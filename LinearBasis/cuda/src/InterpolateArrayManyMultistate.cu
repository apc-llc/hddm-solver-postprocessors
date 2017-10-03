#include "Data.h"
#include "Device.h"
#include "LinearBasis.h"

#include <algorithm> // min & max

#define KERNEL_CAT(name) name##_kernel
#define KERNEL(name) KERNEL_CAT(name)
#define STR1(val) #val
#define STR2(val) STR1(val)
#define STRPARAM_CAT(name, index) STR2(name##_kernel_param_##index)
#define STRPARAM(name, index) STRPARAM_CAT(name, index)

// CUDA 8.0 introduces sm_60_atomic_functions.h with atomicAdd(double*, double)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
inline __attribute__((always_inline)) __device__ double atomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do
	{
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val + __longlong_as_double(assumed)));
	}
	while (assumed != old);

	return __longlong_as_double(old);
}
#endif // __CUDA_ARCH__

using namespace NAMESPACE;
using namespace std;

#ifdef X_IN_CONSTANT_MEMORY
union X
{
	double x[DIM * COUNT];
};
#endif

extern "C" __global__ void KERNEL(FUNCNAME)(
	const int dim, const int nno, const int nnoPerBlock, const int DofPerNode, const int count,
#ifdef X_IN_CONSTANT_MEMORY
	const X x_,
#else	
	const double* x_,
#endif
	const int* nfreqs_, const XPS::Device* xps_, const int* szxps_, double** xpv_, const Chains::Device* chains_,
	const Matrix<double>::Device* surplus_, double* value_, double* value_next_)
{
	// Set the next values array to zero.
	if (blockIdx.x == 0)
	{
		for (int i = threadIdx.x, e = DIM * COUNT; i < e; i += blockDim.x)
			value_next_[i] = 0.0;
	}

	extern __shared__ double temps[];

	for (int many = 0; many < COUNT; many++)
	{
#ifdef X_IN_CONSTANT_MEMORY
		double* x;
		asm("mov.b64 %0, " STRPARAM(FUNCNAME, 5) ";" : "=l"(x));
		x += many * dim;
#else
		const double* x = x_ + many * dim;
#endif
		const int& nfreqs = nfreqs_[many];
		const XPS::Device& xps = xps_[many];
		const Chains::Device& chains = chains_[many];
		const Matrix<double>::Device& surplus = surplus_[many];
		double* value = value_ + many * DOF_PER_NODE;
		
		// Select memory space for the xpv array.
		double* xpv = xpv_ ? xpv_[blockIdx.x] : temps + nnoPerBlock;

		// Loop to calculate all unique xp values.
		for (int i = threadIdx.x, e = szxps_[many]; i < e; i += blockDim.x)
			xpv[i] = LinearBasis(x, (uint32_t*)&xps(i));

		__syncthreads();

		// TODO Store only non-zero temps, store non-zero temp indexes into shared memory as well!

		for (int i = blockIdx.x * nnoPerBlock + threadIdx.x, ii = threadIdx.x,
			e = min(i + nnoPerBlock - threadIdx.x, NNO); i < e; i += blockDim.x, ii += blockDim.x)
		{
			double temp = 1.0;

			for (int ifreq = 0; ifreq < nfreqs; ifreq += 4)
			{
				int4 idx = *(int4*)&chains(i * nfreqs + ifreq);

				if (!idx.x) break;
				double xp = xpv[idx.x];
				if (xp <= 0.0) goto next;
				temp *= xp;

				if (!idx.y) break;
				xp = xpv[idx.y];
				if (xp <= 0.0) goto next;
				temp *= xp;

				if (!idx.z) break;
				xp = xpv[idx.z];
				if (xp <= 0.0) goto next;
				temp *= xp;

				if (!idx.w) break;
				xp = xpv[idx.w];
				if (xp <= 0.0) goto next;
				temp *= xp;
			}

			temps[ii] = temp;
			continue;

		next :
		
			temps[ii] = 0.0;
			continue;
		}

		__syncthreads();

		// Each thread hosts a part of blockDim.x-shared register cache
		// to accumulate nnoPerBlock intermediate additions.
		// blockDim.x -sharing is done due to limited number of registers
		// available per thread.
		double cache = 0.0;

		// Loop to calculate scaled surplus product.
		for (int i = blockIdx.x * nnoPerBlock, e = min(i + nnoPerBlock, NNO), ii = 0; i < e; i++, ii++)
		{
			for (int Dof_choice = threadIdx.x, icache = 0; Dof_choice < DOF_PER_NODE; Dof_choice += blockDim.x, icache++)
				cache += temps[ii] * surplus(i, Dof_choice);
		}

		for (int Dof_choice = threadIdx.x, icache = 0; Dof_choice < DOF_PER_NODE; Dof_choice += blockDim.x, icache++)
			atomicAdd(&value[Dof_choice], cache);
	}
}

namespace {

class InterpolateArrayManyMultistate
{
	Matrix<double>::Device xpvMatrixDev;

public :

	int dim;
	int nno;
	int DofPerNode;
	int count;

	const int szblock;
	const int nnoPerBlock;
	int nblocks;

	vector<double> xHost;
	double* xDev;
	double* valueDev1;
	double* valueDev2;
	vector<int> szxps;
	int* szxpsDev;
#ifdef XPV_IN_GLOBAL_MEMORY
	vector<double*> xpv;
#endif
	double** xpvDev;

	// Keep szxpv for further reference: its change
	// triggers xpv reallocation.
	int szxpv;

	cudaStream_t stream;

	InterpolateArrayManyMultistate(Device* device, int dim, int nno, int DofPerNode, int count, const int* szxps_) :
		szblock(device->getBlockSize()),
		nnoPerBlock(nno / device->getBlockCount() + ((nno % device->getBlockCount()) ? 1 : 0)),
		dim(dim), nno(nno), DofPerNode(DOF_PER_NODE), count(count),
		nblocks(device->getBlockCount()),
		xHost(dim * count), xDev(NULL), valueDev1(NULL), valueDev2(NULL),
		szxps(count), szxpsDev(NULL), szxpv(0), xpvDev(NULL)

	{
		CUDA_ERR_CHECK(cudaMalloc(&xDev, sizeof(double) * count * DOF_PER_NODE));

		// To save on memset-ing output values array to zero, we
		// deploy two output values arrays. Initially, first is zered.
		// Then, second is zeroed during the kernel call. Then arrays
		// are swapped.
		CUDA_ERR_CHECK(cudaMalloc(&valueDev1, sizeof(double) * count * DOF_PER_NODE));
		CUDA_ERR_CHECK(cudaMalloc(&valueDev2, sizeof(double) * count * DOF_PER_NODE));

		CUDA_ERR_CHECK(cudaMalloc(&szxpsDev, sizeof(int) * count));
		CUDA_ERR_CHECK(cudaMemcpy(&szxpsDev[0], &szxps_[0], sizeof(int) * count,
			cudaMemcpyHostToDevice));

		// Prepare the XPV buffer vector sized to max xps.size()
		// across all states. Individual buffer for each CUDA block.
		szxpv = 0;
		memcpy(&szxps[0], &szxps_[0], sizeof(int) * count);
		for (int many = 0; many < count; many++)
			szxpv = max(szxpv, szxps[many]);

		CUDA_ERR_CHECK(cudaMalloc(&xpvDev, sizeof(double*) * nblocks));
		vector<double*> xpv(nblocks);
		xpvMatrixDev.resize(nblocks, szxpv);
		for (int i = 0; i < nblocks; i++)
			xpv[i] = xpvMatrixDev.getData(i, 0);
		CUDA_ERR_CHECK(cudaMemcpy(&xpvDev[0], &xpv[0], sizeof(double*) * nblocks,
			cudaMemcpyHostToDevice));

		CUDA_ERR_CHECK(cudaStreamCreate(&stream));
		CUDA_ERR_CHECK(cudaMemsetAsync(valueDev1, 0, sizeof(double) * count * DOF_PER_NODE, stream));
	}

	template<class T>
	static inline void hashCombine(size_t& seed, const T& v)
	{
		std::hash<T> hasher;
		seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	}

	static size_t hash(int dim, int nno, int DofPerNode, int count)
	{
		size_t seed = 0;
		hashCombine(seed, dim);
		hashCombine(seed, nno);
		hashCombine(seed, DofPerNode);
		hashCombine(seed, count);
		return seed;
	}

	size_t hash() const
	{
		return InterpolateArrayManyMultistate::hash(dim, nno, DofPerNode, count);
	}
	
	void load(const double* const* x_, const int* szxps_)
	{
		for (int i = 0; i < count; i++)
			memcpy(&xHost[i * dim], x_[i], sizeof(double) * dim);

#ifndef X_IN_CONSTANT_MEMORY
		CUDA_ERR_CHECK(cudaHostRegister(&xHost[0], sizeof(double) * dim * count, cudaHostRegisterDefault));

		CUDA_ERR_CHECK(cudaMemcpyAsync(xDev, &xHost[0], sizeof(double) * count * dim, cudaMemcpyHostToDevice, stream));
#endif

		// Copy szxps, if different.
		for (int i = 0; i < count; i++)
			if (szxps_[i] != szxps[i])
			{
				memcpy(&szxps[0], &szxps_[0], sizeof(int) * count);
				CUDA_ERR_CHECK(cudaMemcpyAsync(&szxpsDev[0], &szxps_[0], sizeof(int) * count,
					cudaMemcpyHostToDevice, stream));
				break;
			}

		int szxpvNew = 0;
		for (int many = 0; many < count; many++)
			szxpvNew = max(szxpvNew, szxps_[many]);
		
#ifdef XPV_IN_GLOBAL_MEMORY
		// Reallocate xpv, if not enough space.
		xpv.resize(nblocks);
		CUDA_ERR_CHECK(cudaHostRegister(&xpv[0], sizeof(double*) * nblocks, cudaHostRegisterDefault));
		if (szxpv < szxpvNew)
		{
			szxpv = szxpvNew;
		
			xpvMatrixDev.resize(nblocks, szxpv);
			for (int i = 0; i < nblocks; i++)
				xpv[i] = xpvMatrixDev.getData(i, 0);
			CUDA_ERR_CHECK(cudaMemcpyAsync(&xpvDev[0], &xpv[0], sizeof(double*) * nblocks,
				cudaMemcpyHostToDevice, stream));
		}
#endif
	}

	void save(double** value_)
	{
		for (int i = 0; i < count; i++)
		{
			cudaError_t cudaError = cudaHostRegister(value_[i], sizeof(double) * DOF_PER_NODE, cudaHostRegisterDefault);
			if (cudaError != cudaErrorHostMemoryAlreadyRegistered)
				CUDA_ERR_CHECK(cudaError);
			CUDA_ERR_CHECK(cudaMemcpyAsync(value_[i], &valueDev1[i * DOF_PER_NODE], sizeof(double) * DOF_PER_NODE,
				cudaMemcpyDeviceToHost, stream));
		}

		CUDA_ERR_CHECK(cudaStreamSynchronize(stream));

#ifndef X_IN_CONSTANT_MEMORY
		CUDA_ERR_CHECK(cudaHostUnregister(&xHost[0]));
#endif
		for (int i = 0; i < count; i++)
		{
			cudaError_t cudaError = cudaHostUnregister(value_[i]);
			if (cudaError != cudaErrorHostMemoryNotRegistered)
				CUDA_ERR_CHECK(cudaError);
		}
		
#ifdef XPV_IN_GLOBAL_MEMORY
		CUDA_ERR_CHECK(cudaHostUnregister(&xpv[0]));
#endif

		// Swap values arrays
		double* swap = valueDev1;
		valueDev1 = valueDev2;
		valueDev2 = swap;
	}

	~InterpolateArrayManyMultistate()
	{
		CUDA_ERR_CHECK(cudaFree(xDev));
		CUDA_ERR_CHECK(cudaFree(valueDev1));
		CUDA_ERR_CHECK(cudaFree(valueDev2));
		CUDA_ERR_CHECK(cudaFree(szxpsDev));
#ifdef XPV_IN_GLOBAL_MEMORY		
		CUDA_ERR_CHECK(cudaFree(xpvDev));
#endif
		CUDA_ERR_CHECK(cudaStreamDestroy(stream));
	}
};

unique_ptr<InterpolateArrayManyMultistate> interp;

} // namespace

extern "C" void FUNCNAME(
	Device* device,
	const int dim, const int nno, const int DofPerNode, const int count, const double* const* x_,
	const int* nfreqs_, const XPS::Device* xps_, const int* szxps_, const Chains::Device* chains_,
	const Matrix<double>::Device* surplus_, double** value_)
{
	bool rebuild = false;
	if (!interp.get())
		rebuild = true;
	else
		if (interp->hash() != InterpolateArrayManyMultistate::hash(dim, nno, DOF_PER_NODE, count))
			rebuild = true;
	
	if (rebuild)
		interp.reset(new InterpolateArrayManyMultistate(device, dim, nno, DOF_PER_NODE, count, szxps_));

	interp->load(x_, szxps_);

	// Calculate the size of shared memory.
	// Use shared memory for the XPV vector, if it fits.
	int szshmem = interp->nnoPerBlock * sizeof(double);
	bool useSharedMemoryForXPV = false;
#ifndef XPV_IN_GLOBAL_MEMORY
	int nblocksPerSM = device->getBlockCount() / device->getSM()->getCount();
	int szxpvb = interp->szxpv * sizeof(double);
	if (device->getSM()->getSharedMemorySize() / nblocksPerSM > szshmem + szxpvb)
	{
		szshmem += szxpvb;
		useSharedMemoryForXPV = true;
	}
#endif

	// Launch the kernel.
	KERNEL(FUNCNAME)<<<interp->nblocks, interp->szblock, szshmem, interp->stream>>>(
		dim, nno, interp->nnoPerBlock, DOF_PER_NODE, count,
#ifdef X_IN_CONSTANT_MEMORY
		*(X*)&interp->xHost[0],
#else
		interp->xDev,
#endif
		nfreqs_, xps_, interp->szxpsDev,
		useSharedMemoryForXPV ? NULL : interp->xpvDev,
		chains_, surplus_, interp->valueDev1, interp->valueDev2);

	interp->save(value_);
}

