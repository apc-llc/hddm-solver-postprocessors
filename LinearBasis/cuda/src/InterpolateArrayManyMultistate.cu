#include "LinearBasis.h"
#include "Data.h"

#include <algorithm> // min & max

#define CAT(kernel, name) name##_kernel
#define KERNEL(name) CAT(kernel, name)

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

class Device;

__global__ void KERNEL(FUNCNAME)(
	const int dim, const int nno, const int nnoPerBlock, const int DofPerNode, const int count, const double* const* x_,
	const int* nfreqs_, const XPS::Device* xps_, const int* szxps_, double** xpv_, const Chains::Device* chains_,
	const Matrix<double>::Device* surplus_, double** value_)
{
	extern __shared__ double temps[];

	for (int many = 0; many < COUNT; many++)
	{
		const double* x = x_[many];
		const int& nfreqs = nfreqs_[many];
		const XPS::Device& xps = xps_[many];
		const Chains::Device& chains = chains_[many];
		const Matrix<double>::Device& surplus = surplus_[many];
		double* value = value_[many];

#ifdef XPV_IN_GLOBAL_MEMORY
		double* xpv = xpv_[blockIdx.x];
#else
		double* xpv = temps + nnoPerBlock;
#endif

		// Loop to calculate all unique xp values.
		for (int i = threadIdx.x, e = szxps_[many]; i < e; i += blockDim.x)
			xpv[i] = LinearBasis(x, (uint32_t*)&xps(i));

		__syncthreads();

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
			//for (int Dof_choice = threadIdx.x, icache = 0; Dof_choice < DOF_PER_NODE; Dof_choice += blockDim.x, icache++)
			cache += temps[ii] * surplus(i, threadIdx.x);
		}

		//for (int Dof_choice = threadIdx.x, icache = 0; Dof_choice < DOF_PER_NODE; Dof_choice += blockDim.x, icache++)
		if (threadIdx.x < DOF_PER_NODE)
			atomicAdd(&value[threadIdx.x], cache);
	}
}

namespace {

class InterpolateArrayManyMultistate
{
	Matrix<double>::Device xMatrixDev;
	Matrix<double>::Device valueMatrixDev;
	Matrix<double>::Device xpvMatrixDev;

public :

	int dim;
	int nno;
	int DofPerNode;
	int count;

	static const int szblock = 128;
	static const int nnoPerBlock = 64;
	int nblocks;

	vector<double> xHost;
	double** xDev;
	double** valueDev;
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

	InterpolateArrayManyMultistate(int dim, int nno, int DofPerNode, int count, const int* szxps_) :
		dim(dim), nno(nno), DofPerNode(DOF_PER_NODE), count(count),
		nblocks(nno / nnoPerBlock + (nno % nnoPerBlock ? 1 : 0)),
		xHost(dim * count), xDev(NULL), xMatrixDev(count, dim),
		valueDev(NULL), valueMatrixDev(count, DOF_PER_NODE),
		szxps(count), szxpsDev(NULL), szxpv(0), xpvDev(NULL)

	{
		CUDA_ERR_CHECK(cudaMalloc(&xDev, sizeof(double*) * count));
		vector<double*> x(count);
		for (int i = 0; i < count; i++)
			x[i] = xMatrixDev.getData(i, 0);
		CUDA_ERR_CHECK(cudaMemcpy(&xDev[0], &x[0], sizeof(double*) * count,
			cudaMemcpyHostToDevice));

		CUDA_ERR_CHECK(cudaMalloc(&valueDev, sizeof(double*) * count));
		vector<double*> value(count);
		for (int i = 0; i < count; i++)
			value[i] = valueMatrixDev.getData(i, 0);
		CUDA_ERR_CHECK(cudaMemcpy(&valueDev[0], &value[0], sizeof(double*) * count,
			cudaMemcpyHostToDevice));

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

		CUDA_ERR_CHECK(cudaHostRegister(&xHost[0], sizeof(double) * dim * count, cudaHostRegisterDefault));

		for (int i = 0; i < count; i++)
			CUDA_ERR_CHECK(cudaMemcpyAsync(xMatrixDev.getData(i, 0), &xHost[i * dim], sizeof(double) * dim,
				cudaMemcpyHostToDevice, stream));

		for (int i = 0; i < count; i++)
			CUDA_ERR_CHECK(cudaMemsetAsync(valueMatrixDev.getData(i, 0), 0, sizeof(double) * DOF_PER_NODE, stream));

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
			CUDA_ERR_CHECK(cudaMemcpyAsync(value_[i], valueMatrixDev.getData(i, 0), sizeof(double) * DOF_PER_NODE,
				cudaMemcpyDeviceToHost, stream));
		}

		CUDA_ERR_CHECK(cudaStreamSynchronize(stream));

		CUDA_ERR_CHECK(cudaHostUnregister(&xHost[0]));
		for (int i = 0; i < count; i++)
		{
			cudaError_t cudaError = cudaHostUnregister(value_[i]);
			if (cudaError != cudaErrorHostMemoryNotRegistered)
				CUDA_ERR_CHECK(cudaError);
		}
		
#ifdef XPV_IN_GLOBAL_MEMORY
		CUDA_ERR_CHECK(cudaHostUnregister(&xpv[0]));
#endif
	}

	~InterpolateArrayManyMultistate()
	{
		CUDA_ERR_CHECK(cudaFree(xDev));
		CUDA_ERR_CHECK(cudaFree(valueDev));
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
		interp.reset(new InterpolateArrayManyMultistate(dim, nno, DOF_PER_NODE, count, szxps_));

	interp->load(x_, szxps_);

	// Launch the kernel.
#ifdef XPV_IN_GLOBAL_MEMORY
	KERNEL(FUNCNAME)<<<interp->nblocks, interp->szblock, interp->nnoPerBlock * sizeof(double), interp->stream>>>(
#else
	KERNEL(FUNCNAME)<<<interp->nblocks, interp->szblock, (interp->szxpv + interp->nnoPerBlock) * sizeof(double), interp->stream>>>(
#endif
		dim, nno, interp->nnoPerBlock, DOF_PER_NODE, count, interp->xDev,
		nfreqs_, xps_, interp->szxpsDev, interp->xpvDev, chains_, surplus_, interp->valueDev);

	interp->save(value_);
}

