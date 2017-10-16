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
	const int dim, const int nnoPerBlock, const int DofPerNode, const double* x,
	const int nfreqs, const XPS::Device* xps_, const int szxps, double** xpv_, const Chains::Device* chains_,
	const Matrix<double>::Device* surplus_, double* value)
{
	const XPS::Device& xps = *xps_;
#ifdef XPV_IN_GLOBAL_MEMORY
	double* xpv = xpv_[blockIdx.x];
#else
	extern __shared__ double xpv[];
#endif
	const Chains::Device& chains = *chains_;
	const Matrix<double>::Device& surplus = *surplus_;

	int nno = surplus.dimy();

	// Loop to calculate all unique xp values.
	for (int i = threadIdx.x, e = szxps; i < e; i += blockDim.x)
		xpv[i] = LinearBasis(x, (uint32_t*)&xps(i));

	// Each thread hosts a part of blockDim.x-shared register cache
	// to accumulate nnoPerBlock intermediate additions.
	// blockDim.x -sharing is done due to limited number of registers
	// available per thread.
	double cache = 0.0;

	__syncthreads();

	// Loop to calculate scaled surplus product.
	for (int i = blockIdx.x * nnoPerBlock, e = min(i + nnoPerBlock, nno); i < e; i++)
	{
		double temp = 1.0;
		for (int ifreq = 0; ifreq < nfreqs; ifreq++)
		{
			// Early exit for shorter chains.
			int32_t idx = chains(i * nfreqs + ifreq);
			if (!idx) break;

			double xp = xpv[idx];
			if (xp <= 0.0) goto next;
			
			temp *= xp;
		}

		for (int Dof_choice = threadIdx.x, icache = 0; Dof_choice < DOF_PER_NODE; Dof_choice += blockDim.x, icache++)
			cache += temp * surplus(i, Dof_choice);				
	
	next :

		continue;
	}

	for (int Dof_choice = threadIdx.x, icache = 0; Dof_choice < DOF_PER_NODE; Dof_choice += blockDim.x, icache++)
		atomicAdd(&value[Dof_choice], cache);
}

namespace {

class InterpolateArray
{
	Vector<double>::Device *xVectorDev_, &xVectorDev;
	Vector<double>::Device *valueVectorDev_, &valueVectorDev;
	Matrix<double>::Device *xpvMatrixDev_, &xpvMatrixDev;

public :

	int dim;
	int nno;
	int DofPerNode;

	static const int szblock = 128;
	static const int nnoPerBlock = 64;
	int nblocks;

	const double* xHost;
	double* xDev;
	double* valueDev;
#ifdef XPV_IN_GLOBAL_MEMORY
	vector<double*> xpv;
#endif
	double** xpvDev;
	
	// Keep szxps for further reference: its change
	// triggers xpv reallocation.
	int szxps;

	cudaStream_t stream;

	InterpolateArray(int dim, int nno, int DofPerNode, const int szxps) :
		xDev(NULL), xVectorDev_(new Vector<double>::Device(dim)), xVectorDev(*xVectorDev_),
		valueDev(NULL), valueVectorDev_(new Vector<double>::Device(DOF_PER_NODE)), valueVectorDev(*valueVectorDev_),
		xpvMatrixDev_(new Matrix<double>::Device()), xpvMatrixDev(*xpvMatrixDev_),
		dim(dim), nno(nno), DofPerNode(DOF_PER_NODE),
		nblocks(nno / nnoPerBlock + (nno % nnoPerBlock ? 1 : 0)),
		szxps(szxps), xpvDev(NULL)

	{
		CUDA_ERR_CHECK(cudaStreamCreate(&stream));

		xDev = xVectorDev.getData();
		valueDev = valueVectorDev.getData();

#ifdef XPV_IN_GLOBAL_MEMORY
		// Prepare the XPV buffer vector sized to max xps.size()
		// across all states. Individual buffer for each CUDA block.
		CUDA_ERR_CHECK(cudaMalloc(&xpvDev, sizeof(double*) * nblocks));
		vector<double*> xpv(nblocks);
		xpvMatrixDev.resize(nblocks, szxps);
		for (int i = 0; i < nblocks; i++)
			xpv[i] = xpvMatrixDev.getData(i, 0);
		CUDA_ERR_CHECK(cudaMemcpyAsync(&xpvDev[0], &xpv[0], sizeof(double*) * nblocks,
			cudaMemcpyHostToDevice, stream));
#endif
	}

	template<class T>
	static inline void hashCombine(size_t& seed, const T& v)
	{
		std::hash<T> hasher;
		seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	}

	static size_t hash(int dim, int nno, int DofPerNode)
	{
		size_t seed = 0;
		hashCombine(seed, dim);
		hashCombine(seed, nno);
		hashCombine(seed, DofPerNode);
		return seed;
	}

	size_t hash() const
	{
		return InterpolateArray::hash(dim, nno, DofPerNode);
	}

	void load(const double* x, const int szxps_)
	{
		xHost = x;
		cudaError_t cudaError = cudaHostRegister(const_cast<double*>(x), sizeof(double) * dim, cudaHostRegisterDefault);
		if (cudaError != cudaErrorHostMemoryAlreadyRegistered)
			CUDA_ERR_CHECK(cudaError);
		CUDA_ERR_CHECK(cudaMemcpyAsync(xVectorDev.getData(), x, sizeof(double) * dim,
			cudaMemcpyHostToDevice, stream));

		CUDA_ERR_CHECK(cudaMemsetAsync(valueVectorDev.getData(), 0, sizeof(double) * DOF_PER_NODE, stream));

#ifdef XPV_IN_GLOBAL_MEMORY
		// Reallocate xpv, if not enough space.
		xpv.resize(nblocks);
		CUDA_ERR_CHECK(cudaHostRegister(&xpv[0], sizeof(double*) * nblocks, cudaHostRegisterDefault));
		if (szxps_ < szxps)
		{
			szxps = szxps_;

			xpvMatrixDev.resize(nblocks, szxps);
			for (int i = 0; i < nblocks; i++)
				xpv[i] = xpvMatrixDev.getData(i, 0);
			CUDA_ERR_CHECK(cudaMemcpyAsync(&xpvDev[0], &xpv[0], sizeof(double*) * nblocks,
				cudaMemcpyHostToDevice, async));
		}
#endif
	}

	void save(double* value)
	{
		cudaError_t cudaError = cudaHostRegister(value, sizeof(double) * DOF_PER_NODE, cudaHostRegisterDefault);
		if (cudaError != cudaErrorHostMemoryAlreadyRegistered)
			CUDA_ERR_CHECK(cudaError);
		CUDA_ERR_CHECK(cudaMemcpyAsync(value, valueVectorDev.getData(), sizeof(double) * DOF_PER_NODE,
			cudaMemcpyDeviceToHost, stream));

		CUDA_ERR_CHECK(cudaStreamSynchronize(stream));

		cudaError = cudaHostUnregister(const_cast<double*>(xHost));
		if (cudaError != cudaErrorHostMemoryNotRegistered)
			CUDA_ERR_CHECK(cudaError);
		cudaError = cudaHostUnregister(value);
		if (cudaError != cudaErrorHostMemoryNotRegistered)
			CUDA_ERR_CHECK(cudaError);

#ifdef XPV_IN_GLOBAL_MEMORY
		CUDA_ERR_CHECK(cudaHostUnregister(&xpv[0]));
#endif
	}

	virtual ~InterpolateArray()
	{
		cudaError_t cudaError = cudaStreamDestroy(stream);
		if (cudaError != cudaErrorCudartUnloading)
		{
			CUDA_ERR_CHECK(cudaError);
#ifdef XPV_IN_GLOBAL_MEMORY
			CUDA_ERR_CHECK(cudaFree(xpvDev));
#endif
			delete xVectorDev_;
			delete valueVectorDev_;
			delete xpvMatrixDev_;
		}
	}
};

unique_ptr<InterpolateArray> interp;

} // namespace

extern "C" void FUNCNAME(
	Device* device,
	const int dim, const int nno, const int DofPerNode, const double* x,
	const int nfreqs, const XPS::Device* xps_, const int szxps, const Chains::Device* chains_,
	const Matrix<double>::Device* surplus_, double* value)
{
	bool rebuild = false;
	if (!interp.get())
		rebuild = true;
	else
		if (interp->hash() != InterpolateArray::hash(dim, nno, DOF_PER_NODE))
			rebuild = true;
	
	if (rebuild)
		interp.reset(new InterpolateArray(dim, nno, DOF_PER_NODE, szxps));

	interp->load(x, szxps);

	// Launch the kernel.
#ifdef XPV_IN_GLOBAL_MEMORY
	KERNEL(FUNCNAME)<<<interp->nblocks, interp->szblock, 0, interp->stream>>>(
#else
	KERNEL(FUNCNAME)<<<interp->nblocks, interp->szblock, interp->szxps * sizeof(double), interp->stream>>>(
#endif
		dim, interp->nnoPerBlock, DofPerNode, interp->xDev,
		nfreqs, xps_, szxps, interp->xpvDev, chains_, surplus_, interp->valueDev);

	interp->save(value);
}

