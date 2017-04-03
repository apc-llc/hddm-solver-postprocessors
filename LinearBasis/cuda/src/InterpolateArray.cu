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
	const int dim, const int nno, const int nnoPerBlock, const int DofPerNode, const double* x,
	const int nfreqs, const XPS::Device* xps_, const int szxps, double** xpv_, const Chains::Device* chains_,
	const Matrix<double>::Device* surplus_, double* value)
{
	const XPS::Device& xps = *xps_;
	double* xpv = xpv_[blockIdx.x];
	const Chains::Device& chains = *chains_;
	const Matrix<double>::Device& surplus = *surplus_;

	// Loop to calculate all unique xp values.
	for (int i = threadIdx.x, e = szxps; i < e; i += blockDim.x)
	{
		const Index<uint16_t>& index = xps(i);
		const uint32_t& j = index.index;
		double xp = LinearBasis(x[j], index.i, index.j);
		xpv[i] = fmax(0.0, xp);
	}

	__syncthreads();

#define szcache 4
	// Each thread hosts a part of blockDim.x-shared register cache
	// to accumulate nnoPerBlock intermediate additions.
	// blockDim.x -sharing is done due to limited number of registers
	// available per thread.
	double cache[szcache];
	for (int i = 0; i < szcache; i++)
		cache[i] = 0;
#undef szcache

	// Loop to calculate scaled surplus product.
	for (int i = blockIdx.x * nnoPerBlock, e = min(i + nnoPerBlock, NNO); i < e; i++)
	{
		double temp = 1.0;
		for (int ifreq = 0; ifreq < nfreqs; ifreq++)
		{
			// Early exit for shorter chains.
			int32_t idx = chains(i * nfreqs + ifreq);
			if (!idx) break;

			temp *= xpv[idx];
			if (!temp) goto next;
		}

		for (int Dof_choice = threadIdx.x, icache = 0; Dof_choice < DOF_PER_NODE; Dof_choice += blockDim.x, icache++)
			cache[icache] += temp * surplus(i, Dof_choice);
	
	next :

		continue;
	}

	for (int Dof_choice = threadIdx.x, icache = 0; Dof_choice < DOF_PER_NODE; Dof_choice += blockDim.x, icache++)
		atomicAdd(&value[Dof_choice], cache[icache]);
}

namespace {

class InterpolateArray
{
	Vector<double>::Device xVectorDev;
	Vector<double>::Device valueVectorDev;
	Matrix<double>::Device xpvMatrixDev;

public :

	int dim;
	int DofPerNode;

	int szblock;
	int nnoPerBlock;
	int nblocks;

	double* xDev;
	double* valueDev;
	double** xpvDev;

	InterpolateArray(int dim, int nno, int DofPerNode, const int szxps) :
		dim(dim), DofPerNode(DofPerNode),
		szblock(128), nnoPerBlock(16), nblocks(nno / nnoPerBlock + (nno % nnoPerBlock ? 1 : 0)),
		xDev(NULL), xVectorDev(dim),
		valueDev(NULL), valueVectorDev(dim),
		xpvDev(NULL)

	{
		xDev = xVectorDev.getData();
		valueDev = valueVectorDev.getData();

		// Prepare the XPV buffer vector sized to max xps.size()
		// across all states. Individual buffer for each CUDA block.
		CUDA_ERR_CHECK(cudaMalloc(&xpvDev, sizeof(double*) * nblocks));
		vector<double*> xpv(nblocks);
		xpvMatrixDev.resize(nblocks, szxps);
		for (int i = 0; i < nblocks; i++)
			xpv[i] = xpvMatrixDev.getData(i, 0);
		CUDA_ERR_CHECK(cudaMemcpy(&xpvDev[0], &xpv[0], sizeof(double*) * nblocks,
			cudaMemcpyHostToDevice));
	}

	void load(const double* x)
	{
		CUDA_ERR_CHECK(cudaMemcpy(xVectorDev.getData(), x, sizeof(double) * dim,
			cudaMemcpyHostToDevice));

		CUDA_ERR_CHECK(cudaMemset(valueVectorDev.getData(), 0, sizeof(double) * DOF_PER_NODE));
	}

	void save(double* value)
	{
		CUDA_ERR_CHECK(cudaMemcpy(value, valueVectorDev.getData(), sizeof(double) * DOF_PER_NODE,
			cudaMemcpyDeviceToHost));
	}

	~InterpolateArray()
	{
		CUDA_ERR_CHECK(cudaFree(xpvDev));
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
	if (!interp.get())
		interp.reset(new InterpolateArray(dim, nno, DofPerNode, szxps));

	interp->load(x);

	// Launch the kernel.
	KERNEL(FUNCNAME)<<<interp->nblocks, interp->szblock>>>(
		dim, nno, interp->nnoPerBlock, DofPerNode, interp->xDev,
		nfreqs, xps_, szxps, interp->xpvDev, chains_, surplus_, interp->valueDev);

	CUDA_ERR_CHECK(cudaDeviceSynchronize());

	interp->save(value);
}

