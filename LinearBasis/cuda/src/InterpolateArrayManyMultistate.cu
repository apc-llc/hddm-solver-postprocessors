#define KERNEL_NAME InterpolateArrayManyMultistate_kernel_large_dim

#include "interpolator.h"
#include "Interpolate.h"
#include "LinearBasis.h"

#ifdef DEFERRED
#include <vector>

__device__ double dvalue_deferred[1024];
#endif

extern "C" __global__ void KERNEL_NAME(
#ifdef DEFERRED
	const X x_,
#else
 	const double* x_,
#endif
	const int dim, const int vdim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const int count,
	const Matrix<int>::Device* index_, const Matrix<double>::Device* surplus_, double* value_,
	int length, int nwarps, int nnoPerBlock)
{
	int lane = threadIdx.x % warpSize;
	int warpId = threadIdx.x / warpSize;

	extern __shared__ double temps[];

	for (int many = 0; many < COUNT; many++)
	{
		const Matrix<int>::Device& index = index_[many];
		const Matrix<double>::Device& surplus = surplus_[many];
		double* value = value_ + many * length;

		// The "i" is the index by nno, which could be either grid dimension X,
		// or partitioned between grid dimension X and block dimension Y.
		// In case of no partitioning, threadIdx.y is 0, and "i" falls back to
		// grid dimension X only.
		int i = (blockIdx.x + threadIdx.y * gridDim.x) * nnoPerBlock;
	
		if (i >= nno) continue;

#define szcache 4
		// Each thread hosts a part of blockDim.x-shared register cache
		// to accumulate nnoPerBlock intermediate additions.
		// blockDim.x -sharing is done due to limited number of registers
		// available per thread.
		double cache[szcache];
		for (int i = 0; i < szcache; i++)
			cache[i] = 0;
#undef szcache

		for (int e = i + nnoPerBlock; i < e; i++)
		{
			// Each thread is assigned with a "j" loop index.
			// If DIM is larger than AVX_VECTOR_SIZE, each thread is
			// assigned with multiple "j" loop indexes.
			double temp = 1.0;
			#pragma no unroll
			for (int j = threadIdx.x; j < DIM; j += AVX_VECTOR_SIZE)
			{
				struct IndexPair
				{
					unsigned short i, j;
				};
				union IndexUnion
				{
					int i;
					IndexPair pair;
				};
				IndexUnion iu; iu.i = index(i, j);
				IndexPair& pair = iu.pair;
				if ((pair.i == 0) && (pair.j == 0))
					continue;

				double xp = LinearBasis(x(j + many * DIM), pair.i, pair.j);
				temp *= fmax(0.0, xp);
			}

			// Multiply all partial temps within a warp.
			temp = warpReduceMultiply(temp);

			// Gather temps from all participating warps corresponding to the single DIM
			// into a shared memory array.
			if (lane == 0)
				temps[warpId + threadIdx.y * nwarps] = temp;

			// Wait for all partial reductions.
			__syncthreads();

			// Read from shared memory only if that warp existed.
			temp = (threadIdx.x < blockDim.x / warpSize) ? temps[lane + threadIdx.y * nwarps] : 1.0;

			// Final reduction within the first warp.
			if (warpId == 0)
			{
				temp = warpReduceMultiply(temp);

				// Store result into shared memory to broadcast across all warps.
				if (threadIdx.x == 0)
					temps[threadIdx.y * nwarps] = temp;
			}

			// Wait for the zero thread of the first warp to share temp value in shared memory.
			__syncthreads();

			// Load final reduction value from shared memory.
			temp = temps[threadIdx.y * nwarps];

			// We can only exit at this point, when temp is synchronized for all warps in block.
			if (!temp) continue;
			
			// Collect values into blockDim.x-shared register cache.
			for (int j = Dof_choice_start + threadIdx.x, icache = 0; j <= Dof_choice_end; j += blockDim.x, icache++)
				cache[icache] += temp * surplus(i, j);
		}

		// Collect values into global memory.
		for (int j = Dof_choice_start + threadIdx.x, icache = 0; j <= Dof_choice_end; j += blockDim.x, icache++)
			atomicAdd(&value[j - Dof_choice_start], cache[icache]);
	}
}

extern "C" void FUNCNAME(
	Device* device,
	const int dim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const int count, const double* const* x,
	const Matrix<int>::Device* index, const Matrix<double>::Device* surplus, double** value)
{
	// Configure kernel compute grid.
	int vdim = 1;
	dim3 blockDim(1, 1, 1);
	dim3 gridDim(1, 1, 1);
	int nwarps = 1;
	int nnoPerBlock = 1;
	configureKernel(device, dim, nno, vdim, blockDim, gridDim, nwarps, nnoPerBlock);

#ifdef DEFERRED
	std::vector<double> vx;
	vx.resize(DIM * COUNT);
	X* dx = (X*)&vx[0];
#else	
	double* dx;
	CUDA_ERR_CHECK(cudaMalloc(&dx, sizeof(double) * DIM * COUNT));
#endif

	const int length = Dof_choice_end - Dof_choice_start + 1;
	double* dvalue;
#ifdef DEFERRED
	CUDA_ERR_CHECK(cudaGetSymbolAddress((void**)&dvalue, dvalue_deferred));
#else
	CUDA_ERR_CHECK(cudaMalloc(&dvalue, sizeof(double) * length * COUNT));
#endif

	cudaStream_t stream;
	CUDA_ERR_CHECK(cudaStreamCreate(&stream));
	for (int i = 0; i < COUNT; i++)
	{
#ifdef DEFERRED
		memcpy(&vx[0] + i * DIM, x[i], sizeof(double) * DIM);
#else
		CUDA_ERR_CHECK(cudaMemcpyAsync(dx + i * DIM, x[i], sizeof(double) * DIM, cudaMemcpyHostToDevice, stream));
#endif
	}
	CUDA_ERR_CHECK(cudaMemsetAsync(dvalue, 0, sizeof(double) * length * COUNT, stream));
	CUDA_ERR_CHECK(cudaFuncSetSharedMemConfig(
		InterpolateArrayManyMultistate_kernel_large_dim, cudaSharedMemBankSizeEightByte));
	CUDA_ERR_CHECK(cudaFuncSetCacheConfig(
		InterpolateArrayManyMultistate_kernel_large_dim, cudaFuncCachePreferL1));
	InterpolateArrayManyMultistate_kernel_large_dim<<<gridDim, blockDim, (blockDim.y * nwarps) * sizeof(double), stream>>>(
#ifdef DEFERRED
		*dx,
#else
		dx,
#endif
		dim, vdim, nno, Dof_choice_start, Dof_choice_end, COUNT,
		index, surplus, dvalue, length, nwarps, nnoPerBlock);
	CUDA_ERR_CHECK(cudaGetLastError());
	std::vector<double> vvalue;
	vvalue.resize(length * COUNT);
	CUDA_ERR_CHECK(cudaMemcpyAsync(&vvalue[0], dvalue, sizeof(double) * length * COUNT, cudaMemcpyDeviceToHost, stream));
	for (int i = 0; i < COUNT; i++)
		memcpy(value[i], &vvalue[0] + i * length, sizeof(double) * length);
	CUDA_ERR_CHECK(cudaStreamSynchronize(stream));
	CUDA_ERR_CHECK(cudaStreamDestroy(stream));

#ifndef DEFERRED
	CUDA_ERR_CHECK(cudaFree(dx));	
	CUDA_ERR_CHECK(cudaFree(dvalue));
#endif
}

