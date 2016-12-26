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
	const int dim, const int dimAligned, const int nno, const int nnoAligned,
	const int Dof_choice_start, const int Dof_choice_end, const int count,
	const Matrix::Device::Sparse::CRW<IndexPair, uint32_t>* index_,
	const Matrix::Device::Dense<double>* surplus_, double* value_,
	const int length)
{
	for (int many = 0; many < COUNT; many++)
	{
		const Matrix::Device::Sparse::CRW<IndexPair, uint32_t>& index = index_[many];
		const Matrix::Device::Dense<double>& surplus = surplus_[many];
		double* value = value_ + many * length;
		const int nnzPerRow = index.nnzperrow();

		extern __shared__ double xx[];
		for (int i = threadIdx.x; i < dimAligned; i += blockDim.x)
			xx[i] = x(i + many * DIM);

		int i = blockDim.x * blockIdx.x + threadIdx.x;

		if (i >= nno) continue;

		double temp = 1.0;
		for (int j = 0; j < nnzPerRow; j++)
		{
			int idx = i + nnoAligned * j;

			const IndexUnion iu = { *(const uint32_t*)&index.a(idx) };
			const IndexPair& pair = iu.pair;

			// TODO map "x" vector onto shared memory for speed, if size permits.
			double xp = LinearBasis(xx[index.ja(idx)], pair.i, pair.j);
			temp *= fmax(0.0, xp);
		}

		if (!temp) continue;

		// Collect values into global memory.
		for (int j = Dof_choice_start; j <= Dof_choice_end; j++)
			atomicAdd(&value[j - Dof_choice_start], temp * surplus(j, i));
	}
}

extern "C" void FUNCNAME(
	Device* device,
	const int dim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const int count, const double* const* x,
	const Matrix::Device::Sparse::CRW<IndexPair, uint32_t>* index,
	const Matrix::Device::Dense<double>* surplus, double** value)
{
	// Configure kernel compute grid.
	dim3 blockDim = { AVX_VECTOR_SIZE, 1, 1 };
	dim3 gridDim = { nno / blockDim.x, 1, 1 };

	int nnoAligned = nno;
	if (nno % AVX_VECTOR_SIZE)
		nnoAligned = nno + AVX_VECTOR_SIZE - nno % AVX_VECTOR_SIZE;

	int dimAligned = dim;
	if (dim % blockDim.x)
		dimAligned = dim + blockDim.x - dim % blockDim.x;

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
	InterpolateArrayManyMultistate_kernel_large_dim<<<gridDim, blockDim, dimAligned * sizeof(double), stream>>>(
#ifdef DEFERRED
		*dx,
#else
		dx,
#endif
		dim, dimAligned, nno, nnoAligned,
		Dof_choice_start, Dof_choice_end, COUNT,
		index, surplus, dvalue, length);
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

