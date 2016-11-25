#include "interpolator.h"
#include "LinearBasis.h"

#include "Data.h"
#include "Device.h"

using namespace cuda;

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

static __global__ void InterpolateValue_kernel_large_dim(
	const int dim, const int vdim,
	const int Dof_choice, const double* x,
	const Matrix<int>::Device* index_, const Matrix<double>::Device* surplus_, double* value)
{
	const Matrix<int>::Device& index = *index_;
	const Matrix<double>::Device& surplus = *surplus_;

	// The "i" is the index by nno, which could be either grid dimension X,
	// or partitioned between grid dimension X and block dimension Y.
	// In case of no partitioning, threadIdx.y is 0, and "i" falls back to
	// grid dimension X only.
	int i = blockIdx.x + threadIdx.y * blockDim.x;

	// Each thread is assigned with a "j" loop index.
	// If DIM is larger than AVX_VECTOR_SIZE, each thread is
	// assigned with multiple "j" loop indexes.
	double temp = 1.0;
	for (int j = threadIdx.x; j < DIM; j += AVX_VECTOR_SIZE)
	{
		double xp = LinearBasis(x[j], index(i, j), index(i, j + vdim));
		temp *= max(0.0, xp);
	}
	
	__syncthreads();

	// Multiply all partial temps within warp.
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
		temp *= __shfl_down(temp, offset);
	
	// We can only exit at this point, where all threads are synchronized.
	if (!temp) return;

	// Atomically add to the output value.
	// Uses double precision atomicAdd code above, since all
	// generations of GPUs before Pascal do not have double atomicAdd builtin.
	atomicAdd(value, temp * surplus(i, Dof_choice));
}

extern "C" void FUNCNAME(
	Device* device,
	const int dim, const int nno,
	const int Dof_choice, const double* x,
	const Matrix<int>::Device* index, const Matrix<double>::Device* surplus, double* value)
{
	// Index arrays shall be padded to AVX_VECTOR_SIZE-element
	// boundary to keep up the required alignment.
	int vdim = dim / AVX_VECTOR_SIZE;
	if (dim % AVX_VECTOR_SIZE) vdim++;
	vdim *= AVX_VECTOR_SIZE;
	
	// Choose efficient grid block dimensions.
	dim3 blockDim(1, 1, 1);
	dim3 gridDim(1, 1, 1);

	// If DIM is larger than the warp size, then pick up aligned dim
	// as the first dimension.
	int warpSize = device->getWarpSize();
	if (DIM >= warpSize)
	{
		// If DIM is larger than AVX_VECTOR_SIZE, assign multiple
		// indexes per thread, with stepping.
		if (DIM > AVX_VECTOR_SIZE)
			blockDim.x = AVX_VECTOR_SIZE;
		else
		{
			blockDim.x = DIM;
			if (blockDim.x % warpSize)
				blockDim.x += warpSize - blockDim.x % warpSize;
		}

		// If the first dimension is still smaller than AVX_VECTOR_SIZE,
		// pick up a part of nno to get a close value.
		if (blockDim.x < AVX_VECTOR_SIZE)
		{
			blockDim.y = AVX_VECTOR_SIZE / blockDim.x;
			if (AVX_VECTOR_SIZE % blockDim.x)
				blockDim.y++;
			
			gridDim.x = nno / blockDim.y;
			if (nno % blockDim.y)
				gridDim.x++;
		}
		else
		{
			// Otherwise, whole nno goes as grid dimension.
			gridDim.x = nno;
		}
	}
	else
	{
		// Pick up a part of nno to have a block of at least
		// AVX_VECTOR_SIZE.
		blockDim.x = AVX_VECTOR_SIZE;
		
		// Set the rest of nno for grid dimension.
		gridDim.x = nno / blockDim.x;
		if (nno % blockDim.x)
			gridDim.x++;
	}

	real* dx;
	CUDA_ERR_CHECK(cudaMalloc(&dx, sizeof(real) * DIM));

	real* dvalue;
	CUDA_ERR_CHECK(cudaMalloc(&dvalue, sizeof(real)));
	
	cudaStream_t stream;
	CUDA_ERR_CHECK(cudaStreamCreate(&stream));
	CUDA_ERR_CHECK(cudaMemcpyAsync(dx, x, sizeof(real) * DIM, cudaMemcpyHostToDevice, stream));
	CUDA_ERR_CHECK(cudaMemsetAsync(dvalue, 0, sizeof(real), stream));
	InterpolateValue_kernel_large_dim<<<gridDim, blockDim, 0, stream>>>(
		dim, vdim, Dof_choice, dx, index, surplus, dvalue);
	CUDA_ERR_CHECK(cudaGetLastError());
	CUDA_ERR_CHECK(cudaMemcpyAsync(value, dvalue, sizeof(real), cudaMemcpyDeviceToHost, stream));
	CUDA_ERR_CHECK(cudaStreamSynchronize(stream));
	CUDA_ERR_CHECK(cudaStreamDestroy(stream));
	
	CUDA_ERR_CHECK(cudaFree(dx));	
	CUDA_ERR_CHECK(cudaFree(dvalue));
}

