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

inline __attribute__((always_inline)) __device__ double warpReduceMultiply(double val)
{
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
		val *= __shfl_down(val, offset);
	return val;
}

inline __attribute__((always_inline)) __device__ double warpAllReduceMultiply(double val)
{
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
		val *= __shfl_xor(val, offset);
	return val;
}

#ifdef DEFERRED
// Define a structure-container, that shall be used to pass x vector as
// a single value via kernel argument. As an argument, it will be implicitly
// loaded into device constant memory.
struct X
{
	double values[DIM];
};

// In case of deferred compilation, X vector is loaded as a kernel argument
// for speed (saves on separate memcpy call). However, if used, brain-damaged CUDA
// compiler copies entire array into local memory (STL/LDL). In order to avoid this,
// we do not use the X kernel argument in the code directly, and instead read
// from 5th argument of PTX kernel representation. Note this is a fragile and
// compiler-specific hack.
inline __attribute__((always_inline))  __device__ double x(int j)
{
	double ret;
	asm(
		".reg .u64 ptr, i;\n\t"
		"mov.u64 ptr, InterpolateArray_kernel_large_dim_param_5;\n\t"
		"cvt.u64.u32 i, %1;\n\t"
		"mad.lo.u64 ptr, i, 8, ptr;\n\t"
		"ld.param.cs.f64 %0, [ptr];"  : "=d"(ret) : "r"(j));
	return ret;
}
#else
#define x(j) x_[j]
#endif

extern "C" __global__ void InterpolateArray_kernel_large_dim(
	const int dim, const int vdim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end,
#ifdef DEFERRED
	const X x_,
#else
 	const double* x_,
#endif
	const Matrix<int>::Device* index_, const Matrix<double>::Device* surplus_, double* value)
{
	extern __shared__ double temps[];

	const Matrix<int>::Device& index = *index_;
	const Matrix<double>::Device& surplus = *surplus_;

	// The "i" is the index by nno, which could be either grid dimension X,
	// or partitioned between grid dimension X and block dimension Y.
	// In case of no partitioning, threadIdx.y is 0, and "i" falls back to
	// grid dimension X only.
	int i = blockIdx.x + threadIdx.y * blockDim.x;

	if (i >= nno) return;

	// Each thread is assigned with a "j" loop index.
	// If DIM is larger than AVX_VECTOR_SIZE, each thread is
	// assigned with multiple "j" loop indexes.
	double temp = 1.0;
	#pragma no unroll
	for (int j = threadIdx.x; j < DIM; j += AVX_VECTOR_SIZE)
	{
		double xp = LinearBasis(x(j), index(i, j), index(i, j + vdim));
		temp *= max(0.0, xp);
	}
	
	// Multiply all partial temps within a warp.
	temp = warpReduceMultiply(temp);
	
	// Gather temps from all participating warps corresponding to the single DIM
	// into a shared memory array.
	int lane = threadIdx.x % warpSize;
	int warpId = threadIdx.x / warpSize;
	int nwarps = blockDim.x / warpSize;
	if (lane == 0)
		temps[warpId + threadIdx.y * nwarps] = temp;

	// Wait for all partial reductions.
	__syncthreads();

	// We can only exit at this point, when all threads in block are synchronized.
	if (!temp) return;

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

	// Atomically add to the output value.
	// Uses double precision atomicAdd code above, since all
	// generations of GPUs before Pascal do not have double atomicAdd builtin.
	for (int j = Dof_choice_start + threadIdx.x; j <= Dof_choice_end; j += blockDim.x)
		atomicAdd(&value[j - Dof_choice_start], temp * surplus(i, j));
}

extern "C" void FUNCNAME(
	Device* device,
	const int dim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const double* x,
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
	if (DIM >= device->warpSize)
	{
		// If DIM is larger than AVX_VECTOR_SIZE, assign multiple
		// indexes per thread, with stepping.
		if (DIM > AVX_VECTOR_SIZE)
			blockDim.x = AVX_VECTOR_SIZE;
		else
		{
			blockDim.x = DIM;
			if (blockDim.x % device->warpSize)
				blockDim.x += device->warpSize - blockDim.x % device->warpSize;
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

	// Calculate the number of warps in block.
	// It shall denote the size of shared memory used for
	// inter-warp step of temp value reduction.
	int nwarps = (blockDim.x * blockDim.y) / device->warpSize;

#ifdef DEFERRED
	X* dx = (X*)x;
#else
	double* dx;
	CUDA_ERR_CHECK(cudaMalloc(&dx, sizeof(double) * DIM));
#endif

	const int length = Dof_choice_end - Dof_choice_start + 1;
	double* dvalue;
	CUDA_ERR_CHECK(cudaMalloc(&dvalue, sizeof(double) * length));
	
	cudaStream_t stream;
	CUDA_ERR_CHECK(cudaStreamCreate(&stream));
#ifndef DEFERRED
	CUDA_ERR_CHECK(cudaMemcpyAsync(dx, x, sizeof(double) * DIM, cudaMemcpyHostToDevice, stream));
#endif
	CUDA_ERR_CHECK(cudaMemsetAsync(dvalue, 0, sizeof(double) * length, stream));
	CUDA_ERR_CHECK(cudaFuncSetSharedMemConfig(
		InterpolateArray_kernel_large_dim, cudaSharedMemBankSizeEightByte));
	InterpolateArray_kernel_large_dim<<<gridDim, blockDim, nwarps * sizeof(double), stream>>>(
		dim, vdim, nno, Dof_choice_start, Dof_choice_end,
#ifdef DEFERRED
		*dx,
#else
		dx,
#endif
		index, surplus, dvalue);
	CUDA_ERR_CHECK(cudaGetLastError());
	CUDA_ERR_CHECK(cudaMemcpyAsync(value, dvalue, sizeof(double) * length, cudaMemcpyDeviceToHost, stream));
	CUDA_ERR_CHECK(cudaStreamSynchronize(stream));
	CUDA_ERR_CHECK(cudaStreamDestroy(stream));

#ifndef DEFERRED
	CUDA_ERR_CHECK(cudaFree(dx));
#endif
	CUDA_ERR_CHECK(cudaFree(dvalue));
}

