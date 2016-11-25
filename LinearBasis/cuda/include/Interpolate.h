#ifndef INTERPOLATE_H
#define INTERPOLATE_H

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
// Multi- interpolators define COUNT value. Define unit COUNT for
// all other interpolators.
#ifndef COUNT
#define COUNT 1
#endif

// Define a structure-container, that shall be used to pass x vector as
// a single value via kernel argument. As an argument, it will be implicitly
// loaded into device constant memory.
struct X
{
	double values[DIM * COUNT];
};

#define KERNEL_PARAM_0_STR(name) #name
#define KERNEL_PARAM_0_CAT(name) KERNEL_PARAM_0_STR(name ## _param_0)
#define KERNEL_PARAM_0(name) KERNEL_PARAM_0_CAT(name)

// In case of deferred compilation, X vector is loaded as a kernel argument
// for speed (saves on separate memcpy call). However, if used, brain-damaged CUDA
// compiler copies entire array into local memory (STL/LDL). In order to avoid this,
// we do not use the X kernel argument in the code directly, and instead read
// from 5th argument of PTX kernel representation. Note this is a fragile and
// compiler-specific hack.
inline __attribute__((always_inline))  __device__ double x(int j)
{
	double ret;
	asm("{\n\t"
		".reg .u64 ptr;\n\t"
		"mov.u64 ptr, "
		KERNEL_PARAM_0(KERNEL_NAME)
		";\n\t"
		"mad.wide.u32 ptr, %1, 8, ptr;\n\t"
		"ld.param.f64 %0, [ptr];"
		"}" : "=d"(ret) : "r"(j)
	);
	return ret;
}
#else
#define x(j) x_[j]
#endif

// Configure kernel compute grid.
static void configureKernel(
	Device* device, 
	const int dim, const int nno,
	int& vdim, dim3& blockDim, dim3& gridDim,
	int& nwarps, int& nnoPerBlock)
{
	nnoPerBlock = 16;

	// Index arrays shall be padded to AVX_VECTOR_SIZE-element
	// boundary to keep up the required alignment.
	vdim = dim / AVX_VECTOR_SIZE;
	if (dim % AVX_VECTOR_SIZE) vdim++;
	vdim *= AVX_VECTOR_SIZE;
	
	// Choose efficient grid block dimensions.
	blockDim = dim3(1, 1, 1);
	gridDim = dim3(1, 1, 1);

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
			
			// The rest of nno goes as grid dimension.
			gridDim.x = nno / (nnoPerBlock * blockDim.y);
			if (nno % (nnoPerBlock * blockDim.y))
				gridDim.x++;
		}
		else
		{
			// The rest of nno goes as grid dimension.
			gridDim.x = nno / nnoPerBlock;
			if (nno % nnoPerBlock)
				gridDim.x++;
		}
	}
	else
	{
		assert(false && "Not tested yet");
	
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
	nwarps = (blockDim.x * blockDim.y) / warpSize;
}

#endif // INTERPOLATE_H

