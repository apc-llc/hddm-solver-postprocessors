#include "Device.h"

#include "check.h"
#include "process.h"

using namespace NAMESPACE;

__device__ int maxConcurrentBlocksVar;
__device__ volatile int maxConcurrentBlockEvalDoneVar;

__device__ int* maxConcurrentBlocks()
{
	return &maxConcurrentBlocksVar;
}

__device__ int* maxConcurrentBlockEvalDone()
{
	return (int*)&maxConcurrentBlockEvalDoneVar;
}

__device__ volatile float BigData_[1024 * 1024];

__device__ volatile float* BigData()
{
	return ::BigData_;
}

template<int ITS, int REGS = 32>
class DelayFMADS
{
public:

	__device__
	inline __attribute__((always_inline))
	static void delay()
	{
		float values[REGS];

		#pragma unroll
		for(int r = 0; r < REGS; ++r)
			values[r] = BigData()[threadIdx.x + r * 32];

		#pragma unroll
		for(int i = 0; i < (ITS + REGS - 1) / REGS; ++i)
		{
			#pragma unroll
			for(int r = 0; r < REGS; ++r)
				values[r] += values[r] * values[r];
			__threadfence_block();
		}

		#pragma unroll
		for(int r = 0; r < REGS; ++r)
			BigData()[threadIdx.x + r * 32] = values[r];
	}
};

__global__ void maxConcurrentBlockEval()
{
	if (*maxConcurrentBlockEvalDone() != 0)
		return;

	if (threadIdx.x == 0)
		atomicAdd(maxConcurrentBlocks(), 1);

	DelayFMADS<10000, 4>::delay();
	__syncthreads();

	*maxConcurrentBlockEvalDone() = 1;
	__threadfence();
}

long long Device::getID() const { return id; }
int Device::getBlockSize() const { return 128; }
int Device::getBlockCount() const { return blockCount; }
int Device::getWarpSize() const { return warpSize; }
int Device::getCC() const { return cc; }

Device::Device() : available(1)
{
	maxConcurrentBlockEval<<<1024, getBlockSize()>>>();
	
	CUDA_ERR_CHECK(cudaGetLastError());
	CUDA_ERR_CHECK(cudaDeviceSynchronize());

	CUDA_ERR_CHECK(cudaMemcpyFromSymbol(&blockCount, maxConcurrentBlocksVar, sizeof(int)));
}

