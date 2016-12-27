#include "check.h"
#include "process.h"
#include "AlignedAllocator.h"

using namespace cuda::AlignedAllocator;

bool cuda::AlignedAllocator::deviceMemoryHeapInitialized = false;
__device__ kernelgen_memory_t cuda::AlignedAllocator::deviceMemoryHeap;

__global__ void cuda::AlignedAllocator::setupDeviceMemoryHeap(kernelgen_memory_t heap)
{
	deviceMemoryHeap = heap;
}

