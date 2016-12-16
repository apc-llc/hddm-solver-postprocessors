#include "check.h"
#include "process.h"
#include "AlignedAllocator.h"

using namespace cuda::AlignedAllocator;

namespace {

// Create a static instance of device allocator,
// such that the device memory pool will be initialized
// on host before any device-side invocations.
static cuda::AlignedAllocator::Device<int> initializer;

static bool heapHostInitialized = false;
static kernelgen_memory_t heapHost;

} // namespace

kernelgen_memory_t& cuda::AlignedAllocator::deviceMemoryHeapHost()
{
	if (!heapHostInitialized)
	{
		heapHost.pool = NULL;
		heapHostInitialized = true;
	}
	return heapHost;
}

