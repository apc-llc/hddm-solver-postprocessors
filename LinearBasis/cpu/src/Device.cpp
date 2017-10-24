#include "Device.h"

using namespace NAMESPACE;

Device::Device() : available(1)
{
}

int Device::getThreadsCount() const
{
	return nthreads;
}
