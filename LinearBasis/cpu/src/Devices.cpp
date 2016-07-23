#include "Devices.h"
#include "interpolator.h"

using namespace cpu;

Devices::Devices()
{
	devices.resize(1);
}

Device* Devices::tryAcquire()
{
	// Presuming higher level CPU thread pool is managed by TBB,
	// the only CPU device is always available to all threads.
	return &devices[0];
}

void Devices::release(Device* device)
{
	// Nothing to do	
}

static Devices devices;

extern "C" Device* tryAcquireDevice()
{
	return devices.tryAcquire();
}

extern "C" void releaseDevice(Device* device)
{
	devices.release(device);
}

