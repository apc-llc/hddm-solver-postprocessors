#include "Devices.h"
#include "interpolator.h"

#include <omp.h>

using namespace NAMESPACE;

Devices::Devices()
{
	devices.resize(1);
}

Device* Devices::tryAcquire()
{
	// Presuming higher level CPU thread pool is managed by TBB,
	// the only CPU device is always available to all threads.
	Device* device = &devices[0];

#if defined(_OPENMP)
	#pragma omp parallel
	{
		#pragma omp master
		device->nthreads = omp_get_num_threads();
	}
#else
	device->nthreads = 1;
#endif

        return device;
}

void Devices::release(Device* device)
{
	// Nothing to do
}

static Devices devices;

namespace NAMESPACE
{
	Device* tryAcquireDevice()
	{
		return devices.tryAcquire();
	}

	void releaseDevice(Device* device)
	{
        	devices.release(device);
	}
}

extern "C" Device* tryAcquireDevice()
{
	return devices.tryAcquire();
}

extern "C" void releaseDevice(Device* device)
{
	devices.release(device);
}
