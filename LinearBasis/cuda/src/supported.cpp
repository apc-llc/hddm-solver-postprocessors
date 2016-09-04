#include "Devices.h"

using namespace cuda;

namespace cuda
{
	extern Devices devices;
}

extern "C" bool isSupported()
{
	// GPU interpolator is supported, if at least one GPU is available
	return (devices.getCount() > 0);
}

