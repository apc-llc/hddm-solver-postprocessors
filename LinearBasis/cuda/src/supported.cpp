#include "Devices.h"

#include <memory>

using namespace NAMESPACE;
using namespace std;

namespace NAMESPACE
{
	extern unique_ptr<Devices> devices;
}

extern "C" bool isSupported()
{
	if (!devices)
		devices.reset(new Devices());

	// GPU interpolator is supported, if at least one GPU is available
	return (devices->getCount() > 0);
}

