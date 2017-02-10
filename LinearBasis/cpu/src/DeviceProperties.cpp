#include "DeviceProperties.h"

#include <vector>

using namespace cpu;
using namespace std;

static SIMDVector simdVector;

const SIMDVector* DeviceProperties::getSIMDVector() const
{
	return &simdVector;
}

static DeviceProperties props;
static vector<DeviceProperties*> vprops;

extern "C" vector<DeviceProperties*>& getDeviceProperties()
{
	if (!vprops.size())
	{
		vprops.resize(1);
		vprops[0] = &props;
	}
	
	return vprops;
}

