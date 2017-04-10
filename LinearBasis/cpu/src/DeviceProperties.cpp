#include "DeviceProperties.h"

#include <vector>

using namespace NAMESPACE;
using namespace std;

static SIMDVector simdVector;

const SIMDVector* DeviceProperties::getSIMDVector() const
{
	return &simdVector;
}

static DeviceProperties props;
static vector<DeviceProperties*> vprops;

extern "C" void getDeviceProperties(DeviceProperties*** result, size_t* szresult)
{
	if (!vprops.size())
	{
		vprops.resize(1);
		vprops[0] = &props;
	}
	
	*result = &vprops[0];
	*szresult = vprops.size();
}

