#include "DeviceProperties.h"
#include "Devices.h"

#include <iostream>
#include <memory>

using namespace NAMESPACE;
using namespace std;

DeviceProperties::DeviceProperties(const Device* device) : simdVector(device) { }

const SIMDVector* DeviceProperties::getSIMDVector() const
{
	return &simdVector;
}
static bool propsInitialized = false;
static vector<unique_ptr<DeviceProperties> > uniqueProps;
static vector<DeviceProperties*> props;

namespace NAMESPACE
{
	extern unique_ptr<Devices> devices;
}

vector<DeviceProperties*>& DeviceProperties::getDeviceProperties()
{
	if (!propsInitialized)
	{
		if (!devices)
			devices.reset(new Devices());

		uniqueProps.resize(devices->getCount());
		props.resize(devices->getCount());
		for (int i = 0, e = uniqueProps.size(); i != e; i++)
		{
			uniqueProps[i].reset(new DeviceProperties(devices->getDevice(i)));
			props[i] = uniqueProps[i].get();
		}
		
		propsInitialized = true;
	}
	
	return props;
}

extern "C" void getDeviceProperties(DeviceProperties*** result, size_t* szresult)
{
	vector<DeviceProperties*>& vprops = DeviceProperties::getDeviceProperties();
	*result = &vprops[0];
	*szresult = vprops.size();
}

