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

static vector<unique_ptr<DeviceProperties> > uniqueProps;
static vector<DeviceProperties*> props;

namespace NAMESPACE
{
	extern Devices devices;
}

vector<DeviceProperties*>* DeviceProperties::getDeviceProperties()
{
	if (!uniqueProps.size())
	{
		uniqueProps.resize(devices.getCount());
		props.resize(devices.getCount());
		for (int i = 0, e = uniqueProps.size(); i != e; i++)
		{
			uniqueProps[i].reset(new DeviceProperties(devices.getDevice(i)));
			props[i] = uniqueProps[i].get();
		}
	}
	
	return &props;
}

extern "C" vector<DeviceProperties*>& getDeviceProperties()
{
	return DeviceProperties::getDeviceProperties();
}

