#ifndef DEVICES_H
#define DEVICES_H

#include <vector>

#include "Device.h"

namespace NAMESPACE {

class DeviceProperties;

class Devices
{
	std::vector<Device> devices;
	
	const Device* getDevice(int index) const;

public :

	Devices();

	int getCount();

	Device* tryAcquire();

	void release(Device* device);
	
	friend class DeviceProperties;
};

} // namespace NAMESPACE

extern "C" NAMESPACE::Device* tryAcquireDevice();

extern "C" void releaseDevice(NAMESPACE::Device* device);

#endif // DEVICES_H

