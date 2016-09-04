#ifndef DEVICES_H
#define DEVICES_H

#include <vector>

#include "Device.h"

namespace cuda {

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

} // namespace cuda

#endif // DEVICES_H

