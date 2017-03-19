#ifndef DEVICES_H
#define DEVICES_H

#include <vector>

#include "Device.h"

namespace NAMESPACE {

class Devices
{
	std::vector<Device> devices;

public :

	Devices();

	Device* getDevice(int i);

	Device* tryAcquire();

	void release(Device* device);
};

} // namespace NAMESPACE

#endif // DEVICES_H

