#ifndef DEVICES_H
#define DEVICES_H

#include <vector>

#include "Device.h"

namespace cpu {

class Devices
{
	std::vector<Device> devices;

public :

	Devices();

	Device* getDevice(int i);

	Device* tryAcquire();

	void release(Device* device);
};

} // namespace cpu

#endif // DEVICES_H

