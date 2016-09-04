#ifndef DEVICE_H
#define DEVICE_H

namespace cuda {

class Devices;

class Postprocessor;

// Defines device-specific parameters of interpolator.
class Device
{
	int available;

	Postprocessor* post;

public :
	
	int warpSize;
	
	Device();
	
	friend class Devices;
};

} // namespace cuda

#endif // DEVICE_H

