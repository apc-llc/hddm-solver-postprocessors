#ifndef DEVICE_H
#define DEVICE_H

namespace cpu {

class Devices;

class Postprocessor;

// Defines device-specific parameters of interpolator.
class Device
{
	int available;
	
	Postprocessor* post;

public :

	Device();
	
	friend class Devices;
};

} // namespace cpu

#endif // DEVICE_H

