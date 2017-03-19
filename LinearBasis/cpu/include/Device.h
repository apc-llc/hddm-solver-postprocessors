#ifndef DEVICE_H
#define DEVICE_H

namespace NAMESPACE {

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

} // namespace NAMESPACE

#endif // DEVICE_H

