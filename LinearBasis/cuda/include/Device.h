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

	long long id;
	int blockSize;
	int blockCount;
	int warpSize;
	int cc;

public :

	long long getID() const;
	int getBlockSize() const;
	int getBlockCount() const;
	int getWarpSize() const;
	int getCC() const;
		
	Device();
	
	friend class Devices;
};

} // namespace NAMESPACE

#endif // DEVICE_H

