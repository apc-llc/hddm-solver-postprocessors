#ifndef DEVICE_H
#define DEVICE_H

namespace NAMESPACE {

class Devices;

class Postprocessor;

class SM
{
	int count;
	int szshmem;

public :

	int getCount() const;
	int getSharedMemorySize() const;

	SM();

	SM(int count_, int szshmem_);
};

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
	
	SM sm;

public :

	long long getID() const;
	int getBlockSize() const;
	int getBlockCount() const;
	int getWarpSize() const;
	int getCC() const;

	const SM* getSM() const;
		
	Device();
	
	friend class Devices;
};

} // namespace NAMESPACE

#endif // DEVICE_H

