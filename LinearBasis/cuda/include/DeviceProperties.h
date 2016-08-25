#ifndef DEVICE_PROPERTIES_H
#define DEVICE_PROPERTIES_H

#include "SIMDVector.h"

#include <vector>

namespace cuda {

class DeviceProperties
{
	SIMDVector simdVector;

public :

	DeviceProperties(const Device* device);

	virtual const SIMDVector* getSIMDVector() const;
	
	static std::vector<DeviceProperties*> getDeviceProperties();
};

} // namespace cuda

#endif // DEVICE_PROPERTIES_H

