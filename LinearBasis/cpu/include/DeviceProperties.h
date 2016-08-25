#ifndef DEVICE_PROPERTIES_H
#define DEVICE_PROPERTIES_H

#include "SIMDVector.h"

namespace cpu {

class DeviceProperties
{
public :

	virtual const SIMDVector* getSIMDVector() const;
};

} // namespace cpu

#endif // DEVICE_PROPERTIES_H

