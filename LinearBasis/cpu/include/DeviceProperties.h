#ifndef DEVICE_PROPERTIES_H
#define DEVICE_PROPERTIES_H

#include "SIMDVector.h"

namespace NAMESPACE {

class DeviceProperties
{
public :

	virtual const SIMDVector* getSIMDVector() const;
};

} // namespace NAMESPACE

#endif // DEVICE_PROPERTIES_H

