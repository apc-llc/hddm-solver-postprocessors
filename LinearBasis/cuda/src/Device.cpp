#include "Device.h"

using namespace NAMESPACE;

long long Device::getID() const { return id; }
int Device::getWarpSize() const { return warpSize; }
int Device::getCC() const { return cc; }

Device::Device() : available(1) { }

