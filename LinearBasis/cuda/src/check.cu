#include "check.h"

__device__ cuda::Lock::Device* cuda::checkLock = NULL;
__device__ char cuda::checkMessage[1024];

// TODO Initialize checkLock

