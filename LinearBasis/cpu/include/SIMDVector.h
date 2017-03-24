#ifndef SIMDVECTOR_H
#define SIMDVECTOR_H

#include "check.h"
#include "process.h"

#include <iostream>

namespace NAMESPACE {

class SIMDVector
{
public :

	virtual inline __attribute__((always_inline)) size_t getLength(size_t szelement) const
	{
		// TODO Temporary workaround
		// TODO Privatize this function in different interpolators
		return 16;
	}
};

} // namespace NAMESPACE

#endif // SIMDVECTOR_H

