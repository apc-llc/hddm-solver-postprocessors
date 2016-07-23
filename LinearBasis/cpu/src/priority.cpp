#include "interpolator.h"

using namespace cpu;

extern "C" int getPriority()
{
	// Interpolator tells the priority it has read from the config file.
	const Parameters& params = Interpolator::getInstance()->getParameters();
	return params.priority;
}

