#include "instrset.h"

extern "C" bool isSupported()
{
	// Use this interpolator if AVX2 is supported.
	InstrSet iset = InstrSetDetect();
	if (iset == InstrSetAVX2)
		return true;
	
	return false;
}

