#include "instrset.h"

extern "C" bool isSupported()
{
	// Use this interpolator if only AVX is supported.
	InstrSet iset = InstrSetDetect();
	if (iset == InstrSetAVX)
		return true;
	
	return false;
}

