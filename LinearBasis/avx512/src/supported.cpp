#include "instrset.h"

extern "C" bool isSupported()
{
	// Use this interpolator if AVX512 is supported.
	InstrSet iset = InstrSetDetect();
	if (iset == InstrSetAVX512F)
		return true;
	
	return false;
}

