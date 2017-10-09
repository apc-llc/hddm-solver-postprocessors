#include "instrset.h"

extern "C" bool isSupported()
{
	// Use this interpolator only if no AVX found at all.
	InstrSet iset = InstrSetDetect();
	if (iset < InstrSetAVX)
		return true;
	
	return false;
}

