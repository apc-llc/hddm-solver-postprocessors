#include "Data.h"

using namespace NAMESPACE;

extern "C" Data::Dense* getData(int nstates)
{
	return new Data::Dense(nstates);
}

