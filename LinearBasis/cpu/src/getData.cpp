#include "Data.h"

using namespace NAMESPACE;

extern "C" Data::Sparse* getData(int nstates)
{
	return new Data::Sparse(nstates);
}

