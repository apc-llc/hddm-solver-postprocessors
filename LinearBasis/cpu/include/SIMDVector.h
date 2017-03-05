#ifndef SIMDVECTOR_H
#define SIMDVECTOR_H

#include "check.h"
#include "process.h"

#include <iostream>

namespace cpu {

class SIMDVector
{
public :

	virtual inline __attribute__((always_inline)) size_t getLength(size_t szelement) const
	{
#ifdef HAVE_AVX
		switch (szelement)
		{
		case 4 : return 8;
		case 8 : return 4;
		default :
			{
				MPI_Process* process;
				MPI_ERR_CHECK(MPI_Process_get(&process));
				std::cerr << "Unsupported vector element size = " << szelement << std::endl;
				process->abort();
			}
		}
// #elif TODO: implement SSE, AVX512, etc. here
#else
		return 1;
#endif
		return 0;
	}
};

} // namespace cpu

#endif // SIMDVECTOR_H

