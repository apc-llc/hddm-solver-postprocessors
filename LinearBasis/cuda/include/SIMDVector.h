#ifndef SIMDVECTOR_H
#define SIMDVECTOR_H

#include "check.h"
#include "Device.h"
#include "process.h"

#include <iostream>

namespace cuda {

class SIMDVector
{
	const Device* device;
	
public :

	SIMDVector(const Device* device);

	virtual inline __attribute__((always_inline)) size_t getLength(size_t szelement) const
	{
		switch (szelement)
		{
		case 4 :
		case 8 :
			// Warps do 128-byte or 256-byte coalesced memory transactions
			// for float and double data, respectively.
			return device->getWarpSize();
		default :
			{
				MPI_Process* process;
				MPI_ERR_CHECK(MPI_Process_get(&process));
				std::cerr << "Unsupported vector element size = " << szelement << std::endl;
				process->abort();
			}
		}
	}
};

} // namespace cuda

#endif // SIMDVECTOR_H

