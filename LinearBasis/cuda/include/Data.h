#ifndef DATA_H
#define DATA_H

#include <algorithm>
#include <assert.h>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string.h>
#include <vector>

#include "check.h"
#include "process.h"

#ifndef __CUDACC__
#include <cuda_runtime_api.h>
#endif

#include "Vector.h"
#include "Matrix.h"

namespace cuda {

class Interpolator;

struct IndexPair
{
	unsigned short i, j;
};

union IndexUnion
{
	uint32_t i;
	IndexPair pair;
};

class Data
{
	int nstates, dim, vdim, nno, TotalDof, Level;
	
	class Host
	{
		class DataHost;

		// Opaque internal data container.
		std::unique_ptr<DataHost> data;
	
	public :

		Matrix::Host::Sparse::CSR<IndexPair, uint32_t>* getIndex(int istate);
		Matrix::Host::Dense<real>* getSurplus(int istate);
		Matrix::Host::Dense<real>* getSurplus_t(int istate);

		Host(int nstates);
	
		friend class Data;
	}
	host;
	
	class Device
	{
		class DataDevice;

		// Opaque internal data container.
		std::unique_ptr<DataDevice> data;

		int nstates;

	public :

		Matrix::Device::Sparse::CRW<IndexPair, uint32_t>* getIndex(int istate);
		Matrix::Device::Dense<real>* getSurplus(int istate);
		Matrix::Device::Dense<real>* getSurplus_t(int istate);

		void setIndex(int istate, Matrix::Host::Sparse::CSR<IndexPair, uint32_t>& matrix);
		void setSurplus(int istate, Matrix::Host::Dense<real>& matrix);
		void setSurplus_t(int istate, Matrix::Host::Dense<real>& matrix);
	
		Device(int nstates_);
		
		friend class Data;
	}
	device;

	std::vector<bool> loadedStates;
	
	friend class Interpolator;

public :
	virtual int getNno() const;

	virtual void load(const char* filename, int istate);
	
	virtual void clear();

	Data(int nstates);
};

} // namespace cuda

#endif // DATA_H

