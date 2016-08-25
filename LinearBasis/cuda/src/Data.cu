#include "check.h"
#include "Data.h"
#include "interpolator.h"

#include <fstream>
#include <iostream>
#include <mpi.h>

using namespace cuda;
using namespace std;

int Data::getNno() const { return nno; }

void Data::load(const char* filename, int istate)
{
	MPI_Process* process;
	MPI_ERR_CHECK(MPI_Process_get(&process));
	const Parameters& params = Interpolator::getInstance()->getParameters();

	if (loadedStates[istate])
	{
		cerr << "State " << istate << " data is already loaded" << endl;
		process->abort();
	}

	ifstream infile;
	if (params.binaryio)
		infile.open(filename, ios::in | ios::binary);
	else
		infile.open(filename, ios::in);

	if (!infile.is_open())
	{
		cerr << "Error opening file: " << filename << endl;
		process->abort();
	}

	infile >> dim;
	if (dim != params.nagents)
	{
		cerr << "File \"" << filename << "\" # of dimensions (" << dim << 
			") mismatches config (" << params.nagents << ")" << endl;
		process->abort();
	}
	infile >> nno; 
	infile >> TotalDof;
	infile >> Level;

	// Pad all indexes to 4-element boundary.
	vdim = dim / AVX_VECTOR_SIZE;
	if (dim % AVX_VECTOR_SIZE) vdim++;
	int nsd = 2 * vdim * AVX_VECTOR_SIZE;
	host.index[istate].resize(nno, nsd);
	host.index[istate].fill(0);
	host.surplus[istate].resize(nno, TotalDof);
	host.surplus[istate].fill(0.0);

	// For better caching we use transposed surplus.
	host.surplus_t[istate].resize(TotalDof, nno);
	host.surplus_t[istate].fill(0.0);
	int j = 0;
	while (infile)
	{
		if (j == nno) break;

		for (int i = 0; i < dim; )
		{
			for (int v = 0; (v < AVX_VECTOR_SIZE) && (i < dim); v++, i++)
			{
				int value; infile >> value;
				value = 2 << (value - 2);
				host.index[istate](j, i) = value;
			}
		}
		for (int i = 0; i < dim; )
		{
			for (int v = 0; (v < AVX_VECTOR_SIZE) && (i < dim); v++, i++)
			{
				int value; infile >> value;
				value--;
				// Percompute "j" to merge two cases into one:
				// (((i) == 0) ? (1) : (1 - fabs((x) * (i) - (j)))).
				if (!host.index[istate](j, i)) value = 0;
				host.index[istate](j, i + vdim * AVX_VECTOR_SIZE) = value;
			}
		}
		for (int i = 0; i < TotalDof; i++)
		{
			double value; infile >> value;
			host.surplus[istate](j, i) = value;
			host.surplus_t[istate](i, j) = value;
		}
		j++;
	}
	infile.close();
	
	loadedStates[istate] = true;
	
	device.index[istate] = host.index[istate];
}

void Data::clear()
{
	fill(loadedStates.begin(), loadedStates.end(), false);
}

Data::Data(int nstates_) : nstates(nstates_), host(nstates), device(nstates)
{
	loadedStates.resize(nstates);
	fill(loadedStates.begin(), loadedStates.end(), false);
}

Data::Host::Host(int nstates)
{
	index.resize(nstates);
	surplus.resize(nstates);
	surplus_t.resize(nstates);
}

static __global__ void constructDeviceData(int nstates,
	Matrix<int>::Device* device_index,
	Matrix<real>::Device* device_surplus,
	Matrix<real>::Device* device_surplus_t)
{
	for (int i = 0; i < nstates; i++)
	{
		new(&device_index[i]) Matrix<int>::Device();
		new(&device_surplus[i]) Matrix<real>::Device();
		new(&device_surplus_t[i]) Matrix<real>::Device();
	}
}

Data::Device::Device(int nstates_) : nstates(nstates_)
{
	// Create arrays of matrices in device memory.
	CUDA_ERR_CHECK(cudaMalloc(&index, sizeof(Matrix<int>::Device) * nstates));
	CUDA_ERR_CHECK(cudaMalloc(&surplus, sizeof(Matrix<real>::Device) * nstates));
	CUDA_ERR_CHECK(cudaMalloc(&surplus_t, sizeof(Matrix<real>::Device) * nstates));
	
	// Construct individual matrices from within the device kernel code,
	// using placement new operator.
	constructDeviceData<<<1, 1>>>(nstates, index, surplus, surplus_t);
	CUDA_ERR_CHECK(cudaGetLastError());
	CUDA_ERR_CHECK(cudaDeviceSynchronize());
}

static __global__ void destroyDeviceData(int nstates,
	Matrix<int>::Device* device_index,
	Matrix<real>::Device* device_surplus,
	Matrix<real>::Device* device_surplus_t)
{
	for (int i = 0; i < nstates; i++)
	{
		delete &device_index[i];
		delete &device_surplus[i];
		delete &device_surplus_t[i];
	}
}
		
Data::Device::~Device()
{
	destroyDeviceData<<<1, 1>>>(nstates, index, surplus, surplus_t);
	CUDA_ERR_CHECK(cudaGetLastError());
	CUDA_ERR_CHECK(cudaDeviceSynchronize());			

	// Free temporary device space for pointers.
	CUDA_ERR_CHECK(cudaFree(index));
	CUDA_ERR_CHECK(cudaFree(surplus));
	CUDA_ERR_CHECK(cudaFree(surplus_t));
}

extern "C" Data* getData(int nstates)
{
	return new Data(nstates);
}

