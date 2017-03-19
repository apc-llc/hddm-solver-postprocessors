#include "check.h"
#include "Data.h"
#include "interpolator.h"

#include <fstream>
#include <iostream>
#include <mpi.h>

using namespace NAMESPACE;
using namespace std;

int Data::getNno() const { return nno; }

class Data::Host::DataHost
{
	std::vector<Matrix<int>::Host::Dense> index;
	std::vector<Matrix<real>::Host::Dense> surplus, surplus_t;

public :

	DataHost(int nstates)
	{
		index.resize(nstates);
		surplus.resize(nstates);
		surplus_t.resize(nstates);
	}

	friend class Data::Host;
};

static bool isCompressed(const char* filename)
{
	MPI_Process* process;
	MPI_ERR_CHECK(MPI_Process_get(&process));

	ifstream infile;
	infile.open(filename, ifstream::binary);

	if (!infile.is_open())
	{
		cerr << "Error opening file: " << filename << endl;
		process->abort();
	}

	bool compressed = false;
	
	char format_marker[] = "          \0";
		
	infile.read(format_marker, strlen(format_marker));
	
	infile.close();
	
	if ((string)format_marker == "compressed")
		compressed = true;
	
	return compressed;
}

struct IndexPair
{
	unsigned short i, j;
};

template<typename T>
static void read_index(ifstream& infile, int nno, int dim, Matrix<int>::Host::Dense& index_)
{
	MPI_Process* process;
	MPI_ERR_CHECK(MPI_Process_get(&process));

	char index_marker[] = "index";
	infile.read(index_marker, strlen(index_marker));

	unsigned int index_nonzeros = 0;
	infile.read(reinterpret_cast<char*>(&index_nonzeros), sizeof(unsigned int));

	vector<int> A;
	A.resize(index_nonzeros);
	for (int i = 0, e = A.size(); i != e; i++)
		infile.read(reinterpret_cast<char*>(&A[i]), sizeof(IndexPair));

	vector<int> IA;
	IA.resize(nno + 1);
	{
		T value;
		infile.read(reinterpret_cast<char*>(&value), sizeof(T));
		IA[0] = value;
	}
	for (int i = 1, e = IA.size(); i != e; i++)
	{
		T value;
		infile.read(reinterpret_cast<char*>(&value), sizeof(T));
		IA[i] = (int)value + IA[i - 1];
	}
	
	if (IA[0] != 0)
	{
		cerr << "IA[0] must be 0" << endl;
		process->abort();
	}

	for (int i = 1, e = IA.size(); i != e; i++)
		if (IA[i] < IA[i - 1])
		{
			cerr << "IA[i] must be not less than IA[i - 1] - not true for IA[" << i << "] >= IA[" << (i - 1) <<
				"] : " << IA[i] << " < " << IA[i - 1] << endl;
			process->abort();
		}
	
	vector<int> JA;
	JA.resize(index_nonzeros);
	for (int i = 0, e = JA.size(); i != e; i++)
	{
		T value;
		infile.read(reinterpret_cast<char*>(&value), sizeof(T));
		JA[i] = (int)value;
	}
	
	for (int i = 0, e = JA.size(); i != e; i++)
	{
		if (JA[i] >= dim)
		{
			cerr << "JA[i] must be within column index range - not true for JA[" << i << "] = " << JA[i] << endl;
			process->abort();
		}
	}

	for (int i = 0, row = 0; row < nno; row++)
		for (int col = IA[row]; col < IA[row + 1]; col++, i++)
			index_(row, JA[i]) = A[i];
	
	//cout << (100 - (double)index_nonzeros / (nno * dim) * 100) << "% index sparsity" << endl;
}

template<typename T>
static void read_surplus(ifstream& infile, int nno, int TotalDof, Matrix<double>::Host::Dense& surplus_)
{
	MPI_Process* process;
	MPI_ERR_CHECK(MPI_Process_get(&process));

	char surplus_marker[] = "surplus";
	infile.read(surplus_marker, strlen(surplus_marker));

	unsigned int surplus_nonzeros = 0;
	infile.read(reinterpret_cast<char*>(&surplus_nonzeros), sizeof(unsigned int));

	vector<double> A;
	A.resize(surplus_nonzeros);
	for (int i = 0, e = A.size(); i != e; i++)
		infile.read(reinterpret_cast<char*>(&A[i]), sizeof(double));

	vector<int> IA;
	IA.resize(nno + 1);
	{
		T value;
		infile.read(reinterpret_cast<char*>(&value), sizeof(T));
		IA[0] = value;
	}
	for (int i = 1, e = IA.size(); i != e; i++)
	{
		T value;
		infile.read(reinterpret_cast<char*>(&value), sizeof(T));
		IA[i] = (int)value + IA[i - 1];
	}

	if (IA[0] != 0)
	{
		cerr << "IA[0] must be 0" << endl;
		process->abort();
	}

	for (int i = 1, e = IA.size(); i != e; i++)
		if (IA[i] < IA[i - 1])
		{
			cerr << "IA[i] must be not less than IA[i - 1] - not true for IA[" << i << "] >= IA[" << (i - 1) <<
				"] : " << IA[i] << " < " << IA[i - 1] << endl;
			process->abort();
		}
	
	vector<int> JA;
	JA.resize(surplus_nonzeros);
	for (int i = 0, e = JA.size(); i != e; i++)
	{
		T value;
		infile.read(reinterpret_cast<char*>(&value), sizeof(T));
		JA[i] = (int)value;
	}

	for (int i = 0, e = JA.size(); i != e; i++)
	{
		if (JA[i] >= TotalDof)
		{
			cerr << "JA[i] must be within column index range - not true for JA[" << i << "] = " << JA[i] << endl;
			process->abort();
		}
	}

	for (int i = 0, row = 0; row < nno; row++)
		for (int col = IA[row]; col < IA[row + 1]; col++, i++)
			surplus_(row, JA[i]) = A[i];
	
	//cout << (100 - (double)surplus_nonzeros / (nno * TotalDof) * 100) << "% surplus sparsity" << endl;
}

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

	bool compressed = isCompressed(filename);

	ifstream infile;
	if (compressed)
		infile.open(filename, ifstream::binary);
	else
		infile.open(filename, ios::in);

	if (!infile.is_open())
	{
		cerr << "Error opening file: " << filename << endl;
		process->abort();
	}

	if (compressed)
	{
		char format_marker[] = "          ";		
		infile.read(format_marker, strlen(format_marker));
		infile.read(reinterpret_cast<char*>(&dim), sizeof(int));
		infile.read(reinterpret_cast<char*>(&nno), sizeof(int)); 
		infile.read(reinterpret_cast<char*>(&TotalDof), sizeof(int));
		infile.read(reinterpret_cast<char*>(&Level), sizeof(int));
	}
	else
	{	
		infile >> dim;
		infile >> nno; 
		infile >> TotalDof;
		infile >> Level;
	}

	if (dim != params.nagents)
	{
		cerr << "File \"" << filename << "\" # of dimensions (" << dim << 
			") mismatches config (" << params.nagents << ")" << endl;
		process->abort();
	}

	// Pad all indexes to 4-element boundary.
	vdim = dim / AVX_VECTOR_SIZE;
	if (dim % AVX_VECTOR_SIZE) vdim++;
	int nsd = vdim * AVX_VECTOR_SIZE;

	Matrix<int>::Host::Dense& index = *host.getIndex(istate);
	Matrix<real>::Host::Dense& surplus = *host.getSurplus(istate);
	Matrix<real>::Host::Dense& surplus_t = *host.getSurplus_t(istate);
	
	index.resize(nno, nsd);
	index.fill(0);
	surplus.resize(nno, TotalDof);
	surplus.fill(0.0);

	// For better caching we use transposed surplus.
	surplus_t.resize(TotalDof, nno);
	surplus_t.fill(0.0);

	if (!compressed)
	{	
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
					IndexPair& pair = (IndexPair&)index(j, i);
					pair.i = value;
				}
			}
			for (int i = 0; i < dim; )
			{
				for (int v = 0; (v < AVX_VECTOR_SIZE) && (i < dim); v++, i++)
				{
					int value; infile >> value;
					value--;
					// Precompute "j" to merge two cases into one:
					// (((i) == 0) ? (1) : (1 - fabs((x) * (i) - (j)))).
					IndexPair& pair = (IndexPair&)index(j, i);
					if (!pair.i) value = 0;
					pair.j = value;
				}
			}
			for (int i = 0; i < TotalDof; i++)
			{
				double value; infile >> value;
				surplus(j, i) = value;
				surplus_t(i, j) = value;
			}
			j++;
		}
	}
	else
	{
		int szt = 0;
		if (dim <= numeric_limits<unsigned char>::max())
			szt = 1;
		else if (dim <= numeric_limits<unsigned short>::max())
			szt = 2;
		else
			szt = 4;
	
		//cout << "Using " << szt << "-byte IA/JA size for index" << endl;
	
		switch (szt)
		{
		case 1 :
			read_index<unsigned char>(infile, nno, dim, index);
			break;
		case 2 :
			read_index<unsigned short>(infile, nno, dim, index);
			break;
		case 4 :
			read_index<unsigned int>(infile, nno, dim, index);
			break;
		}

		if (TotalDof <= numeric_limits<unsigned char>::max())
			szt = 1;
		else if (TotalDof <= numeric_limits<unsigned short>::max())
			szt = 2;
		else
			szt = 4;
	
		//cout << endl;
		//cout << "Using " << szt << "-byte IA/JA size for surplus" << endl;
	
		switch (szt)
		{
		case 1 :
			read_surplus<unsigned char>(infile, nno, TotalDof, surplus);
			break;
		case 2 :
			read_surplus<unsigned short>(infile, nno, TotalDof, surplus);
			break;
		case 4 :
			read_surplus<unsigned int>(infile, nno, TotalDof, surplus);
			break;
		}
	}

	infile.close();
	
	loadedStates[istate] = true;

	// Copy data from host to device memory.
	device.setIndex(istate, &index);
	device.setSurplus(istate, &surplus);
//	device.setSurplus_t(istate, &surplus_t);
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

Data::~Data()
{
}

Data::Host::Host(int nstates)
{
	data.reset(new DataHost(nstates));
}

Matrix<int>::Host::Dense* Data::Host::getIndex(int istate)
{
	return &data->index[istate];
}

Matrix<real>::Host::Dense* Data::Host::getSurplus(int istate)
{
	return &data->surplus[istate];
}

Matrix<real>::Host::Dense* Data::Host::getSurplus_t(int istate)
{
	return &data->surplus_t[istate];
}

// Determine the size of type on device.
template<typename T>
static __global__ void deviceSizeOf(size_t* result)
{
	*result = sizeof(T);
}

template<typename T>
static __global__ void constructDeviceData(int length, char* ptr)
{
	for (int i = 0; i < length; i++)
		new(&ptr[i * sizeof(T)]) T();
}

template<typename T>
static __global__ void destroyDeviceData(int length, char* ptr)
{
	for (int i = 0; i < length; i++)
		delete (T*)&ptr[i * sizeof(T)];
}

// Host array of elements in device memory, whose size is
// determined on device in runtime.
template<typename T>
class DeviceSizedArray
{
	int length;
	size_t size;
	char* ptr;

public :

	DeviceSizedArray(int length_) : length(length_)
	{
		// Determine the size of target type.
		size_t* dSize;
		CUDA_ERR_CHECK(cudaMalloc(&dSize, sizeof(size_t)));
		deviceSizeOf<T><<<1, 1>>>(dSize);
		CUDA_ERR_CHECK(cudaGetLastError());
		CUDA_ERR_CHECK(cudaDeviceSynchronize());
		CUDA_ERR_CHECK(cudaMemcpy(&size, dSize, sizeof(size_t), cudaMemcpyDeviceToHost));
		CUDA_ERR_CHECK(cudaFree(dSize));
		
		// Make sure the size of type is the same on host and on device.
		if (size != sizeof(T))
		{
			cerr << "Unexpected unequal sizes of type T in DeviceSizedArray<T> on host and device" << endl;
			MPI_Process* process;
			MPI_ERR_CHECK(MPI_Process_get(&process));
			process->abort();
		}

		// Allocate array.		
		CUDA_ERR_CHECK(cudaMalloc(&ptr, size * length));
		
		// Construct individual array elements from within the device kernel code,
		// using placement new operator.
		constructDeviceData<T><<<1, 1>>>(length, ptr);
		CUDA_ERR_CHECK(cudaGetLastError());
		CUDA_ERR_CHECK(cudaDeviceSynchronize());
	}
	
	~DeviceSizedArray()
	{
		// Destroy individual array elements from within the device kernel code.
		destroyDeviceData<T><<<1, 1>>>(length, ptr);
		CUDA_ERR_CHECK(cudaGetLastError());
		CUDA_ERR_CHECK(cudaDeviceSynchronize());			

		// Delete each indivudual element on host, which triggers deletion of
		// data buffer previously allocated on host with cudaMalloc.
		vector<T> elements;
		elements.resize(length);
		CUDA_ERR_CHECK(cudaMemcpy(&elements[0], ptr, length * sizeof(T), cudaMemcpyDeviceToHost));
		for (int i = 0; i < length; i++)
		{
			// Set that matrix owns its underlying data buffer.
			elements[i].disownData();

			delete &elements[i];
		}

		// Free device memory used for array elements.
		CUDA_ERR_CHECK(cudaFree(ptr));
	}

	T& operator[](int i)
	{
		return (T&)ptr[i * size];
	}
};

class Data::Device::DataDevice
{
	// These containers shall be entirely in device memory, including vectors.
	DeviceSizedArray<Matrix<int>::Device::Dense> index;
	DeviceSizedArray<Matrix<real>::Device::Dense> surplus, surplus_t;

public :

	DataDevice(int nstates) : index(nstates), surplus(nstates), surplus_t(nstates) { }

	friend class Data::Device;
};	

Data::Device::Device(int nstates_) : nstates(nstates_)
{
	data.reset(new DataDevice(nstates_));
}

Matrix<int>::Device::Dense* Data::Device::getIndex(int istate)
{
	return &data->index[istate];
}

Matrix<real>::Device::Dense* Data::Device::getSurplus(int istate)
{
	return &data->surplus[istate];
}

Matrix<real>::Device::Dense* Data::Device::getSurplus_t(int istate)
{
	return &data->surplus_t[istate];
}

template<typename T>
static void setMatrix(int istate,
	MatrixHostDense<T, std::vector<T, AlignedAllocator<T> > > * pMatrixHost, MatrixDeviceDense<T>* pMatrixDevice)
{
	MatrixHostDense<T, std::vector<T, AlignedAllocator<T> > >& matrixHost = *pMatrixHost;

	MatrixDeviceDense<T> matrixDevice;
	CUDA_ERR_CHECK(cudaMemcpy(&matrixDevice, pMatrixDevice, sizeof(MatrixDeviceDense<T>), cudaMemcpyDeviceToHost));
	matrixDevice.resize(matrixHost.dimy(), matrixHost.dimx());

	// It is assumed to be safe to copy padded data from host to device matrix,
	// as they use the same memory allocation policy.
	size_t size = (ptrdiff_t)&matrixHost(matrixHost.dimy() - 1, matrixHost.dimx() - 1) -
		(ptrdiff_t)matrixHost.getData() + sizeof(T);
	CUDA_ERR_CHECK(cudaMemcpy(matrixDevice.getData(), pMatrixHost->getData(), size, cudaMemcpyHostToDevice));

	// Set that matrix does not own its underlying data buffer.
	matrixDevice.disownData();

	CUDA_ERR_CHECK(cudaMemcpy(pMatrixDevice, &matrixDevice, sizeof(MatrixDeviceDense<T>), cudaMemcpyHostToDevice));
}

void Data::Device::setIndex(int istate, Matrix<int>::Host::Dense* matrix)
{
	setMatrix<int>(istate, matrix, &data->index[istate]);
}

void Data::Device::setSurplus(int istate, Matrix<real>::Host::Dense* matrix)
{
	setMatrix<double>(istate, matrix, &data->surplus[istate]);
}

void Data::Device::setSurplus_t(int istate, Matrix<real>::Host::Dense* matrix)
{
	setMatrix<double>(istate, matrix, &data->surplus_t[istate]);
}
		
extern "C" Data* getData(int nstates)
{
	return new Data(nstates);
}

