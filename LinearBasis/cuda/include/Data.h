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

namespace cuda {

// Custom host memory allocator, using code by Sergei Danielian
// https://github.com/gahcep/Allocators
template <class T>
class AlignedAllocator
{
public:
	using value_type = T;
	using pointer = T*;
	using const_pointer = const T*;
	using reference = T&;
	using const_reference = const T&;
	using size_type = std::size_t;
	using difference_type = std::ptrdiff_t;

	// Rebind
	template <class U>
	struct rebind { using other = AlignedAllocator<U>; };

	AlignedAllocator() { }
	AlignedAllocator(const AlignedAllocator&) { }
	template <class U>
	AlignedAllocator(const AlignedAllocator<U>&) { }

	// Allocators are not required to be assignable
	AlignedAllocator& operator=(const AlignedAllocator& other) = delete;
	~AlignedAllocator() { }

	// Obtains the address of an object
	pointer address(reference r) const { return &r; }
	const_pointer address(const_reference r) const { return &r; }

	// Returns the largest supported allocation size
	size_type max_size() const
	{
		return (static_cast<size_t>(0) - static_cast<size_t>(1)) / sizeof(T);
	}

	// Equality of allocators does not imply that they must have exactly
	// the same internal state,  only that they must both be able to
	// deallocate memory that was allocated with either allocator
	bool operator!=(const AlignedAllocator& other) { return !(*this == other); }
	bool operator==(const AlignedAllocator&) { return true; }

	// allocation
	pointer allocate(size_type n, std::allocator<void>::const_pointer = 0) const
	{
		// Align & pad to vector size and zero.
		void* ptr;
		size_t size = n * sizeof(T);
		if (size % (AVX_VECTOR_SIZE * sizeof(T)))
			size += AVX_VECTOR_SIZE * sizeof(T) - size % (AVX_VECTOR_SIZE * sizeof(T)); 
		int err = posix_memalign(&ptr, AVX_VECTOR_SIZE * sizeof(T), size);
		if (err != 0)
		{
			using namespace std;
			cerr << "posix_memalign returned error " << err;
			if (err == EINVAL) cerr << " (EINVAL)";
			else if (err == ENOMEM) cerr << " (ENOMEM)";
			cerr << endl;
			MPI_Process* process;
			MPI_ERR_CHECK(MPI_Process_get(&process));
			process->abort();
		}
		memset(ptr, 0, size);

		return static_cast<pointer>(ptr);
	}

	void deallocate(pointer ptr, size_type n)
	{
		free(ptr);
	}
};

// Host memory aligned vector.
template<typename T, typename VectorType>
class VectorHost
{
	VectorType data;

public :
	VectorHost() : data(VectorType()) { }

	VectorHost(int dim) : data(AlignedAllocator<T>()) { data.resize(dim); }

	VectorHost(int dim, T value) : data(dim, value, AlignedAllocator<T>()) { }

	inline __attribute__((always_inline)) T* getData() { return &data[0]; }

	inline __attribute__((always_inline)) T& operator()(int x)
	{
		assert(x < data.size());
		return data[x];
	}

	inline __attribute__((always_inline)) const T& operator()(int x) const
	{
		assert(x < data.size());
		return data[x];
	}
	
	inline __attribute__((always_inline)) int length() { return data.size(); }
	
	inline __attribute__((always_inline)) void resize(int length) { data.resize(length); }
};

// Device memory aligned vector.
template<typename T>
class VectorDevice
{
	T* data;
	bool dataOwner; // whether the instance owns its data pointer or not
	size_t size;
	int dim, dim_aligned;

public :
	__host__ __device__
	VectorDevice() : data(NULL), dataOwner(true), size(0), dim(0) { }

	__host__ __device__
	VectorDevice(int dim_) : data(NULL), dataOwner(true), size(0), dim(dim_)
	{
		dim_aligned = dim_;
		if (dim_ % AVX_VECTOR_SIZE)
			dim_aligned = dim + AVX_VECTOR_SIZE - dim_ % AVX_VECTOR_SIZE;
#if defined(__CUDA_ARCH__)
		data = new T[dim];
#else
		CUDA_ERR_CHECK(cudaMalloc(&data, dim * sizeof(T)));
#endif
	}
	
	__host__ __device__
	~VectorDevice()
	{
		if (!dataOwner) return;
#if defined(__CUDA_ARCH__)
		if (data) delete[] data;
#else
		CUDA_ERR_CHECK(cudaFree(data));
#endif
	}

	// Become an owner of the underlying data pointer.
	__host__ __device__
	void ownData() { dataOwner = true; }

	// Disown the underlying data pointer.
	__host__ __device__
	void disownData() { dataOwner = false; }

	__host__ __device__
	inline __attribute__((always_inline)) T* getData() { return &data[0]; }
	
	__device__
	inline __attribute__((always_inline)) T& operator()(int x)
	{
		assert(x < dim);
		return data[x];
	}

	__device__
	inline __attribute__((always_inline)) const T& operator()(int x) const
	{
		assert(x < dim);
		return data[x];
	}

	__device__
	inline __attribute__((always_inline)) int length() { return dim; }

	__host__ __device__
	inline __attribute__((always_inline)) void resize(int length)
	{
		dim = length;
		if (dim % AVX_VECTOR_SIZE)
			dim_aligned = dim + AVX_VECTOR_SIZE - dim % AVX_VECTOR_SIZE;
#if defined(__CUDA_ARCH__)
		if (data)
			delete[] data;
#else
		if (data)
			CUDA_ERR_CHECK(cudaFree(data));
#endif
#if defined(__CUDA_ARCH__)
		data = new T[dim];
#else
		CUDA_ERR_CHECK(cudaMalloc(&data, dim * sizeof(T)));
#endif
	}
};

template<typename T>
struct Vector
{
	typedef VectorHost<T, std::vector<T, AlignedAllocator<T> > > Host;
	typedef VectorDevice<T> Device;
};

// Host memory matrix with all rows aligned
template<typename T, typename VectorType>
class MatrixHostDense
{
	VectorType data;
	int dimY, dimX, dimX_aligned;

public :
	MatrixHostDense() : data(VectorType()), dimY(0), dimX(0) { }

	MatrixHostDense(int dimY_, int dimX_) : data(VectorType()), dimY(dimY_), dimX(dimX_)
	{
		dimX_aligned = dimX_;
		if (dimX_ % AVX_VECTOR_SIZE)
			dimX_aligned = dimX + AVX_VECTOR_SIZE - dimX_ % AVX_VECTOR_SIZE;
		data.resize(dimY_ * dimX_aligned);
	}

	inline __attribute__((always_inline)) T* getData() { return &data[0]; }
	
	inline __attribute__((always_inline)) T& operator()(int y, int x)
	{
		assert(x < dimX);
		assert(y < dimY);
		int index = x + dimX_aligned * y;
		assert(index < data.size());
		return data[index];
	}

	inline __attribute__((always_inline)) const T& operator()(int y, int x) const
	{
		assert(x < dimX);
		assert(y < dimY);
		int index = x + dimX_aligned * y;
		assert(index < data.size());
		return data[index];
	}

	inline __attribute__((always_inline)) int dimy() { return dimY; }

	inline __attribute__((always_inline)) int dimx() { return dimX; }
		
	inline __attribute__((always_inline)) void resize(int dimY_, int dimX_)
	{
		dimY = dimY_; dimX = dimX_;
		dimX_aligned = dimX_;
		if (dimX_ % AVX_VECTOR_SIZE)
			dimX_aligned = dimX + AVX_VECTOR_SIZE - dimX_ % AVX_VECTOR_SIZE;
		data.resize(dimY_ * dimX_aligned);
	}
	
	inline __attribute__((always_inline)) void fill(T value)
	{
		std::fill(data.begin(), data.end(), value);
	}
};

// Host memory sparse matrix in CSR format
template<typename TValue, typename TIndex, template<typename, typename> class TVector, template<typename> class TAllocator>
class MatrixHostSparseCSR
{
	TVector<TValue, TAllocator<TValue> > a;
	TVector<TIndex, TAllocator<TIndex> > ia, ja;
	int dimY, dimX, nnZ;

public :
	MatrixHostSparseCSR() :
		a(TVector<TValue, TAllocator<TValue> >()),
		ia(TVector<TIndex, TAllocator<TIndex> >()),
		ja(TVector<TIndex, TAllocator<TIndex> >()),
		dimY(0), dimX(0), nnZ(0)
	{ }

	MatrixHostSparseCSR(int dimY_, int dimX_, int nnz_) :
		a(TVector<TValue, TAllocator<TValue> >()),
		ia(TVector<TIndex, TAllocator<TIndex> >()),
		ja(TVector<TIndex, TAllocator<TIndex> >()),
		dimY(dimY_), dimX(dimX_), nnZ(nnz_)
	{
		a.resize(nnZ);
		ia.resize(dimY + 1);
		ja.resize(nnZ);
	}

	inline __attribute__((always_inline)) TValue& A(int i)
	{
		assert(i < nnZ);
		return a[i];
	}

	inline __attribute__((always_inline)) const TValue& A(int i) const
	{
		assert(i < nnZ);
		return a[i];
	}
	
	inline __attribute__((always_inline)) TIndex& IA(int i)
	{
		assert(i < dimY + 1);
		return ia[i];
	}

	inline __attribute__((always_inline)) const TIndex& IA(int i) const
	{
		assert(i < dimY + 1);
		return ia[i];
	}
	
	inline __attribute__((always_inline)) TIndex& JA(int i)
	{
		assert(i < nnZ);
		return ja[i];
	}

	inline __attribute__((always_inline)) const TIndex& JA(int i) const
	{
		assert(i < nnZ);
		return ja[i];
	}

	inline __attribute__((always_inline)) const TValue& operator()(int y, int x) const
	{
		assert(x < dimX);
		assert(y < dimY);

		for (int i = 0; ; )
		{
			for (int row = 0; row < y; row++)
				for (int col = IA[row]; (col < IA[row + 1]) && (i < nnZ); col++)
					i++;

			assert (i < nnZ);

			for (int col = IA[y]; col < IA[y + 1]; col++, i++)
				if (JA[i] == x) return A[i];

			return (TValue) 0;
		}
	}

	inline __attribute__((always_inline)) int dimy() { return dimY; }

	inline __attribute__((always_inline)) int dimx() { return dimX; }

	inline __attribute__((always_inline)) int nnz() { return nnZ; }
		
	inline __attribute__((always_inline)) void resize(int dimY_, int dimX_, int nnz_)
	{
		dimY = dimY_; dimX = dimX_; nnZ = nnz_;
		a.resize(nnZ);
		ia.resize(dimY + 1);
		ja.resize(nnZ);
	}
	
	inline __attribute__((always_inline)) void fill(TValue value)
	{
		std::fill(a.begin(), a.end(), value);
	}
};

// Device memory matrix with all rows aligned
template<typename T>
class MatrixDeviceDense
{
	T* data;
	bool dataOwner; // whether the instance owns its data pointer or not
	size_t size;
	int dimY, dimX, dimX_aligned;

public :
	__host__ __device__
	MatrixDeviceDense() : data(NULL), dataOwner(true), size(0), dimY(0), dimX(0) { }

	__host__ __device__
	MatrixDeviceDense(int dimY_, int dimX_) : data(NULL), dataOwner(true), size(0), dimY(dimY_), dimX(dimX_)
	{
		dimX_aligned = dimX_;
		if (dimX_ % AVX_VECTOR_SIZE)
			dimX_aligned = dimX + AVX_VECTOR_SIZE - dimX_ % AVX_VECTOR_SIZE;
		size = dimY_ * dimX_aligned;
#if defined(__CUDA_ARCH__)
		data = new T[size];
#else
		CUDA_ERR_CHECK(cudaMalloc(&data, size * sizeof(T)));
#endif
	}
	
	__host__ __device__
	~MatrixDeviceDense()
	{
		if (!dataOwner) return;
#if defined(__CUDA_ARCH__)
		if (data) delete[] data;
#else
		CUDA_ERR_CHECK(cudaFree(data));
#endif
	}

	// Become an owner of the underlying data pointer.
	__host__ __device__
	void ownData() { dataOwner = true; }

	// Disown the underlying data pointer.
	__host__ __device__
	void disownData() { dataOwner = false; }

	__host__ __device__
	inline __attribute__((always_inline)) T* getData() { return &data[0]; }
	
	__device__
	inline __attribute__((always_inline)) T& operator()(int y, int x)
	{
		assert(x < dimX);
		assert(y < dimY);
		int index = x + dimX_aligned * y;
		assert(index < size);
		return data[index];
	}

	__device__
	inline __attribute__((always_inline)) const T& operator()(int y, int x) const
	{
		assert(x < dimX);
		assert(y < dimY);
		int index = x + dimX_aligned * y;
		assert(index < size);
		return data[index];
	}

	__device__
	inline __attribute__((always_inline)) int dimy() { return dimY; }

	__device__
	inline __attribute__((always_inline)) int dimx() { return dimX; }
		
	__host__ __device__
	inline __attribute__((always_inline)) void resize(int dimY_, int dimX_)
	{
		dimY = dimY_; dimX = dimX_;
		dimX_aligned = dimX_;
		if (dimX_ % AVX_VECTOR_SIZE)
			dimX_aligned = dimX + AVX_VECTOR_SIZE - dimX_ % AVX_VECTOR_SIZE;
#if defined(__CUDA_ARCH__)
		if (data)
			delete[] data;
#else
		if (data)
			CUDA_ERR_CHECK(cudaFree(data));
#endif
		size = dimY_ * dimX_aligned;
#if defined(__CUDA_ARCH__)
		data = new T[size];
#else
		CUDA_ERR_CHECK(cudaMalloc(&data, size * sizeof(T)));
#endif
	}
};

// Host memory sparse matrix in CSR format
template<typename TValue, typename TIndex>
class MatrixDeviceSparseCSR
{
	TValue *a;
	TIndex *ia, *ja;
	bool dataOwner; // whether the instance owns its data pointer or not
	int dimY, dimX, nnZ;

public :
	MatrixDeviceSparseCSR() :
		a(NULL), ia(NULL), ja(NULL),
		dimY(0), dimX(0), nnZ(0)
	{ }

	MatrixDeviceSparseCSR(int dimY_, int dimX_, int nnz_) :
		a(NULL), ia(NULL), ja(NULL),
		dimY(dimY_), dimX(dimX_), nnZ(nnz_)
	{
#if defined(__CUDA_ARCH__)
		a = new TValue[nnZ];
		ia = new TIndex[dimY + 1];
		ja = new TIndex[nnZ];
#else
		CUDA_ERR_CHECK(cudaMalloc(&a, nnZ * sizeof(TValue)));
		CUDA_ERR_CHECK(cudaMalloc(&ia, (dimY + 1) * sizeof(TIndex)));
		CUDA_ERR_CHECK(cudaMalloc(&ja, nnZ * sizeof(TIndex)));
#endif
	}

	__host__ __device__
	~MatrixDeviceSparseCSR()
	{
		if (!dataOwner) return;
#if defined(__CUDA_ARCH__)
		if (a) delete[] a;
		if (ia) delete[] ia;
		if (ja) delete[] ja;
#else
		CUDA_ERR_CHECK(cudaFree(a));
		CUDA_ERR_CHECK(cudaFree(ia));
		CUDA_ERR_CHECK(cudaFree(ja));
#endif
	}

	// Become an owner of the underlying data pointer.
	__host__ __device__
	void ownData() { dataOwner = true; }

	// Disown the underlying data pointer.
	__host__ __device__
	void disownData() { dataOwner = false; }

	__device__
	inline __attribute__((always_inline)) TValue& A(int i)
	{
		assert(i < nnZ);
		return a[i];
	}

	__device__
	inline __attribute__((always_inline)) const TValue& A(int i) const
	{
		assert(i < nnZ);
		return a[i];
	}

	__device__
	inline __attribute__((always_inline)) TIndex& IA(int i)
	{
		assert(i < dimY + 1);
		return ia[i];
	}

	__device__
	inline __attribute__((always_inline)) const TIndex& IA(int i) const
	{
		assert(i < dimY + 1);
		return ia[i];
	}
	
	__device__
	inline __attribute__((always_inline)) TIndex& JA(int i)
	{
		assert(i < nnZ);
		return ja[i];
	}

	__device__
	inline __attribute__((always_inline)) const TIndex& JA(int i) const
	{
		assert(i < nnZ);
		return ja[i];
	}

	__device__
	inline __attribute__((always_inline)) const TValue& operator()(int y, int x) const
	{
		assert(x < dimX);
		assert(y < dimY);

		for (int i = 0; ; )
		{
			for (int row = 0; row < y; row++)
				for (int col = IA[row]; (col < IA[row + 1]) && (i < nnZ); col++)
					i++;

			assert (i < nnZ);

			for (int col = IA[y]; col < IA[y + 1]; col++, i++)
				if (JA[i] == x) return A[i];

			return (TValue) 0;
		}
	}

	__device__
	inline __attribute__((always_inline)) int dimy() { return dimY; }

	__device__
	inline __attribute__((always_inline)) int dimx() { return dimX; }

	__device__
	inline __attribute__((always_inline)) int nnz() { return nnZ; }
		
	__host__ __device__
	inline __attribute__((always_inline)) void resize(int dimY_, int dimX_, int nnz_)
	{
		dimY = dimY_; dimX = dimX_; nnZ = nnz_;
#if defined(__CUDA_ARCH__)
		if (a) delete[] a;
		if (ia) delete[] ia;
		if (ja) delete[] ja;
#else
		if (a) CUDA_ERR_CHECK(cudaFree(a));
		if (ia) CUDA_ERR_CHECK(cudaFree(ia));
		if (ja) CUDA_ERR_CHECK(cudaFree(ja));
#endif
#if defined(__CUDA_ARCH__)
		a = new TValue[nnZ];
		ia = new TIndex[dimY + 1];
		ja = new TIndex[nnZ];
#else
		CUDA_ERR_CHECK(cudaMalloc(&a, nnZ * sizeof(TValue)));
		CUDA_ERR_CHECK(cudaMalloc(&ia, (dimY + 1) * sizeof(TIndex)));
		CUDA_ERR_CHECK(cudaMalloc(&ja, nnZ * sizeof(TIndex)));
#endif
	}
};

template<typename T>
struct Matrix
{
	struct Host
	{
		typedef MatrixHostDense<T, std::vector<T, AlignedAllocator<T> > > Dense;
		
		template<class TIndex>
		struct Sparse
		{
			typedef MatrixHostSparseCSR<T, TIndex, std::vector, AlignedAllocator> CSR;
		};
	};

	struct Device
	{
		typedef MatrixDeviceDense<T> Dense;
		
		template<class TIndex>
		struct Sparse
		{
			typedef MatrixDeviceSparseCSR<T, TIndex> CSR;
		};
	};
};

class Interpolator;

class Data
{
	int nstates, dim, vdim, nno, TotalDof, Level;
	
	class Host
	{
		class DataHost;

		// Opaque internal data container.
		std::unique_ptr<DataHost> data;
	
	public :

		Matrix<int>::Host::Dense* getIndex(int istate);
		Matrix<real>::Host::Dense* getSurplus(int istate);
		Matrix<real>::Host::Dense* getSurplus_t(int istate);

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

		Matrix<int>::Device::Dense* getIndex(int istate);
		Matrix<real>::Device::Dense* getSurplus(int istate);
		Matrix<real>::Device::Dense* getSurplus_t(int istate);

		void setIndex(int istate, Matrix<int>::Host::Dense* matrix);
		void setSurplus(int istate, Matrix<real>::Host::Dense* matrix);
		void setSurplus_t(int istate, Matrix<real>::Host::Dense* matrix);
	
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

	virtual ~Data();
};

} // namespace cuda

#endif // DATA_H

