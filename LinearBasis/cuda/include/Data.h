#ifndef DATA_H
#define DATA_H

#include <algorithm>
#include <assert.h>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include <vector>

#include "check.h"
#include "process.h"

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

// Aligned vector.
template<typename T>
class Vector
{
	std::vector<T, AlignedAllocator<T> > data;

public :
	Vector() : data(AlignedAllocator<T>()) { }

	Vector(int dim) : data(AlignedAllocator<T>()) { data.resize(dim); }

	Vector(int dim, T value) : data(dim, value, AlignedAllocator<T>()) { }

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

// Host memory matrix with all rows aligned
template<typename T, typename VectorType>
class MatrixHost
{
	VectorType data;
	int dimY, dimX, dimX_aligned;

public :
	MatrixHost() : data(VectorType()), dimY(0), dimX(0) { }

	MatrixHost(int dimY_, int dimX_) : data(VectorType()), dimY(dimY_), dimX(dimX_)
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

// Device memory matrix with all rows aligned
template<typename T>
class MatrixDevice
{
	T* data;
	bool dataOwner; // whether the instance owns its data pointer or not
	size_t size;
	int dimY, dimX, dimX_aligned;

public :
	__host__ __device__
	MatrixDevice() : data(NULL), dataOwner(true), size(0), dimY(0), dimX(0) { }

	__host__ __device__
	MatrixDevice(int dimY_, int dimX_) : data(NULL), dataOwner(true), size(0), dimY(dimY_), dimX(dimX_)
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
	~MatrixDevice()
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

template<typename T>
struct Matrix
{
	typedef MatrixHost<T, std::vector<T, AlignedAllocator<T> > > Host;
	typedef MatrixDevice<T> Device;
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

		Matrix<int>::Host* getIndex(int istate);
		Matrix<real>::Host* getSurplus(int istate);
		Matrix<real>::Host* getSurplus_t(int istate);

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

		Matrix<int>::Device* getIndex(int istate);
		Matrix<real>::Device* getSurplus(int istate);
		Matrix<real>::Device* getSurplus_t(int istate);

		void setIndex(int istate, Matrix<int>::Host* matrix);
		void setSurplus(int istate, Matrix<real>::Host* matrix);
		void setSurplus_t(int istate, Matrix<real>::Host* matrix);
	
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

