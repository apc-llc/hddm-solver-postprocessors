#ifndef DATA_H
#define DATA_H

#include <algorithm>
#include <assert.h>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string.h>
#include <vector>

#include "check.h"
#include "process.h"

// Custom allocator, using code by Sergei Danielian
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

template<typename T>
class Vector
{
	std::vector<T, AlignedAllocator<T> > data;

public :
	Vector() : data(AlignedAllocator<T>()) { }

	Vector(int dim) : data(AlignedAllocator<T>()) { data.resize(dim); }

	Vector(int dim, T value) : data(dim, value, AlignedAllocator<T>()) { }

	T* getData() { return &data[0]; }

	T& operator()(int x)
	{
		assert(x < data.size());
		return data[x];
	}
	
	int length() { return data.size(); }
	
	void resize(int length) { data.resize(length); }
};

template<typename T>
class Matrix
{
	std::vector<T, AlignedAllocator<T> > data;
	int dimY, dimX;

public :
	Matrix() : data(AlignedAllocator<T>()), dimY(0), dimX(0) { }

	Matrix(int dimY_, int dimX_) : data(AlignedAllocator<T>()), dimY(dimY_), dimX(dimX_)
	{
		data.resize(dimY_ * dimX_);
	}

	T* getData() { return &data[0]; }
	
	T& operator()(int y, int x)
	{
		int index = x + dimX * y;
		assert(index < data.size());
		return data[index];
	}

	int dimy() { return dimY; }

	int dimx() { return dimX; }
		
	void resize(int dimY_, int dimX_)
	{
		dimY = dimY_; dimX = dimX_;
		data.resize(dimY_ * dimX_);
	}
	
	void fill(T value)
	{
		std::fill(data.begin(), data.end(), value);
	}
};

class Interpolator;

class Data
{
	int nstates, dim, vdim, nno, TotalDof, Level;
	std::vector<Matrix<int> > index;
	std::vector<Matrix<real> > surplus, surplus_t;
	std::vector<bool> loadedStates;

	friend class Interpolator;

public :
	int getNno() const;

	void load(const char* filename, int istate);
	
	void clear();

	Data(int nstates);
};

#endif // DATA_H

