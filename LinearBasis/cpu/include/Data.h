#ifndef DATA_H
#define DATA_H

#include <algorithm>
#include <assert.h>
#include <cstddef>
#include <cstdlib>
#include <vector>

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
		throw(std::bad_alloc, std::length_error)
	{
		void* ptr;
		posix_memalign(&ptr, AVX_VECTOR_SIZE * sizeof(T), n * sizeof(T));

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
	int dimX, dimY;

public :
	Matrix() : data(AlignedAllocator<T>()), dimX(0), dimY(0) { }

	Matrix(int dimX_, int dimY_) : data(AlignedAllocator<T>()), dimX(dimX_), dimY(dimY_)
	{
		data.resize(dimX_ * dimY_);
	}

	T* getData() { return &data[0]; }
	
	T& operator()(int x, int y)
	{
		int index = y + dimY * x;
		assert(index < data.size());
		return data[index];
	}

	int dimx() { return dimX; }
	
	int dimy() { return dimY; }
	
	void resize(int dimX_, int dimY_)
	{
		dimX = dimX_; dimY = dimY_;
		data.resize(dimX_ * dimY_);
	}
	
	void fill(T value)
	{
		std::fill(data.begin(), data.end(), value);
	}
};

class Interpolator;

class Data
{
	int dim, vdim, nno, TotalDof, Level;
	Matrix<int> index;
	Matrix<real> surplus, surplus_t;

	friend class Interpolator;

public :
	int getNno() const;

	void load(const char* filename);

	Data();
};

#endif // DATA_H

