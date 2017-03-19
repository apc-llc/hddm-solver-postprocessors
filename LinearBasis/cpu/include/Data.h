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

namespace cpu {

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

		// POSIX requires minimum alignment to be sizeof(void*). 
		int err = posix_memalign(&ptr, std::max(sizeof(void*), AVX_VECTOR_SIZE * sizeof(T)), size);
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
	
	inline __attribute__((always_inline)) int length() const { return data.size(); }
	
	inline __attribute__((always_inline)) void resize(int length) { data.resize(length); }
};

// Matrix with all rows aligned
template<typename T>
class Matrix
{
	std::vector<T, AlignedAllocator<T> > data;
	int dimY, dimX, dimX_aligned;

public :
	Matrix() : data(AlignedAllocator<T>()), dimY(0), dimX(0) { }

	Matrix(int dimY_, int dimX_) : data(AlignedAllocator<T>()), dimY(dimY_), dimX(dimX_)
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

	inline __attribute__((always_inline)) int dimy() const { return dimY; }

	inline __attribute__((always_inline)) int dimx() const { return dimX; }
		
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

struct AVXIndex
{
	uint8_t i[AVX_VECTOR_SIZE], j[AVX_VECTOR_SIZE];
	
	AVXIndex()
	{
		memset(i, 0, sizeof(i));
		memset(j, 0, sizeof(j));
	}
	
	__attribute__((always_inline))
	bool isEmpty() const
	{
		AVXIndex empty;
		if (memcmp(this, &empty, sizeof(AVXIndex)) == 0)
			return true;
		
		return false;
	}

	__attribute__((always_inline))
	bool isEmpty(int k) const
	{
		return (i[k] == 0) && (j[k] == 0);
	}
};

// Compressed index matrix packed into AVX-sized chunks.
// Specialized from vector in order to place the length value
// right before its corresponding row data (for caching).
class AVXIndexes: private std::vector<char, AlignedAllocator<AVXIndex> >
{
	int nnoMax;
	int dim;
	
	static const int szlength = 4 * sizeof(double);

	__attribute__((always_inline))
	static int nnoMaxAlign(int nnoMax_)
	{
		// Pad indexes rows to AVX_VECTOR_SIZE.
		if (nnoMax_ % AVX_VECTOR_SIZE)
			nnoMax_ += AVX_VECTOR_SIZE - nnoMax_ % AVX_VECTOR_SIZE;

		return nnoMax_ / AVX_VECTOR_SIZE;
	}
	
	__attribute__((always_inline))
	int& length(int j)
	{
		return reinterpret_cast<int*>(
			reinterpret_cast<char*>(&this->operator()(0, j)) - sizeof(int))[0];
	}

	__attribute__((always_inline))
	const int& length(int j) const
	{
		return reinterpret_cast<const int*>(
			reinterpret_cast<const char*>(&this->operator()(0, j)) - sizeof(int))[0];
	}
	
	__attribute__((always_inline))
	void setLength(int j, int length_)
	{
		length(j) = length_;
	}

public :

	AVXIndexes() :
		nnoMax(0), dim(0),
		std::vector<char, AlignedAllocator<AVXIndex> >()
	
	{ }

	AVXIndexes(int nnoMax_, int dim_) :
		nnoMax(nnoMaxAlign(nnoMax_)), dim(dim_),
		std::vector<char, AlignedAllocator<AVXIndex> >(
			dim_ * (nnoMaxAlign(nnoMax_) * sizeof(AVXIndex) + szlength))

	{ }
	
	void resize(int nnoMax_, int dim_)
	{
		nnoMax = nnoMaxAlign(nnoMax_);
		dim = dim_;

		vector<char, AlignedAllocator<AVXIndex> >::resize(
			dim_ * (nnoMaxAlign(nnoMax_) * sizeof(AVXIndex) + szlength));
	}
	
	__attribute__((always_inline))
	AVXIndex& operator()(int i, int j)
	{
		return *reinterpret_cast<AVXIndex*>(
			&std::vector<char, AlignedAllocator<AVXIndex> >::operator[]((j * nnoMax + i) * sizeof(AVXIndex) + (j + 1) * szlength));
	}

	__attribute__((always_inline))
	const AVXIndex& operator()(int i, int j) const
	{
		return *reinterpret_cast<const AVXIndex*>(
			&std::vector<char, AlignedAllocator<AVXIndex> >::operator[]((j * nnoMax + i) * sizeof(AVXIndex) + (j + 1) * szlength));
	}
	
	__attribute__((always_inline))
	int getLength(int j) const
	{
		return length(j);
	}
	
	void calculateLengths()
	{
		for (int j = 0; j < dim; j++)
		{
			int length = 0;
			for (int i = 0; i < nnoMax; i++)
			{
				AVXIndex& index = this->operator()(i, j);
				if (index.isEmpty())
					break;
				
				length++;
			}
			
			setLength(j, length);
		}
	}
};

typedef std::vector<AVXIndexes> AVXIndexMatrix;

// Index transition matrix between row indexes of different frequencies.
typedef std::vector<std::vector<uint32_t> > TransMatrix;

class Interpolator;

class Data
{
	int nstates, dim, vdim, nno, TotalDof, Level;
	std::vector<AVXIndexMatrix> avxinds;
	std::vector<TransMatrix> trans;
	std::vector<Matrix<real> > surplus;
	std::vector<bool> loadedStates;
	
	friend class Interpolator;

public :
	virtual int getNno() const;

	virtual void load(const char* filename, int istate);
	
	virtual void clear();

	Data(int nstates);
	
	virtual ~Data();
};

} // namespace cpu

#endif // DATA_H

