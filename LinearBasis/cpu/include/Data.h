#ifndef DATA_H
#define DATA_H

#include <algorithm>
#include <assert.h>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string.h>
#include <utility> // pair
#include <vector>

#include "check.h"
#include "process.h"

namespace NAMESPACE {

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

	// Calculate the maximum number of non-zero values across individual rows.
	inline __attribute__((always_inline)) int maxRowPopulation() const
	{
		int result = 0;
		int vdim = dimX / 2;

		const std::pair<T, T> zero = std::make_pair((T)0, (T)0);

		std::map<int, int> freqs;
		for (int i = 0; i < dimY; i++)
			for (int j = 0; j < vdim; j++)
			{
				// Get pair.
				std::pair<T, T> value = std::make_pair(operator()(i, j), operator()(i, j + vdim));

				// If both indexes are zeros, do nothing.
				if (value == zero)
					continue;
			
				freqs[i]++;
			}

		for (std::map<int, int>::iterator i = freqs.begin(), e = freqs.end(); i != e; i++)
			result = std::max(result, i->second);
		
		return result;
	}
};

class Interpolator;

template<typename TIndex>
struct Index
{
	uint8_t i, j;
	TIndex index;

	inline __attribute__((always_inline))
	TIndex& rowind() { return index; }

	inline __attribute__((always_inline))
	bool isEmpty() { return (i == 0) && (j == 0); }	

	Index() : i(0), j(0), index(0) { }
	
	Index(short i_, short j_, int index_) : i(i_), j(j_), index(index_) { }

	template <class TValue>
	static inline void hashCombine(size_t& seed, const TValue& v)
	{
		std::hash<TValue> hasher;
		seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	}
	
	friend bool operator<(const Index& a, const Index& b)
	{
		size_t seedA = 0;
		hashCombine(seedA, a.i);
		hashCombine(seedA, a.j);
		hashCombine(seedA, a.index);

		size_t seedB = 0;
		hashCombine(seedB, b.i);
		hashCombine(seedB, b.j);
		hashCombine(seedB, b.index);

		return seedA < seedB;
	}
};

typedef std::vector<Index<uint16_t> > XPS;
typedef std::vector<uint32_t> Chains;

class Data
{
	int nstates;
	std::vector<int> nfreqs;
	std::vector<XPS> xps;
	std::vector<Chains> chains;
	std::vector<Matrix<real> > surplus;
	std::vector<bool> loadedStates;
	
	friend class Interpolator;

public :

	virtual void load(const char* filename, int istate);
	
	virtual void load(int dim, int vdim, int nno, int TotalDof, int Level, const Matrix<int>& index, int istate);
	
	virtual void clear();

	Data(int nstates);
	
	virtual ~Data();
};

} // namespace NAMESPACE

#endif // DATA_H

