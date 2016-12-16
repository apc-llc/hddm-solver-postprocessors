#ifndef VECTOR_H
#define VECTOR_H

#include "DataContainer.h"

namespace cuda {

namespace Vector {

// Host memory aligned vector.
template<typename T, typename VectorType = std::vector<T, AlignedAllocator::Host<T> > >
class Host
{
	VectorType data;

public :
	Host() : data(VectorType()) { }

	Host(int dim) : data(VectorType()) { data.resize(dim); }

	Host(int dim, T value) : data(VectorType())
	{
		data.resize(dim, value);
	}

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

#if defined(__CUDACC__)

// Device memory aligned vector.
template<typename T>
class Device : public DataContainer<T>
{
	int dim, dimAligned;

	__host__ __device__
	static int dimAlign(int dim)
	{
		int dimAligned = dim;
		if (dim % AVX_VECTOR_SIZE)
			dimAligned = dim + AVX_VECTOR_SIZE - dim % AVX_VECTOR_SIZE;
		return dimAligned;
	}

public :

	__host__ __device__
	Device() : DataContainer<T>(), dim(0) { }

	__host__ __device__
	Device(int dim_) : DataContainer<T>(dimAlign(dim_)), dim(dim_), dimAligned(dimAlign(dim_)) { }
	
	__host__ __device__
	~Device() { }
	
	__host__ __device__
	inline __attribute__((always_inline)) T& operator()(int x)
	{
		assert(x < dimAligned);
		return DataContainer<T>::modifyDataAt(x);
	}

	__host__ __device__
	inline __attribute__((always_inline)) const T& operator()(int x) const
	{
		assert(x < dimAligned);
		return const_cast<Device*>(this)->accessDataAt(x);
	}

	__host__ __device__
	inline __attribute__((always_inline)) int length() { return dim; }

	__host__ __device__
	inline __attribute__((always_inline)) void resize(int dim_)
	{
		dim = dim_;
		dimAligned = dimAlign(dim_);
		DataContainer<T>::deinitialize();
		DataContainer<T>::initialize(dimAligned);
	}
};

#else

template<typename T>
class Dense;

#endif // __CUDACC__

} // namespace Vector

} // namespace cuda

#endif // VECTOR_H

