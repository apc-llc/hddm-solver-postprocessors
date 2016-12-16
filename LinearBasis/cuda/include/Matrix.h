#ifndef MATRIX_H
#define MATRIX_H

#include "DataContainer.h"

namespace cuda {

namespace Matrix {

namespace Host {

// Host memory matrix with all rows aligned
template<typename T, typename VectorType = std::vector<T, AlignedAllocator::Host<T> > >
class Dense
{
	VectorType data;
	int dimY, dimX, dimX_aligned;

public :
	Dense() : data(VectorType()), dimY(0), dimX(0) { }

	Dense(int dimY_, int dimX_) : data(VectorType()), dimY(dimY_), dimX(dimX_)
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

namespace Sparse {

// Host memory sparse matrix in CSR format
template<typename TValue, typename TIndex, template<typename, typename> class TVector = std::vector, template<typename> class TAllocator = AlignedAllocator::Host>
class CSR
{
	TVector<TValue, TAllocator<TValue> > a;
	TVector<TIndex, TAllocator<TIndex> > ia, ja;
	int dimY, dimX, nnZ;

public :
	CSR() :
		a(TVector<TValue, TAllocator<TValue> >()),
		ia(TVector<TIndex, TAllocator<TIndex> >()),
		ja(TVector<TIndex, TAllocator<TIndex> >()),
		dimY(0), dimX(0), nnZ(0)
	{ }

	CSR(int dimY_, int dimX_, int nnz_) :
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

} // namespace Sparse

} // namespace Host

#if defined(__CUDACC__)

namespace Device {

// Device memory matrix with all rows aligned
template<typename T>
class Dense : public DataContainer<T>
{
	size_t size;
	int dimY, dimX, dimXAligned;

	__host__ __device__
	static int dimAlign(int dim)
	{
		int dimAligned = dim;
		if (dim % AVX_VECTOR_SIZE)
			dimAligned = dim + AVX_VECTOR_SIZE - dim % AVX_VECTOR_SIZE;
		return dimAligned;
	}
	
	__host__ __device__
	static int sizeAlign(int dimY, int dimX)
	{
		return dimY * dimAlign(dimX);
	}

public :
	__host__ __device__
	Dense() : DataContainer<T>(), size(0), dimY(0), dimX(0) { }

	__host__ __device__
	Dense(int dimY_, int dimX_) : DataContainer<T>(sizeAlign(dimY_, dimX_)), size(sizeAlign(dimY_, dimX_)),
		dimY(dimY_), dimX(dimX_), dimXAligned(dimAlign(dimX_)) { }

	__host__ __device__
	~Dense() { }
	
	__host__ __device__
	inline __attribute__((always_inline)) T& operator()(int y, int x)
	{
		assert(x < dimXAligned);
		assert(y < dimY);
		int index = x + dimXAligned * y;
		assert(index < size);
		return DataContainer<T>::modifyDataAt(index);
	}

	__host__ __device__
	inline __attribute__((always_inline)) const T& operator()(int y, int x) const
	{
		assert(x < dimXAligned);
		assert(y < dimY);
		int index = x + dimXAligned * y;
		assert(index < size);
		return const_cast<Dense*>(this)->accessDataAt(index);
	}

	__host__ __device__
	inline __attribute__((always_inline)) int dimy() { return dimY; }

	__host__ __device__
	inline __attribute__((always_inline)) int dimx() { return dimX; }
		
	__host__ __device__
	inline __attribute__((always_inline)) void resize(int dimY_, int dimX_)
	{
		dimY = dimY_; dimX = dimX_;
		dimXAligned = dimAlign(dimX_);
		size = sizeAlign(dimY_, dimX_);
		DataContainer<T>::deinitialize();
		DataContainer<T>::initialize(size);
	}

	template<typename VectorType = std::vector<T, AlignedAllocator::Host<T> > >
	__host__
	void operator=(Matrix::Host::Dense<T, VectorType>& other)
	{
		// Use byte container preventing destruction of matrix
		// mirrored from device. Otherwise it will be destroyed
		// together with newly created data array after resizing.
		std::vector<char> container(sizeof(Matrix::Device::Dense<T>));
		
		Matrix::Device::Dense<T>* matrix =
			reinterpret_cast<Matrix::Device::Dense<T>*>(&container[0]);
		CUDA_ERR_CHECK(cudaMemcpy(matrix, this, sizeof(Matrix::Device::Dense<T>),
			cudaMemcpyDeviceToHost));
		matrix->resize(other.dimy(), other.dimx());

		// It is assumed safe to copy padded data from host to device matrix,
		// as they use the same memory allocation policy.
		size_t size = (ptrdiff_t)&other(other.dimy() - 1, other.dimx() - 1) -
			(ptrdiff_t)other.getData() + sizeof(T);
		CUDA_ERR_CHECK(cudaMemcpy(matrix->getData(), other.getData(), size,
			cudaMemcpyHostToDevice));
		
		CUDA_ERR_CHECK(cudaMemcpy(this, matrix, sizeof(Matrix::Device::Dense<T>),
			cudaMemcpyHostToDevice));
	}
};

namespace Sparse {

// Host memory sparse matrix in CSR format
template<typename TValue, typename TIndex>
class CSR
{
	TValue *a;
	TIndex *ia, *ja;
	bool dataOwner; // whether the instance owns its data pointer or not
	int dimY, dimX, nnZ;

public :
	CSR() :
		a(NULL), ia(NULL), ja(NULL),
		dimY(0), dimX(0), nnZ(0)
	{ }

	CSR(int dimY_, int dimX_, int nnz_) :
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
	~CSR()
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

} // namespace Sparse

} // namespace Device

#else

namespace Device
{
	template<typename T>
	class Dense;
	
	namespace Sparse
	{
		template<typename T>
		class CSR;
	}
}

#endif // __CUDACC__

} // namespace Matrix

} // namespace cuda

#endif // MATRIX_H

