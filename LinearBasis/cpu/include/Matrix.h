#ifndef MATRIX_H
#define MATRIX_H

#include "DataContainer.h"

namespace NAMESPACE {

namespace Matrix {

// Dense matrix with all rows aligned
template<typename T, typename VectorType = std::vector<T, AlignedAllocator<T> > >
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

namespace Sparse {

// Sparse matrix in CSR format
template<typename TValue, typename TIndex, template<typename, typename> class TVector = std::vector, template<typename> class TAllocator = AlignedAllocator>
class CSR
{
	TVector<TValue, TAllocator<TValue> > a_;
	TVector<TIndex, TAllocator<TIndex> > ia_, ja_;
	int dimY, dimX, nnZ;
	TValue zero;

public :
	CSR() :
		a_(TVector<TValue, TAllocator<TValue> >()),
		ia_(TVector<TIndex, TAllocator<TIndex> >()),
		ja_(TVector<TIndex, TAllocator<TIndex> >()),
		dimY(0), dimX(0), nnZ(0), zero(TValue())
	{ }

	CSR(int dimY_, int dimX_, int nnz_) :
		a_(TVector<TValue, TAllocator<TValue> >()),
		ia_(TVector<TIndex, TAllocator<TIndex> >()),
		ja_(TVector<TIndex, TAllocator<TIndex> >()),
		dimY(dimY_), dimX(dimX_), nnZ(nnz_), zero(TValue())
	{
		a_.resize(nnZ);
		ia_.resize(dimY + 1);
		ja_.resize(nnZ);
	}

	inline __attribute__((always_inline)) TValue& a(int i)
	{
		assert(i < nnZ);
		return a_[i];
	}

	inline __attribute__((always_inline)) const TValue& a(int i) const
	{
		assert(i < nnZ);
		return a_[i];
	}
	
	inline __attribute__((always_inline)) TIndex& ia(int i)
	{
		assert(i < dimY + 1);
		return ia_[i];
	}

	inline __attribute__((always_inline)) const TIndex& ia(int i) const
	{
		assert(i < dimY + 1);
		return ia_[i];
	}
	
	inline __attribute__((always_inline)) TIndex& ja(int i)
	{
		assert(i < nnZ);
		return ja_[i];
	}

	inline __attribute__((always_inline)) const TIndex& ja(int i) const
	{
		assert(i < nnZ);
		return ja_[i];
	}

	inline __attribute__((always_inline)) const TValue& operator()(int y, int x) const
	{
		assert(x < dimX);
		assert(y < dimY);

		for (int i = ia_[y]; i < ia_[y + 1]; i++)
			if (ja_[i] == x) return a_[i];

		return zero;
	}

	inline __attribute__((always_inline)) int dimy() const { return dimY; }

	inline __attribute__((always_inline)) int dimx() const { return dimX; }

	inline __attribute__((always_inline)) int nnz() const { return nnZ; }
		
	inline __attribute__((always_inline)) void resize(int dimY_, int dimX_, int nnz_)
	{
		dimY = dimY_; dimX = dimX_; nnZ = nnz_;
		a_.resize(nnZ);
		ia_.resize(dimY + 1);
		ja_.resize(nnZ);
	}
	
	inline __attribute__((always_inline)) void fill(TValue value)
	{
		std::fill(a_.begin(), a_.end(), value);
	}
};

// Sparse matrix in consequtive row-wise format (CRW).
// Essentially, CRW format packs elements into lines
// organized from N non-zero element(s) from each row.
// Example:
// 1 0 0 2
// 0 4 3 0
// 1 0 6 0
// 5 7 0 0
// packs into:
//  a = (1, 4, 1, 5, 2, 3, 6, 7),
// ja = (0, 1, 0, 0, 3, 2, 2, 1)
//
// If the number of non-zeros in rows is not equal,
// line size is the maximum # of non-zeros across all rows,
// missing elements are packed as zeros.
template<typename TValue, typename TIndex>
// Although CSR matrix relies on self-managing A/JA data containers,
// it still inherits from DataContainer to check the type size.
class CRW : public DataContainer<TValue>
{
	Vector::Device<TValue> a_;
	Vector::Device<TIndex> ja_;
	int dimY, dimYAligned, dimX, nnzPerRow;
	TValue zero;

	static int dimAlign(int dim)
	{
		int dimAligned = dim;
		if (dim % AVX_VECTOR_SIZE)
			dimAligned = dim + AVX_VECTOR_SIZE - dim % AVX_VECTOR_SIZE;
		return dimAligned;
	}

public :
	CRW() : DataContainer<TValue>(), dimY(0), dimYAligned(0), dimX(0), nnzPerRow(0), zero(TValue()) { }

	CRW(int dimY_, int dimX_, int nnzPerRow_) :
		DataContainer<TValue>(),
		a_(nnzPerRow_ * dimAlign(dimY_)), ja_(nnzPerRow_ * dimAlign(dimY_)),
		dimY(dimY_), dimYAligned(dimAlign(dimY_)), dimX(dimX_), nnzPerRow(nnzPerRow_), zero(TValue()) { }

	~CRW() { }

	inline __attribute__((always_inline)) TValue& a(int i) { return a_(i); }

	inline __attribute__((always_inline)) const TValue& a(int i) const { return a_(i); }

	inline __attribute__((always_inline)) TIndex& ja(int i) { return ja_(i); }

	inline __attribute__((always_inline)) const TIndex& ja(int i) const { return ja_(i); }

	inline __attribute__((always_inline)) const TValue& operator()(int y, int x) const
	{
		assert(x < dimX);
		assert(y < dimY);

		for (int i = y; i < nnzPerRow * dimYAligned; i += dimYAligned)
			if (ja_(i) == x) return a_(i);
		
		return zero;
	}

	inline __attribute__((always_inline)) int dimy() const { return dimY; }

	inline __attribute__((always_inline)) int dimx() const { return dimX; }

	inline __attribute__((always_inline)) int nnzperrow() const { return nnzPerRow; }
		
	inline __attribute__((always_inline)) void resize(int dimY_, int dimX_, int nnzPerRow_)
	{
		dimY = dimY_; dimYAligned = dimAlign(dimY_);
		dimX = dimX_; nnzPerRow = nnzPerRow_;
		a_.resize(nnzPerRow * dimYAligned);
		ja_.resize(nnzPerRow * dimYAligned);
	}

	template<template<typename, typename> class TVector = std::vector, template<typename> class TAllocator = AlignedAllocator>
	void operator=(Matrix::Sparse::CSR<TValue, TIndex, TVector, TAllocator>& other)
	{
		// Calculate the number of non-zeros per row in the dense matrix.
		int maxNnzPerRow = 0;
		for (int j = 0, je = other.dimy(); j < je; j++)
		{
			int nnzPerRow = other.ia(j + 1) - other.ia(j);

			if (nnzPerRow > maxNnzPerRow)
				maxNnzPerRow = nnzPerRow;
		}
		int nnzPerRow = maxNnzPerRow;

		Matrix::Device::Sparse::CRW<TValue, TIndex>* matrix = this;

		// Use byte container preventing destruction of matrix
		// mirrored from device. Otherwise it will be destroyed
		// together with newly created data array after resizing.
		std::vector<char> container(sizeof(Matrix::Device::Sparse::CRW<TValue, TIndex>));

		// Determine, in which memory the current matrix instance
		// resides.
		cudaPointerAttributes attrs;
		CUDA_ERR_CHECK(cudaPointerGetAttributes(&attrs, this));
		cudaMemcpyKind kind = cudaMemcpyHostToHost;
		if (attrs.memoryType == cudaMemoryTypeDevice)
		{		
			matrix = reinterpret_cast<Matrix::Device::Sparse::CRW<TValue, TIndex>*>(&container[0]);
			CUDA_ERR_CHECK(cudaMemcpy(matrix, this,
				sizeof(Matrix::Device::Sparse::CRW<TValue, TIndex>),
				cudaMemcpyDeviceToHost));
			kind = cudaMemcpyHostToDevice;
		}

		int dimY = other.dimy();
		int dimYAligned = dimAlign(dimY);
		int dimX = other.dimx();
		
		matrix->resize(dimY, dimX, nnzPerRow);
		
		std::vector<TValue> a(dimYAligned * nnzPerRow);
		std::vector<TIndex> ja(dimYAligned * nnzPerRow);
		std::vector<int> nnz(dimY);
		for (int j = 0, je = dimY; j < je; j++)
			for (int i = other.ia(j), ie = other.ia(j + 1); i < ie; i++, nnz[j]++)
			{
				a[dimYAligned * nnz[j] + j] = other.a(i);
				ja[dimYAligned * nnz[j] + j] = other.ja(i);
			}	
		
		// It is assumed safe to copy padded data from host to device matrix,
		// as they use the same memory allocation policy.
		{
			size_t size = (ptrdiff_t)&a[a.size() - 1] -
				(ptrdiff_t)&a[0] + sizeof(TValue);
			CUDA_ERR_CHECK(cudaMemcpy(&matrix->a(0), &a[0], size, kind));
		}
		{
			size_t size = (ptrdiff_t)&ja[ja.size() - 1] -
				(ptrdiff_t)&ja[0] + sizeof(TIndex);
			CUDA_ERR_CHECK(cudaMemcpy(&matrix->ja(0), &ja[0], size, kind));
		}
		
		if (attrs.memoryType == cudaMemoryTypeDevice)
		{
			CUDA_ERR_CHECK(cudaMemcpy(this, matrix,
				sizeof(Matrix::Device::Sparse::CRW<TValue, TIndex>), kind));
		}
	}
};

} // namespace Matrix

} // namespace NAMESPACE

#endif // MATRIX_H

