#ifndef MAPPED_ALLOCATOR_H
#define MAPPED_ALLOCATOR_H

#include <iostream>
#include <mutex>

namespace cuda {

// Custom host memory allocator, using code by Sergei Danielian
// https://github.com/gahcep/Allocators
template <class T>
class MappedAllocator
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
	struct rebind { using other = MappedAllocator<U>; };

	MappedAllocator() { }
	MappedAllocator(const MappedAllocator&) { }
	template <class U>
	MappedAllocator(const MappedAllocator<U>&) { }

	// Allocators are not required to be assignable
	MappedAllocator& operator=(const MappedAllocator& other) = delete;
	~MappedAllocator() { }

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
	bool operator!=(const MappedAllocator& other) { return !(*this == other); }
	bool operator==(const MappedAllocator&) { return true; }

	// allocation
	pointer allocate(size_type n, std::allocator<void>::const_pointer = 0) const
	{
		// TODO
#if 0
		// TODO SIMD vector byte size should be used here instead.
		size_t alignment = AVX_VECTOR_SIZE * sizeof(T);
		
		if (sizeof(T) > sizeof(double))
			alignment = AVX_VECTOR_SIZE * sizeof(double);
#endif
		// Align and pad to vector size.
		void* ptr;
		size_t size = n * sizeof(T);
		if (size % (AVX_VECTOR_SIZE * sizeof(T)))
			size += AVX_VECTOR_SIZE * sizeof(T) - size % (AVX_VECTOR_SIZE * sizeof(T));		
		CUDA_ERR_CHECK(cudaHostAlloc(&ptr, size, cudaHostAllocMapped));

		return static_cast<pointer>(ptr);
	}

	void deallocate(pointer ptr, size_type n)
	{
		CUDA_ERR_CHECK(cudaFreeHost(ptr));
	}
};

}

#endif // MAPPED_ALLOCATOR_H

