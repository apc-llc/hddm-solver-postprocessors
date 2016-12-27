#ifndef ALIGNED_ALLOCATOR_H
#define ALIGNED_ALLOCATOR_H

#include <iostream>
#include <mutex>

namespace cuda {

namespace AlignedAllocator {

// Custom host memory allocator, using code by Sergei Danielian
// https://github.com/gahcep/Allocators
template <class T>
class Host
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
	struct rebind { using other = Host<U>; };

	Host() { }
	Host(const Host&) { }
	template <class U>
	Host(const Host<U>&) { }

	// Allocators are not required to be assignable
	Host& operator=(const Host& other) = delete;
	~Host() { }

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
	bool operator!=(const Host& other) { return !(*this == other); }
	bool operator==(const Host&) { return true; }

	// allocation
	pointer allocate(size_type n, std::allocator<void>::const_pointer = 0) const
	{
		// TODO SIMD vector byte size should be used here instead.
		size_t alignment = AVX_VECTOR_SIZE * sizeof(T);
		
		if (sizeof(T) > sizeof(double))
			alignment = AVX_VECTOR_SIZE * sizeof(double);

		// Align and pad to vector size.
		void* ptr;
		size_t size = n * sizeof(T);
		if (size % (AVX_VECTOR_SIZE * sizeof(T)))
			size += AVX_VECTOR_SIZE * sizeof(T) - size % (AVX_VECTOR_SIZE * sizeof(T));		
		int err = posix_memalign(&ptr, alignment, size);
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

		return static_cast<pointer>(ptr);
	}

	void deallocate(pointer ptr, size_type n)
	{
		free(ptr);
	}
};

#if defined(__CUDACC__)

// Defines the dynamic memory pool configuration.
struct kernelgen_memory_t
{
	// The pointer to the memory pool (a single
	// large array, allocated with cudaMallocManaged).
	char *pool;

	// The size of the memory pool.
	size_t szpool;

	// The size of the used memory.
	size_t szused;

	// The number of MCB records in pool.
	size_t count;
};

extern bool deviceMemoryHeapInitialized;
extern __device__ kernelgen_memory_t deviceMemoryHeap;
kernelgen_memory_t& deviceMemoryHeapHost();

__global__ void setupDeviceMemoryHeap(kernelgen_memory_t heap);

namespace
{
	static std::mutex mtx;
}

// Custom host memory allocator, using code by Sergei Danielian
// https://github.com/gahcep/Allocators
template <class T>
class Device
{
	struct kernelgen_memory_chunk_t
	{
		int is_available;
		int size;	

		// Align structure to 4096, or there will be
		// problems with 128-bit loads/stores in
		// optimized Fermi ISA (nvopencc issue?).
		char padding[4096 - 8];
	};

	__host__ __device__
	static void preallocateHeap()
	{
#if !defined(__CUDA_ARCH__)
		// Critical section
		std::unique_lock<std::mutex> lck(mtx, std::defer_lock);
		lck.lock();
  	
		// Device-side allocator pre-allocates all device memory
		if (!deviceMemoryHeapInitialized)
		{
			kernelgen_memory_t& heapHost = deviceMemoryHeapHost();

			if (!heapHost.pool)
			{
				cudaDeviceProp props;
				CUDA_ERR_CHECK(cudaGetDeviceProperties(&props, 0));
			
				size_t size = props.totalGlobalMem * 0.75;
				cudaError_t status = cudaErrorNotSupported;
				for ( ; size >= 1; size *= 0.9)
				{
					status = cudaMalloc((void**)&heapHost.pool, size);
					if (status == cudaSuccess)
						break;
				}
				if (status != cudaSuccess)
				{
					std::cerr << "Device memory heap creation has failed" << std::endl;
					CUDA_ERR_CHECK(status);
				}
			
				std::cout << "Device memory heap size: " << size << std::endl;

				CUDA_ERR_CHECK(cudaMemset(heapHost.pool, 0, size));

				heapHost.szpool = size;
				heapHost.szused = 0;
				heapHost.count = 0;
			}

			setupDeviceMemoryHeap<<<1, 1>>>(heapHost);
			CUDA_ERR_CHECK(cudaGetLastError());
			CUDA_ERR_CHECK(cudaDeviceSynchronize());
			
			deviceMemoryHeapInitialized = true;
		}
		
		lck.unlock();
#endif // __CUDA_ARCH__
	}

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
	struct rebind { using other = Device<U>; };

	__host__ __device__
	Device()
	{
		preallocateHeap();
	}

	Device(const Device&) { }
	template <class U>
	
	Device(const Device<U>&) { }

	// Allocators are not required to be assignable
	Device& operator=(const Device& other) = delete;
	
	__host__ __device__
	~Device() { }

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
	bool operator!=(const Device& other) { return !(*this == other); }
	bool operator==(const Device&) { return true; }

#define KERNELGEN_MEM_FREE 0
#define KERNELGEN_MEM_IN_USE 1

	// allocation
	__host__ __device__
	pointer allocate(size_type n, std::allocator<void>::const_pointer = 0) const
	{
		// TODO SIMD vector byte size should be used here instead.
		size_t alignment = AVX_VECTOR_SIZE * sizeof(T);
		
		if (sizeof(T) > sizeof(double))
			alignment = AVX_VECTOR_SIZE * sizeof(double);

		// Align and pad to vector size.
		size_t size = n * sizeof(T);
		if (size % (AVX_VECTOR_SIZE * sizeof(T)))
			size += AVX_VECTOR_SIZE * sizeof(T) - size % (AVX_VECTOR_SIZE * sizeof(T));		

		// TODO Handle alignment properly.
		size += alignment;
#if defined(__CUDA_ARCH__)
		kernelgen_memory_t *km = &deviceMemoryHeap;
#else
		kernelgen_memory_t *km = &deviceMemoryHeapHost();
#endif
		// If there is less free space in pool, than requested,
		// then just return NULL.
		if (size + sizeof(kernelgen_memory_chunk_t) >
			km->szpool - (km->szused + km->count * sizeof(kernelgen_memory_chunk_t)))
		{
#if !defined(__CUDA_ARCH__)
			std::cerr << "Out of device memory" << std::endl;
#endif
			return NULL;
		}

		// Find a free memory chunk.
		size_t i = 0;
		for ( ; i + size + sizeof(kernelgen_memory_chunk_t) < km->szpool; )
		{
			kernelgen_memory_chunk_t* p_mcb = (kernelgen_memory_chunk_t *)(km->pool + i);
#if defined(__CUDA_ARCH__)
			kernelgen_memory_chunk_t& mcb = *p_mcb;
#else
			kernelgen_memory_chunk_t mcb;
			CUDA_ERR_CHECK(cudaMemcpy(&mcb, p_mcb, sizeof(kernelgen_memory_chunk_t),
				cudaMemcpyDeviceToHost));
#endif
			if (mcb.is_available == KERNELGEN_MEM_FREE)
			{
				// If this is a new unused chunk in the tail of pool.
				if (mcb.size == 0)
				{
					mcb.is_available = KERNELGEN_MEM_IN_USE;
					mcb.size = size + sizeof(kernelgen_memory_chunk_t);
					km->count++;
					km->szused += size;

#if !defined(__CUDA_ARCH__)
					CUDA_ERR_CHECK(cudaMemcpy(p_mcb, &mcb, sizeof(kernelgen_memory_chunk_t),
						cudaMemcpyHostToDevice));
#endif

					void* ptr = (void*)((char*)p_mcb + sizeof(kernelgen_memory_chunk_t));

					return static_cast<pointer>(ptr);
				}

				// If size of the available chunk is equal to greater
				// than required size, use that chunk.
				if (mcb.size >= (size + sizeof(kernelgen_memory_chunk_t)))
				{
					mcb.is_available = KERNELGEN_MEM_IN_USE;
					size = mcb.size - sizeof(kernelgen_memory_chunk_t);
					km->count++;
					km->szused += size;
					
					// TODO: Mark the rest of the used chunk as a new
					// free chunk?

#if !defined(__CUDA_ARCH__)
					CUDA_ERR_CHECK(cudaMemcpy(p_mcb, &mcb, sizeof(kernelgen_memory_chunk_t),
						cudaMemcpyHostToDevice));
#endif

					void* ptr = (void*)((char*)p_mcb + sizeof(kernelgen_memory_chunk_t));
					
					return static_cast<pointer>(ptr);
				}
			}
			
			i += mcb.size;
		}

#if !defined(__CUDA_ARCH__)
		std::cerr << "Out of device memory" << std::endl;
#endif

		return NULL;
	}

	__host__ __device__
	void deallocate(pointer ptr, size_type n)
	{
#if defined(__CUDA_ARCH__)
		kernelgen_memory_t *km = &deviceMemoryHeap;
#else
		kernelgen_memory_t *km = &deviceMemoryHeapHost();
#endif
		// Mark in MCB that this chunk is free.
		kernelgen_memory_chunk_t* p_mcb = (kernelgen_memory_chunk_t *)ptr - 1;
#if defined(__CUDA_ARCH__)
		kernelgen_memory_chunk_t& mcb = *p_mcb;
#else
		kernelgen_memory_chunk_t mcb;
		CUDA_ERR_CHECK(cudaMemcpy(&mcb, p_mcb, sizeof(kernelgen_memory_chunk_t),
			cudaMemcpyDeviceToHost));
#endif
		if (mcb.is_available != KERNELGEN_MEM_FREE)
		{
			mcb.is_available = KERNELGEN_MEM_FREE;
			km->count--;
			km->szused -= (mcb.size - sizeof(kernelgen_memory_chunk_t));
		}

		// TODO: if the last chunk is freed, then we need
		// to flush its contents to zero.
	}
};

#endif // __CUDA_ARCH__

} // namespace AlignedAllocator

} // namespace cuda

#endif // ALIGNED_ALLOCATOR_H

