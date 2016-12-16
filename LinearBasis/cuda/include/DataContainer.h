#ifndef DATA_CONTAINER_H
#define DATA_CONTAINER_H

#include "AlignedAllocator.h"
#include "ConstructionKernels.h"
#include "MappedAllocator.h"

namespace cuda {

#if defined(__CUDACC__)

class DataModificationIndicator
{
	bool* lastModifiedOnHost;
	
public :

	__host__ __device__
	inline __attribute__((always_inline)) bool isOnHost() const
	{
#if defined(__CUDA_ARCH__)
		return *lastModifiedOnHost;
#else
		bool value;
		CUDA_ERR_CHECK(cudaMemcpy(&value, lastModifiedOnHost, sizeof(bool),
			cudaMemcpyDeviceToHost));
		return value;
#endif // __CUDA_ARCH__
	}
	
	__host__ __device__
	inline __attribute__((always_inline)) bool isOnDevice() const
	{
		return !isOnHost();
	}
	
	__host__ __device__
	inline __attribute__((always_inline)) void setOnHost()
	{
#if defined(__CUDA_ARCH__)
		*lastModifiedOnHost = true;
#else
		bool value = true;
		CUDA_ERR_CHECK(cudaMemcpy(lastModifiedOnHost, &value, sizeof(bool),
			cudaMemcpyHostToDevice));
#endif // __CUDA_ARCH__
	}
	
	__host__ __device__
	inline __attribute__((always_inline)) void setOnDevice()
	{
#if defined(__CUDA_ARCH__)
		*lastModifiedOnHost = false;
#else
		bool value = false;
		CUDA_ERR_CHECK(cudaMemcpy(lastModifiedOnHost, &value, sizeof(bool),
			cudaMemcpyHostToDevice));
#endif // __CUDA_ARCH__
	}

	__host__ __device__
	DataModificationIndicator()
	{
		lastModifiedOnHost = AlignedAllocator::Device<bool>().allocate(1);
#if defined(__CUDA_ARCH__)
		setOnDevice();
#else
		setOnHost();
#endif // __CUDA_ARCH__
	}

	__host__ __device__
	~DataModificationIndicator()
	{
		AlignedAllocator::Device<bool>().deallocate(lastModifiedOnHost, 1);
	}
};

template<typename T>
class DataContainer
{
	T* data;
	
#if !defined(__CUDA_ARCH__)
	std::vector<T, MappedAllocator<T> > dataHost;
#else
	char dataHost[sizeof(std::vector<T, MappedAllocator<T> >)]; // padding
#endif // __CUDA_ARCH__

	// Device-accessible pointer to the host-mapped data.
	T** dataHostPtr;

	int length;

	// Indicate where the data has been last time modified.
	// If the data was last time modified by client different
	// from currently accessing it, then the data has to be refreshed.
	DataModificationIndicator lastModified;

	struct Kernels
	{
		__host__
		Kernels()
		{
			bool neverCalled = true;
			if (!neverCalled)
			{
				deviceSizeOf<T><<<1, 1>>>(NULL);
				constructDeviceData<T><<<1, 1>>>(1, NULL);
				destroyDeviceData<T><<<1, 1>>>(1, NULL);
			}
		}
	} kernels;

	__device__
	inline __attribute__((always_inline)) void ld128(char* dst, char* src) const
	{
#if defined(__CUDA_ARCH__)
#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)
#define __LD128_PTR   "l"
#else
#define __LD128_PTR   "r"
#endif
		float4& ret = *(float4*)dst;
		asm volatile("ld.global.v4.f32 {%0,%1,%2,%3}, [%4];"  :
			"=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w) : __LD128_PTR(src));
#endif // __CUDA_ARCH__
	}

protected :

	__host__ __device__
	inline __attribute__((always_inline)) void initialize(int length)
	{
#if defined(__CUDA_ARCH__)
		data = AlignedAllocator::Device<T>().allocate(length);
		// TODO map onto kernel, if device parallelism is available
		new(data) T[length];
#else
		// Determine the size of target type.
		size_t* dSize = AlignedAllocator::Device<size_t>().allocate(1);
		deviceSizeOf<T><<<1, 1>>>(dSize);
		CUDA_ERR_CHECK(cudaGetLastError());
		CUDA_ERR_CHECK(cudaDeviceSynchronize());
		size_t size;
		CUDA_ERR_CHECK(cudaMemcpy(&size, dSize, sizeof(size_t), cudaMemcpyDeviceToHost));
		AlignedAllocator::Device<size_t>().deallocate(dSize, 1);
		
		// Make sure the size of type is the same on host and on device.
		if (size != sizeof(T))
		{
			std::cerr << "Unexpected unequal sizes of type T in DataContainer<T> on host (" << sizeof(T) <<
				") and device (" << size << ")" << std::endl;
			MPI_Process* process;
			MPI_ERR_CHECK(MPI_Process_get(&process));
			process->abort();
		}

		// Allocate array.
		data = AlignedAllocator::Device<T>().allocate(length);

		// Construct individual array elements from within the device kernel code,
		// using placement new operator.
		unsigned int szblock = CONSTRUCTION_KERNELS_SZBLOCK;
		unsigned int nblocks = length / szblock;
		if (length % szblock) nblocks++;
		constructDeviceData<T><<<nblocks, szblock>>>(length, data);
		CUDA_ERR_CHECK(cudaGetLastError());
		CUDA_ERR_CHECK(cudaDeviceSynchronize());
#endif // __CUDA_ARCH__
	}
	
	__host__ __device__
	inline __attribute__((always_inline)) void deinitialize()
	{
		if (!length) return;
#if defined(__CUDA_ARCH__)
		if (data)
		{
			// TODO map onto kernel, if device parallelism is available
			for (int i = 0; i < length; i++)
				data[i].~T();
			AlignedAllocator::Device<T>().deallocate(data, length);
		}
#else
		// Destroy individual array elements from within the device kernel code.
		unsigned int szblock = CONSTRUCTION_KERNELS_SZBLOCK;
		unsigned int nblocks = length / szblock;
		if (length % szblock) nblocks++;
		destroyDeviceData<T><<<nblocks, szblock>>>(length, data);
		CUDA_ERR_CHECK(cudaGetLastError());
		CUDA_ERR_CHECK(cudaDeviceSynchronize());

		AlignedAllocator::Device<T>().deallocate(data, length);
#endif // __CUDA_ARCH__
	}

	// Disable warning that Kernels class has host-only constructor.
	#pragma hd_warning_disable \
	#pragma nv_exec_check_disable
	__host__ __device__
	DataContainer() : data(NULL), length(0)
	{
		dataHostPtr = AlignedAllocator::Device<T*>().allocate(1);
	}

	// Disable warning that Kernels class has host-only constructor.
	#pragma hd_warning_disable \
	#pragma nv_exec_check_disable
	__host__ __device__
	DataContainer(int length_) : data(NULL), length(length_)
	{
		initialize(length);

		dataHostPtr = AlignedAllocator::Device<T*>().allocate(1);
	}

	__host__ __device__
	~DataContainer()
	{
		deinitialize();

		AlignedAllocator::Device<T*>().deallocate(dataHostPtr, 1);
	}

	__host__ __device__
	inline __attribute__((always_inline)) const T& accessDataAt(int offset)
	{
#if defined(__CUDA_ARCH__)
		if (lastModified.isOnHost())
		{
			#pragma unroll 1
			for (int i = 0, e = length * sizeof(T); i < e; i += 4)
				ld128(&((char*)data)[i], &((char*)(*dataHostPtr))[i]);
		}
		return data[offset];
#else
		if (dataHost.size() != length)
		{
			dataHost.resize(length);
			T* hostPtr = NULL;
			CUDA_ERR_CHECK(cudaHostGetDevicePointer(&hostPtr, &dataHost[0], 0));
			CUDA_ERR_CHECK(cudaMemcpy(dataHostPtr, &hostPtr, sizeof(T*),
				cudaMemcpyHostToDevice));
		}
		if (lastModified.isOnDevice())
		{
			CUDA_ERR_CHECK(cudaMemcpy(&dataHost[0], getData(), length * sizeof(T),
				cudaMemcpyDeviceToHost));
		}
		return dataHost[offset];
#endif
	}
	
	__host__ __device__
	inline __attribute__((always_inline)) T& modifyDataAt(int offset)
	{
#if defined(__CUDA_ARCH__)
		if (lastModified.isOnHost())
		{
			#pragma unroll 1
			for (int i = 0, e = length * sizeof(T); i < e; i += 4)
				ld128(&((char*)data)[i], &((char*)(*dataHostPtr))[i]);

			lastModified.setOnDevice();
		}
		return data[offset];
#else
		if (dataHost.size() != length)
		{
			dataHost.resize(length);
			T* hostPtr = NULL;
			CUDA_ERR_CHECK(cudaHostGetDevicePointer(&hostPtr, &dataHost[0], 0));
			CUDA_ERR_CHECK(cudaMemcpy(dataHostPtr, &hostPtr, sizeof(T*),
				cudaMemcpyHostToDevice));
		}
		if (lastModified.isOnDevice())
		{
			dataHost.resize(length);
			CUDA_ERR_CHECK(cudaMemcpy(&dataHost[0], getData(), length * sizeof(T),
				cudaMemcpyDeviceToHost));

			lastModified.setOnHost();
		}
		return data[offset];
#endif
	}

	__host__ __device__
	inline __attribute__((always_inline)) T* getData() const { return &data[0]; }
};

#endif // __CUDACC__

} // namespace cuda

#endif // DATA_CONTAINER_H

