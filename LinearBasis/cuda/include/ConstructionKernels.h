#ifndef CONSTRUCTION_KERNELS_H
#define CONSTRUCTION_KERNELS_H

// GPU helper kernels for device-side object construction

#if defined(__CUDACC__)

#define CONSTRUCTION_KERNELS_SZBLOCK 128

// Determine the size of type on device.
template<typename T>
static __global__ void deviceSizeOf(size_t* result)
{
	*result = sizeof(T);
}

template<typename T>
static __global__ void constructDeviceData(int length, T* data)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= length) return;
	new(&data[i]) T();
}

template<typename T>
static __global__ void destroyDeviceData(int length, T* data)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= length) return;
	delete (T*)&data[i];
}

#endif // __CUDACC__

#endif // CONSTRUCTION_KERNELS_H

