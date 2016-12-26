#ifndef LOCK_H
#define LOCK_H

#if defined(__CUDACC__)

#include "AlignedAllocator.h"

#define STATE_UNLOCKED 2
#define STATE_LOCKED 10

extern "C" __device__ __cudart_builtin__
int __nvvm_atom_cas_gen_i(volatile int *, int, int);

extern "C" __device__ __cudart_builtin__
int __nvvm_atom_xchg_gen_i(volatile int *, int);

namespace cuda {

namespace Lock {

class Device
{
	volatile int* mutex;

public :
	
	__host__ __device__
	Device()
	{
		mutex = AlignedAllocator::Device<int>().allocate(1);
#if defined(__CUDA_ARCH__)
		*mutex = STATE_UNLOCKED;
#else
		int state = STATE_UNLOCKED;
		CUDA_ERR_CHECK(cudaMemcpy((int*)mutex, &state, sizeof(int), cudaMemcpyHostToDevice));
#endif
	}

	__host__ __device__
	~Device()
	{
		AlignedAllocator::Device<int>().deallocate((int*)mutex, 1);
	}

	__device__
	inline __attribute__((always_inline)) void lock()
	{
		#pragma unroll 1
		for ( ; ; )
		{
			if (__nvvm_atom_cas_gen_i(mutex, STATE_UNLOCKED, STATE_LOCKED) == STATE_LOCKED)
				break;
		}
	}

	__device__
	inline __attribute__((always_inline)) void unlock()
	{
		__nvvm_atom_xchg_gen_i(mutex, STATE_UNLOCKED);
	}
};

} // namespace Lock

namespace ScopedLock {

class Device
{
	Lock::Device& lock;

public :

	__device__
	Device(Lock::Device& lock_) : lock(lock_)
	{
		lock.lock();
	}

	__device__
	~Device()
	{
		lock.unlock();
	}
};

} // namespace ScopedLock

} // namespace cuda

#endif // __CUDACC__

#endif // LOCK_H

