#include "check.h"
#include "Devices.h"
#include "interpolator.h"
#include "process.h"

#include <iostream>
#include <memory>
#include <mutex>

using namespace NAMESPACE;
using namespace std;

Devices::Devices()
{
	MPI_Process* process;
	MPI_ERR_CHECK(MPI_Process_get(&process));

	// Check for available CUDA GPU(s)
	int ngpus = 0;
	CUDA_ERR_CHECK(cudaGetLastError());
	cudaError_t cudaError = cudaGetDeviceCount(&ngpus);
	if (cudaError == cudaErrorNoDevice)
	{
		if (process->isMaster())
			cout << cudaGetErrorString(cudaError) << endl;
		ngpus = 0;
	}
	else
	{
		CUDA_ERR_CHECK(cudaError);
		CUDA_ERR_CHECK(cudaDeviceReset());
	}
	devices.resize(ngpus);
	
	for (int igpu = 0; igpu < ngpus; igpu++)
	{
		struct cudaDeviceProp props;
		CUDA_ERR_CHECK(cudaGetDeviceProperties(&props, igpu));
		int id[2];
		id[0] = props.pciBusID;
		id[1] = props.pciDeviceID;
		devices[igpu].id = *(long long*)id;
		devices[igpu].warpSize = props.warpSize;
		devices[igpu].cc = props.major * 10 + props.minor;
		devices[igpu].sm = SM(props.multiProcessorCount, props.sharedMemPerMultiprocessor);
	}
}

int Devices::getCount()
{
	return devices.size();
}

const Device* Devices::getDevice(int index) const
{
	return &devices[index];
}

Device* Devices::tryAcquire()
{
	Device* device = NULL;

	for (int i = 0; i < devices.size(); i++)
	{
		{
			static std::mutex mutex;
			std::lock_guard<std::mutex> lock(mutex);

			if (devices[i].available && !device)
			{
				devices[i].available = 0;
				device = &devices[i];
			}
		}
		
		if (device) break;
	}
	
	return device;
}

void Devices::release(Device* device)
{
	if (!device) return;

	{
		static std::mutex mutex;
		std::lock_guard<std::mutex> lock(mutex);

		device->available++;
	}
}

namespace NAMESPACE
{
	unique_ptr<Devices> devices;

	NAMESPACE::Device* tryAcquireDevice()
	{
		if (!devices)
			devices.reset(new Devices());

		return devices->tryAcquire();
	}

	void releaseDevice(NAMESPACE::Device* device)
	{
		if (!devices)
			devices.reset(new Devices());

		devices->release(device);
	}
}

extern "C" Device* tryAcquireDevice()
{
	return NAMESPACE::tryAcquireDevice();
}

extern "C" void releaseDevice(Device* device)
{
	NAMESPACE::releaseDevice(device);
}

