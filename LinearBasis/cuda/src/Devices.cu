#include "Devices.h"
#include "interpolator.h"

#include <iostream>

using namespace cuda;
using namespace std;

Devices::Devices()
{
	MPI_Process* process;
	MPI_ERR_CHECK(MPI_Process_get(&process));

	// Check for available CUDA GPU(s)
	int ngpus = 0;
	cudaError_t cudaError = cudaGetDeviceCount(&ngpus);
	if (cudaError == cudaErrorNoDevice)
	{
		if (process->isMaster())
			cout << cudaGetErrorString(cudaError) << endl;
		ngpus = 0;
	}
	else
		CUDA_ERR_CHECK(cudaError);
	devices.resize(ngpus);

	// Find out the total number of GPU(s) available to
	// all participating hosts.
	int ngpus_total = ngpus;
	MPI_ERR_CHECK(MPI_Allreduce(MPI_IN_PLACE, &ngpus_total, 1, MPI_INT,
		MPI_SUM, MPI_COMM_WORLD));

	if (process->isMaster())
		cout << ngpus_total << " GPU(s) available" << endl;
	cout << endl;
	
	struct cudaDeviceProp props;
	for (int igpu = 0; igpu < ngpus; igpu++)
	{
		CUDA_ERR_CHECK(cudaGetDeviceProperties(&props, igpu));
		devices[igpu].warpSize = props.warpSize;
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
		#pragma omp critical
		{
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

	#pragma omp atomic
	device->available++;
}

namespace cuda
{
	Devices devices;
}

extern "C" Device* tryAcquireDevice()
{
	return devices.tryAcquire();
}

extern "C" void releaseDevice(Device* device)
{
	devices.release(device);
}

