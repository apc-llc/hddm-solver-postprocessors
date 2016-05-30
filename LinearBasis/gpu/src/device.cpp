// TODO: interpolator-specific device management

#ifdef HAVE_CUDA		
	// Check if my host has a GPU. If it does, assign GPU
	// to the first node in host communicator.
	// TODO: If host has more than one GPU, then assign GPUs to
	// MPI ranks with cudaSetDevice in a round robin.
	// TODO: On NUMA systems we should assign GPU to the
	// topology-nearest MPI node in host communicator.
	if (rank == hostranks[0])
	{
		CUDA_ERR_CHECK(cudaGetDeviceCount(&ngpus));
		if (ngpus) hasGPU = true;
	}

	// TODO: Turn hasGPU into integer to denote how many GPUs are
	// available to a rank (could be more than one).

	// Tell everyone the total number of available GPUs.
	MPI_ERR_CHECK(MPI_Allreduce(MPI_IN_PLACE, &ngpus, 1, MPI_INT,
		MPI_SUM, MPI_COMM_WORLD));
	
	if (master)
		cout << ngpus << " GPU(s) participating" << endl;
	cout << endl;
#endif

