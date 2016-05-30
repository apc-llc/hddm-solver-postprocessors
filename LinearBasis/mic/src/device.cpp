// TODO: interpolator-specific device management

#ifdef HAVE_MIC
	// Check if my host has Xeon Phi. If it does, assign Xeon Phi
	// to the first node in host communicator.
	// TODO: If host has more that one Xeon Phi, then assign Xeon Phis
	// to MPI ranks in a round robin.
	// TODO: On NUMA systems we should assign Xeon Phi to the
	// topology-nearest MPI node in host communicator.
	if (rank == hostranks[0])
	{
	}
	
	// TODO: Turn hasMIC into integer to denote how many Xeon Phis are
	// available to a rank (could be more than one).
	
	// Tell everyone the total number of available Xeon Phis.
	MPI_ERR_CHECK(MPI_Allreduce(MPI_IN_PLACE, &nmics, 1, MPI_INT,
		MPI_SUM, MPI_COMM_WORLD));

	if (master)
		cout << nmics << " MIC(s) participating" << endl;
	cout << endl;
#endif

