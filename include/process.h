#ifndef HDDM_MPI_PROCESS
#define HDDM_MPI_PROCESS 

class MPI_Process
{
public :
	bool isMaster() const;
	int getRoot() const;
	int getRank() const;
	int getSize() const;

	void abort();
};

extern "C" int MPI_Process_get(MPI_Process**);

#endif // HDDM_MPI_PROCESS

