#include "check.h"
#include "Data.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <mpi.h>

using namespace std;

class Parameters
{
public :
	bool get_binaryio() const;
	int get_nagents() const;
};

class Simulation
{
public :
	const Parameters& getParameters() const;
};

class MPI_Process
{
public :
	Simulation* getSimulation();
	void abort();
};

extern "C" int MPI_Process_get(MPI_Process**);

int Data::getNno() const { return nno; }

void Data::load(const char* filename)
{
	MPI_Process* process;
	MPI_ERR_CHECK(MPI_Process_get(&process));
	const Parameters* params = const_cast<Parameters*>(&process->getSimulation()->getParameters());

	ifstream infile;
	if (params->get_binaryio())
		infile.open(filename, ios::in | ios::binary);
	else
		infile.open(filename, ios::in);

	if (!infile.is_open())
	{
		cerr << "Error opening file: " << filename << endl;
		exit(1);
	}

	infile >> dim;
	if (dim != params->get_nagents())
	{
		cerr << "File \"" << filename << "\" # of dimensions (" << dim << 
			") mismatches config (" << params->get_nagents() << ")" << endl;
		process->abort();
	}
	infile >> nno; 
	infile >> TotalDof;
	infile >> Level;

	// Pad all indexes to 4-element boundary.
	vdim = dim / AVX_VECTOR_SIZE;
	if (dim % AVX_VECTOR_SIZE) vdim++;
	int nsd = 2 * vdim * AVX_VECTOR_SIZE;
	index.resize(nno, nsd);
	index.fill(0);
	surplus.resize(nno, TotalDof);
	surplus.fill(0.0);

	// For better caching we use transposed surplus.
	surplus_t.resize(TotalDof, nno);
	surplus_t.fill(0.0);
	int i = 0;
	while (infile)
	{
		if (i == nno) break;

		for (int j = 0;j < dim; )
		{
			int value_simd[AVX_VECTOR_SIZE];
			for (int v = 0; (v < AVX_VECTOR_SIZE) && (j < dim); v++, j++)
			{
				int value; infile >> value;
				value = 2 << (value - 2);
				index(i, j) = value;
				value_simd[v] = value;
			}
		}
		for (int j = 0; j < dim; )
		{
			int value_simd[AVX_VECTOR_SIZE];
			for (int v = 0; (v < AVX_VECTOR_SIZE) && (j < dim); v++, j++)
			{
				int value; infile >> value;
				value--;
				// Percompute "j" to merge two cases into one:
				// (((i) == 0) ? (1) : (1 - fabs((x) * (i) - (j)))).
				if (!index(i, j)) value = 0;
				index(i, j + vdim * AVX_VECTOR_SIZE) = value;
				value_simd[v] = value;
			}
		}
		for (int j = 0; j < TotalDof; j++)
		{
			double value; infile >> value;
			surplus(i, j) = value;
			surplus_t(j, i) = value;
		}
		i++;
	}
	infile.close();
}

Data::Data() { }

static unique_ptr<Data> data = NULL;

extern "C" Data* getData()
{
	if (!data.get())
		data.reset(new Data());
	
	return data.get();
}

