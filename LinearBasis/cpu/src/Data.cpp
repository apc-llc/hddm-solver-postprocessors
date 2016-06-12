#include "check.h"
#include "Data.h"
#include "interpolator.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <mpi.h>

using namespace std;

int Data::getNno() const { return nno; }

void Data::load(const char* filename, int istate)
{
	MPI_Process* process;
	MPI_ERR_CHECK(MPI_Process_get(&process));
	const Parameters& params = Interpolator::getInstance()->getParameters();

	if (loadedStates[istate])
	{
		cerr << "State " << istate << " data is already loaded" << endl;
		process->abort();
	}

	ifstream infile;
	if (params.binaryio)
		infile.open(filename, ios::in | ios::binary);
	else
		infile.open(filename, ios::in);

	if (!infile.is_open())
	{
		cerr << "Error opening file: " << filename << endl;
		process->abort();
	}

	infile >> dim;
	if (dim != params.nagents)
	{
		cerr << "File \"" << filename << "\" # of dimensions (" << dim << 
			") mismatches config (" << params.nagents << ")" << endl;
		process->abort();
	}
	infile >> nno; 
	infile >> TotalDof;
	infile >> Level;

	// Pad all indexes to 4-element boundary.
	vdim = dim / AVX_VECTOR_SIZE;
	if (dim % AVX_VECTOR_SIZE) vdim++;
	int nsd = 2 * vdim * AVX_VECTOR_SIZE;
	index[istate].resize(nno, nsd);
	index[istate].fill(0);
	surplus[istate].resize(nno, TotalDof);
	surplus[istate].fill(0.0);

	// For better caching we use transposed surplus.
	surplus_t[istate].resize(TotalDof, nno);
	surplus_t[istate].fill(0.0);
	int j = 0;
	while (infile)
	{
		if (j == nno) break;

		for (int i = 0; i < dim; )
		{
			for (int v = 0; (v < AVX_VECTOR_SIZE) && (i < dim); v++, i++)
			{
				int value; infile >> value;
				value = 2 << (value - 2);
				index[istate](j, i) = value;
			}
		}
		for (int i = 0; i < dim; )
		{
			for (int v = 0; (v < AVX_VECTOR_SIZE) && (i < dim); v++, i++)
			{
				int value; infile >> value;
				value--;
				// Percompute "j" to merge two cases into one:
				// (((i) == 0) ? (1) : (1 - fabs((x) * (i) - (j)))).
				if (!index[istate](j, i)) value = 0;
				index[istate](j, i + vdim * AVX_VECTOR_SIZE) = value;
			}
		}
		for (int i = 0; i < TotalDof; i++)
		{
			double value; infile >> value;
			surplus[istate](j, i) = value;
			surplus_t[istate](i, j) = value;
		}
		j++;
	}
	infile.close();
	
	loadedStates[istate] = true;
	
	indexes[istate] = index[istate].getData();
	surpluses[istate] = surplus[istate].getData();
}

void Data::clear()
{
	fill(loadedStates.begin(), loadedStates.end(), false);
}

Data::Data(int nstates_) : nstates(nstates_)
{
	index.resize(nstates);
	surplus.resize(nstates);
	surplus_t.resize(nstates);
	loadedStates.resize(nstates);
	fill(loadedStates.begin(), loadedStates.end(), false);
	indexes.resize(nstates);
	surpluses.resize(nstates);
}

const int* const* Data::getIndexes() const { return &indexes[0]; }
	
const real* const* Data::getSurpluses() const { return &surpluses[0]; }

static unique_ptr<Data> data = NULL;

extern "C" Data* getData(int nstates)
{
	if (!data.get())
		data.reset(new Data(nstates));
	
	return data.get();
}

