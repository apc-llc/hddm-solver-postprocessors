#include "check.h"
#include "interpolator.h"
#include "Data.h"

#include <cstdlib>
#include <map>
#include <mpi.h>
#include <utility> // pair

using namespace NAMESPACE;
using namespace std;

static bool isCompressed(const char* filename)
{
	MPI_Process* process;
	MPI_ERR_CHECK(MPI_Process_get(&process));

	FILE* infile = fopen(filename, "rb");
	if (!infile)
	{
		process->cerr("Error opening file: %s\n", filename);
		process->abort();
	}  

	bool compressed = false;
	
	char format_marker[] = "          ";

	size_t nbytes = fread(reinterpret_cast<void*>(format_marker), sizeof(char), sizeof(format_marker) - 1, infile);
	if (nbytes != sizeof(format_marker) - 1)
	{
		process->cerr("Error reading file: %s\n", filename);
		process->abort();
	}	
	
	fclose(infile);
	
	if ((string)format_marker == "compressed")
		compressed = true;
	
	return compressed;
}

struct IndexPair
{
	unsigned short i, j;
};

template<typename T>
static void read_index(FILE* infile, int nno, int dim, int vdim, Matrix<int>& index_)
{
	MPI_Process* process;
	MPI_ERR_CHECK(MPI_Process_get(&process));

	size_t nbytes = 0;

	char index_marker[] = "index";
	nbytes = fread(index_marker, sizeof(char), sizeof(index_marker) - 1, infile);
	if (nbytes != sizeof(index_marker) - 1)
	{
		if (process->isMaster())
			process->cout("Cannot read index data\n");
		process->abort();
	}

	unsigned int index_nonzeros = 0;
	nbytes = fread(reinterpret_cast<char*>(&index_nonzeros), sizeof(unsigned int), 1, infile);

	vector<int> A;
	A.resize(index_nonzeros);
	nbytes = fread(reinterpret_cast<char*>(&A[0]), sizeof(IndexPair), A.size(), infile);

	vector<int> IA;
	IA.resize(nno + 1);
	{
		vector<T> IAT;
		IAT.resize(nno + 1);
		nbytes = fread(reinterpret_cast<char*>(&IAT[0]), sizeof(T), IAT.size(), infile);
		IA[0] = IAT[0];
		for (int i = 1, e = IA.size(); i != e; i++)
			IA[i] = IAT[i] + IA[i - 1];
	}

	if (IA[0] != 0)
	{
		process->cerr("IA[0] must be 0\n");
		process->abort();
	}

	for (int i = 1, e = IA.size(); i != e; i++)
		if (IA[i] < IA[i - 1])
		{
			process->cerr("IA[i] must be not less than IA[i - 1] - not true for IA[%d] >= IA[%d] : %d < %d\n",
				i, i - 1, IA[i], IA[i - 1]);
			process->abort();
		}
	
	vector<T> JA;
	JA.resize(index_nonzeros);
	nbytes = fread(reinterpret_cast<char*>(&JA[0]), sizeof(T), JA.size(), infile);

	for (int i = 0, e = JA.size(); i != e; i++)
	{
		if (JA[i] >= dim)
		{
			process->cerr("JA[i] must be within column index range - not true for JA[%d] = %d\n",
				i, JA[i]);
			process->abort();
		}
	}

	for (int i = 0, row = 0; row < nno; row++)
		for (int col = IA[row]; col < IA[row + 1]; col++, i++)
		{
			IndexPair& pair = (IndexPair&)A[i];
			index_(row, JA[i]) = pair.i;
			index_(row, JA[i] + vdim * AVX_VECTOR_SIZE) = pair.j;
		}
	
	//cout << (100 - (double)index_nonzeros / (nno * dim) * 100) << "% index sparsity" << endl;
}

template<typename T>
static void read_surplus(FILE* infile, int nno, int TotalDof, Matrix<double>& surplus_)
{
	MPI_Process* process;
	MPI_ERR_CHECK(MPI_Process_get(&process));

	size_t nbytes = 0;

	char surplus_marker[] = "surplus";
	nbytes = fread(surplus_marker, sizeof(char), sizeof(surplus_marker) - 1, infile);
	if (nbytes != sizeof(surplus_marker) - 1)
	{
		if (process->isMaster())
			process->cout("Cannot read surplus data\n");
		process->abort();
	}

	unsigned int surplus_nonzeros = 0;
	nbytes = fread(reinterpret_cast<char*>(&surplus_nonzeros), sizeof(unsigned int), 1, infile);

	vector<double> A;
	A.resize(surplus_nonzeros);
	nbytes = fread(reinterpret_cast<char*>(&A[0]), sizeof(double), A.size(), infile);

	vector<int> IA;
	IA.resize(nno + 1);
	{
		vector<T> IAT;
		IAT.resize(nno + 1);
		nbytes = fread(reinterpret_cast<char*>(&IAT[0]), sizeof(T), IAT.size(), infile);
		IA[0] = (int)IAT[0];
		for (int i = 1, e = IA.size(); i != e; i++)
			IA[i] = (int)IAT[i] + IA[i - 1];
	}

	if (IA[0] != 0)
	{
		process->cerr("IA[0] must be 0\n");
		process->abort();
	}

	for (int i = 1, e = IA.size(); i != e; i++)
		if (IA[i] < IA[i - 1])
		{
			process->cerr("IA[i] must be not less than IA[i - 1] - not true for IA[%d] >= IA[%d] : %d < %d\n",
				i, i - 1, IA[i], IA[i - 1]);
			process->abort();
		}
	
	vector<T> JA;
	JA.resize(surplus_nonzeros);
	nbytes = fread(reinterpret_cast<char*>(&JA[0]), sizeof(T), JA.size(), infile);

	for (int i = 0, e = JA.size(); i != e; i++)
	{
		if (JA[i] >= TotalDof)
		{
			process->cerr("JA[i] must be within column index range - not true for JA[%d] = %d\n", i, JA[i]);
			process->abort();
		}
	}

	for (int i = 0, row = 0; row < nno; row++)
		for (int col = IA[row]; col < IA[row + 1]; col++, i++)
			surplus_(row, JA[i]) = A[i];
	
	//cout << (100 - (double)surplus_nonzeros / (nno * TotalDof) * 100) << "% surplus sparsity" << endl;
}

struct AVXIndex
{
	uint8_t i[AVX_VECTOR_SIZE], j[AVX_VECTOR_SIZE];
	
	AVXIndex()
	{
		memset(i, 0, sizeof(i));
		memset(j, 0, sizeof(j));
	}
	
	__attribute__((always_inline))
	bool isEmpty() const
	{
		AVXIndex empty;
		if (memcmp(this, &empty, sizeof(AVXIndex)) == 0)
			return true;
		
		return false;
	}

	__attribute__((always_inline))
	bool isEmpty(int k) const
	{
		return (i[k] == 0) && (j[k] == 0);
	}
};

// Compressed index matrix packed into AVX-sized chunks.
// Specialized from vector in order to place the length value
// right before its corresponding row data (for caching).
class AVXIndexes: private std::vector<char, AlignedAllocator<AVXIndex> >
{
	int nnoMax;
	int dim;
	
	static const int szlength = 4 * sizeof(double);

	__attribute__((always_inline))
	static int nnoMaxAlign(int nnoMax_)
	{
		// Pad indexes rows to AVX_VECTOR_SIZE.
		if (nnoMax_ % AVX_VECTOR_SIZE)
			nnoMax_ += AVX_VECTOR_SIZE - nnoMax_ % AVX_VECTOR_SIZE;

		return nnoMax_ / AVX_VECTOR_SIZE;
	}
	
	__attribute__((always_inline))
	int& length(int j)
	{
		return reinterpret_cast<int*>(
			reinterpret_cast<char*>(&this->operator()(0, j)) - sizeof(int))[0];
	}

	__attribute__((always_inline))
	const int& length(int j) const
	{
		return reinterpret_cast<const int*>(
			reinterpret_cast<const char*>(&this->operator()(0, j)) - sizeof(int))[0];
	}
	
	__attribute__((always_inline))
	void setLength(int j, int length_)
	{
		length(j) = length_;
	}

public :

	AVXIndexes() :
		nnoMax(0), dim(0),
		std::vector<char, AlignedAllocator<AVXIndex> >()
	
	{ }

	AVXIndexes(int nnoMax_, int dim_) :
		nnoMax(nnoMaxAlign(nnoMax_)), dim(dim_),
		std::vector<char, AlignedAllocator<AVXIndex> >(
			dim_ * (nnoMaxAlign(nnoMax_) * sizeof(AVXIndex) + szlength))

	{ }
	
	void resize(int nnoMax_, int dim_)
	{
		nnoMax = nnoMaxAlign(nnoMax_);
		dim = dim_;

		vector<char, AlignedAllocator<AVXIndex> >::resize(
			dim_ * (nnoMaxAlign(nnoMax_) * sizeof(AVXIndex) + szlength));
	}
	
	__attribute__((always_inline))
	AVXIndex& operator()(int i, int j)
	{
		return *reinterpret_cast<AVXIndex*>(
			&std::vector<char, AlignedAllocator<AVXIndex> >::operator[]((j * nnoMax + i) * sizeof(AVXIndex) + (j + 1) * szlength));
	}

	__attribute__((always_inline))
	const AVXIndex& operator()(int i, int j) const
	{
		return *reinterpret_cast<const AVXIndex*>(
			&std::vector<char, AlignedAllocator<AVXIndex> >::operator[]((j * nnoMax + i) * sizeof(AVXIndex) + (j + 1) * szlength));
	}
	
	__attribute__((always_inline))
	int getLength(int j) const
	{
		return length(j);
	}
	
	void calculateLengths()
	{
		for (int j = 0; j < dim; j++)
		{
			int length = 0;
			for (int i = 0; i < nnoMax; i++)
			{
				AVXIndex& index = this->operator()(i, j);
				if (index.isEmpty())
					break;
				
				length++;
			}
			
			setLength(j, length);
		}
	}
};

void DataDense::load(const char* filename, int istate)
{
	MPI_Process* process;
	MPI_ERR_CHECK(MPI_Process_get(&process));
	const Parameters& params = Interpolator::getInstance()->getParameters();

	if (istate >= nstates)
	{
		process->cerr("State %d index out of range 0 .. %d\n", istate, nstates - 1);
		process->abort();
	}

	if (loadedStates[istate])
	{
		process->cerr("State %d data is already loaded\n", istate);
		process->abort();
	}

	bool compressed = isCompressed(filename);

	// First try to read file as binary.
	bool binaryio = true;
	FILE* infile = NULL;
	infile = fopen(filename, "rb");

	int dim, vdim, nno, TotalDof, Level;

	if (compressed)
	{
		size_t nbytes = 0;
		char format_marker[] = "          ";		
		nbytes += fread(reinterpret_cast<void*>(format_marker), sizeof(char), sizeof(format_marker) - 1, infile);
		nbytes += fread(reinterpret_cast<void*>(&dim), 1, sizeof(dim), infile);
		nbytes += fread(reinterpret_cast<void*>(&nno), 1, sizeof(nno), infile);
		nbytes += fread(reinterpret_cast<void*>(&TotalDof), 1, sizeof(TotalDof), infile);
		nbytes += fread(reinterpret_cast<void*>(&Level), 1, sizeof(Level), infile);
		if (nbytes != sizeof(format_marker) - 1 + sizeof(dim) + sizeof(nno) + sizeof(TotalDof) + sizeof(Level))
		{
			process->cerr("Error reading file: %s\n", filename);
			process->abort();
		}
	}
	else
	{
		// Guess text/binary format by comparing dim with nagents.
		{
			size_t nbytes = 0;
			nbytes += fread(reinterpret_cast<void*>(&dim), 1, sizeof(dim), infile);
			nbytes += fread(reinterpret_cast<void*>(&nno), 1, sizeof(nno), infile);
			nbytes += fread(reinterpret_cast<void*>(&TotalDof), 1, sizeof(TotalDof), infile);
			nbytes += fread(reinterpret_cast<void*>(&Level), 1, sizeof(Level), infile);
			if (nbytes != sizeof(dim) + sizeof(nno) + sizeof(TotalDof) + sizeof(Level))
			{
				process->cerr("Error reading file: %s\n", filename);
				process->abort();
			}
		}

		if (dim != params.nagents)
		{
			// Reopen file for text-mode reading.
			fclose(infile);
			infile = fopen(filename, "r");
			binaryio = false;

			int nbytes = 0;
			nbytes += fscanf(infile, "%d", &dim);
			nbytes += fscanf(infile, "%d", &nno);
			nbytes += fscanf(infile, "%d", &TotalDof);
			nbytes += fscanf(infile, "%d", &Level);
			if (nbytes <= 0)
			{
				process->cerr("Error reading file: %s\n", filename);
				process->abort();
			}
		}
	}

	if (dim != params.nagents)
	{
		process->cerr("File \"%s\" # of dimensions (%d) mismatches config (%d)\n",
			filename, dim, params.nagents);
		process->abort();
	}
	
	// Check is the current state the first loaded state.
	bool firstLoadedState = true;
	for (int i = 0; i < nstates; i++)
	{
		if (loadedStates[i])
		{
			firstLoadedState = false;
			break;
		}
	}
	if (firstLoadedState)
	{
		this->dim = dim;
		this->TotalDof = TotalDof;
	}
	else
	{
		if (dim != this->dim)
		{
			process->cerr("File \"%s\" # of dimensions (%d) mismatches another state (%d)\n",
				filename, dim, this->dim);
			process->abort();
		}
		if (TotalDof != this->TotalDof)
		{
			process->cerr("File \"%s\" TotalDof (%d) mismatches another state (%d)\n",
				filename, TotalDof, this->TotalDof);
			process->abort();
		}
	}

	// Pad all indexes to 4-element boundary.
	vdim = dim / AVX_VECTOR_SIZE;
	if (dim % AVX_VECTOR_SIZE) vdim++;
	int nsd = 2 * vdim * AVX_VECTOR_SIZE;

	index[istate].resize(nno, nsd);
	index[istate].fill(0);

	surplus[istate].resize(nno, TotalDof);
	surplus[istate].fill(0.0);

	if (!compressed)
	{
		int j = 0;
		while (infile)
		{
			if (j == nno) break;

			{
				vector<int> values(2 * dim);
				if (binaryio)
				{
					size_t nbytes = 0;
					nbytes += fread(reinterpret_cast<int*>(&values[0]), 1, sizeof(int) * 2 * dim, infile);
					if (nbytes != sizeof(int) * 2 * dim)
					{
						process->cerr("Error reading file: %s\n", filename);
						process->abort();
					}
				}
				else
				{
					for (int i = 0; i < 2 * dim; i++)
						int nbytes = fscanf(infile, "%d", &values[i]);
				}

				for (int i = 0; i < dim; )
				{
					for (int v = 0; (v < AVX_VECTOR_SIZE) && (i < dim); v++, i++)
					{
						int value = values[i];
						value = 2 << (value - 2);
						index[istate](j, i) = value;
					}
				}
				for (int i = 0; i < dim; )
				{
					for (int v = 0; (v < AVX_VECTOR_SIZE) && (i < dim); v++, i++)
					{
						int value = values[i + dim];
						value--;
						// Precompute "j" to merge two cases into one:
						// (((i) == 0) ? (1) : (1 - fabs((x) * (i) - (j)))).
						if (!index[istate](j, i)) value = 0;
						index[istate](j, i + vdim * AVX_VECTOR_SIZE) = value;
					}
				}
			}

			{
				vector<double> values(TotalDof);
				if (binaryio)
				{
					size_t nbytes = 0;
					nbytes += fread(reinterpret_cast<double*>(&values[0]), 1, sizeof(double) * TotalDof, infile);
					if (nbytes != sizeof(double) * TotalDof)
					{
						process->cerr("Error reading file: %s\n", filename);
						process->abort();
					}
				}
				else
				{
					for (int i = 0; i < TotalDof; i++)
						int nbytes = fscanf(infile, "%lf", &values[i]);
				}

				for (int i = 0; i < TotalDof; i++)
				{
					double value = values[i];
					surplus[istate](j, i) = value;
				}
			}

			j++;
		}
	}
	else
	{
		int szt = 0;
		if (dim <= numeric_limits<unsigned char>::max())
			szt = 1;
		else if (dim <= numeric_limits<unsigned short>::max())
			szt = 2;
		else
			szt = 4;
	
		//cout << "Using " << szt << "-byte IA/JA size for index" << endl;
	
		switch (szt)
		{
		case 1 :
			read_index<unsigned char>(infile, nno, dim, vdim, index[istate]);
			break;
		case 2 :
			read_index<unsigned short>(infile, nno, dim, vdim, index[istate]);
			break;
		case 4 :
			read_index<unsigned int>(infile, nno, dim, vdim, index[istate]);
			break;
		}

		if (TotalDof <= numeric_limits<unsigned char>::max())
			szt = 1;
		else if (TotalDof <= numeric_limits<unsigned short>::max())
			szt = 2;
		else
			szt = 4;
	
		//cout << endl;
		//cout << "Using " << szt << "-byte IA/JA size for surplus" << endl;
	
		switch (szt)
		{
		case 1 :
			read_surplus<unsigned char>(infile, nno, TotalDof, surplus[istate]);
			break;
		case 2 :
			read_surplus<unsigned short>(infile, nno, TotalDof, surplus[istate]);
			break;
		case 4 :
			read_surplus<unsigned int>(infile, nno, TotalDof, surplus[istate]);
			break;
		}
	}
	
	fclose(infile);

#if 0
	for (int i = 0; i < surplus[istate].dimy(); i++)
	{
		for (int j = 0; j < surplus[istate].dimx(); j++)
			process->cout("%e ", surplus[istate](i, j));
		process->cout("\n");
	}
#endif

#if 0
	for (int j = 0; j < index[istate].dimy(); j++)
	{
		for (int i = 0; i < index[istate].dimx() / 2; i++)
			process->cout("(%d, %d) ", index[istate](j, i), index[istate](j, i + vdim));
		process->cout("\n");
	}
#endif

	loadedStates[istate] = true;
}

void DataDense::clear()
{
	fill(loadedStates.begin(), loadedStates.end(), false);
}

DataDense::DataDense(int nstates_) : nstates(nstates_)
{
	index.resize(nstates);
	surplus.resize(nstates);
	loadedStates.resize(nstates);
	fill(loadedStates.begin(), loadedStates.end(), false);
}

DataDense::~DataDense()
{
}

