#include "check.h"
#include "interpolator.h"
#include "Data.h"

#include <cstdlib>
#include <limits>
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
		exit(1);
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

void DataSparse::load(const char* filename, int istate)
{
	DataDense::load(filename, istate);

	loadedStates[istate] = false;

	int vdim = dim / AVX_VECTOR_SIZE;
	if (dim % AVX_VECTOR_SIZE) vdim++;

	int nno = surplus[istate].dimy();

	MPI_Process* process;
	MPI_ERR_CHECK(MPI_Process_get(&process));
	const Parameters& params = Interpolator::getInstance()->getParameters();

	bool showDataAnalysis = false;
	const char* iShowDataAnalysis = getenv("SHOW_DATA_ANALYSIS");
	if (iShowDataAnalysis)
		if (atoi(iShowDataAnalysis))
			showDataAnalysis = true;

	const pair<int, int> zero = make_pair(0, 0);

	vdim *= AVX_VECTOR_SIZE;

	struct State
	{
		int nfreqs;
		XPS& xps;
		Chains& chains;
		
		State(XPS& xps_, Chains& chains_) : xps(xps_), chains(chains_) { }
	}
	state(xps[istate], chains[istate]);

	// Calculate the maximum number of non-zero values across individual rows.
	state.nfreqs = index[istate].maxRowPopulation();

	vector<map<uint32_t, uint32_t> > transMaps(state.nfreqs);
	vector<AVXIndexes> avxinds(state.nfreqs);

	// Decompose single dense index matrix into a set of sparse index matrices.
	// Each sparse matrix shall contain no more than a single element from
	// each dense matrix row.
	for (int ifreq = 0; ifreq < state.nfreqs; ifreq++)
	{
		vector<Index<uint32_t> > indexes;
		indexes.resize(dim);

		// Convert (i, I) indexes matrix to sparse format.
		vector<uint32_t> nnz(dim);
		vector<uint32_t> freqs(nno);
		for (int i = 0; i < nno; i++)
			for (int j = 0; j < dim; j++)
			{
				// Get pair.
				pair<int, int> value = make_pair(index[istate](i, j), index[istate](i, j + vdim));

				// If both indexes are zeros, do nothing.
				if (value == zero)
					continue;

				freqs[i]++;

				if (freqs[i] != ifreq + 1)
					continue;

				// Find free position for non-zero pair.
				// If no free position, add new free row.
				if (nnz[j] == indexes.size() / dim)
					indexes.resize(indexes.size() + dim);

				Index<uint32_t>& index = indexes[nnz[j] * dim + j];
				index.i = value.first;
				index.j = value.second;
				index.rowind() = i;
				nnz[j]++;
			}
	
		// Reorder indexes.
		map<uint32_t, uint32_t>& transMap = transMaps[ifreq];
		for (int j = 0, order = 0; j < dim; j++)
			for (int i = 0, e = indexes.size() / dim; i < e; i++)
			{
				Index<uint32_t>& index = indexes[i * dim + j];

				if ((index.i == 0) && (index.j == 0)) continue;
			
				if (transMap.find(index.rowind()) == transMap.end())
				{
					transMap[index.rowind()] = order;
					index.rowind() = order;
					order++;
				}
				else
					index.rowind() = transMap[index.rowind()];
			}

		// Pad indexes rows to AVX_VECTOR_SIZE.
		if ((indexes.size() / dim) % AVX_VECTOR_SIZE)
			indexes.resize(indexes.size() +
				dim * (AVX_VECTOR_SIZE - (indexes.size() / dim) % AVX_VECTOR_SIZE));
	
		// Do not forget to reorder unseen indexes.
		for (int i = 0, last = transMap.size(); i < nno; i++)
			if (transMap.find(i) == transMap.end())
				transMap[i] = last++;

		AVXIndexes& avxindsFreq = avxinds[ifreq];
		avxindsFreq.resize(nno, dim);

		// Copy sparse indexes matrix for the current frequency
		// into new AVX-aligned sparse matrix, omitting the original
		// dense index matrix row index, which is no longer needed
		// after renumbering.
		for (int j = 0, length = indexes.size() / dim / AVX_VECTOR_SIZE; j < dim; j++)
		{
			for (int i = 0; i < length; i++)
			{
				AVXIndex& index = avxindsFreq(i, j);
				for (int k = 0; k < AVX_VECTOR_SIZE; k++)
				{
					int idx = (i * AVX_VECTOR_SIZE + k) * dim + j;
					index.i[k] = indexes[idx].i;
					index.j[k] = indexes[idx].j;
				}
			}
		}
	
		avxindsFreq.calculateLengths();
	}

	// Reorder surpluses.
	if (state.nfreqs)
	{
		Matrix<double> surplusOld = surplus[istate];
		Matrix<double>& surplusNew = surplus[istate];

		for (map<uint32_t, uint32_t>::iterator i = transMaps[0].begin(), e = transMaps[0].end(); i != e; i++)
		{
			int oldind = i->first;
			int newind = i->second;

			memcpy(&surplusNew(newind, 0), &(surplusOld(oldind, 0)), surplusNew.dimx() * sizeof(double));
		}
	}
	
	// Recalculate translations between frequencies relative to the
	// first frequency.
	for (int ifreq = 1; ifreq < state.nfreqs; ifreq++)
	{
		map<uint32_t, uint32_t> transRelative;
		for (int i = 0; i < nno; i++)
			transRelative[transMaps[0][i]] = transMaps[ifreq][i];
		transMaps[ifreq] = transRelative;
	}

	// Store maps down to vectors.
	vector<vector<uint32_t> > trans(state.nfreqs);
	for (int ifreq = 1; ifreq < state.nfreqs; ifreq++)
	{
		vector<uint32_t>& transFreq = trans[ifreq];
		transFreq.resize(nno);
		for (int i = 0; i < nno; i++)
			transFreq[i] = transMaps[ifreq][i];
	}
	
	int nnoAligned = nno;
	if (nno % AVX_VECTOR_SIZE)
		nnoAligned += AVX_VECTOR_SIZE - nno % AVX_VECTOR_SIZE;

	// One extra vector size for *hi part in AVX code below.
	nnoAligned += AVX_VECTOR_SIZE;

	vector<vector<uint32_t, AlignedAllocator<uint32_t> > > temps(
		state.nfreqs, vector<uint32_t, AlignedAllocator<uint32_t> >(nnoAligned, 0));

	struct Map
	{
		map<Index<uint16_t>, uint32_t> xps;
	}
	map;

	// First index denotes an empty frequency value.
	map.xps[Index<uint16_t>(0, 0, 0)] = 0;
	
	// Loop through all frequences.
	for (int ifreq = 0, ixp = 1; ifreq < state.nfreqs; ifreq++)
	{
		struct Freq
		{
			const AVXIndexes& avxinds;
			vector<uint32_t, AlignedAllocator<uint32_t> >& temps;
			
			Freq(const AVXIndexes& avxinds_, vector<uint32_t, AlignedAllocator<uint32_t> >& temps_) :
				avxinds(avxinds_), temps(temps_) { }
		}
		freq(avxinds[ifreq], temps[ifreq]);

		// In each frequency matrix avxinds, assign each element a global inter-frequency index XPS:
		// either a new index for unseen value, or existing for already known.
		for (int j = 0, itemp = 0; j < dim; j++)
		{
			for (int i = 0, e = freq.avxinds.getLength(j); i < e; i++)
			{
				const AVXIndex& avxind = freq.avxinds(i, j);

				for (int k = 0; k < AVX_VECTOR_SIZE; k++, itemp++)
				{
					Index<uint16_t> ind(avxind.i[k], avxind.j[k], j);

					if (ind.isEmpty()) continue;

					if (map.xps.find(ind) == map.xps.end())
						map.xps[ind] = ixp++;

					freq.temps[itemp] = map.xps[ind];
				}
			}			

			// Do not index tailing empty elements.
			// TODO ???
			if (freq.avxinds.getLength(j))
			{
				const AVXIndex& index = freq.avxinds(freq.avxinds.getLength(j) - 1, j);
				for (int k = AVX_VECTOR_SIZE - 1; k >= 0; k--)
					if (index.isEmpty(k)) itemp--;
			}
		}
	}
	if (showDataAnalysis)
	{
		// The size of XPS index denotes how many meaninful linear basis
		// calculations to perform.
		if (process->isMaster())
			process->cout("%zu unique xp(s) to compute\n", (size_t)map.xps.size());
	}

	// Create all possible chains between frequencies.
	state.chains.resize(nno * state.nfreqs);
	if (state.nfreqs)
	{	
		for (int i = 0, ichain = 0; i < nno; i++)
		{
			state.chains[ichain++] = temps[0][i];

			for (int ifreq = 1; ifreq < state.nfreqs; ifreq++)
				state.chains[ichain++] = temps[ifreq][trans[ifreq][i]];
		}
	}
	if (showDataAnalysis)
	{
		if (process->isMaster())
			process->cout("%zu chains of %d xp(s) to build\n",
				(size_t)(state.chains.size() / state.nfreqs), state.nfreqs);
	}
	
	// Convert xps from map to vector.
	xps[istate].resize(map.xps.size());
	for (std::map<Index<uint16_t>, uint32_t>::const_iterator i = map.xps.begin(), e = map.xps.end(); i != e; i++)
		xps[istate][i->second] = i->first;

	// Sort array of chains first by first frequence, then by second,
	// then by third, etc.
	vector<vector<uint32_t> > vv(nno);
	for (int i = 0, e = vv.size(); i < e; i++)
	{
		vv[i].resize(state.nfreqs + 1);
		for (int ifreq = 0; ifreq < state.nfreqs; ifreq++)
			vv[i][ifreq] = state.chains[i * state.nfreqs + ifreq];
		vv[i][state.nfreqs] = i;
	}
	sort(vv.begin(), vv.end(),
		[](const vector<uint32_t>& a, const vector<uint32_t>& b) -> bool
		{
			return a[0] < b[0];
		}
	);
	for (int ifreq = 1; ifreq < state.nfreqs; ifreq++)
	{
		for (vector<vector<uint32_t> >::iterator i = vv.begin(), e = vv.end(); i != e; )
		{
			vector<vector<uint32_t> >::iterator ii = i + 1;
			for ( ; ii != e; ii++)
				if ((*ii)[ifreq - 1] != (*i)[ifreq - 1])
				{
					sort(i, ii,
						[=](const vector<uint32_t>& a, const vector<uint32_t>& b) -> bool
						{
							return a[ifreq] < b[ifreq];
						}
					);
								
					break;
				}

			i = ii;
		}
	}
	for (int i = 0, e = vv.size(); i < e; i++)
	{
		auto chain = state.chains[0];
		for (int ifreq = 0; ifreq < state.nfreqs; ifreq++)
		{
			uint32_t idx = vv[i][ifreq];
			if (idx > numeric_limits<decltype(chain)>::max())
			{
				MPI_Process* process;
			        MPI_ERR_CHECK(MPI_Process_get(&process));
				process->cerr("Chain index %u is out range predefined by type capacity\n", idx);
				process->abort();
			}
			state.chains[i * state.nfreqs + ifreq] = idx;
		}
	}	

	// Reorder surpluses.
	if (state.nfreqs)
	{
		Matrix<double> surplusOld = surplus[istate];
		Matrix<double>& surplusNew = surplus[istate];

		for (int i = 0; i < nno; i++)
		{
			int oldind = vv[i][state.nfreqs];
			int newind = i;

			memcpy(&surplusNew(newind, 0), &(surplusOld(oldind, 0)), surplusNew.dimx() * sizeof(double));
		}
	}

#if 0
	for (int i = 0; i < xps[istate].size(); i++)
		process->cout("%d -> %d (%d, %d)\n", i, xps[istate][i].index, xps[istate][i].i, xps[istate][i].j);

	for (int i = 0; i < nno; i++)
	{
		process->cout("%d", state.chains[i * state.nfreqs]);
		for (int ifreq = 1; ifreq < state.nfreqs; ifreq++)
			process->cout(" -> %d", state.chains[i * state.nfreqs + ifreq]);
		process->cout("\n");
	}
#endif

	nfreqs[istate] = state.nfreqs;
	
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

DataSparse::DataSparse(int nstates_) : DataDense(nstates_)
{
	nfreqs.resize(nstates);
	xps.resize(nstates);
	chains.resize(nstates);
}

DataDense::~DataDense()
{
}

DataSparse::~DataSparse()
{
}

