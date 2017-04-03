#include "check.h"
#include "Data.h"
#include "interpolator.h"

#include <fstream>
#include <iostream>
#include <mpi.h>

using namespace NAMESPACE;
using namespace std;

class Data::Host::DataHost
{
	std::vector<int> nfreqs;
	std::vector<XPS::Host> xps;
	std::vector<int> szxps;
	std::vector<Chains::Host> chains;
	std::vector<Matrix<real>::Host> surplus;

public :

	DataHost(int nstates) :

		nfreqs(nstates),
		xps(nstates),
		szxps(nstates),
		chains(nstates),
		surplus(nstates)

	{ }

	friend class Data;
	friend class Data::Host;
};

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
static void read_index(FILE* infile, int nno, int dim, int vdim, Matrix<int>::Host& index_)
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
			process->cerr("JA[i] must be within column index range - not true for JA[%d] = %f\n",
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
static void read_surplus(FILE* infile, int nno, int TotalDof, Matrix<double>::Host& surplus_)
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
			process->cerr("IA[i] must be not less than IA[i - 1] - not true for IA[%d] >= IA[%d] : %f < %f\n",
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
			process->cerr("JA[i] must be within column index range - not true for JA[%d] = %f\n", i, JA[i]);
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

void Data::load(const char* filename, int istate)
{
	MPI_Process* process;
	MPI_ERR_CHECK(MPI_Process_get(&process));
	const Parameters& params = Interpolator::getInstance()->getParameters();

	if (loadedStates[istate])
	{
		process->cerr("State %d data is already loaded\n", istate);
		process->abort();
	}

	bool compressed = isCompressed(filename);

	FILE* infile = NULL;
	if (params.binaryio)
		infile = fopen(filename, "r");
	else
		infile = fopen(filename, "rb");

	if (!infile)
	{
		process->cerr("Error opening file: %s\n", filename);
		exit(1);
	}  

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

	if (dim != params.nagents)
	{
		process->cerr("File \"%s\" # of dimensions (%d) mismatches config (%d)\n",
			filename, dim, params.nagents);
		process->abort();
	}

	// Pad all indexes to 4-element boundary.
	vdim = dim / AVX_VECTOR_SIZE;
	if (dim % AVX_VECTOR_SIZE) vdim++;
	int nsd = 2 * vdim * AVX_VECTOR_SIZE;

	Matrix<int>::Host index;
	index.resize(nno, nsd);
	index.fill(0);

	std::vector<Matrix<real>::Host>& surplus = host.data->surplus;
	surplus[istate].resize(nno, TotalDof);
	surplus[istate].fill(0.0);

	if (!compressed)
	{
		int j = 0;
		while (infile)
		{
			if (j == nno) break;

			for (int i = 0; i < dim; )
			{
				for (int v = 0; (v < AVX_VECTOR_SIZE) && (i < dim); v++, i++)
				{
					int value;
					int nbytes = fscanf(infile, "%d", &value);
					value = 2 << (value - 2);
					index(j, i) = value;
				}
			}
			for (int i = 0; i < dim; )
			{
				for (int v = 0; (v < AVX_VECTOR_SIZE) && (i < dim); v++, i++)
				{
					int value;
					int nbytes = fscanf(infile, "%d", &value);
					value--;
					// Precompute "j" to merge two cases into one:
					// (((i) == 0) ? (1) : (1 - fabs((x) * (i) - (j)))).
					if (!index(j, i)) value = 0;
					index(j, i + vdim * AVX_VECTOR_SIZE) = value;
				}
			}
			for (int i = 0; i < TotalDof; i++)
			{
				double value;
				int nbytes = fscanf(infile, "%lf", &value);
				surplus[istate](j, i) = value;
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
			read_index<unsigned char>(infile, nno, dim, vdim, index);
			break;
		case 2 :
			read_index<unsigned short>(infile, nno, dim, vdim, index);
			break;
		case 4 :
			read_index<unsigned int>(infile, nno, dim, vdim, index);
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

	const pair<int, int> zero = make_pair(0, 0);

	vdim *= AVX_VECTOR_SIZE;

	struct State
	{
		int nfreqs;
		XPS::Host& xps;
		int& szxps;
		Chains::Host& chains;
		
		State(XPS::Host& xps_, int& szxps_, Chains::Host& chains_) : xps(xps_), szxps(szxps_), chains(chains_) { }
		
		~State()
		{
			szxps = xps.size();
		}
	}
	state(host.data->xps[istate], host.data->szxps[istate], host.data->chains[istate]);

	// Calculate maximum frequency across row indexes.
	state.nfreqs = 0;
	{
		map<int, int> freqs;
		for (int i = 0; i < nno; i++)
			for (int j = 0; j < dim; j++)
			{
				// Get pair.
				pair<int, int> value = make_pair(index(i, j), index(i, j + vdim));

				// If both indexes are zeros, do nothing.
				if (value == zero)
					continue;
			
				freqs[i]++;
			}

		for (map<int, int>::iterator i = freqs.begin(), e = freqs.end(); i != e; i++)
			state.nfreqs = max(state.nfreqs, i->second);
	}

	vector<map<uint32_t, uint32_t> > transMaps(state.nfreqs);
	vector<AVXIndexes> avxinds(state.nfreqs);

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
				pair<int, int> value = make_pair(index(i, j), index(i, j + vdim));

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
	{
		Matrix<double>::Host surplusOld = surplus[istate];
		Matrix<double>::Host& surplusNew = surplus[istate];

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

		// Loop to calculate temps.
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

			if (freq.avxinds.getLength(j))
			{
				const AVXIndex& index = freq.avxinds(freq.avxinds.getLength(j) - 1, j);
				for (int k = AVX_VECTOR_SIZE - 1; k >= 0; k--)
					if (index.isEmpty(k)) itemp--;
			}
		}
	}	
	if (process->isMaster())
		process->cout("%d unique xp(s) to compute\n", map.xps.size());

	// Create all possible chains between frequencies.
	state.chains.resize(nno * state.nfreqs);
	for (int i = 0, ichain = 0; i < nno; i++)
	{
		state.chains[ichain++] = temps[0][i];

		for (int ifreq = 1; ifreq < state.nfreqs; ifreq++)
			state.chains[ichain++] = temps[ifreq][trans[ifreq][i]];
	}
	if (process->isMaster())
		process->cout("%d chains of %d xp(s) to build\n",
			state.chains.size() / state.nfreqs, state.nfreqs);
	
	// Convert xps from map to vector.
	state.xps.resize(map.xps.size());
	for (std::map<Index<uint16_t>, uint32_t>::const_iterator i = map.xps.begin(), e = map.xps.end(); i != e; i++)
		state.xps[i->second] = i->first;

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
		for (int ifreq = 0; ifreq < state.nfreqs; ifreq++)
			state.chains[i * state.nfreqs + ifreq] = vv[i][ifreq];
	}	

	// Reorder surpluses.
	{
		Matrix<double>::Host surplusOld = surplus[istate];
		Matrix<double>::Host& surplusNew = surplus[istate];

		for (int i = 0; i < nno; i++)
		{
			int oldind = vv[i][state.nfreqs];
			int newind = i;

			memcpy(&surplusNew(newind, 0), &(surplusOld(oldind, 0)), surplusNew.dimx() * sizeof(double));
		}
	}

	host.data->nfreqs[istate] = state.nfreqs;
	
	loadedStates[istate] = true;

	// Copy data from host to device memory.
	device.setNfreqs(istate, state.nfreqs);
	device.setXPS(istate, &state.xps);
	device.setChains(istate, &state.chains);
	device.setSurplus(istate, &surplus[istate]);
}

void Data::clear()
{
	fill(loadedStates.begin(), loadedStates.end(), false);
}

Data::Data(int nstates_) : nstates(nstates_), host(nstates), device(nstates)
{
	loadedStates.resize(nstates);
	fill(loadedStates.begin(), loadedStates.end(), false);
}

Data::~Data()
{
}

Data::Host::Host(int nstates)
{
	data.reset(new DataHost(nstates));
}

int* Data::Host::getNfreqs(int istate)
{
	return &data->nfreqs[istate];
}

XPS::Host* Data::Host::getXPS(int istate)
{
	return &data->xps[istate];
}

int* Data::Host::getSzXPS(int istate)
{
	return &data->szxps[istate];
}

Chains::Host* Data::Host::getChains(int istate)
{
	return &data->chains[istate];
}

Matrix<real>::Host* Data::Host::getSurplus(int istate)
{
	return &data->surplus[istate];
}

// Determine the size of type on device.
template<typename T>
static __global__ void deviceSizeOf(size_t* result)
{
	*result = sizeof(T);
}

template<typename T>
static __global__ void constructDeviceData(int length, char* ptr)
{
	for (int i = 0; i < length; i++)
		new(&ptr[i * sizeof(T)]) T();
}

template<typename T>
static __global__ void destroyDeviceData(int length, char* ptr)
{
	for (int i = 0; i < length; i++)
		delete (T*)&ptr[i * sizeof(T)];
}

// Host array of elements in device memory, whose size is
// determined on device in runtime.
template<typename T>
class DeviceSizedArray
{
	int length;
	size_t size;
	char* ptr;

public :

	DeviceSizedArray(int length_) : length(length_)
	{
		// Determine the size of target type.
		size_t* dSize;
		CUDA_ERR_CHECK(cudaMalloc(&dSize, sizeof(size_t)));
		deviceSizeOf<T><<<1, 1>>>(dSize);
		CUDA_ERR_CHECK(cudaGetLastError());
		CUDA_ERR_CHECK(cudaDeviceSynchronize());
		CUDA_ERR_CHECK(cudaMemcpy(&size, dSize, sizeof(size_t), cudaMemcpyDeviceToHost));
		CUDA_ERR_CHECK(cudaFree(dSize));
		
		// Make sure the size of type is the same on host and on device.
		if (size != sizeof(T))
		{
			cerr << "Unexpected unequal sizes of type T in DeviceSizedArray<T> on host and device" << endl;
			MPI_Process* process;
			MPI_ERR_CHECK(MPI_Process_get(&process));
			process->abort();
		}

		// Allocate array.		
		CUDA_ERR_CHECK(cudaMalloc(&ptr, size * length));
		
		// Construct individual array elements from within the device kernel code,
		// using placement new operator.
		constructDeviceData<T><<<1, 1>>>(length, ptr);
		CUDA_ERR_CHECK(cudaGetLastError());
		CUDA_ERR_CHECK(cudaDeviceSynchronize());
	}
	
	~DeviceSizedArray()
	{
		// Destroy individual array elements from within the device kernel code.
		destroyDeviceData<T><<<1, 1>>>(length, ptr);
		CUDA_ERR_CHECK(cudaGetLastError());
		CUDA_ERR_CHECK(cudaDeviceSynchronize());			

		// Delete each indivudual element on host, which triggers deletion of
		// data buffer previously allocated on host with cudaMalloc.
		vector<T> elements;
		elements.resize(length);
		CUDA_ERR_CHECK(cudaMemcpy(&elements[0], ptr, length * sizeof(T), cudaMemcpyDeviceToHost));
		for (int i = 0; i < length; i++)
		{
			// Set that matrix owns its underlying data buffer.
			elements[i].disownData();

			delete &elements[i];
		}

		// Free device memory used for array elements.
		CUDA_ERR_CHECK(cudaFree(ptr));
	}

	T& operator[](int i)
	{
		return (T&)ptr[i * size];
	}
};

class Data::Device::DataDevice
{
	// These containers shall be entirely in device memory, including vectors.
	int* nfreqs;
	DeviceSizedArray<XPS::Device> xps;
	DeviceSizedArray<Chains::Device> chains;
	DeviceSizedArray<Matrix<real>::Device> surplus;

public :

	DataDevice(int nstates) :

		xps(nstates),
		chains(nstates),
		surplus(nstates)

	{
		CUDA_ERR_CHECK(cudaMalloc(&nfreqs, sizeof(int) * nstates));
	}

	~DataDevice()
	{
		CUDA_ERR_CHECK(cudaFree(nfreqs));
	}

	friend class Data;
	friend class Data::Device;
};	

Data::Device::Device(int nstates_) : nstates(nstates_)
{
	data.reset(new DataDevice(nstates_));
}

int* Data::Device::getNfreqs(int istate)
{
	return &data->nfreqs[istate];
}

XPS::Device* Data::Device::getXPS(int istate)
{
	return &data->xps[istate];
}

Chains::Device* Data::Device::getChains(int istate)
{
	return &data->chains[istate];
}

Matrix<real>::Device* Data::Device::getSurplus(int istate)
{
	return &data->surplus[istate];
}

void Data::Device::setNfreqs(int istate, int nfreqs)
{
	CUDA_ERR_CHECK(cudaMemcpy(&data->nfreqs[istate], &nfreqs, sizeof(int),
		cudaMemcpyHostToDevice));
}

void Data::Device::setXPS(int istate, XPS::Host* xps)
{
	XPS::Host& xpsHost = *xps;
	
	XPS::Device xpsDevice;
	CUDA_ERR_CHECK(cudaMemcpy(&xpsDevice, &data->xps[istate], sizeof(XPS::Device),
		cudaMemcpyDeviceToHost));
	xpsDevice.resize(xpsHost.size());
	
	CUDA_ERR_CHECK(cudaMemcpy(xpsDevice.getData(), &xpsHost[0],
		xpsHost.size() * sizeof(Index<uint16_t>), cudaMemcpyHostToDevice));

	// Set that vector does not own its underlying data buffer.
	xpsDevice.disownData();

	CUDA_ERR_CHECK(cudaMemcpy(&data->xps[istate], &xpsDevice, sizeof(XPS::Device),
		cudaMemcpyHostToDevice));
}

void Data::Device::setChains(int istate, Chains::Host* chains)
{
	Chains::Host& chainsHost = *chains;
	
	Chains::Device chainsDevice;
	CUDA_ERR_CHECK(cudaMemcpy(&chainsDevice, &data->chains[istate], sizeof(Chains::Device),
		cudaMemcpyDeviceToHost));
	chainsDevice.resize(chainsHost.size());
	
	CUDA_ERR_CHECK(cudaMemcpy(chainsDevice.getData(), &chainsHost[0],
		chainsHost.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
	
	// Set that vector does not own its underlying data buffer.
	chainsDevice.disownData();

	CUDA_ERR_CHECK(cudaMemcpy(&data->chains[istate], &chainsDevice, sizeof(Chains::Device),
		cudaMemcpyHostToDevice));
}

void Data::Device::setSurplus(int istate, Matrix<real>::Host* matrix)
{
	Matrix<double>::Host& matrixHost = *matrix;

	Matrix<double>::Device matrixDevice;
	CUDA_ERR_CHECK(cudaMemcpy(&matrixDevice, &data->surplus[istate], sizeof(Matrix<double>::Device),
		cudaMemcpyDeviceToHost));
	matrixDevice.resize(matrixHost.dimy(), matrixHost.dimx());

	// It is assumed to be safe to copy padded data from host to device matrix,
	// as they use the same memory allocation policy.
	size_t size = (ptrdiff_t)&matrixHost(matrixHost.dimy() - 1, matrixHost.dimx() - 1) -
		(ptrdiff_t)matrixHost.getData() + sizeof(double);
	CUDA_ERR_CHECK(cudaMemcpy(matrixDevice.getData(), matrixHost.getData(), size, cudaMemcpyHostToDevice));

	// Set that matrix does not own its underlying data buffer.
	matrixDevice.disownData();

	CUDA_ERR_CHECK(cudaMemcpy(&data->surplus[istate], &matrixDevice, sizeof(Matrix<double>::Device),
		cudaMemcpyHostToDevice));
}

extern "C" Data* getData(int nstates)
{
	return new Data(nstates);
}

