#include "check.h"
#include "Data.h"
#include "interpolator.h"

#include <fstream>
#include <iostream>
#include <map>
#include <mpi.h>
#include <utility> // pair

using namespace NAMESPACE;
using namespace std;

int Data::getNno() const { return nno; }

static bool isCompressed(const char* filename)
{
	MPI_Process* process;
	MPI_ERR_CHECK(MPI_Process_get(&process));

	ifstream infile;
	infile.open(filename, ifstream::binary);

	if (!infile.is_open())
	{
		cerr << "Error opening file: " << filename << endl;
		process->abort();
	}

	bool compressed = false;
	
	char format_marker[] = "          \0";
		
	infile.read(format_marker, strlen(format_marker));
	
	infile.close();
	
	if ((string)format_marker == "compressed")
		compressed = true;
	
	return compressed;
}

struct IndexPair
{
	unsigned short i, j;
};

template<typename T>
static void read_index(ifstream& infile, int nno, int dim, int vdim, Matrix<int>& index_)
{
	MPI_Process* process;
	MPI_ERR_CHECK(MPI_Process_get(&process));

	char index_marker[] = "index";
	infile.read(index_marker, strlen(index_marker));

	unsigned int index_nonzeros = 0;
	infile.read(reinterpret_cast<char*>(&index_nonzeros), sizeof(unsigned int));

	vector<int> A;
	A.resize(index_nonzeros);
	for (int i = 0, e = A.size(); i != e; i++)
		infile.read(reinterpret_cast<char*>(&A[i]), sizeof(IndexPair));

	vector<int> IA;
	IA.resize(nno + 1);
	{
		T value;
		infile.read(reinterpret_cast<char*>(&value), sizeof(T));
		IA[0] = value;
	}
	for (int i = 1, e = IA.size(); i != e; i++)
	{
		T value;
		infile.read(reinterpret_cast<char*>(&value), sizeof(T));
		IA[i] = (int)value + IA[i - 1];
	}

	if (IA[0] != 0)
	{
		cerr << "IA[0] must be 0" << endl;
		process->abort();
	}

	for (int i = 1, e = IA.size(); i != e; i++)
		if (IA[i] < IA[i - 1])
		{
			cerr << "IA[i] must be not less than IA[i - 1] - not true for IA[" << i << "] >= IA[" << (i - 1) <<
				"] : " << IA[i] << " < " << IA[i - 1] << endl;
			process->abort();
		}
	
	vector<int> JA;
	JA.resize(index_nonzeros);
	for (int i = 0, e = JA.size(); i != e; i++)
	{
		T value;
		infile.read(reinterpret_cast<char*>(&value), sizeof(T));
		JA[i] = (int)value;
	}

	for (int i = 0, e = JA.size(); i != e; i++)
	{
		if (JA[i] >= dim)
		{
			cerr << "JA[i] must be within column index range - not true for JA[" << i << "] = " << JA[i] << endl;
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
static void read_surplus(ifstream& infile, int nno, int TotalDof, Matrix<double>& surplus_)
{
	MPI_Process* process;
	MPI_ERR_CHECK(MPI_Process_get(&process));

	char surplus_marker[] = "surplus";
	infile.read(surplus_marker, strlen(surplus_marker));

	unsigned int surplus_nonzeros = 0;
	infile.read(reinterpret_cast<char*>(&surplus_nonzeros), sizeof(unsigned int));

	vector<double> A;
	A.resize(surplus_nonzeros);
	for (int i = 0, e = A.size(); i != e; i++)
		infile.read(reinterpret_cast<char*>(&A[i]), sizeof(double));

	vector<int> IA;
	IA.resize(nno + 1);
	{
		T value;
		infile.read(reinterpret_cast<char*>(&value), sizeof(T));
		IA[0] = value;
	}
	for (int i = 1, e = IA.size(); i != e; i++)
	{
		T value;
		infile.read(reinterpret_cast<char*>(&value), sizeof(T));
		IA[i] = (int)value + IA[i - 1];
	}

	if (IA[0] != 0)
	{
		cerr << "IA[0] must be 0" << endl;
		process->abort();
	}

	for (int i = 1, e = IA.size(); i != e; i++)
		if (IA[i] < IA[i - 1])
		{
			cerr << "IA[i] must be not less than IA[i - 1] - not true for IA[" << i << "] >= IA[" << (i - 1) <<
				"] : " << IA[i] << " < " << IA[i - 1] << endl;
			process->abort();
		}
	
	vector<int> JA;
	JA.resize(surplus_nonzeros);
	for (int i = 0, e = JA.size(); i != e; i++)
	{
		T value;
		infile.read(reinterpret_cast<char*>(&value), sizeof(T));
		JA[i] = (int)value;
	}

	for (int i = 0, e = JA.size(); i != e; i++)
	{
		if (JA[i] >= TotalDof)
		{
			cerr << "JA[i] must be within column index range - not true for JA[" << i << "] = " << JA[i] << endl;
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
		cerr << "State " << istate << " data is already loaded" << endl;
		process->abort();
	}

	bool compressed = isCompressed(filename);

	ifstream infile;
	if (compressed)
		infile.open(filename, ifstream::binary);
	else
		infile.open(filename, ios::in);

	if (!infile.is_open())
	{
		cerr << "Error opening file: " << filename << endl;
		process->abort();
	}

	if (compressed)
	{
		char format_marker[] = "          ";		
		infile.read(format_marker, strlen(format_marker));
		infile.read(reinterpret_cast<char*>(&dim), sizeof(int));
		infile.read(reinterpret_cast<char*>(&nno), sizeof(int)); 
		infile.read(reinterpret_cast<char*>(&TotalDof), sizeof(int));
		infile.read(reinterpret_cast<char*>(&Level), sizeof(int));
	}
	else
	{	
		infile >> dim;
		infile >> nno; 
		infile >> TotalDof;
		infile >> Level;
	}

	if (dim != params.nagents)
	{
		cerr << "File \"" << filename << "\" # of dimensions (" << dim << 
			") mismatches config (" << params.nagents << ")" << endl;
		process->abort();
	}

	// Pad all indexes to 4-element boundary.
	vdim = dim / AVX_VECTOR_SIZE;
	if (dim % AVX_VECTOR_SIZE) vdim++;
	int nsd = 2 * vdim * AVX_VECTOR_SIZE;

	Matrix<int> index;
	index.resize(nno, nsd);
	index.fill(0);

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
					int value; infile >> value;
					value = 2 << (value - 2);
					index(j, i) = value;
				}
			}
			for (int i = 0; i < dim; )
			{
				for (int v = 0; (v < AVX_VECTOR_SIZE) && (i < dim); v++, i++)
				{
					int value; infile >> value;
					value--;
					// Precompute "j" to merge two cases into one:
					// (((i) == 0) ? (1) : (1 - fabs((x) * (i) - (j)))).
					if (!index(j, i)) value = 0;
					index(j, i + vdim * AVX_VECTOR_SIZE) = value;
				}
			}
			for (int i = 0; i < TotalDof; i++)
			{
				double value; infile >> value;
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
	
	infile.close();

	const pair<int, int> zero = make_pair(0, 0);

	vdim *= AVX_VECTOR_SIZE;

	struct State
	{
		int nfreqs;
		vector<Index<uint16_t> >& xps;
		vector<uint32_t>& chains;
		
		State(vector<Index<uint16_t> >& xps_, vector<uint32_t>& chains_) : xps(xps_), chains(chains_) { }
	}
	state(xps[istate], chains[istate]);

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
				bool foundPosition = false;
				for (int irow = 0, nrows = indexes.size() / dim; irow < nrows; irow++)
				{
					Index<uint32_t>& index = indexes[irow * dim + j];
					if (index.isEmpty())
					{
						index.i = value.first;
						index.j = value.second;
						index.rowind() = i;

						foundPosition = true;
						break;
					}
				}
				if (!foundPosition)
				{
					// Add new free row.
					indexes.resize(indexes.size() + dim);

					Index<uint32_t>& index = indexes[indexes.size() - dim + j];

					index.i = value.first;
					index.j = value.second;
					index.rowind() = i;
				}
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

		for (int j = 0, iavx = 0, length = indexes.size() / dim / AVX_VECTOR_SIZE; j < dim; j++)
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

	Matrix<double> surplusOld = surplus[istate];
	Matrix<double>& surplusNew = surplus[istate];

	// Reorder surpluses.
	for (map<uint32_t, uint32_t>::iterator i = transMaps[0].begin(), e = transMaps[0].end(); i != e; i++)
	{
		int oldind = i->first;
		int newind = i->second;

		memcpy(&surplusNew(newind, 0), &(surplusOld(oldind, 0)), surplusNew.dimx() * sizeof(double));
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

	vector<vector<int, AlignedAllocator<int> > > temps(
		state.nfreqs, vector<int, AlignedAllocator<int> >(nnoAligned, -1));

	struct Map
	{
		map<Index<uint16_t>, uint32_t> xps;
	}
	map;
	
	// Loop through all frequences.
	for (int ifreq = 0, ixp = 0; ifreq < state.nfreqs; ifreq++)
	{
		struct Freq
		{
			const AVXIndexes& avxinds;
			vector<int, AlignedAllocator<int> >& temps;
			
			Freq(const AVXIndexes& avxinds_, vector<int, AlignedAllocator<int> >& temps_) :
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
		cout << map.xps.size() << " unique xp(s) to compute" << endl;
	
	// Add extra xp index denoting an empty frequency value.
	int32_t last = map.xps.size();
	map.xps[Index<uint16_t>(0, 0, 0)] = last;

	// Create all possible chains between frequencies.
	state.chains.resize(nno * state.nfreqs);
	for (int i = 0, ichain = 0; i < nno; i++)
	{
		int value = temps[0][i];

		if (value == -1)
			value = map.xps[Index<uint16_t>(0, 0, 0)];

		state.chains[ichain++] = (uint32_t)value;

		for (int ifreq = 1; ifreq < state.nfreqs; ifreq++)
		{
			int value = temps[ifreq][trans[ifreq][i]];

			if (value == -1)
				value = map.xps[Index<uint16_t>(0, 0, 0)];

			state.chains[ichain++] = (uint32_t)value;
		}
	}
	if (process->isMaster())
		cout << (state.chains.size() / state.nfreqs) << " chains of " <<
			state.nfreqs << " xp(s) to build" << endl;
	
	// Convert xps from map to vector.
	xps[istate].resize(map.xps.size());
	for (std::map<Index<uint16_t>, uint32_t>::const_iterator i = map.xps.begin(), e = map.xps.end(); i != e; i++)
		xps[istate][i->second] = i->first;
	
	nfreqs[istate] = state.nfreqs;
	
	loadedStates[istate] = true;
}

void Data::clear()
{
	fill(loadedStates.begin(), loadedStates.end(), false);
}

Data::Data(int nstates_) : nstates(nstates_)
{
	nfreqs.resize(nstates);
	xps.resize(nstates);
	chains.resize(nstates);
	surplus.resize(nstates);
	loadedStates.resize(nstates);
	fill(loadedStates.begin(), loadedStates.end(), false);
}

Data::~Data()
{
}

extern "C" Data* getData(int nstates)
{
	return new Data(nstates);
}

