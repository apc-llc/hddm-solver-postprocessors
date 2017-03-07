#ifdef HAVE_AVX
#include <assert.h>
#include <stdint.h>
#include <x86intrin.h>
#include <utility> // pair
#endif

#include "LinearBasis.h"

#include "Data.h"

using namespace cpu;
using namespace std;

class Device;

static bool initialized = false;

static vector<int> ind_I;
static vector<int> ind_J;
static vector<int> rowinds;

extern "C" void FUNCNAME(
	Device* device,
	const int dim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const double* x,
	const Matrix<int>* index_, const Matrix<double>* surplus_, double* value)
{
#ifdef HAVE_AVX
	assert(((size_t)x % (AVX_VECTOR_SIZE * sizeof(double)) == 0) && "x vector must be sufficiently memory-aligned");
#endif

	const Matrix<int>& index = *index_;
	const Matrix<double>& surplus = *surplus_;

	// Index arrays shall be padded to AVX_VECTOR_SIZE-element
	// boundary to keep up the required alignment.
	int vdim = dim / AVX_VECTOR_SIZE;
	if (dim % AVX_VECTOR_SIZE) vdim++;
	vdim *= AVX_VECTOR_SIZE;

#if 1
	if (!initialized)
	{
		pair<int, int> zero = make_pair(0, 0);

		ind_I.resize(vdim);
		ind_J.resize(vdim);
		rowinds.resize(vdim);

		// Convert (i, I) indexes matrix to sparse format.
		for (int i = 0; i < nno; i++)
			for (int j = 0; j < dim; j++)
			{
				// Get pair.
				pair<int, int> value = make_pair(index(i, j), index(i, j + vdim));
	
				// If both indexes are zeros, do nothing.
				if (value == zero)
					continue;
	
				// Find free position for non-zero pair.
				bool foundPosition = false;
				for (int irow = 0, nrows = rowinds.size() / vdim; irow < nrows; irow++)
				{
					int& positionI = ind_I[irow * vdim + j];
					int& positionJ = ind_J[irow * vdim + j];
					if (make_pair(positionI, positionJ) == zero)
					{
						// Check no any "i" row elements on this "irow" yet.
						bool busyRow = false;
						for (int jrow = 0; jrow < dim; jrow++)
						{
							int& rowind = rowinds[irow * vdim + jrow];

							if (rowind == i) busyRow = true;
						}
						if (busyRow) continue;

						int& rowind = rowinds[irow * vdim + j];
			
						positionI = value.first;
						positionJ = value.second;
						rowind = i;

						foundPosition = true;
						break;
					}
				}
				if (!foundPosition)
				{
					// Add new free row.
					ind_I.resize(ind_I.size() + vdim);
					ind_J.resize(ind_J.size() + vdim);
					rowinds.resize(rowinds.size() + vdim);

					int& positionI = ind_I[ind_I.size() - vdim + j];
					int& positionJ = ind_J[ind_J.size() - vdim + j];

					int& rowind = rowinds[rowinds.size() - vdim + j];
		
					positionI = value.first;
					positionJ = value.second;
					rowind = i;				
				}
			}
		
		initialized = true;
	}

	// Loop to calculate temps.
	// Note temps vector should not be too large to keep up the caching.
	// So, we might need to roll the nno loop as a set of small loops.
	vector<double> temps(nno, 1.0);
	for (int i = 0, e = rowinds.size() / vdim; i < e; i++)
	{
		for (int j = 0; j < dim; j++)
		{
			int& ind_i = ind_I[i * vdim + j];
			int& ind_j = ind_J[i * vdim + j];
			int& rowind = rowinds[i * vdim + j];

			// XXX LinearBasis can be done in AVX.
			double xp = LinearBasis(x[j], ind_i, ind_j);
			xp = fmax(0.0, xp);
		
			// XXX This can be done scalar only.
			temps[rowind] *= xp;
		}			
	}
	
	// Loop to calculate values.
	for (int i = 0; i < nno; i++)
	{
		// XXX This can be done in AVX.
		for (int b = Dof_choice_start, Dof_choice = b, e = Dof_choice_end; Dof_choice <= e; Dof_choice++)
			value[Dof_choice - b] += temps[i] * surplus(i, Dof_choice);
	}		
#else

	for (int i = 0; i < nno; i++)
	{
		double temp = 1.0;
		for (int j = 0; j < DIM; j++)
		{
			double xp = LinearBasis(x[j], index(i, j), index(i, j + vdim));
			if (xp <= 0.0)
				goto zero;
			temp *= xp;
		}
		for (int b = Dof_choice_start, Dof_choice = b, e = Dof_choice_end; Dof_choice <= e; Dof_choice++)
			value[Dof_choice - b] += temp * surplus(i, Dof_choice);

		zero: continue;
	}
#endif
}

