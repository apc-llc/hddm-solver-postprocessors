#include <x86intrin.h>

#include "LinearBasis.h"
#include "Data.h"

using namespace NAMESPACE;
using namespace std;

class Device;

extern "C" void FUNCNAME(
	Device* device,
	const int dim, const int nno, const int DofPerNode, const double* x,
	const int nfreqs, const XPS* xps_, const Chains* chains_, const Matrix<double>* surplus_, double* value)
{
	const XPS& xps = *xps_;
	const Chains& chains = *chains_;
	const Matrix<double>& surplus = *surplus_;

	// Loop to calculate all unique xp values.
	vector<double> xpv(xps.size(), 1.0);
	for (int i = 0, e = xpv.size(); i < e; i++)
	{
		const Index<uint16_t>& index = xps[i];
		const uint32_t& j = index.index;
		double xp = LinearBasis(x[j], index.i, index.j);
		xpv[i] = fmax(0.0, xp);
	}

	// Zero the values array.
	memset(value, 0, sizeof(double) * DOF_PER_NODE);

	// Loop to calculate scaled surplus product.
	for (int i = 0, ichain = 0; i < NNO; i++, ichain += nfreqs)
	{
		double temp = 1.0;
		for (int ifreq = 0; ifreq < nfreqs; ifreq++)
		{
			// Early exit for shorter chains.
			int32_t idx = chains[ichain + ifreq];
			if (!idx) break;

			temp *= xpv[idx];
			if (!temp) goto next;
		}

		{
			const __m512d temp64 = _mm512_set1_pd(temp);

			for (int Dof_choice = 0; Dof_choice <= DOF_PER_NODE;
				Dof_choice += sizeof(temp64) / sizeof(double))
			{
				const __m512d surplus64 = _mm512_load_pd(&surplus(i, Dof_choice));
				__m512d value64 = _mm512_load_pd(&value[Dof_choice]);
				value64 = _mm512_fmadd_pd(temp64, surplus64, value64);
				_mm512_store_pd(&value[Dof_choice], value64);
			}
		}
		
	next :

		continue;
	}
}

