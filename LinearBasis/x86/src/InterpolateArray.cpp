#include "LinearBasis.h"
#include "x86/include/Data.h"

using namespace NAMESPACE;
using namespace std;

class Device;

extern "C" void FUNCNAME(
	Device* device,
	const int dim, const int DofPerNode, const double* x,
	const int nfreqs, const XPS* xps_, const Chains* chains_, const Matrix<double>* surplus_, double* value)
{
	const XPS& xps = *xps_;
	const Chains& chains = *chains_;
	const Matrix<double>& surplus = *surplus_;

	int nno = surplus.dimy();

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
	for (int i = 0, ichain = 0; i < nno; i++, ichain += nfreqs)
	{
		double temp = 1.0;
		for (int ifreq = 0; ifreq < nfreqs; ifreq++)
		{
			// Early exit for shorter chains.
			const auto& idx = chains[ichain + ifreq];
			if (!idx) break;

			temp *= xpv[idx];
			if (!temp) goto next;
		}

		for (int Dof_choice = 0; Dof_choice < DOF_PER_NODE; Dof_choice++)
			value[Dof_choice] += temp * surplus(i, Dof_choice);
		
	next :

		continue;
	}
}

