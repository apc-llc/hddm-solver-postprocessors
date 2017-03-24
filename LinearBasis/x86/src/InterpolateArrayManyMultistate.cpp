#include "LinearBasis.h"
#include "Data.h"

using namespace NAMESPACE;
using namespace std;

class Device;

extern "C" void FUNCNAME(
	Device* device,
	const int dim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const int count, const double* const* x_,
	const int* nfreqs_, const XPS* xps_, const Chains* chains_, const Matrix<double>* surplus_, double** value_)
{
	for (int many = 0; many < COUNT; many++)
	{
		const double* x = x_[many];
		const int& nfreqs = nfreqs_[many];
		const XPS& xps = xps_[many];
		const Chains& chains = chains_[many];
		const Matrix<double>& surplus = surplus_[many];
		double* value = value_[many];

		for (int b = DOF_CHOICE_START, Dof_choice = b, e = DOF_CHOICE_END; Dof_choice <= e; Dof_choice++)
			value[Dof_choice - b] = 0;

		// Loop through all frequences.
		vector<double> temps(NNO, 1.0);
		for (int ifreq = 0; ifreq < nfreqs; ifreq++)
		{
			const vector<Index<uint16_t> >& xpsFreq = xps[ifreq];
			
			vector<double> xpv(xpsFreq.size(), 1.0);
			for (int i = 0, e = xpv.size(); i < e; i++)
			{
				const Index<uint16_t>& index = xpsFreq[i];
				const uint32_t& j = index.index;
				double xp = LinearBasis(x[j], index.i, index.j);
				xpv[i] = fmax(0.0, xp);
			}

			for (int i = 0; i < NNO; i++)
				temps[i] *= xpv[chains[ifreq * nno + i]];
		}

		// Loop to calculate values.
		for (int i = 0; i < NNO; i++)
		{
			double temp = temps[i];

			if (!temp) continue;

			for (int b = DOF_CHOICE_START, Dof_choice = b, e = DOF_CHOICE_END; Dof_choice <= e; Dof_choice++)
			{
				double t = temp * surplus(i, Dof_choice);
				value[Dof_choice - b] += t;
			}
		}
	}

	/*for (int many = 0; many < COUNT; many++)
	{
		const double* x = x_[many];
		const AVXIndexMatrix& avxinds = avxinds_[many];
		const TransMatrix& trans = trans_[many];
		const Matrix<double>& surplus = surplus_[many];
		double* value = value_[many];

		for (int b = DOF_CHOICE_START, Dof_choice = b, e = DOF_CHOICE_END; Dof_choice <= e; Dof_choice++)
			value[Dof_choice - b] = 0;

		int nfreqs = avxinds.size();
		int nnoAligned = nno;
		if (nno % AVX_VECTOR_SIZE)
			nnoAligned += AVX_VECTOR_SIZE - nno % AVX_VECTOR_SIZE;

		// One extra vector size for *hi part in AVX code below.
		nnoAligned += AVX_VECTOR_SIZE;

		vector<vector<double, AlignedAllocator<double> > > temps_(
			nfreqs, vector<double, AlignedAllocator<double> >(nnoAligned, 1.0));

		// Loop through all frequences.
		for (int ifreq = 0; ifreq < nfreqs; ifreq++)
		{
			const AVXIndexes& avxindsFreq = avxinds[ifreq];
			vector<double, AlignedAllocator<double> >& temps = temps_[ifreq];

			// Loop to calculate temps.
			for (int j = 0, itemp = 0; j < DIM; j++)
			{
				double xx = x[j];
		
				for (int i = 0, e = avxindsFreq.getLength(j); i < e; i++)
				{
					const AVXIndex& index = avxindsFreq(i, j);

					for (int k = 0; k < AVX_VECTOR_SIZE; k++, itemp++)
					{
						const uint8_t& ind_i = index.i[k];
						const uint8_t& ind_j = index.j[k];

						double xp = LinearBasis(xx, ind_i, ind_j);

						xp = fmax(0.0, xp);
			
						temps[itemp] = xp;
					}
				}			

				if (avxindsFreq.getLength(j))
				{
					const AVXIndex& index = avxindsFreq(avxindsFreq.getLength(j) - 1, j);
					for (int k = AVX_VECTOR_SIZE - 1; k >= 0; k--)
						if (index.isEmpty(k)) itemp--;
				}
			}
		}

		// Join temps from all frequencies.
		vector<double, AlignedAllocator<double> >& temps = temps_[0];
		for (int i = 0; i < NNO; i++)
			for (int ifreq = 1; ifreq < nfreqs; ifreq++)
				temps[i] *= temps_[ifreq][trans[ifreq][i]];

		// Loop to calculate values.
		for (int i = 0; i < NNO; i++)
		{
			double temp = temps[i];

			if (!temp) continue;

			for (int b = DOF_CHOICE_START, Dof_choice = b, e = DOF_CHOICE_END; Dof_choice <= e; Dof_choice++)
				value[Dof_choice - b] += temp * surplus(i, Dof_choice);
		}
	}*/
}

