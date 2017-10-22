#include <x86intrin.h>

#include "LinearBasis.h"
#include "avx2/include/Data.h"

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
	__m256d zero = _mm256_set1_pd(0.0);
	__m256d one = _mm256_set1_pd(1.0);
	__m256d sign_mask = _mm256_set1_pd(-0.);
	vector<__m256d, AlignedAllocator<__m256d> > xpv64(xps.size());
	for (int i = 0, e = xpv64.size(); i < e; i++)
	{
		// Load Index.index
		__m128i index = _mm_load_si128((const __m128i*)&xps[i]);
		const __m128i index32 = _mm_cvtepu16_epi32(index);
		const __m256d x64 = _mm256_i32gather_pd(x, index32, 8);
		
		// Load Index.i
		index = _mm_shuffle_epi32(index, _MM_SHUFFLE(1, 0, 3, 2));
		const __m256d i32 = _mm256_cvtepi32_pd(_mm_cvtepi8_epi32(index));

		// Load Index.j
		index = _mm_shuffle_epi32(index, _MM_SHUFFLE(3, 2, 0, 1));
		const __m256d j32 = _mm256_cvtepi32_pd(_mm_cvtepi8_epi32(index));

		// Compute xpv[i]
		_mm256_store_pd((double*)&xpv64[i], _mm256_max_pd(zero,
			_mm256_sub_pd(one, _mm256_andnot_pd(sign_mask, _mm256_fmadd_pd(x64, i32, j32)))));
	}

	// Zero the values array.
	memset(value, 0, sizeof(double) * DOF_PER_NODE);

	// Loop to calculate scaled surplus product.
	double* xpv = (double*)&xpv64[0];
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

		{
			const __m256d temp64 = _mm256_set1_pd(temp);

			for (int Dof_choice = 0; Dof_choice < DOF_PER_NODE;
				Dof_choice += sizeof(temp64) / sizeof(double))
			{
				const __m256d surplus64 = _mm256_load_pd(&surplus(i, Dof_choice));
				__m256d value64 = _mm256_load_pd(&value[Dof_choice]);		
				value64 = _mm256_fmadd_pd(temp64, surplus64, value64);
				_mm256_store_pd(&value[Dof_choice], value64);
			}
		}
		
	next :

		continue;
	}
}

