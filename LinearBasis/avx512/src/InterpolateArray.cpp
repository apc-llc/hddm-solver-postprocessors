#include <omp.h>
#include <x86intrin.h>

#include "LinearBasis.h"
#include "avx512/include/Data.h"

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
	__m512d zero = _mm512_set1_pd(0.0);
	__m512d one = _mm512_set1_pd(1.0);
	vector<__m512d, AlignedAllocator<__m512d> > xpv64(xps.size() / DOUBLE_VECTOR_SIZE);
	#pragma omp parallel for
	for (int i = 0, e = xpv64.size(); i < e; i++)
	{
		// Load Index.index
		__m128i index = _mm_load_si128((const __m128i*)&xps[i]);
		const __m256i index32 = _mm256_cvtepu16_epi32(index);
		const __m512d x64 = _mm512_i32gather_pd(index32, x, 8);
		
		// Load Index.i
		index = _mm_load_si128((const __m128i*)&xps[i] + 1);
		const __m512d i32 = _mm512_cvtepi32_pd(_mm256_cvtepi8_epi32(index));

		// Load Index.j
		index = _mm_shuffle_epi32(index, _MM_SHUFFLE(1, 0, 3, 2));
		const __m512d j32 = _mm512_cvtepi32_pd(_mm256_cvtepi8_epi32(index));

		// Compute xpv[i]
		_mm512_store_pd((double*)&xpv64[i], _mm512_max_pd(zero,
			_mm512_sub_pd(one, (__m512d)_mm512_abs_epi64((__m512i)_mm512_fmadd_pd(x64, i32, j32)))));
	}

	size_t szscratch = DOF_PER_NODE;
	if (DOF_PER_NODE % DOUBLE_VECTOR_SIZE)
		szscratch = (DOF_PER_NODE / DOUBLE_VECTOR_SIZE + 1) * DOUBLE_VECTOR_SIZE;

	int nthreads = 0;

	#pragma omp parallel
	{
		#pragma omp master
		nthreads = omp_get_num_threads();
	}
	Vector<double> vscratch(szscratch * (nthreads - 1));
	double* scratch = vscratch.getData();

	// Loop to calculate scaled surplus product.
	double* xpv = (double*)&xpv64[0];
	#pragma omp parallel
	{
		int tid = omp_get_thread_num();

		double* value_private = (tid == 0) ? value : scratch + szscratch * (tid - 1);

		#pragma omp for
		for (int i = 0; i < nno; i++)
		{
			double temp = 1.0;
			for (int ichain = i * nfreqs, ifreq = 0; ifreq < nfreqs; ifreq++)
			{
				// Early exit for shorter chains.
				const auto& idx = chains[ichain + ifreq];
				if (!idx) break;

				temp *= xpv[idx];
				if (!temp) goto next;
			}

			{
				const __m512d temp64 = _mm512_set1_pd(temp);

				for (int Dof_choice = 0; Dof_choice < DOF_PER_NODE; Dof_choice += DOUBLE_VECTOR_SIZE)
				{
					const __m512d surplus64 = _mm512_load_pd(&surplus(i, Dof_choice));
					__m512d value64 = _mm512_load_pd(&value_private[Dof_choice]);
					value64 = _mm512_fmadd_pd(temp64, surplus64, value64);
					_mm512_store_pd(&value_private[Dof_choice], value64);
				}
			}
		
		next :

			continue;
		}

		for (int shift = 1; shift < nthreads; shift <<= 1)
		{
			if (tid % (shift << 1) == 0)
			{
				double* value_private2 = scratch + szscratch * (tid - 1 + shift);
				for (int Dof_choice = 0; Dof_choice < DOF_PER_NODE; Dof_choice += DOUBLE_VECTOR_SIZE)
				{
					__m512d v = _mm512_load_pd(&value_private[Dof_choice]);
					__m512d v2 = _mm512_load_pd(&value_private2[Dof_choice]);
					_mm512_store_pd((__m512d*)&value_private[Dof_choice], _mm512_add_pd(v, v2));
				}
			}
			#pragma omp barrier
		}
	}
}

