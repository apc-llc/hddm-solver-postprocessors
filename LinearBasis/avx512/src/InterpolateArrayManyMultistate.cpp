#include <omp.h>
#include <x86intrin.h>

#include "LinearBasis.h"
#include "avx512/include/Data.h"
#include "avx512/include/Device.h"

using namespace NAMESPACE;
using namespace std;

class Device;

extern "C" void FUNCNAME(
	NAMESPACE::Device* device,
	const int dim, const int DofPerNode, const int count, const double* const* x_,
	const int* nfreqs_, const XPS* xps_, const Chains* chains_, const Matrix<double>* surplus_, double** value_)
{
	size_t szscratch = DOF_PER_NODE;
	if (DOF_PER_NODE % DOUBLE_VECTOR_SIZE)
		szscratch = (DOF_PER_NODE / DOUBLE_VECTOR_SIZE + 1) * DOUBLE_VECTOR_SIZE;

	int nthreads = device->getThreadsCount();

	Vector<double> vscratch(szscratch * (nthreads - 1));
	double* scratch = vscratch.getData();

	const __m512d zero = _mm512_setzero_pd();
        const __m512d one = _mm512_set1_pd(1.0);
	for (int many = 0; many < COUNT; many++)
	{
		const double* x = x_[many];
		const int& nfreqs = nfreqs_[many];
		const XPS& xps = xps_[many];
		const Chains& chains = chains_[many];
		const Matrix<double>& surplus = surplus_[many];
		double* value = value_[many];

		int nno = surplus.dimy();

		// Loop to calculate all unique xp values.
		vector<__m512d, AlignedAllocator<__m512d> > xpv64(xps.size() / DOUBLE_VECTOR_SIZE);
		#pragma omp parallel for
		for (int i = 0, e = xpv64.size(); i < e; i++)
		{
			// Load Index.index
			__m128i index = _mm_load_si128((const __m128i*)&xps[i]);
			const __m256i index32 = _mm256_cvtepu16_epi32(index);
			__m512d x64 = _mm512_i32gather_pd(index32, x, 8);
		
			// Load Index.i
			index = _mm_load_si128((const __m128i*)&xps[i] + 1);
			const __m512d i32 = _mm512_cvtepi32_pd(_mm256_cvtepi8_epi32(index));

			// Load Index.j
			index = _mm_shuffle_epi32(index, _MM_SHUFFLE(1, 0, 3, 2));
			const __m512d j32 = _mm512_cvtepi32_pd(_mm256_cvtepi8_epi32(index));

			// Compute xpv[i]
#if 0
			// Alas, there is no _mm512_andnot_pd on avx512f/pf/er/cd, so we abs in two 256-wide parts.
			x64 = _mm512_fmadd_pd(x64, i32, j32);
			x64 = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_andnot_pd(sign_mask, _mm512_castpd512_pd256(x64))),
				_mm256_andnot_pd(sign_mask, _mm512_extractf64x4_pd(x64, 1)), 1);
			_mm512_store_pd((double*)&xpv64[i], _mm512_max_pd(zero, _mm512_sub_pd(one, x64)));
#else
			_mm512_store_pd((double*)&xpv64[i], _mm512_max_pd(zero, _mm512_sub_pd(one,
				(__m512d)_mm512_abs_epi64((__m512i)_mm512_fmadd_pd(x64, i32, j32)))));
#endif
		}

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
						__m512d surplus64 = _mm512_load_pd(&surplus(i, Dof_choice)/*,
							_MM_UPCONV_PD_NONE, _MM_BROADCAST64_NONE, _MM_HINT_NT*/);
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
						__m512d v2 = _mm512_load_pd(&value_private2[Dof_choice]/*,
							_MM_UPCONV_PD_NONE, _MM_BROADCAST64_NONE, _MM_HINT_NT*/);
						_mm512_store_pd((__m512d*)&value_private[Dof_choice], _mm512_add_pd(v, v2));
					}
				}
				#pragma omp barrier
			}
		}
	}
}

