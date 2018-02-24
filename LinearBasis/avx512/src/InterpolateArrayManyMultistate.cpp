#include <omp.h>
#include <x86intrin.h>

#include "avx512/include/Data.h"
#include "avx512/include/Device.h"

using namespace NAMESPACE;
using namespace std;

#include "ScaledSurplus.h"

extern "C" void FUNCNAME(
	NAMESPACE::Device* device,
	const int dim, const int DofPerNode, const int count, const double* const* x_,
	const int* nfreqs_, const XPS* xps_, const Chains* chains_, const Matrix<double>* surplus_, double** value_)
{
	size_t szscratch = DOF_PER_NODE;
	if (DOF_PER_NODE % DOUBLE_VECTOR_SIZE)
		szscratch = (DOF_PER_NODE / DOUBLE_VECTOR_SIZE + 1) * DOUBLE_VECTOR_SIZE;

	int nthreadsMax = device->getThreadsCount();

	const __m512d zero = _mm512_setzero_pd();
	const __m512d one = _mm512_set1_pd(1.0);
	for (int many = 0; many < COUNT; many++)
	{
		Vector<double> vscratch(szscratch * nthreadsMax);
		double* scratch = vscratch.getData();

		const double* x = x_[many];
		const int& nfreqs = nfreqs_[many];
		const XPS& xps = xps_[many];
		const Chains& chains = chains_[many];
		const Matrix<double>& surplus = surplus_[many];
		double* value = value_[many];

		// Zero the values array.
		memset(value, 0, sizeof(double) * DOF_PER_NODE);

		int nno = surplus.dimy();

		int nthreads = min(nthreadsMax, nno);

		size_t szxpv64 = xps.size() / DOUBLE_VECTOR_SIZE;
		if (xps.size() % DOUBLE_VECTOR_SIZE) szxpv64++;
		vector<__m512d, AlignedAllocator<__m512d> > xpv64(szxpv64);

		// Loop to calculate all unique xp values.
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
			_mm512_store_pd((double*)&xpv64[i], _mm512_max_pd(zero, _mm512_sub_pd(one,
				_mm512_abs_pd(_mm512_fmadd_pd(x64, i32, j32)))));
		}

		// Loop to calculate scaled surplus product.
		double* xpv = (double*)&xpv64[0];
		ScaledSurplus result(DofPerNode, surplus, value);
		#pragma omp parallel num_threads(nthreads)
		{
			int tid = omp_get_thread_num();

			double* value_private = scratch + szscratch * tid;

			// Create empty private copies of result. Due to bug in Intel, we do not assign
			// value_private pointer to it here, and instead do in the reduction loop below.
			#pragma omp declare reduction(ScaledSurplusReduction: ScaledSurplus: omp_out += omp_in) \
				initializer(omp_priv = ScaledSurplus::priv(omp_orig))

			#pragma omp for reduction(ScaledSurplusReduction:result)
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

				result += ScaledSurplus(value_private, i, temp);
				continue;

			next :

				// Only assign private copy a data pointer.
				result += ScaledSurplus(value_private);
				continue;
			}
		}
	}
}

