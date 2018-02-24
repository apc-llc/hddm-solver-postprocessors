#include <omp.h>
#include <x86intrin.h>

#include "avx512/include/Data.h"
#include "avx512/include/Device.h"

using namespace NAMESPACE;
using namespace std;

#include "ScaledSurplus.h"

extern "C" void FUNCNAME(
	Device* device,
	const int dim, const int DofPerNode, const double* x,
	const int nfreqs, const XPS* xps_, const Chains* chains_, const Matrix<double>* surplus_, double* value)
{
	size_t szscratch = DOF_PER_NODE;
	if (DOF_PER_NODE % DOUBLE_VECTOR_SIZE)
		szscratch = (DOF_PER_NODE / DOUBLE_VECTOR_SIZE + 1) * DOUBLE_VECTOR_SIZE;

	int nthreadsMax = device->getThreadsCount();

	Vector<double> vscratch(szscratch * nthreadsMax);
	double* scratch = vscratch.getData();

	const XPS& xps = *xps_;
	const Chains& chains = *chains_;
	const Matrix<double>& surplus = *surplus_;

	int nno = surplus.dimy();

	int nthreads = min(nthreadsMax, nno);

	// Loop to calculate all unique xp values.
	const __m512d zero = _mm512_set1_pd(0.0);
	const __m512d one = _mm512_set1_pd(1.0);
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

