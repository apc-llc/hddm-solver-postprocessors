#ifdef HAVE_AVX
#include <stdint.h>
#include <x86intrin.h>
#else
#include "LinearBasis.h"
#endif

void LinearBasis::CPU::FUNCNAME(
	const int dim, const int nno,
	const int Dof_choice, const double* x,
	const int* index, const double* surplus_t, double* value_)
{
	double value = 0.0;

	// Index arrays shall be padded to 4-element boundary to keep up
	// the alignment.
	int vdim = dim / 4;
	if (dim % 4) vdim++;
	vdim *= 4;
#ifdef HAVE_AVX
	const __m256d double4_0_0_0_0 = _mm256_setzero_pd();
	const __m256d double4_1_1_1_1 = _mm256_set1_pd(1.0);
	const __m256d sign_mask = _mm256_set1_pd(-0.);

	__m256d x4;
#if defined(DEFERRED)
	if (DIM <= 4)
		x4 = _mm256_loadu_pd(x);
#endif
	for (int i = 0; i < nno; i++)
	{
		int zero = 0;
		__m256d temp = double4_1_1_1_1;
		for (int j = 0; j < DIM; j += 4)
		{
#if defined(DEFERRED)
			if (DIM > 4)
#endif
			{
				__m256d x4 = _mm256_loadu_pd(x + j);
			}

			__m128i i4 = _mm_load_si128((const __m128i*)&index[i * 2 * vdim + j]);
			__m128i j4 = _mm_load_si128((const __m128i*)&index[i * 2 * vdim + j + vdim]);
			const __m256d xp = _mm256_sub_pd(double4_1_1_1_1, _mm256_andnot_pd(sign_mask,
				_mm256_sub_pd(_mm256_mul_pd(x4, _mm256_cvtepi32_pd(i4)), _mm256_cvtepi32_pd(j4))));
			const __m256d d = (__m256d)_mm256_cmp_ps((__m256)xp, (__m256)double4_0_0_0_0, _CMP_GT_OS);
			if (_mm256_movemask_pd(d) != 0xf)
			{
				zero = 1;
				break;
			}
			temp = _mm256_mul_pd(temp, xp);
		}
		if (zero) continue;
		const __m128d pairwise_sum = _mm_mul_pd(_mm256_castpd256_pd128(temp), _mm256_extractf128_pd(temp, 1));
		value += _mm_cvtsd_f64(_mm_mul_pd(pairwise_sum, (__m128d)_mm_movehl_ps((__m128)pairwise_sum, (__m128)pairwise_sum))) *
			surplus_t[Dof_choice * nno + i];
	}
#else
	for (int i = 0; i < nno; i++)
	{
		int zero = 0;
		double temp = 1.0;
		for (int j = 0; j < DIM; j++)
		{
			double xp = LinearBasis(x[j], index[i * 2 * vdim + j],
				index[i * 2 * vdim + j + vdim]);
			if (xp <= 0.0)
			{
				zero = 1;
				break;
			}
			temp *= xp;
		}
		if (zero) continue;
		value += temp * surplus_t[Dof_choice * nno + i];
	}
#endif
	*value_ = value;
}

