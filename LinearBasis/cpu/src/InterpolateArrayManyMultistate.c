#ifdef HAVE_AVX
#include <assert.h>
#include <stdint.h>
#include <x86intrin.h>
#else
#include "LinearBasis.h"
#endif

void FUNCNAME(
	const int dim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const int count, const double* const* x_,
	const int* const* index_, const double* const* surplus_, double** value_)
{
	// Index arrays shall be padded to AVX_VECTOR_SIZE-element
	// boundary to keep up the required alignment.
	int vdim = dim / AVX_VECTOR_SIZE;
	if (dim % AVX_VECTOR_SIZE) vdim++;
	vdim *= AVX_VECTOR_SIZE;

	const int TotalDof = Dof_choice_end - Dof_choice_start + 1;

	for (int many = 0; many < count; many++)
	{
		const double* x = x_[many];
		const int* index = index_[many];
		const double* surplus = surplus_[many];
		double* value = value_[many];

#ifdef HAVE_AVX
		assert(((size_t)x % (AVX_VECTOR_SIZE * sizeof(double)) == 0) && "x vector must be sufficiently memory-aligned");
		assert(((size_t)index % (AVX_VECTOR_SIZE * sizeof(int)) == 0) && "index vector must be sufficiently memory-aligned");
		assert(((size_t)surplus % (AVX_VECTOR_SIZE * sizeof(double)) == 0) && "surplus vector must be sufficiently memory-aligned");
#endif

		for (int b = Dof_choice_start, Dof_choice = b, e = Dof_choice_end; Dof_choice <= e; Dof_choice++)
			value[Dof_choice - b] = 0;
#ifdef HAVE_AVX
		const __m256d double4_0_0_0_0 = _mm256_setzero_pd();
		const __m256d double4_1_1_1_1 = _mm256_set1_pd(1.0);
		const __m256d sign_mask = _mm256_set1_pd(-0.);

		__m256d x4;
#if defined(DEFERRED)
		if (DIM <= AVX_VECTOR_SIZE)
			x4 = _mm256_load_pd(x);
#endif
		for (int i = 0; i < nno; i++)
		{
			int zero = 0;
			__m256d temp = double4_1_1_1_1;
			for (int j = 0; j < DIM; j += 4)
			{
#if defined(DEFERRED)
				if (DIM > AVX_VECTOR_SIZE)
#endif
				{
					x4 = _mm256_load_pd(x + j);
				}

				__m128i i4 = _mm_load_si128((const __m128i*)&index[i * 2 * vdim + j]);
				__m128i j4 = _mm_load_si128((const __m128i*)&index[i * 2 * vdim + j + vdim]);
				const __m256d xp = _mm256_sub_pd(double4_1_1_1_1, _mm256_andnot_pd(sign_mask,
					_mm256_sub_pd(_mm256_mul_pd(x4, _mm256_cvtepi32_pd(i4)), _mm256_cvtepi32_pd(j4))));
				const __m256d d = _mm256_cmp_pd(xp, double4_0_0_0_0, _CMP_GT_OQ);
				if (_mm256_movemask_pd(d) != (int)0xf)
				{
					zero = 1;
					break;
				}
				temp = _mm256_mul_pd(temp, xp);
			}
			if (zero) continue;
			const __m128d pairwise_sum = _mm_mul_pd(_mm256_castpd256_pd128(temp), _mm256_extractf128_pd(temp, 1));
			const double temps = _mm_cvtsd_f64(_mm_mul_pd(pairwise_sum,
				(__m128d)_mm_movehl_ps((__m128)pairwise_sum, (__m128)pairwise_sum)));
			for (int b = Dof_choice_start, Dof_choice = b, e = Dof_choice_end; Dof_choice <= e; Dof_choice++)
				value[Dof_choice - b] += temps * surplus[i * TotalDof + Dof_choice];
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
			for (int b = Dof_choice_start, Dof_choice = b, e = Dof_choice_end; Dof_choice <= e; Dof_choice++)
				value[Dof_choice - b] += temp * surplus[i * TotalDof + Dof_choice];
		}
#endif
	}
}

