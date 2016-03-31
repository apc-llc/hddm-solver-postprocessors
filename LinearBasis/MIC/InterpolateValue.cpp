#ifdef HAVE_MIC
#include <stdint.h>
#include <immintrin.h>
#else
#include "LinearBasis.h"
#endif

#ifdef HAVE_MIC

// Number of double precision elements in used AVX vector
#define AVX_VECTOR_SIZE 8

static __m512i sign_mask;
static __m512d double8_0_0_0_0_0_0_0_0;
static __m512d double8_1_1_1_1_1_1_1_1;

__attribute__((constructor)) void init_consts()
{
	sign_mask = _mm512_set_epi64(
		0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF,
		0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF);

	double8_0_0_0_0_0_0_0_0 = _mm512_setzero_pd();

	double8_1_1_1_1_1_1_1_1 = _mm512_set_pd(
		1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
}

// Adopted from C++ vector class library by Agner Fog
static inline __m512d _mm512_abs_pd(const __m512d x)
{
    return _mm512_castsi512_pd(_mm512_castpd_si512(x) & sign_mask);
}

#endif

#define FUNCNAME(mode) LinearBasis_MIC_##mode##_InterpolateValue 

void FUNCNAME(MODE)(
	const int dim, const int nno,
	const int Dof_choice, const double* x,
	const int* index, const double* surplus_t, double* value_)
{
	double value = 0.0;

	// Index arrays shall be padded to AVX_VECTOR_SIZE-element
	// boundary to keep up the required alignment.
	int vdim = dim / AVX_VECTOR_SIZE;
	if (dim % AVX_VECTOR_SIZE) vdim++;
	vdim *= AVX_VECTOR_SIZE;

#ifdef HAVE_MIC
	volatile __m512d x8;
#if defined(DEFERRED)
	if (DIM <= AVX_VECTOR_SIZE)
	{
		x8 = _mm512_load_pd(x);
		//x8 = _mm512_extloadunpacklo_pd(x8, x, _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
		//x8 = _mm512_extloadunpackhi_pd(x8, (uint8_t*)x + 64, _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	}
#endif
	for (int i = 0; i < nno; i++)
	{
		int zero = 0;
		volatile __m512d temp = double8_1_1_1_1_1_1_1_1;
		for (int j = 0; j < DIM; j += AVX_VECTOR_SIZE)
		{
#if defined(DEFERRED)
			if (DIM > AVX_VECTOR_SIZE)
#endif
			{
				__m512d x8 = _mm512_load_pd(x + j);
				//x8 = _mm512_extloadunpacklo_pd(x8, x + j, _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
				//x8 = _mm512_extloadunpackhi_pd(x8, (uint8_t*)(x + j) + 64, _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
			}

			// Read integer indexes, which are already converted to double,
			// because k1om can't load __m256i and convert anyway.
			volatile __m512d i8 = _mm512_load_pd((void*)&index[i * 2 * vdim + j]);
			volatile __m512d j8 = _mm512_load_pd((void*)&index[i * 2 * vdim + j + vdim]);

			volatile const __m512d xp = _mm512_sub_pd(double8_1_1_1_1_1_1_1_1,
				_mm512_abs_pd(_mm512_fmsub_pd (x8, i8, j8)));
			volatile __mmask8 d = _mm512_cmp_pd_mask(xp, double8_0_0_0_0_0_0_0_0, _MM_CMPINT_GT);
			if (d != 0xff)
			{
				zero = 1;
				break;
			}
			temp = _mm512_mul_pd(temp, xp);
		}
		if (zero) continue;

		//value += _mm512_reduce_mul_pd(temp) * surplus_t[Dof_choice * nno + i];
		double reduction = _mm512_reduce_mul_pd(temp);
		__asm__ (
			  "fldl %[temp]\n\t"
			  "fmull %[surplus]\n\t"
			  "faddl %[value]\n\t"
			  "fstpl %[value]\n\t"
			: [value] "+m"(value)
			: [temp] "m"(reduction), [surplus] "m"(surplus_t[Dof_choice * nno + i]));
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

