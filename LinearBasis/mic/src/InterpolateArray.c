#ifdef HAVE_AVX
#include <assert.h>
#include <stdint.h>
#include <immintrin.h>
#else
#include "LinearBasis.h"
#endif

#ifdef HAVE_AVX

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

#endif // HAVE_AVX

// Adopted from C++ vector class library by Agner Fog
static inline __m512d _mm512_abs_pd(const __m512d x)
{
    return _mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(x), sign_mask));
}

static void interpolate(
	const int dim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const double* x,
	const double* index, const double* surplus_t, double* value)
{
#ifdef HAVE_AVX
	assert(((size_t)x % (AVX_VECTOR_SIZE * sizeof(double)) == 0) && "x vector must be sufficiently memory-aligned");
	assert(((size_t)index % (AVX_VECTOR_SIZE * sizeof(double)) == 0) && "index vector must be sufficiently memory-aligned");
	assert(((size_t)surplus_t % (AVX_VECTOR_SIZE * sizeof(double)) == 0) && "surplus_t vector must be sufficiently memory-aligned");
#endif

	__m512d x8;

	// Index arrays shall be padded to AVX_VECTOR_SIZE-element
	// boundary to keep up the required alignment.
	int vdim = dim / AVX_VECTOR_SIZE;
	if (dim % AVX_VECTOR_SIZE) vdim++;
	vdim *= AVX_VECTOR_SIZE;

	for (int b = Dof_choice_start, Dof_choice = b, e = Dof_choice_end; Dof_choice <= e; Dof_choice++)
		value[Dof_choice - b] = 0;

#ifdef HAVE_AVX
	#pragma omp parallel for
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
		double temps = _mm512_reduce_mul_pd(temp);
		#pragma omp critical
		{
			for (int b = Dof_choice_start, Dof_choice = b, e = Dof_choice_end; Dof_choice <= e; Dof_choice++)
				value[Dof_choice - b] += temps * surplus_t[Dof_choice * nno + i];
		}
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
			value[Dof_choice - b] += temp * surplus_t[Dof_choice * nno + i];
	}
#endif
}

void FUNCNAME(void* arg)
{
	typedef struct
	{
		int dim, nno, Dof_choice_start, Dof_choice_end;
		double *x, *surplus_t, *value, *index;
	}
	Args;
	
	Args* args = (Args*)arg;

	interpolate(args->dim, args->nno, args->Dof_choice_start, args->Dof_choice_end, args->x,
		args->index, args->surplus_t, args->value);
}

