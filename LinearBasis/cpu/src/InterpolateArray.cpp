#ifdef HAVE_AVX
#include <assert.h>
#include <stdint.h>
#include <x86intrin.h>
#endif

#include <algorithm> // min & max
#include <mutex>
#include <utility> // pair

#include "LinearBasis.h"

#include "Data.h"

using namespace cpu;
using namespace std;

class Device;

extern "C" void FUNCNAME(
	Device* device,
	const int dim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const double* x,
	const AVXIndexMatrix* avxinds_, const Matrix<double>* surplus_, double* value)
{
#ifdef HAVE_AVX
	assert(((size_t)x % (AVX_VECTOR_SIZE * sizeof(double)) == 0) && "x vector must be sufficiently memory-aligned");
#endif

	const AVXIndexMatrix& avxinds = *avxinds_;
	const Matrix<double>& surplus = *surplus_;

	// Index arrays shall be padded to AVX_VECTOR_SIZE-element
	// boundary to keep up the required alignment.
	int vdim = dim / AVX_VECTOR_SIZE;
	if (dim % AVX_VECTOR_SIZE) vdim++;
	int vdim8 = vdim;
	vdim *= AVX_VECTOR_SIZE;

#ifdef HAVE_AVX
	const __m256d double4_0_0_0_0 = _mm256_setzero_pd();
	const __m256d double4_1_1_1_1 = _mm256_set1_pd(1.0);
	const __m256d sign_mask = _mm256_set1_pd(-0.);
	const __m128i int4_0_0_0_0 = _mm_setzero_si128();

	// Loop to calculate temps.
	// Note temps vector should not be too large to keep up the caching.
	vector<double, AlignedAllocator<double> > temps(nno, 1.0);
	for (int i = 0, e = avxinds.size() / VDIM8; i < e; i++)
	{
		for (int j = 0; j < VDIM8; j++)
		{
			const AVXIndex& index = avxinds[i * VDIM8 + j];

			const __m128i ij8 = _mm_load_si128(reinterpret_cast<const __m128i*>(&index));
			const __m128i i16 = _mm_unpacklo_epi8(ij8, int4_0_0_0_0);
			const __m128i j16 = _mm_unpackhi_epi8(ij8, int4_0_0_0_0);

			const __m128i i32lo = _mm_unpacklo_epi16(i16, int4_0_0_0_0);
			const __m128i i32hi = _mm_unpackhi_epi16(i16, int4_0_0_0_0);

			const __m128i j32lo = _mm_unpacklo_epi16(j16, int4_0_0_0_0);
			const __m128i j32hi = _mm_unpackhi_epi16(j16, int4_0_0_0_0);

			const __m256d x64lo = _mm256_load_pd(&x[j * AVX_VECTOR_SIZE]);

			__m256d xp64lo = _mm256_sub_pd(double4_1_1_1_1, _mm256_andnot_pd(sign_mask,
				_mm256_sub_pd(_mm256_mul_pd(x64lo, _mm256_cvtepi32_pd(i32lo)), _mm256_cvtepi32_pd(j32lo))));

			const __m256d mask64lo = _mm256_cmp_pd(xp64lo, double4_0_0_0_0, _CMP_GT_OQ);
			xp64lo = _mm256_blendv_pd(double4_0_0_0_0, xp64lo, mask64lo);

			const __m256d x64hi = _mm256_load_pd(&x[j * AVX_VECTOR_SIZE + sizeof(x64lo) / sizeof(double)]);

			__m256d xp64hi = _mm256_sub_pd(double4_1_1_1_1, _mm256_andnot_pd(sign_mask,
				_mm256_sub_pd(_mm256_mul_pd(x64hi, _mm256_cvtepi32_pd(i32hi)), _mm256_cvtepi32_pd(j32hi))));

			const __m256d mask64hi = _mm256_cmp_pd(xp64hi, double4_0_0_0_0, _CMP_GT_OQ);
			xp64hi = _mm256_blendv_pd(double4_0_0_0_0, xp64hi, mask64hi);

			double xp[AVX_VECTOR_SIZE] __attribute__((aligned(AVX_VECTOR_SIZE * sizeof(double))));
			_mm256_store_pd(&xp[0], xp64lo);
			_mm256_store_pd(&xp[0 + sizeof(xp64lo) / sizeof(double)], xp64hi);
			for (int k = 0; k < AVX_VECTOR_SIZE; k++)
			{
				const uint16_t& rowind = index.rowind[k];
		
				// This can be done scalar only.
				temps[rowind] *= xp[k];
			}
		}			
	}
	
	// Loop to calculate values.
	for (int i = 0; i < NNO; i++)
	{
		double temp = temps[i];
		if (!temp) continue;

		const __m256d temp64 = _mm256_set1_pd(temp);

		for (int b = DOF_CHOICE_START, Dof_choice = b, e = DOF_CHOICE_END; Dof_choice <= e;
			Dof_choice += sizeof(temp64) / sizeof(double))
		{
			const __m256d surplus64 = _mm256_load_pd(&surplus(i, Dof_choice));
			__m256d value64 = _mm256_load_pd(&value[Dof_choice - b]);
			
			// XXX Can be FMA here, if AVX2 is available
			value64 = _mm256_add_pd(value64, _mm256_mul_pd(temp64, surplus64));

			_mm256_storeu_pd(&value[Dof_choice - b], value64);
		}
	}		
#else
	// Loop to calculate temps.
	// Note temps vector should not be too large to keep up the caching.
	vector<double> temps(nno, 1.0);
	for (int i = 0, e = avxinds.size() / VDIM8; i < e; i++)
	{
		for (int j = 0; j < VDIM8; j++)
		{
			const AVXIndex& index = avxinds[i * VDIM8 + j];

			for (int k = 0; k < 8; k++)
			{
				const uint8_t& ind_i = index.i[k];
				const uint8_t& ind_j = index.j[k];
				const uint16_t& rowind = index.rowind[k];

				double xp = LinearBasis(x[j * AVX_VECTOR_SIZE + k], ind_i, ind_j);

				xp = fmax(0.0, xp);
		
				temps[rowind] *= xp;
			}
		}			
	}
	
	// Loop to calculate values.
	for (int i = 0; i < NNO; i++)
	{
		double temp = temps[i];
		if (!temp) continue;

		for (int b = DOF_CHOICE_START, Dof_choice = b, e = DOF_CHOICE_END; Dof_choice <= e; Dof_choice++)
			value[Dof_choice - b] += temps[i] * surplus(i, Dof_choice);
	}		
#endif
}

