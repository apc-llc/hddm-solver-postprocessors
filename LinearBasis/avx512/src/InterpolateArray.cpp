#include <assert.h>
#include <stdint.h>
#include <x86intrin.h>

#include "LinearBasis.h"
#include "Data.h"

using namespace NAMESPACE;
using namespace std;

class Device;

extern "C" void FUNCNAME(
	Device* device,
	const int dim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const double* x,
	const AVXIndexMatrix* avxinds_, const TransMatrix* trans_, const Matrix<double>* surplus_, double* value)
{
	assert(((size_t)x % (AVX_VECTOR_SIZE * sizeof(double)) == 0) && "x vector must be sufficiently memory-aligned");

	const AVXIndexMatrix& avxinds = *avxinds_;
	const TransMatrix& trans = *trans_;
	const Matrix<double>& surplus = *surplus_;

	// Index arrays shall be padded to AVX_VECTOR_SIZE-element
	// boundary to keep up the required alignment.
	int vdim = dim / AVX_VECTOR_SIZE;
	if (dim % AVX_VECTOR_SIZE) vdim++;
	vdim *= AVX_VECTOR_SIZE;

	const __m512d double8_0_0_0_0_0_0_0_0 = _mm512_setzero_pd();
	const __m512d double8_1_1_1_1_1_1_1_1 = _mm512_set1_pd(1.0);
	const __m512d sign_mask = _mm512_set1_pd(-0.);
	const __m256i int8_0_0_0_0_0_0_0_0 = _mm256_setzero_si256();

	int nfreqs = trans.size();
	int nnoAligned = nno;
	if (nno % AVX_VECTOR_SIZE)
		nnoAligned += AVX_VECTOR_SIZE - nno % AVX_VECTOR_SIZE;
		
	// One extra vector size for *hi part in AVX code below.
	nnoAligned += AVX_VECTOR_SIZE;
	
	vector<vector<double, AlignedAllocator<double> > > temps_(
		nfreqs, vector<double, AlignedAllocator<double> >(nnoAligned, 1.0));

	// Loop through all frequences.
	for (int ifreq = 0; ifreq < nfreqs; ifreq++)
	{
		const AVXIndexes& avxindsFreq = avxinds[ifreq];
		vector<double, AlignedAllocator<double> >& temps = temps_[ifreq];

		// Loop to calculate temps.
		for (int j = 0, itemp = 0; j < DIM; j++)
		{
			const __m512d x64 = _mm512_set1_pd(x[j]);
	
			for (int i = 0, e = avxindsFreq.getLength(j); i < e; i++, itemp += AVX_VECTOR_SIZE)
			{
				const AVXIndex& index = avxindsFreq(i, j);

				const __m256i ij8 = _mm256_load_si256(reinterpret_cast<const __m256i*>(&index));
				const __m256i i16 = _mm256_unpacklo_epi8(ij8, int8_0_0_0_0_0_0_0_0);
				const __m256i j16 = _mm256_unpackhi_epi8(ij8, int8_0_0_0_0_0_0_0_0);

				const __m256i i32lo = _mm256_unpacklo_epi16(i16, int8_0_0_0_0_0_0_0_0);
				const __m256i i32hi = _mm256_unpackhi_epi16(i16, int8_0_0_0_0_0_0_0_0);

				const __m256i j32lo = _mm256_unpacklo_epi16(j16, int8_0_0_0_0_0_0_0_0);
				const __m256i j32hi = _mm256_unpackhi_epi16(j16, int8_0_0_0_0_0_0_0_0);

				__m512d xp64lo = _mm512_sub_pd(double8_1_1_1_1_1_1_1_1, _mm512_andnot_pd(sign_mask,
					_mm512_sub_pd(_mm512_mul_pd(x64, _mm512_cvtepi32_pd(i32lo)), _mm512_cvtepi32_pd(j32lo))));

				const __mmask8 mask64lo = _mm512_cmp_pd_mask(xp64lo, double8_0_0_0_0_0_0_0_0, _MM_CMPINT_GT);
				xp64lo = _mm512_mask_blend_pd(mask64lo, double8_0_0_0_0_0_0_0_0, xp64lo);

				__m512d xp64hi = _mm512_sub_pd(double8_1_1_1_1_1_1_1_1, _mm512_andnot_pd(sign_mask,
					_mm512_sub_pd(_mm512_mul_pd(x64, _mm512_cvtepi32_pd(i32hi)), _mm512_cvtepi32_pd(j32hi))));

				const __mmask8 mask64hi = _mm512_cmp_pd_mask(xp64hi, double8_0_0_0_0_0_0_0_0, _MM_CMPINT_GT);
				xp64hi = _mm512_mask_blend_pd(mask64hi, double8_0_0_0_0_0_0_0_0, xp64hi);

				const __m512d temp64lo = _mm512_loadu_pd(&temps[itemp]);
				_mm512_storeu_pd(&temps[itemp], _mm512_mul_pd(temp64lo, xp64lo));

				const __m512d temp64hi = _mm512_loadu_pd(&temps[itemp + sizeof(xp64lo) / sizeof(double)]);
				_mm512_storeu_pd(&temps[itemp + sizeof(xp64lo) / sizeof(double)], _mm512_mul_pd(temp64hi, xp64hi));
			}			

			if (avxindsFreq.getLength(j))
			{
				const AVXIndex& index = avxindsFreq(avxindsFreq.getLength(j) - 1, j);
				for (int k = AVX_VECTOR_SIZE - 1; k >= 0; k--)
					if (index.isEmpty(k)) itemp--;
			}
		}
	}

	// Join temps from all frequencies.
	vector<double, AlignedAllocator<double> >& temps = temps_[0];
	for (int i = 0; i < NNO; i++)
		for (int ifreq = 1; ifreq < nfreqs; ifreq++)
			temps[i] *= temps_[ifreq][trans[ifreq][i]];

	// Loop to calculate values.
	for (int i = 0; i < NNO; i++)
	{
		double temp = temps[i];
		if (!temp) continue;

		const __m512d temp64 = _mm512_set1_pd(temp);

		for (int b = DOF_CHOICE_START, Dof_choice = b, e = DOF_CHOICE_END; Dof_choice <= e;
			Dof_choice += sizeof(temp64) / sizeof(double))
		{
			const __m512d surplus64 = _mm512_load_pd(&surplus(i, Dof_choice));
			__m512d value64 = _mm512_load_pd(&value[Dof_choice - b]);		
			value64 = _mm512_fmadd_pd(temp64, surplus64, value64);
			_mm512_store_pd(&value[Dof_choice - b], value64);
		}
	}
}

