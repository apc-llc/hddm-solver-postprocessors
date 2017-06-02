#ifndef LINEAR_BASIS_H
#define LINEAR_BASIS_H

#include <cstdio>
#include <math.h>
#include <stdint.h>

namespace NAMESPACE {

#if defined(__CUDACC__)

#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__) || defined(__CUDACC_RTC__)
#define __PTR   "l"
#else
#define __PTR   "r"
#endif

__device__
inline __attribute__((always_inline)) double LinearBasis(const double* x_, const unsigned int* index)
{
	double ret;
	asm(
		"{"
		".reg .u8 i8, j8, idx_hi, idx_lo;"
		".reg .u16 idx16;"
		".reg .b32 idx32;"
		".reg .b64 addr;"
		".reg .f64 x64, i64, j64;"
		"ld.global.v4.u8 {i8, j8, idx_hi, idx_lo}, [%1];" 
		"mov.b16 idx16, {idx_hi, idx_lo};"
		"cvt.u32.u16 idx32, idx16;"
		"mad.wide.u32 addr, idx32, 0x8, %2;"
		"ld.global.f64 x64, [addr];"
		"cvt.rn.f64.u8 i64, i8;"
		"cvt.rn.f64.u8 j64, j8;"
		"neg.f64 j64, j64;"
		"fma.rn.f64 x64, x64, i64, j64;"
		"abs.f64 x64, x64;"
		"fma.rn.f64 %0, x64, -1.0, 1.0;"
		"}" :
		"=d"(ret) : __PTR(index), __PTR(x_));
	return ret;
}

#else

#define LinearBasis(x, i, j) (1.0 - fabs((x) * (i) - (j)))

#endif

} // namespace NAMESPACE

#endif // LINEAR_BASIS_H

