#ifndef LINEAR_BASIS_H
#define LINEAR_BASIS_H

#include <math.h>

namespace NAMESPACE {

#if defined(__CUDACC__)

#if defined(__CUDA_ARCH__)
__device__
#endif
float inline __attribute__((always_inline)) LinearBasis(float x, unsigned short i, unsigned short j)
{ return 1.0f - fabsf(x * i - j); }

#if defined(__CUDA_ARCH__)
__device__
#endif
double inline __attribute__((always_inline)) LinearBasis(double x, unsigned short i, unsigned short j)
{ return 1.0 - fabs(x * i - j); }

#else

#define LinearBasis(x, i, j) (1.0 - fabs((x) * (i) - (j)))

#endif

} // namespace NAMESPACE

#endif // LINEAR_BASIS_H

