#ifdef HAVE_AVX
#include <assert.h>
#include <stdint.h>
#include <x86intrin.h>
#else
#include "LinearBasis.h"
#endif

#include "Data.h"
#include "Device.h"

#include <stdio.h>
#include <stdlib.h>

#define STR(funcname) #funcname

using namespace cuda;

extern "C" void FUNCNAME(
	Device* device,
	const int dim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const int count, double* x,
	const Matrix<int>::Device* index, const Matrix<double>::Device* surplus, double* value)
{
	printf("%s is not implemented\n", STR(FUNCNAME));
	abort();
}

