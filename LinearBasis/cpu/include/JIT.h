#ifndef JIT_H
#define JIT_H

#ifdef HAVE_RUNTIME_OPTIMIZATION

#include "InterpolateKernel.h"

typedef void (*InterpolateValueFunc)(const int dim, const int nno,
	const int Dof_choice, const real* x,
	const int* index, const real* surplus_t,
	real* value);

typedef void (*InterpolateArrayFunc)(const int dim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const real* x,
	const int* index, const real* surplus_t,
	real* value);

typedef void (*InterpolateArrayManyStatelessFunc)(const int dim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const int count, const real* x,
	const int* index, const real* surplus_t,
	real* value);

typedef void (*InterpolateArrayManyMultistateFunc)(const int dim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const int count, const real* x,
	int** index, real** surplus_t,
	real* value);

typedef InterpolateKernel<InterpolateValueFunc> InterpolateValueKernel;
typedef InterpolateKernel<InterpolateArrayFunc> InterpolateArrayKernel;
typedef InterpolateKernel<InterpolateArrayManyStatelessFunc> InterpolateArrayManyStatelessKernel;
typedef InterpolateKernel<InterpolateArrayManyMultistateFunc> InterpolateArrayManyMultistateKernel;

class JIT
{
public :
	static InterpolateValueKernel& jitCompile(
		int dim, const std::string& funcnameTemplate, InterpolateValueFunc fallbackFunc);
	static InterpolateArrayKernel& jitCompile(
		int dim, const std::string& funcnameTemplate, InterpolateArrayFunc fallbackFunc);
	static InterpolateArrayManyStatelessKernel& jitCompile(
		int dim, const std::string& funcnameTemplate, InterpolateArrayManyStatelessFunc fallbackFunc);
	static InterpolateArrayManyMultistateKernel& jitCompile(
		int dim, const std::string& funcnameTemplate, InterpolateArrayManyMultistateFunc fallbackFunc);

	template<typename K, typename F>
	static K& jitCompile(int dim, const std::string& funcnameTemplate, F fallbackFunc);
};

#endif // HAVE_RUNTIME_OPTIMIZATION

#endif // JIT_H

