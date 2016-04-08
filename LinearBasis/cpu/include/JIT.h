#ifndef JIT_H
#define JIT_H

#ifdef HAVE_RUNTIME_OPTIMIZATION

#include "InterpolateKernel.h"

typedef void (*InterpolateArrayFunc)(const int dim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const real* x,
	const int* index, const real* surplus_t,
	real* value);

typedef void (*InterpolateArrayManyFunc)(const int dim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const int count, const real* x,
	const int* index, const real* surplus_t,
	real* value);

typedef void (*InterpolateValueFunc)(const int dim, const int nno,
	const int Dof_choice, const real* x,
	const int* index, const real* surplus_t,
	real* value);

typedef InterpolateKernel<InterpolateArrayFunc> InterpolateArrayKernel;
typedef InterpolateKernel<InterpolateArrayManyFunc> InterpolateArrayManyKernel;
typedef InterpolateKernel<InterpolateValueFunc> InterpolateValueKernel;

class JIT
{
public :
	static InterpolateArrayKernel& jitCompile(int dim, int count, const std::string& funcnameTemplate, InterpolateArrayFunc fallbackFunc);
	static InterpolateArrayManyKernel& jitCompile(int dim, int count, const std::string& funcnameTemplate, InterpolateArrayManyFunc fallbackFunc);
	static InterpolateValueKernel& jitCompile(int dim, int count, const std::string& funcnameTemplate, InterpolateValueFunc fallbackFunc);

	template<typename K, typename F>
	static K& jitCompile(int dim, int count, const std::string& funcnameTemplate, F fallbackFunc);
};

#endif // HAVE_RUNTIME_OPTIMIZATION

#endif // JIT_H

