#ifndef JIT_H
#define JIT_H

#ifdef HAVE_RUNTIME_OPTIMIZATION

#include "Data.h"
#include "InterpolateKernel.h"

namespace NAMESPACE {

class Device;

typedef void (*InterpolateArrayFunc)(
	Device* device,
	const int dim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const real* x,
	const AVXIndexMatrix* avxinds, const TransMatrix* trans_, const Matrix<real>* surplus,
	real* value);

typedef void (*InterpolateArrayManyMultistateFunc)(
	Device* device,
	const int dim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const int count, const real* const* x,
	const AVXIndexMatrix* avxinds, const TransMatrix* trans_, const Matrix<real>* surplus,
	real** value);

typedef InterpolateKernel<InterpolateArrayFunc> InterpolateArrayKernel;
typedef InterpolateKernel<InterpolateArrayManyMultistateFunc> InterpolateArrayManyMultistateKernel;

class JIT
{
public :
	static InterpolateArrayKernel& jitCompile(
		int dim, int nno, int Dof_choice_start, int Dof_choice_end,
		const std::string& funcnameTemplate, InterpolateArrayFunc fallbackFunc);
	static InterpolateArrayManyMultistateKernel& jitCompile(
		int dim, int count, int nno, int Dof_choice_start, int Dof_choice_end,
		const std::string& funcnameTemplate, InterpolateArrayManyMultistateFunc fallbackFunc);

	template<typename K, typename F>
	static K& jitCompile(
		int dim, int count, int nno, int Dof_choice_start, int Dof_choice_end,
		const std::string& funcnameTemplate, F fallbackFunc);
};

} // namespace NAMESPACE

#endif // HAVE_RUNTIME_OPTIMIZATION

#endif // JIT_H

