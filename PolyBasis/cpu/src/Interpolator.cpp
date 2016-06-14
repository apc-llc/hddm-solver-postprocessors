#include <map>
#include <memory>
#include <vector>

#include "interpolator.h"
#include "JIT.h"

using namespace std;

const Parameters& Interpolator::getParameters() const { return params; }

Interpolator::Interpolator(const std::string& targetSuffix, const std::string& configFile) : 

params(targetSuffix, configFile)

{
	jit = params.enableRuntimeOptimization;
}

extern "C" void PolyBasis_CPU_Generic_InterpolateValue(
	const int dim, const int nno,
	const int Dof_choice, const double* x,
	const int* index, const double* surplus_t, double* value_);
	
// Interpolate a single value.
void Interpolator::interpolate(Data* data,
	const int istate, const real* x, const int Dof_choice, real& value)
{
	if (jit)
	{
		typedef void (*Func)(
			const int dim, const int nno,
			const int Dof_choice, const double* x,
			const int* index, const double* surplus_t, double* value_);

		static Func PolyBasis_CPU_RuntimeOpt_InterpolateValue;

		if (!PolyBasis_CPU_RuntimeOpt_InterpolateValue)
		{
			PolyBasis_CPU_RuntimeOpt_InterpolateValue =
				JIT::jitCompile(data->dim, "PolyBasis_CPU_RuntimeOpt_InterpolateValue_",
				(Func)PolyBasis_CPU_Generic_InterpolateValue).getFunc();
		}
		
		PolyBasis_CPU_RuntimeOpt_InterpolateValue(
			data->dim, data->nno, Dof_choice, x,
			data->index[istate].getData(), data->surplus_t[istate].getData(), &value);
	}
	else
	{			
		PolyBasis_CPU_Generic_InterpolateValue(
			data->dim, data->nno, Dof_choice, x,
			data->index[istate].getData(), data->surplus_t[istate].getData(), &value);
	}
}

extern "C" void PolyBasis_CPU_Generic_InterpolateArray(
	const int dim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const double* x,
	const int* index, const double* surplus, double* value);

// Interpolate array of values.
void Interpolator::interpolate(Data* data,
	const int istate, const real* x, const int Dof_choice_start, const int Dof_choice_end, real* value)
{
	if (jit)
	{
		typedef void (*Func)(
			const int dim, const int nno,
			const int Dof_choice_start, const int Dof_choice_end, const double* x,
			const int* index, const double* surplus, double* value);

		static Func PolyBasis_CPU_RuntimeOpt_InterpolateArray;

		if (!PolyBasis_CPU_RuntimeOpt_InterpolateArray)
		{
			PolyBasis_CPU_RuntimeOpt_InterpolateArray =
				JIT::jitCompile(data->dim, "PolyBasis_CPU_RuntimeOpt_InterpolateArray_",
				(Func)PolyBasis_CPU_Generic_InterpolateArray).getFunc();
		}
		
		PolyBasis_CPU_RuntimeOpt_InterpolateArray(
			data->dim, data->nno, Dof_choice_start, Dof_choice_end, x,
			data->index[istate].getData(), data->surplus[istate].getData(), value);
	}
	else
	{
		PolyBasis_CPU_Generic_InterpolateArray(
			data->dim, data->nno, Dof_choice_start, Dof_choice_end, x,
			data->index[istate].getData(), data->surplus[istate].getData(), value);
	}
}

extern "C" void PolyBasis_CPU_Generic_InterpolateArrayManyStateless(
	const int dim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const int count, const double* x_,
	const int* index, const double* surplus_t, double* value);

// Interpolate multiple arrays of values in continuous vector, with single surplus state.
void Interpolator::interpolate(Data* data,
	const int istate, const real* x, const int Dof_choice_start, const int Dof_choice_end, const int count, real* value)
{
	if (jit)
	{
		typedef void (*Func)(
			const int dim, const int nno,
			const int Dof_choice_start, const int Dof_choice_end, const int count, const double* x_,
			const int* index, const double* surplus_t, double* value);

		static Func PolyBasis_CPU_RuntimeOpt_InterpolateArrayManyStateless;

		if (!PolyBasis_CPU_RuntimeOpt_InterpolateArrayManyStateless)
		{
			PolyBasis_CPU_RuntimeOpt_InterpolateArrayManyStateless =
				JIT::jitCompile(data->dim, "PolyBasis_CPU_RuntimeOpt_InterpolateArrayManyStateless_",
				(Func)PolyBasis_CPU_Generic_InterpolateArrayManyStateless).getFunc();
		}

		PolyBasis_CPU_RuntimeOpt_InterpolateArrayManyStateless(
			data->dim, data->nno, Dof_choice_start, Dof_choice_end, count, x,
			data->index[istate].getData(), data->surplus_t[istate].getData(), value);
	}
	else
	{
		PolyBasis_CPU_Generic_InterpolateArrayManyStateless(
			data->dim, data->nno, Dof_choice_start, Dof_choice_end, count, x,
			data->index[istate].getData(), data->surplus_t[istate].getData(), value);
	}
}

extern "C" void PolyBasis_CPU_Generic_InterpolateArrayManyMultistate(
	const int dim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const int count, const double* const* x_,
	const int* const* index, const double* const* surplus_t, double** value);

// Interpolate multiple arrays of values in continuous vector, with multiple surplus states.
void Interpolator::interpolate(Data* data,
	const real** x, const int Dof_choice_start, const int Dof_choice_end, real** value)
{
	if (jit)
	{
		typedef void (*Func)(
			const int dim, const int nno,
			const int Dof_choice_start, const int Dof_choice_end, const int count, const double* const* x_,
			const int* const* index, const double* const* surplus_t, double** value);

		static Func PolyBasis_CPU_RuntimeOpt_InterpolateArrayManyMultistate;

		if (!PolyBasis_CPU_RuntimeOpt_InterpolateArrayManyMultistate)
		{
			PolyBasis_CPU_RuntimeOpt_InterpolateArrayManyMultistate =
				JIT::jitCompile(data->dim, "PolyBasis_CPU_RuntimeOpt_InterpolateArrayManyMultistate_",
				(Func)PolyBasis_CPU_Generic_InterpolateArrayManyMultistate).getFunc();
		}

		PolyBasis_CPU_RuntimeOpt_InterpolateArrayManyMultistate(
			data->dim, data->nno, Dof_choice_start, Dof_choice_end, data->nstates, x,
			data->getIndexes(), data->getSurpluses(), value);
	}
	else
	{
		PolyBasis_CPU_Generic_InterpolateArrayManyMultistate(
			data->dim, data->nno, Dof_choice_start, Dof_choice_end, data->nstates, x,
			data->getIndexes(), data->getSurpluses(), value);
	}
}

Interpolator* Interpolator::getInstance()
{
	static unique_ptr<Interpolator> interp;

	if (!interp.get())
		interp.reset(new Interpolator("CPU"));
	
	return interp.get();
}

extern "C" Interpolator* getInterpolator()
{
	return Interpolator::getInstance();
}

