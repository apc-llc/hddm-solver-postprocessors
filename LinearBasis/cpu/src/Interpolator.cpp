#include <map>
#include <memory>
#include <vector>

#include "interpolator.h"
#include "JIT.h"

using namespace cpu;
using namespace std;

const Parameters& Interpolator::getParameters() const { return params; }

Interpolator::Interpolator(const std::string& targetSuffix, const std::string& configFile) : 

params(targetSuffix, configFile)

{
	jit = params.enableRuntimeOptimization;
}

extern "C" void LinearBasis_CPU_Generic_InterpolateArray(
	Device* device, const int dim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const double* x,
	const AVXIndexMatrix* avxinds, const Matrix<double>* surplus, double* value);

// Interpolate array of values.
void Interpolator::interpolate(Device* device, Data* data,
	const int istate, const real* x, const int Dof_choice_start, const int Dof_choice_end, real* value)
{
	if (jit)
	{
		typedef void (*Func)(
			Device* device, const int dim, const int nno,
			const int Dof_choice_start, const int Dof_choice_end, const double* x,
			const AVXIndexMatrix* avxinds, const Matrix<double>* surplus, double* value);

		static Func LinearBasis_CPU_RuntimeOpt_InterpolateArray;

		if (!LinearBasis_CPU_RuntimeOpt_InterpolateArray)
		{
			LinearBasis_CPU_RuntimeOpt_InterpolateArray =
				JIT::jitCompile(data->dim, data->nno, Dof_choice_start, Dof_choice_end,
				"LinearBasis_CPU_RuntimeOpt_InterpolateArray_",
				(Func)LinearBasis_CPU_Generic_InterpolateArray).getFunc();
		}
		
		LinearBasis_CPU_RuntimeOpt_InterpolateArray(
			device, data->dim, data->nno, Dof_choice_start, Dof_choice_end, x,
			&data->avxinds[istate], &data->surplus[istate], value);
	}
	else
	{
		LinearBasis_CPU_Generic_InterpolateArray(
			device, data->dim, data->nno, Dof_choice_start, Dof_choice_end, x,
			&data->avxinds[istate], &data->surplus[istate], value);
	}
}

extern "C" void LinearBasis_CPU_Generic_InterpolateArrayManyMultistate(
	Device* device, const int dim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const int count, const double* const* x_,
	const AVXIndexMatrix* avxinds, const Matrix<double>* surplus, double** value);

// Interpolate multiple arrays of values, with multiple surplus states.
void Interpolator::interpolate(Device* device, Data* data,
	const real** x, const int Dof_choice_start, const int Dof_choice_end, real** value)
{
	if (jit)
	{
		typedef void (*Func)(
			Device* device, const int dim, const int nno,
			const int Dof_choice_start, const int Dof_choice_end, const int count, const double* const* x_,
			const AVXIndexMatrix* avxinds, const Matrix<double>* surplus, double** value);

		static Func LinearBasis_CPU_RuntimeOpt_InterpolateArrayManyMultistate;

		if (!LinearBasis_CPU_RuntimeOpt_InterpolateArrayManyMultistate)
		{
			LinearBasis_CPU_RuntimeOpt_InterpolateArrayManyMultistate =
				JIT::jitCompile(data->dim, data->nstates, data->nno, Dof_choice_start, Dof_choice_end,
				"LinearBasis_CPU_RuntimeOpt_InterpolateArrayManyMultistate_",
				(Func)LinearBasis_CPU_Generic_InterpolateArrayManyMultistate).getFunc();
		}

		LinearBasis_CPU_RuntimeOpt_InterpolateArrayManyMultistate(
			device, data->dim, data->nno, Dof_choice_start, Dof_choice_end, data->nstates, x,
			&data->avxinds[0], &data->surplus[0], value);
	}
	else
	{
		LinearBasis_CPU_Generic_InterpolateArrayManyMultistate(
			device, data->dim, data->nno, Dof_choice_start, Dof_choice_end, data->nstates, x,
			&data->avxinds[0], &data->surplus[0], value);
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

