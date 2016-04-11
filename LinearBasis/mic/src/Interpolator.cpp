#include <map>
#include <memory>
#include <vector>

#include "Data.h"
#include "JIT.h"

using namespace std;

namespace Optimizer
{
	int count();
}

class Interpolator
{
	bool jit;

public :
	Interpolator(bool jit_);

	// Interpolate a single value.
	void interpolate(Data* data, const real* x, const int Dof_choice, real& value);

	// Interpolate array of values.
	void interpolate(Data* data, const real* x, const int Dof_choice_start, const int Dof_choice_end, real* value);

	// Interpolate multiple arrays of values in continuous vector.
	void interpolate(Data* data, const real* x, const int Dof_choice_start, const int Dof_choice_end, const int count, real* value);
};

Interpolator::Interpolator(bool jit_) : jit(jit_) { }

extern "C" void LinearBasis_CPU_Generic_InterpolateValue(
	const int dim, const int nno,
	const int Dof_choice, const double* x,
	const int* index, const double* surplus_t, double* value_);

// Interpolate a single value.
void Interpolator::interpolate(Data* data, const real* x, const int Dof_choice, real& value)
{
	if (jit)
	{
		typedef void (*Func)(
			const int dim, const int nno,
			const int Dof_choice, const double* x,
			const int* index, const double* surplus_t, double* value_);

		static Func LinearBasis_CPU_RuntimeOpt_InterpolateValue;

		if (!LinearBasis_CPU_RuntimeOpt_InterpolateValue)
		{
			LinearBasis_CPU_RuntimeOpt_InterpolateValue =
				JIT::jitCompile(data->dim, 1, "LinearBasis_CPU_RuntimeOpt_InterpolateValue_",
				(Func)LinearBasis_CPU_Generic_InterpolateValue).getFunc();
		}
		
		LinearBasis_CPU_RuntimeOpt_InterpolateValue(
			data->dim, data->nno, Dof_choice, x,
			data->index.getData(), data->surplus_t.getData(), &value);
	}
	else
	{			
		LinearBasis_CPU_Generic_InterpolateValue(
			data->dim, data->nno, Dof_choice, x,
			data->index.getData(), data->surplus_t.getData(), &value);
	}
}

extern "C" void LinearBasis_CPU_Generic_InterpolateArray(
	const int dim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const double* x,
	const int* index, const double* surplus_t, double* value);

// Interpolate array of values.
void Interpolator::interpolate(Data* data, const real* x, const int Dof_choice_start, const int Dof_choice_end, real* value)
{
	if (jit)
	{
		typedef void (*Func)(
			const int dim, const int nno,
			const int Dof_choice_start, const int Dof_choice_end, const double* x,
			const int* index, const double* surplus_t, double* value);

		static Func LinearBasis_CPU_RuntimeOpt_InterpolateArray;

		if (!LinearBasis_CPU_RuntimeOpt_InterpolateArray)
		{
			LinearBasis_CPU_RuntimeOpt_InterpolateArray =
				JIT::jitCompile(data->dim, 1, "LinearBasis_CPU_RuntimeOpt_InterpolateArray_",
				(Func)LinearBasis_CPU_Generic_InterpolateArray).getFunc();
		}
		
		LinearBasis_CPU_RuntimeOpt_InterpolateArray(
			data->dim, data->nno, Dof_choice_start, Dof_choice_end, x,
			data->index.getData(), data->surplus_t.getData(), value);
	}
	else
	{
		LinearBasis_CPU_Generic_InterpolateArray(
			data->dim, data->nno, Dof_choice_start, Dof_choice_end, x,
			data->index.getData(), data->surplus_t.getData(), value);
	}
}

extern "C" void LinearBasis_CPU_Generic_InterpolateArrayMany(
	const int dim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const int count, const double* x_,
	const int* index, const double* surplus_t, double* value);

// Interpolate multiple arrays of values in continuous vector.
void Interpolator::interpolate(Data* data, const real* x, const int Dof_choice_start, const int Dof_choice_end, const int count, real* value){
	if (jit)
	{
		typedef void (*Func)(
			const int dim, const int nno,
			const int Dof_choice_start, const int Dof_choice_end, const int count, const double* x_,
			const int* index, const double* surplus_t, double* value);

		static Func LinearBasis_CPU_RuntimeOpt_InterpolateArrayMany;

		if (!LinearBasis_CPU_RuntimeOpt_InterpolateArrayMany)
		{
			int count = Optimizer::count();
			LinearBasis_CPU_RuntimeOpt_InterpolateArrayMany =
				JIT::jitCompile(data->dim, count, "LinearBasis_CPU_RuntimeOpt_InterpolateArrayMany_",
				(Func)LinearBasis_CPU_Generic_InterpolateArrayMany).getFunc();
		}

		LinearBasis_CPU_RuntimeOpt_InterpolateArrayMany(
			data->dim, data->nno, Dof_choice_start, Dof_choice_end, count, x,
			data->index.getData(), data->surplus_t.getData(), value);
	}
	else
	{
		LinearBasis_CPU_Generic_InterpolateArrayMany(
			data->dim, data->nno, Dof_choice_start, Dof_choice_end, count, x,
			data->index.getData(), data->surplus_t.getData(), value);
	}
}

static map<bool, unique_ptr<Interpolator> > interp;

extern "C" Interpolator* getInterpolator(bool jit)
{
	if (!interp[jit].get())
		interp[jit].reset(new Interpolator(jit));
	
	return interp[jit].get();
}

