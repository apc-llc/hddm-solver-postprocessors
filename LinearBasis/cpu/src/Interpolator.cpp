#include <map>
#include <memory>
#include <vector>

#include "Data.h"
#include "JIT.h"

using namespace std;

class Interpolator
{
	bool jit;

public :
	Interpolator(bool jit_);

	// Interpolate a single value.
	void interpolate(Data* data,
		const int istate, const real* x, const int Dof_choice, real& value);

	// Interpolate array of values.
	void interpolate(Data* data,
		const int istate, const real* x, const int Dof_choice_start, const int Dof_choice_end, real* value);

	// Interpolate multiple arrays of values in continuous vector, with single surplus state.
	void interpolate(Data* data,
		const int istate, const real* x, const int Dof_choice_start, const int Dof_choice_end, const int count, real* value);

	// Interpolate multiple arrays of values in continuous vector, with multiple surplus states.
	void interpolate(Data* data,
		const real* x, const int Dof_choice_start, const int Dof_choice_end, real* value);
};

Interpolator::Interpolator(bool jit_) : jit(jit_) { }

extern "C" void LinearBasis_CPU_Generic_InterpolateValue(
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

		static Func LinearBasis_CPU_RuntimeOpt_InterpolateValue;

		if (!LinearBasis_CPU_RuntimeOpt_InterpolateValue)
		{
			LinearBasis_CPU_RuntimeOpt_InterpolateValue =
				JIT::jitCompile(data->dim, "LinearBasis_CPU_RuntimeOpt_InterpolateValue_",
				(Func)LinearBasis_CPU_Generic_InterpolateValue).getFunc();
		}
		
		LinearBasis_CPU_RuntimeOpt_InterpolateValue(
			data->dim, data->nno, Dof_choice, x,
			data->index[istate].getData(), data->surplus_t[istate].getData(), &value);
	}
	else
	{			
		LinearBasis_CPU_Generic_InterpolateValue(
			data->dim, data->nno, Dof_choice, x,
			data->index[istate].getData(), data->surplus_t[istate].getData(), &value);
	}
}

extern "C" void LinearBasis_CPU_Generic_InterpolateArray(
	const int dim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const double* x,
	const int* index, const double* surplus_t, double* value);

// Interpolate array of values.
void Interpolator::interpolate(Data* data,
	const int istate, const real* x, const int Dof_choice_start, const int Dof_choice_end, real* value)
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
				JIT::jitCompile(data->dim, "LinearBasis_CPU_RuntimeOpt_InterpolateArray_",
				(Func)LinearBasis_CPU_Generic_InterpolateArray).getFunc();
		}
		
		LinearBasis_CPU_RuntimeOpt_InterpolateArray(
			data->dim, data->nno, Dof_choice_start, Dof_choice_end, x,
			data->index[istate].getData(), data->surplus_t[istate].getData(), value);
	}
	else
	{
		LinearBasis_CPU_Generic_InterpolateArray(
			data->dim, data->nno, Dof_choice_start, Dof_choice_end, x,
			data->index[istate].getData(), data->surplus_t[istate].getData(), value);
	}
}

extern "C" void LinearBasis_CPU_Generic_InterpolateArrayManyStateless(
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

		static Func LinearBasis_CPU_RuntimeOpt_InterpolateArrayManyStateless;

		if (!LinearBasis_CPU_RuntimeOpt_InterpolateArrayManyStateless)
		{
			LinearBasis_CPU_RuntimeOpt_InterpolateArrayManyStateless =
				JIT::jitCompile(data->dim, "LinearBasis_CPU_RuntimeOpt_InterpolateArrayManyStateless_",
				(Func)LinearBasis_CPU_Generic_InterpolateArrayManyStateless).getFunc();
		}

		LinearBasis_CPU_RuntimeOpt_InterpolateArrayManyStateless(
			data->dim, data->nno, Dof_choice_start, Dof_choice_end, count, x,
			data->index[istate].getData(), data->surplus_t[istate].getData(), value);
	}
	else
	{
		LinearBasis_CPU_Generic_InterpolateArrayManyStateless(
			data->dim, data->nno, Dof_choice_start, Dof_choice_end, count, x,
			data->index[istate].getData(), data->surplus_t[istate].getData(), value);
	}
}

extern "C" void LinearBasis_CPU_Generic_InterpolateArrayManyMultistate(
	const int dim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const int count, const double* x_,
	int** index, double** surplus_t, double* value);

// Interpolate multiple arrays of values in continuous vector, with multiple surplus states.
void Interpolator::interpolate(Data* data,
	const real* x, const int Dof_choice_start, const int Dof_choice_end, real* value)
{
	vector<int*> indexes; indexes.resize(data->nstates);
	vector<real*> surpluses; surpluses.resize(data->nstates);
	for (int i = 0; i < data->nstates; i++)
	{
		indexes[i] = data->index[i].getData();
		surpluses[i] = data->surplus_t[i].getData();
	}

	if (jit)
	{
		typedef void (*Func)(
			const int dim, const int nno,
			const int Dof_choice_start, const int Dof_choice_end, const int count, const double* x_,
			int** index, double** surplus_t, double* value);

		static Func LinearBasis_CPU_RuntimeOpt_InterpolateArrayManyMultistate;

		if (!LinearBasis_CPU_RuntimeOpt_InterpolateArrayManyMultistate)
		{
			LinearBasis_CPU_RuntimeOpt_InterpolateArrayManyMultistate =
				JIT::jitCompile(data->dim, "LinearBasis_CPU_RuntimeOpt_InterpolateArrayManyMultistate_",
				(Func)LinearBasis_CPU_Generic_InterpolateArrayManyMultistate).getFunc();
		}

		LinearBasis_CPU_RuntimeOpt_InterpolateArrayManyMultistate(
			data->dim, data->nno, Dof_choice_start, Dof_choice_end, data->nstates, x,
			&indexes[0], &surpluses[0], value);
	}
	else
	{
		LinearBasis_CPU_Generic_InterpolateArrayManyMultistate(
			data->dim, data->nno, Dof_choice_start, Dof_choice_end, data->nstates, x,
			&indexes[0], &surpluses[0], value);
	}
}

static map<bool, unique_ptr<Interpolator> > interp;

extern "C" Interpolator* getInterpolator(bool jit)
{
	if (!interp[jit].get())
		interp[jit].reset(new Interpolator(jit));
	
	return interp[jit].get();
}

