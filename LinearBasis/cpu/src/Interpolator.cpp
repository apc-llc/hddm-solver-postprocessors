#include <map>
#include <memory>
#include <vector>

#include "interpolator.h"
#include "JIT.h"

#define str(x) #x
#define stringize(x) str(x)

using namespace NAMESPACE;
using namespace std;

const Parameters& Interpolator::getParameters() const { return params; }

Interpolator::Interpolator(const std::string& targetSuffix, const std::string& configFile) : 

params(targetSuffix, configFile)

{
	jit = params.enableRuntimeOptimization;
}

extern "C" void INTERPOLATE_ARRAY(
	Device* device, const int dim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const double* x,
	const AVXIndexMatrix* avxinds, const TransMatrix* trans, const Matrix<double>* surplus, double* value);

// Interpolate array of values.
void Interpolator::interpolate(Device* device, Data* data,
	const int istate, const real* x, const int Dof_choice_start, const int Dof_choice_end, real* value)
{
	if (jit)
	{
		typedef void (*Func)(
			Device* device, const int dim, const int nno,
			const int Dof_choice_start, const int Dof_choice_end, const double* x,
			const AVXIndexMatrix* avxinds, const TransMatrix* trans, const Matrix<double>* surplus, double* value);

		static Func INTERPOLATE_ARRAY_RUNTIME_OPT;

		if (!INTERPOLATE_ARRAY_RUNTIME_OPT)
		{
			INTERPOLATE_ARRAY_RUNTIME_OPT =
				JIT::jitCompile(data->dim, data->nno, Dof_choice_start, Dof_choice_end,
				stringize(INTERPOLATE_ARRAY_RUNTIME_OPT) "_",
				(Func)INTERPOLATE_ARRAY).getFunc();
		}
		
		INTERPOLATE_ARRAY_RUNTIME_OPT(
			device, data->dim, data->nno, Dof_choice_start, Dof_choice_end, x,
			&data->avxinds[istate], &data->trans[istate], &data->surplus[istate], value);
	}
	else
	{
		INTERPOLATE_ARRAY(
			device, data->dim, data->nno, Dof_choice_start, Dof_choice_end, x,
			&data->avxinds[istate], &data->trans[istate], &data->surplus[istate], value);
	}
}

extern "C" void INTERPOLATE_ARRAY_MANY_MULTISTATE(
	Device* device, const int dim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const int count, const double* const* x_,
	const AVXIndexMatrix* avxinds, const TransMatrix* trans, const Matrix<double>* surplus, double** value);

// Interpolate multiple arrays of values, with multiple surplus states.
void Interpolator::interpolate(Device* device, Data* data,
	const real** x, const int Dof_choice_start, const int Dof_choice_end, real** value)
{
	typedef void (*Func)(
		Device* device, const int dim, const int nno,
		const int Dof_choice_start, const int Dof_choice_end, const int count, const double* const* x_,
		const AVXIndexMatrix* avxinds, const TransMatrix* trans, const Matrix<double>* surplus, double** value);

	static Func INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT;

	if (jit)
	{
		if (!INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT)
		{
			INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT =
				JIT::jitCompile(data->dim, data->nstates, data->nno, Dof_choice_start, Dof_choice_end,
				stringize(INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT) "_",
				(Func)INTERPOLATE_ARRAY_MANY_MULTISTATE).getFunc();
		}

		INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT(
			device, data->dim, data->nno, Dof_choice_start, Dof_choice_end, data->nstates, x,
			&data->avxinds[0], &data->trans[0], &data->surplus[0], value);
	}
	else
	{
		INTERPOLATE_ARRAY_MANY_MULTISTATE(
			device, data->dim, data->nno, Dof_choice_start, Dof_choice_end, data->nstates, x,
			&data->avxinds[0], &data->trans[0], &data->surplus[0], value);
	}
}

Interpolator* Interpolator::getInstance()
{
	static unique_ptr<Interpolator> interp;

	if (!interp.get())
		interp.reset(new Interpolator("CPU"));
	
	return interp.get();
}

Interpolator::~Interpolator()
{
}

extern "C" Interpolator* getInterpolator()
{
	return Interpolator::getInstance();
}

