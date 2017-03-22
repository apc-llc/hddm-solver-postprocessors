#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "interpolator.h"
#include "JIT.h"

namespace NAMESPACE {

class Device
{
public :

	long long getID() const;
};

} // namespace NAMESPACE

using namespace NAMESPACE;
using namespace std;

const Parameters& Interpolator::getParameters() const { return params; }

Interpolator::Interpolator(const std::string& targetSuffix, const std::string& configFile) : 

params(targetSuffix, configFile)

{
	jit = params.enableRuntimeOptimization;
}

extern "C" void LinearBasis_CUDA_Generic_InterpolateValue(
	Device* device,
	const int dim, const int nno,
	const int Dof_choice, const double* x,
	const Matrix<int>::Device::Dense* index, const Matrix<real>::Device::Dense* surplus, double* value_);
	
// Interpolate a single value.
void Interpolator::interpolate(Device* device, Data* data,
	const int istate, const real* x, const int Dof_choice, real& value)
{
	if (jit)
	{
		typedef void (*Func)(
			Device* device,
			const int dim, const int nno,
			const int Dof_choice, const double* x,
			const Matrix<int>::Device::Dense* index, const Matrix<real>::Device::Dense* surplus, double* value_);

		static map<long long, Func> LinearBasis_CUDA_RuntimeOpt_InterpolateValue;

		if (!LinearBasis_CUDA_RuntimeOpt_InterpolateValue[device->getID()])
		{
			LinearBasis_CUDA_RuntimeOpt_InterpolateValue[device->getID()] =
				JIT::jitCompile(device, data->dim, "LinearBasis_CUDA_RuntimeOpt_InterpolateValue_",
				(Func)LinearBasis_CUDA_Generic_InterpolateValue).getFunc();
		}
		
		LinearBasis_CUDA_RuntimeOpt_InterpolateValue[device->getID()](
			device, data->dim, data->nno, Dof_choice, x,
			data->device.getIndex(istate), data->device.getSurplus(istate), &value);
	}
	else
	{			
		LinearBasis_CUDA_Generic_InterpolateValue(
			device, data->dim, data->nno, Dof_choice, x,
			data->device.getIndex(istate), data->device.getSurplus(istate), &value);
	}
}

extern "C" void LinearBasis_CUDA_Generic_InterpolateArray(
	Device* device, const int dim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const double* x,
	const Matrix<int>::Device::Dense* index, const Matrix<real>::Device::Dense* surplus, double* value);

// Interpolate array of values.
void Interpolator::interpolate(Device* device, Data* data,
	const int istate, const real* x, const int Dof_choice_start, const int Dof_choice_end, real* value)
{
	if (jit)
	{
		typedef void (*Func)(
			Device* device, const int dim, const int nno,
			const int Dof_choice_start, const int Dof_choice_end, const double* x,
			const Matrix<int>::Device::Dense* index, const Matrix<real>::Device::Dense* surplus, double* value);

		static map<long long, Func> LinearBasis_CUDA_RuntimeOpt_InterpolateArray;

		if (!LinearBasis_CUDA_RuntimeOpt_InterpolateArray[device->getID()])
		{
			LinearBasis_CUDA_RuntimeOpt_InterpolateArray[device->getID()] =
				JIT::jitCompile(device, data->dim, "LinearBasis_CUDA_RuntimeOpt_InterpolateArray_",
				(Func)LinearBasis_CUDA_Generic_InterpolateArray).getFunc();
		}
		
		LinearBasis_CUDA_RuntimeOpt_InterpolateArray[device->getID()](
			device, data->dim, data->nno, Dof_choice_start, Dof_choice_end, x,
			data->device.getIndex(istate), data->device.getSurplus(istate), value);
	}
	else
	{
		LinearBasis_CUDA_Generic_InterpolateArray(
			device, data->dim, data->nno, Dof_choice_start, Dof_choice_end, x,
			data->device.getIndex(istate), data->device.getSurplus(istate), value);
	}
}

extern "C" void LinearBasis_CUDA_Generic_InterpolateArrayManyStateless(
	Device* device, const int dim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const int count, const double* x_,
	const Matrix<int>::Device::Dense* index, const Matrix<real>::Device::Dense* surplus, double* value);

// TODO
// Interpolate multiple arrays of values, with single surplus state.
void Interpolator::interpolate(Device* device, Data* data,
	const int istate, const real* x, const int Dof_choice_start, const int Dof_choice_end, const int count, real* value)
{
	if (jit)
	{
		typedef void (*Func)(
			Device* device, const int dim, const int nno,
			const int Dof_choice_start, const int Dof_choice_end, const int count, const double* x_,
			const Matrix<int>::Device::Dense* index, const Matrix<real>::Device::Dense* surplus, double* value);

		static map<long long, Func> LinearBasis_CUDA_RuntimeOpt_InterpolateArrayManyStateless;

		if (!LinearBasis_CUDA_RuntimeOpt_InterpolateArrayManyStateless[device->getID()])
		{
			LinearBasis_CUDA_RuntimeOpt_InterpolateArrayManyStateless[device->getID()] =
				JIT::jitCompile(device, data->dim, count, "LinearBasis_CUDA_RuntimeOpt_InterpolateArrayManyStateless_",
				(Func)LinearBasis_CUDA_Generic_InterpolateArrayManyStateless).getFunc();
		}

		LinearBasis_CUDA_RuntimeOpt_InterpolateArrayManyStateless[device->getID()](
			device, data->dim, data->nno, Dof_choice_start, Dof_choice_end, count, x,
			data->device.getIndex(istate), data->device.getSurplus(istate), value);
	}
	else
	{
		LinearBasis_CUDA_Generic_InterpolateArrayManyStateless(
			device, data->dim, data->nno, Dof_choice_start, Dof_choice_end, count, x,
			data->device.getIndex(istate), data->device.getSurplus(istate), value);
	}
}

extern "C" void LinearBasis_CUDA_Generic_InterpolateArrayManyMultistate(
	Device* device,
	const int dim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const int count, const double* const* x_,
	const Matrix<int>::Device::Dense* index, const Matrix<real>::Device::Dense* surplus, double** value);

// Interpolate multiple arrays of values, with multiple surplus states.
void Interpolator::interpolate(Device* device, Data* data,
	const real** x, const int Dof_choice_start, const int Dof_choice_end, real** value)
{
	if (jit)
	{
		typedef void (*Func)(
			Device* device,
			const int dim, const int nno,
			const int Dof_choice_start, const int Dof_choice_end, const int count, const double* const* x_,
			const Matrix<int>::Device::Dense* index, const Matrix<real>::Device::Dense* surplus, double** value);

		static map<long long, Func> LinearBasis_CUDA_RuntimeOpt_InterpolateArrayManyMultistate;

		if (!LinearBasis_CUDA_RuntimeOpt_InterpolateArrayManyMultistate[device->getID()])
		{
			LinearBasis_CUDA_RuntimeOpt_InterpolateArrayManyMultistate[device->getID()] =
				JIT::jitCompile(device, data->dim, data->nstates, "LinearBasis_CUDA_RuntimeOpt_InterpolateArrayManyMultistate_",
				(Func)LinearBasis_CUDA_Generic_InterpolateArrayManyMultistate).getFunc();
		}

		LinearBasis_CUDA_RuntimeOpt_InterpolateArrayManyMultistate[device->getID()](
			device, data->dim, data->nno, Dof_choice_start, Dof_choice_end, data->nstates, x,
			data->device.getIndex(0), data->device.getSurplus(0), value);
	}
	else
	{
		LinearBasis_CUDA_Generic_InterpolateArrayManyMultistate(
			device, data->dim, data->nno, Dof_choice_start, Dof_choice_end, data->nstates, x,
			data->device.getIndex(0), data->device.getSurplus(0), value);
	}
}

Interpolator* Interpolator::getInstance()
{
	static unique_ptr<Interpolator> interp;

	if (!interp.get())
		interp.reset(new Interpolator("CUDA"));
	
	return interp.get();
}

Interpolator::~Interpolator()
{
}

extern "C" Interpolator* getInterpolator()
{
	return Interpolator::getInstance();
}

