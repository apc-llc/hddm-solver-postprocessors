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

// Interpolate a single value.
void Interpolator::interpolate(Data* data, const real* x, const int Dof_choice, real& value)
{
	real* device_x = NULL;
	MIC_ERROR_CHECK(micMallocAligned((void**)&device_x, data->dim * sizeof(real), AVX_VECTOR_SIZE * sizeof(real)));
	
	real* device_value = NULL;
	MIC_ERROR_CHECK(micMallocAligned((void**)&device_value, sizeof(real), AVX_VECTOR_SIZE * sizeof(real)));
	
	MIC_ERROR_CHECK(micMemcpy(device_x, x, data->dim * sizeof(real), micMemcpyHostToDevice));

	typedef struct
	{
		int dim, nno, Dof_choice;
		double *x, *surplus_t, *value, *index;
	}
	Args;

	Args host_args;
	host_args.dim = data->dim;
	host_args.nno = data->nno;
	host_args.Dof_choice = Dof_choice;
	host_args.x = device_x;
	host_args.surplus_t = data->surplus_t.getData();
	host_args.value = device_value;
	host_args.index = data->index.getData();

	Args* device_args;
	MIC_ERROR_CHECK(micMallocAligned((void**)&device_args, sizeof(Args), AVX_VECTOR_SIZE * sizeof(double)));
	MIC_ERROR_CHECK(micMemcpy(device_args, &host_args, sizeof(Args), micMemcpyHostToDevice));

	if (jit)
	{
		typedef void (*Func)(
			const int dim, const int nno,
			const int Dof_choice, const double* x,
			const double* index, const double* surplus_t, double* value_);

		static Func LinearBasis_MIC_RuntimeOpt_InterpolateValue;

		if (!LinearBasis_MIC_RuntimeOpt_InterpolateValue)
		{
			// TODO
			/*LinearBasis_MIC_RuntimeOpt_InterpolateValue =
				JIT::jitCompile(data->dim, 1, "LinearBasis_MIC_RuntimeOpt_InterpolateValue_",
				(Func)LinearBasis_MIC_Generic_InterpolateValue).getFunc();*/
		}
		
		LinearBasis_MIC_RuntimeOpt_InterpolateValue(
			data->dim, data->nno, Dof_choice, x,
			data->index.getData(), data->surplus_t.getData(), &value);
	}
	else
	{			
		MIC_ERROR_CHECK(micLaunchKernel("LinearBasis_MIC_Generic_InterpolateValue", device_args));
		MIC_ERROR_CHECK(micDeviceSynchronize());
	}
	
	MIC_ERROR_CHECK(micMemcpy(&value, device_value, sizeof(real), micMemcpyDeviceToHost));
	
	MIC_ERROR_CHECK(micFree(device_args));
	MIC_ERROR_CHECK(micFree(device_x));
	MIC_ERROR_CHECK(micFree(device_value));
}

// Interpolate array of values.
void Interpolator::interpolate(Data* data, const real* x, const int Dof_choice_start, const int Dof_choice_end, real* value)
{
	real* device_x = NULL;
	MIC_ERROR_CHECK(micMallocAligned((void**)&device_x, data->dim * sizeof(real), AVX_VECTOR_SIZE * sizeof(real)));

	real* device_value = NULL;
	MIC_ERROR_CHECK(micMallocAligned((void**)&device_value, (Dof_choice_end - Dof_choice_start + 1) * sizeof(real),
		AVX_VECTOR_SIZE * sizeof(real)));
	
	MIC_ERROR_CHECK(micMemcpy(device_x, x, data->dim * sizeof(real), micMemcpyHostToDevice));

	typedef struct
	{
		int dim, nno, Dof_choice_start, Dof_choice_end;
		double *x, *surplus_t, *value, *index;
	}
	Args;

	Args host_args;
	host_args.dim = data->dim;
	host_args.nno = data->nno;
	host_args.Dof_choice_start = Dof_choice_start;
	host_args.Dof_choice_end = Dof_choice_end;
	host_args.x = device_x;
	host_args.surplus_t = data->surplus_t.getData();
	host_args.value = device_value;
	host_args.index = data->index.getData();

	Args* device_args;
	MIC_ERROR_CHECK(micMallocAligned((void**)&device_args, sizeof(Args), AVX_VECTOR_SIZE * sizeof(double)));
	MIC_ERROR_CHECK(micMemcpy(device_args, &host_args, sizeof(Args), micMemcpyHostToDevice));

	if (jit)
	{
		typedef void (*Func)(
			const int dim, const int nno,
			const int Dof_choice_start, const int Dof_choice_end, const double* x,
			const double* index, const double* surplus_t, double* value);

		static Func LinearBasis_MIC_RuntimeOpt_InterpolateArray;

		if (!LinearBasis_MIC_RuntimeOpt_InterpolateArray)
		{
			// TODO
			/*LinearBasis_MIC_RuntimeOpt_InterpolateArray =
				JIT::jitCompile(data->dim, 1, "LinearBasis_MIC_RuntimeOpt_InterpolateArray_",
				(Func)LinearBasis_MIC_Generic_InterpolateArray).getFunc();*/
		}
		
		LinearBasis_MIC_RuntimeOpt_InterpolateArray(
			data->dim, data->nno, Dof_choice_start, Dof_choice_end, x,
			data->index.getData(), data->surplus_t.getData(), value);
	}
	else
	{
		MIC_ERROR_CHECK(micLaunchKernel("LinearBasis_MIC_Generic_InterpolateArray", device_args));
		MIC_ERROR_CHECK(micDeviceSynchronize());
	}

	MIC_ERROR_CHECK(micMemcpy(&value, device_value, (Dof_choice_end - Dof_choice_start + 1) * sizeof(real),
		micMemcpyDeviceToHost));
	
	MIC_ERROR_CHECK(micFree(device_args));
	MIC_ERROR_CHECK(micFree(device_x));
	MIC_ERROR_CHECK(micFree(device_value));
}

// Interpolate multiple arrays of values in continuous vector.
void Interpolator::interpolate(Data* data, const real* x, const int Dof_choice_start, const int Dof_choice_end, const int count, real* value){
	real* device_x = NULL;
	MIC_ERROR_CHECK(micMallocAligned((void**)&device_x, data->dim * count * sizeof(real), AVX_VECTOR_SIZE * sizeof(real)));
	
	real* device_value = NULL;
	MIC_ERROR_CHECK(micMallocAligned((void**)&device_value, (Dof_choice_end - Dof_choice_start + 1) * count * sizeof(real),
		AVX_VECTOR_SIZE * sizeof(real)));
	
	MIC_ERROR_CHECK(micMemcpy(device_x, x, data->dim * count * sizeof(real), micMemcpyHostToDevice));

	typedef struct
	{
		int dim, nno, Dof_choice_start, Dof_choice_end, count;
		double *x, *surplus_t, *value, *index;
	}
	Args;

	Args host_args;
	host_args.dim = data->dim;
	host_args.nno = data->nno;
	host_args.Dof_choice_start = Dof_choice_start;
	host_args.Dof_choice_end = Dof_choice_end;
	host_args.count = count;
	host_args.x = device_x;
	host_args.surplus_t = data->surplus_t.getData();
	host_args.value = device_value;
	host_args.index = data->index.getData();

	Args* device_args;
	MIC_ERROR_CHECK(micMallocAligned((void**)&device_args, sizeof(Args), AVX_VECTOR_SIZE * sizeof(double)));
	MIC_ERROR_CHECK(micMemcpy(device_args, &host_args, sizeof(Args), micMemcpyHostToDevice));

	if (jit)
	{
		typedef void (*Func)(
			const int dim, const int nno,
			const int Dof_choice_start, const int Dof_choice_end, const int count, const double* x_,
			const double* index, const double* surplus_t, double* value);

		static Func LinearBasis_MIC_RuntimeOpt_InterpolateArrayMany;

		if (!LinearBasis_MIC_RuntimeOpt_InterpolateArrayMany)
		{
			// TODO
			/*int count = Optimizer::count();
			LinearBasis_MIC_RuntimeOpt_InterpolateArrayMany =
				JIT::jitCompile(data->dim, count, "LinearBasis_MIC_RuntimeOpt_InterpolateArrayMany_",
				(Func)LinearBasis_MIC_Generic_InterpolateArrayMany).getFunc();*/
		}

		LinearBasis_MIC_RuntimeOpt_InterpolateArrayMany(
			data->dim, data->nno, Dof_choice_start, Dof_choice_end, count, x,
			data->index.getData(), data->surplus_t.getData(), value);
	}
	else
	{
		MIC_ERROR_CHECK(micLaunchKernel("LinearBasis_MIC_Generic_InterpolateArrayMany", device_args));
		MIC_ERROR_CHECK(micDeviceSynchronize());
	}

	MIC_ERROR_CHECK(micMemcpy(&value, device_value, (Dof_choice_end - Dof_choice_start + 1) * count * sizeof(real),
		micMemcpyDeviceToHost));
	
	MIC_ERROR_CHECK(micFree(device_args));
	MIC_ERROR_CHECK(micFree(device_x));
	MIC_ERROR_CHECK(micFree(device_value));
}

static map<bool, unique_ptr<Interpolator> > interp;

extern "C" Interpolator* getInterpolator(bool jit)
{
	if (!interp[jit].get())
		interp[jit].reset(new Interpolator(jit));
	
	return interp[jit].get();
}

