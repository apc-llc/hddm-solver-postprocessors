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

{ }

extern "C" void INTERPOLATE_ARRAY(
	Device* device, const int dim, const int nno, int DofPerNode, const double* x,
	const int nfreqs, const XPS* xps, const Chains* chains, const Matrix<double>* surplus, double* value);

// Interpolate array of values.
void Interpolator::interpolate(Device* device, Data* data,
	const int istate, const real* x, int DofPerNode, real* value)
{
	typedef void (*Func)(
		Device* device, const int dim, const int nno, int DofPerNode, const double* x,
		const int nfreqs, const XPS* xps, const Chains* chains, const Matrix<double>* surplus, double* value);

	static Func INTERPOLATE_ARRAY_RUNTIME_OPT;

	int dim = data->surplus[istate].dimx();
	int nno = data->surplus[istate].dimy();

	if (!INTERPOLATE_ARRAY_RUNTIME_OPT)
	{
		INTERPOLATE_ARRAY_RUNTIME_OPT =
			JIT::jitCompile(dim, nno, DofPerNode,
			stringize(INTERPOLATE_ARRAY_RUNTIME_OPT) "_",
			(Func)INTERPOLATE_ARRAY).getFunc();
	}
	
	INTERPOLATE_ARRAY_RUNTIME_OPT(
		device, dim, nno, DofPerNode, x,
		data->nfreqs[istate], &data->xps[istate], &data->chains[istate], &data->surplus[istate], value);
}

extern "C" void INTERPOLATE_ARRAY_MANY_MULTISTATE(
	Device* device, const int dim, const int nno, int DofPerNode, const int count, const double* const* x_,
	const int* nfreqs, const XPS* xps, const Chains* chains, const Matrix<double>* surplus, double** value);

// Interpolate multiple arrays of values, with multiple surplus states.
void Interpolator::interpolate(Device* device, Data* data,
	const real** x, int DofPerNode, real** value)
{
	typedef void (*Func)(
		Device* device, const int dim, const int nno, int DofPerNode, const int count, const double* const* x_,
		const int* nfreqs, const XPS* xps, const Chains* chains, const Matrix<double>* surplus, double** value);

	static Func INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT;

	int dim = data->surplus[0].dimx();
	int nno = data->surplus[0].dimy();

	if (!INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT)
	{
		INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT =
			JIT::jitCompile(dim, data->nstates, nno, DofPerNode,
			stringize(INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT) "_",
			(Func)INTERPOLATE_ARRAY_MANY_MULTISTATE).getFunc();
	}

	INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT(
		device, dim, nno, DofPerNode, data->nstates, x,
		&data->nfreqs[0], &data->xps[0], &data->chains[0], &data->surplus[0], value);
}

Interpolator* Interpolator::getInstance()
{
	static unique_ptr<Interpolator> interp;

	if (!interp.get())
		interp.reset(new Interpolator(stringize(NAME)));
	
	return interp.get();
}

Interpolator::~Interpolator()
{
}

extern "C" Interpolator* getInterpolator()
{
	return Interpolator::getInstance();
}

