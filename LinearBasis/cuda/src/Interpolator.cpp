#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "interpolator.h"
#include "JIT.h"

#define str(x) #x
#define stringize(x) str(x)

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

{ }

extern "C" void INTERPOLATE_ARRAY(
	Device* device, const int dim, const int nno, int DofPerNode, const double* x,
	const int nfreqs, const XPS* xps, const int szxps, const Chains* chains,
	const Matrix<double>::Device* surplus, double* value);

// Interpolate array of values.
void Interpolator::interpolate(Device* device, Data* data,
	const int istate, const real* x, int DofPerNode, real* value)
{
	typedef void (*Func)(
		Device* device, const int dim, const int nno, int DofPerNode, const double* x,
		const int nfreqs, const XPS::Device* xps, const int szxps, const Chains::Device* chains,
		const Matrix<double>::Device* surplus, double* value);

	int dim = data->host.getSurplus(istate)->dimx();
	int nno = data->host.getSurplus(istate)->dimy();

	Func INTERPOLATE_ARRAY_RUNTIME_OPT =
		JIT::jitCompile(device, dim, nno, DofPerNode,
			stringize(INTERPOLATE_ARRAY_RUNTIME_OPT) "_",
			(Func)INTERPOLATE_ARRAY).getFunc();
	
	INTERPOLATE_ARRAY_RUNTIME_OPT(
		device, dim, nno, DofPerNode, x,
		*data->host.getNfreqs(istate), data->device.getXPS(istate), *data->host.getSzXPS(istate),
		data->device.getChains(istate), data->device.getSurplus(istate), value);
}

extern "C" void INTERPOLATE_ARRAY_MANY_MULTISTATE(
	Device* device, const int dim, const int nno, int DofPerNode, const int count, const double* const* x_,
	const int* nfreqs, const XPS* xps, const int* szxps, const Chains* chains,
	const Matrix<double>::Device* surplus, double** value);

// Interpolate multiple arrays of values, with multiple surplus states.
void Interpolator::interpolate(Device* device, Data* data,
	const real** x, int DofPerNode, real** value)
{
	typedef void (*Func)(
		Device* device, const int dim, const int nno, int DofPerNode, const int count, const double* const* x_,
		const int* nfreqs, const XPS::Device* xps, const int* szxps, const Chains::Device* chains,
		const Matrix<double>::Device* surplus, double** value);

	int dim = data->host.getSurplus(0)->dimx();
	int nno = data->host.getSurplus(0)->dimy();

	Func INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT =
		JIT::jitCompile(device, dim, data->nstates, nno, DofPerNode,
			stringize(INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT) "_",
			(Func)INTERPOLATE_ARRAY_MANY_MULTISTATE).getFunc();

	INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT(
		device, dim, nno, DofPerNode, data->nstates, x,
		data->device.getNfreqs(0), data->device.getXPS(0), data->host.getSzXPS(0),
		data->device.getChains(0), data->device.getSurplus(0), value);
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

