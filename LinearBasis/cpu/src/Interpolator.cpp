#include <map>
#include <memory>
#include <mutex>
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
	Device* device, const int dim, int DofPerNode, const double* x,
	const int nfreqs, const XPS* xps, const Chains* chains, const Matrix<double>* surplus, double* value);

// Interpolate array of values.
void Interpolator::interpolate(Device* device, Data* data_,
	const int istate, const real* x, int DofPerNode, real* value)
{
	Data::Sparse* data = (Data::Sparse*)data_;

	if (DofPerNode != data->TotalDof)
	{
		MPI_Process* process;
		MPI_ERR_CHECK(MPI_Process_get(&process));
		const Parameters& params = Interpolator::getInstance()->getParameters();

		process->cerr("Requested DofPerNode (%d) mismatches actual data TotalDof (%d)\n",
			DofPerNode, data->TotalDof);
		process->abort();
	}

	typedef void (*Func)(
		Device* device, const int dim, int DofPerNode, const double* x,
		const int nfreqs, const XPS* xps, const Chains* chains, const Matrix<double>* surplus, double* value);

	Func INTERPOLATE_ARRAY_RUNTIME_OPT =
		JIT::jitCompile(device, data->dim, DofPerNode,
			stringize(INTERPOLATE_ARRAY_RUNTIME_OPT) "_",
			(Func)INTERPOLATE_ARRAY).getFunc();
	
	INTERPOLATE_ARRAY_RUNTIME_OPT(
		device, data->dim, DofPerNode, x,
		data->nfreqs[istate], &data->xps[istate], &data->chains[istate], &data->surplus[istate], value);
}

extern "C" void INTERPOLATE_ARRAY_MANY_MULTISTATE(
	Device* device, const int dim, int DofPerNode, const int count, const double* const* x_,
	const int* nfreqs, const XPS* xps, const Chains* chains, const Matrix<double>* surplus, double** value);

// Interpolate multiple arrays of values, with multiple surplus states.
void Interpolator::interpolate(Device* device, Data* data_,
	const real** x, int DofPerNode, real** value)
{
	Data::Sparse* data = (Data::Sparse*)data_;

	if (DofPerNode != data->TotalDof)
	{
		MPI_Process* process;
		MPI_ERR_CHECK(MPI_Process_get(&process));
		const Parameters& params = Interpolator::getInstance()->getParameters();

		process->cerr("Requested DofPerNode (%d) mismatches actual data TotalDof (%d)\n",
			DofPerNode, data->TotalDof);
		process->abort();
	}

	typedef void (*Func)(
		Device* device, const int dim, int DofPerNode, const int count, const double* const* x_,
		const int* nfreqs, const XPS* xps, const Chains* chains, const Matrix<double>* surplus, double** value);

	Func INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT =
		JIT::jitCompile(device, data->dim, data->nstates, DofPerNode,
			stringize(INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT) "_",
			(Func)INTERPOLATE_ARRAY_MANY_MULTISTATE).getFunc();

	INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT(
		device, data->dim, DofPerNode, data->nstates, x,
		&data->nfreqs[0], &data->xps[0], &data->chains[0], &data->surplus[0], value);
}

namespace NAMESPACE
{
	unique_ptr<Interpolator> interp;
}

Interpolator* Interpolator::getInstance()
{
	static mutex mtx;
	{
		lock_guard<std::mutex> lck(mtx);
		if (!interp.get())
			interp.reset(new Interpolator(stringize(NAME)));
	}
	
	return interp.get();
}

Interpolator::~Interpolator()
{
}

extern "C" Interpolator* getInterpolator()
{
	return Interpolator::getInstance();
}

