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
	const Matrix<int>* index, const Matrix<double>* surplus, double* value);

// Interpolate array of values.
void Interpolator::interpolate(Device* device, Data* data_,
	const int istate, const real* x, int DofPerNode, real* value)
{
	Data::Dense* data = dynamic_cast<Data::Dense*>(data_);

	int dim = data->surplus[istate].dimx();
	int nno = data->surplus[istate].dimy();

	INTERPOLATE_ARRAY(
		device, dim, nno, DofPerNode, x,
		&data->index[istate], &data->surplus[istate], value);
}

extern "C" void INTERPOLATE_ARRAY_MANY_MULTISTATE(
	Device* device, const int dim, const int nno, int DofPerNode, const int count, const double* const* x_,
	const Matrix<int>* index, const Matrix<double>* surplus, double** value);

// Interpolate multiple arrays of values, with multiple surplus states.
void Interpolator::interpolate(Device* device, Data* data_,
	const real** x, int DofPerNode, real** value)
{
	Data::Sparse* data = dynamic_cast<Data::Sparse*>(data_);

	int dim = data->surplus[0].dimx();
	int nno = data->surplus[0].dimy();

	INTERPOLATE_ARRAY_MANY_MULTISTATE(
		device, dim, nno, DofPerNode, data->nstates, x,
		&data->index[0], &data->surplus[0], value);
}

namespace NAMESPACE
{
	unique_ptr<Interpolator> interp;
}

Interpolator* Interpolator::getInstance()
{
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

