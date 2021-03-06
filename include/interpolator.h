#ifndef INTERPOLATOR_H
#define INTERPOLATOR_H

#include "parameters.h"

namespace NAMESPACE {

class Device;

enum BasisType
{
	LinearBasisType,
	PolynomialBasisType
};

class Data;

class Interpolator
{
	bool jit;

	const Parameters params;

public :
	const Parameters& getParameters() const;
	
	Interpolator(const std::string& targetSuffix, const std::string& configFile = "hddm-solver.cfg");

	static Interpolator* getInstance();

	// Interpolate array of values.
	virtual void interpolate(Device* device, Data* data,
		const int istate, const real* x, const int DofPerNode, real* value);

	// Interpolate multiple arrays of values in continuous vector, with multiple surplus states.
	virtual void interpolate(Device* device, Data* data,
		const real** x, const int DofPerNode, real** value);

	virtual ~Interpolator();
};

} // namespace NAMESPACE

#endif // INTERPOLATOR_H

