#ifndef INTERPOLATOR_H
#define INTERPOLATOR_H

#include "Data.h"
#include "parameters.h"

namespace NAMESPACE {

class Device;

enum Basis
{
	LinearBasis,
	PolynomialBasis
};

class Interpolator
{
	bool jit;

	const Parameters params;

public :
	const Parameters& getParameters() const;
	
	Interpolator(const std::string& targetSuffix, const std::string& configFile = "hddm-solver.cfg");

	static Interpolator* getInstance();

	// Interpolate a single value.
	virtual void interpolate(Device* device, Data* data,
		const int istate, const real* x, const int Dof_choice, real& value);

	// Interpolate array of values.
	virtual void interpolate(Device* device, Data* data,
		const int istate, const real* x, const int Dof_choice_start, const int Dof_choice_end, real* value);

	// Interpolate multiple arrays of values in continuous vector, with single surplus state.
	virtual void interpolate(Device* device, Data* data,
		const int istate, const real* x, const int Dof_choice_start, const int Dof_choice_end, const int count, real* value);

	// Interpolate multiple arrays of values in continuous vector, with multiple surplus states.
	virtual void interpolate(Device* device, Data* data,
		const real** x, const int Dof_choice_start, const int Dof_choice_end, real** value);
};

} // namespace NAMESPACE

#endif // INTERPOLATOR_H

