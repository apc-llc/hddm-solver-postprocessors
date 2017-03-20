#ifndef INTERPOLATOR_H
#define INTERPOLATOR_H

#include "Data.h"
#include "parameters.h"

namespace NAMESPACE {

class Device;

enum BasisType
{
	LinearBasisType,
	PolynomialBasisType
};

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
		const int istate, const real* x, const int Dof_choice_start, const int Dof_choice_end, real* value);

	// Interpolate multiple arrays of values in continuous vector, with multiple surplus states.
	virtual void interpolate(Device* device, Data* data,
		const real** x, const int Dof_choice_start, const int Dof_choice_end, real** value);

	virtual ~Interpolator();
};

} // namespace NAMESPACE

#endif // INTERPOLATOR_H

