#ifndef INTERPOLATOR_H
#define INTERPOLATOR_H

#include "parameters.h"

enum Basis
{
	LinearBasis,
	PolynomialBasis
};

class Interpolator
{
	const Parameters params;

public :
	const Parameters& getParameters() const;
	
	Interpolator(const std::string& configFile = "hddm-solver.cfg");

	static Interpolator* getInstance();
};

#endif // INTERPOLATOR_H

