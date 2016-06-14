#include "check.h"
#include "optimizer.h"

#include <string>

const Parameters& Interpolator::getParameters() const { return params; }
	
Interpolator::Interpolator(const std::string& configFile) : params(configFile) { }

Interpolator* Interpolator::getInstance()
{
	static Interpolator optimizer;	
	return &optimizer;
}

static const Basis basis = LinearBasis;

extern "C" Basis getBasis() { return basis; }

