#ifndef HDDM_PARAMETERS_H
#define HDDM_PARAMETERS_H

#include <string>
#include <vector>

namespace NAMESPACE {

struct Parameters
{
	#define ALIGN_BOOL_4
	#define REFERENCES
	#include "parameters.c"
	#undef REFERENCES
	#undef ALIGN_BOOL_4

	Parameters(const std::string& targetSuffix, const std::string& configFile = "hddm-solver.cfg");
};

} // namespace NAMESPACE

#endif // HDDM_PARAMETERS_H

