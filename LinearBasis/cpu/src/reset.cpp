#include "interpolator.h"

#include <memory>

using namespace std;

namespace NAMESPACE
{
	extern unique_ptr<Interpolator> interp;
}

using namespace NAMESPACE;

extern "C" void resetInterpolator()
{
	interp = nullptr;
}

