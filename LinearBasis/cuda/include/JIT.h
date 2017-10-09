#ifndef JIT_H
#define JIT_H

#include "Data.h"
#include "InterpolateKernel.h"

namespace NAMESPACE {

class Device;

typedef void (*InterpolateArrayFunc)(
	Device* device,
	const int dim, const int nno,
	const int DofPerNode, const real* x,
	const int nfreqs, const XPS::Device* xps, const int szxps,
	const Chains::Device* chains, const Matrix<real>::Device* surplus,
	real* value);

typedef void (*InterpolateArrayManyMultistateFunc)(
	Device* device,
	const int dim, const int* nnos,
	const int DofPerNode, const int count, const real* const* x,
	const int* nfreqs, const XPS::Device* xps, const int* szxps,
	const Chains::Device* chains, const Matrix<real>::Device* surplus,
	real** value);

typedef InterpolateKernel<InterpolateArrayFunc> InterpolateArrayKernel;
typedef InterpolateKernel<InterpolateArrayManyMultistateFunc> InterpolateArrayManyMultistateKernel;

class JIT
{
public :
	static InterpolateArrayKernel& jitCompile(
		Device* device, int dim, int DofPerNode,
		const std::string& funcnameTemplate, InterpolateArrayFunc fallbackFunc);
	static InterpolateArrayManyMultistateKernel& jitCompile(
		Device* device, int dim, int count, int DofPerNode,
		const std::string& funcnameTemplate, InterpolateArrayManyMultistateFunc fallbackFunc);

	template<typename K, typename F>
	static K& jitCompile(
		Device* device, int dim, int count, int DofPerNode,
		const std::string& funcnameTemplate, F fallbackFunc);
};

} // namespace NAMESPACE

#endif // JIT_H

