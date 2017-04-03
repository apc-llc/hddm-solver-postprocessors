#include "LinearBasis.h"
#include "Data.h"

#define CAT(kernel, name) name##_kernel
#define KERNEL(name) CAT(kernel, name)

using namespace NAMESPACE;
using namespace std;

class Device;

__global__ void KERNEL(FUNCNAME)(
	const int dim, const int nno, const int DofPerNode, const int count, const double* const* x_,
	const int* nfreqs_, const XPS* xps_, const int szxps_, const Chains* chains_, const Matrix<double>* surplus_, double** value_)
{
}

extern "C" void FUNCNAME(
	Device* device,
	const int dim, const int nno, const int DofPerNode, const int count, const double* const* x_,
	const int* nfreqs_, const XPS* xps_, const int szxps_, const Chains* chains_, const Matrix<double>* surplus_, double** value_)
{

}

