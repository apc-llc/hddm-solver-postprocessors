#include "LinearBasis.h"
#include "Data.h"

using namespace NAMESPACE;

class Device;

extern "C" void FUNCNAME(
	Device* device,
	const int dim, const int DofPerNode, const int count, const double* const* x_,
	const Matrix<int>* index_, const Matrix<double>* surplus_, double** value_)
{
	// Index arrays shall be padded to AVX_VECTOR_SIZE-element
	// boundary to keep up the required alignment.
	int vdim = dim / AVX_VECTOR_SIZE;
	if (dim % AVX_VECTOR_SIZE) vdim++;
	vdim *= AVX_VECTOR_SIZE;

	for (int many = 0; many < count; many++)
	{
		const double* x = x_[many];
		const Matrix<int>& index = index_[many];
		const Matrix<double>& surplus = surplus_[many];
		double* value = value_[many];

		int nno = surplus.dimy();

		memset(value, 0, sizeof(double) * DOF_PER_NODE);

		for (int i = 0; i < nno; i++)
		{
			double temp = 1.0;
			for (int j = 0; j < DIM; j++)
			{
				double xp = LinearBasis(x[j], index(i, j), index(i, + j + vdim));
				if (xp <= 0.0)
					goto zero;

				temp *= xp;
			}

			for (int Dof_choice = 0; Dof_choice < DOF_PER_NODE; Dof_choice++)
				value[Dof_choice] += temp * surplus(i, Dof_choice);

			zero : continue;
		}
	}
}

