#include <math.h>

__attribute__((always_inline)) static double IndextoCoordinate(int i, int j)
{
	double m = pow(2.0, (i - 1)) + 1;

	if (i == 1) return 0.5;

	m = pow(2.0, i) + 1;
	return (j - 1) / (m - 1.0);
}

/**
 * FlipUpBasis Basis Function
 * @param  x [X val]
 * @param  i [Level Depth]
 * @param  j [Basis Function Index]
 * @return   [Value]
 */
__attribute__((always_inline)) static double FlipUpBasis(double x, int i, int j)
{
	if (i == 1) return 1.0;
	
	double m = pow(2.0, i);
	double invm = 1.0 / m;
	double xp = IndextoCoordinate(i, j);

	// Wing
	if ((x <= invm) && (xp == invm))
	{
		return -1.0 * m * x + 2.0;
	}
	else if ((x >= 1.0 - invm) && (xp == (1.0 - invm)))
	{
		return (1.0 * m * x + (2.0 - m));
	}
	else
	{
		// Body
		if (fabs(x - xp) >= invm)
			return 0.0;
		else
			return (1 - m * fabs(x - xp));
	}
}

/**
 * Polynomial Basis Function
 * @param  x [X val]
 * @param  i [Level Depth]
 * @param  j [Basis Function Index]
 * @return   [Value]
 */
__attribute__((always_inline)) static double PolyBasis(double x, int i, int j)
{
	if (i < 3) {
		return FlipUpBasis(x, i, j);
	} else {

		double m = pow(2.0, i);
		double xp = IndextoCoordinate(i, j);

		// Wings
		if ( x <= (1.0 / m) && xp == 1. / m)  {
			return (-1.0 * m * x + 2.0);
		} else if ( x >= (1.0 - 1.0 / m )  && xp == (1.0 - 1. / m) )  {
			return (1.0 * m * x + (2.0 - m));
		} else {

			// Body
			double x1 = xp - 1.0 / m;
			double x2 = xp + 1.0 / m;
			double temp = (x - x1) * (x - x2) / ((xp - x1) * (xp - x2));

			if (temp > 0) {
				return temp;
			} else {
				return 0.0;
			}
		}
	}
}

#define STR(funcname) #funcname

void FUNCNAME(
	const int dim, const int nno,
	const int Dof_choice_start, const int Dof_choice_end, const double* x,
	const int* index, const double* surplus, double* value)
{
	// Index arrays shall be padded to AVX_VECTOR_SIZE-element
	// boundary to keep up the required alignment.
	int vdim = dim / AVX_VECTOR_SIZE;
	if (dim % AVX_VECTOR_SIZE) vdim++;
	vdim *= AVX_VECTOR_SIZE;

	const int TotalDof = Dof_choice_end - Dof_choice_start + 1;

	for (int i = 0; i < nno; i++)
	{
		int zero = 0;
		double temp = 1.0;
		for (int j = 0; j < DIM; j++)
		{
			double xp = PolyBasis(x[j], index[i * 2 * vdim + j],
				index[i * 2 * vdim + j + vdim]);
			if (xp <= 0.0)
			{
				zero = 1;
				break;
			}
			temp *= xp;
		}
		if (zero) continue;
		for (int b = Dof_choice_start, Dof_choice = b, e = Dof_choice_end; Dof_choice <= e; Dof_choice++)
			value[Dof_choice - b] += temp * surplus[i * TotalDof + Dof_choice];
	}
}

