#ifndef SCALED_SURPLUS_H
#define SCALED_SURPLUS_H

class ScaledSurplus
{
	const int DofPerNode;
	const Matrix<double>* surplus_;
	double* value;
	const int i;
	const double temp;

	// Indicates whether an instance is a "leaf" of reduction binary tree.
	const bool leaf;

	// Indicates the instance was not modified.
	bool zero;

public :

	ScaledSurplus(const int DofPerNode_, const Matrix<double>& surplus, double* value_) : 
		DofPerNode(DofPerNode_), surplus_(&surplus), value(value_), i(0), temp(0.0), leaf(false), zero(false) { }

	ScaledSurplus(double* value_) :
		DofPerNode(0), surplus_(NULL), value(value_), i(-1), temp(0.0), leaf(true), zero(false) { }

	ScaledSurplus(double* value_, const int i_, const double temp_) :
		DofPerNode(0), surplus_(NULL), value(value_), i(i_), temp(temp_), leaf(true), zero(false) { }


	// Create a copy of origin with unassigned value pointer.
	static ScaledSurplus priv(const ScaledSurplus& origin)
	{
		ScaledSurplus result(origin.DofPerNode, *origin.surplus_, NULL);
		result.zero = true;

		return result;
	}

	ScaledSurplus& operator+=(const ScaledSurplus& other)
	{
		if (!value) value = other.value;

		if (other.leaf)
		{
			// Indicates that there is actually no temp to account.
			if (other.i == -1)
				return *this;

			const Matrix<double>& surplus = *surplus_;

			const __m512d temp64 = _mm512_set1_pd(other.temp);

			if (zero)
			{
				// First time assigment to private partial sum: no need to load & sum up with previous value.
				for (int Dof_choice = 0; Dof_choice < DOF_PER_NODE; Dof_choice += DOUBLE_VECTOR_SIZE)
				{
					__m512d surplus64 = _mm512_load_pd(&surplus(other.i, Dof_choice));
					__m512d value64 = _mm512_mul_pd(temp64, surplus64);
					_mm512_store_pd(&value[Dof_choice], value64);
				}

				zero = false;
			}
			else
			{
				for (int Dof_choice = 0; Dof_choice < DOF_PER_NODE; Dof_choice += DOUBLE_VECTOR_SIZE)
				{
					__m512d surplus64 = _mm512_load_pd(&surplus(other.i, Dof_choice));
					__m512d value64 = _mm512_load_pd(&value[Dof_choice]);
					value64 = _mm512_fmadd_pd(temp64, surplus64, value64);
					_mm512_store_pd(&value[Dof_choice], value64);
				}
			}
		}
		else
		{
			// Nothing to do if the other is zero.
			if (other.zero)
				return *this;

			if (zero)
			{
				// If this is zero, just pick up the other one without summing.
				value = other.value;
				zero = false;
				return *this;
			}

			for (int Dof_choice = 0; Dof_choice < DOF_PER_NODE; Dof_choice += DOUBLE_VECTOR_SIZE)
			{
				__m512d v = _mm512_load_pd(&value[Dof_choice]);
				__m512d v2 = _mm512_load_pd(&other.value[Dof_choice]);
				_mm512_store_pd((__m512d*)&value[Dof_choice], _mm512_add_pd(v, v2));
			}
		}

		return *this;
	}
};

#endif // SCALED_SURPLUS_H

