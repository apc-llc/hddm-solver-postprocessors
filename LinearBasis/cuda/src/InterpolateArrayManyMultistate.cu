#include "LinearBasis.h"
#include "Data.h"

#include <algorithm> // min & max

#define CAT(kernel, name) name##_kernel
#define KERNEL(name) CAT(kernel, name)

using namespace NAMESPACE;
using namespace std;

class Device;

__global__ void KERNEL(FUNCNAME)(
	const int dim, const int nno, const int DofPerNode, const int count, const double* const* x_,
	const int* nfreqs_, const XPS::Device* xps_, const int* szxps_, double* xpv_, const Chains::Device* chains_,
	const Matrix<double>::Device* surplus_, double** value_)
{
	if ((blockIdx.x == 0) && (threadIdx.x == 0))
	{
		for (int many = 0; many < COUNT; many++)
		{
			const double* x = x_[many];
			const int& nfreqs = nfreqs_[many];
			const XPS::Device& xps = xps_[many];
			const Chains::Device& chains = chains_[many];
			const Matrix<double>::Device& surplus = surplus_[many];
			double* value = value_[many];

			// Loop to calculate all unique xp values.
			for (int i = 0, e = szxps_[many]; i < e; i++)
			{
				const Index<uint16_t>& index = xps(i);
				const uint32_t& j = index.index;
				double xp = LinearBasis(x[j], index.i, index.j);
				xpv_[i] = fmax(0.0, xp);
			}

			// Zero the values array.
			memset(value, 0, sizeof(double) * DOF_PER_NODE);

			// Loop to calculate scaled surplus product.
			for (int i = 0, ichain = 0; i < NNO; i++, ichain += nfreqs)
			{
				double temp = 1.0;
				for (int ifreq = 0; ifreq < nfreqs; ifreq++)
				{
					// Early exit for shorter chains.
					int32_t idx = chains(ichain + ifreq);
					if (!idx) break;

					temp *= xpv_[idx];
					if (!temp) goto next;
				}

				for (int Dof_choice = 0; Dof_choice < DOF_PER_NODE; Dof_choice++)
					value[Dof_choice] += temp * surplus(i, Dof_choice);
			
			next :

				continue;
			}
		}
	}
}

extern "C" void FUNCNAME(
	Device* device,
	const int dim, const int nno, const int DofPerNode, const int count, const double* const* x_,
	const int* nfreqs_, const XPS::Device* xps_, const int* szxps_, const Chains::Device* chains_,
	const Matrix<double>::Device* surplus_, double** value_)
{
	double** xDev = NULL;
	CUDA_ERR_CHECK(cudaMalloc(&xDev, sizeof(double*) * count));
	vector<double*> x(count);
	Matrix<double>::Device xMatrixDev(count, dim);
	for (int i = 0; i < count; i++)
	{
		CUDA_ERR_CHECK(cudaMemcpy(xMatrixDev.getData(i, 0), x_[i], sizeof(double) * dim,
			cudaMemcpyHostToDevice));
		x[i] = xMatrixDev.getData(i, 0);
	}
	CUDA_ERR_CHECK(cudaMemcpy(&xDev[0], &x[0], sizeof(double*) * count,
		cudaMemcpyHostToDevice));

	double** valueDev = NULL;
	CUDA_ERR_CHECK(cudaMalloc(&valueDev, sizeof(double*) * count));
	vector<double*> value(count);
	Matrix<double>::Device valueMatrixDev(count, dim);
	for (int i = 0; i < count; i++)
	{
		value[i] = valueMatrixDev.getData(i, 0);
		CUDA_ERR_CHECK(cudaMemset(value[i], 0, sizeof(double) * dim));
	}
	CUDA_ERR_CHECK(cudaMemcpy(&valueDev[0], &value[0], sizeof(double*) * count,
		cudaMemcpyHostToDevice));

	int* szxpsDev = NULL;
	CUDA_ERR_CHECK(cudaMalloc(&szxpsDev, sizeof(int) * count));
	CUDA_ERR_CHECK(cudaMemcpy(&szxpsDev[0], &szxps_[0], sizeof(int) * count,
		cudaMemcpyHostToDevice)); 

	// Prepare the XPV buffer vector sized to max xps.size()
	// across all states.
	int szxpv = 0;
	for (int many = 0; many < count; many++)
		szxpv = max(szxpv, szxps_[many]);
	Vector<double>::Device xpvDev(szxpv);
	double* xpv_ = xpvDev.getData();
	
	// Launch the kernel.
	int szblock = 128;
	int nblocks = nno / szblock;
	if (nno % szblock) nblocks++;
	KERNEL(FUNCNAME)<<<nblocks, szblock>>>(dim, nno, DofPerNode, count, xDev,
		nfreqs_, xps_, szxpsDev, xpv_, chains_, surplus_, valueDev);

	CUDA_ERR_CHECK(cudaDeviceSynchronize());
	
	for (int i = 0; i < count; i++)
		CUDA_ERR_CHECK(cudaMemcpy(value_[i], valueMatrixDev.getData(i, 0), sizeof(double) * dim,
			cudaMemcpyDeviceToHost));
	
	CUDA_ERR_CHECK(cudaFree(xDev));
	CUDA_ERR_CHECK(cudaFree(valueDev));
	CUDA_ERR_CHECK(cudaFree(szxpsDev));
}

