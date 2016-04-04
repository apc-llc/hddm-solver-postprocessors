#include <malloc.h>
#include <math.h>
#include <mic_runtime.h>
#include <stdlib.h>
#include <string.h>

#define MIC_ERROR_CHECK(x) do { micError_t err = x; if (( err ) != micSuccess ) { \
	printf ("Error %d at %s :%d \n" , err, __FILE__ , __LINE__ ) ; exit(-1);\
}} while (0)

// Number of double precision elements in used AVX vector
#define AVX_VECTOR_SIZE 8

int main(int argc, char* argv[])
{
	const char* name;
	MIC_ERROR_CHECK(micGetPlatformName(&name));
	printf("CUDA-like runtime API powered by GOMP backend for %s\n", name);
	
	int num_devices;
	MIC_ERROR_CHECK(micGetDeviceCount(&num_devices));
	printf("%d %s device(s) available\n", num_devices, name);
	
	if (!num_devices) return 1;

	double dinvrandmax = 1.0 / (double)RAND_MAX;

	int dim = 8;
	int nno = 5;
	int Dof_choice = 2;

	double* host_x;
	posix_memalign((void**)&host_x, AVX_VECTOR_SIZE * sizeof(double),
		dim * sizeof(double));
	int* host_index;
	posix_memalign((void**)&host_index, AVX_VECTOR_SIZE * sizeof(double),
		2 * dim * (nno + 1) * sizeof(int));
	double* host_index_double;
	posix_memalign((void**)&host_index_double, AVX_VECTOR_SIZE * sizeof(double),
		2 * dim * (nno + 1) * sizeof(double));
	double* host_surplus_t;
	posix_memalign((void**)&host_surplus_t, AVX_VECTOR_SIZE * sizeof(double),
		dim * (nno + 1) * sizeof(double));
	double host_value = 0;

	for (int i = 0; i < dim; i++)
		host_x[i] = rand() * dinvrandmax;
	for (int i = 0; i < dim * nno + nno; i++)
		host_surplus_t[i] = (double) rand() / RAND_MAX;
	for (int i = 0; i < 2 * dim * (nno + 1); i++)
	{
		host_index[i] = 1;
		host_index_double[i] = (double)host_index[i];
	}

	typedef struct
	{
		int dim, nno, Dof_choice;
		double *x, *surplus_t, *value, *index;
	}
	Args;

	Args host_args;
	host_args.dim = dim;
	host_args.nno = nno;
	host_args.Dof_choice = Dof_choice;

	double* device_x;
	MIC_ERROR_CHECK(micMallocAligned((void**)&device_x, dim * sizeof(double), AVX_VECTOR_SIZE * sizeof(double)));
	MIC_ERROR_CHECK(micMemcpy(device_x, host_x, dim * sizeof(double), micMemcpyHostToDevice));
	host_args.x = device_x;
	double* device_index;
	MIC_ERROR_CHECK(micMallocAligned((void**)&device_index, 2 * dim * (nno + 1) * sizeof(double), AVX_VECTOR_SIZE * sizeof(double)));
	MIC_ERROR_CHECK(micMemcpy(device_index, host_index_double, 2 * dim * (nno + 1) * sizeof(double), micMemcpyHostToDevice));
	host_args.index = device_index;
	double* device_surplus_t;
	MIC_ERROR_CHECK(micMallocAligned((void**)&device_surplus_t, dim * (nno + 1) * sizeof(double), AVX_VECTOR_SIZE * sizeof(double)));
	MIC_ERROR_CHECK(micMemcpy(device_surplus_t, host_surplus_t, dim * (nno + 1) * sizeof(double), micMemcpyHostToDevice));
	host_args.surplus_t = device_surplus_t;
	double* device_value;
	MIC_ERROR_CHECK(micMallocAligned((void**)&device_value, sizeof(double), AVX_VECTOR_SIZE * sizeof(double)));
	MIC_ERROR_CHECK(micMemcpy(device_value, &host_value, sizeof(double), micMemcpyHostToDevice));
	host_args.value = device_value;

	Args* device_args;
	MIC_ERROR_CHECK(micMallocAligned((void**)&device_args, sizeof(Args), AVX_VECTOR_SIZE * sizeof(double)));
	MIC_ERROR_CHECK(micMemcpy(device_args, &host_args, sizeof(Args), micMemcpyHostToDevice));

	MIC_ERROR_CHECK(micLaunchKernel("LinearBasis_MIC_Generic_InterpolateValue", device_args));
	MIC_ERROR_CHECK(micDeviceSynchronize());

	double host_value_result;
	MIC_ERROR_CHECK(micMemcpy(&host_value_result, device_value, sizeof(double), micMemcpyDeviceToHost));

	void LinearBasis_CPU_Generic_InterpolateValue(
		const int dim, const int nno,
		const int Dof_choice, const double* x,
		const int* index, const double* surplus_t, double* value_);

	LinearBasis_CPU_Generic_InterpolateValue(dim, nno, Dof_choice, host_x, host_index, host_surplus_t, &host_value);

	// Check results.
	if (fabs(host_value_result - host_value) > 1e-6)
	{
		printf("Results mismatch: %15.10e != %15.10e\n",
			host_value_result, host_value);
	}
	else
		printf("Result is correct\n");

	MIC_ERROR_CHECK(micFreeAligned(device_x));
	MIC_ERROR_CHECK(micFreeAligned(device_index));
	MIC_ERROR_CHECK(micFreeAligned(device_surplus_t));
	MIC_ERROR_CHECK(micFreeAligned(device_value));

	free(host_x);
	free(host_index);
	free(host_index_double);
	free(host_surplus_t);
	
	return 0;
}

