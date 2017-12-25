#include <iostream>
#include <sstream>
#include <time.h>

#include "cpu/include/instrset.h"
#include "gtest/gtest.h"

using namespace std;

#define EPSILON 0.001

// The number of repeations to be averaged when
// calculating time/performance.
static int ntests = 1;

// Get the timer value.
static void get_time(volatile double* ret)
{
	volatile struct timespec val;
	clock_gettime(CLOCK_REALTIME, (struct timespec*)&val);
	*ret = (double)0.000000001 * val.tv_nsec + val.tv_sec;
}

static void init(double* input, int dim)
{
	ASSERT_EQ(dim, 59);

	input[0] = 0.0084745762711864406;
	input[1] = 0.016949152542372881;
	input[2] = 0.025423728813559324;
	input[3] = 0.033898305084745763;
	input[4] = 0.042372881355932202;
	input[5] = 0.050847457627118647;
	input[6] = 0.059322033898305086;
	input[7] = 0.067796610169491525;
	input[8] = 0.076271186440677971;
	input[9] = 0.084745762711864403;
	input[10] = 0.093220338983050849;
	input[11] = 0.10169491525423729;
	input[12] = 0.11016949152542373;
	input[13] = 0.11864406779661017;
	input[14] = 0.1271186440677966;
	input[15] = 0.13559322033898305;
	input[16] = 0.1440677966101695;
	input[17] = 0.15254237288135594;
	input[18] = 0.16101694915254236;
	input[19] = 0.16949152542372881;
	input[20] = 0.17796610169491525;
	input[21] = 0.1864406779661017;
	input[22] = 0.19491525423728814;
	input[23] = 0.20338983050847459;
	input[24] = 0.21186440677966101;
	input[25] = 0.22033898305084745;
	input[26] = 0.2288135593220339;
	input[27] = 0.23728813559322035;
	input[28] = 0.24576271186440679;
	input[29] = 0.25423728813559321;
	input[30] = 0.26271186440677968;
	input[31] = 0.2711864406779661;
	input[32] = 0.27966101694915252;
	input[33] = 0.28813559322033899;
	input[34] = 0.29661016949152541;
	input[35] = 0.30508474576271188;
	input[36] = 0.3135593220338983;
	input[37] = 0.32203389830508472;
	input[38] = 0.33050847457627119;
	input[39] = 0.33898305084745761;
	input[40] = 0.34745762711864409;
	input[41] = 0.3559322033898305;
	input[42] = 0.36440677966101692;
	input[43] = 0.3728813559322034;
	input[44] = 0.38135593220338981;
	input[45] = 0.38983050847457629;
	input[46] = 0.39830508474576271;
	input[47] = 0.40677966101694918;
	input[48] = 0.4152542372881356;
	input[49] = 0.42372881355932202;
	input[50] = 0.43220338983050849;
	input[51] = 0.44067796610169491;
	input[52] = 0.44915254237288138;
	input[53] = 0.4576271186440678;
	input[54] = 0.46610169491525422;
	input[55] = 0.47457627118644069;
	input[56] = 0.48305084745762711;
	input[57] = 0.49152542372881358;
	input[58] = 0.5;
}

static void check(double* result, int TotalDof)
{
	ASSERT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.11149;
	expected[1] = -0.379729;
	expected[2] = -0.515032;
	expected[3] = -0.627058;
	expected[4] = -0.717594;
	expected[5] = -0.788343;
	expected[6] = -0.84101;
	expected[7] = -0.877075;
	expected[8] = -0.897992;
	expected[9] = -0.906151;
	expected[10] = -0.902999;
	expected[11] = -0.889663;
	expected[12] = -0.866978;
	expected[13] = -0.835836;
	expected[14] = -0.796796;
	expected[15] = -0.750255;
	expected[16] = -0.696419;
	expected[17] = -0.634805;
	expected[18] = -0.564002;
	expected[19] = -0.483346;
	expected[20] = -0.404664;
	expected[21] = -0.321162;
	expected[22] = -0.233004;
	expected[23] = -0.140538;
	expected[24] = -0.0436552;
	expected[25] = 0.056987;
	expected[26] = 0.161857;
	expected[27] = 0.271403;
	expected[28] = 0.387293;
	expected[29] = 0.510103;
	expected[30] = 0.640659;
	expected[31] = 0.779955;
	expected[32] = 0.928737;
	expected[33] = 1.08761;
	expected[34] = 1.25646;
	expected[35] = 1.42638;
	expected[36] = 1.59561;
	expected[37] = 1.76706;
	expected[38] = 1.93705;
	expected[39] = 2.10139;
	expected[40] = 2.23533;
	expected[41] = 2.35324;
	expected[42] = 2.45088;
	expected[43] = 2.5237;
	expected[44] = 2.5676;
	expected[45] = 2.57851;
	expected[46] = 2.55415;
	expected[47] = 2.49646;
	expected[48] = 2.40772;
	expected[49] = 2.28923;
	expected[50] = 2.14207;
	expected[51] = 1.99248;
	expected[52] = 1.81894;
	expected[53] = 1.63186;
	expected[54] = 1.42767;
	expected[55] = 1.2033;
	expected[56] = 0.956931;
	expected[57] = 0.685362;
	expected[58] = 0.380448;
	expected[59] = 99.5823;
	expected[60] = 103.628;
	expected[61] = 105.428;
	expected[62] = 106.975;
	expected[63] = 108.293;
	expected[64] = 109.395;
	expected[65] = 110.328;
	expected[66] = 111.11;
	expected[67] = 111.768;
	expected[68] = 112.368;
	expected[69] = 112.929;
	expected[70] = 113.48;
	expected[71] = 114.024;
	expected[72] = 114.568;
	expected[73] = 115.111;
	expected[74] = 115.656;
	expected[75] = 116.213;
	expected[76] = 116.772;
	expected[77] = 117.328;
	expected[78] = 118.059;
	expected[79] = 119.267;
	expected[80] = 120.681;
	expected[81] = 122.436;
	expected[82] = 124.759;
	expected[83] = 127.921;
	expected[84] = 131.505;
	expected[85] = 135.198;
	expected[86] = 139.349;
	expected[87] = 144.185;
	expected[88] = 149.627;
	expected[89] = 155.468;
	expected[90] = 162.182;
	expected[91] = 169.891;
	expected[92] = 178.717;
	expected[93] = 188.678;
	expected[94] = 198.701;
	expected[95] = 203.08;
	expected[96] = 203.372;
	expected[97] = 200.64;
	expected[98] = 195.643;
	expected[99] = 189.227;
	expected[100] = 178.785;
	expected[101] = 165.167;
	expected[102] = 149.245;
	expected[103] = 131.804;
	expected[104] = 113.684;
	expected[105] = 95.5773;
	expected[106] = 77.7606;
	expected[107] = 60.6308;
	expected[108] = 44.8955;
	expected[109] = 31.3791;
	expected[110] = 21.7669;
	expected[111] = 17.7072;
	expected[112] = 15.3923;
	expected[113] = 13.2127;
	expected[114] = 11.0216;
	expected[115] = 8.77193;
	expected[116] = 6.4376;
	expected[117] = 3.9354;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

#define NAMESPACE gold
#include "gold/include/Data.h"
#define INTERPOLATE_ARRAY LinearBasis_gold_Generic_InterpolateArray

namespace gold
{
	class Device;

	extern "C" void INTERPOLATE_ARRAY(
		Device* device,
		const int dim, const int DofPerNode, const double* x,
		const Matrix<int>* index, const Matrix<double>* surplus, double* value);

	class GoogleTest
	{
	public :

		static void run()
		{
			using namespace gold;

			Data::Dense data(1);
			data.load("surplus.plt", 0);

			Vector<double> x(data.dim);
			init(&x(0), data.dim);

			Vector<double> result(data.TotalDof);

			Device* device = NULL;

			volatile double start, finish;
			get_time(&start);
			for (int i = 0; i < ntests; i++)
			{
				INTERPOLATE_ARRAY(device, data.dim, data.TotalDof, &x(0),
					&data.index[0], &data.surplus[0], &result(0));
			}
			get_time(&finish);
			
			cout << "time = " << (finish - start) / ntests <<
				" sec (averaged from " << ntests << " tests)" << endl;

			check(&result(0), data.TotalDof);
		}
	};
}

TEST(InterpolateArray, gold)
{
	gold::GoogleTest::run();
}

#undef NAMESPACE
#define NAMESPACE x86
#undef DATA_H
#include "x86/include/Data.h"
#undef JIT_H
#include "x86/include/JIT.h"
#undef INTERPOLATE_ARRAY
#define INTERPOLATE_ARRAY LinearBasis_x86_Generic_InterpolateArray
#undef INTERPOLATE_ARRAY_RUNTIME_OPT
#define INTERPOLATE_ARRAY_RUNTIME_OPT LinearBasis_x86_RuntimeOpt_InterpolateArray

namespace x86
{
	class Device;

	extern "C" void INTERPOLATE_ARRAY(
		Device* device,
		const int dim, const int DofPerNode, const double* x,
		const int nfreqs, const XPS* xps, const Chains* chains, const Matrix<double>* surplus, double* value);

	class GoogleTest
	{
	public :

		static void run()
		{
			using namespace x86;

			Data::Sparse data(1);
			data.load("surplus.plt", 0);

			Vector<double> x(data.dim);
			init(&x(0), data.dim);

			Vector<double> result(data.TotalDof);

			Device* device = NULL;

			volatile double start, finish;
			get_time(&start);
			for (int i = 0; i < ntests; i++)
			{
				INTERPOLATE_ARRAY(device, data.dim, data.TotalDof, &x(0),
					data.nfreqs[0], &data.xps[0], &data.chains[0], &data.surplus[0], &result(0));
			}
			get_time(&finish);
		
			cout << "time = " << (finish - start) / ntests <<
				" sec (averaged from " << ntests << " tests)" << endl;

			check(&result(0), data.TotalDof);
		}
	};
}

TEST(InterpolateArray, x86)
{
	x86::GoogleTest::run();
}

#undef NAMESPACE
#define NAMESPACE avx
#undef DOUBLE_VECTOR_SIZE
#define DOUBLE_VECTOR_SIZE 4
#undef DATA_H
#include "avx/include/Data.h"
#undef JIT_H
#include "avx/include/JIT.h"
#undef INTERPOLATE_ARRAY
#define INTERPOLATE_ARRAY LinearBasis_avx_Generic_InterpolateArray
#undef INTERPOLATE_ARRAY_RUNTIME_OPT
#define INTERPOLATE_ARRAY_RUNTIME_OPT LinearBasis_avx_RuntimeOpt_InterpolateArray

namespace avx
{
	class Device;

	extern "C" void INTERPOLATE_ARRAY(
		Device* device,
		const int dim, const int DofPerNode, const double* x,
		const int nfreqs, const XPS* xps, const Chains* chains, const Matrix<double>* surplus, double* value);

	class GoogleTest
	{
	public :

		static void run()
		{
			using namespace avx;

			Data::Sparse data(1);
			data.load("surplus.plt", 0);

			Vector<double> x(data.dim);
			init(&x(0), data.dim);

			Vector<double> result(data.TotalDof);

			Device* device = NULL;

			volatile double start, finish;
			get_time(&start);
			for (int i = 0; i < ntests; i++)
			{
				INTERPOLATE_ARRAY(device, data.dim, data.TotalDof, &x(0),
					data.nfreqs[0], &data.xps[0], &data.chains[0], &data.surplus[0], &result(0));
			}
			get_time(&finish);

			cout << "time = " << (finish - start) / ntests <<
				" sec (averaged from " << ntests << " tests)" << endl;

			check(&result(0), data.TotalDof);
		}
	};
}

TEST(InterpolateArray, avx)
{
	avx::GoogleTest::run();
}

#undef NAMESPACE
#define NAMESPACE avx2
#undef DOUBLE_VECTOR_SIZE
#define DOUBLE_VECTOR_SIZE 4
#undef DATA_H
#include "avx2/include/Data.h"
#undef JIT_H
#include "avx2/include/JIT.h"
#undef INTERPOLATE_ARRAY
#define INTERPOLATE_ARRAY LinearBasis_avx2_Generic_InterpolateArray
#undef INTERPOLATE_ARRAY_RUNTIME_OPT
#define INTERPOLATE_ARRAY_RUNTIME_OPT LinearBasis_avx2_RuntimeOpt_InterpolateArray

namespace avx2
{
	class Device;

	extern "C" void INTERPOLATE_ARRAY(
		Device* device,
		const int dim, const int DofPerNode, const double* x,
		const int nfreqs, const XPS* xps, const Chains* chains, const Matrix<double>* surplus, double* value);

	class GoogleTest
	{
	public :

		static void run()
		{
			using namespace avx2;

			Data::Sparse data(1);
			data.load("surplus.plt", 0);

			Vector<double> x(data.dim);
			init(&x(0), data.dim);

			Vector<double> result(data.TotalDof);

			Device* device = NULL;

			volatile double start, finish;
			get_time(&start);
			for (int i = 0; i < ntests; i++)
			{
				INTERPOLATE_ARRAY(device, data.dim, data.TotalDof, &x(0),
					data.nfreqs[0], &data.xps[0], &data.chains[0], &data.surplus[0], &result(0));
			}
			get_time(&finish);

			cout << "time = " << (finish - start) / ntests <<
				" sec (averaged from " << ntests << " tests)" << endl;

			check(&result(0), data.TotalDof);
		}
	};
}

TEST(InterpolateArray, avx2)
{
	avx2::GoogleTest::run();
}

#undef NAMESPACE
#define NAMESPACE avx512
#undef DOUBLE_VECTOR_SIZE
#define DOUBLE_VECTOR_SIZE 8
#undef DATA_H
#include "avx512/include/Data.h"
#undef JIT_H
#include "avx512/include/JIT.h"
#undef INTERPOLATE_ARRAY
#define INTERPOLATE_ARRAY LinearBasis_avx512_Generic_InterpolateArray
#undef INTERPOLATE_ARRAY_RUNTIME_OPT
#define INTERPOLATE_ARRAY_RUNTIME_OPT LinearBasis_avx512_RuntimeOpt_InterpolateArray
#undef DEVICES_H
#include "avx512/include/Devices.h"

namespace avx512
{
	class Device;

	extern "C" void INTERPOLATE_ARRAY(
		Device* device,
		const int dim, const int DofPerNode, const double* x,
		const int nfreqs, const XPS* xps, const Chains* chains, const Matrix<double>* surplus, double* value);

	class GoogleTest
	{
	public :

		static void run()
		{
			using namespace avx512;

			Data::Sparse data(1);
			data.load("surplus.plt", 0);

			Vector<double> x(data.dim);
			init(&x(0), data.dim);

			Vector<double> result(data.TotalDof);

			Device* device = avx512::tryAcquireDevice();
			ASSERT_TRUE(device != NULL);

			volatile double start, finish;
			get_time(&start);
			for (int i = 0; i < ntests; i++)
			{
				INTERPOLATE_ARRAY(device, data.dim, data.TotalDof, &x(0),
					data.nfreqs[0], &data.xps[0], &data.chains[0], &data.surplus[0], &result(0));
			}
			get_time(&finish);

			cout << "time = " << (finish - start) / ntests <<
				" sec (averaged from " << ntests << " tests)" << endl;

			check(&result(0), data.TotalDof);
		}
	};
}

TEST(InterpolateArray, avx512)
{
	avx512::GoogleTest::run();
}

#if defined(NVCC)
#undef NAMESPACE
#define NAMESPACE cuda
#undef DATA_H
#include "cuda/include/Data.h"
#undef JIT_H
#include "cuda/include/JIT.h"
#undef INTERPOLATE_ARRAY
#define INTERPOLATE_ARRAY LinearBasis_cuda_Generic_InterpolateArray
#undef INTERPOLATE_ARRAY_RUNTIME_OPT
#define INTERPOLATE_ARRAY_RUNTIME_OPT LinearBasis_cuda_RuntimeOpt_InterpolateArray
#undef DEVICES_H
#include "cuda/include/Devices.h"

namespace cuda
{
	extern "C" void INTERPOLATE_ARRAY(
		Device* device,
		const int dim, const int nno, const int DofPerNode, const double* x,
		const int nfreqs, const XPS::Device* xps_, const int szxps, const Chains::Device* chains_,
		const Matrix<double>::Device* surplus_, double* value);

	class GoogleTest
	{
	public :

		static void run()
		{
			using namespace cuda;

			Vector<double>::Host result;

			Device* device = cuda::tryAcquireDevice();
			ASSERT_TRUE(device != NULL);

			{
				Data data(1);
				data.load("surplus.plt", 0);
				result.resize(data.TotalDof);

				Vector<double>::Host x(data.dim);
				init(&x(0), data.dim);

				// Run once without timing to do all CUDA-specific internal initializations.
				INTERPOLATE_ARRAY(device, data.dim,
					data.host.getSurplus(0)->dimy(), data.TotalDof, &x(0),
					*data.host.getNfreqs(0), data.device.getXPS(0), *data.host.getSzXPS(0),
					data.device.getChains(0), data.device.getSurplus(0), &result(0));

				volatile double start, finish;
				get_time(&start);
				for (int i = 0; i < ntests; i++)
				{
					INTERPOLATE_ARRAY(device, data.dim,
						data.host.getSurplus(0)->dimy(), data.TotalDof, &x(0),
						*data.host.getNfreqs(0), data.device.getXPS(0), *data.host.getSzXPS(0),
						data.device.getChains(0), data.device.getSurplus(0), &result(0));
				}
				get_time(&finish);

				cout << "time = " << (finish - start) / ntests <<
					" sec (averaged from " << ntests << " tests)" << endl;
			}
			releaseDevice(device);

			check(&result(0), result.length());
		}
	};
}

TEST(InterpolateArray, cuda)
{
	cuda::GoogleTest::run();
}
#endif // NVCC

#include "Postprocessors.h"

class GoogleTest
{
public :

	GoogleTest(int argc, char* argv[])
	{
		MPI_Process::GoogleTest(argc, argv);

		Postprocessors* posts = Postprocessors::getInstance(1, "LinearBasis");

		string& filters = testing::GTEST_FLAG(filter);
		if (filters == "*")
		{
			filters += "*.x86:*.gold";
			if (isSupported(InstrSetAVX512F))
				filters += ":*.avx512";
			if (isSupported(InstrSetAVX2))
				filters += ":*.avx2";
			if (isSupported(InstrSetAVX))
				filters += ":*.avx";
#if defined(NVCC)
			cuda::Device* device = cuda::tryAcquireDevice();
			if (device)
			{
				filters += ":*.cuda";
				cuda::releaseDevice(device);
			}
#endif
		}
	}
};

// Benchmark specific target with e.g.:
// $ build/linux/gnu/release/test --gtest_filter=*.cuda
extern "C" int __wrap_main(int argc, char* argv[])
{
	const char* cntests = getenv("NTESTS");
	if (cntests)
		ntests = atoi(cntests);

	GoogleTest(argc, argv);

	::testing::InitGoogleTest(&argc, argv);

	return RUN_ALL_TESTS();
}

