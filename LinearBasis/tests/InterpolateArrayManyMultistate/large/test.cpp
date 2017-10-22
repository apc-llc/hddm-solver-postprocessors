#include <iostream>
#include <sstream>
#include <time.h>

#include "cpu/include/instrset.h"
#include "gtest/gtest.h"

#define EPSILON 0.001

#define str(x) #x
#define stringize(x) str(x)

using namespace std;

static bool runopt = true;

// Get the timer value.
static void get_time(double* ret)
{
	volatile struct timespec val;
	clock_gettime(CLOCK_REALTIME, (struct timespec*)&val);
	*ret = (double)0.000000001 * val.tv_nsec + val.tv_sec;
}

static void init(double* input, int dim)
{
	EXPECT_EQ(dim, 59);

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

static void check_0(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.0126576;
	expected[1] = -0.0422574;
	expected[2] = -0.0653376;
	expected[3] = -0.0835677;
	expected[4] = -0.0979356;
	expected[5] = -0.107302;
	expected[6] = -0.112555;
	expected[7] = -0.11428;
	expected[8] = -0.112634;
	expected[9] = -0.107759;
	expected[10] = -0.0997429;
	expected[11] = -0.0888063;
	expected[12] = -0.0746691;
	expected[13] = -0.0574801;
	expected[14] = -0.0371691;
	expected[15] = -0.0138759;
	expected[16] = 0.0121533;
	expected[17] = 0.0416433;
	expected[18] = 0.0743528;
	expected[19] = 0.108067;
	expected[20] = 0.139565;
	expected[21] = 0.172445;
	expected[22] = 0.205668;
	expected[23] = 0.240641;
	expected[24] = 0.275789;
	expected[25] = 0.312666;
	expected[26] = 0.349283;
	expected[27] = 0.385985;
	expected[28] = 0.422634;
	expected[29] = 0.459152;
	expected[30] = 0.496059;
	expected[31] = 0.531122;
	expected[32] = 0.564285;
	expected[33] = 0.594772;
	expected[34] = 0.626181;
	expected[35] = 0.650389;
	expected[36] = 0.674316;
	expected[37] = 0.696721;
	expected[38] = 0.717912;
	expected[39] = 0.738239;
	expected[40] = 0.746343;
	expected[41] = 0.753225;
	expected[42] = 0.758065;
	expected[43] = 0.761695;
	expected[44] = 0.759683;
	expected[45] = 0.846831;
	expected[46] = 0.859297;
	expected[47] = 0.871124;
	expected[48] = 0.87681;
	expected[49] = 0.87263;
	expected[50] = 0.857703;
	expected[51] = 0.839995;
	expected[52] = 0.807964;
	expected[53] = 0.759444;
	expected[54] = 0.691976;
	expected[55] = 0.602972;
	expected[56] = 0.490821;
	expected[57] = 0.353235;
	expected[58] = 0.189275;
	expected[59] = 175.936;
	expected[60] = 174.765;
	expected[61] = 173.056;
	expected[62] = 171.057;
	expected[63] = 168.759;
	expected[64] = 166.238;
	expected[65] = 163.518;
	expected[66] = 160.662;
	expected[67] = 157.72;
	expected[68] = 154.772;
	expected[69] = 151.852;
	expected[70] = 149.038;
	expected[71] = 146.29;
	expected[72] = 143.659;
	expected[73] = 141.131;
	expected[74] = 138.729;
	expected[75] = 136.491;
	expected[76] = 138.805;
	expected[77] = 132.874;
	expected[78] = 129.822;
	expected[79] = 127.137;
	expected[80] = 124.526;
	expected[81] = 122.438;
	expected[82] = 120.416;
	expected[83] = 120.41;
	expected[84] = 119.631;
	expected[85] = 121.32;
	expected[86] = 123.623;
	expected[87] = 126.142;
	expected[88] = 127.721;
	expected[89] = 126.569;
	expected[90] = 125.469;
	expected[91] = 130.561;
	expected[92] = 138.405;
	expected[93] = 111.982;
	expected[94] = 112.621;
	expected[95] = 107.399;
	expected[96] = 99.7557;
	expected[97] = 90.8652;
	expected[98] = 80.3716;
	expected[99] = 70.5367;
	expected[100] = 64.3504;
	expected[101] = 59.8474;
	expected[102] = 55.7405;
	expected[103] = 51.0096;
	expected[104] = 46.4509;
	expected[105] = 43.5638;
	expected[106] = 40.7839;
	expected[107] = 37.6053;
	expected[108] = 34.2822;
	expected[109] = 30.8761;
	expected[110] = 27.4053;
	expected[111] = 24.0588;
	expected[112] = 20.8856;
	expected[113] = 18.0306;
	expected[114] = 14.5908;
	expected[115] = 11.6072;
	expected[116] = 8.75984;
	expected[117] = 5.77744;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_1(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.0169148;
	expected[1] = -0.0468095;
	expected[2] = -0.0706319;
	expected[3] = -0.0890039;
	expected[4] = -0.102177;
	expected[5] = -0.111318;
	expected[6] = -0.116715;
	expected[7] = -0.118612;
	expected[8] = -0.117463;
	expected[9] = -0.112659;
	expected[10] = -0.103001;
	expected[11] = -0.0931258;
	expected[12] = -0.0799007;
	expected[13] = -0.0625668;
	expected[14] = -0.0430678;
	expected[15] = -0.0204053;
	expected[16] = 0.00523762;
	expected[17] = 0.0333554;
	expected[18] = 0.0641455;
	expected[19] = 0.0979326;
	expected[20] = 0.129177;
	expected[21] = 0.162355;
	expected[22] = 0.196371;
	expected[23] = 0.231331;
	expected[24] = 0.26703;
	expected[25] = 0.303181;
	expected[26] = 0.339732;
	expected[27] = 0.376343;
	expected[28] = 0.412897;
	expected[29] = 0.449247;
	expected[30] = 0.48561;
	expected[31] = 0.5211;
	expected[32] = 0.554847;
	expected[33] = 0.58686;
	expected[34] = 0.615109;
	expected[35] = 0.641134;
	expected[36] = 0.665037;
	expected[37] = 0.687043;
	expected[38] = 0.708027;
	expected[39] = 0.730918;
	expected[40] = 0.73808;
	expected[41] = 0.7455;
	expected[42] = 0.750766;
	expected[43] = 0.754889;
	expected[44] = 0.750167;
	expected[45] = 0.86672;
	expected[46] = 0.876404;
	expected[47] = 0.886823;
	expected[48] = 0.892282;
	expected[49] = 0.889855;
	expected[50] = 0.875608;
	expected[51] = 0.857846;
	expected[52] = 0.825639;
	expected[53] = 0.776564;
	expected[54] = 0.708362;
	expected[55] = 0.618734;
	expected[56] = 0.504775;
	expected[57] = 0.364638;
	expected[58] = 0.196785;
	expected[59] = 177.099;
	expected[60] = 175.209;
	expected[61] = 174.368;
	expected[62] = 172.382;
	expected[63] = 169.966;
	expected[64] = 167.421;
	expected[65] = 164.701;
	expected[66] = 161.836;
	expected[67] = 158.563;
	expected[68] = 155.999;
	expected[69] = 152.777;
	expected[70] = 150.455;
	expected[71] = 147.507;
	expected[72] = 145.509;
	expected[73] = 142.7;
	expected[74] = 139.816;
	expected[75] = 137.572;
	expected[76] = 140.094;
	expected[77] = 134.125;
	expected[78] = 131.151;
	expected[79] = 128.702;
	expected[80] = 125.673;
	expected[81] = 123.251;
	expected[82] = 121.339;
	expected[83] = 120.334;
	expected[84] = 120.864;
	expected[85] = 122.668;
	expected[86] = 125.156;
	expected[87] = 127.896;
	expected[88] = 131.162;
	expected[89] = 129.228;
	expected[90] = 128.087;
	expected[91] = 131.238;
	expected[92] = 135.129;
	expected[93] = 116.729;
	expected[94] = 113.941;
	expected[95] = 108.719;
	expected[96] = 101.694;
	expected[97] = 92.9039;
	expected[98] = 77.4365;
	expected[99] = 70.427;
	expected[100] = 64.6813;
	expected[101] = 60.1055;
	expected[102] = 55.4388;
	expected[103] = 50.9069;
	expected[104] = 44.7273;
	expected[105] = 42.8156;
	expected[106] = 40.0713;
	expected[107] = 36.9284;
	expected[108] = 33.6416;
	expected[109] = 30.2746;
	expected[110] = 26.8543;
	expected[111] = 23.5434;
	expected[112] = 20.3354;
	expected[113] = 17.2186;
	expected[114] = 14.1807;
	expected[115] = 11.2331;
	expected[116] = 8.34821;
	expected[117] = 5.48014;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_2(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.0126576;
	expected[1] = -0.0422574;
	expected[2] = -0.0653376;
	expected[3] = -0.0835677;
	expected[4] = -0.0979356;
	expected[5] = -0.107302;
	expected[6] = -0.112555;
	expected[7] = -0.11428;
	expected[8] = -0.112634;
	expected[9] = -0.107759;
	expected[10] = -0.0997429;
	expected[11] = -0.0888063;
	expected[12] = -0.0746691;
	expected[13] = -0.0574801;
	expected[14] = -0.0371691;
	expected[15] = -0.0138759;
	expected[16] = 0.0121533;
	expected[17] = 0.0416433;
	expected[18] = 0.0743528;
	expected[19] = 0.108067;
	expected[20] = 0.139565;
	expected[21] = 0.172445;
	expected[22] = 0.205668;
	expected[23] = 0.240641;
	expected[24] = 0.275789;
	expected[25] = 0.312666;
	expected[26] = 0.349283;
	expected[27] = 0.385985;
	expected[28] = 0.422634;
	expected[29] = 0.459152;
	expected[30] = 0.496059;
	expected[31] = 0.531122;
	expected[32] = 0.564285;
	expected[33] = 0.594772;
	expected[34] = 0.626181;
	expected[35] = 0.650389;
	expected[36] = 0.674316;
	expected[37] = 0.696721;
	expected[38] = 0.717912;
	expected[39] = 0.738239;
	expected[40] = 0.746343;
	expected[41] = 0.753225;
	expected[42] = 0.758065;
	expected[43] = 0.761695;
	expected[44] = 0.759683;
	expected[45] = 0.846831;
	expected[46] = 0.859297;
	expected[47] = 0.871124;
	expected[48] = 0.87681;
	expected[49] = 0.87263;
	expected[50] = 0.857703;
	expected[51] = 0.839995;
	expected[52] = 0.807964;
	expected[53] = 0.759444;
	expected[54] = 0.691976;
	expected[55] = 0.602972;
	expected[56] = 0.490821;
	expected[57] = 0.353235;
	expected[58] = 0.189275;
	expected[59] = 175.936;
	expected[60] = 174.764;
	expected[61] = 173.056;
	expected[62] = 171.057;
	expected[63] = 168.758;
	expected[64] = 166.237;
	expected[65] = 163.518;
	expected[66] = 160.662;
	expected[67] = 157.72;
	expected[68] = 154.772;
	expected[69] = 151.852;
	expected[70] = 149.038;
	expected[71] = 146.29;
	expected[72] = 143.659;
	expected[73] = 141.131;
	expected[74] = 138.729;
	expected[75] = 136.491;
	expected[76] = 138.805;
	expected[77] = 132.874;
	expected[78] = 129.822;
	expected[79] = 127.137;
	expected[80] = 124.526;
	expected[81] = 122.438;
	expected[82] = 120.416;
	expected[83] = 120.41;
	expected[84] = 119.631;
	expected[85] = 121.32;
	expected[86] = 123.623;
	expected[87] = 126.142;
	expected[88] = 127.721;
	expected[89] = 126.569;
	expected[90] = 125.469;
	expected[91] = 130.561;
	expected[92] = 138.405;
	expected[93] = 111.982;
	expected[94] = 112.621;
	expected[95] = 107.399;
	expected[96] = 99.7557;
	expected[97] = 90.8652;
	expected[98] = 80.3716;
	expected[99] = 70.5368;
	expected[100] = 64.3504;
	expected[101] = 59.8475;
	expected[102] = 55.7404;
	expected[103] = 51.0096;
	expected[104] = 46.4509;
	expected[105] = 43.5638;
	expected[106] = 40.7839;
	expected[107] = 37.6053;
	expected[108] = 34.2822;
	expected[109] = 30.8761;
	expected[110] = 27.4053;
	expected[111] = 24.0588;
	expected[112] = 20.8856;
	expected[113] = 18.0306;
	expected[114] = 14.5908;
	expected[115] = 11.6072;
	expected[116] = 8.75984;
	expected[117] = 5.77744;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_3(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.0169148;
	expected[1] = -0.0468095;
	expected[2] = -0.0706319;
	expected[3] = -0.0890039;
	expected[4] = -0.102177;
	expected[5] = -0.111318;
	expected[6] = -0.116715;
	expected[7] = -0.118612;
	expected[8] = -0.117463;
	expected[9] = -0.112659;
	expected[10] = -0.103001;
	expected[11] = -0.0931258;
	expected[12] = -0.0799007;
	expected[13] = -0.0625667;
	expected[14] = -0.0430678;
	expected[15] = -0.0204053;
	expected[16] = 0.00523761;
	expected[17] = 0.0333554;
	expected[18] = 0.0641455;
	expected[19] = 0.0979326;
	expected[20] = 0.129177;
	expected[21] = 0.162355;
	expected[22] = 0.196371;
	expected[23] = 0.231331;
	expected[24] = 0.26703;
	expected[25] = 0.303181;
	expected[26] = 0.339732;
	expected[27] = 0.376343;
	expected[28] = 0.412897;
	expected[29] = 0.449247;
	expected[30] = 0.48561;
	expected[31] = 0.5211;
	expected[32] = 0.554847;
	expected[33] = 0.58686;
	expected[34] = 0.615109;
	expected[35] = 0.641134;
	expected[36] = 0.665037;
	expected[37] = 0.687043;
	expected[38] = 0.708027;
	expected[39] = 0.730918;
	expected[40] = 0.73808;
	expected[41] = 0.7455;
	expected[42] = 0.750766;
	expected[43] = 0.754889;
	expected[44] = 0.750167;
	expected[45] = 0.86672;
	expected[46] = 0.876404;
	expected[47] = 0.886823;
	expected[48] = 0.892282;
	expected[49] = 0.889855;
	expected[50] = 0.875608;
	expected[51] = 0.857846;
	expected[52] = 0.825639;
	expected[53] = 0.776564;
	expected[54] = 0.708362;
	expected[55] = 0.618734;
	expected[56] = 0.504775;
	expected[57] = 0.364638;
	expected[58] = 0.196785;
	expected[59] = 177.099;
	expected[60] = 175.209;
	expected[61] = 174.368;
	expected[62] = 172.382;
	expected[63] = 169.966;
	expected[64] = 167.421;
	expected[65] = 164.701;
	expected[66] = 161.836;
	expected[67] = 158.563;
	expected[68] = 155.999;
	expected[69] = 152.777;
	expected[70] = 150.455;
	expected[71] = 147.507;
	expected[72] = 145.509;
	expected[73] = 142.7;
	expected[74] = 139.816;
	expected[75] = 137.572;
	expected[76] = 140.094;
	expected[77] = 134.125;
	expected[78] = 131.151;
	expected[79] = 128.702;
	expected[80] = 125.672;
	expected[81] = 123.251;
	expected[82] = 121.339;
	expected[83] = 120.334;
	expected[84] = 120.864;
	expected[85] = 122.668;
	expected[86] = 125.156;
	expected[87] = 127.896;
	expected[88] = 131.162;
	expected[89] = 129.228;
	expected[90] = 128.087;
	expected[91] = 131.238;
	expected[92] = 135.129;
	expected[93] = 116.729;
	expected[94] = 113.941;
	expected[95] = 108.719;
	expected[96] = 101.694;
	expected[97] = 92.9039;
	expected[98] = 77.4365;
	expected[99] = 70.427;
	expected[100] = 64.6813;
	expected[101] = 60.1055;
	expected[102] = 55.4388;
	expected[103] = 50.9069;
	expected[104] = 44.7274;
	expected[105] = 42.8156;
	expected[106] = 40.0713;
	expected[107] = 36.9284;
	expected[108] = 33.6416;
	expected[109] = 30.2746;
	expected[110] = 26.8543;
	expected[111] = 23.5434;
	expected[112] = 20.3354;
	expected[113] = 17.2186;
	expected[114] = 14.1807;
	expected[115] = 11.2331;
	expected[116] = 8.34821;
	expected[117] = 5.48014;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

#define NAMESPACE gold
#include "gold/include/Data.h"
#define INTERPOLATE_ARRAY_MANY_MULTISTATE LinearBasis_gold_Generic_InterpolateArrayManyMultistate

namespace gold
{
	class Device;

	extern "C" void INTERPOLATE_ARRAY_MANY_MULTISTATE(
		Device* device,
		const int dim, int DofPerNode, const int count, const double* const* x_,
		const Matrix<int>* index, const Matrix<double>* surplus, double** value);

	class GoogleTest
	{
	public :

		GoogleTest()
		{
			using namespace gold;

			int nstates = 16;
			
			Data::Dense data(nstates);
			for (int i = 0; i < nstates; i += 4)
			{
				data.load("surplus_1.plt", i);
				data.load("surplus_2.plt", i + 1);
				data.load("surplus_3.plt", i + 2);
				data.load("surplus_4.plt", i + 3);
			}

			vector<Vector<double> > vx(nstates);
			vector<double*> x(nstates);
			for (int i = 0; i < nstates; i++)
			{
				vx[i].resize(data.dim);
				x[i] = vx[i].getData();
				init(x[i], data.dim);
			}

			vector<Vector<double> > vresults(nstates);
			vector<double*> results(nstates);
			for (int i = 0; i < nstates; i++)
			{
				vresults[i].resize(data.TotalDof);
				results[i] = vresults[i].getData();
			}

			Device* device = NULL;

			double start, finish;
			get_time(&start);
			INTERPOLATE_ARRAY_MANY_MULTISTATE(
				device, data.dim, data.TotalDof, nstates, &x[0],
				&data.index[0], &data.surplus[0], &results[0]);
			get_time(&finish);
			
			cout << "time = " << (finish - start) << " sec" << endl;

			for (int i = 0; i < nstates; i += 4)
			{
				check_0(results[i], data.TotalDof);
				check_1(results[i + 1], data.TotalDof);
				check_2(results[i + 2], data.TotalDof);
				check_3(results[i + 3], data.TotalDof);
			}
		}
	};
}

TEST(InterpolateArrayManyMultistate, gold)
{
	gold::GoogleTest();
}

#undef NAMESPACE
#define NAMESPACE x86
#undef DATA_H
#include "x86/include/Data.h"
#undef JIT_H
#include "x86/include/JIT.h"
#undef INTERPOLATE_ARRAY_MANY_MULTISTATE
#define INTERPOLATE_ARRAY_MANY_MULTISTATE LinearBasis_x86_Generic_InterpolateArrayManyMultistate
#undef INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT
#define INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT LinearBasis_x86_RuntimeOpt_InterpolateArrayManyMultistate

namespace x86
{
	class Device;

	extern "C" void INTERPOLATE_ARRAY_MANY_MULTISTATE(
		Device* device,
		const int dim, int DofPerNode, const int count, const double* const* x_,
		const int* nfreqs, const XPS* xps, const Chains* chains, const Matrix<double>* surplus, double** value);

	class GoogleTest
	{
	public :

		GoogleTest()
		{
			using namespace x86;

			int nstates = 16;
			
			Data::Sparse data(nstates);
			for (int i = 0; i < nstates; i += 4)
			{
				data.load("surplus_1.plt", i);
				data.load("surplus_2.plt", i + 1);
				data.load("surplus_3.plt", i + 2);
				data.load("surplus_4.plt", i + 3);
			}

			vector<Vector<double> > vx(nstates);
			vector<double*> x(nstates);
			for (int i = 0; i < nstates; i++)
			{
				vx[i].resize(data.dim);
				x[i] = vx[i].getData();
				init(x[i], data.dim);
			}

			vector<Vector<double> > vresults(nstates);
			vector<double*> results(nstates);
			for (int i = 0; i < nstates; i++)
			{
				vresults[i].resize(data.TotalDof);
				results[i] = vresults[i].getData();
			}

			Device* device = NULL;

			if (runopt)
			{
				typedef void (*Func)(
					Device* device, const int dim, int DofPerNode, const int count, const double* const* x_,
					const int* nfreqs, const XPS* xps, const Chains* chains, const Matrix<double>* surplus, double** value);

				Func INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT =
					JIT::jitCompile(device, data.dim, nstates, data.TotalDof,
						stringize(LinearBasis_x86_RuntimeOpt_InterpolateArrayManyMultistate) "_",
						(Func)NULL).getFunc();

				double start, finish;
				get_time(&start);
				INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT(
					device, data.dim, data.TotalDof, nstates, &x[0],
					&data.nfreqs[0], &data.xps[0], &data.chains[0], &data.surplus[0], &results[0]);
				get_time(&finish);
			
				cout << "time = " << (finish - start) << " sec" << endl;
			}
			else
			{
				double start, finish;
				get_time(&start);
				INTERPOLATE_ARRAY_MANY_MULTISTATE(
					device, data.dim, data.TotalDof, nstates, &x[0],
					&data.nfreqs[0], &data.xps[0], &data.chains[0], &data.surplus[0], &results[0]);
				get_time(&finish);
			
				cout << "time = " << (finish - start) << " sec" << endl;
			}

			for (int i = 0; i < nstates; i += 4)
			{
				check_0(results[i], data.TotalDof);
				check_1(results[i + 1], data.TotalDof);
				check_2(results[i + 2], data.TotalDof);
				check_3(results[i + 3], data.TotalDof);
			}
		}
	};
}

TEST(InterpolateArrayManyMultistate, x86)
{
	x86::GoogleTest();
}

#undef NAMESPACE
#define NAMESPACE avx
#undef DOUBLE_VECTOR_SIZE
#define DOUBLE_VECTOR_SIZE 4
#undef DATA_H
#include "avx/include/Data.h"
#undef JIT_H
#include "avx/include/JIT.h"
#undef INTERPOLATE_ARRAY_MANY_MULTISTATE
#define INTERPOLATE_ARRAY_MANY_MULTISTATE LinearBasis_avx_Generic_InterpolateArrayManyMultistate
#undef INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT
#define INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT LinearBasis_avx_RuntimeOpt_InterpolateArrayManyMultistate

namespace avx
{
	class Device;

	extern "C" void INTERPOLATE_ARRAY_MANY_MULTISTATE(
		Device* device,
		const int dim, int DofPerNode, const int count, const double* const* x_,
		const int* nfreqs, const XPS* xps, const Chains* chains, const Matrix<double>* surplus, double** value);

	class GoogleTest
	{
	public :

		GoogleTest()
		{
			using namespace avx;

			int nstates = 16;
			
			Data::Sparse data(nstates);
			for (int i = 0; i < nstates; i += 4)
			{
				data.load("surplus_1.plt", i);
				data.load("surplus_2.plt", i + 1);
				data.load("surplus_3.plt", i + 2);
				data.load("surplus_4.plt", i + 3);
			}

			vector<Vector<double> > vx(nstates);
			vector<double*> x(nstates);
			for (int i = 0; i < nstates; i++)
			{
				vx[i].resize(data.dim);
				x[i] = vx[i].getData();
				init(x[i], data.dim);
			}

			vector<Vector<double> > vresults(nstates);
			vector<double*> results(nstates);
			for (int i = 0; i < nstates; i++)
			{
				vresults[i].resize(data.TotalDof);
				results[i] = vresults[i].getData();
			}

			Device* device = NULL;

			if (runopt)
			{
				typedef void (*Func)(
					Device* device, const int dim, int DofPerNode, const int count, const double* const* x_,
					const int* nfreqs, const XPS* xps, const Chains* chains, const Matrix<double>* surplus, double** value);

				Func INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT =
					JIT::jitCompile(device, data.dim, nstates, data.TotalDof,
						stringize(LinearBasis_avx_RuntimeOpt_InterpolateArrayManyMultistate) "_",
						(Func)NULL).getFunc();

				double start, finish;
				get_time(&start);
				INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT(
					device, data.dim, data.TotalDof, nstates, &x[0],
					&data.nfreqs[0], &data.xps[0], &data.chains[0], &data.surplus[0], &results[0]);
				get_time(&finish);

				cout << "time = " << (finish - start) << " sec" << endl;
			}
			else
			{
				double start, finish;
				get_time(&start);
				INTERPOLATE_ARRAY_MANY_MULTISTATE(
					device, data.dim, data.TotalDof, nstates, &x[0],
					&data.nfreqs[0], &data.xps[0], &data.chains[0], &data.surplus[0], &results[0]);
				get_time(&finish);

				cout << "time = " << (finish - start) << " sec" << endl;
			}
			
			for (int i = 0; i < nstates; i += 4)
			{
				check_0(results[i], data.TotalDof);
				check_1(results[i + 1], data.TotalDof);
				check_2(results[i + 2], data.TotalDof);
				check_3(results[i + 3], data.TotalDof);
			}
		}
	};
}

TEST(InterpolateArrayManyMultistate, avx)
{
	avx::GoogleTest();
}

#undef NAMESPACE
#define NAMESPACE avx2
#undef DOUBLE_VECTOR_SIZE
#define DOUBLE_VECTOR_SIZE 4
#undef DATA_H
#include "avx2/include/Data.h"
#undef JIT_H
#include "avx2/include/JIT.h"
#undef INTERPOLATE_ARRAY_MANY_MULTISTATE
#define INTERPOLATE_ARRAY_MANY_MULTISTATE LinearBasis_avx2_Generic_InterpolateArrayManyMultistate
#undef INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT
#define INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT LinearBasis_avx2_RuntimeOpt_InterpolateArrayManyMultistate

namespace avx2
{
	class Device;

	extern "C" void INTERPOLATE_ARRAY_MANY_MULTISTATE(
		Device* device,
		const int dim, int DofPerNode, const int count, const double* const* x_,
		const int* nfreqs, const XPS* xps, const Chains* chains, const Matrix<double>* surplus, double** value);

	class GoogleTest
	{
	public :

		GoogleTest()
		{
			using namespace avx2;

			int nstates = 16;
			
			Data::Sparse data(nstates);
			for (int i = 0; i < nstates; i += 4)
			{
				data.load("surplus_1.plt", i);
				data.load("surplus_2.plt", i + 1);
				data.load("surplus_3.plt", i + 2);
				data.load("surplus_4.plt", i + 3);
			}

			vector<Vector<double> > vx(nstates);
			vector<double*> x(nstates);
			for (int i = 0; i < nstates; i++)
			{
				vx[i].resize(data.dim);
				x[i] = vx[i].getData();
				init(x[i], data.dim);
			}

			vector<Vector<double> > vresults(nstates);
			vector<double*> results(nstates);
			for (int i = 0; i < nstates; i++)
			{
				vresults[i].resize(data.TotalDof);
				results[i] = vresults[i].getData();
			}

			Device* device = NULL;

			if (runopt)
			{
				typedef void (*Func)(
					Device* device, const int dim, int DofPerNode, const int count, const double* const* x_,
					const int* nfreqs, const XPS* xps, const Chains* chains, const Matrix<double>* surplus, double** value);

				Func INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT =
					JIT::jitCompile(device, data.dim, nstates, data.TotalDof,
						stringize(LinearBasis_avx2_RuntimeOpt_InterpolateArrayManyMultistate) "_",
						(Func)NULL).getFunc();

				double start, finish;
				get_time(&start);
				INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT(
					device, data.dim, data.TotalDof, nstates, &x[0],
					&data.nfreqs[0], &data.xps[0], &data.chains[0], &data.surplus[0], &results[0]);
				get_time(&finish);

				cout << "time = " << (finish - start) << " sec" << endl;
			}
			else
			{
				double start, finish;
				get_time(&start);
				INTERPOLATE_ARRAY_MANY_MULTISTATE(
					device, data.dim, data.TotalDof, nstates, &x[0],
					&data.nfreqs[0], &data.xps[0], &data.chains[0], &data.surplus[0], &results[0]);
				get_time(&finish);
			
				cout << "time = " << (finish - start) << " sec" << endl;
			}

			for (int i = 0; i < nstates; i += 4)
			{
				check_0(results[i], data.TotalDof);
				check_1(results[i + 1], data.TotalDof);
				check_2(results[i + 2], data.TotalDof);
				check_3(results[i + 3], data.TotalDof);
			}
		}
	};
}

TEST(InterpolateArrayManyMultistate, avx2)
{
	avx2::GoogleTest();
}

#undef NAMESPACE
#define NAMESPACE avx512
#undef DOUBLE_VECTOR_SIZE
#define DOUBLE_VECTOR_SIZE 8
#undef DATA_H
#include "avx512/include/Data.h"
#undef JIT_H
#include "avx512/include/JIT.h"
#undef INTERPOLATE_ARRAY_MANY_MULTISTATE
#define INTERPOLATE_ARRAY_MANY_MULTISTATE LinearBasis_avx512_Generic_InterpolateArrayManyMultistate
#undef INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT
#define INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT LinearBasis_avx512_RuntimeOpt_InterpolateArrayManyMultistate

namespace avx512
{
	class Device;

	extern "C" void INTERPOLATE_ARRAY_MANY_MULTISTATE(
		Device* device,
		const int dim, int DofPerNode, const int count, const double* const* x_,
		const int* nfreqs, const XPS* xps, const Chains* chains, const Matrix<double>* surplus, double** value);

	class GoogleTest
	{
	public :

		GoogleTest()
		{
			using namespace avx512;

			int nstates = 16;
			
			Data::Sparse data(nstates);
			for (int i = 0; i < nstates; i += 4)
			{
				data.load("surplus_1.plt", i);
				data.load("surplus_2.plt", i + 1);
				data.load("surplus_3.plt", i + 2);
				data.load("surplus_4.plt", i + 3);
			}

			vector<Vector<double> > vx(nstates);
			vector<double*> x(nstates);
			for (int i = 0; i < nstates; i++)
			{
				vx[i].resize(data.dim);
				x[i] = vx[i].getData();
				init(x[i], data.dim);
			}

			vector<Vector<double> > vresults(nstates);
			vector<double*> results(nstates);
			for (int i = 0; i < nstates; i++)
			{
				vresults[i].resize(data.TotalDof);
				results[i] = vresults[i].getData();
			}

			Device* device = NULL;

			if (runopt)
			{
				typedef void (*Func)(
					Device* device, const int dim, int DofPerNode, const int count, const double* const* x_,
					const int* nfreqs, const XPS* xps, const Chains* chains, const Matrix<double>* surplus, double** value);

				Func INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT =
					JIT::jitCompile(device, data.dim, nstates, data.TotalDof,
						stringize(LinearBasis_avx512_RuntimeOpt_InterpolateArrayManyMultistate) "_",
						(Func)NULL).getFunc();

				double start, finish;
				get_time(&start);
				INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT(
					device, data.dim, data.TotalDof, nstates, &x[0],
					&data.nfreqs[0], &data.xps[0], &data.chains[0], &data.surplus[0], &results[0]);
				get_time(&finish);

				cout << "time = " << (finish - start) << " sec" << endl;
			}
			else
			{
				double start, finish;
				get_time(&start);
				INTERPOLATE_ARRAY_MANY_MULTISTATE(
					device, data.dim, data.TotalDof, nstates, &x[0],
					&data.nfreqs[0], &data.xps[0], &data.chains[0], &data.surplus[0], &results[0]);
				get_time(&finish);

				cout << "time = " << (finish - start) << " sec" << endl;
			}
			
			for (int i = 0; i < nstates; i += 4)
			{
				check_0(results[i], data.TotalDof);
				check_1(results[i + 1], data.TotalDof);
				check_2(results[i + 2], data.TotalDof);
				check_3(results[i + 3], data.TotalDof);
			}
		}
	};
}

TEST(InterpolateArrayManyMultistate, avx512)
{
	avx512::GoogleTest();
}

#if defined(NVCC)
#undef NAMESPACE
#define NAMESPACE cuda
#undef DATA_H
#include "cuda/include/Data.h"
#undef JIT_H
#include "cuda/include/JIT.h"
#undef INTERPOLATE_ARRAY_MANY_MULTISTATE
#define INTERPOLATE_ARRAY_MANY_MULTISTATE LinearBasis_cuda_Generic_InterpolateArrayManyMultistate
#undef INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT
#define INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT LinearBasis_cuda_RuntimeOpt_InterpolateArrayManyMultistate
#include "cuda/include/Devices.h"

namespace cuda
{
	extern "C" void INTERPOLATE_ARRAY_MANY_MULTISTATE(
		Device* device,
		const int dim, const int* nnos, int DofPerNode, const int count, const double* const* x_,
		const int* nfreqs, const XPS::Device* xps, const int* szxps, const Chains::Device* chains,
		const Matrix<double>::Device* surplus, double** value);

	class GoogleTest
	{
	public :

		GoogleTest()
		{
			using namespace cuda;

			int nstates = 16;

			vector<Vector<double>::Host > vresults(nstates);
			vector<double*> results(nstates);

			Device* device = cuda::tryAcquireDevice();
			EXPECT_TRUE(device != NULL);
			if (!device) return;
			{			
				Data data(nstates);
				for (int i = 0; i < nstates; i += 4)
				{
					data.load("surplus_1.plt", i);
					data.load("surplus_2.plt", i + 1);
					data.load("surplus_3.plt", i + 2);
					data.load("surplus_4.plt", i + 3);
				}

				vector<Vector<double>::Host > vx(nstates);
				vector<double*> x(nstates);
				for (int i = 0; i < nstates; i++)
				{
					vx[i].resize(data.dim);
					x[i] = vx[i].getData();
					init(x[i], data.dim);
				}

				for (int i = 0; i < nstates; i++)
				{
					vresults[i].resize(data.TotalDof);
					results[i] = vresults[i].getData();
				}

				vector<int> nnos(data.nstates);
				for (int i = 0; i < data.nstates; i++)
					nnos[i] = data.host.getSurplus(i)->dimy();

				if (runopt)
				{
					typedef void (*Func)(
						Device* device, const int dim, const int* nnos, int DofPerNode, const int count, const double* const* x_,
						const int* nfreqs, const XPS::Device* xps, const int* szxps, const Chains::Device* chains,
						const Matrix<double>::Device* surplus, double** value);

					Func INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT =
						JIT::jitCompile(device, data.dim, nstates, data.TotalDof,
							stringize(LinearBasis_cuda_RuntimeOpt_InterpolateArrayManyMultistate) "_",
							(Func)NULL).getFunc();

					// Run once without timing to do all CUDA-specific internal initializations.
					INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT(
						device, data.dim, &nnos[0], data.TotalDof, nstates, &x[0],
						data.device.getNfreqs(0), data.device.getXPS(0), data.host.getSzXPS(0),
						data.device.getChains(0), data.device.getSurplus(0), &results[0]);

					double start, finish;
					get_time(&start);
					INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT(
						device, data.dim, &nnos[0], data.TotalDof, nstates, &x[0],
						data.device.getNfreqs(0), data.device.getXPS(0), data.host.getSzXPS(0),
						data.device.getChains(0), data.device.getSurplus(0), &results[0]);
					get_time(&finish);

					cout << "time = " << (finish - start) << " sec" << endl;
				}
				else
				{
					// Run once without timing to do all CUDA-specific internal initializations.
					INTERPOLATE_ARRAY_MANY_MULTISTATE(
						device, data.dim, &nnos[0], data.TotalDof, nstates, &x[0],
						data.device.getNfreqs(0), data.device.getXPS(0), data.host.getSzXPS(0),
						data.device.getChains(0), data.device.getSurplus(0), &results[0]);

					double start, finish;
					get_time(&start);
					INTERPOLATE_ARRAY_MANY_MULTISTATE(
						device, data.dim, &nnos[0], data.TotalDof, nstates, &x[0],
						data.device.getNfreqs(0), data.device.getXPS(0), data.host.getSzXPS(0),
						data.device.getChains(0), data.device.getSurplus(0), &results[0]);
					get_time(&finish);

					cout << "time = " << (finish - start) << " sec" << endl;
				}
			}
			releaseDevice(device);

			for (int i = 0; i < nstates; i += 4)
			{
				check_0(results[i], vresults[0].length());
				check_1(results[i + 1], vresults[1].length());
				check_2(results[i + 2], vresults[2].length());
				check_3(results[i + 3], vresults[3].length());
			}
		}
	};
}

TEST(InterpolateArrayManyMultistate, cuda)
{
	cuda::GoogleTest();
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

		string filters = "*.x86:*.gold";
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

		testing::GTEST_FLAG(filter) = filters;
	}
};

extern "C" int __wrap_main(int argc, char* argv[])
{
	const char* crunopt = getenv("RUNTIME_OPTIMIZATION");
	if (crunopt)
		runopt = !!atoi(crunopt);

	GoogleTest(argc, argv);

	::testing::InitGoogleTest(&argc, argv);

	return RUN_ALL_TESTS();
}

