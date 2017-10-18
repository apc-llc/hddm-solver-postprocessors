#include <sstream>

#include "cpu/include/instrset.h"
#include "gtest/gtest.h"

using namespace std;

#define EPSILON 0.001

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
	expected[0] = -0.078341;
	expected[1] = -0.194503;
	expected[2] = -0.283318;
	expected[3] = -0.365285;
	expected[4] = -0.439613;
	expected[5] = -0.50559;
	expected[6] = -0.56313;
	expected[7] = -0.611937;
	expected[8] = -0.650852;
	expected[9] = -0.679727;
	expected[10] = -0.698053;
	expected[11] = -0.70583;
	expected[12] = -0.701072;
	expected[13] = -0.684447;
	expected[14] = -0.656505;
	expected[15] = -0.617928;
	expected[16] = -0.568909;
	expected[17] = -0.509985;
	expected[18] = -0.441193;
	expected[19] = -0.364873;
	expected[20] = -0.287477;
	expected[21] = -0.203915;
	expected[22] = -0.115144;
	expected[23] = -0.0213688;
	expected[24] = 0.0763575;
	expected[25] = 0.176866;
	expected[26] = 0.279381;
	expected[27] = 0.384054;
	expected[28] = 0.4908;
	expected[29] = 0.600456;
	expected[30] = 0.713626;
	expected[31] = 0.829469;
	expected[32] = 0.947814;
	expected[33] = 1.07012;
	expected[34] = 1.19605;
	expected[35] = 1.32602;
	expected[36] = 1.46055;
	expected[37] = 1.59748;
	expected[38] = 1.73246;
	expected[39] = 1.86298;
	expected[40] = 1.96273;
	expected[41] = 2.05035;
	expected[42] = 2.12125;
	expected[43] = 2.17186;
	expected[44] = 2.19847;
	expected[45] = 2.19726;
	expected[46] = 2.16527;
	expected[47] = 2.10374;
	expected[48] = 2.01498;
	expected[49] = 1.90078;
	expected[50] = 1.7633;
	expected[51] = 1.62658;
	expected[52] = 1.47374;
	expected[53] = 1.30715;
	expected[54] = 1.12905;
	expected[55] = 0.941378;
	expected[56] = 0.745214;
	expected[57] = 0.539154;
	expected[58] = 0.312587;
	expected[59] = 95.0015;
	expected[60] = 93.6625;
	expected[61] = 90.5537;
	expected[62] = 86.6859;
	expected[63] = 84.1117;
	expected[64] = 83.3354;
	expected[65] = 82.7981;
	expected[66] = 82.266;
	expected[67] = 81.6492;
	expected[68] = 81.0757;
	expected[69] = 80.4071;
	expected[70] = 79.651;
	expected[71] = 78.7828;
	expected[72] = 77.8273;
	expected[73] = 76.7846;
	expected[74] = 75.6698;
	expected[75] = 74.5422;
	expected[76] = 73.4027;
	expected[77] = 72.2666;
	expected[78] = 71.1555;
	expected[79] = 70.0579;
	expected[80] = 68.8855;
	expected[81] = 67.6203;
	expected[82] = 66.315;
	expected[83] = 64.9769;
	expected[84] = 63.6248;
	expected[85] = 62.2639;
	expected[86] = 60.8916;
	expected[87] = 59.5227;
	expected[88] = 58.1277;
	expected[89] = 56.7188;
	expected[90] = 55.2934;
	expected[91] = 53.8467;
	expected[92] = 52.37;
	expected[93] = 50.8643;
	expected[94] = 49.3229;
	expected[95] = 47.7553;
	expected[96] = 46.1807;
	expected[97] = 44.6368;
	expected[98] = 43.1606;
	expected[99] = 41.7707;
	expected[100] = 40.2604;
	expected[101] = 38.6687;
	expected[102] = 37.0184;
	expected[103] = 35.3328;
	expected[104] = 33.6354;
	expected[105] = 31.9397;
	expected[106] = 30.2152;
	expected[107] = 28.4095;
	expected[108] = 26.4226;
	expected[109] = 24.4021;
	expected[110] = 22.2042;
	expected[111] = 19.9961;
	expected[112] = 17.7435;
	expected[113] = 15.4109;
	expected[114] = 12.9582;
	expected[115] = 10.3812;
	expected[116] = 7.62577;
	expected[117] = 4.68565;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_1(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.0872779;
	expected[1] = -0.201061;
	expected[2] = -0.290059;
	expected[3] = -0.372083;
	expected[4] = -0.446874;
	expected[5] = -0.51456;
	expected[6] = -0.572352;
	expected[7] = -0.621409;
	expected[8] = -0.661488;
	expected[9] = -0.690726;
	expected[10] = -0.709473;
	expected[11] = -0.716777;
	expected[12] = -0.712575;
	expected[13] = -0.696594;
	expected[14] = -0.66943;
	expected[15] = -0.631419;
	expected[16] = -0.583312;
	expected[17] = -0.525033;
	expected[18] = -0.457623;
	expected[19] = -0.381465;
	expected[20] = -0.304102;
	expected[21] = -0.220715;
	expected[22] = -0.131773;
	expected[23] = -0.0386992;
	expected[24] = 0.0581765;
	expected[25] = 0.158624;
	expected[26] = 0.261117;
	expected[27] = 0.364958;
	expected[28] = 0.471993;
	expected[29] = 0.581594;
	expected[30] = 0.693299;
	expected[31] = 0.808553;
	expected[32] = 0.927215;
	expected[33] = 1.04931;
	expected[34] = 1.17492;
	expected[35] = 1.3046;
	expected[36] = 1.43865;
	expected[37] = 1.57462;
	expected[38] = 1.70977;
	expected[39] = 1.84037;
	expected[40] = 1.94238;
	expected[41] = 2.03051;
	expected[42] = 2.10274;
	expected[43] = 2.15476;
	expected[44] = 2.18289;
	expected[45] = 2.18341;
	expected[46] = 2.15333;
	expected[47] = 2.09329;
	expected[48] = 2.00579;
	expected[49] = 1.89336;
	expected[50] = 1.75752;
	expected[51] = 1.62096;
	expected[52] = 1.46831;
	expected[53] = 1.30192;
	expected[54] = 1.12404;
	expected[55] = 0.936584;
	expected[56] = 0.740812;
	expected[57] = 0.535327;
	expected[58] = 0.309777;
	expected[59] = 95.3401;
	expected[60] = 93.9947;
	expected[61] = 90.9218;
	expected[62] = 87.0566;
	expected[63] = 84.5258;
	expected[64] = 83.7963;
	expected[65] = 83.2826;
	expected[66] = 82.7479;
	expected[67] = 82.1837;
	expected[68] = 81.6279;
	expected[69] = 80.9738;
	expected[70] = 80.2157;
	expected[71] = 79.3546;
	expected[72] = 78.4083;
	expected[73] = 77.3547;
	expected[74] = 76.2506;
	expected[75] = 75.0849;
	expected[76] = 73.9283;
	expected[77] = 72.7838;
	expected[78] = 71.6529;
	expected[79] = 70.5312;
	expected[80] = 69.3381;
	expected[81] = 68.0293;
	expected[82] = 66.7237;
	expected[83] = 65.3712;
	expected[84] = 63.9918;
	expected[85] = 62.602;
	expected[86] = 61.2093;
	expected[87] = 59.8105;
	expected[88] = 58.4101;
	expected[89] = 56.9714;
	expected[90] = 55.5292;
	expected[91] = 54.0552;
	expected[92] = 52.558;
	expected[93] = 51.0339;
	expected[94] = 49.4859;
	expected[95] = 47.8912;
	expected[96] = 46.3033;
	expected[97] = 44.7458;
	expected[98] = 43.2494;
	expected[99] = 41.8269;
	expected[100] = 40.3025;
	expected[101] = 38.6903;
	expected[102] = 37.0219;
	expected[103] = 35.3207;
	expected[104] = 33.6084;
	expected[105] = 31.899;
	expected[106] = 30.1685;
	expected[107] = 28.3625;
	expected[108] = 26.3729;
	expected[109] = 24.3523;
	expected[110] = 22.1693;
	expected[111] = 19.9762;
	expected[112] = 17.7381;
	expected[113] = 15.4187;
	expected[114] = 12.9806;
	expected[115] = 10.4149;
	expected[116] = 7.66333;
	expected[117] = 4.7188;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_2(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.078341;
	expected[1] = -0.194503;
	expected[2] = -0.283318;
	expected[3] = -0.365285;
	expected[4] = -0.439613;
	expected[5] = -0.50559;
	expected[6] = -0.56313;
	expected[7] = -0.611937;
	expected[8] = -0.650852;
	expected[9] = -0.679727;
	expected[10] = -0.698053;
	expected[11] = -0.70583;
	expected[12] = -0.701072;
	expected[13] = -0.684447;
	expected[14] = -0.656505;
	expected[15] = -0.617928;
	expected[16] = -0.568909;
	expected[17] = -0.509985;
	expected[18] = -0.441193;
	expected[19] = -0.364873;
	expected[20] = -0.287477;
	expected[21] = -0.203915;
	expected[22] = -0.115144;
	expected[23] = -0.0213688;
	expected[24] = 0.0763575;
	expected[25] = 0.176866;
	expected[26] = 0.279381;
	expected[27] = 0.384054;
	expected[28] = 0.4908;
	expected[29] = 0.600456;
	expected[30] = 0.713626;
	expected[31] = 0.829469;
	expected[32] = 0.947814;
	expected[33] = 1.07012;
	expected[34] = 1.19605;
	expected[35] = 1.32602;
	expected[36] = 1.46055;
	expected[37] = 1.59748;
	expected[38] = 1.73246;
	expected[39] = 1.86298;
	expected[40] = 1.96273;
	expected[41] = 2.05035;
	expected[42] = 2.12125;
	expected[43] = 2.17186;
	expected[44] = 2.19847;
	expected[45] = 2.19726;
	expected[46] = 2.16527;
	expected[47] = 2.10374;
	expected[48] = 2.01498;
	expected[49] = 1.90078;
	expected[50] = 1.7633;
	expected[51] = 1.62658;
	expected[52] = 1.47374;
	expected[53] = 1.30715;
	expected[54] = 1.12905;
	expected[55] = 0.941378;
	expected[56] = 0.745214;
	expected[57] = 0.539154;
	expected[58] = 0.312587;
	expected[59] = 95.0015;
	expected[60] = 93.6625;
	expected[61] = 90.5537;
	expected[62] = 86.6859;
	expected[63] = 84.1117;
	expected[64] = 83.3354;
	expected[65] = 82.7981;
	expected[66] = 82.266;
	expected[67] = 81.6492;
	expected[68] = 81.0757;
	expected[69] = 80.4071;
	expected[70] = 79.651;
	expected[71] = 78.7828;
	expected[72] = 77.8273;
	expected[73] = 76.7846;
	expected[74] = 75.6698;
	expected[75] = 74.5422;
	expected[76] = 73.4027;
	expected[77] = 72.2666;
	expected[78] = 71.1555;
	expected[79] = 70.0579;
	expected[80] = 68.8855;
	expected[81] = 67.6203;
	expected[82] = 66.315;
	expected[83] = 64.9769;
	expected[84] = 63.6248;
	expected[85] = 62.2639;
	expected[86] = 60.8916;
	expected[87] = 59.5227;
	expected[88] = 58.1277;
	expected[89] = 56.7188;
	expected[90] = 55.2934;
	expected[91] = 53.8467;
	expected[92] = 52.37;
	expected[93] = 50.8643;
	expected[94] = 49.3229;
	expected[95] = 47.7553;
	expected[96] = 46.1807;
	expected[97] = 44.6368;
	expected[98] = 43.1606;
	expected[99] = 41.7707;
	expected[100] = 40.2604;
	expected[101] = 38.6687;
	expected[102] = 37.0184;
	expected[103] = 35.3328;
	expected[104] = 33.6354;
	expected[105] = 31.9397;
	expected[106] = 30.2152;
	expected[107] = 28.4095;
	expected[108] = 26.4226;
	expected[109] = 24.4021;
	expected[110] = 22.2042;
	expected[111] = 19.9961;
	expected[112] = 17.7435;
	expected[113] = 15.4109;
	expected[114] = 12.9582;
	expected[115] = 10.3812;
	expected[116] = 7.62577;
	expected[117] = 4.68565;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_3(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.0872779;
	expected[1] = -0.201061;
	expected[2] = -0.290059;
	expected[3] = -0.372083;
	expected[4] = -0.446874;
	expected[5] = -0.51456;
	expected[6] = -0.572352;
	expected[7] = -0.621409;
	expected[8] = -0.661488;
	expected[9] = -0.690726;
	expected[10] = -0.709473;
	expected[11] = -0.716777;
	expected[12] = -0.712575;
	expected[13] = -0.696594;
	expected[14] = -0.66943;
	expected[15] = -0.631419;
	expected[16] = -0.583312;
	expected[17] = -0.525033;
	expected[18] = -0.457623;
	expected[19] = -0.381465;
	expected[20] = -0.304102;
	expected[21] = -0.220715;
	expected[22] = -0.131773;
	expected[23] = -0.0386992;
	expected[24] = 0.0581765;
	expected[25] = 0.158624;
	expected[26] = 0.261117;
	expected[27] = 0.364958;
	expected[28] = 0.471993;
	expected[29] = 0.581594;
	expected[30] = 0.693299;
	expected[31] = 0.808553;
	expected[32] = 0.927215;
	expected[33] = 1.04931;
	expected[34] = 1.17492;
	expected[35] = 1.3046;
	expected[36] = 1.43865;
	expected[37] = 1.57462;
	expected[38] = 1.70977;
	expected[39] = 1.84037;
	expected[40] = 1.94238;
	expected[41] = 2.03051;
	expected[42] = 2.10274;
	expected[43] = 2.15476;
	expected[44] = 2.18289;
	expected[45] = 2.18341;
	expected[46] = 2.15333;
	expected[47] = 2.09329;
	expected[48] = 2.00579;
	expected[49] = 1.89336;
	expected[50] = 1.75752;
	expected[51] = 1.62096;
	expected[52] = 1.46831;
	expected[53] = 1.30192;
	expected[54] = 1.12404;
	expected[55] = 0.936584;
	expected[56] = 0.740812;
	expected[57] = 0.535327;
	expected[58] = 0.309777;
	expected[59] = 95.3401;
	expected[60] = 93.9947;
	expected[61] = 90.9218;
	expected[62] = 87.0566;
	expected[63] = 84.5258;
	expected[64] = 83.7963;
	expected[65] = 83.2826;
	expected[66] = 82.7479;
	expected[67] = 82.1837;
	expected[68] = 81.6279;
	expected[69] = 80.9738;
	expected[70] = 80.2157;
	expected[71] = 79.3546;
	expected[72] = 78.4083;
	expected[73] = 77.3547;
	expected[74] = 76.2506;
	expected[75] = 75.0849;
	expected[76] = 73.9283;
	expected[77] = 72.7838;
	expected[78] = 71.6529;
	expected[79] = 70.5312;
	expected[80] = 69.3381;
	expected[81] = 68.0293;
	expected[82] = 66.7237;
	expected[83] = 65.3712;
	expected[84] = 63.9918;
	expected[85] = 62.602;
	expected[86] = 61.2093;
	expected[87] = 59.8105;
	expected[88] = 58.4101;
	expected[89] = 56.9714;
	expected[90] = 55.5292;
	expected[91] = 54.0552;
	expected[92] = 52.558;
	expected[93] = 51.0339;
	expected[94] = 49.4859;
	expected[95] = 47.8912;
	expected[96] = 46.3033;
	expected[97] = 44.7458;
	expected[98] = 43.2494;
	expected[99] = 41.8269;
	expected[100] = 40.3025;
	expected[101] = 38.6903;
	expected[102] = 37.0219;
	expected[103] = 35.3207;
	expected[104] = 33.6084;
	expected[105] = 31.899;
	expected[106] = 30.1685;
	expected[107] = 28.3625;
	expected[108] = 26.3729;
	expected[109] = 24.3523;
	expected[110] = 22.1693;
	expected[111] = 19.9762;
	expected[112] = 17.7381;
	expected[113] = 15.4187;
	expected[114] = 12.9806;
	expected[115] = 10.4149;
	expected[116] = 7.66333;
	expected[117] = 4.7188;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

#define NAMESPACE gold
#include "cpu/include/Data.h"

namespace gold
{
	class Device;

	extern "C" void LinearBasis_gold_Generic_InterpolateArrayManyMultistate(
		Device* device,
		const int dim, int DofPerNode, const int count, const double* const* x_,
		const Matrix<int>* index, const Matrix<double>* surplus, double** value);

	class GoogleTest
	{
	public :

		GoogleTest()
		{
			using namespace gold;

			Data::Dense data(4);
			data.load("surplus_1.plt", 0);
			data.load("surplus_2.plt", 1);
			data.load("surplus_3.plt", 2);
			data.load("surplus_4.plt", 3);

			vector<Vector<double> > vx(4);
			vector<double*> x(4);
			for (int i = 0; i < 4; i++)
			{
				vx[i].resize(data.dim);
				x[i] = vx[i].getData();
				init(x[i], data.dim);
			}

			vector<Vector<double> > vresults(4);
			vector<double*> results(4);
			for (int i = 0; i < 4; i++)
			{
				vresults[i].resize(data.TotalDof);
				results[i] = vresults[i].getData();
			}

			Device* device = NULL;

			LinearBasis_gold_Generic_InterpolateArrayManyMultistate(
				device, data.dim, data.TotalDof, 4, &x[0],
				&data.index[0], &data.surplus[0], &results[0]);

			check_0(results[0], data.TotalDof);
			check_1(results[1], data.TotalDof);
			check_2(results[2], data.TotalDof);
			check_3(results[3], data.TotalDof);
		}
	};
}

TEST(InterpolateArray, gold)
{
	gold::GoogleTest();
}

#undef NAMESPACE
#define NAMESPACE x86
#undef DATA_H
#include "cpu/include/Data.h"

namespace x86
{
	class Device;

	extern "C" void LinearBasis_x86_Generic_InterpolateArrayManyMultistate(
		Device* device,
		const int dim, int DofPerNode, const int count, const double* const* x_,
		const int* nfreqs, const XPS* xps, const Chains* chains, const Matrix<double>* surplus, double** value);

	class GoogleTest
	{
	public :

		GoogleTest()
		{
			using namespace x86;

			Data::Sparse data(4);
			data.load("surplus_1.plt", 0);
			data.load("surplus_2.plt", 1);
			data.load("surplus_3.plt", 2);
			data.load("surplus_4.plt", 3);

			vector<Vector<double> > vx(4);
			vector<double*> x(4);
			for (int i = 0; i < 4; i++)
			{
				vx[i].resize(data.dim);
				x[i] = vx[i].getData();
				init(x[i], data.dim);
			}

			vector<Vector<double> > vresults(4);
			vector<double*> results(4);
			for (int i = 0; i < 4; i++)
			{
				vresults[i].resize(data.TotalDof);
				results[i] = vresults[i].getData();
			}

			Device* device = NULL;

			LinearBasis_x86_Generic_InterpolateArrayManyMultistate(
				device, data.dim, data.TotalDof, 4, &x[0],
				&data.nfreqs[0], &data.xps[0], &data.chains[0], &data.surplus[0], &results[0]);

			check_0(results[0], data.TotalDof);
			check_1(results[1], data.TotalDof);
			check_2(results[2], data.TotalDof);
			check_3(results[3], data.TotalDof);
		}
	};
}

TEST(InterpolateArray, x86)
{
	x86::GoogleTest();
}

#undef NAMESPACE
#define NAMESPACE avx
#undef DATA_H
#include "cpu/include/Data.h"

namespace avx
{
	class Device;

	extern "C" void LinearBasis_avx_Generic_InterpolateArrayManyMultistate(
		Device* device,
		const int dim, int DofPerNode, const int count, const double* const* x_,
		const int* nfreqs, const XPS* xps, const Chains* chains, const Matrix<double>* surplus, double** value);

	class GoogleTest
	{
	public :

		GoogleTest()
		{
			using namespace avx;

			Data::Sparse data(4);
			data.load("surplus_1.plt", 0);
			data.load("surplus_2.plt", 1);
			data.load("surplus_3.plt", 2);
			data.load("surplus_4.plt", 3);

			vector<Vector<double> > vx(4);
			vector<double*> x(4);
			for (int i = 0; i < 4; i++)
			{
				vx[i].resize(data.dim);
				x[i] = vx[i].getData();
				init(x[i], data.dim);
			}

			vector<Vector<double> > vresults(4);
			vector<double*> results(4);
			for (int i = 0; i < 4; i++)
			{
				vresults[i].resize(data.TotalDof);
				results[i] = vresults[i].getData();
			}

			Device* device = NULL;

			LinearBasis_avx_Generic_InterpolateArrayManyMultistate(
				device, data.dim, data.TotalDof, 4, &x[0],
				&data.nfreqs[0], &data.xps[0], &data.chains[0], &data.surplus[0], &results[0]);

			check_0(results[0], data.TotalDof);
			check_1(results[1], data.TotalDof);
			check_2(results[2], data.TotalDof);
			check_3(results[3], data.TotalDof);
		}
	};
}

TEST(InterpolateArray, avx)
{
	avx::GoogleTest();
}

#undef NAMESPACE
#define NAMESPACE avx2
#undef DATA_H
#include "cpu/include/Data.h"

namespace avx2
{
	class Device;

	extern "C" void LinearBasis_avx2_Generic_InterpolateArrayManyMultistate(
		Device* device,
		const int dim, int DofPerNode, const int count, const double* const* x_,
		const int* nfreqs, const XPS* xps, const Chains* chains, const Matrix<double>* surplus, double** value);

	class GoogleTest
	{
	public :

		GoogleTest()
		{
			using namespace avx2;

			Data::Sparse data(4);
			data.load("surplus_1.plt", 0);
			data.load("surplus_2.plt", 1);
			data.load("surplus_3.plt", 2);
			data.load("surplus_4.plt", 3);

			vector<Vector<double> > vx(4);
			vector<double*> x(4);
			for (int i = 0; i < 4; i++)
			{
				vx[i].resize(data.dim);
				x[i] = vx[i].getData();
				init(x[i], data.dim);
			}

			vector<Vector<double> > vresults(4);
			vector<double*> results(4);
			for (int i = 0; i < 4; i++)
			{
				vresults[i].resize(data.TotalDof);
				results[i] = vresults[i].getData();
			}

			Device* device = NULL;

			LinearBasis_avx2_Generic_InterpolateArrayManyMultistate(
				device, data.dim, data.TotalDof, 4, &x[0],
				&data.nfreqs[0], &data.xps[0], &data.chains[0], &data.surplus[0], &results[0]);

			check_0(results[0], data.TotalDof);
			check_1(results[1], data.TotalDof);
			check_2(results[2], data.TotalDof);
			check_3(results[3], data.TotalDof);
		}
	};
}

TEST(InterpolateArray, avx2)
{
	avx2::GoogleTest();
}

#undef NAMESPACE
#define NAMESPACE avx512
#undef DATA_H
#include "cpu/include/Data.h"

namespace avx512
{
	class Device;

	extern "C" void LinearBasis_avx512_Generic_InterpolateArrayManyMultistate(
		Device* device,
		const int dim, int DofPerNode, const int count, const double* const* x_,
		const int* nfreqs, const XPS* xps, const Chains* chains, const Matrix<double>* surplus, double** value);

	class GoogleTest
	{
	public :

		GoogleTest()
		{
			using namespace avx512;

			Data::Sparse data(4);
			data.load("surplus_1.plt", 0);
			data.load("surplus_2.plt", 1);
			data.load("surplus_3.plt", 2);
			data.load("surplus_4.plt", 3);

			vector<Vector<double> > vx(4);
			vector<double*> x(4);
			for (int i = 0; i < 4; i++)
			{
				vx[i].resize(data.dim);
				x[i] = vx[i].getData();
				init(x[i], data.dim);
			}

			vector<Vector<double> > vresults(4);
			vector<double*> results(4);
			for (int i = 0; i < 4; i++)
			{
				vresults[i].resize(data.TotalDof);
				results[i] = vresults[i].getData();
			}

			Device* device = NULL;

			LinearBasis_avx512_Generic_InterpolateArrayManyMultistate(
				device, data.dim, data.TotalDof, 4, &x[0],
				&data.nfreqs[0], &data.xps[0], &data.chains[0], &data.surplus[0], &results[0]);

			check_0(results[0], data.TotalDof);
			check_1(results[1], data.TotalDof);
			check_2(results[2], data.TotalDof);
			check_3(results[3], data.TotalDof);
		}
	};
}

TEST(InterpolateArray, avx512)
{
	avx512::GoogleTest();
}

#if defined(NVCC)
#undef NAMESPACE
#define NAMESPACE cuda
#undef DATA_H
#include "cuda/include/Data.h"
#include "cuda/include/Devices.h"

namespace cuda
{
	extern "C" void LinearBasis_cuda_Generic_InterpolateArrayManyMultistate(
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

			vector<Vector<double>::Host > vresults(4);
			vector<double*> results(4);

			Device* device = cuda::tryAcquireDevice();
			EXPECT_TRUE(device != NULL);
			if (!device) return;
			{
				Data data(4);
				data.load("surplus_1.plt", 0);
				data.load("surplus_2.plt", 1);
				data.load("surplus_3.plt", 2);
				data.load("surplus_4.plt", 3);

				vector<Vector<double>::Host > vx(4);
				vector<double*> x(4);
				for (int i = 0; i < 4; i++)
				{
					vx[i].resize(data.dim);
					x[i] = vx[i].getData();
					init(x[i], data.dim);
				}

				for (int i = 0; i < 4; i++)
				{
					vresults[i].resize(data.TotalDof);
					results[i] = vresults[i].getData();
				}

				vector<int> nnos(data.nstates);
				for (int i = 0; i < data.nstates; i++)
					nnos[i] = data.host.getSurplus(i)->dimy();

				LinearBasis_cuda_Generic_InterpolateArrayManyMultistate(device, data.dim,
					&nnos[0], data.TotalDof, 4, &x[0],
					data.device.getNfreqs(0), data.device.getXPS(0), data.host.getSzXPS(0),
					data.device.getChains(0), data.device.getSurplus(0), &results[0]);
			}
			releaseDevice(device);

			check_0(results[0], vresults[0].length());
			check_1(results[1], vresults[1].length());
			check_2(results[2], vresults[2].length());
			check_3(results[3], vresults[3].length());
		}
	};
}

TEST(InterpolateArray, cuda)
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
	GoogleTest(argc, argv);

	::testing::InitGoogleTest(&argc, argv);

	return RUN_ALL_TESTS();
}

