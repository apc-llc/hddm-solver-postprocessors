#include <iostream>
#include <sstream>
#include <time.h>

#include "cpu/include/instrset.h"
#include "gtest/gtest.h"

#define EPSILON 0.001

#define str(x) #x
#define stringize(x) str(x)

#define LOAD_DATA(filename, state) \
	do { \
		printf("Loading data from %s ... ", filename); fflush(stdout); \
		data.load(filename, state); \
		printf("done\n"); fflush(stdout); \
	} while (0)

using namespace std;

static bool runopt = true;

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
	expected[0] = -0.026351;
	expected[1] = -0.051250;
	expected[2] = -0.073673;
	expected[3] = -0.091655;
	expected[4] = -0.104356;
	expected[5] = -0.113119;
	expected[6] = -0.118184;
	expected[7] = -0.119763;
	expected[8] = -0.118029;
	expected[9] = -0.113143;
	expected[10] = -0.105214;
	expected[11] = -0.094566;
	expected[12] = -0.081089;
	expected[13] = -0.065070;
	expected[14] = -0.045923;
	expected[15] = -0.024232;
	expected[16] = -0.000097;
	expected[17] = 0.026554;
	expected[18] = 0.056039;
	expected[19] = 0.090244;
	expected[20] = 0.121443;
	expected[21] = 0.153987;
	expected[22] = 0.187671;
	expected[23] = 0.222275;
	expected[24] = 0.257564;
	expected[25] = 0.293299;
	expected[26] = 0.329291;
	expected[27] = 0.365172;
	expected[28] = 0.400752;
	expected[29] = 0.435724;
	expected[30] = 0.470515;
	expected[31] = 0.502687;
	expected[32] = 0.534287;
	expected[33] = 0.564713;
	expected[34] = 0.593868;
	expected[35] = 0.620574;
	expected[36] = 0.646897;
	expected[37] = 0.670056;
	expected[38] = 0.692626;
	expected[39] = 0.715262;
	expected[40] = 0.726351;
	expected[41] = 0.735325;
	expected[42] = 0.741261;
	expected[43] = 0.742425;
	expected[44] = 0.735223;
	expected[45] = 0.827506;
	expected[46] = 0.842899;
	expected[47] = 0.855694;
	expected[48] = 0.864061;
	expected[49] = 0.860384;
	expected[50] = 0.845984;
	expected[51] = 0.828763;
	expected[52] = 0.797468;
	expected[53] = 0.750070;
	expected[54] = 0.684427;
	expected[55] = 0.597990;
	expected[56] = 0.488181;
	expected[57] = 0.352671;
	expected[58] = 0.190091;
	expected[59] = 217.648838;
	expected[60] = 207.833938;
	expected[61] = 200.719156;
	expected[62] = 196.019988;
	expected[63] = 191.606805;
	expected[64] = 187.216397;
	expected[65] = 182.792823;
	expected[66] = 178.356059;
	expected[67] = 174.145105;
	expected[68] = 169.775062;
	expected[69] = 165.437768;
	expected[70] = 161.173061;
	expected[71] = 156.990111;
	expected[72] = 152.918856;
	expected[73] = 148.952783;
	expected[74] = 145.128482;
	expected[75] = 141.452998;
	expected[76] = 137.918366;
	expected[77] = 134.501370;
	expected[78] = 131.114643;
	expected[79] = 127.905035;
	expected[80] = 124.679086;
	expected[81] = 121.447996;
	expected[82] = 118.222015;
	expected[83] = 115.010283;
	expected[84] = 111.820436;
	expected[85] = 108.656457;
	expected[86] = 105.528395;
	expected[87] = 102.437469;
	expected[88] = 99.389677;
	expected[89] = 96.413330;
	expected[90] = 93.415099;
	expected[91] = 90.484666;
	expected[92] = 87.618946;
	expected[93] = 84.648477;
	expected[94] = 81.811672;
	expected[95] = 78.901554;
	expected[96] = 76.057555;
	expected[97] = 73.196698;
	expected[98] = 70.256213;
	expected[99] = 67.271028;
	expected[100] = 63.929479;
	expected[101] = 60.331226;
	expected[102] = 56.538756;
	expected[103] = 52.701663;
	expected[104] = 48.462447;
	expected[105] = 45.537227;
	expected[106] = 42.411245;
	expected[107] = 39.193655;
	expected[108] = 35.648011;
	expected[109] = 32.145626;
	expected[110] = 28.578188;
	expected[111] = 25.110651;
	expected[112] = 21.734095;
	expected[113] = 18.530438;
	expected[114] = 15.147845;
	expected[115] = 12.069098;
	expected[116] = 8.839518;
	expected[117] = 5.806405;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_1(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.028965;
	expected[1] = -0.055221;
	expected[2] = -0.076814;
	expected[3] = -0.093851;
	expected[4] = -0.106670;
	expected[5] = -0.115556;
	expected[6] = -0.120748;
	expected[7] = -0.122582;
	expected[8] = -0.121291;
	expected[9] = -0.116547;
	expected[10] = -0.109181;
	expected[11] = -0.098497;
	expected[12] = -0.084959;
	expected[13] = -0.068636;
	expected[14] = -0.049672;
	expected[15] = -0.028193;
	expected[16] = -0.004304;
	expected[17] = 0.022055;
	expected[18] = 0.051186;
	expected[19] = 0.083677;
	expected[20] = 0.114902;
	expected[21] = 0.147934;
	expected[22] = 0.182301;
	expected[23] = 0.216778;
	expected[24] = 0.251945;
	expected[25] = 0.287540;
	expected[26] = 0.323330;
	expected[27] = 0.359029;
	expected[28] = 0.394401;
	expected[29] = 0.429173;
	expected[30] = 0.463133;
	expected[31] = 0.495973;
	expected[32] = 0.527489;
	expected[33] = 0.558092;
	expected[34] = 0.585786;
	expected[35] = 0.612844;
	expected[36] = 0.638328;
	expected[37] = 0.662149;
	expected[38] = 0.685681;
	expected[39] = 0.708088;
	expected[40] = 0.719779;
	expected[41] = 0.729626;
	expected[42] = 0.736805;
	expected[43] = 0.736644;
	expected[44] = 0.726863;
	expected[45] = 0.843120;
	expected[46] = 0.859814;
	expected[47] = 0.873491;
	expected[48] = 0.880363;
	expected[49] = 0.877952;
	expected[50] = 0.863938;
	expected[51] = 0.846670;
	expected[52] = 0.815305;
	expected[53] = 0.767694;
	expected[54] = 0.701580;
	expected[55] = 0.614075;
	expected[56] = 0.502466;
	expected[57] = 0.364398;
	expected[58] = 0.197560;
	expected[59] = 218.204763;
	expected[60] = 208.536722;
	expected[61] = 201.365147;
	expected[62] = 196.613321;
	expected[63] = 192.225449;
	expected[64] = 187.854648;
	expected[65] = 183.444339;
	expected[66] = 179.024727;
	expected[67] = 174.838208;
	expected[68] = 170.462899;
	expected[69] = 166.138283;
	expected[70] = 161.844749;
	expected[71] = 157.628736;
	expected[72] = 153.515558;
	expected[73] = 149.527825;
	expected[74] = 145.682542;
	expected[75] = 141.988690;
	expected[76] = 138.440774;
	expected[77] = 135.017228;
	expected[78] = 131.684056;
	expected[79] = 128.454935;
	expected[80] = 125.186366;
	expected[81] = 121.911516;
	expected[82] = 118.672538;
	expected[83] = 115.447516;
	expected[84] = 112.245432;
	expected[85] = 109.072702;
	expected[86] = 105.935827;
	expected[87] = 102.837515;
	expected[88] = 99.783094;
	expected[89] = 96.771233;
	expected[90] = 93.799117;
	expected[91] = 90.868795;
	expected[92] = 88.056926;
	expected[93] = 85.072909;
	expected[94] = 82.286489;
	expected[95] = 79.283381;
	expected[96] = 76.432239;
	expected[97] = 73.503216;
	expected[98] = 70.590586;
	expected[99] = 67.533985;
	expected[100] = 64.183638;
	expected[101] = 60.609076;
	expected[102] = 56.608306;
	expected[103] = 52.578052;
	expected[104] = 47.845324;
	expected[105] = 44.926503;
	expected[106] = 41.789478;
	expected[107] = 38.491206;
	expected[108] = 35.059797;
	expected[109] = 31.581774;
	expected[110] = 28.048177;
	expected[111] = 24.605836;
	expected[112] = 21.241255;
	expected[113] = 17.932798;
	expected[114] = 14.695594;
	expected[115] = 11.558949;
	expected[116] = 8.480543;
	expected[117] = 5.507076;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_2(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.021237;
	expected[1] = -0.047843;
	expected[2] = -0.068639;
	expected[3] = -0.085020;
	expected[4] = -0.097301;
	expected[5] = -0.105764;
	expected[6] = -0.110635;
	expected[7] = -0.112117;
	expected[8] = -0.111017;
	expected[9] = -0.106460;
	expected[10] = -0.098576;
	expected[11] = -0.088101;
	expected[12] = -0.074913;
	expected[13] = -0.058763;
	expected[14] = -0.039891;
	expected[15] = -0.018500;
	expected[16] = 0.004293;
	expected[17] = 0.030304;
	expected[18] = 0.060141;
	expected[19] = 0.094130;
	expected[20] = 0.122613;
	expected[21] = 0.153425;
	expected[22] = 0.185904;
	expected[23] = 0.219238;
	expected[24] = 0.253186;
	expected[25] = 0.287501;
	expected[26] = 0.321984;
	expected[27] = 0.356323;
	expected[28] = 0.390243;
	expected[29] = 0.423529;
	expected[30] = 0.455975;
	expected[31] = 0.487278;
	expected[32] = 0.517232;
	expected[33] = 0.545688;
	expected[34] = 0.572602;
	expected[35] = 0.597904;
	expected[36] = 0.621862;
	expected[37] = 0.645117;
	expected[38] = 0.667820;
	expected[39] = 0.689442;
	expected[40] = 0.699758;
	expected[41] = 0.708946;
	expected[42] = 0.713845;
	expected[43] = 0.714725;
	expected[44] = 0.705642;
	expected[45] = 0.795698;
	expected[46] = 0.810910;
	expected[47] = 0.823267;
	expected[48] = 0.829694;
	expected[49] = 0.827197;
	expected[50] = 0.812436;
	expected[51] = 0.795233;
	expected[52] = 0.764423;
	expected[53] = 0.718185;
	expected[54] = 0.654780;
	expected[55] = 0.572059;
	expected[56] = 0.467250;
	expected[57] = 0.337881;
	expected[58] = 0.182377;
	expected[59] = 217.289984;
	expected[60] = 207.586750;
	expected[61] = 200.300361;
	expected[62] = 195.459023;
	expected[63] = 191.007986;
	expected[64] = 186.600972;
	expected[65] = 182.180575;
	expected[66] = 177.764526;
	expected[67] = 173.629304;
	expected[68] = 169.319348;
	expected[69] = 165.029708;
	expected[70] = 160.817318;
	expected[71] = 156.691967;
	expected[72] = 152.662494;
	expected[73] = 148.753298;
	expected[74] = 144.981812;
	expected[75] = 141.394077;
	expected[76] = 137.911792;
	expected[77] = 134.506234;
	expected[78] = 131.168117;
	expected[79] = 128.079392;
	expected[80] = 124.938880;
	expected[81] = 121.767590;
	expected[82] = 118.599011;
	expected[83] = 115.442072;
	expected[84] = 112.304654;
	expected[85] = 109.191526;
	expected[86] = 106.110769;
	expected[87] = 103.067822;
	expected[88] = 100.066356;
	expected[89] = 97.106152;
	expected[90] = 94.185079;
	expected[91] = 91.301797;
	expected[92] = 88.448364;
	expected[93] = 85.609442;
	expected[94] = 82.785200;
	expected[95] = 79.966765;
	expected[96] = 77.199096;
	expected[97] = 74.208132;
	expected[98] = 71.400995;
	expected[99] = 68.331903;
	expected[100] = 65.035098;
	expected[101] = 61.422056;
	expected[102] = 57.549787;
	expected[103] = 53.665895;
	expected[104] = 49.399379;
	expected[105] = 46.523667;
	expected[106] = 43.394295;
	expected[107] = 40.062131;
	expected[108] = 36.585960;
	expected[109] = 33.022845;
	expected[110] = 29.397968;
	expected[111] = 25.880070;
	expected[112] = 22.448019;
	expected[113] = 19.074730;
	expected[114] = 15.742950;
	expected[115] = 12.448621;
	expected[116] = 9.195554;
	expected[117] = 5.999285;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_3(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.025698;
	expected[1] = -0.050279;
	expected[2] = -0.071166;
	expected[3] = -0.087626;
	expected[4] = -0.100963;
	expected[5] = -0.109728;
	expected[6] = -0.115205;
	expected[7] = -0.116743;
	expected[8] = -0.115027;
	expected[9] = -0.110245;
	expected[10] = -0.102503;
	expected[11] = -0.091888;
	expected[12] = -0.078484;
	expected[13] = -0.062782;
	expected[14] = -0.044726;
	expected[15] = -0.023674;
	expected[16] = -0.000292;
	expected[17] = 0.025455;
	expected[18] = 0.053843;
	expected[19] = 0.086341;
	expected[20] = 0.116854;
	expected[21] = 0.148194;
	expected[22] = 0.180579;
	expected[23] = 0.213798;
	expected[24] = 0.247631;
	expected[25] = 0.281831;
	expected[26] = 0.316161;
	expected[27] = 0.350327;
	expected[28] = 0.384069;
	expected[29] = 0.417236;
	expected[30] = 0.449572;
	expected[31] = 0.480741;
	expected[32] = 0.510546;
	expected[33] = 0.538833;
	expected[34] = 0.565628;
	expected[35] = 0.590815;
	expected[36] = 0.614420;
	expected[37] = 0.637006;
	expected[38] = 0.659794;
	expected[39] = 0.682398;
	expected[40] = 0.693470;
	expected[41] = 0.702401;
	expected[42] = 0.708365;
	expected[43] = 0.709175;
	expected[44] = 0.697300;
	expected[45] = 0.812864;
	expected[46] = 0.827976;
	expected[47] = 0.840364;
	expected[48] = 0.846591;
	expected[49] = 0.844287;
	expected[50] = 0.830026;
	expected[51] = 0.812817;
	expected[52] = 0.782026;
	expected[53] = 0.735739;
	expected[54] = 0.671976;
	expected[55] = 0.588461;
	expected[56] = 0.481733;
	expected[57] = 0.349693;
	expected[58] = 0.189907;
	expected[59] = 217.994464;
	expected[60] = 208.150142;
	expected[61] = 200.889633;
	expected[62] = 196.069374;
	expected[63] = 191.707725;
	expected[64] = 187.322572;
	expected[65] = 182.935385;
	expected[66] = 178.514530;
	expected[67] = 174.328162;
	expected[68] = 169.990971;
	expected[69] = 165.692336;
	expected[70] = 161.454104;
	expected[71] = 157.297986;
	expected[72] = 153.268002;
	expected[73] = 149.365114;
	expected[74] = 145.580429;
	expected[75] = 141.944388;
	expected[76] = 138.452706;
	expected[77] = 135.086776;
	expected[78] = 131.777553;
	expected[79] = 128.602720;
	expected[80] = 125.425689;
	expected[81] = 122.241869;
	expected[82] = 119.060701;
	expected[83] = 115.890685;
	expected[84] = 112.739989;
	expected[85] = 109.615393;
	expected[86] = 106.524464;
	expected[87] = 103.472253;
	expected[88] = 100.460389;
	expected[89] = 97.490678;
	expected[90] = 94.563634;
	expected[91] = 91.676164;
	expected[92] = 88.820249;
	expected[93] = 85.982298;
	expected[94] = 83.156833;
	expected[95] = 80.335052;
	expected[96] = 77.496482;
	expected[97] = 74.623727;
	expected[98] = 71.634264;
	expected[99] = 68.650211;
	expected[100] = 65.307587;
	expected[101] = 61.615012;
	expected[102] = 57.614011;
	expected[103] = 53.531180;
	expected[104] = 48.739935;
	expected[105] = 45.820905;
	expected[106] = 42.702777;
	expected[107] = 39.392902;
	expected[108] = 35.945608;
	expected[109] = 32.415966;
	expected[110] = 28.828562;
	expected[111] = 25.335339;
	expected[112] = 21.916160;
	expected[113] = 18.548339;
	expected[114] = 15.302942;
	expected[115] = 11.966481;
	expected[116] = 8.756932;
	expected[117] = 5.682378;

	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_4(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.029485;
	expected[1] = -0.055845;
	expected[2] = -0.077477;
	expected[3] = -0.094572;
	expected[4] = -0.107456;
	expected[5] = -0.116736;
	expected[6] = -0.122132;
	expected[7] = -0.123922;
	expected[8] = -0.122750;
	expected[9] = -0.118197;
	expected[10] = -0.110550;
	expected[11] = -0.099962;
	expected[12] = -0.086529;
	expected[13] = -0.070316;
	expected[14] = -0.051462;
	expected[15] = -0.030090;
	expected[16] = -0.006307;
	expected[17] = 0.019945;
	expected[18] = 0.048996;
	expected[19] = 0.081420;
	expected[20] = 0.112564;
	expected[21] = 0.145070;
	expected[22] = 0.179275;
	expected[23] = 0.214043;
	expected[24] = 0.249737;
	expected[25] = 0.285315;
	expected[26] = 0.321087;
	expected[27] = 0.356754;
	expected[28] = 0.392077;
	expected[29] = 0.426872;
	expected[30] = 0.460867;
	expected[31] = 0.493699;
	expected[32] = 0.525241;
	expected[33] = 0.555838;
	expected[34] = 0.583685;
	expected[35] = 0.610590;
	expected[36] = 0.636034;
	expected[37] = 0.660126;
	expected[38] = 0.683890;
	expected[39] = 0.706102;
	expected[40] = 0.717846;
	expected[41] = 0.727847;
	expected[42] = 0.735204;
	expected[43] = 0.736240;
	expected[44] = 0.730032;
	expected[45] = 0.816309;
	expected[46] = 0.833021;
	expected[47] = 0.847814;
	expected[48] = 0.854418;
	expected[49] = 0.854076;
	expected[50] = 0.839381;
	expected[51] = 0.822190;
	expected[52] = 0.790914;
	expected[53] = 0.743502;
	expected[54] = 0.677835;
	expected[55] = 0.591597;
	expected[56] = 0.482065;
	expected[57] = 0.347257;
	expected[58] = 0.185862;
	expected[59] = 218.077187;
	expected[60] = 208.414694;
	expected[61] = 201.244126;
	expected[62] = 196.494624;
	expected[63] = 192.109144;
	expected[64] = 187.768571;
	expected[65] = 183.370716;
	expected[66] = 178.942380;
	expected[67] = 174.754680;
	expected[68] = 170.386116;
	expected[69] = 166.040771;
	expected[70] = 161.747893;
	expected[71] = 157.532757;
	expected[72] = 153.420188;
	expected[73] = 149.432729;
	expected[74] = 145.587243;
	expected[75] = 141.892685;
	expected[76] = 138.343223;
	expected[77] = 134.916496;
	expected[78] = 131.581785;
	expected[79] = 128.352468;
	expected[80] = 125.105379;
	expected[81] = 121.825946;
	expected[82] = 118.572372;
	expected[83] = 115.324488;
	expected[84] = 112.119984;
	expected[85] = 108.944809;
	expected[86] = 105.806135;
	expected[87] = 102.707027;
	expected[88] = 99.649408;
	expected[89] = 96.636560;
	expected[90] = 93.662547;
	expected[91] = 90.727333;
	expected[92] = 87.917784;
	expected[93] = 84.927084;
	expected[94] = 82.092102;
	expected[95] = 79.143203;
	expected[96] = 76.286335;
	expected[97] = 73.356897;
	expected[98] = 70.459835;
	expected[99] = 67.418194;
	expected[100] = 64.086365;
	expected[101] = 60.562340;
	expected[102] = 56.637119;
	expected[103] = 52.777390;
	expected[104] = 48.617428;
	expected[105] = 45.699803;
	expected[106] = 42.636298;
	expected[107] = 39.193548;
	expected[108] = 35.815053;
	expected[109] = 32.221504;
	expected[110] = 28.650484;
	expected[111] = 25.182518;
	expected[112] = 21.801899;
	expected[113] = 18.488880;
	expected[114] = 15.246296;
	expected[115] = 12.049690;
	expected[116] = 9.034642;
	expected[117] = 5.896829;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_5(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.031292;
	expected[1] = -0.058592;
	expected[2] = -0.079822;
	expected[3] = -0.096996;
	expected[4] = -0.110179;
	expected[5] = -0.119414;
	expected[6] = -0.124791;
	expected[7] = -0.126702;
	expected[8] = -0.125325;
	expected[9] = -0.120815;
	expected[10] = -0.113287;
	expected[11] = -0.102829;
	expected[12] = -0.089539;
	expected[13] = -0.073476;
	expected[14] = -0.054786;
	expected[15] = -0.033601;
	expected[16] = -0.010036;
	expected[17] = 0.015954;
	expected[18] = 0.044678;
	expected[19] = 0.076786;
	expected[20] = 0.107801;
	expected[21] = 0.140179;
	expected[22] = 0.173708;
	expected[23] = 0.208165;
	expected[24] = 0.243671;
	expected[25] = 0.279357;
	expected[26] = 0.315389;
	expected[27] = 0.351322;
	expected[28] = 0.386489;
	expected[29] = 0.421104;
	expected[30] = 0.454965;
	expected[31] = 0.487685;
	expected[32] = 0.519053;
	expected[33] = 0.548907;
	expected[34] = 0.577248;
	expected[35] = 0.603922;
	expected[36] = 0.628810;
	expected[37] = 0.653047;
	expected[38] = 0.676419;
	expected[39] = 0.699700;
	expected[40] = 0.712187;
	expected[41] = 0.721639;
	expected[42] = 0.728756;
	expected[43] = 0.730934;
	expected[44] = 0.722167;
	expected[45] = 0.832078;
	expected[46] = 0.848968;
	expected[47] = 0.862808;
	expected[48] = 0.870194;
	expected[49] = 0.868996;
	expected[50] = 0.855494;
	expected[51] = 0.838288;
	expected[52] = 0.807001;
	expected[53] = 0.759758;
	expected[54] = 0.693448;
	expected[55] = 0.606313;
	expected[56] = 0.495016;
	expected[57] = 0.357671;
	expected[58] = 0.192412;
	expected[59] = 218.571212;
	expected[60] = 209.033149;
	expected[61] = 201.855309;
	expected[62] = 197.135208;
	expected[63] = 192.790236;
	expected[64] = 188.453684;
	expected[65] = 184.058431;
	expected[66] = 179.637533;
	expected[67] = 175.430224;
	expected[68] = 171.052369;
	expected[69] = 166.696920;
	expected[70] = 162.389392;
	expected[71] = 158.155684;
	expected[72] = 154.021462;
	expected[73] = 150.010728;
	expected[74] = 146.142222;
	expected[75] = 142.426957;
	expected[76] = 138.861666;
	expected[77] = 135.426063;
	expected[78] = 132.084398;
	expected[79] = 128.843349;
	expected[80] = 125.583533;
	expected[81] = 122.317110;
	expected[82] = 119.055076;
	expected[83] = 115.790187;
	expected[84] = 112.561299;
	expected[85] = 109.358668;
	expected[86] = 106.193284;
	expected[87] = 103.084546;
	expected[88] = 100.020189;
	expected[89] = 96.998266;
	expected[90] = 94.018649;
	expected[91] = 91.081300;
	expected[92] = 88.178872;
	expected[93] = 85.295324;
	expected[94] = 82.442244;
	expected[95] = 79.529023;
	expected[96] = 76.788803;
	expected[97] = 73.724631;
	expected[98] = 70.742032;
	expected[99] = 67.703320;
	expected[100] = 64.314870;
	expected[101] = 60.648937;
	expected[102] = 56.695534;
	expected[103] = 52.642511;
	expected[104] = 48.023532;
	expected[105] = 45.095876;
	expected[106] = 41.946646;
	expected[107] = 38.609640;
	expected[108] = 35.178034;
	expected[109] = 31.665754;
	expected[110] = 28.126959;
	expected[111] = 24.681138;
	expected[112] = 21.381601;
	expected[113] = 18.014303;
	expected[114] = 14.796627;
	expected[115] = 11.638230;
	expected[116] = 8.582664;
	expected[117] = 5.605140;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_6(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.027066;
	expected[1] = -0.050921;
	expected[2] = -0.072570;
	expected[3] = -0.089778;
	expected[4] = -0.102693;
	expected[5] = -0.111257;
	expected[6] = -0.116227;
	expected[7] = -0.117811;
	expected[8] = -0.116182;
	expected[9] = -0.111494;
	expected[10] = -0.103853;
	expected[11] = -0.093536;
	expected[12] = -0.080559;
	expected[13] = -0.065092;
	expected[14] = -0.046600;
	expected[15] = -0.025659;
	expected[16] = -0.002386;
	expected[17] = 0.023260;
	expected[18] = 0.051576;
	expected[19] = 0.083074;
	expected[20] = 0.113254;
	expected[21] = 0.145097;
	expected[22] = 0.177543;
	expected[23] = 0.211177;
	expected[24] = 0.245263;
	expected[25] = 0.279435;
	expected[26] = 0.313739;
	expected[27] = 0.347885;
	expected[28] = 0.381615;
	expected[29] = 0.414758;
	expected[30] = 0.447110;
	expected[31] = 0.478314;
	expected[32] = 0.508163;
	expected[33] = 0.536501;
	expected[34] = 0.563275;
	expected[35] = 0.588512;
	expected[36] = 0.612192;
	expected[37] = 0.634827;
	expected[38] = 0.657657;
	expected[39] = 0.680278;
	expected[40] = 0.690651;
	expected[41] = 0.700246;
	expected[42] = 0.706563;
	expected[43] = 0.708495;
	expected[44] = 0.700353;
	expected[45] = 0.785286;
	expected[46] = 0.801304;
	expected[47] = 0.814336;
	expected[48] = 0.821468;
	expected[49] = 0.819719;
	expected[50] = 0.805681;
	expected[51] = 0.788518;
	expected[52] = 0.757646;
	expected[53] = 0.711434;
	expected[54] = 0.648116;
	expected[55] = 0.565564;
	expected[56] = 0.461058;
	expected[57] = 0.332242;
	expected[58] = 0.178020;
	expected[59] = 217.944306;
	expected[60] = 208.039758;
	expected[61] = 200.838749;
	expected[62] = 196.067901;
	expected[63] = 191.661960;
	expected[64] = 187.259973;
	expected[65] = 182.839421;
	expected[66] = 178.418664;
	expected[67] = 174.234347;
	expected[68] = 169.898973;
	expected[69] = 165.601732;
	expected[70] = 161.375848;
	expected[71] = 157.237222;
	expected[72] = 153.206448;
	expected[73] = 149.281437;
	expected[74] = 145.495824;
	expected[75] = 141.858251;
	expected[76] = 138.363977;
	expected[77] = 134.993852;
	expected[78] = 131.721812;
	expected[79] = 128.552632;
	expected[80] = 125.347472;
	expected[81] = 122.157165;
	expected[82] = 118.957612;
	expected[83] = 115.775156;
	expected[84] = 112.621792;
	expected[85] = 109.494396;
	expected[86] = 106.400459;
	expected[87] = 103.345131;
	expected[88] = 100.331662;
	expected[89] = 97.360004;
	expected[90] = 94.429147;
	expected[91] = 91.538563;
	expected[92] = 88.680742;
	expected[93] = 85.840060;
	expected[94] = 83.013456;
	expected[95] = 80.192268;
	expected[96] = 77.355537;
	expected[97] = 74.469690;
	expected[98] = 71.507383;
	expected[99] = 68.542013;
	expected[100] = 65.212063;
	expected[101] = 61.562872;
	expected[102] = 57.655446;
	expected[103] = 53.746039;
	expected[104] = 49.585619;
	expected[105] = 46.685545;
	expected[106] = 43.536959;
	expected[107] = 40.183560;
	expected[108] = 36.685748;
	expected[109] = 33.103238;
	expected[110] = 29.473626;
	expected[111] = 25.956418;
	expected[112] = 22.526640;
	expected[113] = 19.158390;
	expected[114] = 15.835021;
	expected[115] = 12.551648;
	expected[116] = 9.312875;
	expected[117] = 6.098522;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_7(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.029071;
	expected[1] = -0.054716;
	expected[2] = -0.075742;
	expected[3] = -0.092343;
	expected[4] = -0.104840;
	expected[5] = -0.113512;
	expected[6] = -0.118595;
	expected[7] = -0.120294;
	expected[8] = -0.118781;
	expected[9] = -0.114643;
	expected[10] = -0.107157;
	expected[11] = -0.097218;
	expected[12] = -0.084135;
	expected[13] = -0.068350;
	expected[14] = -0.050013;
	expected[15] = -0.029256;
	expected[16] = -0.006201;
	expected[17] = 0.019178;
	expected[18] = 0.047160;
	expected[19] = 0.078364;
	expected[20] = 0.108362;
	expected[21] = 0.139626;
	expected[22] = 0.171942;
	expected[23] = 0.205101;
	expected[24] = 0.239067;
	expected[25] = 0.273472;
	expected[26] = 0.307710;
	expected[27] = 0.341790;
	expected[28] = 0.375724;
	expected[29] = 0.409097;
	expected[30] = 0.441339;
	expected[31] = 0.472414;
	expected[32] = 0.502123;
	expected[33] = 0.530342;
	expected[34] = 0.557010;
	expected[35] = 0.582043;
	expected[36] = 0.605543;
	expected[37] = 0.628080;
	expected[38] = 0.650152;
	expected[39] = 0.672866;
	expected[40] = 0.684783;
	expected[41] = 0.694323;
	expected[42] = 0.701507;
	expected[43] = 0.703109;
	expected[44] = 0.692439;
	expected[45] = 0.800868;
	expected[46] = 0.816440;
	expected[47] = 0.830376;
	expected[48] = 0.836804;
	expected[49] = 0.835181;
	expected[50] = 0.821683;
	expected[51] = 0.804513;
	expected[52] = 0.773743;
	expected[53] = 0.727473;
	expected[54] = 0.663759;
	expected[55] = 0.580323;
	expected[56] = 0.474126;
	expected[57] = 0.342837;
	expected[58] = 0.184625;
	expected[59] = 218.455357;
	expected[60] = 208.720503;
	expected[61] = 201.483231;
	expected[62] = 196.683181;
	expected[63] = 192.264245;
	expected[64] = 187.879337;
	expected[65] = 183.470816;
	expected[66] = 179.056800;
	expected[67] = 174.874092;
	expected[68] = 170.564076;
	expected[69] = 166.260565;
	expected[70] = 162.031709;
	expected[71] = 157.863394;
	expected[72] = 153.796803;
	expected[73] = 149.853177;
	expected[74] = 146.049145;
	expected[75] = 142.395041;
	expected[76] = 138.888591;
	expected[77] = 135.512508;
	expected[78] = 132.234932;
	expected[79] = 129.058804;
	expected[80] = 125.863495;
	expected[81] = 122.660318;
	expected[82] = 119.459032;
	expected[83] = 116.260204;
	expected[84] = 113.077783;
	expected[85] = 109.935190;
	expected[86] = 106.826661;
	expected[87] = 103.748169;
	expected[88] = 100.711899;
	expected[89] = 97.730880;
	expected[90] = 94.794296;
	expected[91] = 91.899503;
	expected[92] = 89.036967;
	expected[93] = 86.196618;
	expected[94] = 83.372059;
	expected[95] = 80.550904;
	expected[96] = 77.712265;
	expected[97] = 74.835024;
	expected[98] = 71.866003;
	expected[99] = 68.844792;
	expected[100] = 65.477229;
	expected[101] = 61.749255;
	expected[102] = 57.717188;
	expected[103] = 53.605071;
	expected[104] = 48.949754;
	expected[105] = 46.025493;
	expected[106] = 42.980207;
	expected[107] = 39.544186;
	expected[108] = 36.074089;
	expected[109] = 32.523453;
	expected[110] = 28.930278;
	expected[111] = 25.435118;
	expected[112] = 22.018257;
	expected[113] = 18.656542;
	expected[114] = 15.344417;
	expected[115] = 12.088958;
	expected[116] = 8.916150;
	expected[117] = 5.789080;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_8(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.026351;
	expected[1] = -0.051250;
	expected[2] = -0.073673;
	expected[3] = -0.091655;
	expected[4] = -0.104356;
	expected[5] = -0.113119;
	expected[6] = -0.118184;
	expected[7] = -0.119763;
	expected[8] = -0.118029;
	expected[9] = -0.113143;
	expected[10] = -0.105214;
	expected[11] = -0.094566;
	expected[12] = -0.081089;
	expected[13] = -0.065070;
	expected[14] = -0.045923;
	expected[15] = -0.024232;
	expected[16] = -0.000097;
	expected[17] = 0.026554;
	expected[18] = 0.056039;
	expected[19] = 0.090244;
	expected[20] = 0.121443;
	expected[21] = 0.153987;
	expected[22] = 0.187671;
	expected[23] = 0.222275;
	expected[24] = 0.257564;
	expected[25] = 0.293299;
	expected[26] = 0.329291;
	expected[27] = 0.365172;
	expected[28] = 0.400752;
	expected[29] = 0.435724;
	expected[30] = 0.470515;
	expected[31] = 0.502687;
	expected[32] = 0.534287;
	expected[33] = 0.564713;
	expected[34] = 0.593868;
	expected[35] = 0.620574;
	expected[36] = 0.646897;
	expected[37] = 0.670056;
	expected[38] = 0.692626;
	expected[39] = 0.715262;
	expected[40] = 0.726351;
	expected[41] = 0.735325;
	expected[42] = 0.741261;
	expected[43] = 0.742425;
	expected[44] = 0.735223;
	expected[45] = 0.827506;
	expected[46] = 0.842899;
	expected[47] = 0.855694;
	expected[48] = 0.864061;
	expected[49] = 0.860384;
	expected[50] = 0.845984;
	expected[51] = 0.828763;
	expected[52] = 0.797468;
	expected[53] = 0.750070;
	expected[54] = 0.684427;
	expected[55] = 0.597990;
	expected[56] = 0.488181;
	expected[57] = 0.352671;
	expected[58] = 0.190091;
	expected[59] = 217.648838;
	expected[60] = 207.833938;
	expected[61] = 200.719156;
	expected[62] = 196.019988;
	expected[63] = 191.606805;
	expected[64] = 187.216397;
	expected[65] = 182.792823;
	expected[66] = 178.356059;
	expected[67] = 174.145105;
	expected[68] = 169.775062;
	expected[69] = 165.437768;
	expected[70] = 161.173061;
	expected[71] = 156.990111;
	expected[72] = 152.918856;
	expected[73] = 148.952783;
	expected[74] = 145.128482;
	expected[75] = 141.452998;
	expected[76] = 137.918366;
	expected[77] = 134.501370;
	expected[78] = 131.114643;
	expected[79] = 127.905035;
	expected[80] = 124.679086;
	expected[81] = 121.447996;
	expected[82] = 118.222015;
	expected[83] = 115.010283;
	expected[84] = 111.820436;
	expected[85] = 108.656457;
	expected[86] = 105.528395;
	expected[87] = 102.437469;
	expected[88] = 99.389677;
	expected[89] = 96.413330;
	expected[90] = 93.415099;
	expected[91] = 90.484666;
	expected[92] = 87.618946;
	expected[93] = 84.648477;
	expected[94] = 81.811672;
	expected[95] = 78.901554;
	expected[96] = 76.057555;
	expected[97] = 73.196698;
	expected[98] = 70.256213;
	expected[99] = 67.271028;
	expected[100] = 63.929479;
	expected[101] = 60.331226;
	expected[102] = 56.538756;
	expected[103] = 52.701663;
	expected[104] = 48.462447;
	expected[105] = 45.537227;
	expected[106] = 42.411245;
	expected[107] = 39.193655;
	expected[108] = 35.648011;
	expected[109] = 32.145626;
	expected[110] = 28.578188;
	expected[111] = 25.110651;
	expected[112] = 21.734095;
	expected[113] = 18.530438;
	expected[114] = 15.147845;
	expected[115] = 12.069098;
	expected[116] = 8.839518;
	expected[117] = 5.806405;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_9(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.028965;
	expected[1] = -0.055221;
	expected[2] = -0.076814;
	expected[3] = -0.093851;
	expected[4] = -0.106670;
	expected[5] = -0.115556;
	expected[6] = -0.120748;
	expected[7] = -0.122582;
	expected[8] = -0.121291;
	expected[9] = -0.116547;
	expected[10] = -0.109181;
	expected[11] = -0.098497;
	expected[12] = -0.084959;
	expected[13] = -0.068636;
	expected[14] = -0.049672;
	expected[15] = -0.028193;
	expected[16] = -0.004304;
	expected[17] = 0.022055;
	expected[18] = 0.051186;
	expected[19] = 0.083677;
	expected[20] = 0.114902;
	expected[21] = 0.147934;
	expected[22] = 0.182301;
	expected[23] = 0.216778;
	expected[24] = 0.251945;
	expected[25] = 0.287540;
	expected[26] = 0.323330;
	expected[27] = 0.359029;
	expected[28] = 0.394401;
	expected[29] = 0.429173;
	expected[30] = 0.463133;
	expected[31] = 0.495973;
	expected[32] = 0.527489;
	expected[33] = 0.558092;
	expected[34] = 0.585786;
	expected[35] = 0.612844;
	expected[36] = 0.638328;
	expected[37] = 0.662149;
	expected[38] = 0.685681;
	expected[39] = 0.708088;
	expected[40] = 0.719779;
	expected[41] = 0.729626;
	expected[42] = 0.736805;
	expected[43] = 0.736644;
	expected[44] = 0.726863;
	expected[45] = 0.843120;
	expected[46] = 0.859814;
	expected[47] = 0.873491;
	expected[48] = 0.880363;
	expected[49] = 0.877952;
	expected[50] = 0.863938;
	expected[51] = 0.846670;
	expected[52] = 0.815305;
	expected[53] = 0.767694;
	expected[54] = 0.701580;
	expected[55] = 0.614075;
	expected[56] = 0.502466;
	expected[57] = 0.364398;
	expected[58] = 0.197560;
	expected[59] = 218.204763;
	expected[60] = 208.536722;
	expected[61] = 201.365147;
	expected[62] = 196.613321;
	expected[63] = 192.225449;
	expected[64] = 187.854648;
	expected[65] = 183.444339;
	expected[66] = 179.024727;
	expected[67] = 174.838208;
	expected[68] = 170.462899;
	expected[69] = 166.138283;
	expected[70] = 161.844749;
	expected[71] = 157.628736;
	expected[72] = 153.515558;
	expected[73] = 149.527825;
	expected[74] = 145.682542;
	expected[75] = 141.988690;
	expected[76] = 138.440774;
	expected[77] = 135.017228;
	expected[78] = 131.684056;
	expected[79] = 128.454935;
	expected[80] = 125.186366;
	expected[81] = 121.911516;
	expected[82] = 118.672538;
	expected[83] = 115.447516;
	expected[84] = 112.245432;
	expected[85] = 109.072702;
	expected[86] = 105.935827;
	expected[87] = 102.837515;
	expected[88] = 99.783094;
	expected[89] = 96.771233;
	expected[90] = 93.799117;
	expected[91] = 90.868795;
	expected[92] = 88.056926;
	expected[93] = 85.072909;
	expected[94] = 82.286489;
	expected[95] = 79.283381;
	expected[96] = 76.432239;
	expected[97] = 73.503216;
	expected[98] = 70.590586;
	expected[99] = 67.533985;
	expected[100] = 64.183638;
	expected[101] = 60.609076;
	expected[102] = 56.608306;
	expected[103] = 52.578052;
	expected[104] = 47.845324;
	expected[105] = 44.926503;
	expected[106] = 41.789478;
	expected[107] = 38.491206;
	expected[108] = 35.059797;
	expected[109] = 31.581774;
	expected[110] = 28.048177;
	expected[111] = 24.605836;
	expected[112] = 21.241255;
	expected[113] = 17.932798;
	expected[114] = 14.695594;
	expected[115] = 11.558949;
	expected[116] = 8.480543;
	expected[117] = 5.507076;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_10(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.022834;
	expected[1] = -0.048423;
	expected[2] = -0.069212;
	expected[3] = -0.085586;
	expected[4] = -0.097862;
	expected[5] = -0.106320;
	expected[6] = -0.111538;
	expected[7] = -0.113619;
	expected[8] = -0.111840;
	expected[9] = -0.107297;
	expected[10] = -0.099845;
	expected[11] = -0.089158;
	expected[12] = -0.075610;
	expected[13] = -0.059325;
	expected[14] = -0.040437;
	expected[15] = -0.019302;
	expected[16] = 0.003346;
	expected[17] = 0.029371;
	expected[18] = 0.059527;
	expected[19] = 0.092310;
	expected[20] = 0.121411;
	expected[21] = 0.152777;
	expected[22] = 0.185255;
	expected[23] = 0.218574;
	expected[24] = 0.252508;
	expected[25] = 0.286810;
	expected[26] = 0.321260;
	expected[27] = 0.355587;
	expected[28] = 0.389494;
	expected[29] = 0.422773;
	expected[30] = 0.455196;
	expected[31] = 0.486495;
	expected[32] = 0.516447;
	expected[33] = 0.544899;
	expected[34] = 0.571809;
	expected[35] = 0.597107;
	expected[36] = 0.620849;
	expected[37] = 0.644462;
	expected[38] = 0.667114;
	expected[39] = 0.688396;
	expected[40] = 0.699020;
	expected[41] = 0.707511;
	expected[42] = 0.713028;
	expected[43] = 0.713953;
	expected[44] = 0.704877;
	expected[45] = 0.794798;
	expected[46] = 0.810126;
	expected[47] = 0.822475;
	expected[48] = 0.828897;
	expected[49] = 0.826407;
	expected[50] = 0.811647;
	expected[51] = 0.794451;
	expected[52] = 0.763645;
	expected[53] = 0.717410;
	expected[54] = 0.653983;
	expected[55] = 0.571305;
	expected[56] = 0.466533;
	expected[57] = 0.337228;
	expected[58] = 0.181870;
	expected[59] = 217.445262;
	expected[60] = 207.657697;
	expected[61] = 200.372423;
	expected[62] = 195.531646;
	expected[63] = 191.080574;
	expected[64] = 186.672947;
	expected[65] = 182.275737;
	expected[66] = 177.896160;
	expected[67] = 173.712989;
	expected[68] = 169.397849;
	expected[69] = 165.125152;
	expected[70] = 160.898243;
	expected[71] = 156.753229;
	expected[72] = 152.714732;
	expected[73] = 148.801467;
	expected[74] = 145.037999;
	expected[75] = 141.450518;
	expected[76] = 137.966836;
	expected[77] = 134.544511;
	expected[78] = 131.244467;
	expected[79] = 128.136994;
	expected[80] = 124.972354;
	expected[81] = 121.799110;
	expected[82] = 118.628993;
	expected[83] = 115.470473;
	expected[84] = 112.331448;
	expected[85] = 109.217578;
	expected[86] = 106.135222;
	expected[87] = 103.090750;
	expected[88] = 100.087779;
	expected[89] = 97.126864;
	expected[90] = 94.204610;
	expected[91] = 91.320316;
	expected[92] = 88.466173;
	expected[93] = 85.626732;
	expected[94] = 82.802253;
	expected[95] = 79.981882;
	expected[96] = 77.292651;
	expected[97] = 74.223516;
	expected[98] = 71.336932;
	expected[99] = 68.347762;
	expected[100] = 65.057017;
	expected[101] = 61.436678;
	expected[102] = 57.562183;
	expected[103] = 53.677628;
	expected[104] = 49.413265;
	expected[105] = 46.534451;
	expected[106] = 43.405049;
	expected[107] = 40.072385;
	expected[108] = 36.595472;
	expected[109] = 33.031742;
	expected[110] = 29.406375;
	expected[111] = 25.888365;
	expected[112] = 22.456674;
	expected[113] = 19.084796;
	expected[114] = 15.753552;
	expected[115] = 12.460506;
	expected[116] = 9.209182;
	expected[117] = 6.010570;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_11(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.026620;
	expected[1] = -0.050823;
	expected[2] = -0.071707;
	expected[3] = -0.089174;
	expected[4] = -0.102052;
	expected[5] = -0.110832;
	expected[6] = -0.115768;
	expected[7] = -0.117277;
	expected[8] = -0.115566;
	expected[9] = -0.110789;
	expected[10] = -0.103054;
	expected[11] = -0.092444;
	expected[12] = -0.079111;
	expected[13] = -0.063525;
	expected[14] = -0.045329;
	expected[15] = -0.024288;
	expected[16] = -0.000921;
	expected[17] = 0.024809;
	expected[18] = 0.053176;
	expected[19] = 0.085168;
	expected[20] = 0.115567;
	expected[21] = 0.147403;
	expected[22] = 0.179890;
	expected[23] = 0.213094;
	expected[24] = 0.246913;
	expected[25] = 0.281099;
	expected[26] = 0.315415;
	expected[27] = 0.349566;
	expected[28] = 0.383294;
	expected[29] = 0.416435;
	expected[30] = 0.448767;
	expected[31] = 0.479932;
	expected[32] = 0.509734;
	expected[33] = 0.538017;
	expected[34] = 0.564787;
	expected[35] = 0.589971;
	expected[36] = 0.613573;
	expected[37] = 0.636155;
	expected[38] = 0.658933;
	expected[39] = 0.681638;
	expected[40] = 0.692169;
	expected[41] = 0.701564;
	expected[42] = 0.707515;
	expected[43] = 0.708366;
	expected[44] = 0.696503;
	expected[45] = 0.812109;
	expected[46] = 0.827671;
	expected[47] = 0.839569;
	expected[48] = 0.846335;
	expected[49] = 0.843471;
	expected[50] = 0.829216;
	expected[51] = 0.812018;
	expected[52] = 0.781231;
	expected[53] = 0.734948;
	expected[54] = 0.671189;
	expected[55] = 0.587627;
	expected[56] = 0.481008;
	expected[57] = 0.349038;
	expected[58] = 0.189401;
	expected[59] = 218.091644;
	expected[60] = 208.217812;
	expected[61] = 200.959018;
	expected[62] = 196.218266;
	expected[63] = 191.813838;
	expected[64] = 187.427792;
	expected[65] = 183.007010;
	expected[66] = 178.583044;
	expected[67] = 174.395137;
	expected[68] = 170.055935;
	expected[69] = 165.754893;
	expected[70] = 161.513928;
	expected[71] = 157.358277;
	expected[72] = 153.329736;
	expected[73] = 149.417106;
	expected[74] = 145.629258;
	expected[75] = 141.990345;
	expected[76] = 138.496228;
	expected[77] = 135.128390;
	expected[78] = 131.836237;
	expected[79] = 128.661901;
	expected[80] = 125.465262;
	expected[81] = 122.275859;
	expected[82] = 119.093207;
	expected[83] = 115.921678;
	expected[84] = 112.769469;
	expected[85] = 109.643382;
	expected[86] = 106.551018;
	expected[87] = 103.497456;
	expected[88] = 100.485033;
	expected[89] = 97.514045;
	expected[90] = 94.585873;
	expected[91] = 91.697471;
	expected[92] = 88.840826;
	expected[93] = 86.003135;
	expected[94] = 83.177321;
	expected[95] = 80.355353;
	expected[96] = 77.516725;
	expected[97] = 74.623926;
	expected[98] = 71.652901;
	expected[99] = 68.676088;
	expected[100] = 65.325857;
	expected[101] = 61.632038;
	expected[102] = 57.628678;
	expected[103] = 53.544956;
	expected[104] = 48.747040;
	expected[105] = 45.924135;
	expected[106] = 42.714987;
	expected[107] = 39.453596;
	expected[108] = 35.956441;
	expected[109] = 32.426061;
	expected[110] = 28.838014;
	expected[111] = 25.344585;
	expected[112] = 21.925607;
	expected[113] = 18.558392;
	expected[114] = 15.250239;
	expected[115] = 11.976983;
	expected[116] = 8.769766;
	expected[117] = 5.692520;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_12(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.029485;
	expected[1] = -0.055845;
	expected[2] = -0.077477;
	expected[3] = -0.094572;
	expected[4] = -0.107456;
	expected[5] = -0.116736;
	expected[6] = -0.122132;
	expected[7] = -0.123922;
	expected[8] = -0.122750;
	expected[9] = -0.118197;
	expected[10] = -0.110550;
	expected[11] = -0.099962;
	expected[12] = -0.086529;
	expected[13] = -0.070316;
	expected[14] = -0.051462;
	expected[15] = -0.030090;
	expected[16] = -0.006307;
	expected[17] = 0.019945;
	expected[18] = 0.048996;
	expected[19] = 0.081420;
	expected[20] = 0.112564;
	expected[21] = 0.145070;
	expected[22] = 0.179275;
	expected[23] = 0.214043;
	expected[24] = 0.249737;
	expected[25] = 0.285315;
	expected[26] = 0.321087;
	expected[27] = 0.356754;
	expected[28] = 0.392077;
	expected[29] = 0.426872;
	expected[30] = 0.460867;
	expected[31] = 0.493699;
	expected[32] = 0.525241;
	expected[33] = 0.555838;
	expected[34] = 0.583685;
	expected[35] = 0.610590;
	expected[36] = 0.636034;
	expected[37] = 0.660126;
	expected[38] = 0.683890;
	expected[39] = 0.706102;
	expected[40] = 0.717846;
	expected[41] = 0.727847;
	expected[42] = 0.735204;
	expected[43] = 0.736240;
	expected[44] = 0.730032;
	expected[45] = 0.816309;
	expected[46] = 0.833021;
	expected[47] = 0.847814;
	expected[48] = 0.854418;
	expected[49] = 0.854076;
	expected[50] = 0.839381;
	expected[51] = 0.822190;
	expected[52] = 0.790914;
	expected[53] = 0.743502;
	expected[54] = 0.677835;
	expected[55] = 0.591597;
	expected[56] = 0.482065;
	expected[57] = 0.347257;
	expected[58] = 0.185862;
	expected[59] = 218.077187;
	expected[60] = 208.414694;
	expected[61] = 201.244126;
	expected[62] = 196.494624;
	expected[63] = 192.109144;
	expected[64] = 187.768571;
	expected[65] = 183.370716;
	expected[66] = 178.942380;
	expected[67] = 174.754680;
	expected[68] = 170.386116;
	expected[69] = 166.040771;
	expected[70] = 161.747893;
	expected[71] = 157.532757;
	expected[72] = 153.420188;
	expected[73] = 149.432729;
	expected[74] = 145.587243;
	expected[75] = 141.892685;
	expected[76] = 138.343223;
	expected[77] = 134.916496;
	expected[78] = 131.581785;
	expected[79] = 128.352468;
	expected[80] = 125.105379;
	expected[81] = 121.825946;
	expected[82] = 118.572372;
	expected[83] = 115.324488;
	expected[84] = 112.119984;
	expected[85] = 108.944809;
	expected[86] = 105.806135;
	expected[87] = 102.707027;
	expected[88] = 99.649408;
	expected[89] = 96.636560;
	expected[90] = 93.662547;
	expected[91] = 90.727333;
	expected[92] = 87.917784;
	expected[93] = 84.927084;
	expected[94] = 82.092102;
	expected[95] = 79.143203;
	expected[96] = 76.286335;
	expected[97] = 73.356897;
	expected[98] = 70.459835;
	expected[99] = 67.418194;
	expected[100] = 64.086365;
	expected[101] = 60.562340;
	expected[102] = 56.637119;
	expected[103] = 52.777390;
	expected[104] = 48.617428;
	expected[105] = 45.699803;
	expected[106] = 42.636298;
	expected[107] = 39.193548;
	expected[108] = 35.815053;
	expected[109] = 32.221504;
	expected[110] = 28.650484;
	expected[111] = 25.182518;
	expected[112] = 21.801899;
	expected[113] = 18.488880;
	expected[114] = 15.246296;
	expected[115] = 12.049690;
	expected[116] = 9.034642;
	expected[117] = 5.896829;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_13(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.031292;
	expected[1] = -0.058592;
	expected[2] = -0.079822;
	expected[3] = -0.096996;
	expected[4] = -0.110179;
	expected[5] = -0.119414;
	expected[6] = -0.124791;
	expected[7] = -0.126702;
	expected[8] = -0.125325;
	expected[9] = -0.120815;
	expected[10] = -0.113287;
	expected[11] = -0.102829;
	expected[12] = -0.089539;
	expected[13] = -0.073476;
	expected[14] = -0.054786;
	expected[15] = -0.033601;
	expected[16] = -0.010036;
	expected[17] = 0.015954;
	expected[18] = 0.044678;
	expected[19] = 0.076786;
	expected[20] = 0.107801;
	expected[21] = 0.140179;
	expected[22] = 0.173708;
	expected[23] = 0.208165;
	expected[24] = 0.243671;
	expected[25] = 0.279357;
	expected[26] = 0.315389;
	expected[27] = 0.351322;
	expected[28] = 0.386489;
	expected[29] = 0.421104;
	expected[30] = 0.454965;
	expected[31] = 0.487685;
	expected[32] = 0.519053;
	expected[33] = 0.548907;
	expected[34] = 0.577248;
	expected[35] = 0.603922;
	expected[36] = 0.628810;
	expected[37] = 0.653047;
	expected[38] = 0.676419;
	expected[39] = 0.699700;
	expected[40] = 0.712187;
	expected[41] = 0.721639;
	expected[42] = 0.728756;
	expected[43] = 0.730934;
	expected[44] = 0.722167;
	expected[45] = 0.832078;
	expected[46] = 0.848968;
	expected[47] = 0.862808;
	expected[48] = 0.870194;
	expected[49] = 0.868996;
	expected[50] = 0.855494;
	expected[51] = 0.838288;
	expected[52] = 0.807001;
	expected[53] = 0.759758;
	expected[54] = 0.693448;
	expected[55] = 0.606313;
	expected[56] = 0.495016;
	expected[57] = 0.357671;
	expected[58] = 0.192412;
	expected[59] = 218.571212;
	expected[60] = 209.033149;
	expected[61] = 201.855309;
	expected[62] = 197.135208;
	expected[63] = 192.790236;
	expected[64] = 188.453684;
	expected[65] = 184.058431;
	expected[66] = 179.637533;
	expected[67] = 175.430224;
	expected[68] = 171.052369;
	expected[69] = 166.696920;
	expected[70] = 162.389392;
	expected[71] = 158.155684;
	expected[72] = 154.021462;
	expected[73] = 150.010728;
	expected[74] = 146.142222;
	expected[75] = 142.426957;
	expected[76] = 138.861666;
	expected[77] = 135.426063;
	expected[78] = 132.084398;
	expected[79] = 128.843349;
	expected[80] = 125.583533;
	expected[81] = 122.317110;
	expected[82] = 119.055076;
	expected[83] = 115.790187;
	expected[84] = 112.561299;
	expected[85] = 109.358668;
	expected[86] = 106.193284;
	expected[87] = 103.084546;
	expected[88] = 100.020189;
	expected[89] = 96.998266;
	expected[90] = 94.018649;
	expected[91] = 91.081300;
	expected[92] = 88.178872;
	expected[93] = 85.295324;
	expected[94] = 82.442244;
	expected[95] = 79.529023;
	expected[96] = 76.788803;
	expected[97] = 73.724631;
	expected[98] = 70.742032;
	expected[99] = 67.703320;
	expected[100] = 64.314870;
	expected[101] = 60.648937;
	expected[102] = 56.695534;
	expected[103] = 52.642511;
	expected[104] = 48.023532;
	expected[105] = 45.095876;
	expected[106] = 41.946646;
	expected[107] = 38.609640;
	expected[108] = 35.178034;
	expected[109] = 31.665754;
	expected[110] = 28.126959;
	expected[111] = 24.681138;
	expected[112] = 21.381601;
	expected[113] = 18.014303;
	expected[114] = 14.796627;
	expected[115] = 11.638230;
	expected[116] = 8.582664;
	expected[117] = 5.605140;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_14(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.027895;
	expected[1] = -0.051410;
	expected[2] = -0.073881;
	expected[3] = -0.090811;
	expected[4] = -0.103207;
	expected[5] = -0.111775;
	expected[6] = -0.116749;
	expected[7] = -0.118339;
	expected[8] = -0.116715;
	expected[9] = -0.112034;
	expected[10] = -0.104398;
	expected[11] = -0.094383;
	expected[12] = -0.081571;
	expected[13] = -0.065678;
	expected[14] = -0.047197;
	expected[15] = -0.026268;
	expected[16] = -0.003009;
	expected[17] = 0.022620;
	expected[18] = 0.050916;
	expected[19] = 0.082396;
	expected[20] = 0.112488;
	expected[21] = 0.144013;
	expected[22] = 0.176780;
	expected[23] = 0.210052;
	expected[24] = 0.244298;
	expected[25] = 0.278709;
	expected[26] = 0.312998;
	expected[27] = 0.347131;
	expected[28] = 0.380846;
	expected[29] = 0.413981;
	expected[30] = 0.446314;
	expected[31] = 0.477514;
	expected[32] = 0.507360;
	expected[33] = 0.535693;
	expected[34] = 0.562463;
	expected[35] = 0.587677;
	expected[36] = 0.611353;
	expected[37] = 0.633985;
	expected[38] = 0.656831;
	expected[39] = 0.679883;
	expected[40] = 0.689822;
	expected[41] = 0.699194;
	expected[42] = 0.705720;
	expected[43] = 0.707690;
	expected[44] = 0.699555;
	expected[45] = 0.784485;
	expected[46] = 0.800493;
	expected[47] = 0.813519;
	expected[48] = 0.820653;
	expected[49] = 0.818905;
	expected[50] = 0.804870;
	expected[51] = 0.787658;
	expected[52] = 0.756841;
	expected[53] = 0.710634;
	expected[54] = 0.647326;
	expected[55] = 0.564796;
	expected[56] = 0.460331;
	expected[57] = 0.331581;
	expected[58] = 0.177512;
	expected[59] = 218.032210;
	expected[60] = 208.105091;
	expected[61] = 200.967335;
	expected[62] = 196.172800;
	expected[63] = 191.733018;
	expected[64] = 187.331356;
	expected[65] = 182.910448;
	expected[66] = 178.488685;
	expected[67] = 174.302775;
	expected[68] = 169.965311;
	expected[69] = 165.665565;
	expected[70] = 161.453807;
	expected[71] = 157.314249;
	expected[72] = 153.262473;
	expected[73] = 149.334105;
	expected[74] = 145.545240;
	expected[75] = 141.904700;
	expected[76] = 138.407915;
	expected[77] = 135.035810;
	expected[78] = 131.762124;
	expected[79] = 128.594994;
	expected[80] = 125.402219;
	expected[81] = 122.194244;
	expected[82] = 119.005029;
	expected[83] = 115.814920;
	expected[84] = 112.651276;
	expected[85] = 109.522296;
	expected[86] = 106.426823;
	expected[87] = 103.370043;
	expected[88] = 100.355173;
	expected[89] = 97.382930;
	expected[90] = 94.450900;
	expected[91] = 91.559370;
	expected[92] = 88.700847;
	expected[93] = 85.859621;
	expected[94] = 83.033451;
	expected[95] = 80.212176;
	expected[96] = 77.375354;
	expected[97] = 74.481783;
	expected[98] = 71.639427;
	expected[99] = 68.560571;
	expected[100] = 65.232860;
	expected[101] = 61.579488;
	expected[102] = 57.669845;
	expected[103] = 53.759673;
	expected[104] = 49.598598;
	expected[105] = 46.698338;
	expected[106] = 43.549399;
	expected[107] = 40.195256;
	expected[108] = 36.696695;
	expected[109] = 33.113457;
	expected[110] = 29.484611;
	expected[111] = 25.965993;
	expected[112] = 22.536433;
	expected[113] = 19.168731;
	expected[114] = 15.846265;
	expected[115] = 12.563958;
	expected[116] = 9.326735;
	expected[117] = 6.110176;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_15(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.025425;
	expected[1] = -0.055233;
	expected[2] = -0.076263;
	expected[3] = -0.092869;
	expected[4] = -0.105371;
	expected[5] = -0.114048;
	expected[6] = -0.119137;
	expected[7] = -0.120842;
	expected[8] = -0.119758;
	expected[9] = -0.115208;
	expected[10] = -0.108120;
	expected[11] = -0.097804;
	expected[12] = -0.084731;
	expected[13] = -0.068956;
	expected[14] = -0.050630;
	expected[15] = -0.029886;
	expected[16] = -0.006845;
	expected[17] = 0.018515;
	expected[18] = 0.046476;
	expected[19] = 0.077663;
	expected[20] = 0.107645;
	expected[21] = 0.138892;
	expected[22] = 0.171192;
	expected[23] = 0.204335;
	expected[24] = 0.238103;
	expected[25] = 0.272692;
	expected[26] = 0.306915;
	expected[27] = 0.340974;
	expected[28] = 0.374645;
	expected[29] = 0.408069;
	expected[30] = 0.440530;
	expected[31] = 0.471600;
	expected[32] = 0.501304;
	expected[33] = 0.529504;
	expected[34] = 0.556167;
	expected[35] = 0.581195;
	expected[36] = 0.604673;
	expected[37] = 0.627208;
	expected[38] = 0.649278;
	expected[39] = 0.672021;
	expected[40] = 0.683486;
	expected[41] = 0.693448;
	expected[42] = 0.700645;
	expected[43] = 0.702231;
	expected[44] = 0.691617;
	expected[45] = 0.799990;
	expected[46] = 0.815603;
	expected[47] = 0.829221;
	expected[48] = 0.835971;
	expected[49] = 0.834350;
	expected[50] = 0.820863;
	expected[51] = 0.803696;
	expected[52] = 0.772929;
	expected[53] = 0.726663;
	expected[54] = 0.662957;
	expected[55] = 0.579541;
	expected[56] = 0.473388;
	expected[57] = 0.342150;
	expected[58] = 0.184116;
	expected[59] = 218.157846;
	expected[60] = 208.789018;
	expected[61] = 201.554614;
	expected[62] = 196.756658;
	expected[63] = 192.338987;
	expected[64] = 187.954533;
	expected[65] = 183.545674;
	expected[66] = 179.130606;
	expected[67] = 174.975395;
	expected[68] = 170.634734;
	expected[69] = 166.347553;
	expected[70] = 162.097195;
	expected[71] = 157.925496;
	expected[72] = 153.855348;
	expected[73] = 149.908129;
	expected[74] = 146.100627;
	expected[75] = 142.443352;
	expected[76] = 138.934200;
	expected[77] = 135.555969;
	expected[78] = 132.276633;
	expected[79] = 129.099154;
	expected[80] = 125.902377;
	expected[81] = 122.697645;
	expected[82] = 119.494744;
	expected[83] = 116.302382;
	expected[84] = 113.109187;
	expected[85] = 109.964966;
	expected[86] = 106.855053;
	expected[87] = 103.783659;
	expected[88] = 100.744067;
	expected[89] = 97.753921;
	expected[90] = 94.816300;
	expected[91] = 91.920695;
	expected[92] = 89.058252;
	expected[93] = 86.217514;
	expected[94] = 83.392724;
	expected[95] = 80.572117;
	expected[96] = 77.733427;
	expected[97] = 74.856130;
	expected[98] = 71.886111;
	expected[99] = 68.872772;
	expected[100] = 65.496481;
	expected[101] = 61.766844;
	expected[102] = 57.733131;
	expected[103] = 53.619462;
	expected[104] = 48.948289;
	expected[105] = 46.039325;
	expected[106] = 42.953707;
	expected[107] = 39.556464;
	expected[108] = 36.085587;
	expected[109] = 32.534162;
	expected[110] = 28.940502;
	expected[111] = 25.445195;
	expected[112] = 22.028566;
	expected[113] = 18.667367;
	expected[114] = 15.355932;
	expected[115] = 12.099909;
	expected[116] = 8.902120;
	expected[117] = 5.799595;

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
			LOAD_DATA("surplus_1.plt", 0);
			LOAD_DATA("surplus_2.plt", 1);
			LOAD_DATA("surplus_3.plt", 2);
			LOAD_DATA("surplus_4.plt", 3);
			LOAD_DATA("surplus_5.plt", 4);
			LOAD_DATA("surplus_6.plt", 5);
			LOAD_DATA("surplus_7.plt", 6);
			LOAD_DATA("surplus_8.plt", 7);
			LOAD_DATA("surplus_9.plt", 8);
			LOAD_DATA("surplus_10.plt", 9);
			LOAD_DATA("surplus_11.plt", 10);
			LOAD_DATA("surplus_12.plt", 11);
			LOAD_DATA("surplus_13.plt", 12);
			LOAD_DATA("surplus_14.plt", 13);
			LOAD_DATA("surplus_15.plt", 14);
			LOAD_DATA("surplus_16.plt", 15);

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

			volatile double start, finish;
			get_time(&start);
			for (int i = 0; i < ntests; i++)
			{
				INTERPOLATE_ARRAY_MANY_MULTISTATE(
					device, data.dim, data.TotalDof, nstates, &x[0],
					&data.index[0], &data.surplus[0], &results[0]);
			}
			get_time(&finish);
			
			cout << "time = " << (finish - start) / (nstates * ntests) <<
				" sec per state (averaged from " << ntests << " tests)" << endl;

			check_0(results[0], data.TotalDof);
			check_1(results[1], data.TotalDof);
			check_2(results[2], data.TotalDof);
			check_3(results[3], data.TotalDof);
			check_4(results[4], data.TotalDof);
			check_5(results[5], data.TotalDof);
			check_6(results[6], data.TotalDof);
			check_7(results[7], data.TotalDof);
			check_8(results[8], data.TotalDof);
			check_9(results[9], data.TotalDof);
			check_10(results[10], data.TotalDof);
			check_11(results[11], data.TotalDof);
			check_12(results[12], data.TotalDof);
			check_13(results[13], data.TotalDof);
			check_14(results[14], data.TotalDof);
			check_15(results[15], data.TotalDof);
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
			LOAD_DATA("surplus_1.plt", 0);
			LOAD_DATA("surplus_2.plt", 1);
			LOAD_DATA("surplus_3.plt", 2);
			LOAD_DATA("surplus_4.plt", 3);
			LOAD_DATA("surplus_5.plt", 4);
			LOAD_DATA("surplus_6.plt", 5);
			LOAD_DATA("surplus_7.plt", 6);
			LOAD_DATA("surplus_8.plt", 7);
			LOAD_DATA("surplus_9.plt", 8);
			LOAD_DATA("surplus_10.plt", 9);
			LOAD_DATA("surplus_11.plt", 10);
			LOAD_DATA("surplus_12.plt", 11);
			LOAD_DATA("surplus_13.plt", 12);
			LOAD_DATA("surplus_14.plt", 13);
			LOAD_DATA("surplus_15.plt", 14);
			LOAD_DATA("surplus_16.plt", 15);

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

				volatile double start, finish;
				get_time(&start);
				for (int i = 0; i < ntests; i++)
				{
					INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT(
						device, data.dim, data.TotalDof, nstates, &x[0],
						&data.nfreqs[0], &data.xps[0], &data.chains[0], &data.surplus[0], &results[0]);
				}
				get_time(&finish);
			
				cout << "time = " << (finish - start) / (nstates * ntests) <<
					" sec per state (averaged from " << ntests << " tests)" << endl;
			}
			else
			{
				volatile double start, finish;
				get_time(&start);
				for (int i = 0; i < ntests; i++)
				{
					INTERPOLATE_ARRAY_MANY_MULTISTATE(
						device, data.dim, data.TotalDof, nstates, &x[0],
						&data.nfreqs[0], &data.xps[0], &data.chains[0], &data.surplus[0], &results[0]);
				}
				get_time(&finish);
			
				cout << "time = " << (finish - start) / (nstates * ntests) <<
					" sec per state (averaged from " << ntests << " tests)" << endl;
			}

			check_0(results[0], data.TotalDof);
			check_1(results[1], data.TotalDof);
			check_2(results[2], data.TotalDof);
			check_3(results[3], data.TotalDof);
			check_4(results[4], data.TotalDof);
			check_5(results[5], data.TotalDof);
			check_6(results[6], data.TotalDof);
			check_7(results[7], data.TotalDof);
			check_8(results[8], data.TotalDof);
			check_9(results[9], data.TotalDof);
			check_10(results[10], data.TotalDof);
			check_11(results[11], data.TotalDof);
			check_12(results[12], data.TotalDof);
			check_13(results[13], data.TotalDof);
			check_14(results[14], data.TotalDof);
			check_15(results[15], data.TotalDof);
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
			LOAD_DATA("surplus_1.plt", 0);
			LOAD_DATA("surplus_2.plt", 1);
			LOAD_DATA("surplus_3.plt", 2);
			LOAD_DATA("surplus_4.plt", 3);
			LOAD_DATA("surplus_5.plt", 4);
			LOAD_DATA("surplus_6.plt", 5);
			LOAD_DATA("surplus_7.plt", 6);
			LOAD_DATA("surplus_8.plt", 7);
			LOAD_DATA("surplus_9.plt", 8);
			LOAD_DATA("surplus_10.plt", 9);
			LOAD_DATA("surplus_11.plt", 10);
			LOAD_DATA("surplus_12.plt", 11);
			LOAD_DATA("surplus_13.plt", 12);
			LOAD_DATA("surplus_14.plt", 13);
			LOAD_DATA("surplus_15.plt", 14);
			LOAD_DATA("surplus_16.plt", 15);

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

				volatile double start, finish;
				get_time(&start);
				for (int i = 0; i < ntests; i++)
				{
					INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT(
						device, data.dim, data.TotalDof, nstates, &x[0],
						&data.nfreqs[0], &data.xps[0], &data.chains[0], &data.surplus[0], &results[0]);
				}
				get_time(&finish);

				cout << "time = " << (finish - start) / (nstates * ntests) <<
					" sec per state (averaged from " << ntests << " tests)" << endl;
			}
			else
			{
				volatile double start, finish;
				get_time(&start);
				for (int i = 0; i < ntests; i++)
				{
					INTERPOLATE_ARRAY_MANY_MULTISTATE(
						device, data.dim, data.TotalDof, nstates, &x[0],
						&data.nfreqs[0], &data.xps[0], &data.chains[0], &data.surplus[0], &results[0]);
				}
				get_time(&finish);

				cout << "time = " << (finish - start) / (nstates * ntests) <<
					" sec per state (averaged from " << ntests << " tests)" << endl;
			}
			
			check_0(results[0], data.TotalDof);
			check_1(results[1], data.TotalDof);
			check_2(results[2], data.TotalDof);
			check_3(results[3], data.TotalDof);
			check_4(results[4], data.TotalDof);
			check_5(results[5], data.TotalDof);
			check_6(results[6], data.TotalDof);
			check_7(results[7], data.TotalDof);
			check_8(results[8], data.TotalDof);
			check_9(results[9], data.TotalDof);
			check_10(results[10], data.TotalDof);
			check_11(results[11], data.TotalDof);
			check_12(results[12], data.TotalDof);
			check_13(results[13], data.TotalDof);
			check_14(results[14], data.TotalDof);
			check_15(results[15], data.TotalDof);
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
			LOAD_DATA("surplus_1.plt", 0);
			LOAD_DATA("surplus_2.plt", 1);
			LOAD_DATA("surplus_3.plt", 2);
			LOAD_DATA("surplus_4.plt", 3);
			LOAD_DATA("surplus_5.plt", 4);
			LOAD_DATA("surplus_6.plt", 5);
			LOAD_DATA("surplus_7.plt", 6);
			LOAD_DATA("surplus_8.plt", 7);
			LOAD_DATA("surplus_9.plt", 8);
			LOAD_DATA("surplus_10.plt", 9);
			LOAD_DATA("surplus_11.plt", 10);
			LOAD_DATA("surplus_12.plt", 11);
			LOAD_DATA("surplus_13.plt", 12);
			LOAD_DATA("surplus_14.plt", 13);
			LOAD_DATA("surplus_15.plt", 14);
			LOAD_DATA("surplus_16.plt", 15);

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

				volatile double start, finish;
				get_time(&start);
				for (int i = 0; i < ntests; i++)
				{
					INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT(
						device, data.dim, data.TotalDof, nstates, &x[0],
						&data.nfreqs[0], &data.xps[0], &data.chains[0], &data.surplus[0], &results[0]);
				}
				get_time(&finish);

				cout << "time = " << (finish - start) / (nstates * ntests) <<
					" sec per state (averaged from " << ntests << " tests)" << endl;
			}
			else
			{
				volatile double start, finish;
				get_time(&start);
				for (int i = 0; i < ntests; i++)
				{
					INTERPOLATE_ARRAY_MANY_MULTISTATE(
						device, data.dim, data.TotalDof, nstates, &x[0],
						&data.nfreqs[0], &data.xps[0], &data.chains[0], &data.surplus[0], &results[0]);
				}
				get_time(&finish);
			
				cout << "time = " << (finish - start) / (nstates * ntests) <<
					" sec per state (averaged from " << ntests << " tests)" << endl;
			}

			check_0(results[0], data.TotalDof);
			check_1(results[1], data.TotalDof);
			check_2(results[2], data.TotalDof);
			check_3(results[3], data.TotalDof);
			check_4(results[4], data.TotalDof);
			check_5(results[5], data.TotalDof);
			check_6(results[6], data.TotalDof);
			check_7(results[7], data.TotalDof);
			check_8(results[8], data.TotalDof);
			check_9(results[9], data.TotalDof);
			check_10(results[10], data.TotalDof);
			check_11(results[11], data.TotalDof);
			check_12(results[12], data.TotalDof);
			check_13(results[13], data.TotalDof);
			check_14(results[14], data.TotalDof);
			check_15(results[15], data.TotalDof);
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
			LOAD_DATA("surplus_1.plt", 0);
			LOAD_DATA("surplus_2.plt", 1);
			LOAD_DATA("surplus_3.plt", 2);
			LOAD_DATA("surplus_4.plt", 3);
			LOAD_DATA("surplus_5.plt", 4);
			LOAD_DATA("surplus_6.plt", 5);
			LOAD_DATA("surplus_7.plt", 6);
			LOAD_DATA("surplus_8.plt", 7);
			LOAD_DATA("surplus_9.plt", 8);
			LOAD_DATA("surplus_10.plt", 9);
			LOAD_DATA("surplus_11.plt", 10);
			LOAD_DATA("surplus_12.plt", 11);
			LOAD_DATA("surplus_13.plt", 12);
			LOAD_DATA("surplus_14.plt", 13);
			LOAD_DATA("surplus_15.plt", 14);
			LOAD_DATA("surplus_16.plt", 15);

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

				volatile double start, finish;
				get_time(&start);
				for (int i = 0; i < ntests; i++)
				{
					INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT(
						device, data.dim, data.TotalDof, nstates, &x[0],
						&data.nfreqs[0], &data.xps[0], &data.chains[0], &data.surplus[0], &results[0]);
					get_time(&finish);
				}

				cout << "time = " << (finish - start) / (nstates * ntests) <<
					" sec per state (averaged from " << ntests << " tests)" << endl;
			}
			else
			{
				volatile double start, finish;
				get_time(&start);
				for (int i = 0; i < ntests; i++)
				{
					INTERPOLATE_ARRAY_MANY_MULTISTATE(
						device, data.dim, data.TotalDof, nstates, &x[0],
						&data.nfreqs[0], &data.xps[0], &data.chains[0], &data.surplus[0], &results[0]);
				}
				get_time(&finish);

				cout << "time = " << (finish - start) / (nstates * ntests) <<
					" sec per state (averaged from " << ntests << " tests)" << endl;
			}
			
			check_0(results[0], data.TotalDof);
			check_1(results[1], data.TotalDof);
			check_2(results[2], data.TotalDof);
			check_3(results[3], data.TotalDof);
			check_4(results[4], data.TotalDof);
			check_5(results[5], data.TotalDof);
			check_6(results[6], data.TotalDof);
			check_7(results[7], data.TotalDof);
			check_8(results[8], data.TotalDof);
			check_9(results[9], data.TotalDof);
			check_10(results[10], data.TotalDof);
			check_11(results[11], data.TotalDof);
			check_12(results[12], data.TotalDof);
			check_13(results[13], data.TotalDof);
			check_14(results[14], data.TotalDof);
			check_15(results[15], data.TotalDof);
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
				LOAD_DATA("surplus_1.plt", 0);
				LOAD_DATA("surplus_2.plt", 1);
				LOAD_DATA("surplus_3.plt", 2);
				LOAD_DATA("surplus_4.plt", 3);
				LOAD_DATA("surplus_5.plt", 4);
				LOAD_DATA("surplus_6.plt", 5);
				LOAD_DATA("surplus_7.plt", 6);
				LOAD_DATA("surplus_8.plt", 7);
				LOAD_DATA("surplus_9.plt", 8);
				LOAD_DATA("surplus_10.plt", 9);
				LOAD_DATA("surplus_11.plt", 10);
				LOAD_DATA("surplus_12.plt", 11);
				LOAD_DATA("surplus_13.plt", 12);
				LOAD_DATA("surplus_14.plt", 13);
				LOAD_DATA("surplus_15.plt", 14);
				LOAD_DATA("surplus_16.plt", 15);

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

					volatile double start, finish;
					get_time(&start);
					for (int i = 0; i < ntests; i++)
					{
						INTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT(
							device, data.dim, &nnos[0], data.TotalDof, nstates, &x[0],
							data.device.getNfreqs(0), data.device.getXPS(0), data.host.getSzXPS(0),
							data.device.getChains(0), data.device.getSurplus(0), &results[0]);
					}
					get_time(&finish);

					cout << "time = " << (finish - start) / (nstates * ntests) <<
						" sec per state (averaged from " << ntests << " tests)" << endl;
				}
				else
				{
					// Run once without timing to do all CUDA-specific internal initializations.
					INTERPOLATE_ARRAY_MANY_MULTISTATE(
						device, data.dim, &nnos[0], data.TotalDof, nstates, &x[0],
						data.device.getNfreqs(0), data.device.getXPS(0), data.host.getSzXPS(0),
						data.device.getChains(0), data.device.getSurplus(0), &results[0]);

					volatile double start, finish;
					get_time(&start);
					for (int i = 0; i < ntests; i++)
					{
						INTERPOLATE_ARRAY_MANY_MULTISTATE(
							device, data.dim, &nnos[0], data.TotalDof, nstates, &x[0],
							data.device.getNfreqs(0), data.device.getXPS(0), data.host.getSzXPS(0),
							data.device.getChains(0), data.device.getSurplus(0), &results[0]);
					}
					get_time(&finish);

					cout << "time = " << (finish - start) / (nstates * ntests) <<
						" sec per state (averaged from " << ntests << " tests)" << endl;
				}
			}
			releaseDevice(device);

			check_0(results[0], vresults[0].length());
			check_1(results[1], vresults[1].length());
			check_2(results[2], vresults[2].length());
			check_3(results[3], vresults[3].length());
			check_4(results[4], vresults[4].length());
			check_5(results[5], vresults[5].length());
			check_6(results[6], vresults[6].length());
			check_7(results[7], vresults[7].length());
			check_8(results[8], vresults[8].length());
			check_9(results[9], vresults[9].length());
			check_10(results[10], vresults[10].length());
			check_11(results[11], vresults[11].length());
			check_12(results[12], vresults[12].length());
			check_13(results[13], vresults[13].length());
			check_14(results[14], vresults[14].length());
			check_15(results[15], vresults[15].length());

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
	const char* crunopt = getenv("RUNTIME_OPTIMIZATION");
	if (crunopt)
		runopt = !!atoi(crunopt);

	const char* cntests = getenv("NTESTS");
	if (cntests)
		ntests = atoi(cntests);

	GoogleTest(argc, argv);

	::testing::InitGoogleTest(&argc, argv);

	return RUN_ALL_TESTS();
}

