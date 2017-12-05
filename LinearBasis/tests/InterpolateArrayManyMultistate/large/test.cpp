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
	expected[0] = -0.024336;
	expected[1] = -0.054007;
	expected[2] = -0.071207;
	expected[3] = -0.090020;
	expected[4] = -0.105265;
	expected[5] = -0.114130;
	expected[6] = -0.119322;
	expected[7] = -0.121046;
	expected[8] = -0.119471;
	expected[9] = -0.114754;
	expected[10] = -0.107002;
	expected[11] = -0.094333;
	expected[12] = -0.082233;
	expected[13] = -0.064660;
	expected[14] = -0.047240;
	expected[15] = -0.025539;
	expected[16] = -0.001364;
	expected[17] = 0.025393;
	expected[18] = 0.055211;
	expected[19] = 0.087864;
	expected[20] = 0.119090;
	expected[21] = 0.151730;
	expected[22] = 0.185555;
	expected[23] = 0.220341;
	expected[24] = 0.255850;
	expected[25] = 0.292309;
	expected[26] = 0.328513;
	expected[27] = 0.365074;
	expected[28] = 0.401087;
	expected[29] = 0.437596;
	expected[30] = 0.470950;
	expected[31] = 0.503270;
	expected[32] = 0.535476;
	expected[33] = 0.570273;
	expected[34] = 0.592709;
	expected[35] = 0.619647;
	expected[36] = 0.643284;
	expected[37] = 0.667621;
	expected[38] = 0.691448;
	expected[39] = 0.715402;
	expected[40] = 0.726424;
	expected[41] = 0.734378;
	expected[42] = 0.742312;
	expected[43] = 0.743235;
	expected[44] = 0.736867;
	expected[45] = 0.825383;
	expected[46] = 0.842611;
	expected[47] = 0.855929;
	expected[48] = 0.857974;
	expected[49] = 0.860716;
	expected[50] = 0.846466;
	expected[51] = 0.829105;
	expected[52] = 0.797938;
	expected[53] = 0.751129;
	expected[54] = 0.684331;
	expected[55] = 0.598432;
	expected[56] = 0.488126;
	expected[57] = 0.352762;
	expected[58] = 0.190131;
	expected[59] = 217.485350;
	expected[60] = 208.118911;
	expected[61] = 200.548848;
	expected[62] = 195.969121;
	expected[63] = 191.743411;
	expected[64] = 187.361604;
	expected[65] = 182.944717;
	expected[66] = 178.512605;
	expected[67] = 174.304189;
	expected[68] = 169.934834;
	expected[69] = 165.596651;
	expected[70] = 161.189894;
	expected[71] = 157.079704;
	expected[72] = 152.935928;
	expected[73] = 149.033181;
	expected[74] = 145.201090;
	expected[75] = 141.517841;
	expected[76] = 137.974097;
	expected[77] = 134.539630;
	expected[78] = 131.229098;
	expected[79] = 128.011002;
	expected[80] = 124.773909;
	expected[81] = 121.530265;
	expected[82] = 118.290753;
	expected[83] = 115.064835;
	expected[84] = 111.838120;
	expected[85] = 108.663359;
	expected[86] = 105.508107;
	expected[87] = 102.401460;
	expected[88] = 99.389994;
	expected[89] = 96.363831;
	expected[90] = 93.362983;
	expected[91] = 90.434266;
	expected[92] = 87.682147;
	expected[93] = 84.667728;
	expected[94] = 81.809101;
	expected[95] = 78.977284;
	expected[96] = 76.252087;
	expected[97] = 73.165675;
	expected[98] = 70.175219;
	expected[99] = 67.189514;
	expected[100] = 63.864509;
	expected[101] = 60.247028;
	expected[102] = 56.449306;
	expected[103] = 52.606640;
	expected[104] = 48.427472;
	expected[105] = 45.521302;
	expected[106] = 42.345466;
	expected[107] = 38.676627;
	expected[108] = 35.600124;
	expected[109] = 32.111247;
	expected[110] = 28.554832;
	expected[111] = 25.113958;
	expected[112] = 21.941072;
	expected[113] = 18.014876;
	expected[114] = 15.126636;
	expected[115] = 11.693684;
	expected[116] = 8.832424;
	expected[117] = 5.806389;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_1(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.029201;
	expected[1] = -0.053242;
	expected[2] = -0.077685;
	expected[3] = -0.094788;
	expected[4] = -0.107714;
	expected[5] = -0.116741;
	expected[6] = -0.122098;
	expected[7] = -0.122642;
	expected[8] = -0.122033;
	expected[9] = -0.117938;
	expected[10] = -0.108853;
	expected[11] = -0.099620;
	expected[12] = -0.086240;
	expected[13] = -0.070244;
	expected[14] = -0.051718;
	expected[15] = -0.030334;
	expected[16] = -0.006390;
	expected[17] = 0.020083;
	expected[18] = 0.049482;
	expected[19] = 0.082554;
	expected[20] = 0.115926;
	expected[21] = 0.148786;
	expected[22] = 0.179720;
	expected[23] = 0.214415;
	expected[24] = 0.249838;
	expected[25] = 0.285721;
	expected[26] = 0.321839;
	expected[27] = 0.358329;
	expected[28] = 0.394023;
	expected[29] = 0.429137;
	expected[30] = 0.463743;
	expected[31] = 0.496661;
	expected[32] = 0.529695;
	expected[33] = 0.556646;
	expected[34] = 0.586615;
	expected[35] = 0.613978;
	expected[36] = 0.637582;
	expected[37] = 0.665977;
	expected[38] = 0.683534;
	expected[39] = 0.707120;
	expected[40] = 0.719015;
	expected[41] = 0.728040;
	expected[42] = 0.734628;
	expected[43] = 0.737493;
	expected[44] = 0.728627;
	expected[45] = 0.842326;
	expected[46] = 0.858765;
	expected[47] = 0.871902;
	expected[48] = 0.879857;
	expected[49] = 0.879021;
	expected[50] = 0.864172;
	expected[51] = 0.846922;
	expected[52] = 0.815564;
	expected[53] = 0.767945;
	expected[54] = 0.702175;
	expected[55] = 0.614182;
	expected[56] = 0.502615;
	expected[57] = 0.364475;
	expected[58] = 0.197607;
	expected[59] = 218.246039;
	expected[60] = 208.429071;
	expected[61] = 201.494000;
	expected[62] = 196.755163;
	expected[63] = 192.379085;
	expected[64] = 188.018189;
	expected[65] = 183.615274;
	expected[66] = 179.081741;
	expected[67] = 174.937099;
	expected[68] = 170.597544;
	expected[69] = 166.165113;
	expected[70] = 161.937866;
	expected[71] = 157.718557;
	expected[72] = 153.609061;
	expected[73] = 149.629696;
	expected[74] = 145.779913;
	expected[75] = 142.076572;
	expected[76] = 138.518742;
	expected[77] = 135.080878;
	expected[78] = 131.722990;
	expected[79] = 128.383674;
	expected[80] = 125.151573;
	expected[81] = 122.002322;
	expected[82] = 118.748305;
	expected[83] = 115.507808;
	expected[84] = 112.290292;
	expected[85] = 109.101992;
	expected[86] = 105.928981;
	expected[87] = 102.818620;
	expected[88] = 99.751858;
	expected[89] = 96.715876;
	expected[90] = 93.741691;
	expected[91] = 90.931930;
	expected[92] = 87.689118;
	expected[93] = 85.017297;
	expected[94] = 82.165265;
	expected[95] = 79.420283;
	expected[96] = 76.278544;
	expected[97] = 73.614520;
	expected[98] = 70.540283;
	expected[99] = 67.526920;
	expected[100] = 64.125773;
	expected[101] = 60.375256;
	expected[102] = 56.514362;
	expected[103] = 52.478916;
	expected[104] = 47.770618;
	expected[105] = 44.863955;
	expected[106] = 41.743660;
	expected[107] = 38.440510;
	expected[108] = 35.005227;
	expected[109] = 31.542229;
	expected[110] = 28.015000;
	expected[111] = 24.577226;
	expected[112] = 21.216307;
	expected[113] = 17.903733;
	expected[114] = 14.778889;
	expected[115] = 11.547694;
	expected[116] = 8.623105;
	expected[117] = 5.507108;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_2(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.021189;
	expected[1] = -0.049743;
	expected[2] = -0.070521;
	expected[3] = -0.086903;
	expected[4] = -0.099205;
	expected[5] = -0.107596;
	expected[6] = -0.112142;
	expected[7] = -0.113917;
	expected[8] = -0.108769;
	expected[9] = -0.105600;
	expected[10] = -0.100181;
	expected[11] = -0.086686;
	expected[12] = -0.073113;
	expected[13] = -0.057474;
	expected[14] = -0.040090;
	expected[15] = -0.019102;
	expected[16] = 0.011258;
	expected[17] = 0.037320;
	expected[18] = 0.061994;
	expected[19] = 0.089465;
	expected[20] = 0.125198;
	expected[21] = 0.154333;
	expected[22] = 0.183929;
	expected[23] = 0.217146;
	expected[24] = 0.251220;
	expected[25] = 0.285890;
	expected[26] = 0.320820;
	expected[27] = 0.355361;
	expected[28] = 0.389526;
	expected[29] = 0.423224;
	expected[30] = 0.456165;
	expected[31] = 0.487540;
	expected[32] = 0.517554;
	expected[33] = 0.546049;
	expected[34] = 0.572974;
	expected[35] = 0.598255;
	expected[36] = 0.625911;
	expected[37] = 0.642584;
	expected[38] = 0.664771;
	expected[39] = 0.687954;
	expected[40] = 0.697981;
	expected[41] = 0.706784;
	expected[42] = 0.714247;
	expected[43] = 0.715070;
	expected[44] = 0.706946;
	expected[45] = 0.797585;
	expected[46] = 0.811838;
	expected[47] = 0.823373;
	expected[48] = 0.829834;
	expected[49] = 0.827223;
	expected[50] = 0.812661;
	expected[51] = 0.795385;
	expected[52] = 0.764537;
	expected[53] = 0.718261;
	expected[54] = 0.655108;
	expected[55] = 0.572110;
	expected[56] = 0.467282;
	expected[57] = 0.337909;
	expected[58] = 0.182394;
	expected[59] = 217.306694;
	expected[60] = 207.767103;
	expected[61] = 200.481915;
	expected[62] = 195.641606;
	expected[63] = 191.190892;
	expected[64] = 186.785665;
	expected[65] = 182.371020;
	expected[66] = 177.960233;
	expected[67] = 173.531546;
	expected[68] = 169.304282;
	expected[69] = 165.148525;
	expected[70] = 160.795633;
	expected[71] = 156.654956;
	expected[72] = 152.645902;
	expected[73] = 148.790144;
	expected[74] = 145.027659;
	expected[75] = 141.139699;
	expected[76] = 137.634339;
	expected[77] = 134.442485;
	expected[78] = 131.361360;
	expected[79] = 127.993418;
	expected[80] = 124.900896;
	expected[81] = 121.832877;
	expected[82] = 118.662116;
	expected[83] = 115.494841;
	expected[84] = 112.338177;
	expected[85] = 109.203468;
	expected[86] = 106.113784;
	expected[87] = 103.061669;
	expected[88] = 100.046173;
	expected[89] = 97.069115;
	expected[90] = 94.146702;
	expected[91] = 91.262819;
	expected[92] = 88.409743;
	expected[93] = 85.572543;
	expected[94] = 82.751005;
	expected[95] = 80.139071;
	expected[96] = 76.294920;
	expected[97] = 74.260724;
	expected[98] = 71.365803;
	expected[99] = 68.318456;
	expected[100] = 65.005619;
	expected[101] = 61.344804;
	expected[102] = 57.464694;
	expected[103] = 53.576273;
	expected[104] = 49.321158;
	expected[105] = 46.606317;
	expected[106] = 43.349456;
	expected[107] = 40.018281;
	expected[108] = 36.548866;
	expected[109] = 32.991114;
	expected[110] = 29.375713;
	expected[111] = 25.864562;
	expected[112] = 22.437351;
	expected[113] = 19.060961;
	expected[114] = 15.736687;
	expected[115] = 12.444097;
	expected[116] = 9.193882;
	expected[117] = 5.999089;

	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_3(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.025772;
	expected[1] = -0.052260;
	expected[2] = -0.073048;
	expected[3] = -0.089614;
	expected[4] = -0.100531;
	expected[5] = -0.107959;
	expected[6] = -0.114464;
	expected[7] = -0.117625;
	expected[8] = -0.116123;
	expected[9] = -0.111491;
	expected[10] = -0.103917;
	expected[11] = -0.093487;
	expected[12] = -0.080290;
	expected[13] = -0.062290;
	expected[14] = -0.044532;
	expected[15] = -0.024827;
	expected[16] = -0.001517;
	expected[17] = 0.024222;
	expected[18] = 0.052746;
	expected[19] = 0.087539;
	expected[20] = 0.114322;
	expected[21] = 0.145687;
	expected[22] = 0.178144;
	expected[23] = 0.211482;
	expected[24] = 0.245481;
	expected[25] = 0.279891;
	expected[26] = 0.314479;
	expected[27] = 0.348949;
	expected[28] = 0.383249;
	expected[29] = 0.416870;
	expected[30] = 0.449343;
	expected[31] = 0.480639;
	expected[32] = 0.510563;
	expected[33] = 0.539281;
	expected[34] = 0.566304;
	expected[35] = 0.591471;
	expected[36] = 0.615018;
	expected[37] = 0.637756;
	expected[38] = 0.659763;
	expected[39] = 0.680046;
	expected[40] = 0.695526;
	expected[41] = 0.701833;
	expected[42] = 0.708789;
	expected[43] = 0.709652;
	expected[44] = 0.698964;
	expected[45] = 0.811059;
	expected[46] = 0.829921;
	expected[47] = 0.840052;
	expected[48] = 0.846702;
	expected[49] = 0.844494;
	expected[50] = 0.830304;
	expected[51] = 0.812959;
	expected[52] = 0.782164;
	expected[53] = 0.735875;
	expected[54] = 0.672178;
	expected[55] = 0.588728;
	expected[56] = 0.481957;
	expected[57] = 0.349797;
	expected[58] = 0.189952;
	expected[59] = 218.027187;
	expected[60] = 208.354902;
	expected[61] = 201.090235;
	expected[62] = 196.280841;
	expected[63] = 191.735468;
	expected[64] = 187.287777;
	expected[65] = 182.964222;
	expected[66] = 178.633092;
	expected[67] = 174.453957;
	expected[68] = 170.118244;
	expected[69] = 165.820056;
	expected[70] = 161.581377;
	expected[71] = 157.424400;
	expected[72] = 153.266036;
	expected[73] = 149.388452;
	expected[74] = 145.643048;
	expected[75] = 142.002960;
	expected[76] = 138.505910;
	expected[77] = 135.130564;
	expected[78] = 131.751532;
	expected[79] = 128.703552;
	expected[80] = 125.517025;
	expected[81] = 122.322355;
	expected[82] = 119.129346;
	expected[83] = 115.946779;
	expected[84] = 112.783293;
	expected[85] = 109.645712;
	expected[86] = 106.541753;
	expected[87] = 103.466821;
	expected[88] = 100.437076;
	expected[89] = 97.463264;
	expected[90] = 94.532762;
	expected[91] = 91.642483;
	expected[92] = 88.771995;
	expected[93] = 85.927040;
	expected[94] = 83.104235;
	expected[95] = 80.285794;
	expected[96] = 77.443198;
	expected[97] = 75.008346;
	expected[98] = 71.641892;
	expected[99] = 68.597680;
	expected[100] = 65.248157;
	expected[101] = 61.524562;
	expected[102] = 57.516824;
	expected[103] = 53.428680;
	expected[104] = 48.717795;
	expected[105] = 45.749797;
	expected[106] = 42.692890;
	expected[107] = 39.346748;
	expected[108] = 35.906401;
	expected[109] = 32.382177;
	expected[110] = 28.804075;
	expected[111] = 25.316440;
	expected[112] = 21.901450;
	expected[113] = 18.535249;
	expected[114] = 15.337144;
	expected[115] = 12.115889;
	expected[116] = 8.757290;
	expected[117] = 5.682433;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_4(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.029760;
	expected[1] = -0.056675;
	expected[2] = -0.078398;
	expected[3] = -0.095600;
	expected[4] = -0.108617;
	expected[5] = -0.116020;
	expected[6] = -0.122982;
	expected[7] = -0.125128;
	expected[8] = -0.122117;
	expected[9] = -0.119031;
	expected[10] = -0.111350;
	expected[11] = -0.100778;
	expected[12] = -0.087467;
	expected[13] = -0.071546;
	expected[14] = -0.053130;
	expected[15] = -0.032186;
	expected[16] = -0.008861;
	expected[17] = 0.017467;
	expected[18] = 0.046832;
	expected[19] = 0.079900;
	expected[20] = 0.111234;
	expected[21] = 0.144802;
	expected[22] = 0.177842;
	expected[23] = 0.215192;
	expected[24] = 0.247416;
	expected[25] = 0.283314;
	expected[26] = 0.319444;
	expected[27] = 0.355515;
	expected[28] = 0.391668;
	expected[29] = 0.426736;
	expected[30] = 0.460968;
	expected[31] = 0.494461;
	expected[32] = 0.526703;
	expected[33] = 0.555210;
	expected[34] = 0.584537;
	expected[35] = 0.612486;
	expected[36] = 0.639477;
	expected[37] = 0.660571;
	expected[38] = 0.681445;
	expected[39] = 0.705043;
	expected[40] = 0.716860;
	expected[41] = 0.726001;
	expected[42] = 0.734409;
	expected[43] = 0.737092;
	expected[44] = 0.731702;
	expected[45] = 0.818285;
	expected[46] = 0.833320;
	expected[47] = 0.845920;
	expected[48] = 0.854767;
	expected[49] = 0.852184;
	expected[50] = 0.839678;
	expected[51] = 0.822464;
	expected[52] = 0.791159;
	expected[53] = 0.743717;
	expected[54] = 0.678026;
	expected[55] = 0.591848;
	expected[56] = 0.482200;
	expected[57] = 0.347039;
	expected[58] = 0.185902;
	expected[59] = 218.123842;
	expected[60] = 208.534136;
	expected[61] = 201.382370;
	expected[62] = 196.648827;
	expected[63] = 192.276428;
	expected[64] = 187.769642;
	expected[65] = 183.496303;
	expected[66] = 179.086766;
	expected[67] = 174.774423;
	expected[68] = 170.483925;
	expected[69] = 166.126970;
	expected[70] = 161.824791;
	expected[71] = 157.604681;
	expected[72] = 153.493836;
	expected[73] = 149.513941;
	expected[74] = 145.677433;
	expected[75] = 141.995135;
	expected[76] = 138.437999;
	expected[77] = 134.995572;
	expected[78] = 131.633985;
	expected[79] = 128.390420;
	expected[80] = 125.088320;
	expected[81] = 121.868985;
	expected[82] = 118.515383;
	expected[83] = 115.392821;
	expected[84] = 112.171890;
	expected[85] = 108.980229;
	expected[86] = 105.824419;
	expected[87] = 102.689158;
	expected[88] = 99.621931;
	expected[89] = 96.600623;
	expected[90] = 93.601873;
	expected[91] = 90.726719;
	expected[92] = 87.616381;
	expected[93] = 84.862730;
	expected[94] = 82.356495;
	expected[95] = 79.123256;
	expected[96] = 76.228901;
	expected[97] = 73.418840;
	expected[98] = 70.411055;
	expected[99] = 67.387842;
	expected[100] = 64.031069;
	expected[101] = 60.226708;
	expected[102] = 56.543066;
	expected[103] = 52.679452;
	expected[104] = 48.509247;
	expected[105] = 45.628309;
	expected[106] = 42.373968;
	expected[107] = 39.137481;
	expected[108] = 35.614961;
	expected[109] = 32.184715;
	expected[110] = 28.621551;
	expected[111] = 25.159848;
	expected[112] = 21.784066;
	expected[113] = 18.475076;
	expected[114] = 15.502023;
	expected[115] = 12.037885;
	expected[116] = 8.651311;
	expected[117] = 5.896831;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_5(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.027398;
	expected[1] = -0.054156;
	expected[2] = -0.080237;
	expected[3] = -0.097677;
	expected[4] = -0.109439;
	expected[5] = -0.119315;
	expected[6] = -0.125475;
	expected[7] = -0.127396;
	expected[8] = -0.125967;
	expected[9] = -0.121371;
	expected[10] = -0.113758;
	expected[11] = -0.103265;
	expected[12] = -0.089971;
	expected[13] = -0.073993;
	expected[14] = -0.055459;
	expected[15] = -0.034521;
	expected[16] = -0.011469;
	expected[17] = 0.013675;
	expected[18] = 0.041980;
	expected[19] = 0.074414;
	expected[20] = 0.105685;
	expected[21] = 0.138345;
	expected[22] = 0.172175;
	expected[23] = 0.207099;
	expected[24] = 0.243897;
	expected[25] = 0.278444;
	expected[26] = 0.316379;
	expected[27] = 0.349687;
	expected[28] = 0.385318;
	expected[29] = 0.420399;
	expected[30] = 0.454752;
	expected[31] = 0.487617;
	expected[32] = 0.519122;
	expected[33] = 0.549630;
	expected[34] = 0.578771;
	expected[35] = 0.603916;
	expected[36] = 0.629607;
	expected[37] = 0.655260;
	expected[38] = 0.676184;
	expected[39] = 0.697654;
	expected[40] = 0.710024;
	expected[41] = 0.721221;
	expected[42] = 0.729184;
	expected[43] = 0.731472;
	expected[44] = 0.723767;
	expected[45] = 0.830115;
	expected[46] = 0.850611;
	expected[47] = 0.862124;
	expected[48] = 0.870586;
	expected[49] = 0.869055;
	expected[50] = 0.855785;
	expected[51] = 0.838662;
	expected[52] = 0.808126;
	expected[53] = 0.759966;
	expected[54] = 0.693616;
	expected[55] = 0.606158;
	expected[56] = 0.495095;
	expected[57] = 0.357730;
	expected[58] = 0.192443;
	expected[59] = 218.276520;
	expected[60] = 208.676723;
	expected[61] = 201.934325;
	expected[62] = 197.243654;
	expected[63] = 192.795585;
	expected[64] = 188.505087;
	expected[65] = 184.159459;
	expected[66] = 179.732530;
	expected[67] = 175.514105;
	expected[68] = 171.122765;
	expected[69] = 166.753855;
	expected[70] = 162.435212;
	expected[71] = 158.190463;
	expected[72] = 154.048093;
	expected[73] = 150.033013;
	expected[74] = 146.166719;
	expected[75] = 142.465394;
	expected[76] = 138.929423;
	expected[77] = 135.508497;
	expected[78] = 132.153361;
	expected[79] = 128.895382;
	expected[80] = 125.618871;
	expected[81] = 122.336370;
	expected[82] = 119.051969;
	expected[83] = 115.729147;
	expected[84] = 112.562149;
	expected[85] = 109.290583;
	expected[86] = 106.223012;
	expected[87] = 103.098563;
	expected[88] = 100.017170;
	expected[89] = 96.974883;
	expected[90] = 93.991988;
	expected[91] = 91.051608;
	expected[92] = 88.125119;
	expected[93] = 85.437860;
	expected[94] = 82.080588;
	expected[95] = 79.545694;
	expected[96] = 76.127720;
	expected[97] = 73.683760;
	expected[98] = 70.743393;
	expected[99] = 67.675928;
	expected[100] = 64.266925;
	expected[101] = 60.569195;
	expected[102] = 56.610570;
	expected[103] = 52.550868;
	expected[104] = 47.984834;
	expected[105] = 45.003341;
	expected[106] = 41.900377;
	expected[107] = 38.554924;
	expected[108] = 35.135531;
	expected[109] = 31.639991;
	expected[110] = 28.114748;
	expected[111] = 24.766883;
	expected[112] = 21.384394;
	expected[113] = 18.103965;
	expected[114] = 14.787559;
	expected[115] = 11.626707;
	expected[116] = 8.577548;
	expected[117] = 5.605059;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_6(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.024743;
	expected[1] = -0.053522;
	expected[2] = -0.069880;
	expected[3] = -0.087669;
	expected[4] = -0.102744;
	expected[5] = -0.112202;
	expected[6] = -0.117323;
	expected[7] = -0.119048;
	expected[8] = -0.117581;
	expected[9] = -0.113081;
	expected[10] = -0.105647;
	expected[11] = -0.093497;
	expected[12] = -0.082232;
	expected[13] = -0.066489;
	expected[14] = -0.048312;
	expected[15] = -0.027451;
	expected[16] = -0.004228;
	expected[17] = 0.021435;
	expected[18] = 0.049937;
	expected[19] = 0.082171;
	expected[20] = 0.114220;
	expected[21] = 0.143867;
	expected[22] = 0.178497;
	expected[23] = 0.211794;
	expected[24] = 0.242833;
	expected[25] = 0.277248;
	expected[26] = 0.311845;
	expected[27] = 0.346328;
	expected[28] = 0.380440;
	expected[29] = 0.413983;
	expected[30] = 0.446814;
	expected[31] = 0.478147;
	expected[32] = 0.508115;
	expected[33] = 0.536557;
	expected[34] = 0.563857;
	expected[35] = 0.589169;
	expected[36] = 0.612789;
	expected[37] = 0.635331;
	expected[38] = 0.657352;
	expected[39] = 0.677698;
	expected[40] = 0.691089;
	expected[41] = 0.699621;
	expected[42] = 0.706980;
	expected[43] = 0.708923;
	expected[44] = 0.701866;
	expected[45] = 0.785587;
	expected[46] = 0.801592;
	expected[47] = 0.814627;
	expected[48] = 0.821597;
	expected[49] = 0.819856;
	expected[50] = 0.805992;
	expected[51] = 0.788800;
	expected[52] = 0.758292;
	expected[53] = 0.711896;
	expected[54] = 0.648283;
	expected[55] = 0.565668;
	expected[56] = 0.461120;
	expected[57] = 0.332322;
	expected[58] = 0.178057;
	expected[59] = 217.779313;
	expected[60] = 208.292384;
	expected[61] = 200.670719;
	expected[62] = 195.991859;
	expected[63] = 191.732953;
	expected[64] = 187.388555;
	expected[65] = 182.975688;
	expected[66] = 178.559009;
	expected[67] = 174.377318;
	expected[68] = 170.043513;
	expected[69] = 165.746738;
	expected[70] = 161.402756;
	expected[71] = 157.344843;
	expected[72] = 153.290693;
	expected[73] = 149.368514;
	expected[74] = 145.577897;
	expected[75] = 141.935327;
	expected[76] = 138.434864;
	expected[77] = 135.053767;
	expected[78] = 131.750316;
	expected[79] = 128.489811;
	expected[80] = 125.388601;
	expected[81] = 122.116724;
	expected[82] = 118.923269;
	expected[83] = 115.840057;
	expected[84] = 112.673176;
	expected[85] = 109.532212;
	expected[86] = 106.424811;
	expected[87] = 103.355934;
	expected[88] = 100.327928;
	expected[89] = 97.336573;
	expected[90] = 94.402577;
	expected[91] = 91.509493;
	expected[92] = 88.649738;
	expected[93] = 85.790618;
	expected[94] = 82.962985;
	expected[95] = 80.145325;
	expected[96] = 77.311826;
	expected[97] = 74.775013;
	expected[98] = 71.519816;
	expected[99] = 68.497203;
	expected[100] = 65.155470;
	expected[101] = 61.474955;
	expected[102] = 57.561318;
	expected[103] = 53.647402;
	expected[104] = 49.537020;
	expected[105] = 46.631625;
	expected[106] = 43.481631;
	expected[107] = 40.133028;
	expected[108] = 36.641106;
	expected[109] = 33.065450;
	expected[110] = 29.444916;
	expected[111] = 25.924879;
	expected[112] = 22.503779;
	expected[113] = 19.145913;
	expected[114] = 15.826087;
	expected[115] = 12.545825;
	expected[116] = 9.308709;
	expected[117] = 6.098513;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_7(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.029435;
	expected[1] = -0.054213;
	expected[2] = -0.076627;
	expected[3] = -0.093270;
	expected[4] = -0.105848;
	expected[5] = -0.114637;
	expected[6] = -0.119868;
	expected[7] = -0.121743;
	expected[8] = -0.120427;
	expected[9] = -0.115659;
	expected[10] = -0.108498;
	expected[11] = -0.098425;
	expected[12] = -0.085438;
	expected[13] = -0.069892;
	expected[14] = -0.051913;
	expected[15] = -0.031480;
	expected[16] = -0.008707;
	expected[17] = 0.016667;
	expected[18] = 0.044834;
	expected[19] = 0.076560;
	expected[20] = 0.106673;
	expected[21] = 0.138090;
	expected[22] = 0.170593;
	expected[23] = 0.204356;
	expected[24] = 0.239745;
	expected[25] = 0.272154;
	expected[26] = 0.306887;
	expected[27] = 0.343056;
	expected[28] = 0.377392;
	expected[29] = 0.408153;
	expected[30] = 0.440583;
	expected[31] = 0.471831;
	expected[32] = 0.501929;
	expected[33] = 0.530399;
	expected[34] = 0.557158;
	expected[35] = 0.582287;
	expected[36] = 0.606244;
	expected[37] = 0.628687;
	expected[38] = 0.650625;
	expected[39] = 0.672118;
	expected[40] = 0.684080;
	expected[41] = 0.694659;
	expected[42] = 0.701987;
	expected[43] = 0.704873;
	expected[44] = 0.694115;
	expected[45] = 0.801891;
	expected[46] = 0.817930;
	expected[47] = 0.828770;
	expected[48] = 0.837013;
	expected[49] = 0.835433;
	expected[50] = 0.821821;
	expected[51] = 0.804646;
	expected[52] = 0.773871;
	expected[53] = 0.727599;
	expected[54] = 0.663893;
	expected[55] = 0.580620;
	expected[56] = 0.474319;
	expected[57] = 0.343167;
	expected[58] = 0.184673;
	expected[59] = 218.505915;
	expected[60] = 208.732237;
	expected[61] = 201.605402;
	expected[62] = 196.815337;
	expected[63] = 192.405404;
	expected[64] = 188.028204;
	expected[65] = 183.625729;
	expected[66] = 179.216060;
	expected[67] = 175.035878;
	expected[68] = 170.668328;
	expected[69] = 166.375088;
	expected[70] = 162.125132;
	expected[71] = 157.949909;
	expected[72] = 153.881618;
	expected[73] = 149.940621;
	expected[74] = 146.139508;
	expected[75] = 142.488318;
	expected[76] = 138.975902;
	expected[77] = 135.588482;
	expected[78] = 132.289625;
	expected[79] = 129.100580;
	expected[80] = 125.891737;
	expected[81] = 122.674826;
	expected[82] = 119.443024;
	expected[83] = 116.180142;
	expected[84] = 113.087680;
	expected[85] = 109.926073;
	expected[86] = 106.746881;
	expected[87] = 103.655360;
	expected[88] = 100.711653;
	expected[89] = 97.725074;
	expected[90] = 94.783764;
	expected[91] = 91.873565;
	expected[92] = 89.002102;
	expected[93] = 86.160557;
	expected[94] = 83.333651;
	expected[95] = 80.496080;
	expected[96] = 77.660922;
	expected[97] = 74.786417;
	expected[98] = 71.851817;
	expected[99] = 68.803849;
	expected[100] = 65.400076;
	expected[101] = 61.654155;
	expected[102] = 57.598240;
	expected[103] = 53.501594;
	expected[104] = 49.242993;
	expected[105] = 45.951290;
	expected[106] = 42.715341;
	expected[107] = 39.498333;
	expected[108] = 36.032168;
	expected[109] = 32.490964;
	expected[110] = 28.905289;
	expected[111] = 25.415984;
	expected[112] = 22.003491;
	expected[113] = 18.644700;
	expected[114] = 15.331448;
	expected[115] = 12.238228;
	expected[116] = 9.188241;
	expected[117] = 5.789180;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_8(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.024336;
	expected[1] = -0.054007;
	expected[2] = -0.071207;
	expected[3] = -0.090020;
	expected[4] = -0.105265;
	expected[5] = -0.114130;
	expected[6] = -0.119322;
	expected[7] = -0.121046;
	expected[8] = -0.119471;
	expected[9] = -0.114754;
	expected[10] = -0.107002;
	expected[11] = -0.094333;
	expected[12] = -0.082233;
	expected[13] = -0.064660;
	expected[14] = -0.047240;
	expected[15] = -0.025539;
	expected[16] = -0.001364;
	expected[17] = 0.025393;
	expected[18] = 0.055211;
	expected[19] = 0.087864;
	expected[20] = 0.119090;
	expected[21] = 0.151730;
	expected[22] = 0.185555;
	expected[23] = 0.220341;
	expected[24] = 0.255850;
	expected[25] = 0.292309;
	expected[26] = 0.328513;
	expected[27] = 0.365074;
	expected[28] = 0.401087;
	expected[29] = 0.437596;
	expected[30] = 0.470950;
	expected[31] = 0.503270;
	expected[32] = 0.535476;
	expected[33] = 0.570273;
	expected[34] = 0.592709;
	expected[35] = 0.619647;
	expected[36] = 0.643284;
	expected[37] = 0.667621;
	expected[38] = 0.691448;
	expected[39] = 0.715402;
	expected[40] = 0.726424;
	expected[41] = 0.734378;
	expected[42] = 0.742312;
	expected[43] = 0.743235;
	expected[44] = 0.736867;
	expected[45] = 0.825383;
	expected[46] = 0.842611;
	expected[47] = 0.855929;
	expected[48] = 0.857974;
	expected[49] = 0.860716;
	expected[50] = 0.846466;
	expected[51] = 0.829105;
	expected[52] = 0.797938;
	expected[53] = 0.751129;
	expected[54] = 0.684331;
	expected[55] = 0.598432;
	expected[56] = 0.488126;
	expected[57] = 0.352762;
	expected[58] = 0.190131;
	expected[59] = 217.485350;
	expected[60] = 208.118911;
	expected[61] = 200.548848;
	expected[62] = 195.969121;
	expected[63] = 191.743411;
	expected[64] = 187.361604;
	expected[65] = 182.944717;
	expected[66] = 178.512605;
	expected[67] = 174.304189;
	expected[68] = 169.934834;
	expected[69] = 165.596651;
	expected[70] = 161.189894;
	expected[71] = 157.079704;
	expected[72] = 152.935928;
	expected[73] = 149.033181;
	expected[74] = 145.201090;
	expected[75] = 141.517841;
	expected[76] = 137.974097;
	expected[77] = 134.539630;
	expected[78] = 131.229098;
	expected[79] = 128.011002;
	expected[80] = 124.773909;
	expected[81] = 121.530265;
	expected[82] = 118.290753;
	expected[83] = 115.064835;
	expected[84] = 111.838120;
	expected[85] = 108.663359;
	expected[86] = 105.508107;
	expected[87] = 102.401460;
	expected[88] = 99.389994;
	expected[89] = 96.363831;
	expected[90] = 93.362983;
	expected[91] = 90.434266;
	expected[92] = 87.682147;
	expected[93] = 84.667728;
	expected[94] = 81.809101;
	expected[95] = 78.977284;
	expected[96] = 76.252087;
	expected[97] = 73.165675;
	expected[98] = 70.175219;
	expected[99] = 67.189514;
	expected[100] = 63.864509;
	expected[101] = 60.247028;
	expected[102] = 56.449306;
	expected[103] = 52.606640;
	expected[104] = 48.427472;
	expected[105] = 45.521302;
	expected[106] = 42.345466;
	expected[107] = 38.676627;
	expected[108] = 35.600124;
	expected[109] = 32.111247;
	expected[110] = 28.554832;
	expected[111] = 25.113958;
	expected[112] = 21.941072;
	expected[113] = 18.014876;
	expected[114] = 15.126636;
	expected[115] = 11.693684;
	expected[116] = 8.832424;
	expected[117] = 5.806389;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_9(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.029201;
	expected[1] = -0.053242;
	expected[2] = -0.077685;
	expected[3] = -0.094788;
	expected[4] = -0.107714;
	expected[5] = -0.116741;
	expected[6] = -0.122098;
	expected[7] = -0.122642;
	expected[8] = -0.122033;
	expected[9] = -0.117938;
	expected[10] = -0.108853;
	expected[11] = -0.099620;
	expected[12] = -0.086240;
	expected[13] = -0.070244;
	expected[14] = -0.051718;
	expected[15] = -0.030334;
	expected[16] = -0.006390;
	expected[17] = 0.020083;
	expected[18] = 0.049482;
	expected[19] = 0.082554;
	expected[20] = 0.115926;
	expected[21] = 0.148786;
	expected[22] = 0.179720;
	expected[23] = 0.214415;
	expected[24] = 0.249838;
	expected[25] = 0.285721;
	expected[26] = 0.321839;
	expected[27] = 0.358329;
	expected[28] = 0.394023;
	expected[29] = 0.429137;
	expected[30] = 0.463743;
	expected[31] = 0.496661;
	expected[32] = 0.529695;
	expected[33] = 0.556646;
	expected[34] = 0.586615;
	expected[35] = 0.613978;
	expected[36] = 0.637582;
	expected[37] = 0.665977;
	expected[38] = 0.683534;
	expected[39] = 0.707120;
	expected[40] = 0.719015;
	expected[41] = 0.728040;
	expected[42] = 0.734628;
	expected[43] = 0.737493;
	expected[44] = 0.728627;
	expected[45] = 0.842326;
	expected[46] = 0.858765;
	expected[47] = 0.871902;
	expected[48] = 0.879857;
	expected[49] = 0.879021;
	expected[50] = 0.864172;
	expected[51] = 0.846922;
	expected[52] = 0.815564;
	expected[53] = 0.767945;
	expected[54] = 0.702175;
	expected[55] = 0.614182;
	expected[56] = 0.502615;
	expected[57] = 0.364475;
	expected[58] = 0.197607;
	expected[59] = 218.246039;
	expected[60] = 208.429071;
	expected[61] = 201.494000;
	expected[62] = 196.755163;
	expected[63] = 192.379085;
	expected[64] = 188.018189;
	expected[65] = 183.615274;
	expected[66] = 179.081741;
	expected[67] = 174.937099;
	expected[68] = 170.597544;
	expected[69] = 166.165113;
	expected[70] = 161.937866;
	expected[71] = 157.718557;
	expected[72] = 153.609061;
	expected[73] = 149.629696;
	expected[74] = 145.779913;
	expected[75] = 142.076572;
	expected[76] = 138.518742;
	expected[77] = 135.080878;
	expected[78] = 131.722990;
	expected[79] = 128.383674;
	expected[80] = 125.151573;
	expected[81] = 122.002322;
	expected[82] = 118.748305;
	expected[83] = 115.507808;
	expected[84] = 112.290292;
	expected[85] = 109.101992;
	expected[86] = 105.928981;
	expected[87] = 102.818620;
	expected[88] = 99.751858;
	expected[89] = 96.715876;
	expected[90] = 93.741691;
	expected[91] = 90.931930;
	expected[92] = 87.689118;
	expected[93] = 85.017297;
	expected[94] = 82.165265;
	expected[95] = 79.420283;
	expected[96] = 76.278544;
	expected[97] = 73.614520;
	expected[98] = 70.540283;
	expected[99] = 67.526920;
	expected[100] = 64.125773;
	expected[101] = 60.375256;
	expected[102] = 56.514362;
	expected[103] = 52.478916;
	expected[104] = 47.770618;
	expected[105] = 44.863955;
	expected[106] = 41.743660;
	expected[107] = 38.440510;
	expected[108] = 35.005227;
	expected[109] = 31.542229;
	expected[110] = 28.015000;
	expected[111] = 24.577226;
	expected[112] = 21.216307;
	expected[113] = 17.903733;
	expected[114] = 14.778889;
	expected[115] = 11.547694;
	expected[116] = 8.623105;
	expected[117] = 5.507108;

	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_10(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.018479;
	expected[1] = -0.050309;
	expected[2] = -0.071079;
	expected[3] = -0.087453;
	expected[4] = -0.099749;
	expected[5] = -0.108245;
	expected[6] = -0.109450;
	expected[7] = -0.112280;
	expected[8] = -0.113149;
	expected[9] = -0.105731;
	expected[10] = -0.098115;
	expected[11] = -0.089457;
	expected[12] = -0.076895;
	expected[13] = -0.061029;
	expected[14] = -0.042381;
	expected[15] = -0.018651;
	expected[16] = 0.004188;
	expected[17] = 0.030259;
	expected[18] = 0.057278;
	expected[19] = 0.094223;
	expected[20] = 0.122698;
	expected[21] = 0.150856;
	expected[22] = 0.183062;
	expected[23] = 0.216472;
	expected[24] = 0.250542;
	expected[25] = 0.285069;
	expected[26] = 0.320128;
	expected[27] = 0.354668;
	expected[28] = 0.388831;
	expected[29] = 0.422379;
	expected[30] = 0.455395;
	expected[31] = 0.486837;
	expected[32] = 0.516849;
	expected[33] = 0.545342;
	expected[34] = 0.572263;
	expected[35] = 0.597539;
	expected[36] = 0.624139;
	expected[37] = 0.642413;
	expected[38] = 0.664030;
	expected[39] = 0.688506;
	expected[40] = 0.697071;
	expected[41] = 0.709827;
	expected[42] = 0.713316;
	expected[43] = 0.714301;
	expected[44] = 0.706256;
	expected[45] = 0.796566;
	expected[46] = 0.810480;
	expected[47] = 0.822629;
	expected[48] = 0.829081;
	expected[49] = 0.826471;
	expected[50] = 0.811872;
	expected[51] = 0.794646;
	expected[52] = 0.763803;
	expected[53] = 0.717529;
	expected[54] = 0.654266;
	expected[55] = 0.571390;
	expected[56] = 0.466590;
	expected[57] = 0.337270;
	expected[58] = 0.181893;
	expected[59] = 217.092344;
	expected[60] = 207.839472;
	expected[61] = 200.555717;
	expected[62] = 195.716070;
	expected[63] = 191.265432;
	expected[64] = 186.857349;
	expected[65] = 182.176311;
	expected[66] = 177.853917;
	expected[67] = 173.836387;
	expected[68] = 169.372283;
	expected[69] = 165.092814;
	expected[70] = 160.955476;
	expected[71] = 156.846090;
	expected[72] = 152.816901;
	expected[73] = 148.905542;
	expected[74] = 145.017434;
	expected[75] = 141.432295;
	expected[76] = 137.932011;
	expected[77] = 134.646846;
	expected[78] = 131.219689;
	expected[79] = 128.089273;
	expected[80] = 125.043813;
	expected[81] = 121.874008;
	expected[82] = 118.694125;
	expected[83] = 115.524926;
	expected[84] = 112.372566;
	expected[85] = 109.229938;
	expected[86] = 106.138372;
	expected[87] = 103.084409;
	expected[88] = 100.072354;
	expected[89] = 97.091083;
	expected[90] = 94.164726;
	expected[91] = 91.279588;
	expected[92] = 88.425492;
	expected[93] = 85.587527;
	expected[94] = 82.765578;
	expected[95] = 80.024875;
	expected[96] = 76.687446;
	expected[97] = 74.274728;
	expected[98] = 71.736048;
	expected[99] = 68.336290;
	expected[100] = 64.982916;
	expected[101] = 61.359414;
	expected[102] = 57.476734;
	expected[103] = 53.586489;
	expected[104] = 49.336943;
	expected[105] = 46.522814;
	expected[106] = 43.358474;
	expected[107] = 40.026975;
	expected[108] = 36.556994;
	expected[109] = 32.999336;
	expected[110] = 29.382609;
	expected[111] = 25.871252;
	expected[112] = 22.444383;
	expected[113] = 19.071414;
	expected[114] = 15.746104;
	expected[115] = 12.455279;
	expected[116] = 9.205539;
	expected[117] = 6.010417;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_11(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.024644;
	expected[1] = -0.052960;
	expected[2] = -0.073798;
	expected[3] = -0.088857;
	expected[4] = -0.099539;
	expected[5] = -0.110881;
	expected[6] = -0.116382;
	expected[7] = -0.118262;
	expected[8] = -0.116687;
	expected[9] = -0.112066;
	expected[10] = -0.104504;
	expected[11] = -0.094084;
	expected[12] = -0.080083;
	expected[13] = -0.063933;
	expected[14] = -0.046482;
	expected[15] = -0.025527;
	expected[16] = -0.002214;
	expected[17] = 0.023521;
	expected[18] = 0.052039;
	expected[19] = 0.087880;
	expected[20] = 0.116916;
	expected[21] = 0.146556;
	expected[22] = 0.177489;
	expected[23] = 0.210824;
	expected[24] = 0.244821;
	expected[25] = 0.279230;
	expected[26] = 0.313818;
	expected[27] = 0.348288;
	expected[28] = 0.382457;
	expected[29] = 0.416207;
	expected[30] = 0.448680;
	expected[31] = 0.479974;
	expected[32] = 0.509892;
	expected[33] = 0.538418;
	expected[34] = 0.565614;
	expected[35] = 0.590770;
	expected[36] = 0.614304;
	expected[37] = 0.636786;
	expected[38] = 0.658691;
	expected[39] = 0.679301;
	expected[40] = 0.694715;
	expected[41] = 0.701092;
	expected[42] = 0.708060;
	expected[43] = 0.708948;
	expected[44] = 0.698279;
	expected[45] = 0.810585;
	expected[46] = 0.823404;
	expected[47] = 0.839097;
	expected[48] = 0.841462;
	expected[49] = 0.843756;
	expected[50] = 0.829569;
	expected[51] = 0.812225;
	expected[52] = 0.781424;
	expected[53] = 0.735128;
	expected[54] = 0.671425;
	expected[55] = 0.588086;
	expected[56] = 0.481165;
	expected[57] = 0.349138;
	expected[58] = 0.189448;
	expected[59] = 217.952048;
	expected[60] = 208.444287;
	expected[61] = 201.185786;
	expected[62] = 196.256596;
	expected[63] = 191.730583;
	expected[64] = 187.508289;
	expected[65] = 183.120925;
	expected[66] = 178.715944;
	expected[67] = 174.530342;
	expected[68] = 170.192164;
	expected[69] = 165.890994;
	expected[70] = 161.648725;
	expected[71] = 157.443949;
	expected[72] = 153.383786;
	expected[73] = 149.489124;
	expected[74] = 145.696168;
	expected[75] = 142.051759;
	expected[76] = 138.551137;
	expected[77] = 135.172905;
	expected[78] = 131.731980;
	expected[79] = 128.622268;
	expected[80] = 125.496008;
	expected[81] = 122.352748;
	expected[82] = 119.157224;
	expected[83] = 115.972084;
	expected[84] = 112.805896;
	expected[85] = 109.665559;
	expected[86] = 106.558875;
	expected[87] = 103.487395;
	expected[88] = 100.449325;
	expected[89] = 97.473595;
	expected[90] = 94.541713;
	expected[91] = 91.650562;
	expected[92] = 88.786871;
	expected[93] = 85.934814;
	expected[94] = 83.112563;
	expected[95] = 80.294601;
	expected[96] = 77.460249;
	expected[97] = 74.861602;
	expected[98] = 71.652779;
	expected[99] = 68.587347;
	expected[100] = 65.258261;
	expected[101] = 61.533604;
	expected[102] = 57.524016;
	expected[103] = 53.434966;
	expected[104] = 48.782534;
	expected[105] = 45.056301;
	expected[106] = 42.676317;
	expected[107] = 38.958806;
	expected[108] = 35.911909;
	expected[109] = 32.387676;
	expected[110] = 28.809777;
	expected[111] = 25.322744;
	expected[112] = 21.908837;
	expected[113] = 18.544142;
	expected[114] = 15.517585;
	expected[115] = 11.980390;
	expected[116] = 8.760148;
	expected[117] = 5.692581;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_12(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.029760;
	expected[1] = -0.056675;
	expected[2] = -0.078398;
	expected[3] = -0.095600;
	expected[4] = -0.108617;
	expected[5] = -0.116020;
	expected[6] = -0.122982;
	expected[7] = -0.125128;
	expected[8] = -0.122117;
	expected[9] = -0.119031;
	expected[10] = -0.111350;
	expected[11] = -0.100778;
	expected[12] = -0.087467;
	expected[13] = -0.071546;
	expected[14] = -0.053130;
	expected[15] = -0.032186;
	expected[16] = -0.008861;
	expected[17] = 0.017467;
	expected[18] = 0.046832;
	expected[19] = 0.079900;
	expected[20] = 0.111234;
	expected[21] = 0.144802;
	expected[22] = 0.177842;
	expected[23] = 0.215192;
	expected[24] = 0.247416;
	expected[25] = 0.283314;
	expected[26] = 0.319444;
	expected[27] = 0.355515;
	expected[28] = 0.391668;
	expected[29] = 0.426736;
	expected[30] = 0.460968;
	expected[31] = 0.494461;
	expected[32] = 0.526703;
	expected[33] = 0.555210;
	expected[34] = 0.584537;
	expected[35] = 0.612486;
	expected[36] = 0.639477;
	expected[37] = 0.660571;
	expected[38] = 0.681445;
	expected[39] = 0.705043;
	expected[40] = 0.716860;
	expected[41] = 0.726001;
	expected[42] = 0.734409;
	expected[43] = 0.737092;
	expected[44] = 0.731702;
	expected[45] = 0.818285;
	expected[46] = 0.833320;
	expected[47] = 0.845920;
	expected[48] = 0.854767;
	expected[49] = 0.852184;
	expected[50] = 0.839678;
	expected[51] = 0.822464;
	expected[52] = 0.791159;
	expected[53] = 0.743717;
	expected[54] = 0.678026;
	expected[55] = 0.591848;
	expected[56] = 0.482200;
	expected[57] = 0.347039;
	expected[58] = 0.185902;
	expected[59] = 218.123842;
	expected[60] = 208.534136;
	expected[61] = 201.382370;
	expected[62] = 196.648827;
	expected[63] = 192.276428;
	expected[64] = 187.769642;
	expected[65] = 183.496303;
	expected[66] = 179.086766;
	expected[67] = 174.774423;
	expected[68] = 170.483925;
	expected[69] = 166.126970;
	expected[70] = 161.824791;
	expected[71] = 157.604681;
	expected[72] = 153.493836;
	expected[73] = 149.513941;
	expected[74] = 145.677433;
	expected[75] = 141.995135;
	expected[76] = 138.437999;
	expected[77] = 134.995572;
	expected[78] = 131.633985;
	expected[79] = 128.390420;
	expected[80] = 125.088320;
	expected[81] = 121.868985;
	expected[82] = 118.515383;
	expected[83] = 115.392821;
	expected[84] = 112.171890;
	expected[85] = 108.980229;
	expected[86] = 105.824419;
	expected[87] = 102.689158;
	expected[88] = 99.621931;
	expected[89] = 96.600623;
	expected[90] = 93.601873;
	expected[91] = 90.726719;
	expected[92] = 87.616381;
	expected[93] = 84.862730;
	expected[94] = 82.356495;
	expected[95] = 79.123256;
	expected[96] = 76.228901;
	expected[97] = 73.418840;
	expected[98] = 70.411055;
	expected[99] = 67.387842;
	expected[100] = 64.031069;
	expected[101] = 60.226708;
	expected[102] = 56.543066;
	expected[103] = 52.679452;
	expected[104] = 48.509247;
	expected[105] = 45.628309;
	expected[106] = 42.373968;
	expected[107] = 39.137481;
	expected[108] = 35.614961;
	expected[109] = 32.184715;
	expected[110] = 28.621551;
	expected[111] = 25.159848;
	expected[112] = 21.784066;
	expected[113] = 18.475076;
	expected[114] = 15.502023;
	expected[115] = 12.037885;
	expected[116] = 8.651311;
	expected[117] = 5.896831;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_13(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.027398;
	expected[1] = -0.054156;
	expected[2] = -0.080237;
	expected[3] = -0.097677;
	expected[4] = -0.109439;
	expected[5] = -0.119315;
	expected[6] = -0.125475;
	expected[7] = -0.127396;
	expected[8] = -0.125967;
	expected[9] = -0.121371;
	expected[10] = -0.113758;
	expected[11] = -0.103265;
	expected[12] = -0.089971;
	expected[13] = -0.073993;
	expected[14] = -0.055459;
	expected[15] = -0.034521;
	expected[16] = -0.011469;
	expected[17] = 0.013675;
	expected[18] = 0.041980;
	expected[19] = 0.074414;
	expected[20] = 0.105685;
	expected[21] = 0.138345;
	expected[22] = 0.172175;
	expected[23] = 0.207099;
	expected[24] = 0.243897;
	expected[25] = 0.278444;
	expected[26] = 0.316379;
	expected[27] = 0.349687;
	expected[28] = 0.385318;
	expected[29] = 0.420399;
	expected[30] = 0.454752;
	expected[31] = 0.487617;
	expected[32] = 0.519122;
	expected[33] = 0.549630;
	expected[34] = 0.578771;
	expected[35] = 0.603916;
	expected[36] = 0.629607;
	expected[37] = 0.655260;
	expected[38] = 0.676184;
	expected[39] = 0.697654;
	expected[40] = 0.710024;
	expected[41] = 0.721221;
	expected[42] = 0.729184;
	expected[43] = 0.731472;
	expected[44] = 0.723767;
	expected[45] = 0.830115;
	expected[46] = 0.850611;
	expected[47] = 0.862124;
	expected[48] = 0.870586;
	expected[49] = 0.869055;
	expected[50] = 0.855785;
	expected[51] = 0.838662;
	expected[52] = 0.808126;
	expected[53] = 0.759966;
	expected[54] = 0.693616;
	expected[55] = 0.606158;
	expected[56] = 0.495095;
	expected[57] = 0.357730;
	expected[58] = 0.192443;
	expected[59] = 218.276520;
	expected[60] = 208.676723;
	expected[61] = 201.934325;
	expected[62] = 197.243654;
	expected[63] = 192.795585;
	expected[64] = 188.505087;
	expected[65] = 184.159459;
	expected[66] = 179.732530;
	expected[67] = 175.514105;
	expected[68] = 171.122765;
	expected[69] = 166.753855;
	expected[70] = 162.435212;
	expected[71] = 158.190463;
	expected[72] = 154.048093;
	expected[73] = 150.033013;
	expected[74] = 146.166719;
	expected[75] = 142.465394;
	expected[76] = 138.929423;
	expected[77] = 135.508497;
	expected[78] = 132.153361;
	expected[79] = 128.895382;
	expected[80] = 125.618871;
	expected[81] = 122.336370;
	expected[82] = 119.051969;
	expected[83] = 115.729147;
	expected[84] = 112.562149;
	expected[85] = 109.290583;
	expected[86] = 106.223012;
	expected[87] = 103.098563;
	expected[88] = 100.017170;
	expected[89] = 96.974883;
	expected[90] = 93.991988;
	expected[91] = 91.051608;
	expected[92] = 88.125119;
	expected[93] = 85.437860;
	expected[94] = 82.080588;
	expected[95] = 79.545694;
	expected[96] = 76.127720;
	expected[97] = 73.683760;
	expected[98] = 70.743393;
	expected[99] = 67.675928;
	expected[100] = 64.266925;
	expected[101] = 60.569195;
	expected[102] = 56.610570;
	expected[103] = 52.550868;
	expected[104] = 47.984834;
	expected[105] = 45.003341;
	expected[106] = 41.900377;
	expected[107] = 38.554924;
	expected[108] = 35.135531;
	expected[109] = 31.639991;
	expected[110] = 28.114748;
	expected[111] = 24.766883;
	expected[112] = 21.384394;
	expected[113] = 18.103965;
	expected[114] = 14.787559;
	expected[115] = 11.626707;
	expected[116] = 8.577548;
	expected[117] = 5.605059;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_14(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.027213;
	expected[1] = -0.054080;
	expected[2] = -0.071737;
	expected[3] = -0.091343;
	expected[4] = -0.104109;
	expected[5] = -0.112771;
	expected[6] = -0.117867;
	expected[7] = -0.119603;
	expected[8] = -0.118150;
	expected[9] = -0.113662;
	expected[10] = -0.106240;
	expected[11] = -0.095465;
	expected[12] = -0.081278;
	expected[13] = -0.067260;
	expected[14] = -0.049029;
	expected[15] = -0.028163;
	expected[16] = -0.004938;
	expected[17] = 0.020722;
	expected[18] = 0.049219;
	expected[19] = 0.081203;
	expected[20] = 0.111809;
	expected[21] = 0.144949;
	expected[22] = 0.175560;
	expected[23] = 0.211089;
	expected[24] = 0.245018;
	expected[25] = 0.276578;
	expected[26] = 0.311172;
	expected[27] = 0.345655;
	expected[28] = 0.379766;
	expected[29] = 0.413229;
	expected[30] = 0.446132;
	expected[31] = 0.477465;
	expected[32] = 0.507428;
	expected[33] = 0.535865;
	expected[34] = 0.562909;
	expected[35] = 0.588457;
	expected[36] = 0.612067;
	expected[37] = 0.634600;
	expected[38] = 0.656244;
	expected[39] = 0.674079;
	expected[40] = 0.689320;
	expected[41] = 0.701106;
	expected[42] = 0.706229;
	expected[43] = 0.708204;
	expected[44] = 0.701164;
	expected[45] = 0.784864;
	expected[46] = 0.800858;
	expected[47] = 0.813878;
	expected[48] = 0.820844;
	expected[49] = 0.819101;
	expected[50] = 0.805240;
	expected[51] = 0.788489;
	expected[52] = 0.757332;
	expected[53] = 0.711034;
	expected[54] = 0.647528;
	expected[55] = 0.564924;
	expected[56] = 0.460407;
	expected[57] = 0.331671;
	expected[58] = 0.177552;
	expected[59] = 218.006287;
	expected[60] = 208.370029;
	expected[61] = 200.878471;
	expected[62] = 196.277158;
	expected[63] = 191.865375;
	expected[64] = 187.470526;
	expected[65] = 183.054887;
	expected[66] = 178.636942;
	expected[67] = 174.453356;
	expected[68] = 170.116937;
	expected[69] = 165.817023;
	expected[70] = 161.544117;
	expected[71] = 157.342814;
	expected[72] = 153.354872;
	expected[73] = 149.425883;
	expected[74] = 145.630472;
	expected[75] = 141.983645;
	expected[76] = 138.479589;
	expected[77] = 135.095479;
	expected[78] = 131.800717;
	expected[79] = 128.602951;
	expected[80] = 125.331772;
	expected[81] = 122.223069;
	expected[82] = 118.950865;
	expected[83] = 115.766409;
	expected[84] = 112.695177;
	expected[85] = 109.551863;
	expected[86] = 106.442170;
	expected[87] = 103.371164;
	expected[88] = 100.344928;
	expected[89] = 97.348307;
	expected[90] = 94.413157;
	expected[91] = 91.519195;
	expected[92] = 88.658966;
	expected[93] = 85.809499;
	expected[94] = 82.972555;
	expected[95] = 80.154669;
	expected[96] = 77.322332;
	expected[97] = 74.607198;
	expected[98] = 70.621701;
	expected[99] = 68.516697;
	expected[100] = 65.136838;
	expected[101] = 61.485003;
	expected[102] = 57.569723;
	expected[103] = 53.654911;
	expected[104] = 49.543326;
	expected[105] = 46.639253;
	expected[106] = 43.489253;
	expected[107] = 40.140833;
	expected[108] = 36.648589;
	expected[109] = 33.072661;
	expected[110] = 29.441381;
	expected[111] = 25.937244;
	expected[112] = 22.514393;
	expected[113] = 19.155008;
	expected[114] = 15.836444;
	expected[115] = 12.557801;
	expected[116] = 9.322445;
	expected[117] = 6.110204;
	
	for (int i = 0; i < TotalDof; i++)
		EXPECT_NEAR(result[i], expected[i], EPSILON);
}

static void check_15(double* result, int TotalDof)
{
	EXPECT_EQ(TotalDof, 118);

	vector<double> expected(118);
	expected[0] = -0.043403;
	expected[1] = -0.056025;
	expected[2] = -0.077129;
	expected[3] = -0.093787;
	expected[4] = -0.106379;
	expected[5] = -0.115182;
	expected[6] = -0.120426;
	expected[7] = -0.122314;
	expected[8] = -0.120268;
	expected[9] = -0.116577;
	expected[10] = -0.108597;
	expected[11] = -0.098828;
	expected[12] = -0.085834;
	expected[13] = -0.070288;
	expected[14] = -0.052310;
	expected[15] = -0.031886;
	expected[16] = -0.009195;
	expected[17] = 0.016014;
	expected[18] = 0.044094;
	expected[19] = 0.075819;
	expected[20] = 0.105932;
	expected[21] = 0.137347;
	expected[22] = 0.169849;
	expected[23] = 0.203228;
	expected[24] = 0.238674;
	expected[25] = 0.271423;
	expected[26] = 0.305997;
	expected[27] = 0.340705;
	expected[28] = 0.376590;
	expected[29] = 0.409853;
	expected[30] = 0.439852;
	expected[31] = 0.471097;
	expected[32] = 0.501065;
	expected[33] = 0.529658;
	expected[34] = 0.556412;
	expected[35] = 0.581521;
	expected[36] = 0.605484;
	expected[37] = 0.627920;
	expected[38] = 0.649847;
	expected[39] = 0.671304;
	expected[40] = 0.685790;
	expected[41] = 0.693851;
	expected[42] = 0.701195;
	expected[43] = 0.703511;
	expected[44] = 0.693398;
	expected[45] = 0.800622;
	expected[46] = 0.816181;
	expected[47] = 0.830786;
	expected[48] = 0.836243;
	expected[49] = 0.834664;
	expected[50] = 0.821058;
	expected[51] = 0.803889;
	expected[52] = 0.773118;
	expected[53] = 0.726849;
	expected[54] = 0.663148;
	expected[55] = 0.579772;
	expected[56] = 0.473580;
	expected[57] = 0.342351;
	expected[58] = 0.184174;
	expected[59] = 219.754445;
	expected[60] = 208.899932;
	expected[61] = 201.680912;
	expected[62] = 196.893446;
	expected[63] = 192.485122;
	expected[64] = 188.108576;
	expected[65] = 183.705832;
	expected[66] = 179.295037;
	expected[67] = 175.059566;
	expected[68] = 170.764095;
	expected[69] = 166.419985;
	expected[70] = 162.184559;
	expected[71] = 158.005589;
	expected[72] = 153.933374;
	expected[73] = 149.988287;
	expected[74] = 146.183387;
	expected[75] = 142.531085;
	expected[76] = 139.021792;
	expected[77] = 135.634492;
	expected[78] = 132.333298;
	expected[79] = 129.142127;
	expected[80] = 125.931151;
	expected[81] = 122.712114;
	expected[82] = 119.495010;
	expected[83] = 116.226831;
	expected[84] = 113.118177;
	expected[85] = 109.959773;
	expected[86] = 106.827545;
	expected[87] = 103.682233;
	expected[88] = 100.648848;
	expected[89] = 97.746770;
	expected[90] = 94.804068;
	expected[91] = 91.898915;
	expected[92] = 89.020182;
	expected[93] = 86.177789;
	expected[94] = 83.351142;
	expected[95] = 80.512903;
	expected[96] = 77.677773;
	expected[97] = 74.803568;
	expected[98] = 71.861136;
	expected[99] = 68.774637;
	expected[100] = 65.416341;
	expected[101] = 61.669156;
	expected[102] = 57.620249;
	expected[103] = 53.513411;
	expected[104] = 49.150126;
	expected[105] = 45.979179;
	expected[106] = 43.063760;
	expected[107] = 39.508093;
	expected[108] = 36.041360;
	expected[109] = 32.499464;
	expected[110] = 28.913188;
	expected[111] = 25.423676;
	expected[112] = 22.011424;
	expected[113] = 18.653354;
	expected[114] = 15.343834;
	expected[115] = 12.108078;
	expected[116] = 8.983096;
	expected[117] = 5.799786;

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
			data.load("surplus_1.plt", 0);
			data.load("surplus_2.plt", 1);
			data.load("surplus_3.plt", 2);
			data.load("surplus_4.plt", 3);
			data.load("surplus_5.plt", 4);
			data.load("surplus_6.plt", 5);
			data.load("surplus_7.plt", 6);
			data.load("surplus_8.plt", 7);
			data.load("surplus_9.plt", 8);
			data.load("surplus_10.plt", 9);
			data.load("surplus_11.plt", 10);
			data.load("surplus_12.plt", 11);
			data.load("surplus_13.plt", 12);
			data.load("surplus_14.plt", 13);
			data.load("surplus_15.plt", 14);
			data.load("surplus_16.plt", 15);

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
			data.load("surplus_1.plt", 0);
			data.load("surplus_2.plt", 1);
			data.load("surplus_3.plt", 2);
			data.load("surplus_4.plt", 3);
			data.load("surplus_5.plt", 4);
			data.load("surplus_6.plt", 5);
			data.load("surplus_7.plt", 6);
			data.load("surplus_8.plt", 7);
			data.load("surplus_9.plt", 8);
			data.load("surplus_10.plt", 9);
			data.load("surplus_11.plt", 10);
			data.load("surplus_12.plt", 11);
			data.load("surplus_13.plt", 12);
			data.load("surplus_14.plt", 13);
			data.load("surplus_15.plt", 14);
			data.load("surplus_16.plt", 15);

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
			data.load("surplus_1.plt", 0);
			data.load("surplus_2.plt", 1);
			data.load("surplus_3.plt", 2);
			data.load("surplus_4.plt", 3);
			data.load("surplus_5.plt", 4);
			data.load("surplus_6.plt", 5);
			data.load("surplus_7.plt", 6);
			data.load("surplus_8.plt", 7);
			data.load("surplus_9.plt", 8);
			data.load("surplus_10.plt", 9);
			data.load("surplus_11.plt", 10);
			data.load("surplus_12.plt", 11);
			data.load("surplus_13.plt", 12);
			data.load("surplus_14.plt", 13);
			data.load("surplus_15.plt", 14);
			data.load("surplus_16.plt", 15);

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
			data.load("surplus_1.plt", 0);
			data.load("surplus_2.plt", 1);
			data.load("surplus_3.plt", 2);
			data.load("surplus_4.plt", 3);
			data.load("surplus_5.plt", 4);
			data.load("surplus_6.plt", 5);
			data.load("surplus_7.plt", 6);
			data.load("surplus_8.plt", 7);
			data.load("surplus_9.plt", 8);
			data.load("surplus_10.plt", 9);
			data.load("surplus_11.plt", 10);
			data.load("surplus_12.plt", 11);
			data.load("surplus_13.plt", 12);
			data.load("surplus_14.plt", 13);
			data.load("surplus_15.plt", 14);
			data.load("surplus_16.plt", 15);

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
			data.load("surplus_1.plt", 0);
			data.load("surplus_2.plt", 1);
			data.load("surplus_3.plt", 2);
			data.load("surplus_4.plt", 3);
			data.load("surplus_5.plt", 4);
			data.load("surplus_6.plt", 5);
			data.load("surplus_7.plt", 6);
			data.load("surplus_8.plt", 7);
			data.load("surplus_9.plt", 8);
			data.load("surplus_10.plt", 9);
			data.load("surplus_11.plt", 10);
			data.load("surplus_12.plt", 11);
			data.load("surplus_13.plt", 12);
			data.load("surplus_14.plt", 13);
			data.load("surplus_15.plt", 14);
			data.load("surplus_16.plt", 15);

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
				data.load("surplus_1.plt", 0);
				data.load("surplus_2.plt", 1);
				data.load("surplus_3.plt", 2);
				data.load("surplus_4.plt", 3);
				data.load("surplus_5.plt", 4);
				data.load("surplus_6.plt", 5);
				data.load("surplus_7.plt", 6);
				data.load("surplus_8.plt", 7);
				data.load("surplus_9.plt", 8);
				data.load("surplus_10.plt", 9);
				data.load("surplus_11.plt", 10);
				data.load("surplus_12.plt", 11);
				data.load("surplus_13.plt", 12);
				data.load("surplus_14.plt", 13);
				data.load("surplus_15.plt", 14);
				data.load("surplus_16.plt", 15);

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

	const char* cntests = getenv("NTESTS");
	if (cntests)
		ntests = atoi(cntests);

	GoogleTest(argc, argv);

	::testing::InitGoogleTest(&argc, argv);

	return RUN_ALL_TESTS();
}

