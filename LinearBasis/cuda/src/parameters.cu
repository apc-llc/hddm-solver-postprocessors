#include "check.h"
#include "parameters.h"
#include "process.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <mpi.h>
#include <sstream>
#include <string>
#include <string.h>
#include <unistd.h>

using namespace NAMESPACE;
using namespace std;

// For compatibility with Fortran, all parameters are declared
// as global variables. Parameters class stores only references
// to these global variables.
extern "C"
{
	#define ALIGN_BOOL_4 __attribute__((aligned(4)))
	#include "parameters.c"
	#undef ALIGN_BOOL_4
};

#undef ASSIGN

#define ASSIGN(name) name(::name)

Parameters::Parameters(const string& targetSuffix, const string& configFile) :

ASSIGN(ngpus),
ASSIGN(warpSize),
ASSIGN(priority),
ASSIGN(nagents),
ASSIGN(enableRuntimeOptimization),
ASSIGN(binaryio)

// XXX Add new parameters here

#undef ASSIGN

{
	MPI_Process* process;
	MPI_ERR_CHECK(MPI_Process_get(&process));

	if (process->isMaster())
	{
		cout << "------------------------------------" << endl;
		cout << "LOADING \"" << targetSuffix << "\" POSTPROCESSOR CONFIG (" << configFile << ")" << endl;
		cout << "------------------------------------" << endl;
	}
	
	map<string, bool> undefParams;

#if defined(CUDA_INTERPOLATOR)

	undefParams["ngpus"] = true;
	undefParams["warpSize"] = true;

#endif // CUDA_INTERPOLATOR

	undefParams["priority"] = true;
	undefParams["nagents"] = true;
	undefParams["enableRuntimeOptimization"] = true;
	undefParams["binaryio"] = true;

	// Read configuration file on master.
	ifstream cfg(configFile.c_str());
	if (!cfg.is_open())
	{
		cerr << "Cannot open config file \"" << configFile << "\"" << endl;
		process->abort();
	}

	string line;
	while (getline(cfg, line))
	{
		istringstream iss(line);
		string name;
		iss >> name;
		if (name[0] == '!') continue;
		string value;
		if (!(iss >> value))
		{
			if (name == "") continue;
			if (process->isMaster())
				cerr << "Missing \"" << name << "\" parameter value" << endl;
			process->abort();
		}
		if (value[0] == '!')
		{
			if (process->isMaster())
				cerr << "Missing \"" << name << "\" parameter value" << endl;
			process->abort();
		}

		if (name == "priorityCUDA")
		{
			priority = atoi(value.c_str());
			if (process->isMaster())
				cout << "priority : " << priority << endl;
			undefParams["priority"] = false;
		}
		else if ((name == "n_agents") || (name == "nagents"))
		{
			nagents = atoi(value.c_str());
			if (process->isMaster())
				cout << "nagents : " << nagents << endl;
			undefParams["nagents"] = false;
		}
		else if ((name == "enable_runtime_optimization") || (name == "enableRuntimeOptimization"))
		{
			if (process->isMaster())
				cout << "enableRuntimeOptimization : ";
			if ((value == "yes") || (value == "y") || (value == "true"))
			{
				enableRuntimeOptimization = true;
				if (process->isMaster())
					cout << value << endl;
			}
			else if ((value == "no") || (value == "n") || (value == "false"))
			{
				enableRuntimeOptimization = false;
				if (process->isMaster())
					cout << value << endl;
			}
			else
			{
				if (process->isMaster())
					cerr << "unknown" << endl;
				process->abort();
			}
			undefParams["enableRuntimeOptimization"] = false;
		}
		else if (name == "binaryio")
		{
			if (process->isMaster())
				cout << "binary i/o : ";
			if ((value == "yes") || (value == "y") || (value == "true"))
			{
				binaryio = true;
				if (process->isMaster())
					cout << value << endl;
			}
			else if ((value == "no") || (value == "n") || (value == "false"))
			{
				binaryio = false;
				if (process->isMaster())
					cout << value << endl;
			}
			else
			{
				if (process->isMaster())
					cerr << "unknown" << endl;
				process->abort();
			}
			undefParams["binaryio"] = false;
		}
	}
	
	cfg.close();

	// Check for available CUDA GPU(s)
	CUDA_ERR_CHECK(cudaGetDeviceCount(&ngpus));
	undefParams["ngpus"] = false;

	// Find out the total number of GPU(s) available to
	// all participating hosts.
	int ngpus_total = ngpus;
	MPI_ERR_CHECK(MPI_Allreduce(MPI_IN_PLACE, &ngpus_total, 1, MPI_INT,
		MPI_SUM, process->getComm()));

	if (process->isMaster())
		cout << ngpus_total << " GPU(s) available" << endl;
	cout << endl;
	
	// Find out the warp size for each GPU.

	// Check if something is still undefined.
	for (map<string, bool>::iterator i = undefParams.begin(), e = undefParams.end(); i != e; i++)
		if (i->second)
		{
			if (process->isMaster())
				cerr << "Required parameter \"" << i->first << "\" is undefined" << endl;
			process->abort();
		}
}

