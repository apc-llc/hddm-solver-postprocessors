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

ASSIGN(priority),
ASSIGN(nagents),
ASSIGN(binaryio),
ASSIGN(surplusCutoff),
ASSIGN(surplusCutoffDefined)

// XXX Add new parameters here

#undef ASSIGN

{
	MPI_Process* process;
	MPI_ERR_CHECK(MPI_Process_get(&process));

	if (process->isMaster())
	{
		process->cout("------------------------------------\n");
		process->cout("LOADING \"%s\" POSTPROCESSOR CONFIG (%s)\n", targetSuffix.c_str(), configFile.c_str());
		process->cout("------------------------------------\n");
	}
	
	map<string, bool> undefParams;

	undefParams["priority"] = true;
	undefParams["nagents"] = true;
	undefParams["binaryio"] = true;
	undefParams["surplusCutoff"] = true;

	// Read the configuration file.
	FILE* cfg = fopen(configFile.c_str(), "r");
    if (!cfg)
	{
		process->cerr("Cannot open config file \"%s\"\n", configFile.c_str());
		process->abort();
	}

	char* line = NULL;
	size_t szline = 0;
	while (getline(&line, &szline, cfg) != -1)
	{
		char* name = strtok(line, " \t\n");
		if (!name) continue;
		if (name[0] == '!') continue;
		char* value = strtok(NULL, " \t\n");
		if (!value)
		{
			if (process->isMaster())
				process->cerr("Missing \"%s\" parameter value\n", name);
			process->abort();
		}
		if (value[0] == '!')
		{
			if (process->isMaster())
				process->cerr("Missing \"%s\" parameter value\n", name);
			process->abort();
		}

		if ((string)name == "priorityCUDA")
		{
			priority = atoi(value);
			if (process->isMaster())
				process->cout("priority : %d\n", priority);
			undefParams["priority"] = false;
		}
		else if (((string)name == "n_agents") || ((string)name == "nagents"))
		{
			nagents = atoi(value);
			if (process->isMaster())
				process->cout("nagents : %d\n", nagents);
			undefParams["nagents"] = false;
		}
		else if ((string)name == "binaryio")
		{
			if (process->isMaster())
				process->cout("binary i/o : ");
			if (((string)value == "yes") || ((string)value == "y") || ((string)value == "true"))
			{
				binaryio = true;
				if (process->isMaster())
					process->cout("%s\n", value);
			}
			else if (((string)value == "no") || ((string)value == "n") || ((string)value == "false"))
			{
				binaryio = false;
				if (process->isMaster())
					process->cout("%s\n", value);
			}
			else
			{
				if (process->isMaster())
					process->cout("unknown\n");
				process->abort();
			}
			undefParams["binaryio"] = false;
		}
		else if ((string)name == "surplusCutoff")
		{
			surplusCutoff = atof(value);
			if (process->isMaster())
				process->cout("surplusCutoff : %d\n", surplusCutoff);
			undefParams["surplusCutoff"] = false;
		}
	}
	
	fclose(cfg);

	// Check if surplusCutoff has been defined.
	surplusCutoffDefined = true;
	if (undefParams["surplusCutoff"])
	{
		if (process->isMaster())
			process->cout("surplusCutoff : unused\n", surplusCutoff);

		surplusCutoffDefined = false;
		undefParams["surplusCutoff"] = false;
	}

	// Check if something is still undefined.
	for (map<string, bool>::iterator i = undefParams.begin(), e = undefParams.end(); i != e; i++)
		if (i->second)
		{
			if (process->isMaster())
				process->cerr("Required parameter \"%s\" is undefined\n", i->first.c_str());
			process->abort();
		}
}

