#ifdef HAVE_RUNTIME_OPTIMIZATION

#include "JIT.h"

#include <iostream>
#include <fstream>
#include <map>
#include <pthread.h>
#include <pstreams/pstream.h>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <vector>
#include <unistd.h>

using namespace cuda;
using namespace std;

template<>
const string InterpolateValueKernel::sh = INTERPOLATE_VALUE_SH;
template<>
const string InterpolateArrayKernel::sh = INTERPOLATE_ARRAY_SH;
template<>
const string InterpolateArrayManyStatelessKernel::sh = INTERPOLATE_ARRAY_MANY_STATELESS_SH;
template<>
const string InterpolateArrayManyMultistateKernel::sh = INTERPOLATE_ARRAY_MANY_MULTISTATE_SH;

template<typename K, typename F>
K& JIT::jitCompile(int dim, const string& funcnameTemplate, F fallbackFunc)
{
	map<int, K>* kernels_tls = NULL;

	// Already in TLS cache?
	{
		static __thread bool kernels_init = false;
		static __thread char kernels_a[sizeof(map<int, K>)];
		kernels_tls = (map<int, K>*)kernels_a;

		if (!kernels_init)
		{
			kernels_tls = new (kernels_a) map<int, K>();
			kernels_init = true;
		}

		K& kernel = kernels_tls->operator[](dim);

		// Already successfully compiled?
		if (kernel.filename != "")
			return kernel;

		// Already unsuccessfully compiled?
		if (kernel.compilationFailed)
		{
			kernel.dim = dim;
			kernel.fileowner = false;
			kernel.func = fallbackFunc;

			__sync_synchronize();
	
			return kernel;
		}
	}

	// Already in process cache?
	static map<int, K> kernels;

	static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

	PTHREAD_ERR_CHECK(pthread_mutex_lock(&mutex));

	K& kernel = kernels[dim];

	// Already successfully compiled?
	if (kernel.filename != "")
	{
		kernels_tls->operator[](dim) = kernel;
		PTHREAD_ERR_CHECK(pthread_mutex_unlock(&mutex));
		return kernel;
	}

	// Already unsuccessfully compiled?
	if (kernel.compilationFailed)
	{
		kernel.dim = dim;
		kernel.fileowner = false;
		kernel.func = fallbackFunc;

		__sync_synchronize();

		kernels_tls->operator[](dim) = kernel;
		PTHREAD_ERR_CHECK(pthread_mutex_unlock(&mutex));
		return kernel;
	}
	
	MPI_Process* process;
	MPI_ERR_CHECK(MPI_Process_get(&process));
	if (process->isMaster())
	{
		cout << "Performing deferred GPU kernel compilation for dim = " << dim << " ..." << endl;

		char* cwd = get_current_dir_name();
		string dir = (string)cwd + "/.cache";
		mkdir(dir.c_str(), S_IRWXU);
		struct TempFile
		{
			string filename;

			TempFile(const string& mask) : filename("")
			{
				vector<char> vfilename(mask.c_str(), mask.c_str() + mask.size() + 1);
				int fd = mkstemp(&vfilename[0]);
				if (fd == -1)
					return;
				close(fd);
				filename = (char*)&vfilename[0];
				bool keepCache = false;
				const char* keepCacheValue = getenv("KEEP_CACHE");
				if (keepCacheValue)
					keepCache = atoi(keepCacheValue);
				if (!keepCache)
					unlink(filename.c_str());
			}
		}
		tmp((string)cwd + "/.cache/fileXXXXXX");
		free(cwd);

		if (tmp.filename == "")
		{
			cerr << "Deferred GPU kernel temp file creation failed!" << endl;

			kernel.compilationFailed = true;
			kernel.dim = dim;
			kernel.fileowner = false;
			kernel.func = fallbackFunc;

			__sync_synchronize();

			kernels_tls->operator[](dim) = kernel;
			PTHREAD_ERR_CHECK(pthread_mutex_unlock(&mutex));
			return kernel;
		}

		// Generate function name for specific number of arguments.
		stringstream sfuncname;
		sfuncname << funcnameTemplate;
		sfuncname << dim;
		string funcname = sfuncname.str();

		// Read the compile command template.
		stringstream cmd;
		{
			std::ifstream t(kernel.sh.c_str());
			std::string sh((std::istreambuf_iterator<char>(t)),
				std::istreambuf_iterator<char>());
			stringstream snewline;
			snewline << endl;
			string newline = snewline.str();
			for (int pos = sh.find(newline); pos != string::npos; pos = sh.find(newline))
				sh.erase(pos, newline.size());
			cmd << sh;
			cmd << " -DDEFERRED";
			cmd << " -DFUNCNAME=";
			cmd << funcname;
			cmd << " -DDIM=";
			cmd << dim;
	
			cmd << " -o ";
			cmd << tmp.filename;
		}
		cout << cmd.str() << endl;

		// Run compiler as a process and create a streambuf that
		// reads its stdout and stderr.
		{
			redi::ipstream proc(cmd.str(), redi::pstreams::pstderr);

			string line;
			while (std::getline(proc.out(), line))
				cout << line << endl;
			while (std::getline(proc.err(), line))
				cerr << line << endl;
		}

		// If the output file does not exist, there must be some
		// fatal error. However, we can still continue in fallback
		// mode by executing the generic kernel.
		struct stat buffer;
		if (stat(tmp.filename.c_str(), &buffer))
		{
			cerr << "Deferred GPU kernel compilation failed!" << endl;

			kernel.compilationFailed = true;
			kernel.dim = dim;
			kernel.fileowner = false;
			kernel.func = fallbackFunc;

			__sync_synchronize();

			kernels_tls->operator[](dim) = kernel;
			PTHREAD_ERR_CHECK(pthread_mutex_unlock(&mutex));
			return kernel;
		}

		cout << "JIT-compiled GPU kernel for dim = " << dim << endl;

		kernel.dim = dim;
		kernel.filename = tmp.filename;
		kernel.fileowner = true;
		kernel.funcname = funcname;
	
		__sync_synchronize();

		// Send filename to everyone.
		static vector<MPI_Request> vrequests;
		vrequests.resize(vrequests.size() + process->getSize() * 2);
		MPI_Request* requests = &vrequests[vrequests.size() - 1 - process->getSize() * 2];
		for (int i = 0, e = process->getSize(); i != e; i++)
		{
			if (i == process->getRoot()) continue;

			int length = tmp.filename.length();
			MPI_ERR_CHECK(MPI_Isend(&length,
				1, MPI_INT, i, ((int)(size_t)&JIT::jitCompile<K, F>) % 32767, MPI_COMM_WORLD, &requests[2 * i]));
			MPI_ERR_CHECK(MPI_Isend((void*)tmp.filename.c_str(), tmp.filename.length(),
				MPI_BYTE, i, ((int)(size_t)&JIT::jitCompile<K, F> + 1) % 32767, MPI_COMM_WORLD, &requests[2 * i + 1]));
		}
		/*for (int i = 0, e = requests.size(); i != e; i++)
		{
			if (i == 2 * process->getRoot()) continue;
			if (i == 2 * process->getRoot() + 1) continue;
			MPI_ERR_CHECK(MPI_Wait(&requests[i], MPI_STATUS_IGNORE));
		}*/
	}
	else
	{
		// Generate function name for specific number of arguments.
		stringstream sfuncname;
		sfuncname << funcnameTemplate;
		sfuncname << dim;
		string funcname = sfuncname.str();

		// Receive filename from master.
		int length = 0;
		MPI_ERR_CHECK(MPI_Recv(&length, 1, MPI_INT,
			process->getRoot(), ((int)(size_t)&JIT::jitCompile<K, F>) % 32767, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
		vector<char> buffer;
		buffer.resize(length);
		MPI_ERR_CHECK(MPI_Recv(&buffer[0], length, MPI_BYTE,
			process->getRoot(), ((int)(size_t)&JIT::jitCompile<K, F> + 1) % 32767, MPI_COMM_WORLD, MPI_STATUS_IGNORE));

		kernel.dim = dim;
		kernel.filename = string(&buffer[0], buffer.size());
		kernel.fileowner = false;
		kernel.funcname = funcname;
		
		__sync_synchronize();
	}

	kernel.func = kernel.getFunc();
	kernels_tls->operator[](dim) = kernel;
	PTHREAD_ERR_CHECK(pthread_mutex_unlock(&mutex));
	return kernel;
}

InterpolateValueKernel& JIT::jitCompile(
	int dim, const string& funcnameTemplate, InterpolateValueFunc fallbackFunc)
{
	return JIT::jitCompile<InterpolateValueKernel, InterpolateValueFunc>(
		dim, funcnameTemplate, fallbackFunc);
}

InterpolateArrayKernel& JIT::jitCompile(
	int dim, const string& funcnameTemplate, InterpolateArrayFunc fallbackFunc)
{
	return JIT::jitCompile<InterpolateArrayKernel, InterpolateArrayFunc>(
		dim, funcnameTemplate, fallbackFunc);
}

InterpolateArrayManyStatelessKernel& JIT::jitCompile(
	int dim, const string& funcnameTemplate, InterpolateArrayManyStatelessFunc fallbackFunc)
{
	return JIT::jitCompile<InterpolateArrayManyStatelessKernel, InterpolateArrayManyStatelessFunc>(
		dim, funcnameTemplate, fallbackFunc);
}

InterpolateArrayManyMultistateKernel& JIT::jitCompile(
	int dim, const string& funcnameTemplate, InterpolateArrayManyMultistateFunc fallbackFunc)
{
	return JIT::jitCompile<InterpolateArrayManyMultistateKernel, InterpolateArrayManyMultistateFunc>(
		dim, funcnameTemplate, fallbackFunc);
}

#endif // HAVE_RUNTIME_OPTIMIZATION

