#include "JIT.h"

#include <functional>
#include <map>
#include <pthread.h>
#include <exec-stream.h>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <vector>
#include <unistd.h>

using namespace NAMESPACE;
using namespace std;

namespace NAMESPACE {

class Device
{
public :

	int getCC() const;
};

} // namespace NAMESPACE

static int id = 1; // TODO

extern unsigned int INTERPOLATE_ARRAY_SH_LEN;
extern unsigned int INTERPOLATE_ARRAY_MANY_MULTISTATE_SH_LEN;

extern unsigned char INTERPOLATE_ARRAY_SH[];
extern unsigned char INTERPOLATE_ARRAY_MANY_MULTISTATE_SH[];

template<>
const string InterpolateArrayKernel::sh((char*)INTERPOLATE_ARRAY_SH, INTERPOLATE_ARRAY_SH_LEN);
template<>
const string InterpolateArrayManyMultistateKernel::sh((char*)INTERPOLATE_ARRAY_MANY_MULTISTATE_SH, INTERPOLATE_ARRAY_MANY_MULTISTATE_SH_LEN);

struct KSignature
{
	int dim;
	int count;
	int DofPerNode;
	
	KSignature() : 
	
	dim(0), count(0), DofPerNode(0)
	
	{ }
	
	KSignature(int dim_, int count_, int DofPerNode_) :
	
	dim(dim_), count(count_), DofPerNode(DofPerNode_)
	
	{ }
	
	template<class T>
	static inline void hashCombine(size_t& seed, const T& v)
	{
		std::hash<T> hasher;
		seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	}

	size_t hash() const
	{
		size_t seed = 0;
		hashCombine(seed, dim);
		hashCombine(seed, count);
		hashCombine(seed, DofPerNode);
		return seed;
	}

	bool operator()(const KSignature& a, const KSignature& b) const
	{
		return a.hash() < b.hash();
	}
};

template<typename K, typename F>
K& JIT::jitCompile(Device* device, int dim, int count, int DofPerNode,
	const string& funcnameTemplate, F fallbackFunc)
{
	map<KSignature, K, KSignature>* kernels_tls_ = NULL;

	// Already in TLS cache?
	{
		static __thread bool kernels_init = false;
		static __thread char kernels_a[sizeof(map<KSignature, K, KSignature>)];
		kernels_tls_ = (map<KSignature, K, KSignature>*)kernels_a;

		if (!kernels_init)
		{
			kernels_tls_ = new (kernels_a) map<KSignature, K, KSignature>();
			kernels_init = true;
		}
	}

	map<KSignature, K, KSignature>& kernels_tls = *kernels_tls_;

	KSignature signature(dim, count, DofPerNode);

	{
		K& kernel = kernels_tls[signature];

		// Already successfully compiled?
		if (kernel.filename != "")
			return kernel;

		// Already unsuccessfully compiled?
		if (kernel.compilationFailed)
		{
			kernel.dim = dim;
			kernel.count = count;
			kernel.DofPerNode = DofPerNode;
			kernel.fileowner = false;
			kernel.func = fallbackFunc;

			return kernel;
		}
	}

	// Already in process cache?
	static map<KSignature, K, KSignature> kernels;

	static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

	PTHREAD_ERR_CHECK(pthread_mutex_lock(&mutex));

	K& kernel = kernels[signature];

	// Already successfully compiled?
	if (kernel.filename != "")
	{
		kernels_tls[signature] = kernel;
		PTHREAD_ERR_CHECK(pthread_mutex_unlock(&mutex));
		return kernel;
	}

	// Already unsuccessfully compiled?
	if (kernel.compilationFailed)
	{
		kernel.dim = dim;
		kernel.count = count;
		kernel.DofPerNode = DofPerNode;
		kernel.fileowner = false;
		kernel.func = fallbackFunc;

		kernels_tls[signature] = kernel;
		PTHREAD_ERR_CHECK(pthread_mutex_unlock(&mutex));
		return kernel;
	}

	// Generate function name for specific number of arguments.
	int length = snprintf(NULL, 0, "%s%zu", funcnameTemplate.c_str(), signature.hash());
	string funcname;
	funcname.resize(length + 1);
	sprintf(&funcname[0], "%s%zu", funcnameTemplate.c_str(), signature.hash());

	MPI_Process* process;
	MPI_ERR_CHECK(MPI_Process_get(&process));
	if (process->isMaster())
	{
		process->cout("Performing deferred GPU kernel compilation for dim = %d, count = %d, DofPerNode = %d ...\n",
			dim, count, DofPerNode);

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
			process->cerr("Deferred GPU kernel temp file creation failed!\n");

			kernel.compilationFailed = true;
			kernel.dim = dim;
			kernel.count = count;
			kernel.DofPerNode = DofPerNode;
			kernel.fileowner = false;
			kernel.func = fallbackFunc;

			kernels_tls[signature] = kernel;
			PTHREAD_ERR_CHECK(pthread_mutex_unlock(&mutex));
			return kernel;
		}

		// Read the compile command template.
		vector<char> cmd;
		{
			string sh = kernel.sh.c_str();
			size_t length = sh.size();

			// Remove newlines.
			for (long i = 0; i < length; i++)
				if (sh[i] == '\n') sh[i] = ' ';
			
			// Replace all double quotes with sigle quotes.
			for (long i = 0; i < length; i++)
				if (sh[i] == '"') sh[i] = '\'';			

			// Map X vector onto formal parameter constant memory space, if smaller than
			// DEVICE_CONST_X_MAX_SIZE, and the size is runtime-constant.
			bool useConstMemoryForX = false;
			if (dim * count * sizeof(double) < DEVICE_CONST_X_MAX_SIZE)
				useConstMemoryForX = true;

			// Add option for including line-number information for device code in release mode only
			// - debug more already implies it is enabled (compiler warning).
			const char* format = "-c \"%s -arch=sm_%d %s -DFUNCNAME=%s -DDIM=%d "
				"-DCOUNT=%d -DDOF_PER_NODE=%d -o %s %s"
#if defined(NDEBUG)
				" -lineinfo"
#endif
				" 2>&1\"";

			bool keepCache = false;
			const char* keepCacheValue = getenv("KEEP_CACHE");
			if (keepCacheValue)
				keepCache = atoi(keepCacheValue);

			// Add arch specification based on CC of the given device.
			int cc = device->getCC();

			size_t szcmd = snprintf(NULL, 0, format,
				sh.c_str(), cc, useConstMemoryForX ? "-DX_IN_CONSTANT_MEMORY" : "",
				funcname.c_str(), dim, count, DofPerNode, tmp.filename.c_str(),
				keepCache ? "-keep" : "");

			cmd.resize(szcmd + 2);
			cmd[szcmd + 1] = '\0';

			snprintf(&cmd[0], szcmd + 1, format,
				sh.c_str(), cc, useConstMemoryForX ? "-DX_IN_CONSTANT_MEMORY" : "",
				funcname.c_str(), dim, count, DofPerNode, tmp.filename.c_str(),
				keepCache ? "-keep" : "");

			if (keepCache)
				printf("cmd = %s\n", &cmd[0]);
		}

		// Run compiler as a process and create a streambuf that
		// reads its stdout and stderr.
		{
			exec_stream_t es;

			try
			{
				// Start command and wait for it infinitely.
				es.set_wait_timeout(exec_stream_t::s_all, (unsigned long)-1);
				es.start("sh", (string)&cmd[0]);

				int length = 0;
				vector<char> buffer(512 + 1);
				while (length = es.out().rdbuf()->sgetn(&buffer[0], 512))
				{
					buffer[length] = '\0';
					process->cout("%s", &buffer[0]);
				}
			}
			catch (exception const & e)
			{
				const string& out = e.what();
				process->cerr("%s", out.c_str());
			}
		}

		// If the output file does not exist, there must be some
		// fatal error. However, we can still continue in fallback
		// mode by executing the generic kernel.
		struct stat buffer;
		if (stat(tmp.filename.c_str(), &buffer))
		{
			process->cerr("Deferred GPU kernel compilation failed!\n");

			kernel.compilationFailed = true;
			kernel.dim = dim;
			kernel.count = count;
			kernel.DofPerNode = DofPerNode;
			kernel.fileowner = false;
			kernel.func = fallbackFunc;

			kernels_tls[signature] = kernel;
		}
		else
		{
			process->cout("JIT-compiled GPU kernel for dim = %d, count = %d, DofPerNode = %d\n",
				dim, count, DofPerNode);

			kernel.dim = dim;
			kernel.count = count;
			kernel.DofPerNode = DofPerNode;
			kernel.filename = tmp.filename;
			kernel.fileowner = true;
			kernel.funcname = funcname;
		}

		// Convert filename to char array.
		vector<char> vfilename;
		if (!kernel.compilationFailed)
		{
			// Keep zero filename vector size, if runtime compilation has failed.
			vfilename.resize(tmp.filename.length() + 1);
			memcpy(&vfilename[0], tmp.filename.c_str(), vfilename.size());
		}

		// Send filename to everyone.
		vector<MPI_Request> vrequests;
		vrequests.resize(process->getSize() * 2);
		MPI_Request* requests = &vrequests[0];
		for (int i = 0, e = process->getSize(); i != e; i++)
		{
			if (i == process->getRoot()) continue;

			int length = vfilename.size();
			MPI_ERR_CHECK(MPI_Isend(&length, 1, MPI_INT, i, id, process->getComm(), &requests[2 * i]));
			MPI_ERR_CHECK(MPI_Isend(&vfilename[0], length, MPI_BYTE, i, 2 * id, process->getComm(), &requests[2 * i + 1]));
		}
		for (int i = 0, e = vrequests.size(); i != e; i++)
		{
			if (i == 2 * process->getRoot()) continue;
			if (i == 2 * process->getRoot() + 1) continue;
			MPI_ERR_CHECK(MPI_Wait(&requests[i], MPI_STATUS_IGNORE));
		}
	}
	else
	{
		// Receive filename from master.
		int length = 0;
		MPI_ERR_CHECK(MPI_Recv(&length, 1, MPI_INT,
			process->getRoot(), id, process->getComm(), MPI_STATUS_IGNORE));
		vector<char> vfilename;
		vfilename.resize(length);
		MPI_ERR_CHECK(MPI_Recv(&vfilename[0], length, MPI_BYTE,
			process->getRoot(), 2 * id, process->getComm(), MPI_STATUS_IGNORE));

		// Zero filename is an indication of failed runtime compilation.
		if (vfilename.size())
		{
			kernel.dim = dim;
			kernel.count = count;
			kernel.DofPerNode = DofPerNode;
			kernel.filename = string(&vfilename[0], vfilename.size());
			kernel.fileowner = false;
			kernel.funcname = funcname;
		}
		else
		{
			kernel.compilationFailed = true;
			kernel.dim = dim;
			kernel.count = count;
			kernel.DofPerNode = DofPerNode;
			kernel.fileowner = false;
			kernel.func = fallbackFunc;

			kernels_tls[signature] = kernel;
		}
	}

	if (!kernel.compilationFailed)
	{
		kernel.func = kernel.getFunc();
		kernels_tls[signature] = kernel;
	}

	PTHREAD_ERR_CHECK(pthread_mutex_unlock(&mutex));
	return kernel;
}

InterpolateArrayKernel& JIT::jitCompile(
	Device* device, int dim, int DofPerNode,
	const string& funcnameTemplate, InterpolateArrayFunc fallbackFunc)
{
	return JIT::jitCompile<InterpolateArrayKernel, InterpolateArrayFunc>(
		device, dim, 1, DofPerNode, funcnameTemplate, fallbackFunc);
}

InterpolateArrayManyMultistateKernel& JIT::jitCompile(
	Device* device, int dim, int count, int DofPerNode,
	const string& funcnameTemplate, InterpolateArrayManyMultistateFunc fallbackFunc)
{
	return JIT::jitCompile<InterpolateArrayManyMultistateKernel, InterpolateArrayManyMultistateFunc>(
		device, dim, count, DofPerNode, funcnameTemplate, fallbackFunc);
}
