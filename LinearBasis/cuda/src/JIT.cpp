#include "JIT.h"

#include <functional>
#include <map>
#include <pthread.h>
#include <pstreams/pstream.h>
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

template<>
const string InterpolateArrayKernel::sh = INTERPOLATE_ARRAY_SH;
template<>
const string InterpolateArrayManyMultistateKernel::sh = INTERPOLATE_ARRAY_MANY_MULTISTATE_SH;

struct KSignature
{
	int dim;
	int count;
	int nno;
	int DofPerNode;
	
	KSignature() : 
	
	dim(0), count(0), nno(0), DofPerNode(0)
	
	{ }
	
	KSignature(int dim_, int count_, int nno_, int DofPerNode_) :
	
	dim(dim_), count(count_), nno(nno_), DofPerNode(DofPerNode_)
	
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
		hashCombine(seed, nno);
		hashCombine(seed, DofPerNode);
		return seed;
	}

	bool operator()(const KSignature& a, const KSignature& b) const
	{
		return a.hash() < b.hash();
	}
};

template<typename K, typename F>
K& JIT::jitCompile(Device* device, int dim, int count, int nno, int DofPerNode,
	const string& funcnameTemplate, F fallbackFunc)
{
	int vdim = dim / AVX_VECTOR_SIZE;
	if (dim % AVX_VECTOR_SIZE) vdim++;
	vdim *= AVX_VECTOR_SIZE;
	int vdim8 = vdim / AVX_VECTOR_SIZE;

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

	KSignature signature(dim, count, nno, DofPerNode);

	{
		K& kernel = kernels_tls[signature];

		// Already successfully compiled?
		if (kernel.filename != "")
			return kernel;

		// Already unsuccessfully compiled?
		if (kernel.compilationFailed)
		{
			kernel.dim = dim;
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
		process->cout("Performing deferred GPU kernel compilation for dim = %d ...\n", dim);

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
			kernel.fileowner = false;
			kernel.func = fallbackFunc;

			kernels_tls[signature] = kernel;
			PTHREAD_ERR_CHECK(pthread_mutex_unlock(&mutex));
			return kernel;
		}

		// Read the compile command template.
		vector<char> cmd;
		{
			FILE* fsh = fopen(kernel.sh.c_str(), "rb");
			if (!fsh)
			{
				process->cerr("Error opening file: %s\n", kernel.sh.c_str());
				process->cerr("Deferred GPU kernel compilation failed!\n");

				kernel.compilationFailed = true;
				kernel.dim = dim;
				kernel.fileowner = false;
				kernel.func = fallbackFunc;

				kernels_tls[signature] = kernel;
				PTHREAD_ERR_CHECK(pthread_mutex_unlock(&mutex));
				return kernel;
			}

			vector<char> sh;
			fseek(fsh, 0, SEEK_END);
			long length = ftell(fsh);
			fseek(fsh, 0, SEEK_SET);
			sh.resize(length + 1);
			sh[length] = '\0';
			length = fread(&sh[0], 1, length, fsh);
			fclose(fsh);

			// Remove newlines.
			for (long i = 0; i < length; i++)
				if (sh[i] == '\n') sh[i] = ' ';

			const char* format = "%s -arch=sm_%d -lineinfo -DDEFERRED -DFUNCNAME=%s -DDIM=%d "
				"-DCOUNT=%d -DNNO=%d -DVDIM8=%d -DDOF_PER_NODE=%d -o %s";

			bool keepCache = false;
			const char* keepCacheValue = getenv("KEEP_CACHE");
			if (keepCacheValue)
				keepCache = atoi(keepCacheValue);
			if (keepCache)
				format = "%s -arch=sm_%d -lineinfo -DDEFERRED -DFUNCNAME=%s -DDIM=%d "
					"-DCOUNT=%d -DNNO=%d -DVDIM8=%d -DDOF_PER_NODE=%d -o %s -keep";

			// Add arch specification based on CC of the given device.
			int cc = device->getCC();

			size_t szcmd = snprintf(NULL, 0, format,
				&sh[0], cc, funcname.c_str(), dim, count, nno, vdim8, DofPerNode, tmp.filename.c_str());

			cmd.resize(szcmd + 2);
			cmd[szcmd + 1] = '\0';

			snprintf(&cmd[0], szcmd + 1, format,
				&sh[0], cc, funcname.c_str(), dim, count, nno, vdim8, DofPerNode, tmp.filename.c_str());

			if (keepCache)
				printf("cmd = %s\n", &cmd[0]);
		}

		// Run compiler as a process and create a streambuf that
		// reads its stdout and stderr.
		{
			redi::ipstream proc((string)&cmd[0], redi::pstreams::pstderr);

			string line;
			while (std::getline(proc.out(), line))
				process->cout("%s\n", line.c_str());
			while (std::getline(proc.err(), line))
				process->cout("%s\n", line.c_str());
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
			kernel.fileowner = false;
			kernel.func = fallbackFunc;

			kernels_tls[signature] = kernel;
			PTHREAD_ERR_CHECK(pthread_mutex_unlock(&mutex));
			return kernel;
		}

		process->cout("JIT-compiled GPU kernel for dim = %d\n", dim);

		kernel.dim = dim;
		kernel.filename = tmp.filename;
		kernel.fileowner = true;
		kernel.funcname = funcname;

		// Convert filename to char array.
		vector<char> vfilename;
		vfilename.resize(tmp.filename.length() + 1);
		memcpy(&vfilename[0], tmp.filename.c_str(), vfilename.size());

		// Send filename to everyone.
		vector<MPI_Request> vrequests;
		vrequests.resize(process->getSize() * 2);
		MPI_Request* requests = &vrequests[0];
		for (int i = 0, e = process->getSize(); i != e; i++)
		{
			if (i == process->getRoot()) continue;

			int length = vfilename.size();
			MPI_ERR_CHECK(MPI_Isend(&length, 1, MPI_INT, i, i, process->getComm(), &requests[2 * i]));
			MPI_ERR_CHECK(MPI_Isend(&vfilename[0], length, MPI_BYTE, i, i + e, process->getComm(), &requests[2 * i + 1]));
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
			process->getRoot(), process->getRank(), process->getComm(), MPI_STATUS_IGNORE));
		vector<char> vfilename;
		vfilename.resize(length);
		MPI_ERR_CHECK(MPI_Recv(&vfilename[0], length, MPI_BYTE,
			process->getRoot(), process->getRank() + process->getSize(), process->getComm(), MPI_STATUS_IGNORE));

		kernel.dim = dim;
		kernel.filename = string(&vfilename[0], vfilename.size());
		kernel.fileowner = false;
		kernel.funcname = funcname;
	}

	kernel.func = kernel.getFunc();
	kernels_tls[signature] = kernel;
	PTHREAD_ERR_CHECK(pthread_mutex_unlock(&mutex));
	return kernel;
}

InterpolateArrayKernel& JIT::jitCompile(
	Device* device, int dim, int nno, int DofPerNode,
	const string& funcnameTemplate, InterpolateArrayFunc fallbackFunc)
{
	return JIT::jitCompile<InterpolateArrayKernel, InterpolateArrayFunc>(
		device, dim, 1, nno, DofPerNode, funcnameTemplate, fallbackFunc);
}

InterpolateArrayManyMultistateKernel& JIT::jitCompile(
	Device* device, int dim, int count, int nno, int DofPerNode,
	const string& funcnameTemplate, InterpolateArrayManyMultistateFunc fallbackFunc)
{
	return JIT::jitCompile<InterpolateArrayManyMultistateKernel, InterpolateArrayManyMultistateFunc>(
		device, dim, count, nno, DofPerNode, funcnameTemplate, fallbackFunc);
}
