#ifdef HAVE_RUNTIME_OPTIMIZATION

#include "JIT.h"

#include <functional>
#include <fstream>
#include <iostream>
#include <map>
#include <pthread.h>
#include <pstreams/pstream.h>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <vector>
#include <unistd.h>

using namespace cpu;
using namespace std;

template<>
const string InterpolateArrayKernel::sh = INTERPOLATE_ARRAY_SH;
template<>
const string InterpolateArrayManyMultistateKernel::sh = INTERPOLATE_ARRAY_MANY_MULTISTATE_SH;

struct KSignature
{
	int dim;
	int count;
	int nno;
	int Dof_choice_start;
	int Dof_choice_end;
	
	KSignature() : 
	
	dim(0), count(0), nno(0),
	Dof_choice_start(0), Dof_choice_end(0)
	
	{ }
	
	KSignature(int dim_, int count_, int nno_,
		int Dof_choice_start_, int Dof_choice_end_) :
	
	dim(dim_), count(count_), nno(nno_),
	Dof_choice_start(Dof_choice_start_), Dof_choice_end(Dof_choice_end_)
	
	{ }
	
	bool operator()(const KSignature& a, const KSignature& b) const;
};

namespace std {

template<>
struct hash<KSignature>
{
	template <class T>
	static inline void hash_combine(size_t& seed, const T& v)
	{
		hash<T> hasher;
		seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	}

	typedef KSignature argument_type;
	typedef std::size_t result_type;
	result_type operator()(argument_type const& k) const
	{
		size_t seed = 0;
		hash_combine(seed, k.dim);
		hash_combine(seed, k.count);
		hash_combine(seed, k.nno);
		hash_combine(seed, k.Dof_choice_start);
		hash_combine(seed, k.Dof_choice_end);
		return seed;
	}
};

} // namespace std

bool KSignature::operator()(const KSignature& a, const KSignature& b) const
{
	return hash<KSignature>{}(a) < hash<KSignature>{}(b);
}

template<typename K, typename F>
K& JIT::jitCompile(int dim, int count, int nno, int Dof_choice_start, int Dof_choice_end,
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

	{
		K& kernel = kernels_tls[KSignature(dim, count, nno, Dof_choice_start, Dof_choice_end)];

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

	K& kernel = kernels[KSignature(dim, count, nno, Dof_choice_start, Dof_choice_end)];

	// Already successfully compiled?
	if (kernel.filename != "")
	{
		kernels_tls[KSignature(dim, count, nno, Dof_choice_start, Dof_choice_end)] = kernel;
		PTHREAD_ERR_CHECK(pthread_mutex_unlock(&mutex));
		return kernel;
	}

	// Already unsuccessfully compiled?
	if (kernel.compilationFailed)
	{
		kernel.dim = dim;
		kernel.fileowner = false;
		kernel.func = fallbackFunc;

		kernels_tls[KSignature(dim, count, nno, Dof_choice_start, Dof_choice_end)] = kernel;
		PTHREAD_ERR_CHECK(pthread_mutex_unlock(&mutex));
		return kernel;
	}
	
	MPI_Process* process;
	MPI_ERR_CHECK(MPI_Process_get(&process));
	if (process->isMaster())
	{
		cout << "Performing deferred CPU kernel compilation for dim = " << dim << " ..." << endl;

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
			cerr << "Deferred CPU kernel temp file creation failed!" << endl;

			kernel.compilationFailed = true;
			kernel.dim = dim;
			kernel.fileowner = false;
			kernel.func = fallbackFunc;

			kernels_tls[KSignature(dim, count, nno, Dof_choice_start, Dof_choice_end)] = kernel;
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
			if (!t.is_open())
			{
				cerr << "Error opening file: " << kernel.sh << endl;
				cerr << "Deferred CPU kernel compilation failed!" << endl;

				kernel.compilationFailed = true;
				kernel.dim = dim;
				kernel.fileowner = false;
				kernel.func = fallbackFunc;

				kernels_tls[KSignature(dim, count, nno, Dof_choice_start, Dof_choice_end)] = kernel;
				PTHREAD_ERR_CHECK(pthread_mutex_unlock(&mutex));
				return kernel;
			}
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
			cmd << " -DCOUNT=";
			cmd << count;
			cmd << " -DNNO=";
			cmd << nno;
			cmd << " -DVDIM8=";
			cmd << vdim8;
			cmd << " -DDOF_CHOICE_START=";
			cmd << Dof_choice_start;
			cmd << " -DDOF_CHOICE_END=";
			cmd << Dof_choice_end;
	
			cmd << " -o ";
			cmd << tmp.filename;
		}

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
			cerr << "Deferred CPU kernel compilation failed!" << endl;

			kernel.compilationFailed = true;
			kernel.dim = dim;
			kernel.fileowner = false;
			kernel.func = fallbackFunc;

			kernels_tls[KSignature(dim, count, nno, Dof_choice_start, Dof_choice_end)] = kernel;
			PTHREAD_ERR_CHECK(pthread_mutex_unlock(&mutex));
			return kernel;
		}

		cout << "JIT-compiled CPU kernel for dim = " << dim << endl;

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
		// Generate function name for specific number of arguments.
		stringstream sfuncname;
		sfuncname << funcnameTemplate;
		sfuncname << dim;
		string funcname = sfuncname.str();

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
	kernels_tls[KSignature(dim, count, nno, Dof_choice_start, Dof_choice_end)] = kernel;
	PTHREAD_ERR_CHECK(pthread_mutex_unlock(&mutex));
	return kernel;
}

InterpolateArrayKernel& JIT::jitCompile(
	int dim, int nno, int Dof_choice_start, int Dof_choice_end,
	const string& funcnameTemplate, InterpolateArrayFunc fallbackFunc)
{
	return JIT::jitCompile<InterpolateArrayKernel, InterpolateArrayFunc>(
		dim, 1, nno, Dof_choice_start, Dof_choice_end, funcnameTemplate, fallbackFunc);
}

InterpolateArrayManyMultistateKernel& JIT::jitCompile(
	int dim, int count, int nno, int Dof_choice_start, int Dof_choice_end,
	const string& funcnameTemplate, InterpolateArrayManyMultistateFunc fallbackFunc)
{
	return JIT::jitCompile<InterpolateArrayManyMultistateKernel, InterpolateArrayManyMultistateFunc>(
		dim, count, nno, Dof_choice_start, Dof_choice_end, funcnameTemplate, fallbackFunc);
}

#endif // HAVE_RUNTIME_OPTIMIZATION

