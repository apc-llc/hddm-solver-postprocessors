#include <check.h>
#include <cstdio>
#include <dlfcn.h>
#include <process.h>
#include <string>
#include <unistd.h>

namespace NAMESPACE {

class JIT;

template<typename T>
class InterpolateKernel
{
	int dim, count, nno, DofPerNode;
	bool compilationFailed;
	T func;
	static const std::string sh;
	std::string filename;
	bool fileowner;
	std::string funcname;
	
	pthread_mutex_t mutex;

public :
	const T& getFunc()
	{
		using namespace std;
	
		if (compilationFailed) return func;

		if (!func)
		{
			PTHREAD_ERR_CHECK(pthread_mutex_lock(&mutex));
			if (!func)
			{
				MPI_Process* process;
				MPI_ERR_CHECK(MPI_Process_get(&process));

				string type = "CPU";
			
				// Open compiled library and load interpolation function entry point.
#ifndef RTLD_DEEPBIND
				void* handle = dlopen(filename.c_str(), RTLD_NOW | RTLD_GLOBAL);
#else
				void* handle = dlopen(filename.c_str(), RTLD_NOW | RTLD_GLOBAL  | RTLD_DEEPBIND);
#endif
				if (!handle)
				{
					process->cerr("Deferred %s kernel loading failed: %s\n", type.c_str(), dlerror());
					compilationFailed = true;
					PTHREAD_ERR_CHECK(pthread_mutex_unlock(&mutex));
					return func;
				}
				func = (T)dlsym(handle, funcname.c_str());
				if (!func)
				{
					process->cerr("Deferred %s kernel loading failed: %s\n", type.c_str(), dlerror());
					compilationFailed = true;
					PTHREAD_ERR_CHECK(pthread_mutex_unlock(&mutex));
					return func;
				}

				process->cout("Rank #%d loaded %s kernel for dim = %d, count = %d, nno = %d, DofPerNode = %d : %s @ 0x%x\n",
					process->getRank(), type.c_str(), dim, count, nno, DofPerNode, funcname.c_str(), (void*)func);
			}
			PTHREAD_ERR_CHECK(pthread_mutex_unlock(&mutex));
		}

		return func;
	}

	InterpolateKernel() :
		dim(-1), compilationFailed(false), func(NULL), filename(""), fileowner(false)
	{
		pthread_mutex_init(&mutex, NULL); 
	}
	
	InterpolateKernel(const InterpolateKernel& other)
	{
		dim = other.dim;
		compilationFailed = other.compilationFailed;
		func = other.func;
		filename = other.filename;
		fileowner = false;
		pthread_mutex_init(&mutex, NULL);
	}

	~InterpolateKernel()
	{
		if (fileowner && (filename != ""))
		{
			bool keepCache = false;
			const char* keepCacheValue = getenv("KEEP_CACHE");
			if (keepCacheValue)
				keepCache = atoi(keepCacheValue);
			if (!keepCache)
				unlink(filename.c_str());
		}
	}
	
	friend class JIT;
};

} // namespace NAMESPACE

