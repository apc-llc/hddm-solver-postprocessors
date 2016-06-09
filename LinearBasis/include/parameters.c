#ifdef REFERENCES
#define REF &
#else
#define REF
#endif

int REF priority;                    // interpolator's priority

int REF nagents;                     // number of agents in economy
	
bool REF binaryio;                   // input/output surplus files as binary or text

bool REF enableRuntimeOptimization;  // enable the use of runtime code optimization (and if
                                     // it fails - fallback to regular code)

// XXX Add new parameters here

#undef REF

