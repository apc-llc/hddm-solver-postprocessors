#ifdef REFERENCES
#define REF &
#else
#define REF
#endif

int REF priority;                    // interpolator's priority

int REF nagents;                     // number of agents in economy
	
double REF surplusCutoff;            // small surplus values cutoff threshold

bool REF surplusCutoffDefined;       // set if surplus values cutoff threshold has been defined

// XXX Add new parameters here

#undef REF

