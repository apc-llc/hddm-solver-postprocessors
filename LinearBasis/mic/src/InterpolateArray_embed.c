#include <InterpolateArray_embed.h>
#include <mic_runtime.h>

__attribute__((constructor)) static void libInterpolateArray_register(void)
{
	micRegisterModule(libInterpolateArray_so, libInterpolateArray_so_len);
}

