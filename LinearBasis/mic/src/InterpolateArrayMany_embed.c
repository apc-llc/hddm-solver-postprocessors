#include <InterpolateArrayMany_embed.h>
#include <mic_runtime.h>

__attribute__((constructor)) static void libInterpolateArrayMany_register(void)
{
	micRegisterModule(libInterpolateArrayMany_so, libInterpolateArrayMany_so_len);
}

