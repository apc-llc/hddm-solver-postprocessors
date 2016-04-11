#include <InterpolateValue_embed.h>
#include <mic_runtime.h>

__attribute__((constructor)) static void libInterpolateValue_register(void)
{
	micRegisterModule(libInterpolateValue_so, libInterpolateValue_so_len);
}

