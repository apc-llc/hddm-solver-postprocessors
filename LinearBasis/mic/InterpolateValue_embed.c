#include <InterpolateValue_embed.h>
#include <mic_runtime.h>

__attribute__((constructor)) static void libInterpolateValueMIC_register(void)
{
	micRegisterModule(libInterpolateValueMIC_so, libInterpolateValueMIC_so_len);
}

