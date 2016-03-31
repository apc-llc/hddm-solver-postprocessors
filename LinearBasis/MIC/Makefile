#
# Build script for Linux workstation and
# Anselm cluster http://www.it4i.cz/
#

LOCATION = ../../liboffloadmic

INSTALL_PREFIX = ../../install

MIC_PREFIX = k1om-mpss-linux-

include $(LOCATION)/Makefile.inc

all: test_InterpolateValue

ifeq (native,$(target))
$(LOCATION)/tbb/build/linux_release/libtbb.so:
	$(LOAD_ICC_MODULE) && cd $(LOCATION)/tbb/ && make target=mic compiler=icc -j12 && cd build && ln -sf mic_icc_release linux_release
else
$(LOCATION)/tbb/build/linux_release/libtbb.so:
	$(LOAD_GCC_MODULE) && cd $(LOCATION)/tbb/ && make -j12 && cd build && ln -sf `find . -name "linux_intel64_gcc_*_release"` linux_release
endif

test_InterpolateValue: test_InterpolateValue.o InterpolateValueCPU.o InterpolateValue_embed.o
	$(LOAD_GCC_MODULE) && $(HOST_CC) $(HOST_CFLAGS) $^ -o $@ -L$(INSTALL_PREFIX)/host/lib/$(TOOLEXECLIBDIR) -lmicrt -Wl,-rpath=$(INSTALL_PREFIX)/host/lib/$(TOOLEXECLIBDIR)

test_InterpolateValue.o: test_InterpolateValue.c $(INSTALL_PREFIX)/host/include/mic_runtime.h
	$(LOAD_GCC_MODULE) && $(HOST_CC) $(HOST_CFLAGS) -std=c99 -I$(INSTALL_PREFIX)/host/include -c $< -o $@

InterpolateValueCPU.o: ../CPU/InterpolateValue.c
	$(HOST_CC) $(HOST_CFLAGS) -I.. -mavx -std=c99 -DMODE=Generic -DDIM=8 -DDEFERRED -DHAVE_AVX -c $< -o $@

InterpolateValue_embed.o: InterpolateValue_embed.c InterpolateValue_embed.h
	$(LOAD_GCC_MODULE) && $(HOST_CC) $(HOST_CFLAGS) -DFUNCNAME=InterpolateValueMIC -I. -I$(INSTALL_PREFIX)/host/include -c $< -o $@

InterpolateValue_embed.h: libInterpolateValueMIC.so
	xxd -include $< >$@

libInterpolateValueMIC.so: InterpolateValue.cpp $(LOCATION)/tbb/build/linux_release/libtbb.so
	$(LOAD_GCC_MODULE) && $(MIC_CXX) $(MIC_CXXFLAGS) -I.. -mavx512f -DMODE=Generic -DDIM=8 -DDEFERRED -DHAVE_MIC -fPIC -shared $< -o $@ -I$(LOCATION)/tbb/include/

clean:
	rm -rf test_InterpolateValue libInterpolateValueMIC.so InterpolateValue_embed.h *.o

