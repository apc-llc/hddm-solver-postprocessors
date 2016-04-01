#
# Build script for Linux workstation and
# Anselm cluster http://www.it4i.cz/
#

LOCATION = ../../

INSTALL_PREFIX = ../../install

MIC_PREFIX = k1om-mpss-linux-

include $(LOCATION)/Makefile.inc

MPSS_VER := 3.4.2
GCC_EXTRA_CONFIG :=
GCC_CFLAGS :=
GCC_CXXFLAGS :=
ifneq (,$(ANSELM))
MPSS_VER := 3.4.3
GMP_PATH := /apps/libs/gmp/gcc/5.1.2/
MPFR_PATH := /apps/libs/mpfr/gcc/3.1.2/
MPC_PATH := /apps/libs/mpc/gcc/1.0.1/
ANSELM_LIBS := -Wl,-rpath=$(GMP_PATH) -Wl,-rpath=$(MPFR_PATH) -Wl,-rpath=$(MPC_PATH)
GCC_CFLAGS := CFLAGS="-g -O2 $(ANSELM_LIBS)"
GCC_CXXFLAGS := CXXFLAGS="-g -O2 $(ANSELM_LIBS)"
GCC_EXTRA_CONFIG := --with-gmp=$(GMP_PATH) --with-mpfr=$(MPFR_PATH) --with-mpc=$(MPC_PATH) $(GCC_CFLAGS) $(GCC_CXXFLAGS)
endif

all: test_InterpolateValue

ifeq (native,$(target))
$(LOCATION)/tbb/build/linux_release/libtbb.so:
	$(LOAD_ICC_MODULE) && cd $(LOCATION)/tbb/ && make target=mic compiler=icc -j12 && cd build && ln -sf mic_icc_release linux_release
else
$(LOCATION)/tbb/build/linux_release/libtbb.so:
	$(LOAD_GCC_MODULE) && cd $(LOCATION)/tbb/ && make -j12 && cd build && ln -sf `find . -name "linux_intel64_gcc_*_release"` linux_release
endif

mpss/$(MPSS_VER)/sysroots/k1om-mpss-linux/usr/lib:
	cp -rf /opt/mpss . && \
	rm -rf /home/marcusmae/apc/hddm-solver-interpolators/LinearBasis/MIC/mpss/$(MPSS_VER)/sysroots/k1om-mpss-linux/usr/lib && \
	ln -sf $(shell pwd)/mpss/$(MPSS_VER)/sysroots/k1om-mpss-linux/usr/lib64 $(shell pwd)/mpss/$(MPSS_VER)/sysroots/k1om-mpss-linux/usr/lib

$(INSTALL_PREFIX)/host/bin/k1om-mpss-linux-gcc: mpss/$(MPSS_VER)/sysroots/k1om-mpss-linux/usr/lib
	$(LOAD_GCC_MODULE) && $(LOAD_FLEX_MODULE) && \
	cd $(LOCATION)/gcc-5.1.1-knc/ && \
	mkdir -p build/ && \
	cd build/ && \
	export PATH=$(shell pwd)/mpss/$(MPSS_VER)/sysroots/x86_64-mpsssdk-linux/usr/bin/k1om-mpss-linux/:$(PATH) && \
	../configure --build=x86_64-linux-gnu --host=x86_64-linux-gnu --target=k1om-mpss-linux --prefix=$(shell pwd)/$(LOCATION)/install/host --disable-silent-rules --disable-dependency-tracking --with-ld=$(shell pwd)/mpss/$(MPSS_VER)/sysroots/x86_64-mpsssdk-linux/usr/bin/k1om-mpss-linux/k1om-mpss-linux-ld --with-as=$(shell pwd)/mpss/$(MPSS_VER)/sysroots/x86_64-mpsssdk-linux/usr/bin/k1om-mpss-linux/k1om-mpss-linux-as --enable-shared --enable-languages=c,c++ --enable-threads=posix --disable-multilib --enable-c99 --enable-long-long --enable-symvers=gnu --enable-libstdcxx-pch --program-prefix=k1om-mpss-linux- --enable-target-optspace --enable-lto --disable-bootstrap --with-system-zlib --with-linker-hash-style=gnu --enable-cheaders=c_global --with-local-prefix=$(shell pwd)/mpss/$(MPSS_VER)/sysroots/k1om-mpss-linux/usr --with-sysroot=$(shell pwd)/mpss/$(MPSS_VER)/sysroots/k1om-mpss-linux/ --disable-libunwind-exceptions --disable-libssp --disable-libgomp --disable-libmudflap --enable-nls --enable-__cxa_atexit --disable-libitm $(GCC_EXTRA_CONFIG) && \
	make -j64 && \
	make -j64 install

$(INSTALL_PREFIX)/host/lib/libmicrt.so: $(INSTALL_PREFIX)/host/bin/k1om-mpss-linux-gcc
	export PATH=$(shell pwd)/$(LOCATION)install/host/bin:$(PATH) && \
	cd $(LOCATION)/liboffloadmic/ && \
	$(LOAD_ICC_MODULE) && \
	make clean && \
	make target=native INSTALL_PREFIX=$(shell pwd)/$(LOCATION)/install MIC_PREFIX=k1om-mpss-linux-

test_InterpolateValue: test_InterpolateValue.o InterpolateValueCPU.o InterpolateValue_embed.o
	$(LOAD_GCC_MODULE) && $(HOST_CC) $(HOST_CFLAGS) $^ -o $@ -L$(INSTALL_PREFIX)/host/lib/$(TOOLEXECLIBDIR) -lmicrt -Wl,-rpath=$(INSTALL_PREFIX)/host/lib/$(TOOLEXECLIBDIR)

test_InterpolateValue.o: test_InterpolateValue.c $(INSTALL_PREFIX)/host/lib/libmicrt.so
	$(LOAD_GCC_MODULE) && $(HOST_CC) $(HOST_CFLAGS) -std=c99 -I$(INSTALL_PREFIX)/host/include -c $< -o $@

InterpolateValueCPU.o: ../CPU/InterpolateValue.c
	$(HOST_CC) $(HOST_CFLAGS) -I.. -mavx -std=c99 -DMODE=Generic -DDIM=8 -DDEFERRED -DHAVE_AVX -c $< -o $@

InterpolateValue_embed.o: InterpolateValue_embed.c InterpolateValue_embed.h
	$(LOAD_GCC_MODULE) && $(HOST_CC) $(HOST_CFLAGS) -I. -I$(INSTALL_PREFIX)/host/include -c $< -o $@

InterpolateValue_embed.h: libInterpolateValueMIC.so
	xxd -include $< >$@

libInterpolateValueMIC.so: InterpolateValue.cpp $(INSTALL_PREFIX)/host/bin/k1om-mpss-linux-gcc $(INSTALL_PREFIX)/host/lib/libmicrt.so $(LOCATION)/tbb/build/linux_release/libtbb.so
	$(LOAD_GCC_MODULE) && $(MIC_CXX) $(MIC_CXXFLAGS) -I.. -mavx512f -DMODE=Generic -DDIM=8 -DDEFERRED -DHAVE_MIC -fPIC -shared $< -o $@ -I$(LOCATION)/tbb/include/

clean:
	rm -rf test_InterpolateValue libInterpolateValueMIC.so InterpolateValue_embed.h *.o

