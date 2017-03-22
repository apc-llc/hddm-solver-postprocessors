LOCATION = ../..

include $(LOCATION)/Makefile.inc

CINC += -I../include -I../../include
CDIR = mkdir -p $(shell dirname $@)

TARGETS = x86
ifneq (,$(HAVE_AVX))
TARGETS += avx
endif
ifneq (,$(HAVE_AVX2))
TARGETS += avx2
endif
ifneq (,$(which icpc))
TARGETS += avx512
endif
ifneq (,$(NVCC))
TARGETS += cuda
endif

.PHONY: $(TARGETS)

all: $(TARGETS)

x86:
	cd $@ && $(MAKE) -j$(NJOBS)

avx512:
	cd $@ && $(MAKE) -j$(NJOBS)

cuda:
	cd $@ && $(MAKE) -j$(NJOBS)

clean:
	rm -rf $(BUILD) && \
	cd x86 && $(MAKE) clean && cd .. && \
	cd avx && $(MAKE) clean && cd .. && \
	cd avx2 && $(MAKE) clean && cd .. && \
	cd avx512 && $(MAKE) clean && cd .. && \
	cd cuda && $(MAKE) clean && cd ..

