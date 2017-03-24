LOCATION = ../..

include $(LOCATION)/Makefile.inc

CINC += -I../include -I../../include
CDIR = mkdir -p $(shell dirname $@)

TARGETS = x86
ifneq (,$(AVX))
TARGETS += avx
else
$(info AVX target is not supported by host GCC compiler)
endif
ifneq (,$(AVX2))
TARGETS += avx2
else
$(info AVX2 target is not supported by host GCC compiler)
endif
ifneq (,$(shell which icpc))
TARGETS += avx512
else
$(info AVX512 target is not built because Intel compiler is not found)
endif

.PHONY: $(TARGETS)

all: $(TARGETS)

x86:
	cd $@ && $(MAKE) -j$(NJOBS)

avx:
	cd $@ && $(MAKE) -j$(NJOBS)

avx2:
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

