LOCATION = ../../..

include $(LOCATION)/Makefile.inc

CINC += -I. -I.. -I$(INSTALL)/include -I$(BUILD)
COPT += -DAVX_VECTOR_SIZE=8
CDIR = mkdir -p $(shell dirname $@)

all: $(INSTALL)/bin/postprocessors/LinearBasis/mic/libpostprocessor.so

$(INSTALL)/bin/postprocessors/LinearBasis/mic/libpostprocessor.so: \
	$(BUILD)/InterpolateValue_embed.o $(BUILD)/InterpolateArray_embed.o $(BUILD)/InterpolateArrayMany_embed.o \
	$(BUILD)/libInterpolateValue.sh $(BUILD)/libInterpolateArray.sh $(BUILD)/libInterpolateArrayMany.sh \
	$(BUILD)/Interpolator.o $(BUILD)/Data.o $(BUILD)/JIT.o
	mkdir -p $(INSTALL)/bin/postprocessors/mic && $(MPICXX) $(CINC) $(COPT) -Wl,--whole-archive $(filter %.o,$^) -Wl,--no-whole-archive -shared -o $@ -L$(INSTALL)/lib -lmicrt

$(INSTALL)/libmicrt.so:
	cd liboffloadmic/ && $(MAKE) clean && $(MAKE) INSTALL_PREFIX=$(INSTALL)/.. MIC_PREFIX=/usr/linux-k1om-4.7/bin/x86-k1om-linux-

$(BUILD)/InterpolateValue_embed.o: src/InterpolateValue_embed.c $(BUILD)/InterpolateValue_embed.h $(INSTALL)/lib/libmicrt.so
	$(CDIR) && $(MPICC) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/InterpolateArray_embed.o: src/InterpolateArray_embed.c $(BUILD)/InterpolateArray_embed.h $(INSTALL)/lib/libmicrt.so
	$(CDIR) && $(MPICC) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/InterpolateArrayMany_embed.o: src/InterpolateArrayMany_embed.c $(BUILD)/InterpolateArrayMany_embed.h $(INSTALL)/lib/libmicrt.so
	$(CDIR) && $(MPICC) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/InterpolateValue_embed.h: $(BUILD)/libInterpolateValue.so
	$(CDIR) && cd $(BUILD) && xxd -include libInterpolateValue.so >InterpolateValue_embed.h

$(BUILD)/InterpolateArray_embed.h: $(BUILD)/libInterpolateArray.so
	$(CDIR) && cd $(BUILD) && xxd -include libInterpolateArray.so >InterpolateArray_embed.h

$(BUILD)/InterpolateArrayMany_embed.h: $(BUILD)/libInterpolateArrayMany.so
	$(CDIR) && cd $(BUILD) && xxd -include libInterpolateArrayMany.so >InterpolateArrayMany_embed.h

$(BUILD)/libInterpolateValue.so: src/InterpolateValue.c
	$(CDIR) && icc -mmic -DFUNCNAME=LinearBasis_MIC_Generic_InterpolateValue -DDIM=dim $(CINC) -std=c99 -fopenmp $(filter-out -std=c++0x -ffast-math -mavx,$(COPT)) -shared $< -o $@

$(BUILD)/libInterpolateArray.so: src/InterpolateArray.c
	$(CDIR) && icc -mmic -DFUNCNAME=LinearBasis_MIC_Generic_InterpolateArray -DDIM=dim $(CINC) -std=c99 -fopenmp $(filter-out -std=c++0x -ffast-math -mavx,$(COPT)) -shared $< -o $@

$(BUILD)/libInterpolateArrayMany.so: src/InterpolateArrayMany.c
	$(CDIR) && icc -mmic -DFUNCNAME=LinearBasis_MIC_Generic_InterpolateArrayMany -DDIM=dim $(CINC) -std=c99 -fopenmp $(filter-out -std=c++0x -ffast-math -mavx,$(COPT)) -shared $< -o $@

$(BUILD)/libInterpolateValue.sh: src/InterpolateValue.c
	$(CDIR) && echo cd $(shell pwd) \&\& $(MPICC) -std=c99 $(CINC) $(COPT) -shared -fno-lto $^  > $@

$(BUILD)/libInterpolateArray.sh: src/InterpolateArray.c
	$(CDIR) && echo cd $(shell pwd) \&\& $(MPICC) -std=c99 $(CINC) $(COPT) -shared -fno-lto $^  > $@

$(BUILD)/libInterpolateArrayMany.sh: src/InterpolateArrayMany.c
	$(CDIR) && echo cd $(shell pwd) \&\& $(MPICC) -std=c99 $(CINC) $(COPT) -shared -fno-lto $^  > $@

$(BUILD)/Interpolator.o: src/Interpolator.cpp include/JIT.h include/Data.h
	$(CDIR) && $(MPICXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/Data.o: src/Data.cpp include/Data.h
	$(CDIR) && $(MPICXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/JIT.o: src/JIT.cpp include/InterpolateKernel.h
	$(CDIR) && $(MPICXX) -DINTERPOLATE_ARRAY_SH=\"$(shell pwd)/$(BUILD)/libInterpolateArray.sh\" -DINTERPOLATE_ARRAY_MANY_SH=\"$(shell pwd)/$(BUILD)/libInterpolateArrayMany.sh\" -DINTERPOLATE_VALUE_SH=\"$(shell pwd)/$(BUILD)/libInterpolateValue.sh\" $(CINC) $(COPT) -c $< -o $@

clean:
	rm -rf $(BUILD) $(INSTALL)/bin/postprocessors/LinearBasis/mic/libpostprocessor.so

