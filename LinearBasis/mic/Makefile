LOCATION = ../../..

include $(LOCATION)/Makefile.inc

CINC += -I.
COPT += -DAVX_VECTOR_SIZE=8
CDIR = mkdir -p $(shell dirname $@)

all: $(INSTALL)/bin/interpolators/cpu/libinterpolator.so

$(INSTALL)/bin/interpolators/cpu/libinterpolator.so: \
	$(BUILD)/InterpolateValue.o $(BUILD)/InterpolateArray.o $(BUILD)/InterpolateArrayMany.o \
	$(BUILD)/libInterpolateValue.sh $(BUILD)/libInterpolateArray.sh $(BUILD)/libInterpolateArrayMany.sh \
	$(BUILD)/Interpolator.o $(BUILD)/Data.o $(BUILD)/JIT.o
	mkdir -p $(INSTALL)/bin/interpolators/cpu && $(MPICXX) $(CINC) $(COPT) -Wl,--whole-archive $(filter %.o,$^) -Wl,--no-whole-archive -shared -o $@

$(BUILD)/InterpolateValue.o: src/InterpolateValue.c
	$(CDIR) && $(MPICC) -std=c99 -DFUNCNAME=LinearBasis_CPU_Generic_InterpolateValue -DDIM=dim $(CINC) $(COPT) -c $< -o $@

$(BUILD)/InterpolateArray.o: src/InterpolateArray.c
	$(CDIR) && $(MPICC) -std=c99 -DFUNCNAME=LinearBasis_CPU_Generic_InterpolateArray -DDIM=dim $(CINC) $(COPT) -c $< -o $@

$(BUILD)/InterpolateArrayMany.o: src/InterpolateArrayMany.c
	$(CDIR) && $(MPICC) -std=c99 -DFUNCNAME=LinearBasis_CPU_Generic_InterpolateArrayMany -DDIM=dim $(CINC) $(COPT) -c $< -o $@

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
	rm -rf $(BUILD) $(INSTALL)/bin/interpolators/cpu/libinterpolator.so

