NAME = avx2

LOCATION = ../../..

include $(LOCATION)/Makefile.inc

CINC += -I. -I.. -I../include -I../../include
COPT += -DAVX_VECTOR_SIZE=8 -DDOUBLE_VECTOR_SIZE=4 -DNAMESPACE=$(NAME)
CDIR = mkdir -p $(shell dirname $@)

ifeq (gnu,$(COMPILER))
COPT += -mavx -mavx2 -mfma
endif

ifeq (intel,$(COMPILER))
COPT += -xCORE-AVX2
endif

all: $(INSTALL)/bin/postprocessors/LinearBasis/$(NAME)/libpostprocessor.so

$(INSTALL)/bin/postprocessors/LinearBasis/$(NAME)/libpostprocessor.so: \
	$(BUILD)/InterpolateArray.o $(BUILD)/InterpolateArrayManyMultistate.o \
	$(BUILD)/libInterpolateArray.sh $(BUILD)/libInterpolateArrayManyMultistate.sh \
	$(BUILD)/parameters.o $(BUILD)/Interpolator.o $(BUILD)/Data.o $(BUILD)/JIT.o \
	$(BUILD)/priority.o $(BUILD)/instrset.o $(BUILD)/supported.o $(BUILD)/reset.o \
	$(BUILD)/Device.o $(BUILD)/Devices.o $(BUILD)/DeviceProperties.o $(BUILD)/getData.o
	mkdir -p $(INSTALL)/bin/postprocessors/LinearBasis/$(NAME) && $(MPILD) $(CINC) $(COPT) -Wl,--whole-archive $(filter %.o,$^) -Wl,--no-whole-archive -shared -o $@ -L$(INSTALL)/lib/ $(MPILDFLAGS)

$(BUILD)/InterpolateArray.o: src/InterpolateArray.cpp
	$(CDIR) && $(CXX) -DFUNCNAME=LinearBasis_$(NAME)_Generic_InterpolateArray -DDIM=dim -DDOF_PER_NODE=DofPerNode $(CINC) $(COPT) -c $< -o $@

$(BUILD)/InterpolateArrayManyMultistate.o: src/InterpolateArrayManyMultistate.cpp
	$(CDIR) && $(CXX) -DFUNCNAME=LinearBasis_$(NAME)_Generic_InterpolateArrayManyMultistate -DDIM=dim -DCOUNT=count -DDOF_PER_NODE=DofPerNode $(CINC) $(COPT) -c $< -o $@

$(BUILD)/libInterpolateArray.sh: src/InterpolateArray.cpp
	$(CDIR) && echo cd $(shell pwd) \&\& $(CXX) $(CINC) $(COPT) -shared $^  > $@

$(BUILD)/libInterpolateArrayManyMultistate.sh: src/InterpolateArrayManyMultistate.cpp
	$(CDIR) && echo cd $(shell pwd) \&\& $(CXX) $(CINC) $(COPT) -shared $^  > $@

$(BUILD)/parameters.o: src/parameters.cpp $(LOCATION)/include/check.h include/parameters.h include/parameters.c
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/Interpolator.o: src/Interpolator.cpp include/JIT.h include/Data.h
	$(CDIR) && $(CXX) -DNAME=$(NAME) -DINTERPOLATE_ARRAY=LinearBasis_$(NAME)_Generic_InterpolateArray -DINTERPOLATE_ARRAY_RUNTIME_OPT=LinearBasis_$(NAME)_RuntimeOpt_InterpolateArray -DINTERPOLATE_ARRAY_MANY_MULTISTATE=LinearBasis_$(NAME)_Generic_InterpolateArrayManyMultistate -DINTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT=LinearBasis_$(NAME)_RuntimeOpt_InterpolateArrayManyMultistate $(CINC) $(COPT) -c $< -o $@

$(BUILD)/Data.o: src/Data.cpp include/Data.h
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/JIT.o: src/JIT.cpp include/InterpolateKernel.h
	$(CDIR) && $(CXX) -DINTERPOLATE_ARRAY_SH=\"$(shell pwd)/$(BUILD)/libInterpolateArray.sh\" -DINTERPOLATE_ARRAY_MANY_MULTISTATE_SH=\"$(shell pwd)/$(BUILD)/libInterpolateArrayManyMultistate.sh\" $(CINC) $(COPT) -c $< -o $@

$(BUILD)/priority.o: src/priority.cpp
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/instrset.o: src/instrset.cpp include/instrset.h
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/supported.o: src/supported.cpp
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/reset.o: src/reset.cpp
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/Device.o: src/Device.cpp
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/Devices.o: src/Devices.cpp
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/DeviceProperties.o: src/DeviceProperties.cpp
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/getData.o: src/getData.cpp
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@

clean:
	rm -rf $(BUILD) $(INSTALL)/bin/postprocessors/LinearBasis/$(NAME)/libpostprocessor.so

