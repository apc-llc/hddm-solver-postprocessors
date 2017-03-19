NAME = avx512

LOCATION = ../../..

include $(LOCATION)/Makefile.inc

CINC += -I. -I../include -I../../include
COPT_FILTERED := $(filter-out -ffast-math, $(COPT))
COPT = $(COPT_FILTERED) -DAVX_VECTOR_SIZE=16 -DNAMESPACE=$(NAME) -fast -axMIC-AVX512
CDIR = mkdir -p $(shell dirname $@)

all: $(INSTALL)/bin/postprocessors/LinearBasis/avx512/libpostprocessor.so

$(INSTALL)/bin/postprocessors/LinearBasis/$(NAME)/libpostprocessor.so: \
	$(BUILD)/InterpolateArray.o $(BUILD)/InterpolateArrayManyMultistate.o \
	$(BUILD)/libInterpolateArray.sh $(BUILD)/libInterpolateArrayManyMultistate.sh \
	$(BUILD)/Interpolator.o $(BUILD)/Data.o $(BUILD)/JIT.o \
	$(BUILD)/priority.o $(BUILD)/supported.o $(BUILD)/Device.o $(BUILD)/Devices.o $(BUILD)/DeviceProperties.o
	mkdir -p $(INSTALL)/bin/postprocessors/LinearBasis/$(NAME) && $(CXX) $(CINC) $(COPT) -Wl,--whole-archive $(filter %.o,$^) -Wl,--no-whole-archive -shared -o $@ -L$(INSTALL)/lib/ -llinearbasis

$(BUILD)/InterpolateArray.o: src/InterpolateArray.cpp
	$(CDIR) && $(CXX) -DFUNCNAME=LinearBasis_CPU_Generic_InterpolateArray -DDIM=dim -DNNO=nno -DDOF_CHOICE_START=Dof_choice_start -DDOF_CHOICE_END=Dof_choice_end $(CINC) $(COPT) -c $< -o $@

$(BUILD)/InterpolateArrayManyMultistate.o: src/InterpolateArrayManyMultistate.cpp
	$(CDIR) && $(CXX) -DFUNCNAME=LinearBasis_CPU_Generic_InterpolateArrayManyMultistate -DDIM=dim -DCOUNT=count -DNNO=nno -DDOF_CHOICE_START=Dof_choice_start -DDOF_CHOICE_END=Dof_choice_end $(CINC) $(COPT) -c $< -o $@

$(BUILD)/libInterpolateArray.sh: src/InterpolateArray.cpp
	$(CDIR) && echo cd $(shell pwd) \&\& $(CXX) $(CINC) $(COPT) -shared $^  > $@

$(BUILD)/libInterpolateArrayManyMultistate.sh: src/InterpolateArrayManyMultistate.cpp
	$(CDIR) && echo cd $(shell pwd) \&\& $(CXX) $(CINC) $(COPT) -shared $^  > $@

$(BUILD)/Interpolator.o: src/Interpolator.cpp include/JIT.h include/Data.h
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/Data.o: src/Data.cpp include/Data.h
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/JIT.o: src/JIT.cpp include/InterpolateKernel.h
	$(CDIR) && $(CXX) -DINTERPOLATE_ARRAY_SH=\"$(shell pwd)/$(BUILD)/libInterpolateArray.sh\" -DINTERPOLATE_ARRAY_MANY_MULTISTATE_SH=\"$(shell pwd)/$(BUILD)/libInterpolateArrayManyMultistate.sh\" $(CINC) $(COPT) -c $< -o $@

$(BUILD)/priority.o: src/priority.cpp
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/supported.o: src/supported.cpp
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/Device.o: src/Device.cpp
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/Devices.o: src/Devices.cpp
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/DeviceProperties.o: src/DeviceProperties.cpp
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@

clean:
	rm -rf $(BUILD) $(INSTALL)/bin/postprocessors/LinearBasis/$(NAME)/libpostprocessor.so

