LOCATION = ../../..

include $(LOCATION)/Makefile.inc

CINC += -I. -I../include -I../../include
COPT += -DAVX_VECTOR_SIZE=8 -DNAMESPACE=cpu
CDIR = mkdir -p $(shell dirname $@)

all: $(INSTALL)/bin/postprocessors/LinearBasis/cpu/libpostprocessor.so

$(INSTALL)/bin/postprocessors/LinearBasis/cpu/libpostprocessor.so: \
	$(BUILD)/InterpolateValue.o $(BUILD)/InterpolateArray.o \
	$(BUILD)/InterpolateArrayManyStateless.o $(BUILD)/InterpolateArrayManyMultistate.o \
	$(BUILD)/libInterpolateValue.sh $(BUILD)/libInterpolateArray.sh \
	$(BUILD)/libInterpolateArrayManyStateless.sh $(BUILD)/libInterpolateArrayManyMultistate.sh \
	$(BUILD)/Interpolator.o $(BUILD)/Data.o $(BUILD)/JIT.o \
	$(BUILD)/priority.o $(BUILD)/supported.o $(BUILD)/Device.o $(BUILD)/Devices.o $(BUILD)/DeviceProperties.o
	mkdir -p $(INSTALL)/bin/postprocessors/LinearBasis/cpu && $(MPICXX) $(CINC) $(COPT) -Wl,--whole-archive $(filter %.o,$^) -Wl,--no-whole-archive -shared -o $@ -L$(INSTALL)/lib/ -llinearbasis

$(BUILD)/InterpolateValue.o: src/InterpolateValue.cpp
	$(CDIR) && $(MPICXX) -DFUNCNAME=LinearBasis_CPU_Generic_InterpolateValue -DDIM=dim $(CINC) $(COPT) -c $< -o $@

$(BUILD)/InterpolateArray.o: src/InterpolateArray.cpp
	$(CDIR) && $(MPICXX) -DFUNCNAME=LinearBasis_CPU_Generic_InterpolateArray -DDIM=dim $(CINC) $(COPT) -c $< -o $@

$(BUILD)/InterpolateArrayManyStateless.o: src/InterpolateArrayManyStateless.cpp
	$(CDIR) && $(MPICXX) -DFUNCNAME=LinearBasis_CPU_Generic_InterpolateArrayManyStateless -DDIM=dim -DCOUNT=count $(CINC) $(COPT) -c $< -o $@

$(BUILD)/InterpolateArrayManyMultistate.o: src/InterpolateArrayManyMultistate.cpp
	$(CDIR) && $(MPICXX) -DFUNCNAME=LinearBasis_CPU_Generic_InterpolateArrayManyMultistate -DDIM=dim -DCOUNT=count $(CINC) $(COPT) -c $< -o $@

$(BUILD)/libInterpolateValue.sh: src/InterpolateValue.cpp
	$(CDIR) && echo cd $(shell pwd) \&\& $(MPICXX) $(CINC) $(COPT) -shared $^  > $@

$(BUILD)/libInterpolateArray.sh: src/InterpolateArray.cpp
	$(CDIR) && echo cd $(shell pwd) \&\& $(MPICXX) $(CINC) $(COPT) -shared $^  > $@

$(BUILD)/libInterpolateArrayManyStateless.sh: src/InterpolateArrayManyStateless.cpp
	$(CDIR) && echo cd $(shell pwd) \&\& $(MPICXX) $(CINC) $(COPT) -shared $^  > $@

$(BUILD)/libInterpolateArrayManyMultistate.sh: src/InterpolateArrayManyMultistate.cpp
	$(CDIR) && echo cd $(shell pwd) \&\& $(MPICXX) $(CINC) $(COPT) -shared $^  > $@

$(BUILD)/Interpolator.o: src/Interpolator.cpp include/JIT.h include/Data.h
	$(CDIR) && $(MPICXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/Data.o: src/Data.cpp include/Data.h
	$(CDIR) && $(MPICXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/JIT.o: src/JIT.cpp include/InterpolateKernel.h
	$(CDIR) && $(MPICXX) -DINTERPOLATE_ARRAY_SH=\"$(shell pwd)/$(BUILD)/libInterpolateArray.sh\" -DINTERPOLATE_ARRAY_MANY_STATELESS_SH=\"$(shell pwd)/$(BUILD)/libInterpolateArrayManyStateless.sh\" -DINTERPOLATE_ARRAY_MANY_MULTISTATE_SH=\"$(shell pwd)/$(BUILD)/libInterpolateArrayManyMultistate.sh\" -DINTERPOLATE_VALUE_SH=\"$(shell pwd)/$(BUILD)/libInterpolateValue.sh\" $(CINC) $(COPT) -c $< -o $@

$(BUILD)/priority.o: src/priority.cpp
	$(CDIR) && $(MPICXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/supported.o: src/supported.cpp
	$(CDIR) && $(MPICXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/Device.o: src/Device.cpp
	$(CDIR) && $(MPICXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/Devices.o: src/Devices.cpp
	$(CDIR) && $(MPICXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/DeviceProperties.o: src/DeviceProperties.cpp
	$(CDIR) && $(MPICXX) $(CINC) $(COPT) -c $< -o $@

clean:
	rm -rf $(BUILD) $(INSTALL)/bin/postprocessors/LinearBasis/cpu/libpostprocessor.so

