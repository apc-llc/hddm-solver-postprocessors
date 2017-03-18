LOCATION = ../../..

include $(LOCATION)/Makefile.inc

CINC += -I. -I../include -I../../include
COPT += -DAVX_VECTOR_SIZE=8 -DNAMESPACE=cpu
CDIR = mkdir -p $(shell dirname $@)

all: $(INSTALL)/bin/postprocessors/LinearBasis/cpu/libpostprocessor.so

$(INSTALL)/bin/postprocessors/LinearBasis/cpu/libpostprocessor.so: \
	$(BUILD)/InterpolateArray.o $(BUILD)/InterpolateArrayManyMultistate.o \
	$(BUILD)/libInterpolateArray.sh $(BUILD)/libInterpolateArrayManyMultistate.sh \
	$(BUILD)/Interpolator.o $(BUILD)/Data.o $(BUILD)/JIT.o \
	$(BUILD)/priority.o $(BUILD)/supported.o $(BUILD)/Device.o $(BUILD)/Devices.o $(BUILD)/DeviceProperties.o
	mkdir -p $(INSTALL)/bin/postprocessors/LinearBasis/cpu && $(MPICXX) $(CINC) $(COPT) -Wl,--whole-archive $(filter %.o,$^) -Wl,--no-whole-archive -shared -o $@ -L$(INSTALL)/lib/ -llinearbasis

$(BUILD)/InterpolateArray.o: src/InterpolateArray.cpp
	$(CDIR) && $(MPICXX) -DFUNCNAME=LinearBasis_CPU_Generic_InterpolateArray -DDIM=dim -DNNO=nno -DVDIM8=vdim8 -DDOF_CHOICE_START=Dof_choice_start -DDOF_CHOICE_END=Dof_choice_end $(CINC) $(COPT) -c $< -o $@

$(BUILD)/InterpolateArrayManyMultistate.o: src/InterpolateArrayManyMultistate.cpp
	$(CDIR) && $(MPICXX) -DFUNCNAME=LinearBasis_CPU_Generic_InterpolateArrayManyMultistate -DDIM=dim -DCOUNT=count -DNNO=nno -DVDIM8=vdim8 -DDOF_CHOICE_START=Dof_choice_start -DDOF_CHOICE_END=Dof_choice_end $(CINC) $(COPT) -c $< -o $@

$(BUILD)/libInterpolateArray.sh: src/InterpolateArray.cpp
	$(CDIR) && echo cd $(shell pwd) \&\& $(MPICXX) $(CINC) $(COPT) -shared $^  > $@

$(BUILD)/libInterpolateArrayManyMultistate.sh: src/InterpolateArrayManyMultistate.cpp
	$(CDIR) && echo cd $(shell pwd) \&\& $(MPICXX) $(CINC) $(COPT) -shared $^  > $@

$(BUILD)/Interpolator.o: src/Interpolator.cpp include/JIT.h include/Data.h
	$(CDIR) && $(MPICXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/Data.o: src/Data.cpp include/Data.h
	$(CDIR) && $(MPICXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/JIT.o: src/JIT.cpp include/InterpolateKernel.h
	$(CDIR) && $(MPICXX) -DINTERPOLATE_ARRAY_SH=\"$(shell pwd)/$(BUILD)/libInterpolateArray.sh\" -DINTERPOLATE_ARRAY_MANY_MULTISTATE_SH=\"$(shell pwd)/$(BUILD)/libInterpolateArrayManyMultistate.sh\" $(CINC) $(COPT) -c $< -o $@

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

