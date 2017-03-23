NAME = cuda

LOCATION = ../../..

include $(LOCATION)/Makefile.inc

CINC += -I. -I../include -I../../include

# In case of CUDA GPU, AVX_VECTOR_SIZE denotes the size of grid block.
# Empirically good grid block size is (128, 1, 1).
COPT += -DAVX_VECTOR_SIZE=128 -DNAMESPACE=$(NAME)
NVCCOPT += -DAVX_VECTOR_SIZE=128 -DNAMESPACE=$(NAME)
CDIR = mkdir -p $(shell dirname $@)

all: $(INSTALL)/bin/postprocessors/LinearBasis/$(NAME)/libpostprocessor.so

$(INSTALL)/bin/postprocessors/LinearBasis/$(NAME)/libpostprocessor.so: \
	$(BUILD)/InterpolateArray.o $(BUILD)/InterpolateArrayManyMultistate.o \
	$(BUILD)/libInterpolateArray.sh $(BUILD)/libInterpolateArrayManyMultistate.sh \
	$(BUILD)/Interpolator.o $(BUILD)/Data.o $(BUILD)/JIT.o $(BUILD)/SIMDVector.o \
	$(BUILD)/priority.o $(BUILD)/supported.o $(BUILD)/Device.o $(BUILD)/Devices.o $(BUILD)/DeviceProperties.o
	mkdir -p $(INSTALL)/bin/postprocessors/LinearBasis/$(NAME) && $(CXX) $(CINC) $(COPT) -Wl,--whole-archive $(filter %.o,$^) -Wl,--no-whole-archive -shared -o $@ -L$(INSTALL)/lib/ -llinearbasis $(LINK_ACCELERATOR) $(OMPFLAGS) 

$(BUILD)/InterpolateArray.o: src/InterpolateArray.cu include/Interpolate.h
	$(CDIR) && $(NVCC) -DFUNCNAME=LinearBasis_$(NAME)_Generic_InterpolateArray -DDIM=dim $(CINC) $(NVCCOPT) -c $< -o $@

$(BUILD)/InterpolateArrayManyMultistate.o: src/InterpolateArrayManyMultistate.cu include/Interpolate.h
	$(CDIR) && $(NVCC) -DFUNCNAME=LinearBasis_$(NAME)_Generic_InterpolateArrayManyMultistate -DDIM=dim -DCOUNT=count $(CINC) $(NVCCOPT) -c $< -o $@

$(BUILD)/libInterpolateArray.sh: src/InterpolateArray.cu
	$(CDIR) && echo cd $(shell pwd) \&\& $(NVCC) $(CINC) $(NVCCOPT) -shared $< > $@

$(BUILD)/libInterpolateArrayManyMultistate.sh: src/InterpolateArrayManyMultistate.cu
	$(CDIR) && echo cd $(shell pwd) \&\& $(NVCC) $(CINC) $(NVCCOPT) -shared $< > $@

$(BUILD)/parameters.o: src/parameters.cpp $(LOCATION)/include/check.h include/parameters.h include/parameters.c
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/Interpolator.o: src/Interpolator.cpp include/JIT.h include/Data.h
	$(CDIR) && $(CXX) -DINTERPOLATE_ARRAY=LinearBasis_$(NAME)_Generic_InterpolateArray -DINTERPOLATE_ARRAY_RUNTIME_OPT=LinearBasis_$(NAME)_RuntimeOpt_InterpolateArray -DINTERPOLATE_ARRAY_MANY_MULTISTATE=LinearBasis_$(NAME)_Generic_InterpolateArrayManyMultistate -DINTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT=LinearBasis_$(NAME)_RuntimeOpt_InterpolateArrayManyMultistate $(CINC) $(COPT) -c $< -o $@

$(BUILD)/Data.o: src/Data.cu include/Data.h
	$(CDIR) && $(NVCC) $(CINC) $(NVCCOPT) -c $< -o $@

$(BUILD)/JIT.o: src/JIT.cpp include/InterpolateKernel.h
	$(CDIR) && $(CXX) -DINTERPOLATE_ARRAY_SH=\"$(shell pwd)/$(BUILD)/libInterpolateArray.sh\" -DINTERPOLATE_ARRAY_MANY_MULTISTATE_SH=\"$(shell pwd)/$(BUILD)/libInterpolateArrayManyMultistate.sh\" $(CINC) $(COPT) -c $< -o $@

$(BUILD)/SIMDVector.o: src/SIMDVector.cpp include/SIMDVector.h
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/priority.o: src/priority.cpp
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/supported.o: src/supported.cpp
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/Device.o: src/Device.cpp
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/Devices.o: src/Devices.cu
	$(CDIR) && $(NVCC) $(CINC) $(NVCCOPT) $(addprefix -Xcompiler=, $(OMPFLAGS)) -c $< -o $@

$(BUILD)/DeviceProperties.o: src/DeviceProperties.cpp
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@

clean:
	rm -rf $(BUILD) $(INSTALL)/bin/postprocessors/LinearBasis/$(NAME)/libpostprocessor.so

