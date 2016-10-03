LOCATION = ../../..

include $(LOCATION)/Makefile.inc

CINC += -I. -I../include -I../../include

# In case of CUDA GPU, AVX_VECTOR_SIZE denotes the size of grid block.
# Empirically good grid block size is (128, 1, 1).
COPT += -DAVX_VECTOR_SIZE=128 -DNAMESPACE=cuda
NVCCOPT += -DAVX_VECTOR_SIZE=128 -DNAMESPACE=cuda
CDIR = mkdir -p $(shell dirname $@)

all: $(INSTALL)/bin/postprocessors/LinearBasis/cuda/libpostprocessor.so

$(INSTALL)/bin/postprocessors/LinearBasis/cuda/libpostprocessor.so: \
	$(BUILD)/InterpolateValue.o $(BUILD)/InterpolateArray.o \
	$(BUILD)/InterpolateArrayManyStateless.o $(BUILD)/InterpolateArrayManyMultistate.o \
	$(BUILD)/libInterpolateValue.sh $(BUILD)/libInterpolateArray.sh \
	$(BUILD)/libInterpolateArrayManyStateless.sh $(BUILD)/libInterpolateArrayManyMultistate.sh \
	$(BUILD)/Interpolator.o $(BUILD)/Data.o $(BUILD)/JIT.o $(BUILD)/SIMDVector.o \
	$(BUILD)/priority.o $(BUILD)/supported.o $(BUILD)/Device.o $(BUILD)/Devices.o $(BUILD)/DeviceProperties.o
	mkdir -p $(INSTALL)/bin/postprocessors/LinearBasis/cuda && $(MPICXX) $(CINC) $(COPT) -Wl,--whole-archive $(filter %.o,$^) -Wl,--no-whole-archive -shared -o $@ -L$(INSTALL)/lib/ -llinearbasis $(LINK_ACCELERATOR) $(OMPFLAGS)

$(BUILD)/InterpolateValue.o: src/InterpolateValue.cu include/Interpolate.h
	$(CDIR) && $(NVCC) -DFUNCNAME=LinearBasis_CUDA_Generic_InterpolateValue -DDIM=dim $(CINC) $(NVCCOPT) -c $< -o $@

$(BUILD)/InterpolateArray.o: src/InterpolateArray.cu include/Interpolate.h
	$(CDIR) && $(NVCC) -DFUNCNAME=LinearBasis_CUDA_Generic_InterpolateArray -DDIM=dim $(CINC) $(NVCCOPT) -c $< -o $@

$(BUILD)/InterpolateArrayManyStateless.o: src/InterpolateArrayManyStateless.cu include/Interpolate.h
	$(CDIR) && $(NVCC) -DFUNCNAME=LinearBasis_CUDA_Generic_InterpolateArrayManyStateless -DDIM=dim -DCOUNT=count $(CINC) $(NVCCOPT) -c $< -o $@

$(BUILD)/InterpolateArrayManyMultistate.o: src/InterpolateArrayManyMultistate.cu include/Interpolate.h
	$(CDIR) && $(NVCC) -DFUNCNAME=LinearBasis_CUDA_Generic_InterpolateArrayManyMultistate -DDIM=dim -DCOUNT=count $(CINC) $(NVCCOPT) -c $< -o $@

$(BUILD)/libInterpolateValue.sh: src/InterpolateValue.cu
	$(CDIR) && echo cd $(shell pwd) \&\& $(NVCC) $(CINC) $(NVCCOPT) -shared $< > $@

$(BUILD)/libInterpolateArray.sh: src/InterpolateArray.cu
	$(CDIR) && echo cd $(shell pwd) \&\& $(NVCC) $(CINC) $(NVCCOPT) -shared $< > $@

$(BUILD)/libInterpolateArrayManyStateless.sh: src/InterpolateArrayManyStateless.cu
	$(CDIR) && echo cd $(shell pwd) \&\& $(NVCC) $(CINC) $(NVCCOPT) -shared $< > $@

$(BUILD)/libInterpolateArrayManyMultistate.sh: src/InterpolateArrayManyMultistate.cu
	$(CDIR) && echo cd $(shell pwd) \&\& $(NVCC) $(CINC) $(NVCCOPT) -shared $< > $@

$(BUILD)/Interpolator.o: src/Interpolator.cpp include/JIT.h include/Data.h
	$(CDIR) && $(MPICXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/Data.o: src/Data.cu include/Data.h
	$(CDIR) && $(NVCC) $(CINC) $(NVCCOPT) -c $< -o $@

$(BUILD)/JIT.o: src/JIT.cpp include/InterpolateKernel.h
	$(CDIR) && $(MPICXX) -DINTERPOLATE_ARRAY_SH=\"$(shell pwd)/$(BUILD)/libInterpolateArray.sh\" -DINTERPOLATE_ARRAY_MANY_STATELESS_SH=\"$(shell pwd)/$(BUILD)/libInterpolateArrayManyStateless.sh\" -DINTERPOLATE_ARRAY_MANY_MULTISTATE_SH=\"$(shell pwd)/$(BUILD)/libInterpolateArrayManyMultistate.sh\" -DINTERPOLATE_VALUE_SH=\"$(shell pwd)/$(BUILD)/libInterpolateValue.sh\" $(CINC) $(COPT) -c $< -o $@

$(BUILD)/SIMDVector.o: src/SIMDVector.cpp include/SIMDVector.h
	$(CDIR) && $(MPICXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/priority.o: src/priority.cpp
	$(CDIR) && $(MPICXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/supported.o: src/supported.cpp
	$(CDIR) && $(MPICXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/Device.o: src/Device.cpp
	$(CDIR) && $(MPICXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/Devices.o: src/Devices.cu
	$(CDIR) && $(NVCC) $(CINC) $(NVCCOPT) $(addprefix -Xcompiler=, $(OMPFLAGS)) -c $< -o $@

$(BUILD)/DeviceProperties.o: src/DeviceProperties.cpp
	$(CDIR) && $(MPICXX) $(CINC) $(COPT) -c $< -o $@

clean:
	rm -rf $(BUILD) $(INSTALL)/bin/postprocessors/LinearBasis/cuda/libpostprocessor.so

