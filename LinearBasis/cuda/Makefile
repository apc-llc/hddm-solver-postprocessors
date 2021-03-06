NAME = cuda

LOCATION = ../../..

include $(LOCATION)/Makefile.inc

NVCC := $(shell which nvcc 2>/dev/null)

CINC += -I. -I../include -I../../include -DDEVICE_CONST_X_MAX_SIZE=3968 -I$(shell dirname $(NVCC) 2>/dev/null)/../include

NVCCOPT := -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -std=c++11 -ccbin=$(shell which $(firstword $(MPICC)))

ifeq (,$(DEBUG))
NVCCOPT += -use_fast_math
else
NVCCOPT += -G
endif

#
# Platform-dependent path to search for CUDA libraries.
#
CUDA_LIBRARY_PATH := $(shell dirname $(NVCC))/../lib
ifeq (x86_64,$(shell uname -m))
        CUDA_LIBRARY_PATH := $(CUDA_LIBRARY_PATH)64
endif

# In case of CUDA GPU, AVX_VECTOR_SIZE denotes the size of grid block.
# Empirically good grid block size is (128, 1, 1).
COPT += -DAVX_VECTOR_SIZE=128 -DNAMESPACE=$(NAME)
NVCCOPT += -DAVX_VECTOR_SIZE=128 -DNAMESPACE=$(NAME)
CDIR = mkdir -p $(shell dirname $@)

NVCCOPT += $(addsuffix \", $(addprefix -Xcompiler=\", $(subst $(COMMA),\$(COMMA),$(COPT))))

all: $(INSTALL)/bin/postprocessors/LinearBasis/$(NAME)/libpostprocessor.so

$(INSTALL)/bin/postprocessors/LinearBasis/$(NAME)/libpostprocessor.so: \
	$(BUILD)/exec-stream.o \
	$(BUILD)/InterpolateArray.o $(BUILD)/InterpolateArrayManyMultistate.o \
	$(BUILD)/libInterpolateArray.sh.o $(BUILD)/libInterpolateArrayManyMultistate.sh.o \
	$(BUILD)/parameters.o $(BUILD)/Interpolator.o $(BUILD)/Data.o $(BUILD)/JIT.o $(BUILD)/SIMDVector.o \
	$(BUILD)/priority.o $(BUILD)/supported.o $(BUILD)/reset.o \
	$(BUILD)/Device.o $(BUILD)/Devices.o $(BUILD)/DeviceProperties.o
	mkdir -p $(INSTALL)/bin/postprocessors/LinearBasis/$(NAME) && $(CXX) $(CINC) $(COPT) -Wl,--whole-archive $(filter %.o,$^) -Wl,--no-whole-archive -shared -o $@ -L$(INSTALL)/lib/ -L$(CUDA_LIBRARY_PATH) -lcudart_static -Wl,-rpath=$(CUDA_LIBRARY_PATH)

$(BUILD)/exec-stream.o: ../src/exec-stream.cpp ../src/posix/exec-stream-helpers.cpp ../src/posix/exec-stream-impl.cpp
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/InterpolateArray.o: src/InterpolateArray.cu ../include/LinearBasis.h
	$(CDIR) && $(NVCC) -DFUNCNAME=LinearBasis_$(NAME)_Generic_InterpolateArray -DDIM=dim -DDOF_PER_NODE=DofPerNode $(CINC) $(NVCCOPT) -c $< -o $@

$(BUILD)/InterpolateArrayManyMultistate.o: src/InterpolateArrayManyMultistate.cu ../include/LinearBasis.h
	$(CDIR) && $(NVCC) -DFUNCNAME=LinearBasis_$(NAME)_Generic_InterpolateArrayManyMultistate -DDIM=dim -DCOUNT=count -DDOF_PER_NODE=DofPerNode $(CINC) $(NVCCOPT) -c $< -o $@

$(BUILD)/libInterpolateArray.sh: src/InterpolateArray.cu
	$(CDIR) && echo cd $(shell pwd) \&\& $(NVCC) $(CINC) $(filter-out -gencode%,$(NVCCOPT)) -shared $< > $@

$(BUILD)/libInterpolateArray.sh.c: $(BUILD)/libInterpolateArray.sh
	$(CDIR) && xxd -i $< $@

$(BUILD)/libInterpolateArray.sh.o: $(BUILD)/libInterpolateArray.sh.c
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/libInterpolateArrayManyMultistate.sh: src/InterpolateArrayManyMultistate.cu
	$(CDIR) && echo cd $(shell pwd) \&\& $(NVCC) $(CINC) $(filter-out -gencode%,$(NVCCOPT)) -shared $< > $@

$(BUILD)/libInterpolateArrayManyMultistate.sh.c: $(BUILD)/libInterpolateArrayManyMultistate.sh
	$(CDIR) && xxd -i $< $@

$(BUILD)/libInterpolateArrayManyMultistate.sh.o: $(BUILD)/libInterpolateArrayManyMultistate.sh.c
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/parameters.o: src/parameters.cpp $(LOCATION)/include/check.h include/parameters.h include/parameters.c
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/Interpolator.o: src/Interpolator.cpp include/JIT.h include/Data.h
	$(CDIR) && $(CXX) -DNAME=$(NAME) -DINTERPOLATE_ARRAY=LinearBasis_$(NAME)_Generic_InterpolateArray -DINTERPOLATE_ARRAY_RUNTIME_OPT=LinearBasis_$(NAME)_RuntimeOpt_InterpolateArray -DINTERPOLATE_ARRAY_MANY_MULTISTATE=LinearBasis_$(NAME)_Generic_InterpolateArrayManyMultistate -DINTERPOLATE_ARRAY_MANY_MULTISTATE_RUNTIME_OPT=LinearBasis_$(NAME)_RuntimeOpt_InterpolateArrayManyMultistate $(CINC) $(COPT) -c $< -o $@

$(BUILD)/Data.o: src/Data.cu include/Data.h
	$(CDIR) && $(NVCC) $(CINC) $(NVCCOPT) -c $< -o $@

$(BUILD)/JIT.o: src/JIT.cpp include/InterpolateKernel.h
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@ \
		-DINTERPOLATE_ARRAY_SH=$(subst /,_,$(BUILD))_libInterpolateArray_sh \
		-DINTERPOLATE_ARRAY_SH_LEN=$(subst /,_,$(BUILD))_libInterpolateArray_sh_len \
		-DINTERPOLATE_ARRAY_MANY_MULTISTATE_SH=$(subst /,_,$(BUILD))_libInterpolateArrayManyMultistate_sh \
		-DINTERPOLATE_ARRAY_MANY_MULTISTATE_SH_LEN=$(subst /,_,$(BUILD))_libInterpolateArrayManyMultistate_sh_len

$(BUILD)/SIMDVector.o: src/SIMDVector.cpp include/SIMDVector.h
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/priority.o: src/priority.cpp
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/supported.o: src/supported.cpp
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/reset.o: src/reset.cpp
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@

$(BUILD)/Device.o: src/Device.cu
	$(CDIR) && $(NVCC) $(CINC) $(NVCCOPT) $(addprefix -Xcompiler=, $(OMPFLAGS)) -c $< -o $@

$(BUILD)/Devices.o: src/Devices.cu
	$(CDIR) && $(NVCC) $(CINC) $(NVCCOPT) $(addprefix -Xcompiler=, $(OMPFLAGS)) -c $< -o $@

$(BUILD)/DeviceProperties.o: src/DeviceProperties.cpp
	$(CDIR) && $(CXX) $(CINC) $(COPT) -c $< -o $@

clean:
	rm -rf $(BUILD) $(INSTALL)/bin/postprocessors/LinearBasis/$(NAME)/libpostprocessor.so

