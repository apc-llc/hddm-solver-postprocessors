.PHONY: run

LOCATION = ../../../..

include $(LOCATION)/Makefile.inc

GTEST_DIR = ../../googletest/googletest

# Flags passed to the preprocessor.
# Set Google Test's header directory as a system directory, such that
# the compiler doesn't generate warnings in Google Test headers.
CINC += -isystem $(GTEST_DIR)/include

CINC += -I../.. -I../../../include -I$(LOCATION)/postprocess/include

COPT += -DAVX_VECTOR_SIZE=8

NVCC := $(shell which nvcc 2>/dev/null)

ifneq (,$(NVCC))
CINC += -I$(shell dirname $(NVCC) 2>/dev/null)/../include
COPT += -DNVCC
endif

CDIR = mkdir -p $(shell dirname $@)
FDIR = mkdir -p $(shell dirname $@)

# All Google Test headers.  Usually you shouldn't change this
# definition.
GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
                $(GTEST_DIR)/include/gtest/internal/*.h

all: $(BUILD)/test

$(GTEST_DIR)/$(BUILD)/libgtest.a:
	cd .. && $(MAKE) googletest

$(BUILD)/test: test.cpp $(GTEST_HEADERS) \
	$(GTEST_DIR)/$(BUILD)/libgtest.a
	$(CDIR) && $(MPILD) $(CINC) $(COPT) $(FUSE_LINKER_PLUGIN) $< $(GTEST_DIR)/$(BUILD)/libgtest.a -Wl,--whole-archive $(INSTALL)/lib/libfrontend.a -Wl,--no-whole-archive -o $@ -L$(INSTALL)/lib -Wl,-rpath=$(INSTALL)/lib -Wl,--unresolved-symbols=ignore-in-object-files -Wl,-Bstatic $(addprefix -Wl$(COMMA),$(LINK)) $(LINK_ACCELERATOR) -Wl,-Bdynamic -Ltbb/build/release -ltbb -ltbbmalloc -ldl $(MPILDFLAGS) -lrt -lgfortran -lstdc++ -lpostprocess -Wl,-rpath=$(ROOT)/$(BUILD) -lpthread -Wl,--wrap=main

run: $(BUILD)/test
	$(ROOT)/$(BUILD)/test

clean:
	rm -rf $(BUILD)

