LOCATION = ../..

include $(LOCATION)/Makefile.inc

CINC += -I../include -I../../include
CDIR = mkdir -p $(shell dirname $@)

TARGETS = cpu

.PHONY: $(TARGETS)

all: $(TARGETS)

$(INSTALL)/lib/libpolybasis.a: $(BUILD)/parameters.o
	$(CDIR) && ar rcs $@ $^

$(BUILD)/parameters.o: src/parameters.cpp $(LOCATION)/include/check.h include/parameters.h include/parameters.c
	$(CDIR) && $(MPICXX) $(CINC) $(COPT) -c $< -o $@

cpu: $(INSTALL)/lib/libpolybasis.a
	cd $@ && $(MAKE) -j$(NJOBS)

gpu: $(INSTALL)/lib/libpolybasis.a
	cd $@ && $(MAKE) -j$(NJOBS)

mic: $(INSTALL)/lib/libpolybasis.a
	cd $@ && $(MAKE) -j$(NJOBS) target=native

clean:
	rm -rf $(BUILD) $(INSTALL)/lib/libpolybasis.a && cd mic && $(MAKE) clean

