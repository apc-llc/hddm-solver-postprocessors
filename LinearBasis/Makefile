LOCATION = ../..

include $(LOCATION)/Makefile.inc

CINC += -I../include -I../../include
CDIR = mkdir -p $(shell dirname $@)

TARGETS = cpu cuda mic

.PHONY: $(TARGETS)

all: $(TARGETS)

$(INSTALL)/lib/liblinearbasis.a: $(BUILD)/parameters.o
	$(CDIR) && ar rcs $@ $^

$(BUILD)/parameters.o: src/parameters.cpp $(LOCATION)/include/check.h include/parameters.h include/parameters.c
	$(CDIR) && $(MPICXX) $(CINC) $(COPT) -c $< -o $@

cpu: $(INSTALL)/lib/liblinearbasis.a
	cd $@ && $(MAKE) -j$(NJOBS)

cuda: $(INSTALL)/lib/liblinearbasis.a
	cd $@ && $(MAKE) -j$(NJOBS)

mic: $(INSTALL)/lib/liblinearbasis.a
	cd $@ && $(MAKE) -j$(NJOBS) target=native

clean:
	rm -rf $(BUILD) && \
	cd cpu && $(MAKE) clean && cd .. && \
	cd cuda && $(MAKE) clean && cd .. && \
	cd mic && $(MAKE) clean && cd ..

