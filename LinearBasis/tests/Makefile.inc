include $(LOCATION)/Makefile.inc

.PHONY: googletest run

TESTS = $(sort $(dir $(wildcard ./*/)))
TESTS := $(TESTS:./%=%)
TESTS := $(TESTS:%/=%)
TESTS := $(filter-out googletest, $(TESTS))

TESTS_BUILD = $(addsuffix .build, $(TESTS))
TESTS_RUN = $(addsuffix .run, $(TESTS))
TESTS_CLEAN = $(addsuffix .clean, $(TESTS))

%.build: googletest
	cd $(basename $@) && $(MAKE)

%.run: %.build
	cd $(basename $@) && $(MAKE) run

%.clean:
	cd $(basename $@) && $(MAKE) clean

all: $(TESTS_BUILD)

run: $(TESTS_RUN)

clean: $(TESTS_CLEAN)

