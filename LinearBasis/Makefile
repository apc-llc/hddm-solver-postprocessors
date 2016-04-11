LOCATION = ../..

include Makefile.inc

.PHONY: cpu gpu mic

all: cpu mic

cpu:
	cd $@ && $(MAKE) -j$(NJOBS)

mic:
	cd $@ && $(MAKE) -j$(NJOBS) target=native

clean:
	cd mic && $(MAKE) clean

