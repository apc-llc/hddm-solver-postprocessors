LOCATION = ../../..

include $(LOCATION)/Makefile.inc

include Makefile.inc

googletest:
	cd $@/$@ && mkdir -p $(BUILD) && cd $(BUILD) && CC="$(MPICC)" CXX="$(MPICXX)" cmake $(ROOT)/$@/$@ && make -j12 && \
	mkdir -p $(INSTALL)/lib && cp -rf libgtest.a $(INSTALL)/lib && cp -rf libgtest_main.a $(INSTALL)/lib

clean: $(TESTS_CLEAN)
	cd googletest/googletest && rm -rf $(BUILD)

