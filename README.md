### Get the codebase on Ubuntu Linux

Clone the source tree and submodules:

```
$ git clone https://github.com/apc-llc/hddm-solver-interpolators.git
$ cd hddm-solver-interpolators
$ git submodule init
$ git submodule update
$ cd liboffloadmic
$ git submodule init
$ git submodule update
```

### Get the codebase on Anselm

First, load the git module, since the system-default is too old:

```
$ module load git
$ git clone https://github.com/apc-llc/hddm-solver-interpolators.git
$ cd hddm-solver-interpolators
$ git submodule init
$ git submodule update
$ cd liboffloadmic
$ git submodule init
$ git submodule update
```

### Build interpolators for Intel Xeon Phi (k1om) target

```
$ cd gcc-5.1.1-knc/
$ mkdir build/
$ cd build/
$ export PATH=/opt/mpss/3.4.2/sysroots/x86_64-mpsssdk-linux/usr/bin/k1om-mpss-linux/:$PATH
$ sudo ln -s /opt/mpss/3.4.2/sysroots/k1om-mpss-linux/usr/lib64 /opt/mpss/3.4.2/sysroots/k1om-mpss-linux/usr/lib
$ ../configure --build=x86_64-linux-gnu --host=x86_64-linux-gnu --target=k1om-mpss-linux --prefix=$(pwd)/../../install/host --disable-silent-rules --disable-dependency-tracking --with-ld=/opt/mpss/3.4.2/sysroots/x86_64-mpsssdk-linux/usr/bin/k1om-mpss-linux/k1om-mpss-linux-ld --with-as=/opt/mpss/3.4.2/sysroots/x86_64-mpsssdk-linux/usr/bin/k1om-mpss-linux/k1om-mpss-linux-as --enable-shared --enable-languages=c,c++ --enable-threads=posix --disable-multilib --enable-c99 --enable-long-long --enable-symvers=gnu --enable-libstdcxx-pch --program-prefix=k1om-mpss-linux- --enable-target-optspace --enable-lto --disable-bootstrap --with-system-zlib --with-linker-hash-style=gnu --enable-cheaders=c_global --with-local-prefix=/opt/mpss/3.4.2/sysroots/k1om-mpss-linux/usr --with-sysroot=/opt/mpss/3.4.2/sysroots/k1om-mpss-linux/ --disable-libunwind-exceptions --disable-libssp --disable-libgomp --disable-libmudflap --enable-nls --enable-__cxa_atexit --disable-libitm
$ make -j64
$ make -j64 install
$ export PATH=$(pwd)/../../install/host/bin:$PATH
$ cd ..
$ cd liboffloadmic/
$ git submodule init
$ git submodule update
$ make target=native INSTALL_PREFIX=$(pwd)/../install MIC_PREFIX=k1om-mpss-linux-
```

