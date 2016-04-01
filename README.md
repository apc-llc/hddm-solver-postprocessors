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

### Build interpolators for Intel Xeon Phi (k1om) target on Ubuntu Linux

```
$ cd LinearBasis/MIC
$ make target=native
```

### Build interpolators for Intel Xeon Phi (k1om) target on Anselm

```
$ qsub -q qmic -I -A DD-16-7
$ cd LinearBasis/MIC
$ make target=native
```

