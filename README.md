### Get the codebase on Ubuntu Linux

Clone the source tree and submodules:

```
$ git clone https://github.com/apc-llc/hddm-solver-postprocessors.git
$ cd hddm-solver-postprocessors
$ git submodule init
$ git submodule update
```

### Get the codebase on Anselm

First, load the git module, since the system-default is too old:

```
$ module load git
$ git clone https://github.com/apc-llc/hddm-solver-postprocessors.git
$ cd hddm-solver-postprocessors
$ git submodule init
$ git submodule update
```

### Build postprocessors for Intel Xeon Phi (k1om) target on Ubuntu Linux

```
$ cd hddm-solver-postprocessors
$ cd LinearBasis/MIC
$ make target=native
```

### Build postprocessors for Intel Xeon Phi (k1om) target on Anselm

```
$ qsub -q qmic -I -A DD-16-7
$ cd hddm-solver-postprocessors
$ cd LinearBasis/MIC
$ make target=native
```

