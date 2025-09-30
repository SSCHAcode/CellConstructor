# CellConstructor

Welcome to the CellConstructor python package!

## What is CellConstructor?
CellConstructor is a Python library that allows for easy manipulation of crystal structures and phonon dynamical matrices.
It is based on Quantum Espresso and wraps many of the utilities available in the Phonon package in Python for post-processing tools.
It is also interfaced with ASE for I/O of standard file formats (.cif, input for common calculators, etc.) and with SPGLIB for a more stable tool for symmetry identification (the Quantum Espresso symmetry library is also available).


## What can I do with CellConstructor?
CellConstructor is a general-purpose library. Some of the things you can do with one single command:

1. Compute Quasi-Harmonic free energies from a dynamical matrix at any temperature.
2. Impose symmetries on a structure, or on a dynamical matrix.
3. Impose the acoustic sum rule on a dynamical matrix
4. Compute the phonon dispersion along high-symmetry paths
5. Extract harmonic, randomly displaced configurations according to the dynamical matrix.

And also many other cool features!

## Requirements

To correctly install and use the package, you need to have
1. python >= 3.8
2. ASE: Atomic Simulation Environment (suggested but not mandatory)
3. numpy
4. scipy
5. A FORTRAN compiler
6. Lapack and BLAS

Suggested, but not required, is the installation of ASE and spglib.
The presence of a valid ASE installation will enable some more features,
like the possibility to load structures by any ASE-supported file format,
or the possibility to export the structures into a valid ASE Atoms class.

Please note that some Fortran libraries require compilation; therefore, the Python header files should be localized during the compilation process. Starting from version 1.5, the compilation is handled by meson and can be performed automatically using `pip`.

For example, using mamba (or conda), you can install all the binary dependencies with:

```
mamba install -c conda-forge python=3.12 ase spglib=2.2 gfortran libblas lapack openmpi openmpi-mpicc pip numpy scipy
```
To use conda, just replace `mamba` with `conda`. Note that `conda` may take a long time to resolve the environment; therefore, mamba or micromamba are recommended.

## Installation

The code can be compiled and installed from the PyPi repository with

```
pip install CellConstructor
```

Or, alternatively, directly by downloading the GitHub repository and running the following command in the directory containing the code

```
pip install .
```

This will start the automatic compilation. If some libraries or compilers aren't found, please take a look at the next section.

To test if the installation runs properly, you can just run the examples reported
in the test directory. The Python code in these tests should be
almost self-explanatory and introduce you to the potential of this library.

Please note that all the functions of the library have a proper numpy style
docstring.

You can test the installation using the script:
```bash
cellconstructor_test.py
```

To run the complete testsuite, you can use pytest, running the following command:
```bash
pytest
```

For a complete API documentation, you can compile the documentation inside the UserGuide directory.  
To compile it, use the make utility followed by the frontend.
For example, if you want the HTML version, run:
```bash
make html
```
Inside the UserGuide directory. It will generate a build directory that contains the HTML version of the complete documentation.

## Installation using 'Meson'

Here, we follow the manual step to compile the code using `meson`. These steps are required if you 
want to compute the code on an HPC by exploiting the fast optimized libraries of the HPC nodes.

### Compiling with Meson

To compile and install CellConstructor with Meson, follow these typical steps:

### 1. Change to the Source Directory

First, open a terminal and navigate to the root directory of the project source code. This is where the `meson.build` file is located.

```bash
cd /path/to/source/root/cellconstructor
```


### 2. Configure the Build Directory

Create and configure a build directory by running:

```bash
meson setup builddir
```

or if you are in a conda env (the best option for a local installation):
```bash
meson setup builddir --prefix=$CONDA_PREFIX
```

if you want to use Intel MKL:
```bash
setup builddir -Duse_mkl=true
```

This command sets up a separate build directory (`builddir`) where all compiled files and build artifacts will be placed, keeping the source directory clean. After this, change into the build directory:

```bash
cd builddir
```


### 3. Compile the Project

Once inside the build directory, compile the project using:

```bash
meson compile
```

This will compile the source code according to the configuration from the previous step.

### 4. Run Tests (Optional)

The project includes tests, you need to install pytest to work. You can run them with:

```bash
meson test
```

This step helps verify that the build works correctly.

### 5. Install the Project (Optional)

To install the compiled binaries, libraries, and other files system-wide (or to a custom location), run:


```bash
meson install
```

or

```bash
sudo meson install
```

You may need superuser privileges (hence `sudo`) to install into system directories.

***

Following these steps will help you successfully compile, test, and install SSCHA with Meson as its build system.
