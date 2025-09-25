# **CellConstructor Installation Guide**

This guide provides step-by-step instructions to compile and install the CellConstructor library and its dependencies. The project uses the Meson build system to compile C and Fortran extensions for Python.  
We recommend using a Conda environment to manage the dependencies.

## **Prerequisites**

You will need a working C and Fortran compiler on your system, as well as git to clone the repository.  
On a Debian/Ubuntu based system, you can install them with:
```bash
sudo apt-get update  
sudo apt-get install build-essential gfortran git
```

On a RedHat/Fedora/CentOS system, you can install them with:  
```bash
sudo dnf groupinstall "Development Tools"  
sudo dnf install gfortran git
```

## **Step 1: Set up the Conda Environment**

First, we will create a dedicated Conda environment to ensure all dependencies are isolated and have the correct versions.

1. Create a new Conda environment:  
   We will name the environment sscha\_env and use Python 3.10. You can choose a different Python version if needed (the project requires \>=3.8).
   ```bash  
   conda create -n sscha_env python=3.10
   ```

2. Activate the environment:  
   You must activate the environment before proceeding with the installation.
   ```bash  
   conda activate sscha_env
   ```

## **Step 2: Install Dependencies**

Now we will install the necessary build tools, libraries, and Python packages into our active Conda environment.

1. Install system and build dependencies with Conda:  
   This command installs the core libraries needed for compilation (BLAS, LAPACK, MPI) and the build system itself (Meson, Ninja).
   ```bash  
   conda install -c conda-forge openblas lapack openmpi meson ninja
   ```

   *Note: openblas provides optimized implementations for blas and lapack.*  
2. Install Python dependencies with pip:  
   Use pip to install the Python packages listed in requirements.txt.
   ```bash  
   pip install -r requirements.txt
   ```

## **Step 3: Clone the Repository**

If you haven't already, clone the project repository from its source.
```bash
git clone <URL_OF_REPOSITORY>  
cd CellConstructor  # Or your repository's root directory
```

## **Step 4: Compile and Install the Project**

With the environment set up and all dependencies installed, you can now build and install CellConstructor using pip in editable mode or standard mode. The pyproject.toml file tells pip to use meson-python as the build backend.

### **Recommended Method: Editable Install**

An "editable" install is highly recommended for developers. It allows you to modify the source code and have the changes reflected immediately without needing to reinstall.
```bash  
pip install -e .
```

### **Standard Method: Regular Install**

This method is suitable for users who just want to use the library without modifying its source code.
```bash  
pip install .
```

The pip command will automatically invoke Meson, which will configure the project, compile the C and Fortran extensions, and install the complete package into your Conda environment.

## **Build Options**

The project includes custom build options that can be configured.

### **Using Intel MKL**

If you have the Intel Math Kernel Library (MKL) installed and wish to use it for BLAS/LAPACK, you can enable it during the build process.

1. **Ensure MKL is installed in your Conda environment:**  
   ```bash
   conda install mkl mkl-devel
   ```

2. Install the project passing the use\_mkl option:  
   You can pass Meson options through pip's \--config-settings flag.  
   ```bash
   pip install . --config-settings=--setup-args=-Duse_mkl=true
   ```

   Or for an editable install:  
   ```bash
   pip install -e . --config-settings=--setup-args=-Duse_mkl=true
   ```

## **How to Verify the Installation**

After the installation is complete, you can verify that the modules were built correctly by opening a Python interpreter and importing them:
```bash  
python

>>> import cellconstructor  
>>> import cc_linalg  
>>> import symph  
>>> import secondorder  
>>> import thirdorder

>>> print("Successfully imported all CellConstructor modules\!")
```

If no ImportError messages appear, the installation was successful.

### How to Install with a Specific Compiler Path

This guide explains how to compile and install this Meson-based Python project when you need to use specific C, C++, or Fortran compilers that are not the default ones in your system's PATH.
## Prerequisites

Before you begin, ensure you have the following installed:

*    Python 3.8+ and pip

*    The build dependencies for this project, which can be installed via pip:

```bash
    pip install meson ninja
```

*    The C, C++, and Fortran compilers you intend to use.

## Method 1: Using Environment Variables (Recommended for most cases)

Meson automatically detects compilers using standard environment variables. You can set these variables before running the installation command. This is the simplest way to specify a compiler for a single build.

The key variables are:

```bash
    CC: Specifies the C compiler executable.

    CXX: Specifies the C++ compiler executable.

    FC: Specifies the Fortran compiler executable.
```

# Step-by-Step Instructions

1.    Open your terminal. All commands must be run in the same session, as environment variables are typically not permanent.

2.    Set the environment variables to point to your desired compilers.

    Example for C (using a specific gcc):

```bash
    export CC=/path/to/my/custom/gcc
```

    Example for Fortran (using a specific gfortran):

```bash
    export FC=/path/to/my/custom/gfortran
```

3.    Combine them as needed. For this project, you will likely need to set CC and FC.

```bash
    # Example using compilers from a specific toolchain
    export CC=/usr/local/bin/gcc-11
    export FC=/usr/local/bin/gfortran-11
```

4.    Run the pip installation. With the variables set, run pip install from the project's root directory. pip will pass the environment variables down to Meson.

```bash
    # Ensure you are in the project's root directory (where pyproject.toml is)
    pip install .
```

## Method 2: Using a Meson Cross File (Advanced & Reproducible)

For a more permanent or reproducible setup (e.g., in CI/CD pipelines or complex environments), a Meson "cross file" is the best practice. This file explicitly defines the toolchain.

# Step-by-Step Instructions

1.    Create a cross file. In your project's root directory, create a file named native-toolchain.ini.

2.    Edit the file to specify the paths to your compilers in the [binaries] section.

    Example native-toolchain.ini:

```bash
    # native-toolchain.ini
    [binaries]
    c = '/path/to/my/custom/gcc'
    fortran = '/path/to/my/custom/gfortran'
```

3.    Run the pip installation with meson-args.

```bash
    pip install . --config-settings=meson-args="--native-file=native-toolchain.ini"
```


### Advanced Configuration: Using Meson Build Options

Meson allows you to configure the build using options, which act like on/off switches or settings. You can pass these options during installation using the same --config-settings flag. This is useful for enabling features or changing the build type.

The general syntax is -D<option_name>=<value>.

# Example 1: Creating a Debug Build

By default, the project is built in release mode for performance. To build with debugging symbols and fewer optimizations (useful for development), you can change the built-in buildtype option.

```bash
pip install . --config-settings=meson-args="-Dbuildtype=debug"
```

# Example 2: Enabling Project-Specific Features (like MKL)

This project has an option to use Intel's Math Kernel Library (MKL) instead of standard BLAS/LAPACK. This is controlled by the use_mkl option defined in meson.options.

To enable it, set it to true during installation:

```bash
pip install . --config-settings=meson-args="-Duse_mkl=true"
```

## Combining Multiple Options

You can combine multiple Meson arguments by separating them with commas inside the string.

# Example: A debug build using MKL:

```bash
pip install . --config-settings=meson-args="-Dbuildtype=debug,-Duse_mkl=true"
```

# Important: Reconfiguring a Build

If you have already built the project and want to change a configuration option, pip might not automatically rebuild it. It is best practice to first clean any previous build artifacts.

You can force a clean rebuild by uninstalling the package and manually deleting the build directory before running the new install command.

1. Uninstall the package
```bash
pip uninstall CellConstructor
```
2. (Optional but recommended) Remove the build directory
```bash
rm -rf build/
```
3. Reinstall with the new options
```bash
pip install . --config-settings=meson-args="-Dnew_option=value"
```
