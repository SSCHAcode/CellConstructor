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
   conda create \-n sscha\_env python=3.10
   ```

2. Activate the environment:  
   You must activate the environment before proceeding with the installation.
   ```bash  
   conda activate sscha\_env
   ```

## **Step 2: Install Dependencies**

Now we will install the necessary build tools, libraries, and Python packages into our active Conda environment.

1. Install system and build dependencies with Conda:  
   This command installs the core libraries needed for compilation (BLAS, LAPACK, MPI) and the build system itself (Meson, Ninja).
   ```bash  
   conda install \-c conda-forge openblas lapack openmpi meson ninja
   ```

   *Note: openblas provides optimized implementations for blas and lapack.*  
2. Install Python dependencies with pip:  
   Use pip to install the Python packages listed in requirements.txt.
   ```bash  
   pip install \-r requirements.txt
   ```

## **Step 3: Clone the Repository**

If you haven't already, clone the project repository from its source.
```bash
git clone \<URL\_OF\_YOUR\_REPOSITORY\>  
cd CellConstructor  \# Or your repository's root directory
```

## **Step 4: Compile and Install the Project**

With the environment set up and all dependencies installed, you can now build and install CellConstructor using pip in editable mode or standard mode. The pyproject.toml file tells pip to use meson-python as the build backend.

### **Recommended Method: Editable Install**

An "editable" install is highly recommended for developers. It allows you to modify the source code and have the changes reflected immediately without needing to reinstall.
```bash  
pip install \-e .
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
   pip install . \--config-settings=--setup-args=-Duse\_mkl=true
   ```

   Or for an editable install:  
   ```bash
   pip install \-e . \--config-settings=--setup-args=-Duse\_mkl=true
   ```

## **How to Verify the Installation**

After the installation is complete, you can verify that the modules were built correctly by opening a Python interpreter and importing them:
```bash  
python

\>\>\> import cellconstructor  
\>\>\> import cc\_linalg  
\>\>\> import symph  
\>\>\> import secondorder  
\>\>\> import thirdorder

\>\>\> print("Successfully imported all CellConstructor modules\!")
```

If no ImportError messages appear, the installation was successful.
