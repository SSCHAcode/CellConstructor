from numpy.distutils.core import setup, Extension

symph_ext = Extension(name = "symph",
                      sources = ["FModules/symdynph_gq_new.f90", "FModules/symm_base.f90", 
                                 "FModules/sgam_ph.f90", "FModules/invmat.f90", "FModules/set_asr.f90",
                                 "FModules/error_handler.f90", "FModules/io_global.f90",
                                 "FModules/flush_unit.f90", "FModules/symvector.f90"],
                      libraries= ["lapack", "blas"],
                      extra_f90_compile_args = ["-cpp"]
                      )


setup( name = "CellConstructor",
       version = "0.1",
       description = "Python utilities that is interfaced with ASE for atomic crystal analysis",
       author = "Lorenzo Monacelli",
       url = "https://github.com/mesonepigreco/CellConstructor",
       packages = ["cellconstructor"],
       package_dir = {"cellconstructor": "cellconstructor"},
       package_data = {"cellconstructor": ["SymData/*.dat"]},
       install_requires = ["numpy", "ase", "scipy", "gfortran", "lapack"],
       license = "MIT",
       include_package_data = True,
       ext_modules = [symph_ext]
       )

def readme():
    with open("README.md") as f:
        return f.read()