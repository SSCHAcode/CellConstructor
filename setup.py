from numpy.distutils.core import setup, Extension

symph_ext = Extension(name = "symph",
                      sources = ["FModules/symdynph_gq_new.f90"],
                      )


setup( name = "CellConstructor",
       version = "0.1",
       description = "Python utilities that is interfaced with ASE for atomic crystal analysis",
       author = "Lorenzo Monacelli",
       url = "https://github.com/mesonepigreco/CellConstructor",
       packages = ["cellconstructor"],
       package_dir = {"cellconstructor": "cellconstructor"},
       package_data = {"cellconstructor": ["SymData/*.dat"]},
       install_requires = ["numpy", "ase", "scipy"],
       license = "MIT",
       include_package_data = True,
       ext_modules = [symph_ext]
       )

def readme():
    with open("README.md") as f:
        return f.read()