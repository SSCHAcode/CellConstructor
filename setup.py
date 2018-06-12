from distutils.core import setup
setup( name = "CellConstructor",
       version = "0.1",
       description = "Python utilities that is interfaced with ASE for atomic crystal analysis",
       author = "Lorenzo Monacelli",
       url = "https://github.com/mesonepigreco/CellConstructor",
       packages = ["cellconstructor"],
       package_dir = {"cellconstructor": "cellconstructor"},
       package_data = {"cellconstructor": ["SymData/*.dat"]},
       install_requires = ["numpy", "ase"],
       license = "MIT",
       include_package_data = True
       )

def readme():
    with open("README.md") as f:
        return f.read()