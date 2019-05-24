from numpy.distutils.core import setup, Extension

symph_ext = Extension(name = "symph",
                      sources = ["FModules/symdynph_gq_new.f90", "FModules/symm_base.f90", 
                                 "FModules/sgam_ph.f90", "FModules/invmat.f90", "FModules/set_asr.f90",
                                 "FModules/error_handler.f90", "FModules/io_global.f90",
                                 "FModules/flush_unit.f90", "FModules/symvector.f90",
                                 "FModules/fc_supercell_from_dyn.f90",
                                 "FModules/set_tau.f90", "FModules/cryst_to_car.f90",
                                 "FModules/recips.f90", "FModules/q2qstar_out.f90",
                                 "FModules/rotate_and_add_dyn.f90", "FModules/trntnsc.f90",
                                 "FModules/star_q.f90", "FModules/eqvect.f90",
                                 "FModules/symm_matrix.f90", "FModules/from_matdyn.f90",
                                 "FModules/interp.f90", "FModules/q_gen.f90", "FModules/smallgq.f90"],
                      libraries= ["lapack", "blas"],
                      extra_f90_compile_args = ["-cpp"]
                      )

cc_modules_ext = Extension(name = "cc_linalg",
                      sources = ["CModules/LinAlg.c", "CModules/wrapper.c"]
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
       scripts = ["scripts/symmetrize_dynmat.x", "scripts/cellconstructor_test.py"],
       ext_modules = [symph_ext, cc_modules_ext]
       )

def readme():
    with open("README.md") as f:
        return f.read()
