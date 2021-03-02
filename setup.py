from numpy.distutils.core import setup, Extension
import sys

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
                                 "FModules/interp.f90", "FModules/q_gen.f90", "FModules/smallgq.f90",
                                 "FModules/symmetry_high_rank.f90", 
                                 "FModules/unwrap_tensors.f90",
                                 "FModules/get_latvec.f90",
                                 "FModules/contract_two_phonon_propagator.f90",
                                 "FModules/get_q_grid_fast.f90",
                                 "FModules/kind.f90",
                                 "FModules/constants.f90",
                                 "FModules/eff_charge_interp.f90",],
                      libraries= ["lapack", "blas"],
                      extra_f90_compile_args = ["-cpp"]
                      )


secondorder_ext = Extension(name = "secondorder",
                      sources = ["FModules/second_order_centering.f90",
                                 "FModules/second_order_ASR.f90"],
                      libraries= ["lapack", "blas"],
                      extra_f90_compile_args = ["-cpp"]
                      )


thirdorder_ext = Extension(name = "thirdorder",
                      sources = ["FModules/third_order_centering.f90",
                                 "FModules/third_order_ASR.f90",
                                 "FModules/third_order_interpol.f90",
                                 "FModules/third_order_dynbubble.f90"],
                      libraries= ["lapack", "blas"],
                      extra_f90_compile_args = ["-cpp"]
                      )



# The C module extension actually depeds on the python version
WRAPPER = "CModules/wrapper3.c"
if sys.version_info[0] < 3:
    print("Running python2, changing the C wrapper")
    WRAPPER = "CModules/wrapper.c"
    
cc_modules_ext = Extension(name = "cc_linalg",
                      sources = ["CModules/LinAlg.c", WRAPPER]
                      )




setup( name = "CellConstructor",
       version = "1.0alpha2",
       description = "Python utilities that is interfaced with ASE for atomic crystal analysis",
       author = "Lorenzo Monacelli",
       url = "https://github.com/mesonepigreco/CellConstructor",
       packages = ["cellconstructor"],
       package_dir = {"cellconstructor": "cellconstructor"},
       package_data = {"cellconstructor": ["SymData/*.dat"]},
       install_requires = ["numpy", "ase", "scipy"],
       license = "MIT",
       include_package_data = True,
       scripts = ["scripts/symmetrize_dynmat.py", "scripts/cellconstructor_test.py"],
       ext_modules = [symph_ext, cc_modules_ext, thirdorder_ext, secondorder_ext]
       )

def readme():
    with open("README.md") as f:
        return f.read()
