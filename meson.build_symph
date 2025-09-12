project('symph',
        ['c', 'fortran'],
        version : '0.1',
        meson_version: '>= 1.1.0',
        default_options : [
                            'warning_level=1',
                            'buildtype=release',
                          'fortran_args=-O2', 'fortran_args=-cpp'])
fc = meson.get_compiler('fortran')

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

incdir_numpy = run_command(py,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

incdir_f2py = run_command(py,
    ['-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

inc_np = include_directories(incdir_numpy)
np_dep = declare_dependency(include_directories: inc_np)

incdir_f2py = incdir_numpy / '..' / '..' / 'f2py' / 'src'
inc_f2py = include_directories(incdir_f2py)
fortranobject_c = incdir_f2py / 'fortranobject.c'

inc_np = include_directories(incdir_numpy, incdir_f2py)
# gh-25000
quadmath_dep = fc.find_library('quadmath', required: false)




py.extension_module('symph',
                     [
                     'constants.f90',
                                          'error_handler.f90',
                                          'get_latvec.f90',
                                          'io_global.f90',
                                          'rotate_and_add_dyn.f90',
                                          'smallgq.f90',
                                          'symm_matrix.f90',
                                          'contract_two_phonon_propagator.f90',
                                          'fc_supercell_from_dyn.f90',
                                          'get_q_grid_fast.f90',
                                          'kind.f90',
                                          'star_q.f90',
                                          'symvector.f90',
                                          'cryst_to_car.f90',
                                          'flush_unit.f90',
                                          'get_translations.f90',
                                          'q2qstar_out.f90',
                                          'set_asr.f90',
                                          'symdynph_gq_new.f90',
                                          'trntnsc.f90',
                                          'eff_charge_interp.f90',
                                          'from_matdyn.f90',
                                          'interp.f90',
                                          'q_gen.f90',
                                          'set_tau.f90',
                                          'symm_base.f90',
                                          'unwrap_tensors.f90',
                                          'eqvect.f90',
                                          'get_equivalent_atoms.f90',
                                          'invmat.f90',
                                          'recips.f90',
                                          'sgam_ph.f90',
                                          'symmetry_high_rank.f90',
                                          'symphmodule.c',
                                          'symph-f2pywrappers2.f90',
                                          'symph-f2pywrappers.f',
                     fortranobject_c
                     ],
                     include_directories: [
                     inc_np,

                     ],
                     dependencies : [
                     py_dep,
                     quadmath_dep,
                     dependency('mpi')


                     ],
                     install : true)
