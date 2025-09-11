project('CellConstructor', 'fortran', 'c',
  version: '1.5.0',
  license: 'GPL',
  default_options: ['fortran_args=-O2']
)

# --- System and Python Dependencies ---
python = import('python')
python_installation = python.find_installation()
numpy_incdir = run_command(
  python_installation, '-c', 'import numpy; print(numpy.get_include())',
  check: true
).stdout().strip()

# Obtenemos la ruta padre para incluir los headers de numpy anidados
numpy_parent_incdir = run_command(
  python_installation, '-c', 'import os, numpy; print(os.path.dirname(numpy.get_include()))',
  check: true
).stdout().strip()

include_dirs = include_directories(numpy_incdir, numpy_parent_incdir, '.')

f2py_exe = find_program('f2py', required: true)

# Busca las librerías de sistema BLAS, LAPACK y OpenMP
lapack_dep = dependency('lapack', required: true)
blas_dep = dependency('blas', required: true)
openmp_dep = dependency('openmp', required: true)

# --- Definición de cada extensión de Python (Fortran) ---
# Hemos vuelto a una estructura más simple y robusta.

# Extensión 'symph'
symph_fortran_src = files(
    'FModules/symdynph_gq_new.f90', 'FModules/symm_base.f90', 'FModules/sgam_ph.f90',
    'FModules/invmat.f90', 'FModules/set_asr.f90', 'FModules/error_handler.f90',
    'FModules/io_global.f90', 'FModules/flush_unit.f90', 'FModules/symvector.f90',
    'FModules/fc_supercell_from_dyn.f90', 'FModules/set_tau.f90',
    'FModules/cryst_to_car.f90', 'FModules/recips.f90', 'FModules/q2qstar_out.f90',
    'FModules/rotate_and_add_dyn.f90', 'FModules/trntnsc.f90', 'FModules/star_q.f90',
    'FModules/eqvect.f90', 'FModules/symm_matrix.f90', 'FModules/from_matdyn.f90',
    'FModules/interp.f90', 'FModules/q_gen.f90', 'FModules/smallgq.f90',
    'FModules/symmetry_high_rank.f90', 'FModules/unwrap_tensors.f90',
    'FModules/get_latvec.f90', 'FModules/contract_two_phonon_propagator.f90',
    'FModules/get_q_grid_fast.f90', 'FModules/kind.f90', 'FModules/constants.f90',
    'FModules/eff_charge_interp.f90', 'FModules/get_translations.f90',
    'FModules/get_equivalent_atoms.f90'
)

symph_wrapper = custom_target(
    'symph_f2py_wrapper',
    input: symph_fortran_src,
    output: 'symphmodule.c',
    command: [
        f2py_exe,
        '-m', 'symph',
        '@INPUT@',
        '--quiet',
        '--lower'
    ],
    build_by_default: true
)

py_symph_ext = python_installation.extension_module(
    'symph',
    [symph_wrapper, symph_fortran_src],
    install: true,
    include_directories: include_dirs,
    dependencies: [lapack_dep, blas_dep, openmp_dep],
    include_directories: include_directories(numpy_incdir, numpy_parent_incdir, '.'),
    link_args: ['-fopenmp'],
    fortran_args: ['-cpp', '-fopenmp'],
    install_rpath: '$ORIGIN/..'
)

# ---

# Extensión 'secondorder'
secondorder_fortran_src = files(
    'FModules/second_order_centering.f90', 'FModules/second_order_ASR.f90'
)

secondorder_wrapper = custom_target(
    'secondorder_f2py_wrapper',
    input: secondorder_fortran_src,
    output: 'secondordermodule.c',
    command: [f2py_exe, '-m', 'secondorder', '@INPUT@', '--quiet', '--lower'],
    build_by_default: true
)

py_secondorder_ext = python_installation.extension_module(
    'secondorder',
    [secondorder_wrapper, secondorder_fortran_src],
    install: true,
    include_directories: include_dirs,
    dependencies: [lapack_dep, blas_dep],
    include_directories: include_directories(numpy_incdir, numpy_parent_incdir, '.'),
    fortran_args: ['-cpp'],
    install_rpath: '$ORIGIN/..'
)

# ---

# Extensión 'thirdorder'
thirdorder_fortran_src = files(
    'FModules/third_order_centering.f90', 'FModules/third_order_ASR.f90',
    'FModules/third_order_interpol.f90', 'FModules/third_order_dynbubble.f90'
)

thirdorder_wrapper = custom_target(
    'thirdorder_f2py_wrapper',
    input: thirdorder_fortran_src,
    output: 'thirdordermodule.c',
    command: [f2py_exe, '-m', 'thirdorder', '@INPUT@', '--quiet', '--lower'],
    build_by_default: true
)

py_thirdorder_ext = python_installation.extension_module(
    'thirdorder',
    [thirdorder_wrapper, thirdorder_fortran_src],
    install: true,
    include_directories: include_dirs,
    dependencies: [lapack_dep, blas_dep],
    include_directories: include_directories(numpy_incdir, numpy_parent_incdir, '.'),
    fortran_args: ['-cpp'],
    install_rpath: '$ORIGIN/..'
)

# ---

# Extensión 'thermal_conductivity'
cond_fortran_src = files(
    'FModules/get_scattering_q_grid.f90', 'FModules/third_order_cond.f90',
    'FModules/third_order_cond_centering.f90', 'FModules/get_lf.f90'
)

cond_wrapper = custom_target(
    'thermal_conductivity_f2py_wrapper',
    input: cond_fortran_src,
    output: 'thermal_conductivitymodule.c',
    command: [f2py_exe, '-m', 'thermal_conductivity', '@INPUT@', '--quiet', '--lower'],
    build_by_default: true
)

py_cond_ext = python_installation.extension_module(
    'thermal_conductivity',
    [cond_wrapper, cond_fortran_src],
    include_directories: include_dirs,
    dependencies: [lapack_dep, blas_dep, openmp_dep],
    include_directories: include_directories(numpy_incdir, numpy_parent_incdir, '.'),
    link_args: ['-fopenmp'],
    fortran_args: ['-cpp', '-fopenmp'],
    install: true,
    install_rpath: '$ORIGIN/..'
)

# --- Definición de la extensión C 'cc_linalg' ---
wrapper_file = ''
if python_installation.version().version_compare('<3.0')
  wrapper_file = 'CModules/wrapper.c'
else
  wrapper_file = 'CModules/wrapper3.c'
endif

cc_modules_sources = files('CModules/LinAlg.c', wrapper_file)
py_cc_linalg_ext = python_installation.extension_module(
  'cc_linalg',
  cc_modules_sources,
  include_directories: include_directories(numpy_incdir, numpy_parent_incdir),
  install: true,
  install_rpath: '$ORIGIN/..'
)

# --- Instalación del paquete de Python 'cellconstructor' ---
install_data(
  'cellconstructor/__init__.py', 'cellconstructor/AnharmonicForceFields.py', 'cellconstructor/calculators.py',
  'cellconstructor/Methods.py', 'cellconstructor/Phonons.py', 'cellconstructor/Spectral.py',
  'cellconstructor/ThermalConductivity.py', 'cellconstructor/Units.py', 'cellconstructor/Bands.py',
  'cellconstructor/ForceTensor.py', 'cellconstructor/Manipulate.py', 'cellconstructor/Moro_object.py',
  'cellconstructor/Settings.py', 'cellconstructor/Structure.py', 'cellconstructor/symmetries.py',
  'cellconstructor/Timer.py',
  install_dir: python_installation.get_install_dir() / 'cellconstructor',
)

install_data(
  'cellconstructor/SymData/15.dat', 'cellconstructor/SymData/36_red.dat', 'cellconstructor/SymData/36.dat',
  'cellconstructor/SymData/60.dat', 'cellconstructor/SymData/64.bcs', 'cellconstructor/SymData/64.dat',
  'cellconstructor/SymData/convert_sym.py',
  install_dir: python_installation.get_install_dir() / 'cellconstructor' / 'SymData'
)

# --- Instalación de los scripts ejecutables ---
python_installation.install_sources([
  'scripts/symmetrize_dynmat.py',
  'scripts/cellconstructor_test.py',
  'scripts/view_scf_atoms.py'
])

# Set the tests by pytest.
pytest_exe = find_program('pytest', required: false)

if pytest_exe.found()
  test('pytest', pytest_exe,
    args : ['-v'],
    workdir : meson.project_source_root()
  )
else
  message('pytest no encontrado; se omiten tests con pytest.')
endif
