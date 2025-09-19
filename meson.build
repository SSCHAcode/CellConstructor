project('CellConstructor',
  ['c','fortran'],
  version: '1.5.0',
  license: 'GPL',
  meson_version: '>= 1.1.0', # <- set min version of meson.
  default_options : [
        'warning_level=1',
        'buildtype=release',
        'fortran_args=-O2',
        'fortran_args=-cpp'
    ]
    )

# --- Compilation options ---
use_mkl = get_option('use_mkl')

# --- System and Python Dependencies ---
# Find the necessary installations
py = import('python').find_installation(pure: false)
py_dep = py.dependency()
fc = meson.get_compiler('fortran')

# Additional dependencies

# The LAPACK and BLAS libraries are essential for Fortran matrix operations.
#lapack_dep = dependency('lapack')
#blas_dep = dependency('blas')
openblas_dep = dependency('openblas', required: false)
if use_mkl
  mkl_dep = dependency('mkl', required: true)
  blas_dep = mkl_dep
  lapack_dep = mkl_dep
else
  lapack_dep = dependency('lapack', required: true)
  blas_dep = dependency('blas', required: true)
endif

# The openmp dependency is necessary for thermal_transport
openmp_dep = dependency('openmp', required: true)

# --- MPI Detection ---
# This overrides the logic in os.environ["MPICC"] and os.popen("%s -show" % mpicc).
# Meson has a built-in MPI module.
mpi_args = []
mpi_link_args = []
mpi_compile_args = []
has_mpi = false

# Attempts to find the MPI dependency.
# You can specify a specific MPI compiler with the 'mpi_compiler' parameter
# or, if you want, a specific Fortran compiler with 'mpi_fortran_compiler'.
# For OpenMPI, IntelMPI, MPICH, etc., Meson usually finds it automatically.
#mpi_dep = dependency('mpi', required: false, language: ['c', 'fortran'])
mpi_dep = dependency('mpi', required: false)

if mpi_dep.found()
  message('MPI environment detected correctly.')
  has_mpi = true
  # Meson handles adding appropriate flags. We just add the define.
  # If you need specific MPI flags beyond what Meson adds automatically,
  # you can get them via mpi_dep.get_compile_args() and mpi_dep.get_link_args()
  # and add them to extra_compile_args/extra_link_args.
  mpi_compile_args += ['-D_MPI']
else
  # Here you can add warning logic if MPI is not found.
  # Meson prints a warning if required: true and it is not found.
  # For required: false, you can print your own warning.
  warning('No MPI compiler found, please ensure MPI is installed and configured.')
  warning('If you wish to activate MPI acceleration, consider setting MPICC environment variable or providing Meson with appropriate flags.')
endif
# Find the quadmath library, if available
quadmath_dep = fc.find_library('quadmath', required: false)

# --- NUMPY CONFIGURATION ---
# Gets the path to the NumPy and f2py header directories using the Python command
incdir_numpy = run_command(py,
    ['-c', 'import numpy; print(numpy.get_include())'],
    check : true
).stdout().strip()

# f2py also requires the fortranobject.h header
incdir_f2py = run_command(py,
    ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

inc_np = include_directories(incdir_numpy, incdir_f2py)
np_dep = declare_dependency(include_directories: inc_np)

inc_f2py = include_directories(incdir_f2py)
fortranobject_c = incdir_f2py / 'fortranobject.c'

# --- END OF NUMPY CONFIGURATION ---
if openblas_dep.found()
  message('openblas environment detected correctly.')
  list_dep = [py_dep, mpi_dep, quadmath_dep, openblas_dep, lapack_dep, blas_dep]
else
  warning('No openblas found.')
  list_dep = [py_dep, mpi_dep, quadmath_dep, lapack_dep, blas_dep]
endif
# --- Definition of each Python extension (Fortran) ---

# 'symph' extension

# Compilation of the Fortran module: symph
fortran_sources_symph = [
    'FModules/constants.f90',
    'FModules/error_handler.f90',
    'FModules/get_latvec.f90',
    'FModules/io_global.f90',
    'FModules/rotate_and_add_dyn.f90',
    'FModules/smallgq.f90',
    'FModules/symm_matrix.f90',
    'FModules/contract_two_phonon_propagator.f90',
    'FModules/fc_supercell_from_dyn.f90',
    'FModules/get_q_grid_fast.f90',
    'FModules/kind.f90',
    'FModules/star_q.f90',
    'FModules/symvector.f90',
    'FModules/cryst_to_car.f90',
    'FModules/flush_unit.f90',
    'FModules/get_translations.f90',
    'FModules/q2qstar_out.f90',
    'FModules/set_asr.f90',
    'FModules/symdynph_gq_new.f90',
    'FModules/trntnsc.f90',
    'FModules/eff_charge_interp.f90',
    'FModules/from_matdyn.f90',
    'FModules/interp.f90',
    'FModules/q_gen.f90',
    'FModules/set_tau.f90',
    'FModules/symm_base.f90',
    'FModules/unwrap_tensors.f90',
    'FModules/eqvect.f90',
    'FModules/get_equivalent_atoms.f90',
    'FModules/invmat.f90',
    'FModules/recips.f90',
    'FModules/sgam_ph.f90',
    'FModules/symmetry_high_rank.f90'
]
## Generate the C wrapper with f2py using a custom target
f2py_symph_target = custom_target('symph-f2py-wrapper',
    input: fortran_sources_symph,
    output: ['symphmodule.c','symph-f2pywrappers.f','symph-f2pywrappers2.f90'],
#    command: [py.full_path(), '-m', 'numpy.f2py', '--backend', 'meson',
#        '--dep', 'mpi', '--quiet', '-c', '@INPUT0@', '-m', 'symph'],
     command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'symph', '--lower'],
    install: false
)
py.extension_module('symph',
   [
    fortran_sources_symph,
    f2py_symph_target, fortranobject_c],
    include_directories: inc_np,
    dependencies: list_dep,
#    link_args: ['-L' + py.get_install_dir() / 'numpy' / 'f2py' / 'src' / 'fortranobject.c', '-lfortranobject'],
    install: true
)

# 'secondorder' extension

# Compilation of the Fortran module: secondorder
fortran_sources_secondorder = [
    'FModules/second_order_centering.f90',
    'FModules/second_order_ASR.f90'
]
## Generate the C wrapper with f2py using a custom target
f2py_secondorder_target = custom_target('secondorder-f2py-wrapper',
    input: fortran_sources_secondorder,
    output: ['secondordermodule.c','secondorder-f2pywrappers2.f90'],
#    command: [py.full_path(), '-m', 'numpy.f2py', '--backend', 'meson',
#        '--dep', 'mpi', '--quiet', '-c', '@INPUT0@', '-m', 'secondorder'],
    command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'secondorder', '--lower'],
    install: false
)
py.extension_module('secondorder',
   [
    fortran_sources_secondorder,
    f2py_secondorder_target, fortranobject_c],
    include_directories: inc_np,
    dependencies: list_dep,
#    link_args: ['-L' + py.get_install_dir() / 'numpy' / 'f2py' / 'src' / 'fortranobject.c', '-lfortranobject'],
    install: true
)

# 'thirdorder' extension

# Compilation of the Fortran module: thirdorder

fortran_sources_thirdorder = [
    'FModules/third_order_ASR.f90',
    'FModules/third_order_centering.f90',
    'FModules/third_order_cond_centering.f90',
    'FModules/third_order_cond.f90',
    'FModules/third_order_dynbubble.f90',
    'FModules/third_order_interpol.f90'
]
## Generate the C wrapper with f2py using a custom target
f2py_thirdorder_target = custom_target('thirdorder-f2py-wrapper',
    input: fortran_sources_thirdorder,
    output: ['thirdordermodule.c','thirdorder-f2pywrappers2.f90'],
#    command: [py.full_path(), '-m', 'numpy.f2py', '--backend', 'meson',
#        '--dep', 'mpi', '--quiet', '-c', '@INPUT0@', '-m', 'thirdorder'],
    command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'thirdorder', '--lower'],
    install: false
)
py.extension_module('thirdorder',
    [
    fortran_sources_thirdorder,
    f2py_thirdorder_target, fortranobject_c],
    include_directories: inc_np,
    dependencies: list_dep,
#    link_args: ['-L' + py.get_install_dir() / 'numpy' / 'f2py' / 'src' / 'fortranobject.c', '-lfortranobject'],
    install: true
)

# 'thermal_conductivity' extension

fortran_sources_thermal_conductivity = [
'FModules/get_scattering_q_grid.f90',
'FModules/third_order_cond.f90',
'FModules/third_order_cond_centering.f90',
'FModules/get_lf.f90'
]
## Generate the C wrapper with f2py using a custom target
f2py_thermal_conductivity_target = custom_target('thermal_conductivity-f2py-wrapper',
    input: fortran_sources_thermal_conductivity,
    output: ['thermal_conductivitymodule.c','thermal_conductivity-f2pywrappers2.f90'],
#    command: [py.full_path(), '-m', 'numpy.f2py', '--backend', 'meson',
#        '--dep', 'mpi', '--quiet', '-c', '@INPUT0@', '-m', 'thermal_conductivity'],
    command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'thermal_conductivity', '--lower'],
    install: false
)
py.extension_module('thermal_conductivity',
   [
    fortran_sources_thermal_conductivity,
    f2py_thermal_conductivity_target, fortranobject_c],
    include_directories: inc_np,
    dependencies: [list_dep, openmp_dep],
#    link_args: ['-L' + py.get_install_dir() / 'numpy' / 'f2py' / 'src' / 'fortranobject.c', '-lfortranobject'],
    install: true
)

# --- Definition of the 'cc_linalg' C extension ---
wrapper_file = ''
if py.version().version_compare('<3.0')
  wrapper_file = 'CModules/wrapper.c'
else
  wrapper_file = 'CModules/wrapper3.c'
endif

# Compilation of the C module: cc_linalg
c_linalg_sources = [
    'CModules/LinAlg.c',
    wrapper_file
]

py.extension_module('cc_linalg',
    c_linalg_sources,
    include_directories: inc_np, # Add the NumPy header for the C module
    dependencies: py_dep,
    install: true
)


# --- Installation of the 'cellconstructor' Python package ---
#install_data(
#  'cellconstructor/__init__.py', 'cellconstructor/AnharmonicForceFields.py', 'cellconstructor/calculators.py',
#  'cellconstructor/Methods.py', 'cellconstructor/Phonons.py', 'cellconstructor/Spectral.py',
#  'cellconstructor/ThermalConductivity.py', 'cellconstructor/Units.py', 'cellconstructor/Bands.py',
#  'cellconstructor/ForceTensor.py', 'cellconstructor/Manipulate.py', 'cellconstructor/Moro_object.py',
#  'cellconstructor/Settings.py', 'cellconstructor/Structure.py', 'cellconstructor/symmetries.py',
#  'cellconstructor/Timer.py',
#  install_dir: py.get_install_dir() / 'cellconstructor',
#)
py.install_sources(
    [
        'cellconstructor/__init__.py',
        'cellconstructor/AnharmonicForceFields.py',
        'cellconstructor/Bands.py',
        'cellconstructor/calculators.py',
        'cellconstructor/ForceTensor.py',
        'cellconstructor/Manipulate.py',
        'cellconstructor/Methods.py',
        'cellconstructor/Moro_object.py',
        'cellconstructor/Phonons.py',
        'cellconstructor/Settings.py',
        'cellconstructor/Spectral.py',
        'cellconstructor/Structure.py',
        'cellconstructor/symmetries.py',
        'cellconstructor/SymData/15.dat',
        'cellconstructor/SymData/36_red.dat',
        'cellconstructor/SymData/36.dat',
        'cellconstructor/SymData/60.dat',
        'cellconstructor/SymData/64.bcs',
        'cellconstructor/SymData/64.dat',
        'cellconstructor/SymData/convert_sym.py',
        'cellconstructor/ThermalConductivity.py',
        'cellconstructor/Timer.py',
        'cellconstructor/Units.py'
    ],
    subdir: 'cellconstructor'
)

install_data(
  'cellconstructor/SymData/15.dat', 'cellconstructor/SymData/36_red.dat', 'cellconstructor/SymData/36.dat',
  'cellconstructor/SymData/60.dat', 'cellconstructor/SymData/64.bcs', 'cellconstructor/SymData/64.dat',
  'cellconstructor/SymData/convert_sym.py',
  install_dir: py.get_install_dir() / 'cellconstructor' / 'SymData'
)

# --- Installation of executable scripts ---
py.install_sources([
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
  message('pytest not found; pytest tests are skipped.')
endif
