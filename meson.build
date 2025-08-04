project('CellConstructor', 'fortran', 'c',
  version: '1.5.0',
  license: 'GPL',
  # Common compilation options for Fortran.
  # -cpp is essential for the Fortran preprocessor.
  # -O2 is a general optimization you had in setup.py.
  default_options: ['fortran_args=-cpp', 'fortran_args=-O2']
)

# --- System and Python Dependencies ---
# Imports the Meson Python module to handle extensions and Python installation.
python = import('python')
python_installation = python.find_installation()

# Defines the Python dependencies that numpy.distutils used to handle.
# Meson will look for them in the current Python environment.
#numpy_dep = python_installation.dependency(module: 'numpy', required: true)
#ase_dep = python_installation.dependency(module: 'ase', required: true)
#scipy_dep = python_installation.dependency(module: 'scipy', required: true)

# Searches the system for the BLAS and LAPACK libraries.
# Meson can find implementations such as OpenBLAS, BLIS, Intel MKL, or generic ones.
lapack_dep = dependency('lapack', required: true)
blas_dep = dependency('blas', required: true)

# Look for the OpenMP library if it is needed for parallelization.
openmp_dep = dependency('openmp', required: true)


# --- Definition of Fortran Extensions ---

# Extensión 'symph'
symph_sources = files(
  'FModules/symdynph_gq_new.f90', 'FModules/symm_base.f90',
  'FModules/sgam_ph.f90', 'FModules/invmat.f90', 'FModules/set_asr.f90',
  'FModules/error_handler.f90', 'FModules/io_global.f90',
  'FModules/flush_unit.f90', 'FModules/symvector.f90',
  'FModules/fc_supercell_from_dyn.f90', 'FModules/set_tau.f90',
  'FModules/cryst_to_car.f90', 'FModules/recips.f90',
  'FModules/q2qstar_out.f90', 'FModules/rotate_and_add_dyn.f90',
  'FModules/trntnsc.f90', 'FModules/star_q.f90', 'FModules/eqvect.f90',
  'FModules/symm_matrix.f90', 'FModules/from_matdyn.f90',
  'FModules/interp.f90', 'FModules/q_gen.f90', 'FModules/smallgq.f90',
  'FModules/symmetry_high_rank.f90', 'FModules/unwrap_tensors.f90',
  'FModules/get_latvec.f90', 'FModules/contract_two_phonon_propagator.f90',
  'FModules/get_q_grid_fast.f90', 'FModules/kind.f90',
  'FModules/constants.f90', 'FModules/eff_charge_interp.f90',
  'FModules/get_translations.f90', 'FModules/get_equivalent_atoms.f90'
)
# Create a Python extension module using the Fortran sources and dependencies.
py_symph_ext = python_installation.extension_module(
  'symph', # Name of the resulting module (e.g. symph.cpython-310-x86_64-linux-gnu.so)
  symph_sources,
  install: true,
  # Compilation arguments specific to these Fortran sources.
  # -cpp is already defined in default_options, but can be added here if needed.
  # fortran_args: ['-cpp'],
  dependencies: [lapack_dep, blas_dep]
)

# Extensión 'secondorder'
secondorder_sources = files(
  'FModules/second_order_centering.f90',
  'FModules/second_order_ASR.f90'
)
py_secondorder_ext = python_installation.extension_module(
  'secondorder',
  secondorder_sources,
  # fortran_args: ['-cpp'],
  install: true,
  dependencies: [lapack_dep, blas_dep]
)

# Extensión 'thirdorder'
thirdorder_sources = files(
  'FModules/third_order_centering.f90',
  'FModules/third_order_ASR.f90',
  'FModules/third_order_interpol.f90',
  'FModules/third_order_dynbubble.f90'
)
py_thirdorder_ext = python_installation.extension_module(
  'thirdorder',
  thirdorder_sources,
  # fortran_args: ['-cpp'],
  install: true,
  dependencies: [lapack_dep, blas_dep]
)

# Extensión 'thermal_conductivity'
cond_sources = files(
  'FModules/get_scattering_q_grid.f90',
  'FModules/third_order_cond.f90',
  'FModules/third_order_cond_centering.f90',
  'FModules/get_lf.f90'
)
py_cond_ext = python_installation.extension_module(
  'thermal_conductivity',
  cond_sources,
  # '-fopenmp' is explicitly added here for this specific extension.
  # "-O2" is already in default_options.
  fortran_args: ['-fopenmp'],
  install: true,
  dependencies: [lapack_dep, blas_dep, openmp_dep]
)

# --- Definition of the C Extension 'cc_linalg' ---
# Conditional logic for the wrapper file based on the Python version.
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
  install: true
# I assume cc_linalg also needs numpy, if not it can be removed.
  #dependencies: [numpy_dep]
)

# --- Installing Extension Modules ---
# Register each extension module so Meson knows to install them
# in the Python site-packages directory.
#python_installation.install_extmod(py_symph_ext)
#python_installation.install_extmod(py_secondorder_ext)
#python_installation.install_extmod(py_thirdorder_ext)
#python_installation.install_extmod(py_cond_ext)
#python_installation.install_extmod(py_cc_linalg_ext)


# --- Installing the Python package 'cellconstructor' ---
# This installs the .py files from the 'cellconstructor' directory
# into the 'cellconstructor' subdirectory within the Python site-packages.
python_installation.install_sources(
  python_installation.find_sources('cellconstructor/*.py'),
  subdir: 'cellconstructor'
)

# Installing package data (.dat files)
# They are installed in the 'cellconstructor/SymData' directory in the site-packages.
install_data('cellconstructor/SymData/gamma_symm_op.dat',
             'cellconstructor/SymData/space_groups_crystal.dat',
             install_dir: python_installation.get_install_dir() / 'cellconstructor' / 'SymData')

# --- Installing the executable scripts ---
# Installs the scripts directly into the Python environment's scripts directory.
install_scripts(
  'scripts/symmetrize_dynmat.py',
  'scripts/cellconstructor_test.py',
  'scripts/view_scf_atoms.py'
)
