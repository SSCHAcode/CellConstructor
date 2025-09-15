project('CellConstructor',
  ['c','fortran'],
  version: '1.5.0',
  license: 'GPL',
  default_options : [
        'warning_level=1',
        'buildtype=release',
        'fortran_args=-O2',
        'fortran_args=-cpp'
    ]
    )

# --- System and Python Dependencies ---
# Encuentra las instalaciones necesarias
py = import('python').find_installation(pure: false)
py_dep = py.dependency()
fc = meson.get_compiler('fortran')

# Dependencias adicionales
mpi_dep = dependency('mpi')
# Busca la librería quadmath, si está disponible
quadmath_dep = fc.find_library('quadmath', required: false)

# --- CONFIGURACIÓN DE NUMPY ---
# Obtiene la ruta de los directorios de cabecera de NumPy y f2py usando el comando de Python
incdir_numpy = run_command(py,
    ['-c', 'import numpy; print(numpy.get_include())'],
    check : true
).stdout().strip()

# f2py también requiere la cabecera fortranobject.h
incdir_f2py = run_command(py,
    ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

inc_np = include_directories(incdir_numpy, incdir_f2py)
# --- FIN DE LA CONFIGURACIÓN DE NUMPY ---

# --- Definición de cada extensión de Python (Fortran) ---

# Extensión 'symph'

# Compilación del módulo Fortran: symph
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

py.extension_module('symph',
    fortran_sources_symph,
    include_directories: inc_np,
    dependencies: [py_dep, mpi_dep, quadmath_dep],
    install: true
)

# Extensión 'secondorder'

# Compilación del módulo Fortran: secondorder
fortran_sources_secondorder = [
    'FModules/second_order_centering.f90',
    'FModules/second_order_ASR.f90'
]

py.extension_module('secondorder',
    fortran_sources_secondorder,
    include_directories: inc_np,
    dependencies: [py_dep, mpi_dep, quadmath_dep],
    install: true
)

# Extensión 'thirdorder'

# Compilación del módulo Fortran: thirdorder

fortran_sources_thirdorder = [
    'FModules/third_order_ASR.f90',
    'FModules/third_order_centering.f90',
    'FModules/third_order_cond_centering.f90',
    'FModules/third_order_cond.f90',
    'FModules/third_order_dynbubble.f90',
    'FModules/third_order_interpol.f90'
]

py.extension_module('thirdorder',
    fortran_sources_thirdorder,
    include_directories: inc_np,
    dependencies: [py_dep, mpi_dep, quadmath_dep],
    install: true
)

# Extensión 'thermal_conductivity'

thermal_conductivity_fortran_src = files(
'FModules/get_scattering_q_grid.f90',
'FModules/third_order_cond.f90',
'FModules/third_order_cond_centering.f90',
'FModules/get_lf.f90'
)

py.extension_module('thermal_conductivity',
    thermal_conductivity_fortran_src,
    include_directories: inc_np,
    dependencies: [py_dep, mpi_dep, quadmath_dep],
    install: true
)

# --- Definición de la extensión C 'cc_linalg' ---
wrapper_file = ''
if py.version().version_compare('<3.0')
  wrapper_file = 'CModules/wrapper.c'
else
  wrapper_file = 'CModules/wrapper3.c'
endif

# Compilación del módulo C: cc_linalg
c_linalg_sources = [
    'CModules/LinAlg.c',
    wrapper_file
]

py.extension_module('cc_linalg',
    c_linalg_sources,
    include_directories: inc_np, # Se añade la cabecera de NumPy para el módulo de C
    dependencies: py_dep,
    install: true
)


# --- Instalación del paquete de Python 'cellconstructor' ---
install_data(
  'cellconstructor/__init__.py', 'cellconstructor/AnharmonicForceFields.py', 'cellconstructor/calculators.py',
  'cellconstructor/Methods.py', 'cellconstructor/Phonons.py', 'cellconstructor/Spectral.py',
  'cellconstructor/ThermalConductivity.py', 'cellconstructor/Units.py', 'cellconstructor/Bands.py',
  'cellconstructor/ForceTensor.py', 'cellconstructor/Manipulate.py', 'cellconstructor/Moro_object.py',
  'cellconstructor/Settings.py', 'cellconstructor/Structure.py', 'cellconstructor/symmetries.py',
  'cellconstructor/Timer.py',
  install_dir: py.get_install_dir() / 'cellconstructor',
)

install_data(
  'cellconstructor/SymData/15.dat', 'cellconstructor/SymData/36_red.dat', 'cellconstructor/SymData/36.dat',
  'cellconstructor/SymData/60.dat', 'cellconstructor/SymData/64.bcs', 'cellconstructor/SymData/64.dat',
  'cellconstructor/SymData/convert_sym.py',
  install_dir: py.get_install_dir() / 'cellconstructor' / 'SymData'
)

# --- Instalación de los scripts ejecutables ---
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
  message('pytest no encontrado; se omiten tests con pytest.')
endif
