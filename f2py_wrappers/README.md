# This directory contains the f2py constructed Wrappers for meson
If you change the contsnts of the FORTRAN sources take the corresponden files from the f2py runs:

```Bash
python -m numpy.f2py --backend meson --build-dir meson_builddir --dep mpi -c FModules/third_order_ASR.f90  FModules/third_order_centering.f90  FModules/third_order_cond_centering.f90  FModules/third_order_cond.f90  FModules/third_order_dynbubble.f90  FModules/third_order_interpol.f90 -m thirdorder
```

```Bash
python -m numpy.f2py --backend meson --build-dir meson_builddir --dep mpi -c FModules/second_order_centering.f90 FModules/second_order_ASR.f90 -m secondorder
```

```Bash
python -m numpy.f2py --backend meson --build-dir meson_builddir --dep mpi -c constants.f90 error_handler.f90 get_latvec.f90 io_global.f90 rotate_and_add_dyn.f90 smallgq.f90 symm_matrix.f90 contract_two_phonon_propagator.f90 fc_supercell_from_dyn.f90 get_q_grid_fast.f90 kind.f90 star_q.f90 symvector.f90 cryst_to_car.f90 flush_unit.f90 get_translations.f90 q2qstar_out.f90 set_asr.f90 symdynph_gq_new.f90 trntnsc.f90 eff_charge_interp.f90 from_matdyn.f90 interp.f90 q_gen.f90 set_tau.f90 symm_base.f90 unwrap_tensors.f90 eqvect.f90 get_equivalent_atoms.f90 invmat.f90 recips.f90 sgam_ph.f90 symmetry_high_rank.f90 -m symph
```

```Bash
python -m numpy.f2py --backend meson --build-dir meson_builddir --dep mpi -c get_lf.f90 get_scattering_q_grid.f90 third_order_centering.f90 third_order_cond.f90 -m thermal_conductivity
```
