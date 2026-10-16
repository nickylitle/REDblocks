# Assembling a sim from blocks

## Contents
1. The three layers
2. Choosing the integrator and dt
3. Energy ledger bookkeeping
4. Coupling bodies without leaking energy
5. Common assemblies (recipes)
6. Performance

## 1. The three layers

| layer | runs | written as | examples |
|---|---|---|---|
| builders | once, before the loop | plain Python, rich objects allowed | `beam_case`, `plate_case`, `mass_properties`, `load_curve`, `polygon_plate`, `laminate_ABD` |
| kernels | every step | `@kernel` (numba njit), floats/ints/arrays only | `tabular_return`, `hertz_sphere_force`, `tire_force_2d`, `motor_current`, `tip_face_penetration` |
| analytic | sizing and checks | plain Python | `hertz_impact`, `spin_up_time`, `euler_load`, `spin_stress` |

A sim file has exactly this shape (see `scripts/templates/sim_template.py`):
`DEFAULTS` dict -> `build(p)` from builders -> `@njit _loop(...)` calling kernels -> `run(params, dt)`
returning a metrics dict with `_ledger_rel` -> `SimSpec` -> `verify_all`.

Put every number in `DEFAULTS` with units in the name or comment. Nothing physical is hard-coded in the loop.

## 2. Integrator and dt

- **Contact, impact, plasticity:** semi-implicit (symplectic) Euler: update velocity from forces, then
  position from the new velocity. Bounded energy error, stable up to dt = 2/w_max.
- **Smooth drive dynamics, thermal networks:** RK4 is fine and more accurate per step. Do not use RK4
  across contact on/off switches; the substeps sample inconsistent contact states.
- **dt:** compute every critical step in the model and take ~0.1x the smallest:
  - spring k on mass m: `critical_dt_spring(k, m_eff)`; use the EFFECTIVE mass at the contact
    (`effective_mass_at_point`), not the whole machine mass.
  - rotational spring on inertia: `critical_dt_rotational`.
  - Hertz contact: `hertz_impact(...)["t_contact"]` then `dt_for_contact(t_c, 200)`.
  - discretized bar/plate: `courant_dt`.
  The harness re-checks this (`stability_margin`) and proves accuracy with dt halving.

## 3. Energy ledger

Track, in the loop, every way energy enters or leaves:
- sources: motor terminal power, initial KE, external work
- stored: KE (translational + rotational, every body, including wheel/rotor spin), elastic energy in
  every spring at the end (use the force: F^2/(2k)), gravity PE
- sinks: plastic work `F * d(set)` per plastic element, dashpot `c v^2 dt`, friction `F . v_slip dt`,
  copper `I^2 R dt`, gearbox losses
Return `_ledger_rel = (sources - dStored - sinks) / scale`. A residual above a few percent means the
model creates or destroys energy somewhere: usually a force applied to one body whose reaction is
missing or applied at a different arm, or contact state switching mid-step.

## 4. Coupling bodies without leaking energy

- One contact force, two bodies: compute the force once, apply +F to one and -F to the other at the
  SAME material point. Moments use the same arm on both sides.
- Hinged faces (ramps, hinged plates): use `tip_face_penetration(tip_x, root_x, root_z, a)`. Its `r_arm` equals
  `-d pen / d a`, which makes the hinge torque exactly the work-conjugate of the penetration rate.
- Use the state-only penetration (geometry from the current positions), not an accumulated overlap.
- Route every reaction explicitly. A sim that passes only part of a reaction to the supporting body
  can still close its ledger through inertia, which hides the missing force path.

## 5. Recipes

**Spinner strike on a ramp or plate** - `tip_on_circle` + `tip_face_penetration` for geometry;
dent = `plate_case` (point load, clamped or ss) stiffness with yield from
`plate_collapse` (point_interior / point_free_edge) (or `tabular_return` on a curve-derived table);
root fold = `moment_curvature_rect` -> `hinge_table` -> `hinge_to_spring_table` -> `tabular_return`;
rotor = rotational DOF with `mass_properties`/`polygon_plate`; first-contact angle from `sample_first_contact`
for Monte Carlo inputs.

**Drive maneuver** - `three_point_normal_loads` (lagged accelerations), `tire_force_2d` per wheel,
`skid_force_2d` for skids, `motor_current` + `gearbox_output_torque` per side, `bus_voltage` for the pack,
`planar_accel` for the body, RK4.

**Flywheel or rotor spin-up / spin-down** - `spin_up_time` (analytic check), `motor_current` loop, `disk_windage_torque`
or `bar_windage_torque`, `bearing_drag_torque`, `copper_resistance` + `node_dTdt` for winding heating.

**Bolted mount under impact** - impulse from `spinner_strike`, load split by `spring_combination` (parallel),
capacities from `bolt_shear_capacity`, `tear_out_capacity`, `bearing_capacity`, `thread_strip_capacity`.

**3-D bodies carrying rotors (hand-written loop until the 3d compiler tier exists)** - `quat_step` +
`quat_to_matrix` for attitude, `gyrostat_accel` / `gyrostat_energy` for a body with a spinning rotor
(gyroscopic coupling exact), `I_box_tensor` + `parallel_axis_tensor` for mass properties. Contact between
bodies: sample edge points of one body (spacing <= the thinnest box dimension / 4), test them against the
other body's boxes with `point_box_penetration`, apply the SAME force at the same world point to both bodies,
`coulomb_regularized_3d` for friction. Worked example: `examples/case_studies/combat_robot/spinner_vs_spinner.py`.

**Flexible blade or arm in a 3-D sim** - rigid hub + one lumped tooth mass per arm (33/140 of the arm mass,
cantilever first mode) tied to its anchor by axial EA/L plus elastic-plastic flap and lag springs from
`beam_plastic_spring(..., support="cantilever_end")` and `tabular_return`. Points on the arm move by xi times the
tooth deflection (xi linear in radius); split every arm force as +f at the undeformed point, -xi f at the anchor,
+xi f on the tooth. Contact INTO a deforming arm must invert that map to material coordinates and take the
normal from the gradient: a rotated rigid box gives a 190 J ledger error mid-hit in the spinner study, the exact
map gives 2 J.

**Composite panel hit** - `parse_layup` -> `laminate_ABD` -> `laminate_as_isotropic(use="flexural")` for
the Roark plate stiffness; ply stresses from `laminate_response` + `ply_stresses`; `first_ply_failure`
for the onset load.

## 6. Performance

numba compiles the loop on first call (cached afterwards). 20,000 steps of a contact sim take
milliseconds, so a 200-sample Monte Carlo with dt/4 convergence runs in seconds. Keep Python objects out
of the loop; pass tables as numpy arrays.
