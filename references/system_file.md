# System files (format version 1)

A system file is a YAML recipe that the compiler turns into a running, verified sim with no hand-written loop.
Run it with the `simblocks` command (`check`, `run`, `set`, `sweep`, `verify`, `report`), open it in the desktop
editor, or from Python: `load_system("model.yaml").run()`. Teaching examples (1D, 2D, 3D, simple to advanced):
`scripts/examples/learn/*.yaml`.

Rules that hold everywhere:
- **Units on every dimensional value**: `2.5 mm`, `12000 rpm`, `7.6e-4 kg m^2`, `200 kN/m`, `1100 rpm/V`, `450 mAh`.
  A bare number is taken as SI. Unknown units are errors, never guesses. Capacitance is `farad` (`F` is ambiguous).
- **Text values** (material names, layups) are written with letters: `"4140 prehard"`, `"7075-T6"`,
  `"CF_3K_twill [0/45]s"`. Text used where a number is needed is an error that names the field.
- **References**: `$name` is a parameter or derived value; `$plate.k` is a field of a block result.
  **Expressions** start with `=`: `"= sqrt(2 * $g * $h)"` (math functions only: sqrt, sin, cos, ..., min, max).
- **Errors** give file, line, field path and the fix; nothing missing is filled in silently.

## Sections

```yaml
simblocks: 1                 # format version (required)
name: ...                    # required
question: ...                # shown on the report
dim: 1d                      # 1d, 2d or 3d
units: SI                    # the editor's default units for this file (SI, mm or US); values still carry their own

params:                      # inputs; anything here can be swept, overridden or made uncertain
  d: 10 mm
  shaft_mat: "4140 prehard"

derived:                     # computed once per run, in order, from blocks or expressions
  shaft: {block: beam_plastic_spring, material: $shaft_mat, section: {circle: $d}, L: $L}
  v_drop: "= sqrt(2 * $g * $h)"
  K_wind: {block: bar_windage_torque, omega: 1.0, R: $R_bar, h: $h_bar, rho: 1.225, Cd: 1.28}
  E_t: {block: material_property, material: "S7", prop: E}
```

`{block: name, arg: value, ..., take: key}` calls any catalog block (`simblocks catalog <text>` lists them).
Arguments are checked against the block's signature. An argument named `mat` given as text becomes the
catalog material. **A material name given to a property argument** (`E`, `E1`, `E_t`, `nu`, `nu2`, `sy`, `Y`, `Y_o`,
`su`, `rho`, `density`, `G`, with any digit or `_suffix`) takes that property of the material:
`{block: effective_modulus, E1: "S7", nu1: "S7", E2: "AR500", nu2: "AR500"}`.
`take:` picks one field of the result; without it the whole result is kept (`$shaft.k`).
Parameters may carry a note: `k: {value: 400 N/m, note: "spring rate from the catalog"}`.

### bodies (1d)

```yaml
bodies:
  hub:    {dof: translation, mass: 0.28 kg}
  rotor:  {dof: rotation, inertia: 7.6e-4 kg m^2, v0: 12000 rpm}
  frame:  {dof: translation, mass: "= $m_machine - $m_hub", x0: 0, v0: 0}
  floor:  {ground: true}
```

Each body has ONE coordinate (metres or radians) along its own axis. Choose the positive direction once per
body and write every measure with that convention (e.g. "coordinates point DOWN" for a drop). `ground: true`
is a fixed body (coordinate 0). Every non-ground body must appear in at least one element.

**Shapes** (every dim): instead of typing mass and inertia, give the body a shape and a material, density or mass:

```yaml
  flywheel: {dof: rotation, shape: {kind: disc, R: 100 mm, t: 20 mm, material: 1018_CD}}
  arm:      {dof: planar, shape: {kind: rod, L: 500 mm, A: 200 mm^2, material: 6061-T6}, at: [0, 0]}
  top:      {dof: free, shape: {kind: disc, R: 40 mm, t: 8 mm, density: 7850 kg/m^3, at: [0, 0, 50 mm], axis: z}}
```

| kind | dimensions (the shape's own axes; z = its axis) | 2D |
|---|---|---|
| `rod` | L, A (section area, for mass from density) | yes (lies along x) |
| `disc` | R, t | yes (adds a circle feature `outline`) |
| `ring` | Ri, Ro, t | yes (adds `outline`) |
| `plate` | a (x), b (y), t (z) | yes |
| `box` | a, b, c | 3D |
| `cylinder` | R, L | 3D |
| `tube` | Ri, Ro, L | 3D |
| `sphere` | R | 3D (adds a sphere feature `outline`) |

Exactly one of `mass`, `density`, `material`. `at` places the shape's centre in the body frame (it becomes the centre
of mass); `axis` (3D) is the world direction of the shape's z axis at the start. A body with a shape gives no
`mass`, `inertia` or `com` of its own. A 1D rotation body takes the inertia about the shape's axis.

### elements

Every element acts on a **measure**: a linear expression of body coordinates, `s = sum c_i q_i + const`.

```yaml
measure: hub + $R_tip * rotor - target     # tip path (translation + R x rotation) minus target
```

The element returns a force F along its measure (F > 0 resists s increasing). The compiler applies
`-c_i F` to every body in the measure. That one rule supplies every reaction and every lever arm, so a
compiled sim cannot forget an equal-and-opposite force. For a rotation body, `c` is the lever arm (m) and the
generalized force is a torque.

| type | fields | law |
|---|---|---|
| `spring` | `k` | F = k s |
| `dashpot` | `c` | F = c ds/dt |
| `kelvin_voigt` | `k`, `c`, `one_sided` | F = k s + c ds/dt; one_sided: only for s > 0, never pulls |
| `elastoplastic` | `table`, `one_sided`, `break_at`, `strain_rate` | `tabular_return` with a (k, acc_tab, F_tab) table from a block; `break_at` = plastic set at which it tears off / punches through and carries nothing after; `strain_rate: {D, q, length}` or `{material, length}` scales the yield each step by Cowper-Symonds 1 + (rate/D)^(1/q), rate = \|ds/dt\| / length (output `rate_factor`) |
| `hertz` | `K`, `n` (1.5), `alpha` (0) | F = K s^n (1 + alpha ds/dt), compression only |
| `friction` | `F_c`, `v_eps`, `F_s`, `v_s` | regularized Coulomb, optional Stribeck peak F_s at speed scale v_s |
| `drag` | `K` | F = K v abs(v) (windage, quadratic air drag) |
| `force` | `P`, or `csv` | constant push P along +measure, or a table against time or the measure (below) |
| `drive` | `kv`, `R`, `I_max`, `gear`, `cells`, `R0`, `R1`, `C1`, `capacity`, `throttle`, `J_rotor` | DC motor + current-limited ESC (duty capped at the throttle) + LiPo 1-RC pack on one rotation body (or a twist measure in 2D/3D); `J_rotor gear^2` is added to that body |
| `dc_motor` | `kv` or `Kt`, `R`, `V`, `I_max`, `gear`, `throttle`, `J_rotor` | a motor on a fixed supply V (no battery, no circuit): just the motor values; `current` has a time history |
| `impact` | `e`, `mu` (0), `v_rest` (0) | rigid contact: no overlap; an impulse sets the separating speed to e x the approach speed, Coulomb friction up to mu x that impulse; every joule removed is booked as dissipated (history = running total). s > 0 means overlap |

**Element mass (1D).** Any 1D element can carry its own `mass`: it is shared onto the bodies in its measure
(m/2 each between two moving bodies, m/3 when the other end is ground, i.e. Rayleigh). A spring, dashpot or
kelvin_voigt with `mass` and `segments: n` becomes a chain of n pieces with n - 1 hidden node bodies, so it
surges and carries waves (outputs still use the element's name).

**CSV loads.** A force can follow a table:

```yaml
shock:
  type: force
  measure: instrument
  csv: {file: data/shock.csv, against: time, x_unit: ms, y_unit: kN, interpolation: linear, x_col: 1, y_col: 2}
```

- `file`: path relative to the model file (the editor uses the saved model's folder). Or inline `x: [..]`, `y: [..]`.
- `against`: `time` (default) or `measure` (a load-deflection curve; x_unit is a length or an angle).
- `x_unit` (default s or m), `y_unit` (a force, or a torque like `N m` for a rotation; default N).
- `interpolation`: `linear` (default), `step` (hold each row) or `smooth` (monotone cubic, PCHIP, no overshoot).
- `x_col`, `y_col`: column number (1 = first) or header text; default the first two. A header row and `#`
  comment lines are skipped; comma, semicolon, tab or space delimited. x must increase. End values hold outside.

**Standard load shapes.** Instead of `P` or `csv`, a force can take `pulse:` (against time; numbers with units):

```yaml
punch: {type: force, measure: plate, pulse: {kind: half_sine, peak: 2 kN, duration: 5 ms, start: 1 ms}}
shake: {type: force, measure: table, pulse: {kind: sine, amplitude: 50 N, frequency: 30 Hz, offset: 0 N}}
```

| kind | numbers |
|---|---|
| `half_sine`, `haversine` | `peak`, `duration`, `start` |
| `triangle` | `peak`, `duration`, `peak_at` (0-1, default 0.5), `start` |
| `rectangle` | `peak`, `duration`, `rise` (softens both edges), `start` |
| `trapezoid` | `peak`, `duration`, `rise`, `fall` (default = rise), `start` |
| `ramp` | `to`, `duration`, `from`, `start` (holds `to` after) |
| `step` | `to`, `from`, `start` |
| `sine` | `amplitude`, `frequency`, `offset`, `phase`, `start`, `stop` (default: the end of the run) |
| `square_wave` | `amplitude`, `frequency`, `duty`, `offset`, `start`, `stop` |
| `sweep` | `amplitude`, `f_start`, `f_end`, `duration`, `offset`, `start` (frequency rises linearly) |
| `decay` | `peak`, `tau`, `start` |

Before `start` the force is 0 (a ramp or step: `from`; a wave: its `offset`). Pulses end at 0. The automatic time
step resolves the shape (a pulse gets at least 50 steps, a wave 40 per cycle). Torque for a rotation: give N m.

Common element fields: `label`, `source` (shown in the schematic table), `note`, and
`knockout: {expect: {metric: "+"|"-"}, expect_elastic: {...}}` to give the expected direction of change
when the element is removed (or made elastic) in the knockout check. `knockout: {skip: "why"}` leaves an element out of the check
(one that only acts in another setting of the model) and the report shows it as skipped with that reason.

Tables for `elastoplastic` come from builder blocks:
`beam_plastic_spring` (whole beam from its section and the material curve), `plastic_indentation_table`
(rounded tooth denting a surface), `curve_to_spring_table` (axial bar), or any block returning k/acc_tab/F_tab.

### outputs

```yaml
outputs:
  shaft_set_mm: {element: shaft, quantity: plastic_set, reduce: final, abs: true, unit: mm}
  F_peak_kN:    {element: tooth_contact, quantity: force, reduce: max, unit: kN}
  t95_s:        {body: rotor, quantity: velocity, reduce: first_time_above, value: "= 0.95 * $target", unit: s}
  E_target_J:   {body: target, quantity: kinetic_energy}
  fpf_SF:       {expr: "= $plate.F_fpf / $F_plate_peak_N"}
```

- element quantities: `force`, `measure`, `plastic_set`, `broken`, `absorbed_work`, `dissipated`; a drive adds
  `current`, `battery_current`, `bus_voltage`, `soc`; a beam `moment`, `torque`, `shear`, `hinge_set`; an impact
  `impulse` (per step), `friction_impulse`; a rigid joint `drift` (largest gap it closed). A weld, hinge or slider:
  `force` is the push / pull reaction (pin force, or the slider's sideways reaction as one magnitude) and `moment`
  the turn lock's reaction (N m); they are never mixed.
- body quantities: `position`, `velocity`, `kinetic_energy` (final only).
- reduce: `final`, `max`, `min`, `max_abs`, `first_time_above`, `first_time_below` (needs `value`; if never
  crossed, the value is 10 x t_end and the report says "never reached"), `at` (the value at time `value`, read from
  the run's log), `time_of_max`, `time_of_min` (when it peaks).
- `unit` converts the SI result for display; failure limits are written in that same unit.
- `expr` outputs read every other output in SI (rad, rad/s, m, N), whatever its `unit`: a display unit never changes
  an expression. An expression's result is SI too; its own `unit` converts it for display.
- peaks are tracked every step, not from the decimated log.

### failure (user-defined only)

```yaml
failure:
  - {name: shaft bend, metric: shaft_set_mm, limit: 0.1, fails_when: above, origin: "user: ..."}
  - {name: plate cracks, demand: "= $F_plate_peak_N", capacity: "= $plate.F_fpf", origin: "user: ..."}
```

Every mode needs `origin` (who defined failure, in their words). Optional `assumption`, `reach_metric`.

### uncertainty, limits, knockouts

```yaml
uncertain:  {rpm: [10800 rpm, 13200 rpm], R_tooth: [1 mm, 3 mm]}      # Monte Carlo + sensitivity
limits:
  - {name: "no spin -> no damage", set: {rpm: 0.0001 rpm}, expect: {shaft_set_mm: 0}}
knockouts:                                                              # extra, beyond the automatic ones
  rigid_shaft: {set: {d: 50 mm}, expect: {shaft_set_mm: "-"}}
```

Automatic knockouts: every element is switched off once ("no <element>"), and every elastoplastic element
is also made elastic ("<element> elastic only"). A knockout that changes nothing fails (dead wiring).

### levers (sensitivity)

```yaml
study:
  sensitivity: {vary: [k, c, m], of: [peak_force_N], step: 1 %}   # of: blank = every output
```
Study runs the nominal model, then each lever once at +step and once at -step (two runs each). For each output the
levers are ranked by elasticity (% change of the output per % change of the input); the change for +step and the
derivative (output unit per the input's SI unit) are listed too. Up and down changes that differ by more than 20 %
are flagged: a kink or a threshold near the nominal point. A zero-valued lever moves by +-step in its own SI units.

### goal seek

```yaml
study:
  goal_seek: {vary: t_wedge, output: peak_stress_MPa, target: 900, tolerance: 0.1 %, range: [2 mm, 12 mm]}
```
Study runs the model again and again: from the value's own number, a small step, then secant steps from the last
two runs (inside range, or within x10 of the last value), until the output is within tolerance of the target
(the output's own unit). It reports the value found, the runs it took, and why it stopped if it did not converge.
With uncertainty ranges on the levers (uncertain: {p: [lo, hi]}), a Levers study also gives each output's
first-order spread: the derivatives times each input's standard deviation (uniform), added in quadrature.

### run and report

```yaml
run: {t_end: 0.4 ms, dt: 20 ns, mc_n: 150, bounds: {shaft_set_mm: [0, 15]}}   # dt: auto = 0.02 x critical
# accuracy: draft | normal | fine  scales the automatic step x2 | x1 | x0.5 (run: {t_end: 2 s, accuracy: fine})
# first_time_* outputs are interpolated between steps (the crossing of a straight line through the two steps)
# automatic step held inside bounds instead of a fixed one:  run: {t_end: 0.4 ms, dt_min: 5 ns, dt_max: 50 ns}
# motors (drive, circuit motor) do not limit the step for stability: their back-EMF damping on the shaft is stepped
#   exactly (the shaft's speed change over a step is the plain step's x (1 - e^-k)/k, k = c dt / J, with the other
#   loads held over the step). The auto step takes five steps per motor time constant J R / (G Kt)^2.
#   J is the inertia the shaft has with the rigid joints held (worked out each step from the joint rows), so a motor
#   driving a jointed linkage spins up on the whole linkage's inertia.
# adaptive step (changes as it runs):  run: {t_end: 2 s, dt: adaptive}   (dt_min / dt_max still bound it)
#   free motion: the automatic step with the contact elements left out (one-sided, hertz, contact measures),
#     and at most t_end / 20000 when the model has contacts (free flight is first order in the step)
#   contacts: the full automatic step while a contact touches, or would close within 4 free steps at its speed
#   driven fields: a {control: ...} stiffness or damping sets its own step from the signal's value as it runs
#     (fitted from the model compiled at two values), so it needs no max:
#   the step shrinks at once and grows at most x1.25 a step; outputs add _steps, _dt_smallest, _dt_largest,
#   _contact_steps. Verify's dt-convergence halves every step it would take.
# timeout (wall-clock, per run): auto (default) = 5 x estimated run time, at least 10 s | none | a time:  run: {t_end: 3 s, timeout: 5 min}
#   a stopped run returns what it computed so far, with _timed_out = 1 and _t_stopped = the simulated time reached
report:
  plots:
    - {title: Shaft force, unit: kN, series: [{ref: shaft.force, unit: kN}]}
  sweep: {param: d, values: [8 mm, 10 mm, 12 mm], unit: mm, metric: shaft_set_mm,
          variants: {"4140 prehard": {shaft_mat: "4140 prehard"}, "7075-T6": {shaft_mat: "7075-T6"}}}
  input_labels: {rpm: rotor rpm}
assumptions: [...]
sources: [...]
```

Plot refs are `<element>.<quantity>` or `<body>.position|velocity`.

### exports (optional CSV time histories)

Nothing is written unless an export is defined. `simblocks run` and the editor's Run write them; live runs never do.

```yaml
exports:
  - name: hit_force
    columns:                                   # element signals/states, or body position/velocity
      - {element: hit, quantity: force, unit: N, label: F_hit}
      - {body: blade, quantity: velocity, unit: rpm}
      - {part: wpn, quantity: current}        # a circuit part
    folder: out/csv                            # relative to the model file, or absolute
    naming: "{model}_{name}"                   # tokens {model} {name} {date} {time} {run}
    time_unit: ms
    points: 1000                               # rows, or all
    header: units                              # units: Time [ms],F_hit [N] | plain: t_ms,F_hit_N | none
    when: run                                  # run | study (also one numbered file per study run)
```

Element quantities with a time history: force, measure, plus plastic_set and broken (elastoplastic), current,
battery_current, bus_voltage, soc (drive), moment, torque, shear, hinge_set (beam) and impulse, dissipated,
friction_impulse (impact). Otherwise absorbed_work and dissipated are end-of-run numbers.

### Vehicles in flight (2d): rockets, staging, drag, steering, landing guidance, a planet

`gravity: {planet: earth, radius: 6373 km, rotation: $site.omega_plane}` puts the planet's centre at the origin:
inverse-square gravity (mu), the US Standard Atmosphere 1976 above its surface (pressure for engine thrust, density and
temperature for drag), and the air turning with the planet at `rotation` (rad/s about the model's plane; the
`launch_plane(latitude, azimuth)` block gives it, and the pad's speed `v_surface` for the body's v0). A vehicle is one
body with `dof: [x, y]`; its elements take `measure: <body>`. `gravity: {planet: moon, rotation: ...}` is the Moon
(GM 4.9028e12 m^3/s^2, radius 1737.4 km, no air); `atmosphere: us76 | none` overrides either planet's air.

| type | fields | law |
|---|---|---|
| `rocket` | **thrust_vac**, **mass_flow**, exit_area, start, stop, rise, tailoff, cutoff: guidance, throttle: guidance, propellant | F = level x (thrust_vac - p_amb exit_area) along the steering direction; the body loses level x mass_flow; with propellant it stops when that is burnt |
| `jettison` | **at**, **mass** | the body's mass drops by mass at time at (a stage, an interstage, a tower) |
| `aero_drag` | **area**, **Cd** (number or {mach: [...], cd: [...]}) | D = 1/2 rho v^2 Cd area against the air-relative velocity |
| `steering` | pitch_start, pitch_rate, pitch_end, tilt_arrest, guidance_start, target_altitude, target_speed, tgo_freeze | vertical, pitch-over, gravity turn, hold, then explicit guidance; cutoff at target_speed |
| `landing_guidance` | **site_range**, mode (target, rate), start (time or `after`), stop, stop_below, end_before, target, schedule, freeze, tau, contact_altitude, throttle | one phase of Apollo's lunar descent guidance (below); sets the thrust direction and the level of `throttle: guidance` rockets |

`landing_guidance` phases run in the order written: `start: after` begins when the one before it ends (at `stop`, at
`stop_below` altitude, or `end_before` seconds ahead of its target). mode `target` flies Apollo's quadratic law in the
landing-site frame (X up, Z downrange, turning with the planet): a = a_T + 6 (v_T + v) / T + 12 (r_T - r) / T^2 per
axis, with `target: {time, altitude, range, speed_forward, speed_up, accel_forward, accel_up, jerk_forward}` (range
negative short of the site; time after the phase starts). With `jerk_forward` the time to go T is solved every step
from the downrange jerk condition (Draper R-695 eq. 12), otherwise it counts down to `time`; `freeze` holds the last
command over the final seconds. mode `rate` follows `schedule: {time, speed_up, speed_forward}` (times from the phase
start) with a lag `tau` (Apollo P66: 1.5 s); `contact_altitude` stops the engine at that altitude and the run ends at touchdown.
`throttle: {start_level, start_hold, full, recovery, min, max}`: fractions of the rocket's thrust_vac; the engine sits
at start_level until start_hold, then at full until the thrust asked for drops below recovery, then follows the ask
between min and max (above max it goes to full). Outputs: `altitude`, `speed_forward` (over the ground), `speed_up`,
`throttle`, `thrust_command`, `range_to_site`; rocket `propellant` (left).

Engines come from `rocket_engine(fuel, OF, Pc, eps, mdot | F_vac, Isp_sl | Isp_vac | efficiency)`: equilibrium
combustion and shifting-equilibrium nozzle expansion (NASA RP-1311 method) for RP1, LH2, CH4, A50 (Aerozine 50),
N2H4, UDMH or MMH with `oxidizer: LOX` (default) or `N2O4`,
returning F_vac, F_sl, A_e, mdot, delivered and ideal Isp, Tc, c*. Outputs: steering `altitude`, `speed_inertial`,
`airspeed` (earth-fixed), `flight_path_angle`, `pitch` (deg), `downrange` (over the turning ground); aero_drag `drag`,
`dynamic_pressure`, `mach`, `air_density`, `ambient_pressure`; rocket `thrust`, `mass_flow`, `vehicle_mass`, `throttle`.
Examples: examples/showcase/apollo11/ (the Saturn V ascent against NASA's postflight trajectory, and the lunar
module's powered descent to touchdown).

### control (optional): live signals, logic and feedback

Named signals worked out every time step from the running model. An element field, or a circuit part's field
(including its `esc` and `thermal` settings), written `{control: <signal>, init: <first-step value>, max: <the most it
reaches>}` follows that signal as the model runs (a load's `P`, a drive's, rocket's or ESC's `throttle`, a spring's
`k`, a damper's `c`, a drive's `kv`, which the model stores inverted). Signals are SI. Outputs and CSV columns read a
signal with `control: <name>` (any reduce, including at, time_of_max and first_time_*).

The time step: a driven stiffness or damping moves the explicit stability limit. With `max:` the model is also
compiled at max and an automatic time step is sized for it (and Verify's stability margin uses it). Without `max:` on a
field that moves the limit, the step is sized for `init`, and after the run a note gives the most the field reached
and the max to set. A field driven past its `max` is flagged too.

```yaml
control:
  err:     "= $v_set - car.velocity"                                   # an expression
  push:    {pid: {measure: car.velocity, setpoint: $v_set, kp: 800, ki: 120, kd: 0, min: 0, max: 4000, d_filter: 0.02}}
  staged:  {if: "air.ambient_pressure < 2000", then: 1, else: 0, latch: true, for: 0.5 s}
  smooth:  {lag: push, tau: 0.3 s}
  ramp:    {rate_limit: smooth, rate: 500}
  dist:    {integrate: car.velocity, init: 0}
elements:
  engine: {type: force, measure: car - road, P: {control: ramp}}
outputs:
  push_max: {control: push, reduce: max}
```

| kind | fields | law |
|---|---|---|
| expression | a string `= ...`, or `{expr: ...}` | names: other signals, `body.quantity` / `element.quantity` (anything an output reads), `t`, `$params`; min, max, abs, sqrt, exp, log, sin, cos, tan, atan2, clip(x, lo, hi), sign, floor; comparisons give 1 / 0; and, or, not; `a if c else b` |
| `if` | **if** (condition), then, else, latch, for | then while the condition holds (after holding `for` seconds), else otherwise; `latch` keeps it on once true |
| `pid` | **measure**, **setpoint**, kp, ki, kd, min, max, d_filter, bias | u = bias + kp e + ki ∫e dt - kd d(measure)/dt, e = setpoint - measure; clipped to [min, max]; the integral holds while the output is pinned (anti-windup) |
| `lag` | **lag** (input), tau, init | first-order: y += (u - y)(1 - e^(-dt/tau)) |
| `rate_limit` | **rate_limit** (input), **rate**, init | y moves toward u by at most rate x dt per step |
| `integrate` | **integrate** (input), init | y = init + ∫u dt |

Timing: at the end of each step the signals read the state that step ended in (positions, speeds, forces and element
states) and set the fields the next step uses. Signals are worked out in the order written; one that reads a signal
written below it gets that one's value from the step before. A field the compiler turns into several other numbers (a
beam's section, a body's mass, a circuit motor's kv) cannot be driven and is refused. A driven spring-like element (spring, kelvin_voigt, hertz, elastoplastic, pin, beam) does
work that is not its own law: its element check is skipped, the global ledger still closes.

### circuit (optional lumped electrical circuit)

Parts wired between named nets; `ground` is the 0 V net (default `gnd`). Solved every step by nodal analysis
(BDF2 companions for C and L, second order with almost no numerical damping, backward Euler on the first step and
for the battery's RC branch; diodes, ESC current limits and freewheeling settled
by re-solving). Motors couple to the mechanics like a drive: torque `gear Kt i` on their measure, back-EMF
`gear Kt ds/dt`. Sign: a part's current flows from its first net to its second through the part; `voltage` is
v(first) - v(second); `power` is absorbed (negative while delivering). The part powers sum to zero every step
(`_circuit_ledger_rel`). A circuit-only model may leave out `elements` and `bodies`.

```yaml
circuit:
  ground: gnd
  parts:
    pack:  {type: battery, between: [bat, gnd], cells: 6, capacity: 1300 mAh, R0: 20 mohm, R1: 8 mohm, C1: 2000 farad}
    fuse:  {type: fuse, between: [bat, bus], I_rating: 100 A, t_blow: 1 s}
    caps:  {type: capacitor, between: [bus, gnd], C: 2200 uF, ESR: 15 mohm}
    wpn:   {type: motor, between: [bus, gnd], kv: 1100 rpm/V, R: 30 mohm, L: 12 uH, measure: blade,
            esc: {throttle: 1, I_max: 120 A}, thermal: {C_th: 40 J/K, R_th: 4 K/W, T_max_C: 160}}
```

| type | fields (required in bold) | law |
|---|---|---|
| `resistor` | **R** | v = R i |
| `capacitor` | **C**, ESR (0), R_leak (0 = none), v0 | i = C dv/dt behind ESR |
| `inductor` | **L**, R, i0 | v = L di/dt + R i |
| `voltage_source` | **V** (or wave), R (0.1 mohm), I_load (a fixed draw, A, or a wave), wave | v = V - R i, + is the first net |
| `current_source` | **I** (or wave), wave | drives I out of + into the circuit (its current reads negative) |
| `current_load` | **I** (or wave), wave | draws I in at + (supply side), out at − (its current reads positive) |
| `battery` | **cells**, **capacity**, R0, R1, C1, soc0 (1), V_full (4.2), V_empty (3.3), I_load (a fixed draw across it, A, or a wave; its part is `<name>__load`) | LiPo OCV(soc) - R0 i - v1, v1 the R1 \|\| C1 sag |
| `diode` | Vf (0.7 V), R_on (10 mohm), R_off (10 Mohm) | conducts first net to second past Vf |
| `switch` | R_on, R_off, on_at, off_at, pwm_freq, duty, phase | closed from on_at to off_at, chopped at duty with pwm_freq |
| `fuse` | **I_rating**, t_blow (1 s), I_trip, R | opens once integral (i/I_rating)^2 dt > t_blow, or at once above I_trip |
| `motor` | **R**, **measure**, kv or Kt, L, gear, J_rotor, eta, eta_back, esc {throttle, I_max, regen, brake} | v = R i + L di/dt + gear Kt ds/dt |

Function generator: `wave:` on a voltage source, current source or current load (and `I_load:` given as the same
dict on a battery or voltage source) drives it with a waveform. Output = offset + amplitude x shape(t).

```yaml
    sig: {type: current_load, between: [bus, gnd], wave: {shape: square, amplitude: 5 A, offset: 5 A, frequency: 10 Hz, duty: 0.25}}
    ac:  {type: voltage_source, between: [a, gnd], wave: {shape: sine, amplitude: 10 V, frequency: 50 Hz, phase: 90 deg}}
    pack: {type: battery, between: [bat, gnd], cells: 6, capacity: 2200 mAh, I_load: {shape: pulse, amplitude: 20 A, frequency: 5 Hz, duty: 0.2}}
```

Keys: `shape` (sine, square, triangle, sawtooth, pulse = 0 to amplitude, step, ramp), `amplitude`, `offset`
(default: the part's V or I), `frequency` or `period` (periodic shapes), `phase`, `duty` (0 to 1, square and pulse),
`start` (offset before it), `stop` (offset after it), `rise` (ramp: time to climb by the amplitude). The auto step
takes about 50 steps per period.

Motor details:
- **ESC** `throttle` runs -1 (full reverse) to 1: a reversing drive ESC. `regen: true` lets current flow back to
  the bus; `brake: true` is the drag brake: at zero throttle the bridge shorts the winding (current held to `I_max`),
  otherwise zero throttle coasts.
- **Gearbox efficiency** `eta` (forward, default 1): the output gets eta x gear x Kt x i while the motor drives it;
  when the load drives the motor (braking, back-driving) it feels gear x Kt x i / `eta_back` (default from eta and the
  ratio, the planetary/spur relation; give it if measured). The loss is counted in the motor's `heat`.
- **J_rotor** is reflected through the gear (J_rotor gear^2 on the output). On a motor that rides on a moving body
  (`measure: {twist: [chassis, wheel]}` in 3D) the rotor's reaction stays in the chassis, as the gearbox puts it:
  the chassis feels the torque that reaches the wheel plus the rotor's own spin-up, not the full reflected inertia
  torque (a gearmotor launching a robot does not wheelie from rotor inertia it does not have), and the gyroscopic
  pull when the chassis turns is the real rotor's (G J_r w), not the reflected G^2 J_r w.

`thermal` (optional on every part except sources): `{C_th: J/K, R_th: K/W, T_amb_C: 25, T0_C, alpha: 1/K,
T_max_C, kt_tempco, vf_tempco}`. The part heats with its losses and cools to ambient through R_th. Its resistance
scales by 1 + alpha (T - 25 C) (motor and inductor default alpha 0.00393, copper); a motor's Kt by
1 + kt_tempco (T - 25 C) (default -0.0011, NdFeB); a diode's Vf by vf_tempco (default -0.002 V/K). Above T_max_C the
part fails open.

Part outputs and export columns: `{part: wpn, quantity: current, reduce: max}`. Quantities: `current`, `voltage`,
`power`, `energy` (absorbed so far), `heat` (J turned to heat by its resistance so far), `soc` (battery), `torque`
(motor shaft, before the gear), `temperature` (C, needs `thermal`) and `state`: a fuse 1 once blown; a switch or
diode 1 while conducting; a motor its ESC duty (1 without an ESC); any other part 1 while intact, 0 once it has
failed open from overheating.
Reduce: `final`, `max`, `min`, `max_abs`.

### editor (written by the desktop editor, ignored by the solver)

`editor: {layout: {"b:tip": [x, y], ...}, groups: [...], notes: [{text, x, y, width}], images: [{path, x, y, width,
caption}]}` keeps node positions, groups, canvas notes (Markdown and LaTeX) and image blocks (a picture linked from
the model's folder). Hand-written files can leave it out.

## Gears and bolts (type: gear, type: bolt)

Both are measure elements, so they work in 1d, 2d and 3d.

```yaml
mesh: {type: gear, measure: pinion + wheel, module: 1 mm, teeth: [12, 36], face_width: 8 mm,
       backlash: 0.05 mm, material: 4140_normalized}         # 2d: pinion.rz + wheel.rz, 3d: .spin
hold: {type: bolt, measure: lid - frame, size: M5, grip: 6 mm, grade: "12.9", torque: 6 N m}
```

- **gear.** The measure lists the two shafts, gear 1 first (the signs written do not matter; `internal: true` is a ring gear; `teeth: [z, rack]` meshes with a translation body). It becomes pitch-line travel `r1 a1 + r2 a2`. The force is the tangential tooth load Ft. The mesh stiffness defaults to ISO 6336-1 (`c_gamma` 20 N/(mm um) per mm of face, scaled by E), with no force inside the backlash, and teeth that only push. Damping is `zeta` 0.05 of critical by default, or `c`. Outputs: `bending_stress` (Lewis, Ft / (b m Y) for the weaker gear) and `contact_stress` (Hertz at the pitch point, AGMA form with no extra factors).
- **bolt.** The measure is the joint opening from the clamped state. Size is `M5`, `M6x0.75`, `#10-32` or `1/4-20`, or `d` and `pitch`. The preload is `preload` or `torque` (Fi = T / (K d), with `nut_factor` K = 0.2). The bolt stiffness is `Ad At E / (Ad lt + At ld)`; the member stiffness is the 30 deg frustum (Shigley 8-23). `grade` sets Sy, Su and the elongation at break (ISO 898-1, ISO 3506-1, SAE J429). Past yield the bolt hardens linearly to Su over elongation x the thread in the grip, then breaks. Outputs: `bolt_force`, `clamp_force`, `bolt_stress` (Fb / At), `plastic_set` and `broken`. The joint separates at Fi (1 + k_b / k_m).

## Model blocks, ports and sequences (ports:, models:)

A model with **ports** can be used as one block inside another model. Ports carry no value of their own:

```yaml
ports:
  inputs:
    v_in:  {unit: m/s, default: 2 m/s}     # a value: $v_in anywhere a number goes
    F_hit: {kind: series, unit: N}         # a time series: csv: {series: $F_hit} on a force
  outputs:
    v_out: {output: v}                     # a value: one of this model's outputs
    F_out: {element: s, quantity: force}   # a time series: anything an export column can name
```

An input with no default makes the model run only as a block. The using model lists its blocks:

```yaml
models:
  hit:   {file: hit.yaml, inputs: {v_in: $v}}
  frame: {file: frame.yaml, inputs: {F_hit: $hit.F_out, v0: $hit.v_out}}
```

- Blocks run before this model compiles, in wire order: a block that reads `$hit.x` runs after `hit`. A loop is an error.
- Each block's value ports and outputs are numbers here (`$hit.v_out`, also inside `=` expressions). Series ports feed force tables: `csv: {series: $hit.F_out}`.
- Every block's outputs are added to the results as `hit.v_out`, `hit.F`.
- A file with `models:` and no bodies is a sequence (`bodies`, `elements`, `outputs`, `run` and `dim` may be left out).
- Runs are cached by file time and inputs. A study on this model reruns only the blocks whose inputs change.
- Blocks are one-way (no feedback from downstream). Each keeps its own `dim` and time step.

**Values from a CSV** anywhere a number goes: `{from_csv: out/csv/hit_loads.csv, column: va, pick: final}`. `pick` is final, first, max, min, max_abs, mean, or at (with `at: 5 ms`); `unit` overrides the header. A header cell `name [unit]` (every export writes these) gives the column's unit. A force table picks columns by header text, with or without the unit, and takes its `x_unit` and `y_unit` from the header when they are left out.

## Analysis, design studies and extras

- **Modes and static** (editor buttons; `compiler/linear.py`): linearised through the solver itself, about the settled position. Natural frequencies in Hz and rpm, mode shapes, rigid-body modes at 0 Hz. Static: Newton's method on the solver's forces, loads as at t = 0, every output read at rest.
- **Spectra** (`compiler/spectra.py`): FFT amplitude spectrum, and the maximax shock response spectrum (Smallwood, 5 % damping) of any history.
- **Optimizer** `study.optimize: {vary: {p: [lo, hi]}, minimize | maximize: output, subject_to: [{output, min | max}], failure_modes: true, max_runs: 150}`. COBYLA; the best feasible run is kept.
- **Scenarios** `scenarios: {name: {param: value}}`, run side by side.
- **Weight budget** `budget: {limit: 3 lb, bodies: true, items: {name: mass}}` adds outputs `mass_total` and `mass_margin` in the limit's unit.
- **Model blocks, repeated**: `models: {hits: {file: dent.yaml, repeat: 5, carry: {set_in: set_out}}}`; results `hits.out@1` and onwards.
- **Elastoplastic extras**: `set0` / `acc0` (permanent set and plastic travel carried in from an earlier hit), and `softening: {T: 150 C, material: 4140_normalized}` (Johnson-Cook 1 - T*^m on the yield table; or give T_melt and m). Temperatures read 150 C, 300 F or 420 K; a plain number is Celsius.
- **Circuit extras**: `wire` part (awg or area, length, material, `connectors: [XT60, XT60]` with makers' contact resistance, R_contact), with `thermal: {}` filling in the copper's heat capacity and convection; motor `I0` (no-load current: a drag torque Kt I0 on the shaft); ESC losses `esc: {R_on, t_sw, f_sw, C_th, R_th, T_amb_C}` with part outputs `esc_temperature` and `esc_heat`.
- **Machine-element blocks** (`structures/machine_elements.py`): shaft critical speed (Rayleigh and Dunkerley), bearing L10 life and static s0, press fit (Lame), key, set screw, clamp collar, helical spring (Wahl, surge, solid length), belt slip (capstan) and chain, crack check (K_IC), notch sensitivity (Kf), Volkersen lap joint, fillet weld.

## What the compiler checks before running

Unknown keys, missing required fields, wrong field for an element type, `mass` on a rotation body, bodies in
no element, non-linear measures, undefined `$names`, unknown units, unknown blocks or block arguments,
outputs naming missing elements, failure metrics that are not outputs, and 2d/3d-only sections (`directions`,
`angles`) in a 1d file.

## Energy ledger

Per element: absorbed work A = sum F ds. Global: change in kinetic energy + sum A = 0. Per element with its
own energy law (springs, elastoplastic, undamped Hertz, the drive): A = stored change + dissipated - input.
Both residuals form `_ledger_rel` (first order in dt). The verification battery vetoes > 5%.

## 2d models (dim: 2d)

A 2d model is a plane (x right, y up, angles counter-clockwise). Every body is rigid and moves in up to three
coordinates: x and y of its tracked point and its spin angle rz. Bodies carry their own geometry, and elements
measure that geometry, so lever arms, contact normals and sliding come from where things are, not from typed
factors. Same laws, same element types, same outputs, verification and reports as 1d.

```yaml
dim: 2d
gravity: 9.81 m/s^2            # optional; along -y (or a vector [gx, gy])
bodies:
  floor: {ground: true, lines: {top: {from: [-1 m, 0], to: [1 m, 0]}}}   # ground geometry is in world coordinates
  weapon:
    dof: [rz]                  # free coordinates, any of x, y, rz; planar (default) = all three
    inertia: 7.56e-4 kg m^2    # about the centre of mass (mass: needed if x or y is free)
    at: [0, 0]                 # where the body's own drawing origin starts (the pivot, for a spin-only body)
    angle0: -60 deg            # starting angle; omega0: starting spin; v0: [vx, vy] starting velocity
    com: [0, 0]                # centre of mass in the body's drawing frame (default: its origin)
    points: {tooth: [90 mm, 0]}
    circles: {hub: {at: [0, 0], r: 20 mm}}
    lines: {face: {from: [0, 0], to: [30 mm, 0], depth: 4 mm}}
```

- A body free in x or y is tracked at its **centre of mass**: `A.x`, `A.y` and the x, y outputs are the centre
  of mass. A body that only spins (`dof: [rz]`) turns about its drawing origin; give `mass` and `com` for an
  off-pivot centre of mass and the compiler adds m d^2 and the gravity torque.
- Geometry is given in the body's own drawing frame (CAD coordinates are fine; `com` locates the centre of mass).
- `angle0` is the body's starting rotation; points, circles and lines rotate with it.

### Measures in 2d

| Measure | What s is | Typical element |
|---|---|---|
| `'A.x - B.x + $r * A.rz'` | linear sum of coordinates (x, y, rz of each body) | springs, hinges on `face.rz`, drives on `rotor.rz` |
| `{distance: [A.p, B.q], length: L}` | distance between two points minus L (default: the starting distance) | rope, link, strut |
| `{along: [A.p, B.q], dir: x}` | gap between two points along a fixed direction (`x`, `y`, `-x`, `[dx, dy]`) | slider, guide rail |
| `{contact: [A.tip, B.face]}` | penetration of a point or circle into a line segment or circle (> 0 in contact) | hertz, kelvin_voigt, elastoplastic |
| `{twist: [A, B]}` | B's turn relative to A (rz_B - rz_A, zero at the start) | a motor between a frame and its rotor, torsion springs, bearing drag |
| `type: pin`, `measure: [A.p, B.q]` | the two points held together (vector spring-damper k, c, or `rigid: true`) | hinges, axles |
| `type: contact_friction`, `contact: <element>` | slip along that contact's surface: mu x contact force (3D: optional tire curve, below) | sliding, rolling, scraping |

`A.p` names a point, circle or line on body A; `A` alone is its drawing origin.

**Contacts.** The outside of a line is the side the point or circle starts on. A contact only acts while the
circle's centre is over the segment and the penetration is under `depth` (default: the line's length): a tooth
that bites deeper than `depth` tears through and loses purchase. A contact can only start by crossing the line
from outside, never by arriving from beyond the segment's end. Elastic energy still stored when a contact drops
(sliding off an end) is booked as dissipated. kelvin_voigt and elastoplastic on a contact are one-sided
automatically.

**Outputs.** Body quantities: `x`, `y`, `angle`, `vx`, `vy`, `omega`, `speed`, `kinetic_energy`, and `position` /
`velocity` for a body with one free coordinate. Element quantities as in 1d. CSV exports take the same body
quantities.

**Time step.** The automatic step uses each element's stiffness against the largest lever arm of its geometry.
Hertz contacts set no step of their own (as in 1d): give `dt` when a stiff Hertz contact governs.

Worked examples: `scripts/examples/learn/2d_*.yaml` (pendulum on a rigid hinge, a ball bouncing down a ramp with
impacts, a motor-driven four-bar linkage).

## 3d models (dim: 3d)

z is up. Bodies are rigid. Each slides in any of x, y, z and either keeps its orientation, **spins** about one
fixed axis (any vector), or **rotates** freely (full 3D rotation with an inertia tensor: precession, nutation and
gyroscopic moments come out of the physics). Same rule as 1d and 2d: elements act along measures of the geometry.

```yaml
dim: 3d
gravity: 9.81 m/s^2              # along -z (or a vector [gx, gy, gz])
bodies:
  floor: {ground: true, planes: {top: {at: [0, 0, 0], normal: [0, 0, 1], size: [1 m, 1 m]}}}
  weapon:
    dof: [spin]                  # any of x, y, z, plus spin or rotate; free = [x, y, z, rotate] (default)
    axis: [0, 0, 1]              # spin axis: any vector (normalised) or x, y, z, -x...
    inertia: 7.56e-4 kg m^2      # spin: about the axis. rotate: a number, [Ixx, Iyy, Izz] or a 3x3 list (centre of mass)
    angle0: -30 deg              # spin bodies; omega0: the spin rate ([wx, wy, wz] world, for rotate)
    at: [0, 0, 0]                # the pivot of a spin-only body; otherwise where the drawing origin starts
    com: [0, 0, 0]
    orient0: {axis: [1, 0, 0], angle: 20 deg}   # starting orientation of the drawing frame
    v0: [0, 0, 0]
    points: {tooth: [90 mm, 0, 12 mm]}
    spheres: {hub: {at: [0, 0, 0], r: 20 mm}}
    planes: {face: {at: [..], normal: [..], u: [..], size: [a, b], depth: d}}   # a finite rectangular patch
```

- `inertia` of a rotating body: a number, `[Ixx, Iyy, Izz]`, a 3x3 list, or the CAD table
  `{Ixx: .., Iyy: .., Izz: .., Pxy: .., Pxz: .., Pyz: ..}` as SolidWorks / Onshape / Fusion print it about the centre
  of mass (products P = +sum(x y dm); the tensor uses -P; tensor entries `Ixy`.. are accepted instead). Non-physical
  tensors are refused with the reason.
- Measures: coordinates `A.x`, `A.y`, `A.z` and `A.spin` (spin bodies), `{distance: [A.p, B.q]}`,
  `{along: [A.p, B.q], dir: z, turns_with: A}`, `{contact: [A.tip, B.face]}` (a point or sphere against a plane patch
  or a sphere), `{twist: [A, B], axis: z}` (B's turn relative to A about an axis fixed in A, given in world
  directions at the start), `type: pin` (a ball joint) and `contact_friction` (slip in the contact plane).
- **Tire curve** on `contact_friction` (3D): `mu` is the sliding grip; add `mu_peak` and `slip_peak` and grip follows
  slip ratio sigma = slip speed / the larger of the hub's ground speed and the tread's rolling speed (at least `v_min`,
  default 0.05 m/s): a brush-like rise, the peak `mu_peak` at `slip_peak`, settling to `mu` at large slip.
  `{type: contact_friction, contact: tire, mu: 1.67, mu_peak: 2.0, slip_peak: 0.12}` for a wheel's sphere on a floor.
- A plane patch's outside is the side the point starts on; it counts while the point is over the patch
  (`size` along `u` and `normal x u`) and the bite is under `depth`. Contacts start only by crossing from outside.
- Body outputs: `x y z vx vy vz speed`, `angle` and `omega` of a spin body, `wx wy wz` (world angular velocity)
  and `omega` (|w|) of a rotating body, `kinetic_energy`; `position` / `velocity` for a body with one coordinate.
- Free rotation is integrated on the angular momentum (exact for a torque-free body) with the orientation turned
  by the exact rotation of each step.

Worked examples: `scripts/examples/learn/3d_*.yaml` (a spinning top on a ball joint, a drone arm beam under a CSV
thrust, a robot arm with rigid hinges, a weld, a motor on a twist and a rubber stop).

## Joints, ropes and beams (2d and 3d)

```yaml
elements:
  pivot:   {type: hinge, rigid: true, axis: y, measure: [base.shoulder, arm.s]}      # 3D needs axis; 2D does not
  elbow:   {type: hinge, rigid: true, axis: y, k_twist: 60 N m/rad, c_twist: 0.4, measure: [arm.e, fore.e]}
  grip:    {type: weld, rigid: true, measure: [fore.w, payload.grip]}
  rail:    {type: slider, rigid: true, axis: x, k_rot: 0, measure: [frame.o, cart.p]}   # k_rot present: no turning
  tether:  {type: rope, k: 50 kN/m, c: 20 N s/m, measure: {distance: [post.top, load.hook], length: 1.2 m}}
  arm:     {type: beam, measure: [hub.root, motor.mount], section: $tube, E: 70 GPa, G: 5 GPa,
            density: 1550 kg/m^3, segments: 6, damping: 0.02, orient: z}
```

- `rigid: true` (pin, weld, hinge, slider): an exact joint by constraint projection (SHAKE/RATTLE). No stiffness to
  choose and no time-step cost, but each step costs more (3D: about 40 us per hinge per step, the solve dominates a
  drive model); the force output is the reaction, `drift` the largest gap it closed. The two points
  must meet at the start (a gap is refused). Without `rigid`, joints are stiff spring-dampers: `k`, `c` (and `k_rot`, `c_rot` for the
  turn lock).
- hinge: 2D turns about the plane's normal; 3D about `axis` (world, at the start; then fixed in A). `k_twist` /
  `c_twist` add a torsion spring and damper about it.
- slider: B's point stays on the line through A's point along `axis` (fixed in A). Its measure output is how far B
  slid. `friction: <force>` adds Coulomb drag along it.
- beam: a co-rotational beam between two points, rigidly attached at both ends. Stiffness from `material` + a
  `section` block (rectangle, tube, rect_tube, i_beam, given_section) or numbers (`E`, `A`, `I`; 3D also `Iy`, `J`,
  `G`; or `EA`, `EI`/`EIz`, `EIy`, `GJ`). Mass from density x A x L, or `mass`. `segments: n` adds n - 1 node bodies so
  it bends along its length and vibrates; `end_mass: false` stops it adding half a segment's mass to each end body.
  `orient` (3D): the section's h direction (bending across it uses I / Ix). `hinge: auto` forms plastic hinges at
  Mp = sy Z (needs a material and a section with Z), or give a moment table. In 3D each axis yields on its own
  strength (Zy from the section) on the surface (My/Mpy)^2 + (Mz/Mpz)^2 = 1. `damping`: fraction of critical in
  the first bending mode (cantilever estimate; up to about 0.3). The fast segment modes are filtered out of it.
- Contacts in 2D and 3D: penalty (kelvin_voigt, hertz, elastoplastic on a `{contact: ...}` measure), or `impact`
  (rigid) on the same measure. See the editor's Help > Contact and joint methods for when to use which.

## Reserved

`directions:` and `angles:` stay reserved; use `along: dir` and body geometry instead.
