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
| `elastoplastic` | `table`, `one_sided`, `break_at` | `tabular_return` with a (k, acc_tab, F_tab) table from a block; `break_at` = plastic set at which it tears off / punches through and carries nothing after |
| `hertz` | `K`, `n` (1.5), `alpha` (0) | F = K s^n (1 + alpha ds/dt), compression only |
| `friction` | `F_c`, `v_eps`, `F_s`, `v_s` | regularized Coulomb, optional Stribeck peak F_s at speed scale v_s |
| `drag` | `K` | F = K v abs(v) (windage, quadratic air drag) |
| `force` | `P`, or `csv` | constant push P along +measure, or a table against time or the measure (below) |
| `drive` | `kv`, `R`, `I_max`, `gear`, `cells`, `R0`, `R1`, `C1`, `capacity`, `throttle`, `J_rotor` | DC motor + current-limited ESC (duty capped at the throttle) + LiPo 1-RC pack on one rotation body (or a twist measure in 2D/3D); `J_rotor gear^2` is added to that body |
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
when the element is removed (or made elastic) in the knockout check.

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
  crossed, the value is 10 x t_end and the report says "never reached").
- `unit` converts the SI result for display; failure limits are written in that same unit.
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

### run and report

```yaml
run: {t_end: 0.4 ms, dt: 20 ns, mc_n: 150, bounds: {shaft_set_mm: [0, 15]}}   # dt: auto = 0.02 x critical
# automatic step held inside bounds instead of a fixed one:  run: {t_end: 0.4 ms, dt_min: 5 ns, dt_max: 50 ns}
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

### editor (written by the desktop editor, ignored by the solver)

`editor: {layout: {"b:tip": [x, y], ...}, groups: [...], notes: [{text, x, y, width}], images: [{path, x, y, width,
caption}]}` keeps node positions, groups, canvas notes (Markdown and LaTeX) and image blocks (a picture linked from
the model's folder). Hand-written files can leave it out.

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
| `type: contact_friction`, `contact: <element>` | slip along that contact's surface: mu x contact force | sliding, rolling, scraping |

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
  choose and no time-step cost; the force output is the reaction, `drift` the largest gap it closed. The two points
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
