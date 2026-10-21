# How simblocks works

This page explains what happens between "simulate this hit" and the report: how the pieces are built, how
they connect, how data moves every time step, and how the result is checked. It is for the curious. You do
not need any of it to use the skill.

---

## 1. The big picture

A simulation here is **assembled, not written**. Three layers do the work:

| Layer | What it is | Where it lives |
|---|---|---|
| **Blocks** | Small tested functions. Each one does one piece of physics, with a citation. | `scripts/simblocks/` |
| **Assembly** | A YAML system file (compiled automatically), or a hand-written loop for 3D or special cases. | `*.yaml`, case-study `.py` files |
| **Harness** | A battery of checks that every assembled sim must pass before its numbers are trusted. | `simblocks/diagnostics/verify.py` |

```
 your question
      │
      ▼
 physical model ──► blocks picked from the catalog ──► assembly (YAML or loop)
                                                              │
                                                              ▼
                                         harness: converged? energy closes? wired?
                                                              │
                                                              ▼
                                       report: answer, diagrams, verification page
```

The point of the split: **physics is written once and tested once**, then reused. A new sim is mostly wiring,
and wiring is the part the harness checks.

---

## 2. Blocks

Every block is a Python function registered in a catalog with a fixed address:

```
domain / category / geometry / load        e.g.  structures/beams/cantilever/end_point
```

and a **source** (Roark table and case, a textbook equation, a paper). There are about 310 of them across
12 domains: materials, structures, contact, constitutive, friction, actuators, rigid body, geometry, aero,
thermal, integrators and diagnostics.

Blocks come in **three kinds**:

| Kind | When it runs | What it does | Example |
|---|---|---|---|
| **Builder** | Once, before the sim starts | Turns geometry and material into numbers | `beam_plastic_spring`: 30 x 6.35 mm AR500 bar becomes a stiffness and a yield table |
| **Kernel** | Every time step, inside the loop | Turns the current state into a force (or current, or voltage) | `hertz_sphere_force`, `coulomb_tanh`, `esc_supply`, `ecm_constant_power` |
| **Analytic** | On demand | A closed-form answer, used for sizing and for checking sims | `spin_up_time`, Roark deflection formulas |

**Kernels are deliberately simple.** They take plain numbers and return plain numbers, so they can be compiled
to machine code (numba) and run millions of times per second. A kernel never knows what it is connected to.
A contact kernel only sees "how far am I overlapping, and how fast" and answers "this much force".

**Every block is tested** against an independent answer before it goes in the catalog: a hand calculation, a
published table, or a second method. The tests live in `scripts/tests/` (about 195 of them).

---

## 3. Materials

Materials feed the builders. `load_curve("7075 whatever temper")` resolves loose text to a catalog entry and
returns a **published stress-strain fit** with its citation. A curve file you drop in `materials/curves/user/`
overrides it.

Composite plates go through classical laminate theory: `laminate_card("T300/5208 [0/+-45/90]2s")` returns the
full stiffness matrices, first-ply-failure strengths and warnings about coupling.

Builders then turn the curve into what a sim needs. For example, a bending arm becomes a **moment-curvature
table**, which becomes a force-deflection table with a yield point and hardening. The sim never sees the curve
itself, only the table the builder made from it.

---

## 4. Assembly with a system file

For models where each body has **one coordinate** (a translation or a rotation), you describe the model in a
YAML file and the compiler builds the sim. No loop is written by hand.

```yaml
bodies:
  rotor: {dof: rotation, inertia: 8.5e-4 kg m^2, v0: 12000 rpm}
  hub:   {dof: translation, mass: 0.40 kg}
  wedge: {dof: translation, mass: 0.15 kg}

elements:
  tooth: {type: hertz, measure: hub + 0.095 * rotor - wedge, K: 2e10, n: 1.5}
```

### 4.1 The measure: how links are made

Every element acts along a **measure**: a straight sum of body coordinates.

```
s = c1 * q1 + c2 * q2 + ... + constant
```

`s` is how far the element is stretched or compressed. The coefficients are the **lever arms**. A plain spring
between two bodies has coefficients +1 and -1. A blade tip has the blade radius in front of the rotation angle.

The compiler turns every measure into **one row of a table, `C`**: one row per element, one column per body.

| element | rotor | hub | wedge |
|---|---|---|---|
| tooth | 0.095 | 1 | -1 |

**That row is the whole connection.** It is built once, before the run, and never changes.

### 4.2 What the compiler builds

| Array | Holds | Changes during the run? |
|---|---|---|
| `q`, `v` | Position and velocity of every body | Yes, every step |
| `m` | Mass or inertia of every body | No |
| `C` | The link table | No |
| `typ` | Which kernel each element uses | No |
| `P` | Each element's numbers (stiffness, yield table, motor Kv...), filled by builders | No |
| `st`, `aux` | Each element's memory (plastic set, battery charge, ...) | Yes, by its own kernel |

---

## 5. One time step

Every step (from tens of nanoseconds for impacts to a tenth of a millisecond for a battery fight) does the same
five things:

```
   q, v  (body states)
     │
     │  1. s = C · q        how stretched each element is
     │     ṡ = C · v        how fast
     ▼
   kernel for each element row
     │  2. F = law(s, ṡ, numbers, memory)      and the kernel updates its own memory
     ▼
     │  3. Q = -Cᵀ · F      force on every body, through the same row
     ▼
     │  4. v += dt · Q / m  (semi-implicit Euler: velocity first,
     │     q += dt · v       then position with the new velocity)
     ▼
   5. new q, v  ──► next step
```

**Nothing tells a spring how compressed to be.** Compression comes from where the bodies are. The force comes
from the compression. The motion comes from the force. Peak forces and deflections are outputs, never inputs.

**Why the same table is used in both directions:** the force an element puts on body A and the reaction it
puts on body B come from the same row of `C`. They cannot disagree. A lever arm used to compute the stretch is
automatically the lever arm used to apply the torque. This removes the most common hand-written bug, a
reaction applied at the wrong place or not at all.

**Kernels never call each other.** Each reads its own row's `s` and `ṡ`, its own numbers and memory, and
returns a force. That is why a block can be dropped into any model.

### Example: a blade tooth hitting a wedge

1. The tooth has not reached the face: `s` is negative, the contact kernel returns 0, nothing happens.
2. Next step the tooth is 0.01 mm into the face: the kernel returns about 5 kN.
3. That force slows the rotor (as a torque of 0.095 x F), pushes the hub back and shoves the wedge.
4. Over the next steps the overlap grows, peaks and shrinks as the bodies respond, and the force follows it.

### Element memory

Some elements remember things between steps:

- A **plastic spring** keeps its permanent set. Its kernel (`tabular_return`) checks each step whether the load
  exceeds the current yield force, and moves the permanent set if it does.
- A **motor drive** keeps the battery's state of charge and the recovery-voltage state of its equivalent
  circuit.

Only the element's own kernel writes to its memory.

### The same step in 2D

In a 2D model every body has three coordinates (x, y, spin) and carries its own points, circles and lines. A
measure is no longer a fixed row of numbers: each step it is worked out from where the geometry is now.

1. Each element finds its two **application points** (where the tooth touches the face, the two ends of a rope)
   and its **direction** (the face normal, the rope's line).
2. `s` is the penetration, length or gap; `ṡ` is the relative velocity of the two material points along that
   direction.
3. The element pushes body A at its point with `+F` along the direction and body B with `-F`. The push becomes a
   force and a torque on each body from where the point really is, so a contact's lever arm is exactly the
   perpendicular distance from the pivot to the line of push. No arm is typed by hand.
4. Friction on a contact rides on that contact's force and acts along its surface.

Two rules keep contacts honest: a contact can only start by crossing its line from outside (never by appearing
deep inside after sliding round an end), and elastic energy still stored when a contact drops is booked as
dissipated. Gravity acts at each centre of mass and its work enters the energy ledger as an input.

### And in 3D

3D adds a third axis (z up) and real rotation. A body either keeps its orientation, spins about one axis (any
vector), or tumbles freely with its full inertia tensor. A freely rotating body carries its angular momentum: the
torques from every element change it, and the spin rate follows from the momentum and the body's current
orientation. That is why a spinning weapon resists being tilted and why a top precesses instead of falling: nobody
writes a gyroscopic term, it comes out of the same step. Geometry is points, spheres, cylinders (disc slices) and finite plane patches, and
the same application-point rule gives every force its true lever arm in 3D.

---

## 6. Hand-written loops (special cases)

Some physics does not fit "one number per body": two robots tumbling in 3D, contact between a deforming blade
and a box, or a whole fight with a timed script of spin-ups and pushes. Those sims are written as a loop from a
template, using the **same kernels**, in the same order:

```
state  ──►  geometry: where are the contact points?
       ──►  kernels: contact, friction, bending, ESC current, battery voltage
       ──►  apply each force at its point, and the reaction on the partner
       ──►  integrate
```

In 3D the link table becomes an `_apply` helper that puts a force at a point on a body (force plus torque) and
the reaction on the other body at the same point.

Data passes between kernels as plain variables. In the battery fight, each step:

```
blade speed        ──►  esc_supply           ──►  weapon power
wheel speed        ──►  motor_current        ──►  drive power
weapon + drive + idle electronics            ──►  total power
total power        ──►  ecm_constant_power   ──►  bus voltage, battery current
bus voltage        ──►  next step's ESCs
battery current    ──►  drains the charge
```

Fast 3D sims also use a few speed tricks that do not change the answer: bigger steps while nothing can touch
(proved by a separating-axis gap check), stopping once the outputs are decided, and no memory allocation inside
the loop. The spinner sim went from 2.4 s to 0.26 s per engagement with identical results.

---

## 7. The energy ledger

Every sim keeps books, every step:

```
energy in (motors, batteries, starting spin)
   =  energy stored (springs, kinetic energy, battery)
   +  energy dissipated (damping, friction, plastic work, copper loss, windage)
   +  energy that left (into a target, into the floor)
```

Each element computes its part from its own law, independently of the integration. The mismatch is reported
as `_ledger_rel`.

**What it catches:** a force applied to one body but not its partner, a kernel that creates energy, a contact
that returns more energy than it absorbed, or a time step
too coarse for a stiff element. A ledger error above 5% fails the sim.

---

## 8. The harness

The harness wraps the whole sim and runs it many times with changes. It does not look inside. It only asks
whether the outputs behave the way a correct model must.

| Check | What it does | What it catches |
|---|---|---|
| **Determinism** | Same inputs twice | Hidden randomness, uninitialized memory |
| **Finite and bounded** | No NaN or infinity, every output inside physical bounds | Blow-ups, impossible values |
| **Stability margin** | Compares the step to the stiffest element's critical step | A step that is about to go unstable |
| **dt convergence** | Runs at dt, dt/2, dt/4; estimates the true answer (Richardson) | Answers that depend on the step size. Over 1% fails |
| **Energy ledger** | At every step size | Wiring errors, energy sources that should not exist |
| **Limit cases** | Inputs with a known answer ("weapon never spins: only idle draw") | Physics that misbehaves at the edges |
| **Knockouts** | Turns each physics term off, one at a time | Dead wiring: a term that changes nothing was never connected |
| **Sensitivity** | Nudges each uncertain input | Which inputs matter (reported, never judged) |
| **Monte Carlo** | Samples the uncertain inputs (small, 20 runs or fewer here) | The spread of the answer |
| **Failure modes** | Scores every run against your limits | Probability of failure and safety factors |

**Two rules the harness never breaks:**

1. **No monotonicity checks.** Real behavior is often non-monotonic (wedge damage against plate thickness is one real example),
   so the harness never expects "more X gives more Y". Trends are reported, not judged.
2. **A failed check means the model gets fixed.** Tolerances are never loosened to make a sim pass. A sim that
   fails is reported as NOT VERIFIED with the failing checks named.

---

## 9. Failure modes

Failure is **what you say it is** for that part, never an automatic "over yield". Each mode is written down
with its origin:

```yaml
failure:
  - {name: tip drop, metric: tip_drop_mm, limit: 5, fails_when: above, origin: "user: 5 mm permanent drop"}
```

Every run gets a safety factor per mode (limit / demand). The **governing mode** is the one most likely to
fail, with ties broken by the smallest safety factor. A mode the geometry can never reach (a short face that
cannot rotate far enough to drop 5 mm) is reported as UNREACHABLE, not as a pass.

---

## 10. Big studies on your machine

Large Monte Carlo studies are yours to define: which inputs vary, over what ranges, and how many samples.
They are written as a **job file**:

```yaml
simblocks_batch: 1
name: spinner_vs_spinner
sim: spinner_vs_spinner.py
entry: batch_record
dt: 2.5e-7
configs: {elastic-plastic (both): {}, rigid blades: {flexH: 0.0, flexV: 0.0}}
samples:
  n: 64
  sampler: sobol
  uniform: {v_close: [1.0, 5.0], y_off: [-0.040, 0.040]}
dt_check: {n: 16, factors: [2, 4]}
verify: {entry: make_spec, mc_n: 20}
```

`redblocks_runner.exe` (or `simblocks batch job.yaml`) runs it on every core, saves progress as it goes so an
interruption loses nothing, and packs one `<name>_results.zip` of plain JSON. Sobol sampling spreads points
evenly, so a wide space is covered well with tens of samples instead of hundreds.

---

## 11. Output and the readability gate

Every figure uses the dark mission-control theme. Reports contain:

- **Answer page:** the headline number, one tile per failure mode, safety-factor spreads.
- **Time histories** with your limits drawn in.
- **Design-around page:** which inputs drive the governing margin, and a sweep with the failure band.
- **Verification page:** every harness check with its status.
- **Schematic** of the model as built: bodies, springs, contacts, each labelled with its computed value and the
  block it came from. Event results (a double hit, a launch) also get a sequence-of-events diagram.

Before anything is saved, each figure goes through an automatic **layout check**: overlapping text, lines
through labels, text off the page, fonts under 6 pt. A figure that fails is not saved. Then every image is
opened and looked at, because a checker cannot judge crowding or balance.

---

## 12. Exporting loads to another solver

For a rigid-part FE model in another solver, the sim exports the force the flexible part actually passes to the
attachment point:

- only while contact is active, plus the unloading after it (ringing afterwards is vibration, not load);
- ranked by **signed impulse per direction**, never by total magnitude;
- only cases that pass the time-step convergence check;
- simplified to a short table that keeps the impulse exactly (`fe_load_table`), in your axes.

---

## 13. The desktop editor

`simblocks app` (or the installed **REDblocks** app: Windows search, Start Menu, pinnable to the taskbar) opens a native window: a
Blender-style node editor around the same compiler. `packaging/build_exes.bat` builds it and offers to install it for
the current user (no admin rights; uninstall from Settings > Apps).
Nothing runs online and nothing needs Claude.

- **2D mode** (DIM 2D). Bodies get x / y / spin freedoms, a start pose and velocity, and their own points, circles
  and lines. Elements pick what they measure (coordinates, distance, along, contact) and the features on each body;
  Contact, Pin joint and contact_friction come from the library; gravity is on the Model tab. 1D models switch to 2D
  as the same physics (translation bodies move in x, rotation bodies spin).
- **Units on wires.** A value only feeds an input of the same kind (a length cannot feed an angle); a refused wire
  says why in a popup.
- **Nodes are the YAML sections.** Value, Builder, Body, Element, Load, Output, Failure, plus two modifiers: Range of
  values and Monte Carlo. Wires carry values, bodies, signals, element states and metrics, and a socket only accepts the
  type it can use.
- **Single values are typed into the node.** A number box plus a unit dropdown filtered to that field's dimension
  (force fields offer N, kN, lbf). Exponent notation works: 1.1e5 shows "= 110,000" under the field.
- **Varying a value means plugging a modifier into it.** Range of values runs once per value; Monte Carlo samples
  between low and high. **Study** runs every range value against every Monte Carlo sample on the cores picked in the top
  bar, asks first above 100 runs or about a minute, and warns when more cores are picked than there are runs (each run
  is single threaded).
- **Default units** (SI, mm, US) sit in the top bar and are saved in the file. Changing them asks first; values still on
  the old default unit are converted to the new default (same physics), units chosen by hand stay as they are, and
  failure limits follow their output's unit.
- **Run, Verify and Report** call the same functions as the command line. Reports land in an `out` folder next to the
  saved model.
- **Selection and focus.** Clicking a wire selects it: it turns bold and is drawn above every node. Ctrl+click adds
  wires or nodes. With anything selected, nodes and wires not connected to it darken and desaturate, and the connected
  wires come to the front. Body wires are pink, so they never blend with the blue text and highlights.
- **Settings: colours.** Every colour is one named token. Six themes (Blueprint, Paper, Graphite, Graphite vivid,
  Colour-blind safe, Candy) set all of them, and any single colour can be changed with a picker. Saved per user in
  `~/.simblocks/app.json`, never in model files.
- **Timeout.** AUTO estimates a run (time steps x elements / the solver speed measured on this computer) and stops it at
  5x the estimate, at least 10 s; a Study stops at 5x (runs x time per run / cores), at least 30 s. Limit sets your own
  per-run time, Off never stops. A stopped run returns what it computed so far, marked with the simulated time reached.
- **Progress card.** Jobs longer than a moment show their stages (compile, solve, results; each Verify check with the
  Monte Carlo sample count; Study runs) with a bar and time left estimated from earlier runs on this computer.
- **Solver on its own core.** The compiled solver releases Python's lock while it integrates, so on a 2+ core machine
  live runs and big runs leave the editor fully responsive.
- **Live run.** Every edit that compiles re-runs the model (Range and Monte Carlo inputs held at their middle values,
  with a banner saying so) and the Results tab updates in place. Live runs never write files; they pause if one run takes
  over 2.5 s.
- **Templates.** New offers a blank model or a template: spinner hit, drop test, weapon spin-up with battery, drive push.
- **Editing.** Undo/redo, Shift-click and box select, copy, cut, paste and duplicate (names stay unique and `$refs`
  follow), Arrange.
- **Groups.** Ctrl+G folds nodes into one group node whose values stay editable on its face; Tab opens it; groups can be
  saved to a personal library and reused in any model. Positions, groups and notes are saved under `editor:`.
- **CSV export node.** Plug signals, states or bodies into its columns; Run writes a time-vs-value CSV with a chosen
  folder, naming pattern, time unit, row count and header style.
- **Notes.** Free text on the canvas with LaTeX ($...$ and $$...$$) rendered offline by a bundled KaTeX.
- **Your folders.** Folders connects a custom materials folder (YAML files, optional curve CSVs) and a custom blocks
  folder (Python files with `@block` functions). Both work in the editor, the command line and study workers.
- **Block library search** is forgiving (typos, partial words, a few synonyms) with filters by kind and domain.
- **Help** (F1) is the full manual inside the app: every feature, every node and element type, every block with its
  citation, a usage snippet and its Python source to copy, and the materials table. Each node's info popover links to
  its entry.

## 14. Glossary

| Term | Meaning |
|---|---|
| **Body** | Something with mass that moves: a rotor, a chassis, a wedge |
| **Element** | Something that connects bodies and makes force: a spring, a contact, a motor |
| **Measure** | The stretch of an element, written as a sum of body coordinates with lever arms |
| **`C` table** | All measures stacked as rows; used to compute stretches and to apply forces |
| **Builder** | Block that runs once and turns geometry and material into numbers |
| **Kernel** | Block that runs every step and turns the current state into a force |
| **Semi-implicit Euler** | Update velocity from force, then position from the new velocity; stable for contact |
| **Ledger** | The energy books; closure within 5% is required |
| **Knockout** | A rerun with one physics term turned off, to prove it was wired in |
| **Richardson** | Estimating the zero-step-size answer from runs at dt, dt/2 and dt/4 |
| **Sobol** | A sampling pattern that covers a space evenly with few points |


## Fitting parameters to test data (parameter estimation)

`redblocks estimate model.yaml --data test.csv --signal chassis.vx --vary mu_tire=1:2.5 k_tire=5kN/m:50kN/m`
(also the MCP tool `estimate`, and `simblocks.diagnostics.estimate.estimate` in Python) finds the parameter values
that make a model history match a measured one: bounded least squares on the misfit at the measurement times, one
full run per evaluation, started from the best of a spread of trial points (a fit to an oscillation has false
minima a period apart). It reports each value with its 1-sigma uncertainty from the fit's Jacobian (a parameter the
data cannot see shows a huge one), their correlation, the RMS misfit and R^2. The CSV's first column is time and the
second the measurement (or name them with --time-col / --y-col, with --time-unit and --unit).

## Replay (Animate)

The editor's Animate button replays a run in 3D (three.js, bundled): 1d bodies in lanes, 2d bodies in the x-y plane,
3d bodies with their logged orientation quaternions. 1d and 2d are seen straight from above and only pan and zoom.
- 1d: a sliding body a spring holds is scaled up so small travel shows (the scale is printed); a sliding body nothing
  holds (a car) drives along a road with distance marks and the camera follows it. A turning body's links attach at
  fixed points beside it, so they never whip round with the spin.
- Springs, dampers, joints and beams (solid tubes) are drawn between their points. Loads that only push (force, drag,
  friction, motors, circuit motors) are arrows on the body, sized against the largest; a torque on a 1d turning body
  is a curved arrow. Electronics are never drawn as lines.
- Body shapes (box, cylinder, disc, rod, ring, tube, sphere) are drawn as solids; a 2d rod lies along its farthest
  point. A planet (gravity: {planet: ..}) is drawn round the origin with the top of its air; point masses are dots of
  a fixed size on screen.
- A model with beams that moves less than 2 % of its size is drawn moving bigger (the factor is printed).
- F fits everything on screen (it fits when opened), Space plays, the arrow keys step a frame (shift: ten), Home and
  End jump.
- While the run is solved a progress card shows the stage, a bar and the time, with Stop. A watchdog stops a run that
  goes badly wrong (4x the estimate, at least 2 minutes, never shorter than the model's own timeout, even when that
  is off); the replay then shows how far it got.

It reuses the last Run's history when the model is unchanged.
