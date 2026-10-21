---
name: explicit-sim-builder
description: Guided explicit-dynamics sim builder (call it with no details and it walks you through setup). Builds fixed-time-step explicit dynamics sims from a prompt or drawing by assembling tested building blocks (Roark beams/plates/rings, Hertz and penalty contact, plasticity from real stress-strain curves, CLT composite material cards, puncture, fatigue, friction/tire, motors, inertia, aero, thermal), check them against user-defined failure modes, auto-verify (dt convergence, knockouts, Monte Carlo, energy ledger), and deliver dark-theme diagrams and results reports. Use whenever the user wants to simulate an impact, strike, collision, launch, spin-up, drive maneuver, deformation or heating, asks to write a sim, model a hit, or whether a part survives or is thick enough, wants a material curve applied ("it's 7075, whatever temper") or a layup made into a material card, or wants an existing sim re-run, debugged, diagrammed or verified, even without saying explicit or time step.
---

# Explicit sim builder

A library of ~311 registered building blocks (`scripts/simblocks/`), a YAML system-file compiler, a materials catalog with sourced
stress-strain curves and composite material cards, a verification harness every sim must pass, and a
mission-control themed output stack (schematics, signal-flow diagrams, results reports).
You assemble; the blocks carry the physics and their citations; the harness proves the result.

Why this structure: hand-writing a sim each time re-derives stiffnesses, invites unit bugs (one legacy
sim added N.m to radians in its hardening law), and leaves no way to know whether the
answer is converged or energy-consistent. Blocks are tested once against independent solutions; the
harness checks every assembly; the diagrams let the user check the model by eye.

## Start here: guided mode

**If the request is not complete, run the guided session in `references/guided_session.md` before anything else.**
A bare call to this skill, "help me simulate something", or a request missing the question, the geometry or the
failure definition all start it. The user only has to call the skill: each step is offered as one short
multiple-choice round (`AskUserQuestion`, recommended option first), already-known answers are skipped, and
"just run it" takes every default. The steps are:

1. **What are we doing?** New sim, rerun/change an existing sim, FE load export, or a material card.
2. **Setup card.** One table of bodies, geometry, materials, speeds, contacts, supports and outputs, with every
   assumption marked. "Correct anything, or say go."
3. **Failure modes.** Multi-select from `candidate_failure_modes`, each with a proposed limit.
4. **Run size.** One verified run (default), a small sweep here, or a big study on the user's machine.
5. **Build, verify, report** with a task list; answer line, report PDF, rerunnable YAML/sim file.
6. **What next?** Change and rerun, export FE loads, bigger study, or done.

When the request is already complete, skip straight to the workflow below and still end with step 6.

## Workflow

1. **Translate the prompt into a physical model.** In SI units: bodies and their DOFs, contacts (which
   surface hits which), materials, initial conditions, loads, and the outputs that answer the question.
   Drawings become vertex lists (`references/drawing_input.md`). One line per assumption. Ask at most one
   question, only if a wrong guess would change the answer; otherwise state the assumption and continue.

2. **Get the failure definition from the user.** Failure is what the user says it is for this part
   ("the ramp fails at a permanent 5 mm tip drop", "punctured"), never an implicit "over yield". If they have
   not said, propose candidates (`candidate_failure_modes(part_type)`) and ask which apply and what limits.
   Each becomes a `FailureMode` (`references/failure_modes.md`). Make the sim output the metric each mode
   needs (a geometric height, a peak force, a capacity). Placeholder limits are flagged as assumptions.

3. **List the physics terms, included and excluded.** Every included term becomes a knockout. Write the
   exclusions down ("gyro coupling excluded: spin axis parallel to yaw").

4. **Pick blocks.** Search before writing physics: `python scripts/catalog.py --find <word>`,
   `--domain structures --category plates`, `--show <block>`. `references/CATALOG.md` lists all blocks.
   Missing block: add it properly (`references/adding_blocks.md`): registered, cited, independently tested.
   Never inline a formula that could be a block.

5. **Materials.** `load_curve("7075 whatever temper")` resolves loose text (default-temper table) and returns
   the best published fit with its citation in the CSV header; a user CSV in `materials/curves/user/`
   overrides it. Report `NO PUBLISHED FIT FOUND` sources as such. Composites: `laminate_card("T300/5208
   [0/+-45/90]2s")` or a user-defined lamina dict gives a full card (stiffness, first-ply strengths, CTE,
   coupling warnings) that saves, reloads and renders; `card.as_isotropic("flexural")` feeds Roark plate
   blocks for near-isotropic stacks. Details: `references/materials.md`.

6. **Assemble. Prefer a system file** (`references/system_file.md`): if the model is 1D (one coordinate per
   body, elements acting on linear measures, including lever arms like `hub + R*rotor`) or planar 2D (rigid
   bodies in x, y and spin carrying points, circles and lines; elements on distances, gaps, contacts, pins,
   contact friction; gravity) or 3D (rigid bodies sliding in x, y, z that keep their orientation, spin about any
   axis, or rotate freely with an inertia tensor; points, spheres, cylinders and plane patches; ball joints), write a YAML
   system file instead of a loop. The compiler applies every reaction from one rule, generates the energy
   ledger, the automatic knockouts and the report, and the user can rerun it without Claude
   (`simblocks run|set|sweep|report model.yaml`). Give the user the YAML file as a deliverable. Check it with
   `simblocks check` before running. Only when the physics needs something the element types cannot express
   (adaptive dt, flexible bodies, laws no element type has) **assemble from the template**
   (`scripts/templates/sim_template.py`), and consider whether the missing piece should become a new element
   type or builder block instead.
   Template rules: Builders before the loop; the
   `@njit` loop calls only kernels. Semi-implicit Euler for contact/impact, RK4 only for smooth systems.
   dt from the stability blocks (~0.1x the smallest critical step, 200+ steps per contact). For multi-scale
   sims (long free motion + short impacts) use `adaptive_dt` + `stable_step`: stiffness- and gap-driven,
   capped, smooth growth; never deformation-driven (it would be coarse at the instant of first contact).
   An adaptive sim treats its `dt` argument as a refinement scale so the harness can still halve it. Every source
   and sink goes into the energy ledger (`_ledger_rel`). Offer `history=True` logging for the report.
   Patterns: `references/assembly.md`.

7. **Verify. Not optional.** `SimSpec` + `verify_all`: one knockout per physics term (sign only where
   physics forces it), at least one limit case, physical bounds, `critical_dt`, every uncertain input as a
   Monte Carlo range (sample contact angle from `sample_first_contact` for rotating-blade strikes), and the user's
   `failure_modes`. Never add monotonicity expectations: real behavior is often non-monotonic and the
   harness deliberately never judges trends. FAIL means fix the model, never loosen a tolerance.
   Details: `references/verification.md`.
   **Study size and sampled dimensions are the user's to define in the setup.** Propose them (as a job file,
   below), keep verification's own Monte Carlo small (`mc_n` <= 20), and never launch a large Monte Carlo or a
   configuration study unprompted. Prefer a wide space sampled sparsely (Sobol, tens of points) over a narrow
   space sampled densely.

7b. **Big studies run on the user's machine** (one-click: `redblocks_runner.exe`, built once with `scripts/packaging/build_exes.bat`; the user drops `<name>_job.yaml` + its sim `.py` next to it and sends back `<name>_results.zip`; Claude reads it with `--from` / `load_results`). Write a batch job (`simblocks batch job.yaml`; format in the
   header of `scripts/simblocks/batch.py`, worked example `examples/case_studies/combat_robot/
   spinner_vs_spinner_job.yaml`): configs to compare, `samples.n`, `samples.sampler: sobol`, `samples.uniform`
   ranges, optional `dt_check` and `verify`. The sim module exposes `batch_record(params, dt)` (JSON-safe
   record; keep raw event logs so thresholds can change later), `warmup()`, `make_spec(params, uncertain,
   mc_n)` and `pick_nominal(runs)`. The runner uses every core, resumes after interrupts, and packs one
   `<name>_results.zip`; read it back with `simblocks.batch.load_results` for interpretation and graphing.
   Make the sim fast before shipping it: gap-driven big steps while nothing can touch (cap as a multiple of dt
   so convergence still covers it), stop once the outputs are decided, no per-step allocation or BLAS/LAPACK
   calls in the loop. The spinner sim went from 2.4 s to 0.26 s per run this way with identical answers.

8. **Diagram the model. Every report embeds the physical Schematic, and an event-driven result (double hit, launch, failure sequence) also gets a sequence-of-events diagram (`SignalFlow(...).draw(..., tag='SIMBLOCKS // SEQUENCE OF EVENTS')`).** Build a `Schematic` (bodies + springs/sliders/contacts with their computed values
   and source blocks) and a `SignalFlow` (one time step) from the same numbers `build()` produced, so the
   user can check topology and magnitudes by eye. Series vs parallel matters: draw it as the physics is.
   Details: `references/visual_output.md`.

9. **Report.** `results_report(...)` writes the answer page (hero tiles per failure mode, margin
   distributions vs SF = 1), time histories with limits drawn in, design-around page (drivers of the
   governing margin, design-variable sweep with the failure band), verification page, plus one PDF and one
   self-contained HTML. In chat, lead with the answer and the numbers, then notable results, then what to
   design around.

10. **Readability gate before anything is sent.** Every figure goes through `viz.qa.check_figure` at save
   time (`theme.save`, `results_report`, cards, verification plot): text-text overlap, lines through text,
   off-canvas text, fonts under 6 pt. Strict mode raises instead of saving, so fix the layout (move the
   label, widen the box, use `theme.place_point_labels`, move detail into a table) rather than disabling
   the check. Then open every PNG and look at it: the checker cannot judge crowding, balance or emphasis.
   Never deliver a figure you have not looked at.
   Results from a NOT VERIFIED sim are reported as unverified with the failing checks named.

## Rules that prevent the known failure modes

- Energy-consistent coupling: a force doing work on body A is the reaction on body B at the same arm. Use
  `tip_face_penetration`'s `r_arm` (= -d pen / d angle) instead of hand arms.
- Yield and hardening live in constitutive kernels. `lag_return` only reproduces old results; new work uses
  `tabular_return` with a curve, `hinge_to_spring_table`, or a sourced hardening law.
- Strain-rate effects scale yield (Cowper-Symonds, `jc_scale`), never E.
- Governing failure mode = highest P(fail), ties by the smallest margin seen; size against the
  worst-allowable hit (e.g. alpha_max of the first-contact distribution), not the median.
- Compare materials at equal areal mass when the question is "which material".
- Reproducing a legacy sim: match its hard-coded constants (`mat_overrides=dict(E=2.1e11, nu=0.3)`) first.
- Blocks whose source starts with `LEGACY:` came from the user's own code, not a publication. Prefer the
  published equivalent; if a LEGACY block is used, list it under assumptions in the report.
- Friction, tire, drive and contact-damping choices: `references/friction_and_drive.md` (which published
  model replaces which legacy one, how to fit it from measured mu, what data is still unmeasured).
- Loads exported for another solver (rigid-part FE/MATLAB): export the force the flexible part transmits at the
  attachment point (`simblocks.diagnostics.loads`: `pulse_metrics`, `fe_load_table` keeps impulse exactly,
  `write_load_table`), only while contact is active plus unloading; free ringing afterwards is vibration, not load,
  and an axial mode at tens of kHz is averaged out. Rank worst cases by SIGNED impulse per direction, never by
  integrated magnitude, and keep only candidates that pass the harness's dt-convergence criterion.
- Ground reaction changes load response massively: an upward load ahead of the COM pitches the robot about its
  rear supports, the floor pushes back and roughly doubles vertical momentum while cutting rotation by about 2/3
  (spinner study). Any sim or exported load used for launch/vertical questions must include the supports (wheels,
  skid, chassis touchdown) and, for spinning weapons, the gyroscopic coupling (spin turns pitch into roll).
- Check geometry can reach a failure limit at all (a 30 mm face at 60 deg can only rotate the tip down 4 mm).

## Files

- `scripts/simblocks/` library (`registry.py` explains the taxonomy); `scripts/tests/` (run
  `python -m pytest -q scripts/tests` after any library change).
- `scripts/examples/case_studies/` application case studies built from the general blocks (e.g.
  `combat_robot/`: hinged-ramp strike with full report pipeline, vehicle drive sim). Nothing in the library
  depends on them.
- `scripts/simblocks/viz/fonts/` Inter + JetBrains Mono (OFL), bundled so figures look identical everywhere.
- `references/BLOCK_LIBRARY.html` searchable block library (diagram, inputs, outputs, system-file snippet per
  block); regenerate with `simblocks library references/BLOCK_LIBRARY.html` after adding blocks.
- `scripts/simblocks/app/` the desktop node editor (`simblocks app`, or the installed REDblocks app built and installed by
  `scripts/packaging/build_exes.bat`): Blender-style nodes that read and write system files, run, study (sweeps x Monte
  Carlo on chosen cores, asks before big studies), verify and report with no Claude. Point users there for 1D models.
  It also has live results, templates, groups, CSV export nodes, notes with LaTeX, custom materials and blocks folders,
  and a full in-app Help (F1) with every block's citation and source. Custom blocks: `simblocks.registry.CUSTOM_TEMPLATE`;
  custom materials: `simblocks.materials.custom.TEMPLATE`. Sections not in the list: `given_section` (I required; c, Z,
  A, J optional).
- `references/HOW_IT_WORKS.md` how the library, compiler, time step, ledger and harness work, for the curious.
- `references/guided_session.md` the guided intake (start here when the request is incomplete).
- `references/`: CATALOG.md, MATERIAL_CATALOG.md (generated), system_file (YAML format), assembly,
  verification, failure_modes, visual_output, materials, drawing_input, friction_and_drive, adding_blocks.
- `scripts/simblocks/compiler/` system-file compiler (loader with line-numbered errors, units, safe
  expressions, 1D engine); `scripts/simblocks/cli.py` the `simblocks` command; `scripts/pyproject.toml`
  makes `scripts/` pip-installable so the user can run models standalone (`pip install .`).
- `scripts/examples/systems/*.yaml` three verified example system files (flywheel spin-up, rotor shaft impact, panel drop);
  each matches its hand-written twin in `examples/trials/` (tested).
