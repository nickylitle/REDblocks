# User-defined failure modes

Failure is defined by the user for the part in question, then checked on every run.

```python
from simblocks.diagnostics.failure import FailureMode, candidate_failure_modes
modes = [
  FailureMode("tip drop", metric="tip_drop_perm_mm", limit=5.0, fails_when="above",
              origin="user: part fails at a permanent 5 mm tip drop"),
  FailureMode("puncture", check=lambda o: (o["peak_dent_force_N"] >= o["plug_capacity_N"],
                                           o["plug_capacity_N"] / o["peak_dent_force_N"]),
              origin="user: punctured", assumption="tooth patch 6.35 mm x t"),
]
spec = SimSpec(..., failure_modes=modes)
```

- `fails_when="above"`: SF = limit / value. `"below"`: SF = value / limit. Custom `check` returns (failed, SF).
- The harness evaluates every mode on the nominal run and every Monte Carlo sample: P(fail) with a Wilson
  95% interval (0 of 400 still means up to ~0.95%), SF percentiles, lowest SF seen, and the inputs that
  drive each mode's margin (rank correlation).
- Governing mode: highest P(fail); ties broken by the smallest SF seen.
- A failing design gives `failure_modes` = WARN. That is a design result, not a model error: verification
  can still be VERIFIED.

## Turning words into metrics

| user says | sim must output | typical blocks |
|---|---|---|
| "tip/lip drops X mm" | permanent geometric height change (rotation set + local dent, signed) | `tip_face_penetration`, hinge + dent sets |
| "punctured / pushed through" | peak contact force and plug capacity | `shear_plug_capacity`, `punch_perimeter` |
| "bends more than X deg" | permanent rotation set | `hinge_to_spring_table` + `tabular_return` |
| "bolts shear / tear out" | peak joint force per fastener | `bolt_shear_capacity`, `tear_out_capacity`, `bearing_capacity` |
| "plate cracks" (composite) | ply stress ratio | `first_ply_failure`, `ply_stresses` |
| "fatigues out after N fights" | stress amplitude + mean per cycle | `sn_life`, `fatigue_sf`, `miner_damage`, `fatigue_kf` |
| "loses traction / gets stuck" | wheel normal load, slip duration | `three_point_normal_loads`, `tire_force_2d` |

Before running, check the geometry can reach the limit at all. If the mode is unreachable, say so: the
result is "cannot fail this way in this model", which is different from "passes".
