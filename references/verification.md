# Verification battery (`simblocks.diagnostics.verify.verify_all`)

Every assembled sim runs this before results are reported. It is automated: build a `SimSpec`, call
`verify_all(spec)`, write `rep.markdown()` and `plot_report(rep, path)`.

| # | check | what it catches | fails when |
|---|---|---|---|
| 1 | determinism | hidden state, uninitialized arrays, unseeded randomness | repeat run not bit-identical |
| 2 | finite_outputs | NaN/inf blow-ups | any output non-finite |
| 3 | physical_bounds | impossible results (negative set, penetration > thickness) | any declared bound violated |
| 4 | stability_margin | dt above the critical step | critical_dt / dt < 1 (WARN below 5) |
| 5 | dt_convergence | discretization error | Richardson-estimated error of the reported run > conv_tol (1%) |
| 6 | energy_ledger | missing reactions, wrong arms, energy creation | worst `_ledger_rel` across all dt levels > 5% |
| 7 | conservation | momentum/angular momentum drift in free systems | reported drift > tolerance |
| 8 | limit cases | model does not reduce to known answers | expectation not met |
| 9 | knockouts | dead blocks (wiring bugs) and wrong-direction physics | no metric changes, or declared sign wrong |
| 10 | sensitivity | (report only) one-at-a-time low/high trends | never fails |
| 11 | monte_carlo | fragile regions of input space | any sample non-finite, out of bounds or ledger-failing |
| 12 | anchors | drift from a locked field/calibration value | error above stated tolerance |

**Monotonicity is never checked.** Real contact results often show real non-monotonic behavior (dent depth
vs thickness), so trends are reported by the sensitivity table and judged by the engineer.

## Writing a good SimSpec

- `metrics`: the outputs that answer the question. Underscore names are diagnostics, not converged.
- `knockouts`: one per included physics term. Express a knockout as parameter overrides (add an
  `<term>_on` flag to the sim if needed). Give an expected sign only where physics forces it
  (removing plasticity cannot increase permanent set).
- `limits`: at least one. Zero speed -> zero damage; infinite yield -> zero set; elastic limit -> full
  rebound; zero friction -> no traction.
- `uncertain`: every input you are not sure of, with a defensible range. Include contact phase
  (first-contact angle) for strikes; that is usually the dominant driver.
- `bounds`: physical admissibility only, not expectations.
- `critical_dt`: from the stability blocks with effective masses.
- `anchors`: field-observed values (e.g. measured dent depth) with tolerance and a source string.

## When a check fails

Fix the model, never the tolerance. Typical causes:
- ledger: a force without its reaction, inconsistent arm, contact switching inside a step, energy
  stored in a spring not counted at the end.
- convergence with order < 0.5: discontinuous contact on/off; reduce dt or regularize the contact.
- dead knockout: the flag is not wired through `build()` to the loop.
- Monte Carlo failures: a parameter corner the model cannot represent; tighten bounds or extend the model.

The harness itself is tested: `tests/test_convergence_and_harness.py` plants each class of bug and
asserts the harness reports FAIL.
