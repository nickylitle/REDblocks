# Verification report: horizontal wedge strike (legacy physics, block-built)

**horizontal wedge strike (legacy physics, block-built): VERIFIED | PASS 12 WARN 0 FAIL 0 SKIP 0 | 216 runs, 1.3 s**

| check | status | detail |
|---|---|---|
| determinism | PASS | bit-identical repeat |
| finite_outputs | PASS | all outputs finite |
| physical_bounds | PASS | 3 bounds respected |
| stability_margin | PASS | critical dt 7.76e-05 s / dt 1e-07 s = 775.6x |
| dt_convergence | PASS | bending deformation: err 0.015% (order 1.4); denting deformation: err 0.019% (order 0.84); wedge drop: err 0.0054% (order 0.86); weapon_energy_lost: err 0.0057% (order 0.79) |
| energy_ledger | PASS | worst closure residual 0.6% across 3 dt levels (tol 5%) |
| limit:weapon stopped -> no damage | PASS | matches expected limit |
| limit:rigid wedge -> no permanent set | PASS | matches expected limit |
| knockout:no_dent_plasticity | PASS | bending deformation +46.1%, denting deformation -100%, wedge drop +10.5%, weapon_energy_lost -50.7% |
| knockout:no_bend_plasticity | PASS | bending deformation -100%, denting deformation +2.02%, wedge drop -0.625%, weapon_energy_lost -3.13% |
| sensitivity | PASS | 4 inputs swept low/high (trends reported, not judged) |
| monte_carlo | PASS | 200 samples, 0 failed finite/bounds/ledger (0.0%) |

## Monte Carlo

| metric | P5 | P50 | P95 | top driver (rank corr) |
|---|---|---|---|---|
| bending deformation | 0.2596 | 0.7625 | 2.272 | t (-0.80), weapon_angle0_deg (-0.51), weapon_w (+0.18) |
| denting deformation | 0.9942 | 2.261 | 3.898 | weapon_angle0_deg (-0.96), weapon_w (+0.13), t (-0.08) |
| wedge drop | 2.755 | 3.695 | 4.017 | weapon_angle0_deg (-0.88), t (-0.34), weapon_w (+0.14) |
| weapon_energy_lost | 12.22 | 34.69 | 117.7 | weapon_angle0_deg (-0.84), t (+0.44), weapon_w (+0.08) |

## One-at-a-time sensitivity (trend report, not judged)

| parameter | metric | low | nominal | high |
|---|---|---|---|---|
| t | bending deformation | 2.225 | 0.75 | 0.391 |
| t | denting deformation | 2.248 | 2.484 | 2.054 |
| t | wedge drop | 3.982 | 3.584 | 3.439 |
| t | weapon_energy_lost | 25.79 | 34 | 63.36 |
| weapon_angle0_deg | bending deformation | 1.378 | 0.75 | 0.3478 |
| weapon_angle0_deg | denting deformation | 4.056 | 2.484 | 0.6799 |
| weapon_angle0_deg | wedge drop | 4.01 | 3.584 | 2.718 |
| weapon_angle0_deg | weapon_energy_lost | 95.69 | 34 | 6.234 |
| weapon_w | bending deformation | 0.6719 | 0.75 | 0.8203 |
| weapon_w | denting deformation | 2.291 | 2.484 | 2.622 |
| weapon_w | wedge drop | 3.451 | 3.584 | 3.688 |
| weapon_w | weapon_energy_lost | 35.84 | 34 | 51.06 |
| rate_factor | bending deformation | 0.7744 | 0.75 | 0.726 |
| rate_factor | denting deformation | 2.547 | 2.484 | 2.423 |
| rate_factor | wedge drop | 3.571 | 3.584 | 3.596 |
| rate_factor | weapon_energy_lost | 33.98 | 34 | 34.05 |
