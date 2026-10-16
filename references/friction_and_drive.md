# Friction, tire, drive and contact damping

All blocks below carry their citation in `source`. Status words: VERIFIED (equation read in the original
or an official doc), secondary (read in a paper quoting the original), PARTIAL (vendor data, garbled
extraction, or equation numbers unconfirmed). Data with sources: `simblocks.friction.data`.

## Which block to use

| Need | Use | Replaces (LEGACY) | Basis |
|---|---|---|---|
| Wheel traction from measured peak and sliding mu | `brush_mu`, `brush_force_2d` (+ `brush_mu_s_for_peak`) | `tire_mu`, `tire_force_2d` | Svendenius 2007 brush model, two frictions |
| Wheel traction from a full mu-slip curve | `magic_formula` + `magic_formula_fit` | same | Pacejka MF, fit relations per arXiv:2502.05479 |
| Low-speed slip / load-transfer lag | `relaxation_lag` (over distance) | fixed 10 ms lag | Romano et al. 2023; Lee et al. 2017 |
| Sliding skid with stiction | `stribeck_mu` (cited now), `stribeck_mu_delta` | none | Liu 2015 Eq. 6, LuGre g(v) |
| Stick-slip, presliding, dwell | `lugre_step` | none | Canudas de Wit 1995 |
| Smooth friction for control or implicit use | `makkar_friction`, `simscape_friction` | none | Makkar 2005; Simscape |
| Rolling resistance | `rolling_torque` | `rolling_resistance_torque` (same form) | Hofmann 2024 definition |
| Gearbox back-driving | `backdrive_efficiency` + `gearbox_output_torque_dir` | `gearbox_output_torque` | Matsuki 2019; Sim & Ramos |
| Battery sag with recovery | `ecm_constant_power` | `bus_voltage` (R0-only, still valid) | Plett 2015 Ch. 2 |
| Contact restitution, linear | `damping_for_restitution` (cited now), `restitution_force_terminated` | none | Kawaguchi 1992; Schwager & Poschel 2007 |
| Contact restitution, Hertz | `hunt_crossley` + `hysteresis_damping_factor(model="flores")` | none | Machado et al. 2012 review |
| Tooth indentation onset | `blunt_punch_line_force` | `tab_shear_out_force` | Prandtl/Hill; Chaijit 2006 |
| Ploughing drag ratio | `challen_oxley_mu` | (pi M_p term) | Challen & Oxley 1979 |
| Lip tear-out | `tear_out_capacity` (cited now) | none | Melcon & Hoblit; AISC J3-6c |

## Traps the sources exposed

- **Brush mu_s is not the peak.** It is the adhesion limit. With mu_s = 2.0, mu_k = 1.67 the brush curve
  peaks near 1.68. Convert a measured peak with `brush_mu_s_for_peak` (2.0 / 1.67 gives mu_s = 3.10).
- **Back-driven efficiency is lower than forward.** Using eta_f in both directions (the legacy block)
  under-predicts the torque the load must supply to back-drive. G = 20, eta_f = 0.8 gives eta_b = 0.75.
- **Clipped contact raises restitution.** A spring-dashpot tuned for e = 0.5 with the classic formula
  bounces at 0.55 when its force is clipped at zero. Use `restitution_force_terminated` to check.
- **Hunt-Crossley damping is for e near 1.** For dent-type hits (low e) use the Flores factor.
- **Slip is singular at V = 0.** `vmin` regularization keeps it finite; `relaxation_lag` over distance is
  the published fix and simply holds the value at rest.

## Still unmeasured (no published value found)

- Urethane wheels on painted plywood: slip stiffness, mu-slip curve, rolling resistance.
- Steel screw heads on painted plywood: mu_s, mu_k.
Measure them and add `status="MEASURED"` rows to `friction.data`. Until then the report lists the values
used as assumptions.
