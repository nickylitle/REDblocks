# Case study: combat robot (hinged ramp strike, vehicle drive)

User-supplied models rebuilt from the general blocks, kept as worked examples and regression anchors.
Nothing in the library depends on these files.

- `wedge_strike_from_blocks.py`: a spinning blade strikes a hinged ramp; legacy mode reproduces the original
  hand-written sim within 0.05%, curve mode uses tabular plasticity.
- `wedge_report.py`: full pipeline on that model (failure modes, schematic, signal flow, results report).
  `python wedge_report.py cad` or `python wedge_report.py sim`.
- `verify_wedge.py`, `rerun_new_physics.py`, `rerun_first_contact.py`, `compare_legacy.py`: verification and
  comparison runs.
- `flapjack_drive_legacy.py`: the user's original drive sim with switchable sourced tire, gearbox and battery
  models; `drive_sourced_compare.py` compares them.
