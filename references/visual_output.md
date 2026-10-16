# Visual output: theme, diagrams, reports

Everything uses `simblocks.viz.theme` (dark mission-control style; Inter + JetBrains Mono bundled in
`simblocks/viz/fonts`). Call `theme.apply()` once; all builders below do it for you.

## Palette rules (validated with the dataviz validator against the #0b1018 surface)

- Categorical series, fixed order, never cycled: `SERIES = [blue #1b95d8, amber #c27c00, green #14a06e,
  violet #8c72f2, magenta #d84b8b]`. More than 5 series: small multiples.
- Status colors are reserved for pass/warn/fail (`STATUS`) and always carry a text label.
- Thresholds: `theme.limit_line(ax, y=..., label=..., status="FAIL")`. Hero numbers: `theme.stat_tile`.
- Numbers in the monospace face (`theme.style_ticks(ax)`), prose in Inter.

## Readability gate (automatic)

`theme.save()` and every page builder run `viz.qa.check_figure(fig)`: text overlapping text, data lines or
glyph strokes through text (gridlines, ticks and legend handles excluded; only the drawn, clipped part of a
line counts), text off the canvas, fonts under 6 pt. `theme.QA_STRICT = True` (default) raises
`LayoutError` and nothing is written. `theme.place_point_labels(ax, xs, ys, texts)` picks a clear position
for each value label. Schematics keep only short names and numbered callouts on the drawing; every value and
source goes into the parameter table, which is why they cannot collide.

## Physical schematic

```python
from simblocks.viz.diagram import Schematic
s = Schematic("Rotor strike on a hinged plate", "subtitle")
s.body("rotor", "rotor", "Rotor", info=["I = 7.56e-4 kg m^2", "w0 = 1257 rad/s"])
s.body("lip", "point", "")                      # massless junction for SERIES elements
s.body("face", "mass", "Plate face", info=[...]); s.body("frame", "mass", "Frame"); s.body("gnd", "ground", "")
s.link("rotor", "lip", "contact", "Tooth contact", "unilateral", "tip_face_penetration")
s.link("lip", "face", "elastoplastic", "Lip dent", "k = ...\nFy = ...", "circ_clamped_center_point")
s.link("face", "frame", "elastoplastic", "Root fold", "k = ...", "moment_curvature_rect -> hinge_table")
s.probe("face", "tip drop / puncture")
s.draw(path_no_ext)                              # PNG + SVG: drawing with callouts, key, 3-column parameter table
```
Element kinds: spring, dashpot, slider, elastoplastic, contact, rigid, wire, rot_spring, rot_elastoplastic.
Body kinds: mass, rotor, wheel, motor, face, point, ground. Elements in series go through a `point` body;
`lane=` offsets elements that are genuinely in parallel. Values must come from `build()`, never retyped.

## Signal flow (one time step)

`SignalFlow(title).stage(key, title, items=[(block, note)], role)` with roles state / geometry /
constitutive / force / integrate / output; `.edge(a, b, label)`, `.feedback(a, b, "x(t + dt)")`.

## Results report

`results_report(out_dir, name, question, answer, rep, history, history_panels, sweep, findings,
assumptions, sources, extra_svgs=[schematic.svg, signal_flow.svg])` writes answer / history / design /
verification pages (PNG + SVG), a PDF and a self-contained HTML. `sweep` takes `series`, `limit`,
`mark` (current design) and `input_labels` (readable names for Monte Carlo inputs).

Always open the PNGs and look before sending: label collisions, clipped tick labels, legends over data.
`glyph_sheet(path)` renders the symbol key.
