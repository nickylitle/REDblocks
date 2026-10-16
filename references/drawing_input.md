# Geometry from a drawing, photo or CAD file

1. **Image or sketch** (`.png`, `.jpg`, a phone photo of a part on paper):
   `outline_from_image(path, ref_length_m=0.1, ref_length_px=<measured>)` or `px_per_m=...`.
   It thresholds (Otsu), traces the largest silhouette and its holes, simplifies, converts to metres
   with y up and the centroid at the origin. Check the result by plotting it over the image before use.
   For a photo, the part must be shot square-on with a ruler or known dimension in the same plane.
2. **DXF** (laser/waterjet file): `outline_from_dxf(path, units_to_m=1e-3)`.
3. **Simple shapes:** `outline_rounded_bar(L, W, r_corner)` or write the vertex list by hand.

Then feed the vertices to:
- `polygon_plate(outer, t, rho, holes)` -> mass, centroid, polar inertia (rotor MOI from a drawing)
- `polygon_section(xy)` -> A, I, S, Z of a cross-section
- `moment_curvature_polygon(curve, xy)` -> full-curve M(kappa) of an arbitrary section

When the drawing gives dimensions as text instead, use them directly; tracing is for when it doesn't.
State the scale source in the report (ruler, stated dimension, or DXF units).
