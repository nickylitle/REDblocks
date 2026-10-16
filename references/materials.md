# Materials, stress-strain curves and composites

## Loose text to a curve

```python
from simblocks.materials import load_curve, resolve_material, get
c = load_curve("yea it's 7075 with whatever the best common temper is")   # -> 7075-T6
c.source            # model | citation | table/eq | URL | tested: conditions | confidence
c.eng_strain, c.eng_stress, c.true, c.plastic()   # plastic() = (eps_p, sigma) hardening table
c.write_csv(path)   # header carries the full audit trail
```

Resolution: explicit catalog key or full name in the text > hardness hint ("4130 at 45 HRC") >
default-temper table (`curves.DEFAULT_TEMPER`, editable) > error.

Curve priority:
1. `simblocks/materials/curves/user/<key>.csv` (user's own test or digitized data). Any units: header
   words `MPa`, `ksi`, `psi`, `GPa`, `Pa`; strain in % detected automatically; `# measure: true|engineering|plastic`.
2. Best published fit in `published_fits.json` (records sorted best-first: condition match, then confidence).
3. Fallback Ramberg-Osgood through catalog E, yield, ultimate, estimated uniform elongation; the source
   string then starts `NO PUBLISHED FIT FOUND`, and the report must say so.

Supported fit models: ramberg_osgood, johnson_cook, swift, voce, hollomon, ludwik, neo_hookean,
mooney_rivlin, linear. Every record keeps `equation_as_published`, `strain_measure`, `valid_range`,
`test_conditions`, `citation`, `url`, `location`, `confidence`.

## Using a curve in the loop

- axial member: `k, acc_tab, F_tab = curve_to_spring_table(curve, A, L)` -> `tabular_return(...)`
- bending hinge: `moment_curvature_rect(curve, b, h)` or `moment_curvature_polygon(curve, xy)` ->
  `hinge_table(mk, L_hinge)` -> `hinge_to_spring_table(hinge)` -> `tabular_return(...)`
- rate: multiply the yield table by `cowper_symonds_factor(rate, D, q)` or `jc_scale(...)`; never scale E.

## Composites (CLT)

```python
from simblocks.materials import clt, get
lam = get("T300_5208"); ang = clt.parse_layup("[0/+-45/90]2s")
abd = clt.laminate_ABD(lam, ang)                       # A, B, D, z, h
clt.laminate_engineering_constants(abd)               # Ex, Ey, Gxy, nuxy, Efx, Efy
e0, k = clt.laminate_response(abd, N=(Nx, Ny, Nxy), M=(Mx, My, Mxy), dT=-100)
clt.ply_stresses(abd, e0, k, dT=-100)                 # material-axis stresses, top/bottom of each ply
clt.first_ply_failure(abd, N=(1, 0, 0), criterion="tsai_wu")   # load multiplier, ply, angle
iso = clt.laminate_as_isotropic(abd, use="flexural")  # feed Roark plate blocks (near-isotropic stacks only)
```

Mixed stacks: pass a list of Lamina, one per ply. Micromechanics: `lamina_from_constituents(FIBERS["T300"],
MATRICES["epoxy"], Vf=0.6)` (strengths are rough estimates; replace with test data).

`python scripts/catalog.py --materials` lists every key; `references/MATERIAL_CATALOG.md` has the property tables.

## Composite material cards

```python
from simblocks.materials import laminate_card, load_card
card = laminate_card("T300/5208 [0/+-45/90]2s")          # catalog lamina found in the text + layup code
card = laminate_card("my CF", lamina=dict(name="my CF", E1=135e9, E2=9e9, G12=5e9, nu12=0.3, rho=1550,
                     Xt=2000e6, Xc=1200e6, Yt=50e6, Yc=200e6, S=80e6, t_ply=0.2e-3), layup="[0/90]3s")
card.props     # h, areal_mass, Ex/Ey/Gxy/nuxy (membrane), Efx/Efy (flexural), D11/D22/D66,
               # first-ply strengths x/xc/y/shear, first-ply Mx, CTE, specific flexural modulus
card.notes     # UNSYMMETRIC (B != 0), UNBALANCED (A16 != 0), bending-twist D16, quasi-isotropic flags
card.save(); load_card("my CF"); card.render("out/card")   # JSON in materials/cards/user, PNG + SVG card
iso = card.as_isotropic("flexural")                       # Roark plate blocks for near-isotropic stacks
```
Isotropic summary: `isotropic_card("7075")` (properties + stress-strain source).

## Published-fit policy (v0.2.4)

- `simblocks/materials/research/raw/*.json` keeps every research result as found (audit trail).
  `research/curate.py` is the reviewed layer that writes `published_fits.json`; run
  `python -m simblocks.materials.research.curate` after editing it.
- Each fit record says `applies: exact` (same alloy and temper/condition as the catalog key) or `nearest`
  (different temper, hardness, grade or test mode). Only exact fits become the default curve. Nearest fits
  are written into the CSV header as `nearest_published_fit: ... NOT THIS CONDITION`.
- Johnson-Cook rate terms are applied only above the fit's reference rate, never extrapolated below it.
- If an exact fit's 0.2% yield differs from the catalog datasheet yield by more than 15%, the CSV carries
  `CONSISTENCY_WARNING` and the report must list it under assumptions.
- Hyperelastic fits (Ogden, Mooney-Rivlin) are cut at the datasheet UTS (`TRUNCATED` header line).
- Datasheets that give yield = max stress (most polymers, printed parts) fall back to an
  elastic-perfectly-plastic curve, flagged as such.
- Many catalog entries have `_min` twins holding spec minimums; the plain key is the typical value.
