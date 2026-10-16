# Adding blocks (growing the catalog to hundreds)

## Where it goes

`simblocks/<domain>/<category>/<module>.py` (structures has a category folder level; smaller domains keep
categories inside one module until a module passes ~400 lines, then split by category).
Domains are fixed in `registry.DOMAINS`. Add a new domain there only for a genuinely new physics area
(e.g. `fluids` for CFD-lite, `electromagnetics`).

## Template

```python
from ..registry import block      # adjust dots for depth
from .._jit import kernel

@block("structures", "plates", "rectangular_clamped", "uniform_pressure", kind="builder",
       source="Roark 7e Table 11.4 case 8a", tags=("panel", "armor"))
def rect_clamped_pressure(mat, a, b, t):
    """One-line summary that appears in the catalog. Then formula, units, validity limits."""
    ...

@block("contact", "penalty", "general", "normal_force", kind="kernel", source="penalty method")
@kernel                            # kernels: floats/ints/numpy arrays only, return floats or tuples
def penalty_linear(pen, k):
    """..."""
```

Rules:
1. `@block` goes ABOVE `@kernel`.
2. `source` is a real citation: book + table/case or equation, or paper. If you derived it, say
   "derivation" and put the derivation in the docstring.
3. Builders return dicts with consistent keys within a category (beams return `k, delta, M_max, F_y, F_c`;
   sections return `A, Ix, Iy, c, S, Z, J, tau_coef, kappa`).
4. Every block gets a test in `scripts/tests/` that checks it against an INDEPENDENT method: a numerical
   solution (FE, BVP, ODE integration, series), a different closed form, a textbook example value, or a
   conservation law. Re-typing the same formula in the test does not count.
5. Time-integrated kernels also get a fixed-step dt convergence test.
6. Run `python -m pytest -q scripts/tests` and `python scripts/catalog.py --write`.

## Roadmap seeds (not yet built)

structures: curved beams (Roark ch. 9), rings (Table 9.2), thick plates (Mindlin), sandwich panels,
shells of revolution, notch stress concentration (Peterson), fatigue (S-N, Basquin, Goodman).
contact: Hertz elliptical (general curvatures), JKR adhesion, cutting/shearing force models, rubber
indentation. constitutive: Zerilli-Armstrong, viscoelastic (Prony), foam crush, G'Sell-Jonas polymers.
aero: ground effect, Magnus, compressible corrections. thermal: transient fins, semi-infinite contact
flash temperature, motor two-node thermal model. actuators: FOC BLDC with dq currents, belt/chain drives.
