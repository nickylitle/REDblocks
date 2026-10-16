"""Stress-strain curves: generate from cited parameters, import digitized CSVs, convert, export.

Resolution order in `load_curve(name)`:
  1. A user CSV in curves/user/<key>.csv (your digitized test data wins)
  2. The best published fit in published_fits.json (citation, URL, table/equation travel into the CSV)
  3. Fallback: Ramberg-Osgood fit through E, 0.2% yield, ultimate and estimated uniform elongation
     (prefer="jc" swaps in the Johnson-Cook curve where parameters are published)
  4. neo-Hookean for elastomers without a published hyperelastic fit
  5. Linear-elastic to failure (composites, brittle)
Every Curve carries `source` so a sim report can state where its material law came from.

CSV format written and read (header lines start with '#'):
  # material: 7075-T6
  # source: ...
  # measure: engineering        (or: true, plastic)
  strain,stress_MPa
"""
import csv
import math
import os
import re
from dataclasses import dataclass, field

import numpy as np

from .base import Isotropic, Lamina
from ..registry import block
from .._jit import kernel

CURVE_DIR = os.path.join(os.path.dirname(__file__), "curves")

# ------------------------------------------------------------ name resolution
# "best common temper" defaults. Edit freely.
DEFAULT_TEMPER = {
    "7075": "7075-T6", "6061": "6061-T6", "2024": "2024-T3", "5052": "5052-H32",
    "4130": "4130_normalized", "4140": "4140_QT_30HRC", "4340": "4340_QT_315C", "1018": "1018_CD",
    "1045": "1045_CD", "17-4": "17-4PH_H900", "17-4ph": "17-4PH_H900", "304": "304_annealed",
    "ar500": "AR500", "s7": "S7_56HRC", "maraging": "maraging_300", "ti": "Ti6Al4V", "titanium": "Ti6Al4V",
    "ti64": "Ti6Al4V", "ti-6al-4v": "Ti6Al4V", "6al4v": "Ti6Al4V", "grade 5": "Ti6Al4V", "grade 2": "Ti_Gr2",
    "uhmw": "UHMW", "hdpe": "HDPE", "polycarbonate": "PC", "lexan": "PC", "nylon": "PA66_dry",
    "delrin": "POM", "acetal": "POM", "peek": "PEEK", "tpu": "TPU95A_print", "g10": "G10_FR4", "fr4": "G10_FR4",
    "copper": "Cu_C110", "brass": "brass_C360", "tungsten": "WHA_90W", "magnesium": "AZ31B",
}


@block("materials", "curves", "lookup", "name", kind="builder", source="alias table", tags=("temper", "resolve"))
def resolve_material(text):
    """Turn loose text ('yea it's 7075, best common temper', '4130 at 45 HRC') into a catalog key."""
    from . import _ALL
    t = text.lower()
    # exact catalog key or full name mentioned explicitly wins
    for k, m in sorted(_ALL.items(), key=lambda kv: -len(kv[0])):
        if k.lower() in t or getattr(m, "name", "").lower() in t:
            return k
    shore = re.search(r"(\d{2})\s*a\b", t)
    if shore and ("urethane" in t or "polyurethane" in t or "tpu" in t):
        S = int(shore.group(1))
        pool = ["TPU95A_print", "TPU85A_print"] if "tpu" in t else \
            ["urethane_20A", "urethane_40A", "urethane_60A", "urethane_90A"]
        return min(pool, key=lambda k: abs(int(re.search(r"(\d{2})A", k).group(1)) - S))
    hrc = re.search(r"(\d{2})\s*hrc", t)
    if "4130" in t and hrc and int(hrc.group(1)) >= 40:
        return "4130_45HRC"
    for alias in sorted(DEFAULT_TEMPER, key=len, reverse=True):
        if re.search(r"(?<![\w.])" + re.escape(alias) + r"(?![\w])", t):
            return DEFAULT_TEMPER[alias]
    raise KeyError(f"could not resolve a material from {text!r}")


# ------------------------------------------------------------ Curve object


@dataclass
class Curve:
    name: str
    eng_strain: np.ndarray
    eng_stress: np.ndarray
    E: float
    source: str
    extras: dict = field(default_factory=dict)

    @property
    def true(self):
        return eng_to_true(self.eng_strain, self.eng_stress)

    def plastic(self, extend_to=1.0, n_pts=200):
        """(eps_p, sigma_true) hardening table for the plasticity kernels, Hollomon-extended past necking."""
        et, st = self.true
        ep, sp = true_to_plastic(et, st, self.E)
        if extend_to and ep[-1] < extend_to:
            ep, sp = extend_hollomon(ep, sp, extend_to)
        grid = np.concatenate([np.linspace(0, min(ep[-1], 0.02), n_pts // 2, endpoint=False),
                               np.geomspace(max(min(ep[-1], 0.02), 1e-6), ep[-1], n_pts - n_pts // 2)])
        return grid, np.interp(grid, ep, sp)

    def write_csv(self, path=None, measure="engineering"):
        path = path or os.path.join(CURVE_DIR, f"{self.name}.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if measure == "engineering":
            x, y = self.eng_strain, self.eng_stress
        elif measure == "true":
            x, y = self.true
        else:
            x, y = self.plastic()
        with open(path, "w", newline="") as fh:
            fh.write(f"# material: {self.name}\n# source: {self.source}\n# measure: {measure}\n# E_Pa: {self.E:.6g}\n")
            fit = self.extras.get("fit") if self.extras else None
            if fit:
                for k in ("model", "equation_as_published", "params", "strain_measure", "valid_range",
                          "test_conditions", "citation", "url", "location", "confidence", "notes",
                          "verified_by", "applies"):
                    if k in fit:
                        fh.write(f"# fit_{k}: {fit[k]}\n")
            if (self.extras or {}).get("truncated"):
                fh.write(f"# TRUNCATED: {self.extras['truncated']}\n")
            if (self.extras or {}).get("consistency_warning"):
                fh.write(f"# CONSISTENCY_WARNING: {self.extras['consistency_warning']}\n")
            for nf in (self.extras or {}).get("nearest_fits", []):
                fh.write(f"# nearest_published_fit: {nf}\n")
            fh.write(f"# generated_by: simblocks.materials.curves ({self.extras.get('generator','') if self.extras else ''})\n")
            w = csv.writer(fh)
            w.writerow(["strain", "stress_MPa"])
            for a, b in zip(x, y):
                w.writerow([f"{a:.6g}", f"{b / 1e6:.6g}"])
        return path


# ------------------------------------------------------------ conversions


@block("materials", "curves", "uniaxial", "conversion", source="volume constancy (valid to necking)",
       tags=("true stress", "engineering"))
def eng_to_true(eng_strain, eng_stress):
    """Engineering -> true strain/stress: eps = ln(1+e), sigma = s(1+e)."""
    e = np.asarray(eng_strain, float); s = np.asarray(eng_stress, float)
    return np.log1p(e), s * (1.0 + e)


@block("materials", "curves", "uniaxial", "conversion", source="additive elastic-plastic split",
       tags=("plastic strain", "hardening table"))
def true_to_plastic(true_strain, true_stress, E):
    """True curve -> monotonic hardening table (eps_p, sigma) starting at eps_p = 0, sigma = yield."""
    et = np.asarray(true_strain, float); st = np.asarray(true_stress, float)
    ep = et - st / E
    keep = ep > 1e-6
    if not np.any(keep):
        return np.array([0.0]), np.array([st.max()])
    i0 = np.argmax(keep)
    ep = np.concatenate([[0.0], ep[i0:]]); sp = np.concatenate([[st[i0 - 1] if i0 else st[0]], st[i0:]])
    # enforce monotonic (strictly increasing eps_p, non-decreasing sigma)
    out_e, out_s = [ep[0]], [sp[0]]
    for a, b in zip(ep[1:], sp[1:]):
        if a > out_e[-1] + 1e-9:
            out_e.append(a); out_s.append(max(b, out_s[-1]))
    return np.array(out_e), np.array(out_s)


@block("materials", "curves", "uniaxial", "post_necking", source="Hollomon, Considere necking criterion",
       tags=("extrapolation",))
def extend_hollomon(eps_p, sigma, eps_p_max, n_add=60):
    """Extend a hardening table past necking with sigma = K eps^n matched in value and slope at the last point."""
    ep, sp = np.asarray(eps_p, float), np.asarray(sigma, float)
    e1, s1 = ep[-1], sp[-1]
    slope = (sp[-1] - sp[-2]) / (ep[-1] - ep[-2]) if len(ep) > 1 else 0.0
    n = max(min(slope * e1 / s1, 0.6), 1e-3) if e1 > 0 else 0.1
    K = s1 / e1 ** n if e1 > 0 else s1
    xs = np.geomspace(e1 * 1.01 if e1 > 0 else 1e-4, eps_p_max, n_add)
    return np.concatenate([ep, xs]), np.concatenate([sp, K * xs ** n])


# ------------------------------------------------------------ generators


def uniform_elongation_estimate(mat):
    """Uniform (necking) engineering strain estimate when no test value exists.
    Ratio uses family trends: high sy/su (little hardening) -> early necking."""
    el = mat.elong if math.isfinite(mat.elong) else 0.10
    r = mat.sy / mat.su
    frac = 0.75 - 0.45 * r          # ~0.35 of elongation for sy/su ~0.9, ~0.5 for 0.55
    return max(min(el * frac, el), 0.01)


@block("materials", "curves", "metal", "uniaxial", source="Ramberg & Osgood 1943, MMPDS form",
       tags=("ramberg-osgood", "n exponent"))
def ramberg_osgood_n(E, sy, su, e_u):
    """RO exponent n from 0.2%-offset yield and the ultimate at uniform engineering strain e_u."""
    ep_u = max(e_u - su / E, 0.0025)
    return math.log(ep_u / 0.002) / math.log(su / sy)


@block("materials", "curves", "metal", "uniaxial", kind="builder", source="Ramberg & Osgood 1943",
       tags=("ramberg-osgood", "curve"))
def ramberg_osgood_curve(mat, e_u=None, n_pts=160):
    """Engineering curve to UTS: e = s/E + 0.002 (s/sy)^n, n fit to pass through (e_u, su)."""
    given = e_u is not None
    e_u = uniform_elongation_estimate(mat) if e_u is None else e_u
    if mat.su <= mat.sy * 1.005:
        # datasheet gives no hardening (yield = max stress, typical of polymers / brittle prints):
        # elastic-perfectly-plastic plateau to the elongation, flagged as such
        ef = mat.elong if math.isfinite(mat.elong) else 0.05
        e = np.array([0.0, mat.sy / mat.E, max(ef, 1.5 * mat.sy / mat.E)])
        s_ = np.array([0.0, mat.sy, mat.sy])
        return Curve(_key_of(mat), e, s_, mat.E,
                     f"elastic-perfectly-plastic (datasheet has yield = max stress, no hardening data) "
                     f"({mat.name} catalog) to elongation {e[-1]:.3f}", dict(n=float("inf"), e_u=e[-1]))
    n = ramberg_osgood_n(mat.E, mat.sy, mat.su, e_u)
    s = np.concatenate([np.linspace(0, 0.7 * mat.sy, 20, endpoint=False),
                        np.linspace(0.7 * mat.sy, mat.su, n_pts - 20)])
    e = s / mat.E + 0.002 * (s / mat.sy) ** n
    return Curve(_key_of(mat), e, s, mat.E,
                 f"Ramberg-Osgood fit n={n:.2f} through E, sy, su ({mat.name} catalog), e_u={e_u:.3f} "
                 f"({'given' if given else 'estimated from elongation and sy/su'})", dict(n=n, e_u=e_u))


@block("materials", "constitutive_params", "metal", "flow_stress", kind="kernel",
       source="Johnson & Cook 1983", tags=("johnson-cook", "rate", "temperature"))
@kernel
def johnson_cook_stress(eps_p, rate, T, A, B, n, C, m, Tm, Tr, rate0=1.0):
    """Johnson-Cook flow stress (Pa). rate in 1/s, T in K. Rate term floored at rate0."""
    r = rate / rate0
    rate_term = 1.0 + C * math.log(r) if r > 1.0 else 1.0
    Ts = (T - Tr) / (Tm - Tr)
    therm = 1.0 - Ts ** m if Ts > 0.0 else 1.0
    return (A + B * eps_p ** n) * rate_term * therm


@block("materials", "constitutive_params", "metal", "rate_temperature", kind="kernel",
       source="Johnson & Cook 1983", tags=("johnson-cook", "rate", "scale"))
@kernel
def jc_scale(rate, T, C, m, Tm, Tr, rate0=1.0):
    """Johnson-Cook multiplier on a quasi-static curve: (1 + C ln(rate/rate0)) (1 - T*^m)."""
    r = rate / rate0
    a = 1.0 + C * math.log(r) if r > 1.0 else 1.0
    Ts = (T - Tr) / (Tm - Tr)
    return a * (1.0 - Ts ** m if Ts > 0.0 else 1.0)


@block("materials", "curves", "metal", "uniaxial", kind="builder", source="Johnson & Cook 1983",
       tags=("johnson-cook", "curve"))
def johnson_cook_curve(mat, rate=1.0, T=293.0, eps_p_max=None, n_pts=160):
    """Engineering curve from JC parameters (quasi-static by default), cut at necking (Considere)."""
    A, B, n, C, m, Tm, Tr = mat.jc
    ep = np.geomspace(1e-5, eps_p_max or 0.6, n_pts)
    sig = np.array([johnson_cook_stress(x, rate, T, A, B, n, C, m, Tm, Tr) for x in ep])
    et = ep + sig / mat.E
    # Considere: necking where dsigma/deps_true <= sigma
    dsig = np.gradient(sig, et)
    neck = np.nonzero(dsig <= sig)[0]
    iN = neck[0] if len(neck) else len(et) - 1
    et, sig = et[:iN + 1], sig[:iN + 1]
    e = np.expm1(et); s = sig / (1 + e)
    e = np.concatenate([[0.0, A / mat.E], e]); s = np.concatenate([[0.0, A], s])
    return Curve(_key_of(mat), e, s, mat.E, f"Johnson-Cook ({mat.note or 'catalog'}) at {rate:g}/s, {T:.0f} K",
                 dict(jc=mat.jc))


@block("materials", "curves", "brittle", "uniaxial", kind="builder", source="linear elastic",
       tags=("composite", "brittle"))
def linear_to_failure_curve(name, E, s_fail, n_pts=20):
    """Linear-elastic curve to failure (composite fiber direction, brittle ceramics, glass)."""
    s = np.linspace(0, s_fail, n_pts)
    return Curve(name, s / E, s, E, "linear elastic to failure")


@block("materials", "curves", "elastomer", "uniaxial", kind="builder", source="neo-Hookean (incompressible)",
       tags=("elastomer", "urethane", "TPU"))
def neo_hookean_curve(mat, stretch_max=None, n_pts=120):
    """Engineering uniaxial curve s = mu (lam - lam^-2), mu = E/3, to break elongation."""
    lam_max = stretch_max or 1 + (mat.elong if math.isfinite(mat.elong) else 3.0)
    lam = np.linspace(1, lam_max, n_pts)
    mu = mat.E / 3.0
    return Curve(_key_of(mat), lam - 1, mu * (lam - lam ** -2), mat.E, "neo-Hookean from Shore-derived E",
                 dict(mu=mu))


def _key_of(mat):
    from . import _ALL
    for k, v in _ALL.items():
        if v is mat:
            return k
    return mat.name.replace(" ", "_")


# ------------------------------------------------------------ CSV import


@block("materials", "curves", "import", "csv", kind="builder", source="user data",
       tags=("digitized", "WebPlotDigitizer", "csv"))
def read_curve_csv(path, E=None, measure=None, name=None):
    """Read a digitized curve CSV. Detects stress units (Pa, MPa, ksi, psi) from header,
    percent strain (if max > 1.5), and '# measure:' engineering/true. Returns Curve (engineering)."""
    meta = {}; rows = []; header = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                if ":" in line:
                    k, v = line[1:].split(":", 1); meta[k.strip().lower()] = v.strip()
                continue
            parts = [p.strip() for p in re.split(r"[,\t;]", line.strip()) if p.strip()]
            if not parts:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                header = " ".join(parts).lower()
    a = np.array(rows)
    x, y = a[:, 0], a[:, 1]
    h = (header or "") + " " + meta.get("units", "")
    scale = 1e6
    if "ksi" in h: scale = 6.894757e6
    elif "psi" in h: scale = 6894.757
    elif "gpa" in h: scale = 1e9
    elif re.search(r"\bpa\b", h) and "mpa" not in h: scale = 1.0
    y = y * scale
    if x.max() > 1.5 or "%" in h:
        x = x / 100.0
    order = np.argsort(x); x, y = x[order], y[order]
    measure = (measure or meta.get("measure", "engineering")).lower()
    if E is None:
        E = float(meta["e_pa"]) if "e_pa" in meta else _initial_slope(x, y)
    if measure == "true":
        e = np.expm1(x); s = y / (1 + e)
    elif measure == "plastic":
        et = x + y / E; e = np.expm1(et); s = y / (1 + e)
    else:
        e, s = x, y
    if e[0] > 0:
        e = np.concatenate([[0.0], e]); s = np.concatenate([[0.0], s])
    return Curve(name or meta.get("material", os.path.splitext(os.path.basename(path))[0]), e, s, E,
                 meta.get("source", f"user CSV {os.path.basename(path)}"))


def _initial_slope(x, y):
    sel = y <= 0.3 * y.max()
    if sel.sum() >= 2 and np.ptp(x[sel]) > 0:
        return float(np.polyfit(x[sel], y[sel], 1)[0])
    return float(y[1] / x[1]) if x[1] > 0 else float("nan")


# ------------------------------------------------------------ top-level loader


@block("materials", "curves", "lookup", "auto", kind="builder", source="see resolution order",
       tags=("stress strain", "csv", "temper"))
def load_curve(text_or_key, curve_dir=None, prefer=None, write=True, rate=None):
    """Get the stress-strain Curve for loose material text. prefer in {None,'user','jc','ro'}.
    rate (1/s): evaluate Johnson-Cook fits at this rate. Only raises stress above the fit's reference
    rate; never extrapolates below it (default None = the fit's reference rate, stated in the CSV)."""
    from . import get
    key = resolve_material(text_or_key)
    mat = get(key)
    d = curve_dir or CURVE_DIR
    from .custom import CURVES as _CUSTOM_CURVES
    if prefer in (None, "user") and key in _CUSTOM_CURVES:          # custom materials folder curve
        return read_curve_csv(_CUSTOM_CURVES[key], E=getattr(mat, "E", None), name=key)
    user = os.path.join(d, "user", f"{key}.csv")
    if prefer in (None, "user") and os.path.exists(user):
        c = read_curve_csv(user, E=getattr(mat, "E", None), name=key)
        return c
    if isinstance(mat, Lamina):
        return linear_to_failure_curve(key, mat.E1, mat.Xt)
    allfits = published_fits().get(key, [])
    fits = [f for f in allfits if f.get("applies", "exact") == "exact"]     # only exact-condition fits
    nearest = [f for f in allfits if f.get("applies") == "nearest"]
    if prefer == "jc":
        fits = [f for f in fits if f["model"] == "johnson_cook"] or fits
    elif prefer == "ro":
        fits = []
    if fits:
        c = curve_from_fit(mat, fits[0], rate=rate)      # records stored best-first
        y = _offset_yield(c)
        dev = y / mat.sy - 1.0
        if abs(dev) > 0.15 and mat.family != "elastomer":
            c.extras["consistency_warning"] = (
                f"fit 0.2% yield {y / 1e6:.0f} MPa is {100 * dev:+.0f}% vs catalog {mat.sy / 1e6:.0f} MPa "
                f"({mat.name}): fit lot/rate differs from the datasheet; check before sizing")
    elif mat.family == "elastomer":
        c = neo_hookean_curve(mat)
        c.source = "NO PUBLISHED FIT FOR THIS CONDITION -> " + c.source
    else:
        c = ramberg_osgood_curve(mat)
        c.source = "NO PUBLISHED FIT FOR THIS CONDITION -> " + c.source
    if nearest:
        c.extras["nearest_fits"] = [f"{f['model']} | {f['citation']} | {f['location']} | "
                                    f"NOT THIS CONDITION: {f['test_conditions']}" for f in nearest]
    if write:
        c.write_csv(os.path.join(d, f"{key}.csv"))
    return c


def _truncate_at_uts(c, mat):
    """Hyperelastic fits with negative-exponent terms can climb far past the real breaking stress.
    Stop the curve where it first reaches the datasheet UTS and say so."""
    if not (math.isfinite(mat.su) and c.eng_stress.max() > mat.su):
        return
    i = int(np.argmax(c.eng_stress >= mat.su))
    c.eng_strain, c.eng_stress = c.eng_strain[:i + 1], c.eng_stress[:i + 1]
    c.extras = dict(c.extras or {}, truncated=f"curve stopped at datasheet UTS {mat.su / 1e6:.1f} MPa "
                                             f"(strain {c.eng_strain[-1]:.2f}) before the fit's end of range")


def _offset_yield(curve, offset=0.002):
    e, s = curve.eng_strain, curve.eng_stress
    f = e - s / curve.E - offset
    i = np.nonzero(f >= 0)[0]
    return float(s[i[0]]) if len(i) else float(s.max())


# ------------------------------------------------------------ published-fit flow models
# Each takes plastic strain (true) and returns true stress. All are njit-safe kernels.

@block("materials", "constitutive_params", "metal", "flow_stress", kind="kernel", source="Hollomon 1945",
       tags=("power law",))
@kernel
def hollomon_stress(eps_p, K, n):
    """Hollomon power law sigma = K eps^n (eps = true strain; use total or plastic as the source states)."""
    return K * eps_p ** n


@block("materials", "constitutive_params", "metal", "flow_stress", kind="kernel", source="Swift 1952",
       tags=("power law", "pre-strain"))
@kernel
def swift_stress(eps_p, K, eps0, n):
    """Swift law sigma = K (eps0 + eps_p)^n."""
    return K * (eps0 + eps_p) ** n


@block("materials", "constitutive_params", "metal", "flow_stress", kind="kernel", source="Voce 1948",
       tags=("saturation",))
@kernel
def voce_stress(eps_p, sigma0, Q, b):
    """Voce saturation law sigma = sigma0 + Q (1 - exp(-b eps_p))."""
    return sigma0 + Q * (1.0 - math.exp(-b * eps_p))


@block("materials", "constitutive_params", "metal", "flow_stress", kind="kernel", source="Ludwik 1909",
       tags=("power law",))
@kernel
def ludwik_stress(eps_p, sigma0, K, n):
    """Ludwik law sigma = sigma0 + K eps_p^n."""
    return sigma0 + K * eps_p ** n


FIT_MODELS = ("ramberg_osgood", "johnson_cook", "swift", "swift_voce", "voce", "hollomon", "ludwik", "neo_hookean",
              "mooney_rivlin", "ogden", "linear")
FITS_FILE = os.path.join(os.path.dirname(__file__), "published_fits.json")


def published_fits():
    """Dict key -> list of fit records (see published_fits.json). Empty if the file is missing."""
    import json
    if not os.path.exists(FITS_FILE):
        return {}
    with open(FITS_FILE) as fh:
        return json.load(fh)


@block("materials", "curves", "metal", "uniaxial", kind="builder", source="the fit record's citation",
       tags=("published fit", "audit"))
def curve_from_fit(mat, fit, eps_p_max=None, n_pts=200, rate=None):
    """Build an engineering Curve from one published-fit record. The record's citation, URL,
    location and test conditions travel with the Curve into the CSV header."""
    p = fit["params"]; model = fit["model"]; E = mat.E
    measure = fit.get("strain_measure", "true plastic")
    if model == "ramberg_osgood":
        sy = p.get("sy_Pa", mat.sy); su = p.get("su_Pa", mat.su)
        n = p["n"]
        s = np.linspace(0, su, n_pts)
        e = s / p.get("E_Pa", E) + p.get("offset", 0.002) * (s / sy) ** n
        c = Curve(_key_of(mat), e, s, p.get("E_Pa", E), "")
    elif model == "linear":
        c = linear_to_failure_curve(_key_of(mat), E, p.get("s_fail_Pa", mat.su))
    elif model == "ogden":
        lam = np.linspace(1, p.get("stretch_max", 1 + (mat.elong if math.isfinite(mat.elong) else 3)), n_pts)
        mus, als = p["mu_Pa"], p["alpha"]
        # incompressible uniaxial nominal stress for W = sum mu_p/alpha_p (l1^a + l2^a + l3^a - 3)
        s = sum(m_ * (lam ** (a_ - 1) - lam ** (-a_ / 2 - 1)) for m_, a_ in zip(mus, als))
        c = Curve(_key_of(mat), lam - 1, s, 1.5 * sum(m_ * a_ for m_, a_ in zip(mus, als)), "")
        _truncate_at_uts(c, mat)
    elif model in ("neo_hookean", "mooney_rivlin"):
        lam = np.linspace(1, p.get("stretch_max", 1 + (mat.elong if math.isfinite(mat.elong) else 3)), n_pts)
        C10 = p.get("C10_Pa", p.get("mu_Pa", 0) / 2); C01 = p.get("C01_Pa", 0.0)
        s = 2 * (lam - lam ** -2) * (C10 + C01 / lam)
        c = Curve(_key_of(mat), lam - 1, s, 6 * (C10 + C01), "")
        _truncate_at_uts(c, mat)
    else:
        ep = np.geomspace(1e-5, eps_p_max or p.get("eps_max", 0.5), n_pts)
        if model == "johnson_cook":
            sig = p["A_Pa"] + p["B_Pa"] * ep ** p["n"]
            # rate term only ABOVE the fit's reference rate: JC's log term is not valid for extrapolating
            # down (7075-T6 Brar C = 0.059 would cut yield 41% at 1e-3/s)
            if rate and p.get("ref_rate") and rate > p["ref_rate"]:
                sig = sig * (1.0 + p["C"] * math.log(rate / p["ref_rate"]))
        elif model == "swift_voce":
            sw = p["K_Pa"] * (p.get("eps0", 0.0) + ep) ** p["n"]
            vo = p["k0_Pa"] + p["Q_Pa"] * (1 - np.exp(-p["b"] * ep))
            sig = p["alpha"] * sw + (1 - p["alpha"]) * vo
        elif model == "swift":
            sig = p["K_Pa"] * (p.get("eps0", 0.0) + ep) ** p["n"]
        elif model == "voce":
            sig = p["sigma0_Pa"] + p["Q_Pa"] * (1 - np.exp(-p["b"] * ep))
        elif model == "hollomon":
            x = ep + (0 if "plastic" in measure else 0)
            sig = p["K_Pa"] * x ** p["n"]
        elif model == "ludwik":
            sig = p["sigma0_Pa"] + p["K_Pa"] * ep ** p["n"]
        else:
            raise ValueError(f"unknown fit model {model}")
        et = ep + sig / E
        # cut at Considere necking unless the record gives a uniform-strain cap
        cap = fit.get("uniform_true_strain")
        dsig = np.gradient(sig, et)
        neck = np.nonzero(dsig <= sig)[0]
        iN = neck[0] if len(neck) else len(et) - 1
        if cap:
            iN = min(iN, int(np.searchsorted(et, cap)))
        et, sig = et[:iN + 1], sig[:iN + 1]
        e = np.expm1(et); s = sig / (1 + e)
        s0 = sig[0]
        e = np.concatenate([[0.0, s0 / E], e]); s = np.concatenate([[0.0, s0], s])
        c = Curve(_key_of(mat), e, s, E, "")
    rr = p.get("ref_rate")
    rate_txt = (f" @ {max(rate, rr):g}/s (ref {rr:g}/s)" if (rate and rr and model == "johnson_cook") else "")
    c.source = (f"{model}{rate_txt} | {fit.get('citation','')} | {fit.get('location','')} | {fit.get('url','')} | "
                f"tested: {fit.get('test_conditions','')} | confidence: {fit.get('confidence','')}")
    c.extras = dict(c.extras or {}, fit=fit)
    return c


@block("materials", "lookup", "catalog", "property", kind="builder", source="materials catalog (see MATERIAL_CATALOG.md)",
       tags=("E", "yield", "density", "system file"))
def material_property(material, prop):
    """One catalog property of a material given as text: E, nu, rho, sy, su, elong, G, tau_y, k, cp, alpha."""
    from . import get
    m = get(resolve_material(material))
    v = getattr(m, prop, None)
    if v is None or callable(v):
        raise ValueError(f"material has no property '{prop}'")
    return float(v)


_HARDENING = {
    "hollomon": (hollomon_stress, ("K", "n")),
    "swift": (swift_stress, ("K", "eps0", "n")),
    "voce": (voce_stress, ("sigma0", "Q", "b")),
    "ludwik": (ludwik_stress, ("sigma0", "K", "n")),
    "johnson_cook": (johnson_cook_stress, ("rate", "T", "A", "B", "n", "C", "m", "Tm", "Tr")),
}


@block("materials", "constitutive_params", "metal", "flow_stress_any",
       source="Hollomon 1945; Swift 1952; Voce 1948; Ludwik 1909; Johnson & Cook 1983",
       tags=("power law", "saturation", "pre-strain", "johnson-cook", "hardening", "flow stress"))
def hardening_stress(law, eps_p, K=None, n=None, eps0=None, sigma0=None, Q=None, b=None,
                     A=None, B=None, C=None, m=None, Tm=None, Tr=None, rate=1.0, T=None, rate0=1.0):
    """Flow stress (Pa) at plastic strain eps_p from a hardening law.

    law and its parameters:
      hollomon      K, n:              sigma = K eps^n (true strain; total or plastic as the source states)
      swift         K, eps0, n:        sigma = K (eps0 + eps_p)^n
      voce          sigma0, Q, b:      sigma = sigma0 + Q (1 - exp(-b eps_p))
      ludwik        sigma0, K, n:      sigma = sigma0 + K eps_p^n
      johnson_cook  A, B, n, C, m, Tm, Tr, rate (1/s), T (K, default Tr), rate0:
                    (A + B eps_p^n)(1 + C ln(rate/rate0))(1 - T*^m), rate term floored at rate0
    Returns the flow stress (Pa)."""
    key = str(law).strip().lower()
    if key not in _HARDENING:
        raise ValueError(f"hardening_stress: unknown law {law!r}; use one of {', '.join(_HARDENING)}")
    f, names = _HARDENING[key]
    vals = dict(K=K, n=n, eps0=eps0, sigma0=sigma0, Q=Q, b=b, A=A, B=B, C=C, m=m, Tm=Tm, Tr=Tr, rate=rate,
                T=Tr if T is None else T)
    miss = [k for k in names if vals[k] is None]
    if miss:
        raise ValueError(f"hardening_stress '{key}' needs {', '.join(miss)}")
    py = getattr(f, "py_func", f)
    args = [eps_p] + [vals[k] for k in names]
    return py(*args, rate0) if key == "johnson_cook" else py(*args)
