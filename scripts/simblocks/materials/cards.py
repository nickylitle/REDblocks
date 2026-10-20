"""Material cards: one object that carries everything a sim (or a person) needs about a material,
saved as JSON and drawn as a one-page summary image (same dark report style as every figure REDblocks makes).

    card = laminate_card("T300/5208 [0/+-45/90]2s")              # catalog lamina + layup code
    card = laminate_card("my CF", lamina=dict(E1=135e9, E2=9e9, G12=5e9, nu12=0.3, rho=1550,
                         Xt=2000e6, Xc=1200e6, Yt=50e6, Yc=200e6, S=80e6, t_ply=0.2e-3), layup="[0/90]3s")
    card.save(); card.render("card")                              # JSON + PNG/SVG
    load_card("my CF")                                            # later sessions
    iso = card.as_isotropic("flexural")                           # feed Roark plate / beam blocks

Isotropic cards wrap catalog metals/polymers with their stress-strain curve source.
"""
import json
import math
import os
import re
from dataclasses import dataclass, field, asdict

import numpy as np

from .base import Lamina, Isotropic
from . import clt
from ..registry import block

CARD_DIR = os.path.join(os.path.dirname(__file__), "cards")


def _find_lamina(text):
    from .composites import LAMINAE
    t = text.lower().replace(" ", "")
    for k, l in LAMINAE.items():
        if k.lower().replace("_", "/") in t.replace("_", "/") or k.lower() in t or l.name.lower().replace(" ", "") in t:
            return k, l
    alias = {"t300": "T300_5208", "as4": "AS4_3501-6", "im7": "IM7_8552", "eglass": "Eglass_epoxy", "e-glass": "Eglass_epoxy",
             "fiberglass": "Eglass_epoxy", "kevlar": "Kevlar49_epoxy", "aramid": "Kevlar49_epoxy", "twill": "CF_3K_twill",
             "3k": "CF_3K_twill", "7781": "Eglass_7781"}
    for a, k in alias.items():
        if a in t:
            return k, LAMINAE[k]
    raise KeyError(f"no catalog lamina found in {text!r}; pass lamina=dict(...) to define one")


@dataclass
class LaminateCard:
    name: str
    lamina_name: str
    layup_code: str
    angles: list
    lamina: dict
    props: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    source: str = ""

    # ---------------------------------------------------------------- access
    @property
    def lamina_obj(self):
        return Lamina(**self.lamina)

    @property
    def abd(self):
        return clt.laminate_ABD(self.lamina_obj, self.angles)

    def as_isotropic(self, use="flexural"):
        return clt.laminate_as_isotropic(self.abd, use=use, name=self.name)

    def save(self, directory=None):
        d = directory or os.path.join(CARD_DIR, "user"); os.makedirs(d, exist_ok=True)
        p = os.path.join(d, _slug(self.name) + ".json")
        with open(p, "w") as fh:
            json.dump(asdict(self), fh, indent=1, default=float)
        return p

    def markdown(self):
        P = self.props
        rows = [f"# Material card: {self.name}", "", f"- lamina: {self.lamina_name} ({self.source})",
                f"- layup: `{self.layup_code}` = {self.angles}  ({len(self.angles)} plies)", ""]
        rows += ["| property | value |", "|---|---|"]
        for k, (lab, sc, unit) in _PROP_FMT.items():
            if k in P:
                rows.append(f"| {lab} | {P[k] * sc:.4g} {unit} |")
        rows += [""] + [f"- {n}" for n in self.notes]
        return "\n".join(rows) + "\n"

    def render(self, path_no_ext):
        return render_laminate_card(self, path_no_ext)


_PROP_FMT = {
    "h": ("thickness", 1e3, "mm"), "areal_mass": ("areal mass", 1.0, "kg/m^2"), "rho": ("density", 1.0, "kg/m^3"),
    "Ex": ("Ex (membrane)", 1e-9, "GPa"), "Ey": ("Ey (membrane)", 1e-9, "GPa"), "Gxy": ("Gxy", 1e-9, "GPa"),
    "nuxy": ("nu_xy", 1.0, ""), "Efx": ("Ex (flexural)", 1e-9, "GPa"), "Efy": ("Ey (flexural)", 1e-9, "GPa"),
    "D11": ("D11", 1.0, "N m"), "D22": ("D22", 1.0, "N m"), "D66": ("D66", 1.0, "N m"),
    "Sx_fpf": ("first-ply strength, x tension", 1e-6, "MPa"), "Sxc_fpf": ("first-ply strength, x compression", 1e-6, "MPa"),
    "Sy_fpf": ("first-ply strength, y tension", 1e-6, "MPa"), "Sxy_fpf": ("first-ply strength, shear", 1e-6, "MPa"),
    "Mx_fpf": ("first-ply bending moment, Mx", 1.0, "N m/m"), "alpha_x": ("CTE x", 1e6, "ppm/K"),
    "alpha_y": ("CTE y", 1e6, "ppm/K"), "specific_Efx": ("specific flexural modulus", 1e-6, "MN m/kg"),
}


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


@block("materials", "cards", "laminate", "from_text", kind="builder", source="CLT (Jones; Daniel & Ishai)",
       tags=("composite", "layup", "material card", "CF plate"))
def laminate_card(text, lamina=None, layup=None, t_ply=None, name=None, criterion="tsai_wu"):
    """Build a LaminateCard from text like 'T300/5208 [0/+-45/90]2s' or from lamina=dict(...) + layup=..."""
    code = layup or (re.search(r"\[[^\]]+\]\d*s?", text).group(0) if re.search(r"\[[^\]]+\]", text) else None)
    if code is None:
        raise ValueError("give a layup code like [0/+-45/90]2s")
    if lamina is None:
        key, lam = _find_lamina(text)
        src = f"catalog lamina {key}: {lam.note}"
        ld = asdict(lam)
    else:
        key = lamina.get("name", "user lamina")
        ld = dict(name=key, alpha1=0.0, alpha2=0.0, Vf=0.6, note="user-defined", t_ply=0.125e-3)
        ld.update(lamina)
        src = "user-defined lamina"
    if t_ply:
        ld["t_ply"] = t_ply
    angles = clt.parse_layup(code)
    card = LaminateCard(name=name or f"{key} {code}", lamina_name=key, layup_code=code, angles=angles, lamina=ld,
                        source=src)
    _compute(card, criterion)
    return card


def _compute(card, criterion):
    lam = card.lamina_obj; abd = clt.laminate_ABD(lam, card.angles)
    ec = clt.laminate_engineering_constants(abd)
    h = abd["h"]; rho = lam.rho
    P = dict(h=h, rho=rho, areal_mass=rho * h, **{k: ec[k] for k in ("Ex", "Ey", "Gxy", "nuxy", "Efx", "Efy")})
    D = abd["D"]; P.update(D11=D[0, 0], D22=D[1, 1], D66=D[2, 2])
    for key, N, M in (("Sx_fpf", (1, 0, 0), (0, 0, 0)), ("Sxc_fpf", (-1, 0, 0), (0, 0, 0)),
                      ("Sy_fpf", (0, 1, 0), (0, 0, 0)), ("Sxy_fpf", (0, 0, 1), (0, 0, 0))):
        R, _, _ = clt.first_ply_failure(abd, N=N, M=M, criterion=criterion)
        P[key] = R / h
    R, _, _ = clt.first_ply_failure(abd, M=(1, 0, 0), criterion=criterion)
    P["Mx_fpf"] = R
    e0, k = clt.laminate_response(abd, dT=1.0)
    P["alpha_x"], P["alpha_y"] = e0[0], e0[1]
    P["specific_Efx"] = ec["Efx"] / rho
    Bmax = np.abs(abd["B"]).max(); Amax = np.abs(abd["A"]).max()
    notes = []
    if Bmax > 1e-6 * Amax * h:
        notes.append("UNSYMMETRIC: extension-bending coupling (B != 0): plate will warp on cure and under in-plane load.")
    if abs(abd["A"][0, 2]) > 1e-6 * Amax:
        notes.append("UNBALANCED: extension-shear coupling (A16 != 0): in-plane load twists the panel's shape.")
    if abs(abd["D"][0, 2]) > 1e-3 * abs(D[0, 0]):
        notes.append(f"Bending-twisting coupling D16/D11 = {abd['D'][0, 2] / D[0, 0]:.3f} (normal for angle-ply stacks; "
                     "a point hit on the panel induces some twist).")
    iso = abs(ec["Ex"] / ec["Ey"] - 1) < 0.02 and abs(ec["Gxy"] / (ec["Ex"] / (2 * (1 + ec["nuxy"]))) - 1) < 0.02
    if iso:
        notes.append("In-plane quasi-isotropic: membrane stiffness is direction independent.")
    notes.append(f"first-ply failure by {criterion.replace('_', '-')}; laminate may carry more load after FPF (no progressive damage modelled).")
    card.props = P; card.notes = notes


@block("materials", "cards", "laminate", "load", kind="builder", source="saved card JSON", tags=("material card",))
def load_card(name, directory=None):
    """Load a saved LaminateCard by name (from cards/user/)."""
    d = directory or os.path.join(CARD_DIR, "user")
    p = os.path.join(d, _slug(name) + ".json")
    with open(p) as fh:
        return LaminateCard(**json.load(fh))


@block("materials", "cards", "laminate", "render", kind="builder", source="simblocks viz", tags=("material card", "figure"))
def render_laminate_card(card, path_no_ext):
    """Save a one-page picture (PNG + SVG) of a composite laminate, for reports and sanity checks: the ply stack
    drawn top to bottom, a polar plot of in-plane stiffness by direction (shows how anisotropic the layup is), and a
    table of its properties (thickness, density, moduli, first-ply strengths, coupling warnings). It is
    documentation only: sims use the card's numbers, not this picture. "Dark" is just the colour style of the image."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from ..viz import theme as T
    T.apply()
    fig = plt.figure(figsize=(12.5, 7.8))
    top = T.header(fig, card.name, f"{card.lamina_name}  |  {card.layup_code}  |  {len(card.angles)} plies, "
                   f"{card.props['h']*1e3:.3g} mm", tag="SIMBLOCKS // MATERIAL CARD")
    gs = fig.add_gridspec(1, 3, left=0.04, right=0.98, bottom=0.30, top=top - 0.07, width_ratios=[0.8, 1.05, 1.25], wspace=0.25)
    ax = fig.add_subplot(gs[0, 0]); ax.set_title("Stack (top to bottom)")
    n = len(card.angles); hpl = 1.0
    uniq = sorted(set(card.angles), key=lambda a: (abs(a), a))
    colmap = {a: T.SERIES[i % len(T.SERIES)] for i, a in enumerate(uniq)}
    for i, a in enumerate(card.angles):
        y = n - 1 - i
        ax.add_patch(Rectangle((0, y), 1, hpl * 0.92, fc=colmap[a], ec=T.SURFACE, lw=1.5))
        ax.text(1.06, y + 0.45, f"{a:+.0f}°", va="center", fontsize=7, color=T.INK_2, family=T.MONO)
        th = math.radians(a)
        cx, cy = 0.5, y + 0.46
        pos = ax.get_position(); W_, H_ = fig.get_size_inches()
        sx = pos.width * W_ / 1.45; sy = pos.height * H_ / (n + 0.4)        # inches per data unit
        dx, dy = math.cos(th), math.sin(th) * sx / sy                       # true on-screen angle
        Lh = min(0.32, 0.30 / max(abs(dy), 1e-9))
        ax.plot([cx - Lh * dx, cx + Lh * dx], [cy - Lh * dy, cy + Lh * dy], color=T.SURFACE, lw=1.6,
                solid_capstyle="round")
    ax.set_xlim(0, 1.45); ax.set_ylim(-0.2, n + 0.2); ax.axis("off")
    ax = fig.add_subplot(gs[0, 1], projection="polar"); ax.set_facecolor(T.PANEL)
    lam = card.lamina_obj; ths = np.radians(np.arange(0, 361, 3)); Ex = []; Ef = []
    for t_ in ths:
        rot = [a - math.degrees(t_) for a in card.angles]
        ec_ = clt.laminate_engineering_constants(clt.laminate_ABD(lam, rot))
        Ex.append(ec_["Ex"] / 1e9); Ef.append(ec_["Efx"] / 1e9)
    ax.plot(ths, Ex, color=T.SERIES[0], lw=1.8, label="membrane E"); ax.fill(ths, Ex, color=T.SERIES[0], alpha=0.10)
    ax.plot(ths, Ef, color=T.SERIES[1], lw=1.8, label="flexural E")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.16, 0.92), fontsize=7)
    ax.set_title("Modulus vs load direction (GPa)", pad=14)
    # no ring labels (an anisotropic curve always crosses some); print the values that matter instead
    fig.canvas.draw()
    rt = [r_ for r_ in ax.get_yticks() if 0 < r_ <= ax.get_ylim()[1]]
    step = rt[1] - rt[0] if len(rt) > 1 else (rt[0] if rt else 0)
    ax.set_yticklabels([])
    deg = np.degrees(ths) % 360
    def mm(c_):
        c_ = np.asarray(c_); return c_.max(), deg[c_.argmax()], c_.min(), deg[c_.argmin()]
    a1, a2, a3, a4 = mm(Ex); b1, b2, b3, b4 = mm(Ef)
    ax.text(0.5, -0.13, f"rings every {step:g} GPa\nmembrane E  max {a1:.3g} @ {a2:.0f}\u00b0   min {a3:.3g} @ {a4:.0f}\u00b0"
            f"\nflexural E  max {b1:.3g} @ {b2:.0f}\u00b0   min {b3:.3g} @ {b4:.0f}\u00b0",
            transform=ax.transAxes, ha="center", va="top", fontsize=7, color=T.INK_2, family=T.MONO, linespacing=1.5)
    ax.tick_params(colors=T.INK_3, labelsize=7); ax.grid(color=T.HAIRLINE)
    for lab in ax.get_yticklabels():
        lab.set_fontfamily(T.MONO)
    ax = fig.add_subplot(gs[0, 2]); ax.axis("off"); ax.set_title("Properties")
    y = 0.97
    for k, (lab, sc, unit) in _PROP_FMT.items():
        if k not in card.props:
            continue
        ax.text(0.0, y, lab, fontsize=7.6, color=T.INK_2, transform=ax.transAxes, va="top")
        ax.text(1.0, y, f"{card.props[k]*sc:.4g} {unit}", fontsize=7.6, color=T.INK, family=T.MONO,
                transform=ax.transAxes, va="top", ha="right")
        y -= 0.047
    yn = 0.105
    for nt in card.notes:
        col = T.STATUS["WARN"] if nt.startswith(("UNSYM", "UNBAL")) else T.INK_2
        fig.text(0.04, yn, "\u2022 " + nt, fontsize=7, color=col, va="top")
        yn -= 0.028
    T.footer(fig, card.source)
    return T.save(fig, path_no_ext)


@block("materials", "cards", "isotropic", "summary", kind="builder", source="catalog + curve loader", tags=("material card",))
def isotropic_card(text):
    """Card dict for a catalog metal/polymer (resolved from loose text) including its stress-strain source."""
    from . import get
    from .curves import resolve_material, load_curve
    key = resolve_material(text); m = get(key); c = load_curve(key, write=False)
    return dict(key=key, name=m.name, E=m.E, nu=m.nu, rho=m.rho, sy=m.sy, su=m.su, elong=m.elong, G=m.G,
                specific_strength=m.sy / m.rho, curve_source=c.source, note=m.note)
