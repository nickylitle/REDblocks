"""Micromechanics and Classical Laminate Theory (CLT).

Conventions (Jones, "Mechanics of Composite Materials"; Daniel & Ishai):
  * strain vector [eps_x, eps_y, gamma_xy] uses ENGINEERING shear strain
  * ply angle theta (deg) measured from laminate x to fiber direction 1, CCW positive
  * z measured from midplane, positive down through the stack, ply 0 is the top ply
  * N, M are force and moment resultants per unit width (N/m, N.m/m)
"""
import math
import re
import numpy as np

from .base import Lamina, Isotropic
from ..registry import block

# ---------------------------------------------------------------- micromechanics


@block("materials", "micromechanics", "unidirectional", "fiber_direction",
       source="rule of mixtures (Voigt)", tags=("composite", "Vf"))
def rule_of_mixtures(E_f, E_m, Vf):
    """Longitudinal property by rule of mixtures: P = Pf Vf + Pm (1 - Vf). Works for E1, nu12, rho."""
    return E_f * Vf + E_m * (1.0 - Vf)


@block("materials", "micromechanics", "unidirectional", "transverse",
       source="Halpin & Kardos 1976", tags=("composite", "E2", "G12"))
def halpin_tsai(P_f, P_m, Vf, xi):
    """Halpin-Tsai transverse/shear property. xi = 2 for E2 (circular fibers), 1 for G12."""
    eta = (P_f / P_m - 1.0) / (P_f / P_m + xi)
    return P_m * (1.0 + xi * eta * Vf) / (1.0 - eta * Vf)


@block("materials", "micromechanics", "unidirectional", "all", kind="builder",
       source="ROM + Halpin-Tsai", tags=("composite", "lamina"))
def lamina_from_constituents(fiber, matrix, Vf, t_ply=0.125e-3, name=None):
    """Build a Lamina from fiber + matrix + fiber volume fraction. Strengths are rough first estimates:
    Xt from fiber ROM, others matrix-dominated. Replace with test data when you have it."""
    E1 = rule_of_mixtures(fiber.E1, matrix.E, Vf)
    E2 = halpin_tsai(fiber.E2, matrix.E, Vf, 2.0)
    G12 = halpin_tsai(fiber.G12, matrix.G, Vf, 1.0)
    nu12 = rule_of_mixtures(fiber.nu12, matrix.nu, Vf)
    rho = rule_of_mixtures(fiber.rho, matrix.rho, Vf)
    Xt = fiber.Xt * Vf + matrix.Xt * (1 - Vf) * (matrix.E / fiber.E1)
    Xc = 0.6 * Xt
    Yt = matrix.Xt * 0.6
    Yc = 2.5 * matrix.Xt
    S = matrix.Xt * 0.9
    return Lamina(name or f"{fiber.name}/{matrix.name} Vf={Vf:.2f}", E1, E2, G12, nu12, rho,
                  Xt, Xc, Yt, Yc, S, t_ply, 0.0, 0.0, Vf, "micromechanics estimate")


# ---------------------------------------------------------------- lamina stiffness


@block("materials", "laminates", "ply", "plane_stress", kind="builder",
       source="Jones 2.62", tags=("CLT", "Q"))
def lamina_Q(lam):
    """Reduced stiffness matrix Q (3x3) of a ply in its material axes."""
    d = 1.0 - lam.nu12 * lam.nu21
    Q11 = lam.E1 / d
    Q22 = lam.E2 / d
    Q12 = lam.nu12 * lam.E2 / d
    return np.array([[Q11, Q12, 0.0], [Q12, Q22, 0.0], [0.0, 0.0, lam.G12]])


@block("materials", "laminates", "ply", "rotation", kind="builder",
       source="Jones 2.80", tags=("CLT", "Qbar"))
def lamina_Qbar(Q, theta_deg):
    """Transformed reduced stiffness Qbar for a ply at theta (deg)."""
    t = math.radians(theta_deg)
    m, n = math.cos(t), math.sin(t)
    Q11, Q12, Q22, Q66 = Q[0, 0], Q[0, 1], Q[1, 1], Q[2, 2]
    m2, n2 = m * m, n * n
    Qb = np.empty((3, 3))
    Qb[0, 0] = Q11 * m2 * m2 + 2 * (Q12 + 2 * Q66) * m2 * n2 + Q22 * n2 * n2
    Qb[0, 1] = (Q11 + Q22 - 4 * Q66) * m2 * n2 + Q12 * (m2 * m2 + n2 * n2)
    Qb[1, 1] = Q11 * n2 * n2 + 2 * (Q12 + 2 * Q66) * m2 * n2 + Q22 * m2 * m2
    Qb[0, 2] = (Q11 - Q12 - 2 * Q66) * m2 * m * n + (Q12 - Q22 + 2 * Q66) * m * n2 * n
    Qb[1, 2] = (Q11 - Q12 - 2 * Q66) * m * n2 * n + (Q12 - Q22 + 2 * Q66) * m2 * m * n
    Qb[2, 2] = (Q11 + Q22 - 2 * Q12 - 2 * Q66) * m2 * n2 + Q66 * (m2 * m2 + n2 * n2)
    Qb[1, 0], Qb[2, 0], Qb[2, 1] = Qb[0, 1], Qb[0, 2], Qb[1, 2]
    return Qb


def _stress_to_material(theta_deg, s):
    t = math.radians(theta_deg)
    m, n = math.cos(t), math.sin(t)
    sx, sy, txy = s
    return np.array([m * m * sx + n * n * sy + 2 * m * n * txy,
                     n * n * sx + m * m * sy - 2 * m * n * txy,
                     -m * n * sx + m * n * sy + (m * m - n * n) * txy])


def _alpha_global(lam, theta_deg):
    t = math.radians(theta_deg)
    m, n = math.cos(t), math.sin(t)
    return np.array([lam.alpha1 * m * m + lam.alpha2 * n * n,
                     lam.alpha1 * n * n + lam.alpha2 * m * m,
                     2.0 * (lam.alpha1 - lam.alpha2) * m * n])


# ---------------------------------------------------------------- layup handling


@block("materials", "laminates", "stack", "notation", kind="builder",
       source="standard laminate code", tags=("CLT", "layup", "parser"))
def parse_layup(code):
    """Expand laminate code to a ply-angle list. Supports '/', '+-' or '±', repeat counts and s/2s:
    '[0/+-45/90]s' -> [0,45,-45,90,90,-45,45,0];  '[0/90]2s';  '[45/-45]3'."""
    code = code.replace(" ", "").replace("±", "+-")
    mobj = re.fullmatch(r"\[(.+)\](\d*)(s?)", code)
    if not mobj:
        raise ValueError(f"cannot parse layup {code!r}")
    body, rep, sym = mobj.groups()
    plies = []
    for tok in body.split("/"):
        tm = re.fullmatch(r"(\+-|-\+)?([+-]?\d+(?:\.\d+)?)(?:_(\d+))?", tok)
        if not tm:
            raise ValueError(f"bad ply token {tok!r}")
        pm, ang, cnt = tm.groups()
        a = float(ang)
        group = [a, -a] if pm == "+-" else ([-a, a] if pm == "-+" else [a])
        plies += group * (int(cnt) if cnt else 1)
    plies = plies * (int(rep) if rep else 1)
    if sym:
        plies = plies + plies[::-1]
    return plies


def _stack(lamina, angles):
    if isinstance(lamina, Lamina):
        lamina = [lamina] * len(angles)
    return list(zip(lamina, angles))


@block("materials", "laminates", "stack", "stiffness", kind="builder",
       source="Jones 4.22-4.24", tags=("CLT", "ABD"))
def laminate_ABD(lamina, angles):
    """A, B, D matrices and ply z-coordinates. lamina = one Lamina or a list (one per ply)."""
    stack = _stack(lamina, angles)
    h = sum(l.t_ply for l, _ in stack)
    z = np.empty(len(stack) + 1)
    z[0] = -h / 2
    for k, (l, _) in enumerate(stack):
        z[k + 1] = z[k] + l.t_ply
    A = np.zeros((3, 3)); B = np.zeros((3, 3)); D = np.zeros((3, 3))
    for k, (l, th) in enumerate(stack):
        Qb = lamina_Qbar(lamina_Q(l), th)
        A += Qb * (z[k + 1] - z[k])
        B += Qb * (z[k + 1] ** 2 - z[k] ** 2) / 2
        D += Qb * (z[k + 1] ** 3 - z[k] ** 3) / 3
    return dict(A=A, B=B, D=D, z=z, h=h, stack=stack,
                ABD=np.block([[A, B], [B, D]]))


@block("materials", "laminates", "stack", "thermal", kind="builder",
       source="Jones 4.4 (hygrothermal)", tags=("CLT", "thermal", "warp"))
def laminate_thermal_resultants(lam_abd, dT):
    """Thermal force/moment resultants NT, MT (per width) for a uniform temperature change dT."""
    z = lam_abd["z"]
    NT = np.zeros(3); MT = np.zeros(3)
    for k, (l, th) in enumerate(lam_abd["stack"]):
        Qb = lamina_Qbar(lamina_Q(l), th)
        a = _alpha_global(l, th) * dT
        NT += Qb @ a * (z[k + 1] - z[k])
        MT += Qb @ a * (z[k + 1] ** 2 - z[k] ** 2) / 2
    return NT, MT


@block("materials", "laminates", "stack", "response", kind="analytic",
       source="Jones 4.4", tags=("CLT", "strain", "curvature"))
def laminate_response(lam_abd, N=(0, 0, 0), M=(0, 0, 0), dT=0.0):
    """Midplane strains eps0 and curvatures kappa from applied N, M (and optional dT)."""
    NM = np.concatenate([np.asarray(N, float), np.asarray(M, float)])
    if dT:
        NT, MT = laminate_thermal_resultants(lam_abd, dT)
        NM = NM + np.concatenate([NT, MT])
    sol = np.linalg.solve(lam_abd["ABD"], NM)
    return sol[:3], sol[3:]


@block("materials", "laminates", "ply", "stress", kind="analytic",
       source="Jones 4.4", tags=("CLT", "ply stress"))
def ply_stresses(lam_abd, eps0, kappa, dT=0.0):
    """Material-axis stresses [s1, s2, t12] at top and bottom of every ply. Returns list of dicts."""
    z = lam_abd["z"]; out = []
    for k, (l, th) in enumerate(lam_abd["stack"]):
        Qb = lamina_Qbar(lamina_Q(l), th)
        a = _alpha_global(l, th) * dT
        res = {}
        for pos, zz in (("top", z[k]), ("bot", z[k + 1])):
            eps = eps0 + zz * kappa - a
            res[pos] = _stress_to_material(th, Qb @ eps)
        out.append(dict(ply=k, angle=th, lamina=l, **res))
    return out


# ---------------------------------------------------------------- failure criteria


@block("materials", "laminates", "ply", "failure", source="max stress criterion", tags=("CLT", "failure"))
def strength_ratio_max_stress(s, lam):
    """Strength ratio R (load multiplier to failure) by maximum stress. s = [s1, s2, t12]."""
    s1, s2, t12 = s
    r = [lam.Xt / s1 if s1 > 0 else (lam.Xc / -s1 if s1 < 0 else math.inf),
         lam.Yt / s2 if s2 > 0 else (lam.Yc / -s2 if s2 < 0 else math.inf),
         lam.S / abs(t12) if t12 else math.inf]
    return min(r)


@block("materials", "laminates", "ply", "failure", source="Tsai-Hill (Azzi-Tsai)", tags=("CLT", "failure"))
def strength_ratio_tsai_hill(s, lam):
    """Strength ratio by Tsai-Hill with sign-selected X and Y."""
    s1, s2, t12 = s
    X = lam.Xt if s1 >= 0 else lam.Xc
    Y = lam.Yt if s2 >= 0 else lam.Yc
    fi = (s1 / X) ** 2 - s1 * s2 / X ** 2 + (s2 / Y) ** 2 + (t12 / lam.S) ** 2
    return math.inf if fi <= 0 else 1.0 / math.sqrt(fi)


@block("materials", "laminates", "ply", "failure", source="Tsai-Wu 1971, F12 = -0.5 sqrt(F11 F22)",
       tags=("CLT", "failure"))
def strength_ratio_tsai_wu(s, lam):
    """Strength ratio by Tsai-Wu: solve a R^2 + b R - 1 = 0 for the positive root."""
    s1, s2, t12 = s
    F1 = 1 / lam.Xt - 1 / lam.Xc; F2 = 1 / lam.Yt - 1 / lam.Yc
    F11 = 1 / (lam.Xt * lam.Xc); F22 = 1 / (lam.Yt * lam.Yc); F66 = 1 / lam.S ** 2
    F12 = -0.5 * math.sqrt(F11 * F22)
    a = F11 * s1 ** 2 + F22 * s2 ** 2 + F66 * t12 ** 2 + 2 * F12 * s1 * s2
    b = F1 * s1 + F2 * s2
    if a <= 0:
        return math.inf if b <= 0 else 1.0 / b
    return (-b + math.sqrt(b * b + 4 * a)) / (2 * a)


_CRIT = {"max_stress": strength_ratio_max_stress, "tsai_hill": strength_ratio_tsai_hill,
         "tsai_wu": strength_ratio_tsai_wu}


@block("materials", "laminates", "stack", "failure", kind="analytic",
       source="first-ply failure (linear)", tags=("CLT", "FPF"))
def first_ply_failure(lam_abd, N=(0, 0, 0), M=(0, 0, 0), criterion="tsai_wu"):
    """Load multiplier R at first-ply failure for proportional N, M. Returns (R, ply index, angle)."""
    e0, k = laminate_response(lam_abd, N, M)
    f = _CRIT[criterion]; best = (math.inf, -1, 0.0)
    for p in ply_stresses(lam_abd, e0, k):
        for pos in ("top", "bot"):
            r = f(p[pos], p["lamina"])
            if r < best[0]:
                best = (r, p["ply"], p["angle"])
    return best


# ---------------------------------------------------------------- equivalents


@block("materials", "laminates", "stack", "effective", kind="analytic",
       source="Daniel & Ishai 7.9", tags=("CLT", "effective modulus"))
def laminate_engineering_constants(lam_abd):
    """Effective in-plane Ex, Ey, Gxy, nuxy (membrane) and flexural Efx, Efy of the laminate."""
    h = lam_abd["h"]
    a = np.linalg.inv(lam_abd["A"])
    d = np.linalg.inv(lam_abd["D"])
    return dict(Ex=1 / (h * a[0, 0]), Ey=1 / (h * a[1, 1]), Gxy=1 / (h * a[2, 2]),
                nuxy=-a[0, 1] / a[0, 0], Efx=12 / (h ** 3 * d[0, 0]), Efy=12 / (h ** 3 * d[1, 1]), h=h)


@block("materials", "laminates", "stack", "effective", kind="builder",
       source="CLT equivalent", tags=("CLT", "plate formulas"))
def laminate_as_isotropic(lam_abd, use="flexural", name="laminate"):
    """Equivalent Isotropic material so Roark plate/beam blocks can be used on a laminate.
    use='flexural' for bending problems, 'membrane' for in-plane. Valid for near-isotropic stacks.
    Strength is FPF-based uniaxial stress Nx/h (tsai_wu)."""
    ec = laminate_engineering_constants(lam_abd)
    E = 0.5 * (ec["Efx"] + ec["Efy"]) if use == "flexural" else 0.5 * (ec["Ex"] + ec["Ey"])
    rho = sum(l.rho * l.t_ply for l, _ in lam_abd["stack"]) / lam_abd["h"]
    R, _, _ = first_ply_failure(lam_abd, N=(1.0, 0, 0))
    s_fpf = R / lam_abd["h"]
    return Isotropic(name, E, ec["nuxy"], rho, s_fpf, s_fpf, family="laminate",
                     note=f"CLT {use} equivalent, strength = first-ply failure")


@block("materials", "laminates", "ply", "failure_any", source="maximum stress; Tsai-Hill (Azzi-Tsai); "
       "Tsai-Wu 1971 with F12 = -0.5 sqrt(F11 F22)", tags=("CLT", "failure", "composite", "ply"))
def strength_ratio(criterion, s1, s2, t12, lamina=None, Xt=None, Xc=None, Yt=None, Yc=None, S=None):
    """Strength ratio R of one ply: the multiplier on its stress state (s1, s2, t12) that reaches failure.

    criterion
      max_stress  min of Xt/s1 (or Xc/-s1), Yt/s2 (or Yc/-s2), S/|t12|
      tsai_hill   1/sqrt(failure index), X and Y picked by the sign of s1 and s2
      tsai_wu     positive root of a R^2 + b R - 1 = 0
    s1, s2, t12   material-axis stresses (Pa), e.g. from ply_stresses
    lamina        catalog lamina name (T300_5208, IM7_8552, ...); Xt, Xc, Yt, Yc, S (positive strengths, Pa)
                  override it or replace it
    Returns R (R < 1: the ply fails under this stress state; inf: never)."""
    from types import SimpleNamespace
    from .lookup import mat_props
    f = _CRIT.get(str(criterion).strip().lower())
    if f is None:
        raise ValueError(f"strength_ratio: unknown criterion {criterion!r}; use one of {', '.join(_CRIT)}")
    lam = mat_props(lamina, Xt=Xt, Xc=Xc, Yt=Yt, Yc=Yc, S=S)
    miss = [k for k, v in vars(lam).items() if math.isnan(v)]
    if miss:
        raise ValueError(f"strength_ratio needs a lamina or the strengths {', '.join(miss)}")
    return f((s1, s2, t12), SimpleNamespace(**vars(lam)))
