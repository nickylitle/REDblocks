"""Horizontal wedge strike rebuilt from simblocks (proof that the blocks assemble into a real sim).

mode="legacy"  reproduces Horizontal Wedge Impact.py exactly (same parameters, same update order, same
               lag-style permanent-set updates), so old results can be regenerated from blocks.
The builders compute every stiffness / yield from material + geometry; the njit loop only calls kernels.
An energy ledger is added (the original had none) so verify_all can audit the model.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
from numba import njit

from simblocks.materials import get
from simblocks.structures.sections.properties import rectangle
from simblocks.structures.beams.static import cantilever_end_load, effective_width_45
from simblocks.structures.plates.circular import circ_clamped_center_point
from simblocks.structures.plates.yield_line import plate_mp, collapse_point_free_edge
from simblocks.constitutive.plasticity import lag_return, tabular_return, hinge_to_spring_table
from simblocks.materials import load_curve
from simblocks.materials.curves import Curve
from simblocks.structures.sections.moment_curvature import moment_curvature_rect, hinge_table
from simblocks.structures.joints import tab_shear_out_force
from simblocks.integrators.explicit import critical_dt_spring, critical_dt_rotational

DEFAULTS = dict(
    material="4130_45HRC", rate_factor=1.2,       # sigma_y x 1.2 (strain-rate hardening, as original)
    mat_overrides=None,                           # e.g. dict(E=2.1e11, nu=0.3) to match the legacy script
    t=0.002, wedge_l=0.03, wedge_angle0_deg=60.0, wedge_moi_coef=0.015,
    bot_mass=1.3, bot_speed=0.0, bot_offset=0.0,
    weapon_radius=0.09, weapon_w=1257.0, weapon_moi=7.56e-4, weapon_angle0_deg=-30.0,
    dent_plastic_ratio=0.3, bend_plastic_ratio=0.02,
    t_end=2000e-6, dent_on=1.0, bend_on=1.0,
    physics="legacy",          # "legacy" = original lag updates | "curve" = stress-strain-curve hinge + depth-hardening gouge
    hinge_len_over_t=1.0,      # curve mode: plastic hinge length / thickness (assumption, swept in Monte Carlo)
    edge_clamp_factor=1.0,     # curve mode: root collapse capacity multiplier. 1.0 = flat cantilever (this sim as
                               # written); ~2.2 = 3-edge-clamped CAD face (flapjack-wedge-final-model.md)
)


def build(p):
    """All pre-loop physics from blocks. Returns the scalar parameter set the kernel loop needs."""
    m = get(p["material"])
    if p.get("mat_overrides"):
        m = m.with_(**p["mat_overrides"])
    m = m.with_(sy=m.sy * p["rate_factor"], name=m.name + " (rate-scaled)")
    t, L = p["t"], p["wedge_l"]
    mp = plate_mp(m.sy, t)                                       # per unit width
    dent = circ_clamped_center_point(m, L, t)                    # Roark 11.2: k = 16 pi D / L^2
    dent_k = dent["k"]
    dent_yield_F = collapse_point_free_edge(mp)                  # yield-line half fan pi m_p
    root_w = effective_width_45(0.0, L)                          # 45 deg dispersion: 2 L
    sec = rectangle(root_w, t)
    cant = cantilever_end_load(m, sec, L)                        # Roark 8.1 case 1a
    bend_k = cant["k"] * L * L                                   # tip stiffness -> rotational (N.m/rad)
    bend_yield_F = cant["F_c"]                                   # Mp / L at the dispersion width
    extra = {}
    if p["physics"] == "curve":
        c0 = load_curve(p["material"])
        c = Curve(c0.name, c0.eng_strain, c0.eng_stress * p["rate_factor"], m.E, c0.source + " x rate_factor")
        mk = moment_curvature_rect(c, root_w, t, kappa_max=0.6 / t)
        k0, th_tab, M_tab = hinge_to_spring_table(hinge_table(mk, p["hinge_len_over_t"] * t))
        acc_d = np.linspace(0.0, 0.05, 200)
        F_d = np.array([tab_shear_out_force(mp, m.tau_y, t, a) for a in acc_d])
        M_tab = M_tab * p["edge_clamp_factor"]
        extra = dict(bend_tab_x=th_tab, bend_tab_M=M_tab, dent_tab_x=acc_d, dent_tab_F=F_d,
                     Mp_root=mp * root_w * p["edge_clamp_factor"], curve_source=c.source)
    return dict(**extra,
        dent_k=dent_k, dent_kp=p["dent_plastic_ratio"] * dent_k, dent_y0=dent_yield_F / dent_k,
        bend_k=bend_k, bend_kp=p["bend_plastic_ratio"] * bend_k, bend_y0=bend_yield_F * L / bend_k,
        wedge_moi=p["wedge_moi_coef"] * L * L, L=L,
        a0=math.radians(p["wedge_angle0_deg"]), wa0=math.radians(p["weapon_angle0_deg"]),
    )


@njit(cache=True)
def _loop(n, dt, R, w0, I_w, m_bot, v_bot0, x_bot0, L, a0, wa0, I_wedge,
          dk, dkp, dy0, bk, bkp, by0, dent_on, bend_on):
    wa = wa0; a = a0; x = x_bot0; w = w0; wedge_w = 0.0; v = v_bot0
    dset = 0.0; bset = 0.0; dmax = 0.0; bmax = 0.0; amax = a0; fmax = -1e300
    pen_x0 = R * math.cos(wa0) + L * math.cos(a0)
    E0 = 0.5 * I_w * w0 * w0 + 0.5 * m_bot * v_bot0 * v_bot0
    W_pl = 0.0
    for i in range(1, n):
        wa += w * dt; a += wedge_w * dt; x += v * dt
        tip_x = R * math.cos(wa); tip_y = R * math.sin(wa)
        lip_x = x - L * math.cos(a)
        n_x = -math.sin(a)
        # bend (legacy lag update, symmetric)
        th = a - a0
        bset_new = lag_return(th, bset, by0, bkp, bkp / bk, 1) if bend_on > 0.5 else 0.0
        bend_T = -bk * (th - bset_new) if bend_on > 0.5 else -bk * th
        W_pl += -bend_T * (bset_new - bset)
        bset = bset_new
        # dent (legacy lag update, one-sided)
        pen = -(tip_x - lip_x - pen_x0) * n_x
        dset_new = lag_return(pen, dset, dy0, dkp / (dk - dkp), dkp / dk, 0) if dent_on > 0.5 else 0.0
        dF = dk * (pen - dset_new)
        if dF < 0.0:
            dF = 0.0
        W_pl += dF * (dset_new - dset)
        dset = dset_new
        bend_Fx = bend_T / L * n_x
        fmax = max(fmax, bend_Fx)
        w += (-tip_y * dF * n_x) / I_w * dt
        wedge_w += (dF * L + bend_T) / I_wedge * dt
        v += bend_Fx / m_bot * dt
        dmax = max(dmax, dset); bmax = max(bmax, bset); amax = max(amax, a)
    el = dk * (pen - dset) if pen > dset else 0.0
    E1 = 0.5 * I_w * w * w + 0.5 * I_wedge * wedge_w ** 2 + 0.5 * m_bot * v * v + \
        0.5 * el * el / dk + 0.5 * bk * (a - a0 - bset) ** 2
    return bmax, dmax, amax, fmax, E0, E1, W_pl, w


@njit(cache=True)
def _loop_curve(n, dt, R, w0, I_w, m_bot, v_bot0, x_bot0, L, a0, wa0, I_wedge,
                dk, bk, d_x, d_F, b_x, b_M, dent_on, bend_on, log_every):
    """Same geometry and coupling as _loop; plasticity from tabular_return (curve hinge, gouge hardening)."""
    wa = wa0; a = a0; x = x_bot0; w = w0; wedge_w = 0.0; v = v_bot0
    dset = 0.0; dacc = 0.0; bset = 0.0; bacc = 0.0
    dmax = 0.0; bmax = 0.0; amax = a0; Tmax = 0.0; Fdmax = 0.0
    pen_x0 = R * math.cos(wa0) + L * math.cos(a0)
    E0 = 0.5 * I_w * w0 * w0 + 0.5 * m_bot * v_bot0 * v_bot0
    W_pl = 0.0; pen = 0.0
    nl = n // log_every + 1
    H = np.zeros((nl, 10)); j = 0
    for i in range(1, n):
        wa += w * dt; a += wedge_w * dt; x += v * dt
        tip_x = R * math.cos(wa); tip_y = R * math.sin(wa)
        lip_x = x - L * math.cos(a)
        n_x = -math.sin(a)
        th = a - a0
        if bend_on > 0.5:
            Mb, bnew, bacc = tabular_return(th, bset, bacc, bk, b_x, b_M, 0)
        else:
            Mb = bk * th; bnew = 0.0
        bend_T = -Mb
        W_pl += Mb * (bnew - bset); bset = bnew
        pen = -(tip_x - lip_x - pen_x0) * n_x
        if dent_on > 0.5:
            dF, dnew, dacc = tabular_return(pen, dset, dacc, dk, d_x, d_F, 1)
        else:
            dF = dk * pen if pen > 0.0 else 0.0; dnew = 0.0
        W_pl += dF * (dnew - dset); dset = dnew
        bend_Fx = bend_T / L * n_x
        w += (-tip_y * dF * n_x) / I_w * dt
        wedge_w += (dF * L + bend_T) / I_wedge * dt
        v += bend_Fx / m_bot * dt
        dmax = max(dmax, dset); bmax = max(bmax, bset); amax = max(amax, a); Tmax = max(Tmax, abs(bend_T))
        Fdmax = max(Fdmax, dF)
        if i % log_every == 0 and j < nl:
            H[j, 0] = i * dt; H[j, 1] = w; H[j, 2] = a - a0; H[j, 3] = pen; H[j, 4] = dF; H[j, 5] = -bend_T
            H[j, 6] = dset; H[j, 7] = bset; H[j, 8] = 0.5 * I_w * w * w; H[j, 9] = W_pl
            j += 1
    el = dk * (pen - dset) if pen > dset else 0.0
    E1 = 0.5 * I_w * w * w + 0.5 * I_wedge * wedge_w ** 2 + 0.5 * m_bot * v * v + \
        0.5 * el * el / dk + 0.5 * bk * (a - a0 - bset) ** 2
    return bmax, dmax, amax, Tmax, E0, E1, W_pl, w, bset, dset, Fdmax, H[:j]


HIST_COLS = ["t", "weapon_w", "face_rot", "pen", "dent_F", "root_M", "dent_set", "bend_set", "weapon_KE", "W_plastic"]


def wedge_strike(params=None, dt=0.1e-6, history=False):
    p = dict(DEFAULTS); p.update(params or {})
    b = build(p)
    n = int(round(p["t_end"] / dt))
    if p["physics"] == "curve":
        bmax, dmax, amax, Tmax, E0, E1, W_pl, w_end, bfin, dfin, Fdmax, Hh = _loop_curve(
            n, dt, p["weapon_radius"], p["weapon_w"], p["weapon_moi"], p["bot_mass"], p["bot_speed"], p["bot_offset"],
            b["L"], b["a0"], b["wa0"], b["wedge_moi"], b["dent_k"], b["bend_k"], b["dent_tab_x"], b["dent_tab_F"],
            b["bend_tab_x"], b["bend_tab_M"], p["dent_on"], p["bend_on"], max(n // 2000, 1) if history else n + 1)
        L, a0 = b["L"], b["a0"]
        out = {"bending deformation": math.degrees(bmax), "denting deformation": dmax * 1000,
                "wedge drop": L * (math.sin(amax) - math.sin(a0)) * 1000,
                "peak root moment / Mp": Tmax / b["Mp_root"],
                # wedge drop (Nicky's definition): how far the TIP falls after impact from the face rotating about its
                # root. Theoretical max L - L sin(a0) when the face folds to vertical.
                "tip_drop_perm_mm": L * (math.sin(a0 + bfin) - math.sin(a0)) * 1000,
                "tip_drop_with_dent_mm": (L * (math.sin(a0 + bfin) - math.sin(a0)) + dfin * math.cos(a0 + bfin)) * 1000,
                "max_possible_bend_drop_mm": L * (1 - math.sin(a0)) * 1000,
                "peak_dent_force_N": Fdmax,
                "weapon_energy_lost": E0 - 0.5 * p["weapon_moi"] * w_end ** 2,
                "_ledger_rel": (E0 - E1 - W_pl) / E0}
        if history:
            return out, {k: Hh[:, i] for i, k in enumerate(HIST_COLS)}, b
        return out
    bmax, dmax, amax, fmax, E0, E1, W_pl, w_end = _loop(
        n, dt, p["weapon_radius"], p["weapon_w"], p["weapon_moi"], p["bot_mass"], p["bot_speed"], p["bot_offset"],
        b["L"], b["a0"], b["wa0"], b["wedge_moi"], b["dent_k"], b["dent_kp"], b["dent_y0"],
        b["bend_k"], b["bend_kp"], b["bend_y0"], p["dent_on"], p["bend_on"])
    L, a0 = b["L"], b["a0"]
    return {"bending deformation": math.degrees(bmax), "denting deformation": dmax * 1000,
            "wedge drop": L * (math.sin(amax) - math.sin(a0)) * 1000, "force": fmax,
            "weapon_energy_lost": E0 - 0.5 * p["weapon_moi"] * w_end ** 2,
            "_ledger_rel": (E0 - E1 - W_pl) / E0}


def critical_dt(params):
    p = dict(DEFAULTS); p.update(params); b = build(p)
    I_tip = p["weapon_moi"] / p["weapon_radius"] ** 2
    m_eff = 1.0 / (1.0 / I_tip + b["L"] ** 2 / b["wedge_moi"])
    return min(critical_dt_spring(b["dent_k"], m_eff), critical_dt_rotational(b["bend_k"], b["wedge_moi"]))


if __name__ == "__main__":
    for t in (0.0015, 0.002, 0.003):
        print(t, wedge_strike(dict(t=t)))
