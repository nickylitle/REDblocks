"""Vertical spinner vs horizontal spinner: does BLADE BENDING produce a double hit?

3-D explicit sim. Each robot = 6-DOF rigid body carrying a spinning blade (gyrostat coupling). Each blade is a
rigid hub plus two FLEXIBLE ARMS (one per tooth). An arm is a cantilever from the hub clamp to the tip: its
first bending mode is a lumped tooth mass (33/140 of the arm mass) tied to the tip anchor on the rotor by
  - radial: axial bar stiffness EA/L (elastic)
  - lag (in-plane) and flap (out-of-plane): elastic-plastic cantilever springs from beam_plastic_spring
    (moment-curvature of the blade section from its stress-strain curve; permanent set tracked by tabular_return).
The arm's contact box follows the tooth (linear deflection shape: a point at fraction xi of the arm moves by
xi * tooth deflection); forces on the arm are split to rotor and tooth by the same weights (virtual-power
consistent, so the energy ledger closes).

Contact: sampled edge points of one robot vs the boxes of the other (point_box_penetration), Hunt-Crossley
normal law with the Flores restitution factor, regularized Coulomb friction; floor contact for every point;
compliant supports (wheels, skid, forks) with anisotropic friction.

Hit bookkeeping: raw blade-to-blade contact episodes (separations < 5 us are one episode) are logged; Python
merges episodes closer than t_merge into HITS. Double hit = 2 or more hits in one engagement.
"""
import math
import os
import sys

import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
from simblocks.rigid_body.spatial import (quat_to_matrix, quat_step, gyrostat_accel, gyrostat_energy,
                                          point_box_penetration, coulomb_regularized_3d, I_box_tensor,
                                          parallel_axis_tensor, obb_gap)
from simblocks.structures.beams.plastic_spring import beam_plastic_spring
from simblocks.constitutive.plasticity import tabular_return
from simblocks.materials import get
from simblocks.materials.curves import resolve_material

G = 9.81
RHO_STEEL = 7850.0
MODAL_TIP_FRACTION = 33.0 / 140.0      # cantilever first mode, mass referred to the tip (Rayleigh)

DEFAULTS = dict(
    # horizontal robot (H): Flapjack undercutter. 190 x 30 x 6.35 mm bar, 280 g, 600 J at 113 m/s tip
    H_m=1.361, H_R=0.095, H_W=0.030, H_t=0.00635, H_mr=0.28, H_Is=8.5e-4, H_rpm=11360.0,
    H_mat="AR500",            # stand-in for AR600 / SR500 (not in the catalog)
    H_rroot=0.0125,           # ASSUMPTION: arm clamp radius (25 mm retention washer OD)
    H_zblade=0.015,           # ASSUMPTION: blade mid-plane height above the floor
    H_hub_x=0.060, H_zcom=0.025,
    # vertical robot (V): generic beetleweight bar vertical (all ASSUMPTIONS)
    V_m=1.361, V_R=0.060, V_W=0.025, V_t=0.0127, V_rpm=12000.0, V_clear=0.005, V_hub_x=0.070, V_zcom=0.030,
    V_mat="S7", V_rroot=0.015,
    # engagement
    v_close=3.0, y_off=0.0, yaw_off_deg=0.0, phase_H_deg=0.0, phase_V_deg=0.0, gap0=0.003, t_end=0.040,
    # blade flexibility switches (knockouts)
    flexH=1.0, flexV=1.0, plastic=1.0, zeta_arm=0.02,
    H_sy_scale=1.0, V_sy_scale=1.0,   # scale each blade's bending strength (yield + hardening table), not stiffness
    # contact
    k_line=1.0e11, ds_blade=0.0015, ds_body=0.005,
    e_bb=0.4, mu_bb=0.3, e_floor=0.3, mu_floor=0.4, k_line_floor=1.0e10, v_eps=0.5,
    k_sup=3.0e5, zeta_sup=0.3,
    gyro_on=1.0,
    # run-time controls: bigger steps while nothing can touch (gap-driven, capped at 6 dt = 1.5 us, inside the stiffest
    # spring's stability limit of 8.5 us), and stop
    # t_after past the first blade hit (0 = run to t_end)
    adaptive=1.0, dt_mult_max=6, t_after=0.008,
    gravity=9.81, z_lift=0.0,          # z_lift > 0 lifts both robots off the floor (free-body studies)     # big step = up to 6 x dt (1.5 us at the default dt)
)

NT = 64      # common length for the hinge tables
NVIN = 6     # contact-entry-speed memory: up to 5 boxes of the other robot + floor


def _edge_points(lo, hi, ds):
    """Points on the 12 edges of the box [lo, hi] (per axis), spacing <= ds."""
    pts = []
    for ax in range(3):
        o = [i for i in range(3) if i != ax]
        n = max(int(math.ceil((hi[ax] - lo[ax]) / ds)), 1)
        s = np.linspace(lo[ax], hi[ax], n + 1)
        for v1 in (lo[o[0]], hi[o[0]]):
            for v2 in (lo[o[1]], hi[o[1]]):
                P = np.zeros((n + 1, 3)); P[:, ax] = s; P[:, o[0]] = v1; P[:, o[1]] = v2
                pts.append(P)
    return np.unique(np.round(np.vstack(pts), 12), axis=0)


def _rx90():
    return np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])


def _resample(acc, F):
    a = np.linspace(0.0, acc[-1], NT)
    return a, np.interp(a, acc, F)


def _arm_springs(mat, W, t, L, flex, plastic):
    """(k_lag, lag tables, k_flap, flap tables, k_axial) for one arm."""
    m = get(resolve_material(mat))
    k_ax = m.E * W * t / L
    if flex < 0.5:
        big = np.full(NT, 1e12)
        return k_ax, np.linspace(0, 1, NT), big, k_ax, np.linspace(0, 1, NT), big, k_ax, dict(rigid=True)
    flap = beam_plastic_spring(mat, {"rect": (W, t)}, L, support="cantilever_end")
    lag = beam_plastic_spring(mat, {"rect": (t, W)}, L, support="cantilever_end")
    la, lF = _resample(lag["acc_tab"], lag["F_tab"]); fa, fF = _resample(flap["acc_tab"], flap["F_tab"])
    if plastic < 0.5:
        lF = lF * 0 + 1e12; fF = fF * 0 + 1e12
    return lag["k"], la, lF, flap["k"], fa, fF, k_ax, dict(flap=flap, lag=lag)


def build(p):
    p["V_zaxle"] = p["V_R"] + p["V_clear"]          # axle height = blade radius + tip clearance
    bots = []
    # robot-frame geometry (origin at the ROBOT COM, x forward, z up)
    H = dict(m=p["H_m"], mr=p["H_mr"], Is=p["H_Is"], R=p["H_R"], W=p["H_W"], t=p["H_t"], rroot=p["H_rroot"],
             a=np.array([0.0, 0, 1.0]), Rab=np.eye(3), mat=p["H_mat"], flex=p["flexH"],
             d=np.array([p["H_hub_x"], 0.0, p["H_zblade"] - p["H_zcom"]]), chassis_dims=(0.21, 0.20, 0.045),
             body_boxes=[(np.array([-0.0775, 0.0, 0.0275 - p["H_zcom"]]), np.array([0.0425, 0.10, 0.0225]), "H chassis"),
                         (np.array([0.0275, 0.0, 0.028 - p["H_zcom"]]), np.array([0.0625, 0.06, 0.006]), "H nose plate")],
             sup=[(-0.04, 0.09, 0.05, 1.6), (-0.04, -0.09, 0.05, 1.6), (0.0215, 0.0, 0.20, 0.20)],
             zcom=p["H_zcom"], rpm=p["H_rpm"], name="H")
    VR, VW, Vt = p["V_R"], p["V_W"], p["V_t"]
    Vmr = 2 * VR * VW * Vt * RHO_STEEL
    V = dict(m=p["V_m"], mr=Vmr, Is=Vmr * ((2 * VR) ** 2 + VW ** 2) / 12, R=VR, W=VW, t=Vt, rroot=p["V_rroot"],
             a=np.array([0.0, -1.0, 0.0]), Rab=_rx90(), mat=p["V_mat"], flex=p["flexV"],
             d=np.array([p["V_hub_x"], 0.0, p["V_zaxle"] - p["V_zcom"]]), chassis_dims=(0.17, 0.16, 0.065),
             body_boxes=[(np.array([-0.0475, 0.0, 0.0415 - p["V_zcom"]]), np.array([0.0525, 0.08, 0.0335]), "V chassis")],
             sup=[(-0.07, 0.075, 0.05, 1.6), (-0.07, -0.075, 0.05, 1.6), (0.10, 0.06, 0.3, 0.3), (0.10, -0.06, 0.3, 0.3)],
             zcom=p["V_zcom"], rpm=p["V_rpm"], name="V")
    bots = [H, V]
    for b in bots:
        L = b["R"] - b["rroot"]
        m_arm = b["mr"] * L / (2 * b["R"])
        b["L"] = L; b["mt"] = MODAL_TIP_FRACTION * m_arm
        b["Is_r"] = b["Is"] - 2 * b["mt"] * b["R"] ** 2              # rigid part of the rotor
        assert b["Is_r"] > 0
        mc = b["m"] - b["mr"]
        dc = -b["mr"] * b["d"] / mc                                   # chassis centroid: robot COM at origin
        b["mb"] = b["m"] - 2 * b["mt"]                                 # body = chassis + rigid rotor
        cb = (mc * dc + (b["mr"] - 2 * b["mt"]) * b["d"]) / b["mb"]    # body COM (teeth sit symmetric about the hub)
        b["cb"] = cb
        Jc = parallel_axis_tensor(I_box_tensor(mc, *b["chassis_dims"]), mc, dc - cb)
        Jr = parallel_axis_tensor(b["Rab"] @ np.diag([b["Is_r"] / 2, b["Is_r"] / 2, b["Is_r"]]) @ b["Rab"].T,
                                  b["mr"] - 2 * b["mt"], b["d"] - cb)
        b["J"] = Jc + Jr
        b["d_b"] = b["d"] - cb                                         # hub in BODY frame (origin = body COM)
        kl, la, lF, kf, fa, fF, kax, info = _arm_springs(b["mat"], b["W"], b["t"], L, b["flex"], p["plastic"])
        sc = p["H_sy_scale"] if b["name"] == "H" else p["V_sy_scale"]
        if b["flex"] > 0.5 and p["plastic"] > 0.5:
            lF = lF * sc; fF = fF * sc
        info["sy_scale"] = sc
        b.update(k_lag=kl, lag_acc=la, lag_F=lF, k_flap=kf, flap_acc=fa, flap_F=fF, k_ax=kax, spring_info=info)

    # ----- boxes and points
    B_bot, B_kind, B_arm, B_c, B_h, B_slot, B_label, B_rng = [], [], [], [], [], [], [], []
    P_loc, P_xi, P_k, P_kf = [], [], [], []
    ARM = dict(bot=[], sign=[], R=[], root=[], L=[], h=[], m=[], kax=[], kl=[], kf=[], la=[], lF=[], fa=[], fF=[], c=[])
    for bi, b in enumerate(bots):
        slot = 0
        def add_box(kind, arm, c, h, lab, lo, hi, ds):
            nonlocal slot
            start = len(P_loc)
            pts = _edge_points(lo, hi, ds)
            for q in pts:
                P_loc.append(q)
                P_xi.append((abs(q[0]) - b["rroot"]) / b["L"] if kind == 2 else 0.0)
                P_k.append(p["k_line"] * ds); P_kf.append(p["k_line_floor"] * ds)
            B_bot.append(bi); B_kind.append(kind); B_arm.append(arm); B_c.append(c); B_h.append(h)
            B_slot.append(slot); B_label.append(lab); B_rng.append((start, len(P_loc))); slot += 1
        hw, ht = b["W"] / 2, b["t"] / 2
        rr = b["rroot"]
        add_box(1, -1, np.zeros(3), np.array([rr, hw, ht]), f"{b['name']} hub", np.array([-rr, -hw, -ht]),
                np.array([rr, hw, ht]), p["ds_blade"])
        for sgn in (1, -1):
            ai = len(ARM["bot"])
            for k, v in (("bot", bi), ("sign", sgn), ("R", b["R"]), ("root", rr), ("L", b["L"]),
                         ("h", [b["L"] / 2, hw, ht]), ("m", b["mt"]), ("kax", b["k_ax"]), ("kl", b["k_lag"]),
                         ("kf", b["k_flap"]), ("la", b["lag_acc"]), ("lF", b["lag_F"]), ("fa", b["flap_acc"]),
                         ("fF", b["flap_F"]), ("c", 2 * p["zeta_arm"] * math.sqrt(b["k_flap"] * b["mt"]))):
                ARM[k].append(v)
            lo = np.array([rr, -hw, -ht]) if sgn > 0 else np.array([-b["R"], -hw, -ht])
            hi = np.array([b["R"], hw, ht]) if sgn > 0 else np.array([-rr, hw, ht])
            add_box(2, ai, np.zeros(3), np.array([b["L"] / 2, hw, ht]), f"{b['name']} arm {'+' if sgn > 0 else '-'}",
                    lo, hi, p["ds_blade"])
        for (c, h, lab) in b["body_boxes"]:
            cc = c - b["cb"]
            add_box(0, -1, cc, h, lab, cc - h, cc + h, p["ds_body"])
    S_bot, S_loc, S_mux, S_muy, S_c = [], [], [], [], []
    for bi, b in enumerate(bots):
        msh = b["m"] / len(b["sup"])
        for (x, y, mux, muy) in b["sup"]:
            S_bot.append(bi); S_loc.append(np.array([x, y, -b["zcom"]]) - b["cb"]); S_mux.append(mux); S_muy.append(muy)
            S_c.append(2 * p["zeta_sup"] * math.sqrt(p["k_sup"] * msh))
    arr = lambda k, dt=float: np.array(ARM[k], dtype=dt)
    return dict(bots=bots,
                B_bot=np.array(B_bot, np.int64), B_kind=np.array(B_kind, np.int64), B_arm=np.array(B_arm, np.int64),
                B_c=np.array(B_c), B_h=np.array(B_h), B_slot=np.array(B_slot, np.int64),
                B_rng=np.array(B_rng, np.int64), B_label=B_label,
                P_loc=np.array(P_loc), P_xi=np.array(P_xi), P_k=np.array(P_k), P_kf=np.array(P_kf),
                A_bot=arr("bot", np.int64), A_sign=arr("sign"), A_R=arr("R"), A_root=arr("root"), A_L=arr("L"),
                A_h=arr("h"), A_m=arr("m"), A_kax=arr("kax"), A_kl=arr("kl"), A_kf=arr("kf"),
                A_la=arr("la"), A_lF=arr("lF"), A_fa=arr("fa"), A_fF=arr("fF"), A_c=arr("c"),
                S_bot=np.array(S_bot, np.int64), S_loc=np.array(S_loc), S_mux=np.array(S_mux), S_muy=np.array(S_muy),
                S_c=np.array(S_c))


def _yaw_quat(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def initial_state(p, b):
    """Robots face each other on the x axis with their reach envelopes gap0 apart, closing at v_close."""
    p["V_zaxle"] = p["V_R"] + p["V_clear"]
    H, V = b["bots"]
    def v_reach(z_top):                           # V blade's furthest-forward point below height z_top
        dz = p["V_zaxle"] - z_top
        return p["V_hub_x"] + math.hypot(math.sqrt(max(p["V_R"] ** 2 - dz ** 2, 0.0)) + p["V_t"], p["V_W"] / 2)
    sep = max(p["H_hub_x"] + math.hypot(p["H_R"], p["H_W"] / 2) + v_reach(p["H_zblade"] + p["H_t"] / 2),  # blades
              0.09 + v_reach(0.034))                                                                     # H nose plate
    X = np.zeros((2, 3)); Vv = np.zeros((2, 3)); Q = np.zeros((2, 4))
    Q[0] = _yaw_quat(0.0); Q[1] = _yaw_quat(math.pi + math.radians(p["yaw_off_deg"]))
    com = [np.array([0.0, 0.0, p["H_zcom"]]), np.array([sep + p["gap0"], p["y_off"], p["V_zcom"]])]
    for i, bot in enumerate((H, V)):
        X[i] = com[i] + quat_to_matrix(Q[i]) @ bot["cb"]          # body COM from robot COM
        X[i, 2] -= bot["m"] * G / (len(bot["sup"]) * p["k_sup"])
        X[i, 2] += p["z_lift"]
    Vv[0] = [0.5 * p["v_close"], 0, 0]; Vv[1] = [-0.5 * p["v_close"], 0, 0]
    W = np.zeros((2, 3))
    OM = np.array([H["rpm"], V["rpm"]]) * 2 * math.pi / 60
    PSI = np.radians([p["phase_H_deg"], p["phase_V_deg"]])
    # teeth at their anchors, moving with the rotor
    XT = np.zeros((4, 3)); VT = np.zeros((4, 3))
    for j in range(4):
        bi = int(b["A_bot"][j]); bot = (H, V)[bi]
        RB = quat_to_matrix(Q[bi]); c = PSI[bi]; s = math.sin(c); c = math.cos(c)
        RR = RB @ bot["Rab"] @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        CW = X[bi] + RB @ bot["d_b"]
        XT[j] = CW + RR @ np.array([b["A_sign"][j] * b["A_R"][j], 0, 0])
        VT[j] = Vv[bi] + OM[bi] * np.cross(RB @ bot["a"], XT[j] - CW)
    return X, Vv, Q, W, PSI, OM, XT, VT


# ------------------------------------------------------------------ small vector helpers (no allocation)
@njit(cache=True, inline="always")
def _mv(R, x, y, z):
    return (R[0, 0] * x + R[0, 1] * y + R[0, 2] * z, R[1, 0] * x + R[1, 1] * y + R[1, 2] * z,
            R[2, 0] * x + R[2, 1] * y + R[2, 2] * z)


@njit(cache=True, inline="always")
def _mtv(R, x, y, z):
    return (R[0, 0] * x + R[1, 0] * y + R[2, 0] * z, R[0, 1] * x + R[1, 1] * y + R[2, 1] * z,
            R[0, 2] * x + R[1, 2] * y + R[2, 2] * z)


@njit(cache=True, inline="always")
def _cr(ax, ay, az, bx, by, bz):
    return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx


@njit(cache=True)
def _vrig(b, onrot, px, py, pz, X, Vv, WW, CW, AW, OM):
    """Rigid velocity of robot b's body (onrot=0) or rotor (onrot=1) material at world point p."""
    cx, cy, cz = _cr(WW[b, 0], WW[b, 1], WW[b, 2], px - X[b, 0], py - X[b, 1], pz - X[b, 2])
    vx = Vv[b, 0] + cx; vy = Vv[b, 1] + cy; vz = Vv[b, 2] + cz
    if onrot == 1:
        dx, dy, dz = _cr(AW[b, 0], AW[b, 1], AW[b, 2], px - CW[b, 0], py - CW[b, 1], pz - CW[b, 2])
        vx += OM[b] * dx; vy += OM[b] * dy; vz += OM[b] * dz
    return vx, vy, vz


@njit(cache=True)
def _apply(b, onrot, px, py, pz, fx, fy, fz, X, CW, AW, F, TQ, TAX):
    F[b, 0] += fx; F[b, 1] += fy; F[b, 2] += fz
    tx, ty, tz = _cr(px - X[b, 0], py - X[b, 1], pz - X[b, 2], fx, fy, fz)
    TQ[b, 0] += tx; TQ[b, 1] += ty; TQ[b, 2] += tz
    if onrot == 1:
        tx, ty, tz = _cr(px - CW[b, 0], py - CW[b, 1], pz - CW[b, 2], fx, fy, fz)
        TAX[b] += AW[b, 0] * tx + AW[b, 1] * ty + AW[b, 2] * tz


@njit(cache=True)
def _rz(psi):
    c = math.cos(psi); s = math.sin(psi)
    R = np.zeros((3, 3)); R[0, 0] = c; R[0, 1] = -s; R[1, 0] = s; R[1, 1] = c; R[2, 2] = 1.0
    return R


@njit(cache=True)
def _loop(n, dt, X, Vv, Q, W, PSI, OM, XT, VT, XP, ACC, m, J, Is, A, D, RAB,
          B_bot, B_kind, B_arm, B_c, B_h, B_rng, B_slot, P_loc, P_xi, P_k, P_kf,
          A_bot, A_sign, A_R, A_root, A_L, A_h, A_m, A_kax, A_kl, A_kf, A_la, A_lF, A_fa, A_fF, A_c,
          S_bot, S_loc, S_mux, S_muy, S_c, k_sup,
          e_bb, mu_bb, e_fl, mu_fl, v_eps, gyro_on, every, H, EP, VIN, UMAX, mmax, t_after,
          gval, EXT_T, EXT_F, ext_arm, ext_body):
    nB = B_bot.shape[0]; nS = S_bot.shape[0]; nA = A_bot.shape[0]; FL = VIN.shape[1] - 1
    Wnc = 0.0; Wpl = 0.0; E0 = 0.0; resid = 0.0
    nep = 0; in_ep = False; last_c = -1.0
    max_pen_ratio = 0.0; Fbb_peak = 0.0; jrow = 0; t_wb = -1.0
    RB = np.zeros((2, 3, 3)); RR = np.zeros((2, 3, 3)); CW = np.zeros((2, 3)); AW = np.zeros((2, 3))
    WW = np.zeros((2, 3)); BCW = np.zeros((nB, 3)); BOW = np.zeros((nB, 3, 3)); BRAD = np.zeros(nB)
    AW_anchor = np.zeros((nA, 3)); ROOT = np.zeros((nA, 3)); UW = np.zeros((nA, 3))
    F = np.zeros((2, 3)); TQ = np.zeros((2, 3)); TAX = np.zeros(2); FT = np.zeros((nA, 3))
    for k in range(nB):
        BRAD[k] = math.sqrt(B_h[k, 0] ** 2 + B_h[k, 1] ** 2 + B_h[k, 2] ** 2)
    GAP = np.full((nB, nB), 1e9)
    WASNEAR = np.ones(nB, np.bool_)
    TB = np.zeros(3)
    SPF = np.zeros((nA, 3))
    ARM_MARGIN = 0.004       # sheared arm vs its rotated-box stand-in: offsets up to (u/L)(W/2), a few mm
    tick = 0; it = 0; t_bb0 = -1.0; nsteps = 0; mult = 1; next_rec = 0
    while True:
        t = tick * dt
        it = tick
        # -------------------------------------------- kinematics
        for b in range(2):
            qw = Q[b, 0]; qx = Q[b, 1]; qy = Q[b, 2]; qz = Q[b, 3]       # quat_to_matrix, written in place
            RB[b, 0, 0] = 1 - 2 * (qy * qy + qz * qz); RB[b, 0, 1] = 2 * (qx * qy - qw * qz); RB[b, 0, 2] = 2 * (qx * qz + qw * qy)
            RB[b, 1, 0] = 2 * (qx * qy + qw * qz); RB[b, 1, 1] = 1 - 2 * (qx * qx + qz * qz); RB[b, 1, 2] = 2 * (qy * qz - qw * qx)
            RB[b, 2, 0] = 2 * (qx * qz - qw * qy); RB[b, 2, 1] = 2 * (qy * qz + qw * qx); RB[b, 2, 2] = 1 - 2 * (qx * qx + qy * qy)
            cp = math.cos(PSI[b]); sp = math.sin(PSI[b])
            for r_ in range(3):
                m0 = RB[b, r_, 0] * RAB[b, 0, 0] + RB[b, r_, 1] * RAB[b, 1, 0] + RB[b, r_, 2] * RAB[b, 2, 0]
                m1 = RB[b, r_, 0] * RAB[b, 0, 1] + RB[b, r_, 1] * RAB[b, 1, 1] + RB[b, r_, 2] * RAB[b, 2, 1]
                m2 = RB[b, r_, 0] * RAB[b, 0, 2] + RB[b, r_, 1] * RAB[b, 1, 2] + RB[b, r_, 2] * RAB[b, 2, 2]
                RR[b, r_, 0] = m0 * cp + m1 * sp; RR[b, r_, 1] = -m0 * sp + m1 * cp; RR[b, r_, 2] = m2
            x0, x1, x2 = _mv(RB[b], D[b, 0], D[b, 1], D[b, 2])
            CW[b, 0] = X[b, 0] + x0; CW[b, 1] = X[b, 1] + x1; CW[b, 2] = X[b, 2] + x2
            x0, x1, x2 = _mv(RB[b], A[b, 0], A[b, 1], A[b, 2]); AW[b, 0] = x0; AW[b, 1] = x1; AW[b, 2] = x2
            x0, x1, x2 = _mv(RB[b], W[b, 0], W[b, 1], W[b, 2]); WW[b, 0] = x0; WW[b, 1] = x1; WW[b, 2] = x2
        for j in range(nA):
            b = A_bot[j]; sg = A_sign[j]
            x0, x1, x2 = _mv(RR[b], sg * A_R[j], 0.0, 0.0)
            AW_anchor[j, 0] = CW[b, 0] + x0; AW_anchor[j, 1] = CW[b, 1] + x1; AW_anchor[j, 2] = CW[b, 2] + x2
            x0, x1, x2 = _mv(RR[b], sg * A_root[j], 0.0, 0.0)
            ROOT[j, 0] = CW[b, 0] + x0; ROOT[j, 1] = CW[b, 1] + x1; ROOT[j, 2] = CW[b, 2] + x2
            for c in range(3):
                UW[j, c] = XT[j, c] - AW_anchor[j, c]
        for k in range(nB):
            b = B_bot[k]
            if B_kind[k] == 0:
                x0, x1, x2 = _mv(RB[b], B_c[k, 0], B_c[k, 1], B_c[k, 2])
                BCW[k, 0] = X[b, 0] + x0; BCW[k, 1] = X[b, 1] + x1; BCW[k, 2] = X[b, 2] + x2
                BOW[k] = RB[b]
            elif B_kind[k] == 1:
                BCW[k] = CW[b]; BOW[k] = RR[b]
            else:
                j = B_arm[k]
                ex = XT[j, 0] - ROOT[j, 0]; ey = XT[j, 1] - ROOT[j, 1]; ez = XT[j, 2] - ROOT[j, 2]
                ln = math.sqrt(ex * ex + ey * ey + ez * ez); ex /= ln; ey /= ln; ez /= ln
                zx = RR[b, 0, 2]; zy = RR[b, 1, 2]; zz = RR[b, 2, 2]
                dd = zx * ex + zy * ey + zz * ez
                zx -= dd * ex; zy -= dd * ey; zz -= dd * ez
                zn = math.sqrt(zx * zx + zy * zy + zz * zz); zx /= zn; zy /= zn; zz /= zn
                yx, yy, yz = _cr(zx, zy, zz, ex, ey, ez)
                BOW[k, 0, 0] = ex; BOW[k, 1, 0] = ey; BOW[k, 2, 0] = ez
                BOW[k, 0, 1] = yx; BOW[k, 1, 1] = yy; BOW[k, 2, 1] = yz
                BOW[k, 0, 2] = zx; BOW[k, 1, 2] = zy; BOW[k, 2, 2] = zz
                BCW[k, 0] = ROOT[j, 0] + 0.5 * ln * ex; BCW[k, 1] = ROOT[j, 1] + 0.5 * ln * ey
                BCW[k, 2] = ROOT[j, 2] + 0.5 * ln * ez
        for k in range(nB):
            if B_bot[k] != 0:
                continue
            for l in range(nB):
                if B_bot[l] != 1:
                    continue
                dc = math.sqrt((BCW[k, 0] - BCW[l, 0]) ** 2 + (BCW[k, 1] - BCW[l, 1]) ** 2 + (BCW[k, 2] - BCW[l, 2]) ** 2)
                g = dc - BRAD[k] - BRAD[l]                 # sphere bound: valid lower bound, much cheaper
                if g < 0.02:
                    g = obb_gap(BCW[k], BOW[k], B_h[k], BCW[l], BOW[l], B_h[l])
                g -= (ARM_MARGIN if B_kind[k] == 2 else 0.0) + (ARM_MARGIN if B_kind[l] == 2 else 0.0)
                GAP[k, l] = g; GAP[l, k] = g
        F[:] = 0.0; TQ[:] = 0.0; TAX[:] = 0.0; FT[:] = 0.0
        PE = 0.0; Pnc = 0.0
        Fbb = 0.0; FbbA = 0.0; FbbB = 0.0; Fwb = 0.0; Fbody = 0.0
        fmax_step = 0.0; toothH = 0; toothV = 0; zpk = 0.0; xpk = 0.0; touching = False
        # -------------------------------------------- points of box i (robot b) vs boxes of the other robot, floor
        for i in range(nB):
            b = B_bot[i]; ob = 1 - b; ki = B_kind[i]; onrot_i = 1 if ki >= 1 else 0
            ji = B_arm[i]
            zmin = BCW[i, 2] - (abs(BOW[i, 2, 0]) * B_h[i, 0] + abs(BOW[i, 2, 1]) * B_h[i, 1] + abs(BOW[i, 2, 2]) * B_h[i, 2])
            near_any = zmin < 0.0
            for k in range(nB):
                if B_bot[k] == ob and GAP[i, k] < 0.0:
                    near_any = True
            s0 = B_rng[i, 0]; s1 = B_rng[i, 1]
            if not near_any:
                if WASNEAR[i]:
                    for pi in range(s0, s1):
                        for cc in range(VIN.shape[1]):
                            VIN[pi, cc] = -1.0
                    WASNEAR[i] = False
                continue
            WASNEAR[i] = True
            if ki == 0:
                Rg = RB[b]; ox = X[b, 0]; oy = X[b, 1]; oz = X[b, 2]
            else:
                Rg = RR[b]; ox = CW[b, 0]; oy = CW[b, 1]; oz = CW[b, 2]
            for pi in range(s0, s1):
                x0, x1, x2 = _mv(Rg, P_loc[pi, 0], P_loc[pi, 1], P_loc[pi, 2])
                pux = ox + x0; puy = oy + x1; puz = oz + x2          # undeformed (rigid-attached) position
                xi = P_xi[pi]
                if ki == 2:
                    pwx = pux + xi * UW[ji, 0]; pwy = puy + xi * UW[ji, 1]; pwz = puz + xi * UW[ji, 2]
                else:
                    pwx = pux; pwy = puy; pwz = puz
                have_v = False; vpx = 0.0; vpy = 0.0; vpz = 0.0
                for k in range(nB):
                    if B_bot[k] != ob:
                        continue
                    slot = B_slot[k]
                    if GAP[i, k] >= 0.0:
                        VIN[pi, slot] = -1.0
                        continue
                    dx = pwx - BCW[k, 0]; dy = pwy - BCW[k, 1]; dz = pwz - BCW[k, 2]
                    rk = BRAD[k] + (0.01 if B_kind[k] == 2 else 0.0)
                    if dx * dx + dy * dy + dz * dz > rk * rk:
                        VIN[pi, slot] = -1.0
                        continue
                    kk = B_kind[k]; onrot_k = 1 if kk >= 1 else 0
                    xik = 0.0; jk = B_arm[k]
                    if kk == 2:
                        # deformed arm: map the point back to the arm's material coordinates (exact inverse of
                        # x = r0 + xi(r0) u, xi linear in the radial coordinate) so the penalty potential and the
                        # applied force come from the same geometry
                        y0, y1, y2 = _mtv(RR[ob], pwx - CW[ob, 0], pwy - CW[ob, 1], pwz - CW[ob, 2])
                        ur0, ur1, ur2 = _mtv(RR[ob], UW[jk, 0], UW[jk, 1], UW[jk, 2])
                        sgm = A_sign[jk]; Lk = A_L[jk]; rr = A_root[jk]
                        den = 1.0 + sgm * ur0 / Lk
                        sr = (sgm * y0 + sgm * rr * ur0 / Lk) / den
                        xik = (sr - rr) / Lk
                        r1 = y1 - xik * ur1; r2 = y2 - xik * ur2
                        lx = sr - (rr + 0.5 * Lk); ly = r1; lz = r2
                    else:
                        lx, ly, lz = _mtv(BOW[k], dx, dy, dz)
                    pen, ax, sg = point_box_penetration(lx, ly, lz, B_h[k, 0], B_h[k, 1], B_h[k, 2])
                    if pen <= 0.0:
                        VIN[pi, slot] = -1.0
                        continue
                    if not have_v:
                        vpx, vpy, vpz = _vrig(b, onrot_i, pux, puy, puz, X, Vv, WW, CW, AW, OM)
                        if ki == 2:
                            ax_, ay_, az_ = _vrig(b, 1, AW_anchor[ji, 0], AW_anchor[ji, 1], AW_anchor[ji, 2], X, Vv, WW, CW, AW, OM)
                            vpx += xi * (VT[ji, 0] - ax_); vpy += xi * (VT[ji, 1] - ay_); vpz += xi * (VT[ji, 2] - az_)
                        have_v = True
                    qux = pwx; quy = pwy; quz = pwz
                    if kk == 2:
                        # outward normal = -d(pen)/d(x): material normal pulled back through the deformation map
                        v0 = 0.0; v1 = 0.0; v2 = 0.0
                        if ax == 0:
                            v0 = sg * sgm
                        elif ax == 1:
                            v1 = sg
                        else:
                            v2 = sg
                        uv = ur0 * v0 + ur1 * v1 + ur2 * v2
                        v0 -= sgm / Lk * uv / den
                        nx, ny, nz = _mv(RR[ob], v0, v1, v2)
                        qux -= xik * UW[jk, 0]; quy -= xik * UW[jk, 1]; quz -= xik * UW[jk, 2]
                    else:
                        nx = BOW[k, 0, ax] * sg; ny = BOW[k, 1, ax] * sg; nz = BOW[k, 2, ax] * sg
                    vbx, vby, vbz = _vrig(ob, onrot_k, qux, quy, quz, X, Vv, WW, CW, AW, OM)
                    if kk == 2:
                        ax_, ay_, az_ = _vrig(ob, 1, AW_anchor[jk, 0], AW_anchor[jk, 1], AW_anchor[jk, 2], X, Vv, WW, CW, AW, OM)
                        vbx += xik * (VT[jk, 0] - ax_); vby += xik * (VT[jk, 1] - ay_); vbz += xik * (VT[jk, 2] - az_)
                    vrx = vpx - vbx; vry = vpy - vby; vrz = vpz - vbz
                    vn = vrx * nx + vry * ny + vrz * nz
                    pdot = -vn
                    if VIN[pi, slot] < 0.0:
                        VIN[pi, slot] = pdot if pdot > 0.1 else 0.1
                    kp = P_k[pi]
                    alpha = 8.0 * (1.0 - e_bb) / (5.0 * e_bb * VIN[pi, slot])      # Flores (2011)
                    fn = kp * pen * (1.0 + alpha * pdot)
                    if fn < 0.0:
                        fn = 0.0
                    nn = math.sqrt(nx * nx + ny * ny + nz * nz)
                    ux_ = nx / nn; uy_ = ny / nn; uz_ = nz / nn
                    vnu = vrx * ux_ + vry * uy_ + vrz * uz_
                    ftx, fty, ftz = coulomb_regularized_3d(vrx - vnu * ux_, vry - vnu * uy_, vrz - vnu * uz_, fn * nn,
                                                           mu_bb, v_eps)
                    fx = fn * nx + ftx; fy = fn * ny + fty; fz = fn * nz + ftz
                    PE += 0.5 * kp * pen * pen
                    fe = kp * pen
                    Pnc += (fx - fe * nx) * vrx + (fy - fe * ny) * vry + (fz - fe * nz) * vrz
                    # +f on robot b's point, -f on robot ob's box material
                    if ki == 2:
                        _apply(b, 1, pux, puy, puz, fx, fy, fz, X, CW, AW, F, TQ, TAX)
                        _apply(b, 1, AW_anchor[ji, 0], AW_anchor[ji, 1], AW_anchor[ji, 2], -xi * fx, -xi * fy, -xi * fz,
                               X, CW, AW, F, TQ, TAX)
                        FT[ji, 0] += xi * fx; FT[ji, 1] += xi * fy; FT[ji, 2] += xi * fz
                    else:
                        _apply(b, onrot_i, pux, puy, puz, fx, fy, fz, X, CW, AW, F, TQ, TAX)
                    if kk == 2:
                        _apply(ob, 1, qux, quy, quz, -fx, -fy, -fz, X, CW, AW, F, TQ, TAX)
                        _apply(ob, 1, AW_anchor[jk, 0], AW_anchor[jk, 1], AW_anchor[jk, 2], xik * fx, xik * fy, xik * fz,
                               X, CW, AW, F, TQ, TAX)
                        FT[jk, 0] -= xik * fx; FT[jk, 1] -= xik * fy; FT[jk, 2] -= xik * fz
                    else:
                        _apply(ob, onrot_k, qux, quy, quz, -fx, -fy, -fz, X, CW, AW, F, TQ, TAX)
                    rel = pen / B_h[k, ax]
                    if rel > max_pen_ratio:
                        max_pen_ratio = rel
                    if ki >= 1 and kk >= 1:
                        Fbb += fn
                        if b == 0:
                            FbbA += fn
                        else:
                            FbbB += fn
                        if fn > fmax_step:
                            fmax_step = fn; zpk = pwz; xpk = pwx
                            jH = ji if b == 0 else jk
                            jV = jk if b == 0 else ji
                            toothH = 0 if jH < 0 else int(A_sign[jH])
                            toothV = 0 if jV < 0 else int(A_sign[jV])
                    elif ki >= 1 or kk >= 1:
                        Fwb += fn
                    else:
                        Fbody += fn
                # ---- floor
                if pwz < 0.0:
                    touching = True
                    if not have_v:
                        vpx, vpy, vpz = _vrig(b, onrot_i, pux, puy, puz, X, Vv, WW, CW, AW, OM)
                        if ki == 2:
                            ax_, ay_, az_ = _vrig(b, 1, AW_anchor[ji, 0], AW_anchor[ji, 1], AW_anchor[ji, 2], X, Vv, WW, CW, AW, OM)
                            vpx += xi * (VT[ji, 0] - ax_); vpy += xi * (VT[ji, 1] - ay_); vpz += xi * (VT[ji, 2] - az_)
                        have_v = True
                    pen = -pwz
                    if VIN[pi, FL] < 0.0:
                        VIN[pi, FL] = -vpz if -vpz > 0.1 else 0.1
                    alpha = 8.0 * (1.0 - e_fl) / (5.0 * e_fl * VIN[pi, FL])
                    fn = P_kf[pi] * pen * (1.0 - alpha * vpz)
                    if fn < 0.0:
                        fn = 0.0
                    ftx, fty, ftz = coulomb_regularized_3d(vpx, vpy, 0.0, fn, mu_fl, v_eps)
                    PE += 0.5 * P_kf[pi] * pen * pen
                    Pnc += ftx * vpx + fty * vpy + (fn - P_kf[pi] * pen) * vpz
                    if ki == 2:
                        _apply(b, 1, pux, puy, puz, ftx, fty, fn, X, CW, AW, F, TQ, TAX)
                        _apply(b, 1, AW_anchor[ji, 0], AW_anchor[ji, 1], AW_anchor[ji, 2], -xi * ftx, -xi * fty, -xi * fn,
                               X, CW, AW, F, TQ, TAX)
                        FT[ji, 0] += xi * ftx; FT[ji, 1] += xi * fty; FT[ji, 2] += xi * fn
                    else:
                        _apply(b, onrot_i, pux, puy, puz, ftx, fty, fn, X, CW, AW, F, TQ, TAX)
                else:
                    VIN[pi, FL] = -1.0
        # -------------------------------------------- supports
        for s in range(nS):
            b = S_bot[s]
            x0, x1, x2 = _mv(RB[b], S_loc[s, 0], S_loc[s, 1], S_loc[s, 2])
            pwx = X[b, 0] + x0; pwy = X[b, 1] + x1; pwz = X[b, 2] + x2
            if pwz < 0.0:
                vx, vy, vz = _vrig(b, 0, pwx, pwy, pwz, X, Vv, WW, CW, AW, OM)
                pen = -pwz
                fn = k_sup * pen - S_c[s] * vz
                if fn < 0.0:
                    fn = 0.0
                hx = RB[b, 0, 0]; hy = RB[b, 1, 0]; hn = math.sqrt(hx * hx + hy * hy); ex0 = hx / hn; ex1 = hy / hn
                vl = vx * ex0 + vy * ex1; vt = -vx * ex1 + vy * ex0
                den = math.sqrt(vl * vl + vt * vt + v_eps * v_eps)
                fl = -S_mux[s] * fn * vl / den; ftr = -S_muy[s] * fn * vt / den
                fx = fl * ex0 - ftr * ex1; fy = fl * ex1 + ftr * ex0
                PE += 0.5 * k_sup * pen * pen
                Pnc += fx * vx + fy * vy + (fn - k_sup * pen) * vz
                _apply(b, 0, pwx, pwy, pwz, fx, fy, fn, X, CW, AW, F, TQ, TAX)
        # -------------------------------------------- arm springs (tooth mass <-> rotor)
        for j in range(nA):
            b = A_bot[j]
            u0, u1, u2 = _mtv(RR[b], UW[j, 0], UW[j, 1], UW[j, 2])
            fr = A_kax[j] * u0
            xpo0 = XP[j, 0]; xpo1 = XP[j, 1]
            fl, XP[j, 0], ACC[j, 0] = tabular_return(u1, XP[j, 0], ACC[j, 0], A_kl[j], A_la[j], A_lF[j], 0)
            ff, XP[j, 1], ACC[j, 1] = tabular_return(u2, XP[j, 1], ACC[j, 1], A_kf[j], A_fa[j], A_fF[j], 0)
            Wpl += fl * (XP[j, 0] - xpo0) + ff * (XP[j, 1] - xpo1)
            PE += 0.5 * A_kax[j] * u0 * u0 + fl * fl / (2 * A_kl[j]) + ff * ff / (2 * A_kf[j])
            SPF[j, 0] = fr; SPF[j, 1] = fl; SPF[j, 2] = ff
            sx, sy, sz = _mv(RR[b], fr, fl, ff)                      # spring tension: +s on rotor, -s on tooth
            vrx, vry, vrz = _vrig(b, 1, XT[j, 0], XT[j, 1], XT[j, 2], X, Vv, WW, CW, AW, OM)
            vrx = VT[j, 0] - vrx; vry = VT[j, 1] - vry; vrz = VT[j, 2] - vrz
            c = A_c[j]
            sx += c * vrx; sy += c * vry; sz += c * vrz
            Pnc -= c * (vrx * vrx + vry * vry + vrz * vrz)
            _apply(b, 1, XT[j, 0], XT[j, 1], XT[j, 2], sx, sy, sz, X, CW, AW, F, TQ, TAX)
            FT[j, 0] -= sx; FT[j, 1] -= sy; FT[j, 2] -= sz
            if abs(u1) > UMAX[j, 0]:
                UMAX[j, 0] = abs(u1)
            if abs(u2) > UMAX[j, 1]:
                UMAX[j, 1] = abs(u2)
        # -------------------------------------------- optional external tip load (load-table studies)
        if ext_arm >= 0 and EXT_T.shape[0] > 1 and t >= EXT_T[0] and t <= EXT_T[EXT_T.shape[0] - 1]:
            k_ = np.searchsorted(EXT_T, t) - 1
            k_ = 0 if k_ < 0 else (EXT_T.shape[0] - 2 if k_ > EXT_T.shape[0] - 2 else k_)
            w_ = (t - EXT_T[k_]) / (EXT_T[k_ + 1] - EXT_T[k_])
            fa0 = EXT_F[k_, 0] + w_ * (EXT_F[k_ + 1, 0] - EXT_F[k_, 0])
            fa1 = EXT_F[k_, 1] + w_ * (EXT_F[k_ + 1, 1] - EXT_F[k_, 1])
            fa2 = EXT_F[k_, 2] + w_ * (EXT_F[k_ + 1, 2] - EXT_F[k_, 2])
            b = A_bot[ext_arm]; sg_ = A_sign[ext_arm]
            if ext_body == 1:
                # fixed on the robot body at the blade tip position with the blade pointing forward (body +x);
                # arm frame = body frame there (X radial = forward, Y along rotation = left, Z up)
                fx, fy, fz = _mv(RB[b], fa0, fa1, fa2)
                px_, py_, pz_ = _mv(RB[b], D[b, 0] + A_R[ext_arm], D[b, 1], D[b, 2])
                ax_ = X[b, 0] + px_; ay_ = X[b, 1] + py_; az_ = X[b, 2] + pz_
                vx_, vy_, vz_ = _vrig(b, 0, ax_, ay_, az_, X, Vv, WW, CW, AW, OM)
                _apply(b, 0, ax_, ay_, az_, fx, fy, fz, X, CW, AW, F, TQ, TAX)
            else:
                fx, fy, fz = _mv(RR[b], sg_ * fa0, sg_ * fa1, fa2)
                ax_, ay_, az_ = AW_anchor[ext_arm, 0], AW_anchor[ext_arm, 1], AW_anchor[ext_arm, 2]
                vx_, vy_, vz_ = _vrig(b, 1, ax_, ay_, az_, X, Vv, WW, CW, AW, OM)
                _apply(b, 1, ax_, ay_, az_, fx, fy, fz, X, CW, AW, F, TQ, TAX)
            Pnc += fx * vx_ + fy * vy_ + fz * vz_                   # external work enters the ledger as a source
        # -------------------------------------------- energy at the start of this step
        KE = 0.0; PG = 0.0
        for b in range(2):
            KE += 0.5 * m[b] * (Vv[b, 0] ** 2 + Vv[b, 1] ** 2 + Vv[b, 2] ** 2)
            KE += gyrostat_energy(J[b], Is[b], A[b], W[b], OM[b])
            PG += m[b] * gval * X[b, 2]
        for j in range(nA):
            KE += 0.5 * A_m[j] * (VT[j, 0] ** 2 + VT[j, 1] ** 2 + VT[j, 2] ** 2)
            PG += A_m[j] * gval * XT[j, 2]
        Etot = KE + PG + PE + Wpl
        if it == 0:
            E0 = Etot
        resid = Etot - E0 - Wnc
        # -------------------------------------------- hit episodes
        if Fbb > 0.0:
            if not in_ep:
                if nep > 0 and t - last_c < 5e-6:
                    nep -= 1
                elif nep < EP.shape[0]:
                    EP[nep, 0] = t
                    for c in range(2, 6):
                        EP[nep, c] = 0.0
                    EP[nep, 6] = 0.5 * Is[0] * (OM[0] + A[0, 0] * W[0, 0] + A[0, 1] * W[0, 1] + A[0, 2] * W[0, 2]) ** 2
                    EP[nep, 7] = 0.5 * Is[1] * (OM[1] + A[1, 0] * W[1, 0] + A[1, 1] * W[1, 1] + A[1, 2] * W[1, 2]) ** 2
                in_ep = True
            if nep < EP.shape[0]:
                EP[nep, 2] += Fbb * dt; EP[nep, 4] += FbbA * dt; EP[nep, 5] += FbbB * dt
                if Fbb > EP[nep, 3]:
                    EP[nep, 3] = Fbb; EP[nep, 10] = toothH; EP[nep, 11] = toothV
                    EP[nep, 12] = xpk; EP[nep, 13] = zpk
            last_c = t
        elif in_ep:
            if nep < EP.shape[0]:
                EP[nep, 1] = t
                EP[nep, 8] = 0.5 * Is[0] * (OM[0] + A[0, 0] * W[0, 0] + A[0, 1] * W[0, 1] + A[0, 2] * W[0, 2]) ** 2
                EP[nep, 9] = 0.5 * Is[1] * (OM[1] + A[1, 0] * W[1, 0] + A[1, 1] * W[1, 1] + A[1, 2] * W[1, 2]) ** 2
            nep += 1; in_ep = False
        if Fbb > Fbb_peak:
            Fbb_peak = Fbb
        if Fwb > 0.0 and t_wb < 0.0:
            t_wb = t
        # -------------------------------------------- history
        if every > 0 and tick >= next_rec and jrow < H.shape[0]:
            next_rec = (tick // every + 1) * every
            H[jrow, 0] = t; H[jrow, 1] = Fbb; H[jrow, 2] = Fwb; H[jrow, 3] = Fbody
            H[jrow, 4] = OM[0] + A[0, 0] * W[0, 0] + A[0, 1] * W[0, 1] + A[0, 2] * W[0, 2]
            H[jrow, 5] = OM[1] + A[1, 0] * W[1, 0] + A[1, 1] * W[1, 1] + A[1, 2] * W[1, 2]
            for b in range(2):
                o = 6 + 8 * b
                H[jrow, o:o + 3] = X[b]; H[jrow, o + 3:o + 7] = Q[b]; H[jrow, o + 7] = PSI[b]
            H[jrow, 22] = resid; H[jrow, 23] = FbbA; H[jrow, 24] = FbbB
            H[jrow, 25:28] = Vv[0]; H[jrow, 28:31] = Vv[1]
            for j in range(nA):
                u0, u1, u2 = _mtv(RR[A_bot[j]], UW[j, 0], UW[j, 1], UW[j, 2])
                H[jrow, 31 + 4 * j] = u1; H[jrow, 32 + 4 * j] = u2
                H[jrow, 33 + 4 * j] = XP[j, 0]; H[jrow, 34 + 4 * j] = XP[j, 1]
                H[jrow, 47 + 3 * j:50 + 3 * j] = XT[j]
                H[jrow, 59 + 3 * j:62 + 3 * j] = SPF[j]          # radial, lag, flap force on the rigid blade at the tip
            H[jrow, 71:74] = W[0]                                # robot 0 body angular velocity, body axes
            jrow += 1
        if Fbb > 0.0 and t_bb0 < 0.0:
            t_bb0 = t
        if tick >= n or (t_after > 0.0 and t_bb0 >= 0.0 and t > t_bb0 + t_after):
            break
        # ---- step size: base dt whenever anything touches or could touch within this step; otherwise a multiple
        # of it, limited by the gap to the nearest box (gap-driven) and by mmax (stiffest spring's stability)
        mult = 1
        if mmax > 1 and Fbb == 0.0 and Fwb == 0.0 and Fbody == 0.0 and not touching:
            vmax = 1.0
            for b in range(2):
                vmax += math.sqrt(Vv[b, 0] ** 2 + Vv[b, 1] ** 2 + Vv[b, 2] ** 2) + 0.25 * math.sqrt(W[b, 0] ** 2 + W[b, 1] ** 2 + W[b, 2] ** 2)
            vt = 0.0
            for j in range(nA):
                vv = math.sqrt(VT[j, 0] ** 2 + VT[j, 1] ** 2 + VT[j, 2] ** 2)
                if vv > vt:
                    vt = vv
            vmax += 2.0 * vt
            clr = 1e9
            for k in range(nB):
                zmin = BCW[k, 2] - (abs(BOW[k, 2, 0]) * B_h[k, 0] + abs(BOW[k, 2, 1]) * B_h[k, 1] + abs(BOW[k, 2, 2]) * B_h[k, 2])
                if zmin < clr:
                    clr = zmin
                if B_bot[k] == 0:
                    for l in range(nB):
                        if B_bot[l] == 1 and GAP[k, l] < clr:
                            clr = GAP[k, l]
            if clr > 0.0:
                mult = int(0.3 * clr / vmax / dt)
                mult = 1 if mult < 1 else (mmax if mult > mmax else mult)
        if tick + mult > n:
            mult = n - tick if n - tick > 0 else 1
        h = mult * dt
        tick += mult; nsteps += 1
        # -------------------------------------------- integrate (symplectic Euler)
        Wnc += Pnc * h
        for b in range(2):
            tx, ty, tz = _mtv(RB[b], TQ[b, 0], TQ[b, 1], TQ[b, 2])
            TB[0] = tx; TB[1] = ty; TB[2] = tz
            tb = TB
            if gyro_on < 0.5:          # knockout: remove the rotor's angular momentum from w x h
                gx, gy, gz = _cr(W[b, 0], W[b, 1], W[b, 2], A[b, 0], A[b, 1], A[b, 2])
                tb[0] += Is[b] * OM[b] * gx; tb[1] += Is[b] * OM[b] * gy; tb[2] += Is[b] * OM[b] * gz
            wd, od = gyrostat_accel(J[b], Is[b], A[b], W[b], OM[b], tb, TAX[b])
            Vv[b, 0] += h * F[b, 0] / m[b]; Vv[b, 1] += h * F[b, 1] / m[b]; Vv[b, 2] += h * (F[b, 2] / m[b] - gval)
            W[b, 0] += h * wd[0]; W[b, 1] += h * wd[1]; W[b, 2] += h * wd[2]
            OM[b] += h * od
        for j in range(nA):
            VT[j, 0] += h * FT[j, 0] / A_m[j]; VT[j, 1] += h * FT[j, 1] / A_m[j]
            VT[j, 2] += h * (FT[j, 2] / A_m[j] - gval)
        for b in range(2):
            X[b, 0] += h * Vv[b, 0]; X[b, 1] += h * Vv[b, 1]; X[b, 2] += h * Vv[b, 2]
            Q[b] = quat_step(Q[b], W[b], h)
            PSI[b] += h * OM[b]
        for j in range(nA):
            XT[j, 0] += h * VT[j, 0]; XT[j, 1] += h * VT[j, 1]; XT[j, 2] += h * VT[j, 2]
    if in_ep and nep < EP.shape[0]:
        EP[nep, 1] = tick * dt
        EP[nep, 8] = 0.5 * Is[0] * (OM[0] + A[0, 0] * W[0, 0] + A[0, 1] * W[0, 1] + A[0, 2] * W[0, 2]) ** 2
        EP[nep, 9] = 0.5 * Is[1] * (OM[1] + A[1, 0] * W[1, 0] + A[1, 1] * W[1, 1] + A[1, 2] * W[1, 2]) ** 2
        nep += 1
    return nep, jrow, resid / E0, max_pen_ratio, Fbb_peak, E0, Wpl, t_wb, nsteps, tick * dt


EP_COLS = ["t0", "t1", "J", "Fpk", "J_Hpts", "J_Vpts", "EH0", "EV0", "EH1", "EV1", "toothH", "toothV", "x_pk", "z_pk"]


def merge_hits(ep, t_merge):
    """Merge raw blade-to-blade contact episodes separated by less than t_merge into hits."""
    hits = []
    for row in ep:
        e = dict(zip(EP_COLS, (float(v) for v in row)))
        if hits and e["t0"] - hits[-1]["t1"] < t_merge:
            h = hits[-1]
            h["t1"] = e["t1"]; h["J"] += e["J"]; h["J_Hpts"] += e["J_Hpts"]; h["J_Vpts"] += e["J_Vpts"]
            h["EH1"] = e["EH1"]; h["EV1"] = e["EV1"]; h["n_raw"] += 1
            if e["Fpk"] > h["Fpk"]:
                for k in ("Fpk", "toothH", "toothV", "x_pk", "z_pk"):
                    h[k] = e[k]
        else:
            e["n_raw"] = 1; hits.append(e)
    return hits


T_MERGE = 1e-4            # raw episodes closer than this are one hit (contact chatter)
T_FAST = 1.5e-3           # a second hit sooner than this cannot be the next tooth (half-rev >= 2.5 ms)
T_TAIL = 6e-3             # an engagement counts only if the sim runs this long after the first hit


def run(params=None, dt=2.5e-7, history=False, t_merge=T_MERGE, raw=False):
    p = dict(DEFAULTS); p.update(params or {})
    b = build(p)
    X, Vv, Q, W, PSI, OM, XT, VT = initial_state(p, b)
    bots = b["bots"]
    m = np.array([x["mb"] for x in bots]); J = np.array([x["J"] for x in bots]); Is = np.array([x["Is_r"] for x in bots])
    A = np.array([x["a"] for x in bots]); D = np.array([x["d_b"] for x in bots]); RAB = np.array([x["Rab"] for x in bots])
    n = int(round(p["t_end"] / dt))
    every = max(int(round(p.get("hist_dt", 5e-6) / dt)), 1) if history else 0
    Hh = np.zeros((n // every + 2 if history else 1, 74))
    EP = np.zeros((400, 14)); VIN = -np.ones((b["P_loc"].shape[0], NVIN))
    XP = np.zeros((4, 2)); ACC = np.zeros((4, 2)); UMAX = np.zeros((4, 2))
    mmax = max(1, int(p["dt_mult_max"])) if p["adaptive"] > 0.5 else 1      # scales with dt: convergence covers it
    ext = p.get("ext_load")                   # dict(t=..., F=(n,3) arm frame X radial, Y along rotation, Z up, arm=0)
    if ext is not None:
        ext_t = np.ascontiguousarray(ext["t"], dtype=float); ext_F = np.ascontiguousarray(ext["F"], dtype=float)
        ext_arm = int(ext.get("arm", 0)); ext_body = int(ext.get("on_body", 0))
    else:
        ext_t = np.zeros(1); ext_F = np.zeros((1, 3)); ext_arm = -1; ext_body = 0
    nep, jrow, led, pen_ratio, Fpk, E0, Wpl, t_wb, nsteps, t_stop = _loop(
        n, dt, X, Vv, Q, W, PSI, OM, XT, VT, XP, ACC, m, J, Is, A, D, RAB,
        b["B_bot"], b["B_kind"], b["B_arm"], b["B_c"], b["B_h"], b["B_rng"], b["B_slot"],
        b["P_loc"], b["P_xi"], b["P_k"], b["P_kf"],
        b["A_bot"], b["A_sign"], b["A_R"], b["A_root"], b["A_L"], b["A_h"], b["A_m"], b["A_kax"], b["A_kl"], b["A_kf"],
        b["A_la"], b["A_lF"], b["A_fa"], b["A_fF"], b["A_c"],
        b["S_bot"], b["S_loc"], b["S_mux"], b["S_muy"], b["S_c"], p["k_sup"],
        p["e_bb"], p["mu_bb"], p["e_floor"], p["mu_floor"], p["v_eps"], p["gyro_on"], every, Hh, EP, VIN, UMAX,
        mmax, p["t_after"], p["gravity"], ext_t, ext_F, ext_arm, ext_body)
    ep = EP[:nep]
    hits = merge_hits(ep, t_merge)

    def hv(i, k):
        return hits[i][k] if len(hits) > i else 0.0
    mm = 1e3
    out = dict(
        n_hits=float(len(hits)),
        double_hit=1.0 if len(hits) >= 2 else 0.0,
        t_first_ms=hv(0, "t0") * mm,
        gap12_ms=(hits[1]["t0"] - hits[0]["t1"]) * mm if len(hits) > 1 else 0.0,
        J1_Ns=hv(0, "J"), J2_Ns=hv(1, "J"),
        F_peak_kN=Fpk / 1e3,
        set_flap_H_mm=float(np.max(np.abs(XP[:2, 1]))) * mm, set_lag_H_mm=float(np.max(np.abs(XP[:2, 0]))) * mm,
        set_flap_V_mm=float(np.max(np.abs(XP[2:, 1]))) * mm, set_lag_V_mm=float(np.max(np.abs(XP[2:, 0]))) * mm,
        defl_flap_H_mm=float(np.max(UMAX[:2, 1])) * mm, defl_lag_H_mm=float(np.max(UMAX[:2, 0])) * mm,
        defl_flap_V_mm=float(np.max(UMAX[2:, 1])) * mm, defl_lag_V_mm=float(np.max(UMAX[2:, 0])) * mm,
        W_plastic_J=Wpl, pen_ratio_max=pen_ratio, _ledger_rel=led, _steps=float(nsteps), _t_stop_ms=t_stop * 1e3)
    rpm = lambda E, I: math.sqrt(max(2 * E / I, 0.0)) * 60 / (2 * math.pi)
    out["rpmH_at_hit1"] = rpm(hv(0, "EH0"), Is[0]) if hits else 0.0
    out["rpmV_at_hit1"] = rpm(hv(0, "EV0"), Is[1]) if hits else 0.0
    out["t_wb_first_ms"] = t_wb * 1e3 if t_wb >= 0 else -1.0
    out["wb_before_bb"] = 1.0 if (t_wb >= 0 and (not hits or t_wb < hits[0]["t0"])) else 0.0
    valid = 1.0 if hits and hits[0]["t0"] <= t_stop - T_TAIL + 1e-9 else 0.0
    out["valid"] = valid
    out["fast_double"] = 1.0 if (valid and len(hits) > 1 and out["gap12_ms"] < T_FAST * 1e3) else 0.0
    out["double_6ms"] = 1.0 if (valid and len(hits) > 1 and out["gap12_ms"] < T_TAIL * 1e3) else 0.0
    if raw or history:
        extra = dict(hits=hits, episodes=ep, E0=E0, build=b, params=p, XP=XP.copy(), UMAX=UMAX.copy())
        if history:
            extra["hist"] = Hh[:jrow]
        return out, extra
    return out


def critical_dt(params):
    """Stiffest plausible blade contact: 40 sampled points (60 mm of edge) against the lighter tooth mass."""
    p = dict(DEFAULTS); p.update(params or {})
    b = build(p)
    k = 40 * p["k_line"] * p["ds_blade"]
    m_min = float(b["A_m"].min())
    return min(2.0 / math.sqrt(k / m_min), 2.0 / math.sqrt(float(b["A_kax"].max()) / m_min))


# ------------------------------------------------------------------ study definition shared by the report and
# the local batch runner (simblocks batch spinner_vs_spinner_job.yaml)
# Sampled dimensions and sample count are NOT defined here: the user sets them in the job file
# (spinner_vs_spinner_job.yaml, samples.uniform and samples.n). make_spec takes them from there.
CFG = {"elastic-plastic (both)": {}, "elastic only": dict(plastic=0.0), "H blade flexes only": dict(flexV=0.0),
       "V blade flexes only": dict(flexH=0.0), "rigid blades": dict(flexH=0.0, flexV=0.0)}
HIT_KEYS = ["t0", "t1", "J", "Fpk", "J_Hpts", "J_Vpts", "EH0", "EV0", "EH1", "EV1", "toothH", "toothV", "x_pk", "z_pk",
            "n_raw"]


def batch_record(params, dt):
    """One engagement for the batch runner: outputs, merged hits and the raw contact episodes (so the hit
    merge threshold can be changed later without rerunning)."""
    o, ex = run(params, dt=dt, raw=True)
    return dict(out=o, hits=[[h[k] for k in HIT_KEYS] for h in ex["hits"]],
                episodes=np.round(ex["episodes"], 9).tolist())


def warmup():
    run(dict(t_end=2e-4))


def make_spec(params, uncertain=None, dt=6.25e-8, mc_n=20):
    """Verification of one run. uncertain = the job file's sampled ranges {param: (lo, hi)}."""
    from simblocks.diagnostics.verify import SimSpec
    return SimSpec(
        name="Spinner vs spinner double hit", run=lambda p, d: run(p, d), params=dict(params), dt=dt,
        metrics=["fast_double", "J1_Ns", "J2_Ns", "gap12_ms", "n_hits", "defl_lag_H_mm", "set_lag_H_mm",
                 "defl_flap_H_mm", "defl_flap_V_mm", "W_plastic_J"],
        uncertain={k: tuple(v) for k, v in (uncertain or {}).items()},
        knockouts={"rigid blades": (dict(flexH=0.0, flexV=0.0), {"fast_double": "-", "W_plastic_J": "-"}),
                   "no blade plasticity": (dict(plastic=0.0), {"W_plastic_J": "-"}),
                   "no blade friction": (dict(mu_bb=0.0), {}),
                   "no gyroscopic coupling": (dict(gyro_on=0.0), {})},
        limits=[("robots never meet (1 ms run) -> no hits", dict(t_end=0.001), {"n_hits": 0.0, "W_plastic_J": 0.0}),
                ("rigid blades -> no bend", dict(flexH=0.0, flexV=0.0), {"set_flap_V_mm": 0.0, "set_flap_H_mm": 0.0})],
        bounds={"pen_ratio_max": (0.0, 0.5), "n_hits": (0, 30), "set_flap_V_mm": (0.0, 20.0),
                "set_flap_H_mm": (0.0, 20.0), "W_plastic_J": (0.0, 1000.0)},
        critical_dt=critical_dt, mc_n=mc_n, ledger_tol=0.05)


def pick_nominal(runs):
    """The dominant fast mechanism: an H-blade snap-back double hit (V blade rigid), no chassis hit first,
    largest smaller-of-two impulses."""
    best = None
    for r in runs:
        if not r["key"].startswith("run|H blade flexes only|") or not r["record"]:
            continue
        o = r["record"]["out"]
        if o["fast_double"] and not o["wb_before_bb"]:
            score = min(o["J1_Ns"], o["J2_Ns"])
            if best is None or score > best[0]:
                best = (score, r["params"])
    return best[1] if best else runs[0]["params"]


if __name__ == "__main__":
    import time
    t0 = time.time(); o, ex = run({}, raw=True); print(f"first call {time.time()-t0:.1f}s")
    t0 = time.time(); o, ex = run({}, raw=True); print(f"second call {time.time()-t0:.2f}s")
    for k, v in o.items():
        print(f"{k:16s} {v:.5g}")
    for h in ex["hits"]:
        print({k: round(float(v), 5) for k, v in h.items()})
