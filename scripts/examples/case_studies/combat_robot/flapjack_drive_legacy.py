#!/usr/bin/env python3
"""
Flapjack drive dynamics simulator  (v1.0)
==========================================

Planar 3-DOF rigid-body model (x, y, yaw) of a 3-point-stance beetleweight
(two driven wheels + one sliding weapon-bolt skid), with:

  * Per-wheel spin DOF (wheel + gearbox-reflected rotor inertia)
  * DC motor model (back-EMF, winding resistance, ESC current limit,
    drag-brake vs coast), planetary gearbox efficiency with correct
    motoring / back-driving loss direction
  * Shared battery with internal resistance (bus sag)
  * Combined-slip tire model: brush-type stiffness at small slip, peak at
    mu_static, decays to mu_kinetic at large slip (friction circle)
  * Static->kinetic (Stribeck) skid friction on the bolt
  * Fore-aft AND lateral load transfer from COM height, with a first-order
    lag representing tire/chassis compliance (removes the algebraic loop)
  * Rolling resistance of soft urethane
  * External yaw torque input (weapon spin-up reaction)
  * Full energy ledger (terminal energy = dKE + every loss) as a closure check

Integrator: fixed-step RK4, numba-compiled. Run:  python flapjack_drive_sim.py
Outputs:  ./sim_out/  (PDF report, PNG figures, CSV + Markdown tables)
"""
import math, os, csv, time, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from simblocks.friction.published import magic_formula, brush_mu
import numpy as np
from numba import njit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

OUT = "sim_out"

# --------------------------------------------------------------------------
# Parameter vector indices
# --------------------------------------------------------------------------
M, GRAV, IZZ, XC, LB, TRK, HC, RW, JW, GR, ETA, KT, RM, VB, IMAX = range(15)
MUS, MUK, SIGP, VMIN, MUBS, MUBK, VSTRB, EPSB, CRR, TAU, BRAKE, RBAT, SLEW = range(15, 28)
# sourced-model extensions (0 = Nicky's legacy curve, 1 = Pacejka MF, 2 = Svendenius brush)
TMODEL, MFB, MFC, MFE, BMUS, BCS, ETAB = range(28, 35)
NP = 35

# state indices
SX, SY, SPSI, SVX, SVY, SR, SWL, SWR, SAX, SAY = range(10)
E_TERM, E_CU, E_GEAR, E_TIRE, E_BOLT, E_RR, E_EXT, E_BAT = range(10, 18)
NS = 18
# diagnostic indices
D_NL, D_NR, D_NB, D_IL, D_IR, D_SL, D_SR, D_VB, D_AX, D_AY, D_ICRX, D_ICRY, \
    D_MUL, D_MUR, D_VBUS, D_IBAT, D_FXL, D_FXR, D_LIFT, D_SIGL = range(20)
ND = 20

# --------------------------------------------------------------------------
# Physics kernels
# --------------------------------------------------------------------------
@njit(cache=True)
def tire_mu(sig, mus, muk, sigp):
    """Combined-slip friction curve: linear (brush) at small slip, peak ~mu_s at
    sig~sigp, decays to mu_k at large slip."""
    sig0 = 0.5 * sigp
    return muk * math.tanh(sig / sig0) + (mus - muk) * (sig / sigp) * math.exp(1.0 - sig / sigp)


@njit(cache=True)
def motor_I(u, w, vbus, p):
    if p[BRAKE] < 0.5 and abs(u) < 0.02:
        return 0.0                      # coast: ESC open circuit
    I = (u * vbus - p[KT] * p[GR] * w) / p[RM]
    if I > p[IMAX]:
        I = p[IMAX]
    elif I < -p[IMAX]:
        I = -p[IMAX]
    return I


@njit(cache=True)
def deriv(s, uL, uR, mz, p, ds, dg):
    m = p[M]; g = p[GRAV]; xc = p[XC]; L = p[LB]; T = p[TRK]; h = p[HC]; rw = p[RW]
    c = math.cos(s[SPSI]); sn = math.sin(s[SPSI])
    vbx = c * s[SVX] + sn * s[SVY]
    vby = -sn * s[SVX] + c * s[SVY]
    r = s[SR]
    # ---- normal loads (quasi-static pitch/roll with lagged accelerations) ----
    NB = m * (g * xc - h * s[SAX]) / L
    if NB < 0.0:
        NB = 0.0
    Nw = m * g - NB
    dN = -2.0 * h * m * s[SAY] / T
    NL = 0.5 * (Nw + dN); NR = 0.5 * (Nw - dN)
    lift = 0.0
    if NB < 0.0:
        NB = 0.0; lift = 1.0
    if NL < 0.0:
        NL = 0.0; lift = 2.0
    if NR < 0.0:
        NR = 0.0; lift = 2.0
    # ---- motors / battery (fixed-point on bus voltage) ----
    vbus = p[VB]; IL = 0.0; IR = 0.0; Ib = 0.0
    G = p[GR]; Kt = p[KT]
    for _ in range(12):            # relaxed fixed point on bus voltage (converges to <1e-6 V)
        IL = motor_I(uL, s[SWL], vbus, p)
        IR = motor_I(uR, s[SWR], vbus, p)
        PL = (IL * p[RM] + Kt * G * s[SWL]) * IL
        PR = (IR * p[RM] + Kt * G * s[SWR]) * IR
        Ib = (PL + PR) / vbus
        vbus = 0.4 * vbus + 0.6 * (p[VB] - p[RBAT] * Ib)
    IL = motor_I(uL, s[SWL], vbus, p)
    IR = motor_I(uR, s[SWR], vbus, p)
    Ib = ((IL * p[RM] + Kt * G * s[SWL]) * IL + (IR * p[RM] + Kt * G * s[SWR]) * IR) / vbus
    # ---- wheel contacts (left +y, right -y) ----
    Fbx = 0.0; Fby = 0.0; Mzt = mz
    Ptire = 0.0; Pgear = 0.0; Prr = 0.0; Pcu = 0.0; Pterm = 0.0
    for side in range(2):
        if side == 0:
            yi = 0.5 * T; N = NL; w = s[SWL]; I = IL
        else:
            yi = -0.5 * T; N = NR; w = s[SWR]; I = IR
        xi = -xc
        vcx = vbx - r * yi
        vcy = vby + r * xi
        sx = vcx - w * rw; sy = vcy
        sm = math.sqrt(sx * sx + sy * sy)
        vref = max(abs(w * rw), math.sqrt(vcx * vcx + vcy * vcy), p[VMIN])
        sig = sm / vref
        if p[TMODEL] == 1.0:
            mu = magic_formula(sig, p[MFB], p[MFC], p[MUS], p[MFE])
        elif p[TMODEL] == 2.0:
            mu = brush_mu(sig, p[BCS] * N, N, p[BMUS], p[MUK])
        else:
            mu = tire_mu(sig, p[MUS], p[MUK], p[SIGP])
        if sm > 1e-12:
            Fx = -mu * N * sx / sm; Fy = -mu * N * sy / sm
        else:
            Fx = 0.0; Fy = 0.0
        Ptire += -(Fx * sx + Fy * sy)
        Fbx += Fx; Fby += Fy
        Mzt += xi * Fy - yi * Fx
        # motor -> gearbox -> wheel
        Tm = Kt * I
        if Tm * w >= 0.0:
            Tout = p[ETA] * G * Tm
        else:
            Tout = G * Tm / p[ETAB]
        Trr = p[CRR] * N * rw * math.tanh(w * rw / 0.02)
        dw = (Tout - Fx * rw - Trr) / p[JW]
        Pgear += Tm * G * w - Tout * w
        Prr += Trr * w
        Pcu += I * I * p[RM]
        Pterm += (I * p[RM] + Kt * G * w) * I
        if side == 0:
            ds[SWL] = dw; dg[D_SL] = sm; dg[D_MUL] = mu; dg[D_FXL] = Fx; dg[D_SIGL] = sig
        else:
            ds[SWR] = dw; dg[D_SR] = sm; dg[D_MUR] = mu; dg[D_FXR] = Fx
    # ---- bolt skid ----
    xb = L - xc
    vcx = vbx; vcy = vby + r * xb
    vm = math.sqrt(vcx * vcx + vcy * vcy)
    mub = p[MUBK] + (p[MUBS] - p[MUBK]) * math.exp(-(vm / p[VSTRB]) ** 2)
    den = math.sqrt(vm * vm + p[EPSB] * p[EPSB])
    Fx = -mub * NB * vcx / den; Fy = -mub * NB * vcy / den
    Pbolt = -(Fx * vcx + Fy * vcy)
    Fbx += Fx; Fby += Fy
    Mzt += xb * Fy
    # ---- rigid body ----
    axb = Fbx / m; ayb = Fby / m
    ds[SX] = s[SVX]; ds[SY] = s[SVY]; ds[SPSI] = r
    ds[SVX] = c * axb - sn * ayb
    ds[SVY] = sn * axb + c * ayb
    ds[SR] = Mzt / p[IZZ]
    ds[SAX] = (axb - s[SAX]) / p[TAU]
    ds[SAY] = (ayb - s[SAY]) / p[TAU]
    ds[E_TERM] = Pterm; ds[E_CU] = Pcu; ds[E_GEAR] = Pgear; ds[E_TIRE] = Ptire
    ds[E_BOLT] = Pbolt; ds[E_RR] = Prr; ds[E_EXT] = mz * r; ds[E_BAT] = p[RBAT] * Ib * Ib
    # ---- diagnostics ----
    dg[D_NL] = NL; dg[D_NR] = NR; dg[D_NB] = NB; dg[D_IL] = IL; dg[D_IR] = IR
    dg[D_VB] = vm; dg[D_AX] = axb; dg[D_AY] = ayb
    if abs(r) > 1e-3:
        dg[D_ICRX] = -vby / r + xc          # instantaneous centre, measured from axle
        dg[D_ICRY] = vbx / r
    else:
        dg[D_ICRX] = np.nan; dg[D_ICRY] = np.nan
    dg[D_VBUS] = vbus; dg[D_IBAT] = Ib; dg[D_LIFT] = lift


@njit(cache=True)
def simulate(s0, uLc, uRc, mzc, p, dt, nlog):
    n = uLc.shape[0]
    nout = n // nlog + 1
    out = np.zeros((nout, 3 + NS + ND))
    s = s0.copy()
    k1 = np.zeros(NS); k2 = np.zeros(NS); k3 = np.zeros(NS); k4 = np.zeros(NS)
    tmp = np.zeros(NS); dg = np.zeros(ND)
    uL = 0.0; uR = 0.0
    j = 0
    for i in range(n):
        # transmitter slew limiter (rate in 1/s, 0 = off)
        if p[SLEW] > 0.0:
            du = p[SLEW] * dt
            uL += min(max(uLc[i] - uL, -du), du)
            uR += min(max(uRc[i] - uR, -du), du)
        else:
            uL = uLc[i]; uR = uRc[i]
        mz = mzc[i]
        if i % nlog == 0:
            deriv(s, uL, uR, mz, p, k1, dg)
            out[j, 0] = i * dt; out[j, 1] = uL; out[j, 2] = uR
            out[j, 3:3 + NS] = s
            out[j, 3 + NS:] = dg
            j += 1
        deriv(s, uL, uR, mz, p, k1, dg)
        for q in range(NS): tmp[q] = s[q] + 0.5 * dt * k1[q]
        deriv(tmp, uL, uR, mz, p, k2, dg)
        for q in range(NS): tmp[q] = s[q] + 0.5 * dt * k2[q]
        deriv(tmp, uL, uR, mz, p, k3, dg)
        for q in range(NS): tmp[q] = s[q] + dt * k3[q]
        deriv(tmp, uL, uR, mz, p, k4, dg)
        for q in range(NS):
            s[q] += dt / 6.0 * (k1[q] + 2 * k2[q] + 2 * k3[q] + k4[q])
    return out[:j]

# --------------------------------------------------------------------------
# Mass model -> COM and yaw inertia (component build-up, parallel axis)
# --------------------------------------------------------------------------
L_BOLT = 0.100   # axle -> weapon bolt contact (m), measured
def mass_model(pins=False, total=1.361, wheel_share=0.35):
    """Components: (name, mass kg, x fwd of axle, y, own Izz).
    The blade's spin inertia is EXCLUDED from chassis yaw inertia: it is
    bearing-decoupled, so yawing the chassis does not spin it up.
    The remaining chassis mass centroid is solved so the COM gives the
    measured CAD wheel share."""
    rect = lambda mk, a, b: mk * (a * a + b * b) / 12.0
    comps = [
        ("Blade (spin decoupled)", 0.280, 0.100, 0.0, 0.0),
        ("Weapon non-spinning parts", 0.200, 0.100, 0.0, 0.200 * 0.015 ** 2),
        ("Hardened wedge", 0.120, 0.115, 0.0, rect(0.120, 0.120, 0.010)),
        ("Drive motor L", 0.0455, 0.0, 0.060, 0.0),
        ("Drive motor R", 0.0455, 0.0, -0.060, 0.0),
        ("Wheel+bushing L", 0.025, 0.0, 0.090, 0.0),
        ("Wheel+bushing R", 0.025, 0.0, -0.090, 0.0),
        ("Battery", 0.075, 0.035, 0.0, rect(0.075, 0.065, 0.032)),
        ("Electronics + wiring", 0.060, 0.030, 0.0, rect(0.060, 0.06, 0.06)),
    ]
    known = sum(c[1] for c in comps)
    x_target = (1 - wheel_share) * L_BOLT
    m_rem = total - known
    x_rem = (total * x_target - sum(c[1] * c[2] for c in comps)) / m_rem
    comps.append(("Chassis plates/armor/fasteners", m_rem, x_rem, 0.0, rect(m_rem, 0.19, 0.16)))
    if pins:
        # move 40 g of mass into rear pins 20 mm behind axle (total unchanged)
        name, mk, xk, yk, ik = comps[-1]
        comps[-1] = (name, mk - 0.040, xk, yk, rect(mk - 0.040, 0.19, 0.16))
        comps.append(("Rear pins", 0.040, -0.020, 0.0, rect(0.040, 0.10, 0.01)))
    mt = sum(c[1] for c in comps)
    xcom = sum(c[1] * c[2] for c in comps) / mt
    ycom = sum(c[1] * c[3] for c in comps) / mt
    izz = sum(c[1] * ((c[2] - xcom) ** 2 + (c[3] - ycom) ** 2) + c[4] for c in comps)
    return dict(comps=comps, m=mt, xcom=xcom, izz=izz, wheel_share=1 - xcom / L_BOLT,
                x_rem=x_rem)

# --------------------------------------------------------------------------
# Motors (Repeat Robotics Compact, 22.6:1 planetary)
# --------------------------------------------------------------------------
def motor_brushed():
    """Fit to published 12 V curve: 1070 rpm free, 3.85 kg.cm stall, 4.3 A stall."""
    G = 22.6
    w_free_m = 1070 * G * 2 * math.pi / 60
    Ke = 12.0 / w_free_m
    R = 12.0 / 4.3
    eta = 0.3776 / (G * Ke * 4.3)
    return dict(name="Compact brushed", G=G, Kt=Ke, R=R, eta=eta, Imax=1e3, Jrot=3e-7)

def motor_brushless():
    """Compact 1806 2300KV (DC-equivalent); R and rotor inertia estimated; 20 A ESC limit."""
    return dict(name="Compact 1806 brushless", G=22.6, Kt=60 / (2 * math.pi * 2300), R=0.15,
                eta=0.82, Imax=20.0, Jrot=8e-7)

# --------------------------------------------------------------------------
# Baseline parameters (MEASURED values flagged)
# --------------------------------------------------------------------------
def base_params(motor=None, **kw):
    motor = motor or motor_brushless()
    mm = mass_model(pins=kw.pop("pins", False))
    d = dict(
        m=mm["m"], g=9.81, Izz=mm["izz"], wheel_share=mm["wheel_share"],
        L=L_BOLT,            # measured
        T=0.180,             # measured track
        h=0.023,             # CAD COM height
        rw=0.022,            # measured wheel radius
        J_wheel=0.5 * 0.025 * 0.022 ** 2,
        mus=2.0,             # measured (upper bound of janky rig), dusty 20A urethane
        muk=1.67,            # measured kinetic, dusty 20A urethane on painted wood
        sigp=0.12,           # slip at peak grip (estimate, soft urethane)
        vmin=0.05,
        mubs=0.25,           # bolt static (assumed 1.2x kinetic)
        mubk=0.203,          # measured kinetic, bare steel button head on painted wood
        vstrb=0.01, epsb=0.005,
        Crr=0.02,            # rolling resistance soft urethane (estimate)
        tau=0.010,           # load-transfer lag (tire/chassis compliance)
        brake=1.0,           # ESC drag brake on
        Vb=18.5,             # 5S nominal (Nicky's setup)
        Rbat=0.05,           # pack internal resistance
        slew=0.0,
        motor=motor,
    )
    d.update(kw)
    return d

def pack(d):
    mo = d["motor"]
    p = np.zeros(NP)
    p[M] = d["m"]; p[GRAV] = d["g"]; p[IZZ] = d["Izz"]
    p[XC] = (1 - d["wheel_share"]) * d["L"]; p[LB] = d["L"]; p[TRK] = d["T"]; p[HC] = d["h"]
    p[RW] = d["rw"]; p[JW] = d["J_wheel"] + mo["Jrot"] * mo["G"] ** 2
    p[GR] = mo["G"]; p[ETA] = mo["eta"]; p[KT] = mo["Kt"]; p[RM] = mo["R"]; p[VB] = d["Vb"]
    p[IMAX] = mo["Imax"]; p[MUS] = d["mus"]; p[MUK] = d["muk"]; p[SIGP] = d["sigp"]
    p[VMIN] = d["vmin"]; p[MUBS] = d["mubs"]; p[MUBK] = d["mubk"]; p[VSTRB] = d["vstrb"]
    p[EPSB] = d["epsb"]; p[CRR] = d["Crr"]; p[TAU] = d["tau"]; p[BRAKE] = d["brake"]
    p[RBAT] = d["Rbat"]; p[SLEW] = d["slew"]
    p[TMODEL] = d.get("tmodel", 0.0); p[MFB] = d.get("mfB", 0.0); p[MFC] = d.get("mfC", 0.0)
    p[MFE] = d.get("mfE", 0.0); p[BMUS] = d.get("brush_mus", d["mus"]); p[BCS] = d.get("brush_cs", 0.0)
    p[ETAB] = d.get("eta_b", mo["eta"])
    return p

# --------------------------------------------------------------------------
# Run helper + result object
# --------------------------------------------------------------------------
DT = 1e-4
class Run:
    def __init__(self, d, out):
        self.d = d; self.o = out
        self.t = out[:, 0]; self.uL = out[:, 1]; self.uR = out[:, 2]
        S = out[:, 3:3 + NS]; D = out[:, 3 + NS:]
        self.S = S; self.D = D
        self.x, self.y, self.psi = S[:, SX], S[:, SY], S[:, SPSI]
        self.vx, self.vy, self.r = S[:, SVX], S[:, SVY], S[:, SR]
        c, sn = np.cos(self.psi), np.sin(self.psi)
        self.vbx = c * self.vx + sn * self.vy
        self.vby = -sn * self.vx + c * self.vy
        self.speed = np.hypot(self.vx, self.vy)
        self.dist = np.concatenate([[0], np.cumsum(self.speed[1:] * np.diff(self.t))])
        W = d["m"] * d["g"]
        self.wshare = (D[:, D_NL] + D[:, D_NR]) / W
        self.ax_g = D[:, D_AX] / d["g"]; self.ay_g = D[:, D_AY] / d["g"]
        rw = d["rw"]
        self.wsurf = 0.5 * (S[:, SWL] + S[:, SWR]) * rw

    def t_at(self, arr, val):
        idx = np.nonzero(arr >= val)[0]
        if len(idx) == 0: return np.nan
        i = idx[0]
        if i == 0: return self.t[0]
        f = (val - arr[i - 1]) / (arr[i] - arr[i - 1])
        return self.t[i - 1] + f * (self.t[i] - self.t[i - 1])

    def at_t(self, arr, tt):
        return np.interp(tt, self.t, arr)

    def ledger(self):
        d = self.d; S = self.S; mo = d["motor"]
        Jw = d["J_wheel"] + mo["Jrot"] * mo["G"] ** 2
        KE = 0.5 * d["m"] * (S[:, SVX] ** 2 + S[:, SVY] ** 2) + 0.5 * d["Izz"] * S[:, SR] ** 2 \
            + 0.5 * Jw * (S[:, SWL] ** 2 + S[:, SWR] ** 2)
        e = S[-1]
        ins = e[E_TERM] + e[E_EXT]
        dke = KE[-1] - KE[0]
        losses = e[E_CU] + e[E_GEAR] + e[E_TIRE] + e[E_BOLT] + e[E_RR]
        res = ins - dke - losses
        scale = max(abs(ins), abs(dke) + losses, 1e-9)
        return dict(E_in=ins, dKE=dke, cu=e[E_CU], gear=e[E_GEAR], tire=e[E_TIRE],
                    bolt=e[E_BOLT], rr=e[E_RR], bat=e[E_BAT], residual=res, rel=res / scale)

def run(d, t_end, cmd, mz=None, s0=None, dt=None, nlog=20):
    dt = DT if dt is None else dt
    n = int(round(t_end / dt))
    t = np.arange(n) * dt
    uL, uR = cmd(t)
    uL = np.clip(np.broadcast_to(uL, t.shape).astype(float), -1, 1)
    uR = np.clip(np.broadcast_to(uR, t.shape).astype(float), -1, 1)
    mzc = np.zeros(n) if mz is None else np.broadcast_to(mz(t), t.shape).astype(float)
    p = pack(d)
    if s0 is None:
        s0 = np.zeros(NS)
    out = simulate(s0, np.ascontiguousarray(uL), np.ascontiguousarray(uR),
                   np.ascontiguousarray(mzc), p, dt, nlog)
    return Run(d, out)

def arcade(thr, steer):
    """Standard tank mix: left = thr - steer, right = thr + steer (left turn for +steer)."""
    return lambda t: (np.clip(thr(t) - steer(t), -1, 1), np.clip(thr(t) + steer(t), -1, 1))

const = lambda v: (lambda t: np.full_like(t, v, dtype=float))
step = lambda t0, a, b: (lambda t: np.where(t < t0, a, b).astype(float))

# --------------------------------------------------------------------------
# Scenario metrics
# --------------------------------------------------------------------------
FT8 = 2.4384

def launch(d, t_end=2.0):
    R = run(d, t_end, lambda t: (np.ones_like(t), np.ones_like(t)))
    W = d["m"] * d["g"]
    return R, dict(
        t_05=R.t_at(R.dist, 0.5), v_05=np.interp(0.5, R.dist, R.speed) if R.dist[-1] >= 0.5 else 0.0,
        t_1=R.t_at(R.dist, 1.0), t_8ft=R.t_at(R.dist, FT8), v_8ft=np.interp(FT8, R.dist, R.speed),
        a_peak=R.ax_g.max(), slip_peak=np.nanmax(R.D[:, D_SL]),
        wshare_peak=R.wshare.max(), I_peak=np.abs(R.D[:, D_IL]).max(),
        vbus_min=R.D[:, D_VBUS].min(), v_top=R.speed[-1])

def point_turn(d, t_end=1.0):
    R = run(d, t_end, lambda t: (-np.ones_like(t), np.ones_like(t)))
    late = R.t > 0.6 * t_end
    return R, dict(
        t_90=R.t_at(R.psi, math.pi / 2), t_180=R.t_at(R.psi, math.pi), t_360=R.t_at(R.psi, 2 * math.pi),
        r_peak=np.degrees(R.r.max()), r_end=np.degrees(R.r[-1]),
        icr_x=np.nanmedian(R.D[late, D_ICRX]) * 1000,
        drift=np.hypot(R.x[-1], R.y[-1]) * 1000)

def step_steer(d, s_amt, thr0=0.8, thr_turn=None, t_cruise=1.2, t_turn=0.8):
    thr_turn = thr0 if thr_turn is None else thr_turn
    thr = lambda t: np.where(t < t_cruise, thr0, thr_turn)
    ste = lambda t: np.where(t < t_cruise, 0.0, s_amt)
    R = run(d, t_cruise + t_turn, arcade(thr, ste))
    i0 = np.searchsorted(R.t, t_cruise)
    psi0 = R.psi[i0]
    win = R.t > t_cruise
    tail = R.t > t_cruise + t_turn - 0.3
    beta = np.degrees(np.arctan2(R.vby, np.maximum(np.abs(R.vbx), 1e-6)))
    chi = np.unwrap(np.arctan2(R.vy, R.vx))          # direction of travel (course)
    chi0 = chi[i0]
    return R, dict(
        dchi_025=np.degrees(R.at_t(chi, t_cruise + 0.25) - chi0),
        dchi_05=np.degrees(R.at_t(chi, t_cruise + 0.5) - chi0),
        v_cruise=R.speed[i0],
        dpsi_025=np.degrees(R.at_t(R.psi, t_cruise + 0.25) - psi0),
        dpsi_05=np.degrees(R.at_t(R.psi, t_cruise + 0.5) - psi0),
        r_peak=np.degrees(np.abs(R.r[win]).max()),
        v_05=R.at_t(R.speed, t_cruise + 0.5),
        ay_peak=np.abs(R.ay_g[win]).max(),
        icr_x=np.nanmedian(R.D[tail, D_ICRX]) * 1000,
        beta=np.nanmedian(beta[tail]))

def ramp_steer(d, thr0=0.8, t_end=5.0, s_max=0.9):
    thr = const(thr0)
    ste = lambda t: np.clip((t - 0.8) / (t_end - 0.8), 0, 1) * s_max
    R = run(d, t_end, arcade(thr, ste))
    rw, T = d["rw"], d["T"]
    wl, wr = R.S[:, SWL] * rw, R.S[:, SWR] * rw
    vmean = 0.5 * (wl + wr)
    kin = (wr - wl) / (T * np.maximum(np.abs(vmean), 1e-3))
    act = R.r / np.maximum(R.speed, 1e-3)
    mask = (R.t > 0.9) & (np.abs(kin) > 0.05) & (R.D[:, D_SIGL] < d["sigp"])
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(mask, act / kin, np.nan)
    return R, dict(ay_max=np.abs(R.ay_g[R.t > 0.8]).max(),
                   ratio_low=np.nanmedian(ratio[(np.abs(R.ay_g) > 0.05) & (np.abs(R.ay_g) < 0.25) & mask]) if np.any((np.abs(R.ay_g) > 0.05) & (np.abs(R.ay_g) < 0.25) & mask) else np.nan), \
        dict(ay=np.abs(R.ay_g), ratio=ratio, kin=kin, act=act, s=ste(R.t))

def braking(d, mode):
    tc = 1.5
    if mode == "brake":
        cmd = lambda t: (np.where(t < tc, 1.0, 0.0), np.where(t < tc, 1.0, 0.0))
    elif mode == "coast":
        d = dict(d, brake=0.0)
        cmd = lambda t: (np.where(t < tc, 1.0, 0.0), np.where(t < tc, 1.0, 0.0))
    else:
        cmd = lambda t: (np.where(t < tc, 1.0, -1.0), np.where(t < tc, 1.0, -1.0))
    R = run(d, tc + 4.0, cmd)
    i0 = np.searchsorted(R.t, tc)
    after = R.t >= tc
    vsign = R.vbx
    stop_idx = np.nonzero((R.t > tc) & (vsign <= 0.02))[0]
    t_stop = R.t[stop_idx[0]] - tc if len(stop_idx) else np.nan
    d_stop = (R.dist[stop_idx[0]] - R.dist[i0]) if len(stop_idx) else np.nan
    W = d["m"] * d["g"]
    return R, dict(v0=R.speed[i0], t_stop=t_stop, d_stop=d_stop * 1000,
                   decel_peak=-R.ax_g[after].min(), nose_peak=(R.D[after, D_NB] / W).max(),
                   wshare_min=R.wshare[after].min(), lift=R.D[after, D_LIFT].max())

PACK_WH = 5 * 3.8 * 0.450        # 5S 450 mAh HV pack energy (assumed)
H_BLADE = 0.955   # N.m.s  (600 J at 12,000 rpm -> I=7.6e-4)
def spinup(d, t95, brake=True, t_end=3.5):
    # current-limited ESC: ~constant motor torque until speed, so reaction = H / t_spin
    mz = lambda t: np.where(t < t95, -H_BLADE / t95, 0.0)
    dd = dict(d, brake=1.0 if brake else 0.0)
    R = run(dd, t_end, lambda t: (np.zeros_like(t), np.zeros_like(t)), mz=mz)
    return R, dict(yaw_final=np.degrees(R.psi[-1]), r_peak=np.degrees(np.abs(R.r).max()),
                   drift=np.hypot(R.x[-1], R.y[-1]) * 1000)

def duty_cycle(d, reps=6):
    """Synthetic 30 s fight segment: charge, brake, point turns, arcs, reverse, pauses."""
    seg = [  # (duration, thr, steer)
        (0.7, 1.0, 0.0), (0.3, 0.0, 0.0), (0.25, 0.0, 1.0), (1.0, 0.8, 0.4),
        (0.5, -1.0, 0.0), (0.3, 0.0, 0.0), (0.6, 1.0, 0.0), (0.25, 0.0, -1.0),
        (0.8, 0.6, -0.3), (0.3, 0.0, 0.0)]
    tl, th, st = [], [], []
    acc = 0.0
    while acc < 30.0:
        for dur, a, b in seg:
            tl.append(acc); th.append(a); st.append(b); acc += dur
    tl = np.array(tl)
    th_f = lambda t: np.array(th)[np.searchsorted(tl, t, side="right") - 1]
    st_f = lambda t: np.array(st)[np.searchsorted(tl, t, side="right") - 1]
    R = run(dict(d, slew=1 / 0.08), 30.0, arcade(th_f, st_f), nlog=50)
    e = R.S[-1]
    E_batt = e[E_TERM] + e[E_BAT]          # J drawn from cells over 30 s
    return R, dict(Wh_30s=E_batt / 3600, Wh_fight=E_batt / 3600 * 6,
                   I_avg=np.mean(np.abs(R.D[:, D_IBAT])), I_peak=np.abs(R.D[:, D_IBAT]).max(),
                   tire_J=e[E_TIRE], bolt_J=e[E_BOLT], cu_J=e[E_CU])

# --------------------------------------------------------------------------
# Plot styling (validated reference palette, light mode)
# --------------------------------------------------------------------------
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2.0, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
    "legend.fontsize": 8, "figure.dpi": 130})

def fig_page(title, sub, nrows, ncols, size=(11, 8.5), **kw):
    f, ax = plt.subplots(nrows, ncols, figsize=size, **kw)
    H = size[1]
    f.text(0.02, 1 - 0.32 / H, title, ha="left", va="center", fontsize=13, fontweight="bold")
    f.text(0.02, 1 - 0.62 / H, sub, ha="left", va="center", fontsize=9, color=INK2)
    f.rtop = 1 - 0.85 / H
    return f, np.atleast_1d(ax).ravel()

def table_page(title, header, rows, note=""):
    f = plt.figure(figsize=(11, 8.5))
    f.suptitle(title, x=0.02, ha="left", fontsize=13, fontweight="bold")
    ax = f.add_axes([0.03, 0.08, 0.94, 0.80]); ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=header, loc="upper left", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.45)
    for (i, j), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        if i == 0:
            cell.set_text_props(fontweight="bold", color=INK); cell.set_facecolor("#efeeea")
        else:
            cell.set_facecolor(SURF)
    if note:
        f.text(0.03, 0.03, note, fontsize=8, color=INK2, wrap=True)
    return f

def fmt(v, nd=2):
    if isinstance(v, str): return v
    if v is None or (isinstance(v, float) and not np.isfinite(v)): return "n/a"
    return f"{v:.{nd}f}"

# --------------------------------------------------------------------------
# Main study
# --------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    tables = {}       # name -> (header, rows)
    figs = []

    base = base_params()
    bl = dict(base, Vb=14.8)   # 4S reference
    mm = mass_model()

    # ---------- Page: model summary & geometry ----------
    f, ax = fig_page("Flapjack drive model: geometry and mass build-up",
                     f"m = {mm['m']*1000:.0f} g | COM {mm['xcom']*1000:.1f} mm ahead of axle -> wheel share "
                     f"{mm['wheel_share']*100:.1f}% | Izz = {mm['izz']*1e4:.1f}e-4 kg.m^2 | "
                     f"solved chassis-remainder centroid {mm['x_rem']*1000:.1f} mm ahead of axle", 1, 2,
                     gridspec_kw=dict(width_ratios=[1.25, 1]))
    a = ax[0]
    a.plot([90, -90, 0, 90], [0, 0, 100, 0], color=GRID, lw=1.5, zorder=1)
    for i, (nm, mk, xk, yk, ik) in enumerate(mm["comps"]):
        a.scatter(-yk * 1000, xk * 1000, s=mk * 3000, color=C[i % 8], alpha=0.8,
                  edgecolor=SURF, linewidth=2, zorder=3)
        a.annotate(str(i + 1), (-yk * 1000, xk * 1000), xytext=(9, 9 if i % 2 else -13),
                   textcoords="offset points", fontsize=8, fontweight="bold", color=INK)
    a.scatter([90, -90], [0, 0], marker="s", s=90, color=INK, zorder=4)
    a.scatter([0], [100], marker="v", s=90, color=INK, zorder=4)
    a.scatter([0], [mm["xcom"] * 1000], marker="+", s=400, color=INK, linewidth=2.5, zorder=5)
    a.annotate("COM", (0, mm["xcom"] * 1000), xytext=(-40, 0), textcoords="offset points",
               color=INK, fontweight="bold", va="center")
    a.annotate("bolt skid", (0, 100), xytext=(-60, 10), textcoords="offset points", color=INK)
    a.annotate("wheel contacts (squares)", (-90, 0), xytext=(-70, -28), textcoords="offset points", color=INK)
    a.set_xlabel("lateral position (mm, +left)"); a.set_ylabel("fore-aft position from axle (mm)")
    a.set_aspect("equal"); a.set_xlim(-130, 130); a.set_ylim(-45, 140)
    a.set_title("Top view: contacts and lumped masses (area ~ mass)")
    a = ax[1]; a.axis("off")
    rows = [[str(i + 1), nm, f"{mk*1000:.0f}", f"{xk*1000:.0f}", f"{yk*1000:.0f}"]
            for i, (nm, mk, xk, yk, ik) in enumerate(mm["comps"])]
    tb = a.table(cellText=rows, colLabels=["#", "Component", "g", "x mm", "y mm"], loc="upper center",
                 colWidths=[0.07, 0.55, 0.12, 0.12, 0.12])
    tb.auto_set_font_size(False); tb.set_fontsize(7.5); tb.scale(1, 1.3)
    for (i, j), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID); cell.set_facecolor("#efeeea" if i == 0 else SURF)
    mb, ml = motor_brushed(), motor_brushless()
    txt = ("Measured inputs: wheel kinetic mu 1.67 / static 2.0 (dusty 20A urethane),\n"
           "bolt kinetic mu 0.203 (bare steel), track 180 mm, axle-bolt 100 mm,\n"
           "wheel r 22 mm, COM height 23 mm, 35/65 split.\n\n"
           f"Reference brushed Compact fit (12 V curve): Kt {mb['Kt']*1e3:.2f} mN.m/A, R {mb['R']:.2f} ohm,\n"
           f"gearbox eta {mb['eta']:.2f}. Reflected rotor inertia {mb['Jrot']*mb['G']**2*1e4:.2f}e-4 kg.m^2\n"
           f"= {mb['Jrot']*mb['G']**2/0.022**2*1000:.0f} g equivalent mass per wheel.\n"
           f"BASELINE brushless 1806 on 5S: reflected rotor {ml['Jrot']*ml['G']**2/0.022**2*1000:.0f} g equiv/wheel, 20 A limit.\n\n"
           "Excluded: blade spin inertia (bearing-decoupled), gyro coupling (spin axis\n"
           "parallel to yaw axis; pitch is quasi-static), floor seams, arena slope.")
    a.text(0.0, 0.0, txt, fontsize=7.2, color=INK2, va="bottom", family="monospace")
    f.tight_layout(rect=(0, 0, 1, f.rtop)); figs.append(f)

    # ---------- 1. Launch ----------
    L_cases = [("5S, full-throttle step", base),
               ("5S, 0.15 s throttle ramp", dict(base, slew=1 / 0.15)),
               ("5S, 0.30 s throttle ramp", dict(base, slew=1 / 0.30)),
               ("4S, step (reference)", bl),
               ("5S, step, 30 A ESC limit", dict(base, motor=dict(motor_brushless(), Imax=30.0)))]
    rows = []; runsL = []
    for nm, d in L_cases:
        R, mt = launch(d); runsL.append((nm, R))
        rows.append([nm, fmt(mt["t_05"]), fmt(mt["v_05"]), fmt(mt["t_8ft"]), fmt(mt["v_8ft"]),
                     fmt(mt["a_peak"]), fmt(mt["slip_peak"]), fmt(mt["wshare_peak"] * 100, 0),
                     fmt(mt["I_peak"], 1), fmt(mt["vbus_min"], 1), fmt(mt["v_top"])])
    tables["Launch"] = (["Case", "t 0.5 m (s)", "v 0.5 m (m/s)", "t 8 ft (s)", "v 8 ft (m/s)",
                         "peak a (g)", "peak slip (m/s)", "peak wheel load %", "peak I (A)",
                         "min Vbus", "v at 2 s"], rows)
    f, ax = fig_page("Launch from rest (straight line)",
                     "Brushless Compact 1806 on 5S, measured dusty 20A urethane, bare bolt. Ramp = transmitter slew limit. 20 A ESC limit unless noted.", 2, 2)
    for i, (nm, R) in enumerate(runsL):
        ax[0].plot(R.t, R.speed, color=C[i], label=nm)
        ax[1].plot(R.t, R.ax_g, color=C[i], label=nm)
        ax[2].plot(R.t, R.D[:, D_SL], color=C[i], label=nm)
        ax[3].plot(R.t, R.wshare * 100, color=C[i], label=nm)
    ax[0].set(title="Speed", xlabel="time (s)", ylabel="m/s")
    ax[1].set(title="Longitudinal acceleration", xlabel="time (s)", ylabel="g", xlim=(0, 0.8))
    ax[2].set(title="Tire slip speed (wheelspin)", xlabel="time (s)", ylabel="m/s", xlim=(0, 0.8))
    ax[3].set(title="Load on wheels (load transfer)", xlabel="time (s)", ylabel="% of weight", xlim=(0, 0.8))
    ax[0].legend(loc="lower right")
    f.tight_layout(rect=(0, 0, 1, f.rtop)); figs.append(f)

    # ---------- 2. Point turn ----------
    P_cases = [("5S, step", base), ("5S, 0.15 s ramp", dict(base, slew=1 / 0.15)),
               ("4S, step (reference)", bl)]
    rows = []; runsP = []
    for nm, d in P_cases:
        R, mt = point_turn(d); runsP.append((nm, R))
        rows.append([nm, fmt(mt["t_90"], 3), fmt(mt["t_180"], 3), fmt(mt["t_360"], 3),
                     fmt(mt["r_peak"], 0), fmt(mt["icr_x"], 0), fmt(mt["drift"], 0)])
    tables["Point turn"] = (["Case", "t 90 deg (s)", "t 180 deg (s)", "t 360 deg (s)",
                             "peak yaw rate (deg/s)", "pivot x from axle (mm)", "COM drift (mm)"], rows)
    f, ax = fig_page("Point turn (full opposite throttle)",
                     "Pivot = instantaneous centre of rotation. Axle at 0 mm, bolt at 100 mm, COM at 65 mm.", 2, 2)
    for i, (nm, R) in enumerate(runsP):
        ax[0].plot(R.t, np.degrees(R.psi), color=C[i], label=nm)
        ax[1].plot(R.t, np.degrees(R.r), color=C[i], label=nm)
        ax[2].plot(R.t, R.D[:, D_ICRX] * 1000, color=C[i], label=nm)
    for yv, lab in ((90, "90"), (180, "180")):
        ax[0].axhline(yv, color=INK2, lw=0.8, ls="--")
    ax[0].set(title="Heading", xlabel="time (s)", ylabel="deg", ylim=(0, 400))
    ax[1].set(title="Yaw rate", xlabel="time (s)", ylabel="deg/s")
    ax[2].set(title="Pivot point location", xlabel="time (s)", ylabel="mm ahead of axle", ylim=(-20, 140))
    ax[2].axhline(65, color=C[7], lw=0.8, ls="--"); ax[2].text(0.9, 68, "COM", color=INK2, fontsize=8)
    R = runsP[0][1]
    xc = (1 - base["wheel_share"]) * base["L"]
    for lab, (bx, by), col in (("left wheel", (-xc, 0.09), C[0]), ("right wheel", (-xc, -0.09), C[1]),
                                ("bolt", (base["L"] - xc, 0.0), C[2])):
        X = R.x + np.cos(R.psi) * bx - np.sin(R.psi) * by
        Y = R.y + np.sin(R.psi) * bx + np.cos(R.psi) * by
        sel = R.t < 0.5
        ax[3].plot(X[sel] * 1000, Y[sel] * 1000, color=col, label=lab)
    ax[3].plot(R.x * 1000, R.y * 1000, color=INK, lw=1.2, label="COM")
    ax[3].set(title="Contact paths, first 0.5 s (5S step)", xlabel="mm", ylabel="mm")
    ax[3].set_aspect("equal"); ax[3].legend(); ax[0].legend()
    f.tight_layout(rect=(0, 0, 1, f.rtop)); figs.append(f)

    # ---------- 3. Step steer from cruise ----------
    S_amts = [0.1, 0.2, 0.3, 0.5, 0.8]
    rows = []; runsS = []; runsS_lift = []
    for k, s_amt in enumerate(S_amts):
        R, mt = step_steer(base, s_amt); runsS.append((s_amt, R))
        R2, mt2 = step_steer(base, s_amt, thr_turn=0.4); runsS_lift.append((s_amt, R2))
        for lab, m_ in (("hold 0.8", mt), ("lift to 0.4", mt2)):
            rows.append([f"{s_amt:.1f}", lab, fmt(m_["v_cruise"]), fmt(m_["dchi_05"], 0), fmt(m_["dpsi_05"], 0),
                         fmt(m_["r_peak"], 0), fmt(m_["v_05"]), fmt(m_["ay_peak"]), fmt(m_["beta"], 1),
                         fmt(m_["icr_x"], 0)])
    tables["Turning while moving"] = (["steer", "throttle in turn", "cruise v (m/s)", "PATH turn +0.5 s (deg)",
                                       "body rotation +0.5 s (deg)", "peak yaw rate (deg/s)", "v at +0.5 s",
                                       "peak lat g", "sideslip at COM (deg)", "pivot x (mm)"], rows)
    f, ax = fig_page("Turning while moving: step steer from cruise",
                     "Cruise at 80% throttle, then arcade-mix steer step at t = 1.2 s. Paths start at the steer input.", 2, 2)
    for k, (s_amt, R) in enumerate(runsS):
        sel = R.t >= 1.2
        x0, y0 = R.x[sel][0], R.y[sel][0]
        ax[0].plot((R.x[sel] - x0) * 1000, (R.y[sel] - y0) * 1000, color=C[k], label=f"steer {s_amt}")
        ax[2].plot(R.t[sel] - 1.2, np.degrees(R.r[sel]), color=C[k], label=f"steer {s_amt}")
    for k, (s_amt, R) in enumerate(runsS_lift):
        sel = R.t >= 1.2
        x0, y0 = R.x[sel][0], R.y[sel][0]
        ax[1].plot((R.x[sel] - x0) * 1000, (R.y[sel] - y0) * 1000, color=C[k], label=f"steer {s_amt}")
        ax[3].plot(R.t[sel] - 1.2, np.degrees(R.r[sel]), color=C[k], label=f"steer {s_amt}")
    for a in ax[:2]:
        a.set_aspect("equal"); a.set(xlabel="forward (mm)", ylabel="left (mm)")
    ax[0].set_title("COM path, throttle held at 0.8"); ax[1].set_title("COM path, throttle lifted to 0.4")
    ax[2].set(title="Yaw rate, throttle held", xlabel="time after steer (s)", ylabel="deg/s")
    ax[3].set(title="Yaw rate, throttle lifted", xlabel="time after steer (s)", ylabel="deg/s")
    ax[0].legend(loc="upper left")
    f.tight_layout(rect=(0, 0, 1, f.rtop)); figs.append(f)

    # ---------- 4. Ramp steer handling diagram ----------
    f, ax = fig_page("Steady-state handling: slow ramp steer",
                     "Steer ratio = achieved path curvature / kinematic curvature from wheel speeds. 1.0 = tracks exactly; <1 = understeer.", 1, 2, size=(11, 5.5))
    rows = []
    for k, thr in enumerate([0.5, 0.8, 1.0]):
        R, mt, aux = ramp_steer(base, thr0=thr)
        m_ = np.isfinite(aux["ratio"])
        ax[0].plot(aux["ay"][m_], aux["ratio"][m_], color=C[k], label=f"throttle {thr}")
        ax[1].plot(aux["s"], aux["ay"], color=C[k], label=f"throttle {thr}")
        rows.append([f"{thr}", fmt(mt["ay_max"]), fmt(mt["ratio_low"])])
    tables["Handling"] = (["Throttle", "max lateral g", "steer ratio at 0.05-0.25 g"], rows)
    ax[0].set(xlabel="lateral acceleration (g)", ylabel="steer ratio", ylim=(0, 1.2), title="Handling diagram")
    ax[1].set(xlabel="steer input", ylabel="lateral acceleration (g)", title="Lateral g vs steer input")
    ax[0].legend(); f.tight_layout(rect=(0, 0, 1, f.rtop)); figs.append(f)

    # ---------- 5. Braking ----------
    rows = []; runsB = []
    for k, mode in enumerate(["brake", "coast", "reverse"]):
        R, mt = braking(base, mode); runsB.append((mode, R))
        rows.append([mode, fmt(mt["v0"]), fmt(mt["t_stop"]), fmt(mt["d_stop"], 0), fmt(mt["decel_peak"]),
                     fmt(mt["nose_peak"] * 100, 0), fmt(mt["wshare_min"] * 100, 0),
                     "nose lift" if mt["lift"] == 1 else ("wheel lift" if mt["lift"] == 2 else "none")])
    tables["Braking"] = (["Mode", "v0 (m/s)", "time to stop (s)", "stop dist (mm)", "peak decel (g)",
                          "peak nose load %", "min wheel load %", "lift event"], rows)
    f, ax = fig_page("Braking from full speed", "Throttle released (drag brake / coast) or full reverse at t = 1.5 s.", 1, 2, size=(11, 5.5))
    for k, (mode, R) in enumerate(runsB):
        sel = R.t > 1.2
        ax[0].plot(R.t[sel] - 1.5, R.vbx[sel], color=C[k], label=mode)
        ax[1].plot(R.t[sel] - 1.5, R.wshare[sel] * 100, color=C[k], label=mode)
    ax[0].set(title="Forward speed", xlabel="time after input (s)", ylabel="m/s")
    ax[1].set(title="Load on wheels", xlabel="time after input (s)", ylabel="% of weight")
    ax[0].legend(); f.tight_layout(rect=(0, 0, 1, f.rtop)); figs.append(f)

    # ---------- 6. Weapon spin-up reaction ----------
    rows = []
    f, ax = fig_page("Weapon spin-up reaction (bot at rest)",
                     f"Blade H = {H_BLADE} N.m.s. Current-limited spin-up: constant reaction torque H / t_spin until the blade reaches speed.", 1, 2, size=(11, 5.5))
    for k, t95 in enumerate([0.75, 0.85, 1.0, 1.5, 3.0]):
        for bk, ls in ((True, "-"), (False, "--")):
            R, mt = spinup(base, t95, brake=bk)
            ax[0 if bk else 1].plot(R.t, np.degrees(R.psi), color=C[k], ls="-", label=f"spin-up {t95} s")
            rows.append([fmt(t95, 2), "drag brake" if bk else "coast", fmt(mt["yaw_final"], 1),
                         fmt(mt["r_peak"], 0), fmt(mt["drift"], 0)])
    tables["Weapon spin-up"] = (["spin-up time (s)", "ESC", "final yaw (deg)", "peak yaw rate (deg/s)", "COM drift (mm)"], rows)
    ax[0].set(title="Heading, drive drag brake ON", xlabel="time (s)", ylabel="deg")
    ax[1].set(title="Heading, drive COAST", xlabel="time (s)", ylabel="deg")
    ax[0].legend(); f.tight_layout(rect=(0, 0, 1, f.rtop)); figs.append(f)

    # ---------- 7. Configuration matrix ----------
    configs = [("Baseline: 35/65, bare bolt", base_params()),
               ("PTFE bolt (mu 0.10)", base_params(mubk=0.10, mubs=0.12)),
               ("Rear pins (40 g, CAD 38.5%)", base_params(pins=True, wheel_share=0.385)),
               ("Pins + PTFE", base_params(pins=True, wheel_share=0.385, mubk=0.10, mubs=0.12)),
               ("Worst dust (tire 1.4/1.7)", base_params(muk=1.4, mus=1.7)),
               ("4S instead of 5S", base_params(Vb=14.8))]
    rows = []; cmat = []
    for nm, d in configs:
        _, mL = launch(dict(d, slew=1 / 0.15))
        _, mP = point_turn(d)
        _, mS = step_steer(d, 0.5)
        _, mH, _ = ramp_steer(d, 0.8)
        _, mD = duty_cycle(d, reps=6)
        cmat.append((nm, mL["v_05"], 1 / mP["t_90"], mS["dchi_05"], mH["ay_max"]))
        rows.append([nm, fmt(d["wheel_share"] * 100, 1), fmt(mL["v_05"]), fmt(mL["t_8ft"]), fmt(mP["t_90"], 3),
                     fmt(mS["dchi_05"], 0), fmt(mH["ay_max"]), fmt(mD["Wh_fight"], 2), fmt(mD["I_peak"], 1)])
    tables["Configurations"] = (["Config", "wheel %", "v 0.5 m (ramp)", "t 8 ft (s)", "t 90 deg (s)",
                                 "path turn +0.5 s, steer 0.5", "max lat g", "drive Wh / 3 min", "peak batt I (A)"], rows)
    f, ax = fig_page("Configuration comparison (normalized to baseline)",
                     "Bars > 1.0 are better than baseline. Turn speed = 1 / time-to-90.", 1, 1, size=(11, 5.5))
    a = ax[0]
    labels = ["charge speed @0.5 m", "point-turn speed", "path turn at speed", "max lateral g"]
    nb = len(cmat); w = 0.8 / nb
    basev = np.array(cmat[0][1:])
    for k, row in enumerate(cmat):
        vals = np.array(row[1:]) / basev
        xs = np.arange(4) + (k - (nb - 1) / 2) * w
        a.bar(xs, vals, width=w * 0.9, color=C[k], label=row[0], edgecolor=SURF, linewidth=2)
    a.axhline(1.0, color=INK2, lw=1)
    a.set_xticks(np.arange(4)); a.set_xticklabels(labels); a.set_ylabel("relative to baseline")
    a.legend(ncol=3, loc="upper left"); a.set_ylim(0, 1.6)
    f.tight_layout(rect=(0, 0, 1, f.rtop)); figs.append(f)

    # ---------- 8. Sensitivity (tornado) ----------
    def metrics(d):
        _, mL = launch(dict(d, slew=1 / 0.15), t_end=1.0)
        _, mP = point_turn(d, t_end=0.5)
        _, mS = step_steer(d, 0.5, t_turn=0.55)
        return np.array([mL["v_05"], mP["t_90"], mS["dchi_05"]])
    b0 = metrics(base)
    sens = [("Tire kinetic mu 1.4 / 1.8", dict(muk=1.4), dict(muk=1.8)),
            ("Tire static mu 1.8 / 2.2", dict(mus=1.8), dict(mus=2.2)),
            ("Tire peak slip 0.06 / 0.25", dict(sigp=0.06), dict(sigp=0.25)),
            ("Bolt mu 0.15 / 0.25", dict(mubk=0.15, mubs=0.18), dict(mubk=0.25, mubs=0.30)),
            ("Wheel share 33% / 38.5%", dict(wheel_share=0.33), dict(wheel_share=0.385)),
            ("COM height 18 / 28 mm", dict(h=0.018), dict(h=0.028)),
            ("Yaw inertia x0.7 / x1.3", dict(Izz=base["Izz"] * 0.7), dict(Izz=base["Izz"] * 1.3)),
            ("Battery 17.0 / 21.0 V", dict(Vb=17.0), dict(Vb=21.0)),
            ("Rolling resistance 0 / 0.04", dict(Crr=0.0), dict(Crr=0.04)),
            ("Load-transfer lag 5 / 25 ms", dict(tau=0.005), dict(tau=0.025))]
    srows = []; sres = []
    for nm, lo, hi in sens:
        ml_ = metrics(dict(base, **lo)); mh_ = metrics(dict(base, **hi))
        sres.append((nm, ml_, mh_))
        srows.append([nm, fmt(ml_[0]), fmt(mh_[0]), fmt(ml_[1], 3), fmt(mh_[1], 3), fmt(ml_[2], 0), fmt(mh_[2], 0)])
    tables["Sensitivity"] = (["Parameter (low / high)", "v0.5m lo", "v0.5m hi", "t90 lo", "t90 hi",
                              "path turn lo", "path turn hi"], srows)
    f, ax = fig_page("Sensitivity: one-at-a-time swings around baseline",
                     f"Baseline: v@0.5 m {b0[0]:.2f} m/s (0.15 s ramp) | t90 {b0[1]:.3f} s | path turn {b0[2]:.0f} deg in 0.5 s (steer 0.5)", 1, 3, size=(11, 6))
    titles = ["Charge speed at 0.5 m (m/s)", "Time to 90 deg (s)", "Path turn at speed in 0.5 s (deg)"]
    for mi in range(3):
        order = sorted(range(len(sres)), key=lambda i: abs(sres[i][2][mi] - sres[i][1][mi]))
        a = ax[mi]
        for yy, i in enumerate(order):
            lo_, hi_ = sres[i][1][mi] - b0[mi], sres[i][2][mi] - b0[mi]
            a.barh(yy, lo_, color=C[0], height=0.6, edgecolor=SURF)
            a.barh(yy, hi_, color=C[1], height=0.6, edgecolor=SURF)
        a.set_yticks(range(len(order)))
        a.set_yticklabels([sres[i][0] for i in order] if mi == 0 else [""] * len(order), fontsize=7.5)
        a.axvline(0, color=INK, lw=1); a.set_title(titles[mi]); a.set_xlabel("change from baseline")
    from matplotlib.patches import Patch
    ax[0].legend(handles=[Patch(color=C[0], label="low value"), Patch(color=C[1], label="high value")],
                 loc="lower right")
    f.tight_layout(rect=(0, 0, 1, f.rtop)); figs.append(f)

    # ---------- 9. Monte Carlo ----------
    rng = np.random.default_rng(7)
    NMC = 250
    mc = []
    for _ in range(NMC):
        dd = dict(base,
                  muk=rng.uniform(1.4, 1.8), mus=rng.uniform(1.8, 2.2), sigp=rng.uniform(0.06, 0.25),
                  mubk=rng.uniform(0.15, 0.25), wheel_share=rng.uniform(0.33, 0.385),
                  h=rng.uniform(0.018, 0.028), Izz=base["Izz"] * rng.uniform(0.7, 1.3),
                  Vb=rng.uniform(17.0, 21.0), Crr=rng.uniform(0.0, 0.04))
        dd["mubs"] = dd["mubk"] * 1.2
        dd["mus"] = max(dd["mus"], dd["muk"] + 0.05)
        mc.append(metrics(dd))
    mc = np.array(mc)
    rows = []
    for mi, nm in enumerate(["Charge speed at 0.5 m (m/s)", "Time to 90 deg (s)", "Path turn 0.5 s (deg)"]):
        q = np.percentile(mc[:, mi], [5, 50, 95])
        rows.append([nm, fmt(q[0], 3), fmt(q[1], 3), fmt(q[2], 3), fmt(b0[mi], 3)])
    tables["Monte Carlo"] = (["Metric", "P5", "P50", "P95", "Baseline"], rows)
    f, ax = fig_page(f"Monte Carlo: {NMC} runs, all uncertain inputs sampled uniformly",
                     "Ranges: tire mu_k 1.4-1.8, mu_s 1.8-2.2, peak slip 0.06-0.25, bolt 0.15-0.25, wheel share 33-38.5%, "
                     "h 18-28 mm, Izz +/-30%, 17-21 V, Crr 0-0.04", 1, 3, size=(11, 5))
    for mi in range(3):
        a = ax[mi]
        a.hist(mc[:, mi], bins=25, color=C[0], edgecolor=SURF, linewidth=1.5)
        for qv, ls in zip(np.percentile(mc[:, mi], [5, 50, 95]), (":", "-", ":")):
            a.axvline(qv, color=INK, ls=ls, lw=1.2)
        a.set_title(titles[mi]); a.set_ylabel("runs")
    f.tight_layout(rect=(0, 0, 1, f.rtop)); figs.append(f)

    # ---------- 10. Validation ----------
    vrows = []
    led_cases = [("Launch 5S step", runsL[0][1]), ("Launch 4S", runsL[3][1]),
                 ("Point turn", runsP[0][1]), ("Step steer 0.5", runsS[3][1]),
                 ("Brake from cruise", runsB[0][1]), ("Reverse from cruise", runsB[2][1])]
    Rsp, _ = spinup(base, 1.0); led_cases.append(("Weapon spin-up", Rsp))
    Rdc, mdc = duty_cycle(base); led_cases.append(("30 s duty cycle", Rdc))
    leds = []
    for nm, R in led_cases:
        lg = R.ledger(); leds.append((nm, lg))
        vrows.append([nm, fmt(lg["E_in"], 2), fmt(lg["dKE"], 3), fmt(lg["cu"], 2), fmt(lg["gear"], 2),
                      fmt(lg["tire"], 2), fmt(lg["bolt"], 2), fmt(lg["rr"], 3), f"{abs(lg['rel'])*100:.1e}%"])
    tables["Energy ledger"] = (["Scenario", "E in (J)", "dKE (J)", "copper", "gearbox", "tire slip",
                                "bolt", "rolling", "closure error"], vrows)
    # analytic check 1: wheelspin-phase acceleration
    R = runsL[0][1]
    sel = (R.t > 0.10) & (R.t < 0.25)
    a_sim = np.mean(R.ax_g[sel])
    mu_eff = np.mean(R.D[sel, D_MUL])
    xc = (1 - base["wheel_share"]) * base["L"]
    mub = base["mubk"]; L = base["L"]; h = base["h"]
    a_an = (mu_eff - (mu_eff + mub) * xc / L) / (1 - (mu_eff + mub) * h / L)
    # analytic check 2: top speed balance (brushed)
    mo = base["motor"]; W = base["m"] * 9.81
    NB = W * xc / L
    Tload = (0.5 * mub * NB + base["Crr"] * 0.5 * (W - NB)) * base["rw"]
    I = Tload / (mo["eta"] * mo["G"] * mo["Kt"])
    Vterm = base["Vb"] - base["Rbat"] * 2 * I
    w = (Vterm - I * mo["R"]) / (mo["Kt"] * mo["G"])
    v_an = w * base["rw"]
    Rtop = run(base, 4.0, lambda t: (np.ones_like(t), np.ones_like(t)))
    v_sim = Rtop.wsurf[-1]
    # analytic check 3: dt convergence
    dfine = launch(base)[1]; pfine = point_turn(base)[1]
    global DT
    DT_save = DT; DT = 5e-5
    dfine2 = launch(base)[1]; pfine2 = point_turn(base)[1]
    DT = DT_save
    chk = [["Wheelspin-phase accel (g)", fmt(a_an, 3), fmt(a_sim, 3), f"{(a_sim/a_an-1)*100:.2f}%"],
           ["Top wheel-surface speed, 5S (m/s)", fmt(v_an, 3), fmt(v_sim, 3), f"{(v_sim/v_an-1)*100:.2f}%"],
           ["dt halving: v at 0.5 m", fmt(dfine["v_05"], 4), fmt(dfine2["v_05"], 4), f"{(dfine2['v_05']/dfine['v_05']-1)*100:.3f}%"],
           ["dt halving: t 90 deg", fmt(pfine["t_90"], 4), fmt(pfine2["t_90"], 4), f"{(pfine2['t_90']/pfine['t_90']-1)*100:.3f}%"]]
    tables["Analytic checks"] = (["Check", "Analytic / dt=1e-4", "Simulated / dt=5e-5", "Difference"], chk)
    tables["Duty cycle"] = (["Metric", "Value"],
                            [["Drive energy, 30 s segment (Wh)", fmt(mdc["Wh_30s"], 3)],
                             ["Drive energy, 3 min fight (Wh)", fmt(mdc["Wh_fight"], 2)],
                             [f"Share of {PACK_WH:.1f} Wh pack (5S 450 HV, assumed)", f"{mdc['Wh_fight']/PACK_WH*100:.1f}%"],
                             ["Avg battery current (A)", fmt(mdc["I_avg"], 2)],
                             ["Peak battery current (A)", fmt(mdc["I_peak"], 1)],
                             ["Energy lost to tire slip (J/30 s)", fmt(mdc["tire_J"], 1)],
                             ["Energy lost to bolt drag (J/30 s)", fmt(mdc["bolt_J"], 1)],
                             ["Copper loss (J/30 s)", fmt(mdc["cu_J"], 1)]])
    f, ax = fig_page("Validation: energy ledger closure",
                     "Integrator check: terminal energy = dKE + all modeled losses. Validates integration and kinematics, not the friction or motor laws themselves.",
                     1, 1, size=(11, 5.5))
    a = ax[0]
    parts = [("dKE", "dKE"), ("copper", "cu"), ("gearbox", "gear"), ("tire slip", "tire"),
             ("bolt", "bolt"), ("rolling", "rr")]
    bottom = np.zeros(len(leds))
    for k, (lab, key) in enumerate(parts):
        vals = np.array([max(lg[key], 0) / max(lg["E_in"], 1e-9) for _, lg in leds])
        a.bar(range(len(leds)), vals, bottom=bottom, color=C[k], label=lab, edgecolor=SURF, linewidth=2)
        bottom += vals
    a.set_xticks(range(len(leds))); a.set_xticklabels([n for n, _ in leds], fontsize=8)
    a.set_ylabel("fraction of input energy"); a.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    for i, (_, lg) in enumerate(leds):
        a.text(i, 1.02, f"residual {abs(lg['rel'])*100:.1e}%", ha="center", fontsize=7.5, color=INK2)
    a.set_ylim(0, 1.15)
    f.tight_layout(rect=(0, 0, 1, f.rtop)); figs.append(f)

    # ---------- write outputs ----------
    order = ["Launch", "Point turn", "Turning while moving", "Handling", "Braking", "Weapon spin-up",
             "Configurations", "Sensitivity", "Monte Carlo", "Duty cycle", "Energy ledger", "Analytic checks"]
    with PdfPages(os.path.join(OUT, "flapjack_drive_report.pdf")) as pdf:
        for i, fg in enumerate(figs):
            pdf.savefig(fg); fg.savefig(os.path.join(OUT, f"fig{i:02d}.png"))
        for nm in order:
            hdr, rws = tables[nm]
            fg = table_page(f"Table: {nm}", hdr, rws)
            pdf.savefig(fg); plt.close(fg)
    with open(os.path.join(OUT, "results.md"), "w") as fh:
        for nm in order:
            hdr, rws = tables[nm]
            fh.write(f"\n## {nm}\n\n| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
            for r_ in rws: fh.write("| " + " | ".join(r_) + " |\n")
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as fh:
        wtr = csv.writer(fh)
        for nm in order:
            hdr, rws = tables[nm]
            wtr.writerow([f"# {nm}"]); wtr.writerow(hdr); wtr.writerows(rws); wtr.writerow([])
    print(f"done in {time.time()-t0:.1f} s -> {OUT}/")


if False and __name__ == "__main__":
    main()
