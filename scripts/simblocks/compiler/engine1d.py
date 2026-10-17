"""1D engine: lumped bodies with one generalized coordinate each (translation or rotation), joined by
elements that act on a LINEAR MEASURE of the coordinates, s = sum_i c_i q_i + s0.

An element returns a force F along its measure (F > 0 resists s increasing). Its generalized force on body i
is Q_i = -c_i F. This one rule applies every reaction (equal and opposite, and lever arms through c_i), so a
compiled sim cannot forget a reaction. Semi-implicit Euler: v += dt Q / m, then q += dt v.

Energy bookkeeping, per element: absorbed work A = sum F * (s_new - s_old). Global check: KE change + sum A = 0
(integration). Element check: A = stored change + dissipated - input, each from the element's own law
(springs, elastoplastic, Hertz, the motor/battery drive). Both residuals go into _ledger_rel.

Element type codes and their float parameters p[0..9]:
  0 spring        k
  1 dashpot       c
  2 kelvin_voigt  k, c                       (one_sided: no pull, force clipped at 0)
  3 elastoplastic table (k, acc_tab, F_tab)  (one_sided; p[0] = k, p[1] = break_at plastic set)
  4 hertz         K, n, alpha                (F = K s^n (1 + alpha sd), compression only)
  5 friction      F_c, v_eps, F_s, v_s       (regularized Coulomb + Stribeck bump)
  6 drag          K                          (F = K sd |sd|)
  7 force         P                          (constant push of P along +measure: F = -P)
  8 drive         Kt, R, I_max, G, cells, R0, R1, C1, Q, throttle  (DC motor + ESC + LiPo 1-RC pack)
aux per element (4 values): elastoplastic [xp, acc, broken, -]; drive [I_motor, I_batt, v_bus, soc]."""
import math
import numpy as np
from numba import njit

from .circuit import circuit_step
from .control import control_step
from .adaptive import pick_dt

from ..constitutive.plasticity import tabular_return
from ..actuators.motors import esc_supply, esc_supply_thr, lipo_ocv_k

NP = 12      # floats per element
NAUX = 4


@njit(cache=True, nogil=True)
def _force(typ, one_sided, p, s, sd, tab_k, tab_x, tab_F, st, aux, dt):
    """Returns (F, d_diss, d_input, stored). st: mutable state [xp, acc, broken, soc, iR1]."""
    if typ == 0:
        F = p[0] * s
        return F, 0.0, 0.0, 0.5 * p[0] * s * s
    if typ == 1:
        F = p[0] * sd
        return F, F * sd * dt, 0.0, 0.0
    if typ == 2:
        if one_sided == 1:
            if s <= 0.0:
                return 0.0, 0.0, 0.0, 0.0
            F = p[0] * s + p[1] * sd
            if F < 0.0:
                F = 0.0
            return F, -1.0, 0.0, 0.0           # -1: dissipation = absorbed - stored (no independent law)
        F = p[0] * s + p[1] * sd
        return F, p[1] * sd * sd * dt, 0.0, 0.5 * p[0] * s * s
    if typ == 3:
        if st[2] > 0.5:                         # broken (punched through / torn off): carries nothing
            return 0.0, 0.0, 0.0, 0.0
        xp0 = st[0]
        F, xp, acc = tabular_return(s, st[0], st[1], tab_k, tab_x, tab_F, one_sided)
        st[0] = xp; st[1] = acc
        dd = F * (xp - xp0)
        stored = F * F / (2.0 * tab_k)
        if abs(xp) >= p[1]:
            st[2] = 1.0                         # release: stored elastic energy is lost with the break
            dd += stored
            stored = 0.0
        aux[0] = xp; aux[1] = acc; aux[2] = st[2]
        return F, dd, 0.0, stored
    if typ == 4:
        if s <= 0.0:
            return 0.0, 0.0, 0.0, 0.0
        F = p[0] * s ** p[1] * (1.0 + p[2] * sd)
        if F < 0.0:
            F = 0.0
        if p[2] == 0.0:
            return F, 0.0, 0.0, p[0] * s ** (p[1] + 1.0) / (p[1] + 1.0)
        return F, -1.0, 0.0, 0.0
    if typ == 5:
        Fc = p[0]; ve = p[1]; Fs = p[2]; vs = p[3]
        mag = Fc
        if Fs > Fc and vs > 0.0:
            mag = Fc + (Fs - Fc) * math.exp(-(sd / vs) ** 2)
        F = mag * math.tanh(sd / ve)
        return F, F * sd * dt, 0.0, 0.0
    if typ == 6:
        F = p[0] * sd * abs(sd)
        return F, F * sd * dt, 0.0, 0.0
    if typ == 7:
        F = -p[0]
        return F, 0.0, -1.0, 0.0                 # -1: input = -absorbed (a pure source)
    if typ == 13:                                # load from a table (CSV): F(t) or F(s), held at the ends
        x = st[0] if p[1] < 0.5 else s
        nt = tab_x.shape[0]
        if x <= tab_x[0]:
            Fv = tab_F[0]
        elif x >= tab_x[nt - 1]:
            Fv = tab_F[nt - 1]
        else:
            lo = 0; hi = nt - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if tab_x[mid] <= x:
                    lo = mid
                else:
                    hi = mid
            if p[2] > 0.5:                       # step: hold the previous value
                Fv = tab_F[lo]
            else:
                Fv = tab_F[lo] + (tab_F[hi] - tab_F[lo]) * (x - tab_x[lo]) / (tab_x[hi] - tab_x[lo])
        aux[0] = Fv
        return -Fv, 0.0, -1.0, 0.0
    if typ == 8:
        Kt = p[0]; Rm = p[1]; Imax = p[2]; G = p[3]; cells = p[4]; R0 = p[5]; R1 = p[6]; C1 = p[7]; Q = p[8]
        thr = p[9]
        soc = st[3]; iR1 = st[4]
        ocv = lipo_ocv_k(soc, cells)
        Veff = ocv - R1 * iR1
        wm = G * sd
        I, Ib, v, duty = esc_supply_thr(Veff, R0, Kt * wm, Rm, Imax, thr)
        if Ib < 0.0:
            Ib = 0.0
        # below the current limit the drive is a damper on its shaft (current falls as the speed rises): st[0] = its
        # c dt / J for the exact step (motor_step), st[2] = the shaft rate this step started from
        st[0] = 0.0; st[2] = sd
        if p[10] > 0.0 and I > 0.0:
            h = 1e-6 * (abs(wm) + Veff / Kt)
            I2, Ib2, v2, d2 = esc_supply_thr(Veff, R0, Kt * (wm + h), Rm, Imax, thr)
            dIdw = (I - I2) / h
            if dIdw > 0.0:
                st[0] = p[10] * G * G * Kt * dIdw * dt
        Ech = ocv * Ib * dt
        diss = (R0 * Ib * Ib + R1 * iR1 * iR1 + Rm * I * I) * dt
        a = math.exp(-dt / (R1 * C1)) if R1 > 0.0 and C1 > 0.0 else 0.0
        st[4] = a * iR1 + (1.0 - a) * Ib
        st[3] = soc - Ib * dt / Q
        aux[0] = I; aux[1] = Ib; aux[2] = v; aux[3] = st[3]
        cap = 0.5 * C1 * (R1 * st[4]) ** 2 if C1 > 0.0 else 0.0
        return -G * Kt * I, diss, Ech, cap
    if typ == 17:                                # a circuit motor: torque set by the circuit, speed read back
        st[2] = sd
        aux[0] = st[1]; aux[1] = st[3]
        return -st[0], 0.0, -1.0, 0.0
    return 0.0, 0.0, 0.0, 0.0


@njit(cache=True, nogil=True)
def motor_k(typ, p, st, dt):
    """k = c dt / J of a motor's back-EMF damping on its shaft this step (0: none). A drive works it out in _force; a
    circuit motor's damping comes from the circuit solve (how its current falls with its back-EMF)."""
    if typ == 8:
        return st[0]
    if typ == 17:                                    # st[4]: its back-EMF damping, from the circuit (circuit_step)
        return p[10] * st[4] * dt
    return 0.0


@njit(cache=True, nogil=True)
def motor_scale(k):
    """The exact step of a shaft held by a damper: over a step of k = c dt / J its speed change is the plain step's
    times (1 - e^-k) / k, whatever else pushes on it (held steady over the step). Never overshoots, at any step."""
    if k < 1e-9:
        return 1.0
    return (1.0 - math.exp(-k)) / k


@njit(cache=True, nogil=True)
def drive_books(p, st, aux, F0, F1, k, dt, D, IN, A, dsp, e):
    """A drive's books after the exact step changed its force from F0 to F1 (the step's average): the currents
    scale with the force, the heat is the average of I^2 over the exponential decay, the pack charge and the
    R1 branch follow the average battery current."""
    A[e] += 0.5 * (F1 - F0) * dsp[e]
    if F0 == 0.0:
        return
    r = F1 / F0
    I0 = aux[0]; Ib0 = aux[1]
    R0 = p[5]; Rm = p[1]; R1 = p[6]; C1 = p[7]; Q = p[8]
    phi = motor_scale(k)
    phi2 = (1.0 - math.exp(-2.0 * k)) / (2.0 * k) if k > 1e-9 else 1.0
    Ia = I0 * r
    Iinf = (Ia - I0 * phi) / (1.0 - phi) if phi < 1.0 else Ia
    sq = Iinf * Iinf + 2.0 * Iinf * (I0 - Iinf) * phi + (I0 - Iinf) * (I0 - Iinf) * phi2
    if sq < 0.0:
        sq = Ia * Ia
    beta = Ib0 / I0 if I0 != 0.0 else 0.0
    D[e] += (R0 * beta * beta + Rm) * (sq - I0 * I0) * dt
    dIb = Ib0 * (r - 1.0)
    ocv = lipo_ocv_k(st[3] + Ib0 * dt / Q, p[4])
    IN[e] += ocv * dIb * dt
    st[3] -= dIb * dt / Q
    a = math.exp(-dt / (R1 * C1)) if R1 > 0.0 and C1 > 0.0 else 0.0
    st[4] += (1.0 - a) * dIb
    aux[3] = st[3]                             # (current outputs stay the step's starting values: the peak is real)


@njit(cache=True, nogil=True)
def run_engine(n, dt, q0, v0, m, typ, one_sided, active, P, C, s0, tab_off, tab_len, tab_k, tab_x, tab_F,
               log_every, cross_src, cross_idx, cross_thr, cross_dir,
               cpt, cna, cnb, cprm, cpst0, cth, cnn, cmot,
               ckind, cprog, cval, crng, ccp, cssrc, csidx, cae, caslot, casig, caarr, cctlD, adp, adf, adc, stop):
    """stop: int64[3]. Another thread sets stop[0] = 1 to end the run early (timeout); stop[1] returns steps done.
    ckind..cctlD: the control signals (see control.py) and the element fields they drive.
    adp, adf, adc: the adaptive step (see adaptive.py); adp[0] = 0 runs n steps of dt."""
    nb = q0.shape[0]; ne = typ.shape[0]
    q = q0.copy(); v = v0.copy()
    P = P.copy(); cprm = cprm.copy()
    nsig = ckind.shape[0]; nsen = cssrc.shape[0]
    sig = np.zeros(nsig); cst = np.zeros((nsig, 4)); sens = np.zeros(nsen); cstack = np.zeros(64)
    SG = np.empty((nsig, 2))
    for k in range(nsig):
        SG[k, 0] = 1e300; SG[k, 1] = -1e300
    Q = np.zeros(nb)
    st = np.zeros((ne, 5))
    # electrical circuit (optional): parts solved each step, motors read their speed and set their torque
    npart = cpt.shape[0]
    cpst = cpst0.copy(); cpout = np.zeros((npart, 8)); cflg = np.zeros(npart)
    cG = np.zeros((cnn, cnn)); cI = np.zeros(cnn); cvn = np.zeros(cnn + 1)
    CTR = np.empty((npart, 16))
    for k in range(npart):
        for c in range(8):
            CTR[k, 2 * c] = 1e300; CTR[k, 2 * c + 1] = -1e300
    for e in range(ne):
        if typ[e] == 8:
            st[e, 3] = 1.0                     # pack starts full
    aux = np.zeros((ne, NAUX))
    A = np.zeros(ne); D = np.zeros(ne); IN = np.zeros(ne); S0 = np.zeros(ne); S1 = np.zeros(ne)
    Fe = np.zeros(ne); se = np.zeros(ne)
    Fp = np.zeros(ne); dsp = np.zeros(ne)      # previous step's force and stretch increment (trapezoidal work)
    # trackers: [Fmin, Fmax, smin, smax, aux0min, aux0max, aux1min, aux1max, aux2min, aux2max, aux3min, aux3max]
    TR = np.empty((ne, 12))
    for e in range(ne):
        for k in range(6):
            TR[e, 2 * k] = 1e300; TR[e, 2 * k + 1] = -1e300
    BT = np.empty((nb, 4))                     # body [qmin, qmax, vmin, vmax]
    for b in range(nb):
        BT[b, 0] = q[b]; BT[b, 1] = q[b]; BT[b, 2] = v[b]; BT[b, 3] = v[b]
    ncr = cross_src.shape[0]
    t_cross = -np.ones(ncr)
    adapt = adp[0] > 0.5
    t = 0.0; tlog = 0.0; t_stop = adp[5] * (1.0 - 1e-13)
    adsp = np.zeros(ne); inc1 = np.zeros(ne, np.int64)
    nlog = n // log_every + 2 if log_every > 0 else 1      # no history: no log arrays (was n rows of zeros)
    if adapt and log_every > 0:
        nlog = int(adp[5] / adp[7]) + 3
    LOGC = np.zeros((nlog if npart > 0 else 1, npart * 8 if npart > 0 else 1))
    LOGB = np.zeros((nlog, 2 * nb)); LOGE = np.zeros((nlog, ne * (2 + NAUX))); LOGT = np.zeros(nlog); j = 0
    LOGS = np.zeros((nlog if nsig > 0 else 1, nsig if nsig > 0 else 1))
    KE0 = 0.0
    for b in range(nb):
        KE0 += 0.5 * m[b] * v[b] * v[b]
    first = True
    done = n
    KEpk = KE0
    jacc = np.zeros(ne); jcnt = 0          # rigid impacts: force logged as the mean over each log interval
    Dq = np.zeros(nb); nimp = 0
    for e in range(ne):
        if typ[e] == 14:
            nimp += 1
    cand = np.zeros(ne, np.int64); tauv = np.zeros(ne); tgt = np.zeros(ne); jn = np.zeros(ne); invn = np.zeros(ne)
    vst = np.zeros(nb); ucd = np.zeros(ne); u0d = np.zeros(ne); lst = np.zeros(ne); held = np.zeros(ne)
    for i in range(n):
        if (i & 4095) == 0 and stop[0] != 0:
            done = i
            break
        if adapt:
            if t >= t_stop:
                done = i; stop[2] = 2               # reached t_end
                break
            dt = pick_dt(i, t, dt, adp, adf, adc, active, se, adsp, inc1, False, sig, casig)
            tn = t + dt
        else:
            t = i * dt; tn = (i + 1) * dt
        for e in range(ne):
            if typ[e] == 13:
                st[e, 0] = t                # load tables against time read the clock from their state
        for b in range(nb):
            Q[b] = 0.0
        for e in range(ne):
            s = s0[e]; sd = 0.0
            for b in range(nb):
                s += C[e, b] * q[b]; sd += C[e, b] * v[b]
            if active[e] == 0 or typ[e] == 14:
                if typ[e] != 14:
                    Fe[e] = 0.0
                se[e] = s
                continue
            o = tab_off[e]; L = tab_len[e]
            F, dd, din, stored = _force(typ[e], one_sided[e], P[e], s, sd, tab_k[e], tab_x[o:o + L], tab_F[o:o + L],
                                        st[e], aux[e], dt)
            if first:
                S0[e] = stored
            S1[e] = stored
            if dd >= 0.0:
                D[e] += dd
            else:
                D[e] = -1.0                      # flag: no independent dissipation law
            if din >= 0.0:
                IN[e] += din
            else:
                IN[e] = -1.0                     # flag: pure source
            A[e] += 0.5 * (F - Fp[e]) * dsp[e]        # completes the previous interval as a trapezoid
            Fe[e] = F; se[e] = s
            for b in range(nb):
                Q[b] -= C[e, b] * F
        first = False
        for b in range(nb):
            vst[b] = v[b]
            v[b] += dt * Q[b] / m[b]
        # motors: the exact step of a shaft held by back-EMF (stable at any step; see motor_scale)
        for e in range(ne):
            if (typ[e] == 8 or typ[e] == 17) and active[e] != 0 and P[e, 10] > 0.0:
                kk = motor_k(typ[e], P[e], st[e], dt)
                if kk > 1e-9:
                    du = 0.0
                    for b in range(nb):
                        du += C[e, b] * (v[b] - vst[b])
                    J_ = du * (motor_scale(kk) - 1.0) / P[e, 10]
                    for b in range(nb):
                        if C[e, b] != 0.0:
                            v[b] += C[e, b] * J_ / m[b]
                    F0 = Fe[e]; Fe[e] = F0 - J_ / dt
                    if typ[e] == 8:
                        drive_books(P[e], st[e], aux[e], F0, Fe[e], kk, dt, D, IN, A, dsp, e)
                    else:
                        A[e] += 0.5 * (Fe[e] - F0) * dsp[e]
        # rigid impacts (sequential impulses; see engine2d)
        for b in range(nb):
            Dq[b] = 0.0
        if nimp > 0:
            nc = 0
            for e in range(ne):
                if typ[e] != 14:
                    continue
                held[e] = jn[e]; Fe[e] = 0.0; aux[e, 0] = 0.0; jn[e] = 0.0
                if active[e] == 0:
                    continue
                s = s0[e]; sd = 0.0; w_ = 0.0
                for b in range(nb):
                    s += C[e, b] * q[b]; sd += C[e, b] * v[b]; w_ += C[e, b] * C[e, b] / m[b]
                if sd <= 0.0 or s + sd * dt <= 0.0 or w_ <= 0.0:
                    continue
                cand[nc] = e; nc += 1
                tauv[e] = min(1.0, -s / (sd * dt)) if s < 0.0 else 0.0
                tgt[e] = -P[e, 0] * sd if sd > P[e, 2] else 0.0
                jn[e] = 0.0; invn[e] = w_
                aux[e, 3] += 0.5 * sd * sd / w_
                ucd[e] = sd; lst[e] = D[e]; u0d[e] = 0.0
                for b in range(nb):
                    u0d[e] += C[e, b] * vst[b]
            for it in range(16):
                for c in range(nc):
                    e = cand[c]
                    u = 0.0
                    for b in range(nb):
                        u += C[e, b] * v[b]
                    new = jn[e] + (u - tgt[e]) / invn[e]
                    if new < 0.0:
                        new = 0.0
                    dj = new - jn[e]; jn[e] = new
                    if dj != 0.0:
                        for b in range(nb):
                            if C[e, b] != 0.0:
                                dv = -C[e, b] * dj / m[b]
                                v[b] += dv; Dq[b] -= dv * tauv[e] * dt
                        u1 = u - invn[e] * dj
                        loss = 0.5 * dj * (u + u1)
                        D[e] += loss; A[e] += loss
            for c in range(nc):
                e = cand[c]
                Fe[e] = jn[e] / dt; aux[e, 0] = jn[e]
                if held[e] > 0.0:                                 # held contact: books no heat
                    A[e] -= D[e] - lst[e]; D[e] = lst[e]
                    continue
                u0 = u0d[e] if u0d[e] > 0.0 else 0.0              # book only the approach speed it came in with
                corr = 0.5 * (ucd[e] * ucd[e] - u0 * u0) / invn[e]
                if corr > D[e] - lst[e]:
                    corr = D[e] - lst[e]
                if corr > 0.0:
                    D[e] -= corr; A[e] -= corr
        for e in range(ne):
            if typ[e] == 14:
                aux[e, 1] = D[e]
                continue
            ds = 0.0
            for b in range(nb):
                ds += C[e, b] * (v[b] * dt + Dq[b])
            A[e] += Fe[e] * ds
            Fp[e] = Fe[e]; dsp[e] = ds
        for b in range(nb):
            q[b] += dt * v[b] + Dq[b]
        if nsig > 0:                             # control: read the state this step ended in, work out the signals,
                                                 # set the fields the next step uses
            for k in range(nsen):
                src = cssrc[k]; kk = csidx[k]
                if src == 0:
                    sens[k] = q[kk]
                elif src == 1:
                    sens[k] = v[kk]
                elif src == 2:
                    sens[k] = Fe[kk]
                elif src == 3:
                    s_ = s0[kk]
                    for b in range(nb):
                        s_ += C[kk, b] * q[b]
                    sens[k] = s_
                else:
                    sens[k] = aux[kk, src - 4]
            control_step(tn, dt, i == 0, ckind, cprog, cval, crng, ccp, cst, sig, sens, cstack)
            for a_ in range(cae.shape[0]):
                x_ = sig[casig[a_]]
                if caarr[a_] >= 2:                   # the field is stored as its inverse (a motor's kv as 1/kv)
                    x_ = 1.0 / x_ if x_ != 0.0 else 1e300
                if caarr[a_] == 0 or caarr[a_] == 2:
                    P[cae[a_], caslot[a_]] = x_
                else:
                    cprm[cae[a_], caslot[a_]] = x_
            for k in range(nsig):
                if sig[k] < SG[k, 0]: SG[k, 0] = sig[k]
                if sig[k] > SG[k, 1]: SG[k, 1] = sig[k]
        # trackers
        for e in range(ne):
            if Fe[e] < TR[e, 0]: TR[e, 0] = Fe[e]
            if Fe[e] > TR[e, 1]: TR[e, 1] = Fe[e]
            if se[e] < TR[e, 2]: TR[e, 2] = se[e]
            if se[e] > TR[e, 3]: TR[e, 3] = se[e]
            for k in range(NAUX):
                if aux[e, k] < TR[e, 4 + 2 * k]: TR[e, 4 + 2 * k] = aux[e, k]
                if aux[e, k] > TR[e, 5 + 2 * k]: TR[e, 5 + 2 * k] = aux[e, k]
        for b in range(nb):
            if q[b] < BT[b, 0]: BT[b, 0] = q[b]
            if q[b] > BT[b, 1]: BT[b, 1] = q[b]
            if v[b] < BT[b, 2]: BT[b, 2] = v[b]
            if v[b] > BT[b, 3]: BT[b, 3] = v[b]
        for c in range(ncr):
            if t_cross[c] < 0.0:
                src = cross_src[c]; k = cross_idx[c]
                if src == 0:
                    x = q[k]
                elif src == 1:
                    x = v[k]
                elif src == 2:
                    x = Fe[k]
                elif src == 3:
                    x = se[k]
                elif src == 50:
                    x = sig[k]
                else:
                    x = aux[k, src - 4]
                if (cross_dir[c] > 0 and x >= cross_thr[c]) or (cross_dir[c] < 0 and x <= cross_thr[c]):
                    t_cross[c] = tn
        if i % 4 == 0:
            ke_ = 0.0
            for b in range(nb):
                ke_ += 0.5 * m[b] * v[b] * v[b]
            if ke_ > KEpk: KEpk = ke_
        if nimp > 0:
            jcnt += 1
            for e in range(ne):
                if typ[e] == 14:
                    jacc[e] += Fe[e]
        if npart > 0:
            for e in range(ne):
                if typ[e] == 17:                 # the circuit reads each motor's speed at the end of the step
                    u_ = 0.0
                    for b in range(nb):
                        u_ += C[e, b] * v[b]
                    st[e, 2] = u_
            circuit_step(tn, dt, cnn, cpt, cna, cnb, cprm, cpst, cth, cpout, cmot, st, cflg, cG, cI, cvn)
            for k in range(npart):
                for c in range(8):
                    if cpout[k, c] < CTR[k, 2 * c]: CTR[k, 2 * c] = cpout[k, c]
                    if cpout[k, c] > CTR[k, 2 * c + 1]: CTR[k, 2 * c + 1] = cpout[k, c]
        if adapt:
            logit = log_every > 0 and t >= tlog and j < nlog
            if logit:
                while tlog <= t:
                    tlog += adp[7]
        else:
            logit = log_every > 0 and i % log_every == 0 and j < nlog
        if logit:
            if npart > 0:
                for k in range(npart):
                    for c in range(8):
                        LOGC[j, k * 8 + c] = cpout[k, c]
            LOGT[j] = t
            for k in range(nsig):
                LOGS[j, k] = sig[k]
            for b in range(nb):
                LOGB[j, 2 * b] = q[b]; LOGB[j, 2 * b + 1] = v[b]
            for e in range(ne):
                LOGE[j, e * (2 + NAUX)] = (jacc[e] / jcnt if (typ[e] == 14 and jcnt > 0) else Fe[e]); LOGE[j, e * (2 + NAUX) + 1] = se[e]
                for k in range(NAUX):
                    LOGE[j, e * (2 + NAUX) + 2 + k] = aux[e, k]
            j += 1
            jcnt = 0
            for e in range(ne):
                jacc[e] = 0.0
        if adapt:
            t = tn
    stop[1] = done
    adp[14] = t if adapt else done * dt
    for e in range(min(ne, cctlD.shape[0])):
        if cctlD[e] == 1:
            D[e] = -1.0                          # a controlled spring: its stored energy is no longer its own law
    KE1 = 0.0
    for b in range(nb):
        KE1 += 0.5 * m[b] * v[b] * v[b]
    return (q, v, KE0, KE1, A, D, IN, S0, S1, TR, BT, aux, t_cross, LOGT[:j], LOGB[:j], LOGE[:j], Fe, se, KEpk, cpout,
            CTR, LOGC[:j], sig, SG, LOGS[:j])
