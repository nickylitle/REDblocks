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

from ..constitutive.plasticity import tabular_return
from ..actuators.motors import esc_supply, lipo_ocv_k

NP = 10      # floats per element
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
        I, Ib, v, duty = esc_supply(Veff * thr, R0, Kt * wm, Rm, Imax)
        if Ib < 0.0:
            Ib = 0.0
        Ech = ocv * Ib * dt
        diss = (R0 * Ib * Ib + R1 * iR1 * iR1 + Rm * I * I) * dt
        a = math.exp(-dt / (R1 * C1)) if R1 > 0.0 and C1 > 0.0 else 0.0
        st[4] = a * iR1 + (1.0 - a) * Ib
        st[3] = soc - Ib * dt / Q
        aux[0] = I; aux[1] = Ib; aux[2] = v; aux[3] = st[3]
        cap = 0.5 * C1 * (R1 * st[4]) ** 2 if C1 > 0.0 else 0.0
        return -G * Kt * I, diss, Ech, cap
    return 0.0, 0.0, 0.0, 0.0


@njit(cache=True, nogil=True)
def run_engine(n, dt, q0, v0, m, typ, one_sided, active, P, C, s0, tab_off, tab_len, tab_k, tab_x, tab_F,
               log_every, cross_src, cross_idx, cross_thr, cross_dir, stop):
    """stop: int64[2]. Another thread sets stop[0] = 1 to end the run early (timeout); stop[1] returns steps done."""
    nb = q0.shape[0]; ne = typ.shape[0]
    q = q0.copy(); v = v0.copy()
    Q = np.zeros(nb)
    st = np.zeros((ne, 5))
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
    nlog = n // log_every + 2 if log_every > 0 else 1      # no history: no log arrays (was n rows of zeros)
    LOGB = np.zeros((nlog, 2 * nb)); LOGE = np.zeros((nlog, ne * (2 + NAUX))); LOGT = np.zeros(nlog); j = 0
    KE0 = 0.0
    for b in range(nb):
        KE0 += 0.5 * m[b] * v[b] * v[b]
    first = True
    done = n
    Dq = np.zeros(nb); nimp = 0
    for e in range(ne):
        if typ[e] == 14:
            nimp += 1
    cand = np.zeros(ne, np.int64); tauv = np.zeros(ne); tgt = np.zeros(ne); jn = np.zeros(ne); invn = np.zeros(ne)
    for i in range(n):
        if (i & 4095) == 0 and stop[0] != 0:
            done = i
            break
        for e in range(ne):
            if typ[e] == 13:
                st[e, 0] = i * dt                # load tables against time read the clock from their state
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
            v[b] += dt * Q[b] / m[b]
        # rigid impacts (sequential impulses; see engine2d)
        for b in range(nb):
            Dq[b] = 0.0
        if nimp > 0:
            nc = 0
            for e in range(ne):
                if typ[e] != 14:
                    continue
                Fe[e] = 0.0; aux[e, 0] = 0.0
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
                else:
                    x = aux[k, src - 4]
                if (cross_dir[c] > 0 and x >= cross_thr[c]) or (cross_dir[c] < 0 and x <= cross_thr[c]):
                    t_cross[c] = (i + 1) * dt
        if log_every > 0 and i % log_every == 0 and j < nlog:
            LOGT[j] = i * dt
            for b in range(nb):
                LOGB[j, 2 * b] = q[b]; LOGB[j, 2 * b + 1] = v[b]
            for e in range(ne):
                LOGE[j, e * (2 + NAUX)] = Fe[e]; LOGE[j, e * (2 + NAUX) + 1] = se[e]
                for k in range(NAUX):
                    LOGE[j, e * (2 + NAUX) + 2 + k] = aux[e, k]
            j += 1
    stop[1] = done
    KE1 = 0.0
    for b in range(nb):
        KE1 += 0.5 * m[b] * v[b] * v[b]
    return q, v, KE0, KE1, A, D, IN, S0, S1, TR, BT, aux, t_cross, LOGT[:j], LOGB[:j], LOGE[:j], Fe, se
