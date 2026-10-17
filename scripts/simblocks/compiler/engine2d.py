"""2D (planar) engine: rigid bodies with up to three coordinates each (x, y of the body frame origin and the spin
angle rz), joined by elements that act on a SCALAR MEASURE s of the geometry.

The rule is the same as the 1D engine, generalised: an element returns a force F along its measure (F > 0 resists
s increasing). Every measure says how it changes, ds/dt = n . (v_B(p_B) - v_A(p_A)), through two material points
p_A, p_B and a unit direction n (or, for linear measures, a constant row C over the coordinates). The element then
pushes body A at p_A with +F n and body B at p_B with -F n. Forces, torques and every reaction come from that one
rule, so the lever arm of a contact is always the true geometric one (the wedge-face lesson: r_arm = -ds/dq).

Measure kinds (mk):
  0 linear     s = s0 + sum_k C[e,k] q[k]          over the flat coordinate list [x0, y0, rz0, x1, y1, rz1, ...]
  1 distance   s = |p_B - p_A| - L0                points fixed on the bodies
  2 along      s = (p_B - p_A) . d - L0            d a fixed world direction
  3 edge       circle (r >= 0, a point is r = 0) on A against a line segment on B; s = penetration, > 0 in contact,
               only while the circle's centre projects inside the segment and the penetration is under `depth`
  4 circles    circle on A against circle on B; s = r_A + r_B - |c_B - c_A| (overlap), > 0 in contact
  5 pin        two points held together by a stiff spring-damper in both directions (type 9 law k, c)
  6 slip       Coulomb friction on the tangent of another contact element (type 10 law mu, v_eps)
Contact measures carry nothing while out of contact (F = 0, element state untouched).

Integration: semi-implicit Euler per coordinate, v += dt Q / m, then q += dt v (frozen coordinates have 1/m = 0).
Gravity acts at each body's centre of mass and its work is counted as an energy input.
Energy bookkeeping is the 1D scheme: per element absorbed work A = sum F ds (trapezoid), stored and dissipated from
the element's own law; the global check is KE change + sum A - gravity work = 0."""
import math
import numpy as np
from numba import njit

from .circuit import circuit_step

from .engine1d import _force, NP, NAUX
from .control import control_step
from .adaptive import pick_dt

MK_LIN, MK_DIST, MK_ALONG, MK_EDGE, MK_CIRC, MK_PIN, MK_SLIP, MK_BEAM = 0, 1, 2, 3, 4, 5, 6, 7


@njit(cache=True, nogil=True, inline="always")
def _yield_at(acc, tx, tF):
    """Yield moment after accumulated plastic rotation acc (linear in the table, flat past its end)."""
    n = tx.shape[0]
    if n == 0:
        return 1e300
    if acc <= tx[0]:
        return tF[0]
    for i in range(1, n):
        if acc <= tx[i]:
            return tF[i - 1] + (tF[i] - tF[i - 1]) * (acc - tx[i - 1]) / (tx[i] - tx[i - 1])
    return tF[n - 1]


@njit(cache=True, nogil=True)
def _beam2(e, q, ba, bb, GA, GB, P, st, tab_x, tab_F, xg, dt, first):
    """Co-rotational 2D beam. Returns (N, M1, M2, L, th1, th2, ax, ay, bx, by, e1x, e1y, stored, dplastic, ddamp).
    xg[e] holds the previous generalised coordinates [L, th1, th2] (for damping and the work trapezoid)."""
    a = ba[e]; b = bb[e]
    ax, ay = _pt(q, a, GA[e, 0], GA[e, 1]); bx, by = _pt(q, b, GB[e, 0], GB[e, 1])
    dx = bx - ax; dy = by - ay; L = math.sqrt(dx * dx + dy * dy)
    e1x = dx / L; e1y = dy / L
    beta = math.atan2(dy, dx)
    if first:
        st[4] = P[5]
    d = beta - st[4]                                # unwrap against last step's chord angle
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    beta = st[4] + d; st[4] = beta
    phA = q[3 * a + 2] - P[6] if a >= 0 else 0.0
    phB = q[3 * b + 2] - P[7] if b >= 0 else 0.0
    th1 = phA - (beta - P[5]); th2 = phB - (beta - P[5])
    kA = P[0]; kB = P[1]; L0 = P[4]
    # elastic trial, then a return map at each end with a hinge table (plastic rotation st[0], st[1])
    dpl = 0.0
    e1_ = th1 - st[0]; e2_ = th2 - st[1]
    M1 = kB * (4.0 * e1_ + 2.0 * e2_); M2 = kB * (2.0 * e1_ + 4.0 * e2_)
    if P[8] > 0.5:
        for end in range(2):                        # each end sees the other's latest plastic rotation
            e1_ = th1 - st[0]; e2_ = th2 - st[1]
            Mi = kB * (4.0 * e1_ + 2.0 * e2_) if end == 0 else kB * (2.0 * e1_ + 4.0 * e2_)
            Y = _yield_at(st[2 + end], tab_x, tab_F)
            if abs(Mi) > Y:
                sg = 1.0 if Mi > 0 else -1.0
                dth = (abs(Mi) - Y) / (4.0 * kB) * sg
                st[end] += dth; st[2 + end] += abs(dth)
                dpl += 0.5 * (abs(Mi) + Y) * abs(dth)   # energy released: mean moment times plastic rotation
        e1_ = th1 - st[0]; e2_ = th2 - st[1]
        M1 = kB * (4.0 * e1_ + 2.0 * e2_); M2 = kB * (2.0 * e1_ + 4.0 * e2_)
    N = kA * (L - L0)
    stored = 0.5 * kA * (L - L0) ** 2 + kB * (2.0 * e1_ * e1_ + 2.0 * e1_ * e2_ + 2.0 * e2_ * e2_)
    dd = 0.0
    if (P[2] > 0.0 or P[3] > 0.0) and not first:                    # stiffness-proportional damping, backward-difference rate
        rl = (L - xg[e, 0]) / dt; r1 = (th1 - xg[e, 1]) / dt; r2 = (th2 - xg[e, 2]) / dt
        Nd = P[2] * kA * rl
        if P[9] > 0.0:
            # bending: a damper aimed at the first mode, behind a low-pass at a few times its frequency (a Maxwell
            # element), so the fast segment modes the explicit step cannot damp stably never see it
            lp_ = P[9] * dt
            if lp_ > 1.0:
                lp_ = 1.0
            st[5] += lp_ * (P[3] * kB * (4.0 * r1 + 2.0 * r2) - st[5])
            st[6] += lp_ * (P[3] * kB * (2.0 * r1 + 4.0 * r2) - st[6])
            M1d = st[5]; M2d = st[6]
        else:
            bb_ = P[3] if P[3] > 0.0 else P[2]
            M1d = bb_ * kB * (4.0 * r1 + 2.0 * r2); M2d = bb_ * kB * (2.0 * r1 + 4.0 * r2)
        N += Nd; M1 += M1d; M2 += M2d
        dd = (Nd * rl + M1d * r1 + M2d * r2) * dt
    return N, M1, M2, L, th1, th2, ax, ay, bx, by, e1x, e1y, stored, dpl, dd


@njit(cache=True, nogil=True, inline="always")
def _pt(q, b, lx, ly):
    """World position of a body-fixed point (b = -1: ground, the point is already in world coordinates)."""
    if b < 0:
        return lx, ly
    c = math.cos(q[3 * b + 2]); s = math.sin(q[3 * b + 2])
    return q[3 * b] + c * lx - s * ly, q[3 * b + 1] + s * lx + c * ly


@njit(cache=True, nogil=True, inline="always")
def _vel(q, v, b, px, py):
    """Velocity of the material point of body b that is at world position (px, py)."""
    if b < 0:
        return 0.0, 0.0
    w = v[3 * b + 2]
    return v[3 * b] - w * (py - q[3 * b + 1]), v[3 * b + 1] + w * (px - q[3 * b])


@njit(cache=True, nogil=True, inline="always")
def _push(Q, q, b, px, py, fx, fy):
    """Generalised force of a world force (fx, fy) applied at world point (px, py) of body b."""
    if b < 0:
        return
    Q[3 * b] += fx; Q[3 * b + 1] += fy
    Q[3 * b + 2] += (px - q[3 * b]) * fy - (py - q[3 * b + 1]) * fx


@njit(cache=True, nogil=True, inline="always")
def _inv_mass_along(q, minv, b, px, py, nx, ny):
    """Inverse effective mass of body b at world point (px, py) along unit direction (nx, ny)."""
    if b < 0:
        return 0.0
    rx = px - q[3 * b]; ry = py - q[3 * b + 1]
    cr = rx * ny - ry * nx
    return minv[3 * b] * nx * nx + minv[3 * b + 1] * ny * ny + minv[3 * b + 2] * cr * cr


@njit(cache=True, nogil=True)
def _measure(mk, e, q, v, ba, bb, C, s0, GA, GB, GX, geo):
    """Returns (s, sd, in_contact). geo[e] receives [pAx, pAy, pBx, pBy, nx, ny] for the force step."""
    if mk == MK_LIN:
        s = s0[e]; sd = 0.0
        for k in range(q.shape[0]):
            if C[e, k] != 0.0:
                s += C[e, k] * q[k]; sd += C[e, k] * v[k]
        return s, sd, True
    a = ba[e]; b = bb[e]
    if mk == MK_DIST or mk == MK_ALONG or mk == MK_PIN:
        ax, ay = _pt(q, a, GA[e, 0], GA[e, 1]); bx, by = _pt(q, b, GB[e, 0], GB[e, 1])
        dx = bx - ax; dy = by - ay
        if mk == MK_DIST:
            L = math.sqrt(dx * dx + dy * dy)
            if L > 1e-15:
                nx = dx / L; ny = dy / L
            else:
                nx = GX[e, 0]; ny = GX[e, 1]
            s = L - GX[e, 2]
        elif mk == MK_ALONG:
            if GX[e, 3] > 0.5 and a >= 0:              # direction fixed in body A: it turns with A
                c_ = math.cos(q[3 * a + 2]); s_ = math.sin(q[3 * a + 2])
                nx = c_ * GX[e, 0] - s_ * GX[e, 1]; ny = s_ * GX[e, 0] + c_ * GX[e, 1]
                s = dx * nx + dy * ny - GX[e, 2]
                # A's material point at B's point: the rate (and A's torque) include the turning of the direction
                vax, vay = _vel(q, v, a, bx, by); vbx, vby = _vel(q, v, b, bx, by)
                geo[e, 0] = bx; geo[e, 1] = by; geo[e, 2] = bx; geo[e, 3] = by; geo[e, 4] = nx; geo[e, 5] = ny
                return s, nx * (vbx - vax) + ny * (vby - vay), True
            nx = GX[e, 0]; ny = GX[e, 1]
            s = dx * nx + dy * ny - GX[e, 2]
        else:
            L = math.sqrt(dx * dx + dy * dy)
            nx = 1.0; ny = 0.0
            s = L
        vax, vay = _vel(q, v, a, ax, ay); vbx, vby = _vel(q, v, b, bx, by)
        geo[e, 0] = ax; geo[e, 1] = ay; geo[e, 2] = bx; geo[e, 3] = by; geo[e, 4] = nx; geo[e, 5] = ny
        if mk == MK_PIN:
            return s, 0.0, True
        return s, nx * (vbx - vax) + ny * (vby - vay), True
    if mk == MK_EDGE:
        cx, cy = _pt(q, a, GA[e, 0], GA[e, 1]); r = GA[e, 2]
        x0, y0 = _pt(q, b, GB[e, 0], GB[e, 1]); x1, y1 = _pt(q, b, GB[e, 2], GB[e, 3])
        tx = x1 - x0; ty = y1 - y0; L = math.sqrt(tx * tx + ty * ty); tx /= L; ty /= L
        side = GX[e, 3]
        nx = -ty * side; ny = tx * side                     # outward normal of B's solid, toward where A started
        u = (cx - x0) * tx + (cy - y0) * ty
        s = r - ((cx - x0) * nx + (cy - y0) * ny)          # penetration
        px = cx - r * nx; py = cy - r * ny                 # contact point (on the circle, at the line)
        geo[e, 0] = px; geo[e, 1] = py; geo[e, 2] = px; geo[e, 3] = py; geo[e, 4] = nx; geo[e, 5] = ny
        vax, vay = _vel(q, v, a, px, py); vbx, vby = _vel(q, v, b, px, py)
        sd = nx * (vbx - vax) + ny * (vby - vay)
        inside = u >= 0.0 and u <= L and s > 0.0 and s < GB[e, 4]
        return s, sd, inside
    if mk == MK_CIRC:
        ax, ay = _pt(q, a, GA[e, 0], GA[e, 1]); bx, by = _pt(q, b, GB[e, 0], GB[e, 1])
        dx = bx - ax; dy = by - ay; L = math.sqrt(dx * dx + dy * dy)
        if L > 1e-15:
            ux = dx / L; uy = dy / L
        else:
            ux = 1.0; uy = 0.0
        ra = GA[e, 2]; rb = GB[e, 2]
        s = ra + rb - L
        pax = ax + ra * ux; pay = ay + ra * uy; pbx = bx - rb * ux; pby = by - rb * uy
        nx = -ux; ny = -uy                                  # s grows as the centres approach
        geo[e, 0] = pax; geo[e, 1] = pay; geo[e, 2] = pbx; geo[e, 3] = pby; geo[e, 4] = nx; geo[e, 5] = ny
        vax, vay = _vel(q, v, a, pax, pay); vbx, vby = _vel(q, v, b, pbx, pby)
        return s, nx * (vbx - vax) + ny * (vby - vay), s > 0.0
    return 0.0, 0.0, False


@njit(cache=True, nogil=True)
def _edge_span(e, q, ba, bb, GA, GB):
    """Position of a circle's centre along the segment it may hit, as a fraction (0..1 = over the segment)."""
    cx, cy = _pt(q, ba[e], GA[e, 0], GA[e, 1])
    x0, y0 = _pt(q, bb[e], GB[e, 0], GB[e, 1]); x1, y1 = _pt(q, bb[e], GB[e, 2], GB[e, 3])
    tx = x1 - x0; ty = y1 - y0; L2 = tx * tx + ty * ty
    return ((cx - x0) * tx + (cy - y0) * ty) / L2


@njit(cache=True, nogil=True)
def _impulse2(e, dj, which, mk_, q, v, minv, ba, bb, C, geo, Dq, tdt):
    """Apply impulse dj along the contact normal (which = 0) or tangent (1): +dj on A, -dj on B (or -dj C on a
    linear measure). Positions catch up by -dv tdt (the bodies moved at the old speed until the hit).
    Returns the kinetic energy it removes (exact: dj (u_before + u_after) / 2)."""
    u0 = _rate(mk_, e, q, v, ba, bb, C, geo, which)
    if mk_ == MK_LIN:
        for k in range(q.shape[0]):
            if C[e, k] != 0.0 and minv[k] > 0.0:
                dv = -C[e, k] * dj * minv[k]
                v[k] += dv; Dq[k] -= dv * tdt
    else:
        nx = geo[e, 4]; ny = geo[e, 5]
        if which == 1:
            nx, ny = -ny, nx
        for side in range(2):
            b = ba[e] if side == 0 else bb[e]
            if b < 0:
                continue
            sg = dj if side == 0 else -dj
            px = geo[e, 0] if side == 0 else geo[e, 2]; py = geo[e, 1] if side == 0 else geo[e, 3]
            fx = sg * nx; fy = sg * ny
            tq = (px - q[3 * b]) * fy - (py - q[3 * b + 1]) * fx
            for k, g_ in ((0, fx), (1, fy), (2, tq)):
                dv = g_ * minv[3 * b + k]
                v[3 * b + k] += dv; Dq[3 * b + k] -= dv * tdt
    u1 = _rate(mk_, e, q, v, ba, bb, C, geo, which)
    return 0.5 * dj * (u0 + u1)


@njit(cache=True, nogil=True, inline="always")
def _jpt(J, r, q, b, px, py, nx, ny, sg):
    """Row r of a constraint Jacobian: sg * (velocity of body b's material point at p) . n."""
    if b < 0:
        return
    J[r, 3 * b] += sg * nx; J[r, 3 * b + 1] += sg * ny
    J[r, 3 * b + 2] += sg * ((px - q[3 * b]) * ny - (py - q[3 * b + 1]) * nx)


@njit(cache=True, nogil=True)
def _rows2(e, mk_, q, ba, bb, C, s0, GA, GB, GX, ref, J, cpos):
    """Rigid constraint rows of element e: fills J[0:n] (velocity Jacobian rows, generalised) and cpos[0:n]
    (position errors). Returns n. pin: x and y of (pB - pA); linear: s - s_ref; along: s - s_ref."""
    nq = q.shape[0]
    for r in range(2):
        for k in range(nq):
            J[r, k] = 0.0
    a = ba[e]; b = bb[e]
    if mk_ == MK_LIN:
        sv = s0[e]
        for k in range(nq):
            J[0, k] = C[e, k]; sv += C[e, k] * q[k]
        cpos[0] = sv - ref[e]
        return 1
    ax, ay = _pt(q, a, GA[e, 0], GA[e, 1]); bx, by = _pt(q, b, GB[e, 0], GB[e, 1])
    if mk_ == MK_PIN:
        for r in range(2):
            nx = 1.0 if r == 0 else 0.0; ny = 1.0 - nx
            _jpt(J, r, q, b, bx, by, nx, ny, 1.0); _jpt(J, r, q, a, ax, ay, nx, ny, -1.0)
        cpos[0] = bx - ax; cpos[1] = by - ay
        return 2
    # along (a slider's guide): direction fixed in A when GX[e, 3] is set
    nx = GX[e, 0]; ny = GX[e, 1]
    if GX[e, 3] > 0.5 and a >= 0:
        c_ = math.cos(q[3 * a + 2]); s_ = math.sin(q[3 * a + 2])
        nx, ny = c_ * GX[e, 0] - s_ * GX[e, 1], s_ * GX[e, 0] + c_ * GX[e, 1]
        _jpt(J, 0, q, b, bx, by, nx, ny, 1.0); _jpt(J, 0, q, a, bx, by, nx, ny, -1.0)
    else:
        _jpt(J, 0, q, b, bx, by, nx, ny, 1.0); _jpt(J, 0, q, a, ax, ay, nx, ny, -1.0)
    cpos[0] = (bx - ax) * nx + (by - ay) * ny - GX[e, 2] - ref[e]
    return 1



@njit(cache=True, nogil=True)
def _jsolve2(mode, ne, nq, typ, active, mk, q, v, minv, ba, bb, C, s0, GA, GB, GX, refc, Jr, cps, Jold, lam, aux,
             JA, JO, rhs, re, rr, dt, first_pass):
    """Every rigid-joint row at once: K x = rhs with K = J M^-1 J^T (a direct solve, so long levers and light
    bodies converge in one go where row-by-row sweeps crawl). mode 0: velocities onto the constraints (current J);
    mode 1: positions (SHAKE, moving along the start-of-step directions Jold). Returns the largest residual."""
    n = 0
    for e in range(ne):
        if typ[e] != 15 or active[e] == 0:
            continue
        nr = _rows2(e, mk[e], q, ba, bb, C, s0, GA, GB, GX, refc, Jr, cps)
        for r in range(nr):
            u = 0.0
            for k in range(nq):
                JA[n, k] = Jr[r, k]
                JO[n, k] = Jold[e, r, k] if mode == 1 else Jr[r, k]
                u += Jr[r, k] * v[k]
            rhs[n] = -cps[r] if mode == 1 else -u
            if mode == 1 and first_pass and abs(cps[r]) > aux[e, 3]:
                aux[e, 3] = abs(cps[r])                  # largest drift seen before correction
            re[n] = e; rr[n] = r; n += 1
    if n == 0:
        return 0.0
    mx = 0.0
    for i in range(n):
        if abs(rhs[i]) > mx:
            mx = abs(rhs[i])
    if mx == 0.0:
        return 0.0
    K = np.zeros((n, n)); tr = 0.0
    for i in range(n):
        for j in range(n):
            s_ = 0.0
            for k in range(nq):
                if JA[i, k] != 0.0 and JO[j, k] != 0.0:
                    s_ += JA[i, k] * JO[j, k] * minv[k]
            K[i, j] = s_
        tr += K[i, i]
    if tr <= 0.0:
        return mx
    for i in range(n):                                   # a whisker of regularisation: redundant rows (a closed
        K[i, i] += 1e-12 * tr / n                        # loop) still solve, as the smallest impulses
    x = np.linalg.solve(K, rhs[:n].copy())
    for i in range(n):
        if not np.isfinite(x[i]):
            return mx
    for j in range(n):
        e = re[j]; r = rr[j]
        for k in range(nq):
            if JO[j, k] != 0.0:
                if mode == 1:
                    q[k] += minv[k] * JO[j, k] * x[j]
                    v[k] += minv[k] * JO[j, k] * x[j] / dt
                else:
                    v[k] += minv[k] * JO[j, k] * x[j]
        lam[e, r] += x[j] / dt if mode == 1 else x[j]
    return mx

@njit(cache=True, nogil=True)
def _book_hits(cand, nc, jn, held, lossN, lossT, D, A):
    """Heat of rigid contacts this step (the kinetic energy their own impulses removed). A NEW hit books all of it.
    A HELD contact (pressed last step too: resting, or pushed by a motor) books only its sliding friction: the speed
    this step's forces pressed into it never moved anything, so taking it back out is not a loss (the explicit step
    made that energy without work). Energy that reaches a hit through rigid joints, and the explicit step's own
    error, show in the energy ledger instead; both shrink with the time step."""
    for c in range(nc):
        e = cand[c]
        if jn[e] <= 0.0:
            continue
        h = lossT[e] if held[e] > 0.0 else lossN[e] + lossT[e]
        D[e] += h; A[e] += h


@njit(cache=True, nogil=True)
def _rate(mk, e, q, v, ba, bb, C, geo, which):
    """ds/dt with the current velocities and the stored application points (which = 0 normal, 1 tangent)."""
    if mk == MK_LIN:
        sd = 0.0
        for k in range(q.shape[0]):
            if C[e, k] != 0.0:
                sd += C[e, k] * v[k]
        return sd
    nx = geo[e, 4]; ny = geo[e, 5]
    if which == 1:
        nx, ny = -ny, nx
    vax, vay = _vel(q, v, ba[e], geo[e, 0], geo[e, 1]); vbx, vby = _vel(q, v, bb[e], geo[e, 2], geo[e, 3])
    return nx * (vbx - vax) + ny * (vby - vay)


# ---- vehicles in flight: planet gravity, the US Standard Atmosphere 1976, rockets, staging, drag and steering
TYP_ROCKET, TYP_JETTISON, TYP_AERO, TYP_STEER, TYP_LAND = 18, 19, 20, 21, 22
_US76_HB = (0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 84852.0)      # geopotential layer bases, m
_US76_TB = (288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946)
_US76_PB = (101325.0, 22632.06, 5474.889, 868.0187, 110.9063, 66.93887, 3.956420, 0.3734)
_US76_L = (-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002, 0.0)
# above 86 km (geometric): tabulated US76 pressure and density, log-interpolated
_HI_Z = (86000.0, 100000.0, 120000.0, 150000.0, 200000.0, 300000.0, 500000.0, 1000000.0)
_HI_P = (0.3734, 3.201e-2, 2.538e-3, 4.542e-4, 8.474e-5, 8.770e-6, 3.019e-7, 7.514e-9)
_HI_R = (6.958e-6, 5.604e-7, 2.222e-8, 2.076e-9, 2.541e-10, 1.916e-11, 5.215e-13, 3.561e-15)


@njit(cache=True, nogil=True)
def atmosphere(z):
    """US Standard Atmosphere 1976 at geometric altitude z (m): pressure (Pa), density (kg/m^3), temperature (K)."""
    if z < 0.0:
        z = 0.0
    if z >= 86000.0:
        for i in range(7):
            if z < _HI_Z[i + 1] or i == 6:
                f = (z - _HI_Z[i]) / (_HI_Z[i + 1] - _HI_Z[i])
                if f > 1.0:
                    f = 1.0
                p = math.exp(math.log(_HI_P[i]) + f * (math.log(_HI_P[i + 1]) - math.log(_HI_P[i])))
                rho = math.exp(math.log(_HI_R[i]) + f * (math.log(_HI_R[i + 1]) - math.log(_HI_R[i])))
                return p, rho, 186.87 if z < 91000.0 else 186.87 + (z - 91000.0) * 0.005
    h = 6356766.0 * z / (6356766.0 + z)                  # geopotential altitude
    k = 0
    for i in range(7):
        if h >= _US76_HB[i + 1]:
            k = i + 1
    T0 = _US76_TB[k]; P0 = _US76_PB[k]; L = _US76_L[k]; dh = h - _US76_HB[k]
    if L == 0.0:
        T = T0; p = P0 * math.exp(-9.80665 * 0.0289644 * dh / (8.3144598 * T0))
    else:
        T = T0 + L * dh; p = P0 * (T / T0) ** (-9.80665 * 0.0289644 / (8.3144598 * L))
    return p, p * 0.0289644 / (8.3144598 * T), T


@njit(cache=True, nogil=True, inline="always")
def _local(q, v, b, planet):
    """Body b's place over the planet: radius, radial and horizontal (prograde) unit vectors, radial and horizontal
    speed, and the velocity relative to the turning air."""
    x = q[3 * b]; y = q[3 * b + 1]; r = math.sqrt(x * x + y * y)
    urx = x / r; ury = y / r; uhx = ury; uhy = -urx
    vx = v[3 * b]; vy = v[3 * b + 1]
    w = planet[2]
    ax = vx - w * r * uhx; ay = vy - w * r * uhy            # the air turns with the planet
    return r, urx, ury, uhx, uhy, vx * urx + vy * ury, vx * uhx + vy * uhy, ax, ay


@njit(cache=True, nogil=True, inline="always")
def _cd_at(M, tx, tF):
    n = tx.shape[0]
    if n == 0:
        return 0.0
    if M <= tx[0]:
        return tF[0]
    for i in range(1, n):
        if M <= tx[i]:
            return tF[i - 1] + (tF[i] - tF[i - 1]) * (M - tx[i - 1]) / (tx[i] - tx[i - 1])
    return tF[n - 1]


@njit(cache=True, nogil=True, inline="always")
def _ramp(t, on, off, rise, tail):
    """Thrust level 0-1: off before on, rises linearly over rise, full, then falls linearly over tail after off."""
    if t < on:
        return 0.0
    f = 1.0 if rise <= 0.0 else min(1.0, (t - on) / rise)
    if t >= off:
        if tail <= 0.0:
            return 0.0
        g = 1.0 - (t - off) / tail
        f = f * (g if g > 0.0 else 0.0)
    return f


@njit(cache=True, nogil=True)
def run_engine2d(n, dt, q0, v0, minv, mass, com, grav, typ, one_sided, active, mk, ba, bb, ref, P, C, s0, GA, GB, GX,
                 tab_off, tab_len, tab_k, tab_x, tab_F, log_every, cross_src, cross_idx, cross_thr, cross_dir,
               cpt, cna, cnb, cprm, cpst0, cth, cnn, cmot,
               ckind, cprog, cval, crng, ccp, cssrc, csidx, cae, caslot, casig, caarr, cctlD, planet, adp, adf, adc, stop):
    """q0, v0, minv: flat [x, y, rz] per body. mass[b], com[b] = (cx, cy) body-frame centre of mass (for gravity).
    stop: int64[2] as in the 1D engine (timeout)."""
    nb = mass.shape[0]; nq = q0.shape[0]; ne = typ.shape[0]
    q = q0.copy(); v = v0.copy()
    minv = minv.copy(); mass = mass.copy()          # rockets burn and stages drop: the masses change in flight
    P = P.copy(); cprm = cprm.copy()                # control signals may set element fields as it runs
    nsig = ckind.shape[0]; nsen = cssrc.shape[0]
    sig = np.zeros(nsig); cst = np.zeros((nsig, 4)); sens = np.zeros(nsen); cstack = np.zeros(64)
    SG = np.empty((nsig, 2))
    for k in range(nsig):
        SG[k, 0] = 1e300; SG[k, 1] = -1e300
    mu_p = planet[0]; Rp = planet[1]
    nveh = 0
    for e in range(ne):
        if typ[e] >= TYP_ROCKET and typ[e] <= TYP_LAND:
            nveh += 1
    incr = mu_p > 0.0 or nveh > 0                      # gravity work booked step by step (mass and field vary)
    gfx = np.zeros(nb); gfy = np.zeros(nb)
    steer_of = -np.ones(nb, np.int64)
    for e in range(ne):
        if typ[e] == TYP_STEER and ba[e] >= 0 and active[e] == 1:   # a knocked-out steering leaves thrust radial
            steer_of[ba[e]] = e
    Tb = np.zeros(nb); Mb = np.zeros(nb)               # this step's thrust and propellant flow on each body
    frx = np.zeros(ne); fry = np.zeros(ne); dmv = np.zeros(ne); cut_t = np.full(ne, 1e300)
    # landing guidance (Apollo P63 / P64 / P66): this step's thrust direction and throttle level per body
    gdx = np.zeros(nb); gdy = np.zeros(nb); gthr = np.zeros(nb); gon = np.zeros(nb, np.int64)
    grec = np.zeros(nb, np.int64); Frated = np.zeros(nb); landed = False; lastend = np.zeros(nb, np.int64)
    for e in range(ne):
        if typ[e] == TYP_ROCKET and P[e, 8] > 0.5 and ba[e] >= 0:
            Frated[ba[e]] += P[e, 0]
    th0 = np.zeros(nb)
    for b in range(nb):
        th0[b] = math.atan2(q0[3 * b], q0[3 * b + 1])   # starting angle round the planet (for downrange)
    Q = np.zeros(nq)
    st = np.zeros((ne, 10))                    # beams: plastic rotations, accumulators, chord angle, damping state
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
            st[e, 3] = 1.0
    aux = np.zeros((ne, NAUX))
    A = np.zeros(ne); D = np.zeros(ne); IN = np.zeros(ne); S0 = np.zeros(ne); S1 = np.zeros(ne)
    Fe = np.zeros(ne); se = np.zeros(ne); sde = np.zeros(ne); inc = np.zeros(ne, np.int64)
    Fp = np.zeros(ne); dsp = np.zeros(ne)
    gpv = np.zeros((ne, 3)); fpv = np.zeros((ne, 3))       # pin / beam: previous coordinates and forces
    Dq = np.zeros(nq)                                       # rigid impacts: position catch-up this step
    nimp = 0
    for e in range(ne):
        if typ[e] == 14:
            nimp += 1
    cand = np.zeros(ne, np.int64); tauv = np.zeros(ne); tgt = np.zeros(ne); jn = np.zeros(ne); jt = np.zeros(ne)
    invn = np.zeros(ne); invt = np.zeros(ne); ucd = np.zeros(ne); u0d = np.zeros(ne); lst = np.zeros(ne); held = np.zeros(ne)
    lossN = np.zeros(ne); lossT = np.zeros(ne)
    # rigid joints (typ 15): projected constraints, rows in generalised coordinates
    nrig = 0
    for e in range(ne):
        if typ[e] == 15:
            nrig += 1
    nrig_ = nrig
    vst = np.zeros(nq)                                       # velocities at the start of the step
    Jold = np.zeros((ne if nrig_ > 0 else 1, 2, nq)); nrow = np.zeros(ne, np.int64)   # SHAKE directions
    Jr = np.zeros((2, nq)); cps = np.zeros(2); lam = np.zeros((ne, 2)); keff = np.zeros(2); refc = np.zeros(ne)
    nja = 2 * nrig_ if nrig_ > 0 else 1
    JA = np.zeros((nja, nq)); JO = np.zeros((nja, nq)); rhs = np.zeros(nja)
    rje = np.zeros(nja, np.int64); rjr = np.zeros(nja, np.int64)
    for e in range(ne):
        if typ[e] == 15 and mk[e] != MK_PIN:
            nr = _rows2(e, mk[e], q0, ba, bb, C, s0, GA, GB, GX, refc, Jr, cps)
            refc[e] = cps[0]                     # hold the starting value
    # geometric measures book their work on the true change in s (trapezoid), so a measure that turns while
    # loaded (a rope, a whirling link) closes the ledger; linear measures and friction keep the rate form
    exact = np.zeros(ne, np.int64)
    for e in range(ne):
        if mk[e] != MK_LIN and mk[e] != MK_SLIP and typ[e] != 9 and mk[e] != MK_BEAM and typ[e] != 14 and typ[e] != 15:
            exact[e] = 1
    geo = np.zeros((ne, 6))
    TR = np.empty((ne, 12))
    for e in range(ne):
        for k in range(6):
            TR[e, 2 * k] = 1e300; TR[e, 2 * k + 1] = -1e300
    BT = np.empty((nq, 4))
    for k in range(nq):
        BT[k, 0] = q[k]; BT[k, 1] = q[k]; BT[k, 2] = v[k]; BT[k, 3] = v[k]
    SP = np.zeros((nb, 2))                              # speed of the frame origin [min, max]
    for b in range(nb):
        sp = math.sqrt(v[3 * b] ** 2 + v[3 * b + 1] ** 2); SP[b, 0] = sp; SP[b, 1] = sp
    ncr = cross_src.shape[0]
    t_cross = -np.ones(ncr)
    adapt = adp[0] > 0.5
    t = 0.0; tlog = 0.0; t_stop = adp[5] * (1.0 - 1e-13)
    adsp = np.zeros(ne)
    nlog = n // log_every + 2 if log_every > 0 else 1
    if adapt and log_every > 0:
        nlog = int(adp[5] / adp[7]) + 3
    LOGC = np.zeros((nlog if npart > 0 else 1, npart * 8 if npart > 0 else 1))
    LOGB = np.zeros((nlog, 2 * nq)); LOGE = np.zeros((nlog, ne * (2 + NAUX))); LOGT = np.zeros(nlog); j = 0
    LOGS = np.zeros((nlog if nsig > 0 else 1, nsig if nsig > 0 else 1))
    KE0 = 0.0
    for k in range(nq):
        if minv[k] > 0.0:
            KE0 += 0.5 * v[k] * v[k] / minv[k]
    PC0 = np.zeros((nb, 2))                             # gravity work is exact: m g . (com now - com at start)
    for b in range(nb):
        PC0[b, 0], PC0[b, 1] = _pt(q, b, com[b, 0], com[b, 1])
    Wg = 0.0
    first = True
    done = n
    KEpk = KE0
    jacc = np.zeros(ne); jcnt = 0          # rigid impacts: force logged as the mean over each log interval
    for i in range(n):
        if (i & 4095) == 0 and stop[0] != 0:
            done = i
            break
        if adapt:
            if t >= t_stop:
                done = i; stop[2] = 2               # reached t_end
                break
            dt = pick_dt(i, t, dt, adp, adf, adc, active, se, adsp, inc, True, sig, casig)
            tn = t + dt
        else:
            t = i * dt; tn = (i + 1) * dt
        for e in range(ne):
            if typ[e] == 13:
                st[e, 0] = t                # load tables against time read the clock from their state
        for k in range(nq):
            Q[k] = 0.0
        # gravity at each centre of mass: uniform, or a planet's inverse-square pull toward the origin
        for b in range(nb):
            gfx[b] = 0.0; gfy[b] = 0.0
            if mass[b] > 0.0 and mu_p > 0.0 and minv[3 * b] + minv[3 * b + 1] > 0.0:
                px, py = _pt(q, b, com[b, 0], com[b, 1])
                r2 = px * px + py * py; r_ = math.sqrt(r2)
                gfx[b] = -mu_p * mass[b] * px / (r2 * r_); gfy[b] = -mu_p * mass[b] * py / (r2 * r_)
                _push(Q, q, b, px, py, gfx[b], gfy[b])
            elif mass[b] > 0.0 and (grav[0] != 0.0 or grav[1] != 0.0):
                px, py = _pt(q, b, com[b, 0], com[b, 1])
                gfx[b] = mass[b] * grav[0]; gfy[b] = mass[b] * grav[1]
                _push(Q, q, b, px, py, gfx[b], gfy[b])
        # vehicles: steering first (it sets each body's thrust direction), then engines, drag and staging
        if nveh > 0:
            t_ = t
            for b in range(nb):
                Tb[b] = 0.0; Mb[b] = 0.0; gon[b] = 0
            for b in range(nb):
                lastend[b] = 1                                              # the first phase needs no predecessor
            for e in range(ne):
                if typ[e] != TYP_LAND or active[e] == 0:
                    continue
                Pe = P[e]
                b = ba[e]
                r, urx, ury, uhx, uhy, vr, vh, ax, ay = _local(q, v, b, planet)
                w = planet[2]
                phi = math.atan2(q[3 * b], q[3 * b + 1]) - th0[b] - w * t_      # angle travelled over the ground
                xg = Rp * phi - Pe[3]                                       # range to the site (negative: short of it)
                phid = vh / r - w
                xgd = Rp * phid; h = r - Rp
                vf = r * phid                                               # forward speed over the ground
                prev_done = lastend[b]
                lastend[b] = int(st[e, 9])
                if st[e, 8] < 0.5 and st[e, 9] < 0.5:                       # not started: by time, or after the last phase
                    if (Pe[0] >= 0.0 and t_ >= Pe[0]) or (Pe[0] < 0.0 and prev_done == 1):
                        st[e, 8] = 1.0; st[e, 7] = t_                       # started, at this time
                        st[e, 6] = -Pe[4]                                   # first estimate of the target-referenced time
                if st[e, 8] > 0.5 and st[e, 9] < 0.5 and (t_ >= Pe[1] or (Pe[23] > 0.0 and h <= Pe[23])):
                    st[e, 9] = 1.0                                          # its stop time, or down to its stop altitude
                if st[e, 8] < 0.5 or st[e, 9] > 0.5:                        # not this phase: report the state only
                    se[e] = xg; aux[e, 0] = h; aux[e, 1] = vf; aux[e, 2] = vr; aux[e, 3] = 0.0; Fe[e] = 0.0
                    lastend[b] = int(st[e, 9])
                    continue
                if Pe[2] > 0.5 and h <= 0.0:                                # touchdown: the run ends
                    landed = True
                    break
                if Pe[2] < 0.5:
                    # Apollo quadratic guidance in the landing-site frame (X up through the site, Z downrange), which
                    # turns with the Moon: a = a_T + 6 (v_T + v) / T + 12 (r_T - r) / T^2, T the target-referenced time
                    ths = th0[b] + Pe[3] / Rp + w * t_                          # the site's angle round the planet now
                    sux = math.sin(ths); suy = math.cos(ths)                    # up at the site
                    sdx = suy; sdy = -sux                                       # downrange at the site
                    px_ = q[3 * b] - Rp * sux; py_ = q[3 * b + 1] - Rp * suy
                    vfx = v[3 * b] - w * q[3 * b + 1]; vfy = v[3 * b + 1] + w * q[3 * b]   # velocity seen in the frame
                    X = px_ * sux + py_ * suy; Z = px_ * sdx + py_ * sdy
                    Xd = vfx * sux + vfy * suy; Zd = vfx * sdx + vfy * sdy
                    rTz = Pe[6]; vTz = Pe[7]; aTz = Pe[9]
                    if Pe[20] > 0.5:
                        # target-referenced time from the downrange jerk condition (Draper R-695 eq. 12), Newton
                        jTz = Pe[18]
                        T = st[e, 6]
                        for it in range(8):
                            fT = jTz * T ** 3 + 6.0 * aTz * T * T + (18.0 * vTz + 6.0 * Zd) * T + 24.0 * (rTz - Z)
                            dT = 3.0 * jTz * T * T + 12.0 * aTz * T + 18.0 * vTz + 6.0 * Zd
                            if dT == 0.0:
                                break
                            Tn = T - fT / dT
                            if Tn > -1e-3:
                                Tn = 0.5 * T
                            T = Tn
                            if abs(fT) < 1e-9 * (abs(rTz - Z) + 1.0):
                                break
                        st[e, 6] = T
                        TTF = T
                    else:
                        TTF = t_ - (st[e, 7] + Pe[4])
                    if Pe[21] > 0.0 and TTF >= -Pe[21]:                     # the phase ends this far before its target
                        st[e, 9] = 1.0; lastend[b] = 1
                        se[e] = xg; aux[e, 0] = h; aux[e, 1] = vf; aux[e, 2] = vr; aux[e, 3] = 0.0; Fe[e] = 0.0
                        continue
                    if TTF < -Pe[17]:
                        st[e, 3] = aTz + 6.0 * (vTz + Zd) / TTF + 12.0 * (rTz - Z) / (TTF * TTF)
                        st[e, 4] = Pe[10] + 6.0 * (Pe[8] + Xd) / TTF + 12.0 * (Pe[5] - X) / (TTF * TTF)
                        st[e, 5] = 1.0
                    elif st[e, 5] < 0.5:
                        st[e, 3] = aTz; st[e, 4] = Pe[10]
                    # inertial specific force: frame acceleration + Coriolis + centrifugal - gravity
                    aix = st[e, 4] * sux + st[e, 3] * sdx + 2.0 * w * vfy - w * w * q[3 * b]
                    aiy = st[e, 4] * suy + st[e, 3] * sdy - 2.0 * w * vfx - w * w * q[3 * b + 1]
                    r3 = r * r * r
                    fxx = aix + mu_p * q[3 * b] / r3; fyy = aiy + mu_p * q[3 * b + 1] / r3
                    fm = math.sqrt(fxx * fxx + fyy * fyy)
                    if fm > 0.0:
                        gdx[b] = fxx / fm; gdy[b] = fyy / fm
                    else:
                        gdx[b] = urx; gdy[b] = ury
                else:                                                       # rate of descent: follow a schedule
                    o = tab_off[e]; L = tab_len[e] // 2
                    tt = tab_x[o:o + L]
                    tr_ = t_ - st[e, 7]                                     # schedule times count from the phase start
                    vu_c = _cd_at(tr_, tt, tab_F[o:o + L]); vf_c = _cd_at(tr_, tt, tab_F[o + L:o + 2 * L])
                    tau = Pe[18] if Pe[18] > 0.0 else 1.5
                    ahg = (vu_c - vr) / tau
                    axg = (vf_c - vf) / tau * Rp / r
                    # specific force the engine must give (inertial, polar): radial and along-track
                    thd = phid + w
                    fr = ahg - r * thd * thd + mu_p / (r * r)
                    ft = r * (axg / Rp) + 2.0 * vr * thd
                    fm = math.sqrt(fr * fr + ft * ft)
                    if fm > 0.0:
                        gdx[b] = (fr * urx + ft * uhx) / fm; gdy[b] = (fr * ury + ft * uhy) / fm
                    else:
                        gdx[b] = urx; gdy[b] = ury
                cmd = mass[b] * fm / Frated[b] if Frated[b] > 0.0 else 0.0  # thrust asked for, fraction of rated
                if t_ < Pe[12]:
                    lvl = Pe[11]                                            # start at a low trim setting
                elif grec[b] == 0:
                    lvl = Pe[13]                                            # fixed full throttle until the ask falls
                    if cmd < Pe[14]:
                        grec[b] = 1                                         # throttle recovery
                        lvl = max(Pe[15], min(Pe[16], cmd))
                else:
                    lvl = Pe[13] if cmd > Pe[16] else max(Pe[15], cmd)
                if Pe[2] > 0.5 and Pe[19] > 0.0 and h <= Pe[19]:
                    lvl = 0.0                                               # contact light: engine stop
                gthr[b] = lvl; gon[b] = 1
                Fe[e] = cmd * Frated[b]; se[e] = xg
                aux[e, 0] = h; aux[e, 1] = vf; aux[e, 2] = vr; aux[e, 3] = lvl
            if landed:
                done = i
                stop[2] = 1
                break
            for e in range(ne):
                if typ[e] != TYP_STEER or active[e] == 0:
                    continue
                b = ba[e]
                r, urx, ury, uhx, uhy, vr, vh, ax, ay = _local(q, v, b, planet)
                Pe = P[e]
                if t_ < Pe[0]:                                   # vertical rise
                    dx = urx; dy = ury
                elif t_ < Pe[2]:                                 # pitch-over at a set rate from vertical
                    a_ = Pe[1] * (t_ - Pe[0])
                    dx = math.cos(a_) * urx + math.sin(a_) * uhx; dy = math.cos(a_) * ury + math.sin(a_) * uhy
                elif t_ < Pe[3]:                                 # gravity turn: thrust along the air-relative velocity
                    va = math.sqrt(ax * ax + ay * ay)
                    dx = ax / va; dy = ay / va
                elif t_ < Pe[4] or st[e, 2] > 0.5:               # tilt arrested: hold the attitude in space
                    dx = st[e, 0]; dy = st[e, 1]
                else:                                            # explicit guidance to the target orbit
                    aT = st[e, 3]; c_ = st[e, 5]; md = st[e, 6]
                    dv = Pe[6] - vh
                    if aT <= 0.0 or dv <= 0.0:
                        dx = st[e, 0]; dy = st[e, 1]
                    else:
                        if md > 0.0 and c_ > 0.0:
                            tgo = mass[b] / md * (1.0 - math.exp(-dv / c_))
                        else:
                            tgo = dv / aT
                        if tgo > Pe[7]:
                            a0 = 6.0 * (Pe[5] - r) / (tgo * tgo) - 4.0 * vr / tgo
                            ar = a0 + mu_p / (r * r) - vh * vh / r
                            sn = ar / aT
                            if sn > 0.95:
                                sn = 0.95
                            elif sn < -0.95:
                                sn = -0.95
                            st[e, 4] = sn
                        sn = st[e, 4]
                        cs = math.sqrt(1.0 - sn * sn)
                        dx = sn * urx + cs * uhx; dy = sn * ury + cs * uhy
                    if math.sqrt(v[3 * b] ** 2 + v[3 * b + 1] ** 2) >= Pe[6]:
                        st[e, 2] = 1.0                           # target speed reached: guidance cutoff
                        cut_t[e] = t_
                st[e, 0] = dx; st[e, 1] = dy
                Fe[e] = math.degrees(math.asin(max(-1.0, min(1.0, dx * urx + dy * ury))))   # pitch above horizontal
                se[e] = Rp * (math.atan2(q[3 * b], q[3 * b + 1]) - th0[b] - planet[2] * t_)   # surface range over the turning ground
                aux[e, 0] = r - Rp
                aux[e, 1] = math.sqrt(v[3 * b] ** 2 + v[3 * b + 1] ** 2)
                aux[e, 2] = math.degrees(math.atan2(vr, vh))
                aux[e, 3] = math.sqrt(ax * ax + ay * ay)
            for e in range(ne):
                te = typ[e]
                if te == TYP_ROCKET:
                    b = ba[e]; Pe = P[e]
                    off = Pe[4]
                    se_ = steer_of[b]
                    if Pe[7] > 0.5 and se_ >= 0 and st[se_, 2] > 0.5:
                        off = min(off, cut_t[se_])               # guidance cutoff
                    f = _ramp(t_, Pe[3], off, Pe[5], Pe[6]) if active[e] == 1 else 0.0
                    if Pe[8] > 0.5:                                  # throttled by landing guidance
                        f = f * gthr[b] if gon[b] == 1 else 0.0
                    elif Pe[11] > 0.5:                               # a set (or controlled) throttle level
                        f = f * min(max(Pe[10], 0.0), 1.0)
                    if Pe[9] > 0.0:                                  # a propellant load: the engine stops when it is gone
                        left = Pe[9] - st[e, 0]
                        if left <= 0.0:
                            f = 0.0
                        elif f * Pe[2] * dt > left:
                            f = left / (Pe[2] * dt)
                        se[e] = left
                    r, urx, ury, uhx, uhy, vr, vh, ax, ay = _local(q, v, b, planet)
                    if mu_p > 0.0 and planet[3] > 0.5:
                        pa, rho, Ta = atmosphere(r - Rp)
                    else:
                        pa = 0.0
                    F = f * (Pe[0] - pa * Pe[1])
                    if F < 0.0:
                        F = 0.0
                    if Pe[8] > 0.5 and gon[b] == 1:
                        dx = gdx[b]; dy = gdy[b]
                    elif se_ >= 0:
                        dx = st[se_, 0]; dy = st[se_, 1]
                    else:
                        dx = urx; dy = ury
                    frx[e] = F * dx; fry[e] = F * dy
                    Q[3 * b] += frx[e]; Q[3 * b + 1] += fry[e]
                    dmv[e] = f * Pe[2] * dt
                    st[e, 0] += dmv[e]
                    Tb[b] += F; Mb[b] += f * Pe[2]
                    Fe[e] = F; aux[e, 0] = f * Pe[2]; aux[e, 1] = pa; aux[e, 2] = mass[b]; aux[e, 3] = f
                elif te == TYP_JETTISON:
                    dmv[e] = 0.0
                    if st[e, 0] < 0.5 and t_ >= P[e, 0] and active[e] == 1:
                        st[e, 0] = 1.0; dmv[e] = P[e, 1]
                    Fe[e] = st[e, 0] * P[e, 1]
                elif te == TYP_AERO:
                    b = ba[e]
                    r, urx, ury, uhx, uhy, vr, vh, ax, ay = _local(q, v, b, planet)
                    pa, rho, Ta = atmosphere(r - Rp)
                    va = math.sqrt(ax * ax + ay * ay)
                    Mach = va / math.sqrt(1.4 * 287.053 * Ta)
                    o = tab_off[e]; L = tab_len[e]
                    qd = 0.5 * rho * va * va
                    Dg = qd * P[e, 0] * _cd_at(Mach, tab_x[o:o + L], tab_F[o:o + L]) if active[e] == 1 else 0.0
                    if va > 0.0:
                        frx[e] = -Dg * ax / va; fry[e] = -Dg * ay / va
                    else:
                        frx[e] = 0.0; fry[e] = 0.0
                    Q[3 * b] += frx[e]; Q[3 * b + 1] += fry[e]
                    Fe[e] = Dg; se[e] = r - Rp; aux[e, 0] = qd; aux[e, 1] = Mach; aux[e, 2] = rho; aux[e, 3] = pa
            for e in range(ne):                                 # what guidance reads next step
                if typ[e] == TYP_STEER:
                    b = ba[e]
                    st[e, 3] = Tb[b] / mass[b] if mass[b] > 0.0 else 0.0
                    st[e, 5] = Tb[b] / Mb[b] if Mb[b] > 0.0 else 0.0
                    st[e, 6] = Mb[b]
        # elements: contacts and the rest first, then friction on contacts (needs this step's normal force)
        for sweep in range(2):
            for e in range(ne):
                if (typ[e] == 10) != (sweep == 1) or typ[e] == 14 or typ[e] == 15 or \
                        (typ[e] >= TYP_ROCKET and typ[e] <= TYP_LAND):
                    continue
                m_ = mk[e]
                if m_ == MK_SLIP:
                    c = ref[e]
                    for k in range(6):
                        geo[e, k] = geo[c, k]
                    sd = _rate(MK_EDGE, e, q, v, ba, bb, C, geo, 1)
                    s = se[e] + sd * dt
                    ok = inc[c] == 1 and Fe[c] > 0.0
                elif m_ == MK_BEAM:
                    s = se[e]; sd = 0.0; ok = True          # a beam measures itself below
                else:
                    s, sd, ok = _measure(m_, e, q, v, ba, bb, C, s0, GA, GB, GX, geo)
                    # entry rule: an edge contact can only START by crossing the line from outside. A point that
                    # arrives already deep "inside" from beyond the segment's end went round the corner: no contact.
                    # A shallow one (under 0.1% of the segment) is handed over from a neighbouring segment: allowed.
                    if m_ == MK_EDGE and ok and inc[e] == 0 and se[e] > 0.0 and not first:
                        if s > 1e-3 * math.hypot(GB[e, 2] - GB[e, 0], GB[e, 3] - GB[e, 1]):
                            ok = False
                sold = se[e]
                se[e] = s; sde[e] = sd
                if active[e] == 0 or not ok:
                    if exact[e] == 1:
                        if inc[e] == 1:
                            if s <= 0.0:
                                A[e] += 0.5 * Fp[e] * (0.0 - sold)   # unloaded on the way out to s = 0
                            else:
                                A[e] += 0.5 * Fp[e] * (s - sold)
                                if S1[e] > 0.0 and D[e] >= 0.0:
                                    D[e] += S1[e]               # lost while still pressed (slid off an end, tore
                                                                # through): the elastic energy leaves as vibration
                    elif typ[e] != 9:
                        A[e] += 0.5 * (0.0 - Fp[e]) * dsp[e]     # close the last loaded interval
                    Fe[e] = 0.0; inc[e] = 0; Fp[e] = 0.0; dsp[e] = 0.0
                    S1[e] = 0.0
                    continue
                was = inc[e]
                inc[e] = 1
                if m_ == MK_BEAM:                        # co-rotational beam between two bodies
                    if active[e] == 0:
                        Fe[e] = 0.0; continue
                    o = tab_off[e]; Lt = tab_len[e]
                    N, M1, M2, L, th1, th2, ax, ay, bx, by, e1x, e1y, stored, dpl, ddp = _beam2(
                        e, q, ba, bb, GA, GB, P[e], st[e], tab_x[o:o + Lt], tab_F[o:o + Lt], gpv, dt, first)
                    V = (M1 + M2) / L
                    e2x = -e1y; e2y = e1x
                    fax = N * e1x - V * e2x; fay = N * e1y - V * e2y
                    _push(Q, q, ba[e], ax, ay, fax, fay); _push(Q, q, bb[e], bx, by, -fax, -fay)
                    if ba[e] >= 0:
                        Q[3 * ba[e] + 2] -= M1
                    if bb[e] >= 0:
                        Q[3 * bb[e] + 2] -= M2
                    if first:
                        S0[e] = stored
                    else:
                        A[e] += 0.5 * ((fpv[e, 0] + N) * (L - gpv[e, 0]) + (fpv[e, 1] + M1) * (th1 - gpv[e, 1]) +
                                       (fpv[e, 2] + M2) * (th2 - gpv[e, 2]))
                    S1[e] = stored; D[e] += dpl + ddp
                    gpv[e, 0] = L; gpv[e, 1] = th1; gpv[e, 2] = th2; fpv[e, 0] = N; fpv[e, 1] = M1; fpv[e, 2] = M2
                    Fe[e] = N; se[e] = L - P[e, 4]; inc[e] = 1
                    aux[e, 0] = M1 if abs(M1) >= abs(M2) else M2
                    aux[e, 1] = 0.0; aux[e, 2] = V
                    aux[e, 3] = max(abs(st[e, 0]), abs(st[e, 1]))
                    continue
                if typ[e] == 9:                         # pin: vector spring-damper between two points
                    ax = geo[e, 0]; ay = geo[e, 1]; bx = geo[e, 2]; by = geo[e, 3]
                    vax, vay = _vel(q, v, ba[e], ax, ay); vbx, vby = _vel(q, v, bb[e], bx, by)
                    gx = bx - ax; gy = by - ay; gvx = vbx - vax; gvy = vby - vay
                    fx = P[e, 0] * gx + P[e, 1] * gvx; fy = P[e, 0] * gy + P[e, 1] * gvy
                    _push(Q, q, ba[e], ax, ay, fx, fy); _push(Q, q, bb[e], bx, by, -fx, -fy)
                    F = math.sqrt(fx * fx + fy * fy)
                    stored = 0.5 * P[e, 0] * (gx * gx + gy * gy)
                    if first:
                        S0[e] = stored
                    S1[e] = stored
                    D[e] += P[e, 1] * (gvx * gvx + gvy * gvy) * dt
                    if not first:                        # trapezoid on the true change of the gap vector
                        A[e] += 0.5 * ((fpv[e, 0] + fx) * (gx - gpv[e, 0]) + (fpv[e, 1] + fy) * (gy - gpv[e, 1]))
                    gpv[e, 0] = gx; gpv[e, 1] = gy; fpv[e, 0] = fx; fpv[e, 1] = fy
                    Fe[e] = F
                    continue
                if typ[e] == 10:                        # Coulomb friction on the contact tangent
                    Fn = Fe[ref[e]]
                    F = P[e, 0] * Fn * math.tanh(sd / P[e, 1])
                    # stick limit: one step of friction may at most stop the slip, never reverse it. Keeps the
                    # regularised law stable at any v_eps (it is a very stiff damper near zero slip).
                    tx = -geo[e, 5]; ty = geo[e, 4]
                    inv = _inv_mass_along(q, minv, ba[e], geo[e, 0], geo[e, 1], tx, ty) + \
                        _inv_mass_along(q, minv, bb[e], geo[e, 2], geo[e, 3], tx, ty)
                    if inv > 0.0:
                        Fcap = abs(sd) / (inv * dt)
                        if abs(F) > Fcap:
                            F = Fcap if F > 0.0 else -Fcap
                    D[e] += F * sd * dt
                    stored = 0.0
                else:
                    o = tab_off[e]; L = tab_len[e]
                    F, dd, din, stored = _force(typ[e], one_sided[e], P[e], s, sd, tab_k[e], tab_x[o:o + L],
                                                tab_F[o:o + L], st[e], aux[e], dt)
                    if dd >= 0.0:
                        D[e] += dd
                    else:
                        D[e] = -1.0
                    if din >= 0.0:
                        IN[e] += din
                    else:
                        IN[e] = -1.0
                if first:
                    S0[e] = stored
                S1[e] = stored
                if exact[e] == 1:
                    if was == 1:
                        A[e] += 0.5 * (Fp[e] + F) * (s - sold)
                    elif not first:
                        A[e] += 0.5 * F * (s if s > 0.0 else 0.0)  # contact just made: loaded from s = 0
                    Fp[e] = F
                else:
                    A[e] += 0.5 * (F - Fp[e]) * dsp[e]
                Fe[e] = F
                if m_ == MK_LIN:
                    for k in range(nq):
                        if C[e, k] != 0.0:
                            Q[k] -= C[e, k] * F
                else:
                    nx = geo[e, 4]; ny = geo[e, 5]
                    if m_ == MK_SLIP:
                        nx, ny = -ny, nx
                    _push(Q, q, ba[e], geo[e, 0], geo[e, 1], F * nx, F * ny)
                    _push(Q, q, bb[e], geo[e, 2], geo[e, 3], -F * nx, -F * ny)
        first = False
        for k in range(nq):
            vst[k] = v[k]
            v[k] += dt * Q[k] * minv[k]
        # rigid impacts: sequential impulses on this step's velocities (restitution e, Coulomb mu)
        for k in range(nq):
            Dq[k] = 0.0
        nc = 0
        if nimp > 0 or nrig > 0:
            for e in range(ne):
                if typ[e] == 15:
                    lam[e, 0] = 0.0; lam[e, 1] = 0.0
            nc = 0
            for e in range(ne):
                if typ[e] != 14:
                    continue
                held[e] = jn[e]; Fe[e] = 0.0; aux[e, 0] = 0.0; aux[e, 2] = 0.0; jn[e] = 0.0
                if active[e] == 0:
                    continue
                s, sd, ok = _measure(mk[e], e, q, v, ba, bb, C, s0, GA, GB, GX, geo)
                se[e] = s
                if mk[e] == MK_EDGE:
                    fr = _edge_span(e, q, ba, bb, GA, GB)
                    if fr < 0.0 or fr > 1.0 or s >= GB[e, 4]:
                        continue
                if sd <= 0.0 or s + sd * dt <= 0.0:
                    continue
                cand[nc] = e; nc += 1
                tauv[e] = min(1.0, -s / (sd * dt)) if s < 0.0 else 0.0
                # restitution only for a new hit: a contact pressed last step (resting, pushed) gets no bounce,
                # or the speed gravity adds each step would bounce back as chatter and creep
                tgt[e] = -P[e, 0] * sd if (sd > P[e, 2] and held[e] <= 0.0) else 0.0
                jn[e] = 0.0; jt[e] = 0.0
                if mk[e] == MK_LIN:
                    w_ = 0.0
                    for k in range(nq):
                        w_ += C[e, k] * C[e, k] * minv[k]
                    invn[e] = w_; invt[e] = 0.0
                else:
                    invn[e] = (_inv_mass_along(q, minv, ba[e], geo[e, 0], geo[e, 1], geo[e, 4], geo[e, 5]) +
                               _inv_mass_along(q, minv, bb[e], geo[e, 2], geo[e, 3], geo[e, 4], geo[e, 5]))
                    invt[e] = (_inv_mass_along(q, minv, ba[e], geo[e, 0], geo[e, 1], -geo[e, 5], geo[e, 4]) +
                               _inv_mass_along(q, minv, bb[e], geo[e, 2], geo[e, 3], -geo[e, 5], geo[e, 4]))
                if invn[e] > 0.0:
                    aux[e, 3] += 0.5 * sd * sd / invn[e]  # energy brought into the hit (ledger scale)
                ucd[e] = sd; u0d[e] = _rate(MK_LIN if mk[e] == MK_LIN else MK_EDGE, e, q, vst, ba, bb, C, geo, 0); lst[e] = D[e]
            for c in range(nc):
                lossN[cand[c]] = 0.0; lossT[cand[c]] = 0.0
            for it in range(16):
                for c in range(nc):
                    e = cand[c]; mke = MK_LIN if mk[e] == MK_LIN else MK_EDGE
                    tdt = tauv[e] * dt
                    if invn[e] > 0.0:
                        u = _rate(mke, e, q, v, ba, bb, C, geo, 0)
                        new = jn[e] + (u - tgt[e]) / invn[e]
                        if new < 0.0:
                            new = 0.0
                        dj = new - jn[e]; jn[e] = new
                        if dj != 0.0:
                            lossN[e] += _impulse2(e, dj, 0, mke, q, v, minv, ba, bb, C, geo, Dq, tdt)
                    if P[e, 1] > 0.0 and invt[e] > 0.0 and mke != MK_LIN:
                        ut = _rate(mke, e, q, v, ba, bb, C, geo, 1)
                        lim = P[e, 1] * jn[e]
                        new = jt[e] + ut / invt[e]
                        if new > lim:
                            new = lim
                        elif new < -lim:
                            new = -lim
                        dj = new - jt[e]; jt[e] = new
                        if dj != 0.0:
                            lossT[e] += _impulse2(e, dj, 1, mke, q, v, minv, ba, bb, C, geo, Dq, tdt)
                for e in range(ne):              # rigid joints: relative velocity along every row to zero
                    if typ[e] != 15 or active[e] == 0:
                        continue
                    nr = _rows2(e, mk[e], q, ba, bb, C, s0, GA, GB, GX, refc, Jr, cps)
                    for r in range(nr):
                        u = 0.0; kk = 0.0; u0 = 0.0
                        for k in range(nq):
                            if Jr[r, k] != 0.0:
                                u += Jr[r, k] * v[k]; kk += Jr[r, k] * Jr[r, k] * minv[k]; u0 += Jr[r, k] * vst[k]
                        if kk <= 0.0 or u == 0.0:
                            continue
                        dl = -u / kk
                        for k in range(nq):
                            if Jr[r, k] != 0.0:
                                v[k] += minv[k] * Jr[r, k] * dl
                        lam[e, r] += dl
                        if it == 0:
                            # energy really lost: only a relative speed the joint already had at the start of the
                            # step (a catch). The part this step's forces just added is the joint force doing its
                            # job (no motion along the row, no work), not a loss.
                            loss = 0.5 * u0 * u0 / kk
                            D[e] += loss; A[e] += loss
            if nrig > 0:                                 # joints: finish the velocity rows with one direct solve
                _jsolve2(0, ne, nq, typ, active, mk, q, v, minv, ba, bb, C, s0, GA, GB, GX, refc, Jr, cps, Jold, lam,
                         aux, JA, JO, rhs, rje, rjr, dt, False)
            for c in range(nc):
                e = cand[c]
                Fe[e] = jn[e] / dt; aux[e, 0] = jn[e]; aux[e, 2] = jt[e]
                if held[e] > 0.0 and mk[e] != MK_LIN:
                    # a held contact's friction dissipates only while it slides: impulse times the slip left after
                    # the solve (sticking leaves none). The tangential speed the other contacts' pushes put there
                    # and friction took back out was never motion, so it is not a loss.
                    lossT[e] = abs(jt[e] * _rate(MK_EDGE, e, q, v, ba, bb, C, geo, 1))
            _book_hits(cand, nc, jn, held, lossN, lossT, D, A)

        for e in range(ne):
            if typ[e] == 14:
                aux[e, 1] = D[e]; inc[e] = 1 if jn[e] > 0.0 else 0                         # heat so far (J): every joule the impacts took
        # work this step for linear measures and friction (rate form, new velocities, this step's geometry)
        for e in range(ne):
            if exact[e] == 1 or typ[e] == 9 or mk[e] == MK_BEAM or typ[e] == 14 or typ[e] == 15 or \
                    (typ[e] >= TYP_ROCKET and typ[e] <= TYP_LAND):
                continue
            if Fe[e] == 0.0 and Fp[e] == 0.0:
                Fp[e] = 0.0; dsp[e] = 0.0
                continue
            which = 1 if mk[e] == MK_SLIP else 0
            ds = _rate(MK_LIN if mk[e] == MK_LIN else MK_EDGE, e, q, v, ba, bb, C, geo, which) * dt
            if mk[e] == MK_LIN and nimp > 0:
                for k in range(nq):
                    ds += C[e, k] * Dq[k]
            A[e] += Fe[e] * ds
            Fp[e] = Fe[e]; dsp[e] = ds
        if nrig > 0:
            for e in range(ne):
                if typ[e] == 15:
                    nrow[e] = _rows2(e, mk[e], q, ba, bb, C, s0, GA, GB, GX, refc, Jr, cps)
                    for r in range(nrow[e]):
                        for k in range(nq):
                            Jold[e, r, k] = Jr[r, k]
        for k in range(nq):
            q[k] += dt * v[k] + Dq[k]
        if incr:
            for b in range(nb):
                Wg += (gfx[b] * v[3 * b] + gfy[b] * v[3 * b + 1]) * dt
        if nveh > 0:
            for e in range(ne):
                te = typ[e]
                if te == TYP_ROCKET or te == TYP_AERO:
                    b = ba[e]
                    w_ = (frx[e] * v[3 * b] + fry[e] * v[3 * b + 1]) * dt     # work the force did on the vehicle
                    A[e] -= w_
                    if te == TYP_AERO:
                        D[e] -= w_
                    else:
                        IN[e] += w_
                if (te == TYP_ROCKET or te == TYP_JETTISON) and dmv[e] > 0.0:
                    b = ba[e]
                    if dmv[e] > mass[b] * (1.0 - 1e-9):
                        dmv[e] = mass[b] * (1.0 - 1e-9)
                    ke = 0.5 * (v[3 * b] ** 2 + v[3 * b + 1] ** 2) * dmv[e]   # kinetic energy the lost mass carries off
                    A[e] += ke; D[e] += ke
                    mass[b] -= dmv[e]
                    if minv[3 * b] > 0.0:
                        minv[3 * b] = 1.0 / mass[b]
                    if minv[3 * b + 1] > 0.0:
                        minv[3 * b + 1] = 1.0 / mass[b]
                    dmv[e] = 0.0
        if nrig > 0:                                 # rigid joints: Newton steps of the direct position solve
            for it in range(4):
                if _jsolve2(1, ne, nq, typ, active, mk, q, v, minv, ba, bb, C, s0, GA, GB, GX, refc, Jr, cps, Jold, lam,
                            aux, JA, JO, rhs, rje, rjr, dt, it == 0) < 1e-12:
                    break
            for it in range(1):
                _jsolve2(0, ne, nq, typ, active, mk, q, v, minv, ba, bb, C, s0, GA, GB, GX, refc, Jr, cps, Jold, lam,
                         aux, JA, JO, rhs, rje, rjr, dt, False)
            for it in range(4):                     # row-by-row clean-up of anything the direct solve left
                mx_ = 0.0
                for e in range(ne):
                    if typ[e] != 15 or active[e] == 0:
                        continue
                    nr = _rows2(e, mk[e], q, ba, bb, C, s0, GA, GB, GX, refc, Jr, cps)
                    for r in range(nr):
                        # SHAKE: move along the start-of-step constraint directions M^-1 J_old^T (symplectic)
                        kk = 0.0
                        for k in range(nq):
                            kk += Jr[r, k] * Jold[e, r, k] * minv[k]
                        if kk <= 0.0:
                            continue
                        if it == 0:
                            aux[e, 3] = max(aux[e, 3], abs(cps[r]))       # largest drift before correction
                        if abs(cps[r]) > mx_:
                            mx_ = abs(cps[r])
                        dl = -cps[r] / kk
                        lam[e, r] += dl / dt                 # the correction is an impulse dl / dt
                        for k in range(nq):
                            if Jold[e, r, k] != 0.0:
                                q[k] += minv[k] * Jold[e, r, k] * dl
                                v[k] += minv[k] * Jold[e, r, k] * dl / dt   # velocities follow the corrected path
                        if r == 0 and nr == 2:                            # pin: refresh the y row after moving
                            nr = _rows2(e, mk[e], q, ba, bb, C, s0, GA, GB, GX, refc, Jr, cps)
                if mx_ < 1e-13:
                    break
            for it in range(4):                     # RATTLE: velocities tangent to the constraints at the new
                mu_ = 0.0
                for e in range(ne):                  # positions (numerical; any energy change shows in the ledger)
                    if typ[e] != 15 or active[e] == 0:
                        continue
                    nr = _rows2(e, mk[e], q, ba, bb, C, s0, GA, GB, GX, refc, Jr, cps)
                    for r in range(nr):
                        u = 0.0; kk = 0.0
                        for k in range(nq):
                            if Jr[r, k] != 0.0:
                                u += Jr[r, k] * v[k]; kk += Jr[r, k] * Jr[r, k] * minv[k]
                        if abs(u) > mu_:
                            mu_ = abs(u)
                        if kk > 0.0 and u != 0.0:
                            lam[e, r] -= u / kk
                            for k in range(nq):
                                if Jr[r, k] != 0.0:
                                    v[k] -= minv[k] * Jr[r, k] * u / kk
                for c in range(nc):                  # contacts that pushed this step keep pushing: no approach
                    e = cand[c]                      # speed is left for the next step (not booked: numerical)
                    if jn[e] <= 0.0 or invn[e] <= 0.0:
                        continue
                    mke = MK_LIN if mk[e] == MK_LIN else MK_EDGE
                    u = _rate(mke, e, q, v, ba, bb, C, geo, 0)
                    if u > 0.0:
                        _impulse2(e, u / invn[e], 0, mke, q, v, minv, ba, bb, C, geo, Dq, 0.0)
                        if u > mu_:
                            mu_ = u
                if mu_ < 1e-13 and it > 0:
                    break
            for e in range(ne):
                if typ[e] == 15:
                    s_, sd_, ok_ = _measure(mk[e], e, q, v, ba, bb, C, s0, GA, GB, GX, geo)
                    se[e] = s_
                    # joint force: every impulse the joint gave this step (catch, SHAKE, RATTLE) over the step
                    if mk[e] == MK_PIN:
                        Fe[e] = math.sqrt(lam[e, 0] ** 2 + lam[e, 1] ** 2) / dt
                    else:
                        Fe[e] = -lam[e, 0] / dt          # same sign as a spring: pulls the measure back
                    aux[e, 0] = lam[e, 0] / dt; aux[e, 1] = lam[e, 1] / dt; aux[e, 2] = D[e]
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
                    sens[k] = se[kk]
                elif src == 8:
                    sens[k] = math.sqrt(v[3 * kk] ** 2 + v[3 * kk + 1] ** 2)
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
            sv = se[e] if (inc[e] == 1 or mk[e] < MK_EDGE or mk[e] == MK_PIN) else 0.0
            if sv < TR[e, 2]: TR[e, 2] = sv
            if sv > TR[e, 3]: TR[e, 3] = sv
            for k in range(NAUX):
                if aux[e, k] < TR[e, 4 + 2 * k]: TR[e, 4 + 2 * k] = aux[e, k]
                if aux[e, k] > TR[e, 5 + 2 * k]: TR[e, 5 + 2 * k] = aux[e, k]
        for k in range(nq):
            if q[k] < BT[k, 0]: BT[k, 0] = q[k]
            if q[k] > BT[k, 1]: BT[k, 1] = q[k]
            if v[k] < BT[k, 2]: BT[k, 2] = v[k]
            if v[k] > BT[k, 3]: BT[k, 3] = v[k]
        for b in range(nb):
            sp = math.sqrt(v[3 * b] ** 2 + v[3 * b + 1] ** 2)
            if sp < SP[b, 0]: SP[b, 0] = sp
            if sp > SP[b, 1]: SP[b, 1] = sp
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
                elif src == 8:
                    x = math.sqrt(v[3 * k] ** 2 + v[3 * k + 1] ** 2)
                elif src == 50:
                    x = sig[k]
                else:
                    x = aux[k, src - 4]
                if (cross_dir[c] > 0 and x >= cross_thr[c]) or (cross_dir[c] < 0 and x <= cross_thr[c]):
                    t_cross[c] = tn
        if i % 4 == 0:
            ke_ = 0.0
            for k in range(nq):
                if minv[k] > 0.0:
                    ke_ += 0.5 * v[k] * v[k] / minv[k]
            if ke_ > KEpk: KEpk = ke_
        if nimp > 0:
            jcnt += 1
            for e in range(ne):
                if typ[e] == 14:
                    jacc[e] += Fe[e]
        if npart > 0:
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
            for k in range(nq):
                LOGB[j, k] = q[k]; LOGB[j, nq + k] = v[k]
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
    for b in range(nb):
        if mass[b] > 0.0 and not incr:
            px, py = _pt(q, b, com[b, 0], com[b, 1])
            Wg += mass[b] * (grav[0] * (px - PC0[b, 0]) + grav[1] * (py - PC0[b, 1]))
    KE1 = 0.0
    for k in range(nq):
        if minv[k] > 0.0:
            KE1 += 0.5 * v[k] * v[k] / minv[k]
    for e in range(min(ne, cctlD.shape[0])):
        if cctlD[e] == 1:
            D[e] = -1.0                          # a controlled spring: its stored energy is no longer its own law
    return (q, v, KE0, KE1, A, D, IN, S0, S1, TR, BT, SP, aux, t_cross, LOGT[:j], LOGB[:j], LOGE[:j], Fe, se, Wg, KEpk,
            cpout, CTR, LOGC[:j], sig, SG, LOGS[:j])
