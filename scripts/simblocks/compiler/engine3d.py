"""3D engine: rigid bodies with translation (any of x, y, z) and rotation that is either
  none   (the body only translates),
  spin   (about one fixed world axis through its pivot or centre of mass; the axis is any vector you give), or
  free   (full 3D rotation with an inertia tensor: precession, nutation, gyroscopic moments all come out of it).

The rule is the 1D and 2D one: an element returns a force F along a scalar measure s; the measure gives two material
points p_A, p_B and a unit direction n with ds/dt = n . (v_B(p_B) - v_A(p_A)); A is pushed at p_A with +F n, B at p_B
with -F n. Torques come from where the points really are.

Free rotation is integrated on the world angular momentum L (dL/dt = torque exactly, so a torque-free body keeps L
to round-off), with omega = R I^-1 R^T L, and the orientation advanced by a midpoint rule (second order): omega is
read at the half-step orientation and the body turned by exp(omega dt). Spin bodies
integrate one rate about their axis. Translation is semi-implicit Euler as in 1D and 2D.

Per body, 12 tracked quantities: x y z vx vy vz speed angle omega wx wy wz
(angle: spin angle of a spin body; omega: its spin rate, or |omega| of a free body; wx wy wz: world angular velocity).

Measure kinds (mk): 0 linear over coordinates [x, y, z, spin] per body | 1 distance | 2 along | 3 contact of a point or
sphere with a finite plane patch | 4 sphere against sphere | 5 pin (ball joint, vector spring-damper) | 6 slip
(Coulomb friction on another contact's tangent plane)."""
import math
import numpy as np
from numba import njit

from .circuit import circuit_step, NOUT as NOUT_C
from .control import control_step
from .adaptive import pick_dt

from .engine2d import _book_hits
from .engine1d import _force, NP, NAUX, motor_k, motor_scale, drive_books

MK_LIN, MK_DIST, MK_ALONG, MK_PLANE, MK_SPH, MK_PIN, MK_SLIP, MK_ROT, MK_TWIST, MK_BEAM = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9


@njit(cache=True, nogil=True, inline="always")
def _yield3(acc, tx, tF):
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
def _end_rot(E, R, b, GB, off, e):
    """Rotation vector (beam-frame components x, y, z) taking the beam frame E to the end triad of body b."""
    tx = 0.0; ty = 0.0; tz = 0.0
    for i in range(3):
        l0 = GB[e, off + i]; l1 = GB[e, off + 3 + i]; l2 = GB[e, off + 6 + i]     # column i of the stored triad
        if b >= 0:
            c0 = R[b, 0, 0] * l0 + R[b, 0, 1] * l1 + R[b, 0, 2] * l2
            c1 = R[b, 1, 0] * l0 + R[b, 1, 1] * l1 + R[b, 1, 2] * l2
            c2 = R[b, 2, 0] * l0 + R[b, 2, 1] * l1 + R[b, 2, 2] * l2
        else:
            c0 = l0; c1 = l1; c2 = l2
        tx += 0.5 * (E[1, i] * c2 - E[2, i] * c1); ty += 0.5 * (E[2, i] * c0 - E[0, i] * c2)
        tz += 0.5 * (E[0, i] * c1 - E[1, i] * c0)
    return (tx * E[0, 0] + ty * E[1, 0] + tz * E[2, 0], tx * E[0, 1] + ty * E[1, 1] + tz * E[2, 1],
            tx * E[0, 2] + ty * E[1, 2] + tz * E[2, 2])
R_NONE, R_SPIN, R_FREE = 0, 1, 2
NBQ = 12


@njit(cache=True, nogil=True, inline="always")
def _qmat(qw, qx, qy, qz, R):
    R[0, 0] = 1 - 2 * (qy * qy + qz * qz); R[0, 1] = 2 * (qx * qy - qz * qw); R[0, 2] = 2 * (qx * qz + qy * qw)
    R[1, 0] = 2 * (qx * qy + qz * qw); R[1, 1] = 1 - 2 * (qx * qx + qz * qz); R[1, 2] = 2 * (qy * qz - qx * qw)
    R[2, 0] = 2 * (qx * qz - qy * qw); R[2, 1] = 2 * (qy * qz + qx * qw); R[2, 2] = 1 - 2 * (qx * qx + qy * qy)


@njit(cache=True, nogil=True)
def _rotate_quat(quat, b, wx, wy, wz, dt):
    """quat[b] <- exp(w dt) * quat[b] (world-frame angular velocity), exact for constant w over the step."""
    w = math.sqrt(wx * wx + wy * wy + wz * wz)
    if w * dt < 1e-300:
        return
    h = 0.5 * w * dt; s = math.sin(h) / w; c = math.cos(h)
    dw, dx, dy, dz = c, wx * s, wy * s, wz * s
    qw, qx, qy, qz = quat[b, 0], quat[b, 1], quat[b, 2], quat[b, 3]
    nw = dw * qw - dx * qx - dy * qy - dz * qz
    nx = dw * qx + dx * qw + dy * qz - dz * qy
    ny = dw * qy - dx * qz + dy * qw + dz * qx
    nz = dw * qz + dx * qy - dy * qx + dz * qw
    L = math.sqrt(nw * nw + nx * nx + ny * ny + nz * nz)
    quat[b, 0] = nw / L; quat[b, 1] = nx / L; quat[b, 2] = ny / L; quat[b, 3] = nz / L


@njit(cache=True, nogil=True, inline="always")
def _pt(X, R, b, lx, ly, lz):
    if b < 0:
        return lx, ly, lz
    return (X[b, 0] + R[b, 0, 0] * lx + R[b, 0, 1] * ly + R[b, 0, 2] * lz,
            X[b, 1] + R[b, 1, 0] * lx + R[b, 1, 1] * ly + R[b, 1, 2] * lz,
            X[b, 2] + R[b, 2, 0] * lx + R[b, 2, 1] * ly + R[b, 2, 2] * lz)


@njit(cache=True, nogil=True, inline="always")
def _dir(R, b, lx, ly, lz):
    if b < 0:
        return lx, ly, lz
    return (R[b, 0, 0] * lx + R[b, 0, 1] * ly + R[b, 0, 2] * lz,
            R[b, 1, 0] * lx + R[b, 1, 1] * ly + R[b, 1, 2] * lz,
            R[b, 2, 0] * lx + R[b, 2, 1] * ly + R[b, 2, 2] * lz)


@njit(cache=True, nogil=True, inline="always")
def _vel(X, V, W, b, px, py, pz):
    if b < 0:
        return 0.0, 0.0, 0.0
    rx = px - X[b, 0]; ry = py - X[b, 1]; rz = pz - X[b, 2]
    return (V[b, 0] + W[b, 1] * rz - W[b, 2] * ry, V[b, 1] + W[b, 2] * rx - W[b, 0] * rz,
            V[b, 2] + W[b, 0] * ry - W[b, 1] * rx)


@njit(cache=True, nogil=True, inline="always")
def _push(F, T, X, b, px, py, pz, fx, fy, fz):
    if b < 0:
        return
    F[b, 0] += fx; F[b, 1] += fy; F[b, 2] += fz
    rx = px - X[b, 0]; ry = py - X[b, 1]; rz = pz - X[b, 2]
    T[b, 0] += ry * fz - rz * fy; T[b, 1] += rz * fx - rx * fz; T[b, 2] += rx * fy - ry * fx


@njit(cache=True, nogil=True)
def _inv_along(X, R, minv, rmode, Iinv, axis, Iax, b, px, py, pz, nx, ny, nz):
    """Inverse effective mass of body b at world point p along unit direction n."""
    if b < 0:
        return 0.0
    out = minv[b, 0] * nx * nx + minv[b, 1] * ny * ny + minv[b, 2] * nz * nz
    rx = px - X[b, 0]; ry = py - X[b, 1]; rz = pz - X[b, 2]
    cx = ry * nz - rz * ny; cy = rz * nx - rx * nz; cz = rx * ny - ry * nx      # r x n
    if rmode[b] == R_SPIN:
        t = cx * axis[b, 0] + cy * axis[b, 1] + cz * axis[b, 2]
        out += Iax[b] * t * t
    elif rmode[b] == R_FREE:
        # (r x n)^T  R Iinv R^T (r x n)
        bx = R[b, 0, 0] * cx + R[b, 1, 0] * cy + R[b, 2, 0] * cz
        by = R[b, 0, 1] * cx + R[b, 1, 1] * cy + R[b, 2, 1] * cz
        bz = R[b, 0, 2] * cx + R[b, 1, 2] * cy + R[b, 2, 2] * cz
        out += (bx * (Iinv[b, 0, 0] * bx + Iinv[b, 0, 1] * by + Iinv[b, 0, 2] * bz) +
                by * (Iinv[b, 1, 0] * bx + Iinv[b, 1, 1] * by + Iinv[b, 1, 2] * bz) +
                bz * (Iinv[b, 2, 0] * bx + Iinv[b, 2, 1] * by + Iinv[b, 2, 2] * bz))
    return out


@njit(cache=True, nogil=True)
def _omega(R, rmode, Iinv, axis, Lw, wsp, W, nb):
    for b in range(nb):
        if rmode[b] == R_FREE:
            # body-frame L, then I^-1, then back to world
            lx = R[b, 0, 0] * Lw[b, 0] + R[b, 1, 0] * Lw[b, 1] + R[b, 2, 0] * Lw[b, 2]
            ly = R[b, 0, 1] * Lw[b, 0] + R[b, 1, 1] * Lw[b, 1] + R[b, 2, 1] * Lw[b, 2]
            lz = R[b, 0, 2] * Lw[b, 0] + R[b, 1, 2] * Lw[b, 1] + R[b, 2, 2] * Lw[b, 2]
            wbx = Iinv[b, 0, 0] * lx + Iinv[b, 0, 1] * ly + Iinv[b, 0, 2] * lz
            wby = Iinv[b, 1, 0] * lx + Iinv[b, 1, 1] * ly + Iinv[b, 1, 2] * lz
            wbz = Iinv[b, 2, 0] * lx + Iinv[b, 2, 1] * ly + Iinv[b, 2, 2] * lz
            W[b, 0] = R[b, 0, 0] * wbx + R[b, 0, 1] * wby + R[b, 0, 2] * wbz
            W[b, 1] = R[b, 1, 0] * wbx + R[b, 1, 1] * wby + R[b, 1, 2] * wbz
            W[b, 2] = R[b, 2, 0] * wbx + R[b, 2, 1] * wby + R[b, 2, 2] * wbz
        elif rmode[b] == R_SPIN:
            W[b, 0] = axis[b, 0] * wsp[b]; W[b, 1] = axis[b, 1] * wsp[b]; W[b, 2] = axis[b, 2] * wsp[b]
        else:
            W[b, 0] = 0.0; W[b, 1] = 0.0; W[b, 2] = 0.0


@njit(cache=True, nogil=True)
def _measure(mk, e, X, V, W, R, th, ba, bb, C, s0, GA, GB, GX, geo):
    """Returns (s, sd, ok). geo[e] = [pA(3), pB(3), n(3), t(3)]."""
    a = ba[e]; b = bb[e]          # (linear measures are evaluated inline in the step loop)
    if mk == MK_DIST or mk == MK_ALONG or mk == MK_PIN:
        ax, ay, az = _pt(X, R, a, GA[e, 0], GA[e, 1], GA[e, 2])
        bx, by, bz = _pt(X, R, b, GB[e, 0], GB[e, 1], GB[e, 2])
        dx = bx - ax; dy = by - ay; dz = bz - az
        if mk == MK_DIST:
            L = math.sqrt(dx * dx + dy * dy + dz * dz)
            if L > 1e-15:
                nx = dx / L; ny = dy / L; nz = dz / L
            else:
                nx = GX[e, 0]; ny = GX[e, 1]; nz = GX[e, 2]
            s = L - GX[e, 3]
        elif mk == MK_ALONG:
            if GX[e, 4] > 0.5 and a >= 0:              # direction fixed in body A: it turns with A
                nx, ny, nz = _dir(R, a, GX[e, 0], GX[e, 1], GX[e, 2])
                s = dx * nx + dy * ny + dz * nz - GX[e, 3]
                vax, vay, vaz = _vel(X, V, W, a, bx, by, bz); vbx, vby, vbz = _vel(X, V, W, b, bx, by, bz)
                geo[e, 0] = bx; geo[e, 1] = by; geo[e, 2] = bz; geo[e, 3] = bx; geo[e, 4] = by; geo[e, 5] = bz
                geo[e, 6] = nx; geo[e, 7] = ny; geo[e, 8] = nz
                return s, nx * (vbx - vax) + ny * (vby - vay) + nz * (vbz - vaz), True
            nx = GX[e, 0]; ny = GX[e, 1]; nz = GX[e, 2]
            s = dx * nx + dy * ny + dz * nz - GX[e, 3]
        else:
            nx = 1.0; ny = 0.0; nz = 0.0
            s = math.sqrt(dx * dx + dy * dy + dz * dz)
        geo[e, 0] = ax; geo[e, 1] = ay; geo[e, 2] = az; geo[e, 3] = bx; geo[e, 4] = by; geo[e, 5] = bz
        geo[e, 6] = nx; geo[e, 7] = ny; geo[e, 8] = nz
        if mk == MK_PIN:
            return s, 0.0, True
        vax, vay, vaz = _vel(X, V, W, a, ax, ay, az); vbx, vby, vbz = _vel(X, V, W, b, bx, by, bz)
        return s, nx * (vbx - vax) + ny * (vby - vay) + nz * (vbz - vaz), True
    if mk == MK_PLANE:
        cx, cy, cz = _pt(X, R, a, GA[e, 0], GA[e, 1], GA[e, 2]); r = GA[e, 3]
        ox, oy, oz = _pt(X, R, b, GB[e, 0], GB[e, 1], GB[e, 2])
        side = GX[e, 3]
        nx, ny, nz = _dir(R, b, GB[e, 3] * side, GB[e, 4] * side, GB[e, 5] * side)
        ux, uy, uz = _dir(R, b, GB[e, 6], GB[e, 7], GB[e, 8])
        wx = ny * uz - nz * uy; wy = nz * ux - nx * uz; wz = nx * uy - ny * ux
        rx = cx - ox; ry = cy - oy; rz = cz - oz
        s = r - (rx * nx + ry * ny + rz * nz)
        pu = rx * ux + ry * uy + rz * uz; pw = rx * wx + ry * wy + rz * wz
        px = cx - r * nx; py = cy - r * ny; pz = cz - r * nz
        geo[e, 0] = px; geo[e, 1] = py; geo[e, 2] = pz; geo[e, 3] = px; geo[e, 4] = py; geo[e, 5] = pz
        geo[e, 6] = nx; geo[e, 7] = ny; geo[e, 8] = nz
        vax, vay, vaz = _vel(X, V, W, a, px, py, pz); vbx, vby, vbz = _vel(X, V, W, b, px, py, pz)
        sd = nx * (vbx - vax) + ny * (vby - vay) + nz * (vbz - vaz)
        ok = abs(pu) <= GB[e, 9] and abs(pw) <= GB[e, 10] and s > 0.0 and s < GB[e, 11]
        return s, sd, ok
    if mk == MK_SPH:
        ax, ay, az = _pt(X, R, a, GA[e, 0], GA[e, 1], GA[e, 2]); bx, by, bz = _pt(X, R, b, GB[e, 0], GB[e, 1], GB[e, 2])
        dx = bx - ax; dy = by - ay; dz = bz - az; L = math.sqrt(dx * dx + dy * dy + dz * dz)
        if L > 1e-15:
            ux = dx / L; uy = dy / L; uz = dz / L
        else:
            ux = 1.0; uy = 0.0; uz = 0.0
        ra = GA[e, 3]; rb = GB[e, 3]
        s = ra + rb - L
        pax = ax + ra * ux; pay = ay + ra * uy; paz = az + ra * uz
        pbx = bx - rb * ux; pby = by - rb * uy; pbz = bz - rb * uz
        geo[e, 0] = pax; geo[e, 1] = pay; geo[e, 2] = paz; geo[e, 3] = pbx; geo[e, 4] = pby; geo[e, 5] = pbz
        geo[e, 6] = -ux; geo[e, 7] = -uy; geo[e, 8] = -uz
        vax, vay, vaz = _vel(X, V, W, a, pax, pay, paz); vbx, vby, vbz = _vel(X, V, W, b, pbx, pby, pbz)
        return s, -ux * (vbx - vax) - uy * (vby - vay) - uz * (vbz - vaz), s > 0.0
    return 0.0, 0.0, False


@njit(cache=True, nogil=True)
def _omega1(R, rmode, Iinv, axis, Lw, wsp, W, b):
    if rmode[b] == R_FREE:
        lx = R[b, 0, 0] * Lw[b, 0] + R[b, 1, 0] * Lw[b, 1] + R[b, 2, 0] * Lw[b, 2]
        ly = R[b, 0, 1] * Lw[b, 0] + R[b, 1, 1] * Lw[b, 1] + R[b, 2, 1] * Lw[b, 2]
        lz = R[b, 0, 2] * Lw[b, 0] + R[b, 1, 2] * Lw[b, 1] + R[b, 2, 2] * Lw[b, 2]
        wbx = Iinv[b, 0, 0] * lx + Iinv[b, 0, 1] * ly + Iinv[b, 0, 2] * lz
        wby = Iinv[b, 1, 0] * lx + Iinv[b, 1, 1] * ly + Iinv[b, 1, 2] * lz
        wbz = Iinv[b, 2, 0] * lx + Iinv[b, 2, 1] * ly + Iinv[b, 2, 2] * lz
        W[b, 0] = R[b, 0, 0] * wbx + R[b, 0, 1] * wby + R[b, 0, 2] * wbz
        W[b, 1] = R[b, 1, 0] * wbx + R[b, 1, 1] * wby + R[b, 1, 2] * wbz
        W[b, 2] = R[b, 2, 0] * wbx + R[b, 2, 1] * wby + R[b, 2, 2] * wbz
    elif rmode[b] == R_SPIN:
        W[b, 0] = axis[b, 0] * wsp[b]; W[b, 1] = axis[b, 1] * wsp[b]; W[b, 2] = axis[b, 2] * wsp[b]


@njit(cache=True, nogil=True)
def _relv(e, X, V, W, ba, bb, geo):
    vax, vay, vaz = _vel(X, V, W, ba[e], geo[e, 0], geo[e, 1], geo[e, 2])
    vbx, vby, vbz = _vel(X, V, W, bb[e], geo[e, 3], geo[e, 4], geo[e, 5])
    return vbx - vax, vby - vay, vbz - vaz


@njit(cache=True, nogil=True)
def _impulse3(e, jx, jy, jz, X, V, W, R, Lw, wsp, minv, rmode, Iinv, axis, Iax, ba, bb, geo, DX, DW, tdt):
    """Impulse J on A at its contact point and -J on B. Positions catch up by -dv tdt (and turns by -dw tdt).
    Returns the kinetic energy it removes: J . (u_before + u_after) / 2 with u = v_B - v_A at the points."""
    r0x, r0y, r0z = _relv(e, X, V, W, ba, bb, geo)
    for side in range(2):
        b = ba[e] if side == 0 else bb[e]
        if b < 0:
            continue
        sg = 1.0 if side == 0 else -1.0
        fx = sg * jx; fy = sg * jy; fz = sg * jz
        o3 = 0 if side == 0 else 3
        rx = geo[e, o3] - X[b, 0]; ry = geo[e, o3 + 1] - X[b, 1]; rz = geo[e, o3 + 2] - X[b, 2]
        for k, fk in ((0, fx), (1, fy), (2, fz)):
            dv = fk * minv[b, k]
            V[b, k] += dv; DX[b, k] -= dv * tdt
        tx = ry * fz - rz * fy; ty = rz * fx - rx * fz; tz = rx * fy - ry * fx
        w0x = W[b, 0]; w0y = W[b, 1]; w0z = W[b, 2]
        if rmode[b] == R_SPIN:
            wsp[b] += (tx * axis[b, 0] + ty * axis[b, 1] + tz * axis[b, 2]) * Iax[b]
        elif rmode[b] == R_FREE:
            Lw[b, 0] += tx; Lw[b, 1] += ty; Lw[b, 2] += tz
        _omega1(R, rmode, Iinv, axis, Lw, wsp, W, b)
        DW[b, 0] -= (W[b, 0] - w0x) * tdt; DW[b, 1] -= (W[b, 1] - w0y) * tdt; DW[b, 2] -= (W[b, 2] - w0z) * tdt
    r1x, r1y, r1z = _relv(e, X, V, W, ba, bb, geo)
    return 0.5 * (jx * (r0x + r1x) + jy * (r0y + r1y) + jz * (r0z + r1z))


@njit(cache=True, nogil=True, inline="always")
def _jpt3(J, r, X, b, px, py, pz, nx, ny, nz, sg):
    if b < 0:
        return
    rx = px - X[b, 0]; ry = py - X[b, 1]; rz = pz - X[b, 2]
    J[r, b, 0] += sg * nx; J[r, b, 1] += sg * ny; J[r, b, 2] += sg * nz
    J[r, b, 3] += sg * (ry * nz - rz * ny); J[r, b, 4] += sg * (rz * nx - rx * nz); J[r, b, 5] += sg * (rx * ny - ry * nx)


@njit(cache=True, nogil=True)
def _rows3(e, mk_, X, R, th, ba, bb, C, s0, GA, GB, GX, ref, axis, J, cpos):
    """Rigid constraint rows (velocity Jacobian per body [v(3), w(3)] and position errors). Returns the count."""
    nb = X.shape[0]
    for r in range(3):
        for b in range(nb):
            for k in range(6):
                J[r, b, k] = 0.0
        cpos[r] = 0.0
    a = ba[e]; b_ = bb[e]
    if mk_ == MK_LIN:
        sv = s0[e]
        for b in range(nb):
            for k in range(3):
                J[0, b, k] = C[e, 4 * b + k]; sv += C[e, 4 * b + k] * X[b, k]
            c3 = C[e, 4 * b + 3]
            if c3 != 0.0:
                J[0, b, 3] = c3 * axis[b, 0]; J[0, b, 4] = c3 * axis[b, 1]; J[0, b, 5] = c3 * axis[b, 2]
                sv += c3 * th[b]
        cpos[0] = sv - ref[e]
        return 1
    if mk_ == MK_TWIST:
        awx, awy, awz = _dir(R, a, GX[e, 0], GX[e, 1], GX[e, 2])
        if b_ >= 0:
            J[0, b_, 3] = awx; J[0, b_, 4] = awy; J[0, b_, 5] = awz
        if a >= 0:
            J[0, a, 3] = -awx; J[0, a, 4] = -awy; J[0, a, 5] = -awz
        return 1
    if mk_ == MK_ROT:
        phx = 0.0; phy = 0.0; phz = 0.0
        for c3 in range(3):
            t0 = GB[e, c3]; t1 = GB[e, 3 + c3]; t2 = GB[e, 6 + c3]
            if a >= 0:
                tx = R[a, 0, 0] * t0 + R[a, 0, 1] * t1 + R[a, 0, 2] * t2
                ty = R[a, 1, 0] * t0 + R[a, 1, 1] * t1 + R[a, 1, 2] * t2
                tz = R[a, 2, 0] * t0 + R[a, 2, 1] * t1 + R[a, 2, 2] * t2
            else:
                tx = t0; ty = t1; tz = t2
            if b_ >= 0:
                ux = R[b_, 0, c3]; uy = R[b_, 1, c3]; uz = R[b_, 2, c3]
            else:
                ux = 1.0 if c3 == 0 else 0.0; uy = 1.0 if c3 == 1 else 0.0; uz = 1.0 if c3 == 2 else 0.0
            phx += 0.5 * (ty * uz - tz * uy); phy += 0.5 * (tz * ux - tx * uz); phz += 0.5 * (tx * uy - ty * ux)
        nr = 3
        d0x = 1.0; d0y = 0.0; d0z = 0.0; d1x = 0.0; d1y = 1.0; d1z = 0.0; d2x = 0.0; d2y = 0.0; d2z = 1.0
        if GX[e, 3] > 0.5:                       # hinge: lock only the two directions across its axis
            hx, hy, hz = _dir(R, a, GX[e, 0], GX[e, 1], GX[e, 2])
            tx_ = 1.0 if abs(hx) < 0.9 else 0.0; ty_ = 0.0 if abs(hx) < 0.9 else 1.0
            pp = tx_ * hx + ty_ * hy
            d0x = tx_ - pp * hx; d0y = ty_ - pp * hy; d0z = -pp * hz
            nn = math.sqrt(d0x * d0x + d0y * d0y + d0z * d0z); d0x /= nn; d0y /= nn; d0z /= nn
            d1x = hy * d0z - hz * d0y; d1y = hz * d0x - hx * d0z; d1z = hx * d0y - hy * d0x
            nr = 2
        for r in range(nr):
            dx = d0x if r == 0 else (d1x if r == 1 else d2x)
            dy = d0y if r == 0 else (d1y if r == 1 else d2y)
            dz = d0z if r == 0 else (d1z if r == 1 else d2z)
            if b_ >= 0:
                J[r, b_, 3] = dx; J[r, b_, 4] = dy; J[r, b_, 5] = dz
            if a >= 0:
                J[r, a, 3] = -dx; J[r, a, 4] = -dy; J[r, a, 5] = -dz
            cpos[r] = phx * dx + phy * dy + phz * dz
        return nr
    ax, ay, az = _pt(X, R, a, GA[e, 0], GA[e, 1], GA[e, 2]); bx, by, bz = _pt(X, R, b_, GB[e, 0], GB[e, 1], GB[e, 2])
    if mk_ == MK_PIN:
        for r in range(3):
            nx = 1.0 if r == 0 else 0.0; ny = 1.0 if r == 1 else 0.0; nz = 1.0 if r == 2 else 0.0
            _jpt3(J, r, X, b_, bx, by, bz, nx, ny, nz, 1.0); _jpt3(J, r, X, a, ax, ay, az, nx, ny, nz, -1.0)
        cpos[0] = bx - ax; cpos[1] = by - ay; cpos[2] = bz - az
        return 3
    nx = GX[e, 0]; ny = GX[e, 1]; nz = GX[e, 2]
    if GX[e, 4] > 0.5 and a >= 0:
        nx, ny, nz = _dir(R, a, GX[e, 0], GX[e, 1], GX[e, 2])
        _jpt3(J, 0, X, b_, bx, by, bz, nx, ny, nz, 1.0); _jpt3(J, 0, X, a, bx, by, bz, nx, ny, nz, -1.0)
    else:
        _jpt3(J, 0, X, b_, bx, by, bz, nx, ny, nz, 1.0); _jpt3(J, 0, X, a, ax, ay, az, nx, ny, nz, -1.0)
    cpos[0] = (bx - ax) * nx + (by - ay) * ny + (bz - az) * nz - GX[e, 3] - ref[e]
    return 1



@njit(cache=True, nogil=True)
def _jsolve3(mode, ne, nb, typ, active, mk, X, V, W, R, th, quat, Lw, wsp, minv, rmode, Iinv, axis, Iax, ba, bb, C, s0,
             GA, GB, GX, refc, J3, cps, J3o, lam, aux, JA, JO, rhs, re, rr, dt, first_pass):
    """Every rigid-joint row at once: K x = rhs, K = J M^-1 J^T (direct, so long levers on light bodies converge
    where row-by-row sweeps crawl). mode 0: velocities onto the constraints; mode 1: positions (SHAKE along the
    start-of-step directions J3o). Returns the largest residual."""
    n = 0
    for e in range(ne):
        if typ[e] != 15 or active[e] == 0 or (mode == 1 and mk[e] == MK_TWIST):
            continue
        nr = _rows3(e, mk[e], X, R, th, ba, bb, C, s0, GA, GB, GX, refc, axis, J3, cps)
        for r in range(nr):
            u = 0.0
            for b in range(nb):
                for k in range(6):
                    JA[n, b, k] = J3[r, b, k]
                    JO[n, b, k] = J3o[e, r, b, k] if mode == 1 else J3[r, b, k]
                for k in range(3):
                    u += J3[r, b, k] * V[b, k] + J3[r, b, 3 + k] * W[b, k]
            rhs[n] = -cps[r] if mode == 1 else -u
            if mode == 1 and first_pass and abs(cps[r]) > aux[e, 3]:
                aux[e, 3] = abs(cps[r])
            re[n] = e; rr[n] = r; n += 1
    if n == 0:
        return 0.0
    mx = 0.0
    for i in range(n):
        if abs(rhs[i]) > mx:
            mx = abs(rhs[i])
    if mx == 0.0:
        return 0.0
    # M^-1 JO^T per row: linear part minv * J, angular part the change of world omega
    MJ = np.zeros((n, nb, 6))
    for j in range(n):
        for b in range(nb):
            for k in range(3):
                MJ[j, b, k] = minv[b, k] * JO[j, b, k]
            ox = JO[j, b, 3]; oy = JO[j, b, 4]; oz = JO[j, b, 5]
            if ox != 0.0 or oy != 0.0 or oz != 0.0:
                tx, ty, tz = _turn_dir(R, rmode, Iinv, axis, Iax, b, ox, oy, oz)
                MJ[j, b, 3] = tx; MJ[j, b, 4] = ty; MJ[j, b, 5] = tz
    K = np.zeros((n, n)); tr = 0.0
    for i in range(n):
        for j in range(n):
            s_ = 0.0
            for b in range(nb):
                for k in range(6):
                    s_ += JA[i, b, k] * MJ[j, b, k]
            K[i, j] = s_
        tr += K[i, i]
    if tr <= 0.0:
        return mx
    for i in range(n):
        K[i, i] += 1e-12 * tr / n
    x = np.linalg.solve(K, rhs[:n].copy())
    for i in range(n):
        if not np.isfinite(x[i]):
            return mx
    for j in range(n):
        e = re[j]; r = rr[j]; dl = x[j]
        lam[e, r] += dl / dt if mode == 1 else dl
        for b in range(nb):
            for k in range(3):
                if JO[j, b, k] != 0.0:
                    if mode == 1:
                        X[b, k] += minv[b, k] * JO[j, b, k] * dl
                        V[b, k] += minv[b, k] * JO[j, b, k] * dl / dt
                    else:
                        V[b, k] += minv[b, k] * JO[j, b, k] * dl
            jx = JO[j, b, 3] * dl; jy = JO[j, b, 4] * dl; jz = JO[j, b, 5] * dl
            if (jx != 0.0 or jy != 0.0 or jz != 0.0) and rmode[b] != R_NONE:
                f_ = 1.0 / dt if mode == 1 else 1.0
                if rmode[b] == R_SPIN:
                    wsp[b] += (jx * axis[b, 0] + jy * axis[b, 1] + jz * axis[b, 2]) * Iax[b] * f_
                elif rmode[b] == R_FREE:
                    Lw[b, 0] += jx * f_; Lw[b, 1] += jy * f_; Lw[b, 2] += jz * f_
                if mode == 1:
                    tx, ty, tz = _turn_dir(R, rmode, Iinv, axis, Iax, b, jx, jy, jz)
                    _rotate_quat(quat, b, tx, ty, tz, 1.0)
                    _qmat(quat[b, 0], quat[b, 1], quat[b, 2], quat[b, 3], R[b])
                    if rmode[b] == R_SPIN:
                        th[b] += tx * axis[b, 0] + ty * axis[b, 1] + tz * axis[b, 2]
                else:
                    _omega1(R, rmode, Iinv, axis, Lw, wsp, W, b)
    return mx

@njit(cache=True, nogil=True)
def _keff3(J, r, R, minv, rmode, Iinv, axis, Iax, nb):
    """J M^-1 J^T for row r, and fills nothing else."""
    k = 0.0
    for b in range(nb):
        k += minv[b, 0] * J[r, b, 0] ** 2 + minv[b, 1] * J[r, b, 1] ** 2 + minv[b, 2] * J[r, b, 2] ** 2
        jx = J[r, b, 3]; jy = J[r, b, 4]; jz = J[r, b, 5]
        if jx == 0.0 and jy == 0.0 and jz == 0.0:
            continue
        if rmode[b] == R_SPIN:
            t = jx * axis[b, 0] + jy * axis[b, 1] + jz * axis[b, 2]
            k += Iax[b] * t * t
        elif rmode[b] == R_FREE:
            bx = R[b, 0, 0] * jx + R[b, 1, 0] * jy + R[b, 2, 0] * jz
            by = R[b, 0, 1] * jx + R[b, 1, 1] * jy + R[b, 2, 1] * jz
            bz = R[b, 0, 2] * jx + R[b, 1, 2] * jy + R[b, 2, 2] * jz
            k += (bx * (Iinv[b, 0, 0] * bx + Iinv[b, 0, 1] * by + Iinv[b, 0, 2] * bz) +
                  by * (Iinv[b, 1, 0] * bx + Iinv[b, 1, 1] * by + Iinv[b, 1, 2] * bz) +
                  bz * (Iinv[b, 2, 0] * bx + Iinv[b, 2, 1] * by + Iinv[b, 2, 2] * bz))
    return k


@njit(cache=True, nogil=True)
def _kmix3(Jc, Jo, e, r, R, minv, rmode, Iinv, axis, Iax, nb):
    """Jc[r] M^-1 Jo[e, r]^T: the current row against the start-of-step direction (SHAKE)."""
    k = 0.0
    for b in range(nb):
        k += minv[b, 0] * Jc[r, b, 0] * Jo[e, r, b, 0] + minv[b, 1] * Jc[r, b, 1] * Jo[e, r, b, 1] + \
            minv[b, 2] * Jc[r, b, 2] * Jo[e, r, b, 2]
        ox = Jo[e, r, b, 3]; oy = Jo[e, r, b, 4]; oz = Jo[e, r, b, 5]
        if ox == 0.0 and oy == 0.0 and oz == 0.0:
            continue
        tx, ty, tz = _turn_dir(R, rmode, Iinv, axis, Iax, b, ox, oy, oz)
        k += Jc[r, b, 3] * tx + Jc[r, b, 4] * ty + Jc[r, b, 5] * tz
    return k


@njit(cache=True, nogil=True)
def _turn_dir(R, rmode, Iinv, axis, Iax, b, jx, jy, jz):
    """M^-1 applied to an angular impulse (world): the change of world angular velocity."""
    if rmode[b] == R_SPIN:
        t = (jx * axis[b, 0] + jy * axis[b, 1] + jz * axis[b, 2]) * Iax[b]
        return t * axis[b, 0], t * axis[b, 1], t * axis[b, 2]
    if rmode[b] == R_FREE:
        bx = R[b, 0, 0] * jx + R[b, 1, 0] * jy + R[b, 2, 0] * jz
        by = R[b, 0, 1] * jx + R[b, 1, 1] * jy + R[b, 2, 1] * jz
        bz = R[b, 0, 2] * jx + R[b, 1, 2] * jy + R[b, 2, 2] * jz
        wx = Iinv[b, 0, 0] * bx + Iinv[b, 0, 1] * by + Iinv[b, 0, 2] * bz
        wy = Iinv[b, 1, 0] * bx + Iinv[b, 1, 1] * by + Iinv[b, 1, 2] * bz
        wz = Iinv[b, 2, 0] * bx + Iinv[b, 2, 1] * by + Iinv[b, 2, 2] * bz
        return (R[b, 0, 0] * wx + R[b, 0, 1] * wy + R[b, 0, 2] * wz, R[b, 1, 0] * wx + R[b, 1, 1] * wy + R[b, 1, 2] * wz,
                R[b, 2, 0] * wx + R[b, 2, 1] * wy + R[b, 2, 2] * wz)
    return 0.0, 0.0, 0.0


@njit(cache=True, nogil=True)
def _motor_row(e, m_, C, ba, bb, geo, axis, rmode, nb, g):
    """A motor's measure as a velocity row per body [v(3), w(3)] (like a joint row)."""
    for b in range(nb):
        for k in range(6):
            g[b, k] = 0.0
    if m_ == MK_TWIST:
        if bb[e] >= 0:
            g[bb[e], 3] += geo[e, 6]; g[bb[e], 4] += geo[e, 7]; g[bb[e], 5] += geo[e, 8]
        if ba[e] >= 0:
            g[ba[e], 3] -= geo[e, 6]; g[ba[e], 4] -= geo[e, 7]; g[ba[e], 5] -= geo[e, 8]
        return
    for b in range(nb):
        for k in range(3):
            g[b, k] = C[e, 4 * b + k]
        c = C[e, 4 * b + 3]
        if c != 0.0 and rmode[b] == R_SPIN:
            for k in range(3):
                g[b, 3 + k] = c * axis[b, k]


@njit(cache=True, nogil=True)
def _motor_eff3(em, ne, nb, typ, active, mk, X, V, W, Vst, Wst, R, th, minv, rmode, Iinv, axis, Iax, ba, bb, C, s0,
                GA, GB, GX, refc, J3, cps, JA, g, geo):
    """A motor's shaft as the rigid joints leave it free to move: (1 / inertia along its measure, the measure's
    rate change this step), both with the joints held (g M^-1 g' - b' K^-1 b, b = J M^-1 g', K = J M^-1 J')."""
    _motor_row(em, mk[em], C, ba, bb, geo, axis, rmode, nb, g)
    Mg = np.zeros((nb, 6))
    gg = 0.0; du = 0.0
    for b in range(nb):
        for k in range(3):
            Mg[b, k] = minv[b, k] * g[b, k]
        tx, ty, tz = _turn_dir(R, rmode, Iinv, axis, Iax, b, g[b, 3], g[b, 4], g[b, 5])
        Mg[b, 3] = tx; Mg[b, 4] = ty; Mg[b, 5] = tz
        for k in range(3):
            gg += g[b, k] * Mg[b, k] + g[b, 3 + k] * Mg[b, 3 + k]
            du += g[b, k] * (V[b, k] - Vst[b, k]) + g[b, 3 + k] * (W[b, k] - Wst[b, k])
    n = 0
    for e in range(ne):
        if typ[e] != 15 or active[e] == 0:
            continue
        nr = _rows3(e, mk[e], X, R, th, ba, bb, C, s0, GA, GB, GX, refc, axis, J3, cps)
        for r in range(nr):
            for b in range(nb):
                for k in range(6):
                    JA[n, b, k] = J3[r, b, k]
            n += 1
    if n == 0:
        return gg, du
    MJ = np.zeros((n, nb, 6))
    for j in range(n):
        for b in range(nb):
            for k in range(3):
                MJ[j, b, k] = minv[b, k] * JA[j, b, k]
            tx, ty, tz = _turn_dir(R, rmode, Iinv, axis, Iax, b, JA[j, b, 3], JA[j, b, 4], JA[j, b, 5])
            MJ[j, b, 3] = tx; MJ[j, b, 4] = ty; MJ[j, b, 5] = tz
    K = np.zeros((n, n)); bv = np.zeros(n); jd = np.zeros(n); tr = 0.0
    for i in range(n):
        for b in range(nb):
            for k in range(6):
                bv[i] += JA[i, b, k] * Mg[b, k]
            for k in range(3):
                jd[i] += JA[i, b, k] * (V[b, k] - Vst[b, k]) + JA[i, b, 3 + k] * (W[b, k] - Wst[b, k])
        for j in range(n):
            s_ = 0.0
            for b in range(nb):
                for k in range(6):
                    s_ += JA[i, b, k] * MJ[j, b, k]
            K[i, j] = s_
        tr += K[i, i]
    if tr <= 0.0:
        return gg, du
    for i in range(n):
        K[i, i] += 1e-12 * tr / n
    y = np.linalg.solve(K, bv)
    for i in range(n):
        if not np.isfinite(y[i]):
            return gg, du
        gg -= bv[i] * y[i]; du -= y[i] * jd[i]
    return gg, du


@njit(cache=True, nogil=True)
def _motor_rate(e, m_, V, W, wsp, C, ba, bb, geo, nb):
    """Rate of a motor's measure now: a twist (relative spin about the axis fixed in A) or a coordinate sum."""
    if m_ == MK_TWIST:
        u = 0.0
        if bb[e] >= 0:
            u += W[bb[e], 0] * geo[e, 6] + W[bb[e], 1] * geo[e, 7] + W[bb[e], 2] * geo[e, 8]
        if ba[e] >= 0:
            u -= W[ba[e], 0] * geo[e, 6] + W[ba[e], 1] * geo[e, 7] + W[ba[e], 2] * geo[e, 8]
        return u
    return _lin_rate(e, V, wsp, C, nb)


@njit(cache=True, nogil=True)
def _lin_rate(e, V, wsp, C, nb):
    sd = 0.0
    for b in range(nb):
        for k in range(3):
            c = C[e, 4 * b + k]
            if c != 0.0:
                sd += c * V[b, k]
        c = C[e, 4 * b + 3]
        if c != 0.0:
            sd += c * wsp[b]
    return sd


@njit(cache=True, nogil=True)
def _geo_rate(e, X, V, W, ba, bb, geo, use_t):
    o = 9 if use_t else 6
    nx = geo[e, o]; ny = geo[e, o + 1]; nz = geo[e, o + 2]
    vax, vay, vaz = _vel(X, V, W, ba[e], geo[e, 0], geo[e, 1], geo[e, 2])
    vbx, vby, vbz = _vel(X, V, W, bb[e], geo[e, 3], geo[e, 4], geo[e, 5])
    return nx * (vbx - vax) + ny * (vby - vay) + nz * (vbz - vaz)


@njit(cache=True, nogil=True)
def _body_q(X, V, W, th, wsp, rmode, b, out):
    out[0] = X[b, 0]; out[1] = X[b, 1]; out[2] = X[b, 2]
    out[3] = V[b, 0]; out[4] = V[b, 1]; out[5] = V[b, 2]
    out[6] = math.sqrt(V[b, 0] ** 2 + V[b, 1] ** 2 + V[b, 2] ** 2)
    out[7] = th[b]
    out[8] = wsp[b] if rmode[b] == R_SPIN else math.sqrt(W[b, 0] ** 2 + W[b, 1] ** 2 + W[b, 2] ** 2)
    out[9] = W[b, 0]; out[10] = W[b, 1]; out[11] = W[b, 2]


@njit(cache=True, nogil=True)
def _ke(V, W, Lw, wsp, minv, rmode, Iax, nb):
    ke = 0.0
    for b in range(nb):
        for k in range(3):
            if minv[b, k] > 0.0:
                ke += 0.5 * V[b, k] * V[b, k] / minv[b, k]
        if rmode[b] == R_FREE:
            ke += 0.5 * (W[b, 0] * Lw[b, 0] + W[b, 1] * Lw[b, 1] + W[b, 2] * Lw[b, 2])
        elif rmode[b] == R_SPIN and Iax[b] > 0.0:
            ke += 0.5 * wsp[b] * wsp[b] / Iax[b]
    return ke


@njit(cache=True, nogil=True)
def run_engine3d(n, dt, X0, V0, quat0, Lw0, wsp0, th0, minv, rmode, Iinv, axis, Iax, mass, com, grav,
                 typ, one_sided, active, mk, ba, bb, ref, P, C, s0, GA, GB, GX,
                 tab_off, tab_len, tab_k, tab_x, tab_F, log_every, cross_src, cross_idx, cross_thr, cross_dir,
               cpt, cna, cnb, cprm, cpst0, cth, cnn, cmot,
               ckind, cprog, cval, crng, ccp, cssrc, csidx, cae, caslot, casig, caarr, cctlD, adp, adf, adc, stop):
    """Iax: INVERSE inertia about the spin axis for spin bodies. Iinv: body-frame inverse inertia for free bodies.
    com: body-frame centre of mass (non-zero only for spin bodies turning about a pivot off their centre of mass)."""
    nb = X0.shape[0]; ne = typ.shape[0]
    X = X0.copy(); V = V0.copy(); quat = quat0.copy(); Lw = Lw0.copy(); wsp = wsp0.copy()
    P = P.copy(); cprm = cprm.copy()                # control signals may set element fields as it runs
    nsig = ckind.shape[0]; nsen = cssrc.shape[0]
    sig = np.zeros(nsig); cst = np.zeros((nsig, 4)); sens = np.zeros(nsen); cstack = np.zeros(64)
    SG = np.empty((nsig, 2))
    for k in range(nsig):
        SG[k, 0] = 1e300; SG[k, 1] = -1e300
    th = th0.copy()
    R = np.zeros((nb, 3, 3)); W = np.zeros((nb, 3)); Rh = np.zeros((3, 3))
    for b in range(nb):
        _qmat(quat[b, 0], quat[b, 1], quat[b, 2], quat[b, 3], R[b])
    _omega(R, rmode, Iinv, axis, Lw, wsp, W, nb)
    F = np.zeros((nb, 3)); T = np.zeros((nb, 3)); Q4 = np.zeros((nb, 4))
    st = np.zeros((ne, 5))
    # electrical circuit (optional): parts solved each step, motors read their speed and set their torque
    npart = cpt.shape[0]
    cpst = cpst0.copy(); cpout = np.zeros((npart, NOUT_C)); cflg = np.zeros(npart)
    cG = np.zeros((cnn, cnn)); cI = np.zeros(cnn); cvn = np.zeros(cnn + 1)
    CTR = np.empty((npart, 2 * NOUT_C))
    for k in range(npart):
        for c in range(NOUT_C):
            CTR[k, 2 * c] = 1e300; CTR[k, 2 * c + 1] = -1e300
    if npart > 0:      # t = 0+: the instant the circuit closes (capacitors are sources behind their ESR, inductors hold
        p0 = cpst.copy(); o0 = np.zeros((npart, NOUT_C)); f0 = np.zeros(npart); s0c = st.copy()   # their current):
        circuit_step(0.0, 1e-12, cnn, cpt, cna, cnb, cprm, p0, cth, o0, cmot, s0c, f0, cG, cI, cvn)  # an inrush peak
        for k in range(npart):                                                               # is exact, any step
            for c in range(3):
                if o0[k, c] < CTR[k, 2 * c]: CTR[k, 2 * c] = o0[k, c]
                if o0[k, c] > CTR[k, 2 * c + 1]: CTR[k, 2 * c + 1] = o0[k, c]
    for e in range(ne):
        if typ[e] == 8:
            st[e, 3] = 1.0
        if typ[e] == 3:                        # elastoplastic: a permanent set carried in from an earlier hit
            st[e, 0] = P[e, 5]; st[e, 1] = P[e, 6]
    aux = np.zeros((ne, NAUX))
    A = np.zeros(ne); D = np.zeros(ne); IN = np.zeros(ne); S0 = np.zeros(ne); S1 = np.zeros(ne)
    Fe = np.zeros(ne); se = np.zeros(ne); inc = np.zeros(ne, np.int64)
    Fp = np.zeros(ne); dsp = np.zeros(ne)
    gpv = np.zeros((ne, 3)); fpv = np.zeros((ne, 3))
    bx7 = np.zeros((ne, 7)); bg7 = np.zeros((ne, 7)); bst = np.zeros((ne, 10))   # beams: coords, forces, plastic, damping
    Eb = np.zeros((3, 3))
    exact = np.zeros(ne, np.int64)                      # geometric measures: trapezoid on the true change in s
    for e in range(ne):
        if mk[e] != MK_LIN and mk[e] != MK_SLIP and typ[e] != 9 and mk[e] != MK_ROT and mk[e] != MK_TWIST and mk[e] != MK_BEAM:
            exact[e] = 1
    geo = np.zeros((ne, 12))
    DX = np.zeros((nb, 3)); DW = np.zeros((nb, 3)); nimp = 0           # rigid impacts: catch-up this step
    for e in range(ne):
        if typ[e] == 14:
            nimp += 1
            exact[e] = 0
    cand = np.zeros(ne, np.int64); tauv = np.zeros(ne); tgt = np.zeros(ne); jn = np.zeros(ne)
    jt3 = np.zeros((ne, 3)); invn = np.zeros(ne); ucd = np.zeros(ne); u0d = np.zeros(ne); lst = np.zeros(ne); held = np.zeros(ne)
    lossN = np.zeros(ne); lossT = np.zeros(ne)
    nrig = 0
    for e in range(ne):
        if typ[e] == 15:
            nrig += 1; exact[e] = 0
    J3 = np.zeros((3, nb, 6)); cps = np.zeros(3); lam = np.zeros((ne, 3)); refc = np.zeros(ne)
    Vst = np.zeros((nb, 3)); Wst = np.zeros((nb, 3)); gmot = np.zeros((nb, 6))
    J3o = np.zeros((ne if nrig > 0 else 1, 3, nb, 6)); nrow3 = np.zeros(ne, np.int64)   # SHAKE directions
    nja = 3 * nrig if nrig > 0 else 1
    JA3 = np.zeros((nja, nb, 6)); JO3 = np.zeros((nja, nb, 6)); rhs3 = np.zeros(nja)
    rje = np.zeros(nja, np.int64); rjr = np.zeros(nja, np.int64)
    for e in range(ne):
        if typ[e] == 15 and (mk[e] == MK_LIN or mk[e] == MK_ALONG):
            _rows3(e, mk[e], X, R, th, ba, bb, C, s0, GA, GB, GX, refc, axis, J3, cps)
            refc[e] = cps[0]
    TR = np.empty((ne, 4 + 2 * NAUX))
    for e in range(ne):
        for k in range(2 + NAUX):
            TR[e, 2 * k] = 1e300; TR[e, 2 * k + 1] = -1e300
    BQ = np.zeros(NBQ)
    BT = np.empty((nb * NBQ, 2))
    for b in range(nb):
        _body_q(X, V, W, th, wsp, rmode, b, BQ)
        for k in range(NBQ):
            BT[b * NBQ + k, 0] = BQ[k]; BT[b * NBQ + k, 1] = BQ[k]
    ncr = cross_src.shape[0]
    t_cross = -np.ones(ncr)
    cx_prev = np.full(ncr, np.nan)                  # each crossing's value last step: the time is interpolated
    adapt = adp[0] > 0.5
    t = 0.0; tlog = 0.0; t_stop = adp[5] * (1.0 - 1e-13)
    adsp = np.zeros(ne)
    nlog = n // log_every + 4 if log_every > 0 else 1
    if adapt and log_every > 0:
        nlog = int(adp[5] / adp[7]) + 5
    LOGC = np.zeros((nlog if npart > 0 else 1, npart * NOUT_C if npart > 0 else 1))
    LOGB = np.zeros((nlog, nb * NBQ)); LOGE = np.zeros((nlog, ne * (2 + NAUX))); LOGT = np.zeros(nlog); j = 0
    LOGS = np.zeros((nlog if nsig > 0 else 1, nsig if nsig > 0 else 1))
    KE0 = _ke(V, W, Lw, wsp, minv, rmode, Iax, nb)
    Wg = 0.0
    hasg = grav[0] != 0.0 or grav[1] != 0.0 or grav[2] != 0.0
    PC0 = np.zeros((nb, 3))                             # gravity work is exact: m g . (com now - com at start)
    for b in range(nb):
        PC0[b, 0], PC0[b, 1], PC0[b, 2] = _pt(X, R, b, com[b, 0], com[b, 1], com[b, 2])
    first = True
    done = n
    KEpk = KE0
    jacc = np.zeros(ne); jcnt = 0          # rigid impacts: force logged as the mean over each log interval
    if log_every > 0:                      # row 0: the starting state at t = 0 (its forces are step 0's, below)
        LOGT[0] = 0.0
        for b in range(nb):
            _body_q(X, V, W, th, wsp, rmode, b, BQ)
            for k in range(NBQ):
                LOGB[0, b * NBQ + k] = BQ[k]
        j = 1
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
        for b in range(nb):
            for k in range(3):
                F[b, k] = 0.0; T[b, k] = 0.0
            for k in range(4):
                Q4[b, k] = 0.0
        if hasg:
            for b in range(nb):
                if mass[b] > 0.0:
                    px, py, pz = _pt(X, R, b, com[b, 0], com[b, 1], com[b, 2])
                    _push(F, T, X, b, px, py, pz, mass[b] * grav[0], mass[b] * grav[1], mass[b] * grav[2])
        for sweep in range(2):
            for e in range(ne):
                if (typ[e] == 10) != (sweep == 1) or typ[e] == 14 or typ[e] == 15:
                    continue
                m_ = mk[e]
                if m_ == MK_BEAM:                        # co-rotational 3D beam
                    if active[e] == 0:
                        Fe[e] = 0.0; continue
                    a_ = ba[e]; b_ = bb[e]
                    ax_, ay_, az_ = _pt(X, R, a_, GA[e, 0], GA[e, 1], GA[e, 2])
                    bx_, by_, bz_ = _pt(X, R, b_, GB[e, 0], GB[e, 1], GB[e, 2])
                    dx = bx_ - ax_; dy = by_ - ay_; dz = bz_ - az_; L = math.sqrt(dx * dx + dy * dy + dz * dz)
                    Eb[0, 0] = dx / L; Eb[1, 0] = dy / L; Eb[2, 0] = dz / L
                    # reference e2: the ends' second axes averaged, made square to the chord
                    qx = 0.0; qy = 0.0; qz = 0.0
                    for (bb_, off) in ((a_, 3), (b_, 12)):
                        l0 = GB[e, off + 1]; l1 = GB[e, off + 4]; l2 = GB[e, off + 7]
                        if bb_ >= 0:
                            qx += R[bb_, 0, 0] * l0 + R[bb_, 0, 1] * l1 + R[bb_, 0, 2] * l2
                            qy += R[bb_, 1, 0] * l0 + R[bb_, 1, 1] * l1 + R[bb_, 1, 2] * l2
                            qz += R[bb_, 2, 0] * l0 + R[bb_, 2, 1] * l1 + R[bb_, 2, 2] * l2
                        else:
                            qx += l0; qy += l1; qz += l2
                    pq = qx * Eb[0, 0] + qy * Eb[1, 0] + qz * Eb[2, 0]
                    qx -= pq * Eb[0, 0]; qy -= pq * Eb[1, 0]; qz -= pq * Eb[2, 0]
                    nq = math.sqrt(qx * qx + qy * qy + qz * qz)
                    Eb[0, 1] = qx / nq; Eb[1, 1] = qy / nq; Eb[2, 1] = qz / nq
                    Eb[0, 2] = Eb[1, 0] * Eb[2, 1] - Eb[2, 0] * Eb[1, 1]
                    Eb[1, 2] = Eb[2, 0] * Eb[0, 1] - Eb[0, 0] * Eb[2, 1]
                    Eb[2, 2] = Eb[0, 0] * Eb[1, 1] - Eb[1, 0] * Eb[0, 1]
                    txA, tyA, tzA = _end_rot(Eb, R, a_, GB, 3, e)
                    txB, tyB, tzB = _end_rot(Eb, R, b_, GB, 12, e)
                    kA = P[e, 0]; kY = P[e, 1]; kZ = P[e, 2]; kT = P[e, 3]; L0 = P[e, 4]
                    o = tab_off[e]; Lt = tab_len[e]
                    dpl = 0.0
                    if P[e, 6] > 0.5:
                        # plastic hinge at each end: elliptic yield surface (My/Mpy)^2 + (Mz/Mpz)^2 <= 1, with Mpz from
                        # the table and Mpy = P7 * Mpz (sy Zy / sy Z). Radial return in the scaled moments; each end
                        # sees the other end's latest plastic rotation. Dissipation books the energy actually released:
                        # the mean of the moments before and after times the plastic rotation (closes the ledger).
                        for end in range(2):
                            yA = tyA - bst[e, 0]; zA = tzA - bst[e, 1]; yB = tyB - bst[e, 2]; zB = tzB - bst[e, 3]
                            if end == 0:
                                my = kY * (4.0 * yA + 2.0 * yB); mz = kZ * (4.0 * zA + 2.0 * zB)
                            else:
                                my = kY * (2.0 * yA + 4.0 * yB); mz = kZ * (2.0 * zA + 4.0 * zB)
                            Yz = _yield3(bst[e, 4 + end], tab_x[o:o + Lt], tab_F[o:o + Lt]); Yy = Yz * P[e, 7]
                            fy_ = my / Yy; fz_ = mz / Yz
                            fm = math.sqrt(fy_ * fy_ + fz_ * fz_)
                            if fm > 1.0:
                                my2 = my / fm; mz2 = mz / fm
                                dyp = (my - my2) / (4.0 * kY); dzp = (mz - mz2) / (4.0 * kZ)
                                bst[e, 2 * end] += dyp; bst[e, 2 * end + 1] += dzp
                                bst[e, 4 + end] += math.sqrt(dyp * dyp + dzp * dzp)
                                dpl += 0.5 * ((my + my2) * dyp + (mz + mz2) * dzp)
                    yA = tyA - bst[e, 0]; zA = tzA - bst[e, 1]; yB = tyB - bst[e, 2]; zB = tzB - bst[e, 3]
                    MyA = kY * (4.0 * yA + 2.0 * yB); MyB = kY * (2.0 * yA + 4.0 * yB)
                    MzA = kZ * (4.0 * zA + 2.0 * zB); MzB = kZ * (2.0 * zA + 4.0 * zB)
                    N = kA * (L - L0); Tq = kT * (txB - txA)
                    stored = (0.5 * kA * (L - L0) ** 2 + 0.5 * kT * (txB - txA) ** 2 +
                              kY * (2.0 * yA * yA + 2.0 * yA * yB + 2.0 * yB * yB) +
                              kZ * (2.0 * zA * zA + 2.0 * zA * zB + 2.0 * zB * zB))
                    x7 = (L, txA, txB, tyA, tyB, tzA, tzB)
                    g7 = [N, -Tq, Tq, MyA, MyB, MzA, MzB]
                    ddp = 0.0
                    if (P[e, 5] > 0.0 or P[e, 8] > 0.0) and not first:
                        r = [(x7[k] - bx7[e, k]) / dt for k in range(7)]
                        bb_ = P[e, 8] if P[e, 8] > 0.0 else P[e, 5]     # bending at the first-mode rate
                        gd = [P[e, 5] * kA * r[0], -P[e, 5] * kT * (r[2] - r[1]), P[e, 5] * kT * (r[2] - r[1]),
                              bb_ * kY * (4 * r[3] + 2 * r[4]), bb_ * kY * (2 * r[3] + 4 * r[4]),
                              bb_ * kZ * (4 * r[5] + 2 * r[6]), bb_ * kZ * (2 * r[5] + 4 * r[6])]
                        if P[e, 9] > 0.0:
                            # bending damper behind a low-pass (Maxwell): it damps the first mode, while the fast
                            # segment modes the explicit step cannot damp stably never see it
                            lp_ = P[e, 9] * dt
                            if lp_ > 1.0:
                                lp_ = 1.0
                            for k in range(4):
                                bst[e, 6 + k] += lp_ * (gd[3 + k] - bst[e, 6 + k])
                                gd[3 + k] = bst[e, 6 + k]
                        for k in range(7):
                            g7[k] += gd[k]; ddp += gd[k] * r[k] * dt
                    N = g7[0]; Tq = g7[2]; MyA = g7[3]; MyB = g7[4]; MzA = g7[5]; MzB = g7[6]
                    vy = (MzA + MzB) / L; vz = (MyA + MyB) / L
                    fx = N * Eb[0, 0] - vy * Eb[0, 1] + vz * Eb[0, 2]
                    fy = N * Eb[1, 0] - vy * Eb[1, 1] + vz * Eb[1, 2]
                    fz = N * Eb[2, 0] - vy * Eb[2, 1] + vz * Eb[2, 2]
                    _push(F, T, X, a_, ax_, ay_, az_, fx, fy, fz); _push(F, T, X, b_, bx_, by_, bz_, -fx, -fy, -fz)
                    for k in range(3):
                        if a_ >= 0:
                            T[a_, k] += Tq * Eb[k, 0] - MyA * Eb[k, 1] - MzA * Eb[k, 2]
                        if b_ >= 0:
                            T[b_, k] += -Tq * Eb[k, 0] - MyB * Eb[k, 1] - MzB * Eb[k, 2]
                    if first:
                        S0[e] = stored
                    else:
                        for k in range(7):
                            A[e] += 0.5 * (bg7[e, k] + g7[k]) * (x7[k] - bx7[e, k])
                    S1[e] = stored; D[e] += dpl + ddp
                    for k in range(7):
                        bx7[e, k] = x7[k]; bg7[e, k] = g7[k]
                    Fe[e] = N; se[e] = L - L0; inc[e] = 1
                    mA = math.sqrt(MyA * MyA + MzA * MzA); mB = math.sqrt(MyB * MyB + MzB * MzB)
                    aux[e, 0] = mA if mA >= mB else mB; aux[e, 1] = Tq; aux[e, 2] = math.sqrt(vy * vy + vz * vz)
                    aux[e, 3] = bst[e, 4] if bst[e, 4] > bst[e, 5] else bst[e, 5]
                    continue
                if m_ == MK_ROT:                         # rotation lock (weld; hinge = all but one axis)
                    if active[e] == 0:
                        Fe[e] = 0.0; continue
                    a_ = ba[e]; b_ = bb[e]
                    phx = 0.0; phy = 0.0; phz = 0.0
                    for c3 in range(3):
                        # target column: R_A @ Rrel0[:, c3]; actual: R_B[:, c3]
                        t0 = GB[e, c3]; t1 = GB[e, 3 + c3]; t2 = GB[e, 6 + c3]
                        if a_ >= 0:
                            tx = R[a_, 0, 0] * t0 + R[a_, 0, 1] * t1 + R[a_, 0, 2] * t2
                            ty = R[a_, 1, 0] * t0 + R[a_, 1, 1] * t1 + R[a_, 1, 2] * t2
                            tz = R[a_, 2, 0] * t0 + R[a_, 2, 1] * t1 + R[a_, 2, 2] * t2
                        else:
                            tx = t0; ty = t1; tz = t2
                        if b_ >= 0:
                            ux = R[b_, 0, c3]; uy = R[b_, 1, c3]; uz = R[b_, 2, c3]
                        else:
                            ux = 1.0 if c3 == 0 else 0.0; uy = 1.0 if c3 == 1 else 0.0; uz = 1.0 if c3 == 2 else 0.0
                        phx += 0.5 * (ty * uz - tz * uy); phy += 0.5 * (tz * ux - tx * uz); phz += 0.5 * (tx * uy - ty * ux)
                    wrx = (W[b_, 0] if b_ >= 0 else 0.0) - (W[a_, 0] if a_ >= 0 else 0.0)
                    wry = (W[b_, 1] if b_ >= 0 else 0.0) - (W[a_, 1] if a_ >= 0 else 0.0)
                    wrz = (W[b_, 2] if b_ >= 0 else 0.0) - (W[a_, 2] if a_ >= 0 else 0.0)
                    if GX[e, 3] > 0.5:                   # hinge: the axis turns freely
                        awx, awy, awz = _dir(R, a_, GX[e, 0], GX[e, 1], GX[e, 2])
                        pp = phx * awx + phy * awy + phz * awz; ww = wrx * awx + wry * awy + wrz * awz
                        phx -= pp * awx; phy -= pp * awy; phz -= pp * awz
                        wrx -= ww * awx; wry -= ww * awy; wrz -= ww * awz
                    mx = P[e, 0] * phx + P[e, 1] * wrx; my = P[e, 0] * phy + P[e, 1] * wry; mz = P[e, 0] * phz + P[e, 1] * wrz
                    if b_ >= 0:
                        T[b_, 0] -= mx; T[b_, 1] -= my; T[b_, 2] -= mz
                    if a_ >= 0:
                        T[a_, 0] += mx; T[a_, 1] += my; T[a_, 2] += mz
                    stored = 0.5 * P[e, 0] * (phx * phx + phy * phy + phz * phz)
                    if first:
                        S0[e] = stored
                    else:
                        A[e] += 0.5 * ((fpv[e, 0] + mx) * (phx - gpv[e, 0]) + (fpv[e, 1] + my) * (phy - gpv[e, 1]) +
                                       (fpv[e, 2] + mz) * (phz - gpv[e, 2]))
                    S1[e] = stored
                    D[e] += P[e, 1] * (wrx * wrx + wry * wry + wrz * wrz) * dt
                    gpv[e, 0] = phx; gpv[e, 1] = phy; gpv[e, 2] = phz; fpv[e, 0] = mx; fpv[e, 1] = my; fpv[e, 2] = mz
                    Fe[e] = math.sqrt(mx * mx + my * my + mz * mz); se[e] = math.sqrt(phx * phx + phy * phy + phz * phz)
                    inc[e] = 1
                    continue
                if m_ == MK_TWIST:                       # relative turn about an axis fixed in A (bearings, drives)
                    awx, awy, awz = _dir(R, ba[e], GX[e, 0], GX[e, 1], GX[e, 2])
                    geo[e, 6] = awx; geo[e, 7] = awy; geo[e, 8] = awz
                    wbx = W[bb[e], 0] if bb[e] >= 0 else 0.0; wby = W[bb[e], 1] if bb[e] >= 0 else 0.0
                    wbz = W[bb[e], 2] if bb[e] >= 0 else 0.0
                    wax = W[ba[e], 0] if ba[e] >= 0 else 0.0; way = W[ba[e], 1] if ba[e] >= 0 else 0.0
                    waz = W[ba[e], 2] if ba[e] >= 0 else 0.0
                    sd = (wbx - wax) * awx + (wby - way) * awy + (wbz - waz) * awz
                    s = se[e] + sd * dt if not first else 0.0
                    se[e] = s
                    if active[e] == 0:
                        Fe[e] = 0.0; continue
                    o = tab_off[e]; L = tab_len[e]
                    Fv, dd, din, stored = _force(typ[e], one_sided[e], P[e], s, sd, tab_k[e], tab_x[o:o + L],
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
                    A[e] += 0.5 * (Fv - Fp[e]) * dsp[e]
                    Fe[e] = Fv; inc[e] = 1
                    if ba[e] >= 0:
                        T[ba[e], 0] += Fv * awx; T[ba[e], 1] += Fv * awy; T[ba[e], 2] += Fv * awz
                    if bb[e] >= 0:
                        T[bb[e], 0] -= Fv * awx; T[bb[e], 1] -= Fv * awy; T[bb[e], 2] -= Fv * awz
                    continue
                if m_ == MK_SLIP:
                    c = ref[e]
                    for k in range(9):
                        geo[e, k] = geo[c, k]
                    # tangential slip velocity of the contact points
                    vax, vay, vaz = _vel(X, V, W, ba[e], geo[e, 0], geo[e, 1], geo[e, 2])
                    vbx, vby, vbz = _vel(X, V, W, bb[e], geo[e, 3], geo[e, 4], geo[e, 5])
                    rx = vbx - vax; ry = vby - vay; rz = vbz - vaz
                    nx = geo[e, 6]; ny = geo[e, 7]; nz = geo[e, 8]
                    vn = rx * nx + ry * ny + rz * nz
                    tx = rx - vn * nx; ty = ry - vn * ny; tz = rz - vn * nz
                    vt = math.sqrt(tx * tx + ty * ty + tz * tz)
                    if vt > 1e-300:
                        geo[e, 9] = tx / vt; geo[e, 10] = ty / vt; geo[e, 11] = tz / vt
                    sd = vt
                    s = se[e] + sd * dt
                    ok = inc[c] == 1 and Fe[c] > 0.0 and vt > 1e-300
                elif m_ == MK_LIN:
                    s = s0[e]; sd = 0.0
                    for b in range(nb):
                        for k in range(3):
                            cc = C[e, 4 * b + k]
                            if cc != 0.0:
                                s += cc * X[b, k]
                        cc = C[e, 4 * b + 3]
                        if cc != 0.0:
                            s += cc * th[b]
                    sd = _lin_rate(e, V, wsp, C, nb)
                    ok = True
                else:
                    s, sd, ok = _measure(m_, e, X, V, W, R, th, ba, bb, C, s0, GA, GB, GX, geo)
                    # entry rule: only by crossing the plane from outside; a shallow one is a hand-over from a
                    # neighbouring patch (under 0.1% of the patch size): allowed
                    if m_ == MK_PLANE and ok and inc[e] == 0 and se[e] > 0.0 and not first:
                        if s > 2e-3 * min(GB[e, 9], GB[e, 10], 1e3):
                            ok = False
                sold = se[e]
                se[e] = s
                if active[e] == 0 or not ok:
                    if exact[e] == 1:
                        if inc[e] == 1:
                            if s <= 0.0:
                                A[e] += 0.5 * Fp[e] * (0.0 - sold)
                            else:
                                A[e] += 0.5 * Fp[e] * (s - sold)
                                if S1[e] > 0.0 and D[e] >= 0.0:
                                    D[e] += S1[e]           # lost while still pressed: elastic energy released
                    elif typ[e] != 9:
                        A[e] += 0.5 * (0.0 - Fp[e]) * dsp[e]
                    Fe[e] = 0.0; inc[e] = 0; Fp[e] = 0.0; dsp[e] = 0.0; S1[e] = 0.0
                    continue
                was = inc[e]
                inc[e] = 1
                if typ[e] == 9:                         # ball joint
                    ax = geo[e, 0]; ay = geo[e, 1]; az = geo[e, 2]; bx = geo[e, 3]; by = geo[e, 4]; bz = geo[e, 5]
                    vax, vay, vaz = _vel(X, V, W, ba[e], ax, ay, az); vbx, vby, vbz = _vel(X, V, W, bb[e], bx, by, bz)
                    gx = bx - ax; gy = by - ay; gz = bz - az; hx = vbx - vax; hy = vby - vay; hz = vbz - vaz
                    fx = P[e, 0] * gx + P[e, 1] * hx; fy = P[e, 0] * gy + P[e, 1] * hy; fz = P[e, 0] * gz + P[e, 1] * hz
                    _push(F, T, X, ba[e], ax, ay, az, fx, fy, fz); _push(F, T, X, bb[e], bx, by, bz, -fx, -fy, -fz)
                    stored = 0.5 * P[e, 0] * (gx * gx + gy * gy + gz * gz)
                    if first:
                        S0[e] = stored
                    S1[e] = stored
                    D[e] += P[e, 1] * (hx * hx + hy * hy + hz * hz) * dt
                    if not first:
                        A[e] += 0.5 * ((fpv[e, 0] + fx) * (gx - gpv[e, 0]) + (fpv[e, 1] + fy) * (gy - gpv[e, 1]) +
                                       (fpv[e, 2] + fz) * (gz - gpv[e, 2]))
                    gpv[e, 0] = gx; gpv[e, 1] = gy; gpv[e, 2] = gz; fpv[e, 0] = fx; fpv[e, 1] = fy; fpv[e, 2] = fz
                    Fe[e] = math.sqrt(fx * fx + fy * fy + fz * fz)
                    continue
                if typ[e] == 10:
                    Fn = Fe[ref[e]]
                    Fm = P[e, 0] * Fn * math.tanh(sd / P[e, 1])
                    inv = _inv_along(X, R, minv, rmode, Iinv, axis, Iax, ba[e], geo[e, 0], geo[e, 1], geo[e, 2],
                                     geo[e, 9], geo[e, 10], geo[e, 11]) + \
                        _inv_along(X, R, minv, rmode, Iinv, axis, Iax, bb[e], geo[e, 3], geo[e, 4], geo[e, 5],
                                   geo[e, 9], geo[e, 10], geo[e, 11])
                    if inv > 0.0:
                        cap = sd / (inv * dt)
                        if Fm > cap:
                            Fm = cap
                    Fv = Fm
                    D[e] += Fv * sd * dt
                    stored = 0.0
                else:
                    o = tab_off[e]; L = tab_len[e]
                    Fv, dd, din, stored = _force(typ[e], one_sided[e], P[e], s, sd, tab_k[e], tab_x[o:o + L],
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
                        A[e] += 0.5 * (Fp[e] + Fv) * (s - sold)
                    elif not first:
                        A[e] += 0.5 * Fv * (s if s > 0.0 else 0.0)
                    Fp[e] = Fv
                else:
                    A[e] += 0.5 * (Fv - Fp[e]) * dsp[e]
                Fe[e] = Fv
                if m_ == MK_LIN:
                    for b in range(nb):
                        for k in range(4):
                            cc = C[e, 4 * b + k]
                            if cc != 0.0:
                                Q4[b, k] -= cc * Fv
                else:
                    o9 = 9 if m_ == MK_SLIP else 6
                    nx = geo[e, o9]; ny = geo[e, o9 + 1]; nz = geo[e, o9 + 2]
                    _push(F, T, X, ba[e], geo[e, 0], geo[e, 1], geo[e, 2], Fv * nx, Fv * ny, Fv * nz)
                    _push(F, T, X, bb[e], geo[e, 3], geo[e, 4], geo[e, 5], -Fv * nx, -Fv * ny, -Fv * nz)
        first = False
        # velocities
        for b in range(nb):
            for k in range(3):
                Vst[b, k] = V[b, k]; Wst[b, k] = W[b, k]
        for b in range(nb):
            for k in range(3):
                V[b, k] += dt * (F[b, k] + Q4[b, k]) * minv[b, k]
            if rmode[b] == R_SPIN:
                tq = T[b, 0] * axis[b, 0] + T[b, 1] * axis[b, 1] + T[b, 2] * axis[b, 2] + Q4[b, 3]
                wsp[b] += dt * tq * Iax[b]
            elif rmode[b] == R_FREE:
                Lw[b, 0] += dt * T[b, 0]; Lw[b, 1] += dt * T[b, 1]; Lw[b, 2] += dt * T[b, 2]
        _omega(R, rmode, Iinv, axis, Lw, wsp, W, nb)
        # motors: the exact step of a shaft held by back-EMF (stable at any step; see engine1d motor_scale)
        for e in range(ne):
            if (typ[e] == 8 or typ[e] == 17) and active[e] != 0 and P[e, 10] > 0.0:
                kk = motor_k(typ[e], P[e], st[e], dt)
                if kk <= 1e-9:
                    continue
                ie, du = _motor_eff3(e, ne, nb, typ, active, mk, X, V, W, Vst, Wst, R, th, minv, rmode, Iinv, axis,
                                     Iax, ba, bb, C, s0, GA, GB, GX, refc, J3, cps, JA3, gmot, geo)
                if ie <= 0.0:
                    continue
                kk *= ie / P[e, 10]                       # the damping on the inertia the joints leave it
                J_ = du * (motor_scale(kk) - 1.0) / ie
                if mk[e] == MK_TWIST:
                    for b, sg in ((bb[e], 1.0), (ba[e], -1.0)):
                        if b < 0:
                            continue
                        jx = sg * J_ * geo[e, 6]; jy = sg * J_ * geo[e, 7]; jz = sg * J_ * geo[e, 8]
                        if rmode[b] == R_SPIN:
                            wsp[b] += (jx * axis[b, 0] + jy * axis[b, 1] + jz * axis[b, 2]) * Iax[b]
                        elif rmode[b] == R_FREE:
                            Lw[b, 0] += jx; Lw[b, 1] += jy; Lw[b, 2] += jz
                        _omega1(R, rmode, Iinv, axis, Lw, wsp, W, b)
                else:
                    for b in range(nb):
                        for k in range(3):
                            V[b, k] += C[e, 4 * b + k] * minv[b, k] * J_
                        if C[e, 4 * b + 3] != 0.0:
                            wsp[b] += C[e, 4 * b + 3] * Iax[b] * J_
                            _omega1(R, rmode, Iinv, axis, Lw, wsp, W, b)
                F0 = Fe[e]; Fe[e] = F0 - J_ / dt
                if typ[e] == 8:
                    drive_books(P[e], st[e], aux[e], F0, Fe[e], kk, dt, D, IN, A, dsp, e)
                else:
                    A[e] += 0.5 * (Fe[e] - F0) * dsp[e]
        # rigid impacts: sequential impulses on this step's velocities (restitution e, Coulomb mu)
        for b in range(nb):
            DX[b, 0] = 0.0; DX[b, 1] = 0.0; DX[b, 2] = 0.0; DW[b, 0] = 0.0; DW[b, 1] = 0.0; DW[b, 2] = 0.0
        nc = 0
        if nimp > 0 or nrig > 0:
            for e in range(ne):
                if typ[e] == 15:
                    lam[e, 0] = 0.0; lam[e, 1] = 0.0; lam[e, 2] = 0.0
            nc = 0
            for e in range(ne):
                if typ[e] != 14:
                    continue
                held[e] = jn[e]; Fe[e] = 0.0; aux[e, 0] = 0.0; aux[e, 2] = 0.0; jn[e] = 0.0
                if active[e] == 0:
                    continue
                if mk[e] == MK_LIN:
                    s = s0[e]
                    for b in range(nb):
                        for k in range(3):
                            s += C[e, 4 * b + k] * X[b, k]
                        s += C[e, 4 * b + 3] * th[b]
                    sd = _lin_rate(e, V, wsp, C, nb)
                    w_ = 0.0
                    for b in range(nb):
                        for k in range(3):
                            w_ += C[e, 4 * b + k] ** 2 * minv[b, k]
                        w_ += C[e, 4 * b + 3] ** 2 * Iax[b]
                else:
                    s, sd, ok = _measure(mk[e], e, X, V, W, R, th, ba, bb, C, s0, GA, GB, GX, geo)
                    if mk[e] == MK_PLANE:
                        cx, cy, cz = _pt(X, R, ba[e], GA[e, 0], GA[e, 1], GA[e, 2])
                        ox, oy, oz = _pt(X, R, bb[e], GB[e, 0], GB[e, 1], GB[e, 2])
                        ux, uy, uz = _dir(R, bb[e], GB[e, 6], GB[e, 7], GB[e, 8])
                        nx = geo[e, 6]; ny = geo[e, 7]; nz = geo[e, 8]
                        wx = ny * uz - nz * uy; wy = nz * ux - nx * uz; wz = nx * uy - ny * ux
                        pu = (cx - ox) * ux + (cy - oy) * uy + (cz - oz) * uz
                        pw = (cx - ox) * wx + (cy - oy) * wy + (cz - oz) * wz
                        if abs(pu) > GB[e, 9] or abs(pw) > GB[e, 10] or s >= GB[e, 11]:
                            se[e] = s; continue
                    w_ = (_inv_along(X, R, minv, rmode, Iinv, axis, Iax, ba[e], geo[e, 0], geo[e, 1], geo[e, 2],
                                     geo[e, 6], geo[e, 7], geo[e, 8]) +
                          _inv_along(X, R, minv, rmode, Iinv, axis, Iax, bb[e], geo[e, 3], geo[e, 4], geo[e, 5],
                                     geo[e, 6], geo[e, 7], geo[e, 8]))
                se[e] = s
                if sd <= 0.0 or s + sd * dt <= 0.0 or w_ <= 0.0:
                    continue
                cand[nc] = e; nc += 1
                tauv[e] = min(1.0, -s / (sd * dt)) if s < 0.0 else 0.0
                tgt[e] = -P[e, 0] * sd if (sd > P[e, 2] and held[e] <= 0.0) else 0.0     # no bounce when held
                jn[e] = 0.0; jt3[e, 0] = 0.0; jt3[e, 1] = 0.0; jt3[e, 2] = 0.0; invn[e] = w_
                aux[e, 3] += 0.5 * sd * sd / w_
                ucd[e] = sd; lst[e] = D[e]
                if mk[e] == MK_LIN:
                    u0d[e] = _lin_rate(e, Vst, wsp, C, nb)        # (spin rate change of this step ignored: small)
                else:
                    rx0, ry0, rz0 = _relv(e, X, Vst, Wst, ba, bb, geo)
                    u0d[e] = geo[e, 6] * rx0 + geo[e, 7] * ry0 + geo[e, 8] * rz0
            for c in range(nc):
                lossN[cand[c]] = 0.0; lossT[cand[c]] = 0.0
            for it in range(16):
                for c in range(nc):
                    e = cand[c]; tdt = tauv[e] * dt
                    if mk[e] == MK_LIN:
                        u = _lin_rate(e, V, wsp, C, nb)
                        new = jn[e] + (u - tgt[e]) / invn[e]
                        if new < 0.0:
                            new = 0.0
                        dj = new - jn[e]; jn[e] = new
                        if dj != 0.0:
                            for b in range(nb):
                                for k in range(3):
                                    cc = C[e, 4 * b + k]
                                    if cc != 0.0:
                                        dv = -cc * dj * minv[b, k]
                                        V[b, k] += dv; DX[b, k] -= dv * tdt
                                cc = C[e, 4 * b + 3]
                                if cc != 0.0:
                                    dw = -cc * dj * Iax[b]
                                    wsp[b] += dw
                                    W[b, 0] += axis[b, 0] * dw; W[b, 1] += axis[b, 1] * dw; W[b, 2] += axis[b, 2] * dw
                                    DW[b, 0] -= axis[b, 0] * dw * tdt; DW[b, 1] -= axis[b, 1] * dw * tdt
                                    DW[b, 2] -= axis[b, 2] * dw * tdt
                            u1 = _lin_rate(e, V, wsp, C, nb)
                            lossN[e] += 0.5 * dj * (u + u1)
                        continue
                    nx = geo[e, 6]; ny = geo[e, 7]; nz = geo[e, 8]
                    rx, ry, rz = _relv(e, X, V, W, ba, bb, geo)
                    u = nx * rx + ny * ry + nz * rz
                    new = jn[e] + (u - tgt[e]) / invn[e]
                    if new < 0.0:
                        new = 0.0
                    dj = new - jn[e]; jn[e] = new
                    if dj != 0.0:
                        lossN[e] += _impulse3(e, dj * nx, dj * ny, dj * nz, X, V, W, R, Lw, wsp, minv, rmode, Iinv,
                                              axis, Iax, ba, bb, geo, DX, DW, tdt)
                    mu = P[e, 1]
                    if mu > 0.0:
                        rx, ry, rz = _relv(e, X, V, W, ba, bb, geo)
                        un = nx * rx + ny * ry + nz * rz
                        tx = rx - un * nx; ty = ry - un * ny; tz = rz - un * nz
                        vt = math.sqrt(tx * tx + ty * ty + tz * tz)
                        if vt > 1e-15:
                            tx /= vt; ty /= vt; tz /= vt
                            it_ = (_inv_along(X, R, minv, rmode, Iinv, axis, Iax, ba[e], geo[e, 0], geo[e, 1], geo[e, 2],
                                              tx, ty, tz) +
                                   _inv_along(X, R, minv, rmode, Iinv, axis, Iax, bb[e], geo[e, 3], geo[e, 4], geo[e, 5],
                                              tx, ty, tz))
                            if it_ > 0.0:
                                gx = jt3[e, 0] + vt / it_ * tx; gy = jt3[e, 1] + vt / it_ * ty; gz = jt3[e, 2] + vt / it_ * tz
                                gm = math.sqrt(gx * gx + gy * gy + gz * gz); lim = mu * jn[e]
                                if gm > lim:
                                    gx *= lim / gm; gy *= lim / gm; gz *= lim / gm
                                dx_ = gx - jt3[e, 0]; dy_ = gy - jt3[e, 1]; dz_ = gz - jt3[e, 2]
                                jt3[e, 0] = gx; jt3[e, 1] = gy; jt3[e, 2] = gz
                                lossT[e] += _impulse3(e, dx_, dy_, dz_, X, V, W, R, Lw, wsp, minv, rmode, Iinv, axis,
                                                      Iax, ba, bb, geo, DX, DW, tdt)
                for e in range(ne):              # rigid joints: relative velocity along every row to zero
                    if typ[e] != 15 or active[e] == 0:
                        continue
                    nr = _rows3(e, mk[e], X, R, th, ba, bb, C, s0, GA, GB, GX, refc, axis, J3, cps)
                    for r in range(nr):
                        u = 0.0; u0 = 0.0
                        for b in range(nb):
                            for k in range(3):
                                u += J3[r, b, k] * V[b, k] + J3[r, b, 3 + k] * W[b, k]
                                u0 += J3[r, b, k] * Vst[b, k] + J3[r, b, 3 + k] * Wst[b, k]
                        kk = _keff3(J3, r, R, minv, rmode, Iinv, axis, Iax, nb)
                        if kk <= 0.0 or u == 0.0:
                            continue
                        dl = -u / kk
                        for b in range(nb):
                            for k in range(3):
                                V[b, k] += minv[b, k] * J3[r, b, k] * dl
                            jx = J3[r, b, 3] * dl; jy = J3[r, b, 4] * dl; jz = J3[r, b, 5] * dl
                            if jx != 0.0 or jy != 0.0 or jz != 0.0:
                                if rmode[b] == R_SPIN:
                                    wsp[b] += (jx * axis[b, 0] + jy * axis[b, 1] + jz * axis[b, 2]) * Iax[b]
                                elif rmode[b] == R_FREE:
                                    Lw[b, 0] += jx; Lw[b, 1] += jy; Lw[b, 2] += jz
                                _omega1(R, rmode, Iinv, axis, Lw, wsp, W, b)
                        lam[e, r] += dl
                        if it == 0:                      # only a speed the joint already had is a real loss
                            loss = 0.5 * u0 * u0 / kk
                            D[e] += loss; A[e] += loss
            if nrig > 0:                                    # joints: finish the velocity rows with one direct solve
                _jsolve3(0, ne, nb, typ, active, mk, X, V, W, R, th, quat, Lw, wsp, minv, rmode, Iinv, axis, Iax, ba, bb,
                         C, s0, GA, GB, GX, refc, J3, cps, J3o, lam, aux, JA3, JO3, rhs3, rje, rjr, dt, False)
            for c in range(nc):
                e = cand[c]
                Fe[e] = jn[e] / dt; aux[e, 0] = jn[e]

                aux[e, 2] = math.sqrt(jt3[e, 0] ** 2 + jt3[e, 1] ** 2 + jt3[e, 2] ** 2)
                if held[e] > 0.0 and mk[e] != MK_LIN and P[e, 1] > 0.0:
                    # held contact: friction dissipates only the slip left after the solve (sticking: none)
                    rx, ry, rz = _relv(e, X, V, W, ba, bb, geo)
                    nx = geo[e, 6]; ny = geo[e, 7]; nz = geo[e, 8]
                    un = nx * rx + ny * ry + nz * rz
                    lossT[e] = abs(jt3[e, 0] * (rx - un * nx) + jt3[e, 1] * (ry - un * ny) + jt3[e, 2] * (rz - un * nz))
            if nc > 0:
                _book_hits(cand, nc, jn, held, lossN, lossT, D, A)

        for e in range(ne):
            if typ[e] == 14:
                aux[e, 1] = D[e]; inc[e] = 1 if jn[e] > 0.0 else 0
        # work this step for linear measures and friction (rate form, new velocities, this step's geometry)
        for e in range(ne):
            if exact[e] == 1 or typ[e] == 9 or mk[e] == MK_ROT or mk[e] == MK_BEAM or typ[e] == 14 or typ[e] == 15:
                continue
            if Fe[e] == 0.0 and Fp[e] == 0.0:
                continue
            if mk[e] == MK_TWIST:
                wbx = W[bb[e], 0] if bb[e] >= 0 else 0.0; wby = W[bb[e], 1] if bb[e] >= 0 else 0.0
                wbz = W[bb[e], 2] if bb[e] >= 0 else 0.0
                wax = W[ba[e], 0] if ba[e] >= 0 else 0.0; way = W[ba[e], 1] if ba[e] >= 0 else 0.0
                waz = W[ba[e], 2] if ba[e] >= 0 else 0.0
                ds = ((wbx - wax) * geo[e, 6] + (wby - way) * geo[e, 7] + (wbz - waz) * geo[e, 8]) * dt
                A[e] += Fe[e] * ds
                Fp[e] = Fe[e]; dsp[e] = ds
                continue
            if mk[e] == MK_LIN:
                ds = _lin_rate(e, V, wsp, C, nb) * dt
                if nimp > 0:
                    for b in range(nb):
                        for k in range(3):
                            ds += C[e, 4 * b + k] * DX[b, k]
                        ds += C[e, 4 * b + 3] * (DW[b, 0] * axis[b, 0] + DW[b, 1] * axis[b, 1] + DW[b, 2] * axis[b, 2])
            else:
                ds = _geo_rate(e, X, V, W, ba, bb, geo, mk[e] == MK_SLIP) * dt
            A[e] += Fe[e] * ds
            Fp[e] = Fe[e]; dsp[e] = ds
        if nrig > 0:
            for e in range(ne):
                if typ[e] == 15 and mk[e] != MK_TWIST:
                    nrow3[e] = _rows3(e, mk[e], X, R, th, ba, bb, C, s0, GA, GB, GX, refc, axis, J3, cps)
                    for r in range(nrow3[e]):
                        for b in range(nb):
                            for k in range(6):
                                J3o[e, r, b, k] = J3[r, b, k]
        # positions and orientations
        for b in range(nb):
            for k in range(3):
                X[b, k] += dt * V[b, k]
            if rmode[b] == R_FREE:
                # midpoint rule (second order): turn half a step, read omega there from the new L, then turn the
                # whole step from the start with that omega
                q0w = quat[b, 0]; q0x = quat[b, 1]; q0y = quat[b, 2]; q0z = quat[b, 3]
                _rotate_quat(quat, b, W[b, 0], W[b, 1], W[b, 2], 0.5 * dt)
                _qmat(quat[b, 0], quat[b, 1], quat[b, 2], quat[b, 3], Rh)
                lx = Rh[0, 0] * Lw[b, 0] + Rh[1, 0] * Lw[b, 1] + Rh[2, 0] * Lw[b, 2]
                ly = Rh[0, 1] * Lw[b, 0] + Rh[1, 1] * Lw[b, 1] + Rh[2, 1] * Lw[b, 2]
                lz = Rh[0, 2] * Lw[b, 0] + Rh[1, 2] * Lw[b, 1] + Rh[2, 2] * Lw[b, 2]
                bx = Iinv[b, 0, 0] * lx + Iinv[b, 0, 1] * ly + Iinv[b, 0, 2] * lz
                by = Iinv[b, 1, 0] * lx + Iinv[b, 1, 1] * ly + Iinv[b, 1, 2] * lz
                bz = Iinv[b, 2, 0] * lx + Iinv[b, 2, 1] * ly + Iinv[b, 2, 2] * lz
                hx = Rh[0, 0] * bx + Rh[0, 1] * by + Rh[0, 2] * bz
                hy = Rh[1, 0] * bx + Rh[1, 1] * by + Rh[1, 2] * bz
                hz = Rh[2, 0] * bx + Rh[2, 1] * by + Rh[2, 2] * bz
                quat[b, 0] = q0w; quat[b, 1] = q0x; quat[b, 2] = q0y; quat[b, 3] = q0z
                _rotate_quat(quat, b, hx, hy, hz, dt)
                _qmat(quat[b, 0], quat[b, 1], quat[b, 2], quat[b, 3], R[b])
            elif rmode[b] == R_SPIN:
                _rotate_quat(quat, b, W[b, 0], W[b, 1], W[b, 2], dt)
                _qmat(quat[b, 0], quat[b, 1], quat[b, 2], quat[b, 3], R[b])
            if rmode[b] == R_SPIN:
                th[b] += dt * wsp[b]
        if nimp > 0:                                        # impacts: positions catch up (moved at old speed till the hit)
            for b in range(nb):
                X[b, 0] += DX[b, 0]; X[b, 1] += DX[b, 1]; X[b, 2] += DX[b, 2]
                if rmode[b] != R_NONE and (DW[b, 0] != 0.0 or DW[b, 1] != 0.0 or DW[b, 2] != 0.0):
                    _rotate_quat(quat, b, DW[b, 0], DW[b, 1], DW[b, 2], 1.0)
                    _qmat(quat[b, 0], quat[b, 1], quat[b, 2], quat[b, 3], R[b])
                    if rmode[b] == R_SPIN:
                        th[b] += DW[b, 0] * axis[b, 0] + DW[b, 1] * axis[b, 1] + DW[b, 2] * axis[b, 2]
        if nrig > 0:                                        # rigid joints: Newton steps of the direct position solve
            for it in range(4):
                if _jsolve3(1, ne, nb, typ, active, mk, X, V, W, R, th, quat, Lw, wsp, minv, rmode, Iinv, axis, Iax, ba,
                            bb, C, s0, GA, GB, GX, refc, J3, cps, J3o, lam, aux, JA3, JO3, rhs3, rje, rjr, dt,
                            it == 0) < 1e-12:
                    break
            for it in range(4):                            # row-by-row clean-up of anything the direct solve left
                mx_ = 0.0
                for e in range(ne):
                    if typ[e] != 15 or active[e] == 0 or mk[e] == MK_TWIST:
                        continue
                    nr = _rows3(e, mk[e], X, R, th, ba, bb, C, s0, GA, GB, GX, refc, axis, J3, cps)
                    for r in range(nr):
                        if r > 0:
                            nr = _rows3(e, mk[e], X, R, th, ba, bb, C, s0, GA, GB, GX, refc, axis, J3, cps)
                        kk = _kmix3(J3, J3o, e, r, R, minv, rmode, Iinv, axis, Iax, nb)   # SHAKE: old directions
                        if kk <= 0.0:
                            continue
                        if it == 0 and abs(cps[r]) > aux[e, 3]:
                            aux[e, 3] = abs(cps[r])                    # largest drift seen before correction
                        if abs(cps[r]) > mx_:
                            mx_ = abs(cps[r])
                        dl = -cps[r] / kk
                        lam[e, r] += dl / dt
                        for b in range(nb):
                            for k in range(3):
                                X[b, k] += minv[b, k] * J3o[e, r, b, k] * dl
                                V[b, k] += minv[b, k] * J3o[e, r, b, k] * dl / dt   # velocities follow (RATTLE)
                            jx = J3o[e, r, b, 3] * dl; jy = J3o[e, r, b, 4] * dl; jz = J3o[e, r, b, 5] * dl
                            if (jx != 0.0 or jy != 0.0 or jz != 0.0) and rmode[b] != R_NONE:
                                if rmode[b] == R_SPIN:
                                    wsp[b] += (jx * axis[b, 0] + jy * axis[b, 1] + jz * axis[b, 2]) * Iax[b] / dt
                                elif rmode[b] == R_FREE:
                                    Lw[b, 0] += jx / dt; Lw[b, 1] += jy / dt; Lw[b, 2] += jz / dt
                                tx, ty, tz = _turn_dir(R, rmode, Iinv, axis, Iax, b, jx, jy, jz)
                                _rotate_quat(quat, b, tx, ty, tz, 1.0)
                                _qmat(quat[b, 0], quat[b, 1], quat[b, 2], quat[b, 3], R[b])
                                if rmode[b] == R_SPIN:
                                    th[b] += tx * axis[b, 0] + ty * axis[b, 1] + tz * axis[b, 2]
                if mx_ < 1e-13:
                    break
        _omega(R, rmode, Iinv, axis, Lw, wsp, W, nb)        # free bodies: omega follows the new orientation
        if nrig > 0:                                        # RATTLE: velocities tangent to the constraints (numerical)
            for it in range(1):
                _jsolve3(0, ne, nb, typ, active, mk, X, V, W, R, th, quat, Lw, wsp, minv, rmode, Iinv, axis, Iax, ba, bb,
                         C, s0, GA, GB, GX, refc, J3, cps, J3o, lam, aux, JA3, JO3, rhs3, rje, rjr, dt, False)
            for it in range(4):
                mu_ = 0.0
                for e in range(ne):
                    if typ[e] != 15 or active[e] == 0:
                        continue
                    nr = _rows3(e, mk[e], X, R, th, ba, bb, C, s0, GA, GB, GX, refc, axis, J3, cps)
                    for r in range(nr):
                        u = 0.0
                        for b in range(nb):
                            for k in range(3):
                                u += J3[r, b, k] * V[b, k] + J3[r, b, 3 + k] * W[b, k]
                        if abs(u) > mu_:
                            mu_ = abs(u)
                        kk = _keff3(J3, r, R, minv, rmode, Iinv, axis, Iax, nb)
                        if kk <= 0.0 or u == 0.0:
                            continue
                        dl = -u / kk
                        lam[e, r] += dl
                        for b in range(nb):
                            for k in range(3):
                                V[b, k] += minv[b, k] * J3[r, b, k] * dl
                            jx = J3[r, b, 3] * dl; jy = J3[r, b, 4] * dl; jz = J3[r, b, 5] * dl
                            if jx != 0.0 or jy != 0.0 or jz != 0.0:
                                if rmode[b] == R_SPIN:
                                    wsp[b] += (jx * axis[b, 0] + jy * axis[b, 1] + jz * axis[b, 2]) * Iax[b]
                                elif rmode[b] == R_FREE:
                                    Lw[b, 0] += jx; Lw[b, 1] += jy; Lw[b, 2] += jz
                                _omega1(R, rmode, Iinv, axis, Lw, wsp, W, b)
                if mu_ < 1e-13:
                    break
            for it in range(8):
                for c in range(nc):                         # contacts that pushed this step keep pushing (numerical)
                    e = cand[c]
                    if jn[e] <= 0.0 or invn[e] <= 0.0 or mk[e] == MK_LIN:
                        continue
                    rx, ry, rz = _relv(e, X, V, W, ba, bb, geo)
                    u = geo[e, 6] * rx + geo[e, 7] * ry + geo[e, 8] * rz
                    if u > 0.0:
                        dj = u / invn[e]
                        _impulse3(e, dj * geo[e, 6], dj * geo[e, 7], dj * geo[e, 8], X, V, W, R, Lw, wsp, minv, rmode,
                                  Iinv, axis, Iax, ba, bb, geo, DX, DW, 0.0)
        for e in range(ne):
            if typ[e] == 15:                                # joint force: all impulses this step over the step
                if mk[e] == MK_PIN or mk[e] == MK_ROT:
                    Fe[e] = math.sqrt(lam[e, 0] ** 2 + lam[e, 1] ** 2 + lam[e, 2] ** 2) / dt
                else:
                    Fe[e] = -lam[e, 0] / dt
                aux[e, 0] = lam[e, 0] / dt; aux[e, 1] = lam[e, 1] / dt; aux[e, 2] = lam[e, 2] / dt
        # trackers
        for e in range(ne):
            if Fe[e] < TR[e, 0]: TR[e, 0] = Fe[e]
            if Fe[e] > TR[e, 1]: TR[e, 1] = Fe[e]
            sv = se[e] if (inc[e] == 1 or mk[e] < MK_PLANE or mk[e] == MK_PIN) else 0.0
            if sv < TR[e, 2]: TR[e, 2] = sv
            if sv > TR[e, 3]: TR[e, 3] = sv
            for k in range(NAUX):
                if aux[e, k] < TR[e, 4 + 2 * k]: TR[e, 4 + 2 * k] = aux[e, k]
                if aux[e, k] > TR[e, 5 + 2 * k]: TR[e, 5 + 2 * k] = aux[e, k]
        for b in range(nb):
            _body_q(X, V, W, th, wsp, rmode, b, BQ)
            for k in range(NBQ):
                x = BQ[k]
                if x < BT[b * NBQ + k, 0]: BT[b * NBQ + k, 0] = x
                if x > BT[b * NBQ + k, 1]: BT[b * NBQ + k, 1] = x
        if nsig > 0:                             # control: read the state this step ended in, work out the signals,
                                                 # set the fields the next step uses
            for k in range(nsen):
                src = cssrc[k]; kk = csidx[k]
                if src == 0:
                    _body_q(X, V, W, th, wsp, rmode, kk // NBQ, BQ)
                    sens[k] = BQ[kk % NBQ]
                elif src == 2:
                    sens[k] = Fe[kk]
                elif src == 3:
                    sens[k] = se[kk]
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
        for c in range(ncr):
            if t_cross[c] < 0.0:
                src = cross_src[c]; k = cross_idx[c]
                if src == 0:
                    _body_q(X, V, W, th, wsp, rmode, k // NBQ, BQ)
                    x = BQ[k % NBQ]
                elif src == 2:
                    x = Fe[k]
                elif src == 3:
                    x = se[k]
                elif src == 50:
                    x = sig[k]
                else:
                    x = aux[k, src - 4]
                if (cross_dir[c] > 0 and x >= cross_thr[c]) or (cross_dir[c] < 0 and x <= cross_thr[c]):
                    xp_ = cx_prev[c]
                    if xp_ == xp_ and x != xp_:            # straight line between last step and this one
                        fr = (cross_thr[c] - xp_) / (x - xp_)
                        fr = 0.0 if fr < 0.0 else (1.0 if fr > 1.0 else fr)
                        t_cross[c] = tn - dt + fr * dt
                    else:
                        t_cross[c] = tn
                cx_prev[c] = x
        if i % 4 == 0:
            ke_ = _ke(V, W, Lw, wsp, minv, rmode, Iax, nb)
            if ke_ > KEpk: KEpk = ke_
        if nimp > 0:
            jcnt += 1
            for e in range(ne):
                if typ[e] == 14:
                    jacc[e] += Fe[e]
        if npart > 0:
            for e in range(ne):
                if typ[e] == 17:                 # the circuit reads each motor's speed at the end of the step
                    st[e, 2] = _motor_rate(e, mk[e], V, W, wsp, C, ba, bb, geo, nb)
            circuit_step(tn, dt, cnn, cpt, cna, cnb, cprm, cpst, cth, cpout, cmot, st, cflg, cG, cI, cvn)
            for k in range(npart):
                for c in range(NOUT_C):
                    if cpout[k, c] < CTR[k, 2 * c]: CTR[k, 2 * c] = cpout[k, c]
                    if cpout[k, c] > CTR[k, 2 * c + 1]: CTR[k, 2 * c + 1] = cpout[k, c]
        if i == 0 and log_every > 0:       # the forces of the first step are the forces at t = 0
            for k in range(nsig):
                LOGS[0, k] = sig[k]
            for e in range(ne):
                LOGE[0, e * (2 + NAUX)] = Fe[e]; LOGE[0, e * (2 + NAUX) + 1] = se[e]
                for k in range(NAUX):
                    LOGE[0, e * (2 + NAUX) + 2 + k] = aux[e, k]
            if npart > 0:
                for k in range(npart):
                    for c in range(NOUT_C):
                        LOGC[0, k * NOUT_C + c] = cpout[k, c]
        if adapt:
            logit = log_every > 0 and tn >= tlog and j < nlog - 1
            if logit:
                while tlog <= tn:
                    tlog += adp[7]
        else:
            logit = log_every > 0 and (i + 1) % log_every == 0 and j < nlog - 1
        if logit:
            if npart > 0:
                for k in range(npart):
                    for c in range(NOUT_C):
                        LOGC[j, k * NOUT_C + c] = cpout[k, c]
            LOGT[j] = tn                   # the state just reached; element forces are those of this step's start
            for k in range(nsig):
                LOGS[j, k] = sig[k]
            for b in range(nb):
                _body_q(X, V, W, th, wsp, rmode, b, BQ)
                for k in range(NBQ):
                    LOGB[j, b * NBQ + k] = BQ[k]
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
    if log_every > 0 and j < nlog and LOGT[j - 1] < adp[14] * (1.0 - 1e-12):   # the final state is always a row
        if npart > 0:
            for k in range(npart):
                for c in range(NOUT_C):
                    LOGC[j, k * NOUT_C + c] = cpout[k, c]
        LOGT[j] = adp[14]
        for k in range(nsig):
            LOGS[j, k] = sig[k]
        for b in range(nb):
            _body_q(X, V, W, th, wsp, rmode, b, BQ)
            for k in range(NBQ):
                LOGB[j, b * NBQ + k] = BQ[k]
        for e in range(ne):
            LOGE[j, e * (2 + NAUX)] = Fe[e]; LOGE[j, e * (2 + NAUX) + 1] = se[e]
            for k in range(NAUX):
                LOGE[j, e * (2 + NAUX) + 2 + k] = aux[e, k]
        j += 1
    for b in range(nb):
        if mass[b] > 0.0:
            px, py, pz = _pt(X, R, b, com[b, 0], com[b, 1], com[b, 2])
            Wg += mass[b] * (grav[0] * (px - PC0[b, 0]) + grav[1] * (py - PC0[b, 1]) + grav[2] * (pz - PC0[b, 2]))
    KE1 = _ke(V, W, Lw, wsp, minv, rmode, Iax, nb)
    FB = np.zeros(nb * NBQ)
    for b in range(nb):
        _body_q(X, V, W, th, wsp, rmode, b, BQ)
        for k in range(NBQ):
            FB[b * NBQ + k] = BQ[k]
    for e in range(min(ne, cctlD.shape[0])):
        if cctlD[e] == 1:
            D[e] = -1.0                          # a controlled spring: its stored energy is no longer its own law
    return (FB, quat, KE0, KE1, A, D, IN, S0, S1, TR, BT, aux, t_cross, LOGT[:j], LOGB[:j], LOGE[:j], Fe, se, Wg, Lw,
            KEpk, cpout, CTR, LOGC[:j], sig, SG, LOGS[:j])
