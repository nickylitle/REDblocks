"""3-D rigid-body kernels: quaternion attitude, gyrostat (rigid body carrying a spinning rotor), point-vs-box
penetration, regularized 3-D Coulomb friction.

Conventions
- Quaternions are (w, x, y, z), unit, body-to-world: v_world = R(q) v_body.
- Angular velocity w is in BODY axes. q_dot = 0.5 q (x) (0, w).
- Gyrostat: a rigid body with total inertia tensor J about its COM (rotor INCLUDED, as an axisymmetric part)
  plus a rotor of axial inertia Is spinning at Omega RELATIVE to the body about the body-fixed unit axis a.
  Generalized speeds [w, Omega] have mass matrix A = [[J, Is a], [Is a^T, Is]];
  kinetic energy 0.5 [w, Omega]^T A [w, Omega] (+ 0.5 m v^2 for translation).
"""
import math

import numpy as np

from ..registry import block
from .._jit import kernel

SRC_Q = "unit-quaternion kinematics, exponential map (Shuster, J. Astronaut. Sci. 41(4), 1993)"
SRC_G = ("gyrostat equations (Hughes, Spacecraft Attitude Dynamics, 2nd ed., ch. 4 'gyrostats'; "
         "Wittenburg, Dynamics of Multibody Systems, 2nd ed., sec. 5.6)")


@block("rigid_body", "spatial", "attitude", "rotation_matrix", kind="kernel", source=SRC_Q,
       tags=("quaternion", "3d", "frames"))
@kernel
def quat_to_matrix(q):
    """Rotation matrix (body to world) of the unit quaternion q = (w, x, y, z)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = np.empty((3, 3))
    R[0, 0] = 1 - 2 * (y * y + z * z); R[0, 1] = 2 * (x * y - w * z); R[0, 2] = 2 * (x * z + w * y)
    R[1, 0] = 2 * (x * y + w * z); R[1, 1] = 1 - 2 * (x * x + z * z); R[1, 2] = 2 * (y * z - w * x)
    R[2, 0] = 2 * (x * z - w * y); R[2, 1] = 2 * (y * z + w * x); R[2, 2] = 1 - 2 * (x * x + y * y)
    return R


@block("rigid_body", "spatial", "attitude", "integrate", kind="kernel", source=SRC_Q,
       tags=("quaternion", "3d", "time step"))
@kernel
def quat_step(q, w_body, dt):
    """Advance attitude by one step at constant body angular velocity: q <- q (x) exp(w dt / 2), renormalized.
    Exact for constant w (no first-order drift off the unit sphere)."""
    wn = math.sqrt(w_body[0] ** 2 + w_body[1] ** 2 + w_body[2] ** 2)
    h = 0.5 * wn * dt
    if wn > 1e-300:
        s = math.sin(h) / wn
        dq0, dq1, dq2, dq3 = math.cos(h), s * w_body[0], s * w_body[1], s * w_body[2]
    else:
        dq0, dq1, dq2, dq3 = 1.0, 0.5 * dt * w_body[0], 0.5 * dt * w_body[1], 0.5 * dt * w_body[2]
    a0, a1, a2, a3 = q[0], q[1], q[2], q[3]
    out = np.empty(4)
    out[0] = a0 * dq0 - a1 * dq1 - a2 * dq2 - a3 * dq3
    out[1] = a0 * dq1 + a1 * dq0 + a2 * dq3 - a3 * dq2
    out[2] = a0 * dq2 - a1 * dq3 + a2 * dq0 + a3 * dq1
    out[3] = a0 * dq3 + a1 * dq2 - a2 * dq1 + a3 * dq0
    n = math.sqrt(out[0] ** 2 + out[1] ** 2 + out[2] ** 2 + out[3] ** 2)
    return out / n


@block("rigid_body", "spatial", "gyrostat", "accelerations", kind="kernel", source=SRC_G,
       tags=("spinning rotor", "gyroscopic", "3d", "flywheel on a frame"))
@kernel
def gyrostat_accel(J, Is, a, w, Om, tau, tau_axial):
    """Angular accelerations of a body carrying a rotor. J: 3x3 total inertia about the COM (body axes, rotor
    included), Is: rotor axial inertia, a: rotor unit axis (body), w: body angular velocity (body axes),
    Om: rotor spin relative to the body, tau: total external torque about the COM (body axes),
    tau_axial: external torque on the rotor alone about its own axis. Returns (w_dot[3], Om_dot).
    Solves [[J, Is a], [Is a^T, Is]] [w_dot, Om_dot] = [tau - w x h, tau_axial], h = J w + Is Om a."""
    h0 = J[0, 0] * w[0] + J[0, 1] * w[1] + J[0, 2] * w[2] + Is * Om * a[0]
    h1 = J[1, 0] * w[0] + J[1, 1] * w[1] + J[1, 2] * w[2] + Is * Om * a[1]
    h2 = J[2, 0] * w[0] + J[2, 1] * w[1] + J[2, 2] * w[2] + Is * Om * a[2]
    A = np.empty((4, 4)); b = np.empty(4)
    for i in range(3):
        for j in range(3):
            A[i, j] = J[i, j]
        A[i, 3] = Is * a[i]; A[3, i] = Is * a[i]
    A[3, 3] = Is
    b[0] = tau[0] - (w[1] * h2 - w[2] * h1)
    b[1] = tau[1] - (w[2] * h0 - w[0] * h2)
    b[2] = tau[2] - (w[0] * h1 - w[1] * h0)
    b[3] = tau_axial
    # 4x4 symmetric positive-definite system: Gaussian elimination without pivoting (no LAPACK call per step)
    for c in range(4):
        piv = A[c, c]
        for r in range(c + 1, 4):
            f = A[r, c] / piv
            for q in range(c, 4):
                A[r, q] -= f * A[c, q]
            b[r] -= f * b[c]
    x = np.empty(4)
    for r in range(3, -1, -1):
        acc = b[r]
        for q in range(r + 1, 4):
            acc -= A[r, q] * x[q]
        x[r] = acc / A[r, r]
    return x[:3].copy(), x[3]


@block("rigid_body", "spatial", "gyrostat", "kinetic_energy", kind="kernel", source=SRC_G,
       tags=("spinning rotor", "energy ledger", "3d"))
@kernel
def gyrostat_energy(J, Is, a, w, Om):
    """Rotational kinetic energy of body + rotor: 0.5 w^T J w + Is Om (a . w) + 0.5 Is Om^2."""
    e = 0.0
    for i in range(3):
        for j in range(3):
            e += 0.5 * w[i] * J[i, j] * w[j]
    aw = a[0] * w[0] + a[1] * w[1] + a[2] * w[2]
    return e + Is * Om * aw + 0.5 * Is * Om * Om


@block("geometry", "penetration", "point_vs_box", "nearest_face", kind="kernel",
       source="signed distance to an axis-aligned box (interior: depth to the nearest face)",
       tags=("3d", "collision", "blade vs blade", "penalty contact"))
@kernel
def point_box_penetration(px, py, pz, hx, hy, hz):
    """Penetration of a point (box frame, box centred at origin, half-extents hx, hy, hz).
    Returns (pen, axis, sign): pen > 0 inside = depth to the nearest face, whose outward normal is
    sign * e_axis. pen <= 0 outside (not a true distance; only its sign is meaningful).
    The penalty potential 0.5 k pen^2 is continuous across nearest-face switches."""
    dx = hx - abs(px); dy = hy - abs(py); dz = hz - abs(pz)
    if dx <= dy and dx <= dz:
        return dx, 0, 1.0 if px >= 0 else -1.0
    if dy <= dz:
        return dy, 1, 1.0 if py >= 0 else -1.0
    return dz, 2, 1.0 if pz >= 0 else -1.0


@block("friction", "coulomb", "general", "regularized_3d", kind="kernel",
       source="regularized Coulomb friction (Oden & Martins, CMAME 52, 1985; smooth sqrt form)",
       tags=("3d", "sliding", "dissipative"))
@kernel
def coulomb_regularized_3d(vtx, vty, vtz, N, mu, v_eps):
    """Friction force opposing the tangential slip velocity vt: F = -mu N vt / sqrt(|vt|^2 + v_eps^2).
    Reaches mu N for |vt| >> v_eps; always dissipative (F . vt <= 0)."""
    s = mu * N / math.sqrt(vtx * vtx + vty * vty + vtz * vtz + v_eps * v_eps)
    return -s * vtx, -s * vty, -s * vtz


@block("rigid_body", "inertia", "box", "tensor", source="solid cuboid about its centroid (standard result)",
       tags=("3d", "chassis"))
def I_box_tensor(m, lx, ly, lz):
    """Inertia tensor of a solid box (full side lengths) about its centroid, box axes."""
    return np.diag([m * (ly ** 2 + lz ** 2) / 12, m * (lx ** 2 + lz ** 2) / 12, m * (lx ** 2 + ly ** 2) / 12])


@block("rigid_body", "inertia", "general", "parallel_axis_3d", source="Steiner theorem, tensor form")
def parallel_axis_tensor(I_cm, m, d):
    """Inertia tensor about a point displaced by -d from the COM: I_cm + m (|d|^2 E - d d^T)."""
    d = np.asarray(d, float)
    return np.asarray(I_cm, float) + m * (d @ d * np.eye(3) - np.outer(d, d))


@block("geometry", "distance", "box_vs_box", "separating_axis", kind="kernel",
       source="separating axis theorem for oriented boxes (Gottschalk, Lin & Manocha, SIGGRAPH 1996)",
       tags=("3d", "collision", "broad phase", "culling", "time step"))
@kernel
def obb_gap(c1, R1, h1, c2, R2, h2):
    """Lower bound on the distance between two oriented boxes (centres c, rotation matrices R whose columns are
    the box axes in world, half extents h): the largest separation along the 15 SAT axes. > 0 means the boxes
    are disjoint and at least that far apart; <= 0 means they may touch (not a penetration depth)."""
    best = -1e30
    for case in range(15):
        if case < 3:
            ax0 = R1[0, case]; ax1 = R1[1, case]; ax2 = R1[2, case]
        elif case < 6:
            ax0 = R2[0, case - 3]; ax1 = R2[1, case - 3]; ax2 = R2[2, case - 3]
        else:
            i = (case - 6) // 3; j = (case - 6) % 3
            a0 = R1[0, i]; a1 = R1[1, i]; a2 = R1[2, i]; b0 = R2[0, j]; b1 = R2[1, j]; b2 = R2[2, j]
            ax0 = a1 * b2 - a2 * b1; ax1 = a2 * b0 - a0 * b2; ax2 = a0 * b1 - a1 * b0
            nn = math.sqrt(ax0 * ax0 + ax1 * ax1 + ax2 * ax2)
            if nn < 1e-9:
                continue                      # parallel edges: covered by the face axes
            ax0 /= nn; ax1 /= nn; ax2 /= nn
        d = abs((c2[0] - c1[0]) * ax0 + (c2[1] - c1[1]) * ax1 + (c2[2] - c1[2]) * ax2)
        r1 = 0.0; r2 = 0.0
        for k in range(3):
            r1 += h1[k] * abs(R1[0, k] * ax0 + R1[1, k] * ax1 + R1[2, k] * ax2)
            r2 += h2[k] * abs(R2[0, k] * ax0 + R2[1, k] * ax1 + R2[2, k] * ax2)
        g = d - r1 - r2
        if g > best:
            best = g
    return best
