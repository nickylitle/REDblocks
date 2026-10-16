"""Contact geometry: penetration measures and engagement tests (njit-safe kernels) plus the
first-contact angle distribution for a rotating blade closing on a surface."""
import math
import numpy as np

from ..registry import block
from .._jit import kernel


@block("geometry", "distance", "point_line", "signed", kind="kernel", source="vector projection")
@kernel
def point_line_signed_distance(px, py, ax, ay, nx, ny):
    """Signed distance of P from the line through A with unit normal n (positive on the +n side)."""
    return (px - ax) * nx + (py - ay) * ny


@block("geometry", "distance", "point_segment", "closest", kind="kernel", source="clamped projection",
       tags=("finite face", "lip", "slide off"))
@kernel
def point_segment_closest(px, py, ax, ay, bx, by):
    """Closest point on segment AB to P and the segment parameter s in [0,1]. Returns (cx, cy, s, dist)."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    s = ((px - ax) * dx + (py - ay) * dy) / L2 if L2 > 0 else 0.0
    if s < 0.0:
        s = 0.0
    elif s > 1.0:
        s = 1.0
    cx, cy = ax + s * dx, ay + s * dy
    return cx, cy, s, math.sqrt((px - cx) ** 2 + (py - cy) ** 2)


@block("geometry", "kinematics", "circle", "tip_position", kind="kernel", source="R (cos phi, sin phi)",
       tags=("blade tip", "tooth"))
@kernel
def tip_on_circle(R, phi, cx, cy):
    """Position of a tip at radius R and angle phi about (cx, cy). Returns (x, y)."""
    return cx + R * math.cos(phi), cy + R * math.sin(phi)


@block("geometry", "penetration", "tip_vs_hinged_face", "root_line", kind="kernel",
       source="kinematics of a hinged face (state-only penetration through the root)",
       tags=("ramp", "plow", "hinged face", "energy-consistent arm"))
@kernel
def tip_face_penetration(tip_x, root_x, root_z, a):
    """Perpendicular penetration of the tip into a face hinged at its root, face at angle a:
    pen = (tip_x - root_x) sin a + root_z cos a.  The torque arm r_arm = -d pen / d a is returned too,
    and is both the contact locator (0 = root, L = lip; outside = no contact) and the energy-consistent arm.
    Returns (pen, r_arm)."""
    pen = (tip_x - root_x) * math.sin(a) + root_z * math.cos(a)
    r_arm = -((tip_x - root_x) * math.cos(a) - root_z * math.sin(a))
    return pen, r_arm


@block("geometry", "penetration", "circle_vs_plane", "overlap", kind="kernel", source="geometry",
       tags=("wheel ground", "disk vs wall"))
@kernel
def circle_plane_overlap(cx, cy, R, ax, ay, nx, ny):
    """Overlap depth of a circle into the half-space behind the plane (A, n) (n points out of the solid)."""
    d = (cx - ax) * nx + (cy - ay) * ny
    return R - d if d < R else 0.0


@block("geometry", "first_contact", "head_on", "delta", source="blade-angle-first-contact.md",
       tags=("bite", "contact angle"))
def first_contact_delta(v_close, rpm, D, n_teeth=1):
    """Governing group delta = 120 v / (N rpm D): closing distance per tooth pass over blade radius."""
    return 120.0 * v_close / (n_teeth * rpm * D)


@block("geometry", "first_contact", "head_on", "distribution", source="blade-angle-first-contact.md",
       tags=("contact angle", "probability"))
def first_contact_stats(delta):
    """alpha_max, mean, median of the first-contact angle (rad before top-dead-center): f = sin(a)/delta."""
    amax = math.acos(1 - delta)
    mean = (math.sqrt(2 * delta - delta * delta) - (1 - delta) * amax) / delta
    median = math.acos(1 - delta / 2)
    return dict(alpha_max=amax, mean=mean, median=median)


@block("geometry", "first_contact", "head_on", "sample", kind="builder", source="inverse CDF of F = (1-cos a)/delta",
       tags=("Monte Carlo",))
def sample_first_contact(delta, n, rng=None):
    """Random first-contact angles (rad) by inverse transform: a = arccos(1 - delta U)."""
    rng = rng or np.random.default_rng()
    return np.arccos(1 - delta * rng.random(n))
