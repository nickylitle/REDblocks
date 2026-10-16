"""Mass properties: primitive shapes, parallel axis, assemblies, and polygon (drawing/CAD outline) plates.
Axis convention: 'z' = axis normal to the plane of a disk/plate (spin axis for a disk spinning in its own plane)."""
import math
import numpy as np

from ..registry import block

SRC = "standard mass-property tables (e.g. Meriam & Kraige App. B)"


@block("rigid_body", "inertia", "solid_disk", "z", source=SRC, tags=("rotor", "flywheel"))
def I_disk(m, R):
    """Solid disk about its axis: m R^2 / 2."""
    return 0.5 * m * R * R


@block("rigid_body", "inertia", "annulus", "z", source=SRC, tags=("ring", "drum rotor"))
def I_annulus(m, Ri, Ro):
    """Annular disk / thick ring about its axis: m (Ri^2 + Ro^2) / 2."""
    return 0.5 * m * (Ri * Ri + Ro * Ro)


@block("rigid_body", "inertia", "thin_ring", "z", source=SRC)
def I_thin_ring(m, R):
    """Thin hoop about its axis: m R^2."""
    return m * R * R


@block("rigid_body", "inertia", "rod", "center", source=SRC, tags=("rotating bar", "link"))
def I_rod_center(m, L):
    """Slender rod about its center, perpendicular: m L^2 / 12."""
    return m * L * L / 12.0


@block("rigid_body", "inertia", "rod", "end", source=SRC)
def I_rod_end(m, L):
    """Slender rod about one end: m L^2 / 3."""
    return m * L * L / 3.0


@block("rigid_body", "inertia", "rect_plate", "z", source=SRC, tags=("chassis", "panel"))
def I_rect_plate(m, a, b):
    """Rectangular plate a x b about the normal axis through its center: m (a^2 + b^2) / 12."""
    return m * (a * a + b * b) / 12.0


@block("rigid_body", "inertia", "cylinder", "diameter", source=SRC, tags=("drum", "shaft"))
def I_cylinder_transverse(m, R, L):
    """Solid cylinder about a transverse axis through its center: m (3R^2 + L^2) / 12."""
    return m * (3 * R * R + L * L) / 12.0


@block("rigid_body", "inertia", "sphere", "center", source=SRC)
def I_sphere(m, R):
    """Solid sphere: 2 m R^2 / 5."""
    return 0.4 * m * R * R


@block("rigid_body", "inertia", "general", "parallel_axis", source="Steiner theorem")
def parallel_axis(I_cm, m, d):
    """I about a parallel axis at distance d: I_cm + m d^2."""
    return I_cm + m * d * d


@block("rigid_body", "inertia", "polygon_plate", "z", kind="builder",
       source="Green's theorem polygon mass properties", tags=("drawing", "blade outline", "CAD", "holes"))
def polygon_plate(xy, t, rho, holes=()):
    """Mass, centroid and polar inertia (about the centroid and about the origin) of a plate of thickness t
    with outline xy [(x,y),...] and optional hole outlines. Works directly from a traced drawing."""
    def props(P):
        P = np.asarray(P, float)
        x, y = P[:, 0], P[:, 1]
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        cr = x * y1 - x1 * y
        A = cr.sum() / 2
        Sx = ((x + x1) * cr).sum() / 6
        Sy = ((y + y1) * cr).sum() / 6
        Jo = ((x * x + x * x1 + x1 * x1 + y * y + y * y1 + y1 * y1) * cr).sum() / 12
        s = 1.0 if A > 0 else -1.0
        return s * A, s * Sx, s * Sy, s * Jo
    A, Sx, Sy, Jo = props(xy)
    for h in holes:
        a, sx, sy, jo = props(h)
        A -= a; Sx -= sx; Sy -= sy; Jo -= jo
    m = rho * t * A
    cx, cy = Sx / A, Sy / A
    J_origin = rho * t * Jo
    return dict(m=m, area=A, cx=cx, cy=cy, I_origin=J_origin, I_cm=J_origin - m * (cx * cx + cy * cy))


@block("rigid_body", "inertia", "assembly", "planar", kind="builder",
       source="composite-body mass properties (parallel axis); workflow from a user-supplied drive sim", tags=("COM", "yaw inertia", "vehicle"))
def assemble(components):
    """Planar assembly from components [(name, m, x, y, I_own), ...] -> total m, COM, Izz about COM.
    Exclude bearing-decoupled spinning parts from chassis yaw inertia (their I_own = 0)."""
    m = sum(c[1] for c in components)
    xc = sum(c[1] * c[2] for c in components) / m
    yc = sum(c[1] * c[3] for c in components) / m
    I = sum(c[1] * ((c[2] - xc) ** 2 + (c[3] - yc) ** 2) + c[4] for c in components)
    return dict(m=m, xcom=xc, ycom=yc, Izz=I)


@block("rigid_body", "inertia", "assembly", "solve_remainder", kind="builder",
       source="moment balance; workflow from a user-supplied drive sim", tags=("CAD weight share", "unknown chassis"))
def solve_remainder_centroid(components, m_total, x_com_target):
    """Centroid x of the unlisted remainder mass so the assembly hits a measured total mass and COM."""
    known_m = sum(c[1] for c in components)
    m_rem = m_total - known_m
    x_rem = (m_total * x_com_target - sum(c[1] * c[2] for c in components)) / m_rem
    return m_rem, x_rem


# ---------------------------------------------------------------------------------------------- generic primitive
_SHAPE_DIMS = {
    "rod": ("L",), "disc": ("R",), "ring": ("Ri", "Ro"), "thin_ring": ("R",), "rect_plate": ("a", "b"),
    "box": ("a", "b", "c"), "cylinder": ("R", "L"), "tube": ("Ri", "Ro", "L"), "sphere": ("R",),
}
_SHAPE_ALIAS = {"disk": "disc", "solid_disk": "disc", "annulus": "ring", "annular_disk": "ring", "hoop": "thin_ring",
                "plate": "rect_plate", "rod_center": "rod", "bar": "rod"}


@block("rigid_body", "inertia", "primitives", "tensor", kind="builder", source=SRC,
       tags=("rotor", "flywheel", "chassis", "drum", "shaft", "ring", "box", "sphere", "COM", "inertia tensor"))
def mass_properties(shape, mass=None, density=None, material=None, L=None, R=None, Ri=None, Ro=None,
                    a=None, b=None, c=None, t=None, A=None):
    """Mass, centre of mass and inertia tensor of a uniform primitive shape, about its centroid.

    shape and dimensions (axes in brackets):
      rod         L (length along z); slender: Ixx = Iyy = m L^2/12, Izz = 0
      disc        R, thickness t (axis z; t only for mass from density and the transverse terms)
      ring        Ri, Ro, thickness t (annulus, axis z): Izz = m (Ri^2 + Ro^2)/2
      thin_ring   R (hoop, axis z): Izz = m R^2, Ixx = Iyy = m R^2/2; give the wire section area A for mass
      rect_plate  a along x, b along y, thickness t along z: Izz = m (a^2 + b^2)/12
      box         a, b, c along x, y, z
      cylinder    R, L (axis z): Izz = m R^2/2, Ixx = Iyy = m (3R^2 + L^2)/12
      tube        Ri, Ro, L (axis z): Izz = m (Ri^2 + Ro^2)/2, Ixx = Iyy = m (3(Ri^2 + Ro^2) + L^2)/12
      sphere      R: 2 m R^2 / 5 about every axis
    Give one of mass (kg), density (kg/m^3) or material (catalog name; its rho is used). The rod takes A
    (cross-section area) for its volume. Planar (2D) use: take Izz.
    Returns dict(m, com [0, 0, 0], Ixx, Iyy, Izz, Ixy, Ixz, Iyz (all 0 for these shapes), I (3x3 nested list)).
    Use parallel_axis or parallel_axis_tensor to move the result to another point."""
    sh = str(shape).strip().lower()
    sh = _SHAPE_ALIAS.get(sh, sh)
    if sh not in _SHAPE_DIMS:
        raise ValueError(f"mass_properties: unknown shape {shape!r}; use one of {', '.join(_SHAPE_DIMS)}")
    dims = dict(L=L, R=R, Ri=Ri, Ro=Ro, a=a, b=b, c=c)
    miss = [k for k in _SHAPE_DIMS[sh] if dims[k] is None]
    if miss:
        raise ValueError(f"mass_properties '{sh}' needs {', '.join(miss)}")
    tt = 0.0 if t is None else t
    if mass is None:
        rho = density
        if rho is None:
            if material is None:
                raise ValueError("mass_properties needs one of mass, density or material")
            from ..materials.lookup import as_material
            rho = as_material(material).rho
        vol = {"rod": lambda: (A or 0.0) * L, "disc": lambda: math.pi * R * R * tt,
               "ring": lambda: math.pi * (Ro * Ro - Ri * Ri) * tt, "thin_ring": lambda: 2 * math.pi * R * (A or 0.0),
               "rect_plate": lambda: a * b * tt, "box": lambda: a * b * c, "cylinder": lambda: math.pi * R * R * L,
               "tube": lambda: math.pi * (Ro * Ro - Ri * Ri) * L, "sphere": lambda: 4.0 / 3.0 * math.pi * R ** 3}[sh]()
        if vol <= 0.0:
            need_ = {"rod": "A", "thin_ring": "A"}.get(sh, "t")
            raise ValueError(f"mass_properties '{sh}' from density needs '{need_}'")
        m = rho * vol
    else:
        m = mass
    if sh == "rod":
        Ixx = Iyy = I_rod_center(m, L); Izz = 0.0
    elif sh == "disc":
        Izz = I_disk(m, R); Ixx = Iyy = I_cylinder_transverse(m, R, tt)
    elif sh == "ring":
        Izz = I_annulus(m, Ri, Ro); Ixx = Iyy = m * (3 * (Ri * Ri + Ro * Ro) + tt * tt) / 12.0
    elif sh == "thin_ring":
        Izz = I_thin_ring(m, R); Ixx = Iyy = 0.5 * m * R * R
    elif sh == "rect_plate":
        Izz = I_rect_plate(m, a, b); Ixx = m * (b * b + tt * tt) / 12.0; Iyy = m * (a * a + tt * tt) / 12.0
    elif sh == "box":
        from .spatial import I_box_tensor
        T = I_box_tensor(m, a, b, c)
        Ixx, Iyy, Izz = float(T[0, 0]), float(T[1, 1]), float(T[2, 2])
    elif sh == "cylinder":
        Izz = I_disk(m, R); Ixx = Iyy = I_cylinder_transverse(m, R, L)
    elif sh == "tube":
        Izz = I_annulus(m, Ri, Ro); Ixx = Iyy = m * (3 * (Ri * Ri + Ro * Ro) + L * L) / 12.0
    else:
        Ixx = Iyy = Izz = I_sphere(m, R)
    return dict(m=m, com=[0.0, 0.0, 0.0], Ixx=Ixx, Iyy=Iyy, Izz=Izz, Ixy=0.0, Ixz=0.0, Iyz=0.0,
                I=[[Ixx, 0.0, 0.0], [0.0, Iyy, 0.0], [0.0, 0.0, Izz]])
