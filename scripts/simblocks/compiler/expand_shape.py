"""Body shapes: mass, centre of mass and inertia from dimensions and a material, density or mass.

    bodies:
      disc: {dof: planar, shape: {kind: disc, R: 40 mm, t: 6 mm, material: 4130_normalized}}
      bar:  {dof: free, shape: {kind: box, a: 120 mm, b: 20 mm, c: 6 mm, density: 2800 kg/m^3, at: [0, 0, 0.01]}}

kind and dimensions (the shape's own axes; axis z is the shape's axis of symmetry):
  rod        L, A (section area, for mass from density): slender bar along z (2D: along x)
  disc       R, t                                  ring   Ri, Ro, t
  plate      a (x), b (y), t (z)                   box    a, b, c (x, y, z)
  cylinder   R, L                                  tube   Ri, Ro, L                 sphere  R
2D shapes: rod, disc, ring, plate (a by b in the plane, t thick); the inertia is about the out-of-plane axis.
Give one of mass, density or material. Options:
  at:   where the shape's centre is in the body's frame (default the origin): it becomes the body's com
  axis: (3D) the world direction of the shape's z axis at the start (default z)
The body gets mass, com and inertia from the shape (it cannot also give them), and a feature named 'outline':
a circle (2D disc, ring) or sphere (3D sphere) at the shape's centre with its outer radius, ready for contacts."""
import numpy as np

from .loader import SpecError, line_of

KINDS_2D = ("rod", "disc", "ring", "plate")
KINDS_3D = ("rod", "disc", "ring", "plate", "box", "cylinder", "tube", "sphere")
SHAPE_KEYS = {"kind", "L", "A", "R", "Ri", "Ro", "t", "a", "b", "c", "mass", "density", "material", "at", "axis"}


def _rot_z_to(ax):
    """Rotation taking the z axis onto the unit vector ax."""
    z = np.array([0.0, 0.0, 1.0]); v = np.cross(z, ax); s = np.linalg.norm(v); c = float(z @ ax)
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / s
    th = np.arctan2(s, c)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def shape_props(sh, ctx, where, dim, f=""):
    from .model import resolve, num
    from ..rigid_body.inertia import mass_properties
    if not isinstance(sh, dict) or "kind" not in sh:
        raise SpecError("shape is {kind: box, a: .., b: .., c: .., material: ..}", where, None, f)
    bad = set(sh) - SHAPE_KEYS
    if bad:
        raise SpecError("unknown shape key(s): " + ", ".join(sorted(bad)), where, line_of(sh, sorted(bad)[0]), f)
    kind = str(sh["kind"]).strip().lower()
    ok = KINDS_3D if dim == "3d" else KINDS_2D
    if dim == "1d":
        ok = KINDS_3D
    if kind not in ok:
        raise SpecError(f"shape kind '{kind}' is not one of: {', '.join(ok)}" +
                        (" (in 2D a shape is flat: use plate for a rectangle)" if dim == "2d" else ""),
                        where, line_of(sh, "kind"), f)
    n = sum(k in sh for k in ("mass", "density", "material"))
    if n != 1:
        raise SpecError("a shape needs exactly one of mass, density or material", where, line_of(sh, "kind"), f)
    g = lambda k: num(resolve(sh[k], ctx, f"{where}.{k}"), f"{where}.{k}") if k in sh else None
    kw = {k: g(k) for k in ("L", "A", "R", "Ri", "Ro", "t", "a", "b", "c", "mass", "density")}
    if "material" in sh:
        kw["material"] = str(resolve(sh["material"], ctx, where + ".material"))
    mk = {"plate": "rect_plate"}.get(kind, kind)
    if dim == "2d" and kind == "plate" and kw["t"] is None and kw["mass"] is None:
        raise SpecError("a 2D plate from a density or material needs its thickness t", where, line_of(sh, "kind"), f)
    try:
        mp = mass_properties(mk, **{k: v for k, v in kw.items() if v is not None})
    except Exception as exc:
        raise SpecError(f"shape: {exc}", where, line_of(sh, "kind"), f)
    m = float(mp["m"])
    I = np.array(mp["I"], float)
    if dim == "2d":
        Izz = float(mp["Ixx"]) if kind == "rod" else float(mp["Izz"])     # a 2D rod lies in the plane
        return m, Izz, kind, kw
    if dim == "3d" and "axis" in sh:
        from .model3d import _vec, _unit
        Rz = _rot_z_to(_unit(_vec(sh["axis"], ctx, where + ".axis"), where + ".axis"))
        I = Rz @ I @ Rz.T
    return m, I, kind, kw


def expand_shapes(doc, ctx):
    """Fill mass / com / inertia (and an 'outline' feature) on bodies that give a shape. Returns a new doc."""
    bodies = doc.get("bodies") or {}
    if not any(isinstance(d, dict) and "shape" in d for d in bodies.values()):
        return doc
    import copy
    from .model import resolve, num
    f = doc.get("__file__", ""); dim = doc.get("dim", "1d")
    out = copy.copy(doc); out.lines = getattr(doc, "lines", {})
    nb = dict(bodies)
    for bn, d in bodies.items():
        if not isinstance(d, dict) or "shape" not in d:
            continue
        where = f"bodies.{bn}.shape"
        if d.get("ground"):
            raise SpecError("ground does not move: it needs no shape", where, line_of(d, "shape"), f)
        clash = [k for k in ("mass", "inertia", "com") if k in d]
        if clash:
            raise SpecError(f"this body gives both shape and {', '.join(clash)}: the shape sets them", where,
                            line_of(d, clash[0]), f)
        m, I, kind, kw = shape_props(d["shape"], ctx, where, dim, f)
        d = dict(d); d.pop("shape")
        d["mass"] = m
        dof = d.get("dof")
        if dim == "1d":
            if dof == "rotation":                  # a 1D rotation body takes the inertia about the shape's axis
                d.pop("mass"); d["inertia"] = float(I[2][2]) if not np.isscalar(I) else float(I)
            nb[bn] = d; continue
        n = 3 if dim == "3d" else 2
        d_at = [0.0] * n
        if "at" in doc["bodies"][bn]["shape"]:
            from .model2d import _vec as v2
            from .model3d import _vec as v3
            d_at = list((v3 if n == 3 else v2)(doc["bodies"][bn]["shape"]["at"], ctx, where + ".at"))
            d["com"] = d_at
        d["inertia"] = I.tolist() if n == 3 else I
        rr = kw.get("Ro") or kw.get("R")
        if (n == 2 and kind in ("disc", "ring")) or (n == 3 and kind == "sphere"):
            key = "circles" if n == 2 else "spheres"
            feats = dict(d.get(key) or {})
            if "outline" not in feats:
                feats["outline"] = {"at": d_at, "r": rr}
                d[key] = feats
        nb[bn] = d
    out["bodies"] = nb
    return out
