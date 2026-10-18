"""Compile a dim: 3d system file into the 3D engine's arrays (mechanics in engine3d.py). z is up.

Bodies
  name:
    dof: free                  default: [x, y, z, rotate]. Any of x, y, z (translation), plus either
                               rotate (free 3D rotation) or spin (about one fixed world axis, given by axis:)
    axis: [ax, ay, az]         spin axis for a spin body (any vector, normalised; or x, y, z, -x...)  default z
    mass, inertia              inertia about the centre of mass, in the body's own frame: a number (same about every
                               axis), [Ixx, Iyy, Izz], or a 3x3 list. A spin body may give just the number about its axis
    at: [x, y, z]              where the body's drawing origin starts (the pivot, for a body that only spins)
    com: [x, y, z]             centre of mass in the drawing frame (default: the origin)
    orient0: {axis: [..], angle: 30 deg}    starting orientation of the drawing frame (default: aligned with world)
    angle0                     starting spin angle about the axis (spin bodies)
    v0: [vx, vy, vz]           omega0: spin rate (spin) or [wx, wy, wz] world angular velocity (rotate)
    points: {p: [x, y, z]}     spheres: {s: {at: [..], r: ..}}
    planes: {face: {at: [..], normal: [..], u: [..], size: [a, b], depth: d}}   a finite rectangular patch
Measures: 'A.x - B.z + $r * C.spin' (coordinates x, y, z, spin), {distance: [A.p, B.q]}, {along: [A.p, B.q], dir: z},
{contact: [A.tip, B.face]} (point or sphere against a plane patch or a sphere), pin (ball joint), contact_friction."""
import math
import re

import numpy as np

from .loader import SpecError, line_of
from .expr import eval_measure, ExprError
from . import engine3d as E3
from .engine1d import NP

TRANS = ("x", "y", "z")
COORDS3 = ("x", "y", "z", "spin")
FEATURES3 = ("points", "spheres", "planes")
BODY_Q3 = {k: i for i, k in enumerate(("x", "y", "z", "vx", "vy", "vz", "speed", "angle", "omega", "wx", "wy", "wz"))}
NAMED = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1), "-x": (-1, 0, 0), "-y": (0, -1, 0), "-z": (0, 0, -1)}


def _vec(v, ctx, where, n=3):
    from .model import resolve, num
    if isinstance(v, str) and v.strip() in NAMED and n == 3:
        return list(map(float, NAMED[v.strip()]))
    if isinstance(v, str) and not v.strip().startswith(("$", "=")):
        v = [x.strip() for x in v.strip("[]() ").split(",")]
    val = resolve(v, ctx, where)
    if not isinstance(val, (list, tuple)) or len(val) != n:
        raise SpecError(f"expected a list of {n} numbers or quantities, got {v!r}", where)
    return [num(x, f"{where}[{i}]") for i, x in enumerate(val)]


def _unit(v, where):
    v = np.asarray(v, float); L = float(np.linalg.norm(v))
    if L == 0:
        raise SpecError("a direction must not be zero", where)
    return v / L


def rot(axis, ang):
    a = _unit(axis, "axis"); c, s = math.cos(ang), math.sin(ang); x, y, z = a
    return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])


def mat2quat(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q); return q / np.linalg.norm(q)


def parse_dof3(d, where):
    raw = d.get("dof", "free")
    if raw in ("free", None):
        raw = ["x", "y", "z", "rotate"]
    elif raw == "translation":
        raw = ["x", "y", "z"]
    elif raw in ("spin", "rotation"):
        raw = ["spin"]
    elif isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)) or any(c not in ("x", "y", "z", "spin", "rotate") for c in raw):
        raise SpecError("3d dof is a list of x, y, z plus spin (about axis:) or rotate (free), or 'free'", where)
    if "spin" in raw and "rotate" in raw:
        raise SpecError("pick one: spin (about one axis) or rotate (free 3D rotation)", where)
    if not raw:
        raise SpecError("a body with no free coordinate never moves: make it ground: true", where)
    tr = [c in raw for c in TRANS]
    rm = E3.R_SPIN if "spin" in raw else (E3.R_FREE if "rotate" in raw else E3.R_NONE)
    return tr, rm


def _physical(I, where):
    ev = np.linalg.eigvalsh(I)
    if ev[0] > 0 and ev[2] > ev[0] + ev[1] + 1e-9 * ev[2]:
        raise SpecError("this inertia is not physical: no body has one principal moment larger than the other two "
                        "together (" + ", ".join(f"{x:.3g}" for x in ev) + "). Check the numbers and that it is about "
                        "the centre of mass", where)
    return I



def _inertia(v, ctx, where):
    from .model import resolve, num
    val = resolve(v, ctx, where)
    if isinstance(val, (list, tuple)):
        if len(val) == 3 and not isinstance(val[0], (list, tuple)):
            return _physical(np.diag([num(x, where) for x in val]), where)
        if len(val) == 3 and all(isinstance(r, (list, tuple)) and len(r) == 3 for r in val):
            I = np.array([[num(x, where) for x in r] for r in val])
            if not np.allclose(I, I.T, rtol=1e-9, atol=1e-15):
                raise SpecError("an inertia tensor must be symmetric", where)
            return _physical(I, where)
        raise SpecError("inertia is a number, [Ixx, Iyy, Izz], a 3x3 list or {Ixx, Iyy, Izz, Pxy, Pxz, Pyz}", where)
    if isinstance(val, dict):
        # CAD mass-properties form (SolidWorks, Onshape, Fusion): moments Ixx, Iyy, Izz and products of inertia
        # Pxy = integral(x y dm) etc. The tensor's off-diagonal terms are minus the products. Taken about the COM.
        # Ixy / Ixz / Iyz (tensor entries, already negative-signed) are accepted instead of the P terms.
        ks = set(val)
        bad = ks - {"Ixx", "Iyy", "Izz", "Pxy", "Pxz", "Pyz", "Ixy", "Ixz", "Iyz"}
        if bad or not {"Ixx", "Iyy", "Izz"} <= ks:
            raise SpecError("inertia as a table needs Ixx, Iyy, Izz and optionally the products Pxy, Pxz, Pyz (as "
                            "CAD reports them, about the centre of mass); unknown: " + ", ".join(sorted(bad)), where)
        g = lambda k: num(resolve(val[k], ctx, f"{where}.{k}"), f"{where}.{k}") if k in val else 0.0
        off = {}
        for pk, ik in (("Pxy", "Ixy"), ("Pxz", "Ixz"), ("Pyz", "Iyz")):
            if pk in val and ik in val:
                raise SpecError(f"give {pk} (CAD product of inertia) or {ik} (tensor entry = -{pk}), not both", where)
            off[pk] = -g(pk) if pk in val else g(ik)
        I = np.array([[g("Ixx"), off["Pxy"], off["Pxz"]], [off["Pxy"], g("Iyy"), off["Pyz"]],
                      [off["Pxz"], off["Pyz"], g("Izz")]])
        ev = np.linalg.eigvalsh(I)
        if ev[0] <= 0 or ev[2] > ev[0] + ev[1] + 1e-9 * ev[2]:
            raise SpecError("this inertia is not physical (principal moments " + ", ".join(f"{x:.3g}" for x in ev) +
                            "): check the signs of the products (CAD Pxy is +integral(x y dm); the tensor uses "
                            "-Pxy) and that it is about the centre of mass", where)
        return I
    x = num(val, where)
    return np.eye(3) * x


def compile_bodies3(doc, ctx, M):
    from .model import resolve, num
    f = doc.get("__file__", "")
    bodies = doc["bodies"]
    names = [b for b, d in bodies.items() if not d.get("ground")]
    nb = len(names)
    X0 = np.zeros((nb, 3)); V0 = np.zeros((nb, 3)); Q0 = np.zeros((nb, 4)); Q0[:, 0] = 1.0
    Lw0 = np.zeros((nb, 3)); wsp0 = np.zeros(nb); th0 = np.zeros(nb); minv = np.zeros((nb, 3))
    rmode = np.zeros(nb, np.int64); Iinv = np.zeros((nb, 3, 3)); axis = np.zeros((nb, 3)); axis[:, 2] = 1.0
    Iax = np.zeros(nb); mass = np.zeros(nb); com = np.zeros((nb, 3))
    feats = {}; info = {}; reach_rot = np.zeros(nb)
    for bn, d in bodies.items():
        where = f"bodies.{bn}"
        if d.get("ground"):
            sh = np.zeros(3)
        else:
            i = names.index(bn)
            tr, rm = parse_dof3(d, where)
            m = num(resolve(d["mass"], ctx, where + ".mass"), where + ".mass") if "mass" in d else 0.0
            if any(tr) and m <= 0:
                raise SpecError("a body free to translate needs a positive mass", where, line_of(bodies, bn), f)
            at = np.array(_vec(d.get("at", [0, 0, 0]), ctx, where + ".at"))
            cm = np.array(_vec(d.get("com", [0, 0, 0]), ctx, where + ".com"))
            R0 = np.eye(3)
            if "orient0" in d:
                o = d["orient0"]
                if not isinstance(o, dict) or "axis" not in o or "angle" not in o:
                    raise SpecError("orient0 is {axis: [ax, ay, az], angle: 30 deg}", where + ".orient0")
                R0 = rot(_vec(o["axis"], ctx, where + ".orient0.axis"),
                         num(resolve(o["angle"], ctx, where + ".orient0.angle"), where + ".orient0.angle"))
            ax = np.array([0.0, 0.0, 1.0])
            if rm == E3.R_SPIN:
                ax = _unit(_vec(d.get("axis", "z"), ctx, where + ".axis"), where + ".axis")
                a0 = num(resolve(d.get("angle0", 0.0), ctx, where + ".angle0"), where + ".angle0")
                R0 = rot(ax, a0) @ R0
                th0[i] = a0
            elif "axis" in d:
                raise SpecError("axis: is the spin axis of a spin body (dof with spin)", where + ".axis")
            spin_only = rm == E3.R_SPIN and not any(tr)
            Ib = _inertia(d["inertia"], ctx, where + ".inertia") if "inertia" in d else None
            if rm != E3.R_NONE and Ib is None:
                raise SpecError("a body that turns needs inertia (about its centre of mass)", where, line_of(bodies, bn), f)
            cmw = R0 @ cm
            if spin_only:
                sh = np.zeros(3); com[i] = cm; X0[i] = at
                d_perp = cmw - (cmw @ ax) * ax
                Iaxis = float(ax @ (R0 @ Ib @ R0.T) @ ax) + m * float(d_perp @ d_perp)
                if float(d_perp @ d_perp) > 0 and m <= 0:
                    raise SpecError("a spinning body with its centre of mass off the axis needs its mass", where,
                                    line_of(bodies, bn), f)
            else:
                sh = cm; X0[i] = at + cmw
                Iaxis = float(ax @ (R0 @ Ib @ R0.T) @ ax) if Ib is not None else 0.0
            Q0[i] = mat2quat(R0)
            vv = np.array(_vec(d.get("v0", [0, 0, 0]), ctx, where + ".v0"))
            om = d.get("omega0", 0.0)
            if rm == E3.R_FREE:
                ov = resolve(om, ctx, where + ".omega0") if not (isinstance(om, str) and om.strip().startswith("[")) else om
                if isinstance(ov, (int, float)) and not isinstance(ov, bool) and float(ov) == 0.0:
                    w0 = np.zeros(3)
                elif isinstance(ov, (list, tuple)) or isinstance(ov, str):
                    w0 = np.array(_vec(ov, ctx, where + ".omega0"))
                else:
                    raise SpecError("a freely rotating body takes omega0 as a vector [wx, wy, wz] (world axes), "
                                    f"got {om!r}", where + ".omega0", line_of(d, "omega0"), f)
                Iw = R0 @ Ib @ R0.T
                Lw0[i] = Iw @ w0
                try:
                    Iinv[i] = np.linalg.inv(Ib)
                except np.linalg.LinAlgError:
                    raise SpecError("inertia must be invertible (all principal moments > 0)", where + ".inertia")
                if np.any(np.linalg.eigvalsh(Ib) <= 0):
                    raise SpecError("inertia must be positive in every direction", where + ".inertia")
                wvec = w0
            elif rm == E3.R_SPIN:
                if Iaxis <= 0:
                    raise SpecError("inertia about the spin axis must be positive", where + ".inertia")
                wsp0[i] = num(resolve(om, ctx, where + ".omega0"), where + ".omega0")
                Iax[i] = 1.0 / Iaxis; axis[i] = ax; wvec = ax * wsp0[i]
            else:
                wvec = np.zeros(3)
            V0[i] = (vv + np.cross(wvec, X0[i] - at)) if not spin_only else 0.0
            for k in range(3):
                minv[i, k] = (1.0 / m) if tr[k] else 0.0
                if not tr[k]:
                    V0[i, k] = 0.0
            rmode[i] = rm; mass[i] = m
            info[bn] = dict(trans=tr, rmode=rm, spin_only=spin_only, mass=m, R0=R0)
        for kind in FEATURES3:
            for fn, fv in (d.get(kind) or {}).items():
                fw = f"{where}.{kind}.{fn}"
                if fn in COORDS3:
                    raise SpecError(f"'{fn}' is a coordinate name; call the feature something else", fw)
                if (bn, fn) in feats:
                    raise SpecError(f"body '{bn}' has two features called '{fn}'", fw)
                if kind == "points":
                    p = np.array(_vec(fv, ctx, fw)) - sh
                    feats[(bn, fn)] = ("sphere", np.array([*p, 0.0]))
                elif kind == "spheres":
                    if not isinstance(fv, dict) or "r" not in fv:
                        raise SpecError("a sphere is {at: [x, y, z], r: radius}", fw)
                    p = np.array(_vec(fv.get("at", [0, 0, 0]), ctx, fw + ".at")) - sh
                    r = num(resolve(fv["r"], ctx, fw + ".r"), fw + ".r")
                    if r <= 0:
                        raise SpecError("a sphere needs a positive radius (use points for r = 0)", fw)
                    feats[(bn, fn)] = ("sphere", np.array([*p, r]))
                else:
                    if not isinstance(fv, dict) or "normal" not in fv:
                        raise SpecError("a plane is {at: [x, y, z], normal: [nx, ny, nz]} (optional u, size, depth)", fw)
                    o = np.array(_vec(fv.get("at", [0, 0, 0]), ctx, fw + ".at")) - sh
                    nrm = _unit(_vec(fv["normal"], ctx, fw + ".normal"), fw + ".normal")
                    if "u" in fv:
                        u = np.array(_vec(fv["u"], ctx, fw + ".u")); u = u - (u @ nrm) * nrm
                        u = _unit(u, fw + ".u (it must not be along the normal)")
                    else:
                        t = np.array([1.0, 0, 0]) if abs(nrm[0]) < 0.9 else np.array([0, 1.0, 0])
                        u = _unit(t - (t @ nrm) * nrm, fw)
                    if "size" in fv:
                        sz = _vec(fv["size"], ctx, fw + ".size", n=2)
                        hu, hw = 0.5 * sz[0], 0.5 * sz[1]
                    else:
                        hu = hw = 1e9
                    dep = num(resolve(fv["depth"], ctx, fw + ".depth"), fw + ".depth") if "depth" in fv \
                        else (min(2 * hu, 2 * hw) if hu < 1e8 else 1e9)
                    feats[(bn, fn)] = ("plane", np.array([*o, *nrm, *u, hu, hw, dep]))
        feats[(bn, "__origin__")] = ("sphere", np.array([*(-sh), 0.0]))
    M.body_names = names; M.dof = ["3d"] * nb; M.b3 = info
    return dict(X0=X0, V0=V0, quat0=Q0, Lw0=Lw0, wsp0=wsp0, th0=th0, minv=minv, rmode=rmode, Iinv=Iinv, axis=axis,
                Iax=Iax, mass=mass, com=com), feats


_REF = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)(?:\.([A-Za-z_][A-Za-z0-9_]*))?\s*$")


def feature3(ref, doc, feats, where):
    m = _REF.match(str(ref))
    if not m or m.group(1) not in doc["bodies"]:
        raise SpecError(f"'{ref}' is not body.feature (bodies: {', '.join(doc['bodies'])})", where)
    b, fn = m.group(1), m.group(2) or "__origin__"
    if (b, fn) not in feats:
        have = [k[1] for k in feats if k[0] == b and k[1] != "__origin__"]
        raise SpecError(f"body '{b}' has no point, sphere or plane called '{fn}'"
                        + (f" (it has: {', '.join(have)})" if have else " (it has none)"), where)
    return b, feats[(b, fn)]


def _wp(B, i, p):
    if i < 0:
        return np.asarray(p[:3], float)
    return B["X0"][i] + _qR(B["quat0"][i]) @ np.asarray(p[:3], float)


def _qR(q):
    R = np.zeros((3, 3)); E3._qmat(q[0], q[1], q[2], q[3], R); return R


_COORD3 = r"(?<![\w$.])({names})\.(x|y|z|spin)\b"


def linear_measure3(text, doc, names, nums, M):
    pat = re.compile(_COORD3.format(names="|".join(re.escape(b) for b in doc["bodies"])))
    src = pat.sub(lambda m: f"{m.group(1)}__{m.group(2)}", str(text))
    bare = [b for b in doc["bodies"] if re.search(rf"(?<![\w$.]){re.escape(b)}(?![\w.])", src)]
    if bare:
        raise SpecError(f"in 3d a measure names a coordinate: {bare[0]}.x, .y, .z or .spin "
                        "(or use distance / along / contact)", "")
    allb = [f"{b}__{c}" for b in doc["bodies"] for c in COORDS3]
    coef, const = eval_measure(src, allb, nums)
    C = np.zeros(4 * len(names))
    for k, c in coef.items():
        b, co = k.rsplit("__", 1)
        if b in names:
            if co == "spin" and M.b3[b]["rmode"] != E3.R_SPIN:
                raise SpecError(f"'{b}.spin' needs {b} to be a spin body (dof with spin and an axis)", "")
            C[4 * names.index(b) + COORDS3.index(co)] = c
    return C, const


def compile_elements3(doc, ctx, M, B, feats, overrides):
    from .model import resolve, num, TYPE_CODE, KNOCK, ELASTIC, numeric_namespace
    from .model2d import measure_text
    f = doc.get("__file__", "")
    names = M.body_names; nb = len(names)
    els = doc["elements"]; enames = list(els); ne = len(enames)
    typ = np.zeros(ne, np.int64); one = np.zeros(ne, np.int64); act = np.ones(ne, np.int64)
    mk = np.zeros(ne, np.int64); ba = -np.ones(ne, np.int64); bb = -np.ones(ne, np.int64); ref = -np.ones(ne, np.int64)
    P = np.zeros((ne, NP)); C = np.zeros((ne, 4 * nb)); s0 = np.zeros(ne)
    GA = np.zeros((ne, 4)); GB = np.zeros((ne, 24)); GX = np.zeros((ne, 5)); reach = np.zeros(ne)
    tabs_x, tabs_F, tab_k = [], [], np.zeros(ne)
    M.el_info = {}
    nums = numeric_namespace(ctx)
    bidx = lambda b: names.index(b) if b in names else -1
    rr = lambda i, p: float(np.linalg.norm(p[:3])) + (float(p[3]) if len(p) > 3 and p.shape[0] == 4 else 0.0) if i >= 0 else 0.0

    def g(d, key, where, default=None):
        return num(resolve(d[key], ctx, f"{where}.{key}"), f"{where}.{key}") if key in d else default

    for e, en in enumerate(enames):
        d = els[en]; where = f"elements.{en}"; t = d["type"]
        typ[e] = {"pin": 9, "contact_friction": 10, "rope": 2, "__rotlock__": 11, "beam": 12}.get(t, TYPE_CODE.get(t, -1))
        one[e] = 1 if (d.get("one_sided") or t == "rope") else 0
        meas = d.get("measure")
        if t == "rope" and not (isinstance(meas, dict) and "distance" in meas):
            raise SpecError("a rope's measure is {distance: [A.p, B.q], length: L} (it only pulls past L)", where,
                            line_of(d, "measure"), f)
        if t == "__rotlock__":                    # rotation lock of a weld / hinge (built by the expansion)
            a_, b_ = (str(r).split(".")[0] for r in meas)
            ia, ib = bidx(a_), bidx(b_)
            RA = _qR(B["quat0"][ia]) if ia >= 0 else np.eye(3); RB = _qR(B["quat0"][ib]) if ib >= 0 else np.eye(3)
            GB[e, :9] = (RA.T @ RB).reshape(-1)
            mk[e] = E3.MK_ROT; ba[e] = ia; bb[e] = ib
            P[e, 0] = g(d, "k_rot", where, 0.0 if d.get("__rigid__") else None); P[e, 1] = g(d, "c_rot", where, 0.0)
            if "hinge_axis" in d:
                aw = _unit(_vec(d["hinge_axis"], ctx, where + ".axis"), where + ".axis")
                GX[e, :3] = RA.T @ aw; GX[e, 3] = 1.0
            tabs_x.append(np.zeros(0)); tabs_F.append(np.zeros(0))
            if overrides.get(KNOCK + en) or overrides.get(KNOCK + en.split("__")[0]):
                act[e] = 0
            if d.get("__rigid__"):
                typ[e] = 15
            M.el_info[en] = dict(type="rotation lock", measure=f"turn of {b_} against {a_}", params=P[e].copy(), k=P[e, 0])
            continue
        if t == "beam":                          # co-rotational 3D beam segment (built by expand_beam)
            (a_, fa), (b_, fb) = (feature3(r, doc, feats, where + ".measure") for r in meas)
            if fa[0] != "sphere" or fb[0] != "sphere":
                raise SpecError("a beam joins two points", where, line_of(d, "measure"), f)
            ia, ib = bidx(a_), bidx(b_)
            mk[e] = E3.MK_BEAM; ba[e] = ia; bb[e] = ib
            GA[e, :3] = fa[1][:3]; GB[e, :3] = fb[1][:3]
            pa = _wp(B, ia, fa[1]); pb = _wp(B, ib, fb[1]); dv = pb - pa; L0 = float(np.linalg.norm(dv))
            e1 = dv / L0
            if "orient" in d:
                o = np.asarray(_vec(d["orient"], ctx, where + ".orient"), float)
            else:
                o = np.array([0.0, 0.0, 1.0]) if abs(e1[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            o = o - (o @ e1) * e1
            if np.linalg.norm(o) < 1e-9:
                raise SpecError("orient (the direction of the section's second axis) lies along the beam", where,
                                line_of(d, "orient"), f)
            e2 = o / np.linalg.norm(o); e3 = np.cross(e1, e2)
            E0 = np.column_stack([e1, e2, e3])
            RA = _qR(B["quat0"][ia]) if ia >= 0 else np.eye(3); RB = _qR(B["quat0"][ib]) if ib >= 0 else np.eye(3)
            GB[e, 3:12] = (RA.T @ E0).reshape(-1); GB[e, 12:21] = (RB.T @ E0).reshape(-1)
            M.beam_axes = getattr(M, "beam_axes", {}); M.beam_axes[e] = E0
            # e2 is the orient axis (bending about e2 uses EIy, about e3 uses EIz)
            P[e, 0] = d["__EA"] / L0; P[e, 1] = d["__EIy"] / L0; P[e, 2] = d["__EIz"] / L0
            P[e, 3] = d["__GJ"] / L0; P[e, 4] = L0
            zeta = g(d, "damping", where, 0.0)
            if zeta > 0:
                mseg = d.get("__mseg__") or min([1.0 / float(np.max(B["minv"][i])) for i in (ia, ib)
                                                 if i >= 0 and np.max(B["minv"][i]) > 0] or [1.0])
                P[e, 5] = 2.0 * zeta / math.sqrt(P[e, 0] / (0.5 * mseg))
                from .model2d import _beam_beta, beam_bend_damping
                P[e, 8], P[e, 9] = beam_bend_damping(zeta, _beam_beta(d, zeta, min(d["__EIy"], d["__EIz"]), names,
                                                                      B["minv"], None))
            hg = d.get("__hinge__"); tx = np.zeros(0); tF = np.zeros(0)
            if hg is not None:
                P[e, 6] = 1.0; P[e, 7] = float(hg.get("ry", 1.0)); tx = np.ascontiguousarray(hg["acc_tab"], float); tF = np.ascontiguousarray(hg["F_tab"], float)
            reach[e] = max(rr(ia, fa[1]), rr(ib, fb[1]))
            tabs_x.append(tx); tabs_F.append(tF)
            if overrides.get(KNOCK + en) or overrides.get(KNOCK + en.split("__")[0]):
                act[e] = 0
            M.el_info[en] = dict(type="beam", measure=f"beam {meas[0]} / {meas[1]}", params=P[e].copy(), k=P[e, 0],
                                 beam=dict(EA=d["__EA"], EIy=d["__EIy"], EIz=d["__EIz"], GJ=d["__GJ"], L0=L0))
            continue
        if isinstance(meas, dict) and "twist" in meas:
            refs = meas["twist"]
            if not isinstance(refs, (list, tuple)) or len(refs) != 2 or set(meas) - {"twist", "axis"}:
                raise SpecError("twist is {twist: [A, B], axis: z}: B's turn relative to A about an axis fixed in A "
                                "(given in world axes at the start)", where, line_of(d, "measure"), f)
            a_, b_ = (str(r).split(".")[0] for r in refs)
            for x in (a_, b_):
                if x not in doc["bodies"]:
                    raise SpecError(f"no body named '{x}'", where, line_of(d, "measure"), f)
            ia, ib = bidx(a_), bidx(b_)
            aw = _unit(_vec(meas.get("axis", "z"), ctx, where + ".measure.axis"), where + ".axis")
            RA = _qR(B["quat0"][ia]) if ia >= 0 else np.eye(3)
            GX[e, :3] = RA.T @ aw
            mk[e] = E3.MK_TWIST; ba[e] = ia; bb[e] = ib
            if t in ("pin", "rope"):
                raise SpecError(f"a {t} cannot act on a twist", where, line_of(d, "type"), f)
        if t == "contact_friction":
            on = d.get("contact")
            mo = (els.get(on) or {}).get("measure")
            if on not in els or on == en or not (isinstance(mo, dict) and "contact" in mo):
                raise SpecError("contact_friction needs contact: <the name of a contact element>", where,
                                line_of(d, "contact"), f)
            mk[e] = E3.MK_SLIP; ref[e] = enames.index(on)
            P[e, 0] = g(d, "mu", where); P[e, 1] = g(d, "v_eps", where, 0.01)
            if P[e, 0] < 0 or P[e, 1] <= 0:
                raise SpecError("mu must be >= 0 and v_eps > 0", where, line_of(d, "mu"), f)
            tabs_x.append(np.zeros(0)); tabs_F.append(np.zeros(0))
            if overrides.get(KNOCK + en) or overrides.get(KNOCK + en.split("__")[0]):
                act[e] = 0
            M.el_info[en] = dict(type=t, measure=f"slip on {on}", params=P[e].copy(), k=None)
            continue
        if isinstance(meas, dict) and "twist" in meas:
            pass                                       # handled above
        elif t == "pin" or isinstance(meas, dict):
            if t == "pin":
                refs = meas.get("pin") if isinstance(meas, dict) else meas; kind = "pin"
            else:
                kinds = [k for k in ("distance", "along", "contact") if k in meas]
                if len(kinds) != 1 or set(meas) - {"distance", "along", "contact", "length", "dir", "turns_with"}:
                    raise SpecError("a 3d measure is a coordinate expression ('A.z - B.z') or one of {distance: [A.p, "
                                    "B.q]}, {along: [A.p, B.q], dir: z}, {contact: [A.tip, B.face]}", where,
                                    line_of(d, "measure"), f)
                kind = kinds[0]; refs = meas[kind]
            if not isinstance(refs, (list, tuple)) or len(refs) != 2:
                raise SpecError(f"{kind} takes two features: [A.feature, B.feature]", where, line_of(d, "measure"), f)
            (a_, fa), (b_, fb) = (feature3(r, doc, feats, where + ".measure") for r in refs)
            if a_ == b_:
                raise SpecError("both features are on the same body", where, line_of(d, "measure"), f)
            if kind in ("pin", "distance", "along"):
                if fa[0] != "sphere" or fb[0] != "sphere":
                    raise SpecError(f"{kind} joins two points (or sphere centres)", where, line_of(d, "measure"), f)
                ia, ib = bidx(a_), bidx(b_)
                mk[e] = {"pin": E3.MK_PIN, "distance": E3.MK_DIST, "along": E3.MK_ALONG}[kind]
                ba[e] = ia; bb[e] = ib; GA[e, :3] = fa[1][:3]; GB[e, :3] = fb[1][:3]
                pa = _wp(B, ia, fa[1]); pb = _wp(B, ib, fb[1]); dv = pb - pa
                if kind == "distance":
                    L0 = float(np.linalg.norm(dv)); GX[e, :3] = dv / L0 if L0 > 0 else [1, 0, 0]
                elif kind == "along":
                    nvec = _unit(_vec(meas.get("dir", "z"), ctx, where + ".measure.dir"), where + ".dir")
                    GX[e, :3] = nvec; L0 = float(dv @ nvec)
                    tw = meas.get("turns_with")
                    if tw is not None:
                        if tw != a_:
                            raise SpecError("turns_with names the first body of the measure", where, line_of(d, "measure"), f)
                        if ia >= 0:
                            GX[e, :3] = _qR(B["quat0"][ia]).T @ nvec; GX[e, 4] = 1.0
                else:
                    L0 = 0.0
                if kind != "pin":
                    GX[e, 3] = num(resolve(meas["length"], ctx, where + ".length"), where + ".length") \
                        if "length" in meas else L0
                reach[e] = max(rr(ia, fa[1]), rr(ib, fb[1]))
            else:
                if fa[0] == "plane" and fb[0] == "plane":
                    raise SpecError("contact between two planes: put a point or sphere on one side (the corner that "
                                    "hits)", where, line_of(d, "measure"), f)
                if fa[0] == "plane":
                    (a_, fa), (b_, fb) = (b_, fb), (a_, fa)
                ia, ib = bidx(a_), bidx(b_)
                ba[e] = ia; bb[e] = ib; GA[e] = fa[1]
                if fb[0] == "plane":
                    mk[e] = E3.MK_PLANE; GB[e, :len(fb[1])] = fb[1]
                    c = _wp(B, ia, fa[1]); o = _wp(B, ib, fb[1][:3])
                    R = _qR(B["quat0"][ib]) if ib >= 0 else np.eye(3)
                    nw = R @ fb[1][3:6]
                    GX[e, 3] = 1.0 if float((c - o) @ nw) >= 0 else -1.0
                    if fb[1][9] < 1e8:
                        pr = float(np.linalg.norm(fb[1][:3])) + math.hypot(fb[1][9], fb[1][10])
                    else:                         # unbounded plane: the contact sits near where the point is
                        pr = float(np.linalg.norm(fb[1][:3])) + float(np.linalg.norm(c - o))
                    reach[e] = max(rr(ia, fa[1]), pr if ib >= 0 else 0.0)
                else:
                    if fa[1][3] == 0 and fb[1][3] == 0:
                        raise SpecError("contact between two points never touches: give one a radius (a sphere)",
                                        where, line_of(d, "measure"), f)
                    mk[e] = E3.MK_SPH; GB[e, :4] = fb[1]
                    reach[e] = max(rr(ia, fa[1]), rr(ib, fb[1]))
                if t in ("friction", "drag", "force", "drive", "emotor"):
                    raise SpecError(f"a {t} element cannot act on a contact measure (use contact_friction)", where,
                                    line_of(d, "type"), f)
                if t in ("kelvin_voigt", "elastoplastic"):
                    one[e] = 1
        else:
            try:
                C[e], s0[e] = linear_measure3(meas, doc, names, nums, M)
            except ExprError as exc:
                raise SpecError(f"measure: {exc}", where, line_of(d, "measure"), f)
            except SpecError as exc:
                raise SpecError(str(exc).split("] ")[-1].strip(), where, line_of(d, "measure"), f)
            mk[e] = E3.MK_LIN
        tx = np.zeros(0); tF = np.zeros(0)
        gg = lambda key, default=None: g(d, key, where, default)
        if d.get("rigid") is True or d.get("__rigid__"):   # exact joint: projected constraint (no stiffness)
            if mk[e] not in (E3.MK_PIN, E3.MK_LIN, E3.MK_ALONG, E3.MK_TWIST):
                raise SpecError("rigid applies to a pin, a joint (weld, hinge, slider) or a linear measure held "
                                "constant", where, line_of(d, "rigid"), f)
            typ[e] = 15
            if mk[e] == E3.MK_PIN:                       # an exact joint must start closed
                pa = _wp(B, ba[e], GA[e, :3]); pb = _wp(B, bb[e], GB[e, :3])
                gap = float(np.linalg.norm(pb - pa))
                if gap > 1e-6 + 1e-6 * float(np.linalg.norm(pa)):
                    raise SpecError(f"the rigid joint's two points are {gap * 1e3:.4g} mm apart at the start: they must "
                                    "meet (check the points, at and orient0 of both bodies)",
                                    where.split("__")[0], line_of(d, "measure"), f)
            tabs_x.append(tx); tabs_F.append(tF)
            if overrides.get(KNOCK + en) or overrides.get(KNOCK + en.split("__")[0]):
                act[e] = 0
            M.el_info[en] = dict(type="rigid " + t, measure=measure_text(meas), params=P[e].copy(), k=None)
            continue
        if t == "spring":
            P[e, 0] = gg("k")
        elif t == "dashpot":
            P[e, 0] = gg("c")
        elif t in ("kelvin_voigt", "pin", "rope"):
            P[e, 0] = gg("k"); P[e, 1] = gg("c", 0.0)
        elif t == "elastoplastic":
            tb = resolve(d["table"], ctx, f"{where}.table")
            if not isinstance(tb, dict) or not {"k", "acc_tab", "F_tab"} <= set(tb):
                raise SpecError("table must give k, acc_tab and F_tab", where, line_of(d, "table"), f)
            tab_k[e] = float(tb["k"]); P[e, 0] = tab_k[e]
            tx = np.ascontiguousarray(tb["acc_tab"], float); tF = np.ascontiguousarray(tb["F_tab"], float)
            if overrides.get(ELASTIC + en) or overrides.get(ELASTIC + en.split("__")[0]):
                tF = tF * 0 + 1e30
            P[e, 1] = gg("break_at", math.inf)
            from .model import strain_rate_params
            P[e, 2:5] = strain_rate_params(d, ctx, where, f)
            from .model import softening_factor
            tF = tF * softening_factor(d, ctx, where, f)       # Johnson-Cook: a hot part yields lower
            P[e, 5] = gg("set0", 0.0); P[e, 6] = gg("acc0", abs(P[e, 5]))      # carried in: set and its plastic travel
        elif t in ("gear", "bolt"):               # machine.py worked these out from the gear or bolt's geometry
            P[e, :len(d["__P__"])] = d["__P__"]
        elif t == "hertz":
            P[e, 0] = gg("K"); P[e, 1] = gg("n", 1.5); P[e, 2] = gg("alpha", 0.0)
        elif t == "friction":
            P[e, 0] = gg("F_c"); P[e, 1] = gg("v_eps"); P[e, 2] = gg("F_s", 0.0); P[e, 3] = gg("v_s", 0.0)
        elif t == "drag":
            P[e, 0] = gg("K")
        elif t == "impact":                      # rigid impact: restitution e, Coulomb mu, rest speed v_rest
            P[e, 0] = gg("e"); P[e, 1] = gg("mu", 0.0); P[e, 2] = gg("v_rest", 0.0)
            if not 0.0 <= P[e, 0] <= 1.0 or P[e, 1] < 0.0 or P[e, 2] < 0.0:
                raise SpecError("impact needs 0 <= e <= 1, mu >= 0 and v_rest >= 0", where, line_of(d, "e"), f)
        elif t == "force":
            if ("P" in d) + ("csv" in d) + ("pulse" in d) != 1:
                raise SpecError("a force needs one of P (a constant), csv (a table against time or the measure) or pulse "
                                "(a standard shape: half_sine, triangle, sine...)", where, line_of(d, "type"), f)
            if "pulse" in d:
                from .loadtable import compile_pulse
                typ[e], tx, tF = compile_pulse(e, d, ctx, where, doc, P, tab_k)
            elif "csv" in d:
                from .loadtable import compile_load
                typ[e], tx, tF = compile_load(e, d, ctx, where, doc, P, tab_k)
                from .loadtable import check_load_kind
                rot_ = (isinstance(meas, dict) and "twist" in meas) or (isinstance(meas, str) and ".angle" in meas)
                check_load_kind(d, rot_, where, doc)
            else:
                P[e, 0] = gg("P")
        elif t in ("drive", "emotor"):
            G = gg("gear", 1.0)
            if t == "drive":
                kv = gg("kv")
                P[e, :10] = [1.0 / kv, gg("R"), gg("I_max"), G, gg("cells"), gg("R0"), gg("R1", 0.0), gg("C1", 0.0),
                           gg("capacity"), gg("throttle", 1.0)]
            else:                                 # a circuit motor: torque from the circuit, speed back to it
                P[e, :4] = [G, gg("__idx__"), gg("__kt__"), gg("__R__")]
            nz = [k for k in range(4 * nb) if C[e, k] != 0.0]
            if mk[e] == E3.MK_TWIST:
                bi = bb[e]; Jr = gg("J_rotor", 0.0)
                if Jr > 0:
                    if bi < 0 or B["rmode"][bi] != E3.R_SPIN:
                        raise SpecError("J_rotor is added to a spin body; for a freely rotating rotor add it to the "
                                        "rotor's inertia", where, line_of(d, "J_rotor"), f)
                    B["Iax"][bi] = 1.0 / (1.0 / B["Iax"][bi] + Jr * G * G)
            elif mk[e] != E3.MK_LIN or len(nz) != 1 or nz[0] % 4 != 3:
                raise SpecError("a drive turns one spin coordinate ('shaft.spin') or a twist ({twist: [chassis, "
                                "weapon], axis: z} for a motor on a moving body)", where, line_of(d, "measure"), f)
            else:
                bi = nz[0] // 4
                B["Iax"][bi] = 1.0 / (1.0 / B["Iax"][bi] + gg("J_rotor", 0.0) * G * G)
        if overrides.get(KNOCK + en) or overrides.get(KNOCK + en.split("__")[0]):
            act[e] = 0
        tabs_x.append(tx); tabs_F.append(tF)
        M.el_info[en] = dict(type=t, measure=measure_text(meas), params=P[e].copy(),
                             k=tab_k[e] if t == "elastoplastic" else (P[e, 0] if t in ("spring", "kelvin_voigt", "pin") else None))
    for e in range(ne):
        if mk[e] == E3.MK_SLIP:
            ba[e] = ba[ref[e]]; bb[e] = bb[ref[e]]; reach[e] = reach[ref[e]]
    off = np.zeros(ne, np.int64); ln = np.zeros(ne, np.int64); pos = 0
    for e in range(ne):
        off[e] = pos; ln[e] = len(tabs_x[e]); pos += ln[e]
    M.el_names = enames; M.reach = reach
    return dict(typ=typ, one_sided=one, active=act, mk=mk, ba=ba, bb=bb, ref=ref, P=P, C=C, s0=s0, GA=GA, GB=GB,
                GX=GX, tab_off=off, tab_len=ln, tab_k=tab_k,
                tab_x=np.concatenate(tabs_x) if pos else np.zeros(1),
                tab_F=np.concatenate(tabs_F) if pos else np.zeros(1))


def gravity3(doc, ctx):
    from .model import resolve, num
    gv = doc.get("gravity")
    if gv in (None, 0, "none", "off"):
        return np.zeros(3)
    if isinstance(gv, list) or (isinstance(gv, str) and gv.strip().startswith("[")):
        return np.array(_vec(gv, ctx, "gravity"))
    return np.array([0.0, 0.0, -abs(num(resolve(gv, ctx, "gravity"), "gravity"))])


def body_slot3(M, body, qn, where):
    i = M.body_names.index(body); inf = M.b3[body]
    if qn in ("position", "velocity"):
        tr = [k for k, x in enumerate(inf["trans"]) if x]
        if inf["rmode"] == E3.R_NONE and len(tr) == 1:
            return 0, i * E3.NBQ + (tr[0] if qn == "position" else 3 + tr[0])
        if inf["spin_only"]:
            return 0, i * E3.NBQ + (7 if qn == "position" else 8)
        raise SpecError(f"'{qn}' is ambiguous for this 3d body: use x, y, z, vx, vy, vz, speed, angle, omega or wx/wy/wz",
                        where)
    if qn == "angle" and inf["rmode"] != E3.R_SPIN:
        raise SpecError("angle is the spin angle of a spin body; a free body reports wx, wy, wz and omega", where)
    if qn not in BODY_Q3:
        raise SpecError(f"unknown 3d body quantity '{qn}'", where)
    return 0, i * E3.NBQ + BODY_Q3[qn]


def compile3d(doc, ctx, M, overrides):
    B, feats = compile_bodies3(doc, ctx, M)
    EL = compile_elements3(doc, ctx, M, B, feats, overrides)
    M.dim = "3d"; M.feats = feats
    M.arrays = dict(**B, **EL, grav=gravity3(doc, ctx))
    for name, o in (doc.get("outputs") or {}).items():
        if "body" in o and o.get("quantity") != "kinetic_energy":
            if doc["bodies"][o["body"]].get("ground"):
                raise SpecError(f"'{o['body']}' is ground and never moves", f"outputs.{name}")
            body_slot3(M, o["body"], o["quantity"], f"outputs.{name}")
    return M


def rotation_dt3d(M):
    """Free rotation is accurate only while a step turns the body a little: at most 0.01 rad per step at the
    starting spin (orientation error then stays far below the energy tolerance)."""
    a = M.arrays; best = math.inf
    for b in range(len(M.body_names)):
        if a["rmode"][b] == E3.R_FREE:
            Rm = _qR(a["quat0"][b]); w = np.linalg.norm(Rm @ a["Iinv"][b] @ Rm.T @ a["Lw0"][b])
            if w > 0:
                best = min(best, 0.01 / w)
    return best


def critical_dt3d(M, skip=None):
    a = M.arrays; best = math.inf; bbest = math.inf; nb = len(M.body_names); M.dt_motor = math.inf
    rot_inv = np.zeros(nb)
    for b in range(nb):
        if a["rmode"][b] == E3.R_SPIN:
            rot_inv[b] = a["Iax"][b]
        elif a["rmode"][b] == E3.R_FREE:
            rot_inv[b] = float(np.max(np.linalg.eigvalsh(a["Iinv"][b])))
    def twist_inv(e):
        """1 / inertia about a twist element's own axis, summed over its two bodies (not the weakest body axis)."""
        ia = a["ba"][e]
        RA = _qR(a["quat0"][ia]) if ia >= 0 else np.eye(3)
        aw = RA @ np.asarray(a["GX"][e, :3], float)
        tot = 0.0
        for b in (a["ba"][e], a["bb"][e]):
            if b < 0:
                continue
            if a["rmode"][b] == E3.R_FREE:
                Rb = _qR(a["quat0"][b]); tot += float(aw @ (Rb @ a["Iinv"][b] @ Rb.T) @ aw)
            elif a["rmode"][b] == E3.R_SPIN:
                tot += a["Iax"][b] * float(aw @ a["axis"][b]) ** 2
        return tot
    for e in range(len(M.el_names)):
        if skip is not None and skip[e]:
            continue
        t = a["typ"][e]
        k = a["tab_k"][e] if t in (3, 13) else (a["P"][e, 0] if t in (0, 2, 9, 11, 30) else
                                                (a["P"][e, 0] + a["P"][e, 1] if t == 31 else 0.0))
        if t == 12:                              # beam: axial, transverse, bending and torsion modes
            L0 = a["P"][e, 4]; kax = a["P"][e, 0]; kt = a["P"][e, 3]
            E0 = M.beam_axes[e]
            for b in (a["ba"][e], a["bb"][e]):
                if b < 0:
                    continue
                mt = float(np.max(a["minv"][b]))
                if a["rmode"][b] == E3.R_FREE:
                    Rq = _qR(a["quat0"][b]); Iw = Rq @ a["Iinv"][b] @ Rq.T
                    ri = [float(E0[:, j] @ Iw @ E0[:, j]) for j in range(3)]
                elif a["rmode"][b] == E3.R_SPIN:
                    ri = [a["Iax"][b] * float(E0[:, j] @ a["axis"][b]) ** 2 for j in range(3)]
                else:
                    ri = [0.0, 0.0, 0.0]
                modes = ((kax, mt), (12.0 * a["P"][e, 1] / L0 ** 2, mt), (12.0 * a["P"][e, 2] / L0 ** 2, mt),
                         (4.0 * a["P"][e, 1], ri[1]), (4.0 * a["P"][e, 2], ri[2]), (kt, ri[0]))
                for kk, inv in modes:
                    if inv > 0 and kk > 0:
                        bbest = min(bbest, 2.0 / math.sqrt(kk * inv * 2.0))
            continue
        if t == 8 or t == 17:
            Kt, R_, G = ((a["P"][e, 0], a["P"][e, 1], a["P"][e, 3]) if t == 8 else
                        (a["P"][e, 2], a["P"][e, 3], a["P"][e, 0]))
            inv = sum((a["C"][e, 4 * b + 3] ** 2) * a["Iax"][b] for b in range(nb))
            if a["mk"][e] == E3.MK_TWIST:
                inv = twist_inv(e)
            if inv > 0:
                lim = 2.0 * R_ / (inv * (G * Kt) ** 2)
                # a drive steps itself exactly (engine1d _force) and a circuit motor is solved with its shaft
                # (circuit_step): neither is a stability limit. Its time constant only sets the accuracy cap, and the
                # engine needs its inertia (P[e, 10] = 1 / inertia along the measure)
                a["P"][e, 10] = inv
                M.dt_motor = min(M.dt_motor, 0.5 * lim)
            continue
        if k <= 0:
            continue
        if a["mk"][e] == E3.MK_TWIST:
            inv = twist_inv(e)
        elif a["mk"][e] == E3.MK_ROT:
            inv = sum(rot_inv[b] for b in (a["ba"][e], a["bb"][e]) if b >= 0)
        elif a["mk"][e] == E3.MK_LIN:
            inv = sum(a["C"][e, 4 * b + j] ** 2 * a["minv"][b, j] for b in range(nb) for j in range(3)) + \
                sum(a["C"][e, 4 * b + 3] ** 2 * a["Iax"][b] for b in range(nb))
        else:
            inv = 0.0
            for b in (a["ba"][e], a["bb"][e]):
                if b >= 0:
                    inv += float(np.max(a["minv"][b])) + rot_inv[b] * M.reach[e] ** 2
        if t in (30, 31) and a["P"][e, 6] > 0 and inv > 0:      # gear / bolt damping as a fraction of critical
            a["P"][e, 1 if t == 30 else 5] = 2.0 * a["P"][e, 6] * math.sqrt(k / inv)
        if inv > 0:
            best = min(best, 2.0 / math.sqrt(k * inv))
    # beams are kept stable at a larger share of their own limit (see _run_settings); contacts at the usual share
    M.dt_crit_beam = bbest; M.dt_crit_other = best if math.isfinite(best) else math.inf
    best_all = min(best, bbest)
    return best_all if math.isfinite(best_all) else M.t_end / 100.0
