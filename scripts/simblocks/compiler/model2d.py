"""Compile a dim: 2d system file into the planar engine's arrays (see engine2d.py for the mechanics).

Bodies
  name:
    dof: [x, y, rz]          any subset ('planar' = all three; default). Frozen coordinates never move.
    mass, inertia            mass if x or y is free; inertia (about the centre of mass) if rz is free
    at: [x, y]               world position of the body's own drawing frame at t = 0 (default [0, 0])
    com: [x, y]              centre of mass in that frame (default [0, 0])
    angle0, v0: [vx, vy], omega0
    points: {tip: [x, y]}    circles: {rim: {at: [x, y], r: ...}}    lines: {face: {from: [x, y], to: [x, y], depth: ...}}
A body that only spins (dof [rz]) turns about its frame origin (the pivot); the compiler adds m d^2 for a centre of mass
off the pivot. Every other body is tracked at its centre of mass: A.x and A.y in measures and outputs are the centre
of mass. Ground bodies (ground: true) can carry points, circles and lines in world coordinates.

Measures (elements)
  'A.x - B.x + $r * A.rz'                         linear over coordinates (x, y, rz of each body)
  {distance: [A.p, B.q], length: L}               rest length L (default: the starting distance)
  {along: [A.p, B.q], dir: x | y | [dx, dy], length: L}
  {contact: [A.tip, B.face]}                      point or circle against a line or a circle; s = penetration
  pin elements:  measure: [A.hinge, B.hinge]     contact_friction elements:  contact: <contact element>"""
import math
import re

import numpy as np

from .loader import SpecError, line_of
from .expr import eval_measure, ExprError
from . import engine2d as E2
from .engine1d import NP

COORDS = ("x", "y", "rz")
FEATURE_KINDS = ("points", "circles", "lines")



def _beam_beta(d, zeta, EI, names, minv, stride):
    """The beam's first bending frequency, estimated as a cantilever: w1 = sqrt(3 EI / (m_eff L^3)),
    m_eff = 0.24 m_beam + the lighter moving end body (0 when unknown)."""
    L = d.get("__Ltot"); mb = d.get("__mtot") or 0.0
    if not L:
        return 0.0
    tip = []
    for bn in d.get("__ends") or []:
        if bn in names:
            i = names.index(bn)
            mi = minv[stride * i] if stride else float(np.max(minv[i]))
            if mi > 0:
                tip.append(1.0 / mi)
    meff = 0.24 * mb + (min(tip) if tip else 0.0)
    if meff <= 0 or EI <= 0:
        return 0.0
    return math.sqrt(3.0 * EI / (meff * L ** 3))


def beam_bend_damping(zeta, w1):
    """(beta, low-pass frequency) of the beam's bending damper: beta = 2 zeta / w1 aimed at the first mode, behind a
    low-pass at 3 w1. Above the cut-off the damper acts as a spring of beta wf k: capped at 2 k for stability."""
    if w1 <= 0 or zeta <= 0:
        return 0.0, 0.0
    wf = 3.0 * w1
    return min(2.0 * zeta / w1, 2.0 / wf), wf

def _vec(v, ctx, where, n=2):
    from .model import resolve, num
    if isinstance(v, str) and not v.strip().startswith(("$", "=")):
        v = [x.strip() for x in v.strip("[]() ").split(",")]
    val = resolve(v, ctx, where)
    if not isinstance(val, (list, tuple)) or len(val) != n:
        raise SpecError(f"expected [x, y] (a list of {n} numbers or quantities), got {v!r}", where)
    return [num(x, f"{where}[{i}]") for i, x in enumerate(val)]


def parse_dof(d, where):
    raw = d.get("dof", "planar")
    if raw in ("planar", None):
        return [True, True, True]
    if isinstance(raw, str):
        raw = {"translation": ["x", "y"], "rotation": ["rz"], "spin": ["rz"]}.get(raw, [raw])
    if not isinstance(raw, (list, tuple)) or any(c not in COORDS for c in raw):
        raise SpecError("dof is a list of the free coordinates: any of x, y, rz (or 'planar' for all three)", where)
    if not raw:
        raise SpecError("a body with no free coordinate never moves: make it ground: true", where)
    return [c in raw for c in COORDS]


class Body2:
    pass


def compile_bodies(doc, ctx, M):
    from .model import resolve, num
    f = doc.get("__file__", "")
    bodies = doc["bodies"]
    names = [b for b, d in bodies.items() if not d.get("ground")]
    nb = len(names)
    q0 = np.zeros(3 * nb); v0 = np.zeros(3 * nb); minv = np.zeros(3 * nb)
    mass = np.zeros(nb); com = np.zeros((nb, 2))
    feats = {}                        # (body, feature) -> (kind, data in the body frame used by the engine)
    info = {}
    for bn, d in bodies.items():
        where = f"bodies.{bn}"
        ground = bool(d.get("ground"))
        if ground:
            sh = np.zeros(2); rot0 = 0.0
        else:
            i = names.index(bn)
            free = parse_dof(d, where)
            at = _vec(d.get("at", [0, 0]), ctx, where + ".at")
            cm = _vec(d.get("com", [0, 0]), ctx, where + ".com")
            ang = num(resolve(d.get("angle0", 0.0), ctx, where + ".angle0"), where + ".angle0")
            vv = _vec(d.get("v0", [0, 0]), ctx, where + ".v0")
            om = num(resolve(d.get("omega0", 0.0), ctx, where + ".omega0"), where + ".omega0")
            m = num(resolve(d["mass"], ctx, where + ".mass"), where + ".mass") if "mass" in d else 0.0
            I = num(resolve(d["inertia"], ctx, where + ".inertia"), where + ".inertia") if "inertia" in d else 0.0
            if (free[0] or free[1]) and m <= 0:
                raise SpecError("a body free to move in x or y needs a positive mass", where, line_of(bodies, bn), f)
            if free[2] and I <= 0:
                raise SpecError("a body free to spin (rz) needs a positive inertia (about its centre of mass)", where,
                                line_of(bodies, bn), f)
            spin_only = free[2] and not (free[0] or free[1])
            c, s = math.cos(ang), math.sin(ang)
            if spin_only:
                sh = np.zeros(2)                                  # frame origin = pivot
                d2 = cm[0] ** 2 + cm[1] ** 2
                if d2 > 0 and m <= 0 and "com" in d:
                    raise SpecError("a spinning body with its centre of mass off the pivot needs its mass "
                                    "(for m d^2 and gravity)", where, line_of(d, "com"), f)
                Ieff = I + m * d2
                com[i] = cm; origin = at
            else:
                sh = np.array(cm)                                 # track the centre of mass
                Ieff = I; com[i] = (0.0, 0.0)
                origin = [at[0] + c * cm[0] - s * cm[1], at[1] + s * cm[0] + c * cm[1]]
            q0[3 * i:3 * i + 3] = [origin[0], origin[1], ang]
            # velocity of the tracked point: v0 is given for the body frame origin; add omega x r for a shifted COM
            rx, ry = origin[0] - at[0], origin[1] - at[1]
            if not free[2]:
                om = 0.0                                          # it cannot spin: omega0 has nothing to act on
            v0[3 * i:3 * i + 3] = [vv[0] - om * ry, vv[1] + om * rx, om]
            for k in range(3):
                if free[k]:
                    v0[3 * i + k] = v0[3 * i + k]
                    minv[3 * i + k] = 1.0 / (Ieff if k == 2 else m)
                else:
                    v0[3 * i + k] = 0.0
            mass[i] = m
            info[bn] = dict(free=free, mass=m, inertia=Ieff, spin_only=spin_only)
            rot0 = ang
        for kind in FEATURE_KINDS:
            for fn, fv in (d.get(kind) or {}).items():
                fw = f"{where}.{kind}.{fn}"
                if fn in COORDS:
                    raise SpecError(f"'{fn}' is a coordinate name; call the feature something else", fw)
                if (bn, fn) in feats:
                    raise SpecError(f"body '{bn}' has two features called '{fn}'", fw)
                if kind == "points":
                    p = np.array(_vec(fv, ctx, fw)) - sh
                    feats[(bn, fn)] = ("circle", (p[0], p[1], 0.0))
                elif kind == "circles":
                    if not isinstance(fv, dict) or "r" not in fv:
                        raise SpecError("a circle is {at: [x, y], r: radius}", fw)
                    p = np.array(_vec(fv.get("at", [0, 0]), ctx, fw + ".at")) - sh
                    r = num(resolve(fv["r"], ctx, fw + ".r"), fw + ".r")
                    if r <= 0:
                        raise SpecError("a circle needs a positive radius (use points for r = 0)", fw)
                    feats[(bn, fn)] = ("circle", (p[0], p[1], r))
                else:
                    if not isinstance(fv, dict) or "from" not in fv or "to" not in fv:
                        raise SpecError("a line is {from: [x, y], to: [x, y]} (optional depth)", fw)
                    a = np.array(_vec(fv["from"], ctx, fw + ".from")) - sh
                    b = np.array(_vec(fv["to"], ctx, fw + ".to")) - sh
                    L = float(np.hypot(*(b - a)))
                    if L <= 0:
                        raise SpecError("a line needs two different end points", fw)
                    dep = num(resolve(fv["depth"], ctx, fw + ".depth"), fw + ".depth") if "depth" in fv else L
                    feats[(bn, fn)] = ("line", (a[0], a[1], b[0], b[1], dep))
        feats[(bn, "__origin__")] = ("circle", (-sh[0], -sh[1], 0.0))
    M.body_names = names; M.dof = ["planar"] * nb
    M.b2 = info
    return dict(q0=q0, v0=v0, minv=minv, mass=mass, com=com), feats


_REF = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)(?:\.([A-Za-z_][A-Za-z0-9_]*))?\s*$")


def feature(ref, doc, feats, where):
    m = _REF.match(str(ref))
    if not m or m.group(1) not in doc["bodies"]:
        raise SpecError(f"'{ref}' is not body.feature (bodies: {', '.join(doc['bodies'])})", where)
    b, fn = m.group(1), m.group(2) or "__origin__"
    if (b, fn) not in feats:
        have = [k[1] for k in feats if k[0] == b and k[1] != "__origin__"]
        raise SpecError(f"body '{b}' has no point, circle or line called '{fn}'"
                        + (f" (it has: {', '.join(have)})" if have else " (it has none: add points/circles/lines)"), where)
    return b, feats[(b, fn)]


def _bidx(names, b):
    return names.index(b) if b in names else -1


def world_point(q0, i, lx, ly):
    if i < 0:
        return lx, ly
    c, s = math.cos(q0[3 * i + 2]), math.sin(q0[3 * i + 2])
    return q0[3 * i] + c * lx - s * ly, q0[3 * i + 1] + s * lx + c * ly


_COORD_RE = r"(?<![\w$.])({names})\.(x|y|rz)\b"


def linear_measure(text, doc, names, ctx_nums):
    pat = re.compile(_COORD_RE.format(names="|".join(re.escape(b) for b in doc["bodies"])))
    src = pat.sub(lambda m: f"{m.group(1)}__{m.group(2)}", str(text))
    allb = [f"{b}__{c}" for b in doc["bodies"] for c in COORDS]
    bare = [b for b in doc["bodies"] if re.search(rf"(?<![\w$.]){re.escape(b)}(?![\w.])", src)]
    if bare:
        raise SpecError(f"in 2d a measure names a coordinate: {bare[0]}.x, {bare[0]}.y or {bare[0]}.rz "
                        f"(or use distance / along / contact)", "")
    coef, const = eval_measure(src, allb, ctx_nums)
    C = np.zeros(3 * len(names))
    for k, c in coef.items():
        b, co = k.rsplit("__", 1)
        if b in names:
            C[3 * names.index(b) + COORDS.index(co)] = c
    return C, const


def compile_elements(doc, ctx, M, B, feats, overrides):
    """Fills element arrays. Element order: as written (contact_friction handled in a second sweep by the engine)."""
    from .model import resolve, num, TYPE_CODE, KNOCK, ELASTIC, numeric_namespace
    f = doc.get("__file__", "")
    names = M.body_names; nb = len(names)
    els = doc["elements"]; enames = list(els); ne = len(enames)
    typ = np.zeros(ne, np.int64); one = np.zeros(ne, np.int64); act = np.ones(ne, np.int64)
    mk = np.zeros(ne, np.int64); ba = -np.ones(ne, np.int64); bb = -np.ones(ne, np.int64); ref = -np.ones(ne, np.int64)
    P = np.zeros((ne, max(NP, 24))); C = np.zeros((ne, 3 * nb)); s0 = np.zeros(ne)
    GA = np.zeros((ne, 3)); GB = np.zeros((ne, 5)); GX = np.zeros((ne, 4))
    reach = np.zeros(ne)                      # largest lever radius per element (for the critical time step)
    tabs_x, tabs_F, tab_k = [], [], np.zeros(ne)
    M.el_info = {}
    nums = numeric_namespace(ctx)
    q0 = B["q0"]

    def g(d, key, where, default=None):
        return num(resolve(d[key], ctx, f"{where}.{key}"), f"{where}.{key}") if key in d else default

    def frame_r(i, pts):
        return max([math.hypot(x, y) for x, y in pts] + [0.0]) if i >= 0 else 0.0

    for e, en in enumerate(enames):
        d = els[en]; where = f"elements.{en}"; t = d["type"]
        typ[e] = {"pin": 9, "contact_friction": 10, "rope": 2, "beam": 12}.get(t, TYPE_CODE.get(t, -1))
        if t in VEHICLE_TYPES:                            # a vehicle element acts on one body as a whole
            _vehicle_element(e, en, d, where, t, doc, ctx, M, B, names, typ, mk, ba, P, act, tabs_x, tabs_F,
                             overrides, f)
            continue
        one[e] = 1 if (d.get("one_sided") or t == "rope") else 0
        meas = d.get("measure")
        if t == "rope" and not (isinstance(meas, dict) and "distance" in meas):
            raise SpecError("a rope's measure is {distance: [A.p, B.q], length: L} (it only pulls past L)", where,
                            line_of(d, "measure"), f)
        if t == "contact_friction":
            on = d.get("contact")
            if on not in els or on == en:
                raise SpecError("contact_friction needs contact: <the name of a contact element>", where,
                                line_of(d, "contact"), f)
            mo = els[on].get("measure")
            if not (isinstance(mo, dict) and "contact" in mo):
                raise SpecError(f"'{on}' is not a contact element (its measure must be contact: [...])", where,
                                line_of(d, "contact"), f)
            c = enames.index(on)
            mk[e] = E2.MK_SLIP; ref[e] = c
            P[e, 0] = g(d, "mu", where); P[e, 1] = g(d, "v_eps", where, 0.01)
            if P[e, 0] < 0 or P[e, 1] <= 0:
                raise SpecError("mu must be >= 0 and v_eps > 0", where, line_of(d, "mu"), f)
            tabs_x.append(np.zeros(0)); tabs_F.append(np.zeros(0))
            if overrides.get(KNOCK + en) or overrides.get(KNOCK + en.split("__")[0]):
                act[e] = 0
            M.el_info[en] = dict(type=t, measure=f"slip on {on}", params=P[e].copy(), k=None)
            continue
        # ---- measure
        if t == "beam":                                   # co-rotational beam (built by expand_beam)
            (a_, fa), (b_, fb) = (feature(r, doc, feats, where + ".measure") for r in meas)
            ia, ib = _bidx(names, a_), _bidx(names, b_)
            mk[e] = E2.MK_BEAM; ba[e] = ia; bb[e] = ib
            GA[e] = (fa[1][0], fa[1][1], 0.0); GB[e, :3] = (fb[1][0], fb[1][1], 0.0)
            pa = world_point(q0, ia, *fa[1][:2]); pb = world_point(q0, ib, *fb[1][:2])
            L0 = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
            EA, EI = d["__EA"], d["__EIz"]
            P[e, 0] = EA / L0; P[e, 1] = EI / L0; P[e, 4] = L0
            P[e, 5] = math.atan2(pb[1] - pa[1], pb[0] - pa[0])
            P[e, 6] = q0[3 * ia + 2] if ia >= 0 else 0.0; P[e, 7] = q0[3 * ib + 2] if ib >= 0 else 0.0
            zeta = num(resolve(d["damping"], ctx, where + ".damping"), where + ".damping") if "damping" in d else 0.0
            if zeta > 0:
                mseg = d.get("__mseg__") or min([1.0 / B["minv"][3 * i] for i in (ia, ib) if i >= 0 and B["minv"][3 * i] > 0] or [1.0])
                P[e, 2] = 2.0 * zeta / math.sqrt(P[e, 0] / (0.5 * mseg))
                P[e, 3], P[e, 9] = beam_bend_damping(zeta, _beam_beta(d, zeta, EI, names, B["minv"], 3))
            hg = d.get("__hinge__")
            tx = np.zeros(0); tF = np.zeros(0)
            if hg is not None:
                P[e, 8] = 1.0; tx = np.ascontiguousarray(hg["acc_tab"], float); tF = np.ascontiguousarray(hg["F_tab"], float)
            reach[e] = max(frame_r(ia, [fa[1][:2]]), frame_r(ib, [fb[1][:2]]))
            tabs_x.append(tx); tabs_F.append(tF)
            if overrides.get(KNOCK + en) or overrides.get(KNOCK + en.split("__")[0]):
                act[e] = 0
            M.el_info[en] = dict(type="beam", measure=f"beam {meas[0]} / {meas[1]}", params=P[e].copy(), k=P[e, 0],
                                 beam=dict(EA=EA, EI=EI, L0=L0))
            continue
        if t == "pin":
            refs = meas.get("pin") if isinstance(meas, dict) else meas
            if not isinstance(refs, (list, tuple)) or len(refs) != 2:
                raise SpecError("a pin's measure is [A.point, B.point]: the two points it holds together", where,
                                line_of(d, "measure"), f)
            (a_, fa), (b_, fb) = (feature(r, doc, feats, where + ".measure") for r in refs)
            if fa[0] != "circle" or fb[0] != "circle":
                raise SpecError("a pin joins two points (or circle centres)", where, line_of(d, "measure"), f)
            mk[e] = E2.MK_PIN; ba[e] = _bidx(names, a_); bb[e] = _bidx(names, b_)
            GA[e] = (fa[1][0], fa[1][1], 0.0); GB[e, :3] = (fb[1][0], fb[1][1], 0.0)
            reach[e] = max(frame_r(ba[e], [fa[1][:2]]), frame_r(bb[e], [fb[1][:2]]))
        elif isinstance(meas, dict) and "twist" in meas:
            refs = meas["twist"]
            if not isinstance(refs, (list, tuple)) or len(refs) != 2 or set(meas) - {"twist", "axis"}:
                raise SpecError("twist is {twist: [A, B]}: B's turn relative to A (zero at the start)", where,
                                line_of(d, "measure"), f)
            a_, b_ = (str(r).split(".")[0] for r in refs)
            for x in (a_, b_):
                if x not in doc["bodies"]:
                    raise SpecError(f"no body named '{x}'", where, line_of(d, "measure"), f)
            ia, ib = _bidx(names, a_), _bidx(names, b_)
            if ib >= 0:
                C[e, 3 * ib + 2] += 1.0
            if ia >= 0:
                C[e, 3 * ia + 2] -= 1.0
            s0[e] = -float(C[e] @ q0)                       # zero at the start
            mk[e] = E2.MK_LIN
        elif isinstance(meas, dict):
            kinds = [k for k in ("distance", "along", "contact") if k in meas]
            extra = set(meas) - {"distance", "along", "contact", "length", "dir", "turns_with"}
            if len(kinds) != 1 or extra:
                raise SpecError("a 2d measure is a coordinate expression ('A.x - B.x') or one of "
                                "{distance: [A.p, B.q]}, {along: [A.p, B.q], dir: x}, {contact: [A.tip, B.face]}",
                                where, line_of(d, "measure"), f)
            kind = kinds[0]; refs = meas[kind]
            if not isinstance(refs, (list, tuple)) or len(refs) != 2:
                raise SpecError(f"{kind} takes two features: [A.feature, B.feature]", where, line_of(d, "measure"), f)
            (a_, fa), (b_, fb) = (feature(r, doc, feats, where + ".measure") for r in refs)
            if a_ == b_:
                raise SpecError("both features are on the same body: a measure must join two bodies", where,
                                line_of(d, "measure"), f)
            if kind in ("distance", "along"):
                if fa[0] != "circle" or fb[0] != "circle":
                    raise SpecError(f"{kind} joins two points (or circle centres)", where, line_of(d, "measure"), f)
                ia, ib = _bidx(names, a_), _bidx(names, b_)
                mk[e] = E2.MK_DIST if kind == "distance" else E2.MK_ALONG
                ba[e] = ia; bb[e] = ib
                GA[e] = (fa[1][0], fa[1][1], 0.0); GB[e, :3] = (fb[1][0], fb[1][1], 0.0)
                pa = world_point(q0, ia, *fa[1][:2]); pb = world_point(q0, ib, *fb[1][:2])
                dx, dy = pb[0] - pa[0], pb[1] - pa[1]
                if kind == "distance":
                    L0 = math.hypot(dx, dy)
                    GX[e, 0], GX[e, 1] = ((dx / L0, dy / L0) if L0 > 0 else (1.0, 0.0))
                else:
                    dv = meas.get("dir", "x")
                    if isinstance(dv, str) and dv.strip().lstrip("-") in ("x", "y"):
                        sg = -1.0 if dv.strip().startswith("-") else 1.0
                        nx, ny = (sg, 0.0) if dv.strip().endswith("x") else (0.0, sg)
                    else:
                        nx, ny = _vec(dv, ctx, where + ".measure.dir")
                        L_ = math.hypot(nx, ny)
                        if L_ == 0:
                            raise SpecError("dir must not be zero", where, line_of(d, "measure"), f)
                        nx, ny = nx / L_, ny / L_
                    GX[e, 0], GX[e, 1] = nx, ny
                    L0 = dx * nx + dy * ny
                    tw = meas.get("turns_with")
                    if tw is not None:
                        if tw != a_:
                            raise SpecError("turns_with names the first body of the measure (the direction is fixed "
                                            "in it)", where, line_of(d, "measure"), f)
                        if ia >= 0:                       # store the direction in A's own frame
                            c_, s_ = math.cos(q0[3 * ia + 2]), math.sin(q0[3 * ia + 2])
                            GX[e, 0], GX[e, 1] = c_ * nx + s_ * ny, -s_ * nx + c_ * ny
                            GX[e, 3] = 1.0
                GX[e, 2] = num(resolve(meas["length"], ctx, where + ".measure.length"), where + ".length") \
                    if "length" in meas else L0
                reach[e] = max(frame_r(ia, [fa[1][:2]]), frame_r(ib, [fb[1][:2]]))
            else:                                                     # contact
                if fa[0] == "line" and fb[0] == "line":
                    raise SpecError("contact between two lines: put a point or circle on one side "
                                    "(the corner that hits)", where, line_of(d, "measure"), f)
                if fa[0] == "line":
                    (a_, fa), (b_, fb) = (b_, fb), (a_, fa)
                ia, ib = _bidx(names, a_), _bidx(names, b_)
                ba[e] = ia; bb[e] = ib
                GA[e] = fa[1]
                if fb[0] == "line":
                    mk[e] = E2.MK_EDGE; GB[e] = fb[1]
                    cx, cy = world_point(q0, ia, fa[1][0], fa[1][1])
                    x0, y0 = world_point(q0, ib, fb[1][0], fb[1][1]); x1, y1 = world_point(q0, ib, fb[1][2], fb[1][3])
                    tx, ty = x1 - x0, y1 - y0
                    left = (cx - x0) * (-ty) + (cy - y0) * tx
                    GX[e, 3] = 1.0 if left >= 0 else -1.0                 # outside = where the circle starts
                    reach[e] = max(frame_r(ia, [fa[1][:2]]) + fa[1][2], frame_r(ib, [fb[1][:2], fb[1][2:4]]))
                else:
                    if fa[1][2] == 0 and fb[1][2] == 0:
                        raise SpecError("contact between two points never touches: give one a radius (a circle)",
                                        where, line_of(d, "measure"), f)
                    mk[e] = E2.MK_CIRC; GB[e, :3] = fb[1]
                    reach[e] = max(frame_r(ia, [fa[1][:2]]) + fa[1][2], frame_r(ib, [fb[1][:2]]) + fb[1][2])
                if t in ("friction", "drag", "force", "drive", "emotor"):
                    raise SpecError(f"a {t} element cannot act on a contact measure (use contact_friction for "
                                    "sliding friction on a contact)", where, line_of(d, "type"), f)
                if t in ("kelvin_voigt", "elastoplastic"):
                    one[e] = 1                                            # contact pushes, never pulls
        else:
            try:
                C[e], s0[e] = linear_measure(meas, doc, names, nums)
            except ExprError as exc:
                raise SpecError(f"measure: {exc}", where, line_of(d, "measure"), f)
            except SpecError as exc:
                raise SpecError(str(exc).split("] ")[-1].strip(), where, line_of(d, "measure"), f)
            mk[e] = E2.MK_LIN
        # ---- law (same parameters as 1D)
        tx = np.zeros(0); tF = np.zeros(0)
        gg = lambda key, default=None: g(d, key, where, default)
        if d.get("rigid") is True or d.get("__rigid__"):   # exact joint: projected constraint (no stiffness)
            if mk[e] not in (E2.MK_PIN, E2.MK_LIN, E2.MK_ALONG):
                raise SpecError("rigid applies to a pin, a joint (weld, hinge, slider) or a linear measure held "
                                "constant", where, line_of(d, "rigid"), f)
            typ[e] = 15
            if mk[e] == E2.MK_PIN:                       # an exact joint must start closed
                pa = world_point(q0, ba[e], GA[e, 0], GA[e, 1]); pb = world_point(q0, bb[e], GB[e, 0], GB[e, 1])
                gap = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
                if gap > 1e-6 + 1e-6 * math.hypot(pa[0], pa[1]):
                    raise SpecError(f"the rigid joint's two points are {gap * 1e3:.4g} mm apart at the start: they must "
                                    "meet (check the points, at and angle0 of both bodies)",
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
            nz = [k for k in range(3 * nb) if C[e, k] != 0.0]
            if mk[e] != E2.MK_LIN or not nz or any(k % 3 != 2 for k in nz):
                raise SpecError("a drive turns spin coordinates: 'shaft.rz', or {twist: [chassis, shaft]} for a motor "
                                "on a moving body", where, line_of(d, "measure"), f)
            rot = max(nz, key=lambda k: C[e, k])               # the rotor: the body the measure counts positive
            B["minv"][rot] = 1.0 / (1.0 / B["minv"][rot] + gg("J_rotor", 0.0) * G * G) if B["minv"][rot] > 0 else 0.0
        if overrides.get(KNOCK + en) or overrides.get(KNOCK + en.split("__")[0]):
            act[e] = 0
        tabs_x.append(tx); tabs_F.append(tF)
        M.el_info[en] = dict(type=t, measure=measure_text(meas), params=P[e].copy(),
                             k=tab_k[e] if t == "elastoplastic" else (P[e, 0] if t in ("spring", "kelvin_voigt", "pin") else None))
    for e in range(ne):                       # friction rides on its contact's two bodies
        if mk[e] == E2.MK_SLIP:
            ba[e] = ba[ref[e]]; bb[e] = bb[ref[e]]; reach[e] = reach[ref[e]]
    off = np.zeros(ne, np.int64); ln = np.zeros(ne, np.int64); pos = 0
    for e in range(ne):
        off[e] = pos; ln[e] = len(tabs_x[e]); pos += ln[e]
    M.el_names = enames
    M.reach = reach
    return dict(typ=typ, one_sided=one, active=act, mk=mk, ba=ba, bb=bb, ref=ref, P=P, C=C, s0=s0, GA=GA, GB=GB,
                GX=GX, tab_off=off, tab_len=ln, tab_k=tab_k,
                tab_x=np.concatenate(tabs_x) if pos else np.zeros(1),
                tab_F=np.concatenate(tabs_F) if pos else np.zeros(1))


def measure_text(meas):
    """Readable one-line form of a measure for drawings and summaries."""
    if isinstance(meas, dict):
        for k in ("distance", "along", "contact", "pin", "twist"):
            if k in meas:
                return f"{k} {' / '.join(str(x) for x in meas[k])}" + (f" ({meas['dir']})" if "dir" in meas else "")
    if isinstance(meas, (list, tuple)):
        return "pin " + " / ".join(str(x) for x in meas)
    return str(meas)


def gravity(doc, ctx):
    from .model import resolve, num
    gv = doc.get("gravity")
    if gv in (None, 0, "none", "off") or isinstance(gv, dict):
        return np.zeros(2)
    if isinstance(gv, list) or (isinstance(gv, str) and gv.strip().startswith("[")):
        return np.array(_vec(gv, ctx, "gravity"))
    g = num(resolve(gv, ctx, "gravity"), "gravity")
    return np.array([0.0, -abs(g)])


def compile2d(doc, ctx, M, overrides):
    M.planet = planet_of(doc, ctx)
    B, feats = compile_bodies(doc, ctx, M)
    EL = compile_elements(doc, ctx, M, B, feats, overrides)
    M.dim = "2d"
    M.feats = feats
    M.arrays = dict(**B, **EL, grav=gravity(doc, ctx),
                    planet=M.planet if M.planet is not None else np.zeros(4))
    for name, o in (doc.get("outputs") or {}).items():           # body outputs must name one coordinate
        if "body" in o and o.get("quantity") != "kinetic_energy":
            if doc["bodies"][o["body"]].get("ground"):
                raise SpecError(f"'{o['body']}' is ground and never moves", f"outputs.{name}")
            body_slot(M, o["body"], o["quantity"], f"outputs.{name}")
    return M


def critical_dt2d(M, skip=None):
    a = M.arrays; best = math.inf; bbest = math.inf; nb = len(M.body_names); M.dt_motor = math.inf
    mi = a["minv"]
    for e in range(len(M.el_names)):
        if skip is not None and skip[e]:
            continue
        t = a["typ"][e]
        k = a["tab_k"][e] if t in (3, 13) else (a["P"][e, 0] if t in (0, 2, 9) else 0.0)
        if t == 8 or t == 17:
            Kt, R, G = ((a["P"][e, 0], a["P"][e, 1], a["P"][e, 3]) if t == 8 else
                        (a["P"][e, 2], a["P"][e, 3], a["P"][e, 0]))
            inv = sum(a["C"][e, j] ** 2 * mi[j] for j in range(3 * nb))
            if inv > 0:
                lim = 2.0 * R / (inv * (G * Kt) ** 2)
                # a drive steps itself exactly (engine1d _force) and a circuit motor is solved with its shaft
                # (circuit_step): neither is a stability limit. Its time constant only sets the accuracy cap, and the
                # engine needs its inertia (P[e, 10] = 1 / inertia along the measure)
                a["P"][e, 10] = inv
                M.dt_motor = min(M.dt_motor, 0.5 * lim)
            continue
        if t == 12:                                       # beam: axial, transverse and end-rotation modes
            L0 = a["P"][e, 4]; kax = a["P"][e, 0]; kb = a["P"][e, 1]
            for b in (a["ba"][e], a["bb"][e]):
                if b < 0:
                    continue
                mt = max(mi[3 * b], mi[3 * b + 1]); mr = mi[3 * b + 2]
                for kk, inv in ((kax, mt), (12.0 * kb / L0 ** 2, mt), (4.0 * kb, mr)):
                    if inv > 0 and kk > 0:
                        bbest = min(bbest, 2.0 / math.sqrt(kk * inv * 2.0))
            continue
        if k <= 0:
            continue
        if a["mk"][e] == E2.MK_LIN:
            inv = sum(a["C"][e, j] ** 2 * mi[j] for j in range(3 * nb))
        else:
            inv = 0.0
            for b in (a["ba"][e], a["bb"][e]):
                if b >= 0:
                    inv += max(mi[3 * b], mi[3 * b + 1]) + mi[3 * b + 2] * M.reach[e] ** 2
        if inv > 0:
            best = min(best, 2.0 / math.sqrt(k * inv))
    # beams are kept stable at a larger share of their own limit (see _run_settings); contacts at the usual share
    M.dt_crit_beam = bbest; M.dt_crit_other = best if math.isfinite(best) else math.inf
    best_all = min(best, bbest)
    return best_all if math.isfinite(best_all) else M.t_end / 100.0


# ---------------------------------------------------------------------------------------- outputs
BODY_Q2 = {"x": (0, 0), "y": (0, 1), "angle": (0, 2), "vx": (1, 0), "vy": (1, 1), "omega": (1, 2)}


def body_slot(M, body, qn, where):
    """(source, flat index) of a body quantity. position / velocity work for a body with one free coordinate."""
    i = M.body_names.index(body)
    if qn in ("position", "velocity"):
        free = [k for k, fr in enumerate(M.b2[body]["free"]) if fr]
        if len(free) != 1:
            raise SpecError(f"'{qn}' is ambiguous for a body with {len(free)} free coordinates in 2d: use "
                            f"{'x, y or angle' if qn == 'position' else 'vx, vy, omega or speed'}", where)
        return (0 if qn == "position" else 1), 3 * i + free[0]
    if qn == "speed":
        return 8, i
    src, k = BODY_Q2[qn]
    return src, 3 * i + k


# ---------------------------------------------------------------------------------------- vehicles in flight
VEHICLE_TYPES = {"rocket": 18, "jettison": 19, "aero_drag": 20, "steering": 21, "landing_guidance": 22}
THROTTLE_DEFAULTS = dict(start_level=0.10, start_hold=0.0, full=0.925, recovery=0.57, min=0.10, max=0.60)


def _vehicle_element(e, en, d, where, t, doc, ctx, M, B, names, typ, mk, ba, P, act, tabs_x, tabs_F, overrides, f):
    """rocket (engines on a body), jettison (a stage or part dropped at a time), aero_drag (drag from the
    atmosphere with a Cd table against Mach), steering (the thrust direction: vertical rise, pitch-over, gravity
    turn, tilt arrest, then guidance to a target orbit)."""
    from .model import resolve, num, KNOCK
    m = str(d.get("measure", "")).strip()
    if m not in names:
        raise SpecError(f"a {t} acts on one moving body: measure: <body name> (bodies: {', '.join(names)})", where,
                        line_of(d, "measure"), f)
    b = names.index(m)
    if B["minv"][3 * b] <= 0 and B["minv"][3 * b + 1] <= 0:
        raise SpecError(f"a {t} needs a body free to move in x and y", where, line_of(d, "measure"), f)
    if t in ("rocket", "jettison") and B["minv"][3 * b + 2] > 0:
        raise SpecError(f"a {t} changes its body's mass, so the body moves in x and y only (dof: [x, y]); "
                        "point it with a steering element", where, line_of(d, "measure"), f)
    g = lambda key, dflt=None: num(resolve(d[key], ctx, f"{where}.{key}"), f"{where}.{key}") if key in d else dflt
    typ[e] = VEHICLE_TYPES[t]; mk[e] = 0; ba[e] = b
    tx = np.zeros(0); tF = np.zeros(0)
    if t == "rocket":
        for k in ("thrust_vac", "mass_flow"):
            if k not in d:
                raise SpecError(f"a rocket needs {k} (from a rocket_engine block: $eng.F_vac, $eng.mdot)", where,
                                line_of(d, "type"), f)
        thr = d.get("throttle")
        lvl = None
        if thr not in (None, "guidance"):
            try:
                lvl = num(resolve(thr, ctx, f"{where}.throttle"), f"{where}.throttle")
            except Exception:
                raise SpecError("throttle: guidance (a landing_guidance element sets it), a level 0-1, or "
                                "{control: <signal>}", where, line_of(d, "throttle"), f)
        cut = d.get("cutoff")
        if cut not in (None, "guidance", "time"):
            raise SpecError("cutoff is guidance (the steering element's target speed stops it) or time (stop:)", where,
                            line_of(d, "cutoff"), f)
        P[e, :8] = [g("thrust_vac"), g("exit_area", 0.0), g("mass_flow"), g("start", 0.0), g("stop", 1e300),
                    g("rise", 0.0), g("tailoff", 0.0), 1.0 if cut == "guidance" else 0.0]
        P[e, 8] = 1.0 if thr == "guidance" else 0.0
        if lvl is not None:
            P[e, 10] = lvl; P[e, 11] = 1.0             # a throttle level, set or driven by a control signal
        P[e, 9] = g("propellant", 0.0)
        if P[e, 0] <= 0 or P[e, 2] <= 0 or P[e, 1] < 0:
            raise SpecError("thrust_vac and mass_flow must be positive, exit_area >= 0", where, line_of(d, "type"), f)
    elif t == "jettison":
        P[e, :2] = [g("at"), g("mass")]
        if P[e, 1] is None or P[e, 0] is None or P[e, 1] <= 0:
            raise SpecError("a jettison needs at (time) and mass (what is dropped)", where, line_of(d, "type"), f)
    elif t == "aero_drag":
        P[e, 0] = g("area")
        cd = resolve(d.get("Cd"), ctx, where + ".Cd")
        if isinstance(cd, (int, float)):
            tx = np.array([0.0]); tF = np.array([float(cd)])
        elif isinstance(cd, dict) and {"mach", "cd"} <= set(cd):
            tx = np.ascontiguousarray([float(x) for x in cd["mach"]]); tF = np.ascontiguousarray([float(x) for x in cd["cd"]])
            if len(tx) != len(tF) or len(tx) < 1 or np.any(np.diff(tx) <= 0):
                raise SpecError("Cd: {mach: [...], cd: [...]} with rising Mach numbers, one Cd each", where,
                                line_of(d, "Cd"), f)
        else:
            raise SpecError("Cd is a number or a table {mach: [...], cd: [...]}", where, line_of(d, "Cd"), f)
        if not P[e, 0] or P[e, 0] <= 0:
            raise SpecError("aero_drag needs area (the reference area its Cd is quoted on)", where, line_of(d, "area"), f)
    elif t == "landing_guidance":
        _landing_guidance(e, d, where, g, ctx, M, P, tabs_x, tabs_F, f)
        tx = tF = None
    else:                                                       # steering
        R = M.planet[1] if getattr(M, "planet", None) is not None else 0.0
        P[e, :8] = [g("pitch_start", 1e300), g("pitch_rate", 0.0), g("pitch_end", g("pitch_start", 1e300)),
                    g("tilt_arrest", g("pitch_end", 1e300)), g("guidance_start", 1e300),
                    R + g("target_altitude", 0.0), g("target_speed", 1e300), g("tgo_freeze", 10.0)]
        if "guidance_start" in d and ("target_altitude" not in d or "target_speed" not in d):
            raise SpecError("guidance needs target_altitude and target_speed (inertial, at cutoff)", where,
                            line_of(d, "guidance_start"), f)
        if "guidance_start" in d and R <= 0:
            raise SpecError("guidance to an orbit needs a planet: gravity: {planet: earth}", where,
                            line_of(d, "guidance_start"), f)
    if t == "aero_drag" and (getattr(M, "planet", None) is None or M.planet[3] < 0.5):
        raise SpecError("aero_drag needs a planet with an atmosphere: gravity: {planet: earth}", where,
                        line_of(d, "type"), f)
    if overrides.get(KNOCK + en) or overrides.get(KNOCK + en.split("__")[0]):
        act[e] = 0
    if tx is not None:
        tabs_x.append(tx); tabs_F.append(tF)
    M.el_info[en] = dict(type=t, measure=m, params=P[e].copy(), k=None)


def _landing_guidance(e, d, where, g, ctx, M, P, tabs_x, tabs_F, f):
    """Apollo lunar-landing guidance for one phase. mode target (P63 braking, P64 approach): the quadratic law
    a = a_T + 6 (v_T + v) / t_go + 12 (r_T - r) / t_go^2 (t_go negative) in altitude and range to the site, aimed at
    a target state at a set time. mode rate (P66): follow a schedule of climb rate and forward speed with a first-
    order lag (time constant tau). Both set the thrust direction and the throttle of throttle: guidance rockets:
    a low start level, fixed full throttle until the thrust asked for falls below the recovery level, then the
    throttle follows the ask between min and max."""
    from .model import resolve, num
    if getattr(M, "planet", None) is None:
        raise SpecError("landing guidance needs a planet: gravity: {planet: moon}", where, line_of(d, "type"), f)
    mode = str(d.get("mode", "target")).strip().lower()
    if mode not in ("target", "rate"):
        raise SpecError("mode is target (aim at a state at a time: P63, P64) or rate (follow a descent-rate "
                        "schedule: P66)", where, line_of(d, "mode"), f)
    th = dict(THROTTLE_DEFAULTS)
    tv = d.get("throttle") or {}
    if not isinstance(tv, dict) or set(tv) - set(th):
        raise SpecError(f"throttle: {{{', '.join(th)}}} (fractions of the rated thrust; start_hold is a time)", where,
                        line_of(d, "throttle"), f)
    for k in tv:
        th[k] = num(resolve(tv[k], ctx, f"{where}.throttle.{k}"), f"{where}.throttle.{k}")
    after = str(d.get("start", "")).strip().lower() == "after"
    P[e, 0] = -1.0 if after else g("start", 0.0); P[e, 1] = g("stop", 1e300); P[e, 2] = 1.0 if mode == "rate" else 0.0
    P[e, 3] = g("site_range", 0.0)
    P[e, 11:17] = [th["start_level"], th["start_hold"], th["full"], th["recovery"], th["min"], th["max"]]
    P[e, 23] = g("stop_below", 0.0)               # the phase also ends at this altitude
    if mode == "target":
        tg = d.get("target")
        keys = ("altitude", "range", "speed_forward", "speed_up", "accel_forward", "accel_up")
        if not isinstance(tg, dict) or "time" not in tg or set(tg) - set(keys) - {"time", "jerk_forward"}:
            raise SpecError(f"target: {{time, {', '.join(keys)}, jerk_forward}}: the state to arrive at (range: to the "
                            "site, negative short of it; time: how long after the phase starts)", where,
                            line_of(d, "target"), f)
        val = lambda k: num(resolve(tg[k], ctx, f"{where}.target.{k}"), f"{where}.target.{k}") if k in tg else 0.0
        P[e, 4] = val("time")
        P[e, 5:11] = [val(k) for k in keys]
        P[e, 17] = g("freeze", 5.0)
        if "jerk_forward" in tg:              # Apollo: the time to go comes from the downrange jerk condition
            P[e, 18] = val("jerk_forward"); P[e, 20] = 1.0
        P[e, 21] = g("end_before", 0.0)
        tabs_x.append(np.zeros(0)); tabs_F.append(np.zeros(0))
    else:
        sch = d.get("schedule")
        if not isinstance(sch, dict) or not {"time", "speed_up", "speed_forward"} <= set(sch):
            raise SpecError("schedule: {time: [...], speed_up: [...], speed_forward: [...]} (times from the phase start; "
                            "climb rate negative going down)", where, line_of(d, "schedule"), f)
        cols = {k: [num(resolve(x, ctx, f"{where}.schedule.{k}"), f"{where}.schedule.{k}") for x in sch[k]]
                for k in ("time", "speed_up", "speed_forward")}
        n = len(cols["time"])
        if n < 1 or any(len(c) != n for c in cols.values()) or np.any(np.diff(cols["time"]) <= 0):
            raise SpecError("schedule columns need the same length and rising times", where, line_of(d, "schedule"), f)
        tt = np.array(cols["time"], float)
        tabs_x.append(np.concatenate([tt, tt])); tabs_F.append(np.array(cols["speed_up"] + cols["speed_forward"], float))
        P[e, 18] = g("tau", 1.5); P[e, 19] = g("contact_altitude", 0.0)


PLANETS = {"earth": dict(mu=3.986004418e14, radius=6371000.0, rotation=7.2921159e-5, atmosphere="us76"),
           # Moon: GM from the JPL DE ephemerides, IAU mean radius, sidereal spin; no atmosphere
           "moon": dict(mu=4.9028e12, radius=1737400.0, rotation=2.6617e-6, atmosphere="none")}


def planet_of(doc, ctx):
    """gravity: {planet: earth | moon, radius, mu, rotation, atmosphere: us76 | none} -> [mu, R, in-plane
    rotation, 1 if air], else None."""
    from .model import resolve, num
    gv = doc.get("gravity")
    if not isinstance(gv, dict):
        return None
    bad = set(gv) - {"planet", "radius", "mu", "rotation", "atmosphere", "note"}
    if bad or "planet" not in gv:
        raise SpecError("gravity is a number, a vector, or {planet: earth, radius: ..., mu: ..., rotation: ...}",
                        "gravity", line_of(doc, "gravity"))
    base = PLANETS.get(str(gv["planet"]).lower())
    if base is None and "mu" not in gv:
        raise SpecError(f"unknown planet '{gv['planet']}' (known: {', '.join(PLANETS)}; or give mu and radius)",
                        "gravity", line_of(doc, "gravity"))
    base = base or {}
    g = lambda k: num(resolve(gv[k], ctx, f"gravity.{k}"), f"gravity.{k}") if k in gv else base.get(k, 0.0)
    atm = str(gv.get("atmosphere", base.get("atmosphere", "none"))).strip().lower()
    if atm not in ("us76", "none"):
        raise SpecError("atmosphere is us76 (the US Standard Atmosphere 1976) or none", "gravity", line_of(doc, "gravity"))
    return np.array([g("mu"), g("radius"), g("rotation"), 1.0 if atm == "us76" else 0.0])
