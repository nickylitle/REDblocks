"""Animation scene for the editor's viewer: what each body looks like (in its own frame) and where it is at each
logged instant, for 1d, 2d and 3d models. Built from the compiled model and its run history, so the picture is
exactly what the solver used. The editor draws it with three.js.

Frames: every body's position and orientation (quaternion w, x, y, z, world from body) at each time.
- 3d: the solver's own centre-of-mass positions and orientation quaternions.
- 2d: x, y and the angle about z (drawn in the x-y plane, z = 0).
- 1d: each body in its own lane; a sliding body moves along x by its coordinate (scaled up so small travels show,
  the scale is reported), a turning body spins about the lane's z axis by its angle.
"""
import math

import numpy as np

# element types drawn as springs, dampers or stiff links between their two attachment points
SPRINGY = {"spring", "kelvin_voigt", "hertz", "elastoplastic", "rope", "bumper", "spring_nonlinear"}
DAMPY = {"dashpot", "damper"}
LINKY = {"pin", "weld", "hinge", "slider", "beam"}


def _q_axis(axis, ang):
    s = math.sin(0.5 * ang)
    return [math.cos(0.5 * ang), axis[0] * s, axis[1] * s, axis[2] * s]


def _hull(points):
    """Faces of the convex hull of 4+ body points (triangles, indices), or None when they are flat or few."""
    if len(points) < 4:
        return None
    try:
        from scipy.spatial import ConvexHull
        P = np.asarray(points, float)
        if np.linalg.matrix_rank(P - P.mean(axis=0), tol=1e-9) < 3:
            return None
        h = ConvexHull(P)
        return h.simplices.tolist()
    except Exception:
        return None


def _thin(H, n):
    t = np.asarray(H["t"], float)
    if len(t) <= n:
        idx = np.arange(len(t))
    else:
        idx = np.unique(np.round(np.linspace(0, len(t) - 1, n)).astype(int))
    return t[idx], idx


MAX_FRAMES = 8000
MAX_STEP_DEG = 20.0          # the most a body may turn between frames (its spin about its own axis aside) to blend cleanly


def _build(M, H, out, frames):
    dim = getattr(M, "dim", "1d")
    t, idx = _thin(H, frames)
    S = dict(dim=dim, name=M.doc.get("name", "model"), t=t.tolist(), bodies=[], links=[], notes=[],
             timed_out=bool(out.get("_timed_out")), t_stopped=float(out.get("_t_stopped", 0.0) or 0.0))
    if dim == "3d":
        _scene3(M, H, idx, S)
    elif dim == "2d":
        _scene2(M, H, idx, S)
    else:
        _scene1(M, H, idx, S)
    _magnify(S)
    _extent(S)
    S["empty"] = not any(b.get("pos") for b in S["bodies"])
    S["signals"] = S.pop("spin_signals", []) + _signals(M, H, idx)
    return S


def _max_turn_deg(S):
    """The biggest turn of any body between two frames (a fast spinner's own spin left out: that is drawn apart)."""
    worst = 0.0
    for b in S["bodies"]:
        Q = (b.get("spin") or {}).get("swing") or b.get("quat")
        if not Q or len(Q) < 2 or b.get("hidden"):
            continue
        Q = np.asarray(Q, float); d = np.abs(np.sum(Q[1:] * Q[:-1], axis=1)).clip(0, 1)
        worst = max(worst, float(np.degrees(2 * np.arccos(d)).max()))
    return worst


def scene(system, overrides=None, frames=900, H=None, out=None, timeout="model"):
    """The JSON-able scene: bodies, links, frames, extent. Uses the history H of a run already made (the editor's
    last Run) when given, otherwise runs the model once (timeout: seconds of wall clock, as run_model takes it; a
    run stopped by it is replayed as far as it got). A body that turns more than 20 degrees between frames (a fast
    precession, a tumble) gets more frames, up to 8,000, so blending between them stays true."""
    M = system.compile(overrides)

    def fresh(n):
        from ..compiler.model import run_model
        from ..compiler.system import _with_blocks
        r = run_model(M, history=True, n_log=n, timeout=timeout)
        return _with_blocks(M, r, True)
    if H is None:
        out, H = fresh(frames)
    out = out or {}
    S = _build(M, H, out, frames)
    worst = _max_turn_deg(S)
    if worst > MAX_STEP_DEG and frames < MAX_FRAMES and not S["timed_out"]:
        need = int(min(MAX_FRAMES, math.ceil(frames * worst / MAX_STEP_DEG * 1.1)))
        if len(H["t"]) < need:
            out, H = fresh(need)
            need = len(H["t"])                       # every logged point: even steps (thinning would skip some)
        S = _build(M, H, out, need)
        S["frames_note"] = need
    return S


def _qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _spin_split(body, t, Q, w_axis, axis, label):
    """A body turning faster than the frames can follow (a top, a wheel, a flywheel: more than about a third of a turn
    between logged frames) would seem to crawl or run backwards (the wagon-wheel effect). Its orientation is split into
    the slow part (where its axis points: the swing, safe to blend) and the turn about its own axis, sent as a rate;
    the viewer turns it at that rate, slowed to what a screen can show, and the readout gives the true rate."""
    t = np.asarray(t, float); w_axis = np.asarray(w_axis, float)
    if len(t) < 3:
        return None
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(w_axis).all() or float(np.max(np.abs(w_axis))) * dt < 2.0:
        return None
    a = np.asarray(axis, float); a = a / (np.linalg.norm(a) or 1.0)
    sw = []
    for q in np.asarray(Q, float):
        p = float(q[1:] @ a); tw = np.array([q[0], *(p * a)]); n = np.linalg.norm(tw)
        tw = tw / n if n > 1e-12 else np.array([1.0, 0, 0, 0])
        s = _qmul(q, np.array([tw[0], -tw[1], -tw[2], -tw[3]]))
        if sw and float(s @ sw[-1]) < 0:            # keep neighbours on the same side, so blending takes the short way
            s = -s
        sw.append(s)
    body["spin"] = dict(axis=a.tolist(), w=np.round(w_axis, 6).tolist(), swing=np.round(np.array(sw), 7).tolist())
    return dict(key=f"{label}.spin", label=f"{label} spin", unit="rpm", group="body", show=True,
                v=np.round(w_axis * 30 / math.pi, 4).tolist(), lo=float(np.min(w_axis) * 30 / math.pi),
                hi=float(np.max(w_axis) * 30 / math.pi))


def _qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _qrot(q, v):
    return _qmul(_qmul(q, np.array([0.0, *v])), _qconj(q))[1:]


def _smooth(x, sigma=3.0):
    """Frames x 3 smoothed along the frames (a Gaussian of about 3 frames; the ends held)."""
    from scipy.ndimage import gaussian_filter1d
    x = np.asarray(x, float)
    return gaussian_filter1d(x, sigma, axis=0, mode="nearest") if len(x) > 8 else x


def _magnify(S):
    """What "motion x" scales: deformation only. Bodies hang in a tree from the fixed world (or, for a free machine,
    from its heaviest body) along the links that hold them: springs and beams are elastic (their relative motion is
    deformation and gets scaled), joints are not (a hinge's swing is the mechanism moving: drawn as it is, the child
    just follows its parent). Bodies held by nothing (a car on a road, a flying point mass) move as they do. The
    viewer rebuilds each child from its parent's drawn pose and its own pose relative to the parent, scaled about
    where that started. auto: when the largest deformation is under 2 % of the model's size, it is drawn at about 15 %."""
    bodies = S["bodies"]; nb = len(bodies)
    mov = [i for i, b in enumerate(bodies) if b.get("pos")]
    if not mov:
        return
    world = {i for i, b in enumerate(bodies) if not b.get("pos")}      # ground bodies: fixed, the world itself
    adj = {}
    for L in S["links"]:
        if L.get("kind") not in ("spring", "beam", "joint"):
            continue
        a = -1 if (L["a"] < 0 or L["a"] in world) else L["a"]; b = -1 if (L["b"] < 0 or L["b"] in world) else L["b"]
        if a == b:
            continue
        el = L["kind"] != "joint"
        adj.setdefault(a, []).append((b, el)); adj.setdefault(b, []).append((a, el))
    parent = {}; order = []
    seen = {-1}
    def grow(root):
        q = [root]
        while q:
            u = q.pop(0)
            for v, el in sorted(adj.get(u, []), key=lambda x: not x[1] and 0):
                if v in seen:
                    continue
                seen.add(v); parent[v] = (u, el); order.append(v); q.append(v)
    grow(-1)
    for i in sorted(mov, key=lambda i: -(bodies[i].get("mass") or 0.0)):   # free machines: from their heaviest body
        if i not in seen and adj.get(i):
            seen.add(i); grow(i)
    tree = []; worst = 0.0
    for v in order:
        if v < 0 or not bodies[v].get("pos"):
            continue
        u, el = parent[v]
        Pv = np.asarray(bodies[v]["pos"], float); Qv = np.asarray(bodies[v]["quat"], float)
        if u < 0:
            Pu = np.zeros_like(Pv); Qu = np.tile([1.0, 0, 0, 0], (len(Pv), 1))
        else:
            Pu = np.asarray(bodies[u]["pos"], float); Qu = np.asarray(bodies[u]["quat"], float)
        r0 = _qrot(_qconj(Qu[0]), Pv[0] - Pu[0])
        ent = dict(b=v, p=u, el=bool(el))
        if el:
            # the deformation (pose relative to the parent, less where it started), smoothed over a few frames: a
            # stiff link rings far faster than the frames can follow, and scaled up that ringing is only twitching
            r = np.array([_qrot(_qconj(qu), pv - pu) for qu, pv, pu in zip(Qu, Pv, Pu)])
            q0 = _qmul(_qconj(Qu[0]), Qv[0]); rv = []
            for qu, qv in zip(Qu, Qv):
                dq = _qmul(_qconj(q0), _qmul(_qconj(qu), qv))
                if dq[0] < 0:
                    dq = -dq
                sn = float(np.linalg.norm(dq[1:])); ang = 2 * math.atan2(sn, dq[0])
                rv.append(dq[1:] / sn * ang if sn > 1e-15 else np.zeros(3))
            dr = _smooth(r - r0); drv = _smooth(np.array(rv))
            ent["dr"] = np.round(dr, 10).tolist(); ent["drot"] = np.round(drv, 10).tolist()
            worst = max(worst, float(np.max(np.linalg.norm(dr, axis=1))))
        tree.append(ent)
    if not any(t["el"] for t in tree):
        return
    S["deform"] = tree
    if S["dim"] == "1d":                               # 1D: its slides are already drawn at a scale (mag_base)
        return
    P0 = np.array([bodies[i]["pos"][0] for i in mov], float)
    span = float(np.max(np.ptp(np.vstack([P0] + [np.asarray(L["pa"], float)[None] for L in S["links"] if L["a"] < 0]), axis=0)))
    if span > 0 and 0 < worst < 0.02 * span:
        mag = _nice_step(0.15 * span / worst)
        S["notes"].append(f"deformation drawn {_nice(mag)} x bigger (at most {_fmt_len(worst)})")
        S["magnified"] = mag; S["mag_auto"] = mag


PART_Q = (("voltage", "V"), ("current", "A"), ("power", "W"), ("soc", ""), ("temperature", "°C"), ("state", ""))
SHOW_FIRST = {"voltage_source": ("voltage", "current"), "battery": ("voltage", "current", "soc"), "motor": ("current",),
              "capacitor": ("voltage",), "inductor": ("current",), "nonlinear_resistor": ("voltage", "current"),
              "current_source": ("current",), "current_load": ("current",)}


def _signals(M, H, idx):
    """Values to read out next to the replay: every circuit part's voltage, current, power (and charge, temperature,
    state where it has them), every control signal and every element's force, at the replay's frames. `show` marks
    the ones on by default (sources and batteries: voltage, current, charge; motors: current; a few control signals)."""
    doc = M.doc; out = []
    parts = ((doc.get("circuit") or {}).get("parts") or {}) if isinstance(doc.get("circuit"), dict) else {}
    nshow = 0

    def add(key, label, unit, group, show):
        nonlocal nshow
        if key not in H:
            return
        v = np.asarray(H[key], float)[idx]
        if not np.isfinite(v).all() or (np.ptp(v) == 0 and group != "circuit"):
            return
        if group == "circuit" and np.ptp(v) == 0 and not show:
            return
        show = bool(show and nshow < 8); nshow += int(show)
        out.append(dict(key=key, label=label, unit=unit, group=group, show=show, v=np.round(v, 9).tolist(),
                        lo=float(v.min()), hi=float(v.max())))
    for pn, d in parts.items():
        t = (d or {}).get("type", "")
        first = SHOW_FIRST.get(t, ())
        for q, u in PART_Q:
            if q == "soc" and t != "battery":
                continue
            if q == "temperature" and not (d or {}).get("thermal"):
                continue
            if q == "state" and t not in ("fuse", "switch", "diode", "nonlinear_resistor", "motor"):
                continue
            add(f"{pn}.{q}", f"{pn} {q}", "%" if q == "soc" else u, "circuit", q in first)
    for i, cn in enumerate((doc.get("control") or {})):
        add(f"control.{cn}", cn, "", "control", i < 3)
    from ..compiler.exports import series_unit
    known = set(M.el_names) | set(getattr(M, "groups", None) or {})
    for en in (doc.get("elements") or {}):       # the model's own elements (a split one answers as its whole)
        if "__" in en or en not in known:
            continue
        try:
            u = series_unit(M, doc, f"{en}.force")
        except Exception:
            u = ""
        add(f"{en}.force", f"{en} force", u, "element", False)
    if not nshow:                                     # nothing picked: the first few, whatever they are
        for sgl in out[:3]:
            sgl["show"] = True
    for sgl in out:
        if sgl["unit"] == "%":
            sgl["v"] = [x * 100 for x in sgl["v"]]; sgl["lo"] *= 100; sgl["hi"] *= 100
    return out


def _label(en):
    """The name a person gave it: a circuit motor's engine row is __circ_<part>; a joint's or beam's rows <name>__x."""
    if en.startswith("__circ_"):
        return en[len("__circ_"):]
    return en.split("__")[0]


def _el_kind(M, en):
    if en.endswith("__probe"):                       # a beam's measuring row (no stiffness): nothing to draw
        return None
    t = M.el_info.get(en, {}).get("type", "")
    if t in ("drive", "dc_motor", "emotor"):
        return "motor"
    base = en.split("__")[0]
    tb = M.el_info.get(base, {}).get("type", t)
    if t in SPRINGY or tb in SPRINGY:
        return "spring"
    if t in DAMPY or tb in DAMPY:
        return "damper"
    if t in LINKY or tb in LINKY or "__pin" in en or "__rot" in en:
        return "joint"
    return None


# ------------------------------------------------------------------------------------------------ shapes, arrows
ARROWY = {"force", "drag", "friction", "drive", "dc_motor", "emotor"}     # pushes on one body: drawn as an arrow


def _shape_prims(M, bn, dim, i=None, have=()):
    """A body's `shape` (box, cylinder, disc, rod, ...) as solid prims in its centre-of-mass frame (the shape's centre
    is the body's centre of mass). `have`: feature kinds already drawn (a disc's or sphere's outline)."""
    d = (M.doc.get("bodies") or {}).get(bn)
    sh = d.get("shape") if isinstance(d, dict) else None
    if not isinstance(sh, dict):
        return []
    try:
        from ..compiler.expand_shape import shape_props
        _m, _I, kind, kw = shape_props(sh, M.ctx, f"bodies.{bn}.shape", "3d" if dim == "3d" else "2d")
    except Exception:
        return []
    g = lambda k, dflt=0.0: float(kw.get(k) if kw.get(k) is not None else dflt)
    ax = [0.0, 0.0, 1.0]
    if dim == "3d" and "axis" in sh and i is not None:
        try:
            from ..compiler.model3d import _vec, _unit, _qR
            aw = np.asarray(_unit(_vec(sh["axis"], M.ctx, "axis"), "axis"), float)
            ax = (_qR(M.arrays["quat0"][i]).T @ aw).tolist()
        except Exception:
            pass
    o = [0.0, 0.0, 0.0]
    if dim != "3d":                                    # flat: in the x-y plane, thin along z
        if kind in ("disc", "ring") and "circle" in have:
            return []
        R = g("Ro") or g("R")
        if kind in ("disc", "ring"):
            return [dict(kind="disc", name="shape", p=o, r=R, solid=True)]
        if kind == "plate":
            a, b = g("a"), g("b")
            return [dict(kind="block", name="shape", p=o, size=[a, b, max(g("t"), 0.05 * min(a, b))], solid=True)]
        if kind == "rod":                          # its inertia does not say which way it lies: along its farthest point
            L = g("L"); ang = 0.0
            pts = [np.asarray(gg[:2], float) for (bb, fn), (kk, gg) in M.feats.items() if bb == bn and fn != "__origin__"]
            if pts:
                far = max(pts, key=lambda v: float(np.hypot(v[0], v[1])))
                if np.hypot(*far) > 1e-12:
                    ang = math.atan2(far[1], far[0])
            return [dict(kind="block", name="shape", p=o, size=[L, L / 25, L / 25], axis=[0.0, 0.0, 1.0], turn=ang, solid=True)]
        return []
    if kind == "sphere":
        return [] if "sphere" in have else [dict(kind="sphere", name="shape", p=o, r=g("R"), solid=True)]
    if kind in ("box", "plate"):
        a, b = g("a"), g("b"); c = g("c") if kind == "box" else (g("t") or 0.05 * min(a, b))
        return [dict(kind="block", name="shape", p=o, size=[a, b, c], axis=ax, solid=True)]
    if kind in ("cylinder", "disc"):
        return [dict(kind="cylinder", name="shape", p=o, axis=ax, r=g("R"), w=g("L") if kind == "cylinder" else g("t"),
                     solid=True)]
    if kind in ("ring", "tube"):
        return [dict(kind="tube", name="shape", p=o, axis=ax, ri=g("Ri"), ro=g("Ro"), w=g("L") if kind == "tube" else g("t"),
                     solid=True)]
    if kind == "rod":
        L = g("L"); A = g("A"); r = math.sqrt(A / math.pi) if A > 0 else L / 40
        return [dict(kind="cylinder", name="shape", p=o, axis=ax, r=r, w=L, solid=True, plain=True)]
    return []


def _force_arrows(M, H, idx, S, order, names, per_body):
    """Loads that push on a body (a force, drag, friction, a motor on a free body) as arrows at its centre of mass:
    the world force vector at each frame, from the element's force and its measure (a coordinate expression)."""
    a = M.arrays; C = np.asarray(a["C"], float); dim = S["dim"]; w = 4 if dim == "3d" else 3
    nb = len(names)
    for e, en in enumerate(M.el_names):
        t = M.el_info.get(en, {}).get("type", "")
        base = _label(en)
        if t not in ARROWY or f"{en}.force" not in H:
            continue
        if dim == "3d" and int(a["mk"][e]) == 2 and float(a["GX"][e, 4]) > 1.5:     # a thrust: at its point, turning
            from ..compiler.model3d import _qR
            i = int(a["ba"][e]); bn = names[i]
            F = np.asarray(H[f"{en}.force"], float)[idx]
            q = np.c_[[np.asarray(H[f"{bn}.{k}"])[idx] for k in ("qw", "qx", "qy", "qz")]].T
            g = np.asarray(a["GX"][e, :3], float)
            v = np.array([f_ * (_qR(qq) @ g) for f_, qq in zip(F, q)])
            j = per_body.get(i, 0); per_body[i] = j + 1
            S.setdefault("arrows", []).append(dict(name=base, b=order.index(bn), v=np.round(v, 6).tolist(), j=j, unit="N",
                                                  p=np.asarray(a["GA"][e, :3], float).tolist()))
            continue
        row = C[e] if e < C.shape[0] else np.zeros(w * nb)
        on = [i for i in range(nb) if any(row[w * i + k] != 0.0 for k in range(3 if dim == "3d" else 2))]
        if not on or any(row[w * i + w - 1] != 0.0 for i in range(nb)):   # a turn (a torque): not drawn here
            continue
        F = np.asarray(H[f"{en}.force"], float)[idx]
        for i in on:
            if "__" in names[i]:
                continue
            c = row[w * i: w * i + (3 if dim == "3d" else 2)]
            v = -np.outer(F, c)
            if v.shape[1] == 2:
                v = np.c_[v, np.zeros(len(F))]
            if not np.isfinite(v).all() or not np.any(v):
                continue
            j = per_body.get(i, 0); per_body[i] = j + 1
            S.setdefault("arrows", []).append(dict(name=base, b=order.index(names[i]), v=np.round(v, 6).tolist(), j=j,
                                                  unit="N"))


# ------------------------------------------------------------------------------------------------ 3d
def _scene3(M, H, idx, S):
    from ..compiler import engine3d as E3
    names = list(M.body_names)
    groups = {}
    for (b, fn), (kind, g) in M.feats.items():
        if fn == "__origin__":
            continue
        groups.setdefault(b, []).append((fn, kind, np.asarray(g, float)))
    order = names + [b for b in groups if b not in names]
    for b in order:
        moving = b in names
        prims = []; pts = []; have = set()
        for fn, kind, g in groups.get(b, []):
            if kind == "sphere":
                if g[3] > 0:
                    prims.append(dict(kind="sphere", name=fn, p=g[:3].tolist(), r=float(g[3]))); have.add("sphere")
                elif moving:
                    prims.append(dict(kind="point", name=fn, p=g[:3].tolist(), r=0.0))
                else:                                # a fixed mounting point: a small anchor block
                    prims.append(dict(kind="anchor", name=fn, p=g[:3].tolist()))
                pts.append(g[:3])
            elif kind == "disc":                  # a cylinder's contact slices: the cylinder itself is drawn
                continue
            elif kind == "cylinder":
                prims.append(dict(kind="cylinder", name=fn, p=g[:3].tolist(), axis=g[3:6].tolist(), r=float(g[6]),
                                  w=float(g[7])))
                pts.append(g[:3])
            else:
                hu, hw = float(min(g[9], 1e6)), float(min(g[10], 1e6))
                prims.append(dict(kind="plane", name=fn, o=g[:3].tolist(), n=g[3:6].tolist(), u=g[6:9].tolist(),
                                  hu=hu, hw=hw, infinite=bool(g[9] > 1e8 or g[10] > 1e8)))
        shp = _shape_prims(M, b, "3d", names.index(b) if moving else None, have) if moving else []
        prims = shp + prims
        body = dict(name=b, ground=not moving, prims=prims)
        if "__" in b:                              # a beam's inner node: the beam is drawn, not its nodes
            body["hidden"] = True; body["prims"] = []
        hull = _hull(pts) if moving and not shp and "__" not in b else None
        if hull:
            body["hull"] = dict(points=[p.tolist() for p in pts], faces=hull)
        elif moving and pts and "__" not in b:     # too few points for a solid: rods from the centre of mass to each
            body["sticks"] = [[[0.0, 0.0, 0.0], p.tolist()] for p in pts if np.linalg.norm(p) > 0]
        if moving:
            i = names.index(b)
            pos = np.c_[[np.asarray(H[f"{b}.{k}"])[idx] for k in ("x", "y", "z")]].T
            q = np.c_[[np.asarray(H[f"{b}.{k}"])[idx] for k in ("qw", "qx", "qy", "qz")]].T
            body["pos"] = np.round(pos, 7).tolist(); body["quat"] = np.round(q, 7).tolist()
            body["mass"] = float(M.arrays["mass"][i])
            if "__" not in b:                      # a fast spinner: turned about its own axis at a readable rate
                W = np.c_[[np.asarray(H[f"{b}.{k}"])[idx] for k in ("wx", "wy", "wz")]].T
                ax = next((p_["axis"] for p_ in prims if p_.get("kind") in ("cylinder", "tube") and p_.get("axis")), None)
                if ax is None and len(W):           # no round part: the way it turns in its own frame, if that is steady
                    from ..compiler.model3d import _qR
                    wb = np.array([_qR(qq).T @ ww for qq, ww in zip(q, W)])
                    m = wb.mean(axis=0)
                    if np.linalg.norm(m) > 0.95 * np.mean(np.linalg.norm(wb, axis=1)) > 0:
                        ax = (m / np.linalg.norm(m)).tolist()
                if ax is not None:
                    from ..compiler.model3d import _qR
                    wa = np.array([float((_qR(qq) @ np.asarray(ax)) @ ww) for qq, ww in zip(q, W)])
                    sg = _spin_split(body, np.asarray(H["t"])[idx], q, wa, ax, b)
                    if sg:
                        S.setdefault("spin_signals", []).append(sg)
        S["bodies"].append(body)
    a = M.arrays
    for e, en in enumerate(M.el_names):
        kind = _el_kind(M, en)
        mk = int(a["mk"][e])
        if kind is None or mk not in (E3.MK_DIST, E3.MK_PIN, E3.MK_BEAM, E3.MK_ALONG):
            continue
        L = dict(name=en.split("__")[0], kind=kind, a=_bi(order, names, a["ba"][e]),
                 pa=a["GA"][e, :3].tolist(), b=_bi(order, names, a["bb"][e]), pb=a["GB"][e, :3].tolist())
        if mk == E3.MK_BEAM:                       # a tube of about the section's size (from its stiffnesses)
            bi = (M.el_info.get(en) or {}).get("beam") or {}
            L["kind"] = "beam"
            if bi.get("EA"):
                L["r"] = float(math.sqrt(2.0 * min(bi["EIy"], bi["EIz"]) / bi["EA"]))
        S["links"].append(L)
    _force_arrows(M, H, idx, S, order, names, {})


def _bi(order, names, k):
    """Engine body index (-1 = ground) to the scene's body list index (-1 = the world itself)."""
    k = int(k)
    return order.index(names[k]) if k >= 0 else -1


# ------------------------------------------------------------------------------------------------ 2d
def _scene2(M, H, idx, S):
    from ..compiler import engine2d as E2
    names = list(M.body_names)
    groups = {}
    for (b, fn), (kind, g) in M.feats.items():
        if fn == "__origin__":
            continue
        groups.setdefault(b, []).append((fn, kind, np.asarray(g, float)))
    order = names + [b for b in groups if b not in names]
    for b in order:
        moving = b in names
        prims = []; pts = []; have = set()
        for fn, kind, g in groups.get(b, []):
            if kind == "circle":
                if g[2] > 0:
                    prims.append(dict(kind="disc", name=fn, p=[g[0], g[1], 0.0], r=float(g[2]))); have.add("circle")
                else:
                    prims.append(dict(kind="point" if moving else "anchor", name=fn, p=[g[0], g[1], 0.0], r=0.0))
                pts.append([g[0], g[1]])
            else:
                prims.append(dict(kind="line", name=fn, a=[g[0], g[1], 0.0], b=[g[2], g[3], 0.0]))
                pts += [[g[0], g[1]], [g[2], g[3]]]
        shp = _shape_prims(M, b, "2d", None, have) if moving else []
        prims = shp + prims
        body = dict(name=b, ground=not moving, prims=prims)
        if "__" in b:
            body["hidden"] = True; body["prims"] = []; pts = []
        if moving:
            if len(pts) >= 3:
                body["outline"] = _hull2(pts)
            if not body.get("outline"):
                body["sticks"] = [[[0.0, 0.0, 0.0], [p[0], p[1], 0.0]] for p in pts if math.hypot(p[0], p[1]) > 0]
            x = np.asarray(H[f"{b}.x"])[idx]; y = np.asarray(H[f"{b}.y"])[idx]; th = np.asarray(H[f"{b}.angle"])[idx]
            body["pos"] = np.round(np.c_[x, y, np.zeros_like(x)], 7).tolist()
            body["quat"] = [_q_axis((0, 0, 1), float(v)) for v in th]
            if f"{b}.omega" in H and "__" not in b:
                sg = _spin_split(body, np.asarray(H["t"])[idx], body["quat"], np.asarray(H[f"{b}.omega"])[idx], [0, 0, 1], b)
                if sg:
                    S.setdefault("spin_signals", []).append(sg)
            body["mass"] = float(M.arrays["mass"][names.index(b)])
        S["bodies"].append(body)
    a = M.arrays
    for e, en in enumerate(M.el_names):
        kind = _el_kind(M, en); mk = int(a["mk"][e])
        if kind is None or mk not in (E2.MK_DIST, E2.MK_PIN, E2.MK_BEAM, E2.MK_ALONG):
            continue
        L = dict(name=en.split("__")[0], kind=kind, a=_bi(order, names, a["ba"][e]),
                 pa=[float(a["GA"][e, 0]), float(a["GA"][e, 1]), 0.0], b=_bi(order, names, a["bb"][e]),
                 pb=[float(a["GB"][e, 0]), float(a["GB"][e, 1]), 0.0])
        if mk == E2.MK_BEAM:
            L["kind"] = "beam"
        S["links"].append(L)
    _force_arrows(M, H, idx, S, order, names, {})
    try:                                             # a planet (gravity: {planet: ...}): drawn round its centre, the origin
        from ..compiler.model2d import planet_of
        pl = planet_of(M.doc, M.ctx)
    except Exception:
        pl = None
    if pl is not None and pl[1] > 0:
        S["bodies"].append(dict(name=str((M.doc.get("gravity") or {}).get("planet", "planet")), ground=True,
                                prims=[dict(kind="planet", name="planet", p=[0.0, 0.0, 0.0], r=float(pl[1]),
                                            air=float(100e3 if pl[3] else 0.0))]))


def _hull2(pts):
    try:
        from scipy.spatial import ConvexHull
        P = np.asarray(pts, float)
        if np.linalg.matrix_rank(P - P.mean(axis=0), tol=1e-9) < 2:
            return None
        h = ConvexHull(P)
        return [[float(P[i, 0]), float(P[i, 1]), 0.0] for i in h.vertices]
    except Exception:
        return None


# ------------------------------------------------------------------------------------------------ 1d
def _nice_step(x):
    """The largest 1, 2 or 5 x 10^n at or below x."""
    if x <= 0:
        return 1.0
    e = 10 ** math.floor(math.log10(x)); m = x / e
    return (5 if m >= 5 else 2 if m >= 2 else 1) * e


def _scene1(M, H, idx, S):
    """Each body in its own lane. A sliding body held by springs (an oscillator) is drawn at a scale that makes its
    travel about a lane wide; a sliding body nothing holds (a car, a cart pushed along) drives along a road with
    distance marks, the camera riding with it. Springs and dampers to the fixed world are drawn to a wall; loads that
    only push (a force, drag, friction, a motor) are arrows on the body (a curved arrow on a turning one), never
    lines to the wall. A turning body's links attach at fixed points beside it, so they do not whip round with it."""
    names = list(M.body_names); nb = len(names)
    dof = list(getattr(M, "dof", ["translation"] * nb))
    q = {b: np.asarray(H[f"{b}.position"])[idx] for b in names}
    trans = [b for b, d in zip(names, dof) if d != "rotation"]
    C = np.asarray(M.arrays["C"], float)
    info = lambda en: M.el_info.get(en, {}).get("type", "")
    held = set()                                     # bodies a spring holds
    for e, en in enumerate(M.el_names):
        if info(en) in SPRINGY or info(en.split("__")[0]) in SPRINGY:
            row = C[e] if e < C.shape[0] else np.zeros(nb)
            held |= {names[k] for k in range(min(nb, len(row))) if row[k] != 0.0}
    travel = max([float(np.ptp(q[b])) for b in trans] + [0.0])
    span = max([float(np.max(np.abs(q[b]))) for b in trans] + [0.0])
    road = bool(trans) and travel > 0 and not (set(trans) & held)
    if road:                                         # the whole trip about 30 lane widths, marks in real distance
        scale = 30.0 / travel
        S["notes"].append(f"driving along: {_fmt_len(travel)} in all, marks every {_fmt_len(_nice_step(0.5 / scale) * 5)}")
    else:
        scale = 1.0 / max(span, 1e-12) if trans and span > 0 else 1.0
        if trans and span > 0:
            S["notes"].append(f"sliding bodies drawn at {_nice(scale)} x their real travel ({_fmt_len(travel)} peak to peak)")
    S["road"] = road
    S["mag_base"] = S["mag_auto"] = float(scale)     # positions below are already drawn at this scale
    lane = {b: (nb - 1 - i) * 1.4 for i, b in enumerate(names)}
    for i, b in enumerate(names):
        rot = dof[i] == "rotation"
        y = lane[b]
        if rot:
            prims = [dict(kind="wheel", name=b, p=[0, 0, 0], r=0.45)]
            pos = [[0.0, y, 0.0]] * len(idx)
            quat = [_q_axis((0, 0, 1), float(v)) for v in q[b]]
        else:
            prims = [dict(kind="block", name=b, p=[0, 0, 0], size=[0.5, 0.5, 0.5])]
            pos = [[float(v) * scale, y, 0.0] for v in q[b]]
            quat = [[1.0, 0.0, 0.0, 0.0]] * len(idx)
        S["bodies"].append(dict(name=b, ground=False, prims=prims, pos=pos, quat=quat, rotation=rot))
        if rot and f"{b}.velocity" in H:
            sg = _spin_split(S["bodies"][-1], np.asarray(H["t"])[idx], quat, np.asarray(H[f"{b}.velocity"])[idx], [0, 0, 1], b)
            if sg:
                S.setdefault("spin_signals", []).append(sg)
    if nb == 0:                                      # a circuit (or control) only model: nothing to draw
        return
    gprims = []
    if road:
        minor = _nice_step(0.5 / scale)
        for b in trans:
            x0 = float(np.min(q[b])) * scale - 3.0; x1 = float(np.max(q[b])) * scale + 3.0
            gprims.append(dict(kind="road", name="road", y=lane[b] - 0.25, x0=x0, x1=x1, step=minor * scale, major=5,
                               unit_m=1.0 / scale, origin=0.0))
    S["bodies"].append(dict(name="ground", ground=True, prims=gprims))
    gi = len(S["bodies"]) - 1
    per = {}; arrows = {}                            # links on the same body or pair sit side by side
    wall_used = False
    for e, en in enumerate(M.el_names):
        if en.endswith("__probe"):
            continue
        t = info(en); base = _label(en)
        kind = _el_kind(M, en)
        row = C[e] if e < C.shape[0] else np.zeros(nb)
        on = [k for k in range(min(nb, len(row))) if row[k] != 0.0]
        if not on:
            continue
        pushes = kind in (None, "motor") or t in ARROWY
        if len(on) == 1 and road and kind == "damper":
            pushes = True                            # rolling resistance on a car is a drag, not a link to a wall
        if pushes:                                   # an arrow (a curved one on a turning body) at each body it acts on
            if f"{en}.force" not in H:
                continue
            F = np.asarray(H[f"{en}.force"], float)[idx]
            for k in on:
                j = arrows.get(k, 0); arrows[k] = j + 1
                v = (-row[k] * F)
                S.setdefault("arrows", []).append(dict(name=base, b=k, j=j, spin=dof[k] == "rotation",
                                                      v=np.round(np.c_[v * 1.0, 0 * v, 0 * v], 6).tolist(),
                                                      unit="N m" if dof[k] == "rotation" else "N"))
            continue
        key = tuple(on[:2]); j = per.get(key, 0); per[key] = j + 1
        off = [0.0, 0.14, -0.14, 0.28, -0.28][j % 5]
        end = lambda k, side: (gi, [side * 0.47, lane[names[k]] + off, 0.0]) if dof[k] == "rotation" else \
            (k, [side * 0.25, off, 0.0])
        if len(on) == 1:                             # to the wall (the ground body has no pose: world coordinates)
            k = on[0]; wall_used = True
            bk, pk = end(k, -1)
            S["links"].append(dict(name=base, kind=kind or "spring", a=gi, pa=[-1.94, lane[names[k]] + off, 0.0], b=bk, pb=pk))
        else:
            k0, k1 = on[0], on[1]
            if dof[k0] == "rotation" or dof[k1] == "rotation":     # side by side lanes: from one edge to the other
                ya, yb = lane[names[k0]], lane[names[k1]]
                up = 1 if yb > ya else -1
                pa = (gi, [off, ya + up * 0.47, 0.0]) if dof[k0] == "rotation" else (k0, [off, up * 0.25, 0.0])
                pb = (gi, [off, yb - up * 0.47, 0.0]) if dof[k1] == "rotation" else (k1, [off, -up * 0.25, 0.0])
                S["links"].append(dict(name=base, kind=kind or "spring", a=pa[0], pa=pa[1], b=pb[0], pb=pb[1]))
            else:
                S["links"].append(dict(name=base, kind=kind or "spring", a=k0, pa=[off, -0.25, 0.0], b=k1, pb=[off, 0.25, 0.0]))
    if wall_used:
        S["bodies"][gi]["prims"].append(dict(kind="wall", name="wall", p=[-2.0, (nb - 1) * 0.7, 0], size=[0.12, nb * 1.4, 0.8]))


def _nice(x):
    if x <= 0:
        return "1"
    e = math.floor(math.log10(x)); m = x / 10 ** e
    return f"{m:.2g}e{e}" if abs(e) > 3 else f"{x:.3g}"


def _fmt_len(v):
    return f"{v * 1e3:.3g} mm" if v < 1 else f"{v:.3g} m"


def _extent(S):
    """Bounding box of everything that moves (and finite ground features), for the camera and an infinite floor."""
    lo = np.full(3, np.inf); hi = np.full(3, -np.inf)
    for b in S["bodies"]:
        if "pos" in b and b["pos"]:
            P = np.asarray(b["pos"], float)
            lo = np.minimum(lo, P.min(axis=0)); hi = np.maximum(hi, P.max(axis=0))
        for p in b["prims"]:
            if b["ground"] and ((p.get("kind") == "plane" and p.get("infinite")) or p.get("kind") in ("planet", "road")):
                continue
            c = np.asarray(p.get("p", p.get("o", p.get("a", [0, 0, 0]))), float)
            if b["ground"]:
                lo = np.minimum(lo, c); hi = np.maximum(hi, c)
    if not np.isfinite(lo).all():
        lo = np.zeros(3); hi = np.zeros(3)
    size = float(max(np.max(hi - lo), 1e-3))
    # a feature size scale: the bodies' own geometry
    feat = []
    for b in S["bodies"]:
        if b["ground"]:
            continue
        for p in b["prims"]:
            c = np.asarray(p.get("p", p.get("o", p.get("a", [0, 0, 0]))), float)
            feat.append(float(np.linalg.norm(c)) + float(p.get("r", 0.0)) + 0.5 * float(max(p.get("size", [0.0]))))
    S["extent"] = dict(lo=lo.tolist(), hi=hi.tolist(), size=size, body=max(feat) if feat else size * 0.1)
