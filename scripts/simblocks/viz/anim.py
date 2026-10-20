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


def scene(system, overrides=None, frames=900, H=None, out=None):
    """The JSON-able scene: bodies, links, frames, extent. Uses the history H of a run already made (the editor's
    last Run) when given, otherwise runs the model once."""
    M = system.compile(overrides)
    if H is None:
        from ..compiler.model import run_model
        from ..compiler.system import _with_blocks
        r = run_model(M, history=True, n_log=frames)
        out, H = _with_blocks(M, r, True)
    out = out or {}
    dim = getattr(M, "dim", "1d")
    t, idx = _thin(H, frames)
    S = dict(dim=dim, name=M.doc.get("name", "model"), t=t.tolist(), bodies=[], links=[], notes=[],
             timed_out=bool(out.get("_timed_out")))
    if dim == "3d":
        _scene3(M, H, idx, S)
    elif dim == "2d":
        _scene2(M, H, idx, S)
    else:
        _scene1(M, H, idx, S)
    _extent(S)
    S["empty"] = not any(b.get("pos") for b in S["bodies"])
    S["signals"] = _signals(M, H, idx)
    return S


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
    for en in M.el_names:
        if "__" in en or en not in (doc.get("elements") or {}):
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


def _el_kind(M, en):
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
        prims = []; pts = []
        for fn, kind, g in groups.get(b, []):
            if kind == "sphere":
                prims.append(dict(kind="sphere" if g[3] > 0 else "point", name=fn, p=g[:3].tolist(), r=float(g[3])))
                pts.append(g[:3])
            else:
                hu, hw = float(min(g[9], 1e6)), float(min(g[10], 1e6))
                prims.append(dict(kind="plane", name=fn, o=g[:3].tolist(), n=g[3:6].tolist(), u=g[6:9].tolist(),
                                  hu=hu, hw=hw, infinite=bool(g[9] > 1e8 or g[10] > 1e8)))
        body = dict(name=b, ground=not moving, prims=prims)
        hull = _hull(pts) if moving else None
        if hull:
            body["hull"] = dict(points=[p.tolist() for p in pts], faces=hull)
        elif moving and pts:                       # too few points for a solid: rods from the centre of mass to each
            body["sticks"] = [[[0.0, 0.0, 0.0], p.tolist()] for p in pts if np.linalg.norm(p) > 0]
        if moving:
            i = names.index(b)
            pos = np.c_[[np.asarray(H[f"{b}.{k}"])[idx] for k in ("x", "y", "z")]].T
            q = np.c_[[np.asarray(H[f"{b}.{k}"])[idx] for k in ("qw", "qx", "qy", "qz")]].T
            body["pos"] = np.round(pos, 7).tolist(); body["quat"] = np.round(q, 7).tolist()
            body["mass"] = float(M.arrays["mass"][i])
        S["bodies"].append(body)
    a = M.arrays
    for e, en in enumerate(M.el_names):
        kind = _el_kind(M, en)
        mk = int(a["mk"][e])
        if kind is None or mk not in (E3.MK_DIST, E3.MK_PIN, E3.MK_BEAM, E3.MK_ALONG):
            continue
        S["links"].append(dict(name=en.split("__")[0], kind=kind, a=_bi(order, names, a["ba"][e]),
                               pa=a["GA"][e, :3].tolist(), b=_bi(order, names, a["bb"][e]), pb=a["GB"][e, :3].tolist()))


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
        prims = []; pts = []
        for fn, kind, g in groups.get(b, []):
            if kind == "circle":
                prims.append(dict(kind="disc" if g[2] > 0 else "point", name=fn, p=[g[0], g[1], 0.0], r=float(g[2])))
                pts.append([g[0], g[1]])
            else:
                prims.append(dict(kind="line", name=fn, a=[g[0], g[1], 0.0], b=[g[2], g[3], 0.0]))
                pts += [[g[0], g[1]], [g[2], g[3]]]
        body = dict(name=b, ground=not moving, prims=prims)
        if moving:
            if len(pts) >= 3:
                body["outline"] = _hull2(pts)
            if not body.get("outline"):
                body["sticks"] = [[[0.0, 0.0, 0.0], [p[0], p[1], 0.0]] for p in pts if math.hypot(p[0], p[1]) > 0]
            x = np.asarray(H[f"{b}.x"])[idx]; y = np.asarray(H[f"{b}.y"])[idx]; th = np.asarray(H[f"{b}.angle"])[idx]
            body["pos"] = np.round(np.c_[x, y, np.zeros_like(x)], 7).tolist()
            body["quat"] = [_q_axis((0, 0, 1), float(v)) for v in th]
            body["mass"] = float(M.arrays["mass"][names.index(b)])
        S["bodies"].append(body)
    a = M.arrays
    for e, en in enumerate(M.el_names):
        kind = _el_kind(M, en); mk = int(a["mk"][e])
        if kind is None or mk not in (E2.MK_DIST, E2.MK_PIN, E2.MK_BEAM, E2.MK_ALONG):
            continue
        S["links"].append(dict(name=en.split("__")[0], kind=kind, a=_bi(order, names, a["ba"][e]),
                               pa=[float(a["GA"][e, 0]), float(a["GA"][e, 1]), 0.0], b=_bi(order, names, a["bb"][e]),
                               pb=[float(a["GB"][e, 0]), float(a["GB"][e, 1]), 0.0]))


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
def _scene1(M, H, idx, S):
    names = list(M.body_names); nb = len(names)
    dof = list(getattr(M, "dof", ["translation"] * nb))
    q = {b: np.asarray(H[f"{b}.position"])[idx] for b in names}
    trans = [b for b, d in zip(names, dof) if d != "rotation"]
    travel = max([float(np.ptp(q[b])) for b in trans] + [0.0])
    span = max([float(np.max(np.abs(q[b]))) for b in trans] + [0.0])
    # a sliding body moves by scale x its coordinate: about one lane width of travel on screen
    scale = 1.0 / max(span, 1e-12) if trans and span > 0 else 1.0
    if trans and span > 0:
        S["notes"].append(f"sliding bodies drawn at {_nice(scale)} x their real travel ({_fmt_len(travel)} peak to peak)")
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
    if nb == 0:                                      # a circuit (or control) only model: nothing to draw
        return
    S["bodies"].append(dict(name="ground", ground=True, prims=[dict(kind="wall", name="wall", p=[-2.0, (nb - 1) * 0.7, 0],
                                                                      size=[0.12, nb * 1.4, 0.8])]))
    gi = len(S["bodies"]) - 1
    C = np.asarray(M.arrays["C"], float)
    per = {}                                         # links on the same body or pair sit side by side
    for e, en in enumerate(M.el_names):
        kind = _el_kind(M, en) or ("damper" if M.el_info[en]["type"] == "friction" else "spring")
        row = C[e] if e < C.shape[0] else np.zeros(nb)
        on = [k for k in range(min(nb, len(row))) if row[k] != 0.0]
        if not on:
            continue
        key = tuple(on[:2]); j = per.get(key, 0); per[key] = j + 1
        off = [0.0, 0.14, -0.14, 0.28, -0.28][j % 5]
        if len(on) == 1:                             # to the wall (the ground body has no pose: world coordinates)
            k = on[0]
            S["links"].append(dict(name=en, kind=kind, a=gi, pa=[-1.94, lane[names[k]] + off, 0.0],
                                   b=k, pb=[-0.25 if dof[k] != "rotation" else -0.45, off, 0.0]))
        else:
            k0, k1 = on[0], on[1]
            S["links"].append(dict(name=en, kind=kind, a=k0, pa=[off, -0.25, 0.0], b=k1, pb=[off, 0.25, 0.0]))


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
            if b["ground"] and p.get("kind") == "plane" and p.get("infinite"):
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
