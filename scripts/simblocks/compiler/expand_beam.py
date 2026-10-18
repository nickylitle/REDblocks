"""Beam elements: section properties, and splitting a beam into segments with hidden node bodies.

A beam joins a point on body A to a point on body B and is rigidly attached at both ends (the ends turn with the
bodies). Stiffness from E and the section (A, I; 3D also Iy and J with G), or given directly (EA, EI / EIy, EIz, GJ).
Mass from the material density (rho A L) or `mass:`; with `segments: n` the beam becomes n co-rotational elements
with n - 1 node bodies carrying its mass, so it bends, vibrates and whips along its length. Its end mass shares are
added to A and B at the attachment points (end_mass: false to leave them out).

hinge: a moment / plastic-rotation table (k, acc_tab, F_tab, e.g. hinge_table or plate_root_hinge), or 'auto' for
elastic-perfectly-plastic at Mp = sy Z of the section and material. Plastic hinges can form at every segment end.
       In 3D each axis has its own strength: the table (or sy Z) is about the section's x axis, and the other axis
       yields at Zy / Z times that (sections give Zy; given_section takes it), on the yield surface
       (My / Mpy)^2 + (Mz / Mpz)^2 = 1. Without Zy both axes use the table.
damping: fraction of critical in the beam's first bending mode (estimated as a cantilever carrying its own mass
         and the lighter end body). Bending is damped through a low-pass at three times that frequency, so the first
         mode gets what was asked while the fast segment modes (which an explicit step cannot damp stably) are left
         alone; stretch and twist are damped at their own segment frequencies. Up to about 0.3.
orient (3D): the direction of the section's h axis at the start (default world z, or x for a vertical beam).
Bending across orient uses I (Ix of the section), bending across the other axis uses Iy; torsion uses G J."""
import math

import numpy as np

from .loader import SpecError, line_of
from .expand import Group, _num

BEAM_KEYS = ("E", "A", "I", "Iy", "Iz", "J", "G", "EA", "EI", "EIy", "EIz", "GJ", "section", "material", "density",
             "hinge", "damping", "orient", "rotary_inertia", "mass", "segments", "end_mass")


def beam_props(e, ctx, where, dim, L, f=""):
    """Resolved stiffness, mass per length and hinge table of a beam (all SI)."""
    from .model import resolve, num
    from ..materials import get
    from ..materials.curves import resolve_material
    g = lambda k: num(resolve(e[k], ctx, f"{where}.{k}"), f"{where}.{k}") if k in e else None
    mat = None
    if "material" in e:
        mv = resolve(e["material"], ctx, where + ".material")
        try:
            try:
                mat = get(str(mv))
            except Exception:
                mat = get(resolve_material(str(mv)))
        except Exception as exc:
            raise SpecError(f"unknown material '{mv}' ({exc})", where, line_of(e, "material"), f)
    sec = resolve(e["section"], ctx, where + ".section") if "section" in e else None
    if sec is not None and not isinstance(sec, dict):
        raise SpecError("section is a section block result, e.g. {block: rectangle, b: 20 mm, h: 4 mm}", where,
                        line_of(e, "section"), f)
    def fin(v):                               # a NaN from given_section (not given) counts as missing
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) and v > 0 else None
    if sec is not None:
        sec = {k: (fin(v) if k in ("A", "Ix", "Iy", "J", "Z", "Zy") else v) for k, v in sec.items()}
    E = g("E") or (mat.E if mat else None)
    G = g("G") or (mat.G if mat and getattr(mat, "G", None) else (E / 2.6 if E else None))
    A_ = g("A") or (sec.get("A") if sec else None)
    Iz = g("I") or g("Iz") or (sec.get("Ix") if sec else None)
    Iy = g("Iy") or (sec.get("Iy") if sec else Iz)
    if dim == "3d" and not Iy and Iz and not (g("EIy")):
        raise SpecError("a 3D beam needs Iy (bending the other way): the section has none. Give Iy (or EIy), "
                        "or use a section block that has it (rectangle, circle, tube, rect_tube, i_beam)", where,
                        line_of(e, "section") or line_of(e, "type"), f)
    J = g("J") or (sec.get("J") if sec else None)
    EA = g("EA") or (E * A_ if E and A_ else None)
    EIz = g("EI") or g("EIz") or (E * Iz if E and Iz else None)
    EIy = g("EIy") or (E * Iy if E and Iy else EIz)
    GJ = g("GJ") or (G * J if G and J else None)
    miss = [n for n, v in (("EA (E and A)", EA), ("EI (E and I)", EIz)) if not fin(v)]
    if dim == "3d" and not fin(GJ):
        miss.append("GJ (G and J)")
    if miss:
        raise SpecError("a beam needs " + ", ".join(miss) + ": give a material and a section, or the numbers",
                        where, line_of(e, "type"), f)
    rho = g("density") or (mat.rho if mat else None)
    m = g("mass")
    if m is None:
        m = rho * A_ * L if (rho and A_) else 0.0
    hinge = None
    if "hinge" in e:
        hv = e["hinge"]
        if isinstance(hv, str) and hv.strip() == "auto":
            if not (sec and mat and sec.get("Z")):
                raise SpecError("hinge: auto needs a material and a section (it uses Mp = sy Z)", where,
                                line_of(e, "hinge"), f)
            Mp = mat.sy * float(sec["Z"])
            hinge = dict(acc_tab=np.array([0.0, 1.0]), F_tab=np.array([Mp, Mp]))
            if sec.get("Zy"):
                hinge["ry"] = float(sec["Zy"]) / float(sec["Z"])     # yield the other way: Mp_y = sy Zy
        else:
            tb = resolve(hv, ctx, where + ".hinge")
            if isinstance(tb, dict) and {"theta", "M", "k0"} <= set(tb) and "acc_tab" not in tb:
                # hinge_table gives moment against total rotation: the plastic part is theta - M / k0
                th = np.asarray(tb["theta"], float); Mm = np.asarray(tb["M"], float); k0 = float(tb["k0"])
                acc = th - Mm / k0; keep = acc >= -1e-12 * max(1.0, float(np.max(np.abs(th))))
                acc = np.maximum(acc[keep], 0.0); Mm = Mm[keep]
                if len(acc) == 0:
                    raise SpecError("this hinge_table never yields (no plastic rotation)", where, line_of(e, "hinge"), f)
                tb = {"acc_tab": np.concatenate([[0.0], acc]) if acc[0] > 0 else acc,
                      "F_tab": np.concatenate([[Mm[0]], Mm]) if acc[0] > 0 else Mm}
            if not isinstance(tb, dict) or not {"acc_tab", "F_tab"} <= set(tb):
                raise SpecError("hinge is 'auto' or a moment table {k, acc_tab, F_tab} (e.g. from hinge_table or "
                                "plate_root_hinge)", where, line_of(e, "hinge"), f)
            hinge = dict(acc_tab=np.asarray(tb["acc_tab"], float), F_tab=np.asarray(tb["F_tab"], float))
            if sec and sec.get("Zy") and sec.get("Z"):
                hinge["ry"] = float(sec["Zy"]) / float(sec["Z"])
    zeta = g("damping") or 0.0
    return dict(EA=EA, EIz=EIz, EIy=EIy, GJ=GJ, mass=m, hinge=hinge, zeta=zeta, E=E, A=A_, Iz=Iz, Iy=Iy, rho=rho)


def _world(doc, ctx, ref, dim, where):
    """World position at t = 0 of body.feature (a point, circle or sphere centre)."""
    from .model import resolve, num
    b, _, fn = str(ref).partition(".")
    if b not in doc["bodies"]:
        raise SpecError(f"no body named '{b}'", where)
    d = doc["bodies"][b]
    n = 3 if dim == "3d" else 2
    vec = (lambda v, w: _vec3(v, ctx, w)) if n == 3 else (lambda v, w: _vec2(v, ctx, w))
    if fn:
        pts = d.get("points") or {}; circ = d.get("circles" if n == 2 else "spheres") or {}
        if fn in pts:
            p = np.array(vec(pts[fn], where))
        elif fn in circ:
            p = np.array(vec(circ[fn].get("at", [0] * n), where))
        else:
            raise SpecError(f"body '{b}' has no point called '{fn}' (a beam end is a point)", where)
    else:
        p = np.zeros(n)
    if d.get("ground"):
        return p, b, p, None, np.zeros(n), (np.zeros(3) if n == 3 else 0.0)
    at = np.array(vec(d.get("at", [0] * n), where))
    if n == 2:
        a0 = num(resolve(d.get("angle0", 0.0), ctx, where), where)
        R = np.array([[math.cos(a0), -math.sin(a0)], [math.sin(a0), math.cos(a0)]])
    else:
        from .model3d import rot, _unit
        R = np.eye(3)
        if "orient0" in d:
            o = d["orient0"]; R = rot(_vec3(o["axis"], ctx, where), num(resolve(o["angle"], ctx, where), where))
        dl = d.get("dof", "free")
        if (isinstance(dl, list) and "spin" in dl) or dl in ("spin", "rotation"):
            ax = _unit(_vec3(d.get("axis", "z"), ctx, where), where)
            R = rot(ax, num(resolve(d.get("angle0", 0.0), ctx, where), where)) @ R
    # velocity of the point: v0 of the drawing origin + omega x (R p); omega0 is a spin rate (2d / spin) or a vector
    v0 = np.array(vec(d.get("v0", [0] * n), where)) if "v0" in d else np.zeros(n)
    om = d.get("omega0", 0.0)
    if n == 2:
        w = num(resolve(om, ctx, where), where) if om not in (None, 0) else 0.0
        rp = R @ p; vp = v0 + w * np.array([-rp[1], rp[0]])
    else:
        if isinstance(om, (list, tuple)) or (isinstance(om, str) and om.strip().startswith("[")):
            w = np.array(_vec3(om, ctx, where))
        else:
            wv = num(resolve(om, ctx, where), where) if om not in (None, 0) else 0.0
            from .model3d import _unit
            ax = _unit(_vec3(d.get("axis", "z"), ctx, where), where) if wv else np.zeros(3)
            w = ax * wv
        vp = v0 + np.cross(w, R @ p)
    return at + R @ p, b, p, R, vp, w


def _vec2(v, ctx, w):
    from .model2d import _vec
    return _vec(v, ctx, w)


def _vec3(v, ctx, w):
    from .model3d import _vec
    return _vec(v, ctx, w)


def expand_beam(en, e, doc, ctx, bodies, new, groups):
    from .model import resolve, num
    f = doc.get("__file__", ""); where = f"elements.{en}"; dim = doc["dim"]
    refs = e.get("measure")
    if not isinstance(refs, (list, tuple)) or len(refs) != 2:
        raise SpecError("a beam's measure is [A.point, B.point]: the two points it joins", where, line_of(e, "measure"), f)
    pA, A, locA, RA, vA, wA = _world(doc, ctx, refs[0], dim, where + ".measure")
    pB, B, locB, RB, vB, wB = _world(doc, ctx, refs[1], dim, where + ".measure")
    L = float(np.linalg.norm(pB - pA))
    if L <= 0:
        raise SpecError("the beam's two ends are at the same place", where, line_of(e, "measure"), f)
    props = beam_props(e, ctx, where, dim, L, f)
    n = int(_num(e.get("segments", 1), ctx, where + ".segments"))
    if n < 1:
        raise SpecError("segments is a whole number, 1 or more", where, line_of(e, "segments"), f)
    m = props["mass"]; mseg = m / n; Ls = L / n
    base = {k: e[k] for k in ("damping", "orient") if k in e}
    base["__hinge__"] = props["hinge"]                     # resolved table (numbers), shared by every segment
    for k in ("EA", "EIz", "EIy", "GJ"):
        base["__" + k] = props[k]
    # the whole beam, for damping set at its first bending mode
    base["__Ltot"] = L; base["__mtot"] = m; base["__ends"] = [str(refs[0]).partition(".")[0], str(refs[1]).partition(".")[0]]
    nodes = []
    for j in range(1, n):
        pos = pA + (pB - pA) * j / n
        nm = f"{en}__n{j}"
        r2y = props["EIz"] / props["EA"]; r2z = props["EIy"] / props["EA"]      # section radii of gyration^2
        Irot = mseg * Ls * Ls / 12.0 + mseg * max(r2y, r2z)
        fr = j / n
        vel = [float(x) for x in vA + (vB - vA) * fr]          # nodes start moving with the ends (rigid-like)
        if dim == "3d":
            om = [float(x) for x in np.asarray(wA) + (np.asarray(wB) - np.asarray(wA)) * fr]
            e1 = (pB - pA) / L
            Ipol = mseg * (r2y + r2z)                                     # polar: m (Iy + Iz) / A
            Iten = max(Irot, 1e-30) * (np.eye(3) - np.outer(e1, e1)) + Ipol * np.outer(e1, e1)
            bodies[nm] = {"dof": "free", "mass": mseg, "inertia": Iten.tolist(), "at": [float(x) for x in pos],
                          "v0": vel, "omega0": om, "points": {"p": [0.0, 0.0, 0.0]}, "__hidden__": True}
        else:
            om = float(wA + (wB - wA) * fr)
            bodies[nm] = {"dof": "planar", "mass": mseg, "inertia": max(Irot, 1e-30), "at": [float(x) for x in pos],
                          "v0": vel, "omega0": om, "points": {"p": [0.0, 0.0]}, "__hidden__": True}
        nodes.append(nm + ".p")
    chain = [refs[0]] + nodes + [refs[1]]
    segs = []
    for j in range(n):
        sn = en if n == 1 else f"{en}__s{j + 1}"
        new[sn] = dict(base, type="beam", measure=[chain[j], chain[j + 1]], __mseg__=mseg if mseg > 0 else None,
                       **({"__hidden__": True} if n > 1 else {}))
        segs.append(sn)
    # the beam's end shares of mass go onto A and B at the attachment points; the weight budget reads this record
    rec = doc.setdefault("__beam_mass__", {}).setdefault(en, {"total": float(m), "ends": {}})
    if m > 0 and e.get("end_mass", True):
        mend = mseg / 2.0
        for bn, loc in ((A, locA), (B, locB)):
            d = bodies[bn]
            if d.get("ground"):
                continue
            d = dict(d); where_b = f"bodies.{bn}"
            com = np.array((_vec3 if dim == "3d" else _vec2)(d.get("com", [0] * (3 if dim == "3d" else 2)), ctx, where_b))
            r = loc - com
            if "mass" in d:
                d["mass"] = num(resolve(d["mass"], ctx, where_b + ".mass"), where_b) + mend
                rec["ends"][bn] = rec["ends"].get(bn, 0.0) + mend
            if "inertia" in d:
                if dim == "3d":
                    from .model3d import _inertia
                    I = _inertia(d["inertia"], ctx, where_b + ".inertia") + mend * (float(r @ r) * np.eye(3) - np.outer(r, r))
                    d["inertia"] = I.tolist()
                else:
                    d["inertia"] = num(resolve(d["inertia"], ctx, where_b + ".inertia"), where_b) + mend * float(r @ r)
            bodies[bn] = d
    if n > 1:
        probe = f"{en}__probe"
        new[probe] = dict(type="spring", k=0.0, measure={"distance": list(refs)}, __hidden__=True)
        groups[en] = Group(en, segs, probe, kind="beam")
