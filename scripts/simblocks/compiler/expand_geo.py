"""2D / 3D expansion of composite elements into engine pieces (see expand.py for the naming and output rules).

weld   [A.p, B.q]                 pin (k, c) + rotation lock (k_rot, c_rot): the two bodies move as one, elastically
hinge  [A.p, B.q], axis           2D: a pin (plus an optional torsion spring k_twist about it)
                                  3D: pin + lock of the two rotations across the axis (k_rot) + optional k_twist
slider [A.p, B.q], axis           B's point stays on the line through A's point along axis (fixed in A): stiffness k
                                  across it, free along it; k_rot (optional) stops B turning relative to A.
                                  measure = how far B has slid along the axis
beam   [A.p, B.q], segments: n    built in expand_beam (deformable beam with hidden node bodies)"""
import copy
import math

import numpy as np

from .loader import SpecError, line_of
from .expand import Group, _num

COMPOSITE = ("weld", "hinge", "slider", "beam")


def _axis_vec(v, ctx, where, n):
    from .model2d import _vec as v2
    from .model3d import _vec as v3, _unit
    if n == 2:
        a = np.array(v2(v, ctx, where) if not (isinstance(v, str) and v.strip().lstrip("-") in ("x", "y")) else
                     {"x": [1, 0], "-x": [-1, 0], "y": [0, 1], "-y": [0, -1]}[v.strip()], float)
        L = np.linalg.norm(a)
        if L == 0:
            raise SpecError("axis must not be zero", where)
        return a / L
    return _unit(v3(v, ctx, where), where)


def expand_geo(doc, ctx):
    f = doc.get("__file__", "")
    dim = doc["dim"]; n3 = dim == "3d"
    els = doc.get("elements") or {}
    if not any(isinstance(e, dict) and (e.get("type") in COMPOSITE or e.get("segments") not in (None, 1, "1"))
               for e in els.values()):
        return doc, {}
    out = copy.copy(doc); out.lines = getattr(doc, "lines", {})
    bodies = dict(doc["bodies"]); new = {}; groups = {}
    for en, e in els.items():
        t = e.get("type")
        where = f"elements.{en}"
        if t == "beam":
            from .expand_beam import expand_beam
            expand_beam(en, e, doc, ctx, bodies, new, groups)
            continue
        if e.get("segments") not in (None, 1, "1"):
            raise SpecError("in 2d and 3d, segments belongs to a beam (use type: beam)", where, line_of(e, "segments"), f)
        if t not in COMPOSITE:
            new[en] = e; continue
        refs = e.get("measure")
        if not isinstance(refs, (list, tuple)) or len(refs) != 2:
            raise SpecError(f"a {t}'s measure is [A.point, B.point]", where, line_of(e, "measure"), f)
        A = str(refs[0]).split(".")[0]; B = str(refs[1]).split(".")[0]
        k = e.get("k"); c = e.get("c", 0); rg = e.get("rigid") is True
        if rg:                                   # exact joint: projected constraints, no stiffness to choose
            k = 0.0; c = 0
        if k is None:
            raise SpecError(f"a {t} needs k (its stiffness across the joint, N/m)", where, line_of(e, "type"), f)
        hid = dict(__hidden__=True)
        rid = dict(__hidden__=True, __rigid__=True) if rg else hid
        members = []
        if t in ("weld", "hinge"):
            new[f"{en}__pin"] = dict(type="pin", measure=list(refs), k=k, c=c, **rid); members.append(f"{en}__pin")
            if t == "weld":
                if rg:
                    if n3:
                        new[f"{en}__rot"] = dict(type="__rotlock__", measure=[A, B], k_rot=0.0, **rid)
                    else:
                        new[f"{en}__rot"] = dict(type="spring", measure={"twist": [A, B]}, k=0.0, **rid)
                    members.append(f"{en}__rot")
                else:
                    if "k_rot" not in e:
                        raise SpecError("a weld needs k_rot (its stiffness against turning, N m/rad)", where,
                                        line_of(e, "type"), f)
                    if n3:
                        new[f"{en}__rot"] = dict(type="__rotlock__", measure=[A, B], k_rot=e["k_rot"],
                                                 c_rot=e.get("c_rot", 0), **hid)
                    else:
                        new[f"{en}__rot"] = dict(type="kelvin_voigt", measure={"twist": [A, B]}, k=e["k_rot"],
                                                 c=e.get("c_rot", 0), **hid)
                    members.append(f"{en}__rot")
            else:
                if n3 and rg:
                    if "axis" not in e:
                        raise SpecError("a 3d hinge needs axis (the line it turns about)", where, line_of(e, "type"), f)
                    new[f"{en}__rot"] = dict(type="__rotlock__", measure=[A, B], k_rot=0.0, hinge_axis=e["axis"], **rid)
                    members.append(f"{en}__rot")
                elif n3:
                    if "axis" not in e or "k_rot" not in e:
                        raise SpecError("a 3d hinge needs axis (the line it turns about) and k_rot (stiffness against "
                                        "tilting off that axis, N m/rad)", where, line_of(e, "type"), f)
                    new[f"{en}__rot"] = dict(type="__rotlock__", measure=[A, B], k_rot=e["k_rot"], c_rot=e.get("c_rot", 0),
                                             hinge_axis=e["axis"], **hid)
                    members.append(f"{en}__rot")
                if "k_twist" in e:
                    tw = {"twist": [A, B]}
                    if n3:
                        tw["axis"] = e["axis"]
                    new[f"{en}__twist"] = dict(type="kelvin_voigt", measure=tw, k=e["k_twist"], c=e.get("c_twist", 0), **hid)
                    members.append(f"{en}__twist")
            groups[en] = Group(en, members, None, kind=t)
            continue
        # slider
        if "axis" not in e:
            raise SpecError("a slider needs axis (the direction B may slide along, at the start)", where, line_of(e, "type"), f)
        ax = _axis_vec(e["axis"], ctx, where + ".axis", 3 if n3 else 2)
        perps = []
        if n3:
            tmp = np.array([1.0, 0, 0]) if abs(ax[0]) < 0.9 else np.array([0, 1.0, 0])
            u = tmp - (tmp @ ax) * ax; u /= np.linalg.norm(u); w = np.cross(ax, u)
            perps = [u, w]
        else:
            perps = [np.array([-ax[1], ax[0]])]
        for j, pv in enumerate(perps):
            nm = f"{en}__g{j + 1}"
            new[nm] = dict(type="kelvin_voigt", measure={"along": list(refs), "dir": [float(x) for x in pv], "turns_with": A},
                           k=k, c=c, **rid)
            members.append(nm)
        if "k_rot" in e:                         # rigid slider: k_rot (any value) locks the turn exactly
            kr = 0.0 if rg else e["k_rot"]; hr = rid if rg else hid
            if n3:
                new[f"{en}__rot"] = dict(type="__rotlock__", measure=[A, B], k_rot=kr, c_rot=e.get("c_rot", 0), **hr)
            else:
                new[f"{en}__rot"] = dict(type="kelvin_voigt", measure={"twist": [A, B]}, k=kr, c=e.get("c_rot", 0), **hr)
            members.append(f"{en}__rot")
        probe = f"{en}__slide"
        new[probe] = dict(type="spring", k=0.0, measure={"along": list(refs), "dir": [float(x) for x in ax], "turns_with": A},
                             **hid)
        if "friction" in e:          # Coulomb friction along the slide, regularised
            new[f"{en}__fric"] = dict(type="friction", measure={"along": list(refs), "dir": [float(x) for x in ax],
                                                                "turns_with": A}, F_c=e["friction"],
                                      v_eps=e.get("v_eps", 0.01), **hid)
            members.append(f"{en}__fric")
        groups[en] = Group(en, members, probe, kind="slider")
    out["bodies"] = bodies; out["elements"] = new
    return out, groups
