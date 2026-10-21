"""Cylinders (3D): a wheel, roller or tread as a stack of discs.

    bodies:
      wheel:
        cylinders: {tread: {at: [0, 0, 0], axis: y, r: 30 mm, width: 20 mm, slices: 3}}
    elements:
      tire:    {type: kelvin_voigt, measure: {contact: [wheel.tread, floor.top]}, k: 20 kN/m, c: 40}
      tire_mu: {type: contact_friction, contact: tire, mu: 1.2}

Each slice is a disc (its rim, not a sphere) at an even spacing across the width, ends included. A disc touches a
plane at its rim point deepest into the plane, so a cylinder stands flat on its whole width, rolls about its own axis,
and rides up onto one edge when tipped. The contact elements on a cylinder are split into one per slice (`tire__d0`,
`tire__d1`, ...), each carrying its share of the stiffness and damping by the trapezoid rule (1/4, 1/2, 1/4 for three
slices). That share is a uniform line load across the width: a flat cylinder carries the full k, the friction about a
vertical axis (scrub when it turns on the spot) is mu N w / 4, the value of a uniform line, and an edge carries an end
share. The original name answers outputs and history as the whole: force is the sum of the slices (tracked exactly
every step), measure the deepest slice.

Laws a cylinder contact takes: spring, kelvin_voigt, dashpot, hertz (k, c and K are shared out; for hertz the
share is exact for a line of Hertz points, not the line-contact law), impact (each slice its own rigid impact).
"""
import copy

import numpy as np

from .loader import SpecError, line_of

SCALED = {"spring": ("k",), "kelvin_voigt": ("k", "c"), "dashpot": ("c",), "hertz": ("K",), "impact": ()}


def cyl_slices(fv, ctx, where):
    """(r, width, slices) of a cylinder feature, checked."""
    from .model import resolve, num
    r = num(resolve(fv["r"], ctx, where + ".r"), where + ".r")
    wd = num(resolve(fv.get("width", 0), ctx, where + ".width"), where + ".width")
    ns = fv.get("slices", 3 if wd > 0 else 1)
    try:
        ns = int(num(resolve(ns, ctx, where + ".slices"), where + ".slices"))
    except (TypeError, ValueError):
        raise SpecError("slices is a whole number (default 3)", where)
    if r <= 0 or wd < 0:
        raise SpecError("a cylinder needs r > 0 and width >= 0", where)
    if not 1 <= ns <= 16:
        raise SpecError("slices is 1 to 16 (3 catches the edges and the scrub; 5 is smoother when it rides an edge)", where)
    if ns > 1 and wd <= 0:
        raise SpecError("more than one slice needs a width", where)
    return r, wd, ns


def cyl_offsets(wd, ns):
    """[(axial offset, weight)] of the slices: even spacing, ends included, trapezoid weights (sum 1)."""
    ns = int(ns)
    if ns <= 1:
        return [(0.0, 1.0)]
    xs = np.linspace(-0.5 * wd, 0.5 * wd, ns)
    w = np.full(ns, 1.0 / (ns - 1)); w[0] *= 0.5; w[-1] *= 0.5
    return list(zip(xs.tolist(), w.tolist()))


def _cyl_ref(ref, doc):
    """(body, feature, cylinder spec) when ref names a cylinder, else None."""
    b, _, fn = str(ref).partition(".")
    body = (doc.get("bodies") or {}).get(b)
    cyl = (body or {}).get("cylinders") if isinstance(body, dict) else None
    if isinstance(cyl, dict) and fn in cyl:
        return b, fn, cyl[fn]
    return None


def expand_cylinders(doc, ctx):
    """(doc, groups): contact elements on a cylinder split per slice. The doc is not modified."""
    if doc.get("dim") != "3d":
        for bn, b in (doc.get("bodies") or {}).items():
            if isinstance(b, dict) and b.get("cylinders"):
                raise SpecError("cylinders are 3D geometry (in 2D a wheel is a circle)", f"bodies.{bn}.cylinders")
        return doc, {}
    if not any(isinstance(b, dict) and b.get("cylinders") for b in (doc.get("bodies") or {}).values()):
        return doc, {}
    from .model import resolve, num
    from .expand import Group
    f = doc.get("__file__", "")
    els = doc.get("elements") or {}
    split = {}                                   # contact element -> [(slice name, weight)]
    new = {}; groups = {}
    for en, e in els.items():
        meas = e.get("measure") if isinstance(e, dict) else None
        refs = meas.get("contact") if isinstance(meas, dict) else None
        hit = [(_cyl_ref(r, doc), i) for i, r in enumerate(refs or [])] if isinstance(refs, (list, tuple)) else []
        hit = [(h, i) for h, i in hit if h]
        if not hit:
            new[en] = e; continue
        where = f"elements.{en}"
        if len(hit) > 1:
            raise SpecError("contact between two cylinders: give one of them a plane (a flat face) where they touch",
                            where, line_of(e, "measure"), f)
        t = e.get("type")
        if t not in SCALED:
            raise SpecError(f"a cylinder's contact takes {', '.join(SCALED)} (a {t} cannot be shared across slices)",
                            where, line_of(e, "type"), f)
        (b, fn, spec), i = hit[0]
        fw = f"bodies.{b}.cylinders.{fn}"
        _r, wd, ns = cyl_slices(spec, ctx, fw)
        names = []
        for j, (_off, w) in enumerate(cyl_offsets(wd, ns)):
            d = {k: v for k, v in e.items()}
            m = dict(meas); rr = list(refs); rr[i] = f"{b}.{fn}__d{j}"; m["contact"] = rr; d["measure"] = m
            for key in SCALED[t]:
                if key in e:
                    d[key] = num(resolve(e[key], ctx, f"{where}.{key}"), f"{where}.{key}") * w
            d["__hidden__"] = True
            sn = f"{en}__d{j}"; new[sn] = d; names.append(sn)
        split[en] = names
        groups[en] = Group(en, names, None, kind="parallel")
    for en, e in list(new.items()):              # friction on a split contact: one per slice, the same mu
        if isinstance(e, dict) and e.get("type") == "contact_friction" and e.get("contact") in split:
            del new[en]
            names = []
            for j, cn in enumerate(split[e["contact"]]):
                d = dict(e); d["contact"] = cn; d["__hidden__"] = True
                sn = f"{en}__d{j}"; new[sn] = d; names.append(sn)
            groups[en] = Group(en, names, None, kind="parallel")
    out = copy.copy(doc); out.lines = getattr(doc, "lines", {})
    out["elements"] = new
    return out, groups
