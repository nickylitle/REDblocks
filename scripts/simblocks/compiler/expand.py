"""Doc-level expansion before compiling: elements that stand for several engine pieces.

1D  `segments: n` on a spring, dashpot or kelvin_voigt with mass: the element becomes a chain of n springs (n k, n c)
    with n - 1 hidden node bodies of mass m / n between its ends (the ends get m / 2n each). This resolves the
    spring's own dynamics (surge, wave travel) instead of a single lumped mass.

The parent name stays usable in outputs and exports: force reduces over every segment (max, min, max_abs; final
and history = the segment at the measure's positive end), measure is the total stretch (a hidden probe element on
the original measure), absorbed_work and dissipated are summed.

Hidden names: '<element>__n<j>' (bodies), '<element>__s<j>' (segments), '<element>__probe'."""
import copy

from .loader import SpecError, line_of
from .expr import eval_measure, ExprError

SEGMENTABLE_1D = ("spring", "dashpot", "kelvin_voigt")


class Group:
    def __init__(self, parent, segs, probe=None, kind="chain"):
        self.parent = parent; self.segs = segs; self.probe = probe; self.kind = kind


def _num(v, ctx, where):
    from .model import resolve, num
    return num(resolve(v, ctx, where), where)


def _desugar_dc_motors(doc, ctx):
    """dc_motor: a DC motor on a fixed supply V (no battery, no circuit). It runs as a drive whose pack never sags:
    zero internal resistance, a capacity that never empties, cells = V / 4.2 V (the pack's full-charge voltage)."""
    els = doc.get("elements") or {}
    if not any(isinstance(e, dict) and e.get("type") == "dc_motor" for e in els.values()):
        return doc
    from .model import resolve, num
    out = copy.copy(doc); out.lines = getattr(doc, "lines", {}); new = {}
    for en, e in els.items():
        if not (isinstance(e, dict) and e.get("type") == "dc_motor"):
            new[en] = e; continue
        where = f"elements.{en}"
        d = {k: v for k, v in e.items() if k not in ("Kt", "V", "type")}
        d["type"] = "drive"
        if "Kt" in e:
            Kt = num(resolve(e["Kt"], ctx, where + ".Kt"), where + ".Kt")
            if Kt <= 0:
                raise SpecError("Kt must be positive", where, line_of(e, "Kt"), doc.get("__file__", ""))
            d["kv"] = 1.0 / Kt                      # rad/s per volt
        V = num(resolve(e["V"], ctx, where + ".V"), where + ".V")
        if V <= 0:
            raise SpecError("V (the supply voltage) must be positive", where, line_of(e, "V"), doc.get("__file__", ""))
        d["cells"] = V / 4.2; d["R0"] = 0.0; d["capacity"] = 1e18
        d.setdefault("I_max", 1e9)
        new[en] = d
    out["elements"] = new
    return out


def expand(doc, ctx):
    """Returns (doc_for_the_compiler, {parent: Group}). The input doc is not modified."""
    from .expand_shape import expand_shapes
    doc = _desugar_dc_motors(doc, ctx)
    doc = expand_shapes(doc, ctx)
    dim = doc.get("dim", "1d")
    els = doc.get("elements") or {}
    if dim != "1d":
        from .expand_geo import expand_geo
        return expand_geo(doc, ctx)
    if not any(isinstance(e, dict) and e.get("segments") not in (None, 1, "1") for e in els.values()):
        return doc, {}
    from .model import numeric_namespace
    f = doc.get("__file__", "")
    out = copy.copy(doc)
    bodies = dict(doc["bodies"]); new_els = {}
    out.lines = getattr(doc, "lines", {})
    groups = {}
    nums = numeric_namespace(ctx)
    names = [b for b, d in bodies.items() if not d.get("ground")]
    for en, e in els.items():
        n = e.get("segments")
        if n in (None, 1, "1"):
            new_els[en] = e; continue
        where = f"elements.{en}"
        try:
            n = int(_num(n, ctx, where + ".segments"))
        except Exception:
            raise SpecError("segments is a whole number (2 or more)", where, line_of(e, "segments"), f)
        if n < 2:
            new_els[en] = e; continue
        t = e["type"]
        if t not in SEGMENTABLE_1D:
            raise SpecError(f"segments splits a spring, dashpot or kelvin_voigt (a {t} does not divide into equal "
                            "pieces)", where, line_of(e, "segments"), f)
        if "mass" not in e:
            raise SpecError("segments needs the element's mass (it is what the segments carry)", where,
                            line_of(e, "segments"), f)
        if e.get("one_sided"):
            raise SpecError("a one_sided contact does not split into segments", where, line_of(e, "one_sided"), f)
        try:
            coef, const = eval_measure(e["measure"], list(bodies), nums)
        except ExprError as exc:
            raise SpecError(f"measure: {exc}", where, line_of(e, "measure"), f)
        moving = {b: c for b, c in coef.items() if b in names}
        pos = {b: c for b, c in moving.items() if c > 0}; neg = {b: c for b, c in moving.items() if c < 0}
        if len(pos) > 1 or len(neg) > 1 or not moving:
            raise SpecError("segments needs a measure between at most two bodies, like 'a - b' or 'R * rotor - b'",
                            where, line_of(e, "measure"), f)
        m = _num(e["mass"], ctx, where + ".mass")
        # end values: s = eA - eB with eA = cA qA + s0, eB = cB qB (cB >= 0)
        q0 = {b: _num(bodies[b].get("x0", 0.0), ctx, f"bodies.{b}.x0") for b in moving}
        v0 = {b: _num(bodies[b].get("v0", 0.0), ctx, f"bodies.{b}.v0") for b in moving}
        (a, ca), = pos.items() if pos else ((None, 0.0),)
        (b_, cb), = neg.items() if neg else ((None, 0.0),)
        cb = -cb
        eA = (ca * q0[a] if a else 0.0) + const; eB = cb * q0[b_] if b_ else 0.0
        vA = ca * v0[a] if a else 0.0; vB = cb * v0[b_] if b_ else 0.0
        endA = (f"{ca!r} * {a} + {const!r}" if a else f"{const!r}")
        endB = (f"{cb!r} * {b_}" if b_ else "0")
        law = {}
        for k in ("k", "c"):
            if k in e:
                law[k] = _num(e[k], ctx, f"{where}.{k}") * n
        nodes = [f"{en}__n{j}" for j in range(1, n)]
        for j, nd in enumerate(nodes, start=1):
            fr = j / n
            bodies[nd] = {"dof": "translation", "mass": m / n, "x0": eB + fr * (eA - eB), "v0": vB + fr * (vA - vB),
                          "label": f"{en} node {j}", "__hidden__": True}
        chain = [endB] + nodes + [endA]
        segs = []
        for j in range(n):
            sn = f"{en}__s{j + 1}"
            lo, hi = chain[j], chain[j + 1]
            new_els[sn] = dict(type=t, measure=f"({hi}) - ({lo})", mass=m / n, **law, __lump__="ends",
                               label=f"{e.get('label', en)} segment {j + 1}", __hidden__=True)
            segs.append(sn)
        probe = f"{en}__probe"
        new_els[probe] = dict(type="spring", k=0.0, measure=e["measure"], __hidden__=True)
        groups[en] = Group(en, segs[::-1], probe)          # segs[0] = the segment at the positive end
    out["bodies"] = bodies; out["elements"] = new_els
    return out, groups
