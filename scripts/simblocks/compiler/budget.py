"""Weight budget: every body's mass plus listed items, against a limit (a 3 lb class, a payload cap).

    budget:
      limit: 3 lb
      bodies: true                    # count every modelled body's mass (default true)
      items: {wiring: 25 g, receiver: 6 g, screws: 18 g}

Outputs added to the results: mass_total and mass_margin (limit - total), both in the limit's unit. A 1D rotation
body carries an inertia, not a mass: list its mass as an item."""
from .loader import SpecError, line_of
from .units import unit_factor


def _unit(lim):
    s = str(lim).strip().split(None, 1)
    return s[1] if len(s) > 1 else "kg"


def items(M, doc):
    """[(name, kg, where it comes from)]. Every modelled mass once: the bodies you wrote, the mass of each element that
    carries one (a 1D spring's or a segmented chain's, whatever it was split into), the hidden node bodies of an
    expanded beam (counted under the beam), and the items you list. An inertia (kg m^2) is not a weight."""
    from .model import resolve, num
    from .exports import element_is_rotational
    b = doc.get("budget") or {}
    out = []
    if b.get("bodies", True):
        dim = getattr(M, "dim", "1d")
        if dim == "1d":
            for name, bd in (doc.get("bodies") or {}).items():
                if bd.get("ground") or "__" in name or bd.get("__hidden__"):
                    continue
                if "mass" in bd:
                    out.append((name, num(resolve(bd["mass"], M.ctx, f"bodies.{name}.mass"), f"bodies.{name}.mass"), "body"))
        else:
            mass = M.arrays.get("mass")
            beams = (getattr(M, "cdoc", None) or {}).get("__beam_mass__") or {}
            ends = {}
            for bm in beams.values():
                for bn, x in bm["ends"].items():
                    ends[bn] = ends.get(bn, 0.0) + x
            for i, name in enumerate(M.body_names):
                if mass is None or float(mass[i]) <= 0 or "__" in name:   # a beam's node bodies: counted with the beam
                    continue
                own = float(mass[i]) - ends.get(name, 0.0)     # less the beam ends lumped onto it
                if own > 0:
                    out.append((name, own, "body"))
            out += [(k, bm["total"], "element") for k, bm in beams.items() if bm["total"] > 0]
        if getattr(M, "dim", "1d") == "1d":
            udoc = getattr(M, "doc", None) or doc            # the file as written: a segmented element under its own name
            for en, e in (udoc.get("elements") or {}).items():
                if "__" in en or not isinstance(e, dict) or e.get("mass") in (None, ""):
                    continue
                if element_is_rotational(M, en):       # a shaft's 'mass' is its inertia
                    continue
                out.append((en, num(resolve(e["mass"], M.ctx, f"elements.{en}.mass"), f"elements.{en}.mass"), "element"))
    for k, v in (b.get("items") or {}).items():
        out.append((k, num(resolve(v, M.ctx, f"budget.items.{k}"), f"budget.items.{k}"), "item"))
    return out


def summary(M, doc):
    b = doc.get("budget")
    if not b:
        return None
    if not isinstance(b, dict) or "limit" not in b:
        raise SpecError("budget is {limit: 3 lb, items: {name: mass}, bodies: true}", "budget", line_of(doc, "budget"),
                        doc.get("__file__", ""))
    from .model import resolve, num
    lim = num(resolve(b["limit"], M.ctx, "budget.limit"), "budget.limit")
    u = _unit(b["limit"]) if isinstance(b["limit"], str) else "kg"
    f = unit_factor(u)
    its = items(M, doc)
    tot = sum(x for _, x, _ in its)
    return dict(limit=lim / f, total=tot / f, margin=(lim - tot) / f, unit=u,
                items=[dict(name=n, mass=x / f, source=s) for n, x, s in its])
