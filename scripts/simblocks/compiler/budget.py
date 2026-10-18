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
    """[(name, kg, where it comes from)]."""
    from .model import resolve, num
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
            for i, name in enumerate(M.body_names):
                if "__" in name:
                    continue
                if mass is not None and float(mass[i]) > 0:
                    out.append((name, float(mass[i]), "body"))
        for en, info in (getattr(M, "el_info", None) or {}).items():
            if "__" in en:
                continue
            if info.get("mass"):
                out.append((en, float(info["mass"]), "element"))
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
