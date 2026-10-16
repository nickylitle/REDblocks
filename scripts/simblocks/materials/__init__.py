"""Materials catalog: isotropic metals and polymers, composite laminae, fluids, CLT.

    from simblocks.materials import get, catalog
    steel = get("4130_45HRC");  cf = get("T300_5208");  air = get("air_20C")
"""
from .base import Isotropic, Lamina, Fluid
from . import metals, polymers, composites, fluids, clt
from .polymers import shore_A_to_E
from .composites import LAMINAE, FIBERS, MATRICES

_ALL = {**metals.ALL, **polymers.ALL, **composites.LAMINAE, **fluids.FLUIDS}


def get(key):
    """Look up a material by catalog key (case-insensitive)."""
    for k, v in _ALL.items():
        if k.lower() == key.lower():
            return v
    raise KeyError(f"{key!r} not in catalog. Keys: {sorted(_ALL)}")


def catalog():
    """Dict of every catalog material by key."""
    return dict(_ALL)


def catalog_markdown():
    rows = ["# Materials catalog", "",
            "## Isotropic", "", "| key | name | E GPa | nu | rho | sy MPa | su MPa | family |",
            "|---|---|---|---|---|---|---|---|"]
    for k, m in {**metals.ALL, **polymers.ALL}.items():
        rows.append(f"| `{k}` | {m.name} | {m.E/1e9:.3g} | {m.nu} | {m.rho:.0f} | {m.sy/1e6:.0f} | "
                    f"{m.su/1e6:.0f} | {m.family} |")
    rows += ["", "## Composite laminae", "",
             "| key | name | E1 GPa | E2 GPa | G12 GPa | nu12 | Xt/Xc MPa | Yt/Yc MPa | S MPa | t_ply mm |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for k, l in LAMINAE.items():
        rows.append(f"| `{k}` | {l.name} | {l.E1/1e9:.3g} | {l.E2/1e9:.3g} | {l.G12/1e9:.3g} | {l.nu12} | "
                    f"{l.Xt/1e6:.0f}/{l.Xc/1e6:.0f} | {l.Yt/1e6:.0f}/{l.Yc/1e6:.0f} | {l.S/1e6:.0f} | {l.t_ply*1e3:.3g} |")
    rows += ["", "## Constituents (micromechanics)", "", "Fibers: " + ", ".join(f"`{k}`" for k in FIBERS),
             "", "Matrices: " + ", ".join(f"`{k}`" for k in MATRICES),
             "", "## Fluids", "", ", ".join(f"`{k}`" for k in fluids.FLUIDS)]
    return "\n".join(rows) + "\n"

from . import curves  # noqa: E402  (needs _ALL defined)
from .curves import load_curve, resolve_material, read_curve_csv, Curve
from . import cards  # noqa: E402
from .cards import laminate_card, load_card, isotropic_card, LaminateCard
from . import custom  # noqa: E402
custom.load_env()       # SIMBLOCKS_MATERIALS folders (desktop app sets it; study workers inherit it)
