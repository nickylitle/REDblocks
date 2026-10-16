"""Shared helper for the generic blocks: turn a material argument (catalog key, loose text or object) plus explicit
numbers into one object with the fields a formula needs. Explicit numbers win over catalog values."""
import math
from types import SimpleNamespace


def as_material(material):
    """Catalog object for a key, loose text ('7075', '4130 at 45 HRC') or an object (returned as is)."""
    if material is None or not isinstance(material, str):
        return material
    from . import get
    from .curves import resolve_material
    try:
        return get(str(material))
    except KeyError:
        return get(resolve_material(str(material)))


def mat_props(material=None, **given):
    """SimpleNamespace with each requested field: the explicit value when given, else the catalog value, else NaN.
    Example: mat_props('7075', E=None, nu=None, sy=None) -> namespace(E=..., nu=..., sy=...)."""
    m = as_material(material)
    out = {}
    for k, v in given.items():
        if v is None and m is not None:
            v = getattr(m, k, None)
        out[k] = math.nan if v is None else v
    return SimpleNamespace(**out)


def need(name, value, where):
    """Raise a clear error when a required input is missing (None or NaN)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ValueError(f"{where} needs '{name}'")
    return value
