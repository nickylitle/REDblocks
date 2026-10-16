"""Unit parsing for system files. A value is a number (already SI) or a string 'number unit-expression',
e.g. '2.5 mm', '12000 rpm', '7.6e-4 kg m^2', '1100 rpm/V', '200 kN/m', '450 mAh'.
Unit expressions: tokens separated by spaces or '*', one optional '/', exponents with '^' (m^2, s^-1).
Every token must be known; unknown units are errors, never guessed."""
import math
import re

# factor to SI for each token
_U = {
    # length
    "m": 1.0, "mm": 1e-3, "cm": 1e-2, "um": 1e-6, "km": 1e3, "in": 0.0254, "ft": 0.3048,
    # mass
    "kg": 1.0, "g": 1e-3, "lb": 0.45359237, "oz": 0.028349523125,
    # time
    "s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "min": 60.0, "h": 3600.0,
    # force / energy / power
    "N": 1.0, "kN": 1e3, "lbf": 4.4482216152605, "J": 1.0, "kJ": 1e3, "W": 1.0, "kW": 1e3, "Wh": 3600.0,
    # pressure / stress
    "Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "GPa": 1e9, "psi": 6894.757293168, "ksi": 6894757.293168,
    # angle
    "rad": 1.0, "deg": math.pi / 180.0,
    # rotation rate / frequency
    "rpm": 2 * math.pi / 60.0, "rps": 2 * math.pi, "Hz": 1.0,
    # electrical
    "V": 1.0, "A": 1.0, "mA": 1e-3, "ohm": 1.0, "mohm": 1e-3, "mOhm": 1e-3, "F": None,  # F is ambiguous
    "Ah": 3600.0, "mAh": 3.6, "C": 1.0,
    # temperature differences / absolute in K
    "K": 1.0,
    # dimensionless
    "%": 0.01, "1": 1.0,
}
_AMBIGUOUS = {"F": "use 'farad' for capacitance or 'N' for force"}
_U["farad"] = 1.0

_NUM = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(.*)$")


class UnitError(ValueError):
    pass


def _token_factor(tok):
    tok = tok.strip()
    if not tok:
        return 1.0
    p = 1.0
    if "^" in tok:
        base, ex = tok.split("^", 1)
        try:
            p = float(ex)
        except ValueError:
            raise UnitError(f"bad exponent in unit '{tok}'")
        tok = base
    if tok in _AMBIGUOUS:
        raise UnitError(f"unit '{tok}' is ambiguous: {_AMBIGUOUS[tok]}")
    if tok not in _U or _U[tok] is None:
        raise UnitError(f"unknown unit '{tok}'")
    return _U[tok] ** p


def unit_factor(expr):
    """SI factor of a unit expression like 'kN/m', 'kg m^2', 'rpm/V', 'N s/m'."""
    expr = expr.strip()
    if not expr:
        return 1.0
    if expr.count("/") > 1:
        raise UnitError(f"use at most one '/' in unit '{expr}'")
    num, _, den = expr.partition("/")
    f = 1.0
    for t in re.split(r"[\s*]+", num.strip()):
        f *= _token_factor(t)
    for t in re.split(r"[\s*]+", den.strip()):
        if t:
            f /= _token_factor(t)
    return f


def parse_quantity(v):
    """Number -> itself (SI). 'number unit' -> SI float. Raises UnitError on anything else."""
    if isinstance(v, bool):
        raise UnitError("expected a number, got true/false")
    if isinstance(v, (int, float)):
        return float(v)
    if not isinstance(v, str):
        raise UnitError(f"expected a number or 'number unit', got {type(v).__name__}")
    m = _NUM.match(v)
    if not m:
        raise UnitError(f"cannot read '{v}' as 'number unit'")
    return float(m.group(1)) * unit_factor(m.group(2))


def looks_like_quantity(s):
    return isinstance(s, str) and bool(_NUM.match(s))
