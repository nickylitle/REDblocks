"""Loads from a table: a force (or torque) that follows a CSV file or a list of points.

    push: {type: force, measure: plate.y,
           csv: {file: loads/impact.csv, against: time, x_unit: ms, y_unit: kN, interpolation: linear}}

csv keys
  file           path to the CSV (relative to the system file's folder; in the editor, the saved model's folder). Or give the points inline with x: [..] and
                 y: [..] instead of a file.
  against        time (default): F follows the clock. measure: F follows the element's measure (a nonlinear
                 load-deflection curve; it gives energy back on the way out, it is not a damper).
  x_unit         unit of the x column: a time unit (s, ms, us) against time; a length (m, mm, in) or an angle
                 (rad, deg) against a measure. Default s / m.
  y_unit         unit of the y column: a force (N, kN, lbf) or a torque (N m, lbf in) for a rotation. Default N.
  interpolation  linear (default), step (hold each value until the next x) or smooth (monotone cubic, PCHIP: no
                 overshoot between points; next to a flat stretch it rounds the corners and can add a few % of area,
                 so use linear when the impulse must match the table exactly).
  x_col, y_col   which columns: a number (1 = first) or the header text. Default the first two.
Outside the table the end values are held. A header row (any non-numeric cell) is skipped; '#' starts a comment.
The force acts like P on a force element: positive pushes the measure positive."""
import csv
import math
import os

import numpy as np

from .loader import SpecError, line_of
from .units import unit_factor, UnitError

CSV_KEYS = {"file", "x", "y", "against", "x_unit", "y_unit", "interpolation", "x_col", "y_col"}
_TIME = {"s", "ms", "us", "ns", "min", "h"}
_LEN = {"m", "mm", "cm", "um", "km", "in", "ft"}
_ANG = {"rad", "deg"}
_FORCE = {"N", "kN", "lbf"}


def _tokens(u):
    return [t for t in u.replace("*", " ").replace("/", " / ").split() if t]


def _kind(u):
    tk = _tokens(u)
    if "/" in tk:
        return "other"
    tk = [t.split("^")[0] for t in tk]
    if len(tk) == 1:
        t = tk[0]
        return "time" if t in _TIME else "length" if t in _LEN else "angle" if t in _ANG else \
            "force" if t in _FORCE else "other"
    if len(tk) == 2 and sum(t in _FORCE for t in tk) == 1 and sum(t in _LEN for t in tk) == 1:
        return "torque"
    return "other"


def _pchip(x, y, per=40, cap=20000):
    """Monotone cubic (Fritsch-Carlson) resampled to a dense table for linear lookup."""
    n = len(x)
    if n < 3:
        return x, y
    h = np.diff(x); dl = np.diff(y) / h
    d = np.zeros(n)
    for k in range(1, n - 1):
        if dl[k - 1] * dl[k] > 0:
            w1 = 2 * h[k] + h[k - 1]; w2 = h[k] + 2 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / dl[k - 1] + w2 / dl[k])
    for k, (a, b) in ((0, (0, 1)), (n - 1, (n - 2, n - 3))):
        hh0 = h[a] if k == 0 else h[n - 2]; hh1 = h[1] if k == 0 else h[n - 3]
        d0, d1 = (dl[0], dl[1]) if k == 0 else (dl[n - 2], dl[n - 3])
        v = ((2 * hh0 + hh1) * d0 - hh0 * d1) / (hh0 + hh1)
        if v * d0 <= 0:
            v = 0.0
        elif d0 * d1 <= 0 and abs(v) > abs(3 * d0):
            v = 3 * d0
        d[k] = v
    per = max(2, min(per, cap // max(n - 1, 1)))
    xs = [x[0]]; ys = [y[0]]
    for k in range(n - 1):
        for j in range(1, per + 1):
            s = j / per; t = x[k] + s * h[k]
            h00 = 2 * s ** 3 - 3 * s ** 2 + 1; h10 = s ** 3 - 2 * s ** 2 + s
            h01 = -2 * s ** 3 + 3 * s ** 2; h11 = s ** 3 - s ** 2
            xs.append(t); ys.append(h00 * y[k] + h10 * h[k] * d[k] + h01 * y[k + 1] + h11 * h[k] * d[k + 1])
    return np.array(xs), np.array(ys)


def _read_csv(path, xc, yc, where, ln, f):
    try:
        fh = open(path, newline="", encoding="utf-8-sig")
    except OSError as exc:
        raise SpecError(f"cannot open the CSV '{path}' ({exc.strerror})", where, ln, f)
    with fh:
        lines = [ln_ for ln_ in fh.read().splitlines() if ln_.strip() and not ln_.lstrip().startswith("#")]
    sample = "\n".join(lines[:50])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.reader(lines, dialect))
    rows = [[c.strip() for c in r if c.strip() != ""] for r in rows]
    rows = [r for r in rows if r]
    if not rows:
        raise SpecError(f"the CSV '{path}' has no rows", where, ln, f)

    def isnum(c):
        try:
            float(c); return True
        except ValueError:
            return False
    header = None
    if not all(isnum(c) for c in rows[0]):
        header = rows[0]; rows = rows[1:]

    def col(c, name):
        if c is None:
            return 0 if name == "x" else 1
        if isinstance(c, int) or (isinstance(c, str) and c.strip().lstrip("-").isdigit()):
            if int(c) < 1:
                raise SpecError(f"{name}_col counts from 1 (the first column), got {c}", where, ln, f)
            return int(c) - 1
        if header and str(c) in header:
            return header.index(str(c))
        raise SpecError(f"{name}_col '{c}' is not a column number or a header name" +
                        (f" (headers: {', '.join(header)})" if header else " (the file has no header row)"), where, ln, f)
    ix, iy = col(xc, "x"), col(yc, "y")
    xs, ys = [], []
    for k, r in enumerate(rows):
        try:
            xs.append(float(r[ix])); ys.append(float(r[iy]))
        except (IndexError, ValueError):
            raise SpecError(f"CSV '{os.path.basename(path)}' row {k + 1 + (header is not None)}: needs numbers in "
                            f"columns {ix + 1} and {iy + 1} (got {r})", where, ln, f)
    return xs, ys


def load_table(d, ctx, where, doc):
    """Returns (against_measure: bool, step: bool, x SI array, F SI array, max |dF/dx|)."""
    from .model import resolve
    f = doc.get("__file__", ""); spec = d["csv"]; ln = line_of(d, "csv")
    if not isinstance(spec, dict):
        raise SpecError("csv is {file: path.csv, against: time, x_unit: s, y_unit: N, interpolation: linear}",
                        where, ln, f)
    bad = set(spec) - CSV_KEYS
    if bad:
        raise SpecError("unknown csv key(s): " + ", ".join(sorted(bad)) + " (allowed: " + ", ".join(sorted(CSV_KEYS)) + ")",
                        where, ln, f)
    against = str(spec.get("against", "time")).strip()
    if against not in ("time", "measure"):
        raise SpecError("against is time or measure", where, ln, f)
    interp = str(spec.get("interpolation", "linear")).strip().lower()
    if interp not in ("linear", "step", "smooth"):
        raise SpecError("interpolation is linear, step or smooth", where, ln, f)
    xu = str(spec.get("x_unit", "s" if against == "time" else "m")); yu = str(spec.get("y_unit", "N"))
    try:
        fx = unit_factor(xu); fy = unit_factor(yu)
    except UnitError as exc:
        raise SpecError(f"csv units: {exc}", where, ln, f)
    kx = _kind(xu); ky = _kind(yu)
    if against == "time" and kx != "time":
        raise SpecError(f"x_unit '{xu}' is not a time (against: time needs s, ms, us...)", where, ln, f)
    if against == "measure" and kx not in ("length", "angle"):
        raise SpecError(f"x_unit '{xu}' is not a length or an angle (against: measure)", where, ln, f)
    if ky not in ("force", "torque"):
        raise SpecError(f"y_unit '{yu}' is not a force (N, kN, lbf) or a torque (N m)", where, ln, f)
    if "file" in spec:
        if "x" in spec or "y" in spec:
            raise SpecError("give file or the x / y lists, not both", where, ln, f)
        p = str(resolve(spec["file"], ctx, where + ".csv.file"))
        if not os.path.isabs(p):
            # a model typed in the editor has no file: the app sets SIMBLOCKS_MODEL_DIR to the saved model's folder
            base = os.path.dirname(f) if f and not f.startswith("<") else (os.environ.get("SIMBLOCKS_MODEL_DIR") or os.getcwd())
            p = os.path.join(base, p)
        xs, ys = _read_csv(p, spec.get("x_col"), spec.get("y_col"), where, ln, f)
    else:
        if "x" not in spec or "y" not in spec:
            raise SpecError("csv needs file (a path) or both x and y lists", where, ln, f)
        try:
            xs = [float(v) for v in spec["x"]]; ys = [float(v) for v in spec["y"]]
        except (TypeError, ValueError):
            raise SpecError("inline x and y are plain numbers in x_unit and y_unit (e.g. x: [0, 1, 2], x_unit: ms), "
                            "not quantities with units", where, ln, f)
        if len(xs) != len(ys):
            raise SpecError("x and y have different lengths", where, ln, f)
    if len(xs) < 1:
        raise SpecError("the table is empty", where, ln, f)
    x = np.array(xs) * fx; y = np.array(ys) * fy
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise SpecError("the table has a non-finite number", where, ln, f)
    if len(x) > 1 and np.any(np.diff(x) <= 0):
        k = int(np.argmax(np.diff(x) <= 0))
        raise SpecError(f"x must increase down the table (row {k + 2}: {xs[k + 1]} after {xs[k]})", where, ln, f)
    if interp == "smooth":
        x, y = _pchip(x, y)
    slope = float(np.max(np.abs(np.diff(y) / np.diff(x)))) if len(x) > 1 else 0.0
    return against == "measure", interp == "step", np.ascontiguousarray(x, float), np.ascontiguousarray(y, float), slope


def compile_load(e, d, ctx, where, doc, P, tab_k):
    """Fill P and tab_k for a table load; returns (typ code 13, tx, tF)."""
    meas, step, tx, tF, slope = load_table(d, ctx, where, doc)
    P[e, 0] = 0.0; P[e, 1] = 1.0 if meas else 0.0; P[e, 2] = 1.0 if step else 0.0
    tab_k[e] = slope if meas else 0.0              # a load-deflection curve sets a time step like a spring
    return 13, tx, tF


def check_load_kind(d, rotation, where, doc):
    """A table's y unit must match what it pushes: a force on a translation, a torque on a rotation."""
    spec = d.get("csv") or {}
    if not isinstance(spec, dict):
        return
    yu = str(spec.get("y_unit", "N")); ky = _kind(yu)
    f = doc.get("__file__", "")
    if rotation and ky == "force":
        raise SpecError(f"y_unit '{yu}' is a force, but this load turns a rotation: give a torque (N m, lbf in)",
                        where, line_of(d, "csv"), f)
    if not rotation and ky == "torque":
        raise SpecError(f"y_unit '{yu}' is a torque, but this load pushes a translation: give a force (N, kN, lbf)",
                        where, line_of(d, "csv"), f)
