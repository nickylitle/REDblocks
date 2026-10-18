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
  x_col, y_col   which columns: a number (1 = first) or the header text (with or without its '[unit]'). Default
                 the first two. A header written as 'name [unit]' (every CSV export) gives that column's unit, so
                 x_unit and y_unit can be left out: another model's export reads straight in (see chain.py).
Outside the table the end values are held. A header row (any non-numeric cell) is skipped; '#' starts a comment.
The force acts like P on a force element: positive pushes the measure positive."""
import csv
import math
import os

import numpy as np

from .loader import SpecError, line_of
from .units import unit_factor, UnitError

CSV_KEYS = {"file", "series", "x", "y", "against", "x_unit", "y_unit", "interpolation", "x_col", "y_col"}
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


def _parse(path, where, ln=None, f=""):
    """(header list or None, rows of text cells) from a CSV: header row optional, '#' comments, , ; tab or space."""
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
    header = None
    if not all(_isnum(c) for c in rows[0]):
        header = rows[0]; rows = rows[1:]
    return header, rows


def _isnum(c):
    try:
        float(c); return True
    except ValueError:
        return False


def _read_rows(path, where, ln=None, f=""):
    """(header or None, float array of every row) for chained inputs (from_csv)."""
    header, rows = _parse(path, where, ln, f)
    try:
        data = np.array([[float(c) for c in r] for r in rows], float)
    except ValueError:
        raise SpecError(f"CSV '{os.path.basename(path)}' has a cell that is not a number", where, ln, f)
    if data.ndim != 2:
        raise SpecError(f"CSV '{os.path.basename(path)}' rows have different numbers of cells", where, ln, f)
    return header, data


def _read_csv(path, xc, yc, where, ln, f):
    from .chain import find_column, split_header
    header, rows = _parse(path, where, ln, f)
    _read_csv.units = (None, None)

    def col(c, name):
        if c is None:
            return 0 if name == "x" else 1
        if isinstance(c, int) or (isinstance(c, str) and c.strip().lstrip("-").isdigit()):
            if int(c) < 1:
                raise SpecError(f"{name}_col counts from 1 (the first column), got {c}", where, ln, f)
            return int(c) - 1
        j = find_column(header, c)
        if j is not None:
            return j
        raise SpecError(f"{name}_col '{c}' is not a column number or a header name" +
                        (f" (headers: {', '.join(header)})" if header else " (the file has no header row)"), where, ln, f)
    ix, iy = col(xc, "x"), col(yc, "y")
    if header:                                     # an export's header carries each column's unit: 'Fz [kN]'
        _read_csv.units = (split_header(header[ix])[1] if ix < len(header) else None,
                           split_header(header[iy])[1] if iy < len(header) else None)
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
    xs = ys = None
    hx = hy = None
    ser = None
    if "series" in spec:                           # a time series from another model's port (chain.py)
        from .chain import Series
        if against != "time" or set(spec) & {"file", "x", "y", "x_unit", "y_unit", "x_col", "y_col"}:
            raise SpecError("csv: {series: $block.port} is a load against time and takes only interpolation", where, ln, f)
        ser = resolve(spec["series"], ctx, where + ".csv.series")
        if not isinstance(ser, Series):
            raise SpecError(f"csv series must name a time-series port ($F_in or $block.F_out), got {ser!r}", where, ln, f)
        xs = [float(v) for v in ser.t]; ys = [float(v) for v in ser.y]       # already SI
        hx, hy = "s", "N"
    if "file" in spec:
        if "x" in spec or "y" in spec:
            raise SpecError("give file or the x / y lists, not both", where, ln, f)
        p = str(resolve(spec["file"], ctx, where + ".csv.file"))
        if not os.path.isabs(p):
            # a model typed in the editor has no file: the app sets SIMBLOCKS_MODEL_DIR to the saved model's folder
            base = os.path.dirname(f) if f and not f.startswith("<") else (os.environ.get("SIMBLOCKS_MODEL_DIR") or os.getcwd())
            p = os.path.join(base, p)
        xs, ys = _read_csv(p, spec.get("x_col"), spec.get("y_col"), where, ln, f)
        hx, hy = _read_csv.units
    xu = str(spec.get("x_unit") or hx or ("s" if against == "time" else "m"))
    yu = str(spec.get("y_unit") or hy or "N")
    load_table.last_yu = yu if ser is None else (ser.unit or "any")
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
    if xs is None:
        if "x" not in spec or "y" not in spec:
            raise SpecError("csv needs file (a path) or both x and y lists", where, ln, f)
        try:
            # a list of numbers, or $block.result naming a table a block returned (in SI, like every block result)
            xv = resolve(spec["x"], ctx, where + ".csv.x") if isinstance(spec["x"], str) else spec["x"]
            yv = resolve(spec["y"], ctx, where + ".csv.y") if isinstance(spec["y"], str) else spec["y"]
            xs = [float(v) for v in xv]; ys = [float(v) for v in yv]
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
    dx0 = np.diff(x); dx0 = dx0[dx0 > 1e-6 * (x[-1] - x[0] + 1e-30)] if len(x) > 1 else dx0
    row_dx = float(np.median(dx0)) if len(dx0) else 0.0          # the table's own row spacing (before smoothing)
    if interp == "smooth":
        x, y = _pchip(x, y)
    slope = float(np.max(np.abs(np.diff(y) / np.diff(x)))) if len(x) > 1 else 0.0
    load_table.row_dx = row_dx
    return against == "measure", interp == "step", np.ascontiguousarray(x, float), np.ascontiguousarray(y, float), slope


def compile_load(e, d, ctx, where, doc, P, tab_k):
    """Fill P and tab_k for a table load; returns (typ code 13, tx, tF)."""
    meas, step, tx, tF, slope = load_table(d, ctx, where, doc)
    P[e, 0] = 0.0; P[e, 1] = 1.0 if meas else 0.0; P[e, 2] = 1.0 if step else 0.0
    if not meas:                          # a table against time: resolve its rows (half the typical row spacing)
        P[e, 3] = 0.5 * getattr(load_table, "row_dx", 0.0)
    tab_k[e] = slope if meas else 0.0              # a load-deflection curve sets a time step like a spring
    return 13, tx, tF


def check_load_kind(d, rotation, where, doc):
    """A table's y unit must match what it pushes: a force on a translation, a torque on a rotation."""
    spec = d.get("csv") or {}
    if not isinstance(spec, dict):
        return
    yu = str(spec.get("y_unit") or getattr(load_table, "last_yu", None) or "N"); ky = _kind(yu)
    f = doc.get("__file__", "")
    if rotation and ky == "force":
        raise SpecError(f"y_unit '{yu}' is a force, but this load turns a rotation: give a torque (N m, lbf in)",
                        where, line_of(d, "csv"), f)
    if not rotation and ky == "torque":
        raise SpecError(f"y_unit '{yu}' is a torque, but this load pushes a translation: give a force (N, kN, lbf)",
                        where, line_of(d, "csv"), f)


# ---------------------------------------------------------------------------------- standard load shapes
PULSE_KINDS = {
    # kind: (required, optional with defaults)
    "half_sine":  (("peak", "duration"), {"start": 0.0}),
    "haversine":  (("peak", "duration"), {"start": 0.0}),
    "triangle":   (("peak", "duration"), {"start": 0.0, "peak_at": 0.5}),
    "rectangle":  (("peak", "duration"), {"start": 0.0, "rise": 0.0}),
    "trapezoid":  (("peak", "duration", "rise"), {"start": 0.0, "fall": None}),
    "ramp":       (("to", "duration"), {"start": 0.0, "from": 0.0}),
    "step":       (("to",), {"start": 0.0, "from": 0.0}),
    "sine":       (("amplitude", "frequency"), {"offset": 0.0, "phase": 0.0, "start": 0.0, "stop": None}),
    "square_wave": (("amplitude", "frequency"), {"offset": 0.0, "duty": 0.5, "start": 0.0, "stop": None}),
    "sweep":      (("amplitude", "f_start", "f_end", "duration"), {"start": 0.0, "offset": 0.0}),
    "decay":      (("peak", "tau"), {"start": 0.0}),
}
PULSE_HELP = """pulse: a standard load shape against time (instead of P or csv). kind and its numbers (with units):
  half_sine   peak, duration, start          F = peak sin(pi t / duration) for one half wave
  haversine   peak, duration, start          F = peak (1 - cos(2 pi t / duration)) / 2: a smooth bump
  triangle    peak, duration, start, peak_at (0-1 of the duration, default 0.5)
  rectangle   peak, duration, start, rise    a flat pulse; rise (optional) softens both edges
  trapezoid   peak, duration, rise, fall, start
  ramp        to, duration, start, from      from -> to over the duration, then held
  step        to, start, from                jumps at start (one time step wide)
  sine        amplitude, frequency, offset, phase, start, stop     an oscillation (runs to the end if no stop)
  square_wave amplitude, frequency, duty, offset, start, stop
  sweep       amplitude, f_start, f_end, duration, start, offset   a chirp: frequency rises linearly
  decay       peak, tau, start               F = peak exp(-t / tau)
Before start and after the pulse the force is 0 (ramp and step hold their end value; sine and square_wave their
offset)."""


def pulse_table(d, ctx, where, doc):
    """(x seconds, F newtons or N m) dense enough for linear lookup."""
    from .model import resolve, num
    f = doc.get("__file__", ""); spec = d["pulse"]; ln = line_of(d, "pulse")
    if not isinstance(spec, dict) or "kind" not in spec:
        raise SpecError("pulse is {kind: half_sine, peak: 2 kN, duration: 5 ms} (kinds: " + ", ".join(PULSE_KINDS) + ")",
                        where, ln, f)
    kind = str(spec["kind"]).strip().lower()
    if kind not in PULSE_KINDS:
        raise SpecError(f"pulse kind '{kind}' is not one of: " + ", ".join(PULSE_KINDS), where, ln, f)
    req, opt = PULSE_KINDS[kind]
    bad = set(spec) - {"kind"} - set(req) - set(opt)
    if bad:
        raise SpecError(f"a {kind} pulse takes " + ", ".join(list(req) + list(opt)) + "; unknown: " + ", ".join(sorted(bad)),
                        where, ln, f)
    miss = [k for k in req if k not in spec]
    if miss:
        raise SpecError(f"a {kind} pulse needs " + ", ".join(miss), where, ln, f)
    v = {k: num(resolve(spec[k], ctx, f"{where}.pulse.{k}"), f"{where}.pulse.{k}") for k in req}
    for k, dv in opt.items():
        v[k] = num(resolve(spec[k], ctx, f"{where}.pulse.{k}"), f"{where}.pulse.{k}") if k in spec else dv
    try:
        T = float(num(resolve((doc.get("run") or {}).get("t_end", 1.0), ctx, "run.t_end"), "run.t_end"))
    except Exception:
        T = 1.0
    t0 = float(v["start"])
    for k in ("duration", "tau", "frequency", "f_start", "f_end"):
        if k in v and v[k] is not None and v[k] <= 0:
            raise SpecError(f"pulse {k} must be positive", where, ln, f)
    N = 400
    if kind in ("half_sine", "haversine"):
        D = v["duration"]; t = np.linspace(0, D, N + 1)
        y = v["peak"] * (np.sin(np.pi * t / D) if kind == "half_sine" else 0.5 * (1 - np.cos(2 * np.pi * t / D)))
        x, F = t0 + t, y
    elif kind == "triangle":
        D = v["duration"]; a = min(max(float(v["peak_at"]), 0.0), 1.0)
        x = t0 + np.array([0.0, a * D, D]); F = np.array([0.0, v["peak"], 0.0])
    elif kind in ("rectangle", "trapezoid"):
        D = v["duration"]; r = float(v["rise"] or 0.0); fl = r if kind == "rectangle" or v.get("fall") is None else float(v["fall"])
        if r + fl > D:
            raise SpecError("rise + fall is longer than the pulse", where, ln, f)
        e = max(D * 1e-6, 1e-12)
        r = r or e; fl = fl or e
        x = t0 + np.array([0.0, r, D - fl, D]); F = np.array([0.0, v["peak"], v["peak"], 0.0])
    elif kind == "ramp":
        x = t0 + np.array([0.0, v["duration"]]); F = np.array([v["from"], v["to"]])
    elif kind == "step":
        e = max(abs(T) * 1e-7, 1e-12); x = t0 + np.array([0.0, e]); F = np.array([v["from"], v["to"]])
    elif kind in ("sine", "square_wave"):
        stop = float(v["stop"]) if v.get("stop") is not None else max(T, t0)
        dur = max(stop - t0, 0.0); fr = v["frequency"]
        if kind == "sine":
            n = int(min(400000, max(200, 60 * fr * dur)))
            t = np.linspace(0, dur, n + 1); y = v["offset"] + v["amplitude"] * np.sin(2 * np.pi * fr * t + v["phase"])
        else:
            per = 1.0 / fr; duty = min(max(float(v["duty"]), 0.01), 0.99); e = per * 1e-4; ts = []; ys = []
            k = 0
            while k * per < dur and len(ts) < 400000:
                a = k * per; b = a + duty * per
                ts += [a, a + e, b, b + e]; ys += [-1, 1, 1, -1]; k += 1
            t = np.array(ts); y = v["offset"] + v["amplitude"] * np.array(ys, float)
            keep = t <= dur; t, y = t[keep], y[keep]
            if len(t) and t[-1] < dur:                     # hold the level the wave is at until the stop
                t = np.append(t, dur); y = np.append(y, y[-1])
        x, F = t0 + t, y
    elif kind == "sweep":
        D = v["duration"]; f0, f1 = v["f_start"], v["f_end"]
        n = int(min(400000, max(400, 60 * max(f0, f1) * D)))
        t = np.linspace(0, D, n + 1); ph = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / D * t * t)
        x, F = t0 + t, v["offset"] + v["amplitude"] * np.sin(ph)
    else:  # decay
        tau = v["tau"]; t = np.linspace(0, 8 * tau, N + 1); x, F = t0 + t, v["peak"] * np.exp(-t / tau)
    x = np.asarray(x, float); F = np.asarray(F, float)
    # before the start the force is 0 (or the ramp/step's from, or the wave's offset); after: 0 except ramp, step, decay
    pre = {"ramp": v.get("from", 0.0), "step": v.get("from", 0.0)}.get(kind, v.get("offset", 0.0) if kind in ("sine", "square_wave", "sweep") else 0.0)
    if t0 > 0:
        x = np.concatenate([[0.0, t0 - max(t0 * 1e-9, 1e-12)], x]); F = np.concatenate([[pre, pre], F])
    if kind in ("half_sine", "haversine", "triangle", "rectangle", "trapezoid", "sine", "square_wave", "sweep"):
        end = 0.0 if kind not in ("sine", "square_wave", "sweep") else v.get("offset", 0.0)
        x = np.concatenate([x, [x[-1] + max(abs(x[-1]) * 1e-9, 1e-12)]]); F = np.concatenate([F, [end]])
    # strictly increasing x
    keep = np.concatenate([[True], np.diff(x) > 0]); x, F = x[keep], F[keep]
    return x, F


def compile_pulse(e, d, ctx, where, doc, P, tab_k):
    x, F = pulse_table(d, ctx, where, doc)
    P[e, 0] = 0.0; P[e, 1] = 0.0; P[e, 2] = 0.0; tab_k[e] = 0.0
    sp = d["pulse"]; kind = str(sp.get("kind", "")).lower()
    from .model import resolve, num
    def val(k):
        return num(resolve(sp[k], ctx, where), where) if k in sp else None
    if kind in ("sine", "square_wave"):
        feat = 1.0 / val("frequency") / 40.0
    elif kind == "sweep":
        feat = 1.0 / max(val("f_start"), val("f_end")) / 40.0
    elif kind == "decay":
        feat = val("tau") / 20.0
    elif kind == "step":
        feat = 0.0
    else:
        feat = val("duration") / 50.0
        r = val("rise") if kind in ("rectangle", "trapezoid") else None
        if r:
            feat = min(feat, r / 10.0)
    P[e, 3] = feat                     # the time step should resolve the shape (see _run_settings)
    return 13, np.ascontiguousarray(x), np.ascontiguousarray(F)
