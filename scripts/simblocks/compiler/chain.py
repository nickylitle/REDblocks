"""Chained sims: a model with named ports is a block; models wired together run in order, upstream first.

A model declares its ports (they carry no value of their own: whoever uses the model supplies them):

    ports:
      inputs:
        v_in:  {unit: m/s, default: 2 m/s}       # a value: used as $v_in anywhere a number goes
        F_hit: {kind: series, unit: N}           # a time series: a force table, csv: {series: $F_hit}
      outputs:
        v_out: {output: v}                       # a value: one of this model's outputs
        F_out: {element: s, quantity: force}     # a time series: any signal an export column can name

A default lets the model run on its own; an input without one must be supplied by the model that uses it.

Another model uses it as a block:

    models:
      hit:   {file: hit.yaml, inputs: {v_in: $v0}}
      frame: {file: frame.yaml, inputs: {F_hit: $hit.F_out}}

Each block runs before this model compiles, in wire order (a block that reads $hit.x runs after hit). Its value
ports and its outputs are then numbers here: $hit.v_out (also inside an "=" expression); its series ports feed
force tables: csv: {series: $hit.F_out}. Results of every block are listed with this model's own (hit.v_out ...).
A model can be only blocks (no bodies of its own): that is a sequence. A block can run several times in a row,
each run starting from the last one's ending state (repeated hits):

    models:
      hits: {file: hit.yaml, repeat: 5, inputs: {set_in: 0}, carry: {set_in: set_out, x_in: set_out}}

Results list the last run as hits.peak and every run as hits.peak@1 ... hits.peak@5. Studies work across the chain: a sweep or
Monte Carlo on this model's values reruns each block whose inputs changed. Blocks are one-way (the downstream
model cannot push back on the upstream one); each keeps its own time step and dimension.

Values off a CSV file (measured data, another tool's output):

    params:
      v_in: {from_csv: out/csv/hit_state.csv, column: "arm velocity", pick: final}

from_csv keys
  from_csv   the CSV path (relative to this model's folder)
  column     header text (with or without its '[unit]'), or a number counting from 1. Default the last column.
  pick       final (default), first, max, min, max_abs, mean, or at (with at: a time, read off the first column)
  at         the time for pick: at, with a unit ('5 ms'); the first column's header unit is the CSV's time unit
  unit       the column's unit, when the header has none ('[unit]' in the header is used otherwise)

Force tables (csv: on a force element) read the same headers: a column picked by its header text gives its own
'[unit]', so x_unit and y_unit can be left out for a file written by an export."""
import os
import re

import numpy as np

from .loader import SpecError
from .units import unit_factor, UnitError

FROM_CSV_KEYS = {"from_csv", "column", "pick", "at", "unit"}
PICKS = {"final", "first", "max", "min", "max_abs", "mean", "at"}
BASE = [None]                     # the folder relative CSV paths are read from (set while a model compiles)
_HDR = re.compile(r"^\s*(.*?)\s*\[([^\]]*)\]\s*$")


def split_header(h):
    """'Fz [kN]' -> ('Fz', 'kN'); 'Fz' -> ('Fz', None)."""
    m = _HDR.match(str(h))
    return (m.group(1), m.group(2).strip() or None) if m else (str(h).strip(), None)


def find_column(header, c):
    """Index of column c (header text with or without its unit, or a number from 1) or None."""
    if c is None:
        return None
    if isinstance(c, (int, float)) or (isinstance(c, str) and c.strip().isdigit()):
        return int(c) - 1
    c = str(c).strip()
    if not header:
        return None
    if c in header:
        return header.index(c)
    labels = [split_header(h)[0] for h in header]
    if c in labels:
        return labels.index(c)
    low = [x.lower() for x in labels]
    return low.index(c.lower()) if c.lower() in low else None


def model_dir(doc, model_path=None):
    f = model_path or doc.get("__file__", "")
    if f and not str(f).startswith("<"):
        return os.path.dirname(os.path.abspath(f))
    return os.environ.get("SIMBLOCKS_MODEL_DIR") or os.getcwd()


def _abs(p, base):
    p = os.path.expanduser(str(p))
    return p if os.path.isabs(p) else os.path.join(base or os.getcwd(), p)


def read_table(path, where=""):
    """(header list or None, rows as a 2D float array) for a CSV with an optional header row and # comments."""
    from .loadtable import _read_rows
    return _read_rows(path, where)


def csv_value(spec, where):
    """The number a from_csv spec picks, in SI."""
    bad = set(spec) - FROM_CSV_KEYS
    if bad:
        raise SpecError("unknown from_csv key(s): " + ", ".join(sorted(bad)) + " (allowed: "
                        + ", ".join(sorted(FROM_CSV_KEYS)) + ")", where)
    path = _abs(spec["from_csv"], BASE[0])
    header, data = read_table(path, where)
    if data.shape[0] == 0:
        raise SpecError(f"the CSV '{os.path.basename(path)}' has no rows", where)
    c = spec.get("column")
    j = data.shape[1] - 1 if c is None else find_column(header, c)
    if j is None or not 0 <= j < data.shape[1]:
        raise SpecError(f"column '{c}' is not in '{os.path.basename(path)}'"
                        + (f" (columns: {', '.join(header)})" if header else " (it has no header row)"), where)
    pick = str(spec.get("pick", "final")).strip().lower()
    if pick not in PICKS:
        raise SpecError(f"pick is one of {', '.join(sorted(PICKS))}", where)
    y = data[:, j]
    if pick == "at":
        if "at" not in spec:
            raise SpecError("pick: at needs at: a time (e.g. at: 5 ms)", where)
        from .units import parse_quantity
        t_si = parse_quantity(str(spec["at"])) if isinstance(spec["at"], str) else float(spec["at"])
        tu = split_header(header[0])[1] if header else None
        t = data[:, 0] * (unit_factor(tu) if tu else 1.0)
        v = float(np.interp(t_si, t, y))
    else:
        v = float({"final": lambda a: a[-1], "first": lambda a: a[0], "max": np.max, "min": np.min,
                   "max_abs": lambda a: a[np.argmax(np.abs(a))], "mean": np.mean}[pick](y))
    u = spec.get("unit") or (split_header(header[j])[1] if header else None)
    if u:
        try:
            v *= unit_factor(str(u))
        except UnitError as exc:
            raise SpecError(f"from_csv unit: {exc}", where)
    return v


# ---------------------------------------------------------------------------------------------- ports and blocks
PORT_IN_KEYS = {"kind", "unit", "default", "label", "note"}
PORT_OUT_KEYS = {"output", "element", "body", "part", "control", "quantity", "unit", "label", "note"}
MODEL_KEYS = {"file", "inputs", "repeat", "carry", "label", "note"}
_RUNS = {}                        # (path, mtime, inputs) -> (values, series, outputs): reruns only what changed


class Series:
    """A time series passed between models: t in s, y in SI, kind of quantity ('force', 'torque', ...)."""
    def __init__(self, t, y, unit="", label=""):
        self.t = np.ascontiguousarray(t, float); self.y = np.ascontiguousarray(y, float)
        self.unit = unit; self.label = label

    def __repr__(self):
        return f"<series {self.label} {len(self.t)} points>"

    def key(self):
        return (len(self.t), float(self.t[-1]) if len(self.t) else 0.0, float(np.sum(self.y)), float(np.sum(self.y * self.y)))


def port_spec(doc):
    p = doc.get("ports") or {}
    return p.get("inputs") or {}, p.get("outputs") or {}


def check_ports(doc):
    from .loader import line_of
    f = doc.get("__file__", "")
    p = doc.get("ports")
    if p is None:
        return
    if not isinstance(p, dict) or set(p) - {"inputs", "outputs"}:
        raise SpecError("ports has inputs: and outputs:", "ports", line_of(doc, "ports"), f)
    ins, outs = port_spec(doc)
    params = set(doc.get("params") or {}) | set(doc.get("derived") or {})
    for k, v in ins.items():
        w = f"ports.inputs.{k}"
        if not isinstance(v, dict):
            raise SpecError("an input port is {unit: ..., default: ...} or {kind: series, unit: N}", w, line_of(ins, k), f)
        bad = set(v) - PORT_IN_KEYS
        if bad:
            raise SpecError(f"unknown input port key(s): {', '.join(sorted(bad))} (allowed: {', '.join(sorted(PORT_IN_KEYS))})",
                            w, line_of(ins, k), f)
        if str(v.get("kind", "value")) not in ("value", "series"):
            raise SpecError("an input port's kind is value or series", w, line_of(ins, k), f)
        if k in params:
            raise SpecError(f"'{k}' is both an input port and a param: an input port is a param with no value", w,
                            line_of(ins, k), f)
    outn = set(doc.get("outputs") or {})
    for k, v in outs.items():
        w = f"ports.outputs.{k}"
        if not isinstance(v, dict):
            raise SpecError("an output port is {output: name} or {element: e, quantity: force}", w, line_of(outs, k), f)
        bad = set(v) - PORT_OUT_KEYS
        if bad:
            raise SpecError(f"unknown output port key(s): {', '.join(sorted(bad))}", w, line_of(outs, k), f)
        if "output" in v and v["output"] not in outn:
            raise SpecError(f"output '{v['output']}' is not one of this model's outputs", w, line_of(outs, k), f)
        if "output" not in v and not any(x in v for x in ("element", "body", "part", "control")):
            raise SpecError("an output port names an output (a value) or an element, body, part or control (a series)",
                            w, line_of(outs, k), f)


def input_values(doc, ctx, overrides, where_file=""):
    """Each input port's value into ctx: the override (the using model's wire), else the default."""
    from .model import resolve
    ins, _ = port_spec(doc)
    for k, v in ins.items():
        if k in overrides:
            val = overrides.pop(k)
            if isinstance(val, str):
                val = resolve(val, ctx, f"ports.inputs.{k}")
        elif "default" in v:
            val = resolve(v["default"], ctx, f"ports.inputs.{k}.default")
        else:
            raise SpecError(f"input port '{k}' has no value, so this model runs only as a block inside another "
                            "model (wire the port there), or give the port a default to run it on its own",
                            f"ports.inputs.{k}")
        if str(v.get("kind", "value")) == "series" and not isinstance(val, Series):
            raise SpecError(f"input '{k}' is a time series: wire a series port to it (got {val!r})", f"ports.inputs.{k}")
        ctx[k] = val


def _deps(models):
    out = {}
    for m, spec in models.items():
        txt = repr((spec or {}).get("inputs") or {})
        out[m] = [o for o in models if o != m and re.search(r"\$" + re.escape(o) + r"\b", txt)]
    return out


def order(models):
    """Block names upstream first; a loop is an error."""
    deps = _deps(models); seen, done, out = set(), set(), []

    def visit(m, path):
        if m in done:
            return
        if m in seen:
            raise SpecError("models make a loop: " + " -> ".join(path + [m]), "models")
        seen.add(m)
        for d in deps[m]:
            visit(d, path + [m])
        done.add(m); out.append(m)
    for m in models:
        visit(m, [])
    return out


def run_block(path, inputs):
    """Run a model file with its input ports set. Returns (port values, port series, outputs), cached by inputs."""
    from .loader import load, check
    from .model import compile_model, run_model
    from .units import unit_factor as uf
    key = (path, os.path.getmtime(path),
           tuple(sorted((k, v.key() if isinstance(v, Series) else repr(v)) for k, v in inputs.items())))
    if key in _RUNS:
        return _RUNS[key]
    doc = load(path); check(doc)
    ins, outs = port_spec(doc)
    for k in inputs:
        if k not in ins:
            raise SpecError(f"'{os.path.basename(path)}' has no input port '{k}' (its inputs: "
                            f"{', '.join(ins) or 'none'})", "models")
    saved = BASE[0]
    try:
        M = compile_model(doc, dict(inputs))
        R, H = run_model(M, history=True, n_log=max(2000, min(20000, int(round(M.t_end / M.dt)) + 1)))
    finally:
        BASE[0] = saved
    vals, ser = {}, {}
    from .exports import column
    for k, o in outs.items():
        if "output" in o:
            od = doc["outputs"][o["output"]]
            v = R[o["output"]]
            vals[k] = float(v) * (uf(od["unit"]) if od.get("unit") else 1.0)      # back to SI
        else:
            c = {kk: vv for kk, vv in o.items() if kk not in ("unit", "label", "note")}
            y, lab, u = column(doc, H, dict(c, unit=""))
            ser[k] = Series(H["t"], y, o.get("unit") or "", lab)
    outv = {k: v for k, v in R.items() if not k.startswith("_") and isinstance(v, (int, float))}
    _RUNS[key] = (vals, ser, outv)
    if len(_RUNS) > 200:
        _RUNS.pop(next(iter(_RUNS)))
    return _RUNS[key]


def run_models(doc, ctx):
    """Run every block in doc's models: in wire order, each result into ctx[name] (ports and outputs)."""
    from .loader import line_of
    from .model import resolve
    models = doc.get("models") or {}
    if not models:
        return
    f = doc.get("__file__", "")
    base = model_dir(doc)
    for m in order(models):
        spec = models[m] or {}; w = f"models.{m}"
        if not isinstance(spec, dict) or "file" not in spec:
            raise SpecError("a model block is {file: other.yaml, inputs: {port: value}}", w, line_of(models, m), f)
        bad = set(spec) - MODEL_KEYS
        if bad:
            raise SpecError(f"unknown model block key(s): {', '.join(sorted(bad))}", w, line_of(models, m), f)
        p = os.path.normpath(_abs(spec["file"], base))
        if f and not str(f).startswith("<") and os.path.normpath(os.path.abspath(f)) == p:
            raise SpecError("a model cannot be a block inside itself", w, line_of(models, m), f)
        if not os.path.exists(p):
            raise SpecError(f"model file '{spec['file']}' does not exist (looked in {base})", w, line_of(models, m), f)
        inp = {k: resolve(v, ctx, f"{w}.inputs.{k}") for k, v in (spec.get("inputs") or {}).items()}
        n = spec.get("repeat", 1)
        n = int(round(float(resolve(n, ctx, f"{w}.repeat")))) if not isinstance(n, int) else n
        if not 1 <= n <= 500:
            raise SpecError("repeat is how many times the block runs in a row, 1 to 500", w, line_of(spec, "repeat"), f)
        carry = spec.get("carry") or {}
        if not isinstance(carry, dict):
            raise SpecError("carry maps an input port to the output port that feeds it on the next run: "
                            "{set_in: set_out}", w, line_of(spec, "carry"), f)
        hits = []
        for i in range(n):
            try:
                vals, ser, outv = run_block(p, inp)
            except SpecError as exc:
                raise SpecError(f"in block '{m}' ({os.path.basename(p)})" + (f", run {i + 1}" if n > 1 else "") +
                                f": {exc}", w, line_of(models, m), f)
            hits.append(dict(outv, **vals))
            src = dict(vals, **ser)
            for ip, op in carry.items():           # this run's ending state starts the next one (another hit)
                if op not in src:
                    raise SpecError(f"carry: '{op}' is not an output port of {os.path.basename(p)} (its ports: "
                                    f"{', '.join(src) or 'none'})", w, line_of(spec, "carry"), f)
                inp[ip] = src[op]
        ctx[m] = dict(outv, **vals, **ser)
        if n > 1:
            ctx["__hits__" + m] = hits


def block_results(M):
    """Every block's outputs and value ports, named block.output, for this model's results table."""
    out = {}
    for m in (M.doc.get("models") or {}):
        for k, v in (M.ctx.get(m) or {}).items():
            if isinstance(v, (int, float)):
                out[f"{m}.{k}"] = float(v)
        for i, h in enumerate(M.ctx.get("__hits__" + m) or [], start=1):     # a repeated block: every run
            for k, v in h.items():
                if isinstance(v, (int, float)):
                    out[f"{m}.{k}@{i}"] = float(v)
    return out


def describe(path):
    """A model file's ports, for drawing it as a block: {name, dim, inputs: [...], outputs: [...]}."""
    from .loader import load
    doc = load(path)
    ins, outs = port_spec(doc)
    return dict(name=doc.get("name", os.path.basename(path)), dim=doc.get("dim", "1d"),
                inputs=[dict(name=k, kind=str(v.get("kind", "value")), unit=v.get("unit", ""),
                             default=None if v.get("default") is None else str(v.get("default")),
                             label=v.get("label", "")) for k, v in ins.items()],
                outputs=[dict(name=k, kind="value" if "output" in v else "series", unit=v.get("unit", ""),
                              label=v.get("label", "")) for k, v in outs.items()],
                results=list(doc.get("outputs") or {}))
