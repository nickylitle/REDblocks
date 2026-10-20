"""REDblocks as an MCP server: any MCP client (Claude desktop, Claude Code, or another assistant that speaks the Model
Context Protocol) can read, edit, run, sweep, verify and plot models on this computer.

    redblocks-mcp                 # stdio server (what desktop clients launch)
    python -m simblocks mcp       # the same

Every tool works on a model file (.yaml). Edits are checked by the compiler before they are written, and the file
before each edit is kept as <name>.yaml.bak, so a bad edit is one rename away from undone. The editor reloads a file
it has open when the file changes on disk, so you can watch an assistant work on the canvas.

The tools are plain functions (tool_*) so they can be tested without an MCP client; main() registers them."""
import io
import json
import math
import os
import re
import shutil

import numpy as np

INSTRUCTIONS = """REDblocks runs explicit-dynamics sims built from chained closed-form blocks: 1D/2D/3D rigid
bodies, springs, contacts, plates and beams from Roark, plasticity from real stress-strain curves, motors, circuits.
A model is one YAML file. Workflow: read_model (or list_examples) -> format_reference for the syntax you need ->
edit_model / write_model (the compiler checks every edit; invalid edits are refused) -> run_model -> sweep / verify /
plot. A run that may take over ~45 s (check_model gives estimated_run_s): start_job, then job_status until done.
The run watchdog is adjustable per call (timeout: "none" turns it off). Values carry units ("3 mm", "12 V"). Never invent failure criteria: failure modes come from the user.
Outputs are in SI unless a unit is given in the model. Tell the user which file you changed."""


# ------------------------------------------------------------------ helpers
def _root():
    return os.path.abspath(os.environ.get("REDBLOCKS_WORKDIR") or os.getcwd())


def _path(p):
    p = os.path.expanduser(str(p).strip().strip('"'))
    if not os.path.isabs(p):
        p = os.path.join(_root(), p)
    return os.path.abspath(p)


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _model_env(path):
    """Relative CSV tables and model blocks resolve next to the model file."""
    os.environ["SIMBLOCKS_MODEL_DIR"] = os.path.dirname(path)


def _load(text, path=None):
    from .compiler import load_system
    if path:
        _model_env(path)
    return load_system(text)


def _err(exc):
    return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _num(v):
    try:
        f = float(v)
        return None if math.isnan(f) else (round(f, 6 - int(math.floor(math.log10(abs(f))))) if f else 0.0)
    except Exception:
        return v if isinstance(v, (str, bool, type(None))) else str(v)


def _check_text(text, path=None):
    """Compile only: ok, or the error with its line. Also the step, step count and estimated run time."""
    try:
        s = _load(text, path)
        M = s.nominal
        from .compiler.model import estimate_run_s
        return {"ok": True, "dt_s": _num(M.dt), "steps": int(round(M.t_end / M.dt)), "t_end_s": _num(M.t_end),
                "estimated_run_s": _num(estimate_run_s(M)), "outputs": list((s.doc.get("outputs") or {}).keys())}
    except Exception as exc:
        return _err(exc)


def _summary(doc):
    c = doc.get("circuit") or {}
    return {"name": doc.get("name"), "dim": doc.get("dim", "1d"), "params": doc.get("params") or {},
            "bodies": list((doc.get("bodies") or {}).keys()), "elements": {k: (v or {}).get("type") for k, v in (doc.get("elements") or {}).items()},
            "circuit_parts": {k: (v or {}).get("type") for k, v in (c.get("parts") or {}).items()},
            "outputs": list((doc.get("outputs") or {}).keys()), "failure_modes": list((doc.get("failure") or {}).keys()),
            "run": doc.get("run") or {}}


def _write_checked(path, text, allow_invalid=False):
    chk = _check_text(text, path)
    if not chk["ok"] and not allow_invalid:
        return {"ok": False, "written": False, "error": chk["error"], "hint": "nothing was written; fix the YAML and try again"}
    if os.path.exists(path):
        shutil.copy2(path, path + ".bak")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return dict(chk, written=True, path=path, backup=(path + ".bak") if os.path.exists(path + ".bak") else None)


def _tmo(timeout):
    """The run watchdog: None or "model" -> the model's own run.timeout (AUTO by default); "none", "off", 0 or
    False -> no limit; a number or "90 s" / "5 min" -> that many seconds of wall-clock time."""
    if timeout is None or timeout == "model":
        return "model"
    if timeout is False or str(timeout).strip().lower() in ("none", "off", "0", "no", "false", "unlimited"):
        return None
    if isinstance(timeout, (int, float)):
        return float(timeout) if timeout > 0 else None
    from .compiler.units import parse_quantity
    try:
        return float(parse_quantity(str(timeout)))
    except Exception:
        return float(str(timeout).split()[0])


def _hist_stats(t, v):
    v = np.asarray(v, float)
    if not len(v):
        return {}
    i_max, i_min = int(np.nanargmax(v)), int(np.nanargmin(v))
    return {"final": _num(v[-1]), "max": _num(v[i_max]), "t_at_max": _num(t[i_max]), "min": _num(v[i_min]),
            "t_at_min": _num(t[i_min]), "mean": _num(np.nanmean(v))}


# ------------------------------------------------------------------ tools
def tool_list_examples():
    """The bundled example models (1D, 2D, 3D, circuits, the Apollo 11 showcase): name and file path. Copy one with
    write_model to start a new model."""
    from .app.api import Api
    return [{"name": e["name"], "path": os.path.join(e["dir"], e["file"])} for e in Api().examples()]


def tool_read_model(path: str):
    """The model's YAML text and a summary (params, bodies, elements, circuit parts, outputs, failure modes, run)."""
    try:
        p = _path(path)
        text = _read(p)
        from .compiler.loader import load
        return {"ok": True, "path": p, "yaml": text, "summary": _summary(load(text))}
    except Exception as exc:
        return _err(exc)


def tool_check_model(path: str | None = None, yaml: str | None = None):
    """Compile without running: ok or the error and its line, the time step, step count and estimated run time.
    Give a path, or YAML text to check before writing it."""
    if yaml is None and path is None:
        return {"ok": False, "error": "give path or yaml"}
    p = _path(path) if path else None
    return _check_text(yaml if yaml is not None else _read(p), p)


def tool_write_model(path: str, yaml: str, allow_invalid: bool = False):
    """Write a whole model file. The compiler checks it first: an invalid model is refused unless allow_invalid.
    An existing file is kept as <path>.bak."""
    p = _path(path)
    if not p.lower().endswith((".yaml", ".yml")):
        return {"ok": False, "error": "model files end in .yaml or .yml"}
    return _write_checked(p, yaml, allow_invalid)


def tool_edit_model(path: str, old: str, new: str):
    """Replace one exact piece of text in the model file (old must occur exactly once). The result is checked by
    the compiler; an edit that breaks the model is refused and nothing is written. The previous file is kept as .bak."""
    try:
        p = _path(path)
        text = _read(p)
        n = text.count(old)
        if n != 1:
            return {"ok": False, "error": f"'old' occurs {n} times; it must occur exactly once (add surrounding lines)"}
        return _write_checked(p, text.replace(old, new, 1))
    except Exception as exc:
        return _err(exc)


def tool_run_model(path: str, overrides: dict | None = None, histories: list[str] | None = None, points: int = 0,
                   timeout: str | float | None = None):
    """Run the model once. overrides sets params for this run only, e.g. {"thickness": "3 mm"}. Returns every output,
    the energy ledger, and for each history named in histories (e.g. ["hit.force", "wedge.velocity"]; [] for none,
    ["*"] for all) its final, max, time of max, min and mean. points > 0 also returns that many evenly spaced samples.
    timeout is the run watchdog: omit for the model's own (AUTO), "none" to turn it off, or seconds ("10 min").
    A run longer than about 50 s can outlast the connection: use start_job for those (check_model gives the estimate)."""
    try:
        p = _path(path)
        s = _load(_read(p), p)
        out, H = s.run(overrides or None, history=True, timeout=_tmo(timeout))
        res = {"ok": True, "outputs": {k: _num(v) for k, v in out.items() if not k.startswith("_")},
               "checks": {k: _num(v) for k, v in out.items() if k.startswith("_")}}
        t = np.asarray(H.get("t", []), float)
        parts = ((s.doc.get("circuit") or {}).get("parts") or {})
        ONLY = {"soc": ("battery",), "torque": ("motor",), "winding_current": ("motor",)}

        def useful(k):                                  # a circuit part's quantities that do not apply to its type
            nm, _, q = k.partition(".")
            d = parts.get(nm)
            if not isinstance(d, dict):
                return True
            if q in ONLY:
                return d.get("type") in ONLY[q]
            if q == "temperature":
                return "thermal" in d
            if q.startswith("esc_"):
                return isinstance(d.get("esc"), dict)
            return True
        types = {k: (e or {}).get("type") for k, e in (s.doc.get("elements") or {}).items()}

        def live(k):                                    # an element history that never leaves 0 does not apply to it
            nm = k.partition(".")[0]
            return nm not in types or k.endswith((".force", ".measure")) or np.any(np.asarray(H[k], float) != 0.0)
        keys = [k for k in H if k != "t" and "__" not in k.partition(".")[0] and useful(k) and live(k)]
        res["history_keys"] = keys if len(keys) <= 200 else keys[:200] + ["..."]
        want = keys if histories == ["*"] else [k for k in (histories or []) if k in H]
        miss = [k for k in (histories or []) if k != "*" and k not in H]
        if want:
            res["histories"] = {k: _hist_stats(t, H[k]) for k in want}
            if points and int(points) > 0:
                idx = np.linspace(0, len(t) - 1, min(int(points), len(t))).astype(int)
                res["samples"] = {"t": [_num(x) for x in t[idx]], **{k: [_num(x) for x in np.asarray(H[k])[idx]] for k in want}}
        if miss:
            res["unknown_histories"] = miss
        return res
    except Exception as exc:
        return _err(exc)


def tool_sweep(path: str, param: str, values: list[str], outputs: list[str] | None = None,
               timeout: str | float | None = None):
    """Run the model once per value of one param (e.g. param "thickness", values ["2 mm", "3 mm", "4 mm"]) and return
    a table of the outputs (all, or the ones named in outputs). timeout applies to each run (see run_model)."""
    try:
        p = _path(path)
        s = _load(_read(p), p)
        rows = []
        for v in values:
            try:
                o = s.run({param: v}, timeout=_tmo(timeout))
                rows.append({param: v, **{k: _num(x) for k, x in o.items() if not k.startswith("_") and (not outputs or k in outputs)}})
            except Exception as exc:
                rows.append({param: v, "error": str(exc)})
        return {"ok": True, "rows": rows}
    except Exception as exc:
        return _err(exc)


def tool_verify_model(path: str, mc_n: int | None = None):
    """Full verification: time-step convergence, knockouts (does each element matter), Monte Carlo over the
    uncertain values, energy ledger, failure-mode safety factors. Slower than a run. Returns the report as markdown."""
    try:
        p = _path(path)
        rep = _load(_read(p), p).verify(int(mc_n) if mc_n else None)
        return {"ok": True, "verified": bool(rep.ok), "report": rep.markdown()}
    except Exception as exc:
        return _err(exc)


def plot_png(path, keys, overrides=None):
    """PNG bytes of the named histories against time (one panel per unit group)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = _path(path)
    s = _load(_read(p), p)
    _, H = s.run(overrides or None, history=True)
    keys = [k for k in keys if k in H]
    if not keys:
        raise ValueError("none of those histories exist; run_model lists history_keys")
    t = np.asarray(H["t"], float)
    fig, ax = plt.subplots(len(keys), 1, figsize=(8, 2.2 * len(keys) + 0.6), sharex=True, squeeze=False)
    fig.patch.set_facecolor("#121214")
    for a, k in zip(ax[:, 0], keys):
        a.set_facecolor("#18181b"); a.plot(t, H[k], color="#e5383b", lw=1.4)
        a.set_ylabel(k, color="#f1eeee", fontsize=9); a.tick_params(colors="#bdb6b6", labelsize=8)
        a.grid(color="#2a2a2f", lw=0.6)
        for sp in a.spines.values():
            sp.set_color("#3a3a41")
    ax[-1, 0].set_xlabel("time (s)", color="#f1eeee")
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, facecolor=fig.get_facecolor()); plt.close(fig)
    return buf.getvalue()


def tool_find_blocks(query: str = "", limit: int = 25):
    """Search the block catalog (Roark beams and plates, contact, plasticity, motors, aero, fatigue ...) by words.
    Returns name, domain, kind and summary; use block_info for parameters and a usage snippet."""
    from .docs.library import entries
    q = [w for w in re.split(r"\W+", query.lower()) if w]
    hits = []
    for e in entries():
        hay = " ".join([e["name"], e["domain"], e["category"], e["summary"], " ".join(e.get("tags") or [])]).lower()
        score = sum(hay.count(w) for w in q) if q else 1
        if score and not e.get("legacy"):
            hits.append((score, {"name": e["name"], "domain": e["domain"], "kind": e["kind"], "summary": e["summary"]}))
    hits.sort(key=lambda h: -h[0])
    return [h[1] for h in hits[:int(limit)]]


def tool_block_info(name: str):
    """One block's documentation: parameters, outputs, source (the handbook it comes from) and a usage snippet."""
    from .docs.library import entries
    for e in entries():
        if e["name"] == name:
            return {k: e.get(k) for k in ("name", "domain", "category", "kind", "summary", "source", "params", "outputs",
                                         "out_keys", "usage", "doc")}
    return {"ok": False, "error": f"no block named {name}; try find_blocks"}


def tool_find_materials(query: str = ""):
    """Materials in the catalog with E, yield, ultimate and density (SI). Use the key in a model's material: field."""
    from .app.api import Api
    q = query.lower().strip()
    rows = Api().help_materials().get("materials", [])
    return [{k: _num(r.get(k)) if k not in ("key", "name", "family") else r.get(k) for k in ("key", "name", "family", "E", "sy", "su", "rho")}
            for r in rows if not q or q in (r["key"] + " " + r.get("name", "") + " " + r.get("family", "")).lower()][:60]


def _reference_text():
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.path.join(here, "docs", "system_file.md"),
              os.path.abspath(os.path.join(here, "..", "..", "references", "system_file.md"))):
        if os.path.exists(c):
            return _read(c)
    return ""


def tool_format_reference(section: str = ""):
    """The YAML model format reference. With no section: the list of section headings. With a section word (e.g.
    "elements", "circuit", "outputs", "2d", "failure", "exports"): that section's text."""
    text = _reference_text()
    if not text:
        return {"ok": False, "error": "the format reference is not installed with this copy"}
    parts = re.split(r"(?m)^(#{2,3} .*)$", text)
    secs = [(parts[i].lstrip("# ").strip(), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]
    if not section:
        return {"ok": True, "sections": [h for h, _ in secs]}
    w = section.lower()
    got = [f"## {h}\n{b}" for h, b in secs if w in h.lower()]
    return {"ok": bool(got), "text": "\n".join(got)[:20000] if got else f"no section matches '{section}'"}


def _request_file():
    return os.path.join(os.path.expanduser("~"), ".simblocks", "open_request.json")


def tool_open_in_editor(path: str):
    """Open a model in the REDblocks editor window on this computer, so the user can watch it (the editor must be
    running). If the editor has unsaved edits it asks instead of replacing them."""
    import time
    p = _path(path)
    if not os.path.exists(p):
        return {"ok": False, "error": f"no file at {p}"}
    f = _request_file()
    os.makedirs(os.path.dirname(f), exist_ok=True)
    with open(f, "w", encoding="utf-8") as fh:
        json.dump({"path": p, "t": time.time()}, fh)
    for _ in range(20):                                   # the editor deletes the request when it picks it up
        time.sleep(0.25)
        if not os.path.exists(f):
            return {"ok": True, "opened": p}
    return {"ok": True, "queued": p, "note": "REDblocks did not pick it up within 5 s; it opens when the editor next runs"}


# ------------------------------------------------------------------ background jobs (long runs)
JOBS = {}


def _job_view(j, with_result=True):
    import time
    v = {"job_id": j["id"], "kind": j["kind"], "path": j["path"], "state": j["state"],
         "elapsed_s": round((j.get("t1") or time.time()) - j["t0"], 1), "estimated_run_s": j.get("est")}
    if with_result and j["state"] in ("done", "error", "cancelled"):
        v["result"] = j.get("result")
    return v


def tool_start_job(kind: str, path: str, overrides: dict | None = None, histories: list[str] | None = None,
                   points: int = 0, param: str | None = None, values: list[str] | None = None,
                   outputs: list[str] | None = None, mc_n: int | None = None, timeout: str | float | None = "none"):
    """Start a long run in the background and return at once with a job_id. kind is "run" (same arguments as
    run_model), "sweep" (param, values, outputs) or "verify" (mc_n). The watchdog is off by default here
    (timeout "none"); give seconds to cap it. Then call job_status(job_id) until it says done."""
    import threading, time, uuid
    p = _path(path)
    if kind not in ("run", "sweep", "verify"):
        return {"ok": False, "error": "kind is run, sweep or verify"}
    chk = _check_text(_read(p), p) if os.path.exists(p) else {"ok": False, "error": f"no file at {p}"}
    if not chk["ok"]:
        return chk
    j = {"id": uuid.uuid4().hex[:8], "kind": kind, "path": p, "state": "running", "t0": time.time(),
         "est": chk.get("estimated_run_s"), "done": threading.Event()}

    def go():
        from .compiler.model import uncancel
        j["thread"] = threading.get_ident()
        try:
            if kind == "run":
                r = tool_run_model(p, overrides, histories, points, timeout)
            elif kind == "sweep":
                r = tool_sweep(p, param or "", values or [], outputs, timeout)
            else:
                r = tool_verify_model(p, mc_n)
            j["result"] = r
            j["state"] = "cancelled" if j.get("cancel") else ("done" if r.get("ok", True) else "error")
        except Exception as exc:
            j["result"] = _err(exc); j["state"] = "error"
        finally:
            j["t1"] = time.time(); uncancel(threading.get_ident()); j["done"].set()
    th = threading.Thread(target=go, daemon=True)
    JOBS[j["id"]] = j
    th.start()
    return {"ok": True, **_job_view(j, False)}


def tool_job_status(job_id: str, wait_s: float = 45):
    """A background job's state (running, done, error, cancelled), how long it has run, and its result once
    finished. Waits up to wait_s seconds (at most 50) for it to finish before answering."""
    j = JOBS.get(job_id)
    if not j:
        return {"ok": False, "error": f"no job {job_id}", "jobs": [_job_view(x, False) for x in JOBS.values()]}
    j["done"].wait(max(0.0, min(float(wait_s or 0), 50.0)))
    return {"ok": True, **_job_view(j)}


def tool_cancel_job(job_id: str):
    """Stop a background job. A run stops at its next check; a sweep or Verify stops all its remaining runs."""
    from .compiler.model import cancel_runs
    j = JOBS.get(job_id)
    if not j:
        return {"ok": False, "error": f"no job {job_id}"}
    j["cancel"] = True
    if j.get("thread"):
        cancel_runs(j["thread"])
    j["done"].wait(5)
    return {"ok": True, **_job_view(j)}


TOOLS = [tool_list_examples, tool_open_in_editor, tool_read_model, tool_check_model, tool_write_model, tool_edit_model, tool_run_model,
         tool_sweep, tool_verify_model, tool_start_job, tool_job_status, tool_cancel_job, tool_find_blocks, tool_block_info,
         tool_find_materials, tool_format_reference]


# ------------------------------------------------------------------ server
def build_server():
    try:                                                  # mcp 2.x
        from mcp.server.mcpserver import MCPServer as Server, Image
    except ImportError:                                   # mcp 1.x
        from mcp.server.fastmcp import FastMCP as Server, Image
    srv = Server("redblocks", instructions=INSTRUCTIONS)
    for f in TOOLS:
        srv.tool(name=f.__name__[5:], description=f.__doc__)(f)

    def plot(path: str, keys: list[str], overrides: dict | None = None):
        return Image(data=plot_png(path, keys, overrides), format="png")
    srv.tool(name="plot", description="Plot named histories against time and return the picture, so you can look at "
                                      "the response (ringing, a spike, a slow drift) and not only at numbers.")(plot)
    return srv


def main(argv=None):
    try:
        srv = build_server()
    except ImportError:
        import sys
        sys.stderr.write("REDblocks MCP needs the mcp package:  pip install \"mcp>=1.2\"\n")
        return 2
    srv.run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
