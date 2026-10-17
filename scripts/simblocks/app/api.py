"""Bridge between the node editor (editor.html) and the simblocks compiler. Pure Python, no GUI imports, so it runs
inside the desktop window (pywebview js_api) and in tests alike. Every method takes and returns plain JSON types.

Nothing leaves the machine: files are read and written where the user picks, reports go next to the model."""
import io
import json
import math
import os
import threading
import time
import traceback

import yaml

ORDER = ["simblocks", "name", "question", "dim", "units", "gravity", "params", "derived", "bodies", "elements", "outputs", "failure",
         "uncertain", "study", "exports", "limits", "knockouts", "run", "report", "assumptions", "sources", "notes", "editor"]
HOME = os.path.join(os.path.expanduser("~"), ".simblocks")
SETTINGS = os.path.join(HOME, "app.json")
GROUPS = os.path.join(HOME, "groups")
ASK_RUNS = 100            # studies with more runs than this (or longer than ASK_SECONDS) ask before starting
ASK_SECONDS = 60.0



def _clean(x):
    """JSON-safe copy (yaml LDicts, numpy, inf/nan)."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items() if not str(k).startswith("__")}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        try:
            x = x.item()
        except Exception:
            pass
    if isinstance(x, float) and not math.isfinite(x):
        return None if math.isnan(x) else ("inf" if x > 0 else "-inf")
    return x


class _Dumper(yaml.SafeDumper):
    pass


def _str(dumper, data):
    style = '"' if data.startswith("=") or data.strip() != data or "#" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


def _short(v):
    """A mapping or list small enough to write on one line, like hand-written files."""
    if isinstance(v, dict):
        return len(v) <= 9 and all(not isinstance(x, (dict, list)) or _short(x) for x in v.values()) and \
            len(json.dumps(v, default=str)) < 115
    if isinstance(v, list):
        return all(not isinstance(x, (dict, list)) for x in v) and len(json.dumps(v, default=str)) < 90
    return True


def _map(dumper, data):
    return dumper.represent_mapping("tag:yaml.org,2002:map", data.items(), flow_style=_short(data))


def _seq(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=_short(data))


_Dumper.add_representer(str, _str)
_Dumper.add_representer(dict, _map)
_Dumper.add_representer(list, _seq)


def _val(v):
    return yaml.dump(v, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=118).rstrip("\n")


def _pair(k, v, ind):
    pad = " " * ind
    if isinstance(v, (dict, list)) and not _short(v):
        if isinstance(v, dict):
            return f"{pad}{k}:\n" + "".join(_pair(kk, vv, ind + 2) for kk, vv in v.items())
        return f"{pad}{k}:\n" + "".join(f"{pad}  - " + _val(x).replace("\n", "\n" + pad + "    ") + "\n" for x in v)
    txt = _val(v)
    if txt.endswith("\n..."):
        txt = txt[:-4].rstrip("\n")
    return f"{pad}{k}: {txt}\n"


def dump_yaml(doc):
    """Top-level sections in the conventional order; one entry per line; small mappings inline."""
    out = io.StringIO()
    for k in ORDER + [k for k in doc if k not in ORDER]:
        if k not in doc or doc[k] in (None, {}, []):
            continue
        v = doc[k]
        if isinstance(v, dict):
            out.write(f"\n{k}:\n")
            for kk, vv in v.items():
                out.write(_pair(kk, vv, 2))
        elif isinstance(v, list):
            out.write(f"\n{k}:\n")
            for x in v:
                out.write("  - " + _val(x).replace("\n", "\n    ") + "\n")
        else:
            out.write(_pair(k, v, 0))
    return out.getvalue()


def _inflate(x):
    """Fields typed as flow YAML in the editor ({circle: $d}, [1, 2]) become real mappings and lists again."""
    if isinstance(x, dict):
        return {k: _inflate(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_inflate(v) for v in x]
    if isinstance(x, str) and x[:1] in "{[" and x[-1:] in "}]":
        try:
            v = yaml.safe_load(x)
            if isinstance(v, (dict, list)):
                return _inflate(v)
        except Exception:
            pass
    return x


def _err(exc):
    from ..compiler.loader import SpecError
    if isinstance(exc, SpecError):
        return dict(ok=False, error=str(exc), path=getattr(exc, "path", ""), line=getattr(exc, "line", None))
    return dict(ok=False, error=f"{type(exc).__name__}: {exc}", path="", line=None,
                trace=traceback.format_exc(limit=4))


class Api:
    def __init__(self, window=None):
        self._window = window
        self._catalog = None
        self._materials = None
        self._blocks = None
        st = self.load_settings() or {}
        if st.get("materials_dir"):
            self._load_materials(st["materials_dir"])
        if st.get("blocks_dir"):
            self._load_blocks(st["blocks_dir"])

    # ------------------------------------------------------------------ catalog
    def catalog(self):
        """Everything the node menu needs: element types and their fields, output quantities, catalog blocks
        (builders and analytic answers, with parameters), material names."""
        if self._catalog is None:
            from ..compiler import loader as L
            from ..docs.library import entries, ELEMENTS, _el_fig
            from ..docs import figures as FG
            from ..materials import _ALL
            blocks = [dict(name=e["name"], domain=e["domain"], category=e["category"], kind=e["kind"],
                           summary=e["summary"], source=e["source"], params=e["params"], outputs=e["outputs"],
                           legacy=e["legacy"], fig=e["fig"], doc=(e["doc"] or "")[:1400], tags=list(e.get("tags") or []))
                      for e in entries()]
            self._catalog = dict(
                # mass (and segments for springs and dampers) is open to every 1D element; the editor hides it in 2D/3D
                element_types={t: dict(fields=sorted(set(f) | ({"mass"} if t not in L.TYPES_2D_ONLY and t != "impact" else set())
                                                     | ({"segments"} if t in ("spring", "dashpot", "kelvin_voigt") else set())
                                                     | ({"mass", "segments"} if t == "beam" else set())),
                                       required=sorted(L.REQUIRED_ELEMENT[t]))
                               for t, f in L.ELEMENT_KEYS.items()},
                element_quantities=sorted(L.ELEMENT_QUANTITIES), body_quantities=sorted(L.BODY_QUANTITIES),
                body_quantities_2d=["x", "y", "angle", "vx", "vy", "omega", "speed", "position", "velocity", "kinetic_energy"],
                body_quantities_3d=["x", "y", "z", "vx", "vy", "vz", "speed", "angle", "omega", "wx", "wy", "wz", "position",
                                    "velocity", "kinetic_energy"],
                types_2d_only=sorted(L.TYPES_2D_ONLY),
                reduces=sorted(L.REDUCES), blocks=blocks,
                materials=sorted((k, getattr(m, "name", k) + (" (custom)" if k in custom_keys() else "")) for k, m in _ALL.items()),
                symbols=FG.symbols() + "".join(_el_fig(e["name"]) for e in ELEMENTS),
                element_info={e["name"]: dict(law=e["law"], fields=e["fields"], what=e["what"]) for e in ELEMENTS},
                unit_factors={k: v for k, v in __import__("simblocks.compiler.units", fromlist=["_U"])._U.items() if v is not None})
        return self._catalog

    # ------------------------------------------------------------------ YAML <-> document
    def import_yaml(self, text):
        try:
            from ..compiler.loader import load
            doc = _clean(load(text))
            return dict(ok=True, doc=doc, mats=_material_params(doc))
        except Exception as exc:
            return _err(exc)

    def export_yaml(self, doc):
        try:
            return dict(ok=True, text=dump_yaml(_inflate(doc)))
        except Exception as exc:
            return _err(exc)

    # ------------------------------------------------------------------ compile, run, verify, report
    def _system(self, text):
        from ..compiler import load_system
        return load_system(text)

    def check(self, text):
        try:
            s = self._system(text); M = s.nominal
            from ..compiler.model import auto_timeout, estimate_run_s
            to = M.timeout
            return dict(ok=True, timeout_s=(auto_timeout(M) if to == "auto" else to), timeout_auto=(to == "auto"),
                        est_run_s=estimate_run_s(M),
                        bodies=[b for b in M.body_names if '__' not in b], elements=list(M.doc['elements']), dt=M.dt, dt_crit=M.dt_crit,
                        dt_auto=getattr(M, "dt_auto", M.dt),
                        t_end=M.t_end, steps=int(round(M.t_end / M.dt)))
        except Exception as exc:
            return _err(exc)

    def run(self, text, overrides=None, n_hist=600, model_path=None, export=False, track=False):
        """One run at the nominal values. export=True also writes the model's CSV exports (Run button);
        live runs pass export=False so typing never writes files. track=True reports stages to progress()."""
        return _run_payload(text, overrides, n_hist, model_path, export, track)

    def progress(self):
        """What the current Run / Verify / Report is doing, for the progress card."""
        p = dict(_PROG)
        now = time.time()
        p["elapsed"] = now - p.get("t0", now)
        p["solve_elapsed"] = now - p["solve_t0"] if p.get("solve_t0") else 0.0
        return p

    def live_run(self, text, n_hist=600):
        """Live-view run. The compiled solver releases Python's lock (nogil), so on a machine with 2+ cores the time
        integration runs on its own core while the editor's other calls (YAML, checks, Help) carry on."""
        r = self.run(text, None, n_hist)
        r["worker"] = (os.cpu_count() or 1) >= 2
        return r

    def verify(self, text, mc_n=None):
        from ..diagnostics import verify as V
        try:
            _prog_begin("verify", V.STAGES, "Compiling the model")
            V.set_progress_hook(_verify_hook)
            s = self._system(text)
            rep = s.verify(int(mc_n) if mc_n else None)
            checks = [dict(name=c.name, status=c.status, detail=c.detail) for c in rep.checks]
            fm = _clean({k: {kk: vv for kk, vv in m.items() if kk in ("p_fail", "nominal_SF", "SF_p5", "governing")}
                         for k, m in (getattr(rep, "failure", {}) or {}).items()})
            return dict(ok=True, verified=bool(rep.ok), summary=rep.summary(), checks=checks, failure=fm)
        except Exception as exc:
            return _err(exc)
        finally:
            V.set_progress_hook(None); _PROG["active"] = False

    def report(self, text, model_path=None, mc_n=None):
        """Full report (PDF + HTML) into an 'out' folder next to the saved model (or the home folder)."""
        from ..diagnostics import verify as V
        try:
            _prog_begin("report", V.STAGES + ["Writing the report"], "Compiling the model")
            V.set_progress_hook(_verify_hook)
            s = self._system(text)
            base = os.path.dirname(model_path) if model_path else os.path.join(os.path.expanduser("~"), "simblocks")
            stem = os.path.splitext(os.path.basename(model_path))[0] if model_path else "model"
            out_dir = os.path.join(base, "out", stem)
            rep = s.verify(int(mc_n) if mc_n else None)
            _PROG.update(stage=len(V.STAGES), label="Writing the report (plots, PDF, HTML)", sub_done=0, sub_total=0)
            paths = s.report(out_dir, rep=rep)
            return dict(ok=True, verified=bool(rep.ok), paths=_clean(paths), out_dir=out_dir)
        except Exception as exc:
            return _err(exc)
        finally:
            V.set_progress_hook(None); _PROG["active"] = False

    # ------------------------------------------------------------------ machine + settings (stored locally)
    def system_info(self):
        return dict(cores=os.cpu_count() or 1, settings=self.load_settings())

    def load_settings(self):
        try:
            with open(SETTINGS, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return dict(recent=[], workers=None)

    def save_settings(self, settings):
        try:
            os.makedirs(os.path.dirname(SETTINGS), exist_ok=True)
            with open(SETTINGS, "w", encoding="utf-8") as fh:
                json.dump(settings, fh, indent=1)
            return dict(ok=True)
        except Exception as exc:
            return _err(exc)

    def _remember(self, path):
        st = self.load_settings()
        rec = [p for p in st.get("recent", []) if p != path]
        st["recent"] = [path] + rec[:9]
        self.save_settings(st)

    def open_path_text(self, path):
        try:
            with open(path, encoding="utf-8") as fh:
                txt = fh.read()
            self._remember(path)
            return dict(ok=True, path=path, text=txt)
        except Exception as exc:
            return _err(exc)

    # ------------------------------------------------------------------ studies: sweeps x Monte Carlo, all local cores
    def study_estimate(self, text, workers=None):
        """How many runs the study blocks ask for, and how long that takes on the chosen cores. ask=True when the
        user should confirm first."""
        try:
            from ..compiler import load_system
            s = load_system(text)
            plan, sweeps, mc = study_plan(s.doc)
            s.run()                                     # first run compiles the kernels; time a warm one
            t0 = time.time(); s.run(); one = time.time() - t0
            w = max(1, int(workers or os.cpu_count() or 1))
            est = one * len(plan) / min(w, len(plan)) * 1.15
            self._last_one_run = (text, one)
            return dict(ok=True, runs=len(plan), sweeps=sweeps, mc=mc, one_run_s=one, workers=w, est_s=est,
                        ask=len(plan) > ASK_RUNS or est > ASK_SECONDS)
        except Exception as exc:
            return _err(exc)

    def study_start(self, text, workers=None, model_path=None):
        if getattr(self, "_job", None) and self._job.get("running"):
            return dict(ok=False, error="a study is already running")
        try:
            from ..compiler import load_system
            s = load_system(text)
            plan, sweeps, mc = study_plan(s.doc)
        except Exception as exc:
            return _err(exc)
        w = max(1, min(int(workers or os.cpu_count() or 1), len(plan)))
        # AUTO timeout for the whole study: 5 x (runs x time per run / cores), at least 30 s. A user limit is per run.
        from ..compiler.model import AUTO_FACTOR, estimate_run_s
        job_limit = None
        if s.nominal.timeout == "auto":
            prev = getattr(self, "_last_one_run", None)
            one = prev[1] if prev and prev[0] == text else estimate_run_s(s.nominal)
            job_limit = max(30.0, AUTO_FACTOR * len(plan) * one / w)
        job = self._job = dict(job_limit=job_limit, timed_out=False, running=True, done=0, total=len(plan), results=[None] * len(plan), cancel=False,
                               sweeps=sweeps, mc=mc, t0=time.time(), error=None,
                               units={k: o.get("unit", "") for k, o in s.doc["outputs"].items()},
                               failure=_clean(s.doc.get("failure") or []))

        def go():
            try:
                if w == 1:
                    _worker_init(text, model_path)
                    for i, ov in enumerate(plan):
                        if job["cancel"]:
                            break
                        if job_limit and time.time() - job["t0"] > job_limit:
                            job["timed_out"] = True; break
                        job["results"][i] = _worker_run(ov, i); job["done"] += 1
                else:
                    from concurrent.futures import ProcessPoolExecutor, as_completed
                    with ProcessPoolExecutor(max_workers=w, initializer=_worker_init, initargs=(text, model_path)) as ex:
                        futs = {ex.submit(_worker_run, ov, i): i for i, ov in enumerate(plan)}
                        for f in as_completed(futs):
                            job["results"][futs[f]] = f.result(); job["done"] += 1
                            if job_limit and time.time() - job["t0"] > job_limit:
                                job["timed_out"] = True; job["cancel"] = True
                            if job["cancel"]:
                                for g in futs:
                                    g.cancel()
                                break
            except Exception as exc:
                job["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                job["running"] = False
                job["elapsed"] = time.time() - job["t0"]
        threading.Thread(target=go, daemon=True).start()
        return dict(ok=True, total=len(plan), workers=w)

    def study_progress(self):
        j = getattr(self, "_job", None)
        if not j:
            return dict(ok=False, error="no study")
        out = dict(ok=True, running=j["running"], done=j["done"], total=j["total"], error=j["error"],
                   timed_out=j.get("timed_out", False), job_limit=j.get("job_limit"),
                   elapsed=time.time() - j["t0"])
        if not j["running"]:
            out.update(results=[r for r in j["results"] if r is not None], sweeps=j["sweeps"], mc=j["mc"],
                       units=j["units"], failure=j["failure"])
        return out

    def study_cancel(self):
        j = getattr(self, "_job", None)
        if j:
            j["cancel"] = True
        return dict(ok=True)

    def save_csv(self, text, name="study.csv"):
        """Study results as CSV through a save dialog (desktop) or returned for download."""
        try:
            if self._window is None:
                return dict(ok=False, error="no window")
            import webview
            r = self._window.create_file_dialog(webview.SAVE_DIALOG, save_filename=name, file_types=("CSV (*.csv)",))
            if not r:
                return dict(ok=False, cancelled=True)
            path = r if isinstance(r, str) else r[0]
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
            return dict(ok=True, path=path)
        except Exception as exc:
            return _err(exc)

    # ------------------------------------------------------------------ files (native dialogs in the desktop app)
    def open_file(self):
        try:
            if self._window is None:
                return dict(ok=False, error="no window")
            import webview
            r = self._window.create_file_dialog(webview.OPEN_DIALOG, file_types=("Sim files (*.yaml;*.yml)",
                                                                               "All files (*.*)"))
            if not r:
                return dict(ok=False, cancelled=True)
            p = r[0]
            return self.open_path_text(p)
        except Exception as exc:
            return _err(exc)

    def save_file(self, text, path=None, save_as=False):
        try:
            if path is None or save_as:
                if self._window is None:
                    return dict(ok=False, error="no window")
                import webview
                r = self._window.create_file_dialog(webview.SAVE_DIALOG, save_filename="model.yaml",
                                                   file_types=("Sim files (*.yaml)",))
                if not r:
                    return dict(ok=False, cancelled=True)
                path = r if isinstance(r, str) else r[0]
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
            self._remember(path)
            return dict(ok=True, path=path)
        except Exception as exc:
            return _err(exc)

    def open_path(self, path):
        """Open a report or folder with the system's default app."""
        try:
            import subprocess
            import sys
            if sys.platform.startswith("win"):
                os.startfile(path)                       # noqa: S606 (local file chosen by the app)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
            return dict(ok=True)
        except Exception as exc:
            return _err(exc)

    def pick_folder(self, start=None):
        """Native folder picker (desktop app)."""
        try:
            if self._window is None:
                return dict(ok=False, error="no window")
            import webview
            r = self._window.create_file_dialog(webview.FOLDER_DIALOG, directory=start or "")
            if not r:
                return dict(ok=False, cancelled=True)
            return dict(ok=True, path=r if isinstance(r, str) else r[0])
        except Exception as exc:
            return _err(exc)

    def set_model_dir(self, model_path=None):
        """Relative file paths in a model (CSV loads, pictures) are read from the saved model's folder."""
        if model_path:
            os.environ["SIMBLOCKS_MODEL_DIR"] = os.path.dirname(os.path.abspath(model_path))
        else:
            os.environ.pop("SIMBLOCKS_MODEL_DIR", None)
        return dict(ok=True)

    def pick_file(self, start=None, kind="csv"):
        """Native file picker (desktop app). kind: csv (load tables) or image (diagrams on the canvas)."""
        try:
            if self._window is None:
                return dict(ok=False, error="no window")
            import webview
            types = {"csv": ("Tables (*.csv;*.txt;*.dat)", "All files (*.*)"),
                     "image": ("Pictures (*.png;*.jpg;*.jpeg;*.gif;*.webp;*.svg)", "All files (*.*)")}.get(kind, ("All files (*.*)",))
            r = self._window.create_file_dialog(webview.OPEN_DIALOG, directory=start or "", file_types=types)
            if not r:
                return dict(ok=False, cancelled=True)
            return dict(ok=True, path=r if isinstance(r, str) else r[0])
        except Exception as exc:
            return _err(exc)

    def project_path(self, path, model_path=None, copy_to=None):
        """Make a picked file usable from the model: the path relative to the model's folder when it is inside it.
        If it is outside and copy_to is given (e.g. 'images'), copy it into <model folder>/<copy_to>/ first.
        Without a saved model the absolute path is returned (and saving later keeps it)."""
        try:
            import shutil
            path = os.path.abspath(path)
            if not model_path:
                return dict(ok=True, path=path.replace("\\", "/"), relative=False)
            base = os.path.dirname(os.path.abspath(model_path))
            rel = os.path.relpath(path, base)
            if rel.startswith(".."):
                if not copy_to:
                    return dict(ok=True, path=path.replace("\\", "/"), relative=False)
                dst_dir = os.path.join(base, copy_to); os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, os.path.basename(path))
                stem, ext = os.path.splitext(dst); i = 2
                while os.path.exists(dst) and not _same_file(dst, path):
                    dst = f"{stem}_{i}{ext}"; i += 1
                if not os.path.exists(dst):
                    shutil.copy2(path, dst)
                rel = os.path.relpath(dst, base)
                return dict(ok=True, path=rel.replace("\\", "/"), relative=True, copied=True)
            return dict(ok=True, path=rel.replace("\\", "/"), relative=True)
        except Exception as exc:
            return _err(exc)

    def read_image(self, path, model_path=None):
        """A picture for an image block, as a data: URL. A relative path is read from the model's folder."""
        try:
            import base64, mimetypes
            p = path if os.path.isabs(path) or not model_path else os.path.join(os.path.dirname(os.path.abspath(model_path)), path)
            mime = mimetypes.guess_type(p)[0] or ""
            if mime not in ("image/png", "image/jpeg", "image/gif", "image/webp", "image/svg+xml"):
                return dict(ok=False, error="an image block shows PNG, JPEG, GIF, WebP or SVG pictures")
            if os.path.getsize(p) > 25 * 1024 * 1024:
                return dict(ok=False, error="the picture is larger than 25 MB")
            with open(p, "rb") as fh:
                return dict(ok=True, image=f"data:{mime};base64," + base64.b64encode(fh.read()).decode("ascii"))
        except FileNotFoundError:
            return dict(ok=False, error=f"picture not found: {path}")
        except Exception as exc:
            return _err(exc)

    # ------------------------------------------------------------------ custom materials folder
    def _load_materials(self, folder):
        from ..materials import custom
        os.environ["SIMBLOCKS_MATERIALS"] = folder or ""          # study workers and the CLI read this too
        self._materials = custom.load_folders([folder]) if folder else (custom.unload() or None)
        self._catalog = None
        return self._materials

    def materials_status(self):
        st = self.load_settings() or {}
        m = self._materials or {}
        return dict(ok=True, folder=st.get("materials_dir", ""), added=m.get("added", []), replaced=m.get("replaced", []),
                    curves=m.get("curves", []), errors=m.get("errors", []), files=m.get("files", []))

    def set_materials_dir(self, folder):
        """Connect (or with '' disconnect) the custom materials folder. Saved in settings."""
        try:
            st = self.load_settings() or {}
            st["materials_dir"] = folder or ""
            self.save_settings(st)
            self._load_materials(folder or "")
            return self.materials_status()
        except Exception as exc:
            return _err(exc)

    def materials_template(self, folder):
        """Write _template.yaml and an example curve into the folder (never overwrites)."""
        try:
            from ..materials import custom
            return dict(ok=True, written=custom.write_template(folder))
        except Exception as exc:
            return _err(exc)

    def reload_materials(self):
        st = self.load_settings() or {}
        self._load_materials(st.get("materials_dir", ""))
        return self.materials_status()

    # ------------------------------------------------------------------ custom blocks folder
    def _load_blocks(self, folder):
        from .. import registry
        os.environ["SIMBLOCKS_BLOCKS"] = folder or ""
        self._blocks = registry.load_block_folders([folder]) if folder else (registry.unload_block_folders() or None)
        self._catalog = None
        return self._blocks

    def blocks_status(self):
        st = self.load_settings() or {}
        b = self._blocks or {}
        return dict(ok=True, folder=st.get("blocks_dir", ""), added=b.get("added", []), errors=b.get("errors", []),
                    files=b.get("files", []))

    def set_blocks_dir(self, folder):
        try:
            st = self.load_settings() or {}
            st["blocks_dir"] = folder or ""
            self.save_settings(st)
            self._load_blocks(folder or "")
            return self.blocks_status()
        except Exception as exc:
            return _err(exc)

    def reload_blocks(self):
        st = self.load_settings() or {}
        self._load_blocks(st.get("blocks_dir", ""))
        return self.blocks_status()

    def blocks_template_text(self):
        from .. import registry
        return dict(ok=True, text=registry.CUSTOM_TEMPLATE)

    def blocks_template(self, folder):
        try:
            from .. import registry
            return dict(ok=True, written=registry.write_block_template(folder))
        except Exception as exc:
            return _err(exc)

    # ------------------------------------------------------------------ help: every block in full
    def help_catalog(self):
        """Every catalog block with its full docstring, parameters, outputs, citation and a usage snippet."""
        try:
            from ..docs.library import entries
            from ..registry import CUSTOM
            return dict(ok=True, blocks=[dict(name=e["name"], domain=e["domain"], category=e["category"], path=e["path"],
                                              kind=e["kind"], source=e["source"], summary=e["summary"], doc=e["doc"],
                                              params=e["params"], outputs=e["outputs"], usage=e["usage"], fig=e["fig"],
                                              tags=e["tags"], legacy=e["legacy"], element=e.get("element", ""),
                                              custom=CUSTOM.get(e["name"], "")) for e in entries()])
        except Exception as exc:
            return _err(exc)

    def help_materials(self):
        """Every material in the catalog (built-in and custom) with its main properties, for Help."""
        try:
            from ..materials import _ALL, custom
            from ..materials.base import Isotropic, Lamina
            out = []
            for k, m in sorted(_ALL.items(), key=lambda kv: kv[0].lower()):
                row = dict(key=k, name=getattr(m, "name", k), note=getattr(m, "note", ""), custom=k in custom._LOADED)
                if isinstance(m, Isotropic):
                    row.update(type="isotropic", family=m.family, E=m.E, nu=m.nu, rho=m.rho, sy=m.sy, su=m.su,
                               elong=m.elong, hardness=m.hardness)
                elif isinstance(m, Lamina):
                    row.update(type="lamina", family="composite ply", E=m.E1, E2=m.E2, nu=m.nu12, rho=m.rho, sy=m.Xt, su=m.Xc)
                else:
                    row.update(type="fluid", family="fluid", rho=getattr(m, "rho", None))
                out.append(_clean(row))
            return dict(ok=True, materials=out)
        except Exception as exc:
            return _err(exc)

    def block_code(self, name):
        """Python source of one block, for Help's copy button (falls back to the copy bundled at build time)."""
        try:
            from ..registry import REGISTRY
            import inspect
            b = REGISTRY[name]
            py = getattr(b.func, "py_func", b.func)
            try:
                src = inspect.getsource(py)
                file = inspect.getsourcefile(py) or ""
            except (OSError, TypeError):
                src, file = _bundled_sources().get(name, ""), ""
            lines = src.splitlines()
            deco = [ln for ln in lines if ln.lstrip().startswith("@block(")]
            if not deco and src:                          # getsource of the bare function: rebuild the decorator line
                src = (f'@block("{b.domain}", "{b.category}", "{b.geometry}", "{b.load}", kind="{b.kind}",\n'
                       f'       source={b.source!r}, tags={tuple(b.tags)!r})\n') + src
            rel = file.split("simblocks", 1)[-1].lstrip("\\/") if "simblocks" in file else os.path.basename(file)
            return dict(ok=True, code=src, file=rel, kernel=b.kind == "kernel")
        except Exception as exc:
            return _err(exc)

    # ------------------------------------------------------------------ templates and node-group library
    def templates(self):
        """Starting models for New: name, question and the YAML text of each."""
        import re
        p = os.path.join(os.path.dirname(__file__), "templates.yaml")
        out = []
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                parts = re.split(r"\n---\s*\n", fh.read())
            for t in parts:
                try:
                    d = yaml.safe_load(t) or {}
                except Exception:
                    continue
                if isinstance(d, dict) and d.get("name"):
                    out.append(dict(name=d["name"], question=d.get("question", ""), text=t.strip() + "\n"))
        return out

    def list_groups(self):
        out = []
        if os.path.isdir(GROUPS):
            for f in sorted(os.listdir(GROUPS)):
                if f.endswith(".json"):
                    try:
                        with open(os.path.join(GROUPS, f), encoding="utf-8") as fh:
                            g = json.load(fh)
                        out.append(dict(file=f, name=g.get("name", f[:-5]), note=g.get("note", ""), data=g))
                    except Exception:
                        pass
        return out

    def save_group(self, name, data):
        try:
            import re
            os.makedirs(GROUPS, exist_ok=True)
            fn = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "group"
            path = os.path.join(GROUPS, fn + ".json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=1)
            return dict(ok=True, path=path)
        except Exception as exc:
            return _err(exc)

    def delete_group(self, file):
        try:
            os.remove(os.path.join(GROUPS, os.path.basename(file)))
            return dict(ok=True)
        except Exception as exc:
            return _err(exc)

    # ------------------------------------------------------------------ user themes (one folder each, in the per-user config dir)
    def themes_info(self):
        """Where user themes live, so the Settings dialog can say so."""
        return dict(ok=True, dir=themes_dir(), max_image_mb=THEME_IMAGE_MAX // (1024 * 1024), image_types=sorted(set(_IMG_EXT.values())))

    def list_themes(self):
        """Every user theme: slug, theme.json contents and whether it has a background image (fetch it with theme_image)."""
        try:
            return dict(ok=True, themes=list_user_themes())
        except Exception as exc:
            return _err(exc)

    def theme_image(self, slug):
        """The theme's background image as a data: URL, or None."""
        try:
            return dict(ok=True, image=read_theme_image(slug))
        except Exception as exc:
            return _err(exc)

    def new_theme(self, name, theme, image=None):
        """Create a user theme from a full theme dict (colours, fonts, grid, wire settings). Returns its slug."""
        try:
            return dict(ok=True, slug=create_user_theme(name, theme, image))
        except Exception as exc:
            return _err(exc)

    def save_theme(self, slug, theme, image=None):
        """Update a user theme. image: None keeps the current picture, "" removes it, a data: URL replaces it."""
        try:
            write_user_theme(slug, theme, image)
            return dict(ok=True, slug=slug)
        except Exception as exc:
            return _err(exc)

    def copy_theme(self, slug, name):
        """Duplicate a user theme (picture included) under a new name."""
        try:
            return dict(ok=True, slug=copy_user_theme(slug, name))
        except Exception as exc:
            return _err(exc)

    def rename_theme(self, slug, name):
        """Rename a user theme; its folder moves to the new name's slug."""
        try:
            return dict(ok=True, slug=rename_user_theme(slug, name))
        except Exception as exc:
            return _err(exc)

    def delete_theme(self, slug):
        try:
            delete_user_theme(slug)
            return dict(ok=True)
        except Exception as exc:
            return _err(exc)

    def examples(self):
        """Bundled example models, so a new user has something to open."""
        # the nine teaching examples (1D, 2D, 3D, simple to advanced); their CSV tables sit in data/ beside them
        here = os.path.join(os.path.dirname(__file__), "examples")            # packaged app (copied at build time)
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "examples", "learn"))
        d = here if os.path.isdir(here) else root
        out = []
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith((".yaml", ".yml")):
                    with open(os.path.join(d, f), encoding="utf-8") as fh:
                        text = fh.read()
                    title = next((ln.split(":", 1)[1].strip() for ln in text.splitlines() if ln.startswith("name:")), f)
                    out.append(dict(name=title, file=f, text=text, dir=d.replace("\\", "/")))
        return out


# ---------------------------------------------------------------------------------------------- user theme storage
# <config dir>/simblocks/themes/<slug>/theme.json plus an optional background.<png|jpg|webp|gif> in the same folder.
THEME_IMAGE_MAX = 10 * 1024 * 1024
_IMG_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg", "image/webp": "webp", "image/gif": "gif"}
_EXT_MIME = {"png": "image/png", "jpg": "image/jpeg", "webp": "image/webp", "gif": "image/gif"}


def _same_file(a, b):
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def config_dir():
    """Per-user config folder: %APPDATA%\\simblocks on Windows, ~/Library/Application Support/simblocks on macOS,
    $XDG_CONFIG_HOME (or ~/.config)/simblocks elsewhere. SIMBLOCKS_CONFIG_DIR overrides it (tests, portable installs)."""
    import sys
    env = os.environ.get("SIMBLOCKS_CONFIG_DIR")
    if env:
        return os.path.abspath(env)
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Roaming")
    elif sys.platform == "darwin":
        base = os.path.join(os.path.expanduser("~"), "Library", "Application Support")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "simblocks")


def themes_dir():
    return os.path.join(config_dir(), "themes")


def slugify(name):
    import re
    s = re.sub(r"[^a-z0-9]+", "-", str(name or "").lower()).strip("-")[:48].strip("-")
    return s or "theme"


def _check_slug(slug):
    """A slug is lower-case letters, digits and single hyphens; anything else (dots, slashes, '..') is refused."""
    import re
    if not isinstance(slug, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug) or len(slug) > 48:
        raise ValueError(f"not a valid theme id: {slug!r}")
    root = os.path.realpath(themes_dir())
    path = os.path.realpath(os.path.join(root, slug))
    if os.path.dirname(path) != root:
        raise ValueError(f"not a valid theme id: {slug!r}")
    return path


def _clean_name(name):
    name = " ".join(str(name or "").split())[:60]
    if not name:
        raise ValueError("a theme needs a name")
    return name


def _unique_slug(name, skip=None):
    base = slugify(name)
    slug, i = base, 2
    while os.path.exists(os.path.join(themes_dir(), slug)) and slug != skip:
        slug = f"{base[:44]}-{i}"
        i += 1
    return slug


def _decode_image(data_url):
    """data:image/<png|jpeg|webp|gif>;base64,... -> (bytes, ext). Checks type, magic bytes and size."""
    import base64
    if not isinstance(data_url, str) or not data_url.startswith("data:") or ";base64," not in data_url:
        raise ValueError("the background image must be a base64 data: URL")
    head, b64 = data_url[5:].split(";base64,", 1)
    mime = head.strip().lower()
    if mime not in _IMG_EXT:
        raise ValueError("background images must be PNG, JPEG, WebP or GIF")
    if len(b64) > THEME_IMAGE_MAX * 4 // 3 + 16:
        raise ValueError(f"the background image is larger than {THEME_IMAGE_MAX // (1024 * 1024)} MB")
    raw = base64.b64decode(b64, validate=True)
    if len(raw) > THEME_IMAGE_MAX:
        raise ValueError(f"the background image is larger than {THEME_IMAGE_MAX // (1024 * 1024)} MB")
    ext = _IMG_EXT[mime]
    magic = {"png": raw[:8] == b"\x89PNG\r\n\x1a\n", "jpg": raw[:3] == b"\xff\xd8\xff",
             "gif": raw[:6] in (b"GIF87a", b"GIF89a"), "webp": raw[:4] == b"RIFF" and raw[8:12] == b"WEBP"}[ext]
    if not magic:
        raise ValueError("the file content does not match its image type")
    return raw, ext


def _theme_image_path(folder):
    for ext in ("png", "jpg", "webp", "gif"):
        p = os.path.join(folder, "background." + ext)
        if os.path.isfile(p):
            return p
    return None


def _clean_theme(theme, name):
    if not isinstance(theme, dict):
        raise ValueError("theme must be an object")
    t = json.loads(json.dumps(theme))                    # plain JSON copy
    t.pop("image", None)
    t["name"] = name
    if not isinstance(t.get("t"), dict):
        raise ValueError("theme has no colours (t)")
    return t


def _write_theme_files(folder, theme, image):
    os.makedirs(folder, exist_ok=True)
    if image is not None:
        old = _theme_image_path(folder)
        if image == "":
            if old:
                os.remove(old)
        else:
            raw, ext = _decode_image(image)
            if old:
                os.remove(old)
            with open(os.path.join(folder, "background." + ext), "wb") as fh:
                fh.write(raw)
    tmp = os.path.join(folder, "theme.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(theme, fh, indent=1)
    os.replace(tmp, os.path.join(folder, "theme.json"))


def list_user_themes():
    root = themes_dir()
    out = []
    if os.path.isdir(root):
        for slug in sorted(os.listdir(root)):
            folder = os.path.join(root, slug)
            try:
                _check_slug(slug)
                with open(os.path.join(folder, "theme.json"), encoding="utf-8") as fh:
                    th = json.load(fh)
            except Exception:
                continue
            out.append(dict(slug=slug, theme=th, has_image=bool(_theme_image_path(folder))))
    return out


def read_theme_image(slug):
    import base64
    p = _theme_image_path(_check_slug(slug))
    if not p:
        return None
    with open(p, "rb") as fh:
        raw = fh.read()
    return f"data:{_EXT_MIME[p.rsplit('.', 1)[1]]};base64," + base64.b64encode(raw).decode("ascii")


def create_user_theme(name, theme, image=None):
    name = _clean_name(name)
    slug = _unique_slug(name)
    folder = _check_slug(slug)
    _write_theme_files(folder, _clean_theme(theme, name), image or None)
    return slug


def write_user_theme(slug, theme, image=None):
    folder = _check_slug(slug)
    if not os.path.isfile(os.path.join(folder, "theme.json")):
        raise ValueError(f"no user theme {slug!r}")
    _write_theme_files(folder, _clean_theme(theme, _clean_name(theme.get("name") if isinstance(theme, dict) else "")), image)


def copy_user_theme(slug, name):
    import shutil
    src = _check_slug(slug)
    with open(os.path.join(src, "theme.json"), encoding="utf-8") as fh:
        th = json.load(fh)
    name = _clean_name(name)
    new = _unique_slug(name)
    dst = _check_slug(new)
    shutil.copytree(src, dst)
    th["name"] = name
    _write_theme_files(dst, th, None)
    return new


def rename_user_theme(slug, name):
    src = _check_slug(slug)
    with open(os.path.join(src, "theme.json"), encoding="utf-8") as fh:
        th = json.load(fh)
    name = _clean_name(name)
    new = _unique_slug(name, skip=slug)
    if new != slug:
        os.replace(src, _check_slug(new))
    th["name"] = name
    _write_theme_files(_check_slug(new), th, None)
    return new


def delete_user_theme(slug):
    import shutil
    folder = _check_slug(slug)
    if not os.path.isfile(os.path.join(folder, "theme.json")):
        raise ValueError(f"no user theme {slug!r}")
    shutil.rmtree(folder)



def _inline_label_styles(svg):
    """Diagram labels carry their font and theme colour inline. Some WebView2 builds do not apply page CSS to the
    copies a <use> element draws, which left labels black and full-size on dark themes."""
    fl = 'style="font-size:7px;font-family:var(--mono);fill:var(--ink2)"'
    fc = 'style="font-size:8px;font-family:var(--mono);fill:var(--ink);text-anchor:middle"'
    return svg.replace('class="fl"', 'class="fl" ' + fl).replace('class="fc"', 'class="fc" ' + fc)


def _bundled_sources():
    p = os.path.join(os.path.dirname(__file__), "block_sources.json")
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except OSError:
        return {}


def write_block_sources(path=None):
    """Build step: save every block's source so the packaged app (which has no .py files) can show it in Help."""
    import inspect
    from ..registry import REGISTRY, CUSTOM
    import simblocks  # noqa: F401  (registers every block)
    out = {}
    for name, b in REGISTRY.items():
        if name in CUSTOM:
            continue
        py = getattr(b.func, "py_func", b.func)
        try:
            lines, _ = inspect.getsourcelines(py)
            start = 0
            out[name] = "".join(lines[start:])
        except (OSError, TypeError):
            pass
    path = path or os.path.join(os.path.dirname(__file__), "block_sources.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    return path


def custom_keys():
    from ..materials import custom
    return set(custom._LOADED)


# ---------------------------------------------------------------------------------------------- progress
_PROG = dict(active=False)
RUN_STAGES = ["Compile", "Solve", "Results"]


def _prog_begin(kind, stages, label):
    _PROG.clear()
    _PROG.update(active=True, kind=kind, stages=list(stages), stage=0, label=label, t0=time.time(), solve_t0=0.0,
                 est_s=0.0, steps=0, sub_done=0, sub_total=0)


def _verify_hook(i, label, done, total):
    lab = label + (f": sample {done:,} of {total:,}" if total else "")
    _PROG.update(stage=i, label=lab, sub_done=done, sub_total=total)


# ---------------------------------------------------------------------------------------------- one run (shared)
def _run_payload(text, overrides=None, n_hist=600, model_path=None, export=False, track=False):
    from ..compiler import load_system
    try:
        if track:
            _prog_begin("run", RUN_STAGES, "Compiling the model")
        s = load_system(text)
        paths = []
        if track:
            from ..compiler.model import estimate_run_s
            M = s.nominal; steps = int(round(M.t_end / M.dt))
            from ..compiler.exports import wanted as _w
            more = " then writing CSV" if (export and _w(s.doc, "run")) else ""
            _PROG.update(stage=1, label=f"Solving {steps:,} time steps{more}", steps=steps, est_s=estimate_run_s(M),
                         solve_t0=time.time())
        if export:
            out, H, paths = s.run_with_exports(overrides or None, model_path=model_path)
        else:
            out, H = s.run(overrides or None, history=True)
        if track:
            _PROG.update(stage=2, label="Outputs, energy ledger and time history", solve_t0=0.0)
        types = {k: e["type"] for k, e in s.doc["elements"].items()}
        keep = {"elastoplastic": ("plastic_set", "broken"), "drive": ("current", "battery_current", "bus_voltage", "soc")}

        def wanted(k):
            if "." not in k:
                return True
            nm, q = k.split(".", 1)
            if nm not in types:
                return True
            return q in ("force", "measure") or q in keep.get(types[nm], ())
        H = {k: v for k, v in H.items() if wanted(k)}
        n = len(H["t"]); step = max(1, n // n_hist)
        hist = {k: [float(x) for x in v[::step]] for k, v in H.items()}
        return dict(ok=True, outputs=_clean(out), history=hist, exported=paths,
                    units={k: o.get("unit", "") for k, o in s.doc["outputs"].items()})
    except Exception as exc:
        return _err(exc)
    finally:
        if track:
            _PROG["active"] = False


# ---------------------------------------------------------------------------------------------- studies
def parse_values(spec):
    """Sweep values: a list, '1, 2, 5', or 'start:stop:count' (count points, ends included). Units allowed."""
    from ..compiler.units import parse_quantity
    if isinstance(spec, (list, tuple)):
        return list(spec)
    t = str(spec).strip()
    if ":" in t and "," not in t:
        a, b, n = [x.strip() for x in t.split(":")]
        unit = ""
        for part in (a, b):
            bits = part.split(None, 1)
            if len(bits) == 2:
                unit = bits[1]
        lo, hi = parse_quantity(a if " " in a else f"{a} {unit}".strip()), parse_quantity(b if " " in b else f"{b} {unit}".strip())
        n = int(float(n))
        return [lo + (hi - lo) * i / max(n - 1, 1) for i in range(n)]
    return [x.strip() for x in t.split(",") if x.strip()]


def study_plan(doc):
    """Runs = every combination of the sweep values x the Monte Carlo samples (one shared sample set)."""
    import itertools
    st = doc.get("study") or {}
    sweeps = {k: parse_values(v) for k, v in (st.get("sweep") or {}).items()}
    mc = st.get("monte_carlo") or {}
    ranges = mc.get("ranges") or {}
    n_mc = int(mc.get("n", 0) or 0) if ranges else 0
    samples = [{}]
    if ranges and n_mc > 0:
        from ..batch import make_samples
        from ..compiler.units import parse_quantity
        def num(x):
            return float(parse_quantity(x)) if isinstance(x, str) else float(x)
        uni = {k: (num(v[0]), num(v[1])) for k, v in ranges.items()}
        samples = make_samples(dict(samples=dict(n=n_mc, sampler=mc.get("sampler", "sobol"), seed=int(mc.get("seed", 0)),
                                                 uniform=uni)))
    keys = list(sweeps)
    combos = [dict(zip(keys, c)) for c in itertools.product(*[sweeps[k] for k in keys])] if keys else [{}]
    return [{**c, **s} for c in combos for s in samples], keys, list(ranges)


_WORKER = {}


def _worker_init(text, model_path=None):
    from ..compiler import load_system
    from ..compiler.exports import wanted
    _WORKER["sys"] = load_system(text)
    _WORKER["path"] = model_path
    _WORKER["export"] = bool(wanted(_WORKER["sys"].doc, "study"))


def _worker_run(ov, i=0):
    try:
        s = _WORKER["sys"]
        if _WORKER.get("export"):
            out, _, paths = s.run_with_exports(ov or None, model_path=_WORKER.get("path"), run=f"run{i + 1:04d}",
                                               when="study")
        else:
            out = s.run(ov or None)
        return dict(ok=True, inputs=_clean(ov), outputs=_clean(out))
    except Exception as exc:
        return dict(ok=False, inputs=_clean(ov), error=str(exc))


def _material_params(doc):
    """Names of params whose value is a material (a library key or an alias such as '4140 prehard'), so the editor
    shows them as Material blocks."""
    from ..materials import get
    from ..materials.curves import resolve_material
    out = []
    for k, v in (doc.get("params") or {}).items():
        if isinstance(v, dict) and "value" in v:
            v = v["value"]
        if not isinstance(v, str) or v.strip().startswith("=") or "$" in v:
            continue
        for f in (lambda t: get(t), lambda t: get(resolve_material(t))):
            try:
                f(v.strip()); out.append(k); break
            except Exception:
                pass
    return out
