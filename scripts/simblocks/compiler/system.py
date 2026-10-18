"""System: one loaded system file. Runs, verifies and reports without any hand-written code."""
import math
import os

import numpy as np

from .loader import load, check, SpecError, line_of, with_location
from .model import compile_model, run_model, resolve, num, KNOCK, ELASTIC
from .expr import eval_numeric, ExprError
from .units import unit_factor, parse_quantity, UnitError


def _with_blocks(M, r, history):
    """Add each model block's results (block.output) to the outputs."""
    if not M.doc.get("models"):
        return r
    out = r[0] if history else r
    from .chain import block_results
    for k, v in block_results(M).items():
        out.setdefault(k, v)
    return r


def load_system(path_or_text):
    return System(path_or_text)


class System:
    def __init__(self, path_or_text):
        self.doc = load(path_or_text)
        check(self.doc)
        self.path = self.doc["__file__"]
        self.name = self.doc["name"]
        try:
            self.nominal = compile_model(self.doc)      # compiles once now so errors surface at load time
        except SpecError as exc:
            raise with_location(exc, self.doc) from None

    # ------------------------------------------------------------------ running
    def compile(self, overrides=None):
        return compile_model(self.doc, overrides) if overrides else self.nominal

    def run(self, overrides=None, dt=None, history=False, n_log=800, timeout="model"):
        M = self.compile(overrides)
        r = run_model(M, dt=dt, history=history, n_log=n_log, timeout=timeout)
        return _with_blocks(M, r, history)

    def run_with_exports(self, overrides=None, model_path=None, run="nominal", when="run", n_log=800):
        """One run that also writes the file's CSV exports. Returns (outputs, history, paths written)."""
        from . import exports as X
        if not X.wanted(self.doc, when):
            out, H = self.run(overrides, history=True, n_log=n_log)
            return out, H, []
        M = self.compile(overrides)
        n = int(round(M.t_end / M.dt))
        out, H = _with_blocks(M, run_model(M, history=True, n_log=max(n_log, X.log_points(self.doc, n))), True)
        return out, H, X.write(self.doc, H, model_path or self.path, run=run, when=when, M=M)

    def _si(self, v, where, key=None):
        if key is not None and isinstance(self.nominal.ctx.get(key), str):
            return str(v)                              # text parameter (material, layup): never a number
        if isinstance(v, str) and not v.strip().startswith("$"):
            try:
                return parse_quantity(v)
            except UnitError:
                return v                               # text value (material name)
        return v

    # ------------------------------------------------------------------ verification spec
    def failure_modes(self):
        from ..diagnostics.failure import FailureMode
        outs = self.doc["outputs"]; modes = []
        for i, f in enumerate(self.doc.get("failure", []) or []):
            where = f"failure[{i}]"
            if "metric" in f:
                lim = f["limit"]
                if isinstance(lim, str):
                    try:
                        lim = parse_quantity(lim) / unit_factor(outs[f["metric"]].get("unit", ""))
                    except UnitError as exc:
                        raise SpecError(str(exc), where, line_of(f, "limit"), self.path)
                modes.append(FailureMode(f["name"], metric=f["metric"], limit=float(lim),
                                         fails_when=f.get("fails_when", "above"), origin=f["origin"],
                                         assumption=f.get("assumption", ""), reach_metric=f.get("reach_metric")))
            else:
                dem, cap = f["demand"], f["capacity"]
                from .model import numeric_namespace
                params = numeric_namespace(self.nominal.ctx)

                def chk(o, dem=dem, cap=cap, params=params, where=where):
                    ns = dict(params); ns.update({k: v for k, v in o.items() if not k.startswith("_")})
                    try:
                        d = float(eval_numeric(str(dem), ns)); c = float(eval_numeric(str(cap), ns))
                    except ExprError as exc:
                        raise SpecError(str(exc), where)
                    return d >= c, (c / d if d > 0 else math.inf)
                modes.append(FailureMode(f["name"], check=chk, origin=f["origin"], assumption=f.get("assumption", "")))
        return modes

    def spec(self, mc_n=None):
        from ..diagnostics.verify import SimSpec
        doc = self.doc; run = doc["run"]
        unc = {}
        for k, rng in (doc.get("uncertain", {}) or {}).items():
            if k not in (doc.get("params") or {}):
                raise SpecError(f"uncertain '{k}' is not a parameter", "uncertain", line_of(doc["uncertain"], k), self.path)
            lo, hi = (num(resolve(x, self.nominal.ctx, f"uncertain.{k}"), f"uncertain.{k}") for x in rng)
            unc[k] = (lo, hi)
        knock = {}
        for en, e in doc["elements"].items():
            exp = (e.get("knockout") or {}).get("expect", {}) if isinstance(e.get("knockout"), dict) else {}
            if e["type"] in ("force",) and not exp:
                continue
            kd_ = e.get("knockout") if isinstance(e.get("knockout"), dict) else {}
            if kd_.get("skip"):                     # an element that only acts in another setting of the model
                knock[f"no {en}"] = "SKIP: " + str(kd_["skip"])
                continue
            knock[f"no {en}"] = ({KNOCK + en: True}, dict(exp))
            if e["type"] == "elastoplastic":
                ex2 = (e.get("knockout") or {}).get("expect_elastic", {}) if isinstance(e.get("knockout"), dict) else {}
                knock[f"{en} elastic only"] = ({ELASTIC + en: True}, dict(ex2))
        for kn, kd in (doc.get("knockouts", {}) or {}).items():
            ov = {k: self._si(v, f"knockouts.{kn}", k) for k, v in (kd.get("set") or {}).items()}
            knock[kn] = (ov, dict(kd.get("expect", {})))
        limits = []
        for i, L in enumerate(doc.get("limits", []) or []):
            ov = {k: self._si(v, f"limits[{i}]", k) for k, v in (L.get("set") or {}).items()}
            limits.append((L.get("name", f"limit {i}"), ov, {k: float(v) for k, v in (L.get("expect") or {}).items()}))
        bounds = {k: tuple(float(x) for x in v) for k, v in (run.get("bounds", {}) or {}).items()}
        M = self.nominal
        metrics = [k for k in doc["outputs"] if not k.startswith("_")]

        # Verify's runs (finer steps, knockouts, samples) are not cut short by the AUTO timeout, which is sized for one
        # ordinary run; a limit the file sets still applies
        tmo = None if M.timeout == "auto" else "model"

        def runner(p, dt):
            return self.run(p, dt, timeout=tmo)

        def crit(p):
            return compile_model(doc, p).dt_crit if p else M.dt_crit
        sp = SimSpec(name=self.name, run=runner, params={}, dt=M.dt, metrics=metrics, uncertain=unc,
                     knockouts=knock, limits=limits, bounds=bounds, critical_dt=crit,
                     mc_n=int(mc_n or run.get("mc_n", 100)), failure_modes=self.failure_modes())
        seen = {}

        def idle(kname):
            """A knockout 'no <element>' of an element that carried no force at all in the nominal run (a contact
            that never touched): removing it cannot change anything, which is not a wiring bug."""
            en = kname[3:] if kname.startswith("no ") else None
            if en is None or en not in doc["elements"]:
                return False
            if "H" not in seen:
                try:
                    seen["H"] = self.run(None, history=True)[1]
                except Exception:
                    seen["H"] = {}
            f = seen["H"].get(f"{en}.force")
            return f is not None and len(f) > 0 and float(np.max(np.abs(f))) == 0.0
        sp.idle = idle
        return sp

    def verify(self, mc_n=None):
        from ..diagnostics.verify import verify_all
        self.rep = verify_all(self.spec(mc_n))
        return self.rep

    # ------------------------------------------------------------------ outputs
    def schematic(self, path_no_ext, H=None):
        doc = self.doc; M = self.nominal
        if getattr(M, "dim", "1d") == "3d":
            from ..viz.scene3d import draw as draw3
            return draw3(M, path_no_ext, H=H, title=f"{self.name}: 3d model")
        if getattr(M, "dim", "1d") == "2d":
            from ..viz.scene2d import draw
            q_end = None
            if H is not None and len(H.get("t", [])):
                q_end = np.array([H[f"{b}.{k}"][-1] for b in M.body_names for k in ("x", "y", "angle")])
            return draw(M, path_no_ext, q_end=q_end, title=f"{self.name}: 2d model")
        from ..viz.diagram import Schematic
        s = Schematic(f"{self.name}: physical model", "Drawn from the system file; values from the compiled model")
        grounds = [b for b, d in doc["bodies"].items() if d.get("ground")]
        used_ground = set()
        for b, d in doc["bodies"].items():
            if d.get("ground"):
                continue
            i = M.body_names.index(b)
            info = [f"{'m' if d['dof'] == 'translation' else 'I'} = {M.arrays['m'][i]:.4g} "
                    f"{'kg' if d['dof'] == 'translation' else 'kg m^2'}"]
            if M.arrays["v0"][i]:
                info.append(f"v0 = {M.arrays['v0'][i]:.4g} {'m/s' if d['dof'] == 'translation' else 'rad/s'}")
            s.body(b, "rotor" if d["dof"] == "rotation" else "mass", d.get("label", b), info=info)
        glyph = {"spring": "spring", "dashpot": "dashpot", "kelvin_voigt": "contact", "hertz": "contact",
                 "friction": "slider", "drag": "dashpot"}
        links = []
        info = lambda en: M.el_info.get(en) or M.el_info[M.groups[en].segs[0]]
        for en, e in doc["elements"].items():
            t = e["type"]
            members = [b for b in doc["bodies"] if b in str(e["measure"]).replace("-", " ").replace("+", " ")
                       .replace("*", " ").split()]
            used_ground.update(b for b in members if b in grounds)
            if t == "force":
                ctl = (self.doc["elements"][en].get("P") or {}) if isinstance(self.doc["elements"][en].get("P"), dict) else {}
                if ctl.get("control"):
                    s.note(f"{e.get('label', en)}: force set by the control signal '{ctl['control']}' along +({e['measure']})")
                else:
                    s.note(f"{e.get('label', en)}: constant force {info(en)['params'][0]:.4g} N along +({e['measure']})")
                continue
            if t in ("drive", "dc_motor"):
                mk = f"__motor_{en}"
                s.body(mk, "motor", e.get("label", en), info=[f"Kt = {info(en)['params'][0]:.3g} N m/A",
                                                              f"gear {info(en)['params'][3]:g}:1"])
                links.append((mk, members[0], "rigid", e.get("label", en), "drive torque", e.get("source", "drive")))
                continue
            if len(members) > 2:
                # measure combines several coordinates (e.g. hub + R*rotor - target): join the + side at a
                # junction point with rigid wires, then draw the element from the junction to the - side
                coef = M.arrays["C"][M.el_names.index(en if en in M.el_names else M.groups[en].segs[0])]
                sign = {b: coef[M.body_names.index(b)] for b in members if b in M.body_names}
                pos_ = [b for b in members if sign.get(b, 0) > 0]; neg_ = [b for b in members if sign.get(b, 0) < 0]
                jn = f"__j_{en}"
                s.body(jn, "point", "")
                side_a, side_b = (pos_, neg_) if len(pos_) >= len(neg_) else (neg_, pos_)
                for b in side_a:
                    links.append((b, jn, "rigid", "", "", ""))
                members = [jn] + side_b[:1]
                for b in side_b[1:]:
                    links.append((members[1], b, "rigid", "", "", ""))
            if len(members) == 1:
                if grounds:
                    gname = grounds[0]
                else:
                    gname = "__ground"
                    if gname not in s.bodies:
                        s.body(gname, "ground", "")
                used_ground.add(gname)
                members.append(gname)
            k = "contact" if (t == "elastoplastic" and e.get("one_sided")) else \
                ("elastoplastic" if t == "elastoplastic" else glyph[t])
            val = f"{t}, measure: {e['measure']}"
            if info(en).get("k"):
                val += f"\nk = {info(en)['k']:.3g}"
            links.append((members[0], members[1], k, e.get("label", en), val, e.get("source", t)))
        for g_ in grounds:
            if g_ in used_ground:
                s.body(g_, "ground", "")
        for l in links:
            s.link(*l)
        return s.draw(path_no_ext)

    def report(self, out_dir, mc_n=None, rep=None):
        from ..viz.report import results_report
        from ..viz import theme as T
        doc = self.doc; os.makedirs(out_dir, exist_ok=True)
        rep = rep or getattr(self, "rep", None) or self.verify(mc_n)
        nom, H = self.run(history=True)
        r = doc.get("report", {}) or {}
        fm = getattr(rep, "failure", {}) or {}
        # answer tile
        if "answer" in r:
            a = r["answer"]; v = nom[a["metric"]]
            answer = dict(label=a.get("label", a["metric"]), value=f"{v:.4g}", unit=a.get("unit", ""),
                          status=None, sub=a.get("sub", ""))
        else:
            failed = any(m["nominal_failed"] for m in fm.values())
            unreach = any(m.get("reach") and not m["reach"]["reachable"] for m in fm.values())
            answer = dict(label=r.get("question_short", "Survives?"), value="NO" if failed else ("N/A" if unreach else "YES"),
                          unit="", status="FAIL" if failed else ("WARN" if unreach else "PASS"),
                          sub=", ".join(f"{k} SF {m['nominal_SF']:.3g}" for k, m in fm.items())[:90])
        # time histories
        t_end = self.nominal.t_end
        ts, tl = (1e6, "time (us)") if t_end < 2e-3 and t_end < 1e-4 else ((1e3, "time (ms)") if t_end < 2.0 else (1.0, "time (s)"))
        panels = []; hist = dict(t=H["t"])
        plots = r.get("plots") or self._default_plots()
        for pi, pl in enumerate(plots):
            keys = []; labels = []
            for si_, ser in enumerate(pl["series"]):
                ref = ser["ref"]
                if ref not in H:
                    raise SpecError(f"plot series '{ref}' is not a logged quantity (use <element>.force, "
                                    f"<body>.velocity, ...)", f"report.plots[{pi}]")
                key = f"p{pi}_{si_}"; f_ = unit_factor(ser.get("unit", "")) if ser.get("unit") else 1.0
                hist[key] = H[ref] / f_; keys.append(key); labels.append(ser.get("label", ref))
            pnl = dict(title=pl["title"], keys=keys, labels=labels, unit=pl.get("unit", pl["series"][0].get("unit", "")),
                       color=pi % 5, t_scale=ts, t_label=tl)
            if pl.get("limit") is not None:
                pnl["limit"] = float(pl["limit"]); pnl["limit_label"] = pl.get("limit_label", "limit")
            panels.append(pnl)
        # sweep
        sweep = None
        if r.get("sweep"):
            sw = r["sweep"]; metric = sw["metric"]
            xs = [num(resolve(x, self.nominal.ctx, "report.sweep.values"), "report.sweep.values") for x in sw["values"]]
            xunit = sw.get("unit", "")
            xf = unit_factor(xunit) if xunit else 1.0
            variants = sw.get("variants") or {"nominal": {}}
            series = {}
            for lab, ov in variants.items():
                ov = {k: self._si(v, "report.sweep.variants", k) for k, v in (ov or {}).items()}
                ys = []
                for x in xs:
                    o = self.run(dict(ov, **{sw["param"]: x}))
                    ys.append(o[metric] if o.get(f"_{metric}_reached", 1.0) > 0.5 else float("nan"))
                series[lab] = ys
            lim = next((m for m in (doc.get("failure") or []) if m.get("metric") == metric), None)
            mark_raw = (doc.get("params") or {}).get(sw["param"])
            mark = self.nominal.ctx.get(sw["param"]) if isinstance(self.nominal.ctx.get(sw["param"]), float) else None
            sweep = dict(x=np.array(xs), x_scale=1.0 / xf, x_label=sw.get("x_label", f"{sw['param']} ({xunit or 'SI'})"),
                         y_label=sw.get("y_label", metric), title=sw.get("title", f"{doc['outputs'][metric].get('label', metric)} vs {sw.get('x_label', sw['param'])}"),
                         series=series, mark=mark,
                         input_labels=r.get("input_labels", {}))
            if lim:
                sweep["limit"] = float(lim["limit"]) if not isinstance(lim["limit"], str) else \
                    parse_quantity(lim["limit"]) / unit_factor(doc["outputs"][metric].get("unit", ""))
                sweep["limit_label"] = f"{lim['name']} limit"
                sweep["fails_when"] = lim.get("fails_when", "above")
        findings = self._findings(nom, fm)
        sch = self.schematic(os.path.join(out_dir, "schematic"), H)
        paths = results_report(out_dir, self.name, doc.get("question", ""), answer, rep, history=hist,
                               history_panels=panels, sweep=sweep, findings=findings,
                               assumptions=doc.get("assumptions", []) or [], sources=doc.get("sources", []) or [],
                               extra_svgs=[sch[1]])
        return paths

    def _default_plots(self):
        doc = self.doc; plots = []
        seen = []
        for o in doc["outputs"].values():
            if "element" in o and o["element"] not in seen:
                seen.append(o["element"])
                plots.append(dict(title=f"{o['element']} force", series=[dict(ref=f"{o['element']}.force")], unit="N"))
            if len(plots) >= 3:
                break
        if not plots:
            b = self.nominal.body_names[0]
            q = "speed" if getattr(self.nominal, "dim", "1d") in ("2d", "3d") else "velocity"
            plots.append(dict(title=f"{b} {q}", series=[dict(ref=f"{b}.{q}")]))
        return plots

    def _findings(self, nom, fm):
        lines = []
        outs = self.doc["outputs"]
        vals = []
        for k, o in outs.items():
            v = nom[k]
            if not o.get("reduce", "final").startswith("first_time") or nom.get(f"_{k}_reached", 1.0) > 0.5:
                vals.append(f"{o.get('label', k)} = {v:.4g}{(' ' + o['unit']) if o.get('unit') else ''}")
            else:
                vals.append(f"{o.get('label', k)}: never reached within t_end")
        lines.append("Nominal: " + "; ".join(vals) + ".")
        for name, m in fm.items():
            drv = m["drivers"][0] if m.get("drivers") else None
            lines.append(f"{name}: nominal SF {m['nominal_SF']:.3g}, P(fail) {m['p_fail'] * 100:.3g}% over {m['n']} "
                         f"Monte Carlo runs" + (f"; top driver {drv[0]} (rank corr {drv[1]:+.2f})." if drv else "."))
        return lines
