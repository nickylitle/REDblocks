"""Command-line tool. Runs system files with no Claude involved.

  simblocks check  model.yaml                 validate + compile + draw the schematic (no sim)
  simblocks run    model.yaml                 nominal run, print outputs and the energy ledger
  simblocks verify model.yaml [--mc N]        full verification battery, print the summary
  simblocks report model.yaml [--out DIR]     verification + report pages, PDF and HTML
  simblocks sweep  model.yaml PARAM V1 V2 ... nominal outputs at each value (values take units: '8 mm')
  simblocks set    model.yaml PARAM=VALUE ... run with overrides, e.g. d="12 mm" shaft_mat=7075
  simblocks catalog [text]                    list blocks, optionally filtered
  simblocks materials                         list catalog materials
  simblocks library [out.html]                write the searchable block library page
  simblocks batch job.yaml [--workers N] [--n N] [--out results.zip]
                                              run a study on every CPU core, pack raw data into one zip
  simblocks batch-info results.zip            summary of a results file
  simblocks app [model.yaml]                  open the node editor (desktop window, needs pywebview)
Exit code 0 = success (and VERIFIED for verify/report), 1 = model error, 2 = not verified."""
import argparse
import os
import sys


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _print_outputs(out, sysm=None):
    w = max(len(k) for k in out)
    for k, v in out.items():
        if k.startswith("_") and k != "_ledger_rel":
            continue
        unit = ""
        if sysm and k in sysm.doc["outputs"]:
            unit = sysm.doc["outputs"][k].get("unit", "")
        print(f"  {k:<{w}}  {_fmt(v)} {unit}".rstrip())
    for k, v in out.items():
        if k.endswith("_reached") and v < 0.5:
            print(f"  note: {k[1:-8]} never reached its threshold within t_end (value shown is a sentinel)")
    from .compiler.model import control_messages
    for m in control_messages(out):
        print(f"  note: {m}")


def _parse_set(items):
    ov = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"expected PARAM=VALUE, got '{it}'")
        k, v = it.split("=", 1)
        ov[k.strip()] = v.strip()
    return ov


def main(argv=None):
    ap = argparse.ArgumentParser(prog="simblocks", description="Run simblocks system files (YAML) without Claude.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c, opts in (("check", ("out",)), ("run", ("out",)), ("verify", ("mc", "out")), ("report", ("mc", "out"))):
        p = sub.add_parser(c); p.add_argument("file")
        if "mc" in opts:
            p.add_argument("--mc", type=int, default=None, help="Monte Carlo samples (default: run.mc_n)")
        p.add_argument("--out", default=None, help={"check": "folder for the schematic",
                       "run": "folder for outputs.json and history.csv", "verify": "folder for verify.md",
                       "report": "folder for the report"}[c])
    p = sub.add_parser("sweep"); p.add_argument("file"); p.add_argument("param"); p.add_argument("values", nargs="+")
    p = sub.add_parser("set"); p.add_argument("file"); p.add_argument("assign", nargs="+")
    p = sub.add_parser("catalog"); p.add_argument("text", nargs="?", default=None)
    sub.add_parser("materials")
    p = sub.add_parser("library"); p.add_argument("out", nargs="?", default="block_library.html")
    p = sub.add_parser("app"); p.add_argument("file", nargs="?", default=None)
    p = sub.add_parser("batch"); p.add_argument("job")
    p.add_argument("--workers", type=int, default=None, help="processes (default: all CPU cores)")
    p.add_argument("--n", type=int, default=None, help="samples (default: samples.n in the job)")
    p.add_argument("--out", default=None, help="results zip (default: <name>_results.zip next to the job)")
    p.add_argument("--fresh", action="store_true", help="ignore a partial run and start over")
    p = sub.add_parser("batch-info"); p.add_argument("results")
    sub.add_parser("mcp", help="run as an MCP server (stdio) so an AI assistant can read, edit, run and verify models")
    a = ap.parse_args(argv)
    if a.cmd == "mcp":
        from .mcp_server import main as mcp_main
        return mcp_main()
    if a.cmd == "app":
        from .app.main import main as app_main
        return app_main([a.file] if a.file else [])
    if a.cmd == "batch":
        from .batch import run_batch
        run_batch(a.job, workers=a.workers, n=a.n, out=a.out, resume=not a.fresh)
        return 0
    if a.cmd == "batch-info":
        from .batch import summarize
        summarize(a.results)
        return 0

    from .compiler import load_system, SpecError
    try:
        if a.cmd == "catalog":
            from .registry import find
            for b in find(text=a.text):
                print(f"{b.name:<36} {b.path:<58} [{b.kind}] {b.summary[:70]}")
            return 0
        if a.cmd == "library":
            from .docs.library import write
            print(write(a.out))
            return 0
        if a.cmd == "materials":
            from .materials import _ALL
            for k, m in _ALL.items():
                print(f"{k:<28} {getattr(m, 'name', '')}")
            return 0
        sysm = load_system(a.file)
        out_dir = getattr(a, "out", None) or os.path.join(os.path.dirname(os.path.abspath(a.file)), "out",
                                                          os.path.splitext(os.path.basename(a.file))[0])
        if a.cmd == "check":
            os.makedirs(out_dir, exist_ok=True)
            M = sysm.nominal
            print(f"OK  {sysm.name}")
            print(f"    bodies: {', '.join(b for b in M.body_names if '__' not in b)}")
            print(f"    elements: {', '.join(f'{n} ({e['type']}' + (f', {e['segments']} segments' if e.get('segments') else '') + ')' for n, e in M.doc['elements'].items())}")
            if getattr(M, "adaptive", False):
                print(f"    dt adaptive: {M.dt_contact:.3g} s at contacts, {M.dt_free:.3g} s in free motion "
                      f"(critical {M.dt_crit:.3g} s), t_end = {M.t_end:.3g} s, "
                      f"{int(round(M.t_end / M.dt_free))} to {int(round(M.t_end / M.dt_contact))} steps")
            else:
                print(f"    dt = {M.dt:.3g} s (critical {M.dt_crit:.3g} s), t_end = {M.t_end:.3g} s, "
                      f"{int(round(M.t_end / M.dt))} steps")
            paths = sysm.schematic(os.path.join(out_dir, "schematic"))
            print(f"    schematic: {paths[0]}")
            return 0
        if a.cmd == "run":
            print(f"{sysm.name}: nominal run")
            out, _, paths = sysm.run_with_exports(model_path=os.path.abspath(a.file))
            _print_outputs(out, sysm)
            if "_steps" in out:
                print(f"  adaptive step: {out['_steps']:.0f} steps, {out['_dt_smallest']:.3g} s to "
                      f"{out['_dt_largest']:.3g} s, {out['_contact_steps']:.0f} at the contact step")
            if out.get("_timed_out"):
                print(f"  STOPPED BY THE TIMEOUT after {out['_timeout_s']:.0f} s at t = {out['_t_stopped']:.4g} s: "
                      f"outputs cover only that part (raise run.timeout, or timeout: none)")
            for p in paths:
                print(f"  csv: {p}")
            if getattr(a, "out", None):                  # --out: the outputs and the full time history as files
                import csv as _csv, json as _json
                os.makedirs(a.out, exist_ok=True)
                _, H = sysm.run(history=True)
                with open(os.path.join(a.out, "outputs.json"), "w") as fh:
                    _json.dump({k: float(v) for k, v in out.items() if isinstance(v, (int, float))}, fh, indent=1)
                from .compiler.exports import series_unit
                main_q = (".force", ".measure", ".position", ".velocity", ".x", ".y", ".z", ".angle", ".omega")
                keys = ["t"] + [k for k in H if k != "t" and (k.endswith(main_q) or any(float(x) != 0.0 for x in H[k]))]
                with open(os.path.join(a.out, "history.csv"), "w", newline="") as fh:
                    w = _csv.writer(fh)
                    w.writerow([f"{k} [{series_unit(sysm.nominal, sysm.doc, k)}]" if series_unit(sysm.nominal, sysm.doc, k)
                                else k for k in keys])
                    for row in zip(*(H[k] for k in keys)):
                        w.writerow([f"{x:.9g}" for x in row])
                print(f"  wrote {os.path.join(a.out, 'outputs.json')} and history.csv")
            return 0
        if a.cmd == "set":
            ov = _parse_set(a.assign)
            print(f"{sysm.name}: run with {ov}")
            _print_outputs(sysm.run(ov), sysm)
            return 0
        if a.cmd == "sweep":
            print(f"{sysm.name}: sweep {a.param}")
            for v in a.values:
                o = sysm.run({a.param: v})
                print(f"{a.param} = {v}")
                _print_outputs(o, sysm)
            return 0
        if a.cmd == "verify":
            rep = sysm.verify(a.mc)
            print(rep.markdown())
            if a.out:
                os.makedirs(a.out, exist_ok=True)
                open(os.path.join(a.out, "verify.md"), "w").write(rep.markdown())
            return 0 if rep.ok else 2
        if a.cmd == "report":
            rep = sysm.verify(a.mc)
            paths = sysm.report(out_dir, rep=rep)
            print(rep.summary())
            for k, p in paths.items():
                print(f"  {k}: {p}")
            return 0 if rep.ok else 2
    except SpecError as exc:
        print(f"MODEL ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
