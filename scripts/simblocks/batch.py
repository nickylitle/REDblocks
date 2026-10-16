"""Local batch runner: run a sim's Monte Carlo / configuration study on every CPU core and pack the raw data
into ONE results file to send back for interpretation and graphing.

  simblocks batch job.yaml                    # all cores, default sample count from the job
  simblocks batch job.yaml --workers 12 --n 2000
  simblocks batch job.yaml --resume           # continue an interrupted run (default: resume if possible)
  simblocks batch-info results.zip            # quick summary of a finished results file

Job file (YAML):
  simblocks_batch: 1
  name: my_study
  sim: path/to/sim.py            # relative to the job file
  entry: batch_record            # function(params: dict, dt: float) -> JSON-serialisable dict
  dt: 2.5e-7
  base: {}                       # overrides applied to every run
  configs: {name: {param: value}, ...}           # every sample runs under every config
  samples: {n: 64, sampler: sobol, seed: 11, uniform: {param: [lo, hi], ...}}   # sobol | lhs | random
  dt_check: {n: 40, config: name, factors: [2, 4]}   # optional: rerun n samples at dt/2, dt/4
  verify: {entry: make_spec, pick: pick_nominal, mc_n: 20}   # optional: verify_all on one run
                                 # (make_spec(params, uncertain=<samples.uniform>, mc_n=...) -> SimSpec)

Results file (zip): meta.json (machine, versions, timings), job.yaml, runs.jsonl.gz (one JSON line per run:
config, sample index, params, record), dt_check.jsonl.gz, verify.json. Plain JSON: safe to open anywhere.
"""
import gzip
import importlib.util
import json
import math
import os
import platform
import sys
import time
import zipfile

import numpy as np

_SIM = None
_ENTRY = None


def _load_module(path):
    name = os.path.splitext(os.path.basename(path))[0]      # plain name: numba disk cache stays valid for normal imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _init_worker(path, entry):
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    global _SIM, _ENTRY
    _SIM = _load_module(path)
    _ENTRY = getattr(_SIM, entry)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def _run_chunk(chunk):
    """chunk: list of (key, params, dt). Returns list of result dicts."""
    out = []
    for key, params, dt in chunk:
        t0 = time.time()
        try:
            rec = _ENTRY(params, dt)
            err = None
        except Exception as exc:  # a failed sample is data, not a crash of the whole study
            rec, err = None, f"{type(exc).__name__}: {exc}"
        out.append(dict(key=key, params=params, dt=dt, record=_jsonable(rec), error=err,
                        seconds=round(time.time() - t0, 4)))
    return out


def load_job(path):
    import yaml
    with open(path) as fh:
        job = yaml.safe_load(fh)
    if job.get("simblocks_batch") != 1:
        raise SystemExit(f"{path}: not a simblocks batch job (needs 'simblocks_batch: 1')")
    base = os.path.dirname(os.path.abspath(path))
    job["_sim_path"] = os.path.normpath(os.path.join(base, job["sim"]))
    job["_job_path"] = os.path.abspath(path)
    return job


def make_samples(job, n=None):
    """Input samples over the job's ranges. sampler: sobol (default; scrambled, space-filling, best coverage for
    few samples), lhs (Latin hypercube) or random."""
    s = job.get("samples", {})
    n = int(n or s.get("n", 64))
    seed = int(s.get("seed", 0))
    uni = s.get("uniform", {})
    keys = list(uni)
    lo = np.array([float(uni[k][0]) for k in keys]); hi = np.array([float(uni[k][1]) for k in keys])
    kind = s.get("sampler", "sobol")
    if kind == "random" or not keys:
        U = np.random.default_rng(seed).random((n, len(keys)))
    else:
        from scipy.stats import qmc
        eng = qmc.Sobol(len(keys), scramble=True, seed=seed) if kind == "sobol" else \
            qmc.LatinHypercube(len(keys), seed=seed)
        if kind == "sobol":
            m = int(math.ceil(math.log2(max(n, 2))))
            U = eng.random_base2(m)[:n]           # first n points of a 2^m Sobol set
        else:
            U = eng.random(n)
    X = lo + U * (hi - lo)
    return [{k: float(X[i, j]) for j, k in enumerate(keys)} for i in range(n)]


def _tasks(job, samples):
    base = job.get("base", {}) or {}
    cfgs = job.get("configs") or {"nominal": {}}
    dt = float(job["dt"])
    tasks = []
    for i, s in enumerate(samples):                    # sample-major order: partial results stay balanced
        for c, ov in cfgs.items():
            tasks.append((f"run|{c}|{i}", {**base, **s, **(ov or {})}, dt))
    dc = job.get("dt_check")
    if dc:
        c = dc.get("config", next(iter(cfgs)))
        for i in range(min(int(dc.get("n", 20)), len(samples))):
            for f in dc.get("factors", [2, 4]):
                tasks.append((f"dt|{c}|{i}|{f}", {**base, **samples[i], **(cfgs[c] or {})}, dt / float(f)))
    return tasks


def _machine():
    info = dict(python=sys.version.split()[0], platform=platform.platform(), processor=platform.processor(),
                cpu_count=os.cpu_count())
    try:
        import numba
        info["numba"] = numba.__version__
    except ImportError:
        pass
    info["numpy"] = np.__version__
    return info


def run_batch(job_path, workers=None, n=None, out=None, resume=True, chunk=None, quiet=False):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    job = load_job(job_path)
    samples = make_samples(job, n)
    tasks = _tasks(job, samples)
    name = job.get("name", "study")
    out = out or os.path.join(os.path.dirname(job["_job_path"]), f"{name}_results.zip")
    partial = out + ".partial.jsonl"
    done = {}
    if resume and os.path.exists(partial):
        with open(partial) as fh:
            for line in fh:
                try:
                    r = json.loads(line); done[r["key"]] = r
                except json.JSONDecodeError:
                    pass                                  # a line cut off by an interrupt
    todo = [t for t in tasks if t[0] not in done]
    workers = int(workers or os.cpu_count() or 1)
    log = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))
    log(f"{name}: {len(tasks)} runs ({len(samples)} samples x {len(job.get('configs') or [1])} configs"
        f"{' + dt check' if job.get('dt_check') else ''}), {len(done)} already done, {workers} workers")

    # compile once in this process so every worker loads numba's on-disk cache instead of compiling
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    t0 = time.time()
    _init_worker(job["_sim_path"], job.get("entry", "batch_record"))
    warm = getattr(_SIM, "warmup", None)
    if warm:
        warm()
    log(f"compiled in {time.time() - t0:.1f}s")

    chunk = int(chunk or max(1, min(8, len(todo) // (workers * 6) or 1)))
    chunks = [todo[i:i + chunk] for i in range(0, len(todo), chunk)]
    t_start = time.time(); n_done = 0; last = 0.0
    with open(partial, "a") as fh, ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                                        initargs=(job["_sim_path"], job.get("entry", "batch_record"))) as ex:
        futs = [ex.submit(_run_chunk, c) for c in chunks]
        for f in as_completed(futs):
            for r in f.result():
                fh.write(json.dumps(r) + "\n"); done[r["key"]] = r
            fh.flush()
            n_done += len(f.result())
            now = time.time()
            if now - last > 5 or n_done == len(todo):
                rate = n_done / (now - t_start)
                eta = (len(todo) - n_done) / rate if rate > 0 else float("nan")
                log(f"  {n_done}/{len(todo)} runs  {rate:.2f} runs/s  ETA {eta/60:.1f} min")
                last = now
    wall = time.time() - t_start

    verify_json = None
    vj = job.get("verify")
    if vj:
        log("verification of one run (sequential)...")
        from .diagnostics.verify import verify_all
        mk = getattr(_SIM, vj.get("entry", "make_spec"))
        pick = getattr(_SIM, vj.get("pick", "pick_nominal"), None)
        runs = [r for k, r in done.items() if k.startswith("run|")]
        params = pick(runs) if pick else runs[0]["params"]
        kw = dict(uncertain=job.get("samples", {}).get("uniform", {}))
        if "mc_n" in vj:
            kw["mc_n"] = int(vj["mc_n"])
        rep = verify_all(mk(params, **kw))
        verify_json = rep.to_json() if hasattr(rep, "to_json") else None
        log(rep.summary())

    meta = dict(name=name, job=os.path.basename(job["_job_path"]), sim=os.path.basename(job["_sim_path"]),
                n_samples=len(samples), n_runs=len(tasks), workers=workers, wall_seconds=round(wall, 1),
                errors=sum(1 for r in done.values() if r["error"]), machine=_machine(),
                created=time.strftime("%Y-%m-%d %H:%M:%S"))
    runs = [done[t[0]] for t in tasks if t[0] in done and t[0].startswith("run|")]
    dtc = [done[t[0]] for t in tasks if t[0] in done and t[0].startswith("dt|")]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("meta.json", json.dumps(meta, indent=2))
        z.write(job["_job_path"], "job.yaml")
        z.writestr("runs.jsonl.gz", gzip.compress("\n".join(json.dumps(r) for r in runs).encode()))
        if dtc:
            z.writestr("dt_check.jsonl.gz", gzip.compress("\n".join(json.dumps(r) for r in dtc).encode()))
        if verify_json:
            z.writestr("verify.json", verify_json if isinstance(verify_json, str) else json.dumps(verify_json))
    if len(runs) + len(dtc) == len(tasks):
        os.remove(partial)
    log(f"wrote {out}  ({len(runs)} runs, {meta['errors']} errors, {wall/60:.1f} min on {workers} workers)")
    return out


def load_results(path):
    """Read a results zip back: dict(meta, runs, dt_check, verify)."""
    with zipfile.ZipFile(path) as z:
        meta = json.loads(z.read("meta.json"))
        runs = [json.loads(l) for l in gzip.decompress(z.read("runs.jsonl.gz")).decode().splitlines() if l]
        dtc = []
        if "dt_check.jsonl.gz" in z.namelist():
            dtc = [json.loads(l) for l in gzip.decompress(z.read("dt_check.jsonl.gz")).decode().splitlines() if l]
        ver = json.loads(z.read("verify.json")) if "verify.json" in z.namelist() else None
        import yaml
        job = yaml.safe_load(z.read("job.yaml")) if "job.yaml" in z.namelist() else {}
    return dict(meta=meta, runs=runs, dt_check=dtc, verify=ver, job=job)


def summarize(path):
    R = load_results(path)
    m = R["meta"]
    print(f"{m['name']}: {m['n_runs']} runs on {m['workers']} workers in {m['wall_seconds']/60:.1f} min, "
          f"{m['errors']} errors ({m['machine'].get('cpu_count')} CPUs, {m['machine'].get('platform')})")
    by = {}
    for r in R["runs"]:
        c = r["key"].split("|")[1]
        by.setdefault(c, []).append(r)
    for c, rs in by.items():
        outs = [r["record"]["out"] for r in rs if r["record"] and "out" in r["record"]]
        line = f"  {c:<26} n={len(rs):<5}"
        for k in ("fast_double", "double_hit", "valid"):
            if outs and k in outs[0]:
                line += f" {k}={np.mean([o[k] for o in outs]):.3f}"
        if outs and "_ledger_rel" in outs[0]:
            line += f" worst_ledger={max(abs(o['_ledger_rel']) for o in outs)*100:.2f}%"
        print(line)
    if R["dt_check"]:
        print(f"  dt check: {len(R['dt_check'])} reruns at finer steps")
    return R
