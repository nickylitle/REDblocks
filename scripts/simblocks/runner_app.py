"""One-click study runner (the entry point of simblocks_runner.exe).

Double-click the executable with a job file (*_job.yaml) and its sim (*.py) in the same folder, or drag a job
file onto it, or run `simblocks_runner job.yaml [--workers N] [--n N]`. It runs the study on every CPU core and
writes <name>_results.zip next to the job file: hand that zip back for analysis and graphing.
"""
import glob
import multiprocessing
import os
import sys
import time


def _here():
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.getcwd()


def _pause():
    if getattr(sys, "frozen", False) and sys.stdin and sys.stdin.isatty():
        try:
            input("\nPress Enter to close.")
        except EOFError:
            pass


def main(argv=None):
    multiprocessing.freeze_support()                       # required for worker processes in a frozen app
    # numba compiles each kernel once and caches it here, so later launches start in seconds
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".simblocks_numba_cache"))
    argv = list(sys.argv[1:] if argv is None else argv)
    import argparse
    ap = argparse.ArgumentParser(prog="simblocks_runner", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs", nargs="*", help="job files (default: every *_job.yaml next to the program)")
    ap.add_argument("--workers", type=int, default=None, help="processes (default: all CPU cores)")
    ap.add_argument("--n", type=int, default=None, help="override samples.n")
    ap.add_argument("--fresh", action="store_true", help="ignore partial results and start over")
    a = ap.parse_args(argv)
    jobs = a.jobs or sorted(glob.glob(os.path.join(_here(), "*_job.yaml")) +
                            glob.glob(os.path.join(_here(), "*_job.yml")))
    if not jobs:
        print(f"No job file found. Put a *_job.yaml (and the sim .py it names) next to this program:\n  {_here()}")
        _pause()
        return 1
    from simblocks.batch import run_batch
    code = 0
    for j in jobs:
        print(f"\n=== {os.path.basename(j)} ===")
        t0 = time.time()
        try:
            out = run_batch(j, workers=a.workers, n=a.n, resume=not a.fresh)
            print(f"\nDONE in {(time.time() - t0) / 60:.1f} min. Send this file back:\n  {out}")
        except Exception as exc:                            # keep the window open with the reason
            import traceback
            traceback.print_exc()
            print(f"\nFAILED: {exc}")
            code = 2
    _pause()
    return code


if __name__ == "__main__":
    sys.exit(main())
