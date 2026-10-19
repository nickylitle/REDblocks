"""Stale numba caches. numba keys a cached function on its own source file only, so an engine that calls into
circuit.py keeps running the old circuit code after circuit.py changes (a git pull, an edit). On import this hashes
every .py file in the package; when the hash differs from the one stamped beside the caches, every cached .nbi/.nbc
file in the package is removed and the engines recompile once (tens of seconds) on their next run."""
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def _cache_dirs():
    yield HERE
    extra = os.environ.get("NUMBA_CACHE_DIR")
    if extra and os.path.isdir(extra):
        yield extra


def source_hash():
    h = hashlib.sha1()
    for root, dirs, files in os.walk(HERE):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for f in sorted(files):
            if f.endswith(".py"):
                fp = os.path.join(root, f)
                h.update(os.path.relpath(fp, HERE).encode())
                try:
                    with open(fp, "rb") as fh:
                        h.update(fh.read())
                except OSError:
                    pass
    return h.hexdigest()


def check():
    try:
        stamp = os.path.join(HERE, "__pycache__", "numba_source.sha1")
        now = source_hash()
        try:
            with open(stamp) as f:
                if f.read().strip() == now:
                    return False
        except OSError:
            pass
        for top in _cache_dirs():
            for root, _, files in os.walk(top):
                for f in files:
                    if f.endswith((".nbi", ".nbc")):
                        try:
                            os.remove(os.path.join(root, f))
                        except OSError:
                            pass
        os.makedirs(os.path.dirname(stamp), exist_ok=True)
        with open(stamp, "w") as f:
            f.write(now)
        return True
    except Exception:           # a read-only install (frozen app): the caches live elsewhere or not at all
        return False
