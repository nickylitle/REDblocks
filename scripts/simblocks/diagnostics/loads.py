"""Load histories for other solvers: measure a force pulse, build reference pulses, and export a simplified
time-force table (FE / MATLAB transient structural) that keeps the impulse.

Typical use: a detailed explicit sim gives the force a flexible part transmits; a separate structural model
treats that part as rigid and needs the force applied at one point as a table of (t, Fx, Fy, Fz).
"""
import math

import numpy as np

from ..registry import block


def _trapz(y, x):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


@block("diagnostics", "loads", "force_history", "pulse_metrics", source="trapezoidal integration",
       tags=("impulse", "peak", "load pulse", "FE export"))
def pulse_metrics(t, F, frac=0.01):
    """Impulse (trapezoidal), peak, time of peak, start/end where |F| first/last exceeds frac x peak, duration,
    impulse centroid time and the equivalent half-sine duration with the same peak and impulse
    (pi J / (2 F_peak))."""
    t = np.asarray(t, float); F = np.asarray(F, float)
    J = _trapz(F, t)
    i = int(np.argmax(np.abs(F))); pk = float(F[i])
    on = np.nonzero(np.abs(F) > frac * abs(pk))[0] if pk != 0 else np.array([], int)
    t0 = float(t[on[0]]) if on.size else float("nan"); t1 = float(t[on[-1]]) if on.size else float("nan")
    cen = _trapz(F * t, t) / J if J != 0 else float("nan")
    return dict(impulse=J, peak=pk, t_peak=float(t[i]), t_start=t0, t_end=t1, duration=t1 - t0, centroid=cen,
                half_sine_equiv_duration=math.pi * J / (2 * pk) if pk else float("nan"))


@block("diagnostics", "loads", "force_history", "split_pulses", source="threshold segmentation",
       tags=("double hit", "load pulse"))
def split_pulses(t, F, frac=0.02, min_gap=2e-5):
    """Index ranges [(i0, i1), ...] of separate pulses: |F| above frac x overall peak, gaps shorter than
    min_gap (s) merged."""
    t = np.asarray(t, float); F = np.abs(np.asarray(F, float))
    on = F > frac * F.max() if F.max() > 0 else np.zeros_like(F, bool)
    out = []
    i = 0; n = len(F)
    while i < n:
        if on[i]:
            j = i
            while j + 1 < n and on[j + 1]:
                j += 1
            if out and t[i] - t[out[-1][1]] < min_gap:
                out[-1] = (out[-1][0], j)
            else:
                out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


@block("diagnostics", "loads", "reference", "half_sine_train", source="F = F_pk sin(pi (t - t0) / T)",
       tags=("load pulse", "impulse", "reference load"))
def half_sine_train(peaks, durations, starts, t_end, n_per=10):
    """Time and force of a train of half-sine pulses (peak, duration, start time each). Impulse of one pulse:
    2 F_pk T / pi."""
    ts = [0.0]; fs = [0.0]
    for pk, T, t0 in zip(peaks, durations, starts):
        tt = t0 + np.linspace(0, T, n_per + 1)
        ts.extend(tt.tolist()); fs.extend((pk * np.sin(np.pi * (tt - t0) / T)).tolist())
    ts.append(t_end); fs.append(0.0)
    t = np.array(ts); F = np.array(fs)
    o = np.argsort(t, kind="stable")
    return t[o], F[o]


def _rdp(t, Y, tol):
    """Ramer-Douglas-Peucker on several columns at once: keep a point when any column deviates from the chord
    by more than tol (same units as Y)."""
    keep = np.zeros(len(t), bool); keep[0] = keep[-1] = True
    stack = [(0, len(t) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        w = (t[a + 1:b] - t[a]) / (t[b] - t[a])
        chord = Y[a] + np.outer(w, Y[b] - Y[a])
        dev = np.max(np.abs(Y[a + 1:b] - chord), axis=1)
        k = int(np.argmax(dev))
        if dev[k] > tol:
            m = a + 1 + k
            keep[m] = True
            stack.append((a, m)); stack.append((m, b))
    return keep


@block("diagnostics", "loads", "force_history", "fe_table", kind="builder",
       source="Ramer-Douglas-Peucker simplification + per-column impulse correction",
       tags=("FE export", "MATLAB", "ANSYS", "LS-DYNA", "load curve", "impulse"))
def fe_load_table(t, F, tol_frac=0.01, max_points=400):
    """Simplify a sampled force history (F: (n,) or (n, k) columns) to a short piecewise-linear table for
    another solver: keeps every point needed to stay within tol_frac x peak of the original, then scales each
    column so its impulse matches the original exactly. Returns (t_table, F_table, info)."""
    t = np.asarray(t, float); F = np.asarray(F, float)
    Y = F.reshape(len(t), -1)
    pk = float(np.max(np.abs(Y))) or 1.0
    tol = tol_frac * pk
    keep = _rdp(t, Y, tol)
    while keep.sum() > max_points:
        tol *= 1.5; keep = _rdp(t, Y, tol)
    tt, YY = t[keep], Y[keep].copy()
    J0 = np.array([_trapz(Y[:, c], t) for c in range(Y.shape[1])])
    J1 = np.array([_trapz(YY[:, c], tt) for c in range(Y.shape[1])])
    scale = np.where(np.abs(J1) > 1e-15, J0 / np.where(J1 == 0, 1, J1), 1.0)
    YY *= scale
    info = dict(points=int(keep.sum()), tol_N=tol, impulse=J0.tolist(), impulse_scale=scale.tolist(),
                max_dev_N=float(np.max(np.abs(np.array([np.interp(t, tt, YY[:, c]) for c in range(Y.shape[1])]).T - Y))))
    return tt, (YY[:, 0] if F.ndim == 1 else YY), info


@block("diagnostics", "loads", "export", "table_text", kind="builder", source="plain text, one row per time",
       tags=("MATLAB", "FE export", "load curve"))
def write_load_table(path, t, Fx, Fy, Fz, header=True, notes=()):
    """Write 'time  Fx  Fy  Fz' rows (tab separated, SI: s and N), the layout Ansys/MATLAB tabular loads read.
    readmatrix(path) in MATLAB loads it (comment lines start with %). notes: extra comment lines on top."""
    with open(path, "w") as fh:
        for n_ in notes:
            fh.write(f"% {n_}\n")
        if header:
            fh.write("% time_s\tFx_N\tFy_N\tFz_N\n")
        for row in zip(t, Fx, Fy, Fz):
            fh.write("\t".join(f"{v:.6g}" for v in row) + "\n")
    return path
