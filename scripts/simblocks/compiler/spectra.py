"""Spectra of a recorded history: the frequency content (FFT) and the shock response spectrum (SRS).

fft   single-sided amplitude spectrum, |Y(f)| * 2 / N on a uniform resample of the history (no window: a hit is a
      transient that starts and ends inside the record). A sine of amplitude A shows a peak of A.
srs   maximax absolute-acceleration shock response spectrum: for each natural frequency f, the largest response
      of a damped single-degree-of-freedom system (zeta, default 5 %, Q = 10) whose base follows the input. Smallwood's
      ramp-invariant recursive filter (Smallwood 1981, the ISO 18431-4 method). Frequencies at 1/12 octave. Read it
      as: a part on this mount with natural frequency f sees this peak acceleration. At high f it tends to the input's
      own peak."""
import math

import numpy as np


def _uniform(t, y):
    t = np.asarray(t, float); y = np.asarray(y, float)
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, y = t[keep], y[keep]
    if len(t) < 4:
        raise ValueError("the history is too short for a spectrum")
    dt = float(np.median(np.diff(t)))
    tu = np.arange(t[0], t[-1] + 0.5 * dt, dt)
    return tu, np.interp(tu, t, y), dt


def _thin(f, a, n=1500):
    if len(f) <= n:
        return f, a
    edges = np.unique(np.geomspace(max(f[1], 1e-12), f[-1], n))
    idx = np.searchsorted(f, edges)
    fo, ao = [], []
    for i0, i1 in zip(idx[:-1], idx[1:]):
        if i1 > i0:
            j = i0 + int(np.argmax(a[i0:i1])); fo.append(f[j]); ao.append(a[j])   # keep each bin's peak
    return np.array(fo), np.array(ao)


def fft(t, y):
    tu, yu, dt = _uniform(t, y)
    N = len(yu)
    Y = np.fft.rfft(yu); f = np.fft.rfftfreq(N, dt)
    amp = np.abs(Y) * 2.0 / N; amp[0] /= 2.0
    fo, ao = _thin(f[1:], amp[1:])
    k = int(np.argmax(amp[1:])) + 1
    return dict(f=fo.tolist(), amp=ao.tolist(), peak_f=float(f[k]), peak_amp=float(amp[k]), df=float(f[1]),
                f_max=float(f[-1]))


def srs(t, y, zeta=0.05, f_lo=None, f_hi=None, per_octave=12):
    from scipy.signal import lfilter
    tu, x, dt = _uniform(t, y)
    T = tu[-1] - tu[0]
    f_lo = f_lo or max(1.0 / T, 1e-3)
    f_hi = f_hi or 1.0 / (8.0 * dt)                     # 8 points per cycle at the top
    n = int(math.log2(f_hi / f_lo) * per_octave) + 1
    fs = f_lo * 2.0 ** (np.arange(max(n, 2)) / per_octave)
    pad = np.concatenate([x, np.zeros(int(1.0 / (fs[0] * dt)) + 1)])     # ring-down after the input ends
    out = np.zeros(len(fs))
    for i, f in enumerate(fs):
        w = 2 * math.pi * f; wd = w * math.sqrt(1 - zeta * zeta)
        E = math.exp(-zeta * w * dt); K = wd * dt; C = E * math.cos(K); S = E * math.sin(K); Sp = S / K
        b = [1 - Sp, 2 * (Sp - C), E * E - Sp]; a = [1.0, -2 * C, E * E]
        out[i] = float(np.max(np.abs(lfilter(b, a, pad))))
    return dict(f=fs.tolist(), amp=out.tolist(), zeta=zeta, peak_input=float(np.max(np.abs(x))))
