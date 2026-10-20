"""Spectra of a time history, for loads that go into a frequency-domain analysis (Ansys harmonic, random vibration
and response-spectrum analyses take these directly):

- fft: one-sided amplitude spectrum |X(f)| (a sine of amplitude A shows as A at its frequency), with a Hann window
  corrected for its amplitude loss.
- psd: power spectral density by Welch's method (Hann segments, 50 % overlap), in unit^2/Hz. Its integral over
  frequency is the mean square of the signal (Parseval), which the tests check.
- srs: shock response spectrum: the peak absolute acceleration of a single-degree-of-freedom oscillator (natural
  frequency f, damping from Q) mounted on the signal, at each f. Computed with Smallwood's ramp-invariant digital
  filter (ISO 18431-4), the standard method. Maximax: the largest response, during the shock or after it (the signal
  is padded with quiet so the residual ring-down is counted).

All three resample a non-uniform history onto a uniform grid first (linear interpolation at its median step).
"""
import math

import numpy as np

KINDS = ("fft", "psd", "srs")


def uniform(t, x):
    """(dt, x on a uniform grid) from a possibly uneven history."""
    t = np.asarray(t, float); x = np.asarray(x, float)
    if len(t) < 4:
        raise ValueError("a spectrum needs at least 4 samples")
    d = np.diff(t)
    dt = float(np.median(d[d > 0])) if (d > 0).any() else 1.0
    if np.allclose(d, dt, rtol=1e-6, atol=0):
        return dt, x
    n = int(math.floor((t[-1] - t[0]) / dt)) + 1
    tu = t[0] + dt * np.arange(n)
    return dt, np.interp(tu, t, x)


def fft_amplitude(t, x, detrend=True):
    """(f, |X|): one-sided amplitude spectrum, Hann window, amplitude-corrected."""
    dt, y = uniform(t, x)
    if detrend:
        y = y - y.mean()
    n = len(y); w = np.hanning(n)
    X = np.fft.rfft(y * w)
    amp = 2.0 * np.abs(X) / w.sum()
    amp[0] /= 2.0
    return np.fft.rfftfreq(n, dt), amp


def psd_welch(t, x, segment=None, detrend=True):
    """(f, PSD): Welch, Hann, 50 % overlap. segment: seconds per segment (default: an eighth of the record)."""
    from scipy.signal import welch
    dt, y = uniform(t, x)
    n = len(y)
    nper = n if segment is None else max(8, min(n, int(round(segment / dt))))
    if segment is None:
        nper = max(8, n // 8)
    f, P = welch(y, fs=1.0 / dt, window="hann", nperseg=nper, noverlap=nper // 2,
                 detrend="constant" if detrend else False, scaling="density")
    return f, P


def srs(t, a, freqs, Q=10.0, pad=True):
    """(f, peak absolute response acceleration) for a base acceleration a(t), Smallwood ramp-invariant filter."""
    from scipy.signal import lfilter
    dt, y = uniform(t, a)
    freqs = np.asarray(freqs, float)
    if pad:                                          # the residual ring-down after the shock counts too
        fmin = max(float(freqs.min()), 1e-9)
        y = np.concatenate([y, np.zeros(int(min(2e6, math.ceil(2.0 / fmin / dt))))])
    z = 1.0 / (2.0 * Q)
    out = np.zeros(len(freqs))
    for i, fn in enumerate(freqs):
        wn = 2 * math.pi * fn; wd = wn * math.sqrt(1 - z * z)
        E = math.exp(-z * wn * dt); K = wd * dt; C = E * math.cos(K); S = E * math.sin(K); Sp = S / K
        b = [1 - Sp, 2 * (Sp - C), E * E - Sp]
        den = [1, -2 * C, E * E]
        out[i] = np.max(np.abs(lfilter(b, den, y)))
    return freqs, out


def log_freqs(f_min, f_max, per_octave=12):
    n = max(2, int(round(math.log2(f_max / f_min) * per_octave)) + 1)
    return f_min * (f_max / f_min) ** (np.arange(n) / (n - 1))


def spectrum(kind, t, x, f_min=None, f_max=None, Q=10.0, segment=None, per_octave=12):
    """(f, values, unit suffix) for one column. unit suffix: '' (fft, same unit), '^2/Hz' (psd), '' (srs)."""
    if kind not in KINDS:
        raise ValueError(f"spectrum kind is one of {', '.join(KINDS)}")
    dt, _ = uniform(t, x)
    nyq = 0.5 / dt
    if kind == "srs":
        T = float(np.asarray(t)[-1] - np.asarray(t)[0])
        lo = f_min or max(1.0 / max(T, 1e-12), 1e-3)
        hi = min(f_max or nyq / 5.0, nyq / 5.0)        # the filter is accurate to about a fifth of the sample rate
        if hi <= lo:
            raise ValueError("the SRS range is empty: log more points (points: all) or lower f_min")
        f, v = srs(t, x, log_freqs(lo, hi, per_octave), Q)
        return f, v, ""
    f, v = fft_amplitude(t, x) if kind == "fft" else psd_welch(t, x, segment)
    m = np.ones(len(f), bool)
    if f_min is not None:
        m &= f >= f_min
    if f_max is not None:
        m &= f <= f_max
    return f[m], v[m], ("^2/Hz" if kind == "psd" else "")
