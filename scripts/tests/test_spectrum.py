"""Spectrum exports (FFT, PSD, SRS) for frequency-domain loads (2026-10-20)."""
import math
import os
import numpy as np
import pytest
from simblocks.diagnostics.spectrum import fft_amplitude, psd_welch, srs
from simblocks.compiler import load_system
from simblocks.compiler.loader import SpecError


def test_fft_reads_sine_amplitudes():
    t = np.arange(0, 2, 1e-4); x = 3 * np.sin(2 * np.pi * 50 * t) + 1.5 * np.sin(2 * np.pi * 120 * t)
    f, a = fft_amplitude(t, x)
    assert a[np.argmin(abs(f - 50))] == pytest.approx(3.0, rel=1e-3)
    assert a[np.argmin(abs(f - 120))] == pytest.approx(1.5, rel=1e-3)


def test_psd_integrates_to_mean_square():
    rng = np.random.default_rng(0); t = np.arange(0, 4, 1e-4); x = rng.normal(0, 2.0, len(t))
    f, P = psd_welch(t, x)
    assert np.trapezoid(P, f) == pytest.approx(np.mean((x - x.mean()) ** 2), rel=0.02)


def test_srs_of_a_half_sine():
    # 100 gn, 11 ms half-sine (the classic shock test): the maximax SRS peaks near 1.7x around f T ~ 0.8 and
    # tends to the input peak at high frequency, and to small values at low frequency
    T = 0.011; t = np.arange(0, 0.05, 1e-5); a = np.where(t < T, 100 * np.sin(np.pi * t / T), 0.0)
    f, s = srs(t, a, np.array([5.0, 73.0, 3000.0]), Q=10)
    assert 1.6 < s[1] / 100 < 1.75
    assert s[2] == pytest.approx(100, rel=0.02)
    assert s[0] < 0.25 * 100


def test_spectrum_export_end_to_end(tmp_path):
    y = """simblocks: 1
name: msd
dim: 1d
bodies:
  mass: {dof: translation, mass: 1 kg, x0: 10 mm}
elements:
  spring: {type: spring, k: 400 N/m, measure: mass}
  damper: {type: dashpot, c: 0.4 N s/m, measure: mass}
outputs:
  x: {body: mass, quantity: position, reduce: max}
exports:
  - {name: fft, spectrum: fft, columns: [{body: mass, quantity: position, unit: mm}], folder: out}
  - {name: srs, spectrum: {kind: srs, f_min: 0.5 Hz, f_max: 50 Hz}, columns: [{body: mass, quantity: acceleration, unit: gn}], folder: out}
run: {t_end: 20 s}
"""
    p = tmp_path / "m.yaml"; p.write_text(y)
    o, H, paths = load_system(str(p)).run_with_exports(model_path=str(p))
    fft = np.loadtxt(paths[0], delimiter=",", skiprows=1); s = np.loadtxt(paths[1], delimiter=",", skiprows=1)
    fn = 20 / (2 * math.pi)
    assert fft[np.argmax(fft[:, 1]), 0] == pytest.approx(fn, abs=0.06)
    assert open(paths[1]).readline().startswith("Frequency [Hz],mass acceleration SRS Q=10 [gn]")
    assert s[np.argmax(s[:, 1]), 0] == pytest.approx(fn, rel=0.05)
    assert 1.5 < s[:, 1].max() < 4.0            # 0.41 gn sine, amplified by the 5 % damped oscillator


def test_bad_spectrum_is_refused(tmp_path):
    y = "simblocks: 1\nname: t\ndim: 1d\nbodies: {m: {dof: translation, mass: 1 kg}}\nelements: {s: {type: spring, k: 1, measure: m}}\n" \
        "outputs: {x: {body: m, quantity: position}}\nexports: [{name: a, spectrum: wavelet, columns: [{body: m}]}]\n"
    with pytest.raises(SpecError):
        load_system(y)
