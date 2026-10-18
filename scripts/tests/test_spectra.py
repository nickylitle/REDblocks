import math
import numpy as np
import pytest
from simblocks.compiler.spectra import fft, srs


def test_fft_sine_amplitude():
    t = np.linspace(0, 1, 20001); y = 3.0 * np.sin(2 * math.pi * 50 * t)
    r = fft(t, y)
    assert r["peak_f"] == pytest.approx(50, rel=0.01) and r["peak_amp"] == pytest.approx(3.0, rel=0.02)


def test_srs_half_sine_limits():
    """Half-sine base pulse, 1 ms, 100 g: the SRS tends to the peak at high frequency and peaks near 1.7x."""
    t = np.linspace(0, 0.05, 50001); T = 1e-3
    x = np.where(t < T, 100 * np.sin(math.pi * t / T), 0.0)
    r = srs(t, x, zeta=0.05)
    f = np.array(r["f"]); a = np.array(r["amp"])
    assert a[f > 8000][-1] == pytest.approx(100, rel=0.05)
    assert 1.5 < a.max() / 100 < 1.8                   # the classic half-sine maximax amplification
