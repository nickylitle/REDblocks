"""Stress concentrations: exact elasticity solutions where they exist, notch sensitivity for fatigue.
Kirsch 1898 (circular hole), Inglis 1913 (elliptical hole); Peterson/Neuber notch sensitivity (Shigley 10e 6-35)."""
import math

from ..registry import block


@block("structures", "stress_concentration", "circular_hole", "uniaxial_field", source="Kirsch 1898 (exact, infinite plate)",
       tags=("bolt hole", "lightening hole"))
def kirsch_hoop_stress(S, a, r, theta):
    """Hoop stress around a circular hole of radius a in an infinite plate under far-field uniaxial S (along theta=0).
    At r = a: S (1 - 2 cos 2 theta): Kt = 3 at theta = 90 deg."""
    return S / 2 * (1 + a * a / (r * r)) - S / 2 * (1 + 3 * a ** 4 / r ** 4) * math.cos(2 * theta)


@block("structures", "stress_concentration", "elliptical_hole", "uniaxial", source="Inglis 1913 (exact)",
       tags=("slot", "crack-like"))
def kt_ellipse(a_perp, b_par):
    """Kt = 1 + 2 a/b for an elliptical hole with semi-axis a perpendicular to the load and b parallel.
    Equivalent: 1 + 2 sqrt(a/rho) with tip radius rho = b^2/a."""
    return 1.0 + 2.0 * a_perp / b_par


@block("structures", "stress_concentration", "general", "notch_sensitivity", source="Shigley 10e eq. 6-33/6-35 (Neuber-Kuhn, steel bending)",
       tags=("fatigue", "Kf"))
def fatigue_kf(Kt, r, Sut):
    """Fatigue notch factor Kf = 1 + q (Kt - 1), q = 1/(1 + sqrt(a)/sqrt(r)), sqrt(a) [sqrt(in)] from Sut (kpsi) fit.
    Valid for steels, bending/axial, 50 <= Sut <= 250 kpsi. r in m."""
    s = Sut / 6.894757e6 / 1000
    sqrt_a = 0.246 - 3.08e-3 * s + 1.51e-5 * s * s - 2.67e-8 * s ** 3
    q = 1.0 / (1.0 + sqrt_a / math.sqrt(r / 0.0254))
    return 1.0 + q * (Kt - 1.0)
