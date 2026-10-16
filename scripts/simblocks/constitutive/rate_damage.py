"""Rate effects, hyperelasticity and damage kernels."""
import math

from ..registry import block
from .._jit import kernel


@block("constitutive", "rate", "metal", "yield_scale", kind="kernel", source="Cowper & Symonds 1957",
       tags=("strain rate", "dynamic yield"))
@kernel
def cowper_symonds_factor(rate, D, q):
    """Dynamic/static yield ratio 1 + (rate/D)^(1/q). Mild steel D=40.4, q=5; Al D=6500, q=4.
    Acts on YIELD, not on E."""
    if rate <= 0.0:
        return 1.0
    return 1.0 + (rate / D) ** (1.0 / q)


@block("constitutive", "hyperelastic", "elastomer", "uniaxial", kind="kernel", source="neo-Hookean, incompressible",
       tags=("urethane", "TPU", "wheel", "bumper"))
@kernel
def neo_hookean_uniaxial(stretch, mu):
    """Engineering (nominal) uniaxial stress of an incompressible neo-Hookean solid: mu (lam - lam^-2)."""
    return mu * (stretch - 1.0 / (stretch * stretch))


@block("constitutive", "hyperelastic", "elastomer", "uniaxial", kind="kernel", source="Mooney-Rivlin 2-term",
       tags=("urethane", "TPU"))
@kernel
def mooney_rivlin_uniaxial(stretch, C10, C01):
    """Nominal uniaxial stress 2 (lam - lam^-2)(C10 + C01/lam)."""
    return 2.0 * (stretch - 1.0 / (stretch * stretch)) * (C10 + C01 / stretch)


@block("constitutive", "damage", "metal", "fracture_strain", kind="kernel", source="Johnson & Cook 1985",
       tags=("failure", "erosion"))
@kernel
def jc_fracture_strain(triax, rate_star, T_star, D1, D2, D3, D4, D5):
    """JC failure strain eps_f = [D1 + D2 exp(D3 sigma*)][1 + D4 ln rate*][1 + D5 T*]."""
    r = 1.0 + D4 * math.log(rate_star) if rate_star > 1.0 else 1.0
    return (D1 + D2 * math.exp(D3 * triax)) * r * (1.0 + D5 * T_star)


@block("constitutive", "damage", "general", "accumulation", kind="kernel", source="Miner-type linear damage",
       tags=("failure", "erosion"))
@kernel
def damage_increment(D, d_eps_p, eps_f):
    """D += d_eps_p / eps_f; failure when D >= 1. Returns new D."""
    return D + d_eps_p / eps_f


@block("constitutive", "viscous", "general", "dashpot", kind="kernel", source="linear viscous")
@kernel
def dashpot_force(v, c):
    """Linear dashpot F = -c v."""
    return -c * v


@block("constitutive", "hyperelastic", "elastomer", "uniaxial_any", source="neo-Hookean; Mooney 1940, Rivlin 1948",
       tags=("rubber", "urethane", "elastomer", "hyperelastic"))
def hyperelastic_stress(model, stretch, mu=None, C10=None, C01=0.0):
    """Nominal (engineering) uniaxial stress of an incompressible hyperelastic solid at stretch lam.

    model
      neo_hookean    mu (shear modulus, about E/3):  mu (lam - lam^-2)
      mooney_rivlin  C10, C01:                        2 (lam - lam^-2)(C10 + C01/lam)
    Returns the nominal stress (Pa)."""
    key = str(model).strip().lower()
    if key == "neo_hookean":
        if mu is None:
            raise ValueError("hyperelastic_stress 'neo_hookean' needs mu")
        return getattr(neo_hookean_uniaxial, "py_func", neo_hookean_uniaxial)(stretch, mu)
    if key == "mooney_rivlin":
        if C10 is None:
            raise ValueError("hyperelastic_stress 'mooney_rivlin' needs C10 (and C01)")
        return getattr(mooney_rivlin_uniaxial, "py_func", mooney_rivlin_uniaxial)(stretch, C10, C01)
    raise ValueError(f"hyperelastic_stress: unknown model {model!r}; use neo_hookean or mooney_rivlin")
