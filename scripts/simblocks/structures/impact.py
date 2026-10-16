"""Puncture / push-through capacity of plates (quasi-static shear plugging).
Punching-shear capacity is the classical blanking/punching force used in die design:
F = perimeter * t * S_su, with ultimate shear S_su ~ 0.67 S_ut for steels and ~0.6 S_ut for aluminum
(Shigley 10e quotes S_su ~ 0.67 S_ut in its fastener/shear discussion). Quasi-static: dynamic
plugging needs the rate factor applied to S_ut. Dynamic (projectile) perforation blocks are pending a
verified-source pass; see references/adding_blocks.md roadmap."""
import math

from ..registry import block


@block("structures", "impact", "plate", "punch_perimeter", source="geometry", tags=("tooth", "puncture"))
def punch_perimeter(shape="rect", w=None, l=None, d=None):
    """Perimeter of the punched patch: rect (w x l) -> 2(w + l); circle (d) -> pi d."""
    return 2.0 * (w + l) if shape == "rect" else math.pi * d


@block("structures", "impact", "plate", "shear_plug_capacity", source="punching shear F = P t S_su (blanking practice)",
       tags=("puncture", "push-through", "tooth"))
def shear_plug_capacity(perimeter, t, su, ssu_ratio=0.67):
    """Force to shear a plug of the given perimeter out of a plate of thickness t: P t (ssu_ratio su)."""
    return perimeter * t * ssu_ratio * su


@block("structures", "impact", "plate", "blunt_punch_line_force",
       source=("Prandtl/Hill flat-punch limit 2k(1 + pi/2) per unit width (Hill, Mathematical Theory of "
               "Plasticity 1950), as applied to blade indentation by Chaijit et al., J. Mech. Mater. Struct. "
               "1(8), 2006, Sec. 3.1"), tags=("tooth", "indentation", "gouge", "dent onset"))
def blunt_punch_line_force(sigma_y, w, L, criterion="mises"):
    """Fully plastic onset force of a flat tooth of tip width w and engaged length L pressed into a thick
    plate: F = 2k(1 + pi/2) w L, k = sigma_y/sqrt(3) (Mises) or sigma_y/2 (Tresca), i.e. ~2.97 sigma_y w L.
    Chaijit et al. apply it for w/t < 0.23; above that the plate bends and dishes instead."""
    k = sigma_y / math.sqrt(3.0) if criterion == "mises" else sigma_y / 2.0
    return 2.0 * k * (1.0 + math.pi / 2.0) * w * L


@block("contact", "ploughing", "wedge_asperity", "friction_ratio",
       source=("Challen & Oxley, Wear 53:229-243, 1979, wave model, as reproduced in Lubricants 6(2):53, 2018 "
               "Eqs. 2-3 (PARTIAL: standard form restored from a garbled extraction)"),
       tags=("ploughing", "gouge", "tangential drag", "slip-line"))
def challen_oxley_mu(theta, f):
    """Tangential/normal ratio for a hard wedge of attack angle theta (rad) ploughing a plastic surface
    with interface shear factor f (0..1). Wave model; theta = 0 gives f / (1 + pi/2 + acos f + sqrt(1-f^2)).
    Multiply by a normal force (e.g. blunt_punch_line_force) to get the ploughing drag."""
    ac = math.acos(f)
    arg = 0.0 if theta == 0.0 else math.sin(theta) / math.sqrt(max(1 - f, 0.0) or 1e-300)
    if arg > 1.0:
        raise ValueError("wave model invalid here (wedge formation or cutting regime): lower theta or f")
    A1 = 1 + math.pi / 2 + ac - 2 * theta - 2 * math.asin(arg)
    return (A1 * math.sin(theta) + math.cos(ac - theta)) / (A1 * math.cos(theta) + math.sin(ac - theta))
