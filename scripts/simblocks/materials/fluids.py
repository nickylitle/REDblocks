"""Fluids and temperature-dependent air properties (for aero and thermal blocks)."""
import math
from .base import Fluid
from ..registry import block

R_AIR = 287.058      # J/kg/K
GAMMA_AIR = 1.4


@block("materials", "fluids", "air", "temperature", source="Sutherland 1893, ideal gas",
       tags=("air", "viscosity", "density"))
def air_properties(T=293.15, p=101325.0):
    """Dry air at temperature T (K) and pressure p (Pa): returns Fluid with rho, mu, k, cp."""
    mu = 1.716e-5 * (T / 273.15) ** 1.5 * (273.15 + 110.4) / (T + 110.4)
    k = 0.0241 * (T / 273.15) ** 1.5 * (273.15 + 194.0) / (T + 194.0)
    cp = 1005.0
    return Fluid(f"air {T:.0f} K", p / (R_AIR * T), mu, k, cp, "Sutherland")


FLUIDS = {
    "air_20C": air_properties(293.15),
    "water_20C": Fluid("water 20 C", 998.2, 1.002e-3, 0.598, 4182.0, "IAPWS typical"),
    "oil_SAE30": Fluid("SAE 30 oil 40 C", 875.0, 0.08, 0.145, 1900.0, "typical"),
}
