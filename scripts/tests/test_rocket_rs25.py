"""rocket_engine against the Space Shuttle Main Engine (RS-25) station data at 104.5 % power.

Source: cryo-rocket.com, SSME thermodynamic data, sections 1.1 (fuel), 1.2 (oxidizer) and 1.3 (combustion),
tabulated from Boeing / Rocketdyne data:
  https://www.cryo-rocket.com/thermodynamic-data/1.2-oxidizer-system/
Mixture ratio from the tank flows: LOX 423.7 kg/s less 0.9 kg/s of ullage gas, LH2 70.3 less 0.5 -> OF 6.06.
Area ratio 77.5 from the nozzle's exit and throat diameters (2.304 m and 0.262 m); 69 is also quoted, but the page's own
sea-level and vacuum thrusts need an exit area of 4.1 m^2, which is the 77.5 nozzle."""
import pytest

from simblocks.actuators.rocket import PROPELLANTS, ideal_performance, rocket_engine

LBF = 4.4482216152605
OF = (423.7 - 0.9) / (70.3 - 0.5)
PC = 2994 * 6894.757          # 104.5 % power level chamber pressure (the page's 19.79 MPa throat value is 6 % below it)


def test_storage_enthalpies_match_the_ssme_tank_stations():
    """Same enthalpy reference as the page (formation basis): LOX station 17 is -404.0 kJ/kg at 90.4 K and 0.69 MPa,
    LH2 station 1 is -4449 kJ/kg at 20.4 K and 0.21 MPa. The block's values are the CEA ones at the normal boiling
    point (90.17 K, 20.27 K, 1 atm); the warmer, pressurised tank liquid sits a little higher."""
    a, mw, h = PROPELLANTS["LOX"]
    assert h / mw / 1e3 == pytest.approx(-404.0, abs=3.0)
    a, mw, h = PROPELLANTS["LH2"]
    assert h / mw / 1e3 == pytest.approx(-4449.0, abs=25.0)


def test_chamber_equilibrium_matches_the_ssme_throat_station():
    """Station 33 (throat stagnation, 19.79 MPa): 3589 K; mole fractions H2O 0.672, H2 0.252, OH 0.036, H 0.028."""
    i = ideal_performance("LH2", OF, 19.79e6, 77.5)
    assert i["Tc"] == pytest.approx(3589.0, rel=0.01)
    x = i["composition"]
    assert x["H2O"] == pytest.approx(0.672, abs=0.03)
    assert x["H2"] == pytest.approx(0.252, abs=0.02)
    assert x["OH"] == pytest.approx(0.036, abs=0.005)
    assert x["H"] == pytest.approx(0.028, abs=0.005)


def test_rs25_thrust_flow_and_sea_level_isp():
    """Rated at 491,900 lbf and 452 s in vacuum, the block should give the page's 398,400 lbf and 366 s at sea level,
    492.1 kg/s through the chamber, and about 0.02 MPa at the nozzle exit."""
    r = rocket_engine("LH2", OF, PC, 77.5, F_vac=491900 * LBF, Isp_vac=452)
    assert r["mdot"] == pytest.approx(492.1, rel=0.005)
    assert r["F_sl"] / LBF == pytest.approx(398400, rel=0.01)
    assert r["Isp_sl"] == pytest.approx(366, rel=0.01)
    assert r["Pe"] == pytest.approx(0.02e6, rel=0.15)
    assert 0.96 < r["efficiency"] < 0.99            # delivered / ideal vacuum Isp: a staged-combustion engine
