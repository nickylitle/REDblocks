"""Polymers, elastomers, printed and laminated sheet. Typical datasheet values at 23 C.
Polymers are strongly rate- and temperature-dependent; sy here is the quasi-static yield/flow stress.
Printed materials are XY (in-plane) values. Z-direction strength is typically 30-60% of XY."""
from .base import Isotropic
from ..registry import block

GPa, MPa = 1e9, 1e6


@block("materials", "conversions", "elastomer", "hardness", kind="analytic",
       source="Gent 1958 (ASTM D2240 correlation)", tags=("shore", "urethane", "durometer"))
def shore_A_to_E(shore_A):
    """Young's modulus (Pa) of an elastomer from Shore A durometer (Gent correlation, valid ~20-95A)."""
    S = float(shore_A)
    return 1e6 * 0.0981 * (56.0 + 7.62336 * S) / (0.137505 * (254.0 - 2.54 * S))


def _poly(name, E, nu, rho, sy, su, **kw):
    kw.setdefault("family", "polymer")
    return Isotropic(name, E, nu, rho, sy, su, **kw)


def _elastomer(name, shore, rho, su, elong, **kw):
    E = shore_A_to_E(shore)
    return Isotropic(name, E, 0.49, rho, su, su, family="elastomer", elong=elong,
                     hardness=f"Shore {shore}A", note="E from Gent correlation; hyperelastic, use neo-Hookean kernel for big strains", **kw)


POLYMERS = {
    "UHMW": _poly("UHMW-PE (TIVAR 1000 sheet)", 0.75 * GPa, 0.46, 930, 19 * MPa, 40 * MPa, elong=3.0, k=0.41,
                  cp=1900.0, alpha=200e-6,
                  note="MCAM TIVAR 1000 TDS 04/2024: E, yield ISO 527; su, elong ASTM D638 (ASTM E 0.552 GPa). "
                       "nu UNVERIFIED"),
    "HDPE": _poly("HDPE sheet (Proteus HDPE)", 1.45 * GPa, 0.46, 960, 31.7 * MPa, 31.7 * MPa, elong=4.0, k=0.48,
                  cp=1900.0, alpha=120e-6,
                  note="MCAM Proteus HDPE TDS 04/2024 ASTM; yield is the max stress. nu UNVERIFIED"),
    "PC": _poly("Polycarbonate sheet (Lexan 9034 / Makrolon GP)", 2.38 * GPa, 0.37, 1200, 62.1 * MPa, 65.5 * MPa,
                elong=1.1, k=0.20, cp=1200.0, alpha=68e-6,
                note="SABIC Lexan 9034 TDS (ASTM D638). Yield is strongly rate dependent (ARL-TR-5899)"),
    "PA66_dry": _poly("Nylon 6/6 dry as molded (Zytel 101L)", 3.1 * GPa, 0.37, 1140, 82 * MPa, 82 * MPa, elong=0.25,
                      k=0.25, cp=1670.0, alpha=80e-6,
                      note="Celanese Zytel 101L NC010 TDS 2023 (ISO). Conditioned 50% RH: E 1.4 GPa, yield 55 MPa. "
                           "Rate: sy = 112.67 + 14.67 log10(rate) MPa compression (Gomez-del Rio 2024)"),
    "PA66_cond": _poly("Nylon 6/6 conditioned 50% RH (Zytel 101L)", 1.4 * GPa, 0.43, 1140, 55 * MPa, 55 * MPa,
                       elong=0.5, k=0.25, cp=1670.0, alpha=80e-6,
                       note="Celanese Zytel 101L NC010 TDS cond. column; a part in room air drifts here over weeks. "
                            "elong UNVERIFIED"),
    "POM": _poly("Acetal homopolymer (Delrin 150)", 3.1 * GPa, 0.35, 1420, 73 * MPa, 73 * MPa, elong=0.4, k=0.33,
                 cp=1460.0, alpha=110e-6, note="DuPont Delrin 150 NC010 TDS-71946 (ISO). nu UNVERIFIED"),
    "PEEK": _poly("PEEK unfilled (Victrex 450G)", 4.0 * GPa, 0.38, 1300, 98 * MPa, 98 * MPa, elong=0.25, k=0.25,
                  cp=1340.0, alpha=47e-6, note="Victrex PEEK 450G datasheet (ISO 527). nu UNVERIFIED"),
    "PLA_print": _poly("PLA printed XY (Bambu PLA Basic)", 2.68 * GPa, 0.36, 1240, 39 * MPa, 39 * MPa,
                       family="printed", elong=0.122,
                       note="Bambu TDS V2.0, 100% infill, annealed 55 C. Z: 2.16 GPa / 35 MPa / 7.5%. nu UNVERIFIED"),
    "PETG_print": _poly("PETG printed XY (Bambu PETG HF)", 1.81 * GPa, 0.38, 1280, 34 * MPa, 34 * MPa,
                        family="printed", elong=0.086,
                        note="Bambu PETG HF TDS V1.0. Z: 1.54 GPa / 23 MPa / 5.1%. Other PETG grades differ"),
    "PCCF_print": _poly("PC-CF printed XY (3DXTech CarbonX ezPC)", 6.98 * GPa, 0.35, 1360, 73 * MPa, 73 * MPa,
                        family="printed", elong=0.021, note="3DXTech TDS Rev 1.0, XY flat only, no Z data"),
    "PPSCF_print": _poly("PPS-CF printed XY (Bambu, not annealed)", 8.23 * GPa, 0.35, 1260, 87 * MPa, 87 * MPa,
                         family="printed", elong=0.012,
                         note="Bambu PPS-CF TDS V1.0. Z: 2.85 GPa / 24 MPa / 0.7% (very weak across layers)"),
    "PACF_print": _poly("PA6-CF printed XY dry (Bambu)", 4.43 * GPa, 0.36, 1090, 102 * MPa, 102 * MPa,
                        family="printed", elong=0.058,
                        note="Bambu PA6-CF TDS V3.0 annealed+dried. Z: 2.17 GPa / 48 MPa. Wet (Polymaker PA6-CF20): "
                             "E drops ~70%, strength ~50%"),
    "TPU95A_print": _elastomer("TPU 95A printed (NinjaTek Cheetah)", 95, 1220, 39 * MPa, 5.8),
    "TPU85A_print": _elastomer("TPU 85A printed (NinjaFlex)", 85, 1190, 26 * MPa, 6.6),
    "urethane_20A": _elastomer("Cast urethane 20A (VytaFlex 20)", 20, 1000, 1.38 * MPa, 10.0),
    "urethane_40A": _elastomer("Cast urethane 40A (VytaFlex 40)", 40, 1030, 3.6 * MPa, 6.6),
    "urethane_60A": _elastomer("Cast urethane 60A (VytaFlex 60)", 60, 1040, 6.07 * MPa, 4.8),
    "urethane_90A": _elastomer("Cast urethane 90A (PMC-790)", 90, 1070, 13.79 * MPa, 5.5),
    "G10_FR4": _poly("G10 / FR4 garolite (in-plane)", 18.6 * GPa, 0.12, 1850, 280 * MPa, 310 * MPa, family="laminate",
                     elong=0.02, k=0.29, cp=1150.0, alpha=12e-6,
                     note="UNVERIFIED (not yet checked against NEMA LI-1 / supplier data); woven glass-epoxy"),
}

# measured datasheet points kept for audit (not used by the Gent E): 100% modulus from Smooth-On pages
ELASTOMER_DATA = {
    "urethane_20A": dict(M100_MPa=0.345, source="Smooth-On VytaFlex 20 Data At-A-Glance",
                         note="Gent unreliable below 40A; 100%-secant neo-Hookean gives E ~0.59 MPa vs Gent 0.73"),
    "urethane_40A": dict(M100_MPa=0.689, source="Smooth-On VytaFlex 40 Data At-A-Glance"),
    "urethane_60A": dict(M100_MPa=2.068, source="Smooth-On VytaFlex 60 Data At-A-Glance"),
    "urethane_90A": dict(M100_MPa=4.413, source="Smooth-On PMC-790 Data At-A-Glance",
                         note="Gent initial E 20.8 MPa is 2.8x the 100% secant: hard urethane strain-softens"),
    "TPU95A_print": dict(E_datasheet_MPa=26, sy_MPa=9, source="NinjaTek technical specs (Cheetah)"),
    "TPU85A_print": dict(E_datasheet_MPa=12, sy_MPa=4, source="NinjaTek technical specs (NinjaFlex)"),
}

ALL = POLYMERS
