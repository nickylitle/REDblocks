"""Composite lamina catalog plus fiber/matrix constituents for micromechanics.
Every entry's note gives its source and which fields are UNVERIFIED (verification pass 2026-10-15).
Strengths are positive numbers; compressive strengths are magnitudes."""
from dataclasses import dataclass
from .base import Lamina

GPa, MPa = 1e9, 1e6

LAMINAE = {
    "T300_5208": Lamina("T300/5208 carbon/epoxy UD", 181 * GPa, 10.3 * GPa, 7.17 * GPa, 0.28, 1600,
                        1500 * MPa, 1500 * MPa, 40 * MPa, 246 * MPa, 68 * MPa, 0.125e-3, 0.02e-6, 22.5e-6, 0.70,
                        "Tsai & Hahn 1980 via secondary copies (Univ. Mumbai QP 25700; UC3M OCW T5 ex. 5.1; "
                        "lsdyna-online c2.pdf row 5). All values match. MEDIUM confidence"),
    "AS4_3501-6": Lamina("AS4/3501-6 carbon/epoxy UD", 142 * GPa, 10.3 * GPa, 7.2 * GPa, 0.27, 1580,
                         2280 * MPa, 1440 * MPa, 57 * MPa, 228 * MPa, 71 * MPa, 0.125e-3, -0.9e-6, 27e-6, 0.63,
                         "Daniel & Ishai Table A.4 (UNVERIFIED: table not opened). Within published spread; "
                         "openly citable alternative is AS4_3501-6_WWFE"),
    "AS4_3501-6_WWFE": Lamina("AS4/3501-6 carbon/epoxy UD (WWFE)", 126 * GPa, 11 * GPa, 6.6 * GPa, 0.28, 1580,
                              1950 * MPa, 1480 * MPa, 48 * MPa, 200 * MPa, 79 * MPa, 0.125e-3, -0.9e-6, 27e-6, 0.60,
                              "Soden, Hinton & Kaddour, Compos. Sci. Technol. 58 (1998) 1011-1022 (via lsdyna-online "
                              "c2.pdf row 1): E1 E2 G12 and strengths. nu12, rho, t_ply, alphas, Vf UNVERIFIED (catalog)"),
    "IM7_8552": Lamina("IM7/8552 carbon/epoxy UD (190 gsm)", 162.1 * GPa, 8.96 * GPa, 4.69 * GPa, 0.316, 1610,
                       2559 * MPa, 1731 * MPa, 64.1 * MPa, 285.7 * MPa, 91.1 * MPa, 0.183e-3, -0.1e-6, 31e-6, 0.60,
                       "NCAMP NCP-RP-2009-028 Rev B Table 3-3 RTD means (Xt = LT 371 ksi; UNT0-derived 2212 MPa "
                       "is the conservative alternative; S at 5% strain); rho NASA TM-20230001560 Table 31; "
                       "Vf Hexcel IM7 PDS. alphas UNVERIFIED. HIGH confidence"),
    "Eglass_epoxy": Lamina("E-glass/epoxy UD (Scotchply 1002)", 38.6 * GPa, 8.27 * GPa, 4.14 * GPa, 0.26, 1800,
                           1062 * MPa, 610 * MPa, 31 * MPa, 118 * MPa, 72 * MPa, 0.15e-3, 8.6e-6, 22.1e-6, 0.45,
                           "Tsai & Hahn Scotchply 1002 via lsdyna-online c2.pdf row 8: elastic + strengths verified. "
                           "rho, Vf, alphas, t_ply UNVERIFIED (Tsai & Hahn use 0.125 mm)"),
    "Kevlar49_epoxy": Lamina("Kevlar 49/epoxy UD", 76 * GPa, 5.5 * GPa, 2.3 * GPa, 0.34, 1380,
                             1400 * MPa, 235 * MPa, 12 * MPa, 53 * MPa, 34 * MPa, 0.125e-3, -4.0e-6, 79e-6, 0.60,
                             "Tsai & Hahn via lsdyna-online c2.pdf row 9 and UC3M ex. 5.2: elastic, strengths, t_ply "
                             "verified. rho, Vf, alphas UNVERIFIED. Yt 12 MPa is what the source states (low)"),
    "CF_3K_twill": Lamina("3K carbon twill/epoxy fabric ply (infused, T700S)", 64.1 * GPa, 64.1 * GPa, 4.14 * GPa,
                          0.035, 1450, 710 * MPa, 614 * MPa, 710 * MPa, 614 * MPa, 73.1 * MPa, 0.22e-3, 2.1e-6, 2.1e-6,
                          0.48,
                          "Rock West 402-44 twill plate Engineering Properties (vendor CLT reference values, not test "
                          "allowables); Vf DragonPlate twill 48%. t_ply and CTE UNVERIFIED. MEDIUM confidence"),
    "Eglass_7781": Lamina("E-glass 7781 satin/epoxy fabric ply (prepreg)", 25.1 * GPa, 23.4 * GPa, 4.5 * GPa, 0.123,
                          1850, 471 * MPa, 571 * MPa, 425 * MPa, 475 * MPa, 67.6 * MPa, 0.254e-3, 12e-6, 12e-6, 0.50,
                          "NCAMP NCP-RP-2009-009 Rev A (MTM45-1/7781) Table 3-3 RTD means, S at 5% strain. "
                          "G12 UNVERIFIED (possible 2.3 GPa in report, catalog 4.5 may be 2x stiff); rho, Vf, alphas "
                          "UNVERIFIED. Wet layup will be lower"),
}


@dataclass(frozen=True)
class Fiber:
    name: str
    E1: float
    E2: float
    G12: float
    nu12: float
    rho: float
    Xt: float


@dataclass(frozen=True)
class Matrix:
    name: str
    E: float
    nu: float
    rho: float
    Xt: float

    @property
    def G(self):
        return self.E / (2 * (1 + self.nu))


FIBERS = {
    "T300": Fiber("Toray T300", 230 * GPa, 15 * GPa, 15 * GPa, 0.20, 1760, 3530 * MPa),
    "AS4": Fiber("Hexcel AS4", 231 * GPa, 15 * GPa, 15 * GPa, 0.20, 1790, 4410 * MPa),
    "IM7": Fiber("Hexcel IM7", 276 * GPa, 19 * GPa, 27 * GPa, 0.20, 1780, 5670 * MPa),
    "Eglass": Fiber("E-glass", 72.4 * GPa, 72.4 * GPa, 29.7 * GPa, 0.22, 2540, 3450 * MPa),
    "Sglass": Fiber("S-2 glass", 86.9 * GPa, 86.9 * GPa, 35.6 * GPa, 0.22, 2460, 4890 * MPa),
    "Kevlar49": Fiber("Kevlar 49", 124 * GPa, 6.9 * GPa, 2.8 * GPa, 0.36, 1440, 3600 * MPa),
    "Dyneema_SK75": Fiber("Dyneema SK75", 113 * GPa, 3.0 * GPa, 1.0 * GPa, 0.35, 975, 3400 * MPa),
}

MATRICES = {
    "epoxy": Matrix("Generic aerospace epoxy", 3.45 * GPa, 0.35, 1200, 70 * MPa),
    "vinylester": Matrix("Vinyl ester (Derakane 411-350)", 3.2 * GPa, 0.35, 1140, 86 * MPa),
    "PEEK": Matrix("PEEK (Victrex 450G)", 4.0 * GPa, 0.38, 1300, 98 * MPa),
}

CONSTITUENT_SOURCES = {
    "T300": "Toray T300 TDS; E2 G12 nu WWFE via NPTEL Mod. 7 Table 7.1. HIGH",
    "AS4": "Hexcel HexTow AS4 PDS (6K/12K); E2 G12 nu WWFE. HIGH",
    "IM7": "Hexcel HexTow IM7 PDS for E1 rho Xt; E2 19 / G12 27 / nu UNVERIFIED (G12 > E2 is suspicious)",
    "Eglass": "AGY S-Series datasheet E 72.3 GPa, Xt 3445 MPa pristine, rho 2.58 (Jang/ASM 2.54); nu UNVERIFIED",
    "Sglass": "AGY S-Series datasheet; nu and G12 UNVERIFIED",
    "Kevlar49": "DuPont Kevlar Technical Guide Tables II-1, II-4 (impregnated strand); Helius App. A for E2 G12",
    "Dyneema_SK75": "DSM SK75 brochure: 113 GPa, 3.4 GPa; rho derived; E2 G12 nu UNVERIFIED",
    "epoxy": "Within WWFE / Helius ranges (E 3.35-4.2, nu 0.34-0.35); Xt 70 MPa UNVERIFIED (conservative)",
    "vinylester": "Ashland DERAKANE 411-350 TDS p.3 clear cast; nu UNVERIFIED",
    "PEEK": "Victrex PEEK 450G datasheet (ISO 527); nu UNVERIFIED",
}

# Commercial off-the-shelf plates: SMEARED in-plane properties (use as an equivalent isotropic or
# orthotropic sheet, not as a lamina). Vendor CLT or production-sheet values, not design allowables.
SMEARED_PLATES = {
    "CF_plate_QI_RockWest_204-22": dict(
        Ex=45.9 * GPa, Ey=45.9 * GPa, Gxy=17.55 * GPa, nu_xy=0.309, rho=1520, Xt=429.7 * MPa, Xc=384.5 * MPa,
        S=209.9 * MPa, t=3.175e-3, Tg_C=90, layup="[0/90 tw, +-45, 0/90, +-45, 0/90], T700S / Rhino 1411",
        source="Rock West Composites part 204-22 Engineering Properties (CLT-based, reference only)",
        url="https://www.rockwestcomposites.com/204-22.html"),
    "CF_sheet_EasyComposites_HS_2mm": dict(
        Ex=43.9 * GPa, Ey=43.9 * GPa, E45=34.5 * GPa, rho=1410, Xt=558 * MPa, Xt45=622 * MPa, elong=0.0291,
        t=1.9e-3, Tg_C=80, layup="210 twill / 300 +-45 / 650 twill / 300 +-45 / 210 twill",
        source="Easy Composites High Strength Carbon Fibre Sheet TDS p.2 (measured production sheet)",
        url="https://media.easycomposites.co.uk/datasheets/EC-TDS-High-Strength-Carbon-Sheet.pdf"),
}
