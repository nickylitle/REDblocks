"""Metals catalog. Typical room-temperature handbook values (ASM / MatWeb typical).
Heat-treat-dependent strengths vary +/-10%: use vendor certs for final sizing.
Johnson-Cook tuples are (A, B, n, C, m, Tmelt K, Tref K) in Pa / K from the cited papers."""
from .base import Isotropic

GPa, MPa = 1e9, 1e6
_ST = dict(E=205 * GPa, nu=0.29, rho=7850, k=42.0, cp=475.0, alpha=12.3e-6,
           resistivity=2.2e-7, cowper_D=40.4, cowper_q=5.0)

def _steel(name, sy, su, **kw):
    d = dict(_ST); d.update(kw)
    return Isotropic(name=name, sy=sy * MPa, su=su * MPa, family="steel", **d)

STEELS = {
    "1018_CD": _steel("1018 cold drawn bar", 372, 441, elong=0.15, k=51.9, hardness="HB 126",
                      note="Alro mill averages, 1 in round (64/54 ksi). Hot-rolled plate is ~400/220 MPa"),
    "1045_CD": _steel("1045 cold drawn bar", 531, 627, E=207 * GPa, elong=0.12, hardness="HB 179",
                      note="Alro mill averages (91/77 ksi); E SAE FD&E normalized 1045"),
    "4130_normalized": _steel("4130 normalized (bar)", 440, 670, elong=0.26, hardness="HB 200",
                              note="MakeItFrom citing ASM Specialty Handbook / ASTM A322. Thin normalized sheet "
                                   "tests stronger: NACA TN-2324 0.090 in sheet 679/807 MPa, 14%"),
    "4130_45HRC": _steel("4130 Q&T 45 HRC (ESTIMATED)", 1400, 1500, elong=0.10, hardness="HRC 45",
                         note="NO source at 45 HRC: nearest NASA TN D-1574 4130 WQ + "
                              "399 C temper, ~HRC 40: 1358/1427 MPa. Low-temper martensite Sy/Su 0.85-0.9 suggests "
                              "Sy 1300-1400. 4130 is shallow hardening: through-hardens only in thin sections"),
    "4140_QT_30HRC": _steel("4140 Q&T ~30 HRC (prehard)", 890, 940, elong=0.20, hardness="HRC 28-30 (HB 269-285)",
                            note="CORRECTED: was the normalized row. Alro mill averages, oil quench + 595-650 C "
                                 "temper (mean of 1100 F and 1200 F rows)"),
    "4140_normalized": _steel("4140 normalized 870 C", 655, 1020, elong=0.177, hardness="HB 302",
                              note="Callister/ASM appendix row via AmesWeb chart"),
    "4340_QT_315C": _steel("4340 oil Q&T 315 C temper", 1620, 1760, rho=7830, elong=0.12, hardness="HRC 47",
                           note="Callister/ASM appendix row via AmesWeb; HRC from SSA 4340 tempering curve. JC 1983 "
                                "set is for HRC 30 only (see published_fits nearest)"),
    "AR500": _steel("AR500 abrasion plate", 1250, 1640, E=207 * GPa, elong=0.08, hardness="HB 477-534",
                    note="Generic distributor grade (not a standard). Within Armox 500T minimums (Nilsson "
                         "FOI-R--1068--SE Table 1: 1250 / 1450-1750 / 8%) and Chapel AR500F typicals (1289/1703)"),
    "S7_56HRC": _steel("S7 tool steel ~55 HRC (315 C temper)", 1585, 1965, E=207 * GPa, nu=0.295, rho=7830,
                       elong=0.09, hardness="HRC 55",
                       note="ASM-type fan-cooled S7 table (via Compraco, primary not opened), 315 C / 55 HRC row. "
                            "205 C / 58 HRC row: 1450/2170/7%. E, rho Crucible DS314; nu BRL ARBRL-MR-03067"),
    "17-4PH_H900": _steel("17-4PH H900 (typical)", 1240, 1340, E=196 * GPa, rho=7810, k=17.9, alpha=10.8e-6,
                          elong=0.10, hardness="HRC 43",
                          note="ATI 17-4 TDS v1 typical. Spec min (AMS 5604): 1170/1310, HRC 40-47"),
    "17-4PH_H900_min": _steel("17-4PH H900 (AMS 5604 minimums)", 1170, 1310, E=196 * GPa, rho=7810, k=17.9,
                              alpha=10.8e-6, elong=0.10, hardness="HRC 40 min", note="conservative variant"),
    "304_annealed": _steel("304 stainless annealed", 215, 505, E=193 * GPa, rho=8000, k=16.2, cp=500.0,
                           alpha=17.3e-6, elong=0.70, hardness="HRB 70",
                           note="ASM / MatWeb AISI 304 typical (near MIL-HDBK-5 S-basis level). Real sheet often "
                                "250-310 / 580-650 MPa. Strongly rate sensitive"),
    "maraging_300": _steel("Maraging 18Ni(300) aged 900 F (MIL-HDBK-5 S-basis bar)", 1862, 1931, E=182.7 * GPa,
                           nu=0.31, rho=7920, k=25.0, elong=0.05, hardness="HRC 50-55",
                           note="MIL-HDBK-5 Table 2.5.1.0(c) via KasperCalc. Typical ~2000/2050 UNVERIFIED. "
                                "Ec = 197 GPa in compression"),
    "1006_mild": _steel("1006 low-carbon steel cold drawn", 300, 370, E=190 * GPa, rho=7890, elong=0.22,
                        hardness="HB 100", jc=(350e6, 275e6, 0.36, 0.022, 1.0, 1811.0, 298.0),
                        note="CORRECTED (old 350 yield was copied from JC A). MakeItFrom CD 1006 (ASM / ASTM A29). "
                             "JC: Johnson & Cook 1983 Table 1 (1006 at HRF 94)"),
}

TITANIUM = {
    "Ti6Al4V": Isotropic("Ti-6Al-4V Gr5 annealed", 113.8 * GPa, 0.342, 4430, 880 * MPa, 950 * MPa,
                         family="titanium", elong=0.14, k=6.7, cp=526.0, alpha=8.6e-6, resistivity=1.78e-6,
                         hardness="HRC 36", jc=(1098e6, 1092e6, 0.93, 0.014, 1.1, 1878.0, 298.0),
                         note="Typical values: ASM Aerospace Spec. Metals Ti-6Al-4V Gr5 Annealed datasheet (spec min "
                              "828/895 MPa, 10%). JC: Lesuer 2000 DOT/FAA/AR-00/25 Table 8 (AMS 4911 plate)"),
    "Ti_Gr2": Isotropic("Ti CP Grade 2 annealed (typical)", 105 * GPa, 0.37, 4510, 345 * MPa, 485 * MPa,
                        family="titanium", elong=0.28, k=16.4, cp=523.0, alpha=8.6e-6,
                        note="TIMET TIMETAL 50A Table 4 typical anneal; E/nu/rho ASM Gr2 datasheet"),
    "Ti_Gr2_min": Isotropic("Ti CP Grade 2 (ASTM B265 minimums)", 105 * GPa, 0.37, 4510, 275 * MPa, 345 * MPa,
                            family="titanium", elong=0.20, k=16.4, cp=523.0, alpha=8.6e-6,
                            note="ASTM B265 specified minimums: conservative variant"),
}

ALUMINUM = {
    "6061-T6": Isotropic("6061-T6", 68.9 * GPa, 0.33, 2700, 276 * MPa, 310 * MPa, family="aluminum",
                         elong=0.12, k=167.0, cp=896.0, alpha=23.6e-6, resistivity=3.99e-8,
                         cowper_D=6500.0, cowper_q=4.0,
                         jc=(324e6, 114e6, 0.42, 0.002, 1.34, 856.0, 293.0),
                         note="AA typical (MatWeb 6061-T6/T651); elong 12% thin sheet, 17% 1/2 in round. "
                              "T6 / T651 / T6511 share these. JC: standard set (Lesuer et al. UCRL-JC-134118 via "
                              "3 secondary tables), Tmelt 856 K"),
    "7075-T6": Isotropic("7075-T6", 71.7 * GPa, 0.33, 2810, 503 * MPa, 572 * MPa, family="aluminum",
                         elong=0.11, k=130.0, cp=960.0, alpha=23.4e-6, cowper_D=6500.0, cowper_q=4.0,
                         jc=(546e6, 674e6, 0.72, 0.059, 1.56, 893.0, 293.0),
                         note="NASA-CR-123773 typical (T6 and T651 same). JC: Brar, Joshi & Harris 2009 T6 set "
                              "(corrected from a mixed T6/T651 tuple); Tmelt UNVERIFIED. Plate is sold as T651: "
                              "use 7075-T651"),
    "7075-T651": Isotropic("7075-T651 plate", 71.7 * GPa, 0.33, 2810, 503 * MPa, 572 * MPa, family="aluminum",
                           elong=0.11, k=130.0, cp=960.0, alpha=23.4e-6, cowper_D=6500.0, cowper_q=4.0,
                           jc=(527e6, 676e6, 0.71, 0.017, 1.61, 893.0, 293.0),
                           note="Same typical values as T6 (NASA-CR-123773). JC: Brar et al. 2009 T651 set. "
                                "Thick plate runs lower (Liu 2005: 476/552 MPa, 7%)"),
    "2024-T3": Isotropic("2024-T3 sheet (bare)", 73.1 * GPa, 0.33, 2780, 345 * MPa, 483 * MPa, family="aluminum",
                         elong=0.18, k=121.0, cp=875.0, alpha=23.2e-6,
                         jc=(369e6, 684e6, 0.73, 0.0083, 1.7, 775.0, 298.0),
                         note="AA typical via Gabrian; Lesuer 2000 test lot ~5% lower. Alclad is weaker. "
                              "JC: Lesuer DOT/FAA/AR-00/25 Table 8"),
    "5052-H32": Isotropic("5052-H32", 70.3 * GPa, 0.33, 2680, 193 * MPa, 228 * MPa, family="aluminum",
                          elong=0.12, k=138.0, cp=880.0, alpha=23.8e-6,
                          note="AA typical (Protolabs / Clinton datasheet). Real lots often test higher UTS "
                               "(Badke 2022: 265-274 MPa). Slight negative rate sensitivity"),
}

OTHER_METALS = {
    "AZ31B": Isotropic("Magnesium AZ31B-H24 sheet (typical)", 45 * GPa, 0.35, 1770, 214 * MPa, 290 * MPa,
                       family="magnesium", elong=0.16, k=96.0, cp=1000.0, alpha=26e-6,
                       note="SuppliersOnline typical (ASTM B90). E Yang 2018. nu UNVERIFIED. STRONG tension/"
                            "compression asymmetry (twinning): compressive yield well below tensile; an isotropic "
                            "curve mispredicts compression-dominated contact"),
    "AZ31B_min": Isotropic("Magnesium AZ31B-H24 (AMS 4377 minimums, thin sheet)", 45 * GPa, 0.35, 1770, 200 * MPa,
                           269 * MPa, family="magnesium", elong=0.06, k=96.0, cp=1000.0, alpha=26e-6,
                           note="AMS 4377, 0.016-0.249 in; thicker plate lower (0.5-1 in: 152/248 MPa)"),
    "Cu_C110": Isotropic("Copper C110 annealed (OS050)", 117 * GPa, 0.31, 8910, 69 * MPa, 221 * MPa,
                         family="copper", elong=0.45, k=391.0, cp=385.0, alpha=17e-6, resistivity=1.68e-8,
                         jc=(90e6, 292e6, 0.31, 0.025, 1.09, 1356.0, 298.0),
                         note="CDA C11000 typical, flat OS050. nu UNVERIFIED. JC: OFHC, Johnson & Cook 1983 Table 1"),
    "Cu_C110_H04": Isotropic("Copper C110 hard (H04 flat)", 117 * GPa, 0.31, 8910, 310 * MPa, 345 * MPa,
                             family="copper", elong=0.06, k=388.0, cp=385.0, alpha=17e-6, resistivity=1.71e-8,
                             note="CDA C11000 typical flat H04 (elong 6-12%, low end used). nu UNVERIFIED"),
    "brass_C360": Isotropic("Brass C360 half-hard (H02 rod, 25 mm)", 97 * GPa, 0.31, 8490, 310 * MPa, 400 * MPa,
                            family="copper", elong=0.25, k=115.0, cp=380.0, alpha=20.5e-6,
                            note="CDA typical H02 1 in rod (via Eng-Tips quote, MEDIUM); size dependent: 6 mm "
                                 "470/360, 50 mm 380/305. nu UNVERIFIED"),
    "WHA_90W": Isotropic("Tungsten heavy alloy 90W-7Ni-3Fe sintered (B777 Class 1)", 310 * GPa, 0.29, 17000,
                         607 * MPa, 862 * MPa, family="tungsten", elong=0.14, k=75.0, cp=150.0, alpha=4.8e-6,
                         hardness="HRC 25",
                         note="Torrey Hills / Rembar typical; proportional limit only ~359 MPa. nu UNVERIFIED"),
    "WHA_90W_min": Isotropic("Tungsten heavy alloy 90W (ASTM B777 Class 1 minimums)", 310 * GPa, 0.29, 17000,
                             517 * MPa, 758 * MPa, family="tungsten", elong=0.05, k=75.0, cp=150.0, alpha=4.8e-6,
                             note="ASTM B777-07 Tables 1-2 minimums: conservative variant"),
}

ALL = {**STEELS, **TITANIUM, **ALUMINUM, **OTHER_METALS}
