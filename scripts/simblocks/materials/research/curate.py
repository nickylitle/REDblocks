"""Curated published fits -> published_fits.json.

Raw research agent outputs live in research/raw_research.json (one file, keyed by the original file name) (kept for audit). This file is the human-reviewed
layer: which fit applies to which catalog key, and whether it is an EXACT condition match (allowed as the
default curve) or only NEAREST (different temper, alloy or hardness: listed in the CSV header, never used
as the default). Run:  python -m simblocks.materials.research.curate
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "published_fits.json")
MPa = 1e6
LESUER = ("Lesuer, D.R., Experimental Investigations of Material Models for Ti-6Al-4V Titanium and 2024-T3 "
          "Aluminum, DOT/FAA/AR-00/25, LLNL/FAA, Sept 2000")
LESUER_URL = "https://corpora.tika.apache.org/base/docs/govdocs1/925/925499.pdf"
JC83 = ("Johnson, G.R. & Cook, W.H., A constitutive model and data for metals subjected to large strains, "
        "high strain rates and high temperatures, 7th Int. Symp. Ballistics, 1983")
JC83_URL = ("https://ia800406.us.archive.org/4/items/AConstitutiveModelAndDataForMetals/"
            "A%20constitutive%20model%20and%20data%20for%20metals_text.pdf")
MOCKO = ("Mocko, W., Effect of strain rate and temperature on visco-plastic properties of CP titanium and "
         "TiAl6V4 (Polish), Instytut Transportu Samochodowego (c. 2014)")
MOCKO_URL = ("https://yadda.icm.edu.pl/baztech/element/bwmeta1.element.baztech-756890ec-c766-4cc4-8720-"
             "1542c84b9504/c/mocko_wplyw_3_2014.pdf")


def jc(A, B, n, C, m, Tm, Tr, rate, **meta):
    return dict(model="johnson_cook",
                params=dict(A_Pa=A * MPa, B_Pa=B * MPa, n=n, C=C, m=m, Tmelt_K=Tm, Tref_K=Tr, ref_rate=rate),
                strain_measure="true plastic", **meta)


FITS = {
    # ------------------------------------------------------------------ titanium
    "Ti6Al4V": [
        jc(1098, 1092, 0.93, 0.014, 1.1, 1878, 298, 1.0, applies="exact", confidence="high",
           citation=LESUER, url=LESUER_URL, location="Table 8 p.18 (new constants); Eq. 1",
           test_conditions="AMS 4911 hot-rolled + annealed plate, Rc 36; A,B,n fit at ~5000/s",
           notes="A exceeds quasi-static yield: at 1e-4/s the C term gives ~955 MPa. Hardening nearly linear."),
        jc(862, 331, 0.34, 0.012, 0.8, 1878, 298, 1.0, applies="nearest", confidence="medium",
           citation=LESUER + " quoting G.R. Johnson (NSWC)", url=LESUER_URL, location="Table 6 p.17",
           test_conditions="Rc 29 material, softer than typical annealed plate"),
        jc(900, 2295, 1.0, 0.015, 1.0, 1870, 296, 2e-4, applies="nearest", confidence="low",
           citation=MOCKO, url=MOCKO_URL, location="Table 1 p.72",
           test_conditions="Grade 5 compression cylinders, heat treatment not stated"),
    ],
    "Ti_Gr2": [
        jc(400, 775, 0.609, 0.03, 1.0, 1870, 296, 2e-4, applies="exact", confidence="medium",
           citation=MOCKO, url=MOCKO_URL, location="Table 1 p.72",
           test_conditions="CP Ti Grade 2 compression, quasi-static reference 2e-4/s; QS yield ~400 MPa",
           notes="m = 1 appears fixed, not fit. Only complete JC set found for Gr2."),
    ],
    # ------------------------------------------------------------------ copper alloys
    "Cu_C110": [
        jc(90, 292, 0.31, 0.025, 1.09, 1356, 298, 1.0, applies="exact", confidence="high",
           citation=JC83, url=JC83_URL, location="Table 1; also arXiv:1608.03056 Table 2",
           test_conditions="OFHC copper, Rockwell F-30 (soft, annealed)",
           notes="Annealed only. Do not use for H02/H04 stock."),
    ],
    "brass_C360": [
        jc(112, 505, 0.42, 0.009, 1.68, 1189, 298, 1.0, applies="nearest", confidence="medium",
           citation=JC83, url=JC83_URL, location="Table 1",
           test_conditions="Cartridge brass C26000, Rockwell F-67 (annealed): different alloy and temper"),
    ],
    "WHA_90W": [
        jc(1506, 177, 0.12, 0.016, 1.0, 1723, 298, 1.0, applies="nearest", confidence="high",
           citation=JC83, url=JC83_URL, location="Table 1",
           test_conditions="90W-7Ni-3Fe swaged to Rc 47 (as-sintered is Rc 25): ~2.5x too strong for sintered"),
        jc(631, 1258, 0.092, 0.014, 0.94, 1723, 298, 1.0, applies="nearest", confidence="medium",
           citation="Skoglund, P., Constitutive modelling and mechanical properties of a tungsten heavy metal "
                    "alloy, FOI-R--0723--SE, 2002", url="https://www.foi.se/rest-api/report/FOI-R--0723--SE",
           location="Table II p.12", test_conditions="92.5W Kennametal Y925, YS 1300 MPa (worked)"),
    ],
}

# ---------------------------------------------------------------------- aluminum / magnesium
FITS.update({
    "6061-T6": [
        dict(model="ludwik", params=dict(sigma0_Pa=245e6, K_Pa=233e6, n=0.34), strain_measure="true plastic",
             applies="exact", confidence="medium",
             citation="Byun, Joo, Lee & Kim, Materials 16(21):7055, 2023, doi:10.3390/ma16217055",
             url="https://pmc.ncbi.nlm.nih.gov/articles/PMC10650735", location="Table 2 (MJC A, B, n), Eq. 2",
             test_conditions="6061-T6 plate 3-8 mm, tension at 0.01/s (reference rate of the MJC fit)",
             notes="Published as modified JC with power-law rate term (1 + epsdot*)^C, C = 0.03; written here at "
                   "its reference rate. A = 245 MPa is below AA typical 276 MPa (lot or fit choice)."),
        jc(324, 114, 0.42, 0.002, 1.34, 856, 293, 1.0, applies="exact", confidence="medium",
           citation="Lesuer, Kay & LeBlanc, UCRL-JC-134118 (via Al-Badour 2014 / JMSG 2020 Table 4, CJME 2021 "
                    "Table 3, Pierazzo 2008)", url="https://jmsg.springeropen.com/articles/10.1186/s40712-020-00118-y/tables/4",
           location="JMSG Table 4; CJME Table 3", test_conditions="primary report not opened; ref rate assumed 1/s",
           notes="De facto standard set, three secondary tables agree. Very low hardening and rate sensitivity."),
    ],
    "7075-T6": [
        jc(546, 674, 0.72, 0.059, 1.56, 893, 293, 1.0, applies="exact", confidence="high",
           citation="Brar, Joshi & Harris, Constitutive Model Constants for Al7075-T651 and Al7075-T6, SCCM 2009, "
                    "AIP Conf. Proc. 1195:945-948", url="https://meetings-archive.aps.org/shock/2009/b5/2",
           location="APS abstract B5.2 (constants)", test_conditions="tension, RT-250 C, quasi-static to high rate",
           notes="Reference rate and Tmelt are in the AIP paper (not opened): 1/s and 893 K assumed."),
        dict(model="swift_voce", params=dict(K_Pa=855e6, eps0=0.026, n=0.15, k0_Pa=502e6, Q_Pa=194e6, b=10.08,
                                              alpha=0.5), strain_measure="true plastic", applies="exact",
             confidence="medium",
             citation="Li, Pandya, Karathanasopoulos, Roth & Mohr, EPJ Web Conf. 250, 05007 (DYMAT 2021)",
             url="https://www.epj-conferences.org/articles/epjconf/pdf/2021/04/epjconf_dymat2021_05007.pdf",
             location="Table 1, Eqs. 3-5", test_conditions="7075-T6 2 mm sheet, RT quasi-static 0.001/s",
             notes="Mixed Swift-Voce (alpha 0.5): better large-strain extrapolation than a power law."),
    ],
    "7075-T651": [
        jc(527, 676, 0.71, 0.017, 1.61, 893, 293, 1.0, applies="exact", confidence="high",
           citation="Brar, Joshi & Harris, SCCM 2009, AIP Conf. Proc. 1195:945-948",
           url="https://meetings-archive.aps.org/shock/2009/b5/2", location="APS abstract B5.2 (T651 set)",
           test_conditions="tension, RT-250 C", notes="Reference rate and Tmelt assumed (AIP paper not opened)."),
    ],
    "2024-T3": [
        jc(369, 684, 0.73, 0.0083, 1.7, 775, 298, 1.0, applies="exact", confidence="high",
           citation=LESUER, url=LESUER_URL, location="Table 8 p.18; Eq. 1; Table 7 (Tmelt 502 C)",
           test_conditions="Kaiser 2024-T3 4 mm plate, SHPB tension/compression"),
        jc(265, 426, 0.34, 0.015, 1.0, 775, 298, 1.0, applies="nearest", confidence="high",
           citation="Johnson, Hoegfeldt, Lindholm & Nagy 1983 (via " + LESUER + ")", url=LESUER_URL,
           location="Table 6 p.17", test_conditions="2024-T351 plate, torsion; under-predicts T3 flow stress"),
    ],
    "5052-H32": [
        dict(model="hollomon", params=dict(K_Pa=328.5e6, n=0.1127), strain_measure="true plastic",
             applies="exact", confidence="low",
             citation="Badke, dos Santos, Scheuer, Goulart & de Almeida, CONEM2022-0029 (ABCM), 2022",
             url="https://www.sistema.abcm.org.br/articleFiles/download/34350", location="Table 5 (RD 0 deg)",
             test_conditions="5052-H32 2.5 mm cold-rolled plate, quasi-static, rolling direction",
             notes="Conference paper; its elongation column looks wrong. Shape check quality."),
        jc(176.6, 289.4, 0.3712, 0.001, 1.0, 880, 293, 1.0, applies="nearest", confidence="low",
           citation="JMSG 2025 car frontal crash FE article s40712-025-00326-4",
           url="https://jmsg.springeropen.com/articles/10.1186/s40712-025-00326-4/tables/2", location="Table 2",
           test_conditions="AA5052, temper not stated"),
    ],
    "AZ31B": [
        jc(91.487, 4353.112, 1.353, 0.033, 1.0, 905, 293, 0.0021, applies="nearest", confidence="low",
           citation="Yang, Chen, Lan, Zeng & Ma, MATEC Web Conf. 153, 01004, 2018",
           url="https://www.matec-conferences.org/articles/matecconf/pdf/2018/12/matecconf_icmme2018_01004.pdf",
           location="Sec. 4, Eq. 2", test_conditions="COMPRESSION fit (twinning, n > 1); off-spec tensile lot"),
    ],
})

# ---------------------------------------------------------------------- polymers / elastomers
SMITH = ("Smith, L. & MacCurdy, R., Mechanical Characterization and Constitutive Modeling of 3D Printable Soft "
         "Materials, 3D Printing and Additive Manufacturing 11(3), 2024, doi:10.1089/3dp.2023.0010")
SMITH_URL = "https://matterassembly.org/assets/pdf/preprints/2024_smith_modeling_softmaterials.pdf"
FITS.update({
    "PC": [
        jc(80, 75, 2.0, 0.052001, 0.548, 562, 295, 1.0, applies="exact", confidence="high",
           citation="Dwivedi, Bradley & Casem, Mechanical Response of Polycarbonate with Strength Model Fits, "
                    "ARL-TR-5899, 2012", url="https://apps.dtic.mil/sti/pdfs/ADA566369.pdf",
           location="Eqs. 1-2 p.9, Table 2 p.10",
           test_conditions="Makrolon GP cast plate, compression 0.005/s-15000/s",
           notes="Fits high-rate adiabatic data better than low-rate. Compression-calibrated."),
    ],
    "HDPE": [
        jc(8.859, 30.56, 0.3838, 0.0455, 1.0, 400, 296, 0.0033, applies="nearest", confidence="medium",
           citation="Alhourani, Barsoum, Sheikh-Ahmad & Deveci, Proc. MCM'24, paper ICMIE 117, 2024",
           url="https://avestia.com/MCM2024_Proceedings/files/paper/ICMIE/ICMIE_117.pdf", location="Table 1",
           test_conditions="pipe-grade HDPE, E 1.0 GPa, not sheet; 0.0033-2.2/s"),
    ],
    "TPU95A_print": [
        dict(model="ogden", params=dict(mu_Pa=[11.5138e6, -0.33239e6], alpha=[1.457, -7.8825]),
             strain_measure="stretch, uniaxial", applies="exact", confidence="medium", citation=SMITH,
             url=SMITH_URL, location="Eq. 1, Table 1 (Polymaker TPU90, listed 95A)",
             test_conditions="printed 95A TPU (Polymaker brand), ASTM D412 Die C, quasi-static, to ~550%",
             notes="Different 95A brand than the datasheet values (NinjaTek Cheetah); Cheetah's own fit is "
                   "ill-conditioned. Cross-check: 9.9 MPa at 55% vs Cheetah yield 9 MPa at 55%. The negative-"
                   "alpha term climbs far past real UTS at large stretch, so the curve is cut at datasheet UTS."),
    ],
    "TPU85A_print": [
        dict(model="mooney_rivlin", params=dict(C10_Pa=0.385e6, C01_Pa=1.47e6, stretch_max=5.5),
             strain_measure="stretch, uniaxial", applies="exact", confidence="medium",
             citation="Reppel & Weinberg, Technische Mechanik 38(1):104-112, 2018, doi:10.24352/UB.OVGU-2018-010",
             url="https://journals.ub.ovgu.de/index.php/techmech/article/download/564/540/1187",
             location="Table 3 (mu1 0.77, mu2 2.94 MPa; C10 = mu1/2, C01 = mu2/2)",
             test_conditions="printed NinjaFlex, uniaxial, valid to ~450%",
             notes="Cross-check: 3.3 MPa at 65% strain vs NinjaTek datasheet yield 4 MPa at 65%."),
        dict(model="ogden", params=dict(mu_Pa=[5.5438e6, -0.0976e6], alpha=[2.1314, -9.0282]),
             strain_measure="stretch, uniaxial", applies="exact", confidence="low", citation=SMITH,
             url=SMITH_URL, location="Eq. 1, Table 1 (NinjaFlex)",
             test_conditions="printed NinjaFlex 85A, ASTM D412 Die C, quasi-static, to ~550%",
             notes="FAILS CROSS-CHECK: 8.4 MPa at 65% strain, about 2x the NinjaTek datasheet and Reppel 2018."),
    ],
})

# ---------------------------------------------------------------------- armor / tool / stainless
NILSSON = ("Nilsson, M., Constitutive Model for Armox 500T and Armox 600T at Low and Medium Strain Rates, "
           "FOI-R--1068--SE, Swedish Defence Research Agency, 2003")
FITS.update({
    "AR500": [
        jc(1470, 702, 0.199, 0.00549, 0.811, 1800, 293, 1.0, applies="nearest", confidence="high",
           citation=NILSSON, url="https://www.foi.se/rest-api/report/FOI-R--1068--SE",
           location="Table 13 p.37 (Procedure 1)", test_conditions="Armox 500T 10 mm plate (armor grade, same "
           "480-540 HBW class as AR500, tighter chemistry and toughness)"),
        jc(1372.488, 835.022, 0.2467, 0.0617, 0.84, 1800, 293, 1.0, applies="nearest", confidence="medium",
           citation="Iqbal, Senthil, Sharma & Gupta, IJIE 96:146-164, 2016, doi:10.1016/j.ijimpeng.2016.05.017",
           url="https://www.academia.edu/26924907", location="Table 4",
           test_conditions="Armox 500T; B differs between the paper's Tables 2 and 4"),
    ],
    "S7_56HRC": [
        jc(1539, 477, 0.18, 0.012, 1.0, 1763, 298, 1.0, applies="nearest", confidence="medium",
           citation=JC83, url=JC83_URL, location="Table 1", test_conditions="S-7 at Rockwell C-50 (softer than 55 HRC)"),
    ],
    "17-4PH_H900": [
        dict(model="hollomon", params=dict(K_Pa=2068e6, n=0.224), strain_measure="true total",
             applies="nearest", confidence="low",
             citation="Mulkey, J.R., Strain Hardening Exponent for 17-4 PH Stainless Steel, BDX-613-577, Bendix, 1971",
             url="https://www.osti.gov/servlets/purl/4703707", location="Summary; Fig. A-1 to A-5",
             test_conditions="H900, fit through POST-necking RA data: overpredicts pre-necking hardening"),
    ],
    "304_annealed": [
        jc(452, 694, 0.311, 0.0054, 0.996, 1700, 293, 1.0, applies="nearest", confidence="low",
           citation="Sci. Rep. 2024, doi:10.1038/s41598-024-71671-1", url="https://www.nature.com/articles/s41598-024-71671-1",
           location="Table 6", test_conditions="machining inverse fit on 304 rod, condition not stated (A ~2x annealed yield)"),
    ],
    "maraging_300": [
        dict(model="ramberg_osgood", params=dict(n=22), strain_measure="engineering total", applies="exact",
             confidence="medium", citation="MIL-HDBK-5J Sec. 2.5.1.1, Fig. 2.5.1.1.6(b) (via KasperCalc)",
             url="https://kaspercalc.com/Chapter2.5(steel).html", location="Figure 2.5.1.1.6(b)",
             test_conditions="18Ni 280(300) grade aged 900 F, longitudinal bar, tension, RT",
             notes="Compression RT n = 21 with Ec 197 GPa (Fig. 2.5.1.1.6(d)). Curves stop near Ftu."),
    ],
})

# ---------------------------------------------------------------------- carbon / alloy steels
FITS.update({
    "1006_mild": [
        jc(350, 275, 0.36, 0.022, 1.0, 1811, 298, 1.0, applies="exact", confidence="high",
           citation=JC83, url=JC83_URL, location="Table 1 (1006 steel, Rockwell F-94)",
           test_conditions="1006 at HRF 94 (~HB 100, mildly cold worked: matches CD bar); torsion; m assumed 1",
           notes="Yield A = 350 vs CD typical 300: +17% expected warning."),
    ],
    "4340_QT_315C": [
        jc(1339, 849, 0.26, 0.014, 1.03, 1793, 298, 1.0, applies="nearest", confidence="medium",
           citation="Banerjee, B., The MTS model for various tempers of AISI 4340 steel, IJSS 44:834-859, 2007; "
                    "arXiv cond-mat/0510330", url="https://arxiv.org/pdf/cond-mat/0510330",
           location="Appendix C Eqs. 26-28 (A, B evaluated at Rc 47)",
           test_conditions="hardness-scaled JC: A from an ASM yield-vs-Rc curve, B/A, n, C, m held from HRC 30; "
                           "A is ~15% below the handbook yield at 315 C"),
        jc(792, 510, 0.26, 0.014, 1.03, 1793, 298, 1.0, applies="nearest", confidence="high",
           citation=JC83, url=JC83_URL, location="Table 1 (B from arXiv:1608.03056 Table 2)",
           test_conditions="4340 at Rockwell C-30, not the 315 C temper"),
    ],
})


def write():
    existing = {k: list(v) for k, v in FITS.items()}
    # exact matches first, then by confidence
    rank = {"high": 0, "medium": 1, "low": 2}
    for k, recs in existing.items():
        recs.sort(key=lambda r: (r.get("applies") != "exact", rank.get(r.get("confidence"), 3)))
    json.dump(existing, open(OUT, "w"), indent=1)
    return OUT


if __name__ == "__main__":
    print(write())
