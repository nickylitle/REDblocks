"""Friction, rolling-resistance and battery data with sources. Status: VERIFIED (peer-reviewed or official
doc read), PARTIAL (vendor or hobby data, or secondary), MEASURED (the user's own test).
Application-specific pairs (urethane on painted plywood, steel screw heads on plywood) have NO published value:
measure them, then add a MEASURED row."""

FRICTION_DATA = {
    "urethane_20A_dusty_floor": dict(mu_s=2.0, mu_k=1.67, status="MEASURED",
                                     source="user drive tests (measurement)"),
    "steel_bolt_skid_floor": dict(mu_k=0.203, status="MEASURED",
                                  source="user drive tests (measurement)"),
    "steel_on_steel_dry": dict(mu_s=(0.5, 0.8), mu_k=0.42, status="PARTIAL",
                               source="Engineering ToolBox, Friction Coefficients for Common Materials (table)"),
    "wood_on_clean_metal_dry": dict(mu_s=(0.2, 0.6), status="PARTIAL",
                                    source="Engineering ToolBox, Friction Coefficients (row Wood / Clean Metal)"),
    "wood_on_wood_dry": dict(mu_s=(0.25, 0.5), status="PARTIAL", source="Engineering ToolBox (same table)"),
    "rubber_on_dry_concrete": dict(mu_k=(0.6, 0.85), status="PARTIAL", source="Engineering ToolBox (same table)"),
}

ROLLING_RESISTANCE = {
    "PU_95A_caster_epoxy_floor": dict(f_R=(0.025, 0.035), typical=0.03, status="PARTIAL",
                                      note="breakaway 1.5-3x steady; steel plate x0.8, rough concrete x1.2",
                                      source="CasterHQ 'Rolling resistance' vendor table"),
    "PU_85A_caster_epoxy_floor": dict(f_R=(0.035, 0.05), status="PARTIAL", source="CasterHQ (same table)"),
}

STRIBECK_DEFAULTS = dict(delta=2.0, delta_range=(0.5, 2.0), Fs_over_Fc=(1.5, 2.0),
                         source="Alaci et al., Axioms 2024 (delta range); LTH L6 lecture notes (Fs/Fc)")

LUGRE_EXAMPLE = dict(sigma0=1e5, sigma1=316.0, sigma2=0.4, Fc=1.0, Fs=1.5, vs=1e-3, m=1.0,
                     source="Canudas de Wit et al. 1995 example, via arXiv:2508.00175 Table I")

MAGIC_FORMULA_ROAD = {   # passenger-car tires, NOT urethane wheels
    "dry_tarmac": (10, 1.9, 1.0, 0.97), "wet_tarmac": (12, 2.3, 0.82, 1.0),
    "snow": (5, 2, 0.3, 1.0), "ice": (4, 2, 0.1, 1.0),
    "_source": "MathWorks Simscape Tire-Road Interaction (Magic Formula) coefficient table, citing Pacejka 2005",
}

LIPO_IR_PER_CELL = {
    "5000mAh_new": dict(R=(1e-3, 4e-3), status="PARTIAL", note="AC-IR reads lower than DC-IR",
                        source="RCHelicopterFun 'LiPo Internal Resistance' (hobby article)"),
    "100-200mAh_new": dict(R=0.150, status="PARTIAL", source="same"),
    "grading_bands": dict(excellent=(0, 1.5e-3), good=(1.5e-3, 3e-3), average=(3e-3, 5e-3),
                          degraded=(5e-3, 8e-3), status="PARTIAL", source="Grepow blog (manufacturer)"),
}

PLANETARY_EFFICIENCY = dict(eta_f=(0.7, 0.8), status="PARTIAL",
                            source="Sim & Ramos arXiv:2011.02506 Sec. IV-B (legged-robot planetaries, G = 20)")
