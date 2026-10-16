"""Flapjack drive sim: legacy tire/gearbox vs sourced replacements (Pacejka MF, Svendenius brush,
Matsuki back-drive efficiency). Same scenarios as Nicky's v1.0 study."""
import math, os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
import flapjack_drive_legacy as F
from simblocks.friction.published import magic_formula_fit, brush_mu_s_for_peak
from simblocks.actuators.motors import backdrive_efficiency

base = F.base_params()
mus, muk, sigp = base["mus"], base["muk"], base["sigp"]
slope = muk / (0.5 * sigp) + (mus - muk) * math.e / sigp          # legacy curve's d mu / d slip at 0
mf = magic_formula_fit(mus, muk, sigp, slope)
eta_f = base["motor"]["eta"]; G = base["motor"]["G"]
eta_b = backdrive_efficiency(eta_f, G)
brush_mus = brush_mu_s_for_peak(mus, muk)

VARIANTS = {
    "legacy": dict(),
    "MF tire": dict(tmodel=1.0, mfB=mf["B"], mfC=mf["C"], mfE=mf["E"]),
    "brush tire": dict(tmodel=2.0, brush_mus=brush_mus, brush_cs=slope),
    "MF + eta_b": dict(tmodel=1.0, mfB=mf["B"], mfC=mf["C"], mfE=mf["E"], eta_b=eta_b),
}


def metrics(d):
    _, L = F.launch(d)
    _, P = F.point_turn(d)
    _, B = F.braking(d, "brake")
    _, Rv = F.braking(d, "reverse")
    _, S = F.step_steer(d, 0.5)
    return dict(t_8ft=L["t_8ft"], v_8ft=L["v_8ft"], a_peak_g=L["a_peak"], t_180=P["t_180"],
                brake_stop_mm=B["d_stop"], reverse_stop_mm=Rv["d_stop"], reverse_t_stop=Rv["t_stop"],
                dchi_05_deg=S["dchi_05"], ay_peak_g=S["ay_peak"])


def ledger_and_dt(d):
    R1 = F.run(d, 2.0, lambda t: (np.ones_like(t), np.ones_like(t)), dt=1e-4)
    R2 = F.run(d, 2.0, lambda t: (np.ones_like(t), np.ones_like(t)), dt=5e-5)
    x1, x2 = R1.dist[-1], R2.dist[-1]
    return dict(ledger_rel=R1.ledger()["rel"], dt_rel=abs(x1 - x2) / x2)


if __name__ == "__main__":
    out = {"params": dict(mf=mf, slope=slope, eta_f=eta_f, eta_b=eta_b, brush_mus=brush_mus), "runs": {}}
    for k, v in VARIANTS.items():
        d = dict(base, **v)
        out["runs"][k] = dict(metrics(d), **ledger_and_dt(d))
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "out", "drive_sourced_compare.json"), "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
