"""Liquid rocket engines from propellant chemistry (the NASA CEA method, in closed form plus a Newton solve).

Chamber: the propellants (at their storage enthalpy) burn to chemical equilibrium at constant pressure, adiabatic.
Nozzle: isentropic expansion with shifting equilibrium; the throat is where the flow reaches the (equilibrium)
speed of sound; the exit is where the area has grown by the expansion ratio. That gives the ideal exhaust velocity,
c*, exit pressure and thrust coefficient. A real engine then sits a little below ideal: give its rated level
(sea-level or vacuum Isp, vacuum thrust, or the fraction of ideal) and the block works out the rest, including the
exit area that sets how thrust grows as the outside pressure falls: F = F_vac - p_amb A_e.

Species thermodynamics: NASA 7-coefficient polynomials (McBride, Gordon & Reno, NASA TM-4513; the GRI / NASA
thermo set as distributed with Cantera, 200-6000 K). Equilibrium: Gibbs energy minimised by the element-potential
Newton iteration of Gordon & McBride, NASA RP-1311 (1994), chapter 2.

Propellants enter at their tank enthalpy. That is right for every cycle: the pumps' work and the cooling jacket's heat
come from the propellants' own energy and go back into the chamber, so the injector-inlet enthalpies of a staged
engine (warmer LOX, hot preburner gas) would count that energy twice.

Checked against the Space Shuttle Main Engine (RS-25) at 104.5 % power (cryo-rocket.com SSME station data,
tests/test_rocket_rs25.py): tank enthalpies within 2 and 22 kJ/kg, chamber temperature 3609 K vs 3589 K, water
fraction 0.69 vs 0.67, and from its vacuum rating (491,900 lbf, 452 s) the sea-level thrust and Isp within 0.7 %.
"""
import functools
import math

import numpy as np

from ..registry import block

R_U = 8.314462618               # J/mol/K
P_REF = 101325.0                # Pa: the reference pressure these polynomial fits are referred to (1 atm)
G0 = 9.80665

# name: (atoms, T_mid, high-T coefficients a1..a7, low-T coefficients a1..a7)
NASA7 = {
    "CO2": ({'C': 1.0, 'O': 2.0}, 1000.0, (4.636594930e+00, 2.741319910e-03, -9.958285310e-07, 1.603730110e-10, -9.161034680e-15, -4.902493410e+04, -1.935348550e+00), (2.356773520e+00, 8.984596770e-03, -7.123562690e-06, 2.459190220e-09, -1.436995480e-13, -4.837196970e+04, 9.901052220e+00)),
    "CO": ({'C': 1.0, 'O': 1.0}, 1000.0, (3.048485830e+00, 1.351728180e-03, -4.857940750e-07, 7.885364860e-11, -4.698074890e-15, -1.426611710e+04, 6.017097900e+00), (3.579533470e+00, -6.103536800e-04, 1.016814330e-06, 9.070058840e-10, -9.044244990e-13, -1.434408600e+04, 3.508409280e+00)),
    "H2O": ({'H': 2.0, 'O': 1.0}, 1000.0, (2.677037870e+00, 2.973183290e-03, -7.737696900e-07, 9.443366890e-11, -4.269009590e-15, -2.988589380e+04, 6.882555710e+00), (4.198640560e+00, -2.036434100e-03, 6.520402110e-06, -5.487970620e-09, 1.771978170e-12, -3.029372670e+04, -8.490322080e-01)),
    "H2": ({'H': 2.0}, 1000.0, (2.932865790e+00, 8.266079670e-04, -1.464023350e-07, 1.541003590e-11, -6.888044320e-16, -8.130655970e+02, -1.024328870e+00), (2.344331120e+00, 7.980520750e-03, -1.947815100e-05, 2.015720940e-08, -7.376117610e-12, -9.179351730e+02, 6.830102380e-01)),
    "O2": ({'O': 2.0}, 1000.0, (3.660960830e+00, 6.563655230e-04, -1.411494850e-07, 2.057976580e-11, -1.299132480e-15, -1.215977250e+03, 3.415361840e+00), (3.782456360e+00, -2.996734150e-03, 9.847302000e-06, -9.681295080e-09, 3.243728360e-12, -1.063943560e+03, 3.657675730e+00)),
    "OH": ({'H': 1.0, 'O': 1.0}, 1000.0, (2.838646070e+00, 1.107255860e-03, -2.939149780e-07, 4.205242470e-11, -2.421690920e-15, 3.943958520e+03, 5.844526620e+00), (3.992015430e+00, -2.401317520e-03, 4.617938410e-06, -3.881133330e-09, 1.364114700e-12, 3.615080560e+03, -1.039254580e-01)),
    "H": ({'H': 1.0}, 1000.0, (2.500002860e+00, -5.653342140e-09, 3.632517230e-12, -9.199497200e-16, 7.952607460e-20, 2.547365890e+04, -4.466984940e-01), (2.500000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 2.547365990e+04, -4.466828530e-01)),
    "O": ({'O': 1.0}, 1000.0, (2.543636970e+00, -2.731624860e-05, -4.190295200e-09, 4.954818450e-12, -4.795536940e-16, 2.922601200e+04, 4.922294570e+00), (3.168267100e+00, -3.279318840e-03, 6.643063960e-06, -6.128066240e-09, 2.112659710e-12, 2.912225920e+04, 2.051933460e+00)),
    "HO2": ({'H': 1.0, 'O': 2.0}, 1000.0, (4.172287280e+00, 1.881176470e-03, -3.462774080e-07, 1.946578530e-11, 1.762542940e-16, 6.181029640e+01, 2.957677460e+00), (4.301798010e+00, -4.749120510e-03, 2.115828910e-05, -2.427638940e-08, 9.292251240e-12, 2.948080400e+02, 3.716662450e+00)),
    "H2O2": ({'H': 2.0, 'O': 2.0}, 1000.0, (4.573335370e+00, 4.049840700e-03, -1.294794790e-06, 1.972817100e-10, -1.134028460e-14, -1.805481210e+04, 7.042784880e-01), (4.276112690e+00, -5.428224170e-04, 1.673357010e-05, -2.157708130e-08, 8.624543630e-12, -1.775429890e+04, 3.435050740e+00)),
    # nitrogen species, for storable propellants (N2O4 with hydrazines)
    "N2": ({'N': 2.0}, 1000.0, (2.952576260e+00, 1.396900570e-03, -4.926316910e-07, 7.860103670e-11, -4.607553210e-15, -9.239486450e+02, 5.871892520e+00), (3.531005280e+00, -1.236609870e-04, -5.029994370e-07, 2.435306120e-09, -1.408812350e-12, -1.046976280e+03, 2.967474680e+00)),
    "NO": ({'N': 1.0, 'O': 1.0}, 1000.0, (3.260712340e+00, 1.191011350e-03, -4.291226460e-07, 6.944814630e-11, -4.032956810e-15, 9.921431320e+03, 6.369005180e+00), (4.218598960e+00, -4.639881240e-03, 1.104430490e-05, -9.340555070e-09, 2.805548740e-12, 9.845099640e+03, 2.280610010e+00)),
    "N": ({'N': 1.0}, 1000.0, (2.415942930e+00, 1.748906000e-04, -1.190236670e-07, 3.022623870e-11, -2.036097900e-15, 5.613377480e+04, 4.649609860e+00), (2.500000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 0.000000000e+00, 5.610463780e+04, 4.193909320e+00)),
    "NO2": ({'N': 1.0, 'O': 2.0}, 1000.0, (4.884744290e+00, 2.172416390e-03, -8.280790200e-07, 1.574772930e-10, -1.051105490e-14, 2.316484620e+03, -1.173570750e-01), (3.944039070e+00, -1.585474440e-03, 1.665789840e-05, -2.047544780e-08, 7.835032650e-12, 2.896598650e+03, 6.311962250e+00)),
    "N2O": ({'N': 2.0, 'O': 1.0}, 1000.0, (4.823188730e+00, 2.626852790e-03, -9.584260580e-07, 1.599912960e-10, -9.774169390e-15, 8.073356620e+03, -2.202366000e+00), (2.257168600e+00, 1.130463380e-02, -1.367103500e-05, 9.681620980e-09, -2.930555830e-12, 8.741771460e+03, 1.075791540e+01)),
    "NH3": ({'H': 3.0, 'N': 1.0}, 1000.0, (2.717096920e+00, 5.568563380e-03, -1.768863960e-06, 2.674172600e-10, -1.527314190e-14, -6.584519890e+03, 6.092898370e+00), (4.301778080e+00, -4.771273300e-03, 2.193416190e-05, -2.298564890e-08, 8.289922680e-12, -6.748063940e+03, -6.906443930e-01)),
    "NH2": ({'H': 2.0, 'N': 1.0}, 1000.0, (2.847689920e+00, 3.142800350e-03, -8.986414580e-07, 1.303182840e-10, -7.488129260e-15, 2.182390490e+04, 6.471654330e+00), (4.205568570e+00, -2.135613630e-03, 7.268513010e-06, -5.930698760e-09, 1.806909780e-12, 2.153522310e+04, -1.466627700e-01)),
    "NH": ({'H': 1.0, 'N': 1.0}, 1000.0, (2.783726450e+00, 1.329858860e-03, -4.247855650e-07, 7.834944250e-11, -5.504512980e-15, 4.213451630e+04, 5.740848570e+00), (3.492950370e+00, 3.117957220e-04, -1.489066280e-06, 2.481674030e-09, -1.035709160e-12, 4.189429400e+04, 1.848349740e+00)),
    "HNO": ({'H': 1.0, 'N': 1.0, 'O': 1.0}, 1000.0, (3.165547620e+00, 3.000051320e-03, -3.943502820e-07, -3.857874910e-11, 7.080919310e-15, 1.119441690e+04, 7.647646950e+00), (4.535258820e+00, -5.685469100e-03, 1.851999760e-05, -1.718836740e-08, 5.558330900e-12, 1.103988050e+04, 1.743147340e+00)),
}
SPECIES = list(NASA7)
ELEMENTS = ["C", "H", "O", "N"]
A = np.array([[NASA7[s][0].get(el, 0.0) for s in SPECIES] for el in ELEMENTS])      # atoms per molecule
MW_EL = {"C": 12.0107, "H": 1.00794, "O": 15.9994, "N": 14.0067}
MW = np.array([sum(MW_EL[k] * v for k, v in NASA7[s][0].items()) for s in SPECIES]) * 1e-3   # kg/mol

# propellants: (atoms per mole, molar mass kg/mol, enthalpy J/mol at storage temperature), CEA thermo.inp values
PROPELLANTS = {
    "RP1": ({"C": 1.0, "H": 1.9423}, (12.0107 + 1.9423 * 1.00794) * 1e-3, -24717.7),   # RP-1 as CH1.9423, 298 K
    "LH2": ({"H": 2.0}, 2.01588e-3, -9012.0),                                          # liquid H2, 20.27 K
    "CH4": ({"C": 1.0, "H": 4.0}, 16.04246e-3, -89233.0),                              # liquid CH4, 111.6 K
    "LOX": ({"O": 2.0}, 31.9988e-3, -12979.0),                                         # liquid O2, 90.17 K
    # storables at 298 K: N2O4 and the hydrazines (CEA thermo.inp liquids; UDMH 48.3 kJ/mol)
    "N2O4": ({"N": 2.0, "O": 4.0}, 92.011e-3, -19564.0),
    "N2H4": ({"N": 2.0, "H": 4.0}, 32.0452e-3, 50630.0),
    "UDMH": ({"C": 2.0, "H": 8.0, "N": 2.0}, 60.0983e-3, 48300.0),
    "MMH": ({"C": 1.0, "H": 6.0, "N": 2.0}, 46.0717e-3, 54200.0),
}
# Aerozine 50: half hydrazine, half UDMH by mass, written as one 'molecule' per gram of the blend
_a50_n = (0.5 / 32.0452, 0.5 / 60.0983)                  # mol of N2H4 and of UDMH in 1 g
PROPELLANTS["A50"] = ({"N": 2 * (_a50_n[0] + _a50_n[1]), "H": 4 * _a50_n[0] + 8 * _a50_n[1], "C": 2 * _a50_n[1]},
                      1e-3, _a50_n[0] * 50630.0 + _a50_n[1] * 48300.0)
OXIDIZERS = ("LOX", "N2O4")
FUEL_NAMES = {"rp1": "RP1", "rp-1": "RP1", "kerosene": "RP1", "lh2": "LH2", "h2": "LH2", "hydrogen": "LH2",
              "ch4": "CH4", "methane": "CH4", "lch4": "CH4", "a50": "A50", "a-50": "A50", "aerozine": "A50",
              "aerozine50": "A50", "aerozine-50": "A50", "aerozine 50": "A50", "n2h4": "N2H4", "hydrazine": "N2H4",
              "udmh": "UDMH", "mmh": "MMH", "lox": "LOX", "o2": "LOX", "oxygen": "LOX", "n2o4": "N2O4", "nto": "N2O4"}


def _coef(T):
    """(n_species, 7) coefficients valid at T."""
    return np.array([NASA7[s][2] if T >= NASA7[s][1] else NASA7[s][3] for s in SPECIES])


def _thermo(T):
    """cp/R, h/RT, s0/R for every species at T."""
    a = _coef(T)
    cp = a[:, 0] + a[:, 1] * T + a[:, 2] * T ** 2 + a[:, 3] * T ** 3 + a[:, 4] * T ** 4
    h = a[:, 0] + a[:, 1] * T / 2 + a[:, 2] * T ** 2 / 3 + a[:, 3] * T ** 3 / 4 + a[:, 4] * T ** 4 / 5 + a[:, 5] / T
    s = a[:, 0] * math.log(T) + a[:, 1] * T + a[:, 2] * T ** 2 / 2 + a[:, 3] * T ** 3 / 3 + a[:, 4] * T ** 4 / 4 + a[:, 6]
    return cp, h, s


def equilibrium_TP(T, P, b, n0=None):
    """Mole numbers (mol per kg of mixture) at chemical equilibrium at T, P for element totals b (mol/kg)."""
    on = b > 0                                       # elements present; species made only of those
    sp = np.all(A[~on] == 0, axis=0)
    out = np.zeros(len(SPECIES))
    Aa = A[on][:, sp]; ba = b[on]
    ns = int(sp.sum()); ne = int(on.sum())
    nj = np.full(ns, 0.1 / ns) if n0 is None else np.maximum(n0[sp].copy(), 1e-12)
    n = nj.sum()
    _, h, s = _thermo(T)
    g0 = (h - s)[sp]                                 # mu0/RT
    lnP = math.log(P / P_REF)
    M = np.zeros((ne + 1, ne + 1)); r = np.zeros(ne + 1)
    for it in range(300):
        mu = g0 + np.log(nj / n) + lnP               # mu_j / RT
        # reduced Newton system (RP-1311 eqs. 2.24, 2.26)
        An = Aa * nj
        M[:ne, :ne] = An @ Aa.T
        M[:ne, ne] = An.sum(axis=1); M[ne, :ne] = M[:ne, ne]
        M[ne, ne] = nj.sum() - n
        r[:ne] = ba - M[:ne, ne] + An @ mu
        r[ne] = n - nj.sum() + nj @ mu
        try:
            x = np.linalg.solve(M, r)
        except np.linalg.LinAlgError:              # a poor warm start (all trace species): restart evenly
            nj = np.full(ns, 0.1 / ns); n = nj.sum(); continue
        pi, dlnn = x[:ne], x[ne]
        dlnj = -mu + Aa.T @ pi + dlnn
        # damping (RP-1311 eq. 3.1-3.3): major species steps kept under 2 in ln; trace species bounded
        major = nj / n > 1e-8
        big = max(abs(dlnn), float(np.max(np.abs(dlnj[major]))) if major.any() else 0.0)
        lam = min(1.0, 2.0 / big) if big > 0 else 1.0
        step = np.clip(lam * dlnj, -5.0, 5.0)
        nj = np.clip(nj * np.exp(step), 1e-30, 1e3); n = n * math.exp(lam * dlnn)
        if big * lam < 1e-10:                        # converged in every species that matters (> 1e-8 of the mix)
            break
    out[sp] = nj
    return out


def _props(T, P, nj):
    """h (J/kg), s (J/kg/K), molar mass (kg/mol) of the mixture."""
    cp, h, s0 = _thermo(T)
    n = nj.sum(); x = nj / n
    with np.errstate(divide="ignore"):
        lnx = np.where(x > 0, np.log(np.maximum(x, 1e-300)), 0.0)
    H = R_U * T * np.sum(nj * h)
    S = R_U * np.sum(nj * (s0 - lnx - math.log(P / P_REF)))
    return H, S, 1.0 / n


def _elements(fuel, OF, ox="LOX"):
    f_atoms, f_mw, f_h = PROPELLANTS[fuel]; o_atoms, o_mw, o_h = PROPELLANTS[ox]
    wf = 1.0 / (1.0 + OF); wo = OF / (1.0 + OF)                   # mass fractions
    b = np.zeros(len(ELEMENTS))
    for i, el in enumerate(ELEMENTS):
        b[i] = wf / f_mw * f_atoms.get(el, 0.0) + wo / o_mw * o_atoms.get(el, 0.0)
    h0 = wf / f_mw * f_h + wo / o_mw * o_h                        # J/kg
    return b, h0


def _solve_T(target, which, P, b, lo=300.0, hi=5500.0):
    """Temperature at which the equilibrium mixture has enthalpy (which='h') or entropy ('s') = target."""
    cache = {"n": None}

    def f(T):
        nj = equilibrium_TP(T, P, b, cache["n"]); cache["n"] = nj
        H, S, _ = _props(T, P, nj)
        return (H if which == "h" else S) - target
    flo, fhi = f(lo), f(hi)
    for _ in range(100):                             # regula falsi (Illinois) with a bisection guard
        if hi - lo < 1e-5:
            break
        mid = hi - fhi * (hi - lo) / (fhi - flo) if fhi != flo else 0.5 * (lo + hi)
        if not lo < mid < hi:
            mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < 1e-9 * (abs(target) + 1.0):
            lo = hi = mid; break
        if (fm < 0) == (flo < 0):
            lo, flo = mid, fm; fhi *= 0.5
        else:
            hi, fhi = mid, fm; flo *= 0.5
    T = 0.5 * (lo + hi)
    nj = equilibrium_TP(T, P, b, cache["n"])
    return T, nj


def _state_s(S0, P, b):
    T, nj = _solve_T(S0, "s", P, b)
    H, S, M = _props(T, P, nj)
    rho = P * M / (R_U * T)
    return T, H, rho, M


def _sound(S0, P, b):
    dP = P * 1e-4
    r1 = _state_s(S0, P + dP, b)[2]; r2 = _state_s(S0, P - dP, b)[2]
    return math.sqrt(2 * dP / (r1 - r2))


@functools.lru_cache(maxsize=256)
def ideal_performance(fuel, OF, Pc, eps, oxidizer="LOX"):
    """Ideal shifting-equilibrium performance: Tc (K), c* (m/s), exit pressure (Pa), exhaust velocity (m/s), mass
    flux at the throat (kg/s/m^2), vacuum and sea-level Isp (s) for a nozzle of area ratio eps."""
    b, h0 = _elements(fuel, OF, oxidizer)
    Tc, njc = _solve_T(h0, "h", Pc, b, 1000.0, 5500.0)
    Hc, S0, Mc = _props(Tc, Pc, njc)

    def u_of(P):
        T, H, rho, M = _state_s(S0, P, b)
        return math.sqrt(max(2.0 * (Hc - H), 0.0)), rho
    lo, hi = 0.40 * Pc, 0.75 * Pc                   # throat: flow speed = sound speed
    for _ in range(24):
        mid = 0.5 * (lo + hi); u, _ = u_of(mid)
        if u > _sound(S0, mid, b):
            lo = mid
        else:
            hi = mid
    Pt = 0.5 * (lo + hi); ut, rt = u_of(Pt); Gt = rt * ut
    lo, hi = math.log(Pc * 1e-7), math.log(Pt)      # exit: area ratio = eps
    for _ in range(36):
        mid = 0.5 * (lo + hi); u, r = u_of(math.exp(mid))
        if Gt / (r * u) > eps:
            lo = mid
        else:
            hi = mid
    Pe = math.exp(0.5 * (lo + hi)); ue, _ = u_of(Pe)
    Isp_vac = (ue + Pe * eps / Gt) / G0
    Isp_sl = (ue + (Pe - 101325.0) * eps / Gt) / G0
    x = njc / njc.sum()
    return dict(Tc=Tc, cstar=Pc / Gt, Pe=Pe, ue=ue, Gt=Gt, Isp_vac=Isp_vac, Isp_sl=Isp_sl, M_chamber=Mc,
                composition={SPECIES[j]: float(x[j]) for j in range(len(SPECIES)) if x[j] > 1e-3})


@block("actuators", "rocket", "liquid", "equilibrium", kind="builder",
       source="NASA RP-1311 (Gordon & McBride 1994) equilibrium method; NASA TM-4513 thermodynamic data",
       tags=("rocket", "engine", "thrust", "Isp", "nozzle", "combustion", "F-1", "J-2", "Merlin", "RP-1", "LOX",
             "hydrogen", "methane", "CEA", "N2O4", "Aerozine", "hydrazine", "UDMH", "MMH", "storable", "DPS", "LM",
             "RS-25", "SSME", "Space Shuttle", "staged combustion"))
def rocket_engine(fuel, OF, Pc, eps, mdot=None, Isp_sl=None, Isp_vac=None, F_vac=None, efficiency=None,
                  eta_cstar=0.98, n=1, oxidizer="LOX"):
    """Liquid rocket engine from its propellants: thrust against outside pressure F = F_vac - p_amb A_e.
    fuel: RP1 (kerosene), LH2, CH4, A50 (Aerozine 50), N2H4, UDMH or MMH, burnt with oxidizer LOX (liquid oxygen,
    the default) or N2O4 at mixture ratio OF (oxidiser / fuel by mass);
    Pc chamber pressure; eps nozzle area ratio. The chemistry gives the ideal Isp; the real engine's level comes from
    one rating: Isp_sl, Isp_vac, F_vac (with mdot) or efficiency (delivered / ideal vacuum Isp). mdot or F_vac sets
    the size. eta_cstar (combustion efficiency, 0.98 typical) sizes the throat, and so the exit area A_e.
    n engines: thrust, flow and exit area are totals for all of them. Returns F_vac, F_sl, A_e, mdot, Isp_vac,
    Isp_sl (delivered), Isp_vac_ideal, Isp_sl_ideal, efficiency, Tc, cstar, Pe."""
    key = FUEL_NAMES.get(str(fuel).strip().lower(), str(fuel).strip().upper())
    ox = FUEL_NAMES.get(str(oxidizer).strip().lower(), str(oxidizer).strip().upper())
    if key not in PROPELLANTS or key in OXIDIZERS:
        raise ValueError("fuel is RP1 (kerosene), LH2 (hydrogen), CH4 (methane), A50 (Aerozine 50), N2H4, UDMH or MMH")
    if ox not in OXIDIZERS:
        raise ValueError("oxidizer is LOX (liquid oxygen) or N2O4 (nitrogen tetroxide)")
    if not (OF > 0 and Pc > 0 and eps > 1):
        raise ValueError("OF > 0, Pc > 0 and eps > 1 are needed")
    ideal = ideal_performance(key, float(OF), float(Pc), float(eps), ox)
    p0 = 101325.0
    rated = [k for k, x in (("Isp_sl", Isp_sl), ("Isp_vac", Isp_vac), ("efficiency", efficiency)) if x is not None]
    if len(rated) > 1:
        raise ValueError("give one rating: Isp_sl, Isp_vac or efficiency (or F_vac with mdot)")
    if mdot is None and F_vac is None:
        raise ValueError("give the engine's size: mdot (propellant flow) or F_vac (vacuum thrust)")
    if not rated and (mdot is None or F_vac is None):
        raise ValueError("give a rating (Isp_sl, Isp_vac or efficiency), or both F_vac and mdot")

    def exit_area(md):
        return eps * md * ideal["cstar"] * eta_cstar / Pc          # A_e = eps A_t,  A_t = mdot c* / Pc
    if Isp_sl is not None:                                          # sea-level rating: Isp_vac follows from A_e
        if mdot is None:
            # F_vac = mdot g0 Isp_sl + p0 eps mdot c* eta / Pc  ->  mdot
            mdot = F_vac / (G0 * Isp_sl + p0 * eps * ideal["cstar"] * eta_cstar / Pc)
        Ae = exit_area(mdot)
        Iv = Isp_sl + p0 * Ae / (mdot * G0)
    else:
        if Isp_vac is not None:
            Iv = Isp_vac
        elif efficiency is not None:
            Iv = efficiency * ideal["Isp_vac"]
        else:
            Iv = F_vac / (mdot * G0)
        if mdot is None:
            mdot = F_vac / (Iv * G0)
        Ae = exit_area(mdot)
    Fv = mdot * G0 * Iv
    Fs = Fv - p0 * Ae
    return dict(F_vac=n * Fv, F_sl=n * Fs, A_e=n * Ae, mdot=n * mdot, Isp_vac=Iv, Isp_sl=Fs / (mdot * G0),
                Isp_vac_ideal=ideal["Isp_vac"], Isp_sl_ideal=ideal["Isp_sl"], efficiency=Iv / ideal["Isp_vac"],
                Tc=ideal["Tc"], cstar=ideal["cstar"], Pe=ideal["Pe"], M_chamber=ideal["M_chamber"],
                composition=ideal["composition"])


@block("aero", "trajectory", "launch_plane", "site", kind="builder", source="spherical Earth rotation geometry",
       tags=("launch", "rotation", "azimuth", "latitude", "orbit"))
def launch_plane(latitude, azimuth, omega=7.2921159e-5, R=6371000.0):
    """A launch flown in one plane (a 2d model): the part of the Earth's spin about that plane's normal, and the
    speed the pad already has along the launch direction. latitude and azimuth (from north, clockwise) in radians.
    Returns omega_plane (rad/s, the planet's in-plane rotation for gravity: {rotation: ...}), v_surface (m/s, the
    pad's speed along the flight direction: the body's starting v0) and v_cross (the part out of the plane, which a
    2d model leaves out)."""
    w_p = omega * math.cos(latitude) * math.sin(azimuth)
    v_e = omega * R * math.cos(latitude)
    return dict(omega_plane=w_p, v_surface=w_p * R, v_cross=v_e * math.cos(azimuth), v_east=v_e)
