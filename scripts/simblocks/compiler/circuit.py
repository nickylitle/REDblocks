"""Lumped electrical circuits solved in step with the mechanics.

A model may carry a `circuit:` section: parts wired between named nets, one net the ground (0 V).

  circuit:
    ground: gnd
    parts:
      pack:  {type: battery, between: [bus, gnd], cells: 6, capacity: 650 mAh, R0: 25 mohm}
      fuse:  {type: fuse, between: [bus, sw], I_rating: 40 A, t_blow: 0.5 s}
      wpn:   {type: motor, between: [sw, gnd], kv: 1350 rpm/V, R: 45 mohm, L: 15 uH, measure: blade,
              esc: {throttle: 1, I_max: 60 A}}

Every part is a lumped law. Each time step the circuit is solved by nodal analysis: capacitors, inductors, the
battery's RC branch and the motors' winding inductance are replaced by their backward-Euler companions (a
conductance in parallel with a current source), the resulting linear system G v = I is solved for the net
voltages, and the switching parts (diodes, ESC current limits, a motor that would draw current backwards
through its ESC) are settled by re-solving until their states agree (a few passes). Part currents follow
from the net voltages, and part states (capacitor voltage, inductor current, charge, fuse heating) advance.

Sign convention (passive): a part's current i flows from its first net to its second, through the part; its
voltage is v = v(first) - v(second); power p = v i is absorbed (negative for a source). The powers of all parts
sum to zero every step (Tellegen), which is the circuit's own energy ledger.

Motors couple to the mechanics like the `drive` element: the winding current i makes torque Kt i on the motor
shaft (gear G times that on the measured coordinate), and the coordinate's speed makes the back-EMF
e = Kt G (ds/dt). The circuit reads each motor's speed from the previous step and sets its torque for this one.
"""
import math

import numpy as np
from numba import njit

from .loader import SpecError, line_of

# part types
R_, C_, L_, VS, IS, BAT, DIODE, SW, FUSE, MOT, ESC = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
TYPES = {"resistor": R_, "capacitor": C_, "inductor": L_, "voltage_source": VS, "current_source": IS, "current_load": IS,
         "battery": BAT, "diode": DIODE, "switch": SW, "fuse": FUSE, "motor": MOT}
FIELDS = {
    "resistor": {"R"}, "capacitor": {"C", "v0", "R_leak", "ESR"}, "inductor": {"L", "R", "i0"},
    "voltage_source": {"V", "R", "I_load"}, "current_source": {"I"}, "current_load": {"I"},
    "battery": {"cells", "capacity", "R0", "R1", "C1", "soc0", "V_full", "V_empty", "I_load"},
    "diode": {"Vf", "R_on", "R_off"},
    "switch": {"R_on", "R_off", "on_at", "off_at", "pwm_freq", "duty", "phase"},
    "fuse": {"R", "I_rating", "t_blow", "I_trip"},
    "motor": {"kv", "Kt", "R", "L", "gear", "measure", "esc", "J_rotor"},
}
REQUIRED = {"resistor": {"R"}, "capacitor": {"C"}, "inductor": {"L"}, "voltage_source": {"V"},
            "current_source": {"I"}, "current_load": {"I"}, "battery": {"cells", "capacity"}, "diode": set(), "switch": set(),
            "fuse": {"I_rating"}, "motor": {"R", "measure"}}
ESC_FIELDS = {"throttle", "I_max", "regen"}
COMMON = {"type", "between", "label", "note", "source", "thermal"}
QUANTITIES = {"current", "voltage", "power", "energy", "soc", "state", "torque", "temperature", "heat"}
THERMAL = {"C_th", "R_th", "T_amb_C", "T0_C", "alpha", "T_max_C", "kt_tempco", "vf_tempco"}
HAS_THERMAL = {"resistor", "capacitor", "inductor", "battery", "diode", "switch", "fuse", "motor"}
# typical temperature coefficients when a thermal model is on and none is given
ALPHA_DEFAULT = {"motor": 0.00393, "inductor": 0.00393}
KT_TEMPCO_DEFAULT = -0.0011          # NdFeB magnets: about -0.11 % per K
VF_TEMPCO_DEFAULT = -0.002           # silicon diode: about -2 mV per K
NPRM = 12; NST = 6; NOUT = 8       # out: current, voltage, power, energy, soc / state, torque, T (C), heat


def compile_circuit(doc, ctx):
    """Read doc['circuit'] into arrays. Returns (circ or None, hidden elements to add): one 'emotor' element per
    motor, so the mechanical compilers give it its measure like any element."""
    from .model import resolve, num
    c = doc.get("circuit")
    if not c:
        return None, {}
    f = doc.get("__file__", "")
    if not isinstance(c, dict) or "parts" not in c:
        raise SpecError("circuit is {ground: <net>, parts: {name: {type: ..., between: [net, net], ...}}}",
                        "circuit", None, f)
    bad = set(c) - {"ground", "parts", "note", "label"}
    if bad:
        raise SpecError("unknown circuit key(s): " + ", ".join(sorted(bad)), "circuit", None, f)
    gnd = str(c.get("ground", "gnd"))
    parts = c["parts"] or {}
    if not isinstance(parts, dict) or not parts:
        raise SpecError("circuit.parts needs at least one part", "circuit.parts", None, f)
    # a battery's or a supply's I_load: a fixed current drawn straight across its terminals (avionics, a receiver, a
    # constant bench load). It runs as its own current_load part, <name>__load, which its outputs can name
    if any(isinstance(d, dict) and d.get("I_load") not in (None, "") for d in parts.values()):
        p2 = {}
        for pn, d in parts.items():
            if isinstance(d, dict) and d.get("I_load") not in (None, ""):
                d = dict(d); il = d.pop("I_load")
                p2[pn] = d
                p2[f"{pn}__load"] = {"type": "current_load", "between": d.get("between"), "I": il,
                                     "label": f"{pn}'s fixed load"}
            else:
                p2[pn] = d
        parts = p2
    nets = {gnd: 0}
    names = list(parts)
    npart = len(names)
    ptype = np.zeros(npart, np.int64); na = np.zeros(npart, np.int64); nb = np.zeros(npart, np.int64)
    prm = np.zeros((npart, NPRM)); pst = np.zeros((npart, NST)); th = np.zeros((npart, 8))
    extra = {}; motors = []
    for k, pn in enumerate(names):
        d = parts[pn]; where = f"circuit.parts.{pn}"
        if not isinstance(d, dict) or "type" not in d:
            raise SpecError("a part is {type: ..., between: [net, net], ...}", where, line_of(parts, pn), f)
        t = str(d["type"])
        if t not in TYPES:
            raise SpecError(f"unknown part type '{t}' (allowed: {', '.join(sorted(TYPES))})", where,
                            line_of(d, "type"), f)
        for key in set(d) - FIELDS[t] - COMMON:
            raise SpecError(f"'{key}' is not a field of a {t} (allowed: {', '.join(sorted(FIELDS[t] | COMMON))})",
                            where, line_of(d, key), f)
        for key in REQUIRED[t] - set(d):
            raise SpecError(f"a {t} needs '{key}'", where, line_of(parts, pn), f)
        bw = d.get("between")
        if not (isinstance(bw, (list, tuple)) and len(bw) == 2):
            raise SpecError("between is [net, net] (first = the + side for sources, batteries and motors)", where,
                            line_of(d, "between"), f)
        a_, b_ = str(bw[0]), str(bw[1])
        if a_ == b_:
            raise SpecError(f"a part between {a_} and itself carries nothing", where, line_of(d, "between"), f)
        for x in (a_, b_):
            if x not in nets:
                nets[x] = len(nets)
        na[k] = nets[a_]; nb[k] = nets[b_]
        g = lambda key, dflt=None: (num(resolve(d[key], ctx, f"{where}.{key}"), f"{where}.{key}") if key in d
                                    else dflt)
        tp = TYPES[t]; P = prm[k]
        if t == "resistor":
            P[0] = g("R")
            if P[0] <= 0:
                raise SpecError("R must be positive (use a switch for an open or a closed contact)", where,
                                line_of(d, "R"), f)
        elif t == "capacitor":
            P[0] = g("C"); pst[k, 0] = g("v0", 0.0); P[1] = g("R_leak", 0.0); P[2] = g("ESR", 0.0)
            if P[0] <= 0:
                raise SpecError("C must be positive", where, line_of(d, "C"), f)
        elif t == "inductor":
            P[0] = g("L"); P[1] = g("R", 0.0); pst[k, 0] = g("i0", 0.0)
            if P[0] <= 0:
                raise SpecError("L must be positive", where, line_of(d, "L"), f)
        elif t == "voltage_source":
            P[0] = g("V"); P[1] = g("R", 1e-4)
            if P[1] <= 0:
                raise SpecError("a voltage source's R (internal resistance) must be positive", where,
                                line_of(d, "R"), f)
        elif t == "current_source":
            P[0] = g("I")
        elif t == "current_load":                      # a fixed current drawn in at + and out at - (a sink)
            P[0] = -g("I")
        elif t == "battery":
            P[0] = g("cells"); P[1] = g("capacity"); P[2] = g("R0", 0.0); P[3] = g("R1", 0.0); P[4] = g("C1", 0.0)
            P[5] = g("V_full", 4.2); P[6] = g("V_empty", 3.3)
            pst[k, 0] = g("soc0", 1.0)
            if P[1] <= 0 or P[0] <= 0:
                raise SpecError("a battery needs cells > 0 and capacity > 0", where, line_of(parts, pn), f)
            if P[2] <= 0 and P[3] <= 0:
                P[2] = 1e-4                         # an ideal source needs some resistance to be a Norton source
        elif t == "diode":
            P[0] = g("Vf", 0.7); P[1] = g("R_on", 0.01); P[2] = g("R_off", 1e7)
        elif t == "switch":
            P[0] = g("R_on", 1e-3); P[1] = g("R_off", 1e8)
            P[2] = g("on_at", 0.0); P[3] = g("off_at", 1e300)
            P[4] = g("pwm_freq", 0.0); P[5] = g("duty", 1.0); P[6] = g("phase", 0.0)
            if not 0.0 <= P[5] <= 1.0:
                raise SpecError("duty is a fraction, 0 to 1", where, line_of(d, "duty"), f)
        elif t == "fuse":
            P[0] = g("R", 1e-3); P[1] = g("I_rating"); P[2] = g("t_blow", 1.0); P[3] = g("I_trip", 1e300)
        elif t == "motor":
            if ("kv" in d) == ("Kt" in d):
                raise SpecError("a motor needs kv (e.g. 1350 rpm/V) or Kt (N m/A), one of them", where,
                                line_of(parts, pn), f)
            Kt = g("Kt") if "Kt" in d else 1.0 / g("kv")
            P[0] = Kt; P[1] = g("R"); P[2] = g("L", 0.0); P[3] = g("gear", 1.0)
            if P[1] <= 0:
                raise SpecError("a motor's R (winding resistance) must be positive", where, line_of(d, "R"), f)
            esc = d.get("esc")
            if esc is not None:
                if not isinstance(esc, dict):
                    raise SpecError("esc is {throttle: 0-1, I_max: <current>, regen: false}", where, line_of(d, "esc"), f)
                for key in set(esc) - ESC_FIELDS:
                    raise SpecError(f"'{key}' is not an esc field (allowed: {', '.join(sorted(ESC_FIELDS))})", where,
                                    line_of(esc, key), f)
                tp = ESC
                P[4] = num(resolve(esc.get("throttle", 1.0), ctx, where + ".esc.throttle"), where + ".esc.throttle")
                P[5] = num(resolve(esc.get("I_max", 1e9), ctx, where + ".esc.I_max"), where + ".esc.I_max")
                P[6] = 1.0 if esc.get("regen") is True else 0.0
                if not 0.0 <= P[4] <= 1.0:
                    raise SpecError("throttle is a fraction of full duty, 0 to 1", where, line_of(esc, "throttle"), f)
            P[7] = len(motors)
            motors.append(pn)
            el = {"type": "emotor", "measure": d["measure"], "__part__": pn, "label": d.get("label", pn),
                  "gear": float(P[3]), "__kt__": float(Kt), "__idx__": float(len(motors) - 1), "__R__": float(P[1])}
            if "J_rotor" in d:
                el["J_rotor"] = d["J_rotor"]
            extra[f"__circ_{pn}"] = el
        ptype[k] = tp
        pst[k, 4] = 298.15
        thd = d.get("thermal")
        if thd is not None:
            if t not in HAS_THERMAL:
                raise SpecError(f"a {t} has no thermal model", where, line_of(d, "thermal"), f)
            if not isinstance(thd, dict):
                raise SpecError("thermal is {C_th: <J/K>, R_th: <K/W>, T_amb_C: 25, alpha: <1/K>, T_max_C: ...}",
                                where, line_of(d, "thermal"), f)
            for key in set(thd) - THERMAL:
                raise SpecError(f"'{key}' is not a thermal field (allowed: {', '.join(sorted(THERMAL))})", where,
                                line_of(thd, key), f)
            for key in ("C_th", "R_th"):
                if key not in thd:
                    raise SpecError(f"thermal needs {key} ({'heat capacity, J/K' if key == 'C_th' else 'thermal resistance to ambient, K/W'})",
                                    where, line_of(d, "thermal"), f)
            gt = lambda key, dflt: num(resolve(thd[key], ctx, f"{where}.thermal.{key}"), f"{where}.thermal.{key}") \
                if key in thd else dflt
            Ta = gt("T_amb_C", 25.0) + 273.15
            th[k] = [1.0, gt("C_th", 1.0), gt("R_th", 1.0), Ta, gt("alpha", ALPHA_DEFAULT.get(t, 0.0)),
                     gt("T_max_C", 1e6) + 273.15, gt("kt_tempco", KT_TEMPCO_DEFAULT if t == "motor" else 0.0),
                     gt("vf_tempco", VF_TEMPCO_DEFAULT if t == "diode" else 0.0)]
            if th[k, 1] <= 0 or th[k, 2] <= 0:
                raise SpecError("C_th and R_th must be positive", where, line_of(d, "thermal"), f)
            pst[k, 4] = gt("T0_C", Ta - 273.15) + 273.15
    nn = len(nets) - 1
    if nn == 0:
        raise SpecError("the circuit has only the ground net", "circuit", None, f)
    # every net must reach ground through parts (a floating island has no defined voltage)
    adj = {i: set() for i in range(len(nets))}
    for k in range(npart):
        adj[int(na[k])].add(int(nb[k])); adj[int(nb[k])].add(int(na[k]))
    seen = {0}; todo = [0]
    while todo:
        x = todo.pop()
        for y in adj[x]:
            if y not in seen:
                seen.add(y); todo.append(y)
    lost = [n_ for n_, i in nets.items() if i not in seen]
    if lost:
        raise SpecError(f"net(s) {', '.join(lost)} do not connect to the ground net '{gnd}'", "circuit", None, f)
    circ = dict(names=names, nets=nets, ground=gnd, motors=motors, types=[str(parts[p_]["type"]) for p_ in names],
                arrays=dict(ptype=ptype, na=na, nb=nb, prm=prm, pst0=pst, th=th, nn=nn))
    return circ, extra


def circuit_tau(circ):
    """The circuit's fastest time scale, for the automatic step: each capacitor's C (ESR + the smallest other
    resistance), each inductor's or motor's L / (R + that), the battery's R1 C1, and a PWM switch's period.
    Backward Euler is stable at any step; this keeps the fast transients (inrush, ripple) resolved."""
    a = circ["arrays"]; pt, prm = a["ptype"], a["prm"]
    rs = []
    for k in range(len(pt)):
        t = pt[k]; P = prm[k]
        r = {R_: P[0], C_: P[2], L_: P[1], VS: P[1], BAT: P[2] + P[3], DIODE: P[1], SW: P[0], FUSE: P[0],
             MOT: P[1], ESC: P[1]}.get(int(t), 0.0)
        rs.append(r if r > 0 else math.inf)
    taus = []
    for k in range(len(pt)):
        t = int(pt[k]); P = prm[k]
        others = [rs[j] for j in range(len(pt)) if j != k]
        rmin = min(others) if others and math.isfinite(min(others)) else 0.0
        if t == C_:
            taus.append(P[0] * (P[2] + rmin))
        elif t in (L_, MOT, ESC) and P[0 if t == L_ else 2] > 0:
            L = P[0] if t == L_ else P[2]
            taus.append(L / max((P[1] if t == L_ else P[1]) + rmin, 1e-9))
        elif t == BAT and P[3] > 0 and P[4] > 0:
            taus.append(P[3] * P[4])
        elif t == SW and P[4] > 0:
            taus.append(1.0 / P[4])
    taus = [x for x in taus if x > 0]
    return min(taus) if taus else math.inf


@njit(cache=True, nogil=True)
def _gauss(A, b, n, out):
    """Solve A x = b in place (Gaussian elimination, partial pivoting); x goes to out[1:]. A few nets: faster than
    a LAPACK call per pass."""
    for c in range(n):
        p = c; best = abs(A[c, c])
        for r in range(c + 1, n):
            if abs(A[r, c]) > best:
                best = abs(A[r, c]); p = r
        if p != c:
            for j in range(c, n):
                A[c, j], A[p, j] = A[p, j], A[c, j]
            b[c], b[p] = b[p], b[c]
        piv = A[c, c]
        for r in range(c + 1, n):
            f = A[r, c] / piv
            if f != 0.0:
                for j in range(c, n):
                    A[r, j] -= f * A[c, j]
                b[r] -= f * b[c]
    for r in range(n - 1, -1, -1):
        acc = b[r]
        for j in range(r + 1, n):
            acc -= A[r, j] * out[j + 1]
        out[r + 1] = acc / A[r, r]


@njit(cache=True, nogil=True, inline="always")
def _ocv(soc, cells, vfull, vempty):
    """LiPo-shaped open-circuit voltage between V_empty and V_full per cell (the same table as the drive)."""
    xs = (0.0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0)
    fr = (0.0, 0.25 / 0.9, 0.38 / 0.9, 0.45 / 0.9, 0.65 / 0.9, 0.80 / 0.9, 1.0)
    if soc <= 0.0:
        return cells * vempty
    if soc >= 1.0:
        return cells * vfull
    for i in range(6):
        if soc < xs[i + 1]:
            u = fr[i] + (fr[i + 1] - fr[i]) * (soc - xs[i]) / (xs[i + 1] - xs[i])
            return cells * (vempty + (vfull - vempty) * u)
    return cells * vfull


T_REF = 298.15                       # 25 C: the temperature the given R, Kt and Vf are quoted at


@njit(cache=True, nogil=True, inline="always")
def _rf(k, th, pst):
    """Resistance factor 1 + alpha (T - 25 C) for a part with a thermal model (else 1)."""
    if th[k, 0] < 0.5:
        return 1.0
    f = 1.0 + th[k, 4] * (pst[k, 4] - T_REF)
    return f if f > 0.05 else 0.05


@njit(cache=True, nogil=True, inline="always")
def _kt(k, prm, th, pst):
    """Motor torque constant, weakened as the magnets warm (kt_tempco per K)."""
    if th[k, 0] < 0.5:
        return prm[k, 0]
    f = 1.0 + th[k, 6] * (pst[k, 4] - T_REF)
    return prm[k, 0] * (f if f > 0.05 else 0.05)


@njit(cache=True, nogil=True, inline="always")
def _companion(k, ptype, prm, pst, th, flg, t, dt, emf):
    """Norton companion of part k for this solve: (G, J) with i = G v - J (passive sign, i from first to
    second net). flg carries the part's switching state for this pass."""
    tp = ptype[k]
    if pst[k, 5] > 0.5 and tp != 4 and tp != 5:
        return 1e-9, 0.0                               # failed (overheated, or a blown fuse): open
    rf = _rf(k, th, pst)
    if tp == 1:
        return 1.0 / (prm[k, 0] * rf), 0.0
    if tp == 2:                                        # v = ESR i + v_c, i = C (v_c - v_prev)/dt (+ leak)
        Z = prm[k, 2] * rf + dt / prm[k, 0]
        gl = 1.0 / prm[k, 1] if prm[k, 1] > 0.0 else 0.0
        return 1.0 / Z + gl, pst[k, 0] / Z
    if tp == 3:                                        # v = R i + L (i - i_prev)/dt
        Z = prm[k, 1] * rf + prm[k, 0] / dt
        return 1.0 / Z, -(prm[k, 0] / dt) * pst[k, 0] / Z
    if tp == 4:                                        # v = V - R (-i)  ->  i = (v - V)/R
        return 1.0 / prm[k, 1], prm[k, 0] / prm[k, 1]
    if tp == 5:                                        # pushes I into the first net: i = -I
        return 0.0, prm[k, 0]
    if tp == 6:                                        # battery: v = OCV - v1 - R0 (-i), v1 BE-advanced
        ocv = _ocv(pst[k, 0], prm[k, 0], prm[k, 5], prm[k, 6])
        R1 = prm[k, 3] * rf
        if R1 > 0.0 and prm[k, 4] > 0.0:
            den = 1.0 + dt / (R1 * prm[k, 4])
            a = pst[k, 1] / den; b = (dt / prm[k, 4]) / den
        elif R1 > 0.0:
            a = 0.0; b = R1
        else:
            a = 0.0; b = 0.0
        Rt = prm[k, 2] * rf + b                             # terminal: v = (OCV - a) + Rt i
        return 1.0 / Rt, (ocv - a) / Rt
    if tp == 7:                                        # diode: Vf + R_on when on, R_off when off
        if flg[k] > 0.5:
            vf = prm[k, 0] + (th[k, 7] * (pst[k, 4] - T_REF) if th[k, 0] > 0.5 else 0.0)
            return 1.0 / (prm[k, 1] * rf), vf / (prm[k, 1] * rf)
        return 1.0 / prm[k, 2], 0.0
    if tp == 8:
        on = t >= prm[k, 2] and t < prm[k, 3]
        if on and prm[k, 4] > 0.0:
            ph = (t * prm[k, 4] + prm[k, 6]) % 1.0
            on = ph < prm[k, 5]
        return (1.0 / (prm[k, 0] * rf) if on else 1.0 / prm[k, 1]), 0.0
    if tp == 9:
        return 1.0 / (prm[k, 0] * rf), 0.0
    if tp == 10:                                       # motor: v = R i + L di/dt + e
        Z = prm[k, 1] * rf + prm[k, 2] / dt
        return 1.0 / Z, (emf - (prm[k, 2] / dt) * pst[k, 0]) / Z
    if tp == 11:                                       # motor behind an averaged ESC, duty d = flg
        d = flg[k]
        if d <= 0.0:
            return 0.0, 0.0
        Z = prm[k, 1] * rf + prm[k, 2] / dt
        Gm = 1.0 / Z
        # i_m = Gm (d v - emf + (L/dt) i_prev); bus current i = d i_m
        return d * d * Gm, d * Gm * (emf - (prm[k, 2] / dt) * pst[k, 0])
    return 0.0, 0.0


@njit(cache=True, nogil=True)
def _motor_i(k, ptype, prm, pst, th, flg, dt, emf, v):
    """Winding current of motor part k for terminal (or bus, behind an ESC) voltage v."""
    Z = prm[k, 1] * _rf(k, th, pst) + prm[k, 2] / dt
    d = 1.0 if ptype[k] == 10 else flg[k]
    return (d * v - emf + (prm[k, 2] / dt) * pst[k, 0]) / Z


@njit(cache=True, nogil=True)
def circuit_step(t, dt, nn, ptype, na, nb, prm, pst, th, pout, mot_el, st, flg, Gm, Iv, vnet):
    """Advance the circuit one step. st holds the mechanical side: st[e, 2] = the motor element's measure rate
    (ds/dt) from the previous step; on return st[e, 0] = torque on the measure (gear x Kt x i), st[e, 1] = winding
    current, st[e, 3] = motor terminal voltage. pout[k] = current, voltage, power, energy, soc / state, torque,
    temperature (C), heat (J)."""
    npart = ptype.shape[0]
    emf = np.zeros(npart)
    for k in range(npart):
        if ptype[k] == 10 or ptype[k] == 11:
            e = mot_el[int(prm[k, 7])]
            emf[k] = _kt(k, prm, th, pst) * prm[k, 3] * st[e, 2]     # Kt G ds/dt (the engines set it to the
                                                                     # step's end rate before this call)
        if ptype[k] == 11:
            flg[k] = prm[k, 4] if pst[k, 5] < 0.5 else 0.0          # each step starts at the set throttle
    for it in range(8):
        for r in range(nn):
            Iv[r] = 0.0
            for c in range(nn):
                Gm[r, c] = 0.0
        for k in range(npart):
            G, J = _companion(k, ptype, prm, pst, th, flg, t, dt, emf[k])
            a = na[k] - 1; b = nb[k] - 1
            if a >= 0:
                Gm[a, a] += G; Iv[a] += J
            if b >= 0:
                Gm[b, b] += G; Iv[b] -= J
            if a >= 0 and b >= 0:
                Gm[a, b] -= G; Gm[b, a] -= G
        for r in range(nn):
            Gm[r, r] += 1e-12                                   # a hair of leakage keeps a floating net defined
        _gauss(Gm, Iv, nn, vnet)
        vnet[0] = 0.0
        changed = False
        for k in range(npart):
            tp = ptype[k]
            v = vnet[na[k]] - vnet[nb[k]]
            if tp == 7:
                vf = prm[k, 0] + (th[k, 7] * (pst[k, 4] - T_REF) if th[k, 0] > 0.5 else 0.0)
                want = 1.0 if v > vf else 0.0
                if want != flg[k]:
                    flg[k] = want; changed = True
            elif tp == 11 and pst[k, 5] < 0.5:
                d = flg[k]
                Z = prm[k, 1] * _rf(k, th, pst) + prm[k, 2] / dt
                im = (d * v - emf[k] + (prm[k, 2] / dt) * pst[k, 0]) / Z
                Imax = prm[k, 5]
                if im > Imax * 1.000001 and v > 0.0:
                    dn = (Imax * Z + emf[k] - (prm[k, 2] / dt) * pst[k, 0]) / v
                    if dn < 0.0:
                        dn = 0.0
                    if dn > prm[k, 4]:
                        dn = prm[k, 4]
                    if abs(dn - d) > 1e-7 * dn + 1e-12:              # settled to 0.1 ppm of the duty
                        flg[k] = dn; changed = True
                elif im < 0.0 and prm[k, 6] < 0.5 and d > 0.0:
                    flg[k] = 0.0; changed = True                # no regen: the ESC stops driving (freewheels)
        if not changed:
            break
    # each motor's back-EMF damping on its shaft, from the circuit as it is this step: how much its current drops per
    # volt of extra back-EMF (one more solve with that motor's EMF nudged; the circuit is linear once the switches
    # are set). The engines' exact motor step reads it (engine1d motor_k): kg^2 x this, in N m s / rad.
    cdamp = np.zeros(npart); vtmp = np.zeros(nn + 1)
    for km in range(npart):
        tm = ptype[km]
        if (tm != 10 and tm != 11) or pst[km, 5] > 0.5:
            continue
        if tm == 11 and not (flg[km] > 0.0 and flg[km] >= prm[km, 4] * (1.0 - 1e-6)):
            continue                                            # at the current limit or freewheeling: no damping
        de = 1e-3 * (abs(emf[km]) + 1.0)
        i0 = _motor_i(km, ptype, prm, pst, th, flg, dt, emf[km], vnet[na[km]] - vnet[nb[km]])
        emf[km] += de
        for r in range(nn):
            Iv[r] = 0.0
            for c in range(nn):
                Gm[r, c] = 0.0
        for k in range(npart):
            G, J = _companion(k, ptype, prm, pst, th, flg, t, dt, emf[k])
            a = na[k] - 1; b = nb[k] - 1
            if a >= 0:
                Gm[a, a] += G; Iv[a] += J
            if b >= 0:
                Gm[b, b] += G; Iv[b] -= J
            if a >= 0 and b >= 0:
                Gm[a, b] -= G; Gm[b, a] -= G
        for r in range(nn):
            Gm[r, r] += 1e-12
        _gauss(Gm, Iv, nn, vtmp)
        vtmp[0] = 0.0
        i1 = _motor_i(km, ptype, prm, pst, th, flg, dt, emf[km], vtmp[na[km]] - vtmp[nb[km]])
        emf[km] -= de
        cdamp[km] = (i0 - i1) / de if i0 > i1 else 0.0
    # currents, energy, heat, states
    for k in range(npart):
        tp = ptype[k]
        v = vnet[na[k]] - vnet[nb[k]]
        G, J = _companion(k, ptype, prm, pst, th, flg, t, dt, emf[k])
        i = G * v - J
        rf = _rf(k, th, pst)
        pout[k, 0] = i; pout[k, 1] = v; pout[k, 2] = v * i; pout[k, 3] += v * i * dt
        heat = 0.0
        if pst[k, 5] > 0.5 and tp != 4 and tp != 5:
            pass
        elif tp == 1 or tp == 7 or tp == 8 or tp == 9:
            heat = v * i
        elif tp == 2:
            pst[k, 0] += i * dt / prm[k, 0]
            heat = prm[k, 2] * rf * i * i
        elif tp == 3:
            pst[k, 0] = i
            heat = prm[k, 1] * rf * i * i
        elif tp == 6:
            iout = -i
            R1 = prm[k, 3] * rf
            if R1 > 0.0 and prm[k, 4] > 0.0:
                pst[k, 1] = (pst[k, 1] + dt * iout / prm[k, 4]) / (1.0 + dt / (R1 * prm[k, 4]))
            elif R1 > 0.0:
                pst[k, 1] = R1 * iout
            heat = prm[k, 2] * rf * iout * iout + (pst[k, 1] * pst[k, 1] / R1 if R1 > 0.0 else 0.0)
            pst[k, 0] -= iout * dt / prm[k, 1]
            pout[k, 4] = pst[k, 0]
        if tp == 9:
            if pst[k, 5] < 0.5:
                pst[k, 0] += i * i * dt
                if pst[k, 0] > prm[k, 1] * prm[k, 1] * prm[k, 2] or abs(i) > prm[k, 3]:
                    pst[k, 5] = 1.0                             # blown
            pout[k, 4] = pst[k, 5]
        elif tp == 10 or tp == 11:
            if tp == 10:
                im = i; vm = v
            else:
                d = flg[k]
                Z = prm[k, 1] * rf + prm[k, 2] / dt
                im = (d * v - emf[k] + (prm[k, 2] / dt) * pst[k, 0]) / Z if d > 0.0 else 0.0
                vm = d * v
            if pst[k, 5] > 0.5:
                im = 0.0
            pst[k, 0] = im
            kt = _kt(k, prm, th, pst)
            e = mot_el[int(prm[k, 7])]
            st[e, 0] = prm[k, 3] * kt * im                            # torque on the measured coordinate
            st[e, 1] = im; st[e, 3] = vm
            kg = prm[k, 3] * kt
            st[e, 4] = kg * kg * cdamp[k] if (pst[k, 5] < 0.5 and im != 0.0) else 0.0   # back-EMF damping, N m s
            heat = prm[k, 1] * rf * im * im
            pout[k, 4] = (flg[k] if tp == 11 else 1.0) if pst[k, 5] < 0.5 else 0.0
            pout[k, 5] = kt * im                                 # torque on the motor shaft
        elif tp == 7 or tp == 8:
            pout[k, 4] = 1.0 if G > 1e-3 else 0.0
        if tp != 9 and tp != 6 and tp != 10 and tp != 11 and tp != 7 and tp != 8:
            pout[k, 4] = 0.0 if pst[k, 5] > 0.5 else 1.0
        pout[k, 7] += heat * dt
        if th[k, 0] > 0.5:                                       # lumped thermal mass: heat in, conduction out
            T = pst[k, 4]
            T += dt * (heat - (T - th[k, 3]) / th[k, 2]) / th[k, 1]
            pst[k, 4] = T
            if T > th[k, 5] and tp != 4 and tp != 5:
                pst[k, 5] = 1.0                                  # overheated: the part fails open
        pout[k, 6] = pst[k, 4] - 273.15
