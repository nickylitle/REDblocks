"""Machine elements on a linear measure: a spur gear mesh and a preloaded bolted joint. Both are closed-form laws
(Shigley, ISO 6336) evaluated every step; they work in 1d, 2d and 3d like any measure element.

gear: two shafts meshing (or a gear on a rack)

    mesh: {type: gear, measure: pinion + wheel, module: 1 mm, teeth: [12, 36], face_width: 8 mm,
           backlash: 0.05 mm, material: 4140_normalized}

  measure        the two shafts, gear 1 first (in 2d: pinion.rz + wheel.rz). The signs written do not matter: an
                 external mesh turns the shafts opposite ways, internal: true is a ring gear (same way). A rack is
                 a translation body with teeth: [z, rack]; with the pinion turning +, the rack moves - (a rack under
                 a pinion that turns counter-clockwise runs left). rack_reversed: true for a rack on the other side
                 (it moves + as the pinion turns +).
  module, teeth  pitch radius r = module * z / 2 for each gear. The measure is then travel along the pitch line,
                 r1 a1 + r2 a2 (m), and the force is the tangential tooth load Ft (N).
  face_width     tooth width b
  pressure_angle 20 deg by default
  backlash       total free play along the pitch line (default 0): the teeth carry nothing inside it
  k_mesh         mesh stiffness along the pitch line. Default ISO 6336-1: c_gamma = 20 N/(mm um) per mm of face
                 width for steel spur gears (scaled by E / 206 GPa), along the line of action, times cos^2 of the
                 pressure angle for the pitch line.
  c, zeta        mesh damping: c directly, or zeta, a fraction of critical for the two inertias (default 0.05)
  material or E, nu   for the stiffness and the contact stress (default steel: 206 GPa, 0.3)
  outputs        force (Ft), measure, bending_stress (Lewis, Ft / (b m Y), the weaker gear's Y), contact_stress
                 (Hertz at the pitch point: Cp sqrt(Ft / (b d_p I)), the AGMA form with no extra factors: the run
                 itself supplies the dynamic load)

bolt: a preloaded joint, bolt and clamped members in parallel (Shigley ch. 8)

    hold: {type: bolt, measure: lid - frame, size: M5, grip: 6 mm, grade: "12.9", torque: 6 N m}

  measure        the joint opening from the clamped state (m): positive pulls the parts apart
  size           M2 ... M24 (coarse pitch), #4-40 ... 1/2-13 (UNC / UNF), or d and pitch
  grip           clamped length L
  threaded       length of thread inside the grip (default: all of it)
  preload        Fi directly, or torque with nut_factor K (default 0.2): Fi = T / (K d)
  grade          4.6, 8.8, 10.9, 12.9, A2-70, A4-80, grade 2, grade 5, grade 8: yield Sy, ultimate Su and
                 elongation at break (ISO 898-1, ISO 3506-1, SAE J429 minimums); or give Sy, Su, elongation
  Past yield the bolt hardens linearly to Su at a permanent stretch of elongation x the thread length in the grip
  (at least d / 2), then breaks: it carries nothing after that. Real bolts neck in the first engaged thread, so treat
  the stretch at break as an estimate.
  E, E_members   bolt and member moduli (default steel 207 GPa; members default to the bolt's)
  k_bolt, k_members   override the stiffnesses (default: k_b = Ad At E / (Ad lt + At ld); k_m the 30 deg frustum)
  c, zeta        joint damping (default zeta 0.02)
  Law: bolt force Fb = Fi + k_b s (yields at Sy At and keeps the set: preload is lost), member clamp
  Fm = max(Fi - k_m s, 0). The joint resists with Fb - Fm. It separates when Fm reaches 0, at an external load
  Fi (1 + k_b / k_m).
  outputs        force (the joint's resistance), measure, bolt_force, clamp_force, bolt_stress (Fb / At),
                 plastic_set (permanent bolt stretch), broken (1 once it snaps)"""
import math
import re

from .loader import SpecError, line_of

GEAR_KEYS = {"module", "teeth", "face_width", "pressure_angle", "backlash", "k_mesh", "c", "zeta", "material", "E",
             "nu", "internal", "rack_reversed"}
BOLT_KEYS = {"size", "d", "pitch", "grip", "threaded", "preload", "torque", "nut_factor", "grade", "Sy", "E",
             "E_members", "k_bolt", "k_members", "c", "zeta", "Su", "elongation"}

# Lewis form factor Y, 20 deg full-depth teeth (Shigley table 14-2)
LEWIS = [(12, .245), (13, .261), (14, .277), (15, .290), (16, .296), (17, .303), (18, .309), (19, .314), (20, .322),
         (21, .328), (22, .331), (24, .337), (26, .346), (28, .353), (30, .359), (34, .371), (38, .384), (43, .397),
         (50, .409), (60, .422), (75, .435), (100, .447), (150, .460), (300, .472), (400, .480)]
Y_RACK = 0.485
METRIC_PITCH = {2: .4, 2.5: .45, 3: .5, 4: .7, 5: .8, 6: 1.0, 8: 1.25, 10: 1.5, 12: 1.75, 14: 2.0, 16: 2.0, 20: 2.5,
                24: 3.0}
UN_NUMBER = {"#2": .086, "#4": .112, "#5": .125, "#6": .138, "#8": .164, "#10": .190, "#12": .216}
_KSI = 6894.757e3
# property class: (yield Sy, ultimate Su, elongation at break). ISO 898-1 / ISO 3506-1 / SAE J429 minimums
GRADES = {"4.6": (240e6, 400e6, .22), "4.8": (340e6, 420e6, .14), "5.8": (420e6, 520e6, .10),
          "8.8": (640e6, 800e6, .12), "9.8": (720e6, 900e6, .10), "10.9": (940e6, 1040e6, .09),
          "12.9": (1100e6, 1220e6, .08), "A2-70": (450e6, 700e6, .40), "A4-70": (450e6, 700e6, .40),
          "A4-80": (600e6, 800e6, .30), "grade 2": (57 * _KSI, 74 * _KSI, .18), "grade 5": (92 * _KSI, 120 * _KSI, .14),
          "grade 8": (130 * _KSI, 150 * _KSI, .12)}


def lewis_Y(z):
    if z == "rack":
        return Y_RACK
    z = float(z)
    if z < 12:
        return LEWIS[0][1] * z / 12.0              # below the table: few teeth are undercut, weaker still
    for (z0, y0), (z1, y1) in zip(LEWIS, LEWIS[1:]):
        if z <= z1:
            return y0 + (y1 - y0) * (z - z0) / (z1 - z0)
    return LEWIS[-1][1] + (Y_RACK - LEWIS[-1][1]) * min(1.0, (z - 400) / 600)


def _terms(text):
    """Top-level + / - terms of a measure, signs dropped: 'a - 2 * b' -> ['a', '2 * b']."""
    out, depth, cur = [], 0, ""
    s = str(text).strip()
    for i, ch in enumerate(s):
        depth += ch == "("; depth -= ch == ")"
        prev = s[:i].rstrip()
        if depth == 0 and ch in "+-" and prev and not re.search(r"\d[eE]$", prev) and not prev.endswith(("*", "/")):
            if cur.strip():
                out.append(cur.strip())
            cur = ""; continue
        if depth == 0 and ch in "+-" and not prev:
            continue
        cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _material(d, ctx, where, num, resolve):
    E = nu = None
    if "material" in d:
        from ..materials import get
        key = str(resolve(d["material"], ctx, where + ".material"))
        try:
            m = get(key)
        except KeyError:
            raise SpecError(f"material '{key}' is not in the catalog", where)
        E, nu = float(m.E), float(getattr(m, "nu", 0.3))
    if "E" in d:
        E = num(resolve(d["E"], ctx, where + ".E"), where + ".E")
    if "nu" in d:
        nu = num(resolve(d["nu"], ctx, where + ".nu"), where + ".nu")
    return E or 206e9, nu if nu is not None else 0.3


def gear(en, e, ctx, doc):
    """A gear element -> (internal element dict, info)."""
    from .model import resolve, num
    f = doc.get("__file__", ""); where = f"elements.{en}"
    g = lambda k, dflt=None: num(resolve(e[k], ctx, f"{where}.{k}"), f"{where}.{k}") if k in e else dflt
    bad = set(e) - GEAR_KEYS - {"type", "measure", "label", "source", "note", "knockout", "mass"}
    if bad:
        raise SpecError(f"'{sorted(bad)[0]}' is not a field of a gear (fields: {', '.join(sorted(GEAR_KEYS))})", where,
                        line_of(e, sorted(bad)[0]), f)
    for k in ("module", "teeth", "face_width"):
        if k not in e:
            raise SpecError(f"a gear needs '{k}'", where, line_of(e, "type"), f)
    m = g("module"); b = g("face_width")
    teeth = e["teeth"]
    if isinstance(teeth, str):                    # typed in a field: "12, 36" or "[12, rack]"
        teeth = [x.strip() for x in teeth.strip().strip("[]").split(",") if x.strip()]
    if not isinstance(teeth, list) or len(teeth) not in (1, 2):
        raise SpecError("teeth is [z1, z2] (gear 1 first, as in the measure), or [z, rack]", where, line_of(e, "teeth"), f)
    zs = [("rack" if str(z).strip().lower() == "rack" else num(resolve(z, ctx, where + ".teeth"), where + ".teeth"))
          for z in teeth]
    if any(z != "rack" and z < 6 for z in zs):
        raise SpecError("a gear needs at least 6 teeth", where, line_of(e, "teeth"), f)
    if not isinstance(e.get("measure"), str):
        raise SpecError("a gear's measure names its two shafts, gear 1 first: 'pinion + wheel' (2d: 'pinion.rz + "
                        "wheel.rz', 3d: 'pinion.spin + wheel.spin')", where, line_of(e, "measure"), f)
    terms = _terms(e["measure"])
    if len(terms) != len(zs):
        raise SpecError(f"the measure names {len(terms)} shaft(s) and teeth lists {len(zs)}: give one tooth count per "
                        "shaft, in the measure's order", where, line_of(e, "measure"), f)
    phi = g("pressure_angle", math.radians(20.0))
    if not 0.17 < phi < 0.62:
        raise SpecError("pressure_angle is between 10 and 35 deg (20 deg is standard)", where, line_of(e, "pressure_angle"), f)
    internal = bool(e.get("internal"))
    r = [m * z / 2.0 if z != "rack" else 1.0 for z in zs]
    sign = [1.0, -1.0 if internal else 1.0]
    if e.get("rack_reversed"):
        if "rack" not in zs:
            raise SpecError("rack_reversed is for a gear on a rack (teeth: [z, rack])", where, line_of(e, "rack_reversed"), f)
        sign = [s_ * (-1.0 if zs[i] == "rack" else 1.0) for i, s_ in enumerate(sign)]
    meas = " + ".join(f"({sign[i] * r[i]!r}) * ({t})" for i, t in enumerate(terms))
    E, nu = _material(e, ctx, where, num, resolve)
    k = g("k_mesh")
    if k is None:
        k = 20e9 * (E / 206e9) * b * math.cos(phi) ** 2       # ISO 6336-1 c_gamma, to the pitch line
    Ys = [lewis_Y(z) for z in zs]
    sb = 1.0 / (b * m * min(Ys))                              # Lewis: sigma = Ft / (b m Y)
    real = [z for z in zs if z != "rack"]
    zp = min(real) if real else 20.0
    dp = m * zp
    if len(zs) == 2 and "rack" not in zs:
        mG = max(zs) / min(zs)
        I = math.cos(phi) * math.sin(phi) / 2.0 * mG / (mG - 1.0 if internal else mG + 1.0) if not (internal and mG <= 1) else 1e-9
    else:
        I = math.cos(phi) * math.sin(phi) / 2.0
    Cp = math.sqrt(1.0 / (math.pi * 2.0 * (1.0 - nu * nu) / E))
    sh = Cp / math.sqrt(b * dp * I)                           # sigma_c = Cp sqrt(Ft / (b d_p I))
    c = g("c"); zeta = g("zeta", 0.05)
    P = [k, c if c is not None else 0.0, 0.5 * g("backlash", 0.0), sb, sh, 0.0, 0.0 if c is not None else zeta]
    out = {kk: v for kk, v in e.items() if kk in ("label", "source", "note", "knockout", "mass")}
    out.update(type="gear", measure=meas, __P__=P)
    return out


def _size(e, ctx, where, num, resolve, f):
    if "size" in e:
        sz = str(e["size"]).strip().replace(" ", "")
        mm = re.fullmatch(r"[Mm](\d+(?:\.\d+)?)(?:x(\d+(?:\.\d+)?))?", sz)
        if mm:
            d = float(mm.group(1))
            p = float(mm.group(2)) if mm.group(2) else METRIC_PITCH.get(d if d != int(d) else int(d))
            if p is None:
                raise SpecError(f"no coarse pitch on file for {sz}: write it as M{mm.group(1)}x<pitch>", where)
            return d * 1e-3, p * 1e-3, False
        un = re.fullmatch(r"(#\d+|\d+/\d+|\d+(?:\.\d+)?)-(\d+)", sz)
        if un:
            a = un.group(1)
            if a.startswith("#"):
                if a not in UN_NUMBER:
                    raise SpecError(f"unknown screw number {a}", where)
                d_in = UN_NUMBER[a]
            elif "/" in a:
                n_, d_ = a.split("/"); d_in = float(n_) / float(d_)
            else:
                d_in = float(a)
            return d_in * 0.0254, 0.0254 / float(un.group(2)), True
        raise SpecError(f"size '{e['size']}' is not M<d> (M5, M6x0.75) or <d>-<tpi> (#10-32, 1/4-20)", where)
    if "d" not in e or "pitch" not in e:
        raise SpecError("a bolt needs size (M5, 1/4-20) or d and pitch", where)
    return (num(resolve(e["d"], ctx, where + ".d"), where + ".d"), num(resolve(e["pitch"], ctx, where + ".pitch"),
            where + ".pitch"), False)


def bolt(en, e, ctx, doc):
    from .model import resolve, num
    f = doc.get("__file__", ""); where = f"elements.{en}"
    g = lambda k, dflt=None: num(resolve(e[k], ctx, f"{where}.{k}"), f"{where}.{k}") if k in e else dflt
    bad = set(e) - BOLT_KEYS - {"type", "measure", "label", "source", "note", "knockout", "mass"}
    if bad:
        raise SpecError(f"'{sorted(bad)[0]}' is not a field of a bolt (fields: {', '.join(sorted(BOLT_KEYS))})", where,
                        line_of(e, sorted(bad)[0]), f)
    d, p, unified = _size(e, ctx, where, num, resolve, f)
    if "grip" not in e:
        raise SpecError("a bolt needs grip (the clamped length)", where, line_of(e, "type"), f)
    L = g("grip"); lt = min(g("threaded", L), L); ld = L - lt
    Ad = math.pi * d * d / 4.0
    At = math.pi / 4.0 * (d - (0.9743 if unified else 0.9382) * p) ** 2      # tensile stress area
    E = g("E", 207e9); Em = g("E_members", E)
    kb = g("k_bolt") or Ad * At * E / (Ad * lt + At * ld)
    km = g("k_members")
    if km is None:
        km = 0.5774 * math.pi * Em * d / (2.0 * math.log(5.0 * (0.5774 * L + 0.5 * d) / (0.5774 * L + 2.5 * d)))
    if ("preload" in e) == ("torque" in e):
        raise SpecError("a bolt needs one of preload (the clamp force) or torque (with nut_factor, default 0.2)",
                        where, line_of(e, "type"), f)
    Fi = g("preload") if "preload" in e else g("torque") / (g("nut_factor", 0.2) * d)
    Sy = Su = el = None
    if "grade" in e:
        gk = str(e["grade"]).strip()
        key = next((k for k in GRADES if k.lower() == gk.lower() or k.lower() == "grade " + gk.lower()), None)
        if key is None:
            raise SpecError(f"grade '{gk}' is not on file ({', '.join(GRADES)}); give Sy (and Su) instead", where,
                            line_of(e, "grade"), f)
        Sy, Su, el = GRADES[key]
    Sy = g("Sy", Sy); Su = g("Su", Su); el = g("elongation", el)
    if Sy is None:
        Sy = math.inf
    if Su is None or not math.isfinite(Sy):
        Su = Sy                                  # no ultimate given: perfectly plastic past yield, never breaks
    if Su < Sy:
        raise SpecError("Su (ultimate) must be at least Sy (yield)", where, line_of(e, "Su"), f)
    # plastic stretch at break: the elongation over the thread in the grip, where a bolt yields first
    xb = (el if el is not None else 0.1) * max(lt, 0.5 * d) if Su > Sy else math.inf
    kh = (Su - Sy) * At / xb if Su > Sy else 0.0
    if Fi <= 0 or Fi >= Sy * At:
        raise SpecError(f"the preload {Fi:.4g} N must be above 0 and below the bolt's yield load {Sy * At:.4g} N",
                        where, line_of(e, "preload" if "preload" in e else "torque"), f)
    c = g("c"); zeta = g("zeta", 0.02)
    P = [kb, km, Fi, Sy * At, At, c if c is not None else 0.0, 0.0 if c is not None else zeta, Su * At, kh, xb]
    out = {kk: v for kk, v in e.items() if kk in ("label", "source", "note", "knockout", "mass")}
    out.update(type="bolt", measure=e["measure"], __P__=P,
               __info__=dict(d=d, pitch=p, At=At, k_bolt=kb, k_members=km, preload=Fi, yield_load=Sy * At,
                             separation_load=Fi * (1 + kb / km), C=kb / (kb + km),
                             ultimate_load=Su * At, set_at_break=xb))
    return out


def desugar(doc, ctx):
    els = doc.get("elements") or {}
    if not any(isinstance(e, dict) and e.get("type") in ("gear", "bolt") for e in els.values()):
        return doc
    import copy
    out = copy.copy(doc); out.lines = getattr(doc, "lines", {})
    out["elements"] = {en: (gear(en, e, ctx, doc) if e.get("type") == "gear" else
                            bolt(en, e, ctx, doc) if e.get("type") == "bolt" else e) if isinstance(e, dict) else e
                       for en, e in els.items()}
    return out
