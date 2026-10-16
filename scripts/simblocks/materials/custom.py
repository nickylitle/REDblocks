"""Custom materials: a folder of your own YAML files, merged into the catalog by key.

Point the desktop app at the folder (Materials button), or set the environment variable SIMBLOCKS_MATERIALS to one
or more folders (separated by ; on Windows, : elsewhere) for the command line and batch runner.

Layout of the folder:

    my_materials/
      shop_steels.yaml           any number of .yaml files, each with a `materials:` mapping
      polymers.yaml
      curves/
        S7_58HRC_mine.csv        optional digitized stress-strain curve, named after the material key
      _template.yaml             files starting with _ are ignored (the app writes the template here)

Values take units ("207 GPa", "7.83 g/cm^3"); plain numbers are SI. A custom key that matches a built-in key
replaces the built-in one while the folder is connected.
"""
import glob
import os

from .base import Isotropic, Lamina

ISO_REQUIRED = ("E", "nu", "rho", "sy", "su")
ISO_OPTIONAL = ("elong", "k", "cp", "alpha", "resistivity", "cowper_D", "cowper_q")
LAM_REQUIRED = ("E1", "E2", "G12", "nu12", "rho", "Xt", "Xc", "Yt", "Yc", "S")
LAM_OPTIONAL = ("t_ply", "alpha1", "alpha2", "Vf")
TEXT_KEYS = ("type", "name", "family", "hardness", "source", "note", "aliases", "jc")

CURVES = {}          # key -> csv path for custom curves
_LOADED = {}         # key -> built-in it replaced (or None)
_ALIASES = {}        # alias -> previous DEFAULT_TEMPER value (or None)
FOLDERS = []


def _q(v, where):
    from ..compiler.units import parse_quantity, UnitError
    try:
        return float(parse_quantity(v))
    except (UnitError, TypeError, ValueError) as exc:
        raise ValueError(f"{where}: {exc}")


def _build(key, m, where):
    if not isinstance(m, dict):
        raise ValueError(f"{where}: a material is a mapping of properties")
    kind = str(m.get("type", "isotropic")).lower()
    unknown = set(m) - set(TEXT_KEYS) - set(ISO_REQUIRED + ISO_OPTIONAL if kind != "lamina" else LAM_REQUIRED + LAM_OPTIONAL)
    if unknown:
        raise ValueError(f"{where}: unknown propert{'y' if len(unknown) == 1 else 'ies'} {', '.join(sorted(unknown))}")
    note = " | ".join(x for x in (str(m.get("source", "")), str(m.get("note", ""))) if x and x != "None")
    name = str(m.get("name") or key)
    if kind == "lamina":
        miss = [k for k in LAM_REQUIRED if k not in m]
        if miss:
            raise ValueError(f"{where}: lamina needs {', '.join(miss)}")
        kw = {k: _q(m[k], f"{where}.{k}") for k in LAM_REQUIRED + LAM_OPTIONAL if k in m}
        return Lamina(name=name, note=note, **kw)
    if kind != "isotropic":
        raise ValueError(f"{where}: type must be isotropic or lamina")
    miss = [k for k in ISO_REQUIRED if k not in m]
    if miss:
        raise ValueError(f"{where}: isotropic material needs {', '.join(miss)}")
    kw = {k: _q(m[k], f"{where}.{k}") for k in ISO_REQUIRED + ISO_OPTIONAL if k in m}
    if "jc" in m:
        jc = m["jc"]
        if not isinstance(jc, (list, tuple)) or len(jc) != 7:
            raise ValueError(f"{where}.jc: Johnson-Cook is [A, B, n, C, m, T_melt, T_ref]")
        kw["jc"] = tuple(_q(x, f"{where}.jc") for x in jc)
    return Isotropic(name=name, family=str(m.get("family", "custom")), hardness=str(m.get("hardness", "") or ""),
                     note=note, **kw)


def unload():
    from . import _ALL
    from .curves import DEFAULT_TEMPER
    for k, old in _LOADED.items():
        _ALL.pop(k, None)
        if old is not None:
            _ALL[k] = old
    for a, old in _ALIASES.items():
        DEFAULT_TEMPER.pop(a, None)
        if old is not None:
            DEFAULT_TEMPER[a] = old
    _LOADED.clear(); _ALIASES.clear(); CURVES.clear(); FOLDERS.clear()


def load_folders(folders):
    """Replace any previously loaded custom materials with the ones in these folders. Returns a summary dict:
    added (keys), replaced (built-in keys overridden), curves (keys with a CSV), errors (file: message), files."""
    import yaml
    from . import _ALL
    from .curves import DEFAULT_TEMPER
    unload()
    out = dict(added=[], replaced=[], curves=[], errors=[], files=[])
    for folder in [f for f in folders if f]:
        folder = os.path.abspath(os.path.expanduser(folder))
        if not os.path.isdir(folder):
            out["errors"].append(f"{folder}: folder not found"); continue
        FOLDERS.append(folder)
        for path in sorted(glob.glob(os.path.join(folder, "*.yaml")) + glob.glob(os.path.join(folder, "*.yml"))):
            base = os.path.basename(path)
            if base.startswith("_"):
                continue
            out["files"].append(path)
            try:
                with open(path, encoding="utf-8") as fh:
                    doc = yaml.safe_load(fh) or {}
            except Exception as exc:
                out["errors"].append(f"{base}: {exc}"); continue
            mats = doc.get("materials") if isinstance(doc, dict) else None
            if not isinstance(mats, dict):
                out["errors"].append(f"{base}: needs a top-level 'materials:' mapping"); continue
            for key, m in mats.items():
                key = str(key)
                try:
                    mat = _build(key, m, f"{base}: {key}")
                except ValueError as exc:
                    out["errors"].append(str(exc)); continue
                if key not in _LOADED:
                    _LOADED[key] = _ALL.get(key)
                    if _ALL.get(key) is not None:
                        out["replaced"].append(key)
                _ALL[key] = mat; out["added"].append(key)
                for a in (m.get("aliases") or []):
                    a = str(a).lower()
                    if a not in _ALIASES:
                        _ALIASES[a] = DEFAULT_TEMPER.get(a)
                    DEFAULT_TEMPER[a] = key
        for csv in sorted(glob.glob(os.path.join(folder, "curves", "*.csv"))):
            k = os.path.splitext(os.path.basename(csv))[0]
            if k in _ALL:
                CURVES[k] = csv; out["curves"].append(k)
            else:
                out["errors"].append(f"curves/{os.path.basename(csv)}: no material with key '{k}'")
    return out


def load_env():
    v = os.environ.get("SIMBLOCKS_MATERIALS", "")
    if v.strip():
        return load_folders(v.split(os.pathsep))
    return None


TEMPLATE = '''# simblocks custom materials
# ---------------------------------------------------------------------------------------------------------------
# Copy this file to a new name WITHOUT the leading underscore (files starting with _ are ignored), then edit.
# Every .yaml file in this folder is read. Each needs a top-level `materials:` mapping of key -> properties.
#
#   key        how you refer to it in a model (material: S7_58HRC_mine). Letters, digits, _ and - only.
#              A key that matches a built-in material replaces it while this folder is connected.
#   quoting    put text containing a colon in quotes: source: "cert: lot 12"
#   units      write them ("207 GPa", "1650 MPa", "7.83 g/cm^3", "16 %" is NOT allowed: elong is a fraction).
#              Plain numbers are SI (Pa, kg/m^3, W/m/K, J/kg/K, 1/K).
#   curve      optional: put a digitized stress-strain CSV in curves/<key>.csv (format below). Without one the
#              solver builds a Ramberg-Osgood curve through E, sy, su and elong.
#
# Isotropic (metals, plastics):  required  E, nu, rho, sy (0.2% yield), su (ultimate)
#                                optional  elong, k, cp, alpha, resistivity, cowper_D, cowper_q,
#                                          jc: [A, B, n, C, m, T_melt, T_ref]  (Johnson-Cook)
# Lamina (one composite ply):    required  E1, E2, G12, nu12, rho, Xt, Xc, Yt, Yc, S
#                                optional  t_ply, alpha1, alpha2, Vf
# Text for any material:         name, family, hardness, source, note, aliases: [other names that resolve to it]
#
# curves/<key>.csv format (lines starting with # are notes):
#   # material: S7_58HRC_mine
#   # source: my tensile test, 2026-10
#   # measure: engineering        (or: true, plastic)
#   strain,stress_MPa
#   0,0
#   0.0075,1550
#   ...
# ---------------------------------------------------------------------------------------------------------------
simblocks_materials: 1
materials:
  S7_58HRC_mine:
    type: isotropic
    name: S7 tool steel, 58 HRC (my heat treater)
    family: steel
    E: 207 GPa
    nu: 0.29
    rho: 7.83 g/cm^3
    sy: 1650 MPa
    su: 1950 MPa
    elong: 0.07
    hardness: HRC 58
    aliases: [my s7, s7 58]
    source: "supplier cert, lot 1234 (EXAMPLE VALUES, replace with yours)"

  CF_plate_mine:
    type: lamina
    name: "3K twill carbon prepreg ply (EXAMPLE VALUES)"
    E1: 60 GPa
    E2: 60 GPa
    G12: 4.5 GPa
    nu12: 0.05
    rho: 1.55 g/cm^3
    Xt: 600 MPa
    Xc: 550 MPa
    Yt: 600 MPa
    Yc: 550 MPa
    S: 90 MPa
    t_ply: 0.22 mm
    source: "datasheet (EXAMPLE VALUES, replace with yours)"
'''

CURVE_TEMPLATE = '''# material: S7_58HRC_mine
# source: EXAMPLE curve, replace with your digitized test data and rename to <your key>.csv
# measure: engineering
strain,stress_MPa
0,0
0.00797,1650
0.012,1760
0.02,1850
0.035,1920
0.05,1950
0.07,1900
'''


def write_template(folder):
    """Create the folder layout with _template.yaml and curves/_EXAMPLE_S7_58HRC_mine.csv. Never overwrites."""
    folder = os.path.abspath(os.path.expanduser(folder))
    os.makedirs(os.path.join(folder, "curves"), exist_ok=True)
    written = []
    for rel, text in (("_template.yaml", TEMPLATE), (os.path.join("curves", "_EXAMPLE_S7_58HRC_mine.csv.txt"), CURVE_TEMPLATE)):
        p = os.path.join(folder, rel)
        if not os.path.exists(p):
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(text)
            written.append(p)
    return written
