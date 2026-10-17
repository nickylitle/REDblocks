"""CSV exports: time histories written as plain CSV files, ready to paste into a spreadsheet or another tool's table of time vs value.

A system file asks for them with an `exports:` list. Nothing is written unless an export is defined.

    exports:
      - name: tip_loads
        columns:
          - {element: tooth, quantity: force, unit: N, label: Fz}
          - {body: rotor, quantity: velocity, unit: rpm}
        folder: out/csv                  # relative to the model file; absolute paths and ~ work too
        naming: "{model}_{name}"         # tokens: {model} {name} {date} {time} {run}
        time_unit: ms
        points: 1000                     # rows (or 'all' for every time step)
        header: units                    # units: "Time [ms],Fz [N]"   plain: "t_ms,Fz_N"   none: numbers only
        when: run                        # run: the nominal run   study: also every study run ({run} in the name)
"""
import datetime as _dt
import os
import re

import numpy as np

from .units import unit_factor

NAMING_SCHEMES = {
    "{model}_{name}": "model_export.csv (overwritten each run)",
    "{model}_{name}_{date}_{time}": "model_export_2026-10-16_103400.csv (new file each run)",
    "{name}": "export.csv",
    "{name}_{date}_{time}": "export_2026-10-16_103400.csv",
    "{model}_{name}_{run}": "model_export_nominal.csv, model_export_run0001.csv ... (for studies)",
}
SI_UNIT = {"force": "N", "measure": "m", "position": "m", "velocity": "m/s", "plastic_set": "m", "broken": "",
           "current": "A", "battery_current": "A", "bus_voltage": "V", "soc": ""}
ROT_UNIT = {"position": "rad", "velocity": "rad/s"}
SI_UNIT.update(x="m", y="m", z="m", angle="rad", vx="m/s", vy="m/s", vz="m/s", omega="rad/s", speed="m/s",
               wx="rad/s", wy="rad/s", wz="rad/s")


def wanted(doc, when="run"):
    """The exports that fire for a nominal run (when='run') or for each study run (when='study')."""
    ex = doc.get("exports") or []
    if when == "run":
        return list(ex)
    return [x for x in ex if str(x.get("when", "run")) == "study"]


def log_points(doc, n_steps):
    """History rows the run should log so every export gets the rows it asks for."""
    pts = 0
    for x in doc.get("exports") or []:
        p = x.get("points", 1000)
        pts = max(pts, n_steps if str(p) == "all" else int(p))
    return max(pts, 2)


def base_dir(doc, model_path=None):
    mp = model_path or doc.get("__file__") or ""
    if mp and os.path.splitext(mp)[1].lower() in (".yaml", ".yml"):
        return os.path.dirname(os.path.abspath(mp)), os.path.splitext(os.path.basename(mp))[0]
    stem = re.sub(r"\W+", "_", str(doc.get("name", "model"))).strip("_") or "model"
    return os.path.join(os.path.expanduser("~"), "simblocks"), stem


def file_name(pattern, model, name, run="nominal", when=None, force_run=False):
    when = when or _dt.datetime.now()
    pat = pattern or "{model}_{name}"
    if force_run and "{run}" not in pat:
        pat += "_{run}"
    out = pat.format(model=model, name=name, date=when.strftime("%Y-%m-%d"), time=when.strftime("%H%M%S"), run=run)
    out = re.sub(r'[<>:"/\\\\|?*]+', "_", out).strip() or name
    return out if out.lower().endswith(".csv") else out + ".csv"


def column(doc, H, c):
    """(series in the column's unit, header label, unit text) for one column spec."""
    if "control" in c:
        key = f"control.{c['control']}"; si = ""; default_label = c["control"]
    elif "part" in c:
        qn = c.get("quantity", "current"); key = f"{c['part']}.{qn}"
        si = {"current": "A", "voltage": "V", "power": "W", "energy": "J", "torque": "N m", "temperature": "",
              "heat": "J"}.get(qn, ""); default_label = f"{c['part']} {qn}"
    elif "element" in c:
        qn = c.get("quantity", "force"); key = f"{c['element']}.{qn}"; si = SI_UNIT.get(qn, "")
        default_label = f"{c['element']} {qn}"
    else:
        qn = c.get("quantity", "position"); key = f"{c['body']}.{qn}"
        dof = (doc["bodies"][c["body"]] or {}).get("dof")
        rot = dof == "rotation" or (doc.get("dim") == "2d" and dof in (["rz"], "rotation", "spin"))
        si = (ROT_UNIT if rot else SI_UNIT).get(qn, ""); default_label = f"{c['body']} {qn}"
    unit = c.get("unit") or si
    y = np.asarray(H[key], dtype=float) / (unit_factor(unit) if unit else 1.0)
    return y, str(c.get("label") or default_label), unit


def write(doc, H, model_path=None, run="nominal", when="run", stamp=None):
    """Write every export that fires for this run; returns the paths written."""
    exps = wanted(doc, when)
    if not exps:
        return []
    root, model = base_dir(doc, model_path)
    t = np.asarray(H["t"], dtype=float)
    paths = []
    for x in exps:
        folder = os.path.expanduser(str(x.get("folder") or "out/csv"))
        folder = folder if os.path.isabs(folder) else os.path.join(root, folder)
        os.makedirs(folder, exist_ok=True)
        tu = x.get("time_unit") or "s"
        cols = [column(doc, H, c) for c in x["columns"]]
        rows = len(t); pts = x.get("points", 1000)
        idx = np.arange(rows) if str(pts) == "all" or rows <= int(pts) else \
            np.unique(np.round(np.linspace(0, rows - 1, int(pts))).astype(int))
        data = np.column_stack([t[idx] / unit_factor(tu)] + [y[idx] for y, _, _ in cols])
        hdr = str(x.get("header", "units"))
        if hdr == "units":
            head = ",".join([f"Time [{tu}]"] + [f"{lab} [{u}]" if u else lab for _, lab, u in cols])
        elif hdr == "plain":
            safe = lambda s: re.sub(r"\W+", "_", s).strip("_")
            head = ",".join([f"t_{safe(tu)}"] + [safe(lab) + (f"_{safe(u)}" if u else "") for _, lab, u in cols])
        else:
            head = None
        path = os.path.join(folder, file_name(x.get("naming"), model, x["name"], run, stamp,
                                              force_run=(when == "study")))
        with open(path, "w", encoding="utf-8", newline="") as fh:
            if head:
                fh.write(head + "\n")
            np.savetxt(fh, data, delimiter=",", fmt="%.9g")
        paths.append(path)
    return paths
