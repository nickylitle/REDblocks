"""
Block registry.

Every building block is registered with a fixed taxonomy so the catalog can grow
to hundreds of entries without becoming a junk drawer:

    domain    : top-level physics area      (structures, contact, constitutive,
                friction, actuators, rigid_body, geometry, aero, thermal,
                integrators, diagnostics, materials)
    category  : how the block is applied    (beams, plates, rotating, hertz, ...)
    geometry  : the geometric configuration (cantilever, circular_clamped, ...)
    load      : the loading / driving case  (end_point, uniform, spin, ...)
    kind      : "builder"  -> computes parameters once from material + geometry
                "kernel"   -> scalar, numba-compatible function called every step
                "analytic" -> closed-form answer used for checks and sizing
    source    : citation (Roark 7th ed. table/case, textbook, or derivation)

Builders run in plain Python before the loop. Kernels run inside the loop and
take only floats and ints so they can be njit-compiled.
"""
from dataclasses import dataclass, field

DOMAINS = ("materials", "structures", "contact", "constitutive", "friction",
           "actuators", "rigid_body", "geometry", "aero", "thermal",
           "integrators", "diagnostics", "custom")
KINDS = ("builder", "kernel", "analytic")


@dataclass
class BlockInfo:
    name: str
    func: object
    domain: str
    category: str
    geometry: str
    load: str
    kind: str
    source: str
    summary: str
    tags: tuple = field(default_factory=tuple)

    @property
    def path(self):
        return f"{self.domain}/{self.category}/{self.geometry}/{self.load}"


REGISTRY = {}

# Blocks that are not model parts in the editor: plumbing (they run inside the solver's compiled loops because an
# element type already does their job, produce figures and reports, or resolve text), specific cases now reached
# through a generic block (beam_case, plate_case, plate_collapse, mass_properties, spin_stress, strength_ratio,
# hardening_stress, hyperelastic_stress, thermal_resistance, nusselt, spring_combination), duplicates, one-line
# expressions, and robot- or project-specific helpers. They stay importable for hand-written code and callable by
# name, but the editor's library, Help and BLOCK_LIBRARY.html do not list them; CATALOG.md lists them separately.
INTERNAL = {
    # materials plumbing
    "resolve_material", "read_curve_csv", "render_laminate_card", "lamina_Q", "lamina_Qbar", "parse_layup",
    "eng_to_true", "true_to_plastic", "extend_hollomon", "curve_from_fit",
    # in-loop state updates (element types do these)
    "interp_clamped", "tabular_return", "linear_hardening_return", "lag_return", "dashpot_force", "damage_increment",
    "penalty_linear", "penalty_kelvin_voigt", "hunt_crossley", "lugre_step", "relaxation_lag",
    "motor_current", "esc_supply", "lipo_ocv_k", "ecm_constant_power", "node_dTdt", "link_heat_flow",
    # rigid-body and geometry plumbing (the 2d / 3d engines do these)
    "world_to_body", "body_to_world", "point_velocity", "force_moment_about_origin", "planar_accel", "instant_center",
    "quat_to_matrix", "quat_step", "gyrostat_accel", "gyrostat_energy", "point_box_penetration", "obb_gap",
    "point_line_signed_distance", "point_segment_closest", "tip_on_circle", "tip_face_penetration",
    "circle_plane_overlap",
    # geometry studies and outline builders: bodies carry their own shapes now (not library blocks)
    "first_contact_delta", "first_contact_stats", "sample_first_contact", "outline_from_image", "outline_from_dxf",
    "outline_rounded_bar",
    # integrators and the verification / report tooling
    "euler_step", "semi_implicit_step", "velocity_verlet_step", "rk4_step", "run_fixed", "adaptive_dt", "stable_step",
    "glyph_sheet", "results_report", "check_figure", "qa_summary", "EnergyLedger", "dt_convergence", "richardson",
    "verify_all", "plot_report", "assess_failure_modes", "candidate_failure_modes", "split_pulses", "write_load_table",
    "pulse_metrics", "half_sine_train", "fe_load_table",
    # builder-to-builder plumbing (beam_plastic_spring and plate_root_hinge already return spring tables)
    "curve_to_spring_table", "hinge_to_spring_table",
    # merged into generic blocks (each specific case is now an option of the generic one)
    # beam_case
    "cantilever_end_load", "cantilever_point_at", "cantilever_tip_deflection_point_at", "cantilever_uniform",
    "cantilever_end_moment", "simply_supported_center", "simply_supported_point_at", "simply_supported_uniform",
    "fixed_fixed_center", "fixed_fixed_uniform", "fixed_guided_end_load", "propped_center", "propped_uniform",
    # spring_combination
    "springs_in_series", "springs_in_parallel",
    # plate_case
    "circ_clamped_center_point", "circ_ss_center_point", "circ_clamped_pressure", "circ_ss_pressure",
    "circ_clamped_pressure_large", "rect_ss_center_point", "rect_ss_point_at", "rect_ss_pressure",
    "rect_clamped_pressure", "rect_clamped_center_point", "annular_outer_fixed_inner_load",
    # plate_collapse
    "collapse_point_interior", "collapse_point_free_edge", "collapse_circular_pressure", "collapse_rect_pressure",
    "collapse_three_edge_strip",
    # mass_properties
    "I_disk", "I_annulus", "I_thin_ring", "I_rod_center", "I_rod_end", "I_rect_plate", "I_cylinder_transverse",
    "I_sphere", "I_box_tensor",
    # spin_stress
    "solid_disk_spin_stress", "annular_disk_spin_stress", "annular_disk_spin_field", "thin_ring_spin_stress",
    "bar_spin_root_stress", "burst_rpm_solid_disk",
    # strength_ratio, hardening_stress, hyperelastic_stress
    "strength_ratio_max_stress", "strength_ratio_tsai_hill", "strength_ratio_tsai_wu",
    "hollomon_stress", "swift_stress", "voce_stress", "ludwik_stress", "johnson_cook_stress",
    "neo_hookean_uniaxial", "mooney_rivlin_uniaxial",
    # thermal_resistance, nusselt
    "R_plane_wall", "R_cylinder", "R_sphere", "R_convection",
    "nu_flat_plate", "nu_cylinder_crossflow", "nu_vertical_plate_natural", "nu_horizontal_cylinder_natural",
    "nu_pipe_gnielinski",
    # duplicates (the kept one covers it: gearbox_output_torque_dir with eta_b = eta, critical_dt_spring with
    # k = k_theta and m = I)
    "gearbox_output_torque", "critical_dt_rotational",
    # robot- or project-specific (case-study code imports them directly)
    "three_point_normal_loads", "axle_load_transfer", "spinner_strike", "bite_depth", "first_contact_delta",
    "first_contact_stats", "sample_first_contact", "outline_rounded_bar", "solve_remainder_centroid",
    "tab_shear_out_force",
    # one-line expressions (write them as '= ...' in the system file)
    "tip_speed", "spin_energy", "gyro_torque", "copper_loss", "dynamic_pressure", "dt_for_contact", "recommend_dt",
    # per-step friction, tyre, drag and supply kernels: an element type does the job (friction, contact_friction,
    # hertz, drag, drive) or nothing drives them; hand-written loops can still call them
    "coulomb_tanh", "stribeck_mu", "stribeck_mu_delta", "simscape_friction", "makkar_friction", "skid_force_2d",
    "coulomb_regularized_3d", "rolling_torque", "rolling_resistance_torque", "bearing_drag_torque",
    "tire_mu", "tire_force_2d", "brush_mu", "brush_force_2d", "brush_mu_s_for_peak", "magic_formula",
    "magic_formula_fit", "hertz_sphere_force", "drag_force", "bus_voltage",
}


def is_public(name):
    return name not in INTERNAL


def block(domain, category, geometry="general", load="general", kind="analytic",
          source="derivation", tags=()):
    """Decorator that registers a block. Place it ABOVE @kernel."""
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}")
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}")

    def wrap(f):
        py = getattr(f, "py_func", f)
        name = py.__name__
        if name in REGISTRY:
            raise ValueError(f"duplicate block name {name!r}")
        doc = (py.__doc__ or "").strip().splitlines()
        REGISTRY[name] = BlockInfo(name, f, domain, category, geometry, load, kind,
                                   source, doc[0] if doc else "", tuple(tags))
        return f
    return wrap


def find(domain=None, category=None, geometry=None, load=None, kind=None, text=None):
    """Query the catalog. Any argument left None matches everything."""
    out = []
    for b in REGISTRY.values():
        if domain and b.domain != domain: continue
        if category and b.category != category: continue
        if geometry and geometry not in b.geometry: continue
        if load and load not in b.load: continue
        if kind and b.kind != kind: continue
        if text and text.lower() not in (b.name + b.summary + " ".join(b.tags)).lower():
            continue
        out.append(b)
    return sorted(out, key=lambda b: (b.domain, b.category, b.geometry, b.name))


def catalog_markdown():
    pub = [b for b in find() if is_public(b.name)]
    lines = ["# Block catalog", "",
             f"{len(pub)} model blocks (the editor's library) plus {len(REGISTRY) - len(pub)} internal ones at the end. "
             "Path = domain / category / geometry / load. "
             "Kind: builder (pre-loop), kernel (in-loop, njit-safe), analytic (closed form).", ""]
    cur = None
    for b in pub:
        key = (b.domain, b.category)
        if key != cur:
            lines += ["", f"## {b.domain} / {b.category}", "",
                      "| block | geometry | load | kind | source | summary |",
                      "|---|---|---|---|---|---|"]
            cur = key
        lines.append(f"| `{b.name}` | {b.geometry} | {b.load} | {b.kind} | {b.source} | {b.summary} |")
    lines += ["", "## Internal (code only)", "",
              "Solver, report and text plumbing: callable from hand-written sims, not listed in the editor.", "",
              "| block | path | summary |", "|---|---|---|"]
    for b in find():
        if not is_public(b.name):
            lines.append(f"| `{b.name}` | {b.path} | {b.summary} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------- custom block folders
CUSTOM = {}          # block name -> file it came from
CUSTOM_FOLDERS = []

CUSTOM_TEMPLATE = '''"""My custom blocks.

Every .py file in this folder is loaded when simblocks starts (files starting with _ are skipped, so copy this
template to a new name first). Each function decorated with @block becomes a catalog block: it appears in the
Add menu and in Help, and a system file calls it from derived: exactly like a built-in block:

    derived:
      arm: {block: my_cantilever_tip, E: 200 GPa, I: 1.2e-9 m^4, L: 80 mm}
    elements:
      blade: {type: spring, measure: tip, k: $arm.k}

Rules
  * Inputs arrive in SI (the system file converts units before calling you). Return SI.
  * Return a number, or a dict of numbers (use them as $name.key).
  * kind="builder" or "analytic" for blocks used in derived:. kind="kernel" blocks run inside the time loop and
    must be numba-safe (floats and ints only); the 1D solver only calls kernels through its element types.
  * domain is any of the built-in domains or "custom". Names must be unique across the whole catalog.
  * Put the citation in source=: Help shows it next to the block.
"""
import math

from simblocks.registry import block


@block("custom", "beams", "cantilever", "end_point", kind="builder",
       source="Roark 7th ed. Table 8.1 case 1a (replace with your source)", tags=("stiffness", "tip load", "example"))
def my_cantilever_tip(E, I, L, c=None, sy=None):
    """Cantilever tip stiffness, plus the tip load at first yield when c and sy are given.
    E modulus (Pa), I second moment of area (m^4), L length (m), c distance to the longest fibre (m),
    sy yield stress (Pa). Returns k (N/m) and F_yield (N, NaN without c and sy)."""
    k = 3.0 * E * I / L ** 3
    F_y = sy * I / (c * L) if (c and sy) else math.nan
    return dict(k=k, F_yield=F_y)
'''


def unload_block_folders():
    import sys
    for name in list(CUSTOM):
        REGISTRY.pop(name, None)
    for m in [m for m in sys.modules if m.startswith("simblocks_user.")]:
        del sys.modules[m]
    CUSTOM.clear(); CUSTOM_FOLDERS.clear()


def load_block_folders(folders):
    """Import every .py file in the folders (skipping _*.py). Returns dict(added, errors, files)."""
    import glob
    import importlib.util
    import os
    import sys
    import traceback
    unload_block_folders()
    out = dict(added=[], errors=[], files=[])
    for folder in [f for f in folders if f]:
        folder = os.path.abspath(os.path.expanduser(folder))
        if not os.path.isdir(folder):
            out["errors"].append(f"{folder}: folder not found"); continue
        CUSTOM_FOLDERS.append(folder)
        for path in sorted(glob.glob(os.path.join(folder, "*.py"))):
            base = os.path.basename(path)
            if base.startswith("_"):
                continue
            out["files"].append(path)
            before = set(REGISTRY)
            modname = "simblocks_user." + "".join(ch if ch.isalnum() else "_" for ch in base[:-3])
            try:
                spec = importlib.util.spec_from_file_location(modname, path)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[modname] = mod
                spec.loader.exec_module(mod)
            except Exception as exc:
                for n in set(REGISTRY) - before:
                    REGISTRY.pop(n, None)
                sys.modules.pop(modname, None)
                tb = traceback.extract_tb(exc.__traceback__)
                line = next((f.lineno for f in reversed(tb) if f.filename == path), None)
                out["errors"].append(f"{base}{':' + str(line) if line else ''}: {type(exc).__name__}: {exc}")
                continue
            for n in sorted(set(REGISTRY) - before):
                CUSTOM[n] = path; out["added"].append(n)
    return out


def load_env_blocks():
    import os
    v = os.environ.get("SIMBLOCKS_BLOCKS", "")
    return load_block_folders(v.split(os.pathsep)) if v.strip() else None


def write_block_template(folder):
    import os
    folder = os.path.abspath(os.path.expanduser(folder))
    os.makedirs(folder, exist_ok=True)
    p = os.path.join(folder, "_template_blocks.py")
    if os.path.exists(p):
        return []
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(CUSTOM_TEMPLATE)
    return [p]
