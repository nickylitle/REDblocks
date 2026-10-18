"""Plastic hinge at the root of a plate strip (a wedge face, a flap, a lip): moment-rotation spring for a rotational
element. Elastic stiffness from the strip as a tip-loaded cantilever; the plastic part from the fibre-integrated
moment-curvature of the material's stress-strain curve over a hinge length."""
import math

import numpy as np

from ...registry import block


@block("structures", "plates", "strip_root", "plastic_hinge", kind="builder",
       source="Roark 7e Table 8.1 case 1a (k = 3EI/L as a root rotation) + fibre-integrated M(kappa) plastic hinge",
       tags=("wedge face", "plow", "bend", "root hinge", "elastoplastic spring", "moment rotation"))
def plate_root_hinge(material, width, t, L, hinge_len=None, clamp_factor=1.0, rate_factor=1.0, k_factor=1.0,
                     plane_strain=False):
    """Moment-rotation spring (k, acc_tab, F_tab) for a plate strip of the given width and thickness t that bends at
    its root. L = free length from root to the loaded edge (sets the elastic stiffness k = 3 E I / L).
    hinge_len defaults to t. clamp_factor scales the plastic moments for extra edge support (1 = a plain cantilever
    strip; about 2.2 for a face also held along two side edges). rate_factor scales the stress-strain curve for
    strain-rate hardening. k_factor scales the elastic stiffness (the side clamps stiffen it too; an FEA unit load
    on the real face gives the factor). For a rotational element on the face's spin coordinate: 'force' is a moment (N m),
    'measure' a rotation (rad). plane_strain: a strip wider than about 5 t bends with its width held straight, so
    it is E / (1 - nu^2) stiff and yields at 2 / sqrt(3) sy (von Mises, plane strain): about 10 % stiffer and 15 %
    stronger than a narrow beam (the default, conservative). Returns dict(k, acc_tab, F_tab, M_y, Mp, source)."""
    from ...materials import get, load_curve
    from ...materials.curves import resolve_material, Curve
    from ..sections.moment_curvature import moment_curvature_rect, hinge_table
    from ...constitutive.plasticity import hinge_to_spring_table
    key = resolve_material(material); mat = get(key); c0 = load_curve(key)
    ps = 2.0 / math.sqrt(3.0) if plane_strain else 1.0
    Ep = mat.E / (1.0 - mat.nu ** 2) if plane_strain else mat.E
    c = Curve(c0.name, c0.eng_strain, c0.eng_stress * rate_factor * ps, Ep,
              c0.source + f" x {rate_factor:g} (rate)" + (" x 2/sqrt(3) (plane strain)" if plane_strain else ""))
    I = width * t ** 3 / 12.0
    k = 3.0 * Ep * I / L * k_factor
    mk = moment_curvature_rect(c, width, t, kappa_max=0.6 / t)
    _, th, M = hinge_to_spring_table(hinge_table(mk, hinge_len or t))
    M = np.ascontiguousarray(M * clamp_factor, float)
    return dict(k=float(k), acc_tab=np.ascontiguousarray(th, float), F_tab=M, M_y=float(M[0]), Mp=float(M.max()),
                source=f"strip {width * 1e3:.3g} x {t * 1e3:.3g} mm, L {L * 1e3:.3g} mm, clamp x{clamp_factor:g}; "
                       f"curve: {c.source[:80]}")
