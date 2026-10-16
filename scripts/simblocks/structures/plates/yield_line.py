"""Rigid-plastic collapse loads of plates by yield-line theory (upper bounds).
m_p = plastic moment per unit width = sy t^2 / 4 (equal positive and negative capacity assumed).
Sources: Johansen yield-line theory; Jones, "Structural Impact" (2nd ed.) ch. 2-4."""
import math

from ...registry import block

SRC = "yield-line theory (Johansen; Jones Structural Impact)"


@block("structures", "plates", "general", "plastic_moment", source="sy t^2 / 4")
def plate_mp(sy, t):
    """Plastic moment per unit width of a plate: m_p = sy t^2 / 4."""
    return sy * t * t / 4.0


@block("structures", "plates", "any_shape", "point_interior", source=SRC + " (fan mechanism)",
       tags=("collapse", "puncture precursor"))
def collapse_point_interior(mp, clamped=False):
    """Point load collapse on an interior point of any plate: 2 pi m_p (SS edges) or 4 pi m_p (clamped)."""
    return (4.0 if clamped else 2.0) * math.pi * mp


@block("structures", "plates", "free_edge", "point_at_edge", source=SRC + " (half fan at free edge)",
       tags=("lip", "plate edge"))
def collapse_point_free_edge(mp):
    """Point load at a free edge: half fan, pi m_p."""
    return math.pi * mp


@block("structures", "plates", "circular", "uniform_pressure", source=SRC)
def collapse_circular_pressure(mp, a, clamped=False):
    """Uniform-pressure collapse of a circular plate: 6 m_p/a^2 (SS), 12 m_p/a^2 (clamped)."""
    return (12.0 if clamped else 6.0) * mp / (a * a)


@block("structures", "plates", "rectangular", "uniform_pressure", source=SRC + " (Jones eq. 2.84 form)")
def collapse_rect_pressure(mp, a, b, clamped=False):
    """Uniform-pressure collapse of a rectangle (long a, short b), optimized yield-line pattern:
    q = 24 m_p / (b^2 (sqrt(3 + beta^2) - beta)^2), beta = b/a (SS). Clamped doubles capacity."""
    a, b = max(a, b), min(a, b)
    beta = b / a
    q = 24 * mp / (b * b * (math.sqrt(3 + beta * beta) - beta) ** 2)
    return 2.0 * q if clamped else q


@block("structures", "plates", "three_edge_clamped", "point_free_edge", source=SRC + " (approximate)",
       tags=("hinged plate", "3-edge clamped"))
def collapse_three_edge_strip(mp, b_eff, L_slant):
    """Fold collapse capacity of a face clamped on three edges, loaded near the free lip:
    M_c ~ m_p (b_eff + 2 L_slant) (root fold plus two side folds). Returns collapse moment about the root."""
    return mp * (b_eff + 2.0 * L_slant)
