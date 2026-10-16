"""Material data types. All values SI (Pa, kg/m^3, W/m/K, J/kg/K, 1/K)."""
from dataclasses import dataclass, field, replace
import math


@dataclass(frozen=True)
class Isotropic:
    name: str
    E: float                 # Young's modulus
    nu: float                # Poisson ratio
    rho: float               # density
    sy: float                # 0.2% yield (or flow stress for polymers)
    su: float                # ultimate tensile
    family: str = "metal"
    elong: float = float("nan")      # elongation at break (fraction)
    k: float = float("nan")          # thermal conductivity
    cp: float = float("nan")         # specific heat
    alpha: float = float("nan")      # CTE
    resistivity: float = float("nan")  # electrical, ohm.m
    hardness: str = ""
    cowper_D: float = float("nan")   # Cowper-Symonds D (1/s)
    cowper_q: float = float("nan")   # Cowper-Symonds q
    jc: tuple = ()                   # Johnson-Cook (A, B, n, C, m, Tmelt, Tref)
    note: str = ""

    @property
    def G(self):
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def tau_y(self):
        """Shear yield by von Mises (sy / sqrt 3)."""
        return self.sy / math.sqrt(3.0)

    @property
    def wave_speed(self):
        """1-D bar wave speed sqrt(E/rho)."""
        return math.sqrt(self.E / self.rho)

    @property
    def specific_strength(self):
        return self.sy / self.rho

    def plate_D(self, t):
        """Flexural rigidity E t^3 / (12 (1 - nu^2))."""
        return self.E * t ** 3 / (12.0 * (1.0 - self.nu ** 2))

    def with_(self, **kw):
        """Copy with fields changed, e.g. mat.with_(sy=1.2*mat.sy, name='rate-scaled')."""
        return replace(self, **kw)


@dataclass(frozen=True)
class Lamina:
    """Unidirectional (or balanced fabric) ply, principal axes 1 = fiber, 2 = transverse."""
    name: str
    E1: float
    E2: float
    G12: float
    nu12: float
    rho: float
    Xt: float     # longitudinal tensile strength
    Xc: float     # longitudinal compressive strength (positive number)
    Yt: float     # transverse tensile
    Yc: float     # transverse compressive (positive)
    S: float      # in-plane shear strength
    t_ply: float = 0.125e-3
    alpha1: float = 0.0
    alpha2: float = 0.0
    Vf: float = 0.6
    note: str = ""

    @property
    def nu21(self):
        return self.nu12 * self.E2 / self.E1


@dataclass(frozen=True)
class Fluid:
    name: str
    rho: float
    mu: float      # dynamic viscosity
    k: float
    cp: float
    note: str = ""

    @property
    def nu_kin(self):
        return self.mu / self.rho

    @property
    def Pr(self):
        return self.cp * self.mu / self.k
