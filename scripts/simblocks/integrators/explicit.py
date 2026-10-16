"""Fixed-step explicit integrators and stable-time-step estimators.

Pattern for njit sims: write a deriv(s, t, p, ds) kernel, then call rk4_step / semi_implicit_step from an
njit loop. The python-level `run_fixed` driver is for prototyping and tests."""
import math
import numpy as np

from ..registry import block
from .._jit import kernel


@block("integrators", "step", "first_order", "explicit_euler", source="forward Euler",
       tags=("legacy", "symplectic"))
def euler_step(f, s, t, dt, p):
    """s_{n+1} = s_n + dt f(s_n, t_n). Energy drifts UP for oscillators: use only with tiny dt."""
    return s + dt * f(s, t, p)


@block("integrators", "step", "second_order", "semi_implicit_euler", source="symplectic Euler (Hairer et al.)",
       tags=("contact", "penalty", "default for impact"))
def semi_implicit_step(acc, x, v, t, dt, p):
    """v_{n+1} = v_n + dt a(x_n, v_n); x_{n+1} = x_n + dt v_{n+1}. Symplectic: bounded energy error.
    This is what the legacy hinged-plate sim effectively does (velocity updated, then used next step)."""
    v1 = v + dt * acc(x, v, t, p)
    return x + dt * v1, v1


@block("integrators", "step", "second_order", "velocity_verlet", source="Swope et al. 1982",
       tags=("contact", "2nd order"))
def velocity_verlet_step(acc, x, v, a, t, dt, p):
    """Velocity Verlet (velocity-independent forces exact to 2nd order). Returns (x, v, a_new)."""
    x1 = x + dt * v + 0.5 * dt * dt * a
    a1 = acc(x1, v, t + dt, p)
    return x1, v + 0.5 * dt * (a + a1), a1


@block("integrators", "step", "first_order", "rk4", source="classical Runge-Kutta",
       tags=("drive sim", "smooth"))
def rk4_step(f, s, t, dt, p):
    """Classical RK4 step for smooth systems (drive dynamics). Avoid across contact on/off switches."""
    k1 = f(s, t, p)
    k2 = f(s + 0.5 * dt * k1, t + 0.5 * dt, p)
    k3 = f(s + 0.5 * dt * k2, t + 0.5 * dt, p)
    k4 = f(s + dt * k3, t + dt, p)
    return s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


@block("integrators", "driver", "general", "fixed_step", kind="builder", source="loop", tags=("prototype",))
def run_fixed(step, state0, t_end, dt, log_every=1, **kw):
    """Python driver: repeatedly calls step(state, t, dt) -> state. Returns (t array, list of states)."""
    n = int(round(t_end / dt))
    s = state0; out_t = [0.0]; out_s = [state0]
    for i in range(1, n + 1):
        s = step(s, (i - 1) * dt, dt)
        if i % log_every == 0:
            out_t.append(i * dt); out_s.append(s)
    return np.array(out_t), out_s


# ---------------------------------------------------------------- stability
@block("integrators", "stability", "spring_mass", "critical_dt", source="central difference: dt < 2/w_max",
       tags=("dt check", "penalty stiffness"))
def critical_dt_spring(k, m, zeta=0.0):
    """Critical step of explicit central-difference / semi-implicit Euler: (2/w)(sqrt(1+z^2) - z)."""
    w = math.sqrt(k / m)
    return 2.0 / w * (math.sqrt(1 + zeta * zeta) - zeta)


@block("integrators", "stability", "rotational", "critical_dt", source="dt < 2 sqrt(I/k_theta)")
def critical_dt_rotational(k_theta, I):
    """Critical step for a rotational spring on inertia I."""
    return 2.0 * math.sqrt(I / k_theta)


@block("integrators", "stability", "contact", "hertz_dt", source="contact duration / samples",
       tags=("impact", "resolution"))
def dt_for_contact(t_contact, samples=200):
    """Step that resolves a contact event of duration t_contact with `samples` steps."""
    return t_contact / samples


@block("integrators", "stability", "bar_element", "courant", source="CFL: dt < L_e / c")
def courant_dt(L_element, E, rho, safety=0.9):
    """Courant step for a discretized bar/plate element: safety * L / sqrt(E/rho)."""
    return safety * L_element / math.sqrt(E / rho)


@block("integrators", "stability", "general", "recommend", source="min over checks with safety factor",
       tags=("dt check",))
def recommend_dt(checks, safety=0.1):
    """Recommend dt = safety * min(list of critical steps). safety 0.1 keeps penalty contact accurate,
    not just stable. Returns (dt, governing_index)."""
    i = int(np.argmin(checks))
    return safety * checks[i], i
