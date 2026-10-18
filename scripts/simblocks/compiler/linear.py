"""Linear analysis of a compiled model: natural frequencies and mode shapes, and a direct static solve.

Both read the model through its own solver, so every element law, joint, beam, contact state and gravity is
included exactly as a run sees it. One solver step from rest gives the accelerations a(q) = M^-1 Q(q) (semi-implicit
Euler: v1 = dt a(q0)). Central differences over each free coordinate give the Jacobian J = da/dq.

Modes: J = -M^-1 K, so the natural frequencies are omega^2 = eigenvalues of -J and the mode shapes its
eigenvectors. Free directions (a body nothing holds) come out as rigid-body modes at zero frequency. Exact
(rigid: true) joints are kept by the solver's projection; perturbing across one shows up far above anything the time
step can resolve (omega dt > 2.5) and is set aside.

Static: Newton's method on a(q) = 0 from the starting positions, the Jacobian above, a least-squares step (so free
directions are left alone) and a line search. Loads and drives act as they do at t = 0. The model's outputs are then
read at the equilibrium (one step from rest there)."""
import math

import numpy as np

from .loader import SpecError


def _copy(M):
    import copy
    M2 = copy.copy(M)
    M2.arrays = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in M.arrays.items()}
    return M2


def dofs(M):
    """Free coordinates: (label, body index, kind, k). kind: 'q' (1d), 'c' (2d coordinate k), 't' (3d translation k),
    'r' (3d free rotation about world axis k), 's' (3d spin angle)."""
    a = M.arrays; dim = getattr(M, "dim", "1d"); out = []
    for i, b in enumerate(M.body_names):
        if "__" in b and dim != "1d":
            pass
        if dim == "1d":
            out.append((b, i, "q", 0))
        elif dim == "2d":
            for k, nm in enumerate(("x", "y", "rz")):
                if a["minv"][3 * i + k] > 0:
                    out.append((f"{b}.{nm}", i, "c", k))
        else:
            for k, nm in enumerate(("x", "y", "z")):
                if a["minv"][i, k] > 0:
                    out.append((f"{b}.{nm}", i, "t", k))
            from . import engine3d as E3
            if a["rmode"][i] == E3.R_FREE:
                for k, nm in enumerate(("rx", "ry", "rz")):
                    out.append((f"{b}.{nm}", i, "r", k))
            elif a["rmode"][i] == E3.R_SPIN:
                out.append((f"{b}.spin", i, "s", 0))
    return out


def _qmul(p, q):
    w1, x1, y1, z1 = p; w2, x2, y2, z2 = q
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _apply(M, D, x):
    """A copy of M starting at rest at the base positions moved by x (one entry per dof)."""
    M2 = _copy(M); a = M2.arrays; dim = getattr(M, "dim", "1d")
    if dim == "3d":
        a["V0"][:] = 0.0; a["Lw0"][:] = 0.0; a["wsp0"][:] = 0.0
    else:
        a["v0"][:] = 0.0
    for (lab, i, kind, k), dx in zip(D, x):
        if dx == 0.0:
            continue
        if kind == "q":
            a["q0"][i] += dx
        elif kind == "c":
            a["q0"][3 * i + k] += dx
        elif kind == "t":
            a["X0"][i, k] += dx
        elif kind == "s":
            a["th0"][i] += dx
        else:                                    # a small turn about world axis k: dq (x) q
            h = 0.5 * dx; axis = np.zeros(3); axis[k] = 1.0
            dq = np.array([math.cos(h), *(math.sin(h) * axis)])
            qn = _qmul(dq, a["quat0"][i]); a["quat0"][i] = qn / np.linalg.norm(qn)
    return M2


def _step(M2, history=True):
    from .model import run_model
    return run_model(M2, dt=M2.dt_static, history=history, n_log=2, timeout=None, _n=1)


def accel(M, D, x):
    """Accelerations of every dof at positions x (from rest), and the one-step run (outputs, history)."""
    M2 = _apply(M, D, x); M2.dt_static = M.dt_static
    R, H = _step(M2)
    dt = M.dt_static; out = np.zeros(len(D))
    for j, (lab, i, kind, k) in enumerate(D):
        b = M.body_names[i]
        key = {"q": "velocity", "c": ("vx", "vy", "omega")[k], "t": ("vx", "vy", "vz")[k],
               "r": ("wx", "wy", "wz")[k], "s": "omega"}[kind]
        out[j] = float(H[f"{b}.{key}"][-1]) / dt
    return out, R, H


def _scale(M, D):
    """Perturbation per dof: 1e-7 of the model's size for lengths, 1e-7 rad for angles."""
    a = M.arrays
    pos = a["q0"] if "q0" in a else a["X0"]
    L = max(float(np.max(np.abs(pos))) if np.size(pos) else 0.0, 0.01)

    def angle(i, kind, k):
        return kind in ("r", "s") or (kind == "c" and k == 2) or (kind == "q" and M.dof[i] == "rotation")
    return np.array([1e-7 if angle(i, kind, k) else 1e-7 * L for (lab, i, kind, k) in D])


def jacobian(M, D, x0, h):
    n = len(D); J = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n); e[j] = h[j]
        ap, _, _ = accel(M, D, x0 + e); am, _, _ = accel(M, D, x0 - e)
        J[:, j] = (ap - am) / (2 * h[j])
    return J


def _prep(M):
    D = dofs(M)
    if not D:
        raise SpecError("the model has no free coordinates to analyse", "bodies")
    if len(D) > 400:
        raise SpecError(f"{len(D)} free coordinates: modes and static solves stop at 400", "bodies")
    M.dt_static = M.dt
    return D


def modes(M, n_max=12, about="equilibrium"):
    """Natural frequencies (Hz) and mode shapes. about: 'equilibrium' (default: first settle the model under its
    loads and gravity, so preloads count, like a hanging pendulum's pin tension) or 'start' (as written)."""
    D = _prep(M)
    h = _scale(M, D); x0 = np.zeros(len(D)); where = "start"
    if about == "equilibrium":
        st = static(M)
        if st["converged"]:
            x0 = np.array([st["moved"][d[0]] for d in D]); where = "equilibrium"
        else:
            where = "start (no equilibrium: " + (", ".join(st["free"]) or "it keeps moving") + ")"
    J = jacobian(M, D, x0, h)
    lam, vec = np.linalg.eig(-J)
    lam = np.real(lam); vec = np.real(vec)
    dt = M.dt; out = []; rigid = 0; skipped = 0
    wmax = max((math.sqrt(l) for l in lam if l > 0), default=0.0)
    for idx in np.argsort(lam):
        l = lam[idx]
        if l <= max(1e-9 * wmax ** 2, 1e-12):
            rigid += 1; continue
        w = math.sqrt(l)
        if w * dt > 2.5:
            skipped += 1; continue
        v = vec[:, idx]; v = v / (np.max(np.abs(v)) or 1.0)
        if v[np.argmax(np.abs(v))] < 0:
            v = -v
        out.append(dict(f_hz=w / (2 * math.pi), omega=w, rpm=w * 60 / (2 * math.pi),
                        shape={D[j][0]: float(v[j]) for j in range(len(D)) if abs(v[j]) > 1e-3}))
    return dict(modes=out[:n_max], n_found=len(out), rigid=rigid, constraint=skipped, dofs=[d[0] for d in D],
                dt=dt, about=where, note=_note(M))


def _note(M):
    oc = [n for e, n in enumerate(M.el_names) if M.arrays["one_sided"][e] == 1 and "__" not in n]
    return ("one-sided contacts count as they are at the start (touching or not): " + ", ".join(oc)) if oc else ""


def static(M, tol=1e-9, max_iter=40):
    """Equilibrium positions from the start, with every output read there."""
    D = _prep(M)
    h = _scale(M, D); x = np.zeros(len(D))
    a, R, H = accel(M, D, x)
    a0 = max(float(np.linalg.norm(a)), 1e-30)
    J = jacobian(M, D, x, h)
    it = 0
    for it in range(1, max_iter + 1):
        if np.linalg.norm(a) <= tol * a0 or np.linalg.norm(a) < 1e-12:
            break
        dx, *_ = np.linalg.lstsq(J, -a, rcond=1e-10)
        step = 1.0; ok = False
        for _ in range(30):
            a2, R2, H2 = accel(M, D, x + step * dx)
            if np.linalg.norm(a2) < np.linalg.norm(a):
                ok = True; break
            step *= 0.5
        if not ok:
            break
        x = x + step * dx; a, R, H = a2, R2, H2
        if it % 3 == 0 or step < 1.0:
            J = jacobian(M, D, x, h)              # refresh for nonlinear laws (contacts, plasticity, joints)
    res = float(np.linalg.norm(a))
    converged = res <= max(1e-6 * a0, 1e-9)
    free = []
    if not converged:
        w, V = np.linalg.eig(-J)
        for idx in np.where(np.abs(np.real(w)) < 1e-9 * max(np.max(np.abs(np.real(w))), 1e-30))[0]:
            v = np.real(V[:, idx]); free.append(D[int(np.argmax(np.abs(v)))][0])
    outs = {k: v for k, v in R.items() if not k.startswith("_")}
    return dict(converged=bool(converged), iterations=it, residual=res, start_residual=a0,
                moved={D[j][0]: float(x[j]) for j in range(len(D))}, outputs=outs, free=sorted(set(free)),
                note=_note(M))
