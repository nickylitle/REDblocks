"""Adaptive time step (run.dt: adaptive).

The step is still an explicit step of the same closed-form element laws; only its length changes from step to step:
  * free motion runs at dt_free: the automatic step worked out with the contact elements left out;
  * while a contact touches, or its gap would close within the horizon at the speed it is closing, the step is
    dt_contact: the usual automatic step with every element in;
  * a stiffness or damping driven by a control signal sets its own step from the signal's current value
    (dt = C |x|^-p, fitted from the model compiled at two values of the field);
  * the step shrinks at once and grows by at most `growth` a step; it never falls below dt_min.

adp (float64) is the settings in, statistics out:
  0 mode (1 = adaptive)   1 dt_free   2 dt_contact   3 horizon (s)   4 growth   5 t_end   6 dt_min   7 log interval (s)
  10 steps   11 smallest step   12 largest step   13 steps at the contact step or smaller   14 time reached
adf (n_act, 4): [C_up, p_up, C_down, p_down] for each controlled field: the step it needs as the signal rises
  above its first value and as it falls below it (C = 0: that way the field does not set the step)
adc (n_el) int64: 1 = a contact element (the ones dt_free leaves out)
"""
import numpy as np
from numba import njit

NADP = 16


def fixed_adaptive(ne=0, na=0):
    """The arguments for a fixed-step run (adaptive off)."""
    return np.zeros(NADP), np.zeros((na, 4)), np.zeros(ne, np.int64)


@njit(cache=True, nogil=True)
def pick_dt(i, t, dt, adp, adf, adc, active, se, sp, inc, use_inc, sig, casig):
    """Length of the next step. dt: the last step's length. sp: each contact's measure at the last pick."""
    want = adp[1]; H = adp[3]
    touch = False
    for e in range(adc.shape[0]):
        if adc[e] == 0:
            continue
        s = se[e]
        rate = (s - sp[e]) / dt if i > 0 and dt > 0.0 else 0.0
        sp[e] = s
        if active[e] == 0:
            continue
        on = inc[e] == 1 if use_inc else s > 0.0
        if on or (s <= 0.0 and rate > 0.0 and s + rate * H > 0.0):
            touch = True
    if touch:
        want = adp[2]
    for a in range(adf.shape[0]):
        x = abs(sig[casig[a]])
        for c in (0, 2):
            if adf[a, c] > 0.0 and x > 0.0:
                d = adf[a, c] * x ** (-adf[a, c + 1])
                if d < want:
                    want = d
    if want < adp[6]:
        want = adp[6]
    if i == 0:
        new = want if want < adp[2] else adp[2]      # start at the contact step: nothing is known yet
    else:
        g = dt * adp[4]
        new = want if want < g else g
    rem = adp[5] - t
    last = False
    if rem <= new:
        new = rem; last = True
    elif rem < 2.0 * new:
        new = 0.5 * rem; last = True                 # two equal last steps, never a sliver
    adp[10] += 1.0
    if new < adp[11] and not last:
        adp[11] = new
    if new > adp[12]:
        adp[12] = new
    if new <= adp[2] * (1.0 + 1e-12):
        adp[13] += 1.0
    return new
