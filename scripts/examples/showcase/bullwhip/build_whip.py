"""Writes bullwhip_crack.yaml: a tapered whip as a chain of beams, thinner and finer toward the tip.

    python build_whip.py                  # the shipped model
    python build_whip.py --pieces 12 --segments 14 --length 2.5 --d0 16 --d1 1.8

The whip is `pieces` beams, each its own round section (the diameter falls linearly from d0 at the handle to d1 at
the tip), joined at small knot bodies. Each piece is split into segments, more of them where the whip is thin (the
loop is tightest there): the tip piece gets `segments`, a piece of diameter d gets segments x (d_tip / d)^0.7, at
least 2. A finer mesh moves the tip speed by about 1 % (12 x 10 against 12 x 14).
"""
import argparse
import math
import os

ap = argparse.ArgumentParser()
ap.add_argument("--pieces", type=int, default=12)
ap.add_argument("--segments", type=int, default=10, help="segments in the tip piece")
ap.add_argument("--length", type=float, default=2.5, help="whip length, m")
ap.add_argument("--d0", type=float, default=16.0, help="diameter at the handle, mm")
ap.add_argument("--d1", type=float, default=1.8, help="diameter at the tip, mm")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "bullwhip_crack.yaml"))
o = ap.parse_args()

NP, SEG, LW = o.pieces, o.segments, o.length
d0, d1 = o.d0 / 1e3, o.d1 / 1e3
arm = 0.6; th0 = -172.0; z_sh = 1.6
t0 = math.radians(th0); ux, uz = math.cos(t0), -math.sin(t0)
P = lambda s: (round(s * ux, 6), round(z_sh + s * uz, 6))
seg_len = LW / NP
dia = lambda x: d0 + (d1 - d0) * x                    # x: 0 at the handle, 1 at the tip
d_tip = dia((NP - 0.5) / NP)
nseg = lambda d: max(2, round(SEG * (d_tip / d) ** 0.7))

L = [f'''simblocks: 1
name: Bullwhip crack
question: "A {LW:g} m tapered whip is thrown overhead and the arm stops. A loop runs down the whip and speeds up as the
  whip thins. How fast does the tip go? The crack is the tip passing the speed of sound (343 m/s): a small sonic boom."
dim: 3d
gravity: 9.81
notes: >
  Written by build_whip.py (change the taper or the mesh there). The whip is {NP} beams, each its own round section,
  falling from {o.d0:g} mm at the handle to {o.d1:g} mm at the tip, split finer toward the tip where the loop is
  tightest. Every node feels crossflow air drag with a Cd that rises through Mach 1, and the cracker (the tuft of
  thread at the end) has its own. Scenario 'vacuum' switches the air off: the loop then keeps tightening (a whip in
  vacuum focuses its energy without limit), so that run stops just after the crack.
sources:
  - "McMillen T, Goriely A (2003). Whip waves. Physica D 184: 192-225 (the tapered whip, the loop, and why drag sets the tip speed)"
  - "Krehl P, Engemann S, Schwenkel D (1998). The puzzle of whip cracking, uncovered by a correlation of whip-tip kinematics with shock wave emission. Shock Waves 8: 1-9 (tip speeds about Mach 2)"
  - "Hoerner SF (1965). Fluid-Dynamic Drag, ch. 15-16 (crossflow drag of a cylinder, its rise near Mach 1)"

params:
  E_whip: {{value: 2 GPa, note: "braided nylon, bending stiffness of the braid"}}
  G_whip: {{value: 0.75 GPa}}
  rho_whip: {{value: 1150 kg/m^3}}
  swing: {{value: 0.14 s, note: "how long the throw takes, shoulder from behind to forward"}}
  th_start: {{value: {th0:g} deg, note: "arm and whip laid out behind, nearly level"}}
  th_end: {{value: 25 deg, note: "where the arm stops: forward and a little down"}}
  kp: {{value: 3000 N*m/rad, note: "how hard the shoulder follows the throw"}}
  kd: {{value: 60 N*m*s/rad}}
  air: {{value: 1, note: "air drag on the whip and the cracker: 1 on, 0 off (vacuum)"}}
  cracker_mass: {{value: 0.2 g, note: "the tuft of thread at the tip"}}
  cracker_area: {{value: 30 mm^2, note: "its frontal area, frayed threads"}}
  t_stop: {{value: 0.175 s, note: "the crack comes at about 0.135 s; the rest is the follow-through"}}

scenarios:
  vacuum: {{air: 0, t_stop: 0.136 s}}

bodies:
  shoulder:
    ground: true
    points: {{s: [0, 0, {z_sh}]}}
  arm:
    dof: free
    shape: {{kind: rod, L: {arm} m, A: 1.2e-3 m^2, density: 1000 kg/m^3, axis: [{ux:.6f}, 0, {uz:.6f}]}}
    at: [{P(arm / 2)[0]}, 0, {P(arm / 2)[1]}]
    points: {{s: [{-arm / 2 * ux:.6f}, 0, {-arm / 2 * uz:.6f}], h: [{arm / 2 * ux:.6f}, 0, {arm / 2 * uz:.6f}]}}
    label: arm and handle
    note: "forearm, hand and handle as one rod turned about the shoulder"
''']
names = []
for i in range(NP):
    s_end = arm + (i + 1) * seg_len
    nm = f"k{i + 1}" if i < NP - 1 else "cracker"
    x, z = P(s_end)
    if nm == "cracker":
        m, I, lab = "$cracker_mass", "1e-9", "\n    label: cracker"
    else:
        d_end = dia((i + 1) / NP); ns = nseg(d_end)
        mseg = 1150 * math.pi / 4 * d_end ** 2 * (seg_len / ns)
        m, I, lab = f"{0.25 * mseg:.4g}", f"{mseg * (seg_len / ns) ** 2 / 12:.3g}", ""
    L.append(f'''  {nm}:
    dof: free
    mass: {m}
    inertia: {I}
    at: [{x}, 0, {z}]
    points: {{c: [0, 0, 0]}}{lab}
''')
    names.append(nm)
L.append('''
control:
  # the arm's angle about the shoulder from its turn (rotation about y), and the throw: a smooth step from th_start to
  # th_end over the swing time, followed by a stiff shoulder (PD)
  tip_mach: "= cracker.speed / 343"
  drive: "= $kp * ($th_start + ($th_end - $th_start) * clip(t / $swing, 0, 1) ** 2 * (3 - 2 * clip(t / $swing, 0, 1)) - $th_start - 2 * atan2(arm.qy, arm.qw)) - $kd * arm.wy"
  arm_deg: "= ($th_start + 2 * atan2(arm.qy, arm.qw)) * 57.29578"

elements:
  shoulder_joint: {type: hinge, rigid: true, axis: y, measure: [shoulder.s, arm.s]}
  muscle: {type: force, measure: {twist: [shoulder, arm], axis: y}, P: {control: drive, init: 0}, label: shoulder torque}
''')
CD = "{mach: [0, 0.4, 0.6, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0], cd: [1.2, 1.2, 1.3, 1.5, 1.7, 1.8, 1.7, 1.55, 1.45, 1.35]}"
prev = "arm.h"
for i, nm in enumerate(names):
    d_mid = dia((i + 0.5) / NP)
    L.append(f'''  piece{i + 1}: {{type: beam, measure: [{prev}, {nm}.c], section: {{block: circle, d: {d_mid * 1e3:.2f} mm}}, E: $E_whip, G: $G_whip, density: $rho_whip, segments: {nseg(d_mid)}, damping: 0.02, orient: y, air_drag: {{Cd: {CD}, scale: $air}}}}
''')
    prev = f"{nm}.c"
L.append('''  cracker_air: {type: aero_drag, measure: cracker, area: $cracker_area, Cd: {mach: [0, 0.6, 0.9, 1.0, 1.2, 2.0, 3.0], cd: [1.0, 1.05, 1.35, 1.5, 1.45, 1.3, 1.25]}, scale: $air, label: drag on the cracker}

outputs:
  tip_speed: {body: cracker, quantity: speed, reduce: max, unit: m/s, label: fastest the tip goes}
  t_crack: {body: cracker, quantity: speed, reduce: time_of_max, unit: ms, label: when}
  mach: {expr: "= $tip_speed / 343", label: tip speed over the speed of sound}
  arm_rate: {body: arm, quantity: omega, reduce: max_abs, unit: rad/s, label: fastest the arm turns}
  hand_speed: {expr: "= $arm_rate * 0.6", unit: m/s, label: fastest the hand goes (0.6 m from the shoulder)}
  cracker_drag_max: {element: cracker_air, quantity: force, reduce: max, unit: N, label: peak air drag on the cracker}

run:
  t_end: $t_stop
''')
open(o.out, "w").write("".join(L))
print("wrote", o.out)
