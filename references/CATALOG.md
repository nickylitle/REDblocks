# Block catalog

151 model blocks (the editor's library) plus 173 internal ones at the end. Path = domain / category / geometry / load. Kind: builder (pre-loop), kernel (in-loop, njit-safe), analytic (closed form).


## actuators / battery

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `lipo_ocv` | pack | open_circuit | analytic | LiPo typical OCV curve (approx) | Approximate LiPo pack open-circuit voltage vs state of charge (piecewise; fine for sag studies). |

## actuators / gearbox

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `backdrive_efficiency` | general | backdrive_efficiency | analytic | Matsuki, Nagano & Fujimoto, IEEE/ASME T-Mech 24(6), 2019, Eq. 38, as reproduced in Sim & Ramos arXiv:2011.02506 Eq. 15 (secondary) | Back-driven efficiency from forward efficiency for a reduction G (motor turns : output turns, e.g. 20). |
| `gearbox_output_torque_dir` | general | output_torque_directional | kernel | Sim & Ramos, arXiv:2106.01842 Eqs. 2, 5, 6 (directional efficiency: forward eta_f, backward eta_b < eta_f) | Output torque with direction-dependent efficiency: motoring G T eta_f; back-driven (load drives the |

## actuators / motor

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `motor_from_curve` | dc | from_curve | builder | DC motor equations fit to the vendor 12 V curve (fit method from a user-supplied drive sim) | Fit Kt, R and gearbox efficiency to a gearmotor's free speed, stall torque, stall current at V_test. |
| `motor_from_kv` | dc | from_datasheet | builder | DC motor equations | DC-equivalent motor from Kv (rpm/V), winding resistance R (ohm) and ESC current limit. |
| `spin_up_time` | rotor | spin_up_current_limited | analytic | constant torque then back-EMF limited | Time to reach w_target (load rad/s) for a current-limited motor, gear ratio G (motor:load). |

## aero / atmosphere

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `isa_troposphere` | isa | troposphere | analytic | ICAO Standard Atmosphere (Doc 7488) | ISA temperature (K), pressure (Pa), density (kg/m^3) for 0 <= h <= 11 km. |

## aero / drag

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `terminal_velocity` | bluff_body | terminal_velocity | analytic | mg = q Cd A | Terminal velocity sqrt(2 m g / (rho Cd A)). |
| `induced_drag_cd` | finite_wing | induced | analytic | Prandtl lifting line (Anderson 5.3) | Induced drag coefficient CL^2 / (pi e AR). |
| `cd_sphere` | sphere | cd_vs_re | analytic | Clift, Grace & Weber 1978 (Re < 3e5) | Sphere drag coefficient vs Reynolds number (Schiller-Naumann + Clift correction). |

## aero / groups

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `mach` | general | mach | analytic | a = sqrt(gamma R T) | Mach number v / sqrt(gamma R T). |
| `reynolds` | general | reynolds | analytic | definition | Reynolds number rho v L / mu. |

## aero / lift

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `thin_airfoil_cl` | airfoil | thin_airfoil | analytic | Anderson 4.8 (thin airfoil theory) | Section lift coefficient 2 pi (alpha - alpha_L0), radians. |
| `finite_wing_lift_slope` | finite_wing | lift_slope | analytic | Helmbold 1942 (low aspect ratio valid) | 3-D lift-curve slope (per rad), Helmbold/DATCOM form. |

## aero / propulsion

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `ideal_hover_power` | actuator_disk | hover_power | analytic | Rankine-Froude momentum theory | Ideal induced power to produce thrust T statically: P = T^1.5 / sqrt(2 rho A). |
| `propeller_thrust` | propeller | thrust_coefficient | analytic | CT, CP definitions (n in rev/s) | Thrust from thrust coefficient: T = CT rho n^2 D^4. |

## aero / skin_friction

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `flat_plate_cf` | flat_plate | average | analytic | Blasius (laminar), Prandtl 1/5-power (turbulent) | Average skin-friction coefficient over a plate of length L. Transition at Re 5e5 unless forced. |

## aero / windage

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `bar_windage_torque` | bar_spinner | torque | kernel | strip theory with flat_plate_normal Cd | Strip-theory drag torque of a bar of half-length R and height h spinning about its center |
| `disk_windage_torque` | rotating_disk | torque | analytic | Daily & Nece 1960 / von Karman (free disk, both faces) | Aerodynamic drag torque on a free disk spinning in still air (both faces), N.m: |

## constitutive / damage

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `jc_fracture_strain` | metal | fracture_strain | kernel | Johnson & Cook 1985 | JC failure strain eps_f = [D1 + D2 exp(D3 sigma*)][1 + D4 ln rate*][1 + D5 T*]. |

## constitutive / hyperelastic

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `hyperelastic_stress` | elastomer | uniaxial_any | analytic | neo-Hookean; Mooney 1940, Rivlin 1948 | Nominal (engineering) uniaxial stress of an incompressible hyperelastic solid at stretch lam. |

## constitutive / rate

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `cowper_symonds_factor` | metal | yield_scale | kernel | Cowper & Symonds 1957 | Dynamic/static yield ratio 1 + (rate/D)^(1/q). Mild steel D=40.4, q=5; Al D=6500, q=4. |

## contact / hertz

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `hertz_cylinder_stress` | cylinder | line_stress | analytic | Roark 7e Table 14.1 case 2 | Parallel cylinders / cylinder on flat: half width b = sqrt(4 F' R*/(pi E*)), p0 = 2F'/(pi b). |
| `flat_punch_stiffness` | flat_punch | normal_stiffness | analytic | Boussinesq (Johnson eq. 3.38) | Rigid circular flat punch of radius a on a half-space: k = 2 a E*. |
| `effective_modulus` | general | effective | analytic | Johnson, Contact Mechanics ch. 4 | E* = 1 / ((1-nu1^2)/E1 + (1-nu2^2)/E2). |
| `effective_radius` | general | effective | analytic | Johnson, Contact Mechanics ch. 4 | R* = 1/(1/R1 + 1/R2). Flat surface: R2 = inf. Concave: negative radius. |
| `hertz_impact` | sphere | impact | analytic | Johnson, Contact Mechanics sec. 11.4 | Elastic sphere impact: peak indentation, peak force, contact duration. m* = m1 m2/(m1+m2). |
| `hertz_sphere_stress` | sphere | stress | analytic | Roark 7e Table 14.1 case 1 | Contact radius a, peak pressure p0 = 3F/(2 pi a^2), max shear ~0.31 p0 at depth 0.48 a (nu=0.3). |

## contact / impulse

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `collision_impulse` | rigid_body | collision | analytic | Newton restitution, planar | Normal impulse J = (1+e) v_rel / (1/m1 + 1/m2) and energy lost to the collision. |
| `effective_mass_at_point` | rigid_body | effective_mass | analytic | impulse-momentum, planar | Effective translational mass of a rigid body at a contact point: 1/(1/m + r_perp^2 / I). |

## contact / indentation

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `plastic_indentation_table` | sphere_tip | plastic_table | builder | flat-punch stiffness 2 a E* (Johnson eq. 3.38) then fully plastic mean pressure 3Y over a^2 = 2 R delta_p (Johnson, Contact Mechanics ch. 6), contact radius capped at R | Elastoplastic contact spring for a rounded tooth (corner radius R) pressing into a surface of yield Y: |

## contact / penalty

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `damping_for_restitution` | general | damping_from_restitution | analytic | Kawaguchi et al., JSME Trans. B 58(551):79-85, 1992, Eq. 11 as given in Itasca PFC 7.0 'Calibrating the normal critical damping ratio' (position-terminated contact) | Dashpot c giving restitution e for linear contact: zeta = -ln e / sqrt(pi^2 + ln^2 e), c = 2 zeta sqrt(k m). |
| `hysteresis_damping_factor` | general | hysteresis_factor | analytic | Machado, Moreira, Flores & Lankarani, Mech. Mach. Theory 2012, Eqs. 28-47, doi:10.1016/j.mechmachtheory.2012.02.010 (secondary for the originals) | alpha for hunt_crossley (F = K d^n (1 + alpha d_dot)) that targets restitution e at impact speed v. |
| `restitution_force_terminated` | general | restitution_force_terminated | analytic | Schwager & Poschel, Granular Matter 9:465-469, 2007, as given in Bartz arXiv:2204.10917 Eqs. 8-11 (secondary) | Restitution of a linear spring-dashpot whose force is clipped at zero (no pulling), which ends the |

## contact / ploughing

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `challen_oxley_mu` | wedge_asperity | friction_ratio | analytic | Challen & Oxley, Wear 53:229-243, 1979, wave model, as reproduced in Lubricants 6(2):53, 2018 Eqs. 2-3 (PARTIAL: standard form restored from a garbled extraction) | Tangential/normal ratio for a hard wedge of attack angle theta (rad) ploughing a plastic surface |

## geometry / drawing

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `outline_from_dxf` | dxf | outline | builder | ezdxf LWPOLYLINE/POLYLINE | Closed polylines from a DXF (e.g. a waterjet/laser file). Largest = outer, others = holes. mm by default. |
| `outline_from_image` | image | outline | builder | threshold + marching squares (skimage) | Trace a part silhouette from an image file. Scale by px_per_m, or by a known reference |

## integrators / stability

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `courant_dt` | bar_element | courant | analytic | CFL: dt < L_e / c | Courant step for a discretized bar/plate element: safety * L / sqrt(E/rho). |
| `critical_dt_spring` | spring_mass | critical_dt | analytic | central difference: dt < 2/w_max | Critical step of explicit central-difference / semi-implicit Euler: (2/w)(sqrt(1+z^2) - z). |

## materials / cards

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `isotropic_card` | isotropic | summary | builder | catalog + curve loader | Card dict for a catalog metal/polymer (resolved from loose text) including its stress-strain source. |
| `laminate_card` | laminate | from_text | builder | CLT (Jones; Daniel & Ishai) | Build a LaminateCard from text like 'T300/5208 [0/+-45/90]2s' or from lamina=dict(...) + layup=... |
| `load_card` | laminate | load | builder | saved card JSON | Load a saved LaminateCard by name (from cards/user/). |

## materials / constitutive_params

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `hardening_stress` | metal | flow_stress_any | analytic | Hollomon 1945; Swift 1952; Voce 1948; Ludwik 1909; Johnson & Cook 1983 | Flow stress (Pa) at plastic strain eps_p from a hardening law. |
| `jc_scale` | metal | rate_temperature | kernel | Johnson & Cook 1983 | Johnson-Cook multiplier on a quasi-static curve: (1 + C ln(rate/rate0)) (1 - T*^m). |

## materials / conversions

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `shore_A_to_E` | elastomer | hardness | analytic | Gent 1958 (ASTM D2240 correlation) | Young's modulus (Pa) of an elastomer from Shore A durometer (Gent correlation, valid ~20-95A). |

## materials / curves

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `linear_to_failure_curve` | brittle | uniaxial | builder | linear elastic | Linear-elastic curve to failure (composite fiber direction, brittle ceramics, glass). |
| `neo_hookean_curve` | elastomer | uniaxial | builder | neo-Hookean (incompressible) | Engineering uniaxial curve s = mu (lam - lam^-2), mu = E/3, to break elongation. |
| `load_curve` | lookup | auto | builder | see resolution order | Get the stress-strain Curve for loose material text. prefer in {None,'user','jc','ro'}. |
| `johnson_cook_curve` | metal | uniaxial | builder | Johnson & Cook 1983 | Engineering curve from JC parameters (quasi-static by default), cut at necking (Considere). |
| `ramberg_osgood_curve` | metal | uniaxial | builder | Ramberg & Osgood 1943 | Engineering curve to UTS: e = s/E + 0.002 (s/sy)^n, n fit to pass through (e_u, su). |
| `ramberg_osgood_n` | metal | uniaxial | analytic | Ramberg & Osgood 1943, MMPDS form | RO exponent n from 0.2%-offset yield and the ultimate at uniform engineering strain e_u. |

## materials / fluids

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `air_properties` | air | temperature | analytic | Sutherland 1893, ideal gas | Dry air at temperature T (K) and pressure p (Pa): returns Fluid with rho, mu, k, cp. |

## materials / laminates

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `ply_stresses` | ply | stress | analytic | Jones 4.4 | Material-axis stresses [s1, s2, t12] at top and bottom of every ply. Returns list of dicts. |
| `strength_ratio` | ply | failure_any | analytic | maximum stress; Tsai-Hill (Azzi-Tsai); Tsai-Wu 1971 with F12 = -0.5 sqrt(F11 F22) | Strength ratio R of one ply: the multiplier on its stress state (s1, s2, t12) that reaches failure. |
| `first_ply_failure` | stack | failure | analytic | first-ply failure (linear) | Load multiplier R at first-ply failure for proportional N, M. Returns (R, ply index, angle). |
| `laminate_ABD` | stack | stiffness | builder | Jones 4.22-4.24 | A, B, D matrices and ply z-coordinates. lamina = one Lamina or a list (one per ply). |
| `laminate_as_isotropic` | stack | effective | builder | CLT equivalent | Equivalent Isotropic material so Roark plate/beam blocks can be used on a laminate. |
| `laminate_engineering_constants` | stack | effective | analytic | Daniel & Ishai 7.9 | Effective in-plane Ex, Ey, Gxy, nuxy (membrane) and flexural Efx, Efy of the laminate. |
| `laminate_response` | stack | response | analytic | Jones 4.4 | Midplane strains eps0 and curvatures kappa from applied N, M (and optional dT). |
| `laminate_thermal_resultants` | stack | thermal | builder | Jones 4.4 (hygrothermal) | Thermal force/moment resultants NT, MT (per width) for a uniform temperature change dT. |

## materials / lookup

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `material_property` | catalog | property | builder | materials catalog (see MATERIAL_CATALOG.md) | One catalog property of a material given as text: E, nu, rho, sy, su, elong, G, tau_y, k, cp, alpha. |

## materials / micromechanics

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `halpin_tsai` | unidirectional | transverse | analytic | Halpin & Kardos 1976 | Halpin-Tsai transverse/shear property. xi = 2 for E2 (circular fibers), 1 for G12. |
| `lamina_from_constituents` | unidirectional | all | builder | ROM + Halpin-Tsai | Build a Lamina from fiber + matrix + fiber volume fraction. Strengths are rough first estimates: |
| `rule_of_mixtures` | unidirectional | fiber_direction | analytic | rule of mixtures (Voigt) | Longitudinal property by rule of mixtures: P = Pf Vf + Pm (1 - Vf). Works for E1, nu12, rho. |

## rigid_body / inertia

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `assemble` | assembly | planar | builder | composite-body mass properties (parallel axis); workflow from a user-supplied drive sim | Planar assembly from components [(name, m, x, y, I_own), ...] -> total m, COM, Izz about COM. |
| `parallel_axis` | general | parallel_axis | analytic | Steiner theorem | I about a parallel axis at distance d: I_cm + m d^2. |
| `parallel_axis_tensor` | general | parallel_axis_3d | analytic | Steiner theorem, tensor form | Inertia tensor about a point displaced by -d from the COM: I_cm + m (|d|^2 E - d d^T). |
| `polygon_plate` | polygon_plate | z | builder | Green's theorem polygon mass properties | Mass, centroid and polar inertia (about the centroid and about the origin) of a plate of thickness t |
| `mass_properties` | primitives | tensor | builder | standard mass-property tables (e.g. Meriam & Kraige App. B) | Mass, centre of mass and inertia tensor of a uniform primitive shape, about its centroid. |

## structures / beams

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `beam_plastic_spring` | any_section | plastic_spring | builder | Roark 7e Table 8.1 elastic stiffness + plastic-hinge mechanism with fibre-integrated M(kappa) | Force-deflection spring (k, acc_tab, F_tab) of a whole beam at its load point: elastic k from the support |
| `effective_width_45` | dispersion | effective_width | analytic | 45 deg load spread (St-Venant) | Effective resisting width at distance from a concentrated load: load_width + 2*distance, capped. |
| `beam_case` | general | any | builder | Roark 7e Table 8.1 cases 1a-1e, 2a-2e, 3a; rigid-plastic collapse by the mechanism method | Straight beam stiffness, moment and collapse load for a support and load case. |
| `beam_membrane_point_load` | restrained | point_load_membrane | analytic | Jones, Structural Impact 2nd ed. eqs. 1.5, 1.36, 1.37 (Pc, verified); finite-deflection law re-derived (m + n^2 = 1 upper bound), string limit checked vs Shi 2023 eq. 26. PARTIAL | Central point load P on a rigid-plastic rectangular beam (depth H, breadth B, span 2 L_half) at midspan |
| `spring_combination` | series | combination | analytic | compliance addition (series), stiffness addition (parallel) | Combined stiffness of a list of springs. |

## structures / buckling

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `euler_load` | column | axial | analytic | Euler; Roark 7e Table 15.1 | Euler critical load pi^2 E I / (K L)^2. ends in K_FACTORS. |
| `johnson_load` | column | axial_inelastic | analytic | J.B. Johnson parabola (Shigley 4-43) | Critical load with Johnson parabola below the transition slenderness, Euler above. |
| `plate_buckling_stress` | plate | in_plane | analytic | Roark 7e Table 15.2 (long plates) | sigma_cr = k pi^2 E / (12 (1 - nu^2)) (t/b)^2, b = loaded width (short side for long plates). |
| `cylinder_axial_buckling` | thin_cylinder | axial | analytic | Roark 7e Table 15.2 case 15 (classical) | Classical axial buckling of a thin cylinder E t / (R sqrt(3(1-nu^2))) times an imperfection knockdown. |

## structures / curved

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `curved_beam_rect` | rectangle | bending | analytic | Winkler-Bach; Shigley 10e eq. 3-65, Table 3-4 | Stresses at inner and outer fibers of a curved rectangular beam (b wide, h deep, inner radius r_inner) |
| `ring_diametral_load` | thin_ring | diametral_point_loads | builder | Castigliano + Mohr curvature integrals (numerical, exact); cf. Roark 7e Table 9.2 case 1 | Thin ring radius R squeezed by equal and opposite P along a diameter (Euler-Bernoulli, inextensible). |

## structures / fatigue

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `fatigue_sf` | general | mean_stress | analytic | Shigley 10e ch. 6 Table 6-6 / 6-7 | Fatigue factor of safety with mean stress: goodman, gerber, asme_elliptic, soderberg. |
| `marin_surface_factor` | general | marin_surface | analytic | Shigley 10e ch. 6 eq. 6-19, Table 6-2 | Surface factor ka = a Sut^b (Sut in MPa inside the fit). |
| `miner_damage` | general | miner | analytic | Palmgren-Miner linear damage | Miner sum D = sum n_i / N_i (failure predicted at D >= 1). |
| `sn_life` | general | basquin_sn | analytic | Shigley 10e ch. 6 eqs. 6-13 to 6-16 (f-fraction S-N line) | Cycles to failure on the Shigley S-N line through (1e3, f Sut) and (1e6, Se). f defaults by Sut (Fig. 6-18 fit). |
| `marin_size_factor` | round | marin_size | analytic | Shigley 10e ch. 6 eq. 6-20 (bending/torsion) | Size factor kb for a round in bending/torsion, d in m: 1.24 d_mm^-0.107 (2.79-51 mm), |
| `endurance_limit_steel` | steel | endurance_limit | analytic | Shigley 10e ch. 6 eq. 6-8 | Rotating-beam endurance limit Se' (Pa): 0.5 Sut for Sut <= 1400 MPa, else 700 MPa. |

## structures / impact

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `blunt_punch_line_force` | plate | blunt_punch_line_force | analytic | Prandtl/Hill flat-punch limit 2k(1 + pi/2) per unit width (Hill, Mathematical Theory of Plasticity 1950), as applied to blade indentation by Chaijit et al., J. Mech. Mater. Struct. 1(8), 2006, Sec. 3.1 | Fully plastic onset force of a flat tooth of tip width w and engaged length L pressed into a thick |
| `punch_perimeter` | plate | punch_perimeter | analytic | geometry | Perimeter of the punched patch: rect (w x l) -> 2(w + l); circle (d) -> pi d. |
| `shear_plug_capacity` | plate | shear_plug_capacity | analytic | punching shear F = P t S_su (blanking practice) | Force to shear a plug of the given perimeter out of a plate of thickness t: P t (ssu_ratio su). |

## structures / joints

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `bolt_preload_from_torque` | bolt | preload | analytic | T = K F d (Shigley 8-27), K ~ 0.2 dry, 0.15 lubed | Bolt preload from tightening torque. |
| `bolt_shear_capacity` | bolt | shear | analytic | direct shear on shank or thread root | Ultimate shear capacity: tau_u * area * planes. Uses root area if threads are in the shear plane. |
| `bolt_stiffness` | bolt | stiffness | analytic | AE/L | Bolt axial stiffness, shank and threaded portions in series. |
| `bolt_tensile_area` | bolt | tensile_area | analytic | ISO 898 / ASME B1.1: As = pi/4 (d - 0.9382 p)^2 | Tensile stress area of a metric/UN thread. |
| `member_stiffness_wileman` | member | stiffness | analytic | Wileman et al. 1991 (Shigley 8-23) | Clamped-member stiffness k_m = E d A exp(B d / l). Defaults are for steel. |
| `bearing_capacity` | plate_hole | bearing | analytic | Shigley 8-11 | Hole bearing capacity F = s_bearing d t (s_bearing ~ 1.5-2.0 su for metals edge distance >= 2d). |
| `net_tension_capacity` | plate_hole | net_tension | analytic | net section | Net-section tension capacity su (w - n d) t. |
| `tear_out_capacity` | plate_hole | tear_out | analytic | 2 tau t (e - d/2): MechaniCalc Lug Analysis (Melcon & Hoblit 1953; Bruhn 1973); AISC 360-16 J3-6c uses 1.2 l_c t F_u | Shear tear-out with edge distance e (hole center to edge): F = 2 tau_u t (e - d/2). |
| `thread_strip_capacity` | thread | stripping | analytic | FED-STD-H28 approximate shear area | Approximate stripping capacity of internal threads: tau_u * (pi d L_e * 0.75 approx engagement). |

## structures / plates

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `plate_collapse` | any_shape | collapse | analytic | yield-line theory (Johansen; Jones Structural Impact); three_edge approximate fold mechanism | Rigid-plastic collapse capacity of a plate by yield-line theory (upper bound). |
| `equivalent_load_radius` | general | point_load_radius | analytic | Roark 7e section 11.1 | Roark equivalent radius r0' for a load over a small radius r0 (< 0.5 t). |
| `flexural_rigidity` | general | rigidity | analytic | Kirchhoff plate theory | Plate flexural rigidity D = E t^3 / (12 (1 - nu^2)). |
| `plate_case` | general | any | builder | Roark 7e Tables 11.2 (cases 1L, 10a, 10b, 16), 11.4 (cases 1b, 8a, 8b), 11.8 case 2; Navier double series for simply supported rectangles | Small-deflection plate stiffness and stress for a shape, edge condition and load. |
| `plate_mp` | general | plastic_moment | analytic | sy t^2 / 4 | Plastic moment per unit width of a plate: m_p = sy t^2 / 4. |
| `laminate_plate_point_load` | laminate_rect_ss | center_patch_fpf | builder | CLT laminate (Tsai-Wu FPF) + Navier SS plate stiffness + Roark 11.4 centre moment with r0 | Rectangular laminate plate, simply supported, square patch (side `patch`) at the centre. |
| `plate_root_hinge` | strip_root | plastic_hinge | builder | Roark 7e Table 8.1 case 1a (k = 3EI/L as a root rotation) + fibre-integrated M(kappa) plastic hinge | Moment-rotation spring (k, acc_tab, F_tab) for a plate strip of the given width and thickness t that bends at |

## structures / rotating

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `spin_stress` | general | spin | analytic | Timoshenko & Goodier sec. 32; Roark 7e ch. 16; hoop and radial equilibrium | Centrifugal stress in a spinning body, and the burst speed when the strength su is known. |

## structures / sections

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `circle` | circle | properties | builder | Roark 7e Table A.1 case 15, Table 10.1 case 1 | Solid round of diameter d. |
| `hinge_table` | general | moment_rotation | builder | lumped plastic hinge, theta = kappa * L_hinge | Convert M(kappa) to a rotational-spring M(theta) table over hinge length L_hinge |
| `plastic_moment` | general | plastic | analytic | Mp = sigma_y Z | Fully plastic moment Mp = sy * Z (per unit width for plate_per_width). |
| `shape_factor` | general | shape_factor | analytic | Z / S | Plastic reserve Z/S (1.5 rectangle, 1.70 circle, ~1.1-1.2 I-beam). |
| `given_section` | given | properties | builder | user-supplied section properties (CAD, catalog or hand calc) | Any section from its numbers, when it is not a circle or rectangle. Only I is required. |
| `i_beam` | i_beam | properties | builder | Roark 7e Table A.1; torsion open thin-wall sum b t^3/3 | Symmetric I: flange width bf, flange thickness tf, web clear height hw, web thickness tw. |
| `plate_per_width` | plate_strip | properties | builder | Roark 7e Table A.1 case 1 per unit width | Unit-width strip of plate thickness t (all quantities per metre of width). |
| `moment_curvature_polygon` | polygon | moment_curvature | builder | Euler-Bernoulli fiber integration | M(kappa) table for an arbitrary polygon section (bending about its centroidal x axis). |
| `polygon_section` | polygon | properties | builder | Green's theorem (shoelace) area moments | Arbitrary simple polygon section from vertex list [(x,y),...] (drawing/CAD input). |
| `rect_tube` | rect_tube | properties | builder | Roark 7e Table A.1; torsion Bredt thin-wall Table 10.1 case 16 | Rectangular tube, outside b x h, uniform wall t. Torsion by Bredt thin-wall formula. |
| `moment_curvature_rect` | rectangle | moment_curvature | builder | Euler-Bernoulli fiber integration | M(kappa) table for a b x h rectangle from a Curve (simblocks.materials.Curve). |
| `rectangle` | rectangle | properties | builder | Roark 7e Table A.1 case 1; torsion Table 10.1 case 4 | Solid rectangle, width b, depth h (bending direction). Torsion from Roark Table 10.1 case 4. |
| `tube` | tube | properties | builder | Roark 7e Table A.1 case 16, Table 10.1 case 10 | Round tube, outer diameter do, inner di. |

## structures / shells

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `press_fit_pressure` | interference_fit | press_fit | analytic | Shigley 3-56 (general materials) | Interface pressure from RADIAL interference delta_r at nominal radius R. |
| `press_fit_spin_growth` | interference_fit | spin_loss | analytic | Lame + rotating disk, first order | Free radial growth of a spinning hub bore at R (annular disk), to subtract from interference: |
| `press_fit_torque` | interference_fit | torque_capacity | analytic | Coulomb friction on fit area | Slip torque of an interference fit: T = 2 pi mu p R^2 L. |
| `lame_stresses` | thick_cylinder | pressure | analytic | Lame; Roark 7e Table 13.5 case 1 | Thick cylinder radial and hoop stress at r for internal pi and external po pressure. Returns (sr, st). |
| `thin_cylinder_stress` | thin_cylinder | internal_pressure | analytic | Roark 7e Table 13.1 case 1c | Thin cylinder hoop and axial stress (closed ends): (p r/t, p r/2t). |
| `thin_sphere_stress` | thin_sphere | internal_pressure | analytic | Roark 7e Table 13.1 case 3a | Thin sphere membrane stress p r / 2t. |

## structures / stress_concentration

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `kirsch_hoop_stress` | circular_hole | uniaxial_field | analytic | Kirsch 1898 (exact, infinite plate) | Hoop stress around a circular hole of radius a in an infinite plate under far-field uniaxial S (along theta=0). |
| `kt_ellipse` | elliptical_hole | uniaxial | analytic | Inglis 1913 (exact) | Kt = 1 + 2 a/b for an elliptical hole with semi-axis a perpendicular to the load and b parallel. |
| `fatigue_kf` | general | notch_sensitivity | analytic | Shigley 10e eq. 6-33/6-35 (Neuber-Kuhn, steel bending) | Fatigue notch factor Kf = 1 + q (Kt - 1), q = 1/(1 + sqrt(a)/sqrt(r)), sqrt(a) [sqrt(in)] from Sut (kpsi) fit. |

## structures / torsion

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `torsion_stiffness` | any_section | twist | analytic | Roark 7e eq. 10.1-1 | Torsional stiffness k_theta = G J / L (N.m/rad). Use J from a sections builder. |
| `torsion_stress` | any_section | stress | analytic | Roark 7e Table 10.1 | Max shear stress tau = tau_coef * T, using the section's Roark coefficient. |
| `shaft_von_mises` | circle | combined | analytic | von Mises, Shigley 5-15 | Von Mises stress at the surface of a solid round shaft under bending M, torque T, axial N. |
| `torsion_plastic_solid` | circle | plastic | analytic | fully plastic solid shaft | Fully plastic torque of a solid round shaft: T_p = (2/3) pi tau_y r^3 (4/3 of first yield). |
| `key_bearing` | keyed_shaft | bearing | analytic | Shigley 7-8 | Key bearing stress on half key height: sigma = 4T / (d h L). |
| `key_shear` | keyed_shaft | shear | analytic | Shigley 7-8 | Square/rectangular key shear stress tau = 2T / (d w L). |

## structures / vibration

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `bar_wave_speed` | bar | wave_speed | analytic | 1-D elastic wave | Longitudinal bar wave speed sqrt(E/rho). |
| `beam_frequency` | beam | first_mode | analytic | Roark 7e Table 16.1 / Blevins | First bending natural frequency (Hz): f = lambda^2/(2 pi L^2) sqrt(EI/m). |
| `cantilever_tip_mass_frequency` | cantilever | tip_mass | analytic | Rayleigh, 0.2427 m L effective mass | Cantilever with tip mass: f = (1/2pi) sqrt(3EI/L^3 / (M + 0.2427 m L)). |
| `plate_frequency` | plate | first_mode | analytic | Roark 7e Table 16.1 / Blevins (Leissa) | Plate fundamental (Hz): f = lambda^2/(2 pi a^2) sqrt(D/(rho t)); a = radius (circular) or side (square). |
| `drop_impact_deflection` | spring_mass | drop_impact | analytic | Roark 7e eq. 16.3-1 (energy method) | Peak deflection from a weight W dropped height h onto a spring: d_st (1 + sqrt(1 + 2h/d_st)). |
| `moving_mass_impact` | spring_mass | moving_mass_impact | analytic | energy: m v^2/2 = k d^2/2 | Peak deflection and force when mass m at speed v hits a linear spring k: (d, F). |
| `spring_mass_frequency` | spring_mass | frequency | analytic | sqrt(k/m) | Undamped natural frequency (rad/s). |

## thermal / conduction

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `thermal_resistance` | general | resistance | analytic | Incropera 3.6 (plane wall), 3.33 (cylinder), 3.41 (sphere); Newton cooling | Thermal resistance (K/W) of one path in a thermal network. |

## thermal / convection

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `nusselt` | general | nusselt | analytic | Incropera 7.30/7.38 (flat plate), Churchill & Bernstein 1977, Churchill & Chu 1975 (vertical plate, horizontal cylinder), Gnielinski 1976 | Average Nusselt number for a convection case; h = Nu k_fluid / L. |
| `rayleigh` | general | rayleigh | analytic | definition | Rayleigh number g beta dT L^3 / (nu alpha). |

## thermal / deformation

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `plastic_heating` | plastic_work | adiabatic_heating | analytic | Taylor & Quinney 1934 (beta ~ 0.9) | Adiabatic temperature rise from plastic work: dT = beta W / (m cp). |

## thermal / electrical

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `copper_resistance` | winding | resistance_vs_T | kernel | R = R0 (1 + a (T - T0)) | Temperature-dependent winding resistance (alpha_R = 0.00393 /K for copper). |

## thermal / fins

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `fin_efficiency` | straight_fin | efficiency | analytic | Incropera 3.89 (adiabatic tip, corrected length) | Straight rectangular fin efficiency tanh(m Lc)/(m Lc), m = sqrt(2h/(k t)), Lc = L + t/2. |

## thermal / friction

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `friction_heat_partition` | sliding | heat_partition | analytic | Blok partition for semi-infinite bodies | Fraction of frictional heat entering body 1: e1/(e1+e2), effusivity e = sqrt(k rho c). |

## thermal / radiation

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `h_radiation` | surface | linearized_h | analytic | Incropera 1.9 | Linearized radiation coefficient eps sigma (T + Tsur)(T^2 + Tsur^2), W/m^2K. |

## thermal / transient

| block | geometry | load | kind | source | summary |
|---|---|---|---|---|---|
| `biot_number` | lumped | biot | analytic | Incropera 5.10 | Biot number h Lc / k (lumped valid for Bi < 0.1, Lc = V/A). |
| `lumped_temperature` | lumped | analytic | analytic | Incropera 5.6 | Lumped body temperature with convection and constant internal heating q_gen (W). |

## Internal (code only)

Solver, report and text plumbing: callable from hand-written sims, not listed in the editor.

| block | path | summary |
|---|---|---|
| `ecm_constant_power` | actuators/battery/cell/rc_ecm_step | One step of the one-RC equivalent-circuit model under a constant-power load. |
| `bus_voltage` | actuators/battery/pack/bus_voltage | Bus voltage for a constant-power load P on a source V_oc with internal resistance R: |
| `lipo_ocv_k` | actuators/battery/pack/open_circuit_kernel | njit-safe twin of lipo_ocv (same table, nominal 3.75 V, 3.3-4.2 V per cell) for use inside loops. |
| `esc_supply` | actuators/esc/current_limited/supply | Full-throttle ESC between a source (Veff behind R0) and a motor (back-EMF E, winding Rm). |
| `gearbox_output_torque` | actuators/gearbox/general/output_torque | Output torque with efficiency applied in the correct direction: motoring T G eta, back-driven T G / eta. |
| `copper_loss` | actuators/motor/dc/copper_loss | Winding Joule loss I^2 R (W). |
| `motor_current` | actuators/motor/dc/current | Winding current for throttle u in [-1, 1] at motor speed w (rad/s). brake=0: open circuit near |
| `drag_force` | aero/drag/bluff_body/quadratic | Quadratic drag opposing velocity component v: -0.5 rho Cd A v |v|. |
| `dynamic_pressure` | aero/groups/general/dynamic_pressure | q = rho v^2 / 2. |
| `damage_increment` | constitutive/damage/general/accumulation | D += d_eps_p / eps_f; failure when D >= 1. Returns new D. |
| `mooney_rivlin_uniaxial` | constitutive/hyperelastic/elastomer/uniaxial | Nominal uniaxial stress 2 (lam - lam^-2)(C10 + C01/lam). |
| `neo_hookean_uniaxial` | constitutive/hyperelastic/elastomer/uniaxial | Engineering (nominal) uniaxial stress of an incompressible neo-Hookean solid: mu (lam - lam^-2). |
| `interp_clamped` | constitutive/lookup/table/interp | Piecewise-linear lookup with end clamping (ys[0] below, linear extrapolation of the last segment above). |
| `curve_to_spring_table` | constitutive/plasticity/curve/to_spring_table | Convert a Curve's hardening table to a (k, acc_tab, F_tab) for an axial bar of area A, length L. |
| `hinge_to_spring_table` | constitutive/plasticity/hinge/to_spring_table | From structures.sections.hinge_table output to tabular_return inputs (k, acc_tab, M_tab). |
| `lag_return` | constitutive/plasticity/spring_1d/legacy_lag | Original permanent-set update from the legacy hinged-plate sim. x = total deformation measure. |
| `linear_hardening_return` | constitutive/plasticity/spring_1d/linear_hardening | Bilinear elastic-plastic spring, combined isotropic (H_iso) and kinematic (H_kin) hardening. |
| `tabular_return` | constitutive/plasticity/spring_1d/tabular_hardening | Elastic-plastic spring with isotropic hardening from a table F_y(acc). |
| `dashpot_force` | constitutive/viscous/general/dashpot | Linear dashpot F = -c v. |
| `hertz_sphere_force` | contact/hertz/sphere/normal_force | Sphere-on-sphere/flat normal force F = (4/3) E* sqrt(R*) delta^1.5 (0 if delta <= 0). |
| `spinner_strike` | contact/impulse/spinner/strike | One tangential strike: spinning blade (inertia I at tip radius R) hits a target of mass m_target. |
| `hunt_crossley` | contact/penalty/general/normal_force_damped | Hunt-Crossley F = K pen^n (1 + alpha pen_rate), alpha ~ 3(1-e)/(2 v_impact). Zero when not touching. |
| `penalty_kelvin_voigt` | contact/penalty/general/normal_force_damped | Linear spring-dashpot contact, clipped so it never pulls: max(k pen + c pen_rate, 0). |
| `penalty_linear` | contact/penalty/general/normal_force | One-sided linear penalty: F = k * pen if pen > 0 else 0. |
| `dt_convergence` | diagnostics/convergence/general/dt_halving | Run metric(dt) and metric(dt/factor). Returns dict(m1, m2, rel_change, ok). |
| `richardson` | diagnostics/convergence/general/richardson | Extrapolated limit from results at step h and h/factor for a method of given order. |
| `glyph_sheet` | diagnostics/diagram/any_sim/glyph_sheet | Render every glyph on one sheet (reference card for reading diagrams). |
| `EnergyLedger` | diagnostics/energy/general/ledger | Accumulate every source and sink as power*dt, then check closure: |
| `assess_failure_modes` | diagnostics/failure/any_sim/assessment | Evaluate user failure modes on the nominal run and all Monte Carlo samples. Returns dict per mode. |
| `candidate_failure_modes` | diagnostics/failure/any_sim/screening_prompts | Candidate failure statements to PROPOSE to the user (never apply silently). The user picks and sets limits. |
| `write_load_table` | diagnostics/loads/export/table_text | Write 'time  Fx  Fy  Fz' rows (tab separated, SI: s and N), the layout Ansys/MATLAB tabular loads read. |
| `fe_load_table` | diagnostics/loads/force_history/fe_table | Simplify a sampled force history (F: (n,) or (n, k) columns) to a short piecewise-linear table for |
| `pulse_metrics` | diagnostics/loads/force_history/pulse_metrics | Impulse (trapezoidal), peak, time of peak, start/end where |F| first/last exceeds frac x peak, duration, |
| `split_pulses` | diagnostics/loads/force_history/split_pulses | Index ranges [(i0, i1), ...] of separate pulses: |F| above frac x overall peak, gaps shorter than |
| `half_sine_train` | diagnostics/loads/reference/half_sine_train | Time and force of a train of half-sine pulses (peak, duration, start time each). Impulse of one pulse: |
| `check_figure` | diagnostics/report/any_figure/layout_qa | Return a list of human-readable layout problems (empty list = clean). |
| `qa_summary` | diagnostics/report/any_figure/qa_folder | Format {path: [problems]} into a short pass/fail report. |
| `results_report` | diagnostics/report/any_sim/results | Write answer / history / design / verification pages (PNG+SVG), a PDF and a self-contained HTML. |
| `plot_report` | diagnostics/verification/any_sim/report_plot | Convergence, knockout and Monte Carlo figure for a Report (PNG/PDF by extension). |
| `verify_all` | diagnostics/verification/any_sim/full_battery | Run the full verification battery on a SimSpec. Returns a Report (report.ok, .markdown(), .to_json()). |
| `bearing_drag_torque` | friction/bearing/shaft/drag_torque | Bearing friction torque M0 + 0.5 mu F d (SKF simplified), signed against w. |
| `coulomb_regularized_3d` | friction/coulomb/general/regularized_3d | Friction force opposing the tangential slip velocity vt: F = -mu N vt / sqrt(|vt|^2 + v_eps^2). |
| `coulomb_tanh` | friction/coulomb/sliding/regularized | Smooth Coulomb friction F = -mu N tanh(v / v_eps). Pick v_eps << typical sliding speed. |
| `makkar_friction` | friction/coulomb/sliding/smooth_stribeck | g1 (tanh(g2 v) - tanh(g3 v)) + g4 tanh(g5 v) + g6 v. g4 = Coulomb level, g1 + g4 ~ static peak, |
| `simscape_friction` | friction/coulomb/sliding/simscape_translational | sqrt(2e)(F_brk - F_c) exp(-(v/v_St)^2) v/v_St + F_c tanh(v/v_Coul) + f v, |
| `skid_force_2d` | friction/coulomb/sliding/planar_skid | 2-D sliding-contact friction force opposing velocity (vx, vy) with Stribeck coefficient. Returns (Fx, Fy). |
| `stribeck_mu` | friction/coulomb/sliding/stribeck | Stribeck friction coefficient mu_k + (mu_s - mu_k) exp(-(v/v_s)^2). |
| `stribeck_mu_delta` | friction/coulomb/sliding/stribeck_general | Stribeck coefficient mu_k + (mu_s - mu_k) exp(-|v/v_s|^delta). delta = 2 is the LuGre standard. |
| `lugre_step` | friction/lugre/sliding/bristle_step | One LuGre step. F = s0 z + s1 dz/dt + s2 v, dz/dt = v - s0 |v| z / g(v), g = Fc + (Fs-Fc) e^-(v/vs)^2. |
| `rolling_resistance_torque` | friction/rolling/wheel/resistance_torque | Rolling resistance torque Crr N rw tanh(w rw / v_eps). |
| `rolling_torque` | friction/rolling/wheel/resistance_torque_published | Rolling-resistance torque f_R N r tanh(w r / v_eps), opposing spin. Pick v_eps << rolling speed |
| `brush_force_2d` | friction/tire/wheel/brush_force_2d | Planar wheel force from contact-patch velocity (vcx, vcy) and spin w, using brush_mu on the slip |
| `brush_mu` | friction/tire/wheel/brush_two_mu | Brush model with parabolic pressure and separate adhesion (mu_s) and sliding (mu_k) friction. |
| `brush_mu_s_for_peak` | friction/tire/wheel/brush_peak_fit | Brush adhesion limit mu_s that makes the brush curve PEAK at a measured mu_peak. |
| `magic_formula` | friction/tire/wheel/magic_formula | Pacejka pure-slip curve y = D sin(C atan(Bx - E(Bx - atan Bx))). With D = peak mu, y is mu(slip). |
| `magic_formula_fit` | friction/tire/wheel/magic_formula_fit | B, C, D, E from a measured curve: peak mu, fully sliding mu, slip at peak and initial slope dmu/dslip. |
| `relaxation_lag` | friction/tire/wheel/relaxation_lag | First-order lag over DISTANCE: dy/ds = (y_target - y)/lam, ds = |V| dt, integrated exactly. |
| `tire_force_2d` | friction/tire/wheel/combined_slip_force | Planar wheel contact force from contact-patch velocity (vcx, vcy) and wheel spin w. |
| `tire_mu` | friction/tire/wheel/combined_slip_mu | Combined-slip friction curve: linear at small slip, peak ~mu_s at sig_p, decays to mu_k. |
| `obb_gap` | geometry/distance/box_vs_box/separating_axis | Lower bound on the distance between two oriented boxes (centres c, rotation matrices R whose columns are |
| `point_line_signed_distance` | geometry/distance/point_line/signed | Signed distance of P from the line through A with unit normal n (positive on the +n side). |
| `point_segment_closest` | geometry/distance/point_segment/closest | Closest point on segment AB to P and the segment parameter s in [0,1]. Returns (cx, cy, s, dist). |
| `outline_rounded_bar` | geometry/drawing/primitive/outline | Outline of a bar / blade blank L x W centered at the origin, optional corner radius. |
| `first_contact_delta` | geometry/first_contact/head_on/delta | Governing group delta = 120 v / (N rpm D): closing distance per tooth pass over blade radius. |
| `first_contact_stats` | geometry/first_contact/head_on/distribution | alpha_max, mean, median of the first-contact angle (rad before top-dead-center): f = sin(a)/delta. |
| `sample_first_contact` | geometry/first_contact/head_on/sample | Random first-contact angles (rad) by inverse transform: a = arccos(1 - delta U). |
| `tip_on_circle` | geometry/kinematics/circle/tip_position | Position of a tip at radius R and angle phi about (cx, cy). Returns (x, y). |
| `circle_plane_overlap` | geometry/penetration/circle_vs_plane/overlap | Overlap depth of a circle into the half-space behind the plane (A, n) (n points out of the solid). |
| `point_box_penetration` | geometry/penetration/point_vs_box/nearest_face | Penetration of a point (box frame, box centred at origin, half-extents hx, hy, hz). |
| `tip_face_penetration` | geometry/penetration/tip_vs_hinged_face/root_line | Perpendicular penetration of the tip into a face hinged at its root, face at angle a: |
| `adaptive_dt` | integrators/adaptive/contact/step_size | Next step size. gap >= 0 is the smallest open gap, v_close > 0 its closing speed (<= 0 if opening). |
| `stable_step` | integrators/adaptive/contact/stable_step | Stability-based step C * sqrt(m_eff / k) (C ~ 0.1 for accurate penalty contact). |
| `run_fixed` | integrators/driver/general/fixed_step | Python driver: repeatedly calls step(state, t, dt) -> state. Returns (t array, list of states). |
| `dt_for_contact` | integrators/stability/contact/hertz_dt | Step that resolves a contact event of duration t_contact with `samples` steps. |
| `recommend_dt` | integrators/stability/general/recommend | Recommend dt = safety * min(list of critical steps). safety 0.1 keeps penalty contact accurate, |
| `critical_dt_rotational` | integrators/stability/rotational/critical_dt | Critical step for a rotational spring on inertia I. |
| `euler_step` | integrators/step/first_order/explicit_euler | s_{n+1} = s_n + dt f(s_n, t_n). Energy drifts UP for oscillators: use only with tiny dt. |
| `rk4_step` | integrators/step/first_order/rk4 | Classical RK4 step for smooth systems (drive dynamics). Avoid across contact on/off switches. |
| `semi_implicit_step` | integrators/step/second_order/semi_implicit_euler | v_{n+1} = v_n + dt a(x_n, v_n); x_{n+1} = x_n + dt v_{n+1}. Symplectic: bounded energy error. |
| `velocity_verlet_step` | integrators/step/second_order/velocity_verlet | Velocity Verlet (velocity-independent forces exact to 2nd order). Returns (x, v, a_new). |
| `render_laminate_card` | materials/cards/laminate/render | Save a one-page picture (PNG + SVG) of a composite laminate, for reports and sanity checks: the ply stack |
| `hollomon_stress` | materials/constitutive_params/metal/flow_stress | Hollomon power law sigma = K eps^n (eps = true strain; use total or plastic as the source states). |
| `johnson_cook_stress` | materials/constitutive_params/metal/flow_stress | Johnson-Cook flow stress (Pa). rate in 1/s, T in K. Rate term floored at rate0. |
| `ludwik_stress` | materials/constitutive_params/metal/flow_stress | Ludwik law sigma = sigma0 + K eps_p^n. |
| `swift_stress` | materials/constitutive_params/metal/flow_stress | Swift law sigma = K (eps0 + eps_p)^n. |
| `voce_stress` | materials/constitutive_params/metal/flow_stress | Voce saturation law sigma = sigma0 + Q (1 - exp(-b eps_p)). |
| `read_curve_csv` | materials/curves/import/csv | Read a digitized curve CSV. Detects stress units (Pa, MPa, ksi, psi) from header, |
| `resolve_material` | materials/curves/lookup/name | Turn loose text ('yea it's 7075, best common temper', '4130 at 45 HRC') into a catalog key. |
| `curve_from_fit` | materials/curves/metal/uniaxial | Build an engineering Curve from one published-fit record. The record's citation, URL, |
| `eng_to_true` | materials/curves/uniaxial/conversion | Engineering -> true strain/stress: eps = ln(1+e), sigma = s(1+e). |
| `extend_hollomon` | materials/curves/uniaxial/post_necking | Extend a hardening table past necking with sigma = K eps^n matched in value and slope at the last point. |
| `true_to_plastic` | materials/curves/uniaxial/conversion | True curve -> monotonic hardening table (eps_p, sigma) starting at eps_p = 0, sigma = yield. |
| `lamina_Q` | materials/laminates/ply/plane_stress | Reduced stiffness matrix Q (3x3) of a ply in its material axes. |
| `lamina_Qbar` | materials/laminates/ply/rotation | Transformed reduced stiffness Qbar for a ply at theta (deg). |
| `strength_ratio_max_stress` | materials/laminates/ply/failure | Strength ratio R (load multiplier to failure) by maximum stress. s = [s1, s2, t12]. |
| `strength_ratio_tsai_hill` | materials/laminates/ply/failure | Strength ratio by Tsai-Hill with sign-selected X and Y. |
| `strength_ratio_tsai_wu` | materials/laminates/ply/failure | Strength ratio by Tsai-Wu: solve a R^2 + b R - 1 = 0 for the positive root. |
| `parse_layup` | materials/laminates/stack/notation | Expand laminate code to a ply-angle list. Supports '/', '+-' or '±', repeat counts and s/2s: |
| `I_annulus` | rigid_body/inertia/annulus/z | Annular disk / thick ring about its axis: m (Ri^2 + Ro^2) / 2. |
| `solve_remainder_centroid` | rigid_body/inertia/assembly/solve_remainder | Centroid x of the unlisted remainder mass so the assembly hits a measured total mass and COM. |
| `I_box_tensor` | rigid_body/inertia/box/tensor | Inertia tensor of a solid box (full side lengths) about its centroid, box axes. |
| `I_cylinder_transverse` | rigid_body/inertia/cylinder/diameter | Solid cylinder about a transverse axis through its center: m (3R^2 + L^2) / 12. |
| `I_rect_plate` | rigid_body/inertia/rect_plate/z | Rectangular plate a x b about the normal axis through its center: m (a^2 + b^2) / 12. |
| `I_rod_center` | rigid_body/inertia/rod/center | Slender rod about its center, perpendicular: m L^2 / 12. |
| `I_rod_end` | rigid_body/inertia/rod/end | Slender rod about one end: m L^2 / 3. |
| `I_disk` | rigid_body/inertia/solid_disk/z | Solid disk about its axis: m R^2 / 2. |
| `I_sphere` | rigid_body/inertia/sphere/center | Solid sphere: 2 m R^2 / 5. |
| `I_thin_ring` | rigid_body/inertia/thin_ring/z | Thin hoop about its axis: m R^2. |
| `instant_center` | rigid_body/planar/body/ICR | Instantaneous center of rotation in body coordinates (nan if not rotating). |
| `planar_accel` | rigid_body/planar/body/accelerations | World-frame translational accelerations and yaw acceleration. Returns (ax, ay, alpha). |
| `body_to_world` | rigid_body/planar/frames/body_to_world | Rotate a body-frame vector into the world frame. Returns (vx, vy). |
| `world_to_body` | rigid_body/planar/frames/world_to_body | Rotate a world vector into the body frame at heading psi. Returns (bx, by). |
| `force_moment_about_origin` | rigid_body/planar/point/force_moment | Yaw moment of a force (Fx, Fy) applied at (px, py). |
| `point_velocity` | rigid_body/planar/point/velocity | Body-frame velocity of a point at body coords (px, py) on a body with yaw rate r. Returns (vx, vy). |
| `three_point_normal_loads` | rigid_body/planar/three_point_stance/normal_loads | Two wheels on an axle (track T) plus one skid at L ahead of the axle; COM xc ahead of axle at height h. |
| `axle_load_transfer` | rigid_body/planar/two_axle/longitudinal_transfer | Front and rear axle loads on a grade: Wf = W(c cos th - h sin th)/L - m ax h/L, Wr = W cos th - Wf. |
| `quat_step` | rigid_body/spatial/attitude/integrate | Advance attitude by one step at constant body angular velocity: q <- q (x) exp(w dt / 2), renormalized. |
| `quat_to_matrix` | rigid_body/spatial/attitude/rotation_matrix | Rotation matrix (body to world) of the unit quaternion q = (w, x, y, z). |
| `gyrostat_accel` | rigid_body/spatial/gyrostat/accelerations | Angular accelerations of a body carrying a rotor. J: 3x3 total inertia about the COM (body axes, rotor |
| `gyrostat_energy` | rigid_body/spatial/gyrostat/kinetic_energy | Rotational kinetic energy of body + rotor: 0.5 w^T J w + Is Om (a . w) + 0.5 Is Om^2. |
| `cantilever_end_load` | structures/beams/cantilever/end_point | Cantilever, point load at free end. delta = PL^3/3EI (+ PL/kGA). |
| `cantilever_end_moment` | structures/beams/cantilever/end_moment | Cantilever with end couple: k_theta = EI/L, tip delta = ML^2/2EI. |
| `cantilever_point_at` | structures/beams/cantilever/point_at_a | Cantilever (fixed at x=0), point load at distance a from the fixed end; stiffness at the load. |
| `cantilever_tip_deflection_point_at` | structures/beams/cantilever/tip_deflection_point_at_a | Free-end deflection of a cantilever with load P at a from the fixed end: P a^2 (3L - a) / 6EI. |
| `cantilever_uniform` | structures/beams/cantilever/uniform | Cantilever, uniform load w total W = wL. Returns per unit TOTAL load W. tip delta = WL^3/8EI. |
| `fixed_fixed_center` | structures/beams/fixed_fixed/center_point | Both ends fixed, central point load: delta = PL^3/192EI, Mmax = PL/8 (ends and center), collapse 8Mp/L. |
| `fixed_fixed_uniform` | structures/beams/fixed_fixed/uniform | Both ends fixed, uniform load (per total W): delta WL^3/384EI, M_end = WL/12, collapse 16Mp/L. |
| `fixed_guided_end_load` | structures/beams/fixed_guided/end_point | Fixed at one end, guided (no rotation, free translation) at the other, load at the guided end: |
| `springs_in_parallel` | structures/beams/parallel/springs | Combined stiffness of springs in parallel k = sum k_i. |
| `propped_center` | structures/beams/propped_cantilever/center_point | Fixed at x=0, simply supported at x=L, central load: delta at load = 7PL^3/768EI, |
| `propped_uniform` | structures/beams/propped_cantilever/uniform | Fixed-pinned, uniform load (per total W): max delta WL^3/184.6EI, M_fixed = WL/8, |
| `springs_in_series` | structures/beams/series/springs | Combined stiffness of springs in series 1/k = sum 1/k_i (e.g. blade cantilever + chassis cantilever). |
| `simply_supported_center` | structures/beams/simply_supported/center_point | Simply supported, central point load: delta = PL^3/48EI, Mmax = PL/4, collapse 4Mp/L. |
| `simply_supported_point_at` | structures/beams/simply_supported/point_at_a | Simply supported, load at a: delta_load = P a^2 b^2 / (3 E I L), M = P a b / L, collapse Mp L/(a b). |
| `simply_supported_uniform` | structures/beams/simply_supported/uniform | Simply supported, uniform load (per total W): center delta 5WL^3/384EI, M = WL/8, collapse 8Mp/L. |
| `tab_shear_out_force` | structures/joints/tab/shear_out_depth | Gouge/tab shear-out flow force hardening with penetration depth: F = pi m_p + 2 tau_y t pen. |
| `annular_outer_fixed_inner_load` | structures/plates/annular_clamped/free_inner_edge_load | Annular plate, outer edge (radius a) fixed, load P spread on the free inner edge (radius b). |
| `collapse_point_interior` | structures/plates/any_shape/point_interior | Point load collapse on an interior point of any plate: 2 pi m_p (SS edges) or 4 pi m_p (clamped). |
| `collapse_circular_pressure` | structures/plates/circular/uniform_pressure | Uniform-pressure collapse of a circular plate: 6 m_p/a^2 (SS), 12 m_p/a^2 (clamped). |
| `circ_clamped_center_point` | structures/plates/circular_clamped/center_point | Clamped circular plate, central point load: k = 16 pi D / a^2. |
| `circ_clamped_pressure` | structures/plates/circular_clamped/uniform_pressure | Clamped circular plate, uniform pressure q: w0 = q a^4/64D, edge stress 3qa^2/(4t^2). |
| `circ_clamped_pressure_large` | structures/plates/circular_clamped/large_deflection_pressure | Pressure q for center deflection w0 including membrane stiffening (clamped, edge held): |
| `circ_ss_center_point` | structures/plates/circular_ss/center_point | Simply supported circular plate, central point load: k = 16 pi D (1+nu) / ((3+nu) a^2). |
| `circ_ss_pressure` | structures/plates/circular_ss/uniform_pressure | SS circular plate, uniform pressure: w0 = (5+nu) q a^4 / (64 (1+nu) D), center stress 3(3+nu)qa^2/(8t^2). |
| `collapse_point_free_edge` | structures/plates/free_edge/point_at_edge | Point load at a free edge: half fan, pi m_p. |
| `collapse_rect_pressure` | structures/plates/rectangular/uniform_pressure | Uniform-pressure collapse of a rectangle (long a, short b), optimized yield-line pattern: |
| `rect_clamped_center_point` | structures/plates/rectangular_clamped/center_point | All edges clamped, central point load: w = alpha W b^2 / (E t^3). |
| `rect_clamped_pressure` | structures/plates/rectangular_clamped/uniform_pressure | All edges clamped, uniform pressure: w = alpha q b^4/(E t^3), sigma_edge = beta1 q b^2 / t^2. |
| `rect_ss_center_point` | structures/plates/rectangular_ss/center_point | SS rectangle, central point load: k at load, and center stress per N with Roark r0' correction. |
| `rect_ss_point_at` | structures/plates/rectangular_ss/point_anywhere | SS rectangle (0..a, 0..b), load at (x0, y0), optionally over a u x v patch. Stiffness at the load. |
| `rect_ss_pressure` | structures/plates/rectangular_ss/uniform_pressure | SS rectangle under uniform pressure: center deflection per unit pressure and Roark-form alpha. |
| `collapse_three_edge_strip` | structures/plates/three_edge_clamped/point_free_edge | Fold collapse capacity of a face clamped on three edges, loaded near the free lip: |
| `annular_disk_spin_field` | structures/rotating/annular_disk/spin_field | Radial and hoop stress at radius r in a spinning annular disk (free edges). Returns (sr, st). |
| `annular_disk_spin_stress` | structures/rotating/annular_disk/spin | Spinning annular disk with free bore and rim: peak hoop stress at the bore |
| `bar_spin_root_stress` | structures/rotating/bar/spin | Root tensile stress of a uniform bar of length L spinning about one end (or a hub at r_hub): |
| `bite_depth` | structures/rotating/general/bite | Bite (m): distance closed per tooth pass. |
| `burst_rpm_solid_disk` | structures/rotating/general/burst_speed | RPM at which the solid-disk center stress reaches su (no defects, ductile: an upper bound). |
| `gyro_torque` | structures/rotating/general/gyro | Gyroscopic torque magnitude from turning a spinning rotor at rate Omega. |
| `spin_energy` | structures/rotating/general/energy | Stored rotational kinetic energy. |
| `tip_speed` | structures/rotating/general/tip_speed | Tip speed (m/s) from rpm and radius. |
| `solid_disk_spin_stress` | structures/rotating/solid_disk/spin | Peak stress in a solid spinning disk: center sigma_r = sigma_t = (3+nu)/8 rho w^2 R^2. |
| `thin_ring_spin_stress` | structures/rotating/thin_ring/spin | Thin rotating ring hoop stress sigma = rho w^2 R^2 (= rho v^2). |
| `R_cylinder` | thermal/conduction/cylinder_shell/resistance | Radial conduction resistance ln(ro/ri) / (2 pi k L). |
| `R_plane_wall` | thermal/conduction/plane_wall/resistance | Conduction resistance L / (k A), K/W. |
| `R_sphere` | thermal/conduction/sphere_shell/resistance | Spherical shell resistance (1/ri - 1/ro) / (4 pi k). |
| `nu_cylinder_crossflow` | thermal/convection/cylinder/crossflow | Average Nu for a cylinder in crossflow, all Re*Pr > 0.2. |
| `nu_flat_plate` | thermal/convection/flat_plate/forced_average | Average Nusselt number over a flat plate, laminar or mixed laminar-turbulent. |
| `nu_horizontal_cylinder_natural` | thermal/convection/horizontal_cylinder/natural | Average Nu, natural convection on a horizontal cylinder, Ra < 1e12. |
| `nu_pipe_gnielinski` | thermal/convection/pipe/turbulent | Fully developed turbulent pipe Nu (3000 < Re < 5e6), smooth-pipe Petukhov friction factor. |
| `R_convection` | thermal/convection/surface/resistance | Convection resistance 1 / (h A). |
| `nu_vertical_plate_natural` | thermal/convection/vertical_plate/natural | Average Nu, natural convection on a vertical plate, full Ra range. |
| `link_heat_flow` | thermal/transient/node/link_flow | Heat flow a -> b through resistance R (W). |
| `node_dTdt` | thermal/transient/node/rate | dT/dt for one thermal node with heat input Q_in and a resistance R to ambient. |
