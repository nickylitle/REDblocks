"""Tiny line diagrams for the block library: general geometry plus connection points.
Every figure is an SVG <symbol> (viewBox 0 0 160 90) drawn with currentColor for structure,
var(--acc) for loads / motion, var(--pt) for connection points. Colors come from the page's tokens."""
import math

ACC = 'style="stroke:var(--acc);fill:none"'
ACCF = 'style="fill:var(--acc);stroke:none"'
SOFT = 'style="fill:var(--soft);stroke:currentColor"'


def _arrow(x1, y1, x2, y2):
    a = math.atan2(y2 - y1, x2 - x1); h = 5
    p1 = (x2 - h * math.cos(a - 0.45), y2 - h * math.sin(a - 0.45))
    p2 = (x2 - h * math.cos(a + 0.45), y2 - h * math.sin(a + 0.45))
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" {ACC} stroke-width="1.6"/>'
            f'<polygon points="{x2:.1f},{y2:.1f} {p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}" {ACCF}/>')


def _pt(x, y, lab=""):
    s = f'<circle cx="{x}" cy="{y}" r="3.2" style="fill:var(--pt);stroke:var(--bg)" stroke-width="1"/>'
    if lab:
        s += f'<text x="{x + 5}" y="{y - 5}" class="fl" font-size="7" font-family="ui-monospace,Consolas,monospace" fill="currentColor" style="fill:var(--ink2)">{lab}</text>'
    return s


def _wall(x, y0, y1, side=-1):
    s = f'<line x1="{x}" y1="{y0}" x2="{x}" y2="{y1}" stroke="currentColor" stroke-width="1.6"/>'
    for y in range(int(y0), int(y1), 6):
        s += f'<line x1="{x}" y1="{y + 6}" x2="{x + 6 * side}" y2="{y}" stroke="currentColor" stroke-width="0.8"/>'
    return s


def _ground(x0, x1, y):
    s = f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="currentColor" stroke-width="1.6"/>'
    for x in range(int(x0), int(x1), 7):
        s += f'<line x1="{x}" y1="{y + 6}" x2="{x + 6}" y2="{y}" stroke="currentColor" stroke-width="0.8"/>'
    return s


def _pin(x, y):
    return (f'<polygon points="{x},{y} {x - 6},{y + 9} {x + 6},{y + 9}" fill="none" stroke="currentColor" '
            f'stroke-width="1.3"/>' + _ground(x - 9, x + 9, y + 9))


def _beam(x0, x1, y, h=7):
    return f'<rect x="{x0}" y="{y - h / 2}" width="{x1 - x0}" height="{h}" {SOFT} stroke-width="1.3"/>'


def _box(x, y, w, h, lab=""):
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="2" {SOFT} stroke-width="1.3"/>'
    if lab:
        s += f'<text x="{x + w / 2}" y="{y + h / 2 + 3.5}" class="fc" font-size="8" font-family="ui-monospace,Consolas,monospace" text-anchor="middle" fill="currentColor" style="fill:var(--ink)">{lab}</text>'
    return s


def _spring(x0, x1, y, n=6, a=5):
    pts = [(x0, y), (x0 + 6, y)]
    L = x1 - x0 - 12
    for i in range(n):
        pts.append((x0 + 6 + L * (i + 0.5) / n, y + (a if i % 2 else -a)))
    pts += [(x1 - 6, y), (x1, y)]
    return ('<polyline points="' + " ".join(f"{p:.1f},{q:.1f}" for p, q in pts) +
            '" fill="none" stroke="currentColor" stroke-width="1.4"/>')


_NOFILL = 'fill="none" stroke="currentColor"'


def _circle(cx, cy, r, soft=True):
    style = SOFT if soft else _NOFILL
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" {style} stroke-width="1.3"/>'


def _spin(cx, cy, r):
    a0, a1 = -0.3, 1.2
    x0, y0 = cx + r * math.cos(a0), cy - r * math.sin(a0)
    x1, y1 = cx + r * math.cos(a1), cy - r * math.sin(a1)
    s = f'<path d="M{x0:.1f},{y0:.1f} A{r},{r} 0 0 0 {x1:.1f},{y1:.1f}" {ACC} stroke-width="1.6"/>'
    s += _arrow(x1 + 4, y1 + 1, x1 - 0.5, y1 - 0.5)
    return s


_ATTR = {"fl": 'font-size="7" font-family="ui-monospace,Consolas,monospace" fill="currentColor" style="fill:var(--ink2)"', "fc": 'font-size="8" font-family="ui-monospace,Consolas,monospace" text-anchor="middle" fill="currentColor" style="fill:var(--ink)"'}


def _text(x, y, t, cls="fl"):
    return f'<text x="{x}" y="{y}" class="{cls}" {_ATTR.get(cls, "")}>{t}</text>'


def _curve(x0, y0, w, h):
    return (f'<path d="M{x0},{y0} L{x0 + w * 0.25},{y0 - h * 0.7} Q{x0 + w * 0.5},{y0 - h * 0.95} {x0 + w},{y0 - h}" '
            f'fill="none" style="stroke:var(--acc)" stroke-width="1.6"/>'
            f'<line x1="{x0}" y1="{y0}" x2="{x0 + w + 6}" y2="{y0}" stroke="currentColor"/>'
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 - h - 6}" stroke="currentColor"/>')


F = {}


def fig(key):
    def wrap(fn):
        F[key] = fn
        return fn
    return wrap


# ------------------------------------------------------------------ beams
@fig("cantilever")
def _():
    return _wall(22, 25, 65) + _beam(22, 140, 45) + _arrow(138, 18, 138, 40) + _pt(22, 45, "root") + _pt(138, 45, "load")


@fig("cantilever_moment")
def _():
    return (_wall(22, 25, 65) + _beam(22, 140, 45) + _pt(138, 45, "M") +
            '<path d="M130,30 A14,14 0 1 1 150,52" fill="none" style="stroke:var(--acc)" stroke-width="1.6"/>' +
            _arrow(152, 46, 150, 53))


@fig("cantilever_uniform")
def _():
    s = _wall(22, 25, 65) + _beam(22, 140, 45)
    for x in range(30, 141, 14):
        s += _arrow(x, 22, x, 39)
    return s + _pt(22, 45, "root")


@fig("simply_supported")
def _():
    return _beam(20, 140, 40) + _pin(25, 44) + _pin(135, 44) + _arrow(80, 12, 80, 35) + _pt(80, 40, "load")


@fig("ss_uniform")
def _():
    s = _beam(20, 140, 40) + _pin(25, 44) + _pin(135, 44)
    for x in range(28, 136, 13):
        s += _arrow(x, 18, x, 35)
    return s


@fig("fixed_fixed")
def _():
    return _wall(20, 22, 62) + _wall(140, 22, 62, 1) + _beam(20, 140, 42) + _arrow(80, 12, 80, 37) + _pt(80, 42, "load")


@fig("fixed_guided")
def _():
    return (_wall(20, 22, 62) + _beam(20, 130, 42) + f'<rect x="130" y="30" width="8" height="24" fill="none" '
            f'stroke="currentColor"/>' + _wall(142, 22, 62, 1) + _arrow(125, 12, 125, 37) + _pt(125, 42, "load"))


@fig("propped")
def _():
    return _wall(20, 22, 62) + _beam(20, 140, 40) + _pin(135, 44) + _arrow(80, 12, 80, 35) + _pt(80, 40, "load")


@fig("membrane")
def _():
    return (_wall(18, 18, 60) + _wall(142, 18, 60, 1) +
            '<polyline points="18,36 80,58 142,36" fill="none" stroke="currentColor" stroke-width="5" '
            'style="stroke:var(--soft)"/><polyline points="18,36 80,58 142,36" fill="none" stroke="currentColor" '
            'stroke-width="1.2"/>' + _arrow(80, 20, 80, 52) + _pt(80, 58, "W") + _text(98, 78, "string action"))


@fig("springs_series")
def _():
    return _wall(18, 30, 60) + _spring(18, 75, 45) + _spring(75, 132, 45) + _pt(75, 45) + _arrow(132, 45, 152, 45)


@fig("springs_parallel")
def _():
    return (_wall(18, 22, 68) + _spring(18, 120, 32) + _spring(18, 120, 58) +
            '<line x1="120" y1="28" x2="120" y2="62" stroke="currentColor" stroke-width="2"/>' + _arrow(120, 45, 145, 45))


@fig("dispersion")
def _():
    return (_beam(10, 150, 60, 10) + '<polygon points="70,30 90,30 120,55 40,55" style="fill:var(--soft)" '
            'stroke="none"/>' + _arrow(80, 10, 80, 28) + _text(55, 80, "load spreads at 45 deg"))


# ------------------------------------------------------------------ sections
@fig("sec_circle")
def _():
    return _circle(80, 45, 26) + _text(112, 50, "d") + '<line x1="54" y1="45" x2="106" y2="45" stroke="currentColor" stroke-dasharray="3 2"/>'


@fig("sec_tube")
def _():
    return _circle(80, 45, 28) + f'<circle cx="80" cy="45" r="18" style="fill:var(--bg)" stroke="currentColor"/>'


@fig("sec_rect")
def _():
    return _box(55, 20, 50, 50) + _text(78, 82, "b") + _text(110, 48, "h")


@fig("sec_rect_tube")
def _():
    return _box(50, 18, 60, 54) + '<rect x="60" y="28" width="40" height="34" style="fill:var(--bg)" stroke="currentColor"/>'


@fig("sec_ibeam")
def _():
    return ('<path d="M45,18 h70 v9 h-30 v36 h30 v9 h-70 v-9 h30 v-36 h-30 z" style="fill:var(--soft)" '
            'stroke="currentColor" stroke-width="1.3"/>')


@fig("sec_polygon")
def _():
    return '<polygon points="50,70 60,22 100,18 118,50 96,72" style="fill:var(--soft)" stroke="currentColor" stroke-width="1.3"/>' + _pt(82, 46, "centroid")


@fig("sec_strip")
def _():
    return _box(20, 38, 120, 12) + _text(30, 66, "1 m wide strip, thickness t")


@fig("sec_general")
def _():
    return _box(50, 22, 60, 46) + '<line x1="40" y1="45" x2="120" y2="45" stroke="currentColor" stroke-dasharray="4 3"/>' + _text(122, 48, "NA")


@fig("moment_curvature")
def _():
    return _curve(30, 72, 110, 50) + _text(112, 84, "curvature") + _text(34, 16, "M")


# ------------------------------------------------------------------ buckling / curved / fatigue
@fig("column")
def _():
    return (_ground(55, 105, 78) + '<path d="M80,78 Q92,46 80,14" fill="none" stroke="currentColor" stroke-width="3"/>' +
            _arrow(80, 2, 80, 12) + _pt(80, 14, "P"))


@fig("plate_buckling")
def _():
    return (_box(45, 20, 70, 50) + '<path d="M50,45 Q80,30 110,45" fill="none" style="stroke:var(--acc)" stroke-dasharray="3 2"/>' +
            _arrow(20, 45, 43, 45) + _arrow(140, 45, 117, 45))


@fig("cyl_buckling")
def _():
    return ('<ellipse cx="80" cy="20" rx="26" ry="7" fill="none" stroke="currentColor"/>'
            '<path d="M54,20 v50 M106,20 v50" stroke="currentColor" fill="none"/>'
            '<ellipse cx="80" cy="70" rx="26" ry="7" style="fill:var(--soft)" stroke="currentColor"/>' + _arrow(80, 2, 80, 12))


@fig("ring")
def _():
    return (f'<circle cx="80" cy="45" r="26" fill="none" stroke="currentColor" stroke-width="3"/>' +
            _arrow(80, 2, 80, 17) + _arrow(80, 88, 80, 73) + _pt(80, 19, "P") + _pt(80, 71, "P"))


@fig("curved_bar")
def _():
    return ('<path d="M30,75 A55,55 0 0 1 130,75" fill="none" stroke="currentColor" stroke-width="7" '
            'style="stroke:var(--soft)"/><path d="M30,75 A55,55 0 0 1 130,75" fill="none" stroke="currentColor"/>' +
            _arrow(18, 60, 30, 72) + _arrow(142, 60, 130, 72))


@fig("sn_curve")
def _():
    return ('<path d="M24,20 Q60,30 90,55 L145,55" fill="none" style="stroke:var(--acc)" stroke-width="1.6"/>'
            '<line x1="22" y1="72" x2="150" y2="72" stroke="currentColor"/><line x1="22" y1="72" x2="22" y2="12" '
            'stroke="currentColor"/>' + _text(96, 84, "cycles (log)") + _text(26, 12, "S"))


# ------------------------------------------------------------------ plates
@fig("circ_plate_clamped")
def _():
    return (_wall(30, 30, 60) + _wall(130, 30, 60, 1) + _beam(30, 130, 45, 6) + _arrow(80, 12, 80, 40) +
            _pt(80, 45, "load") + _text(56, 78, "clamped edge, radius a"))


@fig("circ_plate_ss")
def _():
    return _beam(28, 132, 42, 6) + _pin(34, 45) + _pin(126, 45) + _arrow(80, 12, 80, 38) + _pt(80, 42, "load") + _text(56, 80, "simple support, radius a")


@fig("circ_pressure")
def _():
    s = _wall(30, 30, 60) + _wall(130, 30, 60, 1) + _beam(30, 130, 45, 6)
    for x in range(38, 126, 12):
        s += _arrow(x, 20, x, 40)
    return s + _text(70, 78, "pressure q")


@fig("annular")
def _():
    return (_wall(20, 30, 60) + _wall(140, 30, 60, 1) + _beam(20, 66, 45, 6) + _beam(94, 140, 45, 6) +
            _arrow(66, 18, 66, 40) + _arrow(94, 18, 94, 40) + _text(58, 80, "load on inner edge"))


@fig("rect_plate_ss")
def _():
    return ('<polygon points="30,60 100,60 130,30 60,30" style="fill:var(--soft)" stroke="currentColor" stroke-width="1.3"/>' +
            _arrow(80, 4, 80, 42) + _pt(80, 45, "load") + _text(26, 78, "simply supported a x b"))


@fig("rect_plate_clamped")
def _():
    return ('<polygon points="30,60 100,60 130,30 60,30" style="fill:var(--soft)" stroke="currentColor" stroke-width="2.6"/>' +
            _arrow(80, 4, 80, 42) + _pt(80, 45, "load") + _text(40, 78, "clamped a x b"))


@fig("plate_free_edge")
def _():
    return ('<polygon points="30,60 100,60 130,30 60,30" style="fill:var(--soft)" stroke="currentColor" stroke-width="1.3"/>' +
            _arrow(65, 26, 65, 58) + _pt(65, 60, "edge load") + _text(98, 78, "free edge"))


@fig("plate_three_edge")
def _():
    return ('<polyline points="100,60 30,60 60,30 130,30" fill="none" stroke="currentColor" stroke-width="3"/>'
            '<line x1="130" y1="30" x2="100" y2="60" stroke="currentColor" stroke-dasharray="3 2"/>' +
            _arrow(115, 14, 115, 43) + _pt(115, 45, "lip") + _text(20, 80, "3 edges clamped, 1 free"))


@fig("plate_point")
def _():
    return '<polygon points="30,60 100,60 130,30 60,30" style="fill:var(--soft)" stroke="currentColor"/>' + _arrow(80, 4, 80, 42) + _pt(80, 45, "point") + _text(30, 80, "any outline: collapse load")


@fig("plate_general")
def _():
    return _beam(20, 140, 45, 8) + _text(60, 30, "plate, thickness t") + _text(40, 70, "D, m_p, frequency")


@fig("punch")
def _():
    return (_beam(20, 140, 55, 7) + '<rect x="70" y="20" width="20" height="30" style="fill:var(--soft)" stroke="currentColor"/>' +
            _arrow(80, 4, 80, 18) + '<line x1="70" y1="51" x2="70" y2="59" style="stroke:var(--acc)" stroke-width="2"/>'
            '<line x1="90" y1="51" x2="90" y2="59" style="stroke:var(--acc)" stroke-width="2"/>' + _text(30, 80, "plug shears on the perimeter"))


# ------------------------------------------------------------------ joints / shells / rotating / torsion
@fig("bolt")
def _():
    return (_box(30, 24, 100, 14) + _box(30, 46, 100, 14) + '<rect x="72" y="14" width="16" height="56" fill="none" stroke="currentColor" stroke-width="1.6"/>'
            '<rect x="66" y="10" width="28" height="6" style="fill:var(--soft)" stroke="currentColor"/>' +
            _arrow(20, 31, 40, 31) + _arrow(140, 53, 120, 53) + _pt(80, 42, "shear plane"))


@fig("plate_hole")
def _():
    return (_box(20, 30, 120, 30) + '<circle cx="110" cy="45" r="7" style="fill:var(--bg)" stroke="currentColor"/>' +
            _arrow(110, 45, 140, 45) + _pt(110, 45, "pin") + _text(110, 78, "e") + _text(30, 22, "bearing / tear-out / net"))


@fig("thread")
def _():
    s = '<rect x="60" y="10" width="40" height="70" style="fill:var(--soft)" stroke="currentColor"/>'
    for y in range(16, 76, 8):
        s += f'<polyline points="60,{y} 52,{y + 4} 60,{y + 8}" fill="none" stroke="currentColor"/><polyline points="100,{y} 108,{y + 4} 100,{y + 8}" fill="none" stroke="currentColor"/>'
    return s + _arrow(80, 88, 80, 82)


@fig("tab")
def _():
    return _box(20, 40, 120, 14) + _arrow(130, 14, 130, 36) + _pt(130, 40, "tooth") + _text(30, 74, "tab shears out with depth")


@fig("cylinder_pressure")
def _():
    return ('<ellipse cx="40" cy="45" rx="8" ry="22" fill="none" stroke="currentColor"/>'
            '<path d="M40,23 H120 M40,67 H120" stroke="currentColor" fill="none" stroke-width="1.4"/>'
            '<ellipse cx="120" cy="45" rx="8" ry="22" style="fill:var(--soft)" stroke="currentColor"/>' +
            _arrow(80, 45, 80, 26) + _arrow(80, 45, 80, 64) + _text(70, 84, "internal p"))


@fig("sphere_pressure")
def _():
    return _circle(80, 45, 28) + _arrow(80, 45, 80, 20) + _arrow(80, 45, 104, 45) + _arrow(80, 45, 60, 63)


@fig("interference")
def _():
    return _circle(80, 45, 32) + f'<circle cx="80" cy="45" r="18" style="fill:var(--bg)" stroke="currentColor"/>' + _circle(80, 45, 19) + _text(104, 10, "hub on shaft")


@fig("spinning_disk")
def _():
    return _circle(80, 45, 30) + '<circle cx="80" cy="45" r="3" fill="currentColor"/>' + _spin(80, 45, 38)


@fig("spinning_annulus")
def _():
    return _circle(80, 45, 30) + f'<circle cx="80" cy="45" r="12" style="fill:var(--bg)" stroke="currentColor"/>' + _spin(80, 45, 38)


@fig("spinning_ring")
def _():
    return '<circle cx="80" cy="45" r="28" fill="none" stroke="currentColor" stroke-width="4"/>' + _spin(80, 45, 38)


@fig("spinning_bar")
def _():
    return '<rect x="30" y="40" width="100" height="10" rx="2" style="fill:var(--soft)" stroke="currentColor"/>' + '<circle cx="80" cy="45" r="3" fill="currentColor"/>' + _spin(80, 45, 34)


@fig("torsion")
def _():
    return (_wall(20, 28, 62) + '<rect x="20" y="37" width="110" height="16" style="fill:var(--soft)" stroke="currentColor"/>' +
            '<path d="M132,28 A18,18 0 0 1 132,62" fill="none" style="stroke:var(--acc)" stroke-width="1.6"/>' +
            _arrow(134, 60, 130, 63) + _pt(130, 45, "T"))


@fig("keyed")
def _():
    return _circle(80, 45, 28) + '<rect x="73" y="13" width="14" height="12" style="fill:var(--acc)" stroke="none"/>' + _text(96, 16, "key")


@fig("spring_mass")
def _():
    return _wall(18, 25, 65) + _spring(18, 92, 45) + _box(92, 30, 40, 30, "m") + _arrow(134, 45, 152, 45)


@fig("beam_vibe")
def _():
    return (_wall(18, 25, 65) + '<path d="M18,45 Q80,30 140,52" fill="none" stroke="currentColor" stroke-width="3"/>'
            '<path d="M18,45 Q80,60 140,38" fill="none" style="stroke:var(--acc)" stroke-dasharray="3 2"/>')


# ------------------------------------------------------------------ contact
@fig("sphere_flat")
def _():
    return _ground(20, 140, 62) + _circle(80, 38, 22) + _arrow(80, 4, 80, 14) + _pt(80, 60, "contact") + _text(106, 30, "R")


@fig("cylinder_flat")
def _():
    return (_ground(20, 140, 62) + '<ellipse cx="80" cy="38" rx="40" ry="22" style="fill:var(--soft)" stroke="currentColor"/>' +
            _pt(80, 60, "line contact"))


@fig("flat_punch")
def _():
    return _ground(20, 140, 62) + '<rect x="62" y="22" width="36" height="40" style="fill:var(--soft)" stroke="currentColor"/>' + _arrow(80, 4, 80, 20) + _text(104, 50, "radius a")


@fig("impulse")
def _():
    return _box(20, 30, 40, 30, "m1") + _box(100, 30, 40, 30, "m2") + _arrow(62, 45, 92, 45) + _pt(100, 45, "hit")


@fig("spinner_hit")
def _():
    return ('<rect x="15" y="40" width="80" height="9" rx="2" style="fill:var(--soft)" stroke="currentColor"/>'
            '<circle cx="55" cy="45" r="3" fill="currentColor"/>' + _spin(55, 45, 30) + _box(108, 30, 36, 30, "target") +
            _pt(95, 45, "tooth"))


@fig("indent")
def _():
    return _ground(20, 140, 55) + '<path d="M66,8 L94,8 L80,52 Z" style="fill:var(--soft)" stroke="currentColor"/>' + _arrow(80, 0, 80, 6) + _pt(80, 52, "tip")


@fig("plough")
def _():
    return (_ground(10, 150, 58) + '<path d="M60,20 L100,20 L100,56 Z" style="fill:var(--soft)" stroke="currentColor"/>' +
            _arrow(110, 40, 140, 40) + _text(28, 80, "hard edge ploughs the surface"))


@fig("penalty")
def _():
    return _box(15, 30, 40, 30, "A") + _box(105, 30, 40, 30, "B") + _spring(55, 105, 45) + _text(62, 76, "acts only when touching")


@fig("hinged_face")
def _():
    return (_ground(10, 150, 72) + '<line x1="120" y1="72" x2="60" y2="30" stroke="currentColor" stroke-width="4"/>' +
            _pt(120, 72) + _text(84, 84, "root hinge") + _pt(60, 30, "lip") + _arrow(20, 30, 54, 30) + _text(18, 22, "tooth"))


@fig("circle_plane")
def _():
    return _ground(20, 140, 60) + _circle(80, 42, 22) + '<line x1="58" y1="60" x2="102" y2="60" style="stroke:var(--acc)" stroke-width="3"/>' + _text(108, 30, "overlap")


@fig("first_contact")
def _():
    return ('<circle cx="60" cy="45" r="32" fill="none" stroke="currentColor" stroke-dasharray="4 3"/>' + _spin(60, 45, 40) +
            '<line x1="118" y1="10" x2="104" y2="80" stroke="currentColor" stroke-width="3"/>' + _pt(90, 30, "alpha") +
            _arrow(140, 45, 120, 45))


@fig("distance")
def _():
    return '<line x1="20" y1="65" x2="140" y2="30" stroke="currentColor" stroke-width="1.6"/>' + _pt(70, 20) + '<line x1="70" y1="20" x2="78" y2="50" style="stroke:var(--acc)" stroke-dasharray="3 2"/>'


@fig("drawing")
def _():
    return ('<path d="M30,65 L30,30 Q40,15 70,18 L130,40 L120,68 Z" fill="none" stroke="currentColor" stroke-width="1.4" '
            'stroke-dasharray="5 3"/>' + _pt(30, 65) + _pt(30, 30) + _pt(130, 40) + _pt(120, 68) + _text(54, 84, "traced outline"))


# ------------------------------------------------------------------ friction
@fig("sliding")
def _():
    return _ground(15, 145, 62) + _box(55, 34, 50, 28, "N") + _arrow(106, 48, 140, 48) + _arrow(54, 58, 30, 58) + _text(18, 80, "friction")


@fig("bristles")
def _():
    s = _ground(15, 145, 66) + _box(50, 30, 60, 22)
    for x in range(56, 108, 8):
        s += f'<line x1="{x}" y1="52" x2="{x + 5}" y2="66" stroke="currentColor"/>'
    return s + _arrow(112, 40, 140, 40)


@fig("wheel")
def _():
    return _ground(15, 145, 72) + _circle(80, 44, 27) + '<circle cx="80" cy="44" r="4" fill="currentColor"/>' + _spin(80, 44, 34) + _pt(80, 71, "patch")


@fig("bearing")
def _():
    s = _circle(80, 45, 32) + f'<circle cx="80" cy="45" r="16" style="fill:var(--bg)" stroke="currentColor"/>'
    for k in range(8):
        a = k * math.pi / 4
        s += f'<circle cx="{80 + 24 * math.cos(a):.1f}" cy="{45 + 24 * math.sin(a):.1f}" r="4" fill="none" stroke="currentColor"/>'
    return s + _spin(80, 45, 12)


# ------------------------------------------------------------------ actuators
@fig("motor")
def _():
    return (_box(30, 25, 60, 40, "motor") + '<rect x="90" y="40" width="40" height="10" style="fill:var(--soft)" stroke="currentColor"/>' +
            _spin(130, 45, 12) + _pt(130, 45, "shaft"))


@fig("battery")
def _():
    return (_box(35, 25, 80, 40, "pack") + '<rect x="115" y="37" width="8" height="16" style="fill:var(--soft)" stroke="currentColor"/>' +
            _pt(123, 45, "+") + _text(40, 80, "OCV, R0, R1 || C1"))


@fig("esc")
def _():
    return _box(10, 30, 36, 30, "batt") + _box(62, 30, 36, 30, "ESC") + _box(114, 30, 36, 30, "motor") + _arrow(46, 45, 60, 45) + _arrow(98, 45, 112, 45)


@fig("gearbox")
def _():
    return (_circle(55, 45, 24) + _circle(100, 45, 12) + _spin(55, 45, 30) + _pt(100, 45, "out") + _text(40, 84, "ratio G"))


# ------------------------------------------------------------------ aero
@fig("bluff_drag")
def _():
    return _circle(95, 45, 20) + _arrow(20, 30, 70, 30) + _arrow(20, 45, 70, 45) + _arrow(20, 60, 70, 60) + _arrow(118, 45, 145, 45)


@fig("airfoil")
def _():
    return ('<path d="M40,50 Q70,26 130,46 Q80,54 40,50 Z" style="fill:var(--soft)" stroke="currentColor"/>' +
            _arrow(10, 56, 36, 52) + _arrow(85, 40, 85, 12) + _text(90, 14, "lift"))


@fig("propeller")
def _():
    return ('<ellipse cx="80" cy="25" rx="7" ry="20" style="fill:var(--soft)" stroke="currentColor"/>'
            '<ellipse cx="80" cy="65" rx="7" ry="20" style="fill:var(--soft)" stroke="currentColor"/>' + _arrow(90, 45, 140, 45) + _text(108, 38, "thrust"))


@fig("atmosphere")
def _():
    return (_ground(20, 140, 78) + '<path d="M30,70 Q70,40 130,14" fill="none" style="stroke:var(--acc)" stroke-width="1.6"/>' +
            _text(30, 20, "rho, p, T vs altitude"))


@fig("groups")
def _():
    return _text(46, 50, "Re  q  Ma", "fc")


# ------------------------------------------------------------------ rigid body
@fig("inertia_axis")
def _():
    return _circle(80, 45, 28) + '<line x1="80" y1="5" x2="80" y2="85" style="stroke:var(--acc)" stroke-dasharray="4 3"/>' + _text(86, 12, "axis")


@fig("inertia_rod")
def _():
    return '<rect x="25" y="40" width="110" height="10" style="fill:var(--soft)" stroke="currentColor"/>' + '<line x1="80" y1="10" x2="80" y2="80" style="stroke:var(--acc)" stroke-dasharray="4 3"/>'


@fig("inertia_plate")
def _():
    return '<polygon points="30,60 100,60 130,30 60,30" style="fill:var(--soft)" stroke="currentColor"/>' + '<line x1="80" y1="5" x2="80" y2="85" style="stroke:var(--acc)" stroke-dasharray="4 3"/>'


@fig("assembly")
def _():
    return _box(20, 40, 40, 26) + _box(70, 20, 30, 30) + _box(108, 46, 32, 22) + _pt(78, 46, "COM")


@fig("parallel_axis")
def _():
    return (_circle(60, 45, 22) + '<line x1="60" y1="10" x2="60" y2="80" stroke="currentColor" stroke-dasharray="4 3"/>'
            '<line x1="120" y1="10" x2="120" y2="80" style="stroke:var(--acc)" stroke-dasharray="4 3"/>' + _arrow(62, 70, 118, 70) + _text(86, 84, "d"))


@fig("frames")
def _():
    return (_arrow(30, 70, 70, 70) + _arrow(30, 70, 30, 30) + _text(72, 74, "X") + _text(20, 28, "Y") +
            '<g transform="rotate(-30 105 50)">' + _arrow(105, 50, 140, 50) + _arrow(105, 50, 105, 15) + '</g>' + _text(112, 80, "body"))


@fig("bot_stance")
def _():
    return (_ground(10, 150, 74) + _box(30, 36, 100, 22) + _circle(46, 62, 12) + '<path d="M128,58 L140,74" stroke="currentColor" stroke-width="2"/>' +
            _pt(46, 74, "wheels") + _pt(140, 74) + _text(126, 86, "skid") + _pt(80, 47, "COM"))


# ------------------------------------------------------------------ thermal
@fig("wall_cond")
def _():
    return '<rect x="60" y="15" width="40" height="60" style="fill:var(--soft)" stroke="currentColor"/>' + _arrow(25, 45, 140, 45) + _text(20, 30, "T1") + _text(124, 30, "T2")


@fig("cyl_cond")
def _():
    return _circle(80, 45, 30) + f'<circle cx="80" cy="45" r="14" style="fill:var(--bg)" stroke="currentColor"/>' + _arrow(94, 45, 140, 45)


@fig("convection")
def _():
    return _box(20, 55, 120, 12) + '<path d="M30,45 Q50,30 70,45 T110,45 T150,40" fill="none" style="stroke:var(--acc)"/>' + _text(30, 22, "h, flow over surface")


@fig("fin")
def _():
    return _wall(30, 20, 70) + '<rect x="30" y="38" width="100" height="14" style="fill:var(--soft)" stroke="currentColor"/>' + _arrow(80, 36, 80, 14) + _arrow(80, 54, 80, 76)


@fig("radiation")
def _():
    return _box(20, 30, 40, 30) + '<path d="M66,45 q6,-8 12,0 t12,0 t12,0 t12,0" fill="none" style="stroke:var(--acc)"/>' + _arrow(112, 45, 140, 45)


@fig("lumped")
def _():
    return _box(55, 28, 50, 34, "m cp T") + _arrow(30, 45, 53, 45) + _arrow(107, 45, 135, 45) + _text(20, 80, "heat in / heat out")


@fig("heat_source")
def _():
    return _box(50, 30, 60, 30) + '<path d="M80,24 q-8,-10 0,-18 q8,8 0,18" style="fill:var(--acc)"/>' + _text(26, 80, "heat from work or current")


# ------------------------------------------------------------------ generic kinds
@fig("stress_strain")
def _():
    return _curve(28, 74, 112, 54) + _text(116, 86, "strain") + _text(32, 16, "stress")


@fig("laminate")
def _():
    s = ""
    for i, a in enumerate((0, 45, -45, 90)):
        y = 22 + i * 13
        s += f'<rect x="30" y="{y}" width="100" height="11" style="fill:var(--soft)" stroke="currentColor"/>' + _text(136, y + 9, f"{a}")
    return s


@fig("micromech")
def _():
    s = '<rect x="35" y="15" width="90" height="60" style="fill:var(--soft)" stroke="currentColor"/>'
    for x in range(45, 125, 16):
        for y in range(25, 75, 16):
            s += f'<circle cx="{x}" cy="{y}" r="5" fill="none" stroke="currentColor"/>'
    return s


@fig("lookup")
def _():
    return _box(30, 25, 100, 40, "catalog") + _arrow(10, 45, 28, 45) + _arrow(132, 45, 152, 45)


@fig("timestep")
def _():
    s = '<line x1="15" y1="60" x2="150" y2="60" stroke="currentColor"/>'
    for x in range(20, 150, 16):
        s += f'<line x1="{x}" y1="54" x2="{x}" y2="66" stroke="currentColor"/>'
    return s + _arrow(52, 40, 68, 40) + _text(50, 32, "dt")


@fig("check")
def _():
    return _box(40, 20, 80, 50) + '<polyline points="62,46 76,58 100,32" fill="none" style="stroke:var(--acc)" stroke-width="3"/>'


@fig("report")
def _():
    return (_box(45, 12, 70, 66) + '<polyline points="54,60 68,46 80,52 104,28" fill="none" style="stroke:var(--acc)" stroke-width="1.6"/>')


@fig("formula")
def _():
    return _arrow(10, 45, 40, 45) + _box(42, 25, 76, 40, "f(x)") + _arrow(120, 45, 150, 45)


def figure_for(b):
    """Pick a figure for a registry block from its taxonomy path."""
    d, c, g, l, n = b.domain, b.category, b.geometry, b.load, b.name
    if d == "structures":
        if c == "beams":
            if g == "cantilever":
                return {"uniform": "cantilever_uniform", "end_moment": "cantilever_moment"}.get(l, "cantilever")
            if g == "simply_supported":
                return "ss_uniform" if l == "uniform" else "simply_supported"
            return {"fixed_fixed": "fixed_fixed", "fixed_guided": "fixed_guided", "propped_cantilever": "propped",
                    "restrained": "membrane", "series": "springs_series", "parallel": "springs_parallel",
                    "dispersion": "dispersion"}.get(g, "simply_supported")
        if c == "sections":
            if "moment_curvature" in l or "moment_rotation" in l:
                return "moment_curvature"
            return {"circle": "sec_circle", "tube": "sec_tube", "rectangle": "sec_rect", "rect_tube": "sec_rect_tube",
                    "i_beam": "sec_ibeam", "polygon": "sec_polygon", "plate_strip": "sec_strip"}.get(g, "sec_general")
        if c == "buckling":
            return {"column": "column", "plate": "plate_buckling", "thin_cylinder": "cyl_buckling"}[g]
        if c == "curved":
            return "ring" if g == "thin_ring" else "curved_bar"
        if c == "fatigue":
            return "sn_curve"
        if c == "impact":
            return "punch"
        if c == "joints":
            return {"bolt": "bolt", "member": "bolt", "plate_hole": "plate_hole", "thread": "thread", "tab": "tab"}[g]
        if c == "plates":
            if l == "uniform_pressure":
                return "circ_pressure" if "circ" in g else "rect_plate_ss"
            return {"circular_clamped": "circ_plate_clamped", "circular_ss": "circ_plate_ss", "circular": "circ_pressure",
                    "annular_clamped": "annular", "rectangular_ss": "rect_plate_ss", "laminate_rect_ss": "rect_plate_ss",
                    "rectangular_clamped": "rect_plate_clamped", "rectangular": "rect_plate_ss",
                    "free_edge": "plate_free_edge", "three_edge_clamped": "plate_three_edge",
                    "any_shape": "plate_point"}.get(g, "plate_general")
        if c == "rotating":
            return {"annular_disk": "spinning_annulus", "solid_disk": "spinning_disk", "thin_ring": "spinning_ring",
                    "bar": "spinning_bar"}.get(g, "spinning_disk")
        if c == "shells":
            return {"interference_fit": "interference", "thin_sphere": "sphere_pressure"}.get(g, "cylinder_pressure")
        if c == "stress_concentration":
            return "plate_hole"
        if c == "torsion":
            return "keyed" if g == "keyed_shaft" else "torsion"
        if c == "vibration":
            return "spring_mass" if g == "spring_mass" or g == "bar" else "beam_vibe"
    if d == "contact":
        if c == "hertz":
            return {"cylinder": "cylinder_flat", "flat_punch": "flat_punch"}.get(g, "sphere_flat")
        if c == "impulse":
            return "spinner_hit" if g == "spinner" else "impulse"
        if c == "indentation":
            return "indent"
        if c == "ploughing":
            return "plough"
        return "penalty"
    if d == "friction":
        if c in ("tire", "rolling"):
            return "wheel"
        if c == "bearing":
            return "bearing"
        if c == "lugre":
            return "bristles"
        return "sliding"
    if d == "actuators":
        return {"battery": "battery", "esc": "esc", "gearbox": "gearbox"}.get(c, "motor")
    if d == "aero":
        if c == "windage":
            return "spinning_bar" if g == "bar_spinner" else "spinning_disk"
        if c == "lift":
            return "airfoil"
        if c == "propulsion":
            return "propeller"
        if c == "atmosphere":
            return "atmosphere"
        if c == "groups":
            return "groups"
        return "bluff_drag"
    if d == "rigid_body":
        if c == "inertia":
            if g in ("rod",):
                return "inertia_rod"
            if g in ("rect_plate", "polygon_plate"):
                return "inertia_plate"
            if g == "assembly":
                return "assembly"
            if g == "general":
                return "parallel_axis"
            return "inertia_axis"
        if g in ("three_point_stance", "two_axle"):
            return "bot_stance"
        return "frames"
    if d == "geometry":
        if g == "tip_vs_hinged_face":
            return "hinged_face"
        if g == "circle_vs_plane":
            return "circle_plane"
        if c == "first_contact":
            return "first_contact"
        if c == "drawing":
            return "drawing"
        if c == "kinematics":
            return "spinning_disk"
        return "distance"
    if d == "thermal":
        if c == "conduction":
            return "wall_cond" if g == "plane_wall" else "cyl_cond"
        if c == "convection":
            return "convection"
        if c == "fins":
            return "fin"
        if c == "radiation":
            return "radiation"
        if c == "transient":
            return "lumped"
        return "heat_source"
    if d == "materials":
        if c in ("laminates", "cards") and g != "isotropic":
            return "laminate"
        if c == "micromechanics":
            return "micromech"
        if c in ("curves", "constitutive_params", "conversions"):
            return "stress_strain"
        return "lookup"
    if d == "constitutive":
        if c == "plasticity" and g == "hinge":
            return "moment_curvature"
        return "stress_strain"
    if d == "integrators":
        return "check" if c == "stability" else "timestep"
    if d == "diagnostics":
        return "report" if c in ("report", "diagram") else "check"
    return "formula"


_FIXES = None


_ARROWS = None


def _fix_arrows(sym_id, inner):
    """Run each load arrow on until its tip touches the part it points at (arrow_fixes.json, measured in a browser:
    the first stroke or fill straight ahead of the tip, up to 5 units away)."""
    global _ARROWS
    if _ARROWS is None:
        import json, os
        try:
            _ARROWS = json.load(open(os.path.join(os.path.dirname(__file__), "arrow_fixes.json")))
        except Exception:
            _ARROWS = {}
    fx = _ARROWS.get(sym_id)
    if not fx:
        return inner
    import re
    count = [-1]
    pat = re.compile(r'(<line x1="([-\d.]+)" y1="([-\d.]+)" x2="([-\d.]+)" y2="([-\d.]+)" ' + re.escape(ACC)
                     + r' stroke-width="1.6"/>)(<polygon points="([^"]*)" ' + re.escape(ACCF) + r'/>)')

    def sub(m):
        count[0] += 1
        p = fx.get(str(count[0]))
        if not p:
            return m.group(0)
        x2, y2 = float(m.group(4)), float(m.group(5)); dx, dy = p[0] - x2, p[1] - y2
        ln = m.group(1).replace(f'x2="{m.group(4)}" y2="{m.group(5)}"', f'x2="{p[0]:.1f}" y2="{p[1]:.1f}"')
        pts = " ".join(f"{float(a) + dx:.1f},{float(b) + dy:.1f}" for a, b in (q.split(",") for q in m.group(7).split()))
        return ln + m.group(6).replace(m.group(7), pts)
    return pat.sub(sub, inner)


def place_labels(sym_id, inner):
    """Move labels that sat on a line or shape to their checked spots (fig_fixes.json, made by measuring every
    figure in a browser: no label overlaps a stroke, a fill, another label or the frame)."""
    global _FIXES
    if _FIXES is None:
        import json, os
        try:
            _FIXES = json.load(open(os.path.join(os.path.dirname(__file__), "fig_fixes.json")))
        except Exception:
            _FIXES = {}
    inner = _fix_arrows(sym_id, inner)
    fx = _FIXES.get(sym_id)
    if not fx:
        return inner
    import re
    count = [-1]

    def sub(m):
        count[0] += 1
        p = fx.get(str(count[0]))
        if not p:
            return m.group(0)
        t = re.sub(r'\bx="[^"]*"', f'x="{p[0]}"', m.group(0), count=1)
        return re.sub(r'\by="[^"]*"', f'y="{p[1]}"', t, count=1)
    return re.sub(r"<text\b[^>]*>", sub, inner)


def symbols():
    out = []
    for k, fn in F.items():
        out.append(f'<symbol id="fig-{k}" viewBox="0 0 160 90">{place_labels("fig-" + k, fn())}</symbol>')
    return "\n".join(out)
