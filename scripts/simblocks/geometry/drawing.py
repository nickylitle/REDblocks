"""Drawing / image / DXF input: turn a picture or sketch of a part outline into metric vertex lists that
the polygon blocks (polygon_section, polygon_plate, moment_curvature_polygon) consume."""
import math
import numpy as np

from ..registry import block


@block("geometry", "drawing", "image", "outline", kind="builder", source="threshold + marching squares (skimage)",
       tags=("drawing", "photo", "sketch", "trace"))
def outline_from_image(path, px_per_m=None, ref_length_m=None, ref_length_px=None, dark_part=True,
                       threshold=None, simplify_tol_px=1.5, min_hole_area_px=50):
    """Trace a part silhouette from an image file. Scale by px_per_m, or by a known reference
    (ref_length_m spans ref_length_px pixels). Returns dict(outer=[(x,y)...] in m, holes=[...], px_per_m).
    y is flipped to point up. Origin is the outline centroid."""
    from PIL import Image
    from skimage import measure, filters
    img = np.asarray(Image.open(path).convert("L"), float)
    thr = threshold if threshold is not None else filters.threshold_otsu(img)
    mask = img < thr if dark_part else img > thr
    if px_per_m is None:
        if not (ref_length_m and ref_length_px):
            raise ValueError("give px_per_m or ref_length_m + ref_length_px")
        px_per_m = ref_length_px / ref_length_m
    cs = measure.find_contours(mask.astype(float), 0.5)
    polys = [measure.approximate_polygon(c, simplify_tol_px) for c in cs]
    areas = [abs(_area(p)) for p in polys]
    io = int(np.argmax(areas))
    outer = polys[io]

    def to_m(p):
        xy = np.c_[p[:, 1], -p[:, 0]] / px_per_m
        return xy

    O = to_m(outer)
    holes = [to_m(p) for i, p in enumerate(polys) if i != io and areas[i] > min_hole_area_px
             and _inside(p[0], outer)]
    cx, cy = _centroid(O)
    return dict(outer=[tuple(v) for v in (O - [cx, cy])[:-1]],
                holes=[[tuple(v) for v in (h - [cx, cy])[:-1]] for h in holes], px_per_m=px_per_m)


@block("geometry", "drawing", "dxf", "outline", kind="builder", source="ezdxf LWPOLYLINE/POLYLINE",
       tags=("CAD", "DXF", "laser cut file"))
def outline_from_dxf(path, units_to_m=1e-3):
    """Closed polylines from a DXF (e.g. a waterjet/laser file). Largest = outer, others = holes. mm by default."""
    import ezdxf
    doc = ezdxf.readfile(path)
    loops = []
    for e in doc.modelspace().query("LWPOLYLINE POLYLINE"):
        pts = [(p[0] * units_to_m, p[1] * units_to_m) for p in (e.get_points("xy") if e.dxftype() == "LWPOLYLINE" else e.points())]
        if len(pts) >= 3:
            loops.append(np.array(pts))
    for e in doc.modelspace().query("CIRCLE"):
        c, r = e.dxf.center, e.dxf.radius
        th = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        loops.append(np.c_[c[0] + r * np.cos(th), c[1] + r * np.sin(th)] * units_to_m)
    areas = [abs(_area(l)) for l in loops]
    io = int(np.argmax(areas))
    return dict(outer=[tuple(v) for v in loops[io]], holes=[[tuple(v) for v in l] for i, l in enumerate(loops) if i != io])


@block("geometry", "drawing", "primitive", "outline", kind="builder", source="geometry", tags=("blade", "rotating bar"))
def outline_rounded_bar(L, W, r_corner=0.0, n_arc=8):
    """Outline of a bar / blade blank L x W centered at the origin, optional corner radius."""
    if r_corner <= 0:
        return [(-L / 2, -W / 2), (L / 2, -W / 2), (L / 2, W / 2), (-L / 2, W / 2)]
    pts = []
    for cx, cy, a0 in ((L / 2 - r_corner, -W / 2 + r_corner, -90), (L / 2 - r_corner, W / 2 - r_corner, 0),
                       (-L / 2 + r_corner, W / 2 - r_corner, 90), (-L / 2 + r_corner, -W / 2 + r_corner, 180)):
        for k in range(n_arc + 1):
            a = math.radians(a0 + 90 * k / n_arc)
            pts.append((cx + r_corner * math.cos(a), cy + r_corner * math.sin(a)))
    return pts


def _area(p):
    p = np.asarray(p, float); x, y = p[:, 0], p[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def _centroid(p):
    p = np.asarray(p, float); x, y = p[:, 0], p[:, 1]; x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y; A = cr.sum() / 2
    return ((x + x1) * cr).sum() / (6 * A), ((y + y1) * cr).sum() / (6 * A)


def _inside(pt, poly):
    from matplotlib.path import Path
    return Path(poly).contains_point(pt)
