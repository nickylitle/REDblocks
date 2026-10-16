import math, os, tempfile
import numpy as np
from PIL import Image, ImageDraw
from simblocks.geometry.drawing import outline_from_image, outline_rounded_bar
from simblocks.rigid_body.inertia import polygon_plate, I_rect_plate, I_disk

rel = lambda a, b: abs(a / b - 1)


def test_trace_bar_with_hole_from_image():
    # synthetic drawing: 200 x 40 mm bar with a 20 mm bore, 5 px/mm
    s = 5; im = Image.new("L", (1200, 400), 255); d = ImageDraw.Draw(im)
    d.rectangle([100, 100, 100 + 200 * s - 1, 100 + 40 * s - 1], fill=0)
    cx, cy = 100 + 100 * s, 100 + 20 * s
    d.ellipse([cx - 10 * s, cy - 10 * s, cx + 10 * s, cy + 10 * s], fill=255)
    with tempfile.TemporaryDirectory() as tdir:
        p = os.path.join(tdir, "bar.png"); im.save(p)
        o = outline_from_image(p, px_per_m=s * 1000)
    rho, t = 7850, 0.005
    got = polygon_plate(o["outer"], t, rho, holes=o["holes"])
    A = 0.2 * 0.04 - math.pi * 0.01 ** 2
    assert len(o["holes"]) == 1 and rel(got["area"], A) < 0.01
    m_bar = rho * t * 0.2 * 0.04; m_hole = rho * t * math.pi * 0.01 ** 2
    I = I_rect_plate(m_bar, 0.2, 0.04) - I_disk(m_hole, 0.01)
    assert rel(got["I_cm"], I) < 0.01


def test_rounded_bar_area():
    pts = outline_rounded_bar(0.2, 0.04, 0.005, 32)
    got = polygon_plate(pts, 1.0, 1.0)
    assert rel(got["area"], 0.2 * 0.04 - (4 - math.pi) * 0.005 ** 2) < 1e-3
