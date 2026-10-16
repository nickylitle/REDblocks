"""User theme storage behind the editor's Settings dialog (simblocks.app.api): one folder per theme holding theme.json
and an optional background image."""
import base64
import json
import os
import struct
import sys
import zlib

import pytest

from simblocks.app import api as A


def _png(w=4, h=3):
    raw = b"".join(b"\x00" + b"\x80\x40\x20" * w for _ in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def _url(raw, mime="image/png"):
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


THEME = dict(name="x", note="mine", scheme="dark", dash=True, wire=2.6, grid=False, ui="atkinson", mono="plexmono",
             bgFit="cover", bgOpacity=0.4, t={"bg": "#101010", "ink": "#eeeeee"})


@pytest.fixture()
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMBLOCKS_CONFIG_DIR", str(tmp_path / "cfg"))
    return A.Api()


def test_config_dir_override_and_platform_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMBLOCKS_CONFIG_DIR", str(tmp_path))
    assert A.themes_dir() == os.path.join(str(tmp_path), "themes")
    monkeypatch.delenv("SIMBLOCKS_CONFIG_DIR")
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    assert A.config_dir() == os.path.join(str(tmp_path / "Roaming"), "simblocks")
    monkeypatch.setattr(sys, "platform", "darwin")
    assert A.config_dir().endswith(os.path.join("Library", "Application Support", "simblocks"))
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert A.config_dir() == os.path.join(str(tmp_path / "xdg"), "simblocks")


def test_round_trip_with_image(api):
    png = _png()
    r = api.new_theme("My Night  Theme", THEME, _url(png))
    assert r["ok"], r
    slug = r["slug"]
    assert slug == "my-night-theme"
    folder = os.path.join(A.themes_dir(), slug)
    assert sorted(os.listdir(folder)) == ["background.png", "theme.json"]
    with open(os.path.join(folder, "theme.json")) as fh:
        stored = json.load(fh)
    assert stored["name"] == "My Night Theme" and stored["ui"] == "atkinson" and stored["grid"] is False and stored["t"]["bg"] == "#101010"

    lst = api.list_themes()
    assert lst["ok"] and [t["slug"] for t in lst["themes"]] == [slug] and lst["themes"][0]["has_image"]
    img = api.theme_image(slug)["image"]
    assert img.startswith("data:image/png;base64,") and base64.b64decode(img.split(",", 1)[1]) == png

    # save keeps the picture when image is None, removes it with ""
    th = dict(stored, note="edited", t=dict(stored["t"], bg="#202020"))
    assert api.save_theme(slug, th)["ok"]
    assert api.theme_image(slug)["image"] == img
    assert api.save_theme(slug, th, "")["ok"]
    assert api.theme_image(slug)["image"] is None
    assert api.list_themes()["themes"][0]["theme"]["t"]["bg"] == "#202020"


def test_copy_rename_delete(api):
    slug = api.new_theme("Sea", THEME, _url(_png()))["slug"]
    c = api.copy_theme(slug, "Sea")
    assert c["ok"] and c["slug"] == "sea-2"
    assert api.theme_image("sea-2")["image"] == api.theme_image(slug)["image"]
    r = api.rename_theme("sea-2", "Deep sea")
    assert r["ok"] and r["slug"] == "deep-sea"
    names = {t["slug"]: t["theme"]["name"] for t in api.list_themes()["themes"]}
    assert names == {"sea": "Sea", "deep-sea": "Deep sea"}
    assert api.rename_theme("deep-sea", "Deep Sea!")["slug"] == "deep-sea"      # same slug: name only
    assert api.delete_theme("sea")["ok"]
    assert [t["slug"] for t in api.list_themes()["themes"]] == ["deep-sea"]
    assert not api.delete_theme("sea")["ok"]


@pytest.mark.parametrize("bad", ["../evil", "..", "a/b", "a\\b", "/etc", "Sea", "sea.", "", "a--b", "x" * 49, None])
def test_traversal_and_bad_slugs_rejected(api, bad, tmp_path):
    api.new_theme("Sea", THEME)
    for call in (lambda: api.theme_image(bad), lambda: api.save_theme(bad, THEME), lambda: api.delete_theme(bad),
                 lambda: api.copy_theme(bad, "x"), lambda: api.rename_theme(bad, "x")):
        r = call()
        assert r.get("ok") is False or r.get("error"), (bad, r)
    assert os.path.isdir(os.path.join(A.themes_dir(), "sea"))
    assert not os.path.exists(tmp_path / "evil")


def test_name_cannot_escape_folder(api):
    r = api.new_theme("../../etc/passwd", THEME)
    assert r["ok"] and r["slug"] == "etc-passwd"
    assert os.path.isdir(os.path.join(A.themes_dir(), "etc-passwd"))


def test_image_validation(api):
    slug = api.new_theme("Img", THEME)["slug"]
    bad = [_url(b"GIF89a" + b"\0" * 10, "image/png"),               # content does not match type
           _url(_png(), "image/svg+xml"),                              # type not allowed
           "file:///etc/passwd",
           _url(b"\x89PNG\r\n\x1a\n" + b"\0" * (A.THEME_IMAGE_MAX + 1))]   # too big
    for img in bad:
        r = api.save_theme(slug, THEME, img)
        assert not r.get("ok"), img[:40]
    assert api.theme_image(slug)["image"] is None
    for raw, mime, ext in [(b"\xff\xd8\xff\xe0" + b"\0" * 20, "image/jpeg", "jpg"), (b"GIF89a" + b"\0" * 20, "image/gif", "gif"),
                           (b"RIFF\0\0\0\0WEBPVP8 " + b"\0" * 20, "image/webp", "webp")]:
        assert api.save_theme(slug, THEME, _url(raw, mime))["ok"]
        files = os.listdir(os.path.join(A.themes_dir(), slug))
        assert sorted(files) == sorted(["theme.json", "background." + ext])       # one picture at a time


def test_empty_name_and_missing_colours_rejected(api):
    assert not api.new_theme("   ", THEME).get("ok")
    assert not api.new_theme("ok", dict(name="ok")).get("ok")
    assert not api.save_theme("nope", THEME).get("ok")
