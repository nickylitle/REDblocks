"""Editor regression scripts from the October 2026 audit (tests/editor/bug_*.py). Each opens the editor in headless
Chromium with Playwright and exits 1 if its bug is back. Skipped where Playwright or its Chromium is not installed."""
import glob
import os
import subprocess
import sys

import pytest

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "editor")
SCRIPTS = sorted(glob.glob(os.path.join(HERE, "bug_*.py")))

try:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as _p:
        _p.chromium.launch().close()
    HAVE = True
except Exception:
    HAVE = False


@pytest.mark.skipif(not HAVE, reason="Playwright with Chromium is not installed")
@pytest.mark.parametrize("script", SCRIPTS, ids=[os.path.basename(s) for s in SCRIPTS])
def test_editor_regression(script):
    r = subprocess.run([sys.executable, script], cwd=HERE, capture_output=True, text=True, timeout=900)   # the round trip runs every example twice
    assert r.returncode == 0, (r.stdout[-2000:] + r.stderr[-2000:])
