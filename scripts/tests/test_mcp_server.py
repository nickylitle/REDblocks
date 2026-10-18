"""REDblocks MCP server: the tool functions, and one real stdio session through the mcp client."""
import asyncio
import json
import os
import sys

import pytest

from simblocks import mcp_server as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LC = """simblocks: 1
name: lc
dim: 1d
params: {Lv: 100 uH}
circuit:
  ground: a
  parts:
    L1: {type: inductor, between: [a, b], L: $Lv, i0: 1 A}
    C1: {type: capacitor, between: [b, a], C: 100 uF}
outputs:
  ipk: {part: L1, quantity: current, reduce: max}
  vpk: {part: C1, quantity: voltage, reduce: max_abs}
run: {t_end: 4 ms}
"""


def test_write_edit_run_sweep(tmp_path, monkeypatch):
    monkeypatch.setenv("REDBLOCKS_WORKDIR", str(tmp_path))
    r = M.tool_write_model("lc.yaml", LC)
    assert r["ok"] and r["written"] and r["steps"] > 0
    bad = M.tool_edit_model("lc.yaml", "L: $Lv", "L: banana")
    assert not bad["ok"] and "banana" not in open(tmp_path / "lc.yaml").read()          # refused, nothing written
    assert not M.tool_edit_model("lc.yaml", "zzz", "y")["ok"]                              # must occur once
    ok = M.tool_edit_model("lc.yaml", "t_end: 4 ms", "t_end: 3 ms")
    assert ok["ok"] and (tmp_path / "lc.yaml.bak").exists() and "4 ms" in open(tmp_path / "lc.yaml.bak").read()
    o = M.tool_run_model("lc.yaml", histories=["L1.current", "nope"], points=7)
    assert o["outputs"]["ipk"] == pytest.approx(1.0, rel=1e-3)
    assert o["outputs"]["vpk"] == pytest.approx(1.0, rel=1e-2)                          # sqrt(L/C) * 1 A = 1 V
    assert o["histories"]["L1.current"]["max"] == pytest.approx(1.0, rel=1e-3) and len(o["samples"]["t"]) == 7
    assert o["unknown_histories"] == ["nope"] and "L1.soc" not in o["history_keys"]     # unused quantities left out
    sw = M.tool_sweep("lc.yaml", "Lv", ["25 uH", "100 uH"], outputs=["vpk"])
    assert [round(r["vpk"], 2) for r in sw["rows"]] == [pytest.approx(0.5, abs=0.01), pytest.approx(1.0, abs=0.01)]


def test_write_refuses_invalid_and_non_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("REDBLOCKS_WORKDIR", str(tmp_path))
    assert not M.tool_write_model("x.yaml", "simblocks: 1\nname: x\n")["written"]
    assert not (tmp_path / "x.yaml").exists()
    assert not M.tool_write_model("x.txt", LC)["ok"]


def test_lookup_tools():
    assert any(b["name"] == "plate_case" for b in M.tool_find_blocks("plate stiffness edge"))
    assert M.tool_block_info("plate_case")["params"]
    assert any(m["key"].startswith("4130") for m in M.tool_find_materials("4130"))
    secs = M.tool_format_reference()["sections"]
    assert any("circuit" in s for s in secs)
    assert "wave" in M.tool_format_reference("circuit")["text"]
    assert len(M.tool_list_examples()) >= 10
    assert M.plot_png(os.path.join(ROOT, "examples", "learn", "1d_1_mass_spring_damper.yaml"), _first_hist()).startswith(b"\x89PNG")


def _first_hist():
    o = M.tool_run_model(os.path.join(ROOT, "examples", "learn", "1d_1_mass_spring_damper.yaml"))
    return o["history_keys"][:2]


def test_stdio_session(tmp_path):
    mcp = pytest.importorskip("mcp")
    from mcp.client.stdio import stdio_client, StdioServerParameters
    from mcp.client import ClientSession

    async def go():
        prm = StdioServerParameters(command=sys.executable, args=["-m", "simblocks", "mcp"], cwd=str(tmp_path),
                                    env=dict(os.environ, PYTHONPATH=ROOT, REDBLOCKS_WORKDIR=str(tmp_path)))
        async with stdio_client(prm) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                names = {t.name for t in (await s.list_tools()).tools}
                assert {"read_model", "edit_model", "run_model", "sweep", "verify_model", "plot", "format_reference"} <= names
                res = await s.call_tool("write_model", {"path": "lc.yaml", "yaml": LC})
                assert json.loads(res.content[0].text)["written"]
                res = await s.call_tool("run_model", {"path": "lc.yaml"})
                assert json.loads(res.content[0].text)["outputs"]["ipk"] == pytest.approx(1.0, rel=1e-3)
                res = await s.call_tool("plot", {"path": "lc.yaml", "keys": ["L1.current"]})
                assert res.content[0].type == "image"
    asyncio.run(asyncio.wait_for(go(), 240))


def test_settings_snippet_and_desktop_link(tmp_path, monkeypatch):
    from simblocks.app.api import Api
    a = Api()
    cfg = tmp_path / "Claude" / "claude_desktop_config.json"; cfg.parent.mkdir()
    cfg.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}, "keep": 1}))
    monkeypatch.setattr(Api, "_claude_desktop_config", lambda self: str(cfg))
    st = a.mcp_status()
    assert st["ok"] and not st["desktop_linked"] and "redblocks" in st["json"] and st["claude_code"].startswith("claude mcp add redblocks -- ")
    assert a.mcp_link_claude_desktop()["ok"]
    d = json.loads(cfg.read_text())
    assert d["keep"] == 1 and "other" in d["mcpServers"] and d["mcpServers"]["redblocks"]["args"][-1] == "mcp"
    assert (tmp_path / "Claude" / "claude_desktop_config.json.bak").exists() and a.mcp_status()["desktop_linked"]
