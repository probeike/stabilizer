import numpy as np
import pytest

from headlock import geometry, resolve_fusion

lupa = pytest.importorskip("lupa")

SRC = (3840, 2160)
SEQ = (1080, 1920)
LOCK = (0.5, 0.2)


def _eval_setting(text):
    """Evaluate the .setting as Lua, with Fusion's constructors stubbed to tag their tables."""
    lua = lupa.LuaRuntime()
    lua.execute("""
        function ordered() return function(t) return t end end
        local function kind(name) return function(t) t.__kind = name; return t end end
        Background, Merge, XYPath, BezierSpline = kind("Background"), kind("Merge"), kind("XYPath"), kind("BezierSpline")
        Input, OperatorInfo = kind("Input"), kind("OperatorInfo")
    """)
    return lua.eval(text)


def _keys(spline, n):
    kf = spline["KeyFrames"]
    assert kf[n] is None
    return np.array([kf[i][1] for i in range(n)])


def test_setting_is_valid_lua_and_pins_the_head():
    rng = np.random.default_rng(0)
    heads = rng.uniform([1200, 300], [2600, 900], size=(40, 2))
    scale = 0.9
    P = geometry.positions(heads, SRC, SEQ, scale, LOCK)
    t = _eval_setting(resolve_fusion.build(P, SEQ, scale))
    tools = t["Tools"]

    canvas, merge = tools["HeadlockCanvas"], tools["HeadlockMerge"]
    assert canvas["__kind"] == "Background" and merge["__kind"] == "Merge"
    assert (canvas["Inputs"]["Width"]["Value"], canvas["Inputs"]["Height"]["Value"]) == SEQ
    assert canvas["Inputs"]["UseFrameFormatSettings"]["Value"] == 0
    assert merge["Inputs"]["Background"]["SourceOp"] == "HeadlockCanvas"
    assert merge["Inputs"]["Foreground"]["SourceOp"] == "MediaIn1"
    assert merge["Inputs"]["Center"]["SourceOp"] == "HeadlockCenter"
    assert merge["Inputs"]["Size"]["Value"] == pytest.approx(scale)
    path = tools["HeadlockCenter"]
    assert path["__kind"] == "XYPath"
    cx = _keys(tools[path["Inputs"]["X"]["SourceOp"]], len(P))
    cy = _keys(tools[path["Inputs"]["Y"]["SourceOp"]], len(P))

    # Fusion Merge: foreground pixel q lands at Center (in bg px, Y flipped) + Size * (q - fg centre).
    center_px = np.column_stack([cx * SEQ[0], (1 - cy) * SEQ[1]])
    landed = center_px + scale * (heads - np.array(SRC) / 2)
    np.testing.assert_allclose(landed, np.tile(geometry.lock_point(SEQ, LOCK), (len(P), 1)), atol=1e-3)


def test_center_mapping_is_y_up():
    c = resolve_fusion.position_to_center(np.array([[540.0, 960.0], [0.0, 0.0], [1080.0, 1920.0]]), SEQ)
    np.testing.assert_allclose(c, [[0.5, 0.5], [0.0, 1.0], [1.0, 0.0]])
