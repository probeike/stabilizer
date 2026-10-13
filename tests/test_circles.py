import math

import numpy as np
import pytest

from headlock import circles
from headlock.head import HEAD_TOP, L_EAR, NECK, R_EAR

CANVAS = (1080, 1920)
SR = circles.SR


def test_slots_sit_at_sixths_along_the_bottom():
    s = circles.slots(CANVAS, diameter=0.30, y=0.85, soft=0.04)
    assert [x.name for x in s] == ["left", "middle", "right"]
    assert [x.center for x in s] == [(180, 1632), (540, 1632), (900, 1632)]
    assert all(x.diameter == pytest.approx(324) for x in s)
    assert all(x.square % 2 == 0 and x.square >= 324 * 1.08 for x in s)
    assert [x.index for x in s] == [0, 1, 2]
    # Neighbouring circles don't touch, and the bottom one stays in frame.
    assert s[1].center[0] - s[0].center[0] > s[0].diameter
    assert s[0].center[1] + s[0].diameter / 2 < CANVAS[1]


def _keypoints(n, head=(400.0, 300.0), head_len=120.0, conf=0.9):
    kpts = np.zeros((n, 26, 2))
    scores = np.full((n, 26), conf)
    hx, hy = head
    kpts[:, NECK] = [hx, hy + head_len * 2 / 3]
    kpts[:, HEAD_TOP] = [hx, hy - head_len / 3]
    kpts[:, L_EAR] = [hx - 30, hy]
    kpts[:, R_EAR] = [hx + 30, hy]
    return kpts, scores


def test_framing_takes_the_median_head_and_size():
    kpts, scores = _keypoints(40)
    kpts[:5] += 500                      # a few wild detections don't move the median
    f = circles.framing_from_keypoints(kpts, scores, (540, 960), (1080, 1920), diameter=324)
    assert f.found and f.note == ""
    np.testing.assert_allclose(f.head, (800, 600), atol=1e-6)           # frame px x2 -> source px
    assert f.zoom == pytest.approx(circles.FACE_FILL * 324 / 240)


def test_framing_never_shows_past_the_picture_edge():
    kpts, scores = _keypoints(40, head=(40.0, 300.0), head_len=200.0)    # head near the left edge, big
    f = circles.framing_from_keypoints(kpts, scores, (1080, 1920), (1080, 1920), diameter=324)
    assert f.found and "inside the picture" in f.note
    assert f.zoom == pytest.approx(162 / 40)


def test_framing_falls_back_to_the_centre_without_a_face():
    kpts, scores = _keypoints(40, conf=0.1)
    f = circles.framing_from_keypoints(kpts, scores, (1080, 1920), (1080, 1920), diameter=324)
    assert not f.found and "no clear face" in f.note
    assert f.head == (540, 960) and f.zoom == pytest.approx(324 / 1080)


def _take(onsets, length, offset, freq, gain, rng, jitter=0.0):
    """A take: decaying notes at ``onsets`` (song time), recording started ``offset`` s into the song."""
    t = np.arange(int(length * SR)) / SR
    x = 0.01 * rng.standard_normal(len(t))
    for o in onsets:
        start = o - offset + rng.uniform(-jitter, jitter)
        if not 0 <= start < length:
            continue
        a = int(start * SR)
        tt = t[a:a + SR // 2] - t[a]
        f = freq * rng.uniform(0.9, 1.1)    # different notes, different harmony line
        x[a:a + len(tt)] += gain * np.exp(-tt * 8) * (np.sin(2 * np.pi * f * tt) + 0.4 * np.sin(4 * np.pi * f * tt))
    return x.astype(np.float32)


def test_sync_recovers_offsets_between_different_parts():
    rng = np.random.default_rng(1)
    onsets = np.cumsum(rng.uniform(0.15, 0.6, 120))        # ~45 s of irregular phrasing
    # Song time when each recording starts; different lengths, pitches, loudness, and each note
    # up to 50 ms early or late (separate takes never start a word at exactly the same moment).
    starts = [2.0, 0.37, 5.81]
    takes = [_take(onsets, 30, starts[0], 220, 0.5, rng, jitter=0.05),
             _take(onsets, 34, starts[1], 330, 0.08, rng, jitter=0.05),
             _take(onsets, 25, starts[2], 165, 1.0, rng, jitter=0.05)]
    r = circles.sync(takes, ["left", "middle", "right"])
    # A take that started recording later in the song starts that much later on the timeline.
    want = np.array(starts) - min(starts)
    np.testing.assert_allclose(r.offsets, want, atol=1 / 30)            # within a frame at 30 fps
    assert min(r.offsets) == 0
    assert all(c > circles.MIN_SYNC_CONFIDENCE for c in r.confidence), r.confidence
    assert r.notes == []


def test_sync_without_audio_is_left_at_zero():
    rng = np.random.default_rng(2)
    onsets = np.cumsum(rng.uniform(0.15, 0.6, 60))
    take = _take(onsets, 10, 0, 220, 0.5, rng)
    r = circles.sync([None, take, take], ["left", "middle", "right"])
    assert r.offsets == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    assert any("left has no audio" in n for n in r.notes)


def test_sync_flags_unrelated_audio():
    rng = np.random.default_rng(3)
    a = _take(np.cumsum(rng.uniform(0.15, 0.6, 60)), 15, 0, 220, 0.5, rng)
    b = _take(np.cumsum(rng.uniform(0.15, 0.6, 60)), 15, 0, 220, 0.5, rng)
    r = circles.sync([a, b], ["left", "right"])
    assert r.confidence[1] < circles.MIN_SYNC_CONFIDENCE
    assert any("uncertain" in n for n in r.notes)


def test_best_lag_sign_and_subsample():
    rng = np.random.default_rng(4)
    ref = rng.standard_normal(500)
    other = ref[37:400]                   # other's step n is ref's step n + 37
    lag, conf = circles.best_lag(ref, other, guard=10)
    assert lag == pytest.approx(37, abs=0.01) and conf > 3


lupa = pytest.importorskip("lupa")


def _eval_setting(text):
    lua = lupa.LuaRuntime()
    lua.execute("""
        function ordered() return function(t) return t end end
        local function kind(name) return function(t) t.__kind = name; return t end end
        Background, Merge, Transform, EllipseMask = kind("Background"), kind("Merge"), kind("Transform"), kind("EllipseMask")
        Input, OperatorInfo = kind("Input"), kind("OperatorInfo")
        function FuID(t) return t[1] end
    """)
    return lua.eval(text)


def _xy(v):
    return np.array([v[1], v[2]], dtype=float)


@pytest.mark.parametrize("slot_index", [0, 1, 2])
def test_setting_wires_the_circle_and_centres_the_head(slot_index):
    src = (1080, 1920)
    slot = circles.slots(CANVAS)[slot_index]
    framing = circles.Framing(head=(612.0, 455.0), zoom=0.8, found=True)
    t = _eval_setting(circles.build_setting(slot, CANVAS, src, framing))["Tools"]
    face, square, mask, zoom, canvas, place = (t[k] for k in (
        "CircleFace", "CircleSquare", "CircleMask", "CircleZoom", "CircleCanvas", "CirclePlace"))

    assert face["__kind"] == "Transform" and face["Inputs"]["Input"]["SourceOp"] == "MediaIn1"
    assert zoom["Inputs"]["Foreground"]["SourceOp"] == "CircleFace"
    assert zoom["Inputs"]["Background"]["SourceOp"] == "CircleSquare"
    assert zoom["Inputs"]["EffectMask"]["SourceOp"] == "CircleMask"
    assert zoom["Inputs"]["EffectMask"]["Source"] == "Mask"
    assert place["Inputs"]["Foreground"]["SourceOp"] == "CircleZoom"
    assert place["Inputs"]["Background"]["SourceOp"] == "CircleCanvas"
    assert place["Inputs"]["Size"]["Value"] == 1
    D = slot.square
    for bg, size in ((square, (D, D)), (canvas, CANVAS)):
        assert bg["__kind"] == "Background"
        assert (bg["Inputs"]["Width"]["Value"], bg["Inputs"]["Height"]["Value"]) == size
        assert bg["Inputs"]["TopLeftAlpha"]["Value"] == 0
    m = mask["Inputs"]
    assert mask["__kind"] == "EllipseMask" and (m["MaskWidth"]["Value"], m["MaskHeight"]["Value"]) == (D, D)
    assert m["Width"]["Value"] == pytest.approx(slot.diameter / D) == m["Height"]["Value"]
    assert m["Width"]["Value"] + 2 * m["SoftEdge"]["Value"] <= 1.0        # the soft edge fits in the square
    np.testing.assert_allclose(_xy(m["Center"]["Value"]), [0.5, 0.5])

    # Follow the head through Fusion's maths (normalized coords, Y up):
    # Transform moves the picture by Center - 0.5; a Merge puts fg px q at Center*bg + Size*(q - fg centre).
    head = np.array([framing.head[0] / src[0], 1 - framing.head[1] / src[1]])
    after_face = head + _xy(face["Inputs"]["Center"]["Value"]) - 0.5
    for size in (0.3, framing.zoom, 2.5):                # any zoom keeps the face in the middle
        in_square = _xy(zoom["Inputs"]["Center"]["Value"]) * D + size * (after_face - 0.5) * np.array(src)
        np.testing.assert_allclose(in_square, [D / 2, D / 2], atol=1e-3)
        on_canvas = _xy(place["Inputs"]["Center"]["Value"]) * CANVAS + 1 * (in_square - D / 2)
        np.testing.assert_allclose([on_canvas[0], CANVAS[1] - on_canvas[1]], slot.center, atol=1e-3)


def test_anchor_point_is_from_frame_centre_y_up():
    left = circles.slots(CANVAS)[0]
    assert circles.anchor_point(left, CANVAS) == (180 - 540, 960 - 1632)
    assert not math.isnan(circles.anchor_point(left, CANVAS)[0])
