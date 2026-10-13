from fractions import Fraction

import pytest

from fake_resolve import FakeResolve, readstring
from headlock import circles, resolve_api

CANVAS = (1080, 1920)


def _job(names=("left", "middle", "right"), offsets=(10, 0, 25)):
    slots = {s.name: s for s in circles.slots(CANVAS)}
    framing = circles.Framing((540.0, 700.0), 0.6, True)
    clips = [resolve_api.Circle(f"/v/{n}.mov", slots[n], 300, off,
                                circles.build_setting(slots[n], CANVAS, CANVAS, framing), f"/v/{n}.{n}.circle.setting")
             for n, off in zip(names, offsets)]
    return resolve_api.Job("/v/main.mp4", 900, Fraction(30000, 1001), CANVAS, clips, "main circles")


@pytest.mark.parametrize("inclusive_end", [True, False])
def test_build_places_links_and_wires_the_circles(inclusive_end):
    resolve = FakeResolve(inclusive_end=inclusive_end)
    pool = resolve.project.pool
    pool.frames = {"/v/main.mp4": 900, "/v/left.mov": 300, "/v/middle.mov": 300, "/v/right.mov": 300}
    job = _job()
    assert resolve_api.build(resolve, job, readstring) == "main circles"
    tl = resolve.project.current
    assert tl.settings == {"useCustomSettings": "1", "timelineResolutionWidth": "1080",
                           "timelineResolutionHeight": "1920", "timelineFrameRate": "29.97"}
    assert tl.tracks == {"video": 4, "audio": 4}
    placed = {(i.clip.path, i.media_type): i for i in tl.items}
    assert {(p, i.track, i.record) for (p, _), i in placed.items() if p.endswith("main.mp4")} == {
        ("/v/main.mp4", 1, 86400)}
    for name, track, offset in (("left", 2, 10), ("middle", 3, 0), ("right", 4, 25)):
        video, audio = placed[(f"/v/{name}.mov", 1)], placed[(f"/v/{name}.mov", 2)]
        assert (video.track, video.record, audio.track, audio.record) == (track, 86400 + offset, track, 86400 + offset)
        comp = video.comps[0]
        assert comp.tools["MediaOut1"].connected["Input"] == "CirclePlace"
        assert comp.tools["CircleFace"].connected["Input"] == "MediaIn1"
        slot = next(c.slot for c in job.circles if c.slot.name == name)
        assert (video.props["AnchorPointX"], video.props["AnchorPointY"]) == circles.anchor_point(slot, CANVAS)
        assert not audio.comps
    # The six chorus items (not the main clip) are linked into one group.
    assert len(tl.linked) == 1 and len(tl.linked[0]) == 6
    assert all(not i.clip.path.endswith("main.mp4") for i in tl.linked[0])
    assert job.warnings == [] and resolve.page == "edit"


def test_build_reuses_media_and_names_timelines_uniquely():
    resolve = FakeResolve()
    pool = resolve.project.pool
    pool.frames = {"/v/main.mp4": 900, "/v/left.mov": 300}
    resolve_api.build(resolve, _job(("left",), (0,)), readstring)
    assert resolve_api.build(resolve, _job(("left",), (0,)), readstring) == "main circles 2"
    assert len(pool.root.clips) == 2
    tl = resolve.project.current
    assert tl.tracks == {"video": 2, "audio": 2}


@pytest.mark.parametrize("parsed", [{"Tools": {}}, None])
def test_build_warns_when_the_paste_doesnt_take(parsed):
    resolve = FakeResolve()
    resolve.project.pool.frames = {"/v/main.mp4": 900, "/v/left.mov": 300}
    job = _job(("left",), (0,))
    resolve_api.build(resolve, job, lambda text: parsed)
    assert len(job.warnings) == 1 and "left.left.circle.setting" in job.warnings[0]


def test_no_project_is_a_clear_error():
    resolve = FakeResolve()
    resolve.project = None
    with pytest.raises(resolve_api.ResolveError, match="No project"):
        resolve_api.build(resolve, _job(), readstring)


@pytest.mark.parametrize("fps,want", [(Fraction(30000, 1001), "29.97"), (Fraction(24000, 1001), "23.976"),
                                      (Fraction(30), "30"), (Fraction(2998, 100), "29.97"), (Fraction(60), "60")])
def test_frame_rate_setting(fps, want):
    assert resolve_api.frame_rate_setting(fps) == want


def test_menu_script_opens_the_app(tmp_path):
    scripts = tmp_path / "Fusion" / "Scripts"
    scripts.mkdir(parents=True)
    path = resolve_api.install_menu_script(tmp_path / "Headlock.command", scripts / "Utility")
    text = path.read_text()
    assert path.name == "Headlock.py" and str(tmp_path / "Headlock.command") in text
    compile(text, str(path), "exec")
    with pytest.raises(resolve_api.ResolveError):
        resolve_api.install_menu_script(tmp_path / "Headlock.command", tmp_path / "nope" / "Utility")
