import io

import pytest
from tqdm import tqdm

from headlock.ui import PROGRESS

tk = pytest.importorskip("tkinter")


@pytest.fixture
def app():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display for Tk")
    root.withdraw()
    from headlock.ui import App
    yield App(root)
    root.destroy()


@pytest.mark.parametrize("ncols", [None, 80])
def test_progress_regex_reads_tqdm(ncols):
    buf = io.StringIO()
    bar = tqdm(total=200, desc="Tracking", unit="f", file=buf, mininterval=0, ncols=ncols)
    bar.update(50)
    bar.refresh()
    m = PROGRESS.search(buf.getvalue().split("\r")[-1])
    assert (m["desc"].strip(), m["pct"], m["n"], m["total"]) == ("Tracking", "25", "50", "200")


def test_flat_resolve_args(app, tmp_path):
    app.video.set(str(tmp_path / "take.mp4"))
    app.is360.set(False)
    app.target.set("resolve")
    app.frame.set("Vertical 1080 x 1920")
    app.lock_y.set(0.25)
    args = app._layout_args()
    assert args[:2] == ["-o", str(tmp_path / "take.headlock.setting")]
    assert args[args.index("--output-resolution") + 1] == "1080x1920"
    assert args[args.index("--lock-y") + 1] == "0.25"
    assert "--max-scale" in args and "--fov" not in args


def test_360_prores_args(app, tmp_path):
    app.video.set(str(tmp_path / "selfie.mp4"))
    app.is360.set(True)
    app.fmt.set("mov")
    app.fov.set(90)
    args = app._layout_args()
    assert args[:2] == ["-o", str(tmp_path / "selfie.headlock.mov")]
    assert args[args.index("--fov") + 1] == "90"
    assert "--max-scale" not in args


def test_progress_regex_reads_face_finding():
    m = PROGRESS.search("Finding the face (left):  50%|#####     | 20/40 [00:02<00:02, 9.1f/s]")
    assert (m["desc"].strip(), m["pct"]) == ("Finding the face (left)", "50")


def test_circles_args(app, tmp_path):
    app.main_video.set(str(tmp_path / "song.mp4"))
    app.chorus["left"].set(str(tmp_path / "a.mov"))
    app.chorus["right"].set(str(tmp_path / "c.mov"))
    app.diameter.set(0.28)
    app.at.set(" 1:23.5 ")
    app.gpu.set(True)
    argv = app._circles_argv(build=True)
    assert argv[:2] == ["circles", str(tmp_path / "song.mp4")]
    assert argv[argv.index("--left") + 1] == str(tmp_path / "a.mov")
    assert argv[argv.index("--right") + 1] == str(tmp_path / "c.mov")
    assert "--middle" not in argv and "--no-resolve" not in argv
    assert argv[argv.index("--diameter") + 1] == "0.28" and argv[argv.index("--at") + 1] == "1:23.5"
    assert argv[argv.index("--device") + 1] == "coreml"
    assert app._circles_argv(build=False)[-1] == "--no-resolve"
    app.at.set("soon")
    with pytest.raises(ValueError):
        app._circles_argv(build=True)


def test_check_and_test_clip_args(app, tmp_path):
    app.check_track.set(str(tmp_path / "take.track.json"))
    app.frame.set("Horizontal 1920 x 1080")
    app.check_scale.set("0.9")
    argv = app._check_argv()
    assert argv[:2] == ["check", str(tmp_path / "take.track.json")]
    assert argv[argv.index("--scale") + 1] == "0.9" and argv[argv.index("--output-resolution") + 1] == "1920x1080"
    app.check_scale.set("big")
    with pytest.raises(ValueError):
        app._check_argv()
    app.test_out.set(str(tmp_path / "dot.mp4"))
    app.test360.set(True)
    assert app._test_clip_argv() == ["test-clip", str(tmp_path / "dot.mp4"), "--360"]


def test_buttons_follow_what_is_chosen(app, tmp_path):
    assert app.build_btn.instate(["disabled"])
    app.main_video.set(str(tmp_path / "song.mp4"))
    app.chorus["middle"].set(str(tmp_path / "b.mov"))
    app._refresh_buttons()
    assert app.build_btn.instate(["!disabled"]) and app.files_btn.instate(["!disabled"])
