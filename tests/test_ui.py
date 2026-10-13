import io
import shutil
import time

import pytest
from tqdm import tqdm

from headlock.ui import PROGRESS

tk = pytest.importorskip("tkinter")


@pytest.fixture
def app(monkeypatch, tmp_path):
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display for Tk")
    root.withdraw()
    from headlock import ui
    monkeypatch.setattr(ui, "LYRICS_FILE", tmp_path / "lyrics.txt")
    yield ui.App(root)
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


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg/ffprobe")
def test_preview_window_plays_and_follows_the_options(app, tmp_path):
    from headlock import cli

    clip = tmp_path / "dot360.mp4"
    cli.main(["test-clip", str(clip), "--360", "--size", "1024x512", "--fps", "10", "--seconds", "2"])
    app.video.set(str(clip))
    app._open_preview()
    win = app.preview_win

    def pump_until(pred, timeout=10.0):
        deadline = time.monotonic() + timeout
        while not pred() and time.monotonic() < deadline:
            app.root.update()
            time.sleep(0.01)
        assert pred()

    pump_until(lambda: win.shown == 0)
    assert (win.photo.width(), win.photo.height()) == (360, 640)
    app.frame.set("Horizontal 1920 x 1080")                 # the preview follows the tab's options
    pump_until(lambda: win.photo.width() == 640)
    win.slider.set(12)                                      # scrubbing seeks
    pump_until(lambda: win.shown == 12)
    win._toggle()
    pump_until(lambda: win.shown == 19)                     # plays to the end
    app.fov.set("12x")                                      # mid-edit garbage keeps the last framing
    win.close()
    assert app.preview_win is None and not win.playback._thread.is_alive()


def test_captions_args_keep_the_lyrics_for_next_time(app, tmp_path):
    app.lyrics.insert("1.0", "[Chorus]\nSing it with me\n")
    app.caption_font.set("IM Fell English")
    app.caption_y.set(0.15)
    argv = app._captions_argv()
    assert argv[:3] == ["captions", "--lyrics", str(tmp_path / "lyrics.txt")]
    assert (tmp_path / "lyrics.txt").read_text() == "[Chorus]\nSing it with me\n"
    assert argv[argv.index("--font") + 1] == "IM Fell English" and argv[argv.index("--y") + 1] == "0.15"
    from headlock import ui
    again = ui.App(app.root)                                    # a new window starts with them
    assert again.lyrics.get("1.0", "end").strip() == "[Chorus]\nSing it with me"
