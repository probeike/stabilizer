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
