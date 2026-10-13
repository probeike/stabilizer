"""A small desktop window for everything headlock does, one tab per job.

- **Head-lock**: choose a video, pick a few options, press Track (flat or 360 footage);
- **Chorus circles**: the main video plus a left, middle and right chorus clip, built in Resolve;
- **Captions**: karaoke captions of pasted lyrics on the current Resolve timeline;
- **Check & test**: the fit check and the synthetic test clips.

It runs exactly what the command line runs (``cli.main``) on a worker thread and shows the output
and progress, so anything done here can also be done with ``headlock ...``.
"""

from __future__ import annotations

import queue
import re
import shutil
import subprocess
import sys
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable

from . import cli
from .circles import NAMES

FRAMES = {
    "Vertical 1080 x 1920": "1080x1920",
    "Vertical 4K 2160 x 3840": "2160x3840",
    "Horizontal 1920 x 1080": "1920x1080",
    "Horizontal 4K 3840 x 2160": "3840x2160",
}
VIDEO_TYPES = [("Video", "*.mp4 *.MP4 *.mov *.MOV *.m4v *.mkv *.insv *.INSV"), ("All files", "*")]
# A tqdm progress line, e.g. "Tracking:  45%|####      | 450/1000 [00:10<00:12, 45.0f/s]"
PROGRESS = re.compile(r"(?:(?P<desc>[A-Za-z ()]+):)?\s*(?P<pct>\d+)%\|.*?\|\s*(?P<n>\d+)/(?P<total>\d+)"
                      r"(?:\s*\[[^<\]]*<(?P<left>[^,\]]*))?")
NEXT_STEPS = {
    "resolve": "Next, in DaVinci Resolve: put the whole clip on a timeline of the output size, select it, open "
               "the Fusion page, drag the .setting file into the node area, and connect HeadlockMerge's output "
               "to MediaOut1.",
    "premiere": "Next, in Premiere Pro: File > Import the .xml. It opens as a sequence with the motion keyframed.",
    "360": "The head-locked video is ready. Import it into Resolve like any other clip.",
}
REPO = Path(__file__).resolve().parents[2]
LYRICS_FILE = Path.home() / ".headlock" / "lyrics.txt"   # the last lyrics pasted, kept for next time


class _QueueStream:
    """Stand-in for stdout/stderr that hands text to the UI thread.

    Says it is a terminal so the tracking/render progress bars (tqdm) are written.
    """

    def __init__(self, q: queue.Queue):
        self.q = q

    def write(self, s: str) -> int:
        self.q.put(("text", s))
        return len(s)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return True


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.q: queue.Queue = queue.Queue()
        self.busy = False
        self.last_output: Path | None = None

        # Head-lock
        self.video = tk.StringVar()
        self.info = tk.StringVar(value="No video chosen.")
        self.is360 = tk.BooleanVar(value=False)
        self.target = tk.StringVar(value="resolve")
        self.fmt = tk.StringVar(value="mp4")
        self.frame = tk.StringVar(value="Vertical 1080 x 1920")
        self.lock_y = tk.DoubleVar(value=0.2)
        self.fov = tk.DoubleVar(value=100)
        self.cutoff = tk.DoubleVar(value=6)
        self.max_zoom = tk.DoubleVar(value=1.0)
        self.gpu = tk.BooleanVar(value=sys.platform == "darwin")
        self.preview = tk.BooleanVar(value=False)
        # Chorus circles
        self.main_video = tk.StringVar()
        self.main_info = tk.StringVar(value="The vertical video the circles go over.")
        self.chorus = {name: tk.StringVar() for name in NAMES}
        self.diameter = tk.DoubleVar(value=0.30)
        self.circle_y = tk.DoubleVar(value=0.85)
        self.soft = tk.DoubleVar(value=0.04)
        self.at = tk.StringVar(value="0:00")
        # Check & test
        self.check_track = tk.StringVar()
        self.check_scale = tk.StringVar(value="auto")
        self.test_out = tk.StringVar(value=str(Path.home() / "Movies" / "headlock-test.mp4"))
        self.test360 = tk.BooleanVar(value=False)
        # Captions
        self.lyrics_file = LYRICS_FILE
        self.caption_font = tk.StringVar(value="IM Fell English")
        self.caption_size = tk.DoubleVar(value=0.07)
        self.caption_y = tk.DoubleVar(value=0.13)

        self.status = tk.StringVar(value="")
        self.preview_win: PreviewWindow | None = None
        root.title("Headlock")
        self._build()
        self.is360.trace_add("write", lambda *_: self._mode_changed())
        self._mode_changed()
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            self._log("ffmpeg/ffprobe not found. Install them first (brew install ffmpeg), then reopen this window.\n")
        root.after(100, self._poll)

    # ---- layout -----------------------------------------------------------------------------

    def _build(self) -> None:
        pad = {"padx": 8, "pady": 4}
        main = ttk.Frame(self.root, padding=12)
        main.grid(sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)

        self.tabs = ttk.Notebook(main)
        self.tabs.grid(row=0, column=0, columnspan=2, sticky="ew")
        for title, build in (("Head-lock", self._build_headlock), ("Chorus circles", self._build_circles),
                             ("Captions", self._build_captions), ("Check & test", self._build_tools)):
            tab = ttk.Frame(self.tabs, padding=8)
            tab.columnconfigure(1, weight=1)
            build(tab, pad)
            self.tabs.add(tab, text=title)

        self.bar = ttk.Progressbar(main, maximum=100)
        self.bar.grid(row=1, column=0, sticky="ew", **pad)
        self.reveal = ttk.Button(main, text="Show in Finder", command=self._reveal)
        self.reveal.grid(row=1, column=1, **pad)
        ttk.Label(main, textvariable=self.status).grid(row=2, column=0, columnspan=2, sticky="w", padx=8)
        self.log = tk.Text(main, height=14, width=100, wrap="word", font=("Menlo", 11))
        self.log.grid(row=3, column=0, columnspan=2, sticky="nsew", **pad)
        main.rowconfigure(3, weight=1)
        self._refresh_buttons()

    def _file_row(self, parent, row: int, label: str, var: tk.StringVar, choose: Callable[[], None], pad) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(parent, textvariable=var, state="readonly").grid(row=row, column=1, sticky="ew", **pad)
        ttk.Button(parent, text="Choose…", command=choose).grid(row=row, column=2, **pad)

    def _spin(self, parent, row: int, label: str, var, lo: float, hi: float, step: float, hint: str) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
        line = ttk.Frame(parent)
        line.grid(row=row, column=1, sticky="w")
        ttk.Spinbox(line, from_=lo, to=hi, increment=step, width=6, textvariable=var).pack(side="left")
        ttk.Label(line, text="  " + hint, foreground="gray").pack(side="left")

    def _build_headlock(self, tab, pad) -> None:
        self._file_row(tab, 0, "Video", self.video, self._choose, pad)
        ttk.Label(tab, textvariable=self.info, foreground="gray").grid(row=1, column=1, columnspan=2, sticky="w", padx=8)
        ttk.Checkbutton(tab, text="This is a 360° video (equirectangular export)", variable=self.is360).grid(
            row=2, column=1, columnspan=2, sticky="w", **pad)

        opts = ttk.LabelFrame(tab, text="Options", padding=8)
        opts.grid(row=3, column=0, columnspan=3, sticky="ew", **pad)
        opts.columnconfigure(1, weight=1)

        self.flat_row = ttk.Frame(opts)
        ttk.Label(self.flat_row, text="Export for").pack(side="left")
        ttk.Radiobutton(self.flat_row, text="DaVinci Resolve", value="resolve", variable=self.target).pack(side="left", padx=6)
        ttk.Radiobutton(self.flat_row, text="Premiere Pro", value="premiere", variable=self.target).pack(side="left", padx=6)
        ttk.Label(self.flat_row, text="   Max zoom").pack(side="left")
        ttk.Spinbox(self.flat_row, from_=1.0, to=3.0, increment=0.1, width=5, textvariable=self.max_zoom).pack(side="left", padx=4)
        ttk.Label(self.flat_row, text="(raise above 1 if the source has little room to crop)",
                  foreground="gray").pack(side="left")
        self.flat_row.grid(row=0, column=0, columnspan=2, sticky="w", pady=3)

        self.row360 = ttk.Frame(opts)
        ttk.Label(self.row360, text="Save as").pack(side="left")
        ttk.Radiobutton(self.row360, text="MP4 (to post)", value="mp4", variable=self.fmt).pack(side="left", padx=6)
        ttk.Radiobutton(self.row360, text="MOV ProRes 10-bit (to grade in Resolve)", value="mov",
                        variable=self.fmt).pack(side="left", padx=6)
        ttk.Label(self.row360, text="   Field of view °").pack(side="left")
        ttk.Spinbox(self.row360, from_=60, to=150, increment=5, width=5, textvariable=self.fov).pack(side="left", padx=4)
        self.row360.grid(row=0, column=0, columnspan=2, sticky="w", pady=3)

        ttk.Label(opts, text="Output frame").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Combobox(opts, textvariable=self.frame, values=list(FRAMES), state="readonly", width=26).grid(
            row=1, column=1, sticky="w", pady=3)
        self._spin(opts, 2, "Head height", self.lock_y, 0.05, 0.95, 0.05, "0 = top of frame, 1 = bottom")
        self._spin(opts, 3, "Smoothing (Hz)", self.cutoff, 1, 15, 1,
                   "lower = steadier, higher = follows fast moves more tightly")
        ttk.Checkbutton(opts, text="Use the Apple GPU (faster tracking)", variable=self.gpu).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=3)
        ttk.Checkbutton(opts, text="Also make a preview video with tracking marks", variable=self.preview).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=3)

        buttons = ttk.Frame(tab)
        buttons.grid(row=4, column=0, columnspan=3, sticky="w", **pad)
        self.go = ttk.Button(buttons, text="Track", command=self._track)
        self.go.pack(side="left")
        self.preview_btn = ttk.Button(buttons, text="Preview framing", command=self._open_preview)
        self.preview_btn.pack(side="left", padx=8)
        self.again = ttk.Button(buttons, text="Export", command=self._export)
        self.again.pack(side="left")
        ttk.Label(tab, text="Track once, then play the result in the preview while you adjust the framing, "
                            "and Export when it looks right. The tracking is kept, so you can export again.",
                  foreground="gray", wraplength=640).grid(row=5, column=0, columnspan=3, sticky="w", padx=8)

    def _build_circles(self, tab, pad) -> None:
        self._file_row(tab, 0, "Main video", self.main_video, self._choose_main, pad)
        ttk.Label(tab, textvariable=self.main_info, foreground="gray").grid(row=1, column=1, columnspan=2,
                                                                            sticky="w", padx=8)
        for i, name in enumerate(NAMES):
            self._file_row(tab, 2 + i, f"{name.capitalize()} circle", self.chorus[name],
                           lambda n=name: self._choose_chorus(n), pad)
        ttk.Label(tab, text="The chorus clips are lined up with each other by their audio (not with the main "
                            "video) and linked, so you drag them into place together.",
                  foreground="gray", wraplength=640).grid(row=5, column=1, columnspan=2, sticky="w", padx=8)

        opts = ttk.LabelFrame(tab, text="Options", padding=8)
        opts.grid(row=6, column=0, columnspan=3, sticky="ew", **pad)
        self._spin(opts, 0, "Circle size", self.diameter, 0.15, 0.33, 0.01, "fraction of the width (0.33 = touching)")
        self._spin(opts, 1, "Height", self.circle_y, 0.5, 0.95, 0.01, "centre of the circles, 0 = top, 1 = bottom")
        self._spin(opts, 2, "Soft edge", self.soft, 0.0, 0.2, 0.01, "fraction of the circle size")
        ttk.Label(opts, text="Start at").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=3)
        line = ttk.Frame(opts)
        line.grid(row=3, column=1, sticky="w")
        ttk.Entry(line, textvariable=self.at, width=8).pack(side="left")
        ttk.Label(line, text="  minutes:seconds into the main video (you can drag them after)",
                  foreground="gray").pack(side="left")
        ttk.Checkbutton(opts, text="Use the Apple GPU (faster face finding)", variable=self.gpu).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=3)

        buttons = ttk.Frame(tab)
        buttons.grid(row=7, column=0, columnspan=3, sticky="w", **pad)
        self.build_btn = ttk.Button(buttons, text="Build in Resolve", command=lambda: self._circles(True))
        self.build_btn.pack(side="left")
        self.files_btn = ttk.Button(buttons, text="Only write the Fusion files", command=lambda: self._circles(False))
        self.files_btn.pack(side="left", padx=8)
        ttk.Button(buttons, text="Add Headlock to Resolve's Scripts menu", command=self._install_menu).pack(side="left")
        ttk.Label(tab, text="Building needs DaVinci Resolve Studio open with a project, and Preferences > System > "
                            "General > External scripting using set to Local.",
                  foreground="gray", wraplength=640).grid(row=8, column=0, columnspan=3, sticky="w", padx=8)

    def _build_captions(self, tab, pad) -> None:
        ttk.Label(tab, text="Lyrics").grid(row=0, column=0, sticky="nw", **pad)
        self.lyrics = tk.Text(tab, height=10, width=60, wrap="word", font=("Menlo", 11), undo=True)
        self.lyrics.grid(row=0, column=1, sticky="ew", **pad)
        try:
            self.lyrics.insert("1.0", self.lyrics_file.read_text())
        except OSError:
            pass
        ttk.Button(tab, text="Open…", command=self._open_lyrics).grid(row=0, column=2, sticky="n", **pad)
        ttk.Label(tab, text="One caption per line, as you'd like it shown; blank lines and headings like [Chorus] "
                            "are skipped. No timing needed: it listens to the song and works out when each word "
                            "is sung.", foreground="gray", wraplength=640).grid(row=1, column=1, columnspan=2,
                                                                               sticky="w", padx=8)

        opts = ttk.LabelFrame(tab, text="Look", padding=8)
        opts.grid(row=2, column=0, columnspan=3, sticky="ew", **pad)
        ttk.Label(opts, text="Font").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(opts, textvariable=self.caption_font, width=24).grid(row=0, column=1, sticky="w", pady=3)
        self._spin(opts, 1, "Size", self.caption_size, 0.03, 0.15, 0.005, "Text+ size; long lines go on two rows")
        self._spin(opts, 2, "Height", self.caption_y, 0.03, 0.5, 0.01,
                   "centre of the captions, 0 = top (the head is at 0.35 in 360 head-locks)")

        buttons = ttk.Frame(tab)
        buttons.grid(row=3, column=0, columnspan=3, sticky="w", **pad)
        self.caption_btn = ttk.Button(buttons, text="Caption the current timeline", command=self._captions)
        self.caption_btn.pack(side="left")
        self.recaption_btn = ttk.Button(buttons, text="Place again from a timing file…", command=self._recaption)
        self.recaption_btn.pack(side="left", padx=8)
        ttk.Label(tab, text="Align your clips first, with the lead vocal on audio track 1, and have that timeline "
                            "open in DaVinci Resolve Studio. Once per project, drag Text+ (Effects > Titles) into "
                            "the Media Pool. Each line becomes a title on a track called Captions (captioning "
                            "again replaces them). The timing is also saved as a .lrc text file next to the main "
                            "video: to fix a word, change its time there and use Place again.",
                  foreground="gray", wraplength=640).grid(row=4, column=0, columnspan=3, sticky="w", padx=8)

    def _build_tools(self, tab, pad) -> None:
        check = ttk.LabelFrame(tab, text="Fit check (normal videos)", padding=8)
        check.grid(row=0, column=0, columnspan=3, sticky="ew", **pad)
        check.columnconfigure(1, weight=1)
        self._file_row(check, 0, "Tracking file", self.check_track, self._choose_track, pad)
        ttk.Label(check, text="Zoom").grid(row=1, column=0, sticky="w", **pad)
        line = ttk.Frame(check)
        line.grid(row=1, column=1, sticky="w")
        ttk.Entry(line, textvariable=self.check_scale, width=8).pack(side="left")
        ttk.Label(line, text="  auto, or a scale such as 0.9. Uses the Head-lock tab's output frame and head height.",
                  foreground="gray").pack(side="left")
        self.check_btn = ttk.Button(check, text="Check fit", command=self._check)
        self.check_btn.grid(row=2, column=0, sticky="w", **pad)

        test = ttk.LabelFrame(tab, text="Test clip (a dot on a known path, to confirm Resolve/Premiere keep it still)",
                              padding=8)
        test.grid(row=1, column=0, columnspan=3, sticky="ew", **pad)
        test.columnconfigure(1, weight=1)
        self._file_row(test, 0, "Save as", self.test_out, self._choose_test_out, pad)
        ttk.Checkbutton(test, text="360° test clip", variable=self.test360).grid(row=1, column=1, sticky="w", **pad)
        self.test_btn = ttk.Button(test, text="Make test clip", command=self._test_clip)
        self.test_btn.grid(row=2, column=0, sticky="w", **pad)

    # ---- state ------------------------------------------------------------------------------

    def _base(self) -> str:
        return str(Path(self.video.get()).with_suffix(""))

    def _track_json(self) -> Path:
        return Path(self._base() + ".track.json")

    def _output(self) -> Path:
        if self.is360.get():
            return Path(f"{self._base()}.headlock.{self.fmt.get()}")
        return Path(self._base() + cli.TARGETS[self.target.get()])

    def _refresh_buttons(self) -> None:
        def enable(button, on: bool) -> None:
            button.state(["!disabled"] if on and not self.busy else ["disabled"])

        has_video = bool(self.video.get())
        tracked = has_video and self._track_json().exists()
        enable(self.go, has_video)
        enable(self.again, tracked)
        self.preview_btn.state(["!disabled"] if tracked else ["disabled"])   # fine to watch while exporting
        can_circle = bool(self.main_video.get()) and any(v.get() for v in self.chorus.values())
        enable(self.build_btn, can_circle)
        enable(self.files_btn, can_circle)
        enable(self.caption_btn, True)
        enable(self.recaption_btn, True)
        enable(self.check_btn, bool(self.check_track.get()))
        enable(self.test_btn, bool(self.test_out.get()))
        self.reveal.state(["!disabled"] if self.last_output is not None and self.last_output.exists() else ["disabled"])

    def _mode_changed(self) -> None:
        if self.is360.get():
            self.flat_row.grid_remove()
            self.row360.grid()
        else:
            self.row360.grid_remove()
            self.flat_row.grid()

    def _choose(self) -> None:
        path = filedialog.askopenfilename(title="Choose a video", filetypes=VIDEO_TYPES)
        if path:
            self.load(path)

    def _choose_main(self) -> None:
        path = filedialog.askopenfilename(title="Choose the main video", filetypes=VIDEO_TYPES)
        if not path:
            return
        info = self._probe(path)
        if info:
            self.main_video.set(path)
            shape = "vertical" if info.height > info.width else "not vertical: the circles still go along its bottom"
            self.main_info.set(f"{info.width} x {info.height} · {float(info.fps):.3g} fps · {shape}")
        self._refresh_buttons()

    def _choose_chorus(self, name: str) -> None:
        path = filedialog.askopenfilename(title=f"Choose the chorus clip for the {name} circle", filetypes=VIDEO_TYPES)
        if path and self._probe(path):
            self.chorus[name].set(path)
        self._refresh_buttons()

    def _choose_track(self) -> None:
        path = filedialog.askopenfilename(title="Choose a tracking file", filetypes=[("Tracking", "*.track.json")])
        if path:
            self.check_track.set(path)
        self._refresh_buttons()

    def _choose_test_out(self) -> None:
        path = filedialog.asksaveasfilename(title="Save the test clip as", defaultextension=".mp4",
                                            filetypes=[("MP4", "*.mp4")])
        if path:
            self.test_out.set(path)
        self._refresh_buttons()

    def _probe(self, path: str):
        from .video import probe
        try:
            return probe(path)
        except Exception as e:  # noqa: BLE001 - shown to the user
            messagebox.showerror("Can't read video", f"ffprobe couldn't read this file:\n{e}")
            return None

    def load(self, path: str) -> None:
        if Path(path).suffix.lower() == ".insv":
            messagebox.showinfo("Export from Insta360 first",
                                "This is the camera's raw .insv file. Export it from Insta360 Studio as a 360° "
                                "MP4 first (with FlowState stabilization and horizon lock on), then choose that.")
            return
        info = self._probe(path)
        if not info:
            return
        if self.preview_win:
            self.preview_win.close()
        self.video.set(path)
        is360 = info.width == 2 * info.height
        if self._track_json().exists():  # an earlier run already decided; re-export must agree with it
            from .trackio import Track
            is360 = Track.load(self._track_json()).projection == "equirect"
        portrait = is360 or info.height > info.width
        secs = info.frame_count / float(info.fps)
        kind = "360° video" if is360 else ("portrait video" if info.height > info.width else "video")
        self.info.set(f"{info.width} x {info.height} · {float(info.fps):.3g} fps · "
                      f"{int(secs // 60)}:{int(secs % 60):02d} · {kind}"
                      + (" · already tracked" if self._track_json().exists() else ""))
        self.is360.set(is360)
        self.frame.set("Vertical 1080 x 1920" if portrait else "Horizontal 1920 x 1080")
        self.lock_y.set(0.35 if is360 else 0.2)
        if self._track_json().exists():
            self.check_track.set(str(self._track_json()))
        self.last_output = self._output() if self._output().exists() else None
        self._refresh_buttons()

    # ---- commands ---------------------------------------------------------------------------

    def _device(self) -> str:
        return "coreml" if self.gpu.get() else "cpu"

    def _layout_args(self) -> list[str]:
        args = ["-o", str(self._output()),
                "--output-resolution", FRAMES[self.frame.get()],
                "--lock-y", f"{self.lock_y.get():g}",
                "--cutoff-hz", f"{self.cutoff.get():g}",
                "--target", self.target.get()]
        args += ["--fov", f"{self.fov.get():g}"] if self.is360.get() else ["--max-scale", f"{self.max_zoom.get():g}"]
        if self.preview.get():
            args += ["--preview", self._base() + ".preview.mp4"]
        return args

    def _next_step(self) -> str:
        return NEXT_STEPS["360" if self.is360.get() else self.target.get()]

    def _options(self):
        """The Head-lock tab's framing, as ``_layout_args`` passes it to export."""
        from .player import Options
        return Options(cli._size(FRAMES[self.frame.get()]), (0.5, self.lock_y.get()), fov=self.fov.get(),
                       cutoff_hz=self.cutoff.get(), max_scale=self.max_zoom.get())

    def _track(self) -> None:
        self._run(lambda: ["track", self.video.get(), "--projection", "360" if self.is360.get() else "flat",
                           "--device", self._device(), "--no-export"] + self._layout_args(), self._track_json(),
                  "Tracked. Adjust the framing in the preview, then press Export.", on_success=self._open_preview)

    def _open_preview(self) -> None:
        if self.preview_win:
            self.preview_win.win.lift()
            return
        try:
            self.preview_win = PreviewWindow(self, self._track_json())
        except Exception as e:  # noqa: BLE001 - shown to the user
            messagebox.showerror("Can't preview", str(e))

    def _export(self) -> None:
        self._run(lambda: ["export", str(self._track_json())] + self._layout_args(), self._output(),
                  self._next_step())

    def _circles_argv(self, build: bool) -> list[str]:
        cli._seconds(self.at.get())   # raises ValueError if it isn't a time
        argv = ["circles", self.main_video.get()]
        for name, var in self.chorus.items():
            if var.get():
                argv += [f"--{name}", var.get()]
        argv += ["--diameter", f"{self.diameter.get():g}", "--y", f"{self.circle_y.get():g}",
                 "--soft", f"{self.soft.get():g}", "--at", self.at.get().strip(), "--device", self._device()]
        return argv + ([] if build else ["--no-resolve"])

    def _circles(self, build: bool) -> None:
        first = next(((n, v.get()) for n, v in self.chorus.items() if v.get()), None)
        output = Path(first[1]).with_suffix(f".{first[0]}.circle.setting") if first else None
        self._run(lambda: self._circles_argv(build), output)

    def _open_lyrics(self) -> None:
        path = filedialog.askopenfilename(title="Open lyrics", filetypes=[("Text", "*.txt *.lrc"), ("All files", "*")])
        if path:
            self.lyrics.delete("1.0", "end")
            self.lyrics.insert("1.0", Path(path).read_text())

    def _caption_look(self) -> list[str]:
        return ["--font", self.caption_font.get().strip() or "IM Fell English",
                "--size", f"{self.caption_size.get():g}", "--y", f"{self.caption_y.get():g}"]

    def _captions_argv(self) -> list[str]:
        look = self._caption_look()
        text = self.lyrics.get("1.0", "end").strip()
        self.lyrics_file.parent.mkdir(parents=True, exist_ok=True)
        self.lyrics_file.write_text(text + "\n")
        return ["captions", "--lyrics", str(self.lyrics_file)] + look

    def _captions(self) -> None:
        if not self.lyrics.get("1.0", "end").strip():
            messagebox.showinfo("Paste the lyrics first", "Paste the song's lyrics into the box, one caption per line.")
            return
        self._run(self._captions_argv)

    def _recaption(self) -> None:
        path = filedialog.askopenfilename(title="Choose the timing file", filetypes=[("Timing", "*.lrc")])
        if path:
            self._run(lambda: ["captions", "--timing", path] + self._caption_look())

    def _check_argv(self) -> list[str]:
        scale = self.check_scale.get().strip() or "auto"
        if scale != "auto":
            float(scale)   # raises ValueError if it isn't a number
        return ["check", self.check_track.get(), "--output-resolution", FRAMES[self.frame.get()],
                "--lock-y", f"{self.lock_y.get():g}", "--scale", scale, "--max-scale", f"{self.max_zoom.get():g}"]

    def _check(self) -> None:
        self._run(self._check_argv, ok_codes=(0, 1, None))   # 1 = doesn't fit, which the report explains

    def _test_clip_argv(self) -> list[str]:
        return ["test-clip", self.test_out.get()] + (["--360"] if self.test360.get() else [])

    def _test_clip(self) -> None:
        Path(self.test_out.get()).parent.mkdir(parents=True, exist_ok=True)
        self._run(self._test_clip_argv, Path(self.test_out.get()))

    def _install_menu(self) -> None:
        from .resolve_api import ResolveError, install_menu_script
        command = REPO / "Headlock.command"
        try:
            if not command.exists():
                raise ResolveError(f"{command} is missing")
            path = install_menu_script(command)
        except (ResolveError, OSError) as e:
            messagebox.showerror("Couldn't add the menu item", str(e))
            return
        messagebox.showinfo("Added to Resolve",
                            "In DaVinci Resolve, choose Workspace > Scripts > Headlock to open this window "
                            f"(restart Resolve if it isn't listed yet).\n\nInstalled {path}")

    # ---- running ----------------------------------------------------------------------------

    def _run(self, make_argv: Callable[[], list[str]], output: Path | None = None, next_step: str = "",
             ok_codes: tuple = (0, None), on_success: Callable[[], None] | None = None) -> None:
        try:
            argv = make_argv()
        except (tk.TclError, ValueError):
            messagebox.showerror("Check the options", "One of the boxes doesn't hold a number (or a time like 1:23).")
            return
        self.busy = True
        self.last_output = None
        self._refresh_buttons()
        self.bar["value"] = 0
        self.status.set("Starting…")
        self.log.delete("1.0", "end")
        self._log("headlock " + " ".join(f'"{a}"' if " " in a else a for a in argv) + "\n\n")

        def work() -> None:
            old = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = _QueueStream(self.q)
            ok = False
            try:
                cli.main(argv)
                ok = True
            except SystemExit as e:
                ok = e.code in ok_codes
                if not ok:
                    print(e.code if isinstance(e.code, str) else f"stopped (exit code {e.code})")
            except Exception:  # noqa: BLE001 - reported in the log
                traceback.print_exc()
            finally:
                sys.stdout, sys.stderr = old
                self.q.put(("done", (ok, output, next_step, on_success)))

        threading.Thread(target=work, daemon=True).start()

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "done":
                    self._finished(*payload)
                else:
                    self._text(payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _text(self, s: str) -> None:
        for piece in s.split("\r"):
            m = PROGRESS.search(piece)
            if m:
                self.bar["value"] = int(m["pct"])
                desc = (m["desc"] or "Working").strip()
                left = f", about {m['left'].strip()} left" if m["left"] and "?" not in m["left"] else ""
                self.status.set(f"{desc}: {m['pct']}% ({m['n']}/{m['total']} frames{left})")
            elif piece:
                self._log(piece)

    def _log(self, s: str) -> None:
        self.log.insert("end", s)
        self.log.see("end")

    def _finished(self, ok: bool, output: Path | None, next_step: str,
                  on_success: Callable[[], None] | None = None) -> None:
        self.busy = False
        if ok:
            self.bar["value"] = 100
            self.last_output = output
            self.status.set(f"Done: {output.name}" if output else "Done.")
            if next_step:
                self._log("\n" + next_step + "\n")
        else:
            self.status.set("Failed. See the messages above.")
        self._refresh_buttons()
        if ok and on_success:
            on_success()

    def _reveal(self) -> None:
        if self.last_output:
            subprocess.run(["open", "-R", str(self.last_output)], check=False)


class PreviewWindow:
    """Plays the tracked clip as it will come out. The framing options are the Head-lock tab's own,
    so changing them here (or there) redraws the picture live, and Export uses them as set."""

    def __init__(self, app: App, track_path: Path):
        from .player import Playback, to_ppm
        from .trackio import Track

        self.app = app
        self.track = Track.load(track_path)
        video = self.track.video if Path(self.track.video).exists() else self.track.source
        self.n = len(self.track.head)
        self.to_ppm = to_ppm
        self.marks = tk.BooleanVar(value=False)
        self.latest: tuple[int, bytes] | None = None
        self.shown = -1
        self.lock = threading.Lock()
        self.closed = False
        self._moving_slider = False

        self.win = tk.Toplevel(app.root)
        self.win.title(f"Framing preview: {Path(video).name}")
        self.photo = tk.PhotoImage()
        self._build()
        try:
            self.framer = self._framer()
        except Exception:
            self.win.destroy()
            raise
        self.playback = Playback(video, self.track.fps, self.n, self._render(self.framer), self._on_frame,
                                 self.framer.decode_height)
        variables = (app.fov, app.lock_y, app.cutoff, app.frame, app.max_zoom, self.marks)
        self.traces = [(v, v.trace_add("write", lambda *_: self._changed())) for v in variables]
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        for key, action in (("<space>", self._toggle), ("<Left>", lambda: self._step(-1)),
                            ("<Right>", lambda: self._step(1))):
            self.win.bind(key, lambda e, a=action: None if isinstance(e.widget, ttk.Entry) else a())  # not while typing
        self._poll()

    def _build(self) -> None:
        pad = {"padx": 8, "pady": 4}
        frame = ttk.Frame(self.win, padding=10)
        frame.grid(sticky="nsew")
        ttk.Label(frame, image=self.photo, background="black").grid(row=0, column=0, columnspan=3, **pad)

        opts = ttk.LabelFrame(frame, text="Framing", padding=8)
        opts.grid(row=0, column=3, sticky="n", **pad)
        if self.track.projection == "equirect":
            self.app._spin(opts, 0, "Field of view °", self.app.fov, 60, 150, 5, "")
        else:
            self.app._spin(opts, 0, "Max zoom", self.app.max_zoom, 1.0, 3.0, 0.1, "")
        self.app._spin(opts, 1, "Head height", self.app.lock_y, 0.05, 0.95, 0.05, "")
        self.app._spin(opts, 2, "Smoothing (Hz)", self.app.cutoff, 1, 15, 1, "")
        ttk.Label(opts, text="Output frame").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Combobox(opts, textvariable=self.app.frame, values=list(FRAMES), state="readonly", width=22).grid(
            row=3, column=1, sticky="w", pady=3)
        ttk.Checkbutton(opts, text="Show tracking marks", variable=self.marks).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=3)
        ttk.Label(opts, text="Green cross: where the head is held.\nRed dot: the raw, unsmoothed tracking.",
                  foreground="gray").grid(row=5, column=0, columnspan=2, sticky="w")
        self.note = ttk.Label(opts, text="", foreground="#b05000", wraplength=260)
        self.note.grid(row=6, column=0, columnspan=2, sticky="w", pady=6)
        ttk.Button(opts, text="Export with these settings", command=self._export).grid(
            row=7, column=0, columnspan=2, sticky="w", pady=(10, 3))
        ttk.Label(opts, text="Space: play/pause. Arrows: one frame.\nThe preview is smaller and silent; the "
                             "export is full quality.", foreground="gray").grid(row=8, column=0, columnspan=2,
                                                                                sticky="w")

        self.play_btn = ttk.Button(frame, text="Play", width=7, command=self._toggle)
        self.play_btn.grid(row=1, column=0, **pad)
        self.slider = ttk.Scale(frame, from_=0, to=max(self.n - 1, 1), orient="horizontal", command=self._scrub)
        self.slider.grid(row=1, column=1, sticky="ew", **pad)
        frame.columnconfigure(1, weight=1)
        self.clock = ttk.Label(frame, text="", width=14)
        self.clock.grid(row=1, column=2, **pad)

    def _framer(self):
        from .player import Framer
        framer = Framer(self.track, self.app._options(), marks=self.marks.get())
        self.note.configure(text=framer.note)
        return framer

    def _render(self, framer):
        return lambda i, frame: self.to_ppm(framer(i, frame))

    def _on_frame(self, i: int, ppm: bytes) -> None:   # worker thread
        with self.lock:
            self.latest = (i, ppm)

    def _changed(self) -> None:
        try:
            self.framer = self._framer()
        except (tk.TclError, ValueError):
            return   # a box is mid-edit; keep the last good framing
        self.playback.set_render(self._render(self.framer))

    def _toggle(self) -> None:
        self.playback.pause() if self.playback.playing else self.playback.play()

    def _step(self, d: int) -> None:
        self.playback.pause()
        self.playback.seek(self.playback.position + d)

    def _scrub(self, value) -> None:
        if not self._moving_slider:
            self.playback.seek(round(float(value)))

    def _export(self) -> None:
        if not self.app.busy:
            self.app._export()

    def _time(self, i: int) -> str:
        s = i / float(self.track.fps)
        return f"{int(s // 60)}:{s % 60:04.1f}"

    def _poll(self) -> None:
        if self.closed:
            return
        with self.lock:
            latest, self.latest = self.latest, None
        if latest:
            i, ppm = latest
            self.photo.configure(data=ppm, format="ppm")
            self.shown = i
            self._moving_slider = True
            self.slider.set(i)
            self._moving_slider = False
            self.clock.configure(text=f"{self._time(i)} / {self._time(self.n)}")
        self.play_btn.configure(text="Pause" if self.playback.playing else "Play")
        self.win.after(15, self._poll)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for var, trace in self.traces:
            var.trace_remove("write", trace)
        self.playback.close()
        self.win.destroy()
        self.app.preview_win = None


def main() -> None:
    root = tk.Tk()
    app = App(root)
    if len(sys.argv) > 1:
        app.load(sys.argv[1])
    root.lift()
    root.attributes("-topmost", True)
    root.after(500, lambda: root.attributes("-topmost", False))
    root.mainloop()


if __name__ == "__main__":
    main()
