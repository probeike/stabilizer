"""A small desktop window for headlock: choose a video, pick a few options, press Track.

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

from . import cli

FRAMES = {
    "Vertical 1080 x 1920": "1080x1920",
    "Vertical 4K 2160 x 3840": "2160x3840",
    "Horizontal 1920 x 1080": "1920x1080",
    "Horizontal 4K 3840 x 2160": "3840x2160",
}
# A tqdm progress line, e.g. "Tracking:  45%|####      | 450/1000 [00:10<00:12, 45.0f/s]"
PROGRESS = re.compile(r"(?:(?P<desc>[A-Za-z ]+):)?\s*(?P<pct>\d+)%\|.*?\|\s*(?P<n>\d+)/(?P<total>\d+)"
                      r"(?:\s*\[[^<\]]*<(?P<left>[^,\]]*))?")
NEXT_STEPS = {
    "resolve": "Next, in DaVinci Resolve: put the whole clip on a timeline of the output size, select it, open "
               "the Fusion page, drag the .setting file into the node area, and connect HeadlockMerge's output "
               "to MediaOut1.",
    "premiere": "Next, in Premiere Pro: File > Import the .xml. It opens as a sequence with the motion keyframed.",
    "360": "The head-locked video is ready. Import it into Resolve like any other clip.",
}


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
        self.status = tk.StringVar(value="")

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
        main.columnconfigure(1, weight=1)

        ttk.Label(main, text="Video").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(main, textvariable=self.video, state="readonly").grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(main, text="Choose…", command=self._choose).grid(row=0, column=2, **pad)
        ttk.Label(main, textvariable=self.info, foreground="gray").grid(row=1, column=1, columnspan=2, sticky="w", padx=8)
        ttk.Checkbutton(main, text="This is a 360° video (equirectangular export)", variable=self.is360).grid(
            row=2, column=1, columnspan=2, sticky="w", **pad)

        opts = ttk.LabelFrame(main, text="Options", padding=8)
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

        ttk.Label(opts, text="Output frame").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Combobox(opts, textvariable=self.frame, values=list(FRAMES), state="readonly", width=26).grid(
            row=1, column=1, sticky="w", pady=3)
        ttk.Label(opts, text="Head height").grid(row=2, column=0, sticky="w", pady=3)
        row = ttk.Frame(opts)
        row.grid(row=2, column=1, sticky="w")
        ttk.Spinbox(row, from_=0.05, to=0.95, increment=0.05, width=5, textvariable=self.lock_y).pack(side="left")
        ttk.Label(row, text="  0 = top of frame, 1 = bottom", foreground="gray").pack(side="left")
        ttk.Label(opts, text="Smoothing (Hz)").grid(row=3, column=0, sticky="w", pady=3)
        row = ttk.Frame(opts)
        row.grid(row=3, column=1, sticky="w")
        ttk.Spinbox(row, from_=1, to=15, increment=1, width=5, textvariable=self.cutoff).pack(side="left")
        ttk.Label(row, text="  lower = steadier, higher = follows fast moves more tightly",
                  foreground="gray").pack(side="left")
        ttk.Checkbutton(opts, text="Use the Apple GPU (faster tracking)", variable=self.gpu).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=3)
        ttk.Checkbutton(opts, text="Also make a preview video with tracking marks", variable=self.preview).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=3)

        buttons = ttk.Frame(main)
        buttons.grid(row=4, column=0, columnspan=3, sticky="w", **pad)
        self.go = ttk.Button(buttons, text="Track & export", command=self._track)
        self.go.pack(side="left")
        self.again = ttk.Button(buttons, text="Re-export (keep tracking)", command=self._export)
        self.again.pack(side="left", padx=8)
        self.reveal = ttk.Button(buttons, text="Show in Finder", command=self._reveal)
        self.reveal.pack(side="left")

        self.bar = ttk.Progressbar(main, maximum=100)
        self.bar.grid(row=5, column=0, columnspan=3, sticky="ew", **pad)
        ttk.Label(main, textvariable=self.status).grid(row=6, column=0, columnspan=3, sticky="w", padx=8)
        self.log = tk.Text(main, height=14, width=100, wrap="word", font=("Menlo", 11))
        self.log.grid(row=7, column=0, columnspan=3, sticky="nsew", **pad)
        main.rowconfigure(7, weight=1)
        self._refresh_buttons()

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
        has_video = bool(self.video.get())
        self.go.state(["!disabled"] if has_video and not self.busy else ["disabled"])
        can_redo = has_video and not self.busy and self._track_json().exists()
        self.again.state(["!disabled"] if can_redo else ["disabled"])
        has_out = self.last_output is not None and self.last_output.exists()
        self.reveal.state(["!disabled"] if has_out else ["disabled"])

    def _mode_changed(self) -> None:
        if self.is360.get():
            self.flat_row.grid_remove()
            self.row360.grid()
        else:
            self.row360.grid_remove()
            self.flat_row.grid()

    def _choose(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a video",
            filetypes=[("Video", "*.mp4 *.MP4 *.mov *.MOV *.m4v *.mkv *.insv *.INSV"), ("All files", "*")])
        if path:
            self.load(path)

    def load(self, path: str) -> None:
        if Path(path).suffix.lower() == ".insv":
            messagebox.showinfo("Export from Insta360 first",
                                "This is the camera's raw .insv file. Export it from Insta360 Studio as a 360° "
                                "MP4 first (with FlowState stabilization and horizon lock on), then choose that.")
            return
        from .video import probe
        try:
            info = probe(path)
        except Exception as e:  # noqa: BLE001 - shown to the user
            messagebox.showerror("Can't read video", f"ffprobe couldn't read this file:\n{e}")
            return
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
        self.last_output = self._output() if self._output().exists() else None
        self._refresh_buttons()

    # ---- running ----------------------------------------------------------------------------

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

    def _track(self) -> None:
        self._run(["track", self.video.get(), "--projection", "360" if self.is360.get() else "flat",
                   "--device", "coreml" if self.gpu.get() else "cpu"] + self._layout_args())

    def _export(self) -> None:
        self._run(["export", str(self._track_json())] + self._layout_args())

    def _run(self, argv: list[str]) -> None:
        try:
            self._layout_args()
        except (tk.TclError, ValueError):
            messagebox.showerror("Check the options", "One of the number boxes isn't a number.")
            return
        self.busy = True
        self.last_output = None
        self._refresh_buttons()
        self.bar["value"] = 0
        self.status.set("Starting…")
        self.log.delete("1.0", "end")
        self._log("headlock " + " ".join(f'"{a}"' if " " in a else a for a in argv) + "\n\n")
        output = self._output()

        def work() -> None:
            old = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = _QueueStream(self.q)
            ok = False
            try:
                cli.main(argv)
                ok = True
            except SystemExit as e:
                ok = e.code in (0, None)
                if not ok:
                    print(e.code if isinstance(e.code, str) else f"stopped (exit code {e.code})")
            except Exception:  # noqa: BLE001 - reported in the log
                traceback.print_exc()
            finally:
                sys.stdout, sys.stderr = old
                self.q.put(("done", (ok, output)))

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

    def _finished(self, ok: bool, output: Path) -> None:
        self.busy = False
        if ok:
            self.bar["value"] = 100
            self.last_output = output
            self.status.set(f"Done: {output.name}")
            self._log("\n" + NEXT_STEPS["360" if self.is360.get() else self.target.get()] + "\n")
        else:
            self.status.set("Failed. See the messages above.")
        self._refresh_buttons()

    def _reveal(self) -> None:
        if self.last_output:
            subprocess.run(["open", "-R", str(self.last_output)], check=False)


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
