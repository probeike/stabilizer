"""Build the chorus-circles timeline in DaVinci Resolve through its scripting API.

Driving Resolve from our own process ("external scripting") needs Resolve **Studio** running with
Preferences > System > General > External scripting using set to **Local**. API names follow
Resolve 21.1's ``DaVinciResolveScript.pyi``. ``build`` only uses the objects it is handed, so the
tests run it against a small fake.

Things not yet confirmed inside Resolve (see docs/chorus-circles-plan.md): whether a fresh
timeline item already has a Fusion comp, that ``comp.Paste`` takes ``readstring`` output from an
external interpreter, the EllipseMask output-size input IDs, and the sign of AnchorPointY. Each
step checks what it can and says what to do by hand when something didn't take.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import circles

MAC_API = "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting"
MAC_LIB = "/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so"
MENU_DIR = Path.home() / "Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/Utility"
# Rates a Resolve timeline accepts.
FRAME_RATES = ["16", "18", "23.976", "24", "25", "29.97", "30", "47.952", "48", "50", "59.94", "60",
               "72", "95.904", "96", "100", "119.88", "120"]
VIDEO, AUDIO = 1, 2   # AppendToTimeline mediaType

ENABLE_SCRIPTING = ("Open DaVinci Resolve Studio with a project, and set Preferences > System > General > "
                    "External scripting using to Local (then restart Resolve).")


class ResolveError(RuntimeError):
    pass


def load_module():
    """Import ``DaVinciResolveScript`` from Resolve's install (macOS paths unless the env vars say otherwise)."""
    if sys.platform == "darwin":
        os.environ.setdefault("RESOLVE_SCRIPT_API", MAC_API)
        os.environ.setdefault("RESOLVE_SCRIPT_LIB", MAC_LIB)
    api = os.environ.get("RESOLVE_SCRIPT_API")
    if api and os.path.join(api, "Modules") not in sys.path:
        sys.path.append(os.path.join(api, "Modules"))
    try:
        import DaVinciResolveScript
    except ImportError as e:
        raise ResolveError(f"Can't load Resolve's scripting module ({e}). Is DaVinci Resolve installed?") from e
    return DaVinciResolveScript


def connect():
    """(resolve, DaVinciResolveScript module), or ResolveError saying how to fix it."""
    module = load_module()
    resolve = module.scriptapp("Resolve")
    if resolve is None:
        raise ResolveError("Couldn't reach DaVinci Resolve. " + ENABLE_SCRIPTING)
    return resolve, module


def frame_rate_setting(fps: Fraction) -> str:
    """The timeline frame-rate setting closest to ``fps`` (phones write rates like 29.98)."""
    return min(FRAME_RATES, key=lambda r: abs(float(r) - float(fps)))


@dataclass
class Circle:
    path: str
    slot: circles.Slot
    frames: int                 # length in the clip's own frames
    offset: int                 # timeline frames after the main clip's first frame
    setting: str                # Fusion .setting text
    setting_path: str = ""      # where that text was also written, for doing it by hand
    has_audio: bool = True


@dataclass
class Job:
    main: str
    main_frames: int
    fps: Fraction
    canvas: tuple[int, int]
    circles: list[Circle]
    name: str
    warnings: list[str] = field(default_factory=list)


def _all_clips(folder):
    yield from folder.GetClipList() or []
    for sub in folder.GetSubFolderList() or []:
        yield from _all_clips(sub)


def _import(media_pool, path: str):
    """The media pool item for ``path``, importing it unless it is already in the pool."""
    want = os.path.realpath(path)
    for clip in _all_clips(media_pool.GetRootFolder()):
        have = clip.GetClipProperty("File Path")
        if have and os.path.realpath(have) == want:
            return clip
    items = media_pool.ImportMedia([path]) or []
    if not items or items[0] is None:
        raise ResolveError(f"Resolve couldn't import {path}")
    return items[0]


def _unique_name(project, name: str) -> str:
    taken = {project.GetTimelineByIndex(i + 1).GetName() for i in range(project.GetTimelineCount())}
    candidate, n = name, 2
    while candidate in taken:
        candidate, n = f"{name} {n}", n + 1
    return candidate


def _append(media_pool, item, frames: int, track: int, record: int, media_type: int | None = None) -> list:
    """Place the whole clip at ``record`` on ``track``. Returns the new timeline items."""
    info = {"mediaPoolItem": item, "startFrame": 0, "trackIndex": track, "recordFrame": record}
    if media_type:
        info["mediaType"] = media_type
    # Whether endFrame is inclusive isn't documented: try the full length, then one less.
    for end in (frames, frames - 1):
        placed = [x for x in (media_pool.AppendToTimeline([{**info, "endFrame": end}]) or []) if x]
        if placed:
            return placed
    raise ResolveError(f"Resolve couldn't place {item.GetName()} on track {track}")


def _clip_frames(item, fallback: int) -> int:
    try:
        return int(item.GetClipProperty("Frames"))
    except (TypeError, ValueError):
        return fallback


def _wire_circle(item, circle: Circle, readstring, warn) -> None:
    """Paste the circle's Fusion nodes into the item's comp and connect them to MediaIn/MediaOut."""
    by_hand = (f"{circle.slot.name}: in the Fusion page, drag {Path(circle.setting_path).name} into the Node "
               "Editor, connect MediaIn1 to CircleFace if it isn't, and CirclePlace to MediaOut1")
    comp = item.GetFusionCompByIndex(1) if item.GetFusionCompCount() else item.AddFusionComp()
    if not comp:
        warn(f"couldn't open a Fusion comp. {by_hand}")
        return
    table = readstring(circle.setting)
    if not table:   # Paste() with nothing would paste whatever is on the clipboard
        warn(f"Resolve couldn't read the circle's Fusion nodes. {by_hand}")
        return
    comp.Paste(table)
    tools = {name: comp.FindTool(name) for name in
             ("MediaIn1", "MediaOut1", "CircleFace", "CirclePlace", "CircleMask")}
    missing = [name for name, tool in tools.items() if not tool]
    if missing:
        warn(f"Fusion comp is missing {', '.join(missing)}. {by_hand}")
        return
    tools["CircleFace"].ConnectInput("Input", tools["MediaIn1"])
    tools["MediaOut1"].ConnectInput("Input", tools["CirclePlace"])
    # The mask's output-size input IDs aren't documented; check they took.
    D = circle.slot.square
    if (tools["CircleMask"].GetInput("MaskWidth"), tools["CircleMask"].GetInput("MaskHeight")) != (D, D):
        warn(f"{circle.slot.name}: check CircleMask's Image tab in Fusion: output size should be custom {D} x {D}, "
             "or the circle may come out oval")


def build(resolve, job: Job, readstring) -> str:
    """Make the timeline: main clip on V1/A1, each circle on V2-V4 (video) and A2-A4 (its audio), linked.

    Returns the timeline's name. Things that need a hand afterwards go into ``job.warnings``.
    """
    warn = job.warnings.append
    project = resolve.GetProjectManager().GetCurrentProject()
    if not project:
        raise ResolveError("No project is open in Resolve. Open (or create) one and try again.")
    pool = project.GetMediaPool()
    main = _import(pool, job.main)
    items = [_import(pool, c.path) for c in job.circles]

    name = _unique_name(project, job.name)
    tl = pool.CreateEmptyTimeline(name)
    if not tl:
        raise ResolveError(f"Resolve couldn't create the timeline {name!r}")
    project.SetCurrentTimeline(tl)
    W, H = job.canvas
    # Before any clip goes on: the frame rate can't change once the timeline has media.
    for key, value in (("useCustomSettings", "1"), ("timelineResolutionWidth", str(W)),
                       ("timelineResolutionHeight", str(H)), ("timelineFrameRate", frame_rate_setting(job.fps))):
        if not tl.SetSetting(key, value):
            warn(f"couldn't set the timeline's {key} to {value}; set it in Timeline > Timeline Settings")

    top = 2 + max(c.slot.index for c in job.circles)
    while tl.GetTrackCount("video") < top:
        tl.AddTrack("video")
    while tl.GetTrackCount("audio") < top:
        tl.AddTrack("audio", "stereo")

    start = tl.GetStartFrame()
    _append(pool, main, _clip_frames(main, job.main_frames), 1, start)
    group = []
    for item, c in zip(items, job.circles):
        track = 2 + c.slot.index
        frames = _clip_frames(item, c.frames)
        video = _append(pool, item, frames, track, start + c.offset, VIDEO)
        group += video
        if c.has_audio:
            group += _append(pool, item, frames, track, start + c.offset, AUDIO)
        _wire_circle(video[0], c, readstring, warn)
        ax, ay = circles.anchor_point(c.slot, job.canvas)
        if not (video[0].SetProperty("AnchorPointX", ax) and video[0].SetProperty("AnchorPointY", ay)):
            warn(f"{c.slot.name}: couldn't set the Anchor Point, so Zoom will scale from the frame centre")
    if len(group) > 1 and not tl.SetClipsLinked(group, True):
        warn("couldn't link the circles; select them all and press Cmd+Alt+L to link them")
    resolve.OpenPage("edit")
    return name


# ---- captions --------------------------------------------------------------------------------

CAPTION_TRACK = "Captions"
CAPTION_TEMPLATES = ("Headlock Caption", "Text+")
NEEDS_TEMPLATE = ("Put a Text+ title in the Media Pool first (once per project): in the Edit page's Effects "
                  "library, open Titles and drag Text+ into the Media Pool.")


@dataclass(frozen=True)
class AudioPiece:
    path: str
    source_start: float     # s into the file
    duration: float         # s
    at: float               # s from the timeline's start


def current_timeline(resolve):
    project = resolve.GetProjectManager().GetCurrentProject()
    tl = project.GetCurrentTimeline() if project else None
    if not tl:
        raise ResolveError("No timeline is open in Resolve. Open the one to caption and try again.")
    return project, tl


def timeline_fps(tl) -> float:
    m = re.match(r"\s*([\d.]+)", str(tl.GetSetting("timelineFrameRate")))
    if not m:
        raise ResolveError("Couldn't read the timeline's frame rate")
    return float(m[1])


def audio_pieces(tl, track: int = 1) -> list[AudioPiece]:
    """The clips on audio track ``track``: which file, which part of it, and where on the timeline.

    The part of the file comes from the clip's left offset (how far it could be extended at the
    head, i.e. where it starts in its media), in timeline frames.
    """
    fps, t0 = timeline_fps(tl), tl.GetStartFrame()
    pieces = []
    for item in tl.GetItemListInTrack("audio", track) or []:
        clip = item.GetMediaPoolItem()
        path = clip.GetClipProperty("File Path") if clip else None
        if path:
            pieces.append(AudioPiece(path, item.GetLeftOffset() / fps, item.GetDuration() / fps,
                                     (item.GetStart() - t0) / fps))
    if not pieces:
        raise ResolveError(f"Audio track {track} of the timeline has no clips to listen to")
    return pieces


def timeline_audio(pieces: list[AudioPiece], sr: int = circles.SR) -> np.ndarray:
    """The track's sound as one waveform from the timeline's start, silent between clips."""
    out = np.zeros(int(max(p.at + p.duration for p in pieces) * sr) + 1, np.float32)
    for p in pieces:
        a = circles.read_audio(p.path, sr, p.source_start, p.duration)
        if a is None:
            continue
        i = int(round(p.at * sr))
        a = a[:len(out) - i]
        out[i:i + len(a)] += a
    return out


def _find_template(pool):
    clips = {}
    for clip in _all_clips(pool.GetRootFolder()):
        clips.setdefault(clip.GetName(), clip)
    for name in CAPTION_TEMPLATES:
        if name in clips:
            return clips[name]
    raise ResolveError(NEEDS_TEMPLATE)


def _caption_track(tl) -> int:
    """The video track named Captions, emptied; or a new one on top."""
    for i in range(1, tl.GetTrackCount("video") + 1):
        if tl.GetTrackName("video", i) == CAPTION_TRACK:
            old = tl.GetItemListInTrack("video", i) or []
            if old:
                tl.DeleteClips(old)
            return i
    tl.AddTrack("video")
    i = tl.GetTrackCount("video")
    tl.SetTrackName("video", i, CAPTION_TRACK)
    return i


def _wire_caption(item, setting: str, readstring) -> str | None:
    """Swap the title's Text+ for the caption nodes. Returns a problem, or None."""
    comp = item.GetFusionCompByIndex(1) if item.GetFusionCompCount() else None
    table = readstring(setting) if comp else None
    if not table:
        return "couldn't open the title's Fusion comp"
    for tool in (comp.GetToolList(False, "TextPlus") or {}).values():
        tool.Delete()
    comp.Paste(table)
    merge, out = comp.FindTool("CaptionMerge"), comp.FindTool("MediaOut1")
    if not (merge and out):
        return "the caption nodes didn't paste"
    out.ConnectInput("Input", merge)
    if comp.FindTool("CaptionLit").GetInput("End") is None:
        return ("Text+ has no input called End, so the words can't light up; tell Claude which input "
                "Text+'s Write On uses")
    return None


def place_captions(resolve, placements: list, readstring) -> tuple[str, list[str]]:
    """One title per lyric line on the Captions track. Returns the timeline's name and any problems.

    Whether AppendToTimeline's endFrame is inclusive isn't documented, so the first title is
    placed, measured and, if one frame long, replaced; the rest follow the same rule.
    """
    project, tl = current_timeline(resolve)
    template = _find_template(project.GetMediaPool())
    pool = project.GetMediaPool()
    track = _caption_track(tl)
    start = tl.GetStartFrame()
    inclusive = None
    problems: list[str] = []
    for p in placements:
        def put(end):
            got = pool.AppendToTimeline([{"mediaPoolItem": template, "startFrame": 0, "endFrame": end,
                                          "trackIndex": track, "recordFrame": start + p.start}]) or []
            return got[0] if got and got[0] else None
        if inclusive is None:
            item = put(p.frames)
            if item and item.GetDuration() == p.frames + 1:
                tl.DeleteClips([item])
                inclusive, item = True, put(p.frames - 1)
            else:
                inclusive = False
        else:
            item = put(p.frames - 1 if inclusive else p.frames)
        if not item:
            problems.append(f"couldn't place the line \"{p.phrase.text(10**6)}\"")
            continue
        problem = _wire_caption(item, p.setting, readstring)
        if problem and problem not in problems:
            problems.append(problem)
    resolve.OpenPage("edit")
    return tl.GetName(), problems


MENU_SCRIPT = '''"""Opens the Headlock window. Added by Headlock's "Add Headlock to Resolve's Scripts menu" button."""
import subprocess

subprocess.Popen(["open", {command!r}])
'''


def install_menu_script(command: str | Path, folder: str | Path = MENU_DIR) -> Path:
    """Write Workspace > Scripts > Headlock, which opens ``command`` (Headlock.command)."""
    folder = Path(folder)
    if not folder.parent.exists():
        raise ResolveError(f"Resolve's scripts folder isn't there ({folder.parent}). Is DaVinci Resolve installed?")
    folder.mkdir(exist_ok=True)
    path = folder / "Headlock.py"
    path.write_text(MENU_SCRIPT.format(command=str(Path(command).resolve())))
    return path
