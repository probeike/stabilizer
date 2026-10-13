"""headlock command line: track, check, export, test-clip, circles."""

from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import geometry, premiere_xml, resolve_fusion, sphere
from .smooth import METHODS, fill_gaps, find_gaps, smooth
from .trackio import Track

TARGETS = {"resolve": ".headlock.setting", "premiere": ".headlock.xml"}


def _size(text: str) -> tuple[int, int]:
    w, h = text.lower().split("x")
    return int(w), int(h)


def _seconds(text: str) -> float:
    """Seconds from "83.4", "1:23.4" or "0:01:23.4"."""
    total = 0.0
    for part in text.strip().split(":"):
        total = total * 60 + float(part)
    return total


def _add_layout_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--output-resolution", type=_size, default=None, metavar="WxH",
                   help="Output frame size; sets the aspect ratio (default 1080x1920 for 360 and "
                        "portrait sources, else 1920x1080)")
    p.add_argument("--lock-x", type=float, default=0.5, help="Lock point, fraction of width from left (default 0.5)")
    p.add_argument("--lock-y", type=float, default=None,
                   help="Lock point, fraction of height from top (default 0.2; 0.35 for 360)")
    p.add_argument("--scale", default="auto",
                   help="Flat only: clip scale as a fraction (e.g. 0.86) or 'auto' = smallest that always fits (default)")
    p.add_argument("--max-scale", type=float, default=1.0,
                   help="Flat only: cap for --scale auto (default 1.0 = never upsample the source)")
    p.add_argument("--fov", type=float, default=100.0,
                   help="360 only: field of view across the output's long side, degrees (default 100)")
    p.add_argument("--smooth", choices=METHODS, default="savgol", help="Zero-phase smoother (default savgol)")
    p.add_argument("--cutoff-hz", type=float, default=6.0,
                   help="Smoothing cutoff; higher = lighter (default 6 Hz, keeps moves slower than ~80 ms)")


def _add_target_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--target", choices=list(TARGETS), default="resolve",
                   help="Flat sources: DaVinci Resolve Fusion .setting (default) or Premiere Pro XML. "
                        "360 sources always render a video")


def _fill_defaults(track: Track, args) -> None:
    """Defaults that depend on the source: 360 selfies and portrait clips want a vertical frame."""
    is_360 = track.projection == "equirect"
    if args.output_resolution is None:
        portrait = is_360 or track.source_size[1] > track.source_size[0]
        args.output_resolution = (1080, 1920) if portrait else (1920, 1080)
    if args.lock_y is None:
        args.lock_y = 0.35 if is_360 else 0.2


def _smoothed(track: Track, args) -> np.ndarray:
    """Gap-filled, smoothed head: normalized frame coords (flat) or unit directions (360).

    360 heads are smoothed as 3D unit vectors, so a head crossing the +-180 deg seam is smooth too.
    """
    fps = float(track.fps)
    valid = ~np.isnan(track.head).any(axis=1)
    long_gap = int(round(0.5 * fps))
    for a, b in find_gaps(valid):
        if b - a >= long_gap:
            print(f"warning: no reliable head for frames {a}-{b - 1} ({(b - a) / fps:.2f}s); interpolated",
                  file=sys.stderr)
    if track.projection == "equirect":
        d = smooth(fill_gaps(sphere.uv_to_dirs(track.head)), fps, args.smooth, args.cutoff_hz)
        return d / np.linalg.norm(d, axis=1, keepdims=True)
    return smooth(fill_gaps(track.head), fps, args.smooth, args.cutoff_hz)


def _prepare(track: Track, args) -> tuple[np.ndarray, float, geometry.Feasibility]:
    """Flat sources: smooth, choose scale, run the feasibility check. Returns smoothed head (normalized)."""
    head = _smoothed(track, args)
    head_px = head * np.array(track.source_size, dtype=float)
    lock = (args.lock_x, args.lock_y)
    seq = args.output_resolution
    if args.scale == "auto":
        scale = geometry.auto_scale(head_px, track.source_size, seq, lock, max_scale=args.max_scale)
    else:
        scale = float(args.scale)
    return head, scale, geometry.check(head_px, track.source_size, seq, scale, lock)


def _default_output(base: str, track: Track, args) -> Path:
    if track.projection == "equirect":
        return Path(base + ".headlock.mp4")
    return Path(base + TARGETS[args.target])


def _export(track: Track, args, out: Path) -> None:
    _fill_defaults(track, args)
    if track.projection == "equirect":
        _export_360(track, args, out)
        return
    head, scale, feas = _prepare(track, args)
    print(feas.report())
    lock = (args.lock_x, args.lock_y)
    head_px = head * np.array(track.source_size, dtype=float)
    P = geometry.positions(head_px, track.source_size, args.output_resolution, scale, lock)
    if args.target == "resolve":
        resolve_fusion.write(out, resolve_fusion.build(P, args.output_resolution, scale))
    else:
        tree = premiere_xml.build(track.source, track.source_size, track.fps, P, args.output_resolution,
                                  scale, has_audio=track.has_audio)
        premiere_xml.write(out, tree)
    print(f"Wrote {out} ({len(P)} keyframes, scale {scale * 100:.2f}%)")
    if getattr(args, "preview", None):
        from .preview import render
        render(track.video, head, track.head, track.source_size, args.output_resolution, scale, lock,
               args.preview)
        print(f"Wrote preview {args.preview}")


def _export_360(track: Track, args, out: Path) -> None:
    from .preview import render_360

    if out.suffix.lower() not in (".mp4", ".mov"):
        sys.exit(f"error: 360 sources render a video; give -o a .mp4 or .mov name, not {out.name}")
    dirs = _smoothed(track, args)
    lock = (args.lock_x, args.lock_y)
    w, h = args.output_resolution
    print(f"Reframing 360 -> {w}x{h}, {args.fov:g} deg across the long side, head at ({lock[0]:g}, {lock[1]:g})")
    render_360(track.source, dirs, None, args.output_resolution, args.fov, lock, out, audio=track.has_audio)
    print(f"Wrote {out}")
    if getattr(args, "preview", None):
        render_360(track.video, dirs, sphere.uv_to_dirs(track.head), args.output_resolution, args.fov, lock,
                   args.preview, overlays=True)
        print(f"Wrote preview {args.preview}")


def cmd_track(args) -> None:
    from tqdm import tqdm

    from .head import head_centers
    from .pose import PoseEstimator
    from .video import iter_frames, probe

    info = probe(args.input)
    src = probe(args.source) if args.source else info
    if abs(info.width / info.height - src.width / src.height) > 0.01:
        sys.exit("error: --source aspect ratio differs from the tracked video")
    if info.fps != src.fps:
        sys.exit(f"error: --source frame rate {src.fps} differs from tracked video {info.fps}")
    projection = args.projection
    if projection == "auto":
        projection = "equirect" if info.width == 2 * info.height else "flat"
        if projection == "equirect":
            print("2:1 frame: treating it as a 360 (equirectangular) video; --projection flat if it isn't")
    elif projection == "360":
        projection = "equirect"

    est = PoseEstimator(args.model, args.device)
    frames = tqdm(iter_frames(args.input, max_height=1080), total=info.frame_count, desc="Tracking",
                  unit="f", disable=not sys.stderr.isatty())
    if projection == "equirect":
        from .track360 import VIEW_SIZE, track_equirect
        try:
            head, conf, fraction, kpts, scores = track_equirect(frames, est, args.min_conf)
        except ValueError as e:
            sys.exit(f"error: {e}")
        size = VIEW_SIZE  # keypoints are in tracking-view pixels
    else:
        kpts, scores, size = [], [], None
        for frame in frames:
            size = (frame.shape[1], frame.shape[0])
            k, s = est(frame)
            kpts.append(k)
            scores.append(s)
        if not kpts:
            sys.exit("error: no frames decoded")
        kpts, scores = np.array(kpts), np.array(scores)
        head, conf, fraction = head_centers(kpts, scores, args.min_conf)
        head = head / np.array(size)
    if args.verbose:
        print(f"{len(head)} frames tracked ({projection}); neck->top fraction {fraction:.3f}; "
              f"{np.isnan(head).any(axis=1).sum()} low-confidence frames")
    if args.keypoints:
        np.savez_compressed(args.keypoints, keypoints=kpts, scores=scores, size=np.array(size))

    track = Track(video=str(Path(args.input).resolve()), source=str(Path(src.path).resolve()),
                  source_size=(src.width, src.height), fps=info.fps, head=head,
                  conf=conf, fraction=fraction, has_audio=src.has_audio, projection=projection)
    base = str(Path(args.input).with_suffix(""))
    track_out = Path(args.track) if args.track else Path(base + ".track.json")
    track.save(track_out)
    print(f"Wrote {track_out}")
    _export(track, args, Path(args.output) if args.output else _default_output(base, track, args))


def cmd_export(args) -> None:
    track = Track.load(args.track)
    if args.source:
        from .video import probe
        src = probe(args.source)
        track.source, track.source_size, track.has_audio = (
            str(Path(args.source).resolve()), (src.width, src.height), src.has_audio)
    base = args.track.removesuffix(".json").removesuffix(".track")
    _export(track, args, Path(args.output) if args.output else _default_output(base, track, args))


def cmd_check(args) -> None:
    track = Track.load(args.track)
    if track.projection == "equirect":
        print("360 source: the view turns to follow the head, so there are no crop limits to check.")
        return
    _fill_defaults(track, args)
    _, _, feas = _prepare(track, args)
    print(feas.report())
    sys.exit(0 if feas.fits else 1)


def cmd_test_clip(args) -> None:
    from .preview import make_test_clip, make_test_clip_360

    if args.equirect:
        track = make_test_clip_360(args.output, args.size or (3840, 1920), Fraction(args.fps), args.seconds)
    else:
        track = make_test_clip(args.output, args.size or (3840, 2160), Fraction(args.fps), args.seconds)
    track_out = Path(str(Path(args.output).with_suffix("")) + ".track.json")
    track.save(track_out)
    print(f"Wrote {args.output} and {track_out}")
    if args.equirect:
        print(f"Next: headlock export {track_out} --preview check.mp4  -> the dot should sit still on the green cross")
    else:
        print(f"Next: headlock export {track_out}  -> load it in Resolve/Premiere; the dot should stay still")


def cmd_circles(args) -> None:
    from . import circles, resolve_api
    from .pose import PoseEstimator
    from .video import probe

    given = [(name, getattr(args, name)) for name in circles.NAMES if getattr(args, name)]
    if not given:
        sys.exit("error: give at least one chorus clip: --left, --middle and/or --right")
    main_info = probe(args.main)
    canvas = (main_info.width, main_info.height)
    if canvas[0] > canvas[1]:
        print(f"warning: the main video is landscape ({canvas[0]}x{canvas[1]}); circles go along its bottom anyway",
              file=sys.stderr)
    slots = {s.name: s for s in circles.slots(canvas, args.diameter, args.y, args.soft)}
    infos = {name: probe(path) for name, path in given}

    print("Lining the chorus clips up by their audio…")
    synced = circles.sync([circles.read_audio(path) if infos[name].has_audio else None for name, path in given],
                          [name for name, _ in given])
    for note in synced.notes:
        print(f"warning: {note}", file=sys.stderr)

    est = PoseEstimator(args.model, args.device)
    at = _seconds(args.at)
    placed = []
    for (name, path), offset, conf in zip(given, synced.offsets, synced.confidence):
        slot, info = slots[name], infos[name]
        framing = circles.find_framing(path, slot.diameter, est, desc=f"Finding the face ({name})")
        if framing.note:
            print(f"warning: {name}: {framing.note}", file=sys.stderr)
        text = circles.build_setting(slot, canvas, (info.width, info.height), framing)
        out = Path(path).with_suffix(f".{name}.circle.setting")
        out.write_text(text)
        sure = "" if conf == float("inf") else f" (sync {conf:.1f}x surer than the next best)"
        print(f"{name}: {Path(path).name} starts {offset:.3f}s after the earliest clip{sure}; "
              f"zoom {framing.zoom * 100:.0f}%. Wrote {out.name}")
        placed.append(resolve_api.Circle(str(Path(path).resolve()), slot, info.frame_count,
                                         round((at + offset) * float(main_info.fps)), text, str(out),
                                         info.has_audio))
    if args.no_resolve:
        print("Not building in Resolve (--no-resolve). By hand: put each clip on its own track above the main "
              "video at the offsets above, open it in the Fusion page, drag its .circle.setting into the Node "
              "Editor, connect MediaIn1 to CircleFace if it isn't, and CirclePlace to MediaOut1.")
        return

    job = resolve_api.Job(str(Path(args.main).resolve()), main_info.frame_count, main_info.fps, canvas, placed,
                          f"{Path(args.main).stem} circles")
    try:
        resolve, module = resolve_api.connect()
        name = resolve_api.build(resolve, job, module.readstring)
    except resolve_api.ResolveError as e:
        sys.exit(f"error: {e}\nThe .circle.setting files are written, so this can also be done by hand "
                 "(see --no-resolve).")
    for w in job.warnings:
        print(f"warning: {w}", file=sys.stderr)
    print(f"Built the timeline \"{name}\" in Resolve. The circles are linked: drag them together to where the "
          "chorus comes in. To resize one, change Zoom in the Inspector (it scales around the circle), or "
          "CircleZoom's Size in the Fusion page to zoom on the face.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="headlock",
                                     description="Head-lock stabilization for DaVinci Resolve and Premiere Pro")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("track", help="Track the head and export a Resolve/Premiere file (or render, for 360)")
    p.add_argument("input", help="Video to track (a 1080p proxy is fine)")
    p.add_argument("--source", help="Full-res media to use for the output (default: input)")
    p.add_argument("-o", "--output",
                   help="Output (default: <input>.headlock.setting / .xml, or .headlock.mp4 for 360)")
    p.add_argument("--track", help="track.json output (default: <input>.track.json)")
    p.add_argument("--projection", choices=["auto", "flat", "360"], default="auto",
                   help="360 = equirectangular video; auto treats 2:1 frames as 360 (default auto)")
    p.add_argument("--keypoints", help="Also save raw keypoints/scores to this .npz for debugging")
    p.add_argument("--min-conf", type=float, default=0.3, help="Neck/head-top confidence floor (default 0.3)")
    p.add_argument("--model", default="rtmpose-halpe26-384", choices=["rtmpose-halpe26-384", "rtmpose-halpe26-256"])
    p.add_argument("--device", default="cpu", choices=["cpu", "coreml", "cuda"])
    p.add_argument("--preview", help="Also render a stabilized preview .mp4 with tracking overlays")
    p.add_argument("-v", "--verbose", action="store_true")
    _add_target_arg(p)
    _add_layout_args(p)
    p.set_defaults(func=cmd_track)

    p = sub.add_parser("export", help="Re-export from a track.json without re-tracking")
    p.add_argument("track")
    p.add_argument("-o", "--output")
    p.add_argument("--source", help="Point the output at different full-res media")
    p.add_argument("--preview", help="Also render a stabilized preview .mp4 with tracking overlays")
    _add_target_arg(p)
    _add_layout_args(p)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("check", help="Report crop feasibility for a track.json")
    p.add_argument("track")
    _add_layout_args(p)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("test-clip", help="Make a synthetic moving-dot clip + exact track for validating output")
    p.add_argument("output")
    p.add_argument("--360", dest="equirect", action="store_true", help="Make a 360 (equirectangular) clip")
    p.add_argument("--size", type=_size, default=None, help="Default 3840x2160, or 3840x1920 with --360")
    p.add_argument("--fps", default="30000/1001")
    p.add_argument("--seconds", type=float, default=10.0)
    p.set_defaults(func=cmd_test_clip)

    p = sub.add_parser("circles", help="Chorus clips in three soft circles along the bottom, built in Resolve")
    p.add_argument("main", help="The main (vertical) video; sets the timeline's size and frame rate")
    p.add_argument("--left", help="Chorus clip for the left circle")
    p.add_argument("--middle", help="Chorus clip for the middle circle")
    p.add_argument("--right", help="Chorus clip for the right circle")
    p.add_argument("--at", default="0",
                   help="Where the chorus clips start on the timeline, e.g. 1:23.4 (default 0; drag them after)")
    p.add_argument("--diameter", type=float, default=0.30, help="Circle size, fraction of the width (default 0.30)")
    p.add_argument("--y", type=float, default=0.85,
                   help="Height of the circles' centres, fraction from the top (default 0.85)")
    p.add_argument("--soft", type=float, default=0.04, help="Soft edge, fraction of the circle size (default 0.04)")
    p.add_argument("--no-resolve", action="store_true",
                   help="Only write the .circle.setting files and print the sync; don't build in Resolve")
    p.add_argument("--model", default="rtmpose-halpe26-384", choices=["rtmpose-halpe26-384", "rtmpose-halpe26-256"])
    p.add_argument("--device", default="cpu", choices=["cpu", "coreml", "cuda"])
    p.set_defaults(func=cmd_circles)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
