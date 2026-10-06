"""headlock command line: track, check, export, test-clip."""

from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import geometry, premiere_xml
from .smooth import METHODS, fill_gaps, find_gaps, smooth
from .trackio import Track


def _size(text: str) -> tuple[int, int]:
    w, h = text.lower().split("x")
    return int(w), int(h)


def _add_layout_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--output-resolution", type=_size, default=(1920, 1080), metavar="WxH",
                   help="Sequence size; sets the aspect ratio (default 1920x1080, e.g. 1080x1920 for 9:16)")
    p.add_argument("--lock-x", type=float, default=0.5, help="Lock point, fraction of width from left (default 0.5)")
    p.add_argument("--lock-y", type=float, default=0.2, help="Lock point, fraction of height from top (default 0.2)")
    p.add_argument("--scale", default="auto",
                   help="Clip scale as a fraction (e.g. 0.86) or 'auto' = smallest that always fits (default)")
    p.add_argument("--max-scale", type=float, default=1.0,
                   help="Cap for --scale auto (default 1.0 = never upsample the source)")
    p.add_argument("--smooth", choices=METHODS, default="savgol", help="Zero-phase smoother (default savgol)")
    p.add_argument("--cutoff-hz", type=float, default=6.0,
                   help="Smoothing cutoff; higher = lighter (default 6 Hz, keeps moves slower than ~80 ms)")


def _prepare(track: Track, args) -> tuple[np.ndarray, float, geometry.Feasibility]:
    """Fill gaps, smooth, choose scale, run the feasibility check. Returns smoothed head (normalized)."""
    fps = float(track.fps)
    valid = ~np.isnan(track.head).any(axis=1)
    long_gap = int(round(0.5 * fps))
    for a, b in find_gaps(valid):
        if b - a >= long_gap:
            print(f"warning: no reliable head for frames {a}-{b - 1} ({(b - a) / fps:.2f}s); interpolated",
                  file=sys.stderr)
    head = smooth(fill_gaps(track.head), fps, args.smooth, args.cutoff_hz)
    head_px = head * np.array(track.source_size, dtype=float)
    lock = (args.lock_x, args.lock_y)
    seq = args.output_resolution
    if args.scale == "auto":
        scale = geometry.auto_scale(head_px, track.source_size, seq, lock, max_scale=args.max_scale)
    else:
        scale = float(args.scale)
    return head, scale, geometry.check(head_px, track.source_size, seq, scale, lock)


def _export(track: Track, args, xml_out: Path) -> None:
    head, scale, feas = _prepare(track, args)
    print(feas.report())
    lock = (args.lock_x, args.lock_y)
    head_px = head * np.array(track.source_size, dtype=float)
    P = geometry.positions(head_px, track.source_size, args.output_resolution, scale, lock)
    tree = premiere_xml.build(track.source, track.source_size, track.fps, P, args.output_resolution,
                              scale, has_audio=track.has_audio)
    premiere_xml.write(xml_out, tree)
    print(f"Wrote {xml_out} ({len(P)} keyframes, scale {scale * 100:.2f}%)")
    if getattr(args, "preview", None):
        from .preview import render
        render(track.video, head, track.head, track.source_size, args.output_resolution, scale, lock,
               args.preview)
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

    est = PoseEstimator(args.model, args.device)
    kpts, scores, size = [], [], None
    for frame in tqdm(iter_frames(args.input, max_height=1080), total=info.frame_count,
                      unit="f", disable=not sys.stderr.isatty()):
        size = (frame.shape[1], frame.shape[0])
        k, s = est(frame)
        kpts.append(k)
        scores.append(s)
    if not kpts:
        sys.exit("error: no frames decoded")
    kpts, scores = np.array(kpts), np.array(scores)
    head, conf, fraction = head_centers(kpts, scores, args.min_conf)
    if args.verbose:
        print(f"{len(head)} frames tracked at {size[0]}x{size[1]}; neck->top fraction {fraction:.3f}; "
              f"{np.isnan(head).any(axis=1).sum()} low-confidence frames")
    if args.keypoints:
        np.savez_compressed(args.keypoints, keypoints=kpts, scores=scores, size=np.array(size))

    track = Track(video=str(Path(args.input).resolve()), source=str(Path(src.path).resolve()),
                  source_size=(src.width, src.height), fps=info.fps, head=head / np.array(size),
                  conf=conf, fraction=fraction, has_audio=src.has_audio)
    base = str(Path(args.input).with_suffix(""))
    track_out = Path(args.track) if args.track else Path(base + ".track.json")
    track.save(track_out)
    print(f"Wrote {track_out}")
    _export(track, args, Path(args.output) if args.output else Path(base + ".headlock.xml"))


def cmd_export(args) -> None:
    track = Track.load(args.track)
    if args.source:
        from .video import probe
        src = probe(args.source)
        track.source, track.source_size, track.has_audio = (
            str(Path(args.source).resolve()), (src.width, src.height), src.has_audio)
    out = Path(args.output) if args.output else Path(args.track.removesuffix(".json").removesuffix(".track") + ".headlock.xml")
    _export(track, args, out)


def cmd_check(args) -> None:
    track = Track.load(args.track)
    _, _, feas = _prepare(track, args)
    print(feas.report())
    sys.exit(0 if feas.fits else 1)


def cmd_test_clip(args) -> None:
    from .preview import make_test_clip

    track = make_test_clip(args.output, args.size, Fraction(args.fps), args.seconds)
    track_out = Path(str(Path(args.output).with_suffix("")) + ".track.json")
    track.save(track_out)
    print(f"Wrote {args.output} and {track_out}")
    print(f"Next: headlock export {track_out}  -> import the XML in Premiere; the dot should stay still")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="headlock", description="Head-lock stabilization for Premiere Pro")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("track", help="Track the head in a video and export a Premiere XML")
    p.add_argument("input", help="Video to track (a 1080p proxy is fine)")
    p.add_argument("--source", help="Full-res media for the XML to reference (default: input)")
    p.add_argument("-o", "--output", help="XML output (default: <input>.headlock.xml)")
    p.add_argument("--track", help="track.json output (default: <input>.track.json)")
    p.add_argument("--keypoints", help="Also save raw keypoints/scores to this .npz for debugging")
    p.add_argument("--min-conf", type=float, default=0.3, help="Neck/head-top confidence floor (default 0.3)")
    p.add_argument("--model", default="rtmpose-halpe26-384", choices=["rtmpose-halpe26-384", "rtmpose-halpe26-256"])
    p.add_argument("--device", default="cpu", choices=["cpu", "coreml", "cuda"])
    p.add_argument("--preview", help="Also render a stabilized preview .mp4")
    p.add_argument("-v", "--verbose", action="store_true")
    _add_layout_args(p)
    p.set_defaults(func=cmd_track)

    p = sub.add_parser("export", help="Re-export an XML from a track.json without re-tracking")
    p.add_argument("track")
    p.add_argument("-o", "--output")
    p.add_argument("--source", help="Point the XML at different full-res media")
    p.add_argument("--preview", help="Also render a stabilized preview .mp4")
    _add_layout_args(p)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("check", help="Report crop feasibility for a track.json")
    p.add_argument("track")
    _add_layout_args(p)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("test-clip", help="Make a synthetic moving-dot clip + exact track for validating the XML")
    p.add_argument("output")
    p.add_argument("--size", type=_size, default=(3840, 2160))
    p.add_argument("--fps", default="30000/1001")
    p.add_argument("--seconds", type=float, default=10.0)
    p.set_defaults(func=cmd_test_clip)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
