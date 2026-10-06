"""FCP7 XML (xmeml v4) export that Premiere Pro imports as a sequence with Motion keyframes.

UNVERIFIED MAPPING: ``position_to_center`` encodes our best understanding of how Premiere maps
Motion > Position (sequence pixels) to xmeml Basic Motion ``center`` values. Confirm it with the
format spike (export a hand-keyframed clip from Premiere as FCP XML) and the synthetic marker
round-trip (``headlock test-clip``) before trusting output; adjust only this function if it's off.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path
from urllib.parse import quote

import numpy as np


def position_to_center(P: np.ndarray, seq_size: tuple[int, int]) -> np.ndarray:
    """Premiere Position (sequence px, origin top-left) -> xmeml center (0,0 = frame centre)."""
    seq = np.array(seq_size, dtype=float)
    return (np.asarray(P, dtype=float) - seq / 2) / seq


def _sub(parent: ET.Element, tag: str, text: object | None = None, **attrib) -> ET.Element:
    el = ET.SubElement(parent, tag, attrib)
    if text is not None:
        el.text = str(text)
    return el


def _rate(parent: ET.Element, fps: Fraction) -> None:
    rate = _sub(parent, "rate")
    _sub(rate, "timebase", round(float(fps)))
    _sub(rate, "ntsc", "TRUE" if fps.denominator == 1001 else "FALSE")


def _video_chars(parent: ET.Element, fps: Fraction, size: tuple[int, int]) -> None:
    sc = _sub(parent, "samplecharacteristics")
    _rate(sc, fps)
    _sub(sc, "width", size[0])
    _sub(sc, "height", size[1])
    _sub(sc, "anamorphic", "FALSE")
    _sub(sc, "pixelaspectratio", "square")
    _sub(sc, "fielddominance", "none")


def _pathurl(path: str | Path) -> str:
    return "file://localhost" + quote(str(Path(path).resolve()))


def _basic_motion(parent: ET.Element, scale: float, centers: np.ndarray) -> None:
    effect = _sub(_sub(parent, "filter"), "effect")
    _sub(effect, "name", "Basic Motion")
    _sub(effect, "effectid", "basic")
    _sub(effect, "effectcategory", "motion")
    _sub(effect, "effecttype", "motion")
    _sub(effect, "mediatype", "video")

    p = _sub(effect, "parameter", authoringApp="PremierePro")
    _sub(p, "parameterid", "scale")
    _sub(p, "name", "Scale")
    _sub(p, "valuemin", 0)
    _sub(p, "valuemax", 1000)
    _sub(p, "value", f"{scale * 100:.4f}")

    p = _sub(effect, "parameter", authoringApp="PremierePro")
    _sub(p, "parameterid", "center")
    _sub(p, "name", "Center")
    v = _sub(p, "value")
    _sub(v, "horiz", f"{centers[0, 0]:.6f}")
    _sub(v, "vert", f"{centers[0, 1]:.6f}")
    for i, (h, vv) in enumerate(centers):
        kf = _sub(p, "keyframe")
        _sub(kf, "when", i)
        val = _sub(kf, "value")
        _sub(val, "horiz", f"{h:.6f}")
        _sub(val, "vert", f"{vv:.6f}")


def build(source: str | Path, source_size: tuple[int, int], fps: Fraction, positions_px: np.ndarray,
          seq_size: tuple[int, int], scale: float, has_audio: bool = False,
          name: str | None = None) -> ET.ElementTree:
    """One sequence, one clip spanning the whole source, with per-frame Position keyframes."""
    n = len(positions_px)
    name = name or f"{Path(source).stem} headlock"
    centers = position_to_center(positions_px, seq_size)

    root = ET.Element("xmeml", version="4")
    seq = _sub(root, "sequence", id="sequence-1")
    _sub(seq, "name", name)
    _sub(seq, "duration", n)
    _rate(seq, fps)
    media = _sub(seq, "media")

    video = _sub(media, "video")
    _video_chars(_sub(video, "format"), fps, seq_size)
    clip = _sub(_sub(video, "track"), "clipitem", id="clipitem-1")
    _sub(clip, "name", Path(source).name)
    _sub(clip, "enabled", "TRUE")
    _sub(clip, "duration", n)
    _rate(clip, fps)
    for tag, val in (("start", 0), ("end", n), ("in", 0), ("out", n)):
        _sub(clip, tag, val)
    f = _sub(clip, "file", id="file-1")
    _sub(f, "name", Path(source).name)
    _sub(f, "pathurl", _pathurl(source))
    _rate(f, fps)
    _sub(f, "duration", n)
    fmedia = _sub(f, "media")
    _video_chars(_sub(fmedia, "video"), fps, source_size)
    if has_audio:
        a = _sub(fmedia, "audio")
        _sub(a, "channelcount", 2)
    _basic_motion(clip, scale, centers)

    if has_audio:
        audio = _sub(media, "audio")
        aclip = _sub(_sub(audio, "track"), "clipitem", id="clipitem-2")
        _sub(aclip, "name", Path(source).name)
        _sub(aclip, "enabled", "TRUE")
        _sub(aclip, "duration", n)
        _rate(aclip, fps)
        for tag, val in (("start", 0), ("end", n), ("in", 0), ("out", n)):
            _sub(aclip, tag, val)
        _sub(aclip, "file", id="file-1")
        st = _sub(aclip, "sourcetrack")
        _sub(st, "mediatype", "audio")
        _sub(st, "trackindex", 1)

    ET.indent(root, space="  ")
    return ET.ElementTree(root)


def write(path: str | Path, tree: ET.ElementTree) -> None:
    with open(path, "wb") as fh:
        fh.write(b'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n')
        tree.write(fh, encoding="utf-8", xml_declaration=False)
