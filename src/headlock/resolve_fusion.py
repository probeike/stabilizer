"""Fusion ``.setting`` export for DaVinci Resolve: a Background + Merge that pins the head.

Dropped into the Fusion page's Node Editor for the clip, it adds a Background the size of the
output frame and a Merge that lays the clip (MediaIn1) over it at a fixed Size, with Center
keyframed on every frame. Wire the Merge into MediaOut1.

Fusion behaviour this relies on (DaVinci Resolve manual, Fusion chapters):
- a single timeline clip opened in Fusion arrives in MediaIn at its source resolution;
- a Merge outputs at the Background's resolution, Size 1.0 maps one foreground pixel to one
  background pixel, and Center is where the foreground's centre lands in normalized background
  coordinates (0-1, origin bottom-left, Y up).
So for the Position ``P`` from ``geometry.positions`` (sequence px, origin top-left, Y down),
Center = (Px / W, 1 - Py / H) and Size = scale. Keyframe ``i`` is frame ``i`` of the Fusion clip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MEDIA_IN = "MediaIn1"


def position_to_center(P: np.ndarray, seq_size: tuple[int, int]) -> np.ndarray:
    """Position (sequence px, origin top-left, Y down) -> Fusion Center (normalized, Y up)."""
    c = np.asarray(P, dtype=float) / np.array(seq_size, dtype=float)
    c[:, 1] = 1.0 - c[:, 1]
    return c


def _spline(name: str, color: str, values: np.ndarray) -> str:
    keys = ",\n".join(f"\t\t\t\t[{i}] = {{ {v:.7f}, Flags = {{ Linear = true }} }}"
                      for i, v in enumerate(values))
    return (f"\t\t{name} = BezierSpline {{\n"
            f"\t\t\tSplineColor = {{ {color} }},\n"
            f"\t\t\tNameSet = true,\n"
            f"\t\t\tKeyFrames = {{\n{keys}\n\t\t\t}}\n"
            f"\t\t}},\n")


def build(positions_px: np.ndarray, seq_size: tuple[int, int], scale: float) -> str:
    """The .setting text: Lua table of tools, the same text Fusion puts on the clipboard."""
    centers = position_to_center(positions_px, seq_size)
    w, h = seq_size
    return (
        "{\n"
        "\tTools = ordered() {\n"
        "\t\tHeadlockCanvas = Background {\n"
        "\t\t\tNameSet = true,\n"
        "\t\t\tInputs = {\n"
        "\t\t\t\tUseFrameFormatSettings = Input { Value = 0, },\n"
        f"\t\t\t\tWidth = Input {{ Value = {w}, }},\n"
        f"\t\t\t\tHeight = Input {{ Value = {h}, }},\n"
        "\t\t\t\tTopLeftAlpha = Input { Value = 1, },\n"
        "\t\t\t},\n"
        "\t\t\tViewInfo = OperatorInfo { Pos = { 0, -66 } },\n"
        "\t\t},\n"
        "\t\tHeadlockMerge = Merge {\n"
        "\t\t\tNameSet = true,\n"
        "\t\t\tInputs = {\n"
        "\t\t\t\tBackground = Input { SourceOp = \"HeadlockCanvas\", Source = \"Output\", },\n"
        f"\t\t\t\tForeground = Input {{ SourceOp = \"{MEDIA_IN}\", Source = \"Output\", }},\n"
        "\t\t\t\tCenter = Input { SourceOp = \"HeadlockCenter\", Source = \"Value\", },\n"
        f"\t\t\t\tSize = Input {{ Value = {scale:.7f}, }},\n"
        "\t\t\t},\n"
        "\t\t\tViewInfo = OperatorInfo { Pos = { 110, 0 } },\n"
        "\t\t},\n"
        "\t\tHeadlockCenter = XYPath {\n"
        "\t\t\tShowKeyPoints = false,\n"
        "\t\t\tDrawMode = \"ModifyOnly\",\n"
        "\t\t\tInputs = {\n"
        "\t\t\t\tX = Input { SourceOp = \"HeadlockCenterX\", Source = \"Value\", },\n"
        "\t\t\t\tY = Input { SourceOp = \"HeadlockCenterY\", Source = \"Value\", },\n"
        "\t\t\t},\n"
        "\t\t},\n"
        + _spline("HeadlockCenterX", "Red = 255, Green = 0, Blue = 0", centers[:, 0])
        + _spline("HeadlockCenterY", "Red = 0, Green = 255, Blue = 0", centers[:, 1])
        + "\t},\n"
        "\tActiveTool = \"HeadlockMerge\"\n"
        "}\n"
    )


def write(path: str | Path, text: str) -> None:
    Path(path).write_text(text)
