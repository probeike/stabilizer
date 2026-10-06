# headlock

Pins a dancer's head to a fixed spot in the frame. Tracks the head on every frame (RTMPose Halpe26),
smooths it lightly with a zero-phase filter, checks the crop fits, and writes a **Final Cut Pro 7 XML**
that Premiere Pro imports as a sequence with per-frame Motion keyframes.

## Setup

```sh
uv sync            # Python 3.12 env with rtmlib/onnxruntime/opencv/scipy
```
Requires `ffmpeg`/`ffprobe` on PATH. Pose models (~300 MB) download on first run and are cached.

## Use

```sh
uv run headlock track take.mp4 --preview take_preview.mp4 --device coreml
```
Writes next to the video:
- `take.headlock.xml` — import into Premiere (File → Import). Opens as a new sequence with the clip
  scaled and keyframed so the head stays at the lock point.
- `take.track.json` — raw head track, so you can re-export without re-tracking.
- `take_preview.mp4` (optional) — the stabilized result rendered locally. Green cross = lock point,
  red dot = raw unsmoothed head estimate.

To get the motion onto a clip in an existing edit: in the imported sequence, select the clip → Copy →
select your clip → Edit → Paste Attributes → Motion.

Tracking a proxy and linking the 4K original: `headlock track proxy.mp4 --source original.mp4`
(same framing and frame rate).

### Framing options (work on `track` and `export`)
| Flag | Default | Meaning |
|---|---|---|
| `--output-resolution` | `1920x1080` | Sequence size (e.g. `1080x1920` for vertical) |
| `--lock-x`, `--lock-y` | `0.5`, `0.2` | Where the head is pinned, fractions from top-left |
| `--scale` | `auto` | Clip scale (`0.86` = 86%); `auto` = smallest that never shows past the source edge |
| `--max-scale` | `1.0` | Cap for auto (1.0 = never upsample the source) |
| `--smooth` | `savgol` | `savgol`, `butter`, or `none` |
| `--cutoff-hz` | `6` | Higher = lighter smoothing. Lower if the head still jitters; raise if it slides on fast whips |

```sh
headlock export take.track.json --output-resolution 1080x1920 -o take_vertical.xml   # re-export
headlock check  take.track.json --scale 0.9                                         # fit report only
```

## Validating the Premiere XML mapping

The xmeml Center-unit mapping (`premiere_xml.position_to_center`) is not yet confirmed against
Premiere. Before trusting real output, validate once:

```sh
headlock test-clip dot.mp4                 # 4K grid clip + dot on a known path, and dot.track.json
headlock export dot.track.json --smooth none --preview dot_preview.mp4
```
Import `dot.headlock.xml` into Premiere: the dot should sit still at the top-centre lock point,
matching `dot_preview.mp4`. If it drifts or moves the wrong way, export any clip with keyframed
Position from Premiere as FCP XML and compare its `center` values to fix the mapping.

## Tests

```sh
uv run pytest
```
