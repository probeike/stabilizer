# headlock

Pins a head to a fixed spot in the frame. Tracks the head on every frame (RTMPose Halpe26),
smooths it lightly with a zero-phase filter, and then:

- **flat video** (a normal camera clip of a dance): checks the crop fits and writes per-frame
  keyframes for **DaVinci Resolve** (a Fusion `.setting`) or **Premiere Pro** (FCP7 XML);
- **360 video** (e.g. an Insta360 X5 equirectangular export of a running selfie): turns a virtual
  camera inside the sphere to keep the head on the lock point and renders the flat, head-locked
  video (9:16 by default). A 360 frame has no edges, so nothing is ever cropped past the picture.

It also builds **chorus circles** in DaVinci Resolve: three extra clips of you singing the chorus,
each in a soft-edged circle along the bottom of the main vertical video, lined up with each other
by their audio. And it adds **karaoke captions** of your lyrics, each word lighting up as it's sung.

## Setup

```sh
uv sync            # Python 3.12 env with rtmlib/onnxruntime/opencv/scipy
```
Requires `ffmpeg`/`ffprobe` on PATH (`brew install ffmpeg`). Pose models (~300 MB) download on first
run and are cached.

## The app

Double-click **`Headlock.command`** in Finder (or run `uv run headlock-ui`). It has three tabs:

- **Head-lock**: choose a video; the app spots 360 footage (2:1 frames) and portrait clips and sets
  sensible defaults. Press **Track**. When it finishes, the **framing preview** opens: it plays the
  clip as it will come out while you change the field of view, head height, smoothing or output
  frame (space plays/pauses, arrows step a frame, *Show tracking marks* draws the lock point and the
  raw tracking). Like Insta360 Studio's preview it redraws every frame through the virtual camera as
  it plays, only smaller and silent. Press **Export** when it looks right; the tracking is kept, so
  you can preview and export again without re-tracking.
- **Chorus circles**: choose the main video and the left, middle and right chorus clips, then
  **Build in Resolve** (see below). **Add Headlock to Resolve's Scripts menu** puts a *Headlock* item
  under Workspace > Scripts in Resolve that opens this window.
- **Captions**: paste the lyrics and caption the timeline open in Resolve (see below).
- **Check & test**: the fit check for a tracked clip, and the synthetic test clips used to validate
  the Resolve/Premiere output.

**Show in Finder** reveals the result. Everything below can also be done from the command line; the
app runs the same code.

## Flat video → DaVinci Resolve

```sh
uv run headlock track take.mp4 --output-resolution 1080x1920 --device coreml
```
Writes next to the video:
- `take.headlock.setting`: Fusion nodes that pin the head (a Background the size of the output and
  a Merge with the clip's Center keyframed on every frame).
- `take.track.json`: raw head track, so you can re-export without re-tracking.

In Resolve:
1. Make a timeline at the output resolution (timeline settings → uncheck *Use Project Settings* →
   1080 x 1920) and the clip's frame rate. Put the **whole clip** on it, from its first frame.
2. Select the clip and open the **Fusion** page.
3. Drag `take.headlock.setting` from Finder into the Node Editor (or open it in TextEdit, copy all,
   and paste into the Node Editor).
4. `HeadlockMerge`'s green foreground input should already be connected to `MediaIn1` (connect it
   if not). Drag `HeadlockMerge`'s output onto `MediaOut1`.

Back on the Edit page the head is pinned. Cut and trim the clip after this, not before: keyframe *n*
is frame *n* of the clip in Fusion.

For Premiere Pro instead: `--target premiere` writes `take.headlock.xml`; import it (File → Import)
and it opens as a sequence with the clip scaled and keyframed. To move the motion onto a clip in an
existing edit: select the clip → Copy → select your clip → Edit → Paste Attributes → Motion.

Tracking a proxy and linking the 4K original: `headlock track proxy.mp4 --source original.mp4`
(same framing and frame rate). `--preview take_preview.mp4` renders the result locally (green cross =
lock point, red dot = raw unsmoothed head).

## 360 video (Insta360 X5) → head-locked vertical video

1. In Insta360 Studio, export the clip as a **360° video** (a 2:1 equirectangular MP4, not the raw
   `.insv`), with FlowState stabilization and horizon lock on. headlock turns the view left/right
   and up/down but never rolls it, so a level horizon stays level.
2. Track and render:
   ```sh
   uv run headlock track selfie.mp4 --device coreml
   ```
   2:1 frames are treated as 360 (`--projection flat` overrides). The head is tracked through a
   normal-looking perspective view that follows it, so the ±180° seam and the stretched poles don't
   confuse the pose model. Writes `selfie.headlock.mp4` (1080x1920, 100° field of view across the
   long side, head 35% down from the top, the clip's audio) and `selfie.track.json`.
3. Change the framing without re-tracking:
   ```sh
   uv run headlock export selfie.track.json --fov 90 --lock-y 0.4
   ```

`-o selfie.headlock.mov` writes ProRes 422 HQ 10-bit, decoded at 16 bits so Insta360 log footage
survives for grading in Resolve. To do the crop in Resolve instead, render larger and wider (e.g.
`--output-resolution 2160x3840 --fov 120`), put that on a 1080x1920 timeline and set a fixed
zoom/position: the head is already locked, so any fixed crop stays locked. Track a small export and
render from the full-resolution one with `--source`. Spatial (4-channel) audio is passed through as is.

## Chorus circles (DaVinci Resolve)

The main vertical video on V1, and three clips of you singing the chorus in equal circles along the
bottom (centred at 1/6, 1/2 and 5/6 of the width), each with a slightly soft edge.

```sh
uv run headlock circles song.mp4 --left chorus1.mov --middle chorus2.mov --right chorus3.mov --device coreml
```
It:
1. lines the three chorus clips up **with each other** by their audio (where the notes and words
   start, so different harmony parts still line up). It doesn't try to match them to the main video,
   which sings different parts. A warning means the match was uncertain: check it by ear;
2. finds each singer's head with the pose model (on 40 frames across the clip) and frames the circle
   on it, with the head filling about half the circle;
3. writes `chorus1.left.circle.setting` (and so on) next to each clip;
4. builds a new timeline in the open Resolve project, `song circles`: the main video on V1/A1 and
   the chorus clips on V2-V4, their audio on A2-A4 so you can mix the extra voices in. The six chorus
   items are **linked**: drag them together to where the chorus comes in (or give `--at 1:23.4`).

Each circle is made in the clip's Fusion comp (`CircleFace` → `CircleZoom` → `CirclePlace` →
`MediaOut1`), so the clip comes out as the full frame, transparent outside its circle. To adjust:
- **Zoom on the face**: Fusion page, `CircleZoom` → Size. It zooms around the face, which stays centred.
- **Make a whole circle bigger or smaller**: Edit page, Inspector → Zoom. The clip's Anchor Point is
  set to the circle's centre, so it grows in place.
- Circle size, height and softness for all three: `--diameter 0.30` (fraction of the width; 0.33
  touches), `--y 0.85` (centre height from the top), `--soft 0.04` (fraction of the circle size).

Resolve setup (once): it must be **DaVinci Resolve Studio** (the free version can't be driven from
outside), with Preferences → System → General → **External scripting using: Local**, and a project
open. `--no-resolve` skips Resolve and only writes the `.circle.setting` files and prints the sync;
by hand, put each clip on its own track, open it in the Fusion page, drag its `.circle.setting` into
the Node Editor, connect `MediaIn1` to `CircleFace` if it isn't already, and `CirclePlace` to `MediaOut1`.

## Karaoke captions (DaVinci Resolve)

The lyrics a line at a time in the band above your head, in IM Fell English: the whole line is
shown dimmed, and each word lights up as you sing it and stays lit.

1. Align your clips, with the lead vocal on audio track 1, and keep that timeline open in Resolve.
2. Once per project: in the Edit page's Effects library, drag **Titles → Text+** into the Media Pool.
3. In the app's **Captions** tab, paste the lyrics (one caption per line, as you'd like it shown;
   blank lines and `[Chorus]`-style headings are skipped) and press **Caption the current timeline**.
   Or: `uv run headlock captions --lyrics lyrics.txt`.

No timing needed: it listens to audio track 1 through your edit with Whisper (`large-v3-turbo`,
downloaded on first use, about 1.6 GB), which gets the lyrics as hints, and then lines what it heard
up with your lyrics. So the words shown are always yours, even where it misheard one ("seeing" for
"sing"). Words it didn't catch are spread between their neighbours. Lines it can't find in the
recording are left out and listed. Each line becomes a title on a video track called **Captions**,
and captioning again replaces them.

The timing is also saved as `<timeline>.lrc` next to the main video, one line per caption:
`[00:12.30]<00:12.55>I <00:12.80>keep <00:13.10>running <00:13.90>`. To fix a word, change its time
there and press **Place again from a timing file** (`--timing song.lrc`), which doesn't listen again.
To restyle without listening again, use the same button. Options: `--font`, `--size` (Text+ Size,
default 0.07; lines over 24 characters go on two rows), `--y` (centre height from the top, default
0.13; the head sits at 0.35 in 360 head-locks), `--track` (vocal track), `--model`, `--language`.

In Fusion each title holds `CaptionBase` (the dimmed line) under `CaptionLit` (the same line, lit),
whose Write On end steps forward as each word starts.

## Framing options (`track` and `export`)
| Flag | Default | Meaning |
|---|---|---|
| `--target` | `resolve` | Flat sources: `resolve` (Fusion `.setting`) or `premiere` (FCP7 XML) |
| `--output-resolution` | `1920x1080`; `1080x1920` for 360 and portrait sources | Output size, sets the aspect ratio |
| `--lock-x`, `--lock-y` | `0.5`, `0.2` (`0.35` for 360) | Where the head is pinned, fractions from top-left |
| `--fov` | `100` | 360 only: field of view across the output's long side, degrees |
| `--scale` | `auto` | Flat only: clip scale (`0.86` = 86%); `auto` = smallest that never shows past the source edge |
| `--max-scale` | `1.0` | Flat only: cap for auto (1.0 = never upsample the source) |
| `--smooth` | `savgol` | `savgol`, `butter`, or `none` |
| `--cutoff-hz` | `6` | Higher = lighter smoothing. Lower if the head still jitters; raise if it slides on fast whips |

```sh
headlock export take.track.json --target premiere --output-resolution 1080x1920 -o take_vertical.xml
headlock check  take.track.json --scale 0.9                    # flat: fit report only
```

## Validating the output

The Resolve and Premiere mappings follow each app's documented coordinate rules but haven't been
confirmed inside the apps yet. Check once with a synthetic clip whose track is exact:

```sh
headlock test-clip dot.mp4                                     # 4K grid + dot on a known path
headlock export dot.track.json --smooth none --output-resolution 1080x1920
```
Load `dot.headlock.setting` as above (or `--target premiere` and import the XML): the dot should sit
still at the lock point. In Resolve, also trim the clip's head afterwards and check the dot stays
still. If it drifts or moves the wrong way, the fix belongs in `resolve_fusion.position_to_center`
(or `premiere_xml.position_to_center`).

360: `headlock test-clip dot360.mp4 --360` then `headlock export dot360.track.json --smooth none
--preview check.mp4`. The dot wanders back and forth across the seam behind the camera and should
sit on the green cross throughout.

## Tests

```sh
uv run pytest
```
