# Plan: chorus circles + one app for everything

Status: **implemented on 2026-10-13** (`circles.py`, `resolve_api.py`, `headlock circles`, the tabbed
app), tested against a fake Resolve API; **not yet run inside Resolve**. Do the "Verify in Resolve"
list below on the target machine (`target-machine.md`) before relying on it.

## What was built, and where it differs from the design below
- `.circle.setting` files are named per slot: `<clip>.<left|middle|right>.circle.setting`.
- The square the circle is cut from is a little bigger than the circle (`D = d·(1 + 2·soft)`, even),
  so the soft edge has room; the mask's Width/Height = d/D, SoftEdge = soft·d/D. The visible
  (half-transparent) edge is at the requested diameter.
- Sync: the onset envelope keeps only its peaks and widens each to 70 ms (a max filter) before the
  cross-correlation. Separate takes never start words at exactly the same moment; plain spiky
  envelopes lost the match with ±40 ms of slop, blurring flattened the true peak below chance
  matches, and widening holds it to under a frame with ±60 ms slop (see `tests/test_circles.py`).
  Zero-padding the audio was also dropped: silence before the first sample reads as an onset at
  every take's start and they lined up with each other at lag 0.
- Framing never zooms out so far that the circle would show past the picture's edge.
- The Resolve build uses `Timeline.SetSetting` / `TimelineItem.SetProperty` one key at a time. It
  tries `endFrame` = clip length and then length - 1 (inclusive or exclusive isn't documented),
  reuses media already in the pool, gives the timeline a unique name, explicitly connects
  `CircleFace` to `MediaIn1` and `CirclePlace` to `MediaOut1` after pasting, and reads back
  `CircleMask`'s `MaskWidth`/`MaskHeight` to warn if those input IDs are wrong.
- The Scripts-menu item and "Add Headlock to Resolve's Scripts menu" button are in the Chorus
  circles tab; Check & test holds the fit check and test clips.
- `video.probe` now reports the upright size of phone clips stored sideways with a rotation flag
  (ffmpeg decodes them upright). Before, a portrait iPhone clip read as landscape, which would have
  made a landscape timeline and garbled the decoded frames (this affected `track` too).

## What the user wants
1. **Chorus circles.** A vertical video of them singing (the *main* video, on V1). Three more clips of
   them singing the chorus into a microphone go in three equal circles along the bottom, centred at
   x = 1/6, 1/2 and 5/6 of the width, with slightly soft (blurred) edges. The user labels which clip is
   left/middle/right, and the tool builds it in a DaVinci Resolve timeline.
2. **Decisions the user made:**
   - **Sync:** only sync the three chorus clips *to each other* (by their audio). Do **not** try to
     match them to the main video: it sings different parts, so matching wouldn't be reliable. The user
     drags the group into place by hand, so link the three clips together so they move as one.
   - **Framing:** fixed (not following), centred on the head, found automatically with the pose
     tracker. Zooming must happen around the circle's centre so the user can rescale by hand.
     (Resolve's Edit-page zoom currently scales from the frame centre, which pushes the circle off
     its slot.)
   - **Audio:** keep the chorus clips' audio on their own tracks (A2–A4) so the user can mix the extra
     voices in.
3. **One app for everything.** A basic UI exposing all of headlock (single-dancer flat tracking, 360,
   re-export, chorus circles, fit check, test clip), so the user picks what to do instead of
   memorising commands. Ideally it can also be opened **from inside Resolve** (Workspace → Scripts).

## Design

### Pieces to add
- `src/headlock/circles.py` (pure, testable):
  - slot geometry: canvas = main video size (expected 1080x1920). Centres at x = W/6, W/2, 5W/6.
    Diameter d = 0.30·W by default (324 px at 1080 wide; W/3 = 360 px would make them touch). Centre
    y around 0.85·H. Expose diameter, y and soft edge as options. Keep the square size D even so the
    placement lands on whole pixels.
  - face framing: run `PoseEstimator` on about 40 frames sampled across each chorus clip (it doesn't
    need every frame for a fixed framing). Take the median of `head.head_centers()` for the head
    centre, and the median neck→head-top length (Halpe26 keypoints 18→17) for head size, scaled to
    source pixels. Initial zoom s0 = 0.55·d / head_len_px (neck→top fills about 55% of the circle).
    Without a confident face, fall back to the frame centre and a fit-to-circle scale, with a warning.
  - audio sync: `ffmpeg -map 0:a:0 -ac 1 -ar 16000 -f f32le -`, then an onset envelope (spectral flux
    of the log-magnitude STFT, 10 ms hop). Cross-correlate the middle and right clips against the left
    one with an FFT and take the peak lag. Confidence = peak ÷ best peak outside ±0.5 s; warn if low.
    The envelope works better than raw waveforms because the takes can be different harmony parts. A
    clip without audio gets offset 0 and a warning.
  - `.setting` builder for one circle (Lua table text, testable with lupa like
    `tests/test_resolve_fusion.py`). Also write it next to each clip (`take.circle.setting`) as a
    drag-into-Fusion fallback.
- `src/headlock/resolve_api.py`: connect to Resolve and build the timeline (below).
- CLI: `headlock circles main.mp4 --left a.mov --middle b.mov --right c.mov [--at 1:23.4]
  [--diameter 0.30] [--y 0.85] [--soft 0.04] [--device coreml]`.
- UI: turn `ui.py` into a `ttk.Notebook` with tabs: **Head-lock** (the current contents),
  **Chorus circles** (main + Left/Middle/Right pickers, options, "Build in Resolve") and **Check & test**
  (`check`, `test-clip`). Add a button **"Add Headlock to Resolve's Scripts menu"** that writes
  `~/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/Utility/Headlock.py`.
  It is a few lines run by ResolvePython that just does `subprocess.Popen(["open", "<repo>/Headlock.command"])`.
  The app runs in our uv env and talks back to Resolve over external scripting.

### Fusion graph per chorus clip
The output is the full timeline size (W×H) and transparent outside the circle, so it composites over
V1 with no Edit-page resizing.
```
MediaIn1 → CircleFace (Transform) → CircleZoom (Merge, fg) → CirclePlace (Merge, fg) → MediaOut1
            Center=(1-hx, 1-hy)       bg = CircleSquare        bg = CircleCanvas
            head → image centre       (Background D×D, a=0)    (Background W×H, a=0)
                                      Center=(0.5,0.5)         Center=(cx/W, 1-cy/H), Size=1
                                      Size = s0  ← face zoom
                                      EffectMask = CircleMask (EllipseMask, D×D, Center .5,.5,
                                                   Width=Height≈1-margin, SoftEdge≈0.04)
```
- (hx, hy) is the head position normalised in Fusion coordinates (x from the left, **y up**). The
  Transform with Size 1 and default Pivot is a pure translation (Center 0.5 = no move), so it's
  unambiguous.
- Merge semantics already relied on in `resolve_fusion.py`: Size 1 maps one fg pixel to one bg pixel,
  the fg centre lands at Center, and scaling is about the fg centre. Because CircleFace puts the head
  at the fg centre, **changing CircleZoom's Size scales around the face, which stays centred in the
  circle**. That's the "zoom from the circle centre" the user asked for, inside Fusion.
- The mask sits on a **square** D×D image, so circle units can't be stretched by aspect ratio. The
  manual (p. 2472) says the Ellipse mask "is a circle by default", so Width/Height share units anyway.
  Give the mask a custom D×D output size (Image tab → Output Size: Default / source input / Custom;
  the `.setting` input IDs are probably `OutputSize = FuID{"Custom"}`, `MaskWidth`, `MaskHeight`, still
  **to verify**). Soft Edge blurs both sides of the edge, so keep the ellipse slightly under 1.0.
- Background alpha 0: `TopLeftAlpha = 0`, `UseFrameFormatSettings = 0`, Width/Height (same input IDs
  as `HeadlockCanvas` in `resolve_fusion.py`).
- Edit page: also set the timeline item's `AnchorPointX = cx - W/2`, `AnchorPointY = H/2 - cy`
  (pixels from the frame centre, **assuming Y up**, to verify). The Inspector's Zoom then grows or
  shrinks the whole circle in place instead of from the frame centre. Manual: "Anchor Point X and Y:
  Defines the coordinate on that clip about which all transforms are centered."

### Resolve build steps (API names checked against the 21.1 `DaVinciResolveScript.pyi`)
```python
resolve = DaVinciResolveScript.scriptapp("Resolve")          # None → tell the user to enable Local scripting
project = resolve.GetProjectManager().GetCurrentProject()
mp = project.GetMediaPool()
main, left, mid, right = mp.ImportMedia([{"FilePath": p} for p in paths])
tl = mp.CreateEmptyTimeline(f"{stem} circles")
tl.SetSettings({"useCustomSettings": "1", "timelineResolutionWidth": "1080",
                "timelineResolutionHeight": "1920", "timelineFrameRate": "<main fps>"})  # before any clip
for _ in range(3): tl.AddTrack("video"); tl.AddTrack("audio", "stereo")
start = tl.GetStartFrame()                                     # usually 86400 (01:00:00:00)
mp.AppendToTimeline([{"mediaPoolItem": main, "trackIndex": 1, "recordFrame": start}])
# each chorus clip: video (mediaType 1) on V2..V4 and audio (mediaType 2) on A2..A4 at
# recordFrame = start + at + offset_i (frames); then link all six:
tl.SetClipsLinked(chorus_items, True)
comp = item.GetFusionCompByIndex(1) if item.GetFusionCompCount() else item.AddFusionComp()
comp.Paste(DaVinciResolveScript.readstring(setting_text))
comp.FindTool("MediaOut1").ConnectInput("Input", comp.FindTool("CirclePlace"))
item.SetProperty({"AnchorPointX": ..., "AnchorPointY": ...})
```
Also available: `Timeline.AutoAlignClips(items, {"SyncUsing": resolve.AUTO_ALIGN_CLIPS_USING_WAVEFORM})`,
Resolve's own waveform alignment. It could replace our sync or act as a fallback, but there's no
confidence value and it can't be unit-tested. `TimelineItem.ImportFusionComp(path)` exists too (needs a
full `.comp` file). Other `TimelineItemProperties`: `Pan`, `Tilt`, `ZoomX`, `ZoomY`, `ZoomGang`,
`CompositeMode`, `Opacity`, `AudioVolume`.

### Verify in Resolve on the target machine
1. Does a freshly appended item already have a comp with MediaIn1→MediaOut1 (count 0 vs 1)? Does
   `AddFusionComp()` create MediaIn1/MediaOut1?
2. Does `comp.Paste(readstring(text))` work from the external interpreter, and does
   `SourceOp = "MediaIn1"` hook onto the existing MediaIn1?
3. Does the Fusion output's alpha composite over V1 on the Edit page (expected yes)?
4. Check the EllipseMask custom output-size input IDs; is SoftEdge relative to the mask width?
5. What is the sign of `AnchorPointY` (and `Tilt`): is +Y up?
6. Does the linked group of six items drag together? Must `timelineFrameRate` be set before the first
   clip?
7. Run the existing `headlock test-clip dot.mp4` check from the README too (the Resolve mapping hasn't
   been confirmed in the app yet).
8. With a portrait phone clip (stored sideways + rotation flag): does Resolve show it upright, and
   does MediaIn1 deliver it upright at the upright size? The framing assumes so.
9. Does `import DaVinciResolveScript` load `fusionscript.so` in our Python 3.12 venv (Resolve bundles
   3.14)? If not, the app can't build the timeline; the `.circle.setting` files still work by hand.

### Tests to write
- slot geometry (centres at 1/6, 1/2, 5/6; diameter; even D);
- `.setting` evaluates as Lua, the wiring is right, and the head lands at the circle centre for any
  CircleZoom Size (same Merge maths as `test_setting_is_valid_lua_and_pins_the_head`);
- sync recovers a known offset within one frame from synthetic clips that share a rhythm but have
  different tones;
- framing picks the median head and scale from synthetic keypoints;
- the Resolve build against a small fake API (track indexes, record frames, mediaType, linking);
- UI: each tab builds the right argv (like `tests/test_ui.py`).
