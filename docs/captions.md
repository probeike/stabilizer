# Karaoke captions: design notes and what to check in Resolve

Status: built on 2026-10-13 (`captions.py`, `resolve_api.place_captions`, `headlock captions`, the
Captions tab), tested against a fake Resolve API and synthetic speech (espeak + Whisper tiny);
**not yet run inside Resolve or on real singing**.

## What the user asked for
Karaoke-style captions in the top band of the vertical video, above the head-locked runner, in IM
Fell English: the whole line on screen, each word highlighting as it is sung and staying
highlighted. The user pastes the lyrics; the tool does the timing. Captions must be editable titles
in Resolve, generated after the clips are aligned on the timeline.

## How it works
- **Listening.** The timeline's audio track 1 is rebuilt from its clips (file, in-point from
  `GetLeftOffset`, position), so times are timeline times and cuts are respected. faster-whisper
  transcribes it with word timestamps, with the lyrics as `hotwords`. Passing the lyrics as
  `initial_prompt` instead made Whisper skip the first line (it took it as already said).
- **Cleaning.** Hotwords make Whisper "sing along" in silence (zero-length words past the end in
  the espeak test). Words with nothing audible under them are dropped, and a long word whose
  sound starts late (Whisper let the first word of a segment swallow 2 s of silence) is moved to
  where the sound starts. A held note is loud from its start, so it keeps its start.
- **Aligning.** Needleman-Wunsch over the two word sequences: same word 3, similar spelling 2
  (`SequenceMatcher` >= 0.65), different word -1 (so a mishearing in place is paired, not
  skipped), skipping -1. A lyric line counts as found only if one of its words matched by sound;
  its other words take their paired heard word's time, or are spread by length between timed
  neighbours. Lines not found are left out and listed.
- **Showing.** A line comes up 0.25 s before its first word and stays 0.6 s after its last, never
  overlapping the next; gaps under 0.4 s are closed so the captions don't flicker.
- **Resolve.** One Text+ title (appended from a Text+ in the Media Pool, as AutoSubs does, so its
  position and length are exact) per line, on a video track named Captions. Its stock Text+ is
  replaced by `CaptionBase` (dim, alpha 0.4) and `CaptionLit` (full white) merged; `CaptionLit`'s
  Write On `End` is keyframed to step to the space after each word as the word starts.

## Check in Resolve on the target machine
1. A Text+ dragged into the Media Pool can be appended with `AppendToTimeline` at an exact
   frame and length, and each appended title has its own comp (`GetFusionCompByIndex(1)`) whose
   Text+ `GetToolList(False, "TextPlus")` finds and `Delete()` removes.
2. **Write On**: is Text+'s input really `End`? The tool warns if it isn't; list the inputs with
   `tool.GetInputList()` to find it. Does it count every character including spaces and the line
   break? If the highlight drifts a letter per word, fix `captions.write_on_steps`.
3. Is frame 0 of a title's Fusion comp the title's first frame (the keyframes assume so)?
4. Text+ Size 0.07 and Center y 0.87 on a 1080x1920 timeline: right size and in the top band?
   `find_font` reads the installed font's own name (IM Fell English calls itself "IM FELL English").
5. `GetLeftOffset()` on an audio item is where the clip starts in its media, in timeline frames
   (captions after a cut land in the right place).
6. `AppendToTimeline`'s `endFrame` for titles: inclusive or not is measured on the first title.
7. Whisper `large-v3-turbo` on the M5's CPU: how long a 3-minute song takes, and how well it hears
   singing while running (wind). `--model medium.en` is lighter if needed.
