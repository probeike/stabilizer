"""Karaoke captions: the lyrics a line at a time, each word lighting up as it's sung and staying lit.

1. ``parse_lyrics``: pasted lyrics -> one phrase per line (blank lines and [Chorus]-style headings
   are skipped).
2. ``transcribe``: Whisper (faster-whisper) hears the vocal with a time for every word. The lyrics
   are passed as ``hotwords``, which steers it toward the right words. (Passed as the prompt
   instead, it skips the first line, taking it as already said.)
   ``clean_heard`` then checks every heard word against the sound itself: words where nothing is
   audible (Whisper "singing along" in a silence, which the hotwords encourage) are dropped, and a
   word that swallowed the silence before it starts where the sound does.
3. ``align``: lines the heard words up with the lyrics, so every lyric word gets a time even where
   Whisper misheard it ("seeing" for "sing"): the words are matched in order allowing for misses,
   extras and mishearings, and words it didn't catch are spread between their timed neighbours.
4. ``show_times``: when each line comes up and goes, and ``write_lrc`` saves it all as enhanced
   LRC (a plain-text karaoke format) to correct by hand and place again without re-listening.
5. ``build_setting``: the Fusion nodes for one line: the text dimmed, the same text lit on top of
   it, revealed word by word with Text+'s Write On.
"""

from __future__ import annotations

import os
import re
import struct
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

SECTION = re.compile(r"^\s*[\[(].*[\])]\s*$")   # "[Chorus]", "(x2)"
SIMILAR = 0.65         # spelling similarity that counts as the same word misheard
WORD_PACE = 0.3        # s per word, for words with no timed neighbour on one side
LEAD = 0.25            # a line comes up this long before its first word
HOLD = 0.6             # and stays this long after its last
JOIN = 0.4             # gaps shorter than this between lines are closed (no flicker)
LINE_CHARS = 24        # longer lines are shown on two rows


@dataclass
class Word:
    text: str
    start: float | None = None
    end: float | None = None
    heard: bool = False     # timed from what Whisper heard (False: spread between neighbours)


@dataclass
class Phrase:
    words: list[Word]
    show: float | None = None
    hide: float | None = None

    @property
    def found(self) -> bool:
        return any(w.heard for w in self.words)

    def text(self, line_chars: int = LINE_CHARS) -> str:
        """The words, broken onto two rows at the space nearest the middle if the line is long."""
        line = " ".join(w.text for w in self.words)
        if len(line) <= line_chars or len(self.words) < 2:
            return line
        cuts = [len(" ".join(w.text for w in self.words[:k])) for k in range(1, len(self.words))]
        k = min(range(len(cuts)), key=lambda c: abs(cuts[c] - len(line) / 2)) + 1
        return " ".join(w.text for w in self.words[:k]) + "\n" + " ".join(w.text for w in self.words[k:])


def parse_lyrics(text: str) -> list[Phrase]:
    phrases = []
    for line in text.splitlines():
        line = line.strip()
        if line and not SECTION.match(line):
            phrases.append(Phrase([Word(w) for w in line.split()]))
    return phrases


def norm(word: str) -> str:
    return re.sub(r"[^\w]", "", word.lower().replace("’", "'").replace("'", ""))


# ---- hearing ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class Heard:
    text: str
    start: float
    end: float


def transcribe(audio, lyrics: str, model: str = "large-v3-turbo", language: str | None = None,
               device: str = "cpu") -> list[Heard]:
    """Words Whisper hears in 16 kHz mono ``audio``, with times in seconds."""
    from faster_whisper import WhisperModel
    from tqdm import tqdm

    whisper = WhisperModel(model, device="cuda" if device == "cuda" else "cpu",
                           compute_type="float16" if device == "cuda" else "int8")
    segments, info = whisper.transcribe(
        audio, language=language, word_timestamps=True, hotwords=" ".join(lyrics.split()),
        condition_on_previous_text=False, hallucination_silence_threshold=2.0)
    heard = []
    with tqdm(total=round(info.duration), desc="Listening", unit="s", disable=not sys.stderr.isatty()) as bar:
        for seg in segments:
            heard += [Heard(w.word.strip(), float(w.start), float(w.end)) for w in seg.words or []]
            bar.update(max(0, round(seg.end) - bar.n))
    return heard


def loudness(audio, sr: int, hop: float = 0.01) -> np.ndarray:
    """RMS level every ``hop`` s (30 ms windows)."""
    a = np.asarray(audio, dtype=np.float32)
    n = max(1, int(round(hop * sr)))
    w = 3 * n
    a = np.pad(a, (w // 2, w))
    frames = np.lib.stride_tricks.sliding_window_view(a, w)[::n]
    return np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))


def clean_heard(heard: list[Heard], audio, sr: int, hop: float = 0.01) -> list[Heard]:
    """Drop words with nothing audible under them; start long words where their sound starts."""
    env = loudness(audio, sr, hop)
    loud = float(np.percentile(env, 95)) if len(env) else 0.0
    duration = len(audio) / sr
    out = []
    for h in heard:
        if h.end - h.start < 0.03 or h.start >= duration:
            continue
        a = int(h.start / hop)
        seg = env[a:max(int(h.end / hop), a + 1)]
        if not seg.size or seg.max() < 0.1 * loud:
            continue
        start = h.start
        if h.end - h.start > 0.3:   # held notes are loud from the start; swallowed silence isn't
            start = max(start, (a + int(np.argmax(seg >= 0.25 * seg.max()))) * hop - 0.02)
        out.append(Heard(h.text, start, h.end))
    return out


# ---- aligning --------------------------------------------------------------------------------

def _score(a: str, b: str) -> float:
    if a == b and a:
        return 3.0
    if a and b and SequenceMatcher(None, a, b).ratio() >= SIMILAR:
        return 2.0
    return -1.0   # a substitution: costs the same as skipping one word, less than skipping two


def match(lyric: list[str], heard: list[str]) -> list[tuple[int | None, bool]]:
    """For each lyric word, the heard word it lines up with (or None), and whether they're alike.

    Global alignment (Needleman-Wunsch) of the two word sequences: same or similar words score,
    and skipping a word on either side costs the same as pairing two different words.
    """
    a, b = [norm(w) for w in lyric], [norm(w) for w in heard]
    n, m = len(a), len(b)
    gap = -1.0
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    move = [[0] * (m + 1) for _ in range(n + 1)]   # 0 pair, 1 skip lyric word, 2 skip heard word
    for i in range(1, n + 1):
        score[i][0], move[i][0] = i * gap, 1
    for j in range(1, m + 1):
        score[0][j], move[0][j] = j * gap, 2
    for i in range(1, n + 1):
        row, prev = score[i], score[i - 1]
        for j in range(1, m + 1):
            best, how = prev[j - 1] + _score(a[i - 1], b[j - 1]), 0
            if prev[j] + gap > best:
                best, how = prev[j] + gap, 1
            if row[j - 1] + gap > best:
                best, how = row[j - 1] + gap, 2
            row[j], move[i][j] = best, how
    out: list[tuple[int | None, bool]] = [(None, False)] * n
    i, j = n, m
    while i > 0 and j > 0:
        how = move[i][j]
        if how == 0:
            out[i - 1] = (j - 1, _score(a[i - 1], b[j - 1]) > 0)
            i, j = i - 1, j - 1
        elif how == 1:
            i -= 1
        else:
            j -= 1
    return out


def align(phrases: list[Phrase], heard: list[Heard]) -> list[str]:
    """Give every lyric word a start and end time. Returns notes for the user.

    A word paired with a word that sounds alike takes its time. A word paired with a different
    word (misheard) takes it too, if its line has at least one alike match to anchor it. A line
    with no alike word at all wasn't found in the recording and stays untimed.
    """
    words = [w for p in phrases for w in p.words]
    for w in words:
        w.start = w.end = None
        w.heard = False
    pairs = match([w.text for w in words], [h.text for h in heard])
    k = 0
    for p in phrases:
        mine = pairs[k:k + len(p.words)]
        k += len(p.words)
        if any(alike for _, alike in mine):
            for w, (j, _) in zip(p.words, mine):
                if j is not None:
                    w.start, w.end, w.heard = heard[j].start, heard[j].end, True
    _spread(phrases)
    missing = [p for p in phrases if not p.found]
    notes = [f"not found in the recording, left out: \"{p.text(10**6)}\"" for p in missing]
    guessed = sum(1 for p in phrases if p.found for w in p.words if not w.heard)
    if guessed:
        notes.append(f"{guessed} word(s) weren't heard clearly; their timing is spread between the words around them")
    return notes


def _spread(phrases: list[Phrase]) -> None:
    """Time the untimed words of found lines between their timed neighbours, by length."""
    found = [p for p in phrases if p.found]
    words = [w for p in found for w in p.words]
    timed = [i for i, w in enumerate(words) if w.heard]
    for i, w in enumerate(words):
        if w.heard:
            continue
        before = max((t for t in timed if t < i), default=None)
        after = min((t for t in timed if t > i), default=None)
        if before is not None and after is not None:
            run = words[before + 1:after]
            lo, hi = words[before].end, words[after].start
            hi = max(hi, lo)
            total = sum(len(x.text) + 1 for x in run)
            done = sum(len(x.text) + 1 for x in words[before + 1:i])
            w.start = lo + (hi - lo) * done / total
            w.end = lo + (hi - lo) * (done + len(w.text) + 1) / total
        elif after is not None:     # before the first heard word: back off at a singing pace
            w.start = words[after].start - WORD_PACE * (after - i)
            w.end = w.start + WORD_PACE
        else:                       # after the last heard word
            w.start = words[before].end + WORD_PACE * (i - before - 1)
            w.end = w.start + WORD_PACE
    # Words keep their order even where a guess overlapped a neighbour.
    for prev, w in zip(words, words[1:]):
        w.start = max(w.start, prev.start)
        w.end = max(w.end, w.start)


def show_times(phrases: list[Phrase], lead: float = LEAD, hold: float = HOLD, join: float = JOIN,
               keep_show: bool = False) -> None:
    """When each found line comes up and goes. Lines never overlap; short gaps are closed.

    ``keep_show`` keeps show times already set (from an edited timing file) and only works out
    when each line goes.
    """
    found = [p for p in phrases if p.found]
    for p in found:
        if not keep_show or p.show is None:
            p.show = max(0.0, p.words[0].start - lead)
        p.show = min(p.show, p.words[0].start)
        p.hide = p.words[-1].end + hold
    for p, nxt in zip(found, found[1:]):
        if not keep_show:   # don't bring a line up before the previous one is sung
            nxt.show = max(nxt.show, min(p.words[-1].end, nxt.words[0].start))
        if nxt.show - p.hide < join:
            p.hide = nxt.show


# ---- timing file (enhanced LRC) --------------------------------------------------------------

def _stamp(t: float) -> str:
    cs = round(t * 100)
    return f"{cs // 6000:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def write_lrc(phrases: list[Phrase], path: str | Path) -> None:
    """One line per lyric line: ``[show]<start>word <start>word ... <end of last word>``.

    Times are from the start of the timeline. Lines not found are kept as plain text, untimed.
    """
    lines = ["[re:headlock]  times are from the start of the timeline; edit them and place again with --timing"]
    for p in phrases:
        if not p.found:
            lines.append(" ".join(w.text for w in p.words))
            continue
        words = " ".join(f"<{_stamp(w.start)}>{w.text}" for w in p.words)
        lines.append(f"[{_stamp(p.show)}]{words} <{_stamp(p.words[-1].end)}>")
    Path(path).write_text("\n".join(lines) + "\n")


STAMP = re.compile(r"<(\d+):(\d+(?:\.\d+)?)>")


def read_lrc(path: str | Path) -> list[Phrase]:
    """Phrases from ``write_lrc``'s format (edited or not). Untimed lines come back unfound."""
    phrases = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("[re:") or SECTION.match(line):
            continue
        head = re.match(r"\[(\d+):(\d+(?:\.\d+)?)\]", line)
        if not head:
            phrases.append(Phrase([Word(w) for w in line.split()]))
            continue
        body = line[head.end():]
        stamps = [(m.start(), m.end(), int(m[1]) * 60 + float(m[2])) for m in STAMP.finditer(body)]
        words = []
        for (_, e, t), nxt in zip(stamps, stamps[1:] + [(len(body), len(body), None)]):
            text = body[e:nxt[0]].strip()
            if text:
                words.append(Word(text, t, nxt[2] if nxt[2] is not None else t + WORD_PACE, heard=True))
        for w, nxt in zip(words, words[1:]):
            w.end = min(w.end, nxt.start) if w.end is not None else nxt.start
        if words:
            phrases.append(Phrase(words, show=int(head[1]) * 60 + float(head[2])))
    show_times(phrases, keep_show=True)
    return phrases


# ---- font ------------------------------------------------------------------------------------

FONT_DIRS = ["~/Library/Fonts", "/Library/Fonts", "/System/Library/Fonts", "/System/Library/Fonts/Supplemental",
             "~/.local/share/fonts", "~/.fonts", "/usr/share/fonts", "/usr/local/share/fonts"]


def font_names(path: str | Path) -> tuple[str, str] | None:
    """(family, style) from a .ttf/.otf's name table, as font menus (and Fusion) list them."""
    try:
        data = Path(path).read_bytes()
        if data[:4] == b"ttcf":
            data_off = struct.unpack(">I", data[12:16])[0]
        else:
            data_off = 0
        num = struct.unpack(">H", data[data_off + 4:data_off + 6])[0]
        for k in range(num):
            rec = data_off + 12 + 16 * k
            if data[rec:rec + 4] == b"name":
                off = struct.unpack(">I", data[rec + 8:rec + 12])[0]
                break
        else:
            return None
        count, strings = struct.unpack(">HH", data[off + 2:off + 6])
        names: dict[tuple[int, int], str] = {}
        for k in range(count):
            platform, _, lang, name_id, length, at = struct.unpack(">HHHHHH", data[off + 6 + 12 * k:off + 18 + 12 * k])
            raw = data[off + strings + at:off + strings + at + length]
            if platform == 3 and lang in (0x409, 0) or platform == 0:
                names.setdefault((3, name_id), raw.decode("utf-16-be", "replace"))
            elif platform == 1:
                names.setdefault((1, name_id), raw.decode("mac_roman", "replace"))
        get = lambda i: names.get((3, i)) or names.get((1, i))   # noqa: E731
        family, style = get(16) or get(1), get(17) or get(2)
        return (family, style or "Regular") if family else None
    except (OSError, struct.error, IndexError):
        return None


def find_font(name: str, dirs: list[str] = FONT_DIRS) -> tuple[str, str] | None:
    """The installed font family matching ``name`` (ignoring case and spaces) and its regular style.

    "IM Fell English" is listed by the font itself as "IM FELL English"; Fusion wants the exact name.
    """
    want = re.sub(r"\s+", "", name.lower())
    styles: dict[str, list[str]] = {}
    for d in dirs:
        for root, _, files in os.walk(os.path.expanduser(d)):
            for f in files:
                if f.lower().endswith((".ttf", ".otf", ".ttc")):
                    names = font_names(os.path.join(root, f))
                    if names and re.sub(r"\s+", "", names[0].lower()) == want:
                        styles.setdefault(names[0], []).append(names[1])
    if not styles:
        return None
    family, found = next(iter(styles.items()))
    style = next((s for s in found if s.lower() in ("regular", "roman", "book")), found[0])
    return family, style


# ---- Fusion ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Style:
    font: str = "IM Fell English"
    font_style: str = "Regular"
    size: float = 0.07                       # Text+ Size
    y: float = 0.13                          # centre of the captions, fraction from the top
    sung: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    unsung: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.4)
    line_chars: int = LINE_CHARS


def lua_string(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def write_on_steps(phrase: Phrase, fps: float, line_chars: int = LINE_CHARS) -> list[tuple[int, float]]:
    """(frame from the line's start, Write On end) as each word starts: the lit text so far.

    Write On counts characters of the whole text; each step ends in the space (or line break)
    after the word, so rounding either way doesn't cut a letter.
    """
    text = phrase.text(line_chars)
    n = len(text)
    steps, pos = [(0, 0.0)], 0
    for w in phrase.words:
        pos = text.index(w.text, pos) + len(w.text)
        frame = max(0, round((w.start - phrase.show) * fps))
        steps.append((frame, min(1.0, (pos + 0.5) / n)))
    out: list[tuple[int, float]] = []
    for frame, value in steps:
        if out and frame <= out[-1][0]:
            out[-1] = (out[-1][0], value)   # two words in one frame: the later one wins
        else:
            out.append((frame, value))
    return out


def _text_tool(name: str, text: str, style: Style, color, pos: int, end_spline: str | None = None) -> str:
    r, g, b, a = color
    end = f"\t\t\t\tEnd = Input {{ SourceOp = \"{end_spline}\", Source = \"Value\", }},\n" if end_spline else ""
    return (
        f"\t\t{name} = TextPlus {{\n"
        "\t\t\tNameSet = true,\n"
        "\t\t\tInputs = {\n"
        f"\t\t\t\tStyledText = Input {{ Value = {lua_string(text)}, }},\n"
        f"\t\t\t\tFont = Input {{ Value = {lua_string(style.font)}, }},\n"
        f"\t\t\t\tStyle = Input {{ Value = {lua_string(style.font_style)}, }},\n"
        f"\t\t\t\tSize = Input {{ Value = {style.size:.5f}, }},\n"
        f"\t\t\t\tCenter = Input {{ Value = {{ 0.5, {1.0 - style.y:.5f} }}, }},\n"
        f"\t\t\t\tRed1 = Input {{ Value = {r:.4f}, }},\n"
        f"\t\t\t\tGreen1 = Input {{ Value = {g:.4f}, }},\n"
        f"\t\t\t\tBlue1 = Input {{ Value = {b:.4f}, }},\n"
        f"\t\t\t\tAlpha1 = Input {{ Value = {a:.4f}, }},\n"
        + end +
        "\t\t\t},\n"
        f"\t\t\tViewInfo = OperatorInfo {{ Pos = {{ {pos}, 0 }} }},\n"
        "\t\t},\n"
    )


def build_setting(phrase: Phrase, style: Style, fps: float) -> str:
    """Fusion nodes for one line: CaptionBase (dim) under CaptionLit (lit), merged in CaptionMerge.

    CaptionLit's Write On end steps up as each word starts, so the sung words light up and stay
    lit. Connect CaptionMerge to MediaOut1. Frame 0 is the start of the line's title clip.
    """
    text = phrase.text(style.line_chars)
    steps = write_on_steps(phrase, fps, style.line_chars)
    keys: dict[int, float] = {}
    prev = 0.0
    for frame, value in steps:
        if frame > 0:
            keys.setdefault(frame - 1, prev)   # hold until the frame before: a step, not a fade
        keys[frame] = value
        prev = value
    spline = ",\n".join(f"\t\t\t\t[{f}] = {{ {v:.6f}, Flags = {{ Linear = true }} }}" for f, v in sorted(keys.items()))
    return (
        "{\n"
        "\tTools = ordered() {\n"
        + _text_tool("CaptionBase", text, style, style.unsung, 0)
        + _text_tool("CaptionLit", text, style, style.sung, 0, "CaptionLitEnd")
        + "\t\tCaptionLitEnd = BezierSpline {\n"
        "\t\t\tSplineColor = { Red = 255, Green = 200, Blue = 0 },\n"
        "\t\t\tNameSet = true,\n"
        f"\t\t\tKeyFrames = {{\n{spline}\n\t\t\t}}\n"
        "\t\t},\n"
        "\t\tCaptionMerge = Merge {\n"
        "\t\t\tNameSet = true,\n"
        "\t\t\tInputs = {\n"
        "\t\t\t\tBackground = Input { SourceOp = \"CaptionBase\", Source = \"Output\", },\n"
        "\t\t\t\tForeground = Input { SourceOp = \"CaptionLit\", Source = \"Output\", },\n"
        "\t\t\t},\n"
        "\t\t\tViewInfo = OperatorInfo { Pos = { 110, 0 } },\n"
        "\t\t},\n"
        "\t},\n"
        "\tActiveTool = \"CaptionMerge\"\n"
        "}\n"
    )


@dataclass
class Placement:
    """A found line, in timeline frames from the timeline's first frame."""
    phrase: Phrase
    start: int
    frames: int
    setting: str = field(repr=False)


def placements(phrases: list[Phrase], style: Style, fps: float) -> list[Placement]:
    out = []
    for p in phrases:
        if not p.found:
            continue
        start = round(p.show * fps)
        frames = max(1, round(p.hide * fps) - start)
        out.append(Placement(p, start, frames, build_setting(p, style, fps)))
    for a, b in zip(out, out[1:]):   # rounding mustn't make neighbours overlap on the track
        a.frames = max(1, min(a.frames, b.start - a.start))
    return out
