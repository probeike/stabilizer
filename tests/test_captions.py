import glob

import numpy as np
import pytest

from headlock import captions
from headlock.captions import Heard, Phrase, Word

LYRICS = """[Verse 1]
I keep running through the night
The road is long but I am not alone

(Chorus)
Sing it with me, sing it loud
"""


def _heard(words_and_times):
    return [Heard(w, t, t + 0.25) for w, t in words_and_times]


def test_parse_lyrics_skips_headings_and_blank_lines():
    phrases = captions.parse_lyrics(LYRICS)
    assert [len(p.words) for p in phrases] == [6, 9, 7]
    assert phrases[2].words[2].text == "with"


def test_long_lines_break_near_the_middle():
    p = captions.parse_lyrics("The road is long but I am not alone")[0]
    assert p.text(24) == "The road is long\nbut I am not alone"
    assert p.text(100) == "The road is long but I am not alone"


def test_align_times_every_word_despite_mishearing():
    phrases = captions.parse_lyrics(LYRICS)
    heard = _heard([("I", 0.0), ("keep", 0.3), ("running", 0.6), ("true", 0.9), ("the", 1.26), ("night,", 1.48),
                    ("um", 2.0),                                                  # an extra word
                    ("the", 2.36), ("road", 2.46), ("is", 2.66), ("long", 2.92),  # "but I" missed
                    ("am", 3.78), ("not", 3.86), ("alone", 4.08),
                    ("seeing", 5.2), ("it", 5.3), ("with", 5.52), ("me", 5.72), ("seeing", 6.26), ("it", 6.46),
                    ("loud", 6.68)])
    notes = captions.align(phrases, heard)
    words = [w for p in phrases for w in p.words]
    assert all(w.start is not None for w in words)
    starts = [w.start for w in words]
    assert starts == sorted(starts)
    by = {(i, w.text): w for i, p in enumerate(phrases) for w in p.words}
    assert by[(0, "through")].start == 0.9 and by[(0, "through")].heard      # misheard, still timed
    assert by[(2, "Sing")].start == 5.2                                       # "seeing"
    but, i = by[(1, "but")], by[(1, "I")]
    assert not but.heard and 2.92 + 0.25 <= but.start < i.start < 3.78      # spread into the gap
    assert any("weren't heard" in n for n in notes)


def test_a_line_that_isnt_sung_is_left_out():
    phrases = captions.parse_lyrics("I keep running through the night\nNever sung at all here\nSing it loud")
    heard = _heard([("I", 1.0), ("keep", 1.3), ("running", 1.6), ("through", 1.9), ("the", 2.2), ("night", 2.4),
                    ("sing", 9.0), ("it", 9.3), ("loud", 9.6)])
    notes = captions.align(phrases, heard)
    assert [p.found for p in phrases] == [True, False, True]
    assert any("Never sung" in n for n in notes)
    captions.show_times(phrases)
    assert phrases[0].show == pytest.approx(0.75) and phrases[0].hide == pytest.approx(2.65 + captions.HOLD)
    assert phrases[2].show == pytest.approx(8.75)


def test_lines_never_overlap_and_short_gaps_close():
    a = Phrase([Word("one", 1.0, 1.4, True), Word("two", 1.5, 2.0, True)])
    b = Phrase([Word("three", 2.3, 2.6, True)])
    c = Phrase([Word("four", 6.0, 6.5, True)])
    captions.show_times([a, b, c])
    assert a.hide == b.show == pytest.approx(2.05)          # no flicker between a and b
    assert b.hide == pytest.approx(2.6 + captions.HOLD) and b.hide < c.show


def test_lrc_round_trip_and_hand_edits(tmp_path):
    phrases = captions.parse_lyrics("I keep running\nNot sung\nSing it loud")
    for p, t0 in zip(phrases, (1.0, None, 4.0)):
        for k, w in enumerate(p.words):
            if t0 is not None:
                w.start, w.end, w.heard = t0 + 0.5 * k, t0 + 0.5 * k + 0.4, True
    captions.show_times(phrases)
    path = tmp_path / "song.lrc"
    captions.write_lrc(phrases, path)
    text = path.read_text()
    assert "[00:00.75]<00:01.00>I <00:01.50>keep <00:02.00>running <00:02.40>" in text
    assert "\nNot sung\n" in text

    back = captions.read_lrc(path)
    assert [p.found for p in back] == [True, False, True]
    assert [w.start for w in back[0].words] == [1.0, 1.5, 2.0] and back[0].words[-1].end == pytest.approx(2.4)
    assert back[0].show == pytest.approx(0.75)

    path.write_text(text.replace("<00:04.50>it", "<00:04.70>it").replace("[00:03.75]", "[00:03.10]"))
    edited = captions.read_lrc(path)
    assert edited[2].words[1].start == pytest.approx(4.7) and edited[2].show == pytest.approx(3.1)
    assert edited[0].hide <= edited[2].show


def test_write_on_steps_end_in_the_space_after_each_word():
    p = Phrase([Word("The", 1.0, 1.2, True), Word("road", 1.3, 1.6, True), Word("is", 2.0, 2.1, True)], show=0.8)
    text = p.text()
    steps = captions.write_on_steps(p, 25)
    assert steps[0] == (0, 0.0)
    for (frame, value), word, end in zip(steps[1:], p.words, (3, 8, 11)):
        assert frame == round((word.start - 0.8) * 25)
        lit = value * len(text)
        assert end <= lit <= end + 0.5 + 1e-9    # the word and at most half the space after it


def test_clean_heard_drops_phantom_words_and_trims_swallowed_silence():
    sr = 16000
    audio = np.zeros(6 * sr, np.float32)
    t = np.arange(sr) / sr
    audio[2 * sr:3 * sr] = 0.3 * np.sin(2 * np.pi * 220 * t)          # "I" sung from 2.0 to 3.0 s
    audio[4 * sr:5 * sr] = 0.3 * np.sin(2 * np.pi * 330 * t)          # a held "night", 4.0 to 5.0 s
    heard = [Heard("I", 0.0, 3.0),            # Whisper let it swallow the silence before it
             Heard("ghost", 3.2, 3.8),        # nothing audible: Whisper singing along to itself
             Heard("night", 4.0, 5.0),
             Heard("loop", 5.9, 5.9), Heard("past", 6.5, 6.6)]
    clean = captions.clean_heard(heard, audio, sr)
    assert [h.text for h in clean] == ["I", "night"]
    assert clean[0].start == pytest.approx(1.98, abs=0.03) and clean[0].end == 3.0   # 20 ms before the sound
    assert clean[1].start == 4.0                                  # held from the start: untouched


lupa = pytest.importorskip("lupa")


def _eval(text):
    lua = lupa.LuaRuntime()
    lua.execute("""
        function ordered() return function(t) return t end end
        local function kind(name) return function(t) t.__kind = name; return t end end
        TextPlus, Merge, BezierSpline = kind("TextPlus"), kind("Merge"), kind("BezierSpline")
        Input, OperatorInfo = kind("Input"), kind("OperatorInfo")
    """)
    return lua.eval(text)


def test_setting_layers_lit_text_over_dim_text():
    p = Phrase([Word('Say "hi"', 2.0, 2.4, True), Word("back\\slash", 3.0, 3.2, True)], show=1.7, hide=4.0)
    style = captions.Style(font="IM FELL English", y=0.13)
    t = _eval(captions.build_setting(p, style, 30))["Tools"]
    base, lit, merge, spline = t["CaptionBase"], t["CaptionLit"], t["CaptionMerge"], t["CaptionLitEnd"]
    assert base["Inputs"]["StyledText"]["Value"] == lit["Inputs"]["StyledText"]["Value"] == 'Say "hi" back\\slash'
    assert base["Inputs"]["Font"]["Value"] == "IM FELL English"
    assert base["Inputs"]["Alpha1"]["Value"] < lit["Inputs"]["Alpha1"]["Value"] == 1
    assert base["Inputs"]["Center"]["Value"][2] == pytest.approx(0.87)
    assert merge["Inputs"]["Background"]["SourceOp"] == "CaptionBase"
    assert merge["Inputs"]["Foreground"]["SourceOp"] == "CaptionLit"
    assert lit["Inputs"]["End"]["SourceOp"] == "CaptionLitEnd"
    keys = {k: v[1] for k, v in spline["KeyFrames"].items()}
    # Nothing lit, then a step as each word starts (held until the frame before).
    assert keys[0] == 0 and keys[9 - 1] == 0 and keys[9] > 0            # "Say" starts 0.3 s in
    assert keys[39 - 1] == keys[9] and keys[39] == 1.0                    # "back\\slash" 1.3 s in


def test_placements_dont_overlap():
    a = Phrase([Word("a", 1.0, 1.5, True)], show=0.75, hide=2.0)
    b = Phrase([Word("b", 2.01, 2.5, True)], show=2.01, hide=3.0)
    out = captions.placements([a, Phrase([Word("x")]), b], captions.Style(), 30)
    assert [(x.start, x.frames) for x in out] == [(22, 38), (60, 30)]


def test_font_names_reads_installed_fonts():
    path = next(iter(glob.glob("/usr/share/fonts/**/LiberationSerif-Regular.ttf", recursive=True)), None)
    if path is None:
        pytest.skip("no Liberation fonts here")
    assert captions.font_names(path) == ("Liberation Serif", "Regular")
    assert captions.find_font("liberation   SERIF", ["/usr/share/fonts"]) == ("Liberation Serif", "Regular")
    assert captions.find_font("No Such Font", ["/usr/share/fonts"]) is None
