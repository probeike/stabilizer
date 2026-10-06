import xml.etree.ElementTree as ET
from fractions import Fraction

import numpy as np

from headlock import premiere_xml
from headlock.trackio import Track


def test_xml_structure_and_keyframes(tmp_path):
    P = np.array([[960.0, 540.0], [400.0, 300.0], [1500.0, 900.0]])
    tree = premiere_xml.build("/media/take 1.mp4", (3840, 2160), Fraction(30000, 1001), P,
                              (1920, 1080), 0.86, has_audio=True)
    out = tmp_path / "t.xml"
    premiere_xml.write(out, tree)
    root = ET.parse(out).getroot()

    assert root.tag == "xmeml"
    assert root.findtext("sequence/rate/timebase") == "30"
    assert root.findtext("sequence/rate/ntsc") == "TRUE"
    assert root.findtext(".//format/samplecharacteristics/width") == "1920"
    assert root.findtext(".//file/pathurl") == "file://localhost/media/take%201.mp4"
    assert root.findtext(".//file/media/video/samplecharacteristics/width") == "3840"

    params = {p.findtext("parameterid"): p for p in root.iter("parameter")}
    assert float(params["scale"].findtext("value")) == 86.0
    kfs = params["center"].findall("keyframe")
    assert [k.findtext("when") for k in kfs] == ["0", "1", "2"]
    got = np.array([[float(k.findtext("value/horiz")), float(k.findtext("value/vert"))] for k in kfs])
    np.testing.assert_allclose(got, premiere_xml.position_to_center(P, (1920, 1080)), atol=1e-6)
    assert root.find("sequence/media/audio/track/clipitem/file").get("id") == "file-1"


def test_track_roundtrip(tmp_path):
    head = np.array([[0.5, 0.2], [np.nan, np.nan], [0.25, 0.75]])
    t = Track("v.mp4", "s.mov", (3840, 2160), Fraction(30000, 1001), head, np.array([0.9, 0.1, 0.8]))
    t.save(tmp_path / "t.json")
    u = Track.load(tmp_path / "t.json")
    np.testing.assert_allclose(u.head, head)
    assert u.fps == t.fps and u.source_size == (3840, 2160)
