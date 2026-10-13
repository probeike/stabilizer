"""A small stand-in for DaVinci Resolve's scripting API: records what ``resolve_api`` does."""

import pytest


class FakeTool:
    def __init__(self, name, kind=None, comp=None):
        self.name, self.kind, self.comp, self.inputs, self.connected = name, kind, comp, {}, {}

    def Delete(self):
        del self.comp.tools[self.name]

    def ConnectInput(self, name, tool):
        self.connected[name] = tool.name
        return True

    def GetInput(self, name):
        return self.inputs.get(name)


class FakeComp:
    def __init__(self):
        self.tools = {"MediaIn1": FakeTool("MediaIn1"), "MediaOut1": FakeTool("MediaOut1")}
        self.tools["MediaOut1"].connected["Input"] = "MediaIn1"

    def Paste(self, table):
        for name, tool in table["Tools"].items():
            self.tools[name] = FakeTool(name, tool.get("__kind"), self)
            for key, inp in tool.get("Inputs", {}).items():
                if "Value" in inp:
                    self.tools[name].inputs[key] = inp["Value"]
                elif table["Tools"].get(inp.get("SourceOp"), {}).get("__kind") == "BezierSpline":
                    self.tools[name].inputs[key] = 0.0   # animated: reads as its value now
        return True

    def FindTool(self, name):
        return self.tools.get(name)

    def GetToolList(self, selected, kind):
        return {i + 1: t for i, t in enumerate(t for t in self.tools.values() if t.kind == kind)}


class FakeTitleComp(FakeComp):
    """A Text+ title's comp: Template (Text+) -> MediaOut1."""

    def __init__(self):
        super().__init__()
        del self.tools["MediaIn1"]
        self.tools["Template"] = FakeTool("Template", "TextPlus", self)
        self.tools["MediaOut1"].connected["Input"] = "Template"


class FakeItem:
    def __init__(self, clip, track, record, media_type, duration=None, left_offset=0):
        self.clip, self.track, self.record, self.media_type = clip, track, record, media_type
        self.duration, self.left_offset = duration, left_offset
        self.comps, self.props = ([FakeTitleComp()] if getattr(clip, "title", False) else []), {}

    def GetMediaPoolItem(self):
        return self.clip

    def GetStart(self):
        return self.record

    def GetDuration(self):
        return self.duration

    def GetLeftOffset(self):
        return self.left_offset

    def GetFusionCompCount(self):
        return len(self.comps)

    def GetFusionCompByIndex(self, i):
        return self.comps[i - 1]

    def AddFusionComp(self):
        self.comps.append(FakeComp())
        return self.comps[-1]

    def SetProperty(self, key, value):
        self.props[key] = value
        return True


class FakeClip:
    def __init__(self, path, frames, title=False):
        self.path, self.frames, self.title = path, frames, title

    def GetName(self):
        return self.path.rsplit("/", 1)[-1]

    def GetClipProperty(self, key):
        return {"File Path": self.path, "Frames": str(self.frames)}[key]


class FakeTimeline:
    def __init__(self, name):
        self.name, self.settings, self.items, self.linked = name, {}, [], []
        self.tracks = {"video": 1, "audio": 1}
        self.track_names = {}

    def GetName(self):
        return self.name

    def GetSetting(self, key):
        return self.settings.get(key, "")

    def GetItemListInTrack(self, kind, track):
        want = 1 if kind == "video" else 2
        return sorted((i for i in self.items if i.track == track and i.media_type == want), key=lambda i: i.record)

    def GetTrackName(self, kind, i):
        return self.track_names.get((kind, i), f"{kind[0].upper()}{i}")

    def SetTrackName(self, kind, i, name):
        self.track_names[(kind, i)] = name
        return True

    def DeleteClips(self, items, ripple=False):
        gone = {id(i) for i in items}
        self.items = [i for i in self.items if id(i) not in gone]
        return True

    def SetSetting(self, key, value):
        assert not self.items, "settings must be set before any clip goes on"
        self.settings[key] = value
        return True

    def GetTrackCount(self, kind):
        return self.tracks[kind]

    def AddTrack(self, kind, sub=None):
        assert kind == "video" or sub == "stereo"
        self.tracks[kind] += 1
        return True

    def GetStartFrame(self):
        return 86400

    def SetClipsLinked(self, items, linked):
        self.linked.append(list(items))
        return True


class FakeFolder:
    def __init__(self, clips):
        self.clips = clips

    def GetClipList(self):
        return self.clips

    def GetSubFolderList(self):
        return []


class FakePool:
    def __init__(self, project, inclusive_end=True):
        self.project, self.root, self.inclusive_end = project, FakeFolder([]), inclusive_end
        self.frames = {}

    def GetRootFolder(self):
        return self.root

    def ImportMedia(self, paths):
        clip = FakeClip(paths[0], self.frames.get(paths[0], 1000))
        self.root.clips.append(clip)
        return [clip]

    def CreateEmptyTimeline(self, name):
        tl = FakeTimeline(name)
        self.project.timelines.append(tl)
        return tl

    def AppendToTimeline(self, infos):
        (info,) = infos
        clip = info["mediaPoolItem"]
        if info["endFrame"] > clip.frames - (1 if self.inclusive_end else 0):
            return []
        tl = self.project.current
        assert info["trackIndex"] <= tl.tracks["video"]
        kinds = [1] if clip.title else [info["mediaType"]] if "mediaType" in info else [1, 2]
        length = info["endFrame"] - info["startFrame"] + (1 if self.inclusive_end else 0)
        made = [FakeItem(clip, info["trackIndex"], info["recordFrame"], k, length) for k in kinds]
        tl.items += made
        return made


class FakeProject:
    def __init__(self, **kw):
        self.timelines, self.current = [], None
        self.pool = FakePool(self, **kw)

    def GetMediaPool(self):
        return self.pool

    def GetTimelineCount(self):
        return len(self.timelines)

    def GetTimelineByIndex(self, i):
        return self.timelines[i - 1]

    def SetCurrentTimeline(self, tl):
        self.current = tl
        return True

    def GetCurrentTimeline(self):
        return self.current


class FakeResolve:
    def __init__(self, **kw):
        self.project, self.page = FakeProject(**kw), None

    def GetProjectManager(self):
        return self

    def GetCurrentProject(self):
        return self.project

    def OpenPage(self, page):
        self.page = page


def readstring(text):
    lupa = pytest.importorskip("lupa")
    lua = lupa.LuaRuntime()
    lua.execute("""
        function ordered() return function(t) return t end end
        local function kind(name) return function(t) t.__kind = name; return t end end
        Background, Merge, Transform, EllipseMask = kind("Background"), kind("Merge"), kind("Transform"), kind("EllipseMask")
        TextPlus, BezierSpline = kind("TextPlus"), kind("BezierSpline")
        Input, OperatorInfo = kind("Input"), kind("OperatorInfo")
        function FuID(t) return t[1] end
    """)
    t = lua.eval(text)
    return {"Tools": {k: {"__kind": v["__kind"],
                          "Inputs": {n: dict(i.items()) for n, i in (v["Inputs"] or {}).items()}}
                      for k, v in t["Tools"].items()}}
