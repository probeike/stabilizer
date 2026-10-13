"""A small stand-in for DaVinci Resolve's scripting API: records what ``resolve_api.build`` does."""

import pytest


class FakeTool:
    def __init__(self, name):
        self.name, self.inputs, self.connected = name, {}, {}

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
            self.tools[name] = FakeTool(name)
            for key, inp in tool.get("Inputs", {}).items():
                if "Value" in inp:
                    self.tools[name].inputs[key] = inp["Value"]
        return True

    def FindTool(self, name):
        return self.tools.get(name)


class FakeItem:
    def __init__(self, clip, track, record, media_type):
        self.clip, self.track, self.record, self.media_type = clip, track, record, media_type
        self.comps, self.props = [], {}

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
    def __init__(self, path, frames):
        self.path, self.frames = path, frames

    def GetName(self):
        return self.path.rsplit("/", 1)[-1]

    def GetClipProperty(self, key):
        return {"File Path": self.path, "Frames": str(self.frames)}[key]


class FakeTimeline:
    def __init__(self, name):
        self.name, self.settings, self.items, self.linked = name, {}, [], []
        self.tracks = {"video": 1, "audio": 1}

    def GetName(self):
        return self.name

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
        kinds = [info["mediaType"]] if "mediaType" in info else [1, 2]
        made = [FakeItem(clip, info["trackIndex"], info["recordFrame"], k) for k in kinds]
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
        Input, OperatorInfo = kind("Input"), kind("OperatorInfo")
        function FuID(t) return t[1] end
    """)
    t = lua.eval(text)
    return {"Tools": {k: {"Inputs": dict(v["Inputs"].items())} for k, v in t["Tools"].items()}}
