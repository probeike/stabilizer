# Target machine

The machine headlock will finally run on, as surveyed on 2026-10-12. Develop elsewhere, but check
anything Resolve-specific here.

## Hardware and OS
- Apple M5 (arm64), 10 cores, 32 GB RAM, Metal 4
- macOS 27.0.1 (build 26A434)

## Toolchain status
- The repo's `.venv` exists: Python 3.12.15, created by uv 0.13.0. `onnxruntime` 1.30.0 with
  `CoreMLExecutionProvider` available (so `--device coreml` should work), OpenCV 5.0.0, lupa installed.
- `.venv/bin/python -m pytest`: 38 passed, 1 skipped (the ffmpeg end-to-end test).
- **Missing: `ffmpeg`/`ffprobe` and `uv` are not on PATH** (checked `/opt/homebrew/bin`, `/usr/local/bin`,
  `~/.local/bin`, `~/.cargo/bin`). Homebrew is installed at `/opt/homebrew`. Before using the app here:
  `brew install ffmpeg uv`. `Headlock.command` refuses to start without uv, and tracking and rendering
  both need ffmpeg.
- RTMPose models: no `~/.cache/rtmlib` yet, so the first run downloads about 300 MB.

## DaVinci Resolve
- **DaVinci Resolve Studio 21.1.1** (`/Applications/DaVinci Resolve/DaVinci Resolve.app`). There is a
  Studio license in `/Library/Application Support/Blackmagic Design/DaVinci Resolve/.license/`.
  Studio matters because external scripting (driving Resolve from our own process) is Studio-only.
- External scripting preference (Preferences → System → General → External scripting) has **not been
  confirmed**. It must be **Local** for the headlock app to connect. With it off, or with Resolve not
  running, `scriptapp("Resolve")` returns `None` (Resolve wasn't running during the survey).
- Scripting docs: `/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/`
  (`README.md`, `DaVinciResolveScript.pyi` with typed API stubs, `Examples/`, `Modules/`).
- Bundled interpreter: `/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Applications/ResolvePython`
  is Python 3.14.4. `import DaVinciResolveScript` works with no env vars. No pip, so it can't run our
  pose code. Use it only for thin API glue.
- From another interpreter (our uv env): set
  `RESOLVE_SCRIPT_API=/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting`,
  `RESOLVE_SCRIPT_LIB=/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so`,
  and add `$RESOLVE_SCRIPT_API/Modules/` to `PYTHONPATH`.
- `DaVinciResolveScript` module functions include `scriptapp`, `readstring`, `writestring`,
  `getclipboard`, `setclipboard`. `readstring(text)` turns `.setting` text into a table for `comp.Paste()`.
- Workspace → Scripts menu folder (per user; exists, subfolders empty):
  `~/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/{Utility,Edit,Comp,Tool,Color,Deliver,Views}`.
  A script in `Utility` shows up on every page. Scripts run from this menu get `resolve`, `fusion` and
  `bmd` as globals and don't need the external-scripting preference.
- User manual PDF: `/Applications/DaVinci Resolve/DaVinci Resolve Manual.pdf` (183 MB). No pdftotext is
  installed; a small Swift/PDFKit program (`PDFDocument.findString`) searches it fine.
