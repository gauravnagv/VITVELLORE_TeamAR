"""
demo_live.py - live, on-screen demo of the eye wrapper (for the 1-2 min demo video).

    blender scene.blend --python wrapper/demo_live.py

Opens Blender looking through the locked render camera and drives the eyes
with eyes.move()/eyes.reset(): centre -> left -> right -> up -> down -> reset,
then the 10-step clockwise sequence, then a smooth circle, then reset.
The current call is shown in the viewport header.  Runs for ~75 seconds.

Self-recording (used by record_demo.py): if the environment variables
DEMO_RECORD (output .mp4) and DEMO_FFMPEG (ffmpeg.exe) are set, Blender goes
fullscreen, screen-records itself with ffmpeg from "READY" to the end, and quits.
"""
import math
import os
import subprocess
import sys
import time

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eyes  # noqa: E402

STEP = 1.6  # seconds per discrete pose

script = [("eyes.reset()", 0, 0, 3.0)]
for name, h, v in [("LEFT", -1, 0), ("RIGHT", 1, 0), ("UP", 0, 1), ("DOWN", 0, -1)]:
    script.append(("eyes.move(%+d, %+d)   # %s" % (h, v, name), h, v, STEP + 0.6))
    script.append(("eyes.reset()   # CENTER", 0, 0, STEP))
for k in range(10):
    th = 2 * math.pi * k / 10
    script.append(("eyes.move(%+.3f, %+.3f)   # clockwise k=%d, theta=%.4f" % (math.sin(th), math.cos(th), k, th),
                   math.sin(th), math.cos(th), STEP))
for i in range(181):  # smooth 2 laps
    th = 2 * math.pi * i / 90
    script.append(("eyes.move(%+.3f, %+.3f)   # smooth circle" % (math.sin(th), math.cos(th)),
                   math.sin(th), math.cos(th), 0.12))
script.append(("eyes.reset()   # done", 0, 0, 3.0))

_state = {"i": 0}


def view3d():
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                return win, area
    return None, None


def setup():
    win, area = view3d()
    if area is None:
        return 0.5
    for space in area.spaces:
        if space.type == "VIEW_3D":
            space.region_3d.view_perspective = "CAMERA"
            space.shading.type = "MATERIAL"
            space.overlay.show_overlays = False
            space.show_gizmo = False
    with bpy.context.temp_override(window=win, area=area):
        bpy.ops.screen.screen_full_area()
        if RECORD:
            bpy.ops.wm.window_fullscreen_toggle()
    _state["t0"] = _state["last"] = time.monotonic()
    bpy.app.timers.register(warmup, first_interval=WARM_DT)
    return None


# The first time the viewport draws in Material Preview, Blender compiles its
# shaders and the UI can freeze for 30+ s.  Wait until the UI has run smoothly
# for CALM_NEEDED seconds (or MAX_WARM seconds have passed) before starting,
# so the recorded demo has no freeze.
WARM_DT, CALM_NEEDED, MIN_WARM, MAX_WARM, COUNTDOWN = 0.25, 10.0, 20.0, 150.0, 5


def warmup():
    now = time.monotonic()
    gap = now - _state["last"]
    _state["last"] = now
    _state["calm"] = 0.0 if gap > 0.75 else _state.get("calm", 0.0) + gap
    waited = now - _state["t0"]
    win, area = view3d()
    if (_state["calm"] >= CALM_NEEDED and waited >= MIN_WARM) or waited >= MAX_WARM:
        print("READY after %.0f s warm-up" % waited, flush=True)
        _state["count"] = COUNTDOWN
        start_recording()
        bpy.app.timers.register(countdown, first_interval=0.0)
        return None
    if area:
        area.header_text_set("Track 01 | Preparing demo, please wait... (%.0f s)" % waited)
        area.tag_redraw()
    return WARM_DT


def countdown():
    win, area = view3d()
    n = _state["count"]
    if n <= 0:
        bpy.app.timers.register(tick, first_interval=0.0)
        return None
    if area:
        area.header_text_set("Track 01 | READY - demo starts in %d s" % n)
        area.tag_redraw()
    _state["count"] = n - 1
    return 1.0


def tick():
    i = _state["i"]
    win, area = view3d()
    if i >= len(script):
        if area:
            area.header_text_set("Track 01 demo finished - eyes.reset()")
        if RECORD:
            bpy.app.timers.register(stop_recording, first_interval=3.0)
        return None
    text, h, v, dur = script[i]
    eyes.move(h, v)
    if area:
        area.header_text_set("Track 01 | %s" % text)
        area.tag_redraw()
    print(text, flush=True)
    _state["i"] = i + 1
    return dur


RECORD = os.environ.get("DEMO_RECORD")
FFMPEG = os.environ.get("DEMO_FFMPEG")


def start_recording():
    if not (RECORD and FFMPEG):
        return
    # ddagrab = Desktop Duplication API; it sees the GPU-drawn fullscreen
    # Blender window (the older gdigrab only captured a stale frame).
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi",
           "-i", "ddagrab=output_idx=0:framerate=30:draw_mouse=0,hwdownload,format=bgra",
           "-fps_mode", "cfr", "-r", "30",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           RECORD]
    log = open(RECORD + ".log", "w")
    _state["rec"] = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=log, stderr=log,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    print("RECORDING ->", RECORD, flush=True)


def stop_recording():
    rec = _state.get("rec")
    if rec:
        try:
            rec.communicate(b"q", timeout=60)  # 'q' makes ffmpeg finish the file cleanly
        except Exception:
            rec.kill()
        print("RECORDING SAVED ->", RECORD, flush=True)
    bpy.ops.wm.quit_blender()
    return None


bpy.app.timers.register(setup, first_interval=1.0)
