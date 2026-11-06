"""
record_demo.py - makes the demo video automatically (no manual screen recording).

    python record_demo.py            ->  demo/demo_video.mp4

Opens Blender fullscreen with wrapper/demo_live.py, which screen-records itself
with ffmpeg (from the imageio-ffmpeg package) and closes when the demo ends.
Do not touch the mouse/keyboard or open other windows while it runs (~2-3 min).
"""
import os
import subprocess
import sys

import imageio_ffmpeg

HERE = os.path.dirname(os.path.abspath(__file__))
BLENDER = os.path.join(HERE, "..", "tools", "blender-4.5.15-windows-x64", "blender.exe")
OUT = os.path.join(HERE, "demo", "demo_video.mp4")

env = dict(os.environ, DEMO_RECORD=OUT, DEMO_FFMPEG=imageio_ffmpeg.get_ffmpeg_exe())
subprocess.run([BLENDER, os.path.join(HERE, "scene.blend"),
                "--python", os.path.join(HERE, "wrapper", "demo_live.py")], env=env, cwd=HERE)
if os.path.exists(OUT) and os.path.getsize(OUT) > 0:
    print("Video saved:", OUT, "(%.1f MB)" % (os.path.getsize(OUT) / 1e6))
else:
    sys.exit("Video was not created - see " + OUT + ".log")
