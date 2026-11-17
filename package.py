"""
package.py - assemble the submission zip in the layout the evaluators expect.

    python package.py                     # dist/VITVellore_AR.zip  (everything, incl. 10 texture sets)
    python package.py --no-textures       # dist/VITVellore_AR_core.zip (scene.blend has set01 packed)
    python package.py --name OTHER_NAME

Checks that every required deliverable exists (and is consistent) before writing the zip:
    scene.blend, wrapper/eyes.py, renders/frame_00..09.png, metadata.json, README.md   (original brief)
    videos/left|right|generate.mp4, expression_timeline.blend, renders/texture_sets/   (evaluation slide)
"""
import argparse
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM = "VITVellore_AR"

REQUIRED = [
    "scene.blend", "expression_timeline.blend", "README.md", "metadata.json",
    "wrapper/eyes.py", "wrapper/motion.py", "wrapper/demo_live.py",
    "verify.py", "build_scene_gnm.py", "polish_look.py", "add_expression_keys.py",
    "add_wrinkles.py", "add_hair.py", "add_tearline.py", "add_sclera_shading.py",   # imported by the build
    "videos/left.mp4", "videos/right.mp4", "videos/generate.mp4",
    "videos/left_states.json", "videos/right_states.json", "videos/generate_states.json",
]
OPTIONAL_FILES = [
    "videos/generate_closeup.mp4",
    "bake_skins.py", "make_texture_sets.py", "make_contact_sheet.py", "make_demo_video.py", "record_demo.py",
    "tools_regen_normal.py", "tools_regen_eye_tiles.py",
    "RUN_DEMO.bat", "VITVellore_AR.pptx",
    "Dockerfile", ".dockerignore", "requirements.txt", "LICENSE",
]
DIRS = ["renders", "demo", "gnm_head"]          # copied recursively (filtered below)
SKIP_EXT = (".pyc", ".log", ".blend1")


def check():
    problems = [p for p in REQUIRED if not os.path.isfile(os.path.join(HERE, p))]
    frames = sorted(f for f in os.listdir(os.path.join(HERE, "renders"))
                    if f.startswith("frame_") and f.endswith(".png")) if os.path.isdir(
        os.path.join(HERE, "renders")) else []
    if frames != ["frame_%02d.png" % k for k in range(10)]:
        problems.append("renders/frame_00..09.png (found %d)" % len(frames))
    meta = os.path.join(HERE, "metadata.json")
    if os.path.isfile(meta):
        recs = json.load(open(meta))
        if len(recs) != 10 or any(k not in recs[0] for k in (
                "frame", "theta", "input", "eye_rotation_L", "eye_rotation_R",
                "pupil_center_2D_L", "pupil_center_2D_R", "camera_matrix")):
            problems.append("metadata.json: expected 10 records with the brief's fields")
        # renders must be newer than the GNM scene they claim to show
        if os.path.getmtime(os.path.join(HERE, "renders", "frame_00.png")) < os.path.getmtime(
                os.path.join(HERE, "build_scene_gnm.py")) - 86400 * 3:
            problems.append("renders/frame_*.png look older than the GNM scene - re-run eyes.py sequence")
    for v in ("left", "right", "generate"):
        sj = os.path.join(HERE, "videos", v + "_states.json")
        if os.path.isfile(sj) and json.load(open(sj))["duration_s"] > 15.0:
            problems.append("videos/%s.mp4 longer than 15 s" % v)
    tex = os.path.join(HERE, "renders", "texture_sets")
    n_tex = len([f for f in os.listdir(tex) if f.endswith(".png")]) if os.path.isdir(tex) else 0
    if not 5 <= n_tex <= 10:
        problems.append("renders/texture_sets: need 5-10 renders (found %d)" % n_tex)
    return problems


def walk(rel_dir):
    for root, dirs, files in os.walk(os.path.join(HERE, rel_dir)):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if not f.endswith(SKIP_EXT):
                full = os.path.join(root, f)
                yield full, os.path.relpath(full, HERE).replace("\\", "/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=TEAM)
    ap.add_argument("--no-textures", action="store_true", help="leave out textures/ (~500 MB)")
    a = ap.parse_args()
    problems = check()
    if problems:
        sys.exit("Not ready:\n  - " + "\n  - ".join(problems))

    name = a.name + ("_core" if a.no_textures else "")
    out = os.path.join(HERE, "dist", name + ".zip")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    dirs = DIRS + ([] if a.no_textures else ["textures"])
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for p in REQUIRED + [p for p in OPTIONAL_FILES if os.path.isfile(os.path.join(HERE, p))]:
            z.write(os.path.join(HERE, p), p)       # brief: scene.blend etc. at the zip root
            n += 1
        for d in dirs:
            for full, rel in walk(d):
                # PNG/blend are already compressed: store them to save time
                ct = zipfile.ZIP_STORED if rel.endswith((".png", ".mp4", ".exr")) else zipfile.ZIP_DEFLATED
                z.write(full, rel, compress_type=ct)
                n += 1
    print("wrote %s  (%d files, %.0f MB)" % (out, n, os.path.getsize(out) / 1e6))


if __name__ == "__main__":
    main()
