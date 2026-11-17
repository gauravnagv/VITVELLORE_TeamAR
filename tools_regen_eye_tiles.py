"""tools_regen_eye_tiles.py - rewrite only the eye tiles (face.1002/1004.png) of existing texture
sets with the current gnm_head/textures.py, reading palette/age from each set.json.

    python tools_regen_eye_tiles.py SRC_ROOT DST_ROOT [--tiles-only]
Copies each SRC set to DST first, unless --tiles-only (then DST/<set>/ gets just the two tiles).
"""
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "gnm_head"))
import textures as tx  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
tiles_only = "--tiles-only" in sys.argv
for name in sorted(os.listdir(src)):
    d0 = os.path.join(src, name)
    if not os.path.isfile(os.path.join(d0, "set.json")):
        continue
    d = os.path.join(dst, name)
    if tiles_only:
        os.makedirs(d, exist_ok=True)
    elif not os.path.isdir(d):
        shutil.copytree(d0, d)
    js = json.load(open(os.path.join(d0, "set.json")))
    seed, pal, age = int(js["id"][3:]), js["iris_palette"], float(js.get("age_factor", 0.0))
    for tile, ns, sd in ((1002, -1.0, seed * 2 + 1), (1004, +1.0, seed * 2 + 2)):
        tx.write_png(os.path.join(d, "face.%d.png" % tile),
                     tx.eye_tile(2048, pal, sd, nasal_sign=ns, sclera_tint=0.6 * age))
    print("[regen]", name, pal, age, flush=True)
print("[regen] done", flush=True)
