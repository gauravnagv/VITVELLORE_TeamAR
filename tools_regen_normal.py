"""Regenerate the shared skin detail normal map (face_normal.1001.png) with new 'peel' settings.

    blender -b --python tools_regen_normal.py -- --peel 0.15 --band 1.0,3.0 --out DIR
    blender -b --python tools_regen_normal.py -- --peel 0.15 --band 1.0,3.0 --install

The normal map is identical in every texture set (detail_maps(seed=11) from the bake regions),
so only this one file changes; albedo/roughness/eye tiles are untouched.  --install backs up the
current map once to textures_v8_normal/ and copies the new one into every textures/<set>/.
"""
import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_texture_sets as mts  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--peel", type=float, required=True)
ap.add_argument("--band", default="1.6,4.5")
ap.add_argument("--pores", default="", help="PORE_BLUR,BIG_BLUR,BIG_DENS,BIG_BASE (default: module values)")
ap.add_argument("--k", type=float, default=None, help="NORMAL_K")
ap.add_argument("--out", default=None)
ap.add_argument("--install", action="store_true")
ap.add_argument("--bake", default=os.path.join(HERE, "textures", "_bake"))
a = ap.parse_args(argv)

mts.PEEL_W = a.peel
mts.PEEL_BAND = tuple(float(x) for x in a.band.split(","))
if a.pores:
    mts.PORE_BLUR, mts.BIG_BLUR, mts.BIG_DENS, mts.BIG_BASE = (float(x) for x in a.pores.split(","))
if a.k is not None:
    mts.NORMAL_K = a.k
ra = mts.load(os.path.join(a.bake, "regions_a.png"))
rb = mts.load(os.path.join(a.bake, "regions_b.png"))
normal, _ = mts.detail_maps(ra.shape[0], ra, rb, seed=11)
out = a.out or os.path.join(HERE, "textures_normal_new")
dst = os.path.join(out, "face_normal.1001.png")
mts.save(dst, normal, non_color=True)
print("[normal] peel %.3f band %s pores %s k %.2f -> %s" % (mts.PEEL_W, mts.PEEL_BAND, (mts.PORE_BLUR, mts.BIG_BLUR,
      mts.BIG_DENS, mts.BIG_BASE), mts.NORMAL_K, dst), flush=True)

if a.install:
    tex = os.path.join(HERE, "textures")
    sets = sorted(d for d in os.listdir(tex) if d.startswith("set"))
    bak = os.path.join(HERE, "textures_v8_normal", "face_normal.1001.png")
    if not os.path.isfile(bak):
        os.makedirs(os.path.dirname(bak), exist_ok=True)
        shutil.copy2(os.path.join(tex, sets[0], "face_normal.1001.png"), bak)
        print("[normal] backed up v8 map ->", bak, flush=True)
    for s in sets:
        shutil.copy2(dst, os.path.join(tex, s, "face_normal.1001.png"))
    print("[normal] installed into %d sets" % len(sets), flush=True)
