"""
make_texture_sets.py - turns raw GNM-UV bakes into finished texture sets.

    blender -b --factory-startup --python make_texture_sets.py -- --bake textures/_bake --out textures

Each set  textures/setNN_<name>/  is a complete GNM UDIM texture set:
    face.1001.png             skin albedo (photographic MakeHuman CC0 skin baked onto GNM UVs,
                              eyebrows composited, wet conjunctiva in the eye sockets)
    face.1002.png / 1004.png  left / right eye interior: distinct procedural irises + sclera
    face.1003.png / 1005.png  left / right cornea (clear)
    face.1006.png / 1007.png  teeth+gums / tongue
    face_normal.1001.png      tangent-space pore + fine-wrinkle normal map (high-frequency detail)
    face_roughness.1001.png   roughness (oily T-zone, glossy lips, wet sockets)
    set.json                  provenance
The file naming follows GNM's UDIM convention (face.<UDIM>.png), so any GNM head with the
official UDIM UVs can use a set directly.
"""
import argparse
import json
import os
import shutil
import sys

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "gnm_head"))
import textures as tx  # noqa: E402

# id, MakeHuman skin, iris palette, eyebrow hair colour (sRGB), age factor (0 young .. 1 old)
SETS = [
    ("set01", "young_caucasian_male", "blue", (0.30, 0.21, 0.14), 0.0),
    ("set02", "young_caucasian_female", "green", (0.36, 0.25, 0.16), 0.0),
    ("set03", "young_african_male", "dark_brown", (0.06, 0.045, 0.04), 0.0),
    ("set04", "young_african_female", "dark_brown", (0.07, 0.05, 0.045), 0.0),
    ("set05", "young_asian_female", "dark_brown", (0.07, 0.055, 0.05), 0.0),
    ("set06", "young_asian_male", "brown", (0.06, 0.05, 0.045), 0.0),
    ("set07", "middleage_caucasian_male", "hazel", (0.26, 0.20, 0.15), 0.5),
    ("set08", "middleage_african_female", "brown", (0.08, 0.06, 0.05), 0.5),
    ("set09", "old_caucasian_female", "grey", (0.55, 0.52, 0.49), 1.0),
    ("set10", "old_asian_male", "brown", (0.45, 0.43, 0.41), 1.0),
]


# --------------------------------------------------------------------------- #
def load(path):
    img = bpy.data.images.load(path)
    img.colorspace_settings.name = "Non-Color"   # raw stored values
    w, h = img.size
    a = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(a)
    bpy.data.images.remove(img)
    return a.reshape(h, w, 4)[::-1, :, :3].copy()   # top row first


def save(path, arr, non_color=False):
    arr = np.asarray(arr, np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, -1)
    h, w = arr.shape[:2]
    img = bpy.data.images.new(os.path.basename(path), w, h, alpha=False)
    img.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
    rgba = np.ones((h, w, 4), np.float32)
    rgba[..., :3] = np.clip(arr, 0, 1)
    img.pixels.foreach_set(rgba[::-1].ravel())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


def blur(a, s):
    return tx.blur(a.astype(np.float32), s).astype(np.float32)


def push_pull_fill(img, valid):
    """Fill invalid pixels from valid neighbours (multi-resolution push-pull)."""
    levels = [(img * valid[..., None], valid.astype(np.float32))]
    while levels[-1][1].shape[0] > 4:
        c, w = levels[-1]
        h2, w2 = c.shape[0] // 2, c.shape[1] // 2
        c2 = c[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2, -1).sum((1, 3))
        w2_ = w[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2).sum((1, 3))
        levels.append((c2, w2_))
    fill = levels[-1][0] / np.maximum(levels[-1][1], 1e-6)[..., None]
    for c, w in reversed(levels[:-1]):
        up = np.repeat(np.repeat(fill, 2, 0), 2, 1)[:c.shape[0], :c.shape[1]]
        cur = c / np.maximum(w, 1e-6)[..., None]
        k = np.clip(w, 0, 1)[..., None]
        fill = cur * k + up * (1 - k)
    return np.where(valid[..., None], img, fill)


# --------------------------------------------------------------------------- #
# v8 'orange peel' relief.  0.30 / (1.6, 4.5) was tuned at 1024 and read as pebbling at 2048
# (17 Nov 2026 fix list); see tools_regen_normal.py for the A/B.
# v8 values were PEEL 0.30 (1.6, 4.5), pores 1.1 / 2.0 / 0.012 / +0.25, k 1.3: pebbled at 2048.
# 17 Nov 'D2': v7 pore sizes + light fine peel; BIG_DENS stays 0.012 so the random pore positions
# still match the v8 albedo's pore darkening (mid-scale bump energy on the cheek -21%).
PEEL_W = 0.10
PEEL_BAND = (1.0, 3.0)
# pore blur (x sc), big-pore blur / density / base weight outside nose+cheeks, normal strength
PORE_BLUR = 0.9
BIG_BLUR = 1.7
BIG_DENS = 0.012
BIG_BASE = 0.0
NORMAL_K = 1.1


def detail_maps(size, ra, rb, seed):
    """Pore / wrinkle height field -> (normal map, pore map) in UV space."""
    rng = np.random.default_rng(seed)
    sc = size / 4096.0
    socket, mouth = ra[..., 0], ra[..., 1]
    lips, nose, cheeks = rb[..., 0], rb[..., 1], rb[..., 2]
    skin = np.clip(1 - socket - mouth, 0, 1)

    def impulses(density):
        imp = np.zeros((size, size), np.float32)
        n = int(size * size * density)
        ys, xs = rng.integers(0, size, n), rng.integers(0, size, n)
        imp[ys, xs] = rng.uniform(0.5, 1.0, n).astype(np.float32)
        return imp

    pores = blur(impulses(0.030), PORE_BLUR * sc)
    big = blur(impulses(BIG_DENS), BIG_BLUR * sc) * np.clip(nose * 1.2 + cheeks * 0.8 + BIG_BASE, 0, 1)
    pores = pores / max(pores.max(), 1e-6) + 1.6 * big / max(big.max(), 1e-6)
    pores *= skin * (1 - 0.85 * lips)
    # fine cross-hatched skin wrinkles + vertical lip creases
    n1 = rng.standard_normal((size, size)).astype(np.float32)
    wr = blur(n1, 1.4 * sc) - blur(n1, 4.0 * sc)
    n2 = rng.standard_normal((size, size)).astype(np.float32)
    lipc = _blur_xy(n2, 0.8 * sc, 7.0 * sc)   # creases run vertically across the lips
    # v8: mid-scale 'orange peel' relief (0.3-1 mm) - breaks up specular highlights at video scale
    n3 = rng.standard_normal((size, size)).astype(np.float32)
    peel = blur(n3, PEEL_BAND[0] * sc) - blur(n3, PEEL_BAND[1] * sc)
    peel = peel / max(np.abs(peel).max(), 1e-6) * skin * (1 - lips)
    height = -pores + 0.35 * wr / max(np.abs(wr).max(), 1e-6) * skin + PEEL_W * peel \
        + 0.8 * lipc / max(np.abs(lipc).max(), 1e-6) * lips
    height = height.astype(np.float32)
    gy, gx = np.gradient(height)          # rows go down = -v
    k = NORMAL_K * sc
    nx, ny = -k * gx, k * gy
    nz = np.ones_like(nx)
    nrm = np.sqrt(nx * nx + ny * ny + nz * nz)
    normal = np.stack([nx / nrm, ny / nrm, nz / nrm], -1) * 0.5 + 0.5
    return normal, np.clip(pores, 0, 2) / 2


def _blur_xy(a, sx, sy):
    a = a.astype(np.float32)
    for axis, s in ((1, sx), (0, sy)):
        r = int(3 * s) + 1
        kk = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2)
        kk /= kk.sum()
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        p = np.pad(a, pad, mode="edge")
        a = sum(kk[i] * (p[i:i + a.shape[0]] if axis == 0 else p[:, i:i + a.shape[1]]) for i in range(2 * r + 1))
    return a.astype(np.float32)


# --------------------------------------------------------------------------- #
# v8 skin realism: colour variation real skin has and the MakeHuman bakes lack
# --------------------------------------------------------------------------- #
def beard_mask(size):
    """Stubble region in GNM UV space (upper lip, chin, jaw up to the sideburns), soft-edged.
    Landmarks measured on the 4096 GNM layout: face centre u=2048 px, nose base y~2010,
    mouth corners ~250 px off-centre at y~2120, sideburns ~1100 px off-centre at y~1650."""
    sc = size / 4096.0
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / sc
    dx = np.abs(x - 2048.0)
    top = np.where(dx < 250, 2015.0 + 0.10 * dx, 2040.0 - (dx - 250) * (230.0 / 850.0))
    bot = 2760.0 - 0.45 * dx
    m = ((y > top) & (y < bot) & (dx < 1150)).astype(np.float32)
    m = blur(m, 40 * sc)
    # thinner toward the cheek line and the neck
    m *= np.clip((y - top) / (170.0), 0, 1) ** 1.2 * np.clip((bot - y) / 220.0, 0, 1)
    return np.clip(m, 0, 1)


def skin_realism(alb, ra, rb, rng, male, age, hair):
    size = alb.shape[0]
    sc = size / 4096.0
    socket, mouth = ra[..., 0], ra[..., 1]
    lips, nose, cheeks = rb[..., 0], rb[..., 1], rb[..., 2]
    face_lum = float(np.median(alb[1400:2400, 1500:2600].mean(-1))) if size == 4096 else 0.5
    fair = np.clip((face_lum - 0.25) / 0.45, 0, 1)        # 0 dark skin .. 1 very fair
    red = np.array([1.0, -0.55, -0.45], np.float32)       # haemoglobin tint direction

    def tint(a, mask, amt, direction):
        return a * (1 + amt * mask[..., None] * direction)

    # haemoglobin mottling: low-frequency blotches, strongest on fair skin
    mott = tx.fbm(size, size, rng, base=24, octaves=4).astype(np.float32)
    mott = (mott - mott.mean()) / max(mott.std(), 1e-6)
    alb = tint(alb, np.clip(mott, 0, None), 0.018 * (0.4 + 0.6 * fair), red)
    lum_n = tx.fbm(size, size, rng, base=48, octaves=3).astype(np.float32)
    alb = alb * (1 + 0.015 * np.clip((lum_n - lum_n.mean()) / max(lum_n.std(), 1e-6), -2.5, 2.5))[..., None]
    # flushed cheeks, nose tip, chin
    alb = tint(alb, blur(cheeks, 60 * sc), 0.06 * (0.3 + 0.7 * fair), red)
    alb = tint(alb, blur(nose, 30 * sc), 0.05 * (0.3 + 0.7 * fair), red)
    # eyelids: pinker and slightly darker; under-eye shadow (faint violet-brown)
    ring = np.clip(blur(socket, 55 * sc) * 2.2 - socket * 3.0, 0, 1)
    alb = tint(alb, ring, 0.07 * (0.3 + 0.7 * fair), red)
    alb = alb * (1 - 0.07 * ring[..., None])
    under = np.roll(ring, int(60 * sc), axis=0) * (1 - ring)
    alb = alb * (1 - 0.06 * under[..., None] * np.array([0.8, 1.0, 0.7]))
    # freckles / melanin spots
    n = int(size * size * 1.2e-4)
    imp = np.zeros((size, size), np.float32)
    ys, xs = rng.integers(0, size, n), rng.integers(0, size, n)
    imp[ys, xs] = rng.uniform(0.3, 1.0, n)
    fr = blur(imp, 2.2 * sc)
    fr = np.clip(fr / max(np.percentile(fr[fr > 0], 99), 1e-6), 0, 1) * (1 - lips) * (1 - socket)
    fr *= 0.5 + 0.5 * np.clip(blur(cheeks + nose, 80 * sc) * 2, 0, 1)
    alb = alb * (1 - (0.10 + 0.10 * fair) * fr[..., None] * np.array([0.6, 0.85, 1.0]))
    # stubble: shaved-beard shadow + individual follicle dots
    if male:
        bm = beard_mask(size) * (1 - np.clip(blur(lips, 6 * sc) * 1.5, 0, 1)) * (1 - nose)
        hc = np.array(hair, np.float32)
        grey = hc.mean() > 0.35                                  # white/grey stubble
        shadow = (0.05 if grey else 0.10) * (0.5 + 0.5 * fair)
        alb = alb * (1 - shadow * bm[..., None] * np.array([1.0, 0.86, 0.74]))   # blue-grey
        n = int(size * size * 0.010)
        imp = np.zeros((size, size), np.float32)
        ys, xs = rng.integers(0, size, n), rng.integers(0, size, n)
        imp[ys, xs] = rng.uniform(0.5, 1.0, n)
        dots = np.clip(blur(imp, 0.6 * sc) * 3.0, 0, 1) * bm
        col = hc * (1.6 if grey else 0.5)
        k = (0.35 if grey else 0.45) * dots[..., None]
        alb = alb * (1 - k) + col * k
    return np.clip(alb, 0, 1)


# --------------------------------------------------------------------------- #
def make_set(bake, out_root, sid, skin, palette, hair, age, size_eye, shared):
    ra, rb, brow_a, brow_rgb, normal, pores = shared
    raw = load(os.path.join(bake, skin, "skin_albedo_raw.png"))
    size = raw.shape[0]
    valid = raw.sum(-1) > 0.004
    alb = push_pull_fill(raw, valid)
    socket = blur(ra[..., 0], 2.0 * size / 4096)[..., None]
    mouth = blur(ra[..., 1], 2.0 * size / 4096)[..., None]
    # pores darker (cavities catch less light)
    alb = alb * (1 - 0.10 * pores[..., None])
    alb = skin_realism(alb, ra, rb, np.random.default_rng(int(sid[3:]) * 31 + 5),
                       male=skin.endswith("_male"), age=age, hair=hair)
    # eyebrows
    a = np.clip(brow_a[..., :1] * 1.15, 0, 1) * 0.9
    lum = brow_rgb.mean(-1, keepdims=True)
    hair_c = np.array(hair, np.float32) * (0.65 + 0.7 * lum)
    alb = alb * (1 - a) + hair_c * a
    # wet conjunctiva in the socket lining, dark red mouth interior
    alb = alb * (1 - 0.75 * socket) + np.array([0.80, 0.52, 0.50]) * 0.75 * socket
    alb = alb * (1 - 0.85 * mouth) + np.array([0.22, 0.09, 0.08]) * 0.85 * mouth
    d = os.path.join(out_root, "%s_%s" % (sid, skin))
    save(os.path.join(d, "face.1001.png"), alb)

    # roughness
    tz, lips = ra[..., 2], rb[..., 0]
    rough = 0.52 + 0.04 * age - 0.11 * tz - 0.05 * rb[..., 1] - 0.16 * lips + 0.06 * pores
    rough = rough * (1 - socket[..., 0]) + 0.06 * socket[..., 0]       # wet conjunctiva / caruncle
    rough = rough * (1 - mouth[..., 0]) + 0.18 * mouth[..., 0]
    rough += 0.03 * (blur(np.random.default_rng(len(sid)).standard_normal(tz.shape).astype(np.float32), 6) * 8)
    save(os.path.join(d, "face_roughness.1001.png"), np.clip(rough, 0.05, 0.9), non_color=True)
    save(os.path.join(d, "face_normal.1001.png"), normal, non_color=True)

    # eyes: distinct L/R irises, same colour family; older eyes get a slightly yellower sclera
    seed = int(sid[3:])
    for tile, ns, sd in ((1002, -1.0, seed * 2 + 1), (1004, +1.0, seed * 2 + 2)):
        save(os.path.join(d, "face.%d.png" % tile),
             tx.eye_tile(size_eye, palette, sd, nasal_sign=ns, sclera_tint=0.6 * age))
    for tile in (1003, 1005):
        save(os.path.join(d, "face.%d.png" % tile), np.ones((64, 64, 3)))
    for tile, nm in ((1006, "teeth"), (1007, "tongue")):
        t = load(os.path.join(bake, nm + ".png"))
        if nm == "teeth":
            t = t * np.array([1.0, 0.98, 0.94 - 0.08 * age])  # teeth yellow with age
        save(os.path.join(d, "face.%d.png" % tile), t)
    with open(os.path.join(d, "set.json"), "w") as f:
        json.dump({"id": sid, "skin_source": "MakeHuman system assets (CC0): skins/%s" % skin,
                   "iris_palette": palette, "eyebrow_colour_srgb": hair, "age_factor": age,
                   "uv_layout": "GNM v3 UDIM (1001 skin, 1002/1003 L eye interior/cornea, "
                                "1004/1005 R, 1006 teeth+gums, 1007 tongue)",
                   "maps": sorted(os.listdir(d))}, f, indent=1)
    print("[sets] wrote", d, flush=True)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake", default=os.path.join(HERE, "textures", "_bake"))
    ap.add_argument("--out", default=os.path.join(HERE, "textures"))
    ap.add_argument("--only", default="")
    ap.add_argument("--eye-size", type=int, default=2048)
    a = ap.parse_args(argv)
    ra = load(os.path.join(a.bake, "regions_a.png"))
    rb = load(os.path.join(a.bake, "regions_b.png"))
    brow_a = load(os.path.join(a.bake, "brow_alpha.png"))
    brow_rgb = load(os.path.join(a.bake, "brow_rgb.png"))
    normal, pores = detail_maps(ra.shape[0], ra, rb, seed=11)
    shared = (ra, rb, brow_a, brow_rgb, normal, pores)
    only = set(x for x in a.only.split(",") if x)
    for sid, skin, pal, hair, age in SETS:
        if only and sid not in only and skin not in only:
            continue
        if not os.path.isdir(os.path.join(a.bake, skin)):
            print("[sets] skip %s (no bake)" % skin, flush=True)
            continue
        make_set(a.bake, a.out, sid, skin, pal, hair, age, a.eye_size, shared)


if __name__ == "__main__":
    main()
