"""
textures.py - procedural texture synthesis in GNM's UDIM layout (numpy only).

Eye interior tiles (1002 left, 1004 right) use GNM's planar eye UVs: centre (0.5, 0.5),
pupil radius 0.05, limbus (iris edge) radius 0.185, +u = subject's left, +v = up,
31 mm per UV unit.  Each eye gets its own random iris (crypts, radial fibres,
collarette, contraction furrows, limbal ring) and sclera (veins on the exposed
nasal / temporal sides, pinker periphery), so left and right irises are distinct
like a real person's while sharing one colour family.

    blender_python textures.py eyes --palette hazel --seed 7 --size 2048 --out textures/set01
"""
import argparse
import math
import os
import struct
import zlib

import numpy as np

R_PUPIL = 0.05
R_LIMBUS = 0.185

# sRGB colours: inner (pupillary zone), outer (ciliary zone), fibre highlight, collarette, limbal ring
PALETTES = {
    "dark_brown": dict(inner=(0.14, 0.075, 0.035), outer=(0.20, 0.11, 0.05), fibre=(0.34, 0.20, 0.10),
                       collar=(0.30, 0.16, 0.07), limbal=(0.05, 0.03, 0.02)),
    "brown":      dict(inner=(0.26, 0.14, 0.05), outer=(0.33, 0.19, 0.08), fibre=(0.52, 0.33, 0.16),
                       collar=(0.47, 0.26, 0.09), limbal=(0.08, 0.05, 0.03)),
    "hazel":      dict(inner=(0.44, 0.26, 0.08), outer=(0.33, 0.34, 0.18), fibre=(0.60, 0.54, 0.32),
                       collar=(0.55, 0.34, 0.10), limbal=(0.10, 0.09, 0.05)),
    "green":      dict(inner=(0.40, 0.31, 0.12), outer=(0.24, 0.38, 0.25), fibre=(0.48, 0.62, 0.44),
                       collar=(0.50, 0.40, 0.15), limbal=(0.07, 0.11, 0.08)),
    "blue":       dict(inner=(0.34, 0.37, 0.38), outer=(0.22, 0.38, 0.56), fibre=(0.58, 0.71, 0.83),
                       collar=(0.48, 0.50, 0.48), limbal=(0.08, 0.12, 0.20)),
    "grey":       dict(inner=(0.36, 0.36, 0.33), outer=(0.38, 0.42, 0.46), fibre=(0.66, 0.69, 0.71),
                       collar=(0.50, 0.47, 0.40), limbal=(0.10, 0.11, 0.13)),
    "amber":      dict(inner=(0.50, 0.30, 0.08), outer=(0.58, 0.38, 0.11), fibre=(0.78, 0.56, 0.22),
                       collar=(0.62, 0.36, 0.09), limbal=(0.14, 0.08, 0.03)),
}


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def write_png(path, img):
    """img: float [0,1] (H,W,3|4) or uint8. Minimal PNG writer (no PIL in Blender's python)."""
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    if a.ndim == 2:
        a = a[:, :, None]
    h, w, c = a.shape
    color_type = {1: 0, 3: 2, 4: 6}[c]
    raw = np.concatenate([np.zeros((h, 1), np.uint8), a.reshape(h, w * c)], axis=1).tobytes()

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(chunk(b"IEND", b""))


def read_png(path):
    """Minimal PNG reader for 8-bit non-interlaced RGB/RGBA/grey files -> uint8 (H,W,C)."""
    with open(path, "rb") as f:
        data = f.read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", path
    pos, idat, hdr = 8, [], None
    while pos < len(data):
        n = struct.unpack(">I", data[pos:pos + 4])[0]
        t = data[pos + 4:pos + 8]
        d = data[pos + 8:pos + 8 + n]
        if t == b"IHDR":
            hdr = struct.unpack(">IIBBBBB", d)
        elif t == b"IDAT":
            idat.append(d)
        pos += 12 + n
    w, h, depth, ctype, _, _, interlace = hdr
    assert depth == 8 and interlace == 0, "only 8-bit non-interlaced PNG supported"
    c = {0: 1, 2: 3, 4: 2, 6: 4}[ctype]
    raw = np.frombuffer(zlib.decompress(b"".join(idat)), np.uint8).reshape(h, w * c + 1)
    out = np.zeros((h, w * c), np.int32)
    prev = np.zeros(w * c, np.int32)
    for y in range(h):
        ft, line = raw[y, 0], raw[y, 1:].astype(np.int32)
        if ft == 0:
            cur = line
        elif ft == 2:
            cur = (line + prev) & 255
        else:
            cur = np.zeros_like(line)
            for x in range(w * c):
                a = cur[x - c] if x >= c else 0
                b = prev[x]
                cc = prev[x - c] if x >= c else 0
                if ft == 1:
                    cur[x] = (line[x] + a) & 255
                elif ft == 3:
                    cur[x] = (line[x] + ((a + b) >> 1)) & 255
                else:
                    p = a + b - cc
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - cc)
                    pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else cc)
                    cur[x] = (line[x] + pr) & 255
        out[y] = cur
        prev = cur
    return out.astype(np.uint8).reshape(h, w, c)


# --------------------------------------------------------------------------- #
# noise helpers
# --------------------------------------------------------------------------- #
def _interp_matrix(n_out, n_cells):
    x = (np.arange(n_out) + 0.5) / n_out * n_cells
    i0 = np.clip(np.floor(x).astype(int), 0, n_cells - 1)
    t = x - i0
    t = t * t * (3 - 2 * t)
    M = np.zeros((n_out, n_cells + 1))
    M[np.arange(n_out), i0] = 1 - t
    M[np.arange(n_out), i0 + 1] = t
    return M


def value_noise(h, w, cells, rng):
    g = rng.random((cells + 1, cells + 1))
    return _interp_matrix(h, cells) @ g @ _interp_matrix(w, cells).T


def fbm(h, w, rng, base=4, octaves=5, gain=0.5):
    out = np.zeros((h, w))
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        out += amp * value_noise(h, w, base * 2 ** o, rng)
        tot += amp
        amp *= gain
    return out / tot


def periodic_noise(angle, rng, n=24, fmin=2, fmax=24, extra=None):
    """Smooth noise periodic in angle (radians); optional extra coordinate warps phases."""
    out = np.zeros_like(angle)
    for _ in range(n):
        f = rng.integers(fmin, fmax + 1)
        ph = rng.random() * 2 * math.pi
        k = 0 if extra is None else rng.normal(0, 3.0)
        out += math.sqrt(1.0 / f) * np.sin(f * angle + ph + (0 if extra is None else k * extra))
    return out / np.abs(out).max()


def blur(img, sigma_px):
    """Separable gaussian blur (numpy)."""
    if sigma_px <= 0:
        return img
    r = int(3 * sigma_px) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_px) ** 2)
    k /= k.sum()
    pad = [(r, r), (0, 0)] + ([(0, 0)] if img.ndim == 3 else [])
    a = np.pad(img, pad, mode="edge")
    a = sum(k[i] * a[i:i + img.shape[0]] for i in range(2 * r + 1))
    pad = [(0, 0), (r, r)] + ([(0, 0)] if img.ndim == 3 else [])
    a = np.pad(a, pad, mode="edge")
    return sum(k[i] * a[:, i:i + img.shape[1]] for i in range(2 * r + 1))


def srgb_to_lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


# --------------------------------------------------------------------------- #
# eye interior tile
# --------------------------------------------------------------------------- #
def eye_tile(size=2048, palette="brown", seed=0, nasal_sign=-1.0, sclera_tint=0.0):
    """RGB float image (sRGB) of one eye-interior UDIM tile.

    nasal_sign: -1 if the nose is toward -u (left eye, subject's left = +u), +1 for the right eye.
    """
    rng = np.random.default_rng(seed)
    P = {k: srgb_to_lin(v) for k, v in PALETTES[palette].items()}
    # jitter the palette per eye (real L/R irises differ slightly in pigment)
    jit = 1.0 + rng.normal(0, 0.05, 3)
    for k in ("inner", "outer", "fibre"):
        P[k] = np.clip(P[k] * jit, 0, 1)

    px = (np.arange(size) + 0.5) / size
    U, Vv = np.meshgrid(px, 1.0 - px)            # row 0 = top = v 1
    du, dv = U - 0.5, Vv - 0.5
    r = np.hypot(du, dv)
    ang = np.arctan2(dv, du)
    rho = (r - R_PUPIL) / (R_LIMBUS - R_PUPIL)   # 0 at pupil edge, 1 at limbus

    img = np.zeros((size, size, 3))

    # ---------------- sclera ----------------
    rs = np.clip((r - R_LIMBUS) / (0.46 - R_LIMBUS), 0, 1)
    sclera_base = srgb_to_lin((0.93, 0.895, 0.85))
    sclera_edge = srgb_to_lin((0.86, 0.70, 0.66))
    # conjunctiva turns pink only toward the fornix (>~55 deg from the axis, hidden by the lids)
    t = (0.85 * smooth(0.62, 1.0, rs))[..., None]
    scl = sclera_base * (1 - t) + sclera_edge * t
    # bluish-grey band just outside the limbus, pinker toward the nasal side (caruncle)
    near_limbus = np.exp(-((r - R_LIMBUS - 0.012) / 0.012) ** 2)[..., None]
    scl = scl * (1 - 0.10 * near_limbus) + srgb_to_lin((0.78, 0.80, 0.84)) * 0.10 * near_limbus
    nasal = np.clip(nasal_sign * du / 0.46, 0, 1) ** 2 * smooth(0.25, 0.9, rs)
    scl = scl * (1 - 0.25 * nasal[..., None]) + srgb_to_lin((0.85, 0.55, 0.52)) * 0.25 * nasal[..., None]
    mott = fbm(size, size, rng, base=6, octaves=5)[..., None]
    scl *= 0.94 + 0.10 * mott
    scl = scl * (1 + sclera_tint * np.array([0.04, 0.02, -0.06]))
    # veins: random walks from the periphery toward the limbus, mostly on exposed sides
    veins = np.zeros((size, size))
    n_veins = rng.integers(22, 34)
    for _ in range(n_veins):
        side = rng.choice([0.0, math.pi])
        a0 = side + rng.normal(0, 0.42)
        rr = rng.uniform(0.40, 0.47)
        width = rng.uniform(1.6, 3.2) * size / 2048
        strength = rng.uniform(0.35, 0.9)
        _walk_vein(veins, rr, a0, width, strength, size, rng, depth=0)
    veins = np.clip(blur(veins, 0.8 * size / 2048), 0, 1)
    vein_col = srgb_to_lin((0.62, 0.12, 0.10))
    vmask = (veins * smooth(0.02, 0.35, rs))[..., None]
    scl = scl * (1 - 0.55 * vmask) + vein_col * 0.55 * vmask

    # ---------------- iris ----------------
    rho_c = np.clip(rho, 0, 1)
    base = P["inner"] * (1 - smooth(0.25, 0.45, rho_c))[..., None] + \
        P["outer"] * smooth(0.25, 0.45, rho_c)[..., None]
    # radial fibres: high angular frequency, slowly varying with radius
    fib = np.zeros_like(r)
    m_iris = r < R_LIMBUS * 1.08
    a_i, rho_i = ang[m_iris], rho_c[m_iris]
    acc = np.zeros_like(a_i)
    for _ in range(90):
        f = rng.integers(30, 260)
        ph = rng.random() * 2 * math.pi
        bend = rng.normal(0, 1.2)
        acc += np.sin(f * a_i + ph + bend * rho_i) / math.sqrt(f)
    acc = acc / np.abs(acc).max()
    fib[m_iris] = acc
    fib = 0.5 + 0.5 * fib
    fib = fib ** 1.6
    # radial modulation: fibres stronger in the ciliary zone
    fib_amt = (0.25 + 0.55 * smooth(0.3, 0.7, rho_c) * (1 - smooth(0.85, 1.0, rho_c)))[..., None]
    iris = base * (1 - fib_amt * fib[..., None]) + P["fibre"] * fib_amt * fib[..., None]
    # collarette: wavy lighter ring
    rc = 0.33 + 0.045 * periodic_noise(ang, rng, n=20, fmin=5, fmax=28)
    col = np.exp(-((rho_c - rc) / 0.035) ** 2)[..., None]
    iris = iris * (1 - 0.55 * col) + P["collar"] * 0.55 * col
    # crypts: dark radially elongated pits
    crypt = np.zeros_like(r)
    for _ in range(rng.integers(28, 46)):
        cr = rng.uniform(0.36, 0.86)
        ca = rng.uniform(-math.pi, math.pi)
        da = np.angle(np.exp(1j * (ang - ca))) * (R_PUPIL + cr * (R_LIMBUS - R_PUPIL))
        dr = (rho_c - cr) * (R_LIMBUS - R_PUPIL)
        sa, sr = rng.uniform(0.0025, 0.007), rng.uniform(0.004, 0.012)
        crypt = np.maximum(crypt, rng.uniform(0.4, 0.9) * np.exp(-(da / sa) ** 2 - (dr / sr) ** 2))
    iris *= (1 - 0.6 * crypt)[..., None]
    # contraction furrows: broken concentric dark rings in the outer zone
    furrow = np.zeros_like(r)
    for fr in rng.uniform(0.6, 0.92, rng.integers(2, 5)):
        brk = smooth(0.0, 0.6, periodic_noise(ang, rng, n=10, fmin=3, fmax=14))
        furrow += np.exp(-((rho_c - fr) / 0.018) ** 2) * brk
    iris *= (1 - 0.18 * np.clip(furrow, 0, 1))[..., None]
    # limbal ring (darkens toward the edge) and pupillary ruff
    lim = smooth(0.80, 1.0, rho_c)[..., None]
    iris = iris * (1 - 0.75 * lim) + P["limbal"] * 0.75 * lim
    ruff = np.exp(-(np.clip(rho, 0, None) / 0.035) ** 2)[..., None] * (r > R_PUPIL)[..., None]
    iris = iris * (1 - 0.6 * ruff) + srgb_to_lin((0.10, 0.06, 0.03)) * 0.6 * ruff
    iris *= (0.88 + 0.24 * fbm(size, size, rng, base=24, octaves=3))[..., None]

    # ---------------- compose ----------------
    w_iris = (1 - smooth(R_LIMBUS - 0.004, R_LIMBUS + 0.006, r))[..., None]  # soft limbus
    img = scl * (1 - w_iris) + iris * w_iris
    pupil = (1 - smooth(R_PUPIL - 0.003, R_PUPIL + 0.001, r))[..., None]
    img = img * (1 - pupil) + srgb_to_lin((0.012, 0.010, 0.010)) * pupil
    return lin_to_srgb(img)


def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _walk_vein(canvas, r0, a0, width, strength, size, rng, depth):
    r, a = r0, a0
    steps = int(rng.integers(40, 90))
    da = rng.normal(0, 0.004)
    for i in range(steps):
        w = max(0.8, width * (1 - 0.6 * i / steps))
        s = strength * (1 - 0.6 * i / steps)
        dr = rng.uniform(0.0016, 0.0032)
        da += rng.normal(0, 0.0035)
        # draw the segment with splats spaced <= w/3 px so the vessel is continuous
        seg_px = math.hypot(dr, r * da) * size
        n_sub = max(1, int(seg_px / (w / 3)))
        for k in range(n_sub):
            f = k / n_sub
            rr, aa = r - f * dr, a + f * da
            _splat(canvas, (0.5 + rr * math.cos(aa)) * size, (0.5 - rr * math.sin(aa)) * size, w, s)
        r -= dr
        a += da
        if r < R_LIMBUS + 0.03:
            break
        if depth < 2 and rng.random() < 0.035:
            _walk_vein(canvas, r, a + rng.normal(0, 0.25), w * 0.7, strength * 0.7, size, rng, depth + 1)


def _splat(canvas, cx, cy, w, s):
    h, wd = canvas.shape
    rad = int(2 * w) + 1
    x0, x1 = max(0, int(cx) - rad), min(wd, int(cx) + rad + 1)
    y0, y1 = max(0, int(cy) - rad), min(h, int(cy) + rad + 1)
    if x0 >= x1 or y0 >= y1:
        return
    xs = np.arange(x0, x1) + 0.5 - cx
    ys = np.arange(y0, y1) + 0.5 - cy
    g = np.exp(-(ys[:, None] ** 2 + xs[None, :] ** 2) / (0.5 * w * w)) * s
    canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], g)


def write_eye_tiles(out_dir, palette="brown", seed=0, size=2048, sclera_tint=0.0):
    """Writes face.1002.png (left eye) and face.1004.png (right eye) with distinct irises,
    plus clear-cornea tiles face.1003/1005.png."""
    os.makedirs(out_dir, exist_ok=True)
    write_png(os.path.join(out_dir, "face.1002.png"),
              eye_tile(size, palette, seed * 2 + 1, nasal_sign=-1.0, sclera_tint=sclera_tint))
    write_png(os.path.join(out_dir, "face.1004.png"),
              eye_tile(size, palette, seed * 2 + 2, nasal_sign=+1.0, sclera_tint=sclera_tint))
    clear = np.ones((64, 64, 3))
    write_png(os.path.join(out_dir, "face.1003.png"), clear)
    write_png(os.path.join(out_dir, "face.1005.png"), clear)


def write_placeholder_tiles(out_dir, skin=(0.80, 0.62, 0.52)):
    """Flat tiles for any of 1001 (skin), 1006 (teeth), 1007 (tongue) that are missing."""
    for tile, col in ((1001, skin), (1006, (0.88, 0.84, 0.76)), (1007, (0.62, 0.30, 0.30))):
        p = os.path.join(out_dir, "face.%d.png" % tile)
        if not os.path.isfile(p):
            write_png(p, np.ones((64, 64, 3)) * np.array(col))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["eyes"])
    ap.add_argument("--palette", default="brown", choices=sorted(PALETTES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.what == "eyes":
        write_eye_tiles(a.out, a.palette, a.seed, a.size)
        print("wrote eye tiles to", a.out)
