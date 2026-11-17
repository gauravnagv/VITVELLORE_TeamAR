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
# --------------------------------------------------------------------------- #
# v8 iris: synthesised in polar space (rows = radius, columns = angle) so fibres are real
# anisotropic streaks rather than summed sines, then resampled onto the planar eye UVs.
# --------------------------------------------------------------------------- #
_RHO0, _RHO1 = -0.06, 1.14          # polar grid radial range (rho: 0 pupil edge .. 1 limbus)


def _gauss_axis(a, sigma, axis, wrap):
    """Gaussian blur along one axis via FFT (wrap=True: periodic, else reflect-padded)."""
    if sigma <= 0:
        return a
    n = a.shape[axis]
    pad = 0 if wrap else min(n - 1, int(3 * sigma) + 1)
    if pad:
        widths = [(0, 0)] * a.ndim
        widths[axis] = (pad, pad)
        a = np.pad(a, widths, mode="reflect")
    m = a.shape[axis]
    f = np.fft.rfftfreq(m)
    g = np.exp(-2.0 * (math.pi * f * sigma) ** 2)
    shape = [1] * a.ndim
    shape[axis] = len(f)
    out = np.fft.irfft(np.fft.rfft(a, axis=axis) * g.reshape(shape), n=m, axis=axis)
    if pad:
        out = np.take(out, np.arange(pad, pad + n), axis=axis)
    return out


def _streaks(rng, NR, NA, s_ang, s_rad):
    """Unit-variance noise stretched along the radius (iris fibres / stroma bundles)."""
    n = rng.standard_normal((NR, NA))
    n = _gauss_axis(n, s_ang, 1, True)
    n = _gauss_axis(n, s_rad, 0, False)
    return n / max(n.std(), 1e-9)


def _warp_cols(field, offset):
    """Shift every pixel of a polar field along the angle axis by offset (columns, float)."""
    NR, NA = field.shape[:2]
    x = np.arange(NA)[None, :] + offset
    x0 = np.floor(x).astype(int)
    t = x - x0
    rows = np.arange(NR)[:, None]
    a, b = field[rows, x0 % NA], field[rows, (x0 + 1) % NA]
    if field.ndim == 3:
        t = t[..., None]
    return a * (1 - t) + b * t


def _polar_lookup(field, rho, ang):
    """Bilinear lookup of a polar field (NR, NA[, C]) at (rho, ang) arrays (angle wraps)."""
    NR, NA = field.shape[:2]
    y = np.clip((rho - _RHO0) / (_RHO1 - _RHO0) * (NR - 1), 0, NR - 1.001)
    x = (ang % (2 * math.pi)) / (2 * math.pi) * NA
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    ty, tx = y - y0, x - x0
    x0 %= NA
    x1 = (x0 + 1) % NA
    if field.ndim == 3:
        ty, tx = ty[..., None], tx[..., None]
    top = field[y0, x0] * (1 - tx) + field[y0, x1] * tx
    bot = field[y0 + 1, x0] * (1 - tx) + field[y0 + 1, x1] * tx
    return top * (1 - ty) + bot * ty


# central heterochromia: a warm amber/brown ring round the pupil, strong in light irises
_HETERO = {"blue": 0.30, "grey": 0.30, "green": 0.55, "hazel": 0.75, "amber": 0.4,
           "brown": 0.25, "dark_brown": 0.1}


def iris_polar(palette, rng, NR=640, NA=4096):
    """Iris albedo (linear RGB) and relief height on a polar grid (NR x NA).
    Rows span rho in [_RHO0, _RHO1]; columns span angle [0, 2pi)."""
    P = {k: srgb_to_lin(v) for k, v in PALETTES[palette].items()}
    jit = 1.0 + rng.normal(0, 0.05, 3)                 # L/R pigment differs slightly
    for k in ("inner", "outer", "fibre"):
        P[k] = np.clip(P[k] * jit, 0, 1)
    rho = np.linspace(_RHO0, _RHO1, NR)[:, None] * np.ones((1, NA))
    acol = np.arange(NA)[None, :] * np.ones((NR, 1))
    ang = acol / NA * 2 * math.pi
    rpx = NR / (_RHO1 - _RHO0)                          # rows per unit rho

    # --- fibre layers (fine / medium / coarse), domain-warped so they wave and fork ---
    warp = _streaks(rng, NR, NA, 90, 0.18 * rpx) * 9 + _streaks(rng, NR, NA, 14, 0.08 * rpx) * 1.6
    f_fine = _warp_cols(_streaks(rng, NR, NA, 0.9, 0.035 * rpx), warp)
    f_med = _warp_cols(_streaks(rng, NR, NA, 2.6, 0.07 * rpx), warp)
    f_coarse = _warp_cols(_streaks(rng, NR, NA, 8.0, 0.16 * rpx), warp)
    f_bund = _warp_cols(_streaks(rng, NR, NA, 16.0, 0.22 * rpx), warp * 1.5)
    fib = 0.28 * f_fine + 0.40 * f_med + 0.45 * f_coarse + 0.35 * f_bund
    fib = 1.0 / (1.0 + np.exp(-1.25 * fib))             # 0..1, soft-contrast
    ridge = 0.45 * np.clip(f_med - 1.0, 0, None) ** 1.3 + 0.5 * np.clip(f_bund - 0.7, 0, None)        # bright trabeculae (stroma bundles)

    # --- zones: wavy collarette separates pupillary (inner) and ciliary (outer) zones ---
    cpro = periodic_noise(ang[0], rng, n=26, fmin=6, fmax=40)
    rc = (0.33 + 0.055 * cpro)[None, :]
    inner = 1.0 - smooth(rc - 0.07, rc + 0.05, rho)
    collar = np.exp(-((rho - rc) / 0.045) ** 2) * (0.6 + 0.4 * np.clip(f_coarse, -1, 1))
    het = _HETERO.get(palette, 0.3)
    warm = srgb_to_lin((0.55, 0.36, 0.14)) if palette != "dark_brown" else P["inner"]
    patch = _streaks(rng, NR, NA, 120, 0.25 * rpx)      # patchy pigment density
    hmix = np.clip(het * (0.75 + 0.35 * _streaks(rng, NR, NA, 40, 0.1 * rpx)), 0, 1)[..., None]
    c_inner = P["outer"][None, None] * 0.35 + P["inner"][None, None] * 0.65
    c_inner = c_inner * (1 - hmix) + warm[None, None] * hmix
    inner = inner * (1 - 0.6 * smooth(0.0, 0.35, rho) * (1 - smooth(-0.1, 0.25, rho - rc + 0.15)))
    base = c_inner * inner[..., None] + P["outer"][None, None] * (1 - inner)[..., None]
    base = base * (1 + 0.10 * patch)[..., None]

    # fibres: darker valleys / lighter fibres, strongest in the ciliary zone
    amt = (0.55 + 0.35 * smooth(0.25, 0.6, rho) * (1 - smooth(0.88, 1.0, rho)))[..., None]
    img = base * (1 - amt * 0.55) + base * amt * 0.55 * 2.0 * fib[..., None] ** 1.2
    img = img * (1 - 0.35 * ridge[..., None]) + P["fibre"][None, None] * 0.35 * ridge[..., None]
    img = img * (1 - 0.45 * collar[..., None]) + P["collar"][None, None] * 0.45 * collar[..., None]

    # --- crypts: dark lens-shaped pits, mostly just outside the collarette ---
    crypt = np.zeros((NR, NA))
    for k in range(int(rng.integers(40, 60))):
        near_collar = k < 26
        cr = rng.uniform(0.36, 0.58) if near_collar else rng.uniform(0.62, 0.9)
        ca = rng.uniform(0, NA)
        sr = rng.uniform(0.035, 0.09) if near_collar else rng.uniform(0.02, 0.05)
        sa = rng.uniform(18, 60) if near_collar else rng.uniform(8, 24)
        da = (acol - ca + NA / 2) % NA - NA / 2
        crypt = np.maximum(crypt, rng.uniform(0.55, 1.0) *
                           np.exp(-(da / sa) ** 2 - ((rho - cr) / sr) ** 2))
    crypt = _warp_cols(crypt, warp * 0.5)
    crypt *= np.clip(0.55 + 0.6 * fib, 0, 1)            # fibres cross the pits
    rim = np.clip(_gauss_axis(_gauss_axis(crypt, 3, 1, True), 0.012 * rpx, 0, False) - crypt, 0, 1)
    img *= (1 - 0.78 * crypt)[..., None]
    img = img + P["fibre"][None, None] * 0.25 * rim[..., None]

    # --- contraction furrows: broken concentric dark arcs in the outer ciliary zone ---
    furrow = np.zeros((NR, NA))
    for fr in rng.uniform(0.62, 0.93, int(rng.integers(3, 6))):
        brk = smooth(0.1, 0.7, periodic_noise(ang[0], rng, n=12, fmin=3, fmax=16))[None, :]
        wob = 0.012 * periodic_noise(ang[0], rng, n=10, fmin=4, fmax=30)[None, :]
        furrow += np.exp(-((rho - fr - wob) / 0.014) ** 2) * brk * 0.7
    img *= (1 - 0.30 * np.clip(furrow, 0, 1))[..., None]

    # --- pigment freckles (iris nevi) ---
    for _ in range(int(rng.integers(0, 4))):
        cr, ca = rng.uniform(0.45, 0.9), rng.uniform(0, NA)
        da = (acol - ca + NA / 2) % NA - NA / 2
        s = rng.uniform(0.012, 0.03)
        spot = np.exp(-(da / (s * NA / 2.3)) ** 2 - ((rho - cr) / s) ** 2)
        img = img * (1 - 0.6 * spot[..., None]) + srgb_to_lin((0.20, 0.10, 0.04)) * 0.6 * spot[..., None]

    # --- limbal ring with a fuzzy, irregular outer edge ---
    edge = 1.0 + 0.010 * periodic_noise(ang[0], rng, n=20, fmin=3, fmax=40)[None, :]
    lim = smooth(0.78, 1.02, rho / edge) ** 1.5
    img = img * (1 - 0.80 * lim[..., None]) + P["limbal"][None, None] * 0.80 * lim[..., None]
    # pupillary ruff: dark brown frill right at the pupil margin
    ruff_w = 0.035 + 0.02 * np.clip(f_coarse, -1, 1)
    ruff = np.exp(-(np.clip(rho, 0, None) / ruff_w) ** 2) * (rho > 0)
    img = img * (1 - 0.75 * ruff[..., None]) + srgb_to_lin((0.07, 0.04, 0.025)) * 0.75 * ruff[..., None]

    # relief: fibres/ridges raised, crypts and furrows sunk, collarette raised
    height = 0.45 * fib + 0.35 * ridge + 0.35 * collar - 0.9 * crypt - 0.4 * np.clip(furrow, 0, 1)
    return np.clip(img, 0, 1), height, edge[0]


def _sclera(size, rng, U, Vv, du, dv, r, nasal_sign, sclera_tint):
    """Sclera / bulbar conjunctiva (linear RGB): off-white, warm toward the corners, veins."""
    rs = np.clip((r - R_LIMBUS) / (0.46 - R_LIMBUS), 0, 1)
    scl = srgb_to_lin((0.86, 0.82, 0.77)) * np.ones((size, size, 3))
    # pinker conjunctiva toward the fornix and in the exposed nasal/temporal corners
    t = (0.85 * smooth(0.55, 1.0, rs))[..., None]
    scl = scl * (1 - t) + srgb_to_lin((0.86, 0.66, 0.62)) * t
    # v9: capillary blush reaches into the visible corners (was 0.22 from rs 0.10)
    side = np.clip(np.abs(du) / 0.46, 0, 1) ** 1.3 * smooth(0.04, 0.6, rs)
    scl = scl * (1 - 0.34 * side[..., None]) + srgb_to_lin((0.88, 0.60, 0.56)) * 0.34 * side[..., None]
    nasal = np.clip(nasal_sign * du / 0.46, 0, 1) ** 2 * smooth(0.2, 0.9, rs)
    scl = scl * (1 - 0.25 * nasal[..., None]) + srgb_to_lin((0.85, 0.52, 0.50)) * 0.25 * nasal[..., None]
    # grey-blue band just outside the limbus
    near = np.exp(-((r - R_LIMBUS - 0.014) / 0.014) ** 2)[..., None]
    scl = scl * (1 - 0.14 * near) + srgb_to_lin((0.74, 0.76, 0.80)) * 0.14 * near
    # faint yellowish pinguecula patches at 3 and 9 o'clock
    for sgn in (-1, 1):
        pg = np.exp(-(((du - sgn * 0.27) / 0.05) ** 2 + (dv / 0.03) ** 2))
        scl = scl * (1 - 0.12 * pg[..., None]) + srgb_to_lin((0.90, 0.82, 0.62)) * 0.12 * pg[..., None]
    mott = fbm(size, size, rng, base=6, octaves=5)[..., None]
    scl *= 0.93 + 0.12 * mott
    scl = scl * (1 + sclera_tint * np.array([0.04, 0.02, -0.07]))
    # veins: thick ones from the periphery plus fine capillaries; visible on exposed sides
    veins = np.zeros((size, size))
    for _ in range(int(rng.integers(14, 22))):
        sidea = rng.choice([0.0, math.pi])
        _walk_vein(veins, rng.uniform(0.34, 0.47), sidea + rng.normal(0, 0.28),
                   rng.uniform(4.0, 9.0) * size / 2048, rng.uniform(0.45, 1.0), size, rng, depth=0)
    fine = np.zeros((size, size))
    for _ in range(int(rng.integers(16, 28))):
        sidea = rng.choice([0.0, math.pi])
        _walk_vein(fine, rng.uniform(0.28, 0.42), sidea + rng.normal(0, 0.32),
                   rng.uniform(2.0, 3.5) * size / 2048, rng.uniform(0.3, 0.7), size, rng, depth=1)
    veins = np.clip(blur(veins, 1.1 * size / 2048), 0, 1)
    fine = np.clip(blur(fine, 0.8 * size / 2048), 0, 1)
    horiz = smooth(0.15, 0.6, np.abs(du) / np.maximum(r, 1e-6))   # exposed 3/9 o'clock sides
    veins *= 0.35 + 0.65 * horiz
    fine *= 0.25 + 0.75 * horiz
    vm = (veins * smooth(0.0, 0.18, rs) * 0.8)[..., None]
    fm = (fine * smooth(0.0, 0.12, rs) * 0.7)[..., None]
    # v9: vessels lie under the translucent conjunctiva - a soft pink halo around each one,
    # and a more saturated, slightly lighter red core, so they read as vessels, not cracks
    halo = np.clip(blur(np.maximum(vm[..., 0], fm[..., 0]), 6.0 * size / 2048) * 2.2, 0, 1)[..., None]
    scl = scl * (1 - 0.30 * halo) + srgb_to_lin((0.84, 0.50, 0.46)) * 0.30 * halo
    scl = scl * (1 - 0.62 * vm) + srgb_to_lin((0.66, 0.11, 0.09)) * 0.62 * vm
    scl = scl * (1 - 0.48 * fm) + srgb_to_lin((0.74, 0.20, 0.16)) * 0.48 * fm
    return scl


def eye_tile(size=2048, palette="brown", seed=0, nasal_sign=-1.0, sclera_tint=0.0, return_height=False):
    """v8 eye-interior UDIM tile (sRGB float RGB): polar-synthesised iris + veined sclera.
    With return_height=True also returns an iris relief height map (0..1, 0.5 = flat)."""
    rng = np.random.default_rng(seed + 7919)
    px = (np.arange(size) + 0.5) / size
    U, Vv = np.meshgrid(px, 1.0 - px)
    du, dv = U - 0.5, Vv - 0.5
    r = np.hypot(du, dv)
    ang = np.arctan2(dv, du)
    rho = (r - R_PUPIL) / (R_LIMBUS - R_PUPIL)

    scl = _sclera(size, rng, U, Vv, du, dv, r, nasal_sign, sclera_tint)
    pol, hpol, edge = iris_polar(palette, rng)
    m = r < R_LIMBUS * 1.2
    iris = np.zeros((size, size, 3))
    iris[m] = _polar_lookup(pol, rho[m], ang[m])
    hgt = np.zeros((size, size))
    hgt[m] = _polar_lookup(hpol, rho[m], ang[m])
    # irregular, soft limbus
    edge_px = np.interp((ang % (2 * math.pi)) / (2 * math.pi) * len(edge), np.arange(len(edge) + 1),
                        np.append(edge, edge[0]))
    rl = R_PUPIL + edge_px * (R_LIMBUS - R_PUPIL)
    w_iris = (1 - smooth(rl - 0.006, rl + 0.008, r))[..., None]
    img = scl * (1 - w_iris) + iris * w_iris
    pupil = (1 - smooth(R_PUPIL - 0.003, R_PUPIL + 0.001, r))[..., None]
    img = img * (1 - pupil) + srgb_to_lin((0.010, 0.008, 0.008)) * pupil
    out = lin_to_srgb(img)
    if not return_height:
        return out
    h = hgt * w_iris[..., 0] * (1 - pupil[..., 0])
    h = 0.5 + 0.5 * h / max(np.abs(h).max(), 1e-6)
    return out, h


def eye_tile_v1(size=2048, palette="brown", seed=0, nasal_sign=-1.0, sclera_tint=0.0):
    """v7 eye-interior tile (kept for comparison).

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
