"""
hair.py - procedural 3D hair strands for a GNM head (numpy only): a short buzz cut on the
scalp, eyebrows grown where the texture's brows are, and fine vellus "peach fuzz".

Each function returns (roots, normals, uvs, strands) where strands is (N, K, 3) points from
root to tip in the GNM frame (metres; +Y up, +Z forward, +X subject's left).  Roots lie on
the head surface (sampled on its triangles, area-weighted), and carry the triangle's UV so the
strands can be attached to the deforming surface in Blender (Deform Curves on Surface).
"""
import numpy as np

from gnm_model import smoothstep


# --------------------------------------------------------------------------- surface sampling
def _tris(head):
    T = np.asarray(head.z["triangles"]).astype(np.int64)
    TUV = np.asarray(head.z["triangle_uvs"]).astype(np.float64)
    return T, TUV


def sample_surface(head, vweight, n, seed=0):
    """n points on triangles, density ~ area * mean vertex weight.  Returns pos, normal, uv, tri, bary."""
    rng = np.random.default_rng(seed)
    T, TUV = _tris(head)
    P = head.bind
    w = vweight[T].mean(axis=1)
    keep = w > 1e-3
    T, TUV, w = T[keep], TUV[keep], w[keep]
    a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    cr = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    p = area * w
    idx = rng.choice(len(T), size=n, p=p / p.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    bary = np.stack([1 - s, s * (1 - r2), s * r2], -1)
    tri = T[idx]
    pos = (P[tri] * bary[:, :, None]).sum(1)
    nrm = cr[idx] / np.linalg.norm(cr[idx], axis=1, keepdims=True)
    # make normals point outward (away from the head centre)
    cen = np.array([0.0, 0.26, -0.01])
    flip = ((pos - cen) * nrm).sum(1) < 0
    nrm[flip] *= -1
    uv = (TUV[idx] * bary[:, :, None]).sum(1)
    # per-vertex weight interpolated at the sample (for acceptance tests)
    wv = (vweight[tri] * bary).sum(1)
    return pos, nrm, uv, wv


def _tangent(n, d):
    t = d - (d * n).sum(1, keepdims=True) * n
    return t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)


def _grow(pos, nrm, direc, length, lift, curl, K, rng, jitter=0.15):
    """Strands that leave the skin at `lift` (0 = flat along direc, 1 = along the normal) and
    bend back towards the surface direction; slight random wave."""
    n = len(pos)
    t = _tangent(nrm, direc + jitter * rng.normal(size=direc.shape))
    out = np.zeros((n, K, 3))
    out[:, 0] = pos
    for k in range(1, K):
        u = k / (K - 1)
        l = lift * (1 - u) ** 1.5                      # rises at the root, lies down at the tip
        d = t * (1 - l) + nrm * l
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        bend = curl * u * rng.normal(size=(n, 3)) * 0.3
        out[:, k] = out[:, k - 1] + (d + bend) * (length / (K - 1))[:, None]
    return out


# --------------------------------------------------------------------------- regions
# Hairline transition below / above the hairline height (m).  Was (0.002, 0.005): a crisp 7 mm
# edge.  17 Nov: ~16 mm, so buzz_cut's wv**1.5 thinning leaves sparse hairs ahead of the line.
HAIRLINE = (0.006, 0.010)


def scalp_weight(head):
    """1 on the hair-bearing scalp (buzz cut), soft hairline at forehead, temples, around the
    ears and at the nape; 0 on the face, ears and neck."""
    P = head.bind
    g = head.groups
    ext = head.mask("skin_exterior")
    brows = (g["left_brow_region"] > 0.3) | (g["right_brow_region"] > 0.3)
    brow_top = np.percentile(P[brows, 1], 95)
    ears = g["ears"] > 1e-4
    ear_y0, ear_y1 = np.percentile(P[ears, 1], [5, 95])
    ear_z1 = np.percentile(P[ears, 2], 90)              # front of the ears
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    # hairline height as a function of azimuth around the vertical axis through the ears:
    # forehead (0 deg) -> mild temple recession (~35 deg) -> sideburn down to mid-ear in front
    # of the ear (~75 deg) -> behind the ear and the nape (>= 100 deg)
    zc = np.mean(P[ears, 2])
    az = np.degrees(np.arctan2(np.abs(x), z - zc))
    ear_mid = 0.5 * (ear_y0 + ear_y1)
    knots_az = [0, 20, 36, 52, 64, 72, 80, 86, 96, 112, 180]
    knots_y = [brow_top + 0.040, brow_top + 0.039, brow_top + 0.033, brow_top + 0.022,
               brow_top + 0.008, ear_y1 - 0.002, ear_mid - 0.002, ear_mid + 0.004,
               ear_y1 + 0.002, ear_y0 + 0.006, ear_y0 - 0.004]
    hl = np.interp(az, knots_az, knots_y)
    w = smoothstep(hl - HAIRLINE[0], hl + HAIRLINE[1], y)
    # keep a gap around the ears
    if ears.any():
        from numpy.linalg import norm
        E = P[ears][::3]
        d = np.min(norm(P[:, None, :] - E[None, :, :], axis=2), axis=1)
        w *= smoothstep(0.0025, 0.0055, d)
    w *= ext & ~ears
    return np.clip(w, 0, 1)


def buzz_cut(head, n=90000, length=0.0045, seed=1):
    """Short (~4-5 mm) hair lying back from the forehead and down at the sides/back, growing
    out of a crown whorl."""
    rng = np.random.default_rng(seed)
    w = scalp_weight(head)
    pos, nrm, uv, wv = sample_surface(head, w, n * 2, seed)
    keep = rng.random(len(pos)) < np.clip(wv, 0, 1) ** 1.5      # thinner along the hairline
    pos, nrm, uv, wv = pos[keep][:n], nrm[keep][:n], uv[keep][:n], wv[keep][:n]
    crown = np.array([0.006, head.bind[:, 1].max() - 0.012, -0.055])
    direc = pos - crown                                   # away from the whorl...
    direc[:, 1] -= 0.04                                   # ...and down with gravity / combing
    L = length * (0.75 + 0.5 * rng.random(len(pos))) * (0.5 + 0.5 * wv)
    strands = _grow(pos, nrm, direc, L, lift=0.35, curl=0.25, K=5, rng=rng)
    return pos, nrm, uv, strands


def eyebrows(head, brow_alpha=None, n_per=750, seed=2):
    """Brow hairs where the brows are: weight from GNM brow regions (or the texture's brow
    alpha sampled at each root UV).  Medial hairs point up, the body sweeps laterally."""
    rng = np.random.default_rng(seed)
    P = head.bind
    g = head.groups
    out = []
    for s, side in (("L", "left"), ("R", "right")):
        w = np.clip(g[side + "_brow_region"], 0, 1) * head.mask("skin_exterior")
        pos, nrm, uv, wv = sample_surface(head, w, n_per * 10, seed + (s == "R"))
        if brow_alpha is not None:
            h_, w_ = brow_alpha.shape[:2]
            u = np.clip((uv[:, 0] % 1.0) * (w_ - 1), 0, w_ - 1).astype(int)
            v = np.clip((1 - uv[:, 1] % 1.0) * (h_ - 1), 0, h_ - 1).astype(int)
            a = brow_alpha[v, u].astype(float)
            a = a / max(a.max(), 1e-6)
        else:
            a = np.clip(wv, 0, 1)
        keep = rng.random(len(pos)) < a ** 1.2
        pos, nrm, uv = pos[keep][:n_per], nrm[keep][:n_per], uv[keep][:n_per]
        xs = np.abs(pos[:, 0])
        x0, x1 = np.percentile(xs, [3, 97])
        t = np.clip((xs - x0) / (x1 - x0), 0, 1)           # 0 medial (brow head) .. 1 lateral (tail)
        lat = np.sign(pos[:, 0])[:, None] * np.array([1.0, 0, 0])
        up = np.array([0, 1.0, 0])[None, :]
        # head: steeply up; body: lateral and slightly up; tail: lateral and down
        ang_up = np.where(t < 0.18, 1.0, np.where(t < 0.75, 0.25, -0.25))[:, None]
        direc = lat * (1 - 0.6 * (t < 0.18))[:, None] + up * ang_up
        L = (0.0065 + 0.004 * rng.random(len(pos))) * (1 - 0.35 * t)
        out.append((pos, nrm, uv, _grow(pos, nrm, direc, L, lift=0.25, curl=0.35, K=6, rng=rng, jitter=0.25)))
    return tuple(np.concatenate([o[i] for o in out]) for i in range(4))


def peach_fuzz(head, n=60000, seed=3):
    """Fine vellus hair (1-2 mm, very thin, light) on the cheeks, forehead, nose, upper lip and
    jaw - softens the silhouette in rim light."""
    rng = np.random.default_rng(seed)
    P = head.bind
    g = head.groups
    w = (head.mask("skin_exterior") & ~(g["ears"] > 1e-4)).astype(float)
    w *= 1 - scalp_weight(head)
    w *= 1 - np.clip(g["upper_lip"] + g["lower_lip"], 0, 1)
    # not on the lids / in the eye openings
    for s in "LR":
        w *= smoothstep(0.010, 0.016, np.linalg.norm(P - head.eye_center(s), axis=1))
    pos, nrm, uv, wv = sample_surface(head, w, n, seed)
    direc = np.zeros_like(pos)
    direc[:, 1] = -1.0
    direc[:, 0] = np.sign(pos[:, 0]) * 0.6                # down and outward
    L = 0.0011 + 0.0008 * rng.random(n)
    return pos, nrm, uv, _grow(pos, nrm, direc, L, lift=0.35, curl=0.2, K=3, rng=rng)
