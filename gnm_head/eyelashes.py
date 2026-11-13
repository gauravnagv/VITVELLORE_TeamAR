"""
eyelashes.py - procedural eyelash geometry rooted on GNM's lid margins (numpy only, GNM frame).

Lashes are thin tapered ribbons: two staggered rows on the upper lid (curled upward,
longest mid-lid, shorter toward the canthi, fanning laterally) and one sparser row on the
lower lid.  `follow(...)` converts any eyelid shape delta of the head into the matching
lash delta (each lash rotates rigidly with its root about the eyeball centre and follows
the root's push-out over the cornea), so lash shape keys can mirror the lid shape keys.
"""
import math

import numpy as np


def _margin_curve(P, idx, canthi):
    """Ordered polyline medial->lateral through margin vertices + both canthi."""
    pts = np.vstack([P[canthi[0]], P[idx], P[canthi[1]]])
    ids = np.r_[canthi[0], idx, canthi[1]]
    d_med = np.abs(pts[:, 0] - P[canthi[0], 0])
    o = np.argsort(d_med)
    pts, ids = pts[o], ids[o]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    t = np.r_[0, np.cumsum(seg)] / seg.sum()
    return pts, ids, t


def _interp(pts, t, s):
    return np.stack([np.interp(s, t, pts[:, k]) for k in range(3)], -1)


def generate(head, ctrl, side, seed=0, rows_upper=(78, 46), n_lower=46,
             len_upper=0.0095, len_lower=0.0055, width=0.00011):
    """Returns verts (N,3) GNM frame, quads (M,4), roots list of (lid, s) per lash, vert->lash map."""
    rng = np.random.default_rng(seed)
    P = head.bind
    sd = ctrl.side[side]
    c = sd["c"]
    lat = np.array([1.0 if side == "L" else -1.0, 0, 0])
    fwd, up = np.array([0, 0, 1.0]), np.array([0, 1.0, 0])
    verts, quads, lash_of_vert, roots = [], [], [], []
    nseg = 7
    for lid, rows, L0, upv in (("upper", rows_upper, len_upper, up), ("lower", (n_lower,), len_lower, -up)):
        pts, ids, t = _margin_curve(P, sd[lid], sd["canthi"])
        for r_i, n in enumerate(rows):
            s_all = np.sort(np.clip(np.linspace(0.04, 0.96, n) + rng.normal(0, 0.006, n), 0.02, 0.98))
            for s in s_all:
                root = _interp(pts, t, np.array([s]))[0]
                nrad = (root - c) / np.linalg.norm(root - c)
                # sit on the anterior lid margin, rows staggered slightly outward
                root = root + nrad * (0.00025 + 0.00018 * r_i) + fwd * 0.0001
                prof = math.sin(math.pi * s) ** 0.6
                L = L0 * (0.35 + 0.65 * prof) * rng.uniform(0.85, 1.12) * (1 - 0.12 * r_i)
                fan = (s - 0.45) * (0.9 if lid == "upper" else 0.6)
                # lashes leave the lid mostly forward (slightly downward), then curl away
                # from the eye; seen from the front they are strongly foreshortened
                d0 = 0.80 * fwd + 0.20 * nrad + fan * lat + (-0.10 if lid == "upper" else -0.15) * up
                d0 += rng.normal(0, 0.05, 3)
                d0 /= np.linalg.norm(d0)
                curl = (0.34 if lid == "upper" else 0.16) * rng.uniform(0.8, 1.2)
                ss = np.linspace(0, 1, nseg + 1)
                cen = root + L * (ss[:, None] * d0 + (ss[:, None] ** 2) * curl * upv)
                tang = np.gradient(cen, axis=0)
                side_v = np.cross(tang, fwd)
                side_v /= np.linalg.norm(side_v, axis=1, keepdims=True) + 1e-12
                w = width * (1 - ss) ** 0.8 + 0.000012
                base = len(verts)
                for k in range(nseg + 1):
                    verts.append(cen[k] - side_v[k] * w[k] / 2)
                    verts.append(cen[k] + side_v[k] * w[k] / 2)
                    lash_of_vert += [len(roots), len(roots)]
                for k in range(nseg):
                    a = base + 2 * k
                    quads.append([a, a + 1, a + 3, a + 2])
                # remember where on the margin this lash grows (for following lid motion)
                roots.append((lid, float(s), root))
    return np.array(verts), np.array(quads, np.int64), roots, np.array(lash_of_vert)


def follow(head, ctrl, side, verts, roots, lash_of_vert, delta):
    """Lash vertex deltas that track a head lid-delta (GNM frame)."""
    P = head.bind
    sd = ctrl.side[side]
    c = sd["c"]
    out = np.zeros_like(verts)
    cache = {}
    for lid in ("upper", "lower"):
        pts, ids, t = _margin_curve(P, sd[lid], sd["canthi"])
        cache[lid] = (pts, ids, t, pts + delta[ids])
    for li, (lid, s, root) in enumerate(roots):
        pts, ids, t, pts_new = cache[lid]
        r0 = _interp(pts, t, np.array([s]))[0]
        r1 = _interp(pts_new, t, np.array([s]))[0]
        # rotation about the eye's x axis that carries r0 to r1's elevation
        a0 = math.atan2(r0[1] - c[1], r0[2] - c[2])
        a1 = math.atan2(r1[1] - c[1], r1[2] - c[2])
        ang = a0 - a1                       # + = down (rotation about +x)
        ca, sa = math.cos(ang), math.sin(ang)
        m = lash_of_vert == li
        d = verts[m] - c
        y = d[:, 1] * ca - d[:, 2] * sa
        z = d[:, 1] * sa + d[:, 2] * ca
        nv = np.stack([d[:, 0], y, z], -1) + c
        # translate so the root lands where the lid margin went (cornea push-out)
        dr = root - r0
        root_new = c + np.array([dr[0] + r0[0] - c[0],
                                 (r0[1] - c[1] + dr[1]) * ca - (r0[2] - c[2] + dr[2]) * sa,
                                 (r0[1] - c[1] + dr[1]) * sa + (r0[2] - c[2] + dr[2]) * ca])
        shift = (r1 + (root - r0)) - root_new
        out[m] = nv + shift - verts[m]
    return out
