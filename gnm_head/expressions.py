"""
expressions.py - whole-face expression shape deltas for one GNM identity (numpy only).

The organiser's reference clips (natural CG expressions) show that a convincing eye
expression is never the eyes alone: a real smile raises the cheeks and pushes the lower
lids up (Duchenne smile, FACS AU6 + AU12), and squinting narrows the eye from below.

Both keys are fitted in GNM's own learned expression spaces (PCA of real scans), so the
surrounding skin moves the way faces actually move; we only specify a sparse target:

  squint(ec, s)   AU6/AU7 cheek raiser + lid tightener for one eye: the lower lid rotates
                  up over the globe (same lid model as the gaze-follow keys) and the
                  infraorbital / zygomatic skin lifts and bulges forward.
  smile(ec)       AU12 lip-corner puller (both sides): corners move up, out and back,
                  cheeks lift; fitted in the lower-face space.

Deltas are in the GNM frame (metres; +Y up, +Z forward, +X subject's left).
"""
import math

import numpy as np

from gnm_model import smoothstep


def _fit(h, comps, region, target, wt, reg):
    B = h.expression_basis[comps].astype(np.float64)                    # (K, V, 3)
    sw = np.sqrt(wt[region])
    A = (B[:, region, :] * sw[None, :, None]).reshape(len(comps), -1).T
    y = (target[region] * sw[:, None]).ravel()
    lam = reg * (A ** 2).sum() / len(comps)
    coef = np.linalg.solve(A.T @ A + lam * np.eye(len(comps)), A.T @ y)
    full = np.zeros(len(h.expression_names))
    full[comps] = coef
    return h.expression_delta(full), coef


def squint(ec, s, lid_deg=9.0, cheek_lift=0.0028, cheek_fwd=0.0010, reg=0.03):
    """Cheek raise + lower-lid tighten for eye s ('L' / 'R')."""
    h = ec.h
    side = "left" if s == "L" else "right"
    c = ec.side[s]["c"]
    lid = ec.lid_pitch(s, "lower", -math.radians(lid_deg))              # lower lid up
    g = h.groups
    w = np.clip(g[side + "_infraorbital_region"] + g[side + "_zygomatic_region"]
                + 0.5 * g[side + "_cheek_region"], 0, 1) * ec.skin
    # strongest just below the eye, fading towards the jaw
    below = np.clip((c[1] - h.bind[:, 1]) / 0.035, 0, 1)
    w = w * (1 - smoothstep(0.6, 1.0, below))
    target = np.zeros((h.V, 3))
    target[:, 1] = cheek_lift * w
    target[:, 2] = cheek_fwd * w
    region = ec.skin & (np.linalg.norm(h.bind - c, axis=1) < 0.06)
    keep = np.maximum(ec.side[s]["w_up"], ec.side[s]["w_lo"]) > 0     # lids: driven by the lid model
    wt = np.where(w > 0.05, 1.0, 0.3)
    wt[keep] = 3.0
    # eye-region space only: the lower-face space would also part the lips
    comps = h.expression_index(side + "_eye_region_")
    cheek, _ = _fit(h, comps, region, target, wt, reg)
    cheek[keep] *= 0.0
    lips = (h.groups["upper_lip"] > 1e-4) | (h.groups["lower_lip"] > 1e-4) | (h.groups["mouth_sock"] > 1e-4)
    cheek[lips] = 0.0
    # blend the fitted cheek into the lid falloff so the lower lid rides up with the cheek
    delta = lid + cheek
    delta[~ec.skin] = 0.0
    err = np.abs(cheek[w > 0.5] - target[w > 0.5]).mean() * 1e3 if np.any(w > 0.5) else 0.0
    return delta, err


def lip_corners(h):
    lips = np.nonzero((h.groups["upper_lip"] > 1e-4) | (h.groups["lower_lip"] > 1e-4))[0]
    P = h.bind[lips]
    return {"L": h.bind[lips[np.argmax(P[:, 0])]], "R": h.bind[lips[np.argmin(P[:, 0])]]}


def smile(ec, lift=0.0055, out=0.0035, back=0.0022, cheek_lift=0.0022, radius=0.024, reg=0.04):
    """Symmetric AU12 smile (lip corners up / out / back, cheeks up)."""
    h = ec.h
    P = h.bind
    corners = lip_corners(h)
    target = np.zeros((h.V, 3))
    wsum = np.zeros(h.V)
    for s, sx in (("L", 1.0), ("R", -1.0)):
        f = 1 - smoothstep(0.0, radius, np.linalg.norm(P - corners[s], axis=1))
        f *= (np.sign(P[:, 0]) == sx) | (np.abs(P[:, 0]) < 0.004)
        target += f[:, None] * np.array([sx * out, lift, -back])
        wsum += f
    g = h.groups
    cheeks = np.clip(g["left_cheek_region"] + g["right_cheek_region"]
                     + 0.6 * (g["left_zygomatic_region"] + g["right_zygomatic_region"]), 0, 1)
    target[:, 1] += cheek_lift * cheeks
    wsum += cheeks
    mouth_c = 0.5 * (corners["L"] + corners["R"])
    eyes = h.mask("eyes")
    region = ~eyes & (np.linalg.norm(P - mouth_c, axis=1) < 0.08)
    wt = np.where(wsum > 0.05, 1.0, 0.15)
    wt[(g["upper_lip"] > 1e-4) | (g["lower_lip"] > 1e-4)] = np.maximum(
        wt[(g["upper_lip"] > 1e-4) | (g["lower_lip"] > 1e-4)], 0.6)
    comps = h.expression_index("lower_face_region_")
    delta, _ = _fit(h, comps, region, target, wt, reg)
    delta[eyes] = 0.0
    sel = wsum > 0.5
    err = np.abs(delta[sel] - target[sel]).mean() * 1e3
    return delta, err
