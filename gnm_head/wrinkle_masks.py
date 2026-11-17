"""
wrinkle_masks.py - where expression wrinkles may appear on a GNM head (numpy only).

Returns per-vertex weights (0..1) for three classic dynamic wrinkle zones:
  forehead   horizontal lines when the brows lift (frontalis): a band from ~1 cm above the
             brows up to mid-forehead
  glabella   one short vertical furrow either side of the midline when the brows knit
             (corrugator), from the brow heads up ~1.5 cm
  crow       3-5 short lines fanning from each outer eye corner when the cheeks lift
plus the reference points the shader needs (lateral canthi).  GNM frame: metres, +Y up,
+Z forward, +X subject's left.
"""
import numpy as np

from gnm_model import smoothstep


def masks(head):
    P = head.bind
    g = head.groups
    ext = head.mask("skin_exterior")
    front = P[:, 2] > np.percentile(P[ext, 2], 55)               # facing the camera
    out = {}

    brows = (g["left_brow_region"] > 0.3) | (g["right_brow_region"] > 0.3)
    brow_top = np.percentile(P[brows, 1], 90)
    brow_mid = np.percentile(P[brows, 1], 50)

    # forehead band: from ~0.8 cm above the brows to ~5 cm, fading at the temples
    y = P[:, 1] - brow_top
    w = smoothstep(0.000, 0.007, y) * (1 - smoothstep(0.032, 0.048, y))
    w *= 1 - smoothstep(0.028, 0.048, np.abs(P[:, 0]))
    out["forehead"] = np.clip(w * ext * front, 0, 1)

    # glabella: between the brow heads, short
    y = P[:, 1] - brow_mid
    w = smoothstep(-0.004, 0.001, y) * (1 - smoothstep(0.010, 0.017, y))
    w *= 1 - smoothstep(0.007, 0.011, np.abs(P[:, 0]))
    out["glabella"] = np.clip(w * ext * front, 0, 1)

    # crow's feet: lateral to each outer eye corner, a fan of +/- ~45 deg around horizontal
    crow = np.zeros(head.V)
    canthi = {}
    for s in "LR":
        up, lo, (med, lat) = head.lid_margin(s)
        c = P[lat]
        canthi[s] = c
        d = np.linalg.norm(P - c, axis=1)
        dx = (P[:, 0] - c[0]) * np.sign(c[0])                       # outward from the corner
        ang = np.arctan2(P[:, 1] - c[1], np.maximum(dx, 1e-6))
        margin = np.concatenate([up, lo])
        dm = np.min(np.linalg.norm(P[:, None, :] - P[margin][None, :, :], axis=2), axis=1)
        w = smoothstep(0.003, 0.006, d) * (1 - smoothstep(0.011, 0.016, d)) * (dx > 0)
        w *= 1 - smoothstep(0.55, 0.85, np.abs(ang))
        w *= smoothstep(0.0015, 0.004, dm)                          # keep off the lid margin
        crow = np.maximum(crow, w)
    out["crow"] = np.clip(crow * ext, 0, 1)
    out["canthi"] = canthi
    return out
