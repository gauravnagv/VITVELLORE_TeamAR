"""
gnm_model.py - minimal, dependency-free (numpy only) access to Google's GNM v3 head.

GNM (Generative aNthropometric Model, Apache-2.0, https://github.com/google/GNM) is a
linear 3D morphable head: template + identity basis + expression basis, posed with
linear blend skinning over 4 joints (neck, head, left_eye, right_eye).  The official
package pulls in tensorflow/jax/trimesh; Blender only needs the maths below, which
mirrors gnm/shape/gnm_common.py exactly.

Also provides the pieces our Blender scene needs:
  * UDIM texture layout (identical to gnm/shape/gnm_udim.py: 1001 skin, 1002/1003 left
    eye interior/cornea, 1004/1005 right, 1006 teeth+gums, 1007 tongue),
  * eyelid margin detection and anatomically-motivated eyelid / brow controls.

GNM frame: metres, +Y up, +Z forward (out of the face), +X = subject's left.
"""
import math

import numpy as np

UDIM_REGIONS = [  # (tile, vertex-group expression) - same table as gnm_udim.UDIM_TILE_REGIONS
    (1001, ("skin",)),
    (1002, ("left_eye", "&eye_interiors")),
    (1003, ("left_eye", "&eye_exteriors")),
    (1004, ("right_eye", "&eye_interiors")),
    (1005, ("right_eye", "&eye_exteriors")),
    (1006, ("upper_teeth_and_gums", "lower_teeth_and_gums")),
    (1007, ("tongue",)),
]
NONZERO = 1e-4
SIDES = {"L": "left", "R": "right"}


def rodrigues(aa):
    aa = np.asarray(aa, dtype=np.float64)
    th = float(np.linalg.norm(aa))
    if th < 1e-12:
        return np.eye(3)
    k = aa / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


class GNMHead:
    def __init__(self, npz_path, identity=None):
        z = np.load(npz_path)
        self.z = z
        f = np.float64
        self.template = z["template_vertex_positions"].astype(f)
        self.V = len(self.template)
        self.identity_basis = z["vertex_identity_basis"]          # (253, V, 3) float32
        self.expression_basis = z["expression_basis"]             # (383, V, 3) float32
        self.expression_names = [str(n) for n in z["expression_names"]]
        self.joint_names = [str(n) for n in z["joint_names"]]
        self.joint_parents = z["joint_parent_indices"].tolist()
        self.skinning_weights = z["skinning_weights"].astype(f)  # (J, V)
        self.quads = z["quads"].astype(np.int64)
        self.quad_uvs = z["quad_uvs"].astype(f)
        self.group_names = [str(n) for n in z["vertex_group_names"]]
        self.groups = {n: z["vertex_groups"][i].astype(f) for i, n in enumerate(self.group_names)}
        self.set_identity(identity)

    # ------------------------------------------------------------------ shape
    def set_identity(self, identity=None):
        z = self.z
        self.identity = np.zeros(self.identity_basis.shape[0]) if identity is None else np.asarray(identity, float)
        self.bind = self.template + np.tensordot(self.identity, self.identity_basis, axes=1).astype(np.float64)
        self.joints = (z["template_joint_positions"].astype(np.float64)
                       + np.tensordot(self.identity, z["joint_identity_basis"], axes=1))
        return self

    def expression_delta(self, coeffs):
        coeffs = np.asarray(coeffs, dtype=np.float64)
        nz = np.nonzero(coeffs)[0]
        if len(nz) == 0:
            return np.zeros((self.V, 3))
        return np.tensordot(coeffs[nz], self.expression_basis[nz].astype(np.float64), axes=1)

    def expression_index(self, prefix):
        return [i for i, n in enumerate(self.expression_names) if n.startswith(prefix)]

    # ------------------------------------------------------------- groups/UV
    def mask(self, *names):
        m = np.zeros(self.V, bool)
        for name in names:
            op = "|"
            if name[0] in "|&-":
                op, name = name[0], name[1:]
            inv = name.startswith("~")
            g = self.groups[name.lstrip("~")] > NONZERO
            g = ~g if inv else g
            m = m | g if op == "|" else (m & g if op == "&" else m & ~g)
        return m

    def quad_tiles(self):
        tiles = np.zeros(len(self.quads), np.int32)
        for tile, expr in UDIM_REGIONS:
            q = np.all(self.mask(*expr)[self.quads], axis=1)
            if np.any(tiles[q]):
                raise ValueError("UDIM regions overlap at tile %d" % tile)
            tiles[q] = tile
        if np.any(tiles == 0):
            raise ValueError("%d quads without a UDIM tile" % int(np.sum(tiles == 0)))
        return tiles

    def quad_uvs_udim(self):
        t = self.quad_tiles() - 1001
        off = np.stack([t % 10, t // 10], -1).astype(np.float64)
        return self.quad_uvs + off[:, None, :]

    # ------------------------------------------------------------ eye helpers
    def eye_center(self, s):
        return self.joints[self.joint_names.index(SIDES[s] + "_eye")]

    def lid_margin(self, s):
        """Lid margin of eye s -> (upper idx, lower idx, canthi idx).

        GNM's skin is closed: it folds inward at the lid margin into an eye-socket
        lining.  The margin is the ring of `skin_exterior` vertices that share an
        edge with non-exterior (socket) skin around this eye."""
        if not hasattr(self, "_margin_cache"):
            self._margin_cache = {}
        if s in self._margin_cache:
            return self._margin_cache[s]
        skin, ext = self.mask("skin"), self.mask("skin_exterior")
        q = self.quads[np.all(skin[self.quads], axis=1)]
        e = np.unique(np.sort(np.stack([q, np.roll(q, -1, axis=1)], -1).reshape(-1, 2), axis=1), axis=0)
        cross = e[ext[e[:, 0]] != ext[e[:, 1]]]
        bnd = np.unique(cross.ravel())
        c = self.eye_center(s)
        bnd = bnd[ext[bnd] & (np.linalg.norm(self.bind[bnd] - c, axis=1) < 0.022)]
        p = self.bind[bnd] - c
        wx = np.abs(self.bind[bnd, 0])
        med = bnd[np.argmin(wx)]   # medial canthus: nearest the facial midline (nose)
        lat = bnd[np.argmax(wx)]   # lateral canthus
        a, b = self.bind[med] - c, self.bind[lat] - c
        # canthal line height at each margin vertex's x
        t = (p[:, 0] - a[0]) / (b[0] - a[0])
        yline = a[1] + t * (b[1] - a[1])
        upper = bnd[(p[:, 1] > yline) & (bnd != med) & (bnd != lat)]
        lower = bnd[(p[:, 1] <= yline) & (bnd != med) & (bnd != lat)]
        self._margin_cache[s] = (upper, lower, np.array([med, lat]))
        return self._margin_cache[s]

    def cornea_profile(self, s, nbins=90):
        """Max radius of the eye's outer (corneal) shell vs. angle from the eye axis."""
        c = self.eye_center(s)
        ex = self.mask(SIDES[s] + "_eye", "&eye_exteriors")
        d = self.bind[ex] - c
        r = np.linalg.norm(d, axis=1)
        ang = np.arccos(np.clip(d[:, 2] / r, -1, 1))
        edges = np.linspace(0, math.pi, nbins + 1)
        prof = np.zeros(nbins)
        idx = np.clip(np.digitize(ang, edges) - 1, 0, nbins - 1)
        for i in range(nbins):
            sel = r[idx == i]
            prof[i] = sel.max() if len(sel) else 0
        prof = np.maximum.accumulate(prof[::-1])[::-1]  # conservative: non-increasing outward
        return edges, prof


class EyeControls:
    """Anatomically-motivated eyelid & brow shape deltas for one GNM identity.

    Eyelids: lid skin slides over the globe, so lid vertices rotate about the eyeball
    centre (about the horizontal axis), weighted by a falloff from the lid margin, and
    are kept outside the cornea.  Brows: fitted in GNM's own learned eye-region
    expression space (100 components per eye), so forehead skin deforms as in scans.
    """

    def __init__(self, head, upper_reach=0.010, lower_reach=0.007):
        self.h = head
        self.upper_reach, self.lower_reach = upper_reach, lower_reach
        self.skin = head.mask("skin")
        self.side = {}
        for s in "LR":
            up, lo, canthi = head.lid_margin(s)
            c = head.eye_center(s)
            self.side[s] = dict(c=c, upper=up, lower=lo, canthi=canthi,
                                cornea=head.cornea_profile(s))
            self.side[s]["w_up"], self.side[s]["near_up"] = self._lid_weights(s, up, upper_reach)
            self.side[s]["w_lo"], self.side[s]["near_lo"] = self._lid_weights(s, lo, lower_reach)

    def _lid_weights(self, s, margin, reach):
        """Falloff weight (1 at margin -> 0 at `reach`) for skin on the margin's side."""
        h, sd = self.h, None
        c = h.eye_center(s)
        P = h.bind
        cand = np.where(self.skin & (np.linalg.norm(P - c, axis=1) < 0.03))[0]
        d = np.linalg.norm(P[cand][:, None, :] - P[margin][None, :, :], axis=2)
        nearest = margin[np.argmin(d, axis=1)]
        dist = d.min(axis=1)
        # only vertices on the same side (above / below) of the canthal line as the margin
        med, lat = h.lid_margin(s)[2]
        a, b = P[med] - c, P[lat] - c
        p = P[cand] - c
        t = (p[:, 0] - a[0]) / (b[0] - a[0])
        yline = a[1] + t * (b[1] - a[1])
        upper_side = np.mean(P[margin][:, 1] - c[1]) > np.mean([a[1], b[1]])
        same = (p[:, 1] > yline) if upper_side else (p[:, 1] <= yline)
        # taper to zero at the canthi so the corners stay pinned
        taper = np.clip(4 * np.clip(t, 0, 1) * (1 - np.clip(t, 0, 1)), 0, 1) ** 0.5
        w = np.zeros(h.V)
        w[cand] = (1 - smoothstep(0.0, reach, dist)) * same * taper
        near = np.full(h.V, -1)
        near[cand] = nearest
        return w, near

    def _phi(self, s, idx):
        d = self.h.bind[idx] - self.side[s]["c"]
        return np.arctan2(d[:, 1], d[:, 2])  # elevation about the eye's x axis

    def _rotate_lids(self, s, ang):
        """Rotate vertices about the eye centre's x axis by per-vertex angle ang (rad, + = down)."""
        sd = self.side[s]
        c = sd["c"]
        P = self.h.bind
        out = np.zeros_like(P)
        idx = np.nonzero(np.abs(ang) > 1e-9)[0]
        if len(idx) == 0:
            return out
        d = P[idx] - c
        ca, sa = np.cos(ang[idx]), np.sin(ang[idx])
        y = d[:, 1] * ca - d[:, 2] * sa
        zz = d[:, 1] * sa + d[:, 2] * ca
        nd = np.stack([d[:, 0], y, zz], -1)
        # keep lids outside the cornea (+0.4 mm tear-film / lid thickness clearance)
        edges, prof = sd["cornea"]
        r = np.linalg.norm(nd, axis=1)
        th = np.arccos(np.clip(nd[:, 2] / r, -1, 1))
        rmin = prof[np.clip(np.digitize(th, edges) - 1, 0, len(prof) - 1)] + 0.0004
        push = np.where(r < rmin, rmin / r, 1.0)
        out[idx] = nd * push[:, None] - d
        return out

    # -- shape deltas (GNM frame, metres) --------------------------------------
    def lid_pitch(self, s, lid, angle_rad):
        """Upper or lower lid rotated by angle (+ = down) with margin falloff."""
        w = self.side[s]["w_up" if lid == "upper" else "w_lo"]
        return self._rotate_lids(s, w * angle_rad)

    def _closure(self, s, gap, upper_share):
        """Lid delta for per-upper-margin-vertex closure angles `gap` (rad)."""
        sd = self.side[s]
        P = self.h.bind
        up = sd["upper"]
        gap_of = dict(zip(up.tolist(), gap.tolist()))
        ux = P[up, 0]
        ou = np.argsort(ux)
        ang = np.zeros(self.h.V)
        for key, share, sign in (("up", upper_share, 1.0), ("lo", 1 - upper_share, -1.0)):
            w, near = sd["w_" + key], sd["near_" + key]
            idx = np.nonzero(w > 0)[0]
            if key == "up":
                g = np.array([gap_of.get(int(n), 0.0) for n in near[idx]])
            else:  # lower lid: closure angle of the upper margin above it (interp in x)
                g = np.interp(P[near[idx], 0], ux[ou], gap[ou])
            ang[idx] += sign * share * w[idx] * g
        return self._rotate_lids(s, ang)

    def margin_gap(self, s, delta):
        """Vertical gap (m) between upper and lower lid margins at each upper-margin x."""
        sd = self.side[s]
        P = self.h.bind + delta
        up, lo = sd["upper"], sd["lower"]
        o = np.argsort(P[lo, 0])
        return P[up, 1] - np.interp(P[up, 0], P[lo, 0][o], P[lo, 1][o])

    def blink(self, s, upper_share=0.85, overlap_m=0.0003, iters=6):
        """Fully closed lids: the upper margin meets the lower lid along its whole length.

        The upper lid does most of the travel (levator relaxes); the lower lid rises a
        little (~15 %), as in real blinks.  Closure angles are refined iteratively because
        the lid is pushed out over the corneal bulge as it descends."""
        sd = self.side[s]
        P = self.h.bind
        up, lo = sd["upper"], sd["lower"]
        o = np.argsort(P[lo, 0])
        lo_phi = np.interp(P[up, 0], P[lo, 0][o], self._phi(s, lo)[o])
        gap = np.clip(self._phi(s, up) - lo_phi, 0, None)
        r = np.linalg.norm(P[up] - sd["c"], axis=1)
        for _ in range(iters):
            delta = self._closure(s, gap, upper_share)
            resid = self.margin_gap(s, delta) + overlap_m
            gap = np.clip(gap + resid / r, 0, None)
        return self._closure(s, gap, upper_share)

    def brow(self, s, lift_m=0.0, inward_m=0.0, reg=0.02):
        """Brow raise (+lift) / lower (-lift) / knit (inward) fitted in GNM's eye-region PCA space."""
        h = self.h
        side_name = "left" if s == "L" else "right"
        comps = h.expression_index(side_name + "_eye_region_")
        B = h.expression_basis[comps].astype(np.float64)               # (K, V, 3)
        brow_w = h.groups[side_name + "_brow_region"] * self.skin
        c = self.side[s]["c"]
        region = self.skin & (np.linalg.norm(h.bind - c, axis=1) < 0.05)
        target = np.zeros((h.V, 3))
        medial = -np.sign(c[0])  # towards the nose
        target[:, 1] = lift_m * brow_w
        target[:, 0] = medial * inward_m * brow_w
        # lids & eyeball must stay put (lid motion is controlled separately)
        keep = np.maximum(self.side[s]["w_up"], self.side[s]["w_lo"]) > 0
        wt = np.where(brow_w > 0.05, 1.0, 0.25) * region
        wt[keep] = 3.0
        A = (B[:, region, :] * np.sqrt(wt[region])[None, :, None]).reshape(len(comps), -1).T
        y = (target[region] * np.sqrt(wt[region])[:, None]).ravel()
        coef = np.linalg.solve(A.T @ A + reg * np.eye(len(comps)) * (A ** 2).sum() / len(comps), A.T @ y)
        full = np.zeros(len(h.expression_names))
        full[comps] = coef
        delta = h.expression_delta(full)
        delta[~self.skin] = 0.0   # never move the eyeballs
        err = np.abs(delta[brow_w > 0.5] - target[brow_w > 0.5]).mean() * 1e3
        return delta, coef, err
