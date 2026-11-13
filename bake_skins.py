"""
bake_skins.py - transfers photographic MakeHuman (CC0) skin textures onto GNM's UV layout.

    blender -b ../tools/mh_source/mh_source.blend --python bake_skins.py -- \
        --npz ../tools/gnm/model/gnm_head.npz --skins ../tools/mh_source/skins \
        --out textures/_bake --size 4096 --sets young_caucasian_male,old_african_female

Pipeline (run inside the old MakeHuman scene, which holds the textured MH head):
  1. Similarity-align the MakeHuman head to GNM (eye centres + nose tip, then ICP on the face).
  2. Fit GNM's 170 head-identity coefficients to the MakeHuman surface (ridge regression on
     closest-point correspondences), so the GNM surface coincides with the textured MH skin.
  3. Cycles "selected to active" EMIT bakes from the MH skin (and eyebrow cards) onto the
     fitted GNM skin, written in GNM UDIM tile 1001.  Because UVs are shared by every GNM
     identity, the baked maps wrap directly onto any GNM head.
  4. Region masks (eye-socket lining, mouth interior, T-zone, lips) baked from GNM vertex
     groups for post-processing (see make_texture_sets.py).
Outputs per set:  <out>/<set>/skin_albedo_raw.png ; shared: brow_rgb.png, brow_alpha.png,
region_masks.png, fit.json (identity coefficients + fit error).
"""
import argparse
import json
import math
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "gnm_head"))
from gnm_model import GNMHead  # noqa: E402


def to_blender(p):
    p = np.asarray(p, float)
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], -1)


def to_gnm(p):
    p = np.asarray(p, float)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1)


def umeyama(src, dst, w=None):
    """Similarity (s, R, t) minimising sum w |s R src + t - dst|^2."""
    w = np.ones(len(src)) if w is None else w
    w = w / w.sum()
    ms, md = (w[:, None] * src).sum(0), (w[:, None] * dst).sum(0)
    xs, xd = src - ms, dst - md
    C = (w[:, None] * xd).T @ xs
    U, S, Vt = np.linalg.svd(C)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    var = (w * (xs ** 2).sum(1)).sum()
    s = np.trace(np.diag(S) @ D) / var
    return s, R, md - s * R @ ms


def nose_tip(P, eye_c):
    y0 = eye_c[1]
    m = (np.abs(P[:, 0]) < 0.006) & (P[:, 1] < y0 - 0.01) & (P[:, 1] > y0 - 0.07)
    return P[np.where(m)[0][np.argmax(P[m, 2])]]


def evaluated_mesh(ob):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    mw = np.array(ev.matrix_world)
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
    me.calc_loop_triangles()
    tris = np.empty(len(me.loop_triangles) * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", tris)
    tri_loops = np.empty(len(me.loop_triangles) * 3, np.int32)
    me.loop_triangles.foreach_get("loops", tri_loops)
    uv = np.empty(len(me.loops) * 2)
    me.uv_layers.active.data.foreach_get("uv", uv)
    ev.to_mesh_clear()
    return co, tris.reshape(-1, 3), tri_loops.reshape(-1, 3), uv.reshape(-1, 2)


def mesh_from_tris(name, co, tris, tri_loops, uv):
    """Object with one triangle per loop triangle, carrying the source UVs."""
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(co))
    me.vertices.foreach_set("co", co.astype(np.float32).ravel())
    n = len(tris)
    me.loops.add(n * 3)
    me.loops.foreach_set("vertex_index", tris.astype(np.int32).ravel())
    me.polygons.add(n)
    me.polygons.foreach_set("loop_start", (np.arange(n) * 3).astype(np.int32))
    me.polygons.foreach_set("loop_total", np.full(n, 3, np.int32))
    me.update(calc_edges=True)
    lay = me.uv_layers.new(name="UV")
    lay.data.foreach_set("uv", uv[tri_loops.ravel()].astype(np.float32).ravel())
    me.polygons.foreach_set("use_smooth", np.ones(n, bool))
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


# --------------------------------------------------------------------------- #
def fit(head, mh_co, mh_tris, cfg_mh, iters_icp=10, iters_id=5, lam=0.08):
    G = head.bind.copy()
    mhg = to_gnm(mh_co)
    # 1. landmarks: eye centres + nose tip
    src = np.array([to_gnm(cfg_mh["center_L"]), to_gnm(cfg_mh["center_R"]),
                    nose_tip(mhg, (to_gnm(cfg_mh["center_L"]) + to_gnm(cfg_mh["center_R"])) / 2)])
    dst = np.array([head.eye_center("L"), head.eye_center("R"),
                    nose_tip(G[head.mask("skin_exterior")], (head.eye_center("L") + head.eye_center("R")) / 2)])
    s, R, t = umeyama(src, dst)
    print("[fit] landmark similarity: scale %.4f" % s, flush=True)

    face = head.mask("skin_exterior", "&hockey_mask") | head.mask("forehead_region")
    face &= head.mask("skin_exterior")
    ext = head.mask("skin_exterior")

    def bvh_of(s, R, t):
        P = s * mhg @ R.T + t
        return BVHTree.FromPolygons([Vector(p) for p in P], mh_tris.tolist()), P

    def closest(bvh, pts):
        out = np.zeros_like(pts)
        d = np.zeros(len(pts))
        for i, p in enumerate(pts):
            loc, nrm, idx, dist = bvh.find_nearest(Vector(p))
            out[i] = loc
            d[i] = dist
        return out, d

    idx_face = np.where(face)[0]
    for it in range(iters_icp):
        bvh, P = bvh_of(s, R, t)
        cp, d = closest(bvh, G[idx_face])
        w = (d < 3 * np.median(d)).astype(float)
        # map the closest points back into MH space and re-solve MH->GNM
        cp_mh = ((cp - t) @ R) / s
        s, R, t = umeyama(cp_mh, G[idx_face], w)
        print("[fit] ICP %d: median %.2f mm mean %.2f mm" % (it, np.median(d) * 1e3, d.mean() * 1e3), flush=True)

    # 2. identity fit (head components only)
    comps = [i for i, n in enumerate(head.z["identity_names"]) if str(n).startswith("head_")]
    B = head.identity_basis[comps].astype(np.float64)
    coef = np.zeros(head.identity_basis.shape[0])
    idx_ext = np.where(ext)[0]
    wts = np.where(face[idx_ext], 1.0, 0.35)
    bvh, P = bvh_of(s, R, t)
    for it in range(iters_id):
        cur = head.template + np.tensordot(coef, head.identity_basis, axes=1)
        cp, d = closest(bvh, cur[idx_ext])
        ok = d < 0.02
        w = wts * ok
        A = (B[:, idx_ext, :] * np.sqrt(w)[None, :, None]).reshape(len(comps), -1)
        r = ((cp - head.template[idx_ext]) * np.sqrt(w)[:, None]).ravel()
        AAt = A @ A.T
        c = np.linalg.solve(AAt + lam * np.trace(AAt) / len(comps) * np.eye(len(comps)), A @ r)
        coef[:] = 0
        coef[comps] = c
        print("[fit] identity %d: face median %.2f mm  all-ext median %.2f mm  |c|max %.2f" % (
            it, np.median(d[face[idx_ext]]) * 1e3, np.median(d) * 1e3, np.abs(c).max()), flush=True)
    head.set_identity(coef)
    cp, d = closest(bvh, head.bind[idx_face])
    stats = {"face_median_mm": float(np.median(d) * 1e3), "face_p95_mm": float(np.percentile(d, 95) * 1e3)}
    print("[fit] final face distance median %.2f mm, p95 %.2f mm" % (stats["face_median_mm"], stats["face_p95_mm"]),
          flush=True)
    return (s, R, t), coef, stats


# --------------------------------------------------------------------------- #
def new_image(name, size, float_buf=False, colorspace="sRGB"):
    img = bpy.data.images.new(name, size, size, alpha=False, float_buffer=float_buf)
    img.colorspace_settings.name = colorspace
    img.generated_color = (0, 0, 0, 1)
    return img


def emission_material(name, img_node_image=None, color_attr=None, alpha_only=False):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    if img_node_image is not None:
        tx = nt.nodes.new("ShaderNodeTexImage")
        tx.image = img_node_image
        tx.interpolation = "Cubic"
        nt.links.new(tx.outputs["Alpha" if alpha_only else "Color"], em.inputs["Color"])
    elif color_attr:
        ca = nt.nodes.new("ShaderNodeVertexColor")
        ca.layer_name = color_attr
        nt.links.new(ca.outputs["Color"], em.inputs["Color"])
    return m


def bake_to(target, sources, img, margin=24, extrusion=0.008, max_dist=0.03):
    """EMIT bake of sources' surface colour onto target's active image node."""
    scn = bpy.context.scene
    for m in target.data.materials:
        nt = m.node_tree
        node = nt.nodes.get("BakeTarget") or nt.nodes.new("ShaderNodeTexImage")
        node.name = "BakeTarget"
        node.image = img
        nt.nodes.active = node
    bpy.ops.object.select_all(action="DESELECT")
    for s in sources:
        s.select_set(True)
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    bk = scn.render.bake
    bk.use_selected_to_active = bool(sources)
    bk.cage_extrusion = extrusion
    bk.max_ray_distance = max_dist
    bk.margin = margin
    bk.margin_type = "EXTEND"
    bk.use_clear = True
    t = time.time()
    bpy.ops.object.bake(type="EMIT")
    print("[bake] %s <- %s  %.1fs" % (img.name, [s.name for s in sources], time.time() - t), flush=True)


def save_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()


# --------------------------------------------------------------------------- #
def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(HERE, "..", "tools", "gnm", "model", "gnm_head.npz"))
    ap.add_argument("--skins", default=os.path.join(HERE, "..", "tools", "mh_source", "skins"))
    ap.add_argument("--out", default=os.path.join(HERE, "textures", "_bake"))
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--sets", default="young_caucasian_male")
    ap.add_argument("--samples", type=int, default=4)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    scn = bpy.context.scene
    cfg_mh = json.loads(scn["track1_config"])
    rig = bpy.data.objects[cfg_mh["armature"]]
    rig.data.pose_position = "REST"
    body = bpy.data.objects["Subject.body"]
    brow = bpy.data.objects.get("Subject.eyebrow001")
    for ob in bpy.data.objects:   # hide hair etc. from the bake
        ob.hide_render = ob not in (body, brow)
    bpy.context.view_layer.update()

    head = GNMHead(os.path.abspath(a.npz))
    co, tris, tri_loops, uv = evaluated_mesh(body)
    (s, R, t), coef, stats = fit(head, co, tris, cfg_mh)

    def mh_to_bake_space(p_blender):
        return to_blender(s * to_gnm(p_blender) @ R.T + t)

    # MH skin source
    src_body = mesh_from_tris("MHSkinSrc", mh_to_bake_space(co), tris, tri_loops, uv)
    src_brow = None
    if brow is not None:
        bco, btris, bloops, buv = evaluated_mesh(brow)
        src_brow = mesh_from_tris("MHBrowSrc", mh_to_bake_space(bco), btris, bloops, buv)
    body.hide_render = True
    if brow:
        brow.hide_render = True

    # GNM bake target: skin faces only (tile 1001 -> UV in [0,1])
    tiles = head.quad_tiles()
    sq = np.where(tiles == 1001)[0]
    quads = head.quads[sq]
    used = np.unique(quads)
    remap = -np.ones(head.V, np.int64)
    remap[used] = np.arange(len(used))
    me = bpy.data.meshes.new("GNMSkinBake")
    vb = to_blender(head.bind[used])
    me.vertices.add(len(vb))
    me.vertices.foreach_set("co", vb.astype(np.float32).ravel())
    me.loops.add(len(sq) * 4)
    me.loops.foreach_set("vertex_index", remap[quads].astype(np.int32).ravel())
    me.polygons.add(len(sq))
    me.polygons.foreach_set("loop_start", (np.arange(len(sq)) * 4).astype(np.int32))
    me.polygons.foreach_set("loop_total", np.full(len(sq), 4, np.int32))
    me.update(calc_edges=True)
    me.uv_layers.new(name="UDIM").data.foreach_set("uv", head.quad_uvs[sq].astype(np.float32).ravel())
    me.polygons.foreach_set("use_smooth", np.ones(len(sq), bool))
    # region masks from GNM vertex groups
    #   regions_a: R socket lining, G mouth interior, B T-zone (forehead / nose / glabella)
    #   regions_b: R lips, G nose, B cheeks
    g = head.groups
    ext = (g["skin_exterior"] > 1e-4).astype(float)
    mouth = np.clip(g["mouth_sock"], 0, 1)
    tzone = np.clip(g["forehead_region"] * 0.8 + g["nose_region"] + g["middle_brow_region"] * 0.8, 0, 1)
    lips = np.clip(g["upper_lip"] + g["lower_lip"], 0, 1)
    cheeks = np.clip(g["left_cheek_region"] + g["right_cheek_region"]
                     + 0.6 * (g["left_zygomatic_region"] + g["right_zygomatic_region"]), 0, 1)
    one = np.ones(head.V)
    for nm, cols in (("regions_a", (np.clip(1 - ext - mouth, 0, 1), mouth, tzone)),
                     ("regions_b", (lips, np.clip(g["nose_region"], 0, 1), cheeks))):
        col = np.stack(list(cols) + [one], -1)[used]
        ca = me.color_attributes.new(nm, "FLOAT_COLOR", "POINT")
        ca.data.foreach_set("color", col.astype(np.float32).ravel())
    target = bpy.data.objects.new("GNMSkinBake", me)
    scn.collection.objects.link(target)
    target.data.materials.append(emission_material("BakeTargetMat", color_attr="regions_a"))

    scn.render.engine = "CYCLES"
    scn.cycles.device = "CPU"
    scn.cycles.samples = a.samples
    scn.cycles.use_denoising = False

    # region masks (no selected-to-active: the target's own colour attributes)
    vc = target.data.materials[0].node_tree.nodes.get("Color Attribute") or next(
        n for n in target.data.materials[0].node_tree.nodes if n.type == "VERTEX_COLOR")
    for nm in ("regions_a", "regions_b"):
        vc.layer_name = nm
        img = new_image(nm, a.size, colorspace="Non-Color")
        bake_to(target, [], img, margin=8)
        save_image(img, os.path.join(a.out, nm + ".png"))

    # teeth/gums (1006) and tongue (1007) tiles, coloured from vertex groups
    for tile, nm in ((1006, "teeth"), (1007, "tongue")):
        tq = np.where(tiles == tile)[0]
        q = head.quads[tq]
        u = np.unique(q)
        rm = -np.ones(head.V, np.int64)
        rm[u] = np.arange(len(u))
        m2 = bpy.data.meshes.new("Bake_" + nm)
        m2.vertices.add(len(u))
        m2.vertices.foreach_set("co", to_blender(head.bind[u]).astype(np.float32).ravel())
        m2.loops.add(len(tq) * 4)
        m2.loops.foreach_set("vertex_index", rm[q].astype(np.int32).ravel())
        m2.polygons.add(len(tq))
        m2.polygons.foreach_set("loop_start", (np.arange(len(tq)) * 4).astype(np.int32))
        m2.polygons.foreach_set("loop_total", np.full(len(tq), 4, np.int32))
        m2.update(calc_edges=True)
        m2.uv_layers.new(name="UDIM").data.foreach_set("uv", head.quad_uvs[tq].astype(np.float32).ravel())
        if nm == "teeth":
            gum = np.clip(g["gums"][u], 0, 1)[:, None]
            col = (1 - gum) * np.array([0.80, 0.74, 0.62]) + gum * np.array([0.62, 0.24, 0.24])
        else:
            col = np.tile([0.55, 0.22, 0.24], (len(u), 1))
        col = np.c_[col, np.ones(len(u))]
        ca = m2.color_attributes.new("c", "FLOAT_COLOR", "POINT")
        ca.data.foreach_set("color", col.astype(np.float32).ravel())
        ob2 = bpy.data.objects.new("Bake_" + nm, m2)
        scn.collection.objects.link(ob2)
        ob2.data.materials.append(emission_material("mat_" + nm, color_attr="c"))
        img = new_image(nm, 1024)
        bake_to(ob2, [], img, margin=16)
        save_image(img, os.path.join(a.out, nm + ".png"))
        ob2.hide_render = True

    # eyebrows: colour and coverage
    if src_brow is not None:
        bimg = brow.active_material.node_tree.nodes
        brow_tex = next((n.image for n in bimg if n.type == "TEX_IMAGE" and n.image), None)
        if brow_tex is not None:
            for nm, alpha in (("brow_rgb", False), ("brow_alpha", True)):
                src_brow.data.materials.clear()
                src_brow.data.materials.append(emission_material(nm + "_mat", brow_tex, alpha_only=alpha))
                img = new_image(nm, a.size, colorspace="sRGB" if not alpha else "Non-Color")
                bake_to(target, [src_brow], img, margin=4, extrusion=0.004, max_dist=0.007)
                save_image(img, os.path.join(a.out, nm + ".png"))

    # skins
    for name in [x for x in a.sets.split(",") if x]:
        d = os.path.join(a.skins, name)
        pngs = [f for f in os.listdir(d) if f.endswith(".png")]
        sk = bpy.data.images.load(os.path.join(d, pngs[0]))
        src_body.data.materials.clear()
        src_body.data.materials.append(emission_material("src_" + name, sk))
        img = new_image("skin_" + name, a.size)
        bake_to(target, [src_body], img)
        save_image(img, os.path.join(a.out, name, "skin_albedo_raw.png"))

    with open(os.path.join(a.out, "fit.json"), "w") as f:
        json.dump({"identity": coef.tolist(), "similarity_scale": float(s), "fit": stats,
                   "note": "GNM identity fitted to the MakeHuman source head used for the bake"}, f)
    print("[bake] done ->", a.out, flush=True)


if __name__ == "__main__":
    main()
