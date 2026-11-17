"""
build_scene_gnm.py - builds scene.blend for Track 01 on Google's GNM v3 head.

    blender -b --factory-startup --python build_scene_gnm.py -- \
        --npz ../tools/gnm/model/gnm_head.npz --textures textures/set01 --out scene.blend

What it creates
  * GNM head mesh (17 821 verts, quads) with the official UDIM UV layout
    (1001 skin, 1002/1003 left eye interior/cornea, 1004/1005 right, 1006 teeth, 1007 tongue).
  * Armature with GNM's 4 joints (neck, head, left_eye, right_eye) and GNM's own skinning
    weights, so a bone rotation reproduces GNM's linear blend skinning exactly.
  * Shape keys for natural eye expressions (see gnm_head/gnm_model.py: EyeControls):
      blink_L/R, lidU_down/up_L/R, lidL_down/up_L/R (lid-gaze coupling),
      brow_up/down_L/R (fitted in GNM's learned eye-region expression space), pupil_dilate/constrict.
  * Skin / eye / cornea / teeth materials reading tiled textures  <set>/face.<UDIM>.png.
  * Locked render camera 40 cm in front of the eye midpoint, fixation target 35 cm, fixed lights.
GNM frame (Y up, Z forward) is converted to Blender (Z up, face looking -Y): (x, y, z) -> (x, -z, y).
"""
import argparse
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "gnm_head"))
from gnm_model import GNMHead, EyeControls  # noqa: E402
import eyelashes  # noqa: E402

CAMERA_DISTANCE_M = 0.40
FIXATION_DISTANCE_M = 0.35
LENS_MM = 70.0
CAMERA_SHIFT_Y = -0.10

# key angles (degrees) of the lid-follow shape keys; eyes.py scales them linearly
LID_KEY_DEG = {"lidU_down": 20.0, "lidU_up": -15.0, "lidL_down": 20.0, "lidL_up": -12.0}
BROW_KEYS = {"brow_up": (0.004, 0.0), "brow_down": (-0.003, 0.002)}   # (lift m, inward m)
PUPIL_COEF = 3.0

TILE_MATERIALS = {1001: "Skin", 1002: "EyeInterior_L", 1003: "Cornea_L", 1004: "EyeInterior_R",
                  1005: "Cornea_R", 1006: "Teeth", 1007: "Tongue"}


def to_blender(p):
    p = np.asarray(p, dtype=np.float64)
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)


def args_after_dashdash():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default=os.path.join(HERE, "..", "tools", "gnm", "model", "gnm_head.npz"))
    p.add_argument("--textures", default=os.path.join(HERE, "textures", "set01"))
    p.add_argument("--identity", default=None, help="optional .npy / .json identity coefficients (253)")
    p.add_argument("--out", default=os.path.join(HERE, "scene.blend"))
    p.add_argument("--lash-color", type=float, nargs=3, default=(0.035, 0.025, 0.02))
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
def build_mesh(head):
    verts = to_blender(head.bind)
    quads = head.quads
    me = bpy.data.meshes.new("GNMHead")
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    me.loops.add(quads.size)
    me.loops.foreach_set("vertex_index", quads.astype(np.int32).ravel())
    me.polygons.add(len(quads))
    me.polygons.foreach_set("loop_start", (np.arange(len(quads)) * 4).astype(np.int32))
    me.polygons.foreach_set("loop_total", np.full(len(quads), 4, np.int32))
    me.update(calc_edges=True)
    me.validate()

    uv = me.uv_layers.new(name="UDIM")
    uv.data.foreach_set("uv", head.quad_uvs_udim().astype(np.float32).ravel())

    tiles = head.quad_tiles()
    order = sorted(TILE_MATERIALS)
    me.polygons.foreach_set("material_index", np.array([order.index(t) for t in tiles], np.int32))
    me.polygons.foreach_set("use_smooth", np.ones(len(quads), bool))
    ob = bpy.data.objects.new("GNMHead", me)
    bpy.context.scene.collection.objects.link(ob)
    return ob, order


def build_armature(head, ob):
    arm_data = bpy.data.armatures.new("GNMRig")
    arm = bpy.data.objects.new("GNMRig", arm_data)
    bpy.context.scene.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    J = to_blender(head.joints)
    ebs = []
    for i, name in enumerate(head.joint_names):
        eb = arm_data.edit_bones.new(name)
        eb.head = Vector(J[i])
        if name.endswith("_eye"):
            eb.tail = Vector(J[i]) + Vector((0, -0.02, 0))   # along the gaze (-Y in Blender)
        else:
            eb.tail = Vector(J[i]) + Vector((0, 0, 0.04))
        eb.roll = 0.0
        ebs.append(eb)
    for i, p in enumerate(head.joint_parents):
        if p >= 0 and p != i:
            ebs[i].parent = ebs[p]
            ebs[i].use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    for i, name in enumerate(head.joint_names):
        w = head.skinning_weights[i]
        vg = ob.vertex_groups.new(name=name)
        for val in np.unique(np.round(w[w > 1e-6], 4)):
            idx = np.nonzero(np.abs(np.round(w, 4) - val) < 1e-9)[0]
            vg.add(idx.tolist(), float(val), "REPLACE")
    mod = ob.modifiers.new("Armature", "ARMATURE")
    mod.object = arm
    mod.use_deform_preserve_volume = False   # plain LBS, as in GNM
    ob.parent = arm
    return arm


def add_shape_keys(head, ob, ec):
    ob.shape_key_add(name="Basis", from_mix=False)
    base = to_blender(head.bind)
    lid_deltas = {}

    def add(name, delta_gnm):
        sk = ob.shape_key_add(name=name, from_mix=False)
        co = (base + to_blender(delta_gnm)).astype(np.float32)
        sk.data.foreach_set("co", co.ravel())
        sk.slider_min = 0.0
        # lid keys may extrapolate to 1.5x, except raising the upper lid (beyond ~15 deg it
        # would expose the socket lining above the sclera)
        sk.slider_max = 1.5 if name.startswith(("lidU_down", "lidL_")) else 1.0
        if name.startswith(("blink_", "lidU_", "lidL_")):
            lid_deltas[name] = delta_gnm
        return sk

    report = {}
    for s in "LR":
        add("blink_" + s, ec.blink(s))
        g = ec.margin_gap(s, ec.blink(s))
        report["blink_gap_mm_" + s] = [round(float(g.mean()) * 1e3, 3), round(float(g.max()) * 1e3, 3)]
        for key, deg in LID_KEY_DEG.items():
            lid = "upper" if key.startswith("lidU") else "lower"
            add("%s_%s" % (key, s), ec.lid_pitch(s, lid, math.radians(deg)))
        for key, (lift, inward) in BROW_KEYS.items():
            d, coef, err = ec.brow(s, lift, inward)
            add("%s_%s" % (key, s), d)
            report["%s_%s_fit_err_mm" % (key, s)] = round(float(err), 3)
    pi = head.expression_names.index("pupils_000")
    for name, sgn in (("pupil_dilate", 1.0), ("pupil_constrict", -1.0)):
        coef = np.zeros(len(head.expression_names))
        coef[pi] = sgn * PUPIL_COEF
        add(name, head.expression_delta(coef))
    margins = {s: {"upper": ec.side[s]["upper"].tolist(), "lower": ec.side[s]["lower"].tolist()} for s in "LR"}
    return report, margins, lid_deltas


def add_eyelashes(head, ec, head_ob, arm, lid_deltas, color=(0.035, 0.025, 0.02)):
    """Lash ribbons with shape keys that mirror the head's lid keys (driven 1:1)."""
    verts_all, quads_all, keys = [], [], {}
    offset = 0
    per_side = {}
    for s in "LR":
        v, q, roots, lov = eyelashes.generate(head, ec, s, seed=7 if s == "L" else 8)
        per_side[s] = (v, roots, lov, offset)
        verts_all.append(v)
        quads_all.append(q + offset)
        offset += len(v)
    V = np.vstack(verts_all)
    Q = np.vstack(quads_all)
    me = bpy.data.meshes.new("Eyelashes")
    vb = to_blender(V)
    me.vertices.add(len(vb))
    me.vertices.foreach_set("co", vb.astype(np.float32).ravel())
    me.loops.add(Q.size)
    me.loops.foreach_set("vertex_index", Q.astype(np.int32).ravel())
    me.polygons.add(len(Q))
    me.polygons.foreach_set("loop_start", (np.arange(len(Q)) * 4).astype(np.int32))
    me.polygons.foreach_set("loop_total", np.full(len(Q), 4, np.int32))
    me.update(calc_edges=True)
    me.polygons.foreach_set("use_smooth", np.ones(len(Q), bool))
    ob = bpy.data.objects.new("Eyelashes", me)
    bpy.context.scene.collection.objects.link(ob)
    # ride on the head bone (bone parenting is relative to the bone's tail frame)
    hb = arm.data.bones["head"]
    ob.parent = arm
    ob.parent_type = "BONE"
    ob.parent_bone = "head"
    ob.matrix_parent_inverse = (arm.matrix_world @ hb.matrix_local
                                @ Matrix.Translation((0, hb.length, 0))).inverted()
    m = bpy.data.materials.new("Eyelashes")
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = 0.45
    b.inputs["Coat Weight"].default_value = 0.2
    me.materials.append(m)

    ob.shape_key_add(name="Basis", from_mix=False)
    head_keys = head_ob.data.shape_keys
    for name, delta in lid_deltas.items():
        s = name[-1]
        v, roots, lov, off = per_side[s]
        d = np.zeros_like(V)
        d[off:off + len(v)] = eyelashes.follow(head, ec, s, v, roots, lov, delta)
        sk = ob.shape_key_add(name=name, from_mix=False)
        sk.data.foreach_set("co", (vb + to_blender(d)).astype(np.float32).ravel())
        sk.slider_min = 0.0
        sk.slider_max = 1.5 if name.startswith(("lidU_down", "lidL_")) else 1.0
        fc = sk.driver_add("value")
        drv = fc.driver
        drv.type = "AVERAGE"
        var = drv.variables.new()
        var.type = "SINGLE_PROP"
        var.targets[0].id_type = "KEY"
        var.targets[0].id = head_keys
        var.targets[0].data_path = 'key_blocks["%s"].value' % name
    return ob


# --------------------------------------------------------------------------- #
def tiled_image(tex_dir, stem, colorspace="sRGB"):
    """Load <tex_dir>/<stem>.<UDIM>.png as one tiled (UDIM) image with every tile present."""
    if not tex_dir or not os.path.isdir(tex_dir):
        return None
    tiles = sorted(int(f.split(".")[-2]) for f in os.listdir(tex_dir)
                   if f.startswith(stem + ".") and f.endswith(".png") and f.split(".")[-2].isdigit())
    if not tiles:
        return None
    img = bpy.data.images.load(os.path.join(tex_dir, "%s.%d.png" % (stem, tiles[0])))
    img.source = "TILED"
    img.filepath = os.path.join(tex_dir, "%s.<UDIM>.png" % stem)
    for t in tiles:
        if img.tiles.get(t) is None:
            img.tiles.new(tile_number=t)
    img.reload()
    img.name = stem
    img.colorspace_settings.name = colorspace
    return img


def build_materials(ob, order, tex_dir):
    albedo = tiled_image(tex_dir, "face") if tex_dir else None
    rough = tiled_image(tex_dir, "face_roughness", "Non-Color") if tex_dir else None
    normal = tiled_image(tex_dir, "face_normal", "Non-Color") if tex_dir else None
    mats = {}
    for tile in order:
        name = TILE_MATERIALS[tile]
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        nt = m.node_tree
        bsdf = nt.nodes["Principled BSDF"]
        out = nt.nodes["Material Output"]
        uvn = nt.nodes.new("ShaderNodeUVMap")
        uvn.uv_map = "UDIM"

        def tex(img, socket, non_color=False):
            if img is None:
                return None
            t = nt.nodes.new("ShaderNodeTexImage")
            t.image = img
            t.interpolation = "Cubic" if not non_color else "Linear"
            nt.links.new(uvn.outputs["UV"], t.inputs["Vector"])
            if socket is not None:
                nt.links.new(t.outputs["Color"], socket)
            return t

        if name == "Skin":
            bsdf.inputs["Base Color"].default_value = (0.80, 0.60, 0.50, 1)
            tex(albedo, bsdf.inputs["Base Color"])
            bsdf.inputs["Roughness"].default_value = 0.48
            if rough is not None:
                t = tex(rough, None, True)
                nt.links.new(t.outputs["Color"], bsdf.inputs["Roughness"])
            if normal is not None:
                t = tex(normal, None, True)
                nm = nt.nodes.new("ShaderNodeNormalMap")
                nm.uv_map = "UDIM"
                nm.inputs["Strength"].default_value = 0.35
                nt.links.new(t.outputs["Color"], nm.inputs["Color"])
                nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
            bsdf.inputs["Subsurface Weight"].default_value = 0.25
            bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.2)
            bsdf.inputs["Subsurface Scale"].default_value = 0.004
            bsdf.inputs["Specular IOR Level"].default_value = 0.45
            bsdf.inputs["Coat Weight"].default_value = 0.05
            bsdf.inputs["Coat Roughness"].default_value = 0.35
        elif name.startswith("EyeInterior"):
            bsdf.inputs["Base Color"].default_value = (0.85, 0.82, 0.78, 1)
            tex(albedo, bsdf.inputs["Base Color"])
            bsdf.inputs["Roughness"].default_value = 0.35
            bsdf.inputs["Subsurface Weight"].default_value = 0.1
            bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.3, 0.3)
            bsdf.inputs["Subsurface Scale"].default_value = 0.002
        elif name.startswith("Cornea"):
            # Clear, wet outer shell: transparent, with a Fresnel-weighted mirror
            # reflection (cornea / tear film IOR ~1.376).  The iris sits recessed behind
            # it in GNM's geometry, so parallax and the specular catch-lights are real;
            # this renders identically in EEVEE and Cycles.
            nt.nodes.remove(bsdf)
            tr = nt.nodes.new("ShaderNodeBsdfTransparent")
            gl = nt.nodes.new("ShaderNodeBsdfGlossy")
            gl.inputs["Roughness"].default_value = 0.03
            fr = nt.nodes.new("ShaderNodeFresnel")
            fr.inputs["IOR"].default_value = 1.376
            mix = nt.nodes.new("ShaderNodeMixShader")
            nt.links.new(fr.outputs["Fac"], mix.inputs["Fac"])
            nt.links.new(tr.outputs["BSDF"], mix.inputs[1])
            nt.links.new(gl.outputs["BSDF"], mix.inputs[2])
            nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
            m.surface_render_method = "BLENDED"   # no dither noise over the iris
            m.use_transparency_overlap = False
            m.use_transparent_shadow = True
        elif name == "Teeth":
            bsdf.inputs["Base Color"].default_value = (0.86, 0.80, 0.70, 1)
            tex(albedo, bsdf.inputs["Base Color"])
            bsdf.inputs["Roughness"].default_value = 0.3
        elif name == "Tongue":
            bsdf.inputs["Base Color"].default_value = (0.55, 0.22, 0.22, 1)
            tex(albedo, bsdf.inputs["Base Color"])
            bsdf.inputs["Roughness"].default_value = 0.4
        ob.data.materials.append(m)
        mats[name] = m
    return mats


# --------------------------------------------------------------------------- #
def add_camera_target_lights(mid):
    scn = bpy.context.scene
    cam_data = bpy.data.cameras.new("RenderCamera")
    cam_data.lens = LENS_MM
    cam_data.sensor_width = 36.0
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.clip_start = 0.01
    cam_data.shift_y = CAMERA_SHIFT_Y
    cam = bpy.data.objects.new("RenderCamera", cam_data)
    scn.collection.objects.link(cam)
    cam.location = mid + Vector((0, -CAMERA_DISTANCE_M, 0))
    cam.rotation_euler = (math.radians(90), 0, 0)
    cam.lock_location = cam.lock_rotation = (True, True, True)
    scn.camera = cam

    tgt = bpy.data.objects.new("FixationTarget", None)
    tgt.empty_display_type = "SPHERE"
    tgt.empty_display_size = 0.005
    tgt.location = mid + Vector((0, -FIXATION_DISTANCE_M, 0))
    tgt.hide_render = True
    scn.collection.objects.link(tgt)

    def area(name, loc, energy, size, color=(1, 1, 1), shape="DISK"):
        # Round lights: the cornea mirrors them, so their shape is the catch-light shape.
        ld = bpy.data.lights.new(name, "AREA")
        ld.energy, ld.size, ld.color, ld.shape = energy, size, color, shape
        ob = bpy.data.objects.new(name, ld)
        scn.collection.objects.link(ob)
        ob.location = mid + Vector(loc)
        d = (mid - ob.location).normalized()
        ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        return ob

    # Key/Fill kept small (same energy, so the skin exposure is unchanged): a large
    # source gives a big dim square reflection that greys out the pupil.
    area("Key", (-0.35, -0.70, 0.30), 22, 0.25, (1.0, 0.97, 0.93))
    area("Fill", (0.45, -0.60, 0.05), 9, 0.40, (0.93, 0.96, 1.0))
    area("Rim", (0.0, 0.50, 0.40), 12, 0.5)

    world = scn.world or bpy.data.worlds.new("World")
    scn.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs[0].default_value = (0.18, 0.19, 0.21, 1)
    bg.inputs[1].default_value = 0.25
    return cam, tgt


def render_settings():
    scn = bpy.context.scene
    r = scn.render
    r.resolution_x = r.resolution_y = 1024
    r.resolution_percentage = 100
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGB"
    r.film_transparent = False
    scn.view_settings.view_transform = "AgX"
    scn.view_settings.look = "None"
    r.engine = "BLENDER_EEVEE_NEXT"
    scn.eevee.taa_render_samples = 64
    scn.eevee.use_raytracing = True          # needed for corneal refraction in EEVEE
    scn.eevee.ray_tracing_options.resolution_scale = "1"
    scn.eevee.use_shadows = True
    scn.cycles.samples = 128
    scn.cycles.seed = 0
    scn.frame_set(1)


def _rel(path):
    try:
        return os.path.relpath(os.path.abspath(path), HERE).replace("\\", "/")
    except ValueError:  # different drive on Windows
        return os.path.abspath(path).replace("\\", "/")


# --------------------------------------------------------------------------- #
def main():
    a = args_after_dashdash()
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)

    identity = None
    if a.identity:
        if a.identity.endswith(".npy"):
            identity = np.load(a.identity)
        else:
            js = json.load(open(a.identity))
            identity = np.array(js["identity"] if isinstance(js, dict) else js, dtype=float)
    head = GNMHead(os.path.abspath(a.npz), identity)
    ob, order = build_mesh(head)
    arm = build_armature(head, ob)
    ec = EyeControls(head)
    report, margins, lid_deltas = add_shape_keys(head, ob, ec)
    lash_ob = add_eyelashes(head, ec, ob, arm, lid_deltas, color=tuple(a.lash_color))
    sys.path.insert(0, HERE)
    import add_expression_keys  # noqa: E402  (smile, squint_L/R: GNM-fitted whole-face keys)
    report.update(add_expression_keys.add_keys(head, ec, ob, lash_ob))
    import add_wrinkles  # noqa: E402  (driven forehead / glabella / crow's-feet creases)
    print("[build] shape keys:", [k.name for k in ob.data.shape_keys.key_blocks], flush=True)
    print("[build] report:", json.dumps(report), flush=True)
    build_materials(ob, order, os.path.abspath(a.textures) if a.textures else None)
    report.update(add_wrinkles.add(head, ob))      # needs the Skin material
    import add_hair  # noqa: E402  (3D buzz cut, eyebrows, vellus hair on the skin surface)
    report["hair"] = add_hair.add(head, ob, 1.0, os.path.abspath(a.textures) if a.textures else None)
    sub = ob.modifiers.new("Subdivision", "SUBSURF")
    sub.levels, sub.render_levels = 1, 2

    # eye geometry for the wrapper (Blender frame)
    cfg = {"armature": arm.name, "bone_L": "left_eye", "bone_R": "right_eye", "eyes_mesh": ob.name,
           "head_bones": ["neck", "head"],
           "head_mesh": ob.name, "model": "GNM v3 (google/gnm-v3, Apache-2.0)"}
    for s, side in (("L", "left"), ("R", "right")):
        c = to_blender(head.eye_center(s))
        pm = head.mask(side + "_eye", "&eye_interiors", "&pupils")
        pupil = to_blender(head.bind[pm].mean(axis=0))
        # project the pupil centroid onto the pupil-disc surface along the eye axis
        cfg["center_" + s] = c.tolist()
        cfg["pupil_rest_" + s] = pupil.tolist()
        cfg["eyeball_radius_" + s] = float(np.linalg.norm(head.bind[head.mask(side + "_eye", "&eye_interiors")]
                                                          - head.eye_center(s), axis=1).max())
    mid = (Vector(cfg["center_L"]) + Vector(cfg["center_R"])) / 2
    cam, tgt = add_camera_target_lights(mid)
    render_settings()
    cfg.update({"camera": cam.name, "target": tgt.name, "eye_midpoint": list(mid),
                "camera_distance_m": CAMERA_DISTANCE_M, "fixation_distance_m": FIXATION_DISTANCE_M,
                "lens_mm": LENS_MM, "lid_key_deg": LID_KEY_DEG, "lid_margins": margins,
                "build_report": report, "textures": _rel(a.textures) if a.textures else None,
                "expression_keys": list(add_expression_keys.KEYS)})
    bpy.context.scene["track1_config"] = json.dumps(cfg)
    import add_tearline  # noqa: E402  (lower-lid tear bead; needs track1_config + eye materials)
    report.update(add_tearline.add())
    import add_sclera_shading  # noqa: E402  (live lid contact shading on the eye whites)
    report.update(add_sclera_shading.add())
    # Final look (portrait key/fill/rims, HDRI ambient hidden from camera, skin shader,
    # sclera tone, AgX Punchy) lives in polish_look.py so the build and the shipped
    # scene.blend share one definition.  Needs track1_config (eye midpoint) set above.
    sys.path.insert(0, HERE)
    import polish_look  # noqa: E402
    polish_look.apply(polish_look.FINAL)
    try:
        bpy.ops.file.pack_all()
    except Exception as e:
        print("[build] pack_all warning:", e, flush=True)
    out = os.path.abspath(a.out)
    bpy.ops.wm.save_as_mainfile(filepath=out, compress=True)
    print("[build] saved", out, flush=True)


if __name__ == "__main__":
    main()
