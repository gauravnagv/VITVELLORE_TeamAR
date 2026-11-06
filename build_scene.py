"""
build_scene.py - builds scene.blend for Track 01 from open (CC0) MakeHuman assets.

Run once (needs the MPFB 2 add-on + makehuman_system_assets_cc0 pack installed):
    blender -b --factory-startup --python build_scene.py -- --out scene.blend

What it creates
    * Textured MakeHuman (MPFB) character with the "default" rig: eye.L / eye.R bones.
    * Eye bones re-centred on least-squares-fitted eyeball centres, and the eye
      mesh skinned 100 % to them, so a bone rotation == eyeball rotation about its centre.
    * Fixation target empty 35 cm in front of the eye midpoint.
    * Locked render camera 40 cm in front of the eye midpoint, 1024x1024.
    * Fixed 3-point area lighting + constant world (constant lighting for all frames).
    * All textures packed into the .blend, so evaluators do NOT need MPFB installed.
"""

import argparse
import glob
import json
import math
import os
import sys
import zipfile

import bpy
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))

CAMERA_DISTANCE_M = 0.40
FIXATION_DISTANCE_M = 0.35
LENS_MM = 70.0
CAMERA_SHIFT_Y = -0.10


def args_after_dashdash():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(HERE, "scene.blend"))
    p.add_argument("--asset-pack", default=os.path.join(HERE, "..", "tools", "downloads",
                                                         "makehuman_system_assets_cc0.zip"))
    p.add_argument("--skin", default="young_caucasian_male")
    p.add_argument("--eyes-texture", default="brown")
    p.add_argument("--gender", type=float, default=1.0)
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# MPFB access
# --------------------------------------------------------------------------- #
def mpfb_module():
    for mod in ("bl_ext.user_default.mpfb", "bl_ext.blender_org.mpfb", "mpfb"):
        try:
            bpy.ops.preferences.addon_enable(module=mod)
            return mod
        except Exception:
            continue
    raise RuntimeError("MPFB add-on not installed. Install it with:\n"
                       "  blender -b --command extension install-file -r user_default -e mpfb.zip")


def ensure_asset_pack(mod, pack_zip):
    import importlib
    ls = importlib.import_module(mod + ".services.locationservice").LocationService
    data_dir = ls.get_user_data()
    if not glob.glob(os.path.join(data_dir, "skins", "*")):
        if not os.path.isfile(pack_zip):
            raise RuntimeError("Asset pack missing: " + pack_zip)
        print("[build] extracting asset pack into", data_dir, flush=True)
        os.makedirs(data_dir, exist_ok=True)
        with zipfile.ZipFile(pack_zip) as z:
            z.extractall(data_dir)
        asvc = importlib.import_module(mod + ".services.assetservice").AssetService
        asvc.update_all_asset_lists()
    return data_dir


def pick(data_dir, sub, pattern, preferred):
    """Return an MPFB asset fragment 'folder/file.ext' (the format MPFB resolves)."""
    files = sorted(glob.glob(os.path.join(data_dir, sub, "**", pattern), recursive=True))
    if not files:
        return ""
    chosen = files[0]
    for f in files:
        if os.path.splitext(os.path.basename(f))[0].lower() == preferred.lower():
            chosen = f
            break
    return os.path.basename(os.path.dirname(chosen)) + "/" + os.path.basename(chosen)


def create_human(mod, data_dir, a):
    import importlib
    HS = importlib.import_module(mod + ".services.humanservice").HumanService
    OS = importlib.import_module(mod + ".services.objectservice").ObjectService

    info = HS._create_default_human_info_dict()
    info["name"] = "Subject"
    info["phenotype"]["gender"] = a.gender
    info["rig"] = "default"
    info["eyes"] = pick(data_dir, "eyes", "*.mhclo", "high-poly")
    info["eyebrows"] = pick(data_dir, "eyebrows", "*.mhclo", "eyebrow001")
    info["eyelashes"] = pick(data_dir, "eyelashes", "*.mhclo", "eyelashes01")
    info["hair"] = pick(data_dir, "hair", "*.mhclo", "short02")
    info["skin_mhmat"] = pick(data_dir, "skins", "*.mhmat", a.skin)
    info["skin_material_type"] = "ENHANCED_SSS"
    info["eyes_material_type"] = "PROCEDURAL_EYES"
    print("[build] assets:", json.dumps({k: info[k] for k in
          ("eyes", "eyebrows", "eyelashes", "skin_mhmat")}, indent=1), flush=True)

    settings = HS.get_default_deserialization_settings()
    settings["load_clothes"] = False
    settings["subdiv_levels"] = 1
    basemesh = HS.deserialize_from_dict(info, settings)

    rig = OS.find_object_of_type_amongst_nearest_relatives(basemesh, "Skeleton")
    eyes = OS.find_object_of_type_amongst_nearest_relatives(basemesh, "Eyes")
    return basemesh, rig, eyes


# --------------------------------------------------------------------------- #
# Eye geometry
# --------------------------------------------------------------------------- #
def fit_sphere(pts):
    A = np.c_[2 * pts, np.ones(len(pts))]
    b = (pts ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r = math.sqrt(sol[3] + c @ c)
    return c, r


def pupil_centre_uvs(eyes_obj):
    """UV coordinates where the eye material draws the pupil centre.

    MPFB's procedural eye (node group 'CenterCoordinates') and the MakeHuman eye
    texture both centre the iris at UV (0.295, 0.3) and (0.705, 0.7)."""
    ng = bpy.data.node_groups.get("CenterCoordinates")
    if ng:
        uvs = [(n.inputs["X"].default_value, n.inputs["Y"].default_value)
               for n in ng.nodes if n.type == "COMBXYZ"]
        if len(uvs) == 2:
            return uvs
    return [(0.295, 0.3), (0.705, 0.7)]


def surface_point_at_uv(eyes_obj, mask, uv_target):
    """3D world point(s) on the mesh whose UV equals uv_target (barycentric)."""
    me = eyes_obj.data
    me.calc_loop_triangles()
    uvl = me.uv_layers.active.data
    mw = eyes_obj.matrix_world
    t = np.array(uv_target)
    hits = []
    for tri in me.loop_triangles:
        if not all(mask[v] for v in tri.vertices):
            continue
        a, b, c = (np.array(uvl[li].uv) for li in tri.loops)
        m = np.array([b - a, c - a]).T
        if abs(np.linalg.det(m)) < 1e-12:
            continue
        u, v = np.linalg.solve(m, t - a)
        if u >= -1e-6 and v >= -1e-6 and u + v <= 1 + 1e-6:
            P = [np.array(mw @ me.vertices[i].co) for i in tri.vertices]
            hits.append(P[0] + u * (P[1] - P[0]) + v * (P[2] - P[0]))
    return hits


def eye_geometry(eyes_obj):
    """Per side: fitted eyeball centre, radius and the 3D point where the pupil is drawn."""
    mw = eyes_obj.matrix_world
    pts = np.array([tuple(mw @ v.co) for v in eyes_obj.data.vertices])
    uv_centres = pupil_centre_uvs(eyes_obj)
    out = {}
    for side, mask in (("L", pts[:, 0] > 0), ("R", pts[:, 0] < 0)):  # subject's left = +X
        p = pts[mask]
        # fit on the back 70 % of the eyeball (exclude the corneal bulge)
        y_cut = np.quantile(p[:, 1], 0.30)
        c, r = fit_sphere(p[p[:, 1] >= y_cut])
        hits = []
        for uv in uv_centres:
            hits += surface_point_at_uv(eyes_obj, mask, uv)
        if hits:
            pupil = min(hits, key=lambda q: q[1])  # front-most hit (face looks -Y)
            src = "uv"
        else:
            pupil = p[np.argmin(p[:, 1])]
            src = "apex"
        out[side] = {"center": c.tolist(), "radius": r, "pupil": list(map(float, pupil)),
                     "pupil_source": src, "index": np.where(mask)[0]}
    return out


def recentre_eye_bones(rig, eyes_obj, geo):
    """Move eye.L/eye.R heads to the fitted centres; skin eye halves 100 % to them."""
    bpy.ops.object.mode_set(mode="OBJECT") if bpy.context.object else None
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    inv = rig.matrix_world.inverted()
    for s in "LR":
        eb = rig.data.edit_bones["eye." + s]
        c = Vector(geo[s]["center"])
        axis = (Vector(geo[s]["pupil"]) - c).normalized()
        eb.head = inv @ c
        eb.tail = inv @ (c + axis * 0.02)
        eb.roll = 0.0
        eb.use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")

    # Rigid skinning: every eye vertex follows exactly one eye bone
    for vg in list(eyes_obj.vertex_groups):
        eyes_obj.vertex_groups.remove(vg)
    for s in "LR":
        vg = eyes_obj.vertex_groups.new(name="eye." + s)
        vg.add([int(i) for i in geo[s]["index"]], 1.0, "REPLACE")
    if not any(m.type == "ARMATURE" for m in eyes_obj.modifiers):
        m = eyes_obj.modifiers.new("Armature", "ARMATURE")
        m.object = rig


# --------------------------------------------------------------------------- #
# Camera, target, lights
# --------------------------------------------------------------------------- #
def add_camera_target_lights(mid):
    scn = bpy.context.scene
    cam_data = bpy.data.cameras.new("RenderCamera")
    cam_data.lens = LENS_MM
    cam_data.sensor_width = 36.0
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.clip_start = 0.01
    cam_data.shift_y = CAMERA_SHIFT_Y  # lens shift only: camera stays 40 cm in front of eye midpoint
    cam = bpy.data.objects.new("RenderCamera", cam_data)
    scn.collection.objects.link(cam)
    cam.location = mid + Vector((0, -CAMERA_DISTANCE_M, 0))
    cam.rotation_euler = (math.radians(90), 0, 0)  # looks along +Y at the face
    cam.lock_location = cam.lock_rotation = (True, True, True)
    scn.camera = cam

    tgt = bpy.data.objects.new("FixationTarget", None)
    tgt.empty_display_type = "SPHERE"
    tgt.empty_display_size = 0.005
    tgt.location = mid + Vector((0, -FIXATION_DISTANCE_M, 0))
    tgt.hide_render = True
    scn.collection.objects.link(tgt)

    def area(name, loc, energy, size, color=(1, 1, 1)):
        ld = bpy.data.lights.new(name, "AREA")
        ld.energy, ld.size, ld.color = energy, size, color
        ob = bpy.data.objects.new(name, ld)
        scn.collection.objects.link(ob)
        ob.location = mid + Vector(loc)
        d = (mid - ob.location).normalized()
        ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        return ob

    area("Key", (-0.35, -0.70, 0.30), 22, 0.6, (1.0, 0.97, 0.93))
    area("Fill", (0.45, -0.60, 0.05), 9, 0.8, (0.93, 0.96, 1.0))
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
    scn.view_settings.view_transform = "AgX" if "AgX" in [
        t.identifier for t in scn.view_settings.bl_rna.properties["view_transform"].enum_items] else "Filmic"
    scn.view_settings.look = "None"
    eng = {e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    r.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in eng else "BLENDER_EEVEE"
    scn.eevee.taa_render_samples = 64
    scn.cycles.samples = 128
    scn.cycles.seed = 0
    scn.frame_set(1)


# --------------------------------------------------------------------------- #
def main():
    a = args_after_dashdash()
    # empty scene
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)

    mod = mpfb_module()
    data_dir = ensure_asset_pack(mod, os.path.abspath(a.asset_pack))
    basemesh, rig, eyes = create_human(mod, data_dir, a)
    if rig is None or eyes is None:
        raise RuntimeError("Human creation failed (rig=%s eyes=%s)" % (rig, eyes))
    rig.name = "SubjectRig"
    eyes.name = "SubjectEyes"
    bpy.context.view_layer.update()

    geo = eye_geometry(eyes)
    recentre_eye_bones(rig, eyes, geo)
    for s in "LR":
        print("[build] eye %s centre=%s r=%.4f pupil=%s (%s)" % (
            s, [round(x, 4) for x in geo[s]["center"]], geo[s]["radius"],
            [round(x, 4) for x in geo[s]["pupil"]], geo[s]["pupil_source"]), flush=True)

    mid = (Vector(geo["L"]["center"]) + Vector(geo["R"]["center"])) / 2
    cam, tgt = add_camera_target_lights(mid)
    render_settings()

    cfg = {
        "armature": rig.name, "bone_L": "eye.L", "bone_R": "eye.R",
        "eyes_mesh": eyes.name, "camera": cam.name, "target": tgt.name,
        "center_L": geo["L"]["center"], "center_R": geo["R"]["center"],
        "pupil_rest_L": geo["L"]["pupil"], "pupil_rest_R": geo["R"]["pupil"],
        "eyeball_radius_L": geo["L"]["radius"], "eyeball_radius_R": geo["R"]["radius"],
        "eye_midpoint": list(mid), "camera_distance_m": CAMERA_DISTANCE_M,
        "fixation_distance_m": FIXATION_DISTANCE_M, "lens_mm": LENS_MM,
    }
    bpy.context.scene["track1_config"] = json.dumps(cfg)

    # self-contained file: pack every image, use relative paths
    try:
        bpy.ops.file.pack_all()
    except Exception as e:
        print("[build] pack_all warning:", e, flush=True)
    out = os.path.abspath(a.out)
    bpy.ops.wm.save_as_mainfile(filepath=out, compress=True)
    print("[build] saved", out, flush=True)


if __name__ == "__main__":
    main()
