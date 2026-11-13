"""Look polish for the GNM scene: lighting, world, skin shader, colour management.

Run inside Blender on scene.blend:
    blender -b scene.blend --python polish_look.py -- --preset v3 --render out.png [--crop out_crop.png] [--save]

--preset   which look to apply (default FINAL = "v3f"; "v2" = leave the scene unchanged)
--render   render the full 1024x1024 frame to this PNG
--crop     also render a 2x-resolution close-up of the right eye + cheek (skin detail check)
--save     write the polished look back into the opened .blend (compressed, textures packed)

The same values are used by build_scene_gnm.py, so a fresh build reproduces the look.
"""
import json
import os
import sys

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))

PRESETS = {
    "v2": None,   # leave the scene as built (reference render)
    "v3": {
        # Short-key portrait lighting: a small round key high to camera-left (round catch-light,
        # shape on the face), a weaker large fill, two warm rims for edge separation, and a
        # low-strength HDRI for soft ambient bounce and natural reflections (hidden from camera).
        "lights": {
            "Key":   {"loc": (-0.50, -0.62, 0.38), "energy": 34.0, "size": 0.22, "color": (1.0, 0.95, 0.89)},
            "Fill":  {"loc": (0.55, -0.58, 0.02), "energy": 5.5, "size": 0.45, "color": (0.92, 0.95, 1.0)},
            "Rim":   {"loc": (0.42, 0.48, 0.30), "energy": 22.0, "size": 0.40, "color": (1.0, 0.93, 0.85)},
            "RimL":  {"loc": (-0.45, 0.45, 0.22), "energy": 14.0, "size": 0.40, "color": (1.0, 0.93, 0.85)},
        },
        "world": {"hdri": "interior.exr", "strength": 0.35, "bg": (0.16, 0.17, 0.19), "bg_strength": 1.0},
        "skin": {
            "normal_strength": 0.85,
            "rough_lo": 0.38, "rough_hi": 0.62,       # remap of the roughness map
            "subsurface_weight": 0.55,
            "subsurface_radius": (1.0, 0.40, 0.22),
            "subsurface_scale": 0.006,
            "specular_ior_level": 0.50,
            "coat_weight": 0.08, "coat_roughness": 0.30,
            "sheen_weight": 0.08, "sheen_roughness": 0.35,
        },
        "view": {"transform": "AgX", "look": "AgX - Medium High Contrast", "exposure": 0.0},
    },
}


def _variant(base, **changes):
    """Deep-copy a preset and override nested keys given as section__key=value."""
    import copy
    p = copy.deepcopy(PRESETS[base])
    for k, v in changes.items():
        path = k.split("__")
        d = p
        for part in path[:-1]:
            d = d[part]
        d[path[-1]] = v
    return p


# Test round 2: v3 was over-exposed and flat (the HDRI fills from everywhere).
PRESETS["v3a"] = _variant("v3", view__exposure=-0.7, world__strength=0.15, world__bg_strength=0.30,
                          lights__Fill__energy=4.0)
PRESETS["v3b"] = _variant("v3a", view__look="AgX - Punchy", view__exposure=-0.55)
PRESETS["v3c"] = _variant("v3a", lights__Key__loc=(-0.68, -0.42, 0.34), lights__Key__energy=40.0,
                          lights__Fill__energy=3.0)
# Round 3: v3b's colour (Punchy) + v3c's side key; then the same with toned-down sclera.
PRESETS["v3d"] = _variant("v3c", view__look="AgX - Punchy", view__exposure=-0.55)
PRESETS["v3e"] = _variant("v3d")
PRESETS["v3e"]["eyes"] = {"white_max": 0.80, "knee": 0.45, "roughness": 0.30}
PRESETS["v3f"] = _variant("v3e", view__exposure=-0.4, lights__Fill__energy=4.0)

# Chosen 13 Nov 2026 after comparing v2..v3f on set01 and on sets 03/05/08/09.
FINAL = "v3f"


def args():
    import argparse
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="polish_look.py")
    p.add_argument("--preset", default=FINAL, choices=sorted(PRESETS))
    p.add_argument("--render", default=None)
    p.add_argument("--crop", default=None)
    p.add_argument("--save", action="store_true")
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--textures", default=None, help="render with another texture set (folder)")
    return p.parse_args(argv)


def eye_midpoint():
    cfg = json.loads(bpy.context.scene.get("track1_config", "{}"))
    if "eye_midpoint" in cfg:
        return Vector(cfg["eye_midpoint"])
    return (Vector(cfg["center_L"]) + Vector(cfg["center_R"])) / 2


def set_lights(spec):
    scn = bpy.context.scene
    mid = eye_midpoint()
    for name, s in spec.items():
        ob = bpy.data.objects.get(name)
        if ob is None:
            ld = bpy.data.lights.new(name, "AREA")
            ob = bpy.data.objects.new(name, ld)
            scn.collection.objects.link(ob)
        ld = ob.data
        ld.shape = "DISK"            # round catch-lights in the cornea
        ld.energy, ld.size, ld.color = s["energy"], s["size"], s["color"]
        ob.location = mid + Vector(s["loc"])
        ob.rotation_euler = (mid - ob.location).normalized().to_track_quat("-Z", "Y").to_euler()


def set_world(spec):
    scn = bpy.context.scene
    world = scn.world or bpy.data.worlds.new("World")
    scn.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    lp = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    bg_cam = nt.nodes.new("ShaderNodeBackground")       # what the camera sees: flat studio grey
    bg_cam.inputs["Color"].default_value = (*spec["bg"], 1)
    bg_cam.inputs["Strength"].default_value = spec["bg_strength"]
    bg_env = nt.nodes.new("ShaderNodeBackground")       # what lights the head: the HDRI
    hdri = os.path.join(os.path.dirname(bpy.app.binary_path), "%d.%d" % bpy.app.version[:2],
                        "datafiles", "studiolights", "world", spec["hdri"])
    if os.path.isfile(hdri):
        env = nt.nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(hdri, check_existing=True)
        nt.links.new(env.outputs["Color"], bg_env.inputs["Color"])
    else:
        print("[polish] HDRI not found, using flat ambient:", hdri, flush=True)
        bg_env.inputs["Color"].default_value = (*spec["bg"], 1)
    bg_env.inputs["Strength"].default_value = spec["strength"]
    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
    nt.links.new(bg_env.outputs["Background"], mix.inputs[1])
    nt.links.new(bg_cam.outputs["Background"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])


def set_skin(spec):
    m = bpy.data.materials.get("Skin")
    if m is None:
        print("[polish] no Skin material", flush=True)
        return
    nt = m.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    for n in nt.nodes:
        if n.type == "NORMAL_MAP":
            n.inputs["Strength"].default_value = spec["normal_strength"]
    # roughness map -> Map Range (remapped into a skin-plausible band)
    link = next((l for l in nt.links if l.to_socket == bsdf.inputs["Roughness"]), None)
    if link is not None:
        src = link.from_socket
        if link.from_node.type == "MAP_RANGE":
            mr = link.from_node
        else:
            nt.links.remove(link)
            mr = nt.nodes.new("ShaderNodeMapRange")
            nt.links.new(src, mr.inputs["Value"])
            nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])
        mr.inputs["To Min"].default_value = spec["rough_lo"]
        mr.inputs["To Max"].default_value = spec["rough_hi"]
    bsdf.inputs["Subsurface Weight"].default_value = spec["subsurface_weight"]
    bsdf.inputs["Subsurface Radius"].default_value = spec["subsurface_radius"]
    bsdf.inputs["Subsurface Scale"].default_value = spec["subsurface_scale"]
    bsdf.inputs["Specular IOR Level"].default_value = spec["specular_ior_level"]
    bsdf.inputs["Coat Weight"].default_value = spec["coat_weight"]
    bsdf.inputs["Coat Roughness"].default_value = spec["coat_roughness"]
    bsdf.inputs["Sheen Weight"].default_value = spec["sheen_weight"]
    bsdf.inputs["Sheen Roughness"].default_value = spec["sheen_roughness"]


def set_eyes(spec):
    """Tone down paper-white sclera: an RGB curve on the eye albedo that compresses only the
    top end (sclera), leaving the darker iris colours untouched."""
    for m in bpy.data.materials:
        if not m.name.startswith("EyeInterior") or not m.use_nodes:
            continue
        nt = m.node_tree
        bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            continue
        link = next((l for l in nt.links if l.to_socket == bsdf.inputs["Base Color"]), None)
        if link is None:
            continue
        if link.from_node.type == "CURVE_RGB" and link.from_node.name == "ScleraTone":
            cv = link.from_node
        else:
            src = link.from_socket
            nt.links.remove(link)
            cv = nt.nodes.new("ShaderNodeRGBCurve")
            cv.name = "ScleraTone"
            nt.links.new(src, cv.inputs["Color"])
            nt.links.new(cv.outputs["Color"], bsdf.inputs["Base Color"])
        c = cv.mapping.curves[3]                  # combined curve
        while len(c.points) > 2:
            c.points.remove(c.points[1])
        c.points[0].location = (0.0, 0.0)
        c.points[1].location = (1.0, spec["white_max"])
        p = c.points.new(spec["knee"], spec["knee"])  # identity below the knee
        cv.mapping.update()
        bsdf.inputs["Roughness"].default_value = spec["roughness"]


def set_view(spec):
    vs = bpy.context.scene.view_settings
    vs.view_transform = spec["transform"]
    try:
        vs.look = spec["look"]
    except TypeError:
        print("[polish] look not available:", spec["look"], flush=True)
    vs.exposure = spec["exposure"]


def apply(preset):
    p = PRESETS[preset]
    if p is None:
        return
    set_lights(p["lights"])
    set_world(p["world"])
    set_skin(p["skin"])
    if "eyes" in p:
        set_eyes(p["eyes"])
    set_view(p["view"])


def render(path, samples, crop=False):
    scn = bpy.context.scene
    r = scn.render
    r.engine = "BLENDER_EEVEE_NEXT"
    scn.eevee.taa_render_samples = samples
    keep = (r.resolution_x, r.resolution_y, r.use_border, r.use_crop_to_border)
    if crop:
        # right eye + upper cheek (image left), at 2x so pores/lashes are visible
        r.resolution_x = r.resolution_y = 2048
        r.use_border, r.use_crop_to_border = True, True
        r.border_min_x, r.border_max_x = 0.24, 0.50
        r.border_min_y, r.border_max_y = 0.40, 0.66
    r.filepath = os.path.abspath(path)
    bpy.ops.render.render(write_still=True)
    r.resolution_x, r.resolution_y, r.use_border, r.use_crop_to_border = keep
    print("[polish] wrote", path, flush=True)


def main():
    a = args()
    apply(a.preset)
    if a.textures:
        sys.path.insert(0, os.path.join(HERE, "wrapper"))
        import eyes
        eyes.set_textures(a.textures)
    if a.render:
        render(a.render, a.samples)
    if a.crop:
        render(a.crop, a.samples, crop=True)
    if a.save:
        try:
            bpy.ops.file.pack_all()
        except Exception as e:
            print("[polish] pack_all warning:", e, flush=True)
        bpy.ops.wm.save_mainfile(compress=True)
        print("[polish] saved", bpy.data.filepath, flush=True)


if __name__ == "__main__":
    main()
