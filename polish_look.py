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

# Cycles variants (path tracing adds real bounce light, which lifted the shadows and flattened
# the EEVEE-tuned look): less HDRI ambient, darker exposure, slightly stronger key.
PRESETS["c1"] = _variant("v3f", world__strength=0.10, view__exposure=-0.65)
PRESETS["c2"] = _variant("c1", lights__Key__energy=48.0, lights__Fill__energy=2.5, view__exposure=-0.8)
PRESETS["c3"] = _variant("c2", view__look="AgX - Medium High Contrast", view__exposure=-0.7)
CYCLES_FINAL = "c2"

# ---- v8 "cinematic" looks (14 Nov 2026), modelled on the organiser's reference videos
# (Chris Jones "Eye Piece" / "Flexpressions"): low-key side light falling off into a dark
# background, glossier skin so pores and creases glint, iris relief, eyes-in-focus depth of
# field, vignette.  c8a = moderate (face fully readable), c8b = dramatic, c8c = Eye-Piece dark.
_C8_SKIN = {
    # 0.85 (v11, 17 Nov): at 1.0 the pore normal still read as pebbly where the key rakes the
    # cheek and in the temple highlight (A/B tools_test_v11.py: shadow-cheek band energy -8..-16 %,
    # pores kept; 0.70 started to look soft).
    "normal_strength": 0.85,
    "rough_lo": 0.30, "rough_hi": 0.58,
    "subsurface_weight": 0.60,
    "subsurface_radius": (1.0, 0.38, 0.20),
    "subsurface_scale": 0.006,
    "specular_ior_level": 0.62,
    "coat_weight": 0.14, "coat_roughness": 0.28,
    "sheen_weight": 0.10, "sheen_roughness": 0.35,
}
PRESETS["c8a"] = {
    "lights": {
        "Key":  {"loc": (-0.72, -0.30, 0.30), "energy": 62.0, "size": 0.26, "color": (1.0, 0.94, 0.87)},
        # glossy=False (17 Nov): on right gaze the cornea mirrored this big dim soft-box as a grey
        # disc over the pupil; it still lights the skin, its specular share is negligible.
        "Fill": {"loc": (0.60, -0.55, 0.00), "energy": 1.6, "size": 0.55, "color": (0.90, 0.94, 1.0),
                 "glossy": False},
        "Rim":  {"loc": (0.45, 0.45, 0.28), "energy": 30.0, "size": 0.30, "color": (1.0, 0.90, 0.80)},
        "RimL": {"loc": (-0.45, 0.45, 0.22), "energy": 10.0, "size": 0.30, "color": (1.0, 0.92, 0.84)},
    },
    "world": {"hdri": "interior.exr", "strength": 0.05, "bg": (0.020, 0.021, 0.024), "bg_strength": 1.0},
    "skin": dict(_C8_SKIN),
    "eyes": {"white_max": 0.74, "knee": 0.42, "roughness": 0.22},
    "iris_bump": 0.35,
    "dof_fstop": 8.0,
    "post": {"vignette": 0.35, "dispersion": 0.006},
    "view": {"transform": "AgX", "look": "AgX - Punchy", "exposure": -0.55},
}
PRESETS["c8b"] = _variant("c8a", lights__Key__loc=(-0.80, -0.12, 0.26), lights__Key__energy=70.0,
                          lights__Fill__energy=0.6, world__strength=0.025,
                          world__bg=(0.010, 0.010, 0.012), view__exposure=-0.45)
PRESETS["c8c"] = _variant("c8b", eyes={"white_max": 0.66, "knee": 0.38, "roughness": 0.20}, lights__Key__size=0.14, lights__Fill__energy=0.0, lights__RimL__energy=0.0,
                          world__strength=0.012, world__bg=(0.004, 0.004, 0.005), view__exposure=-0.35)
CYCLES_V8 = "c8a"


def set_iris_bump(strength):
    """Iris relief: bump from the eye texture's luminance, so fibres, crypts and the
    collarette catch the key light (the sclera is near-uniform, so it stays smooth)."""
    for m in bpy.data.materials:
        if not m.name.startswith("EyeInterior") or not m.use_nodes:
            continue
        nt = m.node_tree
        bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        tex = next((n for n in nt.nodes if n.type == "TEX_IMAGE"), None)
        if bsdf is None or tex is None:
            continue
        bump = nt.nodes.get("IrisBump")
        if bump is None:
            bw = nt.nodes.new("ShaderNodeRGBToBW")
            bump = nt.nodes.new("ShaderNodeBump")
            bump.name = "IrisBump"
            nt.links.new(tex.outputs["Color"], bw.inputs["Color"])
            nt.links.new(bw.outputs["Val"], bump.inputs["Height"])
            nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
        bump.inputs["Strength"].default_value = strength
        bump.inputs["Distance"].default_value = 0.0004


def set_dof(fstop):
    """Focus on the eyes (camera -> eye midpoint); fstop None/0 disables depth of field."""
    cam = bpy.context.scene.camera
    if cam is None:
        return
    d = cam.data.dof
    d.use_dof = bool(fstop)
    if fstop:
        d.focus_object = None
        d.focus_distance = (cam.matrix_world.translation - eye_midpoint()).length
        d.aperture_fstop = fstop
        d.aperture_blades = 7


def set_closeup_camera(eye="R", yaw_deg=-14.0, pitch_deg=4.0, distance=0.17, lens=85.0, fstop=2.8):
    """Cinematic close-up in the style of the reference 'Eye Piece' video: one eye, brow and
    upper cheek, seen from slightly to the temporal side and above, shallow focus on the
    cornea.  The subject's right eye (R) sits at -X (image left), the side the key light
    comes from.  Negative yaw puts the camera on the nasal (shadow) side, so the key rakes
    across the frame and the far side falls off into shadow, as in the reference.
    Call AFTER the motion is baked (gaze targets use the main camera)."""
    import math
    scn = bpy.context.scene
    cfg = json.loads(scn.get("track1_config", "{}"))
    c = Vector(cfg["center_" + eye])
    sgn = -1.0 if eye == "R" else 1.0
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    d = Vector((sgn * math.sin(y) * math.cos(p), -math.cos(y) * math.cos(p), math.sin(p))).normalized()
    cam = scn.camera
    cam.location = c + d * distance
    cam.rotation_euler = (c - cam.location).normalized().to_track_quat("-Z", "Y").to_euler()
    cd = cam.data
    cd.lens, cd.sensor_width, cd.sensor_fit = lens, 36.0, "HORIZONTAL"
    cd.shift_x = cd.shift_y = 0.0
    cd.clip_start = 0.005
    cd.dof.use_dof = True
    cd.dof.focus_object = None
    cd.dof.focus_distance = distance - 0.011          # cornea apex ~11 mm in front of the centre
    cd.dof.aperture_fstop = fstop
    cd.dof.aperture_blades = 7


def _set_in(node, names, value):
    """Set a compositor node parameter that is a property in older Blender and an input
    socket in 4.5+.  Returns True if something was set."""
    for n in names:
        if hasattr(node, n):
            try:
                setattr(node, n, value)
                if n == "size_x" and hasattr(node, "size_y"):
                    node.size_y = value
                return True
            except (AttributeError, TypeError):
                pass
        s = node.inputs.get(n) if hasattr(node, "inputs") else None
        if s is not None:
            try:
                s.default_value = value
            except TypeError:
                s.default_value = (value, value)
            return True
    return False


def set_post(spec):
    """Compositor: subtle lens dispersion (colour fringing toward the corners, no geometric
    distortion, so the 2D pupil ground truth is unchanged) and a soft vignette."""
    scn = bpy.context.scene
    scn.use_nodes = True
    nt = scn.node_tree
    nt.nodes.clear()
    rl = nt.nodes.new("CompositorNodeRLayers")
    comp = nt.nodes.new("CompositorNodeComposite")
    cur = rl.outputs["Image"]
    if spec.get("dispersion"):
        ld = nt.nodes.new("CompositorNodeLensdist")
        _set_in(ld, ("Distortion",), 0.0)
        _set_in(ld, ("Dispersion",), spec["dispersion"])
        nt.links.new(cur, ld.inputs["Image"])
        cur = ld.outputs["Image"]
    if spec.get("vignette"):
        em = nt.nodes.new("CompositorNodeEllipseMask")
        if em.inputs.get("Size") is not None:              # Blender 4.5+: sockets
            em.inputs["Size"].default_value = (0.95, 0.95)
        else:                                              # older: properties
            em.mask_width = em.mask_height = 0.95
        bl = nt.nodes.new("CompositorNodeBlur")
        bl.filter_type = "GAUSS"
        bl.use_relative = False
        px = int(0.25 * bpy.context.scene.render.resolution_x)
        if bl.inputs.get("Size") is not None:
            bl.inputs["Size"].default_value = (px, px)
        else:
            bl.size_x = bl.size_y = px
        mr = nt.nodes.new("CompositorNodeMapRange")
        mr.inputs["To Min"].default_value = 1.0 - spec["vignette"]
        mr.inputs["To Max"].default_value = 1.0
        mix = nt.nodes.new("CompositorNodeMixRGB")
        mix.blend_type = "MULTIPLY"
        mix.inputs["Fac"].default_value = 1.0
        nt.links.new(em.outputs["Mask"], bl.inputs["Image"])
        nt.links.new(bl.outputs["Image"], mr.inputs["Value"])
        nt.links.new(cur, mix.inputs[1])
        nt.links.new(mr.outputs["Value"], mix.inputs[2])
        cur = mix.outputs["Image"]
    nt.links.new(cur, comp.inputs["Image"])


def cycles_settings(samples=128):
    """Path-tracing settings used for the final videos (GPU OptiX/CUDA when available)."""
    scn = bpy.context.scene
    c = scn.cycles
    c.samples = samples
    c.use_denoising = True
    c.denoiser = "OPENIMAGEDENOISE"           # with albedo+normal guides: keeps pores and lashes
    c.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    c.denoising_prefilter = "ACCURATE"
    try:
        c.denoising_use_gpu = True
    except AttributeError:
        pass
    c.max_bounces, c.diffuse_bounces, c.glossy_bounces = 8, 4, 4
    c.transmission_bounces, c.transparent_max_bounces = 8, 12
    c.sample_clamp_indirect = 10.0               # no fireflies from the wet cornea
    c.seed = 0
    c.use_animated_seed = True                   # noise changes per frame like film grain
    scn.render.use_persistent_data = True        # keep textures/shaders on the GPU between frames


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
        ob.visible_glossy = s.get("glossy", True)


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


def fix_eyes_for_cycles(sclera_sss=0.1, sclera_sss_scale=0.0008):
    """Path-tracing fixes for GNM's eyes (harmless in EEVEE):
    * The transparent outer eye shell (eye_exteriors, the whole globe) mixes in a glossy
      layer for the wet corneal reflection.  In Cycles that gloss is partly opaque to shadow
      rays, so at grazing angles the shell cast hard dark wedges onto the sclera.  The shell
      is made fully transparent for shadow rays only (reflections unchanged).
    * The sclera's subsurface radius was Blender's default 5 cm, about four times the eyeball:
      light diffused through the whole globe and picked up the dark socket.  Real sclera
      scatters over well under a millimetre."""
    for m in bpy.data.materials:
        if not m.use_nodes:
            continue
        nt = m.node_tree
        if m.name.startswith("Cornea"):
            outn = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
            link = next((l for l in nt.links if l.to_socket == outn.inputs["Surface"]), None)
            if link is None or link.from_node.name == "ShadowPass":
                continue
            src = link.from_socket
            nt.links.remove(link)
            lp = nt.nodes.new("ShaderNodeLightPath")
            tr = nt.nodes.new("ShaderNodeBsdfTransparent")
            mix = nt.nodes.new("ShaderNodeMixShader")
            mix.name = "ShadowPass"
            nt.links.new(lp.outputs["Is Shadow Ray"], mix.inputs["Fac"])
            nt.links.new(src, mix.inputs[1])
            nt.links.new(tr.outputs["BSDF"], mix.inputs[2])
            nt.links.new(mix.outputs["Shader"], outn.inputs["Surface"])
        elif m.name.startswith("EyeInterior"):
            b = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if b is not None:
                b.inputs["Subsurface Weight"].default_value = sclera_sss
                b.inputs["Subsurface Scale"].default_value = sclera_sss_scale
                if PUPIL_SPEC_FIX:
                    _pupil_no_spec(nt, b)


# 17 Nov 2026: the pupil rendered flat grey at ~25-35 deg gaze.  The pupil is a flat textured
# disc with the interior's dielectric specular (rough 0.22-0.30), so when its normal bisects the
# light and camera it shows a uniform sheen of the key (left gaze) or fill/HDRI (right gaze);
# the iris got the same milky veil.  A real pupil is a hole and the iris sits in aqueous humour,
# so neither has a surface sheen: the wet reflection comes from the cornea shell.  Specular is
# therefore faded to IRIS_SPEC inside the limbus, found from the planar eye UVs (tile centre
# 0.5,0.5, limbus radius 0.185 - gnm_head/textures.py); pupil dilation moves geometry, not UVs.
PUPIL_SPEC_FIX = True
IRIS_SPEC = 0.0
LIMBUS_UV = (0.180, 0.192)       # UV radius: <= lo -> IRIS_SPEC, >= hi -> sclera unchanged


def _pupil_no_spec(nt, bsdf):
    tex = next((n for n in nt.nodes if n.type == "TEX_IMAGE"), None)
    if tex is None:
        return
    mr = nt.nodes.get("PupilSpec")
    if mr is None:
        spec = bsdf.inputs["Specular IOR Level"]
        for l in [l for l in nt.links if l.to_socket == spec]:
            nt.links.remove(l)
        uvl = tex.inputs["Vector"].links
        if uvl:
            uv = uvl[0].from_socket
        else:
            uvn = nt.nodes.new("ShaderNodeUVMap")
            uvn.uv_map = "UDIM"
            uv = uvn.outputs["UV"]
        fr = nt.nodes.new("ShaderNodeVectorMath")       # fract(uv) - 0.5 -> radius in the tile
        fr.operation = "FRACTION"
        sub = nt.nodes.new("ShaderNodeVectorMath")
        sub.operation = "SUBTRACT"
        sub.inputs[1].default_value = (0.5, 0.5, 0.0)
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        comb = nt.nodes.new("ShaderNodeCombineXYZ")     # drop z before taking the length
        ln = nt.nodes.new("ShaderNodeVectorMath")
        ln.operation = "LENGTH"
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.name = "PupilSpec"
        mr.clamp = True
        nt.links.new(uv, fr.inputs[0])
        nt.links.new(fr.outputs["Vector"], sub.inputs[0])
        nt.links.new(sub.outputs["Vector"], sep.inputs["Vector"])
        nt.links.new(sep.outputs["X"], comb.inputs["X"])
        nt.links.new(sep.outputs["Y"], comb.inputs["Y"])
        nt.links.new(comb.outputs["Vector"], ln.inputs[0])
        nt.links.new(ln.outputs["Value"], mr.inputs["Value"])
        nt.links.new(mr.outputs["Result"], spec)
        mr["spec"] = float(spec.default_value)
    mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = LIMBUS_UV
    mr.inputs["To Min"].default_value = IRIS_SPEC
    mr.inputs["To Max"].default_value = mr.get("spec", 0.5)


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
    if "iris_bump" in p:
        set_iris_bump(p["iris_bump"])
    if "dof_fstop" in p:
        set_dof(p["dof_fstop"])
    if "post" in p:
        set_post(p["post"])
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
