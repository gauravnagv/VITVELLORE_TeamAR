"""
add_hair.py - real 3D hair on the GNM head: short buzz cut, eyebrows and fine vellus hair
("peach fuzz"), as Blender hair curves attached to the skin (Deform Curves on Surface), so
they follow the armature (head/neck) and the brow / lid / expression shape keys.  Idempotent.

    blender -b scene.blend --python add_hair.py -- [--save] [--density 1.0]

Strands come from gnm_head/hair.py; colours follow the active texture set (set.json:
eyebrow colour, age) through set_hair_colour(), which eyes.set_textures() calls.
"""
import json
import os
import sys

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "gnm_head"))

GROUPS = {   # object name: (generator, root radius m, tip radius m, material)
    "HairScalp": ("buzz", 0.000032, 0.000012, "HairScalp"),
    "HairBrows": ("brows", 0.000040, 0.000010, "HairBrows"),
    "HairFuzz": ("fuzz", 0.0000045, 0.0000015, "HairFuzz"),
}


def to_blender(p):
    p = np.asarray(p, dtype=np.float64)
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)


def srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _deform_group():
    ng = bpy.data.node_groups.get("HairDeformOnSurface")
    if ng:
        return ng
    ng = bpy.data.node_groups.new("HairDeformOnSurface", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gi = ng.nodes.new("NodeGroupInput")
    go = ng.nodes.new("NodeGroupOutput")
    dn = ng.nodes.new("GeometryNodeDeformCurvesOnSurface")
    ng.links.new(gi.outputs[0], dn.inputs[0])
    ng.links.new(dn.outputs[0], go.inputs[0])
    return ng


def _material(name):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    h = nt.nodes.new("ShaderNodeBsdfHairPrincipled")
    h.name = "HairBSDF"
    try:
        h.model = "CHIANG"
    except (AttributeError, TypeError):
        pass
    h.parametrization = "MELANIN"
    h.inputs["Roughness"].default_value = 0.32 if name != "HairFuzz" else 0.45
    h.inputs["Radial Roughness"].default_value = 0.35
    h.inputs["Coat"].default_value = 0.0
    if "Random Roughness" in h.inputs:
        h.inputs["Random Roughness"].default_value = 0.15
    if "Random Color" in h.inputs:
        h.inputs["Random Color"].default_value = 0.10
    info = nt.nodes.new("ShaderNodeHairInfo")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.name = "TipFade"
    # tips thin out (and vellus hair is translucent overall)
    ramp = nt.nodes.new("ShaderNodeMapRange")
    nt.links.new(info.outputs["Intercept"], ramp.inputs["Value"])
    ramp.inputs["From Min"].default_value = 0.6
    ramp.inputs["To Max"].default_value = 0.6 if name != "HairFuzz" else 0.85
    ramp.inputs["To Min"].default_value = 0.0 if name != "HairFuzz" else 0.55
    nt.links.new(ramp.outputs["Result"], mix.inputs["Fac"])
    nt.links.new(h.outputs[0], mix.inputs[1])
    nt.links.new(tr.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    return m


def hair_pigment(srgb, age=0.0):
    """Melanin model from an observed hair colour: dark -> high melanin, red-brown -> redness."""
    r, g, b = [float(x) for x in srgb]
    lum = 0.3 * r + 0.59 * g + 0.11 * b
    melanin = float(np.clip(1.02 - 1.25 * lum, 0.05, 0.98))
    redness = float(np.clip((r - b) / max(r, 1e-3) * 0.42, 0.0, 1.0))
    if age >= 0.9:                       # grey / white hair: little melanin
        melanin, redness = min(melanin, 0.18), min(redness, 0.15)
    return melanin, redness


SCALP_MEL = 0.16      # was 0.06
SCALP_RED = 0.6       # was 1.0


def set_hair_colour(tex_dir=None):
    """Hair colour from the texture set (set.json eyebrow colour, age)."""
    info = {}
    if tex_dir and os.path.isfile(os.path.join(tex_dir, "set.json")):
        info = json.load(open(os.path.join(tex_dir, "set.json")))
    brow = np.array(info.get("eyebrow_colour_srgb", [0.30, 0.21, 0.14]))
    age = float(info.get("age_factor", 0.0))
    mel, red = hair_pigment(brow, age)
    # 17 Nov: the short scalp hairs scatter more light than the brows and read lighter and more
    # ginger, so they carry extra melanin and less redness to match the brows on screen.
    spec = {"HairScalp": (min(0.98, mel + SCALP_MEL), red * SCALP_RED), "HairBrows": (mel, red),
            "HairFuzz": (mel * 0.35, red * 0.5)}            # vellus: light, low pigment
    for name, (m_, r_) in spec.items():
        m = bpy.data.materials.get(name)
        if m and m.use_nodes and "HairBSDF" in m.node_tree.nodes:
            n = m.node_tree.nodes["HairBSDF"]
            n.inputs["Melanin"].default_value = m_
            n.inputs["Melanin Redness"].default_value = r_
    # scalp skin under the buzz cut: tinted towards the hair colour (dense dark roots)
    skin = bpy.data.materials.get("Skin")
    if skin and "HAIR_tint" in skin.node_tree.nodes:
        dark = srgb_to_linear(np.clip(brow * (0.55 if age < 0.9 else 1.0), 0, 1))
        skin.node_tree.nodes["HAIR_tint"].inputs["B"].default_value = (*dark, 1.0)


def _scalp_shading(head, head_ob):
    """Skin under the hair: a 'hair_mask' attribute (scalp weight) mixes the skin colour towards
    the hair colour, so the buzz cut reads dense (roots and shadowed hair at the skin)."""
    import hair
    me = head_ob.data
    w = hair.scalp_weight(head).astype(np.float32)
    a = me.attributes.get("hair_mask")
    if a is not None:
        me.attributes.remove(a)
    a = me.attributes.new("hair_mask", "FLOAT", "POINT")
    a.data.foreach_set("value", w)
    nt = bpy.data.materials["Skin"].node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    old_tint = nt.nodes.get("HAIR_tint")
    if old_tint is not None:     # re-run: keep feeding what fed the old tint node
        src = next(l.from_socket for l in nt.links if l.to_socket == old_tint.inputs["A"])
    else:
        src = next(l.from_socket for l in nt.links if l.to_socket == bsdf.inputs["Base Color"])
    for n in [n for n in nt.nodes if n.name.startswith("HAIR_")]:
        nt.nodes.remove(n)
    at = nt.nodes.new("ShaderNodeAttribute")
    at.name = "HAIR_mask"
    at.attribute_type, at.attribute_name = "GEOMETRY", "hair_mask"
    at.location = (-900, 500)
    fac = nt.nodes.new("ShaderNodeMath")
    fac.name = "HAIR_fac"
    fac.operation = "MULTIPLY"
    fac.inputs[1].default_value = 0.62
    fac.location = (-700, 500)
    nt.links.new(at.outputs["Fac"], fac.inputs[0])
    mx = nt.nodes.new("ShaderNodeMix")
    mx.name = "HAIR_tint"
    mx.data_type = "RGBA"
    mx.location = (-500, 500)
    nt.links.new(fac.outputs[0], mx.inputs["Factor"])
    nt.links.new(src, mx.inputs["A"])
    nt.links.new(mx.outputs["Result"], bsdf.inputs["Base Color"])


def _build(name, roots_uv, strands, r0, r1, head_ob, mat):
    old = bpy.data.objects.get(name)
    if old is not None:
        data = old.data
        bpy.data.objects.remove(old, do_unlink=True)
        if data and data.users == 0:
            bpy.data.hair_curves.remove(data)
    N, K, _ = strands.shape
    cv = bpy.data.hair_curves.new(name)
    cv.add_curves([K] * N)
    pts = to_blender(strands).reshape(-1, 3).astype(np.float32)
    cv.position_data.foreach_set("vector", pts.ravel())
    rad = np.linspace(r0, r1, K, dtype=np.float32)[None, :].repeat(N, 0).ravel()
    ra = cv.attributes.get("radius") or cv.attributes.new("radius", "FLOAT", "POINT")
    ra.data.foreach_set("value", rad)
    uva = cv.attributes.get("surface_uv_coordinate") or cv.attributes.new("surface_uv_coordinate", "FLOAT2", "CURVE")
    uva.data.foreach_set("vector", roots_uv.astype(np.float32).ravel())
    cv.surface = head_ob
    cv.surface_uv_map = "UDIM"
    cv.materials.append(mat)
    ob = bpy.data.objects.new(name, cv)
    bpy.context.scene.collection.objects.link(ob)
    ob.parent = head_ob
    ob.matrix_parent_inverse = head_ob.matrix_world.inverted()
    mod = ob.modifiers.new("DeformOnSurface", "NODES")
    mod.node_group = _deform_group()
    return ob


def add(head, head_ob, density=1.0, tex_dir=None):
    import hair
    head_ob.add_rest_position_attribute = True
    alpha = None
    bp = os.path.join(HERE, "textures", "_bake", "brow_alpha.png")
    if os.path.isfile(bp):
        img = bpy.data.images.load(bp, check_existing=True)
        w, h = img.size
        px = np.zeros(w * h * 4, np.float32)
        img.pixels.foreach_get(px)
        alpha = px.reshape(h, w, 4)[::-1, :, 0]          # Blender pixels start bottom-left
        bpy.data.images.remove(img)
    _scalp_shading(head, head_ob)
    gens = {
        "buzz": lambda: hair.buzz_cut(head, n=int(240000 * density), length=0.0058),
        "brows": lambda: hair.eyebrows(head, alpha, n_per=int(900 * density)),
        "fuzz": lambda: hair.peach_fuzz(head, n=int(70000 * density)),
    }
    report = {}
    for name, (gen, r0, r1, mname) in GROUPS.items():
        pos, nrm, uv, strands = gens[gen]()
        _build(name, uv, strands, r0, r1, head_ob, _material(mname))
        report[name] = int(len(pos))
    set_hair_colour(tex_dir)
    scn = bpy.context.scene
    try:
        scn.cycles_curves.shape = "THICK"
        scn.cycles_curves.subdivisions = 2
    except AttributeError:
        pass
    scn.render.hair_type = "STRIP"
    scn.render.hair_subdiv = 1
    return report


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    density = float(argv[argv.index("--density") + 1]) if "--density" in argv else 1.0
    from gnm_model import GNMHead
    js = json.load(open(os.path.join(HERE, "textures", "_bake", "fit.json")))
    head = GNMHead(os.path.join(HERE, "..", "tools", "gnm", "model", "gnm_head.npz"),
                   np.array(js["identity"] if isinstance(js, dict) else js, float))
    cfg = json.loads(bpy.context.scene["track1_config"])
    tex = cfg.get("textures")
    tex = os.path.join(HERE, tex) if tex and not os.path.isabs(tex) else tex
    rep = add(head, bpy.data.objects[cfg["head_mesh"]], density, tex)
    cfg["hair"] = rep
    bpy.context.scene["track1_config"] = json.dumps(cfg)
    print("[hair]", json.dumps(rep), flush=True)
    if "--save" in argv:
        bpy.ops.wm.save_mainfile(compress=True)
        print("[hair] saved", bpy.data.filepath, flush=True)


if __name__ == "__main__":
    main()
