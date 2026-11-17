"""
add_sclera_shading.py - lid contact shading on the eye whites.  Idempotent.

    blender -b scene.blend --python add_sclera_shading.py -- [--save | --out OTHER.blend]
          [--reach 2.5] [--strength 0.55]

Why: a real sclera is never one flat white - it darkens and warms toward the lids, where the
lid margin, lashes and the tear meniscus shade it.  Without that, the eye white reads as chalk.

How: a Geometry Nodes modifier "EyeContact" on GNMHead (after the Armature, before the
Subdivision) stores, per vertex, the live distance to the nearest Skin face in the attribute
"lid_dist".  The EyeInterior_L/R shaders turn it into a contact factor (smoothstep over
`reach`) and multiply the albedo toward a warm shadow tone, *before* polish_look's ScleraTone
curve (so the look presets keep working).  Live geometry -> follows blinks, gaze and squints.
"""
import argparse
import sys

import bpy

ATTR = "lid_dist"
MOD, NG = "EyeContact", "EyeContact_GN"
REACH_M = 0.0035                 # contact shading fades out this far from the lids (16 Nov A/B)
STRENGTH = 0.75                  # 0 = off, 1 = albedo fully replaced by the contact tone at the lid
CONTACT_TONE = (0.42, 0.30, 0.27, 1.0)   # warm shadow tone (linear) the albedo is pulled toward


def _node_group(skin_mat):
    ng = bpy.data.node_groups.get(NG)
    if ng:
        bpy.data.node_groups.remove(ng)
    ng = bpy.data.node_groups.new(NG, "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nt = ng.nodes
    gin, gout = nt.new("NodeGroupInput"), nt.new("NodeGroupOutput")
    ms = nt.new("GeometryNodeMaterialSelection")
    ms.inputs["Material"].default_value = skin_mat
    sep = nt.new("GeometryNodeSeparateGeometry")
    sep.domain = "FACE"
    ng.links.new(gin.outputs[0], sep.inputs["Geometry"])
    ng.links.new(ms.outputs[0], sep.inputs["Selection"])
    px = nt.new("GeometryNodeProximity")
    px.target_element = "FACES"
    ng.links.new(sep.outputs["Selection"], px.inputs[0])
    pos = nt.new("GeometryNodeInputPosition")
    ng.links.new(pos.outputs[0], px.inputs["Sample Position"] if "Sample Position" in px.inputs else px.inputs[1])
    st = nt.new("GeometryNodeStoreNamedAttribute")
    st.data_type, st.domain = "FLOAT", "POINT"
    st.inputs["Name"].default_value = ATTR
    ng.links.new(gin.outputs[0], st.inputs["Geometry"])
    ng.links.new(px.outputs["Distance"], st.inputs["Value"])
    ng.links.new(st.outputs[0], gout.inputs[0])
    for i, n in enumerate((gin, ms, sep, pos, px, st, gout)):
        n.location = (i * 200, 0)
    return ng


def _shader(m, reach, strength):
    nt = m.node_tree
    for n in [n for n in nt.nodes if n.name.startswith("EC_")]:
        nt.nodes.remove(n)
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    link = next(l for l in nt.links if l.to_socket == bsdf.inputs["Base Color"])
    # keep polish_look's ScleraTone last: splice in front of it when present
    if link.from_node.type == "CURVE_RGB" and link.from_node.name == "ScleraTone":
        tone = link.from_node
        link = next(l for l in nt.links if l.to_socket == tone.inputs["Color"])
    src, dst = link.from_socket, link.to_socket
    nt.links.remove(link)
    x, y = dst.node.location

    def node(kind, name, dx, dy):
        n = nt.nodes.new(kind)
        n.name = n.label = name
        n.location = (x + dx, y + dy)
        return n
    at = node("ShaderNodeAttribute", "EC_dist", -700, -250)
    at.attribute_name = ATTR
    mr = node("ShaderNodeMapRange", "EC_range", -500, -250)
    mr.interpolation_type = "SMOOTHSTEP"
    mr.inputs["From Min"].default_value = 0.0
    mr.inputs["From Max"].default_value = reach
    mr.inputs["To Min"].default_value = 1.0 - strength     # weight of the albedo at the lid
    mr.inputs["To Max"].default_value = 1.0
    nt.links.new(at.outputs["Fac"], mr.inputs["Value"])
    mix = node("ShaderNodeMix", "EC_mix", -300, -100)
    mix.data_type = "RGBA"
    mix.blend_type = "MIX"
    nt.links.new(mr.outputs["Result"], mix.inputs[0])            # Factor (float)
    # tone * albedo at the lid ... albedo away from it
    mul = node("ShaderNodeMix", "EC_tone", -450, 0)
    mul.data_type = "RGBA"
    mul.blend_type = "MULTIPLY"
    # ShaderNodeMix sockets by index (names repeat per data type): 0 Factor, 6/7 colour A/B, out 2
    mul.inputs[0].default_value = 1.0
    nt.links.new(src, mul.inputs[6])
    mul.inputs[7].default_value = CONTACT_TONE
    nt.links.new(mul.outputs[2], mix.inputs[6])
    nt.links.new(src, mix.inputs[7])
    nt.links.new(mix.outputs[2], dst)


def add(reach=REACH_M, strength=STRENGTH):
    import json
    cfg = json.loads(bpy.context.scene["track1_config"])
    ob = bpy.data.objects[cfg["head_mesh"]]
    old = ob.modifiers.get(MOD)
    if old:
        ob.modifiers.remove(old)
    mod = ob.modifiers.new(MOD, "NODES")
    mod.node_group = _node_group(bpy.data.materials["Skin"])
    # after the Armature, before the Subdivision
    arm_i = next(i for i, m in enumerate(ob.modifiers) if m.type == "ARMATURE")
    ob.modifiers.move(len(ob.modifiers) - 1, arm_i + 1)
    for s in "LR":
        _shader(bpy.data.materials["EyeInterior_" + s], reach, strength)
    cfg["sclera_contact"] = {"reach_mm": reach * 1e3, "strength": strength}
    bpy.context.scene["track1_config"] = json.dumps(cfg)
    return {"head_modifiers": [m.name for m in ob.modifiers]}


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--save", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--reach", type=float, default=REACH_M * 1e3, help="mm")
    p.add_argument("--strength", type=float, default=STRENGTH)
    a = p.parse_args(argv)
    print("[sclera]", add(a.reach * 1e-3, a.strength), flush=True)
    if a.out:
        bpy.ops.wm.save_as_mainfile(filepath=a.out, compress=True)
        print("[sclera] saved", a.out, flush=True)
    elif a.save:
        bpy.ops.wm.save_mainfile(compress=True)
        print("[sclera] saved", bpy.data.filepath, flush=True)
