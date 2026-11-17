"""
add_wrinkles.py - dynamic expression wrinkles on the GNM skin (forehead lines, glabella "11"
lines, crow's feet), driven live by the brow / squint / smile shape keys.  Idempotent.

    blender -b scene.blend --python add_wrinkles.py -- [--save]

How: two mesh attributes (rest position, zone weights from gnm_head/wrinkle_masks.py) feed a
procedural crease pattern in the Skin shader (wavy lines in the right orientation per zone,
computed from the rest shape so they stick to the skin as the head moves).  The crease depth is
a Bump on top of the pore normal map plus a slight darkening in the groove; its strength per
zone is a driver on the corresponding shape keys, so it works in EEVEE, Cycles and on the
Timeline.  build_scene_gnm.py calls add() as well.
"""
import json
import os
import sys

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "gnm_head"))

DEPTH_M = 0.00036          # bump distance (crease depth scale)
ZONES = {   # name: (mask channel, driver expression over key values, key names)
    "WR_forehead": (0, "(a+b)/2", ("brow_up_L", "brow_up_R")),
    "WR_glabella": (1, "(a+b)/2", ("brow_down_L", "brow_down_R")),
    "WR_crow": (2, "max((a+b)/2, 0.8*c)", ("squint_L", "squint_R", "smile")),
}


def to_blender(p):
    p = np.asarray(p, dtype=np.float64)
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)


def _attr(me, name, kind, data):
    a = me.attributes.get(name)
    if a is not None:
        me.attributes.remove(a)
    a = me.attributes.new(name, kind, "POINT")
    a.data.foreach_set("vector" if kind == "FLOAT_VECTOR" else "color", data.astype(np.float32).ravel())


def _math(nt, op, a, b=None, x=0, y=0, clamp=False):
    n = nt.nodes.new("ShaderNodeMath")
    n.name = n.label = "WR_math"
    n.operation = op
    n.use_clamp = clamp
    n.location = (x, y)
    for i, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = v
        else:
            nt.links.new(v, n.inputs[i])
    return n.outputs[0]


def _crease(nt, coord, offset, spacing, sharp, noise, wobble, breakup, x, y):
    """(1-|sin(pi*(coord - offset + wobble*noise)/spacing)|)^sharp : soft lines every `spacing`,
    faded in and out along their length by `breakup` (a low-frequency noise, 0..1)."""
    c = _math(nt, "SUBTRACT", coord, offset, x, y)
    c = _math(nt, "ADD", c, _math(nt, "MULTIPLY", noise, wobble, x, y - 60), x + 160, y)
    c = _math(nt, "MULTIPLY", c, np.pi / spacing, x + 320, y)
    c = _math(nt, "SINE", c, None, x + 480, y)
    c = _math(nt, "ABSOLUTE", c, None, x + 640, y)
    c = _math(nt, "SUBTRACT", 1.0, c, x + 800, y)
    c = _math(nt, "POWER", c, sharp, x + 960, y)
    return _math(nt, "MULTIPLY", c, breakup, x + 1120, y)


def add(head, head_ob):
    import wrinkle_masks
    me = head_ob.data
    W = wrinkle_masks.masks(head)
    _attr(me, "rest_pos", "FLOAT_VECTOR", to_blender(head.bind))
    col = np.zeros((head.V, 4))
    col[:, 0], col[:, 1], col[:, 2], col[:, 3] = W["forehead"], W["glabella"], W["crow"], 1.0
    _attr(me, "wrinkle_mask", "FLOAT_COLOR", col)
    cz = float(np.mean([W["canthi"][s][1] for s in "LR"]))           # GNM y -> Blender z
    cx = float(np.mean([abs(W["canthi"][s][0]) for s in "LR"]))

    m = bpy.data.materials["Skin"]
    nt = m.node_tree
    for n in [n for n in nt.nodes if n.name.startswith("WR")]:
        nt.nodes.remove(n)
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    nmap = next(n for n in nt.nodes if n.type == "NORMAL_MAP")
    base_link = next(l for l in nt.links if l.to_socket == bsdf.inputs["Base Color"])
    base_src = base_link.from_socket
    if base_link.from_node.name == "WR_darken":                      # re-run: find the original
        base_src = next(l.from_socket for l in nt.links if l.to_node.name == "WR_darken"
                        and l.to_socket.name in ("A", "Color1"))

    def node(kind, name, x, y):
        n = nt.nodes.new(kind)
        n.name = n.label = name
        n.location = (x, y)
        return n

    X0, Y0 = -2600, -1400
    rp = node("ShaderNodeAttribute", "WR_rest", X0, Y0)
    rp.attribute_type, rp.attribute_name = "GEOMETRY", "rest_pos"
    wm = node("ShaderNodeAttribute", "WR_mask", X0, Y0 - 250)
    wm.attribute_type, wm.attribute_name = "GEOMETRY", "wrinkle_mask"
    sep = node("ShaderNodeSeparateXYZ", "WR_xyz", X0 + 200, Y0)
    nt.links.new(rp.outputs["Vector"], sep.inputs[0])
    msk = node("ShaderNodeSeparateColor", "WR_msk", X0 + 200, Y0 - 250)
    nt.links.new(wm.outputs["Color"], msk.inputs[0])
    nz = node("ShaderNodeTexNoise", "WR_noise", X0 + 200, Y0 - 500)
    nz.inputs["Scale"].default_value = 140.0
    nz.inputs["Detail"].default_value = 2.0
    nt.links.new(rp.outputs["Vector"], nz.inputs["Vector"])
    noise = _math(nt, "SUBTRACT", nz.outputs["Fac"], 0.5, X0 + 400, Y0 - 500)
    nb = node("ShaderNodeTexNoise", "WR_breakup", X0 + 200, Y0 - 700)
    nb.inputs["Scale"].default_value = 45.0
    nb.inputs["Detail"].default_value = 1.0
    nt.links.new(rp.outputs["Vector"], nb.inputs["Vector"])
    bk = node("ShaderNodeMapRange", "WR_breakup_range", X0 + 400, Y0 - 700)
    bk.inputs["From Min"].default_value, bk.inputs["From Max"].default_value = 0.35, 0.65
    bk.inputs["To Min"].default_value, bk.inputs["To Max"].default_value = 0.15, 1.0
    nt.links.new(nb.outputs["Fac"], bk.inputs["Value"])
    breakup = bk.outputs["Result"]
    absx = _math(nt, "ABSOLUTE", sep.outputs["X"], None, X0 + 400, Y0)
    # forehead: 3-4 soft horizontal furrows ~1 cm apart, wavy and broken
    f = _crease(nt, sep.outputs["Z"], 0.0, 0.0105, 2.6, noise, 0.0045, breakup, X0 + 600, Y0 + 300)
    # glabella: one furrow either side of the midline at |x| ~ 4 mm
    g = _crease(nt, absx, 0.0040, 0.0200, 4.0, noise, 0.0010, breakup, X0 + 600, Y0)
    # crow's feet: a few lines fanning from the outer eye corner (angle about the canthus)
    dx = _math(nt, "SUBTRACT", absx, cx, X0 + 400, Y0 - 300)
    dz = _math(nt, "SUBTRACT", sep.outputs["Z"], cz, X0 + 400, Y0 - 360)
    ang = _math(nt, "ARCTAN2", dz, dx, X0 + 550, Y0 - 330)
    c = _crease(nt, ang, 0.08, 0.36, 3.0, noise, 0.12, breakup, X0 + 600, Y0 - 300)

    strengths = {}
    for i, (name, (ch, expr, keys)) in enumerate(ZONES.items()):
        v = node("ShaderNodeValue", name, X0 + 1700, Y0 + 300 - 300 * i)
        fc = v.outputs[0].driver_add("default_value")
        d = fc.driver
        d.type = "SCRIPTED"
        d.expression = expr
        for k, letter in zip(keys, "abc"):
            var = d.variables.new()
            var.name = letter
            var.type = "SINGLE_PROP"
            var.targets[0].id_type = "KEY"
            var.targets[0].id = head_ob.data.shape_keys
            var.targets[0].data_path = 'key_blocks["%s"].value' % k
        strengths[name] = v.outputs[0]
    chans = [msk.outputs["Red"], msk.outputs["Green"], msk.outputs["Blue"]]
    total = None
    for (name, (ch, _, _)), pat in zip(ZONES.items(), (f, g, c)):
        t = _math(nt, "MULTIPLY", pat, chans[ch], X0 + 1900, Y0 - 300 * ch)
        t = _math(nt, "MULTIPLY", t, strengths[name], X0 + 2050, Y0 - 300 * ch)
        total = t if total is None else _math(nt, "ADD", total, t, X0 + 2200, Y0 - 300 * ch)
    total = _math(nt, "MINIMUM", total, 1.0, X0 + 2350, Y0)

    bump = node("ShaderNodeBump", "WR_bump", X0 + 2550, Y0)
    bump.invert = True                                           # creases are grooves
    bump.inputs["Strength"].default_value = 1.0
    bump.inputs["Distance"].default_value = DEPTH_M
    nt.links.new(total, bump.inputs["Height"])
    nt.links.new(nmap.outputs["Normal"], bump.inputs["Normal"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    dk = node("ShaderNodeMix", "WR_darken", X0 + 2550, Y0 + 300)
    dk.data_type, dk.blend_type = "RGBA", "MULTIPLY"
    nt.links.new(_math(nt, "MULTIPLY", total, 0.28, X0 + 2350, Y0 + 300), dk.inputs["Factor"])
    nt.links.new(base_src, dk.inputs["A"])
    dk.inputs["B"].default_value = (0.70, 0.56, 0.52, 1.0)          # slightly darker, warmer groove
    nt.links.new(dk.outputs["Result"], bsdf.inputs["Base Color"])
    return {"wrinkle_canthus_x_m": round(cx, 4), "wrinkle_canthus_z_m": round(cz, 4),
            "wrinkle_verts": {k: int((W[k] > 0.05).sum()) for k in ("forehead", "glabella", "crow")}}


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    from gnm_model import GNMHead
    js = json.load(open(os.path.join(HERE, "textures", "_bake", "fit.json")))
    head = GNMHead(os.path.join(HERE, "..", "tools", "gnm", "model", "gnm_head.npz"),
                   np.array(js["identity"] if isinstance(js, dict) else js, float))
    cfg = json.loads(bpy.context.scene["track1_config"])
    rep = add(head, bpy.data.objects[cfg["head_mesh"]])
    cfg.setdefault("build_report", {}).update(rep)
    cfg["wrinkles"] = list(ZONES)
    bpy.context.scene["track1_config"] = json.dumps(cfg)
    print("[wrinkles]", json.dumps(rep), flush=True)
    if "--save" in argv:
        bpy.ops.wm.save_mainfile(compress=True)
        print("[wrinkles] saved", bpy.data.filepath, flush=True)


if __name__ == "__main__":
    main()
