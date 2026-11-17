"""
add_tearline.py - lower-lid tear meniscus ("tear line") for both eyes.  Idempotent.

    blender -b scene.blend --python add_tearline.py -- [--save | --out OTHER.blend]
          [--h-eye 0.32] [--h-lid 0.22]

What: a thin bead of tear film along the line where the lower lid meets the eyeball - the wet
glint line that makes a CG eye read as alive.

Why not at the margin vertex: GNM's lower lid drapes almost tangentially onto the eyeball; its
margin vertex (where the skin folds into the socket lining) lies ~0.1-0.3 mm *inside* the eye
surface after subdivision, and the lid skin stays within ~0.05-0.15 mm of the eye for 2-4 mm.
The visible lid edge is where the lid surface crosses the eye surface, so the bead sits there.

How it follows the face: the strip object "TearLine" rides on the head bone and carries, per
vertex, the GNMHead vertex ids it hangs from (lower-margin vertex P, the next two lid-skin
rings O1, O2 going away from the eye, the matching upper-margin vertex U) plus its row/column
parameters.  A Geometry Nodes modifier reads the *evaluated* head every frame (armature + every
shape key + subdivision - subdivision keeps original vertex ids first) and rebuilds the strip:
    R0  eye surface radius in P's direction (nearest point on the live sclera + cornea faces)
    X   where the lid polyline P-O1-O2 crosses radius R0  (the visible lid edge)
    across the strip (row t), the direction swings from h_lid down onto the lid to h_eye up the
    eye; the radius follows the lid (chord X->lid point) or the eye surface, plus a bead
    THICK*(1-u^2) peaked at X, sinking TUCK under the surfaces at both edges.
Sizes shrink as the lids close (blink, squint) and taper to zero at the canthi.
The material is a thin wet film (Fresnel-weighted glossy over transparent), so it adds the
specular glint without refracting the eye behind it, casts no shadow, and works in EEVEE too.
"""
import argparse
import json
import sys

import bpy
import numpy as np
from mathutils import Matrix

H_EYE_M = 0.00035     # tear film reach up the eyeball from the lid/eye crossing
H_LID_M = 0.00025     # ... and down over the lid's edge
THICK_M = 0.00005     # bead thickness at the crossing (where the tear collects)
TUCK_M = 0.000012     # the strip's outer edges sink just under the surfaces (no visible rim)
SUB = 4               # strip columns per margin segment
ROWS = (0.0, 0.08, 0.18, 0.3, 0.42, 0.52, 0.62, 0.72, 0.82, 0.91, 1.0)
OPEN_FULL = 0.60      # lid opening |U-P| / rest opening at which the tear line has full size
OPEN_ZERO = 0.25      # ... and at which it has vanished (blink; the upper lid would cover it)
OBJ, MAT, NG = "TearLine", "TearFilm", "TearLine_GN"
IDS = ("i0", "i1", "a0", "a1", "b0", "b1", "u0", "u1")   # P, O1, O2, U vertex-id pairs


# ----------------------------------------------------------------------------- rest data
def _columns(head_me, cfg, s):
    """Per-column vertex ids (P, O1, O2, U pairs), fraction and taper for eye s."""
    co = np.array([v.co[:] for v in head_me.vertices])
    c = np.array(cfg["center_" + s])
    lo = np.array(cfg["lid_margins"][s]["lower"])
    up = np.array(cfg["lid_margins"][s]["upper"])
    lo = lo[np.argsort(co[lo, 0])]
    margin = set(lo.tolist()) | set(up.tolist())
    nbr = [[] for _ in range(len(co))]
    for e in head_me.edges:
        a, b = e.vertices
        nbr[a].append(b)
        nbr[b].append(a)
    r = lambda i: float(np.linalg.norm(co[i] - c))  # noqa: E731
    ring1, ring2 = [], []
    for i in lo.tolist():
        chain = [i]
        for _ in range(2):   # walk out over the lid skin, away from the margin
            cur = chain[-1]
            cand = [j for j in nbr[cur] if j not in margin and j not in chain]
            cand = [j for j in cand if r(j) >= r(cur) - 1e-4] or cand
            chain.append(max(cand, key=lambda j: float(np.linalg.norm(co[j] - co[i]))))
        ring1.append(chain[1])
        ring2.append(chain[2])
    # upper-margin partner: nearest in x (both lids span the same canthus-to-canthus range)
    upper = [int(up[np.argmin(np.abs(co[up, 0] - co[i, 0]))]) for i in lo.tolist()]
    n = len(lo)
    cols = []
    for k in range(n - 1):
        for j in range(SUB):
            cols.append((k, k + 1, j / SUB))
    cols.append((n - 2, n - 1, 1.0))
    # arc-length parameter for the taper
    pts = np.array([co[lo[a]] * (1 - f) + co[lo[b]] * f for a, b, f in cols])
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    u = arc / arc[-1]
    taper = np.clip(np.minimum(u / 0.10, (1 - u) / 0.14), 0, 1)
    taper = taper * taper * (3 - 2 * taper)
    out = []
    for (a, b, f), tp in zip(cols, taper):
        out.append(dict(i0=int(lo[a]), i1=int(lo[b]), a0=ring1[a], a1=ring1[b], b0=ring2[a], b1=ring2[b],
                        u0=upper[a], u1=upper[b], frac=float(f), taper=float(tp),
                        gap0=float(np.linalg.norm((co[upper[a]] - co[lo[a]]) * (1 - f) + (co[upper[b]] - co[lo[b]]) * f))))
    return out, c


def _build_mesh(head_me, cfg):
    verts, faces, attrs = [], [], {k: [] for k in IDS + ("frac", "taper", "gap0", "t", "eye_c")}
    co = np.array([v.co[:] for v in head_me.vertices])
    for s in "LR":
        cols, c = _columns(head_me, cfg, s)
        base = len(verts)
        R = len(ROWS)
        for ci, col in enumerate(cols):
            p = co[col["i0"]] * (1 - col["frac"]) + co[col["i1"]] * col["frac"]
            for t in ROWS:
                verts.append(p)        # placeholder position; the GN modifier sets the real one
                for k in IDS + ("frac", "taper", "gap0"):
                    attrs[k].append(col[k])
                attrs["t"].append(t)
                attrs["eye_c"].append(c)
            if ci:
                for ri in range(R - 1):
                    a = base + (ci - 1) * R + ri
                    b = base + ci * R + ri
                    faces.append((a, b, b + 1, a + 1))
    me = bpy.data.meshes.new(OBJ)
    me.from_pydata([tuple(v) for v in verts], [], faces)
    me.update()
    for k, v in attrs.items():
        if k == "eye_c":
            a = me.attributes.new(k, "FLOAT_VECTOR", "POINT")
            a.data.foreach_set("vector", np.asarray(v, np.float32).ravel())
        elif k in ("frac", "taper", "gap0", "t"):
            a = me.attributes.new(k, "FLOAT", "POINT")
            a.data.foreach_set("value", np.asarray(v, np.float32))
        else:
            a = me.attributes.new(k, "INT", "POINT")
            a.data.foreach_set("value", np.asarray(v, np.int32))
    return me


# ----------------------------------------------------------------------------- material
def _material():
    m = bpy.data.materials.get(MAT) or bpy.data.materials.new(MAT)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)
    fr = nt.nodes.new("ShaderNodeFresnel")
    fr.inputs["IOR"].default_value = 1.336
    fr.location = (0, 200)
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    tr.location = (0, 0)
    gl = nt.nodes.new("ShaderNodeBsdfGlossy")
    gl.inputs["Roughness"].default_value = 0.06
    gl.location = (0, -150)
    # water F0 is ~2 %; boost a little so the glint survives the low-key grade
    boost = nt.nodes.new("ShaderNodeMath")
    boost.operation = "MULTIPLY_ADD"
    boost.use_clamp = True
    boost.inputs[1].default_value = 1.6
    boost.inputs[2].default_value = 0.03
    boost.location = (200, 200)
    nt.links.new(fr.outputs["Fac"], boost.inputs[0])
    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.location = (400, 0)
    nt.links.new(boost.outputs[0], mix.inputs["Fac"])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(gl.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    # EEVEE: blended transparency
    for attr, val in (("surface_render_method", "BLENDED"), ("blend_method", "BLEND"),
                      ("use_transparency_overlap", True), ("show_transparent_back", True)):
        if hasattr(m, attr):
            try:
                setattr(m, attr, val)
            except Exception:
                pass
    return m


# ----------------------------------------------------------------------------- geometry nodes
class _G:
    def __init__(self, ng):
        self.ng, self.nt, self.x = ng, ng.nodes, 0

    def node(self, kind, **props):
        n = self.nt.new(kind)
        for k, v in props.items():
            setattr(n, k, v)
        n.location = (self.x, 0)
        self.x += 40
        return n

    def link(self, a, b):
        self.ng.links.new(a, b)

    def _in(self, n, i, v):
        if v is None:
            return
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = v
        elif isinstance(v, tuple):
            n.inputs[i].default_value = v
        else:
            self.link(v, n.inputs[i])

    def attr(self, name, kind):
        n = self.node("GeometryNodeInputNamedAttribute", data_type=kind)
        n.inputs["Name"].default_value = name
        return n.outputs["Attribute"]

    def vm(self, op, a, b=None, scale=None):
        n = self.node("ShaderNodeVectorMath", operation=op)
        self._in(n, 0, a)
        self._in(n, 1, b)
        if scale is not None:
            self._in(n, n.inputs.find("Scale"), scale)
        return n.outputs["Value" if op in ("DOT_PRODUCT", "LENGTH", "DISTANCE") else "Vector"]

    def fm(self, op, a, b=None, c=None, clamp=False):
        n = self.node("ShaderNodeMath", operation=op, use_clamp=clamp)
        self._in(n, 0, a)
        self._in(n, 1, b)
        self._in(n, 2, c)
        return n.outputs[0]

    def lerp(self, a, b, f):
        return self.vm("ADD", a, self.vm("SCALE", self.vm("SUBTRACT", b, a), scale=f))

    def sample(self, geo, idx):
        n = self.node("GeometryNodeSampleIndex", data_type="FLOAT_VECTOR", domain="POINT")
        self.link(geo, n.inputs["Geometry"])
        self.link(self.pos, n.inputs["Value"])
        self.link(idx, n.inputs["Index"])
        return n.outputs["Value"]

    def smooth01(self, v, lo, hi):
        n = self.node("ShaderNodeMapRange", data_type="FLOAT", interpolation_type="SMOOTHSTEP", clamp=True)
        self._in(n, 0, v)
        n.inputs[1].default_value, n.inputs[2].default_value = lo, hi
        n.inputs[3].default_value, n.inputs[4].default_value = 0.0, 1.0
        return n.outputs[0]


def _node_group(head_ob, h_eye, h_lid, thick=THICK_M):
    ng = bpy.data.node_groups.get(NG)
    if ng:
        bpy.data.node_groups.remove(ng)
    ng = bpy.data.node_groups.new(NG, "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    g = _G(ng)
    gin = g.node("NodeGroupInput")
    gout = g.node("NodeGroupOutput")
    g.pos = g.node("GeometryNodeInputPosition").outputs[0]

    oi = g.node("GeometryNodeObjectInfo", transform_space="RELATIVE")
    oi.inputs["Object"].default_value = head_ob
    H = oi.outputs["Geometry"]

    # eye outer surface = live sclera + cornea faces of both eyes (they rotate with the gaze)
    sel = None
    for name in ("EyeInterior_L", "Cornea_L", "EyeInterior_R", "Cornea_R"):
        n = g.node("GeometryNodeMaterialSelection")
        n.inputs["Material"].default_value = bpy.data.materials[name]
        if sel is None:
            sel = n.outputs[0]
        else:
            o = g.node("FunctionNodeBooleanMath", operation="OR")
            g.link(sel, o.inputs[0])
            g.link(n.outputs[0], o.inputs[1])
            sel = o.outputs[0]
    sep = g.node("GeometryNodeSeparateGeometry", domain="FACE")
    g.link(H, sep.inputs["Geometry"])
    g.link(sel, sep.inputs["Selection"])
    eye = sep.outputs["Selection"]

    ia = {k: g.attr(k, "INT") for k in IDS}
    frac, taper, t = g.attr("frac", "FLOAT"), g.attr("taper", "FLOAT"), g.attr("t", "FLOAT")
    c = g.attr("eye_c", "FLOAT_VECTOR")

    P = g.lerp(g.sample(H, ia["i0"]), g.sample(H, ia["i1"]), frac)
    O1 = g.lerp(g.sample(H, ia["a0"]), g.sample(H, ia["a1"]), frac)
    O2 = g.lerp(g.sample(H, ia["b0"]), g.sample(H, ia["b1"]), frac)
    U = g.lerp(g.sample(H, ia["u0"]), g.sample(H, ia["u1"]), frac)

    def radius(p):
        return g.vm("LENGTH", g.vm("SUBTRACT", p, c))

    def eye_r(d):
        """Outer eye radius along unit direction d (nearest eye-surface point seen from outside)."""
        q = g.vm("ADD", c, g.vm("SCALE", d, scale=0.0175))
        px = g.node("GeometryNodeProximity", target_element="FACES")
        g.link(eye, px.inputs[0])
        g.link(q, px.inputs["Sample Position"] if "Sample Position" in px.inputs else px.inputs[1])
        return radius(px.outputs["Position"])

    def frac_to(r_a, r_b, r_goal):
        return g.fm("DIVIDE", g.fm("SUBTRACT", r_goal, r_a), g.fm("SUBTRACT", r_b, r_a), clamp=True)

    dP = g.vm("NORMALIZE", g.vm("SUBTRACT", P, c))
    R0 = eye_r(dP)
    rP, r1, r2 = radius(P), radius(O1), radius(O2)
    # X: where the lid polyline P -> O1 -> O2 rises through the eye surface (the visible lid edge)
    X1 = g.lerp(P, O1, frac_to(rP, r1, R0))
    X2 = g.lerp(O1, O2, frac_to(r1, r2, R0))
    use2 = g.fm("LESS_THAN", r1, R0)
    X = g.lerp(X1, X2, use2)
    lid_dir = g.vm("NORMALIZE", g.lerp(g.vm("SUBTRACT", O1, P), g.vm("SUBTRACT", O2, O1), use2))
    dX = g.vm("NORMALIZE", g.vm("SUBTRACT", X, c))
    # size: 0 when the lids close (blink, squint), times the canthus taper
    opening = g.fm("DIVIDE", g.vm("LENGTH", g.vm("SUBTRACT", U, P)), g.attr("gap0", "FLOAT"))
    hs = g.fm("MULTIPLY", g.smooth01(opening, OPEN_ZERO, OPEN_FULL), taper)
    # tangent at X toward the upper lid
    uc = g.vm("SUBTRACT", U, c)
    tup = g.vm("NORMALIZE", g.vm("SUBTRACT", uc, g.vm("SCALE", dX, scale=g.vm("DOT_PRODUCT", uc, dX))))
    # across the strip: t=0 h_lid down over the lid ... t0 at X ... t=1 h_eye up the eye
    t0 = h_lid / (h_lid + h_eye)
    arc = g.fm("MULTIPLY", hs, g.fm("SUBTRACT", g.fm("MULTIPLY", t, h_lid + h_eye), h_lid))  # metres, signed
    d = g.vm("NORMALIZE", g.vm("ADD", dX, g.vm("SCALE", tup, scale=g.fm("DIVIDE", arc, R0))))
    # surface radius under the strip: lid chord (X -> lid point) below X, live eye surface above
    B = g.vm("ADD", X, g.vm("SCALE", lid_dir, scale=g.fm("MULTIPLY", hs, h_lid)))
    lid_u = g.fm("DIVIDE", g.fm("SUBTRACT", t0, t), t0, clamp=True)            # 1 at t=0, 0 at X
    closing = g.fm("MINIMUM", g.fm("MULTIPLY", hs, 4.0), 1.0)                     # 0 when closed
    base_lid = g.fm("ADD", R0, g.fm("MULTIPLY", g.fm("MULTIPLY", g.fm("SUBTRACT", radius(B), R0), lid_u), closing))
    is_lid = g.fm("LESS_THAN", t, t0)
    base = g.fm("ADD", eye_r(d), g.fm("MULTIPLY", is_lid, g.fm("SUBTRACT", base_lid, eye_r(d))))
    # bead profile: THICK at X, -TUCK at both edges
    eye_u = g.fm("DIVIDE", g.fm("SUBTRACT", t, t0), 1.0 - t0, clamp=True)
    u = g.fm("ADD", lid_u, eye_u)                                                 # 0 at X, 1 at edges
    th = g.fm("SUBTRACT", g.fm("MULTIPLY", g.fm("SUBTRACT", 1.0, g.fm("MULTIPLY", u, u)), thick + TUCK_M), TUCK_M)
    th = g.fm("MULTIPLY", th, closing)
    pos = g.vm("ADD", c, g.vm("SCALE", d, scale=g.fm("ADD", base, th)))

    sp = g.node("GeometryNodeSetPosition")
    g.link(gin.outputs[0], sp.inputs["Geometry"])
    g.link(pos, sp.inputs["Position"])
    sm = g.node("GeometryNodeSetShadeSmooth")
    g.link(sp.outputs[0], sm.inputs["Geometry"])
    mat = g.node("GeometryNodeSetMaterial")
    mat.inputs["Material"].default_value = bpy.data.materials[MAT]
    g.link(sm.outputs[0], mat.inputs["Geometry"])
    g.link(mat.outputs[0], gout.inputs[0])
    return ng


# ----------------------------------------------------------------------------- wet lid rims
WET_ATTR = "tear_wet"
WET_REACH_M = 0.0012   # wet band: 1 at the lid margins, 0 this far out over the skin
WET_STRENGTH = 0.6     # peak glossy weight (times Fresnel) on the rim
WET_ROUGH = 0.10       # roughness of the wet layer


def add_wet_rims(head_ob, cfg, reach=WET_REACH_M, strength=WET_STRENGTH, rough=WET_ROUGH):
    """Moist lid rims: a 'tear_wet' mask on GNMHead (from both lid margins) drives an extra
    Fresnel-weighted glossy layer added after the Skin BSDF.  Lives at the shader output, so
    polish_look (which rewires roughness / coat on the Principled BSDF) leaves it alone."""
    from mathutils import kdtree
    me = head_ob.data
    co = np.array([v.co[:] for v in me.vertices])
    ids = sorted({i for s in "LR" for lid in ("upper", "lower") for i in cfg["lid_margins"][s][lid]})
    kd = kdtree.KDTree(len(ids))
    for k, i in enumerate(ids):
        kd.insert(co[i], k)
    kd.balance()
    w = np.zeros(len(co), np.float32)
    for s in "LR":
        c = np.array(cfg["center_" + s])
        near = np.where(np.linalg.norm(co - c, axis=1) < 0.03)[0]
        for i in near:
            d = kd.find(co[i])[2]
            x = max(0.0, 1.0 - d / reach)
            w[i] = max(w[i], x * x * (3 - 2 * x))
    a = me.attributes.get(WET_ATTR)
    if a is not None:
        me.attributes.remove(a)
    a = me.attributes.new(WET_ATTR, "FLOAT", "POINT")
    a.data.foreach_set("value", w)

    m = bpy.data.materials["Skin"]
    nt = m.node_tree
    for n in [n for n in nt.nodes if n.name.startswith("WET_")]:
        nt.nodes.remove(n)
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output)
    link = next(l for l in nt.links if l.to_socket == out.inputs["Surface"])
    src = link.from_socket
    nt.links.remove(link)
    x, y = out.location

    def node(kind, name, dx, dy):
        n = nt.nodes.new(kind)
        n.name = n.label = name
        n.location = (x + dx, y + dy)
        return n
    at = node("ShaderNodeAttribute", "WET_mask", -700, -300)
    at.attribute_name = WET_ATTR
    lw = node("ShaderNodeLayerWeight", "WET_fresnel", -700, -450)
    lw.inputs["Blend"].default_value = 0.35
    m1 = node("ShaderNodeMath", "WET_w", -500, -350)
    m1.operation = "MULTIPLY"
    nt.links.new(at.outputs["Fac"], m1.inputs[0])
    nt.links.new(lw.outputs["Fresnel"], m1.inputs[1])
    m2 = node("ShaderNodeMath", "WET_strength", -350, -350)
    m2.operation = "MULTIPLY"
    m2.inputs[1].default_value = strength
    nt.links.new(m1.outputs[0], m2.inputs[0])
    gl = node("ShaderNodeBsdfGlossy", "WET_gloss", -200, -350)
    gl.inputs["Roughness"].default_value = rough
    nt.links.new(m2.outputs[0], gl.inputs["Color"])
    add = node("ShaderNodeAddShader", "WET_add", -120, -100)
    nt.links.new(src, add.inputs[0])
    nt.links.new(gl.outputs[0], add.inputs[1])
    nt.links.new(add.outputs[0], out.inputs["Surface"])
    return {"wet_rim_verts": int((w > 0).sum())}


# ----------------------------------------------------------------------------- entry points
def add(h_eye=H_EYE_M, h_lid=H_LID_M, thick=THICK_M, wet=False):
    sc = bpy.context.scene
    cfg = json.loads(sc["track1_config"])
    head_ob = bpy.data.objects[cfg["head_mesh"]]
    arm = bpy.data.objects[cfg["armature"]]
    old = bpy.data.objects.get(OBJ)
    if old:
        me_old = old.data
        bpy.data.objects.remove(old, do_unlink=True)
        bpy.data.meshes.remove(me_old)
    me = _build_mesh(head_ob.data, cfg)
    ob = bpy.data.objects.new(OBJ, me)
    sc.collection.objects.link(ob)
    # ride on the head bone (same convention as Eyelashes): rest pose -> identity world matrix,
    # so the strip's local frame is the head's rest frame and the eye centres are constants.
    hb = arm.data.bones["head"]
    ob.parent = arm
    ob.parent_type = "BONE"
    ob.parent_bone = "head"
    ob.matrix_parent_inverse = (arm.matrix_world @ hb.matrix_local
                                @ Matrix.Translation((0, hb.length, 0))).inverted()
    me.materials.append(_material())
    mod = ob.modifiers.new("TearLine", "NODES")
    mod.node_group = _node_group(head_ob, h_eye, h_lid, thick)
    # thin film: no shadows, not seen by other shiny surfaces' shadow rays
    if hasattr(ob, "visible_shadow"):
        ob.visible_shadow = False
    cfg["tear_line"] = {"object": OBJ, "h_eye_mm": h_eye * 1e3, "h_lid_mm": h_lid * 1e3, "thick_mm": thick * 1e3}
    sc["track1_config"] = json.dumps(cfg)
    rep = {"tear_line_verts": len(me.vertices), "tear_line_faces": len(me.polygons)}
    if wet:
        rep.update(add_wet_rims(head_ob, cfg))
    return rep


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--save", action="store_true", help="save into the opened .blend")
    p.add_argument("--out", default=None, help="save as this .blend instead")
    p.add_argument("--h-eye", type=float, default=H_EYE_M * 1e3, help="mm")
    p.add_argument("--h-lid", type=float, default=H_LID_M * 1e3, help="mm")
    p.add_argument("--thick", type=float, default=THICK_M * 1e3, help="bead thickness, mm")
    p.add_argument("--wet", action="store_true", help="also add the wet lid-rim layer (16 Nov tests: no visible effect under the c8 lights)")
    a = p.parse_args(argv)
    print("[tearline]", add(a.h_eye * 1e-3, a.h_lid * 1e-3, a.thick * 1e-3, a.wet), flush=True)
    if a.out:
        bpy.ops.wm.save_as_mainfile(filepath=a.out, compress=True)
        print("[tearline] saved", a.out, flush=True)
    elif a.save:
        bpy.ops.wm.save_mainfile(compress=True)
        print("[tearline] saved", bpy.data.filepath, flush=True)
