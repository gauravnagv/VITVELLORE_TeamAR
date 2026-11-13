"""
verify.py - self-test of scene.blend (GNM head) + wrapper/eyes.py (no rendering).

    blender -b scene.blend --python verify.py

Checks, for many (h, v) inputs:
  1. pupil_center_2D from our K[R|t] matches Blender's own world_to_camera_view.
  2. The evaluated (deformed) eyeballs moved rigidly about their centres:
     every eye vertex == centre + R (rest - centre)   (max error in mm).
  3. Both optical axes pass through the fixation target (convergence).
  4. Pupil 2D shift direction is correct for left / right / up / down; reset is exact.
  5. Eyelids: the upper-lid margin rises when looking up and drops when looking down
     (lid-gaze coupling), and blink() closes the eye (upper margin meets the lower lid).
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "wrapper"))
import eyes  # noqa: E402

r = eyes.rig()
scn = bpy.context.scene
cfg = r.cfg
head = r.head
for m in head.modifiers:            # compare base-resolution vertices (no subdivision)
    if m.type == "SUBSURF":
        m.show_viewport = False
dg = bpy.context.evaluated_depsgraph_get()


def eval_verts():
    dg.update()
    ev = head.evaluated_get(dg)
    me = ev.to_mesh()
    mw = np.array(ev.matrix_world)
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    ev.to_mesh_clear()
    return co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]


def group_idx(name):
    gi = head.vertex_groups[name].index
    return np.array([v.index for v in head.data.vertices
                     if any(g.group == gi and g.weight > 0.999 for g in v.groups)])


eye_idx = {"L": group_idx(cfg["bone_L"]), "R": group_idx(cfg["bone_R"])}
margins = cfg["lid_margins"]

r.arm.data.pose_position = "REST"
rest = eval_verts()
r.arm.data.pose_position = "POSE"
r.reset()

fails = []
worst = {"proj_px": 0.0, "rigid_mm": 0.0, "axis_mm": 0.0}
tests = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (0.7, 0.7), (-0.3, 0.9)]
tests += [(math.sin(2 * math.pi * k / 10), math.cos(2 * math.pi * k / 10)) for k in range(10)]
W = scn.render.resolution_x
for h, v in tests:
    r.move(h, v)
    rec = r.record()
    for s in "LR":
        ndc = world_to_camera_view(scn, r.camera, r.pupil_world(s))
        ref = (ndc.x * W, (1 - ndc.y) * W)
        ours = rec["pupil_center_2D_" + s]
        worst["proj_px"] = max(worst["proj_px"], abs(ref[0] - ours[0]), abs(ref[1] - ours[1]))
    cur = eval_verts()
    for s in "LR":
        idx = eye_idx[s]
        c = np.array(r.center[s])
        R = np.array(r.rot[s])
        pred = (rest[idx] - c) @ R.T + c
        worst["rigid_mm"] = max(worst["rigid_mm"], np.abs(pred - cur[idx]).max() * 1000)
        tgt = r.target_point(h, v)
        axis = (r.pupil_world(s) - r.center[s]).normalized()
        to_t = tgt - r.center[s]
        worst["axis_mm"] = max(worst["axis_mm"], (to_t - axis * to_t.dot(axis)).length * 1000)

r.reset()
c0 = r.record()
for name, h, v, ax, sign in [("left", -1, 0, 0, -1), ("right", 1, 0, 0, 1), ("up", 0, 1, 1, -1), ("down", 0, -1, 1, 1)]:
    rec = r.move(h, v).record()
    for s in "LR":
        d = rec["pupil_center_2D_" + s][ax] - c0["pupil_center_2D_" + s][ax]
        ok = d * sign > 3
        print("  %-5s eye %s  shift %+6.2f px  %s" % (name, s, d, "PASS" if ok else "FAIL"))
        if not ok:
            fails.append(name + s)
rec = r.reset().record()
if rec["pupil_center_2D_L"] != c0["pupil_center_2D_L"] or rec["pupil_center_2D_R"] != c0["pupil_center_2D_R"]:
    fails.append("reset")
print("  reset  exact return to centre: %s" % ("FAIL" if "reset" in fails else "PASS"))

# eyelids (world Z is up)
r.reset()
base = eval_verts()
for name, v, sign in (("up", 1, 1), ("down", -1, -1)):
    r.move(0, v)
    cur = eval_verts()
    for s in "LR":
        dz = (cur[margins[s]["upper"], 2] - base[margins[s]["upper"], 2]).mean() * 1000
        ok = dz * sign > 0.5
        print("  lid follows %-4s eye %s  upper margin %+5.2f mm  %s" % (name, s, dz, "PASS" if ok else "FAIL"))
        if not ok:
            fails.append("lid_" + name + s)
r.reset()
r.blink(1.0)
cur = eval_verts()
for s in "LR":
    up, lo = cur[margins[s]["upper"]], cur[margins[s]["lower"]]
    o = np.argsort(lo[:, 0])
    gap = up[:, 2] - np.interp(up[:, 0], lo[o, 0], lo[o, 2])
    ok = gap.max() < 0.0012
    print("  blink eye %s  lid gap mean %+.2f mm max %+.2f mm  %s" % (s, gap.mean() * 1e3, gap.max() * 1e3,
                                                                    "PASS" if ok else "FAIL"))
    if not ok:
        fails.append("blink" + s)
r.reset()

print("  projection vs Blender world_to_camera_view: max %.4f px" % worst["proj_px"])
print("  rigid rotation about eyeball centre:       max %.5f mm" % worst["rigid_mm"])
print("  optical axis miss distance to target:      max %.5f mm" % worst["axis_mm"])
if worst["proj_px"] > 0.05 or worst["rigid_mm"] > 0.01 or worst["axis_mm"] > 0.01:
    fails.append("geometry")
for m in head.modifiers:
    if m.type == "SUBSURF":
        m.show_viewport = True
print("VERIFY", "PASS" if not fails else "FAIL %s" % fails)
