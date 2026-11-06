"""
verify.py - self-test of scene.blend + wrapper/eyes.py (no rendering).

    blender -b scene.blend --python verify.py

Checks, for many (h, v) inputs:
  1. pupil_center_2D from our K[R|t] matches Blender's own world_to_camera_view.
  2. The evaluated (deformed) eye mesh really moved rigidly about the eyeball
     centre: every eye vertex == centre + R (rest - centre)   (max error in mm).
  3. Both optical axes pass through the fixation target (convergence).
  4. Pupil 2D shift direction is correct for left / right / up / down, and
     reset returns exactly to the centre pose.
"""
import math
import os
import sys

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "wrapper"))
import eyes  # noqa: E402

r = eyes.rig()
scn = bpy.context.scene
cfg = r.cfg
eye_obj = bpy.data.objects[cfg["eyes_mesh"]]
dg = bpy.context.evaluated_depsgraph_get()


def eval_verts():
    ev = eye_obj.evaluated_get(dg)
    me = ev.to_mesh()
    mw = ev.matrix_world
    pts = np.array([tuple(mw @ v.co) for v in me.vertices])
    ev.to_mesh_clear()
    return pts


r.arm.data.pose_position = "REST"   # true modelled rest pose (eyes not rotated)
dg.update()
rest = eval_verts()
r.arm.data.pose_position = "POSE"
r.reset()
side_of = np.where(rest[:, 0] > 0, "L", "R")

fails = []
worst = {"proj_px": 0.0, "rigid_mm": 0.0, "axis_mm": 0.0}
tests = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (0.7, 0.7), (-0.3, 0.9)]
tests += [(math.sin(2 * math.pi * k / 10), math.cos(2 * math.pi * k / 10)) for k in range(10)]
for h, v in tests:
    r.move(h, v)
    dg.update()
    rec = r.record()
    W = scn.render.resolution_x
    for s in "LR":
        ndc = world_to_camera_view(scn, r.camera, r.pupil_world(s))
        ref = (ndc.x * W, (1 - ndc.y) * W)
        ours = rec["pupil_center_2D_" + s]
        worst["proj_px"] = max(worst["proj_px"], abs(ref[0] - ours[0]), abs(ref[1] - ours[1]))
    cur = eval_verts()
    for s in "LR":
        m = side_of == s
        c = np.array(r.center[s])
        R = np.array(r.rot[s])
        pred = (rest[m] - c) @ R.T + c
        worst["rigid_mm"] = max(worst["rigid_mm"], np.abs(pred - cur[m]).max() * 1000)
        tgt = r.target_point(h, v)
        axis = (r.pupil_world(s) - r.center[s]).normalized()
        to_t = tgt - r.center[s]
        miss = (to_t - axis * to_t.dot(axis)).length * 1000
        worst["axis_mm"] = max(worst["axis_mm"], miss)

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

print("  projection vs Blender world_to_camera_view: max %.4f px" % worst["proj_px"])
print("  rigid rotation about eyeball centre:       max %.5f mm" % worst["rigid_mm"])
print("  optical axis miss distance to target:      max %.5f mm" % worst["axis_mm"])
if worst["proj_px"] > 0.05 or worst["rigid_mm"] > 0.01 or worst["axis_mm"] > 0.01:
    fails.append("geometry")
print("VERIFY", "PASS" if not fails else "FAIL %s" % fails)
