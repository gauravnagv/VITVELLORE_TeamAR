"""
eyes.py - deterministic eye-articulation wrapper for the Track 01 scene.

API (inside Blender, with scene.blend open):
    import eyes
    eyes.move(h, v)      # h, v in [-1, +1]; both eyes fixate one shared target
    eyes.reset()         # back to straight-ahead fixation (h = v = 0)
    eyes.render(path)    # renders a 1024x1024 PNG, returns the per-frame metadata dict

Conventions
    * h > 0  -> pupils move to IMAGE RIGHT (the subject's left), h < 0 -> image left.
    * v > 0  -> pupils move UP, v < 0 -> down.
    * The fixation target sits 35 cm in front of the eye midpoint and is moved
      inside a +/-5 cm square (h, v scale that offset).  Each eyeball is rotated
      about its own centre so its optical axis passes through the target, which
      gives correct vergence (both eyes converge on a 35 cm target).
    * Rotations are the minimal rotation from the rest gaze direction (no
      torsion), reported as world-frame XYZ Euler angles in degrees.
    * 2D pupil centres are in pixels, origin top-left, x right, y down.
    * camera_matrix is the OpenCV 3x3 intrinsic matrix K. The record also
      carries the 3x4 world->camera extrinsic [R|t] and the projection P = K[R|t].

Command line (evaluators can run either form from the bundle root):
    python wrapper/eyes.py                       # relaunches itself inside Blender
    blender -b scene.blend --python wrapper/eyes.py -- sequence
    blender -b scene.blend --python wrapper/eyes.py -- directions
    blender -b scene.blend --python wrapper/eyes.py -- move 0.5 -0.3 --out out.png
Options: --engine EEVEE|CYCLES   --samples N   --outdir DIR
"""

import json
import math
import os
import sys

WRAPPER_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_DIR = os.path.dirname(WRAPPER_DIR)

FIXATION_DISTANCE_M = 0.35   # target distance in front of the eye midpoint
MOTION_BOUND_M = 0.05        # +/- range of the target for |h|, |v| = 1
CAMERA_DISTANCE_M = 0.40     # locked render camera distance
RESOLUTION = 1024
N_SEQUENCE = 10

try:
    import bpy
    from mathutils import Matrix, Vector
    IN_BLENDER = True
except ImportError:  # plain Python: we relaunch inside Blender (see __main__)
    IN_BLENDER = False


# --------------------------------------------------------------------------- #
# Core rig controller
# --------------------------------------------------------------------------- #
class EyeRig:
    """Wraps the armature eye bones written into the scene by build_scene.py."""

    def __init__(self, scene=None):
        self.scene = scene or bpy.context.scene
        if "track1_config" not in self.scene:
            raise RuntimeError("scene.blend has no 'track1_config' - build it with build_scene.py")
        cfg = json.loads(self.scene["track1_config"])
        self.cfg = cfg
        self.arm = bpy.data.objects[cfg["armature"]]
        self.bone_names = {"L": cfg["bone_L"], "R": cfg["bone_R"]}
        self.center = {s: Vector(cfg["center_" + s]) for s in "LR"}          # world, metres
        self.pupil_rest = {s: Vector(cfg["pupil_rest_" + s]) for s in "LR"}  # world, metres
        self.camera = bpy.data.objects[cfg["camera"]]
        self.target = bpy.data.objects.get(cfg.get("target", ""))
        self.midpoint = (self.center["L"] + self.center["R"]) / 2.0

        # Axes taken from the locked camera so "right"/"up" always match the image.
        cam_rot = self.camera.matrix_world.to_3x3().normalized()
        self.img_right = (cam_rot @ Vector((1, 0, 0))).normalized()
        self.img_up = (cam_rot @ Vector((0, 1, 0))).normalized()
        self.forward = (cam_rot @ Vector((0, 0, 1))).normalized()  # from face toward camera

        self.rest_dir = {s: (self.pupil_rest[s] - self.center[s]).normalized() for s in "LR"}
        self.h = 0.0
        self.v = 0.0
        self.rot = {s: Matrix.Identity(3) for s in "LR"}
        self.reset()

    # -- articulation ------------------------------------------------------- #
    def target_point(self, h, v):
        return (self.midpoint
                + self.forward * FIXATION_DISTANCE_M
                + self.img_right * (h * MOTION_BOUND_M)
                + self.img_up * (v * MOTION_BOUND_M))

    def move(self, h, v):
        """Rotate both eyeballs about their centres to fixate the (h, v) target."""
        h = max(-1.0, min(1.0, float(h)))
        v = max(-1.0, min(1.0, float(v)))
        self.h, self.v = h, v
        tgt = self.target_point(h, v)
        if self.target is not None:
            self.target.location = tgt

        A = self.arm.matrix_world
        A_inv = A.inverted()
        for s in "LR":
            c = self.center[s]
            gaze = (tgt - c).normalized()
            # Gaze for a straight-ahead target at 35 cm is the reference (rest) pose
            # of the bone; we rotate from the mesh's modelled forward axis.
            q = self.rest_dir[s].rotation_difference(gaze)
            R = q.to_matrix()
            self.rot[s] = R
            pb = self.arm.pose.bones[self.bone_names[s]]
            bone_rest_world = A @ pb.bone.matrix_local
            M_world = Matrix.Translation(c) @ R.to_4x4() @ Matrix.Translation(-c) @ bone_rest_world
            pb.matrix = A_inv @ M_world
        bpy.context.view_layer.update()
        return self

    def reset(self):
        return self.move(0.0, 0.0)

    # -- geometry / metadata ------------------------------------------------ #
    def pupil_world(self, s):
        c = self.center[s]
        return c + self.rot[s] @ (self.pupil_rest[s] - c)

    def camera_K(self):
        cam = self.camera.data
        r = self.scene.render
        w = r.resolution_x * r.resolution_percentage / 100.0
        h = r.resolution_y * r.resolution_percentage / 100.0
        if cam.sensor_fit == "VERTICAL":
            fy = cam.lens / cam.sensor_height * h
            fx = fy
        else:  # AUTO / HORIZONTAL with a square image -> horizontal sensor width
            fx = cam.lens / cam.sensor_width * w
            fy = fx
        cx = w / 2.0 - cam.shift_x * max(w, h)
        cy = h / 2.0 + cam.shift_y * max(w, h)
        return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]

    def camera_extrinsic(self):
        """OpenCV world->camera [R|t] (x right, y down, z forward)."""
        flip = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
        Mw = self.camera.matrix_world.normalized()
        R_bc = Mw.to_3x3().transposed()
        t_bc = -(R_bc @ Mw.translation)
        R = flip @ R_bc
        t = flip @ t_bc
        return [[R[i][0], R[i][1], R[i][2], t[i]] for i in range(3)]

    def project(self, p):
        K = self.camera_K()
        Rt = self.camera_extrinsic()
        xc = [sum(Rt[i][j] * p[j] for j in range(3)) + Rt[i][3] for i in range(3)]
        u = K[0][0] * xc[0] / xc[2] + K[0][2]
        v = K[1][1] * xc[1] / xc[2] + K[1][2]
        return [u, v]

    def record(self, frame=None, theta=None, image=None):
        K = self.camera_K()
        Rt = self.camera_extrinsic()
        P = [[sum(K[i][k] * Rt[k][j] for k in range(3)) for j in range(4)] for i in range(3)]
        rec = {}
        if frame is not None:
            rec["frame"] = frame
        if theta is not None:
            rec["theta"] = round(theta, 6)
        rec["input"] = {"h": round(self.h, 6), "v": round(self.v, 6)}
        for s in "LR":
            e = self.rot[s].to_euler("XYZ")
            rec["eye_rotation_" + s] = [round(math.degrees(a), 4) for a in e]
        for s in "LR":
            rec["pupil_center_2D_" + s] = [round(x, 3) for x in self.project(self.pupil_world(s))]
        rec["camera_matrix"] = [[round(x, 4) for x in row] for row in K]
        # Extra fields (not required, provided for verification / reuse)
        rec["camera_extrinsic_world_to_cam"] = [[round(x, 6) for x in row] for row in Rt]
        rec["projection_matrix"] = [[round(x, 4) for x in row] for row in P]
        for s in "LR":
            rec["pupil_center_3D_" + s] = [round(x, 6) for x in self.pupil_world(s)]
            rec["eye_center_3D_" + s] = [round(x, 6) for x in self.center[s]]
            g = (self.pupil_world(s) - self.center[s]).normalized()
            rec["gaze_dir_" + s] = [round(x, 6) for x in g]
            # gaze angles vs. straight ahead (toward camera): yaw + = image right, pitch + = up
            gx, gy, gz = g.dot(self.img_right), g.dot(self.img_up), g.dot(self.forward)
            rec["gaze_yaw_pitch_deg_" + s] = [round(math.degrees(math.atan2(gx, gz)), 4),
                                              round(math.degrees(math.atan2(gy, math.hypot(gx, gz))), 4)]
        rec["fixation_target_3D"] = [round(x, 6) for x in self.target_point(self.h, self.v)]
        rec["units"] = {"rotation": "degrees, XYZ Euler, world frame",
                        "3D": "metres, Blender world (Z up, face looks -Y)",
                        "2D": "pixels, origin top-left, x right, y down"}
        if image:
            rec["image"] = image
        return rec

    # -- rendering ---------------------------------------------------------- #
    def render(self, path, frame=None, theta=None):
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        r = self.scene.render
        r.resolution_x = r.resolution_y = RESOLUTION
        r.resolution_percentage = 100
        r.image_settings.file_format = "PNG"
        r.image_settings.color_mode = "RGB"
        r.filepath = path
        bpy.ops.render.render(write_still=True)
        rel = os.path.relpath(path, BUNDLE_DIR).replace("\\", "/")
        return self.record(frame=frame, theta=theta, image=rel)


# --------------------------------------------------------------------------- #
# Module-level convenience API:  eyes.move(h, v) / eyes.reset() / eyes.render()
# --------------------------------------------------------------------------- #
_RIG = None


def rig():
    global _RIG
    if _RIG is None or _RIG.scene != bpy.context.scene:
        _RIG = EyeRig()
    return _RIG


def move(h, v):
    return rig().move(h, v)


def reset():
    return rig().reset()


def render(path=None):
    if path is None:
        path = os.path.join(BUNDLE_DIR, "renders", "render_h%+.3f_v%+.3f.png" % (rig().h, rig().v))
    return rig().render(path)


def set_engine(engine="EEVEE", samples=None):
    scn = bpy.context.scene
    if engine.upper().startswith("CYC"):
        scn.render.engine = "CYCLES"
        scn.cycles.samples = samples or 128
        scn.cycles.seed = 0
        scn.cycles.use_denoising = True
        prefs = bpy.context.preferences.addons.get("cycles")
        if prefs:
            cp = prefs.preferences
            for dev_type in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
                try:
                    cp.compute_device_type = dev_type
                    cp.get_devices()
                    if any(d.type == dev_type for d in cp.devices):
                        for d in cp.devices:
                            d.use = True
                        scn.cycles.device = "GPU"
                        break
                except TypeError:
                    continue
            else:
                scn.cycles.device = "CPU"
    else:
        eevee = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in {
            e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
        } else "BLENDER_EEVEE"
        scn.render.engine = eevee
        if samples:
            scn.eevee.taa_render_samples = samples


# --------------------------------------------------------------------------- #
# Batch jobs
# --------------------------------------------------------------------------- #
def run_sequence(outdir=None):
    """10 clockwise renders: theta_k = 2*pi*k/10, h = sin, v = cos, starting at top."""
    outdir = outdir or BUNDLE_DIR
    r = rig()
    records = []
    for k in range(N_SEQUENCE):
        theta = 2.0 * math.pi * k / N_SEQUENCE
        h, v = math.sin(theta), math.cos(theta)
        r.move(h, v)
        img = os.path.join(outdir, "renders", "frame_%02d.png" % k)
        rec = r.render(img, frame=k, theta=theta)
        records.append(rec)
        print("[eyes] frame %d  theta=%.4f  h=%+.3f v=%+.3f  pupilL=%s pupilR=%s" % (
            k, theta, h, v, rec["pupil_center_2D_L"], rec["pupil_center_2D_R"]), flush=True)
    r.reset()
    meta = os.path.join(outdir, "metadata.json")
    with open(meta, "w") as f:
        json.dump(records, f, indent=2)
    print("[eyes] wrote %s" % meta, flush=True)
    return records


def run_directions(outdir=None):
    """Direction check: center, left, right, up, down, reset."""
    outdir = outdir or os.path.join(BUNDLE_DIR, "demo", "direction_check")
    r = rig()
    poses = [("center", 0, 0), ("left", -1, 0), ("right", 1, 0),
             ("up", 0, 1), ("down", 0, -1)]
    records = []
    for name, h, v in poses:
        r.move(h, v)
        rec = r.render(os.path.join(outdir, name + ".png"))
        rec["name"] = name
        records.append(rec)
        print("[eyes] %-6s pupilL=%s pupilR=%s" % (name, rec["pupil_center_2D_L"], rec["pupil_center_2D_R"]), flush=True)
    r.reset()
    rec = r.render(os.path.join(outdir, "reset.png"))
    rec["name"] = "reset"
    records.append(rec)
    with open(os.path.join(outdir, "directions.json"), "w") as f:
        json.dump(records, f, indent=2)
    return records


def _parse(argv):
    import argparse
    p = argparse.ArgumentParser(prog="eyes.py")
    p.add_argument("cmd", nargs="?", default="sequence", choices=["sequence", "directions", "move", "all"])
    p.add_argument("h", nargs="?", type=float, default=0.0)
    p.add_argument("v", nargs="?", type=float, default=0.0)
    p.add_argument("--out", default=None, help="output PNG for 'move'")
    p.add_argument("--outdir", default=None)
    p.add_argument("--engine", default=os.environ.get("EYES_ENGINE", "EEVEE"))
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--blender", default=None, help="Blender executable (plain-Python launch only)")
    return p.parse_args(argv)


def _find_blender(explicit=None):
    import shutil
    import glob
    cands = [explicit, os.environ.get("BLENDER"), shutil.which("blender")]
    pats = [os.path.join(BUNDLE_DIR, "..", "tools", "blender-*", "blender*"),
            r"C:\Program Files\Blender Foundation\Blender*\blender.exe",
            "/Applications/Blender.app/Contents/MacOS/Blender",
            "/usr/bin/blender", "/snap/bin/blender"]
    for pat in pats:
        cands += sorted(glob.glob(pat), reverse=True)
    for c in cands:
        if c and os.path.isfile(c) and os.path.basename(c).lower() in ("blender", "blender.exe"):
            return c
    raise SystemExit("Blender not found: pass --blender PATH or set BLENDER=/path/to/blender")


def _relaunch(argv):
    import subprocess
    args = _parse(argv)
    exe = _find_blender(args.blender)
    if args.blender:
        i = argv.index("--blender")
        argv = argv[:i] + argv[i + 2:]
    scene = os.path.join(BUNDLE_DIR, "scene.blend")
    cmd = [exe, "-b", scene, "--python", os.path.abspath(__file__), "--"] + argv
    print("[eyes] launching:", " ".join('"%s"' % c if " " in c else c for c in cmd), flush=True)
    sys.exit(subprocess.call(cmd))


def main(argv):
    args = _parse(argv)
    set_engine(args.engine, args.samples)
    if args.cmd in ("sequence", "all"):
        run_sequence(args.outdir)
    if args.cmd in ("directions", "all"):
        run_directions(os.path.join(args.outdir, "demo", "direction_check") if args.outdir else None)
    if args.cmd == "move":
        move(args.h, args.v)
        out = args.out or os.path.join(BUNDLE_DIR, "renders", "move_h%+.3f_v%+.3f.png" % (args.h, args.v))
        rec = rig().render(out)
        print(json.dumps(rec, indent=2), flush=True)


if __name__ == "__main__":
    user_argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else (
        [] if IN_BLENDER else sys.argv[1:])
    if IN_BLENDER:
        main(user_argv)
    else:
        _relaunch(user_argv)
