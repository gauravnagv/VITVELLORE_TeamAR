"""
eyes.py - deterministic eye / eyelid / brow controller for the Track 01 GNM scene.

API (inside Blender, with scene.blend open):
    import eyes
    eyes.move(h, v)        # h, v in [-1, +1]; both eyes fixate one shared target
    eyes.reset()           # straight-ahead fixation, lids/brows/pupil neutral
    eyes.blink(a)          # 0 = open .. 1 = closed (both eyes; blink(a, side="L") for one)
    eyes.brows(b)          # -1 = lowered/knit .. 0 .. +1 = raised
    eyes.pupil(p)          # -1 = constricted .. 0 .. +1 = dilated
    eyes.smile(a)          # 0 .. 1 smile (lip corners + cheeks), GNM lower-face space
    eyes.squint(a)         # 0 .. 1 cheek raise + lower-lid tighten (squint_L / squint_R)
    eyes.render(path)      # 1024x1024 PNG, returns the per-frame metadata dict
    eyes.set_textures(dir) # switch to another GNM UDIM texture set (textures/setNN_*)

Conventions
    * h > 0 -> pupils move to IMAGE RIGHT (the subject's left); v > 0 -> up.
    * The fixation target sits 35 cm in front of the eye midpoint inside a +/-5 cm
      square.  Each eyeball rotates about its own centre so its optical axis passes
      through the target (vergence included).  Rotations are the minimal rotation from
      the rest gaze (no torsion), reported as world-frame XYZ Euler angles in degrees.
    * Eyelids follow vertical gaze (lid-gaze coupling): upper lid ~1:1, lower lid
      partially (Becker & Fuchs 1988).  Lids slide over the globe about the eyeball
      centre; blink closes the lids along their whole length.
    * 2D pupil centres in pixels (origin top-left); camera_matrix is OpenCV K.

Command line (from the bundle root):
    python wrapper/eyes.py                       # relaunches itself inside Blender
    blender -b scene.blend --python wrapper/eyes.py -- sequence
    blender -b scene.blend --python wrapper/eyes.py -- directions
    blender -b scene.blend --python wrapper/eyes.py -- move 0.5 -0.3 --out out.png
Options: --engine EEVEE|CYCLES   --samples N   --outdir DIR   --textures DIR
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
PUPIL_DILATE_MAX = 0.55      # GNM pupil_dilate key value reached at eyes.pupil(+1)

# Lid-gaze coupling gains (relative to each eye's pitch change from straight ahead)
LID_FOLLOW = True
LID_GAIN = {
    "upper": {"up": 0.85, "down": 1.0},
    "lower": {"up": 0.25, "down": 0.45},
}

try:
    import bpy
    from mathutils import Matrix, Vector
    IN_BLENDER = True
except ImportError:  # plain Python: we relaunch inside Blender (see __main__)
    IN_BLENDER = False


def _clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, float(x)))


# --------------------------------------------------------------------------- #
# Core rig controller
# --------------------------------------------------------------------------- #
class EyeRig:
    """Drives the GNM armature eye bones and the eyelid / brow / pupil shape keys."""

    def __init__(self, scene=None):
        self.scene = scene or bpy.context.scene
        if "track1_config" not in self.scene:
            raise RuntimeError("scene.blend has no 'track1_config' - build it with build_scene_gnm.py")
        cfg = json.loads(self.scene["track1_config"])
        self.cfg = cfg
        self.arm = bpy.data.objects[cfg["armature"]]
        self.head = bpy.data.objects[cfg.get("head_mesh", cfg.get("eyes_mesh"))]
        self.bone_names = {"L": cfg["bone_L"], "R": cfg["bone_R"]}
        self.center = {s: Vector(cfg["center_" + s]) for s in "LR"}
        self.pupil_rest = {s: Vector(cfg["pupil_rest_" + s]) for s in "LR"}
        self.camera = bpy.data.objects[cfg["camera"]]
        self.target = bpy.data.objects.get(cfg.get("target", ""))
        self.midpoint = (self.center["L"] + self.center["R"]) / 2.0
        self.key_deg = cfg.get("lid_key_deg", {})
        sk = self.head.data.shape_keys
        self.keys = sk.key_blocks if sk else {}

        cam_rot = self.camera.matrix_world.to_3x3().normalized()
        self.img_right = (cam_rot @ Vector((1, 0, 0))).normalized()
        self.img_up = (cam_rot @ Vector((0, 1, 0))).normalized()
        self.forward = (cam_rot @ Vector((0, 0, 1))).normalized()  # from face toward camera

        self.rest_dir = {s: (self.pupil_rest[s] - self.center[s]).normalized() for s in "LR"}
        self.pitch0 = {s: self._pitch((self.target_point(0, 0) - self.center[s]).normalized()) for s in "LR"}
        self.H = Matrix.Identity(4)          # current head-bone deformation (world)
        self.head_angles = (0.0, 0.0, 0.0)   # yaw, pitch, roll (deg)
        self.gaze_target = self.target_point(0, 0)
        self.h = self.v = 0.0
        self.rot = {s: Matrix.Identity(3) for s in "LR"}
        self.lid_angle = {s: {"upper": 0.0, "lower": 0.0} for s in "LR"}
        self.blink_amt = {"L": 0.0, "R": 0.0}
        self.brow_amt = {"L": 0.0, "R": 0.0}
        self.pupil_amt = 0.0
        self.smile_amt = 0.0
        self.squint_amt = {"L": 0.0, "R": 0.0}
        self.reset()

    # -- helpers ------------------------------------------------------------- #
    def _pitch(self, d):
        return math.atan2(d.dot(self.img_up), math.hypot(d.dot(self.img_right), d.dot(self.forward)))

    def _key(self, name, value):
        kb = self.keys.get(name) if self.keys else None
        if kb is not None:
            kb.value = max(0.0, min(kb.slider_max, value))   # lid keys extrapolate to 1.5x

    def target_point(self, h, v):
        return (self.midpoint + self.forward * FIXATION_DISTANCE_M
                + self.img_right * (h * MOTION_BOUND_M) + self.img_up * (v * MOTION_BOUND_M))

    # -- articulation -------------------------------------------------------- #
    def move(self, h, v):
        """Rotate both eyeballs about their centres to fixate the (h, v) target."""
        h, v = _clamp(h), _clamp(v)
        self.h, self.v = h, v
        return self.fixate(self.target_point(h, v))

    def look(self, yaw_deg, pitch_deg, distance=0.6):
        """Free gaze (for animation): fixate a point `distance` m away at the given angles
        relative to the camera axis (yaw + = image right, pitch + = up)."""
        y, p = math.radians(yaw_deg), math.radians(pitch_deg)
        d = (self.forward * (math.cos(p) * math.cos(y)) + self.img_right * (math.cos(p) * math.sin(y))
             + self.img_up * math.sin(p))
        mid = self.H @ self.midpoint
        return self.fixate(mid + d * distance)

    def fixate(self, tgt):
        """Rotate each eyeball about its (head-carried) centre so its axis passes through tgt."""
        self.gaze_target = tgt.copy()
        if self.target is not None:
            self.target.location = tgt
        A = self.arm.matrix_world
        A_inv = A.inverted()
        H3 = self.H.to_3x3()
        H3_inv = H3.inverted()
        for s in "LR":
            c = self.H @ self.center[s]
            gaze = (tgt - c).normalized()
            R = (H3 @ self.rest_dir[s]).rotation_difference(gaze).to_matrix()
            self.rot[s] = R @ H3                      # total rotation of the eyeball vs. rest
            pb = self.arm.pose.bones[self.bone_names[s]]
            rest_world = A @ pb.bone.matrix_local
            pb.matrix = A_inv @ (Matrix.Translation(c) @ R.to_4x4() @ Matrix.Translation(-c) @ self.H @ rest_world)
            # lids follow the eye's pitch relative to the head
            dp = (self._pitch(H3_inv @ gaze) - self.pitch0[s]) if LID_FOLLOW else 0.0
            for lid in ("upper", "lower"):
                self.lid_angle[s][lid] = LID_GAIN[lid]["up" if dp > 0 else "down"] * dp
        self._apply_lids()
        return self

    def set_head(self, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0, keep_gaze=True):
        """Pose neck (40 %) + head (60 %).  yaw + = face turns to image right, pitch + = up,
        roll + = head tilts toward image right.  The eyes keep fixating their target."""
        self.head_angles = (yaw_deg, pitch_deg, roll_deg)
        A = self.arm.matrix_world
        A_inv = A.inverted()

        def rot(f):
            return (Matrix.Rotation(math.radians(yaw_deg * f), 4, self.img_up)
                    @ Matrix.Rotation(math.radians(-pitch_deg * f), 4, self.img_right)
                    @ Matrix.Rotation(math.radians(-roll_deg * f), 4, self.forward))
        names = self.cfg.get("head_bones", ["neck", "head"])
        pn, ph = self.arm.pose.bones[names[0]], self.arm.pose.bones[names[1]]
        n = A @ pn.bone.head_local
        hpos = A @ ph.bone.head_local
        M_neck = Matrix.Translation(n) @ rot(0.4) @ Matrix.Translation(-n)
        pn.matrix = A_inv @ (M_neck @ A @ pn.bone.matrix_local)
        h2 = M_neck @ hpos
        # remaining 60 % about the (moved) head joint; total head rotation = rot(1.0)
        M_head = Matrix.Translation(h2) @ (rot(1.0) @ rot(0.4).inverted()) @ Matrix.Translation(-h2) @ M_neck
        ph.matrix = A_inv @ (M_head @ A @ ph.bone.matrix_local)
        self.H = M_head
        bpy.context.view_layer.update()
        if keep_gaze:
            self.fixate(self.gaze_target)
        return self

    def _apply_lids(self):
        """Lid-follow keys (scaled down while blinking) + blink keys."""
        for s in "LR":
            open_f = 1.0 - self.blink_amt[s]
            for lid, pre in (("upper", "lidU"), ("lower", "lidL")):
                down_deg = -math.degrees(self.lid_angle[s][lid])  # + = lid moves down
                kd = self.key_deg.get(pre + "_down", 20.0)
                ku = self.key_deg.get(pre + "_up", -15.0)
                self._key("%s_down_%s" % (pre, s), open_f * max(0.0, down_deg) / kd)
                self._key("%s_up_%s" % (pre, s), open_f * max(0.0, -down_deg) / -ku)
            self._key("blink_" + s, self.blink_amt[s])
        bpy.context.view_layer.update()

    def blink(self, amount=1.0, side=None):
        for s in ("LR" if side is None else side):
            self.blink_amt[s] = _clamp(amount, 0.0, 1.0)
        self._apply_lids()
        return self

    def brows(self, amount=0.0, side=None):
        for s in ("LR" if side is None else side):
            b = _clamp(amount)
            self.brow_amt[s] = b
            self._key("brow_up_" + s, max(0.0, b))
            self._key("brow_down_" + s, max(0.0, -b))
        bpy.context.view_layer.update()
        return self

    def pupil(self, amount=0.0):
        """-1 = fully constricted .. +1 = fully dilated.  +1 maps to PUPIL_DILATE_MAX of the
        GNM dilation key: a human pupil dilates to ~8 mm in a ~12 mm iris (~65 %); the raw key
        at 1.0 leaves only a thin iris ring, which reads as unnatural."""
        p = _clamp(amount)
        self.pupil_amt = p
        self._key("pupil_dilate", PUPIL_DILATE_MAX * max(0.0, p))
        self._key("pupil_constrict", max(0.0, -p))
        bpy.context.view_layer.update()
        return self

    def smile(self, amount=0.0):
        """0 .. 1 lip-corner puller with cheek lift (AU12), fitted in GNM's lower-face space."""
        self.smile_amt = _clamp(amount, 0.0, 1.0)
        self._key("smile", self.smile_amt)
        bpy.context.view_layer.update()
        return self

    def squint(self, amount=0.0, side=None):
        """0 .. 1 cheek raise + lower-lid tighten (AU6/AU7); a real smile squints too."""
        for s in ("LR" if side is None else side):
            self.squint_amt[s] = _clamp(amount, 0.0, 1.0)
            self._key("squint_" + s, self.squint_amt[s])
        bpy.context.view_layer.update()
        return self

    def reset(self):
        self.blink_amt = {"L": 0.0, "R": 0.0}
        self.brows(0.0)
        self.pupil(0.0)
        self.smile(0.0)
        self.squint(0.0)
        self.set_head(0.0, 0.0, 0.0, keep_gaze=False)
        return self.move(0.0, 0.0)

    # -- geometry / metadata ------------------------------------------------- #
    def pupil_world(self, s):
        c = self.center[s]
        return self.H @ c + self.rot[s] @ (self.pupil_rest[s] - c)

    def center_world(self, s):
        return self.H @ self.center[s]

    def camera_K(self):
        cam = self.camera.data
        r = self.scene.render
        w = r.resolution_x * r.resolution_percentage / 100.0
        h = r.resolution_y * r.resolution_percentage / 100.0
        if cam.sensor_fit == "VERTICAL":
            fy = cam.lens / cam.sensor_height * h
            fx = fy
        else:
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
        return [K[0][0] * xc[0] / xc[2] + K[0][2], K[1][1] * xc[1] / xc[2] + K[1][2]]

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
            rec["eye_rotation_" + s] = [round(math.degrees(a), 4) for a in self.rot[s].to_euler("XYZ")]
        for s in "LR":
            rec["pupil_center_2D_" + s] = [round(x, 3) for x in self.project(self.pupil_world(s))]
        rec["camera_matrix"] = [[round(x, 4) for x in row] for row in K]
        # extra fields (not required; for verification / reuse)
        rec["camera_extrinsic_world_to_cam"] = [[round(x, 6) for x in row] for row in Rt]
        rec["projection_matrix"] = [[round(x, 4) for x in row] for row in P]
        for s in "LR":
            rec["pupil_center_3D_" + s] = [round(x, 6) for x in self.pupil_world(s)]
            rec["eye_center_3D_" + s] = [round(x, 6) for x in self.center_world(s)]
            g = (self.pupil_world(s) - self.center_world(s)).normalized()
            rec["gaze_dir_" + s] = [round(x, 6) for x in g]
            gx, gy, gz = g.dot(self.img_right), g.dot(self.img_up), g.dot(self.forward)
            rec["gaze_yaw_pitch_deg_" + s] = [round(math.degrees(math.atan2(gx, gz)), 4),
                                              round(math.degrees(math.atan2(gy, math.hypot(gx, gz))), 4)]
            rec["eyelid_pitch_deg_" + s] = {k: round(math.degrees(a), 4) for k, a in self.lid_angle[s].items()}
        rec["blink"] = {s: round(self.blink_amt[s], 4) for s in "LR"}
        rec["brows"] = {s: round(self.brow_amt[s], 4) for s in "LR"}
        rec["pupil"] = round(self.pupil_amt, 4)
        rec["smile"] = round(self.smile_amt, 4)
        rec["squint"] = {s: round(self.squint_amt[s], 4) for s in "LR"}
        rec["fixation_target_3D"] = [round(x, 6) for x in self.gaze_target]
        rec["head_yaw_pitch_roll_deg"] = [round(a, 4) for a in self.head_angles]
        rec["units"] = {"rotation": "degrees, XYZ Euler, world frame",
                        "3D": "metres, Blender world (Z up, face looks -Y)",
                        "2D": "pixels, origin top-left, x right, y down"}
        if image:
            rec["image"] = image
        return rec

    # -- rendering ----------------------------------------------------------- #
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
        try:
            rel = os.path.relpath(path, BUNDLE_DIR)
        except ValueError:  # Windows: output on a different drive than the bundle
            rel = path
        return self.record(frame=frame, theta=theta, image=rel.replace("\\", "/"))


# --------------------------------------------------------------------------- #
# Module-level convenience API
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


def blink(amount=1.0, side=None):
    return rig().blink(amount, side)


def brows(amount=0.0, side=None):
    return rig().brows(amount, side)


def pupil(amount=0.0):
    return rig().pupil(amount)


def smile(amount=0.0):
    return rig().smile(amount)


def squint(amount=0.0, side=None):
    return rig().squint(amount, side)


def render(path=None):
    if path is None:
        path = os.path.join(BUNDLE_DIR, "renders", "render_h%+.3f_v%+.3f.png" % (rig().h, rig().v))
    return rig().render(path)


def set_textures(tex_dir):
    """Point every tiled GNM texture (face / face_normal / face_roughness) at another set."""
    tex_dir = os.path.abspath(tex_dir)
    for img in bpy.data.images:
        if img.source != "TILED":
            continue
        stem = img.name.split(".")[0]
        if stem in ("face", "face_normal", "face_roughness"):
            if img.packed_file is not None or img.packed_files:
                try:
                    img.unpack(method="REMOVE")
                except Exception:
                    pass
            img.filepath = os.path.join(tex_dir, "%s.<UDIM>.png" % stem)
            img.reload()


def set_engine(engine="EEVEE", samples=None):
    scn = bpy.context.scene
    if engine.upper().startswith("CYC"):
        scn.render.engine = "CYCLES"
        scn.cycles.samples = samples or 128
        scn.cycles.seed = 0
        scn.cycles.use_denoising = True
        prefs = bpy.context.preferences.addons.get("cycles")
        scn.cycles.device = "CPU"
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
        scn.render.engine = "BLENDER_EEVEE_NEXT"
        if samples:
            scn.eevee.taa_render_samples = samples


# --------------------------------------------------------------------------- #
# Batch jobs (original brief deliverables)
# --------------------------------------------------------------------------- #
def run_sequence(outdir=None):
    """10 clockwise renders: theta_k = 2*pi*k/10, h = sin, v = cos, starting at top."""
    outdir = outdir or BUNDLE_DIR
    r = rig()
    records = []
    for k in range(N_SEQUENCE):
        theta = 2.0 * math.pi * k / N_SEQUENCE
        r.move(math.sin(theta), math.cos(theta))
        rec = r.render(os.path.join(outdir, "renders", "frame_%02d.png" % k), frame=k, theta=theta)
        records.append(rec)
        print("[eyes] frame %d  theta=%.4f  pupilL=%s pupilR=%s" % (
            k, theta, rec["pupil_center_2D_L"], rec["pupil_center_2D_R"]), flush=True)
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
    records = []
    for name, h, v in [("center", 0, 0), ("left", -1, 0), ("right", 1, 0), ("up", 0, 1), ("down", 0, -1)]:
        r.move(h, v)
        rec = r.render(os.path.join(outdir, name + ".png"))
        rec["name"] = name
        records.append(rec)
        print("[eyes] %-6s pupilL=%s pupilR=%s" % (name, rec["pupil_center_2D_L"], rec["pupil_center_2D_R"]),
              flush=True)
    r.reset()
    rec = r.render(os.path.join(outdir, "reset.png"))
    rec["name"] = "reset"
    records.append(rec)
    with open(os.path.join(outdir, "directions.json"), "w") as f:
        json.dump(records, f, indent=2)
    return records


def texture_set_dirs(root=None):
    """Every finished texture set: textures/setNN_<name>/ containing face.1001.png."""
    root = root or os.path.join(BUNDLE_DIR, "textures")
    return [os.path.join(root, d) for d in sorted(os.listdir(root))
            if d.startswith("set") and os.path.isfile(os.path.join(root, d, "face.1001.png"))]


def run_texture_sets(outdir=None, h=0.0, v=0.0):
    """One render per texture set (same pose, camera and lights) -> renders/texture_sets/."""
    outdir = outdir or os.path.join(BUNDLE_DIR, "renders", "texture_sets")
    r = rig()
    r.reset()
    r.move(h, v)
    records = []
    for d in texture_set_dirs():
        name = os.path.basename(d)
        set_textures(d)
        rec = r.render(os.path.join(outdir, name + ".png"))
        rec["texture_set"] = name
        info = os.path.join(d, "set.json")
        if os.path.isfile(info):
            with open(info) as f:
                rec["texture_set_info"] = json.load(f)
        records.append(rec)
        print("[eyes] texture set %-34s -> %s" % (name, rec["image"]), flush=True)
    r.reset()
    with open(os.path.join(outdir, "texture_sets.json"), "w") as f:
        json.dump(records, f, indent=2)
    print("[eyes] wrote %d texture-set renders to %s" % (len(records), outdir), flush=True)
    return records


def _parse(argv):
    import argparse
    p = argparse.ArgumentParser(prog="eyes.py")
    p.add_argument("cmd", nargs="?", default="sequence",
                   choices=["sequence", "directions", "textures", "move", "all"])
    p.add_argument("h", nargs="?", type=float, default=0.0)
    p.add_argument("v", nargs="?", type=float, default=0.0)
    p.add_argument("--out", default=None, help="output PNG for 'move'")
    p.add_argument("--outdir", default=None)
    p.add_argument("--textures", default=None, help="GNM texture-set folder (face.<UDIM>.png)")
    p.add_argument("--engine", default=os.environ.get("EYES_ENGINE", "EEVEE"))
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--blender", default=None, help="Blender executable (plain-Python launch only)")
    return p.parse_args(argv)


def _find_blender(explicit=None):
    import glob
    import shutil
    cands = [explicit, os.environ.get("BLENDER"), shutil.which("blender")]
    pats = [os.path.join(BUNDLE_DIR, "..", "tools", "blender-*", "blender*"),
            r"C:\Program Files\Blender Foundation\Blender*\blender.exe",
            "/Applications/Blender.app/Contents/MacOS/Blender", "/usr/bin/blender", "/snap/bin/blender"]
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
    cmd = [exe, "-b", os.path.join(BUNDLE_DIR, "scene.blend"), "--python", os.path.abspath(__file__), "--"] + argv
    print("[eyes] launching:", " ".join('"%s"' % c if " " in c else c for c in cmd), flush=True)
    sys.exit(subprocess.call(cmd))


def main(argv):
    args = _parse(argv)
    set_engine(args.engine, args.samples)
    if args.textures:
        set_textures(args.textures)
    if args.cmd in ("sequence", "all"):
        run_sequence(args.outdir)
    if args.cmd in ("directions", "all"):
        run_directions(os.path.join(args.outdir, "demo", "direction_check") if args.outdir else None)
    if args.cmd in ("textures", "all"):
        run_texture_sets(os.path.join(args.outdir, "renders", "texture_sets") if args.outdir else None)
    if args.cmd == "move":
        move(args.h, args.v)
        out = args.out or os.path.join(BUNDLE_DIR, "renders", "move_h%+.3f_v%+.3f.png" % (args.h, args.v))
        print(json.dumps(rig().render(out), indent=2), flush=True)


if __name__ == "__main__":
    user_argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else ([] if IN_BLENDER else sys.argv[1:])
    if IN_BLENDER:
        main(user_argv)
    else:
        _relaunch(user_argv)
