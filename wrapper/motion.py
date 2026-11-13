"""
motion.py - natural eye-expression animation for the Track 01 GNM scene.

Three inputs (the evaluation brief), each rendered to frames and an mp4 (<= 15 s):
    left      both eyes move to the LEFT of the image, naturally (blinks, head follow, return)
    right     both eyes move to the RIGHT of the image
    generate  a realistic expression: glances left/right, an upward eye-roll with raised
              brows and a slow blink, a downward "thinking" look, a surprised widening,
              a few reading saccades, back to rest

    blender -b scene.blend --python wrapper/motion.py -- left --out videos
    blender -b scene.blend --python wrapper/motion.py -- generate --save-blend expression_timeline.blend
    python wrapper/motion.py right                     # relaunches itself inside Blender
Options: --fps 25  --res 1024  --samples 24  --engine EEVEE|CYCLES  --seed N
         --no-render (keyframe + save only)   --textures DIR   --no-caption

How the motion is made (all deterministic for a given seed):
  * Saccades follow the human "main sequence": duration = 2.2 ms/deg * amplitude + 21 ms,
    minimum-jerk velocity profile and a small glissade (overshoot that settles).
  * Fixations carry slow drift and occasional microsaccades (0.2-0.5 deg).
  * Large gaze shifts recruit the head (slight, delayed, critically damped); because the
    eyes keep fixating the world target they counter-rotate (vestibulo-ocular reflex).
  * Eyelids follow vertical gaze (eyes.py); blinks are asymmetric (fast close ~90 ms,
    slower open ~200 ms) and often accompany large saccades.
  * Brows and pupil change with the expression; pupil shows small hippus oscillation.
The motion is baked as keyframes on the Timeline (bones + shape keys), so the saved .blend
plays the expression directly in Blender.
"""
import json
import math
import os
import sys

WRAPPER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WRAPPER_DIR)
import eyes  # noqa: E402

try:
    import bpy
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

GAZE_DISTANCE = 0.6   # m, fixation distance for free gaze in the videos


# --------------------------------------------------------------------------- #
# Motion plans
# --------------------------------------------------------------------------- #
def plan(mode):
    """Return (duration_s, events).  Gaze angles in degrees, yaw + = image right, pitch + = up."""
    ev = []

    def fix(t, yaw, pitch):
        ev.append(("saccade", t, yaw, pitch))

    def arc(t0, t1, pts):
        ev.append(("pursuit", t0, t1, pts))

    def blink(t, close=0.09, hold=0.04, open_=0.20, amount=1.0):
        ev.append(("blink", t, close, hold, open_, amount))

    def brow(t0, t1, value, ramp=0.25):
        ev.append(("brow", t0, t1, value, ramp))

    def pupil(t0, t1, value, ramp=0.4):
        ev.append(("pupil", t0, t1, value, ramp))

    def smile(t0, t1, value, ramp=0.45):
        ev.append(("smile", t0, t1, value, ramp))

    def squint(t0, t1, value, ramp=0.3):
        ev.append(("squint", t0, t1, value, ramp))

    if mode in ("left", "right"):
        s = -1.0 if mode == "left" else 1.0
        blink(0.55)
        fix(1.20, s * 10, 1.0)            # first small glance
        brow(1.1, 2.2, 0.12)
        blink(2.02)                       # gaze-evoked blink with the big shift
        fix(2.05, s * 27, -1.5)
        fix(3.30, s * 30, 2.0)            # corrective / exploring saccade
        pupil(2.1, 4.8, 0.25)
        blink(4.70)
        fix(5.40, s * 4, 0.0)             # glance back toward centre
        fix(6.30, s * 20, -3.0)           # and back to the side
        blink(7.60, amount=1.0)
        fix(8.20, 0.0, 0.0)               # settle at rest
        return 9.0, ev
    if mode == "generate":
        blink(0.45)
        blink(0.98)
        fix(1.00, -22, 0.0)               # glance left
        fix(2.10, 22, 2.0)                # glance right
        fix(3.15, 0.0, 0.0)               # back
        # upward eye-roll: brows up, smooth arc over the top.  Peak kept at ~17 deg so the
        # upper lid (max raise ~15 deg relative to the head) can still clear the iris.
        brow(3.8, 6.7, 0.75, ramp=0.35)
        fix(3.95, -15, 11)
        arc(4.15, 5.35, [(-15, 11), (-8, 15.5), (0, 17), (8, 15.5), (15, 11)])
        fix(5.55, 6, -2)
        blink(6.10, close=0.16, hold=0.18, open_=0.38)   # exasperated slow blink
        # look down, "thinking"
        fix(7.00, -4, -18)
        brow(7.0, 8.4, -0.35)
        squint(7.1, 8.4, 0.3)             # concentration: slight lid tightening
        fix(8.15, 3, -16)
        # surprise: widen, pupils dilate, look up to centre
        brow(8.75, 10.2, 0.9, ramp=0.12)
        pupil(8.7, 11.2, 0.8, ramp=0.25)
        fix(8.80, 0, 4)
        blink(9.65)
        blink(9.95, close=0.08, hold=0.02, open_=0.16)
        # reading saccades left-to-right, with a return sweep
        for i, x in enumerate((-9, -3, 3, 9)):
            fix(10.6 + 0.32 * i, x, -6)
        fix(12.0, -9, -9)
        fix(12.3, -2, -9)
        fix(12.9, 0, 0)
        # look up from reading with a warm (Duchenne) smile: cheeks lift, lower lids rise
        smile(12.55, 14.8, 0.85)
        blink(13.35)
        return 14.0, ev
    raise ValueError("mode must be left, right or generate")


# --------------------------------------------------------------------------- #
# Synthesis
# --------------------------------------------------------------------------- #
def _min_jerk(x):
    x = max(0.0, min(1.0, x))
    return x * x * x * (10 - 15 * x + 6 * x * x)


def _smooth(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


class Rand:
    """Tiny deterministic LCG (no numpy dependency in the wrapper)."""

    def __init__(self, seed):
        self.s = (seed * 2654435761 + 12345) & 0xFFFFFFFF

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return self.s / 2 ** 32

    def n(self):
        return math.sqrt(-2 * math.log(max(self.u(), 1e-12))) * math.cos(2 * math.pi * self.u())


def synthesize(mode, fps=25, seed=0):
    """Per-frame state dicts: gaze (yaw,pitch), head (yaw,pitch,roll), blink, brow, pupil."""
    dur, ev = plan(mode)
    n = int(round(dur * fps))
    rnd = Rand(seed + {"left": 1, "right": 2, "generate": 3}[mode])
    # spontaneous blinks are not metronomic: jitter each blink onset by up to +/-0.12 s
    # (separate stream, so gaze drift/microsaccades are unchanged by this)
    # (a deliberate double blink keeps its spacing: blinks with a neighbour within 0.6 s are not moved)
    jit = Rand(seed + {"left": 11, "right": 23, "generate": 37}[mode])
    bt = [e[1] for e in ev if e[0] == "blink"]
    ev = [(e[0], e[1] + 0.24 * (jit.u() - 0.5)) + tuple(e[2:])
          if e[0] == "blink" and not any(0 < abs(e[1] - o) < 0.6 for o in bt) else e for e in ev]
    moves = sorted([e for e in ev if e[0] in ("saccade", "pursuit")], key=lambda e: e[1])

    # ---- gaze: saccades and pursuit arcs in time order (+ drift / microsaccades below)
    def base_gaze(t):
        g = (0.0, 0.0)
        for e in moves:
            if t < e[1]:
                break
            if e[0] == "pursuit":           # smooth arc through the given points
                _, t0, t1, pts = e
                if t >= t1:
                    g = pts[-1]
                    continue
                u = (t - t0) / (t1 - t0) * (len(pts) - 1)
                i = min(int(u), len(pts) - 2)
                f = _smooth(u - i)
                return (pts[i][0] + (pts[i + 1][0] - pts[i][0]) * f, pts[i][1] + (pts[i + 1][1] - pts[i][1]) * f)
            _, ts, yaw, pitch = e
            amp = math.hypot(yaw - g[0], pitch - g[1])
            d = (2.2 * amp + 21.0) / 1000.0     # main sequence
            if t < ts + d:                       # in flight: min-jerk with ~4 % overshoot
                frac = 1.04 * _min_jerk((t - ts) / d)
            else:                                # glissade settles over ~30 ms
                frac = 1.0 + 0.04 * math.exp(-(t - ts - d) / 0.03)
            g = (g[0] + (yaw - g[0]) * frac, g[1] + (pitch - g[1]) * frac)
            if t >= ts + d + 0.2:                # settled: snap exactly onto the target
                g = (yaw, pitch)
        return g

    drift = [0.0, 0.0]
    micro = [0.0, 0.0]
    next_micro = 0.4 + rnd.u()
    states = []
    head = [0.0, 0.0, 0.0]
    head_v = [0.0, 0.0, 0.0]
    dt = 1.0 / fps
    for f in range(n):
        t = f * dt
        g = base_gaze(t)
        # fixational drift (bounded random walk) + microsaccades
        for k in range(2):
            drift[k] = 0.92 * drift[k] + 0.06 * rnd.n()
        if t >= next_micro:
            micro = [0.35 * rnd.n(), 0.25 * rnd.n()]
            next_micro = t + 0.5 + 1.0 * rnd.u()
        gaze = (g[0] + drift[0] + micro[0], g[1] + drift[1] + micro[1])

        # head recruitment: delayed, damped follow of the large part of the gaze shift
        gd = base_gaze(max(0.0, t - 0.06))
        tgt = [math.copysign(max(0.0, abs(gd[0]) - 6.0) * 0.42, gd[0]),
               math.copysign(max(0.0, abs(gd[1]) - 5.0) * 0.30, gd[1]),
               -0.08 * gd[0] * 0.3]
        w = 7.0  # rad/s, critically damped second-order follow
        for k in range(3):
            acc = w * w * (tgt[k] - head[k]) - 2 * w * head_v[k]
            head_v[k] += acc * dt
            head[k] += head_v[k] * dt

        # blink
        b = 0.0
        for e in ev:
            if e[0] != "blink":
                continue
            _, t0, c, hld, o, amt = e
            if t0 <= t < t0 + c:
                b = max(b, amt * ((t - t0) / c) ** 2)
            elif t0 + c <= t < t0 + c + hld:
                b = max(b, amt)
            elif t0 + c + hld <= t < t0 + c + hld + o:
                x = (t - t0 - c - hld) / o
                b = max(b, amt * (1 - x) ** 2 * (1 + 0.5 * x))
        # brows / pupil / smile / squint
        br = sm = sq = 0.0
        pu = 0.06 * math.sin(2 * math.pi * 0.3 * t + 1.3) + 0.03 * math.sin(2 * math.pi * 0.71 * t)
        for e in ev:
            if e[0] in ("brow", "pupil", "smile", "squint"):
                _, t0, t1, val, ramp = e
                if t0 <= t <= t1:
                    k = _smooth((t - t0) / ramp) * _smooth((t1 - t) / ramp)
                    if e[0] == "brow":
                        br += val * k
                    elif e[0] == "pupil":
                        pu += val * k
                    elif e[0] == "smile":
                        sm += val * k
                    else:
                        sq += val * k
        sq += 0.7 * sm      # a genuine smile always raises the cheeks (AU6 with AU12)
        # eyes widen slightly when looking up (brow follows strong up-gaze)
        br += 0.25 * _smooth((g[1] - 10) / 12.0)
        states.append({"t": t, "gaze": gaze, "head": tuple(head), "blink": min(1.0, b),
                       "brow": max(-1.0, min(1.0, br)), "pupil": max(-1.0, min(1.0, pu)),
                       "smile": max(0.0, min(1.0, sm)), "squint": max(0.0, min(1.0, sq))})
    return states, dur


# --------------------------------------------------------------------------- #
# Baking to the Timeline
# --------------------------------------------------------------------------- #
def apply_state(r, st):
    r.blink_amt = {"L": st["blink"], "R": st["blink"]}
    r.set_head(*st["head"], keep_gaze=False)
    r.look(st["gaze"][0], st["gaze"][1], GAZE_DISTANCE)
    r.brows(st["brow"])
    r.pupil(st["pupil"])
    r.smile(st.get("smile", 0.0))
    r.squint(st.get("squint", 0.0))


def bake(states, fps):
    scn = bpy.context.scene
    r = eyes.rig()
    arm, head = r.arm, r.head
    arm.animation_data_clear()
    if head.data.shape_keys:
        head.data.shape_keys.animation_data_clear()
    scn.render.fps = fps
    scn.frame_start, scn.frame_end = 1, len(states)
    bones = [arm.pose.bones[b] for b in r.cfg.get("head_bones", ["neck", "head"]) + [r.cfg["bone_L"], r.cfg["bone_R"]]]
    for pb in bones:
        pb.rotation_mode = "QUATERNION"
    keys = [k for k in head.data.shape_keys.key_blocks[1:]]
    for f, st in enumerate(states, start=1):
        apply_state(r, st)
        for pb in bones:
            pb.keyframe_insert("rotation_quaternion", frame=f)
            pb.keyframe_insert("location", frame=f)
        for k in keys:
            k.keyframe_insert("value", frame=f)
    for idb in (arm, head.data.shape_keys):
        ad = idb.animation_data
        if ad and ad.action:
            for fc in _fcurves(ad):
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"
    scn.frame_set(1)
    r.reset()     # rest pose for scripting; playback/render use the keyframes


def _fcurves(ad):
    act = ad.action
    if hasattr(act, "fcurves") and len(getattr(act, "fcurves", [])):
        return act.fcurves
    out = []   # Blender 4.4+ layered actions
    for layer in getattr(act, "layers", []):
        for strip in layer.strips:
            for cb in strip.channelbags:
                out.extend(cb.fcurves)
    return out


# --------------------------------------------------------------------------- #
# Rendering + encoding (Blender's built-in FFmpeg; no extra dependencies)
# --------------------------------------------------------------------------- #
def render_frames(frames_dir, res, samples, motion_blur=True):
    scn = bpy.context.scene
    os.makedirs(frames_dir, exist_ok=True)
    scn.render.resolution_x = scn.render.resolution_y = res
    scn.render.resolution_percentage = 100
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGB"
    scn.render.filepath = os.path.join(frames_dir, "frame_")
    scn.render.use_motion_blur = motion_blur
    scn.render.motion_blur_shutter = 0.5
    if scn.render.engine.startswith("BLENDER_EEVEE"):
        scn.eevee.taa_render_samples = samples
    else:
        scn.cycles.samples = samples
    scn.render.use_overwrite = False      # resumable: existing frames are skipped
    scn.render.use_placeholder = True
    bpy.ops.render.render(animation=True)


def encode(frames_dir, n, fps, res, out_mp4, caption=None):
    files = sorted(f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(".png"))[:n]
    enc = bpy.data.scenes.new("Encode")
    enc.render.fps = fps
    enc.render.resolution_x = enc.render.resolution_y = res
    enc.render.resolution_percentage = 100
    enc.frame_start, enc.frame_end = 1, len(files)
    se = enc.sequence_editor_create()
    strips = getattr(se, "strips", None) or se.sequences
    s = strips.new_image("frames", os.path.join(frames_dir, files[0]), channel=1, frame_start=1)
    for f in files[1:]:
        s.elements.append(f)
    if caption:
        try:   # Blender 4.x: frame_end (exclusive); 5.x: length
            tx = strips.new_effect("caption", "TEXT", channel=2, frame_start=1, frame_end=len(files) + 1)
        except TypeError:
            tx = strips.new_effect("caption", "TEXT", channel=2, frame_start=1, length=len(files))
        tx.text = caption
        tx.font_size = max(16, res // 36)
        tx.location = (0.5, 0.03)
        tx.anchor_x, tx.anchor_y = "CENTER", "BOTTOM"
        tx.use_box = True
        tx.box_color = (0, 0, 0, 0.45)
        tx.color = (1, 1, 1, 1)
    enc.render.image_settings.file_format = "FFMPEG"
    enc.render.ffmpeg.format = "MPEG4"
    enc.render.ffmpeg.codec = "H264"
    enc.render.ffmpeg.constant_rate_factor = "HIGH"
    enc.render.ffmpeg.ffmpeg_preset = "GOOD"
    enc.render.filepath = out_mp4
    enc.render.use_file_extension = False
    enc.view_settings.view_transform = "Standard"   # frames are already display-referred
    bpy.ops.render.render(animation=True, scene=enc.name)
    bpy.data.scenes.remove(enc)
    print("[motion] video ->", out_mp4, flush=True)


# --------------------------------------------------------------------------- #
def _parse(argv):
    import argparse
    p = argparse.ArgumentParser(prog="motion.py")
    p.add_argument("mode", choices=["left", "right", "generate"])
    p.add_argument("--out", default=os.path.join(eyes.BUNDLE_DIR, "videos"))
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--res", type=int, default=1024)
    p.add_argument("--samples", type=int, default=24)
    p.add_argument("--engine", default="EEVEE")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--textures", default=None)
    p.add_argument("--save-blend", default=None, help="save the keyframed scene (Timeline animation)")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--encode-only", action="store_true", help="only (re)build the mp4 from existing frames")
    p.add_argument("--no-blur", action="store_true")
    p.add_argument("--no-caption", action="store_true")
    p.add_argument("--blender", default=None)
    return p.parse_args(argv)


def main(argv):
    a = _parse(argv)
    a.out = os.path.abspath(a.out)   # Blender resolves bare relative paths against the drive root
    eyes.set_engine(a.engine, a.samples)
    if a.textures:
        eyes.set_textures(a.textures)
    states, dur = synthesize(a.mode, a.fps, a.seed)
    assert dur <= 15.0, "videos must not exceed 15 s"
    cap = None if a.no_caption else "eyes input: %s" % {"left": "Left", "right": "Right",
                                                           "generate": "Generate Motion"}[a.mode]
    frames = os.path.join(a.out, "%s_frames" % a.mode)
    if a.encode_only:
        encode(frames, len(states), a.fps, a.res, os.path.join(a.out, "%s.mp4" % a.mode), cap)
        return
    bake(states, a.fps)
    with open(os.path.join(_mkdir(a.out), "%s_states.json" % a.mode), "w") as f:
        json.dump({"mode": a.mode, "fps": a.fps, "duration_s": dur, "frames": states}, f)
    if a.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.save_blend), copy=True, compress=True)
        print("[motion] saved Timeline animation ->", a.save_blend, flush=True)
    if a.no_render:
        return
    render_frames(frames, a.res, a.samples, motion_blur=not a.no_blur)
    encode(frames, len(states), a.fps, a.res, os.path.join(a.out, "%s.mp4" % a.mode), cap)


def _mkdir(d):
    os.makedirs(d, exist_ok=True)
    return d


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else ([] if IN_BLENDER else sys.argv[1:])
    if IN_BLENDER:
        main(argv)
    else:
        import subprocess
        args = _parse(argv)
        exe = eyes._find_blender(args.blender)
        if args.blender:
            i = argv.index("--blender")
            argv = argv[:i] + argv[i + 2:]
        cmd = [exe, "-b", os.path.join(eyes.BUNDLE_DIR, "scene.blend"), "--python", os.path.abspath(__file__),
               "--"] + argv
        print("[motion] launching:", " ".join(cmd), flush=True)
        sys.exit(subprocess.call(cmd))
