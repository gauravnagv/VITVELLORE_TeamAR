"""
make_demo_video.py - builds the submission demo video (demo/demo_video.mp4, ~100 s, 1920x1080)
from the finished outputs: the three input videos, the texture-set renders, the clockwise
sequence with its metadata, the direction check and the Timeline curves of Generate Motion.

    python make_demo_video.py            # needs: pip install pillow numpy imageio-ffmpeg

Run it after the renders exist (motion.py for the videos, eyes.py -- all for the rest).
"""
import json
import os
import subprocess

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "demo", "demo_video.mp4")
W, H, FPS = 1920, 1080, 25
BG = (10, 15, 30)
CARD = (18, 28, 54)
TEXT = (232, 237, 245)
MUTED = (154, 168, 192)
CYAN = (0, 180, 216)
SIDE = 820                 # right-hand render size
IMX, IMY = W - SIDE - 70, 140
MINT = (0, 229, 176)
AMBER = (245, 158, 11)


def font(size, bold=False):
    for f in (("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def mono(size):
    for f in ("consola.ttf", "cour.ttf", "DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return font(size)


def canvas():
    return Image.new("RGB", (W, H), BG)


def header(d, kicker, title):
    d.text((90, 60), kicker, font=font(28, True), fill=CYAN)
    d.text((90, 100), title, font=font(48, True), fill=TEXT)


def lines(d, x, y, items, size=34, gap=16, color=TEXT, bullet=MINT):
    f = font(size)
    for t in items:
        if bullet:
            d.ellipse((x, y + size * 0.38, x + 12, y + size * 0.38 + 12), fill=bullet)
            d.text((x + 30, y), t, font=f, fill=color)
        else:
            d.text((x, y), t, font=f, fill=color)
        y += size + gap
    return y


def read_video(name):
    path = os.path.join(HERE, "videos", name + ".mp4")
    gen = imageio_ffmpeg.read_frames(path)
    meta = next(gen)
    w, h = meta["size"]
    return [np.frombuffer(b, np.uint8).reshape(h, w, 3) for b in gen]


class Writer:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.p = subprocess.Popen(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
             "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path], stdin=subprocess.PIPE)
        self.n = 0

    def put(self, im, times=1):
        b = im.tobytes()
        for _ in range(times):
            self.p.stdin.write(b)
            self.n += 1

    def hold(self, im, seconds):
        self.put(im, int(round(seconds * FPS)))

    def fade(self, a, b, seconds=0.5):
        n = int(seconds * FPS)
        for i in range(n):
            self.put(Image.blend(a, b, (i + 1) / n))

    def close(self):
        self.p.stdin.close()
        self.p.wait()


# --------------------------------------------------------------------------- scenes
def title_card():
    im = canvas()
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((90, 300, 330, 350), 10, fill=(21, 101, 192))
    d.text((112, 309), "TRACK 01", font=font(28, True), fill=TEXT)
    d.text((90, 380), "Synthetic Eye Movement", font=font(84, True), fill=TEXT)
    d.text((90, 480), "& Virtual Camera Rendering", font=font(84, True), fill=TEXT)
    d.text((90, 620), "Team AR  ·  VIT Vellore", font=font(40, True), fill=CYAN)
    d.text((90, 680), "Vagicharla Gaurav Nag  ·  Marimamula Akshay", font=font(34), fill=MUTED)
    d.text((90, 960), "Samsung PRISM XR Hackathon 2026", font=font(28), fill=MUTED)
    return im


def section(kicker, title, sub):
    im = canvas()
    d = ImageDraw.Draw(im)
    d.text((90, 430), kicker, font=font(34, True), fill=AMBER)
    d.text((90, 480), title, font=font(76, True), fill=TEXT)
    d.text((90, 590), sub, font=font(36), fill=MUTED)
    return im


def video_scene(w, name, kicker, title, notes):
    frames = read_video(name)
    base = canvas()
    d = ImageDraw.Draw(base)
    header(d, kicker, title)
    lines(d, 90, 260, notes, size=32, gap=22)
    d.text((90, 960), "videos/%s.mp4  ·  %d frames @ 25 fps  ·  %.1f s  ·  %d² Cycles" % (
        name, len(frames), len(frames) / 25.0, frames[0].shape[0]), font=font(26), fill=MUTED)
    for f in frames:
        im = base.copy()
        im.paste(Image.fromarray(f).resize((SIDE, SIDE), Image.LANCZOS), (IMX, IMY))
        w.put(im)
    return base


def texture_scene(w):
    tdir = os.path.join(HERE, "renders", "texture_sets")
    info = {r["texture_set"]: r for r in json.load(open(os.path.join(tdir, "texture_sets.json")))}
    names = sorted(info)
    uv = Image.open(os.path.join(HERE, "textures", names[0], "face.1001.png")).convert("RGB")
    prev = None
    for n in names:
        im = canvas()
        d = ImageDraw.Draw(im)
        header(d, "UV TEXTURES ON THE GNM HEAD", "10 texture sets, one UV layout")
        pretty = n[6:].replace("_", " ")
        d.text((90, 250), "set %s: %s" % (n[3:5], pretty), font=font(40, True), fill=MINT)
        uv = Image.open(os.path.join(HERE, "textures", n, "face.1001.png")).convert("RGB").resize((360, 360))
        iris = Image.open(os.path.join(HERE, "textures", n, "face.1002.png")).convert("RGB").resize((360, 360))
        im.paste(uv, (90, 330))
        im.paste(iris, (470, 330))
        d.text((90, 700), "face.1001.png  skin (4096²)", font=font(24), fill=MUTED)
        d.text((470, 700), "face.1002.png  left iris", font=font(24), fill=MUTED)
        lines(d, 90, 770, ["Baked onto GNM's own UDIM UVs: fits any GNM head",
                           "Skin albedo, pore normals, roughness, L/R irises",
                           "eyes.set_textures(\"textures/%s\")" % n[:5]], size=28, gap=14)
        im.paste(Image.open(os.path.join(tdir, n + ".png")).convert("RGB").resize((SIDE, SIDE), Image.LANCZOS),
                 (IMX, IMY))
        if prev is not None:
            w.fade(prev, im, 0.4)
        w.hold(im, 1.1)
        prev = im
    return prev


def clockwise_scene(w):
    recs = json.load(open(os.path.join(HERE, "metadata.json")))
    prev = None
    for r in recs:
        im = canvas()
        d = ImageDraw.Draw(im)
        header(d, "ORIGINAL BRIEF", "Clockwise renders + metadata")
        src = Image.open(os.path.join(HERE, "renders", "frame_%02d.png" % r["frame"])).convert("RGB")
        im.paste(src.resize((SIDE, SIDE), Image.LANCZOS), (IMX, IMY))
        dd = ImageDraw.Draw(im)
        for s, col in (("L", (255, 90, 90)), ("R", (80, 200, 255))):
            x, y = r["pupil_center_2D_" + s]
            x, y = IMX + x * SIDE / 1024, IMY + y * SIDE / 1024
            dd.line((x - 14, y, x + 14, y), fill=col, width=3)
            dd.line((x, y - 14, x, y + 14), fill=col, width=3)
        js = ['"frame": %d,' % r["frame"], '"theta": %.4f,' % r["theta"],
              '"input": {"h": %+.3f, "v": %+.3f},' % (r["input"]["h"], r["input"]["v"]),
              '"eye_rotation_L": [%.2f, %.2f, %.2f],' % tuple(r["eye_rotation_L"]),
              '"eye_rotation_R": [%.2f, %.2f, %.2f],' % tuple(r["eye_rotation_R"]),
              '"pupil_center_2D_L": [%.1f, %.1f],' % tuple(r["pupil_center_2D_L"]),
              '"pupil_center_2D_R": [%.1f, %.1f],' % tuple(r["pupil_center_2D_R"]),
              '"camera_matrix": [[%.1f, 0, %.1f], ...]' % (r["camera_matrix"][0][0], r["camera_matrix"][0][2])]
        d.rounded_rectangle((90, 250, 850, 700), 16, fill=(7, 11, 22))
        y = 280
        for t in js:
            d.text((120, y), t, font=mono(28), fill=MINT)
            y += 50
        lines(d, 90, 760, ["θk = 2πk/10,  h = sin θ,  v = cos θ, starting at the top",
                           "Target 35 cm ahead (±5 cm), camera locked 40 cm, 1024²",
                           "Crosses: pupil_center_2D projected with K[R|t]"], size=28, gap=14)
        if prev is not None:
            w.fade(prev, im, 0.25)
        w.hold(im, 0.85)
        prev = im
    return prev


def directions_scene():
    dd = os.path.join(HERE, "demo", "direction_check")
    im = canvas()
    d = ImageDraw.Draw(im)
    header(d, "DIRECTION CHECK", "Left, Right, Up, Down and Reset: 5 / 5 pass")
    for k, n in enumerate(("center", "left", "right", "up", "down", "reset")):
        crop = Image.open(os.path.join(dd, n + ".png")).convert("RGB").crop((200, 280, 830, 500)).resize((540, 189))
        x, y = 90 + (k % 3) * 590, 280 + (k // 3) * 300
        im.paste(crop, (x, y))
        d.text((x, y + 200), n.upper(), font=font(30, True), fill=MINT if n != "center" else MUTED)
    d.text((90, 930), "blender -b scene.blend --python wrapper/eyes.py -- directions", font=mono(30), fill=MUTED)
    return im


def timeline_scene(w):
    frames = read_video("generate")
    st = json.load(open(os.path.join(HERE, "videos", "generate_states.json")))["frames"]
    n = len(st)
    base = canvas()
    d = ImageDraw.Draw(base)
    header(d, "EXPRESSION ON THE TIMELINE", "Generate Motion, keyframed")
    px0, px1, py0 = 90, 980, 250
    tracks = [("gaze yaw (deg)", [s["gaze"][0] for s in st], -30, 30, CYAN),
              ("gaze pitch (deg)", [s["gaze"][1] for s in st], -25, 25, (120, 160, 255)),
              ("blink", [s["blink"] for s in st], 0, 1, MINT),
              ("brows", [s["brow"] for s in st], -1, 1, AMBER),
              ("pupil", [s["pupil"] for s in st], -1, 1, (230, 120, 200)),
              ("smile / squint", [s.get("smile", 0.0) for s in st], 0, 1, (255, 120, 90))]
    th = 100
    for k, (name, vals, lo, hi, col) in enumerate(tracks):
        y0 = py0 + k * (th + 22)
        d.rounded_rectangle((px0, y0, px1, y0 + th), 10, fill=CARD)
        d.text((px0 + 14, y0 + 8), name, font=font(24, True), fill=col)
        pts = [(px0 + 10 + (px1 - px0 - 20) * i / (n - 1), y0 + th - 12 - (th - 44) * (v - lo) / (hi - lo))
               for i, v in enumerate(vals)]
        d.line(pts, fill=col, width=3)
    d.text((90, 1080 - 70), "bones (eyes, neck, head) + 19 shape-key curves: lids, blink, brows, pupil, smile, squint",
           font=font(26), fill=MUTED)
    for i, f in enumerate(frames):
        im = base.copy()
        dd = ImageDraw.Draw(im)
        x = px0 + 10 + (px1 - px0 - 20) * i / (n - 1)
        dd.line((x, py0 - 6, x, py0 + len(tracks) * (th + 22) - 16), fill=TEXT, width=2)
        im.paste(Image.fromarray(f).resize((SIDE, SIDE), Image.LANCZOS), (IMX, IMY))
        w.put(im)


def results_card():
    im = canvas()
    d = ImageDraw.Draw(im)
    header(d, "VERIFIED", "verify.py: geometry checked without rendering")
    rows = [("0.0005 px", "our K[R|t] projection vs Blender's camera"),
            ("0.00005 mm", "eyeballs rotate rigidly about their centres"),
            ("0.0004 mm", "both optical axes pass through the target"),
            ("8 / 8", "pupil shift directions correct, reset exact"),
            ("PASS", "upper lid follows gaze; blink closes the eye")]
    y = 270
    for big, small in rows:
        d.rounded_rectangle((90, y, 1830, y + 120), 14, fill=CARD)
        d.text((130, y + 30), big, font=font(52, True), fill=MINT)
        d.text((560, y + 42), small, font=font(36), fill=TEXT)
        y += 140
    return im


def end_card():
    im = canvas()
    d = ImageDraw.Draw(im)
    d.text((90, 380), "Thank you", font=font(96, True), fill=TEXT)
    d.text((90, 520), "Team AR  ·  VIT Vellore", font=font(44, True), fill=CYAN)
    d.text((90, 590), "Code, renders, videos, texture sets and the .blend files are in the GitHub repository.",
           font=font(32), fill=MUTED)
    d.text((90, 640), "README.md has the setup and run commands.", font=font(32), fill=MUTED)
    return im


def main():
    w = Writer(OUT)
    t = title_card()
    w.fade(canvas(), t, 0.6)
    w.hold(t, 3.5)
    a = section("THE THREE INPUTS", "Left · Right · Generate Motion",
                "wrapper/motion.py renders frames and an mp4 (15 s or less) for each input")
    w.fade(t, a, 0.5)
    w.hold(a, 2.5)
    video_scene(w, "left", "INPUT 1", "eyes input: Left",
                ["Saccades follow the human main sequence", "Head joins large gaze shifts (VOR)",
                 "Gaze-evoked blinks, microsaccades, drift"])
    video_scene(w, "right", "INPUT 2", "eyes input: Right",
                ["Mirror of Left, own blink timing", "Eyes counter-rotate while the head turns",
                 "Lids follow vertical gaze"])
    video_scene(w, "generate", "INPUT 3", "eyes input: Generate Motion",
                ["Glances, an eye-roll with raised brows", "Slow blink, a thinking look down",
                 "Surprise: brows up, pupils dilate", "Reading saccades",
                 "Warm smile: cheeks lift the lower lids"])
    last = texture_scene(w)
    last2 = clockwise_scene(w)
    dirs = directions_scene()
    w.fade(last2, dirs, 0.4)
    w.hold(dirs, 4.0)
    timeline_scene(w)
    r = results_card()
    w.fade(canvas(), r, 0.4)
    w.hold(r, 5.0)
    e = end_card()
    w.fade(r, e, 0.5)
    w.hold(e, 3.5)
    w.close()
    print("wrote %s  (%.1f s)" % (OUT, w.n / FPS))


if __name__ == "__main__":
    main()
