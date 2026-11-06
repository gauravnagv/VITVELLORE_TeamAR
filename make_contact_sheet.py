"""
make_contact_sheet.py - builds demo/contact_sheet.png and demo/direction_check.png
from renders + metadata.json.  Needs Pillow (pip install pillow).

    python make_contact_sheet.py [bundle_dir]
"""
import json
import math
import os
import sys

from PIL import Image, ImageDraw, ImageFont

BUNDLE = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)))
BG = (24, 26, 30)
FG = (230, 232, 236)
MUTED = (150, 155, 165)
MARK_L = (255, 80, 80)
MARK_R = (80, 200, 255)


def font(size):
    for f in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def cross(d, x, y, col, r=10, w=2):
    d.line((x - r, y, x + r, y), fill=col, width=w)
    d.line((x, y - r, x, y + r), fill=col, width=w)
    d.ellipse((x - 3, y - 3, x + 3, y + 3), outline=col, width=1)


def eye_crop(img, rec, ref, width=560, height=170, out_w=420):
    """Crop around the rest-pose pupil midpoint so all crops share one window."""
    mx = (ref["pupil_center_2D_L"][0] + ref["pupil_center_2D_R"][0]) / 2
    my = (ref["pupil_center_2D_L"][1] + ref["pupil_center_2D_R"][1]) / 2
    box = (int(mx - width / 2), int(my - height / 2), int(mx + width / 2), int(my + height / 2))
    s = out_w / width
    crop = img.crop(box).resize((out_w, int(height * s)), Image.LANCZOS)
    d = ImageDraw.Draw(crop)
    # reference (rest) pupil positions: grey; current: coloured
    for key in ("pupil_center_2D_L", "pupil_center_2D_R"):
        x, y = ref[key]
        cross(d, (x - box[0]) * s, (y - box[1]) * s, (160, 160, 160), r=6, w=1)
    for key, col in (("pupil_center_2D_L", MARK_L), ("pupil_center_2D_R", MARK_R)):
        x, y = rec[key]
        cross(d, (x - box[0]) * s, (y - box[1]) * s, col)
    return crop


def dial(h, v, size=70):
    im = Image.new("RGB", (size, size), BG)
    d = ImageDraw.Draw(im)
    c = size / 2
    d.ellipse((4, 4, size - 4, size - 4), outline=MUTED, width=1)
    d.line((c, c, c + h * (c - 8), c - v * (c - 8)), fill=FG, width=3)
    d.ellipse((c + h * (c - 8) - 4, c - v * (c - 8) - 4, c + h * (c - 8) + 4, c - v * (c - 8) + 4), fill=FG)
    return im


def tile(rec, ref, title, sub):
    img = Image.open(os.path.join(BUNDLE, rec["image"])).convert("RGB")
    W = 420
    thumb = img.resize((W, W), Image.LANCZOS)
    td = ImageDraw.Draw(thumb)
    s = W / img.width
    for key, col in (("pupil_center_2D_L", MARK_L), ("pupil_center_2D_R", MARK_R)):
        x, y = rec[key]
        cross(td, x * s, y * s, col, r=6, w=1)
    crop = eye_crop(img, rec, ref)
    t = Image.new("RGB", (W, 60 + W + crop.height + 8), BG)
    d = ImageDraw.Draw(t)
    d.text((8, 6), title, fill=FG, font=font(22))
    d.text((8, 34), sub, fill=MUTED, font=font(15))
    t.paste(dial(rec["input"]["h"], rec["input"]["v"], 52), (W - 58, 4))
    t.paste(thumb, (0, 60))
    t.paste(crop, (0, 60 + W + 8))
    return t


def sheet(tiles, cols, header, out):
    gap = 14
    tw, th = tiles[0].size
    rows = math.ceil(len(tiles) / cols)
    H = 90
    foot = 40
    S = Image.new("RGB", (cols * tw + (cols + 1) * gap, H + rows * th + (rows + 1) * gap + foot), BG)
    d = ImageDraw.Draw(S)
    d.text((gap, 18), header, fill=FG, font=font(30))
    d.text((gap, 56), "red = left-eye pupil, blue = right-eye pupil (projected ground truth); grey = rest pose. "
           "Lower strip: fixed eye-region crop, 0.75x zoom of the 1024px render.", fill=MUTED, font=font(16))
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        S.paste(t, (gap + c * (tw + gap), H + gap + r * (th + gap)))
    d.text((gap, S.height - foot + 8), "h > 0 = pupils to image right, v > 0 = up.  Fixation target 35 cm, camera 40 cm, "
           "target moves within +/-5 cm.", fill=MUTED, font=font(15))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    S.save(out)
    print("wrote", out)


def main():
    meta = json.load(open(os.path.join(BUNDLE, "metadata.json")))
    dpath = os.path.join(BUNDLE, "demo", "direction_check", "directions.json")
    dirs = json.load(open(dpath)) if os.path.isfile(dpath) else None
    ref = next((r for r in (dirs or []) if r.get("name") == "center"), None) or {
        "pupil_center_2D_L": [sum(r["pupil_center_2D_L"][i] for r in meta) / len(meta) for i in (0, 1)],
        "pupil_center_2D_R": [sum(r["pupil_center_2D_R"][i] for r in meta) / len(meta) for i in (0, 1)],
    }
    tiles = [tile(r, ref, "frame %d" % r["frame"],
                  "theta=%.3f  h=%+.3f  v=%+.3f" % (r["theta"], r["input"]["h"], r["input"]["v"])) for r in meta]
    sheet(tiles, 5, "Clockwise gaze sequence  theta_k = 2*pi*k/10,  h = sin theta,  v = cos theta",
          os.path.join(BUNDLE, "demo", "contact_sheet.png"))
    if dirs:
        tiles = [tile(r, ref, r["name"].upper(), "h=%+.0f  v=%+.0f" % (r["input"]["h"], r["input"]["v"])) for r in dirs]
        sheet(tiles, 6, "Direction check: center, left, right, up, down, reset",
              os.path.join(BUNDLE, "demo", "direction_check.png"))


if __name__ == "__main__":
    main()
