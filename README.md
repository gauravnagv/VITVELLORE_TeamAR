# Team AR: Track 01, Synthetic Eye Movement & Virtual Camera Rendering

A textured, rigged human head in Blender. A Python wrapper rotates both eyeballs about their
true centres so they fixate a common target 35 cm ahead. A locked camera 40 cm ahead renders
1024×1024 PNGs, and every frame comes with ground-truth metadata (3D rotations and projected
2D pupil centres).

## Bundle layout

```
Team_AR.zip
├── scene.blend          # textured head mesh, eye bones, materials (all textures packed)
├── wrapper/eyes.py      # eyes.move(h, v) / eyes.reset() / eyes.render()
├── wrapper/demo_live.py # optional on-screen demo (used for the demo video)
├── renders/             # frame_00.png … frame_09.png  (clockwise, theta_k = 2*pi*k/10)
├── metadata.json        # one record per render
├── demo/                # contact_sheet.png, direction_check.png, direction_check/*.png, demo_video.mp4
├── verify.py            # geometric self-test (optional)
├── build_scene.py       # how scene.blend was generated (optional, for reproducibility)
├── Dockerfile           # headless run in a container (Blender 4.5 LTS + Cycles CPU)
├── requirements.txt     # no pip packages needed for the wrapper (see file)
└── README.md
```

## Setup

### Option A: Blender on your PC

* **Blender 4.2 or newer.** Tested with **Blender 4.5.15 LTS** on Windows 11, EEVEE on Intel UHD graphics.
* No add-ons are needed. The MakeHuman assets are baked into `scene.blend` and all textures are packed.
* No extra Python packages. The wrapper uses only Blender's bundled Python (`bpy`, `mathutils`).
  `requirements.txt` lists only the optional `imageio-ffmpeg`, used to re-record the demo video.

### Option B: Docker (no local Blender install)

```bash
docker build -t track1-eyes .
docker run --rm -v "$PWD/out:/out" track1-eyes          # 10 clockwise renders + metadata.json -> ./out
docker run --rm -v "$PWD/out:/out" track1-eyes directions --engine CYCLES --samples 32 --outdir /out
docker run --rm -v "$PWD/out:/out" track1-eyes move 0.5 -0.3 --engine CYCLES --out /out/pose.png
docker run --rm --entrypoint blender track1-eyes -b scene.blend --python verify.py   # self-test
```

The image downloads Blender 4.5.15 LTS for Linux. A plain container has no OpenGL GPU, so it
renders with **Cycles on the CPU** instead of EEVEE. Eye rotations, 2D pupil centres and the
metadata are identical; only the shading looks slightly different.

## Run

From the bundle root:

```bash
# Regenerate the 10 clockwise renders + metadata.json  (default command)
blender -b scene.blend --python wrapper/eyes.py -- sequence

# 5-direction check: center, left, right, up, down, reset  -> demo/direction_check/
blender -b scene.blend --python wrapper/eyes.py -- directions

# Any single pose
blender -b scene.blend --python wrapper/eyes.py -- move 0.5 -0.3 --out out.png

# Same commands from plain Python. The wrapper finds Blender (PATH, $BLENDER or the
# default install folder) and relaunches itself inside it.
python wrapper/eyes.py sequence
python wrapper/eyes.py move 1 0 --blender "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe"
```

Options: `--engine EEVEE|CYCLES` (default EEVEE; CYCLES uses a GPU if one is present),
`--samples N`, `--outdir DIR`.

Interactive use inside Blender (Scripting tab → Python console):

```python
import sys; sys.path.append(r"<bundle>/wrapper")
import eyes
eyes.move(1, 0)        # look right (image right)
eyes.move(0, -1)       # look down
eyes.reset()           # centre
eyes.render("my_frame.png")   # returns the metadata dict for that frame
```

Live demo: `blender scene.blend --python wrapper/demo_live.py` (recorded in `demo/demo_video.mp4`, 88 s).

## Conventions

| Item | Definition |
|---|---|
| Inputs | `h, v ∈ [-1, +1]`, clamped. **h > 0 moves the pupils to image right** (the subject's left), **v > 0 moves them up**. |
| Fixation | Target = eye midpoint + 35 cm forward + (h·5 cm, v·5 cm) in the camera's right/up directions (motion bounds ±5 cm). |
| Eye rotation | Each eyeball rotates **about its own centre** so its optical axis passes through the target: the minimal rotation from the modelled rest axis, with no torsion. This gives physiological **vergence**: at `reset()` both eyes converge about 4.8° inward on the 35 cm target. |
| Camera | Locked 40 cm in front of the eye midpoint, looking straight at the face. 70 mm lens on a 36 mm sensor, vertical lens shift −0.10 so the whole face is in frame. 1024×1024. |
| Lighting | Fixed 3-point area lights + constant world: identical for every frame. Renders are deterministic. |
| Clockwise sequence | `theta_k = 2πk/10`, `k = 0..9`, `h = sin(theta)`, `v = cos(theta)`. k = 0 is straight up, k = 1 moves toward image right (clockwise as seen in the image). |

Note: the brief's example record shows `v = 0.309` for frame 3. `cos(1.8849) = −0.309`, and
metadata.json stores the mathematically correct value.

## metadata.json

A JSON array with one record per render. These are the required fields:

```jsonc
{
  "frame": 3,
  "theta": 1.884956,                       // theta_k = 2*pi*k/10
  "input": { "h": 0.951057, "v": -0.309017 },
  "eye_rotation_L": [rx, ry, rz],          // degrees, XYZ Euler, world frame, relative to the modelled rest pose
  "eye_rotation_R": [rx, ry, rz],
  "pupil_center_2D_L": [x, y],             // pixels, origin top-left, x right, y down
  "pupil_center_2D_R": [x, y],
  "camera_matrix": [[fx,0,cx],[0,fy,cy],[0,0,1]]   // OpenCV intrinsic K
}
```

Extra fields are provided for verification: `camera_extrinsic_world_to_cam` (OpenCV `[R|t]`),
`projection_matrix` (`P = K[R|t]`), `pupil_center_3D_*`, `eye_center_3D_*`, `gaze_dir_*`,
`gaze_yaw_pitch_deg_*` (gaze angles vs. straight ahead: yaw + = image right, pitch + = up),
`fixation_target_3D`, `units` and `image`. Here **L = the subject's left eye**, which appears on the
image right. The pupil point is the exact 3D spot on the iris where the eye material draws the pupil
centre (UV 0.295/0.3 and 0.705/0.7, read from the material), rotated with the eyeball and projected
with the camera. `eye_rotation_*` is the rotation actually applied to the eyeball, relative to the
modelled MakeHuman rest pose (whose eyes splay slightly outward). For gaze labels, use `gaze_yaw_pitch_deg_*`.
Multiplying `projection_matrix` by `[pupil_center_3D, 1]` reproduces `pupil_center_2D`.

## Validation (`verify.py`, no rendering needed)

`blender -b scene.blend --python verify.py` runs these checks over 17 gaze inputs:

| Check | Result |
|---|---|
| Our `K[R|t]` projection vs Blender `world_to_camera_view` | max error 0.0005 px |
| Deformed eye mesh vs ideal rigid rotation about the eyeball centre | max error 0.0004 mm |
| Optical axis passes through the fixation target | max miss 0.0016 mm |
| Left / Right / Up / Down pupil shift sign (both eyes) | 8 / 8 PASS (±9.7–9.9 px) |
| `reset()` returns exactly to the centre pose | PASS |

## How scene.blend was built (reproducible)

`build_scene.py` creates the scene from scratch with the MPFB 2 add-on (MakeHuman for Blender) and the
**MakeHuman system assets (CC0)**: base mesh, `young_caucasian_male` skin, high-poly eyes with the
MPFB procedural eye material, eyebrow001, eyelashes01 and short02 hair, using MPFB's `default` rig
with `eye.L` / `eye.R` bones. It then:

1. fits a sphere (least squares) to each eyeball and moves the eye-bone heads to the fitted centres,
2. skins each eyeball 100 % to its eye bone, so a bone rotation is exactly a rotation about the eyeball centre,
3. adds the fixation-target empty, the locked camera and the fixed lights, and packs all textures.

```bash
blender -b --command extension install-file -r user_default -e mpfb.zip   # once
blender -b --python build_scene.py -- --out scene.blend
```

## Assets & licences

* MakeHuman / MPFB 2 base mesh and system assets: **CC0**. https://static.makehumancommunity.org/
* All code in this bundle was written by Team AR for this hackathon.
