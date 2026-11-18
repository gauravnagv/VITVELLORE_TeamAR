# Team AR: Track 01, Synthetic Eye Movement & Virtual Camera Rendering

**Samsung PRISM XR Hackathon 2026** · Team **AR** · **VIT Vellore** · Vagicharla Gaurav Nag, Marimamula Akshay

Realistic, programmatically controlled eyes on Google's **GNM v3** head in Blender.

**Demo video and all output videos (Google Drive):** https://drive.google.com/drive/folders/10QTUV_ASYQzRfrDaDwr3dUcDB9cF53Qj?usp=sharing
(`VITVellore_AR_demo.mp4` is the 92 s demo; `left`, `right`, `generate` and `generate_closeup.mp4` are the track outputs.)

| Deliverable | Where |
|---|---|
| Three input videos (Left, Right, Generate Motion), each 15 s or less, 2048², Cycles | `videos/left.mp4`, `videos/right.mp4`, `videos/generate.mp4` |
| Cinematic eye close-up of Generate Motion (1920×1080) | `videos/generate_closeup.mp4` |
| 10 UV texture sets on the GNM UV layout + one render per set | `textures/setNN_*/`, `renders/texture_sets/` |
| Expression animated on the Timeline (eyelids, iris/pupil, brows, eyes, head) | `expression_timeline.blend` |
| Python render script with the 3 inputs | `wrapper/motion.py` (on top of the `wrapper/eyes.py` API) |
| Original brief: 10 clockwise renders + metadata, 5-direction check | `renders/frame_00..09.png`, `metadata.json`, `demo/` |
| Presentation | `VITVellore_AR.pptx` |
| Demo video (92 s) | `demo/demo_video.mp4`, also on [Google Drive](https://drive.google.com/drive/folders/10QTUV_ASYQzRfrDaDwr3dUcDB9cF53Qj?usp=sharing) |
| AI disclosure | [`AI_DISCLOSURE.md`](AI_DISCLOSURE.md) |

* **Head model:** GNM v3 (Generative aNthropometric Model, Apache-2.0), with its official UDIM UV layout.
* **Textures:** 10 photographic texture sets wrapped directly onto the GNM UVs. Each set has skin albedo, pore normal map, roughness, separate left and right irises, cornea, teeth and tongue.
* **Eyes:** eyeballs rotate about their true centres and converge on a shared target. Eyelids follow gaze, blinks close the lid fully, and brows and pupils are controllable. Lash geometry moves with the lids.
* **Motion:** a Python controller renders three inputs (**Left**, **Right**, **Generate Motion**) to video. Each is at most 15 s and uses human saccade, blink and head-eye coordination timing.
* **Skin shading:** pore normal maps, a roughness map remapped to a skin-plausible range, subsurface scattering and a faint sheen. Forehead lines, frown lines and crow's feet deepen live with the brow, squint and smile keys.
* **Hair:** real 3D strands (a short buzz cut, eyebrows and fine vellus hair) grown on the skin surface, so they follow the head, brows and lids. Hair colour follows the active texture set.
* **Eye detail:** the eye whites darken and warm toward the lids from the live lid distance (no chalk-white sclera), a thin tear bead sits where the lower lid meets the eye, and the pupil is a true dark aperture with the wet reflection only on the cornea.
* **Rendering:** the submitted videos are rendered with Cycles at 2048² and 128 samples per pixel (the close-up at 1920×1080).
* **Original brief outputs** are also included: 10 clockwise renders with `metadata.json`, a 5-direction check, and a demo video.

## Bundle layout

```
VITVellore_AR.zip               (python package.py; the GitHub repo has the same layout)
├── scene.blend                 # GNM head, armature, lid/brow/pupil shape keys, lashes, camera, lights (textures packed)
├── expression_timeline.blend   # the same scene with the "Generate Motion" expression keyframed on the Timeline
├── wrapper/eyes.py             # eyes.move(h,v) / reset() / blink() / brows() / pupil() / render() / set_textures()
├── wrapper/motion.py           # 3 inputs: left | right | generate  -> frames + mp4 (<= 15 s)
├── textures/setNN_*/           # 10 GNM UDIM texture sets (face.<UDIM>.png + face_normal / face_roughness)
├── videos/                     # left.mp4, right.mp4, generate.mp4, generate_closeup.mp4 (+ per-frame state json)
├── renders/                    # frame_00..09.png (clockwise) + texture_sets/*.png (one render per texture set)
├── metadata.json               # one record per clockwise render
├── demo/                       # contact sheet, direction check, demo video
├── gnm_head/                   # numpy GNM loader, eyelid/brow controls, eyelashes, texture synthesis
├── build_scene_gnm.py          # rebuilds scene.blend from the GNM model (reproducibility)
├── polish_look.py              # final look: lighting, HDRI ambient, skin shader, sclera tone, colour (used by the build)
├── add_expression_keys.py      # smile / squint keys fitted in GNM's expression space (used by the build)
├── add_wrinkles.py, add_hair.py, add_tearline.py, add_sclera_shading.py   # expression wrinkles, 3D hair, tear bead, lid contact shading (used by the build)
├── make_demo_video.py          # builds demo/demo_video.mp4 from the finished outputs
├── bake_skins.py, make_texture_sets.py   # how the texture sets were made (reproducibility)
├── tools_regen_normal.py, tools_regen_eye_tiles.py   # regenerate the shared skin normal map / the iris-sclera tiles
├── verify.py                   # geometric self-test (no rendering)
├── package.py                  # builds the submission zip and checks every deliverable is present
├── Dockerfile, requirements.txt, README.md, VITVellore_AR.pptx
└── legacy/                     # the first (MakeHuman) scene builder, kept for reference only
```

## Setup

* **Blender 4.2 or newer.** Tested on **Blender 4.5.15 LTS** (Windows 11: EEVEE on Intel UHD graphics and on an NVIDIA RTX A2000).
* No add-ons and no pip packages are needed to run the wrapper. It uses Blender's bundled Python (`bpy`, `mathutils`, `numpy`). Everything the scene needs is packed into `scene.blend`.
* You only need the GNM model file (`gnm_head.npz`, from https://huggingface.co/google/gnm-v3) to *rebuild* the scene with `build_scene_gnm.py`.

## Run

From the bundle root:

```bash
# Evaluation inputs: renders the frames and writes an mp4 (<= 15 s) to videos/
blender -b scene.blend --python wrapper/motion.py -- left
blender -b scene.blend --python wrapper/motion.py -- right
blender -b scene.blend --python wrapper/motion.py -- generate
#   options: --res 1024 --fps 25 --samples 24 --textures textures/set05_young_asian_female
#            --engine CYCLES --look c8a   --shot closeup   --save-blend out.blend (keyframed Timeline)
#            --no-render   --encode-only

# Exactly how the submitted videos were rendered (RTX A2000: about 45 s per 2048² frame)
blender -b scene.blend --python wrapper/motion.py -- left --res 2048 --engine CYCLES --samples 128 --look c8a --textures textures/set01_young_caucasian_male
blender -b scene.blend --python wrapper/motion.py -- generate --shot closeup --res 1080 --engine CYCLES --samples 128 --look c8c --textures textures/set01_young_caucasian_male

# Original brief: 10 clockwise renders + metadata.json, and the 5-direction check
blender -b scene.blend --python wrapper/eyes.py -- sequence
blender -b scene.blend --python wrapper/eyes.py -- directions
blender -b scene.blend --python wrapper/eyes.py -- move 0.5 -0.3 --out out.png

# One render per texture set (same pose/camera/lights) -> renders/texture_sets/*.png + texture_sets.json
blender -b scene.blend --python wrapper/eyes.py -- textures

# Timeline expression file (Generate Motion keyframed; no rendering)
blender -b scene.blend --python wrapper/motion.py -- generate --no-render --save-blend expression_timeline.blend

# Same commands from plain Python (finds Blender on PATH, $BLENDER, or pass --blender PATH)
python wrapper/motion.py generate
python wrapper/eyes.py sequence
```

### Docker (optional)

The container renders with Cycles: on the CPU, or on CUDA/OptiX when started with `--gpus all`. EEVEE needs an OpenGL display, which a container does not have.

```bash
docker build -t track1-eyes .
docker run --rm -v "$PWD/out:/out" track1-eyes                                   # 10 clockwise renders + metadata.json
docker run --rm -v "$PWD/out:/out" track1-eyes eyes directions --outdir /out      # 5-direction check
docker run --rm -v "$PWD/out:/out" track1-eyes motion left --out /out --res 512  # Left video (also right, generate)
docker run --rm track1-eyes verify                                               # geometric self-test
```

Interactive use (Blender, Scripting tab):

```python
import sys; sys.path.append(r"<bundle>/wrapper")
import eyes
eyes.move(-1, 0)          # both eyes look to image-left
eyes.blink(0.6)           # lids 60 % closed
eyes.brows(0.8)           # brows raised
eyes.pupil(0.5)           # pupils dilated
eyes.smile(0.8)           # smile (lip corners + cheeks)
eyes.squint(0.6)          # cheeks up, lower lids rise (eyes.squint(0.6, side="L") for one eye)
eyes.reset()
eyes.set_textures("textures/set03_young_african_male")
eyes.render("frame.png")  # returns the metadata dict for this frame
```

To play the expression: open `expression_timeline.blend` and press **Space** in the Timeline.

## Conventions

| Item | Definition |
|---|---|
| Inputs | `h, v ∈ [-1, +1]`, clamped. **h > 0 moves the pupils to image right** (the subject's left). **v > 0 moves them up**. "Left" in the videos means image-left. |
| Fixation | Target = eye midpoint + 35 cm forward + (h·5 cm, v·5 cm) in the camera's right/up axes. |
| Eye rotation | Each eyeball rotates **about its own centre** (GNM eye joint) so its optical axis passes through the target. This is the minimal rotation, with no torsion, and gives physiological vergence. |
| Eyelids | The upper lid follows vertical gaze about 1:1 and the lower lid partially (Becker & Fuchs 1988). Lid skin slides over the globe about the eyeball centre and stays outside the cornea. A blink closes the lids along their whole length (gap ≤ 0.7 mm). |
| Camera | Locked 40 cm in front of the eye midpoint. 70 mm lens, 36 mm sensor, lens shift −0.10, 1024×1024 by default (the submitted videos use `--res 2048`). The close-up (`--shot closeup`) is an 85 mm f/2.8 camera 17 cm from the subject's right eye. |
| Lighting | Fixed for every frame. The submitted videos use the `c8a` look: a small round key light high to camera-left, a very weak fill (hidden from reflections, so it never paints a grey disc on the cornea), two warm rim lights, and a low-strength studio HDRI for ambient bounce (hidden from the camera, which sees a plain dark background). The lights are round discs, so the corneal catch-lights are round. Colour management is AgX (Punchy look). The close-up uses `c8c` (key and right rim only). All values are in `polish_look.py`. |
| Clockwise sequence | `theta_k = 2πk/10`, `h = sin θ`, `v = cos θ`, k = 0 at the top, moving clockwise in the image. |

## Natural motion (wrapper/motion.py)

* **Saccades** follow the human main sequence (duration = 2.2 ms/° × amplitude + 21 ms), with a minimum-jerk profile and a small glissade.
* **Fixations** include slow drift and microsaccades of 0.2–0.5°. The pupil shows hippus (a slow oscillation in size).
* **Head-eye coordination:** gaze shifts larger than about 6° recruit a delayed, damped head turn. Because the eyes keep fixating the world target, they counter-rotate (vestibulo-ocular reflex).
* **Blinks** are asymmetric: about 90 ms to close and 200 ms to open. Blinks often accompany large saccades, and the "Generate Motion" expression includes a slow, exasperated blink. Each blink onset is jittered by up to ±0.12 s, so the Left and Right videos do not blink in lockstep.
* **Pupil:** `eyes.pupil(+1)` is a physiological full dilation (about 65 % of the iris diameter), not the raw GNM key, which leaves only a thin iris ring.
* **Upward gaze** in the expression peaks at about 17°. That keeps the eye inside the range where the upper lid can still lift clear of the iris (about 15° of lid raise relative to the head).
* **Generate Motion** (14 s): glances left and right, an upward eye-roll with raised brows and the head tipped back, a slow blink, a downward "thinking" look with lowered brows and slightly tightened lids, a surprised widening with dilated pupils, reading saccades, then a look back up with a warm smile.
* **Whole-face expressions:** in real faces the eyes never move alone. A genuine (Duchenne) smile raises the cheeks and pushes the lower lids up (FACS AU6 + AU12). The scene therefore has `smile` and `squint_L/R` shape keys, fitted in GNM's own learned expression spaces (`gnm_head/expressions.py`, added by `add_expression_keys.py`). In the motion, every smile automatically brings the matching squint. The lower lashes follow the squint through driven keys.

## Textures (GNM UDIM layout)

Each `textures/setNN_<name>/` folder is a complete set named in GNM's UDIM convention:

| Tile | File | Content |
|---|---|---|
| 1001 | `face.1001.png` (4096²) | Skin albedo with eyebrows, eye-socket conjunctiva and mouth interior |
| 1001 | `face_normal.1001.png`, `face_roughness.1001.png` | Pore and fine-wrinkle normal map; roughness (oily T-zone, glossy lips, wet sockets) |
| 1002 / 1004 | `face.1002.png`, `face.1004.png` (2048²) | **Separate** left and right irises with sclera: radial fibres, crypts, collarette, contraction furrows, limbal ring, soft scleral vessels with a warm blush toward the corners |
| 1003 / 1005 | `face.1003.png`, `face.1005.png` | Cornea (clear) |
| 1006 / 1007 | `face.1006.png`, `face.1007.png` | Teeth and gums, tongue |

The 10 sets cover young, middle-aged and older faces across African, Asian and Caucasian skin, male and female. Iris colours are dark brown, brown, hazel, green, blue and grey.

**How they were made** (`bake_skins.py` → `make_texture_sets.py`):
1. A textured MakeHuman CC0 head is aligned to GNM using eye-centre and nose landmarks, then ICP.
2. GNM's 170 head-identity coefficients are fitted to it. The surfaces then match to a **0.70 mm** median over the face.
3. Each photographic skin and the eyebrows are baked onto GNM's UVs with Cycles.
4. Pore, roughness and iris maps are synthesised in UV space.

Because every GNM identity shares the same UVs, any set wraps directly onto any GNM head.

## metadata.json

One record per render. The required fields are `frame`, `theta`, `input {h, v}`, `eye_rotation_L/R` (degrees, XYZ Euler, world), `pupil_center_2D_L/R` (pixels, origin top-left) and `camera_matrix` (OpenCV K). Extra fields: `camera_extrinsic_world_to_cam`, `projection_matrix`, `pupil_center_3D_*`, `eye_center_3D_*`, `gaze_dir_*`, `gaze_yaw_pitch_deg_*`, `eyelid_pitch_deg_*`, `blink`, `brows`, `pupil`, `head_yaw_pitch_roll_deg`, `fixation_target_3D`. **L is the subject's left eye**, which appears on the image right.

## Validation (`verify.py`, no rendering)

`blender -b scene.blend --python verify.py` runs these checks:

| Check | Result |
|---|---|
| Our `K[R|t]` projection vs Blender `world_to_camera_view` | max error 0.0005 px |
| Deformed eyeballs vs ideal rigid rotation about the eyeball centre | max error 0.00005 mm |
| Optical axis passes through the fixation target | max miss 0.0004 mm |
| Left / Right / Up / Down pupil shift sign (both eyes) | 8 / 8 PASS (±9.2–9.4 px) |
| `reset()` returns exactly to the centre pose | PASS |
| Upper lid follows up / down gaze | +0.98 mm / −1.20 mm, PASS |
| Blink closes the eye | mean gap −0.09 mm, max +0.7 mm, PASS |

## Rebuilding from source (optional)

```bash
blender -b --factory-startup --python build_scene_gnm.py -- \
    --npz path/to/gnm_head.npz --textures textures/set01_young_caucasian_male \
    --identity textures/_bake/fit.json --out scene.blend
```

## Assets & licences

* GNM v3 head model: © Google LLC, **Apache-2.0**. https://github.com/google/GNM. The numpy evaluation in `gnm_head/gnm_model.py` mirrors the official maths.
* MakeHuman system assets (skin photographs, eyebrows): **CC0**. https://static.makehumancommunity.org/
* All other code, the procedural iris, pore and roughness maps, and the eyelashes were made by Team AR for this hackathon.

## Limitations

* Every texture set is shown on the same generic GNM head shape. Fitting a new GNM identity per set would also need the rig, hair and tear bead re-fitted.
* The hair is a short buzz cut only; longer styles are not modelled.
* Under the single key light the tear bead reads as small glints along the lower lid rather than a continuous wet line.
* Cycles at 2048² takes about 45 s per frame on an RTX A2000. EEVEE (the default engine) renders in about 1 s per 1024² frame, with less accurate skin subsurface scattering.
