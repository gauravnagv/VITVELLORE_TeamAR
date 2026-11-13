"""
add_expression_keys.py - adds the whole-face expression shape keys (smile, squint_L,
squint_R; see gnm_head/expressions.py) to an existing scene.blend, with matching eyelash
keys driven 1:1 from the head keys.  Idempotent: existing keys of the same name are replaced.

    blender -b scene.blend --python add_expression_keys.py -- [--npz PATH] [--identity textures/_bake/fit.json] [--save]

build_scene_gnm.py calls add_keys() too, so a rebuild produces the same keys.
"""
import argparse
import json
import os
import sys

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "gnm_head"))
from gnm_model import GNMHead, EyeControls  # noqa: E402
import eyelashes  # noqa: E402
import expressions  # noqa: E402

KEYS = ("smile", "squint_L", "squint_R")


def to_blender(p):
    p = np.asarray(p, dtype=np.float64)
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)


def _replace_key(ob, name, co):
    kb = ob.data.shape_keys.key_blocks.get(name) if ob.data.shape_keys else None
    if kb is not None:
        ob.shape_key_remove(kb)
    sk = ob.shape_key_add(name=name, from_mix=False)
    sk.data.foreach_set("co", co.astype(np.float32).ravel())
    sk.slider_min, sk.slider_max = 0.0, 1.0
    return sk


def add_keys(head, ec, head_ob, lash_ob=None):
    report = {}
    deltas = {}
    d, err = expressions.smile(ec)
    deltas["smile"] = d
    report["smile_fit_err_mm"] = round(float(err), 3)
    for s in "LR":
        d, err = expressions.squint(ec, s)
        deltas["squint_" + s] = d
        report["squint_%s_cheek_fit_err_mm" % s] = round(float(err), 3)
    base = to_blender(head.bind)
    for name, d in deltas.items():
        _replace_key(head_ob, name, base + to_blender(d))
        report[name + "_max_mm"] = round(float(np.linalg.norm(d, axis=1).max() * 1e3), 2)

    if lash_ob is not None:
        # regenerate the lash strands exactly as the build did (same seeds) to get roots
        per_side, offset = {}, 0
        for s in "LR":
            v, q, roots, lov = eyelashes.generate(head, ec, s, seed=7 if s == "L" else 8)
            per_side[s] = (v, roots, lov, offset)
            offset += len(v)
        if offset != len(lash_ob.data.vertices):
            raise RuntimeError("eyelash mesh does not match the regenerated strands (%d vs %d)"
                               % (offset, len(lash_ob.data.vertices)))
        lb = np.zeros(offset * 3, np.float32)
        lash_ob.data.shape_keys.key_blocks["Basis"].data.foreach_get("co", lb)
        lb = lb.reshape(-1, 3).astype(np.float64)
        head_keys = head_ob.data.shape_keys
        for name in ("squint_L", "squint_R"):
            s = name[-1]
            v, roots, lov, off = per_side[s]
            d = np.zeros((offset, 3))
            d[off:off + len(v)] = eyelashes.follow(head, ec, s, v, roots, lov, deltas[name])
            sk = _replace_key(lash_ob, name, lb + to_blender(d))
            fc = sk.driver_add("value")
            drv = fc.driver
            drv.type = "AVERAGE"
            var = drv.variables.new()
            var.type = "SINGLE_PROP"
            var.targets[0].id_type = "KEY"
            var.targets[0].id = head_keys
            var.targets[0].data_path = 'key_blocks["%s"].value' % name
    return report


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default=os.path.join(HERE, "..", "tools", "gnm", "model", "gnm_head.npz"))
    p.add_argument("--identity", default=os.path.join(HERE, "textures", "_bake", "fit.json"))
    p.add_argument("--save", action="store_true")
    a = p.parse_args(argv)
    js = json.load(open(a.identity))
    identity = np.array(js["identity"] if isinstance(js, dict) else js, dtype=float)
    head = GNMHead(os.path.abspath(a.npz), identity)
    ec = EyeControls(head)
    cfg = json.loads(bpy.context.scene["track1_config"])
    head_ob = bpy.data.objects[cfg["head_mesh"]]
    basis = np.zeros(len(head_ob.data.vertices) * 3, np.float32)
    head_ob.data.shape_keys.key_blocks["Basis"].data.foreach_get("co", basis)
    err = np.abs(basis.reshape(-1, 3) - to_blender(head.bind)).max()
    if err > 1e-5:
        raise RuntimeError("scene mesh does not match this GNM identity (max %.2e m)" % err)
    report = add_keys(head, ec, head_ob, bpy.data.objects.get("Eyelashes"))
    cfg.setdefault("build_report", {}).update(report)
    cfg["expression_keys"] = list(KEYS)
    bpy.context.scene["track1_config"] = json.dumps(cfg)
    print("[expr] added", KEYS, json.dumps(report), flush=True)
    if a.save:
        bpy.ops.wm.save_mainfile(compress=True)
        print("[expr] saved", bpy.data.filepath, flush=True)


if __name__ == "__main__":
    main()
