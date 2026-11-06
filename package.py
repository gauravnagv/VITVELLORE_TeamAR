"""
package.py - assemble <team_name>.zip in the exact layout the evaluators expect.

    python package.py TEAM_NAME
"""
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REQUIRED = ["scene.blend", "wrapper/eyes.py", "metadata.json", "README.md",
            "wrapper/demo_live.py", "verify.py", "build_scene.py"]
OPTIONAL_FILES = ["Dockerfile", ".dockerignore", "requirements.txt"]
OPTIONAL_DIRS = ["renders", "demo"]


def main():
    team = sys.argv[1] if len(sys.argv) > 1 else "Team_AR"
    missing = [p for p in REQUIRED if not os.path.isfile(os.path.join(HERE, p))]
    frames = [f for f in os.listdir(os.path.join(HERE, "renders")) if f.startswith("frame_")] \
        if os.path.isdir(os.path.join(HERE, "renders")) else []
    if missing or len(frames) != 10:
        sys.exit("Not ready: missing=%s, clockwise frames=%d/10" % (missing, len(frames)))

    out = os.path.join(HERE, "dist", team + ".zip")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in REQUIRED:
            z.write(os.path.join(HERE, p), p)
        for p in OPTIONAL_FILES:
            if os.path.isfile(os.path.join(HERE, p)):
                z.write(os.path.join(HERE, p), p)
        for d in OPTIONAL_DIRS:
            for root, _, files in os.walk(os.path.join(HERE, d)):
                for f in files:
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, HERE).replace("\\", "/")
                    if d == "renders" and not f.startswith("frame_"):
                        continue  # only the 10 clockwise frames go in renders/
                    z.write(full, rel)
    print("wrote", out)
    with zipfile.ZipFile(out) as z:
        for n in z.namelist():
            print("  ", n)


if __name__ == "__main__":
    main()
