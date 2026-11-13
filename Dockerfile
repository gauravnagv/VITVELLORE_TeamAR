# Track 01 eye wrapper in a container (headless Blender 4.5 LTS, Cycles on CPU, or GPU with --gpus).
#
#   docker build -t track1-eyes .
#   docker run --rm -v "$PWD/out:/out" track1-eyes                                    # 10 clockwise renders + metadata.json -> ./out
#   docker run --rm -v "$PWD/out:/out" track1-eyes eyes directions --outdir /out       # 5-direction check
#   docker run --rm -v "$PWD/out:/out" track1-eyes eyes move 0.5 -0.3 --out /out/pose.png
#   docker run --rm -v "$PWD/out:/out" track1-eyes motion left --out /out --res 512   # Left video (also: right, generate)
#   docker run --rm track1-eyes verify                                                # geometric self-test (no rendering)
#
# EEVEE needs an OpenGL GPU, which a plain container does not have, so the container renders
# with Cycles (CPU, or CUDA/OptiX when run with `--gpus all` on an NVIDIA host).  Geometry,
# pupil positions and metadata are identical to the EEVEE renders; only the shading differs.
FROM ubuntu:24.04

ARG BLENDER_VERSION=4.5.15
ARG BLENDER_SERIES=4.5

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates wget xz-utils \
        libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libxext6 libxkbcommon0 \
        libsm6 libice6 libgl1 libegl1 libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://download.blender.org/release/Blender${BLENDER_SERIES}/blender-${BLENDER_VERSION}-linux-x64.tar.xz \
    && tar -xJf blender-${BLENDER_VERSION}-linux-x64.tar.xz -C /opt \
    && rm blender-${BLENDER_VERSION}-linux-x64.tar.xz \
    && ln -s /opt/blender-${BLENDER_VERSION}-linux-x64/blender /usr/local/bin/blender

WORKDIR /app
# scene.blend has texture set 01 packed; textures/ (other sets) is optional, see .dockerignore
COPY . .
RUN printf '#!/bin/sh\nset -e\ncmd="${1:-eyes}"; [ $# -gt 0 ] && shift\ncase "$cmd" in\n  eyes)   exec blender -b scene.blend --python wrapper/eyes.py -- "$@" --engine CYCLES ;;\n  motion) exec blender -b scene.blend --python wrapper/motion.py -- "$@" --engine CYCLES ;;\n  verify) exec blender -b scene.blend --python verify.py ;;\n  *) echo "usage: eyes|motion|verify ARGS"; exit 2 ;;\nesac\n' > /usr/local/bin/run-track1 \
    && chmod +x /usr/local/bin/run-track1

ENTRYPOINT ["run-track1"]
CMD ["eyes", "sequence", "--samples", "64", "--outdir", "/out"]
