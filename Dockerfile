# Track 01 eye wrapper in a container (headless Blender, Cycles on CPU).
#
#   docker build -t track1-eyes .
#   docker run --rm -v "$PWD/out:/out" track1-eyes                      # 10 clockwise renders + metadata.json -> ./out
#   docker run --rm -v "$PWD/out:/out" track1-eyes directions --engine CYCLES --samples 32 --outdir /out
#   docker run --rm -v "$PWD/out:/out" track1-eyes move 0.5 -0.3 --engine CYCLES --out /out/pose.png
#   docker run --rm --entrypoint blender track1-eyes -b scene.blend --python verify.py   # geometric self-test
#
# EEVEE needs an OpenGL GPU, which a plain container does not have, so the
# container renders with Cycles (CPU). Geometry, pupil positions and metadata
# are identical to the EEVEE renders; only the shading differs slightly.
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
COPY scene.blend metadata.json README.md verify.py build_scene.py ./
COPY wrapper/eyes.py wrapper/eyes.py

ENTRYPOINT ["blender", "-b", "scene.blend", "--python", "wrapper/eyes.py", "--"]
CMD ["sequence", "--engine", "CYCLES", "--samples", "64", "--outdir", "/out"]
