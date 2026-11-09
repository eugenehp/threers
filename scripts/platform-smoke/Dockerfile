# What test-linux-builds.sh runs the Linux addon and wheel in: Node 24, the
# distribution's Python 3, and Mesa's software Vulkan (lavapipe), so a
# container with no GPU still has a real adapter to render on.
FROM node:24-bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-pip mesa-vulkan-drivers libvulkan1 \
 && rm -rf /var/lib/apt/lists/*
ENV WGPU_BACKEND=vulkan
