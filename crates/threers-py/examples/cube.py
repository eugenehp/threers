#!/usr/bin/env python3
"""
Spinning-cube setup — the three.js "hello world", headless.

Same structure as a minimal three.js app; Python uses the native bindings
for offscreen PNG export. For the full THREE.* API in the browser, use the
npm package (`import THREE from 'threers'`).

three.js (JavaScript)                    →  threers (Python)
─────────────────────────────────────────────────────────────
const scene = new THREE.Scene();         →  scene = Scene()
scene.background = new THREE.Color(0x…);   →  scene.set_background(0x…)
new THREE.BoxGeometry(1,1,1)             →  scene.add_box(1, 1, 1, color=0x…)
new THREE.PerspectiveCamera(50, …)       →  PerspectiveCamera(50, …)
camera.position.set(3, 2, 5)             →  camera.set_position(3, 2, 5)
camera.lookAt(0, 0, 0)                   →  camera.look_at(0, 0, 0)
renderer.render(scene, camera)           →  renderer.render_png(scene, camera)
"""

from __future__ import annotations

import sys
from pathlib import Path

import threers


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "cube.png")

    scene = threers.Scene()
    scene.set_background(0x101010)
    scene.add_box(1.0, 1.0, 1.0, color=0xFF6633, y=0.0)

    camera = threers.PerspectiveCamera(50.0, 16.0 / 9.0, 0.1, 100.0)
    camera.set_position(3.0, 2.0, 5.0)
    camera.look_at(0.0, 0.0, 0.0)

    if threers.HeadlessRenderer is None:
        raise SystemExit("HeadlessRenderer not built — maturin develop --features headless")

    renderer = threers.HeadlessRenderer(800, 600)
    png = renderer.render_png(scene, camera)
    out.write_bytes(png)
    print(f"wrote {out} ({len(png)} bytes)")


if __name__ == "__main__":
    main()
