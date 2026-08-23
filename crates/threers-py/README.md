# threers (Python)

Python bindings for [threers](https://github.com/eugenehp/threers) via
[PyO3](https://pyo3.rs) / [maturin](https://www.maturin.rs).

```bash
pip install threers          # when published
# or from this crate:
maturin develop --features headless,physics,animation
```

## Quick start

```python
import threers

# Same scene graph as three.js — see examples/cube.py for the full mapping.
scene = threers.Scene()
scene.set_background(0x202030)
scene.add_box(1, 1, 1, color=0xff6633, y=0.5)

camera = threers.PerspectiveCamera(50, 16 / 9, 0.1, 100)
camera.set_position(3, 2, 5)
camera.look_at(0, 0, 0)

renderer = threers.HeadlessRenderer(1280, 720)
png = renderer.render_png(scene, camera)
open("out.png", "wb").write(png)
```

For the **full THREE.* API** (browser / npm), use the JavaScript package:
`import THREE from 'threers'`. Example: [`../threers-js/examples/cube.html`](../threers-js/examples/cube.html).

```bash
python examples/cube.py cube.png
```

Physics:

```python
world = threers.PhysicsWorld()
ball = world.add_ball(0.5, y=5.0)
for _ in range(120):
    world.step(1 / 60)
print(world.body_translation(ball))
```

Animation:

```python
tween = threers.Tween(0.0, 1.0, 0.5, easing="cubicOut")
while not tween.is_finished():
    tween.update(1 / 60)
print(tween.value())
```

## Build / publish

```bash
cd crates/threers-py
# editable install for local work
maturin develop --features headless,physics,animation

# wheels for PyPI
maturin build --release --features headless,physics,animation
maturin publish --features headless,physics,animation
```

Requires a working Rust toolchain and, for `HeadlessRenderer`, a GPU that wgpu
can open (Vulkan / Metal / DX12 / GLES).

## Architectures

Wheels are built per platform (see
[`.github/workflows/release-language-packages.yml`](../../.github/workflows/release-language-packages.yml)):

| Host | Target |
|------|--------|
| macOS Apple Silicon | `aarch64-apple-darwin` |
| macOS Intel | `x86_64-apple-darwin` |
| Linux x86_64 | `x86_64-unknown-linux-gnu` |
| Linux arm64 | `aarch64-unknown-linux-gnu` |
| Windows x64 | `x86_64-pc-windows-msvc` |

```bash
maturin build --release --features headless,physics,animation --target aarch64-apple-darwin
```

Language binding overview: [`docs/bindings.md`](../../docs/bindings.md).

## Layout

```text
crates/threers-py/
  Cargo.toml          cdylib + features
  pyproject.toml      maturin / PyPI metadata
  python/threers/     pure-Python package surface
  src/lib.rs          PyO3 module `threers._native`
```
