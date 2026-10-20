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
# add_box uses MeshStandardMaterial: as in three.js it renders black until lit.
scene.add_ambient_light(0xFFFFFF, 0.4)
scene.add_directional_light(0xFFFFFF, 2.5, position=(3, 5, 4))

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

### Quads

Grid geometries carry their quads (each two-triangle cell is one polygon), so
wireframes, `Geometry.edges` and picking work on quads, not triangle halves:

```python
from threers import Geometry, Material

plane = Geometry.plane(2, 2, 4, 4)          # three.js PlaneGeometry
plane.quad_count()                          # 16
plane.polygon_counts()                      # [2, 2, …] triangles per polygon
plane.polygon_loops()[0]                    # [a, b, c, d] vertex indices
plane.polygon_edges()                       # [(a, b), …] quad outline edges
plane.positions(), plane.index()            # the vertex data they index

soup = Geometry.from_arrays(positions, indices)
soup.infer_quads()                          # pair triangles back into quads (40°/40°)
soup.set_polygons([2, 2, 1])                # or group them yourself
smooth = plane.subdivide(2)                 # Catmull–Clark -> a new all-quad Geometry

# Quad wireframes without diagonals: as a material, or as line segments.
scene.add_mesh(plane, Material.basic(color=0xFFFFFF, wireframe=True))
scene.add_line_segments(Geometry.edges(Geometry.box(1, 1, 1)), Material.line(color=0x000000))
scene.add_line_segments(Geometry.wireframe(plane), Material.line(color=0x00FF00))

# 12 numbers, or 4 points (Vector3 or (x, y, z)), in winding order.
scene.add_quad([-2, 0, -2, 2, 0, -2, 2, 0, 2, -2, 0, 2], color=0x33AA55,
               width_segments=8, height_segments=8)

# Which quad is under the cursor? (NDC -1..1, +y up)
for hit in scene.raycast_from_camera(camera, 0.1, -0.2):
    print(hit["handle"], hit["polygon_index"], hit["face_index"], hit["point"], hit["uv"])

# three.js Raycaster settings: first_hit_only, Line/Points thresholds,
# near/far, and intersectObjects over a list of handles.
scene.raycast((0, 0, 5), (0, 0, -1), first_hit_only=True, line_threshold=0.05,
              objects=[group], recursive=True)
```

A hit is a dict of three.js's intersection fields in snake_case —
`distance`, `point`, `face_index`, `face` (`a`, `b`, `c`, `normal`,
`material_index`), `normal`, `uv`, `uv1`, `instance_id`, `index` (lines),
`distance_to_ray` (points) — plus `handle` and the threers extensions
`polygon_index` and `barycoord`; fields that do not apply to what was hit are
`None`.

OBJ, PLY, FBX and USD keep their quads (and n-gons) on load; `get_geometry`
returns a mesh's geometry and `meshes` the handles of a loaded model's parts:

```python
model = scene.load("part.obj")              # .glb .gltf .obj .stl .ply .fbx .usd[a|c|z]
scene.get_geometry(model).quad_count()
parts = scene.meshes(scene.load("assembly.fbx"))
```

### Textures

`Texture` is three.js's `Texture`: RGBA8 pixels (rows top first) or a PNG, with
three.js's sampling state. `set_map` puts it in any slot three.js names:

```python
from threers import Texture

wood = Texture.from_file("wood.png")                     # sRGB colour
rough = Texture.from_file("wood_rough.png", "linear")    # data
wood.wrap_s = wood.wrap_t = "repeat"                    # "repeat" | "mirror" | "clamp"
wood.repeat, wood.rotation, wood.center = (4, 4), 0.3, (0.5, 0.5)
wood.min_filter, wood.mipmap_filter = "linear", "nearest"   # mipmap_filter None = level 0 only
rough.channel = 1                                       # read uv1

floor = Material.physical(color=0xFFFFFF, sheen=0.4, sheen_color=0xFFE0C0)
floor.set_map("map", wood)
floor.set_map("roughnessMap", rough)
floor.set_map("clearcoatMap", Texture(1, 1, bytes([128, 0, 0, 255]), "linear"))
floor.set_map("roughnessMap", None)                     # empty the slot
```

A material keeps the texture state it was given: edit the `Texture` and call
`set_map` again to change it.

### Quad view

A CAD-style quad view — top / front / side orthographic panes plus the
perspective camera, framed on the scene's meshes:

```python
sheet = renderer.render_quad_view(scene, camera)                    # PNG bytes
front = renderer.render_quad_view(scene, camera, maximized="front")
rgba = renderer.render_quad_view(
    scene, camera, format="rgba",
    layout=["top", "perspective", "front", "side"],   # grid order, TL TR BL BR
    gap=4, border=8, gap_color=0x303030, margin=0.1,
    backgrounds={"perspective": 0x1D2330},            # per-pane, over the scene background
)
```

`set_viewport(x, y, w, h)`, `set_scissor(x, y, w, h)` and
`set_scissor_test(on)` restrict `render_png` / `render_rgba` to a rectangle
(output pixels, bottom-left origin, as in three.js); `reset_viewport()` undoes
it. With `set_auto_clear(False)` each render draws over the last, so
`clear(color)` once and render several viewports to compose one image.

### Scene API

| Area | Methods |
|------|---------|
| Meshes | `add_mesh(geometry, material, x, y, z)`, `add_line_segments(geometry, line_material)`, `add_group()`, `set_parent(child, parent=None)`, `add_box` / `add_sphere` / `add_quad` |
| Geometry | `Geometry.box` `plane` `sphere` `cylinder` `cone` `torus` `torus_knot` `ring` `circle` `capsule` `lathe` `tube` `icosahedron` `octahedron` `tetrahedron` `dodecahedron` `quad` `from_arrays` `edges` `wireframe` (three.js parameters and defaults) |
| Materials | `Material.basic` `standard` `physical` `lambert` `phong` `toon` `normal` `line` — keyword-only `color, emissive, emissive_intensity, roughness, metalness, opacity, transparent, wireframe, side, specular, shininess, clearcoat, clearcoat_roughness, transmission, ior, thickness, depth_test`, and MeshPhysicalMaterial's `specular_intensity, specular_color, sheen, sheen_color, sheen_roughness, iridescence, iridescence_ior, iridescence_thickness_range, anisotropy, anisotropy_rotation, attenuation_color, attenuation_distance, dispersion`; keywords a material does not use raise `TypeError`. `material.set_map(slot, texture)` / `get_map(slot)` fill three.js's map slots |
| Lights | `add_ambient_light(color, intensity)`, `add_directional_light(color, intensity, position, cast_shadow)`, `add_point_light(color, intensity, position, distance, decay, cast_shadow)`, `add_spot_light(color, intensity, position, target, angle, penumbra, distance, decay, cast_shadow)`, `add_hemisphere_light(sky_color, ground_color, intensity)`, `light_count()` — three.js r155+ physical units, defaults intensity 1, decay 2 |
| Handles | `set_position` `set_rotation` (radians, XYZ) `set_scale` `set_visible` `set_cast_shadow` `set_receive_shadow` `set_material` `get_position` `get_geometry` `meshes` `remove` `object_count` |
| Loading | `load(path, material=None)`, `load_bytes(data, format, material=None)` |
| Picking | `raycast(origin, direction, near=0, far=inf, *, first_hit_only, line_threshold, points_threshold, recursive, objects)`, `raycast_from_camera(camera, x, y, *, near, far, …same keywords)` |
| Cameras | `PerspectiveCamera`, `OrthographicCamera(left, right, top, bottom, near, far)` + `set_frustum` |
| Renderer | `render_png` / `render_rgba` (either camera), `render_quad_view`, `set_size`, `set_tone_mapping("none" \| "linear" \| "aces" \| "neutral" \| "agx", exposure)`, `set_shadows_enabled`, viewport / scissor / `set_auto_clear` / `clear` |

Bad values raise `ValueError`, wrong types `TypeError`, a missing file
`FileNotFoundError`. Type stubs ship in `threers/_native.pyi` (`py.typed`).

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

## Tests

```bash
maturin develop --features headless,physics,animation
pytest tests
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
  python/threers/     pure-Python package surface + `_native.pyi` stubs
  src/lib.rs          PyO3 module `threers._native`
```
