# threers

A drop-in **three.js replacement** for Rust and the browser, on
[wgpu](https://github.com/gfx-rs/wgpu). One Rust core runs natively, headless,
and as WebAssembly on WebGPU or WebGL2. A JavaScript shim exposes the `THREE.*`
API, so existing three.js code runs with an import change.

[![crates.io](https://img.shields.io/crates/v/threers.svg)](https://crates.io/crates/threers)
[![docs.rs](https://docs.rs/threers/badge.svg)](https://docs.rs/threers)
· **0.0.8** · [Changelog](CHANGELOG.md) · MIT

## Quick start

**Rust**: `cargo add threers`, then render a frame to a PNG with no window:

```rust
use threers::prelude::*;

fn main() {
    let mut scene = Scene::new();
    scene.add(Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.0, 1.0, 1.0),
        Material::Standard(StandardMaterial::new(Color::from_hex(0xff8844))),
    )));
    let mut sun = DirectionalLight::new(Color::WHITE, 3.0);
    sun.direction = Vector3::new(-1.0, -2.0, -1.5).normalize();
    scene.add_light(sun);
    scene.add_light(AmbientLight::new(Color::WHITE, 0.4));

    let mut camera = PerspectiveCamera::new(50.0, 16.0 / 9.0, 0.1, 100.0);
    camera.position = Vector3::new(3.0, 2.0, 4.0);
    camera.look_at(Vector3::ZERO);

    let mut renderer = HeadlessRenderer::builder().size(1280, 720).build().unwrap();
    renderer.set_msaa(4); // antialiasing: off by default, as in three.js
    let rgba = renderer.render_to_rgba(&mut scene, &camera);
    std::fs::write("cube.png", threers::encode_png(1280, 720, &rgba)).unwrap();
}
```

**Browser**: `npm i threers`, and keep writing three.js:

```js
import THREE, { initThreers } from 'threers';

await initThreers();
const renderer = await THREE.WebGLRenderer.create(canvas, { antialias: true });
const scene = new THREE.Scene();
scene.add(new THREE.Mesh(
  new THREE.BoxGeometry(1, 1, 1),
  new THREE.MeshStandardMaterial({ color: 0xff8844 }),
));
scene.add(new THREE.DirectionalLight(0xffffff, 3));
scene.add(new THREE.AmbientLight(0xffffff, 0.4));
const camera = new THREE.PerspectiveCamera(50, 16 / 9, 0.1, 100);
camera.position.set(3, 2, 4);
camera.lookAt(0, 0, 0);
renderer.render(scene, camera);
```

The shader does its own tone mapping and output encoding, because Chrome's
WebGPU canvas is `bgra8unorm` rather than sRGB. Use the canvas's preferred
format and leave colour to threers ([Web](docs/web.md)). Python and Node
bindings: [docs/bindings.md](docs/bindings.md).

## What's in it

The core is unconditional: scene graph, geometry, PBR materials, lights and
shadows, custom WGSL, post-processing, controls, animation, and loaders for
glTF, OBJ, STL, PLY, FBX, Collada, HDR and EXR. Each of the following is a
self-contained subsystem, behind a Cargo feature where it says so.

| | |
|---|---|
| [Materials](docs/materials.md) | Clearcoat, transmission, dispersion, anisotropy, sheen, iridescence; HDR pipeline, IBL, area lights |
| [Fur and hair](docs/materials.md#fur) | Shell fur and strand hair, on static or skinned meshes: wind, drag and inertia, shadows, MSAA per strand, a hand that strokes them; `THREE.Fur` and `THREE.Hair` in the shim, demo in `web/examples/fur.html` |
| [Particles, snow and sand](docs/particles.md) | Emitters with drag, turbulence and buoyancy; snow and sand that pile up at their angle of repose, drift and form dunes; lit, shadowed, black-body fire and smoke; GPU weather; dune seas shaded after Journey's sand; `THREE.Particles`, `THREE.SandBed` and `THREE.DesertSky` in the shim, demos in `web/examples/particles.html` and `dunes.html` |
| [Interactive effects](docs/fx.md) · [SDF](docs/sdf.md) | Glossy liquids that react to the pointer on buttons and any shape (`Venom`, `Gloss`), as components, effects and a stage you can add to; exact 2D/3D signed distance fields with smooth operators, ray marching and meshing, the same in WGSL |
| [Colour](docs/colors.md) | Colour spaces (OkLCh, CIELAB, HSV, …), every CSS Color 4 string, ColorBrewer and Tableau schemes, d3-style log, quantile and threshold scales |
| [Marks](docs/marks.md) | Data layers for maps, from points, paths and polygons to labels, heatmaps and tiles. `f64` precision, GPU filtering and picking, `MapView` and globe controls |
| [Headless](docs/headless-render.md) | Offscreen rendering to RGBA, with no window or display; choosing the GPU, and testing on NVIDIA under Linux |
| [Deploying](docs/deploying.md) | A static site for Vercel, GitHub Pages, Netlify or Cloudflare: hashed wasm, preloading, caching headers, sizes |
| [Web](docs/web.md) · [WebGL2](docs/webgl.md) (`webgl`) | WebAssembly on WebGPU, or WebGL2 where WebGPU is missing |
| [Path tracing](docs/path-tracing.md) (`raytrace`) | Global illumination, refraction and depth of field, on CPU and GPU, with denoising |
| [Fluid](docs/fluid.md) (`fluid`) | 2D and 3D fluids as render passes, so they run on WebGL2 too |
| [Planets](docs/planets.md) (`planet`) | Atmospheres, starfields, Earth from NASA imagery |
| [Mesh BVH and CSG](docs/mesh-bvh-csg.md) (`mesh-bvh`, `bvh-csg`) | Accelerated raycasts, and booleans with three-bvh-csg parity |
| [OpenSCAD](docs/openscad.md) (`openscad`) | `.scad` front end, `Solid` DSL, watertight exact-CSG kernel |
| [NURBS, B-rep, STEP](docs/brep-nurbs-plan.md) (`nurbs`, `brep`, `step`) | Exact curved surfaces, analytic booleans, ISO 10303-21 |
| [Lattices](docs/lattices.md) · [Kirigami](docs/kirigami.md) | 43 lattice generators — TPMS, struts, textiles, infill, foams on random, crystal, relaxed and two-size seeds, grown grains — with physical properties, print checks and simplification; foldable sheets |
| [USD](docs/usd.md) (`usd`) · [glTF](docs/gltf.md) · [Draco](docs/draco.md) · [KTX2](docs/ktx2.md) · [Alembic](docs/alembic.md) | Interchange, read and written where the format allows |
| [Video](docs/codecs.md) (`video`, `native-codec`, `gif`) · [Captions](docs/captions.md) · [SVG](docs/svg.md) | GIF, APNG, WebM, H.264 and HEVC in pure Rust; SRT/WebVTT; vector output |
| [Metal and visionOS](docs/apple-platforms.md) (`metal`, `visionos`) | A second renderer on Metal, and the CompositorServices loop |
| [RLX](docs/rlx.md) (`rlx`, `rlx-geo`) · [Physics](docs/physics.md) | Tensors and learned grading; rigid bodies in `threers-physics` |

Bundles turn on related features together: `cad`, `media`, `gi`, `apple`,
`full` (native) and `wasm-full` (browser). The full table is in the
[crate docs](https://docs.rs/threers). Long form: [docs/features.md](docs/features.md).

## Examples

<a href="docs/lattices.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/lattice_foams.jpg" width="820" alt="Voronoi foams on six seed layouts, open, closed and wet"></a>
<br><sub>Voronoi foams on random, Kelvin, Weaire–Phelan, C15, relaxed and two-size seeds — open, closed and wet (<code>lattice_foams</code>; see <a href="docs/lattices.md">docs/lattices.md</a>).</sub>

76 examples render a picture; each is in [docs/examples.md](docs/examples.md)
with the command that makes it. They all run headless, so they work the same
on a laptop and in CI.

| | | | |
|---|---|---|---|
| <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/backend_parity.jpg" width="200" alt="backend_parity"></a><br><sub><code>backend_parity</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/campfire.jpg" width="200" alt="campfire"></a><br><sub><code>campfire</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/captions_render.jpg" width="200" alt="captions_render"></a><br><sub><code>captions_render</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/cuboct.jpg" width="200" alt="cuboct"></a><br><sub><code>cuboct</code></sub> |
| <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/data_charts.jpg" width="200" alt="data_charts"></a><br><sub><code>data_charts</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/denoise_ab.jpg" width="200" alt="denoise_ab"></a><br><sub><code>denoise_ab</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/denoise_compare.jpg" width="200" alt="denoise_compare"></a><br><sub><code>denoise_compare</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/dunes.jpg" width="200" alt="dunes"></a><br><sub><code>dunes</code></sub> |
| <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/earth_sun_flare.jpg" width="200" alt="earth_sun_flare"></a><br><sub><code>earth_sun_flare</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/fluid3d.jpg" width="200" alt="fluid3d"></a><br><sub><code>fluid3d</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/fluid_render.jpg" width="200" alt="fluid_render"></a><br><sub><code>fluid_render</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/fur.jpg" width="200" alt="fur"></a><br><sub><code>fur</code></sub> |
| <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/fur_closeup.jpg" width="200" alt="fur_closeup"></a><br><sub><code>fur_closeup</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/fur_motion.jpg" width="200" alt="fur_motion"></a><br><sub><code>fur_motion</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/fur_stroke.jpg" width="200" alt="fur_stroke"></a><br><sub><code>fur_stroke</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/glass_modes.jpg" width="200" alt="glass_modes"></a><br><sub><code>glass_modes</code></sub> |
| <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/hair.jpg" width="200" alt="hair"></a><br><sub><code>hair</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/hdri_lighting.jpg" width="200" alt="hdri_lighting"></a><br><sub><code>hdri_lighting</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/ik_servo_demo.jpg" width="200" alt="ik_servo_demo"></a><br><sub><code>ik_servo_demo</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/kirigami.jpg" width="200" alt="kirigami"></a><br><sub><code>kirigami</code></sub> |
| <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/lattice.jpg" width="200" alt="lattice"></a><br><sub><code>lattice</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/lattice_engineering.jpg" width="200" alt="lattice_engineering"></a><br><sub><code>lattice_engineering</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/lattice_foams.jpg" width="200" alt="lattice_foams"></a><br><sub><code>lattice_foams</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/lattice_print.jpg" width="200" alt="lattice_print"></a><br><sub><code>lattice_print</code></sub> |
| <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/lattice_shapes.jpg" width="200" alt="lattice_shapes"></a><br><sub><code>lattice_shapes</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/lattice_textiles.jpg" width="200" alt="lattice_textiles"></a><br><sub><code>lattice_textiles</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/map_demo.jpg" width="200" alt="map_demo"></a><br><sub><code>map_demo</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/marks_map.jpg" width="200" alt="marks_map"></a><br><sub><code>marks_map</code></sub> |
| <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/material_chart.jpg" width="200" alt="material_chart"></a><br><sub><code>material_chart</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/metal_headless.jpg" width="200" alt="metal_headless"></a><br><sub><code>metal_headless</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/metal_stereo.jpg" width="200" alt="metal_stereo"></a><br><sub><code>metal_stereo</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/nema17.jpg" width="200" alt="nema17"></a><br><sub><code>nema17</code></sub> |
| <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/neon_logo.jpg" width="200" alt="neon_logo"></a><br><sub><code>neon_logo</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/ocean_water.jpg" width="200" alt="ocean_water"></a><br><sub><code>ocean_water</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/openscad_gallery.jpg" width="200" alt="openscad_gallery"></a><br><sub><code>openscad_gallery</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/openscad_render.jpg" width="200" alt="openscad_render"></a><br><sub><code>openscad_render</code></sub> |
| <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/origami.jpg" width="200" alt="origami"></a><br><sub><code>origami</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/origami_gallery.jpg" width="200" alt="origami_gallery"></a><br><sub><code>origami_gallery</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/origami_miura.jpg" width="200" alt="origami_miura"></a><br><sub><code>origami_miura</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/origami_twist.jpg" width="200" alt="origami_twist"></a><br><sub><code>origami_twist</code></sub> |
| <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/particles.jpg" width="200" alt="particles"></a><br><sub><code>particles</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/path_trace.jpg" width="200" alt="path_trace"></a><br><sub><code>path_trace</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/path_trace_gpu.jpg" width="200" alt="path_trace_gpu"></a><br><sub><code>path_trace_gpu</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/pole_probe.jpg" width="200" alt="pole_probe"></a><br><sub><code>pole_probe</code></sub> |
| <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/postfx_chain.jpg" width="200" alt="postfx_chain"></a><br><sub><code>postfx_chain</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/printer_assembly.jpg" width="200" alt="printer_assembly"></a><br><sub><code>printer_assembly</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/progressive_denoise.jpg" width="200" alt="progressive_denoise"></a><br><sub><code>progressive_denoise</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/quad_view.jpg" width="200" alt="quad_view"></a><br><sub><code>quad_view</code></sub> |
| <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/raspberry.jpg" width="200" alt="raspberry"></a><br><sub><code>raspberry</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/realistic_earth.jpg" width="200" alt="realistic_earth"></a><br><sub><code>realistic_earth</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/retrofuturism.jpg" width="200" alt="retrofuturism"></a><br><sub><code>retrofuturism</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/rlx_conv_postfx.jpg" width="200" alt="rlx_conv_postfx"></a><br><sub><code>rlx_conv_postfx</code></sub> |
| <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/rlx_fit_grade.jpg" width="200" alt="rlx_fit_grade"></a><br><sub><code>rlx_fit_grade</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/rlx_frame_filter.jpg" width="200" alt="rlx_frame_filter"></a><br><sub><code>rlx_frame_filter</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/rlx_geo_erosion.jpg" width="200" alt="rlx_geo_erosion"></a><br><sub><code>rlx_geo_erosion</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/rlx_geo_materials.jpg" width="200" alt="rlx_geo_materials"></a><br><sub><code>rlx_geo_materials</code></sub> |
| <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/rlx_geo_refine.jpg" width="200" alt="rlx_geo_refine"></a><br><sub><code>rlx_geo_refine</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/rlx_geo_terrain.jpg" width="200" alt="rlx_geo_terrain"></a><br><sub><code>rlx_geo_terrain</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/rlx_mesh_smooth.jpg" width="200" alt="rlx_mesh_smooth"></a><br><sub><code>rlx_mesh_smooth</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/rlx_palette.jpg" width="200" alt="rlx_palette"></a><br><sub><code>rlx_palette</code></sub> |
| <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/sand.jpg" width="200" alt="sand"></a><br><sub><code>sand</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/scad_profile.jpg" width="200" alt="scad_profile"></a><br><sub><code>scad_profile</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/shadow_types.jpg" width="200" alt="shadow_types"></a><br><sub><code>shadow_types</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/simcity_render.jpg" width="200" alt="simcity_render"></a><br><sub><code>simcity_render</code></sub> |
| <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/skinned_animation.jpg" width="200" alt="skinned_animation"></a><br><sub><code>skinned_animation</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/snow.jpg" width="200" alt="snow"></a><br><sub><code>snow</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/spacecraft_render.jpg" width="200" alt="spacecraft_render"></a><br><sub><code>spacecraft_render</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/starfield_probe.jpg" width="200" alt="starfield_probe"></a><br><sub><code>starfield_probe</code></sub> |
| <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/svg_gallery.jpg" width="200" alt="svg_gallery"></a><br><sub><code>svg_gallery</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/tower_collapse.jpg" width="200" alt="tower_collapse"></a><br><sub><code>tower_collapse</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/ui_panel.jpg" width="200" alt="ui_panel"></a><br><sub><code>ui_panel</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/usd_stage.jpg" width="200" alt="usd_stage"></a><br><sub><code>usd_stage</code></sub> |
| <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/venom_button.jpg" width="200" alt="venom_button"></a><br><sub><code>venom_button</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/venom_gallery.jpg" width="200" alt="venom_gallery"></a><br><sub><code>venom_gallery</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/viewport_path_trace.jpg" width="200" alt="viewport_path_trace"></a><br><sub><code>viewport_path_trace</code></sub> | <a href="docs/examples.md"><img src="https://raw.githubusercontent.com/eugenehp/threers/main/docs/images/volume_slices.jpg" width="200" alt="volume_slices"></a><br><sub><code>volume_slices</code></sub> |

Interactive ones open a window, for example `cargo run --release --example
map_demo --features captions`, `--example marks_map` or `--example venom_button`. The rest of
[`examples/`](examples/) export video or print numbers.

## Documentation

Start with [getting started](docs/getting-started.md). Every guide is in
[`docs/`](docs/), and each one stands alone:
[architecture](docs/architecture.md) · [controls](docs/controls.md) ·
[interactive effects](docs/fx.md) · [signed distance fields](docs/sdf.md) ·
[animation](docs/animation.md) · [camera animation](docs/camera-animation.md) ·
[scroll video](docs/scroll-video.md) · [parity with three.js](docs/parity.md) ·
[releasing](docs/releasing.md).

## License

MIT
