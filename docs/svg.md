# Vector SVG

Part of the [threers](../README.md) documentation.

# Vector SVG export

`SvgRenderer` draws a `Scene` to an SVG document — one `<path>` per triangle,
depth-sorted back to front and filled with the colour the scene's lights give
it. It is three.js's `SVGRenderer`, and it needs no GPU, no window and no
adapter: it runs on the CPU, in CI, over SSH, and in a wasm build with no
WebGPU.

```bash
cargo run --example svg_export      # four shading modes into out/
cargo run --example svg_gallery     # the nine cases that are hard to sort correctly
```

```rust
use threers::prelude::*;

let renderer = SvgRenderer::new(1200, 900);
renderer.render_to_file(&mut scene, &camera, "out/frame.svg")?;
// or, for the markup:
let markup = renderer.render_to_string(&mut scene, &camera);
```

Shading tracks `Renderer` rather than inventing its own look: the same
Lambert/Phong accumulation, the same punctual attenuation, the same
tone-mapping curves, and the same linear→sRGB encode on the way out. Set
`tone_mapping` to whatever the wgpu renderer is using and the two agree on
colour.

```rust
let renderer = SvgRenderer::new(1200, 900).with_options(SvgOptions {
    shading: SvgShading::Wireframe,   // or Lit (default) / Flat
    background: false,                // transparent document
    precision: 1,                     // coarser coordinates, smaller file
    ..SvgOptions::default()
});
```

| Option | Default | What it does |
|--------|---------|--------------|
| `shading` | `Lit` | `Lit` / `Flat` (unlit material colour) / `Wireframe` |
| `tone_mapping`, `exposure` | `None`, `1.0` | Match `Renderer::set_tone_mapping` |
| `cull_backfaces` | `true` | Honours the material's `side` |
| `background` | `true` | `scene.background` as a full-canvas rect |
| `seam_stroke` | `0.8` | Hairline per face that hides SVG's anti-aliasing seams |
| `precision` | `2` | Decimal places on coordinates |
| `sort` | `true` | Painter's-algorithm depth sort |
| `curve_tolerance` | `Some(0.15)` | Bends edges onto the real surface past this pixel error |
| `depth_split` | `Some(0.05)` | Splits faces that span too much depth to sort |

**Curves, not polygons.** A tessellated sphere has a polygonal outline, and no
amount of shading hides that its silhouette is a 24-gon — a strange thing to
ship in a format whose whole point is that it has curves in it. Where the
geometry carries vertex normals, the surface between two vertices is known, so
each edge is bent back onto it and emitted as a cubic `C` rather than a straight
`L`. Only edges that need it are curved: a cube's normals are constant across
each face, so its edges stay straight and its document stays small, while a
sphere's are not and its outline comes out round. `curve_tolerance` is the
pixel error that decides.

Measured against an analytic sphere, that puts the outline exactly on the
silhouette from about sixteen segments upwards, where straight edges still
visibly under-fill it. Below that it closes most of the gap and not all — the
outline's corners are mesh vertices, and on a mesh that coarse those already sit
inside the true silhouette, so an eight-segment sphere comes out round but a
couple of pixels small. The answer there is more segments, not a tighter
tolerance.

**What it can and cannot draw.** Faces are flat-shaded, one colour each, so
there are no textures, no shadow maps and no post-processing — a `Sprite` is
skipped for the same reason. `Points` become `<circle>`, `LineSegments` become
`<line>`, and `InstancedMesh` expands to one sorted draw per instance.

**The one thing to know about the painter's algorithm.** One depth per face
cannot order a polygon that spans other geometry, and the stock symptom is a
two-triangle ground plane painting over the bottom of everything standing on
it — objects sliced off at the floor line. Two things keep that from happening.
Faces sort by how far back they *reach* rather than by their centroid, so a
floor, which reaches to the horizon, is painted before anything standing on it.
And `depth_split` cuts up whatever is still deep enough to span something, which
after the sort key mostly means a receding wall in front of a small object.
Tessellated geometry never trips the test — a torus knot emits the same paths
either way — so the default is close to free on scenes that do not need it;
lower it for fewer artefacts and a bigger file.

Geometry is also clipped to the frame, not just to the `viewBox`. A viewer
clips anyway, so this buys nothing on screen — it matters because a floor
running to the horizon projects thousands of units past the edge, and any tool
that fits the *content* bounding box instead (Illustrator, Inkscape, Figma, a
thumbnailer) then shows the artwork as a small offset speck. Clipping keeps the
bounding box and the canvas the same thing.

This is checked rather than asserted: `tests/svg_render.rs` renders a set of
scenes, casts a ray per sampled pixel to find what is actually in front of the
camera, and compares that against what the document paints. Ground truth comes
from the geometry, not from the wgpu renderer — two renderers disagreeing tells
you only that they disagree, and it would need a GPU and an SVG rasteriser to
run at all. Interpenetrating solids are
the one case left that no per-face sort can resolve — two cubes pushed through
each other show the seam. For those, use a z-buffer, which is to say a raster
image:

```rust
// Exactly what the GPU drew, wrapped in SVG as an embedded PNG.
let rgba = headless.render_to_rgba(&mut scene, &camera);
let (w, h) = headless.render_size();
std::fs::write("out/frame.svg", threers::svg_from_rgba(w, h, &rgba))?;
```

**From the browser.** `SVGRenderer` in the three.js shim is a drop-in for the
real one: it owns an `<svg>` element, so you enable it the way you enable any
three.js renderer.

```js
import { SVGRenderer } from 'threers';

const renderer = new SVGRenderer();
renderer.setSize(800, 600);
document.body.appendChild(renderer.domElement);   // this is what turns it on

renderer.setToneMapping(4, 1.0);   // match the WebGL renderer's ACES
renderer.setQuality('low');        // straight edges, no seam stroke, small file
renderer.render(scene, camera);    // draws into domElement, returns the markup

const markup = renderer.renderToString(scene, camera);  // or skip the DOM
```

`setSize`, `setClearColor`, `setPrecision`, `setQuality`, `clear`, `autoClear`
and `info.render` behave as three.js's do; `setShading`, `setCurveTolerance`,
`setSeamStroke`, `setDepthSplit`, `setCullBackfaces`, `setBackground` and
`setSort` are the extras above. Resizing keeps every option — it does not
rebuild the renderer. `svgFromRgba` wraps a canvas readback for the raster
path.

[`web/examples/svg-renderer.html`](../web/examples/svg-renderer.html) is the whole
thing running: a live `<svg>`, the options as controls, and a download button.
Note what the page does *not* contain — no canvas, no adapter request. The
renderer is CPU-side wasm, so it works where WebGPU is unavailable.

# SVG path data (`Path`, `Shape`)

A `Path` is already a sequence of Béziers, arcs and lines — the same vocabulary
SVG's `d` attribute speaks. Both directions are lossless:

```rust
use threers::prelude::*;
use threers::{Path, Shape};

let mut p = Path::new();
p.move_to(Vector2::new(0.0, 0.0));
p.bezier_curve_to(
    Vector2::new(0.0, 8.0),
    Vector2::new(10.0, 8.0),
    Vector2::new(10.0, 0.0),
);
assert_eq!(p.to_svg_path_data(3), "M0 0C0 8 10 8 10 0");

// …and back, including arcs, `S`/`T` reflections and relative commands.
let round_trip = Path::from_svg_path_data("M0 0C0 8 10 8 10 0")?;
```

`Shape::to_svg_path_data` writes the outline then each hole as its own closed
subpath (fill it with `fill-rule="evenodd"`), and `Shape::from_svg_path_data`
reads that back — first subpath is the outline, the rest are holes. Combined
with `ExtrudeGeometry`, that turns SVG artwork into 3D geometry, which is what
three.js's `SVGLoader` is for.

Curve types opt in through `Curve2::svg_segments`; the default returns `None`
and the writer flattens, so a `Curve2` you implement yourself still exports —
just as a polyline. `EllipseCurve` becomes a real `A` arc, and `SplineCurve`
converts to cubics *exactly*, because a Catmull-Rom spline is a cubic Hermite
and Hermite-to-Bézier is a change of basis, not a fit.

The `d` string carries the path's own coordinates untouched. SVG's y axis points
down and these curves are y-up, so wrap them in a `transform="scale(1,-1)"` when
assembling a document rather than expecting the serialiser to flip them.
