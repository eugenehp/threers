# Marks: drawing data

The scene graph is the right model for a few hundred objects, each with its
own material and transform. It is the wrong model for a million rows of a table,
where every row is drawn the same way and differs only in its values. A
**mark** layer covers that second case. It is one draw call for the whole
layer, and each property (position, radius, colour) is read from its own
column of per-instance data.

`threers::marks` provides the pieces an application like kepler.gl or deck.gl
is built from. The application itself (layer configuration, UI, data loading)
is left to the project using them.

| | |
|---|---|
| `PointMarks` | Antialiased circles in pixels or world units, with strokes |
| `SegmentMarks` | Lines of any width, polylines with mitred joints, and arcs |
| `ShapeMarks` | Any `BufferGeometry` instanced per row, lit (bars, hexbins, 3D icons) |
| `PolygonMarks` | Filled and extruded polygons with holes, thousands to a draw (GeoJSON fills, choropleths, buildings) |
| `TextLabels` | Text anchored in the world, facing the camera, overlaps culled by priority |
| `DensityMarks` | A heatmap: Gaussian kernels summed offscreen, then mapped through a colour ramp |
| `MapControls` | Drag the ground, zoom toward the cursor, tilt and turn |
| `PolygonGeometry`, `triangulate_polygon` | Filled and extruded polygons with holes, from GeoJSON-style rings |

![A 3D bar chart, a picked scatter plot and the named colour ramps](images/data_charts.jpg)

*`cargo run --release --features captions --example data_charts`: bars are
`ShapeMarks` on a `ColorScale`, the scatter is `PointMarks` with one row
found by `MarkRenderer::pick` and ringed, axes are `SegmentMarks`, and every
word is a `TextLabels` row.*

None of it needs compute or storage buffers. It uses only instanced vertex
buffers and render passes, so it runs on WebGL2 as well as WebGPU.

## Drawing over a frame

Marks draw over a frame the renderer has already drawn, and depth-test against
it, so they appear correctly in front of or behind meshes:

```rust
use threers::marks::{Mark, MarkRenderer, MarkTarget, MarkView, Origin, PointMarks};

let mut marks = MarkRenderer::new(&device);
let mut points = PointMarks::new();
points.set_positions(&rows);            // [[f64; 3]]
points.set_radii(&radii);               // f32 per row
points.set_colors(&colors);             // sRGB [u8; 4] per row
points.style_mut().size_scale = 4.0;    // pixels, by default

// Every frame:
renderer.render(&mut scene, &camera, &frame_view, false);
let view = MarkView::new(&camera, origin, renderer.drawing_buffer_size());
let target = MarkTarget::over_frame(&renderer, &frame_view);
marks.draw(&device, &queue, &target, &view, &mut [&mut points, &mut labels]);
```

`MarkTarget::over_frame` finds the frame's depth buffer, including the
multisampled one when MSAA is on. Headless rendering uses
`HeadlessRenderer::mark_target()` instead. `MarkTarget::render_target` does
the same for a `RenderTarget`.

Points, lines and labels face the camera, so a disc resting on the ground
would lose the part of it that lies over nearer ground. `style.depth_lift`
depth-tests them nearer the camera by that many of their own sizes. The
default of 2 keeps a mark whole on a level surface seen from as low as 27°
above it; 4 covers down to 14°. Geometry less than that far in front of a
mark no longer hides it. Set it to 0 for the anchor's own depth.

[`examples/marks_map.rs`](../examples/marks_map.rs) puts every piece together:
a heatmap, hour-filtered points with hover picking, hexbins, trips, arcs and
labels among seven hundred `PolygonMarks` buildings, with `MapControls`.

## Columns

Each property is a `Column`. Rows past the end of a column take the column's
default value, so leaving one empty gives every row the same value.
`update_*(start, values)` rewrites only those rows, and only those rows are
uploaded on the next draw, so streaming in new data or animating part of a
layer does not re-send the whole layer.

Each layer also has two **custom columns** of one to four floats
(`common.set_custom(slot, components, values)`). The marks ignore them. They
exist for the hooks.

## High precision

Positions are `f64`. At ten million units from the origin, the spacing
between adjacent `f32` values is a whole unit. Web Mercator in metres puts
every city millions of metres from the origin, so `f32` geometry there snaps
to a one-metre grid and shakes as the camera moves.

Each coordinate is uploaded as two `f32`s: the nearest `f32` (high part) and
the remainder (low part). The `Origin` is split the same way, and the shader
subtracts high from high and low from low. Two nearby `f32`s subtract exactly,
so the result keeps the `f64`'s precision. Moving the origin is a uniform
write, not a re-upload.

The camera and any meshes alongside the marks use origin-relative
coordinates (`Origin::to_local`). When the camera has travelled far enough
that those numbers are no longer small, call `Origin::rebase` and shift them
by the offset it returns.

## Hooks and GPU filtering

`ShaderHooks` replace two WGSL functions that every mark calls:

- `hook_instance(i: MarkInstance) -> MarkInstance` runs in the vertex stage.
  It can move, resize, recolour or hide (`visible = 0`) an instance.
- `hook_color(color, custom0, custom1) -> vec4<f32>` runs on the final linear
  colour.

`style.params` provides four `vec4`s the hooks read as `layer.params`. A
filter range or an animation time goes there, and changing it does not
rebuild the pipeline. `ShaderHooks::range_filter(component, param)` is the
common case: an instance is drawn only while a custom value lies inside
`params[param].xy`. Moving a time slider over a million rows then writes 16
bytes. `ShaderHooks::validate()` compile-checks hooks against every mark kind
without a GPU.

`ShaderHooks::trail(times, param)` animates trips, like deck.gl's
`TripsLayer`. Each segment carries the times its two ends are reached (custom
components `times` and `times + 1`). With `params[param] = [now, trail, _, _]`,
each segment is cut to the part between `now - trail` and `now`, so the head
lands exactly at `now` rather than at the next vertex. The trail fades from
opaque at the head to clear at the tail.

## Geographic data

`marks::geo` converts `[longitude, latitude]` to Web Mercator metres and back.
`units_per_meter(latitude)` gives the map-unit scale for true heights, which
Mercator stretches by `1 / cos(latitude)`. `lon_lat_to_world` goes straight to
world coordinates, with Y up and north toward −Z, the same convention as
`PolygonFeature`.

`marks::geojson::FeatureCollection::parse` reads a FeatureCollection, a
single Feature or a bare geometry. Properties come back typed, and quoted
numbers read as numbers. `shapes(project)` projects every coordinate and sorts
the geometry into points, paths, polygons and closed polygon outlines, ready
for each mark. It records which feature each row came from, so
`per_polygon(&values)` and friends spread a per-feature property over the rows
it produced. Lines crossing the antimeridian are not split.

## Picking

`MarkRenderer::pick` draws `(layer, row)` ids into an integer target around
the cursor and reads them back. Its cost does not grow with the number of
instances. Pass the frame's depth view as `occluder` so marks hidden behind
meshes cannot be picked.

The result arrives asynchronously in a `PickRequest`. Poll the device and call
`try_take()` on a later frame. Blocking (`wait`) is native-only, because
waiting on the GPU from a browser's main thread deadlocks the page.

For segments built from polylines, `SegmentMarks::path_of` maps a segment row
back to its path. For labels, the row is the label's index.

## Labels

Glyphs are rasterized once from any `CaptionFont` at `raster_size` pixels
per em (48 by default), converted to signed distance fields, and packed into
one atlas. The same atlas serves every label size between about a sixth and
twice the raster size, and `halo_width` adds an outline for reading over busy
maps. The built-in font is an 8×8 bitmap and looks like one, so load a
TrueType face for real text.

- `glyph_field = GlyphField::Msdf` builds multi-channel distance fields from
  the outlines instead. Corners stay sharp however large the text is drawn,
  so a smaller `raster_size` (about 32) serves. The fourth channel keeps the
  plain distance, which the halo uses. The bitmap face has no outlines, so its
  glyphs stay single-channel.
- `add_fallback(font)` draws characters the face lacks from another face,
  and further fallbacks in the order added. Kerning applies within one face.
- `add_icon(name, w, h, alpha)` adds an alpha mask as a distance field, and
  `TextLabel::with_icon(LabelIcon::new(name, size))` draws it to the left of
  the text, in its own colour and with the label's halo. Icon and text make
  one symbol with one collision box.

TrueType composite glyphs (most accented letters) are assembled from their
components.

Collision runs on the CPU only when the view changes. Labels are projected,
sorted by `priority`, and placed greedily. `visible()` reports which ones
survived.

## Polygons

`PolygonMarks::set_polygons` takes GeoJSON-style rings (`[x, y]`, holes
allowed, a closing point optional) and triangulates every feature once into
one shared mesh: a roof, and walls hanging from it. Each vertex carries its
feature's colour, elevation and custom values, so ten thousand parcels are one
draw, and `update_colors(k, …)` or `update_elevations(k, …)` re-uploads
only feature `k`'s vertices.

Elevation is in world units, times `size_scale`. At 0 the layer is a flat
fill. A roof always shows its colour exactly, so a choropleth matches its
legend; walls darken as they turn from the light (`ambient`). Custom values
are per feature through `set_feature_custom`, and picking returns the
feature's index.

Fills draw with a depth bias toward the camera. Far from the origin, a fill's
position is rebuilt on the GPU from split `f64`s while a ground mesh's is
local `f32`, so the two land a rounding apart. Without the bias the ground won
about half the pixels in the test that checks this. For outlines, draw the
same rings with `SegmentMarks::set_paths`.

## Density

Each row adds a Gaussian (radius `size_scale`, weight from its `weights` row)
to a half-float field the size of the viewport. One fullscreen pass then maps
the field through the ramp. `max_weight` is the sum that reaches the top of the
ramp. It is fixed rather than measured from the view, so a place keeps its
colour as other places scroll in and out.

Set `auto_max = Some(k)` to scale to the view instead: the top of the ramp is
`k` times the densest pixel this frame. The maximum is found on the GPU by
a chain of render passes, each keeping the largest of a 4×4 block of the
last, down to one texel that the composite reads. Nothing is read back, and
it runs on WebGL2. With `k` below 1, the peaks saturate and the sparse parts
show.

## Map views

`MapView` keeps a map's camera, `MapControls`, `Origin` and shared
`CollisionIndex` in step. `handle(event, cursor)` feeds it input.
`begin_frame()` (or `begin_frame_for(&renderer)`) does the rest each frame:
- moves the origin under the view once it strays more than
  `rebase_distances` view distances away, and returns the shift for
  origin-relative meshes;
- places the camera and fits its clip planes;
- builds the `MarkView`;
- begins the collision frame.

`covering_tiles`, `zoom`, `ground_point` and `look_at` work in absolute
coordinates. Give each label and sprite layer `collision_index:
Some(map.collision.clone())`.

## Building tile and globe renderers on marks

These are the pieces a downstream crate (vector tiles, sprites, H3/S2,
clustering, a globe) builds on.

**Your own mark kinds.** Implement `Mark` and report `MarkKind::Custom`
from a `CustomShader`: WGSL with `vs_main`, `fs_main` and `fs_pick`, written
against the shared `PREAMBLE` (uniforms, `MarkInstance`, `mark_relative`,
`mark_offset_px`, `mark_lift_depth`, `mark_out`). Custom columns are spliced
in at `/*CUSTOM_IN*/`, `/*CUSTOM0*/` and `/*CUSTOM1*/`. `extra_layout` and
`extra_bind_group` bind your own textures as group 2. Hooks, picking, depth,
blending, clipping and local positions then work for it as for the built-in
kinds. `CustomShader::validate` compile-checks it without a GPU.

**Images.** `TextureAtlas` packs RGBA images onto shelves with gutters, grows
in height, keeps optional mip levels (premultiplied, so edges do not darken)
and uploads only what changed. Share one between layers with `.shared()`.
`SpriteMarks` draws its images facing the camera: per row a position, a
size, an atlas rect, a rotation and a colour, in colour, mask or
distance-field mode, with an anchor. `ImageMarks` draws them on quads in the
world (raster tiles, overlays). `ImageQuad::ground` places a north-up quad.

**Tiles.**
- *Local positions.* `set_local_positions(origin, …)` on every kind stores
  `f32` positions relative to a per-layer `f64` origin, at 12 bytes a row
  instead of 24. Polygons and paths are stored locally by default.
- *Building off the main thread.* `PolygonMesh::build` and `PathMesh::build`
  do the triangulation and flattening, are `Send`, and convert to and from
  bytes for a Web Worker. `set_mesh` and `set_path_mesh` then only upload.
- *Tile math.* `geo::TileId` (bounds, parent, children, quadkeys),
  `units_per_pixel_at_zoom`, `zoom_for_units_per_pixel`, and
  `geo::covering_tiles(view, …)`, which walks the pyramid. It culls tiles
  outside the frustum and splits a tile while it covers more than a tile's
  worth of pixels, so a tilted view gets fine tiles near and coarse ones far.
  `MarkView::units_per_pixel(p)` gives the scale at a point.
- *Seams and stacking.* `style.clip` (a ground rectangle) cuts fills, lines
  and images at a tile's edge. It keeps or drops points, sprites, shapes and
  labels by anchor, which also drops the copies of a point a tile repeats in
  its buffer. `style.depth_order` stacks coplanar layers (water under land
  use under roads) without z-fighting.
- *Memory.* `release_gpu()` frees a layer's GPU memory and keeps its data.
  `gpu_bytes()` reports what it holds, for a cache's budget. A layer given
  new data reuses its buffers when they are large enough.

**Map lines.** `SegmentMarks::dash` (dash and gap in line widths, continuing
across a path's joints), `cap` (butt, round, square), and `outline_width` with
`outline_color` for a casing. Width in metres is `SizeUnits::World`. Paths
that arch or follow great circles are mitred at their joints too.

**Labels along lines.** `TextLabel::along(path, text)` sets the text along
a line, such as a road's or river's name. Each glyph follows the line's
direction on screen, centred on its length, reading left to right whichever
way the line was drawn. It is hidden when the line is shorter than the text
or turns more than `TextLabels::max_bend` between neighbouring glyphs.

**Collision across layers.** A `CollisionIndex` shared by label and sprite
layers (`collision_index`) keeps their boxes apart across layers and tiles.
Start each frame with `index.begin_frame(&view)`. Layers place in draw order.
While the view holds still and no layer changes, `begin_frame` keeps last
frame's boxes and every layer skips placement, so a still map costs nothing
to place. A layer that changes takes its own boxes back and places again,
and the next frame rebuilds the index in draw order. After removing or reordering layers,
call `invalidate()`. Calling `clear()` every frame also works, but then every
layer places every frame. `TextLabels::stickiness` favours labels that were
already showing, so a moving view does not flicker.

**Raster tiles.** `MapView::covering_tiles` (or `geo::covering_tiles`)
gives the tiles for a view, nearest first: finer near the camera when tilted,
coarser toward the horizon. Pack the loaded images into a `TextureAtlas` and
draw each tile as an `ImageQuad::ground(lo, hi, 0.0, rect)` from
`TileId::bounds`. While a tile loads, draw the matching sub-rectangle of its
nearest loaded ancestor, so zooming shows a blurred parent rather than a
hole. The atlas repeats each image's edge into its gutter, so tiles laid edge
to edge show no seams. `examples/map_demo.rs` does all of this in about 200
lines, fetching with `curl` on worker threads and caching tiles on disk.

**Picking.** `pick_rect` returns every distinct `(layer, row)` in a box. Set
`common.set_feature_ids(…)` and each hit carries `feature`, an id stable
across tiles. `style.highlight` colours one row through a uniform, so hover
costs no upload.

**Transitions.** `ShaderHooks::transition(offset, size, param)` moves rows
from where they were to where they are as `params[param].x` runs from 0 to
1, for clusters opening or points moving between time steps.
`transition_offsets` computes the column. `ShaderHooks::combine` chains it
with a filter or a trail.

**Globes.**
- `geo::lon_lat_to_globe`, `globe_to_lon_lat`, `great_circle_distance` and
  `great_circle_point` convert and measure on the sphere.
- `PolygonMesh::build_with(features, base, &GlobeGround)` densifies rings
  and subdivides roofs to follow the sphere without cracks, and extrudes
  radially.
- `SegmentMarks::globe` follows great circles.
- `ImageMarks::sphere` bends tiles onto the sphere.
- `style.horizon` culls marks on the planet's far side, for layers that do
  not depth-test against it.
- `GlobeControls` drags the planet, zooms toward the cursor, and tilts and
  turns. It works in `f64`. Place the camera with
  `apply_relative(&mut camera, &controls.origin())` and draw the marks
  against the same `Origin`, and the view stays exact from orbit down to a
  metre off the ground.
- For planet-scale depth, turn on `Renderer::set_reversed_z` (from
  JavaScript, `setReversedZ`). Its depth precision is nearly even with
  distance, so two walls a metre apart 2 km away stay apart with a 5 cm near
  plane, where standard depth lets the back one through.
  `MarkView::from_renderer` picks the mode up for the marks.
  `fit_clip_planes` (on both controls) helps either way.

**Tone mapping and post-processing.** Materials tone-map per fragment.
Marks keep their exact colours unless `style.tone_mapped` is set, in which
case they take the scene's curve and exposure from
`MarkView::from_renderer`. To have post effects (bloom, depth of field)
treat marks as part of the scene, render into a linear render target, draw
the marks into the same target (`MarkTarget::render_target`), then call
`apply_postfx`.

## Performance

`cargo run --release --example marks_bench` times each layer alone at
1920×1080, far from the origin. A frame is one draw plus waiting for the GPU.
On an Apple M4 Pro (integrated GPU, shared with other work at the time, so
±30 % between runs):

| layer | rows | frame | after a 1,000-row edit |
|---|---:|---:|---:|
| points, 2.5 px, range filter | 1,000,000 | 8–10 ms | 8–9 ms |
| path segments, mitred | 100,000 | 2–3 ms | 2–2.6 ms |
| polygons, extruded, ¼ with holes | 50,000 (1.25M vertices) | 1.5–2.8 ms | 1.4–2.3 ms |
| density kernels, 30 px radius | 200,000 | 3.3–4.6 ms | 2.9–4 ms |

The CPU side of a frame is under half a millisecond for each of these, so
the frame time is GPU fill. Points are cheaper when a filter hides some of
them. Triangulating the 50,000 polygons takes about 45 ms, once.

The density field is summed at reduced resolution: twelve texels across a
kernel's radius, never coarser than a quarter of the viewport. At that
resolution it stays within 2.5 % of the full-resolution field. At full
resolution the same 200,000 kernels took 10.6 ms.

## Limits

- Polylines are mitred at their joints, so a translucent line keeps one
  shade along its length. Turns sharper than 120° get round caps instead of
  a long spike, and those caps overlap, so a translucent line is darker on
  such a turn.
- `ShapeMarks` are lit by one directional light (`MarkView::light_direction`)
  plus ambient. They are not the PBR pipeline.
- Hooks apply to marks, not to the scene's built-in materials. For custom
  shading on meshes, use `ShaderMaterial`. It also works on an
  `InstancedMesh`, which has two per-instance custom columns of its own
  (`InstancedMesh::set_custom`), read as `in.icustom0` and `in.icustom1`.
- Marks are not tone-mapped unless `style.tone_mapped` is set. Drawn onto a
  finished frame they come after post-processing. Drawn into the linear
  target before `apply_postfx`, they come before it.
- Labels along a path place each glyph on the line's screen-space
  direction, interpolating the world anchor linearly within a segment.
  Very long segments under strong perspective drift slightly from the line.
