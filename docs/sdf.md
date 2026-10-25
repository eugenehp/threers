# Signed distance fields

`threers::sdf` is a small library of signed distance fields — exact 2D and 3D
primitives, the operators that join them, and the same functions in WGSL.

## Distance fields

A signed distance field answers one question about a point: how far is it
from the surface, negative inside. That one number is enough to draw the
shape anti-aliased at any size, to march a ray to it, to mesh it — and to
melt shapes into each other with a smooth minimum, which is what makes them
worth having.

```rust
use threers::math::{Vector2, Vector3};
use threers::sdf::{self, d2, d3, Sdf3};

// 2D: a disc and a bar, melted together over 8 px.
let blob = |p: Vector2| {
    let disc = d2::circle(p - Vector2::new(40.0, 40.0), 20.0);
    let bar = d2::capsule(p, Vector2::new(10.0, 70.0), Vector2::new(90.0, 70.0), 6.0);
    sdf::smooth_union(disc, bar, 8.0)
};

// 3D: the same idea, meshed and ray-marched.
let field = |p: Vector3| {
    let ball = d3::sphere(p - Vector3::new(0.0, 0.6, 0.0), 0.5);
    let bar = d3::capsule(p, Vector3::new(-1.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0), 0.2);
    sdf::smooth_union(ball, bar, 0.3)
};
let mesh = field.to_geometry(64, Vector3::ZERO, Vector3::new(2.6, 1.8, 1.2));
let hit = field.raymarch(Vector3::new(0.0, 0.6, 5.0), Vector3::new(0.0, 0.0, -1.0), Default::default());
```

| | |
|---|---|
| `sdf::d2` | `circle`, `rect`, `round_rect` (a pill with the radius at half the height), `segment`, `capsule`, `tapered_capsule` (and `tapered_capsule_radius`, which also returns the local thickness), `polygon` |
| `sdf::d3` | `sphere`, `cuboid`, `round_cuboid`, `capsule`, `tapered_capsule`, `torus`, `cylinder`, `plane` |
| operators | `union`, `intersection`, `subtraction`; `smooth_union`, `smooth_intersection`, `smooth_subtraction`; `smooth_union_blend`, which also says how much of the result is each side so a colour or a thickness can blend through the same fillet; `round`, `onion` |
| coverage | `coverage`; `coverage_edge` with a pixel's `footprint` for edges at an angle; `coverage_stroke` for lines thinner than a pixel — below |
| `Sdf2`, `Sdf3` | Implemented for every `Fn(Vector2) -> f32` / `Fn(Vector3) -> f32`: `gradient`, `normal`, `raymarch`, `to_geometry` (marching cubes, watertight and indexed) |

Symmetric primitives are centred on the origin — move them by subtracting
from the point, `d2::circle(p - centre, r)`. Primitives defined by endpoints
take the endpoints.

Every primitive is exact. Unions of exact fields are exact outside;
intersections, subtractions and every smooth operator are *bounds* — they
never overestimate, so sphere tracing stays safe, but a ray takes more steps
near them. `Sdf3::raymarch` reports how many.

### On the GPU

`sdf::WGSL` is the same library as WGSL — `sdf2_*`, `sdf3_*` and `sdf_*` — with
no bindings and no entry points, so it prepends to a `ShaderMaterial` or a
compute shader:

```rust
let fragment = format!("{}\n{}", threers::sdf::WGSL, r#"
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let d = sdf2_circle(in.uv - vec2<f32>(0.5), 0.3);
    let a = clamp(0.5 - d / fwidth(d), 0.0, 1.0);
    return vec4<f32>(vec3<f32>(a), 1.0);
}
"#);
let material = threers::ShaderMaterial::new(fragment);
```

The two sides are written by hand, so `tests/sdf_gpu.rs` evaluates every
function at the same points on an adapter and on the CPU and holds them to
1e-4 (they agree to about 4e-6). WGSL has no overloading, which is why 2D and
3D are prefixed separately; `sdf2_tapered_capsule` returns the distance in
`.x` and the local radius in `.y`.

### On the overlay

`Overlay::distance_field(x0, y0, x1, y1, |p| ...)` rasterises any 2D field,
anti-aliased, with the current fill and stroke — `rect`, `circle` and
`polygon` are this one call underneath. Only the box is evaluated, and whole
8 px tiles are skipped on the strength of one sample, so the field must never
overestimate the distance (smooth unions are fine).

```rust
o.fill(Color::WHITE);
o.distance_field(0.0, 0.0, 100.0, 100.0, |p| {
    sdf::intersection(d2::circle(p - Vector2::new(50.0, 50.0), 40.0), -d2::rect(p - Vector2::new(50.0, 50.0), Vector2::new(10.0, 60.0)))
});
```

## Thin lines and diagonal edges

Turning a distance into a pixel's coverage is usually `0.5 - d`: a ramp one
pixel wide across the edge. That is right for an edge along a pixel row,
and wrong in two common cases.

**Lines thinner than a pixel.** Drawn as a shape (`capsule` and the edge
ramp), a 0.4-pixel line comes out at 0.9 alpha over two pixels: more than
twice its ink, and darker or lighter as it moves between pixel centres.
`coverage_stroke(u, half_width, footprint)` takes the distance `u` to the
line's *centre* and gives the share of the pixel the stroke really covers,
a box filter. It peaks at the line's width, and across the line the
coverages sum to that width wherever the line falls, so a hairline keeps one
weight as it moves. For strokes wider than a pixel it equals the edge ramp.

**Edges at an angle.** A pixel is `√2` wide across a 45° edge, so a ramp one
pixel wide is too sharp there, and thin diagonals bead.
`footprint(gradient)` gives the width (`1` along a row or column, `√2` on
the diagonal), and `coverage_edge(d, footprint)` and `coverage_stroke` take
it.

```rust
use threers::sdf::{self, Sdf2};

let g = field.gradient(p, 0.5);
let w = sdf::footprint(g);
let filled = sdf::coverage_edge(field(p), w);                  // an edge
let line = sdf::coverage_stroke(centre_distance(p), 0.2, w);   // a 0.4-pixel line
```

The overlay does this for you: `line`, `polyline`, outline strokes,
`stroke_field` and `stroke_field_with_width` take stroke coverage, every
edge takes the footprint, and a polyline is one stroke, so its joints are
covered once rather than beading. The `fx` surface carries each strand's
radius through its smooth unions, so a strand thinning to a thread is
covered as a stroke too. `SegmentMarks`, `LineBasicMaterial::line_width`
capsules (transparent ones thinner than a pixel included) and the Metal
renderer's `set_line_width` quads cover their lines the same way. All of it
is in WGSL as `sdf2_footprint`, `sdf_coverage_edge` and `sdf_coverage_stroke`.

## Built on it

[`threers::fx`](fx.md) — glossy liquids that react to the pointer, on
buttons and any other shape — draws every surface from one of these fields,
with `sdf::WGSL` in its shader and `Overlay::distance_field` for its detail.
