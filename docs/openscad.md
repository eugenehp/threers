# OpenSCAD solid modeling

Part of the [threers](../README.md) documentation.

# OpenSCAD solid modeling

The `openscad` feature adds two front ends onto one `Solid` CSG tree — an OpenSCAD
`.scad` interpreter and a Rust DSL — evaluated by a pure-Rust **watertight
exact-CSG kernel** (`to_geometry_exact`), with mesh import/export.

```bash
cargo run --example openscad_gallery --features openscad   # 37-demo language tour → STL
cargo run --example dsl_showcase     --features openscad   # the Rust DSL → STL
# convert a .scad to any mesh format (format = output extension):
cargo run --example scad2stl --features openscad -- model.scad out.glb
```

```rust
// Cargo.toml: threers = { version = "0.0.5", features = ["openscad"] }
use threers::{cube, cylinder, sphere, scad, parse_scad};

// (a) Rust DSL — fluent builder + the `scad!` macro:
let part = scad! {
    difference() {
        cube([30.0, 30.0, 30.0]);
        translate([0.0, 0.0, -1.0]) { cylinder(40.0, 8.0); }
    }
};
let _stl = part.to_stl();          // also .to_obj/.to_off/.to_3mf/.to_glb

// (b) or parse an OpenSCAD program:
let solid = parse_scad("difference(){ cube(20,center=true); sphere(12,$fn=48); }").unwrap();
let _glb = solid.to_glb();
let _ = (sphere(1.0),);            // primitives are also free functions
```

Curved∧curved booleans (e.g. `sphere ∪ sphere`, `cylinder ∩ cylinder`) resolve to
watertight meshes; the kernel falls back to the float evaluator only where its
manifold gate can't verify a result, so it is never wrong — only sometimes
deferential. Try it live in the browser: **`web/openscad-playground.html`**.

# Animating OpenSCAD models

OpenSCAD animates by re-evaluating the whole program with `$t` stepped from 0 to
1 — geometry is a function of time. `openscad::animate` drives that loop and
renders it.

```bash
cargo run --release --example scad_animate --features "openscad,video,native-codec"
```

```rust
use threers::openscad::animate::{ScadAnimation, ScadCamera, ScadRender};

let mut animation = ScadAnimation::from_file("chuck.scad").frames(120).fps(30);
ScadRender::new(1280, 720)
    .supersample(2)
    .camera(ScadCamera::turntable())
    .export_video(&mut animation, "chuck.mp4")?;
```

- **`color()` survives evaluation.** `Solid::parts()` splits the model into
  separately colored pieces — CSS names, `#rrggbb`, `[r,g,b,a]`, and alpha for
  see-through parts. Booleans distribute over the pieces exactly, so a cut
  through a colored body keeps its color. A model with no `color()` is still one
  part, identical to `to_geometry_exact()`.
- **Cameras that frame the model**: `Auto` fits it exactly (a per-corner frustum
  fit, not a loose bounding sphere), `Turntable` orbits at a fixed distance so
  the model does not breathe, `Viewport` obeys the model's own `$vpr`/`$vpt`/
  `$vpd` — including written as functions of `$t` — and `Fixed` takes an
  explicit eye and target. Z-up, like every model written for OpenSCAD.
- **It avoids the work where it can.** A model that never reads `$t` is
  evaluated once and shared by every frame; the rest are evaluated several at a
  time (`concurrency`, default `min(4, cores)`), which is ~3× on the example
  model. `--features parallel` additionally parallelises the CSG inside a frame.
- **Output**: `render_frames` (RGBA), `render_png`, `export_png_sequence`, and
  `export_video` / `export_video_with` — the latter takes a `VideoOptions`, so
  [subtitles](#subtitles-and-captions) and codec settings come along.

**Colors.** A `ScadPalette` is a background plus a cycle of part colors; eight
ship built in (`studio`, `cornfield`, `metallic`, `sunset`, `midnight`,
`blueprint`, `nature`, `monochrome`) and `ScadPalette::from_hex` takes your own.
`ScadColoring` decides where a part's color comes from:

```rust
use threers::openscad::animate::{ScadColoring, ScadPalette, ScadRender};

ScadRender::new(1280, 720)
    .palette(ScadPalette::blueprint())        // also sets the background
    .coloring(ScadColoring::Palette)          // ignore color(), re-skin the model
    // …or ModelThenPalette: keep color() and dress only the untagged parts
    .part_colors(vec![[1.0, 0.3, 0.2, 1.0], [0.2, 0.5, 0.9, 1.0]]);  // explicit list
```

**Lights.** The default is a three-point studio rig placed *relative to the
camera*, so a turntable never rotates the model into its own shadow — the usual
failure of a fixed rig. The key light casts shadows, sized to the model's bounds.

```rust
use threers::openscad::animate::{ScadLight, ScadLighting, ScadRender};

ScadRender::new(1280, 720)
    // Camera-relative key/fill/rim over a sky-and-ground hemisphere.
    .lighting(ScadLighting::studio().warmth(0.5).ambient(0.3))
    // …or a fixed sun, so the shading says which way a face points:
    .lighting(ScadLighting::sun(215.0, 38.0))
    // …or flat and shadowless, like the OpenSCAD GUI:
    .lighting(ScadLighting::flat())
    // …or exactly what you specify:
    .lighting(ScadLighting::custom(vec![ScadLight::Key {
        color: [1.0, 0.95, 0.9], intensity: 2.0, azimuth: 30.0, elevation: 25.0,
    }]));
```

Ambient light comes from the palette — each scheme carries its own sky and
ground colors, so `blueprint` is lit coolly and `sunset` warmly with no extra
setup.

**Speed** comes in three flavours, because "faster" can mean three things:

| Knob | Changes |
|------|---------|
| `speed(2.0)` / `seconds(4.0)` | how fast it *plays* — same frames, different frame rate |
| `ping_pong(true)` / `easing(…)` | how fast the *model moves* through the loop |
| `quality(ScadQuality::Draft)` | how fast it *renders* — kernel, curve resolution, supersampling |

```rust
use threers::openscad::animate::{ScadAnimation, ScadEasing, ScadQuality};

let animation = ScadAnimation::from_file("chuck.scad")
    .frames(120)
    .seconds(4.0)                       // retime the loop
    .speed(0.5)                         // …then play it at half speed
    .ping_pong(true)                    // out and back, no snap at the loop point
    .easing(ScadEasing::EaseInOut)      // accelerate away from rest, settle at the end
    .quality(ScadQuality::Draft);       // rough it in first
```

Driving a model from Rust rather than `$t` works the same way:

```rust
// Any root-scope name can be the animation variable.
let animation = ScadAnimation::from_file("arm.scad")
    .frames(90)
    .var("SHOULDER", |t| 90.0 * t)
    .var("ELBOW", |t| 45.0 * (1.0 - t));

// …or skip `.scad` entirely and animate the Rust DSL.
let animation = ScadAnimation::from_fn(|t| {
    threers::cube([20.0, 20.0, 20.0]).difference(threers::sphere(4.0 + 8.0 * t as f32))
});
```
