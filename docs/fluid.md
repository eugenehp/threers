# Fluid

Part of the [threers](../README.md) documentation.

An interactive 2D fluid on the GPU: dye poured into a velocity field that is
advected, confined, projected and dissipated every frame. Drag a pointer across
it and the ink follows, curls into vortices behind the stroke, and diffuses
away. Behind the `fluid` feature.

```bash
cargo run --release --features fluid --example fluid          # a window, under your hand
cargo run --release --features fluid --example fluid_render   # → out/fluid_render.png
FLUID=1 WEBGL=1 web/build.sh && python3 -m http.server -d web # → /fluid.html
```

<img src="images/fluid_render.jpg" width="820" alt="four scripted strokes, left to curl">

```rust
use threers::fluid::{FluidConfig, FluidSim};

let mut fluid = FluidSim::new(device, queue, 1280, 720, FluidConfig::default())?;
fluid.multiple_splats(20);

// per frame
fluid.pointer_move(-1, mouse_x, mouse_y);
fluid.update(1.0 / 60.0);
fluid.render(&frame.texture);
```

It is a port of Pavel Dobryakov's [WebGL Fluid Simulation][orig] onto wgpu, so
the same fluid runs natively and in a browser from one Rust implementation. The
numerics are Jos Stam's *Stable Fluids*, the shape of the frame is Mark
Harris's [GPU Gems 38][gems], and the look — the vorticity confinement, the
bloom, the sunrays, the dye lit as a height field — is Dobryakov's.

## The frame

Five fields, nineteen passes. Per step, over the velocity and dye grids:

| pass | what it does |
|---|---|
| curl | the local spin of the velocity field |
| vorticity | pushes along concentrated spin, putting back what the grid ate |
| divergence | how much the field is creating or destroying fluid |
| pressure × 20 | Jacobi sweeps of ∇²p = ∇·u — or [a V-cycle](#the-pressure-solver) |
| gradient subtract | removes ∇p, leaving an incompressible field |
| advect velocity | the field carries itself |
| advect dye | and it carries the ink — [first or second order](#advection-and-why-the-dye-goes-muddy) |

then bloom, sunrays and a display composite on top of that. Every one is a
fullscreen fragment pass; **there is no compute stage anywhere in it**, which
is what lets the whole thing run on WebGL2 — see [below](#webgl2).

Two things are worth knowing about the cost. The expensive part is the pressure
solve, because making a field divergence-free is a *global* constraint, and
twenty Jacobi sweeps is not a converged Poisson solve. It does not need to be:
what is wanted is a field whose remaining divergence is smooth and small enough
that the dye cannot show it. And the simulation grid is 128² while the dye it
carries is 1024², because the eye reads the dye's resolution and not the
solver's — the ink is sharp, the motion underneath it need not be.

## Quality

The six numbers that decide what it costs are grouped as named levels, so a
menu can offer one choice instead of six:

| level | dye | sim | sweeps | bloom | fields at 16:9 |
|---|---|---|---|---|---|
| `low` | 256 | 64 | 12 | 6 × 128 | ~1 MB |
| `medium` | 512 | 96 | 16 | 7 × 192 | ~4 MB |
| **`high`** | **1024** | **128** | **20** | **8 × 256** | **~15 MB** |
| `ultra` | 2048 | 256 | 32 | 9 × 512 | ~121 MB |
| `extreme` | 4096 | 512 | 48 | 10 × 1024 | ~485 MB |

`high` is the original's settings and what `FluidConfig::default()` gives. The
two above it are not in the original at all — it was written for 2017 mobile
browsers.

```rust
let config = FluidConfig::default().with_quality(quality("ultra").unwrap());
```

What they buy differs at each end. **Dye** resolution is detail in the ink,
which is what the eye reads. **Sim** resolution is detail in the *motion*, and
decides how fine a vortex can get before the grid eats it — so raising it
changes the picture rather than merely sharpening it. **Sweeps** decide how
incompressible the result is.

What they cost is almost entirely dye: two `rgba16float` textures at the
canvas's aspect ratio. The solver's own grids are a few hundred kilobytes at
any level. A quality level is a *request* — the grids that come out of it
depend on the canvas's aspect and on `max_texture_dimension_2d`, and oversized
ones are scaled down on both axes together so the texels stay square. Ask what
actually happened:

```rust
let ((dye_w, dye_h), (sim_w, sim_h)) = fluid.grid_sizes();
let megabytes = fluid.memory_bytes() / (1024 * 1024);
```

## Speed

Measured by `cargo run --release --features fluid --example fluid_bench`,
which submits each stage on its own and drains the queue before reading the
clock — so the split between stages is meaningful and the total is pessimistic
against a real frame. On an M4 Pro at 1920×1080:

| | `low` | `high` | `ultra` | `extreme` |
|---|---|---|---|---|
| solver step | 0.71 ms | 0.78 | 1.94 | 2.71 |
| bloom | 0.64 | 0.65 | 1.08 | 0.59 |
| sunrays | 0.08 | 0.28 | 0.65 | 1.62 |
| shading + composite | 0.13 | 0.17 | 0.15 | 0.31 |

The solver is 50–75 % of the frame at every level, and inside it the dye
advection dominates — it is the only pass that runs on the 1024- or
2048-texel grid rather than the 128-texel one.

**Splats used to be the worst thing in the frame.** Each was two full-screen
passes and a submit of its own, so a burst of twenty-five was fifty passes
over the dye grid: 18 ms at `extreme`, about four frames, for twenty-five
Gaussians. They are now one pass per field for the whole batch, with the
shader summing the blobs and skipping any whose support does not reach the
texel:

| | 25 splats, before | after |
|---|---|---|
| `low` | 2.5 ms | **0.11** |
| `high` | 2.9 | **0.46** |
| `extreme` | 18.4 | **6.8** |

The gain is largest where the work was submit-bound and smallest where it is
bandwidth-bound — at `extreme` the two passes still read and write thirty
million texels, which nothing can avoid. Batches above thirty-two split, which
is past any plausible number of fingers.

## The pressure solver

Twenty Jacobi sweeps leaves **eighty-two per cent** of a splat's divergence in
the field. That number does come down with more sweeps — but as a rate, not
towards a floor, because a Jacobi sweep moves information exactly one texel and
the pressure field from a localised puff of velocity is spread across the whole
grid.

A multigrid V-cycle answers that by solving the smooth part of the problem on a
grid eight texels across, where "spread across the whole grid" is a handful of
neighbours: smooth, take the residual to a grid half the size, solve *that*,
interpolate the correction back, smooth again.

| solver | residual | cost |
|---|---|---|
| Jacobi, 20 sweeps (the default) | 0.826 | 20 passes at full resolution |
| Jacobi, 1280 sweeps | 0.244 | 1280 |
| **Multigrid, 1 cycle** | **0.217** | ~60, most of them at a quarter resolution or less |
| **Multigrid, 2 cycles** | **0.100** | ~120 |

<table>
<tr><td><img src="images/fluid_render.jpg" width="400" alt="twenty Jacobi sweeps"></td>
    <td><img src="images/fluid_multigrid.jpg" width="400" alt="one multigrid V-cycle"></td></tr>
<tr><td align="center"><b>Jacobi, 20 sweeps</b></td>
    <td align="center"><b>Multigrid, 1 cycle</b></td></tr>
</table>

The same strokes, the same seed, the same everything else. A field that is
actually divergence-free rotates instead of spreading, so the dye winds into
tight vortices and stays coherent, where the under-projected one smears across
the canvas. It is a better fluid and it is **not the same picture**, which is
why it is not the default.

One cycle beats sixty-four times the sweeps. Residual is the mean |∇·u| after
the projection as a fraction of what went in, measured on a splat with `dt = 0`
so that advection — which re-introduces divergence, because this solver
projects and *then* advects — is out of the picture.

```rust
let config = FluidConfig {
    pressure_solver: PressureSolver::Multigrid,
    multigrid_cycles: 1,
    ..FluidConfig::default()
};
```

It is **not the default**, because a visibly more incompressible fluid is a
different fluid and the original's look is what this is a port of. Press `m` in
the example, or tick *multigrid* on the page.

One thing the implementation does not do is damp its smoother. The textbook
weight is 4/5, because plain Jacobi's amplification factor at the checkerboard
mode is exactly −1 — the error flips sign every sweep and never shrinks — and a
V-cycle depends on the smoother removing exactly the frequencies the coarse
grid cannot represent. It does not pay here, measurably, and the reason is that
this right-hand side has no checkerboard content: the divergence is a central
difference over *two* texels, an operator with a zero at the Nyquist frequency.
The weight is still a parameter of the sweep, because that is a property of the
divergence operator rather than of multigrid.

## The missing half

One finding is worth stating on its own, because it looks like a bug and is
not.

GPU Gems 38 subtracts the pressure gradient as `u -= halfrdx * (pR - pL, pT -
pB)`, where `halfrdx` is `0.5 / dx`. `(pR - pL)` is a central difference over
*two* texels, so the 0.5 is what turns it into a gradient — and the divergence
pass and the Jacobi sweep both carry the same factor. The original leaves it off
the gradient and only off the gradient, which subtracts exactly twice the
pressure gradient.

Measured on a single splat — a radial puff of velocity, which is pure
divergence — the residual divergence after one projection, as a fraction of
what went in:

| sweeps | `gradient_scale: 0.5` | `gradient_scale: 1.0` |
|---|---|---|
| 20 (the default) | 0.825 | **0.654** |
| 320 | **0.368** | 1.269 |

So it is a compensation, not a mistake. Jacobi converges slowly on a Poisson
problem whose solution is long-ranged; at twenty sweeps the pressure is too
small and so is the correction, and doubling the correction puts back more of
what the unconverged solve left out. Once the solve *is* converged the order
reverses, and the overshoot leaves the field **more** compressible than not
projecting at all.

It is therefore a knob — `FluidConfig::gradient_scale`, 1.0 by default —
belonging with `pressure_iterations`: 1.0 with twenty of them, 0.5 if you raise
them. `the_missing_half_compensates_for_an_unconverged_solve` in the tests holds
both halves of that, so neither can quietly stop being true.

## Advection, and why the dye goes muddy

Semi-Lagrangian advection — trace backwards, fetch, done — is unconditionally
stable at any time step, which is why it is the standard choice and the
original's. It is also first order, and its error is **diffusion**: the
backward trace lands between texels and the bilinear fetch averages them, every
step. Over a few hundred frames that blur is most of why the dye loses its
colours and why small vortices stop existing.

`Advection::MacCormack` estimates that error and subtracts it. Advect forward;
advect the result back, and whatever fails to return is the scheme's own error;
take off half of it. The correction is an extrapolation and can overshoot, so
it is clamped to the range of the four texels the forward pass actually read —
the limiter that makes it unconditionally stable (Selle et al. 2008) rather
than merely sharper.

Measured under a *uniform* flow, which is the only case where both schemes
carry the dye at identically the right speed — a constant field advects itself
exactly, has zero divergence and zero curl — so every difference is the
scheme's own error. A blob a few texels across, a hundred steps, half a texel
of travel each:

| dye grid | semi-Lagrangian | MacCormack |
|---|---|---|
| 128 | 0.42 of its peak | **0.71** |
| 256 | 0.63 | **0.90** |
| 512 | 0.76 | **0.87** |

1.15× to 1.9× more of the blob survives, most where the feature is smallest
relative to the grid — which is exactly what the theory says, and exactly the
case that matters, since a feature far larger than a texel was never going to
blur.

It costs about **2.7× the step** at `high`: the extra pass runs on the dye grid,
the expensive one, and reads six texels rather than one. Not the default, for
the same reason the multigrid solver is not — a less diffusive fluid is a
different-looking fluid. `a` in the example, a checkbox on the page.

One thing it changes beyond sharpness: semi-Lagrangian advection of the
*velocity* is itself a numerical viscosity, so removing it leaves a markedly
more energetic flow — about twice the mean speed on the same input. That is
more correct and it is not more of the same picture.

## Texture space, and why the picture is a mirror

In OpenGL a texture's origin is at the bottom left; in WebGPU it is at the top
left. The simulation lives entirely in uv, so rather than flipping anything the
whole field runs in a frame where y points *down* the screen.

That is sound, and not only convenient. Reflection is a symmetry of the
Navier-Stokes equations, and of this vorticity-confinement term too: the
confinement force is `ε ω (Nŷ, −Nx̂)` with `N = ∇|ω| / |∇|ω||`, and under
`y → −y` both `ω` and `Nŷ` change sign, so applying the identical formula in
the flipped frame gives exactly the mirrored force. Every shader therefore
ports verbatim and the result is the mirror image of the same initial
conditions — which is the same fluid.

It also means pointer coordinates go in as they arrive from the window system,
with no `1 - y`. `a_splat_lands_where_it_was_asked_for` pins that: get the
handedness wrong and everything still runs, still looks like a fluid, and is
upside down.

## WebGL2

The fluid is the one substantial part of this crate that is *complete* on
WebGL2. Bloom and ambient occlusion have raster paths standing in for compute
ones; this has no compute path to stand in for, because there was never a
compute stage in it. What it does need is a **filterable half-float render
target**, which WebGL2 has only with `EXT_color_buffer_float` —
`FluidFormats::for_adapter` probes for it and widens one channel count at a
time, `r16float` → `rg16float` → `rgba16float`, so a device with only the wide
one still runs the whole simulation and merely carries three unused channels in
the pressure field.

`tests/fluid_webgl.rs` runs in headless Chrome, where WebGL2 comes from
SwiftShader, and checks on read-back pixels that the whole frame draws and that
the splat lands where it was asked for:

```sh
FEATURES=fluid,webgl TEST=fluid_webgl scripts/wasm-browser-test.sh
```

## In the browser

`web/fluid.html` is the demo page: a canvas, the knobs, and the grid sizes the
device actually gave. The browser shell is `src/fluid/web.rs` — it does what the
winit example does, with the platform's own primitives, over the same
`FluidSim`.

```sh
FLUID=1 web/build.sh              # or WEBGL=1 FLUID=1, for browsers without WebGPU
python3 -m http.server -d web     # then open /fluid.html
```

```js
import init, { fluidStart, Fluid } from "./pkg/threers.js";
await init();
const fluid = await fluidStart(canvas, "ultra");
fluid.set("curl", 40);
fluid.splats(15);
JSON.parse(fluid.info());   // {"dye":[3641,2048],"sim":[455,256],…}
```

## As a texture in a scene

The fluid draws into any `wgpu::Texture`, which includes a `RenderTarget` the
renderer knows about — so the dye can be a material map on scene geometry with
no new API on either side:

```rust
// Non-sRGB, deliberately — see below.
let rt = Arc::new(RenderTarget::new(&device, 512, 512, wgpu::TextureFormat::Rgba8Unorm));
fluid.render_capture(&rt.color_texture);

renderer.register_render_target(RT_ID, &rt);
let mut material = BasicMaterial::new(Color::WHITE);
material.map = Some(Arc::new(Texture::from_render_target(
    RT_ID, 512, 512, TextureFormat::Rgba8Unorm,
)));
```

One thing has to line up for the colours to survive. The fluid's display pass
hands back **display-referred** values — the original wrote them straight into
an 8-bit canvas with no transfer function — and `register_render_target` samples
an 8-bit target through its **sRGB** view. So the target's format must be the
*non*-sRGB one: then the fluid writes the bytes untouched, the renderer decodes
them to linear to shade with, and they come back out the far end as they went
in. Give it an sRGB target and the encode happens twice.

`the_dye_can_be_a_material_map` in the tests is that recipe, run and checked on
the pixels — a missing texture gives a flat frame, so what it asserts is *hue*:
pixels whose channels disagree, which no flat fill can produce.

## As a pass in the chain

The other way to put it in a frame: a layer over a scene the composer has
already drawn, rather than a texture on a surface inside it.

```rust
composer.add_pass(RenderPass);
composer.add_pass(DrawPass::new("fluid", move |ctx| {
    fluid.update(1.0 / 60.0);
    fluid.render_into(ctx.target(), true);
}));
```

`DrawPass` is the composer's hook for a pass that owns GPU state and draws
itself — its other passes are descriptions it compiles and runs, which covers
a filter over the previous buffer and nothing more. The context hands over the
view the next pass will read, with its format and size attached, which is
exactly what `render_into` wants. Set `transparent: true` and the dye
composites over what the scene left instead of onto a flat colour of its own.
`the_fluid_can_be_a_pass_in_the_composer` is that, on read-back pixels.

## Where it differs from the original

* The nineteen GLSL programs are one WGSL module with nineteen entry points
  over one bind group. The cost is that every pass binds three textures where
  it reads one; the saving is eighteen bind-group layouts.
* `SHADING` / `BLOOM` / `SUNRAYS` were three `#define`s and up to eight
  compiled display variants. Here they are bits in the uniform block — a
  branch that is coherent across the whole draw.
* The dither under the bloom is interleaved gradient noise computed in the
  shader rather than a sampled blue-noise texture, so the crate ships no asset
  and the simulation stays a function of its own state.
* Splat positions and hues come from a seeded generator, so a headless render
  is the same bytes every run. That is what makes the picture above usable as
  documentation.
* Rendering into an sRGB target applies the inverse transfer first, so the
  bytes that land match the ones GL wrote into a plain `RGBA8` canvas. The
  blend against a non-black background is then done in linear rather than
  display space, which is the one place the two can still differ.
* The one case the original handles that this does not is a device that can
  render to a half-float target but cannot *filter* one. WebGPU guarantees the
  filtering, and WebGL2 has it in core for half float.

## Reference

[`threers::fluid`](https://docs.rs/threers/latest/threers/fluid/) —
`FluidSim`, `FluidConfig`, `FluidQuality`, `FluidFormats`.

[orig]: https://github.com/PavelDoGreat/WebGL-Fluid-Simulation
[gems]: https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu
