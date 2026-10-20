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

## Three dimensions

A volume, as a separate type — the 2D fluid is unchanged and is still what
this module leads with.

```bash
cargo run --release --features fluid --example fluid3d   # → out/fluid3d.png
```

<img src="images/fluid3d.jpg" width="440" alt="a rising plume of coloured smoke">

```rust
use threers::fluid::{FluidConfig3d, FluidSim3d};

let mut fluid = FluidSim3d::new(device, queue, FluidConfig3d {
    buoyancy: 46.0,
    ..FluidConfig3d::default()
})?;
fluid.splat([0.5, 0.1, 0.5], [0.0, 300.0, 0.0], [1.0, 0.3, 0.2]);
fluid.step(1.0 / 60.0);
fluid.render(&camera, &frame.texture);
```

**Less changes than you would expect.** Advection, divergence, the Jacobi
sweep and the projection are the same equations with a third term; the stencil
grows from five points to seven, so the pressure divisor is six rather than
four. Three things genuinely differ.

**Vorticity is a vector.** In the plane the curl is a scalar — one number for
how fast a point spins — and the confinement force is that number times a
quarter turn of the gradient. In space the curl is an *axis*, and the quarter
turn becomes the cross product it was always standing in for: `f = ε (N × ω)`.
That is the only formula in the solver whose shape changes.

**Buoyancy is worth having**, because there is an up. It must be
proportional to the dye at each voxel: a uniform upward push accelerates the
empty fluid too, and the whole volume ends up piled against the ceiling as a
rectangular slab. (It did, while this was being written.)

It is also only *conventionally* up. `buoyancy_axis` points it anywhere —
the stage was written `vel.y = vel.y + …`, which is what buoyancy usually
is but not something the solver needs to assume, and a plume along a tilted
chimney is one vector away. Measured, buoyancy along x, along y and along
−z each drift the dye 4.2 voxels along the axis they were given and about
a ten-thousandth of that along the other two.

The random splats had the same bias and less excuse: their vertical
component was drawn from `rand` rather than `rand − 0.5`, so it was never
negative and every one of them went up. They now take a direction uniformly
over the sphere — not three independent components, which is uniform over a
*cube* and therefore denser towards its corners.

**You cannot simply look at it.** A 2D fluid is already an image; a volume has
no surfaces, so `render` marches rays through it and accumulates emission
against absorption, front to back, stopping a ray once what remains cannot
change the pixel. The first sample on each ray is jittered by a fraction of a
step — without that every ray samples at the same distances, the sample planes
line up across the image, and the volume comes out in wood-grain stripes.

### Why the volume lives in a 2D texture

A render pass writes one attachment, and a 3D attachment is one z-slice. A 64³
volume would be sixty-four passes per stage and about eighteen hundred a
frame, and the pass setup alone would cost more than the arithmetic. So the
slices are tiled across a 2D texture and every stage is one fullscreen pass,
exactly as in two dimensions — which also keeps the property the rest of this
module is built around: **no compute stage**, so it runs on WebGL2, which has
no 3D storage textures at all.

The price is that trilinear filtering is not free: two bilinear fetches from
adjacent slices and a lerp, with the in-tile coordinate clamped half a texel
from the tile edge. That clamp is not optional — without it a fetch near a
boundary averages in the neighbouring slice's opposite face, which looks like
the fluid leaking diagonally through itself.

Each texture carries **its own** atlas description, because a pass can touch
grids of different sizes: advecting the dye reads the velocity, and the dye is
usually the finer of the two. Sampling one with the other's tiling still
produces a plausible-looking plume, which is why
`a_splat_appears_where_it_was_put` projects a splat's known position onto the
screen and checks the brightest pixel is *there*.

### The solve, and why it is multigrid here

A Jacobi sweep moves information one voxel. In a volume that is a much worse
deal than in the plane: the pressure from a puff falls off as 1/r rather than
log r, and six neighbours share the correction instead of four. Measured, as
the fraction of a splat's divergence still there after the projection
(`the_v_cycle_beats_the_sweeps`):

| solve | divergence left |
|---|---|
| 20 Jacobi sweeps | 0.816 |
| 300 Jacobi sweeps | 0.369 |
| **1 V-cycle** | **0.306** |
| 2 V-cycles | 0.107 |

One V-cycle beats three hundred sweeps, for about the cost of twenty. So
**the volume defaults to `PressureSolver::Multigrid`** where the plane
defaults to Jacobi — the 2D default is the original's, and there is no
original here to match.

The cycle is the textbook one and reuses the same Jacobi sweep as its
smoother; the three stages it adds are the residual, the restriction and the
prolongation. Two of those are nearly free in an atlas: a coarse voxel's
centre lands exactly on the corner its eight fine voxels share, so one
trilinear fetch *is* their full-weighted average, and reading a half-size
volume at a fine voxel centre *is* trilinear prolongation. The bottom grid
gets twice its side in sweeps and no more — the arithmetic is free at 4³, the
render passes are not.

### Exact taps

Every stage but the two that cross grids reads its neighbours at whole-voxel
offsets, and a whole voxel of the atlas is one texel. Those were going
through the hand-rolled trilinear path — two filtered fetches and a lerp to
arrive at a value whose interpolation weights are always 1 and 0. They are
`textureLoad` now, which halves the memory traffic in the pressure sweep.
Measured with `examples/fluid3d_bench.rs`, the whole step: **2.4× faster at
64³ and 5× at 96³**.

The arithmetic is unchanged, which is the part worth testing —
`the_taps_hit_the_voxels_they_name` runs a divergence pass on a known
velocity field and recomputes it on the CPU from the field read back,
including the free-slip walls.

### Pushing it, and watching it

A click in the volume puts a puff of dye where the pointer is looking, and a
drag pushes it the way the pointer moved — `impulse_at_screen`. Shift-drag or
right-drag orbits instead.

A screen point is a ray, not a place, so something has to pick the depth, and
there is no right answer without reading the volume back. It lands on the
plane through the volume's centre facing the camera: poking the middle of the
box is what a pointer over the middle of the box looks like it should do, the
plane turns with the camera so it stays stable while orbiting, and it cannot
land behind the viewer or outside the box. `a_click_lands_where_it_was_aimed`
works backwards from the render — aim at a screen position, ask where the
brightest pixel ended up — because a projection and an unprojection that are
not each other's inverse put dye somewhere plausible and wrong.

`show_fps` draws the frame rate over the picture, in both dimensions. It is a
3×5 bitmap font packed into fifteen bits a glyph and drawn in the shader
rather than the renderer's overlay canvas, which would want a texture upload
a frame and a path to get it here; this way it works the same in a browser,
on a phone and in a headless capture. Both solvers share the glyphs and
declare their own entry point, because they name their uniform block
differently.

Under the number is a **sparkline of the last sixty-four frames**, because a
number tells you the rate now and a chart tells you whether it is steady,
which is the question anyone watching a frame counter is actually asking. A
hitch that lasts two frames never moves a smoothed figure and is a spike
here. Bars over the refresh target are drawn warm, under it cool.

Two decisions in it are worth the words. The chart is **scaled to twice the
current frame time**, not to the target: scaling to the target reads well
until the fluid is slower than it, and then every bar clips and the chart is
a solid block saying only "slow", which the number already said. Against the
current frame time a steady run sits at half height whatever the rate, so the
variation — the thing the chart is for — is always visible. The target still
gets a line where it falls, so the absolute reference is not lost; when the
fluid is below half its target the line leaves the top of the chart, which is
itself worth seeing.

The rate behind it is not a frame counter. Two renders inside one frame — a
capture beside a present, a resize that redraws — are not a thousand frames a
second, and letting one in sends the smoothed figure somewhere it takes a
hundred frames to come back from; it first showed up as a read-out pinned at
its 999 clamp while the page was plainly running at sixty. The threshold that
rejects them is **relative**, a fifth of the frame time already measured,
because a fixed one has no right value: a flat millisecond is correct at sixty
frames a second and wrong at eight hundred, which is what the Metal backend
reaches at 720p on an M4 — a real frame there is 1.2 ms, so a 1 ms floor threw
away half of them and the chart had bars for 38 of 90 renders. It is capped at
4 ms so that one slow first frame cannot latch the floor shut and leave the
rate stuck at the value that set it.

And the history travels as a **64×1 `rgba8unorm` texture**, one
`write_texture` a frame, rather than as more uniform: the plane's uniform
block is shared with the Metal backend byte for byte, and sixty-four floats
is not a thing to add to it for a read-out. A byte over a 250 ms range is a
fortieth of a millisecond, and the chart is eighteen pixels tall. The shader
gets the row width and that range substituted in from Rust rather than
hard-coding them, so the draw cannot disagree with the upload about how wide
the row is or what a byte means.

One line of that glyph code shipped broken, in a way worth recording because
nothing in the repository could catch it. It read

```wgsl
let leading = column == 0 && shown < 100 || column == 1 && shown < 10;
```

which WGSL forbids — mixing `&&` and `||` needs parentheses — and which
**naga accepts anyway**. Every native test passed, and so did every browser
test, because the browser tests ran under WebGL2, where naga compiles the
module itself. In a real WebGPU browser the compiler is Tint, which rejects
it, and a rejected *module* takes every pipeline in it down at once: the
canvas was black, the frame loop ran at sixty, the panel worked, and wgpu's
log was empty, because the error belongs to the browser rather than to wgpu.
`no_wgsl_mixes_and_with_or` in `tests/wgsl_portability.rs` now walks every
WGSL literal in the tree looking for it, and headless Chrome needs
`--enable-unsafe-webgpu` to exercise the WebGPU path at all.

Testing on that path immediately found a second one, the same shape: the
volume sampled its atlas with `textureSample`, which needs implicit
derivatives and so is only legal in **uniform control flow** — and the
volume's fetches happen inside the ray march's loop. naga does not enforce
that either, Tint does, and the volume had therefore never rendered in a
WebGPU browser. `textureSampleLevel(…, 0.0)` is legal anywhere and, on a
one-level atlas, identical;
`the_volume_shader_samples_with_an_explicit_level` keeps it that way. The
plane's seventy-odd `textureSample` calls are left alone deliberately: they
sit at the top level of a fragment function, which *is* uniform.

### On a phone

The simulation is unchanged; the shell around it is not. The device pixel
ratio is **capped at two** — a phone reports three, and this is a per-pixel
cost from the splat to the march, so nine device pixels per CSS pixel is most
of a frame for sharpness nobody resolves at arm's length. A Mac's Retina
display reports exactly two, so the cap costs desktop nothing. The default
quality drops to `medium` on a narrow screen, and `?quality=` still wins.

The 232-pixel panel is a corner of a laptop window and two thirds of a phone,
so under 720 pixels it becomes a sheet along the bottom: full width, at most
half the height, scrolling, clear of the notch and the home bar through
`env(safe-area-inset-*)`, and **closed on load** behind a 44-pixel toggle,
because the picture is the point. Controls grow to a finger's worth of target.

Two gestures have no touch equivalent and get one. The volume's push is
`ctrl`-drag on a desktop, and a touch pointer has no modifier to hold — so
`drag pushes` is a switch in the panel, shown only where the pointer cannot
hover, and `pushing` is that switch *or* ctrl. Splats are the space bar on a
desktop and a double-tap on the canvas on touch. The hints under the panel
are rewritten to whichever is true.

All three solvers share one `Rate`: the ring, the smoothing and the rule above
are decided once, because three copies of a threshold that subtle would drift.
The Metal backend has the glyphs and the chart transcribed into MSL and its own
`MTLTexture` for the row — `replaceRegion` rather than a queue write — and
carries the two constants as literals, since `FLUID_SHADER_SOURCE` is public
and has to compile as it stands; `the_msl_read_out_agrees_about_the_row` holds
them to the Rust. The transcription itself is checked by rendering the same
pinned history through both backends and comparing: identical to the byte,
which also establishes that the premultiplied blend Metal uses lands in the
same place as the straight alpha the wgpu pipeline asks for.

### What can be skipped, and what only looks like it can

The stage timings put the solve at ~95% of the step and the advections at
1–5%, which rules out the first idea anyone has: masking the dye advection
to the bricks that hold dye. Four fifths of the volume is empty, the
occupancy pyramid is already there, and a stencil would skip most of that
pass — and it would buy under 2% of a frame, because that pass is not where
the time is. The place to look is the solve, and most of what is there turns
out to be load-bearing.

**Skipped, and now are:**

| | passes at 64³ |
|---|---|
| the two clears, folded into the sweeps that follow | 5 |
| the occupancy reduction, on steps nothing renders | 2 |
| curl and confinement, when `curl` is zero | 2 |
| buoyancy, when it is zero | 1 |

A step went from 49 passes to 42. The occupancy one is the clearest case of
work with no reader: it is built for the ray march, so a simulation stepped a
hundred times and drawn once was reducing the dye a hundred times to use the
answer once.

**Not skippable — measured, each buys its cost:**

| | residual left | passes |
|---|---|---|
| 1 smoothing sweep a level | 0.536 | 34 |
| **2** | **0.324** | **42** |
| 3 | 0.248 | 50 |
| 4 | 0.209 | 58 |

Eight passes for a quarter of the residual, in both directions, so two is a
balance rather than a habit. The coarse levels are the same story — dropping
two of the four costs 23% of the residual to save 10% of the solve.

**What about only solving what the camera can see?** Smoothing is most of
the solve's passes and a volume is mostly off screen, so it is the obvious
next cut. Whether it is sound depends on the equation, not the renderer:
advection is hyperbolic and local, but the pressure solve is *elliptic* — it
makes the velocity divergence-free everywhere at once, and the field that
does that is determined by the whole domain simultaneously.

Measured, it is far more local than that suggests. A splat in one corner,
one step, mean |pressure| in shells going away from it:

| near | | | far |
|---|---|---|---|
| 26.19 | 2.77 | 0.53 | 0.06 |

The far quarter of the box carries two parts in a thousand of what the near
quarter does. So limiting the solve to a region *around the activity* would
be very nearly exact — which is the same shape as the empty-space skip, and
a better mask than the frustum, because it needs no camera and it is where
the pressure actually is.

The mechanical catch is the ping-pong. "Skip" in a render pass cannot mean
"leave that texel alone": the sweeps alternate buffers, so declining to
write leaves a value from two sweeps ago rather than the current one. A
masked texel has to be read and rewritten, which saves the six neighbour
taps and not the pass — unless the mask kills the fragment before the shader
with a stencil, and then the staleness is real and has to be shown to be
harmless. Worth building, and not a one-line change.

**Free, and worth more of:** the carry-over from last frame's pressure costs
nothing, because it is a multiply folded into the first sweep. Keeping all
of it converges better than keeping 0.8 — 0.269 against 0.281 after eight
frames — and the drift that damping exists to prevent does not happen: the
pressure equation's null space should let a constant offset accumulate, and
nine hundred frames of forcing leave the peak at 22 to 26 either way,
including under Jacobi, which has no coarse grid to anchor it. So the volume
carries all of it where the plane carries 0.8.

### Folding passes into the shader

The stage timings said the solve is ~95% of the step, and at small grids it
is bound by how many passes it is rather than by what they compute. Two of
those passes were doing nearly nothing:

* **Starting from a fraction of last frame's answer** was a pass over the
  whole grid to multiply a field by a constant. The Jacobi stencil is linear,
  so scaling the six neighbours the first sweep reads is the same thing —
  `fs_pressure3` takes that scale in `params.z`.
* **A coarse level starting from zero** was a clear-only pass followed by a
  sweep that read the zeros back. The same scale set to 0 collapses the whole
  sweep to `-rhs / 6`: no loads at all, and no clear.

A step at 64³ with four levels went from **49 render passes to 44**, and
`folding_the_clears_removed_five_passes` counts them —
[`passes_per_step`](https://docs.rs/threers/latest/threers/fluid/struct.FluidSim3d.html)
is the deterministic companion to `stage_ms`, because five passes in
forty-nine is inside a GPU timer's noise and is not inside counting's. Every
residual in the test suite is unchanged to three figures, which is the point:
the fusion is arithmetic that was already being done, done in fewer places.

It also turned up a real bug. The bind-group cache was keyed on *the most
recently handed out* uniform slot, which is the right slot only while every
block is bound before the next is filled. A pass that wants two blocks before
it binds either — which is exactly what a first sweep that scales differently
needs — got the second one's bind group for both. The symptom was a solver
where the sweep count stopped mattering: twenty sweeps and three hundred both
left 0.944. The slot now travels with the buffer.

### Stepping over the empty parts

Four fifths of a settled plume is nothing: measured on the default volume,
**80% of its 8³ bricks and 85% of its 4³ bricks** hold no dye a ray could
see. A ray marching through that spends four fifths of its samples fetching
zero and adding it to the total.

So the dye is reduced twice — halving each time, keeping the largest value of
each eight — into a volume at a quarter of its side, and the march steps over
a whole brick wherever that largest value is below what it already treats as
nothing. The reduction is rebuilt when the dye moves, not when a frame is
drawn, because it costs the same whatever the render is.

| | march alone | whole frame |
|---|---|---|
| stepping over empty bricks | 8.6 ms | 15–21 ms |
| every step | 16.9 ms | 25–28 ms |
| | **1.9×** | **~1.3×** |

Marching a settled plume into 2560×1440. The frame-level number is lower
because the solver does not get faster, only the march does.

The skip stops one dye voxel short of the far face of a brick. The dye is
read with a filter, so a sample within half a voxel of that face reads part
of whatever is in the *next* brick — the one case where an empty brick is not
empty. Landing inside the face instead takes the worst pixel of a 128×128
render from 38 levels out to 11, on three channels in sixty-five thousand.

### Averaging the march across frames

Past about 256³ the dye is finer than any affordable number of samples can
resolve, and the picture stops being resolution-limited and starts being
noise-limited — the 272³ plume at 96 samples a ray is visibly stippled.
`temporal_samples` averages the march over frames instead: each one jitters
its samples differently and blends into what came before with a weight of
`1/n`, so 24 samples a frame converge towards what 288 give in one.

Measured against a 288-sample reference, with the fluid and the camera held
still: a single 24-sample march is **0.088/255** away, and 24 accumulated
frames are **0.019** — four times closer, for the cost of one.

It used to reset whenever the camera moved, on the grounds that a volume
has no single depth to reproject by. It has no *surface*, which is not the
same thing: it has an **expected** depth. Weight each sample along a ray by
what it actually contributed to the pixel — its own opacity times the
transmittance that reached it — and the mean is where the picture at that
pixel came from. For a plume that is a few voxels inside the smoke, which is
the thing the eye tracks when the camera moves, and quite good enough to
reproject by.

So the march reports it on a second attachment (one loop, not a second
march), and the resolve unprojects it and projects it through the previous
frame's camera to find where that point was on screen. Accumulating through
a slow orbit now lands **0.045/255** from a twelve-times-longer march of the
final view, where a single short march is 0.104 — and where, before, a
moving camera got exactly that single march, because the history was thrown
away every frame.

A point that was off screen last frame has no history, and takes this
frame alone.

**The fluid keeps moving, and reprojection does not help with that.** Measured, that was
not a minor cost: against a long march of the same moment, eight accumulated
frames of a rising plume came out **thirty times further from the truth**
than a single short march with no history in it. Accumulation as a plain
average is not a feature with a caveat, it is a feature that is wrong
whenever anything moves.

So the history is rejected where it no longer matches. Each frame's march
goes into a buffer of its own, and the resolve clamps the history to within
`temporal_clip` standard deviations of its own 3×3 neighbourhood before
mixing — the colour clamp a temporal anti-aliasing resolve does, for the
same reason. Where the fluid has not moved the history is inside that range
and survives; where it has, it is pulled to the nearest value the pixel
could plausibly have now.

Min and max of the neighbourhood is the obvious box and it is too wide: the
march is jittered, so nine samples carry nine samples' worth of noise, and
clamping to their extremes still left a moving plume eleven times off. The
spread of the signal is the right box, and how wide it should be is a
measurement:

| clip | still camera | rising plume |
|---|---|---|
| 0.25σ | 0.067 | 0.179 |
| **0.5σ** | **0.024** | **0.235** |
| 1.0σ | 0.019 | 0.408 |
| no clip | 0.019 | 1.233 |
| no accumulation | 0.088 | 0.050 |

Half a sigma is the knee and is the default: nearly all of the convergence
for a little over half the ghosting of one sigma, and **5.2× less ghosting
than not rejecting at all**. It is still a still-content feature — a moving
plume is better off without it — but it now degrades rather than falls over,
and `reject` is a slider in the browser panel next to `accumulate`.

**The artefact it could not remove, and why.** Averaging eight frames of the
272³ plume dropped the local roughness from 36.2 to 20.3 and took the
per-pixel stipple out — but left a cross-hatch at the scale of the
empty-space skip's bricks. Averaging cannot touch that, because the bricks
are in the same place every frame.

The cause was in the skip, not the march. A ray that stepped over a brick
landed wherever the skip put it, which is not where its next sample was
going to be — so it carried on offset from every ray that did not skip, and
two neighbouring pixels that skipped a different number of bricks sampled
the volume at different phases. The skip now advances to the next sample the
ray *would* have taken. Roughness fell to 16.6, and in the regime that
exposed it — a coarse march over a fine dye — skipping is now **exactly**
invisible: mean 0.000/255 against marching every step, where off-lattice it
was 0.225 over 532 channels. `skipping_keeps_rays_on_the_sample_lattice`
holds it there, and was checked by putting the bug back.

### Compression, and what it is actually worth

The dye is the largest field in a volume and the one the heaviest pass reads
and writes, and it is never negative — nothing in the volume subtracts from
it — so it does not need the sign bit that `rgba16float` spends. Three
channels of 11-bit float hold it in half the space.

It works — 41 MB down to 24 at the default, and the picture is the same to a
tenth of a level (`the_packed_dye_is_the_same_picture`). Being a *float* is
why: the same four bytes as a ten-bit unorm would quantise the faint tails,
where smoke reads, into two or three levels.

It is **not faster**, on the hardware path. Sampling the packed format costs
more than the bandwidth it saves: the march alone, at an identical sample
count, ran 40% slower on an M4 Pro, and the step about twice as slow. That is
the opposite of what halving a texture is supposed to do, and it is why the
flag exists rather than the packing simply being done.

| | memory | step + march |
|---|---|---|
| `rgba16float` | 41 MB | 3.2 ms |
| `rg11b10ufloat` | 24 MB | ~2× slower |

So `compact_dye` defaults to **false**, and is a runtime switch rather than a
build-time one, because the trade is a property of the texture unit rather
than of the fluid — a software rasteriser, with no filtering hardware to be
slow at unpacking, may well go the other way. Measure it before trusting
either direction:

```bash
FLUID_COMPACT=1 cargo run --release --features fluid --example fluid3d_bench
```

### Picking the optimisations while it runs

All of them are fields on `FluidConfig3d` and all can be changed through
`set_config` on a running simulation, because which ones pay is a property of
the machine rather than of the fluid:

| knob | what it does | what it costs | measured against |
|---|---|---|---|
| `empty_space_skip` | steps the march over empty bricks | GPU, in the march | **1.9×** the march |
| `cache_bindings` | keeps bind groups and uniform blocks between frames | CPU, recording the step | **1.12–1.34×** the CPU encode |
| `compact_dye` | packs the dye into half the memory | both | −41% memory, ~2× slower on Metal |
| `pressure_solver` | multigrid against Jacobi | GPU, in the solve | a V-cycle beats 300 sweeps |

`compact_dye` reallocates the dye and so empties the volume; the rest take
effect on the next frame. The browser demo has all four as live controls
under **Optimisations**.

**Do they stack?** No, and the reason is worth more than the answer.

Wall-clock timing of a whole frame cannot settle it — these are effects of
tens of percent and a shared machine varies by more. So the volume can be
asked for the GPU's own clock instead:

```rust
sim.profile();                 // false if the adapter has no TIMESTAMP_QUERY
// …run some frames…
for (name, ms) in STAGE_NAMES.iter().zip(sim.stage_ms()) {
    println!("{name:>10}  {ms:.3}ms");
}
```

On a settled plume at 64³, per frame:

| curl+vort | divergence | solve | project | advect |
|---|---|---|---|---|
| 0.045 | 0.014 | **1.75** | 0.026 | 0.053 |

**The solve is around 95% of the step**, and not one of the three
optimisations touches it. They are dividing up the other 5% — plus the ray
march, which is a separate pass of about the same size as everything that is
not the solve. That is the whole answer: they cannot add up to much because
the thing they are not optimising is nearly all of it.

Which raises the obvious question about the solve: a V-cycle on a 64³ grid is
about forty render passes and most are on grids from 32³ down to 4³, where
the arithmetic is free and the pass is not. Stopping the coarsening earlier
should then be nearly free. Measured, it is not:

| `multigrid_coarsest` | levels | divergence left | solve |
|---|---|---|---|
| 4³ | 4 | **0.324** | 0.871 ms |
| 8³ | 3 | 0.374 | 0.840 ms |
| 16³ | 2 | 0.399 | 0.787 ms |
| 32³ | 1 | 0.586 | 0.644 ms |

Dropping to two levels costs 23% of the residual to save 10% of the solve.
The coarse levels pay for themselves and the default coarsens all the way;
the solve is mostly real work in the top-level sweeps, not overhead. The knob
stays because that balance is a property of a machine's per-pass cost.

### Three implementations, and what keeps them honest

There are now three of this solver — wgpu, the hand-written Metal backend
(two dimensions only), and the CPU reference — and the failure mode they
share is *silent* divergence: a setting one of them does not implement,
ignored without complaint, producing a picture that still looks like a
fluid. Two guards against that, both of which caught something the day they
were written:

`MetalFluid::unsupported` names what that backend will ignore in a given
configuration, and `metal_fluid_matches_wgpu` asserts the list is empty for
the configuration it compares — otherwise it would be reporting how far
apart two *different algorithms* are and calling it agreement. It
immediately found `colorful` set in the config under test.

`the_two_solvers_agree_about_which_way_is_up` points buoyancy along
`[0.6, 0, -0.8]` and checks the CPU and the GPU put the dye in the same
place. The existing parity test runs with buoyancy off, which is the right
way to isolate the projection and the wrong way to notice that one of them
still believed up meant +Y — which one did, for about an hour, with nothing
failing because nothing asked.

### The same solver on the CPU

[`CpuFluid3d`](https://docs.rs/threers/latest/threers/fluid/struct.CpuFluid3d.html)
is the volume again in plain Rust — same stages, same V-cycle, same
`FluidConfig3d`, flat `Vec`s in voxel order instead of an atlas because
nothing on the CPU has to write a 2D attachment. It exists for two things the
GPU version cannot give you.

The first is a check. Two transcriptions of one algorithm should remove the
same fraction of a puff's divergence, and `the_cpu_and_gpu_solvers_agree`
holds them to it: **0.358 on the GPU, 0.337 on the CPU**, from half floats
through a texture unit on one side and `f32` with hand-rolled trilinear on
the other. A stage wired to the wrong grid in either would show up there
first.

The second is an answer to what the GPU is worth. Step only — no ray march,
since that is the part both of them do:

| solver | dye | CPU, 1 core | CPU, all | GPU | |
|---|---|---|---|---|---|
| 16³ | 32³ | 1.1 ms | 1.1 ms | 0.92 ms | 1× |
| 32³ | 64³ | 8.6 ms | 4.2 ms | 1.21 ms | 3× |
| 48³ | 96³ | 29.0 ms | 10.8 ms | 1.13 ms | 10× |
| 64³ | 128³ | 63.1 ms | 17.3 ms | 1.34 ms | 13× |
| 80³ | 160³ | 122.7 ms | 29.6 ms | 1.67 ms | 18× |
| 96³ | 192³ | 210.0 ms | 50.2 ms | 2.13 ms | 24× |

Two things in that table are worth more than the ratio. **Threading buys
about 3.5×, not twelve** — the stages are memory-bound, and at 16³ it is a
net loss, where rayon's per-call cost exceeds the work in the slice. And the
**GPU is nearly flat from 16³ to 96³**, 0.92 ms to 2.13: at those sizes it is
not doing arithmetic, it is paying for fifty render passes. Which is the same
thing the stage timings say — the solve is 95% of the step and most of a
small solve is pass overhead.

### On iOS and iPadOS

The crate already builds for `aarch64-apple-ios`, device and simulator, with
and without the Metal backend — wgpu reaches Metal there the same way it does
on a Mac. What differs is the way in and the budget.

**The way in.** UIKit has no window handle to hand a graphics library: a view
that draws with Metal has a `CAMetalLayer`, and that layer *is* the surface.
[`GpuContext::from_metal_layer`](https://docs.rs/threers/latest/threers/gpu/struct.GpuContext.html)
takes it.

**The budget.** A volume costs the cube of its side, and on a phone the
penalty for asking too much is not a slow frame, it is the watchdog killing
the app. So `within_memory` answers the question the other way round — the
largest volume that fits a number of megabytes:

| budget | volume |
|---|---|
| 32 MB | 56³ solver, 112³ dye |
| 64 MB | 72³ solver, 144³ dye |
| 128 MB | 88³ solver, 176³ dye |
| 256 MB | 112³ solver, 224³ dye |

That is only worth having if the estimate is the allocation, so
`the_memory_estimate_is_the_memory` holds `memory_for` to `memory_bytes` at
five resolutions including a 50³ that the atlas pads awkwardly.

A trap worth naming, because it was caught by a test rather than by reading:
`FluidFormats::volume_dye` is what the *device* allows, not what the
simulation will use — the packed dye is off by default. Size against the
first while allocating with the second and the biggest field is
underestimated by half. `FluidFormats::resolve` exists to make that
explicit, and `threers-fluid-ios` learned it by asking for 48 MB and getting
58.

`crates/threers-fluid-ios` wraps all of it in a C ABI static library, with a
`CADisplayLink` loop and a `CAMetalLayer` in nine functions. Its
`a_layer_comes_up_and_draws` drives those entry points against a real
`CAMetalLayer` — on macOS, where the class, the surface and the drawables are
the same ones iOS uses.

### Speed

`time_scale` on both configurations is a multiple of real time, applied by
`update` and **not** by `step`. That split is load-bearing: the parity tests,
the determinism test and the CPU/GPU comparison all pass a fixed timestep to
`step` and mean it, so a knob that silently rescaled it would pass every test
about speed and break every test about agreement.
`speed_scales_update_and_not_step` pins both halves.

There is no upper limit at which it blows up — semi-Lagrangian advection is
unconditionally stable at any step. What there is is an accuracy limit: past
about four, a voxel's contents travel several voxels in a step and the
back-trace reads from somewhere the fluid never flowed through, which reads
as tearing rather than motion. The browser panel's **speed** slider stops at
four for that reason, and `threers_fluid_set_speed` clamps at eight.

### How far this goes

Marching into 1280×720 on an M4 Pro, with the dye at twice the solver's side
— whole frames, step and march, on a settled plume:

| solver | dye | frame | | memory |
|---|---|---|---|---|
| 96³ | 192³ | 3.2 ms | 315 fps | 141 MB |
| 128³ | 256³ | 5.3 ms | 188 fps | 328 MB |
| 160³ | 320³ | 8.1 ms | 124 fps | 652 MB |
| 192³ | 384³ | 12.0 ms | 83 fps | 1.1 GB |
| 224³ | 448³ | 19.5 ms | 51 fps | 1.8 GB |
| 256³ | 512³ | 24.2 ms | 41 fps | 2.7 GB |
| **272³** | **544³** | **30.0 ms** | **33 fps** | **3.2 GB** |
| 288³ | 576³ | 36.6 ms | 27 fps | 3.7 GB |

So **about 272³ carrying a 544³ dye** is what thirty frames a second buys,
and memory is what you run into first on most machines: the cube law means
each row is roughly twice the one above it. Dropping the march — simulating
without looking at it — moves that to about 288³.

Against the same budget, the same solver on the CPU:

| | 30 fps ceiling | solver voxels | |
|---|---|---|---|
| CPU, one core | 48³ / 96³ dye | 110 thousand | 1× |
| CPU, all cores | 80³ / 160³ dye | 512 thousand | 4.6× |
| GPU | 288³ / 576³ dye | 24 million | **216×** |

A factor of two hundred in voxels, which is a factor of six in resolution.
The CPU solver is not a fallback — it is the reference the GPU one is checked
against, and the measurement that says what the GPU is for.

### What it costs, and picking a size

The cube law is the whole story: 64³ is a quarter of a million voxels, 128³ is
two million. The default is a 64³ solver carrying a 128³ dye — 41 MB of
fields, the coarse multigrid chain included — where the 2D default carries eight times the dye resolution of its
solver, because in a volume that would be five hundred and twelve times the
memory.

The resolution is a runtime decision, through a named level or by hand:

```rust
use threers::fluid::{quality3d, FluidConfig3d};

// A level, at construction…
let config = FluidConfig3d::default().with_quality(quality3d("ultra").unwrap());
// …or while it is running, which rebuilds the grids.
fluid.set_resolution(96, 192);
```

`QUALITY_LEVELS_3D` holds five levels under the same names the 2D fluid uses
— `low` through `extreme`, 24³ to 128³ — and the numbers behind them are the
volume's own. A resize drops the fluid: an interpolated volume would be eight
textures of resampling to preserve a second of simulation. Setting the size
it already has does nothing, so a slider can fire on every pixel of travel.

`grids()` reports what you actually got after the device's texture limit has
had its say; an oversized volume halves until it fits rather than failing to
allocate. `memory_bytes()` is the number to look at before `extreme`, which
is 324 MB of fields — against a frame that only grows from 3.2 ms at `high`
to 5.2 ms, because half of it is the march and the march is set by the pixels
on the screen rather than by the voxels behind them.

```bash
cargo run --release --features fluid --example fluid3d_bench
```

splits a volume frame into the solve, the march and the memory at each
resolution, and reports what each solver costs.

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
