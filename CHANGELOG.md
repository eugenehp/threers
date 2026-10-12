# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.6] — 2026-10-03

Universal Scene Description, read and written in all three of its forms and
checked against Apple's own tools rather than against itself; **Draco** and the
rest of the glTF compression and interchange formats, each verified against the
reference decoder that produced the files; a **WebGL2 backend** so the browser
build runs where WebGPU is not available; an SVG renderer that draws the scene
graph as vectors; sharper text; a converging accumulation mode for offline
frames; and the fixes that writing all of it turned up.

### Added
- **Splats are batched.** Each was two full-screen passes and a submit of its
  own, so a burst of twenty-five was fifty passes over the dye grid — **18 ms
  at `extreme`, about four frames, for twenty-five Gaussians**. One pass per
  field now handles the whole batch, summing the blobs in the shader and
  skipping any whose support does not reach the texel (below about three and a
  half sigma the Gaussian is under 6e-6 of its peak, which a half-float dye
  texture cannot hold anyway). A burst of twenty-five: 2.5 ms → 0.11 at `low`,
  2.9 → 0.46 at `high`, 18.4 → 6.8 at `extreme`. The gain is largest where the
  work was submit-bound and smallest where it is bandwidth-bound — at the top
  level the two passes still move thirty million texels, which nothing avoids.
  Batches above thirty-two split, which is past any plausible number of
  fingers.

- **`Advection::MacCormack`** — second-order advection, for the blur that is
  most of why a long-running fluid goes muddy. Semi-Lagrangian is first order
  and its error is *diffusion*: the backward trace lands between texels and
  the bilinear fetch averages them, every step. MacCormack advects forward,
  advects the result back, and subtracts half of however far it failed to
  return, clamped to the range of the four texels the forward pass read — the
  limiter that makes it unconditionally stable (Selle et al. 2008).

  Measured under a uniform flow, which is the one case where both schemes
  carry the dye at identically the right speed, so every difference is the
  scheme's own error: a blob keeps **0.90 of its peak against 0.63** at a
  256-texel dye grid, 0.71 against 0.42 at 128. It costs about 2.7× the step
  at `high`, because the extra pass is on the dye grid. Not the default — a
  less diffusive fluid is a different-looking fluid.

- **`FluidSim::set_uniform_velocity`** — impose a constant flow. A wind or a
  current, and the only case where advection has an exact answer, which is
  what makes the measurement above a measurement rather than a comparison of
  two different flows.

- **`examples/fluid_bench.rs`** — where a fluid frame actually goes, per stage
  and per quality level, plus what each pressure solver and each advection
  scheme costs. It is what found the splat problem: the solver is 50–75 % of
  the frame and a space-bar burst was four frames on its own.

- **`PassThreading`** — `Send + Sync` natively and nothing on wasm. wgpu's
  resources are `Arc`-backed on one and `Rc`-backed on the other, so requiring
  `Send + Sync` of every `Pass` meant no pass could own a pipeline in a
  browser — which is most of what a self-drawing pass is for, and is how
  `SsaoPass` failed to compile for the web.

- **`SsaoPass` does ambient occlusion.** It returned `COPY_FRAG` — a
  full-screen copy, costing a pass and doing nothing, with a comment saying
  the real one lived on the JavaScript side. From Rust you added ambient
  occlusion and got none. It now runs `renderer::ssao`, the same
  implementation the renderer's own post-fx uses, against the depth buffer the
  scene render leaves behind, and composites it over the colour.
  `ambient_occlusion_darkens_the_scene` puts a box on a floor under a single
  ambient light — so nothing *but* the pass can produce a gradient — and
  asserts that a real part of the frame is darker, that nothing is brighter,
  and that the strength knob moves it.

  The default radius went from 0.5 to 1.0 world units. There is no
  scene-independent answer, but 0.5 on a scene measured in metres reaches a
  hand's breadth and is nearly invisible, which reads exactly like the pass
  not working.

- **`SsrPass` declines instead of pretending.** It was the same silent copy,
  and unlike ambient occlusion it cannot be done honestly here: screen-space
  reflections need surface normals, the composer's chain carries colour and
  depth, and normals reconstructed from depth derivatives are wrong along
  every silhouette — which is exactly where a reflection is looked at. So
  `shader()` returns `None`, the composer spends no pass on it, and adding one
  logs why. The working implementation is the renderer's own post-fx, which
  has the normal prepass.

- **A camera's lens reaches the path tracer.** `PerspectiveCamera::aperture`,
  `focus_distance`, `f_stop`, `aperture_blades` and `anamorphic_ratio` were
  five public fields that nothing read — the tracer took its lens from
  `RaytraceSettings` — and two of them are *written by the USD loader from the
  file, so a camera authored at f/2.8 rendered as a pinhole and said nothing.

  `Camera::lens()` now returns a `Lens`, the path tracer uses it where the
  settings do not override, and the three shaping knobs do what they say:
  **`f_stop`** converts to an aperture radius (and documents the one thing
  such a conversion must assume — that a world unit is a metre, as Blender
  assumes — and why an explicit `aperture` therefore wins); **`aperture_blades`**
  gives polygonal bokeh from a uniform-by-area sample of a regular n-gon, not
  the bright-cored one a naive polar sample produces; **`anamorphic_ratio`**
  squeezes it. `PerspectiveCamera::film_gauge` (three.js's name and 35 mm
  default) and `focal_length_mm()` came with it.

  The sampler is written once in Rust and once in WGSL, and
  `the_two_lens_samplers_agree` evaluates the shader's arithmetic here and
  holds the two to 1e-5 across four irises and four thousand samples. The path
  tracer's own `concentric_disk` — a tenth copy of a function the crate had
  several of — now forwards to the shared one.

- **A test suite that checks the other suites can run.** A test file whose
  file-level `#![cfg]` the build does not satisfy compiles to an empty binary
  and reports `ok. 0 passed`, which a CI job reads as green;
  `cargo test --features "full,fluid,webgl"` prints exactly that for
  `tests/webgl.rs` and `tests/fluid_webgl.rs`. `tests/suite_gates.rs` — the one
  suite with no gate of its own — fails if a file needs a feature its
  `[[test]]` entry does not require (so the suite could never run anywhere),
  fails if a gated file has no manifest entry at all, and *names* the suites
  that are legitimately inert on this target along with how to run each. It
  found one real orphan on the first run: `captions_overlay_gpu` had no
  `[[test]]` entry, so it vanished under `--no-default-features`.

- **Post-processing passes read their own parameters.** `BloomPass`'s
  `strength`, `radius` and `threshold`, `OutlinePass`'s four, `ToneMappingPass`'s
  `exposure` and `FilmPass`'s four were public fields that **nothing read** —
  the composer's bind-group layout had two entries, texture and sampler, and
  every shader hardcoded its own constants. Setting one compiled, ran, and
  changed nothing, which is worse than the field not existing: the caller
  believes the picture responded.

  `Pass::uniforms()` now returns the sixteen floats behind `params` in the
  shader, re-read every frame so a value that moves — `FilmPass::time`, which
  now actually makes the grain crawl — reaches the GPU without recompiling
  anything. `every_pass_parameter_reaches_its_shader` moves each of the twelve
  one at a time and asserts the frame changes.

  Two defaults changed, to the values their shaders had always used:
  `BloomPass` to `{ strength: 0.9, radius: 0.02, threshold: 0.55 }` and
  `FilmPass::grain_intensity` to 0.06. Keeping the old nominal values and
  making the fields work would have changed every picture that uses these
  passes; this way nothing moves until someone asks it to.

- **A multigrid pressure solver for the fluid** (`PressureSolver::Multigrid`).
  Twenty Jacobi sweeps leaves 82 % of a splat's divergence in the field, and
  sixty-four times the sweeps only reaches 24 % — a Jacobi sweep moves
  information one texel, and the pressure from a localised puff is spread
  across the whole grid, so the shortfall is a rate rather than a floor. A
  V-cycle solves that part of the problem on a grid eight texels across: **one
  cycle leaves 0.217 against 1280 sweeps' 0.244**, at about three times the
  default's cost, and two cycles reach 0.100.

  Not the default — a visibly more incompressible fluid is a different fluid,
  and the original's look is what this is a port of. `m` in the example, a
  checkbox on the page.

  The smoother is *undamped*, against the textbook 4/5, and the reason is
  measured: weighting exists to kill the checkerboard mode, which plain Jacobi
  amplifies by exactly -1 and never removes — but this right-hand side has no
  checkerboard content, because the divergence is a central difference over two
  texels and that operator has a zero at Nyquist. Damping only slows the
  smoother on the modes that are there (0.175 against 0.208 at one cycle).

  A staggered-grid option went in alongside this and came back out. It was
  added on the theory that the floor was an operator mismatch — the divergence
  and gradient are *wide* differences while the sweep inverts the *compact*
  Laplacian — and measurement refuted it twice: three per cent on its own, and
  actively worse in combination with the solver that does work, for reasons
  that did not survive a boundary fix or an advection-free comparison. A knob
  that cannot be justified is worse than no knob.

- **Shared GPU plumbing**, pulled out of the places that had each grown their
  own. Writing the fluid simulation meant writing all of it again, which is the
  kind of evidence that settles the question.

  - **`threers::gpu`** — `GpuContext::headless()` / `with_surface()` for the
    instance, adapter, device and queue, with `renderer_limits` applying the one
    non-obvious rule: the main pass declares twenty sampled textures in the
    fragment stage against a downlevel allowance of sixteen, so a device opened
    with the defaults fails `Renderer::new` with a validation error from inside
    wgpu about a pipeline layout nobody wrote. Every example opened with thirty
    lines of this. `gpu::testing::context` is the same thing for a test, with a
    *printed* skip when there is no adapter — a GPU test that silently returns
    is indistinguishable from one that passed.
  - **`renderer::fullscreen`** — `Fbo`, `DoubleFbo`, the fullscreen-triangle
    vertex stage, `begin`/`draw`, the clamped samplers, and `FormatPipelines`
    for the pipeline-per-target-format cache. Twelve files had their own
    fullscreen vertex stage. Also `Target`, a view bundled with the format and
    size a `wgpu::TextureView` does not carry — the three arguments that have to
    agree, and that stop agreeing at a resize.
  - **`renderer::readback`** — `f16_to_f32`, the row-padding fix, and
    `texture_bytes` / `texture_channel_f32`. **Three of the eight separate
    half-float decoders in this crate were wrong**, each decoding subnormals at
    a half or a quarter of their value through an off-by-one in a renormalising
    loop; one of the eight carried a comment recording that the bug had been
    found and fixed *there*. All nine call sites now forward to one
    implementation, which is checked against all 65 536 half floats rather than
    a handful of spot values. Subnormals are everything below 6.1e-5 — the deep
    shadow end of an EXR, the almost-black of a float texture.
  - **`renderer::colorspace`** — the sRGB transfer function as a WGSL prelude,
    so a shader outside the renderer can have the same curve instead of writing
    out a third copy. The exact piecewise function, not a 2.2 power, and a test
    holds it to `utils::srgb`'s numbers.
  - **`postprocessing::DrawPass`** — a composer pass that draws itself. The
    others are *descriptions*: a name and a fragment shader the composer
    compiles and runs over the previous buffer, which cannot express anything
    that owns GPU state. A simulation, a particle system or a second scene now
    joins the chain rather than wrapping it, and gets the view the next pass
    will read along with its format and size.
  - **`Camera::look_at`** is on the trait, so it works through `&mut dyn
    Camera` and `OrthographicCamera` has it at all — it did not, which at a call
    site reads as the method not existing rather than as a camera that cannot be
    aimed. In three.js it comes from `Object3D` and every camera has it.
  - **Eight duplicate fullscreen vertex stages are gone**, replaced by
    `fullscreen::VS`: bloom (both paths), ambient occlusion (both, including
    its test fixture), the supersample resolve, the accumulation pass, the
    caption overlay, the mip generator and PMREM. The mip generator's
    pipeline-per-format map became `FormatPipelines` as well. The
    post-processing composer keeps its own, deliberately: it puts uv (0,0) at
    the bottom left, which cancels because the chain both reads and writes
    through it, and switching would move `GlitchPass`'s bands and `FilmPass`'s
    scanlines for no gain. The fluid keeps its two, which emit the four
    neighbour taps its stencils read.
  - `web/build.sh` honours `CARGO_TARGET_DIR`. It hardcoded `target/`, so a
    build with the variable set compiled successfully and then failed in
    wasm-bindgen, looking for an artifact cargo had just written elsewhere.

- **A fluid simulation** (`fluid`). Dye poured into a velocity field that is
  advected, confined, made divergence-free and dissipated sixty times a second:
  drag a pointer across it and the ink follows the motion, curls into vortices
  behind the stroke, and diffuses away. A port of Pavel Dobryakov's *WebGL
  Fluid Simulation* onto wgpu, which is Stam's *Stable Fluids* in the shape GPU
  Gems 38 gives it — curl, vorticity confinement, divergence, twenty Jacobi
  sweeps of ∇²p = ∇·u, gradient subtract, and semi-Lagrangian advection of
  first the velocity and then the ink — with bloom, sunrays and the dye lit as a
  height field over the top. `FluidSim` owns its own textures and nothing else:
  hand it a device and a texture to draw into and it does both halves.

  **There is no compute stage anywhere in it.** Nineteen GLSL programs became
  one WGSL module with nineteen entry points over a single bind group, all of
  them fullscreen fragment passes, which makes this the one substantial part of
  the crate that is *complete* on WebGL2 rather than reduced — bloom and ambient
  occlusion have raster paths standing in for compute ones; this has no compute
  path to stand in for. What it needs there instead is a filterable half-float
  render target, so `FluidFormats::for_adapter` probes and widens one channel
  count at a time, `r16float` → `rg16float` → `rgba16float`. Checked in headless
  Chrome on read-back pixels: `tests/fluid_webgl.rs`.

  **Five quality levels**, two of them past anything the original had — it was
  written for 2017 mobile browsers. `ultra` carries a 2048-texel dye field and a
  256-texel solver, `extreme` 4096 and 512, and both are clamped to what the
  device will actually allocate, scaling *both* axes together so the texels stay
  square. `grid_sizes()` and `memory_bytes()` report what happened, because a
  quality level is a request and the device has the last word.

  The finding worth repeating is the one that looks like a bug and is not. The
  original subtracts **twice** the pressure gradient: `(pR - pL, pT - pB)` is a
  central difference over two texels, and the 0.5 that GPU Gems calls `halfrdx`
  — the same 0.5 the divergence pass and the Jacobi sweep both carry — is
  missing from that one shader. Measured on a single splat, it is a
  compensation. At the default twenty sweeps the overshoot leaves 0.654 of the
  divergence where the consistent coefficient leaves 0.825, because Jacobi is
  nowhere near converged and doubling the correction puts back some of what the
  unconverged solve left out. At three hundred and twenty sweeps the order
  reverses and the overshoot leaves 1.269 — a field *more* compressible than not
  projecting at all. So it is a knob (`gradient_scale`, 1.0 by default) that
  belongs with the sweep count, and both halves of that are an assertion rather
  than a comment.

  The simulation also runs in a frame where y points down the screen, because
  WebGPU's texture origin is the top left and OpenGL's is the bottom left.
  Reflection is a symmetry of Navier-Stokes and of this confinement term too, so
  every shader ports verbatim and the result is the mirror image of the same
  initial conditions — the same fluid, and pointer coordinates that need no
  `1 - y`. `a_splat_lands_where_it_was_asked_for` pins it: get the handedness
  wrong and everything still runs, still looks like a fluid, and is upside down.

  `cargo run --features fluid --example fluid` for a window,
  `--example fluid_render` for a deterministic PNG, `FLUID=1 web/build.sh` and
  `web/fluid.html` for the browser. See [docs/fluid.md](docs/fluid.md).

- **`ShaderMaterial` reaches JavaScript.** `WebShaderMaterial` in the wasm
  bindings and `THREE.ShaderMaterial` in the shim, with the vertex stage, the
  uniform block, the three storage slots, transparency, side, depth write and
  `validate()`. The class existed in the shim already — as a stub that built a
  *basic* material and rendered a flat colour, ignoring both shaders, which is
  worse than its being absent: three.js code using it got a picture, just not
  the one it asked for.

  The storage slots are the part that had to be proved in a browser rather than
  on a desktop, because WebGL2 has none: a test draws a shader material whose
  brightness comes out of `s0(3u)` on a real WebGL2 context, and reading index 0
  instead turns the frame black, which is how we know the texture fallback is
  what is answering.

- **Benchmarks.** `cargo bench` covers the four places a scene spends CPU before
  anything is drawn: geometry generation, PMREM prefiltering, asset parsing and
  BVH build/raycast. All of it runs without an adapter, deliberately — a
  benchmark that needs a GPU is one CI will not run, and one CI does not run is
  a number nobody is watching. On an M4 Pro: a 64×32 sphere is 16 µs, a 256-face
  PMREM chain 50 ms, and decoding a 50k-vertex Draco bunny 15 ms.

- **A `ShaderMaterial` can replace the vertex stage.** `vertex: Option<String>`
  takes WGSL defining `@vertex fn vs_custom`, and the whole preamble is in
  scope — including `standard_vertex`, which is the built-in transform split
  out of `vs_main` so a custom stage can *start* from the standard output and
  move it rather than re-derive a transform that already handles displacement
  maps and the sky-depth case.

  Until now a custom material could only change the colour of geometry it was
  handed, so anything whose position has to be computed — a camera-facing
  ribbon, a strand expanded from control points, a billboard — had to be
  rebuilt on the CPU and re-uploaded whenever the camera moved. The user
  bindings are visible to the vertex stage now too, so per-vertex data can come
  from `u_data`, the storage arrays or the user textures, indexed by
  `@builtin(vertex_index)`.

- **`ShaderMaterial::validate`, and shader errors that name your own line.** A
  broken custom shader used to reach `create_shader_module`, which does not
  return an error: it goes to the device's uncaptured-error handler and takes
  the process down, from inside the draw loop, with a line number into a module
  the caller has never seen — you write forty lines and the error says
  `wgsl:1707`.

  The assembled module is now parsed and validated with naga before wgpu sees
  it. At render time a shader that will not compile is logged once and its
  object is skipped; the rest of the frame draws. And `validate(storage_buffers)`
  does the same check with no GPU and no renderer at all, so a broken shader is
  a unit test — including against the *other* backend, which is how you find
  that `u_s0` does not exist on WebGL2 without owning a browser.
  `THREERS_DUMP_SHADER=/tmp/shader.wgsl` writes the module out, which is the
  only way to read line 1707 for yourself. Native only: naga is not in wgpu's
  browser build, and `ShaderMaterial::VALIDATION_AVAILABLE` says so rather than
  the method quietly returning `Ok`.

- **Storage arrays that work on WebGL2.** `s0(i)`/`s1(i)`/`s2(i)` are defined on
  every backend; where the device has no storage buffers the same data is
  uploaded as an `Rgba32Float` texture and read with `textureLoad`. `u_s0`..`u_s2`
  stay for shaders that index them directly, but they only exist where the
  device does — which previously capped a browser-bound `ShaderMaterial` at the
  sixteen `vec4`s of its uniform block and ruled the feature out for anything
  data-driven. Verified on a real device by asking wgpu for one with
  `max_storage_buffers_per_shader_stage = 0`, which is what WebGL2 reports.

- **`ToneMapping::Neutral` and `ToneMapping::AgX`.** Khronos PBR Neutral is
  nearly the identity below the shoulder — it subtracts a small black offset and
  otherwise leaves the value alone, so an authored colour reaches the screen as
  itself where ACES lifts the same midtones from 0.18 to 0.27. AgX has a long
  shoulder: 3.0 lands at 0.82 where ACES is already at 0.95.

  `apply_tone_mapping` in the shader preamble is the shared dispatcher, so a
  `ShaderMaterial` ending `framebuffer_encode(apply_tone_mapping(hdr))` responds
  to `set_tone_mapping` like everything else instead of carrying its own curve.
  `ToneMapping::apply` is the CPU form, which `SvgRenderer` and the path
  tracer's film now share rather than each keeping a copy — and a test compares
  all five modes against the WGSL across 50 colour/exposure pairs, which is how
  the CPU ACES was found to carry a `* 0.6` pre-exposure the shader does not.
  A whole stop, and the only symptom would have been an SVG that came out darker
  than its PNG.

- **Depth of field.** `HeadlessRenderer::set_depth_of_field(focus, range,
  radius, near, far)`: a golden-angle spiral gather weighted so a sample only
  contributes if its own circle of confusion reaches the pixel — which is what
  stops a sharp foreground smearing over a blurred background. The tap count
  scales with the area being gathered, because a fixed 24 taps at a 14-pixel
  radius sit three pixels apart and produce speckle rather than blur. Anything
  at the far plane counts as fully defocused, so a backdrop with no geometry in
  it falls away the way a real one does.

- **`ShaderMaterial::depth_write`** overrides whether a custom material writes
  depth, so a dense transparent shell can self-occlude. **`Scene::add_mesh`**
  puts geometry and a material in a scene in one call rather than three nested
  constructors. **`PerspectiveCamera::fov_degrees`**, `set_fov_degrees` and
  `pixels_per_unit` close a trap this release walked into: `new` takes degrees,
  the field is radians, and `camera.fov.to_radians()` reads correctly, is wrong
  by a factor of 57, and still draws a picture.

- **`examples/raspberry` — a raspberry, procedurally.** 117 translucent
  drupelets, the dried styles, and forty-eight thousand hairs, lit by a studio
  of three soft boxes and nothing else. No model, no textures, no download: the
  geometry, the environment and every map are built in the example.

  It exists because a berry is the shortest route to the three things a
  rasteriser is usually bad at, and each one is a `ShaderMaterial` carrying its
  own WGSL.

  **It is not one surface.** The skin is 117 overlapping spheres with the faces
  nobody can see removed — 50k triangles out of 95k — because the crevices
  between the sacs are where nearly all the shading interest is. Drupelet
  centres and radii travel in a storage buffer and `vertex_color.a` is the index
  into it, which is what lets the fragment stage treat each sac as a *body*
  rather than as a patch of surface.

  **Each sac is a bead of juice.** Light enters on the lit side and leaves past
  the terminator; from the limb you can see the soft boxes through the flesh,
  Beer–Lambert against mean free paths of 2.2 mm, 0.22 mm and 0.13 mm; and there
  is a pale seed inside. All three need the bead's centre and local radius, so
  all three come from the geometry rather than from a rim term. The berry is
  modelled at 1 unit = 1 cm precisely so those distances mean something.

  **The fuzz is most of the silhouette.** A hair is about fifteen microns across
  against a two-centimetre berry, so at any sensible resolution it covers a
  fraction of a pixel: drawn at its true width it is hit or missed depending on
  where the pixel centre lands. Each hair is a camera-facing ribbon that is
  never allowed below a pixel wide — widened and faded by the same factor, so
  coverage survives and the sparkle does not — and the part too fine to draw at
  all is accounted for by a sheen term on the skin.

  The studio is generated as an equirectangular HDR, prefiltered to a CubeUV
  atlas for the reflections, and projected onto nine spherical harmonics for the
  diffuse. The panels sit about a thousand to one above the room, which is not
  decoration: prefiltering averages a dim panel down to the ambient, and at
  thirty to one — where this started — there is no specular highlight anywhere
  in the frame.

  `cargo run --release --example raspberry`, and `--fuzz`, `--drupelets`,
  `--exposure`, `--lamp` and `--turntable` to take it apart.
  `RASPBERRY_DEBUG=diffuse|specular|glow|ao|dome|free|env|normal` shows one
  addend at a time, which is how shading this layered is actually debugged.

- **A size-tuned wasm profile, and pre-compressed output.** The web build now
  uses `release-wasm` rather than `release` — the same optimisations plus
  `opt-level = "z"`, fat LTO, one codegen unit, `panic = "abort"` and `strip`.
  Deliberately a separate profile: `release` also governs the native build, where
  the renderer and the path tracer want speed. On `mini` + `webgl`, through the
  real pipeline, brotli as the browser receives it: **1.089 MiB becomes 0.785**,
  and the raw artifact 5.41 MiB becomes 2.86. Plain `mini` drops from 0.39 to
  0.30 MiB. The build also writes `.br` and `.gz` beside the wasm so a static
  host can serve them with `Content-Encoding`.

  Three measurements worth recording, two of them the opposite of what one would
  guess. **`opt-level = "z"` beats `"s"`** here, 2.86 MiB against 3.17, which is
  not a given since `"z"` gives up loop vectorisation. **Brotli beats zstd** on
  this artifact, 0.785 against 0.830 at `--ultra -22`, so there is no reason to
  add a third encoding, and a larger brotli window buys nothing because the file
  already fits the default one. And **`wasm-opt` moves the file and the download
  in opposite directions** — see below.

- **wasm-opt is part of the toolchain.** It used to run only if it happened to be
  installed, which on this machine it never was, so the build's "optional size
  pass" had quietly never executed. It is now taken from `PATH` if present and
  otherwise fetched at a pinned `binaryen@132` through `npx` — no admin rights,
  nothing installed system-wide, the same route the browser tests use for Chrome.
  `WASM_OPT_BIN` overrides the binary and `WASM_OPT=0` skips the pass.

  It runs by default, and it is a trade rather than a free win. On mini+webgl:

  | | raw | brotli | V8 compile |
  |---|---|---|---|
  | without | 2.86 MiB | **0.785 MiB** | 6.7 ms |
  | with `-Oz` | **2.61 MiB** | 0.812 MiB | **5.1 ms** |

  Nine per cent off the file and a quarter off the browser's compile time,
  against twenty-seven kilobytes more to download — the transfer rises
  consistently across gzip, brotli and zstd, because LTO has already collapsed
  the duplication the compressor was exploiting. Which side wins depends on the
  connection. Stripping alone buys nothing: `strip = true` and wasm-bindgen have
  already removed those sections, and `--strip-debug --strip-producers` leaves
  the file slightly larger from re-encoding.

  `panic = "immediate-abort"` would shave more but needs `cargo-features` at the
  top of the manifest, which makes the whole crate nightly-only — evaluated and
  rejected.

- **The web build can include the WebGL2 fallback.** `WEBGL=1 ./web/build.sh`
  adds it to any variant, and it stays off by default, including in `wasm-full`.
  The cost on the wire is about **half a megabyte** — 0.30 MiB becomes 0.785 —
  for the GL backend's glow and naga GLSL translator, which a WebGPU-only
  deployment never executes. Worth it where the alternative is a blank canvas.
  (An earlier draft of this entry quoted the *uncompressed* delta of 3.85 MiB as
  though it were the download; compression recovers most of it, because a shader
  translator is repetitive code.)
- **WebGL2 renders.** WebGPU is not everywhere, and a browser
  without it got a blank page. The `webgl` feature compiles wgpu's GL backend in
  alongside, preferring WebGPU wherever it exists, and
  [`renderer::caps`](src/renderer/caps.rs) asks the device what it can do rather
  than assuming: the compute-backed passes — bloom, ambient occlusion, the packed
  post chain — now decline on a device with no compute stage instead of building
  a pipeline that was never created, and `ShaderMaterial`'s storage bindings are
  left out of the module, the layout and the bind group together where storage
  buffers do not exist. `WebRenderer.backend()` and `.supportsCompute()` say
  which one you got, because otherwise the only symptom is that some effects
  quietly do nothing.

  Two things about the browser forced the shape of this. wgpu's
  `Instance::new` commits to *one* context implementation on wasm, so
  `BROWSER_WEBGPU | GL` does not mean "try both" — it means WebGPU, and the
  fallback then fails claiming `gl support not compiled in`, which is not what
  has gone wrong. And a canvas hands out exactly one kind of context for its
  whole life, so creating a surface to *test* for WebGPU destroys the WebGL2
  fallback it was testing for. Each backend gets its own instance, and the canvas
  is not touched until one has answered.

  Verified in a real browser rather than by compiling: `tests/webgl.rs` runs in
  headless Chrome and checks that a GL adapter is offered, that the device
  reports what WebGL2 actually has, that the compute passes decline, and that a
  render pass and readback produce exactly the pixels asked for.

  Eight things stood between the backend and a rendered frame, every one found
  by running it: `ShaderMaterial`'s storage buffers against a limit of zero;
  `textureNumLevels` for the PMREM chain length, which GLSL ES 3.0 cannot express
  and which is not recoverable from `textureDimensions` because a PMREM chain is
  shorter than a full one — now carried in the uniform that already held it for
  the CubeUV path; depth `textureLoad` in the refraction pass, the temporal
  resolve and post-processing, all three now binding their depth as unfilterable
  *float* rather than `texture_depth_2d` — which a depth format permits, lowers
  to `texelFetch`, and costs nothing on any backend, so the temporal pass and the
  depth-driven post effects work on WebGL2 as well; `INDEPENDENT_BLEND` for
  order-independent transparency, skipped; **seventeen texture units against
  sixteen**; and `VIEW_FORMATS`, which render targets declare and WebGL2 does not
  have.

  That texture-unit one is worth stating precisely, because it is a layout WebGPU
  validates and WebGL2 cannot run. The standard material binds eight sampled
  textures in the fragment stage from the material group and eight from the
  environment group — sixteen exactly, the per-stage floor. There is also a
  displacement map in the *vertex* stage, and WebGL2 has no per-stage pool:
  wgpu-hal's GL backend assigns units across the whole pipeline from one array of
  sixteen. Nothing is over-declared per stage, so there is no validation error,
  only `index out of bounds: the len is 16 but the index is 16` from inside the
  draw. The fix is that units are assigned per *layout* entry and not per use —
  the draw that panicked sampled none of the textures that overflowed it — so a
  pipeline only has to declare less, not do less. `fs_ss_glass` samples no shadow
  map, so screen-space refraction now has its own group 3 with the capture where
  those three were; the main pipelines keep the shadow maps and drop the capture.
  Both layouts come to fourteen, both fit, and refraction works on WebGL2.
- **Bloom and ambient occlusion run without a compute stage.** Both were
  unavailable on WebGL2 along with the rest of the compute passes, and neither
  needed compute to begin with: bloom is a threshold and downsample followed by a
  separable blur in two passes, and ambient occlusion reads a depth texture and
  writes one value per pixel. All of that is as readily a fullscreen draw.
  `Bloom::new` and `Ssao::new` now pick a raster path where there is no compute
  stage, behind the same interfaces, so nothing downstream changed.

  The occlusion port keeps every line of the arithmetic — the linearisation, the
  nearest-side derivatives, the twelve golden-angle taps — and changes only how
  the invocation learns which pixel it is and where the result goes, plus one
  thing it had to: the depth slot is declared as an unfilterable *float* texture
  rather than a depth texture, since WGSL allows `textureLoad` on the latter and
  GLSL has no `texelFetch` for a depth sampler.

  What is left unavailable on WebGL2 is two things, and neither is a port that
  was skipped. Order-independent transparency needs `INDEPENDENT_BLEND` — two
  attachments with different blend equations — which is a hardware capability,
  and transparency falls back to the sorted path that three.js uses anyway. The
  packed post chain is compute-only but is not a browser concern at all: it
  exists for `HeadlessRenderer`'s video read-back, which is the native path.

  Each raster path is compared against the implementation it reproduces on a
  device that has both — the only place that comparison can be made, since the raster paths exist
  for the backend where the compute ones cannot run. Bloom agrees to 0.000244,
  one half-float step at the peak; occlusion to better than 1e-5. Both bars came
  from the measurement, and both tests were checked by mutation first: dropping a
  blur weight moves bloom by 0.0016, and dropping the nearest-side derivative
  moves occlusion by 0.025.

- **Shader materials work on baseline WebGPU.** They were absent there, not
  broken: the custom layout declared the four user textures on top of the eight
  material maps and the environment group, overflowing the per-stage limit of
  sixteen, so it could not be created at all on a device reporting the floor —
  Chrome included — and the feature was simply missing.

  A `ShaderMaterial` carries a fragment source, a uniform block and its own
  textures; it has no standard maps and no displacement map, so the eight
  material textures were *definitionally* unused by one. A custom pipeline now
  gets a group 2 holding only the sampler and the displacement slot, which the
  shared vertex entry needs whether or not the flag selecting them is set,
  because naga counts the use and not the branch. Seventeen fragment textures
  down to nine.

  There was no render-time coverage of shader materials at all — an example was
  the only thing exercising them — so `tests/shader_material_render.rs` now
  checks that a custom fragment draws with the colour its uniform block carries
  and that a user texture arrives in `u_tex0`.

  Two bugs that were never WebGL-specific surfaced on the way, latent until
  something tried a second canvas or a read-back off WebGPU. A `wgpu::Device`
  cannot be paired with a surface from a *different* instance, so the instance
  that won backend discovery is now kept with the cached device — and on WebGL2
  no sharing is possible at all, because a context belongs to its canvas. And
  `readRenderTargetPixels` blocked the event loop: `wait_indefinitely()` stops
  the very thread that would deliver the mapping callback, which WebGPU's backend
  hid by making `poll` a no-op. It polls and yields now.

  The read-back assertion is on pixels, not on the absence of a panic — a draw
  that binds nothing and writes nothing raises no error at all. See
  [docs/webgl.md](docs/webgl.md).

- **Video decoding, in process.** The crate could encode H.264 and HEVC and mux
  an MP4, and could read none of it back — so any pipeline that wanted a clip
  *in* had to shell out to ffmpeg and read raw frames off a pipe. That is the
  cost `videotoolbox` exists to avoid on the way out: 8 MB a frame at 1080p and
  99 MB at 8K, written by one process and read by another, in the middle of a
  frame loop.

  `codec::mp4_demux::demux_video` reads the video track's sample table — `stsd`
  for the codec and its `avcC`/`hvcC`, `stts` for durations, `stsz`/`stsc`/
  `stco` for where the samples are, `stss` for which of them a decoder can
  start on — and says where every coded frame lives.
  `videotoolbox_decode::VideoToolboxDecoder` hands those to a
  `VTDecompressionSession` and gets BGRA back, asking the hardware to do the YUV
  conversion on the way past rather than writing one. Measured on a 1080p H.264
  clip: **300 frames in 1.23 s, 244 fps**, every frame producing a picture.

  **Decode order is not display order**, and the demuxer says so. A B-frame is
  coded after the picture it is shown before, which MP4 expresses as a `ctts`
  composition offset per sample; `Sample` carries `dts` and `pts`, and
  `VideoTrack::presentation_order` sorts them. `VideoToolboxDecoder` takes the
  timestamps with each sample, decodes with temporal processing on, and queues
  pictures by presentation time — `push` to feed in coded order, `take(pts)` to
  pull in display order. Reading the samples in file order and calling that the
  clip plays it shuffled in a short repeating cycle, one per mini-GOP. Caught
  frame by frame on a real render: with the pose held still the screen's
  frame-to-frame difference ran 4.8, 2.3, 1.6, 0.8, 1.6, 0.9 and round again
  while the source's was a flat 1.23, and its correlation against the source
  was 0.07. In display order it is 0.875 at lag zero.

  Round-trip tests mux an MP4 with `codec::mp4` and read it back sample for
  sample, which is the one reader/writer pair where both halves are here, and
  a hand-built `ctts` pins the reorder.

  Fragmented MP4, multiple sample descriptions and edit lists are refused rather
  than guessed at. `CMBlockBufferCreateWithMemoryBlock` is given
  `kCFAllocatorNull` and not a null allocator — the parameter is what CoreMedia
  will *deallocate* the block with, and the default one happily tries to free a
  Rust `Vec`, which aborts the process.

- **A converging accumulation mode**, and multisampling no longer switches the
  temporal pass off. `Renderer::set_taa_converge(true)` turns the TAA pass from
  a feedback blend into a running mean: hold the pose, call `render` `n` times,
  and the frame target holds the average of `n` jittered samples. The two steps
  that make ordinary TAA work on a moving image — reprojecting the history and
  clamping it to the current pixel's neighbourhood — are exactly what stops it
  settling when nothing is moving, so the converge path skips both. Measured on
  a shallow synthetic edge, the deviation from the straight line it should be:

  ```text
  1 sample                    0.285 px
  8 accumulated               0.047 px
  32 accumulated              0.023 px
  32 through ordinary TAA     0.249 px   (the clamp pins it near one sample)
  ```

  The point of it over `supersample` is memory. Supersampling renders at
  `size × factor`, which runs out of texture before it runs out of usefulness:
  3× of 4320×7680 asks for 23040×12960, and wgpu refuses. Accumulation holds one
  buffer at output size however many samples are taken, so the same convergence
  is available at any resolution. It is also *better per sample* — the jitter is
  a Halton sequence rather than a regular grid, so what is left is noise that
  averages out rather than a pattern that folds back. On a rotating phone's side
  rail, 32 accumulated samples measured 0.21 rms banding against 0.30 for the
  36 samples of `supersample(6)`, and 0.17 against 0.23 at 8K, where 32 samples
  a pixel cannot be had any other way.

  `HeadlessRenderer::render_accumulated(scene, camera, n)` is the one-call form,
  and it is also where the cost comes down. Two things were being paid for on
  every sample and needed once:

  * **The copy back into the frame target.** The mean lives in the history
    buffer; only the last sample has to land anywhere a caller will read. Every
    copy before it moved a full-resolution image nobody looked at — 133
    megapixels, twice, per sample, at 8K.
  * **Accumulating in the framebuffer's format.** The history was built at the
    colour format, so an eight-bit target meant an eight-bit running mean — and
    a running mean folds each sample in at `1/(n+1)`, which is below the quantum
    long before it has settled. By the thirty-second sample the increment
    rounded to nothing and every sample after it was thrown away: the caller
    kept paying and the image stopped improving. Converging now accumulates in
    `Rgba16Float`, one byte a pixel more at output size. On the synthetic edge
    with 4x MSAA, thirty-two samples went from 0.0174 px to **0.0119**.

  Together those took a three-hundred-frame 8K render from 160 s to 125 s, and
  1080p from 27 s to 22 s.

  A third thing was being paid for on every sample and is now paid once per
  frame at a quarter the size: **the mean itself**. It ran where the temporal
  pass sits, which is *before* the supersample resolve, so at `supersample(2)`
  it read and wrote four times the pixels of the frame it was building, and held
  two histories at that size. Averaging and the resolve are both linear, so
  doing the resolve first and the mean after is the same picture — as long as
  both means are taken in the same space, which is the fix above. The mean now
  runs at output resolution in `HeadlessRenderer`, in one buffer rather than a
  ping-pong pair.

  The saving is memory, not time. At 8K the internal history was two buffers of
  8640×15360×8 bytes; peak footprint for a render fell from **11.07 GB to
  8.78 GB**. Wall time did not move: 74.3 ms a frame against 76.9 ms at 1080p,
  inside a ±6 ms run-to-run spread. The bandwidth the reordering saves is real
  but the resolve it now runs sixteen times instead of once gives most of it
  back, and the first attempt was 11% *slower* until the per-sample texture
  views and bind groups were cached. Worth having for the headroom; not worth
  reaching for as a speed-up.

  The second half of this is a fix. `HeadlessRenderer::render` took a
  surface-style path whenever MSAA was on, and that path passed no colour source
  — which is what the temporal pass is gated on — so **MSAA silently disabled
  TAA**: no error, no warning, and a quality setting that did nothing. Measured
  on that same rail, 32 accumulated samples took the banding from 0.99 to 0.28
  with MSAA off and changed it by nothing at all with MSAA on. There is now
  `Renderer::render_to_multisampled`, which hands the pass what it needs, and a
  test pinning that the two compose.
- **Draco decodes.** `KHR_draco_mesh_compression` is the compression most `.glb`
  in the wild uses, and such a file used to be refused outright. It is several
  formats stacked: an rANS entropy layer, an edgebreaker connectivity coder that
  has to be *replayed* into a corner table, attribute seams that split one
  position into several points, and per-attribute prediction schemes on top.
  All of it is a port of Draco's own decoder rather than a reading of a spec.

  Both traversals are implemented, including the **valence** one — which matters
  more than it sounds: `draco_encoder` defaults to `-cl 7`, and anything at
  decode speed 4 or below codes the CLERS symbols by vertex valence instead of
  writing them out. Those symbols cannot be read ahead of the replay at all, so
  the replay pulls them one at a time from six streams chosen by the valence of
  the vertex it is standing on *in the partly-built mesh*. Prediction: difference,
  parallelogram, constrained multi-parallelogram, geometric normal, and the
  portable texture-coordinate predictor that reconstructs each UV from the
  triangle's shape in position space. Transforms: wrap and canonicalized
  octahedral. Quantized floats, octahedral normals and plain integers all come
  out.

  Verified against Google's `draco3d` on the same files, stage by stage: the
  connectivity by its **valence histogram** over unique positions, the index
  buffer **index for index**, and every attribute value **bit for bit** — both
  implementations do the same arithmetic in the same order on `f32`, so agreement
  is exact and anything approximate would mean an operation had been reordered.
  Four fixtures across four topologies and three encoder settings: a closed
  sphere, the Stanford bunny with its five holes, an open disc, and a
  default-encoded grid. Through glTF, the decoded face set is compared against an
  **uncompressed twin** of the same mesh — geometry that never went through this
  code. See [docs/draco.md](docs/draco.md).

  Still refused, each by name: the prediction-degree traversal (`-cl 10` only),
  multi-parallelogram and the deprecated UV predictor, uncanonicalized octahedral
  normals, the delta transform, sequential connectivity, point clouds and
  metadata blocks. A geometry decoder that guesses returns a plausible mesh of
  the wrong shape, so none of them fall through to a default.
- **glTF export.** The format was a one-way door: read since the beginning, and
  no way to write it, so a scene built or edited here could not go back to the
  tools it came from. `scene_to_glb` and `scene_to_gltf` write the scene graph
  with its transforms, positions, normals, texture coordinates, vertex colours,
  indices, and materials as `pbrMetallicRoughness` with their maps embedded as
  PNG. A shared geometry or texture is written once, so a forest of identical
  trees costs one mesh rather than a thousand. Both forms pass the glTF
  validator with no errors, warnings, infos or hints.
- **The new loaders are integrated, not merely present.** `draco` and `alembic`
  were not re-exported at all; every new type now appears in `loaders`, the crate
  root and the prelude, matching how the older loaders are exposed. Browser
  bindings were added for KTX2, Alembic, glTF export and — for the first time —
  **USD**, which had none. The `THREE.*` shim gains `AlembicLoader` and
  `USDLoader`. A binding that exists only in some builds is looked up on a
  namespace import rather than named, because a named ESM import of a missing
  symbol fails at load time and takes the whole shim down with it.
- **KTX2 textures.** glTF's container for anything GPU-compressed, and what an
  ARKit `.usdz` leans on. The decode target already existed — `TextureFormat`
  carries block-compressed data with a mip chain — so what was missing was the
  container: the header, the level index, the data format descriptor, key/value
  metadata, and zlib supercompression. Mip chains arrive largest-first, which is
  the order the chain wants. Basis Universal (`ETC1S`/`UASTC`) and zstd are
  refused *by name*, because "unsupported" alone leaves a caller guessing which
  thing to reach for. So is a **linear** BC payload: this crate's block formats
  are all sRGB, and labelling a normal map sRGB is wrong in a way that looks like
  bad lighting rather than like a bug.
- **Alembic (`.abc`).** The Ogawa container — groups of 64-bit references and
  data blocks — then the object hierarchy, property headers, and `PolyMesh`
  geometry: points, face counts and face indices, which is what a baked cache
  is. Two details wreck everything after them if missed, and both are now
  handled: an object-header block ends with **32 bytes of hashes** that are not
  headers, and every array sample begins with a **16-byte key** that is not
  data. A file whose completion byte is still clear — a cache being written — is
  refused rather than read from offsets that are not final yet.
- **`EXT_meshopt_compression`.** A `.glb` compressed with meshopt used to fail
  outright, which is a large share of the real files on the web. The vertex
  codec (both versions, including v1's per-channel control modes), the triangle
  codec with its edge and vertex FIFOs, the index-sequence codec, and all four
  filters — octahedral, quaternion, exponential and YCoCg colour. Ported from
  meshoptimizer's *reference* decoder rather than from the specification: a
  codec written from recollection decodes to plausible numbers instead of to an
  error, and the numbers here are vertex positions.
- **FBX, read as a scene.** The loader searched the file for the first
  `Vertices` array and stopped — one mesh, positions only, no normals, UVs,
  hierarchy or materials, and no tests at all. Both serialisations now parse to
  one node tree, and what comes out is every mesh in the file, in its place in
  the hierarchy, with its transform and material. Checked against `assimp`:
  agreement to the digit on a 2174-triangle torus/sphere/cone file.

- **USD — `.usda`, `.usdz` and `.usdc`**, in pure Rust with no external
  dependency, so it works on wasm where OpenUSD's C++ cannot go.
  `UsdLoader::parse` takes any of the three and tells them apart by content
  rather than by file name; `geometry_to_usda`, `geometry_to_usdz` and
  `scene_to_usdz` write them. Meshes carry points, face counts and indices,
  normals and `primvars:st`; transforms come from the `xformOp` stack that
  `xformOpOrder` names; `UsdPreviewSurface` materials and the prim hierarchy
  come across too. Faces of any size are fanned into triangles and a
  `faceVarying` primvar expands the mesh to unindexed triangles, because an
  indexed buffer cannot give one point two normals. `.usdz` is written to the
  rules that make it mappable — stored, never compressed, every file's data on a
  64-byte boundary — while still reading the deflated archives some tools
  produce. USD lives behind the **`usd` feature**, off by default: it is pure
  code with no dependencies of its own, so turning it on costs compile time and
  nothing else, and most scenes never touch it.
  **`.usdc` is read and written in full** — LZ4 sections, USD's two-bit integer
  packing, the `ValueRep` word, the path table's depth-first walk and the spec
  table — through the same entry point as the other two. It was written against
  files produced by OpenUSD's own `usdcat`, which is the only way to get it
  right: the type enumeration numbers matrices ahead of vectors, a quaternion
  is stored imaginary-part-first but written real-part-first, an inlined double
  holds `f32` bits, and a string is indexed one level deeper than a token — four
  things a reader checked only against its own writer would agree with itself
  about. Writing turned up four more of the same kind: a `float` must be carried
  inline or USD reads the file offset as its bits; a relationship's variability
  defaults to *uniform* where an attribute's defaults to varying; a connection
  is the attribute it connects rather than a property with a dot in its name,
  which is not a name USD can hold; and a metadata field's type comes from the
  schema, so `timeCodesPerSecond = 24` is a double however it was written.
  `geometry_to_usdc`, `scene_to_usdc`, `animated_scene_to_usdc` and
  `layer_to_usdc` write it, `layer_to_usdz` puts either form inside an archive,
  and names, values and field sets are interned so repeated structure costs
  almost nothing. The tests read `.usda` and `.usdc` of one document through
  independent paths and require them to agree; read back a document this crate
  exported after `usdcat` converted it to binary; and hand the crate files this
  crate writes to `usdcat`, diffing its output against OpenUSD's own
  normalisation of the source — identical, which is more than "it loads".
- **USD schemas beyond `Mesh`.** Lights (`DistantLight`, `SphereLight`,
  `DiskLight`, `CylinderLight`, `RectLight`, `DomeLight`, with `ShapingAPI`
  cones making spots), the quadric gprims (`Cube`, `Sphere`, `Cylinder`, `Cone`,
  `Capsule`, `Plane`, each turned to the `axis` it declares), `Points`,
  `BasisCurves` and `Camera` — which comes back on `UsdScene::cameras`, since
  nothing in `ObjectKind` is a camera. `purpose = "guide"` is not drawn and
  `visibility = "invisible"` is present but not visible. Before this, every one
  of those prims loaded as an empty group: the stage looked like it had opened
  and most of it was not there.
- **The USD code is tested in a browser.** `tests/usd_wasm.rs` runs the parser,
  the crate reader and writer, the `.usdz` container, the PNG codec and
  composition — twice, natively as `#[test]` and in headless Chrome as
  `#[wasm_bindgen_test]`, from one body, so the two cannot drift. The crate form
  is the part worth running there: it is all byte layout and shifts, and wasm is
  32-bit. `scripts/wasm-browser-test.sh` fetches a version-matched Chrome for
  Testing and chromedriver through `@puppeteer/browsers`, needing no admin
  rights — Homebrew's `chromedriver` cask is disabled for failing Gatekeeper,
  and Safari will not open a WebDriver session without `sudo safaridriver
  --enable`.
- **`UsdExport` finishes the three forms.** A document is a layer *and* the
  images it names, and only the package form knew that: `scene_to_usdc` handed
  back bytes referring to `textures/*.png` and dropped the pictures on the
  floor, so a crate written from a textured scene was dangling by construction.
  One type now builds the document once and renders it as `.usda`, `.usdc` or
  `.usdz`, with `write_to` putting the images beside the two forms that refer to
  them and inside the one that carries them. `attach_textures_from_dir` is the
  way back for those two, and declines a path that climbs out of the directory
  it was given.
- **Vertex colours survive, and every mesh carries a fallback colour.**
  `primvars:displayColor` was read and never written, so a vertex-coloured mesh
  exported from here arrived grey with the colours simply gone. It is written
  per vertex now, and where there are none the material's own colour goes out as
  a constant — which is what `usdview` flat-shaded, thumbnailers and every
  importer that reads geometry and skips materials actually use. Reading it back
  needed the matching care: a *constant* `displayColor` is the material's colour,
  not a vertex attribute, and treating it as one gave the first vertex the colour
  and every other vertex black. `displayOpacity` and `inputs:opacityThreshold`
  come across too, the latter being what makes cutout foliage cut out.
- **A `.usdz` comes back with its pictures.** The reader reported what a
  material wanted and stopped, which is right for a `.usda` naming files beside
  itself and wrong for a package, whose whole point is that the bytes are
  already in hand. A `.usdz` now arrives with its PNGs decoded onto the
  materials, carrying the wrap modes USD authored and the colour space each slot
  needs — a normal map read through an sRGB curve is the classic way to get
  lighting that is subtly wrong everywhere. Anything unresolvable (a JPEG, a
  file the archive does not hold) is still reported rather than dropped.
- **Textures export the right way up.** `flip_y` is this crate's flag for "the
  top row is stored first", which is what an image file holds and what USD
  expects. A `DataTexture` or render target is authored bottom-up, and its rows
  went into the PNG verbatim — upside down in every USD reader, with no flag in
  the format to carry the difference. Which way round USD wants was settled by
  rendering a half-red, half-blue image and seeing which half Hydra put on top.
- **A boolean written as a digit is a boolean.** `usdcat` writes `1` and `0`
  where this crate writes `true` and `false`, and the digits are
  indistinguishable from an integer until the declared type is consulted — so
  every `bool` attribute in a file OpenUSD had written read back as a number.
  `uniform bool doubleSided = 1` quietly stopped meaning anything. Found by
  exporting a scene, flattening it with `usdcat`, and reading *that* back rather
  than reading back what this crate itself wrote.
- **Materials survive the trip out and back.** A physical material exported with
  an `ior` and a clear coat came home as a standard one, because the importer
  built a `StandardMaterial` and neither input has a home on it; it now builds a
  `PhysicalMaterial` when the file authors either. `doubleSided` moves between
  the mesh, where USD keeps it, and the material, where this crate does.
- **Textures export.** A material's maps became a `UsdPreviewSurface` network
  rather than being dropped: a `UsdUVTexture` per map, a
  `UsdPrimvarReader_float2` for the UVs, a `UsdTransform2d` when the texture has
  one (USD takes degrees where this crate keeps radians), correct
  `sourceColorSpace` per slot, and the `scale`/`bias` a normal map needs to mean
  [-1,1] rather than [0,1]. Images are PNG-encoded; a `.usdz` now carries them,
  because a package whose maps resolve to nothing is a valid archive and a
  broken asset. Two bugs OpenUSD caught at once: `inputs:varname` must be a
  `string` and not a `token`, and a connection is a *field* of one property
  rather than a second property beside it — authoring both left two specs at one
  path, which the text form tolerated and the crate form refused to open.
  `inputs:ior` and the clear coat come across from a physical material, and
  `doubleSided` lands on the mesh, where USD keeps it.
- **Every material kind exports its colour.** `Lambert`, `Phong`, `Toon`,
  `Matcap`, `Sprite`, `Points`, `Line` and `Mirror` all fell through to a grey
  `(0.8, 0.8, 0.8)` default, losing colour, emissive and opacity on the way out.
  The round-trip test did not catch it because the reader read back the grey the
  writer wrote and the two agreed; exporting a gallery and asking `usdcat` what
  was in it is what showed it. Phong now arrives as USD's specular workflow with
  its exponent converted to a roughness, and a mirror as the smooth metal it is.
- **Population masks.** `usd::mask` narrows a stage to a set of prims, matching
  `usdcat --mask`: each path keeps what it names, everything beneath it, and the
  ancestors that position it — an ancestor holds on to its own transform while
  losing the children that lead nowhere. It prunes after composing rather than
  instead of, so it narrows the result without saving the work.
- **Collection membership expressions.** `pathExpression` is carried through
  the crate format, and carries the file's version with it: it is a 0.10.0 value
  type, so a file holding one is stamped 0.10.0 and a file without one stays
  0.8.0 rather than claiming to need a newer reader than it does.
- **Integration tests declare the features they need.** `cargo test` on default
  features did not build: `tests/` is auto-discovered, and the two dozen files
  reaching into `codec`, `brep`, `planet` and friends were compiled whether or
  not those modules existed. Unrelated to USD, found while checking that
  `cargo test --features usd` was clean, which it also was not.
- **Expression variables and value blocking.** A layer can declare variables
  and write them into its asset paths — `` @`"${COLOR}_asset.usda"`@ `` — which
  is how one shot layer drives what a whole stage pulls in; substitution is
  evaluated and anything beyond it declined rather than guessed. And `foo =
  None` now **blocks** a weaker opinion: it had been treated as "nothing said",
  so a blocked attribute picked up the value it existed to suppress. `None` is a
  distinct value now, not an absent one. `timecode` and `SdfValueBlock` are
  carried through the crate format — a time code written *inlined*, as every
  other double is, reads back from USD as zero.
- **The last three composition gaps.** `active = false` was ignored, so a shot
  that switched a prop off got the prop; it now takes the prim and its children
  off the stage. **A sublayer's time offset** was read and written faithfully
  and then ignored — the same bug as arcs, one level up — so a shot composed
  unshifted; offsets now compose down a chain of sublayers. And **`relocates`**
  is implemented: a layer-level map renaming prims an arc brought in. It needed
  parser work, since a relocates map is `<from>: <to>` rather than the
  `type name = value` every other dictionary uses. A relocation that would move
  a prim to a different parent is refused rather than half-done. All three
  checked against `usdcat --flatten`.
- **QC through OpenUSD's own tools** (`scripts/usd-qc.sh`). Every fixture is
  written out as a crate and handed back to `usdchecker`, `usdtree`,
  `usdcat --flatten` and `usdrecord` — the last rendering it through Hydra,
  which exercises composition, schemas, geometry and shading in one step. All 33
  pass. The script reports how many actually drew something, because a stage
  with nothing in view renders blank and two blank images agreeing is not
  evidence; 10 of the 33 have geometry in frame.
- **The remaining animation kinds.** A light's `intensity` and `color`, a
  bound material's `diffuseColor`, and `visibility` are all time-sampled in the
  files people write and none of them is a transform op, so the op-stack pass
  never saw them: a lamp that switched on halfway through a shot stayed as it
  started. They are tracks now, which needed two new `TrackTarget`s —
  `Intensity` and `Visibility` — and `Color` extended to reach lights as well as
  meshes, since a free scalar cannot say what it drives.

  **Animated vertices** become morph targets: each frame after the first holds
  its difference from the first, and the weights ramp one into the next, so the
  blend between neighbours is exactly the linear interpolation USD specifies.
  That is how a cached simulation plays back in three.js, and it costs one
  target per frame rather than one mesh per frame. The base geometry stays the
  first frame, so a stage opened at its start looks right before anything plays.
- **Every kind of USD animation.** Blend shapes were missing entirely:
  `BlendShape` prims become morph targets — sparse `offsets` expanded to a delta
  per vertex, since a smile moves the mouth and names only those points — and
  the `SkelAnimation`'s `blendShapeWeights` become `MorphWeight` tracks, each
  shape's curve a column out of samples that key all of them at once. The
  targets come back on `UsdScene::morph_targets` because `Mesh` holds the
  influences and has nowhere to put them.

  Two interpolation bugs went with it. **Quaternions were blended component by
  component**, which is not a rotation: the halfway point between `(1,0,0,0)`
  and `(0,1,0,0)` came out 0.707 long, and a rotation blended that way slows
  down mid-arc. They are interpolated spherically now, taking the shorter arc,
  since `q` and `-q` are the same rotation and a pair written with opposite
  signs would otherwise tumble the long way round. And **held interpolation**
  was not available at all — `at_time_held` steps between samples rather than
  blending, which is what a mocap or simulation cache wants, every frame being
  a measurement rather than a hint.
- **Conformance to the `.usda` grammar**, found by writing a document that
  uses every part of it and diffing against OpenUSD. Nine things were missing
  or wrong. `reorder nameChildren` and `reorder properties` were parsed as
  *properties* called `nameChildren` — they are statements about the prim, and
  a crate calls them `primOrder` and `propertyOrder`. `reorder` was not
  recognised as a list-operation qualifier anywhere. A bare string in a
  metadata block is the `comment` field, which is a different field from
  `documentation` — a layer may carry both, and this kept one. `custom` was
  parsed and dropped. A sublayer's `(offset = 5; scale = 2)` was replaced with
  identities on the way out. A dictionary entry's declared type was discarded,
  so `double[] n = [1, 2, 3]` came back as integers and `token deep = "yes"` as
  a string. An `add references` had its prim path left out of the path table,
  so the arc pointed at nothing. A relationship with no targets was read as an
  attribute, because the reader inferred what the spec type states outright.
  And a field stated twice with different qualifiers — `add references` and
  `delete references` — was written as two fields where a spec can only hold
  one, losing whichever came second.
- **Composition of list operations across layers.** `delete` and `reorder` were
  parsed and then ignored, so a shot that deletes a reference got it anyway.
  They are applied now, and applied *across* the layer stack: a list operation
  composes over the result of the weaker ones, which is the only way the shot's
  `delete` ever meets the asset's `prepend`. Checked against `usdcat --flatten`.
- **Robustness against files that are not what they claim.** A crate is full
  of offsets and counts read out of the file itself, so a corrupt one can point
  anywhere. Every prefix, every single-byte corruption and a sample of paired
  corruptions of real crates now go through the reader, and the archive reader
  gets the same treatment. That found three panics on untrusted input: a table
  of contents whose offset overflowed on being range-checked, an element count
  whose size calculation overflowed a multiply, and — outside this module —
  face indices pointing past the end of the points, which took
  `compute_vertex_normals` out of bounds. The last is refused at the loader
  now, because everything downstream trusts an index buffer to be in range.
- **A differential sweep over every fixture**, each written as a crate by this
  code, converted back by `usdcat`, and diffed against OpenUSD's own
  normalisation of the source. All 23 are byte-identical, and each scores
  exactly as its source does under `usdchecker --arkit`. Four bugs came out of
  it. **Dictionaries were dropped entirely** — a dictionary carries no USD type
  name and the writer gave up on anything whose type it could not resolve, so
  value clips did not survive `.usdc` at all. **An array of assets indexes the
  string table where a single asset indexes the tokens**, so asset arrays read
  back as whatever names happened to sit at those indices. A **relationship's
  list operation** (`prepend rel`) was parsed and then discarded, turning one
  composition arc into a different one. And `path_list_op` read only the
  explicit and appended sub-lists, so `prepend rel foo = [...]` came back as a
  relationship with **no targets at all**.
- **Apple / RealityKit validation.** Checked what this crate exports against
  `usdchecker --arkit`, Apple's compliance checker for AR Quick Look. It found
  two real bugs. A connection was being spelled twice —
  `outputs:surface.connect.connect` — which OpenUSD refuses to parse at all, a
  regression from unifying the connection syntax that this crate's own
  round-trip tests missed because both halves agreed. And **`apiSchemas` was
  written as a token vector rather than a token list operation**, so every
  schema applied to a prim was silently lost through `.usdc` in both
  directions: anchoring, physics and material binding alike. `usdz::arkit_issues`
  now reports the package constraints RealityKit adds — file types, alignment,
  layer order — which the general `.usdz` writer does not enforce because the
  format permits more.
- **`UsdGeomSubset` material assignment.** A mesh with several materials
  carries `GeomSubset` children rather than a material list, and each becomes a
  child mesh here — plus one for the faces no subset claimed. Ignoring them drew
  the whole mesh in one colour with nothing to say the rest had been lost. Only
  the `materialBind` family assigns materials; a subset for a modeller's
  selection set does not.
- **UsdShade as a graph.** Material values are resolved by following
  connections — `outputs:surface` to its shader, `diffuseColor` to its texture,
  `st` to its primvar reader — through `NodeGraph` pass-throughs. Before this
  only constants sitting directly on the surface shader were read, so every
  texture in every DCC-authored asset was silently missed. `material:binding` is
  inherited down the namespace, closest winning, so a mesh with no binding of its
  own is no longer grey. Textures are reported as requests on
  `UsdScene::textures` — file, slot, wrap modes, scale, bias, colour space —
  rather than loaded, since decoding belongs to the caller.
- **USD value clips.** Animation streamed from a sequence of files: the `clips`
  dictionary and the older flat `clipAssetPaths` spelling, with `active`
  choosing the file and `times` mapping stage time onto clip time. Clips supply
  values for attributes that already exist rather than creating them, which is
  what OpenUSD does. Checked against `usdcat --flatten` sample for sample. The
  dictionary parser gained array types along the way — `asset[] assetPaths` has
  its brackets as separate tokens, and without stepping over them the type was
  taken for the field name.
- **Catmull–Clark subdivision.** `subdivisionScheme` defaults to
  `catmullClark`, so a mesh that says nothing is a subdivision surface whose
  points are only a control cage — and drawing the cage draws something visibly
  more angular than the asset, quietly. `SceneOptions::subdivision_level` refines
  it, with creases, corners and `interpolateBoundary` honoured. Off by default:
  each level multiplies the face count by four, and the level is the renderer's
  call rather than the file's.
- **UsdSkel.** `SkelRoot`, `Skeleton`, `SkelBindingAPI` and `SkelAnimation`:
  a bound mesh becomes a `SkinnedMesh` with joints, weights and a posed
  skeleton, and the animation that drives it becomes an `AnimationClip`. Joint
  paths build the hierarchy, bind transforms are inverted once rather than every
  frame, and a rig with more influences per vertex than the shader reads keeps
  the heaviest four and renormalises. The animation is read from the layer
  *before* it is resolved to an instant — a `SkelAnimation` is nothing but time
  samples, and a resolved layer has already replaced them with one frame's pose.
- **USD point instancers.** A `PointInstancer` becomes one `InstancedMesh` per
  prototype, with `positions`, `orientations`, `scales` and `invisibleIds` all
  applied. Grouping by prototype is the point: an instanced draw carries one
  geometry, so a scatter of trees and rocks is two draws rather than one — or,
  the way a loader without this ends up doing it, a hundred thousand.
- **USD composition.** `UsdLoader::open` follows what a layer points at —
  `subLayers`, `references`, `payload`, `inherits`, `variants` and
  `specializes` — and hands back the scene they compose to, ranked by USD's
  LIVRPS strength order. `UsdLoader::parse` still reads a single layer, which
  for anything a pipeline built is not a reading of it that means much: a shot
  layer on its own is a handful of `over`s with nothing underneath. Arcs resolve
  against the layer that authored them, so an `inherits` inside a referenced
  asset finds a class in *that* asset; variant selections resolve across the
  whole opinion stack, so a shot can override a look the asset chose; cycles
  terminate; and payloads can be left unloaded, which is the reason USD
  distinguishes them from references. Where a name comes from is the caller's:
  `FileResolver`, `MemoryResolver`, or a `UsdzArchive` resolving the layers
  inside itself. Checked against `usdcat --flatten` over a three-layer pipeline,
  prim by prim and value by value, with nothing missing and nothing invented.
  The parser gained what this needs: `variantSet` blocks, `@layer@</Prim>` read
  as one arc rather than an asset beside a path, and commas as real tokens —
  `[@a@, </P>]` is two arcs where `[@a@</P>]` is one, and treating a comma as
  whitespace erases the difference.

  Arcs survive `.usdc` in both directions, which is what makes composition work
  on the binary layers real assets ship as. `ReferenceListOp`, `PayloadListOp`,
  `StringListOp`, `VariantSelectionMap`, `StringVector` and `LayerOffsetVector`
  are read and written; a crate calls three of these fields by names a document
  never uses (`inheritPaths`, `variantSelection`, `variantSetNames`), and a
  variant's body hangs off its prim at a path of its own rather than sitting
  inside it. Checked by handing a pipeline of crate files *this crate wrote* to
  `usdcat --flatten`, whose output is byte-identical to its flattening of the
  original text pipeline.

  A third: **layer offsets on arcs were read and written faithfully and then
  ignored**, so a clip referenced with `(offset = 100; scale = 2)` composed on
  the wrong time line. They apply now, and compose through nested arcs. Fixing
  it turned up a fourth — `@clip.usda@ (offset = 100)`, an arc with a shift and
  no prim named, did not parse at all. The parentheses cannot be consumed on
  sight, because an `asset` attribute is followed by its own metadata block in
  the same position, so the decision is made on the first word inside them.

  Two more came out of a fixture built for the awkward cases — empty arrays,
  integers at the limits of their types, matrices, quaternions, `faceVarying`
  primvars, two variant sets on one prim, negative time codes. **Halves were
  truncated rather than rounded**, biasing every one of them toward zero:
  `0.333` came back as `0.332764` where USD gives `0.333008`, and `half` is what
  normals and colours are usually stored as. And a **`uint64` above `i64::MAX`
  saturated** to a different number entirely; values are held as `i128` now, so
  both signed and unsigned 64-bit types survive whole.

  Two bugs fell out of that check. The path table's slots are **not dense** —
  they are a global numbering shared with the paths arcs point at, so sizing the
  table by the number of encoded entries silently dropped everything at the far
  end; it is a sparse map now. And USD requires `numPaths` to be exactly one
  past the highest index used, so the spare slot standing for the empty path has
  to sit at index zero rather than after the walk.
- **USD animation.** `timeSamples` are read from both the text and the binary
  form and come back as an `AnimationClip` on the `UsdScene`, with the scene
  posed at the first frame of the layer's range. Each prim's whole
  `xformOpOrder` stack is evaluated at every time code any of its ops is keyed
  at and then decomposed, so a rotation authored as `rotateXYZ` and one authored
  as `orient` yield the same track, and ops keyed on different frames still
  compose correctly; channels that never change are dropped rather than written
  flat. `to_scene_at` builds any other instant and `UsdLayer::at_time` resolves
  a whole layer to one moment. `animated_scene_to_usda` and
  `animated_scene_to_usdz` write clips back out — the archive plays in Quick
  Look. Time codes convert to seconds on the way in and back on the way out.
  Animated *vertices* are read and resolvable at any time but do not become a
  track, since a `KeyframeTrack` drives a transform or a morph weight rather
  than an arbitrary vertex buffer.
- **SVG path data, both directions.** A `Path` is already a sequence of
  Béziers, arcs and lines, which is the vocabulary SVG's `d` attribute speaks,
  so flattening one to a polyline to draw it threw away exactly the information
  the format exists to carry. `Path::to_svg_path_data` and
  `Shape::to_svg_path_data` write the curves as curves — a cubic is one `C`, an
  ellipse is a real `A` arc, and a `SplineCurve` converts to cubics *exactly*,
  because a Catmull-Rom spline is a cubic Hermite and Hermite-to-Bézier is a
  change of basis rather than a fit. Curve types opt in through the new
  `Curve2::svg_segments`, whose default is `None`; a `Curve2` implemented
  outside the crate still exports, just as a polyline.

  `Path::from_svg_path_data`, `Shape::from_svg_path_data` and
  `parse_svg_subpaths` read it back: the whole grammar — `M L H V C S Q T A Z`,
  the relative forms, the implicit repeat where parameters recur without the
  letter, `S`/`T`'s reflected control points — with arcs converted through the
  spec's endpoint-to-centre formulas so they stay arcs. Two details that are
  easy to get wrong and quietly wrong when you do: a `moveto` followed by more
  coordinate pairs is an implicit `lineto`, and the two arc flags are single
  characters needing no separator, so `a5 5 0 011 1` is flags `0`,`1` then the
  point `1,1` — lexing them as numbers eats the coordinates. Both are tested,
  as is the round trip through text. With `ExtrudeGeometry` this is three.js's
  `SVGLoader`: artwork in, geometry out.

- **Vector SVG export.** `SvgRenderer` draws a `Scene` to an SVG document —
  one `<path>` per triangle, depth-sorted back to front and filled with the
  colour the scene's lights give it, which is three.js's `SVGRenderer`. There
  was a stub by that name before: it emitted triangles in traversal order with
  no depth sort, no lighting, no backface cull and no near clipping, wrote
  linear colour into attributes the browser reads as sRGB, and turned geometry
  behind the camera into coordinates in the millions. It now shares the wgpu
  renderer's shading — the same Lambert/Phong accumulation, the same punctual
  attenuation, the same tone-mapping curves, the same linear→sRGB encode — so
  setting `tone_mapping` to match makes the two agree on colour. `Points`
  become `<circle>`, `LineSegments` become `<line>`, `InstancedMesh` expands to
  one sorted draw per instance, and `render_order` outranks depth the way it
  does on the GPU. It needs no GPU, no window and no adapter, which is what
  makes it work in CI, over SSH, and in a wasm build with no WebGPU. See
  `examples/svg_export.rs` and `SvgOptions`.

  Three things a painter's algorithm gets wrong were worth fixing rather than
  documenting, and the symptom of all three is the same: objects sliced off
  where a big flat surface crosses them.

  *Sorting on NDC z looks right and is not.* NDC z is hyperbolic, so a face
  with one vertex left on the near plane by the clipper reads as far nearer
  than it is — a ground plane running off behind the camera sorted to the
  *front* of the scene and painted over everything. The sort runs on linear
  view-space depth.

  *The centroid is the textbook depth key and it is the wrong one.* The case
  that actually turns up is a large polygon with small objects sitting on it,
  and a floor's centroid lands out among those objects: whichever ones it sorts
  in front of get their bases painted over. Sorting by how far back a face
  *reaches* — its farthest vertex — puts the floor behind everything standing
  on it, because a floor reaches to the horizon. Measured over ten scenes
  against ray-cast ground truth, this was exactly right where the centroid was
  wrong on three of them and no worse on any.

  *One number still cannot order a polygon that spans other geometry.*
  `depth_split` cuts faces up until each is shallow enough that it does not
  span anything. What is left for it after the depth key does its job is the
  narrower case of a large polygon that occludes something while reaching past
  it — a receding wall in front of a small object. The default is tuned to the
  bend in that curve: exact on nine of the ten scenes, and it leaves
  tessellated geometry completely alone, so a torus knot emits the same number
  of paths as with splitting switched off.

  **Curves, not polygons.** A tessellated sphere has a polygonal outline, and
  no amount of shading hides that its silhouette is a 24-gon — which is a
  strange thing to ship in a format whose entire point is that it has curves in
  it. Where the geometry carries vertex normals the surface between two
  vertices is known, so each edge is bent back onto it and written as a cubic
  `C` instead of a straight `L`. The construction is the edge half of a PN
  triangle, and the reason neighbouring faces still tile with no crack is that
  it depends only on the edge's two endpoints and their normals — the two faces
  sharing an edge compute the same curve, one of them backwards. Nothing is
  per-face.

  Only edges that need it are curved, which is what makes it affordable: a
  cube's normals are constant across each face, so both control points land on
  the chord, the deviation is zero and every edge stays an `L`. Scored against
  an analytic sphere rather than against the tessellation — a ray cast hits the
  same polyhedron the renderer draws, so it cannot measure this at all — the
  outline lands *exactly* on the silhouette from sixteen segments upwards,
  where straight edges still visibly under-fill it. Below that it closes most
  of the gap and not all, because the outline's corners are mesh vertices and
  on a mesh that coarse those already sit inside the true silhouette; bending
  the edges between them cannot move the corners.

  Moving them was tried. Cutting each triangle along the silhouette contour
  instead of culling it whole does put the boundary on the real silhouette —
  the farthest painted pixel lands on it to a tenth of a pixel — but the curves
  between those contour points run through slivers and come out lumpy, and it
  measured worse overall than leaving the outline on the vertices. It is not in
  the build.

  Geometry is clipped to the frame as well as to the near plane. A viewer
  clips to the `viewBox` regardless, so this changes nothing on screen; it
  matters because a document is not only ever looked at in a browser. A floor
  running to the horizon projected to coordinates thousands of units outside
  the frame, and an editor or thumbnailer that fits the *content* bounding box
  rather than the `viewBox` then renders the artwork as a small offset speck —
  which reads as the picture being cut off. The clip keeps the bounding box and
  the canvas the same thing, and makes the files smaller besides.

  None of that is asserted. `tests/svg_render.rs` renders a set of scenes — a
  ground plane with things standing on it, a floor running off behind the
  camera, a camera down among the geometry, a self-occluding knot, overlapping
  spheres, geometry off all four frame edges, nested transforms, a receding
  wall hiding a ball, interpenetrating solids, an orthographic view — and for a
  grid of pixels in each, casts a ray to find what is actually in front of the
  camera and compares it against what the document paints. Ground truth comes
  from the geometry rather than from the wgpu renderer, so the test needs
  neither a GPU nor an SVG rasteriser, runs in CI, and cannot drift.
  `examples/svg_gallery.rs` writes the same scenes out to look at.

  For what no sort can fix — interpenetrating geometry, textures, shadows,
  post-fx, path-traced GI — `svg_from_rgba` wraps an already-rendered RGBA
  frame in an SVG document as an embedded PNG, so the same container works
  either way.

  **In the browser it is now a renderer rather than an export function.** The
  three.js shim's `SVGRenderer` was a stub with no `domElement` and no
  `render()` — which is the whole of how a three.js renderer is turned on, so
  there was no way to enable it in a page at all. It now owns a live `<svg>`
  element, draws into it, and clears between frames (`autoClear`), and reports
  what it drew through `info.render`. `setSize` resizes in place instead of
  rebuilding the handle, which used to throw away every option set before it.
  `setClearColor`, `setPrecision`, `setQuality`, `clear` and `setPixelRatio`
  match three.js; `setShading`, `setToneMapping`, `setBackground`,
  `setCullBackfaces`, `setSeamStroke`, `setCurveTolerance`, `setDepthSplit` and
  `setSort` expose the rest. `SvgRenderer::set_clear_color` backs the first of
  those on the Rust side — an override rather than "set `scene.background`",
  because the same scene is often drawn to a dark canvas and a white page, and
  the background belongs to where it is going.

  `web/examples/svg-renderer.html` is the whole thing running — a live `<svg>`,
  the options as controls, a spin loop and a download button — and the page
  contains no canvas and no adapter request, which is the point: the renderer
  is CPU-side wasm and works where WebGPU is not available.

  `crates/threers-js/test/svg-renderer.test.mjs` covers the JavaScript half
  against a stubbed handle and a small DOM, so it runs without waiting on
  `wasm-pack`. Two of its checks earn their place. One is static: every
  `_w.someMethod()` the shim calls must exist on `WebSvgRenderer`, a mismatch
  that otherwise shows up only at runtime, in a browser, on the one line that
  uses it. The other pins the scene/camera upload — `renderToString` pushes the
  graph into wasm the way `WebGLRenderer.render` does, and writing the example
  is what turned up that it did not: the spin loop moved a mesh every frame and
  the picture never changed, because the renderer was drawing whatever pose had
  last been uploaded.
- **`examples/neon_logo`** — an iridescent logo mark: black metal under a
  drifting four-colour simplex field, with a ten-stop ramp indexed by the angle
  to a single spot light composited over it. Most of that ramp is opaque black,
  so the shape reads as a hole and only four thin stops near the lit end carve
  the bright edge. Because the ramp keys off the light, moving the light sweeps
  the edge around the mark — which is what the animation does.

  Both solids are **swept from a 2D outline**, not loaded: the surface is
  `(1 + sd/D)² + (z/Z)² = 1` over a distance field built from the silhouette, an
  elliptical inflation meshed as two height-field sheets. Normals come from the
  field gradient rather than averaged from triangles, which is what keeps the
  rim clean — the ramp's top stops are only 0.012-0.015 apart in `t`, about two
  degrees of normal rotation each, so the edge is unusually sensitive to normal
  quality. For the same reason the stored outlines are sampled at uniform arc
  length: unevenly spaced knots make Catmull-Rom wobble in curvature, and the
  ramp turns that wobble into a visible ripple. `NEON_MESH=<dir>` loads
  `body.obj` / `leaf.obj` instead.

  **A 15-second timeline**, `NEON_ANIM_FPS=<n>`, with an `.mp4` through this
  crate's own H.264 encoder under `native-codec`. Three chained tweens: the spot
  tours four positions and comes home, while both meshes morph their noise
  fields underneath. Geometry is built once and only the two materials and the
  light move, so the renderer's per-mesh GPU buffers survive the whole clip, and
  frames stream back off disk for encoding rather than being held — 450 at this
  size would be 3.4 GB.

  The post chain — bright-pass bloom, then saturation — runs entirely in linear
  light. Grading the encoded bytes instead pins green to full on saturated
  colours and turns the broad cyan sweep white.

- **The small things d3 and p5 have and this did not.** Every one of these was
  being hand-rolled at the call site — twenty-six separate copies of the random
  generator alone — which is the signal that it belonged in the crate. All of
  them are in the prelude.

  `sketch(options, setup)` is a window, a scene, a camera, a renderer and a
  loop in one call, where the shortest path from nothing to a moving picture
  used to be eighty lines of instance, adapter, device-with-two-limits-raised,
  surface, configuration and a `match` on window events. `setup` returns the
  draw closure, so whatever setup built is simply captured — no shared state
  threaded through a struct and no `Option` that is `None` until the first
  frame. `Frame` hands out the scene, the camera, the overlay and the clock.

  `Frame::scale_factor` is p5's `pixelDensity()`, and it exists because the two
  halves of the API use different units on purpose: `SketchOptions::size` is
  **logical** — asking for a 960x720 window means 960x720 points of desk space —
  while `Frame::width`, the overlay and `Frame::pointer` are **physical**, so
  they agree with each other and with `Camera::world_to_screen` without
  conversion. What that costs is that a 15-pixel label is half as tall on a 2x
  display as the window size implies, so anything meant to be read gets
  multiplied by this.

  `Rng` is a seeded PCG32 with the methods generative work actually reaches
  for: `range`, `below`, `chance`, `gaussian`, `on_sphere`, `in_sphere`, `pick`,
  `shuffle` and `fork`. `ColorRamp` carries viridis, magma, inferno, plasma,
  turbo and a diverging cool-warm, interpolated through OkLab by default
  because a gradient blended in linear RGB goes grey through the middle.
  `Color` grew `from_hsl`, `to_hsl`, `lerp`, `lerp_oklab`, `to_oklab`,
  `from_oklab`, `luminance` and `scaled`. `map_range`, `map_range_clamped`,
  `inverse_lerp`, `smoothstep`, `smootherstep`, `ScaleLinear`, `ScaleLog`,
  `ScalePow` and `ticks` are d3-scale's useful half.

  `easing` **moved from `threers-animation` into `threers`**, because a curve
  from 0 to 1 is not animation and keeping it there meant `cargo add threers`
  got you none of it. `threers_animation::easing` re-exports it, so both paths
  work. `SimplexNoise` gained `noise3`, `fbm2`, `fbm3`, `turbulence2`,
  `ridged2` and `curl2` — curl noise being the divergence-free one, so a flow
  field swirls instead of collecting every particle into a few sinks.

- **An immediate-mode 2D overlay** (`overlay::Overlay`). p5's `rect`,
  `circle`, `ellipse`, `line`, `polyline`, `polygon`, `rounded_rect` and `text`
  on a CPU canvas composited over the rendered frame, with `fill`/`stroke`/
  `stroke_weight` sticky the way p5's are. Every project reaches the point where
  the render is right and now a frame counter has to go on top of it, and doing
  that in 3D — a billboarded textured quad per glyph, depth disabled,
  re-uploaded when the string changes — is a disproportionate amount of
  machinery for a line of text. `sketch` wires it up: cleared before `draw`,
  composited after, and skipped entirely on a frame that drew nothing 2D.

  Every primitive is one signed distance function and coverage is `0.5 - d`
  clamped, so all of them anti-alias through the same rasteriser with no
  supersampling cost. Compositing runs in linear light — `Color` is linear and
  the buffer holds sRGB bytes, so overlapping shapes decode, mix and re-encode.
  Blending coverage in the encoded bytes is the classic source of dark fringes
  on anti-aliased edges, and it is ~30 bytes off the right answer at half
  opacity; `tests/overlay_gpu.rs` pins that against both of the ways it usually
  goes wrong by reading pixels back off a real target.

  Text comes in three alignments — `text`, `text_right`, `text_centered`.
  Right-aligned is the one a read-out wants, so the digits grow leftwards from
  a fixed edge instead of shoving the line about as the value changes; the
  alternative at the call site is a `measure_text` and a subtraction, written
  twice in every HUD. `CaptionPainter` gained the public `draw_text_at_center`
  they share.

  **Only the rows that changed are uploaded.** `Renderer::draw_overlay` takes
  the `Overlay` rather than its bytes and uploads the band that differs from
  what the texture already holds: measured over a frame that clears, draws a
  two-line panel and composites it, **1.1 ms against 223 µs — 4.9x** at
  1920x1080, where the panel is 95 rows out of 1080. Worth it because the
  upload was nearly all of the cost: 728 µs of it, against ~124 µs for the
  full-screen quad and the submit together.

  The band is a delta, which makes the bookkeeping the interesting part, and all
  of it is about the texture holding something the buffer does not. Erasing a
  panel changes rows without drawing on any, so a clear marks the rows it
  erased. A frame that draws nothing leaves that outstanding rather than
  dropping it, so skipping a composite cannot lose it. A renderer seeing an
  overlay for the first time has no previous frame to delta against and uploads
  all of it. And text is drawn by the caption painter straight into the buffer,
  so the overlay *asks* which rows it used (`CaptionPainter::last_rows`) rather
  than predicting them — the painter re-wraps against the space left of the
  anchor and slides the block up off the bottom edge, so a line of type does not
  land on `y..y + measured height`. `tests/overlay_gpu.rs` runs a sequence of
  frames through both paths and compares the composited pixels, because no
  single frame can catch any of that.

  Two things about the rasteriser are worth recording. It walks in tiles and
  settles whole tiles from a single distance evaluation, which a signed distance
  field licenses because it is 1-Lipschitz: eight full-viewport Voronoi cells
  went from 96 ms a frame to 5 ms, and the read-out case that most frames
  actually draw is 183 µs either way. And the sRGB encode on the overlap path
  is a table rather than a `powf`, because the answer is one of 256 bytes, so
  the real question is which of 255 switchover points a value has passed.
  Deriving those thresholds analytically as `decode((b - 0.5) / 255)` is the
  obvious way and lands on the wrong side of **14 of the 255** — the midpoint
  between two bytes is exactly where `f32` rounding decides — so they are found
  by bisecting the curve itself over the bit patterns of `0.0..=1.0`, which
  makes the table a memoisation of the curve and not an approximation of it.
  2.29 ns a call against 6.00 ns, and verified byte-for-byte at every
  switchover and the representable value either side of it.

- **`Camera::world_to_screen` and `world_to_ndc`.** The inverse,
  `Vector3::unproject`, was already there; the forward direction was not, which
  made the overlay above unusable for the thing it is most wanted for — putting
  a label where an object is. Returns `None` at or behind the eye, because both
  projections get that wrong in their own way and neither looks like an error:
  a perspective divide by a negative `w` puts the point back in front,
  mirrored, and an orthographic camera has no divide at all, so a point behind
  the viewer landed squarely on top of the thing the camera was looking at.
  Screen coordinates are pixels from the top-left, which is the frame the
  overlay draws in and the one `Frame::pointer` reports, so a label on an
  object and a hit test against the cursor need no conversion between them.

  `Camera::projector` is the same thing for a list rather than a point:
  `world_to_screen` rebuilds a `look_at` and a projection matrix on every call,
  which is right for one label and wrong for a thousand. `ScreenProjector`
  folds them into one matrix once, and the per-point methods are now defined in
  terms of it so there is a single implementation of the rule.

- **Delaunay triangulation, 2D convex hull and Voronoi cells**
  (`math::delaunay`, `math::convex_hull`, `math::voronoi_cells`). A scattered
  set of points is the most common input in generative work, and all three of
  these existed in the tree already with none of them reachable: the exact
  integer-predicate Delaunay sat behind the `rlx-geo` feature and an external
  crate, the constrained triangulator in the CSG kernel is specialised to one
  convex triangle, and the 2D hull was private to the OpenSCAD front end.

  **There is deliberately no super-triangle**, which is the usual way to write
  this. Wrapping the input in one huge triangle and discarding whatever still
  touches it quietly loses thin triangles along the hull: those have enormous
  circumcircles — a chord of 3 with a sagitta of 0.004 gives a radius of 271 —
  so a super-triangle vertex at any finite distance eventually lands inside
  one, the sliver fails the circumcircle test, and the result is a crack
  between the triangulation and its own hull. On a jittered 8×8 grid that was
  two triangles out of 114 and 0.1% of the area. Pushing the super-triangle
  further out only trades it for a loss of conditioning, since the fourth
  powers of the coordinates swamp the term being tested. Insertion starts from
  a real triangle instead and extends the mesh across the hull edges a new
  point can see, which has neither problem and no magic constant.

  What is worth repeating is how that was found. The mesh was *consistent* the
  whole time — closed boundary, no edge used three times, no point dropped, and
  the full triangulation including the super-triangle conserved area exactly at
  every insertion. Comparing the triangles' area against the hull polygon's was
  the only check that showed the crack, and it is now the test.

  `voronoi_cells` clips the bounding rectangle back by perpendicular bisectors,
  which closes the cells on the hull for free — they are unbounded in theory and
  bounded by the rectangle in practice, which is what a caller drawing one
  wants. *Which* bisectors comes from the dual: a cell is bounded only by the
  ones it shares with its Delaunay neighbours, of which a site has about six
  however many points there are. Clipping against all of them is the obvious way
  to write it and took **379 ms for two thousand sites against 17 ms**, most of
  which is now the triangulation it no longer skips. All-pairs remains as the
  fallback for input the triangulation cannot describe — fewer than three sites,
  collinear sites, duplicates — selected by the one symptom they share, a site
  with no neighbours, and a test holds the two paths to the same cells rather
  than to the same areas.

### Changed

- **The supersample resolve reconstructs with a real filter.** It averaged the
  `factor × factor` block, which is the pixel's area coverage and sounds like
  the right answer. It is not: a box one output pixel wide is a poor low-pass —
  its response is `sinc`, first zero at twice Nyquist — so it passes much of the
  detail it was meant to remove and that folds back as aliasing. The tell is a
  shallow edge, which climbs one pixel row every `1/slope` columns and so
  staircases at exactly that period.

  Adding samples inside the box does not fix it, because the support is what is
  wrong, not how finely it is sampled — which is why supersampling harder stops
  paying off so early. Measured on a rotating phone, tracking the sub-pixel
  position of one near-horizontal rail edge across eight frames and taking how
  far it strays from the rigid motion the whole edge shares:

  ```text
  1080p, 3× supersample, 4× MSAA        wobble   staircase period
  box (what this replaces)              0.239 px   15.3 px  = 1/slope
  tent                                  0.103 px  none
  Mitchell (the new default)            0.105 px  none
  Lanczos-2                             0.101 px  none
  8K rendered and resolved 4× (16× the cost)
                                        0.067 px  none
  ```

  Genuine motion on that edge is 0.78 px a frame, so the box was adding
  sampling error worth a third of the movement — an edge that crawls rather
  than glides, and what a viewer reads as a turn that is being interpolated
  rather than rendered. The wider filter leaves 13% and costs nothing
  measurable: the resolve is a bandwidth-bound pass either way, and 2.3% of
  gradient detail is the whole price.

  `HeadlessBuilder::downsample_filter` and `Downsampler::with_filter` pick the
  filter; `DownsampleFilter::Box` restores the old pixels exactly and remains
  the only one that matches `average_blocks_srgb`, which is unchanged. The
  filters are separable and run in two passes through an `Rgba16Float`
  intermediate — `4 × factor` taps an axis rather than `16 × factor²` in one
  pass, and signed, because Mitchell and Lanczos both land outside `[0, 1]` and
  clamping in between would quietly turn them back into worse boxes. The Metal
  backend's `MetalDownsampler` got the same default, so the two paths still
  agree.

  Lanczos-3 was tried and is *worse than the box* on this measurement: it rings
  hard enough to displace the edge it is meant to stabilise. Lanczos-2 measures
  best and undershoots to −0.041 in linear light, which against a 100:1 edge is
  a dark halo you can see, so Mitchell is the default and Lanczos-2 is opt-in.

### Fixed

- **Spot shadows on the Metal backend**, on their own perspective map — a
  frustum of twice the outer cone angle, one compare tap, matching the wgpu
  path. Metal and wgpu agree on where the shadow falls to a correlation of
  **0.9999**. All three casters share one depth-only vertex function and write
  their own matrix into `shadow_vp` before each pass, rather than having three
  functions that differ only in which matrix they read.

- **Point shadows on the Metal backend**, and two bugs underneath them that
  had to be found in order.

  The cube stored the wrong *quantity*. The lookup is by direction and compares
  `length(world - light) / radius`, but the pass ran with no fragment stage at
  all and stored **projected** depth — per-face, discontinuous at the seams,
  and never matching that comparison. Both renderers did this, so both had
  point shadows that silently never appeared and no test that noticed.
  `fs_shadow_point` writes radial distance now, which makes this the one shadow
  pass with a fragment stage.

  And the cube faces were **mirrored**. Cube maps are left-handed in every
  graphics API and `Matrix4::look_at` is right-handed, so each face was laid
  down flipped against what the sampler would later expect. Every face looks
  perfectly reasonable on its own, which is what makes it hard to see: the
  shadow simply lands on the wrong side. Negating clip-space x took the
  darkening from 0.0021 over three tiles to **0.0785 over eight** — against
  0.0786 over eight for a spot light in the same place, which is the answer,
  since a point and a spot light at one position throw one shadow.

  The wgpu path had both bugs too and is fixed with it — but it wants the *y*
  axis negated where Metal wants *x*. That is measured rather than guessed, by
  `tests/shadow_fixtures.rs`, in pixels from where the geometry puts the
  shadow:

  ```text
             none      x       y      xy
    wgpu     79.2   133.1     2.1   107.0
    metal   107.0     2.1   133.1    79.2
  ```

  The two tables are each other's shifted by one flip: the backends'
  framebuffer-y conventions are opposite, so the correction is too.

- **Muxed files stated their colour in the bitstream and not in the
  container.** The encoder sets `ColorPrimaries`, `TransferFunction` and
  `YCbCrMatrix` on the VideoToolbox session, so BT.709 reaches the HEVC VUI —
  and the sample entry carried no `colr` box at all. Players that read the VUI
  were right; players that do not fell back to guessing from the frame size,
  and a tall 4K frame is not reliably guessed as BT.709. The same file then
  looks different in different players, most visibly on high-contrast edges,
  which reads as a *rendering* fault rather than a container one.

  Both places now, for both codecs: the VUI survives a remux that rebuilds the
  container, and `colr` is read by players that never look at the VUI.
  `the_sample_entry_states_its_colour` pins it.

- **Bloom turned the picture grey, and upside down.** Two separate bugs in one
  pass, and the reason the phone render came out in black and white and
  inverted under `--backend metal` while every material, texture, tone-map,
  orientation and antialiasing test passed: the app enables bloom by default
  and none of those tests did.

  * **`CompositeParams` was not the same struct in both languages.** Metal
    aligns `float2` to eight bytes and Rust aligns `[f32; 2]` to four, so with
    one `float` in front of it the two disagreed about where everything after
    it lived — the shader read `chroma_sat` out of the bytes Rust had written
    `chroma_lift` into. A saturation scale of zero turns the chroma step into
    `rgb = maxc`, the max channel in all three components: a perfectly grey
    frame, from a shader whose arithmetic is a no-op when its parameters are
    right. The struct is five scalars now, four-byte aligned on both sides,
    with a `size_of` assertion so the next member to be added cannot
    reintroduce it quietly. `bytemuck` says a type is plain old data; it says
    nothing about whether the other language agrees.

  * **`vs_fullscreen` handed out a `uv` with the wrong origin.** Clip space has
    +y up and the framebuffer has y down, so the vertex whose clip y is +1
    lands on the top row — while a texture coordinate of v = 1 is the *bottom*
    of the source. Passes sampling by `in.uv` were therefore inverted and
    passes sampling by `in.position.xy` were not, so a chain mixing the two
    flipped only some of its stages. Flipped at the vertex stage rather than at
    the three call sites, so the next pass to sample by `uv` is right without
    having to know any of this.

  `metal_bloom_keeps_the_colour` and
  `metal_post_processing_keeps_the_picture_upright` pin both. The second walks
  the stages one at a time — plain, supersample, tone map, bloom — so a failure
  names the stage rather than the chain.

- **A shadow fixture that takes its reference from geometry, not from the other
  renderer.** `tests/shadow_fixtures.rs`. Every shadow test before it asked
  whether two backends agreed, which caught real bugs and has one failure mode
  this project walked straight into: when both are wrong in the same place,
  agreement is perfect and means nothing. Point shadows agreed at **0.9996**
  while neither of them cast anything.

  So the reference is arithmetic. A light at `L`, an occluder at `O` and a floor
  at a known depth put the shadow somewhere exactly computable — the ray `L → O`
  extended to the floor — and the camera's own projection says where that lands
  on screen. The answer is an error in **pixels**. It measures placement,
  strength and extent separately because they fail differently: a shadow in the
  wrong place, a shadow barely there, and a bias so large the whole frame is in
  shadow are three different bugs and only the first is obvious.

  All six rows — three light types, two backends — now land 1.5 to 2.2 px from
  the prediction, and the two backends land within 2 px of each other.

- **Directional shadow maps on the Metal backend.** A depth-only pass from the
  light fills a `Depth32Float` map, and the fragment stage samples it through a
  comparison sampler with a 3x3 percentage-closer filter — nine hardware depth
  *tests*, not nine fetches plus arithmetic.

  Both halves mirror the wgpu path deliberately. The bias is a **normal** bias
  rather than a large constant depth one: a constant has to cover the worst
  case, a face nearly edge-on to the light where a single texel spans a long
  slope, and sized for that it is far too much everywhere else, so shadows
  visibly detach from their casters. And the light's frustum is a fixed cube on
  the origin rather than fitted to the visible meshes, because fitting makes the
  map's contents depend on the camera and the shadow then swims as the view
  moves.

  `cast_shadow` and `receive_shadow` are read per object. The first version of
  this shadowed every opaque mesh, which looks right in a test scene and wrong
  in a real one where the floor casts onto itself — and it disagreed with wgpu,
  which reads both. Reading them took the agreement on where the shadow falls
  from a correlation of 0.0 to **1.0000**.

  Point and spot casters are still `Renderer`'s: they need a cube map and a
  perspective frustum, not another orthographic one.

- **Skinning on the Metal backend.** A `SkinnedMesh` used to be drawn in its
  bind pose there — geometry right, animation silently absent, which reads as a
  rigging problem in the asset rather than a missing feature. `vs_skinned`
  blends each vertex through its skeleton's bone matrices, four influences a
  vertex, normalised so an exporter that dropped a fifth influence does not
  shrink the mesh toward the origin. The joint and weight attributes go in a
  buffer of their own rather than widening `Vertex`, because skinned meshes are
  a minority and 32 bytes a vertex on every other mesh is not a trade worth
  making. Rotating one bone moves the mesh by 0.13 rms and lands **0.0000 rms**
  from wgpu in the same pose.

- **Vignette and grain on the Metal backend**, which completes the
  post-processing chain: it had bloom, SSAO, a tone map and the resamplers, and
  no film grade at all, so a caller asking for one got it on wgpu and nothing on
  Metal. Ported line for line from `rgb_pack.rs`, including the mid-tone weight
  on the grain — film has almost none in the toe or the shoulder, and an even
  sprinkle over black reads as sensor noise — and the per-frame seed, because
  static grain is dirt on the lens. `set_film_grade` and `set_grain_seed` match
  the wgpu names.

- **Corrected: morph targets are not implemented by *either* renderer.** The
  Metal backend's docs listed them among the things
  `Renderer` handles instead. It does not:
  `MorphAttributes` is produced by the USD loader and consumed by nothing, and
  `BufferGeometry` has no field to hang targets off. Implementing them is a
  change to the geometry model and both backends.

- **`MetalHeadlessRenderer::set_tonemap` did nothing on an 8-bit target.** The
  tone-map pass ran only when the render target was a float format — the case
  it is *required* for, since an 8-bit readback of float pixels is garbage
  without it — so on an ordinary target the exposure a caller passed was
  silently discarded. Three different exposures produced byte-identical frames.

  It runs whenever a caller asks for it now. On an 8-bit target the pass is
  still well defined: an sRGB-formatted texture decodes on sample and
  re-encodes on write, so the curve is applied in linear light either way. It
  is lossier than grading in float, the input having already been quantised,
  but that is the caller's trade to make rather than this function's to make on
  their behalf and without saying so. `metal_exposure_reaches_the_pixels` pins
  it across every supersample and MSAA combination.

- **Image-based lighting and the material map slots, on the Metal backend.**
  It used to hand environment/IBL and the normal, roughness, metalness, AO and
  emissive map slots to [`Renderer`]; it implements them now, and
  `tests/backend_parity.rs` holds each one to the wgpu renderer:

  ```text
  emissive map                      0.0024 rms   (colour and orientation both)
  environment, roughness 0.05       0.0009 rms
  environment, roughness 0.60       0.0049 rms
  normal / roughness maps           0.11   rms   (see below)
  ```

  The environment is a plain cube with its own mip chain, walked by
  `roughness²` — and it is built by the *same* `mip_chain` the wgpu path uses,
  a box downsample in linear light, because two renderers that filter an
  environment differently disagree about what a rough surface reflects and
  nobody can see that is a bug. The split-sum approximation and its page of
  constants are mirrored line for line.

  Normal mapping reconstructs its tangent frame from screen-space derivatives
  (Mikkelsen 2010) rather than a tangent attribute, because the vertex stage
  here carries none and adding one would mean touching every geometry path for
  a slot most of them never fill.

  The extra maps must be uploaded as *linear* — they are quantities and not
  colours, glTF §3.9.2 — and since a Metal texture's pixel format is fixed at
  creation, an image used as both a colour map and a data map is uploaded twice
  and cached under a key that carries the role.

- **Metal's direct lighting brought to the wgpu path's.** Adding the map slots
  made an older difference measurable: a roughness map is the thing that makes
  a BRDF disagreement visible, and `roughness_map` opened at 0.0945 rms. Two
  causes, both in [`pbr_direct`]:

  * The diffuse lobe was missing its `1/pi`. A Lambertian surface scatters its
    albedo over a hemisphere rather than into one direction, and leaving the
    normalisation off made the diffuse pi times too strong *relative to the
    specular* — invisible on a flat-coloured surface, obvious the moment a map
    gives the specular something to vary against.
  * The geometry term was the separable Schlick-GGX approximation where the
    wgpu path uses height-correlated Smith visibility, which also folds in the
    `1/(4·NdotV·NdotL)` the specular would otherwise divide by.

  And ambient occlusion was attenuating the *direct* lights along with the
  indirect ones. Occlusion answers "how much of the surrounding room can this
  point see", which is a question about light arriving from everywhere; whether
  a particular light reaches a particular point is what a shadow map is for.
  Multiplying the whole shade by it darkens the key light in a crevice the key
  light is shining straight into.

  ```text
                 before   after
  roughness_map  0.0945   0.0160
  normal_map     0.0280   0.0161
  ao_map         0.0331   0.0046
  ```

  What remains is the roughness floor and geometric specular antialiasing the
  wgpu path applies and this one does not.

- **A backend parity suite, and the Metal backend cleared of a bug it never
  had.** `tests/backend_parity.rs` renders one scene through every backend and
  compares them — a coarse luma grid and where the bright parts are, not a
  pixel diff, because two rasterisers never agree to the bit and a test that
  demanded it would be switched off within a week. What they must agree on is
  what is where.

  It was written to find a flip. The phone render this crate exists for draws a
  blank white screen with its Dynamic Island along the *bottom* edge under
  `--backend metal`, which had been recorded as the Metal backend being broken.
  It is not. Metal matches wgpu to **0.0002 rms or better** on geometry, parent
  transforms, texture orientation, unlit textured materials, and all four
  supersample/MSAA combinations — including, specifically, drawing an
  asymmetric texture the same way up.

  The screen is an emissive-mapped material, because a phone screen emits its
  picture rather than reflecting it, and the emissive map slot is one of the
  ones `MetalRenderer` documents as belonging to [`Renderer`] instead. The
  backend is exactly right about everything it claims to do and silent about
  the rest, which is the harder failure to spot — and the reason the suite
  stays inside the set both renderers implement rather than testing a promise
  nobody made.

- **The default colour format double-encoded sRGB, and every frame was washed
  out.** `HeadlessConfig::color_format` defaults to `Rgba8UnormSrgb`, which
  makes the *hardware* apply the transfer curve on write — but the mesh shader
  applied it too, because the flag telling it not to was `format ==
  Rgba16Float` and an sRGB format is not that. A pixel with exactly half
  coverage of white stored 0.7379 of the light where it should have stored
  0.5000. The same confusion ran the other way in `downsample`, which decoded
  bytes the hardware had already decoded for it.

  There is one predicate now — `shader_applies_transfer` — and it is true only
  for a plain `Unorm` format, the one case where nothing else will do it. Float
  stores linear; sRGB converts in hardware both ways. Write and read always
  agree, so it is one question and not two.

  Fixing it also fixes the multisample resolve, which is what went looking for
  it. The resolve is hardware and averages whatever the bytes hold, so on a
  plain `Unorm` target carrying hand-encoded sRGB it is a mean over a curve:
  two covered samples and two not gave 0.216 of the light instead of 0.500, an
  edge at 43% of its proper brightness. On the sRGB default the hardware
  decodes, averages and re-encodes, and it measures 0.5029.

  No existing test caught any of this, because every one of them measured a
  *difference* — an edge's position, a residual, a ratio — and all of those are
  blind to the value in the pixel being wrong by a factor of two.
  `a_half_covered_pixel_carries_half_the_light` asks the absolute question, over
  every format and every antialiasing path.

- **Three silent antialiasing bugs, found by one new test.**
  `tests/interpolation.rs` slides a white quad across a pixel a sixteenth at a
  time and measures where the edge lands, for every renderer and every backend.
  Against the straight line that should trace, a hard-edged rasteriser scores
  1/√12 ≈ 0.289 px — the rms of a uniform quantiser — and anything that
  antialiases scores less. It also measures the vertical and horizontal edges of
  the same quad separately, which is the check that earned its keep: rasterising
  is symmetric, so any gap between the two axes is something downstream of it.

  * **Jitter was applied in the wrong matrix element for orthographic
    cameras.** A perspective matrix divides by `-z`, so an offset in the *z*
    column arrives in NDC as a constant: `x_clip += jx·z` over `w = -z` is `-jx`
    at every depth. Orthographic does not divide — `w` is 1 — so the same offset
    landed as `jx·z`, a shear growing with distance from the camera rather than
    a sub-pixel step. With the camera ten units back it spread a hard edge over
    **nine pixels**. Orthographic now takes the translation column, and the
    element asked about is `elements[11]` rather than the camera type guessed
    at. Edge width 7.62 px → 1.39.

  * **The two axes converged at different rates.** Halton is the right sequence
    when the sample count is not known ahead of time, because it is spread at
    every prefix length — but its one-dimensional marginals are only uniform in
    the limit, and at any particular `n` the two bases disagree about how
    uniform they are. At sixteen samples base 2 lands on the exact sixteenths
    and base 3 does not, so a straight edge was placed **fifteen times** less
    precisely along one axis than the other: 0.0253 px against 0.0017. The means
    are off centre too, and by different amounts per axis, which translates the
    whole frame — at eight samples, 0.055 px in x against 0.000 in y.

    `stratified_jitter(n)` is new and public: an N-rooks arrangement where both
    axes get the same exact ladder of `n` positions, paired by *ranking* a
    Halton sequence rather than using its values, which keeps the
    two-dimensional spread and the progressive property both. Exactly
    zero-mean and exactly stratified on both axes, for any `n`.
    `render_accumulated` uses it, because it knows the sample count; the
    streaming temporal pass still uses Halton, because it does not.
    Both axes now measure 0.0017 px.

  * **MSAA did nothing on a linear framebuffer.** `use_msaa` was gated on
    `!linear_framebuffer`, so asking a `Rgba16Float` target for 4× or 8×
    multisampling got single-sample rendering — no error, no warning, and a
    quality setting that did nothing. Measured at 1, 4 and 8 samples it returned
    *precisely* the no-antialiasing number, 0.2524 px, three times over. The
    restriction now lifts for the explicit multisampled entry point, which is
    the path that asked for it; float targets measure 0.0698 px, the same as an
    sRGB one.

  One thing the test found and did not fix, recorded because it is worth
  knowing: MSAA on `Rgba8Unorm` measures 0.1230 px where the same scene on
  `Rgba8UnormSrgb` measures 0.0698. The resolve is hardware, and on a target
  that holds *manually* encoded sRGB it averages the encoded bytes — the same
  mean-over-a-curve as the entry below, in the one place the renderer cannot
  reach. Prefer an sRGB-suffixed format when multisampling.

- **The temporal mean was taken over sRGB bytes, not over light.** Every render
  target but `Rgba16Float` holds sRGB-*encoded* values, and the converging
  accumulation summed them as they stood. Averaging encoded bytes is averaging
  over a curve: sRGB is convex, so by Jensen's inequality the mean of the
  encoded samples is not the encoded mean, and the gap is proportional to how
  much the samples disagree — which is largest exactly where a renderer is
  judged, on edges and fine detail. On a black-to-white edge the accumulated
  frame read 21/255 where the light says 0. The supersample resolve had always
  decoded before averaging its four subpixels; the temporal pass beside it had
  not, so the two halves of the same anti-aliasing disagreed about what space
  they were in.

  Both now average in linear light. The converge path decodes the current frame
  and the history, mixes, and re-encodes — the history stays in the
  framebuffer's own space, which is what lets the blit stay a plain copy, and
  the round trip is exact because that history is float.

  This was measured as a regression before it was understood as a fix. The test
  suite's straightness metric locates an edge from a gradient centroid, and a
  centroid only reads as coverage when the values are linear in coverage; run
  over encoded bytes it called the same frame 0.078 px where the linear values
  say 0.022. The fix moved that metric from 0.0216 to 0.0803 while leaving the
  edge exactly as straight as it had been. The metric decodes now.

- **A plain cube environment had no mip chain, so roughness never blurred a
  reflection.** `GpuCubeTexture::upload` wrote one level for a cube that had no
  PMREM atlas, which made `textureNumLevels` 1 and resolved every roughness to
  mip 0: a rough metal reflected the environment exactly as sharply as a mirror,
  the only difference between them coming from the BRDF. Plain cubes now get a
  full chain, box-downsampled in linear light — sRGB is a transfer function and
  not a quantity you can take the mean of, and an environment is mostly edges.

  The mapping on top of it was wrong too. `(roughness + 2) / 10 * 7` puts the
  whole of `0..1` on mips 1.40 to 2.10 — seven tenths of one level — and assumes
  a chain eight long. It is `roughness² × (levels − 1)` now, with the chain's
  real length asked for. Measured as the contrast a mirror sphere reflects from
  a hard checker cube:

  ```text
  roughness    before    after
  0.02          92.6      89.2
  0.35          77.5      53.7
  0.90          67.4       0.0
  ```

  Nothing reaching the environment through `PmremGenerator` is affected — that
  path has its own atlas and the three.js roughness-to-LOD mapping, and is what
  most callers use.

- **Note on reflection aliasing (no code change).** A strongly foreshortened
  curved specular surface — a phone's side rail, a chrome pipe seen edge-on —
  breaks into bands along its length, and the cause is plain undersampling of
  the specular term rather than anything the crate filters wrongly. Measured on
  one such rail as the rms of row-to-row brightness: 0.99 at `supersample(1)`,
  0.62 at 3, 0.30 at 6, against 0.22 for the same frame rendered at 8K and
  scaled down.

  Four fixes were written and measured before that conclusion, and all four are
  no-ops on this case: widening the lobe by the normal's screen derivative
  (`SPEC_AA_SIGMA` swept 0.25 → 4.0 — the normal turns too gently per pixel for
  there to be variance to find), prefiltering the environment by
  `dpdx(reflect_vec)`, four-tap stochastic sampling across that footprint, and
  raising the environment cube from 256 to 1024. The last three were reverted
  rather than left in; three extra cube fetches a pixel that change nothing are
  not worth carrying. The `dpdx(reflect_vec)` finding is recorded as a comment
  where it would otherwise be reached for again.

- **A transparent surface keeps its reflection.** Alpha blending composites
  `src × α + dst × (1 − α)`, scaling the whole shaded colour by alpha —
  reflection included. But alpha says how much of what is *behind* the surface
  shows through, and a reflection happens in front of it and is not transmitted
  at all. Blending it away is what made a polished surface at `alphaMode:
  BLEND` read as grey film rather than as glass, and authors without
  `KHR_materials_transmission` fake glass exactly that way, so it was the common
  case and not the corner one.

  A transparent PBR material now raises its coverage by its own specular
  reflectance: `α + (1 − α) × reflectance`, taken from the split-sum term the
  IBL path already computes, so it costs nothing. Measured on a metal sphere in
  a uniform environment at `opacity = 0.25`, it keeps **96%** of its reflection
  instead of 25%. A dielectric at the same opacity keeps 27% against 25% — a
  few percent of grazing rim, which is correct, and transparency still means
  transparency. Opaque materials go through the same gate unchanged; the IBL
  energy tests still hold.

  The transmitted half is over-weighted by the same factor, which is exactly
  right for a metal (it transmits nothing) and negligible for glass, so the
  approximation errs the right way at both ends. Premultiplied output would make
  both exact, at the cost of splitting the accumulator every light contributes
  to. `KHR_materials_transmission` still computes its own Fresnel opacity and is
  untouched.
- **Text was rendered at whole-pixel positions, so letter spacing was uneven.**
  Advances are fractional — 13.7 pixels is an ordinary one — and the pen was
  rounded to a whole pixel for every glyph, throwing the fraction away each time.
  The gaps then alternate: measured on a row of stems they came out
  `[5, 7, 7, 7, 7]`, a two-pixel swing between the first pair and the rest. That
  is the wobble that distinguishes rendered text from the same text out of a real
  renderer, and no amount of care in the compositing fixes it, because the glyph
  has to be *rasterized* for the position it actually occupies. Glyphs are now
  rasterized at quarter-pixel phases and cached per phase — four entries a glyph,
  for a placement error of an eighth of a pixel instead of a half.
- **Built-in glyph coverage was sampled rather than computed.** Sixteen sample
  centres per pixel quantize coverage to seventeen levels, which steps visibly
  along a diagonal. The source is a bitmap, so every lit cell is an axis-aligned
  rectangle and a pixel's coverage is the *exact* sum of its overlaps — now
  computed, giving all 256 levels, conserving ink at any scale, and costing less
  at large sizes because the work is per overlapping cell rather than per sample.
- **Outline glyphs were vertically under-sampled.** Horizontal coverage was
  already exact, but only four vertical samples per row meant five levels on any
  nearly horizontal edge — the top and bottom of every round letter. Raised to
  sixteen, which costs nothing at runtime since glyphs are cached per size and
  phase.
- **Outlines and shadows grew by whole pixels, in the shape of an octagon.**
  The mask was grown by repeated max-filtering with 4- and 8-neighbour kernels,
  which can only step a whole pixel at a time — a 1.5-pixel outline became 2 and
  anything under half a pixel vanished — and alternating the two kernels
  approximates a disc with an octagon, plainly visible at the one to three pixels
  an outline actually uses. It is a **distance field** now, thresholded with a
  half-pixel ramp: exact Euclidean distance by two one-dimensional passes, seeded
  from the coverage itself so the field is accurate to a fraction of a pixel
  rather than to the grid. Fractional radii work, the shape is round, and the
  grown edge is as smooth as the one it came from.

  One bug the tests caught on the way: a radius under half a pixel *dimmed* the
  glyph. A pixel well inside a shape is a negative distance from its boundary,
  which the field does not carry, so the ramp came out below full. Growing a shape
  is a union with it, and now it says so.
- **Kerning.** Advance widths say what each glyph needs on its own; kerning is
  what a *pair* needs, and there was none — so `AV` and `To` sat visibly loose,
  the diagonal and the overhang leaving a gap neither advance knows about. The
  `kern` table is read now (format 0, horizontal, non-cross-stream — what a
  TrueType font with kerning actually ships) and applied in both measurement and
  drawing. Measuring without it would put every centred line off by the sum of
  its pairs. Apple's variant of the table starts with a different header and is
  left alone rather than guessed at, and kerning held in `GPOS` — where a modern
  font usually keeps it — is not read.

  Checked against the fonts on this machine rather than only against a synthetic
  table: `AV`, `To` and `LT` all tighten in Arial Bold, Times New Roman Italic
  and Trebuchet MS, while `oo` is exactly zero, which is the control that says
  the numbers are the font's and not the parser's.
- **The other caption compositing path still mixed encodings.**
  `CaptionOverlay::composite` blends a prepared overlay onto a frame in integer
  arithmetic on sRGB bytes — the same error as `blend_pixel`, in the path that
  had not been touched, and so inconsistent with both the CPU painter and the GPU
  pass once those were fixed. Now in linear too.
- **Text was composited on encoded bytes instead of on light.** An sRGB byte of
  128 is about a fifth of the light of 255, not half, so mixing the encodings put
  half coverage of white on black at 128 where the light says 188 — a pixel less
  than half as bright as it should be. That is why antialiased text read thin on
  a dark background and heavy on a light one, and at a quarter coverage the error
  was worse: 64 against a correct 137, so fine strokes all but vanished.
  Compositing is now done in linear on both sides.

  Getting there needed the GPU path to agree, and the reason it did not was not
  the obvious one. Fixed-function blending is linear only on an *sRGB
  attachment*; the caption pass was blending into a plain `Unorm` view of a
  framebuffer that already held encoded colour, so the hardware was mixing
  encodings. Measuring rather than reasoning is what found it — GPU output came
  out at 73 where the CPU had 146, and `encode(decode(decode(146))) = 73`
  exactly, one decode adrift. The pass now composites through an **sRGB view** of
  the target, which `RenderTarget` already declares in `view_formats`, so the
  blend is linear whichever format the target itself is, and the overlay texture
  is declared sRGB so its own colours decode. `tests/captions_overlay_gpu.rs`
  holds the two paths within a channel or two of each other.
- **Glyphs sharing a pixel took the stronger coverage instead of combining.**
  Once glyphs sit at subpixel positions, neighbours genuinely share edge pixels,
  each covering part of one. `max` discards the other's contribution and thins
  the gap between letters; the union `a + b − ab` is what two partial coverages
  of one pixel actually come to.

- **glTF ignored `extensionsRequired`.** It means what it says: the asset cannot
  be read correctly without that extension. The only such files that failed were
  the ones that happened to become unreadable — a Draco mesh errors because its
  accessors point at compressed bytes, which is luck rather than a check — so a
  file requiring `KHR_texture_basisu` or `KHR_materials_variants` loaded with
  that part silently missing. Unknown required extensions are now refused by
  name; `extensionsUsed` is still a refinement a reader may ignore, because
  treating it as fatal would refuse most real files.
- **The wasm mesh-BVH path did not compile.** `src/wasm.rs` built
  `BuildOptions` field by field, so when `split_degenerate` was added to the
  struct the browser build broke while every native check stayed green — the
  failure only appears under `wasm32` *and* `mesh-bvh`, which nothing was
  checking together. Fixed by spreading the default rather than by adding the
  field, since exhaustive initialization is what made it fragile, and the
  default is also what keeps these APIs behaving as their native twins.

- **glTF ignored `byteStride`.** Any interleaved buffer read positions partly
  made of normals. Every meshopt-compressed file is interleaved, and plenty of
  uncompressed ones are.
- **glTF normalized every integer attribute.** The accessor's `normalized` flag
  was not consulted, so `KHR_mesh_quantization`'s plain-short positions were
  divided by 65535 and the mesh collapsed to a speck at the origin. Signed
  `BYTE`/`SHORT` and `UNSIGNED_INT` were not read at all.
- **FBX panicked on a truncated file.** Lengths and offsets come out of the
  bytes being read, and the binary reader indexed on them directly. Every read
  is bounds-checked now, with truncation and corruption sweeps to prove it.
- **FBX ignored layer-element addressing.** A normal array may be per corner,
  per control point, per polygon or one value for the mesh, directly or through
  an index array. Assuming one of those gives the right *count* of wrong
  normals, which reads as bad lighting rather than as a bug.

- **Every ellipse and arc came out rotated a quarter turn.**
  `EllipseCurve::get_point` destructured `rotation.sin_cos()` as `(cr, sr)`, but
  it returns `(sin, cos)` — in that order — so the rotation matrix was built
  from the two swapped. A full circle is unchanged by a quarter turn, which is
  why `Path::arc` looked right and this survived; every partial arc was wrong,
  and so was the pen position it left behind for whatever was drawn next.
  Found by writing arcs out as SVG and noticing a semicircle that should start
  at `(5, 0)` starting at `(0, 5)` instead.
- **Orthographic picking never hit anything.**
  `Raycaster::set_from_camera_ortho` unprojected NDC `z = -1` for the near
  plane, which is where OpenGL puts it. This crate's projection matrices use
  the wgpu/D3D range `[0, 1]`, so that landed the ray origin most of a frustum
  *behind* the camera: every hit then measured past `far` and was discarded,
  and picking under an orthographic camera silently returned nothing at all.
  Found by using the raycaster as ground truth for the SVG renderer's tests,
  where it claimed an entire scene was empty sky.

- **The post passes made callers hand them the camera's clip planes.**
  `set_ssao` took six arguments and `set_depth_of_field` five, four of them
  properties of the camera, each documented as *"wrong otherwise, and wrong
  silently"*. The renderer has had them all along — it writes `cam_near` and
  `cam_far` into the frame uniform every frame — so they are read from the frame
  now and the setters take two and three arguments.

  The one caller in the tree was indeed passing the wrong ones:
  `examples/simcity_render.rs` computed its field of view as
  `camera.fov.to_radians()`, and `fov` is already radians, so the city's ambient
  occlusion ran with a lens 57 times too narrow and still produced something
  that looked like ambient occlusion. `PerspectiveCamera::fov_degrees` and
  `pixels_per_unit` close that trap for callers who still need the numbers.

- **Nothing was checking that a malformed asset is an error rather than a
  panic.** Six formats here are parsed by hand and every one reads offsets and
  lengths out of the file before using them; the distance between "returns an
  error" and "indexes out of bounds" is one missing check in code that looks the
  same either way. 896 corrupted inputs — truncations, scattered corruption,
  length fields blown up to 2 GB — now run through every parser in under a
  second. All six were already clean; the test is what keeps them that way.

  Mutation-testing it was the point. The first version ran twice as many inputs
  and could not catch an unchecked `bytes[64]` added to `decode_mesh`, because
  every truncation was a *fraction* of the file and the shortest input it ever
  produced was 350 bytes — it never handed a header parser a file shorter than
  its own header.

- **`PmremGenerator::from_cube` returned its argument unchanged.** The name
  promises the one step that makes `roughness` mean anything, so a caller who
  reached for it got an environment with no mip chain, no CubeUV atlas and no
  prefiltering — which does not look broken, it looks like a slightly dull
  render with no specular highlight anywhere in it. It prefilters now, and is a
  no-op on an environment that already is one.

- **Bloom, ambient occlusion and the film grade never reached the RGBA
  readback.** They live in the compute pack that feeds a video encoder, so
  asking for them and then calling `render_to_rgba_resolved` — which is what
  every headless example does — got a frame with none of them applied and no
  warning. A setter that returns quietly and does nothing is the worst shape a
  setting has. That readback runs the same pack and widens its output now, so
  there is still only one implementation of the composite and both paths get it.

- **`framebuffer_encode` clamped on a half-float target.** `.w > 0.5` means
  "write linear, this is an HDR render target", and it was clamping to 1.0
  anyway — which made the float format behave like an 8-bit one, left bloom
  nothing to threshold, and gave a `ShaderMaterial` no way to hand HDR onwards.
  Clamping is right for the 8-bit canvas, where the sRGB encode is only defined
  on [0, 1], and wrong for a format chosen because the values exceed it.

- **Every documented helper vanished from the shader-material preamble.** A
  `ShaderMaterial` is compiled against "everything in `SHADER_SOURCE` above the
  first `@fragment`", and the contract its module docs state is that the whole
  built-in environment comes with it. Adding `fs_shadow_point` — the point-cube
  depth pass, which has to exist — put a fragment entry point nine hundred lines
  above `fs_main`, and the preamble was silently cut off at it. Everything from
  `linear_to_srgb` to `sample_env_cube` to `pbr_brdf` went with it, including
  `framebuffer_encode`, which `examples/shader_material.rs` calls: that example
  could no longer create its pipeline.

  Nothing about the failure points at the cause. It arrives as
  `no definition in scope for identifier`, at pipeline-creation time, naming an
  identifier in the *user's* shader. The entry point has moved below `fs_main`,
  which is where any future one has to go, and two tests say so — one checking
  the preamble still carries the helpers, one stating the ordering rule directly
  so a violation names the cause rather than the symptom.

- **Twelve test targets ran no tests and reported success.** Each gates itself
  with a file-level `#![cfg(feature = ...)]` stricter than the
  `required-features` its `[[test]]` entry declared — or than nothing at all,
  for the nine that had no entry. Cargo therefore built an empty binary, ran it,
  and printed `0 passed`, which reads exactly like a target that is simply
  small. `tests/brep.rs` declared `brep` and reaches into `brep::planar`
  (`brep-kernel`) and `brep::intersect` (`brep-csg`), so it did not even
  compile under the features it asked for.

  Declaring what each one actually needs turns **115 tests back on**:
  `brep_boolean` 45, `brep` 27, `brep_csg` 13, `assembly_real_geometry` 10,
  `captions_video` 8, `rlx_bridge` 4, `video_export_streaming` 2, `h264_gpu` 1,
  plus `backend_parity`, `metal_backend`, `h264_metal`, `mp4_demux` and
  `simcity_bake`, which were running under `--all-features` but not under the
  features they named. The same audit is now a few lines of Python: parse
  `[[test]]`, expand the feature graph, and compare against each file's inner
  `cfg`.

  `cargo test` on default features is also quicker for it, since nine empty
  binaries no longer get built and linked.

- **Two Metal ABI assertions had gone stale, which is the one kind of stale
  that matters.** `uniform_structs_match_the_shader` exists to catch a Rust
  struct drifting from its `shaders.metal` twin, and it was asserting
  `size_of::<DrawUniforms>() == 240` against an actual 288 — the three `uint4`
  and `float4` rows the extra material maps and `receive_shadow` added. The
  `FrameUniforms` formula was short by 176 bytes, exactly the directional, spot
  and point shadow blocks. Both sides of the ABI were in fact correct and the
  Metal render tests pass, so the only thing broken was the guard. Both numbers
  now spell out their terms, the way the `FrameUniforms` one already did, and
  deleting one row from either struct makes the test fail.

- **Nineteen broken rustdoc links, three doctests that did not compile, and the
  cause of most of them.** A `mod` declaration that carries an outer `///`
  comment *and* an inner `//!` block has the two fragments merged, and the
  merged text resolves in the **parent's** scope — so every `[`Item`]` in
  `videotoolbox`, `raytrace`, `openscad::freecad` and friends pointed at a name
  that is not in scope there, with no span in the warning to say where. Those
  are fully qualified now. Links to private helpers and to feature-gated items
  are code spans, which is what they can be in every configuration.

  The doctests: `codec::mp4_stream` ended in a `# fn _unused() {}` that made
  rustdoc read the whole block as items, so its `let`s were global variables;
  `videotoolbox_decode` called a `decode()` that does not exist (the API is
  `push` then `take`, which is the part that matters — decode order is not
  display order); and `UsdLoader::open` imported `ComposeOptions` and
  `FileResolver` from `loaders` rather than `loaders::usd`. `cargo doc` is clean
  in all three configurations that ship — default, the docs.rs feature set, and
  `--all-features` — and all 132 doctests pass.

- **`renderer::accumulate` was dead code on wasm.** Only `headless` uses it and
  that is native-only, so the module now carries the same gate. Four dead-code
  warnings on `wasm32`, and weight in a bundle this release spent some effort
  shrinking.

- **Three examples did not compile with `--no-default-features`.**
  `captions_render`, `fontprobe` and `retrofuturism` use `threers::captions`,
  which is a *default* feature — so they built everywhere except the one
  configuration nobody runs. They declare `required-features = ["captions"]`
  now.

- **Four publish roots were still on 0.0.5** — `crates/threers-js/deno.json`,
  `web/package.json` and both lockfiles — and `release-versions.sh` printed six
  of the thirteen. It now prints all of them, and `CHECK=1` exits non-zero on
  drift; `release-all.sh publish` runs that check before it uploads anything,
  because npm and crates.io both refuse to replace a version and there is no
  undo.

  It also caught the real blocker: `threers-physics` and `threers-probe` were
  still at **0.0.1, which has been on crates.io since 2026-09-11**. Both have
  changed since, so both go out as 0.0.2 — and because the bindings pin
  `threers-physics` by version as well as by path, `threers-py` and
  `threers-node` had to move with it or nothing resolved at all. That pairing is
  checked too.

- **One `### Fixed` section in this entry had become two**, 900 lines apart,
  from an insertion made while the rest of it was being written.

## [0.0.5] — 2026-09-10

<!-- Still to write up for this release. The sections below cover the lattice
     work only; these areas also changed since v0.0.4 and are not described
     anywhere in this entry (added lines against the tag):

       src/raytrace          5447   motion blur / shutter on RtCamera, denoise,
                                    film, gpu backend
       src/codec             3382
       src/metal              905   post-fx chain (new src/metal/postfx.rs)
       src/renderer           501
       crates/threers-physics 412   three-bearing swivel examples
       src/videotoolbox.rs    360
       src/mesh_bvh           102   degenerate-split fallback + front-to-back
                                    traversal — see the note in Known issues
       crates/threers-connectome    new crate (untracked)

     `git diff --stat v0.0.4` for the full picture. -->

Lattices stopped being a geometry generator and became something you can size a
part with: foams, cells that follow the part instead of being cut by it, and
numbers — stiffness, conductivity, strength, pore size — that come out of a
solve rather than a rule of thumb.

### Added
- **Stochastic lattices** — six cells with no repeating unit at all:
  `Voronoi` (open-cell struts on the Voronoi edges), `VoronoiWall`
  (closed-cell), and four spinodal random-field classes — isotropic, lamellar,
  columnar and cubic (the spinodoid taxonomy of Kumar et al., *npj Comput.
  Mater.* 2020). `Lattice::seed` picks the foam and `Lattice::jitter` sweeps
  from a regular grid to a fully stochastic one. No point set and no RNG is
  stored: the seeds are a hash of their cell index and the wave directions a
  hash of their own, so the field is a pure function of the point and the seed
  at any resolution, on any thread, in any process.
- **Conformal lattices** — `Lattice::conform` maps the point into cell space
  before the periodic field is evaluated, so a tiling can close around a nozzle
  or stack whole layers through a wall that curves, instead of being cut
  mid-strut where the part ends. `Conform::cylindrical` / `spherical` /
  `depth(region)` / `new(map)`, plus `Conform::ring_pitch` for the cell size
  that closes a ring without a seam. Each map reports its stretch and the
  thickness is divided by it, so a wall is a length and not a coordinate.
- **`Field`** — the adapter between a solver result and
  `Lattice::grade`, which takes a closure and is therefore the least useful
  thing to be handed. Build one from a grid, a function, scattered points
  (solver output, sensor readings, a point cloud) or a value per mesh vertex;
  `into_grade(at_min, at_max)` reads its range once and maps it onto two
  thickness multipliers. `CuboctFrame::element_stress` and `stress_field` close
  the loop for beam lattices: solve the block, grade the lattice on what it
  said.
- **Homogenisation** — `Lattice::homogenize` voxelises one periodic cell, solves
  six unit macroscopic strains on it with periodic boundaries, and reads the
  effective 6×6 stiffness off the strain energy. `Stiffness` gives Young's and
  shear moduli, Poisson ratios, `directional_modulus` off the axes, and
  `anisotropy`. A fully solid cell returns the base material exactly.
  `homogenize_window` measures several cells at once, which a foam needs
  because it has no cell to repeat.
- **Effective conductivity** — `Lattice::conductivity`, the scalar cousin of the
  same solve: one unknown a node, three unit gradients, a 3×3 tensor with
  `principal` (eigenvalues, so orientation does not matter), `anisotropy` and
  `tortuosity_factor`. Heat, electricity, diffusion and permittivity are the
  same equation, so it is the same number for all of them.
- **Collapse strength** — `Lattice::strength` reads the local von Mises stress
  per unit of macroscopic stress out of the fluctuation fields the stiffness
  solve already produced, so it is one set of solves and not two. `Strength`
  gives `yield_strength`, `uniaxial`, `efficiency` (the share of the material at
  yield when the cell gives) and `collapse_strain`. `resolved()` reports whether
  the grid was fine enough to believe, because this one converges from *below*
  and a coarse grid overstates strength.
- **Metrics** — `Lattice::metrics` returns porosity, internal (`wetted`) versus
  total surface area, area per unit of part and per unit of material, pore
  diameter and ligament thickness by Euclidean distance transform, hydraulic
  diameter, and a Kozeny–Carman permeability estimate. It samples fine enough to
  see the wall before it measures anything, and `wall_samples` on the result
  says whether it managed.
- **Void connectivity** — porous, connected and flowing are three different
  questions, so the void is flood-filled and asked all three: `open_porosity`
  and `closed_porosity` (what can be drained), `percolates` per axis (what can
  flow across), and `largest_void_fraction` — 1 for an open-cell foam, about a
  half for a sheet TPMS's two labyrinths, near zero for a closed-cell one.
- **GPU solves** — `Lattice::solver(Solver::Gpu)` runs the homogenisation
  conjugate gradient as a wgpu compute pipeline: 7–13× faster than the CPU on a
  cell worth the trouble, and the moduli agree to four or five significant
  figures despite the device solving in `f32` — 0.00 % apart on three of four
  measured cells and 0.01 % on the fourth. (The effective tensor is read off an energy, and
  energy is stationary at the solution, so an error in the displacement field
  appears squared in the answer.) Opt-in rather than automatic because the two
  are not bit-identical; falls back to the CPU with no adapter and on wasm, and
  `solver` on the result says which one ran.
- **`Region` is `Clone`** — filling a shell and conforming to the same surface
  is one region rather than two, and cloning is a reference count.
- **`examples/lattice_engineering`** — foams, conformal cells, a field-driven
  grade, and the two tables the rest of this is for.

### Changed
- **`fit_relative_density` is about twelve times faster** — 87 s to 7.3 s across
  all 35 generators. The density sampler was inheriting the build grid's cull
  margin, which exists to keep the gradient exact when contouring and is waste
  for a test that only reads a sign; samples-per-cell was the wrong knob for a
  part a hundred cells across, so the estimate is now bounded above *and* below
  by a total sample count; and twenty bisection steps were resolving thickness
  to a nanometre. Accuracy cost, measured against an independent voxel count:
  0.3 %.
- **`resolve_walls` and `wall_samples` see the grade** — they read the nominal
  thickness before, so a part graded down to two fifths sized its grid for the
  thick end and came out as gravel at the thin one. The grade is swept on a
  coarse grid and the minimum is what the sampling is sized for.
- **`Stiffness` and `Conductivity` report `voxels`** — the grid tops out at 48
  and a multi-cell window multiplies into that ceiling, so the number actually
  used comes back rather than the number asked for.
- **`LatticeKind::all()` is 35** — the six stochastic cells join the gallery,
  which `examples/lattice` picks up without changes.

### Fixed
- **Concentric infill homogenised to nothing.** It is the one pattern that reads
  how deep into the part it is, and "deep inside" was passed as infinity, which
  put its first ring infinitely far away: the cell voxelised empty and the
  stiffness came back zero, silently. A sweep over every generator now guards
  against the whole class.
- **`--features video` on its own did not compile.** The native H.264 export
  path used `crate::codec` without the `native-codec` gate its only caller
  already had, so it only built when something else happened to turn that
  feature on. Every leaf feature now builds as a library by itself.
- **Two dead branches** that clippy found and that were doing nothing: an `if`
  in the raytracer's line primitive whose arms were both `i + 1`, and one in the
  procedural-city sign code choosing between `o` and `o`.
- **`BuildOptions::split_degenerate`** — when the chosen plane separates
  nothing, the builder used to give up and turn the whole subtree into a single
  leaf, which a query then scans linearly; one degenerate split at the root
  costs every later query the entire scene, and a handful of triangles far
  larger than the rest (ground planes, backdrops) is enough to cause it. The
  fallback splits at the median centroid instead. It is **off by default**: the
  tree's shape is read, not just queried, by `bvhcast`'s leaf-pair enumeration
  and through it by the CSG evaluator, and the default reproduces
  three-mesh-bvh's. Turn it on for raycast and closest-point trees, where
  nothing reads the shape.
- **Throughput assertions failed every debug run.** The BVH raycast tests
  asserted rays per second unconditionally, and the release checklist runs
  `cargo test` without `--release`, where the same traversal is several times
  slower. The rate is still printed on every run and still enforced where
  optimisations are on.

### Housekeeping
- `cargo clippy --workspace --all-targets --all-features` is clean, as are both
  wasm targets. Where a lint was a genuine disagreement rather than a defect it
  is allowed at the narrowest scope that covers it, with the reason written
  down — graphics signatures are wide because the quantities are independent,
  and an `Id` in the Metal backend always comes from the Metal runtime.

### Known issues
- **A boolean's result depends on the shape of the BVH used to find its
  candidate pairs**, which it should not.
  `collect_intersecting_triangles` marks any *coplanar* pair `bvhcast` hands it
  as intersecting, whether or not the two triangles are anywhere near each
  other — so a coarser tree, whose leaves enumerate more pairs, marks more
  triangles and produces more geometry. On the CSG parity meshes the tree
  degenerates to three nodes, `bvhcast` becomes brute force, and every coplanar
  pair in both meshes gets marked; a properly split tree returns 3816 pairs
  instead of 6240 and the window frame comes out at 751 vertices instead of
  4074. Neither enumeration is wrong — the tighter one was verified against
  brute force to miss no genuinely overlapping pair — but the evaluator should
  decide coplanarity from the geometry rather than from what the accelerator
  happened to deliver. Until it does,
  [`BuildOptions::split_degenerate`] keeps the two apart.

[`BuildOptions::split_degenerate`]: https://docs.rs/threers/latest/threers/mesh_bvh/struct.BuildOptions.html

## [0.0.4] — 2026-08-22

The renderer grew a second way to make a picture, a second backend to make it
on, and a second crate to make things move.

### Added
- **Browser wasm variants** — npm `threers` ships `./mini` (default, smallest)
  and `./full` (`wasm-full`: OpenSCAD/CSG, NURBS, codecs, planet, raytrace) so
  browsers only download the entry they import. `VARIANT=mini|full web/build.sh`.
- **Unified release scripts** — `./scripts/release-versions.sh` and
  `./scripts/release-all.sh` (build / pack / publish) for crates.io, npm ESM,
  npm native, and PyPI; multi-arch via tag push to CI.
- **Meta feature bundles** — `cad`, `media`, `gi`, `apple`, and `full` compose
  existing leaf flags; feature table in `src/lib.rs` now documents `nurbs` /
  `brep*` / `step` / `assembly-check` / `videotoolbox` / `learned-denoise*`.
  Also `wasm-full` for the browser kitchen sink.
- **`threers::wgpu`** — the `wgpu` this crate was built against, re-exported, so
  a caller on the native path cannot end up holding a `Device` from a different
  version of it than the renderer expects.
- **Nested prelude modules** — `prelude::{controls,animation,loaders,helpers}`
  plus feature-gated `csg` / `openscad` / `nurbs` / `raytrace` / `captions`.
- **Native Node package `threers-node`** — napi-rs (`#[napi]`) bindings mirroring
  the Python surface, with multi-arch optional deps (`@threers/node-*`). Deno and
  browsers stay on the wasm ESM package; Neon is not used. See
  [`docs/bindings.md`](docs/bindings.md).
- **Multi-arch release workflow** — `.github/workflows/release-language-packages.yml`
  builds npm ESM (wasm), napi platform binaries (arm64/x64), and PyPI wheels.
- **Published JS package** — `crates/threers-js` is the npm / Deno / Node ESM
  publish root (`threers`), staging the THREE shim + wasm via `./build.sh`.
- **Published Python package** — `crates/threers-py` is the PyPI wheel
  (maturin / PyO3): headless render, math, Tween, and PhysicsWorld.
- **Browser cinematic camera helpers** — `THREE.CameraPath`, `THREE.ShotTimeline`,
  and a richer `THREE.CameraAnimator` (speed ramps, path follow, Vertigo, look/FOV)
  on the three.js-compatible shim. Wasm `WebCamera.setView` batches eye + look-at + FOV
  for cinematic frames. Demo: `web/examples/camera-shots.html`. See
  [`docs/camera-animation.md`](docs/camera-animation.md).
- **General animation stack** — real `PropertyBinding` / `PropertyMixer`, mixer
  `fadeIn` / `fadeOut` / `crossFadeFrom` / `crossFadeTo`, additive blend mode,
  morph / material / bone path binding, `TrackModifier` (noise / cycles / stepped),
  plus `Tween`, `Spring`, `Timeline` (markers + time remap), `ObjectSpring`, and
  `PoseBlend`. Rust mixer gains weights, fades, cross-fade, morph weights, and
  glTF `weights` channels. Demo: `web/examples/animation-demo.html`.
- **Animation ↔ physics handoff** — `PhysicsBlend` freezes the limp pose (no
  yank from a continuing clip), inherits stride velocity, clears velocity on
  recover, and writes via `apply_to_scene`. `World::sync_to_scene_where` skips
  bodies the blend still owns so scene sync cannot undo the crossfade.
  Example: `cargo run -p threers-animation --example anim_physics_blend --features physics`.
- **Path tracing (`raytrace`)** — a physically based integrator beside the
  raster renderer: global illumination, area lights, refraction and depth of
  field, running on the CPU or as a wgpu compute pass (`pathtrace.wgsl`).
  Multiple-importance sampling, a BSDF set, HDRI environment lighting, and a
  progressive film that can be read back mid-render.
  - **Denoising** — an À-Trous edge-avoiding filter guided by albedo and normal
    G-buffers, and behind `learned-denoise`, a trained U-Net run natively
    (`learned-denoise-metal` / `learned-denoise-gpu` pick the backend).
- **NURBS, B-rep and STEP** — a four-step ladder, each implying the one below:
  - `nurbs`: rational curves and surfaces, knot insertion/refinement,
    derivatives, and tessellation.
  - `brep`: analytic surface provenance carried on geometry, so a face
    remembers the plane, cylinder or sphere it was cut from.
  - `brep-csg`: closed-form surface/surface-intersection fast paths in the CSG
    kernel, taken when both operands are analytic.
  - `brep-kernel`: real B-rep topology with its own boolean and fillets.
  - `step`: AP203/214 import and export over a Part 21 reader/writer.
- **Metal backend (`metal`)** — a second renderer talking to Metal through the
  Objective-C runtime on macOS/iOS, with its own headless path. `visionos` adds
  CompositorServices and ARKit on top.
- **`videotoolbox`** — in-process VideoToolbox encode on macOS, no ffmpeg
  process and no temporary frame directory.
- **Captions (`captions`, on by default)** — SRT and WebVTT parsing, layout and
  rasterization, drawn as an overlay pass or burned into an export.
- **H.264/MP4** in `native-codec`, joining the existing HEVC, VP9/WebM, APNG and
  GIF encoders — still pure Rust, still wasm-safe.
- **Planets (`planet`)** — planet and moon surfaces, starfields, and baked map
  generation, including NASA imagery ingest.
- **RLX bridge (`rlx`, `rlx-geo`)** — tensors to and from geometry and pixels, a
  graph runner, convolution post-fx, mesh smoothing, palette extraction, and
  exact Delaunay with adaptive refinement and Voronoi cell textures.
- **Geometry** — kirigami corrugations; a lattice family (struts, TPMS,
  cuboctahedral cells, voxel meshing and infill); rigid origami with crease
  patterns, folding and development sheets; NURBS geometry.
- **Kinematics** — chains, actuators, servos, transmissions and a plant model,
  with material and design helpers.
- **`assembly-check`** — body correspondence, axis recovery and swept-volume
  interference checks on an assembly's motion.
- **`parallel` / `async`** — rayon-backed CSG and BVH work (a deliberate no-op
  on wasm32, where rayon has no threads), and an optional tokio runtime for
  callers driving loaders concurrently.
- **`manifold`** — the Manifold kernel as an alternative backend behind the
  OpenSCAD booleans.
- **Post-processing** — bloom, SSAO, and a downsample pass; BC1 texture
  compression.
- **`threers::prelude`** — the common imports in one `use`.
- **Companion crates**:
  - [`threers-physics`](crates/threers-physics) — rigid bodies, shapes, joints,
    sleeping, collision filtering, sensors, raycasts and shape casts, FABRIK/CCD
    inverse kinematics, a kinematic character controller, and colliders built
    straight from a `Solid`. Optional `parallel`, `gpu` (a wgpu compute broad
    phase that works on wasm32/WebGPU, where rayon cannot), `async`, `assembly`
    and `openscad`.
  - [`threers-probe`](crates/threers-probe) — screen-space neural GI from a
    G-buffer and lighting probes, trained in RLX.
  - [`threers-animation`](crates/threers-animation) — easing, tweens, springs,
    timelines, scene-node clips, and **cinematic cameras** with Blender-parity
    extras (Track/Damped/Locked, Follow Path+tilt, F-curve modifiers, NLA,
    marker binds, drivers, lens shift, panoramics, sensor fit, DoF blades,
    stereo pivots, guides, walk/fly record, time remap, rolling shutter).
    `OrbitControls` gained damping + auto-rotate. Not published with 0.0.4;
    see [docs/releasing.md](docs/releasing.md).

### Changed
- **BREAKING (native Rust callers): wgpu 0.20 → 30.** `Renderer::new` takes a
  `wgpu::Device`, a `wgpu::Queue` and a `wgpu::TextureFormat`, and 36 other
  public functions take or return `wgpu` types. A Rust type belongs to the exact
  crate *version* it came from, so `wgpu::Device` from 0.20 and `wgpu::Device`
  from 30 are unrelated types that print identically — and cargo builds both into
  one graph without complaint, because they are semver-incompatible and allowed
  to coexist. A caller who stays on 0.20 therefore gets `expected `wgpu::Device`,
  found `wgpu::Device`` with nothing on the line to explain it. Move your own
  `wgpu` to 30, or better, use the `threers::wgpu` re-export added below and stop
  having a second copy to keep in step. `HeadlessRenderer` and the wasm/JS paths
  are unaffected: they never name a `wgpu` type.

  Ten major versions in one step, and it removes the last
  use of the unmaintained `block 0.1.6` from the tree: wgpu's Metal backend
  moved from `metal-rs` to `objc2` in 29, so the soundness warning that shipped
  with every macOS build of 0.0.3 is simply gone. It also collapses a duplicate
  — with `rlx` enabled the build compiled wgpu 0.20 *and* wgpu 30 side by side,
  two complete graphics stacks, and now compiles one.

  What changed on the surface: the `ImageCopy*` types are `TexelCopy*`,
  `Maintain` is `PollType` and returns a `Result`, `request_adapter` returns a
  `Result` rather than an `Option`, `present` moved from the surface texture to
  the queue, `get_current_texture` returns a `CurrentSurfaceTexture` enum
  instead of a `Result`, mapping a buffer range is fallible, and `BufferViewMut`
  is write-only because mapped memory may be write-combining.

  One thing was a real bug the upgrade exposed rather than caused: the main
  pass declares 20 sampled textures in its fragment stage while the headless
  device asked for `downlevel_defaults()`, which allows 16. wgpu 0.20 never
  checked; wgpu 30 does. The device now asks for the binding counts the adapter
  actually has, and the renderer has presumably been over that line since it
  was written.

  It also unlocks something that was blocked on the version skew: rlx's GPU
  backend is now available on wasm32. Two wgpu majors in one wasm binary both
  generate bindings for the same WebGPU interfaces, and `wasm-bindgen` rejected
  that outright, so the browser build had rlx pinned to its CPU backend. One
  major, one set of bindings, and the build goes through.
- `default` now includes `captions`; `video` implies it.
- `Vector3` gained `RIGHT` and `FORWARD` beside the existing `UP`.
- `KirigamiPreset::all` and `::count` are `const fn`, so a preset count can size
  an array.
- The crate `exclude`s `tests/parity/scenes/**` — 22 MB of reference dumps that
  only the parity harness reads, and that were counting against the package.
- The repository is a Cargo workspace; the library remains its root member.

### Fixed
- **`manifold` backend: a boolean could return the wrong solid, confidently.**
  The backend verified its *result* and not its *operands*, and a mesh Manifold
  declines to import behaves as an empty solid rather than as an error. So a
  subtraction whose second operand failed to import returned the first operand
  untouched — and since that operand was closed to begin with, the
  watertightness gate downstream had nothing to object to. A `sphere($fn=32)`
  bitten out of a cube came back as the whole cube. The operands are now checked
  on the way in, and a boolean that cannot be trusted falls through to the
  arrangement kernel as it was always meant to.
- **Shadows were lost entirely in any scene without an environment map.**
  Raising `map_size` reallocates the shadow depth texture, and the environment
  bind group that samples it was invalidated by clearing its two cache keys to
  `None`. A scene with no environment map already has `None` for both, so the
  `!=` guard detected nothing, the group was never rebuilt, and every frame
  sampled the *old* texture — one that nothing renders into. The effect was that
  a shadow-casting directional light lost its own contribution on the surfaces it
  lit: a rig with several lights merely looked dim, and a single-sun rig looked
  unlit. Invalidation is now an explicit flag, because it cannot be expressed by
  clearing keys that were already empty.
- A point light's six shadow cube faces all rendered with the last face's
  view-projection. The faces were encoded into one command buffer while writing
  their matrices into one shared uniform buffer, and `queue.write_buffer` lands
  before *any* submitted command — so the last write won for all six. Each face
  now has its own uniform buffer and bind group.
- `ShadowSettings::normal_bias` is read. It was declared, documented, and then
  never looked at — the same failure the renderer's own note about `map_size`
  describes: a caller sets it, sees no error, and gets nothing. The shadow
  lookup now steps off the surface along its normal before sampling.
- The OpenSCAD render's shadow frustum is fitted to the model instead of
  spanning `0.01 .. 8 × reach`. The depth bias is applied in NDC, so what a
  given bias is worth depends on how much world depth the near–far range covers;
  most of that range was empty space.
- **Two features did not compile on their own.** `native-codec` uses
  `crate::captions` in its muxers without depending on it, and `brep` calls the
  surface/surface intersector that only `brep-csg` compiles. Both are invisible
  from `--all-features` and from the default set — `captions` is a default, and
  anything that enables `brep` in a normal build tends to enable `brep-csg` too —
  so they only fail for the person who asks for one feature and nothing else,
  which is exactly what `default-features = false` is for. `native-codec` now
  depends on `captions` as `video` already did, and the `brep` curve falls back
  to its own parameterisation when the intersector is not there.
- **`Solid::parts` appended its pieces where it should have united them.**
  `parts` splits a model by `color()` and distributes the booleans over the
  groups, which is exact — `(a ∪ b) − c` becomes `(a − c)` and `(b − c)` — and
  then has to put the pieces sharing a colour back together. It joined them with
  a triangle-soup concatenation instead of a union, so geometry the model had
  merged came back as separate overlapping shells: an uncolored `cube ∪ cube`
  yielded 72 vertices of z-fighting where `to_geometry_exact` gives 36. Only the
  display path was affected — the export path was always correct — which is why
  it showed as shimmering interior faces and twice the triangles rather than a
  wrong mesh on disk. Each colour group is now folded back into one `Solid` and
  evaluated once, as its own documentation always said it did.
- Rigid origami: `fold_twist` walked its candidate crease assignments but `?`
  returned from the whole function on the first one that would not fold, so a
  later, foldable assignment was never reached.

## [0.0.3] — 2026-08-10

### Added
- `openscad` feature — OpenSCAD-style solid modeling, two front ends onto one
  `Solid` CSG tree:
  - **`.scad` interpreter** (`parse_scad` / `parse_scad_file`): primitives,
    transforms, booleans, `linear_extrude`/`rotate_extrude` (twist/scale/slices),
    `hull`/`minkowski`/`offset`/`projection`/`fill`, modules, first-class
    functions, list comprehensions, the `*`/`%`/`!`/`#` modifiers, special
    variables, and builtins including `rands()` and `fill()`.
  - **Rust DSL**: the `Solid` builder, the `scad!` macro, and
    `union!`/`difference!`/`intersection!`/`hull!` combinators; `rotate` (degrees),
    `rotate_axis`, `mirror`, and a `solid()` escape hatch.
  - **Watertight exact-CSG kernel** (`exact_csg`): mesh co-refinement + per-face
    constrained-Delaunay triangulation + winding/ray-parity classification behind
    a closed-2-manifold gate. Resolves genuinely *curved∧curved* booleans
    (sphere∪sphere, cylinder∩cylinder, cone/mixed), which the float kernel cannot;
    falls back to the float kernel only where a result can't be verified (never
    emits an unproven mesh).
  - **Mesh import**: STL, OBJ, OFF, 3MF, AMF, and 2D DXF/SVG; `surface()`
    heightmaps from `.dat` and `.png`.
  - **Mesh export**: STL, OBJ, OFF, 3MF (OPC/ZIP), and binary glTF 2.0 (`.glb`) —
    `Solid::to_stl`/`to_obj`/`to_off`/`to_3mf`/`to_glb`; the `scad2stl` example
    picks format by output extension.
  - **Browser**: a live [OpenSCAD playground](web/openscad-playground.html)
    (edit code → WebGPU render + STL/OBJ/OFF/3MF/GLB download), a demo gallery, and
    parametric 3D-printer / NEMA-17 assemblies. wasm bindings `scad_geometry`,
    `scadExport`, `scad_register_file`.

### Changed
- Renderer + headless: hardware 4× MSAA (`set_msaa`).
- Refactored the OpenSCAD format parsers into a `scad::import` submodule; gated the
  CSG parity/debug scaffolding behind `#[cfg(test)]` (clean, warning-free build).

### Fixed
- Kernel robustness (curved-boolean paths that previously hung or ground):
  - Bounded the dual-BVH `bvhcast` descent — a degenerate input could explore an
    exponential tree of node pairs and hang; now capped with partial-result bail.
  - Rewrote CDT constraint-edge recovery from ~O(n⁴) to O(crossings·n) via an
    edge→triangle adjacency map, with a linear flip cap.
  - Split-fragment and per-face CDT-size guards keep pathological booleans
    terminating — identically on wasm (no thread to time out).

## [0.0.2] — 2026-07-21

### Added
- `native-codec` feature: pure-Rust, wasm-safe media codecs (no ffmpeg / C bindings).
  - HEVC / H.265 encoder and MP4 mux helpers.
  - VP9 encoder (intra + inter ladder) and WebM mux with alpha (`BlockAdditional`).
  - APNG encoder from RGBA frames.
  - Full GIF89a encoder: local/global/auto palettes, octree & median-cut quantization, Floyd–Steinberg dithering, dirty-rect differencing, transparency, disposal modes, lossy indexing, deferred/adaptive LZW clears, comments, interlacing, and incremental `GifWriter`.
  - Native GIF decoder (`GifDecoder` / `decode_gif`) with disposal compositing, interlace, and structured errors.
- `video` feature: `export_video` frame-sequence export via system ffmpeg; with `native-codec`, `VideoCodec::Gif` streams through `GifWriter`.
- Headless RGBA rendering path and related examples/tests for codec round-trips.

### Changed
- Public re-exports for codec and video APIs when the corresponding features are enabled.
- Web package versions (`web/package.json`, `web/pkg`) aligned to the crate version.
- README updated with table of contents, headless/video/codec quick starts, and feature overview.
- Expanded rustdoc on GIF encode/decode, video export, headless rendering, and `ShaderMaterial`.
- Per-format `export_*` examples, `encode_animation_rgba` / `BrowserCodec`, and browser export demo (`NATIVE_CODEC=1`).
- JS/TS video export API: `encodeVideoFrames`, `exportSceneVideo`, `BrowserVideoFormat` (`web/video-export.js` + shim).

## [0.0.1] — 2026-07-16

### Added
- Initial public crate: three.js–shaped wgpu renderer for native and wasm.
- Scene graph, geometries, PBR materials, lights, post-processing, loaders, controls, helpers.
- Opt-in `mesh-bvh` and `bvh-csg` (three-mesh-bvh / three-bvh-csg parity).
- Web shim (`THREE.*`) and parity tooling.

[0.0.6]: https://github.com/eugenehp/threers/compare/v0.0.5...v0.0.6
[0.0.5]: https://github.com/eugenehp/threers/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/eugenehp/threers/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/eugenehp/threers/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/eugenehp/threers/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/eugenehp/threers/releases/tag/v0.0.1
