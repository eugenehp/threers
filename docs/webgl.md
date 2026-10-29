# WebGL2

Part of the [threers](../README.md) documentation.

WebGPU is not everywhere yet. The `webgl` feature compiles wgpu's GL backend in
as well, so a browser without WebGPU gets WebGL2 instead of a blank page.

```toml
threers = { version = "0.0.9", features = ["webgl"] }
```

WebGPU is still preferred wherever it exists — turning this on cannot downgrade
a canvas that was already working.

**It is not free**, and the honest figure is the compressed one, since that is
what a browser downloads. Through the real build pipeline, `mini`, brotli:

| | on the wire | uncompressed |
|---|---|---|
| `mini` | **0.30 MiB** | 1.03 MiB |
| `mini` + `webgl` | **0.785 MiB** | 2.86 MiB |

So about **half a megabyte** of transfer. The GL backend brings glow and naga's
GLSL translator — a shader compiler, shipped to the browser — and a WebGPU-only
deployment never executes a line of it. It is therefore off by default
everywhere, including in `wasm-full`, and the web build takes it as a flag that
composes with any variant:

```sh
WEBGL=1 VARIANT=full ./web/build.sh
```

Earlier versions of this page quoted the *uncompressed* delta, 3.85 MiB, as
though it were the download. It is not; compression recovers most of it, because
a shader translator is repetitive code.

```js
const renderer = await WebRenderer.new(canvas);
renderer.backend();          // "webgpu" | "gl"
renderer.supportsCompute();  // false on WebGL2
```

## State of it

![One scene on wgpu, Metal, Chrome WebGPU and Chrome WebGL2](images/backend_parity.jpg)

*The same scene four ways: wgpu natively and threers' Metal renderer (top),
then Chrome on WebGPU and on WebGL2 (bottom), from
`cargo run --release --features metal --example backend_parity` and
`web/examples/parity.html` (`?gl=1` for WebGL2). The mean difference from
the wgpu frame is 0.01 of a level (Metal), 0.09 (WebGPU) and 0.003 (WebGL2).*

**It renders.** `tests/webgl.rs` runs in headless Chrome and checks, against a
real WebGL2 device, that a GL adapter is offered, that the device reports what
WebGL2 actually has, that the compute passes decline rather than silently doing
nothing, that a raw render pass plus readback produces exactly the pixels asked
for, that the renderer compiles every one of its own pipelines and runs a frame,
and that **a lit mesh, a skinned mesh and a refracting one drawn to a render
target come back with their colours in them**. Those last assertions are on read-back pixels rather
than on the absence of a panic: a draw that binds nothing and writes nothing
raises no error at all.

The **fluid simulation** (`fluid`) is the one substantial part of this crate
that is *complete* here rather than reduced. Bloom and ambient occlusion have
raster paths standing in for compute ones; the fluid has no compute stage to
stand in for, because it was always nineteen fullscreen fragment passes. What it
needs instead is a filterable half-float render target, which WebGL2 has only
with `EXT_color_buffer_float` — so it probes the adapter and widens one channel
count at a time rather than assuming. `tests/fluid_webgl.rs` runs the whole
frame in headless Chrome and checks on read-back pixels that it drew, and drew
the right way up. See [Fluid](fluid.md).

A richer scene is covered too — a textured material, a shadow-casting light, a
transparent surface on the sorted path, and a post-processing pass over the
result — because the narrow tests each exercise one pipeline and cannot find what
a combination does. Nothing in it is missing on WebGL2.

One invariant underpins all of it and is worth knowing before changing a shader:
**neither shader module may declare a storage buffer.** One `var<storage>` stops
the whole module compiling on a GLSL backend whether or not any entry point reads
it and whether or not the layout offers it — which is why skinning and the light
arrays are fixed-size uniform blocks. `the_shader_modules_declare_no_storage_buffers`
fails if one comes back.

Seven things stood between the backend and that frame, every one found by running
it rather than by reading a specification. Three were then fixed properly rather
than worked around, so the temporal pass and the depth-driven post effects run
there too:

| Problem | Resolution |
|---|---|
| `ShaderMaterial`'s three storage buffers, against a limit of zero | Left out of the module, the layout and the bind group together — all three have to agree. The same data is uploaded as an `Rgba32Float` texture instead and read through `s0(i)`/`s1(i)`/`s2(i)`, which are defined on every backend |
| `textureNumLevels` for the PMREM chain length; no GLSL ES 3.0 equivalent, and not recoverable from `textureDimensions` because a PMREM chain is shorter than a full one | Carried in `env_map_params.z`, which already held exactly this for the CubeUV path |
| `textureLoad` from a depth texture, in the refraction pass, the temporal resolve and post-processing — GLSL has no `texelFetch` for a depth sampler, and naga compiles *both* arms of a branch whether either is taken | All three bind their depth as unfilterable **float** rather than `texture_depth_2d`, which a depth format permits and which lowers to `texelFetch`. The temporal resolve and the depth-driven post effects work on WebGL2 as a result, at no cost on any backend |
| `INDEPENDENT_BLEND`, which order-independent transparency needs for its two differently-blended attachments | Pass skipped; `Oit` glass draws as `Glass` (one layer, tinted and lit) |
| A multisampled texture read in a shader (GLSL ES 3.0 has no `sampler2DMS`), which resolving the opaque depth for the refraction pass needs | The opaque scene's depth is drawn again, single-sampled, for the glass to test against and march; the frame stays multisampled (`Renderer::force_depth_replay` takes this path on any backend, for tests) |
| **Seventeen texture units against sixteen** — a panic *inside* wgpu-hal rather than a validation error, so it cannot be guarded against. See below | The groups were split by entry point: the refraction pass gets its own group 3 with the capture in place of the shadow maps. Its glass's shadows reach it packed into one integer texture, written by a pass over the same surfaces with the lit layout (`fs_glass_shadow`) |
| Cube map arrays, for a shadow cube per point light | Every kind's shadow maps are one 2D depth array, six layers a point light, the cube face chosen by hand in the shader. (The GL backend makes a square texture with a multiple of six layers a cube map, and one with a single layer a plain 2D texture; neither binds as an array, so the arrays never have those counts) |
| `VIEW_FORMATS`, which render targets declare so a later pass can reinterpret their bytes | Not declared where unavailable; such a view cannot then be taken |

### Sixteen per stage is not sixteen in total

Worth spelling out, because it is a layout WebGPU validates and WebGL2 still
cannot run.

The standard material binds **eight** sampled textures in the fragment stage from
the material group and **eight** from the environment group — the environment
cube, the CubeUV atlas, three shadow maps, and the refraction pass's capture with
its two depth buffers. Sixteen exactly, which is the per-stage floor, so WebGPU
accepts it. There is also a displacement map bound in the **vertex** stage, and
WebGL2 has no per-stage pool: wgpu-hal's GL backend assigns units across the
whole pipeline from one array of sixteen. Seventeen, and the last has nowhere to
go.

Nothing is over-declared per stage, so there is no validation error — only
`index out of bounds: the len is 16 but the index is 16`, from inside the draw.

The fix is to split the groups by entry point rather than to drop a feature. Note
what makes it possible: units are assigned per *layout* entry, not per use — the
draw that panicked sampled none of the textures that overflowed it — so a
pipeline only has to **declare** less, not do less.

`fs_ss_glass` samples no shadow map (its material parameters come from uniforms,
and it reads only the environment and the capture), so the refraction pass gets a
group 3 with the capture where the three shadow maps would be. The main pipelines
get one with the shadow maps and no capture. Both come to fourteen, and both fit.

The same move then fixed shader materials, which were unavailable on **baseline
WebGPU** entirely — their four user textures on top of everything else overflowed
the same sixteen, so the layout could not even be created and the feature was
absent wherever a device reports the floor, Chrome included.

A `ShaderMaterial` carries a fragment source, a uniform block and its own
textures; it has no standard maps and no displacement map. So the eight material
textures in group 2 are *definitionally* unused by one, and a custom pipeline now
gets a group 2 holding only the sampler and the displacement slot — the two the
shared vertex entry needs whether or not the flag selecting them is set, because
naga counts the use and not the branch. That takes the custom layout from
seventeen fragment textures to nine.

### Two bugs that were never WebGL-specific

Both latent until something tried a second canvas or a read-back off WebGPU:

- **A device cannot be paired with a surface from a different instance.** The
  device is cached per tab and the backend discovery makes a fresh instance per
  attempt, so the winning instance has to be kept with it. The symptom is
  `Cannot get non-existent resource DeviceId`, which says nothing about the
  cause. On WebGL2 the sharing cannot happen at all — a WebGL2 context *belongs*
  to its canvas, so each canvas gets its own instance, surface and device.
  Reusing one gets as far as `Surface::configure` and fails with
  `Surface does not support the adapter's queue family`.
- **Read-back blocked the event loop.** `PollType::wait_indefinitely()` blocks
  the calling thread, which in a browser is the thread that would deliver the
  mapping callback — so the wait could never end. It went unnoticed because
  WebGPU's wgpu backend makes `poll` a no-op and resolves through a JS promise;
  on WebGL2, which goes through wgpu-core, polling really waits and
  `readRenderTargetPixels` hung the page. It now polls and yields.

## What WebGL2 cannot do, and what happens instead

WebGL2 has no compute stage and no storage buffers. wgpu does not refuse such a
device — it reports the shortfall by zeroing the matching limits, so code that
assumes otherwise gets a pipeline that was never created and a frame that
quietly misses a step. [`renderer::caps`](../src/renderer/caps.rs) asks instead
of assuming:

| | WebGL2 | What happens |
|---|---|---|
| Compute shaders | no | Nothing on the browser path needs one any more — see the two rows below |
| Bloom | **yes** | Threshold and downsample, then a separable blur in two passes — none of it needs a compute stage, so on a device without one the stages are fullscreen draws instead. Held against the compute path on a device with both, agreeing to the last half-float bit |
| Ambient occlusion | **yes** | Same route: it reads a depth texture and writes one value per pixel. Held against the compute path the same way, agreeing to within 1e-5 |
| Temporal anti-aliasing | **yes** | Its depth slot is an unfilterable float texture, so the resolve compiles |
| Depth-driven post effects | **yes** | The same change to the post-processing depth binding; the hardware depth buffer is read directly rather than through a copy |
| Screen-space refraction | **yes** | The pass has its own group 3 — the screen-space capture where the shadow maps used to be, and one integer texture carrying the glass's shadows — inside sixteen texture units. A refracting draw is tested on pixels. Multisampled too: the depth it marches is the opaque scene's drawn again single-sampled, since a multisampled one cannot be resolved by a shader here. Overlapping glass composites in layers (`setRefractionLayers`) |
| Storage buffers | no | `ShaderMaterial`'s three storage bindings are left out of the module, the layout and the bind group together, so a shader reading `u_s0` directly will not compile. Read them through `s0(i)`/`s1(i)`/`s2(i)` instead: those are defined everywhere, and where there are no storage buffers the same data arrives as an `Rgba32Float` texture read with `textureLoad`. `ShaderMaterial::validate(false)` checks a shader against this backend without a browser, and `THREE.ShaderMaterial` now reaches it from JavaScript |
| Skinning | **yes** | Bone matrices were moved out of a storage buffer into a fixed 128-matrix uniform block (8 KB, inside WebGL2's 16 KB floor). A skinned draw is tested on pixels |
| Order-independent transparency | no | Needs `INDEPENDENT_BLEND`: it accumulates weighted colour and coverage into two attachments with *different* blend equations, and WebGL2 blends every attachment alike. This is a hardware capability rather than a shader shape, so there is nothing to port — `Oit` glass draws as `Glass` instead, one layer deep, tinted, lit and shadowed |
| GPU path tracing, lattice solve | no | Behind `raytrace` / lattice features, both compute-only |
| The packed post chain | n/a | Compute-only, and not a browser concern: it exists for `HeadlessRenderer`'s video read-back, which is the native path. An earlier version of this page listed it as a WebGL2 gap, which was wrong |

## Two things that are not obvious

**One wgpu instance per backend.** On wasm, `wgpu::Instance::new` commits to a
single context implementation. Asking for `BROWSER_WEBGPU | GL` does not mean
"try both" — you get the WebGPU context, which contains no GL whatever the Cargo
features say, and on a browser without WebGPU the adapter request then fails
with `gl support not compiled in`. That message is misleading: the backend *is*
compiled in, it is simply not in the instance you made. Each backend therefore
gets its own instance, tried in order.

**The canvas must not be touched until a backend is settled.** A canvas hands
out exactly one kind of context for its whole life. Once `getContext("webgpu")`
has been called, `getContext("webgl2")` on that canvas returns null forever — so
creating a surface to *test* WebGPU destroys the fallback it was testing for, and
the GL attempt fails with `canvas already in use`. WebGPU can enumerate adapters
with no surface at all, so it is asked first and the surface is created only once
an adapter has answered. GL is the other way round: a WebGL2 context *is* the
canvas, so there the surface has to exist before there is an adapter to ask
about.

## Build size

The `release-wasm` profile in `Cargo.toml` is what the web build uses — not
`release`, which also governs the native build, where this renderer and the path
tracer want speed and `opt-level = "z"` would be a bad trade.

Measured on `mini` + `webgl`, after `wasm-bindgen`, as the browser receives it:

| build | raw | gzip -9 | brotli -q 11 |
|---|---|---|---|
| default `release` | 5.41 MiB | 1.567 MiB | 1.089 MiB |
| `release-wasm` | **2.86 MiB** | 1.001 MiB | **0.785 MiB** |
| `release-wasm` + `wasm-opt -Oz` | 2.61 MiB | 1.041 MiB | 0.812 MiB |

Three things worth knowing, all of them measured rather than assumed:

**`wasm-opt` is a trade, not a free win.** Measured on the same artifact:

| | raw | brotli | V8 compile |
|---|---|---|---|
| without | 2.86 MiB | **0.785 MiB** | 6.7 ms |
| with `-Oz` | **2.61 MiB** | 0.812 MiB | **5.1 ms** |

Nine per cent off the file and a quarter off the browser's compile time, against
twenty-seven kilobytes more to download — the transfer goes *up* consistently
across gzip, brotli and zstd, because link-time optimisation has already
collapsed the duplication the compressor was exploiting and wasm-opt's further
rewriting leaves less of it. (On a build without LTO the sign flips and wasm-opt
helps, which is presumably how it earned its reputation.) Which side wins depends
on the connection: on a slow link the extra bytes cost more than the compile
saves. It runs by default, because the file size and the compile time are the
parts this project controls; `WASM_OPT=0` skips it when transfer is what matters.

Stripping alone buys nothing — `strip = true` and wasm-bindgen have already
removed the debug and producer sections, so `--strip-debug --strip-producers`
leaves the file a few hundred bytes *larger* from re-encoding.

**`opt-level = "z"` beats `"s"` here**, 2.86 MiB against 3.17. Not a given: `"z"`
disables loop vectorisation and often costs more than it saves.

**Brotli is the compressor to serve.** Against this artifact, brotli -q 11 gives
0.785 MiB, gzip -9 gives 1.001, and zstd at `--ultra -22` gives 0.830 — so zstd
loses despite its reputation, and a larger brotli window buys nothing because the
file is already inside the default one. The build writes `.br` and `.gz` beside
the wasm so a static host can serve them with `Content-Encoding` rather than
compressing per request.

**wasm-opt is part of the toolchain, not an ambient convenience.** It is taken
from `PATH` if present and otherwise fetched at a pinned version through `npx -p
binaryen@132`, which needs no admin rights and installs nothing system-wide —
the same route `scripts/wasm-browser-test.sh` uses for Chrome. `WASM_OPT_BIN`
overrides the binary; `WASM_OPT=0` skips the pass. Its `--enable-*` flags are not
optional: wasm-bindgen's output uses features wasm-opt will not assume, and
without them it refuses the input with a validation error about bulk memory.

`panic = "immediate-abort"` was tried and rejected: it would take another slice
off, but it needs `cargo-features` at the top of the manifest, which makes the
whole crate nightly-only.

## Testing

```sh
FEATURES=webgl TEST=webgl scripts/wasm-browser-test.sh
```

The script launches Chrome with ANGLE over SwiftShader and *without*
`--disable-gpu`. That flag switches off WebGL as well as WebGPU, which makes a
missing adapter look like "this browser has no WebGL2" rather than like a flag;
recent Chrome also declines to use SwiftShader for WebGL unless told that is
acceptable. The result is a browser with no WebGPU and a real WebGL2 context,
which is exactly the case this feature exists for.
