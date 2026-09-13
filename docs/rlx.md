# RLX bridge

Part of the [threers](../README.md) documentation.

## RLX bridge

[RLX](https://crates.io/crates/rlx) is an ML compiler and runtime. The `rlx`
and `rlx-geo` features put it beside the renderer — the renderer is unchanged
either way — and convert between the two crates' data: vertex attributes and
pixels become tensors of a declared shape, tensors become attributes and
textures. Both features are off by default and both build for wasm32 as well as
native.

```toml
# Cargo.toml
threers = { version = "0.0.5", features = ["rlx", "rlx-geo"] }
```

```rust
use threers::rlx::{mesh, preferred_device, ColorGrade, ConvFilter, FitOptions, Kernel3x3};

// A 3×3 convolution, compiled once and run per frame.
let mut sharpen = ConvFilter::new(width, height, Kernel3x3::SHARPEN, preferred_device());
let crisp = sharpen.apply(&frame)?;

// Mesh smoothing as a graph over the [n, 3] positions tensor.
mesh::taubin_smooth(&mut geometry, 12, 0.5, -0.53, preferred_device())?;

// A colour grade *learned* from a reference, by gradient descent.
let report = ColorGrade::fit(&render, &reference, &FitOptions::default(), preferred_device())?;
let graded = report.grade.apply(&other_frame);
```

| Example | Features | What it shows |
|---|---|---|
| `rlx_frame_filter` | `rlx` | A Reinhard tone map as three IR nodes over a rendered frame |
| `rlx_conv_postfx` | `rlx` | Blur / sharpen / Sobel / Laplacian / emboss, plus CPU-vs-GPU timings |
| `rlx_mesh_smooth` | `rlx` | Laplacian against Taubin smoothing, with the volume each one keeps |
| `rlx_fit_grade` | `rlx` | Learns a colour grade from a reference, then transfers it to another shot |
| `rlx_video_grade` | `rlx,video,native-codec` | The compiled grade applied across 60 frames → GIF |
| `rlx_palette` | `rlx,rlx-geo` | k-means palette, posterise, and a Voronoi mosaic snapped to it |
| `rlx_geo_terrain` | `rlx-geo` | Scattered samples → exact Delaunay mesh, plus a Voronoi map |
| `rlx_geo_materials` | `rlx-geo` | Cell textures: albedo, roughness and normals from wall distance |
| `rlx_geo_refine` | `rlx-geo` | Adaptive refinement against a uniform grid at equal vertex budget |
| `rlx_geo_erosion` | `rlx,rlx-geo` | 120 diffusion passes over a terrain, one compiled graph |

```bash
cargo run --release --features rlx --example rlx_fit_grade
cargo run --release --features "rlx,rlx-geo" --example rlx_palette
```

# In the browser

`RLX=1 web/build.sh` (and `RLX_GEO=1`) exports the same work as `rlxConvolve`,
`rlxDiffuse`, `rlxSmoothMesh`, `rlxFitGrade`, `rlxApplyGrade`, `rlxPalette`,
`rlxPosterize`, `geoDelaunay`, `geoHeightfield`, `geoVoronoiLabels`,
`geoVoronoiWallDistance` and `geoNormalMap` — plain typed arrays in and out.

Run them in a worker. `web/rlx-worker.js` is one, and `web/rlx-bench.html`
drives it: every op is tens of milliseconds, which on the main thread is a
dropped frame or twenty. Pixel buffers go in the transfer list rather than
being cloned — a 4K frame is 33 MB, and copying it costs more than filtering
it does.

```js
const worker = new Worker('./rlx-worker.js', { type: 'module' });
worker.postMessage({ id: 1, op: 'convolve',
                     args: { rgba, width, height, kernel: 'sharpen' } },
                   [rgba.buffer]);   // ← transferred, not copied
```

**rlx's GPU backend is available on wasm32 as of the wgpu 30 upgrade.** It used
not to be: rlx was on wgpu 30 while this renderer was on wgpu 0.20, and two
majors in one wasm binary both generate bindings for the same WebGPU interfaces,
which `wasm-bindgen` rejects outright (`duplicate string enums:
GpuAutoLayoutMode`). The Rust compiled — it was the bindgen step that could not
resolve it — so the browser build left rlx's `gpu` feature off. Both are wgpu 30
now, one set of bindings, and `wasm-bindgen` accepts the build. Ask
`rlxDevices()` at startup rather than assuming: whether a given browser then
hands you a working WebGPU adapter is still its own question.

Two numbers from those examples, since both cut against the feature as much as
for it. Adaptive refinement reaches **~9× lower error** than a uniform grid of
the same vertex count — worth having. A 640×640 convolution costs **~10 ms per
frame** through host memory against ~5 ms to render the frame in the first
place — so for effects that belong to the frame, the renderer's own
post-processing chain is still the better tool. What the bridge buys is a
kernel in the same graph as the rest of the pipeline, and one autodiff can see.

# Speed

`preferred_device()` returns the **CPU**
for a reason — the GPU column is what rlx's own priority order would have
picked:

Milliseconds, best of five. Native is an M-series laptop; the browser column is
Chrome 131 running `web/rlx-bench.html`, measured by
`web/scripts/rlx-bench-validate.mjs`. `preferred_device()` returns the **CPU**
for a reason — the GPU column is what rlx's own priority order would pick:

| operation | native CPU | native GPU | Chrome (wasm) |
|---|---|---|---|
| `ConvFilter` gaussian, 512² | 11.3 | 17.6 | 45.3 |
| `Palette::extract`, 8 colours | 29.7 | 67.0 | 34.1 |
| `posterize`, 512² | 22.8 | 47.6 | 53.0 |
| `ColorGrade::fit`, 400 steps | 479 | 29 312 | 481 |
| `Diffusion`, 20 passes, 256² | 3.9 | 15.6 | 6.6 |
| `taubin_smooth`, 8 iterations | 8.9 | 33.3 | 12.5 |
| `voronoi_labels`, 400 sites, 512² | 106 | — | 87 |
| `delaunay`, 5 000 points | 0.9 | — | 8.3 |

Wasm costs 1–4× native across most of the table, and the **fit — the one that
matters — is at parity** (481 ms against 479). `delaunay` is the outlier at 9×,
and that is not the engine: the browser takes rlx-geo's Guibas-Stolfi path
because the Dwyer one calls a clock wasm does not have (see below). Measured on
the same target, that path costs 1.9–2.4× the Dwyer build for identical output.
Run-to-run variance on this machine is up to ~1.5×, and the fit is the noisiest
row.

The GPU column is the finding: every one of these is either small or a chain of
dependent steps, and dispatch latency eats the arithmetic. `fit` is 400 round
trips, hence 61×. Convolution crosses over around four megapixels — at
3840×2160 the GPU takes 373 ms against the CPU's 920 — so pass `Device::Gpu`
explicitly for work that size.

# Why the worker, measured

Five 512² convolutions, driven from the page, counting animation frames:

| | time | frames painted |
|---|---|---|
| on the main thread | 338 ms | **0** |
| in the worker | 341 ms | **22** |

Same work, same wasm, no measurable throughput cost — and the page keeps
painting instead of freezing solid. `node web/scripts/rlx-bench-validate.mjs`
reproduces both columns in headless Chrome.

# Payload

`wasm-bindgen --target web`, bytes, against the same build without the feature:

| build | raw | gzip | brotli |
|---|---|---|---|
| baseline (default features) | 1 368 963 | 395 840 | 300 916 |
| `+rlx-geo` | 1 439 365 | 418 025 | 317 030 |
| `+rlx` | 6 511 319 | 1 476 290 | 976 191 |
| `+rlx,rlx-geo` | 6 583 531 | 1 497 478 | 990 578 |

`rlx-geo` costs **16 KB** brotli — it is a triangulator and a predicate kernel.
`rlx` costs **675 KB** brotli, a 3.2× payload, because it brings a whole ML
compiler: IR, autodiff, fusion, the compile pipeline and a CPU backend. That is
a real decision for a web build, not a rounding error, which is why it is its
own flag and off by default.
