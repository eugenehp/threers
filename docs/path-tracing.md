# Path tracing

Part of the [threers](../README.md) documentation.

# Path tracing (`raytrace`)

`Renderer` draws a scene the way a GPU pipeline does: project the triangles,
shade each fragment from a fixed set of lights, approximate everything else.
Those approximations are what make it run at 60 fps and what make it wrong in
familiar ways — light does not bounce, a mirror cannot show what is behind the
camera, and a glass ball does not bend the room.

`RaytraceRenderer` answers the same question by simulating light transport. It
takes the **same `Scene`** — same materials, same lights, nothing authored twice.

```bash
cargo run --release --example path_trace     --features raytrace,parallel   # Cornell box, CPU
cargo run --release --example path_trace_gpu --features raytrace,parallel   # GPU, timed against the CPU
cargo run --release --example hdri_lighting  --features raytrace,parallel   # image-based lighting
```

```rust
use threers::raytrace::{RaytraceRenderer, RaytraceSettings};

let mut renderer = RaytraceRenderer::new(1920, 1080);
renderer.set_settings(RaytraceSettings::default().with_samples(256));
let rgba = renderer.render_to_rgba(&mut scene, &camera);   // same layout as HeadlessRenderer
std::fs::write("out.png", threers::encode_png(1920, 1080, &rgba))?;
```

Progressive, so a preview refines instead of appearing all at once — the scene is
flattened and its BVH built once, then samples are added to the same film:

```rust
renderer.prepare(&mut scene, &camera);
for _ in 0..20 {
    renderer.accumulate(8)?;
    let preview = renderer.resolve_rgba();   // sharper each time
}
```

**Backends.** `CpuBackend` is the reference — portable, wasm32 included, and the
definition of correct for the module. `gpu::GpuBackend` runs the same integrator
as a wgpu compute kernel over the same scene, and is selected by swapping one
argument:

```rust
use threers::raytrace::gpu::GpuBackend;
let backend = GpuBackend::headless()?;               // or ::with_device(device, queue)
let mut renderer = RaytraceRenderer::with_backend(1920, 1080, Box::new(backend));
```

The kernel is packed for throughput rather than convenience: the whole scene goes
into four storage buffers plus one texture atlas (WebGPU's baseline guarantees
only four storage buffers per stage, and bindless is a native-only
wgpu feature this kernel does not assume), so there is
one bind group for any scene; samples are traced in batches per dispatch, because
a single dispatch that runs for seconds trips the GPU watchdog on every platform;
the film accumulates on the device and is read back once, not once per batch; and
one invocation owns one pixel, so there are no atomics anywhere.

On an Apple M-series integrated GPU the `path_trace_gpu` example runs about
**7× faster** than the rayon-parallel CPU backend at identical settings, and the
two frames agree to four decimal places — which is what the CPU↔GPU test in the
suite checks, along with a white-furnace test on each.

**Image-based lighting.** The usual way to light a realistic render is one
HDRI and no lights at all — `hdri_lighting` does exactly that: no
`DirectionalLight`, no `AmbientLight`, every shadow and highlight and bit of
fill coming out of `scene.environment`. Point it at a real `.hdr` with
`HDRI=studio.hdr`, or let it synthesise a sky with a sun four orders of
magnitude brighter than the blue around it.

```rust
let (pixels, w, h) = HdrLoader::parse_f32(&std::fs::read("studio.hdr")?)?;
scene.environment = Some(Arc::new(PmremGenerator::from_equirect_f32(&pixels, w, h, 256)));
settings.background = BackgroundMode::Environment;   // lit by it *and* seen behind
```

That needs three things to be right, and an HDRI is unforgiving about all
three: the full dynamic range has to reach the integrator (the 8-bit copy
`CubeTexture` keeps for display is Reinhard-compressed, and integrating it caps
the sky at about 1.0), the sun has to be importance-sampled, and the cube faces
have to be read in the orientation they were written in.

**Sampling.** Three strategies, all combined by the power heuristic so none is
trusted where it is poor: the BSDF's own lobes; the lights, including emissive
geometry by an area×luminance distribution; and the environment, by a 256×128
density built over its own brightness.

Underneath all of them, the *numbers* are stratified rather than independent.
Nearly every decision a path makes is two-dimensional — a point on a light, a
direction off a BSDF, a point on the lens — and each one draws a pair from an
Owen-scrambled Sobol (0,2)-sequence, so sixteen samples of a pixel cover the
unit square evenly instead of clumping the way sixteen independent draws do.
Dimensions are allocated in a fixed block per bounce so that the same decision
draws the same dimension on every sample, which is what makes the stratification
worth anything. Measured against a converged reference: 1.5× lower RMS on a
plain area-light scene, 3× on one with an environment, a sun and an emitter —
the latter being roughly nine times the effective samples. That last one is not a refinement — an
HDRI puts most of its energy into a sun about 6·10⁻⁵ of the sphere across, which
a cosine-weighted direction finds roughly once in twenty thousand samples, so
without it an HDRI-lit render is white speckle that does not resolve at any
sample count you would wait for.

**Denoising.** Cycles uses OpenImageDenoise — a pretrained U-Net — which is
not something this crate can ship. What it *can* take from Cycles is the part
that makes any guided denoiser work, and that is how the guide passes are
built: the albedo and normal are not "whatever the first hit was". A mirror's
own colour and normal describe the mirror, not the image in it, and a glass
ball's describe neither the glass nor the room behind it. So the guides follow
the path through specular and transmissive surfaces, carrying their tint, until
they reach something rough enough to describe — Cycles' scheme, thresholds
included.

The reconstruction filter is edge-avoiding À-Trous, but the colour threshold is
the pixel's own accumulated variance rather than a constant: two estimates of
the same value differ by about their combined standard deviation, so dividing
by it asks "is this difference more than noise?" instead of comparing against a
number that is wrong at every sample count but one. Where a pixel has already
converged the filter stands down entirely — there is nothing left to remove and
a smooth gradient is all it could damage.

And the filter's parameters are **fitted, not guessed**. `denoise_fit` renders
a handful of scenes twice — 32 samples and 8192 — and searches the four widths
against that ground truth, judged on a held-out interior:

```bash
cargo run --release --example denoise_fit --features raytrace,parallel
```

Which was worth doing. The hand-picked values it replaced were worse on every
scene tried, and on a smooth area-lit one they were worse than not denoising at
all. Relative error on the held-out scene went 0.096 → 0.084, and on glass
beside rough metal 0.182 → 0.157. `DenoiseParams::fit` is public, so the same
can be done against your own content.

**A trained denoiser** (`--features learned-denoise`) sits beside the fitted
À-Trous filter. It is a U-Net over the same guides — colour, albedo, normal —
trained on pairs this renderer generates itself, so no corpus is collected and
no licence attaches to the result.

Measured against **the OpenImageDenoise library inside Blender.app**, driven at
the settings Cycles sets in `intern/cycles/integrator/denoiser_oidn_base.cpp`
(filter `RT`, hdr on, srgb off, quality `HIGH`) — so the comparison is with what
Blender actually ships, not an approximation of it:

| scored on | ours | Blender's OIDN | |
|---|---|---|---|
| renders of this repo's NEMA 17 and printer | **0.03545** (5.51×) | 0.03720 (5.25×) | 4.7% ahead |
| the furnace test | **0.0125** | 0.0132 | 5.2% ahead |
| Cornell box | 0.0558 | **0.0492** | 13.3% behind |
| Veach MIS | 0.1016 | **0.0919** | 10.6% behind |

Both directions are worth stating. On real machined geometry it is ahead of
what Cycles ships. On the classic demonstration scenes it is behind, and the
per-scene split says why: the training corpus had no tight saturated box and no
frame holding a pinpoint light beside a broad one. Those regimes are in the
generator now.

The **furnace test** is the row to keep. A grey sphere in a uniform emissive
environment must converge to exactly the environment's radiance — the
multiple-scattering series sums to one — so the correct answer is known in
closed form and any structure a denoiser draws there is *provably invented*.
Being ahead on that one says the filter hallucinates less, which is the property
a renderer actually wants and which none of the other numbers measure.

The forward pass is `raytrace::denoise_net`: plain Rust, no dependencies,
`wasm32` included. It is checked against the implementation that trained it
rather than by eye — worst per-pixel disagreement **0.000000**
(`examples/denoise_parity.rs`). Training lives in `rlx-denoise`, which is
GPL-3.0-only and is not linked by this crate.

**In a browser.** `wasm_denoise` splits the work into `plan` / `extract` /
`run` / `merge`, where only `run` is expensive and belongs on a worker; tiles
share no state, so N workers produce the same frame as one
(`tiling_a_frame_matches_denoising_it_whole` asserts exactly that). The module
docs carry the worker wiring.

Measured: ~0.7 s a 128×128 tile at the default widths on one core, ~1.1 s at
`wide`. Four times the parameters costs 1.6× the time, so the kernel is
memory-bound, not compute-bound. A 1080p frame is ~135 tiles, which across
eight workers is **12–19 seconds** — enough to denoise a finished render in the
browser, or to sharpen a progressive one between passes, and **not** an
interactive viewport. Getting there means putting the convolutions on the GPU,
which the crate already has the wgpu plumbing for.

**Progressive denoising.** `render_progressive` traces in batches and denoises
the accumulated film after each, so a viewport shows a usable image from the
first sample rather than noise until the end — which is what Cycles does, and
Blender's default is to denoise from sample 1. Worth knowing what that buys: on
this content the denoiser at ~4 spp matches a raw render at ~110 spp, about 29×
fewer samples.

**Adaptive sampling** tracks each pixel's own standard error and stops it once
that falls below `adaptive_threshold` (1 % by default). Noise falls as
`1/sqrt(n)`, so the last halving of the error costs three quarters of the
render, and most of an image gets there long before its worst pixels do —
typically 10–40 % of the sample budget goes unspent for an image that is
indistinguishable. It does not bias anything: a pixel that stops early is still
the mean of its own samples, and the film divides by each pixel's own count.
The decision reads only what the pixel has accumulated, so an adaptive render is
still independent of how it was batched. On the CPU the saved samples are saved
time; on the GPU a workgroup runs until its slowest pixel, so the saving only
lands where whole tiles converge. `RaytraceSettings::final_quality()` turns it
off.

**What is simulated:** multi-bounce diffuse and glossy GI; the principled BSDF
(Lambert + anisotropic GGX + rough dielectric transmission + clearcoat) with
Kulla–Conty energy compensation; emissive geometry as area lights;
`Directional` / `Point` / `Spot` / `RectArea` lights, optionally with an angular
or spherical size for soft shadows; `Ambient` and `Hemisphere` lights and
`scene.environment` as importance-sampled image-based lighting; alpha cutouts
whose shadows are cutout-shaped; Beer–Lambert absorption inside transmissive
solids; depth of field.

**What is not:** participating media, subsurface scattering, spectral dispersion,
and caustics through next-event estimation. Materials with no physical reading —
`MeshNormalMaterial`, `MeshToonMaterial`, `ShaderMaterial` — are mapped to the
nearest surface that has one, and every such choice is listed in
`renderer.report().approximated` rather than made silently.
