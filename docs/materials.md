# Materials and environments

Part of the [threers](../README.md) documentation.

# Materials: metals, glass, anisotropy, thin film

`PhysicalMaterial` (three.js `MeshPhysicalMaterial`) carries the full extended
PBR stack — clearcoat, transmission + IOR + dispersion, anisotropy with an
in-plane rotation, sheen, iridescence, and Beer-Lambert volume absorption — all
honored by the renderer, plus normal / roughness / metalness / AO / emissive maps.

```bash
cargo run --release --example material_chart        # contact sheet → PNG
cargo run --release --example spacecraft_materials  # interactive scene
cargo run --release --example spacecraft_render     # same scene, headless → PNG
```

In the browser, **`web/examples/materials.html`** renders all of them live via
wasm/wgpu (also procedural — no HDRI or texture downloads). The JS shim accepts
three.js's own option names, so this is portable three.js code:

```js
new THREE.MeshPhysicalMaterial({
    color: 0xffffff, roughness: 0.03,
    transmission: 1.0, ior: 1.5, thickness: 0.6, dispersion: 0.2,
    attenuationColor: 0x66ddaa, attenuationDistance: 0.8,
    transparencyMode: 'refract',
});
// …or start from a preset and tweak:
THREE.MeshPhysicalMaterial.preset('gold_foil', 0, { normalMap: crinkle });
```

Measured-reflectance presets live in `materials::presets` (gold and silver MLI
foil, aluminium, brushed aluminium, titanium, heat-tinted titanium, solar cell,
thermal paint, black kapton, optical glass):

```rust
use threers::materials::presets;

let mut foil = presets::gold_foil();     // F0 (1.000, 0.766, 0.336), metalness 1
foil.normal_map = Some(crinkle);         // the crinkle is what makes it read as foil
let panel = presets::solar_cell();       // dark cells + cover-glass clearcoat
let bell  = presets::anodized_titanium(320.0);  // thin-film hue by thickness (nm)
```

## Glass

A `PhysicalMaterial` with `transmission` is glass. `transparency` picks how
it is drawn:

| `TransparencyMode` | How it works | Use it for |
|---|---|---|
| `Blend` (default) | three.js's transmission. The opaque scene is captured, and each pixel reads it where the refracted ray leaves after `thickness`. | three.js parity |
| `Refract` | The same physics on a sharper capture. The path length is measured from the glass's own back faces, reflections are traced in screen space and fall back to the environment, and dispersion splits the channels. | solid glass that should look its best |
| `Glass` | Thin glass: one layer, drawn in two halves — what is behind is multiplied by the glass's transmittance, then its reflections, highlights and emission are added at full strength. Nothing bends. | windows, bottles, visors |
| `Oit` | Order-independent blending: overlapping layers in any order, shaded as every other mode is. On WebGL2, which cannot blend two attachments differently, it draws as `Glass`. | many overlapping panes |

`examples/glass_modes.rs` renders each mode next to the path tracer, for
clear, tinted, frosted, dispersive and thin-walled glass.

In the three.js shim the mode is `MeshPhysicalMaterial.transparencyMode`
(`'blend'`, `'refract'`, `'glass'` or `'oit'`), a threers extension. The Metal
renderer has no screen-space refraction, so there `Refract` draws as `Blend`
(three.js's transmission); `Oit` is the same weighted-blended OIT as wgpu's,
to the byte (`tests/oit_parity.rs`).

- **Glass behind glass.** `Refract` draws its objects back to front in up to
  `Renderer::set_refraction_layers` groups (2 by default), each refracting a
  capture of everything drawn before it, so glass shows through glass;
  within a group a pixel shows the front-most surface. OIT surfaces sit
  among the groups by depth: one behind a group's glass is composited into
  that group's capture, so the glass refracts it; one in front of all of it
  is composited over it. `tests/glass_layers.rs` checks both.

- **Lights.** Every mode shows a GGX highlight from each directional, point
  and spot light, shadowed where the light is blocked, and reflects
  `scene.environment`. Glass has no diffuse term, so a light's colour
  reaches it only as a highlight. (`Refract`'s pass has no room for the
  shadow maps under WebGL2's sixteen texture units, so a pass just before it
  looks each light's shadow up on the glass and hands them over packed in
  one texture.)
- **Shadows.** Solid glass casts a full shadow. The light it lets through is
  bent into a caustic that only the path tracer draws, so a lighter but
  evenly spread shadow would be wrong. Thin glass (`thickness` 0, or `Glass`
  mode) bends nothing, so it casts a light shadow, tinted by what it
  transmits. It is dithered into the shadow map, and the shadow filter
  smooths it.
- **Thin-walled glass.** `thickness` 0 means a wall with no inside, as in
  three.js. Light goes straight through, in every mode and in the path
  tracer.
- **Tint and absorption.** `color` filters what passes through.
  `attenuation_color` over `attenuation_distance` absorbs along the path
  (Beer–Lambert): measured from the back faces in `Refract`, `thickness` in
  the others. Every mode filters by channel: green glass over a white wall
  shows green, in `Glass` and `Oit` too (their blends multiply what is behind
  by the transmittance per channel; OIT keeps a coloured revealage).
- **Antialiasing.** Every mode multisamples with the rest of the frame.
  `Oit` accumulates into multisampled targets, resolved before the
  composite; under `Refract` the opaque scene multisamples into the capture
  and its depth is resolved for the march, while the glass surface itself
  smooths its silhouettes analytically. WebGL2 cannot read a multisampled
  depth to resolve it, so there the opaque scene's depth is drawn again,
  single-sampled, for the march — every backend multisamples refraction.
  `Renderer::last_frame_multisampled` says what a frame got, and
  `tests/glass_occlusion.rs` checks glass behind an opaque object stays
  hidden through both depth paths. TAA works with all of them.
- **The `Refract` capture.** The screen-space capture is the opaque frame as
  drawn, tone-mapped and encoded. The pass decodes what it reads and adds
  its own light tone-mapped once, so glass does not wash out under ACES.

## Lights and shadows

A frame carries up to eight lights of each kind: directional, point, spot,
hemisphere and rect-area (four directional and spot lights on the Metal
renderer). Further lights of a kind are left out. Every directional, point
and spot light with `cast_shadow` set casts shadows, up to sixteen at once.

`light.shadow` (`ShadowSettings`, three.js's `LightShadow`) controls them:

| Field | |
|---|---|
| `map_size` | The shadow map's side in texels (directional and spot; the largest any light of the kind asks for). Point faces are 512. |
| `camera_size`, `camera_near`, `camera_far` | The directional shadow frustum: a fixed cube on the origin. Tighter is sharper. |
| `bias`, `normal_bias` | Depth and surface-normal offsets against shadow acne. |
| `radius` | How soft a filtered edge is: the radius, in texels, it is filtered over. 1 (the default) anti-aliases, larger values give a penumbra, 0 is one comparison. |
| `cascades`, `cascade_distance`, `cascade_split` | Cascaded shadow maps for a directional light: 2–4 maps over the camera's view out to `cascade_distance`, fine near the camera and coarse far off, in place of the fixed cube. |
| `light_size` | The light's size for PCSS: world units for point and spot lights, radians for a directional one. |
| `filter` | This light's filter, in place of the renderer's. |

**Filters.** `Renderer::set_shadow_type` (three.js's `shadowMap.type`)
picks how every map is read:

| `ShadowType` | three.js | |
|---|---|---|
| `Basic` | `BasicShadowMap` | One comparison: hard, aliased edges. |
| `Pcf` (default) | `PCFShadowMap` | Twelve comparisons over a Vogel disc `radius` texels wide, turned per point in the world so a wide penumbra shows no banding and does not crawl. |
| `PcfSoft` | `PCFSoftShadowMap` | Twenty-four over twice the disc, at least two texels. |
| `Pcss` | `VSMShadowMap` | Percentage-closer soft shadows: the penumbra widens with the gap between receiver and caster, by `light_size` — sharp where an object meets the ground, soft far off. |

PCSS's blocker search compares rather than reading depths, because WebGL2
can only read a depth texture through a comparison sampler: compares at four
depths give the fraction of the search disc blocked nearer than each, and the
mean blocker depth comes from integrating that. `tests/shadow_types.rs`
checks the filters' order of edge width, PCSS's hardening toward contact,
and that four cascades put a small shadow near the camera where a tightly
fitted map does when one map stretched over the whole reach loses it.

**Storage.** Each casting light owns a shadow record (its kind, layers,
matrices, biases and filter) and names it in its own colour's alpha, so
every light loop asks one lookup, and a fragment works out every record's
shadow once and the lights read it from there. Each kind's maps are one depth
array, a layer per light, a cascade each, six per point light (looked up by
direction with the face chosen by hand: WebGL2 has no cube arrays). Several
lights of a kind share a budget of two 4096² maps.

**Cost.** A shadow pass is redrawn only when what it shows changed — its
light, or a caster — so a camera-following cascade does not redraw a still
point light's six faces, and a still scene redraws nothing. Each pass draws
only the casters inside its light's frustum. `Renderer::shadow_stats` counts
both.

In the three.js shim, `renderer.shadowMap.type`, `light.shadow.bias`,
`.normalBias`, `.radius` and `.mapSize` reach the renderer as three.js has
them (bias negative); from `WebLight`, `setShadowParams`, `setShadowMapSize`,
`setShadowCascades` and `setShadowSoftness`.

# HDR environments and tone mapping

An 8-bit environment cannot hold a sun: capped at 1.0, a solar disc can only be
made to *read* bright by making it physically huge, and PMREM then smears it
across tens of degrees so every metal reflects the same grey wash. Load a real
one instead:

```rust
use threers::{loaders::HdrLoader, PmremGenerator, ToneMapping};

let (pixels, w, h) = HdrLoader::parse_f32(&bytes)?;      // linear f32, unclipped
let cube = PmremGenerator::from_equirect_f32(&pixels, w, h, 256);
scene.environment = Some(Arc::new(PmremGenerator::generate_pmrem(&cube, 256)));

// Without this, anything above 1.0 hard-clips to white and loses its hue.
renderer.set_tone_mapping(ToneMapping::AcesFilmic, 1.0);
```

Tone mapping is **off by default** so existing scenes render unchanged.

Two things worth knowing before you reach for metals:

- **Metals need `scene.environment`.** At `metalness = 1.0` the BRDF has no
  diffuse term, so with no environment a gold sphere renders near-black no matter
  how many lights you add. Prefilter one with `PmremGenerator::generate_pmrem`;
  that mip chain is also what makes `roughness` mean anything for reflections.
- **Pick a non-sRGB offscreen format.** The mesh shader does its own
  linear→sRGB encode, so a `HeadlessRenderer` left on the default
  `Rgba8UnormSrgb` target encodes twice and washes the image out (mid-greys land
  ~2.5× too bright). Use `.color_format(wgpu::TextureFormat::Rgba8Unorm)`, as the
  examples above do.

For a *brushed* look, `anisotropy_rotation` aims the streak in UV tangent space;
the tangent frame is derived from UV screen-space derivatives, so it needs a mesh
with UVs (without them the frame falls back to a view-dependent one and the
rotation is arbitrary).
