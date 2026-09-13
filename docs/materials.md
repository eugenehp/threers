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
