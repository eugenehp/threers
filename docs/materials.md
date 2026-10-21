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

![The four filters on one scene: Basic, PCF, soft PCF and PCSS](images/shadow_types.jpg)

*`cargo run --release --example shadow_types`: `Basic` and `Pcf` above,
`PcfSoft` and `Pcss` below. PCSS is as sharp as `Basic` where the pole and
the ball touch the floor and softer than `PcfSoft` at the far ends of their
shadows.*

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

## Fur

![Shell fur: a short drooping coat and long curly tufts](images/fur.jpg)

`threers::fur` grows shell fur on any mesh — the real-time technique for
coats, plush, carpet and moss. The mesh is drawn `shells + 1` times in one
instanced draw: the skin, then copies pushed out along the normal, drooped by
gravity, each kept only where a strand crosses it. Strands are scattered over
the UVs (one per cell of a grid fitted to the UV stretch, so they come out
round in world space), each with its own length, taper, curl and clump, and
lit along their tangent (Kajiya-Kay) with shadows from every shadow-casting
light.

```rust
use threers::fur::{fur_object, FurGroom, FurMaterial};

let coat = FurMaterial {
    groom: FurGroom { length: 0.22, density: 140.0, clumping: 0.4, ..Default::default() },
    root_color: Color::from_hex(0x2a1a0e),
    tip_color: Color::from_hex(0xd6a86a),
    shells: 40,
    ..Default::default()
};
scene.add(fur_object(SphereGeometry::new(1.0, 96, 48), &coat));
```

| `FurGroom` | |
|---|---|
| `length`, `density` | Strand length (world units) and strands per UV unit along each axis |
| `thickness`, `taper` | Root radius (fraction of a cell) and tip radius (fraction of the root) |
| `gravity` | Droop at the tips, world units per unit of length |
| `curl`, `clumping`, `clump_size` | Turns per strand; how far tips gather into tufts, and how big |
| `length_variation`, `seed` | Per-strand length spread; a different scatter |

`FurMaterial` adds the shell count, root and tip colours, root `occlusion` and
the highlight (`specular`, `specular_exponent`, `specular_shift`). A
geometry's vertex colours paint the groom: RGB tints, alpha scales length (0
is bald).

**Where strands grow.** `FurPlacement::Uv` (the default) roots them on a grid
over the UVs, fitted to the UV stretch so they come out round, and tiling
across a UV seam — but following the UV layout, poles included.
`FurPlacement::Surface { cell }` scatters them over the surface in object
space instead: no UVs needed, no seams or poles, for meshes that do not
deform.

**Motion.** Gravity is a world-space droop, so it stays down however the
object turns. `wind` comes in gusts that travel across the coat downwind
(`wind_frequency` gusts a second), and `flutter` gives each strand its own
small sway. A `FurAnimator` drives the clock and adds inertia: fur trails a
moving object and springs back when it stops (`FurDynamics`: stiffness,
damping, response, the largest swing). It also grows fur on a skinned mesh
(`grow_on_skinned`), posing it with the skeleton every frame and giving every
vertex its own spring, so a turning head or a swinging tail drags its coat.

```rust
let mut animator = FurAnimator::new();
animator.track(ball, &coat);                                // rigid: one spring
let tail = animator.grow_on_skinned(&mut scene, skinned_tail, &coat); // per vertex
// each frame, after moving things:
animator.update(&mut scene, time, dt);
```

**Where it runs.** WebGPU, WebGL2 and native wgpu (a built-in shader
material), and the Metal renderer (a port of the same shader, which draws
what wgpu draws). From JavaScript: `new WebFurMaterial()`, its setters,
`scene.addFur(geometry, fur)` and `scene.stepFur(time, dt)`; in the three.js
shim, `new THREE.Fur(geometry, { length, density, rootColor, tipColor, wind,
flutter, stiffness, placement: 'surface', … })`, animated by the renderer at
every `render`. `cargo run --release --example fur` renders the picture
above.

## Strand hair

![Strand hair: a head of long hair under gravity, and a curly mane](images/hair.jpg)

For manes, long hair and close-ups, where shells run out, `threers::fur::strands`
grows real curves from the same `FurGroom` and shades them with the same
`FurMaterial`:

```rust
use threers::fur::strands::{hair_object, StrandOptions};

let options = StrandOptions { count: 30_000, segments: 12, width: 0.004 };
scene.add(hair_object(&scalp, &hair, &options));
```

Roots are spread evenly by area from `groom.seed`; each strand grows out along
the surface normal, curls (`curl`), shortens at random (`length_variation`)
and gathers into clumps around guide strands (`clumping`, `clump_size`).
Vertex colours paint it as they paint fur: alpha 0 is bald. `shells`,
`density`, `thickness` and `placement` are shell settings and do nothing here.

Each strand is drawn as a ribbon `width` wide at the root, tapered by `taper`,
turned to face the camera in the vertex stage. Gravity, wind, flutter and a
`FurAnimator`'s inertia bend it there too, in world space, by the same rule as
fur's droop, but walked at constant speed so a strand bends without
stretching. It is lit with its root's normal, so a head of hair shades as a
volume, and along its length by Kajiya-Kay.

`grow_strands` returns the curves themselves (`HairStrands`), to pose by hand
and draw with `hair_object_from`, or to turn into real triangles with
`tube_geometry(sides, radius, taper)` for the path tracer or an exporter.

Hair runs where fur does: WebGPU, WebGL2, native wgpu, and Metal (the same
port, pixel for pixel). From JavaScript, `scene.addHair(geometry, fur, count,
segments, width)`; in the shim, `new THREE.Hair(geometry, { count, segments,
width, length, gravity, curl, … })`. `cargo run --release --example hair`
renders the picture above.

**On a skinned mesh**, `FurAnimator::grow_hair_on_skinned(&mut scene, skinned,
&hair, &options)` grows the hair as the mesh's child and carries it through the
animation: each strand is rooted on a triangle, so it moves with the surface
there and turns as it turns, and trails its root's own motion on a spring
(`FurDynamics`), settling when the mesh stops. Call `animator.update` once a
frame, as for fur. From JavaScript, `scene.growHairOnSkinned(skinned, fur,
count, segments, width)` (or `createSkinnedHair` in the scene graph, linked as
the mesh's child); in the shim, `new THREE.Hair(skinnedMesh, { count, … })`,
which adds itself as the mesh's child — leave its transform at the identity.

**Shadows.** Hair casts and receives shadows by default, so it shadows the
body it grows on, the ground, and itself. A ribbon has no width seen from a
light, so the hair carries invisible three-sided tubes around a share of its
strands (`shadow_strands`, 0.25: one strand in four, each four times as thick)
that are drawn into the shadow maps only. They are bent on the CPU by the same
rule as the ribbons. `FurAnimator::update` re-bends them each frame for the
object's real place, wind and inertia; without an animator they are bent once,
for an untransformed object, so turned or windy hair wants one.
`shadow_strands: 0.0` leaves them out. In JavaScript, `shadowStrands` is the
last argument of `addHair` and its relatives, and an option of `THREE.Hair`.

Shell fur casts too: `FurMaterial::shadow_height` (0.6) pushes a copy of the
skin out to that share of the coat's length, drooped as the shells are, and
draws it into the shadow maps only — so a coat shadows the floor as wide as
the fur reaches, and shadows its own far side. 0 casts none. In JavaScript,
`WebFurMaterial.setShadowHeight` and the shim's `shadowHeight` option.

Fur and hair objects now turn `receive_shadow` on themselves, and their shader
honours it on every backend (WebGPU and WebGL2 used to shadow them always,
Metal only when asked).

One limit: strands do not collide, so long hair with weak gravity passes
through the body it hangs past.

## Fur and hair: how they move

The motion is physical in its parts, cheap in its whole. Everything that
bends a strand is a *pull*, per unit of its length; a strand's direction a
share `u` up is its root's normal plus `2u` times the pull, walked at
constant speed, so a strand bends and never stretches — shells and hair
alike.

- **Gravity** pulls as `FurGroom::gravity` says.
- **Air** pulls by drag, as the square of its speed past the strand:
  `drag × |u| u`, `u` the wind (`FurMaterial::wind`, m/s) less the strands'
  own velocity. A coat moving through still air streams back; one carried
  downwind stands up.
- **Gusts** are turbulence frozen into the air and carried downwind at the
  wind's speed: smooth noise eddies `|wind| / wind_frequency` across, of
  `turbulence` times the wind's speed. One sweeps across the coat; no two
  strands feel it alike.
- **Flutter** — vortices shed off each strand — is noise in time on the
  strand's own phase, faster in a faster wind, and none in still air.
- **Inertia**: with a `FurAnimator`, the coat's pull is a damped oscillator
  (`FurDynamics`: stiffness, damping, response): it rests where gravity and
  drag put it, the object's acceleration swings it off, and a gust or a stop
  eases in, overshoots and settles. On a skinned mesh every vertex (every
  strand's root, for hair) has its own, with its own velocity through the
  air.

![A ball that sets off, brakes and stops: its coat trails, swings forward and
settles](images/fur_motion.jpg)

*`cargo run --release --example fur_motion`: at rest, accelerating, braking,
settled.*

**A hand in the coat.** `FurAnimator::stroke(fur, at, radius, push, part)`
pushes the strands within `radius` of `at` along the hand's motion and parts
them away from it, fading to nothing at the edge; call it as the hand moves,
and let go — it fades in a fifth of a second and the strands spring back.
From JavaScript `WebScene.strokeFur(...)` or the shim's `fur.stroke(point,
push, radius, part)` (the demo raycasts the pointer onto invisible shapes
over each coat and strokes whatever it is over); from Python
`Scene.stroke_fur(...)`.

![A hand sweeping across a ball's coat: before, mid-stroke, and half a second
after letting go](images/fur_stroke.jpg)

*`cargo run --release --example fur_stroke`: before, while a hand sweeps
right across the upper left, and half a second after it lets go.*

Seen close, shells are slices: a strand drawn as a stack of them breaks into
dots once the slices are further apart on screen than it is wide. So each
shell draws its strands swept across its slab of height as the line of
sight crosses it, relative to the strand's own lean — a strand is one column
through the shells, at any distance.

![Fur close up, one sample per pixel beside 4× MSAA](images/fur_closeup.jpg)

*`cargo run --release --example fur_closeup`: one sample (left) and 4× MSAA
(right), a hand's width from the coat.*

## Fur and hair: anti-aliasing and speed

**Turn MSAA on** (`set_msaa(4)`; in JavaScript `renderer.setMsaa(4)` or a
render target's `samples`). Fur and hair then test each of the pixel's four
samples against the strands themselves and write the ones they cover —
`@builtin(sample_mask)` on native wgpu, `[[sample_mask]]` on Metal — so strand edges and strands thinner than a pixel resolve as real
geometry does, with no dither and nothing lost where strands overlap.
Against a 16×-supersampled render of the same scene, MSAA halves the error
(fur 16.7 → 8.4, hair 23.1 → 10.2). Hair ribbons are drawn at least 1.5
pixels wide and cut to their true width within, so thin hair fades rather
than breaking into dots; the highlight eases where the strand turns across a
pixel, so it does not sparkle. Frames holding skinned or instanced meshes, or
opaque shader materials, now stay multisampled too (they used to fall back
to one sample for the whole frame). Single-sampled, fur and hair draw as
before.

**In a browser** fur and hair cut their strands at the pixel's centre, MSAA
or not: Safari's WebGPU shades a multisampled fragment that discards, takes
derivatives, writes a sample mask or uses alpha-to-coverage once per sample —
fur does all of them, and a coat that runs at 60 fps single-sampled ran at 5
under MSAA. So a browser build leaves fur's coverage to the hardware's MSAA
of geometry edges (`sample_mask_is_cheap` in the renderer), and the demo
renders at the display's pixel ratio with MSAA off by default instead. The
demo is `web/examples/fur.html` (`THREE.Fur`, `THREE.Hair`, wind, inertia,
shadows): serve the repository root and open `/web/examples/fur.html`.

![The browser demo: a furry ball, a head of long hair and a furry torus in a
breeze, casting shadows on the floor](images/fur_demo.jpg)

*The demo in Chrome on WebGPU. Drag over a coat to stroke it; the slider
sets the wind. `?gl=1` forces WebGL2, `?msaa=1` turns MSAA on,
`?quality=low` draws a smaller canvas and fewer strands.*

**What it costs.** `cargo run --release --example fur_bench` measures it
(1280×720, a shadow-casting sun; median frame). Fur's strand search hashes
each candidate strand once, skips the ones too far away to reach the pixel,
and stops at the first hit; a coat draws as many shells as its length covers
pixels twice over (`FurMaterial::lod`, at least four), so a distant coat
costs little. The shadow casters move into child meshes the first time the
animator runs, so animating fur or hair re-uploads the caster alone, and only
when gravity, wind, inertia or the pose changed. Hair on a skinned mesh is
still posed on the CPU, its ribbons re-uploaded each frame: the costliest
case.

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
