# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.8] — 2026-10-24

### Added

- **Signed distance fields** (`threers::sdf`): exact 2D primitives (circle,
  box, rounded box, segment, capsule, tapered capsule, polygon) and 3D ones
  (sphere, box, rounded box, capsule, tapered capsule, torus, cylinder, plane),
  the hard and smooth union / intersection / subtraction, `round` and `onion`,
  and `smooth_union_blend`, which also returns how much of the result is each
  side so a colour or a thickness can blend through the same fillet as the
  surface. Any `Fn(Vector2) -> f32` or `Fn(Vector3) -> f32` is a field, with
  gradients, normals, sphere tracing (`Sdf3::raymarch`) and meshing through
  marching cubes (`Sdf3::to_geometry`). `sdf::WGSL` is the same library as
  `sdf2_*` / `sdf3_*` / `sdf_*` WGSL functions to prepend to a
  `ShaderMaterial` or a compute shader; `tests/sdf_gpu.rs` evaluates both
  sides on an adapter and holds them to 1e-4.
  `Overlay::distance_field` rasterises any such field with the current fill
  and stroke, and the overlay's own shapes now come from `sdf::d2` instead of
  private copies.

- **Coverage for thin lines and diagonal edges** (`threers::sdf`,
  `Overlay`): `sdf::coverage_stroke` box-filters a stroke against a pixel
  from the distance to its centre line, so a line thinner than a pixel draws
  at its own weight — a 0.4-pixel line peaks at 0.4 alpha and its ink sums
  to 0.4 across it wherever it falls, where the edge ramp drew it at 0.9
  over two pixels. `sdf::footprint` widens the ramp to a pixel's real width
  across an edge (`√2` at 45°), which stops thin diagonals beading. The
  overlay uses both for every line, polyline and outline, and adds
  `stroke_field` and `stroke_field_with_width` for any centre line; the `fx`
  surface (wgpu and Metal) carries each strand's radius through its smooth
  unions and covers thin strands the same way, so strand tips now taper to
  0.3 px instead of stopping at 0.55. WGSL as `sdf2_footprint`,
  `sdf_coverage_edge`, `sdf_coverage_stroke`.
  `SegmentMarks` take the same coverage across the line and around round
  ends and disc joints, in place of deck.gl's `smoothstep` edge: thin paths
  keep one weight as they pan instead of shimmering, and their edges are a
  linear ramp over the pixel's footprint rather than deck.gl's curve, so
  antialiased segments no longer match deck.gl pixel for pixel (aliased ones
  still do). `LineBasicMaterial::line_width` capsules do the same, and a
  transparent line thinner than a pixel is now drawn as one (opaque, it stays
  a line primitive).

- **Interactive effects** (`threers::fx`): glossy liquids that react to the
  pointer, built from three parts that each take additions — a `Component`
  (what an element is: `Button`, `Figure`), an `Effect` (what it wears:
  `Venom`, `Gloss`) and a `Stage` that lays elements out, hit-tests the
  pointer and raises `Enter`/`Leave`/`Press`/`Release`/`Click`, keeps one quad
  per element in the scene and paints the overlay in layers with one shared
  `Cursor`. `Venom` reaches for the pointer with strands of goo, grabs the
  cartoon glove finger by finger and snaps back with spikes and droplets;
  `Gloss` is the same material held still. Both draw a `Surface` — any
  `Shape` (pill, rounded box, circle, star, polygon, SVG path) plus strands,
  spikes, droplets and a mound, joined by smooth unions that carry a
  thickness for the shading — in a `Look` (`LACQUER`, `CHROME`, `GOLD`,
  `TOXIC`, or your own). `Outline` walks any shape's edge by arc length.
  Guide: `docs/fx.md`; examples `venom_button` and `venom_gallery`, both with
  a headless `--png` mode.
  **On the Metal renderer** too: it runs no WGSL, so it recognises surface
  materials and draws them with its own port (`src/metal/fx_surface.metal`),
  blending on encoded values as the wgpu path does; `tests/fx_metal.rs`
  renders one stage through both and they agree to 0.06 levels on average.

- **`sketch::Frame::pointer_inside` and `Frame::cursor_visible`**: whether the
  pointer is over the window (hover effects otherwise keep reacting to its last
  position after it leaves), and a switch to hide the system cursor for a
  sketch that draws its own.

- **A joining face composites a run by maximum, not by union**
  (`threers::loaders::overlap`): where two glyphs of a run put ink in the same
  cell, the union `a + b - ab/255` is what two partial coverages of one pixel
  come to — and for any script whose letters **join** it is wrong. The tail of
  one Arabic letter and the head of the next are the *same* stroke drawn twice,
  so a union darkens every joint, by 30 to 57 of 255 at the sizes measured. The
  rasteriser now takes the maximum for a face whose file says its letters join,
  and keeps the union for every other face: a Latin pair piled on top of itself
  still unions, because its glyphs really do each cover part of a shared edge
  pixel.
  Core Graphics has both composites and picks one **per font**, which is read
  rather than guessed: `CTFontDrawGlyphs` takes a glyph array and a position
  array, so the reference can be asked for glyph A alone, glyph B alone and the
  two together at the same pens, and the rule read off its own operands —
  `SFNS` gives `255 - floor((255-A)(255-B)/255)` and `SFArabic` gives
  `max(A, B)`. It is a maximum and not a geometric union: offsetting one copy
  of a glyph *vertically* puts a sliver of each in one cell side by side, where
  a union would be their sum, and the maximum still wins 663 of 663 cells.
  The selector is `CoreText`'s
  `EnumerateOverlappingGlyphs(CGContext*, TFont const&, …, void (CFRange, bool))`,
  whose `bool` chooses between `draw_glyph_bitmaps` and
  `draw_glyph_bitmap_run_with_compositing`, and this computes it from four
  tables: a `MERG` matrix bit, a `meta`/`dlng` language in a set of 28, three
  hard-coded PostScript names, an eight-range `cmap` test, and a `GPOS` `curs`
  feature. Three of those could not have been guessed — the ranges are
  **derived from the `cmap`** rather than read from `OS/2`, so Rockwell
  composites because its `cmap` covers one Arabic codepoint (`U+060B`, the
  Afghani sign) while Menlo *claims* the Arabic bit and has no Arabic at all;
  the 28 languages are a different set from the 8 ranges; and **`curs` is not
  unused on macOS** — an earlier note in this repository said cursive
  attachment was unmeasurable because no font has it, and `Kannada MN` and
  `Kannada Sangam MN` do, they are the only two fonts whose answer comes from
  that term, and without it they are the only two the chain gets wrong. So
  `curs` is implementable and testable here after all.
  Ported from `rswift`'s `swiftui_font::Font::composites_run`, which is checked
  against every font file on a macOS machine with the real path read off a
  breakpoint rather than inferred from pixels — 239 of 239 — and whose pixel
  guarantee for the identical rule is 0 differing cells of 9,365,885 against
  Core Text over Arabic, Hebrew and mixed runs at 1x, 2x and 3x. The pixel half
  cannot live here, for the same reason `tests/arabic_shaping.rs` gives: threers
  has no Core Text binding and should not grow one. See `tests/glyph_overlap.rs`
  and the unit tests in `src/loaders/overlap.rs`.

- **PostScript outlines and font collections** (`threers::loaders`): `.otf`
  files and the `CFF `/`CFF2` tables inside `.ttc` collections now load, where
  before `TtfFont::parse` returned `UnsupportedFormat` for both. These are not
  a fringe format — **60 of the 369 font files macOS ships** keep their
  outlines in `CFF`, and they are the ones Asian and Indic text needs: every
  Hiragino face, Apple SD Gothic Neo, Hiragino Sans GB and the whole Kohinoor
  family. All of them ship as collections, so neither piece reaches them
  alone. Hiragino Kaku Gothic W3 now loads 20,320 glyphs with outlines where
  it loaded none.
  A charstring is a little stack program rather than a point list, so the
  operand count is part of the operator — `hvcurveto` draws as many curves as
  it has arguments for and its last group may carry a fifth the others do not
  — and a glyph is not self-contained: fragments live in global and local
  subroutine arrays, biased by their own size, and on a CID-keyed font the
  local array depends on which font DICT `FDSelect` assigns the glyph.
  `CFF2` additionally blends each coordinate through its own variation store,
  which is **inside the `CFF2` table** rather than in `GDEF`.
  Ported from `rswift`'s `swiftui-font` and verified against it: reading the
  wrong one of those details draws a different glyph rather than failing, so
  `rswift`'s `swiftui-map/examples/cff_oracle` compares every curve of every
  glyph, sampled at five parameters, against a reader that is pixel-exact
  against `CTFontCreatePathForGlyph` — **2407 of 2407 glyphs over seven faces**
  agree to 0.00017 font units, which is `f32` storage rather than a difference
  in reading. It caught one real porting error: `CFF2`'s variation store wired
  to `GDEF`'s, which drew nothing at all for every variable glyph and looked
  like the format being unsupported. See `tests/cff_fonts.rs`.
- **Arabic marks are positioned** (`threers::loaders::mark`): `GPOS` types 4, 5 and 6,
  the `mark` and `mkmk` features. This is not only about combining characters, which is
  the part that surprises: `SFArabic` shapes a letter into a skeleton **plus its dots as
  separate mark glyphs**, so `كتاب` — four letters, nothing combining in the string at
  all — has a mark in it, and without this the dot sat at its skeleton's origin instead
  of 0.3857pt to the right of it. A dot the shaper produced has **no codepoint**, so
  `GDEF`'s glyph class is the only thing that can identify it.
  The painter now places in two passes, because a mark hangs off another glyph's pen and
  in a right-to-left run its base comes *after* it on screen. Letter spacing skips a
  mark — spacing is between letters, and charging a dot for a gap pushes the word apart —
  and `measure` skips it the same way, which is what keeps a centred line centred.
  `GPOS` and `GDEF`'s bytes are kept only when a font has mark lookups at all: `SF Pro`
  has none and pays nothing, `SF Arabic` pays 75 kB. Resolving the anchors ahead of time
  instead would mean tabulating every (mark, base) pair, the product of two coverages.
  Ported from `rswift`'s `swiftui-font`, where the same reader took the pens from 7 of 13
  right-to-left cases to 12 of 13 against `CTRunGetPositions`. `tests/arabic_shaping.rs`
  checks the number as well as the property: the dot lands 0.3857pt from its base at
  24pt, which is Core Text's own answer.
  `SFArabic` has **no `curs`**, so cursive attachment is absent — its census is `kern` 2
  pair, `mark` 5 mark-to-base and 2 mark-to-ligature, `mkmk` 2 mark-to-mark.
- **Captions are shaped** (`threers::loaders::shape`): Arabic letters join. An Arabic
  letter has up to four shapes and which is drawn depends on its neighbours, so a
  pipeline that maps one codepoint to one glyph draws every letter in isolation —
  readable to nobody. `GSUB` lookup types 1–7 including chaining contextual formats
  1, 2 and 3, per-script feature selection, joining types generated from ICU, and a
  generated script table that resolves a digit or a space to the script around it.
  `CaptionFont` grew a glyph-id-keyed layer beside its character-keyed one — `gid`,
  `advance_of`, `kerning_of`, `glyph_of`, `grown_glyph_of`, `shape_run` — and the
  character-keyed calls now delegate through it, so behaviour is unchanged where
  nothing shapes. The index means a glyph id for a TrueType or PostScript face and the
  codepoint for the three character-keyed faces, which is what lets one layout path
  serve all four: the built-in bitmap face reports the identity and nothing it can draw
  joins.
  Ported from `rswift`'s `swiftui-shape`, and checked against a **third**
  implementation rather than against either: `scripts/shapecmp.py` runs every
  non-AAT TrueType font on the machine through both this and `hb-shape`, and **178
  fonts agree on all four texts, 0 differ**. That found a real `cmap` bug — the
  `0xF000` symbol aliasing was gated on the subtable being `(3, 0)`, and
  `Hoefler Text Ornaments` puts the same content in a `(0, 3)` table that the
  preference order picks first, so every letter in it was unmapped. See
  `tests/arabic_shaping.rs`, whose assertions need no oracle: they check that a
  letter's glyph *changes* with its neighbours, which is false for any unshaped
  pipeline and true for any correct one.
- **Captions draw text, not codepoints** (`threers::captions::order`): between
  a caption's string and a row of glyphs there are three transformations the
  painter used to skip, and each one was visible. `"Noe\u{0308}l"` — an `e`
  followed by a combining diaeresis — drew a floating accent *after* the
  letter instead of on it, and measured as five letters rather than four, so
  every centred line sat half an accent to one side. Hebrew and Arabic read
  backwards. A bracket inside a right-to-left phrase opened the wrong way.
  Now: Unicode canonical ordering and composition (`captions::compose`, 403
  combining-class ranges and 961 composition pairs), the full bidirectional
  algorithm (`captions::bidi`, UAX #9 — paragraph level, embeddings,
  overrides, isolates, isolating run sequences, the weak types, paired
  brackets, implicit levels and reordering) and mirroring (**L4**). Both the
  width and the drawing walk the same order, which is what keeps a centred
  line centred.
  Kerning is looked up in **logical** order, which for a reversed run is the
  reverse of what is on screen — a pair taken visually finds a different pair
  or none, and every letter lands a fraction of a pixel out: the words read
  correctly and the spacing is subtly wrong. Ported from `rswift`'s
  `swiftui-bidi` and `swiftui-normalize`, where the ordering is measured
  against ICU over 360,000 strings (9 differ, all on several explicit
  embedding codes) and the composition against Core Text over 20,000 strings
  per font in six fonts (0 differ). Tables are **generated** from the
  platform's own ICU rather than transcribed.
  Shaping is still absent, so Arabic letters do not join — that needs `GSUB`.
- **Wide lines**: `LineBasicMaterial::line_width` now draws `LineSegments`
  that many device pixels wide (× the pixel ratio, as a point's size), where
  three.js's `linewidth` does nothing under WebGL and WebGPU has no line width.
  Each segment is an instanced, round-capped quad (`vs_wide_lines`), so a
  chain of short segments — a traced neuron, a sampled curve — reads as one
  smooth stroke with closed joints; indexed or not, with draw ranges, vertex
  colours, dashes, fog and clipping as before. Opaque lines cut at half
  coverage and draw under MSAA; blended ones fade over their last pixel and
  sort with the transparent draws. A width of 1 draws line primitives as
  before. See `tests/wide_lines.rs`.
- **Particles, snow and sand** (`threers::particles`): particle systems with
  physics — emitters (point, ball, box, disc, a mesh's surface), gravity,
  quadratic drag from each particle's terminal speed (integrated implicitly),
  wind, divergence-free curl turbulence with a Kolmogorov spectrum, a log-law
  boundary layer, attractors and vortices, buoyancy that decays as smoke
  cools, flutter for falling flakes, and collisions that bounce, stick, die,
  splash into spray, or deposit. Run by a `ParticleAnimator`, emitters
  following their objects. See [docs/particles.md](docs/particles.md).
  - **Snow and sand that pile up**: `SnowCover` and `SandBed`, heightfields
    lying on whatever is in the scene (found by raycasting), slumping to the
    angle of repose (50° and 34°, held to within 3°), drifted by the wind and
    shaped into dunes that migrate (Werner's slab model). Drawn by their own
    shaders: snow's subsurface light, grain and glitter; sand's wind ripples,
    grain speckle and quartz glints.
  - **Drawing**: billboards, motion streaks, lit sphere grains and screen-space
    heat-haze distortion; alpha (depth-sorted), additive and opaque. Billowing
    noise with its relief lit, Henyey–Greenstein scattering, self-shadowing,
    six-armed snowflakes, glints, black-body colour over life, shadows cast
    by smoke and received by every particle as the volume it is, and
    `light_with` — a point light flickering with a fire's flames. Each look
    compiles to its own pipeline with its settings as constants.
  - **`Weather`**: snow or rain filling a box round the camera, every flake
    placed by the vertex shader — no CPU cost, nothing uploaded per frame.
  - **Light through smoke**: with `extinction`, a particle is a sphere of
    medium: Beer–Lambert opacity, the sun and sky dimmed by the optical depth
    of the rest of the plume (a light-aligned grid each frame), Wrenninge's
    multiple-scattering octaves and a two-lobe phase function.
  - **Anisotropy**: raindrops' rainbow at 42° (and the secondary at 51°,
    reversed), plate-oriented flakes glinting in pillars, surface-layer
    turbulence (σ_u : σ_v : σ_w ≈ 2.5 : 1.9 : 1.25, still divergence-free),
    sand's sheen stretched along the wind (anisotropic GGX) and the opposition
    surge of granular surfaces.
  - **Embers that burn** (`Thermal`): radiation, Ranz–Marshall convection,
    diffusion- and Arrhenius-limited char oxidation; they shrink as they burn
    and are quenched on snow.
  - **Dunes** (`SandBed::dunes`, `particles::dunes`): procedural crest
    networks worked by the wind into slip faces at 34°, sand shaded after
    Journey's (grains, steep and shallow ripples, sheen, rim, glitter, wind
    streaks), spindrift off the crests, a desert sky at four times of day
    whose horizon the haze meets. Every surface detail band-limited by its
    size on screen: no moiré. Example `dunes`; browser demo
    `web/examples/dunes.html` (`THREE.DesertSky`, `THREE.Spindrift`,
    `SandBed.dunes()` / `.look()`).
  - Presets: `fire`, `embers`, `heat_haze`, `smoke`, `sparks`, `rain`,
    `snowfall`, `sand_stream`, `dust`, `fountain`.
  - **JavaScript**: `WebParticleSystem`, `WebScene.addParticles` /
    `createSnowCover` / `createSandBed` / `createWeather` / `stepParticles`;
    in the three.js shim `THREE.Particles`, `THREE.SnowCover`,
    `THREE.SandBed` and `THREE.Weather`, stepped by the renderer.
    `web/examples/particles.html`: a winter camp and a desert.
  - Examples `particles`, `snow`, `sand`, `campfire`, `dunes` and
    `particles_bench`.


- **Lattices: new families, print checks and faster builds.**
  - Foams: `Stochastic::VoronoiWet` (rounded bubbles, liquid in Plateau
    borders; `Lattice::plateau_border`), `GaussianField` (pore sizes spread
    ±45 %), `Hyperuniform` (evenly spread fibres) and `ReactionDiffusion` (a
    Gray–Scott maze grown once per seed and cached). `Lattice::seed_layout`
    puts Voronoi seeds on `Kelvin`, `Fcc`, `WeairePhelan` or `C15` crystals,
    or generates a periodic tile of `Poisson`, Lloyd-`Relaxed` or two-size
    `Bimodal` seeds (`seeds_per_cell`, `size_ratio`; bimodal foams are power
    diagrams, so walls stay flat). Every crystal cell has its textbook face
    count; open Kelvin foam equals `Strut::Kelvin`.
  - `Tpms::Helicoid` (double-screw columns) and the textiles `Strut::Weave`,
    `Braid` and `HelixBundle` (`Strut::is_textile`).
  - `utils::mesh_check`: `edge_report` (closed?), `triangle_quality`
    (needles, worst angles), `split_pieces` (solids and sealed cavities),
    `remove_loose_pieces` and `print_report` (overhang bands, support area
    and footprint, plate contact). `Lattice::drop_specks` removes contour
    debris but never more than 2 % of the solid, and never cavities.
  - `utils::simplify`: quadric edge collapse that keeps a closed mesh closed
    (link condition, no flips) — a finely cut box comes back to 12 triangles.
  - Speed, single thread over all 35 original lattices: 2.7× (strut and
    cuboct cells 2.5–9× from per-cell strut bins, TPMS 1.2–1.5× from per-axis
    trig tables, spinodals 3–5× from row-rotated waves, honeycomb 2.8×), with
    31 meshes bit-identical and the spinodals within 10⁻³ of a step. Voronoi
    foams precompute their exact cell neighbours once (cached across the
    samplers a density fit builds): closed foams read their walls from it
    (1.4–2.6×, bimodal 10×) and open foams their edges as binned segments
    (3–21×); a 5³ open foam now builds in 0.09 s cold, 0.04 s warm. With
    `parallel` the contour is drawn in slabs and welded — byte-identical to
    one sweep, 2.3× on a 200³ grid.
  - Examples `lattice_foams` (six seed layouts as open, closed and wet
    foams), `lattice_textiles` (weave, braid, rope, helicoid and the grown
    grains) and `lattice_print` (overhangs that need support, before and
    after simplification), with images in `docs/lattices.md` and the README.

### Changed

- **`threers-physics` now shares the `threers` version.** It goes from 0.0.1
  on crates.io straight to 0.0.8 and is bumped with the main crate from here
  on; `scripts/release-versions.sh` checks it with the other roots, and under
  `CHECK=1` it also fails if `threers` or `threers-physics` is not ahead of
  the version already on crates.io.

### Fixed

- **Extruded polygon walls faced inward when a ring ran clockwise.** Their
  normals pointed out, but each wall quad was wound the same way whatever
  the ring's direction, so `PolygonMarks::to_buffer_geometry` handed a scene
  renderer walls it culled — and so no shadows from them. Outer rings now
  always run clockwise (deck.gl's winding), which made every building do
  this; walls are wound to face their normals.

- **Metal line quads were half as wide as asked.** `set_line_width` quads
  took a pixel as 1/width of clip space, not 2/width, and twisted into a
  bowtie, each end measuring the segment from itself; with the quad never
  under a pixel wide, a sub-pixel filament lit a whole pixel. They are now as wide as asked,
  box-filtered when transparent, and cut at half a width (never under a
  pixel) when opaque.

- **Overlay polylines beaded at every joint.** `polyline` drew each segment
  on its own, so the anti-aliased pixels two segments share were blended
  twice: a dark spot at every vertex of a thin or translucent line. It is
  one stroke now.

- **SVG paths: the segment after an arc started where the arc began.**
  `parse_svg_subpaths` advanced its own pen past an `A`/`a` command but not the
  `Path`'s, which `line_to` and the curve builders start from — so in
  `M9 14V3.6a2.5 2.5 0 0 1 5 0V12.2` the last line ran diagonally from (9, 3.6)
  instead of straight down from (14, 3.6). Any outline with an arc followed by
  another command was distorted, rounded fingertips and rounded-rectangle
  corners most visibly.

- **Voronoi foam walls were not the thickness asked for.** The field read
  half the gap between the nearest seed distances, which is zero on the
  skeleton but not a distance: its gradient ran from 0.3 to 1, so a wall of
  `thickness` t came out up to three times as thick. It is the exact distance
  to the walls and edges now, so densities fitted with
  `fit_relative_density` are unchanged but fixed-thickness foams are thinner.

- **Marching cubes put several vertices at one point** when a sample sat
  exactly on the surface (a box face on the grid, two spheres touching):
  `sdf::to_geometry` gave a 0.5 cube 388 coincident vertices and 776
  zero-area triangles. Crossings are kept 10⁻⁴ of an edge off the samples.

## [0.0.7] — 2026-10-21

### Added

- **Fur** (`threers::fur`): shell fur on any mesh. `fur_object(geometry,
  &FurMaterial)` draws the skin and `shells` offset copies of it in one
  instanced draw, cut to procedural strands, each with its own length, taper,
  curl and clump (Worley clumps, so no grid lines), lit along the strand
  (Kajiya-Kay) with shadows. Strands are rooted on the UVs (a grid fitted to
  their stretch) or, with `FurPlacement::Surface`, over the surface in object
  space with no UVs at all. Vertex colours paint it: RGB tints, alpha scales
  length.
  - **Motion**: gravity in world space; wind in gusts that travel across the
    coat downwind; per-strand `flutter`; and a `FurAnimator` that adds inertia
    (`FurDynamics`: a spring per object, measured trailing an accelerating
    ball by 4 px and settling to 0) and grows fur on skinned meshes,
    re-posed every frame with a spring per vertex.
  - **Everywhere**: WebGPU, WebGL2 and native wgpu, and Metal through a port
    of the shader that draws what wgpu draws (same coverage, same mean
    brightness). JavaScript: `WebFurMaterial`, `WebScene.addFur` /
    `createFur` / `stepFur`; the three.js shim's `THREE.Fur`, animated at
    every `render`.
  - `tests/fur.rs`, `tests/fur_metal.rs` (under the validation layer), a
    browser test on WebGL2 and WebGPU, `crates/threers-js/test/fur.test.mjs`;
    `examples/fur.rs` renders it; `fur_motion`, `fur_stroke` and
    `fur_closeup` show inertia, a hand stroking the coat and the close-up
    with and without MSAA, all pictured in `docs/examples.md` and
    `docs/materials.md`.
  - **Strand hair** (`fur::strands`): `hair_object(geometry, &FurMaterial,
    &StrandOptions)` grows `count` curves from the same `FurGroom` (by area,
    seeded; curl, clumps around guide strands, length variation, vertex-colour
    paint) and draws them as camera-facing tapered ribbons with fur's shading
    and motion: gravity, wind and inertia bend each strand without stretching
    it. `grow_strands` returns the curves; `HairStrands::tube_geometry` makes
    real tubes of them. Metal draws it pixel for pixel as wgpu does.
    `FurAnimator::grow_hair_on_skinned` carries hair through a skinned
    mesh's animation, each strand on its triangle and trailing its root on
    its own spring; from JavaScript, `WebScene.growHairOnSkinned` /
    `createSkinnedHair` and `new THREE.Hair(skinnedMesh, options)`.
    Hair casts shadows — on the body, the ground and itself — through
    invisible tubes around a share of its strands (`shadow_strands`),
    bent as the ribbons are and kept in step by the animator; the same on
    Metal.
  - **Fur shadows**: shell fur casts through an invisible copy of its skin
    pushed out to `FurMaterial::shadow_height` of the coat (0.6), drooped
    as the shells are — a shadow as wide as the fur, and the coat shadowing
    itself. Tested natively, on Metal and in the browser on both backends.
  - **Anti-aliased under MSAA**: fur and hair test each sample against the
    strands and write the ones they cover (`@builtin(sample_mask)`; alpha-to-
    coverage on WebGL2) — half the error of one sample against a 16×-
    supersampled render. Hair ribbons are at least 1.5 px wide, cut to their
    true width; the highlight eases with the strand's turn across a pixel.
    Metal multisamples them pixel for pixel as wgpu does.
  - **Physical motion**: strands bend without stretching (shells too);
    wind pulls by drag as the square of the air's speed past them (`wind`
    in m/s, `FurMaterial::drag`), so a moving coat streams back; gusts are
    turbulence carried downwind at the wind's speed (`turbulence`); flutter
    is per-strand noise that needs a wind; the animator's spring now drives
    the whole pull — gravity, drag and inertia — so changes ease in and
    overshoot. Shadow casters bend by the same rules on the CPU.
  - **Stroke it**: `FurAnimator::stroke` puts a hand in a coat — strands
    pushed along its motion and parted around it, springing back when let
    go. JavaScript `WebScene.strokeFur` / `fur.stroke`, Python
    `Scene.stroke_fur`; the demo strokes whatever the pointer is over.
  - **No dots up close**: each shell sweeps its strands across its slab as
    the line of sight crosses it, so a strand stays one column.
  - **Faster**: fur's strand search about twice as fast (hashes once, skips
    strands out of reach, stops at the first hit, no integer division); shell
    level of detail by on-screen size (`FurMaterial::lod`); shadow casters
    moved to child meshes by the animator and rebent only when their inputs
    change — still hair with no wind costs nothing a frame. `examples/
    fur_bench.rs` measures it.
- `ShaderMaterial::alpha_to_coverage` (three.js `alphaToCoverage`;
  `coverage_alpha(c)` in the shader), and the preamble's `THREERS_SAMPLES`,
  `threers_sample_offset(i)` and `frag_out(colour, mask)` for a shader that
  works out its own MSAA coverage. JavaScript: `ShaderMaterial({
  alphaToCoverage })`.
- Python: `FurMaterial`, `Scene.add_fur`, `Scene.add_hair`, `Scene.step_fur`,
  `HeadlessRenderer.set_msaa`.
- `web/examples/fur.html`: a browser demo of shell fur, strand hair and curly
  fur — swinging, in the wind, casting shadows — on the three.js shim.
    JavaScript: `WebScene.addHair` / `createHair`, the shim's `THREE.Hair`.
    `examples/hair.rs`; tested natively, on Metal, in the browser on both
    backends, and in Node.
- `ShaderMaterial::fallback_color`: what the Metal renderer, which cannot run
  custom WGSL, draws a shader material in (white by default, as before).
- **`ui::Ui::set_layout_units`** and **`ui::layout_unit`**: boxes on
  Blink's `LayoutUnit` grid (1/64 px) where float error is all that keeps
  them off it (within 1/512 px); `layout_unit` is `LayoutUnit(float)`'s
  truncation, for a caller's own geometry. Skia draws glyphs at
  quarter-pixel positions, so a run a hair left of where a page would put
  it (976.62463 px against Blink's 976.625) is drawn a quarter pixel off;
  a UI matching a page pixel for pixel turns this on. It also wraps text
  as Blink does: a line's width is taken between its ends' offsets in the
  whole text, each rounded up to 1/64 px, so a word on a later line fits a
  box Blink sized for it. Off by default.
- **`ui::canvas::CanvasPainter`** (wasm32): a `ui::Painter` on the page's 2D
  canvas — boxes snapped to device pixels as Chrome paints them, rounded
  boxes, gradients, SVG paths, clips, text with Blink's half-leading and at
  the `LayoutUnit` (1/64 px) a page would put it, so `fillText` lands on the
  same quarter pixel as the page's text — and
  `CanvasPainter::with_dom_text`, which lays the text as positioned DOM over
  the canvas instead of `fillText` (each run's baseline where `fillText`
  would put it, measured once per font and text), since Chrome rasterizes
  some text differently on a canvas (translucent colours, for one); and
  `with_dom_text_for`, the same for the runs a predicate picks, the rest on
  the canvas. Painting order is kept with layers (anything painted after
  DOM text goes on a canvas above it); clips apply to the text too. Call
  `begin` before painting a frame and `finish` after;
  `begin_with_backdrop` draws another canvas (a WebGPU one) under the UI.
  A blurred gradient (`fill_linear_gradient` with `blur`) is blurred on a
  surface of its own and composited at whole device pixels, as Chrome
  composites an element with `filter: blur()`.
- **`renderer::readback::AsyncReadback`**: a buffer read back without
  blocking, native and web alike — copy into it, `map(wake)`, then `read()`
  on later frames until the bytes come (the pattern for pick windows,
  values a compute pass works out, and the like, where `texture_bytes`'s
  blocking wait is not an option).
- **`loaders::draco::decode_mesh_with` and `DecodeOptions`**: decode a Draco
  mesh with the transforms of chosen attribute types skipped (Draco's
  `Decoder::SetSkipAttributeTransform`), so quantized positions come out as
  the integers stored. Formats that define their own dequantization (such as
  neuroglancer's multi-resolution meshes, which read positions as fractions
  of each octree node) need these rather than Draco's floats.
  `attributes::decode_attributes_skipping` is the same at the attribute level.
- **`GpuContext::open_with_optional_features`** (and
  `with_surface_and_optional_features`, `headless_with_optional_features`):
  `open`, also asking for extra features where the adapter has them (such as
  `FLOAT32_BLENDABLE`), for renderers that use them when present and do
  without otherwise.
- **`volume::SliceRenderer`: `slice_color_at`.** A colour function may
  define `slice_color_at(voxel, params)` instead of `slice_color`: it gets
  the voxel position (bounds-checking it itself) and reads the `volume`
  texture as it needs, for data laid out across the texture (channels
  stacked along z, say).
- **`ui`: `text-overflow: ellipsis`.** A clipping box (`overflow: hidden`)
  whose text does not wrap cuts a run too wide for it before `…`, as CSS
  does; `Style::ellipsis`.
- **Examples with pictures**, each in `docs/examples.md` and the README
  (66 now): `shadow_types` (the four filters on one scene),
  `skinned_animation` (a rig built in code, two clips cross-faded),
  `postfx_chain` (`EffectComposer` natively: SSAO, bloom, depth of field,
  outline, pixel art), `usd_stage` (a class, references, a variant set,
  `UsdPreviewSurface`, a `.usdz` round trip), `backend_parity` (wgpu and
  Metal; `web/examples/parity.html` is the same scene in a browser),
  `data_charts` (bars, a picked scatter plot and the colour ramps from
  `marks`), `volume_slices` (a Shepp–Logan phantom through `SliceRenderer`),
  `ui_panel` (`threers::ui` over a render, hit-tested) and
  `threers-physics`'s `tower_collapse`. `ocean_water`, `simcity_render`,
  `svg_gallery` and `openscad_gallery` already drew pictures and are in the
  gallery too, as are `retrofuturism` (its revealed frame) and `neon_logo`,
  which takes `NEON_MARK=drop` now to sweep a neutral mark — an egg and a
  lens — in place of the traced outlines, so a picture of the technique is
  not a picture of someone's logo.

### Fixed

- A `ShaderMaterial` with `screen_space` set failed to build its pipeline
  ("binding is missing from the pipeline layout") and drew nothing: the
  screen-space capture moved out of the main layout's group 3 in 0.0.6, and
  shader materials kept the main layout. They get the refraction pass's now.
  `tests/screen_space_shader.rs`.

- The `ocean_water` example drew no water: its shader read `ss_depth_tex`
  as a depth texture, which it has not been since 0.0.6, and called
  `g_smith`, which is gone. It reads `.r` (as standard depth) and uses
  `v_ggx_smith_correlated`.
  Its clouds drew as a skyline of rectangles in a light deck (`moonlit`):
  the density was two-dimensional noise stood up through the whole 800 m
  slab, so every cloud was a flat-topped pillar. A cloud is as tall as it is
  dense now, domed on top and leaning downwind, and the noise's octaves are
  turned against each other so a high threshold no longer picks out its
  lattice. Same ten-step march, same cost.

- `extras::MarchingCubes` was not marching cubes: it fanned each cell's
  edge crossings in whatever order it found them, so its surfaces were
  torn, inconsistently wound and unindexed. It is the real algorithm now,
  on the 256-case tables three.js ships: one vertex per grid edge shared by
  every triangle on it, normals from the field's gradient, every triangle
  wound outward. A sphere comes out watertight and within 1% of its volume
  and area, which its tests check. The field is inside where it is above
  `iso`, as three.js's is.

- USD: a referenced prim's own variant selection was ignored, so a class
  selecting `enamel` for itself came back with its first variant instead.
  It holds now, weaker than any selection the referencing prim makes and
  stronger than a `variant_fallbacks` entry. The composition guide's
  import of `ComposeOptions` and `FileResolver` named a path that does not
  export them (`threers::loaders::usd::…` does).

- WebGL2: a draw with `colorWrite: false` left the whole frame black. The
  renderer now skips colour-pass draws that write neither colour nor depth
  (fur's and hair's shadow casters among them), which fixes them; a
  `colorWrite: false` draw that does write depth (an invisible occluder) is
  still affected there.

- MSAA fell back to one sample for any frame holding a skinned mesh, an
  instanced mesh or a shader material. They have multisampled pipelines now
  (an opaque shader material's built on demand), so such frames stay
  multisampled.

- Fur received shadows on WebGPU and WebGL2 whatever its `receive_shadow`,
  and on Metal only with it on. It now honours the flag everywhere, and
  `fur_object` and the hair constructors turn it on.

- A `ShaderMaterial` on a geometry with `uv1`/`uv2`/`uv3` read them as zeros
  on wgpu unless the geometry was also morphed: the draw was flagged as having
  uv sets but bound the shared empty texture. It now binds the geometry's.

- **`ui`: a row's height from its items at their flexed widths.** A flex
  row with a definite width now sizes its cross axis after resolving its
  items' main sizes, as CSS does, so text that wraps in a narrowed item
  (`white-space: normal`) makes the row taller instead of overflowing it.
  The main-size resolution is shared between sizing and layout.

### Changed

- **`ui::controls::checkbox`**, checked at 13px: painted as Chrome
  rasterizes its native checkbox (opaque but for the corner pixels, a 0.443
  blend), not as a rounded fill, whose canvas coverage comes in quarters.
- **GIF is its own feature, `gif`** (implies `native-codec`; `media`, `full`
  and `wasm-full` include it). `native-codec` alone no longer compiles
  `codec::gif`. The animation and video exporters keep `BrowserCodec::Gif` /
  `VideoCodec::Gif` in every build; choosing it without the feature is the new
  `AnimationEncodeError::FeatureDisabled("gif")` rather than a missing variant.
- **`codec::gif::DecodedFrame` is now `GifFrame`**, so it is not mistaken for
  the video decoder's `codec::decode::DecodedFrame`. The old name remains as a
  deprecated alias (in `codec::gif`, `codec` and the crate root) for this
  release.

## [0.0.6] — 2026-10-19

Universal Scene Description, read and written in all three of its forms and
checked against Apple's own tools rather than against itself; **Draco** and the
rest of the glTF compression and interchange formats, each verified against the
reference decoder that produced the files; a **WebGL2 backend** so the browser
build runs where WebGPU is not available; an SVG renderer that draws the scene
graph as vectors; sharper text; a converging accumulation mode for offline
frames; **marks**, the data layers a kepler.gl- or Mapbox-style application is
built from (points, paths, polygons, labels, heatmaps and map tiles, with
`f64` positions, GPU filtering and picking); reversed depth for planet-scale
views; native **H.264 and HEVC decoding** in pure Rust, behind one `Decoder`
that uses VideoToolbox where the build has it and the software decoder
everywhere else; a **retained UI** laid out by CSS rules, with form controls
and a code editor; and the fixes that writing all of it turned up.

### Added

- **More of the crate reachable from downstream code.**
  - The prelude has `ShadowType` and `TransparencyMode` beside the lights and
    materials that take them, and `ColorSpace`, `ColorScale`, `ScaleKind` and
    `Palette` beside `ColorRamp`; a new `prelude::video` gathers the encoders,
    MP4 and (with `video`) the decoders. The crate root exports `ShadowType`
    and the colour types too, as the prelude's docs promise for everything.
  - `MeshBvh::node(i)` returns a `BvhNodeView` (internal with two children,
    or a leaf with its triangle range), so a BVH can be walked by hand for
    queries `shapecast` does not cover.
  - `curves::interpolations::{catmull_rom, quadratic_bezier, cubic_bezier}`:
    three.js's `Interpolations`, in `f64`.
  - `exact_csg::is_watertight(&geometry)`: whether a mesh is a closed,
    consistently wound solid.
  - On docs.rs every feature-gated item is badged with the features it needs.
  `tests/public_api.rs` uses each through the crate's public paths.
- **A static site for any CDN** (`node web/scripts/stage-site.mjs`, see
  `docs/deploying.md`): stages `web/` for Vercel, GitHub Pages, Netlify or
  Cloudflare Pages with content-hashed wasm (cacheable for a year, every
  reference rewritten), `modulepreload`/`preload` hints so the wasm download
  starts with the HTML (one request in Chrome, started by the preload),
  minified JavaScript with source maps (the shim 0.70 → 0.37 MB, 0.09 MB
  brotli), `vercel.json` and `_headers` with `application/wasm` and the
  caching rules, and a 25 MB per-file check.
- **Native H.264 and HEVC decoding** (`threers::codec::decode`, features
  `native-codec` and `video`): from-scratch, pure-Rust decoders that also
  build for `wasm32`. An app that only encodes leaves `video` off and does not
  compile them; the parameter-set parsers the MP4 muxer needs stay available. `NativeDecoder` has the same interface as
  `videotoolbox_decode::VideoToolboxDecoder` (open from an `mp4_demux`
  track, `push` samples in decode order, `take` BGRA by presentation time),
  plus `take_frame` for the decoded planes and `decode_annexb` for raw
  elementary streams. H.264 covers progressive Baseline/Main/High (CAVLC,
  CABAC, B-frames with spatial/temporal direct, weighted prediction, 8×8
  transform, scaling matrices, long-term references, monochrome, high bit
  depth). HEVC covers Main/Main 10/Main Still Picture and 4:2:0 Main 12,
  including tiles, WPP, dependent slices, PCM, AMP, TMVP, SAO and scaling
  lists. Output is bit-exact against ffmpeg (H.264), x265's reconstruction
  (HEVC), the crate's own encoders, and VideoToolbox-encoded streams;
  unsupported tools (interlacing, 4:2:2/4:4:4, FMO) are refused with an
  error naming them.
- `codec::decode::Decoder`, the best decoder this build has: VideoToolbox
  on macOS with the `videotoolbox` feature, falling back to `NativeDecoder`
  when VideoToolbox refuses a stream, and native everywhere else. It has the
  same `push`/`take` interface. `Decoder::with_backend` picks one backend
  explicitly, and `Backend::preferred()` / `Backend::available()` report what
  the build offers. `NativeDecoder::from_config` and `Decoder::from_config`
  open a decoder from a raw `avcC`/`hvcC` without building a `VideoTrack`.
- `h264::CompressedEncoder::avcc()`, `encode_sample()` and
  `encode_sample_at()`, which return the length-prefixed MP4 sample directly.
- `Yuv420Frame::from_bgra` for BGRA input, which is what decoders return.
  `Yuv420Frame` is also re-exported as `codec::h264::Yuv420Frame`.
- **A retained UI** (`threers::ui`, feature `ui`), for interfaces specified
  in CSS terms and drawn the same way natively and in the browser. A `Ui` is
  a tree of boxes, text runs and SVG paths, styled with a subset of CSS
  (`display: block | flex | none`, flex direction/grow/shrink/basis, align
  and justify, gap, width/height with min and max, margin, padding and
  border in content-box or border-box sizing, static/relative/absolute
  positioning, colour, font, opacity, clipping). `Ui::layout` follows the
  CSS flexbox algorithm in its common path, including freezing of items that
  reach their limits and an `auto` minimum size of the content. `Ui::paint`
  draws through a `Painter`. `OverlayPainter` targets `Overlay`, and a
  browser app can implement the trait on its 2D canvas so that text
  rasterizes exactly as the page's own would. `Ui::hit_test` maps a pointer
  position to a node, counting the pointer's pixel as Chrome does (a box
  starting at 60.95 is under a pointer at 60). A box with a uniform border, a
  `border-radius` and a background paints its border rounded too (the border
  colour over the rounded box, the background inset by the border).
- **Form controls** (`threers::ui::controls`, feature `ui`), built as
  ordinary subtrees with Chrome's default look and sizes: `checkbox`,
  `radio`, `button`, `select` and its `select_popup`, `range` (with
  `range_fraction` for input), `color_well`, a text input's `text_contents`
  (selection, caret, placeholder) and `focus_ring`, `code_view`, a full
  `code_editor` (gutters, fold markers, selection, caret, scrolling;
  `code_column` maps a pointer to a column), and `overlay` / `overlay_top`
  (a dimmed page and a box centred on it, or near its top).
- **Multi-line editing** (`threers::ui::TextArea`): lines, a selection, and
  CodeMirror's default keys (Enter keeps the indentation, Up/Down keep a
  goal column, Home/End per line, Ctrl/⌘ for the document).
- **Code folding** (`threers::ui::code`): `brace_folds` as CodeMirror's
  `brace-fold` finds ranges, `Folds` (`fold_all`, `unfold`, `toggle`) and
  `shown_lines`.
- **`Style::subpixel`**: a subtree painted at fractional positions (as a
  CSS `transform` places it); painters may implement
  `Painter::set_subpixel`.
- **`controls::color_picker`**: a colour picker popup (saturation and value
  over the hue, a hue strip, a swatch and the hex value) for where the
  system has none, with `color_picker_field`/`color_picker_hue` for input
  and `hsv_to_rgb`/`rgb_to_hsv`. `controls::range` now places its thumb at
  whole pixels and insets its track 1px, as Chrome does.
- **`border-top`, `border-right`, `border-bottom`, `border-left`** in
  `style!` and `css::apply`: one edge's width (and the box's border colour).
- **`Ui::reparent`**: moves a node, with what is inside it, to the end of
  another parent's children (the DOM's `appendChild` of an attached node).
- **`style!` and `ui!` macros** (feature `ui`). `style! { display: flex;
  margin: 0 0 3px; border: 1px solid #ccc; font-family: sans-serif; }` builds
  a `Style` from CSS declarations, including shorthands (`flex`, `border`,
  one to four `margin`/`padding` values), colours (`#rgb`, `#rrggbb`,
  `rgba(…)`, named), `%` and `auto`. `{expr}` passes a computed Rust value,
  and `..base;` starts from another style. `ui!(tree, parent, div(…) { "text"
  div(…) {} })` builds nested boxes and text. CSS mistakes panic with the
  declaration. `ui::css` exposes the parser.
- More of CSS in the UI:
  - `border-radius` (elliptical for percentages);
  - `text-decoration: line-through`;
  - a stepped or smooth `linear-gradient` background, optionally with
    `filter: blur()`.
- `Painter` gains hooks for rounded rectangles and gradients, each with a
  plain fallback.
- `OverlayPainter::with_scale` paints a UI laid out in logical pixels onto a
  HiDPI overlay.
- `circle_path`/`parse_circle` carry SVG `<circle>`s through path data, so
  that painters can stroke them as ovals, as browsers do (a circle drawn as
  arcs rasterizes differently).

- More of the UI toolkit:
  - scrolling boxes (`overflow: auto`, `scroll_top`; `Ui::scroll_extent`,
    `Ui::scroll_container`);
  - `background-clip: content-box`;
  - clipping to rounded boxes (`Painter::push_clip_rounded`);
  - `Ui::move_child` (CSS `order`) and `Ui::offset`;
  - more named colours in `style!`.
- `ui::TextField`: a single-line field's editing (selection, Shift-extended
  caret moves, Home/End, select-all, filtered typing, delete), reporting
  Enter, Escape and Tab.
- **Lighting options.**
  - `ShadowSettings::radius` (three.js's `LightShadow.radius`) sets how soft
    shadows are, for directional, spot and point lights. A
    `ShadowSettings { .. }` literal that lists every field needs `radius: 1.0`
    (or `..Default::default()`).
  - A frame carries eight lights of each kind, up from four. The fifth and
    later lights used to be dropped silently; `tests/many_lights.rs` lights
    six.
  - **Every** directional, point and spot light with `cast_shadow` casts
    shadows, on both renderers. Only the first of each kind used to: each
    kind's maps are now layers of one depth array (six a point light, looked
    up by direction with the cube face chosen by hand, since WebGL2 has no
    cube arrays), and several lights of a kind share a budget of two 4096²
    maps. `tests/multi_light_shadows.rs` takes where each of three lights'
    shadows falls from ray casts, on wgpu and Metal.
  - `WebLight.setShadowParams(bias, normalBias, radius)` sets them from
    JavaScript, in three.js's conventions, and the three.js shim forwards
    `light.shadow.bias`, `.normalBias` and `.radius` to it.
  - `Renderer::last_frame_multisampled` (`WebRenderer.lastFrameMultisampled`
    in JavaScript) says whether the last frame got the MSAA asked for.
  - **Multisampled render targets**, three.js's `renderTarget.samples`:
    `RenderTarget::set_samples(4)` (`WebRenderTarget.setSamples`, and the
    shim's `new WebGLRenderTarget(w, h, { samples: 4 })`) draws the target
    multisampled whatever the renderer's own MSAA, which covers the canvas
    only, as `antialias` does in three.js. Colour is resolved into the
    target, and so is depth (resolved by a shader, or on WebGL2 redrawn),
    so post-processing and refraction glass read it as they would a
    single-sampled target's. Half-float targets stay single-sampled. Before,
    a render target never got MSAA at all. A canvas and a target of another
    size alternating each frame keep both sets of multisampled buffers
    rather than reallocating them. `tests/render_target_msaa.rs`.
  - **Metal**: a multisampled `MetalRenderTarget` now resolves its depth
    as well — into a single-sampled `resolved_depth_texture()`, by the main
    pass's own depth attachment, whenever depth is stored
    (`set_store_depth`; `PassAttachments::with_depth_resolve` for textures
    of your own). Headless SSAO used to be skipped with a warning whenever
    MSAA was on; it now runs, and darkens the same pixels at 4x as at 1x,
    with OIT glass in the frame too. `tests/metal_ssao_msaa.rs`.
  - **Shadow filters**, three.js's `shadowMap.type`:
    `Renderer::set_shadow_type(ShadowType::{Basic, Pcf, PcfSoft, Pcss})`, or
    per light with `ShadowSettings::filter`. PCSS softens a shadow with the
    distance from its caster (`ShadowSettings::light_size`), its blocker
    search done with comparisons alone so it runs on WebGL2; three.js's
    `VSMShadowMap` maps to it.
  - **Cascaded shadow maps** for directional lights
    (`ShadowSettings::cascades`, `cascade_distance`, `cascade_split`): up to
    four maps over the camera's view, sphere-fitted and snapped to their
    texel grid so they do not shimmer. `tests/shadow_types.rs` checks a small
    shadow near the camera lands where a tightly fitted map puts it, where
    one map over the whole reach loses it — on both renderers.
  - Shadows are stored per light: a record for each casting light, named
    from the light, one lookup for every kind, every record's shadow worked
    out once per fragment (which also halved the WebGL2 lit shader's compile
    time).
  - Shadow passes are reused one by one — a camera-following cascade no
    longer redraws a still point light's six faces — and draw only the
    casters inside their light's frustum: the nine-shadow benchmark went
    from 25 passes and about 5,000 caster draws a frame to 4 and 700.
    `Renderer::shadow_stats` counts both. The Metal renderer culls each
    view's draws to its frustum too, which it did not before.
  - **Overlapping refraction glass**: `Renderer::set_refraction_layers`
    (2 by default) draws `Refract` objects back to front in that many
    groups, each refracting a capture of what was drawn before it, so glass
    shows through glass. OIT surfaces sit among the groups by depth.
  - **The Metal renderer's OIT** is weighted-blended OIT, as wgpu's, in
    passes of its own, multisampled when the frame is — the same colour to
    the byte (`tests/oit_parity.rs`), where it used to sort and blend.
  - **The path tracer's bounce limits**, as Cycles has them:
    `diffuse_max_bounces`, `glossy_max_bounces` and
    `transmission_max_bounces` (`with_bounce_limits`), on CPU and GPU alike.
  - **Colour**: every CSS Color 4 string (`Color::from_css`: `lab()`,
    `oklch()`, `hwb()`, `color(display-p3 …)`, `#rrggbbaa`, …, also behind
    the shim's `Color.setStyle`), conversions to HSV, HWB, XYZ, CIELAB/LCh
    and OkLCh, `Color::mix` in any of them, WCAG contrast; d3's named
    schemes (ColorBrewer, Tableau, cividis, twilight) for `ColorRamp` and
    `Palette`; and `ColorScale`, d3's sequential, diverging, log, pow,
    symlog, quantize, quantile and threshold scales over any ramp
    (`WebColorScale` from JavaScript). See `docs/colors.md`.
  - The three.js shim takes threers's glass modes as
    `MeshPhysicalMaterial.transparencyMode` (`'blend'`, `'refract'`,
    `'glass'`, `'oit'`), in the constructor or set later, and forwards
    `light.shadow.mapSize` (`WebLight.setShadowMapSize`).
    `WebMaterial.transparencyMode` and `WebLight.shadowSettings` read back
    what the renderer holds; `crates/threers-js/test/glass-and-shadow-settings.test.mjs`
    checks the shim through them.
- **Speed.** Every draw's uniforms now live in one buffer, written once a
  frame, instead of one buffer and one `write_buffer` call per draw. 2,000
  lit, shadowed meshes went from 11.9 ms to 4.0 ms of CPU a frame (17.8 to
  5.3 ms with the GPU).
- **Shadow maps are redrawn only when they would change.** With
  `shadowMap.autoUpdate` on, every caster used to be redrawn into every
  shadow map on every render. Now each render fingerprints what the maps
  depend on (casters' geometry, transforms, sides and cut-outs, skinned
  poses, instance data, custom shaders' uniforms and sources, and the lights'
  projections) and keeps the maps when nothing changed. A still scene under a
  moving camera stops paying for its shadows each frame. Only custom-shader
  casters that sample textures still redraw every render.
  `tests/shadow_reuse.rs` checks both directions for plain, instanced, skinned
  and custom-shader casters, and that reused maps draw the same pixels. The
  Metal renderer reuses its maps by the same rule.

- `examples/glass_modes.rs`: each way of drawing glass beside the path
  tracer, for clear, tinted, frosted, dispersive and thin-walled glass.

- **Marks** (`threers::marks`), the primitives a data-on-a-map application
  like kepler.gl is built from. Each layer is one instanced draw, with every
  property read from its own column, and runs on WebGL2 as well as WebGPU.
  `PointMarks` draws antialiased circles with strokes. `SegmentMarks` draws
  lines of any width, polylines with mitred joints (so translucent ones do not
  darken where segments meet) and arcs (wgpu has no line width; these are
  quads expanded in screen space). `ShapeMarks` instances any geometry, and
  `TextLabels` draws signed-distance-field text with collision culling.
  `DensityMarks` is a heatmap rendered through an offscreen half-float field.
  Sizes are in pixels or world units.
  - **Columns** take partial updates (`update_*(start, …)`); only the changed
    rows are uploaded.
  - **`f64` positions** are drawn relative to a movable `Origin` as split
    high/low `f32`s, so geometry ten million units out keeps quarter-unit
    spacing.
  - **`PolygonMarks`** fills and extrudes GeoJSON-style polygons with holes,
    every feature in one draw, with per-feature colour, elevation, custom
    values and picking; changing one feature re-uploads only its vertices.
    Fills are depth-biased so they do not fight a coincident ground mesh.
  - `DensityMarks` sums its field at reduced resolution (twelve texels a
    kernel radius, within 2.5 % of full resolution), which cut 200,000 kernels
    from 10.6 ms to about 4 ms. `examples/marks_bench.rs` times every layer at
    kepler.gl sizes.
  - **For tile and globe renderers.** Custom mark kinds (`CustomShader`,
    `MarkKind::Custom`, with their own bind groups). `TextureAtlas`,
    `SpriteMarks` and `ImageMarks`. Positions local to a per-layer origin
    (`set_local_positions`, half the memory), on by default for polygons and
    paths. `PolygonMesh` and `PathMesh`, built off-thread and converted to
    bytes for Web Workers. `release_gpu` and `gpu_bytes`. Tile math
    (`geo::TileId`, `covering_tiles`, `MarkView::units_per_pixel`). Per-layer
    `clip` rectangles and `depth_order`. Dashes, caps and casings on
    `SegmentMarks`. A `CollisionIndex` shared by labels and sprites, with
    label `stickiness`. `pick_rect`, feature ids and `highlight`.
    `ShaderHooks::transition` and `combine`. Globe support: sphere
    conversions and great circles, `GlobeGround` meshes extruded radially,
    great-circle segments, `ImageMarks::sphere`, horizon culling,
    `GlobeControls`, and `fit_clip_planes` on both controls. Opt-in
    `tone_mapped` marks through `MarkView::from_renderer`.
  - **Reversed depth** (`Renderer::set_reversed_z`, JS `setReversedZ`) for
    planet-scale views. Off by default. Shadow maps stay standard, and
    everything that reads the view's depth back converts it: post effects,
    glass and screen-space reflection, TAA, the headless depth of field and
    motion blur, and `Ssao::set_reversed_depth`. Marks follow through
    `MarkView::from_renderer`. `tests/reversed_z.rs` renders each feature
    both ways and requires the same frame. Each conversion is checked by a
    test that fails without it.
  - Labels set along a line (`TextLabel::along`), and mitred joints on
    arched and great-circle paths.
  - Fixed: a pipeline variant for a material's depth state did not record
    its recipe, so `cull_variant` silently kept the stock culling for
    materials with non-default depth state and a non-default side.
  - Fixed: the composer's post effects (`OutlinePass`, `SaoPass`, `SsrPass`,
    `BokehPass`, through `Renderer::run_postfx`) read reversed depth as
    standard, so outlines vanished under `set_reversed_z`. Tested for the
    composer passes, cube-face capture, `QuadView` panes with and without
    MSAA, and TAA under a moving camera.
  - All of a `MarkRenderer`'s layer uniforms share one buffer, written once
    a frame, and each layer binds it at a dynamic offset. A map of hundreds
    of tile layers no longer makes hundreds of buffer writes.
    `MarkRenderer::gpu_bytes` reports the buffer.
  - `CollisionIndex::begin_frame(&view)` keeps last frame's boxes while the
    view and the layers hold still, and label and sprite layers then skip
    placement entirely. A layer that changes takes back its own boxes and
    places again. Call `invalidate()` after removing or reordering layers.
  - Fixed: `SpriteMarks` collision ignored moved, resized, re-imaged or
    turned icons until the view changed. Changing `TextLabels::font_size`
    after the first frame never laid the text out again.
  - Point quads reach half a pixel past the edge rather than a whole one
    when single-sampled. Up to a third fewer fragments for small points,
    with the same pixels. Multisampled targets keep the full pixel their
    off-centre samples need.
  - `DensityMarks::auto_max` scales the heatmap to the densest pixel in
    view. A chain of render passes finds the maximum on the GPU, so nothing
    is read back and it runs on WebGL2.
  - **`MapView`** keeps a map's camera, `MapControls`, `Origin` and shared
    `CollisionIndex` in step. Each frame it rebases the origin as the view
    travels, fits the clip planes, builds the `MarkView` and begins the
    collision frame. It also offers `covering_tiles`, `zoom`, `ground_point`
    and `look_at` in absolute coordinates.
  - `GlobeControls` works in `f64`. `apply_relative` places the camera
    against an `Origin` at the look-at point (`origin()`), so a globe view
    stays exact down to a metre off a real planet.
    *(Breaking within this release: `centre`, `radius` and `distance` are
    now `f64`, and `update` no longer moves the camera.)*
  - `PolygonMarks::to_buffer_geometry` and `ShapeMarks::to_instanced_mesh`
    hand a layer to the scene renderer to be lit, cast and receive shadows,
    and go through post effects. Positions are relative to an `Origin`, and
    colours become vertex or instance colours.
  - Text:
    - multi-channel distance fields (`GlyphField::Msdf`) keep corners sharp
      at any size;
    - `raster_size` sets the atlas resolution;
    - `add_fallback` adds faces for missing characters;
    - `add_icon` and `TextLabel::with_icon` make icon-and-text symbols that
      collide as one.
  - Fixed: TrueType composite glyphs, which include most accented letters,
    loaded as empty outlines. They are now assembled from their components.
  - Fixed: `decode_png` refused 1, 2 and 4-bit greyscale and palette images,
    which is how OpenStreetMap serves open sea.
  - `examples/map_demo.rs`: a small map on `MapView`. It has OpenStreetMap
    raster tiles fetched in the background and cached on disk (or any
    `--tiles` template, or procedural tiles offline), European cities as
    icon-and-name symbols, and an auto-scaled heatmap.
  - `TextureAtlas` repeats each image's edge into its gutter. Images laid
    edge to edge, such as map tiles and opaque sprites, no longer show dark
    seams.
  - `TextureAtlas::write` replaces an image in place. Uploads send only the
    regions that changed, each on its own, without copying the whole atlas
    first. Panning the map demo used to repack and re-upload every tile
    whenever the view crossed one: p95 109 ms and worst 1.2 s per frame. Its
    tiles now live in a fixed pool of slots: p95 12 ms, worst 33 ms.
  - **Geographic data.** `marks::geo` projects longitude and latitude to Web
    Mercator and back, with the per-latitude height scale.
    `marks::geojson::FeatureCollection` parses GeoJSON, and `shapes()` turns
    it into points, paths, polygons and outlines for the marks, with each
    row mapped back to its feature.
  - **`ShaderHooks`** splice WGSL into every mark's vertex and colour stages,
    with per-frame `params`. `ShaderHooks::range_filter` filters rows on the
    GPU without rebuilding a pipeline, and `ShaderHooks::trail` animates trips:
    segments cut to a time window with the head exactly at the clock.
  - **GPU picking** (`MarkRenderer::pick`) reads `(layer, row)` back from an id
    pass, asynchronously, optionally occluded by the frame's depth.
  - Marks draw over a rendered frame and depth-test against it
    (`MarkTarget::over_frame`, `HeadlessRenderer::mark_target`), backed by the
    new `Renderer::overlay_attachments`.
  - Camera-facing marks are depth-tested `style.depth_lift` of their own
    sizes nearer the camera (2 by default), so a point or line resting on the
    ground draws whole instead of half-sunk into it.
  - `examples/marks_map.rs`: a kepler.gl-style map using all of the above.
- **`ShaderMaterial` on `InstancedMesh`.** It used to draw with the built-in
  instanced pipeline whatever its shader. Now it gets its own pipeline on the
  instanced layout. Without a vertex stage it draws every instance at its
  matrix and tint. A custom stage takes `InstancedCustomIn` and starts from
  `standard_vertex_instanced(instanced_in(in), vid, iid)`.
  **`InstancedMesh::set_custom`** adds two per-instance columns of one to
  four floats, read as `in.icustom0`/`in.icustom1`. They are vertex
  attributes, so they work on WebGL2.
- **`MapControls`**: drag the ground (the grabbed point stays under the
  cursor), zoom toward the cursor, tilt and turn, from a look-at point,
  distance, bearing and pitch.
- **`triangulate_polygon`** and **`polygon_points`** in `threers::math`: ear
  clipping of a polygon with holes, from GeoJSON-style rings.
  **`PolygonGeometry`** fills such a polygon, or extrudes it into a closed,
  correctly wound solid.
- **Volumes** (`threers::volume`): **`Data3DTexture`** (three.js's, also at
  the crate root) and **`GpuVolume`**, a 3D texture written whole or one
  region at a time with `write_region`, so a volume can stream in as its
  chunks arrive. Integer formats come back exactly, up to 64-bit labels
  (`Rg32Uint`). **`SliceRenderer`** draws an arbitrary plane through a volume
  into a rectangle of the target, over a frame the `Renderer` already drew,
  so one target can hold a 3D pane and cross-sections. The colour is a WGSL
  `slice_color` hook: window/level grey (`WINDOW_LEVEL`) and a colour per
  label (`LABELS`) are built in. `plane_to_voxel` builds the slice camera,
  with image rows running down or up. Both built-in shaders translate for
  WebGL2 in the tests; window/level has also been checked on WebGL2 in a
  browser. `SliceRenderer::shader_source` returns the module for checking a
  colour function without a device.
  `SliceRenderer::with_target` takes a **`SliceTarget`**: the blend state
  (or none, replacing colour and alpha) and an optional depth attachment,
  where every fragment sits at depth 0 so the first draw to reach a pixel
  keeps it (fine data first, coarse data fills gaps). `draw_in_pass` draws
  within a pass you began, so many draws share one pass.
  `SliceRenderer::with_lookup` adds a lookup table the colour function reads
  (`slice_lookup`, a `texture_2d<u32>`: a palette, or a sorted set of labels
  to search), bound with `create_binding_with_lookup`.
  `ShaderMaterial::volumes` binds two volumes to a custom shader as `u_vol0`
  and `u_vol1`. This works everywhere except WebGL2, where the rest of the
  layout already uses all sixteen texture units
  (`caps::supports_shader_material_volumes`).

- **MeshPhysicalMaterial follows three.js r165's `lights_physical_fragment`,
  on wgpu (WebGPU and WebGL2 included) and Metal.** The reflectance at normal
  incidence comes from `ior`, `specularIntensity` and `specularColor` (new),
  and at grazing from `specularIntensity` (`specularF90`), where it was a flat
  0.04 and 1. The thin film's Fresnel mixes into the specular after
  `F_Schlick` and into the environment's multiscattering, as three.js does,
  with three.js's `iridescenceThicknessRange` (`[min, max]`, the film at
  `max` without a thickness map). Sheen takes three.js's saturated Neubelt
  visibility and its environment term `iblIrradiance · sheenColor ·
  IBLSheenBRDF`. Anisotropy uses three.js's `alphaT = mix(α², 1,
  anisotropy²)` against `α` along the bitangent, its anisotropy vector, and
  `getTangentFrame`'s unnormalised frame, and bends the environment's normal
  as `getIBLAnisotropyRadiance` does. The clearcoat reads the unperturbed
  normal (the base's normal and bump maps are under it) and reflects the
  environment pulled toward that normal by roughness². Emissive now sits
  under the sheen and clearcoat, which compose over it in that order. Metal
  had none of this — it shaded Physical as Standard, with no clearcoat,
  sheen, iridescence or anisotropy — and now matches wgpu on each to within
  three levels inside the surface, with and without an environment (the
  anisotropic highlight to a mean of 0.3, its screen-space frame jumping on
  triangle edges). Five parity scenes — `physical-specular`, `-sheen`,
  `-iridescence`, `-anisotropy`, and `physical-env` under an environment —
  match three.js, and a flat dielectric's reflectance is checked against
  r165's formulas on both backends.
- **MeshPhysicalMaterial's maps**: `specularIntensityMap`, `specularColorMap`,
  `clearcoatMap`, `clearcoatRoughnessMap`, `clearcoatNormalMap` (with
  `clearcoatNormalScale`), `iridescenceMap`, `sheenColorMap`,
  `sheenRoughnessMap`, `transmissionMap`, `thicknessMap` and `anisotropyMap`,
  each read on three.js's channel, with its own sampler and uv transform, on
  wgpu and Metal; `iridescenceThicknessMap` now runs between the range's two
  ends and reaches Metal. Each is checked to draw exactly what its scalar
  does with the texel folded in, on both backends. The glTF loader fills them
  from `KHR_materials_*` as three.js's does. Metal's material samplers are a
  pool of ten bound once per distinct filter-and-wrap, which is what lets
  every map keep its own within Metal's sixteen sampler slots.
- **MeshPhysicalMaterial's properties from JavaScript**: `ior`,
  `reflectivity`, `specularIntensity`, `specularColor`, `sheen`, `sheenColor`,
  `sheenRoughness`, `iridescence`, `iridescenceIOR`,
  `iridescenceThicknessRange`, `anisotropy`, `anisotropyRotation`,
  `transmission`, `thickness`, `attenuationDistance`, `attenuationColor`,
  `dispersion` and every map above are live properties with three.js's
  defaults; the shim's Physical material forwarded `clearcoat` and
  `clearcoatRoughness` and dropped the rest. wasm gains `setPhysical`,
  `setSheenColor`, `setSpecularColor`, `setAttenuationColor`,
  `setClearcoatNormalScale` and `setNamedMap` / `setNamedMapData` for any
  map slot by its three.js name.
- **`Renderer::set_specular_antialiasing`** (and on `MetalRenderer`;
  `setSpecularAntialiasing` from JavaScript, `renderer.specularAntialiasing`
  in the shim): geometric specular antialiasing on Standard and Physical
  highlights, off by default — see *Changed*.
- **three.js's transmission pass, on wgpu (WebGPU and WebGL2 included) and
  Metal.** A visible transmissive material (`transmission > 0`) makes the
  renderer draw the opaque scene first, linear and untone-mapped, into a
  half-float target with a full mip chain, then draw the back faces of
  double-sided transmissive materials over it, as r165's
  `renderTransmissionPass` does; `getIBLVolumeRefraction` then reads it along
  the ray refracted through `thickness` — bicubic, at the mip three.js picks
  from roughness and `ior`, three samples apart under `dispersion` — and
  attenuates it by `attenuationColor` over `attenuationDistance`. Transmission
  replaces the diffuse light, the specular stays on top, and
  `transmissionMap`/`thicknessMap` scale the two. Transmissive materials sort
  between opaque and transparent, drawn opaque, as three.js's transmissive
  list is. The previous approximation — the environment refracted plus a rim
  term — is gone. Plain, instanced and skinned meshes all take the capture.
  On Metal every view captures its own: stereo (layered, or side by side in
  viewports) gets a slice per eye, sized to the eye's viewport, so each eye
  refracts what it sees — each matches the mono render from its camera
  exactly.
  A `transmission` parity scene of clear, rough tinted and dispersive glass
  over coloured boxes matches three.js to six pixels of 480,000, and Metal
  matches wgpu.
- **`RectAreaLight` through three.js's LTC tables**, on wgpu and Metal: r165's
  `LTC_Evaluate` over `ltc_1`/`ltc_2` (bundled, uploaded once as a half-float
  2-layer array), for the specular and the diffuse alike, where both backends
  used a representative-point approximation. A `rect-area` parity scene
  matches three.js to six pixels, and Metal matches wgpu on smooth, rough and
  metal spheres under two scaled, rotated lights. The shim exports
  `RectAreaLightUniformsLib` and `RectAreaLightTexturesLib` (their `init()`
  has nothing left to do).
- **Second to fourth uv sets and `Texture.channel`**, on wgpu and Metal:
  `uv1`, `uv2` and `uv3` reach the shader, and each map reads the set its
  texture's `channel` names, as three.js r165 does. From JavaScript
  `texture.channel` is a live property.
- **Metal reads the `tangent` attribute** for normal maps, the clearcoat
  normal map and anisotropy, as wgpu does, where it always derived the frame
  from screen-space derivatives.
- **`Renderer::set_reflective_transparency`** (and on `MetalRenderer`;
  `setReflectiveTransparency` from JavaScript, `renderer.reflectiveTransparency`
  in the shim) — see *Changed*.
- **three-bvh-csg in the browser again.** `web/bvh-csg-addon.js` re-exported
  `bvh-csg-impl.js`, which 0.0.4 deleted with the JavaScript CSG port it
  wrapped, so every bvh-csg page failed to load. The boolean now runs in wasm
  (`csgEvaluate`, the Rust port), and `bvh-csg-impl.js` keeps three-bvh-csg's
  `Brush`, `Operation`, `OperationGroup` and `Evaluator` as the library has
  them — `useGroups` and `consolidateGroups` with its material list,
  `attributes`, operation arrays, `evaluateHierarchy` and its dirty tracking —
  plus `computeMeshVolume`. The nine bvh-csg parity scenes match three-bvh-csg
  from nine viewpoints each (0.01% at worst), where all nine failed.
- **A `BatchedMesh` draw is cast at `matrixWorld · instanceMatrix`** from the
  mesh's matrix as it stands, as three.js's is.
- **`BufferGeometry::to_non_indexed`**, three.js's `toNonIndexed`
  (attributes and morph attributes expanded, groups kept).
- **`CsgEvaluator` carries three-bvh-csg's `attributes`** (`position`, `uv`,
  `normal` by default) and, with `use_groups` (now on by default, as in
  three-bvh-csg), a group per source group; `evaluate_ops` returns one result
  per operation.
- **`WebGLRenderer.create` takes three.js's parameters**: `create(canvas,
  { antialias })` or `create({ canvas, antialias })`. See *Changed*.
- **MeshToonMaterial's `gradientMap`**, on wgpu (WebGPU and WebGL2
  included) and Metal: each light's irradiance times the map's rgb at
  `(n·l/2 + ½, 0)`, as r165's `getGradientIrradiance` reads it — any ramp,
  coloured or uneven, filtered as the texture asks and decoded by its colour
  space. `ToonMaterial::steps`, which stood in for an evenly spaced grey ramp
  by its texel count, is replaced by `gradient_map` (see *Changed*). The shim
  binds `gradientMap` (it was ignored, so every toon material drew two-tone),
  Python and Node gain `Material.toon`, and a `toon-gradient` parity scene of
  a nearest colour ramp and a linear grey one matches three.js to four pixels
  of 480,000; both backends are checked against the formula texel by texel.
- **Textures from Python and Node.** `Texture` takes RGBA8 pixels or a PNG
  (`Texture.from_file` / `Texture.fromFile`), in sRGB or linear, with
  three.js's wrap, mag/min/mipmap filters, `offset`, `repeat`, `rotation`,
  `center`, `channel` and `flip_y`; `material.set_map(slot, texture)` (and
  `get_map`) fills any map slot three.js names, `None`/`null` empties it.
  `Material.physical` takes the rest of MeshPhysicalMaterial's options:
  `specular_intensity`, `specular_color`, `sheen`, `sheen_color`,
  `sheen_roughness`, `iridescence`, `iridescence_ior`,
  `iridescence_thickness_range`, `anisotropy`, `anisotropy_rotation`,
  `attenuation_color`, `attenuation_distance` and `dispersion` (camelCase from
  Node, `iridescenceIOR` as three.js spells it).

- **`alphaMap`** on Basic, Lambert, Phong, Toon, Standard, Physical, Matcap and
  Sprite materials, on wgpu (WebGPU and WebGL2 included) and Metal: its green
  channel scales the opacity after the colour map and before `alphaTest`, as
  three.js's `alphamap_fragment` does, and a shadow caster's `alphaMap` cuts its
  shadow too, as three.js's depth material takes it. Checked against three.js on
  every kind and light type, both native backends and both browser backends.
  `Material::map_slot_mut` names a material's map slots as three.js does, and
  from JavaScript `material.alphaMap = null` (or any map slot) empties the slot,
  where it used to leave the old texture bound. `metalnessMap` reaches Standard
  and Physical from JavaScript; it had no binding.

- **`lightMap`, `bumpMap` and `specularMap`**, on wgpu (WebGPU and WebGL2
  included) and Metal, as three.js r165 shades them. `lightMap` ×
  `lightMapIntensity` adds to the indirect irradiance on Lambert, Phong, Toon,
  Standard and Physical (`lights_fragment_maps`), under the `aoMap` like the
  rest of the indirect light, and on Basic replaces its indirect 1 with texel ×
  intensity ÷ π. `bumpMap` × `bumpScale` perturbs the normal through
  `perturbNormalArb` on Lambert, Phong, Toon, Standard and Physical, and is
  ignored under a `normalMap`, as three.js does. `specularMap`'s red channel
  scales Phong's specular (stored on Basic and Lambert too, where three.js uses
  it only for an environment map). Each is checked against three.js's formulas
  on both native backends and in both browsers, and all are live properties
  from JavaScript.

- **`envMap` on Basic, Lambert and Phong**, with `combine` (`Multiply`, `Mix`,
  `Add`), `reflectivity`, `refractionRatio`, `envMapRotation` and the cube's
  `mapping` (`CubeReflectionMapping` / `CubeRefractionMapping`), on wgpu
  (WebGPU and WebGL2 included) and Metal, as three.js r165's `envmap_fragment`
  computes them: the view ray (the camera's forward axis for an orthographic
  camera) reflected or refracted about the normal, looked up mirrored in x
  (`flipEnvMap`) through `envMapRotation` in three.js's sign convention, and
  weighted by `specularStrength × reflectivity` — so a `specularMap` now masks
  reflections too. Checked case by case against three.js on both native
  backends and both browsers, and an `envmap` parity scene of reflecting,
  refracting and rotated spheres over a marked cube matches three.js to 70
  pixels of 480,000. Basic reflects per fragment where three.js reflects per
  vertex; the two agree on any flat face. `Material::map_slot_mut`'s
  `clearMap("envMap")` empties it.
- **`CubeTexture` and `CubeTextureLoader` from JavaScript.** The shim's
  `CubeTexture` was a stub — `CubeTextureLoader` stored the six images and
  nothing ever uploaded them, and `new CubeTexture()` built a cube from
  `undefined`. It now takes three.js's constructor, uploads its images when all
  six are present and again on `needsUpdate`, re-binds wherever it is held (a
  material's `envMap`, `scene.environment`), and has live `mapping` and
  `colorSpace` (the loader marks its result sRGB, as three.js's does) and
  `clone()`. `scene.environment = null` now removes a cube environment; it was
  left bound.

- **three.js's mipmapped filter modes.** `minFilter` has six values and was
  folded into two: every texture was sampled through its mip chain, and how
  levels blended followed the in-level filter. `Texture` and `CubeTexture` now
  carry `mipmap_filter` — `Some(Linear)` (`*MipmapLinearFilter`, three.js's
  default), `Some(Nearest)` (`*MipmapNearestFilter`) or `None` (`LinearFilter` /
  `NearestFilter`, level 0 only) — on wgpu and Metal alike, and a render
  target's texture has none, as three.js's has `LinearFilter`. From JavaScript
  `setFilters` takes three.js's own constants, so every `magFilter`,
  `minFilter`, `wrapS` and `wrapT` arrives as set, cube textures' filters too.
  Metal's environment sampler was looked up but never created, so the scene
  environment and a material's `envMap` were read through whatever sampler the
  material map had; both now have their own, trilinear by default.

- **`Points` on wgpu are drawn as three.js draws them.** Every point was its
  own draw — a quad built on the CPU, sized against a hard-coded 800-pixel
  canvas, shaded as a Basic material — so vertex colours were dropped,
  `sizeAttenuation` was not three.js's formula, and a cloud of a hundred
  thousand points was a hundred thousand draws. A `Points` object is now one
  instanced draw over its own vertex buffer (morph targets included), each
  point expanded to a quad in the vertex stage: `size × pixelRatio` pixels of
  the real viewport, attenuated by `(height / 2) / −z` under a perspective
  camera, as `points_vert` does. Opaque points write depth; transparent ones
  blend, sorted with the other transparent draws, in the encoded colours on an
  sRGB target. `PointsMaterial` gains `map` and `alphaMap`, sampled at three.js's
  `(gl_PointCoord.x, 1 − gl_PointCoord.y)` through the map's transform, on wgpu
  and Metal; Metal's point map was upside down and ignored the transform, it
  blended transparent points in linear light, and it discarded alpha-0
  fragments that three.js keeps at `alphaTest` 0. From JavaScript, `size`,
  `sizeAttenuation`, `map` and `alphaMap` are live.
- **`MeshMatcapMaterial.map`**: the colour map multiplies the diffuse colour
  before the matcap, as `meshmatcap.glsl` does, on wgpu and Metal. Metal's
  matcap texture had shared the colour map's slot, and its alpha was read as
  the material's.
- `setOpacity` from JavaScript reaches every material with an opacity; Points,
  Sprite, Matcap, Normal, Line and Depth materials ignored it.

- **Image-based light follows three.js r165 on Standard and Physical.** A
  material's own `envMap` now replaces `scene.environment` for it, with its own
  `envMapIntensity` and `envMapRotation`; without one it reads the scene's at
  `scene.environmentIntensity` and `scene.environmentRotation` (all new).
  Every cube texture lighting a Standard/Physical material is prefiltered
  through PMREM, once and cached, as three.js's `WebGLCubeUVMaps` does — the
  renderers used to read a raw environment cube through its box-filtered mips,
  so a rough metal showed sharp edges through its blur, and mirrored in x
  against three.js's PMREM. Metal now reads the same CubeUV atlas with a port
  of three.js's `textureCubeUV`, and agrees with wgpu to 0.0001 rms where it
  was allowed 0.08. On wgpu the environment's parameters, intensity and
  rotation moved from the frame uniform into the environment's own group, so a
  draw switches environment by switching that group. Checked by case against
  three.js on both backends, and a `pbr-env` parity scene — mirrors, a rough
  metal, a diffuse sphere, a material's own map and rotations — matches three.js
  to 0.08%.
- **Cube textures were cached by address**, on both backends: a cube made after
  another was dropped could land at its address and be drawn with its faces.
  `CubeTexture` now has a stable `id`, as `Texture` does, and the caches key on
  it, its format and its size.

- **Quad and n-gon faces survive loading** — `BufferGeometry::polygons`
  records how many consecutive triangles each source polygon became. The OBJ,
  PLY, Collada, FBX, USD and Alembic loaders fill it in, `merge_geometries`
  keeps it, and wireframes (wgpu, `WireframeGeometry`, SVG) and `EdgesGeometry`
  draw polygon outlines instead of every fan diagonal. USD export writes the
  polygons back, so a quad leaves as `faceVertexCounts = [4]`; checked in both
  directions against `usdcat`. The built-in generators record their own quads
  too — plane, box, cylinder and cone sides, sphere, torus, torus knot, lathe,
  ring, capsule, parametric, tube, extrude walls, kirigami and NURBS grids — so
  their wireframes draw cells rather than diagonals; cells that collapse at a
  pole or tip stay triangles. Instanced and skinned meshes now draw wireframe
  outlines on wgpu (they were drawn filled), and `set_polygons` bumps
  `geometry_version`, so a table changed after upload reaches the GPU. `sketch`
  follows the window's scale factor and takes a `QuadView`. Counting triangles rather than corners means the
  table can be checked against the index however the polygon was triangulated,
  and `polygon_loop` recovers the corners by chaining the edges only one of the
  polygon's triangles uses. New `PolyMesh::geometry()` for Alembic.

- **Quads through the rest of the mesh tooling.** `BufferGeometry::faces()`
  yields one corner loop per recoverable polygon and triangles otherwise, and
  every writer that can hold a polygon uses it: OBJ and OFF export write one
  face per quad or n-gon, a new `geometry_to_ply` / `Solid::to_ply` does the
  same, and each round-trips through its loader polygon for polygon; assimp
  reads the files as n-gons too. Raycast hits carry `polygon_index`
  (`polygonIndex` in JS). `infer_quads` recovers quads from triangle-only
  meshes — glTF, STL — pairing triangles by Blender's join error under fold,
  convexity and corner-angle limits, changing only the index: a shuffled grid
  comes back cell for cell, a glTF sphere gives 336 quads and leaves its 48 pole
  triangles alone, and a UV seam blocks a pair while a flat-shading normal split
  does not. `subdivide(levels)` is Catmull–Clark on any geometry without the
  `usd` feature, its cage code moved out of the USD loader, which now numbers
  edge points deterministically; in JS as `geometry.subdivide(levels)`. The
  shim's `OBJExporter` and `PLYExporter` take `{ polygons: true }` to write
  n-gon faces, and write triangles by default, as three.js does. `QuadGeometry` carries brep provenance: a
  plane, or the bilinear patch as a degree-1×1 NURBS.

- **`QuadGeometry`** — a quad from four arbitrary corners. Subdivided, a
  non-planar quad meshes as a bilinear patch with analytic normals
  (`∂P/∂u × ∂P/∂v`); UVs, vertex order and indices match `PlaneGeometry`
  exactly when given its corners. Each cell splits along the diagonal whose two
  triangles fold least, not the shorter one: on a concave dart the shorter
  diagonal lies outside the quad. Degenerate corners never produce NaN.
  `is_planar` / `is_convex` helpers, and `WebGeometry.quad` in JS.

- **Viewports, scissor and a CAD-style quad view** — `Renderer::set_viewport`,
  `set_scissor`, `set_scissor_test`, `set_pixel_ratio`, `set_auto_clear` and
  `clear`, following three.js: bottom-left origin in logical pixels, and the
  clear limited to the scissor while the test is on, so several renders share
  one target. WebGPU has no partial clear, so a scissored fullscreen triangle
  writes exactly the value a clear would. With nothing set, the old path runs
  and the output is byte-identical. MSAA works in a sub-rectangle: the pane is
  drawn into a pane-sized multisampled target and composited into its rectangle
  with a scissored `textureLoad` draw — a draw rather than a copy because surface
  textures usually refuse copies — and comes out identical to a full-target
  render of the same view. TAA and screen-space refraction are off in a
  sub-rectangle, and post-processing applies to the finished image.
  `Renderer::shadow_map()` takes three.js's `auto_update` / `needs_update`; no
  shadow map here depends on the camera, so the quad view draws them once a
  frame and shares them — 2.66 ms of shadow GPU time a frame down to 0.66 ms,
  byte-identical panes. `QuadViewControls` orbits the perspective pane and pans
  and zooms the flat ones about the cursor with one shared scale and centre,
  each ortho camera following only the part of a pan across its own view;
  double-click maximises a pane. `helpers::QuadView` lays
  out top, front and side orthographic panes at one shared scale beside a
  perspective pane — third-angle, Y-up — with gaps, per-pane backgrounds, a
  maximise toggle and `pane_at` hit-testing. Metal has the same API and
  semantics, sharing one planner with wgpu, and the two backends' quad views
  are byte-identical in a cross-backend test; `MetalViewSurface` draws the
  panes into a window and presents once. `QuadView::metal_views` stays as the
  cheaper single-pass option, which keeps MSAA but has no gap colour or
  per-pane backgrounds (`MAX_VIEWS` is now 4). Also on wasm and the three.js
  shim. `cargo run --release --example quad_view`.

- **three.js sizing and multi-view on the web** — the shim's `WebGLRenderer`
  takes `setSize` in CSS pixels times `setPixelRatio`, and gains `getSize`,
  `getPixelRatio`, `getDrawingBufferSize`, `getViewport`, `getScissor`,
  `getScissorTest`, `setClearColor` / `getClearColor` / `setClearAlpha` /
  `getClearAlpha` and a `clear()` that honours the scissor test, so three.js's
  multi-view idiom works unchanged. Also `QuadGeometry`, `QuadView` (and
  `WebQuadView`), and `BufferGeometry.setPolygons` with polygon-aware
  `WireframeGeometry` and `EdgesGeometry`. Browser tests run
  on WebGL2 and, new, on WebGPU under Chrome's own shader compiler:
  `WEBGPU=1 FEATURES=captions TEST=webgpu scripts/wasm-browser-test.sh`.

- **Node and Python packages, quads first** — the two bindings grow from a
  box-and-sphere demo into a usable API, matched name for name. Quads: geometry
  objects with polygon counts, loops and edges, `setPolygons`, `inferQuads`,
  Catmull–Clark `subdivide`, `QuadGeometry`, quad-aware edges and wireframe as
  line segments, raycasts that report the polygon hit, loaders that keep quads,
  and `renderQuadView` with every `QuadView` option plus viewport, scissor,
  auto-clear and `clear`. Around them: the built-in geometry set, six material
  kinds and line material, ambient, directional, point, spot and hemisphere
  lights with shadows, object handles (transform, visibility, parenting,
  removal), `OrthographicCamera`, glTF/GLB, OBJ, STL, PLY, FBX and USD loading,
  `setSize` and tone mapping. Python ships type stubs.

- **Raycasting at three.js r165 parity, and fast.** Hits respect the
  material's side — a `FrontSide` mesh no longer reports its back faces — and
  `near` / `far` are exact under non-uniform scale, where the old approximation
  could drop a valid hit. Lines, points, sprites, instanced meshes (with
  `instance_id`) and skinned meshes are hit-tested; they used to return nothing.
  An `Intersection` carries `face`, `normal`, `uv`, `uv1`, `instance_id`,
  `index` and `distance_to_ray` as three.js's does, plus `barycoord` and
  `polygon_index`. `set_from_camera` takes either camera kind,
  `intersect_object` / `intersect_object_list` match three.js's, and
  `first_hit_only` keeps the nearest hit through the BVH's closest-hit query.
  Bounds and the polygon lookup are cached on the geometry the first time a ray
  needs them, so a ray no longer copies every position or walks the polygon
  table: BVH picking on a 1M-quad mesh went from 2.7 ms to 3.4 µs a ray. Fixes a
  BVH first-hit traversal that opened every missed box under an unbounded `far`.
  `DragControls` picks the nearest object and works with orthographic cameras.
  Struct literals of `Intersection` and `Raycaster` need `..Default::default()`
  or `Intersection::new`.

  The same in every binding. wasm's `WebRaycaster` gains `set`, `near` / `far`,
  line and points thresholds, `firstHitOnly` and `intersectObject(s)` returning
  three.js-shaped hits; Node and Python `raycast` / `raycastFromCamera` take the
  same options and return the same fields. The shim's `Raycaster` stays in
  JavaScript — the wasm scene mirror cannot answer for objects outside a
  `Scene`, nested groups or multi-material meshes — rewritten as a port of
  r165 covering material sides, groups and material arrays, draw range, morph
  targets, skinned, instanced, lines, points and sprites, and checked hit for
  hit against three@0.165 to 10⁻⁹. As in three.js, the shim's `recursive` now
  defaults to true, invisible objects are hit, and `uv` is present only when the
  geometry has one. `Ray` gained r165's methods, which also fixes `DragControls`
  calling a missing `ray.intersectPlane`.

- **Geometry groups, material arrays, `drawRange` and morph targets** in the
  core, at three.js r165 semantics, drawn natively by every renderer instead of
  being faked in the JavaScript layer. A mesh with `materials` draws each
  geometry group with its own material — sorted and blended per group, as
  three.js does — on wgpu, Metal, the path tracer and SVG, and the raycaster
  reports the group's `material_index`. Morph targets blend on the GPU before
  skinning, normals and colours included, from a morph texture so WebGL2 runs
  them too (a one-layer texture array reads zeros there, so morph textures
  always have two); the CPU consumers share one morph-then-skin pose. wgpu and
  Metal agree to the pixel on groups, absolute and relative morphs, and skinned
  morphs. OBJ `usemtl` runs, FBX multi-material meshes and USD GeomSubsets load
  as groups over a material array; glTF (sparse targets included), FBX and USD
  morph targets load onto the geometry, and the animation mixer drives them on
  skinned meshes too. `Object3D` gains three.js's `matrix_auto_update`,
  arbitrary (sheared) matrices, all six Euler orders and three.js `look_at` —
  lights turn −Z to the target and everything else +Z, where before everything
  turned −Z. `Raycaster` defaults to layer 0, as in three.js. Skinned meshes cast
  posed shadows on wgpu, where before they cast none, and path-traced sprites
  face the camera.

- **Shapes, paths and text run in Rust**, bit for bit against three.js r165:
  `Path`, `Shape` with holes, `ShapePath.toShapes`, `ShapeUtils` with an exact
  port of three.js's Earcut, `ShapeGeometry`, and `ExtrudeGeometry` with bevels,
  steps, `extrudePath`, cap and side groups and a custom `UVGenerator`.
  `FontLoader`, `Font` and `TextGeometry` extrude real typeface.js glyphs, where
  the JavaScript layer used to return a plane; the JavaScript Earcut dependency
  is gone. The JSON reader garbled every non-ASCII string — each UTF-8 byte
  became its own character — and rejected `\u` escapes; both are fixed.

- **Geometry generators run in Rust**, buffer for buffer against three.js r165:
  `CapsuleGeometry`, `LatheGeometry` (r165 leaves its last meridian's normal
  unnormalised, and so does this), `TubeGeometry` with three.js's Frenet frames
  from any curve, `ConvexGeometry` / `ConvexHull` as a QuickHull port with
  three.js's face order, a real projecting and clipping `DecalGeometry`, and new
  `RoundedBoxGeometry` and `TeapotGeometry`. The last four used to return a box,
  a box, a box and a sphere. `EdgesGeometry` and `WireframeGeometry` are
  generated in Rust for any geometry and follow its groups.

- **Exporters run in Rust.** OBJ, STL (ASCII and binary) and PLY (ASCII and
  both binary byte orders) write output byte-identical to three.js r165's
  exporters, world and normal matrices included, and keep `{ polygons: true }`.
  `GLTFExporter` exports the whole scene graph — binary GLB, materials,
  `onlyVisible`, `parseAsync` — where it used to return JSON with no buffers.
  Exporting a `Scene`, `Group` or `Mesh` from JavaScript produced nothing,
  because only `Object3D` had `traverse`; every scene-graph class now has
  `traverse`, `traverseVisible`, `traverseAncestors` and `getObjectBy*`. The
  OpenSCAD exports share these writers, so their PLY declares `vertex_index`
  and their STL normals use three.js's formula.

- **Helpers, KTX2, Draco and noise run in Rust.** Light, camera, arrow, box,
  vertex-normal, plane, skeleton and rect-area-light helpers draw three.js
  r165's exact geometry; the last three, and `VertexNormalsHelper`, were empty
  stubs. `KTX2Loader` reads zlib supercompression, mip chains and BC1/BC7
  blocks through the crate's decoder instead of accepting RGBA8 only, and
  `DRACOLoader` uses the crate's Draco decoder, which matches Google's on every
  fixture, so Google's decoder no longer ships. `SimplexNoise` is three.js's
  seeded implementation, bit for bit.

- **Post-processing runs in Rust.** `EffectComposer` and every three.js r165
  pass — UnrealBloom with its five-mip chain, SSAO with all five outputs, SAO,
  SSR, Outline with hidden edges, glow and pulse, real SMAA, TAA and SSAA, LUT,
  Bokeh, Afterimage, Glitch, DotScreen, Halftone, Film, Output, Mask and
  ClearMask, Texture and CubeTexture, RenderPixelated, RenderTransition and
  AdaptiveToneMapping — are `threers::postprocessing` passes shared by native
  code and the browser, and the JavaScript classes are thin wrappers whose
  `uniforms.X.value` and properties forward to Rust. The CPU and GLSL fallbacks
  are gone, about 1,950 lines of them with the JavaScript pass sequencing. FXAA
  had been a CPU copy with no anti-aliasing at all and is now an exact port, as
  is DigitalGlitch; a stricter byte check than the parity threshold shows FXAA,
  the glitch copy and SSR going from far off three.js to matching it, and every
  post-fx parity scene is pixel-perfect or within one pixel. `seedRandom` seeds
  one random stream shared by JavaScript and the passes, drawn in three.js's
  order. Native: `RenderPass::new()`, and `OutputPass` replaces
  `ToneMappingPass`. Passes written in JavaScript are no longer accepted.

- **`GLTFLoader` is three.js r165's, in Rust** (`src/loaders/gltf`), and builds
  the scene graph directly: the JavaScript loader wraps the nodes Rust made
  rather than parsing JSON and rebuilding everything. Multi-primitive meshes,
  every attribute, sparse, normalised and quantised accessors, morph targets,
  skins, cameras, punctual lights, every `KHR_materials_*` extension r165
  supports, texture transforms and samplers, Draco, meshopt, GPU instancing,
  KTX2 where it can be decoded, animations, `userData`, `parser.associations`
  and `getDependency` — checked field for field against three.js's own loader
  on the repo's fixtures and 39 Khronos sample models. Objects moved between
  scenes keep their nodes, and `renderer.render` accepts a root that is not a
  `Scene`, as three.js does. The native `add_to_scene` used to leave a skin's
  bones pointing into the arena it was loaded into.

- **Animation is three.js r165's animation system, in Rust**
  (`threers::animation::three`): clips and every keyframe track type, discrete,
  linear, quaternion and cubic interpolants with each ending, glTF cubic
  splines, `PropertyBinding` paths (materials, morph targets by name, bones,
  uuids, and any JavaScript property), additive blending, fades, cross-fades,
  warps, loop and finish events dispatched mid-update as three.js does, and
  object groups. It is replayed against three.js run in node over 16 mixer
  scenarios and 377 frames, to 10⁻⁶. The browser `AnimationMixer` runs in wasm
  and writes animated values back into the JavaScript objects once per update;
  the 711-line JavaScript mixer is gone. The `color-track` parity scene went from
  13.44% different to identical: the old binding overwrote a material's `Color`
  with a number. The existing native `AnimationMixer` plays on the new engine,
  and weights below one now blend toward the property's original value, as in
  three.js.

- **Curves are three.js r165's, in Rust**, f64: every 2D and 3D curve,
  `CurvePath`, and the base class's arc-length machinery — `getLengths` with
  three.js's caching, `getUtoTmapping`, `getPointAt`, `getTangent(At)`,
  `computeFrenetFrames` — plus `toJSON` / `fromJSON` / `clone`, agreeing with
  three.js to 10⁻¹². The base `Curve` used to return (1, 0, 0) as every tangent.
  A JavaScript `Curve` subclass works wherever a built-in one does, Rust calling
  back into its `getPoint`, including as a `TubeGeometry` path or an
  `ExtrudeGeometry` `extrudePath`, which now take the curve itself. `Path` and
  `Shape` are `CurvePath`s.

- **`TransformControls` is three.js r165's, in Rust.** It was a stand-in that
  moved objects by the average of the pointer's two coordinates and never drew
  its gizmo. The gizmo, picker and helper are built as three.js builds them,
  with translate, rotate and scale, world and local space, axis, plane and XYZE
  handles, snapping and three.js's events, and a drag lands exactly where
  three.js's does. `FlyControls` is three.js's, in Rust; `MapControls` is
  `OrbitControls` with three.js's map defaults; `OrbitControls` gains
  `mouseButtons`, `touches`, `screenSpacePanning` and its events.

- **Math value types are complete.** `Box3` had no `min` or `max`, `Frustum`,
  `Spherical` and `Cylindrical` were stubs and `OBB` ignored its rotation; they
  and the other value types are now three.js r165's own code, checked against
  three.js to 10⁻¹². `Box3.setFromObject` (precise mode too),
  `Frustum.intersectsObject` and `computeBoundingBox` / `computeBoundingSphere`
  on skinned and instanced meshes read posed bounds from Rust.

- **The JavaScript layer drives one exact Rust scene graph.** Every
  `Object3D` — groups, bones, meshes of every kind, lines, points, sprites,
  lights, cameras and helpers — has one Rust node with the same parent and the
  same child order, synced by dirty flags so a frame with nothing changed makes
  no calls, and objects not yet in a `Scene` are mirrored on demand into a
  hidden graph. The old mirror flattened nested groups by *adding* their Euler
  angles and dropped their scale, froze the transform of a mesh inside a group
  at the moment it was added, did not mirror children of meshes at all, split
  multi-material meshes into copies that lost their second UV set, colours,
  skin and morphs, blended morph targets on the CPU in JavaScript every frame,
  and ignored manual matrices, Euler order, `drawRange`, instance updates, light
  positions set per component and a skinned mesh's bind matrix. Now groups and
  material arrays, GPU morph targets, `drawRange`, `instanceMatrix.needsUpdate`
  and `count`, bind matrix and bind mode, `matrixAutoUpdate = false` with shear,
  and all six Euler orders reach the renderer as they are. Raycasting,
  three-mesh-bvh's `acceleratedRaycast` and `DragControls` run in Rust against
  that graph — the 760-line JavaScript raycaster is gone — and still match
  three.js r165 hit for hit, within an f32 bound, with user `raycast` overrides
  called as three.js calls them. `BoxGeometry` is three.js's exact layout, with
  six groups. Camera, spot-light and directional-light defaults follow three.js.
  The `spot-light` and `parametric` parity scenes, which failed to load, now
  match exactly.

- **A three-dimensional fluid** — `FluidSim3d`, as a separate type beside the
  2D one, which is unchanged. `cargo run --features fluid --example fluid3d`
  renders a rising plume of coloured smoke.

  Less changes than you would expect: advection, divergence, the Jacobi sweep
  and the projection are the same equations with a third term, and the stencil
  grows from five points to seven so the pressure divisor is six. Three things
  genuinely differ. **Vorticity becomes a vector** — in the plane the curl is a
  scalar and the confinement force is a quarter turn of its gradient; in space
  the curl is an axis and the quarter turn becomes the cross product it was
  standing in for, `f = ε (N × ω)`. **Buoyancy becomes worth having**, because
  there is an up. And **you cannot simply look at a volume**, so `render`
  marches rays through it, front to back, accumulating emission against
  absorption.

  The volume lives in a 2D texture with its z-slices tiled across it. A render
  pass writes one attachment and a 3D attachment is one slice, so a 64³ volume
  would be sixty-four passes a stage and about eighteen hundred a frame — the
  setup alone would cost more than the arithmetic. Tiled, every stage is a
  single fullscreen pass, which also keeps what the rest of the module is built
  around: no compute stage, so it runs on WebGL2, which has no 3D storage
  textures at all. The price is doing trilinear filtering by hand, with the
  in-tile coordinate clamped half a texel from the edge — without which a fetch
  near a boundary averages in the neighbouring slice's opposite face.

  Two bugs found while building it, both of the kind that still produce a
  plausible plume. The dye advection sampled the **velocity** atlas through the
  **dye's** tile layout, because one uniform described both; each texture now
  carries its own grid. And buoyancy was a uniform upward push rather than one
  proportional to the dye, so it accelerated the empty fluid too and piled the
  whole volume against the ceiling as a rectangular slab.
  `a_splat_appears_where_it_was_put` is the guard: it projects a splat's known
  position onto the screen and checks the brightest pixel is there, which can
  only pass if the voxel addressing, the trilinear fetch and the ray march all
  agree.

  The cube law sets the defaults: a 64³ solver carrying a 128³ dye is 40 MB,
  where the 2D module carries eight times its solver's resolution in dye
  because in a volume that would be five hundred and twelve times the memory.
  An oversized volume halves until it fits the device's texture limit rather
  than failing to allocate.

- **The volume solves with a V-cycle, and picks its own resolution.** A Jacobi
  sweep moves information one voxel, which is a much worse deal in a volume
  than in the plane: the pressure from a puff falls off as 1/r rather than
  log r, and six neighbours share the correction instead of four. Twenty
  sweeps left 82% of a splat's divergence in place; three hundred left 37%.
  One multigrid V-cycle leaves 31%, and two leave 11% — so a cycle beats three
  hundred sweeps for about the cost of twenty, and `FluidConfig3d` now
  defaults to `PressureSolver::Multigrid` where the 2D default stays Jacobi
  (that one is the original's, and there is no original here).

  **Every whole-voxel tap is a `textureLoad`.** The solver reads its
  neighbours at integer offsets, and a voxel of the atlas is one texel — but
  those reads were going through the hand-rolled trilinear path, two filtered
  fetches and a lerp to reach a value whose interpolation weights are always
  1 and 0. The arithmetic is identical and the step is **2.4× faster at 64³
  and 5× at 96³**, which is what makes the bigger grids usable.
  `the_taps_hit_the_voxels_they_name` recomputes a divergence pass on the CPU
  from the field read back, walls included, so the indexing cannot drift.

  **`FluidSim3d::set_resolution`** rebuilds the grids while it runs, and
  `QUALITY_LEVELS_3D` names five sizes — `low` through `extreme`, 24³ to 128³
  — under the same names the 2D fluid uses. A resize empties the volume:
  interpolating one would be eight textures of resampling to preserve a second
  of simulation. The browser demo has the dropdown and two sliders for it, and
  `examples/fluid3d_bench.rs` is where the numbers above come from — it splits
  a volume frame into the solve, the march and the memory at each resolution.

- **The volume steps over the parts of itself that are empty.** Four fifths
  of a settled plume is nothing — 80% of its 8³ bricks, 85% of its 4³ — so a
  ray was spending four fifths of its samples fetching zero. The dye is now
  reduced twice, keeping the largest value of each eight, into a volume at a
  quarter of its side; the march steps over a whole brick wherever that
  largest value is below what it already treats as nothing. **1.9× on the
  march, about 1.3× on a whole frame** at 2560×1440, and the picture is the
  same to three channels in sixty-five thousand.

  The reduction is rebuilt when the dye moves rather than when a frame is
  drawn — it costs the same whatever the render is, so tying it to the render
  would have made a small one mostly reduction. And the skip stops one dye
  voxel short of a brick's far face, because the dye is read with a filter
  and a sample within half a voxel of that face reads part of the next brick;
  that detail is worth the quarter of the skip it costs, taking the worst
  pixel from 38 levels out to 11.

- **The ray march can average itself across frames.** `temporal_samples`
  jitters each frame's samples differently and blends into what came before,
  so 24 samples a frame converge on what 288 give in one: measured against a
  288-sample reference, a single short march is 0.088/255 away and 24
  accumulated frames are 0.019. It resets when the camera moves — nothing
  reprojects, because a volume has no single depth to reproject by — so it
  costs nothing while you drag and sharpens the moment you stop. In the
  browser panel and as a `THREERS_FLUID_ACCUMULATE` key on iOS.

  **It reprojects, so a moving camera keeps its history.** The original
  version threw the accumulation away on any camera movement, on the grounds
  that a volume has no single depth to reproject by. It has no *surface*,
  which is not the same thing: weighting each sample by what it contributed
  — its opacity times the transmittance that reached it — gives an expected
  depth, which for a plume is a few voxels inside the smoke. The march
  reports it on a second attachment, in the same loop, and the resolve
  projects that point through the previous frame's camera. Accumulating
  through a slow orbit lands 0.045/255 from a twelve-times-longer march,
  against 0.104 for a single one — which is what a moving camera used to get.

  **It needed history rejection to be usable at all.** As a plain average it
  was not a feature with a caveat — against a long march of the same moment,
  eight accumulated frames of a rising plume came out *thirty times further
  from the truth* than one short march with no history in it. The resolve now
  clamps the history to within `temporal_clip` standard deviations of its own
  3×3 neighbourhood before mixing, which is the colour clamp a temporal
  anti-aliasing resolve does. Min and max of that neighbourhood is the
  obvious box and is too wide — the march is jittered, so nine samples carry
  nine samples' worth of noise — and clipping to the spread instead, at half
  a sigma, gives **5.2× less ghosting** while keeping nearly all of the
  convergence. The default came off the measured curve rather than a guess;
  both columns are in the `temporal_clip` docs.

  It also turned up an artefact it could not remove, which turned out not to
  be the march's fault. Averaging took the stipple out of a 272³ plume
  (local roughness 36.2 to 20.3) and left a cross-hatch at the scale of the
  empty-space skip's bricks — averaging cannot touch that, because the
  bricks are in the same place every frame. A ray that stepped over a brick
  was landing wherever the skip put it rather than where its next sample was
  due, so it carried on offset from every ray that had not skipped, and
  neighbouring pixels that skipped different amounts sampled at different
  phases. Snapping back to the lattice took the roughness to 16.6 and made
  skipping **exactly** invisible in that regime: 0.000/255 against marching
  every step, where before it was 0.225 over 532 channels.
  `skipping_keeps_rays_on_the_sample_lattice` guards it, and was verified by
  putting the bug back — the first version of that test, which measured
  local roughness, passed with the bug in place and had to be rewritten.

- **Measured and not built: limiting the solve to where the fluid is doing
  something.** The solve is 95% of the step and the pressure from a
  disturbance falls off by a factor of four hundred across the box, so
  smoothing only the active part is the obvious next cut. On a settled plume
  11.8% of voxels carry divergence above a hundredth of the peak — but their
  bounding box is 85.9% of the volume, because a plume is a thin convoluted
  thing threading through it, so no scissor helps. And the field being
  *solved for* is not sparse at all: 45.9% of voxels hold pressure above a
  hundredth of peak and 98.9% above a thousandth, so a mask drawn round the
  divergence would clamp the pressure's tail and change the answer inside
  the mask as well as outside. Best case about 2× on the sweeps for a
  truncation error and a stencil; recorded here so it is not rediscovered.

- **Two silent divergences between the implementations, found and guarded.**
  There are three of this solver now, and what they share is a failure mode:
  a setting one of them does not implement, ignored without complaint,
  producing a picture that still looks like a fluid.

  `CpuFluid3d` had kept `vel.y +=` and a hardcoded two smoothing sweeps
  after `buoyancy_axis` and `multigrid_smooth` were added to the shader. The
  existing parity test runs with buoyancy off — the right way to isolate the
  projection, the wrong way to notice — so nothing failed.
  `the_two_solvers_agree_about_which_way_is_up` points buoyancy along
  `[0.6, 0, -0.8]` and now holds the two to the same answer, which they give
  to within two hundredths of a voxel.

  `MetalFluid::unsupported` names what that backend ignores for a given
  configuration, and `metal_fluid_matches_wgpu` asserts the list is empty
  for the one it compares — otherwise it reports how far apart two different
  algorithms are and calls it agreement. It found `colorful` set on its
  first run. The example prints the list rather than ignoring it quietly.

- **Nothing in the volume is pinned to +Y any more.** Two places were, for
  different reasons. Buoyancy was written `vel.y = vel.y + …`, which is what
  buoyancy usually is but not what the solver has to assume — `buoyancy_axis`
  points it anywhere, and along x, y and −z the dye drifts 4.2 voxels on the
  axis it was given and a ten-thousandth of that on the others. And
  `multiple_splats` drew its vertical component from `rand` rather than
  `rand - 0.5`, so it was never negative and every random push went up; the
  direction is now uniform over the sphere, which is not the same as three
  independent components (that is uniform over a cube, and denser towards
  its corners). 210 voxels up against 213 down, where it used to be all up.

- **Click and drag push the volume.** `FluidSim3d::impulse_at_screen` turns a
  pointer position and a drag into a puff of dye and a shove, landing on the
  plane through the volume's centre facing the camera — a screen point is a
  ray, and picking the depth any other way needs a readback. In the browser a
  drag pushes and shift-drag or right-drag orbits.
  `a_click_lands_where_it_was_aimed` works backwards from the render, because
  a projection and an unprojection that are not each other's inverse put dye
  somewhere plausible and wrong.

- **`show_fps` draws the frame rate over the picture, in both dimensions.** A
  3×5 bitmap font packed into fifteen bits a glyph, drawn in the shader
  rather than through the renderer's overlay canvas — which would want a
  texture upload a frame and a path to get it here. Both solvers share the
  glyphs and declare their own entry point. Each test checks it draws *and*
  that it changes nothing outside its own box, which is what a `discard` gets
  wrong quietly.

  Under the number is a **sparkline of the last sixty-four frames**, because
  a number tells you the rate now and a chart tells you whether it is
  steady — the question anyone watching a frame counter is actually asking.
  A hitch that lasts two frames never moves a smoothed figure and is a spike
  here. Bars over the refresh target are warm, under it cool.

  It is scaled to **twice the current frame time**, not to the target.
  Scaling to the target reads well until the fluid is slower than it, and
  then every bar clips and the chart is a solid block saying only "slow",
  which the number already said; against the current frame time a steady run
  sits at half height whatever the rate, so the variation stays visible. The
  target keeps a line where it falls, and leaves the top of the chart when
  the fluid is below half of it.

  The history travels as a 64×1 `rgba8unorm` texture, one `write_texture` a
  frame, rather than as more uniform — the plane's uniform block is shared
  with the Metal backend byte for byte, and sixty-four floats is not a thing
  to add to it for a read-out. The shader gets the row width and the
  millisecond range substituted in from Rust, so the draw cannot disagree
  with the upload about how wide the row is or what a byte means.

  `the_chart_shows_the_frame_times_it_was_given` feeds it a flat history and
  then one with a spike, and checks the bars: it exists because the first
  version drew a confident chart of nothing at all, having inherited the
  dye binding from the pass beside it.

  **On the Metal backend too**, which is the first thing the two backends
  share Rust for rather than only a shape: one `Rate` owns the ring of frame
  times, the smoothing and the rule for what counts as a frame, and all
  three solvers call it. Only the glyphs and the chart are transcribed into
  MSL, with the row width and the millisecond range as literals because
  `FLUID_SHADER_SOURCE` is public and has to compile as it stands —
  `the_msl_read_out_agrees_about_the_row` holds those to the Rust, and
  `the_two_read_outs_agree` renders the same pinned history through both and
  compares the pictures. They came back **identical to the byte**, which
  also settles that the premultiplied blend Metal uses lands where the wgpu
  pipeline's straight alpha does. `FLUID_FPS=1` turns it on in
  `examples/metal_fluid`.

  And it shipped with one line of WGSL that only wgpu's own parser accepts:
  `column == 0 && shown < 100 || column == 1 && shown < 10`, which the
  specification forbids without parentheses. **naga takes it; Tint does
  not** — so every native test passed, every browser test passed because
  they ran under WebGL2 where naga compiles the module itself, and in any
  real WebGPU browser the whole module was rejected and with it every
  pipeline in the fluid. The canvas was black while the frame loop ran at
  sixty, the panel worked and wgpu's log was empty, because the error is the
  browser's rather than wgpu's. Fixed, and guarded: `tests/wgsl_portability.rs`
  walks every WGSL literal in the tree for a parenthesised group holding both
  operators, and headless Chrome needs `--enable-unsafe-webgpu` before it has
  a WebGPU adapter to test against at all.

  Testing on the WebGPU path then found a second one of the same kind: the
  **volume** sampled its atlas with `textureSample`, which needs implicit
  derivatives and is therefore legal only in uniform control flow — and the
  volume samples inside the ray march's loop. naga allows it, Tint does not,
  so the 3D fluid had never rendered in a WebGPU browser either.
  `textureSampleLevel(…, 0.0)` is legal anywhere and identical on a
  one-level atlas. The plane's seventy-odd `textureSample` calls stay as
  they are: they sit at the top level of a fragment function, which is
  uniform control flow.

  Two things that made it invisible are also fixed. **wgpu had no logger in
  the browser**, so a rejected pipeline or a failed surface configuration
  went nowhere; `install_console_logger` forwards `log` records to the
  console and `?log=info` raises the level. And **the demo failed silently**:
  the start-up guard covered `init` only, `#msg` is hidden as soon as the
  fluid is running and the panel is unhidden at the very end, so anything
  that threw in between left a black page and no message.

- **The demo works on a phone.** The device pixel ratio is capped at two — a
  phone reports three, and the fluid is a per-pixel cost from the splat to
  the march, so nine device pixels per CSS pixel is most of a frame for
  sharpness nobody resolves at arm's length; a Retina Mac reports exactly
  two, so the cap costs desktop nothing. Default quality drops to `medium`
  on a narrow screen.

  The panel becomes a bottom sheet under 720 pixels — full width, half the
  height at most, scrolling, inset clear of the notch and the home bar, and
  closed on load behind a 44-pixel toggle. Controls grow to a finger's worth
  of target. The two gestures that assume a desktop get touch equivalents:
  the volume's `ctrl`-drag push becomes a **drag pushes** switch, shown only
  where the pointer cannot hover, and the space bar's splats become a
  double-tap. The hints say whichever is true.

  Writing it found the rate counter's threshold to be wrong. It rejected any
  gap under a millisecond, to stop a capture beside a present reading as a
  thousand frames a second; that is right at sixty and wrong at eight
  hundred, which is what this backend reaches at 720p on an M4 — a frame
  there is 1.2 ms, so half of them fell under the floor and the chart had
  bars for 38 of 90 renders. The threshold is a fifth of the frame time
  already measured now, capped at 4 ms so one slow first frame cannot latch
  it shut and leave the rate stuck at the value that set it.
  `the_rate_counts_frames_and_not_duplicate_renders` pins all three cases;
  nothing pinned the old one.

- **Work with no reader, removed.** The occupancy reduction is built for the
  ray march, and it was being built at the end of every step — so a
  simulation stepped a hundred times and drawn once reduced the dye a
  hundred times to use the answer once. It happens at render now, two passes
  of forty-four.

  The inventory that came out of looking: the solve is ~95% of the step and
  the advections are 1–5%, so masking the dye advection to the bricks that
  hold dye — the obvious idea, with the occupancy pyramid already there —
  would have bought under 2% of a frame. Inside the solve, the smoothing
  sweeps and the coarse levels both pay for themselves (one sweep a level
  instead of two leaves 0.536 of the divergence against 0.324, for eight
  fewer passes). What was free was the carry-over.

- **The volume carries all of last frame's pressure**, where the plane keeps
  0.8 after the original. More converges better — 0.269 of a splat's
  divergence after eight frames against 0.281 — and the drift the damping
  exists to prevent does not happen here: the pressure equation's null space
  should let a constant offset accumulate in a half-float field, and nine
  hundred frames of forcing leave the peak at 22 to 26 either way, including
  under Jacobi, which has no coarse grid to anchor the constant and is the
  configuration the damping is really for.
  `the_pressure_carry_over_is_damped_against_drift` is what would catch that
  being wrong.

- **Two solver passes folded into the sweep that follows them.** Starting a
  solve from a fraction of last frame's answer was a pass over the whole grid
  to multiply a field by a constant; the stencil is linear, so scaling what
  the first sweep reads is the same thing. A coarse level starting from zero
  was a clear-only pass and then a sweep that read the zeros back; the same
  scale set to zero collapses it to `-rhs / 6`, no loads at all. **49 render
  passes to 44** at 64³, with every residual in the suite unchanged to three
  figures. `FluidSim3d::passes_per_step` counts them, because five passes in
  forty-nine is inside a GPU timer's noise and not inside counting's.

  That change exposed a latent bug in the bind-group cache: it was keyed on
  the most recently handed-out uniform slot, which is correct only while
  every block is bound before the next is filled. A pass wanting two blocks
  before binding either got one bind group for both — a solver where the
  sweep count stopped mattering, twenty and three hundred sweeps both leaving
  0.944. The slot now travels with the buffer.

- **The iOS C ABI is five functions, not ten.** `create`, `set`, `frame`,
  `info`, `destroy`. Resize, touch, pinch, splat and speed were five calling
  conventions to get wrong across a language boundary; they are now one key
  and three floats, which is the shape the browser build has always had.
  An unknown key does nothing, so an app built against an older header keeps
  working, and `the_header_matches_the_library` now checks the count as well
  as the names. A budget of 0 takes the default, which retired the sixth.

- **iOS and iPadOS.** The crate already built for `aarch64-apple-ios`; what
  was missing was the way in and the budget. `GpuContext::from_metal_layer`
  takes a `CAMetalLayer`, which is what UIKit has instead of a window handle.
  `within_memory` answers the resolution question the way a phone needs it
  asked — the largest volume that fits a number of megabytes, rather than a
  quality name that may get the app killed — and
  `the_memory_estimate_is_the_memory` holds the estimate to what actually
  gets allocated, atlas padding included.

  `crates/threers-fluid-ios` is that as a C ABI static library: nine
  functions, a hand-written header checked against the exports by
  `the_header_matches_the_library`, and a Swift `CADisplayLink` loop in the
  README. `a_layer_comes_up_and_draws` runs the entry points against a real
  `CAMetalLayer` on macOS, where the class, the wgpu surface and the Metal
  drawables are the same ones iOS uses.

  One bug, found by that test rather than by reading: `FluidFormats::volume_dye`
  is what the device *allows*, not what the simulation will use — the packed
  dye is off by default — so sizing against it while allocating with the
  other underestimated the largest field by half. Asking for 48 MB got 58.
  `FluidFormats::resolve` makes the distinction explicit.

- **A speed control**, as `time_scale` on both configurations, in the browser
  panel and as `threers_fluid_set_speed`. Applied by `update` and not by
  `step`, which is the part that matters: every parity, determinism and
  CPU-against-GPU test passes a fixed timestep to `step` and means it, so a
  knob that rescaled it there would pass every test about speed and quietly
  break every test about agreement. `speed_scales_update_and_not_step` pins
  both halves — 2.5× lifts the dye 11.2 slices against 1×'s 2.9, and driving
  `step` directly matches 1× exactly.

- **`CpuFluid3d`** — the volume solver again in plain Rust. Same stages, same
  V-cycle, same `FluidConfig3d`; flat `Vec`s in voxel order rather than an
  atlas, because nothing on the CPU has to write a 2D attachment. Rayon over
  z-slices where the `parallel` feature is on, and `threading_changes_nothing`
  holds it to a bit-identical result either way.

  It is a reference, not a fallback. Two transcriptions of one algorithm
  should remove the same fraction of a puff's divergence, and
  `the_cpu_and_gpu_solvers_agree` holds them to it: 0.358 on the GPU against
  0.337 on the CPU, from half floats through a texture unit on one side and
  `f32` with hand-rolled trilinear on the other.

  And it answers what the GPU is worth. At thirty frames a second, stepping
  only: **one core manages 48³, twelve manage 80³, the GPU manages 288³** —
  a factor of two hundred in voxels. Two things in the table are worth more
  than the ratio: threading buys about 3.5× rather than twelve, because the
  stages are memory-bound and at 16³ it is a net loss; and the GPU is nearly
  flat from 16³ to 96³, 0.92 ms to 2.13, because at those sizes it is paying
  for fifty render passes rather than doing arithmetic.

- **The volume will tell you where its GPU time goes.** `FluidSim3d::profile`
  turns on timestamp queries and `stage_ms` reports the curl, divergence,
  solve, projection and advection separately, off the GPU's own clock rather
  than the wall's — which on a machine with anything else running is the
  difference between a measurement and a number. `GpuTimer` grew from two
  marks to N for it, and `GpuContext` now asks for `TIMESTAMP_QUERY` when the
  adapter has it, because a device opened without it cannot be told to
  measure later.

  It settles a question the wall clock could not. **The solve is about 95% of
  the step** — 1.75 ms against 0.014 to 0.053 for everything else at 64³ — so
  the optimisations above, none of which touch it, are dividing up the
  remainder. That is why they do not stack.

  Two things worth recording from building it. Writing a timestamp at both
  ends of a single pass is the obvious way to time one, and on this GPU it
  returns a pair in the wrong order on most frames and a plausible number on
  some; marks across passes are reliable, so every stage opens on a pass and
  the last is closed by whatever comes next. And the first version took each
  frame's five numbers together, from whichever frame had the cheapest total
  — which reported the solve, identical work in all eight configurations,
  varying by two and a half times. Each stage now keeps its own minimum.

- **`multigrid_coarsest`**, after the measurement it was added to disprove. A
  V-cycle at 64³ is about forty passes and most are on grids from 32³ down to
  4³, where the arithmetic is free and the pass is not, so stopping earlier
  looked like free speed. It is not: two levels instead of four costs 23% of
  the residual to save 10% of the solve. The default coarsens all the way and
  the knob stays, because that balance is a property of a machine's per-pass
  cost rather than of the fluid.

- **The occupancy reduction moved into the step**, where the dye has just been
  finalised, rather than running at render time. The march no longer waits on
  a reduction of the thing it is about to read, and a render that follows a
  splat still rebuilds it so a fresh blob cannot fall into a brick marked
  empty.

- **The volume's resolution ceiling, measured.** Marching into 1280×720 on an
  M4 Pro: 128³ solver with a 256³ dye runs at 188 fps in 328 MB, 192³ at 83
  fps in 1.1 GB, and **272³ carrying a 544³ dye is thirty frames a second**,
  in 3.2 GB. Memory is what most machines run into first — the cube law makes
  each step roughly twice the last.

- **The volume's optimisations are runtime switches, and the bind groups are
  kept between frames.** A step records about fifty passes whose descriptions
  are identical frame to frame — same uniform blocks in the same order, same
  textures until the grids are rebuilt — and rebuilding them was most of what
  the CPU did in a frame. Keyed on the slot and on which textures, they are
  now built once: **1.12–1.34× on the CPU encode**. It does not show in the
  frame on an M4 Pro, where 0.6–0.8 ms of recording already hides behind 2.3
  ms of GPU work; it is there for where the CPU is the binding constraint.

  `empty_space_skip`, `cache_bindings`, `compact_dye`, `pressure_solver` and
  the resolution are all changeable on a running simulation, and all are live
  controls in the browser demo, because which of them pays is a property of
  the machine rather than of the fluid.

  The bench grew a `spread` column with them. These are effects of tens of
  percent and a whole frame on a shared machine varies by more than that
  round to round; a column that shows it is the difference between a
  measurement and a number.

- **The dye can be packed into `rg11b10ufloat`, and mostly should not be.**
  It is the largest field in a volume and never negative, so it does not need
  the sign bit `rgba16float` spends: three 11-bit floats hold it in half the
  memory, 41 MB down to 24 at the default, and the picture is the same to a
  tenth of a level. On an M4 Pro it is also **twice as slow** — sampling the
  packed format costs more than the bandwidth it saves, with the march alone
  40% slower at an identical sample count. So `compact_dye` is off by
  default, documented with the measurement, and a runtime switch rather than
  a build-time one — the trade is a property of the texture unit, and a
  software rasteriser with no filtering hardware to be slow at unpacking may
  well go the other way. `FLUID_COMPACT=1 examples/fluid3d_bench.rs` is how
  to find out which a machine is.

- **The fluid on the Metal backend** — `threers::metal::MetalFluid`, the same
  solver through the Objective-C runtime with no wgpu underneath it at all,
  for an app that already owns an `MTLDevice` and does not want a second
  graphics abstraction inside it. `src/metal/fluid.metal` is a transcription
  of the WGSL function for function, and the two share the uniform block so
  they cannot drift apart.

  `cargo run --features "metal,fluid" --example metal_fluid -- --compare`
  renders the same simulation both ways and reports the difference.
  `metal_fluid_matches_wgpu` keeps it honest at five frames, where the two
  are still the same picture: a fluid is chaotic, so the mean difference goes
  0.00/255 at zero steps, 0.04 at five and 4.79 at ninety from half-float
  rounding alone.

  Two bugs that the comparison found, both invisible from the Metal side
  alone. A fresh `MTLTexture` holds whatever was in that memory — wgpu clears
  one before anything can read it — so a NaN in the velocity reached the first
  advection and the dye it fetched came back as nothing: a picture that stayed
  empty however many splats went in. And the display pass replaced the target
  instead of compositing over it, so the fluid's zero alpha where there is no
  dye erased the background that had just been drawn.

- **The browser demo does both fluids.** `web/fluid.html` has a dimension
  dropdown: the plane, or smoke in a volume with the orbit camera. The panel
  shows the knobs that mode has and reports the grids it actually got.

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

- **rlx comes from crates.io, at 0.2.17** (`rlx`, `rlx-geo`, and in
  `threers-probe` and `threers-ocean`), with `Cargo.lock` resolving the whole
  family from the registry. The committed lockfile had recorded local path
  checkouts and 268 unused patch entries from a machine-local override. 0.2.17
  rather than the newest, 0.2.18: the published `rlx-runtime` 0.2.18 imports
  `rlx_ir::NumberFormat`, which the published `rlx-ir` 0.2.18 lacks, so it
  does not compile from crates.io.
- **The JavaScript API has no `MOUSE`.** `THREE.MOUSE` and the `MOUSE`
  export are gone; the controls keep three.js's button numbers internally, so
  `mouseButtons = { LEFT: 0, MIDDLE: 1, RIGHT: 2 }` still works.
- The npm package no longer ships the `.wasm.br` / `.wasm.gz` copies (package
  CDNs and bundlers ignore them), which had tripled the wasm's share of it.
- **Standard and Physical roughness gains three.js's `geometryRoughness`.**
  Highlights widened by geometric specular antialiasing (the shading
  normal's variance across a pixel, Kaplanyan et al. 2016) in place of
  three.js r165's term, the largest screen-space change of the view-space
  normal; that was the whole remaining difference from three.js on Physical
  highlights (up to 27 levels, 79 on an anisotropic one, against 1–4 now).
  The antialiasing is kept as an option, `set_specular_antialiasing(true)`,
  for crinkled foil or fine normal maps that sparkle under motion. Metal had
  neither term and now follows the same rule.
- **`PhysicalMaterial::iridescence_thickness` is
  `iridescence_thickness_range: [f32; 2]`**, three.js's
  `iridescenceThicknessRange` (default `[100, 400]`); `with_iridescence`
  still takes one thickness, the top of the range.
- The legacy material layout no longer binds `iridescenceThicknessMap` or
  `displacementMap`; a material with either draws through a shader variant.
  That frees two texture units on WebGL2, where the legacy layout had reached
  seventeen against a limit of sixteen and every draw panicked in wgpu's GL
  backend.
- **The shim's renderer antialiases only when asked**, as three.js's does: it
  forced 4× MSAA, so every edge differed from three.js's default
  `antialias: false` — 515 pixels on the `cube` parity scene, a wireframe 0.81%
  of the frame. Pass `{ antialias: true }` for MSAA; the demos and examples
  now do. 120 of 132 parity scenes are pixel-identical to three.js (107
  before), the rest within 0.21%.
- **Removed the CSG's stand-ins for evaluating it**:
  `build_bvh_csg_hierarchy_geometry` (it returned geometry exported from the
  deleted JavaScript port with one live step on top), the step helpers
  (`step1_shell_cut` … `step4_win_frame`, `evaluate_through`,
  `evaluate_live_through`, `js_target_verts`, `JS_STEP*_VERTS`),
  `load_positions_geometry_bin` and the `TriKey` topology helpers, with the
  tests, investigation scripts and 45 fixtures that pinned the CSG to that
  port. `evaluate_hierarchy` matches three-bvh-csg directly now; the
  `bvh_csg_hierarchy` example evaluates live and takes an optional step
  (it absorbs `bvh_csg_steps`).
- **`CsgBrush`'s transforms are `matrix` / `matrix_world: [f64; 16]`** (were
  `local_matrix` / `matrix_world: Matrix4`); `set_matrix_world` replaces
  `set_matrix_world_js`.
- **`ToonMaterial::steps` is `gradient_map: Option<Arc<Texture>>`**, three.js's
  `gradientMap`. `ToonMaterial::gradient_steps(n)` builds the evenly spaced
  `n`-texel ramp `steps: n` used to mean.
- **Reflective transparency is off by default.** A transparent Standard or
  Physical surface's alpha was raised by its specular reflectance
  (`opacity + (1 − opacity) · reflectance`); three.js blends the whole colour
  by `opacity`, so threers now does too, and the boost is
  `set_reflective_transparency(true)`. Metal had no such term and gains it,
  matching wgpu with and without an environment.

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

- **OpenSCAD booleans with a cylinder came out empty.** `difference() {
  cube(…); cylinder(…); }` evaluated to no geometry: the float CSG kernel now
  reads three-bvh-csg's `position`, `uv` and `normal` from both operands, an
  OpenSCAD cylinder has no `uv`, and the panic that followed was caught and
  turned into an empty mesh. OpenSCAD now evaluates each boolean over the
  attributes both operands carry (a caller's `CsgEvaluator::attributes` is
  restored after).
- **Coaxial cylinders resolved exactly again.** `CylinderGeometry` now
  computes in `f64` to match three.js bit for bit, and on those positions the
  exact kernel deferred `intersection` and `difference` of two cylinders offset
  along their axis. OpenSCAD's `cylinder`, `cone` and `frustum` use the new
  `CylinderGeometry::single_precision`, the same mesh with `f32` positions;
  `CylinderGeometry::new` keeps three.js's numbers.
- The shim's `BufferGeometry.dispose()` did nothing: an empty second
  `dispose()` later in the class replaced the one that drops the BVH and the
  half-edge structure. The `THREE` namespace object also listed twelve names
  twice; esbuild now reports nothing over `web/`.
- `cargo build --workspace` failed to link the Python extension on macOS
  (`Undefined symbols … _PyBaseObject_Type`): an extension module leaves the
  interpreter's symbols to be found at load time, which maturin arranged and
  a plain cargo build did not. `crates/threers-py/build.rs` now asks for it.
- `hevc::CompressedEncoder::tiles(..).deblock(false)` wrote a PPS with
  deblocking on, so its stream no longer matched its own reconstruction.
  Tile grids are also reduced to HEVC's minimum tile size (256 × 64 luma);
  narrower tiles were rejected by the reference decoder.
- The H.264 encoder's SPS now carries a VUI saying BT.601, limited range.
  `Yuv420Frame::from_rgba` converts with BT.601, but players assumed BT.709
  for HD without a VUI, which shifted colours (a red of (220,30,40) came back
  as (235,49,37)).
- The software HEVC encoder's SPS says BT.601, limited range, the same way
  (the alpha layer still states only its full range). The MP4 `colr` box now
  repeats what the stream's SPS says, so the two never disagree: it was
  hard-coded to BT.709, which mislabelled every file from threers' own
  encoders. VideoToolbox streams still say BT.709 in both places. The golden
  MP4 checksums in `tests/h264_animation_parity.rs` and
  `web/scripts/video-export-test-lib.mjs` change with it.
- A malformed SPS, PPS or slice header could panic the native decoders
  (arithmetic overflow on all-ones Exp-Golomb codes, crops past the picture,
  unbounded tile counts). Each is now range-checked and refused with an
  error. Fuzzed with 50,000 random parameter sets.
- `DecodedFrame::to_bgra` / `to_rgba` (and `NativeDecoder::take`) upsample
  chroma bilinearly from where H.264 and HEVC site it by default, rather
  than nearest-neighbour. Colour edges no longer step, and the output is
  within 39 dB of VideoToolbox's (31 dB before). The conversion is integer
  fixed-point and runs over rows on several threads for large pictures
  (one on `wasm32`): about 1 ms rather than 6 ms for a 1080p frame.
- UI layout follows the browser more closely:
  - A flex item's automatic minimum size is the smaller of its content and
    its specified size, and zero for a box that clips.
  - Centred and end-aligned flex items overflow both ends when they do not
    fit.
  - Text widths are rounded up to 1/64 px, as Blink's `LayoutUnit` does.
  - Positioned boxes paint, and hit-test, after in-flow siblings.
- Spot and point shadows took a single depth comparison, so their edges
  were a staircase. All shadow kinds now filter over a disc (twelve
  comparisons, up from the directional's nine). `ShadowSettings::map_size`
  was documented as ignored long after it took effect.

- **Glass.**
  - `TransparencyMode::Refract` was tuned to one scene. It forced a blur of
    at least mip 2, clamped refraction to 2.5% of the screen, added a fake
    caustic and density tint, and reflected black where screen-space tracing
    missed. It now follows three.js r165's volume model: the refracted ray
    runs through the measured back-face distance, blur follows roughness,
    dispersion splits the channels, tint and attenuation absorb, Fresnel uses
    the environment BRDF, and misses fall back to the environment. Lights now
    give highlights. The capture is decoded and composited in display space,
    so glass no longer washes out under tone mapping.
  - `TransparencyMode::Glass` and `Oit` with `transmission` drew nothing:
    they read the transmission capture, which those modes never make. They
    now pass what is behind through by channel — green glass tints green,
    where one blend alpha could only darken evenly — and keep reflections
    and highlights at full strength. (`Glass` draws in two halves, a tint
    multiplied over what is behind and then the light added; dividing the
    light by a small alpha to survive one blend had clamped a highlight to
    a twentieth of itself.) `tests/glass_tint.rs` holds both to `Refract`.
  - `Oit` shaded with a Lambert of its own — no transmission, no point or
    spot lights, no shadows — so a glass ball in that mode came out an
    opaque white sphere. It now shades as every other mode does, with a
    coloured revealage. On WebGL2, where OIT cannot run, its draws vanished;
    they now draw as `Glass`.
  - The Metal renderer drew `Glass`, `Oit` and `Refract` glass from a black
    stand-in capture. `Glass` and `Oit` now blend there as one layer tinted
    by channel — composited in the shader on the encoded values, in one draw
    — and `Refract` draws as three.js's transmission over the capture.
    `tests/glass_tint.rs` and `tests/glass_shadows.rs` hold Metal to the same
    numbers as wgpu.
  - Glass highlights ignored shadows in `Refract`, whose pass has no room
    for the shadow maps under WebGL2's sixteen texture units. A pass just
    before it looks every light's shadow up on the glass and packs them into
    one texture. `tests/glass_shadows.rs` checks every mode.
  - Thin-walled glass (`thickness` 0) bent light in `Refract` and in the
    path tracer. Now it goes straight through everywhere.
  - Thin glass casts a light shadow (dithered into the shadow map). Solid
    glass keeps a full one.
  - The path tracer read `dispersion` as a fraction of the IOR, so
    `dispersion: 1` gave red an IOR of 0 and drew a magenta ball. It now
    uses three.js's spread, `(ior − 1) × 0.025 × dispersion`. Shadow rays
    pass thin-walled glass, picking up its transmittance.
- **Antialiasing.** One `Glass`, `Oit` or `Refract` material in view turned
  MSAA off for the whole frame, on screen as well as off. All three
  multisample now, on every backend: `Glass` with multisampled pipelines,
  `Oit` into multisampled accumulation targets, `Refract` over a
  multisampled opaque pass whose depth is resolved for the glass's march —
  or, on WebGL2, which cannot read a multisampled depth, drawn again
  single-sampled. `tests/glass_occlusion.rs` checks both depth paths keep
  glass behind an opaque object hidden.
- In a frame with refraction glass, OIT surfaces were composited and then
  overwritten by the capture laid back down; they now composite into the
  capture behind the glass, or over it in front. `Refract` also
  smooths its silhouettes analytically. `tests/glass_antialiasing.rs` checks
  that every mode keeps its edges smooth and its MSAA.
- The Metal renderer's layout test held `Draw` and `Frame` to formulas that
  had fallen behind the structs, and failed.
- The Metal renderer bound nil in most sampler slots for a draw when no
  texture had been uploaded yet, which Metal's validation layer
  (`MTL_DEBUG_LAYER=1`) rejects; its default and environment samplers are
  now made with the renderer. Every Metal suite passes under validation.

- **`GpuContext::open` on a WebGL2 adapter** (e.g. from `open_browser_gpu`'s
  fallback) failed in `request_device`: `renderer_limits` started from the
  general downlevel floor, which asks for compute limits WebGL2 does not
  have. It now starts from WebGL2's floor on a GL adapter.

- **CSG over-split and kept the wrong pieces.** The port's triangle–triangle
  test was not three-mesh-bvh's `ExtendedTriangle.intersectsTriangle`: it
  read the `suppressLog` flag every caller passes as "coplanar" and so
  counted any two triangles whose planes straddle each other as
  intersecting. Faces were split by triangles that never touched them — a
  sphere ∪ box came out as 4,239 triangles where three-bvh-csg gives 1,081 —
  and `Triangle::intersects_triangle_ext` gave the same wrong answer. It is a
  faithful f64 port now, and three-bvh-csg@0.0.16 itself is the oracle:
  `tests/csg_upstream.rs` replays every evaluation the parity scenes make,
  recorded from the library in Node, and each comes out with the same
  triangles, area and volume — exactly, which took the rest of this list.
  Groups were never assigned (`groupIndices` were all 0) and `b`'s offset was
  fixed at 1; both follow three-bvh-csg now.
- **The CSG cuts every face as three-bvh-csg does.** A face is cut in the
  order the BVH meets its cutting triangles, and a vertex lying on another
  face is classified by the last bit, so matching three-bvh-csg triangle for
  triangle took:
  - **three-mesh-bvh 0.7.6's tree**: per-triangle bounds as its f32 centre and
    padded half-extent, CENTER on the centroid bounds (it split at the node
    bounds' centre), AVERAGE and SAH as three-mesh-bvh computes them, and
    Hoare partitioning. `tests/mesh_bvh_raycast.rs` checks the triangle order
    against three-mesh-bvh's own.
  - **three.js's f64 arithmetic, operand order and all**: `Triangle.getNormal`
    is `(c − b) × (a − b)`, `Vector3.normalize` multiplies by the reciprocal,
    `distanceTo` is `sqrt(distanceToSquared)` (not a hypot), `getMidpoint`
    multiplies by `1/3`, and `Plane.intersectLine` solves for `t` — each was
    a near miss.
  - **`raycastFirst` as three-mesh-bvh walks it**, in f64, for the
    inside/outside test: children in split-axis order, the far child skipped
    when the near one's hit is outside it, and the root box not tested. Nodes
    record their split axis for it.
  - **f64 transforms**: `CsgBrush`, `CsgOperation` and `CsgOperationGroup`
    hold `matrix` and `matrix_world` as `[f64; 16]`, compose them as three.js
    does, and take the determinant and normal matrix in f64. A brush's
    position no longer bypasses its parent's transform.
- **Every primitive is three.js's to the bit**: `BoxGeometry`,
  `SphereGeometry`, `PlaneGeometry`, `CylinderGeometry`, `ConeGeometry`,
  `CircleGeometry`, `RingGeometry`, `TorusGeometry`, `TorusKnotGeometry`,
  `Tetrahedron`/`Octahedron`/`Icosahedron`/`DodecahedronGeometry`,
  `CapsuleGeometry` and `LatheGeometry` — index, positions, normals, uvs and
  groups — against three r165's own classes over a spread of parameters
  (`tests/primitives_upstream.rs`, recorded by
  `tests/parity/export-primitives-upstream.mjs`). They computed in f32, which
  put a sphere's equator vertices on the wrong side of an axis — enough to
  double a ray's hits there — and several had their own layout: the
  cylinder's caps shared one centre vertex where three.js gives each segment
  one (and swapped the cap uvs' sine and cosine) and had no groups; the
  polyhedra's flat normals and seam uvs, the torus knot's frame (an extra
  normalisation), the lathe's normals (divided rather than scaled by the
  reciprocal) and the capsule's poles (forced onto the axis) each differed.
  Each generator now has an f64 core (`from_f64` / `with_f64`) taking
  JavaScript's numbers; the `f32` constructors recover the literal the caller
  wrote.
- **The shim's `OrbitControls` ignored a pose set from JavaScript**: `update()`
  drove the camera from its own state, so `camera.position` and
  `controls.target` set from outside (a saved view, a tween) snapped back. It
  re-reads them as three.js's does. It also wrote the camera's tilted y axis
  into `camera.up`, rolling every later `lookAt`; `up` stays the world up.
- **A ray through a shared vertex or edge hits as three.js's does.** The
  raycaster, the BVH and the CSG each had their own ray–triangle test, the
  BVH's a Möller–Trumbore in f32; all three are now three.js's
  `Ray.intersectTriangle` in f64 (`Ray::intersect_triangle_weights`).
- **The shim's primitives are `BufferGeometry`s**, as three.js's are: each
  takes three.js's full constructor (the plane's, ring's, cylinder's and
  cone's segment counts, open ends and angle ranges were dropped), has its
  `type`, `parameters`, index, attributes and groups, and works with
  `translate`, `computeBoundingBox` and the rest. They are built in Rust from
  JavaScript's numbers (`WebGeometry.threePrimitive`), bit for bit as three.js
  builds them.
- **The shim's `BufferGeometry.toNonIndexed` dropped the groups**, the morph
  attributes and `morphTargetsRelative`; it keeps them, as three.js's does. A
  multi-material CSG built from unindexed primitives came out differently for
  it.
- **The shim's raycaster casts against `matrixWorld` as it stands**, as
  three.js's does: it brought every object's world matrix up to date first,
  so an object moved since its last `updateMatrixWorld` or render was hit
  where it now is, where three.js hits it where it was.
- **`side: BackSide` drew front faces** on wgpu's main, encoded and overlay
  passes: the cull mode only knew front and double. Back-sided materials now
  cull front faces, as three.js's do.
- **`RectAreaLight` took its scale into its direction.** The light's frame is
  its world rotation alone, its half-extents `width/2` and `height/2`, as
  three.js's `RectAreaLight` uniforms are; a scaled light leaned and grew.
- **`iridescenceThicknessMap` could not be set by name**: `map_slot_mut` (and
  so `setNamedMap` from JavaScript) did not know it, and the shim's
  assignment was silently dropped. A test now holds every map the shader
  samples to a slot three.js names.
- The shim's per-render material sync skipped a material with vector
  properties but no colour properties.

- **Background and clear colours were decoded to linear twice** on sRGB and
  float targets, and left unencoded on plain `Unorm` ones such as a
  `bgra8unorm` canvas, so `Color::from_hex(0x20262c)` came out as `(0, 1, 1)`.
  A regression from earlier in this release: before it, the clear was only
  decoded on the linear framebuffer, which is why the example images in the docs
  were unaffected. One rule now picks the encoding by target format, and a hex
  background lands as that hex on every format and on Metal alike, checked
  across twelve colours, three formats, MSAA, supersampling and scissored
  clears. The three.js shim's `Scene.background` had been compensating and now
  passes the linear colour, as three.js does.

- **Several renders a frame to a WebGL2 canvas lost the earlier ones.** Each
  render gets the canvas as a fresh texture that wgpu marks uninitialised, so a
  pass that loads it is zero-filled first; browser WebGPU never showed it. Once
  a page sets a viewport, scissor, `autoClear = false` or calls `clear()`,
  canvas draws on WebGL2 go through a persistent buffer copied to the canvas
  after each draw, without MSAA.

- **`CapsuleGeometry` left its bottom open.** The bottom hemisphere's rings
  started one step late, so the band ended below the equator and the last ring
  passed the pole at a negative radius. It is now a `LatheGeometry` of three.js
  r165's own path, which also doubles the cap segments as r165's
  `CurvePath.getPoints` does, and matches an r165 dump buffer for buffer.

- **`TubeGeometry` dropped its last segment when open**, ending half a segment
  short of the curve, and a closed tube's seam cracked because its last ring was
  the transported end frame rather than a copy of the first. Every segment is
  emitted now, and a closed tube repeats its first ring and spreads the frame
  twist along its length, as three.js does.

- **Post-processing flipped the picture vertically.** The post-fx vertex stage
  put uv (0,0) at the bottom-left while wgpu textures start at the top row, and
  a flag flipped it back only for the canvas — so a pass into a render target,
  an odd-length chain, UnrealBloom's bloom and SSAO's debug outputs came out
  upside down, and SSAO reconstructed positions from flipped depth against
  unflipped normals, smearing occlusion over open floor (0.70 where it should
  read 1.00). The native `EffectComposer` had the same bug in its own vertex
  stage. Every pass now preserves orientation; effects that need three.js's
  bottom-left `vUv` get it explicitly. The flag and the shim's compensating
  double flips are gone (`Renderer::apply_postfx_ex` lost its `flip_y`
  parameter), and AfterimagePass, TAARenderPass and SSRPass's blur no longer
  draw into the target they sample. Checked per effect kind, natively and in
  WebGL2 and WebGPU, including where SSAO darkens and which side SSR reflects
  on; most parity scenes are vertically symmetric, which is why they never
  caught it.

- **SSAO read no depth on WebGL2.** The hardware depth was fetched with
  `textureLoad` and no sampler, so GL fell back to the texture's own default
  filter; GLES treats a depth texture without `NEAREST` filtering as incomplete,
  and an incomplete texture reads 0. Reading it through a nearest sampler makes
  WebGL2 agree with WebGPU to 4·10⁻⁶ in depth and 10⁻⁴ in mean occlusion.

- **`AdaptiveToneMappingPass` and `BokehPass` ran the glitch effect** — the
  glitch branch claimed kinds 25 and 26 before their own branches were reached.
  Each now runs its own shader; the glitch variants moved to 31 and 32.
  `SSAOPass`'s default output multiplies the blurred occlusion into the beauty
  pass instead of only copying the scene, and `GlitchPass`'s snow hashes pixel
  centres from the bottom-left, as `gl_FragCoord` does, instead of mirrored.
  The parity runner finds Chrome for Testing and takes `CHROME`, `SCENES`,
  `PORT` and `PKG_DIR`.

- **Post-processing on WebGL2 stored different bytes from WebGPU.** Every
  target and the canvas are sRGB there, with no view formats, so a computed pass
  was encoded once more than on WebGPU, and normal and depth prepasses were
  stored encoded. Each pass now knows its output and input formats and decodes
  or re-encodes so the stored byte matches WebGPU's — checked across copy, film,
  tone mapping, luminosity, resampling and SSAO blur — and normal and depth
  materials write unencoded values on sRGB targets. A second post-fx write to a
  WebGL2 canvas no longer starts from a cleared surface, so `UnrealBloomPass`
  and `OutlinePass` keep their base image.

- **Shim post-processing checked against three.js's own source.**
  `AdaptiveToneMappingPass` (last shipped in r152) starts from zeroed,
  full-resolution luminance targets and adapts per pixel; `BokehPass` follows
  `setSize`; `EffectComposer` uses its clock when `render()` gets no delta and
  sizes passes on `addPass`; `GlitchPass` and `SSAOPass` draw into the write
  buffer unless last; the SSAO normal prepass clears to the linear scene
  background (the `ssao-normal` parity scene went from 57.76% different to
  0.00%); SSAO counts off-screen samples and tests the background with
  `depth == 1`; and `OutputPass` uses the exact sRGB curve. New bokeh and
  adaptive tone-mapping parity scenes match three.js.

- **Skinned meshes were drawn in three different places.** wgpu skinned without
  the mesh's own transform, Metal applied it after normalising the weights, and
  the path tracer normalised and then transformed; the SVG renderer drew the rest
  pose. All of them, and the raycaster, now skin as three.js does — attached
  bind mode, identity bind matrix, weights used as given — and a moved, rotated,
  scaled mesh with posed bones and weights summing to 0.85 lights the same 798
  pixels on Metal and wgpu, with rays agreeing on all 3538 pixels inside and
  outside it. Metal's skinned normals had ignored the skin entirely.

- **Sprites ignored `center` and `rotation`** on wgpu, and Metal drew them
  unbillboarded. Both now follow three.js's sprite vertex stage, with a new
  `SpriteMaterial::size_attenuation`, so a sprite is picked where it is drawn.

- **Raycasting edge cases.** A ray grazing a bounding-box face — a click on a
  plane's edge — missed, because the slab test let `0 · ∞ = NaN` through; the
  BVH's first-hit traversal had the same flaw, and a vertex lying exactly on the
  bounding sphere was culled. Orthographic rays start in the camera plane, as in
  three.js, rather than at the near plane, which made every distance short by
  `near`. `set_from_camera` no longer overwrites the raycaster's layers. After
  writing a geometry's public fields directly, `bump_version()` tells the
  renderers and the cached bounds.

- **Material models matched neither three.js nor each other.** Each is now
  checked per pixel against three.js r165's formulas evaluated analytically on a
  ray-cast sphere — not against the other backend — and lands within one byte
  on both. Metal had no 1/π on any non-PBR material and was about π× too
  bright; wgpu left ambient and hemisphere light undivided by π for Lambert and
  Phong, used a non-three.js Phong specular, skipped tone mapping and encoding
  for both on plain canvases, and was off by 31 (Lambert), 95 (Phong), 154
  (Toon) and 116 (Matcap) levels. Both PBR paths weighted diffuse by (1−F),
  which three.js does not; Toon now follows `getGradientIrradiance`
  (`ToonMaterial.steps` defaults to three.js's two-tone ramp); Matcap uses
  three.js's view frame; Metal's normal material writes view-space normals; and
  Phong's default specular is `0x111111`. Lit scenes look different from before,
  and correct.

- **Lambert, Phong and Toon received no shadows on wgpu**, `ShadowMaterial`
  ignored spot and point shadows there and was missing on Metal, and a point
  light without `distance` cast no shadow on Metal and a made-up one on wgpu.
  All of them now receive directional, spot and point shadows as three.js does.

- **Transparent surfaces blended in linear light** on sRGB targets — the
  headless default — where three.js's default renderer blends the encoded
  bytes: 50% red over blue gave (188, 0, 188). Metal now blends encoded values
  through programmable blending and wgpu through a Unorm view of the same
  target, with the MSAA resolve still linear. WebGL2 still blends linearly: its
  canvas is sRGB and it cannot reinterpret a view.

- **Instanced meshes** take material arrays (one instanced draw per group, and
  `material_index` on raycast hits), shared or per-instance morph targets
  (`set_morph_at`, as in r159+), and three.js's instance normals — wgpu ignored
  the instance rotation for normals entirely. Metal vertex colours carry alpha
  and morph all four channels; custom shader materials get `morph_position`,
  `morph_normal` and `morph_color` in the preamble. `polygon_offset` (three.js
  `polygonOffset`) works on both backends, which takes the decal parity scene
  from 8% different to identical. USD export writes morph targets as blend
  shapes and the importer reads them on unskinned meshes, checked with
  `usdcat` and `usdchecker`.

- **Tone mapping.** `ACESFilmicToneMapping` on wgpu was a different curve fit
  (Narkowicz) from three.js's, putting mid grey at 0.267 instead of 0.213; it is
  three.js r165's formula now. Basic, line, points, sprite, sky and reflector
  materials are tone mapped as three.js does, every material honours
  `tone_mapped: false`, and Metal — which had no three.js tone mapping at all —
  gains `set_tone_mapping` with the same curves.

- **Encoded blending everywhere.** Sprites, instanced, skinned and custom
  shader draws, viewport panes drawn through MSAA, refraction frames, and
  WebGL2 canvases now blend transparent surfaces on encoded bytes as three.js's
  canvas does; WebGL2 does it through an 8-bit scratch copy, since it cannot
  reinterpret a texture's format. Transparent instanced and skinned meshes used
  to draw opaque on wgpu. Instanced meshes cast shadows on wgpu, per-instance
  morphs included; Metal applies instance colours; and point-light shadows only
  draw objects that cast shadows, where before they drew everything.

- **Material `depthTest`, `depthWrite`, `depthFunc` and `colorWrite`** were
  ignored by both backends: every draw tested and wrote depth. They are honoured
  now on wgpu and Metal for every kind of draw, including custom shader
  materials, and forwarded from the JavaScript materials, so a gizmo drawn with
  `depthTest: false` shows through what is in front of it, as in three.js.

- **`Color` is three.js r165's own code** in the JavaScript layer, with
  `ColorManagement`, instead of allocating a wasm object on every setter call.
  `InterleavedBuffer`, `InterleavedBufferAttribute` and
  `InstancedInterleavedBuffer` have three.js's full API, and interleaved
  geometry renders — it is de-interleaved where it crosses into Rust, and
  `needsUpdate` on a shared buffer refreshes every geometry drawing from it.

- **`alphaTest` only worked on `MeshBasicMaterial`, and `transparent` only on
  Basic and shader materials.** Every material that has them in three.js now
  takes both, on wgpu and Metal: a fragment whose opacity times map alpha falls
  below `alphaTest` is discarded before lighting, as `alphatest_fragment` does —
  `MeshNormalMaterial` excepted, whose three.js shader has no alpha test — and a
  material that is not `transparent` draws opaquely whatever its opacity, where
  before any opacity below 1 blended. The JavaScript materials always send
  three.js's flag; native materials that never set `transparent` keep blending
  below opacity 1, the rule they had. A sprite's map alpha now multiplies its
  opacity, as three.js's sprite shader does, and sprites default to
  `transparent: true` as three.js's do.

- **Shadows follow three.js's shadow map.** A caster with `alphaTest` and a
  map casts the shadow of what the test leaves — three.js's shadow depth
  material takes both — where before a cut-out leaf or fence cast a solid
  square; on wgpu and Metal, for directional, spot and point lights, and for
  plain, skinned and instanced meshes. A caster's `side` picks the faces its
  shadow records, as three.js's `shadowSide` does: every caster used to record
  its back faces whatever its side, so a double-sided plane facing the light
  cast nothing on wgpu, and Metal culled nothing at all. Point-light shadows
  swap the culled face, because the cube faces mirror one axis. Transparent
  meshes cast their whole shadow, as in three.js — both backends skipped them.
  On Metal, a Basic material at opacity below 1 without `transparent` now draws
  opaquely, as it always did on wgpu. A `shadow-cutout` parity scene matches
  three.js pixel for pixel.

- **Lambert, Phong and Toon had no maps.** They now take three.js's `map`,
  `emissiveMap`, `normalMap` (with `normalScale`), `aoMap` (with
  `aoMapIntensity`) and `displacementMap` (with scale and bias), on wgpu and
  Metal, from Rust and from the JavaScript materials, which until now dropped
  every map but Lambert's colour map. Ambient occlusion darkens the indirect
  light only, as three.js's `aomap_fragment` does, and `aoMap` reaches
  Standard and Physical from JavaScript too; it had no binding at all. Each
  case is checked against three.js's shading to within a level on both
  backends, and a `lit-maps` parity scene matches three.js pixel for pixel.

- **A texture changed after binding never reached its material.** The
  JavaScript layer bound a texture's wasm object once, so an image arriving
  later — every `TextureLoader` texture assigned before it loaded —
  `texture.colorSpace = SRGBColorSpace` set after binding, `needsUpdate`, and a
  `CanvasTexture` redrawn all kept the first upload; `colorSpace` was not
  passed to wasm at all, so sRGB colour maps were read as linear and came out
  washed out. A texture now re-binds into every material slot holding it on
  any of these, its `colorSpace` decides whether the sampler decodes it, and a
  `CanvasTexture`'s `needsUpdate` re-reads its canvas. A material without a
  colour map samples its other maps with the first one's filtering (an
  `aoMap` alone, a nearest-filtered `emissiveMap`), where it used to fall back
  to linear filtering.

- **Every texture now keeps its own sampler and uv transform, as in
  three.js.** A material's maps all shared the colour map's sampler and its
  offset and repeat, so an `emissiveMap` tiled differently from the `map`, a
  lone `normalMap` with `repeat`, or a nearest-filtered map next to a linear
  one all read wrongly. On wgpu (and so WebGPU and WebGL2) a material whose
  maps disagree is drawn with a shader variant that declares each map with its
  own sampler and transform; on Metal each map slot has its own sampler and
  transform. Texture `rotation` and `center` (and a hand-set `matrix` with
  `matrixAutoUpdate` off) now apply as `Matrix3.setUvTransform` composes them —
  wgpu ignored rotation, Metal turned it about (½, ½) rather than `center`.
  `MirroredRepeatWrapping` mirrors, where it repeated; `wrapS` and `wrapT` are
  separate, where either one repeating made both repeat; and `minFilter` is
  honoured on its own rather than following `magFilter`. Two materials over
  one image with different wrap or filter no longer share a bind group.
  Checked against three.js's sampling on both backends and in both browser
  backends.

- **The JavaScript layer's texture state now follows three.js.** Its
  `LinearMipmapNearestFilter` and `NearestMipmapLinearFilter` constants were
  swapped (1005 and 1007), and its classifier treated `LinearMipmapNearest` as
  nearest; a texture's `offset`, `repeat`, `rotation`, `center` and `matrix`
  are now read before every render, as three.js reads the texture matrix each
  frame, so `map.offset.x += 0.01` scrolls a map — edits in place after binding
  never reached the material. Filter and wrap changes apply at the first
  render or on `needsUpdate`, as three.js sets them on upload, replacing a
  one-microtask re-sync that only `MeshStandardMaterial` had. `CanvasTexture`
  and `VideoTexture` keep three.js's default `offset` (0, 0) and `repeat`
  (1, 1) and read their top-first pixels through `flipY`, where they folded a
  flip into `offset` and `repeat` that user values overwrote.
  `SpriteMaterial.map` is live like every other material's.
  `MeshPhysicalMaterial`'s maps beyond `map` (`normalMap`, `roughnessMap`,
  `aoMap`, `emissiveMap`, …) were not passed to wasm at all; they now are, live,
  as on `MeshStandardMaterial`.

- **`readRenderTargetPixels` returned BGRA on WebGPU.** A render target there
  takes the canvas's `bgra8unorm` format and its bytes were returned as they
  were, red and blue swapped; three.js returns RGBA whatever the target, and so
  does this now. The browser tests had chosen colours that read the same either
  way round, which is how it went unnoticed; the WebGL2 and WebGPU suites now
  share their scene tests (lit, skinned, refracting, shadowed, post-processed,
  `ShaderMaterial`, morph targets and groups) and both check read-back colour
  exactly. The WebGL2 suite hides `navigator.gpu` before starting a renderer,
  since Chrome for Testing 155 offers headless pages a WebGPU adapter even with
  WebGPU disabled, and the whole suite had quietly run on WebGPU.

- **Render targets follow three.js's configuration.** Rendering into a target
  applied the renderer's tone mapping, which three.js r165 applies only to the
  final output — so a composer's render pass was tone-mapped and its
  `OutputPass` tone-mapped it again. A target now stores untone-mapped light
  unless it stands in for the picture (`RenderTarget::set_presents`: a headless
  renderer's own target, the WebGL2 canvas buffer), and cube faces never take
  it. Read-back follows `renderTarget.texture.colorSpace`, also settable from
  the `WebGLRenderTarget` options: the default `NoColorSpace` reads back linear
  and `SRGBColorSpace` encoded, as three.js returns them
  (`RenderTarget::bytes_as_color_space`; storage stays encoded, which keeps one
  pipeline per format on every backend). A `HalfFloatType` target reads back
  into a `Uint16Array` of half floats; it returned garbage at four bytes a
  pixel. `setSize` reallocates the target in place under the same id, so its
  `texture` and every material sampling it follow — it was a no-op in wasm,
  and the shim's replacement allocation left `texture` on the old target.
  `MeshPhysicalMaterial` bound only its colour map: its `normalMap`,
  `roughnessMap`, `emissiveMap`, `aoMap` and `displacementMap` were never
  passed on, and are now, live, as Standard's are.

- **An `iridescenceThicknessMap` was never uploaded**, so the first frame that
  drew one panicked looking it up. Every map a material binds is now uploaded
  from the same list the bindings are built from.

- **`EdgesGeometry` matched corners by index**, so an unindexed mesh — FBX,
  face-varying USD — drew every triangle edge, and a `BoxGeometry`, split at its
  seams, drew each edge twice. Corners now match by position rounded to four
  places, as three.js does: indexed and unindexed copies give the same edges, and
  a box gives 12, not 24.

- **Metal wireframes showed the triangulation diagonal.** They now draw the
  polygon outline line list as 1px hardware lines, the same as wgpu; the two
  backends light the same pixels on a test quad.

- **Collada `<polylist>` read every face as a triangle**, ignoring `<vcount>`,
  so any quad scrambled the faces after it. Faces are now fanned by their
  corner count.

- **Metal views after the first in a shared target** lost their MSAA content
  and loaded depth that had never been stored. Both showed up once four views
  shared one target; every view now keeps its colour and clears its own depth.

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

[Unreleased]: https://github.com/eugenehp/threers/compare/v0.0.8...HEAD
[0.0.8]: https://github.com/eugenehp/threers/compare/v0.0.7...v0.0.8
[0.0.7]: https://github.com/eugenehp/threers/compare/v0.0.6...v0.0.7
[0.0.6]: https://github.com/eugenehp/threers/compare/v0.0.5...v0.0.6
[0.0.5]: https://github.com/eugenehp/threers/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/eugenehp/threers/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/eugenehp/threers/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/eugenehp/threers/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/eugenehp/threers/releases/tag/v0.0.1
