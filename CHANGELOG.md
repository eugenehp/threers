# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.6] — 2026-09-13

Universal Scene Description, read and written in all three of its forms and
checked against Apple's own tools rather than against itself; an SVG renderer
that draws the scene graph as vectors; and the fixes that writing both turned
up.

### Added
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

### Fixed
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
