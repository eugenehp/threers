# Universal Scene Description

Part of the [threers](../README.md) documentation.

# USD (`.usda`, `.usdz`, `.usdc`)

Universal Scene Description, in pure Rust with no external dependency — which is
what keeps it working on wasm, where OpenUSD's C++ cannot go. It is opt-in
because most scenes never touch USD; being pure code, it costs compile time and
nothing else.

```rust
// Cargo.toml: threers = { version = "0.0.5", features = ["usd"] }
use threers::loaders::UsdLoader;

// One entry point for all three: the format is told apart by content, not by
// the file name.
let scene = UsdLoader::parse(&std::fs::read("model.usdz")?)?;
for root in &scene.roots {
    println!("{}", scene.arena.get(*root).unwrap().name);
}
```

```rust
use threers::loaders::{geometry_to_usda, geometry_to_usdc, geometry_to_usdz};
use threers::loaders::{layer_to_usdz, scene_to_usdc, scene_to_usdz, UsdzLayer};

let text = geometry_to_usda(&geometry, "Widget");    // a layer you can read
let binary = geometry_to_usdc(&geometry, "Widget");  // the same document, packed
let archive = geometry_to_usdz(&geometry, "Widget"); // what Quick Look opens
let packed = scene_to_usdz(&arena, &roots, &[]);     // a whole scene graph
let crated = layer_to_usdz(&layer, UsdzLayer::Crate, &[]); // with a binary layer
```

| Form | | Read | Write |
|---|---|---|---|
| `.usda` | the text layer | ✅ | ✅ |
| `.usdc` | the binary *crate* layer | ✅ | ✅ |
| `.usdz` | a zip of layers and textures | ✅ | ✅ (either layer form) |

Composition — sublayers, references, payloads, variants, inherits, specializes —
is implemented over all three ([below](#composition)).

Meshes come across with their points, face counts and indices, normals and
`primvars:st`; transforms as the `xformOp` stack `xformOpOrder` actually names;
the prim hierarchy; and `UsdPreviewSurface` materials. Two things USD does that
a `BufferGeometry` cannot are resolved on the way in rather than pushed onto the
caller: **faces of any size** are fanned into triangles, and a **`faceVarying`
primvar** — a normal or UV that differs per face corner — expands the mesh to
unindexed triangles, since an indexed buffer cannot express it.

QC runs through OpenUSD's own command-line tools, which ship with macOS and are
the authority on what a USD file is. `scripts/usd-qc.sh` writes every test
fixture out as a crate and asks four questions of each: `usdchecker` whether it
is valid and no less valid than the source, `usdtree` whether the prim hierarchy
is the same one, `usdcat --flatten` whether the composed stage is the same, and
`usdrecord` whether **Hydra draws the same picture** from it. The last goes
through composition, schema resolution, geometry and shading in one step — and
carries a trap worth stating: a stage with nothing in view renders blank, and
two blank images match. The report says which fixtures actually drew something,
so the render check is read as evidence only where it is one. Currently 33 pass,
10 of them with geometry in frame.

The `.usda` grammar is covered in full: `reorder` statements, every
list-operation qualifier (`prepend`, `append`, `delete`, `add`, `reorder`) on
every field that takes one, `custom`, `comment` and `documentation` as the
separate fields they are, sublayer time offsets, nested dictionaries with
declared types, and variants with bodies and children of their own. The test
for it is a document that uses all of it, written as a crate by this code and
diffed against OpenUSD's own normalisation — thirty fixtures, byte-identical.

A USD file usually arrives from somewhere else, so the readers are fuzzed
against it: every prefix, every single-byte corruption and a sample of paired
corruptions of real crate files, plus the same for archives and malformed text.
Anything unreadable is an error rather than a panic. An index pointing past the
end of the points is refused at the loader, because everything downstream —
normals, tangents, the renderer — trusts an index buffer to be in range.

## Apple / RealityKit

An asset bound for AR Quick Look is judged against more than the `.usdz` format
itself: RealityKit will only open certain file types inside a package, the
layer must come first, and nothing may be compressed.
`usdz::arkit_issues` reports what can be judged from the bytes — file types,
alignment, order — so a package can be checked before it ships:

```rust
use threers::loaders::arkit_issues;

for issue in arkit_issues(&archive) {
    eprintln!("{issue}");   // "notes.txt" has extension "txt", which RealityKit will not open
}
```

The general `.usdz` writer is deliberately not restricted to that list — the
format permits more, and in-house packages use it — so the check is offered
rather than enforced. What this crate exports passes `usdchecker --arkit`, which
is the authority; the checker here mirrors it for when it is not to hand.

Every material becomes a `UsdPreviewSurface`, because that is the one surface
shader USD has. The ones that are already a colour and a roughness — `Standard`,
`Physical`, `Basic` — cross over intact. `Phong` becomes USD's *specular*
workflow, its exponent converted to a roughness, so a shininess of 200 stays
smoother than one of 30 instead of both landing on the same number. `Mirror`
becomes the smooth metal it is. `Toon`'s banding and `Normal`'s and `Depth`'s
computed colours have no equivalent and do not survive; their colour, where they
have one, still does. A material's *maps* export too, as the network USD reads textures through: a
`UsdUVTexture` per map sampled by a `UsdPrimvarReader_float2`, with a
`UsdTransform2d` between them when the texture is offset, scaled or rotated.
Each slot gets the colour space it needs — `sRGB` for colour, `raw` for data —
and a normal map gets the `scale` and `bias` that turn eight-bit pixels into a
direction, without which every normal points into the surface. The images are
PNG-encoded and come back alongside the layer:

`UsdExport` is the whole of it: a document is a layer *and* the images it
names, and the three forms differ only in what becomes of those.

```rust
let out = usd::UsdExport::scene(&arena, &roots);        // or ::animated, ::geometry
std::fs::write("model.usdz", out.usdz())?;              // carries its images
out.write_to("model.usda", Path::new("out"))?;          // and writes them beside it
out.write_to("model.usdc", Path::new("out"))?;          // same document, binary
```

| | carries images | refers to them |
|---|---|---|
| `.usdz` | yes — a package must be self-contained | — |
| `.usda` | — | `textures/…` beside the layer |
| `.usdc` | — | `textures/…` beside the layer |

Coming back the other way, a `.usdz` needs nothing: `UsdLoader::parse` decodes
its images onto the materials. For the other two, `attach_textures_from_dir`
says where the layer was read from and loads what is beside it — declining any
path that climbs out of that directory, since it was handed a directory and not
the disk. Whatever cannot be resolved stays in `UsdScene::textures` rather than
being dropped.

`scene_to_usdz` needs no such help — a package is required to be self-contained,
so the images travel inside it, and a `.usdz` read back comes with its pictures
already on its materials rather than with a list of ones to go and find. A
`.usda` still reports what it names in `UsdScene::textures`, because the files
beside it are not this crate's to reach for.

Every mesh also carries `primvars:displayColor` — per vertex where the geometry
has colours, otherwise the material's own colour as a constant. It is USD's
fallback shading, and what a reader that never evaluates a shader graph uses, so
a file exported from here looks right in a thumbnailer as well as in a renderer.

The whole of it runs in a browser as well as natively, and is tested there:

```sh
scripts/wasm-browser-test.sh          # headless Chrome, fetched on first run
```

`tests/usd_wasm.rs` carries each test once and runs it twice — as a `#[test]`
and as a `#[wasm_bindgen_test]` — so the two cannot drift apart. The crate
format is the part that earns it: it is all byte layout and shifts, and wasm is
32-bit.

What is exported is checked by exporting one mesh per material kind and asking
`usdcat` what is in the result, then `usdrecord` whether Hydra draws it —
reading it back through this crate only proves the writer and the reader share
an opinion.

Apple's AR extensions — `Preliminary_AnchoringAPI`, the physics and behaviour
schemas, `Preliminary_Text` — are ordinary prims with typed attributes and
survive a round trip because nothing here discards what it does not recognise.

A `.usdz` is a zip with two rules that make it mappable: everything is stored
rather than compressed, and every file's data begins on a 64-byte boundary. Both
are honoured on write, so a texture inside one can be handed to a decoder as a
slice without unpacking. (Deflated entries are still *read*, because files in
the wild are written by tools that did not read the rule.)

**`.usdc`, the binary crate format**, is read and written in full: the LZ4
sections (both directions — there is an encoder as well as a decoder), USD's
two-bit integer packing, the `ValueRep` word that carries a value inline or
points at one, the path table's depth-first walk with its jump table, and the
spec table that ties them together. Names, values and whole field sets are
interned, so a scene of near-identical prims costs a few bytes each rather than
a full copy apiece.

What this crate writes is a little larger than what OpenUSD writes — on a 2.1 MB
source, 1.0 MB against its 706 KB. Two reasons, both deliberate: the LZ4 matcher
here is a simple greedy one, and integer arrays are written plainly where USD
additionally packs them with the delta coder. Both are read either way, so this
costs bytes and nothing else.

Every part of it was worked out against files written by OpenUSD's own
`usdcat`, and the fixtures under `src/loaders/usd/testdata/` are those files —
which matters more for a binary format than for a text one. A sample of what is
not what you would guess: matrices are numbered *ahead* of vectors in the type
enumeration; a quaternion is stored imaginary-part-first but written
real-part-first; an inlined double holds `f32` bits; a string is indexed one
level deeper than a token; a `float` must be inlined or USD reads its own file
offset as the value; and a relationship's variability defaults to *uniform*
where an attribute's defaults to varying.

The tests close the loop in every direction. `.usda` and `.usdc` of one document
are read by independent code paths and required to agree. A document this crate
exported is converted by `usdcat` and read back from the binary. And the crate
files this crate *writes* are handed to `usdcat`, whose output is diffed against
OpenUSD's own normalisation of the source — byte for byte identical, which is a
stronger statement than "it loads".

## Schemas

A stage is not only meshes, and a loader that handles only meshes opens one and
shows a fraction of it — silently, since every other prim is still there as an
empty group.

| USD | becomes |
|---|---|
| `Mesh` | `Mesh`, triangulated, with primvars |
| `Cube` `Sphere` `Cylinder` `Cone` `Capsule` `Plane` | `Mesh`, generated, turned to the `axis` it declares |
| `DistantLight` | `DirectionalLight` |
| `SphereLight` `DiskLight` `CylinderLight` | `PointLight`, or `SpotLight` where a `ShapingAPI` cone is applied |
| `RectLight` | `RectAreaLight` |
| `DomeLight` | `AmbientLight` |
| `Points` | `Points`, with per-point colour and width |
| `BasisCurves` `NurbsCurves` | `LineSegments` through the control points |
| `Camera` | `PerspectiveCamera` or `OrthographicCamera`, on `UsdScene::cameras` |
| `PointInstancer` | one `InstancedMesh` per prototype |
| `SkelRoot` + `Skeleton` + `SkelAnimation` | `SkinnedMesh` with a posed `Skeleton` |

A `PointInstancer` is how a stage holds a forest: one tree and a hundred
thousand positions. Instances are grouped by the prototype they name, because an
instanced draw carries one geometry — a scatter of trees and rocks is two draws,
not one and not a hundred thousand. `orientations`, `scales` and `invisibleIds`
are all applied.

## Materials

A USD material is not a struct of values — it is a graph. The material's
`outputs:surface` is *connected* to a shader, that shader's `diffuseColor` may
be connected to a texture reader, and that reader's `st` to a primvar reader.
Reading only the values sitting on the surface shader gets the constants and
silently misses every texture in the asset, which for anything authored in a DCC
is most of what the material is. The connections are followed, including through
`NodeGraph` pass-throughs.

A mesh with more than one material does not carry a list of them — it carries
`GeomSubset` children, each naming a set of faces and binding its own material.
Since this crate's `Mesh` holds one material, such a mesh becomes a group with
one child per subset, plus one for the faces no subset claimed. Only the
`materialBind` family counts: a mesh may carry subsets for a modeller's
selection sets or a simulation's regions, and binding off one of those invents
an assignment nobody made.

`material:binding` is **inherited**: it applies to the prim it is on and
everything beneath it, closest binding winning. A mesh with no binding of its own
is not unbound, and missing that turns a bound asset grey.

Textures come back as requests rather than images — which file, into which slot,
with which wrap modes, scale, bias and colour space — on `UsdScene::textures`.
Decoding needs an image decoder and the files beside the layer, neither of which
belongs in a scene-description reader. The scale and bias are not decoration: an
8-bit normal map is *required* to carry `(2,2,2,1)` and `(-1,-1,-1,0)` so that
`[0,1]` becomes `[-1,1]`, and ignoring them leaves every normal pointing into the
surface.

## Value clips

A film shot does not hold its animation in the layer that describes it. The
layer says *where the animation lives*: a list of files, which is active over
which frames, and how stage time maps onto the time inside them. That is how a
crowd or an effects cache reaches a stage without any one file holding all of
it, and it is read here — both the modern `clips` dictionary and the older flat
`clipAssetPaths` spelling, which plenty of caches were written with.

Two tables do different jobs and are easy to swap: `active` picks which *file*,
`times` picks *when inside it*. Getting them the wrong way round produces an
animation that plays, and plays wrongly — which is why the test compares against
`usdcat --flatten` on the same files rather than against a reading of the spec.

Clips supply values for attributes that already exist; they do not bring
attributes into being. A prim with no `xformOp:translate` declared stays without
one however many clips name it, which is what OpenUSD does and why a topology
layer usually sits underneath.

## Subdivision surfaces

Easy to miss: `subdivisionScheme` defaults to **`catmullClark`**, so a
`UsdGeomMesh` that says nothing at all is a subdivision surface and the points
it carries are a *control cage* rather than the surface. Drawing the cage draws
something visibly more angular than the asset — quietly, since it is a perfectly
good mesh, just the wrong one.

```rust
use threers::loaders::{to_scene_with, SceneOptions};

let scene = to_scene_with(&layer, 0.0, &SceneOptions {
    subdivision_level: 2,
    ..Default::default()
});
```

Refinement is off by default, because each level multiplies the face count by
four and the level is a renderer's decision rather than the file's — which is
why usdview has a complexity slider instead of reading one out of the stage.
Creases, corners and `interpolateBoundary` are honoured; primvars are dropped on
a refined mesh rather than carried across, since a UV authored per corner of the
cage does not index the refined surface.

A skinned character is the one thing USD does not describe where it is used:
the mesh says which skeleton binds it and which joints touch each vertex, the
skeleton says where those joints rest and what the mesh was bound at, and a
third prim says where they are at each frame. All three are read.
`joints = ["Root", "Root/Hip"]` is a hierarchy written flat — the slashes say
what hangs off what — and the order of that list is what every other array in
the schema is indexed by. Rigs with more influences per vertex than the shader
reads keep the heaviest four and renormalise, rather than the first four:
dropping a 0.9 to keep a 0.05 is how a limb ends up on the wrong bone.

`purpose = "guide"` is not drawn, `visibility = "invisible"` is present but not
visible, and `intensity` is scaled by `exposure` as a power of two — which is
how a lighter states a stop rather than a multiplier. A camera's field of view
comes from its aperture and focal length, the conversion every DCC applies, and
`focusDistance` and `fStop` come across for depth of field.

## Composition

A `.usda` file is not a scene — it is one *opinion* about a scene. What a prim
actually is comes from combining opinions across sublayers, references,
payloads, inherits, variants and specializes. That combining is what makes USD a
scene description rather than a mesh container, and `UsdLoader::open` does it;
`UsdLoader::parse` reads a single layer and stops.

```rust
use threers::loaders::{ComposeOptions, FileResolver, UsdLoader};

let bytes = std::fs::read("shots/010/shot.usda")?;
let scene = UsdLoader::open(&bytes, "shots/010/shot.usda", &FileResolver, &ComposeOptions::default())?;
```

An arc may also shift and stretch the animation it pulls in —
`references = @walk.usda@ (offset = 100; scale = 2)` maps a sample at `t` to
`t * scale + offset`, which is what lets one walk cycle start at frame 100 in
one shot and frame 340 in another. Nested arcs compose their shifts.

A layer may declare variables and write them into the paths it references —
`@`"${COLOR}_asset.usda"`@` — which is how one shot layer drives what a whole
stage pulls in. String literals and `${VAR}` substitution are evaluated; an
expression using one of USD's functions is left unresolved rather than guessed
at, so it fails loudly instead of opening the wrong asset quietly.

`foo = None` **blocks** a weaker layer's opinion, which is not the same as a
declaration with no value: the latter takes what is below it and the former
exists to stop it.

`active = false` takes a prim and its children off the stage. A sublayer's time
offset shifts everything in it, composing down a chain of sublayers. `relocates`
renames prims that an arc brought in.

Opinions are ranked strongest-first by USD's **LIVRPS** order — **l**ocal,
**i**nherits, **v**ariants, **r**eferences, **p**ayloads, **s**pecializes — the
first opinion found for a field wins, and children are the union across all of
them. Arcs resolve against the layer that *authored* them, so an
`inherits = </Class>` inside a referenced asset means a class in that asset, not
one in the shot that referenced it. Cycles terminate. Payloads can be left
unloaded (`load_payloads: false`), which is the whole reason USD distinguishes
them from references.

Arcs survive the binary form in both directions. That takes more than reading a
new value type: a crate stores three of these fields under names a document
never uses — `inherits` is `inheritPaths`, `variants` is `variantSelection`,
`variantSets` is `variantSetNames` — and a variant's *body* is not inside the
prim at all but hangs off it at a path of its own, `/Chair{look=oak}`. A
reference item is an asset, a prim path and a layer offset, followed by a
`customData` word that is part of the stride whether or not anything uses it.

Where a name comes from is the caller's business: `FileResolver` reads the disk,
`MemoryResolver` holds layers in memory, and a `UsdzArchive` resolves the layers
*inside itself* — which is what `UsdLoader::open_archive` uses, since there is no
filesystem inside a zip.

The tests check this against `usdcat --flatten`: the same three-layer pipeline —
an asset with a variant set, a base layer that references it, and a shot that
sublayers the base and overrides the variant from a stronger layer — composed by
this crate and by OpenUSD, compared prim by prim and value by value, with no
prim missing and none invented. The same pipeline composes identically when
every layer is a `.usdc`, and — the check that covers the writer as well as the
reader — when every layer is a `.usdc` *this crate wrote*, handed back to
`usdcat --flatten`, whose output is byte-identical to its flattening of the
original text.

`usd::mask` narrows a stage to a set of prims the way `usdcat --mask` does —
each path keeps what it names, everything under it, and the ancestors that
position it, so masking on `/World/Keep` leaves `World`'s transform in place
and drops its other children. A path naming nothing still keeps the ancestors it
does name, which is decided by the path asked for rather than by what is found
along it:

```rust
let stage = usd::compose(&layer, "shot.usda", &resolver, &usd::ComposeOptions::default())?;
let shot = usd::mask(&stage, &["/World/Hero", "/World/Lights"]);
```

What this does not do: there is no lazy stage and no prim index cached between
edits — composition is eager and produces a flat layer, so a mask narrows what
you get without saving the work of composing it, which is the opposite of why
USD has one. `instanceable` composes correctly but does not share prototypes,
which costs memory rather than correctness. One format gap is worth naming: `relocates`
round-trips through `.usdc` as a dictionary rather than as `SdfRelocatesMap`,
which needs crate version 0.11.0 — the data survives, the type label does not.
There is no ground truth to match there: the OpenUSD shipped with macOS cannot
write a relocation map into a crate at all, at layer or prim level, failing with
*"Attempted to pack unsupported type `map<SdfPath, SdfPath>`"*. So a dictionary
that survives is strictly more than the reference implementation manages, and
guessing at the real layout would mean guessing at something unverifiable.

A crate's version is raised only as far as what is in it needs. `pathExpression`
— a collection's `membershipExpression`, the thing that decides which prims a
light lights — is a 0.10.0 value type, so a file holding one is stamped 0.10.0
and a file of plain geometry stays 0.8.0. Stamping everything 0.10.0 would work
and would also be a lie about what the reader has to understand.

## Animation

`timeSamples` are read from both forms and turned into an `AnimationClip`
alongside the scene:

```rust
use threers::loaders::UsdLoader;

let scene = UsdLoader::parse(&std::fs::read("spin.usdc")?)?;
if let Some(clip) = scene.animations.first() {
    println!("{} tracks over {}s", clip.tracks.len(), clip.duration);
}
```

Every kind of animation USD expresses becomes a track: transform stacks,
skeletal joints, blend-shape weights, light intensity and colour, a bound
material's colour, visibility, animated vertices, and value clips. Blend shapes come back on `UsdScene::morph_targets` with sparse offsets
expanded to a delta per vertex. Quaternions interpolate spherically rather than
component by component — the componentwise blend of two rotations is not a
rotation — and `at_time_held` steps between samples for the caches where every
frame is a measurement.

A prim's whole `xformOpOrder` stack is evaluated at every time code any of its
ops is keyed at and then decomposed, so a rotation authored as `rotateXYZ` and
one authored as `orient` produce the same track, and a prim whose translate and
scale are keyed on *different* frames still animates correctly. Channels that
never change are dropped rather than written as flat tracks. USD counts in time
codes and this crate counts in seconds, so `timeCodesPerSecond` is applied on
the way in and undone on the way out.

The scene itself is built at the first frame of the layer's range —
`to_scene_at` gives any other instant, and `UsdLayer::at_time` resolves a whole
layer to one moment.

```rust
use threers::loaders::{animated_scene_to_usda, animated_scene_to_usdc, animated_scene_to_usdz};

let text = animated_scene_to_usda(&arena, &roots, &clips);
let binary = animated_scene_to_usdc(&arena, &roots, &clips);
let archive = animated_scene_to_usdz(&arena, &roots, &clips, &[]);  // plays in Quick Look
```

Animated *vertices* — `points.timeSamples` — are read and resolvable at any
time, but do not become a track: a `KeyframeTrack` drives a transform or a
morph weight, not an arbitrary vertex buffer. The geometry is built at the
frame the scene was taken at.
