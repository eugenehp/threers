# Draco geometry compression

Part of the [threers](../README.md) documentation.

Draco is the compression most `.glb` in the wild uses. A Draco primitive keeps
its geometry in an extension rather than in the accessors it declares, so a
loader that ignores `KHR_draco_mesh_compression` and reads those accessors gets
compressed bytes interpreted as floats — which is why this used to refuse such
files outright.

```rust
// Through glTF, which is how you will normally meet it.
let scene = threers::loaders::gltf::GltfLoader::parse_glb(&bytes)?;

// Or directly, for a bare `.drc`.
use threers::loaders::draco;
let mesh = draco::decode_mesh(&std::fs::read("bunny.drc")?)?;
let indices = mesh.indices();                                  // point ids
let positions = mesh.attribute(draco::attributes::attribute_type::POSITION);
```

## What it does

Draco is several formats stacked, and all of them are needed before a single
vertex comes out:

- **rANS** entropy coding, in two shapes: a bit decoder and a tagged symbol
  decoder.
- **Edgebreaker connectivity.** The mesh is walked one triangle at a time,
  emitting one of five symbols (`C`, `S`, `L`, `R`, `E`) that says how each new
  triangle attaches to the last. Decoding replays that walk into a *corner
  table*, which means maintaining a stack of active boundaries — `E` opens one,
  `S` merges two and fuses a pair of vertices, and topology-split events stitch
  branches back together across the traversal.
- **Two traversals.** The *standard* one writes the symbols out. The *valence*
  one does not: it keeps six symbol streams, one per vertex valence from 2 to 7,
  and picks the stream for the next symbol from the valence of the vertex the
  traversal is standing on **in the partly-built mesh**. So symbols cannot be
  read ahead of the replay at all — the replay pulls them one at a time and
  reports back what it built.
- **Attribute seams.** A corner-table vertex is one position; a *point* is one
  unique combination of attribute values. A texture seam gives one position
  several points. Each non-position attribute therefore gets its own corner
  table with the seam edges cut, and the index buffer is in points.
- **Prediction.** Values are stored as corrections to a prediction made from
  neighbours already decoded, in an order set by a depth-first walk of the
  surface.

## What decodes

| | |
|---|---|
| Traversals | standard, **valence** |
| Connectivity | edgebreaker, topology splits, boundaries, multiple components |
| Prediction | difference, parallelogram, constrained multi-parallelogram, geometric normal, portable texture coordinates |
| Transforms | wrap, canonicalized octahedral |
| Attributes | quantized floats, octahedral normals, plain integers, uncompressed |

That covers **Draco's own encoder at its default setting**. `draco_encoder`
defaults to `-cl 7`, which selects the valence traversal *and* the portable UV
predictor; both are implemented, and there is a fixture pinning that exact case.

## What is refused, and why by name

- **Prediction-degree traversal** — `-cl 10` only. It orders attribute values by
  a different traverser (`MaxPredictionDegreeTraverser`) rather than depth-first.
- **Multi-parallelogram** and the **deprecated** texture-coordinate predictor —
  older encoders.
- **Uncanonicalized octahedral** normals, and the **delta** prediction transform.
- **Sequential connectivity**, **point clouds**, **metadata** blocks, per-face
  attributes, and **bitstream 1.x**.

Each of these comes back as `DracoError::Unsupported` naming the feature. A
geometry decoder that guesses produces a plausible mesh that is quietly the wrong
shape, so none of them fall through to a default.

## How it is verified

Every stage is compared against **Google's own `draco3d` decoder on the same
file**, not against this decoder's previous output:

- Connectivity, by the **valence histogram** over unique positions — a
  fingerprint that a differently-stitched mesh of the same Euler characteristic
  fails.
- The index buffer, **index for index**, via a checksum over all of it.
- Every attribute value, **bit for bit** — both implementations do the same
  arithmetic in the same order on `f32`, so agreement is exact, and anything
  approximate would mean an operation had been reordered or widened.

Four fixtures across four topologies and three encoder settings: a closed sphere
(χ=2), the Stanford bunny with five holes (χ=−3), an open disc (χ=1), and a
default-encoded (`-cl 7`) grid. Through glTF, the decoded face set is compared
against an **uncompressed twin** of the same mesh — geometry that never went
through this code.

## Notes for anyone changing this

Three orderings are load-bearing and each reads like a mistake:

- Constrained multi-parallelogram reads its crease flags **before** its
  transform's bounds. Geometric normal reads its transform data **before** its
  flip bits. Opposite ways round.
- Within one attribute decoder, *every* attribute's corrections are read, then
  *every* attribute's quantization parameters, then the transforms are applied —
  not attribute by attribute.
- Building an attribute table, finding a ring's first corner swings with the
  seam-aware `swing_left`, and the walk that follows swings on the **base**
  table. Using the seam-aware swing for both stops at the first cut and numbers
  only part of the ring.

Corrections are zig-zag coded unless the *transform* says they are already
positive — the octahedral one does, the wrap one does not. Reading that from the
transform rather than assuming it is what keeps normals from being halved.

The integer arithmetic in the predictors wraps in the unsigned domain where
Draco's does, and `int_sqrt` is an exact integer Newton iteration rather than a
float `sqrt`. Both are for agreement on the last bit, not for speed.
