# glTF

Part of the [threers](../README.md) documentation.

Read and written, `.gltf` and `.glb`.

## Reading

```rust
use threers::loaders::UsdLoader;          // see also GltfLoader
let scene = threers::loaders::GltfLoader::parse_glb(&std::fs::read("model.glb")?)?;
```

Extensions understood: `KHR_lights_punctual`, `KHR_materials_clearcoat`,
`KHR_materials_emissive_strength`, `KHR_materials_ior`,
`KHR_materials_iridescence`, `KHR_materials_sheen`,
`KHR_materials_transmission`, `KHR_materials_unlit`, `KHR_mesh_quantization`
and **`EXT_meshopt_compression`**.

### Compression

A large share of the `.glb` on the web is compressed, and a loader that cannot
decode it fails on real files while passing every test built from files it wrote
itself. `EXT_meshopt_compression` is decoded in full — the vertex codec at both
versions, the triangle and index-sequence codecs, and the octahedral,
quaternion, exponential and YCoCg-colour filters.

It is a port of meshoptimizer's *reference* decoder, not a reading of the
specification. A codec written from recollection decodes to plausible numbers
rather than to an error, and the numbers here are vertex positions, so being
wrong looks like a modelling mistake.

`KHR_draco_mesh_compression` is **not** decoded. Draco is listed in
`extensionsRequired`, so such a file is refused rather than returned as an empty
or half-read mesh — a test pins that, because coming back with the wrong
geometry would be worse than an error.

## Writing

```rust
use threers::loaders::{scene_to_glb, scene_to_gltf};

std::fs::write("model.glb", scene_to_glb(&arena, &roots))?;

let (json, bin) = scene_to_gltf(&arena, &roots);   // the pair, for a .gltf
std::fs::write("model.gltf", json)?;
std::fs::write("scene.bin", bin)?;                  // the name the JSON uses
```

What goes out: the scene graph with its transforms as TRS, positions, normals,
texture coordinates, vertex colours and indices, and materials as
`pbrMetallicRoughness` with `emissiveFactor`, `alphaMode`/`alphaCutoff` and
`doubleSided`. Maps are embedded as PNG.

A geometry or texture shared by several objects is written once — it is the
`Arc` that says they are the same — so a forest of a thousand identical trees
costs one mesh and one image.

Both forms pass the glTF validator with no errors, warnings, infos or hints.
