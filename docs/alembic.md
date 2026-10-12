# Alembic

Part of the [threers](../README.md) documentation.

`.abc` is what VFX pipelines use for baked geometry and point caches — the
natural neighbour to [USD](usd.md): same audience, same kind of file, and the
same sampling model, since an Alembic property holds a sequence of samples
against a time sampling.

```rust
use threers::loaders::alembic::Archive;

let bytes = std::fs::read("crowd.abc")?;
let archive = Archive::open(&bytes)?;
for mesh in archive.poly_meshes() {
    println!("{}: {} points", mesh.name, mesh.points.len() / 3);
}
```

## The container

Ogawa is two node kinds and nothing else: a **group** is a 64-bit child count
followed by that many 64-bit references, and a **data** block is a 64-bit size
followed by bytes. A reference's top bit says which it points at. Everything
above that is convention about which child means what — an object group holds its
properties first, then its children, then a block naming them.

Two details wreck everything after them if missed:

- an object-header block ends with **32 bytes of hashes** that are not headers,
  and reading them as one yields a name of several gigabytes;
- every array sample begins with a **16-byte key** that is not part of the data,
  and reading it as values shifts a mesh by four floats — which produces a mesh
  that is almost right.

Property headers are a packed bitfield whose size hint decides how wide the
integers after it are, so a wrong hint reads the name as a number.

## What is read

The object hierarchy, and `PolyMesh` geometry from it: `P`, `.faceCounts` and
`.faceIndices`. A file whose completion byte is still clear — a cache that is
mid-write — is refused rather than read from offsets that are not final.

`Xform` transforms, UVs, normals and velocities are named in the property tree
but not yet turned into a scene; subdivision meshes, curves and point clouds are
not read. The HDF5 backing, which predates Ogawa, is a different container and is
not read at all.
