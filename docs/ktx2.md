# KTX2 textures

Part of the [threers](../README.md) documentation.

KTX2 is glTF's container for GPU-compressed textures (`KHR_texture_basisu`), and
what an ARKit `.usdz` leans on. The decode target already existed here —
`TextureFormat` carries block-compressed data with a mip chain — so what this
adds is the container.

```rust
let ktx = threers::loaders::ktx2::parse(&std::fs::read("wood.ktx2")?)?;
let texture = ktx.to_texture()?;        // mip chain included
```

Read: the header, the level index, the data format descriptor's transfer
function, key/value metadata (`KTXorientation` and friends), and **zlib**
supercompression. Levels come out of the file largest-first, which is the order
the mip chain wants.

Formats mapped: `R8_UNORM`, `R8G8B8A8_UNORM`/`_SRGB`, `R16G16B16A16_SFLOAT`,
`BC1_RGBA_SRGB_BLOCK`, `BC7_SRGB_BLOCK`.

## What is refused, and why by name

- **Basis Universal** (`ETC1S`, `UASTC`) — a payload that has to be *transcoded*
  to whatever the GPU takes. That is a codec, not a container; ETC1S has its own
  entropy coder and codebooks.
- **zstd** supercompression — this crate has an inflate and no zstd.
- **Linear BC** payloads (`BC1_RGBA_UNORM_BLOCK`, `BC7_UNORM_BLOCK`) — the block
  formats here are all sRGB. A normal map is stored linear, and reading one
  through an sRGB curve is wrong in a way that looks like bad lighting rather
  than like a bug, so it is refused instead of mislabelled. Adding a linear BC
  variant to `TextureFormat` is what would fix it.

Each of those errors names the thing it could not do, because "unsupported" on
its own leaves a caller guessing what to reach for.
