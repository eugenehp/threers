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

Formats mapped: every `VkFormat` that has a [`TextureFormat`](texture-formats.md)
— the 8-, 16- and 32-bit uncompressed ones, the packed `A2B10G10R10`,
`B10G11R11` and `E5B9G9R9`, BC1, BC3, BC4, BC5, BC6H and BC7 (sRGB and linear),
ETC2 RGB and RGBA, and ASTC at all fourteen footprints. `B8G8R8A8` is swizzled
to RGBA on load. A format the GPU cannot sample is decoded at upload (see
[Texture formats](texture-formats.md)), so a KTX2 file draws everywhere.

## What is refused, and why by name

- **Basis Universal** (`ETC1S`, `UASTC`) — a payload that has to be *transcoded*
  to whatever the GPU takes. That is a codec, not a container; ETC1S has its own
  entropy coder and codebooks.
- **zstd** supercompression — this crate has an inflate and no zstd.
- **BC2**, **ETC2 RGB8A1** (punch-through alpha), **EAC R11/RG11** and the
  **signed BC4/BC5** — formats with no `TextureFormat` equivalent yet.

Each of those errors names the thing it could not do, because "unsupported" on
its own leaves a caller guessing what to reach for. sRGB and linear payloads
stay apart: a normal map stored linear and read through an sRGB curve is wrong
in a way that looks like bad lighting rather than like a bug.
