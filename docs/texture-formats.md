# Texture formats

Part of the [threers](../README.md) documentation.

`TextureFormat` names how a texture's bytes are laid out. A texture stays in
its own format on the GPU wherever the GPU can sample it, and is converted on
upload where it cannot — so the same texture draws on a desktop, a phone, a
headset and a browser, small where the hardware allows and correct everywhere.

| Family | Formats | Bytes |
|---|---|---|
| 8-bit | `Rgba8UnormSrgb`, `Rgba8Unorm`, `R8Unorm`, `Rg8Unorm` | 1–4 a pixel |
| 16-bit unorm | `R16Unorm`, `Rgba16Unorm` (16-bit PNGs, height maps) | 2, 8 |
| Packed | `Rgb10a2Unorm`, `Rg11b10Ufloat`, `Rgb9e5Ufloat` (compact HDR) | 4 |
| Float | `R16Float`, `Rg16Float`, `Rgba16Float`, `R32Float`, `Rg32Float`, `Rgba32Float` | 2–16 |
| BC | `Bc1`, `Bc3`, `Bc7` (sRGB and linear), `Bc4RUnorm`, `Bc5RgUnorm`, `Bc6hRgbUfloat`, `Bc6hRgbFloat` | 8 or 16 a 4x4 block |
| ETC2 | `Etc2Rgb8`, `Etc2Rgba8` (sRGB and linear) | 8 or 16 a 4x4 block |
| ASTC | `Astc { block, srgb }`, LDR, every footprint from 4x4 to 12x12 | 16 a block |

`TextureFormat::with_srgb` gives a format's sRGB or linear twin (three.js's
`colorSpace`), and `code` a number that never changes between versions, which
cache keys and the `.tex` blob store.

## On a GPU that lacks a format

`textures::codec::for_device` decides, from what the device samples
(`codec::Support`, or `gpu_texture::support` for a wgpu device's features),
and both the wgpu and Metal backends call it on upload:

- **A compressed family the GPU lacks** — BC on phones and older Apple GPUs,
  ETC2 and ASTC on most desktops — decodes on the CPU, to RGBA8 for colour
  (sRGB kept), R8 for BC4, RG8 for BC5 and half-float RGBA for BC6H. A
  compressed base level that is not a whole number of blocks is refused on
  every device, by name: no API accepts one.
- **16-bit unorm** without `TEXTURE_FORMAT_16BIT_NORM` becomes 32-bit float
  where that filters, else half float (11 of the 16 bits).
- **32-bit float** without `FLOAT32_FILTERABLE` becomes half float, since the
  renderer's texture bindings filter.

Every device the crate creates asks for all of these features the adapter has
(`gpu::TEXTURE_FEATURES`); asking cannot fail, and the conversion covers the
rest.

## The decoders

`textures::codec::decode` expands any block format; the CPU path tracer, the
post-processing upload, the USD exporter and the tensor conversions read pixels
through it too. Each follows its specification's integer arithmetic. BC1–BC5
follow what Apple's GPUs do inside the tolerance the D3D specification allows
(BC7's weights in 64ths, rounded).

They are held to hardware, not to themselves: `tests/texture_formats.rs` uploads
random blocks of every format natively, reads each texel back through
`textureLoad`, and compares. The blocks are steered to cover every BC7 and BC6H
mode, and, for ASTC, valid blocks of every grid, partition count and endpoint
mode, plus void-extent and malformed ones. On an M4 Pro: BC1, BC3's colour,
BC6H, BC7, ETC2 and sRGB ASTC decode bit for bit; BC4, BC5 and linear ASTC,
which the GPU returns at more than 8 bits, to within half a step, which is
what an 8-bit decode can reach. The same test runs the upload through a device
with none of the features and checks it samples what the native device does.
`tests/metal_texture_formats.rs` draws every format through the Metal backend
against wgpu.

Two decoder details only the hardware settled:

- ASTC's last partial group of trits or quints reads its missing bits as zero,
  not as whatever follows in the block — reading on corrupted one valid block in
  seven at the trit and quint ranges.
- An ASTC block with an HDR endpoint mode in one partition gives the error
  colour (magenta) for that partition's texels only; the rest decode normally.

There are no encoders here other than BC1 (`textures::bc1`): compressed
textures come from offline tools, through [KTX2](ktx2.md) or the `.tex` blob.
