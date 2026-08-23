//! Dense f32 tensors, and the conversions between them and the two things
//! threers has a lot of: vertex attributes and pixels.
//!
//! [`crate::rlx::tensor::Tensor`] is deliberately the smallest type that can cross the boundary —
//! a flat `Vec<f32>` plus static dimensions. rlx's [`Shape`] is *derived* from
//! it rather than stored in it, because a shape can carry dynamic dimensions
//! and a buffer cannot: a tensor whose second dimension is only known at
//! runtime has no defensible `len()`, and every conversion here needs one.

use ::rlx::{DType, Dim, Shape};

use crate::core::{BufferAttribute, BufferGeometry};
use crate::textures::{pack_rgba16f, Texture, TextureFormat};

use super::{linear_to_srgb, srgb_to_linear};

/// Where the channels sit in an image tensor.
///
/// Both carry a leading batch dimension of 1, so a converted frame drops
/// straight into a graph that was traced on batched input. Rank-3 tensors
/// (no batch) are accepted on the way *in*; the way out is always rank 4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// `[1, height, width, channels]` — the pixel order the framebuffer is
    /// already in, so the conversion is a copy and a scale.
    Nhwc,
    /// `[1, channels, height, width]` — one plane per channel, which is what
    /// convolution stacks ported from PyTorch expect.
    Nchw,
}

/// What the *tensor's* numbers mean — a request, not a description of the
/// source. Which side of the transfer function the source is on is the texture
/// format's business: `Rgba8UnormSrgb` holds encoded values (as do the bytes a
/// headless render reads back, whose target is that format by default), and
/// `Rgba8Unorm`, `R8Unorm` and `Rgba16Float` hold linear ones.
///
/// So the conversion applies the transfer function when the two disagree and
/// copies when they agree. Alpha is exempt in both directions: it is a
/// coverage fraction, and encoding it would be encoding a number that was
/// never a colour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    /// sRGB-encoded, `0..=1`. What a network trained on ordinary images
    /// expects, because the images it saw were encoded too.
    Srgb,
    /// Linear light. What arithmetic on the values wants — blending,
    /// filtering, exposure are only meaningful here — and, above 8 bits, what
    /// values greater than 1 mean.
    Linear,
}

/// What a conversion can refuse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    /// `data.len()` is not the product of the dimensions.
    ShapeMismatch { dims: Vec<usize>, len: usize },
    /// The rlx shape has a dimension only known at runtime, so the tensor it
    /// describes has no fixed size.
    DynamicDim,
    /// The rlx shape is not `f32`. Every tensor on this side is.
    NotF32(DType),
    /// Wrong rank for the conversion asked for.
    Rank { expected: &'static str, got: usize },
    /// A channel count no pixel format can express. 1, 3 and 4 can be.
    Channels(usize),
    /// The channel count and the target texture format disagree.
    FormatChannels {
        format: TextureFormat,
        channels: usize,
    },
    /// Byte count and declared size disagree.
    ByteCount { expected: usize, got: usize },
    /// There was nothing to work on.
    ///
    /// Kept as an error rather than answered with an empty result, because the
    /// operations that raise it — fitting a colour transform, clustering a
    /// palette — have no meaningful answer for no data, and because a
    /// zero-length tensor is not a shape any backend accepts: rlx's wgpu
    /// backend panics on a zero-size buffer allocation rather than returning
    /// one.
    Empty,
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeMismatch { dims, len } => {
                write!(f, "{len} values do not fill dims {dims:?}")
            }
            Self::DynamicDim => write!(f, "shape has a dynamic dimension"),
            Self::NotF32(d) => write!(f, "expected an f32 shape, got {d:?}"),
            Self::Rank { expected, got } => write!(f, "expected rank {expected}, got {got}"),
            Self::Channels(c) => write!(f, "{c} channels is not 1, 3 or 4"),
            Self::FormatChannels { format, channels } => {
                write!(f, "{channels} channels cannot be written as {format:?}")
            }
            Self::ByteCount { expected, got } => {
                write!(f, "expected {expected} bytes, got {got}")
            }
            Self::Empty => write!(f, "nothing to work on"),
        }
    }
}

impl std::error::Error for TensorError {}

/// A dense, row-major, statically-shaped f32 tensor.
///
/// The invariant — `data.len() == dims.product()` — is why the fields are
/// private: a tensor that lies about its shape is worse than no tensor,
/// because the graph it feeds will read past the end of one row and call the
/// next row's numbers its own. [`as_mut_slice`](Self::as_mut_slice) hands out
/// a slice rather than the `Vec` for the same reason: values are yours to
/// change, lengths are not.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    data: Vec<f32>,
    dims: Vec<usize>,
}

impl Tensor {
    /// Wrap `data` in `dims`, or fail if it does not fill them exactly.
    pub fn new(data: Vec<f32>, dims: &[usize]) -> Result<Self, TensorError> {
        let want: usize = dims.iter().product();
        if want != data.len() {
            return Err(TensorError::ShapeMismatch {
                dims: dims.to_vec(),
                len: data.len(),
            });
        }
        Ok(Self {
            data,
            dims: dims.to_vec(),
        })
    }

    /// Zeros of the given shape.
    pub fn zeros(dims: &[usize]) -> Self {
        Self {
            data: vec![0.0; dims.iter().product()],
            dims: dims.to_vec(),
        }
    }

    /// A rank-1 tensor over the values as they are.
    pub fn from_flat(data: Vec<f32>) -> Self {
        Self {
            dims: vec![data.len()],
            data,
        }
    }

    /// Wrap `data` in an rlx [`Shape`]. Refuses dynamic dimensions and any
    /// dtype other than `F32` — see the type-level note on why.
    pub fn from_shape(shape: &Shape, data: Vec<f32>) -> Result<Self, TensorError> {
        if shape.dtype() != DType::F32 {
            return Err(TensorError::NotF32(shape.dtype()));
        }
        let mut dims = Vec::with_capacity(shape.rank());
        for d in shape.dims() {
            match d {
                Dim::Static(n) => dims.push(*n),
                Dim::Dynamic(_) => return Err(TensorError::DynamicDim),
            }
        }
        Self::new(data, &dims)
    }

    /// The rlx shape this tensor satisfies — `F32`, all dimensions static.
    /// This is what you hand `Graph::input` so the graph and the buffer agree.
    pub fn shape(&self) -> Shape {
        Shape::new(&self.dims, DType::F32)
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn into_data(self) -> Vec<f32> {
        self.data
    }

    /// Reinterpret the same values under new dimensions.
    pub fn reshape(self, dims: &[usize]) -> Result<Self, TensorError> {
        Self::new(self.data, dims)
    }
}

// ── Vertex attributes ───────────────────────────────────────────────────

/// A `[count, item_size]` tensor over an attribute's values.
///
/// The item size stays a dimension rather than being flattened away, so a
/// positions tensor is `[n, 3]` and a graph can reduce over the components of
/// a vertex without being told separately how many there are.
impl From<&BufferAttribute> for Tensor {
    fn from(attr: &BufferAttribute) -> Self {
        Self {
            dims: vec![attr.count(), attr.item_size],
            data: attr.array.clone(),
        }
    }
}

/// The inverse. Rank 2 only: the second dimension is the item size, and
/// guessing it from a flat buffer would be guessing.
impl TryFrom<&Tensor> for BufferAttribute {
    type Error = TensorError;

    fn try_from(t: &Tensor) -> Result<Self, Self::Error> {
        if t.rank() != 2 {
            return Err(TensorError::Rank {
                expected: "2 ([count, item_size])",
                got: t.rank(),
            });
        }
        Ok(BufferAttribute::new(t.data.clone(), t.dims[1].max(1)))
    }
}

/// One of a geometry's attributes as a `[count, item_size]` tensor, or `None`
/// if it has no attribute by that name.
pub fn geometry_to_tensor(geometry: &BufferGeometry, name: &str) -> Option<Tensor> {
    geometry.get_attribute(name).map(Tensor::from)
}

/// Write a `[count, item_size]` tensor back over a named attribute.
///
/// Goes through `set_attribute`, so the geometry's version stamp, cached
/// bounds and any BVH or surface sidecar are invalidated exactly as they
/// would be by any other write — a tensor that came back from a graph is not
/// a special case.
pub fn tensor_to_geometry(
    geometry: &mut BufferGeometry,
    name: &str,
    tensor: &Tensor,
) -> Result<(), TensorError> {
    let attr = BufferAttribute::try_from(tensor)?;
    geometry.set_attribute(name, attr);
    Ok(())
}

// ── Pixels ──────────────────────────────────────────────────────────────

/// An RGBA8 frame — a headless readback, say — as an image tensor scaled to
/// `0..=1`.
///
/// The bytes are read as sRGB-encoded, which is what
/// [`HeadlessRenderer::read_rgba`](crate::renderer::HeadlessRenderer::read_rgba)
/// returns: its colour target is `Rgba8UnormSrgb` unless you asked for
/// something else. Pass [`ColorSpace::Linear`] to undo that encoding on the
/// way in.
pub fn frame_to_tensor(
    rgba: &[u8],
    width: u32,
    height: u32,
    layout: Layout,
    space: ColorSpace,
) -> Result<Tensor, TensorError> {
    let expected = width as usize * height as usize * 4;
    if rgba.len() != expected {
        return Err(TensorError::ByteCount {
            expected,
            got: rgba.len(),
        });
    }
    let decode = matches!(space, ColorSpace::Linear);
    Ok(pack(width, height, 4, layout, |i, c| {
        let v = rgba[i * 4 + c] as f32 / 255.0;
        if decode && c < 3 {
            srgb_to_linear(v)
        } else {
            v
        }
    }))
}

/// An image tensor back to RGBA8 bytes, with the width and height it turned
/// out to have — a graph is free to return a different size than it was given,
/// and this reports what it did rather than assuming it did not.
///
/// Values outside `0..=1` are clamped: 8 bits cannot hold them, and letting
/// them wrap would turn an over-bright highlight into a dark one.
pub fn tensor_to_frame(
    tensor: &Tensor,
    layout: Layout,
    space: ColorSpace,
) -> Result<(Vec<u8>, u32, u32), TensorError> {
    let (width, height, channels) = image_dims(tensor, layout)?;
    let encode = matches!(space, ColorSpace::Linear);
    let mut out = vec![0u8; width as usize * height as usize * 4];
    for i in 0..(width as usize * height as usize) {
        for c in 0..4 {
            out[i * 4 + c] = if c < channels {
                let v = at(tensor, layout, width, height, channels, i, c);
                let v = if encode && c < 3 {
                    linear_to_srgb(v)
                } else {
                    v
                };
                quantize(v)
            } else if c == 3 {
                255
            } else {
                // A 1-channel tensor is grey, not red.
                let v = at(tensor, layout, width, height, channels, i, 0);
                let v = if encode { linear_to_srgb(v) } else { v };
                quantize(v)
            };
        }
    }
    Ok((out, width, height))
}

/// A [`Texture`]'s pixels as an image tensor.
///
/// The decode is driven by the texture's own format, which is the only thing
/// that knows whether its bytes are encoded: `Rgba8UnormSrgb` is, the rest are
/// not. Asking for [`ColorSpace::Linear`] on a linear format is therefore not
/// an error and not a conversion — it is already true.
pub fn texture_to_tensor(
    texture: &Texture,
    layout: Layout,
    space: ColorSpace,
) -> Result<Tensor, TensorError> {
    let (width, height) = (texture.width, texture.height);
    let pixels = width as usize * height as usize;
    let bpp = texture.bytes_per_pixel();
    let expected = pixels * bpp;
    if texture.data.len() != expected {
        return Err(TensorError::ByteCount {
            expected,
            got: texture.data.len(),
        });
    }
    let channels = match texture.format {
        TextureFormat::R8Unorm => 1,
        _ => 4,
    };
    let stored_srgb = matches!(texture.format, TextureFormat::Rgba8UnormSrgb);
    let want_linear = matches!(space, ColorSpace::Linear);
    let bytes = &texture.data;
    Ok(pack(width, height, channels, layout, |i, c| {
        let v = match texture.format {
            TextureFormat::Rgba16Float => {
                let o = i * 8 + c * 2;
                f16_to_f32(u16::from_le_bytes([bytes[o], bytes[o + 1]]))
            }
            _ => bytes[i * bpp + c] as f32 / 255.0,
        };
        match (stored_srgb, want_linear, c) {
            // Alpha is a coverage fraction, not a colour: never transferred.
            (_, _, 3) => v,
            (true, true, _) => srgb_to_linear(v),
            (false, false, _) => linear_to_srgb(v),
            _ => v,
        }
    }))
}

/// An image tensor as a [`Texture`] in the requested format, ready to hand to
/// a material.
///
/// The encoding mirrors [`texture_to_tensor`]: a linear tensor written to
/// `Rgba8UnormSrgb` is encoded, a linear tensor written to a linear format is
/// copied, and alpha is never transferred. Channel counts must fit the format
/// — 1 for `R8Unorm`, 3 or 4 for the rest (3 gets an opaque alpha).
pub fn tensor_to_texture(
    tensor: &Tensor,
    format: TextureFormat,
    layout: Layout,
    space: ColorSpace,
) -> Result<Texture, TensorError> {
    let (width, height, channels) = image_dims(tensor, layout)?;
    let pixels = width as usize * height as usize;
    let fits = match format {
        TextureFormat::R8Unorm => channels == 1,
        _ => channels == 3 || channels == 4,
    };
    if !fits {
        return Err(TensorError::FormatChannels { format, channels });
    }
    let stored_srgb = matches!(format, TextureFormat::Rgba8UnormSrgb);
    let want_linear = matches!(space, ColorSpace::Linear);
    let sample = |i: usize, c: usize| -> f32 {
        let v = at(tensor, layout, width, height, channels, i, c);
        match (stored_srgb, want_linear, c) {
            (_, _, 3) => v,
            (true, true, _) => linear_to_srgb(v),
            (false, false, _) => srgb_to_linear(v),
            _ => v,
        }
    };
    let data = match format {
        TextureFormat::R8Unorm => (0..pixels).map(|i| quantize(sample(i, 0))).collect(),
        TextureFormat::Rgba16Float => {
            let mut rgba = Vec::with_capacity(pixels * 4);
            for i in 0..pixels {
                for c in 0..4 {
                    rgba.push(if c < channels { sample(i, c) } else { 1.0 });
                }
            }
            pack_rgba16f(&rgba)
        }
        _ => {
            let mut out = Vec::with_capacity(pixels * 4);
            for i in 0..pixels {
                for c in 0..4 {
                    out.push(if c < channels {
                        quantize(sample(i, c))
                    } else {
                        255
                    });
                }
            }
            out
        }
    };
    Ok(Texture::new(width, height, format, data))
}

// ── Internals ───────────────────────────────────────────────────────────

/// Build an image tensor by asking `value(pixel_index, channel)` for every
/// component, in whichever order `layout` wants them.
fn pack(
    width: u32,
    height: u32,
    channels: usize,
    layout: Layout,
    value: impl Fn(usize, usize) -> f32,
) -> Tensor {
    let pixels = width as usize * height as usize;
    let mut data = Vec::with_capacity(pixels * channels);
    match layout {
        Layout::Nhwc => {
            for i in 0..pixels {
                for c in 0..channels {
                    data.push(value(i, c));
                }
            }
        }
        Layout::Nchw => {
            for c in 0..channels {
                for i in 0..pixels {
                    data.push(value(i, c));
                }
            }
        }
    }
    let dims = match layout {
        Layout::Nhwc => vec![1, height as usize, width as usize, channels],
        Layout::Nchw => vec![1, channels, height as usize, width as usize],
    };
    Tensor { data, dims }
}

/// One component out of an image tensor, whichever way round it is stored.
fn at(
    tensor: &Tensor,
    layout: Layout,
    width: u32,
    height: u32,
    channels: usize,
    pixel: usize,
    channel: usize,
) -> f32 {
    let pixels = width as usize * height as usize;
    let index = match layout {
        Layout::Nhwc => pixel * channels + channel,
        Layout::Nchw => channel * pixels + pixel,
    };
    tensor.data[index]
}

/// `(width, height, channels)` for a rank-3 or rank-4 image tensor.
///
/// Rank 3 is read as the same thing without the batch dimension, which is what
/// a graph that never batched will hand back.
fn image_dims(tensor: &Tensor, layout: Layout) -> Result<(u32, u32, usize), TensorError> {
    let d = tensor.dims();
    let spatial = match d.len() {
        4 => &d[1..],
        3 => d,
        got => {
            return Err(TensorError::Rank {
                expected: "3 or 4 (an image)",
                got,
            })
        }
    };
    let (height, width, channels) = match layout {
        Layout::Nhwc => (spatial[0], spatial[1], spatial[2]),
        Layout::Nchw => (spatial[1], spatial[2], spatial[0]),
    };
    if !matches!(channels, 1 | 3 | 4) {
        return Err(TensorError::Channels(channels));
    }
    Ok((width as u32, height as u32, channels))
}

fn quantize(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// IEEE half → f32. The encode direction is [`pack_rgba16f`]'s.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let out = match exp {
        // Zero or subnormal: scale the mantissa in as a float rather than
        // rebuilding the exponent by hand.
        0 => {
            if mant == 0 {
                sign << 31
            } else {
                let v = mant as f32 * (1.0 / 16_777_216.0); // 2^-24
                return if sign == 1 { -v } else { v };
            }
        }
        // Inf / NaN.
        0x1f => (sign << 31) | 0x7f80_0000 | (mant << 13),
        _ => (sign << 31) | ((exp + 112) << 23) | (mant << 13),
    };
    f32::from_bits(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_attribute_round_trips_through_a_tensor() {
        let attr = BufferAttribute::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 3);
        let t = Tensor::from(&attr);
        assert_eq!(t.dims(), &[2, 3]);
        let back = BufferAttribute::try_from(&t).unwrap();
        assert_eq!(back.array, attr.array);
        assert_eq!(back.item_size, 3);
    }

    #[test]
    fn a_shape_with_a_dynamic_dim_is_refused() {
        let shape = Shape::from_dims(&[Dim::Static(2), Dim::Dynamic(0)], DType::F32);
        assert_eq!(
            Tensor::from_shape(&shape, vec![0.0; 4]),
            Err(TensorError::DynamicDim)
        );
    }

    #[test]
    // `1 * w + x` is row-and-column arithmetic written out: the row index
    // stays visible next to the column instead of being folded away.
    #[allow(clippy::identity_op)]
    fn the_two_layouts_hold_the_same_pixels_in_a_different_order() {
        let rgba: Vec<u8> = (0..(2 * 2 * 4) as u8).collect();
        let nhwc = frame_to_tensor(&rgba, 2, 2, Layout::Nhwc, ColorSpace::Srgb).unwrap();
        let nchw = frame_to_tensor(&rgba, 2, 2, Layout::Nchw, ColorSpace::Srgb).unwrap();
        assert_eq!(nhwc.dims(), &[1, 2, 2, 4]);
        assert_eq!(nchw.dims(), &[1, 4, 2, 2]);
        // Pixel 1, channel 2 is the same value from either side.
        assert_eq!(nhwc.data()[1 * 4 + 2], nchw.data()[2 * 4 + 1]);
        for layout in [Layout::Nhwc, Layout::Nchw] {
            let t = frame_to_tensor(&rgba, 2, 2, layout, ColorSpace::Srgb).unwrap();
            let (back, w, h) = tensor_to_frame(&t, layout, ColorSpace::Srgb).unwrap();
            assert_eq!((w, h), (2, 2));
            assert_eq!(back, rgba);
        }
    }

    #[test]
    fn an_srgb_texture_decodes_to_linear_and_back() {
        let tex = Texture::new(1, 1, TextureFormat::Rgba8UnormSrgb, vec![188, 128, 0, 200]);
        let t = texture_to_tensor(&tex, Layout::Nhwc, ColorSpace::Linear).unwrap();
        // 188/255 encoded ≈ 0.5 linear, and alpha stays a fraction.
        assert!((t.data()[0] - 0.5).abs() < 0.01, "{}", t.data()[0]);
        assert!((t.data()[3] - 200.0 / 255.0).abs() < 1e-6);
        let back = tensor_to_texture(
            &t,
            TextureFormat::Rgba8UnormSrgb,
            Layout::Nhwc,
            ColorSpace::Linear,
        )
        .unwrap();
        assert_eq!(*back.data, *tex.data);
    }

    #[test]
    fn a_linear_texture_is_copied_not_transferred() {
        let tex = Texture::new(1, 1, TextureFormat::Rgba8Unorm, vec![188, 128, 0, 200]);
        let t = texture_to_tensor(&tex, Layout::Nhwc, ColorSpace::Linear).unwrap();
        assert!((t.data()[0] - 188.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn half_floats_survive_the_round_trip() {
        let tex = Texture::new(
            1,
            1,
            TextureFormat::Rgba16Float,
            pack_rgba16f(&[4.5, 0.25, 0.0, 1.0]),
        );
        let t = texture_to_tensor(&tex, Layout::Nhwc, ColorSpace::Linear).unwrap();
        assert_eq!(t.data(), &[4.5, 0.25, 0.0, 1.0]);
        let back = tensor_to_texture(
            &t,
            TextureFormat::Rgba16Float,
            Layout::Nhwc,
            ColorSpace::Linear,
        )
        .unwrap();
        assert_eq!(*back.data, *tex.data);
    }

    #[test]
    fn a_wrong_length_buffer_is_refused_rather_than_padded() {
        assert_eq!(
            Tensor::new(vec![0.0; 5], &[2, 3]),
            Err(TensorError::ShapeMismatch {
                dims: vec![2, 3],
                len: 5
            })
        );
        assert!(frame_to_tensor(&[0; 3], 2, 2, Layout::Nhwc, ColorSpace::Srgb).is_err());
    }
}
