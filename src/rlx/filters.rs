//! Image and height-field convolution, as compiled rlx graphs.
//!
//! A 3×3 convolution is a shader in the renderer and an `Op::Conv2d` here, and
//! the difference is not the arithmetic — it is that this one is a *graph*.
//! Build it once, run it per frame, fuse it with whatever else the graph does,
//! and differentiate it if you need to. The renderer's post-processing chain
//! remains the right place for effects that belong to the frame; this is the
//! right place for effects that belong to a pipeline that also does other
//! tensor work.
//!
//! ```no_run
//! use threers::rlx::{preferred_device, ConvFilter, Kernel3x3};
//!
//! # let (width, height) = (64, 64);
//! # let frame = vec![0u8; (width * height * 4) as usize];
//! let mut sharpen = ConvFilter::new(width, height, Kernel3x3::SHARPEN, preferred_device());
//! let out = sharpen.apply(&frame).unwrap();
//! ```
//!
//! # Edges
//!
//! `Op::Conv2d` pads with zeros, which pulls a blurred image towards black in
//! a one-pixel frame and a height field towards sea level at its border. So
//! the input is edge-replicated on the host *before* it becomes a tensor and
//! the convolution runs unpadded: the border is handled correctly rather than
//! handled afterwards.

use ::rlx::{DType, Device, Graph, GraphExt, Shape};

use super::session::GraphRunner;
use super::tensor::{tensor_to_frame, ColorSpace, Layout, Tensor, TensorError};

/// A 3×3 kernel, row-major from the top-left, with the presentation it wants.
///
/// `bias` and `rectify` are part of the kernel rather than of the filter
/// because they are part of what the kernel *means*: an emboss without its
/// half-scale offset is a black image with a few highlights, and a Sobel
/// without a rectifier throws away every edge whose gradient runs the other
/// way. A preset that needs handling carries its handling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Kernel3x3 {
    /// Taps, row-major: `[(-1,-1), (0,-1), (1,-1), (-1,0), …]`.
    pub weights: [f32; 9],
    /// Added to every sample after the convolution.
    pub bias: f32,
    /// Take the absolute value of the response. For a derivative kernel, whose
    /// output is signed and would otherwise clamp half of every edge to zero.
    pub rectify: bool,
}

impl Kernel3x3 {
    /// A kernel with no offset and no rectifier.
    pub const fn new(weights: [f32; 9]) -> Self {
        Self {
            weights,
            bias: 0.0,
            rectify: false,
        }
    }

    /// Passes the image through unchanged. Useful as a control: anything this
    /// filter does to an identity-convolved frame is the conversion, not the
    /// convolution.
    pub const IDENTITY: Self = Self::new([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

    /// Unweighted 3×3 mean.
    pub const BOX_BLUR: Self = Self::new([
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
    ]);

    /// Binomial (1 2 1)ᵀ(1 2 1) / 16 — the 3×3 Gaussian.
    pub const GAUSSIAN: Self = Self::new([
        1.0 / 16.0,
        2.0 / 16.0,
        1.0 / 16.0,
        2.0 / 16.0,
        4.0 / 16.0,
        2.0 / 16.0,
        1.0 / 16.0,
        2.0 / 16.0,
        1.0 / 16.0,
    ]);

    /// Identity plus the negative Laplacian: an unsharp mask in one pass.
    pub const SHARPEN: Self = Self::new([0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0]);

    /// The 4-neighbour Laplacian. Zero on flat regions, so it reads as an edge
    /// detector once rectified.
    pub const LAPLACIAN: Self = Self {
        weights: [0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0],
        bias: 0.0,
        rectify: true,
    };

    /// Horizontal gradient (vertical edges).
    pub const SOBEL_X: Self = Self {
        weights: [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0],
        bias: 0.0,
        rectify: true,
    };

    /// Vertical gradient (horizontal edges).
    pub const SOBEL_Y: Self = Self {
        weights: [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0],
        bias: 0.0,
        rectify: true,
    };

    /// Directional derivative rendered as relief: mid-grey is flat, and the
    /// bias is what puts flat at mid-grey.
    pub const EMBOSS: Self = Self {
        weights: [-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0],
        bias: 0.5,
        rectify: false,
    };

    /// The same taps with the sum scaled to 1, for a kernel written by hand.
    /// A kernel that sums to zero is returned unchanged — it is a derivative,
    /// and normalising it would divide by zero.
    pub fn normalized(mut self) -> Self {
        let sum: f32 = self.weights.iter().sum();
        if sum.abs() > 1e-6 {
            for w in &mut self.weights {
                *w /= sum;
            }
        }
        self
    }
}

/// A compiled 3×3 convolution over RGBA frames.
///
/// The convolution is depthwise (`groups = 4`), so channels do not mix — a
/// blur blurs red with red. Alpha is convolved along with the rest and then
/// *discarded*: the output carries the input's alpha, because coverage is not
/// a colour and a sharpened coverage mask is not a sharper picture.
pub struct ConvFilter {
    runner: GraphRunner,
    width: u32,
    height: u32,
    space: ColorSpace,
}

impl ConvFilter {
    /// Compile a filter for frames of exactly `width × height`.
    ///
    /// Panics on an empty frame: the graph's shapes are fixed at compile time,
    /// and there is no useful convolution of nothing.
    pub fn new(width: u32, height: u32, kernel: Kernel3x3, device: Device) -> Self {
        assert!(
            width > 0 && height > 0,
            "ConvFilter on a {width}×{height} frame"
        );
        let graph = conv_graph(width, height, 4, kernel);
        let mut runner = GraphRunner::new(graph, device);
        // Depthwise: one plane of taps per channel, all four the same.
        let mut taps = Vec::with_capacity(36);
        for _ in 0..4 {
            taps.extend_from_slice(&kernel.weights);
        }
        runner.set_param("kernel", &taps);
        Self {
            runner,
            width,
            height,
            space: ColorSpace::Linear,
        }
    }

    /// What the convolution's numbers mean. `Linear` by default: a blur is a
    /// weighted average of light, and averaging sRGB-encoded values darkens
    /// the result.
    ///
    /// `Srgb` is not wrong, only different — it is what an image editor does,
    /// and what a filter tuned on encoded images expects.
    pub fn color_space(mut self, space: ColorSpace) -> Self {
        self.space = space;
        self
    }

    /// Convolve one RGBA8 frame. The result is the same size, with the input's
    /// alpha.
    pub fn apply(&mut self, rgba: &[u8]) -> Result<Vec<u8>, TensorError> {
        let expected = self.width as usize * self.height as usize * 4;
        if rgba.len() != expected {
            return Err(TensorError::ByteCount {
                expected,
                got: rgba.len(),
            });
        }
        let input = padded_frame(rgba, self.width, self.height, self.space);
        let out = self.runner.run(&[("frame", &input)]).remove(0);
        let (mut bytes, _, _) = tensor_to_frame(&out, Layout::Nchw, self.space)?;
        for (dst, src) in bytes.chunks_exact_mut(4).zip(rgba.chunks_exact(4)) {
            dst[3] = src[3];
        }
        Ok(bytes)
    }

    pub fn runner_mut(&mut self) -> &mut GraphRunner {
        &mut self.runner
    }
}

/// Iterated Laplacian diffusion over a single-channel grid — `h ← h + κ∇²h`.
///
/// This is thermal erosion, not hydraulic: material moves down-slope in
/// proportion to curvature, which rounds peaks and fills pits. There is no
/// water, no sediment and no flow accumulation, so it produces smooth hills
/// rather than drainage networks — say so when showing it off.
///
/// One compiled graph run `n` times, with the border held fixed by the
/// edge-replicated padding, so a terrain does not sink at its edges.
pub struct Diffusion {
    runner: GraphRunner,
    width: u32,
    height: u32,
}

impl Diffusion {
    /// `rate` is κ per iteration. Above 0.25 the explicit update is unstable
    /// for a 4-neighbour Laplacian — the grid oscillates instead of settling —
    /// so it is clamped there.
    pub fn new(width: u32, height: u32, rate: f32, device: Device) -> Self {
        assert!(
            width > 0 && height > 0,
            "Diffusion on a {width}×{height} grid"
        );
        let graph = diffusion_graph(width, height);
        let mut runner = GraphRunner::new(graph, device);
        let k = rate.clamp(0.0, 0.25);
        runner.set_param("kernel", &[0.0, k, 0.0, k, 1.0 - 4.0 * k, k, 0.0, k, 0.0]);
        Self {
            runner,
            width,
            height,
        }
    }

    /// Run `iterations` passes, feeding each result back in.
    pub fn run(&mut self, field: &[f32], iterations: usize) -> Result<Vec<f32>, TensorError> {
        let expected = self.width as usize * self.height as usize;
        if field.len() != expected {
            return Err(TensorError::ShapeMismatch {
                dims: vec![self.height as usize, self.width as usize],
                len: field.len(),
            });
        }
        let mut current = field.to_vec();
        for _ in 0..iterations {
            let input = padded_grid(&current, self.width, self.height);
            current = self.runner.run(&[("field", &input)]).remove(0).into_data();
        }
        Ok(current)
    }
}

/// `y = |x ⊛ k| + bias`, depthwise over `channels` planes of `width × height`,
/// taking an input already padded by one pixel on every side.
fn conv_graph(width: u32, height: u32, channels: usize, kernel: Kernel3x3) -> Graph {
    let mut g = Graph::new("conv3x3");
    let x = g.input(
        "frame",
        Shape::new(
            &[1, channels, height as usize + 2, width as usize + 2],
            DType::F32,
        ),
    );
    let w = g.param("kernel", Shape::new(&[channels, 1, 3, 3], DType::F32));
    let mut y = g.conv2d(x, w, [3, 3], [1, 1], [0, 0], [1, 1], channels);
    if kernel.rectify {
        y = g.abs(y);
    }
    if kernel.bias != 0.0 {
        let b = g.constant(kernel.bias as f64, DType::F32);
        y = g.add(y, b);
    }
    g.set_outputs(vec![y]);
    g
}

/// One diffusion step as a single convolution: the `1 − 4κ` centre tap folds
/// `h + κ∇²h` into the kernel, so a step is one op rather than three.
fn diffusion_graph(width: u32, height: u32) -> Graph {
    let mut g = Graph::new("diffuse");
    let x = g.input(
        "field",
        Shape::new(&[1, 1, height as usize + 2, width as usize + 2], DType::F32),
    );
    let w = g.param("kernel", Shape::new(&[1, 1, 3, 3], DType::F32));
    let y = g.conv2d(x, w, [3, 3], [1, 1], [0, 0], [1, 1], 1);
    let flat = g.reshape_(y, vec![(height as i64) * (width as i64)]);
    g.set_outputs(vec![flat]);
    g
}

/// RGBA8 → an edge-replicated NCHW tensor of `[1, 4, h+2, w+2]`.
fn padded_frame(rgba: &[u8], width: u32, height: u32, space: ColorSpace) -> Tensor {
    let decode = matches!(space, ColorSpace::Linear);
    plane_stack(width, height, 4, |x, y, c| {
        let v = rgba[(y * width as usize + x) * 4 + c] as f32 / 255.0;
        if decode && c < 3 {
            super::srgb_to_linear(v)
        } else {
            v
        }
    })
}

/// A height grid → an edge-replicated NCHW tensor of `[1, 1, h+2, w+2]`.
fn padded_grid(field: &[f32], width: u32, height: u32) -> Tensor {
    plane_stack(width, height, 1, |x, y, _| field[y * width as usize + x])
}

/// Build `[1, channels, h+2, w+2]` by sampling with clamped coordinates, which
/// is edge replication.
///
/// Empty in either axis has no interior to replicate from, and both callers
/// have already rejected it.
fn plane_stack(
    width: u32,
    height: u32,
    channels: usize,
    sample: impl Fn(usize, usize, usize) -> f32,
) -> Tensor {
    let (w, h) = (width as usize, height as usize);
    debug_assert!(w > 0 && h > 0, "plane_stack on an empty grid");
    let mut data = Vec::with_capacity(channels * (h + 2) * (w + 2));
    for c in 0..channels {
        for py in 0..h + 2 {
            let y = py.saturating_sub(1).min(h - 1);
            for px in 0..w + 2 {
                let x = px.saturating_sub(1).min(w - 1);
                data.push(sample(x, y, c));
            }
        }
    }
    Tensor::new(data, &[1, channels, h + 2, w + 2]).expect("plane_stack fills its own dims")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rlx::preferred_device;

    /// A frame with a vertical edge: dark left half, bright right half.
    fn edge_frame(width: u32, height: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity((width * height * 4) as usize);
        for _ in 0..height {
            for x in 0..width {
                let v = if x < width / 2 { 0 } else { 255 };
                out.extend_from_slice(&[v, v, v, 255]);
            }
        }
        out
    }

    #[test]
    fn the_identity_kernel_returns_the_frame() {
        let frame = edge_frame(8, 4);
        let mut filter = ConvFilter::new(8, 4, Kernel3x3::IDENTITY, preferred_device());
        assert_eq!(filter.apply(&frame).unwrap(), frame);
    }

    #[test]
    fn a_blur_of_a_flat_frame_is_the_same_flat_frame() {
        // The border test: with zero padding the edge pixels would come back
        // darker than the middle, which is exactly the artefact the
        // edge-replicated input exists to avoid.
        let frame = vec![128u8; 6 * 5 * 4];
        let mut filter = ConvFilter::new(6, 5, Kernel3x3::BOX_BLUR, preferred_device());
        let out = filter.apply(&frame).unwrap();
        for (i, px) in out.chunks_exact(4).enumerate() {
            assert!(
                (px[0] as i32 - 128).abs() <= 1,
                "pixel {i} came back {}",
                px[0]
            );
        }
    }

    #[test]
    // `1 * w + x` is row-and-column arithmetic written out: the row index
    // stays visible next to the column instead of being folded away.
    #[allow(clippy::identity_op)]
    fn sobel_finds_the_edge_and_only_the_edge() {
        let frame = edge_frame(8, 4);
        let mut filter = ConvFilter::new(8, 4, Kernel3x3::SOBEL_X, preferred_device());
        let out = filter.apply(&frame).unwrap();
        let row = |x: usize| out[(1 * 8 + x) * 4] as i32;
        assert!(row(3) > 100 || row(4) > 100, "no response at the edge");
        assert_eq!(row(0), 0, "flat region responded");
        assert_eq!(row(7), 0, "flat region responded");
    }

    #[test]
    fn alpha_is_carried_through_not_convolved() {
        let mut frame = edge_frame(8, 4);
        for (i, px) in frame.chunks_exact_mut(4).enumerate() {
            px[3] = (i * 7 % 256) as u8;
        }
        let mut filter = ConvFilter::new(8, 4, Kernel3x3::SHARPEN, preferred_device());
        let out = filter.apply(&frame).unwrap();
        for (a, b) in out.chunks_exact(4).zip(frame.chunks_exact(4)) {
            assert_eq!(a[3], b[3]);
        }
    }

    #[test]
    fn diffusion_flattens_a_spike_without_moving_the_mean() {
        let (w, h) = (16u32, 16u32);
        let mut field = vec![0.0f32; (w * h) as usize];
        field[(8 * w + 8) as usize] = 1.0;
        let before: f32 = field.iter().sum();

        let mut diffusion = Diffusion::new(w, h, 0.2, preferred_device());
        let after_field = diffusion.run(&field, 12).unwrap();

        assert!(
            after_field[(8 * w + 8) as usize] < 0.3,
            "peak survived: {}",
            after_field[(8 * w + 8) as usize]
        );
        assert!(after_field[(8 * w + 7) as usize] > 0.01, "nothing spread");
        let after: f32 = after_field.iter().sum();
        assert!(
            (after - before).abs() < 0.05,
            "mass changed: {before} → {after}"
        );
    }

    #[test]
    fn a_wrongly_sized_frame_is_refused() {
        let mut filter = ConvFilter::new(8, 4, Kernel3x3::IDENTITY, preferred_device());
        assert!(filter.apply(&[0; 16]).is_err());
    }
}
