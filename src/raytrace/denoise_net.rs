//! The trained denoiser's forward pass, in plain Rust with no dependencies.
//!
//! The network was trained by `rlx-denoise`, which compiles a graph and runs it
//! on Metal or CUDA. That is the right tool for training and the wrong one for
//! shipping: it is native-only, so it cannot run in a browser, and it is
//! GPL-3.0-only, so linking it would relicense this crate. Both problems have
//! the same fix — the network is *small* and uses four operations:
//!
//! | | |
//! |---|---|
//! | 3x3 convolution | stride 1, padding 1, no bias |
//! | ReLU | |
//! | 2x2 average pool | fixed, not learned |
//! | 2x nearest upsample | fixed, not learned |
//!
//! Twelve convolutions in a U-Net, and a residual add at the end. Written out
//! directly that is a few hundred lines that compile anywhere Rust does,
//! `wasm32` included, and carry no licence but this crate's.
//!
//! # Shape
//!
//! ```text
//! enc0 = relu(conv(input))                    9 -> w0
//! enc1 = relu(conv(pool(enc0)))              w0 -> w1     half resolution
//! enc2 = relu(conv(pool(enc1)))              w1 -> w2     quarter
//! enc3 = relu(conv(pool(enc2)))              w2 -> w3     eighth
//! mid  = relu(conv(enc3))                    w3 -> w3
//! d3   = relu(conv(cat(up(mid),  enc2)))
//! d2   = relu(conv(cat(up(d3),   enc1)))
//! d1   = relu(conv(cat(up(d2),   enc0)))
//! out  = input[0..3] + conv(d1)               residual
//! ```
//!
//! The output is *residual*: the network predicts a correction to the colour it
//! was given, not the colour itself. An all-zero final convolution is therefore
//! the identity, which is why an untrained network returns its input rather
//! than grey.
//!
//! # Speed, measured
//!
//! Roughly **0.7 s** for a 128x128 tile at the default widths on one core, and
//! **1.1 s** at the wide preset. Four times the parameters costs only 1.6x the
//! time, which says this is memory-bound rather than compute-bound: the weights
//! of a wide layer do not fit in cache and the arithmetic waits on them.
//!
//! What that buys, and what it does not. A 1080p frame is about 135 tiles, so
//! across eight workers it lands near **12-19 seconds**. That is usable for
//! denoising a finished render in a browser, and for a progressive render where
//! each pass improves what is already on screen. It is **not** an interactive
//! viewport, and nothing here should be read as claiming one — reaching that
//! needs the convolutions on the GPU, which this crate already has the wgpu
//! plumbing for.
//!
//! [`crate::raytrace::denoise_net::TilePlan`] is what makes the parallel part possible: the pieces share no
//! state and none reads another's output, so N workers produce the same frame
//! as one, which `tiling_a_frame_matches_denoising_it_whole` asserts.

use std::fmt;

/// Input planes the network expects: colour, albedo, normal.
pub const IN_CHANNELS: usize = 9;
/// Output planes: the denoised colour.
pub const OUT_CHANNELS: usize = 3;
/// Both tile dimensions must divide by this — the encoder halves three times.
pub const TILE_MULTIPLE: usize = 8;

const K: usize = 3;
const MAGIC: &[u8; 8] = b"RLXDW004";
const MAGIC_003: &[u8; 8] = b"RLXDW003";
const MAGIC_002: &[u8; 8] = b"RLXDW002";
const MAGIC_001: &[u8; 8] = b"RLXDW001";

/// Why a checkpoint could not be read, or a tile could not be denoised.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DenoiseError {
    /// The file is not a denoiser checkpoint.
    NotACheckpoint,
    /// The file ended in the middle of a field or a tensor.
    Truncated { wanted: usize, got: usize },
    /// A checkpoint this build cannot run, and why.
    Unsupported(String),
    /// The tile is not a multiple of [`TILE_MULTIPLE`], or is empty.
    BadTileSize { width: usize, height: usize },
    /// The input buffer is not `IN_CHANNELS * width * height` long.
    BadInputLength { wanted: usize, got: usize },
}

impl fmt::Display for DenoiseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotACheckpoint => write!(f, "not a denoiser checkpoint"),
            Self::Truncated { wanted, got } => {
                write!(f, "checkpoint is {got} bytes, needs {wanted}")
            }
            Self::Unsupported(what) => write!(f, "unsupported checkpoint: {what}"),
            Self::BadTileSize { width, height } => write!(
                f,
                "tile {width}x{height} must be non-empty and a multiple of {TILE_MULTIPLE}"
            ),
            Self::BadInputLength { wanted, got } => {
                write!(f, "input is {got} floats, needs {wanted}")
            }
        }
    }
}

impl std::error::Error for DenoiseError {}

/// What the eleventh input plane holds, when a network takes one.
///
/// The two are not interchangeable: an absolute standard error has the scale of
/// the scene's brightness, a relative one is divided by the pixel's own mean.
/// Weights trained against one and fed the other load without complaint and
/// denoise slightly wrong for good, so the checkpoint records which it was and
/// the renderer refuses the mismatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Guides {
    /// Plane 10 is the standard error as measured.
    AbsoluteError,
    /// Plane 10 is the standard error over the pixel's own mean.
    RelativeError,
}

/// Channel counts at each level of the U-Net.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Widths {
    pub level0: usize,
    pub level1: usize,
    pub level2: usize,
    pub level3: usize,
}

/// One convolution: `[out, in, 3, 3]` weights, flattened.
struct Conv {
    out_channels: usize,
    in_channels: usize,
    weights: Vec<f32>,
}

impl Conv {
    /// 3x3, stride 1, padding 1, no bias. Zero padding outside the tile, which
    /// is what the network was trained with — tiles overlap and are blended, so
    /// the edge a border pixel sees is discarded anyway.
    ///
    /// Ordered output-channel then tap then row: the innermost loop walks two
    /// contiguous rows with a fixed offset between them, and the row bounds are
    /// clamped once per tap rather than tested per pixel.
    fn apply(&self, src: &[f32], w: usize, h: usize, out: &mut Vec<f32>) {
        let plane = w * h;
        out.clear();
        out.resize(self.out_channels * plane, 0.0);
        // One output row at a time, held here across every input channel and
        // every tap.
        //
        // The obvious ordering — sweep the whole output plane once per input
        // channel per tap — touches `plane * 4` bytes on each of `9 * in`
        // passes. At 128x128 that is 64 KB a pass, which lives in L2 and is
        // re-fetched thousands of times for a wide layer. A row is 512 bytes
        // and stays in L1 for all of them, so the same arithmetic reads memory
        // once per row instead of once per pass.
        let mut acc = vec![0.0f32; w];
        for oc in 0..self.out_channels {
            let obase = oc * plane;
            for y in 0..h {
                acc.iter_mut().for_each(|v| *v = 0.0);
                for ic in 0..self.in_channels {
                    let ibase = ic * plane;
                    let wbase = (oc * self.in_channels + ic) * K * K;
                    for ky in 0..K {
                        // Padding 1 with a 3x3 kernel: tap ky reads row
                        // y + ky - 1, and rows outside the tile contribute
                        // nothing.
                        let sy = y as isize + ky as isize - 1;
                        if sy < 0 || sy >= h as isize {
                            continue;
                        }
                        let irow = ibase + sy as usize * w;
                        let k0 = self.weights[wbase + ky * K];
                        let k1 = self.weights[wbase + ky * K + 1];
                        let k2 = self.weights[wbase + ky * K + 2];
                        // The centre tap is a straight aligned run; the two
                        // shifted ones are the same run offset by a pixel, and
                        // each drops the column that would fall off the edge.
                        let row = &src[irow..irow + w];
                        for (a, v) in acc.iter_mut().zip(row) {
                            *a += k1 * *v;
                        }
                        for (a, v) in acc[1..].iter_mut().zip(&row[..w - 1]) {
                            *a += k0 * *v;
                        }
                        for (a, v) in acc[..w - 1].iter_mut().zip(&row[1..]) {
                            *a += k2 * *v;
                        }
                    }
                }
                out[obase + y * w..obase + y * w + w].copy_from_slice(&acc);
            }
        }
    }
}

/// The trained network: twelve convolutions and the widths they were sized for.
pub struct Denoiser {
    widths: Widths,
    inputs: usize,
    guides: Guides,
    convs: Vec<Conv>,
}

impl fmt::Debug for Denoiser {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Denoiser")
            .field("widths", &self.widths)
            .field("inputs", &self.inputs)
            .field("parameters", &self.parameter_count())
            .finish()
    }
}

impl Denoiser {
    /// Read a checkpoint written by `rlx-denoise`.
    ///
    /// Only the architecture this crate implements is accepted: a residual head
    /// and pooled resampling. A checkpoint for the kernel-predicting head or
    /// strided resampling is rejected by name rather than run as if it were
    /// this one, which would produce a plausible and wrong image.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DenoiseError> {
        let mut r = Reader::new(bytes);
        let magic = r.take(8)?;
        // 001 and 002 predate the header carrying its own shape and are not
        // worth supporting here: nothing trained with them is current.
        if magic == MAGIC_001 || magic == MAGIC_002 {
            return Err(DenoiseError::Unsupported(
                "checkpoint predates the self-describing header; retrain or convert".into(),
            ));
        }
        let v4 = magic == MAGIC;
        if !v4 && magic != MAGIC_003 {
            return Err(DenoiseError::NotACheckpoint);
        }
        let inputs = r.u32()? as usize;
        let widths = Widths {
            level0: r.u32()? as usize,
            level1: r.u32()? as usize,
            level2: r.u32()? as usize,
            level3: r.u32()? as usize,
        };
        let head = r.u32()?;
        let _radius = r.u32()?;
        let sampling = r.u32()?;
        // Guide encoding. Only `004` records it; everything older predates the
        // relative form, so a `003` file means absolute by construction.
        let guides = if v4 {
            match r.u32()? {
                0 => Guides::AbsoluteError,
                1 => Guides::RelativeError,
                other => {
                    return Err(DenoiseError::Unsupported(format!(
                        "guide encoding {other} is not one this build knows"
                    )))
                }
            }
        } else {
            Guides::AbsoluteError
        };
        if head != 0 {
            return Err(DenoiseError::Unsupported(
                "kernel-predicting head; this build runs the residual head only".into(),
            ));
        }
        if sampling != 1 {
            return Err(DenoiseError::Unsupported(
                "strided resampling; this build runs pooled resampling only".into(),
            ));
        }
        let count = r.u32()? as usize;
        let shapes = param_shapes(inputs, widths);
        if count != shapes.len() {
            return Err(DenoiseError::Unsupported(format!(
                "{count} tensors for a network that takes {}",
                shapes.len()
            )));
        }
        let mut convs = Vec::with_capacity(count);
        for [oc, ic, _, _] in shapes {
            // Each tensor carries its own length before its floats. Reading the
            // shape table alone and skipping this would shift every tensor after
            // the first by four bytes, which loads without complaint and
            // produces an image that is confidently wrong.
            let elems = oc * ic * K * K;
            let stored = r.u32()? as usize;
            if stored != elems {
                return Err(DenoiseError::Unsupported(format!(
                    "a {oc}x{ic}x{K}x{K} tensor holds {stored} values, not {elems}"
                )));
            }
            let weights = r.f32s(elems)?;
            convs.push(Conv {
                out_channels: oc,
                in_channels: ic,
                weights,
            });
        }
        Ok(Self {
            widths,
            inputs,
            guides,
            convs,
        })
    }

    /// Read a checkpoint from disk. Native only — a browser has no filesystem,
    /// so there [`Self::from_bytes`] is the way in.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, DenoiseError> {
        let bytes = std::fs::read(path).map_err(|e| DenoiseError::Unsupported(e.to_string()))?;
        Self::from_bytes(&bytes)
    }

    pub fn widths(&self) -> Widths {
        self.widths
    }

    /// Input planes this network was trained with.
    pub fn inputs(&self) -> usize {
        self.inputs
    }

    /// What plane 10 meant to whoever trained this, when there is one.
    pub fn guides(&self) -> Guides {
        self.guides
    }

    pub fn parameter_count(&self) -> usize {
        self.convs.iter().map(|c| c.weights.len()).sum()
    }

    /// Denoise one tile.
    ///
    /// `input` is `inputs() * width * height` floats, planar: every plane in
    /// full before the next starts. The result is `3 * width * height` in the
    /// same layout.
    pub fn denoise(
        &self,
        input: &[f32],
        width: usize,
        height: usize,
    ) -> Result<Vec<f32>, DenoiseError> {
        if width == 0 || height == 0 || !width.is_multiple_of(TILE_MULTIPLE) || !height.is_multiple_of(TILE_MULTIPLE) {
            return Err(DenoiseError::BadTileSize { width, height });
        }
        let wanted = self.inputs * width * height;
        if input.len() != wanted {
            return Err(DenoiseError::BadInputLength {
                wanted,
                got: input.len(),
            });
        }

        let mut scratch = Scratch::default();
        let enc0 = self.conv_relu(0, input, width, height, &mut scratch);
        let (p1, w1, h1) = pool2x(&enc0, self.widths.level0, width, height);
        let enc1 = self.conv_relu(1, &p1, w1, h1, &mut scratch);
        let (p2, w2, h2) = pool2x(&enc1, self.widths.level1, w1, h1);
        let enc2 = self.conv_relu(2, &p2, w2, h2, &mut scratch);
        let (p3, w3, h3) = pool2x(&enc2, self.widths.level2, w2, h2);
        let enc3 = self.conv_relu(3, &p3, w3, h3, &mut scratch);
        let mid = self.conv_relu(4, &enc3, w3, h3, &mut scratch);

        // Each `up` convolution also narrows: w3 -> w2, w2 -> w1, w1 -> w0. So
        // the upsample carries the *incoming* width and the concat sees two
        // stacks of the level's own width.
        let u3 = up2x(&mid, self.widths.level3, w3, h3);
        let u3 = self.conv_relu(5, &u3, w2, h2, &mut scratch);
        let c3 = concat(&u3, self.widths.level2, &enc2, self.widths.level2, w2 * h2);
        let d3 = self.conv_relu(6, &c3, w2, h2, &mut scratch);

        let u2 = up2x(&d3, self.widths.level2, w2, h2);
        let u2 = self.conv_relu(7, &u2, w1, h1, &mut scratch);
        let c2 = concat(&u2, self.widths.level1, &enc1, self.widths.level1, w1 * h1);
        let d2 = self.conv_relu(8, &c2, w1, h1, &mut scratch);

        let u1 = up2x(&d2, self.widths.level1, w1, h1);
        let u1 = self.conv_relu(9, &u1, width, height, &mut scratch);
        let c1 = concat(
            &u1,
            self.widths.level0,
            &enc0,
            self.widths.level0,
            width * height,
        );
        let d1 = self.conv_relu(10, &c1, width, height, &mut scratch);

        // The final convolution has no ReLU: it is a correction and may be
        // negative.
        let mut out = Vec::new();
        self.convs[11].apply(&d1, width, height, &mut out);
        let plane = width * height;
        for c in 0..OUT_CHANNELS {
            for i in 0..plane {
                out[c * plane + i] += input[c * plane + i];
            }
        }
        Ok(out)
    }

    /// Cut `input` down to one tile of a frame, ready for [`Self::denoise`].
    ///
    /// `input` is the whole frame, planar, `inputs()` planes of
    /// `frame_width * frame_height`. The result is the tile's own planes, padded
    /// on the right and bottom to a multiple of [`TILE_MULTIPLE`] by repeating
    /// the edge — zeros there would put a hard black border inside the network's
    /// receptive field, which it would try to reconstruct.
    ///
    /// Pure, and takes only what it needs, so a caller can hand each tile to a
    /// different thread or post it to a different worker.
    pub fn extract_tile(
        &self,
        input: &[f32],
        frame_width: usize,
        frame_height: usize,
        tile: &Tile,
    ) -> (Vec<f32>, usize, usize) {
        let pw = round_up(tile.width, TILE_MULTIPLE);
        let ph = round_up(tile.height, TILE_MULTIPLE);
        let frame_plane = frame_width * frame_height;
        let mut out = vec![0.0; self.inputs * pw * ph];
        for c in 0..self.inputs {
            let sbase = c * frame_plane;
            let dbase = c * pw * ph;
            for y in 0..ph {
                let sy = (tile.y + y.min(tile.height - 1)).min(frame_height - 1);
                for x in 0..pw {
                    let sx = (tile.x + x.min(tile.width - 1)).min(frame_width - 1);
                    out[dbase + y * pw + x] = input[sbase + sy * frame_width + sx];
                }
            }
        }
        (out, pw, ph)
    }

    /// Write a denoised tile's interior into a frame-sized buffer.
    ///
    /// Only the region the plan marked as `keep` is copied, so the overlap that
    /// existed to give the network context is discarded rather than blended —
    /// with a margin wider than the receptive field the interiors already agree,
    /// and a blend would only soften a seam that is not there.
    pub fn merge_tile(
        &self,
        denoised: &[f32],
        tile_width: usize,
        tile: &Tile,
        frame: &mut [f32],
        frame_width: usize,
        frame_height: usize,
    ) {
        let frame_plane = frame_width * frame_height;
        let tile_plane = tile_width * round_up(tile.height, TILE_MULTIPLE);
        for c in 0..OUT_CHANNELS {
            for y in 0..tile.keep_height {
                let fy = tile.keep_y + y;
                let ty = tile.keep_y - tile.y + y;
                for x in 0..tile.keep_width {
                    let fx = tile.keep_x + x;
                    let tx = tile.keep_x - tile.x + x;
                    frame[c * frame_plane + fy * frame_width + fx] =
                        denoised[c * tile_plane + ty * tile_width + tx];
                }
            }
        }
    }

    /// Denoise a whole frame, one tile at a time on this thread.
    ///
    /// The straightforward path: correct, and as slow as the sum of its tiles.
    /// For a viewport, walk [`TilePlan::tiles`] yourself and run them
    /// concurrently — [`Self::extract_tile`], [`Self::denoise`] and
    /// [`Self::merge_tile`] share no state, so N workers each doing a share of
    /// the list produce the same frame as this does.
    pub fn denoise_frame(
        &self,
        input: &[f32],
        frame_width: usize,
        frame_height: usize,
        plan: &TilePlan,
    ) -> Result<Vec<f32>, DenoiseError> {
        let wanted = self.inputs * frame_width * frame_height;
        if input.len() != wanted {
            return Err(DenoiseError::BadInputLength {
                wanted,
                got: input.len(),
            });
        }
        let mut frame = vec![0.0; OUT_CHANNELS * frame_width * frame_height];
        for tile in &plan.tiles {
            let (patch, pw, ph) = self.extract_tile(input, frame_width, frame_height, tile);
            let out = self.denoise(&patch, pw, ph)?;
            self.merge_tile(&out, pw, tile, &mut frame, frame_width, frame_height);
        }
        Ok(frame)
    }

    fn conv_relu(
        &self,
        index: usize,
        x: &[f32],
        w: usize,
        h: usize,
        scratch: &mut Scratch,
    ) -> Vec<f32> {
        let mut out = std::mem::take(&mut scratch.buffer);
        self.convs[index].apply(x, w, h, &mut out);
        for v in out.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        out
    }
}

#[derive(Default)]
struct Scratch {
    buffer: Vec<f32>,
}

/// One tile of a frame: where to read it, and which part of the result to keep.
///
/// Tiles overlap by a margin and only their interiors are kept. A convolution
/// stack pads with zeros at its edges, so the outermost pixels of any tile are
/// computed against a border that is not there — an overlap large enough to
/// cover the network's receptive field pushes that error outside the region
/// that gets used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tile {
    /// Left edge of the region to feed the network.
    pub x: usize,
    /// Top edge of the region to feed the network.
    pub y: usize,
    /// Width of the region to feed the network.
    pub width: usize,
    /// Height of the region to feed the network.
    pub height: usize,
    /// Left edge of the part of this tile to keep, in frame coordinates.
    pub keep_x: usize,
    /// Top edge of the part of this tile to keep, in frame coordinates.
    pub keep_y: usize,
    /// Width of the part to keep.
    pub keep_width: usize,
    /// Height of the part to keep.
    pub keep_height: usize,
}

/// How a frame is cut into tiles.
///
/// The pieces are independent: nothing is shared between them and none reads
/// another's output, so they can go to a thread pool, to a set of web workers,
/// or be run one at a time on whatever budget a frame has left. That
/// independence is the point — it is what turns a two-second frame into eight
/// quarter-second ones running at the same time.
#[derive(Debug, Clone)]
pub struct TilePlan {
    pub tiles: Vec<Tile>,
    pub frame_width: usize,
    pub frame_height: usize,
}

impl TilePlan {
    /// Cut a frame into tiles of about `target` pixels a side, overlapping by
    /// `margin`.
    ///
    /// `target` and `margin` are both rounded so every tile is a multiple of
    /// [`TILE_MULTIPLE`]; a frame smaller than one tile becomes a single tile
    /// padded up to that multiple by the caller.
    pub fn new(frame_width: usize, frame_height: usize, target: usize, margin: usize) -> Self {
        let step = round_down(target.max(TILE_MULTIPLE * 2), TILE_MULTIPLE);
        let margin = round_up(margin, TILE_MULTIPLE);
        let mut tiles = Vec::new();
        let mut y = 0;
        while y < frame_height {
            let keep_height = step.min(frame_height - y);
            let top = y.saturating_sub(margin);
            let bottom = (y + keep_height + margin).min(frame_height);
            let mut x = 0;
            while x < frame_width {
                let keep_width = step.min(frame_width - x);
                let left = x.saturating_sub(margin);
                let right = (x + keep_width + margin).min(frame_width);
                tiles.push(Tile {
                    x: left,
                    y: top,
                    width: right - left,
                    height: bottom - top,
                    keep_x: x,
                    keep_y: y,
                    keep_width,
                    keep_height,
                });
                x += step;
            }
            y += step;
        }
        Self {
            tiles,
            frame_width,
            frame_height,
        }
    }
}

fn round_up(v: usize, to: usize) -> usize {
    v.div_ceil(to) * to
}

fn round_down(v: usize, to: usize) -> usize {
    (v / to).max(1) * to
}

/// Shapes of the twelve convolutions, in the order the checkpoint stores them.
fn param_shapes(inputs: usize, w: Widths) -> Vec<[usize; 4]> {
    let Widths {
        level0: w0,
        level1: w1,
        level2: w2,
        level3: w3,
    } = w;
    // The `up` convolutions change the channel count on the way back — w3 to
    // w2, w2 to w1, w1 to w0 — so each `dec` sees the upsampled stack and the
    // encoder's at the *same* width, hence `w * 2` rather than a sum of two
    // different levels.
    vec![
        [w0, inputs, K, K],       // enc0
        [w1, w0, K, K],           // down1
        [w2, w1, K, K],           // down2
        [w3, w2, K, K],           // down3
        [w3, w3, K, K],           // bottleneck
        [w2, w3, K, K],           // up3
        [w2, w2 * 2, K, K],       // dec3
        [w1, w2, K, K],           // up2
        [w1, w1 * 2, K, K],       // dec2
        [w0, w1, K, K],           // up1
        [w0, w0 * 2, K, K],       // dec1
        [OUT_CHANNELS, w0, K, K], // out
    ]
}

/// Fixed 2x2 average pool. Halves both dimensions.
fn pool2x(src: &[f32], channels: usize, w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let (ow, oh) = (w / 2, h / 2);
    let mut out = vec![0.0; channels * ow * oh];
    for c in 0..channels {
        let sbase = c * w * h;
        let dbase = c * ow * oh;
        for y in 0..oh {
            let r0 = sbase + (y * 2) * w;
            let r1 = r0 + w;
            for x in 0..ow {
                let x0 = x * 2;
                out[dbase + y * ow + x] =
                    0.25 * (src[r0 + x0] + src[r0 + x0 + 1] + src[r1 + x0] + src[r1 + x0 + 1]);
            }
        }
    }
    (out, ow, oh)
}

/// Fixed nearest-neighbour 2x upsample. Every pixel becomes a 2x2 block.
fn up2x(src: &[f32], channels: usize, w: usize, h: usize) -> Vec<f32> {
    let (ow, oh) = (w * 2, h * 2);
    let mut out = vec![0.0; channels * ow * oh];
    for c in 0..channels {
        let sbase = c * w * h;
        let dbase = c * ow * oh;
        for y in 0..h {
            let srow = sbase + y * w;
            let d0 = dbase + (y * 2) * ow;
            let d1 = d0 + ow;
            for x in 0..w {
                let v = src[srow + x];
                let x0 = x * 2;
                out[d0 + x0] = v;
                out[d0 + x0 + 1] = v;
                out[d1 + x0] = v;
                out[d1 + x0 + 1] = v;
            }
        }
    }
    out
}

/// Concatenate two feature stacks along the channel axis.
fn concat(a: &[f32], a_channels: usize, b: &[f32], b_channels: usize, plane: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity((a_channels + b_channels) * plane);
    out.extend_from_slice(&a[..a_channels * plane]);
    out.extend_from_slice(&b[..b_channels * plane]);
    out
}

/// Little-endian cursor that reports what it wanted rather than panicking.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], DenoiseError> {
        let end = self.at + n;
        if end > self.bytes.len() {
            return Err(DenoiseError::Truncated {
                wanted: end,
                got: self.bytes.len(),
            });
        }
        let slice = &self.bytes[self.at..end];
        self.at = end;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32, DenoiseError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, DenoiseError> {
        let b = self.take(n * 4)?;
        Ok(b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TINY: Widths = Widths {
        level0: 4,
        level1: 6,
        level2: 8,
        level3: 10,
    };

    /// A checkpoint whose tensors are all `fill`, in the format `from_bytes`
    /// reads.
    fn checkpoint(widths: Widths, inputs: usize, fill: f32) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&(inputs as u32).to_le_bytes());
        for v in [widths.level0, widths.level1, widths.level2, widths.level3] {
            b.extend_from_slice(&(v as u32).to_le_bytes());
        }
        // head = residual, radius unused, sampling = pooled, guides = relative.
        for v in [0u32, 0, 1, 1] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        let shapes = param_shapes(inputs, widths);
        b.extend_from_slice(&(shapes.len() as u32).to_le_bytes());
        for [oc, ic, _, _] in shapes {
            let elems = oc * ic * K * K;
            b.extend_from_slice(&(elems as u32).to_le_bytes());
            for _ in 0..elems {
                b.extend_from_slice(&fill.to_le_bytes());
            }
        }
        b
    }

    /// The head is residual, so a network whose final convolution is zero must
    /// return its input colour untouched. Every other weight being zero makes
    /// the whole stack zero, which is the case this pins: an all-zero
    /// checkpoint is the identity, *not* black. If this fails the residual add
    /// is missing or reading the wrong planes.
    #[test]
    fn a_zero_network_returns_the_colour_it_was_given() {
        let net = Denoiser::from_bytes(&checkpoint(TINY, IN_CHANNELS, 0.0)).expect("read");
        let (w, h) = (16, 8);
        let mut input = vec![0.0f32; IN_CHANNELS * w * h];
        for (i, v) in input.iter_mut().enumerate() {
            *v = (i % 37) as f32 * 0.01;
        }
        let out = net.denoise(&input, w, h).expect("denoise");
        assert_eq!(out.len(), OUT_CHANNELS * w * h);
        for i in 0..OUT_CHANNELS * w * h {
            assert!(
                (out[i] - input[i]).abs() < 1e-6,
                "plane {} pixel {}: {} vs {}",
                i / (w * h),
                i % (w * h),
                out[i],
                input[i]
            );
        }
    }

    /// Pool then upsample is not the identity, but it is idempotent on an image
    /// that is already constant over 2x2 blocks — which is what makes the
    /// skip connections line up. A mistake in either direction (transposed
    /// indices, wrong stride) breaks this.
    #[test]
    fn pooling_and_upsampling_are_inverse_on_block_constant_input() {
        let (w, h) = (8, 4);
        let mut src = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                src[y * w + x] = ((y / 2) * (w / 2) + (x / 2)) as f32;
            }
        }
        let (pooled, pw, ph) = pool2x(&src, 1, w, h);
        assert_eq!((pw, ph), (4, 2));
        let back = up2x(&pooled, 1, pw, ph);
        assert_eq!(back, src);
    }

    /// An impulse in the middle of the input must reproduce the kernel,
    /// spatially reversed. That is the signature of cross-correlation, which is
    /// what every ML framework calls "conv2d" — a true convolution would
    /// reproduce the kernel unreversed. Getting this backwards is invisible on
    /// a symmetric kernel and wrecks a trained network on any other, so it is
    /// worth pinning explicitly rather than inferring from an all-ones test.
    #[test]
    // `1 * w + x` is row-and-column arithmetic written out: the row index
    // stays visible next to the column instead of being folded away.
    #[allow(clippy::identity_op)]
    fn an_impulse_reproduces_the_reversed_kernel() {
        let conv = Conv {
            out_channels: 1,
            in_channels: 1,
            weights: (1..=9).map(|v| v as f32).collect(),
        };
        let (w, h) = (3, 3);
        let mut src = vec![0.0f32; w * h];
        src[1 * w + 1] = 1.0;
        let mut out = Vec::new();
        conv.apply(&src, w, h, &mut out);
        assert_eq!(
            out,
            vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "cross-correlation reverses the kernel about the impulse"
        );
    }

    /// Zero padding, not wrapping or clamping: a kernel that sums its
    /// neighbourhood must produce a smaller total at a corner than in the
    /// middle, because the taps that fall outside contribute nothing.
    #[test]
    fn convolution_pads_with_zeros() {
        let conv = Conv {
            out_channels: 1,
            in_channels: 1,
            weights: vec![1.0; K * K],
        };
        let (w, h) = (8, 8);
        let src = vec![1.0f32; w * h];
        let mut out = Vec::new();
        conv.apply(&src, w, h, &mut out);
        assert_eq!(out[0], 4.0, "a corner sees 2x2 of the 3x3");
        assert_eq!(out[1], 6.0, "an edge sees 2x3");
        assert_eq!(out[w + 1], 9.0, "the interior sees all nine");
    }

    /// A checkpoint for an architecture this build does not implement has to be
    /// refused by name. Running it anyway would produce an image that looks
    /// like a denoise and is not one, which is worse than an error.
    #[test]
    fn an_architecture_this_build_cannot_run_is_refused() {
        let mut kernel_head = checkpoint(TINY, IN_CHANNELS, 0.0);
        kernel_head[8 + 4 * 5] = 1; // head = kernel-predicting
        match Denoiser::from_bytes(&kernel_head) {
            Err(DenoiseError::Unsupported(why)) => assert!(why.contains("kernel")),
            other => panic!("expected a refusal, got {other:?}"),
        }

        let mut strided = checkpoint(TINY, IN_CHANNELS, 0.0);
        strided[8 + 4 * 7] = 0; // sampling = strided
        match Denoiser::from_bytes(&strided) {
            Err(DenoiseError::Unsupported(why)) => assert!(why.contains("strided")),
            other => panic!("expected a refusal, got {other:?}"),
        }

        assert_eq!(
            Denoiser::from_bytes(b"not a checkpoint").unwrap_err(),
            DenoiseError::NotACheckpoint
        );
    }

    /// The encoder halves three times, so a tile that is not a multiple of
    /// eight would silently lose a row at some level and misalign the skips.
    #[test]
    fn a_tile_that_does_not_halve_three_times_is_refused() {
        let net = Denoiser::from_bytes(&checkpoint(TINY, IN_CHANNELS, 0.0)).expect("read");
        let input = vec![0.0f32; IN_CHANNELS * 12 * 8];
        assert_eq!(
            net.denoise(&input, 12, 8).unwrap_err(),
            DenoiseError::BadTileSize {
                width: 12,
                height: 8
            }
        );
        let short = vec![0.0f32; 3];
        assert!(matches!(
            net.denoise(&short, 8, 8).unwrap_err(),
            DenoiseError::BadInputLength { .. }
        ));
    }

    /// Tiling must cover every pixel exactly once and overlap only where the
    /// margin says. A gap leaves undenoised pixels; an uncovered edge leaves a
    /// band of raw noise down one side of the frame.
    #[test]
    fn a_tile_plan_covers_the_frame_exactly_once() {
        for (fw, fh) in [(256, 128), (300, 200), (64, 64), (129, 97)] {
            let plan = TilePlan::new(fw, fh, 64, 16);
            let mut hits = vec![0u32; fw * fh];
            for t in &plan.tiles {
                assert!(t.keep_x + t.keep_width <= fw);
                assert!(t.keep_y + t.keep_height <= fh);
                assert!(t.x <= t.keep_x && t.y <= t.keep_y);
                assert!(t.x + t.width >= t.keep_x + t.keep_width);
                for y in t.keep_y..t.keep_y + t.keep_height {
                    for x in t.keep_x..t.keep_x + t.keep_width {
                        hits[y * fw + x] += 1;
                    }
                }
            }
            assert!(
                hits.iter().all(|&n| n == 1),
                "{fw}x{fh}: every pixel is kept by exactly one tile"
            );
        }
    }

    /// A frame put through the tiled path must equal one put through in a
    /// single piece, wherever the network's receptive field fits inside the
    /// margin. Anything else means the tiling is visible in the output, which
    /// is the failure the overlap exists to prevent.
    #[test]
    fn tiling_a_frame_matches_denoising_it_whole() {
        let net = Denoiser::from_bytes(&checkpoint(TINY, IN_CHANNELS, 0.0)).expect("read");
        let (fw, fh) = (64, 32);
        let mut input = vec![0.0f32; IN_CHANNELS * fw * fh];
        for (i, v) in input.iter_mut().enumerate() {
            *v = ((i * 7) % 53) as f32 * 0.017;
        }
        let whole = net.denoise(&input, fw, fh).expect("whole");
        let plan = TilePlan::new(fw, fh, 32, 16);
        assert!(plan.tiles.len() > 1, "the frame should split");
        let tiled = net.denoise_frame(&input, fw, fh, &plan).expect("tiled");
        for (i, (a, b)) in whole.iter().zip(&tiled).enumerate() {
            assert!((a - b).abs() < 1e-5, "pixel {i}: whole {a} vs tiled {b}");
        }
    }

    /// A truncated file must say so rather than read past the end.
    #[test]
    fn a_truncated_checkpoint_reports_what_it_wanted() {
        let full = checkpoint(TINY, IN_CHANNELS, 0.0);
        let cut = &full[..full.len() - 400];
        assert!(matches!(
            Denoiser::from_bytes(cut).unwrap_err(),
            DenoiseError::Truncated { .. }
        ));
    }
}
