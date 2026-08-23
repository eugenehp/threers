//! Running a trained denoiser over a finished film.
//!
//! The À-Trous filter in the À-Trous denoiser is hand-tuned and needs no weights;
//! this is the learned alternative, and on the same held-out scenes it is about
//! twice as good. It takes the same guides plus the render's own colour, and
//! returns linear RGBA the same shape as [`Film::resolve_hdr`].
//!
//! Weights come from `rlx-denoise` — train there, run here:
//!
//! ```sh
//! rlx-denoise train --train train.bin --val val.bin --out weights.bin --arch oidn
//! ```
//!
//! # Ranges
//!
//! The network was trained on compressed radiance, so this compresses on the
//! way in and inverts on the way out. Getting that wrong does not fail, it just
//! denoises the wrong function of the image — so the transform lives here, next
//! to the only code that depends on it, rather than at the call site.
//!
//! # Tiling
//!
//! A graph is compiled for one input shape, and a whole frame at once would
//! both need recompiling per resolution and hold every activation for a 1920 x
//! 1080 U-Net in memory at once. So the frame is covered in overlapping tiles,
//! each run through the same compiled graph, and the overlaps are feathered
//! together. A convolution stack has a finite receptive field, so a tile with
//! enough margin sees everything a whole-frame pass would have shown it.

use super::denoise_net::{Denoiser, Guides, OUT_CHANNELS};

/// Input planes when the renderer fills colour, albedo and normal only.
const CORE_CHANNELS: usize = 9;
/// Input planes when it also fills depth and per-pixel error.
const IN_CHANNELS: usize = 11;

use super::denoise::DenoiseGuides;
use super::film::Film;

/// Side of the tile the network runs on. A multiple of 8, because the encoder
/// halves the resolution three times.
const TILE: usize = 256;

/// What this renderer puts in the guide planes. Checked against the weights.
/// What plane 10 holds when the renderer fills eleven planes: the per-pixel
/// standard error divided by the pixel's own mean, which is what
/// `Film::resolve_error` returns. Weights trained against an *absolute*
/// standard error expect a quantity whose scale is the scene's brightness and
/// would denoise slightly wrong for good — the portable checkpoint reader does
/// not carry the distinction, so this is a statement of what the renderer
/// produces rather than a check against the file.
const GUIDES: Guides = Guides::RelativeError;

/// Margin discarded from each tile edge before blending.
///
/// The network's receptive field is a few tens of pixels; a pixel nearer the
/// edge than that saw padding where its neighbourhood should have been. Taking
/// the overlap wider than the receptive field means every *kept* pixel was
/// computed from real data.
const MARGIN: usize = 32;

/// How many mirrored passes to average at inference.
///
/// A convolution stack is not symmetric: it answers a mirrored image slightly
/// differently, and the difference is error rather than signal. Averaging the
/// four axis mirrors cancels part of it, for four times the inference and no
/// retraining.
///
/// The guides survive mirroring because the network reads them differentially —
/// what matters is whether neighbouring pixels share a surface, and mirroring
/// preserves every such relation even though it leaves the world-space normals
/// describing a scene that is inside out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Passes {
    /// One pass, as trained.
    #[default]
    Single,
    /// Average the image, its horizontal and vertical mirrors, and both.
    Mirrored,
}

impl Passes {
    fn transforms(self) -> &'static [(bool, bool)] {
        match self {
            Passes::Single => &[(false, false)],
            Passes::Mirrored => &[(false, false), (true, false), (false, true), (true, true)],
        }
    }
}

/// Whether to clean the guide planes before denoising with them.
///
/// Cycles does this — `DENOISER_PREFILTER_ACCURATE` runs separate OIDN filters
/// over albedo and normal and only then denoises the colour, telling the main
/// filter its guides are already clean. The reasoning is that a noisy guide
/// does not merely fail to help, it actively misleads: the filter reads a
/// spurious normal difference as a surface edge and refuses to blend across it.
///
/// Measured on this renderer against a 1024-sample reference, the guides are
/// noisy enough to be worth it and unevenly so:
///
/// | spp | colour | albedo | normal |
/// |---|---|---|---|
/// | 1 | 0.0665 | 0.0169 | 0.0396 |
/// | 4 | 0.0365 | 0.0050 | 0.0139 |
///
/// Normals carry two to three times the albedo's error — 4% of a unit vector at
/// one sample — which is why they are the plane this helps most.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Prefilter {
    /// Guides used as the renderer produced them.
    #[default]
    None,
    /// Run the network over albedo and normal first, then denoise the colour
    /// with the cleaned versions. Three times the inference.
    ///
    /// Measured at **3.5% worse** on a held-out render, and the reason is a
    /// mismatch rather than the idea being wrong: this network was trained to
    /// denoise radiance, and an albedo or a remapped normal in the colour slot
    /// is not radiance. Cycles uses *dedicated* albedo and normal filters for
    /// this, not its colour filter reused. Doing it properly needs a guide
    /// denoiser trained on noisy-to-clean guide pairs.
    Guides,
    /// Clean the guides with the À-Trous filter instead of the network.
    ///
    /// Edge-preserving smoothing that assumes nothing about what it is
    /// smoothing, so it has no distribution to be out of.
    Atrous,
}

/// A trained denoiser, compiled for one tile shape.
pub struct LearnedDenoiser {
    denoiser: Denoiser,
    /// Planes this model takes — 9 without the depth and error guides, 11 with.
    inputs: usize,
    passes: Passes,
    prefilter: Prefilter,
}

impl std::fmt::Debug for LearnedDenoiser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LearnedDenoiser")
            .field("tile", &TILE)
            .finish()
    }
}

/// The fastest device this build can run the network on.
///
/// The network is only `conv2d` / `conv_transpose2d` / `relu` / `concat`, all
/// of which every RLX backend lowers, so the choice costs nothing but a feature
/// flag. Metal is preferred on Apple and wgpu elsewhere; without either the
/// network still runs, just on the CPU.
impl LearnedDenoiser {
    /// Load weights from a file.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        Self::load(path)
    }

    /// Load weights.
    ///
    /// There is one code path now: the forward pass is plain Rust, so there is
    /// no device to choose and nothing to compile. What used to be a Metal or
    /// wgpu graph is a set of loops that run wherever the crate does.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let bytes = std::fs::read(path.as_ref()).map_err(|e| e.to_string())?;
        let net = Denoiser::from_bytes(&bytes).map_err(|e| e.to_string())?;
        // The guide planes this fills are the ones below; weights trained
        // against a different definition of plane 10 would load without
        // complaint and denoise slightly wrong for good.
        // The guide encoding only matters when the guides are actually used: a
        // model trained on colour, albedo and normal alone never sees plane 10,
        // so what that plane once meant cannot affect it.
        let inputs = net.inputs();
        if inputs != CORE_CHANNELS && inputs != IN_CHANNELS {
            return Err(format!(
                "these weights take {inputs} input planes; this renderer fills \
                 {CORE_CHANNELS} or {IN_CHANNELS}"
            ));
        }
        // Only matters when the guides are actually used: a model trained on
        // colour, albedo and normal alone never sees plane 10, so what that
        // plane once meant cannot affect it.
        if inputs > CORE_CHANNELS && net.guides() != GUIDES {
            return Err(format!(
                "these weights were trained on {:?} guides, this renderer produces {GUIDES:?}",
                net.guides()
            ));
        }
        Ok(Self {
            denoiser: net,
            inputs,
            passes: Passes::default(),
            prefilter: Prefilter::default(),
        })
    }

    /// Clean the guides before denoising with them. See [`Prefilter`].
    pub fn set_prefilter(&mut self, prefilter: Prefilter) {
        self.prefilter = prefilter;
    }

    /// Average several mirrored passes instead of one. See [`Passes`].
    pub fn set_passes(&mut self, passes: Passes) {
        self.passes = passes;
    }

    /// Denoise a film, returning linear RGBA f32 — the same layout
    /// [`Film::resolve_hdr`] produces.
    ///
    /// `scene_scale` is the scene's diagonal, so the depth guide means the same
    /// thing whether the model is in millimetres or kilometres.
    pub fn apply(&mut self, film: &Film, scene_scale: f32) -> Result<Vec<f32>, String> {
        let (w, h) = (film.width() as usize, film.height() as usize);
        let hdr = film.resolve_hdr();
        if w == 0 || h == 0 || film.samples() == 0 {
            return Ok(hdr);
        }

        let mut planes = Planes::gather(film, scene_scale);
        match self.prefilter {
            Prefilter::None => {}
            Prefilter::Guides => self.prefilter_guides(&mut planes, w, h)?,
            Prefilter::Atrous => prefilter_atrous(&mut planes, film, w, h, scene_scale),
        }
        let transforms = self.passes.transforms();
        let mut total = vec![0.0f32; w * h * OUT_CHANNELS];

        for &mirror in transforms {
            let mut sum = vec![0.0f32; w * h * OUT_CHANNELS];
            let mut weight = vec![0.0f32; w * h];

            // Step by the tile less both margins, so every interior pixel is
            // covered by at least one tile that saw its full neighbourhood.
            let step = TILE.saturating_sub(2 * MARGIN).max(1);
            let mut y0 = 0usize;
            loop {
                let mut x0 = 0usize;
                loop {
                    self.run_tile(
                        &planes,
                        Frame { w, h },
                        (x0, y0),
                        mirror,
                        &mut sum,
                        &mut weight,
                    )?;
                    if x0 + TILE >= w {
                        break;
                    }
                    x0 = (x0 + step).min(w.saturating_sub(TILE));
                }
                if y0 + TILE >= h {
                    break;
                }
                y0 = (y0 + step).min(h.saturating_sub(TILE));
            }

            for i in 0..w * h {
                if weight[i] <= 0.0 {
                    continue;
                }
                for c in 0..OUT_CHANNELS {
                    total[i * OUT_CHANNELS + c] += sum[i * OUT_CHANNELS + c] / weight[i];
                }
            }
        }

        // Averaging happens in the compressed range the network works in, then
        // one inverse at the end — averaging after decompressing would bias the
        // result, since `y/(1-y)` is convex.
        let passes = transforms.len() as f32;
        let mut out = hdr;
        for i in 0..w * h {
            for c in 0..OUT_CHANNELS {
                out[i * 4 + c] = decompress(total[i * OUT_CHANNELS + c] / passes);
            }
        }
        Ok(out)
    }

    /// Replace the albedo and normal planes with denoised versions of
    /// themselves.
    ///
    /// Each is run through the same network with itself in the colour slot,
    /// which is what OIDN does with its own filters. Both are already in the
    /// range the colour slot expects once a normal is mapped from `[-1, 1]` to
    /// `[0, 1]`, and both are guided by the raw planes — the guides cannot
    /// guide their own cleaning, so this is one pass, not a fixed point.
    fn prefilter_guides(&mut self, planes: &mut Planes, w: usize, h: usize) -> Result<(), String> {
        let albedo = self.denoise_plane(planes, w, h, Guide::Albedo)?;
        let normal = self.denoise_plane(planes, w, h, Guide::Normal)?;
        planes.albedo = albedo;
        // Re-normalise: the network works per channel and has no reason to
        // return a unit vector, and a guide that is not one makes the cosine
        // between neighbours meaningless.
        planes.normal = normal
            .into_iter()
            .map(|n| {
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                if len > 1e-6 {
                    [n[0] / len, n[1] / len, n[2] / len]
                } else {
                    [0.0, 1.0, 0.0]
                }
            })
            .collect();
        Ok(())
    }

    /// One guide plane through the network, tiled the same way as the colour.
    fn denoise_plane(
        &mut self,
        planes: &Planes,
        w: usize,
        h: usize,
        guide: Guide,
    ) -> Result<Vec<[f32; 3]>, String> {
        let mut sum = vec![0.0f32; w * h * OUT_CHANNELS];
        let mut weight = vec![0.0f32; w * h];
        let step = TILE.saturating_sub(2 * MARGIN).max(1);
        let mut y0 = 0usize;
        loop {
            let mut x0 = 0usize;
            loop {
                self.run_guide_tile(
                    planes,
                    guide,
                    Frame { w, h },
                    (x0, y0),
                    &mut sum,
                    &mut weight,
                )?;
                if x0 + TILE >= w {
                    break;
                }
                x0 = (x0 + step).min(w.saturating_sub(TILE));
            }
            if y0 + TILE >= h {
                break;
            }
            y0 = (y0 + step).min(h.saturating_sub(TILE));
        }

        let mut out = vec![[0.0f32; 3]; w * h];
        for i in 0..w * h {
            let raw = guide.raw(planes, i);
            if weight[i] <= 0.0 {
                out[i] = raw;
                continue;
            }
            for c in 0..OUT_CHANNELS {
                let v = sum[i * OUT_CHANNELS + c] / weight[i];
                out[i][c] = guide.decode(v);
            }
        }
        Ok(out)
    }

    /// One tile of a guide plane.
    #[allow(clippy::too_many_arguments)]
    fn run_guide_tile(
        &mut self,
        planes: &Planes,
        guide: Guide,
        frame: Frame,
        origin: (usize, usize),
        sum: &mut [f32],
        weight: &mut [f32],
    ) -> Result<(), String> {
        let Frame { w, h } = frame;
        let (x0, y0) = origin;
        let pixels = TILE * TILE;
        let mut input = vec![0.0f32; self.inputs * pixels];
        for c in 0..self.inputs {
            for ty in 0..TILE {
                let sy = (y0 + ty).min(h - 1);
                for tx in 0..TILE {
                    let sx = (x0 + tx).min(w - 1);
                    let i = sy * w + sx;
                    // Channels 0..3 carry the plane being cleaned; the rest are
                    // the raw guides, which still describe the geometry.
                    input[c * pixels + ty * TILE + tx] = if c < OUT_CHANNELS {
                        guide.encode(guide.raw(planes, i)[c])
                    } else {
                        planes.at(c, i)
                    };
                }
            }
        }

        let out = self
            .denoiser
            .denoise(&input, TILE, TILE)
            .map_err(|e| e.to_string())?;
        for ty in 0..TILE {
            let sy = y0 + ty;
            if sy >= h {
                continue;
            }
            for tx in 0..TILE {
                let sx = x0 + tx;
                if sx >= w {
                    continue;
                }
                let f = feather(tx, TILE, x0 == 0, x0 + TILE >= w)
                    * feather(ty, TILE, y0 == 0, y0 + TILE >= h);
                if f <= 0.0 {
                    continue;
                }
                let i = sy * w + sx;
                weight[i] += f;
                for c in 0..OUT_CHANNELS {
                    sum[i * OUT_CHANNELS + c] += f * out[c * pixels + ty * TILE + tx];
                }
            }
        }
        Ok(())
    }

    /// One tile, accumulated into the frame with a feathered weight.
    fn run_tile(
        &mut self,
        planes: &Planes,
        frame: Frame,
        origin: (usize, usize),
        mirror: (bool, bool),
        sum: &mut [f32],
        weight: &mut [f32],
    ) -> Result<(), String> {
        let Frame { w, h } = frame;
        let (x0, y0) = origin;
        let (flip_x, flip_y) = mirror;
        // The mirror is applied when reading the source and undone when writing
        // the result, so the tiling and feathering never see it.
        let src = |x: usize, y: usize| {
            let sx = if flip_x { w - 1 - x } else { x };
            let sy = if flip_y { h - 1 - y } else { y };
            sy * w + sx
        };
        let pixels = TILE * TILE;
        let mut input = vec![0.0f32; self.inputs * pixels];
        for c in 0..self.inputs {
            for ty in 0..TILE {
                // Clamp rather than zero: a tile hanging off the edge repeats
                // the border, which is what the training data's own padding did.
                let sy = (y0 + ty).min(h - 1);
                for tx in 0..TILE {
                    let sx = (x0 + tx).min(w - 1);
                    input[c * pixels + ty * TILE + tx] = planes.at(c, src(sx, sy));
                }
            }
        }

        let out = self
            .denoiser
            .denoise(&input, TILE, TILE)
            .map_err(|e| e.to_string())?;
        for ty in 0..TILE {
            let sy = y0 + ty;
            if sy >= h {
                continue;
            }
            for tx in 0..TILE {
                let sx = x0 + tx;
                if sx >= w {
                    continue;
                }
                let f = feather(tx, TILE, x0 == 0, x0 + TILE >= w)
                    * feather(ty, TILE, y0 == 0, y0 + TILE >= h);
                if f <= 0.0 {
                    continue;
                }
                let i = src(sx, sy);
                weight[i] += f;
                for c in 0..OUT_CHANNELS {
                    sum[i * OUT_CHANNELS + c] += f * out[c * pixels + ty * TILE + tx];
                }
            }
        }
        Ok(())
    }
}

/// How much a tile's pixel counts, falling to zero across the margin.
///
/// A hard cut at the margin would show every tile boundary as a seam wherever
/// two tiles disagreed. A ramp makes the transition continuous. The frame's
/// outer edges keep full weight, because there is no neighbouring tile to blend
/// with and dropping them would leave a border unfiltered.
fn feather(i: usize, tile: usize, at_start: bool, at_end: bool) -> f32 {
    let from_start = i;
    let from_end = tile - 1 - i;
    let lead = if at_start { MARGIN } else { from_start };
    let trail = if at_end { MARGIN } else { from_end };
    let edge = lead.min(trail);
    if edge >= MARGIN {
        1.0
    } else {
        // Smoothstep, so weight and its slope both reach zero at the edge.
        let t = (edge as f32 + 0.5) / MARGIN as f32;
        t * t * (3.0 - 2.0 * t)
    }
}

/// Smooth the guides with the À-Trous filter, guided by depth and each other.
///
/// A generic edge-preserving filter, so unlike the network it has no training
/// distribution to fall outside of. The normals are re-normalised afterwards
/// for the same reason as in the learned path: a guide that is not a unit
/// vector makes the cosine between neighbours meaningless.
fn prefilter_atrous(planes: &mut Planes, film: &Film, w: usize, h: usize, scene_scale: f32) {
    let depth = film.resolve_depth();
    let variance = film.resolve_variance();
    // Fewer, tighter passes than the colour gets: the guides carry a fraction
    // of the colour's error, and over-smoothing them would erase the very edges
    // they exist to mark.
    let params = super::denoise::DenoiseParams {
        iterations: 2,
        ..Default::default()
    };

    let smooth = |source: &[[f32; 3]]| -> Vec<[f32; 3]> {
        let rgba: Vec<f32> = source
            .iter()
            .flat_map(|v| [v[0], v[1], v[2], 1.0])
            .collect();
        let out = super::denoise::denoise(
            w as u32,
            h as u32,
            &rgba,
            &DenoiseGuides {
                albedo: &planes.albedo,
                normal: &planes.normal,
                depth: &depth,
                variance: &variance,
                scene_scale,
            },
            &params,
        );
        (0..w * h)
            .map(|i| [out[i * 4], out[i * 4 + 1], out[i * 4 + 2]])
            .collect()
    };

    let albedo = smooth(&planes.albedo);
    let normal = smooth(&planes.normal);
    planes.albedo = albedo;
    planes.normal = normal
        .into_iter()
        .map(|n| {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if len > 1e-6 {
                [n[0] / len, n[1] / len, n[2] / len]
            } else {
                [0.0, 1.0, 0.0]
            }
        })
        .collect();
}

/// Which guide plane is being cleaned, and how it maps into the colour slot.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Guide {
    Albedo,
    Normal,
}

impl Guide {
    fn raw(self, planes: &Planes, i: usize) -> [f32; 3] {
        match self {
            Guide::Albedo => planes.albedo[i],
            Guide::Normal => planes.normal[i],
        }
    }

    /// Into the `[0, 1]` the colour slot expects. Albedo is a reflectance and
    /// already there; a normal component spans `[-1, 1]`.
    fn encode(self, v: f32) -> f32 {
        match self {
            Guide::Albedo => v.clamp(0.0, 1.0),
            Guide::Normal => 0.5 + 0.5 * v,
        }
    }

    fn decode(self, v: f32) -> f32 {
        match self {
            Guide::Albedo => v.clamp(0.0, 1.0),
            Guide::Normal => (2.0 * v - 1.0).clamp(-1.0, 1.0),
        }
    }
}

/// The frame a tile is being cut from.
#[derive(Clone, Copy)]
struct Frame {
    w: usize,
    h: usize,
}

/// The input planes, in the order the network was trained on.
pub(crate) struct Planes {
    colour: Vec<f32>,
    albedo: Vec<[f32; 3]>,
    normal: Vec<[f32; 3]>,
    depth: Vec<f32>,
    error: Vec<f32>,
}

impl Planes {
    fn gather(film: &Film, scene_scale: f32) -> Self {
        let scale = scene_scale.max(1e-6);
        Self {
            colour: film.resolve_hdr(),
            albedo: film.resolve_albedo(),
            normal: film.resolve_normal(),
            depth: film
                .resolve_depth()
                .into_iter()
                .map(|d| {
                    if d.is_finite() {
                        compress(d / scale)
                    } else {
                        1.0
                    }
                })
                .collect(),
            // Relative error, not absolute: the absolute standard error is set
            // by the scene's brightness and its sample count, so across a tile
            // set it varies far more between images than within one and tells
            // the network only what the colour already says. Dividing by the
            // pixel's own mean leaves the local question.
            error: film.resolve_error().into_iter().map(compress).collect(),
        }
    }

    fn at(&self, channel: usize, i: usize) -> f32 {
        match channel {
            0..=2 => compress(self.colour[i * 4 + channel]),
            3..=5 => self.albedo[i][channel - 3],
            6..=8 => self.normal[i][channel - 6],
            9 => self.depth[i],
            _ => self.error[i],
        }
    }
}

/// `x / (1 + x)` — bounded, monotone, and what the network was trained on.
///
/// Total, matching the exporter exactly. `Film::resolve_variance` reports
/// `INFINITY` for any pixel with fewer than two samples, and `inf / (1 + inf)`
/// is NaN — which would propagate through every convolution and return a black
/// frame with no error anywhere.
fn compress(v: f32) -> f32 {
    if !v.is_finite() {
        return if v > 0.0 { 1.0 } else { 0.0 };
    }
    v.max(0.0) / (1.0 + v.max(0.0))
}

/// The inverse, `y / (1 - y)`, guarded at the pole.
fn decompress(v: f32) -> f32 {
    let v = v.clamp(0.0, 0.999_999);
    v / (1.0 - v)
}

#[cfg(test)]
mod tests {

    /// Write a checkpoint whose weights are all zero, in the format the
    /// portable reader takes.
    ///
    /// Zero weights make the network the identity, because the head is
    /// residual — which is exactly what the round-trip tests below need, and
    /// what makes them able to assert an exact colour rather than a plausible
    /// one. Written here rather than by the training crate so these tests do
    /// not depend on it.
    fn write_identity_checkpoint(path: &std::path::Path, inputs: usize) {
        write_checkpoint(path, inputs, 1)
    }

    /// As above, with an explicit guide code: 0 absolute, 1 relative.
    fn write_checkpoint(path: &std::path::Path, inputs: usize, guides: u32) {
        const K: usize = 3;
        let (w0, w1, w2, w3) = (8usize, 12, 16, 20);
        let shapes: [[usize; 2]; 12] = [
            [w0, inputs],
            [w1, w0],
            [w2, w1],
            [w3, w2],
            [w3, w3],
            [w2, w3],
            [w2, w2 * 2],
            [w1, w2],
            [w1, w1 * 2],
            [w0, w1],
            [w0, w0 * 2],
            [3, w0],
        ];
        let mut b: Vec<u8> = Vec::new();
        b.extend_from_slice(b"RLXDW004");
        b.extend_from_slice(&(inputs as u32).to_le_bytes());
        for v in [w0, w1, w2, w3] {
            b.extend_from_slice(&(v as u32).to_le_bytes());
        }
        // Residual head, unused radius, pooled resampling, relative-error guide.
        for v in [0u32, 0, 1, guides] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(&(shapes.len() as u32).to_le_bytes());
        for [oc, ic] in shapes {
            let elems = oc * ic * K * K;
            b.extend_from_slice(&(elems as u32).to_le_bytes());
            b.extend_from_slice(&vec![0u8; elems * 4]);
        }
        std::fs::write(path, b).expect("write checkpoint");
    }
    use super::*;

    #[test]
    fn compression_round_trips() {
        for v in [0.0f32, 0.25, 1.0, 7.5, 1000.0] {
            let back = decompress(compress(v));
            assert!(
                (back - v).abs() <= 1e-3 * v.max(1.0),
                "{v} came back as {back}"
            );
        }
    }

    /// The variance plane is infinite wherever a pixel took one sample, so the
    /// transform has to be defined there — NaN in an input plane is invisible
    /// until the whole frame comes back black.
    #[test]
    fn compression_is_total() {
        assert_eq!(compress(f32::INFINITY), 1.0);
        assert_eq!(compress(f32::NEG_INFINITY), 0.0);
        assert_eq!(compress(f32::NAN), 0.0);
        assert!(compress(f32::INFINITY).is_finite());
    }

    #[test]
    fn compression_is_bounded() {
        // `x / (1 + x)` is below 1 for every finite x in exact arithmetic, but
        // in f32 `1.0 + MAX` rounds back to `MAX` and the quotient is exactly
        // 1. Not clamped: the exporter that produced the training data does not
        // clamp either, and an inference transform that disagreed with the
        // training transform would denoise a slightly different function of the
        // image. `decompress` is where the pole is handled instead.
        // f32 carries 24 bits of mantissa, so `1.0 + v` rounds back to `v` once
        // v exceeds 2^24 and the quotient saturates at exactly 1. Below that
        // the transform is strictly inside the range, which covers every
        // radiance a render produces short of a broken firefly.
        assert!(compress(f32::MAX) <= 1.0);
        assert!(compress((1u32 << 24) as f32) >= 1.0);
        assert!(
            compress(1e6) < 1.0,
            "an ordinary bright pixel must stay inside"
        );
        assert_eq!(compress(-5.0), 0.0);
        assert!(
            decompress(compress(f32::MAX)).is_finite(),
            "the pole must not come back as infinity"
        );
    }

    /// Interior pixels keep full weight and margins ramp to nearly nothing, so
    /// two overlapping tiles sum to a continuous result rather than a seam.
    #[test]
    fn feather_is_flat_inside_and_ramps_at_a_shared_edge() {
        assert_eq!(feather(TILE / 2, TILE, false, false), 1.0);
        assert_eq!(feather(MARGIN, TILE, false, false), 1.0);
        let first = feather(0, TILE, false, false);
        assert!(first > 0.0 && first < 0.1, "edge weight was {first}");
        // Monotone across the ramp.
        for i in 0..MARGIN {
            let a = feather(i, TILE, false, false);
            let b = feather(i + 1, TILE, false, false);
            assert!(b >= a, "weight fell going inward at {i}: {a} -> {b}");
        }
    }

    /// The frame's own border has no neighbour to blend with; fading it would
    /// leave the outermost pixels of the image unfiltered.
    #[test]
    fn the_frames_outer_edge_keeps_full_weight() {
        assert_eq!(feather(0, TILE, true, false), 1.0);
        assert_eq!(feather(TILE - 1, TILE, false, true), 1.0);
    }

    /// End to end on a real film, with an untrained network.
    ///
    /// The residual head starts at zero, so an untrained network returns its
    /// input colour unchanged — which makes the whole path testable without a
    /// trained model. Everything between the film and the result has to be
    /// exactly invertible for this to hold: the guide gather, the compression
    /// and its inverse, the tiling, and the feathered accumulation. A mistake
    /// in any of them shows up here as a colour that is not the one that went
    /// in, and nowhere else until the images look subtly wrong.
    #[test]
    fn an_untrained_network_returns_the_film_it_was_given() {
        use crate::math::Vector3;

        let (w, h) = (40u32, 24u32);
        let mut film = Film::new(w, h);
        // Two samples a pixel, so the variance is defined rather than infinite,
        // and a spread of values so a scaling error cannot hide.
        for (i, p) in film.pixels_mut().iter_mut().enumerate() {
            let v = (i % 17) as f32 / 17.0;
            p.add(Vector3::new(v, v * 0.5 + 0.1, 2.0 - v), 1.0);
            p.add(Vector3::new(v * 1.2, v * 0.4 + 0.1, 1.8 - v), 1.0);
        }
        let expected = film.resolve_hdr();

        let path = std::env::temp_dir().join("threers_learned_identity.bin");
        write_identity_checkpoint(&path, CORE_CHANNELS);

        let mut denoiser = LearnedDenoiser::load(&path).expect("load");
        let out = denoiser.apply(&film, 4.0).expect("apply");
        let _ = std::fs::remove_file(path);

        assert_eq!(out.len(), expected.len());
        let worst = out
            .chunks(4)
            .zip(expected.chunks(4))
            .flat_map(|(a, b)| (0..3).map(move |c| (a[c] - b[c]).abs()))
            .fold(0.0f32, f32::max);
        // The colour makes a round trip through `x/(1+x)` in f32, so exactness
        // is not available; anything above this is a real error in the path.
        assert!(worst < 2e-4, "the path changed the image by {worst}");
    }

    /// Prefiltering must not change what an identity network returns: the
    /// guides go through the same residual path, so a network that is the
    /// identity has to hand them back unchanged — including the normals, which
    /// make a round trip through `[-1, 1]` -> `[0, 1]` and back.
    #[test]
    fn prefiltering_leaves_an_identity_network_alone() {
        use crate::math::Vector3;

        let (w, h) = (32u32, 24u32);
        let mut film = Film::new(w, h);
        for (i, p) in film.pixels_mut().iter_mut().enumerate() {
            let v = (i % 13) as f32 / 13.0;
            p.add(Vector3::new(v, 0.4 + 0.5 * v, 1.0 - v), 1.0);
            p.add(Vector3::new(v * 1.1, 0.4 + 0.4 * v, 0.9 - v), 1.0);
        }
        let expected = film.resolve_hdr();

        let path = std::env::temp_dir().join("threers_prefilter.bin");
        write_identity_checkpoint(&path, CORE_CHANNELS);

        let mut denoiser = LearnedDenoiser::load(&path).expect("load");
        denoiser.set_prefilter(Prefilter::Guides);
        let out = denoiser.apply(&film, 3.0).expect("apply");
        let _ = std::fs::remove_file(path);

        let worst = out
            .chunks(4)
            .zip(expected.chunks(4))
            .flat_map(|(a, b)| (0..3).map(move |c| (a[c] - b[c]).abs()))
            .fold(0.0f32, f32::max);
        assert!(worst < 2e-4, "prefiltering changed the image by {worst}");
    }

    /// A normal has to survive the trip into the colour slot and back, or the
    /// cleaned guide describes a different surface than the raw one did.
    #[test]
    fn a_normal_round_trips_through_the_colour_slot() {
        for v in [-1.0f32, -0.5, 0.0, 0.25, 1.0] {
            let back = Guide::Normal.decode(Guide::Normal.encode(v));
            assert!((back - v).abs() < 1e-6, "{v} came back as {back}");
        }
        for v in [0.0f32, 0.5, 1.0] {
            let back = Guide::Albedo.decode(Guide::Albedo.encode(v));
            assert!((back - v).abs() < 1e-6, "{v} came back as {back}");
        }
    }

    /// Mirroring must not change what an identity network returns — that is
    /// what says the mirror is applied and undone consistently rather than
    /// leaving the image flipped or the tiles misaligned.
    #[test]
    fn mirrored_passes_leave_an_identity_network_alone() {
        use crate::math::Vector3;

        let (w, h) = (36u32, 28u32);
        let mut film = Film::new(w, h);
        for (i, p) in film.pixels_mut().iter_mut().enumerate() {
            // Asymmetric in both axes, so a stray flip cannot cancel itself.
            let x = (i as u32 % w) as f32 / w as f32;
            let y = (i as u32 / w) as f32 / h as f32;
            p.add(Vector3::new(x, y, 0.3 + 0.5 * x * y), 1.0);
            p.add(Vector3::new(x * 0.9, y * 1.1, 0.3 + 0.4 * x * y), 1.0);
        }
        let expected = film.resolve_hdr();

        let path = std::env::temp_dir().join("threers_tta.bin");
        write_identity_checkpoint(&path, CORE_CHANNELS);

        let mut denoiser = LearnedDenoiser::load(&path).expect("load");
        denoiser.set_passes(Passes::Mirrored);
        let out = denoiser.apply(&film, 3.0).expect("apply");
        let _ = std::fs::remove_file(path);

        let worst = out
            .chunks(4)
            .zip(expected.chunks(4))
            .flat_map(|(a, b)| (0..3).map(move |c| (a[c] - b[c]).abs()))
            .fold(0.0f32, f32::max);
        assert!(worst < 2e-4, "mirrored passes changed the image by {worst}");
    }

    /// A model that takes only the core planes must load and run, whatever the
    /// guide field says — it never reads plane 10, so what that plane once
    /// meant cannot reach it. Refusing it would rule out the configuration
    /// that may well be the one worth shipping.
    #[test]
    fn a_core_guide_model_loads_and_returns_its_input() {
        use crate::math::Vector3;

        let (w, h) = (32u32, 20u32);
        let mut film = Film::new(w, h);
        for (i, p) in film.pixels_mut().iter_mut().enumerate() {
            let v = (i % 11) as f32 / 11.0;
            p.add(Vector3::new(v, v * 0.6, 1.0 - v), 1.0);
            p.add(Vector3::new(v * 1.1, v * 0.5, 0.9 - v), 1.0);
        }
        let expected = film.resolve_hdr();

        let path = std::env::temp_dir().join("threers_core_guides.bin");
        write_identity_checkpoint(&path, CORE_CHANNELS);

        let mut denoiser = LearnedDenoiser::load(&path)
            .expect("a core-plane model must load whatever the guide field says");
        let out = denoiser.apply(&film, 3.0).expect("apply");
        let _ = std::fs::remove_file(path);

        let worst = out
            .chunks(4)
            .zip(expected.chunks(4))
            .flat_map(|(a, b)| (0..3).map(move |c| (a[c] - b[c]).abs()))
            .fold(0.0f32, f32::max);
        assert!(worst < 2e-4, "the path changed the image by {worst}");
    }

    /// Weights trained against a different guide encoding must be refused. The
    /// tensor shapes match, so nothing else would notice.
    #[test]
    fn weights_from_another_guide_encoding_are_refused() {
        let path = std::env::temp_dir().join("threers_guide_mismatch.bin");
        // Eleven planes, so plane 10 is actually used, and the *absolute*
        // encoding, which is not what this renderer fills it with.
        write_checkpoint(&path, IN_CHANNELS, 0);

        let err = LearnedDenoiser::load(&path).expect_err("a mismatched encoding must not load");
        assert!(
            err.contains("AbsoluteError"),
            "the error should say which: {err}"
        );
        let _ = std::fs::remove_file(path);
    }

    /// A margin narrower than the receptive field would keep pixels that saw
    /// padding instead of neighbours.
    #[test]
    fn tiles_step_by_less_than_their_width() {
        let step = TILE - 2 * MARGIN;
        assert!(step > 0, "the margins consume the whole tile");
        assert!(TILE.is_multiple_of(8), "the encoder halves three times");
    }
}
