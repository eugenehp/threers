//! Palette extraction: k-means over a frame's pixels, with rlx doing the
//! expensive half.
//!
//! Lloyd's algorithm alternates two steps. **Assignment** — which of the `k`
//! colours is each of the quarter-million pixels nearest to? — is a distance
//! matrix and an `argmin`, and it is what rlx's `vector_quantize` op is: one
//! matrix multiply and a reduction, on whatever device the session holds.
//! **Update** — recompute each colour as the mean of the pixels that chose it
//! — is a scatter-add over a handful of accumulators, and the host does it.
//!
//! Splitting it that way is not a compromise; it is where each step belongs.
//!
//! ```no_run
//! use threers::rlx::{preferred_device, Palette, PaletteOptions};
//!
//! # let frame = vec![0u8; 4];
//! let palette = Palette::extract(&frame, 8, &PaletteOptions::default(), preferred_device())
//!     .unwrap();
//! let posterized = palette.posterize(&frame, preferred_device()).unwrap();
//! ```
//!
//! # Which space to cluster in
//!
//! sRGB, by default. k-means minimises *squared distance*, and in linear light
//! the distance between two bright colours dwarfs the distance between two
//! dark ones — so a linear palette spends its entries on highlights and
//! renders the shadows as one flat black. Encoded values are roughly
//! perceptually spaced, which is the spacing a palette wants. Set
//! [`PaletteOptions::color_space`] to `Linear` when the palette is going to be
//! used for arithmetic rather than for looking at.

use ::rlx::ir::ops::vq::VqMetric;
use ::rlx::{DType, Device, Graph, Shape};

use crate::math::Color;

use super::session::GraphRunner;
use super::tensor::{ColorSpace, Tensor, TensorError};

/// A short list of colours, in linear light like every other [`Color`] in the
/// crate.
#[derive(Debug, Clone, PartialEq)]
pub struct Palette {
    pub colors: Vec<Color>,
    /// The space the clustering ran in, which is the space
    /// [`Palette::nearest`] measures distance in.
    space: ColorSpace,
}

/// How to run the clustering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PaletteOptions {
    /// Lloyd iterations. Convergence is fast — the first five do most of it.
    pub iterations: usize,
    /// How many pixels to cluster over. The palette of a 4K frame is the
    /// palette of twenty thousand of its pixels, and the assignment step is
    /// linear in this number.
    pub samples: usize,
    /// Which space to measure colour distance in. See the module note.
    pub color_space: ColorSpace,
}

impl Default for PaletteOptions {
    fn default() -> Self {
        Self {
            iterations: 12,
            samples: 20_000,
            color_space: ColorSpace::Srgb,
        }
    }
}

impl Default for Palette {
    /// The empty palette. Its space is sRGB only because a palette with no
    /// colours has no distances to measure, and something has to be written.
    fn default() -> Self {
        Self {
            colors: Vec::new(),
            space: ColorSpace::Srgb,
        }
    }
}

impl Palette {
    /// A palette built by hand rather than found. Treated as sRGB-spaced for
    /// nearest-colour queries, which is how a hand-picked palette is spaced.
    pub fn new(colors: Vec<Color>) -> Self {
        Self {
            colors,
            space: ColorSpace::Srgb,
        }
    }

    pub fn len(&self) -> usize {
        self.colors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.colors.is_empty()
    }

    /// Cluster `rgba` into `count` colours.
    ///
    /// Fewer distinct colours than `count` gives a shorter palette rather than
    /// duplicates: an entry no pixel chose is dropped, because a palette with
    /// two identical swatches is a palette that lied about its size.
    pub fn extract(
        rgba: &[u8],
        count: usize,
        options: &PaletteOptions,
        device: Device,
    ) -> Result<Self, TensorError> {
        let count = count.max(1);
        let samples = sample_pixels(rgba, options.samples, options.color_space);
        let n = samples.dims()[0];
        if n == 0 {
            return Ok(Self {
                colors: Vec::new(),
                space: options.color_space,
            });
        }

        let mut centroids = seed_centroids(&samples, count);
        let mut runner = assignment_runner(n, centroids.len() / 3, device);

        for _ in 0..options.iterations {
            runner.set_param("codebook", &centroids);
            let assignment = runner.run(&[("pixels", &samples)]).remove(0).into_data();
            let (next, moved) = update(&samples, &assignment, &centroids);
            centroids = next;
            if !moved {
                break;
            }
        }

        // One more assignment, against the centroids that will actually be
        // returned. The loop's last assignment described the centroids *before*
        // its last update, so using it here would report which of the previous
        // colours were used — and with `iterations: 0` there would be no
        // assignment at all, and every seeded colour would look unused.
        runner.set_param("codebook", &centroids);
        let assignment = runner.run(&[("pixels", &samples)]).remove(0).into_data();

        // Drop entries nothing chose. An unclaimed centroid is wherever it was
        // seeded, which is a colour the image does not contain.
        let mut used: Vec<bool> = vec![false; centroids.len() / 3];
        for a in &assignment {
            let i = *a as usize;
            if i < used.len() {
                used[i] = true;
            }
        }
        let colors = centroids
            .chunks_exact(3)
            .zip(used)
            .filter(|(_, used)| *used)
            .map(|(c, _)| Self::to_linear([c[0], c[1], c[2]], options.color_space))
            .collect();

        Ok(Self {
            colors,
            space: options.color_space,
        })
    }

    /// Which palette entry each pixel belongs to, row-major.
    ///
    /// An empty frame gets an empty answer rather than an empty *graph*: a
    /// zero-row input is not a shape the backends accept.
    pub fn assign(&self, rgba: &[u8], device: Device) -> Result<Vec<u32>, TensorError> {
        if self.colors.is_empty() || rgba.len() < 4 {
            return Ok(vec![0; rgba.len() / 4]);
        }
        let pixels = all_pixels(rgba, self.space);
        let n = pixels.dims()[0];
        let mut runner = assignment_runner(n, self.colors.len(), device);
        runner.set_param("codebook", &self.codebook());
        let assignment = runner.run(&[("pixels", &pixels)]).remove(0);
        Ok(assignment.data().iter().map(|a| *a as u32).collect())
    }

    /// The frame with every pixel replaced by its nearest palette entry.
    /// Alpha is untouched.
    pub fn posterize(&self, rgba: &[u8], device: Device) -> Result<Vec<u8>, TensorError> {
        let assignment = self.assign(rgba, device)?;
        let mut out = Vec::with_capacity(rgba.len());
        for (px, index) in rgba.chunks_exact(4).zip(assignment) {
            let c = self
                .colors
                .get(index as usize)
                .copied()
                .unwrap_or(Color::BLACK);
            let encoded = Self::from_linear(c, self.space);
            for v in encoded {
                out.push((v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            }
            out.push(px[3]);
        }
        Ok(out)
    }

    /// The entry nearest `color`, measured in the palette's own space. `None`
    /// for an empty palette.
    pub fn nearest(&self, color: Color) -> Option<usize> {
        let target = Self::from_linear(color, self.space);
        self.colors
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let c = Self::from_linear(*c, self.space);
                let d: f32 = (0..3).map(|k| (c[k] - target[k]).powi(2)).sum();
                (i, d)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    }

    /// The palette as the `[k, 3]` codebook the assignment graph takes.
    fn codebook(&self) -> Vec<f32> {
        self.colors
            .iter()
            .flat_map(|c| Self::from_linear(*c, self.space))
            .collect()
    }

    fn to_linear(rgb: [f32; 3], space: ColorSpace) -> Color {
        match space {
            ColorSpace::Linear => Color::new(rgb[0], rgb[1], rgb[2]),
            ColorSpace::Srgb => Color::new(
                super::srgb_to_linear(rgb[0]),
                super::srgb_to_linear(rgb[1]),
                super::srgb_to_linear(rgb[2]),
            ),
        }
    }

    fn from_linear(color: Color, space: ColorSpace) -> [f32; 3] {
        match space {
            ColorSpace::Linear => [color.r, color.g, color.b],
            ColorSpace::Srgb => [
                super::linear_to_srgb(color.r),
                super::linear_to_srgb(color.g),
                super::linear_to_srgb(color.b),
            ],
        }
    }
}

/// `argmin_k ‖pixel − codebook[k]‖²`, for every pixel at once.
fn assignment_runner(pixels: usize, codes: usize, device: Device) -> GraphRunner {
    let mut g = Graph::new("assign");
    let x = g.input("pixels", Shape::new(&[pixels, 3], DType::F32));
    let codebook = g.param("codebook", Shape::new(&[codes, 3], DType::F32));
    let (indices, _quantized) = g.vector_quantize(x, codebook, VqMetric::L2);
    g.set_outputs(vec![indices]);
    GraphRunner::new(g, device)
}

/// Every pixel of the frame as an `[n, 3]` tensor.
fn all_pixels(rgba: &[u8], space: ColorSpace) -> Tensor {
    let linear = matches!(space, ColorSpace::Linear);
    let mut data = Vec::with_capacity(rgba.len() / 4 * 3);
    for px in rgba.chunks_exact(4) {
        for &channel in &px[..3] {
            let v = channel as f32 / 255.0;
            data.push(if linear { super::srgb_to_linear(v) } else { v });
        }
    }
    let n = data.len() / 3;
    Tensor::new(data, &[n, 3]).expect("three per pixel")
}

/// A stride-sampled subset, for the clustering itself.
fn sample_pixels(rgba: &[u8], wanted: usize, space: ColorSpace) -> Tensor {
    let pixels = rgba.len() / 4;
    if pixels == 0 {
        return Tensor::new(Vec::new(), &[0, 3]).expect("empty");
    }
    let wanted = wanted.clamp(1, pixels);
    let stride = (pixels / wanted).max(1);
    let linear = matches!(space, ColorSpace::Linear);
    let mut data = Vec::with_capacity(wanted * 3);
    for p in (0..pixels).step_by(stride) {
        for c in 0..3 {
            let v = rgba[p * 4 + c] as f32 / 255.0;
            data.push(if linear { super::srgb_to_linear(v) } else { v });
        }
    }
    let n = data.len() / 3;
    Tensor::new(data, &[n, 3]).expect("three per pixel")
}

/// Initial centroids, spread by *colour* rather than by position: the first
/// sample, then repeatedly whichever sample is furthest from everything
/// already chosen.
///
/// Deterministic, which k-means++ is not — a palette that changes between runs
/// changes every frame of a sequence that uses it. And unlike picking every
/// `n/k`-th sample, it cannot alias: a striped or checkered image is periodic
/// in sample *order*, so a stride lands on the same colour every time and
/// seeds `k` copies of it. Farthest-point seeding looks at the colours.
///
/// O(k·n), which for a few dozen colours over twenty thousand samples is
/// nothing next to the assignment step it precedes.
fn seed_centroids(samples: &Tensor, count: usize) -> Vec<f32> {
    let data = samples.data();
    let n = samples.dims()[0];
    let count = count.min(n.max(1));
    if n == 0 {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(count * 3);
    out.extend_from_slice(&data[0..3]);

    // Distance from each sample to the nearest chosen seed, updated as seeds
    // are added rather than recomputed against all of them.
    let mut nearest: Vec<f32> = (0..n)
        .map(|p| {
            (0..3)
                .map(|j| (data[p * 3 + j] - out[j]).powi(2))
                .sum::<f32>()
        })
        .collect();

    for _ in 1..count {
        let (best, spread) =
            nearest
                .iter()
                .enumerate()
                .fold(
                    (0usize, -1.0f32),
                    |(bi, bd), (i, d)| {
                        if *d > bd {
                            (i, *d)
                        } else {
                            (bi, bd)
                        }
                    },
                );
        // Everything left is already a seed: a shorter palette is the honest
        // answer, and duplicates would be the dishonest one.
        if spread <= 0.0 {
            break;
        }
        let base = out.len();
        out.extend_from_slice(&data[best * 3..best * 3 + 3]);
        for (p, near) in nearest.iter_mut().enumerate() {
            let d: f32 = (0..3)
                .map(|j| (data[p * 3 + j] - out[base + j]).powi(2))
                .sum();
            *near = near.min(d);
        }
    }
    out
}

/// The update step: each centroid becomes the mean of what chose it.
///
/// Returns the new centroids and whether any of them moved appreciably, so a
/// converged run can stop early instead of grinding through its iteration
/// budget. A centroid nothing chose is re-seeded onto the sample furthest from
/// its own centroid — the standard fix, and the one that turns a wasted entry
/// into the one the palette was missing.
fn update(samples: &Tensor, assignment: &[f32], centroids: &[f32]) -> (Vec<f32>, bool) {
    let k = centroids.len() / 3;
    let data = samples.data();
    let n = data.len() / 3;

    let mut sums = vec![0.0f32; k * 3];
    let mut counts = vec![0u32; k];
    for (p, a) in assignment.iter().enumerate().take(n) {
        let c = (*a as usize).min(k.saturating_sub(1));
        counts[c] += 1;
        for j in 0..3 {
            sums[c * 3 + j] += data[p * 3 + j];
        }
    }

    let mut next = centroids.to_vec();
    let mut moved = false;
    for c in 0..k {
        if counts[c] == 0 {
            continue;
        }
        for j in 0..3 {
            let mean = sums[c * 3 + j] / counts[c] as f32;
            if (mean - next[c * 3 + j]).abs() > 1e-4 {
                moved = true;
            }
            next[c * 3 + j] = mean;
        }
    }

    // Re-seed the empties onto the worst-served sample.
    let empty: Vec<usize> = (0..k).filter(|c| counts[*c] == 0).collect();
    if !empty.is_empty() {
        let mut worst: Vec<(f32, usize)> = (0..n)
            .map(|p| {
                let c = (assignment[p] as usize).min(k.saturating_sub(1));
                let d: f32 = (0..3)
                    .map(|j| (data[p * 3 + j] - next[c * 3 + j]).powi(2))
                    .sum();
                (d, p)
            })
            .collect();
        worst.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (slot, c) in empty.into_iter().enumerate() {
            if let Some((_, p)) = worst.get(slot) {
                next[c * 3..c * 3 + 3].copy_from_slice(&data[p * 3..p * 3 + 3]);
                moved = true;
            }
        }
    }
    (next, moved)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rlx::preferred_device;

    /// A frame made of exactly `colors`, repeated.
    fn frame_of(colors: &[[u8; 3]], per_color: usize) -> Vec<u8> {
        let mut out = Vec::new();
        for _ in 0..per_color {
            for c in colors {
                out.extend_from_slice(&[c[0], c[1], c[2], 255]);
            }
        }
        out
    }

    #[test]
    fn an_image_of_three_colours_yields_those_three_colours() {
        let wanted = [[220u8, 30, 30], [30, 200, 60], [40, 60, 210]];
        let frame = frame_of(&wanted, 300);
        let palette =
            Palette::extract(&frame, 3, &PaletteOptions::default(), preferred_device()).unwrap();

        assert_eq!(palette.len(), 3);
        for c in wanted {
            let target = Color::new(
                super::super::srgb_to_linear(c[0] as f32 / 255.0),
                super::super::srgb_to_linear(c[1] as f32 / 255.0),
                super::super::srgb_to_linear(c[2] as f32 / 255.0),
            );
            let nearest = palette.nearest(target).unwrap();
            let found = Palette::from_linear(palette.colors[nearest], palette.space);
            for j in 0..3 {
                let want = c[j] as f32 / 255.0;
                assert!(
                    (found[j] - want).abs() < 0.02,
                    "channel {j}: found {}, wanted {want}",
                    found[j]
                );
            }
        }
    }

    #[test]
    fn asking_for_more_colours_than_exist_gives_the_ones_that_do() {
        let frame = frame_of(&[[255, 255, 255], [0, 0, 0]], 200);
        let palette =
            Palette::extract(&frame, 8, &PaletteOptions::default(), preferred_device()).unwrap();
        assert!(
            palette.len() <= 2,
            "expected at most 2 entries, got {}",
            palette.len()
        );
    }

    #[test]
    fn posterizing_snaps_every_pixel_onto_the_palette() {
        let wanted = [[220u8, 30, 30], [30, 200, 60]];
        let frame = frame_of(&wanted, 200);
        let device = preferred_device();
        let palette = Palette::extract(&frame, 2, &PaletteOptions::default(), device).unwrap();
        let posterized = palette.posterize(&frame, device).unwrap();

        let swatches: Vec<[u8; 3]> = palette
            .colors
            .iter()
            .map(|c| {
                let e = Palette::from_linear(*c, palette.space);
                [
                    (e[0] * 255.0 + 0.5) as u8,
                    (e[1] * 255.0 + 0.5) as u8,
                    (e[2] * 255.0 + 0.5) as u8,
                ]
            })
            .collect();
        for px in posterized.chunks_exact(4) {
            assert!(
                swatches.iter().any(|s| s
                    .iter()
                    .zip(px)
                    .all(|(a, b)| (*a as i32 - *b as i32).abs() <= 1)),
                "pixel {:?} is not a palette entry ({swatches:?})",
                &px[..3]
            );
            assert_eq!(px[3], 255);
        }
    }

    #[test]
    fn zero_iterations_still_returns_the_seeded_colours() {
        // Two things at once. The used-set is computed from an assignment
        // against the centroids actually returned, so it never describes a
        // previous round — and the seeds are spread by colour, so this
        // strictly alternating frame does not seed two copies of the red.
        let frame = frame_of(&[[220u8, 30, 30], [30, 200, 60]], 200);
        let palette = Palette::extract(
            &frame,
            2,
            &PaletteOptions {
                iterations: 0,
                ..Default::default()
            },
            preferred_device(),
        )
        .unwrap();
        assert_eq!(palette.len(), 2, "seeded colours were reported unused");
    }

    #[test]
    fn an_empty_frame_assigns_and_posterizes_to_nothing() {
        // Not a graph with zero rows — that is a zero-size buffer, which the
        // wgpu backend panics on rather than allocating.
        let device = preferred_device();
        let palette = Palette::new(vec![Color::WHITE, Color::BLACK]);
        assert!(palette.assign(&[], device).unwrap().is_empty());
        assert!(palette.posterize(&[], device).unwrap().is_empty());
    }

    #[test]
    fn an_empty_frame_gives_an_empty_palette() {
        let palette =
            Palette::extract(&[], 4, &PaletteOptions::default(), preferred_device()).unwrap();
        assert!(palette.is_empty());
        assert_eq!(palette.nearest(Color::WHITE), None);
    }
}
