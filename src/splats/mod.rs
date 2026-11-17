//! 3D Gaussian splats: scenes captured as millions of coloured, translucent
//! ellipsoids rather than triangles (Kerbl et al., *3D Gaussian Splatting for
//! Real-Time Radiance Field Rendering*, SIGGRAPH 2023). Behind the opt-in
//! `splats` feature.
//!
//! ```
//! use threers::splats::{splat_object, GaussianSplats, SplatOptions};
//! use threers::prelude::*;
//!
//! let mut splats = GaussianSplats::default();
//! splats.push(Vector3::ZERO, [0.2, 0.05, 0.05], [0.0, 0.0, 0.0, 1.0], 0.9, [1.0, 0.4, 0.1]);
//!
//! let mut scene = Scene::new();
//! let (object, mut sorter) = splat_object(&splats, &SplatOptions::default()).unwrap();
//! let id = scene.add(object);
//!
//! let camera = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
//! // Once a frame, before rendering: back to front for this camera.
//! sorter.update(&mut scene, id, &camera);
//! ```
//!
//! **Loading.** [`GaussianSplats::from_ply`] reads the PLY that the reference
//! trainer and most of its descendants write (`x y z`, `f_dc_*`, `f_rest_*`,
//! `opacity`, `scale_*`, `rot_*`, with the log-scales and logit opacities
//! decoded), at any spherical-harmonic degree up to 3.
//! [`GaussianSplats::from_splat`] reads the 32-byte-a-splat `.splat` format
//! web viewers pass around. Both write back out
//! ([`to_ply`](GaussianSplats::to_ply), [`to_splat`](GaussianSplats::to_splat)).
//! [`from_ply`](GaussianSplats::from_ply) also reads SuperSplat's
//! chunk-compressed PLY, and [`GaussianSplats::from_spz`] Niantic's `.spz`
//! (every version; degree-4 harmonics are cut to 3). Read only, both:
//! [`GaussianSplats::from_bytes`] picks the reader from a file's first bytes.
//! Files keep their own axes: trainers write `y` down (turn the object over
//! about `x` to stand it up), `.spz` is stored `y` up.
//!
//! **Drawing** ([`splat_object`]). Each splat is projected to the screen as
//! the reference renderer does it — the 3D covariance through the camera's
//! local affine approximation (EWA splatting), plus a 0.3 px² low-pass — and
//! drawn as a quad covering three standard deviations, blended front over
//! back. The splats sit in the scene with everything else: they are depth
//! tested against opaque meshes and sorted with other transparent objects as
//! a whole. The object's transform applies, scale and rotation included, and
//! view-dependent colour is evaluated in the splats' own frame.
//!
//! **Sorting** ([`SplatSorter`]). Blending needs the splats back to front, so
//! they are sorted on the CPU for each new view direction and only the order
//! (four bytes a splat) is uploaded; a camera that only moves sideways along
//! its own view plane, or stands still, costs nothing. Call
//! [`SplatSorter::update`] once a frame. The sort is the
//! `threers-splat-sort` crate: a radix sort over keys made in SIMD-shaped
//! loops, about 2.5 ms a million splats natively and 5 in the browser. It is
//! a crate of its own so the size-optimised wasm profile can still compile it
//! for speed; build the web package with `SIMD=1` for its `simd128` keys.
//!
//! Quads reach only as far as a splat stays visible — three standard
//! deviations for an opaque one, fewer for a faint one, which never reaches
//! 1/255 past `√(2 ln 255α)` — and a splat that never reaches it is skipped.
//! Captured scenes are full of large, faint splats; in one this cut the
//! frame from 42 ms to 17.
//!
//! **What it runs on.** WebGPU, WebGL2 and native wgpu, and the native Metal
//! renderer, through [`ShaderMaterial`](crate::ShaderMaterial): the splat
//! data lives in its storage arrays, read through `s0`/`s1`/`s2`, so on
//! WebGL2 it comes from a texture. That sets the limits: up to 2²⁴ splats
//! (the order is stored as exact `f32`s), and as many as fit in one storage
//! binding — about 2.1 million at wgpu's default 128 MiB
//! ([`SplatOptions::max_storage_bytes`]). On WebGL2 the slot is a texture
//! 2048 texels wide and as tall as the browser's largest texture: at the
//! 2048 wgpu assumes as a floor that is 1 million splats, but browsers
//! report more (8192 and up), and 2 million draw in Chrome's WebGL2.

mod io;
mod render;
mod spz;

pub use io::SplatError;
/// The depth sort itself (`threers-splat-sort`), to run on another thread:
/// see [`SplatSorter::view_axis`].
pub use threers_splat_sort::DepthSort;
pub use render::{
    splat_material, splat_object, splat_sh_degree, SplatOptions, SplatSorter,
    DEFAULT_MAX_STORAGE_BYTES, SPLATS_PER_INSTANCE,
};

use crate::math::{Box3, Vector3};

/// The degree-0 spherical-harmonic basis constant, `1 / (2√π)`: a colour is
/// `0.5 + SH_C0 * f_dc`.
pub const SH_C0: f32 = 0.282_094_8;

/// Spherical-harmonic coefficients a colour channel has beyond the first, at
/// `degree`: 0, 3, 8 or 15.
pub fn sh_rest_coefficients(degree: u32) -> usize {
    let k = (degree.min(3) + 1) as usize;
    k * k - 1
}

/// A set of 3D Gaussians, one per index across the per-splat vectors (which
/// all have [`len`](Self::len) entries).
///
/// Everything is in the units the splats are drawn in: scales are standard
/// deviations, not their logarithms, and opacities are 0..1, not logits —
/// the loaders decode what the files store.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GaussianSplats {
    /// Centres, in object space.
    pub positions: Vec<[f32; 3]>,
    /// Standard deviation along each of the splat's own axes, before
    /// rotation.
    pub scales: Vec<[f32; 3]>,
    /// Unit quaternions `[x, y, z, w]` (three.js order) turning the splat's
    /// axes into object space.
    pub rotations: Vec<[f32; 4]>,
    /// Peak opacity, 0..1.
    pub opacities: Vec<f32>,
    /// The view-independent colour (spherical-harmonic band 0, `0.5 + SH_C0 *
    /// f_dc`), in the display (sRGB-encoded) space splats are trained in. It
    /// may stray outside 0..1; the higher bands pull it back, and what is drawn
    /// is clamped.
    pub colors: Vec<[f32; 3]>,
    /// Degree of the view-dependent colour in [`sh`](Self::sh), 0..=3.
    pub sh_degree: u32,
    /// The higher spherical-harmonic bands:
    /// [`sh_rest_coefficients`]`(sh_degree)` RGB triples a splat,
    /// coefficient-major — splat `i`'s coefficient `k` is
    /// `sh[(i * n + k) * 3..][..3]`. Empty at degree 0.
    pub sh: Vec<f32>,
    /// Trained with Mip-Splatting's opacity compensation, as `.spz` files
    /// can say: drawn as [`SplatOptions::antialiased`] draws them.
    pub antialiased: bool,
}

impl GaussianSplats {
    /// How many splats there are.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Add a splat with no view-dependent colour (zero higher bands, if the set
    /// has any). `rotation` is `[x, y, z, w]` and is normalised.
    pub fn push(
        &mut self,
        position: Vector3,
        scale: [f32; 3],
        rotation: [f32; 4],
        opacity: f32,
        color: [f32; 3],
    ) {
        self.positions.push([position.x, position.y, position.z]);
        self.scales.push(scale);
        self.rotations.push(normalize_quat(rotation));
        self.opacities.push(opacity);
        self.colors.push(color);
        let rest = sh_rest_coefficients(self.sh_degree) * 3;
        self.sh.extend(std::iter::repeat_n(0.0, rest));
    }

    /// Check that every per-splat vector has [`len`](Self::len) entries and
    /// [`sh`](Self::sh) the size its degree implies.
    pub fn validate(&self) -> Result<(), SplatError> {
        let n = self.len();
        let lens = [
            ("scales", self.scales.len()),
            ("rotations", self.rotations.len()),
            ("opacities", self.opacities.len()),
            ("colors", self.colors.len()),
        ];
        if self.sh_degree > 3 {
            return Err(SplatError::new(format!(
                "spherical-harmonic degree {} is past 3",
                self.sh_degree
            )));
        }
        let sh_expected = n * sh_rest_coefficients(self.sh_degree) * 3;
        if self.sh.len() != sh_expected {
            return Err(SplatError::new(format!(
                "sh has {} floats, but {n} splats at degree {} need {sh_expected}",
                self.sh.len(),
                self.sh_degree
            )));
        }
        for (name, len) in lens {
            if len != n {
                return Err(SplatError::new(format!(
                    "{name} has {len} entries for {n} positions"
                )));
            }
        }
        Ok(())
    }

    /// The covariance of splat `i` in object space, `R S Sᵀ Rᵀ`, as its upper
    /// triangle `[xx, xy, xz, yy, yz, zz]`.
    pub fn covariance(&self, i: usize) -> [f32; 6] {
        covariance(self.scales[i], self.rotations[i])
    }

    /// The box round every splat's centre.
    pub fn bounding_box(&self) -> Box3 {
        let mut b = Box3::empty();
        for p in &self.positions {
            b.expand_by_point(Vector3::new(p[0], p[1], p[2]));
        }
        b
    }

    /// Drop the spherical-harmonic bands past `degree`.
    pub fn truncate_sh(&mut self, degree: u32) {
        if degree >= self.sh_degree {
            return;
        }
        let from = sh_rest_coefficients(self.sh_degree) * 3;
        let to = sh_rest_coefficients(degree) * 3;
        self.sh = self
            .sh
            .chunks_exact(from)
            .flat_map(|c| c[..to].iter().copied())
            .collect();
        self.sh_degree = degree;
    }

    /// Where to aim a camera at these splats, in their own space: the
    /// per-axis median of their centres (trained scenes have far-flung
    /// outliers that would drag a box's centre off), and how far from it to
    /// look from. For a scanned object, far enough to see all of it with a
    /// 50° field of view; for a captured scene, whose background stretches
    /// far past its subject, close — among where its cameras were, since from
    /// outside a scene shows the floaters its training never had to clear.
    ///
    /// The two are told apart by how far the splats spread past their median
    /// distance: in captured scenes the 90th percentile was 5.5–10 times it,
    /// in scanned objects 1.7–3.1 times.
    pub fn framing(&self) -> (Vector3, f32) {
        let finite = || self.positions.iter().filter(|p| p.iter().all(|v| v.is_finite()));
        if finite().next().is_none() {
            return (Vector3::ZERO, 1.0);
        }
        let at = |v: &mut Vec<f32>, f: f32| {
            let k = ((v.len() - 1) as f32 * f) as usize;
            *v.select_nth_unstable_by(k, f32::total_cmp).1
        };
        let median_on = |k: usize| at(&mut finite().map(|p| p[k]).collect(), 0.5);
        let c = Vector3::new(median_on(0), median_on(1), median_on(2));
        let mut d: Vec<f32> = finite()
            .map(|p| (Vector3::new(p[0], p[1], p[2]) - c).length())
            .collect();
        let (median, p90) = (at(&mut d, 0.5), at(&mut d, 0.9));
        let distance = if p90 < 4.0 * median {
            // An object: its 90th percentile inside the view, with margin.
            2.0 * p90
        } else {
            1.2 * median
        };
        (c, distance.max(1e-3))
    }

    /// The colour splat `i` shows looking along `dir` (object space, from the
    /// camera towards the splat; need not be unit length): band 0 plus the
    /// higher bands, clamped below at zero — what the shader computes.
    pub fn color_towards(&self, i: usize, dir: Vector3) -> [f32; 3] {
        let n = sh_rest_coefficients(self.sh_degree);
        let rest = &self.sh[i * n * 3..][..n * 3];
        let c = eval_sh(self.colors[i], self.sh_degree, rest, dir.normalize());
        c.map(|v| v.max(0.0))
    }
}

/// `R S Sᵀ Rᵀ` for scales `s` and the unit quaternion `q = [x, y, z, w]`, as
/// `[xx, xy, xz, yy, yz, zz]`.
pub fn covariance(s: [f32; 3], q: [f32; 4]) -> [f32; 6] {
    let [x, y, z, w] = normalize_quat(q);
    let r = [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ];
    // M = R S: column j of R scaled by s[j]; Σ = M Mᵀ.
    let m = |i: usize, j: usize| r[i][j] * s[j];
    let e = |a: usize, b: usize| (0..3).map(|k| m(a, k) * m(b, k)).sum::<f32>();
    [e(0, 0), e(0, 1), e(0, 2), e(1, 1), e(1, 2), e(2, 2)]
}

pub(crate) fn normalize_quat(q: [f32; 4]) -> [f32; 4] {
    let len = q.iter().map(|v| v * v).sum::<f32>().sqrt();
    if len > 0.0 && len.is_finite() {
        q.map(|v| v / len)
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

const SH_C1: f32 = 0.488_602_5;
const SH_C2: [f32; 5] = [1.092_548_4, -1.092_548_4, 0.315_391_57, -1.092_548_4, 0.546_274_2];
const SH_C3: [f32; 7] = [
    -0.590_043_6,
    2.890_611_4,
    -0.457_045_8,
    0.373_176_33,
    -0.457_045_8,
    1.445_305_7,
    -0.590_043_6,
];

/// Band 0 (`base`) plus the bands up to `degree` from `rest` (coefficient-
/// major RGB) towards the unit direction `d`, as the reference rasteriser's
/// `computeColorFromSH` evaluates them.
fn eval_sh(base: [f32; 3], degree: u32, rest: &[f32], d: Vector3) -> [f32; 3] {
    let (x, y, z) = (d.x, d.y, d.z);
    let mut basis = Vec::with_capacity(15);
    if degree >= 1 {
        basis.extend([-SH_C1 * y, SH_C1 * z, -SH_C1 * x]);
    }
    if degree >= 2 {
        let (xx, yy, zz) = (x * x, y * y, z * z);
        basis.extend([
            SH_C2[0] * x * y,
            SH_C2[1] * y * z,
            SH_C2[2] * (2.0 * zz - xx - yy),
            SH_C2[3] * x * z,
            SH_C2[4] * (xx - yy),
        ]);
    }
    if degree >= 3 {
        let (xx, yy, zz) = (x * x, y * y, z * z);
        basis.extend([
            SH_C3[0] * y * (3.0 * xx - yy),
            SH_C3[1] * x * y * z,
            SH_C3[2] * y * (4.0 * zz - xx - yy),
            SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
            SH_C3[4] * x * (4.0 * zz - xx - yy),
            SH_C3[5] * z * (xx - yy),
            SH_C3[6] * x * (xx - 3.0 * yy),
        ]);
    }
    let mut c = base;
    for (k, b) in basis.iter().enumerate() {
        for ch in 0..3 {
            c[ch] += b * rest[k * 3 + ch];
        }
    }
    c
}

