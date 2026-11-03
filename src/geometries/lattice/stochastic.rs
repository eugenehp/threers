//! Stochastic lattices — foams, not tilings.
//!
//! The other three families repeat one cell on a lattice of integers. These
//! two do not repeat at all: they are built from a seeded random field, so
//! every cell is a different shape and no direction is special. That is the
//! point of them. A beam lattice is stiff along its struts and soft between
//! them, and however the part is loaded some of those directions are wasted;
//! a foam approaches isotropy at a few cells across, which is why real foams,
//! bone and every energy-absorbing pad are built this way.
//!
//! | Family | Field | What it is |
//! |--------|-------|------------|
//! | [`Stochastic::Voronoi`] | distance to the Voronoi edges of a jittered point set | open-cell foam: struts, fully connected void |
//! | [`Stochastic::VoronoiWall`] | distance to the Voronoi faces | closed-cell foam: sealed bubbles |
//! | [`Stochastic::Spinodal`] and friends | a Gaussian random field of plane waves | spinodal decomposition — what a quenched alloy or a phase-separating polymer freezes into |
//!
//! # Randomness that does not move
//!
//! Nothing here holds a point set or an RNG. The seeds are a hash of their
//! cell index and the wave directions a hash of their own index, so the field
//! is a pure function of the point and the [`seed`](super::Lattice::seed): the
//! same lattice samples the same everywhere, at any resolution, on any thread,
//! in any process. A stored point set would have to be sorted, spatially
//! indexed and serialised to make the same promise.
//!
//! # Spinodoids
//!
//! [`Stochastic::Spinodal`] draws its wave vectors uniformly over the sphere,
//! which gives an isotropic solid. Restricting them to cones instead gives the
//! anisotropic classes of Kumar et al. (2020) — lamellar, columnar and cubic —
//! whose stiffness can be tuned by direction without changing the density.
//! Waves *along* an axis stack plates across it; waves *around* the equator
//! leave columns along it.

use crate::math::Vector3;

mod graph;
pub(crate) use graph::{shared as shared_graph, EdgeBins, SeedGraph};
use std::f32::consts::TAU;

/// How many plane waves make a spinodal field.
///
/// The central limit theorem is what makes the sum Gaussian, and it arrives
/// quickly: past about sixteen waves the one-point statistics stop changing
/// and only the cost keeps going up. Sixteen is also enough that the cubic
/// class gets a fair share on each of its three axes.
const WAVES: usize = 24;

/// Half-angle of the cones the anisotropic classes draw their waves from.
const CONE: f32 = 0.261_799_4; // 15°

/// A random-field cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stochastic {
    /// Open-cell foam: round struts along the edges of a Voronoi diagram.
    ///
    /// Every void connects to every other, which is what a filter, a scaffold
    /// or anything that has to drain powder needs.
    Voronoi,
    /// Closed-cell foam: the Voronoi faces themselves, as walls.
    ///
    /// Sealed bubbles — buoyant and good at absorbing a blast, impossible to
    /// clear un-fused powder out of.
    VoronoiWall,
    /// Wet foam: rounded bubbles parted by thin films, with the liquid gathered
    /// into Plateau borders — the three-sided channels along every cell edge,
    /// and the nodes where four meet. A real liquid or polymer foam before it
    /// sets looks like this, and so does the part printed from one: thick,
    /// rounded struts and thin membranes at once. See
    /// [`plateau_border`](super::Lattice::plateau_border).
    VoronoiWet,
    /// Isotropic spinodal decomposition: waves drawn over the whole sphere.
    Spinodal,
    /// Waves within 15° of z: plates stacked across the axis. Stiff in plane,
    /// soft across it.
    SpinodalLamellar,
    /// Waves within 15° of the xy plane: columns running along z. Stiff along
    /// the axis, soft across it.
    SpinodalColumnar,
    /// Waves within 15° of each of the three axes: cubic symmetry, stiffer on
    /// the axes than on the diagonals.
    SpinodalCubic,
    /// A Gaussian random field with a broad spectrum: the waves point every
    /// way, as [`Spinodal`](Stochastic::Spinodal)'s do, but their wavelengths
    /// spread ±45 % about the cell instead of all matching it. The pores come
    /// in a range of sizes rather than one — closer to a sintered powder or a
    /// trabecular bone than to a quenched alloy.
    GaussianField,
    /// Fibres: one elongated Gaussian kernel per cell, at a jittered point and
    /// pointing anywhere, and the solid where they add up past a level. The
    /// jittered grid is what makes it hyperuniform — the fibres are as evenly
    /// spread as a lattice's at long range, with none of a lattice's order up
    /// close, so there are no clumps and no voids much bigger than a cell.
    Hyperuniform,
    /// A Turing pattern: two chemicals that react and diffuse (Gray–Scott)
    /// run on a periodic tile until they settle into a labyrinth, and the
    /// solid where the activator is high. The same mechanism lays out
    /// zebrafish stripes and coral; as a lattice it is a bicontinuous maze
    /// with walls of one width and no preferred direction, grown rather than
    /// summed. The pattern is simulated once per seed — about a second on one
    /// core — and cached for the process.
    ReactionDiffusion,
}

impl Stochastic {
    /// Every stochastic cell, in declaration order.
    pub const ALL: [Stochastic; 10] = [
        Stochastic::Voronoi,
        Stochastic::VoronoiWall,
        Stochastic::VoronoiWet,
        Stochastic::Spinodal,
        Stochastic::SpinodalLamellar,
        Stochastic::SpinodalColumnar,
        Stochastic::SpinodalCubic,
        Stochastic::GaussianField,
        Stochastic::Hyperuniform,
        Stochastic::ReactionDiffusion,
    ];

    /// Lower-case identifier, unique across every lattice family.
    pub fn name(self) -> &'static str {
        match self {
            Stochastic::Voronoi => "voronoi",
            Stochastic::VoronoiWall => "voronoi-wall",
            Stochastic::VoronoiWet => "voronoi-wet",
            Stochastic::Spinodal => "spinodal",
            Stochastic::SpinodalLamellar => "spinodal-lamellar",
            Stochastic::SpinodalColumnar => "spinodal-columnar",
            Stochastic::SpinodalCubic => "spinodal-cubic",
            Stochastic::GaussianField => "gaussian-field",
            Stochastic::Hyperuniform => "hyperuniform",
            Stochastic::ReactionDiffusion => "reaction-diffusion",
        }
    }

    /// The cell with this [`name`](Self::name).
    pub fn from_name(name: &str) -> Option<Stochastic> {
        Stochastic::ALL.into_iter().find(|s| s.name() == name)
    }

    /// Whether this cell is a level set of a random field rather than a
    /// distance to a skeleton.
    ///
    /// The field cells behave like a [`Tpms`](super::Tpms): they answer to
    /// [`LatticeStyle`](super::LatticeStyle), so the same generator gives
    /// either a sheet with two open void networks or a solid with one. The
    /// Voronoi cells are distances to a skeleton, like a
    /// [`Strut`](super::Strut), and ignore the style.
    pub fn is_field(self) -> bool {
        !matches!(
            self,
            Stochastic::Voronoi | Stochastic::VoronoiWall | Stochastic::VoronoiWet
        )
    }
}

/// Which part of a Voronoi diagram [`voronoi_distance`] measures to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Skeleton {
    /// The edges: struts, an open-cell foam.
    Edges,
    /// The faces: walls, a closed-cell foam.
    Walls,
    /// The bubbles of a wet foam: every cell shrunk by `border` and grown back
    /// by it, which rounds its edges and corners off. The answer is the
    /// distance to the nearest bubble, negative inside one.
    Wet { border: f32 },
}

/// Where the seeds of a Voronoi foam sit.
///
/// [`Random`](SeedLayout::Random) is the foam proper: one seed per cell,
/// wandering by [`jitter`](super::Lattice::jitter). The rest put the seeds on
/// a crystal lattice instead, which makes every cell the same handful of
/// shapes — the ordered foams the theory is written about, and the ones a
/// test rig can compare against. Lattice layouts ignore the jitter.
///
/// | Layout | Seeds a cell | Cells |
/// |--------|--------------|-------|
/// | [`Kelvin`](SeedLayout::Kelvin) | 2 (body-centred cubic) | truncated octahedra, 14 faces — Kelvin's 1887 answer |
/// | [`Fcc`](SeedLayout::Fcc) | 4 (face-centred cubic) | rhombic dodecahedra, 12 faces |
/// | [`WeairePhelan`](SeedLayout::WeairePhelan) | 8 (A15) | 2 of 12 faces and 6 of 14 — 0.3 % less wall than Kelvin's for the same volume (Weaire & Phelan 1994) |
/// | [`C15`](SeedLayout::C15) | 24 (Laves C15) | 8 of 16 faces and 16 of 12 |
/// | [`Poisson`](SeedLayout::Poisson) | [`seeds_per_cell`](super::Lattice::seeds_per_cell), well spread | irregular, of similar size |
/// | [`Relaxed`](SeedLayout::Relaxed) | the same, relaxed | irregular, nearly equal in volume |
///
/// The last two are generated once per lattice as one periodic tile, so the
/// foam repeats with the cell like the crystals do.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SeedLayout {
    #[default]
    Random,
    Kelvin,
    Fcc,
    WeairePhelan,
    C15,
    /// [`seeds_per_cell`](super::Lattice::seeds_per_cell) random seeds a cell,
    /// spread by best-candidate sampling: each new seed the furthest from the
    /// rest of a few dozen tries. No two crowd each other, which is what a
    /// Poisson-disk set promises, and the cells come out more even in size
    /// than [`Random`](SeedLayout::Random)'s without any order.
    Poisson,
    /// [`Poisson`](SeedLayout::Poisson) seeds moved, a dozen times over, to the
    /// centres of their own cells (Lloyd's relaxation). The cells settle to
    /// nearly equal volumes and shapes close to the ones soap films find — a
    /// random foam that looks grown rather than scattered.
    Relaxed,
    /// Two bubble sizes: a quarter of
    /// [`seeds_per_cell`](super::Lattice::seeds_per_cell) large, the rest
    /// small, as a power diagram so the walls stay flat — the mix that packs
    /// denser than one size, as gravel does with sand.
    /// [`size_ratio`](super::Lattice::size_ratio) sets how much bigger.
    /// Open, closed and wet foams alike — a wet one's bubbles come in the
    /// two sizes too.
    Bimodal,
}

impl SeedLayout {
    pub const ALL: [SeedLayout; 8] = [
        SeedLayout::Random,
        SeedLayout::Kelvin,
        SeedLayout::Fcc,
        SeedLayout::WeairePhelan,
        SeedLayout::C15,
        SeedLayout::Poisson,
        SeedLayout::Relaxed,
        SeedLayout::Bimodal,
    ];

    pub fn name(self) -> &'static str {
        match self {
            SeedLayout::Random => "random",
            SeedLayout::Kelvin => "kelvin",
            SeedLayout::Fcc => "fcc",
            SeedLayout::WeairePhelan => "weaire-phelan",
            SeedLayout::C15 => "c15",
            SeedLayout::Poisson => "poisson",
            SeedLayout::Relaxed => "relaxed",
            SeedLayout::Bimodal => "bimodal",
        }
    }

    pub fn from_name(name: &str) -> Option<SeedLayout> {
        SeedLayout::ALL.into_iter().find(|s| s.name() == name)
    }

    /// The fixed seeds of one cell, in cell units in `[0, 1)³`. Empty for
    /// [`Random`](SeedLayout::Random), whose one seed is hashed per cell, and
    /// for the generated layouts — see [`seed_tile`].
    pub(crate) fn sites(self) -> &'static [[f64; 3]] {
        match self {
            SeedLayout::Random
            | SeedLayout::Poisson
            | SeedLayout::Relaxed
            | SeedLayout::Bimodal => &[],
            SeedLayout::Kelvin => &[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            SeedLayout::Fcc => &[
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ],
            // A15: 2a at the corner and centre, 6c in pairs on the faces.
            SeedLayout::WeairePhelan => &[
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [0.25, 0.0, 0.5],
                [0.75, 0.0, 0.5],
                [0.5, 0.25, 0.0],
                [0.5, 0.75, 0.0],
                [0.0, 0.5, 0.25],
                [0.0, 0.5, 0.75],
            ],
            // C15 (Fd-3m, origin choice 1): 8a at (0,0,0) and (¾,¼,¾), 16d at
            // (⅝,⅝,⅝) and its three images, each under the face centring.
            SeedLayout::C15 => &[
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
                [0.75, 0.25, 0.75],
                [0.75, 0.75, 0.25],
                [0.25, 0.25, 0.25],
                [0.25, 0.75, 0.75],
                [0.625, 0.625, 0.625],
                [0.625, 0.125, 0.125],
                [0.125, 0.625, 0.125],
                [0.125, 0.125, 0.625],
                [0.375, 0.875, 0.125],
                [0.375, 0.375, 0.625],
                [0.875, 0.875, 0.625],
                [0.875, 0.375, 0.125],
                [0.875, 0.125, 0.375],
                [0.875, 0.625, 0.875],
                [0.375, 0.125, 0.875],
                [0.375, 0.625, 0.375],
                [0.125, 0.375, 0.875],
                [0.125, 0.875, 0.375],
                [0.625, 0.375, 0.375],
                [0.625, 0.875, 0.875],
            ],
        }
    }
}

/// Every cell's seeds for a layout, in cell units: the crystal sites, or the
/// generated tile of `count` — the same tile in every cell, so the foam repeats
/// with the cell and is still a pure function of the point and the seed.
/// Empty for [`SeedLayout::Random`].
pub(crate) fn seed_tile(
    layout: SeedLayout,
    count: usize,
    cell: Vector3,
    seed: u32,
) -> Vec<[f64; 3]> {
    match layout {
        SeedLayout::Poisson => best_candidate(count.max(1), cell, seed),
        SeedLayout::Relaxed => lloyd(best_candidate(count.max(1), cell, seed), cell, 12),
        // The sampler draws a bimodal tile itself, weights and all; this is
        // the positions at the default ratio, for callers that want only them.
        SeedLayout::Bimodal => bimodal_tile(count, 2.0, cell, seed).0,
        other => other.sites().to_vec(),
    }
}

/// Squared distance between two points of the periodic unit tile, in world
/// units: the nearest of the images.
fn tile_distance_sq(a: [f64; 3], b: [f64; 3], size: [f64; 3]) -> f64 {
    (0..3)
        .map(|i| {
            let d = a[i] - b[i];
            ((d - d.round()) * size[i]).powi(2)
        })
        .sum()
}

/// Distance to the edges, or the walls, of a power diagram: the Voronoi
/// diagram of weighted seeds, where seed `i` claims the points with the least
/// `|p − sᵢ|² − wᵢ`. A heavier seed has a bigger cell, which is how a foam gets
/// two bubble sizes and still has flat walls.
///
/// Every seed of the tile in the 3×3×3 cells round the point is a candidate,
/// so the answer is exact while the foam's cells are small beside a lattice
/// cell — as they are for the tiles that carry weights — and the pruning is
/// by the planes alone: a cell beyond its bisector with the point's own is no
/// nearer than that bisector, a plane further than the best answer so far
/// cannot cut a segment nearer than it.
pub(crate) fn power_distance(
    p: Vector3,
    origin: Vector3,
    cell: Vector3,
    wall: bool,
    tile: &[[f64; 3]],
    weights: &[f64],
    cap: f32,
) -> f32 {
    let size = [cell.x as f64, cell.y as f64, cell.z as f64];
    let u = [
        (p.x - origin.x) as f64 / size[0],
        (p.y - origin.y) as f64 / size[1],
        (p.z - origin.z) as f64 / size[2],
    ];
    let base = [u[0].floor(), u[1].floor(), u[2].floor()];
    // Each candidate: the vector to it and its power at p.
    let mut seeds: Vec<([f64; 3], f64)> = Vec::with_capacity(27 * tile.len());
    for oz in -1..=1 {
        for oy in -1..=1 {
            for ox in -1..=1 {
                let o = [ox as f64, oy as f64, oz as f64];
                for (s, &w) in tile.iter().zip(weights) {
                    let v = [0, 1, 2].map(|a| (base[a] + o[a] + s[a] - u[a]) * size[a]);
                    seeds.push((v, dot(v, v) - w));
                }
            }
        }
    }
    // Only the point's own cell has to come first; the rest are bounded
    // individually, so no order is needed beyond that.
    if let Some(own) = (0..seeds.len()).min_by(|&a, &b| seeds[a].1.total_cmp(&seeds[b].1)) {
        seeds.swap(0, own);
    }
    if seeds.len() < 2 {
        return cap;
    }
    // No seed beyond R₂ can matter. A cell j is only measured if its bisector
    // with p's could be nearer than the cap, which bounds |vⱼ| by R₁; a seed k
    // only bounds such a cell within the cap if their bisector could be, which
    // bounds |vₖ| by R₂ in the same way. (Both from c ≥ Δpower / 2(|vⱼ|+|vₖ|).)
    {
        let cap = (cap as f64).max(0.0);
        let w_max = weights.iter().copied().fold(0.0f64, f64::max);
        let (v0, p0) = seeds[0];
        let n0 = dot(v0, v0).sqrt();
        let r1 = cap + (cap * cap + p0 + w_max + 2.0 * cap * n0).max(0.0).sqrt();
        let r2 = cap + (cap * cap + r1 * r1 + w_max + 2.0 * cap * r1).sqrt();
        let own = seeds[0];
        seeds.retain(|(v, _)| dot(*v, *v) < r2 * r2);
        // The own seed is within R₁ by construction; keep it first regardless.
        if seeds.first() != Some(&own) {
            seeds.retain(|s| *s != own);
            seeds.insert(0, own);
        }
    }
    let norm: Vec<f64> = seeds.iter().map(|(v, _)| dot(*v, *v).sqrt()).collect();
    // The bisector between j and k is at least this far, since the seeds are
    // no further apart than their two distances from p.
    let lower =
        |j: usize, k: usize| (seeds[k].1 - seeds[j].1) / (2.0 * (norm[j] + norm[k]).max(1e-12));
    let plane = |j: usize, k: usize| -> Option<Plane> {
        let (vj, pj) = seeds[j];
        let (vk, pk) = seeds[k];
        let w = [vk[0] - vj[0], vk[1] - vj[1], vk[2] - vj[2]];
        let l = dot(w, w).sqrt();
        (l > 1e-12).then(|| Plane {
            n: [w[0] / l, w[1] / l, w[2] / l],
            c: (pk - pj) / (2.0 * l),
        })
    };
    let cap = (cap as f64).max(0.0);
    if wall {
        let best = (1..seeds.len())
            .filter_map(|k| plane(0, k))
            .map(|pl| pl.c)
            .fold(cap, f64::min);
        return best.max(0.0) as f32;
    }
    let eps = 1e-9 * size[0].max(size[1]).max(size[2]);
    let mut best = cap;
    let mut planes = Vec::with_capacity(MAX_PLANES);
    for j in 0..seeds.len() {
        if j > 0 {
            if lower(0, j) >= best {
                continue;
            }
            match plane(0, j) {
                Some(pl) if pl.c < best => {}
                _ => continue,
            }
        }
        planes.clear();
        let mut beyond = false;
        for k in 0..seeds.len() {
            if k == j || lower(j, k) >= best {
                continue;
            }
            let Some(pl) = plane(j, k) else { continue };
            if pl.c <= -best {
                beyond = true;
                break;
            }
            if pl.c.abs() < best {
                planes.push(pl);
            }
        }
        if beyond {
            continue;
        }
        planes.sort_by(|a, b| a.c.abs().total_cmp(&b.c.abs()));
        planes.truncate(MAX_PLANES);
        best = nearest_edge(&planes, best, eps);
    }
    best as f32
}

/// A two-size tile: `count` seeds, a quarter of them large, with weights so a
/// large cell holds about `ratio` times a small one's volume — the weight of a
/// seed is the square of the radius of the bubble it stands for. Large seeds
/// are placed first, each new seed the furthest, net of both radii, from the
/// rest of a few dozen tries.
pub(crate) fn bimodal_tile(
    count: usize,
    ratio: f64,
    cell: Vector3,
    seed: u32,
) -> (Vec<[f64; 3]>, Vec<f64>) {
    let size = [cell.x as f64, cell.y as f64, cell.z as f64];
    let count = count.max(2);
    let large = ((count as f64 * 0.25).round() as usize).clamp(1, count - 1);
    let ratio = ratio.max(1.0);
    // Radii in world units, from the tile's volume shared out by size.
    let volume = size[0] * size[1] * size[2];
    let small_r = (0.7 * volume / ((count - large) as f64 + large as f64 * ratio) * 3.0
        / (4.0 * std::f64::consts::PI))
        .cbrt();
    let large_r = small_r * ratio.cbrt();
    let mut sites: Vec<[f64; 3]> = Vec::with_capacity(count);
    let mut radii: Vec<f64> = Vec::with_capacity(count);
    for i in 0..count {
        let r = if i < large { large_r } else { small_r };
        let mut best = ([0.0; 3], f64::NEG_INFINITY);
        for t in 0..20 * i + 1 {
            let h = hash_unit(i as i32, t as i32, 0xb1d0, seed);
            let q = [h[0] as f64, h[1] as f64, h[2] as f64];
            let room = sites
                .iter()
                .zip(&radii)
                .map(|(&s, &rs)| tile_distance_sq(s, q, size).sqrt() - rs - r)
                .fold(f64::INFINITY, f64::min);
            if room > best.1 {
                best = (q, room);
            }
        }
        sites.push(best.0);
        radii.push(r);
    }
    // Sphere radii are only a first guess at the weights that give the
    // volumes asked for: measure the cells on a grid of the tile and grow the
    // large ones by the cube root of the shortfall until they match.
    let mut grow = 1.0f64;
    let weigh = |grow: f64| -> Vec<f64> {
        radii
            .iter()
            .enumerate()
            .map(|(i, r)| if i < large { (r * grow).powi(2) } else { r * r })
            .collect()
    };
    for _ in 0..8 {
        let weights = weigh(grow);
        const GRID: usize = 24;
        let mut volume = vec![0usize; count];
        for k in 0..GRID {
            for j in 0..GRID {
                for i in 0..GRID {
                    let q = [i, j, k].map(|c| (c as f64 + 0.5) / GRID as f64);
                    let mut owner = (0, f64::INFINITY);
                    for (n, (&s, &w)) in sites.iter().zip(&weights).enumerate() {
                        let power = tile_distance_sq(s, q, size) - w;
                        if power < owner.1 {
                            owner = (n, power);
                        }
                    }
                    volume[owner.0] += 1;
                }
            }
        }
        let mean = |v: &[usize]| (v.iter().sum::<usize>() as f64 / v.len() as f64).max(1.0);
        let measured = mean(&volume[..large]) / mean(&volume[large..]);
        grow *= (ratio / measured).cbrt().clamp(0.8, 1.25);
    }
    (sites, weigh(grow))
}

/// Mitchell's best-candidate sampling on the periodic tile.
fn best_candidate(count: usize, cell: Vector3, seed: u32) -> Vec<[f64; 3]> {
    let size = [cell.x as f64, cell.y as f64, cell.z as f64];
    let mut out: Vec<[f64; 3]> = Vec::with_capacity(count);
    for i in 0..count {
        let tries = 20 * i + 1;
        let mut best = ([0.0; 3], -1.0f64);
        for t in 0..tries {
            let h = hash_unit(i as i32, t as i32, 0x9015, seed);
            let q = [h[0] as f64, h[1] as f64, h[2] as f64];
            let gap = out
                .iter()
                .map(|&p| tile_distance_sq(p, q, size))
                .fold(f64::INFINITY, f64::min);
            if gap > best.1 {
                best = (q, gap);
            }
        }
        out.push(best.0);
    }
    out
}

/// Lloyd's relaxation on the periodic tile: each seed moved to the centroid of
/// its cell, the cells measured on a 32³ grid of the tile.
fn lloyd(mut seeds: Vec<[f64; 3]>, cell: Vector3, rounds: usize) -> Vec<[f64; 3]> {
    const GRID: usize = 32;
    let size = [cell.x as f64, cell.y as f64, cell.z as f64];
    for _ in 0..rounds {
        let mut sum = vec![[0.0f64; 3]; seeds.len()];
        let mut count = vec![0usize; seeds.len()];
        for k in 0..GRID {
            for j in 0..GRID {
                for i in 0..GRID {
                    let q = [i, j, k].map(|c| (c as f64 + 0.5) / GRID as f64);
                    let (mut near, mut best) = (0, f64::INFINITY);
                    for (n, &s) in seeds.iter().enumerate() {
                        let d = tile_distance_sq(s, q, size);
                        if d < best {
                            (near, best) = (n, d);
                        }
                    }
                    // Displacement to the nearest image, so a cell that wraps
                    // round the tile has its centroid in one piece.
                    for a in 0..3 {
                        let d = q[a] - seeds[near][a];
                        sum[near][a] += d - d.round();
                    }
                    count[near] += 1;
                }
            }
        }
        for (n, s) in seeds.iter_mut().enumerate() {
            if count[n] > 0 {
                for a in 0..3 {
                    let moved = s[a] + sum[n][a] / count[n] as f64;
                    s[a] = moved - moved.floor();
                }
            }
        }
    }
    seeds
}

/// Three values in `[0, 1)` from an integer key — a hash, not a sequence, so
/// the same cell always seeds the same point however the samples are ordered.
#[inline(always)]
fn hash_unit(i: i32, j: i32, k: i32, seed: u32) -> [f32; 3] {
    let mut h = (i as i64 as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (j as i64 as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
        ^ (k as i64 as u64).wrapping_mul(0x94d0_49bb_1331_11eb)
        ^ (seed as u64).wrapping_mul(0xd6e8_feb8_6659_fd93);
    let mut out = [0.0f32; 3];
    for axis in &mut out {
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        *axis = (h >> 40) as f32 / (1u32 << 24) as f32;
    }
    out
}

/// The most seeds one sample collects.
///
/// Only seeds within `d₀ + 2·cap` of the point can shape the skeleton within
/// `cap` of it (see [`voronoi_distance`]): six or seven on average for a random
/// foam at a fifth density, a few times that for the 24-seed C15 layout. Past
/// the bound the nearest are kept and the rest — the ones least able to matter
/// — dropped. Kept small on purpose: the array is cleared for every sample.
const MAX_SEEDS: usize = 64;

/// The most bisector planes one cell keeps near a sample.
const MAX_PLANES: usize = 48;

/// Cells searched each way along an axis. Seeds stay inside their own cell,
/// so a cell `r` away is at least `r − 1` cells away.
const MAX_RING: i32 = 4;

/// Distance from the sample (the origin of the seeds' frame) to cell `j`
/// moved in by `inset` on every face, up to `limit`; zero inside it.
///
/// A convex polytope's nearest point is on a face, an edge or a corner. Only
/// the faces the sample is outside of can hold it; an edge is the segment the
/// other faces leave of two faces' line, and a corner is an end of one. As in
/// the edge search, a face further than `limit` either way cannot matter
/// inside that radius — beyond it on the inside it contains the whole ball,
/// on the outside it excludes it.
fn shrunk_cell_distance(
    seeds: &[([f64; 3], f64)],
    j: usize,
    inset: f64,
    limit: f64,
    eps: f64,
) -> f64 {
    let (vj, dj) = seeds[j];
    let mut planes = [Plane {
        n: [0.0; 3],
        c: 0.0,
    }; MAX_PLANES];
    let mut near = 0usize;
    let mut inside = true;
    for (k, &(vk, dk)) in seeds.iter().enumerate() {
        if k == j {
            continue;
        }
        if (dk - dj) * 0.5 - inset >= limit {
            break;
        }
        let Some(mut plane) = bisector(vj, dj, vk, dk) else {
            continue;
        };
        plane.c -= inset;
        if plane.c <= -limit {
            return limit;
        }
        if plane.c < 0.0 {
            inside = false;
        }
        if plane.c.abs() < limit && near < MAX_PLANES {
            planes[near] = plane;
            near += 1;
        }
    }
    if inside {
        return 0.0;
    }
    outside_distance(&planes[..near], limit, eps)
}

/// Distance from the origin, known to be outside it, to the convex cell the
/// `planes` bound (`n · x ≤ c`), up to `limit` — given every plane within
/// `limit` of the origin either way.
fn outside_distance(planes: &[Plane], limit: f64, eps: f64) -> f64 {
    let mut best = limit;
    // Faces: the foot of the perpendicular, if no other face cuts it off.
    for (a, pa) in planes.iter().enumerate() {
        if pa.c >= 0.0 || -pa.c >= best {
            continue;
        }
        let foot = pa.n.map(|x| x * pa.c);
        if planes
            .iter()
            .enumerate()
            .all(|(m, pm)| m == a || dot(pm.n, foot) <= pm.c + eps)
        {
            best = best.min(-pa.c);
        }
    }
    // Edges and their ends.
    nearest_edge(planes, best, eps)
}

/// The nearest point to the origin of any edge of the cell `planes` bound —
/// each pair's line, clipped to what the other planes leave of it — or `best`
/// if none is nearer. A corner is a clipped end.
fn nearest_edge(planes: &[Plane], mut best: f64, eps: f64) -> f64 {
    for b in 1..planes.len() {
        for a in 0..b {
            let (pa, pb) = (planes[a], planes[b]);
            let cos = dot(pa.n, pb.n);
            let det = 1.0 - cos * cos;
            if det < 1e-12 {
                continue;
            }
            let alpha = (pa.c - pb.c * cos) / det;
            let beta = (pb.c - pa.c * cos) / det;
            if alpha * pa.c + beta * pb.c >= best * best {
                continue;
            }
            let x0 = [
                alpha * pa.n[0] + beta * pb.n[0],
                alpha * pa.n[1] + beta * pb.n[1],
                alpha * pa.n[2] + beta * pb.n[2],
            ];
            let inv = 1.0 / det.sqrt();
            let dir = [
                (pa.n[1] * pb.n[2] - pa.n[2] * pb.n[1]) * inv,
                (pa.n[2] * pb.n[0] - pa.n[0] * pb.n[2]) * inv,
                (pa.n[0] * pb.n[1] - pa.n[1] * pb.n[0]) * inv,
            ];
            let (mut lo, mut hi) = (f64::NEG_INFINITY, f64::INFINITY);
            let mut empty = false;
            for (m, pm) in planes.iter().enumerate() {
                if m == a || m == b {
                    continue;
                }
                let along = dot(pm.n, dir);
                let room = pm.c - dot(pm.n, x0);
                if along.abs() < 1e-12 {
                    if room < -eps {
                        empty = true;
                        break;
                    }
                } else if along > 0.0 {
                    hi = hi.min((room + eps) / along);
                } else {
                    lo = lo.max((room + eps) / along);
                }
                if lo > hi {
                    empty = true;
                    break;
                }
            }
            if empty {
                continue;
            }
            let t = 0.0f64.clamp(lo, hi);
            let x = [x0[0] + t * dir[0], x0[1] + t * dir[1], x0[2] + t * dir[2]];
            best = best.min(dot(x, x).sqrt());
        }
    }
    best
}

/// The seeds near a sample, nearest first: each the vector from the sample to
/// it and its length.
struct Gather {
    seeds: [([f64; 3], f64); MAX_SEEDS],
    count: usize,
    /// Seeds further than this cannot matter: `d₀ + reach`.
    limit: f64,
    reach: f64,
}

impl Gather {
    #[inline(always)]
    fn push(&mut self, v: [f64; 3], d: f64) {
        if d > self.limit {
            return;
        }
        // In order; when full, the furthest falls off.
        let mut at = self.count.min(MAX_SEEDS - 1);
        if self.count == MAX_SEEDS && d >= self.seeds[at].1 {
            return;
        }
        while at > 0 && self.seeds[at - 1].1 > d {
            self.seeds[at] = self.seeds[at - 1];
            at -= 1;
        }
        self.seeds[at] = (v, d);
        self.count = (self.count + 1).min(MAX_SEEDS);
        if at == 0 {
            self.limit = d + self.reach;
        }
    }
}

/// What [`voronoi_distance`] needs to place one cell's seed, in one place so
/// the per-cell step inlines.
/// Every seed of a block of cells, worked out once.
///
/// A sample looks at a few dozen cells and a lattice has a few hundred, so a
/// build re-derives each seed thousands of times; the table derives it once,
/// by the same expression, and a sample reads it back. Cells outside the block
/// are hashed as before, so the table can only ever be a shortcut.
pub(crate) struct SeedTable {
    lo: [i32; 3],
    dims: [i32; 3],
    /// Seed position in cell units, `c + ½ + jitter·(h − ½)`, per cell.
    at: Vec<[f64; 3]>,
}

impl SeedTable {
    /// The seeds of cells `lo..=hi` on each axis, or `None` for a block too big
    /// to be worth holding.
    pub(crate) fn new(lo: [i32; 3], hi: [i32; 3], jitter: f32, seed: u32) -> Option<SeedTable> {
        let dims = [0, 1, 2].map(|a| hi[a] - lo[a] + 1);
        if dims.iter().any(|&d| d <= 0) {
            return None;
        }
        let count = dims.iter().map(|&d| d as usize).product::<usize>();
        if count > 1 << 22 {
            return None;
        }
        let jitter = jitter.clamp(0.0, 1.0) as f64;
        let mut at = Vec::with_capacity(count);
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    let c = [lo[0] + x, lo[1] + y, lo[2] + z];
                    at.push(seed_position(c, jitter, seed));
                }
            }
        }
        Some(SeedTable { lo, dims, at })
    }

    #[inline(always)]
    fn get(&self, c: [i32; 3]) -> Option<[f64; 3]> {
        let r = [c[0] - self.lo[0], c[1] - self.lo[1], c[2] - self.lo[2]];
        if (0..3).any(|a| r[a] < 0 || r[a] >= self.dims[a]) {
            return None;
        }
        Some(self.at[((r[2] * self.dims[1] + r[1]) * self.dims[0] + r[0]) as usize])
    }
}

/// Where cell `c`'s seed sits, in cell units.
#[inline(always)]
fn seed_position(c: [i32; 3], jitter: f64, seed: u32) -> [f64; 3] {
    let h = hash_unit(c[0], c[1], c[2], seed);
    [
        c[0] as f64 + 0.5 + jitter * (h[0] as f64 - 0.5),
        c[1] as f64 + 0.5 + jitter * (h[1] as f64 - 0.5),
        c[2] as f64 + 0.5 + jitter * (h[2] as f64 - 0.5),
    ]
}

struct Lookup<'a> {
    table: Option<&'a SeedTable>,
    /// The layout's seeds per cell, or empty for one hashed seed.
    sites: &'a [[f64; 3]],
    gap: &'a [[f64; 2 * MAX_RING as usize + 1]; 3],
    base: [i32; 3],
    u: [f64; 3],
    size: [f64; 3],
    jitter: f64,
    seed: u32,
}

impl Lookup<'_> {
    /// Offer the seed of the cell at this offset, unless its box is out of
    /// reach already.
    #[inline(always)]
    fn consider(&self, found: &mut Gather, ox: i32, oy: i32, oz: i32) {
        let lower = self.gap[0][(ox + MAX_RING) as usize]
            + self.gap[1][(oy + MAX_RING) as usize]
            + self.gap[2][(oz + MAX_RING) as usize];
        if lower > found.limit * found.limit {
            return;
        }
        let c = [self.base[0] + ox, self.base[1] + oy, self.base[2] + oz];
        if !self.sites.is_empty() {
            for site in self.sites {
                let v = [
                    (c[0] as f64 + site[0] - self.u[0]) * self.size[0],
                    (c[1] as f64 + site[1] - self.u[1]) * self.size[1],
                    (c[2] as f64 + site[2] - self.u[2]) * self.size[2],
                ];
                found.push(v, dot(v, v).sqrt());
            }
            return;
        }
        let s = match self.table.and_then(|t| t.get(c)) {
            Some(s) => s,
            None => seed_position(c, self.jitter, self.seed),
        };
        let v = [
            (s[0] - self.u[0]) * self.size[0],
            (s[1] - self.u[1]) * self.size[1],
            (s[2] - self.u[2]) * self.size[2],
        ];
        found.push(v, dot(v, v).sqrt());
    }
}

/// One face of a Voronoi cell, in a frame centred on the sample: the cell is
/// where `n · x ≤ c`, so `c` is the sample's signed distance to the plane,
/// positive on the cell's side.
#[derive(Clone, Copy)]
struct Plane {
    n: [f64; 3],
    c: f64,
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The bisector between the seeds at `vj` and `vk` (both relative to the
/// sample, with lengths `dj`, `dk`), seen from `vj`'s cell. `None` when the
/// seeds coincide.
fn bisector(vj: [f64; 3], dj: f64, vk: [f64; 3], dk: f64) -> Option<Plane> {
    let w = [vk[0] - vj[0], vk[1] - vj[1], vk[2] - vj[2]];
    let l = dot(w, w).sqrt();
    if l < 1e-12 {
        return None;
    }
    // x is nearer vj than vk  ⇔  2x·(vk − vj) < |vk|² − |vj|².
    Some(Plane {
        n: [w[0] / l, w[1] / l, w[2] / l],
        c: (dk * dk - dj * dj) / (2.0 * l),
    })
}

/// Distance from `p` to the nearest Voronoi edge, or face when `wall`, up to
/// `cap`: points further than that from the skeleton report `cap` exactly, as
/// the strut distance does past its cull.
///
/// `floor` is the other end. A point that close to the skeleton is deep inside
/// the strut or wall, where a mesher reads only the sign, so the first member
/// found within `floor` is returned as it is — an upper bound that is itself
/// within `floor`, so on the right side of it. Every answer strictly between
/// the two is exact.
///
/// With [`SeedLayout::Random`], one seed per cell of the integer grid,
/// displaced from the cell centre by `jitter` of a cell — 0 for a regular grid,
/// 1 for a seed that can land anywhere in its own cell. Any other layout puts
/// its own sites in every cell and ignores the jitter.
///
/// # Exact, not implicit
///
/// The usual reading, `(d₂ − d₁)/2` for a face and `(d₃ − d₁)/2` for an edge,
/// is zero on the skeleton but is not a distance. On a fully jittered foam its
/// gradient has a median of 0.75 and a fifth percentile of 0.3, so a wall of
/// "thickness t" comes out anywhere from about t to three times it depending
/// on how the seeds either side happen to lie. This measures the real thing.
///
/// Every face of the cell holding `p` is a bisector plane at signed distance
/// `c = (dₖ² − d₀²) / 2|sₖ − s₀|`, and since `p` is inside a convex cell its
/// distance to the walls is the least `c`. The edges are where two faces
/// meet: for each pair of planes, the segment of their line that the cell's
/// other planes leave, and the point of it nearest `p`. An edge can belong to
/// a neighbouring cell and not to `p`'s, so the neighbours are measured too —
/// which is also what keeps the field continuous across a face.
///
/// Three bounds keep it cheap and keep it exact up to `cap`:
///
/// - A cell `j` lies beyond the bisector between it and `p`'s cell, so it is
///   at least `c₀ⱼ ≥ (dⱼ − d₀)/2` away; cells further than the best distance
///   so far are skipped, and only seeds within `d₀ + 2·cap` are ever looked
///   at — no seed further out can bound a cell that comes within `cap`.
/// - A plane further than the best distance cannot cut a segment inside that
///   radius, so only the near ones clip.
/// - A line lying in a plane is at least that plane's distance away.
pub(crate) fn voronoi_distance(
    p: Vector3,
    origin: Vector3,
    cell: Vector3,
    skeleton: Skeleton,
    jitter: f32,
    seed: u32,
    sites: &[[f64; 3]],
    cap: f32,
    floor: f32,
    table: Option<&SeedTable>,
) -> f32 {
    let floor = floor as f64;
    let cap64 = (cap as f64).max(0.0);
    let wall = skeleton == Skeleton::Walls;
    // A wet foam's bubbles are its cells moved in by the border, so seeds that
    // far further out still shape them.
    let border = match skeleton {
        Skeleton::Wet { border } => (border as f64).max(0.0),
        _ => 0.0,
    };
    let jitter = jitter.clamp(0.0, 1.0) as f64;
    let size = [cell.x as f64, cell.y as f64, cell.z as f64];
    let u = [
        (p.x - origin.x) as f64 / size[0],
        (p.y - origin.y) as f64 / size[1],
        (p.z - origin.z) as f64 / size[2],
    ];
    let base = [
        u[0].floor() as i32,
        u[1].floor() as i32,
        u[2].floor() as i32,
    ];

    // The seeds that matter, nearest first, each as the vector from `p` to it
    // and its length — measured in world units, so an anisotropic cell
    // stretches the foam rather than distorting which seed is nearest.
    // How far the nearest point of each cell's seed box is from `p` along
    // each axis, for offsets −MAX_RING..=MAX_RING: seeds sit within
    // `jitter / 2` of their cell's centre. Squared, ready to add.
    const SPAN: usize = 2 * MAX_RING as usize + 1;
    let mut gap = [[0.0f64; SPAN]; 3];
    for (axis, row) in gap.iter_mut().enumerate() {
        for (o, g) in row.iter_mut().enumerate() {
            let centre = (base[axis] + o as i32 - MAX_RING) as f64 + 0.5;
            // A lattice layout's seeds can be anywhere in their cell.
            let reach = if sites.is_empty() { 0.5 * jitter } else { 0.5 };
            let d = (centre - reach - u[axis])
                .max(u[axis] - centre - reach)
                .max(0.0)
                * size[axis];
            *g = d * d;
        }
    }

    let mut found = Gather {
        seeds: [([0.0f64; 3], 0.0f64); MAX_SEEDS],
        count: 0,
        limit: f64::INFINITY,
        reach: 2.0 * (cap64 + border),
    };
    let at = Lookup {
        table,
        sites,
        gap: &gap,
        base,
        u,
        size,
        jitter,
        seed,
    };
    let consider = |found: &mut Gather, ox: i32, oy: i32, oz: i32| at.consider(found, ox, oy, oz);
    // The point's own cell first, which settles a nearest seed and so the
    // radius anything else has to fall within; then its neighbours; then only
    // the offsets that radius still reaches on each axis — usually none.
    consider(&mut found, 0, 0, 0);
    for oz in -1..=1 {
        for oy in -1..=1 {
            for ox in -1..=1 {
                if (ox, oy, oz) != (0, 0, 0) {
                    consider(&mut found, ox, oy, oz);
                }
            }
        }
    }
    let reach = |axis: usize, limit: f64| -> (i32, i32) {
        let within = |o: i32| gap[axis][(o + MAX_RING) as usize] <= limit * limit;
        let (mut lo, mut hi) = (-1, 1);
        while lo > -MAX_RING && within(lo - 1) {
            lo -= 1;
        }
        while hi < MAX_RING && within(hi + 1) {
            hi += 1;
        }
        (lo, hi)
    };
    let (sx, sy, sz) = (
        reach(0, found.limit),
        reach(1, found.limit),
        reach(2, found.limit),
    );
    if (sx, sy, sz) != ((-1, 1), (-1, 1), (-1, 1)) {
        for oz in sz.0..=sz.1 {
            for oy in sy.0..=sy.1 {
                for ox in sx.0..=sx.1 {
                    if ox.abs() > 1 || oy.abs() > 1 || oz.abs() > 1 {
                        consider(&mut found, ox, oy, oz);
                    }
                }
            }
        }
    }
    let mut count = found.count;
    while count > 0 && found.seeds[count - 1].1 > found.limit {
        count -= 1;
    }
    let seeds = &found.seeds[..count];
    if seeds.is_empty() {
        return cap;
    }
    let (v0, d0) = seeds[0];

    if let Skeleton::Wet { .. } = skeleton {
        let eps = 1e-9 * size[0].max(size[1]).max(size[2]);
        // Bubble j lies inside cell j, which lies beyond its bisector with
        // p's cell: no nearer than that bisector, and so no nearer than
        // (dⱼ − d₀)/2.
        let mut best = cap64;
        for (j, &(vj, dj)) in seeds.iter().enumerate() {
            if (dj - d0) * 0.5 >= best {
                break;
            }
            if j > 0 {
                match bisector(v0, d0, vj, dj) {
                    Some(plane) if plane.c < best => {}
                    _ => continue,
                }
            }
            let e = shrunk_cell_distance(seeds, j, border, best + border, eps) - border;
            best = best.min(e);
        }
        return best as f32;
    }

    // The implicit reading is a lower bound on both: every bisector is at
    // least (dₖ − d₀)/2 away, and a point within δ of an edge has three seeds
    // within 2δ of each other. Out in the void it already clears the cap, and
    // that is most samples.
    let rank = if wall { 1 } else { 2 };
    if seeds.len() <= rank || (seeds[rank].1 - d0) * 0.5 >= cap64 {
        return cap;
    }

    if wall {
        let mut best = cap64;
        for &(vk, dk) in &seeds[1..] {
            if (dk - d0) * 0.5 >= best {
                break;
            }
            if let Some(plane) = bisector(v0, d0, vk, dk) {
                best = best.min(plane.c);
                if best <= floor {
                    break;
                }
            }
        }
        return best.max(0.0) as f32;
    }

    // An edge point is in three cells' closures, so at least two of them are
    // not p's own, and each cell j lies beyond its bisector with p's: the
    // second-nearest of p's faces bounds every edge from below. Unlike the
    // implicit reading it is tight, and it settles most of the samples left.
    let (mut first, mut second) = (f64::INFINITY, f64::INFINITY);
    for &(vk, dk) in &seeds[1..] {
        if (dk - d0) * 0.5 >= second.min(cap64) {
            break;
        }
        let w = [vk[0] - v0[0], vk[1] - v0[1], vk[2] - v0[2]];
        let c = (dk * dk - d0 * d0) / (2.0 * dot(w, w).sqrt().max(1e-12));
        if c < first {
            second = first;
            first = c;
        } else if c < second {
            second = c;
        }
    }
    if second >= cap64 {
        return cap;
    }

    // Edges. Tolerances scale with the cell, for the planes a regular grid
    // makes that contain an edge rather than cross it.
    let eps = 1e-9 * size[0].max(size[1]).max(size[2]);
    let mut best = cap64;
    let mut planes = [Plane {
        n: [0.0; 3],
        c: 0.0,
    }; MAX_PLANES];
    for (j, &(vj, dj)) in seeds.iter().enumerate() {
        if (dj - d0) * 0.5 >= best {
            break;
        }
        if j > 0 {
            match bisector(v0, d0, vj, dj) {
                Some(plane) if plane.c < best => {}
                _ => continue,
            }
        }
        // Cell j's faces within `best` of p, nearest first.
        let mut near = 0usize;
        let mut beyond = false;
        for (k, &(vk, dk)) in seeds.iter().enumerate() {
            if k == j {
                continue;
            }
            // Further seeds only make further planes: c ≥ (dₖ − dⱼ)/2.
            if (dk - dj) * 0.5 >= best {
                break;
            }
            let Some(plane) = bisector(vj, dj, vk, dk) else {
                continue;
            };
            if plane.c <= -best {
                // The whole ball of radius `best` is outside cell j.
                beyond = true;
                break;
            }
            if plane.c.abs() >= best {
                continue;
            }
            let mut at = near.min(MAX_PLANES - 1);
            if near == MAX_PLANES && plane.c.abs() >= planes[at].c.abs() {
                continue;
            }
            while at > 0 && planes[at - 1].c.abs() > plane.c.abs() {
                planes[at] = planes[at - 1];
                at -= 1;
            }
            planes[at] = plane;
            near = (near + 1).min(MAX_PLANES);
        }
        if beyond {
            continue;
        }
        let planes = &planes[..near];
        for b in 1..planes.len() {
            let pb = planes[b];
            if pb.c.abs() >= best {
                break;
            }
            for a in 0..b {
                let pa = planes[a];
                let cos = dot(pa.n, pb.n);
                let det = 1.0 - cos * cos;
                if det < 1e-12 {
                    continue;
                }
                // The point of the line nearest p, and the line's direction.
                let alpha = (pa.c - pb.c * cos) / det;
                let beta = (pb.c - pa.c * cos) / det;
                if alpha * pa.c + beta * pb.c >= best * best {
                    continue;
                }
                let x0 = [
                    alpha * pa.n[0] + beta * pb.n[0],
                    alpha * pa.n[1] + beta * pb.n[1],
                    alpha * pa.n[2] + beta * pb.n[2],
                ];
                let inv = 1.0 / det.sqrt();
                let dir = [
                    (pa.n[1] * pb.n[2] - pa.n[2] * pb.n[1]) * inv,
                    (pa.n[2] * pb.n[0] - pa.n[0] * pb.n[2]) * inv,
                    (pa.n[0] * pb.n[1] - pa.n[1] * pb.n[0]) * inv,
                ];
                // What the cell's other faces leave of the line.
                let (mut lo, mut hi) = (f64::NEG_INFINITY, f64::INFINITY);
                let mut empty = false;
                for (k, pk) in planes.iter().enumerate() {
                    if k == a || k == b {
                        continue;
                    }
                    let along = dot(pk.n, dir);
                    let room = pk.c - dot(pk.n, x0);
                    if along.abs() < 1e-12 {
                        if room < -eps {
                            empty = true;
                            break;
                        }
                    } else if along > 0.0 {
                        hi = hi.min((room + eps) / along);
                    } else {
                        lo = lo.max((room + eps) / along);
                    }
                    if lo > hi {
                        empty = true;
                        break;
                    }
                }
                if empty {
                    continue;
                }
                let t = 0.0f64.clamp(lo, hi);
                let x = [x0[0] + t * dir[0], x0[1] + t * dir[1], x0[2] + t * dir[2]];
                best = best.min(dot(x, x).sqrt());
                if best <= floor {
                    return best as f32;
                }
            }
        }
    }
    best as f32
}

/// Fibre half-length along its axis, and its half-width across, in cells —
/// the proportions of the F13LD grain explorer's defaults (an aspect of four)
/// at one fibre per cell.
const FIBRE_LENGTH: f64 = 0.35;
const FIBRE_WIDTH: f64 = 0.087;
/// Where the fibres' sum is cut into solid.
const FIBRE_LEVEL: f64 = 0.3;
/// Kernel terms below `e^−16` (about 10⁻⁷) are left out: invisible against
/// the level, and what bounds how many cells a sample has to look at.
const FIBRE_CUT: f64 = 16.0;

/// The hyperuniform fibre field and its gradient at `p`: positive in a fibre.
///
/// One kernel per cell, at a seed jittered by 0.9 of a cell — the jittered
/// grid, the textbook hyperuniform point pattern — pointing in a direction
/// hashed uniformly over the sphere. Each kernel is `exp(−R)` with `R` the
/// squared distance from its axis scaled by length along it and width
/// across, so a fibre is a prolate blob that melts into its neighbours where
/// they cross.
pub(crate) fn hyperuniform(
    p: Vector3,
    origin: Vector3,
    cell: Vector3,
    seed: u32,
) -> (f32, Vector3) {
    let size = [cell.x as f64, cell.y as f64, cell.z as f64];
    let u = [
        (p.x - origin.x) as f64 / size[0],
        (p.y - origin.y) as f64 / size[1],
        (p.z - origin.z) as f64 / size[2],
    ];
    let base = [
        u[0].floor() as i32,
        u[1].floor() as i32,
        u[2].floor() as i32,
    ];
    // A kernel's reach is √CUT lengths along its axis; its seed is within
    // 0.45 of its cell's centre.
    let reach = FIBRE_CUT.sqrt() * FIBRE_LENGTH + 0.45;
    let ring = (reach + 0.5).ceil() as i32;
    let (mut value, mut grad) = (0.0f64, [0.0f64; 3]);
    let smallest = size[0].min(size[1]).min(size[2]);
    for oz in -ring..=ring {
        for oy in -ring..=ring {
            for ox in -ring..=ring {
                let c = [base[0] + ox, base[1] + oy, base[2] + oz];
                // The cell's centre is within `reach` cells of p, or no kernel
                // seeded in it can touch p.
                let near = (0..3)
                    .map(|a| ((c[a] as f64 + 0.5 - u[a]).abs() - 0.5).max(0.0))
                    .map(|d| d * d)
                    .sum::<f64>();
                if near > reach * reach {
                    continue;
                }
                let h = hash_unit(c[0], c[1], c[2], seed ^ 0x48_55);
                let g = hash_unit(c[0], c[1], c[2], seed ^ 0x48_56);
                // Displacement p − seed in world units.
                let d = [0, 1, 2]
                    .map(|a| (u[a] - (c[a] as f64 + 0.5 + 0.9 * (h[a] as f64 - 0.5))) * size[a]);
                let t = from_polar(Vector3::new(0.0, 0.0, 1.0), 2.0 * g[0] - 1.0, g[1] * TAU);
                let t = [t.x as f64, t.y as f64, t.z as f64];
                let along = d[0] * t[0] + d[1] * t[1] + d[2] * t[2];
                let across = [
                    d[0] - along * t[0],
                    d[1] - along * t[1],
                    d[2] - along * t[2],
                ];
                let (a, b) = (FIBRE_LENGTH * smallest, FIBRE_WIDTH * smallest);
                let r = (along / a).powi(2)
                    + (across[0] * across[0] + across[1] * across[1] + across[2] * across[2])
                        / (b * b);
                if r >= FIBRE_CUT {
                    continue;
                }
                let k = (-r).exp();
                value += k;
                // ∇R = 2·along/a²·t + 2·across/b².
                for i in 0..3 {
                    grad[i] -= k * (2.0 * along / (a * a) * t[i] + 2.0 * across[i] / (b * b));
                }
            }
        }
    }
    (
        (value - FIBRE_LEVEL) as f32,
        Vector3::new(grad[0] as f32, grad[1] as f32, grad[2] as f32),
    )
}

/// Samples a side of a reaction–diffusion tile.
const TURING_GRID: usize = 40;
/// Steps the pattern is grown for. The seeded spots have met and the
/// concentrations' range stopped changing by about 500; four times that lets
/// the maze itself settle.
const TURING_STEPS: usize = 2000;

/// A settled Gray–Scott pattern on one periodic cell, ready to sample.
pub(crate) struct Turing {
    /// Activator concentration, `TURING_GRID³`, x fastest.
    v: Vec<f32>,
    /// Where it is cut into solid: halfway between its lowest and highest.
    level: f32,
}

/// The pattern for `seed`, simulated once and kept for the process — fitting a
/// density asks for it a few dozen times over.
pub(crate) fn turing(seed: u32) -> std::sync::Arc<Turing> {
    use std::sync::{Arc, Mutex, OnceLock};
    /// The most recent patterns, by seed.
    type Grown = Mutex<Vec<(u32, Arc<Turing>)>>;
    static CACHE: OnceLock<Grown> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    if let Some((_, t)) = cache.lock().unwrap().iter().find(|(s, _)| *s == seed) {
        return t.clone();
    }
    let t = Arc::new(grow_turing(seed));
    let mut held = cache.lock().unwrap();
    if held.len() >= 8 {
        held.remove(0);
    }
    held.push((seed, t.clone()));
    t
}

/// Gray–Scott in the labyrinth regime (feed 0.037, kill 0.06), on a periodic
/// grid with the 7-point Laplacian normalised to neighbour mean minus centre.
///
/// That Laplacian takes a checkerboard to −2× itself, so an explicit unit
/// step multiplies one by `1 − 2D − feed`: past `D ≈ 0.98` it grows without
/// bound. The usual 2D rates (1 and ½, for a 9-point stencil) blow up here
/// within a few hundred steps; 0.8 and 0.4 keep their ratio, which is what
/// sets the pattern, and are stable.
/// Seeded with a few dozen hashed blobs of activator, so the pattern is a pure
/// function of the seed.
fn grow_turing(seed: u32) -> Turing {
    let n = TURING_GRID;
    let at = |i: usize, j: usize, k: usize| (k * n + j) * n + i;
    let (mut u, mut v) = (vec![1.0f32; n * n * n], vec![0.0f32; n * n * n]);
    for b in 0..24 {
        let h = hash_unit(b, 0x7a7, 1, seed);
        let c = h.map(|x| (x * n as f32) as i32);
        for dz in -2i32..=2 {
            for dy in -2i32..=2 {
                for dx in -2i32..=2 {
                    let w = |a: i32, d: i32| (a + d).rem_euclid(n as i32) as usize;
                    let q = at(w(c[0], dx), w(c[1], dy), w(c[2], dz));
                    u[q] = 0.5;
                    v[q] = 0.25;
                }
            }
        }
    }
    let (du, dv, feed, kill) = (0.8f32, 0.4f32, 0.037f32, 0.06f32);
    let (mut u2, mut v2) = (u.clone(), v.clone());
    for _ in 0..TURING_STEPS {
        for k in 0..n {
            let (km, kp) = ((k + n - 1) % n, (k + 1) % n);
            for j in 0..n {
                let (jm, jp) = ((j + n - 1) % n, (j + 1) % n);
                for i in 0..n {
                    let (im, ip) = ((i + n - 1) % n, (i + 1) % n);
                    let q = at(i, j, k);
                    let lap = |f: &[f32]| {
                        (f[at(im, j, k)]
                            + f[at(ip, j, k)]
                            + f[at(i, jm, k)]
                            + f[at(i, jp, k)]
                            + f[at(i, j, km)]
                            + f[at(i, j, kp)])
                            / 6.0
                            - f[q]
                    };
                    let (a, b) = (u[q], v[q]);
                    let react = a * b * b;
                    u2[q] = a + du * lap(&u) - react + feed * (1.0 - a);
                    v2[q] = b + dv * lap(&v) + react - (kill + feed) * b;
                }
            }
        }
        std::mem::swap(&mut u, &mut u2);
        std::mem::swap(&mut v, &mut v2);
    }
    let (lo, hi) = v
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &x| {
            (l.min(x), h.max(x))
        });
    Turing {
        v,
        level: 0.5 * (lo + hi),
    }
}

impl Turing {
    /// Activator minus the level, and its gradient in cell units, at `u` (cell
    /// units, periodic): trilinear between the grid's samples.
    pub(crate) fn sample(&self, u: [f64; 3]) -> (f32, [f32; 3]) {
        let n = TURING_GRID;
        let g = u.map(|x| (x - x.floor()) * n as f64);
        let i0 = g.map(|x| (x.floor() as usize).min(n - 1));
        let f = [0, 1, 2].map(|a| (g[a] - i0[a] as f64) as f32);
        let i1 = i0.map(|i| (i + 1) % n);
        let at = |x: usize, y: usize, z: usize| self.v[(z * n + y) * n + x];
        let c = |ix: usize, iy: usize, iz: usize| {
            at(
                if ix == 0 { i0[0] } else { i1[0] },
                if iy == 0 { i0[1] } else { i1[1] },
                if iz == 0 { i0[2] } else { i1[2] },
            )
        };
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        // Along x first, then y, then z, keeping the differences for the
        // gradient.
        let x00 = lerp(c(0, 0, 0), c(1, 0, 0), f[0]);
        let x10 = lerp(c(0, 1, 0), c(1, 1, 0), f[0]);
        let x01 = lerp(c(0, 0, 1), c(1, 0, 1), f[0]);
        let x11 = lerp(c(0, 1, 1), c(1, 1, 1), f[0]);
        let y0 = lerp(x00, x10, f[1]);
        let y1 = lerp(x01, x11, f[1]);
        let value = lerp(y0, y1, f[2]);
        let dz = y1 - y0;
        let dy = lerp(x10 - x00, x11 - x01, f[2]);
        let dx = {
            let d = |iy: usize, iz: usize| c(1, iy, iz) - c(0, iy, iz);
            lerp(
                lerp(d(0, 0), d(1, 0), f[1]),
                lerp(d(0, 1), d(1, 1), f[1]),
                f[2],
            )
        };
        let scale = n as f32;
        (value - self.level, [dx * scale, dy * scale, dz * scale])
    }
}

/// One plane wave of a spinodal field: a world-space wave vector and a phase.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Wave {
    pub(crate) k: Vector3,
    pub(crate) phase: f32,
}

/// The wave set for a cell type, in world units.
///
/// Built once per lattice and reused for every sample — a few hundred bytes
/// against `WAVES` trigonometric calls a point, which is the whole cost of the
/// field.
pub(crate) fn waves(kind: Stochastic, cell: Vector3, seed: u32) -> Vec<Wave> {
    (0..WAVES)
        .map(|i| {
            let h = hash_unit(i as i32, 0x5eed, 0, seed);
            let n = direction(kind, i, h) * wavenumber(kind, i, seed);
            Wave {
                // Per axis, so an anisotropic cell stretches the field the
                // same way it stretches every other family.
                k: Vector3::new(
                    TAU * n.x / cell.x,
                    TAU * n.y / cell.y,
                    TAU * n.z / cell.z,
                ),
                phase: h[2] * TAU,
            }
        })
        .collect()
}

/// Wavenumber of wave `i` relative to one cycle a cell: 1 for every spinodal
/// class, and for a Gaussian field drawn from a normal of spread 0.45 about
/// it — clipped below at 0.1, since a wave that long is a constant over any
/// part this field would fill.
fn wavenumber(kind: Stochastic, i: usize, seed: u32) -> f32 {
    match kind {
        Stochastic::GaussianField => {
            // Box–Muller from two hashed uniforms, nudged off zero for the log.
            let h = hash_unit(i as i32, 0x6a55, 1, seed);
            let normal = (-2.0 * h[0].max(1e-7).ln()).sqrt() * (TAU * h[1]).cos();
            (1.0 + 0.45 * normal).max(0.1)
        }
        _ => 1.0,
    }
}

/// A wave direction for the class: uniform over the sphere, or inside a cone.
fn direction(kind: Stochastic, i: usize, h: [f32; 3]) -> Vector3 {
    let azimuth = h[1] * TAU;
    match kind {
        // Uniform in cos θ, not in θ — otherwise the poles get crowded and the
        // "isotropic" field is stiffer along z than across it.
        Stochastic::Spinodal | Stochastic::GaussianField => {
            from_polar(Vector3::new(0.0, 0.0, 1.0), 2.0 * h[0] - 1.0, azimuth)
        }
        Stochastic::SpinodalLamellar => from_polar(
            Vector3::new(0.0, 0.0, 1.0),
            1.0 - h[0] * (1.0 - CONE.cos()),
            azimuth,
        ),
        // A band about the equator rather than a cap about the pole.
        Stochastic::SpinodalColumnar => from_polar(
            Vector3::new(0.0, 0.0, 1.0),
            (2.0 * h[0] - 1.0) * CONE.sin(),
            azimuth,
        ),
        Stochastic::SpinodalCubic => {
            let axis = match i % 3 {
                0 => Vector3::new(1.0, 0.0, 0.0),
                1 => Vector3::new(0.0, 1.0, 0.0),
                _ => Vector3::new(0.0, 0.0, 1.0),
            };
            from_polar(axis, 1.0 - h[0] * (1.0 - CONE.cos()), azimuth)
        }
        // Not a field cell; any direction will do and none is ever asked for.
        _ => Vector3::new(0.0, 0.0, 1.0),
    }
}

/// A unit vector at `cos_polar` from `axis`, turned `azimuth` about it.
fn from_polar(axis: Vector3, cos_polar: f32, azimuth: f32) -> Vector3 {
    let cos_polar = cos_polar.clamp(-1.0, 1.0);
    let sin_polar = (1.0 - cos_polar * cos_polar).max(0.0).sqrt();
    // Any perpendicular will do — the azimuth is random about it.
    let seed = if axis.x.abs() < 0.9 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let u = axis.cross(seed).normalize();
    let v = axis.cross(u);
    let (s, c) = azimuth.sin_cos();
    axis * cos_polar + u * (sin_polar * c) + v * (sin_polar * s)
}

/// The field and its gradient at a world point.
///
/// Scaled to unit variance so that a level of `l` cuts the same fraction of
/// space whatever the wave count is: the sum of `n` unit cosines with random
/// phases has variance `n / 2`.
pub(crate) fn value_and_gradient(waves: &[Wave], p: Vector3) -> (f32, Vector3) {
    let mut value = 0.0f32;
    let mut grad = Vector3::ZERO;
    for w in waves {
        let (s, c) = (w.k.dot(p) + w.phase).sin_cos();
        value += c;
        grad = grad - w.k * s;
    }
    let scale = (2.0 / waves.len().max(1) as f32).sqrt();
    (value * scale, grad * scale)
}

/// [`value_and_gradient`] from each wave's `(sin, cos)` at the point, for a
/// caller that has them already — a grid has them for a few multiplies each,
/// where working them out per point is most of a sample.
pub(crate) fn value_and_gradient_sc(waves: &[Wave], sc: &[(f32, f32)]) -> (f32, Vector3) {
    let mut value = 0.0f32;
    let mut grad = Vector3::ZERO;
    for (w, &(s, c)) in waves.iter().zip(sc) {
        value += c;
        grad = grad - w.k * s;
    }
    let scale = (2.0 / waves.len().max(1) as f32).sqrt();
    (value * scale, grad * scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip() {
        for s in Stochastic::ALL {
            assert_eq!(Stochastic::from_name(s.name()), Some(s));
        }
        assert_eq!(Stochastic::from_name("nope"), None);
    }

    #[test]
    fn voronoi_edges_are_further_apart_than_faces() {
        // Every point is at least as far from an edge as from a face: the
        // edges are a subset of the faces' boundaries.
        let cell = Vector3::new(1.0, 1.0, 1.0);
        for i in 0..200 {
            let h = hash_unit(i, 7, 11, 3);
            let p = Vector3::new(h[0] * 3.0, h[1] * 3.0, h[2] * 3.0);
            let face = voronoi_distance(
                p,
                Vector3::ZERO,
                cell,
                Skeleton::Walls,
                0.8,
                1,
                &[],
                2.0,
                0.0,
                None,
            );
            let edge = voronoi_distance(
                p,
                Vector3::ZERO,
                cell,
                Skeleton::Edges,
                0.8,
                1,
                &[],
                2.0,
                0.0,
                None,
            );
            assert!(edge >= face - 1e-5, "edge {edge} < face {face}");
        }
    }

    /// The distance to the nearest integer, in cells.
    fn off_grid(x: f32) -> f32 {
        (x - x.round()).abs()
    }

    #[test]
    fn a_regular_foam_is_measured_exactly() {
        // Seeds at the centres of a box grid make box cells: the faces are the
        // grid planes and the edges the grid lines, so both distances are
        // known in closed form. The implicit (d₃ − d₁)/2 reading gets the edge
        // one wrong: 0.037 at (0.05, 0.05, 0.4) of a unit grid, where the edge
        // is 0.071 away.
        let cell = Vector3::new(2.0, 1.0, 1.5);
        for i in 0..300 {
            let h = hash_unit(i, 3, 5, 9);
            let p = Vector3::new(h[0] * 6.0 - 3.0, h[1] * 3.0 - 1.5, h[2] * 4.5 - 2.25);
            let f = [
                off_grid(p.x / cell.x) * cell.x,
                off_grid(p.y / cell.y) * cell.y,
                off_grid(p.z / cell.z) * cell.z,
            ];
            let face = f[0].min(f[1]).min(f[2]);
            let edge = (f[0].hypot(f[1]))
                .min(f[1].hypot(f[2]))
                .min(f[0].hypot(f[2]));
            let got_face = voronoi_distance(
                p,
                Vector3::ZERO,
                cell,
                Skeleton::Walls,
                0.0,
                1,
                &[],
                4.0,
                0.0,
                None,
            );
            let got_edge = voronoi_distance(
                p,
                Vector3::ZERO,
                cell,
                Skeleton::Edges,
                0.0,
                1,
                &[],
                4.0,
                0.0,
                None,
            );
            assert!(
                (got_face - face).abs() < 1e-5,
                "{p:?}: face {got_face} vs {face}"
            );
            assert!(
                (got_edge - edge).abs() < 1e-5,
                "{p:?}: edge {got_edge} vs {edge}"
            );
        }
    }

    #[test]
    fn a_jittered_foam_field_is_a_distance() {
        // A distance has a unit gradient wherever it is smooth, and stepping
        // down that gradient by the value lands on the skeleton. The implicit
        // reading fails both: 2–3% of these samples have a unit gradient.
        let cell = Vector3::new(1.0, 1.0, 1.0);
        let cap = 2.0;
        for wall in [false, true] {
            let field = |p: Vector3| {
                voronoi_distance(
                    p,
                    Vector3::ZERO,
                    cell,
                    if wall {
                        Skeleton::Walls
                    } else {
                        Skeleton::Edges
                    },
                    1.0,
                    7,
                    &[],
                    cap,
                    0.0,
                    None,
                )
            };
            let (mut tried, mut unit, mut landed) = (0, 0, 0);
            for i in 0..400 {
                let h = hash_unit(i, 1, 2, 3);
                let p = Vector3::new(h[0] * 4.0, h[1] * 4.0, h[2] * 4.0);
                let d = field(p);
                if d > 0.3 {
                    continue;
                }
                tried += 1;
                let e = 1e-3;
                let g = Vector3::new(
                    field(p + Vector3::new(e, 0.0, 0.0)) - field(p - Vector3::new(e, 0.0, 0.0)),
                    field(p + Vector3::new(0.0, e, 0.0)) - field(p - Vector3::new(0.0, e, 0.0)),
                    field(p + Vector3::new(0.0, 0.0, e)) - field(p - Vector3::new(0.0, 0.0, e)),
                ) * (0.5 / e);
                if (g.length() - 1.0).abs() < 1e-2 {
                    unit += 1;
                }
                if field(p - g.normalize() * d) < 2e-3 {
                    landed += 1;
                }
            }
            // A few samples straddle a ridge, where two members are equally
            // near and the gradient is undefined; nearly all of them don't.
            assert!(
                tried > 50,
                "wall {wall}: only {tried} samples near the foam"
            );
            assert!(
                unit * 100 >= tried * 95,
                "wall {wall}: unit gradient at {unit} of {tried}"
            );
            assert!(
                landed * 100 >= tried * 95,
                "wall {wall}: landed {landed} of {tried}"
            );
        }
    }

    #[test]
    fn the_foam_field_is_continuous_and_capped() {
        // Across cell faces especially: an edge that belongs to the next cell
        // and not this one must still be seen from here.
        let cell = Vector3::new(1.0, 1.0, 1.0);
        let cap = 0.25;
        let field = |p: Vector3| {
            voronoi_distance(
                p,
                Vector3::ZERO,
                cell,
                Skeleton::Edges,
                1.0,
                11,
                &[],
                cap,
                0.0,
                None,
            )
        };
        let step = 1e-3;
        let mut p = Vector3::new(0.13, 0.41, 0.07);
        let dir = Vector3::new(0.71, 0.53, 0.46).normalize();
        let mut last = field(p);
        for _ in 0..4000 {
            p = p + dir * step;
            let v = field(p);
            assert!(v <= cap, "{v} above the cap");
            assert!(
                (v - last).abs() <= step * 1.001 + 1e-6,
                "jump {last} → {v} at {p:?}"
            );
            last = v;
        }
    }

    /// How many faces each seed of one cell of `layout` has, by casting rays
    /// from it and noting which bisector each meets first.
    fn face_counts(layout: SeedLayout) -> Vec<usize> {
        let sites = layout.sites();
        let mut others = Vec::new();
        for oz in -2..=2 {
            for oy in -2..=2 {
                for ox in -2..=2 {
                    for s in sites {
                        others.push([s[0] + ox as f64, s[1] + oy as f64, s[2] + oz as f64]);
                    }
                }
            }
        }
        sites
            .iter()
            .map(|me| {
                let mut hit = std::collections::BTreeSet::new();
                for r in 0..20_000 {
                    let h = hash_unit(r, 31, 7, 5);
                    let d = from_polar(Vector3::new(0.0, 0.0, 1.0), 2.0 * h[0] - 1.0, h[1] * TAU);
                    let d = [d.x as f64, d.y as f64, d.z as f64];
                    let mut first = (f64::INFINITY, usize::MAX);
                    for (k, o) in others.iter().enumerate() {
                        let w = [o[0] - me[0], o[1] - me[1], o[2] - me[2]];
                        let along = dot(w, d);
                        if dot(w, w) < 1e-12 || along <= 1e-12 {
                            continue;
                        }
                        // The ray reaches the bisector at t = |w|² / 2(w·d).
                        let t = dot(w, w) / (2.0 * along);
                        if t < first.0 {
                            first = (t, k);
                        }
                    }
                    hit.insert(first.1);
                }
                hit.len()
            })
            .collect()
    }

    #[test]
    fn crystal_layouts_make_their_textbook_cells() {
        let tally = |layout| {
            let mut counts = face_counts(layout);
            counts.sort_unstable();
            counts
        };
        assert_eq!(tally(SeedLayout::Kelvin), vec![14; 2]);
        assert_eq!(tally(SeedLayout::Fcc), vec![12; 4]);
        // Weaire–Phelan: two pentagonal dodecahedra, six tetrakaidecahedra.
        assert_eq!(
            tally(SeedLayout::WeairePhelan),
            [vec![12; 2], vec![14; 6]].concat()
        );
        // C15: sixteen dodecahedra (Z12), eight hexadecahedra (Z16).
        assert_eq!(tally(SeedLayout::C15), [vec![12; 16], vec![16; 8]].concat());
    }

    #[test]
    fn kelvin_foam_is_the_kelvin_strut_lattice() {
        // Two routes to the same thing, sharing no code: the edges of the
        // body-centred cubic Voronoi diagram, and the struts a Kelvin cell is
        // drawn with.
        let cell = Vector3::new(1.0, 1.0, 1.0);
        let segments = super::super::Strut::Kelvin.segments();
        let cap = 0.2;
        let mut worst = 0.0f32;
        for i in 0..3000 {
            let h = hash_unit(i, 4, 4, 4);
            let p = Vector3::new(h[0] * 2.0 - 0.5, h[1] * 2.0 - 0.5, h[2] * 2.0 - 0.5);
            let foam = voronoi_distance(
                p,
                Vector3::ZERO,
                cell,
                Skeleton::Edges,
                0.0,
                1,
                SeedLayout::Kelvin.sites(),
                cap,
                0.0,
                None,
            );
            let strut = super::super::strut::distance(p, Vector3::ZERO, cell, segments, cap);
            worst = worst.max((foam - strut).abs());
        }
        assert!(
            worst < 1e-5,
            "Kelvin foam and Kelvin struts differ by {worst}"
        );
    }

    #[test]
    fn gaussian_field_wavelengths_spread_about_the_cell() {
        let cell = Vector3::new(1.0, 1.0, 1.0);
        let (mut sum, mut sum_sq, mut n) = (0.0f64, 0.0f64, 0.0f64);
        for seed in 0..40 {
            for w in waves(Stochastic::GaussianField, cell, seed) {
                let m = (w.k.length() / TAU) as f64;
                sum += m;
                sum_sq += m * m;
                n += 1.0;
            }
        }
        let mean = sum / n;
        let spread = (sum_sq / n - mean * mean).sqrt();
        assert!((mean - 1.0).abs() < 0.05, "mean {mean}");
        assert!((spread - 0.45).abs() < 0.05, "spread {spread}");
        // And the spinodal's stay put.
        for w in waves(Stochastic::Spinodal, cell, 3) {
            assert!((w.k.length() / TAU - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn hyperuniform_gradient_matches_differences() {
        let cell = Vector3::new(1.0, 1.2, 0.9);
        let mut checked = 0;
        // Fibres are thin: most of space is a long way from one.
        for i in 0..2000 {
            let q = hash_unit(i, 9, 9, 9);
            let p = Vector3::new(q[0] * 3.0, q[1] * 3.0, q[2] * 3.0);
            let (v, g) = hyperuniform(p, Vector3::ZERO, cell, 4);
            if v < -0.29 {
                continue; // nowhere near a fibre: the field is flat
            }
            checked += 1;
            let e = 1e-3;
            for axis in 0..3 {
                let mut d = Vector3::ZERO;
                match axis {
                    0 => d.x = e,
                    1 => d.y = e,
                    _ => d.z = e,
                }
                let numeric = (hyperuniform(p + d, Vector3::ZERO, cell, 4).0
                    - hyperuniform(p - d, Vector3::ZERO, cell, 4).0)
                    / (2.0 * e);
                let analytic = [g.x, g.y, g.z][axis];
                assert!(
                    (numeric - analytic).abs() < 2e-2 * analytic.abs().max(1.0),
                    "{numeric} vs {analytic}"
                );
            }
        }
        assert!(checked > 30, "only {checked} samples near a fibre");
    }

    #[test]
    fn a_wet_bubble_is_its_cell_where_the_cell_is_flat() {
        // Away from edges a bubble is just its cell, face moved in by the
        // border and out again — so within the border of a face, the distance
        // into the bubble is the distance to the wall.
        let cell = Vector3::new(1.0, 1.0, 1.0);
        let border = 0.12f32;
        let mut checked = 0;
        // Points near a face and far from every edge are a thin slice of space.
        for i in 0..20_000 {
            let h = hash_unit(i, 6, 1, 8);
            let p = Vector3::new(h[0] * 3.0, h[1] * 3.0, h[2] * 3.0);
            let measure = |skeleton| {
                voronoi_distance(
                    p,
                    Vector3::ZERO,
                    cell,
                    skeleton,
                    1.0,
                    3,
                    &[],
                    1.0,
                    f32::NEG_INFINITY,
                    None,
                )
            };
            let wall = measure(Skeleton::Walls);
            let edge = measure(Skeleton::Edges);
            if wall >= 0.5 * border || edge < wall + 3.0 * border {
                continue;
            }
            checked += 1;
            let wet = measure(Skeleton::Wet { border });
            assert!(
                (wet + wall).abs() < 1e-5,
                "at {p:?}: wet {wet}, wall {wall}"
            );
        }
        assert!(checked > 50, "only {checked} samples near a flat wall");
    }

    /// Coefficient of variation of the cell volumes of a tile of seeds: how
    /// unequal the foam's cells are, measured on a grid of the tile.
    fn volume_spread(seeds: &[[f64; 3]]) -> f64 {
        const GRID: usize = 40;
        let size = [1.0; 3];
        let mut count = vec![0usize; seeds.len()];
        for k in 0..GRID {
            for j in 0..GRID {
                for i in 0..GRID {
                    let q = [i, j, k].map(|c| (c as f64 + 0.5) / GRID as f64);
                    let near = (0..seeds.len())
                        .min_by(|&a, &b| {
                            tile_distance_sq(seeds[a], q, size)
                                .total_cmp(&tile_distance_sq(seeds[b], q, size))
                        })
                        .unwrap();
                    count[near] += 1;
                }
            }
        }
        let mean = count.iter().sum::<usize>() as f64 / count.len() as f64;
        let var = count
            .iter()
            .map(|&c| (c as f64 - mean).powi(2))
            .sum::<f64>()
            / count.len() as f64;
        var.sqrt() / mean
    }

    #[test]
    fn relaxing_evens_the_cells_out() {
        let cell = Vector3::new(1.0, 1.0, 1.0);
        let (mut random, mut poisson, mut relaxed) = (0.0, 0.0, 0.0);
        for seed in 0..4 {
            let scattered: Vec<[f64; 3]> = (0..27)
                .map(|i| {
                    let h = hash_unit(i, 77, 0, seed);
                    [h[0] as f64, h[1] as f64, h[2] as f64]
                })
                .collect();
            random += volume_spread(&scattered);
            poisson += volume_spread(&seed_tile(SeedLayout::Poisson, 27, cell, seed));
            relaxed += volume_spread(&seed_tile(SeedLayout::Relaxed, 27, cell, seed));
        }
        // A random point set's cells vary by about 40 %; well-spread seeds by
        // much less, and relaxed ones by less again.
        assert!(
            poisson < 0.6 * random,
            "poisson {poisson} vs random {random}"
        );
        assert!(
            relaxed < 0.5 * poisson,
            "relaxed {relaxed} vs poisson {poisson}"
        );
        // The same seed, the same tile.
        assert_eq!(
            seed_tile(SeedLayout::Relaxed, 8, cell, 3),
            seed_tile(SeedLayout::Relaxed, 8, cell, 3)
        );
    }

    #[test]
    fn a_turing_pattern_settles_into_a_maze() {
        let start = std::time::Instant::now();
        let t = grow_turing(5);
        let took = start.elapsed();
        assert!(t.v.iter().all(|x| x.is_finite()), "the simulation blew up");
        let (lo, hi) =
            t.v.iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &x| {
                    (l.min(x), h.max(x))
                });
        assert!(hi - lo > 0.15, "no pattern formed: {lo}..{hi}");
        // Both phases, neither one a speck.
        let above = t.v.iter().filter(|&&x| x > t.level).count() as f32 / t.v.len() as f32;
        assert!((0.15..0.85).contains(&above), "{above} above the level");
        println!("grown in {took:?}, {:.0} % above the level", above * 100.0);
        // Periodic, and its gradient is its slope.
        let (a, _) = t.sample([0.13, 0.42, 0.77]);
        let (b, _) = t.sample([1.13, -0.58, 2.77]);
        assert!((a - b).abs() < 1e-6);
        let p = [0.311, 0.527, 0.204];
        let (_, g) = t.sample(p);
        let e = 1e-4;
        for axis in 0..3 {
            let (mut up, mut dn) = (p, p);
            up[axis] += e;
            dn[axis] -= e;
            let numeric = (t.sample(up).0 - t.sample(dn).0) / (2.0 * e as f32);
            assert!(
                (numeric - g[axis]).abs() < 1e-2 * g[axis].abs().max(0.1),
                "axis {axis}: {numeric} vs {}",
                g[axis]
            );
        }
    }

    #[test]
    fn a_power_diagram_without_weights_is_the_voronoi_diagram() {
        let cell = Vector3::new(1.0, 1.1, 0.9);
        let sites = SeedLayout::Kelvin.sites();
        let zero = vec![0.0; sites.len()];
        let mut worst = 0.0f32;
        for i in 0..2000 {
            let h = hash_unit(i, 2, 7, 1);
            let p = Vector3::new(h[0] * 3.0 - 1.0, h[1] * 3.0 - 1.0, h[2] * 3.0 - 1.0);
            for (wall, skeleton) in [(true, Skeleton::Walls), (false, Skeleton::Edges)] {
                let power = power_distance(p, Vector3::ZERO, cell, wall, sites, &zero, 0.3);
                let voronoi = voronoi_distance(
                    p,
                    Vector3::ZERO,
                    cell,
                    skeleton,
                    0.0,
                    1,
                    sites,
                    0.3,
                    0.0,
                    None,
                );
                worst = worst.max((power - voronoi).abs());
            }
        }
        assert!(worst < 1e-5, "power and Voronoi differ by {worst}");
    }

    #[test]
    fn bimodal_cells_come_in_two_sizes() {
        let cell = Vector3::new(1.0, 1.0, 1.0);
        for ratio in [2.0, 4.0] {
            let (sites, weights) = bimodal_tile(16, ratio, cell, 3);
            let large = 4;
            // Power-cell volumes, counted on a grid of the tile.
            const GRID: usize = 40;
            let mut volume = vec![0usize; sites.len()];
            for k in 0..GRID {
                for j in 0..GRID {
                    for i in 0..GRID {
                        let q = [i, j, k].map(|c| (c as f64 + 0.5) / GRID as f64);
                        let owner = (0..sites.len())
                            .min_by(|&a, &b| {
                                let pa = tile_distance_sq(sites[a], q, [1.0; 3]) - weights[a];
                                let pb = tile_distance_sq(sites[b], q, [1.0; 3]) - weights[b];
                                pa.total_cmp(&pb)
                            })
                            .unwrap();
                        volume[owner] += 1;
                    }
                }
            }
            let mean = |v: &[usize]| v.iter().sum::<usize>() as f64 / v.len() as f64;
            let measured = mean(&volume[..large]) / mean(&volume[large..]);
            assert!(
                (measured / ratio - 1.0).abs() < 0.1,
                "asked {ratio}, got {measured:.2}"
            );
        }
    }

    #[test]
    fn the_seed_graph_answers_as_the_search_does() {
        let cell = Vector3::new(1.0, 1.1, 0.9);
        let size = [1.0f64, 1.1, 0.9];
        let cap = 0.3f32;
        let layouts = [
            SeedLayout::Random,
            SeedLayout::Kelvin,
            SeedLayout::C15,
            SeedLayout::WeairePhelan,
            SeedLayout::Relaxed,
            SeedLayout::Poisson,
            SeedLayout::Fcc,
            SeedLayout::Bimodal,
        ];
        for layout in layouts {
            let (tile, weights) = if layout == SeedLayout::Bimodal {
                bimodal_tile(8, 3.0, cell, 4)
            } else {
                (seed_tile(layout, 8, cell, 4), Vec::new())
            };
            let graph =
                SeedGraph::new([0; 3], [2; 3], size, &tile, &weights, 1.0, 4).expect("small block");
            let mut fell_back = 0;
            let mut worst = 0.0f32;
            for i in 0..1500 {
                let h = hash_unit(i, 3, 3, 3);
                let p = Vector3::new(h[0] * 2.0 + 0.5, h[1] * 2.2 + 0.55, h[2] * 1.8 + 0.45);
                let rel = [p.x as f64, p.y as f64, p.z as f64];
                let mut modes = vec![Skeleton::Walls, Skeleton::Edges];
                if weights.is_empty() {
                    modes.push(Skeleton::Wet { border: 0.1 });
                }
                for mode in modes {
                    let want = if weights.is_empty() {
                        voronoi_distance(
                            p,
                            Vector3::ZERO,
                            cell,
                            mode,
                            1.0,
                            4,
                            &tile,
                            cap,
                            f32::NEG_INFINITY,
                            None,
                        )
                    } else {
                        power_distance(
                            p,
                            Vector3::ZERO,
                            cell,
                            mode == Skeleton::Walls,
                            &tile,
                            &weights,
                            cap,
                        )
                    };
                    match graph.distance(rel, mode, cap as f64, f64::NEG_INFINITY) {
                        Some(got) => worst = worst.max((got - want).abs()),
                        None => fell_back += 1,
                    }
                }
            }
            assert!(
                worst < 1e-5,
                "{}: graph and search differ by {worst}",
                layout.name()
            );
            assert!(
                fell_back == 0,
                "{}: {fell_back} samples fell back",
                layout.name()
            );
        }
    }

    #[test]
    fn a_two_size_wet_bubble_is_its_power_cell_where_flat() {
        // As for one size: away from edges a bubble is its cell, moved in and
        // out — here a power cell, which only the graph can measure.
        let cell = Vector3::new(1.0, 1.0, 1.0);
        let (tile, weights) = bimodal_tile(8, 3.0, cell, 6);
        let graph = SeedGraph::new([0; 3], [2; 3], [1.0; 3], &tile, &weights, 1.0, 6).unwrap();
        // Eight foam cells a lattice cell are small: a thin border, so some
        // of every face is clear of its edges.
        let border = 0.03;
        let mut checked = 0;
        for i in 0..20_000 {
            let h = hash_unit(i, 8, 8, 8);
            let p = [
                h[0] as f64 * 2.0 + 0.5,
                h[1] as f64 * 2.0 + 0.5,
                h[2] as f64 * 2.0 + 0.5,
            ];
            let measure = |mode| graph.distance(p, mode, 1.0, f64::NEG_INFINITY).unwrap();
            let wall = measure(Skeleton::Walls);
            let edge = measure(Skeleton::Edges);
            if wall >= 0.5 * border || edge < wall + 3.0 * border {
                continue;
            }
            checked += 1;
            let wet = measure(Skeleton::Wet { border });
            assert!(
                (wet + wall).abs() < 1e-5,
                "at {p:?}: wet {wet}, wall {wall}"
            );
        }
        assert!(checked > 50, "only {checked} samples near a flat wall");
    }

    #[test]
    fn binned_edges_are_the_open_foam() {
        // The open foam is its cells' edges: measured as segments, binned,
        // it reads what the plane search reads.
        let cell = Vector3::new(1.0, 1.1, 0.9);
        let size = [1.0f64, 1.1, 0.9];
        let cap = 0.25f32;
        for layout in [
            SeedLayout::Random,
            SeedLayout::Kelvin,
            SeedLayout::C15,
            SeedLayout::Relaxed,
            SeedLayout::Bimodal,
        ] {
            let (tile, weights) = if layout == SeedLayout::Bimodal {
                bimodal_tile(8, 3.0, cell, 4)
            } else {
                (seed_tile(layout, 8, cell, 4), Vec::new())
            };
            let graph = SeedGraph::new([0; 3], [2; 3], size, &tile, &weights, 1.0, 4).unwrap();
            let bins = EdgeBins::new(&graph, [0.4, 0.4, 0.4], [2.6, 2.9, 2.4], cap as f64).unwrap();
            let mut worst = 0.0f32;
            for i in 0..3000 {
                let h = hash_unit(i, 4, 4, 4);
                let p = Vector3::new(h[0] * 2.0 + 0.5, h[1] * 2.2 + 0.55, h[2] * 1.8 + 0.45);
                let want = if weights.is_empty() {
                    voronoi_distance(
                        p,
                        Vector3::ZERO,
                        cell,
                        Skeleton::Edges,
                        1.0,
                        4,
                        &tile,
                        cap,
                        f32::NEG_INFINITY,
                        None,
                    )
                } else {
                    power_distance(p, Vector3::ZERO, cell, false, &tile, &weights, cap)
                };
                let got = bins
                    .distance([p.x as f64, p.y as f64, p.z as f64], cap as f64)
                    .expect("inside the bins");
                worst = worst.max((got - want).abs());
            }
            assert!(
                worst < 1e-5,
                "{}: binned edges differ by {worst}",
                layout.name()
            );
        }
    }

    #[test]
    fn voronoi_is_deterministic() {
        let cell = Vector3::new(2.0, 3.0, 1.5);
        let p = Vector3::new(0.37, -1.2, 4.8);
        let a = voronoi_distance(
            p,
            Vector3::ZERO,
            cell,
            Skeleton::Edges,
            1.0,
            42,
            &[],
            2.0,
            0.0,
            None,
        );
        let b = voronoi_distance(
            p,
            Vector3::ZERO,
            cell,
            Skeleton::Edges,
            1.0,
            42,
            &[],
            2.0,
            0.0,
            None,
        );
        assert_eq!(a, b);
        let c = voronoi_distance(
            p,
            Vector3::ZERO,
            cell,
            Skeleton::Edges,
            1.0,
            43,
            &[],
            2.0,
            0.0,
            None,
        );
        assert!((a - c).abs() > 1e-6, "the seed did nothing");
    }

    #[test]
    fn spinodal_gradient_matches_differences() {
        let w = waves(Stochastic::Spinodal, Vector3::new(1.0, 1.0, 1.0), 5);
        let p = Vector3::new(0.31, -0.22, 0.77);
        let (_, g) = value_and_gradient(&w, p);
        let h = 1e-3;
        for axis in 0..3 {
            let mut d = Vector3::ZERO;
            match axis {
                0 => d.x = h,
                1 => d.y = h,
                _ => d.z = h,
            }
            let (up, _) = value_and_gradient(&w, p + d);
            let (dn, _) = value_and_gradient(&w, p - d);
            let numeric = (up - dn) / (2.0 * h);
            let analytic = match axis {
                0 => g.x,
                1 => g.y,
                _ => g.z,
            };
            assert!(
                (numeric - analytic).abs() < 1e-2,
                "axis {axis}: {numeric} vs {analytic}"
            );
        }
    }

    #[test]
    fn spinodal_has_unit_variance() {
        let w = waves(Stochastic::Spinodal, Vector3::new(1.0, 1.0, 1.0), 9);
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let n = 20;
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let p = Vector3::new(i as f32 * 0.31, j as f32 * 0.29, k as f32 * 0.37);
                    let (v, _) = value_and_gradient(&w, p);
                    sum += v as f64;
                    sum_sq += (v * v) as f64;
                }
            }
        }
        let count = (n * n * n) as f64;
        let mean = sum / count;
        let var = sum_sq / count - mean * mean;
        assert!(mean.abs() < 0.15, "mean {mean}");
        assert!((var - 1.0).abs() < 0.25, "variance {var}");
    }

    #[test]
    fn lamellar_waves_point_along_z() {
        let w = waves(Stochastic::SpinodalLamellar, Vector3::new(1.0, 1.0, 1.0), 2);
        for wave in &w {
            let n = wave.k.normalize();
            assert!(n.z > CONE.cos() - 1e-3, "wave off axis: {:?}", n);
        }
    }

    #[test]
    fn columnar_waves_lie_near_the_equator() {
        let w = waves(Stochastic::SpinodalColumnar, Vector3::new(1.0, 1.0, 1.0), 2);
        for wave in &w {
            let n = wave.k.normalize();
            assert!(n.z.abs() < CONE.sin() + 1e-3, "wave off equator: {:?}", n);
        }
    }

    #[test]
    fn cubic_waves_sit_on_the_axes() {
        let w = waves(Stochastic::SpinodalCubic, Vector3::new(1.0, 1.0, 1.0), 2);
        for wave in &w {
            let n = wave.k.normalize();
            let best = n.x.abs().max(n.y.abs()).max(n.z.abs());
            assert!(best > CONE.cos() - 1e-3, "wave off the axes: {:?}", n);
        }
    }
}
