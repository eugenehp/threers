//! Scale-rich networks — ligaments whose length, thickness and connectivity
//! all span orders of magnitude in one part (Both, Chen et al., *Nature* 659,
//! 2026).
//!
//! Every other lattice here has a scale: a cell size, a strut diameter, a
//! coordination number, the same everywhere or nearly so. This one has none.
//! It is grown, one ligament at a time, in a unit square:
//!
//! 1. pick a void, with probability proportional to its area, and a point in
//!    it, uniformly; draw a direction;
//! 2. lay a straight ligament of width `λₜ = λ₀ · t^(−α)` through the point,
//!    running both ways until it meets the material already there or the wall.
//!
//! The void the point was in is cut in three: the ligament, and a smaller void
//! on either side of it. Every void starts as the square and is only ever cut
//! by a straight strip, so every void is convex and the ligaments tile exactly
//! what the voids do not. Two numbers decide everything: `λ₀` how thick the
//! first ligament is, `α` how fast the later ones thin out.
//!
//! What comes out has no characteristic size. Widths follow a power law
//! `P(λ) ∝ λ^−(1+1/α)` by construction; lengths, because later ligaments land
//! in smaller voids, have a power-law tail; and degrees — counting each
//! ligament as a node and each junction as a link, as a road network does —
//! go as `P(k) ∝ k⁻³`, because a long ligament is hit by linearly more of the
//! later ones. The paper's measurements put the payoff in a stiffness whose
//! anisotropy is tunable at fixed density, and crush behaviour that does not
//! localise into bands.
//!
//! ```
//! use threers::ScaleRich;
//!
//! // One of the paper's samples: α = 0.3, λ₀ = 0.03, 500 ligaments.
//! let net = ScaleRich::new(0.3, 0.03).steps(500).seed(1).grow();
//! assert!((net.relative_density() - 0.40).abs() < 0.05);
//!
//! // A 40 mm tile, 5 mm thick, ready for a slicer.
//! let geom = net.geometry(40.0, 5.0);
//! assert!(geom.get_attribute("position").is_some());
//! ```
//!
//! # Fidelity to the reference
//!
//! The growth rule is the authors' (`Barabasi-Lab/Scale-Rich-Metamaterials`),
//! down to the parts the paper leaves out: the direction is drawn once per
//! step and only the point redrawn, up to [`attempts`](ScaleRich::attempts)
//! times; a point is accepted only if a strip `λ + gap` wide fits across it
//! inside its void, which is what keeps the pores printable
//! ([`gap`](ScaleRich::gap), 0.03 of the side by default, as theirs); a step
//! that finds no room is skipped, so `t` still advances; and a ligament that
//! ends up touching nothing but the walls is left out of the part.
//!
//! # Domains, directions and gradients
//!
//! The paper's Fig. 6 grows the same rule into other shapes and patterns, and
//! so does this:
//!
//! - [`domain`](ScaleRich::domain) grows inside any polygon, holes allowed.
//!   A convex one is a single starting void, as the square is. Anything else
//!   is triangulated, and the edges between triangles are left open: a
//!   ligament's strip carries on through them into the next triangle, and
//!   stops only at a wall or at material, so the triangulation never shows.
//! - [`ScaleRichAngles::Field`] reads each ligament's direction from a field
//!   at its nucleation point — the paper's polarised networks.
//! - [`nucleation`](ScaleRich::nucleation) makes some places likelier to seed
//!   a ligament than others: a density gradient.
//!
//! ```
//! use threers::{ScaleRich, ScaleRichAngles};
//!
//! // A ring, its ligaments running round it, denser towards the outside.
//! let circle = |r: f64, ccw: bool| -> Vec<[f64; 2]> {
//!     let n = 64;
//!     (0..n)
//!         .map(|i| {
//!             let a = std::f64::consts::TAU * i as f64 / n as f64;
//!             let a = if ccw { a } else { -a };
//!             [r * a.cos(), r * a.sin()]
//!         })
//!         .collect()
//! };
//! let net = ScaleRich::new(0.3, 0.02)
//!     .domain(circle(1.0, true), vec![circle(0.4, false)])
//!     .angles(ScaleRichAngles::field(|p| p[1].atan2(p[0]) + std::f64::consts::FRAC_PI_2))
//!     .nucleation(|p| (p[0] * p[0] + p[1] * p[1]).sqrt())
//!     .grow();
//! assert!(net.relative_density() > 0.1);
//! ```
//!
//! # Why not a field
//!
//! The other lattices are scalar fields contoured on a grid. That would undo
//! the point of this one: at `α = 0.9` and 500 steps the widest ligament is
//! 270 times the thinnest, and a grid fine enough for the thin ones is
//! hopeless for the part. The outline here is polygons instead — cut in
//! integer coordinates to a nanometre on a 100 mm part, and extruded.

use crate::core::BufferGeometry;
use crate::curves::{Path, Shape};
use crate::math::polygon_ops::{
    Boolean64, ClipType, EndType, FillRule, JoinType, Path64, PathOffset, Paths64, Point64,
    PolyPath64,
};
use crate::math::{Rng, Vector2};
use std::collections::HashMap;
use std::f64::consts::TAU;
use std::sync::Arc;

/// Bits the largest coordinate is scaled to for the boolean: a nanometre on
/// a 100 mm part, and far inside the 62 the algorithm takes.
const SCALE_BITS: i32 = 30;

/// How far each pore is grown, in grid units, before it is cut from the part.
///
/// The pores and the loose ligaments meet one another and the domain's edge
/// exactly — a T-junction at every ligament end — and once rounded to the
/// grid a vertex sits within a unit of the edge it was cut from. The boolean
/// decides such near-touches with `f64` cross products, and it decides them
/// wrongly often enough to matter: 28 pieces covering 0.402 unioned to 0.592,
/// in the C++ library as in the port, and which inputs fail moves with the
/// scale. Grown by two units, every near-touch is a proper overlap, which it
/// handles; the part is two units thinner round each pore.
const PORE_GROWTH: f64 = 2.0;

/// A function of a point, shared so the generator stays `Clone`.
type PointFn = Arc<dyn Fn([f64; 2]) -> f64 + Send + Sync>;

/// How each ligament's direction is drawn.
#[derive(Clone)]
pub enum ScaleRichAngles {
    /// Uniformly over the circle: no direction special, on average.
    Uniform,
    /// Uniformly from these angles, in radians from +x: an aligned network,
    /// stiff along them.
    Set(Vec<f64>),
    /// From a field, in radians from +x, read at the nucleation point: a
    /// polarised network whose ligaments follow the field. Build it with
    /// [`ScaleRichAngles::field`].
    Field(PointFn),
}

impl ScaleRichAngles {
    /// Directions read from `angle` at each nucleation point.
    pub fn field(angle: impl Fn([f64; 2]) -> f64 + Send + Sync + 'static) -> Self {
        ScaleRichAngles::Field(Arc::new(angle))
    }
}

impl std::fmt::Debug for ScaleRichAngles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScaleRichAngles::Uniform => f.write_str("Uniform"),
            ScaleRichAngles::Set(set) => f.debug_tuple("Set").field(set).finish(),
            ScaleRichAngles::Field(_) => f.write_str("Field(..)"),
        }
    }
}

/// A scale-rich network generator. See the [module docs](self).
#[derive(Clone)]
pub struct ScaleRich {
    alpha: f64,
    lambda0: f64,
    steps: usize,
    gap: f64,
    angles: ScaleRichAngles,
    seed: u64,
    attempts: usize,
    /// Outline counter-clockwise, then holes clockwise.
    domain: Vec<Vec<[f64; 2]>>,
    nucleation: Option<PointFn>,
}

impl std::fmt::Debug for ScaleRich {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScaleRich")
            .field("alpha", &self.alpha)
            .field("lambda0", &self.lambda0)
            .field("steps", &self.steps)
            .field("gap", &self.gap)
            .field("angles", &self.angles)
            .field("seed", &self.seed)
            .field("attempts", &self.attempts)
            .field(
                "domain",
                &self.domain.iter().map(Vec::len).collect::<Vec<_>>(),
            )
            .field("nucleation", &self.nucleation.is_some())
            .finish()
    }
}

impl ScaleRich {
    /// A network whose `t`-th ligament is `lambda0 · t^(−alpha)` wide, in
    /// the domain's units — the unit square's unless
    /// [`domain`](Self::domain) says otherwise. 500 steps, a 0.03 gap,
    /// uniform directions, seed 0.
    pub fn new(alpha: f64, lambda0: f64) -> Self {
        assert!(alpha >= 0.0, "alpha must not be negative");
        assert!(lambda0 > 0.0, "lambda0 must be positive");
        Self {
            alpha,
            lambda0,
            steps: 500,
            gap: 0.03,
            angles: ScaleRichAngles::Uniform,
            seed: 0,
            attempts: 1000,
            domain: vec![vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
            nucleation: None,
        }
    }

    /// How many ligaments to try to place. Steps with no room are skipped, so
    /// a jammed network has fewer.
    pub fn steps(mut self, steps: usize) -> Self {
        self.steps = steps;
        self
    }

    /// The narrowest void a new ligament may leave beside it — not quite: a
    /// point is accepted when a strip `λ + gap` wide fits across it, so the
    /// gap is shared between the two sides. 0.03 by default, as the authors';
    /// 0 is the paper's analytical model.
    pub fn gap(mut self, gap: f64) -> Self {
        assert!(gap >= 0.0, "gap must not be negative");
        self.gap = gap;
        self
    }

    /// How directions are drawn.
    pub fn angles(mut self, angles: ScaleRichAngles) -> Self {
        if let ScaleRichAngles::Set(set) = &angles {
            assert!(!set.is_empty(), "an angle set needs at least one angle");
        }
        self.angles = angles;
        self
    }

    /// The random seed. Every seed is a different network; the same seed is
    /// always the same one.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Points tried per step before the step is skipped, and skipped steps
    /// before the network is called jammed and growth stops. 1000, as the
    /// authors'.
    pub fn attempts(mut self, attempts: usize) -> Self {
        self.attempts = attempts.max(1);
        self
    }

    /// Grow inside this polygon, less these holes, instead of the unit
    /// square. Either winding will do; the polygon must not cross itself and
    /// the holes must lie inside it, apart. Widths and the gap are in the
    /// same units.
    pub fn domain(mut self, outline: Vec<[f64; 2]>, holes: Vec<Vec<[f64; 2]>>) -> Self {
        assert!(outline.len() >= 3, "a domain needs at least three corners");
        let wind = |mut ring: Vec<[f64; 2]>, ccw: bool| {
            ring.dedup();
            if ring.len() > 1 && ring.first() == ring.last() {
                ring.pop();
            }
            if (shoelace(&ring) > 0.0) != ccw {
                ring.reverse();
            }
            ring
        };
        let outline = wind(outline, true);
        assert!(outline.len() >= 3, "a domain needs at least three corners");
        self.domain = std::iter::once(outline)
            .chain(
                holes
                    .into_iter()
                    .filter(|h| h.len() >= 3)
                    .map(|h| wind(h, false)),
            )
            .collect();
        self
    }

    /// How likely a point is to seed a ligament, from 0 to 1: a candidate
    /// point is kept with this chance, and a rejected one counts as an
    /// attempt. A gradient here is a density gradient in the part.
    pub fn nucleation(mut self, weight: impl Fn([f64; 2]) -> f64 + Send + Sync + 'static) -> Self {
        self.nucleation = Some(Arc::new(weight));
        self
    }

    /// The width of the `t`-th ligament, counting from 1.
    pub fn width(&self, t: usize) -> f64 {
        self.lambda0 * (t.max(1) as f64).powf(-self.alpha)
    }

    /// Grows the network.
    pub fn grow(&self) -> ScaleRichNet {
        let mut rng = Rng::new(self.seed);
        let mut voids = Voids::new(domain_cells(&self.domain), self.steps + 1);
        let mut ligaments: Vec<ScaleRichLigament> = Vec::new();
        let mut skipped = 0;
        let mut jammed = false;

        for t in 1..=self.steps {
            let width = self.width(t);
            // Drawn once per step, as the reference does, unless it depends
            // on where the ligament starts.
            let drawn = match &self.angles {
                ScaleRichAngles::Uniform => Some(unit(&mut rng) * TAU),
                ScaleRichAngles::Set(set) => Some(set[rng.below(set.len())]),
                ScaleRichAngles::Field(_) => None,
            };
            let reach = 0.5 * (width + self.gap);

            let mut found = None;
            for _ in 0..self.attempts {
                let c = voids.pick(&mut rng);
                // Rounding can leave a filled cell a sliver of weight.
                if !voids.cells[c].alive() {
                    continue;
                }
                let p = voids.cells[c].sample(&mut rng);
                if let Some(weight) = &self.nucleation
                    && unit(&mut rng) >= weight(p)
                {
                    continue;
                }
                let angle = match (&self.angles, drawn) {
                    (_, Some(angle)) => angle,
                    (ScaleRichAngles::Field(field), None) => field(p),
                    _ => unreachable!(),
                };
                let normal = [-angle.sin(), angle.cos()];
                if voids.reaches(c, p, normal, reach)
                    && voids.reaches(c, p, [-normal[0], -normal[1]], reach)
                {
                    found = Some((c, p, angle));
                    break;
                }
            }
            let Some((c, p, angle)) = found else {
                skipped += 1;
                if skipped >= self.attempts {
                    jammed = true;
                    break;
                }
                continue;
            };

            let id = ligaments.len();
            let laid = voids.lay(c, p, angle, 0.5 * width, id as u32);
            for &j in &laid.neighbours {
                ligaments[j].neighbours.push(id);
            }
            ligaments.push(ScaleRichLigament {
                born: t,
                width,
                angle,
                centreline: laid.centreline,
                pieces: laid.pieces,
                neighbours: laid.neighbours,
                on_wall: laid.on_wall,
            });
        }

        ScaleRichNet {
            ligaments,
            voids: voids
                .cells
                .into_iter()
                .filter(|v| v.alive())
                .map(|v| v.pts)
                .collect(),
            domain: self.domain.clone(),
            steps: self.steps,
            jammed,
        }
    }

    /// The paper's analytical relative density for an unbounded run with no
    /// gap in the unit square, `ρ ≈ ℓ₀ λ₀ ζ(α + ½)` with `ℓ₀ ≈ 0.946` —
    /// `None` where the sum
    /// diverges and the network jams instead (`α ≤ ½`), or where the estimate
    /// passes 1, which also means it jams.
    pub fn predicted_density(&self) -> Option<f64> {
        let s = self.alpha + 0.5;
        if s <= 1.0 {
            return None;
        }
        let rho = MEAN_FIRST_LENGTH * self.lambda0 * zeta(s);
        (rho < 1.0).then_some(rho)
    }
}

/// `ℓ₀`, the expected length of the first chord through a unit square at a
/// uniform point and direction, as the paper quotes it.
const MEAN_FIRST_LENGTH: f64 = 0.946;

/// The Riemann zeta function for real `s > 1`: a direct sum and an
/// Euler–Maclaurin tail, good to about 1e-12.
fn zeta(s: f64) -> f64 {
    const N: usize = 64;
    let head: f64 = (1..N).map(|n| (n as f64).powf(-s)).sum();
    let n = N as f64;
    head + n.powf(1.0 - s) / (s - 1.0) + 0.5 * n.powf(-s) + s * n.powf(-s - 1.0) / 12.0
        - s * (s + 1.0) * (s + 2.0) * n.powf(-s - 3.0) / 720.0
}

/// A uniform `f64` in `[0, 1)`.
fn unit(rng: &mut Rng) -> f64 {
    (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// One ligament of a [`ScaleRichNet`].
#[derive(Clone, Debug, PartialEq)]
pub struct ScaleRichLigament {
    /// The step it was laid at, counting from 1 — its width is the
    /// generator's [`width`](ScaleRich::width) of this.
    pub born: usize,
    /// Its width, in the domain's units.
    pub width: f64,
    /// Its direction, in radians from +x.
    pub angle: f64,
    /// Where its centreline meets what it ends on.
    pub centreline: [[f64; 2]; 2],
    /// Its outline, in convex pieces, each counter-clockwise: its sides along
    /// its direction and its ends along whatever it ends on. One piece unless
    /// it crossed between the triangles of a non-convex
    /// [`domain`](ScaleRich::domain).
    pub pieces: Vec<Vec<[f64; 2]>>,
    /// Indices of the ligaments it touches: those it ends on and those that
    /// end on it.
    pub neighbours: Vec<usize>,
    /// Whether it ends on the domain's edge.
    pub on_wall: bool,
}

impl ScaleRichLigament {
    /// Its centreline length.
    pub fn length(&self) -> f64 {
        let [a, b] = self.centreline;
        (b[0] - a[0]).hypot(b[1] - a[1])
    }

    /// Its area.
    pub fn area(&self) -> f64 {
        self.pieces.iter().map(|p| shoelace(p)).sum()
    }

    /// The number of ligaments it touches.
    pub fn degree(&self) -> usize {
        self.neighbours.len()
    }

    /// Touches no other ligament: it would come out of the part loose, so the
    /// part leaves it out.
    pub fn is_floating(&self) -> bool {
        self.neighbours.is_empty()
    }
}

/// A grown scale-rich network, in its domain's coordinates. See
/// [`ScaleRich`].
#[derive(Clone, Debug)]
pub struct ScaleRichNet {
    ligaments: Vec<ScaleRichLigament>,
    voids: Vec<Vec<[f64; 2]>>,
    domain: Vec<Vec<[f64; 2]>>,
    steps: usize,
    jammed: bool,
}

impl ScaleRichNet {
    /// The ligaments, in the order they were laid.
    pub fn ligaments(&self) -> &[ScaleRichLigament] {
        &self.ligaments
    }

    /// The voids between them, each convex and counter-clockwise. In a
    /// non-convex domain a pore can be several of these, side by side.
    pub fn voids(&self) -> &[Vec<[f64; 2]>] {
        &self.voids
    }

    /// Steps asked for; [`ligaments`](Self::ligaments) has one per step that
    /// found room.
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Growth stopped early because nothing more would fit.
    pub fn jammed(&self) -> bool {
        self.jammed
    }

    /// The domain it grew in: outline counter-clockwise, then holes
    /// clockwise.
    pub fn domain(&self) -> &[Vec<[f64; 2]>] {
        &self.domain
    }

    /// Fraction of the domain that is material, floating ligaments left out
    /// as the part leaves them out.
    pub fn relative_density(&self) -> f64 {
        let solid: f64 = self
            .ligaments
            .iter()
            .filter(|l| !l.is_floating())
            .map(|l| l.area())
            .sum();
        solid / self.domain.iter().map(|r| shoelace(r)).sum::<f64>()
    }

    /// Every ligament's degree, in order.
    pub fn degrees(&self) -> Vec<usize> {
        self.ligaments.iter().map(|l| l.degree()).collect()
    }

    /// The paper's orientation tensor `M = Σ Aₜ dₜ ⊗ dₜ`: each ligament's
    /// direction weighted by its area.
    pub fn orientation_tensor(&self) -> [[f64; 2]; 2] {
        let mut m = [[0.0; 2]; 2];
        for l in self.ligaments.iter().filter(|l| !l.is_floating()) {
            let (a, d) = (l.area(), [l.angle.cos(), l.angle.sin()]);
            for (i, row) in m.iter_mut().enumerate() {
                for (j, v) in row.iter_mut().enumerate() {
                    *v += a * d[i] * d[j];
                }
            }
        }
        m
    }

    /// `η_G = μ₁/μ₂ ∈ [0, 1]`, the ratio of the orientation tensor's smaller
    /// eigenvalue to its larger: 1 for material laid equally every way, 0 for
    /// all of it along one direction. The paper finds it tracks the elastic
    /// anisotropy `E_min/E_max`.
    pub fn geometric_anisotropy(&self) -> f64 {
        let [[a, b], [_, d]] = self.orientation_tensor();
        let mean = 0.5 * (a + d);
        let spread = (0.25 * (a - d) * (a - d) + b * b).sqrt();
        if mean + spread <= 0.0 {
            return 1.0;
        }
        ((mean - spread) / (mean + spread)).max(0.0)
    }

    /// The part's outline, in the domain's coordinates: one entry per
    /// connected piece, its outline first and counter-clockwise, then its
    /// pores, clockwise. Every vertex is on a grid of 2⁻³⁰ of the domain's
    /// size, and each pore is two grid units larger than it grew (see
    /// `PORE_GROWTH` in the source), so the part is that much thinner round
    /// each pore.
    pub fn outlines(&self) -> Vec<Vec<Vec<[f64; 2]>>> {
        self.outlines_at(SCALE_BITS)
    }

    fn outlines_at(&self, bits: i32) -> Vec<Vec<Vec<[f64; 2]>>> {
        let extent = self
            .domain
            .iter()
            .flatten()
            .fold(0.0f64, |m, q| m.max(q[0].abs()).max(q[1].abs()));
        let scale = 2f64.powi(bits - extent.max(f64::MIN_POSITIVE).log2().ceil() as i32);
        let ring = |pts: &[[f64; 2]]| -> Path64 {
            pts.iter()
                .map(|q| Point64::new((q[0] * scale).round() as i64, (q[1] * scale).round() as i64))
                .collect()
        };
        // The part is the domain less its pores and less any loose ligament,
        // each grown a little on its own so that the ones that meet overlap
        // (see `PORE_GROWTH`). Mitred, so a convex polygon stays one, with a
        // limit so that a needle-sharp corner does not reach across a thin
        // ligament.
        let grow = |pts: &[[f64; 2]]| -> Paths64 {
            let mut o = PathOffset::new(2.0, 0.0, false, false);
            o.add_path(&ring(pts), JoinType::Miter, EndType::Polygon);
            o.execute_paths(PORE_GROWTH)
        };
        let mut cutters: Paths64 = self.voids.iter().flat_map(|v| grow(v)).collect();
        cutters.extend(
            self.ligaments
                .iter()
                .filter(|l| l.is_floating())
                .flat_map(|l| l.pieces.iter().flat_map(|p| grow(p))),
        );
        let mut boolean = Boolean64::new();
        boolean.add_subject(&self.domain.iter().map(|r| ring(r)).collect::<Vec<_>>());
        boolean.add_clip(&cutters);
        let (_, tree, _) = boolean.execute_tree(ClipType::Difference, FillRule::NonZero);

        let back = |ring: &Path64, ccw: bool| {
            let mut v: Vec<[f64; 2]> = ring
                .iter()
                .map(|q| [q.x as f64 / scale, q.y as f64 / scale])
                .collect();
            if (shoelace(&v) > 0.0) != ccw {
                v.reverse();
            }
            v
        };
        fn outers<'a>(node: &'a PolyPath64, out: &mut Vec<&'a PolyPath64>) {
            for outer in &node.childs {
                out.push(outer);
                for hole in &outer.childs {
                    outers(hole, out);
                }
            }
        }
        let mut found = Vec::new();
        outers(&tree, &mut found);
        found
            .into_iter()
            .map(|outer| {
                std::iter::once(back(&outer.polygon, true))
                    .chain(outer.childs.iter().map(|h| back(&h.polygon, false)))
                    .collect()
            })
            .collect()
    }

    /// The [`outlines`](Self::outlines) as shapes, every coordinate
    /// multiplied by `size` — the side of the part for the default unit
    /// square — for anything that takes a [`Shape`].
    pub fn shapes(&self, size: f32) -> Vec<Shape> {
        let path = |ring: &[[f64; 2]]| {
            let mut p = Path::new();
            let at = |q: &[f64; 2]| {
                Vector2::new((q[0] * size as f64) as f32, (q[1] * size as f64) as f32)
            };
            for (i, q) in ring.iter().enumerate() {
                if i == 0 {
                    p.move_to(at(q));
                } else {
                    p.line_to(at(q));
                }
            }
            if let Some(q) = ring.first() {
                p.line_to(at(q));
            }
            p
        };
        self.outlines()
            .iter()
            .map(|piece| {
                let mut shape = Shape::from_path(path(&piece[0]));
                for hole in &piece[1..] {
                    shape.add_hole(path(hole));
                }
                shape
            })
            .collect()
    }

    /// The part as a mesh: the [`outlines`](Self::outlines), every coordinate
    /// multiplied by `size` — the side of the part for the default unit
    /// square — extruded `depth` along +z: every vertex where the
    /// boolean put it, the caps triangulated with their pores, and a flat
    /// wall on every side.
    pub fn geometry(&self, size: f32, depth: f32) -> BufferGeometry {
        let mut positions: Vec<f32> = Vec::new();
        let mut normals: Vec<f32> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        for piece in self.outlines() {
            let contours: Vec<Vec<Vector2>> = piece
                .iter()
                .map(|ring| {
                    ring.iter()
                        .map(|q| {
                            Vector2::new((q[0] * size as f64) as f32, (q[1] * size as f64) as f32)
                        })
                        .collect()
                })
                .collect();
            let flat: Vec<Vector2> = contours.iter().flatten().copied().collect();
            let mut hole_starts = Vec::new();
            let mut at = 0;
            for c in &contours[..contours.len() - 1] {
                at += c.len();
                hole_starts.push(at);
            }
            // Caps: the bottom faces down, so its triangles turn the other way.
            for (z, nz, flip) in [(0.0, -1.0, true), (depth, 1.0, false)] {
                let base = (positions.len() / 3) as u32;
                for p in &flat {
                    positions.extend_from_slice(&[p.x, p.y, z]);
                    normals.extend_from_slice(&[0.0, 0.0, nz]);
                }
                let tris = crate::curves::earcut::earcut(&flat, &hole_starts);
                for tri in tris.as_chunks::<3>().0 {
                    let [a, b, c] = tri.map(|i| base + i);
                    indices.extend_from_slice(&if flip { [a, c, b] } else { [a, b, c] });
                }
            }
            // Walls. Outlines run counter-clockwise and pores clockwise, so the
            // material is always on the left and the outward normal is the
            // side turned right.
            for c in &contours {
                for i in 0..c.len() {
                    let (a, b) = (c[i], c[(i + 1) % c.len()]);
                    let n = Vector2::new(b.y - a.y, a.x - b.x);
                    let len = n.length();
                    if len == 0.0 {
                        continue;
                    }
                    let base = (positions.len() / 3) as u32;
                    for (p, z) in [(a, 0.0), (b, 0.0), (b, depth), (a, depth)] {
                        positions.extend_from_slice(&[p.x, p.y, z]);
                        normals.extend_from_slice(&[n.x / len, n.y / len, 0.0]);
                    }
                    indices.extend_from_slice(&[
                        base,
                        base + 1,
                        base + 2,
                        base,
                        base + 2,
                        base + 3,
                    ]);
                }
            }
        }
        let mut g = BufferGeometry::new();
        g.set_attribute("position", crate::core::BufferAttribute::new(positions, 3));
        g.set_attribute("normal", crate::core::BufferAttribute::new(normals, 3));
        g.set_index(indices);
        g
    }
}

/// What one side of a void lies along.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Owner {
    /// The domain's edge.
    Wall,
    /// Nothing: the void carries on into this cell. Only between the
    /// triangles of a non-convex domain.
    Open(u32),
    /// A ligament, by index.
    Ligament(u32),
    /// The cut that made this piece — only on the strip being laid, whose
    /// cut sides face the new voids.
    Cut,
}

/// A convex polygon, counter-clockwise, with what each side `pts[i] →
/// pts[i + 1]` lies along.
#[derive(Clone, Debug, Default)]
struct Cell {
    pts: Vec<[f64; 2]>,
    on: Vec<Owner>,
}

impl Cell {
    fn area(&self) -> f64 {
        shoelace(&self.pts)
    }

    /// Not yet filled by a ligament.
    fn alive(&self) -> bool {
        self.pts.len() >= 3
    }

    /// The side a ray from `from` (inside or on the boundary) along `dir`
    /// leaves through, and how far along it does.
    fn exit(&self, from: [f64; 2], dir: [f64; 2]) -> Option<(f64, usize)> {
        let n = self.pts.len();
        let mut best: Option<(f64, usize)> = None;
        for i in 0..n {
            let (a, b) = (self.pts[i], self.pts[(i + 1) % n]);
            let out = [b[1] - a[1], a[0] - b[0]];
            let towards = dir[0] * out[0] + dir[1] * out[1];
            if towards <= 0.0 {
                continue;
            }
            let t = (((a[0] - from[0]) * out[0] + (a[1] - from[1]) * out[1]) / towards).max(0.0);
            if best.is_none_or(|(bt, _)| t < bt) {
                best = Some((t, i));
            }
        }
        best
    }

    /// Strictly inside.
    fn contains(&self, q: [f64; 2]) -> bool {
        let n = self.pts.len();
        (0..n).all(|i| {
            let (a, b) = (self.pts[i], self.pts[(i + 1) % n]);
            (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]) > 0.0
        })
    }

    /// A point uniformly inside: a fan triangle by area, then a point in it.
    fn sample(&self, rng: &mut Rng) -> [f64; 2] {
        let o = self.pts[0];
        let tri = |i: usize| {
            let (a, b) = (self.pts[i], self.pts[i + 1]);
            0.5 * ((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))
        };
        let total: f64 = (1..self.pts.len() - 1).map(tri).sum();
        let mut target = unit(rng) * total;
        let mut i = 1;
        while i < self.pts.len() - 2 {
            let a = tri(i);
            if target < a {
                break;
            }
            target -= a;
            i += 1;
        }
        let (a, b) = (self.pts[i], self.pts[i + 1]);
        let (mut u, mut v) = (unit(rng), unit(rng));
        if u + v > 1.0 {
            (u, v) = (1.0 - u, 1.0 - v);
        }
        [
            o[0] + u * (a[0] - o[0]) + v * (b[0] - o[0]),
            o[1] + u * (a[1] - o[1]) + v * (b[1] - o[1]),
        ]
    }

    /// The part of the cell with `lo ≤ s ≤ hi`, `s` the signed distance of
    /// each vertex across the cut. Sides along a cut line are owned by `cut`.
    ///
    /// Each crossing is interpolated along the original side, from its own
    /// start, whichever piece asks for it — so the void either side of a new
    /// ligament and the ligament itself get the same point to the bit, and
    /// meet exactly.
    fn clip(&self, s: &[f64], lo: f64, hi: f64, cut: Owner) -> Cell {
        let n = self.pts.len();
        let inside = |v: f64| lo <= v && v <= hi;
        let mut pts = Vec::with_capacity(n + 2);
        let mut on = Vec::with_capacity(n + 2);
        for i in 0..n {
            let j = (i + 1) % n;
            let (a, b, sa, sb) = (self.pts[i], self.pts[j], s[i], s[j]);
            if inside(sa) {
                pts.push(a);
                on.push(self.on[i]);
            }
            // The levels this side crosses strictly, in the order it meets
            // them.
            let mut levels = [lo, hi];
            if sb < sa {
                levels.swap(0, 1);
            }
            for c in levels {
                if !c.is_finite() || !((sa < c && c < sb) || (sb < c && c < sa)) {
                    continue;
                }
                let f = (c - sa) / (sb - sa);
                let x = [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])];
                // Past the crossing the side is inside or it is not; if it is,
                // what follows lies along this side, and if not, along the cut
                // until the polygon comes back in.
                let entering = if sb > sa { c == lo } else { c == hi };
                pts.push(x);
                on.push(if entering { self.on[i] } else { cut });
            }
        }
        // A crossing that rounded onto a vertex leaves a side of no length.
        let mut k = 0;
        while k < pts.len() && pts.len() > 3 {
            let next = (k + 1) % pts.len();
            if pts[k] == pts[next] {
                pts.remove(k);
                on.remove(k);
            } else {
                k += 1;
            }
        }
        Cell { pts, on }
    }

    /// Where the cut's zero line — the new ligament's centreline — crosses
    /// the cell's sides.
    fn crossings(&self, s: &[f64]) -> Vec<[f64; 2]> {
        let n = self.pts.len();
        let mut ends = Vec::new();
        for i in 0..n {
            let j = (i + 1) % n;
            let (a, b, sa, sb) = (self.pts[i], self.pts[j], s[i], s[j]);
            if sa == 0.0 {
                ends.push(a);
            } else if (sa < 0.0 && sb > 0.0) || (sa > 0.0 && sb < 0.0) {
                let f = sa / (sa - sb);
                ends.push([a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])]);
            }
        }
        ends
    }
}

/// Whether a counter-clockwise ring turns left, or goes straight, at every
/// corner.
fn is_convex(ring: &[[f64; 2]]) -> bool {
    let n = ring.len();
    (0..n).all(|i| {
        let (a, b, c) = (ring[i], ring[(i + 1) % n], ring[(i + 2) % n]);
        (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) >= 0.0
    })
}

/// The starting voids for a domain: the domain itself if it is convex, its
/// triangles joined by open sides if not.
fn domain_cells(domain: &[Vec<[f64; 2]>]) -> Vec<Cell> {
    let outline = &domain[0];
    if domain.len() == 1 && is_convex(outline) {
        return vec![Cell {
            pts: outline.clone(),
            on: vec![Owner::Wall; outline.len()],
        }];
    }
    let pts: Vec<[f64; 2]> = domain.iter().flatten().copied().collect();
    // The triangulation only picks which corners to join; the corners
    // themselves stay the f64 ones.
    let flat: Vec<Vector2> = pts
        .iter()
        .map(|q| Vector2::new(q[0] as f32, q[1] as f32))
        .collect();
    let mut hole_starts = Vec::new();
    let mut at = 0;
    for ring in &domain[..domain.len() - 1] {
        at += ring.len();
        hole_starts.push(at);
    }
    let mut corners: Vec<[usize; 3]> = Vec::new();
    for t in crate::curves::earcut::earcut(&flat, &hole_starts)
        .as_chunks::<3>()
        .0
    {
        let mut v = t.map(|i| i as usize);
        let area = shoelace(&v.map(|i| pts[i]));
        if area == 0.0 {
            continue;
        }
        if area < 0.0 {
            v.swap(1, 2);
        }
        corners.push(v);
    }
    let mut cells: Vec<Cell> = corners
        .iter()
        .map(|v| Cell {
            pts: v.map(|i| pts[i]).to_vec(),
            on: vec![Owner::Wall; 3],
        })
        .collect();
    let mut sides: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
    for (c, v) in corners.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (v[k], v[(k + 1) % 3]);
            match sides.remove(&(a.min(b), a.max(b))) {
                Some((d, j)) => {
                    cells[c].on[k] = Owner::Open(d as u32);
                    cells[d].on[j] = Owner::Open(c as u32);
                }
                None => {
                    sides.insert((a.min(b), a.max(b)), (c, k));
                }
            }
        }
    }
    cells
}

/// What laying one ligament produced.
struct Laid {
    pieces: Vec<Vec<[f64; 2]>>,
    centreline: [[f64; 2]; 2],
    neighbours: Vec<usize>,
    on_wall: bool,
}

/// The voids still open, and their areas for picking one by area.
struct Voids {
    cells: Vec<Cell>,
    weights: Fenwick,
}

impl Voids {
    fn new(cells: Vec<Cell>, capacity: usize) -> Self {
        let mut weights = Fenwick::with_capacity(capacity.max(cells.len()));
        for c in &cells {
            weights.push(c.area());
        }
        Self { cells, weights }
    }

    /// A cell, with probability proportional to its area.
    fn pick(&self, rng: &mut Rng) -> usize {
        self.weights
            .find(unit(rng) * self.weights.total())
            .min(self.cells.len() - 1)
    }

    /// Whether the void runs at least `r` from `p`, in cell `c`, along `dir`
    /// — through open sides into the cells beyond, if it has to.
    fn reaches(&self, c: usize, p: [f64; 2], dir: [f64; 2], r: f64) -> bool {
        let (mut c, mut from, mut left) = (c, p, r);
        for _ in 0..=self.cells.len() {
            let cell = &self.cells[c];
            if cell.contains([from[0] + dir[0] * left, from[1] + dir[1] * left]) {
                return true;
            }
            let Some((t, i)) = cell.exit(from, dir) else {
                return false;
            };
            match cell.on[i] {
                Owner::Open(next) if t < left => {
                    from = [from[0] + dir[0] * t, from[1] + dir[1] * t];
                    left -= t;
                    c = next as usize;
                }
                _ => return false,
            }
        }
        false
    }

    /// Lays ligament `id` through `p` in cell `c0`: cuts every cell its strip
    /// reaches through open sides, and keeps what is left either side as
    /// voids.
    fn lay(&mut self, c0: usize, p: [f64; 2], angle: f64, half: f64, id: u32) -> Laid {
        let dir = [angle.cos(), angle.sin()];
        let normal = [-dir[1], dir[0]];
        let across = |q: [f64; 2]| (q[0] - p[0]) * normal[0] + (q[1] - p[1]) * normal[1];
        let mut laid = Laid {
            pieces: Vec::new(),
            centreline: [[0.0; 2]; 2],
            neighbours: Vec::new(),
            on_wall: false,
        };
        let mut ends: Vec<[f64; 2]> = Vec::new();
        // Each cut cell, and what became of it: the void left above the
        // strip and the one below, by index, if any.
        let mut split: Vec<(usize, Option<usize>, Option<usize>)> = Vec::new();
        // Cells beside a cut one, whose open sides may now point at a piece.
        let mut beside: Vec<usize> = Vec::new();
        let mut queue = vec![c0];
        while let Some(c) = queue.pop() {
            if split.iter().any(|e| e.0 == c) {
                continue;
            }
            let cell = std::mem::take(&mut self.cells[c]);
            let s: Vec<f64> = cell.pts.iter().map(|&q| across(q)).collect();
            let above = cell.clip(&s, half, f64::INFINITY, Owner::Ligament(id));
            let below = cell.clip(&s, f64::NEG_INFINITY, -half, Owner::Ligament(id));
            let strip = cell.clip(&s, -half, half, Owner::Cut);

            // What the ligament ends on is whatever its outline shares with
            // the void it was cut from; an open side is where it carries on.
            if strip.alive() {
                for (i, owner) in strip.on.iter().enumerate() {
                    let (a, b) = (strip.pts[i], strip.pts[(i + 1) % strip.pts.len()]);
                    if a == b {
                        continue;
                    }
                    match *owner {
                        Owner::Wall => laid.on_wall = true,
                        Owner::Ligament(j) => {
                            if !laid.neighbours.contains(&(j as usize)) {
                                laid.neighbours.push(j as usize);
                            }
                        }
                        Owner::Open(next) => queue.push(next as usize),
                        Owner::Cut => {}
                    }
                }
            }
            ends.extend(cell.crossings(&s));
            beside.extend(cell.on.iter().filter_map(|o| match o {
                Owner::Open(n) => Some(*n as usize),
                _ => None,
            }));

            let old = cell.area();
            let (a_above, a_below) = (above.area(), below.area());
            let (keep_above, keep_below) = (
                above.alive() && a_above > 0.0,
                below.alive() && a_below > 0.0,
            );
            if strip.alive() {
                laid.pieces.push(strip.pts);
            }
            match (keep_above, keep_below) {
                (true, true) => {
                    self.cells[c] = above;
                    self.weights.add(c, a_above - old);
                    self.cells.push(below);
                    self.weights.push(a_below);
                    split.push((c, Some(c), Some(self.cells.len() - 1)));
                }
                (true, false) => {
                    self.cells[c] = above;
                    self.weights.add(c, a_above - old);
                    split.push((c, Some(c), None));
                }
                (false, true) => {
                    self.cells[c] = below;
                    self.weights.add(c, a_below - old);
                    split.push((c, None, Some(c)));
                }
                (false, false) => {
                    self.weights.add(c, -old);
                    split.push((c, None, None));
                }
            }
        }

        // An open side that pointed at a cut cell now points at whichever
        // piece of it lies on the same side of the strip.
        if !beside.is_empty() {
            let mut scan = beside;
            scan.extend(split.iter().flat_map(|e| [e.1, e.2]).flatten());
            scan.sort_unstable();
            scan.dedup();
            for y in scan {
                let cell = &mut self.cells[y];
                for i in 0..cell.on.len() {
                    let Owner::Open(x) = cell.on[i] else {
                        continue;
                    };
                    let Some(&(_, up, down)) = split.iter().find(|e| e.0 == x as usize) else {
                        continue;
                    };
                    let (a, b) = (cell.pts[i], cell.pts[(i + 1) % cell.pts.len()]);
                    let side = across([0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])]);
                    let target = if side > 0.0 { up.or(down) } else { down.or(up) };
                    cell.on[i] = target.map_or(Owner::Wall, |t| Owner::Open(t as u32));
                }
            }
        }

        let along = |q: &[f64; 2]| q[0] * dir[0] + q[1] * dir[1];
        let lo = ends
            .iter()
            .copied()
            .min_by(|a, b| along(a).total_cmp(&along(b)));
        let hi = ends
            .iter()
            .copied()
            .max_by(|a, b| along(a).total_cmp(&along(b)));
        if let (Some(a), Some(b)) = (lo, hi) {
            laid.centreline = [a, b];
        }
        laid
    }
}

fn shoelace(pts: &[[f64; 2]]) -> f64 {
    let n = pts.len();
    0.5 * (0..n)
        .map(|i| {
            let (a, b) = (pts[i], pts[(i + 1) % n]);
            a[0] * b[1] - b[0] * a[1]
        })
        .sum::<f64>()
}

/// A Fenwick tree of void areas, so a void is picked by area in `log n`
/// rather than a scan — the difference between seconds and minutes at the
/// paper's 30,000 steps.
struct Fenwick {
    tree: Vec<f64>,
    values: Vec<f64>,
}

impl Fenwick {
    fn with_capacity(n: usize) -> Self {
        Self {
            tree: vec![0.0; n + 1],
            values: Vec::with_capacity(n),
        }
    }

    fn add(&mut self, i: usize, delta: f64) {
        self.values[i] += delta;
        let mut i = i + 1;
        while i < self.tree.len() {
            self.tree[i] += delta;
            i += i.isolate_lowest_one();
        }
    }

    /// A new last entry, growing the tree if it is full.
    fn push(&mut self, value: f64) {
        if self.values.len() + 1 >= self.tree.len() {
            let values = std::mem::take(&mut self.values);
            *self = Self::with_capacity(2 * self.tree.len());
            for v in values {
                self.push(v);
            }
        }
        self.values.push(0.0);
        self.add(self.values.len() - 1, value);
    }

    fn total(&self) -> f64 {
        let mut i = self.tree.len() - 1;
        let mut sum = 0.0;
        while i > 0 {
            sum += self.tree[i];
            i -= i.isolate_lowest_one();
        }
        sum
    }

    /// The first index whose running sum passes `target`.
    fn find(&self, mut target: f64) -> usize {
        let mut pos = 0;
        let mut step = (self.tree.len() - 1).next_power_of_two();
        while step > 0 {
            let next = pos + step;
            if next < self.tree.len() && self.tree[next] <= target {
                pos = next;
                target -= self.tree[next];
            }
            step >>= 1;
        }
        pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_network() {
        let a = ScaleRich::new(0.3, 0.03).steps(200).seed(7).grow();
        let b = ScaleRich::new(0.3, 0.03).steps(200).seed(7).grow();
        let c = ScaleRich::new(0.3, 0.03).steps(200).seed(8).grow();
        assert_eq!(a.ligaments(), b.ligaments());
        assert_ne!(a.ligaments(), c.ligaments());
    }

    #[test]
    fn ligaments_and_voids_tile_the_square() {
        for (alpha, lambda0) in [(0.0, 0.01), (0.5, 0.06), (0.9, 0.15)] {
            let net = ScaleRich::new(alpha, lambda0).steps(500).seed(3).grow();
            let solid: f64 = net.ligaments().iter().map(|l| l.area()).sum();
            let void: f64 = net.voids().iter().map(|v| shoelace(v)).sum();
            assert!((solid + void - 1.0).abs() < 1e-12, "{solid} + {void}");
            assert!(net.voids().iter().all(|v| shoelace(v) > 0.0));
            for l in net.ligaments() {
                assert!(l.area() > 0.0);
                // Every outline vertex lies within the strip it was cut from.
                let n = [-l.angle.sin(), l.angle.cos()];
                let [c, _] = l.centreline;
                for q in l.pieces.iter().flatten() {
                    let s = (q[0] - c[0]) * n[0] + (q[1] - c[1]) * n[1];
                    assert!(s.abs() <= 0.5 * l.width + 1e-12);
                }
            }
        }
    }

    #[test]
    fn widths_follow_the_schedule() {
        let g = ScaleRich::new(0.5, 0.06).steps(300).seed(1);
        for l in g.grow().ligaments() {
            assert_eq!(l.width, g.width(l.born));
        }
    }

    #[test]
    fn neighbours_are_mutual() {
        let net = ScaleRich::new(0.3, 0.03).steps(500).seed(2).grow();
        for (i, l) in net.ligaments().iter().enumerate() {
            for &j in &l.neighbours {
                assert!(net.ligaments()[j].neighbours.contains(&i));
            }
        }
        // Junctions: one per ligament end that is not on the wall, about two
        // per ligament.
        let links: usize = net.degrees().iter().sum::<usize>() / 2;
        let n = net.ligaments().len();
        assert!(links > n && links < 3 * n, "{links} links for {n}");
    }

    /// The authors' published samples (`Sample_files/` in their repository,
    /// 500 ligaments each) come out at these densities; a seed of ours lands
    /// within the spread between seeds.
    #[test]
    fn densities_match_the_reference_samples() {
        for (alpha, lambda0, reference) in
            [(0.0, 0.01, 0.431), (0.3, 0.03, 0.4006), (0.9, 0.15, 0.408)]
        {
            let runs: Vec<f64> = (0..6)
                .map(|seed| {
                    ScaleRich::new(alpha, lambda0)
                        .steps(500)
                        .seed(seed)
                        .grow()
                        .relative_density()
                })
                .collect();
            let mean = runs.iter().sum::<f64>() / runs.len() as f64;
            assert!(
                (mean - reference).abs() < 0.03,
                "α {alpha}: {mean} vs {reference}"
            );
        }
    }

    #[test]
    fn aligned_angles_are_anisotropic() {
        // One direction alone never crosses itself: every ligament runs wall
        // to wall and floats. Two close ones do cross.
        let aligned = ScaleRich::new(0.3, 0.03)
            .angles(ScaleRichAngles::Set(vec![0.0, 0.2]))
            .grow();
        assert!(aligned.geometric_anisotropy() < 0.02);
    }

    /// The paper's Fig. 2: at one density, anisotropy grows with α — equal
    /// widths spread the material every way, while a fast decay leaves it to a
    /// few thick early ligaments that happen to point somewhere.
    #[test]
    fn anisotropy_grows_with_alpha() {
        let mean = |alpha: f64, lambda0: f64| {
            (0..6)
                .map(|seed| {
                    ScaleRich::new(alpha, lambda0)
                        .seed(seed)
                        .grow()
                        .geometric_anisotropy()
                })
                .sum::<f64>()
                / 6.0
        };
        // Both about 40 % dense.
        let (even, decaying) = (mean(0.0, 0.01), mean(0.9, 0.15));
        assert!(even > 0.65, "{even}");
        assert!(decaying < 0.45, "{decaying}");
    }

    #[test]
    fn the_part_is_the_ligaments() {
        let net = ScaleRich::new(0.5, 0.06).steps(500).seed(4).grow();
        let pieces = net.outlines();
        assert!(!pieces.is_empty());
        for piece in &pieces {
            assert!(shoelace(&piece[0]) > 0.0);
            assert!(piece[1..].iter().all(|h| shoelace(h) < 0.0));
        }
        let area: f64 = pieces.iter().flatten().map(|r| shoelace(r)).sum();
        assert!(
            // To the grid the boolean rounds to.
            (area - net.relative_density()).abs() < 1e-6,
            "{area} vs {}",
            net.relative_density()
        );
        assert_eq!(net.shapes(1.0).len(), pieces.len());
        // Divergence theorem: wound outward and closed, the signed volume is
        // the outline's area times the depth.
        let geom = net.geometry(10.0, 2.0);
        let pos = &geom.get_attribute("position").unwrap().array;
        let p = |i: u32| {
            let i = i as usize * 3;
            [pos[i] as f64, pos[i + 1] as f64, pos[i + 2] as f64]
        };
        let index = geom.index.as_ref().unwrap();
        let volume: f64 = index
            .as_chunks::<3>()
            .0
            .iter()
            .map(|t| {
                let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
                (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                    + a[2] * (b[0] * c[1] - b[1] * c[0]))
                    / 6.0
            })
            .sum();
        let expected = net.relative_density() * 100.0 * 2.0;
        assert!(
            (volume - expected).abs() < 1e-4 * expected,
            "{volume} vs {expected}"
        );
    }

    /// The paper's Fig. 1g: degrees go as `P(k) ∝ k⁻³`, so the fraction of
    /// ligaments with degree at least `k` falls as `k⁻²`.
    #[test]
    fn degrees_have_a_cubic_tail() {
        let net = ScaleRich::new(0.9, 0.1)
            .steps(20_000)
            .gap(0.0)
            .seed(1)
            .grow();
        let degrees = net.degrees();
        let at_least =
            |k: usize| degrees.iter().filter(|&&d| d >= k).count() as f64 / degrees.len() as f64;
        let slope = (at_least(32).ln() - at_least(8).ln()) / 4f64.ln();
        assert!((slope + 2.0).abs() < 0.3, "{slope}");
    }

    /// Without a gap and in the tunable phase, density settles near the
    /// paper's `ℓ₀ λ₀ ζ(α + ½)` — an estimate, good to about a tenth.
    #[test]
    fn density_follows_the_zeta_estimate() {
        for (alpha, lambda0) in [(0.7, 0.1), (0.9, 0.1), (1.2, 0.3)] {
            let g = ScaleRich::new(alpha, lambda0).steps(20_000).gap(0.0);
            let predicted = g.predicted_density().unwrap();
            let measured = (0..3)
                .map(|seed| g.clone().seed(seed).grow().relative_density())
                .sum::<f64>()
                / 3.0;
            assert!(
                (measured / predicted - 1.0).abs() < 0.15,
                "α {alpha}: {measured} vs {predicted}"
            );
        }
        assert_eq!(ScaleRich::new(0.5, 0.1).predicted_density(), None);
    }

    /// An L, and a disc with two holes: shapes a single convex void cannot
    /// start from.
    fn awkward_domains() -> Vec<ScaleRich> {
        let l_shape = vec![
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0],
        ];
        let circle = |cx: f64, cy: f64, r: f64| -> Vec<[f64; 2]> {
            (0..48)
                .map(|i| {
                    let a = TAU * i as f64 / 48.0;
                    [cx + r * a.cos(), cy + r * a.sin()]
                })
                .collect()
        };
        vec![
            ScaleRich::new(0.3, 0.06).domain(l_shape, vec![]),
            ScaleRich::new(0.5, 0.05).domain(
                circle(0.0, 0.0, 1.0),
                vec![circle(-0.4, 0.1, 0.25), circle(0.35, -0.2, 0.3)],
            ),
        ]
    }

    fn segment_distance(q: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len2 = dx * dx + dy * dy;
        let t = if len2 > 0.0 {
            (((q[0] - a[0]) * dx + (q[1] - a[1]) * dy) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (q[0] - a[0] - t * dx).hypot(q[1] - a[1] - t * dy)
    }

    fn near_ring(q: [f64; 2], ring: &[[f64; 2]]) -> f64 {
        (0..ring.len())
            .map(|i| segment_distance(q, ring[i], ring[(i + 1) % ring.len()]))
            .fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn awkward_domains_are_tiled_exactly() {
        for g in awkward_domains() {
            let net = g.steps(400).seed(5).grow();
            let domain: f64 = net.domain().iter().map(|r| shoelace(r)).sum();
            let solid: f64 = net.ligaments().iter().map(|l| l.area()).sum();
            let void: f64 = net.voids().iter().map(|v| shoelace(v)).sum();
            assert!(
                (solid + void - domain).abs() < 1e-10 * domain,
                "{solid} + {void} vs {domain}"
            );
            assert!(net
                .voids()
                .iter()
                .all(|v| shoelace(v) > 0.0 && is_convex(v)));
            assert!(net.relative_density() > 0.15, "{}", net.relative_density());
            // Some ligament crossed from one triangle into the next.
            assert!(net.ligaments().iter().any(|l| l.pieces.len() > 1));
        }
    }

    /// The triangles a non-convex domain starts as must not show: every
    /// ligament runs until it meets the domain's edge or a ligament laid
    /// before it, never a seam between triangles.
    #[test]
    fn ligaments_end_only_on_walls_or_material() {
        for g in awkward_domains() {
            let net = g.steps(300).seed(9).grow();
            let ligaments = net.ligaments();
            for (i, l) in ligaments.iter().enumerate() {
                for end in l.centreline {
                    let wall = net
                        .domain()
                        .iter()
                        .map(|r| near_ring(end, r))
                        .fold(f64::INFINITY, f64::min);
                    let material = ligaments[..i]
                        .iter()
                        .flat_map(|m| m.pieces.iter())
                        .map(|p| near_ring(end, p))
                        .fold(f64::INFINITY, f64::min);
                    assert!(
                        wall.min(material) < 1e-9,
                        "ligament {i} ends at {end:?}, {wall} from a wall and {material} from material"
                    );
                }
            }
        }
    }

    /// The outline's area is the ligaments', less the band the pores are
    /// grown by — at every scale, for every shape. The boolean's failures on
    /// near-touching polygons move with the scale, so one scale passing
    /// proves little; this is what caught them.
    #[test]
    fn outlines_hold_their_area_at_any_scale() {
        let mut networks = vec![ScaleRich::new(0.5, 0.06)];
        networks.extend(awkward_domains());
        for g in networks {
            for seed in 0..3 {
                let net = g.clone().steps(300).seed(seed).grow();
                let domain: f64 = net.domain().iter().map(|r| shoelace(r)).sum();
                let expected = net.relative_density() * domain;
                let perimeter: f64 = net
                    .voids()
                    .iter()
                    .map(|v| {
                        (0..v.len())
                            .map(|i| {
                                let (a, b) = (v[i], v[(i + 1) % v.len()]);
                                (b[0] - a[0]).hypot(b[1] - a[1])
                            })
                            .sum::<f64>()
                    })
                    .sum();
                let extent = net
                    .domain()
                    .iter()
                    .flatten()
                    .fold(0.0f64, |m, q| m.max(q[0].abs()).max(q[1].abs()));
                for bits in [24, 30, 38, 44] {
                    let unit = extent.log2().ceil().exp2() / 2f64.powi(bits);
                    let area: f64 = net
                        .outlines_at(bits)
                        .iter()
                        .flatten()
                        .map(|r| shoelace(r))
                        .sum();
                    // The growth's band, and a unit of rounding on top.
                    let band = perimeter * (PORE_GROWTH + 1.0) * unit;
                    assert!(
                        area <= expected + band && area >= expected - 2.0 * band,
                        "seed {seed}, 2^{bits}: {area} vs {expected} ± {band}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_convex_domain_is_one_void() {
        let hexagon: Vec<[f64; 2]> = (0..6)
            .map(|i| {
                let a = TAU * i as f64 / 6.0;
                [a.cos(), a.sin()]
            })
            .collect();
        assert_eq!(domain_cells(std::slice::from_ref(&hexagon)).len(), 1);
        let net = ScaleRich::new(0.3, 0.05).domain(hexagon, vec![]).grow();
        assert!(net.ligaments().iter().all(|l| l.pieces.len() == 1));
    }

    /// A ligament's direction is the field's at its nucleation point, and
    /// that point is on its centreline: with the angle a function of `x`
    /// alone, the `x` it implies must lie within the centreline's span.
    #[test]
    fn a_field_sets_the_directions() {
        let net = ScaleRich::new(0.3, 0.03)
            .angles(ScaleRichAngles::field(|p| {
                std::f64::consts::FRAC_PI_2 * p[0]
            }))
            .grow();
        // Neighbours nearly parallel jam sooner than random ones.
        assert!(net.ligaments().len() > 100, "{}", net.ligaments().len());
        for l in net.ligaments() {
            let x = l.angle / std::f64::consts::FRAC_PI_2;
            let [a, b] = l.centreline;
            assert!(
                a[0].min(b[0]) - 1e-9 <= x && x <= a[0].max(b[0]) + 1e-9,
                "{l:?}"
            );
        }
    }

    /// Seeding likelier on the right leaves the right with more pores.
    #[test]
    fn nucleation_grades_the_density() {
        let (mut left, mut right) = (0, 0);
        for seed in 0..3 {
            let net = ScaleRich::new(0.3, 0.03)
                .nucleation(|p| p[0] * p[0])
                .seed(seed)
                .grow();
            for v in net.voids() {
                let x = v.iter().map(|q| q[0]).sum::<f64>() / v.len() as f64;
                if x < 0.5 {
                    left += 1;
                } else {
                    right += 1;
                }
            }
        }
        assert!(right > 2 * left, "{left} pores left, {right} right");
    }

    #[test]
    fn zeta_matches_known_values() {
        assert!((zeta(2.0) - std::f64::consts::PI.powi(2) / 6.0).abs() < 1e-12);
        assert!((zeta(4.0) - std::f64::consts::PI.powi(4) / 90.0).abs() < 1e-12);
        assert!((zeta(1.5) - 2.612_375_348_685_488).abs() < 1e-10);
    }

    #[test]
    fn fenwick_picks_by_weight() {
        // Capacity 2, so the tree regrows on the way.
        let mut f = Fenwick::with_capacity(2);
        for w in [1.0, 0.0, 2.0, 3.0] {
            f.push(w);
        }
        assert_eq!(f.total(), 6.0);
        assert_eq!(f.find(0.5), 0);
        assert_eq!(f.find(1.5), 2);
        assert_eq!(f.find(2.999), 2);
        assert_eq!(f.find(3.0), 3);
        assert_eq!(f.find(5.9), 3);
    }
}
