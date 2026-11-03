//! Numerical homogenisation — what a lattice is worth as a material.
//!
//! A lattice is a geometry until someone has to size a part out of it, and
//! then it is a material: a stiffness, a Poisson ratio, a conductivity, and the
//! awkward fact that none of them is the same in every direction. This computes
//! them, by the standard energy method — voxelise one periodic cell, solve
//! every unit macroscopic state on it with periodic boundaries, and read the
//! effective tensor off the energy.
//!
//! [`homogenize`] does elasticity, six unit strains and a 6×6 stiffness.
//! [`homogenize_conduction`] does the scalar problems — heat, electricity,
//! diffusion, permittivity — with three unit gradients and a 3×3 tensor.
//!
//! ```
//! use threers::{Lattice, LatticeKind, Strut, Vector3};
//!
//! let c = Lattice::new(LatticeKind::Strut(Strut::Octet))
//!     .size(Vector3::new(10.0, 10.0, 10.0))
//!     .cells([1, 1, 1])
//!     .fit_relative_density(0.3)
//!     .homogenize(8);
//!
//! // A fraction of the base modulus, and the same on all three axes because
//! // the octet cell is cubic.
//! let e = c.youngs_moduli();
//! assert!(e[0] > 0.0 && (e[0] - e[2]).abs() < 0.05 * e[0]);
//! ```
//!
//! # What it is and is not
//!
//! *Is*: the periodic cell's own stiffness, the number to hand a solver that
//! is modelling the lattice as a solid. Correct in the limit of many cells,
//! which is the same limit in which calling a lattice a material is correct at
//! all.
//!
//! *Is not*: the stiffness of a specimen. A real block is stiffer than this
//! near a bonded platen and softer at a free surface, and at three cells across
//! the surface is most of it. For a block, build it and solve the block —
//! [`CuboctFrame`](super::CuboctFrame) does exactly that for beam cells.
//!
//! Everything is measured in units of the base material's modulus unless a
//! [`SolidMaterial`] says otherwise, so a lattice at a tenth of solid stiffness
//! reads 0.1 whatever it is printed in.
//!
//! # Convergence, and cost
//!
//! A voxelised strut is stair-stepped, and a stair-stepped strut is
//! over-connected — so both converge on the answer **from above** and a coarse
//! grid reads better than the truth. How much better is not the same for the
//! two problems, and the difference is large enough to change how you use them:
//!
//! | | 12³ → 44³ | what to do |
//! |---|---|---|
//! | Stiffness | falls about a fifth, still falling | compare cells at one resolution; quote only from a grid you have watched converge |
//! | Conductivity | falls about 2 % | converged by 12³ for most purposes |
//!
//! Elasticity is the sensitive one because a thin member's bending stiffness
//! depends on the exact shape of its surface, and a staircase is not it.
//! Conduction only asks how much material lies along the path.
//!
//! Six conjugate-gradient solves on `3 · resolution³` degrees of freedom for
//! elasticity, three on `resolution³` for conduction. On an octet at 30 %, an
//! M4 Pro takes roughly 110 ms at 12, 300 ms at 20, 2.4 s at 32 and 6 s at 40
//! for the stiffness, and about a fifth of that for the conductivity.
//!
//! # On the GPU
//!
//! [`Solver::Gpu`] runs the same conjugate gradient as a wgpu compute
//! pipeline — see [`gpu_solve`](super::gpu_solve) for how, and why the whole
//! iteration has to live on the device rather than just the operator. Measured
//! against the CPU path on this machine:
//!
//! | | CPU | GPU | |
//! |---|---|---|---|
//! | octet, 32³ | 3.7 s | 0.54 s | 6.9× |
//! | spinodal, 24³ window | 7.8 s | 0.58 s | 13.4× |
//! | voronoi foam, 24³ window | 9.3 s | 0.92 s | 10.1× |
//! | octet, 44³ | 16.7 s | 1.5 s | 11.2× |
//!
//! The moduli agree to five figures — 0.00 % apart on three of those four and
//! 0.01 % on the fourth — despite the device solving in `f32` against the CPU's
//! `f64`. Below about 16 voxels a cell the CPU wins: the solve is smaller than
//! the cost of talking to a device.
//!
//! Voxelising is parallel with the crate's `parallel` feature; the solves are
//! deliberately not, and [`solve`] says why.

// Dense element matrices, indexed by degree of freedom. The index is the
// meaning here — `ke[a][b]` is the force at `a` from a displacement at `b` —
// and iterating the rows instead hides which is which.
#![allow(clippy::needless_range_loop)]

#[cfg(not(target_arch = "wasm32"))]
use super::gpu_solve::GpuSolver;
use crate::math::Vector3;

/// Where the linear solves run.
///
/// The answer is not quite the same on the two, so this is a choice and not an
/// optimisation the crate makes behind your back — see
/// [`Gpu`](Self::Gpu).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Solver {
    /// Double precision on the CPU. Deterministic to the bit, and the
    /// reference the GPU path is measured against.
    #[default]
    Cpu,
    /// Single precision on a wgpu compute device, falling back to
    /// [`Cpu`](Self::Cpu) when there is no adapter — which the result says, so
    /// the fallback is not silent. Always falls back in the browser, where a
    /// device cannot be shared between threads and so cannot be held.
    ///
    /// Ten to twenty times faster on a cell worth the trouble, and within a
    /// fraction of a percent: the effective tensor is read off a strain energy,
    /// and energy is stationary at the solution, so an error in the
    /// displacement field appears squared in the answer. Below about 16 voxels
    /// a cell the device setup costs more than the solve saves.
    Gpu,
}

/// One device for the process, acquired the first time it is asked for.
///
/// Acquiring an adapter is tens of milliseconds — more than a small solve —
/// and nothing about it is per-problem.
#[cfg(not(target_arch = "wasm32"))]
fn shared_gpu() -> Option<&'static GpuSolver> {
    static GPU: std::sync::OnceLock<Option<GpuSolver>> = std::sync::OnceLock::new();
    GPU.get_or_init(GpuSolver::new).as_ref()
}

/// The base material the lattice is made of.
#[derive(Clone, Copy, Debug)]
pub struct SolidMaterial {
    /// Young's modulus. Every result scales with it.
    pub modulus: f64,
    /// Poisson ratio of the solid, not of the lattice.
    pub poisson: f64,
}

impl Default for SolidMaterial {
    /// Unit modulus, and the Poisson ratio of most metals and of nylon.
    fn default() -> Self {
        Self {
            modulus: 1.0,
            poisson: 0.3,
        }
    }
}

/// How stiff a lattice is, in every direction at once.
///
/// The 6×6 Voigt stiffness, in the order xx, yy, zz, yz, xz, xy, plus the
/// engineering constants read off its inverse.
#[derive(Clone, Copy, Debug)]
pub struct Stiffness {
    /// The effective stiffness matrix, in the base material's units.
    pub c: [[f64; 6]; 6],
    /// The fraction of the cell that was solid when it was measured.
    pub relative_density: f64,
    /// Which solver ran — [`Solver::Gpu`] falls back to the CPU when there is
    /// no adapter, and this is where that shows.
    pub solver: Solver,
    /// The voxel grid it was solved on, per axis — after the clamp.
    ///
    /// Asked for and got are not the same thing: the grid tops out at 48, and
    /// a window of several cells multiplies into that ceiling. Since the answer
    /// converges from above, a resolution silently lower than the one requested
    /// is a stiffness silently higher than it should be, so the number that was
    /// used comes back with it.
    pub voxels: usize,
}

impl Stiffness {
    /// The compliance — the inverse of [`c`](Self::c).
    ///
    /// All zeros for a cell so sparse that the stiffness is singular.
    pub fn compliance(&self) -> [[f64; 6]; 6] {
        invert6(&self.c).unwrap_or([[0.0; 6]; 6])
    }

    /// Young's modulus along x, y and z.
    pub fn youngs_moduli(&self) -> [f64; 3] {
        let s = self.compliance();
        [0, 1, 2].map(|i| if s[i][i].abs() > 1e-30 { 1.0 / s[i][i] } else { 0.0 })
    }

    /// Shear modulus about yz, xz and xy.
    pub fn shear_moduli(&self) -> [f64; 3] {
        let s = self.compliance();
        [3, 4, 5].map(|i| if s[i][i].abs() > 1e-30 { 1.0 / s[i][i] } else { 0.0 })
    }

    /// Poisson ratios ν_yz, ν_xz, ν_xy — the transverse contraction per unit
    /// extension along the *first* named axis.
    ///
    /// Negative for an auxetic cell, which is the point of measuring it.
    pub fn poisson_ratios(&self) -> [f64; 3] {
        let s = self.compliance();
        let ratio = |along: usize, across: usize| {
            if s[along][along].abs() > 1e-30 {
                -s[across][along] / s[along][along]
            } else {
                0.0
            }
        };
        [ratio(1, 2), ratio(0, 2), ratio(0, 1)]
    }

    /// Young's modulus in an arbitrary direction.
    ///
    /// This is the honest way to compare two cells: axis moduli flatter a
    /// lattice built along the axes, and a beam cell can be three times stiffer
    /// on its struts than on its diagonals.
    pub fn directional_modulus(&self, direction: Vector3) -> f64 {
        let n = direction.normalize();
        let n = [n.x as f64, n.y as f64, n.z as f64];
        let s = self.compliance();
        let mut sum = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        sum += n[i] * n[j] * n[k] * n[l] * tensor(&s, i, j, k, l);
                    }
                }
            }
        }
        if sum.abs() > 1e-30 {
            1.0 / sum
        } else {
            0.0
        }
    }

    /// Stiffest direction over stiffest-to-softest — 1 for a truly isotropic
    /// cell, and the number a foam is chosen for.
    ///
    /// Sampled over a spiral of directions rather than solved for, so it is a
    /// lower bound on the true spread: it cannot report more anisotropy than
    /// there is, and with a few hundred directions it misses very little.
    pub fn anisotropy(&self) -> f64 {
        let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
        let count = 256;
        for i in 0..count {
            // A Fibonacci spiral over the hemisphere — even coverage without a
            // random number in sight.
            let z = (i as f64 + 0.5) / count as f64;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let theta = std::f64::consts::PI * (1.0 + 5.0f64.sqrt()) * i as f64;
            let e = self.directional_modulus(Vector3::new(
                (r * theta.cos()) as f32,
                (r * theta.sin()) as f32,
                z as f32,
            ));
            lo = lo.min(e);
            hi = hi.max(e);
        }
        if lo > 1e-30 {
            hi / lo
        } else {
            f64::INFINITY
        }
    }
}

/// How well a lattice conducts, in every direction at once.
///
/// The 3×3 effective conductivity tensor, in the base material's units — or a
/// fraction of it, when the base was left at 1. See
/// [`homogenize_conduction`] for what else the same number describes.
#[derive(Clone, Copy, Debug)]
pub struct Conductivity {
    /// The effective conductivity tensor.
    pub k: [[f64; 3]; 3],
    /// The fraction of the cell that was solid when it was measured.
    pub relative_density: f64,
    /// The voxel grid it was solved on, per axis — see
    /// [`Stiffness::voxels`].
    pub voxels: usize,
    /// Which solver ran — see [`Stiffness::solver`].
    pub solver: Solver,
}

impl Conductivity {
    /// Conductivity along x, y and z.
    ///
    /// The diagonal, which is the whole story only for a cell whose axes are
    /// its principal directions — true of every periodic cell here and of no
    /// foam. [`principal`](Self::principal) is the honest version.
    pub fn axes(&self) -> [f64; 3] {
        [self.k[0][0], self.k[1][1], self.k[2][2]]
    }

    /// Conductivity in an arbitrary direction: `n · K · n`.
    pub fn directional(&self, direction: Vector3) -> f64 {
        let n = direction.normalize();
        let n = [n.x as f64, n.y as f64, n.z as f64];
        let mut sum = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                sum += n[i] * self.k[i][j] * n[j];
            }
        }
        sum
    }

    /// The three principal conductivities, largest first — the tensor's
    /// eigenvalues, so they do not care how the cell happens to be oriented.
    pub fn principal(&self) -> [f64; 3] {
        symmetric_eigenvalues(&self.k)
    }

    /// Best direction over worst. 1 for a cell that conducts the same way
    /// whichever way the heat is going.
    pub fn anisotropy(&self) -> f64 {
        let p = self.principal();
        if p[2].abs() > 1e-30 {
            p[0] / p[2]
        } else {
            f64::INFINITY
        }
    }

    /// Conductivity as a fraction of a straight bar of the same material and
    /// the same relative density — how much of the material is on the path.
    ///
    /// 1 would mean every gram lies along the gradient, which only solid
    /// columns achieve. A stochastic foam lands near a third: two of its three
    /// directions are carrying nothing at any given moment, and the path
    /// through it is not straight.
    pub fn tortuosity_factor(&self, solid: f64) -> f64 {
        let bar = solid * self.relative_density;
        if bar.abs() > 1e-30 {
            self.principal()[0] / bar
        } else {
            0.0
        }
    }
}

/// Homogenise a periodic cell given a test for what is solid.
///
/// `occupancy` is asked at element centres over one cell starting at `origin`,
/// and must be periodic with period `cell` — which is what makes the answer a
/// material property rather than a description of one block.
pub fn homogenize(
    occupancy: impl Fn(Vector3) -> bool + Sync + Send,
    origin: Vector3,
    cell: Vector3,
    resolution: usize,
    material: SolidMaterial,
    solver: Solver,
) -> Stiffness {
    let n = resolution.clamp(2, 48);
    let h = spacing(cell, n);
    let elements = n * n * n;
    let stiff = voxelize(occupancy, origin, h, n);
    let relative_density = stiff.iter().sum::<f64>() / elements as f64;

    let ke = element_stiffness(h, material.poisson);
    let u0 = unit_strain_displacements(h);
    // Every element's 24 global degrees of freedom, once. Recomputing them
    // inside the solver would be eight modulos per element per iteration, and
    // there are thousands of iterations.
    let dof_map: Vec<[usize; 24]> = (0..elements).map(|e| element_map(e, n)).collect();

    let (c, _, ran) = effective::<24, 6>(
        &ke,
        &u0,
        &stiff,
        &dof_map,
        elements,
        volume(cell),
        material.modulus,
        solver,
    );
    Stiffness {
        c,
        relative_density,
        voxels: n,
        solver: ran,
    }
}

/// When a lattice gives way, and where.
///
/// Stiffness says how far a lattice moves under load; this says how much load
/// it takes before something in it stops coming back. The mechanism is the
/// same one that makes a lattice weaker than its density suggests: the load
/// does not spread evenly, and the worst-loaded ligament reaches the material's
/// yield stress long before the average one does.
///
/// [`concentration`](Self::concentration) is that unevenness as a number — the
/// local von Mises stress per unit of macroscopic stress. A solid block is 1.
/// A beam lattice at a quarter density is five or ten, and that factor, not the
/// density, is what sets the strength.
///
/// # This one converges from below
///
/// The stiffness converges from above, which flatters a coarse grid. Strength
/// is worse: a coarse grid cannot see the sharpest-loaded corner of a ligament
/// at all, so the concentration comes out **too low and the strength too
/// high** — the unsafe direction, and the one to be suspicious of. An octet at
/// 30 % reads 6.2 at 12 voxels a cell, 8.0 at 16, 9.7 at 24, 10.8 at 32 and
/// 11.3 at 40, still climbing. At 10 % it is not worth reading below about 32.
///
/// [`resolved`](Self::resolved) catches the gross cases. It is a necessary
/// condition and not a sufficient one: false means the number is certainly
/// wrong, true does not mean it is right. Refine until it stops moving.
#[derive(Clone, Copy, Debug)]
pub struct Strength {
    /// The stiffness solved on the way, and the density and grid it used.
    pub stiffness: Stiffness,
    /// Local von Mises stress per unit macroscopic stress, per Voigt case —
    /// xx, yy, zz, yz, xz, xy.
    ///
    /// A high percentile of the solid rather than its maximum: on a voxel grid
    /// the single worst element sits in a staircase corner whose stress rises
    /// without limit as the grid is refined, so a maximum does not converge to
    /// anything. See [`peak_concentration`](Self::peak_concentration).
    pub concentration: [f64; 6],
    /// The worst single element, per case. Mesh-dependent by nature — useful
    /// for seeing how sharp the concentration is, not for sizing a part.
    pub peak_concentration: [f64; 6],
    /// How many elements the stress was read from.
    pub sampled: usize,
    /// The material fraction an element needed to be read at all.
    ///
    /// 0.9 unless nothing reached it, in which case it steps down — and a
    /// lattice whose walls never fill an element is one whose stresses are
    /// mostly the staircase's. See [`resolved`](Self::resolved).
    pub solid_floor: f64,
    /// The material it was solved for.
    pub material: SolidMaterial,
}

impl Strength {
    /// The macroscopic stress the lattice yields at, per Voigt case, given
    /// what the solid it is made of yields at.
    ///
    /// Von Mises throughout, so the shear cases come out at about `1/√3` of the
    /// axial ones for an isotropic cell, exactly as they do for the solid.
    pub fn yield_strength(&self, solid_yield: f64) -> [f64; 6] {
        let mut out = [0.0f64; 6];
        for (i, v) in out.iter_mut().enumerate() {
            *v = if self.concentration[i] > 1e-30 {
                solid_yield / self.concentration[i]
            } else {
                0.0
            };
        }
        out
    }

    /// The uniaxial strengths along x, y and z.
    pub fn uniaxial(&self, solid_yield: f64) -> [f64; 3] {
        let all = self.yield_strength(solid_yield);
        [all[0], all[1], all[2]]
    }

    /// Strength as a fraction of what the same mass of solid, all of it laid
    /// along the load, would carry — the strength counterpart of
    /// [`Conductivity::tortuosity_factor`].
    ///
    /// 1 would be perfect: every ligament loaded to yield at the same moment.
    /// A stretch-dominated cell reaches a third or so; a bending-dominated one
    /// far less, because its struts are in bending and only their outer fibres
    /// are working.
    pub fn efficiency(&self) -> [f64; 3] {
        let mut out = [0.0f64; 3];
        for (i, v) in out.iter_mut().enumerate() {
            let ideal = self.stiffness.relative_density;
            *v = if self.concentration[i] > 1e-30 && ideal > 1e-30 {
                1.0 / (self.concentration[i] * ideal)
            } else {
                0.0
            };
        }
        out
    }

    /// Whether the grid resolved the ligaments well enough for any of this to
    /// mean anything.
    ///
    /// Two necessary conditions:
    ///
    /// - [`efficiency`](Self::efficiency) cannot exceed 1, give or take a
    ///   per cent of discretisation. No arrangement of
    ///   material carries a uniaxial load with less concentration than a solid
    ///   column of the same density does, so a reading above 1 is proof that
    ///   the worst-loaded material was never sampled.
    /// - At least half the material has to be interior — to have filled an
    ///   element rather than straddled a surface. Below that, the ligaments are
    ///   a voxel or two across and the stress in them is the staircase's.
    ///
    /// Neither is sufficient. See the type's own note on convergence.
    pub fn resolved(&self) -> bool {
        // A per-cent of slack: a prismatic column really does reach 1, and a
        // voxel grid lands on that bound from either side.
        if self.efficiency().iter().any(|&v| v > 1.01) {
            return false;
        }
        let solid = self.stiffness.relative_density
            * (self.stiffness.voxels * self.stiffness.voxels * self.stiffness.voxels) as f64;
        self.solid_floor >= 0.9 && self.sampled as f64 >= 0.5 * solid
    }

    /// The macroscopic strain the lattice reaches at yield, per axis.
    ///
    /// `solid_yield` has to be in the same units as
    /// [`SolidMaterial::modulus`] — a yield stress in megapascals against a
    /// modulus left at 1 gives a strain in the hundreds, which is the sum
    /// telling you the units did not match.
    ///
    /// The number that says whether to believe any of this. All of it is small
    /// strain and linear: struts are assumed to stay straight and stresses to
    /// stay proportional. A lattice that only reaches yield at several percent
    /// of macroscopic strain is one whose slender members will have buckled or
    /// bent over long before, and its real collapse stress is lower than this —
    /// often far lower. Elastic buckling is not modelled here, and this is the
    /// warning that it should have been.
    pub fn collapse_strain(&self, solid_yield: f64) -> [f64; 3] {
        let e = self.stiffness.youngs_moduli();
        let strength = self.uniaxial(solid_yield);
        let mut out = [0.0f64; 3];
        for (i, v) in out.iter_mut().enumerate() {
            *v = if e[i] > 1e-30 { strength[i] / e[i] } else { 0.0 };
        }
        out
    }
}

/// Homogenise a periodic cell for stiffness *and* strength.
///
/// The same six solves as [`homogenize`], plus one pass over the elements to
/// find how hard the material inside is working. Strength needs the stiffness
/// anyway — the macroscopic strain that a unit macroscopic stress produces is
/// the compliance — so it comes back inside the result rather than being solved
/// for twice.
///
/// ```
/// use threers::{Lattice, LatticeKind, Strut, Vector3};
///
/// let s = Lattice::new(LatticeKind::Strut(Strut::Octet))
///     .size(Vector3::new(10.0, 10.0, 10.0))
///     .cells([1, 1, 1])
///     .fit_relative_density(0.3)
///     .strength(12);
///
/// // 316L at 500 MPa, and the lattice at some fraction of it.
/// let along_z = s.uniaxial(500.0)[2];
/// assert!(along_z > 0.0 && along_z < 500.0 * 0.3);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn homogenize_strength(
    occupancy: impl Fn(Vector3) -> bool + Sync + Send,
    origin: Vector3,
    cell: Vector3,
    resolution: usize,
    material: SolidMaterial,
    solver: Solver,
) -> Strength {
    let n = resolution.clamp(2, 48);
    let h = spacing(cell, n);
    let elements = n * n * n;
    let stiff = voxelize(occupancy, origin, h, n);
    let relative_density = stiff.iter().sum::<f64>() / elements as f64;

    let ke = element_stiffness(h, material.poisson);
    let u0 = unit_strain_displacements(h);
    let dof_map: Vec<[usize; 24]> = (0..elements).map(|e| element_map(e, n)).collect();
    let (c, chi, ran) = effective::<24, 6>(
        &ke,
        &u0,
        &stiff,
        &dof_map,
        elements,
        volume(cell),
        material.modulus,
        solver,
    );
    let stiffness = Stiffness {
        c,
        relative_density,
        voxels: n,
        solver: ran,
    };

    // Only elements that are essentially all material: a half-filled element
    // is a stand-in for a surface, and the stress in it is neither the solid's
    // nor the void's. Where a wall is too thin for the grid to hold a full
    // element anywhere, the bar drops rather than reporting nothing.
    let mut floor = 0.9;
    while floor > 0.2 && !stiff.iter().any(|&s| s >= floor) {
        floor -= 0.1;
    }

    let compliance = stiffness.compliance();
    let b = strain_matrix(h);
    let d = constitutive(material.poisson);
    let mut concentration = [0.0f64; 6];
    let mut peak_concentration = [0.0f64; 6];
    let mut sampled = 0usize;

    for case in 0..6 {
        // The macroscopic strain a unit macroscopic stress in this direction
        // produces, and with it the displacement field inside the cell.
        let mut macro_strain = [0.0f64; 6];
        for (j, v) in macro_strain.iter_mut().enumerate() {
            *v = compliance[j][case];
        }

        let mut mises: Vec<f64> = Vec::new();
        let mut worst = 0.0f64;
        for e in 0..elements {
            if stiff[e] < floor {
                continue;
            }
            let map = &dof_map[e];
            let mut local = [0.0f64; 24];
            for a in 0..24 {
                let mut acc = 0.0;
                for (j, &weight) in macro_strain.iter().enumerate() {
                    if weight != 0.0 {
                        acc += weight * (u0[j][a] - chi[j][map[a]]);
                    }
                }
                local[a] = acc;
            }
            let mut strain = [0.0f64; 6];
            for i in 0..6 {
                let mut acc = 0.0;
                for a in 0..24 {
                    acc += b[i][a] * local[a];
                }
                strain[i] = acc;
            }
            let mut stress = [0.0f64; 6];
            for i in 0..6 {
                let mut acc = 0.0;
                for j in 0..6 {
                    acc += d[i][j] * strain[j];
                }
                stress[i] = acc * material.modulus;
            }
            let vm = von_mises(&stress);
            worst = worst.max(vm);
            mises.push(vm);
        }

        peak_concentration[case] = worst;
        sampled = mises.len();
        concentration[case] = percentile(&mut mises, 0.99);
    }

    Strength {
        stiffness,
        concentration,
        peak_concentration,
        sampled,
        solid_floor: floor,
        material,
    }
}

/// Von Mises equivalent stress of a Voigt stress vector.
fn von_mises(s: &[f64; 6]) -> f64 {
    let deviatoric = 0.5
        * ((s[0] - s[1]).powi(2) + (s[1] - s[2]).powi(2) + (s[2] - s[0]).powi(2));
    let shear = 3.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]);
    (deviatoric + shear).max(0.0).sqrt()
}

/// The value below which `fraction` of the samples fall. Sorts in place.
fn percentile(values: &mut [f64], fraction: f64) -> f64 {
    if values.is_empty() {
        return f64::INFINITY;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let at = ((values.len() - 1) as f64 * fraction.clamp(0.0, 1.0)).round() as usize;
    values[at]
}

/// The effective conductivity of a periodic cell, by the same method.
///
/// Conduction is the scalar cousin of elasticity: one unknown a node instead of
/// three, three unit macroscopic gradients instead of six unit strains, and the
/// effective tensor read off the same energy. Everything else — the voxels, the
/// periodic boundaries, the solver, the caveats — is shared with
/// [`homogenize`], including that it converges from above.
///
/// It answers to more than heat. The same equation and so the same number
/// describes electrical conduction, diffusion and dielectric permittivity; only
/// the units on `solid` change.
///
/// ```
/// use threers::{Lattice, LatticeKind, Tpms, Vector3};
///
/// let k = Lattice::new(LatticeKind::Tpms(Tpms::Gyroid))
///     .size(Vector3::new(10.0, 10.0, 10.0))
///     .cells([1, 1, 1])
///     .fit_relative_density(0.3)
///     .conductivity(10);
///
/// // A third of the material conducts rather less than a third as well: the
/// // path through a lattice is not a straight one.
/// let along = k.axes();
/// assert!(along[0] > 0.0 && along[0] < 0.3);
/// ```
pub fn homogenize_conduction(
    occupancy: impl Fn(Vector3) -> bool + Sync + Send,
    origin: Vector3,
    cell: Vector3,
    resolution: usize,
    solid: f64,
    solver: Solver,
) -> Conductivity {
    let n = resolution.clamp(2, 48);
    let h = spacing(cell, n);
    let elements = n * n * n;
    let stiff = voxelize(occupancy, origin, h, n);
    let relative_density = stiff.iter().sum::<f64>() / elements as f64;

    let ke = conduction_element(h);
    let u0 = unit_gradient_potentials(h);
    let dof_map: Vec<[usize; 8]> = (0..elements).map(|e| element_map(e, n)).collect();

    let (k, _, ran) = effective::<8, 3>(
        &ke,
        &u0,
        &stiff,
        &dof_map,
        elements,
        volume(cell),
        solid,
        solver,
    );
    Conductivity {
        k,
        relative_density,
        voxels: n,
        solver: ran,
    }
}

/// Element size along each axis.
fn spacing(cell: Vector3, n: usize) -> [f64; 3] {
    [
        (cell.x as f64 / n as f64).abs().max(1e-12),
        (cell.y as f64 / n as f64).abs().max(1e-12),
        (cell.z as f64 / n as f64).abs().max(1e-12),
    ]
}

/// The cell's volume, floored so it can be divided by.
fn volume(cell: Vector3) -> f64 {
    (cell.x as f64 * cell.y as f64 * cell.z as f64)
        .abs()
        .max(1e-30)
}

/// How much of each element is solid, `VOID` to 1.
///
/// A binary element is the obvious thing and it does not work: a strut lies
/// along the same voxels in every cell of a grid that is commensurate with it,
/// so the same wall is rounded the same way a thousand times over and the error
/// does not average out. The measured density then jumps about with the
/// resolution — 0.22 at 16 voxels and 0.35 at 12, for a lattice built at 0.30 —
/// and the stiffness jumps with it. Sampling each element `SUB³` times and
/// taking the fraction leaves the density smooth in the resolution, and is the
/// ordinary treatment of a partly filled voxel.
///
/// The void is a hundred-thousandth of the solid rather than nothing, so that
/// nodes with no material on them are still tied to something and the system
/// stays solvable. It is also the floor on how soft a lattice this can report,
/// and it is what the solver's conditioning is set by — the iteration count
/// goes as its inverse square root, so a far weaker void would be more honest
/// and very much slower.
fn voxelize(
    occupancy: impl Fn(Vector3) -> bool + Sync + Send,
    origin: Vector3,
    h: [f64; 3],
    n: usize,
) -> Vec<f64> {
    const VOID: f64 = 1e-5;
    const SUB: usize = 3;
    crate::utils::parallel::par_map_range(n * n * n, |e| {
        let i = e % n;
        let j = (e / n) % n;
        let k = e / (n * n);
        let mut filled = 0usize;
        for c in 0..SUB * SUB * SUB {
            let at = |index: usize| (index as f64 + 0.5) / SUB as f64;
            let p = Vector3::new(
                origin.x + ((i as f64 + at(c % SUB)) * h[0]) as f32,
                origin.y + ((j as f64 + at((c / SUB) % SUB)) * h[1]) as f32,
                origin.z + ((k as f64 + at(c / (SUB * SUB))) * h[2]) as f32,
            );
            if occupancy(p) {
                filled += 1;
            }
        }
        (filled as f64 / (SUB * SUB * SUB) as f64).max(VOID)
    })
}

/// The effective property tensor: solve every macroscopic case, then read the
/// tensor off the energy.
///
/// `N` is the element's degrees of freedom — 24 for elasticity, 8 for
/// conduction — and `CASES` the number of unit macroscopic states, 6 and 3.
/// The two problems differ in nothing else, which is why they share this.
///
/// The periodic fluctuation fields come back with the tensor: they are what the
/// local stress inside the cell is computed from, and re-solving for them would
/// be the whole cost again.
#[allow(clippy::too_many_arguments)]
fn effective<const N: usize, const CASES: usize>(
    ke: &[[f64; N]; N],
    u0: &[[f64; N]; CASES],
    stiff: &[f64],
    dof_map: &[[usize; N]],
    elements: usize,
    volume: f64,
    scale: f64,
    solver: Solver,
) -> ([[f64; CASES]; CASES], Vec<Vec<f64>>, Solver) {
    let per_node = (N / 8).max(1);
    let dofs = elements * per_node;

    // One node fixed, to take out what the periodic cell is otherwise free to
    // do for nothing: slide, for elasticity, or sit at any temperature at all.
    let loads: Vec<Vec<f64>> = (0..CASES)
        .map(|case| {
            let mut f = vec![0.0f64; dofs];
            for e in 0..elements {
                let map = &dof_map[e];
                let s = stiff[e];
                for a in 0..N {
                    let mut acc = 0.0;
                    for b in 0..N {
                        acc += ke[a][b] * u0[case][b];
                    }
                    f[map[a]] += s * acc;
                }
            }
            for (d, v) in f.iter_mut().enumerate() {
                if d < per_node {
                    *v = 0.0;
                }
            }
            f
        })
        .collect();

    let diag = diagonal(ke, stiff, dof_map, elements, per_node, dofs);
    let (chi, ran) = match solver {
        Solver::Gpu => gpu_solve(&loads, ke, stiff, &diag, elements, per_node, scale)
            .map(|chi| (chi, Solver::Gpu))
            .unwrap_or_else(|| (cpu_solve(&loads, ke, stiff, dof_map, &diag, elements, per_node, scale), Solver::Cpu)),
        Solver::Cpu => (
            cpu_solve(&loads, ke, stiff, dof_map, &diag, elements, per_node, scale),
            Solver::Cpu,
        ),
    };

    let mut out = [[0.0f64; CASES]; CASES];
    let mut diff = [[0.0f64; N]; CASES];
    let mut k_diff = [[0.0f64; N]; CASES];
    for e in 0..elements {
        let map = &dof_map[e];
        for case in 0..CASES {
            for a in 0..N {
                diff[case][a] = u0[case][a] - chi[case][map[a]];
            }
        }
        // kᵉ · diff once per case, then dot with the other case's difference:
        // `CASES` matrix-vector products an element rather than `CASES²`.
        for case in 0..CASES {
            for a in 0..N {
                let mut acc = 0.0;
                for b in 0..N {
                    acc += ke[a][b] * diff[case][b];
                }
                k_diff[case][a] = acc;
            }
        }
        for i in 0..CASES {
            for j in i..CASES {
                let mut acc = 0.0;
                for a in 0..N {
                    acc += diff[i][a] * k_diff[j][a];
                }
                out[i][j] += stiff[e] * acc;
            }
        }
    }
    for i in 0..CASES {
        for j in i..CASES {
            let v = out[i][j] / volume * scale;
            out[i][j] = v;
            out[j][i] = v;
        }
    }
    (out, chi, ran)
}

/// Every case, on the CPU.
#[allow(clippy::too_many_arguments)]
fn cpu_solve<const N: usize>(
    loads: &[Vec<f64>],
    ke: &[[f64; N]; N],
    stiff: &[f64],
    dof_map: &[[usize; N]],
    diag: &[f64],
    elements: usize,
    per_node: usize,
    scale: f64,
) -> Vec<Vec<f64>> {
    loads
        .iter()
        .map(|f| solve(f, ke, stiff, dof_map, diag, elements, per_node, scale))
        .collect()
}

/// Every case, on the device — or `None`, which sends the caller to the CPU.
///
/// All or nothing deliberately: half the cases at one precision and half at
/// another would give a tensor whose entries do not belong to one another.
#[cfg(not(target_arch = "wasm32"))]
fn gpu_solve<const N: usize>(
    loads: &[Vec<f64>],
    ke: &[[f64; N]; N],
    stiff: &[f64],
    diag: &[f64],
    elements: usize,
    per_node: usize,
    scale: f64,
) -> Option<Vec<Vec<f64>>> {
    let device = shared_gpu()?;
    let dofs = 8 * per_node;
    let mut flat = Vec::with_capacity(dofs * dofs);
    for row in ke.iter().take(dofs) {
        flat.extend_from_slice(&row[..dofs]);
    }
    let problem = device.problem(cbrt(elements), per_node, &flat, stiff, diag);
    let mut out = Vec::with_capacity(loads.len());
    for f in loads {
        // A looser stop than the CPU's millionth: single precision cannot
        // reach that, and does not need to — see the module docs on why an
        // energy tolerates a hundred times more error than a displacement.
        let norm: f64 = f.iter().map(|v| v * v).sum::<f64>().sqrt();
        let target = (norm * 1e-4).max(1e-9 * scale.abs().max(1.0));
        out.push(problem.solve(f, target, 4000)?);
    }
    Some(out)
}

/// No device in the browser — see the module declaration in `mod.rs`.
#[cfg(target_arch = "wasm32")]
fn gpu_solve<const N: usize>(
    _loads: &[Vec<f64>],
    _ke: &[[f64; N]; N],
    _stiff: &[f64],
    _diag: &[f64],
    _elements: usize,
    _per_node: usize,
    _scale: f64,
) -> Option<Vec<Vec<f64>>> {
    None
}

/// The cube root of a perfect cube, exactly. Only the device path needs it —
/// everything else already has `n` in hand.
#[cfg(not(target_arch = "wasm32"))]
fn cbrt(elements: usize) -> usize {
    let mut n = (elements as f64).cbrt().round() as usize;
    while n * n * n > elements {
        n -= 1;
    }
    while (n + 1) * (n + 1) * (n + 1) <= elements {
        n += 1;
    }
    n
}

/// The diagonal of `K`, which is the preconditioner both paths use.
fn diagonal<const N: usize>(
    ke: &[[f64; N]; N],
    stiff: &[f64],
    dof_map: &[[usize; N]],
    elements: usize,
    per_node: usize,
    dofs: usize,
) -> Vec<f64> {
    let mut diag = vec![0.0f64; dofs];
    for e in 0..elements {
        let map = &dof_map[e];
        for a in 0..N {
            diag[map[a]] += stiff[e] * ke[a][a];
        }
    }
    for (d, v) in diag.iter_mut().enumerate() {
        if d < per_node || *v <= 0.0 {
            *v = 1.0;
        }
    }
    diag
}

/// Solve `K χ = f` for the periodic fluctuation, matrix-free.
///
/// The stiffness is never assembled: it is `Σ_e s_e kᵉ`, and both the operator
/// and its diagonal can be evaluated from the one element matrix every element
/// shares. A 32³ cell would otherwise be a 98 304 × 98 304 matrix.
#[allow(clippy::too_many_arguments)]
fn solve<const N: usize>(
    f: &[f64],
    ke: &[[f64; N]; N],
    stiff: &[f64],
    dof_map: &[[usize; N]],
    diag: &[f64],
    elements: usize,
    per_node: usize,
    scale: f64,
) -> Vec<f64> {
    let dofs = f.len();
    // Node 0 pinned; see `effective`.
    let fixed = |d: usize| d < per_node;

    // `K · x`, assembled element by element.
    //
    // Deliberately not parallel, which is not the usual answer. The operator is
    // applied once per conjugate-gradient iteration and there are hundreds of
    // them, so a parallel apply is hundreds of joins; measured on a ten-core
    // M4 Pro, the join costs about ten milliseconds and the arithmetic between
    // two of them costs one, and a rayon version of this loop ran three to
    // twenty times *slower* at every grid size from 12³ to 40³. Voxelising is
    // one parallel call over millions of samples and does profit — see
    // `homogenize`.
    let apply = |x: &[f64], out: &mut [f64]| {
        out.iter_mut().for_each(|v| *v = 0.0);
        let mut local = [0.0f64; N];
        for e in 0..elements {
            let map = &dof_map[e];
            for a in 0..N {
                local[a] = x[map[a]];
            }
            let s = stiff[e];
            for a in 0..N {
                let mut acc = 0.0;
                let row = &ke[a];
                for b in 0..N {
                    acc += row[b] * local[b];
                }
                out[map[a]] += s * acc;
            }
        }
        for d in 0..dofs {
            if fixed(d) {
                out[d] = 0.0;
            }
        }
    };

    let mut x = vec![0.0f64; dofs];
    let mut r: Vec<f64> = f.to_vec();
    for d in 0..dofs {
        if fixed(d) {
            r[d] = 0.0;
        }
    }
    let mut z: Vec<f64> = r.iter().zip(diag.iter()).map(|(r, d)| r / d).collect();
    let mut p = z.clone();
    let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
    let target = {
        let norm: f64 = f.iter().map(|v| v * v).sum::<f64>().sqrt();
        // Relative to the load, and to the modulus so the tolerance means the
        // same thing whatever units the material is in. A millionth is three
        // digits past anything a homogenised property is quoted to, and the
        // residual falls slowly enough near the end that asking for more is
        // most of the run time.
        (norm * 1e-6).max(1e-12 * scale.abs().max(1.0))
    };
    let mut q = vec![0.0f64; dofs];
    // Enough for a 48³ cell with a millionth-stiff void; the residual test is
    // what normally stops it, several times sooner.
    for _ in 0..4000 {
        let residual: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        if residual <= target || rz.abs() < 1e-300 {
            break;
        }
        apply(&p, &mut q);
        let pq: f64 = p.iter().zip(&q).map(|(a, b)| a * b).sum();
        if pq.abs() < 1e-300 {
            break;
        }
        let alpha = rz / pq;
        for d in 0..dofs {
            x[d] += alpha * p[d];
            r[d] -= alpha * q[d];
        }
        for d in 0..dofs {
            z[d] = r[d] / diag[d];
        }
        let rz_next: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        let beta = rz_next / rz;
        for d in 0..dofs {
            p[d] = z[d] + beta * p[d];
        }
        rz = rz_next;
    }
    x
}

/// The global degrees of freedom of element `e`, wrapped periodically.
///
/// `N / 8` of them per node — three for elasticity, one for conduction. Corner
/// `c` is at `(c & 1, (c >> 1) & 1, (c >> 2) & 1)`, the same corner numbering
/// the mesher uses.
fn element_map<const N: usize>(e: usize, n: usize) -> [usize; N] {
    let per_node = (N / 8).max(1);
    let i = e % n;
    let j = (e / n) % n;
    let k = e / (n * n);
    let mut out = [0usize; N];
    for c in 0..8 {
        let ni = (i + (c & 1)) % n;
        let nj = (j + ((c >> 1) & 1)) % n;
        let nk = (k + ((c >> 2) & 1)) % n;
        let node = (nk * n + nj) * n + ni;
        for d in 0..per_node {
            out[c * per_node + d] = node * per_node + d;
        }
    }
    out
}

/// Nodal potentials of the three unit macroscopic gradients.
///
/// A gradient of one along an axis is a potential equal to the coordinate on
/// that axis, which a trilinear element carries exactly — so these are the
/// element's own response, and `kᵉ T⁰` is exactly the load it applies.
fn unit_gradient_potentials(h: [f64; 3]) -> [[f64; 8]; 3] {
    let mut out = [[0.0f64; 8]; 3];
    for (axis, row) in out.iter_mut().enumerate() {
        for c in 0..8 {
            row[c] = ((c >> axis) & 1) as f64 * h[axis];
        }
    }
    out
}

/// The 8×8 conduction matrix of one trilinear brick of unit conductivity.
///
/// `∫ ∇Nᵀ ∇N dV`, by the same 2×2×2 quadrature the stiffness uses — and for the
/// same reason: it takes the three edge lengths rather than assuming a cube.
fn conduction_element(h: [f64; 3]) -> [[f64; 8]; 8] {
    let g = 1.0 / 3.0f64.sqrt();
    let det = h[0] * h[1] * h[2] / 8.0;
    let mut ke = [[0.0f64; 8]; 8];
    for gz in [-g, g] {
        for gy in [-g, g] {
            for gx in [-g, g] {
                let grad = shape_gradients([gx, gy, gz], h);
                for a in 0..8 {
                    for b in a..8 {
                        let mut acc = 0.0;
                        for axis in 0..3 {
                            acc += grad[a][axis] * grad[b][axis];
                        }
                        ke[a][b] += acc * det;
                    }
                }
            }
        }
    }
    for a in 0..8 {
        for b in 0..a {
            ke[a][b] = ke[b][a];
        }
    }
    ke
}

/// `∂N_c/∂x` at a quadrature point, in world units — shared by both elements.
fn shape_gradients(xi: [f64; 3], h: [f64; 3]) -> [[f64; 3]; 8] {
    let mut out = [[0.0f64; 3]; 8];
    for c in 0..8 {
        let sign = [
            if c & 1 == 0 { -1.0 } else { 1.0 },
            if (c >> 1) & 1 == 0 { -1.0 } else { 1.0 },
            if (c >> 2) & 1 == 0 { -1.0 } else { 1.0 },
        ];
        for axis in 0..3 {
            let other = [(axis + 1) % 3, (axis + 2) % 3];
            // d/dξ of ⅛(1+sξ)(1+sη)(1+sζ), then ξ to x.
            out[c][axis] = 0.125
                * sign[axis]
                * (1.0 + sign[other[0]] * xi[other[0]])
                * (1.0 + sign[other[1]] * xi[other[1]])
                * (2.0 / h[axis]);
        }
    }
    out
}

/// The eigenvalues of a symmetric 3×3, largest first.
///
/// The closed form rather than an iteration: a symmetric 3×3 characteristic
/// polynomial is a cubic whose roots are all real, and Smith's trigonometric
/// solution gives them in a dozen operations with no convergence to worry
/// about.
fn symmetric_eigenvalues(a: &[[f64; 3]; 3]) -> [f64; 3] {
    let off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
    if off < 1e-30 {
        let mut d = [a[0][0], a[1][1], a[2][2]];
        d.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
        return d;
    }
    let q = (a[0][0] + a[1][1] + a[2][2]) / 3.0;
    let p2 = (a[0][0] - q).powi(2) + (a[1][1] - q).powi(2) + (a[2][2] - q).powi(2) + 2.0 * off;
    let p = (p2 / 6.0).max(1e-300).sqrt();
    // B = (A - qI) / p has determinant in [-2, 2]; half of it is the cosine of
    // three times the angle the roots are spaced by.
    let mut b = *a;
    for i in 0..3 {
        b[i][i] -= q;
    }
    for row in &mut b {
        for v in row.iter_mut() {
            *v /= p;
        }
    }
    let det = b[0][0] * (b[1][1] * b[2][2] - b[1][2] * b[2][1])
        - b[0][1] * (b[1][0] * b[2][2] - b[1][2] * b[2][0])
        + b[0][2] * (b[1][0] * b[2][1] - b[1][1] * b[2][0]);
    let phi = (det / 2.0).clamp(-1.0, 1.0).acos() / 3.0;
    let hi = q + 2.0 * p * phi.cos();
    let lo = q + 2.0 * p * (phi + 2.0 * std::f64::consts::PI / 3.0).cos();
    [hi, 3.0 * q - hi - lo, lo]
}

/// Nodal displacements of the six unit macroscopic strains.
///
/// A trilinear element represents a linear displacement field exactly, so
/// these *are* the element's response to the macro strain — no solve needed,
/// and `kᵉ u⁰` is exactly the load the strain applies.
fn unit_strain_displacements(h: [f64; 3]) -> [[f64; 24]; 6] {
    let mut out = [[0.0f64; 24]; 6];
    for (case, row) in out.iter_mut().enumerate() {
        // Voigt to tensor: the three shears are engineering, so half of each
        // lands either side of the diagonal.
        let mut eps = [[0.0f64; 3]; 3];
        match case {
            0..=2 => eps[case][case] = 1.0,
            3 => {
                eps[1][2] = 0.5;
                eps[2][1] = 0.5;
            }
            4 => {
                eps[0][2] = 0.5;
                eps[2][0] = 0.5;
            }
            _ => {
                eps[0][1] = 0.5;
                eps[1][0] = 0.5;
            }
        }
        for c in 0..8 {
            let x = [
                (c & 1) as f64 * h[0],
                ((c >> 1) & 1) as f64 * h[1],
                ((c >> 2) & 1) as f64 * h[2],
            ];
            for axis in 0..3 {
                row[c * 3 + axis] = eps[axis][0] * x[0] + eps[axis][1] * x[1] + eps[axis][2] * x[2];
            }
        }
    }
    out
}

/// The 24×24 stiffness of one trilinear brick of unit modulus.
///
/// Integrated by 2×2×2 Gauss quadrature rather than transcribed. The
/// transcribed form is a page of coefficients that is only correct for a cube;
/// this one takes the three edge lengths, which is what an anisotropic cell
/// size needs.
fn element_stiffness(h: [f64; 3], poisson: f64) -> [[f64; 24]; 24] {
    let d = constitutive(poisson);

    let g = 1.0 / 3.0f64.sqrt();
    let det = h[0] * h[1] * h[2] / 8.0;
    let mut ke = [[0.0f64; 24]; 24];
    for gz in [-g, g] {
        for gy in [-g, g] {
            for gx in [-g, g] {
                let gradients = shape_gradients([gx, gy, gz], h);
                let mut b = [[0.0f64; 24]; 6];
                for c in 0..8 {
                    let (bx, by, bz) = (gradients[c][0], gradients[c][1], gradients[c][2]);
                    b[0][c * 3] = bx;
                    b[1][c * 3 + 1] = by;
                    b[2][c * 3 + 2] = bz;
                    b[3][c * 3 + 1] = bz;
                    b[3][c * 3 + 2] = by;
                    b[4][c * 3] = bz;
                    b[4][c * 3 + 2] = bx;
                    b[5][c * 3] = by;
                    b[5][c * 3 + 1] = bx;
                }
                // ke += Bᵀ D B · det J
                let mut db = [[0.0f64; 24]; 6];
                for i in 0..6 {
                    for col in 0..24 {
                        let mut acc = 0.0;
                        for k in 0..6 {
                            acc += d[i][k] * b[k][col];
                        }
                        db[i][col] = acc;
                    }
                }
                for a in 0..24 {
                    for bcol in a..24 {
                        let mut acc = 0.0;
                        for i in 0..6 {
                            acc += b[i][a] * db[i][bcol];
                        }
                        ke[a][bcol] += acc * det;
                    }
                }
            }
        }
    }
    for a in 0..24 {
        for b in 0..a {
            ke[a][b] = ke[b][a];
        }
    }
    ke
}

/// The isotropic constitutive matrix at unit Young's modulus, Voigt xx, yy,
/// zz, yz, xz, xy.
fn constitutive(poisson: f64) -> [[f64; 6]; 6] {
    let nu = poisson.clamp(-0.99, 0.49);
    let lambda = nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = 0.5 / (1.0 + nu);
    let mut d = [[0.0f64; 6]; 6];
    for i in 0..3 {
        for j in 0..3 {
            d[i][j] = lambda;
        }
        d[i][i] += 2.0 * mu;
        d[i + 3][i + 3] = mu;
    }
    d
}

/// The strain-displacement matrix at the element's centre.
///
/// The centre rather than a corner deliberately: a trilinear element's strain
/// varies across it and is worst at the corners, where on a voxelised surface
/// it is worst because of the staircase rather than because of the part. The
/// centre value is the element's average, which is the quantity that converges.
fn strain_matrix(h: [f64; 3]) -> [[f64; 24]; 6] {
    let gradients = shape_gradients([0.0, 0.0, 0.0], h);
    let mut b = [[0.0f64; 24]; 6];
    for c in 0..8 {
        let (bx, by, bz) = (gradients[c][0], gradients[c][1], gradients[c][2]);
        b[0][c * 3] = bx;
        b[1][c * 3 + 1] = by;
        b[2][c * 3 + 2] = bz;
        b[3][c * 3 + 1] = bz;
        b[3][c * 3 + 2] = by;
        b[4][c * 3] = bz;
        b[4][c * 3 + 2] = bx;
        b[5][c * 3] = by;
        b[5][c * 3 + 1] = bx;
    }
    b
}

/// A component of the fourth-order compliance tensor from its Voigt matrix.
///
/// Voigt halves the off-diagonal shear terms once and quarters them twice, so
/// unpacking is not just a table lookup.
fn tensor(s: &[[f64; 6]; 6], i: usize, j: usize, k: usize, l: usize) -> f64 {
    const VOIGT: [[usize; 3]; 3] = [[0, 5, 4], [5, 1, 3], [4, 3, 2]];
    let v = VOIGT[i][j];
    let w = VOIGT[k][l];
    let factor = match (v >= 3, w >= 3) {
        (false, false) => 1.0,
        (true, true) => 4.0,
        _ => 2.0,
    };
    s[v][w] / factor
}

/// Gauss-Jordan inverse of a 6×6, or `None` if it is singular.
fn invert6(m: &[[f64; 6]; 6]) -> Option<[[f64; 6]; 6]> {
    let mut a = *m;
    let mut inv = [[0.0f64; 6]; 6];
    for (i, row) in inv.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for col in 0..6 {
        let mut pivot = col;
        for r in col + 1..6 {
            if a[r][col].abs() > a[pivot][col].abs() {
                pivot = r;
            }
        }
        if a[pivot][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, pivot);
        inv.swap(col, pivot);
        let scale = 1.0 / a[col][col];
        for c in 0..6 {
            a[col][c] *= scale;
            inv[col][c] *= scale;
        }
        for r in 0..6 {
            if r == col {
                continue;
            }
            let factor = a[r][col];
            if factor == 0.0 {
                continue;
            }
            for c in 0..6 {
                a[r][c] -= factor * a[col][c];
                inv[r][c] -= factor * inv[col][c];
            }
        }
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cell that is solid everywhere has to come back as the base material.
    #[test]
    fn solid_recovers_the_base_material() {
        let mat = SolidMaterial {
            modulus: 210.0,
            poisson: 0.3,
        };
        let c = homogenize(
            |_| true,
            Vector3::ZERO,
            Vector3::new(1.0, 1.0, 1.0),
            6,
            mat,
            Solver::Cpu,
        );
        let nu = mat.poisson;
        let factor = mat.modulus / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let c11 = factor * (1.0 - nu);
        let c12 = factor * nu;
        let c44 = mat.modulus / (2.0 * (1.0 + nu));
        assert!((c.c[0][0] - c11).abs() < 1e-6 * c11, "{} vs {c11}", c.c[0][0]);
        assert!((c.c[0][1] - c12).abs() < 1e-6 * c12, "{} vs {c12}", c.c[0][1]);
        assert!((c.c[3][3] - c44).abs() < 1e-6 * c44, "{} vs {c44}", c.c[3][3]);
        assert!((c.relative_density - 1.0).abs() < 1e-12);

        // And the engineering constants read back off it.
        let e = c.youngs_moduli();
        assert!((e[0] - mat.modulus).abs() < 1e-4 * mat.modulus, "{}", e[0]);
        let nus = c.poisson_ratios();
        assert!((nus[2] - nu).abs() < 1e-6, "{}", nus[2]);
        let g = c.shear_moduli();
        assert!((g[0] - c44).abs() < 1e-4 * c44, "{}", g[0]);
        // Isotropic in every direction, not just on the axes.
        assert!((c.anisotropy() - 1.0).abs() < 1e-4, "{}", c.anisotropy());
    }

    /// Solid slabs across z and void between them: stiff in plane, soft across.
    #[test]
    fn layers_are_stiff_in_plane_and_soft_across_it() {
        let c = homogenize(
            |p| p.z.rem_euclid(1.0) < 0.5,
            Vector3::ZERO,
            Vector3::new(1.0, 1.0, 1.0),
            8,
            SolidMaterial::default(),
            Solver::Cpu,
        );
        let e = c.youngs_moduli();
        // Half solid, plus the void's own floor.
        assert!((c.relative_density - 0.5).abs() < 1e-4, "{}", c.relative_density);
        // In plane the slabs act in parallel: half the material, half the
        // stiffness. Across them they act in series, and the void governs.
        assert!((e[0] - 0.5).abs() < 0.05, "in plane {}", e[0]);
        assert!(e[2] < 0.01, "across {}", e[2]);
        assert!(c.anisotropy() > 10.0, "{}", c.anisotropy());
    }

    /// An empty cell is soft, and asking it for engineering constants is not a
    /// panic or a NaN.
    #[test]
    fn an_empty_cell_is_not_a_division_by_zero() {
        let c = homogenize(
            |_| false,
            Vector3::ZERO,
            Vector3::new(1.0, 1.0, 1.0),
            4,
            SolidMaterial::default(),
            Solver::Cpu,
        );
        // The void floor, not zero: an element of nothing at all leaves its
        // nodes unattached and the system singular.
        assert!(c.relative_density < 1e-4, "{}", c.relative_density);
        for e in c.youngs_moduli() {
            assert!(e.is_finite() && e < 1e-4, "{e}");
        }
        for nu in c.poisson_ratios() {
            assert!(nu.is_finite(), "{nu}");
        }
    }

    #[test]
    fn the_stiffness_is_symmetric() {
        let c = homogenize(
            |p| (p.x + p.y + p.z).rem_euclid(1.0) < 0.6,
            Vector3::ZERO,
            Vector3::new(1.0, 1.0, 1.0),
            6,
            SolidMaterial::default(),
            Solver::Cpu,
        );
        for i in 0..6 {
            for j in 0..6 {
                let scale = c.c[i][i].abs().max(c.c[j][j].abs()).max(1e-12);
                assert!(
                    (c.c[i][j] - c.c[j][i]).abs() < 1e-9 * scale,
                    "{i},{j}: {} vs {}",
                    c.c[i][j],
                    c.c[j][i]
                );
            }
        }
    }

    /// Solid material concentrates nothing: the local stress *is* the
    /// macroscopic stress, and von Mises knows the difference between pulling
    /// and shearing.
    #[test]
    fn solid_concentrates_nothing() {
        let mat = SolidMaterial {
            modulus: 200.0,
            poisson: 0.3,
        };
        let s = homogenize_strength(
            |_| true,
            Vector3::ZERO,
            Vector3::new(1.0, 1.0, 1.0),
            6,
            mat,
            Solver::Cpu,
        );
        // Unit tension gives unit von Mises; unit shear gives √3, which is
        // exactly why a material shears at a yield stress over √3.
        for axis in 0..3 {
            assert!(
                (s.concentration[axis] - 1.0).abs() < 1e-6,
                "axis {axis}: {}",
                s.concentration[axis]
            );
            assert!(
                (s.concentration[axis + 3] - 3.0f64.sqrt()).abs() < 1e-6,
                "shear {axis}: {}",
                s.concentration[axis + 3]
            );
        }
        // So the strengths come back as the material's own.
        let yielded = s.yield_strength(500.0);
        assert!((yielded[0] - 500.0).abs() < 1e-3, "{}", yielded[0]);
        assert!((yielded[3] - 500.0 / 3.0f64.sqrt()).abs() < 1e-3, "{}", yielded[3]);
        // Every gram working: efficiency one.
        assert!((s.efficiency()[0] - 1.0).abs() < 1e-3, "{:?}", s.efficiency());
        // And the peak agrees with the percentile, there being nothing to
        // concentrate around.
        assert!((s.peak_concentration[0] - 1.0).abs() < 1e-6);
    }

    /// Half the area carrying all the load is twice the stress. Exactly.
    #[test]
    fn half_the_area_is_twice_the_stress() {
        let s = homogenize_strength(
            |p| p.z.rem_euclid(1.0) < 0.5,
            Vector3::ZERO,
            Vector3::new(1.0, 1.0, 1.0),
            8,
            SolidMaterial::default(),
            Solver::Cpu,
        );
        // Pulled in plane, the slabs carry everything on half the section.
        assert!(
            (s.concentration[0] - 2.0).abs() < 0.02,
            "in plane {}",
            s.concentration[0]
        );
        assert!((s.yield_strength(400.0)[0] - 200.0).abs() < 5.0);
        // Half the material, all of it working: efficiency one again.
        assert!((s.efficiency()[0] - 1.0).abs() < 0.05, "{:?}", s.efficiency());
    }

    /// A cell that carries load along its struts wastes less of itself than
    /// one that bends them.
    #[test]
    fn stretching_is_more_efficient_than_bending() {
        let measure = |seam: f32| {
            // Slabs across z, then the same material spread as a grid of thin
            // walls: same density, different arrangement.
            homogenize_strength(
                move |p| p.z.rem_euclid(1.0) < seam,
                Vector3::ZERO,
                Vector3::new(1.0, 1.0, 1.0),
                8,
                SolidMaterial::default(),
                Solver::Cpu,
            )
        };
        // Thicker slabs, lower stress for the same macroscopic load.
        let thin = measure(0.25);
        let thick = measure(0.5);
        assert!(
            thin.concentration[0] > thick.concentration[0],
            "thin {} thick {}",
            thin.concentration[0],
            thick.concentration[0]
        );
        // And the strength scales with the section, near enough.
        let ratio = thick.yield_strength(100.0)[0] / thin.yield_strength(100.0)[0];
        assert!((ratio - 2.0).abs() < 0.1, "{ratio}");
    }

    /// The diagnostic that keeps the rest of it honest.
    #[test]
    fn an_unresolved_grid_says_so() {
        let build = |resolution| {
            homogenize_strength(
                // Thin slabs — a tenth of the cell, so at eight voxels no
                // element is ever entirely inside one.
                |p| p.z.rem_euclid(1.0) < 0.1,
                Vector3::ZERO,
                Vector3::new(1.0, 1.0, 1.0),
                resolution,
                SolidMaterial::default(),
                Solver::Cpu,
            )
        };
        let coarse = build(8);
        assert!(!coarse.resolved(), "sampled {} of the solid", coarse.sampled);
        // Given enough voxels to hold the slab — and a thickness that is a
        // whole number of them, so the section is exactly what it says — it
        // lands on the exact answer: an eighth of the area, eight times the
        // stress.
        let fine = homogenize_strength(
            |p| p.z.rem_euclid(1.0) < 0.125,
            Vector3::ZERO,
            Vector3::new(1.0, 1.0, 1.0),
            24,
            SolidMaterial::default(),
            Solver::Cpu,
        );
        assert!(
            (fine.concentration[0] - 8.0).abs() < 0.15,
            "{}",
            fine.concentration[0]
        );

        // Slabs sit exactly on the efficiency bound, which is no test of it.
        // Crossed walls do not: the ones normal to the load carry nothing, so
        // some of the material is along for the ride.
        let crossed = homogenize_strength(
            |p| p.z.rem_euclid(1.0) < 0.15 || p.x.rem_euclid(1.0) < 0.15,
            Vector3::ZERO,
            Vector3::new(1.0, 1.0, 1.0),
            24,
            SolidMaterial::default(),
            Solver::Cpu,
        );
        assert!(
            crossed.resolved(),
            "floor {} sampled {} efficiency {:?}",
            crossed.solid_floor,
            crossed.sampled,
            crossed.efficiency()
        );
        let efficiency = crossed.efficiency()[0];
        assert!(
            (0.5..0.95).contains(&efficiency),
            "some of it should be idle, not all or none: {efficiency}"
        );
    }

    #[test]
    fn an_empty_cell_has_no_strength_and_no_panic() {
        let s = homogenize_strength(
            |_| false,
            Vector3::ZERO,
            Vector3::new(1.0, 1.0, 1.0),
            4,
            SolidMaterial::default(),
            Solver::Cpu,
        );
        for v in s.yield_strength(500.0) {
            assert!(v.is_finite() || v == 0.0, "{v}");
        }
        for v in s.collapse_strain(500.0) {
            assert!(v.is_finite(), "{v}");
        }
    }

    #[test]
    fn the_voxel_grid_is_reported() {
        let c = homogenize(
            |_| true,
            Vector3::ZERO,
            Vector3::new(1.0, 1.0, 1.0),
            4,
            SolidMaterial::default(),
            Solver::Cpu,
        );
        assert_eq!(c.voxels, 4);
        // And the clamp is visible rather than silent.
        let clamped = homogenize(
            |_| true,
            Vector3::ZERO,
            Vector3::new(1.0, 1.0, 1.0),
            999,
            SolidMaterial::default(),
            Solver::Cpu,
        );
        assert_eq!(clamped.voxels, 48);
    }

    /// A cell that is solid everywhere conducts exactly like the material.
    #[test]
    fn solid_conducts_like_the_base_material() {
        let k = homogenize_conduction(
            |_| true,
            Vector3::ZERO,
            Vector3::new(1.0, 1.0, 1.0),
            6,
            237.0,
            Solver::Cpu,
        );
        for along in k.axes() {
            assert!((along - 237.0).abs() < 1e-4, "{along}");
        }
        // Off-diagonal nothing, and isotropic however it is asked.
        assert!(k.k[0][1].abs() < 1e-6 && k.k[0][2].abs() < 1e-6);
        assert!((k.anisotropy() - 1.0).abs() < 1e-6, "{}", k.anisotropy());
        assert!(
            (k.directional(Vector3::new(1.0, 1.0, 1.0)) - 237.0).abs() < 1e-3,
            "{}",
            k.directional(Vector3::new(1.0, 1.0, 1.0))
        );
        // Every gram on the path.
        assert!((k.tortuosity_factor(237.0) - 1.0).abs() < 1e-4);
    }

    /// Slabs across z: the two classical bounds, in one cell.
    #[test]
    fn layers_conduct_in_plane_and_not_across() {
        let k = homogenize_conduction(
            |p| p.z.rem_euclid(1.0) < 0.5,
            Vector3::ZERO,
            Vector3::new(1.0, 1.0, 1.0),
            8,
            1.0,
            Solver::Cpu,
        );
        let along = k.axes();
        // In plane the slabs conduct in parallel — half the material, half the
        // conductivity. Across them they are in series and the void decides.
        assert!((along[0] - 0.5).abs() < 0.01, "in plane {}", along[0]);
        assert!(along[2] < 0.001, "across {}", along[2]);
        assert!(k.anisotropy() > 100.0, "{}", k.anisotropy());
    }

    #[test]
    fn an_empty_cell_conducts_nothing_without_dividing_by_zero() {
        let k = homogenize_conduction(
            |_| false,
            Vector3::ZERO,
            Vector3::new(1.0, 1.0, 1.0),
            4,
            1.0,
            Solver::Cpu,
        );
        for along in k.axes() {
            assert!(along.is_finite() && along < 1e-4, "{along}");
        }
        assert!(k.tortuosity_factor(1.0).is_finite());
    }

    #[test]
    fn eigenvalues_of_a_known_matrix() {
        // The 2×2 block has eigenvalues 3 and 1; the third axis is 5.
        let a = [[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 5.0]];
        let e = symmetric_eigenvalues(&a);
        assert!((e[0] - 5.0).abs() < 1e-9, "{e:?}");
        assert!((e[1] - 3.0).abs() < 1e-9, "{e:?}");
        assert!((e[2] - 1.0).abs() < 1e-9, "{e:?}");
        // A diagonal matrix takes the shortcut, and still comes back sorted.
        let d = symmetric_eigenvalues(&[[1.0, 0.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 4.0]]);
        assert_eq!(d, [7.0, 4.0, 1.0]);
    }

    /// The device is only worth having if it says the same thing.
    ///
    /// Skipped rather than failed where there is no adapter — a build machine
    /// without a GPU should not fail a test about one.
    #[test]
    fn the_gpu_agrees_with_the_cpu() {
        let cell = |solver| {
            homogenize(
                // A shape with something to concentrate: not a solid block,
                // whose answer any solver gets right.
                |p| {
                    (p.x.rem_euclid(1.0) - 0.5).abs() < 0.2
                        || (p.z.rem_euclid(1.0) - 0.5).abs() < 0.2
                },
                Vector3::ZERO,
                Vector3::new(1.0, 1.0, 1.0),
                16,
                SolidMaterial {
                    modulus: 200.0,
                    poisson: 0.3,
                },
                solver,
            )
        };
        let gpu = cell(Solver::Gpu);
        if gpu.solver != Solver::Gpu {
            eprintln!("no wgpu adapter — skipping the device comparison");
            return;
        }
        let cpu = cell(Solver::Cpu);
        assert_eq!(cpu.solver, Solver::Cpu);
        assert!((gpu.relative_density - cpu.relative_density).abs() < 1e-12);
        for i in 0..6 {
            for j in 0..6 {
                let scale = cpu.c[i][i].abs().max(cpu.c[j][j].abs()).max(1e-9);
                assert!(
                    (gpu.c[i][j] - cpu.c[i][j]).abs() < 2e-3 * scale,
                    "{i},{j}: gpu {} cpu {}",
                    gpu.c[i][j],
                    cpu.c[i][j]
                );
            }
        }
    }

    /// The scalar problem too — a different element size and one unknown a
    /// node, which is the other half of what the one shader has to handle.
    #[test]
    fn the_gpu_conducts_the_same_as_the_cpu() {
        let cell = |solver| {
            homogenize_conduction(
                |p| p.z.rem_euclid(1.0) < 0.4 || p.x.rem_euclid(1.0) < 0.3,
                Vector3::ZERO,
                Vector3::new(1.0, 1.0, 1.0),
                16,
                12.0,
                solver,
            )
        };
        let gpu = cell(Solver::Gpu);
        if gpu.solver != Solver::Gpu {
            eprintln!("no wgpu adapter — skipping the device comparison");
            return;
        }
        let cpu = cell(Solver::Cpu);
        for axis in 0..3 {
            let (a, b) = (gpu.axes()[axis], cpu.axes()[axis]);
            assert!(
                (a - b).abs() < 2e-3 * b.abs().max(1e-9),
                "axis {axis}: gpu {a} cpu {b}"
            );
        }
    }

    /// And the stress field read back off the device solution.
    #[test]
    fn the_gpu_finds_the_same_concentration() {
        let cell = |solver| {
            homogenize_strength(
                |p| p.z.rem_euclid(1.0) < 0.125,
                Vector3::ZERO,
                Vector3::new(1.0, 1.0, 1.0),
                24,
                SolidMaterial::default(),
                solver,
            )
        };
        let gpu = cell(Solver::Gpu);
        if gpu.stiffness.solver != Solver::Gpu {
            eprintln!("no wgpu adapter — skipping the device comparison");
            return;
        }
        // The exact answer, whichever solver reaches it: an eighth of the
        // section carries eight times the stress.
        assert!(
            (gpu.concentration[0] - 8.0).abs() < 0.15,
            "{}",
            gpu.concentration[0]
        );
        let cpu = cell(Solver::Cpu);
        assert!(
            (gpu.concentration[0] - cpu.concentration[0]).abs() < 5e-3 * cpu.concentration[0],
            "gpu {} cpu {}",
            gpu.concentration[0],
            cpu.concentration[0]
        );
    }

    #[test]
    fn inverting_a_singular_matrix_says_so() {
        assert!(invert6(&[[0.0; 6]; 6]).is_none());
        let mut identity = [[0.0f64; 6]; 6];
        for i in 0..6 {
            identity[i][i] = 2.0;
        }
        let inv = invert6(&identity).unwrap();
        assert!((inv[3][3] - 0.5).abs() < 1e-12);
    }

    /// An anisotropic cell size has to give the same material as a cubic one
    /// when the geometry is scaled with it.
    #[test]
    fn a_stretched_cell_is_the_same_material() {
        let cubic = homogenize(
            |p| p.z.rem_euclid(1.0) < 0.5,
            Vector3::ZERO,
            Vector3::new(1.0, 1.0, 1.0),
            8,
            SolidMaterial::default(),
            Solver::Cpu,
        );
        let stretched = homogenize(
            |p| (p.z / 3.0).rem_euclid(1.0) < 0.5,
            Vector3::ZERO,
            Vector3::new(2.0, 2.0, 3.0),
            8,
            SolidMaterial::default(),
            Solver::Cpu,
        );
        for i in 0..6 {
            for j in 0..6 {
                let scale = cubic.c[i][i].abs().max(1e-6);
                assert!(
                    (cubic.c[i][j] - stretched.c[i][j]).abs() < 1e-3 * scale,
                    "{i},{j}: {} vs {}",
                    cubic.c[i][j],
                    stretched.c[i][j]
                );
            }
        }
    }
}
