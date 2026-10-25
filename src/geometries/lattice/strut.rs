//! Strut (beam) lattices — a unit cell of line segments, tiled.
//!
//! Each topology is a list of segments in unit-cell coordinates, chosen so that
//! tiling the cell over the integer grid reproduces the infinite lattice with
//! every strut listed exactly once. The solid is then everything within half a
//! strut thickness of any segment, evaluated as a distance field: the nodes come
//! out as smooth unions rather than the interpenetrating cylinders you get from
//! meshing each beam separately, which is what makes the result watertight.

use crate::math::Vector3;
use std::sync::OnceLock;

/// A strut in unit-cell coordinates, `[start, end]`.
pub type Segment = [[f32; 3]; 2];

/// Simple cubic: one strut along each axis through every lattice point.
const CUBIC: [Segment; 3] = [
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
];

/// Body-centred cubic: the cell centre tied to all eight corners.
const BCC: [Segment; 8] = [
    [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],
    [[0.5, 0.5, 0.5], [1.0, 0.0, 0.0]],
    [[0.5, 0.5, 0.5], [0.0, 1.0, 0.0]],
    [[0.5, 0.5, 0.5], [1.0, 1.0, 0.0]],
    [[0.5, 0.5, 0.5], [0.0, 0.0, 1.0]],
    [[0.5, 0.5, 0.5], [1.0, 0.0, 1.0]],
    [[0.5, 0.5, 0.5], [0.0, 1.0, 1.0]],
    [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]],
];

/// BCC plus the axis-aligned struts that carry load straight through.
const BCC_Z: [Segment; 11] = [
    [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],
    [[0.5, 0.5, 0.5], [1.0, 0.0, 0.0]],
    [[0.5, 0.5, 0.5], [0.0, 1.0, 0.0]],
    [[0.5, 0.5, 0.5], [1.0, 1.0, 0.0]],
    [[0.5, 0.5, 0.5], [0.0, 0.0, 1.0]],
    [[0.5, 0.5, 0.5], [1.0, 0.0, 1.0]],
    [[0.5, 0.5, 0.5], [0.0, 1.0, 1.0]],
    [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]],
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
];

/// Face-centred cubic: each face centre tied to the four corners of its face.
/// Only the three low faces are listed — the high ones belong to the neighbour.
const FCC: [Segment; 12] = [
    [[0.0, 0.5, 0.5], [0.0, 0.0, 0.0]],
    [[0.0, 0.5, 0.5], [0.0, 1.0, 0.0]],
    [[0.0, 0.5, 0.5], [0.0, 1.0, 1.0]],
    [[0.0, 0.5, 0.5], [0.0, 0.0, 1.0]],
    [[0.5, 0.0, 0.5], [0.0, 0.0, 0.0]],
    [[0.5, 0.0, 0.5], [1.0, 0.0, 0.0]],
    [[0.5, 0.0, 0.5], [1.0, 0.0, 1.0]],
    [[0.5, 0.0, 0.5], [0.0, 0.0, 1.0]],
    [[0.5, 0.5, 0.0], [0.0, 0.0, 0.0]],
    [[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]],
    [[0.5, 0.5, 0.0], [1.0, 1.0, 0.0]],
    [[0.5, 0.5, 0.0], [0.0, 1.0, 0.0]],
];

/// Octet truss: FCC's corner-to-face struts plus the twelve edges of the
/// octahedron the six face centres form. That is the full nearest-neighbour
/// graph of the FCC lattice — every strut is a tetrahedron or octahedron edge,
/// so the truss is stretch-dominated and stiff for its weight.
const OCTET: [Segment; 24] = [
    // Corner to face centre (the tetrahedra).
    [[0.0, 0.5, 0.5], [0.0, 0.0, 0.0]],
    [[0.0, 0.5, 0.5], [0.0, 1.0, 0.0]],
    [[0.0, 0.5, 0.5], [0.0, 1.0, 1.0]],
    [[0.0, 0.5, 0.5], [0.0, 0.0, 1.0]],
    [[0.5, 0.0, 0.5], [0.0, 0.0, 0.0]],
    [[0.5, 0.0, 0.5], [1.0, 0.0, 0.0]],
    [[0.5, 0.0, 0.5], [1.0, 0.0, 1.0]],
    [[0.5, 0.0, 0.5], [0.0, 0.0, 1.0]],
    [[0.5, 0.5, 0.0], [0.0, 0.0, 0.0]],
    [[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]],
    [[0.5, 0.5, 0.0], [1.0, 1.0, 0.0]],
    [[0.5, 0.5, 0.0], [0.0, 1.0, 0.0]],
    // Face centre to face centre (the octahedron).
    [[0.5, 0.5, 0.0], [0.5, 0.0, 0.5]],
    [[0.5, 0.5, 0.0], [0.5, 1.0, 0.5]],
    [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]],
    [[0.5, 0.5, 0.0], [1.0, 0.5, 0.5]],
    [[0.5, 0.5, 1.0], [0.5, 0.0, 0.5]],
    [[0.5, 0.5, 1.0], [0.5, 1.0, 0.5]],
    [[0.5, 0.5, 1.0], [0.0, 0.5, 0.5]],
    [[0.5, 0.5, 1.0], [1.0, 0.5, 0.5]],
    [[0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
    [[0.5, 0.0, 0.5], [1.0, 0.5, 0.5]],
    [[0.5, 1.0, 0.5], [0.0, 0.5, 0.5]],
    [[0.5, 1.0, 0.5], [1.0, 0.5, 0.5]],
];

/// Cubic diamond: the four tetrahedral bonds of each of the cell's four
/// interior atoms. Every atom on the shifted sublattice lies strictly inside
/// one cell, so this lists each bond once.
const DIAMOND: [Segment; 16] = [
    [[0.25, 0.25, 0.25], [0.0, 0.0, 0.0]],
    [[0.25, 0.25, 0.25], [0.5, 0.5, 0.0]],
    [[0.25, 0.25, 0.25], [0.5, 0.0, 0.5]],
    [[0.25, 0.25, 0.25], [0.0, 0.5, 0.5]],
    [[0.75, 0.75, 0.25], [0.5, 0.5, 0.0]],
    [[0.75, 0.75, 0.25], [1.0, 1.0, 0.0]],
    [[0.75, 0.75, 0.25], [1.0, 0.5, 0.5]],
    [[0.75, 0.75, 0.25], [0.5, 1.0, 0.5]],
    [[0.75, 0.25, 0.75], [0.5, 0.0, 0.5]],
    [[0.75, 0.25, 0.75], [1.0, 0.5, 0.5]],
    [[0.75, 0.25, 0.75], [1.0, 0.0, 1.0]],
    [[0.75, 0.25, 0.75], [0.5, 0.5, 1.0]],
    [[0.25, 0.75, 0.75], [0.0, 0.5, 0.5]],
    [[0.25, 0.75, 0.75], [0.5, 1.0, 0.5]],
    [[0.25, 0.75, 0.75], [0.5, 0.5, 1.0]],
    [[0.25, 0.75, 0.75], [0.0, 1.0, 1.0]],
];

/// The edges of the Kelvin cell — the truncated octahedron that is the Voronoi
/// cell of a body-centred cubic lattice, and so the shape a foam's bubbles take
/// when they are all the same size.
///
/// Built rather than tabulated: the 24 vertices are every permutation of
/// `(±½, ±¼, 0)` about the cell centre, and the 36 edges are the vertex pairs
/// one edge-length apart. Written out, that is 36 lines of six numbers each
/// that say nothing about why they are those numbers.
///
/// One cell per cube is enough for the whole foam even though truncated
/// octahedra pack body-centred, not simple cubic — every edge of a
/// corner-centred cell is also an edge of a neighbouring centre-centred one, so
/// tiling the centres alone still lays down every strut. `corner_cells_are_covered`
/// is the test that keeps that claim honest.
fn kelvin() -> &'static [Segment] {
    static EDGES: OnceLock<Vec<Segment>> = OnceLock::new();
    EDGES.get_or_init(|| {
        let mut verts: Vec<[f32; 3]> = Vec::with_capacity(24);
        // Which axis carries the ½, which the ¼; the third is 0.
        for (long, short) in [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)] {
            for sign_long in [0.5f32, -0.5] {
                for sign_short in [0.25f32, -0.25] {
                    let mut v = [0.5f32; 3];
                    v[long] += sign_long;
                    v[short] += sign_short;
                    verts.push(v);
                }
            }
        }
        // Edge length is √2/4 for a unit cell — the gap between a vertex and
        // the one that swaps its ½ and ¼ across two axes.
        let edge_sq = 0.125f32;
        let mut edges = Vec::with_capacity(36);
        for (i, a) in verts.iter().enumerate() {
            for b in &verts[i + 1..] {
                let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let len_sq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if (len_sq - edge_sq).abs() < 1e-4 {
                    edges.push([*a, *b]);
                }
            }
        }
        edges
    })
}

/// Segments a period of a smooth centreline is cut into. Each chord sits
/// within `(π/24)²/8 ≈ 0.2 %` of the curve's amplitude of it — invisible at
/// any resolution a lattice is built at — and the tube around a polyline is
/// exactly what [`distance`] measures, binning and all.
const CURVE_SEGMENTS: usize = 24;

/// A closed periodic centreline `f(t)`, `t ∈ [0, 1]`, as segments.
fn polyline(f: impl Fn(f32) -> [f32; 3]) -> Vec<Segment> {
    (0..CURVE_SEGMENTS)
        .map(|i| {
            [
                f(i as f32 / CURVE_SEGMENTS as f32),
                f((i + 1) as f32 / CURVE_SEGMENTS as f32),
            ]
        })
        .collect()
}

/// Amplitude a weave's yarns rise and fall by, in cells: centrelines pass a
/// quarter cell apart at every crossing.
const WEAVE_RISE: f32 = 0.125;

/// Plain weave. Warps run along x at y = ¼ and ¾, wefts along y at x = 0 and
/// ½; each rises at one crossing and dips at the next, in opposite phase to
/// the yarns it crosses, so over and under alternate both ways.
fn weave() -> &'static [Segment] {
    static SEGMENTS: OnceLock<Vec<Segment>> = OnceLock::new();
    SEGMENTS.get_or_init(|| {
        let wave = |t: f32| WEAVE_RISE * (std::f32::consts::TAU * t).cos();
        let mut out = Vec::new();
        // Warps: up at x = 0, down at x = ½ — and the other one the reverse.
        out.extend(polyline(|t| [t, 0.25, 0.5 + wave(t)]));
        out.extend(polyline(|t| [t, 0.75, 0.5 - wave(t)]));
        // Wefts: under the first warp and over the second at x = 0, the other
        // way round at x = ½.
        out.extend(polyline(|t| [0.0, t, 0.5 - wave(t - 0.25)]));
        out.extend(polyline(|t| [0.5, t, 0.5 + wave(t - 0.25)]));
        out
    })
}

/// Three-strand braid about the cell's vertical axis: strand `k` follows
/// `(sin(τ + 2πk/3), ½ sin 2(τ + 2πk/3))`, scaled into the cell, rising one
/// period a cell — the figure-eight that makes a plait.
fn braid() -> &'static [Segment] {
    static SEGMENTS: OnceLock<Vec<Segment>> = OnceLock::new();
    SEGMENTS.get_or_init(|| {
        let mut out = Vec::new();
        for k in 0..3 {
            let phase = k as f32 * std::f32::consts::TAU / 3.0;
            out.extend(polyline(|t| {
                let a = std::f32::consts::TAU * t + phase;
                [0.5 + 0.32 * a.sin(), 0.5 + 0.16 * (2.0 * a).sin(), t]
            }));
        }
        out
    })
}

/// Six helices on a ring of radius 0.3 about the cell's vertical axis, one
/// turn a cell.
fn helix_bundle() -> &'static [Segment] {
    static SEGMENTS: OnceLock<Vec<Segment>> = OnceLock::new();
    SEGMENTS.get_or_init(|| {
        let mut out = Vec::new();
        for k in 0..6 {
            let phase = k as f32 * std::f32::consts::TAU / 6.0;
            out.extend(polyline(|t| {
                let a = std::f32::consts::TAU * t + phase;
                [0.5 + 0.3 * a.cos(), 0.5 + 0.3 * a.sin(), t]
            }));
        }
        out
    })
}

/// A beam-lattice unit cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Strut {
    /// Simple cubic — struts along the three axes only. Very stiff along them,
    /// very compliant in shear; the cheapest lattice to print and the weakest
    /// off-axis.
    Cubic,
    /// Body-centred cubic — a node in the middle of every cell, tied to the
    /// eight corners. Bending-dominated, so it absorbs energy rather than
    /// resisting load: the usual choice for crash structures and cushioning.
    Bcc,
    /// BCC with the axis-aligned struts added back ("BCC-Z"), trading some of
    /// the compliance for stiffness along the axes.
    BccZ,
    /// Face-centred cubic — face centres tied to their four corners.
    Fcc,
    /// Octet truss — the FCC nearest-neighbour graph. Stretch-dominated and
    /// near-isotropic; the standard high stiffness-to-weight lattice.
    Octet,
    /// Cubic diamond — four bonds per node at the tetrahedral angle. Fewer
    /// struts than the octet at the same node count, and no strut is axis
    /// aligned, which prints cleanly in any orientation.
    Diamond,
    /// Kelvin cell — the edges of a truncated octahedron, three struts meeting
    /// at every node at 120°. It is the shape equal-sized bubbles settle into,
    /// so it is the reference lattice for anything standing in for a foam:
    /// energy absorption, filters, bone scaffolds.
    Kelvin,
    /// Plain weave: two warps along x and two wefts along y a cell, each
    /// passing over one crossing and under the next. Yarns of a diameter under
    /// a quarter of a cell never touch — four interlocked families of yarn,
    /// the way a cloth is — and thicker ones fuse where they cross into one
    /// sheet that flexes like cloth.
    Weave,
    /// A three-strand braid up the middle of every cell column, one period a
    /// cell: the strands wrap one another without meeting — they pass 0.148
    /// of a cell apart, centre to centre, where they cross — so below that
    /// diameter each column is three interlocked strands held by their own
    /// geometry, and above it one fused plait.
    Braid,
    /// Six fibres twisted one turn a cell around each column's axis — a rope.
    /// Twist turns axial load into radial grip, which is what makes a rope
    /// stronger than its strands laid straight. Neighbouring fibres pass 0.147
    /// of a cell apart, centre to centre, so they stay separate below that
    /// diameter and fuse into a tube above it.
    HelixBundle,
}

impl Strut {
    /// Every topology, in declaration order.
    pub const ALL: [Strut; 10] = [
        Strut::Cubic,
        Strut::Bcc,
        Strut::BccZ,
        Strut::Fcc,
        Strut::Octet,
        Strut::Diamond,
        Strut::Kelvin,
        Strut::Weave,
        Strut::Braid,
        Strut::HelixBundle,
    ];

    /// Yarns rather than beams: [`Weave`](Strut::Weave),
    /// [`Braid`](Strut::Braid) and [`HelixBundle`](Strut::HelixBundle) have
    /// no nodes — their strands pass one another instead of joining — so a
    /// thin one is several interlocked pieces, not one frame.
    pub fn is_textile(self) -> bool {
        matches!(self, Strut::Weave | Strut::Braid | Strut::HelixBundle)
    }

    /// The cell's struts, in unit-cell coordinates.
    pub fn segments(self) -> &'static [Segment] {
        match self {
            Strut::Cubic => &CUBIC,
            Strut::Bcc => &BCC,
            Strut::BccZ => &BCC_Z,
            Strut::Fcc => &FCC,
            Strut::Octet => &OCTET,
            Strut::Diamond => &DIAMOND,
            Strut::Kelvin => kelvin(),
            Strut::Weave => weave(),
            Strut::Braid => braid(),
            Strut::HelixBundle => helix_bundle(),
        }
    }

    /// Lower-case identifier, for logs and CLI arguments.
    pub fn name(self) -> &'static str {
        match self {
            // Not "cubic": a slicer means something else by that, and this
            // module has that one too — see `Infill::Cubic`.
            Strut::Cubic => "simple-cubic",
            Strut::Bcc => "bcc",
            Strut::BccZ => "bcc-z",
            Strut::Fcc => "fcc",
            Strut::Octet => "octet",
            Strut::Diamond => "diamond-strut",
            Strut::Kelvin => "kelvin",
            Strut::Weave => "weave",
            Strut::Braid => "braid",
            Strut::HelixBundle => "helix-bundle",
        }
    }

    /// The topology with this [`name`](Self::name), if any.
    pub fn from_name(name: &str) -> Option<Strut> {
        Strut::ALL.into_iter().find(|s| s.name() == name)
    }
}

/// Distance from `p` to the nearest strut of the tiled lattice, giving up at
/// `cull`.
///
/// Only cells whose world-space box comes within `cull` of `p` are opened, so
/// the cost is the handful of struts actually near the point rather than the
/// whole 3×3×3 neighbourhood. Points further out than `cull` report `cull`
/// exactly: marching cubes only reads the field's sign out there, and the
/// gradient it reads near the surface is still exact as long as `cull` clears
/// the surface by a few samples.
pub(crate) fn distance(
    p: Vector3,
    origin: Vector3,
    cell: Vector3,
    segments: &[Segment],
    cull: f32,
) -> f32 {
    let u = [
        (p.x - origin.x) / cell.x,
        (p.y - origin.y) / cell.y,
        (p.z - origin.z) / cell.z,
    ];
    let base = [u[0].floor(), u[1].floor(), u[2].floor()];
    let mut best = cull;

    for oz in -1..=1 {
        for oy in -1..=1 {
            for ox in -1..=1 {
                let c = [
                    base[0] + ox as f32,
                    base[1] + oy as f32,
                    base[2] + oz as f32,
                ];
                // Struts live inside the cell box, so the distance to that box
                // is a lower bound on the distance to any of them.
                let mut bound = 0.0f32;
                for a in 0..3 {
                    let local = u[a] - c[a];
                    let out = if local < 0.0 {
                        -local
                    } else if local > 1.0 {
                        local - 1.0
                    } else {
                        0.0
                    };
                    let world = out * cell_axis(cell, a);
                    bound += world * world;
                }
                if bound >= best * best {
                    continue;
                }
                let cell_origin = Vector3::new(
                    origin.x + c[0] * cell.x,
                    origin.y + c[1] * cell.y,
                    origin.z + c[2] * cell.z,
                );
                for seg in segments {
                    let a = Vector3::new(
                        cell_origin.x + seg[0][0] * cell.x,
                        cell_origin.y + seg[0][1] * cell.y,
                        cell_origin.z + seg[0][2] * cell.z,
                    );
                    let b = Vector3::new(
                        cell_origin.x + seg[1][0] * cell.x,
                        cell_origin.y + seg[1][1] * cell.y,
                        cell_origin.z + seg[1][2] * cell.z,
                    );
                    let d = segment_distance(p, a, b);
                    if d < best {
                        best = d;
                    }
                }
            }
        }
    }
    best
}

/// Bins per axis of the unit cell in [`Bins`].
const BINS: usize = 8;

/// Which struts can matter where, worked out once per lattice.
///
/// The cell is cut into `BINS³` boxes, and each box lists the struts — of its
/// own cell and of the 26 around it — that come within `cull` of some point
/// in it. A sample then measures only its own box's list instead of every
/// strut of every cell its cull reaches: the same distances, by the same
/// arithmetic, so [`distance_binned`] returns exactly what [`distance`]
/// does. A strut left out of a box is further than `cull` from all of it,
/// and `distance` never reports more than `cull` anyway.
pub(crate) struct Bins {
    /// The reach the lists were drawn up for, in world units. A sample that
    /// needs more falls back to [`distance`].
    cull: f32,
    /// `entries[start[b]..start[b + 1]]` belongs to box `b`.
    start: Vec<u32>,
    /// Neighbouring cell offset and strut index.
    entries: Vec<([i8; 3], u16)>,
}

impl Bins {
    pub(crate) fn new(segments: &[Segment], cell: Vector3, cull: f32) -> Option<Bins> {
        // Struts reaching further than one cell, or more of them than the
        // index can name, are outside what the lists can describe. Negated so
        // a NaN cull or cell is refused too, which `cull < 0.0` would let by.
        let size = [cell.x, cell.y, cell.z];
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        let unusable = !(cull >= 0.0) || size.iter().any(|&c| !(cull < c));
        if unusable || segments.len() > u16::MAX as usize {
            return None;
        }
        // Drawn up a little wider than asked, so that rounding in a sample's
        // own bound can never leave out a strut `distance` would have found.
        let reach = cull * 1.001 + 1e-6 * size.iter().fold(0.0f32, |m, &c| m.max(c));
        let mut start = Vec::with_capacity(BINS * BINS * BINS + 1);
        let mut entries = Vec::new();
        start.push(0);
        for bz in 0..BINS {
            for by in 0..BINS {
                for bx in 0..BINS {
                    let lo = [bx, by, bz].map(|b| b as f32 / BINS as f32);
                    let hi = [bx, by, bz].map(|b| (b + 1) as f32 / BINS as f32);
                    for oz in -1i8..=1 {
                        for oy in -1i8..=1 {
                            for ox in -1i8..=1 {
                                let o = [ox, oy, oz];
                                for (index, seg) in segments.iter().enumerate() {
                                    // Gap between the box and the strut's own
                                    // box, which no point of either can beat.
                                    let mut gap = 0.0f32;
                                    for a in 0..3 {
                                        let s0 = seg[0][a].min(seg[1][a]) + o[a] as f32;
                                        let s1 = seg[0][a].max(seg[1][a]) + o[a] as f32;
                                        let g = (s0 - hi[a]).max(lo[a] - s1).max(0.0) * size[a];
                                        gap += g * g;
                                    }
                                    if gap <= reach * reach {
                                        entries.push((o, index as u16));
                                    }
                                }
                            }
                        }
                    }
                    start.push(entries.len() as u32);
                }
            }
        }
        Some(Bins {
            cull: reach.min(cull),
            start,
            entries,
        })
    }
}

/// [`distance`], reading only the struts `bins` lists for the point's box.
pub(crate) fn distance_binned(
    p: Vector3,
    origin: Vector3,
    cell: Vector3,
    segments: &[Segment],
    bins: &Bins,
    cull: f32,
) -> f32 {
    // Negated so a NaN cull takes the full search, as it would have before.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(cull <= bins.cull) {
        return distance(p, origin, cell, segments, cull);
    }
    let u = [
        (p.x - origin.x) / cell.x,
        (p.y - origin.y) / cell.y,
        (p.z - origin.z) / cell.z,
    ];
    let base = [u[0].floor(), u[1].floor(), u[2].floor()];
    let bin = |a: usize| (((u[a] - base[a]) * BINS as f32) as usize).min(BINS - 1);
    let b = (bin(2) * BINS + bin(1)) * BINS + bin(0);
    let mut best = cull;
    for &(o, index) in &bins.entries[bins.start[b] as usize..bins.start[b + 1] as usize] {
        // Exactly as `distance` places a strut, so the same distance comes out.
        let c = [
            base[0] + o[0] as f32,
            base[1] + o[1] as f32,
            base[2] + o[2] as f32,
        ];
        let cell_origin = Vector3::new(
            origin.x + c[0] * cell.x,
            origin.y + c[1] * cell.y,
            origin.z + c[2] * cell.z,
        );
        let seg = &segments[index as usize];
        let a = Vector3::new(
            cell_origin.x + seg[0][0] * cell.x,
            cell_origin.y + seg[0][1] * cell.y,
            cell_origin.z + seg[0][2] * cell.z,
        );
        let b = Vector3::new(
            cell_origin.x + seg[1][0] * cell.x,
            cell_origin.y + seg[1][1] * cell.y,
            cell_origin.z + seg[1][2] * cell.z,
        );
        let d = segment_distance(p, a, b);
        if d < best {
            best = d;
        }
    }
    best
}

#[inline]
fn cell_axis(cell: Vector3, axis: usize) -> f32 {
    match axis {
        0 => cell.x,
        1 => cell.y,
        _ => cell.z,
    }
}

/// Distance from a point to a line segment. Shared with
/// [`Region`](super::Region) and the cuboct voxel cells.
#[inline]
pub(crate) fn segment_distance(p: Vector3, a: Vector3, b: Vector3) -> f32 {
    let ab = b - a;
    let ap = p - a;
    let len_sq = ab.length_sq();
    let t = if len_sq > f32::EPSILON {
        (ap.dot(ab) / len_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (ap - ab * t).length()
}

#[cfg(test)]
mod tests {
    use super::*;

    const CELL: Vector3 = Vector3::ONE;

    fn dist(kind: Strut, p: Vector3) -> f32 {
        distance(p, Vector3::ZERO, CELL, kind.segments(), 10.0)
    }

    #[test]
    fn binned_distance_is_the_full_search() {
        // Same bits, not just close: the bins only decide which struts are
        // measured, never how. An anisotropic cell and an offset origin, so
        // the boxes are not all unit cubes at zero.
        let cell = Vector3::new(1.3, 0.8, 1.1);
        let origin = Vector3::new(-0.37, 0.21, 0.05);
        let mut kinds: Vec<Vec<Segment>> =
            Strut::ALL.iter().map(|k| k.segments().to_vec()).collect();
        kinds.extend(super::super::Cuboct::ALL.iter().map(|c| c.segments(1.0)));
        for (n, segments) in kinds.iter().enumerate() {
            for cull in [0.05f32, 0.2, 0.5] {
                let bins = Bins::new(segments, cell, cull).expect("in range");
                for i in 0..2000 {
                    let f = |a: u32| {
                        ((i * 2654435761u32.wrapping_add(a * 40503)) % 100_000) as f32 / 100_000.0
                    };
                    let p = Vector3::new(f(1) * 4.0 - 2.0, f(2) * 3.0 - 1.5, f(3) * 4.0 - 2.0);
                    // At the bins' cull and below it, and above it — which
                    // has to fall back to the full search.
                    for c in [cull, cull * 0.5, cull * 2.0] {
                        let full = distance(p, origin, cell, segments, c);
                        let binned = distance_binned(p, origin, cell, segments, &bins, c);
                        assert_eq!(
                            full.to_bits(),
                            binned.to_bits(),
                            "kind {n} cull {c} at {p:?}"
                        );
                    }
                }
            }
        }
    }

    /// The closest two different yarns of a textile cell come, centreline to
    /// centreline, over the cell and its neighbours — which caps the diameter
    /// at which they stay separate.
    fn yarn_gap(kind: Strut, yarns: usize) -> f32 {
        let segments = kind.segments();
        let per = segments.len() / yarns;
        let mut gap = f32::INFINITY;
        for (i, a) in segments.iter().enumerate() {
            for (j, b) in segments.iter().enumerate() {
                // A yarn's own images in other cells are the same yarn running
                // on, or a parallel one a whole cell away; the constraint is
                // between different yarns.
                if i / per == j / per {
                    continue;
                }
                for o in [
                    [0.0f32, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                ] {
                    // Sample one segment densely against the other.
                    for s in 0..=16 {
                        let t = s as f32 / 16.0;
                        let p = Vector3::new(
                            a[0][0] + (a[1][0] - a[0][0]) * t,
                            a[0][1] + (a[1][1] - a[0][1]) * t,
                            a[0][2] + (a[1][2] - a[0][2]) * t,
                        );
                        let q0 = Vector3::new(b[0][0] + o[0], b[0][1] + o[1], b[0][2] + o[2]);
                        let q1 = Vector3::new(b[1][0] + o[0], b[1][1] + o[1], b[1][2] + o[2]);
                        gap = gap.min(segment_distance(p, q0, q1));
                    }
                }
            }
        }
        gap
    }

    #[test]
    fn textile_yarns_keep_apart() {
        // Weave: centrelines pass 2 × 0.125 apart where they cross.
        let weave = yarn_gap(Strut::Weave, 4);
        assert!((weave - 0.25).abs() < 0.01, "weave gap {weave}");
        // The plait's strands pass closest where they cross over: 0.148.
        let braid = yarn_gap(Strut::Braid, 3);
        assert!(
            (braid - 0.148).abs() < 0.005,
            "braid strands come within {braid}"
        );
        // Six on a ring of 0.3 are a chord of 0.3 apart at any one height, but
        // a fibre leans 28° off horizontal, so its neighbour is nearer across
        // the lean: a sixth of a cell up, times cos 28°, is 0.147.
        let rope = yarn_gap(Strut::HelixBundle, 6);
        assert!((rope - 0.147).abs() < 0.005, "helix gap {rope}");
    }

    #[test]
    fn segments_stay_inside_their_cell() {
        // A segment reaching outside [0, 1]³ would need a wider neighbourhood
        // than the 3×3×3 the culling assumes.
        for kind in Strut::ALL {
            for seg in kind.segments() {
                for end in seg {
                    for a in end {
                        assert!(
                            (0.0..=1.0).contains(a),
                            "{} leaves the unit cell at {a}",
                            kind.name()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn field_is_periodic() {
        for kind in Strut::ALL {
            for p in [
                Vector3::new(0.31, 0.62, 0.17),
                Vector3::new(0.5, 0.5, 0.5),
                Vector3::new(0.9, 0.05, 0.44),
            ] {
                let base = dist(kind, p);
                for shift in [
                    Vector3::new(1.0, 0.0, 0.0),
                    Vector3::new(0.0, 3.0, 0.0),
                    Vector3::new(-2.0, 1.0, 4.0),
                ] {
                    let moved = dist(kind, p + shift);
                    assert!(
                        (base - moved).abs() < 1e-4,
                        "{} is not periodic: {base} vs {moved}",
                        kind.name()
                    );
                }
            }
        }
    }

    #[test]
    fn nodes_sit_on_the_lattice() {
        // Every topology built on the cubic lattice passes through the cell
        // corners, so the corner is a node and the distance there is zero.
        // Kelvin is the exception and has to be: its cells are bubbles centred
        // on the cubic lattice, so a corner is the middle of one — the furthest
        // point from any strut rather than the nearest.
        // Textiles have no nodes at all.
        for kind in Strut::ALL
            .into_iter()
            .filter(|k| *k != Strut::Kelvin && !k.is_textile())
        {
            assert!(dist(kind, Vector3::ZERO) < 1e-5, "{}", kind.name());
            assert!(
                dist(kind, Vector3::new(1.0, 1.0, 0.0)) < 1e-5,
                "{}",
                kind.name()
            );
        }
        assert!(dist(Strut::Bcc, Vector3::new(0.5, 0.5, 0.5)) < 1e-5);
        assert!(dist(Strut::Fcc, Vector3::new(0.5, 0.5, 0.0)) < 1e-5);
        assert!(dist(Strut::Diamond, Vector3::new(0.25, 0.25, 0.25)) < 1e-5);
        // A Kelvin node is a vertex of the truncated octahedron.
        assert!(dist(Strut::Kelvin, Vector3::new(1.0, 0.75, 0.5)) < 1e-5);
        assert!(dist(Strut::Kelvin, Vector3::ZERO) > 0.3, "corner is a void");
    }

    #[test]
    fn names_round_trip() {
        for kind in Strut::ALL {
            assert_eq!(Strut::from_name(kind.name()), Some(kind));
        }
        assert_eq!(Strut::from_name("not-a-lattice"), None);
    }

    #[test]
    fn cell_interiors_stay_open() {
        // A lattice whose voids had closed up would be a solid block; the
        // furthest point from any strut has to be a real distance away.
        for kind in Strut::ALL {
            let p = match kind {
                // BCC-family cells have a node at the centre; probe off it.
                Strut::Bcc | Strut::BccZ => Vector3::new(0.5, 0.5, 0.15),
                // A weft runs through the centre; the gap is between yarns.
                Strut::Weave => Vector3::new(0.25, 0.5, 0.5),
                // The braid plaits up the cell's axis; its column's corner is
                // the open space between columns.
                Strut::Braid => Vector3::new(0.05, 0.05, 0.5),
                _ => Vector3::new(0.5, 0.5, 0.5),
            };
            assert!(dist(kind, p) > 0.1, "{} has no void at {p:?}", kind.name());
        }
    }

    #[test]
    fn culling_matches_the_full_search() {
        // The cull is an optimisation, not a different field: inside the
        // radius it must agree with an uncapped search exactly.
        for kind in Strut::ALL {
            for i in 0..40 {
                let t = i as f32 / 40.0;
                let p = Vector3::new(t * 1.7 - 0.3, t * t * 2.0, 0.4 - t);
                let full = dist(kind, p);
                let culled = distance(p, Vector3::ZERO, CELL, kind.segments(), 0.3);
                if full < 0.3 {
                    assert!((full - culled).abs() < 1e-5, "{}", kind.name());
                } else {
                    assert_eq!(culled, 0.3, "{}", kind.name());
                }
            }
        }
    }

    #[test]
    fn the_kelvin_cell_is_a_truncated_octahedron() {
        let edges = kelvin();
        assert_eq!(edges.len(), 36, "a truncated octahedron has 36 edges");

        // 24 vertices, each on exactly 3 edges — the defining property of a
        // foam node, where three films meet.
        let key = |v: &[f32; 3]| {
            (
                (v[0] * 1000.0).round() as i32,
                (v[1] * 1000.0).round() as i32,
                (v[2] * 1000.0).round() as i32,
            )
        };
        let mut degree: std::collections::HashMap<(i32, i32, i32), usize> =
            std::collections::HashMap::new();
        for e in edges {
            for v in e {
                *degree.entry(key(v)).or_insert(0) += 1;
            }
        }
        assert_eq!(degree.len(), 24, "a truncated octahedron has 24 vertices");
        assert!(
            degree.values().all(|&d| d == 3),
            "every node joins exactly three struts"
        );
    }

    #[test]
    fn kelvin_corner_cells_are_covered() {
        // Only the centre-of-cell octahedron is listed, on the claim that the
        // corner-centred ones add no edges of their own. Their edges have to
        // come out solid anyway, or the foam has gaps along every cell corner.
        // The corner cell at the origin has vertices at every permutation of
        // (±½, ±¼, 0); these are midpoints of three of its edges.
        for mid in [
            Vector3::new(0.5, 0.125, 0.125),
            Vector3::new(0.125, 0.5, 0.125),
            Vector3::new(0.125, 0.125, 0.5),
            Vector3::new(0.375, 0.375, 0.0),
        ] {
            let d = dist(Strut::Kelvin, mid);
            assert!(d < 1e-4, "corner cell edge missed at {mid:?}: {d}");
        }
    }

    #[test]
    fn anisotropic_cells_scale_the_struts() {
        let cell = Vector3::new(2.0, 1.0, 1.0);
        let segs = Strut::Cubic.segments();
        // Halfway along the stretched x strut is still on it.
        let on = distance(Vector3::new(1.0, 0.0, 0.0), Vector3::ZERO, cell, segs, 10.0);
        assert!(on < 1e-5);
        // And the neighbouring x strut is a full cell away in y.
        let off = distance(Vector3::new(1.0, 0.5, 0.0), Vector3::ZERO, cell, segs, 10.0);
        assert!((off - 0.5).abs() < 1e-5, "{off}");
    }
}
