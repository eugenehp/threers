//! Face-connected cuboctahedra voxels — Jenett et al., Sci. Adv. 6:eabc9943 (2020).
//!
//! A voxel is six planar faces on the cube; each face is a diamond of beams
//! between the four edge midpoints. Tiling only the three low faces reproduces
//! the cuboct lattice without listing any beam twice, the same bookkeeping as
//! [`super::Strut::Fcc`].
//!
//! Beam *shape* is what picks the metamaterial, not connectivity:
//!
//! | Cell | Local mechanism | [`Lattice::shape`](super::Lattice::shape) |
//! |------|-----------------|-------------------------------------------|
//! | [`Cuboct::Rigid`] | Straight beams, stretch-dominated once neighbours triangulate | ignored |
//! | [`Cuboct::Compliant`] | In-plane corrugated flexures | amplitude \(a/P\) |
//! | [`Cuboct::Auxetic`] | Reentrant face mechanisms | indent \(d/P\) |
//! | [`Cuboct::ChiralCw`] / [`Cuboct::ChiralCcw`] | Rotated inner motif | radius \(r/P\) |
//!
//! Chirality can also be programmed per cell (and, with [`ChiralRule`], per
//! face) so a column can twist without internal faces cancelling.

use super::strut::{segment_distance, Segment};
use crate::math::Vector3;
use std::f32::consts::TAU;

/// Upper bound on segments emitted for one cell. Compliant is the worst case:
/// 3 faces × 4 beams × 8 pieces.
const MAX_SEGS: usize = 128;
const CORRUGATION_STEPS: usize = 8;
const CORRUGATION_PERIODS: f32 = 2.0;
/// Pinwheel offset, in radians, of the inner motif relative to the outer nodes.
const CHIRAL_DELTA: f32 = 0.4;

/// Four edge-midpoint nodes of a cube face, in the face's UV square.
/// Walking order: south, east, north, west.
pub(crate) const FACE_NODES_UV: [[f32; 2]; 4] = [[0.5, 0.0], [1.0, 0.5], [0.5, 1.0], [0.0, 0.5]];

/// Handedness of a chiral face, looking along the face normal into the cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Hand {
    /// Clockwise pinwheel.
    Cw,
    /// Counter-clockwise pinwheel.
    Ccw,
}

/// How to orient chiral faces across a column so neighbouring cells do not
/// cancel each other's twist.
///
/// The paper's experimental columns are an odd or even voxel width; each width
/// has a rule that was arrived at empirically. Both are hierarchical: a rule-1
/// 5×5 contains a 3×3 and a 1×1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChiralRule {
    /// Odd column widths. Face chirality points away from the column interior.
    R1,
    /// Even column widths. Interior faces run clockwise around the load axis,
    /// used when rule 1 leaves those faces ambiguous.
    R2,
}

/// A face-connected cuboctahedron voxel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cuboct {
    /// Straight beams. Linear modulus–density scaling once the cell count is
    /// large enough that internal beams dominate.
    Rigid,
    /// Corrugated flexure beams. Near-quadratic scaling and a Poisson ratio
    /// that drops toward zero as the corrugation grows.
    Compliant,
    /// Reentrant faces. Negative Poisson ratio, stronger as the indent grows.
    Auxetic,
    /// Chiral faces, clockwise about each face normal.
    ChiralCw,
    /// Chiral faces, counter-clockwise about each face normal.
    ChiralCcw,
}

impl Cuboct {
    /// Every cell type, in declaration order.
    pub const ALL: [Cuboct; 5] = [
        Cuboct::Rigid,
        Cuboct::Compliant,
        Cuboct::Auxetic,
        Cuboct::ChiralCw,
        Cuboct::ChiralCcw,
    ];

    /// Lower-case identifier, for logs and CLI arguments.
    pub fn name(self) -> &'static str {
        match self {
            Cuboct::Rigid => "cuboct",
            Cuboct::Compliant => "cuboct-compliant",
            Cuboct::Auxetic => "cuboct-auxetic",
            Cuboct::ChiralCw => "cuboct-chiral-cw",
            Cuboct::ChiralCcw => "cuboct-chiral-ccw",
        }
    }

    /// The cell with this [`name`](Self::name), if any.
    pub fn from_name(name: &str) -> Option<Cuboct> {
        Cuboct::ALL.into_iter().find(|c| c.name() == name)
    }

    /// Default [`Lattice::shape`](super::Lattice::shape) for this cell: the
    /// paper's mid-range amplitude, indent, or chiral radius, or zero for
    /// rigid beams that do not read it.
    pub fn default_shape(self) -> f32 {
        match self {
            Cuboct::Rigid => 0.0,
            Cuboct::Compliant | Cuboct::Auxetic | Cuboct::ChiralCw | Cuboct::ChiralCcw => 0.15,
        }
    }

    /// Whether this cell is a chiral pinwheel (either hand).
    pub fn is_chiral(self) -> bool {
        matches!(self, Cuboct::ChiralCw | Cuboct::ChiralCcw)
    }

    /// Half-and-half chiral column used in the paper's twist tests: the top
    /// half is CCW, the bottom half CW, so net twist peaks at the midplane
    /// while the ends can be held fixed.
    ///
    /// `n` is the column width in voxels, `m` the height; `k` is the voxel
    /// index along the load axis. Pair with [`ChiralRule`] on the lattice so
    /// the in-plane faces follow rule 1 or 2.
    pub fn column_half(k: i32, m: i32) -> Cuboct {
        if k * 2 >= m {
            Cuboct::ChiralCcw
        } else {
            Cuboct::ChiralCw
        }
    }

    /// Beams of one cell at the origin, in unit-cell coordinates.
    pub fn segments(self, shape: f32) -> Vec<Segment> {
        cell_segments(self, shape, 0, 0, 0, None, 0)
    }

    /// Handedness of a chiral face at lattice cell `(i, j, k)`. Non-chiral
    /// cells return [`Hand::Ccw`] and ignore `rule`.
    pub fn hand_on_face(
        self,
        axis: usize,
        i: i32,
        j: i32,
        k: i32,
        rule: Option<ChiralRule>,
        n: i32,
    ) -> Hand {
        face_hand(self, axis, i, j, k, rule, n)
    }
}

/// Beams of one cube face, in the face UV square. Axis 2 so `x,y` are `u,v`.
pub(crate) fn uv_beams(kind: Cuboct, shape: f32, hand: Hand) -> Vec<[[f32; 2]; 2]> {
    let mut buf = dummy_buf();
    let mut len = 0;
    let shape = shape.clamp(0.0, 0.35);
    match kind {
        Cuboct::Rigid => rigid_face(&mut buf, &mut len, 2),
        Cuboct::Compliant => compliant_face(&mut buf, &mut len, 2, shape),
        Cuboct::Auxetic => auxetic_face(&mut buf, &mut len, 2, shape),
        Cuboct::ChiralCw | Cuboct::ChiralCcw => {
            chiral_face(&mut buf, &mut len, 2, shape.max(0.04), hand);
        }
    }
    buf[..len]
        .iter()
        .map(|s| [[s[0][0], s[0][1]], [s[1][0], s[1][1]]])
        .collect()
}

pub(crate) fn cell_segments(
    kind: Cuboct,
    shape: f32,
    i: i32,
    j: i32,
    k: i32,
    rule: Option<ChiralRule>,
    n: i32,
) -> Vec<Segment> {
    let mut buf = dummy_buf();
    let len = fill(&mut buf, kind, shape, i, j, k, rule, n);
    buf[..len].to_vec()
}

fn dummy_buf() -> [Segment; MAX_SEGS] {
    [[[0.0; 3]; 2]; MAX_SEGS]
}

#[allow(clippy::too_many_arguments)]
fn fill(
    out: &mut [Segment],
    kind: Cuboct,
    shape: f32,
    i: i32,
    j: i32,
    k: i32,
    rule: Option<ChiralRule>,
    n: i32,
) -> usize {
    let mut len = 0;
    let shape = shape.clamp(0.0, 0.35);
    for axis in 0..3 {
        match kind {
            Cuboct::Rigid => rigid_face(out, &mut len, axis),
            Cuboct::Compliant => compliant_face(out, &mut len, axis, shape),
            Cuboct::Auxetic => auxetic_face(out, &mut len, axis, shape),
            Cuboct::ChiralCw | Cuboct::ChiralCcw => {
                let hand = face_hand(kind, axis, i, j, k, rule, n);
                chiral_face(out, &mut len, axis, shape.max(0.04), hand);
            }
        }
    }
    len
}

fn face_hand(
    kind: Cuboct,
    axis: usize,
    i: i32,
    j: i32,
    _k: i32,
    rule: Option<ChiralRule>,
    n: i32,
) -> Hand {
    let uniform = match kind {
        Cuboct::ChiralCw => Hand::Cw,
        _ => Hand::Ccw,
    };
    let Some(rule) = rule else {
        return uniform;
    };
    if n <= 0 {
        return uniform;
    }
    match rule {
        ChiralRule::R1 => {
            // Horizontal faces keep the half-column hand; vertical faces
            // point away from the interior.
            if axis == 2 {
                return uniform;
            }
            let coord = if axis == 0 { i } else { j };
            if coord * 2 < n {
                Hand::Ccw
            } else {
                Hand::Cw
            }
        }
        ChiralRule::R2 => {
            // Circumferential CW around z on the vertical faces.
            match axis {
                0 => {
                    if j * 2 < n {
                        Hand::Cw
                    } else {
                        Hand::Ccw
                    }
                }
                1 => {
                    if i * 2 < n {
                        Hand::Ccw
                    } else {
                        Hand::Cw
                    }
                }
                _ => uniform,
            }
        }
    }
}

fn face_xyz(axis: usize, u: f32, v: f32) -> [f32; 3] {
    match axis {
        0 => [0.0, u, v],
        1 => [u, 0.0, v],
        _ => [u, v, 0.0],
    }
}

fn face_center(axis: usize) -> [f32; 3] {
    face_xyz(axis, 0.5, 0.5)
}

fn node(axis: usize, i: usize) -> [f32; 3] {
    let uv = FACE_NODES_UV[i];
    face_xyz(axis, uv[0], uv[1])
}

fn push(out: &mut [Segment], len: &mut usize, a: [f32; 3], b: [f32; 3]) {
    if *len < out.len() {
        out[*len] = [a, b];
        *len += 1;
    }
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn lerp(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    add(a, scale(sub(b, a), t))
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn length(a: [f32; 3]) -> f32 {
    dot(a, a).sqrt()
}

fn rigid_face(out: &mut [Segment], len: &mut usize, axis: usize) {
    for i in 0..4 {
        push(out, len, node(axis, i), node(axis, (i + 1) % 4));
    }
}

fn compliant_face(out: &mut [Segment], len: &mut usize, axis: usize, amp: f32) {
    let c = face_center(axis);
    for i in 0..4 {
        corrugated(out, len, node(axis, i), node(axis, (i + 1) % 4), c, amp);
    }
}

fn auxetic_face(out: &mut [Segment], len: &mut usize, axis: usize, d: f32) {
    let c = face_center(axis);
    for i in 0..4 {
        reentrant(out, len, node(axis, i), node(axis, (i + 1) % 4), c, d);
    }
}

fn chiral_face(out: &mut [Segment], len: &mut usize, axis: usize, r: f32, hand: Hand) {
    let delta = match hand {
        Hand::Ccw => CHIRAL_DELTA,
        Hand::Cw => -CHIRAL_DELTA,
    };
    let mut inner = [[0.0f32; 3]; 4];
    for i in 0..4 {
        let uv = FACE_NODES_UV[i];
        let ang = (uv[1] - 0.5).atan2(uv[0] - 0.5) + delta;
        inner[i] = face_xyz(axis, 0.5 + r * ang.cos(), 0.5 + r * ang.sin());
    }
    for (i, &mid) in inner.iter().enumerate() {
        let a = node(axis, i);
        let b = node(axis, (i + 1) % 4);
        push(out, len, a, mid);
        push(out, len, mid, b);
    }
}

fn corrugated(
    out: &mut [Segment],
    len: &mut usize,
    a: [f32; 3],
    b: [f32; 3],
    center: [f32; 3],
    amp: f32,
) {
    if amp < 1e-4 {
        push(out, len, a, b);
        return;
    }
    let ab = sub(b, a);
    let ab_len = length(ab);
    if ab_len < 1e-8 {
        return;
    }
    let mid = lerp(a, b, 0.5);
    let to_c = sub(center, mid);
    let mut perp = sub(to_c, scale(ab, dot(to_c, ab) / (ab_len * ab_len)));
    let pl = length(perp);
    if pl < 1e-8 {
        push(out, len, a, b);
        return;
    }
    perp = scale(perp, amp / pl);
    let mut prev = a;
    for s in 1..=CORRUGATION_STEPS {
        let t = s as f32 / CORRUGATION_STEPS as f32;
        // One-sided: 0 at the nodes, peak toward the face interior. A signed
        // sine would bow the other way, off the cube face and out of the cell.
        let wave = 0.5 * (1.0 - (TAU * CORRUGATION_PERIODS * t).cos());
        let p = add(lerp(a, b, t), scale(perp, wave));
        push(out, len, prev, p);
        prev = p;
    }
}

fn reentrant(
    out: &mut [Segment],
    len: &mut usize,
    a: [f32; 3],
    b: [f32; 3],
    center: [f32; 3],
    d: f32,
) {
    if d < 1e-4 {
        push(out, len, a, b);
        return;
    }
    let mid = lerp(a, b, 0.5);
    let to_c = sub(center, mid);
    let pl = length(to_c);
    if pl < 1e-8 {
        push(out, len, a, b);
        return;
    }
    let indent = add(mid, scale(to_c, d / pl));
    push(out, len, a, indent);
    push(out, len, indent, b);
}

/// Distance from `p` to the nearest cuboct beam, giving up at `cull`.
///
/// `cell_kind` is the voxel type at integer cell coordinates; homogeneous
/// lattices pass a constant. `n` is the column width [`ChiralRule`] reads.
#[allow(clippy::too_many_arguments)]
pub(crate) fn distance(
    p: Vector3,
    origin: Vector3,
    cell: Vector3,
    cull: f32,
    mut cell_kind: impl FnMut(i32, i32, i32) -> Cuboct,
    shape: f32,
    rule: Option<ChiralRule>,
    n: i32,
) -> f32 {
    let u = [
        (p.x - origin.x) / cell.x,
        (p.y - origin.y) / cell.y,
        (p.z - origin.z) / cell.z,
    ];
    let base = [u[0].floor(), u[1].floor(), u[2].floor()];
    let mut best = cull;
    let mut buf = dummy_buf();

    for oz in -1..=1 {
        for oy in -1..=1 {
            for ox in -1..=1 {
                let c = [
                    base[0] + ox as f32,
                    base[1] + oy as f32,
                    base[2] + oz as f32,
                ];
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
                    let world = out
                        * match a {
                            0 => cell.x,
                            1 => cell.y,
                            _ => cell.z,
                        };
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
                let ix = c[0] as i32;
                let iy = c[1] as i32;
                let iz = c[2] as i32;
                let len = fill(&mut buf, cell_kind(ix, iy, iz), shape, ix, iy, iz, rule, n);
                for seg in &buf[..len] {
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

#[cfg(test)]
mod tests {
    use super::*;

    const CELL: Vector3 = Vector3::ONE;

    fn dist(kind: Cuboct, shape: f32, p: Vector3) -> f32 {
        distance(p, Vector3::ZERO, CELL, 10.0, |_, _, _| kind, shape, None, 0)
    }

    #[test]
    fn segments_stay_inside_their_cell() {
        for kind in Cuboct::ALL {
            let shape = kind.default_shape();
            for seg in kind.segments(shape) {
                for end in seg {
                    for a in end {
                        assert!(
                            (-1e-4..=1.0 + 1e-4).contains(&a),
                            "{} leaves the unit cell at {a} (shape {shape})",
                            kind.name()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn rigid_has_twelve_beams() {
        assert_eq!(Cuboct::Rigid.segments(0.0).len(), 12);
    }

    #[test]
    fn auxetic_at_zero_indent_matches_rigid_count() {
        // d = 0 collapses each side to a single beam.
        assert_eq!(Cuboct::Auxetic.segments(0.0).len(), 12);
    }

    #[test]
    fn field_is_periodic() {
        for kind in Cuboct::ALL {
            let shape = kind.default_shape();
            for p in [
                Vector3::new(0.31, 0.62, 0.17),
                Vector3::new(0.5, 0.5, 0.5),
                Vector3::new(0.9, 0.05, 0.44),
            ] {
                let base = dist(kind, shape, p);
                for shift in [
                    Vector3::new(1.0, 0.0, 0.0),
                    Vector3::new(0.0, 3.0, 0.0),
                    Vector3::new(-2.0, 1.0, 4.0),
                ] {
                    let moved = dist(kind, shape, p + shift);
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
    fn edge_midpoints_are_nodes() {
        // Every cuboct face is built on the twelve edge midpoints of the cube,
        // so those points sit on the lattice for every part type.
        for kind in Cuboct::ALL {
            let shape = kind.default_shape();
            for p in [
                Vector3::new(0.5, 0.0, 0.0),
                Vector3::new(0.0, 0.5, 0.0),
                Vector3::new(0.0, 0.0, 0.5),
                Vector3::new(1.0, 0.5, 0.0),
            ] {
                let d = dist(kind, shape, p);
                assert!(d < 1e-4, "{} missed node at {p:?}: {d}", kind.name());
            }
        }
    }

    #[test]
    fn cell_interior_stays_open() {
        for kind in Cuboct::ALL {
            let d = dist(kind, kind.default_shape(), Vector3::new(0.5, 0.5, 0.5));
            assert!(d > 0.15, "{} has no void at the centre: {d}", kind.name());
        }
    }

    #[test]
    fn compliant_at_zero_amplitude_matches_rigid() {
        let p = Vector3::new(0.25, 0.25, 0.0);
        let rigid = dist(Cuboct::Rigid, 0.0, p);
        let flat = dist(Cuboct::Compliant, 0.0, p);
        assert!((rigid - flat).abs() < 1e-5, "{rigid} vs {flat}");
    }

    #[test]
    fn chiral_hands_disagree_off_the_spokes() {
        // A point on the pinwheel's inner motif of one hand should not sit on
        // the other hand's beams.
        let p = Vector3::new(0.5 + 0.15, 0.5, 0.0);
        let cw = dist(Cuboct::ChiralCw, 0.15, p);
        let ccw = dist(Cuboct::ChiralCcw, 0.15, p);
        assert!(
            (cw - ccw).abs() > 1e-3,
            "CW and CCW collapsed at {p:?}: {cw} vs {ccw}"
        );
    }

    #[test]
    fn names_round_trip() {
        for kind in Cuboct::ALL {
            assert_eq!(Cuboct::from_name(kind.name()), Some(kind));
        }
        assert_eq!(Cuboct::from_name("cuboct-chiral"), None);
    }

    #[test]
    fn column_half_splits_at_midplane() {
        assert_eq!(Cuboct::column_half(0, 4), Cuboct::ChiralCw);
        assert_eq!(Cuboct::column_half(1, 4), Cuboct::ChiralCw);
        assert_eq!(Cuboct::column_half(2, 4), Cuboct::ChiralCcw);
        assert_eq!(Cuboct::column_half(3, 4), Cuboct::ChiralCcw);
    }

    #[test]
    fn fill_never_overruns_the_buffer() {
        for kind in Cuboct::ALL {
            let n = kind.segments(kind.default_shape()).len();
            assert!(n <= MAX_SEGS, "{} emitted {n} segments", kind.name());
            assert!(n > 0);
        }
        let wavy = Cuboct::Compliant.segments(0.35).len();
        assert!(wavy <= MAX_SEGS, "max-amp compliant emitted {wavy}");
    }
}
