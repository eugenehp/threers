//! The wall patterns a 3D printer's slicer offers as infill.
//!
//! Where a [`Tpms`](super::Tpms) is a curved surface and a
//! [`Strut`](super::Strut) is a beam cell, these are what a slicer draws:
//! families of straight walls, repeated across the part. Each pattern reduces
//! to *distance to the nearest wall*, and the solid is everything within half a
//! line width of one — so the same field pipeline meshes them, and the same
//! density fit sizes them.
//!
//! Two of them are layer-dependent, exactly as they are on the printer:
//! [`Infill::Rectilinear`] turns ninety degrees each layer, and
//! [`Infill::QuarterCubic`] shifts every other layer of cubes by half a cube.
//! One layer's walls sit on the previous layer's, and it is those crossings
//! that tie the pattern into one solid rather than a stack of loose fins.
//!
//! # What is not here
//!
//! - **Gyroid** — it is a minimal surface, not a wall pattern:
//!   [`Tpms::Gyroid`](super::Tpms::Gyroid).
//! - **Cubic subdivision / adaptive cubic** — [`Infill::Cubic`] with a
//!   [`grade`](super::Lattice::grade) that thickens near the surface. The
//!   pattern is the same; what varies is the density, which is what grading is.
//! - **Lightning** — a branching tree grown from the overhangs it has to hold
//!   up. It is not periodic and it is not a field, so it needs a different
//!   generator entirely.
//! - **Cross and Cross 3D** — space-filling curves. Same reason: the pattern is
//!   defined by a recursive path, not a distance.

use crate::math::Vector3;

/// A slicer's infill pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Infill {
    /// Parallel walls that turn ninety degrees every layer — the default in
    /// most slicers ("rectilinear", or "lines"). Cheapest to print and stiffest
    /// in-plane; each layer is held up by its crossings with the one below.
    Rectilinear,
    /// Parallel walls in one direction, every layer ("aligned rectilinear").
    /// Directional on purpose: stiff across the walls, compliant along them.
    AlignedRectilinear,
    /// Both wall directions in every layer. Twice the material of rectilinear
    /// at the same spacing, and the walls cross at every intersection rather
    /// than layer by layer.
    Grid,
    /// Three wall families sixty degrees apart, all meeting at a point.
    /// In-plane isotropic, and rigid because every cell is a triangle.
    Triangles,
    /// The same three families, offset so they enclose hexagons instead of
    /// meeting — the "tri-hexagon" or "stars" pattern. Fewer crossings than
    /// [`Triangles`](Self::Triangles), so less material and less drag on the
    /// nozzle.
    TriHexagon,
    /// Hexagonal cells — walls on the boundaries between neighbouring cells.
    /// The most enclosed area per unit of wall, which is why it is the standard
    /// for stiff panels.
    Honeycomb,
    /// Cubes standing on a corner: three wall families whose normals are the
    /// faces of a cube tipped onto its body diagonal. Equal stiffness on all
    /// three axes, and no wall is horizontal, so nothing has to bridge.
    Cubic,
    /// [`Cubic`](Self::Cubic) with every other layer of cubes shifted half a
    /// cube, which splits the cells into alternating tetrahedra and octahedra.
    /// More crossings per unit of wall than cubic, and more isotropic.
    QuarterCubic,
    /// Rings following the part outline inwards, one every spacing, extruded
    /// through the layers — a slicer's concentric infill. The first ring sits
    /// half a spacing in, clear of the perimeter.
    ///
    /// Against the bounding box the offsets are measured in the layer plane, so
    /// the rings come out as vertical tubes the way a slicer draws them. Given
    /// a [`fill`](super::Lattice::fill) region or a
    /// [`trim`](super::Lattice::trim), they follow *its* field instead, which
    /// is a three-dimensional distance — so the result is nested shells rather
    /// than extruded rings. For a sphere that is the shape you wanted anyway;
    /// for a prism it is not, and a
    /// [`Region::new`](super::Region::new) whose field ignores `z` gets the
    /// rings back.
    Concentric,
}

impl Infill {
    /// Every pattern, in declaration order.
    pub const ALL: [Infill; 9] = [
        Infill::Rectilinear,
        Infill::AlignedRectilinear,
        Infill::Grid,
        Infill::Triangles,
        Infill::TriHexagon,
        Infill::Honeycomb,
        Infill::Cubic,
        Infill::QuarterCubic,
        Infill::Concentric,
    ];

    /// Lower-case identifier, for logs and CLI arguments.
    pub fn name(self) -> &'static str {
        match self {
            Infill::Rectilinear => "rectilinear",
            Infill::AlignedRectilinear => "aligned-rectilinear",
            Infill::Grid => "grid",
            Infill::Triangles => "triangles",
            Infill::TriHexagon => "tri-hexagon",
            Infill::Honeycomb => "honeycomb",
            Infill::Cubic => "cubic",
            Infill::QuarterCubic => "quarter-cubic",
            Infill::Concentric => "concentric",
        }
    }

    /// The pattern with this [`name`](Self::name), if any.
    pub fn from_name(name: &str) -> Option<Infill> {
        Infill::ALL.into_iter().find(|i| i.name() == name)
    }

    /// True when the pattern reads the layer it is on, and so changes with
    /// height rather than repeating every cell.
    pub fn is_layered(self) -> bool {
        matches!(self, Infill::Rectilinear | Infill::QuarterCubic)
    }
}

/// The three face normals of a cube tipped onto its body diagonal, so that
/// diagonal points up. They are orthonormal and sum to `√3 ẑ`, which is what
/// makes the pattern equally steep — and equally stiff — on all three axes.
const CUBE_FACES: [[f32; 3]; 3] = [
    // √⅔, −1/√6, ±1/√2 and 1/√3 — the rotation that stands a cube on its
    // corner, written out rather than derived so the constant stays readable.
    [0.816_496_6, 0.0, 0.577_350_3],
    [-0.408_248_3, std::f32::consts::FRAC_1_SQRT_2, 0.577_350_3],
    [-0.408_248_3, -std::f32::consts::FRAC_1_SQRT_2, 0.577_350_3],
];

/// How far a cube on its corner rises per cube: the body diagonal, in cells.
const CUBE_RISE: f32 = 1.732_050_8;

/// `sin 60°`, the row spacing of a triangular lattice with unit pitch.
const ROW: f32 = 0.866_025_4;

/// Distance from `p` to the nearest wall of the pattern.
///
/// `depth` is how far inside the part the point is *in the layer plane*, which
/// only [`Infill::Concentric`] reads — the rest are periodic and do not care
/// where the part's edge is. In the plane, because a slicer offsets each
/// layer's outline: measured in three dimensions the shells would close over
/// the top and bottom and seal a void inside each one.
///
/// # Which patterns the cell stretches
///
/// The axis-aligned patterns — rectilinear, grid, concentric — take one spacing
/// per axis, so an anisotropic cell stretches them and `cells` means exactly
/// what it says on all three axes.
///
/// The rest do not: a regular hexagon does not tile a rectangle, and neither
/// does a cube standing on its corner. Shearing them to fit would leave the
/// walls at the wrong angles to each other, which is the whole point of
/// choosing one. They take their spacing from `cell.x` and keep their
/// proportions, exactly as a slicer does with a single line-distance setting.
pub(crate) fn distance(kind: Infill, p: Vector3, phase: Vector3, cell: Vector3, depth: f32) -> f32 {
    let v = p - phase;
    // Sixty degrees apart in the layer plane, for the triangular patterns.
    const A: [f32; 3] = [0.5, ROW, 0.0];
    const B: [f32; 3] = [-0.5, ROW, 0.0];
    let pitch = cell.x;

    match kind {
        Infill::AlignedRectilinear => axis(v.x, cell.x),
        Infill::Rectilinear => {
            // The layer index, not the height: within a layer the walls run one
            // way, and the layer above crosses them.
            if (v.z / cell.z).floor() as i64 % 2 == 0 {
                axis(v.x, cell.x)
            } else {
                axis(v.y, cell.y)
            }
        }
        Infill::Grid => axis(v.x, cell.x).min(axis(v.y, cell.y)),
        Infill::Triangles => axis(v.x, pitch)
            .min(family(v, A, 0.0, pitch))
            .min(family(v, B, 0.0, pitch)),
        // A third of a spacing between the three families opens a hexagon where
        // they would otherwise have met at a point.
        Infill::TriHexagon => axis(v.x, pitch)
            .min(family(v, A, 1.0 / 3.0, pitch))
            .min(family(v, B, 2.0 / 3.0, pitch)),
        Infill::Honeycomb => honeycomb(v, pitch),
        Infill::Cubic => cube_faces(v, 0.0, pitch),
        Infill::QuarterCubic => {
            let layer = (v.z / (CUBE_RISE * pitch)).floor() as i64;
            // Half a period on all three families at once is a translation of
            // half a body diagonal — that is, half a cube straight up.
            let shift = if layer.rem_euclid(2) == 0 { 0.0 } else { 0.5 };
            cube_faces(v, shift, pitch)
        }
        // Shells at every spacing inwards from the surface, the first of them
        // half a spacing in. Putting one *on* the surface instead would seal
        // the part in a skin and double up on the wall the perimeter already
        // prints there.
        Infill::Concentric => {
            let spacing = cell.x.min(cell.y);
            axis(depth - spacing * 0.5, spacing)
        }
    }
}

/// Distance to the nearest of a set of planes `spacing` apart, one of them
/// through the origin.
#[inline]
fn axis(offset: f32, spacing: f32) -> f32 {
    let t = offset / spacing.max(1e-12);
    (t - t.round()).abs() * spacing
}

/// Distance to the nearest plane of the family `n · v ≡ phase · spacing`, for a
/// unit normal `n` in world space.
#[inline]
fn family(v: Vector3, n: [f32; 3], phase: f32, spacing: f32) -> f32 {
    let t = (n[0] * v.x + n[1] * v.y + n[2] * v.z) / spacing.max(1e-12) - phase;
    (t - t.round()).abs() * spacing
}

#[inline]
fn cube_faces(v: Vector3, phase: f32, spacing: f32) -> f32 {
    family(v, CUBE_FACES[0], phase, spacing)
        .min(family(v, CUBE_FACES[1], phase, spacing))
        .min(family(v, CUBE_FACES[2], phase, spacing))
}

/// Distance to the nearest wall of a hexagonal tiling, extruded in z.
///
/// The hexagons are the Voronoi cells of a triangular lattice of centres, one
/// `pitch` apart, so a wall is a perpendicular bisector between two centres and
/// the distance to it has a closed form. Taking the nearest centre first and
/// then the smallest bisector distance to any other is what makes it exact —
/// the usual shortcut of halving the gap between the two nearest centres is
/// only right when the sample happens to lie between them.
fn honeycomb(v: Vector3, pitch: f32) -> f32 {
    // Rows are `ROW` pitches apart, each offset half a pitch from the last.
    let (x, y) = (v.x / pitch, v.y / (pitch * ROW));
    let j0 = y.round() as i64;
    let i0 = (x - j0 as f32 * 0.5).round() as i64;

    let mut sites = [(0.0f32, Vector3::ZERO); 25];
    let mut count = 0;
    for dj in -2..=2i64 {
        for di in -2..=2i64 {
            let (i, j) = (i0 + di, j0 + dj);
            // Offset from the sample to the centre, in world units. The pattern
            // is extruded, so z plays no part.
            let d = Vector3::new(
                (x - (i as f32 + j as f32 * 0.5)) * pitch,
                (y - j as f32) * pitch * ROW,
                0.0,
            );
            sites[count] = (d.length_sq(), d);
            count += 1;
        }
    }

    let mut nearest = 0;
    for k in 1..count {
        if sites[k].0 < sites[nearest].0 {
            nearest = k;
        }
    }
    let (near_sq, near) = sites[nearest];

    let mut wall = f32::MAX;
    for (k, &(far_sq, far)) in sites.iter().enumerate().take(count) {
        if k == nearest {
            continue;
        }
        // Distance from the sample to the bisector of the two centres.
        let span = (near - far).length();
        if span > 1e-9 {
            wall = wall.min((far_sq - near_sq) / (2.0 * span));
        }
    }
    wall.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CELL: Vector3 = Vector3::ONE;

    fn dist(kind: Infill, p: Vector3) -> f32 {
        distance(kind, p, Vector3::ZERO, CELL, 1.0)
    }

    /// Everything but concentric, which follows the part rather than repeating.
    fn periodic() -> impl Iterator<Item = Infill> {
        Infill::ALL.into_iter().filter(|k| *k != Infill::Concentric)
    }

    /// Translations that must leave a pattern exactly where it was, for a unit
    /// cell. Only the axis-aligned patterns repeat every cell: the triangular
    /// and cube-on-corner ones keep their own proportions, so their periods are
    /// their own.
    fn periods(kind: Infill) -> Vec<Vector3> {
        let z = Vector3::new(0.0, 0.0, 1.0);
        match kind {
            Infill::AlignedRectilinear => vec![Vector3::new(1.0, 0.0, 0.0), z * 5.0],
            Infill::Rectilinear => vec![
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
                // Two layers, because the walls turn on every one.
                z * 2.0,
            ],
            Infill::Grid => vec![
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
                z * 7.0,
            ],
            Infill::Triangles | Infill::TriHexagon => vec![
                Vector3::new(2.0, 0.0, 0.0),
                Vector3::new(0.0, 1.0 / ROW, 0.0),
                z * 3.0,
            ],
            Infill::Honeycomb => vec![
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::new(0.0, 2.0 * ROW, 0.0),
                z * 4.0,
            ],
            // A cube's own lattice: one face-normal step, and the body diagonal
            // that carries it straight up.
            Infill::Cubic => vec![
                Vector3::new(
                    CUBE_FACES[0][0] - CUBE_FACES[1][0],
                    CUBE_FACES[0][1] - CUBE_FACES[1][1],
                    CUBE_FACES[0][2] - CUBE_FACES[1][2],
                ),
                z * CUBE_RISE,
            ],
            Infill::QuarterCubic => vec![
                Vector3::new(
                    CUBE_FACES[0][0] - CUBE_FACES[1][0],
                    CUBE_FACES[0][1] - CUBE_FACES[1][1],
                    CUBE_FACES[0][2] - CUBE_FACES[1][2],
                ),
                // Twice, because every other layer of cubes is shifted.
                z * (2.0 * CUBE_RISE),
            ],
            Infill::Concentric => vec![],
        }
    }

    #[test]
    fn every_pattern_repeats_on_its_own_lattice() {
        for kind in periodic() {
            for p in [
                Vector3::new(0.21, 0.37, 0.63),
                Vector3::new(0.5, 0.5, 0.1),
                Vector3::new(0.83, 0.04, 0.29),
            ] {
                let base = dist(kind, p);
                for shift in periods(kind) {
                    for repeat in [1.0, -1.0, 3.0] {
                        let moved = dist(kind, p + shift * repeat);
                        assert!(
                            (base - moved).abs() < 1e-4,
                            "{} is not periodic over {shift:?}: {base} vs {moved}",
                            kind.name()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn walls_pass_through_the_cell_origin() {
        // Every line pattern is phased from the cell corner, so a wall runs
        // through it — otherwise the pattern would drift against the bounds.
        // Honeycomb is the exception: its cell corner is a hexagon's centre.
        for kind in periodic().filter(|k| *k != Infill::Honeycomb) {
            assert!(
                dist(kind, Vector3::ZERO) < 1e-5,
                "{} has no wall at the origin: {}",
                kind.name(),
                dist(kind, Vector3::ZERO)
            );
        }
    }

    #[test]
    fn cells_have_room_between_the_walls() {
        // A pattern whose walls filled the cell would be a solid block. Half a
        // spacing is the most any of these can offer; a tenth is a floor no
        // real pattern should fall under.
        for kind in periodic() {
            let mut best = 0.0f32;
            let n = 32;
            for k in 0..n {
                for j in 0..n {
                    for i in 0..n {
                        // Two cells across, so the patterns whose period is
                        // longer than one still show their widest gap.
                        let s = |a: usize| (a as f32 + 0.5) / n as f32 * 2.0;
                        best = best.max(dist(kind, Vector3::new(s(i), s(j), s(k))));
                    }
                }
            }
            assert!(best > 0.1, "{} leaves no void: {best}", kind.name());
        }
    }

    #[test]
    fn distance_is_a_length() {
        // Walking away from a wall perpendicular to it, the reported distance
        // has to track the distance actually walked. Start at the middle of a
        // cell in y and z so no other family is the nearer one.
        for kind in [Infill::Grid, Infill::AlignedRectilinear] {
            for i in 1..12 {
                let step = i as f32 * 0.02;
                let d = dist(kind, Vector3::new(step, 0.5, 0.5));
                assert!(
                    (d - step).abs() < 1e-4,
                    "{} reported {d} at {step}",
                    kind.name()
                );
            }
        }
    }

    #[test]
    fn anisotropic_cells_stretch_rather_than_clip() {
        // Twice the cell in x, so the walls normal to x are twice as far apart
        // — and the distance halfway between them is a full half-cell.
        let cell = Vector3::new(2.0, 1.0, 1.0);
        let mid = distance(
            Infill::AlignedRectilinear,
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::ZERO,
            cell,
            1.0,
        );
        assert!((mid - 1.0).abs() < 1e-4, "{mid}");
    }

    #[test]
    fn rectilinear_turns_ninety_degrees_each_layer() {
        // On a wall of the first layer, off it on the second.
        let on_x = Vector3::new(0.0, 0.4, 0.5);
        let on_y = Vector3::new(0.4, 0.0, 1.5);
        assert!(dist(Infill::Rectilinear, on_x) < 1e-5);
        assert!(dist(Infill::Rectilinear, on_y) < 1e-5);
        // And the other way round on the layer between.
        assert!(dist(Infill::Rectilinear, Vector3::new(0.4, 0.0, 0.5)) > 0.3);
        assert!(dist(Infill::Rectilinear, Vector3::new(0.0, 0.4, 1.5)) > 0.3);
    }

    #[test]
    fn quarter_cubic_shifts_every_other_layer() {
        // Two samples a whole cube-rise apart see the same pattern only after
        // two layers, not one.
        let p = Vector3::new(0.23, 0.41, 0.15);
        let one = Vector3::new(p.x, p.y, p.z + CUBE_RISE);
        let two = Vector3::new(p.x, p.y, p.z + 2.0 * CUBE_RISE);
        let base = dist(Infill::QuarterCubic, p);
        assert!((base - dist(Infill::QuarterCubic, two)).abs() < 1e-4);
        assert!((base - dist(Infill::QuarterCubic, one)).abs() > 1e-3);
        // Plain cubic repeats every layer, which is the difference.
        assert!((dist(Infill::Cubic, p) - dist(Infill::Cubic, one)).abs() < 1e-4);
    }

    #[test]
    fn honeycomb_rings_each_cell_with_wall() {
        // The origin is a hexagon's centre, so it is the furthest point from
        // any wall: half the pitch, since the walls bisect neighbouring
        // centres one pitch apart.
        let middle = honeycomb(Vector3::ZERO, 1.0);
        assert!(
            (middle - 0.5).abs() < 1e-4,
            "hexagon is not a pitch across: {middle}"
        );
        // Halfway to a neighbouring centre is on the wall between them.
        assert!(honeycomb(Vector3::new(0.5, 0.0, 0.0), 1.0) < 1e-5);
        // And it is a regular hexagon: a corner is where three centres are
        // equidistant, which for an equilateral triangle of centres is its
        // centroid.
        let corner = Vector3::new(0.5, ROW / 3.0, 0.0);
        assert!(honeycomb(corner, 1.0) < 1e-4, "not a hexagon corner");
        // Extruded, so height changes nothing.
        assert!(
            (honeycomb(Vector3::new(0.2, 0.1, 9.0), 1.0)
                - honeycomb(Vector3::new(0.2, 0.1, 0.0), 1.0))
            .abs()
                < 1e-6
        );
    }

    #[test]
    fn concentric_shells_follow_the_boundary() {
        // Shells at half a spacing, then every spacing after — so the surface
        // itself (depth 0) is half a spacing clear of the first one.
        let cell = Vector3::ONE;
        for (depth, want) in [
            (0.0, 0.5),
            (0.5, 0.0),
            (0.75, 0.25),
            (1.5, 0.0),
            (1.9, 0.4),
            (2.5, 0.0),
        ] {
            let d = distance(
                Infill::Concentric,
                Vector3::ZERO,
                Vector3::ZERO,
                cell,
                depth,
            );
            assert!((d - want).abs() < 1e-5, "at depth {depth}: {d} not {want}");
        }
    }
}
