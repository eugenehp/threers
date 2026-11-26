//! Fillets and chamfers on a B-rep body.
//!
//! The same two outcomes the boolean suite allows, and no third: a closed solid
//! of the right volume, or a `BlendDeclined` naming the reason. Every volume
//! here has a closed form, so "right" is a number rather than a look.

#![cfg(feature = "brep-kernel")]

use std::f64::consts::PI;
use threers::brep::{BlendDeclined, Body, BooleanOp, Surface};

const TOL: f64 = 1e-4;

/// Volume of a body's tessellation, by the divergence theorem. Asserts closure
/// on the way past, since an open mesh has no volume to speak of.
fn volume(body: &Body, tolerance: f64) -> f64 {
    let mut b = body.clone();
    b.refine_edges(tolerance);
    let (mesh, report) = b.tessellate(tolerance);
    assert!(
        report.is_closed() && report.carried_through == 0,
        "{} open edges, {} faces unfilled",
        report.boundary_edges,
        report.carried_through
    );
    let pos = &mesh.get_attribute("position").unwrap().array;
    let idx = mesh.index.as_ref().unwrap();
    let v = |i: u32| -> [f64; 3] {
        let o = i as usize * 3;
        [pos[o] as f64, pos[o + 1] as f64, pos[o + 2] as f64]
    };
    idx.as_chunks::<3>()
        .0
        .iter()
        .map(|t| {
            let (a, b, c) = (v(t[0]), v(t[1]), v(t[2]));
            (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]))
                / 6.0
        })
        .sum::<f64>()
        .abs()
}

/// A solid, with the volume it should have.
#[track_caller]
fn assert_solid(body: &Body, expected: f64, relative: f64) {
    assert!(body.is_valid_solid(TOL), "{:?}", body.defects(TOL));
    let shells = body.shells();
    assert_eq!(shells.len(), 1, "one shell");
    let got = volume(body, TOL);
    assert!(
        (got - expected).abs() <= expected * relative,
        "volume {got}, expected {expected}"
    );
}

/// Open edges whose two ends both satisfy `keep`.
fn edges_where(body: &Body, keep: impl Fn([f64; 3]) -> bool) -> Vec<usize> {
    (0..body.edges().len())
        .filter(|&e| {
            let edge = &body.edges()[e];
            let (a, b) = (edge.vertices[0], *edge.vertices.last().unwrap());
            !edge.closed && keep(body.vertices()[a]) && keep(body.vertices()[b])
        })
        .collect()
}

fn closed_edges_where(body: &Body, keep: impl Fn([f64; 3]) -> bool) -> Vec<usize> {
    (0..body.edges().len())
        .filter(|&e| {
            let edge = &body.edges()[e];
            edge.closed && edge.vertices.iter().all(|&v| keep(body.vertices()[v]))
        })
        .collect()
}

fn all_edges(body: &Body) -> Vec<usize> {
    (0..body.edges().len()).collect()
}

fn kinds(body: &Body, kind: &str) -> usize {
    body.faces()
        .iter()
        .filter(|f| body.surfaces()[f.surface].kind() == kind)
        .count()
}

// -- one edge ---------------------------------------------------------------

#[test]
fn chamfering_one_edge_of_a_cube_cuts_off_a_prism() {
    let cube = Body::cuboid([2.0, 2.0, 2.0]);
    let edge = edges_where(&cube, |p| p[0] > 0.0 && p[1] > 0.0);
    assert_eq!(edge.len(), 1);
    let out = cube.chamfer(&edge, 0.5, TOL).unwrap();

    assert_eq!(out.faces().len(), 7);
    assert_eq!(out.edges().len(), 15);
    assert_eq!(out.vertices().len(), 10);
    assert_eq!(out.shells()[0].euler, 2, "10 − 15 + 7");
    assert_solid(&out, 8.0 - 0.5 * 0.5 * 0.5 * 2.0, 1e-6);
}

#[test]
fn filleting_one_edge_of_a_cube_leaves_a_quarter_round() {
    let cube = Body::cuboid([2.0, 2.0, 2.0]);
    let edge = edges_where(&cube, |p| p[0] > 0.0 && p[1] > 0.0);
    let r = 0.5;
    let out = cube.fillet(&edge, r, TOL).unwrap();

    assert_eq!(out.faces().len(), 7);
    assert_eq!(
        kinds(&out, "cylinder"),
        1,
        "the blend is a cylinder, exactly"
    );
    let blend = out
        .surfaces()
        .iter()
        .find(|s| s.kind() == "cylinder")
        .unwrap();
    assert!(matches!(blend, Surface::Cylinder { radius, .. } if (*radius - r).abs() < 1e-12));
    assert_solid(&out, 8.0 - (1.0 - PI / 4.0) * r * r * 2.0, 1e-4);
}

#[test]
fn a_blend_on_a_box_that_is_not_a_cube() {
    // Nothing above distinguishes one axis from another.
    let (x, y, z) = (3.0, 2.0, 5.0);
    let slab = Body::cuboid([x, y, z]);
    for (edges, length) in [
        (edges_where(&slab, |p| p[0] > 0.0 && p[1] > 0.0), z),
        (edges_where(&slab, |p| p[0] < 0.0 && p[2] > 0.0), y),
        (edges_where(&slab, |p| p[1] < 0.0 && p[2] < 0.0), x),
    ] {
        assert_eq!(edges.len(), 1);
        let (d, r) = (0.4, 0.7);
        let bevel = slab.chamfer(&edges, d, TOL).unwrap();
        assert_solid(&bevel, x * y * z - 0.5 * d * d * length, 1e-6);
        let round = slab.fillet(&edges, r, TOL).unwrap();
        assert_solid(&round, x * y * z - (1.0 - PI / 4.0) * r * r * length, 1e-4);
    }
}

// -- corners ----------------------------------------------------------------

#[test]
fn two_chamfers_at_a_corner_mitre() {
    let cube = Body::cuboid([2.0, 2.0, 2.0]);
    // Two edges of the top face, sharing a corner.
    let mut edges = edges_where(&cube, |p| p[0] > 0.0 && p[2] > 0.0);
    edges.extend(edges_where(&cube, |p| p[1] > 0.0 && p[2] > 0.0));
    assert_eq!(edges.len(), 2);
    let d = 0.5;
    let out = cube.chamfer(&edges, d, TOL).unwrap();
    assert_eq!(out.faces().len(), 8);
    // Two prisms, less the wedge both of them remove.
    assert_solid(
        &out,
        8.0 - 2.0 * (0.5 * d * d * 2.0) + d * d * d / 3.0,
        1e-6,
    );
}

#[test]
fn two_fillets_at_a_corner_mitre() {
    let cube = Body::cuboid([2.0, 2.0, 2.0]);
    let mut edges = edges_where(&cube, |p| p[0] > 0.0 && p[2] > 0.0);
    edges.extend(edges_where(&cube, |p| p[1] > 0.0 && p[2] > 0.0));
    let r = 0.5;
    let out = cube.fillet(&edges, r, TOL).unwrap();
    assert_eq!(out.faces().len(), 8);
    assert_eq!(kinds(&out, "cylinder"), 2);
    // ∫₀ʳ (r − √(r² − p²))² dp is what the two quarter-rounds both remove.
    let both = r * r * r * (5.0 / 3.0 - PI / 2.0);
    assert_solid(
        &out,
        8.0 - 2.0 * (1.0 - PI / 4.0) * r * r * 2.0 + both,
        1e-4,
    );
}

#[test]
fn chamfering_every_edge_of_a_cube() {
    let cube = Body::cuboid([2.0, 2.0, 2.0]);
    let d = 0.3;
    let out = cube.chamfer(&all_edges(&cube), d, TOL).unwrap();

    assert_eq!(out.faces().len(), 18, "six faces and twelve bevels");
    assert_eq!(out.edges().len(), 48);
    assert_eq!(out.vertices().len(), 32);
    assert_eq!(out.shells()[0].euler, 2, "32 − 48 + 18");
    // Twelve prisms, less eight corners' worth of pairwise overlap, plus the
    // part all three share: 12·d²·h − 8·d³ + 2·d³ on a cube of half-side h.
    assert_solid(&out, 8.0 - (12.0 * d * d - 6.0 * d * d * d), 1e-6);
}

/// The other shape a chamfered corner can have, chosen in
/// [`Settings`](threers::brep::Settings): a triangular facet through the
/// three points where the bevels' edges meet on the cube's faces. Each
/// corner loses a further tetrahedron of volume `d³ / 12`, and gains a face.
#[test]
fn chamfering_every_edge_of_a_cube_with_faceted_corners() {
    use threers::brep::{with_settings, ChamferCorner, Settings};
    let (s, d) = (2.0, 0.25);
    let cube = Body::cuboid([s, s, s]);
    let pointed = cube.chamfer(&all_edges(&cube), d, TOL).unwrap();
    let faceted = with_settings(
        Settings {
            chamfer_corner: ChamferCorner::Facet,
            ..Settings::default()
        },
        || cube.chamfer(&all_edges(&cube), d, TOL),
    )
    .unwrap();
    let tetrahedra = 8.0 * d * d * d / 12.0;
    assert_solid(&faceted, volume(&pointed, TOL) - tetrahedra, 1e-9);
    // Six faces, twelve bevels, and now eight corners.
    assert_eq!(faceted.faces().len(), pointed.faces().len() + 8);
    assert_eq!(kinds(&faceted, "plane"), 26);
    // The setting was scoped to that call.
    let again = cube.chamfer(&all_edges(&cube), d, TOL).unwrap();
    assert_eq!(again.faces().len(), pointed.faces().len());
}

#[test]
fn filleting_every_edge_of_a_cube_rounds_its_corners_with_spheres() {
    let a = 2.0;
    let cube = Body::cuboid([a, a, a]);
    let r = 0.4;
    let out = cube.fillet(&all_edges(&cube), r, TOL).unwrap();

    assert_eq!(out.faces().len(), 26);
    assert_eq!(kinds(&out, "plane"), 6);
    assert_eq!(kinds(&out, "cylinder"), 12);
    assert_eq!(kinds(&out, "sphere"), 8);
    // A core, six slabs, twelve quarter cylinders and eight octants of a ball.
    let c = a - 2.0 * r;
    let expected = c * c * c + 6.0 * c * c * r + 3.0 * PI * r * r * c + 4.0 / 3.0 * PI * r * r * r;
    assert_solid(&out, expected, 1e-4);
}

#[test]
fn a_rounded_box_with_unequal_sides() {
    let (x, y, z) = (4.0, 3.0, 2.0);
    let slab = Body::cuboid([x, y, z]);
    let r = 0.5;
    let out = slab.fillet(&all_edges(&slab), r, TOL).unwrap();
    let (cx, cy, cz) = (x - 2.0 * r, y - 2.0 * r, z - 2.0 * r);
    let expected = cx * cy * cz
        + 2.0 * r * (cx * cy + cy * cz + cz * cx)
        + PI * r * r * (cx + cy + cz)
        + 4.0 / 3.0 * PI * r * r * r;
    assert_solid(&out, expected, 1e-4);
}

#[test]
fn the_top_of_a_box_rounded_all_the_way_round() {
    // Four edges, every corner a mitre between two of them.
    let slab = Body::cuboid([4.0, 3.0, 2.0]);
    let top = edges_where(&slab, |p| p[2] > 0.0);
    assert_eq!(top.len(), 4);
    let r = 0.5;
    let out = slab.fillet(&top, r, TOL).unwrap();
    assert_eq!(out.faces().len(), 10);
    let quarter = (1.0 - PI / 4.0) * r * r;
    let both = r * r * r * (5.0 / 3.0 - PI / 2.0);
    assert_solid(
        &out,
        24.0 - quarter * (2.0 * 4.0 + 2.0 * 3.0) + 4.0 * both,
        1e-4,
    );
}

// -- concave ----------------------------------------------------------------

/// An L: a 4 × 4 × 2 block with one 2 × 2 quadrant cut away.
fn ell() -> Body {
    let block = Body::cuboid([4.0, 4.0, 2.0]);
    let notch = Body::cuboid([3.0, 3.0, 4.0])
        .translated([1.5, 1.5, 0.0])
        .unwrap();
    block.boolean(&notch, BooleanOp::Difference, TOL).unwrap()
}

#[test]
fn a_concave_edge_gains_material() {
    let ell = ell();
    assert_solid(&ell, 24.0, 1e-6);
    // The inside corner of the L runs along z at the origin.
    let inner = edges_where(&ell, |p| p[0].abs() < 1e-6 && p[1].abs() < 1e-6);
    assert_eq!(inner.len(), 1);

    let r = 0.5;
    let round = ell.fillet(&inner, r, TOL).unwrap();
    assert_solid(&round, 24.0 + (1.0 - PI / 4.0) * r * r * 2.0, 1e-4);

    let d = 0.5;
    let bevel = ell.chamfer(&inner, d, TOL).unwrap();
    assert_solid(&bevel, 24.0 + 0.5 * d * d * 2.0, 1e-6);
}

#[test]
fn a_blend_carried_round_an_inside_corner_mitres_into_it() {
    // The top outline of the L: six convex edges, five outside corners and one
    // inside. At the inside corner the top face is reflex, so its two setback
    // lines cross *behind* the vertex and each blend runs on past the wall it
    // was following, tapering to the mitre.
    let ell = ell();
    let top = edges_where(&ell, |p| p[2] > 0.0);
    assert_eq!(top.len(), 6);

    // Outside, two blends overlap; inside, each gains a taper instead.
    let r = 0.4;
    let round = ell.fillet(&top, r, TOL).unwrap();
    assert_eq!(kinds(&round, "cylinder"), 6);
    let quarter = (1.0 - PI / 4.0) * r * r;
    let both = r * r * r * (5.0 / 3.0 - PI / 2.0);
    assert_solid(&round, 24.0 - quarter * 16.0 + 5.0 * both - both, 1e-4);

    let d = 0.4;
    let bevel = ell.chamfer(&top, d, TOL).unwrap();
    assert_solid(
        &bevel,
        24.0 - 0.5 * d * d * 16.0 + 5.0 * d * d * d / 3.0 - d * d * d / 3.0,
        1e-6,
    );
}

#[test]
fn convex_and_concave_at_one_vertex_is_declined() {
    let ell = ell();
    let inner = edges_where(&ell, |p| p[0].abs() < 1e-6 && p[1].abs() < 1e-6);
    // An edge of the top face that ends at the inside corner.
    let top = edges_where(&ell, |p| p[2] > 0.0 && p[1].abs() < 1e-6);
    assert_eq!(top.len(), 1);
    let got = ell.fillet(&[inner[0], top[0]], 0.3, TOL);
    assert!(
        matches!(got, Err(BlendDeclined::MixedConvexity { .. })),
        "{got:?}"
    );
}

// -- rims -------------------------------------------------------------------

#[test]
fn the_end_of_a_shaft() {
    let (radius, height) = (2.0, 5.0);
    let shaft = Body::cylinder([0.0; 3], [0.0, 0.0, 1.0], radius, height);
    let whole = PI * radius * radius * height;
    let top = closed_edges_where(&shaft, |p| p[2] > 1.0);
    assert_eq!(top.len(), 1);

    // By Pappus: the section removed, times the circle its centroid runs.
    let d = 0.5;
    let bevel = shaft.chamfer(&top, d, TOL).unwrap();
    assert_eq!(kinds(&bevel, "cone"), 1);
    assert_solid(&bevel, whole - PI * d * d * (radius - d / 3.0), 1e-4);

    let r = 0.5;
    let round = shaft.fillet(&top, r, TOL).unwrap();
    assert_eq!(kinds(&round, "torus"), 1);
    let removed = 2.0 * PI * r * r * ((1.0 - PI / 4.0) * (radius - r) + r / 6.0);
    assert_solid(&round, whole - removed, 1e-4);

    // Both ends at once.
    let both = shaft.fillet(&all_edges(&shaft), r, TOL).unwrap();
    assert_eq!(kinds(&both, "torus"), 2);
    assert_solid(&both, whole - 2.0 * removed, 1e-4);
}

#[test]
fn the_mouth_of_a_hole() {
    let radius = 1.0;
    let plate = Body::plate_with_hole([6.0, 6.0, 2.0], radius).unwrap();
    let whole = 72.0 - PI * radius * radius * 2.0;
    assert_solid(&plate, whole, 1e-4);
    let mouth = closed_edges_where(&plate, |p| p[2] > 0.0);
    assert_eq!(mouth.len(), 1);

    // A countersink.
    let d = 0.4;
    let bevel = plate.chamfer(&mouth, d, TOL).unwrap();
    assert_eq!(kinds(&bevel, "cone"), 1);
    assert_solid(&bevel, whole - PI * d * d * (radius + d / 3.0), 1e-4);

    let r = 0.4;
    let round = plate.fillet(&mouth, r, TOL).unwrap();
    assert_eq!(kinds(&round, "torus"), 1);
    let removed = 2.0 * PI * r * r * ((1.0 - PI / 4.0) * (radius + r) - r / 6.0);
    assert_solid(&round, whole - removed, 1e-4);
}

#[test]
fn a_plate_rounded_everywhere() {
    // Straight edges and rims in one call.
    let radius = 1.0;
    let plate = Body::plate_with_hole([6.0, 6.0, 2.0], radius).unwrap();
    let r = 0.3;
    let out = plate.fillet(&all_edges(&plate), r, TOL).unwrap();
    assert_eq!(kinds(&out, "torus"), 2);
    assert_eq!(kinds(&out, "sphere"), 8);
    assert_eq!(kinds(&out, "cylinder"), 13, "twelve fillets and the bore");

    let (cx, cy, cz) = (6.0 - 2.0 * r, 6.0 - 2.0 * r, 2.0 - 2.0 * r);
    let rounded_box = cx * cy * cz
        + 2.0 * r * (cx * cy + cy * cz + cz * cx)
        + PI * r * r * (cx + cy + cz)
        + 4.0 / 3.0 * PI * r * r * r;
    let bore = PI * radius * radius * 2.0;
    let mouth = 2.0 * PI * r * r * ((1.0 - PI / 4.0) * (radius + r) - r / 6.0);
    assert_solid(&out, rounded_box - bore - 2.0 * mouth, 1e-4);
}

#[test]
fn the_foot_of_a_boss_and_the_floor_of_a_blind_hole() {
    // Concave rims, both made by a boolean.
    let plate = Body::cuboid([6.0, 6.0, 2.0]);
    let radius = 1.0;

    // A boss standing on the plate: its foot is where the fillet adds a skirt.
    let post = Body::cylinder([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], radius, 3.0);
    let bossed = plate.boolean(&post, BooleanOp::Union, TOL).unwrap();
    let whole = 72.0 + PI * radius * radius * 2.0;
    assert_solid(&bossed, whole, 1e-3);
    let foot = closed_edges_where(&bossed, |p| (p[2] - 1.0).abs() < 1e-6);
    assert_eq!(foot.len(), 1);
    let r = 0.4;
    let out = bossed.fillet(&foot, r, TOL).unwrap();
    assert_eq!(kinds(&out, "torus"), 1);
    let added = 2.0 * PI * r * r * ((1.0 - PI / 4.0) * (radius + r) - r / 6.0);
    assert_solid(&out, whole + added, 1e-3);
    let out = bossed.chamfer(&foot, r, TOL).unwrap();
    assert_solid(&out, whole + PI * r * r * (radius + r / 3.0), 1e-3);

    // A blind hole: its floor is a disc inside the bore.
    let drill = Body::cylinder([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], radius, 3.0);
    let drilled = plate.boolean(&drill, BooleanOp::Difference, TOL).unwrap();
    let whole = 72.0 - PI * radius * radius;
    assert_solid(&drilled, whole, 1e-3);
    let floor = closed_edges_where(&drilled, |p| p[2].abs() < 1e-6);
    assert_eq!(floor.len(), 1);
    let out = drilled.fillet(&floor, r, TOL).unwrap();
    assert_eq!(kinds(&out, "torus"), 1);
    let added = 2.0 * PI * r * r * ((1.0 - PI / 4.0) * (radius - r) + r / 6.0);
    assert_solid(&out, whole + added, 1e-3);
    let out = drilled.chamfer(&floor, r, TOL).unwrap();
    assert_solid(&out, whole + PI * r * r * (radius - r / 3.0), 1e-3);
}

// -- after a boolean --------------------------------------------------------

#[test]
fn a_drilled_plate_can_still_have_its_edges_rounded() {
    let plate = Body::cuboid([6.0, 6.0, 2.0]);
    let drill = Body::cylinder([0.0, 0.0, -2.0], [0.0, 0.0, 1.0], 1.0, 4.0);
    let drilled = plate.boolean(&drill, BooleanOp::Difference, TOL).unwrap();
    let whole = 72.0 - PI * 2.0;
    assert_solid(&drilled, whole, 1e-3);

    // The four upright edges: their end faces are the ones the drill trimmed.
    let upright = edges_where(&drilled, |p| p[0].abs() > 2.9 && p[1].abs() > 2.9);
    let upright: Vec<usize> = upright
        .into_iter()
        .filter(|&e| {
            let edge = &drilled.edges()[e];
            let (a, b) = (
                drilled.vertices()[edge.vertices[0]],
                drilled.vertices()[*edge.vertices.last().unwrap()],
            );
            (a[2] - b[2]).abs() > 1.0
        })
        .collect();
    assert_eq!(upright.len(), 4);
    let r = 0.5;
    let out = drilled.fillet(&upright, r, TOL).unwrap();
    assert_solid(&out, whole - 4.0 * (1.0 - PI / 4.0) * r * r * 2.0, 1e-3);
}

// -- declines ---------------------------------------------------------------

#[test]
fn a_size_that_is_not_a_size_is_declined() {
    let cube = Body::cuboid([2.0, 2.0, 2.0]);
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            cube.fillet(&[0], bad, TOL).err(),
            Some(BlendDeclined::BadSize)
        );
        assert_eq!(
            cube.chamfer(&[0], bad, TOL).err(),
            Some(BlendDeclined::BadSize)
        );
    }
    assert_eq!(
        cube.fillet(&[0], 0.1, 0.0).err(),
        Some(BlendDeclined::BadSize)
    );
}

#[test]
fn an_edge_that_is_not_there_is_declined() {
    let cube = Body::cuboid([2.0, 2.0, 2.0]);
    assert_eq!(
        cube.fillet(&[12], 0.1, TOL).err(),
        Some(BlendDeclined::NoSuchEdge { edge: 12 })
    );
    let ball = Body::sphere([0.0; 3], 1.0);
    assert_eq!(
        ball.fillet(&[0], 0.1, TOL).err(),
        Some(BlendDeclined::NoSuchEdge { edge: 0 })
    );
}

#[test]
fn no_edges_is_no_change() {
    let cube = Body::cuboid([2.0, 2.0, 2.0]);
    let out = cube.fillet(&[], 0.1, TOL).unwrap();
    assert_eq!(out.faces().len(), 6);
    assert_eq!(out.edges().len(), 12);
}

#[test]
fn a_blend_too_large_for_its_edge_is_declined() {
    let cube = Body::cuboid([2.0, 2.0, 2.0]);
    let edge = edges_where(&cube, |p| p[0] > 0.0 && p[1] > 0.0);
    // Wider than the faces beside it.
    assert!(matches!(
        cube.chamfer(&edge, 2.5, TOL),
        Err(BlendDeclined::TooLarge { .. })
    ));
    // All twelve, each taking more than half of every face.
    assert!(matches!(
        cube.fillet(&all_edges(&cube), 1.2, TOL),
        Err(BlendDeclined::TooLarge { .. })
    ));

    let shaft = Body::cylinder([0.0; 3], [0.0, 0.0, 1.0], 1.0, 5.0);
    let top = closed_edges_where(&shaft, |p| p[2] > 1.0);
    assert!(matches!(
        shaft.fillet(&top, 1.5, TOL),
        Err(BlendDeclined::TooLarge { .. })
    ));

    // A countersink deeper than the plate, and one that fits in depth but
    // would run off the plate's sides — between its corners, not at them.
    let plate = Body::plate_with_hole([6.0, 6.0, 2.0], 1.0).unwrap();
    let mouth = closed_edges_where(&plate, |p| p[2] > 0.0);
    assert!(matches!(
        plate.chamfer(&mouth, 2.5, TOL),
        Err(BlendDeclined::TooLarge { .. })
    ));
    let thick = Body::plate_with_hole([6.0, 6.0, 6.0], 1.0).unwrap();
    let mouth = closed_edges_where(&thick, |p| p[2] > 0.0);
    assert!(matches!(
        thick.chamfer(&mouth, 2.5, TOL),
        Err(BlendDeclined::TooLarge { .. })
    ));
    assert!(thick.chamfer(&mouth, 1.5, TOL).is_ok());

    // An outer edge rounded so far that it reaches the hole.
    let edge = edges_where(&thick, |p| p[0] > 0.0 && p[2] > 0.0);
    assert_eq!(edge.len(), 1);
    assert!(matches!(
        thick.chamfer(&edge, 2.5, TOL),
        Err(BlendDeclined::TooLarge { .. })
    ));
    assert!(thick.chamfer(&edge, 1.5, TOL).is_ok());
}

/// A box with its four upright edges rounded, and those of its top: four
/// straight runs and four arcs, tangent all the way round.
fn rounded_slab(corner: f64) -> (Body, Vec<usize>) {
    let slab = Body::cuboid([4.0, 3.0, 2.0]);
    let upright = edges_where(&slab, |_| true)
        .into_iter()
        .filter(|&e| {
            let edge = &slab.edges()[e];
            let (a, b) = (
                slab.vertices()[edge.vertices[0]],
                slab.vertices()[edge.vertices[1]],
            );
            (a[2] - b[2]).abs() > 1.0
        })
        .collect::<Vec<_>>();
    assert_eq!(upright.len(), 4);
    let round = slab.fillet(&upright, corner, TOL).unwrap();
    assert_solid(
        &round,
        24.0 - 4.0 * (1.0 - PI / 4.0) * corner * corner * 2.0,
        1e-4,
    );
    let top = edges_where(&round, |p| p[2] > 0.0);
    assert_eq!(top.len(), 8, "four straight runs and four arcs");
    (round, top)
}

#[test]
fn a_rounded_boxs_top_is_filleted_all_the_way_round() {
    // Each top edge ends on a cylinder, so no edge can be rounded by itself;
    // together they are one ring, cylinders along the runs and a torus at
    // each corner.
    let (corner, r) = (0.5, 0.2);
    let (round, top) = rounded_slab(corner);
    let got = round.fillet(&top, r, TOL).unwrap();
    // What is taken off is the same section all the way: by Pappus, round the
    // corners' full turn as far as its centroid travels.
    let section = r * r * (1.0 - PI / 4.0);
    let from_wall = (r.powi(3) / 2.0 - PI * r * r / 4.0 * (r - 4.0 * r / (3.0 * PI))) / section;
    let runs = 2.0 * (4.0 - 2.0 * corner) + 2.0 * (3.0 - 2.0 * corner);
    let off = section * (runs + 2.0 * PI * (corner - from_wall));
    let before = 24.0 - 4.0 * (1.0 - PI / 4.0) * corner * corner * 2.0;
    assert_solid(&got, before - off, 5e-5);
    let kinds = |k: &str| {
        got.faces()
            .iter()
            .filter(|f| got.surfaces()[f.surface].kind() == k)
            .count()
    };
    assert_eq!(
        (kinds("torus"), kinds("cylinder"), kinds("plane")),
        (4, 8, 6)
    );
}

#[test]
fn joints_between_tangent_faces_are_refined() {
    // Where a corner's torus runs out into the top, a chord of the joint lies
    // in the top and leaves the torus only at second order: nothing of its
    // distance from either says how far it is off the arc.
    let (round, top) = rounded_slab(0.5);
    let mut got = round.fillet(&top, 0.2, TOL).unwrap();
    let tolerance = 1e-5;
    got.refine_edges(tolerance);
    let mut arcs = 0;
    for e in got.edges() {
        let kinds = [e.surfaces.0, e.surfaces.1].map(|s| got.surfaces()[s].kind());
        let points: Vec<_> = e.vertices.iter().map(|&v| got.vertices()[v]).collect();
        let on_top = points.iter().all(|p| (p[2] - 1.0).abs() < 1e-9);
        if !(kinds.contains(&"torus") && kinds.contains(&"plane") && on_top) {
            continue;
        }
        arcs += 1;
        // Round the corner at (±1.5, ±1.0), 0.3 out.
        let (cx, cy) = (1.5f64.copysign(points[0][0]), 1.0f64.copysign(points[0][1]));
        for w in points.windows(2) {
            let m = [
                (w[0][0] + w[1][0]) / 2.0 - cx,
                (w[0][1] + w[1][1]) / 2.0 - cy,
            ];
            let sag = 0.3 - m[0].hypot(m[1]);
            assert!(sag <= tolerance * 1.01, "a chord {sag} off its arc");
        }
    }
    assert_eq!(arcs, 4);
}

#[test]
fn any_edge_of_a_smooth_ring_blends_all_of_it() {
    // As a blend that follows tangent edges does: OpenCASCADE makes the same
    // eighteen faces of one run, of all but one edge, and of them all.
    let (round, top) = rounded_slab(0.5);
    let straight = top
        .iter()
        .copied()
        .find(|&e| round.edges()[e].vertices.len() == 2)
        .expect("a straight run");
    let whole = volume(&round.fillet(&top, 0.2, TOL).unwrap(), TOL);
    for part in [&[straight][..], &top[1..]] {
        let got = round.fillet(part, 0.2, TOL).unwrap();
        assert_eq!(got.faces().len(), 18);
        assert_solid(&got, whole, 1e-9);
    }
}

#[test]
fn an_edge_ending_on_a_curved_face_is_declined() {
    // A corner bitten out of a box: the top edges beside it end square on the
    // bite's round wall, which the blend would have to be trimmed by.
    let slab = Body::cuboid([4.0, 3.0, 2.0]);
    let bite = Body::cylinder([2.0, 1.5, -2.0], [0.0, 0.0, 1.0], 0.8, 4.0);
    let bitten = slab.boolean(&bite, BooleanOp::Difference, TOL).unwrap();
    let beside = edges_where(&bitten, |p| p[2] > 0.99 && (p[0] - 2.0).abs() < 1e-6);
    assert_eq!(beside.len(), 1, "what is left of the top's x = 2 side");
    let got = bitten.fillet(&beside, 0.2, TOL);
    assert!(
        matches!(
            got,
            Err(BlendDeclined::UnsupportedEdge { .. } | BlendDeclined::UnsupportedVertex { .. })
        ),
        "{got:?}"
    );

    // And a ring wider than its corners would turn inside out there.
    let (round, top) = rounded_slab(0.5);
    let wide = round.fillet(&top, 0.6, TOL);
    assert!(
        matches!(wide, Err(BlendDeclined::TooLarge { .. })),
        "{wide:?}"
    );
}

#[test]
fn a_blend_can_be_blended_again_where_it_left_plane_corners() {
    // A chamfer leaves nothing but planes, so its own edges are fair game.
    let cube = Body::cuboid([2.0, 2.0, 2.0]);
    let edge = edges_where(&cube, |p| p[0] > 0.0 && p[1] > 0.0);
    let d = 0.5;
    let once = cube.chamfer(&edge, d, TOL).unwrap();
    // The two rails of the bevel.
    let mut rails = edges_where(&once, |p| p[0] > 0.99 && (p[1] - 0.5).abs() < 1e-9);
    rails.extend(edges_where(&once, |p| {
        p[1] > 0.99 && (p[0] - 0.5).abs() < 1e-9
    }));
    assert_eq!(rails.len(), 2);
    let twice = once.fillet(&rails, 0.2, TOL).unwrap();
    assert_eq!(kinds(&twice, "cylinder"), 2);
    // Each rail turns through 45°: the ball removes r²·(tan(φ/2) − φ/2) of
    // section there.
    let phi = PI / 4.0;
    let section = 0.2 * 0.2 * ((phi / 2.0).tan() - phi / 2.0);
    assert_solid(&twice, 8.0 - 0.5 * d * d * 2.0 - 2.0 * section * 2.0, 1e-4);
}

#[test]
fn a_finer_tolerance_is_a_better_answer() {
    let cube = Body::cuboid([2.0, 2.0, 2.0]);
    let r = 0.4;
    let c = 2.0 - 2.0 * r;
    let exact = c * c * c + 6.0 * c * c * r + 3.0 * PI * r * r * c + 4.0 / 3.0 * PI * r * r * r;
    let mut last = f64::MAX;
    for tol in [1e-2, 1e-3, 1e-4] {
        let out = cube.fillet(&all_edges(&cube), r, tol).unwrap();
        let err = (volume(&out, tol) - exact).abs();
        assert!(err < last, "tol {tol}: error {err} after {last}");
        last = err;
    }
}
