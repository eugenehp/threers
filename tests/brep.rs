//! Stage 1 acceptance suite for the `brep` feature.
//!
//! The criteria from `docs/brep-nurbs-plan.md` § Stage 1, as executable checks
//! against the public API. Unit tests inside `src/brep/` cover mechanism; this
//! covers the contract:
//!
//! * every built-in primitive's tag is *true* — its triangles really do lie on
//!   the surface they claim
//! * analytic normals beat averaged ones, by a margin that grows as the mesh
//!   coarsens
//! * a coarse sphere re-tessellates to an arbitrary tolerance
//! * transforms compose: transform-then-sample equals sample-then-transform
//! * the safety property — provenance never alters a mesh, and dropping it
//!   leaves every consumer behaving as it did before the feature existed

#![cfg(feature = "brep")]

use threers::brep::{
    exact_normals, exact_uvs, retessellate, Body, Surface, SurfaceTable, TrimLoop,
};
use threers::core::BufferGeometry;
use threers::geometries::{
    BoxGeometry, CircleGeometry, ConeGeometry, CylinderGeometry, PlaneGeometry, RingGeometry,
    SphereGeometry, TorusGeometry,
};
use threers::math::{Matrix4, Quaternion, Vector3};

const TAU: f32 = std::f32::consts::PI * 2.0;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Vertices actually used by a triangle. The per-vertex writers work through
/// triangles, so an unreferenced vertex is never assigned and keeps whatever the
/// generator gave it.
fn referenced_vertices(g: &BufferGeometry) -> Vec<usize> {
    match &g.index {
        Some(idx) => {
            let mut v: Vec<usize> = idx.iter().map(|&i| i as usize).collect();
            v.sort_unstable();
            v.dedup();
            v
        }
        None => (0..g.get_attribute("position").map_or(0, |a| a.count())).collect(),
    }
}

fn positions(g: &BufferGeometry) -> Vec<f32> {
    g.get_attribute("position").unwrap().array.clone()
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Worst angular error between a mesh's stored normals and the true normals of
/// the surface each vertex sits on, as `1 − cos θ`.
fn normal_error(g: &BufferGeometry) -> f64 {
    let table = g.surface_table().expect("tagged");
    let nor = &g.get_attribute("normal").unwrap().array;
    let mut worst = 0.0f64;

    for tri in 0..table.triangle_count() {
        let (Some(surface), Some(verts), Some(idx)) = (
            table.surface_of(tri),
            threers::brep::triangle_vertices(g, tri),
            threers::brep::triangle_indices(g, tri),
        ) else {
            continue;
        };
        for (k, &vi) in idx.iter().enumerate() {
            let Some((u, v)) = surface.invert(verts[k]) else {
                continue;
            };
            let Some(truth) = surface.normal(u, v) else {
                continue;
            };
            let o = vi * 3;
            let stored = [nor[o] as f64, nor[o + 1] as f64, nor[o + 2] as f64];
            let len = (stored[0].powi(2) + stored[1].powi(2) + stored[2].powi(2)).sqrt();
            if len < 1e-9 {
                continue;
            }
            let dot = (stored[0] * truth[0] + stored[1] * truth[1] + stored[2] * truth[2]) / len;
            worst = worst.max(1.0 - dot.abs());
        }
    }
    worst
}

/// Worst distance from a triangle-edge midpoint to the surface — the mesh's
/// real error, which its vertices (exactly on the surface) do not show.
fn chord_error(g: &BufferGeometry) -> f64 {
    let table = g.surface_table().expect("tagged");
    let mut worst = 0.0f64;
    for tri in 0..table.triangle_count() {
        let (Some(surface), Some(v)) = (
            table.surface_of(tri),
            threers::brep::triangle_vertices(g, tri),
        ) else {
            continue;
        };
        for k in 0..3 {
            let a = v[k];
            let b = v[(k + 1) % 3];
            let mid = [
                0.5 * (a[0] + b[0]),
                0.5 * (a[1] + b[1]),
                0.5 * (a[2] + b[2]),
            ];
            worst = worst.max(surface.distance(mid));
        }
    }
    worst
}

/// Every built-in primitive, with a characteristic scale for tolerancing.
fn all_primitives() -> Vec<(&'static str, BufferGeometry, f64)> {
    vec![
        ("box", BoxGeometry::new(2.0, 3.0, 4.0), 4.0),
        ("sphere", SphereGeometry::new(2.0, 32, 16), 2.0),
        ("sphere/coarse", SphereGeometry::new(1.0, 6, 4), 1.0),
        (
            "cylinder",
            CylinderGeometry::new(1.5, 1.5, 4.0, 24, 2, false, 0.0, TAU),
            4.0,
        ),
        (
            "cylinder/open",
            CylinderGeometry::new(1.0, 1.0, 2.0, 16, 1, true, 0.0, TAU),
            2.0,
        ),
        (
            "cylinder/arc",
            CylinderGeometry::new(1.0, 1.0, 2.0, 16, 1, false, 0.4, 2.1),
            2.0,
        ),
        (
            "truncated cone",
            CylinderGeometry::new(1.0, 3.0, 4.0, 24, 1, false, 0.0, TAU),
            4.0,
        ),
        (
            "cone",
            ConeGeometry::new(2.0, 5.0, 24, 1, false, 0.0, TAU),
            5.0,
        ),
        ("torus", TorusGeometry::new(4.0, 1.0, 16, 32, TAU), 5.0),
        ("plane", PlaneGeometry::with_segments(3.0, 2.0, 4, 3), 3.0),
        ("circle", CircleGeometry::new(2.0, 24, 0.0, TAU), 2.0),
        ("ring", RingGeometry::new(1.0, 2.0, 24, 1, 0.0, TAU), 2.0),
    ]
}

// ---------------------------------------------------------------------------
// provenance is truthful
// ---------------------------------------------------------------------------

#[test]
fn every_primitive_is_tagged_and_its_tag_is_true() {
    // The central claim of Stage 1. A tag says "these triangles came from that
    // surface"; this measures the distance from every tagged vertex to the
    // surface it names. Anything else in this file is downstream of it.
    for (name, g, scale) in all_primitives() {
        let table = g
            .surface_table()
            .unwrap_or_else(|| panic!("{name}: no provenance attached"));
        assert_eq!(
            table.triangle_count(),
            threers::brep::triangle_count(&g),
            "{name}: table does not describe the geometry"
        );
        let dev = table.max_deviation(&g);
        // f32 positions, so the floor is f32 epsilon times the model scale.
        assert!(
            dev < 1e-5 * scale,
            "{name}: a vertex is {dev} from the surface it claims (scale {scale})"
        );
    }
}

#[test]
fn every_tagged_triangle_belongs_to_exactly_one_surface() {
    for (name, g, _) in all_primitives() {
        let table = g.surface_table().unwrap();
        let covered: usize = table.groups().iter().map(|(_, t)| t.len()).sum();
        assert_eq!(
            covered,
            table.triangle_count(),
            "{name}: groups do not partition the triangles"
        );
        for tri in 0..table.triangle_count() {
            assert!(
                table.surface_of(tri).is_some(),
                "{name}: triangle {tri} untagged"
            );
        }
    }
}

#[test]
fn surface_kinds_are_the_ones_a_reader_would_expect() {
    let cases: Vec<(&str, BufferGeometry, Vec<&str>)> = vec![
        ("box", BoxGeometry::new(1.0, 1.0, 1.0), vec!["plane"; 6]),
        ("sphere", SphereGeometry::new(1.0, 8, 6), vec!["sphere"]),
        (
            "capped cylinder",
            CylinderGeometry::new(1.0, 1.0, 2.0, 12, 1, false, 0.0, TAU),
            vec!["cylinder", "plane", "plane"],
        ),
        (
            "cone",
            ConeGeometry::new(1.0, 2.0, 12, 1, false, 0.0, TAU),
            // radius_top = 0 → no top cap.
            vec!["cone", "plane"],
        ),
        (
            "torus",
            TorusGeometry::new(3.0, 1.0, 8, 12, TAU),
            vec!["torus"],
        ),
    ];
    for (name, g, want) in cases {
        let got: Vec<&str> = g
            .surface_table()
            .unwrap()
            .surfaces()
            .iter()
            .map(|s| s.kind())
            .collect();
        assert_eq!(got, want, "{name}");
    }
}

// ---------------------------------------------------------------------------
// exact normals
// ---------------------------------------------------------------------------

#[test]
fn analytic_normals_beat_averaged_ones_and_the_gap_grows_with_coarseness() {
    // The error in averaged normals is bounded by the tessellation, so it should
    // shrink as the mesh refines while the analytic error stays at zero.
    let mut previous_gap = 0.0f64;
    for &(w, h) in &[(32usize, 16usize), (12, 8), (6, 4)] {
        let base = SphereGeometry::new(1.0, w, h);

        let mut averaged = base.clone();
        threers::compute_vertex_normals(&mut averaged);
        // `compute_vertex_normals` writes an attribute, which drops the table.
        let table = base.surface_table().unwrap().clone();
        averaged.set_surfaces(table);
        let e_avg = normal_error(&averaged);

        let mut exact = base.clone();
        exact_normals(&mut exact).expect("provenance is attached");
        let e_exact = normal_error(&exact);

        assert!(
            e_exact < 1e-9,
            "{w}x{h}: analytic normals are off by {e_exact}"
        );
        assert!(
            e_avg > e_exact,
            "{w}x{h}: averaging ({e_avg}) was not worse than analytic ({e_exact})"
        );
        assert!(
            e_avg > previous_gap,
            "{w}x{h}: averaging error {e_avg} did not grow as the mesh coarsened"
        );
        previous_gap = e_avg;
    }
}

#[test]
fn analytic_normals_work_on_every_curved_primitive() {
    for (name, mut g, _) in all_primitives() {
        if g.surface_table().is_none() {
            continue;
        }
        exact_normals(&mut g).unwrap_or_else(|| panic!("{name}: normals declined"));
        let err = normal_error(&g);
        assert!(err < 1e-9, "{name}: analytic normal error {err}");

        // And every normal a triangle uses is unit.
        let nor = &g.get_attribute("normal").unwrap().array;
        for vi in referenced_vertices(&g) {
            let c = &nor[vi * 3..vi * 3 + 3];
            let l = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
            assert!((l - 1.0).abs() < 1e-4, "{name}: normal length {l}");
        }
    }
}

#[test]
fn uvs_from_provenance_stay_inside_the_unit_square() {
    for (name, mut g, _) in all_primitives() {
        let report = exact_uvs(&mut g).unwrap_or_else(|| panic!("{name}: uvs declined"));
        assert!(report.vertices > 0, "{name}: no vertices assigned");
        // The guarantee covers *assigned* vertices. Ones no triangle references
        // are left as found — a UV sphere's pole columns carry a deliberate
        // half-texel offset that sits outside [0, 1], and overwriting it with
        // zero to satisfy a blanket assertion would be destroying data to make
        // a test pass.
        let uv = &g.get_attribute("uv").unwrap().array;
        for vi in referenced_vertices(&g) {
            let (u, v) = (uv[vi * 2], uv[vi * 2 + 1]);
            assert!(u.is_finite() && v.is_finite(), "{name}: non-finite uv");
            assert!(
                (-1e-5..=1.0 + 1e-5).contains(&u) && (-1e-5..=1.0 + 1e-5).contains(&v),
                "{name}: uv ({u}, {v}) outside the unit square"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// re-tessellation
// ---------------------------------------------------------------------------

#[test]
fn a_coarse_sphere_retessellates_to_an_arbitrary_tolerance() {
    // The plan's acceptance criterion, and the reason to carry surfaces at all:
    // resolution stops being a decision made once, at build time.
    let coarse = SphereGeometry::new(1.0, 16, 8);
    let before = chord_error(&coarse);
    assert!(before > 1e-2, "the coarse mesh should be coarse: {before}");

    for &tol in &[1e-2, 1e-3, 1e-4] {
        let (fine, report) = retessellate(&coarse, tol).expect("provenance is attached");
        assert_eq!(report.patches, 1);
        assert!(
            report.is_closed(),
            "{tol}: {} open edges",
            report.boundary_edges
        );

        let after = chord_error(&fine);
        assert!(
            after <= tol,
            "{tol}: chord error {after} exceeds the tolerance"
        );
    }
}

#[test]
fn retessellation_is_idempotent_in_shape() {
    // Re-meshing twice at the same tolerance must not drift: the second pass
    // reads the same surfaces, not the first pass's triangles.
    let g = TorusGeometry::new(3.0, 1.0, 8, 12, TAU);
    let (once, r1) = retessellate(&g, 1e-3).unwrap();
    let (twice, r2) = retessellate(&once, 1e-3).unwrap();
    assert_eq!(r1.triangles, r2.triangles);
    assert_eq!(positions(&once).len(), positions(&twice).len());
    for (a, b) in positions(&once).iter().zip(positions(&twice).iter()) {
        assert!((a - b).abs() < 1e-5, "re-meshing drifted: {a} vs {b}");
    }
}

#[test]
fn retessellation_reports_what_it_produced_rather_than_implying_more() {
    // Multi-surface input may leave patch boundaries unmatched. That is a real
    // limitation until Stage 3 trims, and the report is how a caller finds out
    // without discovering it in a slicer.
    let cyl = CylinderGeometry::new(1.0, 1.0, 3.0, 24, 1, false, 0.0, TAU);
    let (_, report) = retessellate(&cyl, 1e-3).unwrap();
    assert_eq!(report.patches, 3, "side plus two caps");
    assert!(report.triangles > 0 && report.vertices > 0);
    // Whether it closed or not, the report must agree with itself.
    assert_eq!(report.is_closed(), report.boundary_edges == 0);
}

#[test]
fn a_coarser_tolerance_costs_fewer_triangles() {
    let g = SphereGeometry::new(1.0, 16, 8);
    let (_, fine) = retessellate(&g, 1e-4).unwrap();
    let (_, coarse) = retessellate(&g, 1e-2).unwrap();
    assert!(
        coarse.triangles < fine.triangles,
        "coarse {} vs fine {}",
        coarse.triangles,
        fine.triangles
    );
}

// ---------------------------------------------------------------------------
// transforms
// ---------------------------------------------------------------------------

#[test]
fn transforming_the_surface_equals_transforming_the_mesh() {
    // The round-trip that makes provenance survive a scene graph: a table
    // mapped through a matrix must describe the mesh mapped through the same
    // matrix. Anything else and a transformed primitive's tag becomes a lie.
    let transforms = [
        (
            "translate",
            Matrix4::translation(Vector3::new(1.0, -2.0, 3.0)),
        ),
        (
            "rotate",
            Matrix4::from_quaternion(Quaternion::from_axis_angle(
                Vector3::new(0.3, 1.0, -0.2).normalize(),
                0.9,
            )),
        ),
        ("uniform scale", Matrix4::scale(Vector3::new(2.5, 2.5, 2.5))),
        (
            "rigid + scale",
            Matrix4::translation(Vector3::new(4.0, 0.0, 1.0))
                .multiply(&Matrix4::scale(Vector3::new(0.5, 0.5, 0.5))),
        ),
    ];

    for (tname, m) in transforms {
        for (gname, g, scale) in all_primitives() {
            let table = g.surface_table().unwrap();
            let Some(moved_table) = table.transform(&m) else {
                panic!("{gname} under {tname}: a similarity must be representable");
            };

            // Move the mesh by the same matrix and check the moved table
            // describes it.
            let mut moved = g.clone();
            let pos: Vec<f32> = positions(&g)
                .chunks_exact(3)
                .flat_map(|c| {
                    let p = Vector3::new(c[0], c[1], c[2]).apply_matrix4(&m);
                    [p.x, p.y, p.z]
                })
                .collect();
            moved.set_attribute("position", threers::core::BufferAttribute::new(pos, 3));
            moved.set_surfaces(moved_table);

            let dev = moved.surface_table().unwrap().max_deviation(&moved);
            assert!(
                dev < 1e-4 * scale.max(1.0),
                "{gname} under {tname}: deviation {dev} after transform"
            );
        }
    }
}

#[test]
fn a_non_uniform_scale_drops_the_table_instead_of_lying() {
    // A squashed sphere is an ellipsoid. There is no `Sphere` radius that
    // describes it, so the honest answer is no provenance at all — which every
    // consumer already handles.
    let squash = Matrix4::scale(Vector3::new(1.0, 2.0, 1.0));
    let g = SphereGeometry::new(1.0, 16, 8);
    assert!(g.surface_table().unwrap().transform(&squash).is_none());

    // A box of planes survives it, because planes do.
    let b = BoxGeometry::new(1.0, 1.0, 1.0);
    assert!(b.surface_table().unwrap().transform(&squash).is_some());
}

// ---------------------------------------------------------------------------
// the safety property
// ---------------------------------------------------------------------------

#[test]
fn tagging_never_alters_the_mesh() {
    // Provenance is a sidecar. Every buffer a primitive produces must be
    // identical, bit for bit, to what it produced before the feature existed —
    // which is also why the pre-existing three.js parity tests still pass with
    // `brep` enabled.
    for (name, g, _) in all_primitives() {
        let before = (
            positions(&g),
            g.get_attribute("normal").map(|a| a.array.clone()),
            g.get_attribute("uv").map(|a| a.array.clone()),
            g.index.clone(),
        );

        let mut stripped = g.clone();
        stripped.surfaces = None;

        assert_eq!(positions(&stripped), before.0, "{name}: positions moved");
        assert_eq!(
            stripped.get_attribute("normal").map(|a| a.array.clone()),
            before.1,
            "{name}: normals moved"
        );
        assert_eq!(
            stripped.get_attribute("uv").map(|a| a.array.clone()),
            before.2,
            "{name}: uvs moved"
        );
        assert_eq!(stripped.index, before.3, "{name}: index moved");
    }
}

#[test]
fn every_consumer_declines_without_provenance() {
    // Dropping the table must leave a build behaving exactly as it did before
    // the feature: no consumer may guess.
    let mut g = SphereGeometry::new(1.0, 8, 6);
    g.surfaces = None;

    assert!(exact_normals(&mut g).is_none());
    assert!(exact_uvs(&mut g).is_none());
    assert!(retessellate(&g, 1e-3).is_none());
    assert!(g.surface_table().is_none());
}

#[test]
fn editing_the_geometry_invalidates_the_table() {
    // A sidecar describing geometry that has since changed is worse than no
    // sidecar, because consumers trust it.
    let mut g = SphereGeometry::new(1.0, 8, 6);
    assert!(g.surface_table().is_some());

    let pos = positions(&g);
    g.set_attribute("position", threers::core::BufferAttribute::new(pos, 3));
    assert!(
        g.surface_table().is_none(),
        "writing positions must clear the provenance"
    );
}

#[test]
fn a_mismatched_table_is_refused_rather_than_stored() {
    let mut g = SphereGeometry::new(1.0, 8, 6);
    g.surfaces = None;
    // Claims seven triangles; the sphere has hundreds.
    g.set_surfaces(SurfaceTable::uniform(Surface::sphere([0.0; 3], 1.0), 7));
    assert!(
        g.surface_table().is_none(),
        "a table that does not describe this geometry must not be stored"
    );
}

#[test]
fn provenance_survives_a_clone() {
    let g = TorusGeometry::new(3.0, 1.0, 8, 12, TAU);
    let c = g.clone();
    assert_eq!(
        g.surface_table().map(|t| t.triangle_count()),
        c.surface_table().map(|t| t.triangle_count())
    );
    assert!(c.surface_table().unwrap().max_deviation(&c) < 1e-4);
}

// ---------------------------------------------------------------------------
// what a surface knows that a mesh cannot
// ---------------------------------------------------------------------------

#[test]
fn a_truncated_cones_apex_is_recovered_though_no_vertex_is_there() {
    // The sharpest illustration of why provenance is not derivable from the
    // mesh: the apex of a truncated cone is not a vertex, not on any face, and
    // not inside the bounding box. Only the construction knew where it was.
    let g = CylinderGeometry::new(1.0, 3.0, 4.0, 24, 1, false, 0.0, TAU);
    let table = g.surface_table().unwrap();

    match &table.surfaces()[0] {
        Surface::Cone { apex, .. } => {
            assert!(dist(*apex, [0.0, 4.0, 0.0]) < 1e-9, "apex at {apex:?}");
            // Confirm it really is outside the mesh.
            let top_y = positions(&g)
                .chunks_exact(3)
                .map(|c| c[1] as f64)
                .fold(f64::MIN, f64::max);
            assert!(
                apex[1] > top_y + 1.0,
                "apex {} vs mesh top {top_y}",
                apex[1]
            );
        }
        other => panic!("expected a cone, got {}", other.kind()),
    }
}

#[test]
fn a_partial_arc_still_names_the_whole_surface() {
    // A face is a surface *plus a trim*. Without trims (Stage 3) the surface is
    // the unbounded one and the triangles carry "which part" — so a 120° wedge
    // of a cylinder is tagged with the same infinite cylinder as a full one.
    let full = CylinderGeometry::new(1.0, 1.0, 2.0, 24, 1, true, 0.0, TAU);
    let wedge = CylinderGeometry::new(1.0, 1.0, 2.0, 8, 1, true, 0.0, TAU / 3.0);

    let a = &full.surface_table().unwrap().surfaces()[0];
    let b = &wedge.surface_table().unwrap().surfaces()[0];
    assert_eq!(a, b, "the same cylinder, sampled over different ranges");

    assert!(wedge.surface_table().unwrap().max_deviation(&wedge) < 1e-5);
}

/// Volume of a body's tessellation, by the divergence theorem.
fn plate_volume(body: &Body, tolerance: f64) -> (f64, usize) {
    let mut b = body.clone();
    b.refine_edges(tolerance);
    let (mesh, report) = b.tessellate(tolerance);
    let pos = &mesh.get_attribute("position").unwrap().array;
    let idx = mesh.index.as_ref().unwrap();
    let v = |i: u32| -> [f64; 3] {
        let o = i as usize * 3;
        [pos[o] as f64, pos[o + 1] as f64, pos[o + 2] as f64]
    };
    let volume = idx
        .chunks_exact(3)
        .map(|t| {
            let (a, b, c) = (v(t[0]), v(t[1]), v(t[2]));
            (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]))
                / 6.0
        })
        .sum::<f64>()
        .abs();
    (volume, report.boundary_edges)
}

#[test]
fn a_face_with_several_holes_triangulates_watertight() {
    // Triangulating a face with holes means bridging each one into the outer
    // ring. With a single hole that is trivial — the outer boundary is the only
    // thing there and is always reachable. Every failure lives past that:
    //
    // * a bridge that runs *through* an already-merged hole, leaving a
    //   self-intersecting ring the ear clip fills only part of;
    // * a bridge that arrives *tangent* to one, since the rightmost point of a
    //   circle has a vertical tangent — the ear there is exactly flat, the clip
    //   drops it, and with it a rim vertex the bore wall still has.
    //
    // * a chain of bridges hole-to-hole-to-hole, which leaves the ring only
    //   *weakly* simple — the channels meet at shared vertices, and an ear clip
    //   has no guarantee there. It ends up clipping a zero-area sliver whose
    //   edges pair with nothing.
    //
    // All three need two or more holes to happen at all, which is why they
    // survived a suite in which every holed face had exactly one.
    let arrangements: Vec<Vec<([f64; 2], f64)>> = vec![
        vec![([0.0, 0.0], 2.5)],
        vec![([-8.0, 0.0], 2.5), ([8.0, 0.0], 2.5)],
        vec![([-6.0, -4.0], 2.0), ([6.0, 4.0], 2.0)],
        // Collinear and evenly spaced, at two different spacings.
        vec![([-8.0, 0.0], 2.5), ([0.0, 0.0], 2.5), ([8.0, 0.0], 2.5)],
        vec![([-14.0, 0.0], 2.5), ([0.0, 0.0], 2.5), ([14.0, 0.0], 2.5)],
        vec![
            ([-12.0, 0.0], 2.0),
            ([-4.0, 0.0], 2.0),
            ([4.0, 0.0], 2.0),
            ([12.0, 0.0], 2.0),
        ],
        vec![([-10.0, 4.0], 1.5), ([0.0, -4.0], 1.5), ([10.0, 4.0], 1.5)],
        // Collinear on a diagonal, both slopes — the arrangement that makes
        // consecutive bridges parallel.
        vec![([-12.0, -6.0], 2.0), ([0.0, 0.0], 2.0), ([12.0, 6.0], 2.0)],
        vec![([-12.0, 6.0], 2.0), ([0.0, 0.0], 2.0), ([12.0, -6.0], 2.0)],
        vec![
            ([-15.0, -9.0], 1.0),
            ([-5.0, -3.0], 1.0),
            ([5.0, 3.0], 1.0),
            ([15.0, 9.0], 1.0),
        ],
        // Collinear along the short axis, and five in a row along the long one.
        vec![([0.0, 0.0], 1.0), ([0.0, 8.0], 1.0), ([0.0, -8.0], 1.0)],
        vec![
            ([-16.0, 0.0], 1.0),
            ([-8.0, 0.0], 1.0),
            ([0.0, 0.0], 1.0),
            ([8.0, 0.0], 1.0),
            ([16.0, 0.0], 1.0),
        ],
        vec![
            ([-12.0, -6.0], 2.0),
            ([12.0, -6.0], 2.0),
            ([-12.0, 6.0], 2.0),
            ([12.0, 6.0], 2.0),
            ([0.0, 0.0], 3.0),
        ],
    ];

    for holes in arrangements {
        let body = Body::plate_with_holes([40.0, 24.0, 4.0], &holes).unwrap();
        let (volume, open) = plate_volume(&body, 1e-3);
        let expected = 40.0 * 24.0 * 4.0
            - holes
                .iter()
                .map(|(_, r)| std::f64::consts::PI * r * r * 4.0)
                .sum::<f64>();
        let centres: Vec<[f64; 2]> = holes.iter().map(|h| h.0).collect();
        assert_eq!(open, 0, "{centres:?} left {open} open edges");
        assert!(
            (volume - expected).abs() / expected < 0.01,
            "{centres:?}: volume {volume}, expected {expected}"
        );
    }
}

#[test]
fn holes_that_do_not_fit_are_refused() {
    // Overlapping bores, and one running off the edge, are not solids.
    assert!(Body::plate_with_holes([10.0, 10.0, 2.0], &[]).is_none());
    assert!(Body::plate_with_holes([10.0, 10.0, 2.0], &[([0.0, 0.0], 6.0)]).is_none());
    assert!(Body::plate_with_holes([10.0, 10.0, 2.0], &[([4.5, 0.0], 1.0)]).is_none());
    assert!(
        Body::plate_with_holes([20.0, 10.0, 2.0], &[([-1.0, 0.0], 2.0), ([1.0, 0.0], 2.0)])
            .is_none(),
        "overlapping bores"
    );
    assert!(
        Body::plate_with_holes([20.0, 10.0, 2.0], &[([-4.0, 0.0], 2.0), ([4.0, 0.0], 2.0)])
            .is_some()
    );
}

#[test]
fn a_traced_polyline_and_the_curve_through_it_agree_at_every_sample() {
    // The parameter has to keep meaning what it meant. For a traced polyline `t`
    // indexes the samples, and everything downstream — clipping, seam splitting,
    // the arrangement — is written against that. A curve through the same points
    // is only useful if it answers the same question the same way, and then
    // answers *between* the samples too, which the polyline cannot.
    use std::f64::consts::TAU;
    let n = 32;
    let ring: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let t = TAU * i as f64 / n as f64;
            [3.0 * t.cos(), 3.0 * t.sin(), 1.5]
        })
        .collect();

    use threers::brep::Curve3d;
    let polyline = Curve3d::Sampled {
        points: ring.clone(),
        closed: true,
    };
    let curve = Curve3d::spline_through(&ring, true).expect("a ring reads as a curve");
    assert_eq!(curve.kind(), "spline");

    // At every sample, the same point.
    for i in 0..n {
        let (a, b) = (polyline.point(i as f64), curve.point(i as f64));
        let d = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
        assert!(d < 1e-9, "sample {i}: polyline and curve are {d} apart");
    }

    // Between them, the curve is on the circle and the polyline is not. The
    // chord of a 32-gon at this radius sags 1.44e-2, and that sag is what a
    // boundary made of chords hands to everything downstream; the curve through
    // the same points is out by 1.04e-4, a hundred and forty times less. It is
    // an interpolation, not the circle, so it is not exact — but it is wrong by
    // less than the tolerances this kernel is asked to hold, where the chord is
    // wrong by more.
    let (mut chord, mut spline) = (0.0f64, 0.0f64);
    for k in 0..400 {
        let t = n as f64 * k as f64 / 400.0;
        chord = chord.max((polyline.point(t)[0].hypot(polyline.point(t)[1]) - 3.0).abs());
        spline = spline.max((curve.point(t)[0].hypot(curve.point(t)[1]) - 3.0).abs());
    }
    assert!(
        chord > 1e-2,
        "the polyline should sag; it is out by {chord}"
    );
    assert!(spline < 1e-3, "the curve should not; it is out by {spline}");

    // And it still projects and moves like a curve.
    let near = curve.project([5.0, 0.0, 1.5]);
    assert!(
        (near[0].hypot(near[1]) - 3.0).abs() < 1e-6,
        "projected off the curve: {near:?}"
    );
    let moved = curve.translated([0.0, 0.0, 2.0]);
    assert!(
        (moved.point(0.0)[2] - 3.5).abs() < 1e-9,
        "translation did not carry"
    );
}

#[test]
fn a_ring_that_crosses_itself_becomes_a_graph_that_does_not() {
    // The fill's precondition today is a simple polygon, and 17 of 49 trim loops
    // in a chaining corpus are not one: a traced curve's closure keeps a sample
    // that has already gone round, so the ring laps itself by a fraction of a
    // segment. Five repairs at the ring have failed, every one of them because a
    // ring is a curve *projected* — editing one projection breaks its agreement
    // with the face across the boundary.
    //
    // An arrangement has no such precondition. The crossing becomes a vertex.
    use threers::brep::planar::arrange;

    // A square whose last corner overshoots the first, exactly as a closure
    // does: the closing segment runs back across the opening one.
    let lapped = vec![
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.05, -0.05],
    ];
    let (pts, edges) = arrange(std::slice::from_ref(&lapped), 1e-9);

    // The lap is now a point of the graph, not a crossing in a polygon.
    assert!(
        pts.len() > lapped.len(),
        "the crossing should have become a vertex: {} points from {}",
        pts.len(),
        lapped.len()
    );
    assert!(
        edges.len() > lapped.len(),
        "and the two segments it splits should have become four"
    );

    // Nothing crosses any more.
    let side = |p: [f64; 2], q: [f64; 2], r: [f64; 2]| {
        (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    };
    for i in 0..edges.len() {
        for j in i + 1..edges.len() {
            // Sharing an endpoint is not crossing: the orientation is zero
            // there, and a zero reads as a sign change.
            let (e, f) = (edges[i], edges[j]);
            if e[0] == f[0] || e[0] == f[1] || e[1] == f[0] || e[1] == f[1] {
                continue;
            }
            let (a, b) = (pts[edges[i][0]], pts[edges[i][1]]);
            let (c, d) = (pts[edges[j][0]], pts[edges[j][1]]);
            let (d1, d2) = (side(c, d, a), side(c, d, b));
            let (d3, d4) = (side(a, b, c), side(a, b, d));
            assert!(
                !(((d1 > 0.0) != (d2 > 0.0)) && ((d3 > 0.0) != (d4 > 0.0))),
                "edges {i} and {j} still cross"
            );
        }
    }

    // A ring that was already simple is left as it was.
    let square = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let (p2, e2) = arrange(&[square], 1e-9);
    assert_eq!((p2.len(), e2.len()), (4, 4), "a simple ring gains nothing");
}

#[test]
fn the_region_a_lapped_ring_meant_survives_its_lap() {
    // The point of an arrangement is that it does not need the ring to be
    // right. A trim loop whose closure went round too far has no consistent
    // orientation and no simple polygon in it — and the region it *meant* is
    // still perfectly well defined, as the part the loop winds around.
    use threers::brep::planar::regions_of;

    // A unit square whose last corner laps the first, as a traced closure does.
    let lapped = vec![
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.05, -0.05],
    ];
    let regions = regions_of(&[lapped], 1e-9);
    let total: f64 = regions
        .iter()
        .map(|r| {
            let n = r.len();
            (0..n)
                .map(|i| {
                    let (a, b) = (r[i], r[(i + 1) % n]);
                    (a[0] * b[1] - b[0] * a[1]) * 0.5
                })
                .sum::<f64>()
        })
        .sum();
    // Not quite the unit square: the lap shaves a corner off it, so the region
    // the loop actually winds around is a little smaller. The shoelace of the
    // lapped ring is 0.975 and the arrangement gives 0.9774 — they differ
    // because the shoelace counts the lapped triangle *negatively* and the
    // arrangement counts it once, which is the more honest reading of "the part
    // this loop encloses".
    assert!(
        (total - 0.977).abs() < 0.005,
        "the region it winds around is 0.977; got {total} from {} regions",
        regions.len()
    );

    // A bowtie is two triangles, and neither of them is the crossing.
    let bowtie = vec![[0.0, 0.0], [2.0, 2.0], [2.0, 0.0], [0.0, 2.0]];
    let two = regions_of(&[bowtie], 1e-9);
    assert_eq!(two.len(), 2, "a bowtie encloses two triangles");
    for t in &two {
        assert_eq!(t.len(), 3, "each is a triangle: {t:?}");
    }
}

#[test]
fn a_curve_asked_between_its_samples_is_still_on_both_surfaces() {
    // A traced curve kept as its samples is only true *at* them. Between them it
    // is a chord, and a chord leaves the surfaces: on two crossing cylinders the
    // midpoints sit 2.7e-3 off, nearly three times the tolerance they were
    // traced at. Everything downstream inherits that — a boundary made of chords
    // is a boundary in the wrong place.
    //
    // Carrying the surfaces instead makes the curve answerable. The spline says
    // where to look next and the surfaces say what is there, so a point comes
    // back on the intersection wherever it is asked.
    use threers::brep::intersect::march;
    use threers::brep::{Body, Curve3d};

    let rod = Body::cylinder([0.0, 0.0, -4.0], [0.0, 0.0, 1.0], 2.0, 8.0);
    let cross = Body::cylinder([-6.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.8, 12.0);
    let a = rod
        .surfaces()
        .iter()
        .find(|s| s.kind() == "cylinder")
        .expect("rod wall");
    let b = cross
        .surfaces()
        .iter()
        .find(|s| s.kind() == "cylinder")
        .expect("bore wall");

    let traced = march(a, b, ([-7.0; 3], [7.0; 3]), 1e-3);
    let Some(Curve3d::Sampled { points, closed }) = traced.first().cloned() else {
        panic!("two crossing cylinders trace to a sampled curve");
    };
    assert!(points.len() > 8, "expected a curve of some length");

    let curve = Curve3d::on_surfaces(a, b, &points, closed, 1e-3).expect("it parameterises");
    let polyline = Curve3d::Sampled {
        points: points.clone(),
        closed,
    };

    // Halfway between every pair of samples, where a polyline is at its worst.
    let (mut worst_curve, mut worst_chord) = (0.0f64, 0.0f64);
    for i in 0..points.len() {
        let t = i as f64 + 0.5;
        let off = |p| a.distance(p).abs().max(b.distance(p).abs());
        worst_curve = worst_curve.max(off(curve.point(t)));
        worst_chord = worst_chord.max(off(polyline.point(t)));
    }
    // Against each other, not against a constant.
    //
    // This asked `worst_chord > 1e-3` — that a polyline leaves the surfaces by
    // more than the tolerance — which was true only while the trace stepped by a
    // hundredth of its bounding box. A step of `sqrt(8·r·tol)` puts a chord's
    // sagitta at the tolerance *by construction*, so the figure came to 0.00099
    // and the assertion failed for the sampling being right. The claim was never
    // about the number: it is that reading the curve beats reading its chords.
    assert!(
        worst_curve * 20.0 < worst_chord,
        "the curve should hug the surfaces far closer than the chord: \
         curve out by {worst_curve}, chord by {worst_chord}"
    );
    assert!(
        worst_curve < 1e-5,
        "the curve should be on the surfaces; it is out by {worst_curve}"
    );
}

#[test]
fn a_surface_says_its_own_period_and_answers_on_the_branch_asked_for() {
    // `invert` answers in the canonical period, which is the right answer to a
    // question almost nobody is asking. A face's rings live wherever the
    // modelling left them, and a point inverted canonically can miss its own
    // face by a whole turn — which is exactly how a sphere cut twice by one
    // sphere came back an eighth smaller, its cavity wall's `u` running from
    // `pi` to `3pi` while every point of it inverted into `(-pi, pi]`.
    use std::f64::consts::TAU;
    use threers::brep::Surface;

    let cyl = Surface::Cylinder {
        origin: [0.0, 0.0, 0.0],
        axis: [0.0, 0.0, 1.0],
        x_dir: [1.0, 0.0, 0.0],
        radius: 2.0,
    };
    let plane = Surface::Plane {
        origin: [0.0; 3],
        normal: [0.0, 0.0, 1.0],
        x_dir: [1.0, 0.0, 0.0],
    };

    // The surface knows what a turn is; the caller no longer assumes.
    assert_eq!(cyl.period(), (Some(TAU), None));
    assert_eq!(plane.period(), (None, None));

    // One point of the cylinder, asked for on three different branches.
    let p = cyl.point(0.5, 3.0);
    for turns in [-1.0, 0.0, 1.0, 2.0] {
        let anchor = 0.5 + TAU * turns;
        let (u, v) = cyl
            .invert_near(p, (anchor, 0.0))
            .expect("a cylinder inverts");
        assert!(
            (u - anchor).abs() < 1e-9,
            "asked near {anchor}, answered {u}"
        );
        assert!((v - 3.0).abs() < 1e-9, "the height should not move: {v}");
        // And it is the same point, whichever branch names it.
        let q = cyl.point(u, v);
        let d = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt();
        assert!(d < 1e-9, "branch {turns} names a different point, {d} away");
    }

    // A plane has no branches to choose between, so it answers plainly.
    let flat = plane.point(1.5, -2.5);
    let (u, v) = plane
        .invert_near(flat, (100.0, 100.0))
        .expect("a plane inverts");
    assert!((u - 1.5).abs() < 1e-9 && (v + 2.5).abs() < 1e-9, "{u} {v}");
}

/// A rim is not a hole, and the ring can say so itself.
///
/// A closed ring on a cylinder is two different faces depending on how it is
/// read, and signed area cannot choose: a bore's rim read as a hole leaves the
/// tool's wall whole, and read as a divider cuts it into bands. Only the second
/// keeps the band inside the rod. What separates them is that a rim goes all
/// the way around the surface, and a hole — by being a hole *in* something —
/// cannot.
#[test]
fn a_ring_that_goes_all_the_way_around_is_not_a_hole_in_anything() {
    let period = (Some(std::f64::consts::TAU), None);

    let rim: Vec<[f64; 2]> = (0..24)
        .map(|i| {
            let u = std::f64::consts::TAU * i as f64 / 24.0;
            [u, 4.0 + 0.24 * u.cos()]
        })
        .collect();
    let rim = TrimLoop {
        vertices: vec![],
        area: signed_area(&rim),
        uv: rim,
    };
    assert_eq!(rim.wraps(period), [true, false], "a rim wraps the cylinder");

    let hole: Vec<[f64; 2]> = (0..24)
        .map(|i| {
            let a = std::f64::consts::TAU * i as f64 / 24.0;
            [3.0 + 0.4 * a.cos(), 4.0 + 0.4 * a.sin()]
        })
        .collect();
    let hole = TrimLoop {
        vertices: vec![],
        area: signed_area(&hole),
        uv: hole,
    };
    assert_eq!(
        hole.wraps(period),
        [false, false],
        "a real hole does not wrap"
    );

    // The two are indistinguishable by the measure that decides every other
    // question about a ring, which is the whole reason this one is needed.
    assert!(rim.area.abs() > 0.0 && hole.area.abs() > 0.0);
}

fn signed_area(uv: &[[f64; 2]]) -> f64 {
    let mut a = 0.0;
    for i in 0..uv.len() {
        let p = uv[i];
        let q = uv[(i + 1) % uv.len()];
        a += p[0] * q[1] - q[0] * p[1];
    }
    a / 2.0
}
