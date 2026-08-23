//! Provenance for the built-in geometry generators.
//!
//! Each generator calls one function from here as its last act, so the
//! instrumentation is a single line at each site rather than a `#[cfg]` block
//! threaded through the meshing code.
//!
//! # Why these live here and not in the generators
//!
//! Two reasons. The tagging needs the *axis conventions* of every primitive in
//! one place — `SphereGeometry` and `CylinderGeometry` are Y-up, `TorusGeometry`
//! is Z-up, `PlaneGeometry` faces +Z — and having them scattered is how they
//! drift. And every function here is checked the same way, by
//! [`crate::brep::SurfaceTable::max_deviation`]: a tag is a claim about where
//! the triangles are, and the tests measure it rather than trusting it.
//!
//! `!(r > 0.0)` guards a degenerate primitive rather than tagging it wrongly.
//! The negated form is deliberate — it is true for NaN, where `r <= 0.0` is
//! false and a NaN radius would sail through into the surface. Same reasoning as
//! `crate::nurbs::curve`.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use crate::core::BufferGeometry;

use super::{Surface, SurfaceTable};

/// Tag a whole geometry with one surface.
fn tag_uniform(g: &mut BufferGeometry, surface: Surface) {
    let tris = super::triangle_count(g);
    g.set_surfaces(SurfaceTable::uniform(surface, tris));
}

/// `SphereGeometry` — Y-up, `θ = 0` at `+Y`, `φ = 0` at `-X`.
pub fn tag_sphere(g: &mut BufferGeometry, radius: f32) {
    if !(radius > 0.0) {
        return;
    }
    let s = Surface::sphere([0.0; 3], radius as f64)
        .with_axis([0.0, 1.0, 0.0])
        .with_x_dir([-1.0, 0.0, 0.0]);
    tag_uniform(g, s);
}

/// `TorusGeometry` — the ring lies in the XY plane, so the axis is `+Z` and
/// `u = 0` is `+X`.
pub fn tag_torus(g: &mut BufferGeometry, radius: f32, tube: f32) {
    if !(radius > 0.0) || !(tube > 0.0) {
        return;
    }
    let s = Surface::torus([0.0; 3], [0.0, 0.0, 1.0], radius as f64, tube as f64)
        .with_x_dir([1.0, 0.0, 0.0]);
    tag_uniform(g, s);
}

/// `PlaneGeometry` / `CircleGeometry` / `RingGeometry` — the XY plane facing `+Z`.
pub fn tag_xy_plane(g: &mut BufferGeometry) {
    let s = Surface::plane([0.0; 3], [0.0, 0.0, 1.0]).with_x_dir([1.0, 0.0, 0.0]);
    tag_uniform(g, s);
}

/// `BoxGeometry` — six planes, in the generator's face order (+X, −X, +Y, −Y,
/// +Z, −Z), two triangles each.
pub fn tag_box(g: &mut BufferGeometry, width: f32, height: f32, depth: f32) {
    let (hx, hy, hz) = (width as f64 * 0.5, height as f64 * 0.5, depth as f64 * 0.5);
    let faces = [
        ([hx, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ([-hx, 0.0, 0.0], [-1.0, 0.0, 0.0]),
        ([0.0, hy, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, -hy, 0.0], [0.0, -1.0, 0.0]),
        ([0.0, 0.0, hz], [0.0, 0.0, 1.0]),
        ([0.0, 0.0, -hz], [0.0, 0.0, -1.0]),
    ];
    let surfaces: Vec<Surface> = faces.iter().map(|&(o, n)| Surface::plane(o, n)).collect();

    let tris = super::triangle_count(g);
    // Two triangles per face, in face order. If the generator's triangle count
    // ever stops being 12 this no longer describes it, so tag nothing rather
    // than tag it wrongly.
    if tris != 12 {
        return;
    }
    if let Some(t) = SurfaceTable::new(surfaces, (0..12u32).map(|i| i / 2).collect()) {
        g.set_surfaces(t);
    }
}

/// `CylinderGeometry` — Y-up, `θ = 0` at `+Z`, caps at `±height/2`.
///
/// The side is a [`Surface::Cylinder`] when the radii match and a
/// [`Surface::Cone`] when they do not, with the apex extrapolated from the two
/// radii. That extrapolation is the whole point: a *truncated* cone's apex is
/// not on the mesh at all, and nothing downstream could recover it from the
/// triangles.
///
/// The triangle counts are passed in rather than recomputed, because only the
/// generator knows how many it emitted — caps are skipped for zero radii and for
/// `open_ended`, and re-deriving that here would be a second copy of the rule.
pub fn tag_cylinder(
    g: &mut BufferGeometry,
    radius_top: f32,
    radius_bottom: f32,
    height: f32,
    side_triangles: usize,
    top_cap_triangles: usize,
    bottom_cap_triangles: usize,
) {
    let (rt, rb, h) = (radius_top as f64, radius_bottom as f64, height as f64);
    if h.abs() < 1e-12 {
        return;
    }
    let half = h * 0.5;

    let side = if (rt - rb).abs() < 1e-12 {
        if !(rt > 0.0) {
            return;
        }
        Surface::cylinder([0.0; 3], [0.0, 1.0, 0.0], rt)
    } else {
        // r(y) is linear with r(+half) = rt and r(−half) = rb, so the apex sits
        // where it reaches zero — above the top when the bottom is wider.
        let y_apex = half + h * rt / (rb - rt);
        // The axis points the way the radius grows.
        let (axis, far_r, far_d) = if rb > rt {
            ([0.0, -1.0, 0.0], rb, y_apex + half)
        } else {
            ([0.0, 1.0, 0.0], rt, half - y_apex)
        };
        match Surface::cone_from_rim([0.0, y_apex, 0.0], axis, far_r, far_d) {
            Some(c) => c,
            None => return,
        }
    }
    .with_x_dir([0.0, 0.0, 1.0]);

    let mut surfaces = vec![side];
    let mut tri_face = vec![0u32; side_triangles];

    if top_cap_triangles > 0 {
        surfaces
            .push(Surface::plane([0.0, half, 0.0], [0.0, 1.0, 0.0]).with_x_dir([0.0, 0.0, 1.0]));
        tri_face.extend(std::iter::repeat_n(
            (surfaces.len() - 1) as u32,
            top_cap_triangles,
        ));
    }
    if bottom_cap_triangles > 0 {
        surfaces
            .push(Surface::plane([0.0, -half, 0.0], [0.0, -1.0, 0.0]).with_x_dir([0.0, 0.0, 1.0]));
        tri_face.extend(std::iter::repeat_n(
            (surfaces.len() - 1) as u32,
            bottom_cap_triangles,
        ));
    }

    if tri_face.len() != super::triangle_count(g) {
        return;
    }
    if let Some(t) = SurfaceTable::new(surfaces, tri_face) {
        g.set_surfaces(t);
    }
}


/// `LatheGeometry` — a profile polyline revolved about `+Y`.
///
/// Exact: the mesh's vertices are `(x·sin φ, y, x·cos φ)` at the same angles the
/// NURBS revolution places them, and the profile between samples is a straight
/// segment in both. So a degree-1 profile curve revolved analytically is the
/// surface the triangles were sampled from, not an approximation of it.
pub fn tag_lathe(
    g: &mut BufferGeometry,
    points: &[crate::math::Vector2],
    phi_start: f32,
    phi_length: f32,
) {
    use crate::nurbs::{construct, NurbsCurve};

    if points.len() < 2 || !(phi_length > 0.0) {
        return;
    }
    // The profile, placed where the mesh starts it. `LatheGeometry` maps a
    // profile point `(x, y)` to `(x·sin φ, y, x·cos φ)`.
    let (s, c) = (phi_start as f64).sin_cos();
    let profile: Vec<[f64; 3]> = points
        .iter()
        .map(|p| {
            let (x, y) = (p.x as f64, p.y as f64);
            [x * s, y, x * c]
        })
        .collect();

    let knots = crate::nurbs::uniform_clamped_knots(1, profile.len());
    let Ok(curve) = NurbsCurve::new(1, knots, &profile, None) else {
        return;
    };
    let surface = construct::revolve(&curve, [0.0; 3], [0.0, 1.0, 0.0], phi_length as f64);
    tag_uniform(g, Surface::nurbs(surface));
}

/// `TubeGeometry` — circular sections carried along a path.
///
/// The sections are lofted rather than swept. A sweep would build its *own*
/// rotation-minimising frame, and the generator has already chosen one; a
/// surface built on different frames places its vertices somewhere else, so the
/// tag would be false. Lofting through the generator's own circles cannot be
/// wrong that way — a loft interpolates its sections exactly, and every mesh
/// vertex lies on one.
pub fn tag_tube(
    g: &mut BufferGeometry,
    centers: &[crate::math::Vector3],
    frames: &[(crate::math::Vector3, crate::math::Vector3)],
    radius: f32,
) {
    use crate::nurbs::construct;

    if centers.len() < 2 || centers.len() != frames.len() || !(radius > 0.0) {
        return;
    }
    // The generator sweeps `-radius·cos v · normal - radius·sin v · binormal`,
    // so the circle's own `x` axis is `-normal` and its `y` axis is `-binormal`.
    let sections: Vec<_> = centers
        .iter()
        .zip(frames)
        .map(|(p, (n, b))| {
            construct::circle(
                [p.x as f64, p.y as f64, p.z as f64],
                [-n.x as f64, -n.y as f64, -n.z as f64],
                [-b.x as f64, -b.y as f64, -b.z as f64],
                radius as f64,
            )
        })
        .collect();

    let Ok(surface) = construct::loft(&sections, 3) else {
        return;
    };
    tag_uniform(g, Surface::nurbs(surface));
}

/// `ExtrudeGeometry` — two cap planes plus one plane per contour edge.
///
/// Every face of a straight extrusion is planar, so this needs no NURBS at all:
/// the caps are the `z = 0` and `z = depth` planes, and each wall is the
/// vertical plane through its contour edge. Which makes an extrusion the one
/// swept primitive whose provenance the *analytic* kernel path in Stage 2 can
/// actually use.
///
/// `cap_triangles` is how many triangles the cap triangulator emitted for one
/// cap; the generator interleaves bottom and top, then appends the walls.
pub fn tag_extrude(
    g: &mut BufferGeometry,
    contour: &[crate::math::Vector2],
    depth: f32,
    cap_triangles: usize,
) {
    let n = contour.len();
    if n < 3 {
        return;
    }
    let d = depth as f64;

    let mut surfaces = vec![
        Surface::plane([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]).with_x_dir([1.0, 0.0, 0.0]),
        Surface::plane([0.0, 0.0, d], [0.0, 0.0, 1.0]).with_x_dir([1.0, 0.0, 0.0]),
    ];
    // Caps first, interleaved bottom/top, one pair per triangulated triangle.
    let mut tri_face: Vec<u32> = Vec::with_capacity(2 * cap_triangles + 2 * n);
    for _ in 0..cap_triangles {
        tri_face.push(0);
        tri_face.push(1);
    }

    // A closed contour sampled per-segment repeats its join points, so some
    // "walls" are degenerate quads. They still have a plane — all four vertices
    // are `(a, 0)` and `(a, depth)`, which lie in *any* vertical plane through
    // `a` — so the neighbouring edge's normal is exactly right rather than
    // merely convenient. Bailing on them instead would leave every extrusion
    // untagged, which is how this was first written and why it never fired.
    let mut normals: Vec<Option<[f64; 2]>> = Vec::with_capacity(n);
    for i in 0..n {
        let a = contour[i];
        let b = contour[(i + 1) % n];
        let (ex, ey) = ((b.x - a.x) as f64, (b.y - a.y) as f64);
        let len = (ex * ex + ey * ey).sqrt();
        normals.push((len > 1e-12).then(|| [ey / len, -ex / len]));
    }
    if normals.iter().all(|x| x.is_none()) {
        return; // no contour at all
    }

    for i in 0..n {
        let a = contour[i];
        // Nearest preceding real edge, wrapping.
        let nrm = (0..n)
            .find_map(|k| normals[(i + n - k) % n])
            .expect("at least one edge is non-degenerate");
        surfaces.push(
            Surface::plane([a.x as f64, a.y as f64, 0.0], [nrm[0], nrm[1], 0.0])
                .with_x_dir([0.0, 0.0, 1.0]),
        );
        let si = (surfaces.len() - 1) as u32;
        tri_face.push(si);
        tri_face.push(si);
    }

    if tri_face.len() != super::triangle_count(g) {
        return;
    }
    if let Some(t) = SurfaceTable::new(surfaces, tri_face) {
        g.set_surfaces(t);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometries::{
        BoxGeometry, CircleGeometry, ConeGeometry, CylinderGeometry, PlaneGeometry, SphereGeometry,
        TorusGeometry,
    };

    /// Every generator's tag is a claim about where its triangles are; this is
    /// the measurement that makes it falsifiable. The bound is f32-scale
    /// because the positions are f32 in the buffer.
    fn assert_tagged(g: &BufferGeometry, what: &str, scale: f64) {
        let t = g
            .surface_table()
            .unwrap_or_else(|| panic!("{what}: no provenance attached"));
        assert_eq!(t.triangle_count(), super::super::triangle_count(g));
        let dev = t.max_deviation(g);
        assert!(
            dev < 1e-5 * scale.max(1.0),
            "{what}: a tagged vertex is {dev} from the surface it claims"
        );
    }

    #[test]
    fn sphere_is_tagged() {
        for &(r, w, h) in &[(1.0f32, 32, 16), (5.0, 8, 6), (0.25, 64, 32)] {
            let g = SphereGeometry::new(r, w, h);
            assert_tagged(&g, "sphere", r as f64);
            assert_eq!(g.surface_table().unwrap().surfaces()[0].kind(), "sphere");
        }
    }

    #[test]
    fn sphere_parameterization_matches_the_generators_own() {
        // `invert` must return the angles the mesh was built with, not ones
        // rotated by an arbitrary frame choice — otherwise UVs from provenance
        // and UVs from the generator disagree about where the seam is.
        let g = SphereGeometry::new(1.0, 32, 16);
        let s = &g.surface_table().unwrap().surfaces()[0];
        // The generator puts φ = 0 at −X and θ = 0 (v = +π/2) at +Y.
        let north = s.point(0.0, std::f64::consts::FRAC_PI_2);
        assert!(
            crate::nurbs::v3::dist(north, [0.0, 1.0, 0.0]) < 1e-12,
            "{north:?}"
        );
        let seam = s.point(0.0, 0.0);
        assert!(
            crate::nurbs::v3::dist(seam, [-1.0, 0.0, 0.0]) < 1e-12,
            "{seam:?}"
        );
    }

    #[test]
    fn box_is_tagged_with_six_planes() {
        let g = BoxGeometry::new(2.0, 3.0, 4.0);
        assert_tagged(&g, "box", 4.0);
        let t = g.surface_table().unwrap();
        assert_eq!(t.surfaces().len(), 6);
        assert!(t.surfaces().iter().all(|s| s.kind() == "plane"));
        assert_eq!(t.groups().len(), 6, "each face must own triangles");
        for (_, tris) in t.groups() {
            assert_eq!(tris.len(), 2);
        }
    }

    #[test]
    fn cylinder_is_tagged_as_a_cylinder_plus_two_caps() {
        let g = CylinderGeometry::new(2.0, 2.0, 5.0, 24, 2, false, 0.0, std::f32::consts::PI * 2.0);
        assert_tagged(&g, "cylinder", 5.0);
        let t = g.surface_table().unwrap();
        assert_eq!(t.surfaces().len(), 3);
        assert_eq!(t.surfaces()[0].kind(), "cylinder");
        assert_eq!(t.surfaces()[1].kind(), "plane");
        assert_eq!(t.surfaces()[2].kind(), "plane");
    }

    #[test]
    fn an_open_cylinder_has_no_cap_surfaces() {
        let g = CylinderGeometry::new(1.0, 1.0, 3.0, 16, 1, true, 0.0, std::f32::consts::PI * 2.0);
        assert_tagged(&g, "open cylinder", 3.0);
        let t = g.surface_table().unwrap();
        assert_eq!(t.surfaces().len(), 1);
        assert_eq!(t.surfaces()[0].kind(), "cylinder");
    }

    #[test]
    fn a_cone_recovers_an_apex_that_is_not_on_the_mesh() {
        // radius_top = 0, so the apex is a mesh vertex and easy. The real test
        // is the truncated case below.
        let g = ConeGeometry::new(2.0, 4.0, 24, 1, false, 0.0, std::f32::consts::PI * 2.0);
        assert_tagged(&g, "cone", 4.0);
        let t = g.surface_table().unwrap();
        assert_eq!(t.surfaces()[0].kind(), "cone");
        match &t.surfaces()[0] {
            Surface::Cone {
                apex, half_angle, ..
            } => {
                assert!(
                    crate::nurbs::v3::dist(*apex, [0.0, 2.0, 0.0]) < 1e-9,
                    "{apex:?}"
                );
                assert!((half_angle - (2.0f64 / 4.0).atan()).abs() < 1e-9);
            }
            other => panic!("expected a cone, got {}", other.kind()),
        }
    }

    #[test]
    fn a_truncated_cone_extrapolates_its_missing_apex() {
        // Radii 1 → 3 over height 4. The slope is 0.5 per unit, so the apex is
        // 2 above the top rim: at y = 2 + 2 = 4. Nothing on the mesh is there.
        let g = CylinderGeometry::new(1.0, 3.0, 4.0, 24, 1, false, 0.0, std::f32::consts::PI * 2.0);
        assert_tagged(&g, "truncated cone", 4.0);
        let t = g.surface_table().unwrap();
        match &t.surfaces()[0] {
            Surface::Cone {
                apex,
                axis,
                half_angle,
                ..
            } => {
                assert!(
                    crate::nurbs::v3::dist(*apex, [0.0, 4.0, 0.0]) < 1e-9,
                    "{apex:?}"
                );
                assert!(crate::nurbs::v3::dist(*axis, [0.0, -1.0, 0.0]) < 1e-12);
                assert!((half_angle - 0.5f64.atan()).abs() < 1e-9, "{half_angle}");
            }
            other => panic!("expected a cone, got {}", other.kind()),
        }
    }

    #[test]
    fn an_inverted_truncated_cone_points_the_other_way() {
        // Wider at the top: the apex is below, and the axis points up.
        let g = CylinderGeometry::new(3.0, 1.0, 4.0, 24, 1, false, 0.0, std::f32::consts::PI * 2.0);
        assert_tagged(&g, "inverted truncated cone", 4.0);
        match &g.surface_table().unwrap().surfaces()[0] {
            Surface::Cone { apex, axis, .. } => {
                assert!(
                    crate::nurbs::v3::dist(*apex, [0.0, -4.0, 0.0]) < 1e-9,
                    "{apex:?}"
                );
                assert!(crate::nurbs::v3::dist(*axis, [0.0, 1.0, 0.0]) < 1e-12);
            }
            other => panic!("expected a cone, got {}", other.kind()),
        }
    }

    #[test]
    fn torus_is_tagged() {
        let g = TorusGeometry::new(4.0, 1.0, 16, 24, std::f32::consts::PI * 2.0);
        assert_tagged(&g, "torus", 5.0);
        assert_eq!(g.surface_table().unwrap().surfaces()[0].kind(), "torus");
    }

    #[test]
    fn plane_and_circle_are_tagged() {
        let p = PlaneGeometry::with_segments(3.0, 2.0, 4, 3);
        assert_tagged(&p, "plane", 3.0);
        assert_eq!(p.surface_table().unwrap().surfaces()[0].kind(), "plane");

        let c = CircleGeometry::new(2.0, 24, 0.0, std::f32::consts::PI * 2.0);
        assert_tagged(&c, "circle", 2.0);
        assert_eq!(c.surface_table().unwrap().surfaces()[0].kind(), "plane");
    }

    #[test]
    fn a_partial_cylinder_arc_is_still_tagged() {
        // A θ-range less than a full turn samples part of the same unbounded
        // cylinder; the surface is unchanged, only the triangles are fewer.
        let g = CylinderGeometry::new(1.0, 1.0, 2.0, 16, 1, false, 0.3, 1.9);
        assert_tagged(&g, "partial cylinder", 2.0);
    }

    /// The exact-CSG kernel recovers "one flat face" by hashing a quantized
    /// plane out of the triangles (`exact_csg::plane_key`). With real face
    /// identity available that guess becomes checkable: on a box, the hash must
    /// produce exactly the partition the provenance already knows.
    ///
    /// Only planar primitives are compared. The hash is *defined* to group
    /// coplanar triangles, so on a sphere it correctly produces one group per
    /// facet and disagreeing with `tri_face` there is the point, not a bug —
    /// which is the whole argument for Stage 2.
    #[cfg(feature = "openscad")]
    #[test]
    fn the_kernels_plane_hash_agrees_with_provenance_on_planar_faces() {
        use crate::brep::triangle_vertices;
        use std::collections::HashMap;

        let g = BoxGeometry::new(2.0, 3.0, 4.0);
        let t = g.surface_table().expect("box is tagged");

        let mut hash_of: HashMap<(i64, i64, i64, i64), Vec<usize>> = HashMap::new();
        for tri in 0..t.triangle_count() {
            let v = triangle_vertices(&g, tri).unwrap();
            hash_of
                .entry(crate::exact_csg::plane_key(&v))
                .or_default()
                .push(tri);
        }

        let mut by_hash: Vec<Vec<usize>> = hash_of.into_values().collect();
        let mut by_provenance: Vec<Vec<usize>> =
            t.groups().into_iter().map(|(_, tris)| tris).collect();
        by_hash.iter_mut().for_each(|v| v.sort_unstable());
        by_provenance.iter_mut().for_each(|v| v.sort_unstable());
        by_hash.sort();
        by_provenance.sort();

        assert_eq!(
            by_hash, by_provenance,
            "the kernel's plane hash and the provenance disagree about a box's faces"
        );
    }

    /// On a curved surface the hash cannot help — it makes one "face" per facet
    /// where provenance sees one sphere. Asserting the gap keeps it visible:
    /// closing it is what Stage 2 (`brep-csg`) is for.
    #[cfg(feature = "openscad")]
    #[test]
    fn the_kernels_plane_hash_cannot_see_a_curved_face() {
        use crate::brep::triangle_vertices;
        use std::collections::HashSet;

        let g = SphereGeometry::new(1.0, 16, 8);
        let t = g.surface_table().expect("sphere is tagged");

        let keys: HashSet<_> = (0..t.triangle_count())
            .filter_map(|tri| triangle_vertices(&g, tri))
            .map(|v| crate::exact_csg::plane_key(&v))
            .collect();

        assert_eq!(t.groups().len(), 1, "provenance sees one sphere");
        assert!(
            keys.len() > 50,
            "the hash saw {} faces where there is one surface",
            keys.len()
        );
    }

    #[test]
    fn lathe_tube_and_extrude_are_tagged() {
        use crate::geometries::{ExtrudeGeometry, LatheGeometry, TubeGeometry};
        use crate::math::{Vector2, Vector3};

        // A revolved profile: exact, because the mesh samples the same
        // revolution the NURBS surface describes.
        let profile = [
            Vector2::new(1.0, 0.0),
            Vector2::new(2.0, 1.0),
            Vector2::new(1.5, 2.5),
            Vector2::new(0.5, 3.0),
        ];
        let lathe = LatheGeometry::new(&profile, 24, 0.0, std::f32::consts::PI * 2.0);
        assert_tagged(&lathe, "lathe", 3.0);
        assert_eq!(lathe.surface_table().unwrap().surfaces()[0].kind(), "nurbs");

        // A partial revolution names the same surface over a shorter sweep.
        let partial = LatheGeometry::new(&profile, 12, 0.3, 1.9);
        assert_tagged(&partial, "partial lathe", 3.0);

        // A tube: lofted through the generator's own circles, so every ring
        // vertex is on the surface.
        let path = crate::curves::CatmullRomCurve3::new(vec![
            Vector3::new(-3.0, 0.0, 0.0),
            Vector3::new(-1.0, 2.0, 1.0),
            Vector3::new(1.0, -1.0, -1.0),
            Vector3::new(3.0, 0.5, 0.0),
        ]);
        let tube = TubeGeometry::new(&path, 16, 0.4, 12, false);
        assert_tagged(&tube, "tube", 3.0);
        assert_eq!(tube.surface_table().unwrap().surfaces()[0].kind(), "nurbs");

        // An extrusion is all planes — the one swept primitive whose provenance
        // the analytic CSG path can use.
        let mut shape = crate::curves::Shape::new();
        shape
            .outline
            .curve_path
            .add(Box::new(crate::curves::LineCurve::new(
                Vector2::new(0.0, 0.0),
                Vector2::new(2.0, 0.0),
            )));
        shape
            .outline
            .curve_path
            .add(Box::new(crate::curves::LineCurve::new(
                Vector2::new(2.0, 0.0),
                Vector2::new(2.0, 1.0),
            )));
        shape
            .outline
            .curve_path
            .add(Box::new(crate::curves::LineCurve::new(
                Vector2::new(2.0, 1.0),
                Vector2::new(0.0, 1.0),
            )));
        shape
            .outline
            .curve_path
            .add(Box::new(crate::curves::LineCurve::new(
                Vector2::new(0.0, 1.0),
                Vector2::new(0.0, 0.0),
            )));
        let ex = ExtrudeGeometry::new(&shape, 3.0, 2);
        assert_tagged(&ex, "extrude", 3.0);
        let t = ex.surface_table().unwrap();
        assert!(
            t.surfaces().iter().all(|s| s.kind() == "plane"),
            "a straight extrusion has no curved face"
        );
        assert!(t.surfaces().len() >= 3, "two caps and at least one wall");
    }

    #[test]
    fn a_degenerate_lathe_is_left_untagged() {
        use crate::geometries::LatheGeometry;
        use crate::math::Vector2;
        let profile = [Vector2::new(1.0, 0.0), Vector2::new(1.0, 2.0)];
        // A zero sweep has no surface to name.
        let g = LatheGeometry::new(&profile, 12, 0.0, 0.0);
        assert!(g.surface_table().is_none());
    }

    #[test]
    fn a_degenerate_primitive_is_left_untagged_rather_than_mistagged() {
        let g = SphereGeometry::new(0.0, 8, 6);
        assert!(
            g.surface_table().is_none(),
            "a zero-radius sphere has no radius to claim"
        );
    }
}
