//! Convert non-mesh drawables into triangle soup the path tracer can hit.

use std::sync::Arc;

use crate::core::{BufferAttribute, BufferGeometry};
use crate::materials::Material;
use crate::math::{Matrix4, Vector3};

/// A camera-facing quad in world space, as two triangles.
pub fn sprite_quads(world: &Matrix4, size: f32) -> Arc<BufferGeometry> {
    let hs = size.max(1e-4) * 0.5;
    let mut positions = vec![
        -hs, -hs, 0.0, //
        hs, -hs, 0.0, //
        hs, hs, 0.0, //
        -hs, hs, 0.0,
    ];
    for chunk in positions.chunks_exact_mut(3) {
        let p = Vector3::new(chunk[0], chunk[1], chunk[2]).apply_matrix4(world);
        chunk[0] = p.x;
        chunk[1] = p.y;
        chunk[2] = p.z;
    }
    let mut geom = BufferGeometry::new();
    geom.set_attribute("position", BufferAttribute::new(positions, 3));
    geom.set_index(vec![0, 1, 2, 0, 2, 3]);
    Arc::new(geom)
}

/// Expand line segments into thin world-space quads (view-independent ribbon).
/// Two unit vectors perpendicular to `n` and to each other.
///
/// Duff et al., "Building an Orthonormal Basis, Revisited" (JCGT 2017).
/// Branchless and stable for every direction, including the poles.
///
/// The previous construction crossed with a fixed UP and fell back to RIGHT
/// when that got short. Near the fallback the cross product is tiny and
/// normalising it amplifies whatever float error is in it, so a segment nearly
/// parallel to UP got a blade orientation that wandered — and the threshold
/// made the wander worst just before the branch rescued it.
fn orthonormal_basis(n: Vector3) -> (Vector3, Vector3) {
    let sign = if n.z >= 0.0 { 1.0f32 } else { -1.0 };
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    (
        Vector3::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x),
        Vector3::new(b, sign + n.y * n.y * a, -n.y),
    )
}

pub fn line_segment_quads(
    geometry: &BufferGeometry,
    world: &Matrix4,
    width: f32,
) -> Option<Arc<BufferGeometry>> {
    let pos = geometry.get_attribute("position")?;
    if pos.item_size < 3 || pos.count() < 2 {
        return None;
    }
    let half = width.max(1e-4) * 0.5;
    let mut out_pos = Vec::new();
    let mut out_idx = Vec::new();
    // Carry vertex colours onto the quads.
    //
    // These expansions built a geometry holding only positions, so a line set
    // with a colour per segment reached the tracer as untinted white and the
    // traced image came out one flat shade however carefully the integrator
    // handled colour downstream.
    let col = geometry.get_attribute("color").filter(|a| a.item_size >= 3);
    let mut out_col: Vec<f32> = Vec::new();
    let read_col = |i: usize| -> [f32; 3] {
        match col {
            Some(a) => {
                let b = i * a.item_size;
                [a.array[b], a.array[b + 1], a.array[b + 2]]
            }
            None => [1.0, 1.0, 1.0],
        }
    };
    let read = |i: usize| -> Vector3 {
        let b = i * pos.item_size;
        Vector3::new(pos.array[b], pos.array[b + 1], pos.array[b + 2]).apply_matrix4(world)
    };
    let n_verts = pos.count();
    // Topology comes from the index when there is one, and from vertex order
    // otherwise. Two things were wrong here:
    //
    // - The index buffer was ignored entirely, so an indexed `LineSegments`
    //   was expanded by position order and traced as different geometry from
    //   the one the rasteriser draws with the same buffers.
    // - Segments-versus-strip was guessed from `n_verts % 2 == 0`. There is no
    //   strip variant — `LineSegments` means pairs — so the parity test could
    //   only ever mis-handle the odd case, silently chaining vertices that were
    //   meant to be disjoint rather than dropping the unpaired last one.
    let pairs: Vec<(usize, usize)> = match geometry.index.as_ref() {
        Some(idx) => idx
            .chunks_exact(2)
            .map(|c| (c[0] as usize, c[1] as usize))
            .filter(|(a, b)| *a < n_verts && *b < n_verts)
            .collect(),
        None => (0..n_verts)
            .step_by(2)
            .filter(|i| i + 1 < n_verts)
            .map(|i| (i, i + 1))
            .collect(),
    };
    for &(i, j) in &pairs {
        let a = read(i);
        let b = read(j);
        let dir = b - a;
        if dir.length_sq() < 1e-12 {
            continue;
        }
        let t = dir.normalize();
        let (side, other) = orthonormal_basis(t);
        // Each end's own colour, so a segment that changes colour along its
        // length interpolates across the quad rather than snapping.
        let (ca, cb) = (read_col(i), read_col(j));

        // Two perpendicular blades, not one ribbon.
        //
        // A single ribbon lies in a fixed world plane, so a ray arriving along
        // its normal sees it edge-on and it disappears. Sampling random segment
        // and view directions: a lone ribbon keeps only 0.500 of its nominal
        // width on average and falls below a tenth of it for 10.0% of
        // segment/view pairs. Crossing a second blade through it takes those to
        // 0.707 and 0.6%. Costs twice the triangles, which is the cheapest
        // honest fix — the alternative is orienting toward the camera, and a
        // path tracer has no single camera direction to orient to.
        for plane in [side, other] {
            let base = (out_pos.len() / 3) as u32;
            for (corner, c) in [
                (a - plane * half, ca),
                (a + plane * half, ca),
                (b + plane * half, cb),
                (b - plane * half, cb),
            ] {
                out_pos.extend_from_slice(&[corner.x, corner.y, corner.z]);
                out_col.extend_from_slice(&c);
            }
            out_idx.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }
    if out_idx.is_empty() {
        return None;
    }
    let mut geom = BufferGeometry::new();
    geom.set_attribute("position", BufferAttribute::new(out_pos, 3));
    if col.is_some() {
        geom.set_attribute("color", BufferAttribute::new(out_col, 3));
    }
    geom.set_index(out_idx);
    Some(Arc::new(geom))
}

/// Expand each point into a small octahedron (8 triangles).
pub fn point_spheres(
    geometry: &BufferGeometry,
    world: &Matrix4,
    radius: f32,
) -> Option<Arc<BufferGeometry>> {
    let pos = geometry.get_attribute("position")?;
    if pos.item_size < 3 || pos.count() == 0 {
        return None;
    }
    let r = radius.max(1e-4);
    let mut out_pos = Vec::new();
    let mut out_idx = Vec::new();
    // Same as the line case: without this a coloured point cloud traces white.
    let col = geometry.get_attribute("color").filter(|a| a.item_size >= 3);
    let mut out_col: Vec<f32> = Vec::new();
    for i in 0..pos.count() {
        let b = i * pos.item_size;
        let c = Vector3::new(pos.array[b], pos.array[b + 1], pos.array[b + 2]).apply_matrix4(world);
        let vc = match col {
            Some(a) => {
                let cb = i * a.item_size;
                [a.array[cb], a.array[cb + 1], a.array[cb + 2]]
            }
            None => [1.0, 1.0, 1.0],
        };
        let base = (out_pos.len() / 3) as u32;
        let axes = [
            Vector3::new(r, 0.0, 0.0),
            Vector3::new(-r, 0.0, 0.0),
            Vector3::new(0.0, r, 0.0),
            Vector3::new(0.0, -r, 0.0),
            Vector3::new(0.0, 0.0, r),
            Vector3::new(0.0, 0.0, -r),
        ];
        for a in &axes {
            let p = c + *a;
            out_pos.extend_from_slice(&[p.x, p.y, p.z]);
            out_col.extend_from_slice(&vc);
        }
        let v = |k: u32| base + k;
        for tri in [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ] {
            out_idx.extend_from_slice(&[v(tri[0]), v(tri[1]), v(tri[2])]);
        }
    }
    let mut geom = BufferGeometry::new();
    geom.set_attribute("position", BufferAttribute::new(out_pos, 3));
    if col.is_some() {
        geom.set_attribute("color", BufferAttribute::new(out_col, 3));
    }
    geom.set_index(out_idx);
    Some(Arc::new(geom))
}

/// Default line width in world units when the material does not specify one.
pub fn line_width_for(material: &Material) -> f32 {
    // Honour the material rather than assuming a scale.
    //
    // This was a flat 0.02 world units for every line. That is a reasonable
    // guess only if the scene is a few units across; in one measured in
    // micrometres and spanning ~1000 of them, every line became a sliver
    // 50 000x smaller than the subject — far below a pixel, so the traced image
    // was sparse noise rather than geometry. `LineBasicMaterial::line_width`
    // already existed and was simply ignored.
    //
    // The 0.02 fallback is kept for materials that carry no width of their own.
    match material {
        Material::Line(m) if m.line_width > 0.0 => m.line_width,
        _ => 0.02,
    }
}

/// Point radius in world units, from the material.
pub fn point_radius_for(material: &Material) -> f32 {
    // Same fault the line width had: a flat 0.05 world units for every points
    // material, with `PointsMaterial::size` sitting there unread. A scene
    // measured in micrometres got points 20 000x smaller than itself.
    //
    // `size` is a diameter in three.js, so halve it. The 0.05 fallback stays
    // for materials that carry no size.
    match material {
        Material::Points(m) if m.size > 0.0 => m.size * 0.5,
        _ => 0.05,
    }
}


#[cfg(test)]
mod color_tests {
    use super::*;
    use crate::core::BufferAttribute;

    fn two_segment_lines() -> BufferGeometry {
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], 3),
        );
        g.set_attribute(
            "color",
            BufferAttribute::new(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], 3),
        );
        g
    }

    #[test]
    fn quads_keep_the_line_colours() {
        // The expansion built a positions-only geometry, so every coloured line
        // reached the tracer white and the traced image was one flat shade.
        let g = line_segment_quads(&two_segment_lines(), &Matrix4::identity(), 0.1)
            .expect("quads");
        let c = g.get_attribute("color").expect("colour survived the expansion");
        assert_eq!(c.count(), 16, "four corners per blade, two blades per segment");
        assert_eq!(&c.array[0..3], &[1.0, 0.0, 0.0]);
        assert_eq!(
            &c.array[24..27],
            &[0.0, 0.0, 1.0],
            "second segment keeps its own colour"
        );
    }

    #[test]
    fn a_segment_is_visible_from_any_direction() {
        // The bug: one ribbon lies in a fixed plane and vanishes edge-on. Two
        // perpendicular blades cannot both be edge-on to the same ray.
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 3),
        );
        let q = line_segment_quads(&g, &Matrix4::identity(), 0.2).expect("quads");
        let idx = q.index.as_ref().expect("indexed");
        assert_eq!(idx.len(), 12, "two quads, six indices each");

        // Face normals of the two blades must not be parallel.
        let p = q.get_attribute("position").expect("positions");
        let v = |i: usize| {
            Vector3::new(p.array[i * 3], p.array[i * 3 + 1], p.array[i * 3 + 2])
        };
        let n0 = (v(1) - v(0)).cross(v(2) - v(0)).normalize();
        let n1 = (v(5) - v(4)).cross(v(6) - v(4)).normalize();
        assert!(
            n0.dot(n1).abs() < 0.1,
            "blades are nearly coplanar ({}), so both vanish from the same view",
            n0.dot(n1)
        );
    }

    #[test]
    fn an_indexed_line_set_follows_its_index() {
        // The expansion read positions in order and ignored the index, so an
        // indexed LineSegments traced as different geometry from the one the
        // rasteriser draws with the same buffers.
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0], 3),
        );
        // One segment, from vertex 0 to vertex 2 — not 0->1.
        g.set_index(vec![0, 2]);
        let q = line_segment_quads(&g, &Matrix4::identity(), 0.1).expect("quads");
        let p = q.get_attribute("position").expect("positions");
        let xs: Vec<f32> = (0..p.count()).map(|i| p.array[i * 3]).collect();
        assert!(
            xs.iter().any(|x| *x > 1.5),
            "the quad stopped at vertex 1, so the index was ignored: {xs:?}"
        );
    }

    #[test]
    fn an_unpaired_last_vertex_is_dropped_not_chained() {
        // Three vertices is one segment plus a leftover. The parity guess used
        // to read this as a strip and invent a second, connected segment.
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0], 3),
        );
        let q = line_segment_quads(&g, &Matrix4::identity(), 0.1).expect("quads");
        let idx = q.index.as_ref().expect("indexed");
        assert_eq!(idx.len(), 12, "one segment: two blades, six indices each");
    }

    #[test]
    fn the_basis_is_orthonormal_for_every_direction() {
        // Including the poles, where the old UP-cross construction degenerated
        // and its fallback threshold made the orientation wander just before
        // rescuing it.
        let dirs = [
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, -1.0),
            Vector3::new(1e-7, 1.0, 0.0).normalize(),
            Vector3::new(0.577, 0.577, 0.577).normalize(),
        ];
        for n in dirs {
            let (a, b) = orthonormal_basis(n);
            assert!((a.length() - 1.0).abs() < 1e-4, "a not unit for {n:?}");
            assert!((b.length() - 1.0).abs() < 1e-4, "b not unit for {n:?}");
            assert!(a.dot(n).abs() < 1e-4, "a not perpendicular to {n:?}");
            assert!(b.dot(n).abs() < 1e-4, "b not perpendicular to {n:?}");
            assert!(a.dot(b).abs() < 1e-4, "a and b not perpendicular for {n:?}");
        }
    }

    #[test]
    fn a_geometry_without_colour_gets_no_colour_attribute() {
        // Rather than a white attribute, which would cost memory to say nothing.
        let mut g = two_segment_lines();
        g.attributes.remove("color");
        let q = line_segment_quads(&g, &Matrix4::identity(), 0.1).expect("quads");
        assert!(q.get_attribute("color").is_none());
    }

    #[test]
    fn point_spheres_keep_their_colours() {
        let g = point_spheres(&two_segment_lines(), &Matrix4::identity(), 0.1).expect("spheres");
        let c = g.get_attribute("color").expect("colour survived");
        assert_eq!(c.count(), 4 * 6, "six vertices per octahedron");
        assert_eq!(&c.array[0..3], &[1.0, 0.0, 0.0]);
    }
}

#[cfg(test)]
mod size_tests {
    use super::*;
    use crate::materials::{LineBasicMaterial, PointsMaterial, StandardMaterial};
    use crate::Color;

    #[test]
    fn line_width_comes_from_the_material() {
        // The bug this guards: both of these returned a hardcoded constant,
        // with the material parameter named `_material`. In a scene measured in
        // micrometres that made every line 50 000x too thin to hit.
        let mut m = LineBasicMaterial::new(Color::WHITE);
        m.line_width = 7.5;
        assert_eq!(line_width_for(&Material::Line(m)), 7.5);
    }

    #[test]
    fn point_radius_is_half_the_material_size() {
        // three.js `size` is a diameter; the tracer wants a radius.
        let m = PointsMaterial {
            size: 4.0,
            ..Default::default()
        };
        assert_eq!(point_radius_for(&Material::Points(m)), 2.0);
    }

    #[test]
    fn materials_without_a_size_keep_the_fallback() {
        let other = Material::Standard(StandardMaterial::new(Color::WHITE));
        assert_eq!(line_width_for(&other), 0.02);
        assert_eq!(point_radius_for(&other), 0.05);
    }

    #[test]
    fn a_zero_width_falls_back_rather_than_vanishing() {
        // Zero would make the quad degenerate and the line invisible, which is
        // the failure the fallback exists to avoid.
        let mut m = LineBasicMaterial::new(Color::WHITE);
        m.line_width = 0.0;
        assert!(line_width_for(&Material::Line(m)) > 0.0);
        let p = PointsMaterial {
            size: 0.0,
            ..Default::default()
        };
        assert!(point_radius_for(&Material::Points(p)) > 0.0);
    }
}
