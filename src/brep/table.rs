//! The provenance sidecar: which surface each triangle came from.

use crate::core::BufferGeometry;
use crate::math::Matrix4;
use crate::nurbs::v3;
use crate::nurbs::V3;

use super::Surface;

/// Maps every triangle of a geometry to the analytic surface it was sampled
/// from.
///
/// This is the whole of Stage 1. It is deliberately *not* a topology: there are
/// no edges, no loops, no adjacency. A "face" here is just the set of triangles
/// carrying one surface index — enough to recover exact normals, exact UVs and
/// a re-tessellation, and not enough for booleans or fillets, which need the
/// `Loop`/`Edge` structure of Stage 3.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceTable {
    surfaces: Vec<Surface>,
    /// `tri_face[i]` indexes `surfaces` for triangle `i`.
    tri_face: Vec<u32>,
}

impl SurfaceTable {
    /// Build and validate. Returns `None` if any triangle references a surface
    /// that does not exist — a table that indexes out of bounds would panic in
    /// the renderer rather than at the point the mistake was made.
    pub fn new(surfaces: Vec<Surface>, tri_face: Vec<u32>) -> Option<Self> {
        if surfaces.is_empty() && !tri_face.is_empty() {
            return None;
        }
        if tri_face.iter().any(|&i| i as usize >= surfaces.len()) {
            return None;
        }
        Some(Self { surfaces, tri_face })
    }

    /// Every triangle comes from the same surface — spheres, tori, cylinder
    /// sides, and any single NURBS patch.
    pub fn uniform(surface: Surface, triangles: usize) -> Self {
        Self {
            surfaces: vec![surface],
            tri_face: vec![0; triangles],
        }
    }

    pub fn surfaces(&self) -> &[Surface] {
        &self.surfaces
    }

    pub fn triangle_count(&self) -> usize {
        self.tri_face.len()
    }

    pub fn surface_index_of(&self, triangle: usize) -> Option<usize> {
        self.tri_face.get(triangle).map(|&i| i as usize)
    }

    pub fn surface_of(&self, triangle: usize) -> Option<&Surface> {
        self.surface_index_of(triangle).map(|i| &self.surfaces[i])
    }

    /// Triangle indices grouped by surface, in surface order. Surfaces with no
    /// triangles are omitted — they carry no geometry to act on.
    pub fn groups(&self) -> Vec<(usize, Vec<usize>)> {
        let mut out: Vec<(usize, Vec<usize>)> = Vec::new();
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); self.surfaces.len()];
        for (tri, &s) in self.tri_face.iter().enumerate() {
            buckets[s as usize].push(tri);
        }
        for (i, b) in buckets.into_iter().enumerate() {
            if !b.is_empty() {
                out.push((i, b));
            }
        }
        out
    }

    /// Does this table describe `geometry`? Checked before every use — a table
    /// that outlived an edit to the positions it described is worse than none.
    pub fn matches(&self, geometry: &BufferGeometry) -> bool {
        triangle_count(geometry) == self.tri_face.len()
    }

    /// The worst distance from any tagged vertex to the surface it claims to
    /// come from.
    ///
    /// The self-check for provenance. A tag is a *claim*, and this is what makes
    /// it falsifiable — every generator that populates a table is tested against
    /// this, so a mis-tagged face is caught where it is created rather than
    /// three stages downstream when a boolean quietly produces the wrong seam.
    pub fn max_deviation(&self, geometry: &BufferGeometry) -> f64 {
        if !self.matches(geometry) {
            return f64::INFINITY;
        }
        let mut worst: f64 = 0.0;
        for tri in 0..self.tri_face.len() {
            let Some(surface) = self.surface_of(tri) else {
                continue;
            };
            let Some(verts) = triangle_vertices(geometry, tri) else {
                return f64::INFINITY;
            };
            for p in verts {
                worst = worst.max(surface.distance(p));
            }
        }
        worst
    }

    /// Map every surface through an affine transform.
    ///
    /// All-or-nothing: if any surface cannot be represented after the transform
    /// (a non-uniformly scaled quadric), the whole table is dropped. A partial
    /// table would leave some triangles claiming a surface and others not, and
    /// every consumer would have to carry that distinction — where "no
    /// provenance" is already a supported, safe state.
    pub fn transform(&self, m: &Matrix4) -> Option<SurfaceTable> {
        let surfaces: Option<Vec<Surface>> = self.surfaces.iter().map(|s| s.transform(m)).collect();
        Some(SurfaceTable {
            surfaces: surfaces?,
            tri_face: self.tri_face.clone(),
        })
    }

    /// Concatenate, offsetting the second table's surface indices. Used when
    /// merging geometries (a cylinder's side and its two caps).
    pub fn concat(mut self, other: &SurfaceTable) -> SurfaceTable {
        let offset = self.surfaces.len() as u32;
        self.surfaces.extend(other.surfaces.iter().cloned());
        self.tri_face
            .extend(other.tri_face.iter().map(|&i| i + offset));
        self
    }
}

/// Number of triangles in a geometry, indexed or not.
pub fn triangle_count(g: &BufferGeometry) -> usize {
    match &g.index {
        Some(idx) => idx.len() / 3,
        None => g
            .get_attribute("position")
            .map(|a| a.count() / 3)
            .unwrap_or(0),
    }
}

/// The three f64 vertices of triangle `i`.
pub fn triangle_vertices(g: &BufferGeometry, i: usize) -> Option<[V3; 3]> {
    let pos = g.get_attribute("position")?;
    if pos.item_size != 3 {
        return None;
    }
    let vertex = |vi: usize| -> Option<V3> {
        let o = vi * 3;
        Some([
            *pos.array.get(o)? as f64,
            *pos.array.get(o + 1)? as f64,
            *pos.array.get(o + 2)? as f64,
        ])
    };
    match &g.index {
        Some(idx) => {
            let base = i * 3;
            Some([
                vertex(*idx.get(base)? as usize)?,
                vertex(*idx.get(base + 1)? as usize)?,
                vertex(*idx.get(base + 2)? as usize)?,
            ])
        }
        None => Some([vertex(i * 3)?, vertex(i * 3 + 1)?, vertex(i * 3 + 2)?]),
    }
}

/// The vertex indices of triangle `i`, for callers that need to write per-vertex
/// data (normals, UVs) rather than read positions.
pub fn triangle_indices(g: &BufferGeometry, i: usize) -> Option<[usize; 3]> {
    match &g.index {
        Some(idx) => {
            let b = i * 3;
            Some([
                *idx.get(b)? as usize,
                *idx.get(b + 1)? as usize,
                *idx.get(b + 2)? as usize,
            ])
        }
        None => Some([i * 3, i * 3 + 1, i * 3 + 2]),
    }
}

/// Outward-ish reference normal of a triangle, used to decide whether a face's
/// winding agrees with its surface's canonical sense.
pub fn triangle_normal(v: &[V3; 3]) -> Option<V3> {
    v3::normalize(v3::cross(v3::sub(v[1], v[0]), v3::sub(v[2], v[0])))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BufferAttribute, BufferGeometry};

    fn two_triangle_quad() -> BufferGeometry {
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(
                vec![
                    0.0, 0.0, 0.0, //
                    1.0, 0.0, 0.0, //
                    1.0, 1.0, 0.0, //
                    0.0, 1.0, 0.0,
                ],
                3,
            ),
        );
        g.set_index(vec![0, 1, 2, 0, 2, 3]);
        g
    }

    #[test]
    fn rejects_out_of_range_surface_indices() {
        assert!(SurfaceTable::new(vec![Surface::sphere([0.0; 3], 1.0)], vec![0, 1]).is_none());
        assert!(SurfaceTable::new(vec![], vec![0]).is_none());
        assert!(SurfaceTable::new(vec![], vec![]).is_some());
    }

    #[test]
    fn counts_triangles_indexed_and_not() {
        let g = two_triangle_quad();
        assert_eq!(triangle_count(&g), 2);

        let mut soup = BufferGeometry::new();
        soup.set_attribute("position", BufferAttribute::new(vec![0.0; 9 * 3], 3));
        assert_eq!(triangle_count(&soup), 3);
    }

    #[test]
    fn max_deviation_catches_a_wrong_tag() {
        let g = two_triangle_quad();

        let right = SurfaceTable::uniform(Surface::plane([0.0; 3], [0.0, 0.0, 1.0]), 2);
        assert!(right.max_deviation(&g) < 1e-15, "correct tag reads as zero");

        let wrong = SurfaceTable::uniform(Surface::plane([0.0, 0.0, 5.0], [0.0, 0.0, 1.0]), 2);
        assert!(
            (wrong.max_deviation(&g) - 5.0).abs() < 1e-12,
            "wrong tag is measurable"
        );

        let stale = SurfaceTable::uniform(Surface::plane([0.0; 3], [0.0, 0.0, 1.0]), 7);
        assert!(
            stale.max_deviation(&g).is_infinite(),
            "a table that does not describe this geometry is not merely inaccurate"
        );
    }

    #[test]
    fn groups_partition_the_triangles() {
        let t = SurfaceTable::new(
            vec![
                Surface::plane([0.0; 3], [0.0, 0.0, 1.0]),
                Surface::sphere([0.0; 3], 1.0),
                Surface::sphere([9.0; 3], 1.0),
            ],
            vec![1, 0, 1, 1, 0],
        )
        .unwrap();
        let g = t.groups();
        // Surface 2 has no triangles and is omitted.
        assert_eq!(g.len(), 2);
        assert_eq!(g[0], (0, vec![1, 4]));
        assert_eq!(g[1], (1, vec![0, 2, 3]));
        let total: usize = g.iter().map(|(_, v)| v.len()).sum();
        assert_eq!(total, t.triangle_count());
    }

    #[test]
    fn concat_offsets_the_second_tables_indices() {
        let a = SurfaceTable::uniform(Surface::plane([0.0; 3], [0.0, 0.0, 1.0]), 2);
        let b = SurfaceTable::uniform(Surface::sphere([0.0; 3], 1.0), 3);
        let c = a.concat(&b);
        assert_eq!(c.surfaces().len(), 2);
        assert_eq!(c.triangle_count(), 5);
        assert_eq!(c.surface_index_of(1), Some(0));
        assert_eq!(c.surface_index_of(2), Some(1));
        assert_eq!(c.surface_of(4).unwrap().kind(), "sphere");
    }

    #[test]
    fn transform_is_all_or_nothing() {
        use crate::math::Vector3;
        let t = SurfaceTable::new(
            vec![
                Surface::plane([0.0; 3], [0.0, 0.0, 1.0]),
                Surface::sphere([0.0; 3], 1.0),
            ],
            vec![0, 1],
        )
        .unwrap();

        assert!(t
            .transform(&Matrix4::scale(Vector3::new(2.0, 2.0, 2.0)))
            .is_some());
        assert!(
            t.transform(&Matrix4::scale(Vector3::new(2.0, 1.0, 1.0)))
                .is_none(),
            "the plane could survive, but a half-populated table is worse than none"
        );
    }
}
