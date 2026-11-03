//! Heightfields: terrain as a grid of heights rather than a triangle soup.
//!
//! A triangle mesh stores three vertices and an index per triangle and needs a
//! BVH over all of it. A heightfield stores one number per sample and needs no
//! acceleration structure at all: the cell under a point is an array index, not
//! a search. For terrain — which is a function of `x` and `z` by definition —
//! that is both far smaller and far faster.
//!
//! A 512×512 terrain is a megabyte of heights here, against roughly 25 MB of
//! vertices, indices and tree nodes as a mesh.
//!
//! # Layout
//!
//! Samples are laid out row-major in `z` then `x`, centred on the local origin
//! and spanning `scale.x` by `scale.z`. Heights are multiplied by `scale.y`, so
//! the same sample grid can be reused at different vertical exaggerations.

use crate::math::Aabb;
use threers::math::{Ray, Vector3};

/// A grid of heights.
#[derive(Debug, Clone, PartialEq)]
pub struct HeightField {
    heights: Vec<f32>,
    /// Samples along x and z. Both at least 2.
    columns: usize,
    rows: usize,
    scale: Vector3,
    bounds: Aabb,
}

impl HeightField {
    /// Build from row-major samples, `columns * rows` of them.
    ///
    /// `scale` is the total extent: `x` and `z` are the ground footprint, `y`
    /// multiplies the sample values.
    ///
    /// `None` if the grid is smaller than 2×2 — a single row has no cells and so
    /// no surface.
    pub fn new(heights: Vec<f32>, columns: usize, rows: usize, scale: Vector3) -> Option<Self> {
        if columns < 2 || rows < 2 || heights.len() != columns * rows {
            return None;
        }
        if !scale.x.is_finite() || !scale.y.is_finite() || !scale.z.is_finite() {
            return None;
        }

        let (mut lowest, mut highest) = (f32::MAX, f32::MIN);
        for &h in &heights {
            if h.is_finite() {
                lowest = lowest.min(h);
                highest = highest.max(h);
            }
        }
        if lowest > highest {
            return None; // every sample was NaN
        }

        let half = Vector3::new(scale.x * 0.5, 0.0, scale.z * 0.5);
        let bounds = Aabb::new(
            Vector3::new(-half.x, lowest * scale.y, -half.z),
            Vector3::new(half.x, highest * scale.y, half.z),
        );

        Some(Self {
            heights,
            columns,
            rows,
            scale,
            bounds,
        })
    }

    /// A flat field, for testing and for placeholder ground.
    pub fn flat(columns: usize, rows: usize, scale: Vector3) -> Option<Self> {
        Self::new(vec![0.0; columns * rows], columns, rows, scale)
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn scale(&self) -> Vector3 {
        self.scale
    }

    pub fn heights(&self) -> &[f32] {
        &self.heights
    }

    pub fn aabb(&self) -> Aabb {
        self.bounds
    }

    /// Cells, which is one fewer than samples in each direction.
    pub fn cell_count(&self) -> usize {
        (self.columns - 1) * (self.rows - 1)
    }

    /// Equivalent triangle count, for comparing against a mesh.
    pub fn triangle_count(&self) -> usize {
        self.cell_count() * 2
    }

    /// Height of one sample, in local space.
    #[inline]
    pub fn sample(&self, column: usize, row: usize) -> f32 {
        let c = column.min(self.columns - 1);
        let r = row.min(self.rows - 1);
        self.heights[r * self.columns + c] * self.scale.y
    }

    /// Local position of a sample.
    #[inline]
    pub fn vertex(&self, column: usize, row: usize) -> Vector3 {
        let u = column as f32 / (self.columns - 1) as f32;
        let v = row as f32 / (self.rows - 1) as f32;
        Vector3::new(
            (u - 0.5) * self.scale.x,
            self.sample(column, row),
            (v - 0.5) * self.scale.z,
        )
    }

    /// Interpolated height at a local `x`, `z`, or `None` outside the field.
    ///
    /// Uses the same triangle split as the collision geometry, so a body resting
    /// on the surface and a query about it agree.
    pub fn height_at(&self, x: f32, z: f32) -> Option<f32> {
        let (u, v) = self.to_grid(x, z)?;
        let (c, r) = (u.floor() as usize, v.floor() as usize);
        let (c, r) = (c.min(self.columns - 2), r.min(self.rows - 2));
        let (fx, fz) = (u - c as f32, v - r as f32);

        let h00 = self.sample(c, r);
        let h10 = self.sample(c + 1, r);
        let h01 = self.sample(c, r + 1);
        let h11 = self.sample(c + 1, r + 1);

        // Each cell is two triangles split along the diagonal; which one the
        // point falls in decides the interpolation.
        Some(if fx + fz <= 1.0 {
            h00 + (h10 - h00) * fx + (h01 - h00) * fz
        } else {
            h11 + (h01 - h11) * (1.0 - fx) + (h10 - h11) * (1.0 - fz)
        })
    }

    /// Continuous grid coordinates for a local position, or `None` if outside.
    fn to_grid(&self, x: f32, z: f32) -> Option<(f32, f32)> {
        if self.scale.x <= 0.0 || self.scale.z <= 0.0 {
            return None;
        }
        let u = (x / self.scale.x + 0.5) * (self.columns - 1) as f32;
        let v = (z / self.scale.z + 0.5) * (self.rows - 1) as f32;
        let (max_u, max_v) = ((self.columns - 1) as f32, (self.rows - 1) as f32);
        if u < 0.0 || v < 0.0 || u > max_u || v > max_v {
            return None;
        }
        Some((u, v))
    }

    /// Visit every triangle overlapping `query`, in local space.
    ///
    /// The cell range comes straight from the query's `x` and `z` extent — no
    /// tree traversal, because a heightfield already knows where everything is.
    pub fn for_each_triangle_in_aabb(&self, query: &Aabb, mut visit: impl FnMut([Vector3; 3])) {
        if !self.bounds.intersects_box(query) {
            return;
        }
        let to_cell = |value: f32, extent: f32, samples: usize| -> f32 {
            (value / extent + 0.5) * (samples - 1) as f32
        };

        let c0 = to_cell(query.min.x, self.scale.x, self.columns).floor();
        let c1 = to_cell(query.max.x, self.scale.x, self.columns).ceil();
        let r0 = to_cell(query.min.z, self.scale.z, self.rows).floor();
        let r1 = to_cell(query.max.z, self.scale.z, self.rows).ceil();

        let c0 = (c0.max(0.0) as usize).min(self.columns - 2);
        let c1 = (c1.max(0.0) as usize).min(self.columns - 2);
        let r0 = (r0.max(0.0) as usize).min(self.rows - 2);
        let r1 = (r1.max(0.0) as usize).min(self.rows - 2);

        for r in r0..=r1 {
            for c in c0..=c1 {
                let v00 = self.vertex(c, r);
                let v10 = self.vertex(c + 1, r);
                let v01 = self.vertex(c, r + 1);
                let v11 = self.vertex(c + 1, r + 1);

                // Vertical range of this cell, to skip cells the query is
                // entirely above or below.
                let lo = v00.y.min(v10.y).min(v01.y).min(v11.y);
                let hi = v00.y.max(v10.y).max(v01.y).max(v11.y);
                if hi < query.min.y || lo > query.max.y {
                    continue;
                }

                // Wound counter-clockwise seen from above, so face normals
                // point up and the narrow phase's internal-edge correction
                // orients the right way.
                visit([v00, v01, v10]);
                visit([v10, v01, v11]);
            }
        }
    }

    /// Nearest ray hit, as `(toi, face normal)`, in local space.
    ///
    /// Marches cell by cell along the ray rather than testing every triangle,
    /// so cost depends on how far the ray travels, not on how large the terrain
    /// is.
    pub fn raycast(&self, ray: &Ray, max_toi: f32) -> Option<(f32, Vector3)> {
        // Clip to the field's bounds first, so a ray aimed elsewhere costs one
        // box test.
        let entry = if self.bounds.contains_point(ray.origin) {
            0.0
        } else {
            match ray.intersect_box(&self.bounds) {
                Some(t) if t <= max_toi => t,
                _ => return None,
            }
        };

        // Step in units of roughly one cell. A true DDA would be tighter, but
        // this keeps the arithmetic simple and never skips a cell, which is the
        // property that matters.
        let cell_x = self.scale.x / (self.columns - 1) as f32;
        let cell_z = self.scale.z / (self.rows - 1) as f32;
        let step = cell_x.min(cell_z).max(1e-4) * 0.5;

        let mut best: Option<(f32, Vector3)> = None;
        let mut t = entry;
        let mut guard = 0;
        while t <= max_toi && guard < 100_000 {
            guard += 1;
            let point = ray.at(t);
            let probe = Aabb::new(
                Vector3::new(point.x - cell_x, point.y - step, point.z - cell_z),
                Vector3::new(point.x + cell_x, point.y + step, point.z + cell_z),
            );
            self.for_each_triangle_in_aabb(&probe, |tri| {
                let triangle = threers::math::Triangle::new(tri[0], tri[1], tri[2]);
                if let Some(hit) = ray.intersect_triangle(&triangle, false) {
                    if hit <= max_toi && best.is_none_or(|(b, _)| hit < b) {
                        if let Some(n) =
                            crate::math::try_normalize((tri[1] - tri[0]).cross(tri[2] - tri[0]))
                        {
                            best = Some((hit, n));
                        }
                    }
                }
            });
            if let Some((hit, _)) = best {
                // Anything found within the span just marched is the nearest;
                // cells further along cannot beat it.
                if hit <= t + step {
                    break;
                }
            }
            t += step;
        }

        best.map(|(toi, normal)| {
            (
                toi,
                if normal.dot(ray.direction) > 0.0 {
                    -normal
                } else {
                    normal
                },
            )
        })
    }

    /// Whether a local point is below the surface.
    pub fn contains_point(&self, p: Vector3) -> bool {
        match self.height_at(p.x, p.z) {
            // Solid below the surface, down to the field's lower bound — which
            // makes a heightfield a solid, not a sheet, so a body that gets
            // under it is pushed back out rather than falling through.
            Some(surface) => p.y <= surface && p.y >= self.bounds.min.y,
            None => false,
        }
    }

    /// Build the drawable mesh, for rendering the same data.
    pub fn to_geometry(&self) -> threers::core::BufferGeometry {
        use threers::core::{BufferAttribute, BufferGeometry};
        let mut positions = Vec::with_capacity(self.columns * self.rows * 3);
        for r in 0..self.rows {
            for c in 0..self.columns {
                let v = self.vertex(c, r);
                positions.extend_from_slice(&[v.x, v.y, v.z]);
            }
        }
        let mut indices = Vec::with_capacity(self.cell_count() * 6);
        for r in 0..self.rows - 1 {
            for c in 0..self.columns - 1 {
                let i = (r * self.columns + c) as u32;
                let right = i + 1;
                let down = i + self.columns as u32;
                let diagonal = down + 1;
                indices.extend_from_slice(&[i, down, right, right, down, diagonal]);
            }
        }
        let mut geometry = BufferGeometry::new();
        geometry.set_attribute("position", BufferAttribute::new(positions, 3));
        geometry.set_index(indices);
        geometry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp() -> HeightField {
        // Height rises with x, from 0 to 1.
        let (columns, rows) = (9, 9);
        let mut heights = Vec::new();
        for _ in 0..rows {
            for c in 0..columns {
                heights.push(c as f32 / (columns - 1) as f32);
            }
        }
        HeightField::new(heights, columns, rows, Vector3::new(8.0, 4.0, 8.0)).unwrap()
    }

    #[test]
    fn degenerate_grids_are_refused() {
        assert!(HeightField::new(vec![0.0], 1, 1, Vector3::ONE).is_none());
        assert!(HeightField::new(vec![0.0; 3], 2, 2, Vector3::ONE).is_none());
        assert!(HeightField::flat(2, 2, Vector3::ONE).is_some());
        assert!(HeightField::new(vec![f32::NAN; 4], 2, 2, Vector3::ONE).is_none());
    }

    #[test]
    fn bounds_cover_the_whole_surface() {
        let field = ramp();
        let bounds = field.aabb();
        assert!((bounds.min.x + 4.0).abs() < 1e-4);
        assert!((bounds.max.x - 4.0).abs() < 1e-4);
        // Heights run 0..1, scaled by 4.
        assert!(bounds.min.y.abs() < 1e-4);
        assert!((bounds.max.y - 4.0).abs() < 1e-4);
    }

    #[test]
    fn height_lookup_matches_the_collision_triangles() {
        let field = ramp();
        // Sample the surface, and confirm the triangles agree with it.
        for (x, z) in [(-3.0f32, -3.0f32), (0.0, 0.0), (2.5, -1.0), (3.9, 3.9)] {
            let h = field.height_at(x, z).expect("inside the field");
            let probe = Aabb::new(
                Vector3::new(x - 0.01, h - 0.5, z - 0.01),
                Vector3::new(x + 0.01, h + 0.5, z + 0.01),
            );
            let mut covered = false;
            field.for_each_triangle_in_aabb(&probe, |tri| {
                let lo = tri[0].y.min(tri[1].y).min(tri[2].y) - 0.2;
                let hi = tri[0].y.max(tri[1].y).max(tri[2].y) + 0.2;
                if h >= lo && h <= hi {
                    covered = true;
                }
            });
            assert!(covered, "no triangle at ({x}, {z}) matched height {h}");
        }
    }

    #[test]
    fn height_rises_along_the_ramp_and_is_none_outside() {
        let field = ramp();
        let low = field.height_at(-3.9, 0.0).unwrap();
        let high = field.height_at(3.9, 0.0).unwrap();
        assert!(high > low + 3.0, "the ramp should rise: {low} to {high}");
        assert!(field.height_at(100.0, 0.0).is_none());
        assert!(field.height_at(0.0, -50.0).is_none());
    }

    #[test]
    fn a_flat_field_is_flat_everywhere() {
        let field = HeightField::flat(16, 16, Vector3::new(10.0, 1.0, 10.0)).unwrap();
        for i in 0..20 {
            let t = i as f32 / 19.0 - 0.5;
            let h = field.height_at(t * 9.9, t * 9.9).unwrap();
            assert!(h.abs() < 1e-5, "flat field returned {h}");
        }
    }

    #[test]
    fn the_aabb_query_returns_only_nearby_cells() {
        let field = ramp();
        let mut all = 0;
        field.for_each_triangle_in_aabb(&field.aabb(), |_| all += 1);
        assert_eq!(all, field.triangle_count(), "a full query should cover it");

        let mut few = 0;
        let corner = Aabb::new(
            Vector3::new(-4.0, -1.0, -4.0),
            Vector3::new(-3.0, 5.0, -3.0),
        );
        field.for_each_triangle_in_aabb(&corner, |_| few += 1);
        assert!(few > 0 && few < all / 4, "corner query returned {few} of {all}");
    }

    #[test]
    fn a_query_far_above_the_surface_finds_nothing() {
        let field = ramp();
        let sky = Aabb::new(
            Vector3::new(-4.0, 100.0, -4.0),
            Vector3::new(4.0, 101.0, 4.0),
        );
        let mut hits = 0;
        field.for_each_triangle_in_aabb(&sky, |_| hits += 1);
        assert_eq!(hits, 0);
    }

    #[test]
    fn rays_hit_the_surface_at_the_right_height() {
        let field = ramp();
        for x in [-3.0f32, -1.0, 0.0, 2.0, 3.5] {
            let expected = field.height_at(x, 0.0).unwrap();
            let ray = Ray::new(Vector3::new(x, 20.0, 0.0), Vector3::new(0.0, -1.0, 0.0));
            let (toi, normal) = field
                .raycast(&ray, 100.0)
                .unwrap_or_else(|| panic!("no hit at x = {x}"));
            let hit_height = 20.0 - toi;
            assert!(
                (hit_height - expected).abs() < 0.1,
                "at x = {x}: ray hit {hit_height}, surface is {expected}"
            );
            assert!(normal.y > 0.0, "normal should point up, got {normal:?}");
        }
    }

    #[test]
    fn a_ray_that_misses_the_field_returns_nothing() {
        let field = ramp();
        let ray = Ray::new(Vector3::new(100.0, 20.0, 0.0), Vector3::new(0.0, -1.0, 0.0));
        assert!(field.raycast(&ray, 100.0).is_none());
        // And one that stops short.
        let ray = Ray::new(Vector3::new(0.0, 20.0, 0.0), Vector3::new(0.0, -1.0, 0.0));
        assert!(field.raycast(&ray, 1.0).is_none());
    }

    #[test]
    fn containment_treats_the_field_as_solid_below_the_surface() {
        let field = ramp();
        let surface = field.height_at(0.0, 0.0).unwrap();
        assert!(field.contains_point(Vector3::new(0.0, surface - 0.5, 0.0)));
        assert!(!field.contains_point(Vector3::new(0.0, surface + 0.5, 0.0)));
        assert!(!field.contains_point(Vector3::new(50.0, 0.0, 0.0)));
    }

    #[test]
    fn the_drawable_mesh_matches_the_collision_surface() {
        let field = ramp();
        let geometry = field.to_geometry();
        let positions = geometry.get_attribute("position").unwrap();
        assert_eq!(positions.count(), field.columns() * field.rows());
        assert_eq!(
            geometry.index.as_ref().unwrap().len() / 3,
            field.triangle_count()
        );

        // Every drawn vertex must sit on the collision surface.
        for chunk in positions.array.chunks_exact(3) {
            let h = field.height_at(chunk[0], chunk[2]).unwrap();
            assert!(
                (chunk[1] - h).abs() < 1e-3,
                "drawn vertex at y = {} but surface is {h}",
                chunk[1]
            );
        }
    }

    #[test]
    fn a_large_field_is_far_smaller_than_the_equivalent_mesh() {
        let field = HeightField::flat(256, 256, Vector3::new(100.0, 1.0, 100.0)).unwrap();
        let heightfield_bytes = field.heights().len() * 4;
        // A mesh would need three vertices and three indices per triangle.
        let mesh_bytes = field.triangle_count() * (3 * 12 + 3 * 4);
        assert!(
            heightfield_bytes * 10 < mesh_bytes,
            "heightfield {heightfield_bytes} vs mesh {mesh_bytes}"
        );
    }
}
