//! Polygon tables for the grid generators.
//!
//! Most built-in geometries are quad grids that three.js — and these ports of
//! it — emit as two consecutive triangles per cell. Recording that grouping
//! ([`BufferGeometry::polygons`]) lets a wireframe draw the cells instead of
//! every diagonal, the way a DCC tool shows the same mesh. The triangle order
//! itself is untouched, so the buffers still match three.js exactly.

use crate::core::BufferGeometry;

/// A polygon table built cell by cell as a generator emits its triangles.
#[derive(Debug, Default)]
pub(crate) struct CellPolygons(Vec<u32>);

impl CellPolygons {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// A cell just emitted as two consecutive triangles over `corners`.
    ///
    /// Recorded as one quad unless two of its corners sit at the same point —
    /// a sphere or capsule pole, a cone tip, a lathe profile point on the axis,
    /// a ring with no hole. Such a cell is a triangle with a zero-length edge:
    /// it has no four-corner outline for
    /// [`polygon_loop`](BufferGeometry::polygon_loop) to recover, so its two
    /// triangles stay separate polygons. The wireframe looks the same either
    /// way; only the topology claim differs.
    pub(crate) fn quad(&mut self, positions: &[f32], corners: [u32; 4]) {
        let at = |v: u32| {
            let i = v as usize * 3;
            [positions[i], positions[i + 1], positions[i + 2]]
        };
        let p = corners.map(at);
        // Corners a rounding error apart are one point: three.js puts a
        // pole at `r·cos(3π/2)`, ~1e-17 off the axis, and these buffers are
        // three.js's to the bit. Measured against the cell's own size.
        let d2 = |a: [f32; 3], b: [f32; 3]| {
            let (x, y, z) = ((a[0] - b[0]) as f64, (a[1] - b[1]) as f64, (a[2] - b[2]) as f64);
            x * x + y * y + z * z
        };
        let extent = (0..4).flat_map(|i| (i + 1..4).map(move |j| (i, j))).map(|(i, j)| d2(p[i], p[j])).fold(0.0, f64::max);
        let collapsed = (0..4).any(|i| (i + 1..4).any(|j| d2(p[i], p[j]) <= extent * 1e-12));
        if collapsed {
            self.0.extend_from_slice(&[1, 1]);
        } else {
            self.0.push(2);
        }
    }

    /// `n` triangles that are each their own polygon: a pole row, or a
    /// centre-fanned cap (a fan around an interior point is not one simple
    /// polygon in [`polygon_loop`](BufferGeometry::polygon_loop)'s sense).
    pub(crate) fn triangles(&mut self, n: usize) {
        self.0.extend(std::iter::repeat_n(1, n));
    }

    /// Attach the table. Call after the geometry's final `set_index`.
    pub(crate) fn apply(self, geometry: &mut BufferGeometry) {
        debug_assert_eq!(
            self.0.iter().map(|&n| n as usize).sum::<usize>(),
            geometry.triangle_count(),
            "polygon table out of step with the emitted triangles"
        );
        geometry.set_polygons(self.0);
    }
}

#[cfg(test)]
mod tests {
    use crate::core::BufferGeometry;
    use crate::geometries::*;
    use crate::math::{Vector2, Vector3};
    use std::f32::consts::PI;

    /// The table is attached, has `quads` two-triangle polygons, every one of
    /// them comes back from `polygon_loop` as four distinct corners, and the
    /// wireframe drops exactly one diagonal per quad.
    fn assert_quads(g: &BufferGeometry, quads: usize) {
        let counts = g.polygon_counts().expect("polygon table attached");
        assert!(counts.iter().all(|&n| n == 1 || n == 2), "{counts:?}");
        assert_eq!(counts.iter().filter(|&&n| n == 2).count(), quads);
        for range in g.polygon_ranges().filter(|r| r.len() == 2) {
            let corners = g.polygon_loop(range.clone()).expect("quad outline");
            assert_eq!(corners.len(), 4, "triangles {range:?}");
        }
    }

    #[test]
    fn plane() {
        assert_quads(&PlaneGeometry::new(2.0, 1.0), 1);
        let g = PlaneGeometry::with_segments(2.0, 1.0, 4, 3);
        assert_quads(&g, 12);
        // Outline only: a 4x3 grid has 4*4 + 5*3 = 31 edges, no diagonals.
        assert_eq!(g.polygon_edges().len(), 31);
    }

    #[test]
    fn box_faces() {
        let g = BoxGeometry::new(1.0, 2.0, 3.0);
        assert_quads(&g, 6);
        assert_eq!(g.polygon_counts().unwrap().len(), 6);
        // A cube's 12 edges; faces do not share vertices but do share points.
        assert_eq!(g.polygon_edges().len(), 12);
    }

    #[test]
    fn cylinder_sides_are_quads_caps_are_triangles() {
        let g = CylinderGeometry::new(1.0, 1.5, 2.0, 8, 3, false, 0.0, PI * 2.0);
        assert_quads(&g, 8 * 3);
        // Two fanned caps of 8 triangles each, left as triangles.
        assert_eq!(g.polygon_counts().unwrap().len(), 8 * 3 + 2 * 8);
        let open = CylinderGeometry::new(1.0, 1.0, 2.0, 6, 1, true, 0.0, PI);
        assert_quads(&open, 6);
    }

    #[test]
    fn cone_tip_row_is_triangles() {
        // radius_top = 0: the top row collapses onto the tip.
        let g = ConeGeometry::new(1.0, 2.0, 8, 2, false, 0.0, PI * 2.0);
        assert_quads(&g, 8);
        let counts = g.polygon_counts().unwrap();
        assert_eq!(
            counts.iter().map(|&n| n as usize).sum::<usize>(),
            g.triangle_count()
        );
    }

    #[test]
    fn sphere_quads_except_poles() {
        let g = SphereGeometry::new(1.0, 8, 6);
        // Rows 1..h-1 are quads; the two pole rows emit one triangle a cell.
        assert_quads(&g, 8 * 4);
        assert_eq!(g.polygon_counts().unwrap().len(), 8 * 4 + 2 * 8);
        // A band away from the poles: every cell is a quad.
        let band = SphereGeometry::with_range(1.0, 8, 4, 0.0, PI * 2.0, 0.5, 2.0);
        assert_quads(&band, 8 * 4);
    }

    #[test]
    fn torus() {
        assert_quads(&TorusGeometry::new(1.0, 0.3, 6, 10, PI * 2.0), 60);
    }

    #[test]
    fn torus_knot() {
        assert_quads(&TorusKnotGeometry::new(1.0, 0.3, 32, 6, 2, 3), 32 * 6);
    }

    #[test]
    fn lathe() {
        let profile = [
            Vector2::new(0.5, -1.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(0.5, 1.0),
        ];
        assert_quads(&LatheGeometry::new(&profile, 12, 0.0, PI * 2.0), 24);
        // A profile point on the axis collapses that row to triangles.
        let closed = [Vector2::new(0.0, -1.0), Vector2::new(1.0, 0.0)];
        let g = LatheGeometry::new(&closed, 12, 0.0, PI * 2.0);
        assert_quads(&g, 0);
        assert_eq!(g.polygon_counts().unwrap().len(), 24);
    }

    #[test]
    fn ring() {
        assert_quads(&RingGeometry::new(0.5, 1.0, 16, 2, 0.0, PI * 2.0), 32);
        // No hole: the inner row is a fan around the centre.
        assert_quads(&RingGeometry::new(0.0, 1.0, 16, 2, 0.0, PI * 2.0), 16);
    }

    #[test]
    fn capsule_quads_except_poles() {
        let (cs, rs) = (4, 12);
        let g = CapsuleGeometry::new(0.5, 1.0, cs, rs);
        // three.js samples each cap with 2*cs divisions: a lathe profile of
        // 4*cs + 2 points, so 4*cs + 1 bands of rs cells. All quads except
        // the two bands at the poles, which collapse to triangles.
        assert_eq!(g.triangle_count(), (4 * cs + 1) * rs * 2);
        assert_quads(&g, (4 * cs - 1) * rs);
    }

    #[test]
    fn parametric() {
        let g = ParametricGeometry::new(|u, v| Vector3::new(u, v, u * v), 5, 4);
        assert_quads(&g, 20);
    }

    #[test]
    fn tube() {
        let path = crate::curves::LineCurve3::new(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 4.0),
        );
        // 8 bands of 6 cells, open or closed, every one a quad.
        for closed in [false, true] {
            let g = TubeGeometry::new(&path, 8, 0.25, 6, closed);
            assert_eq!(g.triangle_count(), 8 * 6 * 2);
            assert_quads(&g, 8 * 6);
        }
    }

    #[test]
    fn extrude_side_walls() {
        let mut shape = crate::curves::Shape::new();
        shape
            .outline
            .move_to(Vector2::new(0.0, 0.0))
            .line_to(Vector2::new(2.0, 0.0))
            .line_to(Vector2::new(2.0, 1.0))
            .line_to(Vector2::new(0.0, 1.0))
            .line_to(Vector2::new(0.0, 0.0));
        let g = ExtrudeGeometry::new(&shape, 0.5, 12);
        let counts = g.polygon_counts().expect("polygon table attached");
        let quads = counts.iter().filter(|&&n| n == 2).count();
        assert!(quads >= 4, "{counts:?}");
        assert_quads(&g, quads);
    }

    #[cfg(feature = "nurbs")]
    #[test]
    fn nurbs_grid() {
        let s = crate::nurbs::construct::cylinder([0.0; 3], [0.0, 0.0, 1.0], 1.0, 2.0);
        assert_quads(&NurbsGeometry::with_segments(&s, 16, 4), 64);
    }

    #[test]
    fn kirigami_faces() {
        // Flat panels and thickened prisms: every face is one quad.
        let miura = KirigamiExpandedMiura::new(3, 4)
            .cell(28.5, 29.5, 68.0)
            .height(50.0)
            .base(11.0, 11.0);
        for g in [miura.build(), miura.clone().thickness(2.0).build()] {
            let quads = g.triangle_count() / 2;
            assert!(quads > 0);
            assert_quads(&g, quads);
        }
    }
}
