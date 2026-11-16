use crate::math::Vector2;

/// Triangulates a polygon given as its outer ring followed by any holes,
/// `hole_indices` saying where in `points` each hole starts (mapbox earcut's
/// convention). Either ring orientation is accepted. Returns indices into
/// `points`, three per triangle, each triangle counter-clockwise.
///
/// The work is done by [`shapes::earcut`](super::shapes::earcut), the full
/// mapbox earcut port, which copes with holes, collinear and repeated points
/// and self-touching outlines, in about O(n log n).
pub fn earcut(points: &[Vector2], hole_indices: &[usize]) -> Vec<u32> {
    if points.len() < 3 {
        return Vec::new();
    }
    let data: Vec<f64> = points
        .iter()
        .flat_map(|p| [p.x as f64, p.y as f64])
        .collect();
    let mut tris = super::shapes::earcut::triangulate(&data, hole_indices, 2);
    for t in tris.chunks_mut(3) {
        let [a, b, c] = [t[0], t[1], t[2]].map(|i| points[i as usize]);
        if (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x) < 0.0 {
            t.swap(1, 2);
        }
    }
    tris
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_into_two_triangles() {
        let pts = [
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0),
            Vector2::new(0.0, 1.0),
        ];
        let tris = earcut(&pts, &[]);
        assert_eq!(tris.len(), 6, "got {:?}", tris);
    }

    #[test]
    fn concave_quad() {
        // Arrow-shaped concave polygon — fan triangulation would fail; earcut
        // should still produce 3 triangles.
        let pts = [
            Vector2::new(0.0, 0.0),
            Vector2::new(2.0, 0.0),
            Vector2::new(1.0, 0.5),
            Vector2::new(2.0, 1.0),
            Vector2::new(0.0, 1.0),
        ];
        let tris = earcut(&pts, &[]);
        assert_eq!(tris.len(), 9);
    }

    fn covered(pts: &[Vector2], tris: &[u32]) -> f32 {
        tris.chunks(3)
            .map(|t| {
                let [a, b, c] = [t[0], t[1], t[2]].map(|i| pts[i as usize]);
                let twice = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
                assert!(twice >= 0.0, "counter-clockwise");
                twice / 2.0
            })
            .sum()
    }

    #[test]
    fn holes_are_left_open() {
        // A 10 × 6 plate with two 2 × 2 windows, outer ring clockwise this
        // time, holes either way.
        let mut pts = vec![
            Vector2::new(0.0, 0.0),
            Vector2::new(0.0, 6.0),
            Vector2::new(10.0, 6.0),
            Vector2::new(10.0, 0.0),
        ];
        let square = |x: f32, y: f32| {
            [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)].map(|(dx, dy)| Vector2::new(x + dx, y + dy))
        };
        pts.extend(square(2.0, 2.0));
        pts.extend(square(6.0, 2.0).into_iter().rev());
        let tris = earcut(&pts, &[4, 8]);
        assert!((covered(&pts, &tris) - (60.0 - 8.0)).abs() < 1e-4);
    }

    #[test]
    fn collinear_and_repeated_points() {
        // A square with points along its edges and one repeated: the old
        // ear clipper gave up on these and left the polygon part-covered.
        let mut pts = vec![];
        for i in 0..10 {
            pts.push(Vector2::new(i as f32, 0.0));
        }
        pts.push(Vector2::new(10.0, 0.0));
        pts.push(Vector2::new(10.0, 0.0));
        for i in 0..=10 {
            pts.push(Vector2::new(10.0 - i as f32, 10.0));
        }
        let tris = earcut(&pts, &[]);
        assert!((covered(&pts, &tris) - 100.0).abs() < 1e-3);
    }

    #[test]
    fn many_points_quickly() {
        // A 2000-point circle with a 1000-point hole: the old O(n³) clipper
        // took seconds here.
        let ring = |n: usize, r: f32| (0..n).map(move |i| {
            let a = i as f32 / n as f32 * std::f32::consts::TAU;
            Vector2::new(r * a.cos(), r * a.sin())
        });
        let mut pts: Vec<Vector2> = ring(2000, 10.0).collect();
        pts.extend(ring(1000, 5.0));
        let start = std::time::Instant::now();
        let tris = earcut(&pts, &[2000]);
        assert!(start.elapsed().as_millis() < 500);
        let want = std::f32::consts::PI * (100.0 - 25.0);
        assert!((covered(&pts, &tris) - want).abs() / want < 1e-3);
    }
}
