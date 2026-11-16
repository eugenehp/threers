//! Hatching: parallel lines across the inside of polygons with holes,
//! found by sweeping each line across the edges (even-odd rule).

use super::P2;

/// The pieces of lines at `angle` (radians from +x), `spacing` apart,
/// that lie inside `polygons` (each a closed ring; rings inside rings are
/// holes, by the even-odd rule). The lines are placed on a grid through
/// the origin, so neighbouring areas hatched alike line up.
pub fn hatch_lines(polygons: &[Vec<P2>], angle: f64, spacing: f64) -> Vec<[P2; 2]> {
    if spacing <= 0.0 || polygons.iter().all(|p| p.len() < 3) {
        return Vec::new();
    }

    // work in a frame where the lines run along x: rotate by -angle
    let (s, c) = angle.sin_cos();
    let to_local = |p: P2| [p[0] * c + p[1] * s, -p[0] * s + p[1] * c];
    let to_world = |p: P2| [p[0] * c - p[1] * s, p[0] * s + p[1] * c];

    let edges: Vec<(P2, P2)> = polygons
        .iter()
        .filter(|p| p.len() >= 3)
        .flat_map(|ring| (0..ring.len()).map(move |i| (ring[i], ring[(i + 1) % ring.len()])))
        .map(|(a, b)| (to_local(a), to_local(b)))
        .collect();

    let (lo, hi) = edges
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), (a, b)| {
            (lo.min(a[1]).min(b[1]), hi.max(a[1]).max(b[1]))
        });
    let mut out = Vec::new();
    let mut k = (lo / spacing).ceil();

    while k * spacing <= hi {
        let y = k * spacing;
        let mut xs: Vec<f64> = edges
            .iter()
            // half-open in y, so a vertex on the line counts once
            .filter(|(a, b)| (a[1] > y) != (b[1] > y))
            .map(|(a, b)| a[0] + (y - a[1]) / (b[1] - a[1]) * (b[0] - a[0]))
            .collect();

        xs.sort_by(f64::total_cmp);

        for pair in xs.chunks_exact(2) {
            if pair[1] > pair[0] {
                out.push([to_world([pair[0], y]), to_world([pair[1], y])]);
            }
        }

        k += 1.0;
    }

    out
}

/// Cross-hatching: [`hatch_lines`] at `angle` and at a right angle to it.
pub fn cross_hatch_lines(polygons: &[Vec<P2>], angle: f64, spacing: f64) -> Vec<[P2; 2]> {
    let mut out = hatch_lines(polygons, angle, spacing);
    out.extend(hatch_lines(
        polygons,
        angle + std::f64::consts::FRAC_PI_2,
        spacing,
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_with_hole() {
        let outer = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let hole = vec![[4.0, 4.0], [6.0, 4.0], [6.0, 6.0], [4.0, 6.0]];
        let lines = hatch_lines(&[outer.clone(), hole], 0.0, 1.0);

        // lines at y = 0 .. 9 (the top edge, half-open, takes none); the
        // two through the hole, at y = 4 and 5, lose its width
        let total: f64 = lines.iter().map(|l| (l[1][0] - l[0][0]).abs()).sum();
        assert!((total - (10.0 * 10.0 - 2.0 * 2.0)).abs() < 1e-9, "{total}");

        // at 45 degrees every piece is inside the square
        for l in hatch_lines(&[outer], std::f64::consts::FRAC_PI_4, 1.0) {
            for p in l {
                assert!(p[0] > -1e-9 && p[0] < 10.0 + 1e-9 && p[1] > -1e-9 && p[1] < 10.0 + 1e-9);
            }
        }
    }
}
