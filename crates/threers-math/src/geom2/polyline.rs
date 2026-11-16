//! Polylines and closed polygons as point lists: length, area, points at a
//! distance, nearest points, containment and resampling.

use super::segment::Segment;
use super::{cross, dist, lerp, P2};

/// The segments of a polyline; closed, with the one back to the start.
pub fn segments(points: &[P2], closed: bool) -> impl Iterator<Item = Segment> + '_ {
    let n = points.len();
    let count = if closed && n > 2 {
        n
    } else {
        n.saturating_sub(1)
    };
    (0..count).map(move |i| Segment::new(points[i], points[(i + 1) % n]))
}

/// The length; closed, with the segment back to the start.
pub fn length(points: &[P2], closed: bool) -> f64 {
    segments(points, closed).map(|s| s.length()).sum()
}

/// The signed area of a closed polygon: positive counter-clockwise (y up).
pub fn signed_area(points: &[P2]) -> f64 {
    let n = points.len();

    if n < 3 {
        return 0.0;
    }

    0.5 * (0..n)
        .map(|i| cross(points[i], points[(i + 1) % n]))
        .sum::<f64>()
}

/// The centroid of a closed polygon's area; the mean of its points where
/// it has none.
pub fn centroid(points: &[P2]) -> P2 {
    let a = signed_area(points);

    if a.abs() < f64::EPSILON || points.len() < 3 {
        let n = points.len().max(1) as f64;
        let s = points
            .iter()
            .fold([0.0, 0.0], |acc, p| [acc[0] + p[0], acc[1] + p[1]]);
        return [s[0] / n, s[1] / n];
    }

    let n = points.len();
    let mut c = [0.0, 0.0];

    for i in 0..n {
        let (p, q) = (points[i], points[(i + 1) % n]);
        let k = cross(p, q);
        c[0] += (p[0] + q[0]) * k;
        c[1] += (p[1] + q[1]) * k;
    }

    [c[0] / (6.0 * a), c[1] / (6.0 * a)]
}

/// Whether `p` is inside a closed polygon, by the even-odd rule; a point
/// on an edge counts as inside.
pub fn contains(points: &[P2], p: P2) -> bool {
    let n = points.len();
    let mut inside = false;

    for i in 0..n {
        let (a, b) = (points[i], points[(i + 1) % n]);

        if Segment::new(a, b).distance(p) == 0.0 {
            return true;
        }

        if (a[1] > p[1]) != (b[1] > p[1]) {
            let x = a[0] + (p[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]);

            if p[0] < x {
                inside = !inside;
            }
        }
    }

    inside
}

/// The point `d` along the polyline from its start, clamped to its ends.
pub fn point_at_distance(points: &[P2], closed: bool, d: f64) -> Option<P2> {
    let first = *points.first()?;

    if d <= 0.0 {
        return Some(first);
    }

    let mut walked = 0.0;
    let mut last = first;

    for s in segments(points, closed) {
        let l = s.length();

        if walked + l >= d && l > 0.0 {
            return Some(s.at((d - walked) / l));
        }

        walked += l;
        last = s.b;
    }

    Some(last)
}

/// The point of the polyline nearest `p`, and its distance along it.
pub fn nearest(points: &[P2], closed: bool, p: P2) -> Option<(P2, f64)> {
    let mut best: Option<(P2, f64, f64)> = None;
    let mut walked = 0.0;

    if points.len() == 1 {
        return Some((points[0], 0.0));
    }

    for s in segments(points, closed) {
        let t = s.project(p);
        let q = s.at(t);
        let d = dist(p, q);

        if best.is_none_or(|b| d < b.2) {
            best = Some((q, walked + t * s.length(), d));
        }

        walked += s.length();
    }

    best.map(|(q, along, _)| (q, along))
}

/// The polyline resampled at points `spacing` apart along it, its ends
/// kept; a closed one's last point not repeating its first.
pub fn resample(points: &[P2], closed: bool, spacing: f64) -> Vec<P2> {
    let total = length(points, closed);

    if points.is_empty() || spacing <= 0.0 || total == 0.0 {
        return points.to_vec();
    }

    let n = (total / spacing).round().max(1.0) as usize;
    let count = if closed { n } else { n + 1 };
    (0..count)
        .filter_map(|i| point_at_distance(points, closed, total * i as f64 / n as f64))
        .collect()
}

/// The polyline with points closer than `tolerance` to the line through
/// their neighbours removed (Ramer–Douglas–Peucker), its ends kept.
pub fn simplify(points: &[P2], tolerance: f64) -> Vec<P2> {
    if points.len() < 3 {
        return points.to_vec();
    }

    let mut keep = vec![false; points.len()];
    keep[0] = true;
    keep[points.len() - 1] = true;
    let mut stack = vec![(0, points.len() - 1)];

    while let Some((first, last)) = stack.pop() {
        let chord = Segment::new(points[first], points[last]);
        let (mut worst, mut index) = (0.0, first);

        for (i, p) in points.iter().enumerate().take(last).skip(first + 1) {
            let d = chord.distance(*p);

            if d > worst {
                worst = d;
                index = i;
            }
        }

        if worst > tolerance {
            keep[index] = true;
            stack.push((first, index));
            stack.push((index, last));
        }
    }

    points
        .iter()
        .zip(keep)
        .filter(|(_, k)| *k)
        .map(|(p, _)| *p)
        .collect()
}

/// A point `t` of the way along each segment, for drawing dashes or
/// placing marks: the midpoints at 0.5.
pub fn along_segments(points: &[P2], closed: bool, t: f64) -> Vec<P2> {
    segments(points, closed)
        .map(|s| lerp(s.a, s.b, t))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SQUARE: [P2; 4] = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];

    #[test]
    fn measures() {
        assert_eq!(length(&SQUARE, true), 40.0);
        assert_eq!(length(&SQUARE, false), 30.0);
        assert_eq!(signed_area(&SQUARE), 100.0);
        assert_eq!(centroid(&SQUARE), [5.0, 5.0]);
        assert!(
            contains(&SQUARE, [5.0, 5.0])
                && contains(&SQUARE, [10.0, 5.0])
                && !contains(&SQUARE, [11.0, 5.0])
        );
    }

    #[test]
    fn walking() {
        assert_eq!(point_at_distance(&SQUARE, true, 15.0), Some([10.0, 5.0]));
        assert_eq!(point_at_distance(&SQUARE, true, 35.0), Some([0.0, 5.0]));
        assert_eq!(
            nearest(&SQUARE, true, [12.0, 3.0]),
            Some(([10.0, 3.0], 13.0))
        );
        assert_eq!(resample(&SQUARE, true, 5.0).len(), 8);
    }

    #[test]
    fn simplifying() {
        let line = [[0.0, 0.0], [1.0, 0.01], [2.0, 0.0], [3.0, 5.0], [4.0, 0.0]];
        assert_eq!(
            simplify(&line, 0.1),
            vec![[0.0, 0.0], [2.0, 0.0], [3.0, 5.0], [4.0, 0.0]]
        );
    }
}
