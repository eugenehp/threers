//! Closed outlines of common shapes as polygons within a chord error:
//! circles, rounded and chamfered rectangles, stadiums (ovals), regular
//! polygons, and a segment or polyline thickened into a band.

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use super::{add, circle_step, norm, scale, sub, P2};

/// The corners of a circle's quarter (or any) arc, `from` through `sweep`.
fn arc_points(center: P2, r: f64, from: f64, sweep: f64, max_error: f64, out: &mut Vec<P2>) {
    let n = (sweep.abs() / circle_step(r, max_error)).ceil().max(1.0) as usize;

    for i in 0..=n {
        let a = from + sweep * i as f64 / n as f64;
        out.push([center[0] + r * a.cos(), center[1] + r * a.sin()]);
    }
}

/// A circle, counter-clockwise from angle 0, the last point not repeated.
pub fn circle(center: P2, r: f64, max_error: f64) -> Vec<P2> {
    let n = ((TAU / circle_step(r, max_error.max(f64::MIN_POSITIVE))).ceil() as usize).max(3);
    (0..n)
        .map(|i| {
            let a = TAU * i as f64 / n as f64;
            [center[0] + r * a.cos(), center[1] + r * a.sin()]
        })
        .collect()
}

/// A regular polygon of `sides` about `center` whose corners lie on a
/// circle of `r`, the first at `rotation`.
pub fn regular_polygon(center: P2, r: f64, sides: usize, rotation: f64) -> Vec<P2> {
    let sides = sides.max(3);
    (0..sides)
        .map(|i| {
            let a = rotation + TAU * i as f64 / sides as f64;
            [center[0] + r * a.cos(), center[1] + r * a.sin()]
        })
        .collect()
}

/// A rectangle from `min` to `max` with each corner rounded by `radius`
/// (clamped to half the shorter side), counter-clockwise from the bottom
/// right corner's arc (y up).
pub fn rounded_rect(min: P2, max: P2, radius: f64, max_error: f64) -> Vec<P2> {
    let (w, h) = (max[0] - min[0], max[1] - min[1]);
    let r = radius.clamp(0.0, 0.5 * w.min(h));

    if r <= 0.0 {
        return vec![min, [max[0], min[1]], max, [min[0], max[1]]];
    }

    let mut out = Vec::new();
    let e = max_error.max(f64::MIN_POSITIVE);
    arc_points(
        [max[0] - r, min[1] + r],
        r,
        -FRAC_PI_2,
        FRAC_PI_2,
        e,
        &mut out,
    );
    arc_points([max[0] - r, max[1] - r], r, 0.0, FRAC_PI_2, e, &mut out);
    arc_points(
        [min[0] + r, max[1] - r],
        r,
        FRAC_PI_2,
        FRAC_PI_2,
        e,
        &mut out,
    );
    arc_points([min[0] + r, min[1] + r], r, PI, FRAC_PI_2, e, &mut out);
    out
}

/// A rectangle from `min` to `max` with each corner cut off `chamfer`
/// along both sides (clamped to half the shorter side).
pub fn chamfered_rect(min: P2, max: P2, chamfer: f64) -> Vec<P2> {
    let c = chamfer.clamp(0.0, 0.5 * (max[0] - min[0]).min(max[1] - min[1]));

    if c <= 0.0 {
        return vec![min, [max[0], min[1]], max, [min[0], max[1]]];
    }

    vec![
        [min[0] + c, min[1]],
        [max[0] - c, min[1]],
        [max[0], min[1] + c],
        [max[0], max[1] - c],
        [max[0] - c, max[1]],
        [min[0] + c, max[1]],
        [min[0], max[1] - c],
        [min[0], min[1] + c],
    ]
}

/// The band `width` wide about the segment from `a` to `b`, its ends
/// rounded: a stadium, or a circle where the ends meet.
pub fn stadium(a: P2, b: P2, width: f64, max_error: f64) -> Vec<P2> {
    let r = 0.5 * width;
    let d = sub(b, a);
    let len = norm(d);

    if len == 0.0 {
        return circle(a, r, max_error);
    }

    let heading = d[1].atan2(d[0]);
    let mut out = Vec::new();
    let e = max_error.max(f64::MIN_POSITIVE);
    arc_points(b, r, heading - FRAC_PI_2, PI, e, &mut out);
    arc_points(a, r, heading + FRAC_PI_2, PI, e, &mut out);
    out
}

/// A polyline thickened to `width` with square-cut ends and mitred joins
/// (each join's mitre capped at `miter_limit` widths, bevelled past it),
/// as one closed outline; it may cross itself where the line turns
/// sharply back.
pub fn thicken(points: &[P2], width: f64, miter_limit: f64) -> Vec<P2> {
    let pts: Vec<P2> = points.iter().copied().fold(Vec::new(), |mut v, p| {
        if v.last() != Some(&p) {
            v.push(p);
        }

        v
    });

    if pts.len() < 2 {
        return Vec::new();
    }

    let half = 0.5 * width;
    let normal = |a: P2, b: P2| {
        let d = sub(b, a);
        let l = norm(d);
        [-d[1] / l, d[0] / l]
    };

    // one side of the band, offset by `sign` half widths
    let side = |sign: f64| {
        let mut out = vec![add(pts[0], scale(normal(pts[0], pts[1]), sign * half))];

        for i in 1..pts.len() - 1 {
            let (n0, n1) = (normal(pts[i - 1], pts[i]), normal(pts[i], pts[i + 1]));
            let m = add(n0, n1);
            let ml = norm(m);
            let cos_half = ml / 2.0;

            if ml > 1e-12 && 1.0 / cos_half <= miter_limit {
                out.push(add(pts[i], scale(m, sign * half / (cos_half * ml))));
            } else {
                out.push(add(pts[i], scale(n0, sign * half)));
                out.push(add(pts[i], scale(n1, sign * half)));
            }
        }

        let n = pts.len();
        out.push(add(
            pts[n - 1],
            scale(normal(pts[n - 2], pts[n - 1]), sign * half),
        ));
        out
    };

    let mut out = side(1.0);
    out.extend(side(-1.0).into_iter().rev());
    out
}

#[cfg(test)]
mod tests {
    use super::super::polyline::signed_area;
    use super::*;

    #[test]
    fn areas() {
        // a circle's area within the error band
        let r = 10.0;
        let c = circle([0.0, 0.0], r, 0.01);
        let a = signed_area(&c);
        assert!(a < PI * r * r && a > PI * (r - 0.01) * (r - 0.01));

        // a rounded rectangle: the box less four corners' worth of (1 - π/4) r²
        let rr = rounded_rect([0.0, 0.0], [20.0, 10.0], 2.0, 0.001);
        let expect = 200.0 - 4.0 * (1.0 - PI / 4.0) * 4.0;
        assert!((signed_area(&rr) - expect).abs() < 0.01);

        assert_eq!(
            signed_area(&chamfered_rect([0.0, 0.0], [10.0, 10.0], 1.0)),
            98.0
        );

        // a stadium: the band plus the two half circles
        let s = stadium([0.0, 0.0], [10.0, 0.0], 2.0, 0.0001);
        assert!((signed_area(&s) - (20.0 + PI)).abs() < 0.01);
    }

    #[test]
    fn thickened() {
        let band = thicken(&[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]], 2.0, 4.0);
        assert!((signed_area(&band).abs() - 40.0).abs() < 1e-9);
    }
}
