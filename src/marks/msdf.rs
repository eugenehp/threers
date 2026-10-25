//! Multi-channel signed distance fields from flattened glyph outlines.
//!
//! A single distance field rounds every corner: a texel near a corner holds
//! one distance, to the nearest point, and bilinear filtering between such
//! texels cuts the corner off. Chlumsky's multi-channel field keeps it by
//! giving the edges either side of a corner different colours — channel
//! pairs — and storing per channel the distance to the nearest edge of that
//! colour, extended past its ends as a straight line. The median of the three
//! channels then meets the outline exactly at the corner, and filtering the
//! channels separately keeps it sharp.
//!
//! The fourth channel holds the ordinary signed distance. The halo reads it,
//! since it is valid far from the outline where the median is not, and it
//! overrides the three where their median disagrees with it about which side
//! of the outline a texel is on — the artifacts a field built without
//! msdfgen's full error correction would otherwise show as specks.

/// A field over a glyph's box plus `spread` pixels: `(r, g, b)` multi-channel,
/// `a` the true distance, each 0.5 on the outline, rising inside and reaching
/// 0 `spread` pixels out.
pub(crate) struct Field {
    pub width: u32,
    pub height: u32,
    pub texels: Vec<[u8; 4]>,
    /// The field's top-left texel, in pixels from the pen on the baseline.
    pub offset_x: i32,
    pub offset_y: i32,
}

type P = [f32; 2];

fn sub(a: P, b: P) -> P {
    [a[0] - b[0], a[1] - b[1]]
}
fn dot(a: P, b: P) -> f32 {
    a[0] * b[0] + a[1] * b[1]
}
fn cross(a: P, b: P) -> f32 {
    a[0] * b[1] - a[1] * b[0]
}
fn len(a: P) -> f32 {
    dot(a, a).sqrt()
}
fn unit(a: P) -> P {
    let l = len(a).max(1e-12);
    [a[0] / l, a[1] / l]
}

const RED: u8 = 1;
const GREEN: u8 = 2;
const BLUE: u8 = 4;
const WHITE: u8 = RED | GREEN | BLUE;
const CYAN: u8 = GREEN | BLUE;
const MAGENTA: u8 = RED | BLUE;
const YELLOW: u8 = RED | GREEN;

/// A run of segments between two corners, and the channels it writes.
struct Edge {
    points: Vec<P>,
    mask: u8,
}

/// Whether the turn from direction `a` to `b` is a corner: more than about
/// 8° (msdfgen's threshold of 3 radians on the sine), or doubling back.
fn is_corner(a: P, b: P) -> bool {
    dot(a, b) <= 0.0 || cross(a, b).abs() > 3f32.sin()
}

/// The first and last non-degenerate directions of a polyline.
fn end_directions(points: &[P]) -> Option<(P, P)> {
    let first = points
        .windows(2)
        .map(|w| sub(w[1], w[0]))
        .find(|d| len(*d) > 1e-6)?;
    let last = points
        .windows(2)
        .rev()
        .map(|w| sub(w[1], w[0]))
        .find(|d| len(*d) > 1e-6)?;
    Some((unit(first), unit(last)))
}

/// Split one contour (a list of curves) into coloured edges.
fn color_contour(curves: &[Vec<P>]) -> Vec<Edge> {
    let curves: Vec<&Vec<P>> = curves
        .iter()
        .filter(|c| end_directions(c).is_some())
        .collect();
    let n = curves.len();
    if n == 0 {
        return Vec::new();
    }
    let dirs: Vec<(P, P)> = curves.iter().map(|c| end_directions(c).unwrap()).collect();
    // Curve k starts at a corner when the previous one's last direction
    // turns sharply into its first.
    let corners: Vec<usize> = (0..n)
        .filter(|&k| is_corner(dirs[(k + n - 1) % n].1, dirs[k].0))
        .collect();
    let join = |range: &mut dyn Iterator<Item = usize>| -> Vec<P> {
        let mut pts: Vec<P> = Vec::new();
        for k in range {
            let c = curves[k];
            let skip = usize::from(!pts.is_empty());
            pts.extend_from_slice(&c[skip..]);
        }
        pts
    };
    if corners.is_empty() {
        return vec![Edge {
            points: join(&mut (0..n)),
            mask: WHITE,
        }];
    }
    let mut edges: Vec<Vec<P>> = Vec::new();
    for (i, &start) in corners.iter().enumerate() {
        let end = corners[(i + 1) % corners.len()];
        let count = (end + n - start - 1) % n + 1;
        edges.push(join(&mut (0..count).map(|j| (start + j) % n)));
    }
    if edges.len() == 1 {
        // One corner, a teardrop: split its single edge in three so the
        // corner still has two colours either side.
        let pts = edges.pop().unwrap();
        if pts.len() < 4 {
            return vec![Edge {
                points: pts,
                mask: WHITE,
            }];
        }
        let a = pts.len() / 3;
        let b = 2 * pts.len() / 3;
        edges = vec![pts[..=a].to_vec(), pts[a..=b].to_vec(), pts[b..].to_vec()];
    }
    let cycle = [CYAN, MAGENTA, YELLOW];
    let m = edges.len();
    let mut out: Vec<Edge> = Vec::with_capacity(m);
    for (i, points) in edges.into_iter().enumerate() {
        let mut mask = cycle[i % 3];
        // The last edge meets the first at a corner too: they must differ.
        if i == m - 1 && i > 0 && mask == out[0].mask {
            mask = cycle[(i + 1) % 3];
        }
        out.push(Edge { points, mask });
    }
    out
}

/// Twice the signed area of a closed polyline.
fn area(points: &[P]) -> f32 {
    points
        .iter()
        .zip(points.iter().cycle().skip(1))
        .map(|(a, b)| cross(*a, *b))
        .sum()
}

/// Non-zero winding of the closed polylines about `p`.
fn inside(contours: &[Vec<P>], p: P) -> bool {
    let mut winding = 0i32;
    for c in contours {
        for (a, b) in c.iter().zip(c.iter().cycle().skip(1)) {
            if (a[1] <= p[1]) != (b[1] <= p[1]) {
                let x = a[0] + (p[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]);
                if x > p[0] {
                    winding += if b[1] > a[1] { 1 } else { -1 };
                }
            }
        }
    }
    winding != 0
}

fn median(a: f32, b: f32, c: f32) -> f32 {
    a.min(b).max(a.max(b).min(c))
}

/// The field of `contours` — per contour, per curve, points in pixels, y
/// down — reaching `spread` pixels either side of the outline. `None` when
/// there is no outline.
///
/// `grow` moves the outline out by that many pixels — a synthetic bold —
/// which for a distance field is exact: the field of the grown shape is the
/// field less the growth, corners rounded as growing rounds them. The box
/// widens by as much, so the field still reaches `spread` past the new edge.
pub(crate) fn msdf(contours: &[Vec<Vec<P>>], spread: i32, grow: f32) -> Option<Field> {
    let grow = grow.max(0.0);
    let margin = spread + grow.ceil() as i32;
    let edges: Vec<Edge> = contours.iter().flat_map(|c| color_contour(c)).collect();
    let closed: Vec<Vec<P>> = contours
        .iter()
        .map(|c| c.iter().flatten().copied().collect::<Vec<P>>())
        .filter(|c| c.len() > 2)
        .collect();
    if edges.is_empty() || closed.is_empty() {
        return None;
    }
    // Which side is inside: the glyph's largest contour is an outer one,
    // and its fill lies on the side its signed area says.
    let flip = closed
        .iter()
        .map(|c| area(c))
        .max_by(|a, b| a.abs().total_cmp(&b.abs()))
        .map_or(1.0, f32::signum);
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for p in closed.iter().flatten() {
        x0 = x0.min(p[0]);
        y0 = y0.min(p[1]);
        x1 = x1.max(p[0]);
        y1 = y1.max(p[1]);
    }
    let ox = x0.floor() as i32 - margin;
    let oy = y0.floor() as i32 - margin;
    let w = (x1.ceil() as i32 - x0.floor() as i32 + 2 * margin).max(1) as u32;
    let h = (y1.ceil() as i32 - y0.floor() as i32 + 2 * margin).max(1) as u32;
    if w > 4096 || h > 4096 {
        return None;
    }
    let encode = |d: f32| ((0.5 - (d - grow) / (2.0 * spread as f32)).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    let mut texels = Vec::with_capacity((w * h) as usize);
    for j in 0..h {
        for i in 0..w {
            let p = [ox as f32 + i as f32 + 0.5, oy as f32 + j as f32 + 0.5];
            // Per channel: (distance, orthogonality, edge, segment, t).
            let mut best = [(f32::MAX, 0.0f32, 0usize, 0usize, 0.0f32); 3];
            let mut nearest = f32::MAX;
            for (ei, e) in edges.iter().enumerate() {
                for (si, s) in e.points.windows(2).enumerate() {
                    let ab = sub(s[1], s[0]);
                    let l2 = dot(ab, ab);
                    if l2 < 1e-12 {
                        continue;
                    }
                    let t = dot(sub(p, s[0]), ab) / l2;
                    let tc = t.clamp(0.0, 1.0);
                    let q = [s[0][0] + ab[0] * tc, s[0][1] + ab[1] * tc];
                    let d = len(sub(p, q));
                    nearest = nearest.min(d);
                    // Ties — two edges meeting at the nearest point — go to
                    // the edge the point is more nearly square to.
                    let ortho = cross(unit(ab), unit(sub(p, q))).abs();
                    for (c, b) in best.iter_mut().enumerate() {
                        if e.mask & (1 << c) == 0 {
                            continue;
                        }
                        if d < b.0 - 1e-5 || ((d - b.0).abs() <= 1e-5 && ortho > b.1) {
                            *b = (d, ortho, ei, si, t);
                        }
                    }
                }
            }
            let mut ch = [0.0f32; 3];
            for (c, &(d, _, ei, si, t)) in best.iter().enumerate() {
                if d == f32::MAX {
                    ch[c] = nearest;
                    continue;
                }
                let pts = &edges[ei].points;
                let (a, b) = (pts[si], pts[si + 1]);
                let ab = sub(b, a);
                let side = cross(ab, sub(p, a));
                // Past either end of the edge, the distance to its line
                // extended: what makes the corner sharp.
                let last = si + 2 == pts.len();
                let mut dist = d;
                if (si == 0 && t < 0.0) || (last && t > 1.0) {
                    let pseudo = (side / len(ab)).abs();
                    if pseudo <= dist {
                        dist = pseudo;
                    }
                }
                // Positive outside.
                ch[c] = if side * flip > 0.0 { -dist } else { dist };
            }
            let truth = if inside(&closed, p) {
                -nearest
            } else {
                nearest
            };
            if median(ch[0], ch[1], ch[2]).signum() != truth.signum() {
                ch = [truth; 3];
            }
            texels.push([encode(ch[0]), encode(ch[1]), encode(ch[2]), encode(truth)]);
        }
    }
    Some(Field {
        width: w,
        height: h,
        texels,
        offset_x: ox,
        offset_y: oy,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(v: u8, spread: f32) -> f32 {
        (0.5 - v as f32 / 255.0) * 2.0 * spread
    }

    /// A square, drawn as four straight curves: four corners, so four edges
    /// in alternating colours, and the median is the distance to the square —
    /// with a sharp corner where the true distance is round.
    #[test]
    fn a_square_keeps_its_corner() {
        let sq = vec![vec![
            vec![[0.0, 0.0], [20.0, 0.0]],
            vec![[20.0, 0.0], [20.0, 20.0]],
            vec![[20.0, 20.0], [0.0, 20.0]],
            vec![[0.0, 20.0], [0.0, 0.0]],
        ]];
        let edges = color_contour(&sq[0]);
        assert_eq!(edges.len(), 4);
        for k in 0..4 {
            let (a, b) = (edges[k].mask, edges[(k + 1) % 4].mask);
            assert_ne!(a, b, "neighbours {k} differ");
            assert_ne!(a & b, 0, "neighbours {k} share a channel");
        }
        let spread = 4;
        let f = msdf(&sq, spread, 0.0).unwrap();
        let at = |x: f32, y: f32| {
            let i = (x - f.offset_x as f32 - 0.5) as usize;
            let j = (y - f.offset_y as f32 - 0.5) as usize;
            f.texels[j * f.width as usize + i]
        };
        // Diagonally out from the corner (20, 20), 1.5 px each way.
        let t = at(21.5, 21.5);
        let m = median(
            decode(t[0], spread as f32),
            decode(t[1], spread as f32),
            decode(t[2], spread as f32),
        );
        let truth = decode(t[3], spread as f32);
        assert!(
            (truth - 1.5 * 2f32.sqrt()).abs() < 0.05,
            "true distance {truth}"
        );
        // The median: the distance to the square's sides extended — 1.5.
        assert!((m - 1.5).abs() < 0.05, "median {m}");
        // Inside, the same either way.
        let t = at(10.5, 10.5);
        assert!(t[3] > 200 && median(t[0] as f32, t[1] as f32, t[2] as f32) > 200.0);
    }

    /// Grown by `g`, every texel reads its distance less `g`, the median and
    /// the true distance both, and the box takes the growth on each side.
    #[test]
    fn growing_moves_the_field_by_the_growth() {
        let sq = vec![vec![
            vec![[0.0, 0.0], [20.0, 0.0]],
            vec![[20.0, 0.0], [20.0, 20.0]],
            vec![[20.0, 20.0], [0.0, 20.0]],
            vec![[0.0, 20.0], [0.0, 0.0]],
        ]];
        let spread = 4;
        let (plain, grown) = (msdf(&sq, spread, 0.0).unwrap(), msdf(&sq, spread, 1.5).unwrap());
        assert_eq!(grown.width, plain.width + 4);
        assert_eq!(grown.offset_x, plain.offset_x - 2);
        let at = |f: &Field, x: f32, y: f32| {
            let i = (x - f.offset_x as f32 - 0.5) as usize;
            let j = (y - f.offset_y as f32 - 0.5) as usize;
            f.texels[j * f.width as usize + i]
        };
        let s = spread as f32;
        for (x, y) in [(21.5, 10.5), (18.5, 10.5), (22.5, 22.5)] {
            let (p, g) = (at(&plain, x, y), at(&grown, x, y));
            let m = |t: [u8; 4]| median(decode(t[0], s), decode(t[1], s), decode(t[2], s));
            assert!((decode(g[3], s) - (decode(p[3], s) - 1.5)).abs() < 0.05, "true at {x},{y}");
            assert!((m(g) - (m(p) - 1.5)).abs() < 0.05, "median at {x},{y}");
        }
        // A pixel past the old edge is inside the grown one.
        assert!(at(&grown, 20.5, 10.5)[3] > 128 && at(&plain, 20.5, 10.5)[3] < 128);
    }

    /// A ring (an outer contour and a hole wound the other way): inside
    /// the band is inside, the hole outside.
    #[test]
    fn a_hole_is_outside() {
        let ring = |r: f32, flip: bool| {
            let mut pts: Vec<P> = (0..=32)
                .map(|k| {
                    let a = k as f32 / 32.0 * std::f32::consts::TAU;
                    [20.0 + r * a.cos(), 20.0 + r * a.sin()]
                })
                .collect();
            if flip {
                pts.reverse();
            }
            vec![pts]
        };
        let f = msdf(&[ring(16.0, false), ring(8.0, true)], 4, 0.0).unwrap();
        let at = |x: f32, y: f32| {
            let i = (x - f.offset_x as f32 - 0.5) as usize;
            let j = (y - f.offset_y as f32 - 0.5) as usize;
            f.texels[j * f.width as usize + i]
        };
        let m = |t: [u8; 4]| median(t[0] as f32, t[1] as f32, t[2] as f32);
        let band = at(32.5, 20.5);
        let hole = at(20.5, 20.5);
        assert!(m(band) > 128.0 && band[3] > 128, "band {band:?}");
        assert!(m(hole) < 128.0 && hole[3] < 128, "hole {hole:?}");
    }
}
