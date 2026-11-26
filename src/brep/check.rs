//! What a body's parts being sound does not say: that they make a solid that
//! does not run through itself.
//!
//! [`Body::defects`] asks each edge and face about itself. These ask them about
//! each other, and cost accordingly: they are for tests and for finding out
//! what is wrong with a result, not for every result. What to ask was read off
//! springrep's checker.

use super::body::{Body, Defect, TrimLoop};
use crate::nurbs::{v3, V3};

/// Something wrong between a body's parts. See [`Body::tangles`].
#[derive(Debug, Clone, PartialEq)]
pub enum Tangle {
    /// A part is unsound by itself.
    Defect(Defect),
    /// An edge that turns back on itself at a vertex.
    EdgeFolds { edge: usize, at: V3 },
    /// Two vertices in one place.
    VerticesCoincide { a: usize, b: usize },
    /// A face's loop whose parameters are not where its vertices are.
    LoopOffVertices { face: usize, off: f64 },
    /// A loop that crosses itself or another of its face.
    LoopsCross { face: usize },
    /// A hole that is not inside its face.
    HoleOutside { face: usize },
    /// An edge that goes through the inside of a face that is not its own.
    EdgeThroughFace { edge: usize, face: usize, at: V3 },
}

type P2 = [f64; 2];

fn cross2(a: P2, b: P2, c: P2) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

fn inside(ring: &[P2], q: P2) -> bool {
    let n = ring.len();
    let mut odd = false;
    for i in 0..n {
        let (a, b) = (ring[i], ring[(i + 1) % n]);
        if (a[1] > q[1]) != (b[1] > q[1]) {
            let x = a[0] + (q[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]);
            odd ^= x > q[0];
        }
    }
    odd
}

/// Two segments that cross, each ending clear of the other.
fn cross(a: P2, b: P2, c: P2, d: P2) -> bool {
    let size = |p: P2, q: P2| (p[0] - q[0]).hypot(p[1] - q[1]);
    let slack = 1e-9 * size(a, b) * size(c, d);
    let (p, q) = (cross2(a, b, c), cross2(a, b, d));
    let (r, s) = (cross2(c, d, a), cross2(c, d, b));
    (p > slack && q < -slack || p < -slack && q > slack)
        && (r > slack && s < -slack || r < -slack && s > slack)
}

fn to_segment(p: V3, a: V3, b: V3) -> f64 {
    let d = v3::sub(b, a);
    let l = v3::dot(d, d);
    let t = if l > 0.0 {
        (v3::dot(v3::sub(p, a), d) / l).clamp(0.0, 1.0)
    } else {
        0.0
    };
    v3::dist(p, v3::add(a, v3::scale(d, t)))
}

impl Body {
    /// Everything wrong with this body, its parts by themselves
    /// ([`Self::defects`]) and between them.
    ///
    /// A face that wraps all the way round is asked only about itself: its
    /// loops do not close in its parameters, so that they bound nothing there.
    pub fn tangles(&self, tolerance: f64) -> Vec<Tangle> {
        let mut out: Vec<Tangle> = self
            .defects(tolerance)
            .into_iter()
            .map(Tangle::Defect)
            .collect();
        // What does not resolve cannot be asked about.
        if out
            .iter()
            .any(|t| !matches!(t, Tangle::Defect(Defect::EdgeOffSurface { .. })))
        {
            return out;
        }
        self.folds(&mut out);
        self.coincident(tolerance, &mut out);
        let loops: Vec<Vec<TrimLoop>> = (0..self.faces().len())
            .map(|f| self.face_loops(f))
            .collect();
        for (f, loops) in loops.iter().enumerate() {
            self.rings(f, loops, tolerance, &mut out);
        }
        self.pierced(&loops, tolerance, &mut out);
        out
    }

    fn folds(&self, out: &mut Vec<Tangle>) {
        for (edge, e) in self.edges().iter().enumerate() {
            for w in e.vertices.windows(3) {
                let [a, b, c] = [w[0], w[1], w[2]].map(|v| self.vertices()[v]);
                let (p, q) = (v3::sub(b, a), v3::sub(c, b));
                // Sharper than square.
                if v3::dot(p, q) < 0.0 {
                    out.push(Tangle::EdgeFolds { edge, at: b });
                }
            }
        }
    }

    fn coincident(&self, tolerance: f64, out: &mut Vec<Tangle>) {
        let used: std::collections::BTreeSet<usize> = self
            .edges()
            .iter()
            .flat_map(|e| e.vertices.iter().copied())
            .collect();
        let within = tolerance * 1e-3;
        let mut sorted: Vec<usize> = used.into_iter().collect();
        let at = |v: usize| self.vertices()[v];
        sorted.sort_by(|&a, &b| at(a)[0].total_cmp(&at(b)[0]));
        for (i, &a) in sorted.iter().enumerate() {
            for &b in &sorted[i + 1..] {
                if at(b)[0] - at(a)[0] > within {
                    break;
                }
                if v3::dist(at(a), at(b)) <= within {
                    out.push(Tangle::VerticesCoincide {
                        a: a.min(b),
                        b: a.max(b),
                    });
                }
            }
        }
    }

    fn rings(&self, face: usize, loops: &[TrimLoop], tolerance: f64, out: &mut Vec<Tangle>) {
        let f = &self.faces()[face];
        let surface = &self.surfaces()[f.surface];
        let off = loops
            .iter()
            .flat_map(|l| l.vertices.iter().zip(&l.uv))
            .map(|(&v, q)| v3::dist(surface.point(q[0], q[1]), self.vertices()[v]))
            .fold(0.0, f64::max);
        if off > tolerance * 4.0 {
            out.push(Tangle::LoopOffVertices { face, off });
        }
        if f.u_wraps || f.v_wraps {
            return;
        }
        let mut segments: Vec<(P2, P2)> = loops
            .iter()
            .flat_map(|l| (0..l.uv.len()).map(|i| (l.uv[i], l.uv[(i + 1) % l.uv.len()])))
            .collect();
        segments.sort_by(|s, t| s.0[0].min(s.1[0]).total_cmp(&t.0[0].min(t.1[0])));
        let crossed = segments.iter().enumerate().any(|(i, s)| {
            let end = s.0[0].max(s.1[0]);
            segments[i + 1..]
                .iter()
                .take_while(|t| t.0[0].min(t.1[0]) <= end)
                .any(|t| cross(s.0, s.1, t.0, t.1))
        });
        if crossed {
            out.push(Tangle::LoopsCross { face });
        }
        let Some(outer) = loops
            .iter()
            .max_by(|a, b| a.area.abs().total_cmp(&b.area.abs()))
        else {
            return;
        };
        for l in loops.iter().filter(|l| !std::ptr::eq(*l, outer)) {
            // Most of its points: one may be where the two touch.
            let held = l.uv.iter().filter(|q| inside(&outer.uv, **q)).count();
            if 2 * held < l.uv.len() {
                out.push(Tangle::HoleOutside { face });
            }
        }
    }

    fn pierced(&self, loops: &[Vec<TrimLoop>], tolerance: f64, out: &mut Vec<Tangle>) {
        let clear = tolerance * 10.0;
        let bounds = |points: &mut dyn Iterator<Item = V3>| {
            let mut b = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
            for p in points {
                for k in 0..3 {
                    b.0[k] = b.0[k].min(p[k]);
                    b.1[k] = b.1[k].max(p[k]);
                }
            }
            b
        };
        for (face, f) in self.faces().iter().enumerate() {
            if f.u_wraps || f.v_wraps || loops[face].is_empty() {
                continue;
            }
            let surface = &self.surfaces()[f.surface];
            let rim: Vec<(V3, V3)> = loops[face]
                .iter()
                .flat_map(|l| {
                    let n = l.vertices.len();
                    (0..n).map(move |i| {
                        (
                            self.vertices()[l.vertices[i]],
                            self.vertices()[l.vertices[(i + 1) % n]],
                        )
                    })
                })
                .collect();
            // Its box, with what it may bulge past its rim.
            let (lo, hi) = bounds(&mut rim.iter().map(|s| s.0));
            let pad = match surface {
                super::Surface::Plane { .. } => clear,
                _ => clear + v3::dist(lo, hi),
            };
            for (edge, e) in self.edges().iter().enumerate() {
                if f.edges.contains(&edge) {
                    continue;
                }
                let points: Vec<V3> = e.vertices.iter().map(|&v| self.vertices()[v]).collect();
                let (a, b) = bounds(&mut points.iter().copied());
                if (0..3).any(|k| a[k] > hi[k] + pad || b[k] < lo[k] - pad) {
                    continue;
                }
                // Well to one side, and then well to the other.
                let mut last: Option<(usize, f64)> = None;
                for (i, p) in points.iter().enumerate() {
                    let d = surface.signed_distance(*p);
                    if d.abs() <= clear {
                        continue;
                    }
                    if let Some((j, was)) = last {
                        if was * d < 0.0 {
                            let (mut x, mut y) = (points[j], points[i]);
                            for k in j..i {
                                // The step it goes over in.
                                let (p, q) = (points[k], points[k + 1]);
                                let (dp, dq) =
                                    (surface.signed_distance(p), surface.signed_distance(q));
                                if dp * was > 0.0 && dq * was <= 0.0 {
                                    (x, y) = (p, q);
                                }
                            }
                            let (dx, dy) = (surface.signed_distance(x), surface.signed_distance(y));
                            let t = dx / (dx - dy);
                            let at = v3::add(x, v3::scale(v3::sub(y, x), t));
                            let near = rim
                                .iter()
                                .map(|s| to_segment(at, s.0, s.1))
                                .fold(f64::INFINITY, f64::min);
                            let within = surface.invert(at).is_some_and(|(u, v)| {
                                // A sign change away from the surface is the
                                // far side of one that ends.
                                let on = v3::dist(surface.point(u, v), at) <= clear;
                                let (pu, pv) = surface.period();
                                let shifts = |p: Option<f64>| match p {
                                    Some(t) => vec![0.0, -t, t],
                                    None => vec![0.0],
                                };
                                on && shifts(pu).into_iter().any(|du| {
                                    shifts(pv).into_iter().any(|dv| {
                                        let q = [u + du, v + dv];
                                        loops[face].iter().filter(|l| inside(&l.uv, q)).count() % 2
                                            == 1
                                    })
                                })
                            });
                            if near > clear && within {
                                out.push(Tangle::EdgeThroughFace { edge, face, at });
                            }
                        }
                    }
                    last = Some((i, d));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Body, Surface};
    use super::*;

    /// Two boxes as one body: apart it is sound, and run into each other their
    /// edges go through each other's faces.
    #[test]
    fn boxes_run_into_each_other_are_tangled() {
        let one = at(Body::cuboid([2.0; 3]), [0.0; 3]);
        let apart = at(Body::cuboid([2.0; 3]), [5.0, 0.0, 0.0]);
        let into = at(Body::cuboid([2.0, 1.0, 1.0]), [1.0, 0.1, 0.1]);
        assert_eq!(one.tangles(1e-3), vec![]);
        assert_eq!(together(&one, &apart).tangles(1e-3), vec![]);
        let found = together(&one, &into).tangles(1e-3);
        assert!(
            found
                .iter()
                .filter(|t| matches!(t, Tangle::EdgeThroughFace { .. }))
                .count()
                >= 4,
            "{found:?}"
        );
    }

    fn at(b: Body, by: V3) -> Body {
        b.translated(by).expect("a box moves")
    }

    fn together(a: &Body, b: &Body) -> Body {
        let (s, v, e) = (a.surfaces().len(), a.vertices().len(), a.edges().len());
        let mut surfaces: Vec<Surface> = a.surfaces().to_vec();
        surfaces.extend(b.surfaces().iter().cloned());
        let mut vertices = a.vertices().to_vec();
        vertices.extend_from_slice(b.vertices());
        let mut edges = a.edges().to_vec();
        edges.extend(b.edges().iter().cloned().map(|mut x| {
            x.surfaces = (x.surfaces.0 + s, x.surfaces.1 + s);
            x.vertices.iter_mut().for_each(|i| *i += v);
            x
        }));
        let mut faces = a.faces().to_vec();
        faces.extend(b.faces().iter().cloned().map(|mut f| {
            f.surface += s;
            f.edges.iter_mut().for_each(|i| *i += e);
            for l in f.loops.iter_mut().flatten() {
                l.vertices.iter_mut().for_each(|i| *i += v);
            }
            f
        }));
        Body::from_parts(surfaces, vertices, edges, faces)
    }
}
