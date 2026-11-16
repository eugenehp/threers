//! Closed triangle surfaces, in double precision: what the mesher fills and
//! the boundary element method discretises.
//!
//! [`TriSurface::from_geometry`] takes a `BufferGeometry` as the renderer
//! would draw it — indexed or not, with vertices split along creases for
//! their normals — and welds it back into one connected surface, since a
//! seam the eye cannot see is a crack to a solver.

use crate::vec3::*;
use crate::FemError;
use std::collections::HashMap;
use threers::core::{BufferAttribute, BufferGeometry};

/// An indexed triangle surface. Triangles wind counter-clockwise seen from
/// outside, as three.js geometry does, so normals point out of the solid.
#[derive(Clone, Debug, Default)]
pub struct TriSurface {
    pub vertices: Vec<V3>,
    pub triangles: Vec<[u32; 3]>,
}

impl TriSurface {
    pub fn new(vertices: Vec<V3>, triangles: Vec<[u32; 3]>) -> Self {
        Self {
            vertices,
            triangles,
        }
    }

    /// Weld a renderable geometry into a solver surface: vertices within
    /// `1e-7` of the bounding-box diagonal merge, and triangles that collapse
    /// in the merge are dropped.
    pub fn from_geometry(geometry: &BufferGeometry) -> Result<Self, FemError> {
        let pos = geometry
            .get_attribute("position")
            .ok_or(FemError::InvalidSurface("geometry has no position attribute"))?;
        if pos.item_size != 3 {
            return Err(FemError::InvalidSurface("position is not 3-component"));
        }
        let raw: Vec<V3> = pos
            .array
            .as_chunks::<3>()
            .0
            .iter()
            .map(|c| c.map(f64::from))
            .collect();
        let index: Vec<u32> = match &geometry.index {
            Some(ix) => ix.clone(),
            None => (0..raw.len() as u32).collect(),
        };
        if !index.len().is_multiple_of(3) || index.is_empty() {
            return Err(FemError::InvalidSurface("index is not a list of triangles"));
        }
        Ok(Self::welded(&raw, &index))
    }

    /// Merge coincident vertices of a triangle soup.
    pub fn welded(points: &[V3], index: &[u32]) -> Self {
        let (lo, hi) = bounds(points);
        let diag = norm(sub(hi, lo)).max(1e-300);
        let q = 1e-7 * diag;
        let key = |p: V3| {
            [
                ((p[0] - lo[0]) / q).round() as i64,
                ((p[1] - lo[1]) / q).round() as i64,
                ((p[2] - lo[2]) / q).round() as i64,
            ]
        };
        let mut map: HashMap<[i64; 3], u32> = HashMap::new();
        let mut vertices = Vec::new();
        let mut remap = vec![0u32; points.len()];
        for (i, p) in points.iter().enumerate() {
            let k = key(*p);
            // Check the neighbouring cells too, so two points straddling a
            // rounding boundary still merge.
            let mut found = None;
            'search: for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(&v) = map.get(&[k[0] + dx, k[1] + dy, k[2] + dz])
                            && dist(vertices[v as usize], *p) <= q
                        {
                            found = Some(v);
                            break 'search;
                        }
                    }
                }
            }
            remap[i] = match found {
                Some(v) => v,
                None => {
                    let v = vertices.len() as u32;
                    vertices.push(*p);
                    map.insert(k, v);
                    v
                }
            };
        }
        let triangles = index
            .as_chunks::<3>()
            .0
            .iter()
            .map(|t| t.map(|i| remap[i as usize]))
            .filter(|t| t[0] != t[1] && t[1] != t[2] && t[0] != t[2])
            .collect();
        Self {
            vertices,
            triangles,
        }
    }

    /// Back to a renderable geometry, with flat-shading-free vertex normals.
    pub fn to_geometry(&self) -> BufferGeometry {
        let mut g = BufferGeometry::new();
        let pos: Vec<f64> = self.vertices.iter().flatten().copied().collect();
        g.set_attribute("position", BufferAttribute::from_f64(&pos, 3));
        let normals: Vec<f64> = self.vertex_normals().iter().flatten().copied().collect();
        g.set_attribute("normal", BufferAttribute::from_f64(&normals, 3));
        g.set_index(self.triangles.iter().flatten().copied().collect());
        g
    }

    pub fn triangle(&self, t: usize) -> [V3; 3] {
        let [a, b, c] = self.triangles[t];
        [
            self.vertices[a as usize],
            self.vertices[b as usize],
            self.vertices[c as usize],
        ]
    }

    /// Unit normal and area of triangle `t`.
    pub fn normal_area(&self, t: usize) -> (V3, f64) {
        let [a, b, c] = self.triangle(t);
        let n = cross(sub(b, a), sub(c, a));
        let l = norm(n);
        (scale(n, 1.0 / l.max(1e-300)), 0.5 * l)
    }

    pub fn bounds(&self) -> (V3, V3) {
        bounds(&self.vertices)
    }

    pub fn area(&self) -> f64 {
        (0..self.triangles.len()).map(|t| self.normal_area(t).1).sum()
    }

    /// Enclosed volume, by the divergence theorem. Negative if the surface is
    /// wound inside out.
    pub fn volume(&self) -> f64 {
        (0..self.triangles.len())
            .map(|t| {
                let [a, b, c] = self.triangle(t);
                dot(a, cross(b, c)) / 6.0
            })
            .sum()
    }

    /// Area-weighted vertex normals.
    pub fn vertex_normals(&self) -> Vec<V3> {
        let mut n = vec![[0.0; 3]; self.vertices.len()];
        for t in &self.triangles {
            let [a, b, c] = t.map(|i| self.vertices[i as usize]);
            let f = cross(sub(b, a), sub(c, a));
            for &i in t {
                n[i as usize] = add(n[i as usize], f);
            }
        }
        n.into_iter().map(normalize).collect()
    }

    /// Edges used by other than exactly two triangles, or used twice in the
    /// same direction. Zero for a closed, consistently wound surface.
    pub fn defects(&self) -> usize {
        let mut count: HashMap<(u32, u32), i32> = HashMap::new();
        for t in &self.triangles {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                *count.entry((a, b)).or_default() += 1;
            }
        }
        count
            .iter()
            .filter(|((a, b), c)| **c != 1 || count.get(&(*b, *a)) != Some(&1))
            .count()
    }

    /// Generalised winding number at `p`: 1 inside a closed surface, 0
    /// outside, ½ on it. Exact sum of solid angles, O(triangles).
    pub fn winding_number(&self, p: V3) -> f64 {
        let mut omega = 0.0;
        for t in 0..self.triangles.len() {
            let [a, b, c] = self.triangle(t);
            let (a, b, c) = (sub(a, p), sub(b, p), sub(c, p));
            let (la, lb, lc) = (norm(a), norm(b), norm(c));
            let num = dot(a, cross(b, c));
            let den = la * lb * lc + dot(a, b) * lc + dot(b, c) * la + dot(c, a) * lb;
            omega += 2.0 * num.atan2(den);
        }
        omega / (4.0 * std::f64::consts::PI)
    }
}

/// Signed distance to a closed triangle surface — negative inside — by a
/// bounding volume hierarchy and angle-weighted pseudonormals (Bærentzen and
/// Aanæs): the sign comes from the normal of whichever feature, face, edge or
/// vertex, the closest point lies on, which is exact for a closed manifold.
pub struct SignedDistance {
    surface: TriSurface,
    face_normals: Vec<V3>,
    edge_normals: HashMap<(u32, u32), V3>,
    vertex_normals: Vec<V3>,
    bvh: Bvh,
}

impl SignedDistance {
    pub fn new(surface: &TriSurface) -> Result<Self, FemError> {
        if surface.triangles.is_empty() {
            return Err(FemError::InvalidSurface("no triangles"));
        }
        if surface.defects() != 0 {
            return Err(FemError::InvalidSurface(
                "surface is not closed and consistently wound",
            ));
        }
        let mut surface = surface.clone();
        if surface.volume() < 0.0 {
            for t in &mut surface.triangles {
                t.swap(1, 2);
            }
        }
        let n = surface.triangles.len();
        let mut face_normals = Vec::with_capacity(n);
        let mut edge_normals: HashMap<(u32, u32), V3> = HashMap::new();
        let mut vertex_normals = vec![[0.0; 3]; surface.vertices.len()];
        for t in 0..n {
            let (fn_, _) = surface.normal_area(t);
            face_normals.push(fn_);
            let tri = surface.triangles[t];
            let p = surface.triangle(t);
            for k in 0..3 {
                let (a, b) = (tri[k], tri[(k + 1) % 3]);
                let key = (a.min(b), a.max(b));
                let e = edge_normals.entry(key).or_insert([0.0; 3]);
                *e = add(*e, fn_);
                // Angle at corner k.
                let u = normalize(sub(p[(k + 1) % 3], p[k]));
                let v = normalize(sub(p[(k + 2) % 3], p[k]));
                let angle = dot(u, v).clamp(-1.0, 1.0).acos();
                let vn = &mut vertex_normals[tri[k] as usize];
                *vn = add(*vn, scale(fn_, angle));
            }
        }
        let bvh = Bvh::build(&surface);
        Ok(Self {
            surface,
            face_normals,
            edge_normals,
            vertex_normals,
            bvh,
        })
    }

    pub fn surface(&self) -> &TriSurface {
        &self.surface
    }

    /// Closest point on the surface to `p`, and the triangle it lies on.
    pub fn closest_point(&self, p: V3) -> (V3, usize) {
        let (c, t, _) = self.bvh.closest(&self.surface, p);
        (c, t)
    }

    /// Signed distance from `p`: negative inside.
    pub fn distance(&self, p: V3) -> f64 {
        let (c, t, feature) = self.bvh.closest(&self.surface, p);
        let tri = self.surface.triangles[t];
        let n = match feature {
            Feature::Face => self.face_normals[t],
            Feature::Edge(k) => {
                let (a, b) = (tri[k], tri[(k + 1) % 3]);
                self.edge_normals[&(a.min(b), a.max(b))]
            }
            Feature::Vertex(k) => self.vertex_normals[tri[k] as usize],
        };
        let d = sub(p, c);
        let len = norm(d);
        if dot(d, n) < 0.0 { -len } else { len }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Feature {
    Face,
    /// Edge from corner k to k+1.
    Edge(usize),
    Vertex(usize),
}

/// Closest point on triangle `abc` to `p` (Ericson, Real-Time Collision
/// Detection §5.1.5), with the feature it lies on.
fn closest_on_triangle(p: V3, a: V3, b: V3, c: V3) -> (V3, Feature) {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, Feature::Vertex(0));
    }
    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, Feature::Vertex(1));
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (add(a, scale(ab, v)), Feature::Edge(0));
    }
    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, Feature::Vertex(2));
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (add(a, scale(ac, w)), Feature::Edge(2));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (add(b, scale(sub(c, b), w)), Feature::Edge(1));
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (add(a, add(scale(ab, v), scale(ac, w))), Feature::Face)
}

struct BvhNode {
    lo: V3,
    hi: V3,
    /// Leaf (`count > 0`): triangles `start..start + count` of `order`.
    /// Inner (`count == 0`): children `left` and `start`.
    start: u32,
    count: u32,
    left: u32,
}

struct Bvh {
    nodes: Vec<BvhNode>,
    order: Vec<u32>,
}

impl Bvh {
    fn build(s: &TriSurface) -> Self {
        let n = s.triangles.len();
        let centroids: Vec<V3> = (0..n)
            .map(|t| {
                let [a, b, c] = s.triangle(t);
                scale(add(add(a, b), c), 1.0 / 3.0)
            })
            .collect();
        let mut order: Vec<u32> = (0..n as u32).collect();
        let mut nodes = Vec::with_capacity(2 * n / 4 + 1);
        fn recurse(
            s: &TriSurface,
            centroids: &[V3],
            order: &mut [u32],
            offset: usize,
            nodes: &mut Vec<BvhNode>,
        ) -> u32 {
            let mut lo = [f64::INFINITY; 3];
            let mut hi = [f64::NEG_INFINITY; 3];
            for &t in order.iter() {
                for p in s.triangle(t as usize) {
                    lo = min3(lo, p);
                    hi = max3(hi, p);
                }
            }
            let id = nodes.len() as u32;
            nodes.push(BvhNode {
                lo,
                hi,
                start: offset as u32,
                count: order.len() as u32,
                left: 0,
            });
            if order.len() <= 4 {
                return id;
            }
            let mut clo = [f64::INFINITY; 3];
            let mut chi = [f64::NEG_INFINITY; 3];
            for &t in order.iter() {
                clo = min3(clo, centroids[t as usize]);
                chi = max3(chi, centroids[t as usize]);
            }
            let ext = sub(chi, clo);
            let axis = if ext[0] >= ext[1] && ext[0] >= ext[2] {
                0
            } else if ext[1] >= ext[2] {
                1
            } else {
                2
            };
            let mid = order.len() / 2;
            order.select_nth_unstable_by(mid, |&a, &b| {
                centroids[a as usize][axis].total_cmp(&centroids[b as usize][axis])
            });
            let (l, r) = order.split_at_mut(mid);
            let left = recurse(s, centroids, l, offset, nodes);
            let right = recurse(s, centroids, r, offset + mid, nodes);
            nodes[id as usize].count = 0;
            nodes[id as usize].left = left;
            nodes[id as usize].start = right;
            id
        }
        recurse(s, &centroids, &mut order, 0, &mut nodes);
        Self { nodes, order }
    }

    fn closest(&self, s: &TriSurface, p: V3) -> (V3, usize, Feature) {
        let box_d2 = |n: &BvhNode| {
            let mut d2 = 0.0;
            for k in 0..3 {
                let v = if p[k] < n.lo[k] {
                    n.lo[k] - p[k]
                } else if p[k] > n.hi[k] {
                    p[k] - n.hi[k]
                } else {
                    0.0
                };
                d2 += v * v;
            }
            d2
        };
        let mut best = (f64::INFINITY, [0.0; 3], 0usize, Feature::Face);
        let mut stack = vec![(0u32, box_d2(&self.nodes[0]))];
        while let Some((id, d2)) = stack.pop() {
            if d2 >= best.0 {
                continue;
            }
            let node = &self.nodes[id as usize];
            if node.count > 0 {
                for &t in &self.order[node.start as usize..(node.start + node.count) as usize] {
                    let [a, b, c] = s.triangle(t as usize);
                    let (q, f) = closest_on_triangle(p, a, b, c);
                    let d2 = norm2(sub(p, q));
                    if d2 < best.0 {
                        best = (d2, q, t as usize, f);
                    }
                }
            } else {
                let l = node.left;
                let r = node.start;
                let dl = box_d2(&self.nodes[l as usize]);
                let dr = box_d2(&self.nodes[r as usize]);
                // Push the farther first so the nearer is searched first.
                if dl < dr {
                    stack.push((r, dr));
                    stack.push((l, dl));
                } else {
                    stack.push((l, dl));
                    stack.push((r, dr));
                }
            }
        }
        (best.1, best.2, best.3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use threers::geometries::{BoxGeometry, SphereGeometry};

    #[test]
    fn box_welds_closed_and_signs_correctly() {
        let s = TriSurface::from_geometry(&BoxGeometry::new(2.0, 2.0, 2.0)).unwrap();
        assert_eq!(s.vertices.len(), 8);
        assert_eq!(s.defects(), 0);
        assert!((s.volume() - 8.0).abs() < 1e-5);
        let sd = SignedDistance::new(&s).unwrap();
        assert!((sd.distance([0.0, 0.0, 0.0]) + 1.0).abs() < 1e-6);
        assert!((sd.distance([3.0, 0.0, 0.0]) - 2.0).abs() < 1e-6);
        // Off a corner: the vertex pseudonormal decides.
        let d = sd.distance([2.0, 2.0, 2.0]);
        assert!((d - 3f64.sqrt()).abs() < 1e-6);
        // Just inside an edge.
        assert!(sd.distance([0.99, 0.99, 0.0]) < 0.0);
        assert!((s.winding_number([0.2, 0.1, -0.3]) - 1.0).abs() < 1e-9);
        assert!(s.winding_number([1.2, 0.1, -0.3]).abs() < 1e-9);
    }

    #[test]
    fn sphere_distance() {
        let s = TriSurface::from_geometry(&SphereGeometry::new(1.0, 48, 24)).unwrap();
        assert_eq!(s.defects(), 0, "UV sphere should weld closed");
        let sd = SignedDistance::new(&s).unwrap();
        for p in [[0.3, 0.2, 0.1], [0.0, 0.0, 0.0], [1.5, 0.0, 0.2], [0.0, -2.0, 0.0]] {
            let want = norm(p) - 1.0;
            assert!((sd.distance(p) - want).abs() < 5e-3, "{p:?}");
        }
    }
}
