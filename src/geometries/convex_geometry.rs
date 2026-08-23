use crate::core::{BufferAttribute, BufferGeometry};
use crate::math::Vector3;

pub struct ConvexGeometry;

impl ConvexGeometry {
    /// Build the convex hull of a 3D point set. Uses an O(n²) gift-wrapping
    /// variant: bootstrap with the extreme tetrahedron, then for each face find
    /// the vertex farthest in the outward direction and connect a triangle fan.
    /// Sufficient for the small-to-medium clouds that three.js's `ConvexGeometry`
    /// is typically called with.
    pub fn new(points: &[Vector3]) -> BufferGeometry {
        let mut g = BufferGeometry::new();
        if points.len() < 4 {
            return g;
        }

        // Start with an arbitrary tetrahedron. Pick the 4 most-spread points.
        let (a, b, c, d) = bootstrap(points);
        if a == b || b == c {
            return g;
        }

        // Faces stored as outward-oriented triangles.
        let mut faces: Vec<[usize; 3]> = vec![[a, b, c], [a, c, d], [a, d, b], [b, d, c]];
        ensure_outward(points, &mut faces);

        // Iteratively add the point that lies farthest beyond any face.
        let mut included: Vec<bool> = vec![false; points.len()];
        for i in [a, b, c, d] {
            included[i] = true;
        }

        let mut guard = 4 * points.len();
        loop {
            guard = guard.saturating_sub(1);
            if guard == 0 {
                break;
            }
            let mut best: Option<(usize, usize, f32)> = None; // face, point, dist
            for (fi, face) in faces.iter().enumerate() {
                let n = face_normal(points, face);
                let p0 = points[face[0]];
                for (pi, p) in points.iter().enumerate() {
                    if included[pi] {
                        continue;
                    }
                    let d = (*p - p0).dot(n);
                    if d > 1e-5 && (best.is_none() || d > best.unwrap().2) {
                        best = Some((fi, pi, d));
                    }
                }
            }
            let Some((_, new_pt, _)) = best else {
                break;
            };
            // Find all visible faces from `new_pt`.
            let visible: Vec<usize> = faces
                .iter()
                .enumerate()
                .filter(|(_, f)| {
                    let n = face_normal(points, f);
                    (points[new_pt] - points[f[0]]).dot(n) > 1e-5
                })
                .map(|(i, _)| i)
                .collect();
            // Find horizon edges (edges of visible faces shared with no other visible face).
            let mut horizon: Vec<(usize, usize)> = Vec::new();
            for &vi in &visible {
                let f = faces[vi];
                for &(a, b) in &[(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
                    let shared = visible.iter().any(|&oi| {
                        if oi == vi {
                            return false;
                        }
                        let of = faces[oi];
                        edge_of(&of, b, a)
                    });
                    if !shared {
                        horizon.push((a, b));
                    }
                }
            }
            // Remove visible faces (back to front).
            let mut sorted = visible.clone();
            sorted.sort_unstable();
            for vi in sorted.iter().rev() {
                faces.swap_remove(*vi);
            }
            // Add new faces fanning from new_pt to horizon edges.
            for (a, b) in horizon {
                faces.push([a, b, new_pt]);
            }
            included[new_pt] = true;
        }

        // Emit as a flat-shaded triangle list with face normals.
        let mut positions = Vec::with_capacity(faces.len() * 9);
        let mut normals = Vec::with_capacity(faces.len() * 9);
        let mut uvs = Vec::with_capacity(faces.len() * 6);
        let mut indices = Vec::with_capacity(faces.len() * 3);
        for (i, face) in faces.iter().enumerate() {
            let p0 = points[face[0]];
            let p1 = points[face[1]];
            let p2 = points[face[2]];
            let n = (p1 - p0).cross(p2 - p0).normalize();
            for p in [p0, p1, p2] {
                positions.extend_from_slice(&[p.x, p.y, p.z]);
            }
            for _ in 0..3 {
                normals.extend_from_slice(&[n.x, n.y, n.z]);
            }
            uvs.extend_from_slice(&[0.0, 0.0, 1.0, 0.0, 0.5, 1.0]);
            let base = (i * 3) as u32;
            indices.extend_from_slice(&[base, base + 1, base + 2]);
        }
        g.set_attribute("position", BufferAttribute::new(positions, 3));
        g.set_attribute("normal", BufferAttribute::new(normals, 3));
        g.set_attribute("uv", BufferAttribute::new(uvs, 2));
        g.set_index(indices);
        g
    }
}

fn bootstrap(points: &[Vector3]) -> (usize, usize, usize, usize) {
    let mut min_x = 0;
    let mut max_x = 0;
    for (i, p) in points.iter().enumerate() {
        if p.x < points[min_x].x {
            min_x = i;
        }
        if p.x > points[max_x].x {
            max_x = i;
        }
    }
    let mut far_idx = 0;
    let mut far_dist = 0.0;
    for (i, p) in points.iter().enumerate() {
        let line_dir = (points[max_x] - points[min_x]).normalize();
        let to_p = *p - points[min_x];
        let perp = to_p - line_dir * to_p.dot(line_dir);
        let d = perp.length();
        if d > far_dist {
            far_dist = d;
            far_idx = i;
        }
    }
    let n = face_normal(points, &[min_x, max_x, far_idx]);
    let mut top_idx = 0;
    let mut top_dist = 0.0;
    for (i, p) in points.iter().enumerate() {
        let d = (*p - points[min_x]).dot(n).abs();
        if d > top_dist {
            top_dist = d;
            top_idx = i;
        }
    }
    (min_x, max_x, far_idx, top_idx)
}

fn face_normal(points: &[Vector3], face: &[usize; 3]) -> Vector3 {
    let p0 = points[face[0]];
    let p1 = points[face[1]];
    let p2 = points[face[2]];
    (p1 - p0).cross(p2 - p0).normalize()
}

fn ensure_outward(points: &[Vector3], faces: &mut [[usize; 3]]) {
    let centroid = {
        let mut sum = Vector3::ZERO;
        let mut count = 0;
        for f in faces.iter() {
            for &i in f {
                sum = sum + points[i];
                count += 1;
            }
        }
        sum * (1.0 / count.max(1) as f32)
    };
    for f in faces.iter_mut() {
        let n = face_normal(points, f);
        if (points[f[0]] - centroid).dot(n) < 0.0 {
            f.swap(1, 2);
        }
    }
}

fn edge_of(face: &[usize; 3], a: usize, b: usize) -> bool {
    (face[0] == a && face[1] == b)
        || (face[1] == a && face[2] == b)
        || (face[2] == a && face[0] == b)
}
