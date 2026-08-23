//! Mesh diagnostics that say **where** a solid is broken, not just that it is.
//!
//! A boolean that fails in this kernel usually fails at one feature, and the
//! rest of the mesh is fine. Knowing "4 of 3062 edges are non-manifold" tells
//! you nothing about which feature; knowing those 4 edges sit in one cluster at
//! (43, 33, 40) tells you immediately. This module reports the defect
//! *locations*, clustered, so a broken model can be diagnosed by reading rather
//! than by bisecting the source.
//!
//! ```ignore
//! use threers::{mesh_report, parse_scad_file};
//!
//! let g = parse_scad_file("part.scad").unwrap().to_geometry_exact();
//! let r = mesh_report(&g);
//! if !r.watertight {
//!     for c in &r.clusters {
//!         println!("{} defects near [{:.1}, {:.1}, {:.1}]", c.count, c.at[0], c.at[1], c.at[2]);
//!     }
//! }
//! ```

use std::collections::HashMap;

use crate::core::BufferGeometry;

use super::{triangles, V3};

/// A group of defective edges that lie close together — almost always one
/// offending feature.
#[derive(Debug, Clone)]
pub struct DefectCluster {
    /// Centroid of the cluster.
    pub at: V3,
    /// How many defective edges fell into it.
    pub count: usize,
    /// Extent of the cluster, as a radius about `at`.
    pub radius: f64,
}

/// What [`mesh_report`] found.
#[derive(Debug, Clone)]
pub struct MeshReport {
    pub triangles: usize,
    /// Axis-aligned bounds, `(min, max)`.
    pub bounds: (V3, V3),
    /// Signed volume. Negative means the mesh is inside-out.
    pub volume: f64,
    /// Every edge used by exactly two triangles.
    pub watertight: bool,
    /// Edges used by exactly one triangle — holes in the surface.
    pub boundary_edges: usize,
    /// Edges used by three or more — self-intersection or duplicated surface.
    pub overused_edges: usize,
    /// Disjoint solids in the mesh. More than one is fine for a multi-body
    /// export and a red flag for anything meant to be a single part.
    pub shells: usize,
    /// Shells that are not closed. This, not the raw edge counts, is what makes
    /// a multi-body mesh sound or not.
    pub open_shells: usize,
    /// Defect locations, largest cluster first. Empty when watertight.
    pub clusters: Vec<DefectCluster>,
}

impl MeshReport {
    /// One-line summary suitable for a build log.
    pub fn summary(&self) -> String {
        let [sx, sy, sz] = [
            self.bounds.1[0] - self.bounds.0[0],
            self.bounds.1[1] - self.bounds.0[1],
            self.bounds.1[2] - self.bounds.0[2],
        ];
        format!(
            "{} tris · {sx:.3} x {sy:.3} x {sz:.3} mm · {:.2} cm^3 · {}",
            self.triangles,
            self.volume / 1000.0,
            if self.watertight {
                format!(
                    "watertight{}",
                    if self.shells > 1 {
                        format!(", {} shells", self.shells)
                    } else {
                        String::new()
                    }
                )
            } else {
                format!(
                    "BROKEN: {} of {} shells open ({} boundary, {} overused)",
                    self.open_shells, self.shells, self.boundary_edges, self.overused_edges
                )
            }
        )
    }
}

fn key(p: V3) -> (i64, i64, i64) {
    (
        (p[0] * 1e4).round() as i64,
        (p[1] * 1e4).round() as i64,
        (p[2] * 1e4).round() as i64,
    )
}

/// Inspect a mesh: bounds, volume, watertightness, and **where** any defects are.
pub fn mesh_report(g: &BufferGeometry) -> MeshReport {
    let tris = triangles(g);
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    let mut volume = 0.0f64;

    for t in &tris {
        for v in t {
            for k in 0..3 {
                lo[k] = lo[k].min(v[k]);
                hi[k] = hi[k].max(v[k]);
            }
        }
        let (a, b, c) = (t[0], t[1], t[2]);
        volume += (a[0] * (b[1] * c[2] - c[1] * b[2]) - a[1] * (b[0] * c[2] - c[0] * b[2])
            + a[2] * (b[0] * c[1] - c[0] * b[1]))
            / 6.0;
    }
    if tris.is_empty() {
        lo = [0.0; 3];
        hi = [0.0; 3];
    }

    // Count edges PER SHELL. A mesh built by `assembly()` is several separate
    // bodies in one buffer; two of them touching along an edge gives that edge
    // four triangles, which is perfectly legitimate — the bodies are distinct
    // solids. Judging the whole buffer as one solid reports those contacts as
    // defects. What actually matters is that EVERY SHELL is closed.
    let (labels, shells) = shell_labels(&tris);
    /// Edge (two snapped endpoints) -> how many faces use it, and its direction.
    type EdgeUse = HashMap<[(i64, i64, i64); 2], (u32, V3)>;
    let mut per_shell: Vec<EdgeUse> = vec![HashMap::new(); shells.max(1)];
    for (ti, t) in tris.iter().enumerate() {
        let sh = labels.get(ti).copied().unwrap_or(0);
        for e in 0..3 {
            let (p, q) = (t[e], t[(e + 1) % 3]);
            let (kp, kq) = (key(p), key(q));
            let k = if kp <= kq { [kp, kq] } else { [kq, kp] };
            let mid = [
                (p[0] + q[0]) / 2.0,
                (p[1] + q[1]) / 2.0,
                (p[2] + q[2]) / 2.0,
            ];
            per_shell[sh].entry(k).or_insert((0, mid)).0 += 1;
        }
    }
    let mut boundary = 0usize;
    let mut overused = 0usize;
    let mut open_shells = 0usize;
    let mut bad: Vec<V3> = Vec::new();
    for m in &per_shell {
        let before = boundary + overused;
        for (count, mid) in m.values() {
            match *count {
                2 => {}
                1 => {
                    boundary += 1;
                    bad.push(*mid);
                }
                _ => {
                    overused += 1;
                    bad.push(*mid);
                }
            }
        }
        if boundary + overused > before {
            open_shells += 1;
        }
    }

    // Cluster defects by proximity, so one broken feature reads as one entry.
    // The span-relative radius keeps this scale-free.
    let span = (hi[0] - lo[0])
        .max(hi[1] - lo[1])
        .max(hi[2] - lo[2])
        .max(1e-9);
    let tol = span * 0.05;
    let mut clusters: Vec<DefectCluster> = Vec::new();
    for p in bad {
        let mut placed = false;
        for c in clusters.iter_mut() {
            let d =
                ((p[0] - c.at[0]).powi(2) + (p[1] - c.at[1]).powi(2) + (p[2] - c.at[2]).powi(2))
                    .sqrt();
            if d <= tol {
                // running mean
                let n = c.count as f64;
                for (k, at) in c.at.iter_mut().enumerate() {
                    *at = (*at * n + p[k]) / (n + 1.0);
                }
                c.count += 1;
                c.radius = c.radius.max(d);
                placed = true;
                break;
            }
        }
        if !placed {
            clusters.push(DefectCluster {
                at: p,
                count: 1,
                radius: 0.0,
            });
        }
    }
    clusters.sort_by_key(|c| std::cmp::Reverse(c.count));

    MeshReport {
        triangles: tris.len(),
        bounds: (lo, hi),
        volume,
        watertight: boundary == 0 && overused == 0,
        boundary_edges: boundary,
        overused_edges: overused,
        shells,
        open_shells,
        clusters,
    }
}

/// Connected-component label per triangle, and the component count.
fn shell_labels(tris: &[[V3; 3]]) -> (Vec<usize>, usize) {
    if tris.is_empty() {
        return (Vec::new(), 0);
    }
    let mut label = vec![usize::MAX; tris.len()];
    let mut of_vertex: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (i, t) in tris.iter().enumerate() {
        for v in t {
            of_vertex.entry(key(*v)).or_default().push(i);
        }
    }
    let mut shells = 0;
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..tris.len() {
        if label[start] != usize::MAX {
            continue;
        }
        label[start] = shells;
        stack.push(start);
        while let Some(i) = stack.pop() {
            for v in &tris[i] {
                if let Some(nb) = of_vertex.get(&key(*v)) {
                    for &j in nb {
                        if label[j] == usize::MAX {
                            label[j] = shells;
                            stack.push(j);
                        }
                    }
                }
            }
        }
        shells += 1;
    }
    (label, shells)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openscad::{cube, sphere_fn};

    #[test]
    fn clean_cube_reports_watertight() {
        let r = mesh_report(&cube([10.0, 20.0, 30.0]).to_geometry_exact());
        assert!(r.watertight, "{}", r.summary());
        assert_eq!(r.shells, 1);
        assert!(r.clusters.is_empty());
        assert!((r.volume - 6000.0).abs() < 1e-6, "volume {}", r.volume);
    }

    #[test]
    fn two_disjoint_solids_report_two_shells() {
        let s = cube([4.0, 4.0, 4.0]).union(sphere_fn(2.0, 16).translate([40.0, 0.0, 0.0]));
        let r = mesh_report(&s.to_geometry_exact());
        assert_eq!(r.shells, 2, "{}", r.summary());
        assert!(r.watertight);
    }

    /// An open surface must report boundary edges and locate them.
    #[test]
    fn open_mesh_locates_its_hole() {
        use crate::{BufferAttribute, BufferGeometry};
        // A single triangle: three edges, each used once.
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(vec![0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0, 0.0], 3),
        );
        let r = mesh_report(&g);
        assert!(!r.watertight);
        assert_eq!(r.boundary_edges, 3);
        // Every defect is accounted for by some cluster...
        assert_eq!(r.clusters.iter().map(|c| c.count).sum::<usize>(), 3);
        // ...and each cluster is located on the triangle, which is the point.
        for c in &r.clusters {
            assert!(c.at[0] >= -1e-9 && c.at[0] <= 10.0, "x {}", c.at[0]);
            assert!(c.at[1] >= -1e-9 && c.at[1] <= 10.0, "y {}", c.at[1]);
            assert!(c.at[2].abs() < 1e-9, "z {}", c.at[2]);
        }
    }
}
