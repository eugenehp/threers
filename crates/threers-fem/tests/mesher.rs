//! The tetrahedral mesher: closed, conforming, positive, on the surface,
//! and of usable quality.

use threers::geometries::{BoxGeometry, IcosahedronGeometry, TorusGeometry};
use threers_fem::{SignedDistance, TetMesh, TriSurface};

fn check(mesh: &TetMesh, name: &str) {
    assert!(!mesh.tets.is_empty(), "{name}: empty");
    assert_eq!(mesh.defects(), 0, "{name}: not conforming / not closed");
    for t in 0..mesh.tets.len() {
        assert!(mesh.tet_volume(t) > 0.0, "{name}: tet {t} inverted");
    }
    let q = mesh.quality();
    assert!(q.min_dihedral > 1.0, "{name}: min dihedral {}", q.min_dihedral);
    assert!(q.mean_radius_ratio > 0.6, "{name}: mean radius ratio {}", q.mean_radius_ratio);
}

#[test]
fn sphere_from_sdf() {
    for h in [0.3, 0.15, 0.08] {
        let mesh = TetMesh::from_sdf([-1.0; 3], [1.0; 3], h, |p| {
            (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() - 1.0
        })
        .unwrap();
        check(&mesh, &format!("sphere h={h}"));
        // Inscribed facets: volume a little under 4π/3, closing in as h².
        let v = mesh.volume();
        let want = 4.0 / 3.0 * std::f64::consts::PI;
        assert!(v < want && v > want * (1.0 - 0.6 * h * h), "h={h}: {v}");
        // Boundary nodes on the sphere.
        let (s, _) = mesh.boundary_surface();
        for p in &s.vertices {
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!((r - 1.0).abs() < 1e-5, "h={h}: boundary node at r={r}");
        }
    }
}

#[test]
fn torus_from_a_triangle_mesh() {
    let geom = TorusGeometry::new(1.0, 0.35, 32, 64, std::f32::consts::TAU);
    let surface = TriSurface::from_geometry(&geom).unwrap();
    assert_eq!(surface.defects(), 0, "torus should weld closed");
    let mesh = TetMesh::from_surface(&surface, 0.08).unwrap();
    check(&mesh, "torus");
    let want = surface.volume();
    assert!((mesh.volume() / want - 1.0).abs() < 0.03, "{} vs {want}", mesh.volume());
    // Genus 1 survives: the hole's centre is outside the mesh.
    let sd = SignedDistance::new(&surface).unwrap();
    assert!(sd.distance([0.0, 0.0, 0.0]) > 0.5);
    let (b, _) = mesh.boundary_surface();
    assert!(b.winding_number([0.0, 0.0, 0.0]).abs() < 1e-6);
    assert!((b.winding_number([1.0, 0.0, 0.0]) - 1.0).abs() < 1e-6);
}

#[test]
fn box_keeps_its_faces_and_bevels_its_edges() {
    let surface = TriSurface::from_geometry(&BoxGeometry::new(2.0, 1.0, 1.0)).unwrap();
    let h = 0.1;
    let mesh = TetMesh::from_surface(&surface, h).unwrap();
    check(&mesh, "box");
    // Volume lost only along the 12 edges, a bevel at most ~h across.
    let v = mesh.volume();
    assert!(v <= 2.0 + 1e-9 && v > 2.0 - 12.0 * 1.5 * 0.5 * h * h, "{v}");
    // The faces are flat: boundary nodes lie on the planes.
    let (s, _) = mesh.boundary_surface();
    for p in &s.vertices {
        let on = (p[0].abs() - 1.0).abs() < 1e-6 || (p[1].abs() - 0.5).abs() < 1e-6 || (p[2].abs() - 0.5).abs() < 1e-6;
        assert!(on, "boundary node {p:?} off the box");
    }
}

#[test]
fn icosphere_input_and_rejection_of_open_surfaces() {
    let s = TriSurface::from_geometry(&IcosahedronGeometry::new(1.0, 2)).unwrap();
    let mesh = TetMesh::from_surface(&s, 0.25).unwrap();
    check(&mesh, "icosphere");
    let mut open = s.clone();
    open.triangles.pop();
    assert!(TetMesh::from_surface(&open, 0.25).is_err());
}
