//! The voxel solver against what is known: a box's volume, Hooke's law in a
//! stretched block, beam theory in a cantilever, a symmetry plane standing
//! in for a clamp, and the errors it gives for what it cannot solve.

use threers::{BoxGeometry, BufferAttribute, BufferGeometry};
use threers_fem::fem::voxel::{self, overlap, VoxelBody, VoxelGrid};
use threers_fem::fem::Material;
use threers_fem::FemError;

/// The nodes at the two ends of a bar along x.
fn ends(body: &VoxelBody) -> (Vec<u32>, Vec<u32>) {
    let lo = body.nodes.iter().map(|q| q[0]).fold(f64::MAX, f64::min);
    let hi = body.nodes.iter().map(|q| q[0]).fold(f64::MIN, f64::max);
    let at = |x: f64| (0..body.nodes.len() as u32).filter(|&n| (body.nodes[n as usize][0] - x).abs() < 0.1).collect();
    (at(lo), at(hi))
}

fn clamped(nodes: Vec<u32>) -> Vec<(u32, [bool; 3])> {
    nodes.into_iter().map(|n| (n, [true; 3])).collect()
}

fn moved(g: &BufferGeometry, dx: f32) -> BufferGeometry {
    let mut out = g.clone();
    let p: Vec<f32> = g.get_attribute("position").unwrap().array.chunks(3).flat_map(|v| [v[0] + dx, v[1], v[2]]).collect();
    out.set_attribute("position", BufferAttribute::new(p, 3));
    out
}

#[test]
fn a_filled_box_has_its_volume_and_two_apart_share_none() {
    let g = BoxGeometry::new(10.0, 6.0, 4.0);
    let grid = VoxelGrid::around(&[&g], 0.5);
    assert_eq!(grid.volume(&grid.fill(&g)), 240.0);
    assert!((overlap(&g, &g, 0.5) - 240.0).abs() < 1e-3);
    assert_eq!(overlap(&g, &moved(&g, 20.0), 0.5), 0.0);
    assert!((overlap(&g, &moved(&g, 5.0), 0.5) - 120.0).abs() < 1e-3, "half over");
}

#[test]
fn a_stretched_block_is_uniform_and_hookes() {
    // 10 × 4 × 4, pulled along its length by 16 N: 1 MPa through it.
    let body = VoxelBody::new(&BoxGeometry::new(10.0, 4.0, 4.0), 1.0).unwrap();
    assert_eq!(body.elements(), 160);
    let (held, end) = ends(&body);
    let pull: Vec<(u32, [f64; 3])> = end.iter().map(|&n| (n, [16.0 / end.len() as f64, 0.0, 0.0])).collect();
    let s = voxel::solve(&body, &Material::new(1000.0, 0.0), &clamped(held), &pull).unwrap();
    // σ / E × L = 0.01 mm (loads lumped at nodes scatter it a little).
    let tip: f64 = end.iter().map(|&n| s.displacement[n as usize][0]).sum::<f64>() / end.len() as f64;
    assert!((tip - 0.01).abs() < 0.0015, "{tip}");
    let mid = body.nearest([0.0, 0.0, 0.0]).unwrap();
    assert!((s.stress[mid as usize] - 1.0).abs() < 0.1, "{}", s.stress[mid as usize]);
}

#[test]
fn a_cantilever_bends_as_the_beam_does() {
    // 60 long, 4 × 4, held at one end, 2 N at the other: four cubes through
    // the thickness, which plain bricks would make far too stiff.
    let (l, h, f, e) = (60.0f64, 4.0f64, 2.0f64, 1800.0f64);
    let body = VoxelBody::new(&BoxGeometry::new(l as f32, h as f32, h as f32), 1.0).unwrap();
    let (held, end) = ends(&body);
    let push: Vec<(u32, [f64; 3])> = end.iter().map(|&n| (n, [0.0, -f / end.len() as f64, 0.0])).collect();
    let s = voxel::solve(&body, &Material::new(e, 0.3), &clamped(held), &push).unwrap();
    let i = h.powi(4) / 12.0;
    let beam_tip = f * l.powi(3) / (3.0 * e * i);
    let tip = -end.iter().map(|&n| s.displacement[n as usize][1]).sum::<f64>() / end.len() as f64;
    assert!((tip / beam_tip - 1.0).abs() < 0.05, "tip {tip} against the beam's {beam_tip}");
    // The stress a cube in from the root: M c / I.
    let at = -l / 2.0 + 2.0;
    let fem = (0..body.nodes.len()).filter(|&n| (body.nodes[n][0] - at).abs() < 0.1).map(|n| s.stress[n]).fold(0.0, f64::max);
    let beam = f * (l / 2.0 - at) * (h / 2.0) / i;
    assert!((fem / beam - 1.0).abs() < 0.12, "stress {fem} against the beam's {beam}");
}

#[test]
fn a_symmetry_plane_holds_half_as_a_clamp_would_the_whole() {
    // A bar pulled at both ends, held only across its middle plane and at
    // two nodes there against drifting and turning: it stretches as one
    // pulled from a clamp.
    let body = VoxelBody::new(&BoxGeometry::new(20.0, 4.0, 4.0), 1.0).unwrap();
    let (lo, hi) = ends(&body);
    let middle: Vec<u32> = (0..body.nodes.len() as u32).filter(|&n| body.nodes[n as usize][0].abs() < 1e-9).collect();
    assert!(!middle.is_empty(), "a layer of nodes on the plane through the origin");
    let y = |n: &&u32| body.nodes[**n as usize][1];
    let low = *middle.iter().min_by(|a, b| y(a).total_cmp(&y(b))).unwrap();
    let high = *middle.iter().max_by(|a, b| y(a).total_cmp(&y(b))).unwrap();
    let mut held: Vec<(u32, [bool; 3])> = middle.iter().map(|&n| (n, [true, false, false])).collect();
    held.push((low, [true; 3]));
    held.push((high, [true, false, true]));
    let mut pull: Vec<(u32, [f64; 3])> = hi.iter().map(|&n| (n, [16.0 / hi.len() as f64, 0.0, 0.0])).collect();
    pull.extend(lo.iter().map(|&n| (n, [-16.0 / lo.len() as f64, 0.0, 0.0])));
    let s = voxel::solve(&body, &Material::new(1000.0, 0.3), &held, &pull).unwrap();
    let tip: f64 = hi.iter().map(|&n| s.displacement[n as usize][0]).sum::<f64>() / hi.len() as f64;
    assert!((tip - 0.01).abs() < 0.0015, "{tip}");
}

#[test]
fn what_cannot_be_solved_says_why() {
    let body = VoxelBody::new(&BoxGeometry::new(10.0, 4.0, 4.0), 1.0).unwrap();
    let push = [(0u32, [1.0, 0.0, 0.0])];
    // Nothing held: it would fly off, and never converges.
    assert!(matches!(voxel::solve(&body, &Material::new(1000.0, 0.3), &[], &push), Err(FemError::NoConvergence { .. })));
    assert!(matches!(voxel::solve(&body, &Material::new(1000.0, 0.5), &[], &push), Err(FemError::InvalidInput(_))));
    let past = [(u32::MAX, [1.0, 0.0, 0.0])];
    assert!(matches!(voxel::solve(&body, &Material::new(1000.0, 0.3), &[], &past), Err(FemError::InvalidInput(_))));
    // Smaller than a cube, nothing inside.
    assert!(matches!(VoxelBody::new(&BoxGeometry::new(0.2, 0.2, 0.2), 1.0), Err(FemError::InvalidSurface(_))));
    assert!(matches!(VoxelBody::new(&BoxGeometry::new(1.0, 1.0, 1.0), 0.0), Err(FemError::InvalidInput(_))));
}
