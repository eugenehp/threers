//! Classic 15-step origami bird (traditional crane / bird base).
//!
//! Traced from the standard step-by-step instructional diagram. Like
//! [`super::giang_cardinal`], this is traditional origami — 3D stages are
//! hand-guided poses, not rigid-fold solver output.

use super::fold::FoldedState;
use super::pattern::{CreasePattern, Edge, EdgeKind};
use super::V2;

/// One illustrated fold step.
#[derive(Debug, Clone)]
pub struct ClassicBirdStage {
    pub label: &'static str,
    pub folded: FoldedState,
}

/// Crease pattern on a unit square (steps 1–11 accumulated).
pub fn crease_pattern() -> CreasePattern {
    let bl = v2(0.0, 0.0);
    let br = v2(1.0, 0.0);
    let tr = v2(1.0, 1.0);
    let tl = v2(0.0, 1.0);
    let c = v2(0.5, 0.5);
    let bm = v2(0.5, 0.0);
    let rm = v2(1.0, 0.5);
    let tm = v2(0.5, 1.0);
    let lm = v2(0.0, 0.5);
    let q1 = v2(0.25, 0.75);
    let q2 = v2(0.75, 0.75);
    let q3 = v2(0.75, 0.25);
    let q4 = v2(0.25, 0.25);
    let p1 = v2(0.5, 0.75);
    let p2 = v2(0.75, 0.5);
    let p3 = v2(0.5, 0.25);
    let p4 = v2(0.25, 0.5);

    let verts = vec![bl, br, tr, tl, c, bm, rm, tm, lm, q1, q2, q3, q4, p1, p2, p3, p4];
    let mut edges = Vec::new();

    bound(&mut edges, 0, 1);
    bound(&mut edges, 1, 2);
    bound(&mut edges, 2, 3);
    bound(&mut edges, 3, 0);

    // Step 1 — diagonals + centre cross.
    hinge(&mut edges, 0, 2);
    hinge(&mut edges, 1, 3);
    hinge(&mut edges, 4, 5);
    hinge(&mut edges, 4, 6);
    hinge(&mut edges, 4, 7);
    hinge(&mut edges, 4, 8);
    hinge(&mut edges, 0, 4);
    hinge(&mut edges, 1, 4);
    hinge(&mut edges, 2, 4);
    hinge(&mut edges, 3, 4);

    // Preliminary / bird-base creases (steps 2–11).
    hinge(&mut edges, 8, 7);
    hinge(&mut edges, 6, 7);
    hinge(&mut edges, 1, 7);
    hinge(&mut edges, 0, 7);
    hinge(&mut edges, 9, 13);
    hinge(&mut edges, 10, 13);
    hinge(&mut edges, 11, 15);
    hinge(&mut edges, 12, 15);
    hinge(&mut edges, 13, 14);
    hinge(&mut edges, 14, 15);
    hinge(&mut edges, 13, 4);
    hinge(&mut edges, 15, 4);
    hinge(&mut edges, 9, 3);
    hinge(&mut edges, 10, 2);
    hinge(&mut edges, 11, 1);
    hinge(&mut edges, 12, 0);

    // Narrow + head/tail (steps 12–14).
    hinge(&mut edges, 5, 15);
    hinge(&mut edges, 5, 13);
    hinge(&mut edges, 3, 13);
    hinge(&mut edges, 1, 15);

    CreasePattern::new(verts, edges)
}

/// Fifteen poses matching the instructional strip.
pub fn fold_stages() -> Vec<ClassicBirdStage> {
    vec![
        stage("1 — creases", flat_square()),
        stage("2 — to centre", corners_in()),
        stage("3 — prelim base", prelim_base()),
        stage("4 — open flap", open_flap()),
        stage("5 — squash", squash_mid()),
        stage("6 — squash flat", squash_flat()),
        stage("7 — turn over", turn_over()),
        stage("8 — petal prep", petal_prep()),
        stage("9 — petal fold", petal_fold()),
        stage("10 — bird base", bird_base()),
        stage("11 — narrow", narrow_legs()),
        stage("12 — reverse", reverse_head_tail()),
        stage("13 — head/tail up", head_tail_up()),
        stage("14 — wings down", wings_down()),
        stage("15 — finished", finished()),
    ]
}

fn stage(label: &'static str, folded: FoldedState) -> ClassicBirdStage {
    ClassicBirdStage { label, folded }
}

fn v2(x: f64, y: f64) -> V2 {
    [x, y]
}

fn hinge(edges: &mut Vec<Edge>, a: usize, b: usize) {
    edges.push(Edge {
        a,
        b,
        kind: EdgeKind::Hinge,
    });
}

fn bound(edges: &mut Vec<Edge>, a: usize, b: usize) {
    edges.push(Edge {
        a,
        b,
        kind: EdgeKind::Boundary,
    });
}

fn flat_square() -> FoldedState {
    let s = 0.9;
    let z = 0.0;
    let verts = vec![
        [-s, -s, z],
        [s, -s, z],
        [s, s, z],
        [-s, s, z],
    ];
    let faces = vec![vec![0, 1, 2], vec![0, 2, 3]];
    FoldedState { verts, faces }
}

fn corners_in() -> FoldedState {
    let verts = vec![
        [0.0, 0.82, 0.0],
        [0.58, 0.0, 0.0],
        [0.0, -0.82, 0.0],
        [-0.58, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [-0.42, 0.38, 0.08],
        [0.42, 0.38, 0.08],
    ];
    let faces = vec![
        vec![0, 4, 5, 3],
        vec![0, 6, 4],
        vec![4, 6, 1],
        vec![4, 1, 2],
        vec![4, 2, 3],
        vec![4, 3, 5],
    ];
    FoldedState { verts, faces }
}

fn prelim_base() -> FoldedState {
    let verts = vec![
        [0.0, 0.75, 0.18],
        [0.52, 0.0, 0.0],
        [0.0, -0.75, 0.0],
        [-0.52, 0.0, 0.0],
        [0.0, 0.0, 0.08],
        [-0.36, 0.0, 0.12],
        [0.36, 0.0, 0.12],
        [0.0, 0.75, 0.0],
    ];
    let faces = vec![
        vec![7, 4, 5, 3],
        vec![7, 6, 4],
        vec![0, 7, 6],
        vec![4, 6, 1],
        vec![4, 1, 2],
        vec![4, 2, 3],
        vec![4, 3, 5],
    ];
    FoldedState { verts, faces }
}

fn open_flap() -> FoldedState {
    let mut s = prelim_base();
    s.verts.push([0.28, 0.42, 0.22]);
    s.verts.push([0.0, 0.55, 0.28]);
    s.faces.push(vec![0, 8, 9]);
    s.faces.push(vec![6, 8, 0]);
    s
}

fn squash_mid() -> FoldedState {
    let verts = vec![
        [0.0, 0.72, 0.2],
        [0.48, 0.05, 0.05],
        [0.0, -0.72, 0.0],
        [-0.48, 0.05, 0.05],
        [0.0, 0.05, 0.1],
        [-0.32, 0.05, 0.12],
        [0.32, 0.05, 0.12],
        [0.22, 0.38, 0.24],
        [-0.22, 0.38, 0.24],
        [0.0, 0.52, 0.3],
    ];
    let faces = vec![
        vec![0, 9, 7, 6, 4],
        vec![0, 8, 9],
        vec![0, 5, 8, 3],
        vec![4, 5, 3],
        vec![4, 3, 2],
        vec![4, 2, 1],
        vec![4, 1, 6],
        vec![7, 9, 8],
    ];
    FoldedState { verts, faces }
}

fn squash_flat() -> FoldedState {
    let verts = vec![
        [0.0, 0.68, 0.22],
        [0.45, 0.08, 0.06],
        [0.0, -0.68, 0.0],
        [-0.45, 0.08, 0.06],
        [0.0, 0.08, 0.12],
        [-0.3, 0.08, 0.14],
        [0.3, 0.08, 0.14],
        [0.18, 0.32, 0.26],
        [-0.18, 0.32, 0.26],
        [0.0, 0.44, 0.32],
        [0.12, 0.22, 0.2],
        [-0.12, 0.22, 0.2],
    ];
    let faces = vec![
        vec![0, 9, 7, 10, 6, 4],
        vec![0, 8, 11, 9],
        vec![4, 5, 11, 8, 3],
        vec![4, 3, 2],
        vec![4, 2, 1],
        vec![4, 1, 10, 6],
        vec![7, 10, 11, 8],
    ];
    FoldedState { verts, faces }
}

fn turn_over() -> FoldedState {
    let mut s = squash_flat();
    for v in &mut s.verts {
        v[2] = -v[2];
    }
    s
}

fn petal_prep() -> FoldedState {
    let verts = vec![
        [0.0, 0.95, 0.0],
        [0.22, 0.42, 0.08],
        [0.0, -0.82, 0.0],
        [-0.22, 0.42, 0.08],
        [0.0, 0.42, 0.05],
        [-0.38, 0.42, 0.02],
        [0.38, 0.42, 0.02],
        [0.0, 0.62, 0.12],
    ];
    let faces = vec![
        vec![0, 7, 4],
        vec![0, 5, 7, 3],
        vec![0, 6, 7],
        vec![4, 5, 3],
        vec![4, 3, 2],
        vec![4, 2, 1],
        vec![4, 1, 6],
        vec![7, 6, 1],
    ];
    FoldedState { verts, faces }
}

fn petal_fold() -> FoldedState {
    let verts = vec![
        [0.0, 1.05, 0.05],
        [0.18, 0.48, 0.1],
        [0.0, -0.78, 0.0],
        [-0.18, 0.48, 0.1],
        [0.0, 0.48, 0.08],
        [-0.28, 0.52, 0.06],
        [0.28, 0.52, 0.06],
        [0.0, 0.72, 0.18],
        [0.0, 0.88, 0.12],
    ];
    let faces = vec![
        vec![8, 0, 7, 4],
        vec![0, 5, 7, 3],
        vec![0, 6, 7],
        vec![4, 5, 3],
        vec![4, 3, 2],
        vec![4, 2, 1],
        vec![4, 1, 6],
        vec![7, 6, 1],
        vec![8, 7, 5],
    ];
    FoldedState { verts, faces }
}

fn bird_base() -> FoldedState {
    let verts = vec![
        [0.0, 1.0, 0.0],
        [0.14, 0.52, 0.08],
        [-0.14, 0.52, 0.08],
        [0.0, 0.52, 0.06],
        [0.0, -0.72, 0.0],
        [-0.12, -0.55, 0.04],
        [0.12, -0.55, 0.04],
        [0.0, -0.55, 0.02],
    ];
    let faces = vec![
        vec![0, 3, 2],
        vec![0, 1, 3],
        vec![3, 2, 5, 7],
        vec![3, 7, 6, 1],
        vec![7, 5, 4],
        vec![7, 4, 6],
        vec![2, 5, 4],
        vec![1, 6, 4],
    ];
    FoldedState { verts, faces }
}

fn narrow_legs() -> FoldedState {
    let verts = vec![
        [0.0, 1.0, 0.0],
        [0.1, 0.52, 0.08],
        [-0.1, 0.52, 0.08],
        [0.0, 0.52, 0.06],
        [0.0, -0.72, 0.0],
        [-0.06, -0.58, 0.05],
        [0.06, -0.58, 0.05],
        [0.0, -0.58, 0.04],
    ];
    let faces = vec![
        vec![0, 3, 2],
        vec![0, 1, 3],
        vec![3, 2, 5, 7],
        vec![3, 7, 6, 1],
        vec![7, 5, 4],
        vec![7, 4, 6],
        vec![2, 5, 4],
        vec![1, 6, 4],
    ];
    FoldedState { verts, faces }
}

fn reverse_head_tail() -> FoldedState {
    let verts = vec![
        [0.0, 0.55, 0.22],
        [0.1, 0.48, 0.1],
        [-0.1, 0.48, 0.1],
        [0.0, 0.48, 0.08],
        [0.0, -0.55, 0.22],
        [-0.08, 0.38, 0.18],
        [0.08, 0.38, 0.18],
        [0.0, 0.38, 0.16],
        [0.0, 0.72, 0.05],
        [0.0, -0.82, 0.05],
    ];
    let faces = vec![
        vec![8, 0, 7, 3],
        vec![0, 5, 7, 2],
        vec![0, 6, 7, 1],
        vec![3, 7, 6, 1],
        vec![3, 1, 2, 5],
        vec![4, 9, 5, 2],
        vec![4, 1, 6, 9],
        vec![7, 5, 2],
        vec![7, 1, 6],
    ];
    FoldedState { verts, faces }
}

fn head_tail_up() -> FoldedState {
    let verts = vec![
        [0.0, 0.62, 0.32],
        [0.12, 0.48, 0.12],
        [-0.12, 0.48, 0.12],
        [0.0, 0.48, 0.1],
        [0.0, -0.48, 0.12],
        [0.0, 0.82, 0.08],
        [0.0, -0.85, 0.08],
        [0.08, 0.58, 0.28],
        [-0.08, 0.58, 0.28],
    ];
    let faces = vec![
        vec![5, 0, 8, 3],
        vec![0, 7, 8, 2],
        vec![0, 1, 7, 3],
        vec![3, 7, 1],
        vec![3, 2, 8],
        vec![4, 6, 2, 8],
        vec![4, 1, 7, 6],
        vec![0, 8, 2],
        vec![0, 1, 7],
    ];
    FoldedState { verts, faces }
}

fn wings_down() -> FoldedState {
    let verts = vec![
        [0.0, 0.35, 0.18],
        [0.55, 0.22, 0.02],
        [-0.55, 0.22, 0.02],
        [0.0, 0.22, 0.08],
        [0.0, -0.42, 0.1],
        [0.0, 0.72, 0.06],
        [0.0, -0.78, 0.06],
        [0.1, 0.62, 0.12],
        [0.0, 0.58, 0.1],
    ];
    let faces = vec![
        vec![5, 8, 7, 3],
        vec![3, 7, 0],
        vec![3, 0, 4],
        vec![3, 4, 6],
        vec![3, 1, 0],
        vec![3, 2, 0],
        vec![1, 0, 4],
        vec![2, 0, 4],
        vec![5, 3, 7],
    ];
    FoldedState { verts, faces }
}

fn finished() -> FoldedState {
    let verts = vec![
        [0.0, 0.28, 0.16],
        [0.62, 0.18, -0.02],
        [-0.62, 0.18, -0.02],
        [0.0, 0.18, 0.1],
        [0.0, -0.38, 0.08],
        [0.0, 0.78, 0.04],
        [0.0, -0.82, 0.04],
        [0.14, 0.68, 0.1],
        [0.06, 0.72, 0.08],
        [0.18, 0.62, 0.06],
    ];
    let faces = vec![
        vec![5, 9, 8, 7, 3],
        vec![3, 7, 0],
        vec![3, 0, 4],
        vec![3, 4, 6],
        vec![3, 1, 0],
        vec![3, 2, 0],
        vec![1, 0, 4],
        vec![2, 0, 4],
        vec![8, 9, 7],
        vec![5, 8, 9],
    ];
    FoldedState { verts, faces }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classic_cp_valid() {
        let cp = crease_pattern();
        assert!(cp.verts.len() >= 12);
        assert!(cp.hinge_indices().count() >= 20);
    }

    #[test]
    fn fifteen_stages() {
        assert_eq!(fold_stages().len(), 15);
    }
}
