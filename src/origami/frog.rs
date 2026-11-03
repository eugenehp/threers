//! Classic 17-step origami frog (green/white instructional diagram).
//!
//! Preliminary base → squash → bird base → splayed legs → head fold.
//! Traditional origami: 3D stages are hand-guided poses, not rigid-fold solver output.

use super::fold::FoldedState;
use super::pattern::{CreasePattern, Edge, EdgeKind};
use super::V2;

/// One illustrated fold step.
#[derive(Debug, Clone)]
pub struct FrogStage {
    pub label: &'static str,
    pub folded: FoldedState,
}

/// Crease pattern on a unit square (frog from bird base + leg reverse folds).
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

    let k_tl = v2(0.25, 0.75);
    let k_tr = v2(0.75, 0.75);
    let k_br = v2(0.75, 0.25);
    let k_bl = v2(0.25, 0.25);
    let mid_t = v2(0.5, 0.75);
    let mid_b = v2(0.5, 0.25);

    // Front / back leg reverse-fold points.
    let fl = v2(0.35, 0.85);
    let fr = v2(0.65, 0.85);
    let bl_leg = v2(0.3, 0.15);
    let br_leg = v2(0.7, 0.15);
    let head = v2(0.5, 0.92);

    let verts = vec![
        bl, br, tr, tl, c, bm, rm, tm, lm, k_tl, k_tr, k_br, k_bl, mid_t, mid_b, fl, fr, bl_leg,
        br_leg, head,
    ];
    let mut edges = Vec::new();

    bound(&mut edges, 0, 1);
    bound(&mut edges, 1, 2);
    bound(&mut edges, 2, 3);
    bound(&mut edges, 3, 0);

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

    hinge(&mut edges, 9, 13);
    hinge(&mut edges, 10, 13);
    hinge(&mut edges, 11, 14);
    hinge(&mut edges, 12, 14);
    hinge(&mut edges, 13, 4);
    hinge(&mut edges, 14, 4);
    hinge(&mut edges, 9, 3);
    hinge(&mut edges, 10, 2);
    hinge(&mut edges, 11, 1);
    hinge(&mut edges, 12, 0);
    hinge(&mut edges, 8, 7);
    hinge(&mut edges, 6, 7);

    hinge(&mut edges, 7, 15);
    hinge(&mut edges, 7, 16);
    hinge(&mut edges, 15, 13);
    hinge(&mut edges, 16, 13);
    hinge(&mut edges, 5, 17);
    hinge(&mut edges, 5, 18);
    hinge(&mut edges, 17, 14);
    hinge(&mut edges, 18, 14);
    hinge(&mut edges, 7, 19);
    hinge(&mut edges, 19, 13);

    CreasePattern::new(verts, edges)
}

/// Seventeen poses matching the instructional strip.
pub fn fold_stages() -> Vec<FrogStage> {
    vec![
        stage("1 — diamond", diamond()),
        stage("2 — fold down", fold_down()),
        stage("3 — fold across", fold_across()),
        stage("4 — open flap", open_flap()),
        stage("5 — squash", squash()),
        stage("6 — turn over", turn_over()),
        stage("7 — square base", square_base()),
        stage("8 — sides in", sides_in()),
        stage("9 — petal prep", petal_prep()),
        stage("10 — petal fold", petal_fold()),
        stage("11 — bird base", bird_base()),
        stage("12 — narrow", narrow()),
        stage("13 — front legs", front_legs()),
        stage("14 — back legs", back_legs()),
        stage("15 — feet", feet()),
        stage("16 — head", head_fold()),
        stage("17 — frog", finished()),
    ]
}

fn stage(label: &'static str, folded: FoldedState) -> FrogStage {
    FrogStage { label, folded }
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

fn diamond() -> FoldedState {
    let s = 0.9;
    FoldedState {
        verts: vec![
            [0.0, s, 0.0],
            [s, 0.0, 0.0],
            [0.0, -s, 0.0],
            [-s, 0.0, 0.0],
        ],
        faces: vec![vec![0, 1, 2], vec![0, 2, 3]],
    }
}

fn fold_down() -> FoldedState {
    let s = 0.88;
    FoldedState {
        verts: vec![
            [0.0, 0.0, 0.0],
            [s, 0.0, 0.0],
            [0.0, -s, 0.0],
            [-s, 0.0, 0.0],
            [0.0, 0.0, 0.06],
        ],
        faces: vec![
            vec![4, 1, 2],
            vec![4, 2, 3],
            vec![0, 1, 4],
            vec![0, 4, 3],
        ],
    }
}

fn fold_across() -> FoldedState {
    FoldedState {
        verts: vec![
            [0.0, 0.0, 0.08],
            [0.0, 0.0, 0.0],
            [0.0, -0.82, 0.0],
            [-0.62, 0.0, 0.0],
            [0.0, 0.0, 0.12],
            [-0.35, -0.35, 0.06],
        ],
        faces: vec![
            vec![0, 4, 5, 3],
            vec![4, 1, 2, 5],
            vec![0, 3, 5],
            vec![1, 5, 2],
        ],
    }
}

fn open_flap() -> FoldedState {
    let mut s = fold_across();
    s.verts.push([0.28, -0.2, 0.18]);
    s.verts.push([0.0, -0.35, 0.22]);
    s.faces.push(vec![1, 6, 7]);
    s.faces.push(vec![1, 7, 2]);
    s
}

fn squash() -> FoldedState {
    FoldedState {
        verts: vec![
            [0.0, 0.55, 0.1],
            [0.42, 0.0, 0.05],
            [0.0, -0.55, 0.0],
            [-0.42, 0.0, 0.05],
            [0.0, 0.0, 0.12],
            [0.22, -0.22, 0.16],
            [-0.22, -0.22, 0.08],
        ],
        faces: vec![
            vec![0, 4, 3],
            vec![0, 1, 4],
            vec![4, 1, 5, 2],
            vec![4, 2, 6, 3],
            vec![5, 1, 2],
            vec![6, 2, 3],
        ],
    }
}

fn turn_over() -> FoldedState {
    let mut s = squash();
    for v in &mut s.verts {
        v[2] = -v[2];
    }
    s
}

fn square_base() -> FoldedState {
    FoldedState {
        verts: vec![
            [0.0, 0.58, 0.14],
            [0.45, 0.0, 0.06],
            [0.0, -0.58, 0.0],
            [-0.45, 0.0, 0.06],
            [0.0, 0.0, 0.1],
            [0.2, 0.2, 0.14],
            [-0.2, 0.2, 0.14],
            [0.2, -0.2, 0.08],
            [-0.2, -0.2, 0.08],
        ],
        faces: vec![
            vec![0, 5, 4, 6],
            vec![0, 1, 5],
            vec![0, 6, 3],
            vec![4, 5, 1, 7, 2],
            vec![4, 2, 8, 3, 6],
            vec![7, 1, 2],
            vec![8, 2, 3],
        ],
    }
}

fn sides_in() -> FoldedState {
    FoldedState {
        verts: vec![
            [0.0, 0.62, 0.12],
            [0.22, 0.12, 0.08],
            [0.0, -0.55, 0.0],
            [-0.22, 0.12, 0.08],
            [0.0, 0.12, 0.1],
            [-0.38, 0.12, 0.04],
            [0.38, 0.12, 0.04],
        ],
        faces: vec![
            vec![0, 4, 3],
            vec![0, 1, 4],
            vec![4, 5, 3],
            vec![4, 3, 2],
            vec![4, 2, 1],
            vec![4, 1, 6],
            vec![0, 6, 1],
        ],
    }
}

fn petal_prep() -> FoldedState {
    let mut s = sides_in();
    s.verts.push([0.0, 0.35, 0.18]);
    s.faces.push(vec![0, 7, 4]);
    s
}

fn petal_fold() -> FoldedState {
    FoldedState {
        verts: vec![
            [0.0, 0.95, 0.05],
            [0.15, 0.4, 0.1],
            [0.0, -0.7, 0.0],
            [-0.15, 0.4, 0.1],
            [0.0, 0.4, 0.08],
            [-0.28, 0.45, 0.06],
            [0.28, 0.45, 0.06],
            [0.0, 0.7, 0.14],
        ],
        faces: vec![
            vec![0, 7, 4],
            vec![0, 5, 7, 3],
            vec![0, 6, 7],
            vec![4, 5, 3],
            vec![4, 3, 2],
            vec![4, 2, 1],
            vec![4, 1, 6],
            vec![7, 6, 1],
        ],
    }
}

fn bird_base() -> FoldedState {
    FoldedState {
        verts: vec![
            [0.0, 0.95, 0.0],
            [0.12, 0.45, 0.08],
            [-0.12, 0.45, 0.08],
            [0.0, 0.45, 0.06],
            [0.0, -0.72, 0.0],
            [-0.1, -0.52, 0.04],
            [0.1, -0.52, 0.04],
        ],
        faces: vec![
            vec![0, 3, 2],
            vec![0, 1, 3],
            vec![3, 2, 5],
            vec![3, 5, 4, 6, 1],
            vec![1, 6, 4],
            vec![2, 5, 4],
        ],
    }
}

fn narrow() -> FoldedState {
    FoldedState {
        verts: vec![
            [0.0, 0.95, 0.0],
            [0.08, 0.45, 0.08],
            [-0.08, 0.45, 0.08],
            [0.0, 0.45, 0.06],
            [0.0, -0.72, 0.0],
            [-0.06, -0.55, 0.05],
            [0.06, -0.55, 0.05],
        ],
        faces: vec![
            vec![0, 3, 2],
            vec![0, 1, 3],
            vec![3, 2, 5],
            vec![3, 5, 4, 6, 1],
            vec![1, 6, 4],
            vec![2, 5, 4],
        ],
    }
}

fn front_legs() -> FoldedState {
    FoldedState {
        verts: vec![
            [0.0, 0.55, 0.12],
            [0.08, 0.35, 0.1],
            [-0.08, 0.35, 0.1],
            [0.0, 0.35, 0.08],
            [0.0, -0.55, 0.05],
            [-0.42, 0.55, 0.06],
            [0.42, 0.55, 0.06],
            [-0.22, 0.48, 0.1],
            [0.22, 0.48, 0.1],
        ],
        faces: vec![
            vec![0, 3, 2],
            vec![0, 1, 3],
            vec![0, 7, 2],
            vec![0, 1, 8],
            vec![7, 5, 2],
            vec![8, 1, 6],
            vec![3, 2, 4],
            vec![3, 4, 1],
        ],
    }
}

fn back_legs() -> FoldedState {
    let mut s = front_legs();
    s.verts.extend_from_slice(&[
        [-0.38, -0.48, 0.08],
        [0.38, -0.48, 0.08],
        [-0.18, -0.42, 0.1],
        [0.18, -0.42, 0.1],
    ]);
    s.faces.push(vec![4, 11, 9]);
    s.faces.push(vec![4, 10, 12]);
    s.faces.push(vec![2, 4, 11]);
    s.faces.push(vec![1, 12, 4]);
    s
}

fn feet() -> FoldedState {
    let mut s = back_legs();
    // Outward foot tips on all four legs.
    s.verts.extend_from_slice(&[
        [-0.55, 0.62, 0.02],
        [0.55, 0.62, 0.02],
        [-0.52, -0.55, 0.02],
        [0.52, -0.55, 0.02],
    ]);
    let n = s.verts.len();
    s.faces.push(vec![5, n - 4, 7]);
    s.faces.push(vec![6, 8, n - 3]);
    s.faces.push(vec![9, n - 2, 11]);
    s.faces.push(vec![10, 12, n - 1]);
    s
}

fn head_fold() -> FoldedState {
    let mut s = feet();
    s.verts.push([0.0, 0.42, 0.18]);
    s.verts.push([0.0, 0.48, 0.22]);
    let n = s.verts.len();
    s.faces.push(vec![0, n - 1, n - 2]);
    s.faces.push(vec![0, n - 2, 3]);
    s
}

fn finished() -> FoldedState {
    // Top-down frog: body diamond, four splayed legs with feet, head tip folded.
    FoldedState {
        verts: vec![
            // 0 body centre
            [0.0, 0.0, 0.12],
            // 1–4 body diamond (head / left / rear / right)
            [0.0, 0.38, 0.14],
            [-0.28, 0.0, 0.1],
            [0.0, -0.32, 0.1],
            [0.28, 0.0, 0.1],
            // 5 head tip (folded down)
            [0.0, 0.52, 0.18],
            // 6–7 eyes (raised bumps)
            [-0.08, 0.42, 0.2],
            [0.08, 0.42, 0.2],
            // front-left leg
            [-0.35, 0.32, 0.08],
            [-0.55, 0.48, 0.04],
            [-0.62, 0.38, 0.02],
            // front-right leg
            [0.35, 0.32, 0.08],
            [0.55, 0.48, 0.04],
            [0.62, 0.38, 0.02],
            // back-left leg
            [-0.32, -0.28, 0.08],
            [-0.52, -0.42, 0.04],
            [-0.58, -0.32, 0.02],
            // back-right leg
            [0.32, -0.28, 0.08],
            [0.52, -0.42, 0.04],
            [0.58, -0.32, 0.02],
            // belly underside
            [0.0, 0.0, 0.02],
        ],
        faces: vec![
            // body
            vec![0, 1, 2],
            vec![0, 2, 3],
            vec![0, 3, 4],
            vec![0, 4, 1],
            vec![20, 2, 1],
            vec![20, 3, 2],
            vec![20, 4, 3],
            vec![20, 1, 4],
            // head + eyes
            vec![1, 5, 6],
            vec![1, 7, 5],
            vec![1, 6, 2],
            vec![1, 4, 7],
            // front left
            vec![2, 8, 9],
            vec![8, 10, 9],
            vec![2, 1, 8],
            // front right
            vec![4, 12, 11],
            vec![11, 12, 13],
            vec![4, 11, 1],
            // back left
            vec![2, 14, 15],
            vec![14, 16, 15],
            vec![2, 3, 14],
            // back right
            vec![4, 18, 17],
            vec![17, 18, 19],
            vec![4, 3, 17],
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frog_cp_valid() {
        let cp = crease_pattern();
        assert!(cp.verts.len() >= 16);
        assert!(cp.hinge_indices().count() >= 24);
    }

    #[test]
    fn seventeen_stages() {
        assert_eq!(fold_stages().len(), 17);
        for s in fold_stages() {
            assert!(!s.folded.faces.is_empty(), "{}", s.label);
        }
    }
}
