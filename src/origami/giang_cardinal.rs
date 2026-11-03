//! Giang Dinh's Northern Cardinal rough diagram (2014).
//!
//! Single-square model from [giangdinh.com](https://giangdinh.com/2014/northern-cardinal/cardinal-rough-dia/).
//! Traditional wet-fold origami — not globally rigid-foldable — so 3D stages are
//! hand-guided poses matching the photo sequence in the diagram.

use super::fold::FoldedState;
use super::pattern::{CreasePattern, Edge, EdgeKind};
use super::V2;

/// One step from the rough diagram photo strip.
#[derive(Debug, Clone)]
pub struct GiangCardinalStage {
    pub label: &'static str,
    pub folded: FoldedState,
}

/// Traced crease pattern (diamond-square, unit half-diagonal 1).
pub fn crease_pattern() -> CreasePattern {
    // Diamond square: top N, right E, bottom S, left W.
    let n = v2(0.0, 1.0);
    let e = v2(1.0, 0.0);
    let s = v2(0.0, -1.0);
    let w = v2(-1.0, 0.0);
    let o = v2(0.0, 0.0);
    let h = v2(0.0, 0.34);
    let m = v2(0.0, 0.56);
    let f = v2(0.13, 0.67);
    let bl = v2(-0.16, 0.54);
    let br = v2(0.05, 0.52);
    let cl = v2(-0.24, 0.86);
    let cr = v2(0.24, 0.86);
    let el = v2(-0.55, 0.38);
    let er = v2(0.55, 0.38);
    let wl = v2(-0.68, -0.06);
    let wr = v2(0.68, -0.06);
    let leg = v2(0.0, -0.42);
    let tl = v2(-0.16, -0.68);
    let tr = v2(0.16, -0.68);

    let verts = vec![
        n, e, s, w, o, h, m, f, bl, br, cl, cr, el, er, wl, wr, leg, tl, tr,
    ];
    let mut edges = Vec::new();

    bound(&mut edges, 0, 1);
    bound(&mut edges, 1, 2);
    bound(&mut edges, 2, 3);
    bound(&mut edges, 3, 0);

    // Axes and preliminaries (steps 1–2).
    hinge(&mut edges, 0, 4);
    hinge(&mut edges, 1, 4);
    hinge(&mut edges, 2, 4);
    hinge(&mut edges, 3, 4);
    hinge(&mut edges, 0, 5);
    hinge(&mut edges, 1, 5);
    hinge(&mut edges, 3, 5);
    hinge(&mut edges, 1, 13);
    hinge(&mut edges, 3, 12);

    // Head, crest, black mask (steps 3–4).
    hinge(&mut edges, 0, 6);
    hinge(&mut edges, 6, 7);
    hinge(&mut edges, 7, 8);
    hinge(&mut edges, 7, 9);
    hinge(&mut edges, 8, 6);
    hinge(&mut edges, 9, 6);
    hinge(&mut edges, 0, 10);
    hinge(&mut edges, 0, 11);
    hinge(&mut edges, 10, 11);
    hinge(&mut edges, 10, 12);
    hinge(&mut edges, 11, 13);

    // Body, wing-or-leg option, tail (steps 5–8).
    hinge(&mut edges, 4, 5);
    hinge(&mut edges, 4, 16);
    hinge(&mut edges, 2, 16);
    hinge(&mut edges, 14, 16);
    hinge(&mut edges, 15, 16);
    hinge(&mut edges, 14, 2);
    hinge(&mut edges, 15, 2);
    hinge(&mut edges, 2, 17);
    hinge(&mut edges, 2, 18);
    hinge(&mut edges, 17, 18);
    hinge(&mut edges, 17, 16);
    hinge(&mut edges, 18, 16);

    CreasePattern::new(verts, edges)
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

/// Eight poses following Giang's photo strip (flat → standing cardinal).
pub fn fold_stages() -> Vec<GiangCardinalStage> {
    vec![
        stage("1 — square", flat_diamond()),
        stage("2 — preliminaries", preliminaries()),
        stage("3 — head", head_kite()),
        stage("4 — mask", mask()),
        stage("5 — wing/leg", wing_or_leg()),
        stage("6 — body", body_half()),
        stage("7 — round body", body_round()),
        stage("8 — finished", finished()),
    ]
}

fn stage(label: &'static str, folded: FoldedState) -> GiangCardinalStage {
    GiangCardinalStage { label, folded }
}

fn v2(x: f64, y: f64) -> V2 {
    [x, y]
}

fn flat_diamond() -> FoldedState {
    let s = 0.85;
    let verts = vec![
        [0.0, s, 0.0],
        [s, 0.0, 0.0],
        [0.0, -s, 0.0],
        [-s, 0.0, 0.0],
    ];
    let faces = vec![vec![0, 1, 2], vec![0, 2, 3]];
    FoldedState { verts, faces }
}

fn preliminaries() -> FoldedState {
    let verts = vec![
        [0.0, 0.72, 0.0],
        [0.62, 0.0, 0.0],
        [0.0, -0.72, 0.0],
        [-0.62, 0.0, 0.0],
        [0.0, 0.28, 0.0],
        [-0.38, 0.26, 0.06],
        [0.38, 0.26, 0.06],
        [0.0, 0.72, 0.14],
    ];
    let faces = vec![
        vec![0, 4, 5, 3],
        vec![0, 6, 4],
        vec![0, 7, 6],
        vec![4, 6, 1],
        vec![4, 1, 2],
        vec![4, 2, 3],
        vec![4, 3, 5],
    ];
    FoldedState { verts, faces }
}

fn head_kite() -> FoldedState {
    let verts = vec![
        [0.0, 0.92, 0.08],
        [0.48, 0.08, 0.0],
        [0.0, -0.72, 0.0],
        [-0.48, 0.08, 0.0],
        [0.0, 0.36, 0.02],
        [-0.22, 0.34, 0.08],
        [0.22, 0.34, 0.08],
        [0.0, 0.62, 0.18],
    ];
    let faces = vec![
        vec![7, 0, 5, 4],
        vec![7, 6, 0],
        vec![4, 5, 3],
        vec![4, 3, 2],
        vec![4, 2, 1],
        vec![4, 1, 6],
        vec![4, 6, 5],
    ];
    FoldedState { verts, faces }
}

fn mask() -> FoldedState {
    let verts = vec![
        [0.0, 0.95, 0.1],
        [0.45, 0.05, 0.0],
        [0.0, -0.72, 0.0],
        [-0.45, 0.05, 0.0],
        [0.0, 0.38, 0.04],
        [-0.2, 0.36, 0.1],
        [0.2, 0.36, 0.1],
        [0.1, 0.52, 0.2],
        [-0.06, 0.48, 0.18],
    ];
    let faces = vec![
        vec![0, 7, 8, 5, 4],
        vec![0, 6, 7],
        vec![4, 5, 3],
        vec![4, 3, 2],
        vec![4, 2, 1],
        vec![4, 1, 6],
        vec![7, 6, 4],
        vec![7, 8, 5],
    ];
    FoldedState { verts, faces }
}

fn wing_or_leg() -> FoldedState {
    let mut s = mask();
    s.verts.extend_from_slice(&[
        [-0.42, -0.08, 0.12],
        [0.42, -0.08, 0.12],
        [0.0, -0.38, 0.08],
    ]);
    s.faces.push(vec![2, 10, 9]);
    s.faces.push(vec![2, 1, 10]);
    s
}

fn body_half() -> FoldedState {
    let verts = vec![
        [0.0, 0.88, 0.12],
        [0.12, 0.48, 0.22],
        [0.0, -0.55, 0.05],
        [-0.12, 0.48, 0.22],
        [0.0, 0.28, 0.0],
        [-0.18, 0.12, -0.08],
        [0.18, 0.12, -0.08],
        [0.08, 0.52, 0.24],
        [-0.05, 0.5, 0.22],
        [0.0, -0.62, 0.02],
    ];
    let faces = vec![
        vec![0, 7, 8, 3, 4],
        vec![0, 4, 1, 7],
        vec![4, 3, 5],
        vec![4, 5, 2, 9],
        vec![4, 9, 6, 1],
        vec![1, 6, 7],
    ];
    FoldedState { verts, faces }
}

fn body_round() -> FoldedState {
    let verts = vec![
        [0.0, 0.82, 0.14],
        [0.14, 0.42, 0.26],
        [0.0, -0.48, 0.1],
        [-0.14, 0.42, 0.26],
        [0.0, 0.18, 0.06],
        [-0.22, 0.02, 0.02],
        [0.22, 0.02, 0.02],
        [0.1, 0.52, 0.28],
        [-0.06, 0.5, 0.26],
        [0.0, -0.68, 0.06],
        [-0.07, -0.05, 0.12],
        [0.07, -0.05, 0.12],
    ];
    let faces = vec![
        vec![0, 7, 8, 3, 4],
        vec![0, 4, 1, 7],
        vec![4, 3, 5, 10],
        vec![4, 10, 2, 9],
        vec![4, 9, 6, 11, 1],
        vec![1, 11, 7],
        vec![2, 10, 11, 6, 9],
    ];
    FoldedState { verts, faces }
}

fn finished() -> FoldedState {
    let verts = vec![
        [0.0, 0.78, 0.18],
        [0.1, 0.46, 0.28],
        [0.0, -0.12, 0.14],
        [-0.1, 0.46, 0.28],
        [0.0, 0.22, 0.1],
        [-0.24, 0.08, 0.04],
        [0.24, 0.08, 0.04],
        [0.12, 0.56, 0.32],
        [-0.04, 0.54, 0.3],
        [0.0, -0.72, 0.08],
        [-0.08, -0.06, 0.16],
        [0.08, -0.06, 0.16],
        [0.06, 0.48, 0.34],
    ];
    let faces = vec![
        vec![0, 7, 12, 8, 3, 4],
        vec![0, 4, 1, 12],
        vec![4, 3, 5, 10],
        vec![4, 10, 2, 11, 6, 1],
        vec![2, 10, 11],
        vec![2, 11, 9],
        vec![1, 6, 12],
        vec![12, 6, 7],
    ];
    FoldedState { verts, faces }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn giang_cp_has_faces() {
        let cp = crease_pattern();
        assert!(cp.verts.len() >= 15);
        assert!(cp.faces.len() >= 4);
    }

    #[test]
    fn eight_stages() {
        assert_eq!(fold_stages().len(), 8);
    }
}
