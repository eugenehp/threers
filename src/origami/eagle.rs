//! Origami Eagle — standing bird with accordion wings and hooked beak.
//!
//! Based on the [origamiok.com](https://origamiok.com) photographic tutorial
//! (20 cm square → ~13 cm tall). Traditional origami: 3D stages are hand-guided
//! poses matching the photo strip, not rigid-fold solver output.

use super::fold::FoldedState;
use super::pattern::{CreasePattern, Edge, EdgeKind};
use super::V2;

/// One illustrated fold step.
#[derive(Debug, Clone)]
pub struct EagleStage {
    pub label: &'static str,
    pub folded: FoldedState,
}

/// Crease pattern on a unit square (preliminary + kite + wing accordion + beak).
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

    // Kite / bird-base midpoints.
    let k_tl = v2(0.25, 0.75);
    let k_tr = v2(0.75, 0.75);
    let k_br = v2(0.75, 0.25);
    let k_bl = v2(0.25, 0.25);
    let mid_t = v2(0.5, 0.75);
    let mid_b = v2(0.5, 0.25);

    // Accordion wing ribs (parallel to body axis).
    let w1l = v2(0.12, 0.55);
    let w2l = v2(0.22, 0.55);
    let w3l = v2(0.32, 0.55);
    let w1r = v2(0.88, 0.55);
    let w2r = v2(0.78, 0.55);
    let w3r = v2(0.68, 0.55);

    // Beak / head reverse-fold points.
    let beak = v2(0.5, 0.92);
    let beak_l = v2(0.42, 0.85);
    let beak_r = v2(0.58, 0.85);

    // Tail fan.
    let t1 = v2(0.35, 0.08);
    let t2 = v2(0.5, 0.12);
    let t3 = v2(0.65, 0.08);

    let verts = vec![
        bl, br, tr, tl, c, bm, rm, tm, lm, k_tl, k_tr, k_br, k_bl, mid_t, mid_b, w1l, w2l, w3l,
        w1r, w2r, w3r, beak, beak_l, beak_r, t1, t2, t3,
    ];
    let mut edges = Vec::new();

    bound(&mut edges, 0, 1);
    bound(&mut edges, 1, 2);
    bound(&mut edges, 2, 3);
    bound(&mut edges, 3, 0);

    // Diagonals + centre cross (preliminary).
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

    // Kite / bird base.
    hinge(&mut edges, 8, 7);
    hinge(&mut edges, 6, 7);
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

    // Wing accordion ribs.
    hinge(&mut edges, 15, 8);
    hinge(&mut edges, 16, 8);
    hinge(&mut edges, 17, 4);
    hinge(&mut edges, 18, 6);
    hinge(&mut edges, 19, 6);
    hinge(&mut edges, 20, 4);
    hinge(&mut edges, 15, 16);
    hinge(&mut edges, 16, 17);
    hinge(&mut edges, 18, 19);
    hinge(&mut edges, 19, 20);
    hinge(&mut edges, 15, 0);
    hinge(&mut edges, 18, 1);

    // Beak reverse folds.
    hinge(&mut edges, 7, 21);
    hinge(&mut edges, 21, 22);
    hinge(&mut edges, 21, 23);
    hinge(&mut edges, 22, 13);
    hinge(&mut edges, 23, 13);

    // Tail fan.
    hinge(&mut edges, 5, 24);
    hinge(&mut edges, 5, 25);
    hinge(&mut edges, 5, 26);
    hinge(&mut edges, 24, 25);
    hinge(&mut edges, 25, 26);
    hinge(&mut edges, 24, 0);
    hinge(&mut edges, 26, 1);

    CreasePattern::new(verts, edges)
}

/// Fourteen poses matching the origamiok photographic strip.
pub fn fold_stages() -> Vec<EagleStage> {
    vec![
        stage("1 — square", flat_diamond()),
        stage("2 — diagonals", diagonals()),
        stage("3 — prelim base", prelim_base()),
        stage("4 — kite", kite()),
        stage("5 — bird base", bird_base()),
        stage("6 — open wings", open_wings()),
        stage("7 — accordion L", accordion_l()),
        stage("8 — accordion R", accordion_r()),
        stage("9 — body stand", body_stand()),
        stage("10 — beak prep", beak_prep()),
        stage("11 — hooked beak", hooked_beak()),
        stage("12 — tail fan", tail_fan()),
        stage("13 — spread", spread()),
        stage("14 — finished", finished()),
    ]
}

fn stage(label: &'static str, folded: FoldedState) -> EagleStage {
    EagleStage { label, folded }
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

fn flat_diamond() -> FoldedState {
    let s = 0.9;
    let verts = vec![
        [0.0, s, 0.0],
        [s, 0.0, 0.0],
        [0.0, -s, 0.0],
        [-s, 0.0, 0.0],
    ];
    FoldedState {
        verts,
        faces: vec![vec![0, 1, 2], vec![0, 2, 3]],
    }
}

fn diagonals() -> FoldedState {
    let s = 0.88;
    let verts = vec![
        [0.0, s, 0.0],
        [s, 0.0, 0.0],
        [0.0, -s, 0.0],
        [-s, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ];
    FoldedState {
        verts,
        faces: vec![
            vec![0, 1, 4],
            vec![1, 2, 4],
            vec![2, 3, 4],
            vec![3, 0, 4],
        ],
    }
}

fn prelim_base() -> FoldedState {
    let verts = vec![
        [0.0, 0.72, 0.16],
        [0.5, 0.0, 0.0],
        [0.0, -0.72, 0.0],
        [-0.5, 0.0, 0.0],
        [0.0, 0.0, 0.08],
        [-0.32, 0.0, 0.1],
        [0.32, 0.0, 0.1],
    ];
    FoldedState {
        verts,
        faces: vec![
            vec![0, 4, 5, 3],
            vec![0, 6, 4],
            vec![4, 6, 1],
            vec![4, 1, 2],
            vec![4, 2, 3],
            vec![4, 3, 5],
        ],
    }
}

fn kite() -> FoldedState {
    let verts = vec![
        [0.0, 0.95, 0.0],
        [0.28, 0.35, 0.06],
        [0.0, -0.78, 0.0],
        [-0.28, 0.35, 0.06],
        [0.0, 0.35, 0.04],
        [-0.42, 0.35, 0.02],
        [0.42, 0.35, 0.02],
    ];
    FoldedState {
        verts,
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

fn bird_base() -> FoldedState {
    let verts = vec![
        [0.0, 1.0, 0.0],
        [0.12, 0.48, 0.08],
        [-0.12, 0.48, 0.08],
        [0.0, 0.48, 0.06],
        [0.0, -0.7, 0.0],
        [-0.1, -0.52, 0.04],
        [0.1, -0.52, 0.04],
    ];
    FoldedState {
        verts,
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

fn open_wings() -> FoldedState {
    let verts = vec![
        [0.0, 0.55, 0.2],
        [0.72, 0.28, 0.02],
        [-0.72, 0.28, 0.02],
        [0.0, 0.28, 0.1],
        [0.0, -0.45, 0.08],
        [0.0, 0.85, 0.05],
        [0.0, -0.75, 0.04],
    ];
    FoldedState {
        verts,
        faces: vec![
            vec![5, 0, 3],
            vec![0, 1, 3],
            vec![0, 3, 2],
            vec![3, 1, 4],
            vec![3, 4, 2],
            vec![1, 4, 6],
            vec![2, 6, 4],
        ],
    }
}

fn accordion_l() -> FoldedState {
    let verts = vec![
        [0.0, 0.52, 0.22],
        [0.55, 0.32, 0.08],
        [-0.55, 0.32, 0.08],
        [0.0, 0.28, 0.12],
        [0.0, -0.42, 0.1],
        [0.0, 0.82, 0.06],
        [0.0, -0.72, 0.05],
        [-0.72, 0.38, 0.0],
        [-0.42, 0.35, 0.12],
        [-0.28, 0.33, 0.04],
    ];
    FoldedState {
        verts,
        faces: vec![
            vec![5, 0, 3],
            vec![0, 1, 3],
            vec![0, 3, 8, 2],
            vec![2, 8, 9, 7],
            vec![8, 3, 9],
            vec![3, 1, 4],
            vec![3, 4, 9],
            vec![4, 6, 1],
            vec![4, 7, 6],
            vec![9, 7, 4],
        ],
    }
}

fn accordion_r() -> FoldedState {
    let verts = vec![
        [0.0, 0.52, 0.22],
        [0.55, 0.32, 0.08],
        [-0.55, 0.32, 0.08],
        [0.0, 0.28, 0.12],
        [0.0, -0.42, 0.1],
        [0.0, 0.82, 0.06],
        [0.0, -0.72, 0.05],
        [-0.72, 0.38, 0.0],
        [-0.42, 0.35, 0.12],
        [-0.28, 0.33, 0.04],
        [0.72, 0.38, 0.0],
        [0.42, 0.35, 0.12],
        [0.28, 0.33, 0.04],
    ];
    FoldedState {
        verts,
        faces: vec![
            vec![5, 0, 3],
            vec![0, 11, 3],
            vec![0, 3, 8, 2],
            vec![2, 8, 9, 7],
            vec![8, 3, 9],
            vec![1, 10, 11],
            vec![1, 11, 12, 3],
            vec![3, 12, 4],
            vec![3, 4, 9],
            vec![4, 6, 10, 1],
            vec![4, 7, 6],
            vec![9, 7, 4],
            vec![12, 10, 4],
        ],
    }
}

fn body_stand() -> FoldedState {
    let verts = vec![
        [0.0, 0.48, 0.35],
        [0.58, 0.22, 0.1],
        [-0.58, 0.22, 0.1],
        [0.0, 0.18, 0.2],
        [0.0, -0.35, 0.18],
        [0.0, 0.78, 0.12],
        [0.0, -0.68, 0.08],
        [-0.78, 0.28, 0.02],
        [-0.48, 0.25, 0.16],
        [0.78, 0.28, 0.02],
        [0.48, 0.25, 0.16],
        [-0.12, -0.15, 0.05],
        [0.12, -0.15, 0.05],
    ];
    FoldedState {
        verts,
        faces: vec![
            vec![5, 0, 3],
            vec![0, 10, 3],
            vec![0, 3, 8, 2],
            vec![2, 8, 7],
            vec![1, 9, 10],
            vec![1, 10, 3],
            vec![3, 10, 12, 4],
            vec![3, 4, 11, 8],
            vec![4, 6, 12],
            vec![4, 11, 6],
            vec![8, 7, 11],
            vec![10, 9, 12],
        ],
    }
}

fn beak_prep() -> FoldedState {
    let mut s = body_stand();
    s.verts.push([0.08, 0.72, 0.18]);
    s.verts.push([-0.05, 0.75, 0.14]);
    s.faces.push(vec![5, 13, 14]);
    s.faces.push(vec![5, 0, 13]);
    s
}

fn hooked_beak() -> FoldedState {
    let verts = vec![
        [0.0, 0.45, 0.38],
        [0.58, 0.2, 0.1],
        [-0.58, 0.2, 0.1],
        [0.0, 0.16, 0.22],
        [0.0, -0.35, 0.18],
        [0.12, 0.72, 0.2],
        [0.0, -0.68, 0.08],
        [-0.78, 0.26, 0.02],
        [-0.48, 0.23, 0.16],
        [0.78, 0.26, 0.02],
        [0.48, 0.23, 0.16],
        [-0.12, -0.15, 0.05],
        [0.12, -0.15, 0.05],
        [0.22, 0.68, 0.14],
        [0.05, 0.78, 0.1],
    ];
    FoldedState {
        verts,
        faces: vec![
            vec![14, 5, 13, 0],
            vec![5, 0, 3],
            vec![0, 10, 3],
            vec![0, 3, 8, 2],
            vec![2, 8, 7],
            vec![1, 9, 10],
            vec![1, 10, 3],
            vec![3, 10, 12, 4],
            vec![3, 4, 11, 8],
            vec![4, 6, 12],
            vec![4, 11, 6],
            vec![8, 7, 11],
            vec![10, 9, 12],
            vec![14, 13, 5],
        ],
    }
}

fn tail_fan() -> FoldedState {
    let mut s = hooked_beak();
    s.verts.extend_from_slice(&[
        [-0.22, -0.72, 0.04],
        [0.0, -0.78, 0.1],
        [0.22, -0.72, 0.04],
    ]);
    let n = s.verts.len();
    s.faces.push(vec![6, n - 3, n - 2]);
    s.faces.push(vec![6, n - 2, n - 1]);
    s.faces.push(vec![11, 6, n - 3]);
    s.faces.push(vec![12, n - 1, 6]);
    s
}

fn spread() -> FoldedState {
    let verts = vec![
        [0.0, 0.42, 0.4],
        [0.62, 0.18, 0.08],
        [-0.62, 0.18, 0.08],
        [0.0, 0.12, 0.24],
        [0.0, -0.32, 0.2],
        [0.18, 0.7, 0.22],
        [0.0, -0.55, 0.12],
        [-0.85, 0.22, -0.02],
        [-0.52, 0.2, 0.14],
        [0.85, 0.22, -0.02],
        [0.52, 0.2, 0.14],
        [-0.14, -0.12, 0.06],
        [0.14, -0.12, 0.06],
        [0.28, 0.65, 0.12],
        [0.08, 0.76, 0.08],
        [-0.28, -0.75, 0.02],
        [0.0, -0.82, 0.08],
        [0.28, -0.75, 0.02],
        [-0.35, 0.22, 0.02],
        [0.35, 0.22, 0.02],
    ];
    FoldedState {
        verts,
        faces: vec![
            vec![14, 5, 13, 0],
            vec![5, 0, 3],
            vec![0, 10, 3],
            vec![0, 3, 8, 2],
            vec![2, 8, 18, 7],
            vec![1, 9, 19, 10],
            vec![1, 10, 3],
            vec![3, 10, 12, 4],
            vec![3, 4, 11, 8],
            vec![4, 6, 12],
            vec![4, 11, 6],
            vec![8, 7, 11],
            vec![10, 9, 12],
            vec![14, 13, 5],
            vec![6, 15, 16],
            vec![6, 16, 17],
            vec![11, 6, 15],
            vec![12, 17, 6],
            vec![18, 8, 11],
            vec![19, 12, 10],
        ],
    }
}

fn finished() -> FoldedState {
    // Standing eagle: +Y up, wings along ±X, beak at top, fan tail at bottom.
    // Faces are a manifold paper net (each interior crease shared by ≤2 panels).
    let verts = vec![
        // 0 chest / body centre
        [0.0, 0.05, 0.28],
        // 1 back
        [0.0, 0.15, 0.08],
        // 2 belly
        [0.0, -0.15, 0.35],
        // 3–4 shoulder
        [-0.18, 0.22, 0.2],
        [0.18, 0.22, 0.2],
        // 5–6 hip
        [-0.12, -0.25, 0.22],
        [0.12, -0.25, 0.22],
        // 7–11 left wing accordion (shoulder → tip)
        [-0.35, 0.28, 0.18],
        [-0.48, 0.22, 0.08],
        [-0.62, 0.3, 0.16],
        [-0.78, 0.18, 0.04],
        [-0.95, 0.25, 0.1],
        // 12–16 right wing accordion
        [0.35, 0.28, 0.18],
        [0.48, 0.22, 0.08],
        [0.62, 0.3, 0.16],
        [0.78, 0.18, 0.04],
        [0.95, 0.25, 0.1],
        // 17 neck
        [0.0, 0.42, 0.22],
        // 18–20 hooked beak (outside reverse fold)
        [0.0, 0.58, 0.28],
        [0.14, 0.62, 0.18],
        [0.06, 0.52, 0.12],
        // 21 crest
        [-0.04, 0.68, 0.2],
        // 22–24 fan tail
        [-0.28, -0.55, 0.12],
        [0.0, -0.62, 0.22],
        [0.28, -0.55, 0.12],
        // 25–26 stand / feet
        [-0.08, -0.38, 0.02],
        [0.08, -0.38, 0.02],
    ];
    FoldedState {
        verts,
        faces: vec![
            // body: chest diamond, left/right flanks, back plate
            vec![0, 3, 4],
            vec![0, 4, 6, 2],
            vec![0, 2, 5, 3],
            vec![1, 3, 5, 6, 4],
            // left wing accordion strip
            vec![3, 7, 8],
            vec![7, 8, 9],
            vec![8, 9, 10],
            vec![9, 10, 11],
            vec![5, 11, 10, 8],
            // right wing accordion strip
            vec![4, 13, 12],
            vec![12, 13, 14],
            vec![13, 14, 15],
            vec![14, 15, 16],
            vec![6, 16, 15, 13],
            // head / hooked beak (attaches along back top edge 3–1–4)
            vec![1, 17, 3],
            vec![1, 4, 17],
            vec![17, 18, 21],
            vec![17, 0, 20, 21],
            vec![18, 19, 20, 21],
            vec![18, 17, 20],
            // belly / feet
            vec![2, 5, 25],
            vec![2, 26, 6],
            vec![2, 25, 26],
            // fan tail
            vec![5, 22, 23, 25],
            vec![6, 26, 23, 24],
            vec![22, 23, 24],
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eagle_cp_valid() {
        let cp = crease_pattern();
        assert!(cp.verts.len() >= 20);
        assert!(cp.hinge_indices().count() >= 30);
    }

    #[test]
    fn fourteen_stages() {
        assert_eq!(fold_stages().len(), 14);
        for s in fold_stages() {
            assert!(!s.folded.faces.is_empty(), "{}", s.label);
            assert!(s.folded.verts.len() >= 4, "{}", s.label);
        }
    }
}
