//! Book Cardinal from *Origami Birds* (steps 8–14 shown on the reference page).
//!
//! Red/black two-tone paper: narrow body, pulled crest, black face mask, beak.
//! Early stages (1–7) are the usual bird-base path; 8–14 match the book page.
//! Traditional origami: hand-guided poses, not rigid-fold solver output.

use super::fold::FoldedState;
use super::pattern::{CreasePattern, Edge, EdgeKind};
use super::V2;

/// One illustrated fold step.
#[derive(Debug, Clone)]
pub struct BookCardinalStage {
    pub label: &'static str,
    pub folded: FoldedState,
}

/// Crease pattern on a unit square (bird base + crest/mask/beak detail).
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

    let mid_t = v2(0.5, 0.75);
    let mid_b = v2(0.5, 0.25);
    let k_tl = v2(0.25, 0.75);
    let k_tr = v2(0.75, 0.75);
    let k_br = v2(0.75, 0.25);
    let k_bl = v2(0.25, 0.25);

    // Crest pull + black mask + beak.
    let crest = v2(0.5, 0.95);
    let neck = v2(0.5, 0.82);
    let mask_l = v2(0.42, 0.78);
    let mask_r = v2(0.58, 0.78);
    let beak = v2(0.62, 0.72);
    let beak_fold = v2(0.55, 0.74);

    let verts = vec![
        bl, br, tr, tl, c, bm, rm, tm, lm, k_tl, k_tr, k_br, k_bl, mid_t, mid_b, crest, neck,
        mask_l, mask_r, beak, beak_fold,
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

    hinge(&mut edges, 7, 15);
    hinge(&mut edges, 15, 16);
    hinge(&mut edges, 16, 17);
    hinge(&mut edges, 16, 18);
    hinge(&mut edges, 17, 13);
    hinge(&mut edges, 18, 13);
    hinge(&mut edges, 18, 20);
    hinge(&mut edges, 20, 19);
    hinge(&mut edges, 16, 20);

    CreasePattern::new(verts, edges)
}

/// Fourteen poses: bird-base prelude (1–7) then book page steps 8–14.
pub fn fold_stages() -> Vec<BookCardinalStage> {
    vec![
        stage("1 — square", square()),
        stage("2 — prelim", prelim()),
        stage("3 — kite", kite()),
        stage("4 — petal", petal()),
        stage("5 — bird base", bird_base()),
        stage("6 — narrow", narrow()),
        stage("7 — reverse", reverse_up()),
        stage("8 — fold corner", fold_corner()),
        stage("9 — mountain inside", mountain_inside()),
        stage("10 — crest + tail", crest_and_tail()),
        stage("11 — head corner", head_corner()),
        stage("12 — edge up", edge_up()),
        stage("13 — beak", beak()),
        stage("14 — finished", finished()),
    ]
}

fn stage(label: &'static str, folded: FoldedState) -> BookCardinalStage {
    BookCardinalStage { label, folded }
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

fn square() -> FoldedState {
    let s = 0.9;
    FoldedState {
        verts: vec![[-s, -s, 0.0], [s, -s, 0.0], [s, s, 0.0], [-s, s, 0.0]],
        faces: vec![vec![0, 1, 2], vec![0, 2, 3]],
    }
}

fn prelim() -> FoldedState {
    FoldedState {
        verts: vec![
            [0.0, 0.72, 0.14],
            [0.5, 0.0, 0.0],
            [0.0, -0.72, 0.0],
            [-0.5, 0.0, 0.0],
            [0.0, 0.0, 0.08],
        ],
        faces: vec![
            vec![0, 4, 3],
            vec![0, 1, 4],
            vec![4, 1, 2],
            vec![4, 2, 3],
        ],
    }
}

fn kite() -> FoldedState {
    FoldedState {
        verts: vec![
            [0.0, 0.95, 0.0],
            [0.25, 0.35, 0.06],
            [0.0, -0.75, 0.0],
            [-0.25, 0.35, 0.06],
            [0.0, 0.35, 0.04],
        ],
        faces: vec![
            vec![0, 4, 3],
            vec![0, 1, 4],
            vec![4, 3, 2],
            vec![4, 2, 1],
        ],
    }
}

fn petal() -> FoldedState {
    FoldedState {
        verts: vec![
            [0.0, 1.0, 0.04],
            [0.14, 0.42, 0.1],
            [0.0, -0.72, 0.0],
            [-0.14, 0.42, 0.1],
            [0.0, 0.42, 0.08],
            [0.0, 0.72, 0.14],
        ],
        faces: vec![
            vec![0, 5, 4],
            vec![0, 4, 3],
            vec![0, 1, 4],
            vec![4, 3, 2],
            vec![4, 2, 1],
            vec![5, 1, 4],
        ],
    }
}

fn bird_base() -> FoldedState {
    FoldedState {
        verts: vec![
            [0.0, 1.0, 0.0],
            [0.12, 0.48, 0.08],
            [-0.12, 0.48, 0.08],
            [0.0, 0.48, 0.06],
            [0.0, -0.7, 0.0],
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
            [0.0, 1.0, 0.0],
            [0.08, 0.48, 0.08],
            [-0.08, 0.48, 0.08],
            [0.0, 0.48, 0.06],
            [0.0, -0.7, 0.0],
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

fn reverse_up() -> FoldedState {
    // Horizontal bird: head left, tail right (book orients this way by step 10).
    FoldedState {
        verts: vec![
            [0.55, 0.12, 0.08],
            [0.2, 0.22, 0.1],
            [0.2, 0.02, 0.1],
            [0.2, 0.12, 0.06],
            [-0.55, 0.12, 0.05],
            [-0.35, 0.18, 0.08],
            [-0.35, 0.06, 0.08],
            [0.72, 0.12, 0.04],
        ],
        faces: vec![
            vec![7, 0, 1],
            vec![7, 2, 0],
            vec![0, 1, 3],
            vec![0, 3, 2],
            vec![3, 1, 5, 4],
            vec![3, 4, 6, 2],
            vec![1, 5, 4],
            vec![2, 4, 6],
        ],
    }
}

fn fold_corner() -> FoldedState {
    // Step 8: tall triangle, fold top corner down (repeat behind).
    FoldedState {
        verts: vec![
            [0.0, 0.55, 0.08],
            [0.18, -0.55, 0.0],
            [-0.18, -0.55, 0.0],
            [0.0, -0.55, 0.04],
            [0.0, 0.78, 0.14],
            [0.08, 0.62, 0.18],
            [-0.08, 0.62, 0.12],
        ],
        faces: vec![
            vec![4, 5, 0],
            vec![4, 0, 6],
            vec![0, 5, 1],
            vec![0, 2, 6],
            vec![0, 1, 3],
            vec![0, 3, 2],
            vec![5, 1, 3],
            vec![6, 3, 2],
        ],
    }
}

fn mountain_inside() -> FoldedState {
    // Step 9: thinner; mountain-fold bottom edges inside.
    FoldedState {
        verts: vec![
            [0.0, 0.52, 0.1],
            [0.1, -0.48, 0.04],
            [-0.1, -0.48, 0.04],
            [0.0, -0.42, 0.08],
            [0.0, 0.72, 0.16],
            [0.06, 0.58, 0.18],
            [-0.06, 0.58, 0.14],
        ],
        faces: vec![
            vec![4, 5, 0],
            vec![4, 0, 6],
            vec![0, 5, 1],
            vec![0, 2, 6],
            vec![0, 1, 3],
            vec![0, 3, 2],
        ],
    }
}

fn crest_and_tail() -> FoldedState {
    // Step 10: horizontal; pull crest (black at neck), fold tail papers down.
    FoldedState {
        verts: vec![
            // 0 body (ridge)
            [0.12, 0.08, 0.18],
            // 1–2 wing roots
            [0.06, 0.26, 0.12],
            [0.08, -0.08, 0.08],
            // 3 neck
            [-0.14, 0.12, 0.2],
            // 4–5 crest (pulled up, red tip)
            [-0.2, 0.48, 0.26],
            [-0.26, 0.34, 0.12],
            // 6–7 black mask
            [-0.34, 0.2, 0.2],
            [-0.3, 0.06, 0.1],
            // 8–10 tail papers down
            [0.58, 0.22, 0.08],
            [0.82, 0.08, 0.02],
            [0.58, -0.06, 0.06],
            // 11 belly tuck
            [0.18, -0.04, 0.02],
            // 12 breast
            [-0.02, 0.0, 0.14],
        ],
        faces: vec![
            vec![0, 1, 3],
            vec![0, 3, 12],
            vec![0, 12, 2],
            vec![1, 4, 5, 3],
            vec![3, 5, 6],
            vec![3, 6, 7, 12],
            vec![12, 7, 2],
            vec![0, 1, 8],
            vec![0, 8, 9, 10],
            vec![0, 10, 2],
            vec![0, 2, 11],
            vec![4, 5, 6],
        ],
    }
}

fn head_corner() -> FoldedState {
    let mut s = crest_and_tail();
    s.verts.push([-0.18, 0.05, 0.1]);
    s.faces.push(vec![7, 12, 2]);
    s.faces.push(vec![3, 7, 12]);
    s
}

fn edge_up() -> FoldedState {
    let mut s = head_corner();
    s.verts.push([-0.38, 0.14, 0.14]);
    s.verts.push([-0.35, 0.1, 0.1]);
    let n = s.verts.len();
    s.faces.push(vec![6, n - 2, n - 1]);
    s.faces.push(vec![6, n - 1, 7]);
    s
}

fn beak() -> FoldedState {
    let mut s = edge_up();
    // Mountain/valley tip for the beak.
    s.verts.push([-0.42, 0.12, 0.12]);
    s.verts.push([-0.4, 0.08, 0.08]);
    let n = s.verts.len();
    s.faces.push(vec![n - 4, n - 2, n - 1]);
    s.faces.push(vec![n - 4, n - 1, n - 3]);
    s
}

fn finished() -> FoldedState {
    // Finished Cardinal: long red body/tail, pointed crest, black face, small beak.
    // Stronger dihedrals so key light separates crest / mask / body / tail.
    FoldedState {
        verts: vec![
            // 0 body centre (raised ridge)
            [0.08, 0.08, 0.22],
            // 1 back / wing (higher)
            [0.02, 0.26, 0.16],
            // 2 belly (lower)
            [0.1, -0.06, 0.04],
            // 3 neck
            [-0.18, 0.12, 0.2],
            // 4 crest tip (red, pointed, tall)
            [-0.16, 0.55, 0.28],
            // 5 crest base
            [-0.2, 0.32, 0.14],
            // 6–7 black face mask (recessed)
            [-0.38, 0.18, 0.2],
            [-0.34, 0.04, 0.1],
            // 8–9 beak
            [-0.5, 0.14, 0.16],
            [-0.46, 0.06, 0.08],
            // 10–12 long tail (spread slightly)
            [0.52, 0.2, 0.1],
            [0.88, 0.06, 0.02],
            [0.52, -0.08, 0.06],
            // 13 stand
            [0.14, -0.14, -0.02],
            // 14 wing tip fold
            [-0.02, 0.34, 0.06],
            // 15 breast fold
            [-0.05, 0.0, 0.16],
        ],
        faces: vec![
            // body red — ridge catches light
            vec![0, 1, 3],
            vec![0, 3, 15],
            vec![0, 15, 2],
            vec![0, 1, 10],
            vec![0, 10, 11, 12],
            vec![0, 12, 2],
            vec![1, 14, 3],
            // crest
            vec![1, 4, 5, 3],
            vec![4, 5, 6],
            // black mask (dark reverse panels)
            vec![3, 5, 6],
            vec![3, 6, 7, 15],
            vec![15, 7, 2],
            vec![6, 8, 9, 7],
            // belly / stand
            vec![2, 7, 13],
            vec![2, 13, 12],
            // wing
            vec![1, 14, 10],
            vec![14, 1, 0],
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn book_cardinal_cp_valid() {
        let cp = crease_pattern();
        assert!(cp.verts.len() >= 18);
        assert!(cp.hinge_indices().count() >= 24);
    }

    #[test]
    fn fourteen_stages() {
        assert_eq!(fold_stages().len(), 14);
        for s in fold_stages() {
            assert!(!s.folded.faces.is_empty(), "{}", s.label);
        }
    }
}
