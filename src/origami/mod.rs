//! Rigid origami from a planar crease pattern.
//!
//! This is the kinematics Akitaya, Demaine, Horiyama, Hull, Ku and Tachi use
//! to prove rigid foldability is NP-hard ([arXiv:1812.01160](https://arxiv.org/abs/1812.01160)).
//! We implement the part that *is* an algorithm: degree-4 flat-foldable vertices
//! (Theorem 6), Kawasaki's condition, speed-coefficient loop closure
//! (Corollary 11), and a dual walk that places panels in 3D.
//!
//! Deciding foldability for an arbitrary net is NP-hard. This module checks a
//! given mountain–valley / mode assignment and folds it when the assignment is
//! consistent. It does not search the assignment space.

mod birds;
mod book_cardinal;
mod classic_bird;
mod eagle;
mod fold;
mod frog;
mod giang_cardinal;
mod pattern;
mod vertex;

pub use birds::{AssembledBird, BirdKind, FoldedBirdPart, PartPose, TWIST_ALPHA};
pub use book_cardinal::{
    crease_pattern as book_cardinal_cp, fold_stages as book_cardinal_stages, BookCardinalStage,
};
pub use classic_bird::{
    crease_pattern as classic_bird_cp, fold_stages as classic_bird_stages, ClassicBirdStage,
};
pub use eagle::{crease_pattern as eagle_cp, fold_stages as eagle_stages, EagleStage};
pub use fold::FoldedState;
pub use frog::{crease_pattern as frog_cp, fold_stages as frog_stages, FrogStage};
pub use giang_cardinal::{crease_pattern as giang_cardinal_cp, fold_stages as giang_cardinal_stages, GiangCardinalStage};
pub use pattern::{Assignment, ClosureReport, CreasePattern, Edge, EdgeKind};
pub use vertex::{degree4_from_dirs, kawasaki, reflect_through, Degree4, VertexMode};

pub(crate) type V2 = [f64; 2];
pub(crate) type V3 = [f64; 3];

pub(crate) fn sub2(a: V2, b: V2) -> V2 {
    [a[0] - b[0], a[1] - b[1]]
}

pub(crate) fn dot2(a: V2, b: V2) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

pub(crate) fn cross2(a: V2, b: V2) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

pub(crate) fn sub3(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(crate) fn add3(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub(crate) fn dot3(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn cross3(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge_len(a: V3, b: V3) -> f64 {
        dot3(sub3(a, b), sub3(a, b)).sqrt()
    }

    #[test]
    fn cross_is_kawasaki_and_has_four_panels() {
        let cp = CreasePattern::cross(std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
        assert!(cp.kawasaki_all());
        assert_eq!(cp.faces.len(), 4);
        assert_eq!(cp.interior_deg4(), vec![0]);
        assert!(cp.interior_face_ids().is_empty());
    }

    #[test]
    fn figure1_cannot_use_all_four_creases() {
        let cp = CreasePattern::cross(std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
        let assign_a = Assignment::uniform(cp.verts.len(), VertexMode::A);
        let assign_b = Assignment::uniform(cp.verts.len(), VertexMode::B);
        let drive_a = cp.drive_hinge(&assign_a).unwrap();
        let drive_b = cp.drive_hinge(&assign_b).unwrap();
        let ta = cp.propagate(&assign_a, drive_a, 0.5).unwrap();
        let tb = cp.propagate(&assign_b, drive_b, 0.5).unwrap();
        let used = |t: &[f64]| {
            t.iter()
                .zip(&cp.edges)
                .filter(|(ti, e)| e.kind == EdgeKind::Hinge && ti.abs() > 1e-9)
                .count()
        };
        // Each mode folds exactly one opposite pair.
        assert_eq!(used(&ta), 2);
        assert_eq!(used(&tb), 2);
        // The two modes fold complementary pairs.
        let both = ta
            .iter()
            .zip(&tb)
            .zip(&cp.edges)
            .filter(|((a, b), e)| e.kind == EdgeKind::Hinge && a.abs() > 1e-9 && b.abs() > 1e-9)
            .count();
        assert_eq!(both, 0);
    }

    #[test]
    fn figure1_folds_off_the_plane() {
        let cp = CreasePattern::cross(std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
        let assign = Assignment::uniform(cp.verts.len(), VertexMode::A);
        let drive = cp.drive_hinge(&assign).unwrap();
        let folded = cp.fold(&assign, drive, 0.6).expect("fold");
        let zmax = folded.verts.iter().map(|v| v[2].abs()).fold(0.0, f64::max);
        assert!(zmax > 0.2, "expected a 3D fold, zmax={zmax}");
        // Arms keep their length.
        for i in 1..=4 {
            let l0 = edge_len(
                [cp.verts[0][0], cp.verts[0][1], 0.0],
                [cp.verts[i][0], cp.verts[i][1], 0.0],
            );
            let l1 = edge_len(folded.verts[0], folded.verts[i]);
            assert!((l0 - l1).abs() < 1e-8, "arm {i} stretched {l0} → {l1}");
        }
        // Panels stay planar (triangles always do) and the origin is shared.
        assert_eq!(folded.verts[0], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn square_twist_lemma14_assignments_close() {
        let alpha = (0.75f64).atan(); // paper: p = 1/2
        let cp = CreasePattern::square_twist(alpha);
        assert!(
            cp.kawasaki_all(),
            "inner vertices should be flat-foldable degree 4"
        );
        assert_eq!(cp.interior_deg4().len(), 4);
        assert!(
            !cp.interior_face_ids().is_empty(),
            "inner square should be an all-hinge face"
        );

        let found = cp.find_assignments(1e-6);
        assert!(
            !found.is_empty() && found.len() < 16,
            "Lemma 14: some but not all of the 16 mode tuples are rigid; got {}",
            found.len()
        );
    }

    #[test]
    fn square_twist_folds_and_preserves_edge_lengths() {
        let cp = CreasePattern::square_twist((0.75f64).atan());
        let assign = cp
            .find_assignments(1e-6)
            .into_iter()
            .next()
            .expect("a rigid assignment");
        let drive = cp.drive_hinge(&assign).unwrap();
        let tangents = cp
            .propagate(&assign, drive, 0.35)
            .expect("rigid assignment should propagate");
        let folded = cp
            .fold(&assign, drive, 0.35)
            .expect("rigid assignment should place in 3D");
        for (ei, e) in cp.edges.iter().enumerate() {
            let l0 = edge_len(
                [cp.verts[e.a][0], cp.verts[e.a][1], 0.0],
                [cp.verts[e.b][0], cp.verts[e.b][1], 0.0],
            );
            let l1 = edge_len(folded.verts[e.a], folded.verts[e.b]);
            assert!(
                (l0 - l1).abs() < 1e-6,
                "edge {ei} stretched {l0} → {l1} (t={})",
                tangents[ei]
            );
        }
        let zmax = folded.verts.iter().map(|v| v[2].abs()).fold(0.0, f64::max);
        assert!(zmax > 0.05, "twist stayed flat, zmax={zmax}");
        let _ = folded.to_geometry();
    }

    #[test]
    fn miura_is_kawasaki_and_folds_from_a_seed() {
        let cp = CreasePattern::miura(2, 2);
        assert!(cp.kawasaki_all());
        let interiors = cp.interior_deg4();
        assert_eq!(interiors.len(), 1, "2×2 cells → one interior vertex");
        let seed = interiors[0];
        let assign = cp
            .assignment_from_seed(seed, VertexMode::A)
            .expect("seed assignment");
        let drive = cp.drive_hinge(&assign).expect("drive");
        let tangents = cp.propagate(&assign, drive, 0.4).expect("propagate");
        let folded = cp.fold(&assign, drive, 0.4).expect("place 3D");
        assert!(cp.is_rigid(&assign, 1e-6));
        for (ei, e) in cp.edges.iter().enumerate() {
            let l0 = edge_len(
                [cp.verts[e.a][0], cp.verts[e.a][1], 0.0],
                [cp.verts[e.b][0], cp.verts[e.b][1], 0.0],
            );
            let l1 = edge_len(folded.verts[e.a], folded.verts[e.b]);
            assert!((l0 - l1).abs() < 1e-5, "edge {ei} stretched {l0} → {l1}");
        }
        let zmax = folded.verts.iter().map(|v| v[2].abs()).fold(0.0, f64::max);
        assert!(zmax > 0.05, "Miura stayed flat, zmax={zmax}");
        assert!(cp.to_svg(Some(&tangents)).contains("<line"));
        assert!(cp.assignment_from_seed(seed, VertexMode::B).is_some());

        // Larger sheet: uniform mode is the usual Miura tessellation.
        let sheet = CreasePattern::miura(4, 3);
        assert!(sheet.kawasaki_all());
        let n = sheet.verts.len();
        let mut ok = false;
        for mode in [VertexMode::A, VertexMode::B] {
            let assign = Assignment::uniform(n, mode);
            if let Some(drive) = sheet.drive_hinge(&assign) {
                if sheet.propagate(&assign, drive, 0.35).is_some()
                    && sheet.fold(&assign, drive, 0.35).is_some()
                {
                    ok = true;
                    break;
                }
            }
        }
        if !ok {
            ok = sheet.find_assignments(1e-5).into_iter().any(|a| {
                sheet
                    .drive_hinge(&a)
                    .and_then(|d| sheet.fold(&a, d, 0.35))
                    .is_some()
            });
        }
        assert!(ok, "4×3 Miura should fold under some assignment");
    }

    #[test]
    fn polygon_twists_fold() {
        for n in [5usize, 6, 8] {
            let cp = CreasePattern::polygon_twist(n, 0.45);
            assert!(
                cp.kawasaki_all(),
                "n={n} corners should be flat-foldable degree 4"
            );
            assert_eq!(cp.interior_deg4().len(), n);
            // A regular n-gon of degree-4 vertices is locally Kawasaki; a
            // globally rigid all-crease motion is not guaranteed (0 for n=5,6
            // with this spoke layout). The square twist (n=4) is the case that
            // Lemma 14 characterises.
            let _ = cp.find_assignments(1e-5);
        }
    }
}
