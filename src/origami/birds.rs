//! Bird-shaped rigid origami — modular parts folded and posed in 3D.
//!
//! Each species is built from degree-4 gadgets (square twist body, crest/wing/tail
//! crosses) that rigid-fold independently, then assemble into a bird silhouette.
//! [`BirdKind::layout_sheet`] lays the part nets on one development sheet (SVG).

use super::fold::FoldedState;
use super::pattern::{Assignment, CreasePattern, Edge};
use super::vertex::VertexMode;
use super::V2;

/// Canonical square-twist sector (α = arctan ¾), same as the square twist.
pub const TWIST_ALPHA: f64 = 0.6435011087932844;

/// A rigid-origami bird species.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BirdKind {
    Cardinal,
    Crane,
    Owl,
    Sparrow,
    Flamingo,
}

impl BirdKind {
    pub const ALL: [BirdKind; 5] = [
        BirdKind::Cardinal,
        BirdKind::Crane,
        BirdKind::Owl,
        BirdKind::Sparrow,
        BirdKind::Flamingo,
    ];

    pub fn name(self) -> &'static str {
        match self {
            BirdKind::Cardinal => "cardinal",
            BirdKind::Crane => "crane",
            BirdKind::Owl => "owl",
            BirdKind::Sparrow => "sparrow",
            BirdKind::Flamingo => "flamingo",
        }
    }

    /// Fold every anatomical part and return poses that form the bird in 3D.
    pub fn assemble(self, t: f64) -> Option<AssembledBird> {
        let mut parts = Vec::new();
        for spec in self.part_specs(t) {
            let folded = spec.fold()?;
            parts.push((folded, spec.pose));
        }
        Some(AssembledBird { kind: self, parts })
    }

    /// Development sheet: each part net placed in anatomical layout (disconnected panels).
    pub fn layout_sheet(self) -> CreasePattern {
        let mut verts = Vec::new();
        let mut edges = Vec::new();
        for spec in self.layout_parts() {
            append_pattern(&mut verts, &mut edges, &spec.pattern, spec.offset);
        }
        CreasePattern::new(verts, edges)
    }

    /// Back-compat alias for [`Self::layout_sheet`].
    pub fn full_net(self) -> CreasePattern {
        self.layout_sheet()
    }

    fn part_specs(self, t: f64) -> Vec<PartSpec> {
        match self {
            BirdKind::Cardinal => cardinal_parts(t),
            BirdKind::Crane => crane_parts(t),
            BirdKind::Owl => owl_parts(t),
            BirdKind::Sparrow => sparrow_parts(t),
            BirdKind::Flamingo => flamingo_parts(t),
        }
    }

    fn layout_parts(self) -> Vec<LayoutPart> {
        match self {
            BirdKind::Cardinal => cardinal_layout(),
            BirdKind::Crane => crane_layout(),
            BirdKind::Owl => owl_layout(),
            BirdKind::Sparrow => sparrow_layout(),
            BirdKind::Flamingo => flamingo_layout(),
        }
    }
}

impl CreasePattern {
    /// Folded primary part for `kind` — body twist only (legacy helper).
    pub fn bird(_kind: BirdKind) -> CreasePattern {
        CreasePattern::square_twist(TWIST_ALPHA)
    }
}

/// One rigid-folded anatomical part plus its local pose in bird space.
#[derive(Debug, Clone)]
pub struct FoldedBirdPart {
    pub name: &'static str,
    pub cp: CreasePattern,
    pub tangents: Vec<f64>,
    pub folded: FoldedState,
}

/// Local transform of a part within the assembled bird (before scene/view transform).
#[derive(Debug, Clone, Copy)]
pub struct PartPose {
    pub position: [f32; 3],
    pub euler: [f32; 3],
    pub scale: f32,
}

/// A fully folded bird ready for rendering.
#[derive(Debug, Clone)]
pub struct AssembledBird {
    pub kind: BirdKind,
    pub parts: Vec<(FoldedBirdPart, PartPose)>,
}

struct PartSpec {
    name: &'static str,
    pattern: CreasePattern,
    mode: Option<VertexMode>,
    t: f64,
    pose: PartPose,
}

impl PartSpec {
    fn fold(&self) -> Option<FoldedBirdPart> {
        let (assign, tangents, folded) = if let Some(mode) = self.mode {
            fold_cross(&self.pattern, mode, self.t)?
        } else {
            fold_twist(&self.pattern, self.t)?
        };
        let _ = assign;
        Some(FoldedBirdPart {
            name: self.name,
            cp: self.pattern.clone(),
            tangents,
            folded,
        })
    }
}

struct LayoutPart {
    pattern: CreasePattern,
    offset: V2,
}

fn fold_twist(cp: &CreasePattern, t: f64) -> Option<(Assignment, Vec<f64>, FoldedState)> {
    // Each candidate assignment is a guess, so a failure is a reason to try the
    // next one rather than to give up: `?` here would have made the first
    // unfoldable candidate the answer for the whole pattern.
    for assign in cp.find_assignments(1e-8) {
        let Some(drive) = cp.drive_hinge(&assign) else {
            continue;
        };
        let Some(tangents) = cp.propagate(&assign, drive, t) else {
            continue;
        };
        let Some(folded) = cp.fold(&assign, drive, t) else {
            continue;
        };
        return Some((assign, tangents, folded));
    }
    None
}

fn fold_cross(
    cp: &CreasePattern,
    mode: VertexMode,
    t: f64,
) -> Option<(Assignment, Vec<f64>, FoldedState)> {
    let assign = Assignment::uniform(cp.verts.len(), mode);
    let drive = cp.drive_hinge(&assign)?;
    let tangents = cp.propagate(&assign, drive, t)?;
    let folded = cp.fold(&assign, drive, t)?;
    Some((assign, tangents, folded))
}

fn body_pose(scale: f32) -> PartPose {
    PartPose {
        position: [0.0, 0.0, 0.0],
        euler: [-std::f32::consts::FRAC_PI_2, 0.0, 0.0],
        scale,
    }
}

fn cardinal_parts(t: f64) -> Vec<PartSpec> {
    vec![
        PartSpec {
            name: "body",
            pattern: CreasePattern::square_twist(TWIST_ALPHA),
            mode: None,
            t,
            pose: body_pose(1.05),
        },
        PartSpec {
            name: "crest",
            pattern: CreasePattern::cross(0.48, 1.18),
            mode: Some(VertexMode::B),
            t: t * 1.15,
            pose: PartPose {
                position: [0.0, 1.05, 0.18],
                euler: [-std::f32::consts::FRAC_PI_2, 0.0, std::f32::consts::FRAC_PI_2],
                scale: 0.82,
            },
        },
        PartSpec {
            name: "wing_l",
            pattern: CreasePattern::cross(0.68, 1.05),
            mode: Some(VertexMode::A),
            t: t * 0.95,
            pose: PartPose {
                position: [-0.92, 0.22, 0.05],
                euler: [-std::f32::consts::FRAC_PI_2, -0.75, 0.15],
                scale: 0.78,
            },
        },
        PartSpec {
            name: "wing_r",
            pattern: CreasePattern::cross(0.68, 1.05),
            mode: Some(VertexMode::A),
            t: t * 0.95,
            pose: PartPose {
                position: [0.92, 0.22, 0.05],
                euler: [-std::f32::consts::FRAC_PI_2, 0.75, -0.15],
                scale: 0.78,
            },
        },
        PartSpec {
            name: "tail",
            pattern: CreasePattern::cross(0.92, 0.78),
            mode: Some(VertexMode::A),
            t: t * 0.85,
            pose: PartPose {
                position: [0.0, -0.82, -0.12],
                euler: [-std::f32::consts::FRAC_PI_2, std::f32::consts::PI, 0.0],
                scale: 0.62,
            },
        },
    ]
}

fn crane_parts(t: f64) -> Vec<PartSpec> {
    vec![
        PartSpec {
            name: "body",
            pattern: CreasePattern::square_twist(TWIST_ALPHA),
            mode: None,
            t: t * 0.9,
            pose: PartPose {
                position: [0.0, 0.0, 0.0],
                euler: [-std::f32::consts::FRAC_PI_2, 0.0, std::f32::consts::FRAC_PI_4],
                scale: 1.15,
            },
        },
        PartSpec {
            name: "neck",
            pattern: CreasePattern::cross(0.35, 0.95),
            mode: Some(VertexMode::B),
            t: t * 1.2,
            pose: PartPose {
                position: [0.0, 1.45, 0.05],
                euler: [-std::f32::consts::FRAC_PI_2, 0.0, std::f32::consts::FRAC_PI_2],
                scale: 1.1,
            },
        },
        PartSpec {
            name: "wing_l",
            pattern: CreasePattern::cross(0.55, 1.15),
            mode: Some(VertexMode::A),
            t,
            pose: PartPose {
                position: [-1.15, 0.15, 0.0],
                euler: [0.05, -std::f32::consts::FRAC_PI_2, -0.2],
                scale: 1.05,
            },
        },
        PartSpec {
            name: "wing_r",
            pattern: CreasePattern::cross(0.55, 1.15),
            mode: Some(VertexMode::A),
            t,
            pose: PartPose {
                position: [1.15, 0.15, 0.0],
                euler: [0.05, std::f32::consts::FRAC_PI_2, 0.2],
                scale: 1.05,
            },
        },
        PartSpec {
            name: "tail",
            pattern: CreasePattern::cross(0.42, 1.05),
            mode: Some(VertexMode::A),
            t: t * 0.8,
            pose: PartPose {
                position: [0.0, -1.1, -0.08],
                euler: [-std::f32::consts::FRAC_PI_2, std::f32::consts::PI, 0.15],
                scale: 0.9,
            },
        },
    ]
}

fn owl_parts(t: f64) -> Vec<PartSpec> {
    vec![
        PartSpec {
            name: "body",
            pattern: CreasePattern::square_twist(TWIST_ALPHA),
            mode: None,
            t: t * 0.85,
            pose: body_pose(1.1),
        },
        PartSpec {
            name: "head",
            pattern: CreasePattern::cross(0.82, 1.35),
            mode: Some(VertexMode::A),
            t: t * 0.75,
            pose: PartPose {
                position: [0.0, 0.95, 0.35],
                euler: [-0.35, 0.0, 0.0],
                scale: 1.05,
            },
        },
        PartSpec {
            name: "wing_l",
            pattern: CreasePattern::cross(0.78, 1.22),
            mode: Some(VertexMode::A),
            t: t * 0.7,
            pose: PartPose {
                position: [-1.0, 0.15, 0.0],
                euler: [-std::f32::consts::FRAC_PI_2, -0.55, 0.45],
                scale: 0.9,
            },
        },
        PartSpec {
            name: "wing_r",
            pattern: CreasePattern::cross(0.78, 1.22),
            mode: Some(VertexMode::A),
            t: t * 0.7,
            pose: PartPose {
                position: [1.0, 0.15, 0.0],
                euler: [-std::f32::consts::FRAC_PI_2, 0.55, -0.45],
                scale: 0.9,
            },
        },
    ]
}

fn sparrow_parts(t: f64) -> Vec<PartSpec> {
    let s = 0.72;
    vec![
        PartSpec {
            name: "body",
            pattern: CreasePattern::square_twist(TWIST_ALPHA),
            mode: None,
            t: t * 0.95,
            pose: body_pose(0.82 * s),
        },
        PartSpec {
            name: "head",
            pattern: CreasePattern::cross(0.58, 0.95),
            mode: Some(VertexMode::B),
            t,
            pose: PartPose {
                position: [0.0, 0.62 * s, 0.12],
                euler: [-std::f32::consts::FRAC_PI_2, 0.0, std::f32::consts::FRAC_PI_2],
                scale: 0.68 * s,
            },
        },
        PartSpec {
            name: "wing_l",
            pattern: CreasePattern::cross(0.62, 0.98),
            mode: Some(VertexMode::A),
            t: t * 0.9,
            pose: PartPose {
                position: [-0.58 * s, 0.12, 0.0],
                euler: [-std::f32::consts::FRAC_PI_2, -0.85, 0.1],
                scale: 0.62 * s,
            },
        },
        PartSpec {
            name: "wing_r",
            pattern: CreasePattern::cross(0.62, 0.98),
            mode: Some(VertexMode::A),
            t: t * 0.9,
            pose: PartPose {
                position: [0.58 * s, 0.12, 0.0],
                euler: [-std::f32::consts::FRAC_PI_2, 0.85, -0.1],
                scale: 0.62 * s,
            },
        },
        PartSpec {
            name: "tail",
            pattern: CreasePattern::cross(0.7, 0.88),
            mode: Some(VertexMode::A),
            t: t * 0.75,
            pose: PartPose {
                position: [0.0, -0.52 * s, -0.08],
                euler: [-std::f32::consts::FRAC_PI_2, std::f32::consts::PI, 0.0],
                scale: 0.48 * s,
            },
        },
    ]
}

fn flamingo_parts(t: f64) -> Vec<PartSpec> {
    vec![
        PartSpec {
            name: "body",
            pattern: CreasePattern::square_twist(TWIST_ALPHA),
            mode: None,
            t: t * 0.8,
            pose: PartPose {
                position: [0.2, -0.15, 0.0],
                euler: [-std::f32::consts::FRAC_PI_2, 0.35, -0.15],
                scale: 0.85,
            },
        },
        PartSpec {
            name: "neck",
            pattern: CreasePattern::cross(0.32, 0.88),
            mode: Some(VertexMode::B),
            t: t * 1.3,
            pose: PartPose {
                position: [0.55, 1.55, 0.25],
                euler: [-0.85, 0.55, 0.65],
                scale: 1.05,
            },
        },
        PartSpec {
            name: "wing_up",
            pattern: CreasePattern::cross(0.5, 1.08),
            mode: Some(VertexMode::A),
            t: t * 0.85,
            pose: PartPose {
                position: [-0.45, 0.55, 0.15],
                euler: [-std::f32::consts::FRAC_PI_2, -0.25, -1.05],
                scale: 0.88,
            },
        },
        PartSpec {
            name: "wing_down",
            pattern: CreasePattern::cross(0.45, 1.0),
            mode: Some(VertexMode::A),
            t: t * 0.65,
            pose: PartPose {
                position: [0.75, 0.15, -0.1],
                euler: [-std::f32::consts::FRAC_PI_2, 0.65, 0.55],
                scale: 0.62,
            },
        },
        PartSpec {
            name: "tail",
            pattern: CreasePattern::cross(0.85, 0.72),
            mode: Some(VertexMode::A),
            t: t * 0.7,
            pose: PartPose {
                position: [0.0, -0.78, -0.12],
                euler: [-std::f32::consts::FRAC_PI_2, std::f32::consts::PI, -0.25],
                scale: 0.52,
            },
        },
    ]
}

fn cardinal_layout() -> Vec<LayoutPart> {
    vec![
        layout_twist([0.0, 0.0]),
        layout_cross([0.0, 3.6], 0.48, 1.18),
        layout_cross([-3.8, 0.4], 0.68, 1.05),
        layout_cross([3.8, 0.4], 0.68, 1.05),
        layout_cross([0.0, -3.4], 0.92, 0.78),
    ]
}

fn crane_layout() -> Vec<LayoutPart> {
    vec![
        layout_twist([0.0, -0.3]),
        layout_cross([0.0, 3.9], 0.35, 0.95),
        layout_cross([-4.2, 0.2], 0.55, 1.15),
        layout_cross([4.2, 0.2], 0.55, 1.15),
        layout_cross([0.0, -3.8], 0.42, 1.05),
    ]
}

fn owl_layout() -> Vec<LayoutPart> {
    vec![
        layout_twist([0.0, 0.0]),
        layout_cross([0.0, 3.2], 0.82, 1.35),
        layout_cross([-4.0, 0.5], 0.78, 1.22),
        layout_cross([4.0, 0.5], 0.78, 1.22),
    ]
}

fn sparrow_layout() -> Vec<LayoutPart> {
    let s = 0.75;
    vec![
        layout_twist_scaled([0.0, 0.0], s),
        layout_cross_scaled([0.0, 2.5 * s], 0.58, 0.95, s),
        layout_cross_scaled([-2.8 * s, 0.2], 0.62, 0.98, s),
        layout_cross_scaled([2.8 * s, 0.2], 0.62, 0.98, s),
        layout_cross_scaled([0.0, -2.2 * s], 0.7, 0.88, s),
    ]
}

fn flamingo_layout() -> Vec<LayoutPart> {
    vec![
        layout_twist([0.5, -0.2]),
        layout_cross([1.0, 3.8], 0.32, 0.88),
        layout_cross([-2.5, 1.0], 0.5, 1.08),
        layout_cross([3.2, 0.6], 0.45, 1.0),
        layout_cross([0.4, -3.0], 0.85, 0.72),
    ]
}

fn layout_twist(offset: V2) -> LayoutPart {
    LayoutPart {
        pattern: CreasePattern::square_twist(TWIST_ALPHA),
        offset,
    }
}

fn layout_twist_scaled(offset: V2, scale: f64) -> LayoutPart {
    LayoutPart {
        pattern: scale_pattern(&CreasePattern::square_twist(TWIST_ALPHA), scale),
        offset,
    }
}

fn layout_cross(offset: V2, alpha: f64, beta: f64) -> LayoutPart {
    LayoutPart {
        pattern: CreasePattern::cross(alpha, beta),
        offset,
    }
}

fn layout_cross_scaled(offset: V2, alpha: f64, beta: f64, scale: f64) -> LayoutPart {
    LayoutPart {
        pattern: scale_pattern(&CreasePattern::cross(alpha, beta), scale),
        offset,
    }
}

fn scale_pattern(cp: &CreasePattern, scale: f64) -> CreasePattern {
    let verts: Vec<V2> = cp.verts.iter().map(|v| [v[0] * scale, v[1] * scale]).collect();
    CreasePattern::new(verts, cp.edges.clone())
}

fn append_pattern(verts: &mut Vec<V2>, edges: &mut Vec<Edge>, cp: &CreasePattern, offset: V2) {
    let base = verts.len();
    for v in &cp.verts {
        verts.push([v[0] + offset[0], v[1] + offset[1]]);
    }
    for e in &cp.edges {
        edges.push(Edge {
            a: base + e.a,
            b: base + e.b,
            kind: e.kind,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_birds_assemble() {
        for kind in BirdKind::ALL {
            let bird = kind
                .assemble(0.42)
                .unwrap_or_else(|| panic!("{} should assemble", kind.name()));
            assert!(bird.parts.len() >= 4, "{}", kind.name());
        }
    }

    #[test]
    fn layout_sheets_are_kawasaki() {
        for kind in BirdKind::ALL {
            let cp = kind.layout_sheet();
            assert!(
                cp.kawasaki_all(),
                "{} layout should be locally flat-foldable",
                kind.name()
            );
        }
    }
}
