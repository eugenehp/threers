//! Gantry-style assembly plans for discrete cuboct voxels.
//!
//! Emits a pick → place → fasten sequence for every face part, following a
//! bottom-up row order. Mobile robots can reuse the same joint graph with
//! [`super::CuboctAssembly::joints`] as local alignment features.

use super::voxel::{face_axis_low, CuboctAssembly, CuboctJointKind};
use super::Cuboct;
use crate::math::Vector3;

/// One step for an assembler or robot motion planner.
#[derive(Clone, Debug, PartialEq)]
pub enum CuboctAssemblyStep {
    /// Move end effector to a safe pose (mm).
    Travel {
        to: [f64; 3],
    },
    /// Pick a face part from a tray (flat on XY).
    PickPart {
        voxel: usize,
        face: usize,
        kind: Cuboct,
    },
    /// Place a face on the voxel at its assembled pose.
    PlacePart {
        voxel: usize,
        face: usize,
        origin: [f64; 3],
        normal: [f64; 3],
    },
    /// Install a rivet at a joint (inner or inter).
    Fasten {
        joint: usize,
        kind: CuboctJointKind,
        at: [f64; 3],
    },
}

/// Ordered assembly instructions.
#[derive(Clone, Debug, Default)]
pub struct CuboctAssemblyPlan {
    pub steps: Vec<CuboctAssemblyStep>,
}

impl CuboctAssemblyPlan {
    /// Build a plan from an assembly configuration.
    pub fn from_assembly(asm: &CuboctAssembly<'_>) -> Self {
        let mut steps = Vec::new();
        let [nx, ny, nz] = asm.grid_cells();
        let pitch = asm.assembly_pitch() as f64;
        let home = [0.0, 0.0, pitch * (nz as f64 + 2.0)];

        steps.push(CuboctAssemblyStep::Travel { to: home });

        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let v = asm.voxel_flat(ix as i32, iy as i32, iz as i32);
                    let origin = asm.voxel_origin(ix as i32, iy as i32, iz as i32);
                    let kind = asm.kind_at(ix as i32, iy as i32, iz as i32);
                    for face in 0..6 {
                        steps.push(CuboctAssemblyStep::PickPart {
                            voxel: v,
                            face,
                            kind,
                        });
                        let (axis, low) = face_axis_low(face);
                        let n = outward_normal(axis, low);
                        let plane = face_plane_offset(axis, low, pitch as f32);
                        let at = origin + plane;
                        steps.push(CuboctAssemblyStep::PlacePart {
                            voxel: v,
                            face,
                            origin: [at.x as f64, at.y as f64, at.z as f64],
                            normal: [n.x as f64, n.y as f64, n.z as f64],
                        });
                    }
                }
            }
        }

        for (ji, j) in asm.joints().iter().enumerate() {
            steps.push(CuboctAssemblyStep::Fasten {
                joint: ji,
                kind: j.kind,
                at: [j.at.x as f64, j.at.y as f64, j.at.z as f64],
            });
        }

        steps.push(CuboctAssemblyStep::Travel { to: home });
        CuboctAssemblyPlan { steps }
    }

    /// Export as a simple CSV log for simulation or shop-floor tools.
    pub fn to_csv(&self) -> String {
        let mut out = String::from("step,action,detail\n");
        for (i, s) in self.steps.iter().enumerate() {
            match s {
                CuboctAssemblyStep::Travel { to } => {
                    out.push_str(&format!("{i},travel,\"{to:?}\"\n"));
                }
                CuboctAssemblyStep::PickPart { voxel, face, kind } => {
                    out.push_str(&format!(
                        "{i},pick,\"voxel={voxel} face={face} kind={}\"\n",
                        kind.name()
                    ));
                }
                CuboctAssemblyStep::PlacePart {
                    voxel,
                    face,
                    origin,
                    normal,
                } => {
                    out.push_str(&format!(
                        "{i},place,\"voxel={voxel} face={face} at={origin:?} n={normal:?}\"\n"
                    ));
                }
                CuboctAssemblyStep::Fasten { joint, kind, at } => {
                    out.push_str(&format!(
                        "{i},fasten,\"joint={joint} {:?} at={at:?}\"\n",
                        kind
                    ));
                }
            }
        }
        out
    }
}

fn outward_normal(axis: usize, low: bool) -> Vector3 {
    match (axis, low) {
        (0, true) => Vector3::new(-1.0, 0.0, 0.0),
        (0, false) => Vector3::new(1.0, 0.0, 0.0),
        (1, true) => Vector3::new(0.0, -1.0, 0.0),
        (1, false) => Vector3::new(0.0, 1.0, 0.0),
        (_, true) => Vector3::new(0.0, 0.0, -1.0),
        (_, false) => Vector3::new(0.0, 0.0, 1.0),
    }
}

fn face_plane_offset(axis: usize, low: bool, pitch: f32) -> Vector3 {
    if low {
        Vector3::ZERO
    } else {
        match axis {
            0 => Vector3::new(pitch, 0.0, 0.0),
            1 => Vector3::new(0.0, pitch, 0.0),
            _ => Vector3::new(0.0, 0.0, pitch),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_covers_every_part_and_joint() {
        let asm = CuboctAssembly::new(Cuboct::Rigid)
            .pitch(20.0)
            .cells([2, 1, 1]);
        let plan = CuboctAssemblyPlan::from_assembly(&asm);
        let picks = plan
            .steps
            .iter()
            .filter(|s| matches!(s, CuboctAssemblyStep::PickPart { .. }))
            .count();
        let fastens = plan
            .steps
            .iter()
            .filter(|s| matches!(s, CuboctAssemblyStep::Fasten { .. }))
            .count();
        assert_eq!(picks, 12);
        assert_eq!(fastens, asm.joints().len());
        assert!(plan.to_csv().contains("pick"));
    }
}
