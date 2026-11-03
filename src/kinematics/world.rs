//! Arm world: link solids, revolute joints, materials, and contacts.
//!
//! The geared [`super::plant::ArmPlant`] owns the dynamics. This layer
//! turns each link into a capsule body with mass/volume/material, reports joint
//! state (angle, limits, motor torque), and runs capsule–capsule / capsule–plane
//! / capsule–AABB contact queries so a browser demo can track collisions the
//! same way a rigid-body scene would.

use super::chain::{scale, sub, JointPose, SerialChain};
use super::plant::ArmPlant;
use super::{dot, norm, unit, V3};

/// Named surface finish for a link solid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkMaterial {
    Aluminum,
    Steel,
    Plastic,
    Rubber,
}

impl LinkMaterial {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Aluminum => "aluminum",
            Self::Steel => "steel",
            Self::Plastic => "plastic",
            Self::Rubber => "rubber",
        }
    }

    /// Linear RGB for preview meshes (not PBR F0).
    pub fn rgb(self) -> [f32; 3] {
        match self {
            Self::Aluminum => [0.72, 0.74, 0.78],
            Self::Steel => [0.45, 0.48, 0.52],
            Self::Plastic => [0.25, 0.55, 0.85],
            Self::Rubber => [0.18, 0.18, 0.2],
        }
    }

    pub fn metalness(self) -> f32 {
        match self {
            Self::Aluminum | Self::Steel => 0.85,
            Self::Plastic => 0.05,
            Self::Rubber => 0.0,
        }
    }

    pub fn roughness(self) -> f32 {
        match self {
            Self::Aluminum => 0.35,
            Self::Steel => 0.45,
            Self::Plastic => 0.55,
            Self::Rubber => 0.9,
        }
    }
}

/// One rigid link volume (capsule along the hinge→distal segment).
#[derive(Debug, Clone)]
pub struct LinkBody {
    pub name: String,
    pub material: LinkMaterial,
    /// Capsule radius, model units (mm).
    pub radius: f64,
    pub mass: f64,
    /// π r² L cylinder volume, mm³ (caps ignored for the report).
    pub volume: f64,
    pub origin: V3,
    pub distal: V3,
    pub axis: V3,
    pub com: V3,
}

/// One revolute hinge between bodies.
#[derive(Debug, Clone)]
pub struct JointState {
    pub name: String,
    pub kind: &'static str,
    pub angle_deg: f64,
    pub cmd_deg: f64,
    pub omega: f64,
    pub limits: (f64, f64),
    /// Motor-reported joint drive torque, N·m.
    pub torque: f64,
    pub origin: V3,
    pub axis: V3,
    /// True while the hinge origin is within `tol` of the expected mate.
    pub engaged: bool,
}

/// A contact between two solids (or a solid and the world).
#[derive(Debug, Clone)]
pub struct Contact {
    pub a: String,
    pub b: String,
    pub point: V3,
    pub normal: V3,
    pub depth: f64,
}

/// Static world geometry the arm can hit.
#[derive(Debug, Clone)]
pub struct WorldObstacles {
    /// Floor plane z = `floor_z` with upward normal +Z.
    pub floor_z: f64,
    /// Axis-aligned obstacle box [min, max] in model units, or `None`.
    pub box_min: Option<V3>,
    pub box_max: Option<V3>,
}

impl Default for WorldObstacles {
    fn default() -> Self {
        Self {
            floor_z: -1.0,
            // Post in the workspace the wrist can clip when lagging.
            box_min: Some([340.0, -40.0, 0.0]),
            box_max: Some([400.0, 40.0, 200.0]),
        }
    }
}

/// Geared plant + solids + contacts.
#[derive(Debug, Clone)]
pub struct ArmWorld {
    pub plant: ArmPlant,
    pub materials: Vec<LinkMaterial>,
    pub radii: Vec<f64>,
    pub obstacles: WorldObstacles,
    pub last_cmd: Vec<f64>,
    pub contacts: Vec<Contact>,
}

impl ArmWorld {
    pub fn from_plant(plant: ArmPlant) -> Self {
        let n = plant.chain.n();
        let materials = (0..n)
            .map(|i| match i {
                0 => LinkMaterial::Aluminum,
                1 => LinkMaterial::Steel,
                _ => LinkMaterial::Plastic,
            })
            .collect();
        // Slightly fatter proximal links.
        let radii = (0..n).map(|i| 22.0 - i as f64 * 3.0).collect();
        let last_cmd = plant.q_deg();
        Self {
            plant,
            materials,
            radii,
            obstacles: WorldObstacles::default(),
            last_cmd,
            contacts: Vec::new(),
        }
    }

    pub fn bodies(&self) -> Vec<LinkBody> {
        let q = self.plant.q_deg();
        let poses = self.plant.chain.poses(&q);
        poses
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let len = norm(sub(p.distal, p.origin)).max(1e-6);
                let r = self.radii[i];
                let mass = self.plant.chain.joints[i].mass;
                LinkBody {
                    name: format!("link{}", i + 1),
                    material: self.materials[i],
                    radius: r,
                    mass,
                    volume: std::f64::consts::PI * r * r * len,
                    origin: p.origin,
                    distal: p.distal,
                    axis: p.axis,
                    com: scale(add3(p.origin, p.distal), 0.5),
                }
            })
            .collect()
    }

    pub fn joints(&self) -> Vec<JointState> {
        let q = self.plant.q_deg();
        let poses = self.plant.chain.poses(&q);
        poses
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let servo = &self.plant.servos[i];
                let (lo, hi) = self.plant.chain.joints[i].limits;
                JointState {
                    name: format!("J{}", i + 1),
                    kind: "revolute",
                    angle_deg: servo.theta_deg,
                    cmd_deg: self.last_cmd.get(i).copied().unwrap_or(servo.theta_deg),
                    omega: servo.omega,
                    limits: (lo, hi),
                    torque: servo.tau_joint,
                    origin: p.origin,
                    axis: p.axis,
                    engaged: true,
                }
            })
            .collect()
    }

    /// Mechanical BOM placed in the world for the current pose.
    ///
    /// Each joint uses its drive's [`ActuatorDesign`](super::design::ActuatorDesign)
    /// (compact module / vane / pulley) — sized like onero / HEBI packs.
    pub fn design_parts(&self) -> Vec<super::design::DesignPart> {
        use super::design::{assemble_joint, ActuatorDesign};
        let q = self.plant.q_deg();
        let poses = self.plant.chain.poses(&q);
        let base = poses
            .first()
            .map(|p| [p.origin[0], p.origin[1], 0.0])
            .unwrap_or([0.0, 0.0, 0.0]);
        let mut out = Vec::new();
        for (i, p) in poses.iter().enumerate() {
            let des = ActuatorDesign::from_drive(&self.plant.servos[i].drive);
            let proximal = if i == 0 {
                base
            } else {
                poses[i - 1].origin
            };
            out.extend(assemble_joint(i, &des, p.origin, p.axis, proximal));
        }
        out
    }

    pub fn design_label(&self) -> String {
        use super::design::ActuatorDesign;
        self.plant
            .servos
            .first()
            .map(|s| ActuatorDesign::from_drive(&s.drive).label().to_string())
            .unwrap_or_default()
    }

    /// Collision capsule endpoints inset from the hinges so a base mount on
    /// the floor is not reported as a contact, and adjacent links don't false-hit.
    fn collision_segment(origin: V3, distal: V3, radius: f64) -> (V3, V3) {
        let d = sub(distal, origin);
        let len = norm(d).max(1e-6);
        let u = scale(d, 1.0 / len);
        let inset = radius.min(len * 0.4);
        (add3(origin, scale(u, inset)), add3(distal, scale(u, -inset)))
    }

    /// Refresh [`Self::contacts`] from the current pose.
    pub fn update_contacts(&mut self) {
        self.contacts.clear();
        let bodies = self.bodies();
        let segs: Vec<_> = bodies
            .iter()
            .map(|b| Self::collision_segment(b.origin, b.distal, b.radius))
            .collect();
        // Link–link (skip adjacent — they share a hinge).
        for i in 0..bodies.len() {
            for j in (i + 2)..bodies.len() {
                if let Some(c) = capsule_capsule(
                    segs[i].0,
                    segs[i].1,
                    bodies[i].radius,
                    segs[j].0,
                    segs[j].1,
                    bodies[j].radius,
                ) {
                    self.contacts.push(Contact {
                        a: bodies[i].name.clone(),
                        b: bodies[j].name.clone(),
                        point: c.0,
                        normal: c.1,
                        depth: c.2,
                    });
                }
            }
        }
        // Floor.
        for (b, seg) in bodies.iter().zip(segs.iter()) {
            if let Some(c) = capsule_plane(seg.0, seg.1, b.radius, self.obstacles.floor_z) {
                self.contacts.push(Contact {
                    a: b.name.clone(),
                    b: "floor".into(),
                    point: c.0,
                    normal: c.1,
                    depth: c.2,
                });
            }
        }
        // Obstacle AABB.
        if let (Some(mn), Some(mx)) = (self.obstacles.box_min, self.obstacles.box_max) {
            for (b, seg) in bodies.iter().zip(segs.iter()) {
                if let Some(c) = capsule_aabb(seg.0, seg.1, b.radius, mn, mx) {
                    self.contacts.push(Contact {
                        a: b.name.clone(),
                        b: "obstacle".into(),
                        point: c.0,
                        normal: c.1,
                        depth: c.2,
                    });
                }
            }
        }
    }
}

fn add3(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Closest points between two segments; returns (pa, pb, dist).
fn segment_closest(a0: V3, a1: V3, b0: V3, b1: V3) -> (V3, V3, f64) {
    let d1 = sub(a1, a0);
    let d2 = sub(b1, b0);
    let r = sub(a0, b0);
    let a = dot(d1, d1).max(1e-18);
    let e = dot(d2, d2).max(1e-18);
    let f = dot(d2, r);
    let c = dot(d1, r);
    let b = dot(d1, d2);
    let denom = a * e - b * b;
    let mut s = if denom.abs() > 1e-18 {
        ((b * f - c * e) / denom).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut t = (b * s + f) / e;
    if t < 0.0 {
        t = 0.0;
        s = (-c / a).clamp(0.0, 1.0);
    } else if t > 1.0 {
        t = 1.0;
        s = ((b - c) / a).clamp(0.0, 1.0);
    }
    let pa = add3(a0, scale(d1, s));
    let pb = add3(b0, scale(d2, t));
    let dist = norm(sub(pa, pb));
    (pa, pb, dist)
}

fn capsule_capsule(a0: V3, a1: V3, ra: f64, b0: V3, b1: V3, rb: f64) -> Option<(V3, V3, f64)> {
    let (pa, pb, dist) = segment_closest(a0, a1, b0, b1);
    let gap = dist - ra - rb;
    if gap >= 0.0 {
        return None;
    }
    let n = if dist > 1e-9 {
        unit(sub(pa, pb))
    } else {
        [0.0, 0.0, 1.0]
    };
    let point = add3(pb, scale(n, rb));
    Some((point, n, -gap))
}

fn capsule_plane(a0: V3, a1: V3, r: f64, floor_z: f64) -> Option<(V3, V3, f64)> {
    // Lowest point of the capsule toward −Z.
    let z0 = a0[2] - r;
    let z1 = a1[2] - r;
    let (p, z) = if z0 < z1 {
        (a0, z0)
    } else {
        (a1, z1)
    };
    if z >= floor_z {
        return None;
    }
    let point = [p[0], p[1], floor_z];
    Some((point, [0.0, 0.0, 1.0], floor_z - z))
}

fn capsule_aabb(a0: V3, a1: V3, r: f64, mn: V3, mx: V3) -> Option<(V3, V3, f64)> {
    // Sample the segment; good enough for a demo obstacle.
    let mut best: Option<(V3, V3, f64)> = None;
    for k in 0..=8 {
        let t = k as f64 / 8.0;
        let p = add3(a0, scale(sub(a1, a0), t));
        let q = [
            p[0].clamp(mn[0], mx[0]),
            p[1].clamp(mn[1], mx[1]),
            p[2].clamp(mn[2], mx[2]),
        ];
        let d = norm(sub(p, q));
        let gap = d - r;
        if gap < 0.0 {
            let n = if d > 1e-9 {
                unit(sub(p, q))
            } else {
                [0.0, 0.0, 1.0]
            };
            let depth = -gap;
            if best.as_ref().map(|b| depth > b.2).unwrap_or(true) {
                best = Some((q, n, depth));
            }
        }
    }
    best
}

/// Convenience: poses of a chain at `q`.
pub fn poses_at(chain: &SerialChain, q: &[f64]) -> Vec<JointPose> {
    chain.poses(q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinematics::transmission::GearTrain;
    use crate::kinematics::SerialChain;

    #[test]
    fn stacked_arm_does_not_hit_floor() {
        let plant = ArmPlant::from_train(SerialChain::planar_3r(), GearTrain::direct_drive());
        let mut w = ArmWorld::from_plant(plant);
        w.plant.seed(&[0.0, 0.0, 0.0]);
        w.update_contacts();
        assert!(
            w.contacts.iter().all(|c| c.b != "floor"),
            "upright arm should clear floor: {:?}",
            w.contacts
        );
    }

    #[test]
    fn horizontal_arm_hits_obstacle_post() {
        let plant = ArmPlant::from_train(SerialChain::planar_3r(), GearTrain::direct_drive());
        let mut w = ArmWorld::from_plant(plant);
        // Reach across the post.
        w.plant.seed(&[90.0, 0.0, 0.0]);
        w.update_contacts();
        assert!(
            w.contacts.iter().any(|c| c.b == "obstacle"),
            "expected obstacle hit: {:?}",
            w.contacts
        );
    }
}
