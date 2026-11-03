//! Mechanical designs for each joint drive — sized like real modules.
//!
//! References (open hardware / product CAD):
//! - [onero XD5746 / XD5757](https://github.com/OpenWonderLabs/RobotDoc): Φ57×46 / Φ57×56.5 mm
//! - [HEBI X5](https://docs.hebi.us/resources/datasheets/X5-Dimensions.pdf): ~110×73×31 mm pack
//! - [ALTO / OS-ARM](https://github.com/liaochikon/ALTO-3D-Printed-6-Axis-Robotic-Arm): CSF harmonic + NEMA behind
//! - Industrial rotary vanes: short disk body, ports on the OD, lines along the link
//! - Tendon arms: small joint pulley, cables run **along the link** to base winches
//!
//! One compact silhouette per joint — not an exploded BOM of oversized cans.

use super::actuator::Drive;
use super::chain::{add, cross, scale, sub};
use super::materials::{HydraulicFluid, TendonMaterial};
use super::transmission::GearTrain;
use super::{norm, unit, V3};

/// What a design solid is for (renderer / HUD).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PartRole {
    Motor = 0,
    GearStage = 1,
    Flange = 2,
    Shaft = 3,
    HarmonicCup = 4,
    VaneBody = 5,
    FluidPort = 6,
    Hose = 7,
    Reservoir = 8,
    Pulley = 9,
    Cable = 10,
    Winch = 11,
    Sheath = 12,
    Bearing = 13,
    Encoder = 14,
    /// Outer actuator can (the product silhouette).
    Housing = 15,
}

impl PartRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Motor => "motor",
            Self::GearStage => "gear_stage",
            Self::Flange => "flange",
            Self::Shaft => "shaft",
            Self::HarmonicCup => "harmonic_cup",
            Self::VaneBody => "vane_body",
            Self::FluidPort => "fluid_port",
            Self::Hose => "hose",
            Self::Reservoir => "reservoir",
            Self::Pulley => "pulley",
            Self::Cable => "cable",
            Self::Winch => "winch",
            Self::Sheath => "sheath",
            Self::Bearing => "bearing",
            Self::Encoder => "encoder",
            Self::Housing => "housing",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PartShape {
    Capsule = 0,
    Segment = 1,
    Disk = 2,
}

#[derive(Debug, Clone)]
pub struct DesignPart {
    pub role: PartRole,
    pub shape: PartShape,
    pub name: &'static str,
    pub origin: V3,
    pub axis: V3,
    pub radius_mm: f64,
    pub length_mm: f64,
    pub rgb: [f32; 3],
    pub metalness: f32,
    pub roughness: f32,
    pub joint: i32,
}

#[derive(Debug, Clone)]
pub enum ActuatorDesign {
    Gear(GearMechanism),
    Hydraulic(HydraulicMechanism),
    Tendon(TendonMechanism),
}

/// Compact serial-arm joint module (HEBI / onero style).
#[derive(Debug, Clone)]
pub struct GearMechanism {
    pub label: &'static str,
    pub ratio: f64,
    /// Outer can OD, mm (onero = 57).
    pub od_mm: f64,
    /// Axial length of the main can, mm.
    pub len_mm: f64,
    pub show_harmonic_ring: bool,
}

impl GearMechanism {
    pub fn from_train(train: &GearTrain) -> Self {
        // Sealed can only — motor lives inside the housing (no rear stub).
        if train.ratio <= 1.0 + 1e-9 {
            Self {
                label: "direct pancake module",
                ratio: 1.0,
                od_mm: 70.0,
                len_mm: 28.0,
                show_harmonic_ring: false,
            }
        } else if (train.ratio - 15.0).abs() < 1e-6 {
            Self {
                label: "QDD module Φ57×46",
                ratio: 15.0,
                od_mm: 57.0,
                len_mm: 46.0,
                show_harmonic_ring: false,
            }
        } else {
            Self {
                label: "harmonic module Φ57×56",
                ratio: train.ratio,
                od_mm: 57.0,
                len_mm: 56.0,
                show_harmonic_ring: true,
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct HydraulicMechanism {
    pub label: &'static str,
    pub fluid: HydraulicFluid,
    pub od_mm: f64,
    pub width_mm: f64,
    pub hose_od_mm: f64,
}

impl HydraulicMechanism {
    pub fn from_fluid(fluid: HydraulicFluid) -> Self {
        let hose = match fluid.name {
            "silicone" => 5.5,
            "ISO VG 68" => 5.0,
            _ => 4.5,
        };
        Self {
            label: "rotary vane cartridge",
            fluid,
            od_mm: 58.0,
            width_mm: 34.0,
            hose_od_mm: hose,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TendonMechanism {
    pub label: &'static str,
    pub material: TendonMaterial,
    pub pulley_od_mm: f64,
    pub pulley_w_mm: f64,
    pub cable_od_mm: f64,
    pub winch_od_mm: f64,
}

impl TendonMechanism {
    pub fn from_material(material: TendonMaterial) -> Self {
        let cable = match material.name {
            "steel" => 1.6,
            "uhmwpe" => 1.2,
            "aramid" => 1.4,
            _ => 1.8,
        };
        Self {
            label: "pulley + base winch",
            material,
            pulley_od_mm: 28.0,
            pulley_w_mm: 10.0,
            cable_od_mm: cable,
            winch_od_mm: 24.0,
        }
    }
}

impl ActuatorDesign {
    pub fn from_drive(drive: &Drive) -> Self {
        match drive {
            Drive::Gear(g) => Self::Gear(GearMechanism::from_train(g)),
            Drive::Hydraulic(h) => Self::Hydraulic(HydraulicMechanism::from_fluid(h.fluid)),
            Drive::Tendon(t) => Self::Tendon(TendonMechanism::from_material(t.material)),
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Gear(g) => g.label,
            Self::Hydraulic(h) => h.label,
            Self::Tendon(t) => t.label,
        }
    }

    pub fn family(&self) -> &'static str {
        match self {
            Self::Gear(_) => "gear",
            Self::Hydraulic(_) => "hydraulic",
            Self::Tendon(_) => "tendon",
        }
    }
}

fn fluid_tint(f: &HydraulicFluid) -> [f32; 3] {
    match f.name {
        "ISO VG 32" => [0.82, 0.72, 0.22],
        "ISO VG 46" => [0.72, 0.52, 0.12],
        "ISO VG 68" => [0.55, 0.38, 0.08],
        "water-glycol" => [0.35, 0.62, 0.82],
        "silicone" => [0.72, 0.74, 0.78],
        _ => [0.65, 0.5, 0.15],
    }
}

fn cable_tint(m: &TendonMaterial) -> [f32; 3] {
    match m.name {
        "steel" => [0.68, 0.7, 0.74],
        "uhmwpe" => [0.9, 0.9, 0.86],
        "aramid" => [0.72, 0.52, 0.14],
        _ => [0.18, 0.18, 0.48],
    }
}

fn orthonormal(axis: V3) -> (V3, V3) {
    let a = unit(axis);
    let tmp = if a[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let u = unit(cross(a, tmp));
    let v = unit(cross(a, u));
    (u, v)
}

fn segment(a: V3, b: V3) -> (V3, V3, f64) {
    let d = sub(b, a);
    let len = norm(d).max(1.0);
    (a, scale(d, 1.0 / len), len)
}

/// Place parts for one hinge.
///
/// `proximal` = previous joint origin (or base mount). Hoses/cables follow the
/// link back toward that point — the way real arms route utilities.
pub fn assemble_joint(
    joint_i: usize,
    design: &ActuatorDesign,
    origin: V3,
    axis: V3,
    proximal: V3,
) -> Vec<DesignPart> {
    match design {
        ActuatorDesign::Gear(g) => assemble_gear(joint_i, g, origin, axis),
        ActuatorDesign::Hydraulic(h) => assemble_hydraulic(joint_i, h, origin, axis, proximal),
        ActuatorDesign::Tendon(t) => assemble_tendon(joint_i, t, origin, axis, proximal),
    }
}

fn assemble_gear(ji: usize, g: &GearMechanism, origin: V3, axis: V3) -> Vec<DesignPart> {
    let a = unit(axis);
    let r = g.od_mm * 0.5;
    let mut parts = Vec::with_capacity(6);

    // Main can — the module you actually buy.
    parts.push(DesignPart {
        role: PartRole::Housing,
        shape: PartShape::Capsule,
        name: "actuator_can",
        origin,
        axis: a,
        radius_mm: r,
        length_mm: g.len_mm,
        rgb: [0.42, 0.46, 0.52],
        metalness: 0.75,
        roughness: 0.38,
        joint: ji as i32,
    });

    // Thin output faces (mounting flanges), flush with the can ends.
    let half = g.len_mm * 0.5;
    for sign in [-1.0_f64, 1.0] {
        parts.push(DesignPart {
            role: PartRole::Flange,
            shape: PartShape::Disk,
            name: "mount_flange",
            origin: add(origin, scale(a, sign * (half + 2.0))),
            axis: a,
            radius_mm: r * 1.08,
            length_mm: 3.5,
            rgb: [0.58, 0.62, 0.68],
            metalness: 0.85,
            roughness: 0.3,
            joint: ji as i32,
        });
    }

    // Harmonic: one gold ring on the OD (circular spline cue), not a second can.
    if g.show_harmonic_ring {
        parts.push(DesignPart {
            role: PartRole::HarmonicCup,
            shape: PartShape::Disk,
            name: "circular_spline",
            origin: add(origin, scale(a, half * 0.35)),
            axis: a,
            radius_mm: r * 1.02,
            length_mm: 5.0,
            rgb: [0.72, 0.58, 0.32],
            metalness: 0.8,
            roughness: 0.35,
            joint: ji as i32,
        });
    }

    // Hollow-bore cue (HEBI 15 mm thru).
    parts.push(DesignPart {
        role: PartRole::Shaft,
        shape: PartShape::Capsule,
        name: "thru_bore",
        origin,
        axis: a,
        radius_mm: 6.0,
        length_mm: g.len_mm + 4.0,
        rgb: [0.15, 0.16, 0.18],
        metalness: 0.4,
        roughness: 0.6,
        joint: ji as i32,
    });

    parts
}

fn assemble_hydraulic(
    ji: usize,
    h: &HydraulicMechanism,
    origin: V3,
    axis: V3,
    proximal: V3,
) -> Vec<DesignPart> {
    let a = unit(axis);
    let (u, _v) = orthonormal(a);
    let tint = fluid_tint(&h.fluid);
    let r = h.od_mm * 0.5;
    let mut parts = Vec::with_capacity(8);

    // Compact vane cartridge (Micromatic-style short disk).
    parts.push(DesignPart {
        role: PartRole::VaneBody,
        shape: PartShape::Capsule,
        name: "vane_cartridge",
        origin,
        axis: a,
        radius_mm: r,
        length_mm: h.width_mm,
        rgb: [0.48, 0.5, 0.46],
        metalness: 0.7,
        roughness: 0.4,
        joint: ji as i32,
    });
    for sign in [-1.0_f64, 1.0] {
        parts.push(DesignPart {
            role: PartRole::Flange,
            shape: PartShape::Disk,
            name: "vane_end",
            origin: add(origin, scale(a, sign * (h.width_mm * 0.5 + 2.0))),
            axis: a,
            radius_mm: r * 1.05,
            length_mm: 3.0,
            rgb: [0.55, 0.58, 0.54],
            metalness: 0.8,
            roughness: 0.35,
            joint: ji as i32,
        });
    }

    // Ports as short nipples on the OD.
    let port = add(origin, scale(u, r + 4.0));
    parts.push(DesignPart {
        role: PartRole::FluidPort,
        shape: PartShape::Capsule,
        name: "ports",
        origin: port,
        axis: u,
        radius_mm: 3.5,
        length_mm: 10.0,
        rgb: [0.55, 0.22, 0.14],
        metalness: 0.65,
        roughness: 0.4,
        joint: ji as i32,
    });

    // Two lines hug the link back to the proximal joint (not a spaghetti drop).
    let along = unit(sub(proximal, origin));
    let side = unit(cross(a, along));
    for (k, name) in [(1.0, "hose_p"), (-1.0, "hose_t")] {
        let start = add(port, scale(side, k * 5.0));
        // Stay outside the can, land near proximal on the same side.
        let end = add(
            add(proximal, scale(side, k * 12.0)),
            scale(along, -8.0),
        );
        let (o, dir, len) = segment(start, end);
        parts.push(DesignPart {
            role: PartRole::Hose,
            shape: PartShape::Segment,
            name,
            origin: o,
            axis: dir,
            radius_mm: h.hose_od_mm * 0.5,
            length_mm: len,
            rgb: tint,
            metalness: 0.05,
            roughness: 0.85,
            joint: ji as i32,
        });
    }

    // One small reservoir under the base joint only.
    if ji == 0 {
        let res = add(proximal, [0.0, 0.0, 28.0]);
        parts.push(DesignPart {
            role: PartRole::Reservoir,
            shape: PartShape::Capsule,
            name: "reservoir",
            origin: res,
            axis: [0.0, 0.0, 1.0],
            radius_mm: 18.0,
            length_mm: 40.0,
            rgb: tint,
            metalness: 0.08,
            roughness: 0.5,
            joint: -1,
        });
    }

    parts
}

fn assemble_tendon(
    ji: usize,
    t: &TendonMechanism,
    origin: V3,
    axis: V3,
    proximal: V3,
) -> Vec<DesignPart> {
    let a = unit(axis);
    let along = unit(sub(proximal, origin));
    let side = unit(cross(a, along));
    let tint = cable_tint(&t.material);
    let mut parts = Vec::with_capacity(8);

    // Small joint pulley on the hinge — not a fat gear can.
    parts.push(DesignPart {
        role: PartRole::Pulley,
        shape: PartShape::Disk,
        name: "joint_pulley",
        origin,
        axis: a,
        radius_mm: t.pulley_od_mm * 0.5,
        length_mm: t.pulley_w_mm,
        rgb: [0.38, 0.4, 0.44],
        metalness: 0.7,
        roughness: 0.4,
        joint: ji as i32,
    });
    for sign in [-1.0_f64, 1.0] {
        parts.push(DesignPart {
            role: PartRole::Bearing,
            shape: PartShape::Disk,
            name: "cheek",
            origin: add(origin, scale(a, sign * (t.pulley_w_mm * 0.55 + 1.5))),
            axis: a,
            radius_mm: t.pulley_od_mm * 0.42,
            length_mm: 2.5,
            rgb: [0.55, 0.58, 0.62],
            metalness: 0.8,
            roughness: 0.35,
            joint: ji as i32,
        });
    }

    // Antagonistic pair runs along the link (top/bottom), like a cable-driven arm.
    let leave_r = t.pulley_od_mm * 0.5;
    for (k, name) in [(1.0, "cable_plus"), (-1.0, "cable_minus")] {
        let start = add(origin, add(scale(side, k * leave_r), scale(along, 4.0)));
        let end = add(proximal, scale(side, k * 10.0));
        let (o, dir, len) = segment(start, end);
        parts.push(DesignPart {
            role: PartRole::Cable,
            shape: PartShape::Segment,
            name,
            origin: o,
            axis: dir,
            radius_mm: t.cable_od_mm * 0.5,
            length_mm: len,
            rgb: tint,
            metalness: if t.material.name == "steel" { 0.85 } else { 0.05 },
            roughness: if t.material.name == "steel" { 0.35 } else { 0.7 },
            joint: ji as i32,
        });
    }

    // Dual winch drum only at the base (joint 0) — motors live here, not at every hinge.
    if ji == 0 {
        let w = add(proximal, [0.0, 0.0, 22.0]);
        parts.push(DesignPart {
            role: PartRole::Winch,
            shape: PartShape::Disk,
            name: "base_winch",
            origin: w,
            axis: [0.0, 1.0, 0.0],
            radius_mm: t.winch_od_mm * 0.5,
            length_mm: 22.0,
            rgb: [0.28, 0.3, 0.34],
            metalness: 0.7,
            roughness: 0.4,
            joint: -1,
        });
        parts.push(DesignPart {
            role: PartRole::Motor,
            shape: PartShape::Capsule,
            name: "winch_motor",
            origin: add(w, [0.0, 18.0, 0.0]),
            axis: [0.0, 1.0, 0.0],
            radius_mm: 11.0,
            length_mm: 20.0,
            rgb: [0.18, 0.2, 0.24],
            metalness: 0.4,
            roughness: 0.55,
            joint: -1,
        });
    }

    parts
}

pub const PART_STRIDE: usize = 16;

pub fn parts_flat(parts: &[DesignPart]) -> Vec<f64> {
    let mut out = Vec::with_capacity(parts.len() * PART_STRIDE);
    for p in parts {
        out.push(p.joint as f64);
        out.push(p.role as u8 as f64);
        out.extend_from_slice(&p.origin);
        out.extend_from_slice(&p.axis);
        out.push(p.radius_mm);
        out.push(p.length_mm);
        out.push(p.rgb[0] as f64);
        out.push(p.rgb[1] as f64);
        out.push(p.rgb[2] as f64);
        out.push(p.metalness as f64);
        out.push(p.roughness as f64);
        out.push(p.shape as u8 as f64);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinematics::Drive;

    fn parts_for(d: Drive) -> Vec<DesignPart> {
        let des = ActuatorDesign::from_drive(&d);
        assemble_joint(
            0,
            &des,
            [100.0, 0.0, 80.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        )
    }

    #[test]
    fn modules_stay_compact() {
        for d in [Drive::direct(), Drive::qdd(), Drive::high_ratio()] {
            let parts = parts_for(d);
            let housing = parts
                .iter()
                .find(|p| p.role == PartRole::Housing)
                .expect("gear module needs a housing");
            assert!(
                housing.radius_mm <= 40.0 && housing.length_mm <= 58.0,
                "oversized module: r={} L={}",
                housing.radius_mm,
                housing.length_mm
            );
            assert!(parts.len() <= 8, "too many parts: {}", parts.len());
        }
    }

    #[test]
    fn hydraulic_hoses_follow_link_not_giant_drop() {
        let parts = parts_for(Drive::hydraulic());
        let hose = parts
            .iter()
            .find(|p| p.role == PartRole::Hose)
            .expect("hose");
        assert!(hose.radius_mm < 4.0);
        assert!(hose.length_mm < 250.0, "hose should track the link, got {}", hose.length_mm);
        assert!(parts.iter().any(|p| p.role == PartRole::VaneBody));
    }

    #[test]
    fn tendon_is_pulley_plus_cables() {
        let parts = parts_for(Drive::tendon());
        let pulley = parts
            .iter()
            .find(|p| p.role == PartRole::Pulley)
            .expect("pulley");
        assert!(pulley.radius_mm <= 18.0);
        assert!(parts.iter().filter(|p| p.role == PartRole::Cable).count() >= 2);
        assert!(parts.iter().any(|p| p.role == PartRole::Winch));
    }

    #[test]
    fn high_ratio_has_harmonic_ring_not_second_can() {
        let parts = parts_for(Drive::high_ratio());
        assert!(parts.iter().any(|p| p.role == PartRole::HarmonicCup));
        assert_eq!(
            parts.iter().filter(|p| p.role == PartRole::Housing).count(),
            1
        );
    }
}
