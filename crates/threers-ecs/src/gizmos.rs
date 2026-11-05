//! Lines drawn for one frame, to see what code is thinking: where an AI aims,
//! what a ray hit, how big a trigger is.
//!
//! ```
//! # use threers_ecs::prelude::*;
//! fn show_aim(w: &mut World) {
//!     let gizmos = w.resource_mut::<Gizmos>();
//!     gizmos.line(Vector3::ZERO, Vector3::new(0.0, 0.0, -5.0), Color::from_hex(0xff0000));
//!     gizmos.sphere(Vector3::ZERO, 1.0, Color::from_hex(0x00ff00));
//! }
//! ```
//!
//! What a fixed system draws stays up until the next fixed step, since a frame
//! may run none; what a frame system draws lasts that frame. Every [`App`](crate::App)
//! has a [`Gizmos`], and a release build can turn it off with `enabled`.

use crate::world::{Entity, World};
use std::sync::Arc;
use threers::core::{BufferAttribute, BufferGeometry, LineSegments, Object3D, ObjectKind};
use threers::materials::{LineBasicMaterial, Material};
use threers::math::{Color, Vector3};

#[derive(Debug, Clone, Copy)]
struct Segment(Vector3, Vector3, u32);

/// The lines to draw, as a resource.
#[derive(Debug)]
pub struct Gizmos {
    /// Off, everything drawn is dropped.
    pub enabled: bool,
    frame: Vec<Segment>,
    fixed: Vec<Segment>,
    in_fixed: bool,
    /// One node per colour, reused from frame to frame.
    nodes: Vec<(u32, Entity)>,
}

impl Default for Gizmos {
    fn default() -> Self {
        Self {
            enabled: true,
            frame: Vec::new(),
            fixed: Vec::new(),
            in_fixed: false,
            nodes: Vec::new(),
        }
    }
}

fn key(color: Color) -> u32 {
    let c = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u32;
    c(color.r) << 16 | c(color.g) << 8 | c(color.b)
}

impl Gizmos {
    pub fn line(&mut self, from: Vector3, to: Vector3, color: Color) -> &mut Self {
        if self.enabled {
            let segment = Segment(from, to, key(color));
            if self.in_fixed {
                self.fixed.push(segment)
            } else {
                self.frame.push(segment)
            }
        }
        self
    }

    /// From `origin` along `direction`, as far as it is long.
    pub fn ray(&mut self, origin: Vector3, direction: Vector3, color: Color) -> &mut Self {
        self.line(origin, origin + direction, color)
    }

    /// A line with a head at `to`.
    pub fn arrow(&mut self, from: Vector3, to: Vector3, color: Color) -> &mut Self {
        let along = to - from;
        let length = along.length();
        if length <= f32::EPSILON {
            return self;
        }
        let dir = along * (1.0 / length);
        let side = if dir.y.abs() < 0.9 {
            dir.cross(Vector3::UP)
        } else {
            dir.cross(Vector3::new(1.0, 0.0, 0.0))
        };
        let (side, up) = (side.normalize(), dir.cross(side).normalize());
        let head = length.min(1.0) * 0.25;
        self.line(from, to, color);
        for wing in [side, side * -1.0, up, up * -1.0] {
            self.line(to, to - dir * head + wing * head * 0.5, color);
        }
        self
    }

    /// Three short lines crossing at `at`.
    pub fn point(&mut self, at: Vector3, size: f32, color: Color) -> &mut Self {
        let h = size / 2.0;
        for axis in [
            Vector3::new(h, 0.0, 0.0),
            Vector3::new(0.0, h, 0.0),
            Vector3::new(0.0, 0.0, h),
        ] {
            self.line(at - axis, at + axis, color);
        }
        self
    }

    /// The edges of a box, `half` its size each way, not turned.
    pub fn cube(&mut self, center: Vector3, half: Vector3, color: Color) -> &mut Self {
        let corner = |i: u32| {
            let s = |bit: u32| if i & bit != 0 { 1.0 } else { -1.0 };
            center + Vector3::new(half.x * s(1), half.y * s(2), half.z * s(4))
        };
        for a in 0..8u32 {
            for bit in [1, 2, 4] {
                if a & bit == 0 {
                    self.line(corner(a), corner(a | bit), color);
                }
            }
        }
        self
    }

    /// Three circles round `center`.
    pub fn sphere(&mut self, center: Vector3, radius: f32, color: Color) -> &mut Self {
        let x = Vector3::new(1.0, 0.0, 0.0);
        let y = Vector3::UP;
        let z = Vector3::new(0.0, 0.0, 1.0);
        self.circle(center, x, y, radius, color);
        self.circle(center, y, z, radius, color);
        self.circle(center, z, x, radius, color)
    }

    /// A circle in the plane of `a` and `b`, which should be at right angles.
    pub fn circle(
        &mut self,
        center: Vector3,
        a: Vector3,
        b: Vector3,
        radius: f32,
        color: Color,
    ) -> &mut Self {
        const SIDES: usize = 24;
        let at = |i: usize| {
            let t = i as f32 / SIDES as f32 * std::f32::consts::TAU;
            center + a * (t.cos() * radius) + b * (t.sin() * radius)
        };
        for i in 0..SIDES {
            self.line(at(i), at(i + 1), color);
        }
        self
    }

    /// Through each point in turn; `closed` joins the last to the first.
    pub fn path(&mut self, points: &[Vector3], closed: bool, color: Color) -> &mut Self {
        for pair in points.windows(2) {
            self.line(pair[0], pair[1], color);
        }
        if closed && points.len() > 2 {
            self.line(points[points.len() - 1], points[0], color);
        }
        self
    }

    /// Lines waiting to be drawn this frame.
    pub fn len(&self) -> usize {
        self.frame.len() + self.fixed.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn enter_fixed_step(&mut self) {
        self.fixed.clear();
        self.in_fixed = true;
    }

    pub(crate) fn leave_fixed(&mut self) {
        self.in_fixed = false;
    }
}

/// Turn this frame's lines into nodes, one per colour, and forget the frame's.
pub(crate) fn draw_gizmos(w: &mut World) {
    let Some(gizmos) = w.get_resource_mut::<Gizmos>() else {
        return;
    };
    let mut by_color: Vec<(u32, Vec<f32>)> = Vec::new();
    for Segment(a, b, color) in gizmos.fixed.iter().chain(&gizmos.frame) {
        let at = match by_color.iter().position(|(c, _)| c == color) {
            Some(at) => at,
            None => {
                by_color.push((*color, Vec::new()));
                by_color.len() - 1
            }
        };
        by_color[at].1.extend([a.x, a.y, a.z, b.x, b.y, b.z]);
    }
    gizmos.frame.clear();
    let mut nodes = std::mem::take(&mut gizmos.nodes);
    nodes.retain(|&(_, id)| w.is_alive(id));
    for (color, id) in &nodes {
        if !by_color.iter().any(|(c, _)| c == color) {
            if let Some(node) = w.scene.get_mut(*id) {
                node.visible = false;
            }
        }
    }
    for (color, positions) in by_color {
        let mut geometry = BufferGeometry::new();
        geometry.set_attribute("position", BufferAttribute::new(positions, 3));
        geometry.compute_bounding_sphere();
        let found = nodes.iter().find(|(c, _)| *c == color).map(|&(_, id)| id);
        match found.and_then(|id| w.scene.get_mut(id)) {
            Some(node) => {
                node.visible = true;
                if let ObjectKind::LineSegments(lines) = &mut node.kind {
                    lines.geometry = Arc::new(geometry);
                }
            }
            None => {
                let material = LineBasicMaterial::new(Color::from_hex(color));
                let mut node =
                    Object3D::line_segments(LineSegments::new(geometry, Material::Line(material)));
                node.name = "gizmos".to_owned();
                let id = w.scene.add(node);
                nodes.push((color, id));
            }
        }
    }
    w.resource_mut::<Gizmos>().nodes = nodes;
}
