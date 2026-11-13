//! Motion for any box: spinning, rocking, rolling, bouncing, buzzing,
//! pulsing or floating, on its own and for as long as it is set. A button
//! that buzzes when something is wrong, a badge that pulses, an icon that
//! bobs, a card that turns round and round.
//!
//! [`Ui::set_motion`] drives the box's [3D transform](crate::depth) from
//! the UI's clock, over whatever transform it has of its own: the box and
//! everything in it move as drawn, with nothing laid out again, in every
//! painter. It asks for frames only while the box shows, and stands still
//! where the UI asks for reduced motion
//! ([`Ui::set_reduce_motion`]). The [indicators](crate::indicators) move
//! the same ways inside their shaders, where solid ones turn in true 3D.
//!
//! ```
//! use threers_ui::motion::{Motion, MotionSpec};
//! use threers_ui::{style, FixedWidthText, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let badge = ui.add_box(root, style! { width: 40px; height: 40px; border-radius: 20px; background: #e11d48; });
//! ui.set_motion(badge, MotionSpec::new(Motion::Bounce).amount(0.5));
//! ui.layout(200.0, 200.0, &mut FixedWidthText(7.0));
//! assert!(ui.needs_frame());
//! ui.step(0.3);
//! assert!(ui.transform3d(badge).unwrap().translate_y < 0.0, "in the air");
//! ui.set_motion(badge, Motion::None);
//! assert_eq!(ui.transform3d(badge), None, "back as it was");
//! ```

use crate::tree::{NodeId, Ui};

/// How a box moves on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Motion {
    #[default]
    None,
    /// Turning round its vertical axis, on and on.
    Spin,
    /// Turning to one side and back, and nodding a little.
    Rock,
    /// Turning in its own plane (an indicator's bar tumbles end over end
    /// about its length instead).
    Roll,
    /// Hopping, squashed where it lands: a dribbled ball.
    Bounce,
    /// Shaking in short bursts, as a phone buzzes.
    Buzz,
    /// Breathing: a little larger, a little smaller.
    Pulse,
    /// Bobbing gently, as if on water.
    Float,
}

impl Motion {
    /// Every motion, in order.
    pub const ALL: [Motion; 8] = [Motion::None, Motion::Spin, Motion::Rock, Motion::Roll, Motion::Bounce, Motion::Buzz, Motion::Pulse, Motion::Float];

    pub fn name(self) -> &'static str {
        match self {
            Motion::None => "None",
            Motion::Spin => "Spin",
            Motion::Rock => "Rock",
            Motion::Roll => "Roll",
            Motion::Bounce => "Bounce",
            Motion::Buzz => "Buzz",
            Motion::Pulse => "Pulse",
            Motion::Float => "Float",
        }
    }
}

/// A motion and how it is played.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionSpec {
    pub motion: Motion,
    /// How fast, 1 as designed.
    pub speed: f32,
    /// How far, 1 as designed.
    pub amount: f32,
}

impl MotionSpec {
    pub fn new(motion: Motion) -> Self {
        Self { motion, speed: 1.0, amount: 1.0 }
    }

    pub fn speed(self, speed: f32) -> Self {
        Self { speed: speed.max(0.0), ..self }
    }

    pub fn amount(self, amount: f32) -> Self {
        Self { amount: amount.max(0.0), ..self }
    }
}

impl From<Motion> for MotionSpec {
    fn from(m: Motion) -> Self {
        MotionSpec::new(m)
    }
}

/// Where a moving box is `t` seconds in, `hop` pixels its highest hop:
/// turned (yaw, pitch, roll, radians), moved (x, y pixels, down positive),
/// and scaled (x, y). The indicators' shaders play the same.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Pose {
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    pub offset: (f32, f32),
    pub scale: (f32, f32),
}

impl MotionSpec {
    pub fn pose(&self, time: f32, hop: f32) -> Pose {
        let t = time * self.speed;
        let a = self.amount;
        let mut p = Pose { scale: (1.0, 1.0), ..Pose::default() };
        match self.motion {
            Motion::None => {}
            Motion::Spin => p.yaw = t * 2.4,
            Motion::Rock => {
                p.yaw = (t * 2.2).sin() * 0.65 * a;
                p.pitch = (t * 1.1 + 1.0).sin() * 0.18 * a;
            }
            Motion::Roll => p.roll = t * 2.6,
            Motion::Bounce => {
                let ph = (t * 1.25).rem_euclid(1.0);
                let up = 4.0 * ph * (1.0 - ph);
                let land = ph.min(1.0 - ph) * 14.0;
                let s = 0.2 * a * (-land * land).exp();
                p.scale = (1.0 + s, 1.0 - s);
                p.offset.1 = -up * hop * a;
            }
            Motion::Buzz => {
                let b = (t * 0.9).rem_euclid(1.0);
                let burst = smoothstep(0.0, 0.03, b) * (1.0 - smoothstep(0.3, 0.36, b));
                p.offset = ((t * 97.0).sin() * hop * 0.12 * a * burst, (t * 83.0).cos() * hop * 0.12 * a * burst);
                p.roll = (t * 71.0).sin() * 0.05 * a * burst;
            }
            Motion::Pulse => {
                let s = 1.0 + 0.07 * a * (t * 3.4).sin();
                p.scale = (s, s);
            }
            Motion::Float => {
                p.offset.1 = (t * 1.7).sin() * hop * 0.3 * a;
                p.roll = (t * 1.2).sin() * 0.06 * a;
                p.pitch = (t * 1.4).sin() * 0.15 * a;
                p.yaw = (t * 0.9).sin() * 0.2 * a;
            }
        }
        p
    }
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl Ui {
    /// Moves `node` on its own (see the [module docs](self)); `Motion::None`
    /// stops it and puts back the transform it had.
    pub fn set_motion(&mut self, node: NodeId, motion: impl Into<MotionSpec>) {
        let spec = motion.into();
        if spec.motion == Motion::None {
            if let Some((_, base)) = self.env.motions.remove(&node) {
                match base {
                    Some(t) => self.set_transform3d(node, t),
                    None => self.clear_transform3d(node),
                }
                self.env.repaint.set(true);
            }
            return;
        }
        let base = match self.env.motions.get(&node) {
            Some(&(_, base)) => base,
            None => self.transform3d(node),
        };
        self.env.motions.insert(node, (spec, base));
        self.place_motion(node);
    }

    /// How `node` moves on its own, if it does.
    pub fn motion(&self, node: NodeId) -> Option<MotionSpec> {
        self.env.motions.get(&node).map(|m| m.0)
    }

    /// Whether a moving box shows, so frames must keep coming.
    pub(crate) fn motions_animate(&self) -> bool {
        !self.env.reduce_motion && self.env.motions.keys().any(|&n| self.is_on_screen(n))
    }

    /// Every moving box where the clock has it (the ones out of sight
    /// wait), dropping those removed.
    pub(crate) fn step_motions(&mut self) {
        if self.env.motions.is_empty() {
            return;
        }
        let gone: Vec<NodeId> = self.env.motions.keys().copied().filter(|&n| !self.contains(n)).collect();
        for n in gone {
            self.env.motions.remove(&n);
        }
        let nodes: Vec<NodeId> = self.env.motions.keys().copied().collect();
        for n in nodes {
            if self.env.reduce_motion || self.is_on_screen(n) {
                self.place_motion(n);
            }
        }
    }

    /// `node`'s transform: its own, moved where its motion is now (or
    /// still, with reduced motion).
    fn place_motion(&mut self, node: NodeId) {
        let Some(&(spec, base)) = self.env.motions.get(&node) else { return };
        let mut t = base.unwrap_or_default();
        if !self.env.reduce_motion {
            let r = self.rect(node);
            let hop = (r.width.min(r.height) * 0.3).clamp(4.0, 40.0);
            let p = spec.pose(self.clock as f32, hop);
            t.rotate_y += p.yaw;
            t.rotate_x += p.pitch;
            t.rotate_z += p.roll;
            t.translate_x += p.offset.0;
            // Squashed on the ground, not about its middle: lowered by
            // what it lost.
            t.translate_y += p.offset.1 + r.height * 0.5 * (1.0 - p.scale.1);
            // One scale: the mean of the two (a 2D painter has no squash).
            t.scale *= (p.scale.0 + p.scale.1) * 0.5;
            if p.yaw != 0.0 || p.pitch != 0.0 {
                t.perspective = t.perspective.min(r.width.max(r.height) * 3.0).max(200.0);
            }
        }
        self.set_transform3d(node, t);
        self.env.repaint.set(true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::depth::Transform3d;
    use crate::{style, FixedWidthText};

    #[test]
    fn every_motion_moves_and_none_stays() {
        for m in Motion::ALL {
            let spec = MotionSpec::new(m);
            let a = spec.pose(0.13, 20.0);
            let b = spec.pose(0.41, 20.0);
            assert_eq!(a == b, m == Motion::None, "{m:?}");
        }
    }

    #[test]
    fn a_moving_box_waits_out_of_sight_and_for_reduced_motion() {
        let mut ui = Ui::new();
        let root = ui.root();
        let page = ui.add_box(root, style! { width: 200px; height: 200px; });
        let b = ui.add_box(page, style! { width: 40px; height: 40px; });
        ui.set_motion(b, Motion::Spin);
        ui.layout(300.0, 300.0, &mut FixedWidthText(7.0));
        ui.step(0.2);
        let turned = ui.transform3d(b).unwrap().rotate_y;
        assert!(turned > 0.0);
        // Hidden: no frames, and it does not move.
        ui.update_style(page, |s| s.display = crate::Display::None);
        ui.layout(300.0, 300.0, &mut FixedWidthText(7.0));
        ui.paint(&mut crate::scene::Scene::default().recorder());
        assert!(!ui.needs_frame());
        ui.step(0.2);
        assert_eq!(ui.transform3d(b).unwrap().rotate_y, turned);
        // Shown, with reduced motion: still, and at rest.
        ui.update_style(page, |s| s.display = crate::Display::Block);
        ui.set_reduce_motion(true);
        ui.layout(300.0, 300.0, &mut FixedWidthText(7.0));
        ui.step(0.2);
        ui.paint(&mut crate::scene::Scene::default().recorder());
        assert_eq!(ui.transform3d(b).unwrap().rotate_y, 0.0);
        assert!(!ui.needs_frame());
    }

    #[test]
    fn stopping_puts_back_its_own_transform() {
        let mut ui = Ui::new();
        let root = ui.root();
        let b = ui.add_box(root, style! { width: 40px; height: 40px; });
        let own = Transform3d { rotate_x: 0.3, ..Transform3d::default() };
        ui.set_transform3d(b, own);
        ui.set_motion(b, MotionSpec::new(Motion::Rock));
        ui.layout(300.0, 300.0, &mut FixedWidthText(7.0));
        ui.step(0.5);
        assert_ne!(ui.transform3d(b), Some(own));
        ui.set_motion(b, Motion::None);
        assert_eq!(ui.transform3d(b), Some(own));
    }
}
