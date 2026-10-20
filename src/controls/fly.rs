//! three.js r165 `FlyControls`: keys and the pointer set a move state; each
//! `update(delta)` translates and turns the object by it.
//!
//! The DOM side (which event, which key code) is the caller's: it forwards
//! `keydown` / `keyup` codes and pointer events here, and applies the pose
//! [`FlyControls::update`] returns. All arithmetic is `f64`, as three.js's.

use super::transform::math::{self, Q, V3};

/// three.js's `moveState`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MoveState {
    pub up: f64,
    pub down: f64,
    pub left: f64,
    pub right: f64,
    pub forward: f64,
    pub back: f64,
    pub pitch_up: f64,
    pub pitch_down: f64,
    pub yaw_left: f64,
    pub yaw_right: f64,
    pub roll_left: f64,
    pub roll_right: f64,
}

/// three.js r165 `FlyControls`.
#[derive(Debug, Clone)]
pub struct FlyControls {
    pub enabled: bool,
    pub movement_speed: f64,
    pub roll_speed: f64,
    pub drag_to_look: bool,
    pub auto_forward: bool,
    /// Set by Shift (three.js keeps it but never reads it).
    pub movement_speed_multiplier: f64,
    pub status: i32,
    pub move_state: MoveState,
    pub move_vector: V3,
    pub rotation_vector: V3,
    last_quaternion: Q,
    last_position: V3,
}

impl Default for FlyControls {
    fn default() -> Self {
        Self::new()
    }
}

const EPS: f64 = 0.000001;

impl FlyControls {
    pub fn new() -> Self {
        let mut c = Self {
            enabled: true,
            movement_speed: 1.0,
            roll_speed: 0.005,
            drag_to_look: false,
            auto_forward: false,
            movement_speed_multiplier: 1.0,
            status: 0,
            move_state: MoveState::default(),
            move_vector: [0.0; 3],
            rotation_vector: [0.0; 3],
            last_quaternion: math::IDENTITY_Q,
            last_position: [0.0; 3],
        };
        c.update_movement_vector();
        c.update_rotation_vector();
        c
    }

    /// `keydown` with `event.code`; `alt` is `event.altKey`.
    pub fn key_down(&mut self, code: &str, alt: bool) {
        if alt || !self.enabled {
            return;
        }
        let s = &mut self.move_state;
        match code {
            "ShiftLeft" | "ShiftRight" => self.movement_speed_multiplier = 0.1,
            "KeyW" => s.forward = 1.0,
            "KeyS" => s.back = 1.0,
            "KeyA" => s.left = 1.0,
            "KeyD" => s.right = 1.0,
            "KeyR" => s.up = 1.0,
            "KeyF" => s.down = 1.0,
            "ArrowUp" => s.pitch_up = 1.0,
            "ArrowDown" => s.pitch_down = 1.0,
            "ArrowLeft" => s.yaw_left = 1.0,
            "ArrowRight" => s.yaw_right = 1.0,
            "KeyQ" => s.roll_left = 1.0,
            "KeyE" => s.roll_right = 1.0,
            _ => {}
        }
        self.update_movement_vector();
        self.update_rotation_vector();
    }

    /// `keyup` with `event.code`.
    pub fn key_up(&mut self, code: &str) {
        if !self.enabled {
            return;
        }
        let s = &mut self.move_state;
        match code {
            "ShiftLeft" | "ShiftRight" => self.movement_speed_multiplier = 1.0,
            "KeyW" => s.forward = 0.0,
            "KeyS" => s.back = 0.0,
            "KeyA" => s.left = 0.0,
            "KeyD" => s.right = 0.0,
            "KeyR" => s.up = 0.0,
            "KeyF" => s.down = 0.0,
            "ArrowUp" => s.pitch_up = 0.0,
            "ArrowDown" => s.pitch_down = 0.0,
            "ArrowLeft" => s.yaw_left = 0.0,
            "ArrowRight" => s.yaw_right = 0.0,
            "KeyQ" => s.roll_left = 0.0,
            "KeyE" => s.roll_right = 0.0,
            _ => {}
        }
        self.update_movement_vector();
        self.update_rotation_vector();
    }

    /// `pointerdown` with `event.button`.
    pub fn pointer_down(&mut self, button: i32) {
        if !self.enabled {
            return;
        }
        if self.drag_to_look {
            self.status += 1;
        } else {
            match button {
                0 => self.move_state.forward = 1.0,
                2 => self.move_state.back = 1.0,
                _ => {}
            }
            self.update_movement_vector();
        }
    }

    /// `pointermove` at page `(x, y)` over a container of `size` at `offset`.
    pub fn pointer_move(&mut self, page_x: f64, page_y: f64, size: [f64; 2], offset: [f64; 2]) {
        if !self.enabled {
            return;
        }
        if !self.drag_to_look || self.status > 0 {
            let half_w = size[0] / 2.0;
            let half_h = size[1] / 2.0;
            self.move_state.yaw_left = -((page_x - offset[0]) - half_w) / half_w;
            self.move_state.pitch_down = ((page_y - offset[1]) - half_h) / half_h;
            self.update_rotation_vector();
        }
    }

    /// `pointerup` with `event.button`.
    pub fn pointer_up(&mut self, button: i32) {
        if !self.enabled {
            return;
        }
        if self.drag_to_look {
            self.status -= 1;
            self.move_state.yaw_left = 0.0;
            self.move_state.pitch_down = 0.0;
        } else {
            match button {
                0 => self.move_state.forward = 0.0,
                2 => self.move_state.back = 0.0,
                _ => {}
            }
            self.update_movement_vector();
        }
        self.update_rotation_vector();
    }

    /// `pointercancel`.
    pub fn pointer_cancel(&mut self) {
        if !self.enabled {
            return;
        }
        if self.drag_to_look {
            self.status = 0;
            self.move_state.yaw_left = 0.0;
            self.move_state.pitch_down = 0.0;
        } else {
            self.move_state.forward = 0.0;
            self.move_state.back = 0.0;
            self.update_movement_vector();
        }
        self.update_rotation_vector();
    }

    pub fn update_movement_vector(&mut self) {
        let s = &self.move_state;
        let forward = if s.forward != 0.0 || (self.auto_forward && s.back == 0.0) {
            1.0
        } else {
            0.0
        };
        self.move_vector = [-s.left + s.right, -s.down + s.up, -forward + s.back];
    }

    pub fn update_rotation_vector(&mut self) {
        let s = &self.move_state;
        self.rotation_vector = [
            -s.pitch_down + s.pitch_up,
            -s.yaw_right + s.yaw_left,
            -s.roll_right + s.roll_left,
        ];
    }

    /// `update(delta)` on an object at `position` / `quaternion`: the new
    /// pose, and whether three.js would dispatch `change`. `None` when
    /// disabled (three.js leaves the object alone).
    pub fn update(&mut self, delta: f64, position: V3, quaternion: Q) -> Option<(V3, Q, bool)> {
        if !self.enabled {
            return None;
        }
        let move_mult = delta * self.movement_speed;
        let rot_mult = delta * self.roll_speed;
        let mut p = position;
        let axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        for (k, axis) in axes.into_iter().enumerate() {
            // translateOnAxis: `position.add(axis.applyQuaternion(q) * d)`.
            let v = math::scale(
                math::apply_quat(axis, quaternion),
                self.move_vector[k] * move_mult,
            );
            p = math::add(p, v);
        }
        let r = self.rotation_vector;
        let tmp = math::quat_normalize([r[0] * rot_mult, r[1] * rot_mult, r[2] * rot_mult, 1.0]);
        let q = math::quat_mul(quaternion, tmp);
        let dp = math::sub(self.last_position, p);
        let changed = math::length_sq(dp) > EPS
            || 8.0 * (1.0 - math::quat_dot(self.last_quaternion, q)) > EPS;
        if changed {
            self.last_quaternion = q;
            self.last_position = p;
        }
        Some((p, q, changed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w_moves_forward_along_minus_z() {
        let mut f = FlyControls::new();
        f.movement_speed = 2.0;
        f.key_down("KeyW", false);
        let (p, q, changed) = f.update(0.5, [0.0; 3], math::IDENTITY_Q).unwrap();
        assert_eq!(p, [0.0, 0.0, -1.0]);
        assert_eq!(q, math::IDENTITY_Q);
        assert!(changed);
        f.key_up("KeyW");
        let (p2, _, changed) = f.update(0.5, p, q).unwrap();
        assert_eq!(p2, p);
        assert!(!changed);
    }

    #[test]
    fn alt_ignores_keys() {
        let mut f = FlyControls::new();
        f.key_down("KeyW", true);
        assert_eq!(f.move_vector, [0.0, 0.0, 0.0]);
    }
}
