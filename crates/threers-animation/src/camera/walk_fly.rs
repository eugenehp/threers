//! Walk / fly navigation recording → pose samples / paths.

use super::bake::OrbitRecorder;
use super::pose::CameraPose;
use threers::math::Vector3;

/// First-person walk / fly state (Blender walk/fly navigate lite).
#[derive(Debug, Clone, PartialEq)]
pub struct WalkFlyNav {
    pub eye: Vector3,
    pub yaw: f32,
    pub pitch: f32,
    pub move_speed: f32,
    pub look_speed: f32,
    pub flying: bool,
    /// Gravity when walking (world -Y).
    pub gravity: f32,
    velocity_y: f32,
    pub recorder: OrbitRecorder,
    time: f32,
}

impl Default for WalkFlyNav {
    fn default() -> Self {
        Self {
            eye: Vector3::new(0.0, 1.7, 5.0),
            yaw: 0.0,
            pitch: 0.0,
            move_speed: 4.0,
            look_speed: 1.5,
            flying: false,
            gravity: 9.8,
            velocity_y: 0.0,
            recorder: OrbitRecorder::new(),
            time: 0.0,
        }
    }
}

/// One frame of walk/fly input.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct WalkFlyInput {
    pub forward: f32,
    pub strafe: f32,
    pub up: f32,
    pub yaw_delta: f32,
    pub pitch_delta: f32,
    pub jump: bool,
}

impl WalkFlyNav {
    pub fn new(eye: Vector3) -> Self {
        Self {
            eye,
            ..Self::default()
        }
    }

    pub fn forward_dir(&self) -> Vector3 {
        Vector3::new(self.yaw.sin(), 0.0, -self.yaw.cos())
    }

    pub fn right_dir(&self) -> Vector3 {
        Vector3::new(self.yaw.cos(), 0.0, self.yaw.sin())
    }

    pub fn look_dir(&self) -> Vector3 {
        let cp = self.pitch.cos();
        Vector3::new(self.yaw.sin() * cp, self.pitch.sin(), -self.yaw.cos() * cp).normalize()
    }

    pub fn pose(&self, fov: f32) -> CameraPose {
        let target = self.eye + self.look_dir();
        CameraPose::from_look_at(self.eye, target, fov)
    }

    pub fn update(&mut self, input: WalkFlyInput, dt: f32, record: bool) {
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        self.time += dt;
        self.yaw += input.yaw_delta * self.look_speed;
        self.pitch = (self.pitch + input.pitch_delta * self.look_speed)
            .clamp(-1.5, 1.5);

        let f = self.forward_dir();
        let r = self.right_dir();
        let mut move_dir =
            f * input.forward + r * input.strafe + Vector3::UP * input.up;
        if self.flying {
            move_dir = move_dir + self.look_dir() * input.forward * 0.0; // already in f
        }
        if move_dir.length_sq() > 1e-8 {
            self.eye = self.eye + move_dir.normalize() * self.move_speed * dt;
        }

        if !self.flying {
            if input.jump && self.eye.y <= 1.7 + 1e-3 {
                self.velocity_y = 4.0;
            }
            self.velocity_y -= self.gravity * dt;
            self.eye.y += self.velocity_y * dt;
            if self.eye.y < 1.7 {
                self.eye.y = 1.7;
                self.velocity_y = 0.0;
            }
        }

        if record {
            self.recorder
                .push(self.time, self.pose(50.0_f32.to_radians()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walk_moves_forward() {
        let mut nav = WalkFlyNav::new(Vector3::new(0.0, 1.7, 0.0));
        nav.update(
            WalkFlyInput {
                forward: 1.0,
                ..WalkFlyInput::default()
            },
            0.5,
            true,
        );
        assert!(nav.eye.z < 0.0);
        assert!(!nav.recorder.is_empty());
    }
}
