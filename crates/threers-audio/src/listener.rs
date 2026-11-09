use threers_math::Vector3;

/// Receiver of positional audio. One per scene; the renderer (or the user)
/// updates `position`/`forward`/`up` to match the camera each frame.
#[derive(Debug, Clone, Copy)]
pub struct AudioListener {
    pub position: Vector3,
    pub forward: Vector3,
    pub up: Vector3,
    pub master_volume: f32,
}

impl Default for AudioListener {
    fn default() -> Self {
        Self {
            position: Vector3::ZERO,
            forward: Vector3::new(0.0, 0.0, -1.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            master_volume: 1.0,
        }
    }
}
