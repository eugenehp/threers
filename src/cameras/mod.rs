//! Cameras: position/orientation in world space plus a projection matrix.

mod orthographic;
mod perspective;

pub use orthographic::OrthographicCamera;
pub use perspective::{PerspectiveCamera, ProjectionKind, SensorFit};

use crate::core::Layers;
use crate::math::{Matrix4, Vector3};

/// Unified camera trait used by the Renderer.
pub trait Camera {
    fn view_matrix(&self) -> Matrix4;
    fn projection_matrix(&self) -> Matrix4;
    fn position(&self) -> Vector3;
    fn set_aspect(&mut self, aspect: f32);

    /// Clip-space near/far planes (for SSAO depth encoding in normal prepass).
    fn near_far(&self) -> (f32, f32) {
        (0.1, 1000.0)
    }

    /// Layer mask. Meshes are visible to this camera iff `camera.layers().test(&mesh.layers)`.
    fn layers(&self) -> Layers {
        Layers::default()
    }
}
