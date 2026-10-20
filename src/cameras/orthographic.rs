use super::Camera;
use crate::math::{Matrix4, Vector3};

#[derive(Debug, Clone)]
pub struct OrthographicCamera {
    pub position: Vector3,
    pub target: Vector3,
    pub up: Vector3,
    pub left: f32,
    pub right: f32,
    pub top: f32,
    pub bottom: f32,
    pub near: f32,
    pub far: f32,
    /// Channel mask — meshes are drawn iff this intersects `Object3D::layers`
    /// (three.js `camera.layers`).
    pub layers: crate::core::Layers,
    /// When set, the view matrix as given (see
    /// [`PerspectiveCamera::view_override`](crate::PerspectiveCamera::view_override)).
    pub view_override: Option<super::ViewOverride>,
}

impl OrthographicCamera {
    pub fn new(left: f32, right: f32, top: f32, bottom: f32, near: f32, far: f32) -> Self {
        Self {
            position: Vector3::new(0.0, 0.0, 5.0),
            target: Vector3::ZERO,
            up: Vector3::UP,
            left,
            right,
            top,
            bottom,
            near,
            far,
            layers: crate::core::Layers::default(),
            view_override: None,
        }
    }

    /// Point the camera at a world-space position. Chainable, mirroring
    /// [`PerspectiveCamera::look_at`](crate::PerspectiveCamera::look_at).
    pub fn look_at(&mut self, target: Vector3) -> &mut Self {
        self.target = target;
        self
    }
}

impl Camera for OrthographicCamera {
    fn view_matrix(&self) -> Matrix4 {
        self.view_override
            .and_then(|o| o.for_pose(self.position, self.target, self.up))
            .unwrap_or_else(|| Matrix4::look_at(self.position, self.target, self.up))
    }
    fn projection_matrix(&self) -> Matrix4 {
        Matrix4::orthographic(
            self.left,
            self.right,
            self.top,
            self.bottom,
            self.near,
            self.far,
        )
    }
    fn position(&self) -> Vector3 {
        self.position
    }
    fn look_at(&mut self, target: Vector3) {
        self.target = target;
    }
    fn set_aspect(&mut self, aspect: f32) {
        // Preserve vertical span; rescale horizontal symmetrically.
        let v_half = (self.top - self.bottom).abs() * 0.5;
        let h_half = v_half * aspect;
        let cx = (self.left + self.right) * 0.5;
        self.left = cx - h_half;
        self.right = cx + h_half;
    }
    fn near_far(&self) -> (f32, f32) {
        (self.near, self.far)
    }
    fn layers(&self) -> crate::core::Layers {
        self.layers
    }
}
