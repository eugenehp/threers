use super::Camera;
use crate::math::{Matrix4, Vector3};

/// How FOV maps to the film back (Blender sensor fit).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SensorFit {
    #[default]
    Auto,
    Horizontal,
    Vertical,
}

/// Projection model for the camera.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProjectionKind {
    #[default]
    Perspective,
    /// 360° × 180° equirectangular (latitude-longitude).
    Equirectangular,
    /// Fisheye with field covering up to ~180°+ (uses `fov` as max angle).
    FisheyeEquidistant,
    /// Reflective mirror-ball / angular map.
    MirrorBall,
}

#[derive(Debug, Clone)]
pub struct PerspectiveCamera {
    pub position: Vector3,
    pub target: Vector3,
    pub up: Vector3,
    pub fov: f32, // radians
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
    /// Focus distance for DoF / path-tracer aperture (world units). `0` = auto.
    pub focus_distance: f32,
    /// Aperture radius for shallow DoF. `0` = pinhole.
    pub aperture: f32,
    /// F-stop (alternative to raw aperture). `0` = unused.
    pub f_stop: f32,
    /// Diaphragm blade count for DoF bokeh (informational / path-tracer).
    pub aperture_blades: u32,
    /// Anamorphic squeeze ratio for DoF / bokeh (`1` = spherical).
    pub anamorphic_ratio: f32,
    /// Lens shift as a fraction of the sensor (±0.5 ≈ half-frame).
    pub shift_x: f32,
    pub shift_y: f32,
    pub sensor_fit: SensorFit,
    pub projection: ProjectionKind,
    /// When set, overrides the computed perspective matrix (Reflector oblique clip).
    pub projection_override: Option<[f32; 16]>,
}

impl PerspectiveCamera {
    /// `fov_deg` matches three.js's degree-based API.
    pub fn new(fov_deg: f32, aspect: f32, near: f32, far: f32) -> Self {
        Self {
            position: Vector3::new(0.0, 0.0, 5.0),
            target: Vector3::ZERO,
            up: Vector3::UP,
            fov: fov_deg.to_radians(),
            aspect,
            near,
            far,
            focus_distance: 0.0,
            aperture: 0.0,
            f_stop: 0.0,
            aperture_blades: 0,
            anamorphic_ratio: 1.0,
            shift_x: 0.0,
            shift_y: 0.0,
            sensor_fit: SensorFit::Auto,
            projection: ProjectionKind::Perspective,
            projection_override: None,
        }
    }

    pub fn look_at(&mut self, target: Vector3) -> &mut Self {
        self.target = target;
        self
    }
}

impl Camera for PerspectiveCamera {
    fn view_matrix(&self) -> Matrix4 {
        Matrix4::look_at(self.position, self.target, self.up)
    }
    fn projection_matrix(&self) -> Matrix4 {
        if let Some(m) = self.projection_override {
            return Matrix4 { elements: m };
        }
        // Panoramic kinds still expose a perspective matrix for the raster
        // path; specialised integrators read `projection` directly.
        if self.shift_x.abs() > 1e-8 || self.shift_y.abs() > 1e-8 {
            Matrix4::perspective_with_shift(
                self.fov,
                self.aspect,
                self.near,
                self.far,
                self.shift_x,
                self.shift_y,
            )
        } else {
            Matrix4::perspective(self.fov, self.aspect, self.near, self.far)
        }
    }
    fn position(&self) -> Vector3 {
        self.position
    }
    fn set_aspect(&mut self, aspect: f32) {
        self.aspect = aspect;
    }
    fn near_far(&self) -> (f32, f32) {
        (self.near, self.far)
    }
}
