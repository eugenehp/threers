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
    /// F-stop — the photographic way to say the same thing as
    /// [`aperture`](Self::aperture), which takes precedence when both are set.
    /// `0` = unused. See [`PerspectiveCamera::lens`] for the conversion and
    /// what it assumes about scale.
    pub f_stop: f32,
    /// Diaphragm blades, shaping the bokeh. Below three the aperture is a
    /// circle; at or above it is a regular polygon, as a stopped-down iris is.
    pub aperture_blades: u32,
    /// Anamorphic squeeze (`1` = spherical). `2` gives bokeh twice as tall as
    /// it is wide, as a 2x anamorphic lens does.
    pub anamorphic_ratio: f32,
    /// Sensor width in millimetres, three.js's `filmGauge`. Only the f-stop
    /// conversion reads it; the projection matrix is driven by `fov`.
    pub film_gauge: f32,
    /// Lens shift as a fraction of the sensor (±0.5 ≈ half-frame).
    pub shift_x: f32,
    pub shift_y: f32,
    pub sensor_fit: SensorFit,
    pub projection: ProjectionKind,
    /// When set, overrides the computed perspective matrix (Reflector oblique clip).
    pub projection_override: Option<[f32; 16]>,
    /// Channel mask — meshes are drawn iff this intersects `Object3D::layers`.
    pub layers: crate::core::Layers,
}

impl PerspectiveCamera {
    /// `fov_deg` matches three.js's degree-based API.
    /// The vertical field of view in **degrees**, which is the unit
    /// [`new`](Self::new) takes and the unit three.js uses.
    ///
    /// [`fov`](Self::fov) itself is radians. Both spellings exist because the
    /// mismatch is a trap worth closing: reading `camera.fov` and calling
    /// `.to_radians()` on it — which looks right, and which `new` makes look
    /// right — scales it by 57 and fails silently, because almost everything
    /// downstream still produces a picture.
    pub fn fov_degrees(&self) -> f32 {
        self.fov.to_degrees()
    }

    /// Set the vertical field of view in degrees.
    pub fn set_fov_degrees(&mut self, degrees: f32) {
        self.fov = degrees.to_radians();
    }

    /// How many pixels one world unit covers at `distance` from the camera,
    /// given a viewport `height` in pixels.
    ///
    /// The number anything sub-pixel needs: a hair, a thread, a spark, a point
    /// sprite — anything that has to be widened to a pixel and faded rather
    /// than left to alias. Deriving it by hand means getting the fov units
    /// right, which is the trap above.
    pub fn pixels_per_unit(&self, distance: f32, height: f32) -> f32 {
        height * 0.5 / (self.fov * 0.5).tan() / distance.max(1e-6)
    }

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
            film_gauge: 35.0,
            shift_x: 0.0,
            shift_y: 0.0,
            sensor_fit: SensorFit::Auto,
            projection: ProjectionKind::Perspective,
            projection_override: None,
            layers: crate::core::Layers::default(),
        }
    }

    pub fn look_at(&mut self, target: Vector3) -> &mut Self {
        self.target = target;
        self
    }

    /// Focal length in millimetres, from the field of view and the sensor —
    /// three.js's `getFocalLength()`.
    pub fn focal_length_mm(&self) -> f32 {
        0.5 * self.film_gauge / (self.fov * 0.5).tan()
    }

    /// This camera's lens, with [`f_stop`](Self::f_stop) resolved into an
    /// aperture radius.
    ///
    /// # The one assumption
    ///
    /// An f-number is a ratio of two lengths *on the camera* — focal length
    /// over aperture diameter — and a ray tracer needs the aperture as a
    /// radius **in the scene**. Nothing in a scene graph says how large a
    /// millimetre is, so the conversion has to assume, and this assumes what
    /// Blender assumes: **one world unit is one metre**.
    ///
    /// So a 50 mm lens at f/2.8 gives `0.050 / (2 x 2.8)` = 8.9 mm of aperture
    /// radius, and defocuses a scene built in metres the way the real lens
    /// would. In a scene built at some other scale the blur will be wrong by
    /// that scale, which is why [`aperture`](Self::aperture) exists and why it
    /// wins when both are set: it is in world units and assumes nothing.
    pub fn lens(&self) -> super::Lens {
        let aperture = if self.aperture > 0.0 {
            self.aperture
        } else if self.f_stop > 0.0 {
            // Radius, not diameter, and metres, not millimetres.
            (self.focal_length_mm() / 1000.0) / (2.0 * self.f_stop)
        } else {
            0.0
        };
        super::Lens {
            aperture,
            focus_distance: self.focus_distance.max(0.0),
            blades: self.aperture_blades,
            anamorphic: if self.anamorphic_ratio > 0.0 {
                self.anamorphic_ratio
            } else {
                1.0
            },
        }
    }
}

impl Camera for PerspectiveCamera {
    fn view_matrix(&self) -> Matrix4 {
        Matrix4::look_at(self.position, self.target, self.up)
    }
    fn look_at(&mut self, target: Vector3) {
        self.target = target;
    }
    fn lens(&self) -> super::Lens {
        PerspectiveCamera::lens(self)
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
    fn layers(&self) -> crate::core::Layers {
        self.layers
    }
}
