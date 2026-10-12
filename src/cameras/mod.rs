//! Cameras: position/orientation in world space plus a projection matrix.

mod orthographic;
mod perspective;

pub use orthographic::OrthographicCamera;
pub use perspective::{PerspectiveCamera, ProjectionKind, SensorFit};

use crate::core::Layers;
use crate::math::{Matrix4, Vector2, Vector3};

/// The lens a camera looks through, for renderers that model one.
///
/// The rasteriser is a pinhole and ignores all of it; the path tracer traces
/// real rays through an aperture, so this is where depth of field, bokeh shape
/// and anamorphic squeeze come from.
///
/// Everything is in **world units**, which is the only form a ray tracer can
/// use. [`PerspectiveCamera::lens`] is what converts a photographic f-number
/// into one, and says what it assumes about scale to do so.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lens {
    /// Aperture radius. `0` is a pinhole — everything in focus, no bokeh.
    pub aperture: f32,
    /// Distance to the plane that stays sharp. `0` leaves it to the renderer,
    /// which focuses on what is in front of the camera.
    pub focus_distance: f32,
    /// Diaphragm blades. Below three the aperture is a circle, which is what a
    /// wide-open lens is; stopped down, a real iris is a polygon and so is the
    /// bokeh.
    pub blades: u32,
    /// Anamorphic squeeze. `1` is spherical; `2` is a 2x anamorphic, whose
    /// bokeh is twice as tall as it is wide.
    pub anamorphic: f32,
}

impl Default for Lens {
    /// A pinhole: no aperture, nothing to shape.
    fn default() -> Self {
        Self {
            aperture: 0.0,
            focus_distance: 0.0,
            blades: 0,
            anamorphic: 1.0,
        }
    }
}

impl Lens {
    /// Whether this lens actually defocuses anything.
    pub fn is_thin_lens(&self) -> bool {
        self.aperture > 0.0
    }

    /// A point on the aperture, from two uniform randoms in `[0, 1)`.
    ///
    /// Returns coordinates on the unit aperture — multiply by
    /// [`aperture`](Self::aperture) for world units. Circular below three
    /// blades, a regular polygon at or above, and squeezed horizontally by
    /// [`anamorphic`](Self::anamorphic) either way.
    ///
    /// Shared with the GPU path tracer, which carries the same arithmetic in
    /// WGSL; `the_two_lens_samplers_agree` holds them together.
    pub fn sample(&self, u1: f32, u2: f32) -> (f32, f32) {
        let (mut x, mut y) = if self.blades >= 3 {
            polygon_sample(self.blades, u1, u2)
        } else {
            concentric_disk(u1, u2)
        };
        // A 2x squeeze narrows the bokeh horizontally; it does not widen it,
        // which is why this divides rather than multiplying the other axis.
        if self.anamorphic > 0.0 && self.anamorphic != 1.0 {
            x /= self.anamorphic;
        }
        // Keep the sample inside the unit disk whatever the squeeze did, so
        // `aperture` stays a bound on how far a ray can start from the axis.
        let r = (x * x + y * y).sqrt();
        if r > 1.0 {
            x /= r;
            y /= r;
        }
        (x, y)
    }
}

/// A uniform point in the unit disk, by the concentric map (Shirley & Chiu).
///
/// Not polar `(sqrt(u1), 2*pi*u2)`: that is also uniform, but it maps the unit
/// square to the disk with a twist that clumps low-discrepancy samples along
/// spokes — and every sample here comes from a stratified sequence.
pub fn concentric_disk(u1: f32, u2: f32) -> (f32, f32) {
    let a = 2.0 * u1 - 1.0;
    let b = 2.0 * u2 - 1.0;
    if a == 0.0 && b == 0.0 {
        return (0.0, 0.0);
    }
    let (r, theta) = if a * a > b * b {
        (a, std::f32::consts::FRAC_PI_4 * (b / a))
    } else {
        (
            b,
            std::f32::consts::FRAC_PI_2 - std::f32::consts::FRAC_PI_4 * (a / b),
        )
    };
    (r * theta.cos(), r * theta.sin())
}

/// A uniform point in a regular `blades`-gon inscribed in the unit circle.
///
/// Pick a wedge, then a point in that wedge's triangle. The square root is
/// what makes it uniform by area rather than by parameter — without it the
/// samples bunch at the centre and a stopped-down aperture produces bokeh with
/// a bright core, which is the artefact this is avoiding.
pub fn polygon_sample(blades: u32, u1: f32, u2: f32) -> (f32, f32) {
    let n = blades.max(3) as f32;
    let wedge = std::f32::consts::TAU / n;
    // Reuse u1 for both the wedge and the position inside it: scaling by n and
    // taking the fraction is a standard trick that costs no extra dimension of
    // the sample sequence, which matters when it is a stratified one.
    let scaled = (u1 * n).min(n - f32::EPSILON);
    let index = scaled.floor();
    let mut a = scaled - index;
    let mut b = u2;
    // Fold the square onto the lower triangle.
    if a + b > 1.0 {
        a = 1.0 - a;
        b = 1.0 - b;
    }
    let (t0, t1) = (index * wedge, (index + 1.0) * wedge);
    let (c0, s0) = (t0.cos(), t0.sin());
    let (c1, s1) = (t1.cos(), t1.sin());
    (a * c0 + b * c1, a * s0 + b * s1)
}

/// Unified camera trait used by the Renderer.
pub trait Camera {
    fn view_matrix(&self) -> Matrix4;
    fn projection_matrix(&self) -> Matrix4;
    fn position(&self) -> Vector3;
    fn set_aspect(&mut self, aspect: f32);

    /// The lens this camera looks through. A pinhole unless overridden.
    fn lens(&self) -> Lens {
        Lens::default()
    }

    /// Point the camera at a world-space position.
    ///
    /// On the trait rather than on each camera because in three.js this comes
    /// from `Object3D` and every camera therefore has it — and because the one
    /// that did not have it was `OrthographicCamera`, which reads at the call
    /// site as the method not existing at all rather than as a camera that
    /// cannot be aimed.
    fn look_at(&mut self, target: Vector3);

    /// Clip-space near/far planes (for SSAO depth encoding in normal prepass).
    fn near_far(&self) -> (f32, f32) {
        (0.1, 1000.0)
    }

    /// Layer mask. Meshes are visible to this camera iff `camera.layers().test(&mesh.layers)`.
    fn layers(&self) -> Layers {
        Layers::default()
    }

    /// Project a world-space point into normalized device coordinates: `x` and
    /// `y` in `-1..1` with y **up**, `z` in `0..1` across the depth range.
    ///
    /// `None` when the point is at or behind the eye, where there is nothing to
    /// place. Both projections get that wrong in their own way if asked: a
    /// perspective divide by a negative `w` puts the point back in front,
    /// mirrored, and an orthographic camera has no divide at all so a point
    /// behind the viewer lands wherever it would be if you could see through
    /// your own head. Neither looks like an error at the call site, which is why
    /// this returns an `Option` rather than a `Vector3`.
    ///
    /// A point outside the frustum but still in front does project, to
    /// coordinates outside the cube. That is deliberate: it is what lets a
    /// caller clamp an off-screen marker to the edge of the view instead of
    /// losing it.
    ///
    /// The inverse is [`Vector3::unproject`](crate::math::Vector3::unproject).
    ///
    /// Projecting more than a handful of points wants
    /// [`projector`](Self::projector), which resolves the matrices once instead
    /// of per point.
    fn world_to_ndc(&self, point: Vector3) -> Option<Vector3> {
        self.projector(1, 1).ndc(point)
    }

    /// Project a world-space point to pixels from the **top-left** of a
    /// `width × height` view.
    ///
    /// The frame the overlay draws in and the one
    /// [`sketch::Frame::pointer`](crate::sketch::Frame::pointer) reports, so a
    /// label on an object and a hit test against the cursor are in the same
    /// coordinates:
    ///
    /// ```no_run
    /// # use threers::prelude::*;
    /// # use threers::sketch::Frame;
    /// # fn draw(f: &mut Frame, thing: Vector3) {
    /// if let Some(p) = f.camera.world_to_screen(thing, f.width, f.height) {
    ///     f.overlay.fill(Color::WHITE);
    ///     f.overlay.text(p.x + 8.0, p.y - 8.0, "here");
    /// }
    /// # }
    /// ```
    fn world_to_screen(&self, point: Vector3, width: u32, height: u32) -> Option<Vector2> {
        self.projector(width, height).screen(point)
    }

    /// This camera's view and projection resolved once, for projecting many
    /// points.
    ///
    /// [`world_to_screen`](Self::world_to_screen) rebuilds both matrices on
    /// every call — a `look_at` with its normalizations and cross products, and
    /// a projection — which is the right trade for one point and the wrong one
    /// for a thousand. Labelling a scene is the loop this exists for:
    ///
    /// ```no_run
    /// # use threers::prelude::*;
    /// # use threers::sketch::Frame;
    /// # fn draw(f: &mut Frame, things: &[Vector3]) {
    /// let project = f.camera.projector(f.width, f.height);
    /// for (i, thing) in things.iter().enumerate() {
    ///     if let Some(p) = project.screen(*thing) {
    ///         f.overlay.circle(p.x, p.y, 4.0);
    ///     }
    /// }
    /// # }
    /// ```
    fn projector(&self, width: u32, height: u32) -> ScreenProjector {
        let view = self.view_matrix();
        ScreenProjector {
            // Row 2 of the view matrix, which is all that is needed of it: the
            // eye-space z of a point, for the in-front test.
            eye_z: [
                view.elements[2],
                view.elements[6],
                view.elements[10],
                view.elements[14],
            ],
            view_projection: self.projection_matrix().multiply(&view),
            width: width as f32,
            height: height as f32,
        }
    }
}

/// A camera's view and projection, resolved once. From [`Camera::projector`].
#[derive(Debug, Clone)]
pub struct ScreenProjector {
    eye_z: [f32; 4],
    view_projection: Matrix4,
    width: f32,
    height: f32,
}

impl ScreenProjector {
    /// As [`Camera::world_to_ndc`], and with the same `None`.
    pub fn ndc(&self, point: Vector3) -> Option<Vector3> {
        // Eye space is right-handed and looks down -z, so in front is negative.
        // Tested on the eye-space z rather than on the clip-space `w` because
        // `w` is 1 for an orthographic projection and so says nothing, and
        // because this needs no assumption about which row of the projection
        // matrix carries it.
        let z = self.eye_z[0] * point.x
            + self.eye_z[1] * point.y
            + self.eye_z[2] * point.z
            + self.eye_z[3];
        if z >= 0.0 {
            return None;
        }
        Some(point.apply_matrix4(&self.view_projection))
    }

    /// As [`Camera::world_to_screen`], and with the same `None`.
    pub fn screen(&self, point: Vector3) -> Option<Vector2> {
        let ndc = self.ndc(point)?;
        Some(Vector2::new(
            (ndc.x * 0.5 + 0.5) * self.width,
            // NDC y is up and pixel y is down, which is the one place this is
            // easy to get backwards and hard to notice: a symmetric scene looks
            // right either way.
            (1.0 - (ndc.y * 0.5 + 0.5)) * self.height,
        ))
    }

    /// The combined view-projection, for handing to a shader or reusing.
    pub fn view_projection(&self) -> &Matrix4 {
        &self.view_projection
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera() -> PerspectiveCamera {
        let mut c = PerspectiveCamera::new(60.0, 2.0, 0.1, 100.0);
        c.position = Vector3::new(0.0, 0.0, 5.0);
        c.look_at(Vector3::ZERO);
        c
    }

    #[test]
    fn the_point_a_camera_looks_at_lands_in_the_middle() {
        let c = camera();
        let p = c
            .world_to_screen(Vector3::ZERO, 800, 400)
            .expect("in front");
        assert!(
            (p.x - 400.0).abs() < 0.01 && (p.y - 200.0).abs() < 0.01,
            "expected the centre of an 800x400 view, got {p:?}"
        );
    }

    /// Pixel y runs down while NDC y runs up, so something above the target has
    /// to come back with a *smaller* y than something below it.
    #[test]
    fn up_in_the_world_is_up_on_the_screen() {
        let c = camera();
        let up = c
            .world_to_screen(Vector3::new(0.0, 1.0, 0.0), 800, 400)
            .unwrap();
        let down = c
            .world_to_screen(Vector3::new(0.0, -1.0, 0.0), 800, 400)
            .unwrap();
        assert!(
            up.y < 200.0,
            "a point above the target should be high: {up:?}"
        );
        assert!(down.y > 200.0, "and one below should be low: {down:?}");
        assert!(
            ((up.y + down.y) * 0.5 - 200.0).abs() < 0.01,
            "symmetric about the middle: {up:?} {down:?}"
        );
        let right = c
            .world_to_screen(Vector3::new(1.0, 0.0, 0.0), 800, 400)
            .unwrap();
        assert!(right.x > 400.0, "+x should be to the right: {right:?}");
    }

    #[test]
    fn a_point_behind_the_eye_has_no_projection() {
        let c = camera();
        // The camera sits at z = 5 looking at the origin, so z = 6 is behind it.
        assert!(c
            .world_to_screen(Vector3::new(0.0, 0.0, 6.0), 800, 400)
            .is_none());
        assert!(
            c.world_to_ndc(Vector3::new(0.0, 0.0, 5.0)).is_none(),
            "at the eye"
        );
        // And just in front still works, which is what makes the test above a
        // statement about the eye rather than about the whole axis.
        assert!(c
            .world_to_screen(Vector3::new(0.0, 0.0, 4.0), 800, 400)
            .is_some());
    }

    /// The projector folds view and projection into one matrix, where the
    /// per-point method used to apply them in turn. Same answer, or the
    /// multiply is the wrong way round — which would still put the centre in
    /// the centre and so survive every other test here.
    #[test]
    fn the_projector_matches_applying_the_matrices_in_turn() {
        let c = camera();
        let (view, projection) = (c.view_matrix(), c.projection_matrix());
        let project = c.projector(800, 400);
        let mut rng = crate::math::Rng::new(0xCA1);
        for _ in 0..200 {
            let p = Vector3::new(
                rng.range(-8.0, 8.0),
                rng.range(-8.0, 8.0),
                rng.range(-8.0, 4.9),
            );
            let eye = p.apply_matrix4(&view);
            let step_by_step = (eye.z < 0.0).then(|| {
                let ndc = eye.apply_matrix4(&projection);
                Vector2::new(
                    (ndc.x * 0.5 + 0.5) * 800.0,
                    (1.0 - (ndc.y * 0.5 + 0.5)) * 400.0,
                )
            });
            match (project.screen(p), step_by_step) {
                (None, None) => {}
                (Some(a), Some(b)) => assert!(
                    (a.x - b.x).abs() < 1e-2 && (a.y - b.y).abs() < 1e-2,
                    "{p:?}: combined {a:?} vs stepwise {b:?}"
                ),
                (a, b) => panic!("{p:?}: disagreed on whether it projects at all: {a:?} {b:?}"),
            }
        }
    }

    /// And the convenience method is the projector, so there is one rule.
    #[test]
    fn the_per_point_method_and_the_projector_agree() {
        let c = camera();
        let project = c.projector(640, 480);
        for p in [
            Vector3::ZERO,
            Vector3::new(2.0, -1.0, 1.0),
            Vector3::new(0.0, 0.0, 9.0),
        ] {
            assert_eq!(
                c.world_to_screen(p, 640, 480).map(|v| (v.x, v.y)),
                project.screen(p).map(|v| (v.x, v.y)),
                "{p:?}"
            );
        }
    }

    /// Against the inverse that already existed, so the two cannot drift.
    #[test]
    fn it_round_trips_through_unproject() {
        let c = camera();
        let (view, projection) = (c.view_matrix(), c.projection_matrix());
        for p in [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.5, -0.5, 2.0),
            Vector3::new(-2.0, 1.0, -3.0),
        ] {
            let ndc = c.world_to_ndc(p).expect("in front");
            let back = ndc.unproject(&view, &projection);
            assert!(
                (back.x - p.x).abs() < 1e-3
                    && (back.y - p.y).abs() < 1e-3
                    && (back.z - p.z).abs() < 1e-3,
                "{p:?} projected to {ndc:?} and came back as {back:?}"
            );
        }
    }

    /// An orthographic camera has no perspective divide, so this is where a
    /// hand-rolled `w` would be wrong.
    #[test]
    fn it_works_for_an_orthographic_camera_too() {
        let mut c = OrthographicCamera::new(-2.0, 2.0, 1.0, -1.0, 0.1, 100.0);
        c.position = Vector3::new(0.0, 0.0, 5.0);
        c.target = Vector3::ZERO;
        let mid = c.world_to_screen(Vector3::ZERO, 400, 200).expect("in view");
        assert!(
            (mid.x - 200.0).abs() < 0.01 && (mid.y - 100.0).abs() < 0.01,
            "got {mid:?}"
        );
        // Half the half-width across should be a quarter of the way over.
        let off = c
            .world_to_screen(Vector3::new(1.0, 0.0, 0.0), 400, 200)
            .unwrap();
        assert!((off.x - 300.0).abs() < 0.01, "got {off:?}");

        // And behind it is still nowhere. An orthographic projection has no
        // perspective divide to go wrong, so this used to come back as the
        // centre of the screen — a marker for something behind the viewer,
        // sitting squarely on top of the thing it is looking at.
        assert!(
            c.world_to_screen(Vector3::new(0.0, 0.0, 6.0), 400, 200)
                .is_none(),
            "a point behind an orthographic camera is not placeable"
        );
    }

    /// `look_at` is on the trait, so it works through a `&mut dyn Camera` —
    /// and both cameras actually move when it is called. `OrthographicCamera`
    /// had no `look_at` at all, which at the call site reads as the method not
    /// existing rather than as a camera that cannot be aimed.
    #[test]
    fn every_camera_can_be_aimed_through_the_trait() {
        let mut perspective = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
        perspective.position = Vector3::new(0.0, 0.0, 5.0);
        let mut orthographic = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 100.0);
        orthographic.position = Vector3::new(0.0, 0.0, 5.0);

        let aim = Vector3::new(2.0, 1.0, 0.0);
        let cameras: [&mut dyn Camera; 2] = [&mut perspective, &mut orthographic];
        let before: Vec<_> = cameras.iter().map(|c| c.view_matrix().elements).collect();
        for (camera, was) in cameras.into_iter().zip(before) {
            camera.look_at(aim);
            assert_ne!(
                camera.view_matrix().elements,
                was,
                "the view matrix did not follow look_at"
            );
            // Aiming somewhere is aiming *there*: the camera's forward axis
            // should point from its position towards the target.
            let forward = (aim - camera.position()).normalize();
            let view = camera.view_matrix().elements;
            // Column-major: the third column's first three entries are the
            // camera's own -Z in world space.
            let cam_forward = Vector3::new(-view[2], -view[6], -view[10]).normalize();
            let dot =
                forward.x * cam_forward.x + forward.y * cam_forward.y + forward.z * cam_forward.z;
            assert!(
                dot > 0.999,
                "camera is aimed {dot} of the way at the target"
            );
        }

        // The inherent methods still chain, so existing callers are untouched.
        let aimed = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0)
            .look_at(Vector3::ONE)
            .position;
        assert_eq!(aimed, Vector3::new(0.0, 0.0, 5.0));
        let mut ortho = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 100.0);
        assert_eq!(ortho.look_at(Vector3::ONE).target, Vector3::ONE);
    }
}

#[cfg(test)]
mod lens_tests {
    use super::*;

    /// An f-number is a ratio on the camera and a ray tracer needs a radius in
    /// the scene, so the conversion has to assume a scale. This pins the one
    /// it assumes — a 50 mm lens at f/2.8 is 8.9 mm of aperture radius in a
    /// scene measured in metres — because the assumption is the whole content
    /// of the conversion and a silent change to it would be a silent change to
    /// every defocused render.
    #[test]
    fn an_f_stop_becomes_an_aperture_in_world_units() {
        let mut c = PerspectiveCamera::new(0.0, 1.0, 0.1, 100.0);
        // 50 mm on a 35 mm gauge: 2 * atan(17.5 / 50).
        c.fov = 2.0 * (17.5f32 / 50.0).atan();
        assert!(
            (c.focal_length_mm() - 50.0).abs() < 0.01,
            "focal length {} mm",
            c.focal_length_mm()
        );

        c.f_stop = 2.8;
        let lens = c.lens();
        let expected = 0.050 / (2.0 * 2.8);
        assert!(
            (lens.aperture - expected).abs() < 1e-6,
            "f/2.8 gave {} , wanted {expected}",
            lens.aperture
        );
        // Halving the f-number doubles the aperture, as it must.
        c.f_stop = 1.4;
        assert!((c.lens().aperture - expected * 2.0).abs() < 1e-6);

        // An explicit radius wins: it is in world units and assumes nothing.
        c.aperture = 0.25;
        assert_eq!(c.lens().aperture, 0.25);

        // And neither set is a pinhole, which is what the rasteriser wants.
        let plain = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
        assert_eq!(plain.lens().aperture, 0.0);
        assert!(!plain.lens().is_thin_lens());
    }

    /// Blades shape the bokeh, and the shape has to be the polygon asked for —
    /// not a circle, not a polygon with a bright core.
    #[test]
    fn blades_make_the_aperture_a_polygon() {
        let hex = Lens {
            aperture: 1.0,
            blades: 6,
            ..Lens::default()
        };
        let round = Lens {
            aperture: 1.0,
            ..Lens::default()
        };
        // Stratified-ish coverage of the unit square.
        let samples = |lens: &Lens| -> Vec<(f32, f32)> {
            let n = 160;
            (0..n * n)
                .map(|i| {
                    let (a, b) = ((i % n) as f32, (i / n) as f32);
                    lens.sample((a + 0.5) / n as f32, (b + 0.5) / n as f32)
                })
                .collect()
        };
        let h = samples(&hex);
        let r = samples(&round);

        // Everything is inside the unit circle either way.
        for (x, y) in h.iter().chain(r.iter()) {
            assert!(x * x + y * y <= 1.0 + 1e-4, "({x}, {y}) is outside");
        }

        // The hexagon's radius varies with angle and the circle's does not.
        // Six-fold: the maximum reach at a vertex is 1 and at an edge midpoint
        // is cos(pi/6) = 0.866.
        let max_at = |pts: &[(f32, f32)], angle: f32| -> f32 {
            pts.iter()
                .filter(|(x, y)| {
                    let a = y.atan2(*x);
                    let d = (a - angle).abs();
                    d.min(std::f32::consts::TAU - d) < 0.05
                })
                .map(|(x, y)| (x * x + y * y).sqrt())
                .fold(0.0f32, f32::max)
        };
        let vertex = max_at(&h, 0.0);
        let edge = max_at(&h, std::f32::consts::PI / 6.0);
        assert!(vertex > 0.97, "hexagon vertex reach {vertex}");
        assert!(
            (edge - 0.866).abs() < 0.03,
            "hexagon edge-midpoint reach {edge}, wanted 0.866"
        );
        // The circle reaches the same distance in both directions.
        assert!(
            (max_at(&r, 0.0) - max_at(&r, std::f32::consts::PI / 6.0)).abs() < 0.03,
            "the round aperture is not round"
        );

        // Uniform by area, not by parameter: half the samples should fall
        // outside radius 1/sqrt(2) of the polygon's own extent.
        let inner = h
            .iter()
            .filter(|(x, y)| (x * x + y * y).sqrt() < 0.866 / std::f32::consts::SQRT_2)
            .count() as f32
            / h.len() as f32;
        assert!(
            (0.35..0.55).contains(&inner),
            "{inner:.2} of the samples are in the inner half-area — the \
             aperture has a bright core"
        );
    }

    /// An anamorphic lens squeezes its bokeh horizontally.
    #[test]
    fn an_anamorphic_lens_has_oval_bokeh() {
        let lens = Lens {
            aperture: 1.0,
            anamorphic: 2.0,
            ..Lens::default()
        };
        let mut wide: f32 = 0.0;
        let mut tall: f32 = 0.0;
        let n = 128;
        for i in 0..n * n {
            let (a, b) = ((i % n) as f32, (i / n) as f32);
            let (x, y) = lens.sample((a + 0.5) / n as f32, (b + 0.5) / n as f32);
            wide = wide.max(x.abs());
            tall = tall.max(y.abs());
        }
        assert!(
            (tall / wide - 2.0).abs() < 0.1,
            "a 2x squeeze gave {:.2}:1 bokeh",
            tall / wide
        );
        // And 1.0 leaves it round, so the default costs nothing.
        let round = Lens {
            aperture: 1.0,
            ..Lens::default()
        };
        let (x, y) = round.sample(0.9, 0.1);
        let (x2, y2) = Lens {
            anamorphic: 1.0,
            ..round
        }
        .sample(0.9, 0.1);
        assert_eq!((x, y), (x2, y2));
    }

    /// The WGSL sampler is a transcription of the Rust one, and a
    /// transcription can drift. This evaluates the shader's arithmetic here —
    /// same constants, same branches — and holds the two together.
    #[test]
    fn the_two_lens_samplers_agree() {
        // `lens_sample` from `pathtrace.wgsl`, in Rust.
        let wgsl = |u1: f32, u2: f32, blades: f32, anamorphic: f32| -> (f32, f32) {
            let (mut x, mut y) = if blades >= 3.0 {
                let n = blades.max(3.0);
                // The WGSL spells tau out as a literal; this is that literal,
                // named, so clippy does not read it as an approximation.
                let wedge = std::f32::consts::TAU / n;
                let scaled = (u1 * n).min(n - 1e-6);
                let index = scaled.floor();
                let mut a = scaled - index;
                let mut b = u2;
                if a + b > 1.0 {
                    a = 1.0 - a;
                    b = 1.0 - b;
                }
                let (t0, t1) = (index * wedge, (index + 1.0) * wedge);
                (a * t0.cos() + b * t1.cos(), a * t0.sin() + b * t1.sin())
            } else {
                concentric_disk(u1, u2)
            };
            if anamorphic > 0.0 && anamorphic != 1.0 {
                x /= anamorphic;
            }
            let r = (x * x + y * y).sqrt();
            if r > 1.0 {
                x /= r;
                y /= r;
            }
            (x, y)
        };
        for &(blades, anamorphic) in &[(0u32, 1.0f32), (5, 1.0), (6, 2.0), (8, 0.5)] {
            let lens = Lens {
                aperture: 1.0,
                blades,
                anamorphic,
                ..Lens::default()
            };
            for i in 0..64 {
                for j in 0..64 {
                    let (u1, u2) = ((i as f32 + 0.5) / 64.0, (j as f32 + 0.5) / 64.0);
                    let a = lens.sample(u1, u2);
                    let b = wgsl(u1, u2, blades as f32, anamorphic);
                    assert!(
                        (a.0 - b.0).abs() < 1e-5 && (a.1 - b.1).abs() < 1e-5,
                        "blades {blades}, squeeze {anamorphic}, u ({u1}, {u2}): \
                         Rust {a:?} vs WGSL {b:?}"
                    );
                }
            }
        }
    }
}
