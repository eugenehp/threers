//! Fixtures for shadow work: scenes whose right answer comes from geometry.
//!
//! Every shadow test before this one asked whether two renderers agreed. That
//! is a useful question and it caught real bugs, but it has a failure mode this
//! project walked straight into: when both backends are wrong in the same
//! place, agreement is perfect and means nothing. Point shadows agreed at
//! 0.9996 while neither of them cast anything.
//!
//! So the reference here is not the other renderer. It is arithmetic. A point
//! light at `L`, an occluder at `O` and a floor at a known depth put the shadow
//! somewhere exactly computable — the ray `L → O` extended to the floor — and
//! the same projection the camera uses says where that lands on screen. What
//! comes back is an error in **pixels**, which is a number you can argue with.
//!
//! Three things are measured, and they fail differently:
//!
//! * **Placement.** Where the darkening actually is, against where the geometry
//!   says it must be. A mirrored cube face, a transposed matrix or a sign slip
//!   all move the shadow somewhere specific, and the offset names which.
//! * **Strength.** How much of the light is blocked at the darkest point. A
//!   shadow in the right place but at a twentieth of the depth is a comparison
//!   that nearly always passes — the failure that hid behind "it works a bit".
//! * **Extent.** How much of the frame is darkened, which catches the opposite
//!   failure: a bias so large that everything is in shadow.
//!
//! The fixture is deliberately reusable across light types. A point light and a
//! spot light in the same place throw the same shadow, so any of the three can
//! be held to the same predicted position, and a discrepancy is about the light
//! rather than about the scene.

#![cfg(any(target_os = "macos", target_os = "ios"))]

use threers::cameras::{Camera, PerspectiveCamera};
use threers::core::{Mesh, Object3D};
use threers::geometries::{BoxGeometry, PlaneGeometry};
use threers::lights::{DirectionalLight, Light, PointLight, SpotLight};
use threers::materials::StandardMaterial;
use threers::math::{Color, Vector3};
use threers::scene::Scene;

/// Frame size. Large enough that a shadow spans tens of pixels, small enough
/// that the path tracer could join in later.
const N: u32 = 192;

/// Where the light sits. Off both axes, so a mirror in either one moves the
/// shadow somewhere distinguishable — a light on an axis of symmetry hides
/// exactly the bug this fixture exists to find.
const LIGHT: Vector3 = Vector3::new(16.0, 11.0, 30.0);
/// The occluder's centre, and the floor's plane.
const OCCLUDER: Vector3 = Vector3::new(0.0, 0.0, 8.0);
const FLOOR_Z: f32 = -8.0;
const OCCLUDER_SIZE: f32 = 10.0;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    Directional,
    Spot,
    Point,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Directional => "directional",
            Kind::Spot => "spot",
            Kind::Point => "point",
        }
    }
}

fn camera() -> PerspectiveCamera {
    let fov: f32 = 40.0;
    let d = 120.0;
    let mut c = PerspectiveCamera::new(fov, 1.0, 0.5, 1000.0);
    c.position = Vector3::new(0.0, 0.0, d);
    c.look_at(Vector3::ZERO);
    c
}

/// Where the occluder's shadow centre lands on the floor, from geometry alone.
///
/// A directional light has no position, so its rays are parallel and the offset
/// is the direction scaled to the floor. The others cast from a point, so the
/// shadow is the ray through the occluder extended — and grows with distance,
/// which is why a point light's shadow is bigger than a directional one's over
/// the same occluder.
fn predicted_shadow_centre(kind: Kind) -> Vector3 {
    match kind {
        Kind::Directional => {
            // Rays parallel to LIGHT -> origin.
            let dir = (Vector3::ZERO - LIGHT).normalize();
            let t = (FLOOR_Z - OCCLUDER.z) / dir.z;
            OCCLUDER + dir * t
        }
        Kind::Spot | Kind::Point => {
            let dir = OCCLUDER - LIGHT;
            let t = (FLOOR_Z - LIGHT.z) / dir.z;
            LIGHT + dir * t
        }
    }
}

/// Project a world point to pixel coordinates, the way the camera does.
fn to_pixels(p: Vector3, cam: &PerspectiveCamera) -> (f64, f64) {
    let vp = cam.projection_matrix().multiply(&cam.view_matrix());
    let e = vp.elements;
    // Column-major, so column i is elements[4i..4i+4].
    let x = (e[0] * p.x + e[4] * p.y + e[8] * p.z + e[12]) as f64;
    let y = (e[1] * p.x + e[5] * p.y + e[9] * p.z + e[13]) as f64;
    let w = (e[3] * p.x + e[7] * p.y + e[11] * p.z + e[15]) as f64;
    let (ndc_x, ndc_y) = (x / w, y / w);
    (
        (ndc_x * 0.5 + 0.5) * N as f64,
        (0.5 - ndc_y * 0.5) * N as f64,
    )
}

fn scene(kind: Kind, occluder: bool) -> Scene {
    let mut s = Scene::new();
    s.background = Color::from_hex(0x000000);
    let light = match kind {
        Kind::Directional => {
            let mut l = DirectionalLight::new(Color::WHITE, 2.2);
            l.direction = (Vector3::ZERO - LIGHT).normalize();
            l.cast_shadow = true;
            l.shadow.map_size = 2048;
            l.shadow.camera_size = 40.0;
            l.shadow.camera_near = 1.0;
            l.shadow.camera_far = 200.0;
            l.shadow.normal_bias = 0.05;
            Light::Directional(l)
        }
        Kind::Spot => {
            let mut l = SpotLight::new(Color::WHITE, 2400.0);
            l.angle = 0.75;
            l.penumbra = 0.1;
            l.distance = 200.0;
            l.cast_shadow = true;
            l.shadow.map_size = 2048;
            l.shadow.camera_near = 1.0;
            l.shadow.camera_far = 200.0;
            l.direction = (Vector3::ZERO - LIGHT).normalize();
            Light::Spot(l)
        }
        Kind::Point => {
            let mut l = PointLight::new(Color::WHITE, 2400.0);
            l.distance = 200.0;
            l.cast_shadow = true;
            Light::Point(l)
        }
    };
    let mut obj = Object3D::light(light);
    obj.position = LIGHT;
    s.add(obj);

    let mut floor_mat = StandardMaterial::new(Color::WHITE);
    floor_mat.metalness = 0.0;
    floor_mat.roughness = 1.0;
    let mut floor = Object3D::mesh(Mesh::new(PlaneGeometry::new(200.0, 200.0), floor_mat.into()));
    floor.position = Vector3::new(0.0, 0.0, FLOOR_Z);
    floor.receive_shadow = true;
    s.add(floor);

    if occluder {
        let mut m = StandardMaterial::new(Color::WHITE);
        m.metalness = 0.0;
        m.roughness = 1.0;
        let mut b = Object3D::mesh(Mesh::new(
            BoxGeometry::new(OCCLUDER_SIZE, OCCLUDER_SIZE, 1.0),
            m.into(),
        ));
        b.position = OCCLUDER;
        b.cast_shadow = true;
        s.add(b);
    }
    s
}

/// What a rendered shadow actually did.
#[derive(Debug)]
struct Measured {
    /// Where the darkening is, in pixels.
    centre: (f64, f64),
    /// How far that is from where geometry says it should be.
    offset_px: f64,
    /// The largest fraction of light any pixel lost.
    strength: f64,
    /// What fraction of the frame lost more than a tenth of its light.
    extent: f64,
}

fn to_linear(b: u8) -> f64 {
    let v = b as f64 / 255.0;
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Compare a lit frame with a shadowed one and locate the shadow.
///
/// The centroid is weighted by how much each pixel *lost*, not by its
/// brightness — a shadow is a difference, and weighting by the difference is
/// what makes the answer independent of how bright the floor happens to be.
fn measure(lit: &[u8], shadowed: &[u8], predicted: (f64, f64)) -> Measured {
    let mut sum_w = 0.0;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut peak: f64 = 0.0;
    let mut darkened = 0usize;
    let mut total = 0usize;
    for y in 0..N as usize {
        for x in 0..N as usize {
            let i = (y * N as usize + x) * 4;
            let a = to_linear(lit[i]) + to_linear(lit[i + 1]) + to_linear(lit[i + 2]);
            let b = to_linear(shadowed[i]) + to_linear(shadowed[i + 1]) + to_linear(shadowed[i + 2]);
            if a <= 1e-4 {
                continue;
            }
            total += 1;
            let lost = ((a - b) / a).max(0.0);
            if lost > 0.1 {
                darkened += 1;
            }
            peak = peak.max(lost);
            sum_w += lost;
            sum_x += lost * x as f64;
            sum_y += lost * y as f64;
        }
    }
    let centre = if sum_w > 1e-6 {
        (sum_x / sum_w, sum_y / sum_w)
    } else {
        (f64::NAN, f64::NAN)
    };
    let offset_px = if centre.0.is_finite() {
        ((centre.0 - predicted.0).powi(2) + (centre.1 - predicted.1).powi(2)).sqrt()
    } else {
        f64::INFINITY
    };
    Measured {
        centre,
        offset_px,
        strength: peak,
        extent: if total > 0 {
            darkened as f64 / total as f64
        } else {
            0.0
        },
    }
}

/// Render one kind through the wgpu renderer and measure it.
fn probe_wgpu(kind: Kind) -> Option<Measured> {
    let mut r = threers::HeadlessRenderer::builder().size(N, N).build().ok()?;
    let cam = camera();
    let lit = r.render_to_rgba(&mut scene(kind, false), &cam);
    let shadowed = r.render_to_rgba(&mut scene(kind, true), &cam);
    Some(measure(&lit, &shadowed, to_pixels(predicted_shadow_centre(kind), &cam)))
}

/// The same through the Metal backend.
#[cfg(feature = "metal")]
fn probe_metal(kind: Kind) -> Option<Measured> {
    use threers::metal::{MetalError, MetalHeadlessRenderer};
    let mut r = match MetalHeadlessRenderer::builder().size(N, N).build() {
        Ok(r) => r,
        Err(MetalError::NoDevice) => return None,
        Err(e) => panic!("metal renderer: {e:?}"),
    };
    let cam = camera();
    let lit = r.render_to_rgba(&mut scene(kind, false), &cam).expect("metal");
    let shadowed = r.render_to_rgba(&mut scene(kind, true), &cam).expect("metal");
    Some(measure(&lit, &shadowed, to_pixels(predicted_shadow_centre(kind), &cam)))
}

/// The fixture reporting on itself: every light type, both backends, against
/// geometry.
///
/// Printed as a table rather than asserted piecemeal, because the useful thing
/// when a shadow is wrong is *how* it is wrong, and six numbers next to each
/// other say that faster than six failures would.
#[test]
fn shadow_placement_against_geometry() {
    let cam = camera();
    eprintln!(
        "{:<12} {:<8} {:>18} {:>18} {:>9} {:>9} {:>8}",
        "light", "backend", "predicted px", "measured px", "offset", "strength", "extent"
    );
    for kind in [Kind::Directional, Kind::Spot, Kind::Point] {
        let predicted = to_pixels(predicted_shadow_centre(kind), &cam);
        let rows: Vec<(&str, Option<Measured>)> = vec![
            ("wgpu", probe_wgpu(kind)),
            #[cfg(feature = "metal")]
            ("metal", probe_metal(kind)),
        ];
        for (backend, m) in rows {
            match m {
                Some(m) => eprintln!(
                    "{:<12} {:<8} {:>8.1},{:>8.1} {:>8.1},{:>8.1} {:>9.1} {:>9.3} {:>8.3}",
                    kind.name(),
                    backend,
                    predicted.0,
                    predicted.1,
                    m.centre.0,
                    m.centre.1,
                    m.offset_px,
                    m.strength,
                    m.extent
                ),
                None => eprintln!("{:<12} {:<8}  skipped (no device)", kind.name(), backend),
            }
        }
    }
}

/// And the same thing as assertions, so it guards rather than reports.
///
/// The budget is six pixels against a measured 1.5 to 2.2. That headroom is
/// deliberate: the centroid of a penumbra is not the centre of the geometric
/// shadow to the last pixel, and a filter width or a map resolution can move it
/// a little without anything being wrong. What it will not absorb is a mirrored
/// face or a transposed matrix — every wrong variant of the cube flip measured
/// 79, 107 or 133 pixels out.
#[test]
fn every_light_casts_where_the_geometry_says() {
    let cam = camera();
    for kind in [Kind::Directional, Kind::Spot, Kind::Point] {
        let predicted = to_pixels(predicted_shadow_centre(kind), &cam);
        let mut probes: Vec<(&str, Measured)> = Vec::new();
        if let Some(m) = probe_wgpu(kind) {
            probes.push(("wgpu", m));
        }
        #[cfg(feature = "metal")]
        if let Some(m) = probe_metal(kind) {
            probes.push(("metal", m));
        }
        if probes.is_empty() {
            eprintln!("skipping: no device");
            return;
        }
        for (backend, m) in &probes {
            assert!(
                m.offset_px < 6.0,
                "the {} {backend} shadow lands {:.1} px from where the geometry \
                 puts it — measured at ({:.1}, {:.1}), predicted ({:.1}, {:.1})",
                kind.name(),
                m.offset_px,
                m.centre.0,
                m.centre.1,
                predicted.0,
                predicted.1
            );
            assert!(
                m.strength > 0.5,
                "the {} {backend} shadow blocks only {:.3} of the light at its \
                 darkest — in the right place and barely there",
                kind.name(),
                m.strength
            );
            assert!(
                m.extent < 0.30,
                "the {} {backend} shadow covers {:.3} of the lit frame — that \
                 is a bias or a matrix problem rather than a shadow",
                kind.name(),
                m.extent
            );
        }
        // And the two backends must agree with each other, which is a tighter
        // bound than either's agreement with the geometry: they share the
        // prediction's error and not their own.
        if probes.len() == 2 {
            let d = ((probes[0].1.centre.0 - probes[1].1.centre.0).powi(2)
                + (probes[0].1.centre.1 - probes[1].1.centre.1).powi(2))
            .sqrt();
            assert!(
                d < 2.0,
                "the {} shadow lands {d:.1} px apart on the two backends",
                kind.name()
            );
        }
    }
}
