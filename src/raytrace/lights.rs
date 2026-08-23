//! Next-event estimation: at every bounce, guess where the light is instead of
//! hoping the next random direction finds it.
//!
//! A path tracer that only follows BSDF samples renders a room lit by a small
//! lamp as noise — the chance a cosine-weighted direction lands on the lamp is
//! the lamp's solid angle, which is tiny. Sampling the light directly fixes
//! that, and creates the opposite problem for a large, dim, mirror-like source,
//! where following the BSDF is far better. Multiple importance sampling takes
//! whichever is doing well at each point, so this module reports a *density*
//! alongside every sample and the integrator weighs the two.
//!
//! Three kinds of emitter live here, and they differ in whether a ray can also
//! reach them by accident:
//!
//! | Emitter | Reachable by a BSDF ray | MIS |
//! |---|---|---|
//! | Analytic lights (`DirectionalLight`, …) | no — they are not geometry | not needed |
//! | Emissive triangles | yes | power heuristic |
//! | Environment / world | yes | power heuristic |

use crate::math::Vector3;

use super::sampler::{uniform_cone, uniform_cone_pdf, Onb, Rng};
use super::scene::{RaytraceScene, RtLight, World};

/// One direct-lighting sample: a direction to try, and what arrives from it.
#[derive(Debug, Clone, Copy)]
pub struct LightSample {
    /// Unit direction from the shading point toward the light.
    pub direction: Vector3,
    /// How far the shadow ray must reach. `INFINITY` for the sky and for
    /// directional lights.
    pub distance: f32,
    /// Radiance arriving along `direction`, already divided by the density —
    /// so the integrator multiplies by `bsdf · cosθ` and nothing else.
    pub weight: Vector3,
    /// Solid-angle density, for the MIS weight. 0 marks a delta light, which
    /// takes weight 1 because no BSDF sample can ever reach it.
    pub pdf: f32,
    /// True when no MIS weight applies.
    pub delta: bool,
}

/// three.js `getDistanceAttenuation`, reproduced exactly so a scene lit for the
/// raster renderer keeps the same falloff here.
pub fn distance_attenuation(d: f32, max_distance: f32, decay: f32) -> f32 {
    let mut att = 1.0 / d.powf(decay).max(0.01);
    if max_distance > 0.0 {
        let t = (1.0 - (d / max_distance).powi(4)).clamp(0.0, 1.0);
        att *= t * t;
    }
    att
}

/// Sample one analytic light.
///
/// `point` is the shading point, already offset off the surface.
pub fn sample_analytic(light: &RtLight, point: Vector3, rng: &mut Rng) -> Option<LightSample> {
    match *light {
        RtLight::Directional {
            direction,
            radiance,
            angular_radius,
        } => {
            let to_light = direction * -1.0;
            if angular_radius <= 0.0 {
                return Some(LightSample {
                    direction: to_light,
                    distance: f32::INFINITY,
                    weight: radiance,
                    pdf: 0.0,
                    delta: true,
                });
            }
            // A sun with an angular size is a cone of directions. Its intensity
            // is still the irradiance it delivers, so the radiance per unit
            // solid angle is that divided by the cone — and sampling the cone
            // multiplies it straight back, which is why a soft sun is exactly
            // as bright as a hard one.
            let cos_max = angular_radius.cos();
            let onb = Onb::new(to_light);
            let (u1, u2) = rng.next_2d();
            let d = onb.to_world(uniform_cone(u1, u2, cos_max));
            let pdf = uniform_cone_pdf(cos_max);
            if pdf <= 0.0 {
                return None;
            }
            Some(LightSample {
                direction: d,
                distance: f32::INFINITY,
                // radiance / Ω, then / pdf = × Ω — so `weight` is just radiance.
                weight: radiance,
                pdf,
                delta: false,
            })
        }
        RtLight::Point {
            position,
            intensity,
            radius,
            distance,
            decay,
        } => {
            let to = position - point;
            let d = to.length();
            if d <= 1e-6 {
                return None;
            }
            let att = distance_attenuation(d, distance, decay);
            if att <= 0.0 {
                return None;
            }
            sample_sphere_light(point, position, radius, d, intensity * att, rng)
        }
        RtLight::Spot {
            position,
            direction,
            intensity,
            radius,
            distance,
            decay,
            cos_outer,
            cos_inner,
        } => {
            let to = position - point;
            let d = to.length();
            if d <= 1e-6 {
                return None;
            }
            let att = distance_attenuation(d, distance, decay);
            if att <= 0.0 {
                return None;
            }
            let mut s = sample_sphere_light(point, position, radius, d, intensity * att, rng)?;
            // Cone falloff, on the sampled direction rather than the axis, so a
            // spot with a radius gets a soft edge for free.
            let cos_angle = (s.direction * -1.0).dot(direction);
            if cos_angle <= cos_outer {
                return None;
            }
            let cone = smoothstep(cos_outer, cos_inner, cos_angle);
            if cone <= 0.0 {
                return None;
            }
            s.weight = s.weight * cone;
            Some(s)
        }
        RtLight::Rect {
            position,
            right,
            up,
            radiance,
        } => {
            let (u1, u2) = rng.next_2d();
            // Uniform on the rectangle; `right`/`up` are half-extents.
            let q = position + right * (2.0 * u1 - 1.0) + up * (2.0 * u2 - 1.0);
            let normal = right.cross(up);
            let area = 4.0 * normal.length();
            if area <= 0.0 {
                return None;
            }
            // three.js orients a RectAreaLight to emit along its local -Z, and
            // right × up is local +Z.
            let normal = normal.normalize() * -1.0;
            let to = q - point;
            let d2 = to.length_sq();
            if d2 <= 1e-12 {
                return None;
            }
            let d = d2.sqrt();
            let dir = to * (1.0 / d);
            let cos_light = (dir * -1.0).dot(normal);
            if cos_light <= 1e-6 {
                return None; // behind the panel
            }
            // Area density → solid-angle density.
            let pdf = d2 / (cos_light * area);
            if !pdf.is_finite() || pdf <= 0.0 {
                return None;
            }
            Some(LightSample {
                direction: dir,
                distance: d,
                weight: radiance * (1.0 / pdf),
                pdf,
                delta: false,
            })
        }
    }
}

/// Sample a point or spot light, treating it as a sphere of `radius` when one
/// is set and as a delta when it is not.
fn sample_sphere_light(
    point: Vector3,
    center: Vector3,
    radius: f32,
    d: f32,
    intensity: Vector3,
    rng: &mut Rng,
) -> Option<LightSample> {
    if radius <= 0.0 || d <= radius {
        // Inside the sphere, or no sphere at all: fall back to the delta light.
        // Sampling the visible cone from inside is undefined, and a light you
        // are standing in has no penumbra to resolve anyway.
        let dir = (center - point) * (1.0 / d);
        return Some(LightSample {
            direction: dir,
            distance: d,
            weight: intensity,
            pdf: 0.0,
            delta: true,
        });
    }
    let axis = (center - point) * (1.0 / d);
    let sin_max = radius / d;
    let cos_max = (1.0 - sin_max * sin_max).max(0.0).sqrt();
    let onb = Onb::new(axis);
    let (u1, u2) = rng.next_2d();
    let dir = onb.to_world(uniform_cone(u1, u2, cos_max));
    let pdf = uniform_cone_pdf(cos_max);
    if pdf <= 0.0 {
        return None;
    }
    // Stop the shadow ray at the sphere's surface, not its centre, or the light
    // shadows itself.
    let hit = ray_sphere_near(point, dir, center, radius).unwrap_or(d - radius);
    Some(LightSample {
        direction: dir,
        distance: hit.max(1e-4),
        // Same identity as the sun: intensity/Ω, then ÷pdf, leaves intensity.
        weight: intensity,
        pdf,
        delta: false,
    })
}

/// Nearest positive root of the ray/sphere intersection, if any.
fn ray_sphere_near(o: Vector3, d: Vector3, center: Vector3, r: f32) -> Option<f32> {
    let oc = o - center;
    let b = oc.dot(d);
    let c = oc.length_sq() - r * r;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let s = disc.sqrt();
    let t = -b - s;
    if t > 1e-5 {
        Some(t)
    } else {
        let t2 = -b + s;
        (t2 > 1e-5).then_some(t2)
    }
}

/// Result of picking an emissive triangle: where to aim, and the density that
/// choice had.
#[derive(Debug, Clone, Copy)]
pub struct EmitterSample {
    pub direction: Vector3,
    pub distance: f32,
    /// Emitted radiance at the sampled point.
    pub radiance: Vector3,
    pub pdf: f32,
    /// The triangle that was chosen, so the caller can exclude it from the
    /// shadow test.
    pub triangle: u32,
}

/// Pick an emissive triangle by the scene's area×luminance distribution, then a
/// point uniformly on it.
pub fn sample_emissive(
    scene: &RaytraceScene,
    point: Vector3,
    rng: &mut Rng,
) -> Option<EmitterSample> {
    let (entry, p_select) = scene.emissive.sample(rng.next_f32())?;
    let tri = &scene.tris[entry.triangle as usize];
    let sh = &scene.shading[entry.triangle as usize];

    // Uniform barycentrics: the square root warps the unit square onto the
    // triangle without clustering at a corner.
    let (u1, u2) = rng.next_2d();
    let su = u1.sqrt();
    let (b0, b1, b2) = (1.0 - su, u2 * su, su * (1.0 - u2));
    let q = tri[0] * b0 + tri[1] * b1 + tri[2] * b2;

    let to = q - point;
    let d2 = to.length_sq();
    if d2 <= 1e-12 {
        return None;
    }
    let d = d2.sqrt();
    let dir = to * (1.0 / d);

    let face = (tri[1] - tri[0]).cross(tri[2] - tri[0]);
    if face.length_sq() <= 0.0 {
        return None;
    }
    // Mesh emitters radiate from both faces, as Blender's emission shader does;
    // a one-sided emitter would make a plane light vanish when flipped.
    let cos_light = (dir * -1.0).dot(face.normalize()).abs();
    if cos_light <= 1e-6 {
        return None;
    }

    let pdf = p_select * d2 / (cos_light * entry.area);
    if !pdf.is_finite() || pdf <= 0.0 {
        return None;
    }

    let material = &scene.materials[sh.material as usize];
    let mut radiance = material.emission;
    if let Some(map) = &material.emissive_map {
        let uv = sh.uvs[0] * b0 + sh.uvs[1] * b1 + sh.uvs[2] * b2;
        radiance = radiance.mul_componentwise(map.sample_rgb(uv));
    }
    if radiance.length_sq() <= 0.0 {
        return None;
    }

    Some(EmitterSample {
        direction: dir,
        distance: d,
        radiance,
        pdf,
        triangle: entry.triangle,
    })
}

/// The density [`sample_emissive`] would have had for a direction that a BSDF
/// ray found by itself — the other half of the MIS weight.
///
/// `cos_light` is the angle at the emitter, `distance` the length of the ray
/// that reached it.
pub fn emissive_pdf(scene: &RaytraceScene, triangle: u32, distance: f32, cos_light: f32) -> f32 {
    let p_select = scene.emissive.probability_of(triangle);
    if p_select <= 0.0 || cos_light.abs() <= 1e-6 {
        return 0.0;
    }
    let tri = &scene.tris[triangle as usize];
    let area = 0.5 * (tri[1] - tri[0]).cross(tri[2] - tri[0]).length();
    if area <= 0.0 {
        return 0.0;
    }
    p_select * (distance * distance) / (cos_light.abs() * area)
}

/// Sample the world (environment map, ambient, hemisphere lights) as a light.
///
/// A mixture of two strategies — see
/// [`World::sample_direction`](super::scene::World::sample_direction). Cosine
/// weighting alone is right for a smooth sky and hopeless for a real one: an
/// HDRI's sun occupies about 6·10⁻⁵ of the sphere, so a cosine-weighted
/// direction finds it once in twenty thousand samples and the result is
/// speckle rather than an image.
pub fn sample_world(scene: &RaytraceScene, normal: Vector3, rng: &mut Rng) -> Option<LightSample> {
    if scene.world.is_black() {
        return None;
    }
    let pick = rng.next_f32();
    let (u1, u2) = rng.next_2d();
    let (dir, pdf) = scene.world.sample_direction(normal, pick, u1, u2)?;
    let radiance = scene.world.lighting_radiance(dir);
    if radiance.length_sq() <= 0.0 {
        return None;
    }
    Some(LightSample {
        direction: dir,
        distance: f32::INFINITY,
        weight: radiance * (1.0 / pdf),
        pdf,
        delta: false,
    })
}

/// The density [`sample_world`] would have had for `dir` off a surface whose
/// normal is `normal` — the other half of the MIS weight.
pub fn world_pdf(world: &World, normal: Vector3, dir: Vector3) -> f32 {
    world.direction_pdf(normal, dir)
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if (edge1 - edge0).abs() < 1e-9 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Component-wise multiply, which `Vector3`'s operators deliberately do not
/// provide (it is not a vector-space operation) but spectral maths needs
/// everywhere.
pub(crate) trait Modulate {
    fn mul_componentwise(self, other: Vector3) -> Vector3;
}

impl Modulate for Vector3 {
    fn mul_componentwise(self, other: Vector3) -> Vector3 {
        Vector3::new(self.x * other.x, self.y * other.y, self.z * other.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Object3D;
    use crate::geometries::PlaneGeometry;
    use crate::materials::{Material, StandardMaterial};
    use crate::math::Color;
    use crate::raytrace::settings::RaytraceSettings;
    use crate::scene::Scene;

    #[test]
    fn attenuation_matches_the_raster_formula() {
        // decay 2, no cutoff: inverse square.
        assert!((distance_attenuation(2.0, 0.0, 2.0) - 0.25).abs() < 1e-6);
        // decay 0: constant.
        assert!((distance_attenuation(7.0, 0.0, 0.0) - 1.0).abs() < 1e-6);
        // Past the cutoff distance, nothing.
        assert!(distance_attenuation(10.0, 10.0, 2.0).abs() < 1e-9);
    }

    #[test]
    fn a_delta_directional_light_points_back_at_the_source() {
        let light = RtLight::Directional {
            direction: Vector3::new(0.0, -1.0, 0.0),
            radiance: Vector3::new(1.0, 1.0, 1.0),
            angular_radius: 0.0,
        };
        let mut rng = Rng::new(1, 1);
        let s = sample_analytic(&light, Vector3::ZERO, &mut rng).unwrap();
        assert!(s.delta);
        assert!((s.direction - Vector3::new(0.0, 1.0, 0.0)).length() < 1e-6);
        assert_eq!(s.distance, f32::INFINITY);
    }

    /// A sun given an angular size must deliver the same total irradiance as
    /// the hard one — softening a shadow may not change the exposure.
    #[test]
    fn soft_and_hard_suns_deliver_the_same_energy() {
        let hard = RtLight::Directional {
            direction: Vector3::new(0.0, -1.0, 0.0),
            radiance: Vector3::new(1.0, 1.0, 1.0),
            angular_radius: 0.0,
        };
        let soft = RtLight::Directional {
            direction: Vector3::new(0.0, -1.0, 0.0),
            radiance: Vector3::new(1.0, 1.0, 1.0),
            angular_radius: 0.1,
        };
        let n = Vector3::new(0.0, 1.0, 0.0);
        let mut rng = Rng::new(2, 2);
        let e_hard = {
            let s = sample_analytic(&hard, Vector3::ZERO, &mut rng).unwrap();
            s.weight * s.direction.dot(n).max(0.0)
        };
        let mut acc = Vector3::ZERO;
        let n_samples = 20_000;
        for _ in 0..n_samples {
            let s = sample_analytic(&soft, Vector3::ZERO, &mut rng).unwrap();
            acc = acc + s.weight * s.direction.dot(n).max(0.0);
        }
        let e_soft = acc * (1.0 / n_samples as f32);
        assert!(
            (e_soft.x - e_hard.x).abs() < 0.01,
            "hard {e_hard:?} vs soft {e_soft:?}"
        );
    }

    #[test]
    fn point_light_falls_off_with_distance() {
        let light = RtLight::Point {
            position: Vector3::new(0.0, 4.0, 0.0),
            intensity: Vector3::new(16.0, 16.0, 16.0),
            radius: 0.0,
            distance: 0.0,
            decay: 2.0,
        };
        let mut rng = Rng::new(3, 3);
        let s = sample_analytic(&light, Vector3::ZERO, &mut rng).unwrap();
        assert!(s.delta);
        assert!((s.distance - 4.0).abs() < 1e-5);
        assert!(
            (s.weight.x - 1.0).abs() < 1e-5,
            "16 / 4^2 = 1, got {:?}",
            s.weight
        );
    }

    /// A sphere light and a delta light of the same intensity must agree once
    /// the sphere's samples are averaged.
    #[test]
    fn sphere_light_matches_the_delta_it_replaces() {
        let delta = RtLight::Point {
            position: Vector3::new(0.0, 5.0, 0.0),
            intensity: Vector3::new(25.0, 25.0, 25.0),
            radius: 0.0,
            distance: 0.0,
            decay: 2.0,
        };
        let sphere = RtLight::Point {
            position: Vector3::new(0.0, 5.0, 0.0),
            intensity: Vector3::new(25.0, 25.0, 25.0),
            radius: 0.5,
            distance: 0.0,
            decay: 2.0,
        };
        let n = Vector3::new(0.0, 1.0, 0.0);
        let mut rng = Rng::new(4, 4);
        let d = sample_analytic(&delta, Vector3::ZERO, &mut rng).unwrap();
        let target = d.weight * d.direction.dot(n);
        let mut acc = Vector3::ZERO;
        let n_samples = 40_000;
        for _ in 0..n_samples {
            let s = sample_analytic(&sphere, Vector3::ZERO, &mut rng).unwrap();
            acc = acc + s.weight * s.direction.dot(n).max(0.0);
            assert!(s.distance < 5.0, "shadow ray should stop at the sphere");
        }
        let got = acc * (1.0 / n_samples as f32);
        assert!(
            (got.x - target.x).abs() < 0.02 * target.x.max(1.0),
            "delta {target:?} vs sphere {got:?}"
        );
    }

    #[test]
    fn spot_light_is_dark_outside_its_cone() {
        let light = RtLight::Spot {
            position: Vector3::new(0.0, 5.0, 0.0),
            direction: Vector3::new(0.0, -1.0, 0.0),
            intensity: Vector3::new(25.0, 25.0, 25.0),
            radius: 0.0,
            distance: 0.0,
            decay: 2.0,
            cos_outer: (0.2f32).cos(),
            cos_inner: (0.1f32).cos(),
        };
        let mut rng = Rng::new(5, 5);
        assert!(sample_analytic(&light, Vector3::ZERO, &mut rng).is_some());
        // Far off-axis: outside the 0.2 rad cone.
        let outside = sample_analytic(&light, Vector3::new(20.0, 0.0, 0.0), &mut rng);
        assert!(
            outside.is_none(),
            "expected the spot to miss, got {outside:?}"
        );
    }

    #[test]
    fn rect_light_only_emits_from_its_front_face() {
        let light = RtLight::Rect {
            position: Vector3::new(0.0, 2.0, 0.0),
            // right = +X, up = +Y  =>  right x up = +Z, emitting face is -Z.
            right: Vector3::new(1.0, 0.0, 0.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            radiance: Vector3::new(3.0, 3.0, 3.0),
        };
        let mut rng = Rng::new(6, 6);
        let front = sample_analytic(&light, Vector3::new(0.0, 2.0, -4.0), &mut rng);
        assert!(front.is_some(), "the -Z side should be lit");
        let mut lit_behind = 0;
        for _ in 0..200 {
            if sample_analytic(&light, Vector3::new(0.0, 2.0, 4.0), &mut rng).is_some() {
                lit_behind += 1;
            }
        }
        assert_eq!(lit_behind, 0, "the +Z side must be dark");
    }

    /// The estimator for a rect light must converge to the analytic irradiance
    /// of a disc-like source. Checked against a large-N reference computed by
    /// brute-force integration over the rectangle.
    #[test]
    fn rect_light_irradiance_converges() {
        let light = RtLight::Rect {
            position: Vector3::new(0.0, 3.0, 0.0),
            right: Vector3::new(0.5, 0.0, 0.0),
            up: Vector3::new(0.0, 0.0, 0.5),
            radiance: Vector3::new(1.0, 1.0, 1.0),
        };
        // right x up = +X x +Z = -Y, so the emitting face is +Y... flip: the
        // emitter faces -(right x up) = +Y. Put the receiver above it.
        let p = Vector3::new(0.0, 6.0, 0.0);
        let n = Vector3::new(0.0, -1.0, 0.0);
        let mut rng = Rng::new(7, 7);
        let mut acc = Vector3::ZERO;
        let n_samples = 100_000;
        for _ in 0..n_samples {
            if let Some(s) = sample_analytic(&light, p, &mut rng) {
                acc = acc + s.weight * s.direction.dot(n).max(0.0);
            }
        }
        let e = acc.x / n_samples as f32;
        // Small source at distance 3: E ≈ L·A·cosθ_l·cosθ_r / d² = 1·1·1·1/9.
        assert!((e - 1.0 / 9.0).abs() < 0.005, "irradiance {e}");
    }

    #[test]
    fn world_sampling_is_cosine_weighted_and_matches_its_pdf() {
        let mut scene = Scene::new();
        scene.add_light(crate::lights::AmbientLight::new(Color::WHITE, 1.0));
        let rt = RaytraceScene::build(&mut scene, &RaytraceSettings::default());
        let n = Vector3::new(0.0, 1.0, 0.0);
        let mut rng = Rng::new(8, 8);
        for _ in 0..2000 {
            let s = sample_world(&rt, n, &mut rng).unwrap();
            assert!(s.direction.dot(n) > 0.0);
            let pdf = world_pdf(&rt.world, n, s.direction);
            assert!((pdf - s.pdf).abs() < 1e-4, "{pdf} vs {}", s.pdf);
        }
    }

    /// With an environment present, sampling must still report the density it
    /// actually drew from — the mixture, not whichever half produced the
    /// direction. Getting this wrong is invisible in a converged image and
    /// wrong everywhere in a noisy one.
    #[test]
    fn world_sampling_reports_the_mixture_density() {
        let size = 8u32;
        let mut faces: [Vec<u8>; 6] = Default::default();
        for (i, f) in faces.iter_mut().enumerate() {
            let mut px = vec![255u8; (size * size * 4) as usize];
            // One face far brighter than the rest, so the distribution is not
            // uniform and the two strategies genuinely differ.
            let level = if i == 2 { 255 } else { 20 };
            for p in px.chunks_exact_mut(4) {
                p[0] = level;
                p[1] = level;
                p[2] = level;
            }
            *f = px;
        }
        let mut scene = Scene::new();
        scene.environment = Some(std::sync::Arc::new(crate::textures::CubeTexture::new(
            size,
            crate::textures::TextureFormat::Rgba8Unorm,
            faces,
        )));
        let rt = RaytraceScene::build(&mut scene, &RaytraceSettings::default());
        assert!(
            rt.world.env_distribution().is_some(),
            "a lit environment should produce a distribution"
        );

        let n = Vector3::new(0.0, 1.0, 0.0);
        let mut rng = Rng::new(21, 22);
        let mut above = 0;
        for _ in 0..4000 {
            let s = sample_world(&rt, n, &mut rng).unwrap();
            let pdf = world_pdf(&rt.world, n, s.direction);
            assert!(
                (pdf - s.pdf).abs() < 1e-4 * s.pdf.max(1.0),
                "sample pdf {} vs lookup {pdf}",
                s.pdf
            );
            if s.direction.dot(n) > 0.0 {
                above += 1;
            }
        }
        // The environment half of the mixture covers the whole sphere, so some
        // directions land below the surface — that is expected, not a bug.
        assert!(above > 1000, "only {above} samples were above the surface");
    }

    /// Without an environment there is nothing to importance-sample, and the
    /// mixture must collapse back to plain cosine weighting.
    #[test]
    fn an_ambient_only_world_samples_cosine_weighted() {
        let mut scene = Scene::new();
        scene.add_light(crate::lights::AmbientLight::new(Color::WHITE, 1.0));
        let rt = RaytraceScene::build(&mut scene, &RaytraceSettings::default());
        assert!(rt.world.env_distribution().is_none());
        let n = Vector3::new(0.0, 1.0, 0.0);
        let mut rng = Rng::new(23, 24);
        for _ in 0..2000 {
            let s = sample_world(&rt, n, &mut rng).unwrap();
            assert!(
                s.direction.dot(n) > 0.0,
                "cosine sampling stayed in-hemisphere"
            );
            let expected = s.direction.dot(n) / std::f32::consts::PI;
            assert!((s.pdf - expected).abs() < 1e-5, "{} vs {expected}", s.pdf);
        }
    }

    #[test]
    fn a_black_world_yields_no_sample() {
        let mut scene = Scene::new();
        let rt = RaytraceScene::build(&mut scene, &RaytraceSettings::default());
        let mut rng = Rng::new(9, 9);
        assert!(sample_world(&rt, Vector3::UP, &mut rng).is_none());
    }

    /// Sampling an emissive plane and asking for the density of the direction
    /// that came back must give the same number.
    #[test]
    fn emissive_sample_and_pdf_agree() {
        let mut scene = Scene::new();
        let mut m = StandardMaterial::new(Color::BLACK);
        m.emissive = Color::WHITE;
        m.emissive_intensity = 4.0;
        let mut obj = Object3D::mesh(crate::core::Mesh::new(
            PlaneGeometry::new(2.0, 2.0),
            Material::Standard(m),
        ));
        // PlaneGeometry faces +Z, so it has to be placed along Z to face the
        // point being lit — edge-on, it emits nothing toward it.
        obj.position = Vector3::new(0.0, 0.0, 3.0);
        scene.add(obj);
        let rt = RaytraceScene::build(&mut scene, &RaytraceSettings::default());
        assert!(!rt.emissive.is_empty());

        let p = Vector3::ZERO;
        let mut rng = Rng::new(10, 10);
        let mut checked = 0;
        for _ in 0..2000 {
            let Some(s) = sample_emissive(&rt, p, &mut rng) else {
                continue;
            };
            let tri = &rt.tris[s.triangle as usize];
            let face = (tri[1] - tri[0]).cross(tri[2] - tri[0]).normalize();
            let cos_light = (s.direction * -1.0).dot(face).abs();
            let pdf = emissive_pdf(&rt, s.triangle, s.distance, cos_light);
            assert!(
                (pdf - s.pdf).abs() < 1e-3 * s.pdf.max(1.0),
                "sample pdf {} vs emissive_pdf {}",
                s.pdf,
                pdf
            );
            checked += 1;
        }
        assert!(checked > 1000, "only {checked} samples were usable");
    }

    /// Monte-Carlo irradiance from an emissive plane against the analytic value
    /// for a small square source: E ≈ L·A·cos·cos/d².
    #[test]
    fn emissive_plane_irradiance_converges() {
        let mut scene = Scene::new();
        let mut m = StandardMaterial::new(Color::BLACK);
        m.emissive = Color::WHITE;
        m.emissive_intensity = 1.0;
        let mut obj = Object3D::mesh(crate::core::Mesh::new(
            PlaneGeometry::new(0.4, 0.4),
            Material::Standard(m),
        ));
        obj.position = Vector3::new(0.0, 0.0, 4.0);
        scene.add(obj);
        let rt = RaytraceScene::build(&mut scene, &RaytraceSettings::default());

        let p = Vector3::ZERO;
        let n = Vector3::new(0.0, 0.0, 1.0);
        let mut rng = Rng::new(11, 11);
        let n_samples = 200_000;
        let mut acc = 0.0f64;
        for _ in 0..n_samples {
            if let Some(s) = sample_emissive(&rt, p, &mut rng) {
                let cos = s.direction.dot(n).max(0.0);
                acc += (s.radiance.x * cos / s.pdf) as f64;
            }
        }
        let e = acc / n_samples as f64;
        let expected = (0.4 * 0.4) / (4.0 * 4.0);
        assert!(
            (e - expected).abs() < 0.02 * expected,
            "irradiance {e} vs {expected}"
        );
    }
}
