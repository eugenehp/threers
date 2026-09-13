//! The path-tracing integrator: follow a ray from the camera, and at every
//! surface it meets, add what the lights contribute there and scatter onward.
//!
//! The estimator is unidirectional path tracing with next-event estimation and
//! multiple importance sampling — the same construction Cycles uses. Each
//! bounce contributes twice: once by sampling a light and testing whether it is
//! visible (which resolves small bright sources), and once by following the
//! BSDF and taking whatever it lands on (which resolves large dim ones and
//! sharp reflections). The power heuristic weights the two so that neither is
//! double-counted and each dominates where it is the better estimator.
//!
//! Everything here is a *mean*, not a sum: one call to [`Integrator::trace`] is
//! one sample of an integral, and the image is the average of many. That is why
//! the output is noisy at low sample counts and why the noise is unbiased —
//! it converges to the right answer rather than to a smoothed one.

use crate::math::{Vector2, Vector3};

use super::bsdf::{Bsdf, Surface};
use super::bvh::RtHit;
use super::lights::{
    emissive_pdf, sample_analytic, sample_emissive, sample_world, world_pdf, Modulate,
};
use super::sampler::{power_heuristic, uniform_sphere, Rng};
use super::scene::RaytraceScene;
use super::settings::RaytraceSettings;

/// One traced path, and the auxiliary channels its first hit produced.
#[derive(Debug, Clone, Copy)]
pub struct PathResult {
    pub radiance: Vector3,
    pub alpha: f32,
    /// Base colour at the first scattering surface, unlit.
    pub albedo: Vector3,
    /// Shading normal there.
    pub normal: Vector3,
    /// Distance from the camera to it, `INFINITY` if the path never scattered.
    pub depth: f32,
}

/// Per-thread reusable buffers, so tracing a path allocates nothing.
#[derive(Debug, Default, Clone)]
pub struct Scratch {
    hits: Vec<RtHit>,
}

/// Everything a hit needs before it can be shaded.
///
/// Two normals, and the difference matters. `ns`/`ng` follow the triangle's
/// *winding*, so their sign says which side of the surface the ray is on —
/// which is the only way a BSDF can tell entering glass from leaving it, and
/// therefore whether the relative index of refraction is `1/ior` or `ior`.
/// `ns_facing`/`ng_facing` are flipped toward the ray, which is what light
/// sampling and ray offsets want.
struct HitInfo {
    /// World-space hit point.
    p: Vector3,
    /// Geometric normal by winding.
    ng: Vector3,
    /// Shading normal, aligned with `ng`.
    ns: Vector3,
    /// `ng`, flipped to the side the ray arrived from.
    ng_facing: Vector3,
    /// `ns`, likewise.
    ns_facing: Vector3,
    /// The triangle's own normal, *not* flipped — the emissive pdf needs the
    /// true orientation.
    face: Vector3,
    uv: Vector2,
    /// Interpolated vertex colour, white when the geometry had none.
    vcolor: Vector3,
    material: u32,
    /// The ray struck the side the winding calls the front.
    front: bool,
    /// Which triangle was hit, so the tangent frame can be rebuilt from it.
    triangle: usize,
}

/// Binds a scene and its settings for tracing.
pub struct Integrator<'a> {
    scene: &'a RaytraceScene,
    settings: &'a RaytraceSettings,
    /// Ray offset, scaled to the scene so that a model in millimetres and the
    /// same model in kilometres both work. A fixed epsilon cannot: too small
    /// and a large scene self-shadows, too large and a small one leaks.
    epsilon: f32,
}

impl<'a> Integrator<'a> {
    pub fn new(scene: &'a RaytraceScene, settings: &'a RaytraceSettings) -> Self {
        let epsilon = ray_epsilon(scene, settings);
        Self {
            scene,
            settings,
            epsilon,
        }
    }

    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Trace one path and return its contribution.
    pub fn trace(
        &self,
        mut origin: Vector3,
        mut dir: Vector3,
        rng: &mut Rng,
        scratch: &mut Scratch,
    ) -> PathResult {
        let mut throughput = Vector3::ONE;
        let mut radiance = Vector3::ZERO;
        let mut alpha = 1.0f32;
        let mut albedo = Vector3::ZERO;
        let mut normal = Vector3::ZERO;
        let mut depth = f32::INFINITY;
        let mut depth_recorded = false;
        // The denoiser's guide passes are not "whatever the first hit was".
        // A mirror's own colour and normal say nothing about the image in it,
        // and a glass ball's say nothing about the room behind it — so the
        // guides follow the path through specular and transmissive surfaces,
        // carrying their tint, until they reach something rough enough to
        // describe. This is Cycles' scheme; it is what lets a guided denoiser
        // clean a reflection instead of smearing it.
        let mut feature_throughput = Vector3::ONE;
        let mut features_open = true;

        let mut travelled = 0.0f32;
        let mut bounce = 0u32;
        let mut transparent_bounces = 0u32;
        // A camera ray sees `scene.background`; every later ray sees the
        // environment. Keeping them apart is what lets a subject be lit by an
        // HDRI while sitting on a flat backdrop.
        let mut camera_ray = true;
        // MIS state carried from the previous scattering event.
        let mut prev_specular = true;
        let mut prev_bsdf_pdf = 0.0f32;
        let mut prev_normal = dir;
        // Extinction of the medium the ray is currently inside, if any.
        let mut medium: Option<Vector3> = None;

        loop {
            let hit = self.scene.bvh.intersect(
                &self.scene.tris,
                origin,
                dir,
                self.epsilon,
                f32::INFINITY,
            );

            // Beer–Lambert absorption over whatever was crossed. Applied before
            // the hit is shaded, because it attenuates the light *arriving*.
            if let Some(sigma) = medium {
                let t = hit.map(|h| h.t).unwrap_or(0.0);
                if t > 0.0 {
                    throughput = throughput.mul_componentwise(exp3(sigma * -t));
                }
            }

            let Some(hit) = hit else {
                let (l, a) = if camera_ray {
                    self.scene.world.camera_radiance(dir)
                } else {
                    (self.scene.world.lighting_radiance(dir), 1.0)
                };
                if camera_ray {
                    alpha = a;
                }
                let weight = if prev_specular {
                    1.0
                } else {
                    power_heuristic(
                        1.0,
                        prev_bsdf_pdf,
                        1.0,
                        world_pdf(&self.scene.world, prev_normal, dir),
                    )
                };
                self.emit(
                    &mut radiance,
                    throughput.mul_componentwise(l) * weight,
                    bounce,
                );
                if features_open {
                    // A pixel showing only sky is not noisy; giving the
                    // denoiser the sky's colour keeps it from treating the
                    // background as detail to preserve. Compressed into [0, 1)
                    // first, because the guide is a reflectance-like quantity
                    // and an HDRI's sun is four orders of magnitude outside
                    // that range.
                    albedo = albedo + reinhard(l).mul_componentwise(feature_throughput);
                    normal = normal + (-dir) * average(feature_throughput);
                }
                break;
            };

            // Point the sampler at this bounce's block of dimensions before
            // anything draws from it, so the same decision lands on the same
            // dimension in every sample of this pixel — which is the whole
            // reason a stratified sequence beats an independent one.
            rng.set_bounce(bounce + 1);

            travelled += hit.t;
            let mut info = self.hit_info(&hit, dir);
            let material = &self.scene.materials[info.material as usize];

            // --- coverage: below 1, the surface is partly not there.
            let mut surface_alpha = material.opacity;
            // Vertex colour multiplies the material, matching the raster path.
            // Dropping it turned a line set with a colour per segment into one
            // flat shade.
            let mut base_color = material.base_color.mul_componentwise(info.vcolor);
            if let Some(map) = &material.base_color_map {
                let s = map.sample(info.uv);
                base_color = base_color.mul_componentwise(Vector3::new(s[0], s[1], s[2]));
                surface_alpha *= s[3];
            }
            if material.alpha_test > 0.0 && surface_alpha < material.alpha_test {
                surface_alpha = 0.0;
            }

            // A normal map replaces the shading normal outright, so it has to be
            // applied before anything reads one: the cosine term in the
            // rendering equation is taken against the normal the BSDF was built
            // on, and light sampling has to work in that normal's hemisphere.
            // Perturbing inside the BSDF constructor instead left every other
            // use of `ns` on the interpolated normal, which biases every
            // normal-mapped surface.
            if let Some(map) = &material.normal_map {
                let ns = self.perturb_normal(&info, map, material.normal_scale);
                info.ns = ns;
                info.ns_facing = if info.front { ns } else { -ns };
            }
            if surface_alpha < 1.0 && rng.next_f32() >= surface_alpha {
                transparent_bounces += 1;
                if transparent_bounces > self.settings.transparent_max_bounces {
                    if camera_ray {
                        alpha = 0.0;
                    }
                    break;
                }
                // Straight through, undeviated and unattenuated: the ray missed
                // the material rather than passing through it.
                origin = info.p - info.ng_facing * self.epsilon;
                continue;
            }

            // --- emission, weighted against the light sample that could have
            // found this same surface.
            let mut emission = material.emission.mul_componentwise(info.vcolor);
            if let Some(map) = &material.emissive_map {
                emission = emission.mul_componentwise(map.sample_rgb(info.uv));
            }
            if emission.length_sq() > 0.0 {
                let weight = if prev_specular {
                    1.0
                } else {
                    let cos_light = dir.dot(info.face).abs();
                    let light_pdf = emissive_pdf(self.scene, hit.triangle, hit.t, cos_light);
                    power_heuristic(1.0, prev_bsdf_pdf, 1.0, light_pdf)
                };
                self.emit(
                    &mut radiance,
                    throughput.mul_componentwise(emission) * weight,
                    bounce,
                );
            }

            let surface = self.surface_at(&info, base_color, material);

            if features_open {
                if material.unlit {
                    // No BSDF to describe; the surface is its own emission.
                    albedo = albedo + reinhard(emission).mul_componentwise(feature_throughput);
                    normal = normal + info.ns_facing * average(feature_throughput);
                    features_open = false;
                } else {
                    let (weight, diffuse_albedo, deferred) = guide_split(&surface);
                    if weight > 0.0 {
                        albedo =
                            albedo + diffuse_albedo.mul_componentwise(feature_throughput) * weight;
                        normal = normal + info.ns_facing * (weight * average(feature_throughput));
                    }
                    feature_throughput = feature_throughput.mul_componentwise(deferred);
                    if feature_throughput
                        .x
                        .max(feature_throughput.y)
                        .max(feature_throughput.z)
                        < 1e-4
                    {
                        features_open = false;
                    }
                }
            }
            if !depth_recorded {
                // Depth stays the distance to the first *hit*, not to wherever
                // the guides ended up. Its job is to say "this is a different
                // surface from that one", and for a mirror that is the mirror.
                depth = travelled;
                depth_recorded = true;
            }

            // An unlit surface scatters nothing; the path stops at it.
            if material.unlit || bounce >= self.settings.max_bounces {
                break;
            }

            let bsdf = self.make_bsdf(&info, surface);
            let wo = -dir;
            let shadow_origin = info.p + info.ng_facing * self.epsilon;

            if !bsdf.is_delta() {
                self.direct_light(
                    &bsdf,
                    &info,
                    wo,
                    shadow_origin,
                    throughput,
                    bounce,
                    &mut radiance,
                    rng,
                    scratch,
                );
                if material.subsurface > 0.0 {
                    self.subsurface_walk(
                        material,
                        &info,
                        base_color,
                        &surface,
                        wo,
                        throughput,
                        bounce,
                        &mut radiance,
                        rng,
                        scratch,
                    );
                }
            }

            let Some(sample) = bsdf.sample(wo, rng) else {
                break;
            };
            if sample.pdf <= 0.0 {
                break;
            }
            let cos = sample.direction.dot(info.ns).abs();
            if cos <= 0.0 {
                break;
            }
            throughput = throughput.mul_componentwise(sample.value * (cos / sample.pdf));
            if throughput.length_sq() <= 0.0 || !throughput.x.is_finite() {
                break;
            }

            if sample.transmitted {
                medium = if info.front {
                    medium_extinction(material)
                } else {
                    None
                };
            }

            origin = if sample.transmitted {
                info.p - info.ng_facing * self.epsilon
            } else {
                shadow_origin
            };
            dir = sample.direction;
            prev_specular = sample.specular;
            prev_bsdf_pdf = sample.pdf;
            prev_normal = info.ns_facing;
            camera_ray = false;
            bounce += 1;

            // Russian roulette: rather than truncate every path at the same
            // depth (which darkens the image), kill dim paths at random and
            // scale the survivors up. Unbiased, and it spends the remaining
            // budget on the paths that still carry energy.
            if bounce > self.settings.min_bounces {
                let q = throughput
                    .x
                    .max(throughput.y)
                    .max(throughput.z)
                    .clamp(0.0, 0.95);
                if q <= 0.0 || rng.next_f32() >= q {
                    break;
                }
                throughput = throughput * (1.0 / q);
            }
        }

        if self.scene.world.fog_mode > 0 {
            let d = if depth_recorded { depth } else { self.scene.scale() * 4.0 };
            let (vis, fog) = self.scene.world.fog_at(d);
            radiance = radiance * vis + fog;
        }

        PathResult {
            radiance: sanitise(radiance),
            alpha,
            albedo: sanitise(albedo),
            normal,
            depth,
        }
    }

    /// Next-event estimation over all three kinds of emitter.
    #[allow(clippy::too_many_arguments)]
    fn direct_light(
        &self,
        bsdf: &Bsdf,
        info: &HitInfo,
        wo: Vector3,
        shadow_origin: Vector3,
        throughput: Vector3,
        bounce: u32,
        radiance: &mut Vector3,
        rng: &mut Rng,
        scratch: &mut Scratch,
    ) {
        // --- analytic lights. One is picked uniformly and its contribution
        // scaled by the count, rather than shadow-raying all of them: the cost
        // of direct lighting then does not grow with the number of lights, only
        // its variance does.
        let n_lights = self.scene.lights.len();
        if n_lights > 0 {
            let idx = rng.next_index(n_lights);
            if let Some(ls) = sample_analytic(&self.scene.lights[idx], info.p, rng) {
                let (f, _) = bsdf.eval(wo, ls.direction);
                let cos = ls.direction.dot(info.ns).abs();
                if cos > 0.0 && f.length_sq() > 0.0 {
                    let vis = self.visibility(shadow_origin, ls.direction, ls.distance, scratch);
                    if vis.length_sq() > 0.0 {
                        let c = throughput
                            .mul_componentwise(f)
                            .mul_componentwise(ls.weight)
                            .mul_componentwise(vis)
                            * (cos * n_lights as f32);
                        self.emit(radiance, c, bounce);
                    }
                }
            }
        }

        // --- emissive geometry, with MIS against the BSDF.
        if !self.scene.emissive.is_empty() {
            if let Some(es) = sample_emissive(self.scene, info.p, rng) {
                let (f, bsdf_pdf) = bsdf.eval(wo, es.direction);
                let cos = es.direction.dot(info.ns).abs();
                if cos > 0.0 && es.pdf > 0.0 && f.length_sq() > 0.0 {
                    // Stop just short of the emitter so the shadow ray does not
                    // hit the very triangle it is aimed at.
                    let t_max = es.distance * (1.0 - 1e-3);
                    let vis = self.visibility(shadow_origin, es.direction, t_max, scratch);
                    if vis.length_sq() > 0.0 {
                        let mis = power_heuristic(1.0, es.pdf, 1.0, bsdf_pdf);
                        let c = throughput
                            .mul_componentwise(f)
                            .mul_componentwise(es.radiance)
                            .mul_componentwise(vis)
                            * (cos * mis / es.pdf);
                        self.emit(radiance, c, bounce);
                    }
                }
            }
        }

        // --- the world, likewise.
        if let Some(ws) = sample_world(self.scene, info.ns_facing, rng) {
            let (f, bsdf_pdf) = bsdf.eval(wo, ws.direction);
            let cos = ws.direction.dot(info.ns).abs();
            if cos > 0.0 && f.length_sq() > 0.0 {
                let vis = self.visibility(shadow_origin, ws.direction, f32::INFINITY, scratch);
                if vis.length_sq() > 0.0 {
                    let mis = power_heuristic(1.0, ws.pdf, 1.0, bsdf_pdf);
                    let c = throughput
                        .mul_componentwise(f)
                        .mul_componentwise(ws.weight)
                        .mul_componentwise(vis)
                        * (cos * mis);
                    self.emit(radiance, c, bounce);
                }
            }
        }
    }

    /// A short random walk inside subsurface materials, adding direct lighting
    /// at each scatter point. Biased but stable; complements the grazing boost
    /// in the BSDF.
    #[allow(clippy::too_many_arguments)]
    fn subsurface_walk(
        &self,
        material: &super::scene::RtMaterial,
        info: &HitInfo,
        base_color: Vector3,
        surface: &Surface,
        wo: Vector3,
        throughput: Vector3,
        bounce: u32,
        radiance: &mut Vector3,
        rng: &mut Rng,
        scratch: &mut Scratch,
    ) {
        let ss = material.subsurface.clamp(0.0, 1.0);
        if ss <= 0.0 {
            return;
        }
        let scale = self.scene.scale();
        let mfp = ((material.subsurface_radius.x
            + material.subsurface_radius.y
            + material.subsurface_radius.z)
            / 3.0)
            * scale
            * 0.02;
        let mfp = mfp.max(self.epsilon * 4.0);
        let max_steps = 6u32;
        let mut pos = info.p - info.ng_facing * self.epsilon * 2.0;
        let mut walk_tp = throughput.mul_componentwise(base_color) * ss;
        let bsdf = Bsdf::new(*surface, info.ns, info.ng);
        let shadow_bias = self.epsilon * 2.0;

        for _ in 0..max_steps {
            let u = rng.next_f32().max(1e-6);
            let step = -mfp * u.ln();
            let scatter_dir = uniform_sphere(rng.next_f32(), rng.next_f32());
            pos = pos + scatter_dir * step;

            // Reconnect to the surface for lighting.
            let to_surface = -info.ng_facing;
            if let Some(exit) = self.scene.bvh.intersect(
                &self.scene.tris,
                pos,
                to_surface,
                self.epsilon,
                mfp * 8.0,
            ) {
                let exit_info = self.hit_info(&exit, to_surface);
                if exit_info.material == info.material {
                    let exit_origin = exit_info.p + exit_info.ng_facing * shadow_bias;
                    self.direct_light(
                        &bsdf,
                        &HitInfo {
                            p: exit_info.p,
                            ng: exit_info.ng,
                            ns: info.ns,
                            ng_facing: exit_info.ng_facing,
                            ns_facing: info.ns_facing,
                            face: exit_info.face,
                            uv: exit_info.uv,
                            // The exit point's own colour: this is a different
                            // surface, even though the material is shared.
                            vcolor: exit_info.vcolor,
                            material: info.material,
                            front: exit_info.front,
                            triangle: exit_info.triangle,
                        },
                        wo,
                        exit_origin,
                        walk_tp,
                        bounce,
                        radiance,
                        rng,
                        scratch,
                    );
                }
            }

            let q = walk_tp
                .x
                .max(walk_tp.y)
                .max(walk_tp.z)
                .clamp(0.05, 0.95);
            if rng.next_f32() >= q {
                break;
            }
            walk_tp = walk_tp * (1.0 / q);
        }
    }

    /// How much of a shadow ray survives: `1` for clear, `0` for blocked, and
    /// something between when it crossed cutouts.
    fn visibility(
        &self,
        origin: Vector3,
        dir: Vector3,
        t_max: f32,
        scratch: &mut Scratch,
    ) -> Vector3 {
        let t_max = if t_max.is_finite() {
            t_max - self.epsilon
        } else {
            f32::INFINITY
        };
        if t_max <= self.epsilon {
            return Vector3::ONE;
        }
        if self.scene.shadows_all_opaque {
            return if self
                .scene
                .bvh
                .occluded(&self.scene.tris, origin, dir, self.epsilon, t_max)
            {
                Vector3::ZERO
            } else {
                Vector3::ONE
            };
        }

        self.scene.bvh.intersect_all(
            &self.scene.tris,
            origin,
            dir,
            self.epsilon,
            t_max,
            &mut scratch.hits,
        );
        let mut transmittance = Vector3::ONE;
        let mut layers = 0u32;
        for h in &scratch.hits {
            let sh = &self.scene.shading[h.triangle as usize];
            let m = &self.scene.materials[sh.material as usize];
            // Refractive glass blocks the shadow ray. Letting light through it
            // unbent would be a lie about where the light went; the light that
            // really does get through arrives along BSDF-sampled paths, which
            // is also where the caustic comes from.
            if m.transmission > 0.0 && m.opacity >= 1.0 {
                if self.settings.caustic_glass_shadows {
                    transmittance = transmittance.mul_componentwise(Vector3::new(0.12, 0.12, 0.12));
                    layers += 1;
                    continue;
                }
                return Vector3::ZERO;
            }
            let w = 1.0 - h.u - h.v;
            let uv = sh.uvs[0] * w + sh.uvs[1] * h.u + sh.uvs[2] * h.v;
            let mut a = m.opacity;
            if let Some(map) = &m.base_color_map {
                a *= map.sample(uv)[3];
            }
            if m.alpha_test > 0.0 && a < m.alpha_test {
                a = 0.0;
            }
            if a >= 1.0 {
                return Vector3::ZERO;
            }
            transmittance = transmittance * (1.0 - a);
            layers += 1;
            if layers > self.settings.transparent_max_bounces
                || transmittance.x.max(transmittance.y).max(transmittance.z) < 1e-4
            {
                break;
            }
        }
        transmittance
    }

    /// Interpolate everything the shading code needs from a raw BVH hit.
    fn hit_info(&self, hit: &RtHit, dir: Vector3) -> HitInfo {
        let tri = &self.scene.tris[hit.triangle as usize];
        let sh = &self.scene.shading[hit.triangle as usize];
        let w = 1.0 - hit.u - hit.v;
        let p = tri[0] * w + tri[1] * hit.u + tri[2] * hit.v;
        let uv = sh.uvs[0] * w + sh.uvs[1] * hit.u + sh.uvs[2] * hit.v;
        let vcolor = sh.colors[0] * w + sh.colors[1] * hit.u + sh.colors[2] * hit.v;

        let face_raw = (tri[1] - tri[0]).cross(tri[2] - tri[0]);
        let face = if face_raw.length_sq() > 0.0 {
            face_raw.normalize()
        } else {
            -dir
        };
        let front = dir.dot(face) < 0.0;
        let ng = face;

        let mut ns = sh.normals[0] * w + sh.normals[1] * hit.u + sh.normals[2] * hit.v;
        ns = if ns.length_sq() > 1e-12 {
            ns.normalize()
        } else {
            ng
        };
        // A vertex normal that disagrees with its own triangle — an
        // inconsistently wound mesh, or interpolation near a silhouette — is
        // aligned to the triangle rather than to the ray, so that the winding
        // stays the single definition of which side is outside.
        if ns.dot(ng) < 0.0 {
            ns = -ns;
        }

        HitInfo {
            p,
            ng,
            ns,
            ng_facing: if front { ng } else { -ng },
            ns_facing: if front { ns } else { -ns },
            face,
            uv,
            vcolor,
            material: sh.material,
            front,
            triangle: hit.triangle as usize,
        }
    }

    /// Sample the material's maps and bind the result to the shading frame.
    /// The textured surface parameters at a hit.
    ///
    /// Separate from [`Self::make_bsdf`] because the denoiser's guide passes
    /// need the same roughness and metalness the BSDF gets — deciding whether a
    /// surface is "describable" or a mirror to see through is exactly a
    /// question about those numbers.
    fn surface_at(
        &self,
        info: &HitInfo,
        base_color: Vector3,
        material: &super::scene::RtMaterial,
    ) -> Surface {
        // three.js reads roughness from green and metalness from blue, so one
        // packed ORM texture serves both. Matching that is what lets a glTF
        // model render the same in both renderers.
        let mut roughness = material.roughness;
        if let Some(map) = &material.roughness_map {
            roughness *= map.sample(info.uv)[1];
        }
        let mut metallic = material.metallic;
        if let Some(map) = &material.metallic_map {
            metallic *= map.sample(info.uv)[2];
        }

        let mut iridescence_thickness = material.iridescence_thickness;
        if let Some(map) = &material.iridescence_thickness_map {
            iridescence_thickness *= map.sample(info.uv)[1];
        }

        Surface {
            base_color,
            roughness: roughness.clamp(0.0, 1.0),
            metallic: metallic.clamp(0.0, 1.0),
            transmission: material.transmission,
            ior: material.ior,
            clearcoat: material.clearcoat,
            clearcoat_roughness: material.clearcoat_roughness,
            specular_tint: material.specular_tint,
            anisotropy: material.anisotropy,
            anisotropy_rotation: material.anisotropy_rotation,
            sheen: material.sheen,
            sheen_color: material.sheen_color,
            sheen_roughness: material.sheen_roughness,
            iridescence: material.iridescence,
            iridescence_ior: material.iridescence_ior,
            iridescence_thickness,
            dispersion: material.dispersion,
            subsurface: material.subsurface,
            subsurface_radius: material.subsurface_radius,
        }
    }

    /// Bind a surface to its shading frame. `info.ns` is already normal-mapped
    /// by the time this runs.
    fn make_bsdf(&self, info: &HitInfo, surface: Surface) -> Bsdf {
        Bsdf::new(surface, info.ns, info.ng)
    }

    /// Apply a tangent-space normal map.
    ///
    /// The tangent frame is derived from the triangle's own UV gradient rather
    /// than from a stored `tangent` attribute, so a normal map works on any
    /// geometry with UVs — including one built by a `*Geometry` constructor,
    /// which does not compute tangents.
    fn perturb_normal(
        &self,
        info: &HitInfo,
        map: &super::texture::CpuTexture,
        scale: Vector2,
    ) -> Vector3 {
        let idx = info.triangle;
        let tri = &self.scene.tris[idx];
        let sh = &self.scene.shading[idx];
        let duv1 = sh.uvs[1] - sh.uvs[0];
        let duv2 = sh.uvs[2] - sh.uvs[0];
        let det = duv1.x * duv2.y - duv2.x * duv1.y;
        if det.abs() < 1e-12 {
            return info.ns;
        }
        let e1 = tri[1] - tri[0];
        let e2 = tri[2] - tri[0];
        let r = 1.0 / det;
        let mut t = (e1 * duv2.y - e2 * duv1.y) * r;
        // Gram-Schmidt against the shading normal, so the frame stays
        // orthonormal after interpolation has tilted it.
        t = t - info.ns * info.ns.dot(t);
        if t.length_sq() < 1e-16 {
            return info.ns;
        }
        let t = t.normalize();
        let b = info.ns.cross(t);
        let s = map.sample(info.uv);
        let n = Vector3::new(
            (s[0] * 2.0 - 1.0) * scale.x,
            (s[1] * 2.0 - 1.0) * scale.y,
            s[2] * 2.0 - 1.0,
        );
        let world = t * n.x + b * n.y + info.ns * n.z;
        if world.length_sq() < 1e-16 {
            info.ns
        } else {
            world.normalize()
        }
    }

    /// Add a contribution, clamped according to how far along the path it is.
    #[inline]
    fn emit(&self, radiance: &mut Vector3, value: Vector3, bounce: u32) {
        let limit = if bounce == 0 {
            self.settings.clamp_direct
        } else {
            self.settings.clamp_indirect
        };
        let v = if limit > 0.0 {
            let m = value.x.max(value.y).max(value.z);
            if m > limit {
                value * (limit / m)
            } else {
                value
            }
        } else {
            value
        };
        if v.x.is_finite() && v.y.is_finite() && v.z.is_finite() {
            *radiance = *radiance + v;
        }
    }
}

/// The offset a secondary ray starts at, in world units.
///
/// Relative to the scene's own size, not absolute: a model measured in metres
/// and the same model measured in millimetres need the same *fraction* of their
/// extent, and a fixed value is simultaneously too small for one and too large
/// for the other. Clamping the scale up to 1 — which this used to do — meant a
/// centimetre-scale scene got an offset a hundredth of its own diameter, enough
/// to push shadow rays clean through thin geometry.
///
/// The floor is there only so a degenerate scene cannot produce a zero offset.
pub(crate) fn ray_epsilon(scene: &RaytraceScene, settings: &RaytraceSettings) -> f32 {
    (settings.ray_epsilon * scene.scale()).max(1e-9)
}

/// How much of the denoiser's guide a surface writes, what it writes, and what
/// it defers to whatever it reflects or transmits.
///
/// Returns `(weight, diffuse_albedo, deferred_tint)`. A rough surface writes
/// everything and defers nothing; a mirror or clear glass writes nothing and
/// defers its own tint, so the guide is taken from what it shows instead. The
/// thresholds are Cycles': a lobe stops counting as specular once its roughness
/// passes about 0.15, and a surface writes in proportion to how much of its
/// response is diffuse-like.
fn guide_split(s: &Surface) -> (f32, Vector3, Vector3) {
    let rough = smoothstep(0.0, 0.15, s.roughness.clamp(0.0, 1.0));
    let metallic = s.metallic.clamp(0.0, 1.0);
    let transmission = s.transmission.clamp(0.0, 1.0);

    let w_diffuse = (1.0 - metallic) * (1.0 - transmission);
    let w_sharp = metallic + (1.0 - metallic) * transmission;
    let total = (w_diffuse + w_sharp).max(1e-6);

    // The share of the response that is rough enough to be worth describing.
    let describable = (w_diffuse + w_sharp * rough) / total;
    let weight = smoothstep(0.0, 0.5, describable);
    let diffuse_albedo = s.base_color * describable;
    let deferred = s.base_color * ((w_sharp / total) * (1.0 - weight));
    (weight, diffuse_albedo, deferred)
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn average(v: Vector3) -> f32 {
    (v.x + v.y + v.z) / 3.0
}

/// Compress an unbounded radiance into `[0, 1)`, preserving order.
///
/// The albedo guide is a reflectance-like quantity, and a denoiser compares
/// guide values to decide whether two pixels show the same thing. Feeding it an
/// HDRI's sun at 12000 makes every neighbour look infinitely different and the
/// filter does nothing at all in the sky.
fn reinhard(v: Vector3) -> Vector3 {
    Vector3::new(
        v.x / (1.0 + v.x.max(0.0)),
        v.y / (1.0 + v.y.max(0.0)),
        v.z / (1.0 + v.z.max(0.0)),
    )
}

/// Extinction coefficient from a material's attenuation colour and distance:
/// `σ = -ln(colour) / distance`, so that a ray travelling exactly `distance`
/// through the medium is attenuated to exactly `colour`.
fn medium_extinction(material: &super::scene::RtMaterial) -> Option<Vector3> {
    if material.attenuation_distance <= 0.0 {
        return None;
    }
    let c = material.attenuation_color;
    let f = |v: f32| -v.clamp(1e-4, 1.0).ln() / material.attenuation_distance;
    let sigma = Vector3::new(f(c.x), f(c.y), f(c.z));
    (sigma.length_sq() > 0.0).then_some(sigma)
}

fn exp3(v: Vector3) -> Vector3 {
    Vector3::new(v.x.exp(), v.y.exp(), v.z.exp())
}

/// Replace non-finite components with zero. A single NaN in a film is
/// permanent — it propagates through the mean, the denoiser and the tone map —
/// so it is stopped at the one place it can enter.
fn sanitise(v: Vector3) -> Vector3 {
    Vector3::new(
        if v.x.is_finite() { v.x.max(0.0) } else { 0.0 },
        if v.y.is_finite() { v.y.max(0.0) } else { 0.0 },
        if v.z.is_finite() { v.z.max(0.0) } else { 0.0 },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Object3D;
    use crate::geometries::BoxGeometry;
    use crate::materials::{Material, StandardMaterial};
    use crate::math::Color;
    use crate::raytrace::settings::RaytraceSettings;
    use crate::scene::Scene;

    fn box_scene(size: f32) -> Scene {
        let mut scene = Scene::new();
        scene.add(Object3D::mesh(crate::core::Mesh::new(
            BoxGeometry::new(size, size, size),
            Material::Standard(StandardMaterial::new(Color::WHITE)),
        )));
        scene
    }

    /// The ray offset has to track the scene's size in *both* directions. A
    /// fixed floor of 1 world unit made a centimetre-scale model use an offset
    /// a hundredth of its own diameter.
    #[test]
    fn ray_epsilon_scales_with_the_scene() {
        let settings = RaytraceSettings::default();
        for size in [0.001f32, 0.05, 1.0, 100.0, 10_000.0] {
            let mut scene = box_scene(size);
            let rt = RaytraceScene::build(&mut scene, &settings);
            let eps = Integrator::new(&rt, &settings).epsilon();
            let scale = rt.scale();
            assert!(eps > 0.0, "size {size}: epsilon collapsed to {eps}");
            assert!(
                eps < scale * 1e-3,
                "size {size}: epsilon {eps} is {:.4}% of a scene {scale} across",
                100.0 * eps / scale
            );
        }
    }

    #[test]
    fn an_empty_scene_still_has_a_usable_epsilon() {
        let settings = RaytraceSettings::default();
        let mut scene = Scene::new();
        let rt = RaytraceScene::build(&mut scene, &settings);
        assert!(Integrator::new(&rt, &settings).epsilon() > 0.0);
    }
}
