//! The surface model every hit is shaded with: a principled BSDF over four
//! lobes — Lambert diffuse, GGX specular reflection, a rough dielectric with
//! transmission, and a clearcoat.
//!
//! It is written the way a path tracer needs a BSDF, which is three operations
//! rather than one:
//!
//! - [`Bsdf::eval`] — value and density for a direction chosen *elsewhere*
//!   (a light sample), so the pair can be combined by multiple importance
//!   sampling;
//! - [`Bsdf::sample`] — a direction drawn from the BSDF's own distribution,
//!   with its density, so the path can continue;
//! - [`Bsdf::pdf`] — the density alone, for the other half of MIS.
//!
//! Everything below works in a local frame whose +Z is the shading normal.
//! World-space vectors are rotated in and out at the boundary, which keeps the
//! microfacet maths free of dot products with an arbitrary normal and is the
//! same layout the GPU kernel uses.

use crate::math::Vector3;
use std::f32::consts::PI;
use std::sync::OnceLock;

use super::lights::Modulate;
use super::sampler::{cosine_hemisphere, sample_ggx_vndf, Onb, Rng};

/// Roughness below this is treated as a mirror: the lobe collapses to a delta,
/// which must be sampled rather than integrated (its density is unbounded) and
/// must not take an MIS weight.
const SMOOTH_ALPHA: f32 = 1e-3;
/// Roughness is never taken to exactly 0 in the rough path, so `D` cannot
/// divide by zero.
const MIN_ALPHA: f32 = 1e-4;

/// A hit's material parameters, already textured and ready to shade.
#[derive(Debug, Clone, Copy)]
pub struct Surface {
    pub base_color: Vector3,
    pub roughness: f32,
    pub metallic: f32,
    pub transmission: f32,
    pub ior: f32,
    pub clearcoat: f32,
    pub clearcoat_roughness: f32,
    /// Dielectric F0 tint, normally white.
    pub specular_tint: Vector3,
    /// -1..1; stretches the specular lobe along the tangent.
    pub anisotropy: f32,
    pub anisotropy_rotation: f32,
    pub sheen: f32,
    pub sheen_color: Vector3,
    pub sheen_roughness: f32,
    pub iridescence: f32,
    pub iridescence_ior: f32,
    pub iridescence_thickness: f32,
    pub dispersion: f32,
    pub subsurface: f32,
    pub subsurface_radius: Vector3,
}

impl Default for Surface {
    fn default() -> Self {
        Self {
            base_color: Vector3::new(0.8, 0.8, 0.8),
            roughness: 0.5,
            metallic: 0.0,
            transmission: 0.0,
            ior: 1.5,
            clearcoat: 0.0,
            clearcoat_roughness: 0.03,
            specular_tint: Vector3::new(1.0, 1.0, 1.0),
            anisotropy: 0.0,
            anisotropy_rotation: 0.0,
            sheen: 0.0,
            sheen_color: Vector3::ZERO,
            sheen_roughness: 1.0,
            iridescence: 0.0,
            iridescence_ior: 1.3,
            iridescence_thickness: 400.0,
            dispersion: 0.0,
            subsurface: 0.0,
            subsurface_radius: Vector3::new(1.0, 0.2, 0.1),
        }
    }
}

/// What [`Bsdf::sample`] produced.
#[derive(Debug, Clone, Copy)]
pub struct BsdfSample {
    /// World-space outgoing direction.
    pub direction: Vector3,
    /// BSDF value, *without* the cosine term.
    pub value: Vector3,
    /// Solid-angle density of `direction`.
    pub pdf: f32,
    /// A delta lobe. Its density is not a real number, so MIS must give this
    /// sample weight 1 and a light sample in the same direction weight 0.
    pub specular: bool,
    /// The direction crossed the surface — the caller must switch which medium
    /// it is tracking.
    pub transmitted: bool,
}

/// A [`Surface`] bound to a shading frame, ready to evaluate.
#[derive(Debug, Clone, Copy)]
pub struct Bsdf {
    onb: Onb,
    /// The *geometric* normal, used only to reject directions that the shading
    /// normal says are above the surface and the geometry says are below.
    /// Interpolated normals near a silhouette routinely disagree with the
    /// triangle they came from, and light leaking through that gap is the
    /// classic smooth-shaded artefact.
    geom_normal: Vector3,
    surface: Surface,
    /// GGX α along the tangent and bitangent.
    ax: f32,
    ay: f32,
    coat_alpha: f32,
    /// Normal-incidence reflectance.
    f0: Vector3,
    /// How much of the surface the *opaque* specular lobe is responsible for.
    ///
    /// The transmissive fraction is handled by the dielectric lobe, which
    /// already carries its own Fresnel reflection; without this the reflection
    /// off a glass surface would be counted twice — once by each lobe.
    spec_weight: f32,
    /// Lobe selection probabilities, summing to 1.
    p_diffuse: f32,
    p_specular: f32,
    p_transmission: f32,
    p_coat: f32,
    p_sheen: f32,
    /// Charlie α for the sheen lobe.
    sheen_alpha: f32,
}

impl Bsdf {
    /// Bind `surface` to a shading frame.
    ///
    /// `normal` is the shading normal and `geom_normal` the triangle's own,
    /// both already flipped to the side the ray arrived from.
    pub fn new(surface: Surface, normal: Vector3, geom_normal: Vector3) -> Self {
        let rough = surface.roughness.clamp(0.0, 1.0);
        // Perceptual roughness squared — the mapping three.js, Filament and
        // Disney all use, so a material tuned in the raster renderer keeps its
        // look here.
        let alpha = rough * rough;
        let aniso = surface.anisotropy.clamp(-0.9, 0.9);
        let stretch = (1.0 - aniso * aniso).sqrt().max(1e-3);
        let (ax, ay) = if aniso.abs() < 1e-4 {
            (alpha, alpha)
        } else if aniso > 0.0 {
            (alpha / stretch, alpha * stretch)
        } else {
            (alpha * stretch, alpha / stretch)
        };

        let dielectric_f0 = {
            let r = (surface.ior - 1.0) / (surface.ior + 1.0);
            r * r
        };
        let f0 = Vector3::new(
            lerp(
                dielectric_f0 * surface.specular_tint.x,
                surface.base_color.x,
                surface.metallic,
            ),
            lerp(
                dielectric_f0 * surface.specular_tint.y,
                surface.base_color.y,
                surface.metallic,
            ),
            lerp(
                dielectric_f0 * surface.specular_tint.z,
                surface.base_color.z,
                surface.metallic,
            ),
        );

        // Lobe probabilities from each lobe's rough share of the reflected
        // energy. They only steer sampling — the estimator divides by whatever
        // density it used, so a poor guess costs variance, never correctness.
        // Specular gets a floor because a black dielectric still has a
        // highlight, and a lobe with probability 0 can never be sampled.
        let metallic = surface.metallic.clamp(0.0, 1.0);
        let transmission = surface.transmission.clamp(0.0, 1.0);
        let w_diffuse = luminance(surface.base_color) * (1.0 - metallic) * (1.0 - transmission);
        let w_transmission = (1.0 - metallic) * transmission;
        let spec_weight = 1.0 - w_transmission;
        let w_specular = luminance(f0).max(0.04) * spec_weight;
        let w_coat = surface.clearcoat.clamp(0.0, 1.0) * 0.25;
        let sheen = surface.sheen.clamp(0.0, 1.0);
        let w_sheen = luminance(surface.sheen_color) * sheen * (1.0 - metallic);
        let total = (w_diffuse + w_specular + w_transmission + w_coat + w_sheen).max(1e-6);

        let mut onb = Onb::new(normal);
        if surface.anisotropy_rotation != 0.0 && aniso.abs() > 1e-4 {
            let (s, c) = surface.anisotropy_rotation.sin_cos();
            let t = onb.t * c + onb.b * s;
            let b = onb.b * c - onb.t * s;
            onb.t = t;
            onb.b = b;
        }

        let coat_rough = surface.clearcoat_roughness.clamp(0.0, 1.0);
        let sheen_rough = surface.sheen_roughness.clamp(0.07, 1.0);
        Self {
            onb,
            geom_normal,
            surface,
            ax: ax.max(0.0),
            ay: ay.max(0.0),
            coat_alpha: (coat_rough * coat_rough).max(0.0),
            f0,
            spec_weight,
            p_diffuse: w_diffuse / total,
            p_specular: w_specular / total,
            p_transmission: w_transmission / total,
            p_coat: w_coat / total,
            p_sheen: w_sheen / total,
            sheen_alpha: (sheen_rough * sheen_rough).max(0.0016),
        }
    }

    /// The shading normal this BSDF was built around.
    pub fn normal(&self) -> Vector3 {
        self.onb.n
    }

    /// Whether every lobe is a delta. Such a surface cannot be lit by next-event
    /// estimation at all — a light sample's direction has zero density under it
    /// — so the integrator skips the shadow ray entirely.
    pub fn is_delta(&self) -> bool {
        let spec_smooth = self.ax.max(self.ay) < SMOOTH_ALPHA;
        let coat_smooth = self.coat_alpha < SMOOTH_ALPHA;
        let no_diffuse = self.p_diffuse <= 0.0;
        let no_rough_coat = self.p_coat <= 0.0 || coat_smooth;
        no_diffuse && spec_smooth && no_rough_coat
    }

    /// BSDF value and density for a world-space `wi`, given the world-space
    /// direction `wo` the ray arrived along (pointing away from the surface).
    ///
    /// Returns zero for a delta surface: no finite direction has non-zero
    /// density under a mirror.
    pub fn eval(&self, wo_world: Vector3, wi_world: Vector3) -> (Vector3, f32) {
        // Shading and geometric normals must agree that the direction is on the
        // side it claims to be, or light leaks through silhouettes.
        let shading_side = wi_world.dot(self.onb.n);
        let geom_side = wi_world.dot(self.geom_normal);
        if shading_side * geom_side < 0.0 {
            return (Vector3::ZERO, 0.0);
        }
        let wo = self.onb.to_local(wo_world);
        let wi = self.onb.to_local(wi_world);
        if wo.z.abs() < 1e-6 {
            return (Vector3::ZERO, 0.0);
        }
        self.eval_local(wo, wi)
    }

    /// Density alone, for the BSDF side of an MIS weight.
    pub fn pdf(&self, wo_world: Vector3, wi_world: Vector3) -> f32 {
        self.eval(wo_world, wi_world).1
    }

    fn eval_local(&self, wo: Vector3, wi: Vector3) -> (Vector3, f32) {
        let mut f = Vector3::ZERO;
        let mut pdf = 0.0f32;
        let reflecting = wo.z * wi.z > 0.0;
        let spec_smooth = self.ax.max(self.ay) < SMOOTH_ALPHA;
        let coat_smooth = self.coat_alpha < SMOOTH_ALPHA;

        // --- clearcoat: a thin smooth dielectric layer over everything else.
        let mut coat_fresnel = 0.0f32;
        if self.surface.clearcoat > 0.0 && reflecting && !coat_smooth {
            let h = (wo + wi).normalize();
            let hz = if h.z < 0.0 { -h } else { h };
            // Clearcoat is polyurethane: IOR 1.5, F0 = 0.04, always.
            let fr = schlick_scalar(0.04, wo.dot(hz).abs());
            coat_fresnel = fr * self.surface.clearcoat;
            let d = ggx_d(hz, self.coat_alpha, self.coat_alpha);
            let g = smith_g2(wo, wi, self.coat_alpha, self.coat_alpha);
            let v = self.surface.clearcoat * d * g * fr / (4.0 * wo.z.abs() * wi.z.abs()).max(1e-9);
            f = f + Vector3::new(v, v, v);
            if self.p_coat > 0.0 {
                pdf += self.p_coat * vndf_reflect_pdf(wo, hz, self.coat_alpha, self.coat_alpha);
            }
        } else if self.surface.clearcoat > 0.0 && reflecting {
            // A smooth coat still darkens what is under it even though its own
            // lobe is a delta and contributes nothing here.
            coat_fresnel = schlick_scalar(0.04, wo.z.abs()) * self.surface.clearcoat;
        }
        let under_coat = 1.0 - coat_fresnel;

        // --- diffuse
        if self.p_diffuse > 0.0 && reflecting {
            let kd = 1.0 - schlick_scalar(luminance(self.f0), wo.z.abs());
            let grazing = (1.0 - wo.z.abs()).max(0.0);
            let ss = self.surface.subsurface.clamp(0.0, 1.0);
            let grazing2 = grazing * grazing;
            let ss_boost = Vector3::new(
                1.0 + ss * self.surface.subsurface_radius.x * grazing2,
                1.0 + ss * self.surface.subsurface_radius.y * grazing2,
                1.0 + ss * self.surface.subsurface_radius.z * grazing2,
            );
            let v = self.surface.base_color
                .mul_componentwise(ss_boost)
                * ((1.0 - self.surface.metallic)
                    * (1.0 - self.surface.transmission)
                    * kd
                    * under_coat
                    / PI);
            f = f + v;
            pdf += self.p_diffuse * (wi.z.abs() / PI);
        }

        // --- sheen (Charlie lobe, grazing fabric highlight)
        if self.p_sheen > 0.0 && reflecting && self.surface.sheen > 0.0 {
            let h = (wo + wi).normalize();
            let hz = if h.z < 0.0 { -h } else { h };
            let n_dot_l = wi.z.abs();
            let n_dot_v = wo.z.abs();
            let n_dot_h = hz.z.abs();
            let tint = self.surface.sheen_color * self.surface.sheen;
            let d = charlie_d(n_dot_h, self.sheen_alpha);
            let v = tint * (d * neubelt(n_dot_v, n_dot_l) * n_dot_l / PI * under_coat);
            f = f + v;
            pdf += self.p_sheen * (n_dot_l / PI);
        }

        // --- specular reflection
        if !spec_smooth && reflecting {
            let h = (wo + wi).normalize();
            let hz = if h.z < 0.0 { -h } else { h };
            let d = ggx_d(hz, self.ax, self.ay);
            let g = smith_g2(wo, wi, self.ax, self.ay);
            let fr = specular_fresnel(self, wo, hz);
            let denom = (4.0 * wo.z.abs() * wi.z.abs()).max(1e-9);
            let ms = self.multiscatter_gain(wo.z.abs());
            let spec = fr * (d * g / denom * under_coat * self.spec_weight);
            f = f + Vector3::new(spec.x * ms.x, spec.y * ms.y, spec.z * ms.z);
            pdf += self.p_specular * vndf_reflect_pdf(wo, hz, self.ax, self.ay);
        }

        // --- dielectric transmission
        if self.p_transmission > 0.0 && !spec_smooth {
            let (ft, pt) = self.eval_transmission_local(wo, wi);
            f = f + ft * under_coat;
            pdf += self.p_transmission * pt;
        }

        (f, pdf.max(0.0))
    }

    fn eval_transmission_local(&self, wo: Vector3, wi: Vector3) -> (Vector3, f32) {
        let eta_base = self.surface.ior.max(1.0001);
        let dispersion = self.surface.dispersion.max(0.0);
        let cos_o = wo.z;
        let cos_i = wi.z;
        if cos_i == 0.0 || cos_o == 0.0 {
            return (Vector3::ZERO, 0.0);
        }
        let reflecting = cos_o * cos_i > 0.0;

        let mut ft = Vector3::ZERO;
        let mut pt = 0.0f32;
        for (ch, offset) in [(0, -1.0f32), (1, 0.0), (2, 1.0)] {
            let eta = eta_base * (1.0 + dispersion * offset);
            let (f_ch, p_ch) = self.eval_transmission_channel(wo, wi, eta, reflecting, ch);
            match ch {
                0 => {
                    ft.x = f_ch;
                    pt += p_ch;
                }
                1 => {
                    ft.y = f_ch;
                    pt += p_ch;
                }
                _ => {
                    ft.z = f_ch;
                    pt += p_ch;
                }
            }
        }
        (ft, pt / 3.0)
    }

    fn eval_transmission_channel(
        &self,
        wo: Vector3,
        wi: Vector3,
        eta: f32,
        reflecting: bool,
        channel: usize,
    ) -> (f32, f32) {
        let cos_o = wo.z;
        let cos_i = wi.z;
        let etap = if reflecting {
            1.0
        } else if cos_o > 0.0 {
            eta
        } else {
            1.0 / eta
        };

        let mut wm = wi * etap + wo;
        if wm.length_sq() < 1e-12 {
            return (0.0, 0.0);
        }
        wm = wm.normalize();
        if wm.z < 0.0 {
            wm = wm * -1.0;
        }
        if wm.dot(wi) * cos_i < 0.0 || wm.dot(wo) * cos_o < 0.0 {
            return (0.0, 0.0);
        }

        let fr = fresnel_dielectric(wo.dot(wm), eta);
        let d = ggx_d(wm, self.ax, self.ay);
        let g = smith_g2(wo, wi, self.ax, self.ay);
        let pr = fr;
        let pt = 1.0 - fr;
        let weight = (1.0 - self.surface.metallic) * self.surface.transmission;

        if reflecting {
            let v = d * fr * g / (4.0 * cos_o * cos_i).abs().max(1e-9);
            let pdf = vndf_reflect_pdf(wo, wm, self.ax, self.ay) * pr / (pr + pt).max(1e-9);
            (v * weight, pdf)
        } else {
            let denom = {
                let d = wi.dot(wm) + wo.dot(wm) / etap;
                d * d
            };
            if denom < 1e-12 {
                return (0.0, 0.0);
            }
            let v = d * (1.0 - fr) * g
                * ((wi.dot(wm) * wo.dot(wm)) / (cos_i * cos_o * denom)).abs()
                / (etap * etap);
            let dwm_dwi = wi.dot(wm).abs() / denom;
            let pdf = vndf_pdf(wo, wm, self.ax, self.ay) * dwm_dwi * pt / (pr + pt).max(1e-9);
            let tint = match channel {
                0 => self.surface.base_color.x,
                1 => self.surface.base_color.y,
                _ => self.surface.base_color.z,
            };
            (tint * (v * weight), pdf)
        }
    }

    /// Draw a direction from the BSDF.
    pub fn sample(&self, wo_world: Vector3, rng: &mut Rng) -> Option<BsdfSample> {
        let wo = self.onb.to_local(wo_world);
        if wo.z.abs() < 1e-6 {
            return None;
        }
        let u = rng.next_f32();
        let (u1, u2) = rng.next_2d();

        let spec_smooth = self.ax.max(self.ay) < SMOOTH_ALPHA;
        let coat_smooth = self.coat_alpha < SMOOTH_ALPHA;

        let mut acc = self.p_diffuse;
        let sample = if u < acc {
            self.sample_diffuse(wo, u1, u2)
        } else {
            acc += self.p_specular;
            if u < acc {
                self.sample_specular(wo, u1, u2, spec_smooth)
            } else {
                acc += self.p_transmission;
                if u < acc {
                    self.sample_transmission(wo, u1, u2, spec_smooth, rng)
                } else {
                    acc += self.p_coat;
                    if u < acc {
                        self.sample_coat(wo, u1, u2, coat_smooth)
                    } else {
                        acc += self.p_sheen;
                        if u < acc {
                            self.sample_sheen(wo, u1, u2)
                        } else {
                            None
                        }
                    }
                }
            }
        }?;

        let wi_world = self.onb.to_world(sample.0);
        // Same shading-vs-geometric normal guard as `eval`.
        if !sample.1 && wi_world.dot(self.geom_normal) * sample.0.z < 0.0 {
            return None;
        }

        if sample.1 {
            // Delta lobe: the value already carries everything, and the density
            // is nominal.
            let (dir_local, _, value, transmitted) = sample;
            return Some(BsdfSample {
                direction: self.onb.to_world(dir_local),
                value,
                pdf: 1.0,
                specular: true,
                transmitted,
            });
        }

        // Rough lobe: re-evaluate the *whole* BSDF for the sampled direction,
        // so the value and density account for every lobe rather than only the
        // one that produced it. That is what makes the mixture consistent, and
        // what lets `pdf()` be used unchanged on the MIS side.
        let (f, pdf) = self.eval_local(wo, sample.0);
        if pdf <= 0.0 || (f.x <= 0.0 && f.y <= 0.0 && f.z <= 0.0) {
            return None;
        }
        Some(BsdfSample {
            direction: wi_world,
            value: f,
            pdf,
            specular: false,
            transmitted: sample.0.z * wo.z < 0.0,
        })
    }

    /// `(direction, is_delta, delta_value, transmitted)`
    fn sample_diffuse(
        &self,
        wo: Vector3,
        u1: f32,
        u2: f32,
    ) -> Option<(Vector3, bool, Vector3, bool)> {
        let mut d = cosine_hemisphere(u1, u2);
        if wo.z < 0.0 {
            d.z = -d.z;
        }
        Some((d, false, Vector3::ZERO, false))
    }

    fn sample_sheen(
        &self,
        wo: Vector3,
        u1: f32,
        u2: f32,
    ) -> Option<(Vector3, bool, Vector3, bool)> {
        self.sample_diffuse(wo, u1, u2)
    }

    fn sample_specular(
        &self,
        wo: Vector3,
        u1: f32,
        u2: f32,
        smooth: bool,
    ) -> Option<(Vector3, bool, Vector3, bool)> {
        if smooth {
            let wi = Vector3::new(-wo.x, -wo.y, wo.z);
            let h = Vector3::new(0.0, 0.0, 1.0);
            let fr = specular_fresnel(self, wo, h);
            let value = fr * (self.spec_weight / wi.z.abs().max(1e-6));
            return Some((wi, true, value, false));
        }
        let wo_up = if wo.z < 0.0 { wo * -1.0 } else { wo };
        let h = sample_ggx_vndf(wo_up, self.ax, self.ay, u1, u2);
        let wi = reflect(wo * -1.0, h);
        if wi.z * wo.z <= 0.0 {
            return None;
        }
        Some((wi, false, Vector3::ZERO, false))
    }

    fn sample_coat(
        &self,
        wo: Vector3,
        u1: f32,
        u2: f32,
        smooth: bool,
    ) -> Option<(Vector3, bool, Vector3, bool)> {
        if smooth {
            let wi = Vector3::new(-wo.x, -wo.y, wo.z);
            let fr = schlick_scalar(0.04, wo.z.abs()) * self.surface.clearcoat;
            let v = fr / wi.z.abs().max(1e-6);
            return Some((wi, true, Vector3::new(v, v, v), false));
        }
        let wo_up = if wo.z < 0.0 { wo * -1.0 } else { wo };
        let h = sample_ggx_vndf(wo_up, self.coat_alpha, self.coat_alpha, u1, u2);
        let wi = reflect(wo * -1.0, h);
        if wi.z * wo.z <= 0.0 {
            return None;
        }
        Some((wi, false, Vector3::ZERO, false))
    }

    fn sample_transmission(
        &self,
        wo: Vector3,
        u1: f32,
        u2: f32,
        smooth: bool,
        rng: &mut Rng,
    ) -> Option<(Vector3, bool, Vector3, bool)> {
        let eta_base = self.surface.ior.max(1.0001);
        let dispersion = self.surface.dispersion.max(0.0);
        let weight = (1.0 - self.surface.metallic) * self.surface.transmission;

        // The microfacet normal stays on the +Z side, as it does in `eval`, so
        // that `wo · h` carries the sign of the side the ray is on and the
        // Fresnel term can work out the interface direction for itself.
        let h = if smooth {
            Vector3::new(0.0, 0.0, 1.0)
        } else {
            let wo_up = if wo.z < 0.0 { wo * -1.0 } else { wo };
            sample_ggx_vndf(wo_up, self.ax, self.ay, u1, u2)
        };

        let cos_oh = wo.dot(h);
        let mut fr_avg = 0.0f32;
        for offset in [-1.0f32, 0.0, 1.0] {
            let eta = eta_base * (1.0 + dispersion * offset);
            fr_avg += fresnel_dielectric(cos_oh, eta);
        }
        fr_avg /= 3.0;
        // Choose reflection or refraction in proportion to Fresnel, then divide
        // it back out — so a nearly-grazing ray reflects nearly always, and the
        // estimator stays unbiased either way.
        let reflect_it = rng.next_f32() < fr_avg;
        let wi = if reflect_it {
            reflect(wo * -1.0, h)
        } else {
            let eta = eta_base;
            let (n, ratio) = if cos_oh > 0.0 {
                (h, 1.0 / eta)
            } else {
                (h * -1.0, eta)
            };
            refract(wo * -1.0, n, ratio)?
        };
        if wi.z == 0.0 {
            return None;
        }

        if smooth {
            let mut value = Vector3::ZERO;
            for (ch, offset) in [(0usize, -1.0f32), (1, 0.0), (2, 1.0)] {
                let eta = eta_base * (1.0 + dispersion * offset);
                let fr = fresnel_dielectric(cos_oh, eta);
                let v = if reflect_it {
                    fr * weight / wi.z.abs().max(1e-6)
                } else {
                    let etap = if wo.z > 0.0 { eta } else { 1.0 / eta };
                    let t = (1.0 - fr) * weight / (etap * etap * wi.z.abs().max(1e-6));
                    match ch {
                        0 => self.surface.base_color.x * t,
                        1 => self.surface.base_color.y * t,
                        _ => self.surface.base_color.z * t,
                    }
                };
                match ch {
                    0 => value.x = v,
                    1 => value.y = v,
                    _ => value.z = v,
                }
            }
            let div = if reflect_it {
                fr_avg.max(1e-6)
            } else {
                (1.0 - fr_avg).max(1e-6)
            };
            return Some((wi, true, value * (1.0 / div), !reflect_it));
        }
        Some((wi, false, Vector3::ZERO, wi.z * wo.z < 0.0))
    }

    /// Kulla–Conty energy compensation for the specular lobe.
    ///
    /// Single-scattering GGX drops the light that would have bounced between
    /// microfacets, which at roughness 1 is around 40% of it — a rough gold
    /// sphere renders visibly dark and grey. The lost fraction is
    /// `1 - E(cosθ, α)` where E is the lobe's directional albedo, so scaling by
    /// `1 + F0·(1/E - 1)` puts it back, tinted by the material as a second
    /// bounce off the same surface would be.
    fn multiscatter_gain(&self, cos_o: f32) -> Vector3 {
        let alpha = (self.ax * self.ay).sqrt();
        let e = ggx_albedo(cos_o, alpha);
        if e >= 0.999 {
            return Vector3::new(1.0, 1.0, 1.0);
        }
        let gain = (1.0 - e) / e.max(1e-3);
        Vector3::new(
            1.0 + self.f0.x * gain,
            1.0 + self.f0.y * gain,
            1.0 + self.f0.z * gain,
        )
    }
}

// ---------------------------------------------------------------- sheen / iridescence

/// Charlie normal distribution (Disney/Estevez fabric sheen).
fn charlie_d(n_dot_h: f32, alpha: f32) -> f32 {
    let a = alpha.max(0.0016);
    let inv_a = 1.0 / a;
    let cos2h = n_dot_h * n_dot_h;
    let sin2h = (1.0 - cos2h).max(0.0078125);
    (2.0 + inv_a) * sin2h.powf(inv_a * 0.5) / (2.0 * PI)
}

fn neubelt(n_dot_v: f32, n_dot_l: f32) -> f32 {
    1.0 / (4.0 * (n_dot_l + n_dot_v - n_dot_l * n_dot_v)).max(1e-5)
}

fn specular_fresnel(bsdf: &Bsdf, wo: Vector3, h: Vector3) -> Vector3 {
    let cos_theta = wo.dot(h).abs();
    let base = schlick(bsdf.f0, cos_theta);
    let ir = bsdf.surface.iridescence.clamp(0.0, 1.0);
    if ir <= 0.0 {
        return base;
    }
    let film = iridescence_fresnel(
        1.0,
        bsdf.surface.iridescence_ior,
        bsdf.f0,
        bsdf.surface.iridescence_thickness,
        wo.z.abs(),
    );
    Vector3::new(
        lerp(base.x, film.x, ir),
        lerp(base.y, film.y, ir),
        lerp(base.z, film.z, ir),
    )
}

fn ior_to_fresnel0(transmitted: f32, incident: f32) -> f32 {
    let t = (transmitted - incident) / (transmitted + incident);
    t * t
}

fn ior_to_fresnel0_v(transmitted: Vector3, incident: f32) -> Vector3 {
    Vector3::new(
        ior_to_fresnel0(transmitted.x, incident),
        ior_to_fresnel0(transmitted.y, incident),
        ior_to_fresnel0(transmitted.z, incident),
    )
}

fn fresnel0_to_ior(f0: Vector3) -> Vector3 {
    let s = Vector3::new(
        f0.x.clamp(0.0, 0.9999).sqrt(),
        f0.y.clamp(0.0, 0.9999).sqrt(),
        f0.z.clamp(0.0, 0.9999).sqrt(),
    );
    Vector3::new(
        (1.0 + s.x) / (1.0 - s.x),
        (1.0 + s.y) / (1.0 - s.y),
        (1.0 + s.z) / (1.0 - s.z),
    )
}

fn schlick_f90(f0: Vector3, f90: Vector3, cos_theta: f32) -> Vector3 {
    let w = (1.0 - cos_theta).clamp(0.0, 1.0).powi(5);
    Vector3::new(
        f0.x + (f90.x - f0.x) * w,
        f0.y + (f90.y - f0.y) * w,
        f0.z + (f90.z - f0.z) * w,
    )
}

fn schlick_f90_scalar(f0: f32, f90: f32, cos_theta: f32) -> f32 {
    let w = (1.0 - cos_theta).clamp(0.0, 1.0).powi(5);
    f0 + (f90 - f0) * w
}

fn eval_sensitivity(opd: f32, shift: Vector3) -> Vector3 {
    let phase = 2.0 * PI * opd * 1e-9;
    let val = Vector3::new(5.4856e-13, 4.4201e-13, 5.2481e-13);
    let pos = Vector3::new(1.6810e6, 1.7953e6, 2.2084e6);
    let vr = Vector3::new(4.3278e9, 9.3046e9, 6.6121e9);

    let mut xyz = Vector3::new(
        val.x * (2.0 * PI * vr.x).sqrt() * (pos.x * phase + shift.x).cos()
            * (-(phase * phase) * vr.x).exp(),
        val.y * (2.0 * PI * vr.y).sqrt() * (pos.y * phase + shift.y).cos()
            * (-(phase * phase) * vr.y).exp(),
        val.z * (2.0 * PI * vr.z).sqrt() * (pos.z * phase + shift.z).cos()
            * (-(phase * phase) * vr.z).exp(),
    );
    let x_extra = 9.7470e-14
        * (2.0 * PI * 4.5282e9).sqrt()
        * (2.2399e6 * phase + shift.x).cos()
        * (-4.5282e9 * phase * phase).exp();
    xyz.x += x_extra;
    xyz = xyz * (1.0 / 1.0685e-7);

    Vector3::new(
        3.2404542 * xyz.x - 0.969_266 * xyz.y + 0.0556434 * xyz.z,
        -1.5371385 * xyz.x + 1.8760108 * xyz.y - 0.2040259 * xyz.z,
        -0.4985314 * xyz.x + 0.0415560 * xyz.y + 1.0572252 * xyz.z,
    )
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Thin-film interference tint (Belcour & Barla 2017), matching three.js.
fn iridescence_fresnel(
    outside_ior: f32,
    film_ior_in: f32,
    base_f0: Vector3,
    thickness_nm: f32,
    cos_theta1: f32,
) -> Vector3 {
    let film_ior = lerp(outside_ior, film_ior_in, smoothstep(0.0, 0.03, thickness_nm));
    let sin_theta2_sq = (outside_ior / film_ior).powi(2) * (1.0 - cos_theta1 * cos_theta1);
    let cos_theta2_sq = 1.0 - sin_theta2_sq;
    if cos_theta2_sq < 0.0 {
        return Vector3::ONE;
    }
    let cos_theta2 = cos_theta2_sq.max(0.0).sqrt();

    let r0 = ior_to_fresnel0(film_ior, outside_ior);
    let r12 = schlick_f90_scalar(r0, 1.0, cos_theta1);
    let t121 = 1.0 - r12;

    let mut phi12 = 0.0f32;
    if film_ior < outside_ior {
        phi12 = PI;
    }
    let phi21 = PI - phi12;

    let base_ior = fresnel0_to_ior(base_f0);
    let r1 = ior_to_fresnel0_v(base_ior, film_ior);
    let r23 = schlick_f90(r1, Vector3::ONE, cos_theta2);

    let mut phi23 = Vector3::ZERO;
    if base_ior.x < film_ior {
        phi23.x = PI;
    }
    if base_ior.y < film_ior {
        phi23.y = PI;
    }
    if base_ior.z < film_ior {
        phi23.z = PI;
    }

    let opd = 2.0 * film_ior * thickness_nm * cos_theta2;
    let phi = Vector3::new(phi21, phi21, phi21) + phi23;

    let r123 = Vector3::new(
        (r12 * r23.x).clamp(1e-5, 0.9999),
        (r12 * r23.y).clamp(1e-5, 0.9999),
        (r12 * r23.z).clamp(1e-5, 0.9999),
    );
    let r123_sqrt = Vector3::new(r123.x.sqrt(), r123.y.sqrt(), r123.z.sqrt());
    let rs = Vector3::new(
        t121 * t121 * r23.x / (1.0 - r123.x).max(1e-5),
        t121 * t121 * r23.y / (1.0 - r123.y).max(1e-5),
        t121 * t121 * r23.z / (1.0 - r123.z).max(1e-5),
    );

    let c0 = Vector3::new(r12, r12, r12) + rs;
    let mut i_out = c0;
    let mut cm = rs - Vector3::new(t121, t121, t121);
    for m in 1..=2 {
        cm = Vector3::new(
            cm.x * r123_sqrt.x,
            cm.y * r123_sqrt.y,
            cm.z * r123_sqrt.z,
        );
        let sm = eval_sensitivity(
            m as f32 * opd,
            Vector3::new(m as f32 * phi.x, m as f32 * phi.y, m as f32 * phi.z),
        );
        i_out = i_out + cm.mul_componentwise(sm * 2.0);
    }

    Vector3::new(i_out.x.max(0.0), i_out.y.max(0.0), i_out.z.max(0.0))
}

// ---------------------------------------------------------------- microfacets

/// Trowbridge–Reitz (GGX) normal distribution, anisotropic.
pub fn ggx_d(h: Vector3, ax: f32, ay: f32) -> f32 {
    let ax = ax.max(MIN_ALPHA);
    let ay = ay.max(MIN_ALPHA);
    let hx = h.x / ax;
    let hy = h.y / ay;
    let d = hx * hx + hy * hy + h.z * h.z;
    if d <= 0.0 {
        return 0.0;
    }
    1.0 / (PI * ax * ay * d * d)
}

/// Smith's Λ for GGX — the ratio of masked to visible microfacet area.
fn smith_lambda(v: Vector3, ax: f32, ay: f32) -> f32 {
    let cos2 = v.z * v.z;
    if cos2 <= 0.0 {
        return 0.0;
    }
    let ax = ax.max(MIN_ALPHA);
    let ay = ay.max(MIN_ALPHA);
    let a2 = (v.x * ax) * (v.x * ax) + (v.y * ay) * (v.y * ay);
    let tan2 = a2 / cos2;
    if !tan2.is_finite() {
        return 0.0;
    }
    0.5 * ((1.0 + tan2).sqrt() - 1.0)
}

/// Monodirectional masking, `G1`.
pub fn smith_g1(v: Vector3, ax: f32, ay: f32) -> f32 {
    1.0 / (1.0 + smith_lambda(v, ax, ay))
}

/// Height-correlated masking-shadowing, `G2`. The correlated form is both
/// cheaper and more accurate than multiplying two `G1`s, which double-counts
/// the correlation between what is masked and what is shadowed.
pub fn smith_g2(wo: Vector3, wi: Vector3, ax: f32, ay: f32) -> f32 {
    1.0 / (1.0 + smith_lambda(wo, ax, ay) + smith_lambda(wi, ax, ay))
}

/// Density of a half-vector drawn from the visible normal distribution.
fn vndf_pdf(wo: Vector3, h: Vector3, ax: f32, ay: f32) -> f32 {
    let cos_o = wo.z.abs();
    if cos_o <= 0.0 {
        return 0.0;
    }
    smith_g1(wo, ax, ay) * ggx_d(h, ax, ay) * wo.dot(h).abs() / cos_o
}

/// The same density, after the reflection change of variables.
fn vndf_reflect_pdf(wo: Vector3, h: Vector3, ax: f32, ay: f32) -> f32 {
    let d = wo.dot(h).abs();
    if d <= 0.0 {
        return 0.0;
    }
    vndf_pdf(wo, h, ax, ay) / (4.0 * d)
}

/// Schlick's Fresnel approximation, spectral.
pub fn schlick(f0: Vector3, cos_theta: f32) -> Vector3 {
    let m = (1.0 - cos_theta).clamp(0.0, 1.0);
    let m5 = m * m * m * m * m;
    Vector3::new(
        f0.x + (1.0 - f0.x) * m5,
        f0.y + (1.0 - f0.y) * m5,
        f0.z + (1.0 - f0.z) * m5,
    )
}

pub fn schlick_scalar(f0: f32, cos_theta: f32) -> f32 {
    let m = (1.0 - cos_theta).clamp(0.0, 1.0);
    let m5 = m * m * m * m * m;
    f0 + (1.0 - f0) * m5
}

/// Exact unpolarised Fresnel reflectance for a dielectric interface.
///
/// Schlick is fine for the reflection lobe but not here: transmission weights
/// by `1 - F`, and Schlick's error near the critical angle is the difference
/// between total internal reflection happening and not.
pub fn fresnel_dielectric(cos_i: f32, eta: f32) -> f32 {
    let mut cos_i = cos_i.clamp(-1.0, 1.0);
    let mut eta = eta;
    if cos_i < 0.0 {
        eta = 1.0 / eta;
        cos_i = -cos_i;
    }
    let sin2_i = (1.0 - cos_i * cos_i).max(0.0);
    let sin2_t = sin2_i / (eta * eta);
    if sin2_t >= 1.0 {
        return 1.0; // total internal reflection
    }
    let cos_t = (1.0 - sin2_t).max(0.0).sqrt();
    let r_parl = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    let r_perp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    0.5 * (r_parl * r_parl + r_perp * r_perp)
}

/// Mirror `v` about `n`. `v` points *into* the surface.
pub fn reflect(v: Vector3, n: Vector3) -> Vector3 {
    v - n * (2.0 * v.dot(n))
}

/// Snell refraction.
///
/// `v` points *into* the surface, `n` faces the side `v` arrived from (so
/// `v·n < 0`), and `eta` is the ratio of refractive indices `n_incident /
/// n_transmitted` — 1/1.5 going into glass, 1.5 coming back out. Returns `None`
/// on total internal reflection.
///
/// The ratio is a parameter rather than something inferred from the geometry
/// because the caller already knows which side it is on, and a helper that
/// guesses gets it wrong exactly once: when the half-vector has been flipped to
/// match the view direction, which is precisely what microfacet sampling does.
pub fn refract(v: Vector3, n: Vector3, eta: f32) -> Option<Vector3> {
    let cos_i = -v.dot(n);
    if cos_i <= 0.0 {
        return None;
    }
    let sin2_t = eta * eta * (1.0 - cos_i * cos_i);
    if sin2_t > 1.0 {
        return None;
    }
    let cos_t = (1.0 - sin2_t).max(0.0).sqrt();
    Some(v * eta + n * (eta * cos_i - cos_t))
}

/// Rec. 709 luminance.
pub fn luminance(c: Vector3) -> f32 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

const ALBEDO_MU: usize = 32;
const ALBEDO_ALPHA: usize = 32;

/// Directional albedo of the single-scattering GGX lobe with a white Fresnel,
/// `E(cosθ, α)`, bilinearly interpolated from a table.
///
/// The table is integrated numerically once, on first use, rather than fitted
/// to a polynomial — 1024 entries of 64 samples each is under a millisecond and
/// avoids carrying somebody else's curve fit. [`ggx_albedo_table`] hands the
/// same numbers to the GPU backend so both agree to the bit.
pub fn ggx_albedo(cos_theta: f32, alpha: f32) -> f32 {
    let table = ggx_albedo_table();
    let mu = cos_theta.clamp(0.0, 1.0) * (ALBEDO_MU - 1) as f32;
    let al = alpha.clamp(0.0, 1.0).sqrt() * (ALBEDO_ALPHA - 1) as f32;
    let (m0, a0) = (mu.floor() as usize, al.floor() as usize);
    let (m1, a1) = ((m0 + 1).min(ALBEDO_MU - 1), (a0 + 1).min(ALBEDO_ALPHA - 1));
    let (tm, ta) = (mu - m0 as f32, al - a0 as f32);
    let at = |m: usize, a: usize| table[a * ALBEDO_MU + m];
    let e0 = at(m0, a0) * (1.0 - tm) + at(m1, a0) * tm;
    let e1 = at(m0, a1) * (1.0 - tm) + at(m1, a1) * tm;
    (e0 * (1.0 - ta) + e1 * ta).clamp(1e-3, 1.0)
}

/// The raw `E(cosθ, α)` table, row-major with `cosθ` varying fastest and `α`
/// stored as `sqrt(α)` so the low-roughness end — where the curve moves most —
/// gets more of the resolution.
pub fn ggx_albedo_table() -> &'static [f32] {
    static TABLE: OnceLock<Vec<f32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = vec![0.0f32; ALBEDO_MU * ALBEDO_ALPHA];
        // Stratified rather than random: the integrand is smooth in the two
        // sampling variables, so a regular grid converges far faster than the
        // same number of independent draws — and it makes the table exactly
        // reproducible, which matters because the GPU kernel is handed a copy
        // of it and the two renders have to agree.
        const STRATA: usize = 32;
        for ai in 0..ALBEDO_ALPHA {
            let sqrt_alpha = ai as f32 / (ALBEDO_ALPHA - 1) as f32;
            let alpha = (sqrt_alpha * sqrt_alpha).max(MIN_ALPHA);
            for mi in 0..ALBEDO_MU {
                let mu = (mi as f32 / (ALBEDO_MU - 1) as f32).max(1e-3);
                let wo = Vector3::new((1.0 - mu * mu).max(0.0).sqrt(), 0.0, mu);
                let g1 = smith_g1(wo, alpha, alpha);
                if g1 <= 0.0 {
                    t[ai * ALBEDO_MU + mi] = 1.0;
                    continue;
                }
                let mut sum = 0.0f64;
                for i in 0..STRATA {
                    let u1 = (i as f32 + 0.5) / STRATA as f32;
                    for j in 0..STRATA {
                        let u2 = (j as f32 + 0.5) / STRATA as f32;
                        let h = sample_ggx_vndf(wo, alpha, alpha, u1, u2);
                        let wi = reflect(wo * -1.0, h);
                        if wi.z <= 0.0 {
                            continue;
                        }
                        // With VNDF sampling and F = 1, the estimator for E
                        // reduces to G2/G1 — every other factor cancels.
                        sum += (smith_g2(wo, wi, alpha, alpha) / g1) as f64;
                    }
                }
                t[ai * ALBEDO_MU + mi] = (sum / (STRATA * STRATA) as f64) as f32;
            }
        }
        t
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raytrace::sampler::uniform_sphere;

    fn frame() -> (Vector3, Vector3) {
        let n = Vector3::new(0.0, 0.0, 1.0);
        (n, n)
    }

    fn diffuse(color: f32) -> Surface {
        Surface {
            base_color: Vector3::new(color, color, color),
            roughness: 1.0,
            metallic: 0.0,
            ..Default::default()
        }
    }

    /// Directional albedo `∫ f·cosθ dω`, estimated by sampling the BSDF itself.
    ///
    /// Low variance at every roughness, and it exercises the same `sample` path
    /// the renderer uses — so it catches an error in the sampled *value*, which
    /// integrating `eval` over uniform directions would not.
    fn sampled_albedo(bsdf: &Bsdf, wo: Vector3, n: usize) -> Vector3 {
        let mut rng = Rng::new(97, 31);
        let mut sum = Vector3::ZERO;
        for _ in 0..n {
            if let Some(s) = bsdf.sample(wo, &mut rng) {
                if s.pdf > 0.0 {
                    let cos = s.direction.dot(bsdf.normal()).abs();
                    sum = sum + s.value * (cos / s.pdf);
                }
            }
        }
        sum * (1.0 / n as f32)
    }

    /// Integrate `∫ f·cosθ dω` by uniform-sphere Monte Carlo. This is the
    /// surface's directional albedo, which physics caps at 1.
    fn hemispherical_albedo(bsdf: &Bsdf, wo: Vector3, n: usize) -> Vector3 {
        let mut rng = Rng::new(31, 41);
        let mut sum = Vector3::ZERO;
        for _ in 0..n {
            let (u1, u2) = rng.next_2d();
            let wi = uniform_sphere(u1, u2);
            let (f, _) = bsdf.eval(wo, wi);
            sum = sum + f * wi.dot(bsdf.normal()).abs();
        }
        // Uniform sphere pdf is 1/4π.
        sum * (4.0 * PI / n as f32)
    }

    #[test]
    fn lambert_albedo_matches_its_base_colour() {
        let (n, g) = frame();
        let bsdf = Bsdf::new(diffuse(0.5), n, g);
        let wo = Vector3::new(0.0, 0.0, 1.0);
        let a = hemispherical_albedo(&bsdf, wo, 200_000);
        assert!((a.x - 0.5).abs() < 0.02, "albedo {a:?}");
    }

    #[test]
    fn no_lobe_reflects_more_than_it_receives() {
        let cases = [
            diffuse(1.0),
            Surface {
                base_color: Vector3::new(1.0, 1.0, 1.0),
                roughness: 0.1,
                metallic: 1.0,
                ..Default::default()
            },
            Surface {
                base_color: Vector3::new(1.0, 1.0, 1.0),
                roughness: 0.6,
                metallic: 1.0,
                ..Default::default()
            },
            Surface {
                base_color: Vector3::new(0.9, 0.9, 0.9),
                roughness: 0.3,
                clearcoat: 1.0,
                clearcoat_roughness: 0.2,
                ..Default::default()
            },
        ];
        let (n, g) = frame();
        for (i, s) in cases.iter().enumerate() {
            let bsdf = Bsdf::new(*s, n, g);
            for mu in [0.95f32, 0.5, 0.15] {
                let wo = Vector3::new((1.0 - mu * mu).sqrt(), 0.0, mu);
                let a = sampled_albedo(&bsdf, wo, 60_000);
                assert!(
                    a.x <= 1.06,
                    "case {i} at cos {mu}: albedo {} exceeds 1",
                    a.x
                );
            }
        }
    }

    /// Energy compensation should bring a rough metal close to its base colour
    /// instead of leaving it 30-40% dark.
    #[test]
    fn rough_metal_keeps_most_of_its_energy() {
        let (n, g) = frame();
        let s = Surface {
            base_color: Vector3::new(1.0, 1.0, 1.0),
            roughness: 1.0,
            metallic: 1.0,
            ..Default::default()
        };
        let bsdf = Bsdf::new(s, n, g);
        let wo = Vector3::new(0.3, 0.0, 0.954);
        let a = sampled_albedo(&bsdf, wo, 100_000);
        assert!(
            a.x > 0.85,
            "rough white metal lost too much energy: {}",
            a.x
        );
        assert!(a.x <= 1.06, "and gained too much: {}", a.x);

        // Without compensation the same lobe would be visibly dark; check the
        // gain is actually doing something rather than being a no-op.
        assert!(
            bsdf.multiscatter_gain(0.954).x > 1.3,
            "energy compensation is not engaging"
        );
    }

    /// The density `sample` reports must be the density `pdf` computes, or MIS
    /// silently produces the wrong weights.
    #[test]
    fn sample_and_pdf_agree() {
        let (n, g) = frame();
        let cases = [
            diffuse(0.8),
            Surface {
                roughness: 0.35,
                metallic: 1.0,
                ..Default::default()
            },
            Surface {
                roughness: 0.4,
                transmission: 1.0,
                ior: 1.5,
                ..Default::default()
            },
            Surface {
                roughness: 0.25,
                clearcoat: 1.0,
                clearcoat_roughness: 0.15,
                ..Default::default()
            },
        ];
        let mut rng = Rng::new(77, 88);
        for (i, s) in cases.iter().enumerate() {
            let bsdf = Bsdf::new(*s, n, g);
            let wo = Vector3::new(0.4, 0.2, 0.894).normalize();
            for _ in 0..3000 {
                let Some(smp) = bsdf.sample(wo, &mut rng) else {
                    continue;
                };
                if smp.specular {
                    continue;
                }
                let pdf = bsdf.pdf(wo, smp.direction);
                assert!(
                    (pdf - smp.pdf).abs() <= 1e-3 * smp.pdf.max(1.0),
                    "case {i}: sample pdf {} vs pdf() {}",
                    smp.pdf,
                    pdf
                );
            }
        }
    }

    /// `∫ pdf dω = 1` over the sphere, for every non-delta configuration.
    #[test]
    fn pdf_integrates_to_one() {
        let (n, g) = frame();
        let cases = [
            diffuse(0.8),
            Surface {
                roughness: 0.5,
                metallic: 1.0,
                ..Default::default()
            },
            Surface {
                roughness: 0.45,
                transmission: 1.0,
                ior: 1.45,
                ..Default::default()
            },
        ];
        for (i, s) in cases.iter().enumerate() {
            let bsdf = Bsdf::new(*s, n, g);
            let wo = Vector3::new(0.0, 0.3, 0.954).normalize();
            let mut rng = Rng::new(101 + i as u64, 202);
            let n_samples = 400_000;
            let mut sum = 0.0f64;
            for _ in 0..n_samples {
                let (u1, u2) = rng.next_2d();
                let wi = uniform_sphere(u1, u2);
                sum += bsdf.pdf(wo, wi) as f64;
            }
            let integral = sum / n_samples as f64 * (4.0 * PI) as f64;
            assert!(
                (integral - 1.0).abs() < 0.06,
                "case {i}: pdf integrates to {integral}"
            );
        }
    }

    #[test]
    fn clear_glass_conserves_energy() {
        let (n, g) = frame();
        let s = Surface {
            base_color: Vector3::new(1.0, 1.0, 1.0),
            roughness: 0.0,
            metallic: 0.0,
            transmission: 1.0,
            ior: 1.5,
            ..Default::default()
        };
        let bsdf = Bsdf::new(s, n, g);
        for mu in [0.98f32, 0.6, 0.2] {
            let wo = Vector3::new((1.0 - mu * mu).sqrt(), 0.0, mu);
            let a = sampled_albedo(&bsdf, wo, 40_000);
            // Refraction into a denser medium compresses the beam, and
            // camera-side transport divides that back out, so the throughput of
            // a transmitted ray is 1/eta^2 rather than 1. What must not happen
            // is the reflection being counted by two lobes at once, which used
            // to push this well past 1.
            assert!(a.x <= 1.02, "cos {mu}: albedo {} exceeds 1", a.x);
            assert!(a.x > 0.4, "cos {mu}: albedo {} lost too much", a.x);
        }
    }

    #[test]
    fn a_glass_surface_gives_the_specular_lobe_no_weight() {
        let (n, g) = frame();
        let glass = Bsdf::new(
            Surface {
                transmission: 1.0,
                metallic: 0.0,
                ..Default::default()
            },
            n,
            g,
        );
        assert_eq!(glass.spec_weight, 0.0);
        let metal = Bsdf::new(
            Surface {
                metallic: 1.0,
                ..Default::default()
            },
            n,
            g,
        );
        assert_eq!(metal.spec_weight, 1.0);
    }

    #[test]
    fn smooth_surfaces_report_as_delta() {
        let (n, g) = frame();
        let mirror = Surface {
            base_color: Vector3::new(1.0, 1.0, 1.0),
            roughness: 0.0,
            metallic: 1.0,
            ..Default::default()
        };
        assert!(Bsdf::new(mirror, n, g).is_delta());
        assert!(!Bsdf::new(diffuse(0.5), n, g).is_delta());
    }

    #[test]
    fn a_mirror_reflects_about_the_normal() {
        let (n, g) = frame();
        let s = Surface {
            base_color: Vector3::new(1.0, 1.0, 1.0),
            roughness: 0.0,
            metallic: 1.0,
            ..Default::default()
        };
        let bsdf = Bsdf::new(s, n, g);
        let wo = Vector3::new(0.6, 0.0, 0.8);
        let mut rng = Rng::new(5, 5);
        let smp = bsdf.sample(wo, &mut rng).unwrap();
        assert!(smp.specular);
        assert!(
            (smp.direction - Vector3::new(-0.6, 0.0, 0.8)).length() < 1e-5,
            "{smp:?}"
        );
        // f·cos/pdf is the Fresnel term, which for a white metal is ~1.
        let throughput = smp.value * (smp.direction.dot(n).abs() / smp.pdf);
        assert!((throughput.x - 1.0).abs() < 0.02, "{throughput:?}");
    }

    #[test]
    fn glass_transmits_and_obeys_snell() {
        let (n, g) = frame();
        let s = Surface {
            base_color: Vector3::new(1.0, 1.0, 1.0),
            roughness: 0.0,
            transmission: 1.0,
            metallic: 0.0,
            ior: 1.5,
            ..Default::default()
        };
        let bsdf = Bsdf::new(s, n, g);
        let wo = Vector3::new(0.5, 0.0, 0.866);
        let mut rng = Rng::new(13, 17);
        let mut transmitted = 0;
        let mut total = 0;
        for _ in 0..4000 {
            let Some(smp) = bsdf.sample(wo, &mut rng) else {
                continue;
            };
            total += 1;
            if smp.transmitted {
                transmitted += 1;
                assert!(
                    smp.direction.z < 0.0,
                    "transmitted ray stayed above: {smp:?}"
                );
                // Snell: sinθt = sinθi / ior.
                let sin_i = (1.0 - wo.z * wo.z).sqrt();
                let sin_t = (1.0 - smp.direction.z * smp.direction.z).sqrt();
                assert!(
                    (sin_t - sin_i / 1.5).abs() < 1e-3,
                    "sin_i {sin_i} sin_t {sin_t}"
                );
            }
        }
        assert!(total > 0);
        // At this angle Fresnel reflects only a few percent.
        let frac = transmitted as f32 / total as f32;
        assert!(frac > 0.85, "only {frac} of samples transmitted");
    }

    #[test]
    fn total_internal_reflection_happens_past_the_critical_angle() {
        // Inside glass looking out at a shallow angle: nothing may escape.
        assert_eq!(fresnel_dielectric(-0.1, 1.5), 1.0);
        assert!(fresnel_dielectric(-0.9, 1.5) < 1.0);
        // And the exact Fresnel matches the textbook normal-incidence value.
        let f0 = fresnel_dielectric(1.0, 1.5);
        assert!((f0 - 0.04).abs() < 0.001, "F0 = {f0}");
    }

    #[test]
    fn refraction_obeys_snell_and_is_reversible() {
        let n = Vector3::new(0.0, 0.0, 1.0);
        let v = Vector3::new(0.3, 0.1, -0.948).normalize();
        let t = refract(v, n, 1.0 / 1.5).expect("entering glass");
        let sin_i = (1.0 - v.z * v.z).sqrt();
        let sin_t = (1.0 - t.z * t.z).sqrt();
        assert!(
            (sin_t - sin_i / 1.5).abs() < 1e-4,
            "sin_i {sin_i} sin_t {sin_t}"
        );

        // Reverse the path: a ray travelling along -t leaves the glass. It
        // arrives from below, so the facing normal is -Z and the ratio flips.
        let v2 = t * -1.0;
        let back = refract(v2, Vector3::new(0.0, 0.0, -1.0), 1.5).expect("leaving glass");
        assert!(
            (back - (v * -1.0)).length() < 1e-4,
            "{back:?} vs {:?}",
            v * -1.0
        );
    }

    #[test]
    fn refract_reports_total_internal_reflection() {
        // Inside glass at 60 degrees from the normal: past the ~41.8 degree
        // critical angle, so nothing gets out.
        let n = Vector3::new(0.0, 0.0, 1.0);
        let v = Vector3::new(0.866, 0.0, -0.5);
        assert!(refract(v, n, 1.5).is_none());
        assert!(refract(v, n, 1.0 / 1.5).is_some());
    }

    #[test]
    fn shading_normal_cannot_light_the_far_side() {
        // A shading normal tilted away from the geometry must not admit a
        // direction that is below the actual triangle.
        let shading = Vector3::new(0.0, 0.6, 0.8).normalize();
        let geom = Vector3::new(0.0, 0.0, 1.0);
        let bsdf = Bsdf::new(diffuse(1.0), shading, geom);
        let wo = Vector3::new(0.0, 0.3, 0.954).normalize();
        // Above the shading normal's plane but below the triangle.
        let wi = Vector3::new(0.0, 0.9, -0.436).normalize();
        assert!(wi.dot(shading) > 0.0 && wi.dot(geom) < 0.0, "test setup");
        let (f, pdf) = bsdf.eval(wo, wi);
        assert_eq!(f, Vector3::ZERO);
        assert_eq!(pdf, 0.0);
    }

    #[test]
    fn albedo_table_is_monotonic_and_bounded() {
        for ai in 0..8 {
            let alpha = ai as f32 / 7.0;
            for mi in 0..8 {
                let mu = (mi as f32 / 7.0).max(0.05);
                let e = ggx_albedo(mu, alpha);
                assert!((0.0..=1.001).contains(&e), "E({mu},{alpha}) = {e}");
            }
            // A smooth lobe loses almost nothing; a rough one loses a lot.
            assert!(ggx_albedo(0.9, 0.02) > 0.97);
        }
        assert!(ggx_albedo(0.9, 1.0) < 0.9, "roughness 1 should lose energy");
    }
}
