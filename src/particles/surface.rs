//! The surfaces of lying snow and sand: what a [`SnowCover`](super::snow::SnowCover)
//! or [`SandBed`](super::sand::SandBed) is drawn with.
//!
//! Both are lit by every light in the scene and shadowed by it, and both
//! carry detail finer than their grid:
//!
//! - **Sand** after the shading of sand in *Journey* (John Edwards, GDC
//!   2013), each part of it put on a physical footing:
//!   - **Grains**: the normal tilted toward a random direction per grain
//!     (nlerp), so the surface is granular under a low sun;
//!   - **Ripples**: shallow ones across the wind on the flats and steep ones
//!     along the contours on slopes, blended by how level the surface is,
//!     in patches — a long gentle stoss side and a short steep lee;
//!   - **Diffuse**: Oren–Nayar's rough Lambert, with the **opposition
//!     surge** of a granular surface (Hapke): brighter looking down-sun,
//!     where every grain hides its own shadow. `stylize` blends toward
//!     Journey's high-contrast term, `4 (N ⊙ [1, 0.3, 1]) · L`, with its
//!     coloured shadows;
//!   - **Anisotropic specular**: a GGX lobe rougher across the ripples than
//!     along their crests — the sheen of wind-polished sand ("ocean
//!     specular") stretched along the wind, the ripples too small to see
//!     becoming the roughness they are — and a rim at grazing view;
//!   - **Glitter**: quartz facets, one per grain at its own tilt, flashing
//!     where one mirrors a light into the eye;
//!   - **Streaks**: sand blown up the windward faces, running downwind.
//! - **Snow** is bright and soft — light goes into it and comes out round the
//!   corner (a wrap term, bluer as it goes deeper), it scatters forward at
//!   grazing light (an ice specular, stretched along the wind on wind-packed
//!   snow), has a faint opposition hotspot, a fine relief of grains, and
//!   **glitters**.
//!
//! Detail fades out with distance before it would shimmer; past it, its
//! slopes become roughness.

use crate::materials::{Material, RenderState};
use crate::math::{Color, Vector3};
use crate::ShaderMaterial;

/// Which surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SurfaceKind {
    Snow,
    Sand,
}

/// A snow or sand surface's look.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceLook {
    pub kind: SurfaceKind,
    /// Its colour (linear).
    pub color: Color,
    /// Glints, 0 and up.
    pub glint: f32,
    /// The wind the ripples lie across and the sheen is stretched along
    /// (any length; only its direction in x and z counts).
    pub wind: Vector3,
    /// Sand: metres from ripple to ripple.
    pub ripple_wavelength: f32,
    /// Sand: how strongly they mark it, 0 to 1.
    pub ripple_strength: f32,
    /// How far a grain's normal tilts toward a random direction, 0 to 1.
    pub grain: f32,
    /// Roughness (GGX α) along the wind and across it, for the sheen.
    pub roughness: [f32; 2],
    /// 0: physical shading. 1: Journey's high-contrast diffuse, its shadows
    /// in `shadow_color`.
    pub stylize: f32,
    /// The colour of the shadowed side, when stylized (linear).
    pub shadow_color: Color,
    /// Sand blown up the windward faces, 0 for none: how strongly the
    /// streaks show. Animated by the time in the material's data.
    pub streaks: f32,
    /// The haze over it brighter toward the sun by `1 + fog_glow ·
    /// max(cos θ, 0)⁸` (θ from the first directional light): forward
    /// scattering in the air, matching [`DesertTime`](super::dunes::DesertTime)'s
    /// sky at the horizon. 0: the fog's colour as it is.
    pub fog_glow: f32,
    /// Blended where thin (snow's coverage): see-through as the vertex
    /// colour's alpha says.
    pub coverage: bool,
}

impl SurfaceLook {
    pub fn snow(color: Color) -> Self {
        Self {
            kind: SurfaceKind::Snow,
            color,
            glint: 1.0,
            wind: Vector3::new(1.0, 0.0, 0.0),
            ripple_wavelength: 0.0,
            ripple_strength: 0.0,
            grain: 0.0,
            roughness: [0.45, 0.3],
            stylize: 0.0,
            shadow_color: Color::new(0.2, 0.25, 0.35),
            streaks: 0.0,
            fog_glow: 0.0,
            coverage: true,
        }
    }

    pub fn sand(color: Color) -> Self {
        Self {
            kind: SurfaceKind::Sand,
            color,
            glint: 0.35,
            wind: Vector3::new(1.0, 0.0, 0.0),
            ripple_wavelength: 0.08,
            ripple_strength: 0.7,
            grain: 0.25,
            roughness: [0.55, 0.22],
            stylize: 0.0,
            shadow_color: Color::new(0.34, 0.22, 0.13),
            streaks: 0.0,
            fog_glow: 0.0,
            coverage: false,
        }
    }

    pub fn material(&self) -> Material {
        let w = Vector3::new(self.wind.x, 0.0, self.wind.z);
        let w = if w.length() > 1e-6 {
            w.normalize()
        } else {
            Vector3::new(1.0, 0.0, 0.0)
        };
        let mut state = RenderState::DEFAULT;
        if self.coverage {
            // Blended where thin, but hiding what is behind it where solid.
            state.depth_write = Some(true);
        }
        // The look compiled in, as for particles: snow or sand, and their
        // numbers, as constants.
        let consts = format!(
            "const S_SAND : bool = {sand};\n\
             const S_GLINT : f32 = {glint:?};\n\
             const S_LAMBDA : f32 = {lambda:?};\n\
             const S_RIPPLE : f32 = {ripple:?};\n\
             const S_WIND : vec2<f32> = vec2<f32>({wx:?}, {wz:?});\n\
             const S_ALBEDO : vec3<f32> = vec3<f32>({r:?}, {g:?}, {b:?});\n\
             const S_SHADOW : vec3<f32> = vec3<f32>({sr:?}, {sg:?}, {sb:?});\n\
             const S_GRAIN : f32 = {grain:?};\n\
             const S_ROUGH : vec2<f32> = vec2<f32>({ra:?}, {rb:?});\n\
             const S_STYLIZE : f32 = {stylize:?};\n\
             const S_STREAKS : f32 = {streaks:?};\n\
             const S_FOG_GLOW : f32 = {fog_glow:?};\n\
             const S_COVERAGE : bool = {coverage};\n",
            sand = self.kind == SurfaceKind::Sand,
            glint = self.glint.max(0.0),
            lambda = self.ripple_wavelength.max(1e-3),
            ripple = self.ripple_strength.clamp(0.0, 1.0),
            wx = w.x,
            wz = w.z,
            r = self.color.r,
            g = self.color.g,
            b = self.color.b,
            sr = self.shadow_color.r,
            sg = self.shadow_color.g,
            sb = self.shadow_color.b,
            grain = self.grain.clamp(0.0, 1.0),
            ra = self.roughness[0].clamp(0.02, 1.0),
            rb = self.roughness[1].clamp(0.02, 1.0),
            stylize = self.stylize.clamp(0.0, 1.0),
            streaks = self.streaks.max(0.0),
            fog_glow = self.fog_glow.max(0.0),
            coverage = self.coverage,
        );
        Material::Shader(ShaderMaterial {
            fragment: consts + SURFACE_WGSL,
            // data[0].x: the time, seconds, for the streaks.
            data: vec![[0.0; 4]],
            transparent: self.coverage,
            side: 2,
            render_state: state,
            fallback_color: self.color,
            ..Default::default()
        })
    }
}

/// `material`'s clock, for animated streaks: a copy with the time set.
pub(crate) fn with_time(material: &Material, time: f32) -> Option<Material> {
    if let Material::Shader(m) = material {
        let mut m = m.clone();
        m.data = vec![[time, 0.0, 0.0, 0.0]];
        return Some(Material::Shader(m));
    }
    None
}

pub(crate) const SURFACE_WGSL: &str = r#"
fn surface_hash2(p : vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(443.897, 441.423));
    q = q + vec2<f32>(dot(q, q.yx + 19.19));
    return fract((q.x + q.y) * q.x);
}

fn surface_noise(p : vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(surface_hash2(i), surface_hash2(i + vec2<f32>(1.0, 0.0)), u.x),
               mix(surface_hash2(i + vec2<f32>(0.0, 1.0)), surface_hash2(i + vec2<f32>(1.0, 1.0)), u.x), u.y);
}

// How much of a detail of wavelength `size` metres to draw where a pixel
// covers `footprint`: all of it at 8 pixels a wavelength, none by 3.
fn surface_band(size : f32, footprint : f32) -> f32 {
    return smoothstep(3.0, 8.0, size / max(footprint, 1e-6));
}

// A random direction per cell, roughly Gaussian (a sum of three uniforms
// per axis), for a grain's facet.
fn surface_grain(c : vec2<f32>) -> vec3<f32> {
    let g = vec3<f32>(
        surface_hash2(c) + surface_hash2(c + 1.3) + surface_hash2(c + 2.9) - 1.5,
        surface_hash2(c + 4.1) + surface_hash2(c + 5.7) + surface_hash2(c + 6.3) - 1.5,
        surface_hash2(c + 7.7) + surface_hash2(c + 8.5) + surface_hash2(c + 9.9) - 1.5);
    return normalize(g + vec3<f32>(0.0, 1e-3, 0.0));
}

// Glints: the surface divided into cells `cell` metres across, a facet in
// each, tilted at random (more steeply in `spread`) and present with
// probability `chance`. It flashes when it mirrors `l` into `v`.
fn surface_glints(xz : vec2<f32>, n : vec3<f32>, l : vec3<f32>, v : vec3<f32>, cell : f32, chance : f32, spread : f32) -> f32 {
    let c = floor(xz / cell);
    let r1 = surface_hash2(c);
    if (r1 > chance) {
        return 0.0;
    }
    // Only near the facet's centre: points of light, not squares.
    let centre = (c + vec2<f32>(surface_hash2(c + 7.1), surface_hash2(c + 3.3)) * 0.6 + 0.2) * cell;
    let d = length(xz - centre) / cell;
    if (d > 0.18) {
        return 0.0;
    }
    let tilt = vec3<f32>(surface_hash2(c + 1.7) - 0.5, 0.0, surface_hash2(c + 5.9) - 0.5) * spread;
    let facet = normalize(n + tilt);
    let h = normalize(l + v);
    return pow(max(dot(facet, h), 0.0), 2000.0) * (1.0 - d / 0.18);
}

// Anisotropic GGX: the specular of a surface rougher along `t` (α.x) than
// along `b` (α.y), with Smith's shadowing and Schlick's Fresnel at `f0`.
fn surface_aniso_spec(n : vec3<f32>, t : vec3<f32>, b : vec3<f32>, l : vec3<f32>, v : vec3<f32>, alpha : vec2<f32>, f0 : f32) -> f32 {
    let ndl = dot(n, l);
    let ndv = max(dot(n, v), 1e-3);
    if (ndl <= 0.0) {
        return 0.0;
    }
    let h = normalize(l + v);
    let th = dot(t, h) / alpha.x;
    let bh = dot(b, h) / alpha.y;
    let nh = dot(n, h);
    let k = th * th + bh * bh + nh * nh;
    let d = 1.0 / (3.14159265 * alpha.x * alpha.y * k * k);
    // Smith (height-correlated, anisotropic α along each direction).
    let al = sqrt(pow(dot(t, l) * alpha.x, 2.0) + pow(dot(b, l) * alpha.y, 2.0) + ndl * ndl);
    let av = sqrt(pow(dot(t, v) * alpha.x, 2.0) + pow(dot(b, v) * alpha.y, 2.0) + ndv * ndv);
    let vis = 0.5 / (ndl * av + ndv * al);
    let f = f0 + (1.0 - f0) * pow(1.0 - max(dot(l, h), 0.0), 5.0);
    return d * vis * f * ndl;
}

@fragment
fn fs_main(in : VsOut) -> @location(0) vec4<f32> {
    let sand = S_SAND;
    let v = normalize(frame.camera_position.xyz - in.world_pos);
    // A skirt (a vertical edge) faces the eye whichever side it sees: its
    // winding is not always its normal's. The surface itself keeps its
    // normal — at a silhouette, where a slope falling away from the eye
    // peeks past the crest in front, flipping it would turn it to the
    // ground and draw it black.
    var n = normalize(in.world_normal);
    if (dot(n, v) < 0.0 && abs(n.y) < 0.05) {
        n = -n;
    }
    let geo_n = n;
    let xz = in.world_pos.xz;
    // How much ground one pixel covers here, metres: its size at this
    // distance, stretched along the slope as the surface turns edge-on.
    // Every detail below is band-limited by it — faded out as its own
    // wavelength comes down toward a few pixels, where it would alias into
    // moiré — and what fades out of the normal goes into the roughness.
    let pixel = 2.0 * in.view_z / (frame.projection[1][1] * frame.viewport_size.y);
    let footprint = pixel / max(abs(dot(geo_n, v)), 0.15);
    // Glints: points of light a pixel or two across, from facets a few
    // millimetres apart — gone before each pixel holds many.
    let near = surface_band(0.05, footprint);
    let w = S_WIND;
    let w3 = vec3<f32>(w.x, 0.0, w.y);
    var albedo = S_ALBEDO;
    // How much unresolved ripple slope there is, for the roughness.
    var ripple_var = 0.0;
    if (sand) {
        // Ripples, in patches: where the wind has worked the surface.
        let worked = smoothstep(0.35, 0.7, surface_noise(xz * 0.11) * 0.7 + surface_noise(xz * 0.37) * 0.3);
        let amount = S_RIPPLE * mix(0.35, 1.0, worked);
        // Shallow ripples across the wind on the flats: a sawtooth, bent a
        // little — the windward 80% of the wavelength rising gently, the lee
        // 20% falling steeply.
        let bend = surface_noise(xz * 0.9) * 2.5 + surface_noise(xz * 3.1) * 0.6;
        let phase = fract(dot(xz, w) / S_LAMBDA + bend);
        // (Its corners rounded: a hard step's harmonics would alias where
        // the ripple itself is wide enough to draw.)
        let lee = smoothstep(0.76, 0.82, phase) * (1.0 - smoothstep(0.95, 1.0, phase));
        let shallow = mix(0.65, -2.5, lee);
        // Steep ripples along the contours on slopes: across the way down,
        // closer together.
        var down = vec2<f32>(geo_n.x, geo_n.z);
        let dl = length(down);
        down = select(w, down / max(dl, 1e-4), dl > 1e-3);
        let sphase = fract(dot(xz, down) / (S_LAMBDA * 0.6) + bend * 0.5);
        let steep = sin(sphase * 6.2831853) * 1.6;
        // Blended by how level the surface is (Journey's steepness, sharp).
        let level = pow(clamp(geo_n.y, 0.0, 1.0), 6.0);
        let slope_dir = normalize(mix(vec3<f32>(down.x, 0.0, down.y), w3, level));
        // Each kind drawn only where it is wide enough on screen; the rest
        // of its slope becomes roughness.
        let shown = mix(surface_band(S_LAMBDA * 0.6, footprint), surface_band(S_LAMBDA, footprint), level);
        let slope = mix(steep * surface_band(S_LAMBDA * 0.6, footprint), shallow * surface_band(S_LAMBDA, footprint), level);
        n = normalize(n - slope_dir * (slope * amount * 0.1));
        ripple_var = amount * 0.12 * (1.0 - shown);
        // Grains: a speckle of light and dark, and each its own facet.
        let gcell = floor(xz / 0.004);
        let fine = surface_band(1.0 / 180.0, footprint);
        let coarse = surface_band(1.0 / 47.0, footprint);
        let g = (surface_noise(xz * 180.0) - 0.5) * 0.6 * fine + (surface_noise(xz * 47.0) - 0.5) * 0.4 * coarse;
        albedo = albedo * (1.0 + 0.36 * g);
        n = normalize(mix(n, surface_grain(gcell), S_GRAIN * surface_band(0.004, footprint)));
        // Sand blown up the windward faces: streaks running downwind.
        if (S_STREAKS > 0.0) {
            let windward = clamp(-4.0 * dot(geo_n, w3), 0.0, 1.0);
            let along = dot(xz, w);
            let across = dot(xz, vec2<f32>(-w.y, w.x));
            let t = u_data.data[0].x;
            let s = surface_noise(vec2<f32>(along * 0.25 - t * 1.6, across * 3.0))
                * surface_noise(vec2<f32>(along * 0.9 - t * 2.4, across * 7.0));
            // Gone before the streamers would alias.
            let seen = surface_band(1.0 / 7.0, footprint);
            albedo = albedo * (1.0 + S_STREAKS * windward * seen * smoothstep(0.2, 0.6, s) * 0.35);
        }
    } else if (surface_band(1.0 / 60.0, footprint) > 0.0) {
        // Snow's grainy relief (none where too fine on screen to show).
        let e = 0.004;
        let s = surface_noise(xz * 60.0) + 0.5 * surface_noise(xz * 140.0);
        let sx = surface_noise((xz + vec2<f32>(e, 0.0)) * 60.0) + 0.5 * surface_noise((xz + vec2<f32>(e, 0.0)) * 140.0);
        let sz = surface_noise((xz + vec2<f32>(0.0, e)) * 60.0) + 0.5 * surface_noise((xz + vec2<f32>(0.0, e)) * 140.0);
        n = normalize(n - vec3<f32>(sx - s, 0.0, sz - s) * (1.2 * surface_band(1.0 / 60.0, footprint)));
    }
    // The sheen's frame: rough along the wind, smooth along the crests.
    var t = w3 - n * dot(w3, n);
    t = select(normalize(cross(n, vec3<f32>(0.0, 0.0, 1.0))), normalize(t), length(t) > 1e-3);
    let b = cross(n, t);
    let alpha = vec2<f32>(sqrt(S_ROUGH.x * S_ROUGH.x + ripple_var), S_ROUGH.y);

    if ((mesh.flags.z & FLAG_RECEIVE_SHADOW) != 0u) {
        compute_shadows(in.world_pos, geo_n);
    } else {
        for (var r = 0u; r < 16u; r = r + 1u) {
            shadow_cache[r] = 1.0;
        }
    }
    let RECIP_PI = 0.3183098861837907;
    // Skylight; stylized, the shadowed side takes the shadow colour.
    let ambient_albedo = mix(albedo, S_SHADOW * 2.0, S_STYLIZE * 0.6);
    var col = ambient_albedo * frame.ambient.rgb * RECIP_PI;
    for (var k = 0u; k < frame.light_counts.w; k = k + 1u) {
        if (k >= 8u) { break; }
        col = col + ambient_albedo * shade_hemi(n, frame.hemi_lights[k]) * RECIP_PI;
    }
    let glint = S_GLINT * near;
    for (var k = 0u; k < frame.light_counts.x; k = k + 1u) {
        if (k >= 8u) { break; }
        let l = -normalize(frame.dir_lights[k].direction.xyz);
        let light = frame.dir_lights[k].color.rgb * shadow_factor_dir(k);
        col = col + light * surface_light(n, t, b, alpha, l, v, albedo, sand, xz, glint);
    }
    for (var k = 0u; k < frame.light_counts.y; k = k + 1u) {
        if (k >= 8u) { break; }
        let pl = frame.point_lights[k];
        let to_light = pl.position.xyz - in.world_pos;
        let d = length(to_light);
        let att = punctual_attenuation(d, pl.params.x, pl.params.y) * shadow_factor_point(k);
        col = col + pl.color.rgb * att * surface_light(n, t, b, alpha, to_light / max(d, 1e-4), v, albedo, sand, xz, glint);
    }
    for (var k = 0u; k < frame.light_counts.z; k = k + 1u) {
        if (k >= 8u) { break; }
        let sl = frame.spot_lights[k];
        let to_light = sl.position.xyz - in.world_pos;
        let d = length(to_light);
        let l = to_light / max(d, 1e-4);
        let cos_angle = dot(-l, normalize(sl.direction.xyz));
        var cone = 0.0;
        if (cos_angle > sl.params.z) {
            cone = smoothstep(sl.params.z, sl.params.w, cos_angle);
        }
        let att = punctual_attenuation(d, sl.params.x, sl.params.y) * cone * shadow_factor_spot(k);
        col = col + sl.color.rgb * att * surface_light(n, t, b, alpha, l, v, albedo, sand, xz, glint);
    }
    var a = 1.0;
    if (S_COVERAGE) {
        a = in.vertex_color.a;
    }
    // The haze, brighter toward the sun as the sky's horizon is.
    var haze = frame.fog_color.rgb;
    if (S_FOG_GLOW > 0.0 && frame.light_counts.x > 0u) {
        let sd = -normalize(frame.dir_lights[0].direction.xyz);
        haze = haze * (1.0 + S_FOG_GLOW * pow(max(dot(-v, sd), 0.0), 8.0));
    }
    col = mix(col, haze, fog_factor(in.view_z));
    return vec4<f32>(framebuffer_encode(apply_tone_mapping(col)), a);
}

// One light's share (radiance per unit of the light's irradiance).
fn surface_light(n : vec3<f32>, t : vec3<f32>, b : vec3<f32>, alpha : vec2<f32>, l : vec3<f32>, v : vec3<f32>,
                 albedo : vec3<f32>, sand : bool, xz : vec2<f32>, glint : f32) -> vec3<f32> {
    let RECIP_PI = 0.3183098861837907;
    let ndl = dot(n, l);
    let ndv = max(dot(n, v), 0.0);
    // The phase angle, between the light and the eye, for the opposition
    // surge: Hapke's B(g) = B₀ / (1 + tan(g/2) / h).
    let g = acos(clamp(dot(l, v), -1.0, 1.0));
    if (sand) {
        // Oren–Nayar, σ = 0.5 rad: a rough surface of grains — in the form
        // without angles (s / t for sin·tan), which stays finite at grazing.
        let s2 = 0.25;
        let oa = 1.0 - 0.5 * s2 / (s2 + 0.33);
        let ob = 0.45 * s2 / (s2 + 0.09);
        let nl = max(ndl, 0.0);
        let st = dot(l, v) - nl * ndv;
        let tt = select(1.0, max(max(nl, ndv), 1e-3), st > 0.0);
        let on = nl * (oa + ob * st / tt);
        let surge = 1.0 + 0.6 / (1.0 + tan(g * 0.5) / 0.06);
        // Journey's high-contrast term, its normal flattened.
        let journey = clamp(4.0 * dot(n * vec3<f32>(1.0, 0.3, 1.0), l), 0.0, 1.0);
        let diffuse = mix(on * surge, journey, S_STYLIZE);
        // The sheen, stretched along the wind; a rim at grazing view; the
        // larger of the two, as Journey takes them.
        let sheen = surface_aniso_spec(n, t, b, l, v, alpha, 0.045) * 1.5;
        let rim = pow(1.0 - ndv, 8.0) * 0.04 * max(ndl + 0.2, 0.0);
        var sparkle = 0.0;
        if (glint > 0.0) {
            sparkle = surface_glints(xz, n, l, v, 0.012, 0.25, 0.9) * glint * 6.0;
        }
        return albedo * diffuse * RECIP_PI + vec3<f32>(max(sheen, rim) + sparkle);
    }
    // Snow: light into it and out round the edge, bluer for having gone
    // deeper; a faint opposition hotspot; ice's forward specular, stretched
    // along the wind on wind-packed snow.
    let wrap = max((ndl + 0.35) / 1.35, 0.0);
    let through = vec3<f32>(0.80, 0.90, 1.0) * (wrap - max(ndl, 0.0)) * 0.6;
    let surge = 1.0 + 0.25 / (1.0 + tan(g * 0.5) / 0.05);
    let ice = surface_aniso_spec(n, t, b, l, v, alpha, 0.02);
    var sparkle = 0.0;
    if (glint > 0.0) {
        sparkle = (surface_glints(xz, n, l, v, 0.02, 0.35, 0.6) + surface_glints(xz + 0.37, n, l, v, 0.007, 0.25, 0.8))
            * glint * 12.0;
    }
    return albedo * (max(ndl, 0.0) * surge + through) * RECIP_PI + vec3<f32>(ice + sparkle);
}
"#;
