//! The water and sky fragments, and the uniform block they share.
//!
//! Both are compiled against threers' standard preamble, so `VsOut`, `frame`,
//! `PI`, `d_ggx`, `g_smith` and `framebuffer_encode` are already in scope, and
//! `u_data` / `u_s0` are the [`ShaderMaterial`](threers::ShaderMaterial) user
//! bindings. Note that the preamble is literally everything in threers' shader
//! ahead of its first `@fragment`, so the built-in materials' own shading —
//! including its Preetham sky — is *not* reachable from here. Hence the copy.
//!
//! Slot indices appear exactly once each, in the `u_*` accessors at the top of
//! [`COMMON_WGSL`]. [`slots`] is the Rust side of that same contract; the two
//! have to be edited together.

use std::sync::Arc;

use threers::{Material, ShaderMaterial};

use crate::ocean_fft::Cascades;
use crate::preset::Preset;
use crate::terrain::SHORE_RADIUS;

/// Pack a preset into the shader's uniform slots.
///
/// This is the whole of "loading a preset": no pipeline is rebuilt, nothing is
/// recompiled — the values are just written, which is why switching one at
/// runtime is free.
///
/// Slot 11 (the disc centre) and slot 0's time are written per frame by
/// [`crate::world::World::update`].
pub fn slots(
    p: &Preset,
    significant_height: f32,
    footprint_scale: f32,
    state_extent: f32,
) -> Vec<[f32; 4]> {
    let sun = sun_direction(p.sun_elevation, p.sun_azimuth);
    let mut d = vec![[0.0f32; 4]; 16];
    d[0] = [0.0, p.foam_amount, p.clouds, 0.0];
    d[1] = [sun[0], sun[1], sun[2], p.sun_intensity];
    d[2] = [
        p.turbidity,
        p.rayleigh,
        p.mie_coefficient,
        p.mie_directional_g,
    ];
    d[3] = [
        p.absorption[0],
        p.absorption[1],
        p.absorption[2],
        p.absorption_scale,
    ];
    d[4] = [
        p.scatter_color[0],
        p.scatter_color[1],
        p.scatter_color[2],
        p.sss,
    ];
    d[5] = [
        p.foam_color[0],
        p.foam_color[1],
        p.foam_color[2],
        p.foam_threshold,
    ];
    d[6] = [
        p.sand_color[0],
        p.sand_color[1],
        p.sand_color[2],
        SHORE_RADIUS,
    ];
    d[7] = [p.sun_color[0], p.sun_color[1], p.sun_color[2], p.roughness];
    d[8] = [footprint_scale, p.haze_near, p.haze_far, p.foam_softness];
    d[9] = [
        p.shore_depth,
        p.sky_reflect,
        p.fresnel_f0,
        1.0 / significant_height.max(0.05),
    ];
    d[10] = [p.exposure, p.sparkle, 0.0, 0.0];
    d[12] = [p.foam_slope, p.foam_slope_softness, p.wind_dir, 0.0];
    d[13] = [
        p.overcast,
        state_extent,
        p.wake_strength,
        p.foam_persistence,
    ];
    d[14] = [p.refraction, p.ssr, 0.0, p.caustics];
    d[15] = [0.0, 0.0, 0.0, 0.0]; // water mask, written per frame
    d
}

/// Sun direction as a unit vector, three.js `Sky` convention.
pub fn sun_direction(elevation_deg: f32, azimuth_deg: f32) -> [f32; 3] {
    let phi = (90.0 - elevation_deg).to_radians();
    let theta = azimuth_deg.to_radians();
    [phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin()]
}

pub fn water_material(slots: Vec<[f32; 4]>, peak_wavelength: f32, cas: &Cascades) -> Material {
    let src = format!("{COMMON_WGSL}{WATER_BODY}").replace(
        "CASCADE_BODY_PLACEHOLDER",
        &crate::waves_gpu::cascade_wgsl(
            "u_tex1",
            "u_tex2",
            "u_tex3",
            "u_samp",
            peak_wavelength,
            cas,
        ),
    );
    ShaderMaterial::new(src)
        .with_data(slots)
        // Screen space: the surface needs the opaque scene behind it to refract,
        // and on screen to reflect. Double-sided: from below the waterline you
        // are looking at the underside of it.
        .with_screen_space(true)
        .with_side(2)
        .into()
}

/// The sea floor.
///
/// It used to be a `StandardMaterial`, which meant it was lit as though it were
/// dry — the capture the water refracts had no ocean in it, so the floor came
/// back implausibly bright and the water shader had to subtract the missing
/// water back out. Doing it here instead splits the two paths the way the physics
/// does: **this** shader attenuates the light coming *down* through the column
/// and focuses it into caustics, and the water attenuates what comes back *up* to
/// the eye. Neither has to know about the other's half.
pub fn seabed_material(
    slots: Vec<[f32; 4]>,
    peak_wavelength: f32,
    cascades: [Arc<wgpu::TextureView>; 3],
    cas: &Cascades,
) -> Material {
    let src = format!("{COMMON_WGSL}{SEABED_BODY}").replace(
        "CASCADE_BODY_PLACEHOLDER",
        &crate::waves_gpu::cascade_wgsl(
            "u_tex1",
            "u_tex2",
            "u_tex3",
            "u_samp",
            peak_wavelength,
            cas,
        ),
    );
    ShaderMaterial::new(src)
        .with_data(slots)
        // Slot 0 is left for the foam field the water uses, so the cascades sit
        // at the same indices in both and the shared code needs no variant.
        .with_textures(vec![
            cascades[0].clone(),
            cascades[0].clone(),
            cascades[1].clone(),
            cascades[2].clone(),
        ])
        .into()
}

const SEABED_BODY: &str = r#"
CASCADE_BODY_PLACEHOLDER

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let p = in.world_pos.xz;
    // Depth of the water standing over *this* point of the floor.
    let depth = max(-in.world_pos.y, 0.0);
    let dist = length(frame.camera_position.xyz - in.world_pos);

    // The bed is analytic, so its normal can be finer than the 4 m mesh that
    // carries it. Widen the difference with distance to stay band-limited.
    var n = seabed_normal(p, max(dist * 0.004, 1.2));

    // Sand ripples. Orbital motion under waves builds them perpendicular to the
    // direction of travel, in bands of a few tens of centimetres, and they fade
    // out in deep water where the orbits no longer reach the bed. Purely a
    // normal perturbation — the mesh knows nothing about them.
    // Only underwater, and only where the orbits still reach the bed: ripples are
    // built by wave motion, so dry sand above the waterline has none and deep
    // water below the wave base has none either.
    let ripple_reach = smoothstep(0.0, 0.6, depth) * (1.0 - smoothstep(6.0, 26.0, depth));
    if (ripple_reach > 0.01) {
        let f = 1.05;
        let dir = vec2<f32>(cos(u_wind_dir()), sin(u_wind_dir()));
        let along = dot(p, dir) * f + vnoise(p * 0.06) * 3.0;
        let amp = 0.22 * ripple_reach * (0.6 + 0.4 * vnoise(p * 0.13));
        // d/dp of amp*sin(along) gives the slope the ripples add.
        let slope = amp * f * cos(along);
        n = normalize(n + vec3<f32>(dir.x * slope, 0.0, dir.y * slope));
    }

    let sun_dir = u_sun_dir();
    let sun_col = u_sun_color() * u_sun_intensity();

    // Calibrated against the `HemisphereLight` + `DirectionalLight` pair this
    // replaced, so the dry part of the island looks the same as it always did.
    var sun = sun_col * max(dot(n, sun_dir), 0.0) * 0.85;
    // Sky fill, split above and below the horizon the way a hemisphere light is.
    var sky = mix(u_scatter() * 2.0, vec3<f32>(0.42, 0.48, 0.60), 0.5 + 0.5 * n.y)
            * sun_col * 0.45;

    if (depth > 0.01) {
        // Downwelling: this light has already come through `depth` of water.
        //
        // At a fraction of the nominal coefficient, deliberately. A single-pass
        // Beer-Lambert overestimates attenuation in a medium that scatters:
        // light reaches the bed from the whole sky hemisphere and after multiple
        // scattering events, not just straight down the one ray. Using the full
        // coefficient here on top of the water's own view-path term buries the
        // bed at depths where you would plainly see it.
        sun = sun * exp(-u_absorption() * depth * 0.45);
        sky = sky * exp(-u_absorption() * depth * 0.3);

        // Caustics. The surface overhead is a lens; where it converges, the same
        // light lands on less floor. This is the *right* place for it — the
        // pattern belongs to the floor's illumination, not to the water in front
        // of it, which is why it now shows when you look at the bed directly.
        if (u_caustics() > 0.001) {
            let s = sample_surface(p, max(depth * 0.12, 0.4));
            let det = max(1.0 - s.fold, 0.25);
            // The pattern blurs out with depth as the focal cone spreads.
            let reach = exp(-depth * 0.05);
            let focus = min(u_caustics() * reach * (1.0 / det - 1.0) * 1.7, 1.6);
            sun = sun * (1.0 + focus);
        }
    }

    // Wind ripples on the dry sand. The submerged bed has its own set above,
    // built by wave orbits; these are built by wind, so they run across it and
    // they only exist where there is no water standing on them. Normal-only, and
    // faded with distance so they stop before they alias.
    let dry = 1.0 - smoothstep(-0.2, 1.2, depth);
    let near = 1.0 - smoothstep(60.0, 300.0, dist);
    let wind_ripple = dry * near;
    if (wind_ripple > 0.01) {
        let dir = vec2<f32>(cos(u_wind_dir() + 1.2), sin(u_wind_dir() + 1.2));
        let f = 1.7;
        let along = dot(p, dir) * f + vnoise(p * 0.11) * 4.0;
        let amp = 0.05 * wind_ripple;
        let slope = amp * f * cos(along);
        n = normalize(n + vec3<f32>(dir.x * slope, 0.0, dir.y * slope));
    }

    // Albedo. Wet sand is darker than dry — the band just above the waterline is
    // the most obvious tell that a beach is a beach — and the bed silts and
    // greens as it deepens.
    var albedo = u_sand();
    let wet = 1.0 - smoothstep(0.0, 1.6, in.world_pos.y);
    albedo = albedo * mix(1.0, 0.62, wet);
    albedo = mix(albedo, albedo * vec3<f32>(0.55, 0.72, 0.62), smoothstep(4.0, 30.0, depth));

    // Sand is not one colour over a whole island. Two things move it: how steep
    // the ground is, and how high. Steep faces shed the loose pale top layer and
    // sit darker and greyer; the crest is the driest and palest thing on the
    // island. Without either, an analytic dome shades as a single flat tone and
    // reads as clay rather than as sand.
    let steep = smoothstep(0.72, 0.97, 1.0 - n.y);
    albedo = mix(albedo, albedo * vec3<f32>(0.78, 0.76, 0.72), steep * 0.85);
    let crest = smoothstep(2.0, 14.0, in.world_pos.y);
    albedo = albedo * (1.0 + 0.10 * crest);

    // Grain, so a hundred metres of bed is not one flat colour. Three scales:
    // drifts, coarse patches of shell and silt, and a fine tooth for a close-up.
    albedo = albedo * (0.86 + 0.16 * vnoise(p * 0.05)
                            + 0.13 * vnoise(p * 0.22)
                            + 0.07 * vnoise(p * 1.7));

    let col = albedo * (sun + sky + 0.06);
    return vec4<f32>(framebuffer_encode(soft_clip(apply_fog(col * u_exposure(), in.view_z))), 1.0);
}
"#;

pub fn sky_material(slots: Vec<[f32; 4]>) -> Material {
    ShaderMaterial::new(format!("{COMMON_WGSL}{SKY_BODY}"))
        .with_data(slots)
        .into()
}

pub const COMMON_WGSL: &str = r#"
// ---- uniform slots -------------------------------------------------------
// The only place a slot index appears. Mirrors `slots()` in shader.rs.
fn u_time()           -> f32       { return u_data.data[0].x; }
fn u_foam_amount()    -> f32       { return u_data.data[0].y; }
fn u_clouds()         -> f32       { return u_data.data[0].z; }
// Waterline overlay only: how far in front of the lens the surface is
// probed, and how wet the lens still is after coming back up.
fn u_probe()          -> f32       { return u_data.data[0].w; }
fn u_wetness()        -> f32       { return u_data.data[10].z; }
fn u_sun_dir()        -> vec3<f32> { return normalize(u_data.data[1].xyz); }
fn u_sun_intensity()  -> f32       { return u_data.data[1].w; }
fn u_turbidity()      -> f32       { return u_data.data[2].x; }
fn u_rayleigh()       -> f32       { return u_data.data[2].y; }
fn u_mie_coeff()      -> f32       { return u_data.data[2].z; }
fn u_mie_g()          -> f32       { return u_data.data[2].w; }
fn u_absorption()     -> vec3<f32> { return u_data.data[3].xyz * u_data.data[3].w; }
fn u_scatter()        -> vec3<f32> { return u_data.data[4].xyz; }
fn u_sss()            -> f32       { return u_data.data[4].w; }
fn u_foam_color()     -> vec3<f32> { return u_data.data[5].xyz; }
fn u_foam_threshold() -> f32       { return u_data.data[5].w; }
fn u_sand()           -> vec3<f32> { return u_data.data[6].xyz; }
fn u_shore_radius()   -> f32       { return u_data.data[6].w; }
fn u_sun_color()      -> vec3<f32> { return u_data.data[7].xyz; }
fn u_roughness()      -> f32       { return u_data.data[7].w; }
fn u_footprint()      -> f32       { return u_data.data[8].x; }
fn u_haze_near()      -> f32       { return u_data.data[8].y; }
fn u_haze_far()       -> f32       { return u_data.data[8].z; }
fn u_foam_softness()  -> f32       { return u_data.data[8].w; }
fn u_shore_depth()    -> f32       { return u_data.data[9].x; }
fn u_sky_reflect()    -> f32       { return u_data.data[9].y; }
fn u_fresnel_f0()     -> f32       { return u_data.data[9].z; }
fn u_inv_wave_height()-> f32       { return u_data.data[9].w; }
fn u_exposure()       -> f32       { return u_data.data[10].x; }
fn u_disc_centre()    -> vec2<f32> { return u_data.data[11].xy; }
fn u_foam_slope()     -> vec2<f32> { return u_data.data[12].xy; }
fn u_overcast()       -> f32       { return u_data.data[13].x; }
fn u_wind_dir()       -> f32       { return u_data.data[12].z; }
fn u_refraction()     -> f32       { return u_data.data[14].x; }
fn u_ssr_strength()   -> f32       { return u_data.data[14].y; }
// Signed metres of water over the camera: positive when submerged.
fn u_submersion()     -> f32       { return u_data.data[14].z; }
fn u_caustics()       -> f32       { return u_data.data[14].w; }
fn u_sparkle()        -> f32       { return u_data.data[10].y; }
// Water is cut away inside this world-space circle: xy centre, z radius,
// w feather. Radius 0 disables it.
fn u_mask()           -> vec4<f32> { return u_data.data[15]; }
// Side of the world square the persistent surface texture covers.
fn u_state_extent()   -> f32       { return u_data.data[13].y; }
fn u_wake_strength()  -> f32       { return u_data.data[13].z; }
fn u_foam_persist()   -> f32       { return u_data.data[13].w; }

// Persistent foam (r) and wake (g), written by the surface-state compute pass.
// World-anchored, so it is indexed by position rather than by screen or by UV.
// u_tex0 is the persistent foam field; u_tex1..3 are the cascades, bound into
// the shared sampling code by name.
fn surface_state(p: vec2<f32>) -> vec2<f32> {
    // Jitter the lookup by a couple of metres. The field is ~2 m per texel, and
    // reading it straight draws visible contours along the bilinear seams.
    let jitter = vec2<f32>(vnoise(p * 0.9), vnoise(p * 0.9 + 31.7)) - 0.5;
    // No bounds check: the field tiles, and the sampler repeats. Foam therefore
    // exists everywhere rather than stopping at the edge of a box the camera can
    // orbit outside of.
    let uv = (p + jitter * 4.0) / u_state_extent() + vec2<f32>(0.5);
    return textureSampleLevel(u_tex0, u_samp, uv, 0.0).rg;
}

// The captured opaque scene, at @group(3). Only real when the material asks for
// screen space; otherwise these are 1x1 placeholders.
fn screen_dims() -> vec2<f32> { return vec2<f32>(textureDimensions(ss_depth_tex)); }

// NDC depth back to metres along the view ray. Inverse of the renderer's own
// `view_z_to_depth_tex`, which is the function that wrote this buffer.
fn linear_depth(d: f32) -> f32 {
    let n = frame.viewport_size.z;
    let f = frame.viewport_size.w;
    return n * f / max(f - d * (f - n), 1e-6);
}

// March a ray through the captured depth buffer. Returns uv in xy and 1 in z on
// a hit. This is the whole of screen-space reflection: anything not on screen
// this frame cannot be reflected, which is why the sky is still the fallback.
fn ss_march(origin: vec3<f32>, dir: vec3<f32>, max_dist: f32) -> vec3<f32> {
    let dims = screen_dims();
    var prev_uv = vec2<f32>(0.0);
    var i = 0;
    loop {
        if (i >= 24) { break; }
        let t = (f32(i) + 1.0) / 24.0;
        // Quadratic spacing: fine near the surface where contacts matter, coarse
        // far away where a miss just falls back to sky.
        let p = origin + dir * (max_dist * t * t);
        let clip = frame.view_proj * vec4<f32>(p, 1.0);
        if (clip.w <= 0.0) { break; }
        let uv = clip.xy / clip.w * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) { break; }
        let ray_d = clip.z / clip.w;
        let scene_d = textureLoad(ss_depth_tex, vec2<i32>(uv * dims), 0);
        // Require a thin crossing, or a ray that dives far behind a silhouette
        // would "hit" whatever happened to be in front of it.
        if (scene_d < 0.9999 && ray_d > scene_d + 2e-5 && ray_d < scene_d + 0.02) {
            return vec3<f32>(mix(prev_uv, uv, 0.5), 1.0);
        }
        prev_uv = uv;
        i = i + 1;
    }
    return vec3<f32>(prev_uv, 0.0);
}

fn sat(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

// A shoulder, not a clip. Sea glitter and the sun's Mie glow both run into the
// thousands; clamping them paints a flat white hole where the brightest, most
// interesting part of the picture should be. Water and sky share this curve, so
// the horizon stays seamless — the reason it lives here rather than in either.
//
// Compressing each channel on its own would drive anything bright toward white,
// which costs a sunset exactly the colour it is there for. Compress the
// brightest channel and scale the other two with it: the hue survives.
fn soft_clip(c: vec3<f32>) -> vec3<f32> {
    let knee = 0.65;
    let m = max(max(c.r, c.g), c.b);
    if (m <= knee) { return c; }
    let over = m - knee;
    let squashed = knee + over / (1.0 + over / (1.0 - knee));
    return c * (squashed / m);
}

fn hash21(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(123.34, 456.21));
    q = q + vec2<f32>(dot(q, q + 45.32));
    return fract(q.x * q.y);
}

fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm2(p: vec2<f32>) -> f32 {
    var s = 0.0;
    var a = 0.5;
    var q = p;
    for (var i = 0; i < 4; i = i + 1) {
        s = s + a * vnoise(q);
        q = q * 2.03 + vec2<f32>(17.0, -9.0);
        a = a * 0.5;
    }
    return s;
}

// ---- volumetric clouds ---------------------------------------------------
// A raymarched slab, not a painted dome. The reason is reflection: the water
// reflects whatever the sky function returns, so clouds that lived only in the
// dome's shader would be missing from the sea — which is exactly where a low sun
// makes them most obvious. Marching them here means one implementation feeds
// both.
//
// It is deliberately cheap: eight steps through an 800 m slab, two octaves of
// value noise for density, and two taps toward the sun for self-shadowing.
// Clouds are low-frequency and read mostly as silhouette; spending more here
// buys detail that is not visible at two kilometres.

const CLOUD_BASE: f32 = 900.0;
const CLOUD_TOP: f32 = 1700.0;

/// Three octaves, and no more.
///
/// The general-purpose `fbm2` runs four, and its fourth has features about 100 m
/// across — finer than the march can step through the slab, so it does not add
/// detail, it adds aliasing. This is the same sum stopped where the sampling can
/// still follow it.
fn cloud_fbm(p: vec2<f32>) -> f32 {
    var s = 0.0;
    var a = 0.55;
    var q = p;
    for (var i = 0; i < 3; i = i + 1) {
        s = s + a * vnoise(q);
        q = q * 2.07 + vec2<f32>(11.0, -7.0);
        a = a * 0.5;
    }
    return s;
}

fn cloud_density(pos: vec3<f32>, cover: f32) -> f32 {
    // The deck drifts downwind.
    let wind = vec2<f32>(cos(u_wind_dir()), sin(u_wind_dir()));
    let t = u_time() * 0.004;
    let q = pos.xz * 0.0012 + wind * t * 3.0;
    var d = cloud_fbm(q);
    // Coverage as a threshold on the noise. The bounds are the noise's actual
    // range, not 0..1: four octaves of value noise land in roughly 0.15..0.85,
    // so thresholding against 1 - cover leaves a clear sky at every setting
    // below about 0.9 — which is what a cloud knob that does nothing looks like.
    d = d - (0.82 - 0.70 * cover);
    if (d <= 0.0) { return 0.0; }
    // A vertical profile, so the slab has a shape instead of two hard faces.
    let h = clamp((pos.y - CLOUD_BASE) / (CLOUD_TOP - CLOUD_BASE), 0.0, 1.0);
    // Back up to order 1, since the threshold above took most of it away.
    return d * smoothstep(0.0, 0.25, h) * (1.0 - smoothstep(0.5, 1.0, h)) * 4.5;
}

/// Scattered radiance in `rgb`, coverage along the ray in `w`.
fn cloud_layer(direction: vec3<f32>, sky: vec3<f32>) -> vec4<f32> {
    let cover = u_clouds();
    // Fade the deck out as the ray flattens. This is not cosmetic: the slab is
    // entered at `base / dir.y`, so a ray 1 degree above the horizon starts its
    // march 50 km out and steps 6 km at a time through noise whose features are
    // 800 m across. Every sample is then uncorrelated with the last, and what
    // that looks like is a picket fence painted across the sky. Real cloud decks
    // are lost in haze at that range anyway.
    let horizon = smoothstep(0.035, 0.25, direction.y);
    if (cover <= 0.001 || horizon <= 0.001) { return vec4<f32>(0.0); }

    // Entry and exit, from a camera taken to be at sea level: the deck is a
    // kilometre up and the viewer is metres, so the parallax of the real height
    // is not visible.
    let t0 = CLOUD_BASE / direction.y;
    let t1 = CLOUD_TOP / direction.y;
    let sun = u_sun_dir();
    let sun_col = u_sun_color() * u_sun_intensity();
    // Forward scattering: the edge of a cloud facing the sun lights up, which is
    // the whole reason a cloudscape reads as three-dimensional.
    let silver = 0.55 + 0.75 * pow(sat(dot(direction, sun)), 8.0);

    let steps = 10;
    let dt = (t1 - t0) / f32(steps);
    // Dither the march. Ten shells through a slab is not enough to resolve it,
    // and the failure mode of a regular march is *stacked arcs* — every pixel
    // sampling the same set of distances, so the shells themselves become the
    // picture. Offsetting each ray by a hash of its own direction turns that
    // structure into noise, which on something as low-frequency as a cloud is
    // invisible where the banding was not.
    let jitter = hash21(direction.xz * 331.0 + direction.y * 71.0);
    var transmittance = 1.0;
    var scattered = vec3<f32>(0.0);
    for (var i = 0; i < steps; i = i + 1) {
        if (transmittance < 0.02) { break; }
        let pos = direction * (t0 + (f32(i) + jitter) * dt);
        let d = cloud_density(pos, cover);
        if (d > 0.0) {
            // One tap up the sun direction: enough to darken a cloud's base and
            // leave its top lit, which is most of what shading one is.
            let shadow = cloud_density(pos + sun * 380.0, cover);
            let lit = exp(-shadow * 2.4);
            let col = sun_col * (0.30 + 1.25 * lit) * silver + sky * 0.6;
            // Extinction per metre. The slab is 800 m thick, so this is an
            // optical depth of about 3 straight up through a solid cloud and
            // more at a slant — which is why a deck thickens toward the horizon
            // rather than staying an even grey.
            let a = 1.0 - exp(-d * dt * 0.0038);
            scattered = scattered + col * a * transmittance;
            transmittance = transmittance * (1.0 - a);
        }
    }
    return vec4<f32>(scattered * horizon, (1.0 - transmittance) * horizon);
}

/// The sky with its cloud deck.
///
/// Kept separate from [`sky_radiance`] rather than folded into it: the water
/// evaluates the sky up to three times per pixel and only one of those — the
/// reflection — is worth marching a cloud through. The horizon blend uses the
/// clear sky, and agrees with this one there anyway, because the deck fades out
/// as the ray flattens.
fn sky_with_clouds(direction: vec3<f32>) -> vec3<f32> {
    let base = sky_radiance(direction);
    if (u_clouds() <= 0.001) { return base; }
    let c = cloud_layer(direction, base);
    return base * (1.0 - c.w) + c.rgb;
}

// Preetham atmospheric scattering, the same model threers' own MAT_SKY branch
// runs. Duplicated here on purpose: the water has to reflect the *same* sky the
// dome draws, and the only way to guarantee that is to evaluate it, not
// approximate it.
fn sky_radiance(direction: vec3<f32>) -> vec3<f32> {
    let sunDir    = u_sun_dir();
    let turbidity = u_turbidity();
    let rayleigh  = u_rayleigh();
    let mieCoeff  = u_mie_coeff();
    let mieDirG   = u_mie_g();

    let up_v = vec3<f32>(0.0, 1.0, 0.0);
    let E_CONST = 2.718281828459045;
    let CUTOFF_ANGLE = 1.6110731556870734;
    let STEEPNESS = 1.5;
    let EE = 1000.0;

    let vSunfade = 1.0 - clamp(1.0 - exp(sunDir.y / 450000.0), 0.0, 1.0);
    let rayleighCoefficient = rayleigh - (1.0 - vSunfade);
    let totalRayleigh = vec3<f32>(5.804542996261093e-6, 1.3562911419845635e-5, 3.0265902468824876e-5);
    let vBetaR = totalRayleigh * rayleighCoefficient;
    let MieConst = vec3<f32>(1.8399918514433978e14, 2.7798023919660528e14, 4.0790479543861094e14);
    let c_mie = 0.2 * turbidity * 1.0e-17;
    let vBetaM = (0.434 * c_mie * MieConst) * mieCoeff;

    let zenithAngleCos_sun = clamp(dot(sunDir, up_v), -1.0, 1.0);
    let vSunE = EE * max(0.0, 1.0 - pow(E_CONST, -((CUTOFF_ANGLE - acos(zenithAngleCos_sun)) / STEEPNESS)));

    // Optical path along the view ray, cut off at the horizon to dodge the
    // singularity in the analytic thickness.
    let zenithAngle = acos(max(0.0, dot(up_v, direction)));
    let inv = 1.0 / (cos(zenithAngle) + 0.15 * pow(max(0.001, 93.885 - (zenithAngle * 180.0 / PI)), -1.253));
    let Fex = exp(-(vBetaR * (8.4e3 * inv) + vBetaM * (1.25e3 * inv)));

    let cosTheta = dot(direction, sunDir);
    let rPhase = 0.05968310365946075 * (1.0 + pow(cosTheta * 0.5 + 0.5, 2.0));
    let g2 = mieDirG * mieDirG;
    let inv_mie = 1.0 / pow(max(0.0001, 1.0 - 2.0 * mieDirG * cosTheta + g2), 1.5);
    let mPhase = 0.07957747154594767 * ((1.0 - g2) * inv_mie);

    let denom = max(vBetaR + vBetaM, vec3<f32>(1.0e-12));
    let scatter_ratio = (vBetaR * rPhase + vBetaM * mPhase) / denom;
    var Lin = pow(max(vSunE * scatter_ratio * (vec3<f32>(1.0) - Fex), vec3<f32>(0.0)), vec3<f32>(1.5));
    let sun_horizon_t = clamp(pow(1.0 - dot(up_v, sunDir), 5.0), 0.0, 1.0);
    let Lin_mix = pow(max(vSunE * scatter_ratio * Fex, vec3<f32>(0.0)), vec3<f32>(0.5));
    Lin = Lin * mix(vec3<f32>(1.0), Lin_mix, sun_horizon_t);

    var L0 = vec3<f32>(0.1) * Fex;
    let sundisk = smoothstep(0.999956676946448, 0.999976676946448, cosTheta);
    L0 = L0 + (vSunE * 19000.0 * Fex) * sundisk;

    let texColor = (Lin + L0) * 0.04 + vec3<f32>(0.0, 0.0003, 0.00075);
    let clear = pow(max(texColor, vec3<f32>(0.0)), vec3<f32>(1.0 / (1.2 + 1.2 * vSunfade)));

    // Cloud cover, as a bulk desaturate-and-darken. Preetham models a clear
    // atmosphere: an overcast day is not a turbidity setting, and cranking
    // turbidity to fake one gives a *brighter* white sky, not a heavier one.
    let grey = vec3<f32>(dot(clear, vec3<f32>(0.30, 0.59, 0.11)));
    return mix(clear, grey * 0.78, u_overcast());
}
"#;

/// The sky dome. Same `sky_radiance` the water reflects, same shoulder — the
/// horizon is where a mismatch between the two would be most obvious.
const SKY_BODY: &str = r#"
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let dir = normalize(in.world_pos - frame.camera_position.xyz);
    return vec4<f32>(framebuffer_encode(soft_clip(sky_with_clouds(dir) * u_exposure())), 1.0);
}
"#;

const WATER_BODY: &str = r#"
CASCADE_BODY_PLACEHOLDER

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let cam = frame.camera_position.xyz;
    let to_cam = cam - in.world_pos;
    let dist = length(to_cam);
    let v = to_cam / max(dist, 1e-4);

    // Undisplaced sample coordinate: the vertex colour carries the disc-local
    // grid position, and slot 11 carries where the disc currently sits. The sum
    // is world space, which is the frame the wave field and the seabed live in.
    let p = in.vertex_color.xy + u_disc_centre();

    // Pixel footprint on the water. It grows with distance and blows up at
    // grazing angles, which is exactly where a fixed-detail ocean shimmers.
    let grazing = max(abs(v.y), 0.05);
    let fp = dist * u_footprint() / grazing;

    // Water masking: cut a hole in the surface, so a hull or a dock can sit in
    // the sea without the sea sitting inside it. Alpha rather than `discard` so
    // the edge can feather instead of stair-stepping.
    let mask = u_mask();
    var cutout = 1.0;
    if (mask.z > 0.0) {
        cutout = smoothstep(mask.z, mask.z + max(mask.w, 1e-3), distance(p, mask.xy));
        if (cutout <= 0.001) { discard; }
    }

    let w = sample_surface(p, fp);
    var n = w.normal;

    // Broken water is turbulent, not a mirror.
    //
    // The depth limit caps wave height in the shallows — a wave cannot be taller
    // than the water it stands in — but capping a height also flattens its
    // *slope*, and `tanh` does that to every wave in shallow water, not only the
    // ones that are breaking. The result was a beach shelf that shaded as
    // polished glass: a flat white plate sitting inside the surf line. Churning
    // the normal where the wave is breaking puts the roughness back exactly
    // where the real surface stops being smooth.
    let churn = smoothstep(0.4, 0.95, w.breaking);
    if (churn > 0.01) {
        let q = p * 1.1 + vec2<f32>(u_time() * 0.6, -u_time() * 0.45);
        let gx = vnoise(q + vec2<f32>(0.5, 0.0)) - vnoise(q - vec2<f32>(0.5, 0.0));
        let gz = vnoise(q + vec2<f32>(0.0, 0.5)) - vnoise(q - vec2<f32>(0.0, 0.5));
        n = normalize(n + vec3<f32>(-gx, 0.0, -gz) * churn * 1.1);
    }

    let sun_dir = u_sun_dir();
    let sun_col = u_sun_color() * u_sun_intensity();
    // Rougher water blurs its reflection; the capture is mipped for exactly this.
    let rough_lod = u_roughness() * 24.0;
    let scatter = u_scatter();
    let n_dot_v = max(dot(n, v), 1e-3);
    let n_dot_l = max(dot(n, sun_dir), 0.0);

    // ---- the body of the water ----
    // What is actually behind the surface, refracted. The opaque scene was
    // captured before this pass, so this is the real seabed, the real island and
    // the submerged half of anything floating — not a guess at their colour.
    let dims = screen_dims();
    let suv = in.clip_pos.xy / dims;
    let refr = refract(-v, n, 0.75);
    let analytic_depth = water_depth(p);

    // Aim the sample where the refracted ray would reach the floor. The march is
    // capped: a full Snell march at a grazing angle lands most of a screen away,
    // where it mostly misses and drags in whatever it happens to land on.
    let surf_dist = in.view_z;
    let march = min(analytic_depth / max(-refr.y, 0.15), 60.0);
    let hit_clip = frame.view_proj * vec4<f32>(in.world_pos + refr * march, 1.0);
    let ruv_bent = hit_clip.xy / hit_clip.w * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    let bent_d = textureLoad(ss_depth_tex,
        vec2<i32>(clamp(ruv_bent, vec2<f32>(0.0), vec2<f32>(1.0)) * dims), 0);

    // A sample that turns out to be in *front* of the water would drag foreground
    // objects down into it — the classic refraction artefact. Blend the offset
    // away rather than switching it off: a hard test draws a visible seam across
    // the surface wherever it flips.
    let edge = min(min(ruv_bent.x, 1.0 - ruv_bent.x), min(ruv_bent.y, 1.0 - ruv_bent.y));
    // Fade the offset out at grazing angles. A screen-space refraction offset is
    // a guess about geometry that is not on screen, and the guess gets worse the
    // flatter the view: the march is longest, it crosses the horizon, and it
    // lands on whatever happens to be there. It is also the regime where Fresnel
    // means you can barely see into the water anyway, so there is nothing to
    // lose — and leaving it on is what painted the far troughs cyan, by
    // refracting bright sky and letting Beer-Lambert eat the red out of it.
    let valid = smoothstep(0.0, 0.02, edge)
              * smoothstep(0.0, 2.0, linear_depth(bent_d) - surf_dist)
              * smoothstep(0.08, 0.35, n_dot_v);
    let ruv = mix(suv, ruv_bent, valid * u_refraction());

    let floor_d = textureLoad(ss_depth_tex, vec2<i32>(clamp(ruv, vec2<f32>(0.0), vec2<f32>(1.0)) * dims), 0);

    // Is there actually a floor behind this pixel? The sky dome is opaque and in
    // the capture too, and at 9 km it still lands just inside a naive
    // "nothing here" test — so a refracted ray aimed past the horizon samples the
    // *sky* and hands it back as seabed. Beer-Lambert then eats the red out of it
    // and paints the sea cyan. The cut is at ~950 m, which is where the seabed
    // mesh ends anyway: past that there is nothing down there to see.
    let has_floor = floor_d < 0.999;

    // Optical path through the column. The depth buffer measures along the view
    // *axis*, so scale by dist/view_z to get true length along the ray — the two
    // differ by a factor of five at the grazing angles most of this frame is.
    let slant = dist / max(surf_dist, 1e-3);
    var path = analytic_depth / max(-refr.y, 0.2);
    if (has_floor) {
        path = min(path, max(linear_depth(floor_d) - surf_dist, 0.0) * slant);
    }
    let thickness = path;
    let trans = exp(-u_absorption() * path);

    var body = scatter * sun_col;
    // Past a few metres of clear water nothing comes back through, so skip the
    // sample entirely. Depth is coherent across a tile, so the branch is free.
    if (has_floor && max(trans.r, max(trans.g, trans.b)) > 0.003) {
        // Water scatters, so what is behind it softens with distance through it —
        // but gently. At a quarter-mip per metre this reached mip 3 in twelve
        // metres of water and erased the sea floor's ripples, caustics and grain
        // before they could be seen through it.
        let lod = clamp(thickness * 0.06, 0.0, 3.0);
        var behind = textureSampleLevel(ss_color_tex, ss_color_samp, ruv, lod).rgb;
        // Where the offset sample is not trustworthy, fall back to the medium's
        // own colour rather than to whatever the ray happened to land on.
        behind = mix(scatter * sun_col, behind, valid);
        // No compensation for the missing water column here any more: the floor
        // attenuates its own downwelling light, so what the capture holds is
        // already what it should be.

        // Caustics are not applied here: they are the *floor's* illumination and
        // the floor now computes them, so adding them again would square them.
        body = behind * trans + scatter * sun_col * (vec3<f32>(1.0) - trans);
    }

    // Light that entered the back of a crest and came out the front.
    let crest = sat(w.disp.y * u_inv_wave_height());
    let through = pow(sat(dot(v, -sun_dir)), 4.0);
    body = body + scatter * sun_col * (through * crest * u_sss());

    // ---- reflection ----
    var r = reflect(-v, n);
    // A reflection ray aimed below the horizon would hit the sea it came from.
    r.y = abs(r.y) + 0.001;
    var sky = sky_with_clouds(normalize(r)) * u_sky_reflect();
    // Screen-space reflection over the top. Only what is on screen can reflect,
    // so the sky stays the fallback — but it is what puts the island in the
    // water instead of leaving a hole where its reflection should be.
    if (u_ssr_strength() > 0.001) {
        let hit = ss_march(in.world_pos, normalize(r), 900.0);
        if (hit.z > 0.5) {
            // Fade at the frame edge, where the reflection runs out of screen.
            let edge = min(min(hit.x, 1.0 - hit.x), min(hit.y, 1.0 - hit.y));
            let conf = smoothstep(0.0, 0.08, edge) * u_ssr_strength();
            let refl = textureSampleLevel(ss_color_tex, ss_color_samp, hit.xy,
                                          clamp(rough_lod, 0.0, 4.0)).rgb;
            sky = mix(sky, refl, conf);
        }
    }

    let f0 = u_fresnel_f0();
    var fres = f0 + (1.0 - f0) * pow(1.0 - n_dot_v, 5.0);
    // Thin water shows its bottom. Schlick against the macro normal sends
    // reflectance to 1 at a grazing angle, which is right for an ocean and wrong
    // for the last metre of a beach: there, the surface is broken by ripples
    // finer than the sampler can resolve, and the sand underneath is bright and
    // close. Left alone, the shallows became a mirror-bright plate with a hard
    // rim at the waterline instead of water you can see the beach through.
    fres = fres * mix(1.0, 0.35, 1.0 - smoothstep(0.0, 1.6, thickness));

    // ---- sun glint ----
    // Cook-Torrance: D·G·F / (4·n·v), the n·l of the reflectance equation having
    // already cancelled the one in the denominator.
    let rough = max(u_roughness(), 0.02);
    let a_r = rough * rough;
    let hv = normalize(v + sun_dir);
    var spec = sun_col * d_ggx(max(dot(n, hv), 0.0), a_r)
             * g_smith(n_dot_v, max(n_dot_l, 1e-3), a_r)
             * fres * 0.25 / n_dot_v
             * smoothstep(0.0, 0.03, dot(n, sun_dir));
    // Sparkle: real sea glitter is thousands of individual facets winking in and
    // out, which a smooth GGX lobe averages away into a soft wedge. Modulating
    // it with high-frequency noise puts the individual glints back.
    if (u_sparkle() > 0.001) {
        let tw = vnoise(p * 5.3 + vec2<f32>(u_time() * 0.7, -u_time() * 0.5));
        let tw2 = vnoise(p * 11.7 - vec2<f32>(u_time() * 1.1, u_time() * 0.9));
        // Only where the lobe is already lit, and fading out with distance as the
        // facets drop below a pixel.
        let grain = mix(1.0, smoothstep(0.35, 0.95, tw * 0.6 + tw2 * 0.4) * 2.4,
                        u_sparkle() * (1.0 - smoothstep(60.0, 700.0, dist)));
        spec = spec * grain;
    }
    // Bound the lobe before it reaches the shoulder, or the glitter path spreads
    // into a single blown-out wedge instead of thousands of separate facets.
    spec = spec / (vec3<f32>(1.0) + spec * 0.9);

    var color = mix(body, sky, fres) + spec;

    // ---- seen from below ----
    // Underwater the surface is a mirror everywhere outside Snell's window: past
    // the critical angle (48.6 deg for water) nothing gets in, and the ceiling
    // reflects the sea floor back at you. Inside the window you see the whole
    // sky, squeezed into that cone.
    if (u_submersion() > 0.0 && dot(n, v) < 0.0) {
        let up_n = -n;
        let cos_i = clamp(dot(up_n, v), 0.0, 1.0);
        let sin_i = sqrt(max(1.0 - cos_i * cos_i, 0.0));
        // sin(theta_c) = 1/1.333
        let window = 1.0 - smoothstep(0.70, 0.751, sin_i);
        let out_dir = refract(-v, up_n, 1.333);
        var above = sky_radiance(normalize(vec3<f32>(out_dir.x, abs(out_dir.y) + 0.001, out_dir.z)));
        // Beyond the critical angle `refract` returns zero; that is the total
        // internal reflection case, and the ceiling shows the water below it.
        let mirrored = scatter * sun_col * 1.4 + spec;
        color = mix(mirrored, above, window);
        // Looking up through the window, the sun is a bright disc that the
        // surface's own roughness smears out.
        color = color + spec * window;
    }

    // ---- foam ----
    // Whitecaps. Folding alone under-reports breaking for a broad spectrum: with
    // forty components at independent phases the Jacobian only dips below one at
    // the rare crest where most of them line up, so a 19 m/s sea comes out
    // glassy. Steepness is the other half of the criterion — a crest spills once
    // its face passes an angle, wind speed setting where that angle is.
    let slope = u_foam_slope();
    let cap = smoothstep(slope.x, slope.x + slope.y, 1.0 - n.y)
            * smoothstep(-0.1, 0.55, w.disp.y * u_inv_wave_height());
    let broken = max(smoothstep(u_foam_threshold(), u_foam_threshold() + max(u_foam_softness(), 1e-3), w.fold), cap);
    // Shoreline foam surges with the swell: the band tracks depth *plus* the
    // wave riding over it, so it runs up the beach and drains back.
    // Written as 1 - smoothstep rather than a reversed pair: WGSL leaves
    // smoothstep undefined when the low edge exceeds the high one, and on this
    // backend it silently returns zero — which is a shoreline with no surf.
    // Surf. `breaking` is the wave's height against what the depth can carry, so
    // this is the real criterion — a wave spilling because it has run out of
    // water under it — rather than a band painted on wherever the sea is shallow.
    let surf = smoothstep(0.55, 0.95, w.breaking);
    // Shallow water is not automatically white. Saturating on depth alone paints
    // a solid sheet over every sandbar and turns a wide gentle beach — which is
    // what this bed deliberately has — into one flat white apron. Kept well
    // below saturation, it is a *tendency* to foam that the break-up noise then
    // bands into streaks. What actually turns surf white is `breaking`: a wave
    // running out of water under it, rather than a depth reading.
    let shallow = 1.0 - smoothstep(0.0, u_shore_depth(), min(analytic_depth, thickness) + w.disp.y);
    let shore = max(surf, shallow * 0.32);

    // What broke earlier and has not dissolved yet, plus anything stirring the
    // water. Neither is a function of the current wave field, which is why they
    // come from a texture with memory rather than from the spectrum.
    let state = surface_state(p);
    let lingering = state.r * u_foam_persist() + state.g * u_wake_strength();

    if (broken > 0.001 || shore > 0.001 || lingering > 0.003) {
        // Two scales of break-up, and four octaves of noise, so this stays
        // behind the test above: most of the sea is not foaming.
        let drift = vec2<f32>(u_time() * 0.05, u_time() * 0.02);
        let brk = fbm2(p * 0.7 + drift) * 0.65 + vnoise(p * 3.1 - drift * 2.0) * 0.35;
        var foam = broken * smoothstep(0.62 - broken * 0.5, 1.0, brk + 0.35);
        foam = max(foam, shore * smoothstep(0.05, 0.6, brk + 0.35));
        // Older foam is thinner and more broken up than fresh foam. Multiplying
        // by the noise rather than thresholding on it matters: the state texture
        // is ~2 m per texel, and a hard cut on a bilinear field draws contours.
        foam = max(foam, lingering * (0.35 + 0.9 * brk));
        foam = sat(foam * u_foam_amount()) * (1.0 - 0.6 * smoothstep(400.0, 1400.0, dist));
        let foam_col = u_foam_color() * (sun_col * (0.35 + 0.65 * n_dot_l) + sky * 0.25);
        color = mix(color, foam_col, foam);
    }

    // ---- underwater haze ----
    // Visibility underwater is metres, not kilometres, and it is the water's own
    // colour that closes in. This is the surface's share of it; the scene fog
    // handles everything else.
    if (u_submersion() > 0.0) {
        // God rays. The surface is a lens, so the light reaching a point down
        // here is already patterned by it; marching a few steps up the sun
        // direction and sampling that pattern is the cheap volumetric version.
        // Three taps, because the shafts are low-frequency and a fourth is not
        // visible.
        var shaft = 0.0;
        for (var g = 1; g <= 3; g = g + 1) {
            let up_dist = f32(g) * 6.0;
            let at = in.world_pos + sun_dir * up_dist;
            let lit = sample_surface(at.xz, 3.0);
            shaft = shaft + max(1.0 - (1.0 - lit.fold), 0.0);
        }
        shaft = clamp(shaft * 0.5, 0.0, 1.0);
        // Only where you are looking anywhere near the sun; shafts are forward
        // scattering and vanish looking away from it.
        let toward = pow(sat(dot(v, -sun_dir)), 2.0);
        let murk = 1.0 - exp(-dist * 0.012);
        color = mix(color, scatter * sun_col * 2.2, murk);
        color = color + sun_col * shaft * toward * murk * u_caustics() * 0.35;
    }

    // ---- horizon ----
    // Far enough out, the water *is* the sky it reflects. This is also what
    // hides the outer edge of the disc. Near water skips the second atmosphere
    // evaluation entirely.
    let hz = smoothstep(u_haze_near(), u_haze_far(), dist);
    if (hz > 0.002) {
        color = mix(color, sky_radiance(normalize(-v)), hz);
    }

    return vec4<f32>(framebuffer_encode(soft_clip(color * u_exposure())), cutout);
}
"#;
