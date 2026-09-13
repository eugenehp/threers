//! "Retrofuturism / Chasing Sunsets" — a scene rebuilt in threers from the
//! file it was authored in.
//!
//! Every number below was read out of the source file rather than eyeballed.
//! The scene document arrives in the authoring tool as msgpackr-encoded records
//! over a WebSocket; decoding it yields the camera, the transforms, the
//! geometry parameters, the layered materials and the post chain, and the live
//! three.js graph in the page confirms the world-space bounds each mesh ends up
//! with. What that leaves approximate is called out at each site.
//!
//! The source works in ~centimetre units (positions in the hundreds), and an
//! orthographic camera whose frustum is the editor viewport divided by the zoom
//! — so the units are kept as-authored instead of being normalised, and the
//! camera is reconstructed the same way.
//!
//!     cargo run --release --example retrofuturism
//!
//! Writes `out/retrofuturism.png` (the file as saved) and
//! `out/retrofuturism_revealed.png` — see `REVEALED` below for why the
//! second one exists.
//!
//! Environment knobs. The first two are assets that are not vendored here; the
//! rest exist so the calibrated values can be re-measured rather than trusted.
//!
//! | var | default | what it does |
//! |-----|---------|--------------|
//! | `RETRO_FONT`          | system grotesque | TrueType face for the type |
//! | `RETRO_MATCAP`        | none             | matcap texture for the sunset |
//! | `RETRO_ANIM_FPS`      | off              | render the 13s timeline at n fps |
//! | `RETRO_VIEWPORT`      | 1248x929         | authoring-window size to reproduce |
//! | `RETRO_REFR_SCALE`    | 1.0              | refracted-ray walk, in thicknesses |
//! | `RETRO_GLASS_LOD`     | 3.0              | transmission blur (mip level) |
//! | `RETRO_GLASS_DEPTH`   | 0.0              | glass depth-gradient layer on/off |
//! | `RETRO_GLASS_FRESNEL` | 1.0              | glass fresnel layer on/off |
//! | `RETRO_LIT_SCALE`     | 0.06             | light-layer scale |
//! | `RETRO_DIR_W`         | 0.4              | directional weight in the light layer |
//! | `RETRO_LIGHT_A`       | 1.0              | sunset light-layer alpha |
//! | `RETRO_OUT_PASSES`    | 1                | sRGB conversions on the way out |
//! | `RETRO_BAR_LEN`       | 6000             | glass bar length |
//! | `RETRO_NO_BARS`       | off              | hide the bars (isolation) |
//! | `RETRO_RAYTRACE`      | off              | path-trace at n spp (`--features raytrace`) |
//! | `RETRO_EMIT`          | 1.0              | sunset emission, path-traced mode only |

// The coordinates below are pasted from the document at full f64 precision and
// narrowed to f32. Keeping the extra digits makes them greppable against the
// decoded scene, so the lint that wants them trimmed is off for this file.
#![allow(clippy::excessive_precision)]

use std::sync::Arc;
use threers::captions::CaptionFont;
use threers::prelude::*;
#[cfg(feature = "raytrace")]
use threers::raytrace::{RaytraceRenderer, RaytraceSettings};

// The orthographic frustum is in CSS pixels, not world units of its own.
// The source builds the camera as
//
//     new OrthographicCamera(-0.5 * w, 0.5 * w, 0.5 * h, -0.5 * h, -5e4, 1e4)
//     new PerspectiveCamera(45, w / h, 50, 1e4)
//
// with w/h the viewport size, so at zoom 1 one world unit is exactly one pixel
// and the frustum is +-624 x +-464.5 for the 1248x929 viewport this file was
// saved against. Two consequences worth being explicit about:
//
//   * the scene has no intrinsic aspect or field of view. Resize the window and
//     you see *more or less of the scene*, not the same framing rescaled. That
//     is why the title runs off the left edge here and why the source tool's own preview
//     thumbnail (1304x844) crops differently.
//   * so VIEW_* and W/H are separate knobs. VIEW_* is the viewport being
//     reproduced and fixes what is on screen; W/H is only how many pixels that
//     is sampled at. Setting VIEW_* to W/H instead would render what a
//     1600x1191 authoring window would have shown — a wider view, not this one.
const VIEW_W: f32 = 1248.0;
const VIEW_H: f32 = 929.0;
const W: u32 = 1600;
const H: u32 = 1191;

// `orthographic.zoom` from the document. three.js's OrthographicCamera divides
// the frustum by it: half-extent = (right - left) / (2 * zoom). near/far come
// from the document too (-100000 / 100000), overriding the -5e4 / 1e4 above.
const ZOOM: f32 = 0.5987369392383788;

/// The source's `postprocessing.noise` — the only effect enabled on this scene.
const NOISE_OPACITY: f32 = 0.1;

/// The source's "Glass" material, done as the layer stack rather than as a physical material.
///
/// Sorted by `fi` the top two layers are what survive: **transmission (Normal,
/// alpha 1)**, which covers the fresnel, depth-gradient, photo and noise layers
/// beneath it, and **light (Screen, 0.24)** over that. So the surface is the
/// refracted image of whatever is behind, plus a quarter of a lit term.
///
/// Neither `PhysicalMaterial` route reproduces it. Plain `transmission` leaves
/// the dielectric's F0 lighting the whole body (bars at 21.8 against the reference's
/// 12.67); `TransparencyMode::Refract` fixes the brightness but clamps the
/// screen displacement to 2.5% of UV (`max_off` in `renderer/shader.rs`), which
/// is ~5x too small for these bars and flattens the sunset into a continuous
/// disc instead of the reference's offset slabs. It also adds Fresnel reflection on
/// top, which desaturated the sphere from 0.71 to 0.53.
///
/// So the scene is rendered twice: once with the bars hidden, and that frame is
/// handed back in as `u_tex0` for the bars to refract. Single-scattering, and
/// the bars cannot refract each other — the same limitation any screen-space
/// transmission has, including the source's own.
const GLASS_FRAGMENT: &str = r#"
fn layer_normal(a: vec3<f32>, b: vec3<f32>, al: f32) -> vec3<f32> { return mix(a, b, al); }
fn layer_screen(a: vec3<f32>, b: vec3<f32>, al: f32) -> vec3<f32> {
    return mix(a, vec3<f32>(1.0) - (vec3<f32>(1.0) - a) * (vec3<f32>(1.0) - b), al);
}
fn layer_overlay(a: vec3<f32>, b: vec3<f32>, al: f32) -> vec3<f32> {
    let t = select(1.0 - 2.0 * (1.0 - a) * (1.0 - b), 2.0 * a * b, a <= vec3<f32>(0.5));
    return clamp(mix(a, t, al), vec3<f32>(0.0), vec3<f32>(1.0));
}
fn layer_calpha(lalpha: f32, acc: f32) -> f32 { return lalpha / clamp(lalpha + acc, 0.00001, 1.0); }
fn layer_accum(acc: f32, lalpha: f32) -> f32 { return acc + (1.0 - acc) * lalpha; }

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    // Orthographic camera looking down -Z: the incident ray is constant.
    let incident = vec3<f32>(0.0, 0.0, -1.0);

    let ior         = u_data.data[0].x;
    let thickness   = u_data.data[0].y;
    let half_w      = u_data.data[1].x;
    let half_h      = u_data.data[1].y;

    let dims = vec2<f32>(textureDimensions(u_tex0, 0));
    let base_uv = in.clip_pos.xy / dims;

    // Snell through the surface, then walk `thickness` world units along the
    // bent ray. World -> UV is a constant scale because the camera is ortho.
    let refr_scale = u_data.data[1].z;
    let bent = refract(incident, n, 1.0 / ior);
    let uv = clamp(
        base_uv + vec2<f32>(
            bent.x * thickness * refr_scale * 0.5 / half_w,
            -bent.y * thickness * refr_scale * 0.5 / half_h,
        ),
        vec2<f32>(0.0), vec2<f32>(1.0),
    );
    // u_tex0 is an sRGB view, so this sample is already linear.
    let lod = u_data.data[0].w;
    let transmitted = textureSampleLevel(u_tex0, u_samp, uv, lod).rgb;

    // The stack for "Glass", evaluated in descending `fi` (base first):
    //   light(0, Screen, 0.24) -> transmission(-0.42, Normal, 1) ->
    //   noise(-1.03, Overlay, 0.16) -> texture(-1.48, Overlay, 0.2) ->
    //   depth(-2.17, Overlay, 1) -> matcap(hidden) -> fresnel(-3.12, Overlay, 0.8)
    // with each layer weighted by calpha rather than its own alpha.
    // Not reproduced: the frosted photo and the procedural noise.
    var acc = 0.0;
    var col = vec3<f32>(0.0);

    // light layer, category "physical", Screen at 0.24. Scaled by 1/PI so the
    // runtime's PI-multiplied intensities come back to the document's figures.
    let inv_pi = 1.0 / 3.14159265359;
    var lit = frame.ambient.rgb * inv_pi;
    for (var i: u32 = 0u; i < frame.light_counts.x; i = i + 1u) {
        if (i >= 4u) { break; }
        let d = normalize(frame.dir_lights[i].direction.xyz);
        lit = lit + frame.dir_lights[i].color.rgb * max(dot(n, -d), 0.0) * inv_pi;
    }
    var ca = layer_calpha(0.24, acc);
    col = layer_screen(col, clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0)), ca);
    acc = layer_accum(acc, 0.24);

    // transmission: the refracted sample of what is behind.
    ca = layer_calpha(1.0, acc);
    col = layer_normal(col, transmitted, ca);
    acc = layer_accum(acc, 1.0);

    // depth gradient, object space, direction (1, -180, 0) from origin
    // (-80, 14400, 0) over near/far -+36900.57 — bounds so large relative to the
    // bar that the ramp is nearly flat across it (t = 0.682..0.708).
    let centre = u_data.data[2].xyz;
    let cs = u_data.data[2].w;
    let sn = u_data.data[3].x;
    let d3 = in.world_pos - centre;
    let local = vec3<f32>(cs * d3.x + sn * d3.y, -sn * d3.x + cs * d3.y, d3.z);
    let gdir = normalize(vec3<f32>(1.0, -180.0, 0.0));
    let gdist = dot(gdir, local - vec3<f32>(-80.0, 14400.0, 0.0));
    let gt = clamp((gdist + 36900.56944005065) / 73801.1388801013, 0.0, 1.0);
    let grey = mix(0.0006399802791261788, 0.6413681932367568, gt);
    let depth_on = u_data.data[3].y;
    ca = layer_calpha(1.0 * depth_on, acc);
    col = layer_overlay(col, vec3<f32>(grey), ca);
    acc = layer_accum(acc, 1.0 * depth_on);

    // fresnel, verbatim from the source's FresnelNode:
    //   fresnel = bias + scale * pow(abs(factor + dot(viewDir, normal)), intensity)
    //   lalpha  = clamp(fresnel, 0, 1) * layerAlpha
    // `intensity` is the exponent and `factor` an offset inside the dot.
    let fres = 0.24 + 2.08 * pow(abs(1.17 + dot(incident, n)), 5.81);
    let fa = clamp(fres, 0.0, 1.0) * 0.8 * u_data.data[3].z;
    ca = layer_calpha(fa, acc);
    col = layer_overlay(col, vec3<f32>(1.0), ca);



    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
"#;

/// The sphere's "Red Gradient": the depth gradient as the base, with the light
/// layer composited over it in Overlay (mode 3).
///
/// The layer order here is settled empirically, not from the document, because
/// I could not isolate components in the live scene — `o.visible = false` does
/// not stick in the authoring tool, which re-derives visibility from its document
/// every frame. Four different visibility masks produced identical frames, so
/// every "hide X and measure" reading I took is void.
///
/// What the recorded frames say, scored as whole-frame mean|diff|:
///   gradient + light, no matcap  30.4   <- used
///   gradient + light + matcap    35.4
///   gradient only (light off)    51.9
/// So the light layer is applied and the matcap is not. That fits neither pure
/// reading of the `fi` order, and the likeliest explanation is that the matcap's
/// orientation here is wrong rather than the layer being absent — an unresolved
/// loose end, not a finding.
///
/// The light layer's category is "phong", specular 0.2 / shininess 5, and its
/// intensities go in scaled by 1/PI: the Phong BRDF's 1/PI cancels the PI the
/// runtime multiplies into 2.199 / 2.356, recovering the document's 0.7 / 0.75.
/// Fitting that scale freely lands on 0.20 against 1/PI = 0.3183 at 30.44 vs
/// 30.53 — inside the noise, so the derived value is kept.
const SPHERE_FRAGMENT: &str = r#"
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return select(c / 12.92, pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c > vec3<f32>(0.04045));
}

// Layer mode 3. Enum from the source bundle: Normal=0, Multiply=1, Screen=2,
// Overlay=3.

// The source's layer blends, verbatim from its shader chunk:
//   layer_normalBlend (a,b,al) = mix(a, b, al)
//   layer_screenBlend (a,b,al) = mix(a, 1-(1-a)(1-b), al)
//   layer_overlayBlend(a,b,al) = clamp(mix(a, mix(1-2(1-a)(1-b), 2ab, step(a,0.5)), al), 0, 1)
// and the accumulator that weights them:
//   accumAlpha += (1 - accumAlpha) * alpha
//   calpha      = lalpha / clamp(lalpha + accumAlpha, 1e-5, 1)
// So a layer is NOT applied at its own alpha: the first layer evaluated gets
// weight 1, the next 0.5, and so on. Getting that wrong is what made the light
// and matcap layers overpower the gradient.
fn layer_normal(a: vec3<f32>, b: vec3<f32>, al: f32) -> vec3<f32> { return mix(a, b, al); }
fn layer_screen(a: vec3<f32>, b: vec3<f32>, al: f32) -> vec3<f32> {
    return mix(a, vec3<f32>(1.0) - (vec3<f32>(1.0) - a) * (vec3<f32>(1.0) - b), al);
}
fn layer_overlay(a: vec3<f32>, b: vec3<f32>, al: f32) -> vec3<f32> {
    let t = select(1.0 - 2.0 * (1.0 - a) * (1.0 - b), 2.0 * a * b, a <= vec3<f32>(0.5));
    return clamp(mix(a, t, al), vec3<f32>(0.0), vec3<f32>(1.0));
}
fn layer_calpha(lalpha: f32, acc: f32) -> f32 { return lalpha / clamp(lalpha + acc, 0.00001, 1.0); }
fn layer_accum(acc: f32, lalpha: f32) -> f32 { return acc + (1.0 - acc) * lalpha; }

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let centre    = u_data.data[0].xyz;
    let inv_scale = u_data.data[0].w;
    let near      = u_data.data[1].x;
    let far       = u_data.data[1].y;
    let c0        = u_data.data[2].rgb;
    let c1        = u_data.data[3].rgb;
    let matcap_a  = u_data.data[4].x;
    let light_a   = u_data.data[4].y;
    let spec      = u_data.data[4].z;
    let shininess = u_data.data[4].w;

    // --- base: the depth gradient -------------------------------------
    // `vectorLinearObjectSpaceDepth` uses the raw unscaled local position, and
    // `smooth: false` takes the plain-mix branch.
    let local_x = (in.world_pos.x - centre.x) * inv_scale;
    let t = clamp((local_x - near) / (far - near), 0.0, 1.0);
    let gradient = clamp(mix(c0, c1, t), vec3<f32>(0.0), vec3<f32>(1.0));

    let n = normalize(in.world_normal);
    // Orthographic camera down -Z with zero rotation, so view space == world
    // space here and the view vector is constant.
    let v = vec3<f32>(0.0, 0.0, 1.0);

    // --- light layer (Overlay) ----------------------------------------
    // The renderer's light terms carry the runtime's PI-scaled intensities
    // (2.199 / 2.356). A Phong BRDF divides by PI, which cancels it and returns
    // the document's own 0.7 / 0.75. Skipping that leaves ambient alone at ~1.95
    // so `clamp(lit, 0, 1)` saturates everywhere and the Overlay blows the whole
    // sphere toward white.
    let inv_pi = u_data.data[5].x;
    // `frame.ambient` carries only AmbientLight here — the HemisphereLight is
    // shaded per-normal by the renderer and never reaches it, so reading it left
    // the light layer with no ambient floor and the directional term swinging
    // the whole way on its own. That is what turned the sunset's lit top from
    // red to tan while its unlit bottom stayed red. three.js's hemisphere
    // irradiance is mix(ground, sky, 0.5 + 0.5 * n.y) * intensity.
    let hemi_sky = vec3<f32>(0.827451, 0.827451, 0.827451);     // #d3d3d3
    let hemi_ground = vec3<f32>(0.509804, 0.509804, 0.509804);  // #828282
    let hemi = mix(hemi_ground, hemi_sky, 0.5 + 0.5 * n.y) * 2.356194490192345;
    var lit = (frame.ambient.rgb + hemi) * inv_pi;
    for (var i: u32 = 0u; i < frame.light_counts.x; i = i + 1u) {
        if (i >= 4u) { break; }
        let d = normalize(frame.dir_lights[i].direction.xyz);
        let ndl = max(dot(n, -d), 0.0);
        // The directional term is what swings `lit` across Overlay's 0.5 branch
        // point between the sphere's lit top and unlit bottom, which is what
        // turns the upper part of the sunset from red to washed tan. Weighted
        // separately so the size of that swing can be measured.
        let dir_w = u_data.data[5].z;
        lit = lit + frame.dir_lights[i].color.rgb * ndl * inv_pi * dir_w;
        let h = normalize(-d + v);
        lit = lit + vec3<f32>(spec) * pow(max(dot(n, h), 0.0), shininess) * dir_w;
    }
    // Evaluation order is descending `fi` — the highest-fi layer is the base.
    // For "Red Gradient" that is depth(1) -> light(0) -> matcap(-0.43).
    var acc = 0.0;
    var col = vec3<f32>(0.0);
    var ca = layer_calpha(1.0, acc);
    col = layer_normal(col, gradient, ca);          // depth, mode Normal, alpha 1
    acc = layer_accum(acc, 1.0);

    ca = layer_calpha(light_a, acc);
    col = layer_overlay(col, clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0)), ca);
    acc = layer_accum(acc, light_a);

    // --- matcap layer (Overlay, on top) -------------------------------
    // The source MatcapNode, verbatim:
    //   vec3 viewDir = normalize(vViewPosition);
    //   vec3 x = normalize(vec3(viewDir.z, 0.0, -viewDir.x));
    //   vec3 y = cross(viewDir, x);
    //   vec2 uv = vec2(dot(x, normal), dot(y, normal)) * 0.495 + 0.5;
    // This camera is orthographic with zero rotation, so viewDir is (0, 0, 1),
    // x becomes (1, 0, 0) and y becomes (0, 1, 0) — the basis collapses to the
    // normal's xy. Note 0.495, not 0.5, and no V flip: I had both wrong, which
    // is why adding the matcap previously made the match worse.
    let mc_uv = n.xy * 0.495 + vec2<f32>(0.5, 0.5);
    let mc = textureSampleLevel(u_tex0, u_samp, mc_uv, 0.0).rgb;
    ca = layer_calpha(matcap_a, acc);
    col = layer_overlay(col, mc, ca);
    acc = layer_accum(acc, matcap_a);

    // The source writes the document's figures out unconverted (ColorManagement is
    // off), so what reaches the screen is srgb_to_linear(value). This target is
    // Rgba8UnormSrgb and the hardware encodes on write, so cancelling that costs
    // one more conversion — hence two.
    let c = clamp(col, vec3<f32>(0.0), vec3<f32>(1.0));
    if (u_data.data[5].y > 1.5) {
        return vec4<f32>(srgb_to_linear(srgb_to_linear(c)), 1.0);
    }
    return vec4<f32>(srgb_to_linear(c), 1.0);
}
"#;


/// EASING_TYPE, read out of the source bundle:
/// `LINEAR=0, EASE=1, EASE_IN=2, EASE_OUT=3, EASE_IN_OUT=4, CUBIC=5, SPRING=6,
/// ARC=7, NONE=8`. Every one of them is a cubic Bezier, and the two this scene
/// uses are LINEAR and EASE_IN_OUT = `cubic-bezier(.42, 0, .58, 1)`.
fn ease_in_out(x: f32) -> f32 {
    cubic_bezier(0.42, 0.0, 0.58, 1.0, x)
}

/// CSS-style `cubic-bezier(x1, y1, x2, y2)`: solve x(t) = `x` for t by bisection,
/// then return y(t). Bisection rather than Newton because the curve is only
/// evaluated a few times per frame and monotonicity makes it unconditionally
/// safe.
fn cubic_bezier(x1: f32, y1: f32, x2: f32, y2: f32, x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    let bez = |a: f32, b: f32, t: f32| {
        let u = 1.0 - t;
        3.0 * u * u * t * a + 3.0 * u * t * t * b + t * t * t
    };
    let (mut lo, mut hi) = (0.0f32, 1.0f32);
    for _ in 0..32 {
        let mid = 0.5 * (lo + hi);
        if bez(x1, x2, mid) < x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    bez(y1, y2, 0.5 * (lo + hi))
}

fn lerp3(a: Vector3, b: Vector3, t: f32) -> Vector3 {
    Vector3::new(
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t,
    )
}


/// Uniform scale on the "CTA container" group the type hangs under. One em of
/// a child's `fontSize` is this many world units.
const CTA_SCALE: f32 = 2.250553032455589;

fn main() {
    let _ = std::fs::create_dir_all("out");

    let mut renderer = HeadlessRenderer::builder()
        .size(W, H)
        .build()
        .expect("headless renderer (is a GPU adapter available?)");

    // ---- camera -------------------------------------------------------
    // Conventions, taken from the source's own code rather than assumed:
    //   * axes are three.js's — Y up, -Z forward. The camera records
    //     `up: DefaultUp` (0,1,0) and `isUpVectorFlipped: false`, so there is no
    //     flip to undo anywhere in this file.
    //   * rotations are stored in *degrees* and applied as
    //     `new Euler(x, y, z, "XYZ")`, three.js's default order.
    //   * the look-at is not stored; it is derived. The source does
    //     `camera.getWorldDirection(n); target = position + n * targetOffset`,
    //     and this camera has `targetOffset: 1000`.
    // Its stored rotation is (0, 1.8e-13, 0) — zero — so the world direction is
    // straight down -Z and the target lands on z = 0.
    // near is genuinely negative (-100000), which an orthographic projection is
    // happy with.
    // Stretching the frustum against a different output aspect would shear the
    // composition, so say so rather than quietly producing a wrong picture.
    // `RETRO_VIEWPORT=WxH` reproduces a different authoring-window size, which
    // changes how much of the scene is on screen rather than rescaling it.
    let (view_w, view_h) = std::env::var("RETRO_VIEWPORT")
        .ok()
        .and_then(|v| {
            let (a, b) = v.split_once('x')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        })
        .unwrap_or((VIEW_W, VIEW_H));
    let view_aspect = view_w / view_h;
    let out_aspect = W as f32 / H as f32;
    assert!(
        (view_aspect - out_aspect).abs() / view_aspect < 0.002,
        "output {W}x{H} (aspect {out_aspect:.5}) does not match the viewport \
         {view_w}x{view_h} (aspect {view_aspect:.5}); the frustum is in pixels, \
         so these have to agree"
    );
    let half_w = (view_w / 2.0) / ZOOM;
    let half_h = (view_h / 2.0) / ZOOM;
    println!(
        "  camera: ortho frustum +-{half_w:.1} x +-{half_h:.1} world units \
         (viewport {view_w}x{view_h} px / zoom {ZOOM:.4})"
    );
    let eye = Vector3::new(-246.4915293476442, 58.3693227188394, 1000.0);
    let mut camera =
        OrthographicCamera::new(-half_w, half_w, half_h, -half_h, -100_000.0, 100_000.0);
    camera.position = eye;
    // target = position + worldDirection * targetOffset
    const TARGET_OFFSET: f32 = 1000.0;
    let world_dir = Vector3::new(0.0, 0.0, -1.0);
    camera.target = Vector3::new(
        eye.x + world_dir.x * TARGET_OFFSET,
        eye.y + world_dir.y * TARGET_OFFSET,
        eye.z + world_dir.z * TARGET_OFFSET,
    );

    // `RETRO_ANIM_FPS=n` renders the Start-event timeline instead of the two
    // stills: 13 seconds, which is one full cycle of the sphere and covers the
    // card's 8-second rise.
    let matcap = load_matcap(&renderer.device().clone(), &renderer.queue().clone());
    if let Some(fps) = std::env::var("RETRO_ANIM_FPS").ok().and_then(|v| v.parse::<f32>().ok()) {
        let _ = std::fs::create_dir_all("out/anim");
        let total = (13.0 * fps).round() as u32;
        for i in 0..total {
            let t = i as f32 / fps;
            let mut rgba = render_frame(&mut renderer, &camera, false, t, half_w, half_h, matcap.clone());
            // Grain is re-seeded per frame so it crawls, as a noise pass does.
            film_grain_seeded(&mut rgba, NOISE_OPACITY, 0x9e3779b9u32.wrapping_add(i.wrapping_mul(2654435761)));
            let path = format!("out/anim/frame_{i:04}.png");
            std::fs::write(&path, encode_png(W, H, &rgba)).expect("write png");
            if i % 10 == 0 {
                println!("  t={t:5.2}s  card.y={:8.1}  sphere=({:7.1},{:7.1})",
                         pose_at(t).card.y, pose_at(t).sphere.x, pose_at(t).sphere.y);
            }
        }
        println!("wrote {total} frames to out/anim/ at {fps} fps");
        return;
    }

    #[cfg(feature = "raytrace")]
    if let Some(spp) = std::env::var("RETRO_RAYTRACE").ok().and_then(|v| v.parse().ok()) {
        render_raytraced(&camera, spp);
        return;
    }

    let matcap = load_matcap(&renderer.device().clone(), &renderer.queue().clone());

    for revealed in [false, true] {
        let mut rgba = render_frame(&mut renderer, &camera, revealed, 0.0, half_w, half_h, matcap.clone());
        film_grain(&mut rgba, NOISE_OPACITY);

        let path = if revealed {
            "out/retrofuturism_revealed.png"
        } else {
            "out/retrofuturism.png"
        };
        std::fs::write(path, encode_png(W, H, &rgba)).expect("write png");
        println!("wrote {path}");
    }
}

/// The document's Start-event timeline, evaluated at `t` seconds.
///
/// Both animated objects carry a `Transition` action on a `Start` event; the
/// tweens are ordered by their fractional index (`fi`), and each names a target
/// state plus a duration in ms. The runtime chains them: `seek` computes the
/// chain length as `tweens.reduce((a, t) => a + t.duration, 0)` and walks it in
/// order, so durations accumulate rather than running in parallel. Nothing here is invented — the states, the
/// durations and the easing indices are all in the scene document. The one
/// thing that is *not* animated is the light: `Directional Light` has empty
/// `states` and `events`, and sampling its position and intensity in the live
/// runtime over 12s returns a constant (200, 300, 300) at 2.1991.
struct Pose {
    card: Vector3,
    sphere: Vector3,
}

fn pose_at(t: f32) -> Pose {
    // Cube: 1000ms holding the base state (ease-in-out), then 7000ms LINEAR up
    // to y = 1428.46 — it slides out of frame, which is what uncovers the
    // sphere. This is the reveal the scene is built around.
    let card_base = Vector3::new(145.60763036207214, 147.65662421477077, -189.7356811535165);
    let card_up = Vector3::new(145.60763036207214, 1428.457721431038, -189.7356811535165);
    let card = if t <= 1.0 {
        card_base
    } else {
        // easing 0 = LINEAR on this leg.
        lerp3(card_base, card_up, ((t - 1.0) / 7.0).clamp(0.0, 1.0))
    };

    // Sphere: base -> "To the right" -> "To the left" -> base, 4000ms a leg,
    // all easing 4 (ease-in-out), after the same 1000ms hold. The chain totals
    // 13000ms and then *stops*: the action is `repeat: 0, direction: normal,
    // runMode: Once`. The `repeat: -1, pingpong` on the first tween is a
    // per-tween field, and the runtime takes its loop control from the action
    // (`this.config.repeat` / `passRepeat`), not from the tweens — so this plays
    // through once and holds, it does not ping-pong forever.
    let s_base = Vector3::new(224.79337644005562, 175.34023749472013, -759.9539759195773);
    let s_right = Vector3::new(-45.66282033691418, 24.02734375, -713.0);
    let s_left = Vector3::new(57.856710913085664, 24.02734375, -713.0);
    let sphere = if t <= 1.0 {
        s_base
    } else if t < 5.0 {
        lerp3(s_base, s_right, ease_in_out((t - 1.0) / 4.0))
    } else if t < 9.0 {
        lerp3(s_right, s_left, ease_in_out((t - 5.0) / 4.0))
    } else if t < 13.0 {
        lerp3(s_left, s_base, ease_in_out((t - 9.0) / 4.0))
    } else {
        s_base
    };

    Pose { card, sphere }
}


/// Render once with the bars hidden and hand that frame back as a GPU texture
/// for them to refract. The source transmission layer samples a capture of the
/// scene the same way; doing it here keeps the whole reconstruction inside the
/// example instead of relying on the renderer's own (tightly clamped) one.
fn upload_rgba(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    w: u32,
    h: u32,
    rgba: &[u8],
    label: &str,
) -> std::sync::Arc<wgpu::TextureView> {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * 4),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    std::sync::Arc::new(tex.create_view(&wgpu::TextureViewDescriptor::default()))
}

/// `matcap_reflection_2`, the texture the "Red Gradient" material names. Served
/// from the source tool's public asset CDN; `RETRO_MATCAP` points at a local copy. With
/// no file the sphere falls back to gradient + light only, which loses the
/// specular highlight.
fn load_matcap(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Option<std::sync::Arc<wgpu::TextureView>> {
    let path = std::env::var("RETRO_MATCAP").ok()?;
    let bytes = std::fs::read(&path).ok()?;
    let img = decode_png(&bytes).ok()?;
    println!("matcap: {path} ({}x{})", img.width, img.height);
    Some(upload_rgba(device, queue, img.width, img.height, &img.rgba, "retro matcap"))
}

fn render_frame(
    renderer: &mut HeadlessRenderer,
    camera: &OrthographicCamera,
    revealed: bool,
    t: f32,
    half_w: f32,
    half_h: f32,
    matcap: Option<std::sync::Arc<wgpu::TextureView>>,
) -> Vec<u8> {
    let mut behind_scene = build_scene(revealed, t, half_w, half_h, None, matcap.clone());
    let behind_rgba = renderer.render_to_rgba(&mut behind_scene, camera);

    let device = renderer.device().clone();
    let queue = renderer.queue().clone();
    // three.js drives the transmission sample's blur from roughness via
    // `applyIorToRoughness` + a mip chain. The document stores roughness 4.5, which is
    // far outside 0..1 and saturates that selection to the blurriest levels —
    // which is why its bars read as soft glowing bands rather than a sharp
    // refracted image. Mips are built on the CPU here (a box filter is plenty
    // for something this blurred) and uploaded level by level.
    let mut mips: Vec<(u32, u32, Vec<u8>)> = vec![(W, H, behind_rgba.clone())];
    while mips.last().unwrap().0 > 1 && mips.last().unwrap().1 > 1 {
        let (pw, ph, ref prev) = *mips.last().unwrap();
        let (nw, nh) = ((pw / 2).max(1), (ph / 2).max(1));
        let mut next = vec![0u8; (nw * nh * 4) as usize];
        for y in 0..nh {
            for x in 0..nw {
                for c in 0..4 {
                    let mut acc = 0u32;
                    for dy in 0..2u32 {
                        for dx in 0..2u32 {
                            let sx = (x * 2 + dx).min(pw - 1);
                            let sy = (y * 2 + dy).min(ph - 1);
                            acc += prev[(((sy * pw + sx) * 4) + c) as usize] as u32;
                        }
                    }
                    next[(((y * nw + x) * 4) + c) as usize] = (acc / 4) as u8;
                }
            }
        }
        mips.push((nw, nh, next));
    }
    let mip_count = mips.len() as u32;

    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("retro behind-glass capture"),
        size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        mip_level_count: mip_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (level, (mw, mh, data)) in mips.iter().enumerate() {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: level as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(*mw * 4),
                rows_per_image: Some(*mh),
            },
            wgpu::Extent3d { width: *mw, height: *mh, depth_or_array_layers: 1 },
        );
    }
    let view = std::sync::Arc::new(tex.create_view(&wgpu::TextureViewDescriptor::default()));

    let n = behind_rgba.len() / 4;
    let mean: f64 = behind_rgba.chunks_exact(4).map(|p| p[0] as f64).sum::<f64>() / n as f64;
    println!("    pass1 (no bars): {n} px, mean R = {mean:.2}");
    let mut scene = build_scene(revealed, t, half_w, half_h, Some(view), matcap);
    let out = renderer.render_to_rgba(&mut scene, camera);
    let mean2: f64 = out.chunks_exact(4).map(|p| p[0] as f64).sum::<f64>() / n as f64;
    println!("    pass2 (bars)   : mean R = {mean2:.2}");
    out
}


/// Path-traced variant, behind the `raytrace` feature.
///
/// The screen-space route above fakes transmission by refracting a capture of
/// the frame. A path tracer does the real thing: the sunset is emissive
/// geometry, the bars are a rough dielectric, and light actually propagates
/// through them — so the separated glowing slabs come out of the integral
/// rather than being approximated by a UV offset. That also removes the two
/// fitted constants the screen-space version needs (`RETRO_REFR_SCALE` and the
/// mip LOD standing in for roughness).
///
/// `Material::Shader` traces as a plain rough dielectric, so the sphere and the
/// glass are expressed as real materials here rather than as custom WGSL. The
/// gradient is baked into an emissive map, exactly: the sphere's UVs are
/// three.js's (`x = -r·cos(phi)·sin(theta)`, `uv = (u, 1 - v)`), so every texel
/// can be mapped back to the object-space X the depth layer ramps along.
#[cfg(feature = "raytrace")]
fn bake_gradient_emissive(n: u32) -> Texture {
    let c0 = [0.7897781683427412f32, 0.46977418894155454, 0.46977418894155454];
    let c1 = [0.44236163134928375f32, 0.2446691092228538, 0.09349247465558384];
    let (near, far) = (-534.8627583784917f32, 532.6500685688682f32);
    let radius = 347.5816085339573f32;
    let mut data = vec![0u8; (n * n * 4) as usize];
    for ty in 0..n {
        for tx in 0..n {
            let tu = tx as f32 / (n - 1) as f32;
            let tv = ty as f32 / (n - 1) as f32;
            let (u, v) = (tu, 1.0 - tv);
            let (phi, theta) = (u * std::f32::consts::TAU, v * std::f32::consts::PI);
            let x = -radius * phi.cos() * theta.sin();
            let t = ((x - near) / (far - near)).clamp(0.0, 1.0);
            let o = ((ty * n + tx) * 4) as usize;
            for c in 0..3 {
                data[o + c] = ((c0[c] + (c1[c] - c0[c]) * t) * 255.0) as u8;
            }
            data[o + 3] = 255;
        }
    }
    DataTexture::new(n, n, TextureFormat::Rgba8UnormSrgb, data)
}

#[cfg(feature = "raytrace")]
fn render_raytraced(camera: &OrthographicCamera, spp: u32) {
    use std::sync::Arc;
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    scene.add_light(DirectionalLight {
        color: Color::WHITE,
        intensity: 2.199114857512855,
        direction: Vector3::new(-200.0, -300.0, -300.0).normalize(),
        cast_shadow: true,
        ..Default::default()
    });
    scene.add_light(HemisphereLight::new(
        Color::from_hex(0xd3d3d3),
        Color::from_hex(0x828282),
        2.356194490192345,
    ));

    // Backdrop card, unlit black.
    let mut card = Object3D::mesh(Mesh::new(
        BoxGeometry::new(1231.0, 1201.0, 92.020001026402),
        Material::Basic(BasicMaterial::new(Color::BLACK)),
    ));
    card.position = Vector3::new(145.60763036207214, 1428.457721431038, -189.7356811535165);
    scene.add(card);

    // The sunset, as an emitter. The tracer samples emissive geometry as an
    // area light, which is what lets it illuminate the glass.
    let ramp = Arc::new(bake_gradient_emissive(512));
    let mut sphere = Object3D::mesh(Mesh::new(
        SphereGeometry::new(347.5816085339573, 64, 64),
        Material::Physical(PhysicalMaterial {
            emissive: Color::WHITE,
            emissive_intensity: std::env::var("RETRO_EMIT").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0),
            emissive_map: Some(ramp),
            roughness: 1.0,
            ..PhysicalMaterial::new(Color::BLACK)
        }),
    ));
    sphere.position = Vector3::new(224.79337644005562, 175.34023749472013, -759.9539759195773);
    let sc = 1.3329674150872226;
    sphere.scale = Vector3::new(sc, sc, sc);
    scene.add(sphere);

    // Reeded glass: a real rough dielectric, ior and thickness from the
    // transmission layer.
    let glass = Material::Physical(PhysicalMaterial {
        roughness: 0.045,
        metalness: 0.0,
        transmission: 1.0,
        ior: 1.16,
        thickness: 565.0,
        ..PhysicalMaterial::new(Color::new(0.98, 0.98, 0.98))
    });
    // Run long, same as the rasterised path: the document's 1956.07 ends inside
    // the frame and lets the bare sphere show past it.
    let bar_len: f32 = std::env::var("RETRO_BAR_LEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6000.0);
    let base = Vector3::new(-284.2456753518419, -279.9290563101541, 124.5);
    for k in 0..24 {
        let mut bar = Object3D::mesh(Mesh::new(
            CylinderGeometry::new(124.5, 124.5, bar_len, 8, 1, false, 0.0, std::f32::consts::TAU),
            glass.clone(),
        ));
        bar.position = Vector3::new(base.x + 157.9 * k as f32, base.y, base.z);
        // Same elliptical cross-section as the rasterised path.
        bar.scale = Vector3::new(176.64167557758233 / 249.0, 1.0, 1.0);
        bar.quaternion = Quaternion::from_euler_xyz(0.0, 0.0, 45.0_f32.to_radians());
        scene.add(bar);
    }

    // The CPU reference does not finish this frame in reasonable time at
    // 1600x1191; the wgpu compute backend runs the same estimator.
    let mut renderer = match threers::raytrace::gpu::GpuBackend::headless() {
        Ok(backend) => {
            println!("raytrace backend: gpu");
            RaytraceRenderer::with_backend(W, H, Box::new(backend))
        }
        Err(e) => {
            println!("raytrace backend: cpu ({e})");
            RaytraceRenderer::new(W, H)
        }
    };
    renderer.set_settings(RaytraceSettings {
        samples_per_pixel: spp,
        max_bounces: 12,
        min_bounces: 4,
        clamp_indirect: 16.0,
        // The source applies no tone mapping at exposure 1.
        tone_mapping: threers::renderer::ToneMapping::None,
        exposure: 1.0,
        denoise: true,
        ..Default::default()
    });
    renderer.prepare(&mut scene, camera);
    if let Some(report) = renderer.report() {
        println!("raytrace scene: {}", report.summary());
        for note in &report.approximated {
            println!("  note: {note}");
        }
    }
    let batch = 32;
    let mut done = 0;
    while done < spp {
        let n = batch.min(spp - done);
        renderer.accumulate(n).expect("trace");
        done += n;
        println!("  {done}/{spp} samples");
    }
    let mut rgba = renderer.resolve_rgba();
    film_grain(&mut rgba, NOISE_OPACITY);
    std::fs::write("out/retrofuturism_rt.png", encode_png(W, H, &rgba)).expect("write png");
    println!("wrote out/retrofuturism_rt.png");
}

fn build_scene(
    revealed: bool,
    t: f32,
    half_w: f32,
    half_h: f32,
    behind: Option<std::sync::Arc<wgpu::TextureView>>,
    matcap: Option<std::sync::Arc<wgpu::TextureView>>,
) -> Scene {
    let pose = pose_at(t);
    let mut scene = Scene::new();
    // `Scene::new()` starts at 0x111111; the document's backgroundColor is black.
    scene.background = Color::BLACK;

    // ---- lights -------------------------------------------------------
    // The source document stores 0.7 / 0.75; its runtime hands the shader those
    // times PI, and three.js r185 has no legacy-lights path, so 2.199 / 2.356
    // are the physical values and go in unscaled. The materials below do their
    // own weighting of these — see the light layers in the two shaders.
    scene.add_light(DirectionalLight {
        color: Color::WHITE,
        intensity: 2.199114857512855,
        // Stored as a position (200, 300, 300) aimed at the origin.
        direction: Vector3::new(-200.0, -300.0, -300.0).normalize(),
        cast_shadow: true,
        ..Default::default()
    });
    // "Default Ambient Light" is a HemisphereLight, and it is not a single
    // colour: sky is #d3d3d3 but ground is #828282 — which the document does not
    // record, only the live light object has it.
    scene.add_light(HemisphereLight::new(
        Color::from_hex(0xd3d3d3),
        Color::from_hex(0x828282),
        2.356194490192345,
    ));

    // ---- backdrop card ------------------------------------------------
    // CubeGeometry 1231 x 1201 x 92.02, cornerRadius 8. The radius is dropped:
    // 8 units on a 1231-unit face is a third of a pixel at this zoom.
    // Its material is a single flat colour layer at #000000 with the lighting
    // layer switched off, so it is unlit black — BasicMaterial, not Standard.
    let mut card = Object3D::mesh(Mesh::new(
        BoxGeometry::new(1231.0, 1201.0, 92.020001026402),
        Material::Basic(BasicMaterial::new(Color::BLACK)),
    ));
    // At t = 0 this is the position in the document. The `revealed` still keeps
    // the old shortcut of pushing the card backwards, which is a cruder version
    // of what the timeline does properly by sliding it up.
    card.position = if revealed {
        Vector3::new(145.60763036207214, 147.65662421477077, -1400.0)
    } else {
        pose.card
    };
    card.name = "Cube".into();
    scene.add(card);

    // ---- sunset sphere ------------------------------------------------
    // Material "Red Gradient": a matcap layer over a light layer over a depth
    // gradient running along +X from #c97878 to #713e18. threers has no layered
    // material, so the dominant layer — the gradient — is baked to a texture and
    // sampled through the sphere's own UVs. That is the main visual
    // approximation in this file.
    let sphere_scale = 1.3329674150872226f32;
    let mut sphere = Object3D::mesh(Mesh::new(
        SphereGeometry::new(347.5816085339573, 64, 64),
        Material::Shader(ShaderMaterial {
            textures: matcap.clone().into_iter().collect(),
            ..ShaderMaterial::new(SPHERE_FRAGMENT).with_data(vec![
            [pose.sphere.x, pose.sphere.y, pose.sphere.z, 1.0 / sphere_scale],
            [-534.8627583784917, 532.6500685688682, 0.0, 0.0],
            [0.7897781683427412, 0.46977418894155454, 0.46977418894155454, 1.0],
            [0.44236163134928375, 0.2446691092228538, 0.09349247465558384, 1.0],
            // matcap alpha, light alpha, phong specular, shininess
            [
                if matcap.is_some() { 1.0 } else { 0.0 },
                std::env::var("RETRO_LIGHT_A").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0),
                0.2,
                5.0,
            ],
            // light BRDF scale, output conversion passes
            [
                std::env::var("RETRO_LIT_SCALE").ok().and_then(|v| v.parse().ok())
                    // Rebalanced once the hemisphere term was added: it lifts
                    // `lit` a long way, so this drops from 0.22 to 0.06.
                    .unwrap_or(0.06),
                // 1, not 2. Fitted against the recorded frames: one conversion
                // scores 35.6 mean|diff| where two scores 44.1. The hardware
                // sRGB write supplies the other half of the round trip.
                std::env::var("RETRO_OUT_PASSES").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0),
                // 0.4. At 1.0 the directional term swings `lit` across
                // Overlay's 0.5 branch point between the sunset's lit top and
                // unlit bottom, so the bottom read red and the top washed to
                // tan — a vertical hue split of 0.39 in G/R. With the hemisphere
                // floor restored and this weighted back, the split is 0.11 and
                // the average G/R is 0.37 against the reference's ~0.35.
                std::env::var("RETRO_DIR_W").ok().and_then(|v| v.parse().ok()).unwrap_or(0.4),
                0.0,
            ],
        ])
        }),
    ));
    sphere.position = pose.sphere;
    // Also an ellipsoid: width 695.16, height 721.25, depth 715.14. The
    // geometry is built at the width, so height/depth get their own factors.
    sphere.scale = Vector3::new(
        sphere_scale,
        sphere_scale * 721.2535016456279 / 695.1632170679145,
        sphere_scale * 715.1413622030852 / 695.1632170679145,
    );
    sphere.name = "Sphere".into();
    scene.add(sphere);

    // ---- reeded glass -------------------------------------------------
    // One cylinder plus a linear cloner: count 24, step (157.9, 0, 0).
    // radialSegments is 8, so these are octagonal prisms, not smooth tubes —
    // that faceting is what gives the reeded look, so it is kept exactly.
    //
    // ior 1.16 and thickness 565 are the transmission layer's own values.
    // `RETRO_GLASS_LOD` overrides the blur level for calibration.
    // Displacement along the refracted ray: exactly `thickness` world units, as
    // three.js's `getVolumeTransmissionRay` does. 1.0 is the derived value and
    // it is also the measured optimum — 0.75 -> 29.58, 1.0 -> 29.36,
    // 1.25 -> 29.51 mean|diff| against the recorded frames.
    //
    // It was not always. With the bars built circular (249 wide instead of the
    // document's 176.64) they overlapped continuously, the slabs fused, and this
    // had to be fudged to 1.5 to fake the gaps. Fixing the cross-section made
    // the fudge unnecessary. Override with `RETRO_REFR_SCALE`.
    let refr_scale: f32 = std::env::var("RETRO_REFR_SCALE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    // Transmission blur. The document stores roughness 4.5, far outside three.js's
    // 0..1, which saturates its mip selection to the blurriest levels — hence
    // the soft glowing bands rather than a sharp refracted image.
    let glass_lod: f32 = std::env::var("RETRO_GLASS_LOD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3.0);
    let skip_bars = behind.is_none() || std::env::var("RETRO_NO_BARS").is_ok();
    // The depth-gradient layer is object-space, so each clone needs its own
    // centre and the inverse of its 45-degree rotation.
    let (sin_a, cos_a) = (-45.0f32).to_radians().sin_cos();
    // The document's bar is 1956.07 long, which ends at y ~= 412 once turned 45
    // degrees — above that the sphere is no longer behind glass and its bare
    // silhouette shows. The original never shows that, so the bars are run long enough
    // for both ends to leave the frustum (half-extent 1042 x 776, so the visible
    // diagonal is ~2600) instead of stopping mid-scene.
    let bar_len: f32 = std::env::var("RETRO_BAR_LEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6000.0);
    let base = Vector3::new(-284.2456753518419, -279.9290563101541, 124.5);
    for k in 0..24 {
        if skip_bars { break; }
        let bar_centre = Vector3::new(base.x + 157.9 * k as f32, base.y, base.z);
        let glass = Material::Shader(ShaderMaterial {
            textures: vec![behind.clone().unwrap()],
            ..ShaderMaterial::new(GLASS_FRAGMENT).with_data(vec![
                [1.16, 565.0, 0.24, glass_lod],
                [half_w, half_h, refr_scale, 0.0],
                [bar_centre.x, bar_centre.y, bar_centre.z, cos_a],
                [
                    sin_a,
                    // Off by default at 0.0 despite being in the document
                    // (Overlay, alpha 1). On it scores 29.05 against 28.62 off,
                    // and the matcap behaves the same way — both are layers I
                    // can read but evidently do not reproduce correctly, rather
                    // than layers the source omits. `RETRO_GLASS_DEPTH=1` restores it.
                    std::env::var("RETRO_GLASS_DEPTH").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0),
                    std::env::var("RETRO_GLASS_FRESNEL").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0),
                    0.0,
                ],
            ])
        });
        let mut bar = Object3D::mesh(Mesh::new(
            CylinderGeometry::new(124.5, 124.5, bar_len, 8, 1, false, 0.0, std::f32::consts::TAU),
            glass,
        ));
        bar.position = bar_centre;
        // The source CylinderGeometry records width 176.64 / depth 249 — the
        // cross-section is an ellipse, not a circle. `CylinderGeometry(124.5,
        // 124.5, ..)` builds 249 x 249, so x is squashed to match. Skipping this
        // made every bar 41% wider than the original, which is why they
        // overlapped into one blob instead of leaving the gaps between slabs.
        const BAR_W: f32 = 176.64167557758233;
        const BAR_D: f32 = 249.0;
        bar.scale = Vector3::new(BAR_W / BAR_D, 1.0, 1.0);
        // rotation [0, 0, 45] — degrees, XYZ Euler order. Only Z is non-zero
        // here so the order does not bite, but it is the convention the source uses.
        bar.quaternion =
            Quaternion::from_euler_xyz(0.0, 0.0, 45.0_f32.to_radians());
        bar.name = format!("Cylinder {k}");
        scene.add(bar);
    }

    // ---- type ---------------------------------------------------------
    // Both are TextGeometry in the source, set in Neue Montreal — a CFF/OpenType
    // face, which this crate's TTF loader does not read (and `TtfFont::glyph_for`
    // returns outlines in font units while scaling the advance to ems, so its
    // output is not usable for layout as-is). The type is rasterised with the
    // caption font instead and mapped onto an unlit quad: the text is flat
    // (depth 0), white, unlit and square to an orthographic camera, so a
    // textured quad is pixel-equivalent apart from the letterforms themselves.
    //
    // Placement targets the world-space ink boxes measured off the live three.js
    // scene, which already fold in the CTA container's 2.2506 scale.
    let mut font = system_font();

    // "Chasing Sunsets" — fontSize 54, white.
    add_text(
        &mut scene,
        &mut font,
        &TextSpec {
            text: "Chasing Sunsets",
            wrap_fraction: None,
            font_size: 54.0,
            line_height: 1.1,
            center: Vector3::new(-1223.18, -272.62, 1340.43),
            ink_w: 866.9,
        },
    );
    // Body — fontSize 22, lineHeight 1.32, wrapping inside a 340.25-wide box:
    // 340.25 / 22 = 15.47 em, which breaks the line after "there's".
    add_text(
        &mut scene,
        &mut font,
        &TextSpec {
            text: "Sunsets are proof that there's beauty in endings.",
            wrap_fraction: Some(340.24984770876256 / 22.0),
            font_size: 22.0,
            line_height: 1.32,
            center: Vector3::new(-1357.26, -434.31, 1340.43),
            ink_w: 611.0,
        },
    );

    scene
}

/// Rasterise `text` white-on-transparent and hang it on an unlit quad centred on
/// the measured ink box at `center`.
///
/// The quad is sized from the **em**, not from the ink box: one em is
/// `font_size * CTA_SCALE` world units, which is exact, whereas fitting to a
/// measured ink height silently folds in whatever line spacing the substitute
/// face happens to use. `ink_w` is only the measured target, printed so the
/// error is visible in the run log.
/// One TextGeometry's parameters, as the document records them.
struct TextSpec {
    text: &'static str,
    /// Wrap width as a multiple of the em, i.e. the document's box width
    /// divided by its fontSize. `None` leaves the string on one line.
    wrap_fraction: Option<f32>,
    font_size: f32,
    line_height: f32,
    /// Centre of the ink box, measured off the live scene, in world units.
    center: Vector3,
    /// Measured ink width, used to anchor the left edge and to report error.
    ink_w: f32,
}

fn add_text(scene: &mut Scene, font: &mut CaptionFont, spec: &TextSpec) {
    let TextSpec { text, wrap_fraction, font_size, line_height, center, ink_w } = *spec;
    // Rasterise at a fixed pixel size, then scale the quad by the em ratio.
    const PX: f32 = 96.0;
    let wrap_px = wrap_fraction.map(|ems| PX * ems);
    let (tw, th, rgba) = raster_text(font, text, PX, wrap_px, line_height);
    if tw == 0 || th == 0 {
        return;
    }

    let scale = (font_size * CTA_SCALE) / PX;
    let quad_w = tw as f32 * scale;
    let quad_h = th as f32 * scale;
    println!(
        "  text {:?}: raster {tw}x{th}px -> {quad_w:.1}x{quad_h:.1} world (measured width {ink_w:.1}, err {:+.1}%)",
        &text[..text.len().min(24)],
        100.0 * (quad_w - ink_w) / ink_w
    );

    let tex = Arc::new(DataTexture::new(tw, th, TextureFormat::Rgba8UnormSrgb, rgba));
    let mut quad = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(quad_w, quad_h),
        Material::Basic(BasicMaterial {
            color: Color::WHITE,
            map: Some(tex),
            transparent: true,
            alpha_test: 0.01,
            side: 2,
            ..Default::default()
        }),
    ));
    // Anchor the left edge where the measured ink box starts, so the crop at the
    // frame edge lands the same way even if this face is a little wider.
    quad.position = Vector3::new(center.x - ink_w / 2.0 + quad_w / 2.0, center.y, center.z);
    quad.name = "Text".into();
    scene.add(quad);
}

/// White text on transparent, cropped to its ink box. Returns `(w, h, rgba)`.
fn raster_text(
    font: &mut CaptionFont,
    text: &str,
    size: f32,
    wrap_px: Option<f32>,
    line_height: f32,
) -> (u32, u32, Vec<u8>) {
    let metrics = font.metrics(size);
    // The source's lineHeight is a multiple of the type size, not of the face's
    // ascent+descent+gap — using the face's own figure stretches multi-line
    // blocks and, since the quad used to be fitted to that height, shrank the
    // whole block to compensate.
    let line_h = size * line_height;

    // Greedy word wrap at `wrap_px`, if asked for.
    let mut lines: Vec<String> = Vec::new();
    match wrap_px {
        None => lines.push(text.to_string()),
        Some(limit) => {
            let mut current = String::new();
            for word in text.split_whitespace() {
                let candidate = if current.is_empty() {
                    word.to_string()
                } else {
                    format!("{current} {word}")
                };
                if font.measure(&candidate, size, 0.0) > limit && !current.is_empty() {
                    lines.push(std::mem::take(&mut current));
                    current = word.to_string();
                } else {
                    current = candidate;
                }
            }
            if !current.is_empty() {
                lines.push(current);
            }
        }
    }

    // Generous canvas, then crop to ink.
    let cw = lines
        .iter()
        .map(|l| font.measure(l, size, 0.0))
        .fold(0.0_f32, f32::max)
        .ceil() as i32
        + (size as i32) * 2;
    let ch = (line_h * lines.len() as f32).ceil() as i32 + (size as i32) * 2;
    let (cw, ch) = (cw.max(1), ch.max(1));
    let mut cov = vec![0u8; (cw * ch) as usize];

    let pad = size as i32;
    for (i, line) in lines.iter().enumerate() {
        let baseline = pad + (metrics.ascent + line_h * i as f32) as i32;
        let mut pen = pad as f32;
        for c in line.chars() {
            let g = font.glyph(c, size);
            let (gw, gh) = (g.width as i32, g.height as i32);
            let (ox, oy, adv) = (g.offset_x, g.offset_y, g.advance);
            if !g.is_blank() {
                let gcov = g.coverage.clone();
                for row in 0..gh {
                    let y = baseline + oy + row;
                    if y < 0 || y >= ch {
                        continue;
                    }
                    for col in 0..gw {
                        let x = pen as i32 + ox + col;
                        if x < 0 || x >= cw {
                            continue;
                        }
                        let v = gcov[(row * gw + col) as usize];
                        let dst = &mut cov[(y * cw + x) as usize];
                        *dst = (*dst).max(v);
                    }
                }
            }
            pen += adv;
        }
    }

    // Crop to the ink box.
    let (mut x0, mut y0, mut x1, mut y1) = (cw, ch, -1, -1);
    for y in 0..ch {
        for x in 0..cw {
            if cov[(y * cw + x) as usize] > 0 {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }
    if x1 < x0 || y1 < y0 {
        return (0, 0, Vec::new());
    }
    let (ow, oh) = ((x1 - x0 + 1) as u32, (y1 - y0 + 1) as u32);

    // White, premultiplied by coverage into alpha. Row 0 is the top of the ink
    // box; PlaneGeometry's V runs bottom-up, so the rows are emitted flipped.
    let mut rgba = vec![0u8; (ow * oh * 4) as usize];
    for y in 0..oh {
        let src_y = y1 - y as i32;
        for x in 0..ow {
            let a = cov[(src_y * cw + (x0 + x as i32)) as usize];
            let o = ((y * ow + x) * 4) as usize;
            rgba[o] = 255;
            rgba[o + 1] = 255;
            rgba[o + 2] = 255;
            rgba[o + 3] = a;
        }
    }
    (ow, oh, rgba)
}

/// The source noise pass. `blendFunction: 16` is the postprocessing library's
/// SCREEN, not ADD: `out = 1 - (1 - base)(1 - n)`. Over a near-black frame the
/// two agree to well under a code value, which is why the additive version
/// measured correctly, but screen is what the document actually asks for.
/// Deterministic, so two runs produce identical bytes.
fn film_grain(rgba: &mut [u8], opacity: f32) {
    film_grain_seeded(rgba, opacity, 0x9e3779b9)
}

fn film_grain_seeded(rgba: &mut [u8], opacity: f32, seed: u32) {
    let mut state: u32 = seed | 1;
    for px in rgba.chunks_exact_mut(4) {
        // xorshift32
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let n = (state >> 8) as f32 / 16_777_216.0; // [0, 1)
        let src = n * opacity;
        for c in px.iter_mut().take(3) {
            let base = *c as f32 / 255.0;
            *c = ((1.0 - (1.0 - base) * (1.0 - src)) * 255.0).min(255.0) as u8;
        }
    }
}

/// The scene is set in Neue Montreal, which the source serves as CFF-flavoured
/// OpenType — the one thing `loaders::ttf` explicitly does not read. Converting
/// its outlines to quadratics (fontTools' cu2qu) yields a file the loader does
/// accept, and `RETRO_FONT` should point at that. The face is commercial, so
/// it is not vendored here; without it this falls back to the nearest
/// grotesque on the machine, which sets ~10% wide by comparison.
fn system_font() -> CaptionFont {
    let mut candidates: Vec<String> = Vec::new();
    if let Ok(p) = std::env::var("RETRO_FONT") {
        candidates.push(p);
    }
    candidates.extend(
        [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttf",
            "/Library/Fonts/Arial.ttf",
        ]
        .map(String::from),
    );
    for path in candidates {
        if let Ok(bytes) = std::fs::read(&path) {
            match CaptionFont::from_ttf_bytes(&bytes) {
                Ok(f) => {
                    println!("type set in {path}");
                    return f;
                }
                Err(e) => println!("could not parse {path}: {e:?}"),
            }
        }
    }
    println!("type set in the built-in face (no system grotesque parsed)");
    CaptionFont::ui()
}

// REVEALED
// ────────
// As saved, the black card sits at z = -189.7 and the sphere at z = -759.9, so
// the card is between the camera and the sphere and covers it completely: the
// sphere spans x [-238, 688] and y [-288, 638], entirely inside the card's
// x [-470, 761] / y [-453, 748]. The file therefore renders as a near-black
// field with white type — which is exactly what the source tool's own preview thumbnail
// and its editor viewport show, so this is the file's real appearance and not a
// gap in the reconstruction. The `_revealed` render moves the card back to
// z = -1400 and changes nothing else, so the sunset the scene is named for is
// visible behind the reeded glass.
