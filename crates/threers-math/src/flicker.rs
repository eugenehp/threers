//! A flame's flicker, as one signal everything near a flame can share: the
//! flame's stretch and brightness, the light it gives, the glow it casts.
//! A few unrelated sines for the fast flutter, and a slower draught that
//! now and then shakes it harder; `seed` sets one flame apart from another.
//!
//! ```
//! use threers_math::flicker::{flame, sway};
//! let f = flame(1.25, 0.0);
//! assert!((0.8..=1.2).contains(&f));
//! let (x, z) = sway(1.25, 0.0);
//! assert!(x.abs() <= 1.0 && z.abs() <= 1.0);
//! ```

/// The flame's brightness and stretch at `time` seconds: about 1, from 0.8
/// as it gutters to 1.2 as it flares.
pub fn flame(time: f32, seed: f32) -> f32 {
    let t = time + seed * 17.3;
    let fast =
        0.035 * (t * 11.7).sin() + 0.025 * (t * 17.3 + 1.3).sin() + 0.02 * (t * 29.1 + 2.9).sin();
    let draught = 0.5 + 0.5 * (t * 0.37).sin() * (t * 0.61 + 0.7).sin();
    let gust = draught * draught * (0.06 * (t * 7.3).sin() + 0.04 * (t * 13.9 + 0.4).sin());
    (1.0 + fast + gust).clamp(0.8, 1.2)
}

/// Which way the flame leans at `time`: sideways and towards or away from
/// the viewer, each within ±1.
pub fn sway(time: f32, seed: f32) -> (f32, f32) {
    let t = time + seed * 9.1;
    let draught = 0.5 + 0.5 * (t * 0.43).sin();
    (
        draught * (0.6 * (t * 1.3).sin() + 0.4 * (t * 3.1).sin()),
        draught * (0.6 * (t * 1.1 + 1.0).cos() + 0.4 * (t * 2.7).sin()),
    )
}

/// The volumetric flame's gas, in WGSL, shared by the 3D candle
/// (`threers::extras::candle`) and the UI's (`threers_ui::candle`), each
/// adding its own way through it. `flame_gas(q, t, flicker, lean, seed)` is
/// the light a point of the flame gives off: `q.xz` across it (-1..1),
/// `q.y` up it (0 at the base, 1 at the tip as it burns steadily), at time
/// `t`, stretched by [`flame`]'s `flicker`, leaned by [`sway`]'s `lean`.
/// A teardrop pinched at the wick, widest a quarter of the way up and drawn
/// out to a point, pushed about by rising noise more the higher it goes;
/// its soot glowing as a black body by its temperature — white-yellow in
/// the core, deep orange at the rim and the tip — with a dark cone of
/// unburnt vapour round the wick, a thin translucent blue cup at the base,
/// and a faint orange haze just outside. Linear light, brighter than 1 in
/// the core: tone map it (the core then reads white-hot).
pub const FLAME_WGSL: &str = r#"
fn flame_hash(p: vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453);
}

fn flame_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = mix(flame_hash(i), flame_hash(i + vec3<f32>(1.0, 0.0, 0.0)), u.x);
    let b = mix(flame_hash(i + vec3<f32>(0.0, 1.0, 0.0)), flame_hash(i + vec3<f32>(1.0, 1.0, 0.0)), u.x);
    let c = mix(flame_hash(i + vec3<f32>(0.0, 0.0, 1.0)), flame_hash(i + vec3<f32>(1.0, 0.0, 1.0)), u.x);
    let d = mix(flame_hash(i + vec3<f32>(0.0, 1.0, 1.0)), flame_hash(i + vec3<f32>(1.0, 1.0, 1.0)), u.x);
    return mix(mix(a, b, u.y), mix(c, d, u.y), u.z);
}

fn flame_fbm(p: vec3<f32>) -> f32 {
    return 0.55 * flame_noise(p) + 0.3 * flame_noise(p * 2.03 + 1.7) + 0.15 * flame_noise(p * 4.01 + 3.1);
}

fn flame_blackbody(k: f32) -> vec3<f32> {
    let t = k / 100.0;
    let g = clamp(0.390081579 * log(t) - 0.631841444, 0.0, 1.0);
    let b = select(clamp(0.543206789 * log(max(t - 10.0, 1.0)) - 1.19625409, 0.0, 1.0), 0.0, t <= 19.0);
    return pow(vec3<f32>(1.0, g, b), vec3<f32>(2.2));
}

fn flame_gas(q: vec3<f32>, t: f32, flicker: f32, lean_in: vec2<f32>, seed: f32) -> vec3<f32> {
    // Up the flame 0..1, stretched as it flares.
    let y = q.y / (0.86 + 0.14 * flicker);
    if (y <= 0.0 || y >= 1.08) {
        return vec3<f32>(0.0);
    }
    // Hot gas rising: noise drawn upward, faster than the flame grows, its
    // pull on the gas growing with height (the base is held by the wick;
    // the tip wanders).
    let rise = vec3<f32>(q.x * 2.1, y * 2.4 - t * 3.3, q.z * 2.1) + vec3<f32>(seed * 3.7);
    let n = flame_fbm(rise) - 0.5;
    let m = flame_fbm(rise.zyx + vec3<f32>(7.3, 0.0, 2.1)) - 0.5;
    let lean = lean_in * vec2<f32>(0.14, 0.1) * pow(max(y, 1e-4), 1.5);
    let wander = vec2<f32>(n, m) * 0.34 * pow(max(y, 1e-4), 2.2);
    let p = q.xz - lean - wander;
    // A teardrop: pinched where it leaves the wick, widest a third of the
    // way up, drawn out to a point; the noise narrows and swells it near
    // the tip, where a real flame necks and lets go.
    // (No `pow` of 0: Metal's fast maths makes NaN of it.)
    let r = 0.94 * pow(max(y, 1e-4), 0.7) * pow(max(1.0 - y, 1e-4), 1.15) / 0.293 * (1.0 + 0.5 * n * y * y);
    let d = length(p) / max(r, 1e-3);
    // The luminous body: glowing soot, soft at its edge, from just above
    // the blue base to the tip, which cools and thins.
    var body = 1.0 - smoothstep(0.5, 1.04, d);
    // It wraps round the dark cone: it starts low at its edge and higher
    // in the middle, a rounded bottom over the blue.
    let onset = 0.13 + 0.18 * (1.0 - smoothstep(0.3, 1.0, d));
    body = body * smoothstep(onset - 0.06, onset + 0.14, y) * (1.0 - 0.7 * smoothstep(0.72, 1.04, y));
    // The dark zone: unburnt vapour round the wick, a cone of little light.
    let dark = (1.0 - smoothstep(0.0, 0.6, d)) * (1.0 - smoothstep(0.16, 0.44, y));
    body = body * (1.0 - 0.9 * dark);
    // Hottest in the middle of the body, cooler out to the edge and up to
    // the tip: white-yellow at the core, deep orange at the rim.
    let core = pow(max(1.0 - d, 1e-4), 0.65);
    let kelvin = 1300.0 + 1500.0 * core * smoothstep(0.12, 0.5, y) - 380.0 * smoothstep(0.6, 1.0, y);
    let glow = flame_blackbody(kelvin) * body * (1.4 + 7.0 * core);
    // The blue cup: a thin sheet of burning vapour round the base, brightest
    // a little above the wick, fading out below and above.
    let k = (d - 0.88) / 0.08;
    let shell = exp(-k * k);
    let cup = shell * smoothstep(0.0, 0.05, y) * (1.0 - smoothstep(0.07, 0.18, y));
    // A faint orange haze just outside the body.
    let haze = exp(-max(d - 0.95, 0.0) * 7.0) * smoothstep(0.2, 0.5, y) * (1.0 - smoothstep(0.85, 1.08, y)) * select(0.0, 1.0, d > 0.8);
    return glow + vec3<f32>(0.1, 0.28, 1.0) * cup * 0.8 + vec3<f32>(1.0, 0.36, 0.07) * haze * 0.12;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_flicker_stays_in_range_and_differs_by_seed() {
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for i in 0..20_000 {
            let f = flame(i as f32 * 0.01, 0.0);
            lo = lo.min(f);
            hi = hi.max(f);
            let (x, z) = sway(i as f32 * 0.01, 0.0);
            assert!(x.abs() <= 1.0 && z.abs() <= 1.0);
        }
        assert!(
            (0.8..0.95).contains(&lo) && (1.05..=1.2).contains(&hi),
            "{lo} {hi}"
        );
        assert_ne!(flame(3.0, 0.0), flame(3.0, 1.0));
    }
}
