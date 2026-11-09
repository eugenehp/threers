//! Tone mapping: compressing linear HDR radiance into displayable range.
//!
//! The renderer runs these curves in WGSL; [`ToneMapping::apply`] runs the same
//! ones on the CPU, for the renderers without a shader (`SvgRenderer`) and the
//! path tracer's film, so it lives here rather than with the GPU code.

/// How the renderer compresses linear HDR radiance into displayable range.
///
/// Defaults to [`ToneMapping::None`], which matches the renderer's historical
/// behaviour — values above 1.0 hard-clip. That is fine for LDR content but
/// destroys hue in highlights: a clipped gold specular goes white, not gold.
/// Any scene using an HDR environment wants [`ToneMapping::AcesFilmic`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToneMapping {
    /// No curve; linear values are clamped to [0, 1] at output.
    #[default]
    None,
    /// Plain exposure scale, then clamp.
    Linear,
    /// ACES filmic approximation — rolls highlights off smoothly and keeps
    /// their hue, which is what makes bright metal read as metal.
    AcesFilmic,
    /// Khronos PBR Neutral — what glTF viewers use, and what to reach for when
    /// the authored colour is the point.
    ///
    /// Below the shoulder it is **almost the identity**: it subtracts a small
    /// black offset and otherwise leaves the value alone (0.18 comes out at
    /// 0.14, 0.45 at 0.432), so a material's colour survives to the screen
    /// instead of being restyled. [`AcesFilmic`](Self::AcesFilmic) lifts the
    /// same midtones hard — 0.18 to 0.267 — which is flattering on a lit scene
    /// and wrong on a colour chart.
    ///
    /// Only the top is compressed, and it desaturates as it compresses, so an
    /// over-exposed highlight resolves towards white the way an over-exposed
    /// highlight does. ACES, as the Narkowicz fit used here, is a *per-channel*
    /// curve with no cross-channel term at all, so a blown red stays a blown
    /// red rather than going white.
    Neutral,
    /// AgX, as three.js ships it.
    ///
    /// A long shoulder — it holds 3.0 down at 0.82 where ACES is already at
    /// 0.95 — so a very bright environment keeps detail in its highlights
    /// instead of flaring to a flat patch. It pays for that with its shadows,
    /// which it desaturates noticeably: a near-black saturated colour comes
    /// back closer to grey than it went in. That is the AgX look, not a defect.
    AgX,
}

impl ToneMapping {
    /// The mode index the WGSL side switches on.
    #[doc(hidden)]
    pub fn shader_mode(self) -> f32 {
        match self {
            ToneMapping::None => 0.0,
            ToneMapping::Linear => 1.0,
            ToneMapping::AcesFilmic => 2.0,
            ToneMapping::Neutral => 3.0,
            ToneMapping::AgX => 4.0,
        }
    }

    /// The same curves on the CPU, for the renderers that have no shader to run
    /// them in (`SvgRenderer`) and the path tracer's film. Takes an *exposed*
    /// linear colour, like the WGSL `apply_tone_mapping` does after it scales.
    pub fn apply(self, c: [f32; 3]) -> [f32; 3] {
        match self {
            ToneMapping::None => c,
            // three.js's `LinearToneMapping` clamps, as `linear_tonemap` does.
            ToneMapping::Linear => c.map(|x| x.clamp(0.0, 1.0)),
            ToneMapping::AcesFilmic => aces_cpu(c),
            ToneMapping::Neutral => neutral_cpu(c),
            ToneMapping::AgX => agx_cpu(c),
        }
    }
}

/// three.js r165 `ACESFilmicToneMapping` on an exposed colour, matching
/// `aces_tonemap` in the shader: the 1/0.6 pre-scale, the input matrix, Hill's
/// RRT + ODT fit, the output matrix, then a clamp. Column-major matrices as in
/// the GLSL, so the rows below are its columns.
fn aces_cpu(c: [f32; 3]) -> [f32; 3] {
    const INPUT: [[f32; 3]; 3] = [
        [0.59719, 0.07600, 0.02840],
        [0.35458, 0.90834, 0.13383],
        [0.04823, 0.01566, 0.83777],
    ];
    const OUTPUT: [[f32; 3]; 3] = [
        [1.60475, -0.10208, -0.00327],
        [-0.53108, 1.10813, -0.07276],
        [-0.07367, -0.00605, 1.07602],
    ];
    let mul = |m: [[f32; 3]; 3], v: [f32; 3]| -> [f32; 3] {
        [
            m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
            m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
            m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
        ]
    };
    let v = mul(INPUT, [c[0] / 0.6, c[1] / 0.6, c[2] / 0.6]);
    let fit =
        v.map(|x| (x * (x + 0.0245786) - 0.000090537) / (x * (0.983729 * x + 0.432951) + 0.238081));
    mul(OUTPUT, fit).map(|x| x.clamp(0.0, 1.0))
}

/// Khronos PBR Neutral, matching `neutral_tonemap` in the shader.
fn neutral_cpu(c: [f32; 3]) -> [f32; 3] {
    const START: f32 = 0.8 - 0.04;
    const DESAT: f32 = 0.15;
    let x = c[0].min(c[1]).min(c[2]);
    let offset = if x < 0.08 { x - 6.25 * x * x } else { 0.04 };
    let mut c = [c[0] - offset, c[1] - offset, c[2] - offset];
    let peak = c[0].max(c[1]).max(c[2]);
    if peak < START {
        return c;
    }
    let d = 1.0 - START;
    let new_peak = 1.0 - d * d / (peak + d - START);
    for v in c.iter_mut() {
        *v *= new_peak / peak.max(1e-6);
    }
    let g = 1.0 - 1.0 / (DESAT * (peak - new_peak) + 1.0);
    [
        c[0] + (new_peak - c[0]) * g,
        c[1] + (new_peak - c[1]) * g,
        c[2] + (new_peak - c[2]) * g,
    ]
}

/// AgX, matching `agx_tonemap` in the shader. Column-major 3x3s, applied as
/// `m * v`, so the rows below are the shader's columns.
fn agx_cpu(c: [f32; 3]) -> [f32; 3] {
    fn mul(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
        [
            m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
            m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
            m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
        ]
    }
    const TO_2020: [[f32; 3]; 3] = [
        [0.6274, 0.0691, 0.0164],
        [0.3293, 0.9195, 0.0880],
        [0.0433, 0.0113, 0.8956],
    ];
    const TO_SRGB: [[f32; 3]; 3] = [
        [1.6605, -0.1246, -0.0182],
        [-0.5876, 1.1329, -0.1006],
        [-0.0728, -0.0083, 1.1187],
    ];
    const INSET: [[f32; 3]; 3] = [
        [0.856_627_2, 0.137_318_97, 0.111_898_21],
        [0.095_121_24, 0.761_242, 0.076_799_42],
        [0.048_251_607, 0.101_439_04, 0.811_302_4],
    ];
    const OUTSET: [[f32; 3]; 3] = [
        [1.127_100_6, -0.141_329_76, -0.141_329_76],
        [-0.110_606_64, 1.157_823_7, -0.110_606_64],
        [-0.016_493_939, -0.016_493_939, 1.251_936_4],
    ];
    const MIN_EV: f32 = -12.473_93;
    const MAX_EV: f32 = 4.026_069;

    let mut v = mul(INSET, mul(TO_2020, c));
    for x in v.iter_mut() {
        *x = ((x.max(1e-10).log2() - MIN_EV) / (MAX_EV - MIN_EV)).clamp(0.0, 1.0);
        let (x2, x4) = (*x * *x, *x * *x * *x * *x);
        *x = 15.5 * x4 * x2 - 40.14 * x4 * *x + 31.96 * x4 - 6.868 * x2 * *x
            + 0.4298 * x2
            + 0.1191 * *x
            - 0.00232;
    }
    let mut v = mul(OUTSET, v);
    for x in v.iter_mut() {
        *x = x.max(0.0).powf(2.2);
    }
    let v = mul(TO_SRGB, v);
    [
        v[0].clamp(0.0, 1.0),
        v[1].clamp(0.0, 1.0),
        v[2].clamp(0.0, 1.0),
    ]
}
