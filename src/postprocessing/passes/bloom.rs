//! UnrealBloomPass and BloomPass.

use crate::math::Color;
use crate::postprocessing::context::{kinds, Dest, Fx, PassContext};
use crate::postprocessing::params::{assign_array, assign_f32, ParamInfo, ParamKind, ParamValue};
use crate::postprocessing::pass::{pass_common, Pass, PassState};
use crate::postprocessing::target::Target;
use crate::renderer::PostFxBlend;

/// JavaScript's `Math.round`, half up.
fn js_round(v: f32) -> u32 {
    (v + 0.5).floor().max(1.0) as u32
}

const N_MIPS: usize = 5;
const KERNEL_SIZES: [u32; N_MIPS] = [3, 5, 7, 9, 11];
const BLOOM_FACTORS: [f32; N_MIPS] = [1.0, 0.8, 0.6, 0.4, 0.2];

/// three.js `UnrealBloomPass(resolution, strength, radius, threshold)`.
///
/// The bright parts (a luminosity high pass) are blurred at five halving
/// resolutions with growing Gaussian kernels (3, 5, 7, 9 and 11 taps a side),
/// the mips are weighted by `bloomFactors` bent by `radius`, tinted, scaled by
/// `strength`, and the sum added over `readBuffer` — three.js's mip chain,
/// target for target.
#[derive(Debug)]
pub struct UnrealBloomPass {
    pub state: PassState,
    pub strength: f32,
    pub radius: f32,
    pub threshold: f32,
    pub resolution: [f32; 2],
    pub clear_color: Color,
    /// Five RGB tints, one per mip.
    pub tint_colors: [[f32; 3]; N_MIPS],
    width: u32,
    height: u32,
    bright: Option<Target>,
    horizontal: [Option<Target>; N_MIPS],
    vertical: [Option<Target>; N_MIPS],
}

impl UnrealBloomPass {
    pub fn new(width: u32, height: u32, strength: f32, radius: f32, threshold: f32) -> Self {
        Self {
            state: PassState::swapping(false),
            strength,
            radius,
            threshold,
            resolution: [width as f32, height as f32],
            clear_color: Color::BLACK,
            tint_colors: [[1.0; 3]; N_MIPS],
            width: width.max(1),
            height: height.max(1),
            bright: None,
            horizontal: Default::default(),
            vertical: Default::default(),
        }
    }

    /// The mips' sizes: half the pass's size, then halving, each rounded as
    /// three.js rounds.
    pub fn mip_sizes(&self) -> [(u32, u32); N_MIPS] {
        let mut sizes = [(0, 0); N_MIPS];
        let (mut x, mut y) = (js_round(self.width as f32 / 2.0), js_round(self.height as f32 / 2.0));
        for s in sizes.iter_mut() {
            *s = (x, y);
            x = js_round(x as f32 / 2.0);
            y = js_round(y as f32 / 2.0);
        }
        sizes
    }
}

impl Pass for UnrealBloomPass {
    pass_common!("UnrealBloomPass");

    fn set_size(&mut self, width: u32, height: u32) {
        self.width = width.max(1);
        self.height = height.max(1);
    }

    fn render(&mut self, ctx: &mut PassContext<'_, '_>, _write: &Target, read: &Target, _d: f32, _m: bool) {
        let sizes = self.mip_sizes();
        let transparent = [
            self.clear_color.r as f64,
            self.clear_color.g as f64,
            self.clear_color.b as f64,
            0.0,
        ];

        // Onto the screen three.js first copies readBuffer with a
        // MeshBasicMaterial and then adds the bloom over it; both go into a
        // stand-in and reach the screen once.
        let stand_in = if self.state.render_to_screen {
            let Some(s) = ctx.screen_stand_in() else {
                return;
            };
            ctx.clear(Dest::StandIn(&s), self.clear_color, 0.0);
            ctx.fx(Fx::basic_copy(read).blend(PostFxBlend::Overwrite), Dest::StandIn(&s));
            Some(s)
        } else {
            None
        };

        // 1. Extract the bright areas, at the first mip's size.
        let bright = ctx.target(&mut self.bright, sizes[0].0, sizes[0].1);
        ctx.fx(
            Fx::new(kinds::LUMINOSITY_HIGH_PASS)
                .input(read)
                .p2([self.threshold, 0.01, 0.0, 0.0])
                .clear_color(transparent),
            Dest::Target(&bright),
        );

        // 2. Blur each mip from the last.
        let mut input = bright;
        let mut verticals = Vec::with_capacity(N_MIPS);
        for i in 0..N_MIPS {
            let (w, h) = sizes[i];
            let k = KERNEL_SIZES[i] as f32;
            let hz = ctx.target(&mut self.horizontal[i], w, h);
            ctx.fx(
                Fx::new(kinds::BLOOM_BLUR)
                    .input(&input)
                    .p2([k, 1.0, 0.0, 0.0])
                    .clear_color(transparent),
                Dest::Target(&hz),
            );
            let vt = ctx.target(&mut self.vertical[i], w, h);
            ctx.fx(
                Fx::new(kinds::BLOOM_BLUR)
                    .input(&hz)
                    .p2([k, 0.0, 1.0, 0.0])
                    .clear_color(transparent),
                Dest::Target(&vt),
            );
            verticals.push(vt.clone());
            input = vt;
        }

        // 3. Composite the mips into the first horizontal target.
        let weights: Vec<[f32; 4]> = (0..N_MIPS)
            .map(|i| {
                let t = self.tint_colors[i];
                [t[0], t[1], t[2], BLOOM_FACTORS[i]]
            })
            .collect();
        ctx.renderer.set_postfx_kernel(&weights);
        let composite = ctx.target(&mut self.horizontal[0], sizes[0].0, sizes[0].1);
        ctx.fx(
            Fx::new(kinds::BLOOM_COMPOSITE)
                .input(&verticals[0])
                .depth(&verticals[1])
                .normal(&verticals[2])
                .linear_aux()
                .p2([self.strength, self.radius, 0.0, 3.0])
                .clear_color(transparent),
            Dest::Target(&composite),
        );
        ctx.fx(
            Fx::new(kinds::BLOOM_COMPOSITE)
                .input(&verticals[3])
                .depth(&verticals[4])
                .linear_aux()
                .p2([self.strength, self.radius, 3.0, 2.0])
                .blend(PostFxBlend::Add),
            Dest::Target(&composite),
        );

        // 4. Add it over the input — `AdditiveBlending`, straight alpha.
        let add = Fx::copy(&composite, 1.0).blend(PostFxBlend::AddAlpha);
        match stand_in {
            Some(s) => {
                ctx.fx(add, Dest::StandIn(&s));
                ctx.present_stand_in(&s);
            }
            None => ctx.fx(add, Dest::Target(read)),
        }
    }

    fn params(&self) -> Vec<ParamInfo> {
        vec![
            ParamInfo::prop("strength", ParamKind::Float),
            ParamInfo::prop("radius", ParamKind::Float),
            ParamInfo::prop("threshold", ParamKind::Float),
            ParamInfo::prop("resolution", ParamKind::Vec2),
            ParamInfo::prop("clearColor", ParamKind::Color),
            ParamInfo::prop("bloomTintColors", ParamKind::Floats),
        ]
    }

    fn get(&self, name: &str) -> Option<ParamValue> {
        Some(match name {
            "strength" => ParamValue::Float(self.strength),
            "radius" => ParamValue::Float(self.radius),
            "threshold" => ParamValue::Float(self.threshold),
            "resolution" => ParamValue::Vec2(self.resolution),
            "clearColor" => ParamValue::Color([self.clear_color.r, self.clear_color.g, self.clear_color.b]),
            "bloomTintColors" => ParamValue::Floats(self.tint_colors.iter().flatten().copied().collect()),
            _ => return None,
        })
    }

    fn set(&mut self, name: &str, value: ParamValue) -> bool {
        match name {
            "strength" => assign_f32(&mut self.strength, &value),
            "radius" => assign_f32(&mut self.radius, &value),
            "threshold" => assign_f32(&mut self.threshold, &value),
            "resolution" => assign_array(&mut self.resolution, &value),
            "clearColor" => value
                .as_array::<3>()
                .map(|c| self.clear_color = Color::new(c[0], c[1], c[2]))
                .is_some(),
            "bloomTintColors" => match value.as_floats() {
                Some(v) if v.len() >= 3 * N_MIPS => {
                    for (i, t) in self.tint_colors.iter_mut().enumerate() {
                        *t = [v[3 * i], v[3 * i + 1], v[3 * i + 2]];
                    }
                    true
                }
                _ => false,
            },
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------

/// three.js `ConvolutionShader.buildKernel(sigma)`.
pub fn convolution_kernel(sigma: f32) -> Vec<f32> {
    let sigma = sigma as f64;
    let max = 25usize;
    let mut size = 2 * (sigma * 3.0).ceil() as usize + 1;
    if size > max {
        size = max;
    }
    let half = (size as f64 - 1.0) * 0.5;
    let gauss = |x: f64| (-(x * x) / (2.0 * sigma * sigma)).exp();
    let mut values: Vec<f64> = (0..size).map(|i| gauss(i as f64 - half)).collect();
    let sum: f64 = values.iter().sum();
    for v in &mut values {
        *v /= sum;
    }
    values.into_iter().map(|v| v as f32).collect()
}

/// three.js `BloomPass(strength, kernelSize, sigma)`: a separable convolution
/// of `readBuffer` (steps of 1/512 of the frame, as three.js has them)
/// added back over it by `strength`. It draws into `readBuffer` only — as in
/// three.js, it puts nothing on the screen itself.
#[derive(Debug)]
pub struct BloomPass {
    pub state: PassState,
    pub strength: f32,
    pub kernel_size: u32,
    pub sigma: f32,
    kernel: Vec<f32>,
    width: u32,
    height: u32,
    x: Option<Target>,
    y: Option<Target>,
}

impl Default for BloomPass {
    fn default() -> Self {
        Self::new(1.0, 25, 4.0)
    }
}

impl BloomPass {
    pub fn new(strength: f32, kernel_size: u32, sigma: f32) -> Self {
        Self {
            state: PassState::swapping(false),
            strength,
            kernel_size: kernel_size.clamp(1, 128),
            sigma,
            kernel: convolution_kernel(sigma),
            width: 1,
            height: 1,
            x: None,
            y: None,
        }
    }
}

impl Pass for BloomPass {
    pass_common!("BloomPass");

    fn set_size(&mut self, width: u32, height: u32) {
        self.width = width.max(1);
        self.height = height.max(1);
    }

    fn render(&mut self, ctx: &mut PassContext<'_, '_>, _write: &Target, read: &Target, _d: f32, _m: bool) {
        let (w, h) = (self.width, self.height);
        let mut weights = vec![[0f32; 4]; 32];
        for (i, k) in self.kernel.iter().enumerate().take(128) {
            weights[i / 4][i % 4] = *k;
        }
        ctx.renderer.set_postfx_kernel(&weights);
        let step = 0.001953125;
        let n = self.kernel_size as f32;
        let x = ctx.target(&mut self.x, w, h);
        ctx.fx(
            Fx::new(kinds::CONVOLUTION).input(read).p2([step, 0.0, n, 0.0]),
            Dest::Target(&x),
        );
        let y = ctx.target(&mut self.y, w, h);
        ctx.fx(
            Fx::new(kinds::CONVOLUTION).input(&x).p2([0.0, step, n, 0.0]),
            Dest::Target(&y),
        );
        if self.state.clear {
            ctx.clear(Dest::Target(read), Color::BLACK, 0.0);
        }
        ctx.fx(
            Fx::new(kinds::COMBINE)
                .input(&y)
                .p2([self.strength, 0.0, 0.0, 0.0])
                .blend(PostFxBlend::AddAlpha),
            Dest::Target(read),
        );
    }

    fn params(&self) -> Vec<ParamInfo> {
        vec![ParamInfo::uniform("strength", ParamKind::Float)]
    }

    fn get(&self, name: &str) -> Option<ParamValue> {
        (name == "strength").then_some(ParamValue::Float(self.strength))
    }

    fn set(&mut self, name: &str, value: ParamValue) -> bool {
        name == "strength" && assign_f32(&mut self.strength, &value)
    }
}
