//! Passes that render the scene their own way: SSAARenderPass and
//! TAARenderPass (jittered accumulation), RenderPixelatedPass,
//! RenderTransitionPass, and AdaptiveToneMappingPass.

use std::sync::Arc;

use super::basic::out;
use crate::materials::{Material, NormalMaterial};
use crate::math::Color;
use crate::postprocessing::context::{kinds, Dest, Fx, PassContext, SceneDraw};
use crate::postprocessing::params::{assign_bool, assign_f32, ParamInfo, ParamKind, ParamValue};
use crate::postprocessing::pass::{pass_common, Pass, PassState};
use crate::postprocessing::target::Target;
use crate::postprocessing::upload;
use crate::renderer::PostFxBlend;
use crate::textures::Texture;

/// three.js's sample patterns, in sixteenths of a pixel, for `2^level`
/// samples.
pub const JITTER_VECTORS: [&[[i8; 2]]; 6] = [
    &[[0, 0]],
    &[[4, 4], [-4, -4]],
    &[[-2, -6], [6, -2], [-6, 2], [2, 6]],
    &[[1, -3], [-1, 3], [5, 1], [-3, -5], [-5, 5], [-7, -1], [3, 7], [7, -7]],
    &[
        [1, 1], [-1, -3], [-3, 2], [4, -1], [-5, -2], [2, 5], [5, 3], [3, -5],
        [-2, 6], [0, -7], [-4, -6], [-6, 4], [-8, 0], [7, -4], [6, 7], [-7, -8],
    ],
    &[
        [-4, -7], [-7, -5], [-3, -5], [-5, -4], [-1, -4], [-2, -2], [-6, -1], [-4, 0],
        [-7, 1], [-1, 2], [-6, 3], [-3, 3], [-7, 6], [-3, 6], [-5, 7], [-1, 7],
        [5, -7], [1, -6], [6, -5], [4, -4], [2, -3], [7, -2], [1, -1], [4, -1],
        [2, 1], [6, 2], [0, 4], [4, 4], [2, 5], [7, 5], [5, 6], [3, 7],
    ],
];

/// The jittered renders of SSAARenderPass, summed into `dest` with the
/// weights three.js gives them.
#[allow(clippy::too_many_arguments)]
fn accumulate_ssaa(
    ctx: &mut PassContext<'_, '_>,
    scene: usize,
    camera: usize,
    sample_level: u32,
    unbiased: bool,
    sample: &Target,
    dest: Dest<'_>,
) {
    let jitters = JITTER_VECTORS[sample_level.min(5) as usize];
    let base = 1.0 / jitters.len() as f32;
    let rounding = 1.0 / 32.0;
    for (i, j) in jitters.iter().enumerate() {
        let mut weight = base;
        if unbiased {
            weight += rounding * (-0.5 + (i as f32 + 0.5) / jitters.len() as f32);
        }
        ctx.render_scene(
            Dest::Target(sample),
            &SceneDraw {
                scene,
                camera,
                view_offset: Some((j[0] as f32 * 0.0625, j[1] as f32 * 0.0625)),
                ..SceneDraw::default()
            },
        );
        if i == 0 {
            ctx.clear(dest, Color::BLACK, 0.0);
        }
        // `premultipliedAlpha` + `AdditiveBlending`: ONE, ONE.
        ctx.fx(Fx::copy(sample, weight).blend(PostFxBlend::Add), dest);
    }
}

/// three.js `SSAARenderPass(scene, camera)`: `2^sampleLevel` renders with
/// sub-pixel camera offsets, averaged.
#[derive(Debug)]
pub struct SsaaRenderPass {
    pub state: PassState,
    pub scene: usize,
    pub camera: usize,
    pub sample_level: u32,
    pub unbiased: bool,
    sample: Option<Target>,
}

impl SsaaRenderPass {
    pub fn new(scene: usize, camera: usize) -> Self {
        Self {
            state: PassState::default(),
            scene,
            camera,
            sample_level: 4,
            unbiased: true,
            sample: None,
        }
    }
}

fn ssaa_render(
    ctx: &mut PassContext<'_, '_>,
    state: &PassState,
    scene: usize,
    camera: usize,
    sample_level: u32,
    unbiased: bool,
    sample_slot: &mut Option<Target>,
    write: &Target,
    read: &Target,
) {
    let sample = ctx.target(sample_slot, read.width(), read.height());
    if state.render_to_screen {
        // Each sample is added onto the screen; sum them in a stand-in.
        let Some(s) = ctx.screen_stand_in() else {
            return;
        };
        accumulate_ssaa(ctx, scene, camera, sample_level, unbiased, &sample, Dest::StandIn(&s));
        ctx.present_stand_in(&s);
    } else {
        accumulate_ssaa(ctx, scene, camera, sample_level, unbiased, &sample, Dest::Target(write));
    }
}

fn ssaa_params() -> Vec<ParamInfo> {
    vec![
        ParamInfo::prop("sampleLevel", ParamKind::Int),
        ParamInfo::prop("unbiased", ParamKind::Bool),
    ]
}

impl Pass for SsaaRenderPass {
    pass_common!("SSAARenderPass");

    fn render(&mut self, ctx: &mut PassContext<'_, '_>, write: &Target, read: &Target, _d: f32, _m: bool) {
        ssaa_render(
            ctx,
            &self.state,
            self.scene,
            self.camera,
            self.sample_level,
            self.unbiased,
            &mut self.sample,
            write,
            read,
        );
    }

    fn params(&self) -> Vec<ParamInfo> {
        ssaa_params()
    }

    fn get(&self, name: &str) -> Option<ParamValue> {
        match name {
            "sampleLevel" => Some(ParamValue::Int(self.sample_level as i32)),
            "unbiased" => Some(ParamValue::Bool(self.unbiased)),
            _ => None,
        }
    }

    fn set(&mut self, name: &str, value: ParamValue) -> bool {
        match name {
            "sampleLevel" => value.as_i32().map(|v| self.sample_level = v.max(0) as u32).is_some(),
            "unbiased" => assign_bool(&mut self.unbiased, &value),
            _ => false,
        }
    }

    fn scene_slots(&self) -> Vec<(usize, usize)> {
        vec![(self.scene, self.camera)]
    }

    fn set_scene_slot(&mut self, n: usize, scene: usize, camera: usize) {
        if n == 0 {
            self.scene = scene;
            self.camera = camera;
        }
    }
}

/// three.js `TAARenderPass(scene, camera)`: SSAA while the view moves
/// (`accumulate` off); with `accumulate` on, one more jittered sample a
/// frame into a running average over thirty-two.
#[derive(Debug)]
pub struct TaaRenderPass {
    pub state: PassState,
    pub scene: usize,
    pub camera: usize,
    pub sample_level: u32,
    pub unbiased: bool,
    pub accumulate: bool,
    pub accumulate_index: i32,
    sample: Option<Target>,
    hold: Option<Target>,
}

impl TaaRenderPass {
    pub fn new(scene: usize, camera: usize) -> Self {
        Self {
            state: PassState::default(),
            scene,
            camera,
            sample_level: 0,
            unbiased: true,
            accumulate: false,
            accumulate_index: -1,
            sample: None,
            hold: None,
        }
    }
}

impl Pass for TaaRenderPass {
    pass_common!("TAARenderPass");

    fn render(&mut self, ctx: &mut PassContext<'_, '_>, write: &Target, read: &Target, _d: f32, _m: bool) {
        if !self.accumulate {
            ssaa_render(
                ctx,
                &self.state,
                self.scene,
                self.camera,
                self.sample_level,
                self.unbiased,
                &mut self.sample,
                write,
                read,
            );
            self.accumulate_index = -1;
            return;
        }
        let jitters = JITTER_VECTORS[5];
        let (w, h) = read.size();
        let sample = ctx.target(&mut self.sample, w, h);
        let hold = ctx.target(&mut self.hold, w, h);
        if self.accumulate_index == -1 {
            // three.js renders this first average to the screen when the pass
            // is the last; it is the history, so it goes where it is kept.
            accumulate_ssaa(ctx, self.scene, self.camera, self.sample_level, self.unbiased, &sample, Dest::Target(&hold));
            self.accumulate_index = 0;
        }
        let weight = 1.0 / jitters.len() as f32;
        if self.accumulate_index >= 0 && (self.accumulate_index as usize) < jitters.len() {
            let per_frame = 1u32 << self.sample_level.min(5);
            for _ in 0..per_frame {
                let j = jitters[self.accumulate_index as usize];
                ctx.render_scene(
                    Dest::Target(write),
                    &SceneDraw {
                        scene: self.scene,
                        camera: self.camera,
                        view_offset: Some((j[0] as f32 * 0.0625, j[1] as f32 * 0.0625)),
                        ..SceneDraw::default()
                    },
                );
                if self.accumulate_index == 0 {
                    ctx.clear(Dest::Target(&sample), Color::BLACK, 0.0);
                }
                ctx.fx(Fx::copy(write, weight).blend(PostFxBlend::Add), Dest::Target(&sample));
                self.accumulate_index += 1;
                if self.accumulate_index as usize >= jitters.len() {
                    break;
                }
            }
        }
        let accumulation = self.accumulate_index as f32 * weight;
        if accumulation > 0.0 {
            ctx.clear(Dest::Target(write), Color::BLACK, 0.0);
            ctx.fx(Fx::copy(&sample, 1.0).blend(PostFxBlend::Add), Dest::Target(write));
        }
        if accumulation < 1.0 {
            ctx.fx(Fx::copy(&hold, 1.0 - accumulation).blend(PostFxBlend::Add), Dest::Target(write));
        }
    }

    fn params(&self) -> Vec<ParamInfo> {
        let mut p = ssaa_params();
        p.push(ParamInfo::prop("accumulate", ParamKind::Bool));
        p.push(ParamInfo::prop("accumulateIndex", ParamKind::Int));
        p
    }

    fn get(&self, name: &str) -> Option<ParamValue> {
        match name {
            "sampleLevel" => Some(ParamValue::Int(self.sample_level as i32)),
            "unbiased" => Some(ParamValue::Bool(self.unbiased)),
            "accumulate" => Some(ParamValue::Bool(self.accumulate)),
            "accumulateIndex" => Some(ParamValue::Int(self.accumulate_index)),
            _ => None,
        }
    }

    fn set(&mut self, name: &str, value: ParamValue) -> bool {
        match name {
            "sampleLevel" => value.as_i32().map(|v| self.sample_level = v.max(0) as u32).is_some(),
            "unbiased" => assign_bool(&mut self.unbiased, &value),
            "accumulate" => assign_bool(&mut self.accumulate, &value),
            "accumulateIndex" => value.as_i32().map(|v| self.accumulate_index = v).is_some(),
            _ => false,
        }
    }

    fn scene_slots(&self) -> Vec<(usize, usize)> {
        vec![(self.scene, self.camera)]
    }

    fn set_scene_slot(&mut self, n: usize, scene: usize, camera: usize) {
        if n == 0 {
            self.scene = scene;
            self.camera = camera;
        }
    }
}

// ---------------------------------------------------------------------------

/// three.js `RenderPixelatedPass(pixelSize, scene, camera, options)`: the
/// scene at `1/pixelSize` resolution, scaled up nearest, with depth and
/// normal edges picked out.
#[derive(Debug)]
pub struct RenderPixelatedPass {
    pub state: PassState,
    pub scene: usize,
    pub camera: usize,
    pub pixel_size: u32,
    pub normal_edge_strength: f32,
    pub depth_edge_strength: f32,
    width: u32,
    height: u32,
    normal_material: Arc<Material>,
    beauty: Option<Target>,
    normal: Option<Target>,
}

impl RenderPixelatedPass {
    pub fn new(pixel_size: u32, scene: usize, camera: usize) -> Self {
        Self {
            state: PassState::default(),
            scene,
            camera,
            pixel_size: pixel_size.max(1),
            normal_edge_strength: 0.3,
            depth_edge_strength: 0.4,
            width: 1,
            height: 1,
            normal_material: Arc::new(Material::Normal(NormalMaterial::new())),
            beauty: None,
            normal: None,
        }
    }

    fn render_resolution(&self) -> (u32, u32) {
        (
            (self.width / self.pixel_size).max(1),
            (self.height / self.pixel_size).max(1),
        )
    }
}

impl Pass for RenderPixelatedPass {
    pass_common!("RenderPixelatedPass");

    fn set_size(&mut self, width: u32, height: u32) {
        self.width = width.max(1);
        self.height = height.max(1);
    }

    fn render(&mut self, ctx: &mut PassContext<'_, '_>, write: &Target, _read: &Target, _d: f32, _m: bool) {
        let (rw, rh) = self.render_resolution();
        let beauty = ctx.target(&mut self.beauty, rw, rh);
        let draw = SceneDraw {
            scene: self.scene,
            camera: self.camera,
            ..SceneDraw::default()
        };
        ctx.render_scene(Dest::Target(&beauty), &draw);
        let normal = ctx.target(&mut self.normal, rw, rh);
        ctx.render_scene(
            Dest::Target(&normal),
            &SceneDraw {
                override_material: Some(self.normal_material.clone()),
                ..draw
            },
        );
        let dest = out(&self.state, write);
        if self.state.clear && !self.state.render_to_screen {
            ctx.clear(dest, Color::BLACK, 0.0);
        }
        ctx.fx(
            Fx::new(kinds::PIXELATED)
                .input(&beauty)
                .hw_depth(&beauty)
                .normal(&normal)
                .p2([self.normal_edge_strength, self.depth_edge_strength, 0.0, 0.0])
                .p3([rw as f32, rh as f32, 1.0 / rw as f32, 1.0 / rh as f32])
                .blend(PostFxBlend::Overwrite),
            dest,
        );
    }

    fn params(&self) -> Vec<ParamInfo> {
        vec![
            ParamInfo::prop("pixelSize", ParamKind::Int),
            ParamInfo::prop("normalEdgeStrength", ParamKind::Float),
            ParamInfo::prop("depthEdgeStrength", ParamKind::Float),
        ]
    }

    fn get(&self, name: &str) -> Option<ParamValue> {
        match name {
            "pixelSize" => Some(ParamValue::Int(self.pixel_size as i32)),
            "normalEdgeStrength" => Some(ParamValue::Float(self.normal_edge_strength)),
            "depthEdgeStrength" => Some(ParamValue::Float(self.depth_edge_strength)),
            _ => None,
        }
    }

    fn set(&mut self, name: &str, value: ParamValue) -> bool {
        match name {
            "pixelSize" => value.as_i32().map(|v| self.pixel_size = v.max(1) as u32).is_some(),
            "normalEdgeStrength" => assign_f32(&mut self.normal_edge_strength, &value),
            "depthEdgeStrength" => assign_f32(&mut self.depth_edge_strength, &value),
            _ => false,
        }
    }

    fn scene_slots(&self) -> Vec<(usize, usize)> {
        vec![(self.scene, self.camera)]
    }

    fn set_scene_slot(&mut self, n: usize, scene: usize, camera: usize) {
        if n == 0 {
            self.scene = scene;
            self.camera = camera;
        }
    }
}

// ---------------------------------------------------------------------------

/// three.js `RenderTransitionPass(sceneA, cameraA, sceneB, cameraB)`:
/// `mixRatio` 1 shows A, 0 shows B; with `useTexture` the cut follows a
/// texture's red channel.
#[derive(Debug)]
pub struct RenderTransitionPass {
    pub state: PassState,
    pub scene_a: usize,
    pub camera_a: usize,
    pub scene_b: usize,
    pub camera_b: usize,
    pub mix_ratio: f32,
    pub threshold: f32,
    pub use_texture: bool,
    pub texture: Option<Arc<Texture>>,
    rt_a: Option<Target>,
    rt_b: Option<Target>,
    uploaded: Option<(u64, Target)>,
}

impl RenderTransitionPass {
    pub fn new(scene_a: usize, camera_a: usize, scene_b: usize, camera_b: usize) -> Self {
        Self {
            state: PassState::default(),
            scene_a,
            camera_a,
            scene_b,
            camera_b,
            mix_ratio: 0.0,
            threshold: 0.1,
            use_texture: true,
            texture: None,
            rt_a: None,
            rt_b: None,
            uploaded: None,
        }
    }
}

impl Pass for RenderTransitionPass {
    pass_common!("RenderTransitionPass");

    fn render(&mut self, ctx: &mut PassContext<'_, '_>, write: &Target, read: &Target, _d: f32, _m: bool) {
        let dest = out(&self.state, write);
        let draw = |scene, camera| SceneDraw {
            scene,
            camera,
            ..SceneDraw::default()
        };
        if self.mix_ratio == 0.0 {
            ctx.render_scene(dest, &draw(self.scene_b, self.camera_b));
            return;
        }
        if self.mix_ratio == 1.0 {
            ctx.render_scene(dest, &draw(self.scene_a, self.camera_a));
            return;
        }
        let (w, h) = read.size();
        let a = ctx.target(&mut self.rt_a, w, h);
        ctx.render_scene(Dest::Target(&a), &draw(self.scene_a, self.camera_a));
        let b = ctx.target(&mut self.rt_b, w, h);
        ctx.render_scene(Dest::Target(&b), &draw(self.scene_b, self.camera_b));
        let mut fx = Fx::new(kinds::TRANSITION)
            .input(&a)
            .normal(&b)
            .linear_aux()
            .p2([self.mix_ratio, self.threshold, if self.use_texture { 1.0 } else { 0.0 }, 0.0]);
        if self.use_texture {
            if let Some(tex) = &self.texture {
                if let Some(t) = upload::texture_target(ctx.renderer, tex, &mut self.uploaded) {
                    fx = fx.depth(&t);
                }
            }
        }
        ctx.fx(fx, dest);
    }

    fn params(&self) -> Vec<ParamInfo> {
        vec![
            ParamInfo::uniform("mixRatio", ParamKind::Float),
            ParamInfo::uniform("threshold", ParamKind::Float),
            ParamInfo::uniform("useTexture", ParamKind::Int),
            ParamInfo::uniform("tMixTexture", ParamKind::Texture),
        ]
    }

    fn get(&self, name: &str) -> Option<ParamValue> {
        match name {
            "mixRatio" => Some(ParamValue::Float(self.mix_ratio)),
            "threshold" => Some(ParamValue::Float(self.threshold)),
            "useTexture" => Some(ParamValue::Int(self.use_texture as i32)),
            "tMixTexture" => Some(ParamValue::Texture(self.texture.clone())),
            _ => None,
        }
    }

    fn set(&mut self, name: &str, value: ParamValue) -> bool {
        match (name, value) {
            ("mixRatio", v) => assign_f32(&mut self.mix_ratio, &v),
            ("threshold", v) => assign_f32(&mut self.threshold, &v),
            ("useTexture", v) => assign_bool(&mut self.use_texture, &v),
            ("tMixTexture", ParamValue::Texture(t)) => {
                self.texture = t;
                true
            }
            _ => false,
        }
    }

    fn scene_slots(&self) -> Vec<(usize, usize)> {
        vec![(self.scene_a, self.camera_a), (self.scene_b, self.camera_b)]
    }

    fn set_scene_slot(&mut self, n: usize, scene: usize, camera: usize) {
        match n {
            0 => {
                self.scene_a = scene;
                self.camera_a = camera;
            }
            1 => {
                self.scene_b = scene;
                self.camera_b = camera;
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------

/// `AdaptiveToneMappingPass(adaptive, resolution)` as three.js shipped it
/// until r152 (r153 removed it): the luminance of `readBuffer`, averaged down
/// to one value, adapted towards over time from last frame's, and the frame
/// tone mapped against it (Reinhard's modified operator). Not adaptive, it
/// tone maps against `averageLuminance`.
///
/// three.js reads the average from the luminance target's 1×1 mip; here it is
/// halved down to 1×1 by bilinear taps between texels, the same average.
#[derive(Debug)]
pub struct AdaptiveToneMappingPass {
    pub state: PassState,
    pub adaptive: bool,
    pub resolution: u32,
    pub middle_grey: f32,
    pub min_luminance: f32,
    pub max_luminance: f32,
    pub average_luminance: f32,
    pub adaption_rate: f32,
    needs_init: bool,
    luminance: Option<Target>,
    previous: Option<Target>,
    current: Option<Target>,
    halvings: Vec<Option<Target>>,
}

impl AdaptiveToneMappingPass {
    pub fn new(adaptive: bool, resolution: u32) -> Self {
        Self {
            state: PassState::default(),
            adaptive,
            resolution: resolution.max(1),
            middle_grey: 0.6,
            min_luminance: 0.01,
            max_luminance: 16.0,
            average_luminance: 1.0,
            adaption_rate: 1.0,
            needs_init: true,
            luminance: None,
            previous: None,
            current: None,
            halvings: Vec::new(),
        }
    }

    /// three.js `reset`: fresh (zeroed) luminance targets.
    pub fn reset(&mut self) {
        self.needs_init = true;
    }
}

impl Pass for AdaptiveToneMappingPass {
    pass_common!("AdaptiveToneMappingPass");

    fn render(&mut self, ctx: &mut PassContext<'_, '_>, write: &Target, read: &Target, delta: f32, _m: bool) {
        let n = self.resolution;
        if self.needs_init {
            // New targets, zeroed: three.js's start from zero luminance too.
            for slot in [&mut self.luminance, &mut self.previous, &mut self.current] {
                let t = ctx.target(slot, n, n);
                ctx.clear(Dest::Target(&t), Color::BLACK, 0.0);
            }
            self.halvings.clear();
            let mut s = n;
            while s > 1 {
                s = (s >> 1).max(1);
                self.halvings.push(None);
            }
            self.needs_init = false;
        }
        let luminance = ctx.target(&mut self.luminance, n, n);
        if self.adaptive {
            let current = ctx.target(&mut self.current, n, n);
            let previous = ctx.target(&mut self.previous, n, n);
            ctx.fx(Fx::new(kinds::LUMINOSITY).input(read), Dest::Target(&current));
            let mut average = current;
            let mut s = n;
            for slot in self.halvings.iter_mut() {
                s = (s >> 1).max(1);
                let t = ctx.target(slot, s, s);
                ctx.fx(Fx::new(kinds::HALVE).input(&average), Dest::Target(&t));
                average = t;
            }
            ctx.fx(
                Fx::new(kinds::ADAPT_LUMINANCE)
                    .input(&average)
                    .normal(&previous)
                    .p2([self.min_luminance, delta, self.adaption_rate, 0.0]),
                Dest::Target(&luminance),
            );
            ctx.fx(Fx::copy(&luminance, 1.0), Dest::Target(&previous));
        }
        let mut fx = Fx::new(kinds::TONE_MAP)
            .input(read)
            .p2([self.middle_grey, self.min_luminance, self.max_luminance, self.average_luminance])
            .p3([if self.adaptive { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0]);
        if self.adaptive {
            fx = fx.normal(&luminance);
        }
        let dest = out(&self.state, write);
        if self.state.clear && !self.state.render_to_screen {
            ctx.clear(dest, Color::BLACK, 0.0);
        }
        ctx.fx(fx.blend(PostFxBlend::Overwrite), dest);
    }

    fn params(&self) -> Vec<ParamInfo> {
        vec![
            ParamInfo::prop("adaptive", ParamKind::Bool),
            ParamInfo::prop("resolution", ParamKind::Int),
            ParamInfo::prop("middleGrey", ParamKind::Float),
            ParamInfo::prop("minLuminance", ParamKind::Float),
            ParamInfo::prop("maxLuminance", ParamKind::Float),
            ParamInfo::prop("averageLuminance", ParamKind::Float),
            ParamInfo::prop("adaptionRate", ParamKind::Float),
            ParamInfo::prop("needsInit", ParamKind::Bool),
        ]
    }

    fn get(&self, name: &str) -> Option<ParamValue> {
        Some(match name {
            "adaptive" => ParamValue::Bool(self.adaptive),
            "resolution" => ParamValue::Int(self.resolution as i32),
            "middleGrey" => ParamValue::Float(self.middle_grey),
            "minLuminance" => ParamValue::Float(self.min_luminance),
            "maxLuminance" => ParamValue::Float(self.max_luminance),
            "averageLuminance" => ParamValue::Float(self.average_luminance),
            "adaptionRate" => ParamValue::Float(self.adaption_rate),
            "needsInit" => ParamValue::Bool(self.needs_init),
            _ => return None,
        })
    }

    fn set(&mut self, name: &str, value: ParamValue) -> bool {
        match name {
            "adaptive" => assign_bool(&mut self.adaptive, &value),
            "resolution" => value
                .as_i32()
                .map(|v| {
                    self.resolution = v.max(1) as u32;
                    self.needs_init = true;
                })
                .is_some(),
            "middleGrey" => assign_f32(&mut self.middle_grey, &value),
            "minLuminance" => assign_f32(&mut self.min_luminance, &value),
            "maxLuminance" => assign_f32(&mut self.max_luminance, &value),
            "averageLuminance" => assign_f32(&mut self.average_luminance, &value),
            "adaptionRate" => assign_f32(&mut self.adaption_rate, &value),
            "needsInit" => assign_bool(&mut self.needs_init, &value),
            _ => false,
        }
    }
}
