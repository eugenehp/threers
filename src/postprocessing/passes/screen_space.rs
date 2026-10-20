//! The passes that read the scene's geometry back from depth and normal
//! prepasses: SSAOPass, SAOPass, SSRPass and OutlinePass.

use std::sync::Arc;

use super::basic::out;
use crate::core::ObjectId;
use crate::materials::{BasicMaterial, Material, NormalMaterial};
use crate::math::Color;
use crate::postprocessing::context::{kinds, Dest, Fx, PassContext, SceneDraw, Visibility};
use crate::postprocessing::params::{assign_bool, assign_f32, ParamInfo, ParamKind, ParamValue};
use crate::postprocessing::pass::{pass_common, Pass, PassState};
use crate::postprocessing::random::{random, SimplexNoise};
use crate::postprocessing::target::Target;
use crate::postprocessing::upload;
use crate::renderer::{PostFxBlend, PostFxNoise};
use crate::textures::Texture;

fn normal_material() -> Arc<Material> {
    Arc::new(Material::Normal(NormalMaterial::new()))
}

/// An opaque material for a depth-only pass, both faces drawn.
fn depth_material(double_sided: bool) -> Arc<Material> {
    let mut m = BasicMaterial::new(Color::WHITE);
    if double_sided {
        m.side = 2;
    }
    Arc::new(Material::Basic(m))
}

/// Where several draws meant for `dest` go: a stand-in when it is the
/// screen (see [`PassContext::screen_stand_in`]), else `dest` itself.
fn composite_dest(ctx: &mut PassContext<'_, '_>, to_screen: bool, target: &Target) -> Option<Target> {
    if to_screen {
        ctx.screen_stand_in()
    } else {
        Some(target.clone())
    }
}

fn composite_into<'t>(to_screen: bool, t: &'t Target) -> Dest<'t> {
    if to_screen {
        Dest::StandIn(t)
    } else {
        Dest::Target(t)
    }
}

// ---------------------------------------------------------------------------

/// three.js `SSAOPass.OUTPUT`.
pub mod ssao_output {
    pub const DEFAULT: i32 = 0;
    pub const SSAO: i32 = 1;
    pub const BLUR: i32 = 2;
    pub const DEPTH: i32 = 3;
    pub const NORMAL: i32 = 4;
}

/// three.js `SSAOPass(scene, camera, width, height, kernelSize)`.
///
/// The sample kernel and the rotation noise are drawn from
/// [`random`](crate::postprocessing::random::random) in three.js's order —
/// the kernel, then `SimplexNoise`'s permutation, then the noise
/// coordinates — so a seeded run builds three.js's exactly.
#[derive(Debug)]
pub struct SsaoPass {
    pub state: PassState,
    pub scene: usize,
    pub camera: usize,
    pub width: u32,
    pub height: u32,
    pub kernel_radius: f32,
    pub min_distance: f32,
    pub max_distance: f32,
    /// One of [`ssao_output`].
    pub output: i32,
    kernel_size: u32,
    kernel: Vec<[f32; 3]>,
    noise: [f32; 16],
    normal_material: Arc<Material>,
    normal_rt: Option<Target>,
    ssao_rt: Option<Target>,
    blur_rt: Option<Target>,
}

impl SsaoPass {
    pub fn new(scene: usize, camera: usize, width: u32, height: u32, kernel_size: u32) -> Self {
        let mut p = Self {
            state: PassState {
                clear: true,
                ..PassState::default()
            },
            scene,
            camera,
            width: width.max(1),
            height: height.max(1),
            kernel_radius: 8.0,
            min_distance: 0.005,
            max_distance: 0.1,
            output: ssao_output::DEFAULT,
            kernel_size: kernel_size.clamp(1, 32),
            kernel: Vec::new(),
            noise: [0.0; 16],
            normal_material: normal_material(),
            normal_rt: None,
            ssao_rt: None,
            blur_rt: None,
        };
        p.generate_sample_kernel(kernel_size);
        p.generate_random_kernel_rotations();
        p
    }

    /// three.js `generateSampleKernel`: points in the +z hemisphere, denser
    /// towards the centre. (The shader's kernel holds 32.)
    pub fn generate_sample_kernel(&mut self, kernel_size: u32) {
        let n = kernel_size.max(1);
        self.kernel_size = n.min(32);
        self.kernel = (0..n)
            .map(|i| {
                let x = random() * 2.0 - 1.0;
                let y = random() * 2.0 - 1.0;
                let z = random();
                let len = (x * x + y * y + z * z).sqrt();
                let (x, y, z) = if len > 0.0 { (x / len, y / len, z / len) } else { (0.0, 0.0, 0.0) };
                let mut scale = i as f64 / n as f64;
                scale = 0.1 + (1.0 - 0.1) * (scale * scale);
                [(x * scale) as f32, (y * scale) as f32, (z * scale) as f32]
            })
            .collect();
    }

    /// three.js `generateRandomKernelRotations`: a 4×4 simplex-noise texture.
    pub fn generate_random_kernel_rotations(&mut self) {
        let simplex = SimplexNoise::new();
        for v in self.noise.iter_mut() {
            let x = random() * 2.0 - 1.0;
            let y = random() * 2.0 - 1.0;
            *v = simplex.noise3d(x, y, 0.0) as f32;
        }
    }

    pub fn kernel(&self) -> &[[f32; 3]] {
        &self.kernel
    }

    pub fn noise(&self) -> &[f32; 16] {
        &self.noise
    }
}

impl Pass for SsaoPass {
    pass_common!("SSAOPass");

    fn set_size(&mut self, width: u32, height: u32) {
        self.width = width.max(1);
        self.height = height.max(1);
    }

    fn render(&mut self, ctx: &mut PassContext<'_, '_>, write: &Target, read: &Target, _d: f32, _m: bool) {
        let (w, h) = (self.width, self.height);
        let normal = ctx.target(&mut self.normal_rt, w, h);
        // Normals and depth, meshes only. three.js clears to 0x7777ff and the
        // scene's colour background clears again over it; the second wins.
        ctx.render_scene(
            Dest::Target(&normal),
            &SceneDraw {
                scene: self.scene,
                camera: self.camera,
                override_material: Some(self.normal_material.clone()),
                visibility: Visibility::MeshesOnly,
                ..SceneDraw::default()
            },
        );

        let weights: Vec<[f32; 4]> = self
            .kernel
            .iter()
            .take(32)
            .map(|k| [k[0], k[1], k[2], 0.0])
            .collect();
        ctx.renderer.set_postfx_kernel(&weights);
        ctx.renderer.set_ssao_noise(&self.noise);
        let mut camera = ctx.postfx_camera(self.camera);
        camera.kernel_radius = self.kernel_radius;
        camera.kernel_size = self.kernel_size;

        let ssao = ctx.target(&mut self.ssao_rt, w, h);
        ctx.fx(
            Fx::new(kinds::SSAO)
                .input(read)
                .normal(&normal)
                .hw_depth(&normal)
                .camera(camera.clone())
                .p2([self.min_distance, self.max_distance, 0.0, 0.0])
                .blend(PostFxBlend::Overwrite),
            Dest::Target(&ssao),
        );
        let blur = ctx.target(&mut self.blur_rt, w, h);
        ctx.fx(
            Fx::new(kinds::SSAO_BLUR).input(&ssao).blend(PostFxBlend::Overwrite),
            Dest::Target(&blur),
        );

        let to_screen = self.state.render_to_screen;
        let dest = out(&self.state, write);
        match self.output {
            ssao_output::SSAO => ctx.fx(Fx::copy(&ssao, 1.0).blend(PostFxBlend::Overwrite), dest),
            ssao_output::BLUR => ctx.fx(Fx::copy(&blur, 1.0).blend(PostFxBlend::Overwrite), dest),
            ssao_output::DEPTH => ctx.fx(
                Fx::new(kinds::DEPTH_VIEW)
                    .hw_depth(&normal)
                    .camera(camera)
                    .blend(PostFxBlend::Overwrite),
                dest,
            ),
            ssao_output::NORMAL => ctx.fx(Fx::copy(&normal, 1.0).blend(PostFxBlend::Overwrite), dest),
            _ => {
                // The beauty, then the blurred occlusion multiplied in.
                let Some(t) = composite_dest(ctx, to_screen, write) else {
                    return;
                };
                let d = composite_into(to_screen, &t);
                ctx.fx(Fx::copy(read, 1.0).blend(PostFxBlend::Overwrite), d);
                ctx.fx(Fx::copy(&blur, 1.0).blend(PostFxBlend::Multiply), d);
                if to_screen {
                    ctx.present_stand_in(&t);
                }
            }
        }
    }

    fn params(&self) -> Vec<ParamInfo> {
        vec![
            ParamInfo::prop("kernelRadius", ParamKind::Float),
            ParamInfo::prop("minDistance", ParamKind::Float),
            ParamInfo::prop("maxDistance", ParamKind::Float),
            ParamInfo::prop("output", ParamKind::Int),
            ParamInfo::prop("width", ParamKind::Int),
            ParamInfo::prop("height", ParamKind::Int),
            ParamInfo::prop("kernelSize", ParamKind::Int),
        ]
    }

    fn get(&self, name: &str) -> Option<ParamValue> {
        Some(match name {
            "kernelRadius" => ParamValue::Float(self.kernel_radius),
            "minDistance" => ParamValue::Float(self.min_distance),
            "maxDistance" => ParamValue::Float(self.max_distance),
            "output" => ParamValue::Int(self.output),
            "width" => ParamValue::Int(self.width as i32),
            "height" => ParamValue::Int(self.height as i32),
            "kernelSize" => ParamValue::Int(self.kernel_size as i32),
            _ => return None,
        })
    }

    fn set(&mut self, name: &str, value: ParamValue) -> bool {
        match name {
            "kernelRadius" => assign_f32(&mut self.kernel_radius, &value),
            "minDistance" => assign_f32(&mut self.min_distance, &value),
            "maxDistance" => assign_f32(&mut self.max_distance, &value),
            "output" => value.as_i32().map(|v| self.output = v).is_some(),
            "width" => value.as_i32().map(|v| self.width = v.max(1) as u32).is_some(),
            "height" => value.as_i32().map(|v| self.height = v.max(1) as u32).is_some(),
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

/// three.js `SAOPass.OUTPUT`.
pub mod sao_output {
    pub const DEFAULT: i32 = 0;
    pub const SAO: i32 = 1;
    pub const NORMAL: i32 = 2;
}

/// three.js `SAOPass(scene, camera, resolution)`: scalable ambient obscurance
/// from a normal-and-depth prepass, blurred with the depth-limited blur,
/// multiplied into `readBuffer`. Its `params` object's fields are the
/// parameters here (`saoBias`, `saoIntensity`, …).
#[derive(Debug)]
pub struct SaoPass {
    pub state: PassState,
    pub scene: usize,
    pub camera: usize,
    pub output: i32,
    pub sao_bias: f32,
    pub sao_intensity: f32,
    pub sao_scale: f32,
    pub sao_kernel_radius: f32,
    pub sao_min_resolution: f32,
    pub sao_blur: bool,
    pub sao_blur_radius: f32,
    pub sao_blur_std_dev: f32,
    pub sao_blur_depth_cutoff: f32,
    width: u32,
    height: u32,
    normal_material: Arc<Material>,
    normal_rt: Option<Target>,
    sao_rt: Option<Target>,
    blur_rt: Option<Target>,
}

impl SaoPass {
    pub fn new(scene: usize, camera: usize, width: u32, height: u32) -> Self {
        Self {
            state: PassState {
                clear: true,
                ..PassState::swapping(false)
            },
            scene,
            camera,
            output: sao_output::DEFAULT,
            sao_bias: 0.5,
            sao_intensity: 0.18,
            sao_scale: 1.0,
            sao_kernel_radius: 100.0,
            sao_min_resolution: 0.0,
            sao_blur: true,
            sao_blur_radius: 8.0,
            sao_blur_std_dev: 4.0,
            sao_blur_depth_cutoff: 0.01,
            width: width.max(1),
            height: height.max(1),
            normal_material: normal_material(),
            normal_rt: None,
            sao_rt: None,
            blur_rt: None,
        }
    }
}

/// three.js `BlurShaderUtils`' `gaussian`.
fn gaussian(x: f64, std_dev: f64) -> f64 {
    (-(x * x) / (2.0 * (std_dev * std_dev))).exp() / ((2.0 * std::f64::consts::PI).sqrt() * std_dev)
}

impl Pass for SaoPass {
    pass_common!("SAOPass");

    fn set_size(&mut self, width: u32, height: u32) {
        self.width = width.max(1);
        self.height = height.max(1);
    }

    fn render(&mut self, ctx: &mut PassContext<'_, '_>, _write: &Target, read: &Target, _d: f32, _m: bool) {
        let (w, h) = (self.width, self.height);
        let to_screen = self.state.render_to_screen;
        // Onto the screen three.js first copies readBuffer and later
        // multiplies the occlusion over it: both into a stand-in.
        let stand_in = if to_screen {
            let Some(s) = ctx.screen_stand_in() else {
                return;
            };
            ctx.fx(Fx::copy(read, 1.0).blend(PostFxBlend::Overwrite), Dest::StandIn(&s));
            Some(s)
        } else {
            None
        };

        let normal = ctx.target(&mut self.normal_rt, w, h);
        ctx.render_scene(
            Dest::Target(&normal),
            &SceneDraw {
                scene: self.scene,
                camera: self.camera,
                override_material: Some(self.normal_material.clone()),
                ..SceneDraw::default()
            },
        );

        let camera = ctx.postfx_camera(self.camera);
        let perspective = if ctx.camera_is_perspective(self.camera) { 1.0 } else { 0.0 };
        let white = [1.0, 1.0, 1.0, 1.0];
        let sao = ctx.target(&mut self.sao_rt, w, h);
        ctx.fx(
            Fx::new(kinds::SAO)
                .normal(&normal)
                .hw_depth(&normal)
                .camera(camera.clone())
                .p2([self.sao_bias, self.sao_intensity, self.sao_scale, self.sao_kernel_radius])
                .p3([self.sao_min_resolution, 0.0, perspective, 0.0])
                .clear_color(white),
            Dest::Target(&sao),
        );

        if self.sao_blur {
            let radius = self.sao_blur_radius.floor().clamp(0.0, 127.0) as usize;
            let std_dev = self.sao_blur_std_dev as f64;
            let mut weights = vec![[0f32; 4]; 32];
            for i in 0..=radius {
                weights[i / 4][i % 4] = gaussian(i as f64, std_dev) as f32;
            }
            ctx.renderer.set_postfx_kernel(&weights);
            let cutoff = self.sao_blur_depth_cutoff * (camera.far - camera.near);
            let blur = ctx.target(&mut self.blur_rt, w, h);
            let blur_fx = |input: &Target, dir: [f32; 2]| {
                Fx::new(kinds::DEPTH_LIMITED_BLUR)
                    .input(input)
                    .hw_depth(&normal)
                    .camera(camera.clone())
                    .p2([radius as f32, dir[0], dir[1], cutoff])
                    .p3([perspective, 0.0, 0.0, 0.0])
                    .clear_color(white)
            };
            ctx.fx(blur_fx(&sao, [0.0, 1.0]), Dest::Target(&blur));
            ctx.fx(blur_fx(&blur, [1.0, 0.0]), Dest::Target(&sao));
        }

        let source = if self.output == sao_output::NORMAL { &normal } else { &sao };
        let blend = if self.output == sao_output::DEFAULT {
            PostFxBlend::Multiply
        } else {
            PostFxBlend::Overwrite
        };
        let copy = Fx::copy(source, 1.0).blend(blend);
        match stand_in {
            Some(s) => {
                ctx.fx(copy, Dest::StandIn(&s));
                ctx.present_stand_in(&s);
            }
            None => ctx.fx(copy, Dest::Target(read)),
        }
    }

    fn params(&self) -> Vec<ParamInfo> {
        [
            ("output", ParamKind::Int),
            ("saoBias", ParamKind::Float),
            ("saoIntensity", ParamKind::Float),
            ("saoScale", ParamKind::Float),
            ("saoKernelRadius", ParamKind::Float),
            ("saoMinResolution", ParamKind::Float),
            ("saoBlur", ParamKind::Bool),
            ("saoBlurRadius", ParamKind::Float),
            ("saoBlurStdDev", ParamKind::Float),
            ("saoBlurDepthCutoff", ParamKind::Float),
        ]
        .into_iter()
        .map(|(n, k)| ParamInfo::prop(n, k))
        .collect()
    }

    fn get(&self, name: &str) -> Option<ParamValue> {
        Some(match name {
            "output" => ParamValue::Int(self.output),
            "saoBias" => ParamValue::Float(self.sao_bias),
            "saoIntensity" => ParamValue::Float(self.sao_intensity),
            "saoScale" => ParamValue::Float(self.sao_scale),
            "saoKernelRadius" => ParamValue::Float(self.sao_kernel_radius),
            "saoMinResolution" => ParamValue::Float(self.sao_min_resolution),
            "saoBlur" => ParamValue::Bool(self.sao_blur),
            "saoBlurRadius" => ParamValue::Float(self.sao_blur_radius),
            "saoBlurStdDev" => ParamValue::Float(self.sao_blur_std_dev),
            "saoBlurDepthCutoff" => ParamValue::Float(self.sao_blur_depth_cutoff),
            _ => return None,
        })
    }

    fn set(&mut self, name: &str, value: ParamValue) -> bool {
        match name {
            "output" => value.as_i32().map(|v| self.output = v).is_some(),
            "saoBias" => assign_f32(&mut self.sao_bias, &value),
            "saoIntensity" => assign_f32(&mut self.sao_intensity, &value),
            "saoScale" => assign_f32(&mut self.sao_scale, &value),
            "saoKernelRadius" => assign_f32(&mut self.sao_kernel_radius, &value),
            "saoMinResolution" => assign_f32(&mut self.sao_min_resolution, &value),
            "saoBlur" => assign_bool(&mut self.sao_blur, &value),
            "saoBlurRadius" => assign_f32(&mut self.sao_blur_radius, &value),
            "saoBlurStdDev" => assign_f32(&mut self.sao_blur_std_dev, &value),
            "saoBlurDepthCutoff" => assign_f32(&mut self.sao_blur_depth_cutoff, &value),
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

/// three.js `SSRPass.OUTPUT`.
pub mod ssr_output {
    pub const DEFAULT: i32 = 0;
    pub const SSR: i32 = 1;
    pub const BEAUTY: i32 = 3;
    pub const DEPTH: i32 = 4;
    pub const NORMAL: i32 = 5;
    pub const METALNESS: i32 = 7;
}

/// three.js `SSRPass({ scene, camera, width, height, selects, bouncing })`:
/// screen-space reflections, ray-marched over its own beauty, normal and
/// depth renders. (`groundReflector` — a `ReflectorForSSRPass` — has no
/// counterpart here.)
#[derive(Debug)]
pub struct SsrPass {
    pub state: PassState,
    pub scene: usize,
    pub camera: usize,
    pub width: u32,
    pub height: u32,
    pub opacity: f32,
    pub output: i32,
    pub max_distance: f32,
    pub thickness: f32,
    pub blur: bool,
    pub distance_attenuation: bool,
    pub fresnel: bool,
    pub infinite_thick: bool,
    pub bouncing: bool,
    /// three.js `selects`: when set, only these reflect.
    pub selects: Option<Vec<ObjectId>>,
    normal_material: Arc<Material>,
    metal_on: Arc<Material>,
    metal_off: Arc<Material>,
    beauty: Option<Target>,
    prev: Option<Target>,
    normal_rt: Option<Target>,
    metalness: Option<Target>,
    ssr: Option<Target>,
    blur1: Option<Target>,
    blur2: Option<Target>,
}

impl SsrPass {
    pub fn new(scene: usize, camera: usize, width: u32, height: u32) -> Self {
        Self {
            state: PassState {
                clear: true,
                ..PassState::default()
            },
            scene,
            camera,
            width: width.max(1),
            height: height.max(1),
            opacity: 0.5,
            output: ssr_output::DEFAULT,
            max_distance: 180.0,
            thickness: 0.018,
            blur: true,
            distance_attenuation: true,
            fresnel: true,
            infinite_thick: false,
            bouncing: false,
            selects: None,
            normal_material: normal_material(),
            metal_on: Arc::new(Material::Basic(BasicMaterial::new(Color::WHITE))),
            metal_off: Arc::new(Material::Basic(BasicMaterial::new(Color::BLACK))),
            beauty: None,
            prev: None,
            normal_rt: None,
            metalness: None,
            ssr: None,
            blur1: None,
            blur2: None,
        }
    }
}

impl Pass for SsrPass {
    pass_common!("SSRPass");

    fn set_size(&mut self, width: u32, height: u32) {
        self.width = width.max(1);
        self.height = height.max(1);
    }

    fn render(&mut self, ctx: &mut PassContext<'_, '_>, write: &Target, _read: &Target, _d: f32, _m: bool) {
        let (w, h) = (self.width, self.height);
        let draw = |m: Option<Arc<Material>>| SceneDraw {
            scene: self.scene,
            camera: self.camera,
            override_material: m,
            ..SceneDraw::default()
        };
        let beauty = ctx.target(&mut self.beauty, w, h);
        ctx.render_scene(Dest::Target(&beauty), &draw(None));
        let normal = ctx.target(&mut self.normal_rt, w, h);
        ctx.render_scene(Dest::Target(&normal), &draw(Some(self.normal_material.clone())));
        let metalness = ctx.target(&mut self.metalness, w, h);
        if let Some(selects) = &self.selects {
            ctx.render_scene(
                Dest::Target(&metalness),
                &SceneDraw {
                    override_by_selection: Some((
                        selects.clone(),
                        self.metal_on.clone(),
                        self.metal_off.clone(),
                    )),
                    ..draw(None)
                },
            );
        }
        let prev = ctx.target(&mut self.prev, w, h);

        let camera = ctx.postfx_camera(self.camera);
        let perspective = if ctx.camera_is_perspective(self.camera) { 1.0 } else { 0.0 };
        let flags = u32::from(self.distance_attenuation)
            | u32::from(self.fresnel) << 1
            | u32::from(self.infinite_thick) << 2
            | u32::from(self.selects.is_some()) << 3;
        let max_step = ((w * w + h * h) as f32).sqrt();
        let ssr = ctx.target(&mut self.ssr, w, h);
        ctx.fx(
            Fx::new(kinds::SSR)
                .input(if self.bouncing { &prev } else { &beauty })
                .hw_depth(&beauty)
                .normal(&normal)
                .depth(&metalness)
                .camera(camera.clone())
                .p2([self.opacity, self.max_distance, self.thickness, flags as f32])
                .p3([max_step, perspective, 0.0, 0.0])
                .blend(PostFxBlend::Overwrite),
            Dest::Target(&ssr),
        );
        let blur2 = if self.blur {
            let b1 = ctx.target(&mut self.blur1, w, h);
            ctx.fx(Fx::new(kinds::SSR_BLUR).input(&ssr).blend(PostFxBlend::Overwrite), Dest::Target(&b1));
            let b2 = ctx.target(&mut self.blur2, w, h);
            ctx.fx(Fx::new(kinds::SSR_BLUR).input(&b1).blend(PostFxBlend::Overwrite), Dest::Target(&b2));
            Some(b2)
        } else {
            None
        };
        let reflection = blur2.as_ref().unwrap_or(&ssr);

        let to_screen = self.state.render_to_screen;
        let dest = out(&self.state, write);
        let copy = |t: &Target, blend| Fx::copy(t, 1.0).blend(blend);
        match self.output {
            ssr_output::SSR => {
                ctx.fx(copy(reflection, PostFxBlend::Overwrite), dest);
                if self.bouncing {
                    ctx.fx(copy(blur2.as_ref().unwrap_or(&beauty), PostFxBlend::Overwrite), Dest::Target(&prev));
                    ctx.fx(copy(&ssr, PostFxBlend::Normal), Dest::Target(&prev));
                }
            }
            ssr_output::BEAUTY => ctx.fx(copy(&beauty, PostFxBlend::Overwrite), dest),
            ssr_output::DEPTH => ctx.fx(
                Fx::new(kinds::DEPTH_VIEW)
                    .hw_depth(&beauty)
                    .camera(camera)
                    .p2([1.0, 0.0, 0.0, 0.0])
                    .blend(PostFxBlend::Overwrite),
                dest,
            ),
            ssr_output::NORMAL => ctx.fx(copy(&normal, PostFxBlend::Overwrite), dest),
            ssr_output::METALNESS => ctx.fx(copy(&metalness, PostFxBlend::Overwrite), dest),
            _ => {
                if self.bouncing {
                    ctx.fx(copy(&beauty, PostFxBlend::Overwrite), Dest::Target(&prev));
                    ctx.fx(copy(reflection, PostFxBlend::Normal), Dest::Target(&prev));
                    ctx.fx(copy(&prev, PostFxBlend::Overwrite), dest);
                } else {
                    let Some(t) = composite_dest(ctx, to_screen, write) else {
                        return;
                    };
                    let d = composite_into(to_screen, &t);
                    ctx.fx(copy(&beauty, PostFxBlend::Overwrite), d);
                    ctx.fx(copy(reflection, PostFxBlend::Normal), d);
                    if to_screen {
                        ctx.present_stand_in(&t);
                    }
                }
            }
        }
    }

    fn params(&self) -> Vec<ParamInfo> {
        vec![
            ParamInfo::prop("opacity", ParamKind::Float),
            ParamInfo::prop("output", ParamKind::Int),
            ParamInfo::prop("maxDistance", ParamKind::Float),
            ParamInfo::prop("thickness", ParamKind::Float),
            ParamInfo::prop("blur", ParamKind::Bool),
            ParamInfo::prop("distanceAttenuation", ParamKind::Bool),
            ParamInfo::prop("fresnel", ParamKind::Bool),
            ParamInfo::prop("infiniteThick", ParamKind::Bool),
            ParamInfo::prop("bouncing", ParamKind::Bool),
            ParamInfo::prop("selects", ParamKind::Objects),
            ParamInfo::prop("width", ParamKind::Int),
            ParamInfo::prop("height", ParamKind::Int),
        ]
    }

    fn get(&self, name: &str) -> Option<ParamValue> {
        Some(match name {
            "opacity" => ParamValue::Float(self.opacity),
            "output" => ParamValue::Int(self.output),
            "maxDistance" => ParamValue::Float(self.max_distance),
            "thickness" => ParamValue::Float(self.thickness),
            "blur" => ParamValue::Bool(self.blur),
            "distanceAttenuation" => ParamValue::Bool(self.distance_attenuation),
            "fresnel" => ParamValue::Bool(self.fresnel),
            "infiniteThick" => ParamValue::Bool(self.infinite_thick),
            "bouncing" => ParamValue::Bool(self.bouncing),
            "selects" => ParamValue::Objects(self.selects.clone().unwrap_or_default()),
            "width" => ParamValue::Int(self.width as i32),
            "height" => ParamValue::Int(self.height as i32),
            _ => return None,
        })
    }

    fn set(&mut self, name: &str, value: ParamValue) -> bool {
        match name {
            "opacity" => assign_f32(&mut self.opacity, &value),
            "output" => value.as_i32().map(|v| self.output = v).is_some(),
            "maxDistance" => assign_f32(&mut self.max_distance, &value),
            "thickness" => assign_f32(&mut self.thickness, &value),
            "blur" => assign_bool(&mut self.blur, &value),
            "distanceAttenuation" => assign_bool(&mut self.distance_attenuation, &value),
            "fresnel" => assign_bool(&mut self.fresnel, &value),
            "infiniteThick" => assign_bool(&mut self.infinite_thick, &value),
            "bouncing" => assign_bool(&mut self.bouncing, &value),
            "width" => value.as_i32().map(|v| self.width = v.max(1) as u32).is_some(),
            "height" => value.as_i32().map(|v| self.height = v.max(1) as u32).is_some(),
            "selects" => match value {
                ParamValue::Objects(ids) => {
                    self.selects = Some(ids);
                    true
                }
                _ => {
                    self.selects = None;
                    true
                }
            },
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

const MAX_EDGE_THICKNESS: f32 = 4.0;
const MAX_EDGE_GLOW: f32 = 4.0;

/// Milliseconds since the first call — three.js's `performance.now()` for
/// OutlinePass's pulse.
fn now_ms() -> f64 {
    static START: std::sync::OnceLock<web_time::Instant> = std::sync::OnceLock::new();
    START.get_or_init(web_time::Instant::now).elapsed().as_secs_f64() * 1000.0
}

/// three.js `OutlinePass(resolution, scene, camera, selectedObjects)`: a glow
/// round the selected objects, in `visibleEdgeColor` where they are in view
/// and `hiddenEdgeColor` where something else hides them.
#[derive(Debug)]
pub struct OutlinePass {
    pub state: PassState,
    pub scene: usize,
    pub camera: usize,
    pub selected: Vec<ObjectId>,
    pub visible_edge_color: Color,
    pub hidden_edge_color: Color,
    pub edge_glow: f32,
    pub use_pattern_texture: bool,
    pub edge_thickness: f32,
    pub edge_strength: f32,
    pub down_sample_ratio: f32,
    pub pulse_period: f32,
    pub pattern_texture: Option<Arc<Texture>>,
    pub resolution: [f32; 2],
    width: u32,
    height: u32,
    depth_material: Arc<Material>,
    depth_rt: Option<Target>,
    selected_rt: Option<Target>,
    mask_rt: Option<Target>,
    mask_down: Option<Target>,
    blur1: Option<Target>,
    edge1: Option<Target>,
    blur2: Option<Target>,
    edge2: Option<Target>,
    pattern: Option<(u64, Target)>,
}

impl OutlinePass {
    pub fn new(width: u32, height: u32, scene: usize, camera: usize, selected: Vec<ObjectId>) -> Self {
        Self {
            state: PassState::swapping(false),
            scene,
            camera,
            selected,
            visible_edge_color: Color::new(1.0, 1.0, 1.0),
            hidden_edge_color: Color::new(0.1, 0.04, 0.02),
            edge_glow: 0.0,
            use_pattern_texture: false,
            edge_thickness: 1.0,
            edge_strength: 3.0,
            down_sample_ratio: 2.0,
            pulse_period: 0.0,
            pattern_texture: None,
            resolution: [width as f32, height as f32],
            width: width.max(1),
            height: height.max(1),
            depth_material: depth_material(true),
            depth_rt: None,
            selected_rt: None,
            mask_rt: None,
            mask_down: None,
            blur1: None,
            edge1: None,
            blur2: None,
            edge2: None,
            pattern: None,
        }
    }
}

impl Pass for OutlinePass {
    pass_common!("OutlinePass");

    fn set_size(&mut self, width: u32, height: u32) {
        self.width = width.max(1);
        self.height = height.max(1);
    }

    fn render(&mut self, ctx: &mut PassContext<'_, '_>, _write: &Target, read: &Target, _d: f32, _m: bool) {
        if !self.selected.is_empty() {
            let (w, h) = (self.width, self.height);
            let round = |v: f32| (v + 0.5).floor().max(1.0) as u32;
            let (hw, hh) = (
                round(w as f32 / self.down_sample_ratio),
                round(h as f32 / self.down_sample_ratio),
            );
            let (qw, qh) = (round(hw as f32 / 2.0), round(hh as f32 / 2.0));

            // 1. Depth of everything but the selection, and of the selection
            //    alone; the mask compares the two (three.js renders the
            //    selection comparing against a packed depth texture).
            let depth = ctx.target(&mut self.depth_rt, w, h);
            let draw = |visibility| SceneDraw {
                scene: self.scene,
                camera: self.camera,
                override_material: Some(self.depth_material.clone()),
                visibility,
                clear: Some((Color::WHITE, 1.0)),
                ..SceneDraw::default()
            };
            ctx.render_scene(Dest::Target(&depth), &draw(Visibility::Except(self.selected.clone())));
            let selected = ctx.target(&mut self.selected_rt, w, h);
            ctx.render_scene(Dest::Target(&selected), &draw(Visibility::Only(self.selected.clone())));
            let mask = ctx.target(&mut self.mask_rt, w, h);
            ctx.fx(
                Fx::new(kinds::OUTLINE_MASK)
                    .hw_depth(&depth)
                    .noise(PostFxNoise::Depth(selected.id())),
                Dest::Target(&mask),
            );

            // 2. Downsample to half resolution.
            let mask_down = ctx.target(&mut self.mask_down, hw, hh);
            ctx.fx(Fx::copy(&mask, 1.0), Dest::Target(&mask_down));

            let mut visible = self.visible_edge_color;
            let mut hidden = self.hidden_edge_color;
            if self.pulse_period > 0.0 {
                let s = (1.0 + 0.25) / 2.0
                    + (now_ms() * 0.01 / self.pulse_period as f64).cos() as f32 * (1.0 - 0.25) / 2.0;
                visible = Color::new(visible.r * s, visible.g * s, visible.b * s);
                hidden = Color::new(hidden.r * s, hidden.g * s, hidden.b * s);
            }

            // 3. Edge detection.
            let edge1 = ctx.target(&mut self.edge1, hw, hh);
            ctx.fx(
                Fx::new(kinds::OUTLINE_EDGE)
                    .input(&mask_down)
                    .p2([visible.r, visible.g, visible.b, 0.0])
                    .p3([hidden.r, hidden.g, hidden.b, 0.0]),
                Dest::Target(&edge1),
            );

            // 4. Blur at half and at quarter resolution.
            let blur = |input: &Target, radius: f32, dir: [f32; 2], max: f32| {
                Fx::new(kinds::OUTLINE_BLUR)
                    .input(input)
                    .p2([radius, dir[0], dir[1], max])
            };
            let blur1 = ctx.target(&mut self.blur1, hw, hh);
            ctx.fx(blur(&edge1, self.edge_thickness, [1.0, 0.0], MAX_EDGE_THICKNESS), Dest::Target(&blur1));
            ctx.fx(blur(&blur1, self.edge_thickness, [0.0, 1.0], MAX_EDGE_THICKNESS), Dest::Target(&edge1));
            let blur2 = ctx.target(&mut self.blur2, qw, qh);
            ctx.fx(blur(&edge1, MAX_EDGE_GLOW, [1.0, 0.0], MAX_EDGE_GLOW), Dest::Target(&blur2));
            let edge2 = ctx.target(&mut self.edge2, qw, qh);
            ctx.fx(blur(&blur2, MAX_EDGE_GLOW, [0.0, 1.0], MAX_EDGE_GLOW), Dest::Target(&edge2));

            // 5. Added over readBuffer.
            let mut overlay = Fx::new(kinds::OUTLINE_OVERLAY)
                .input(&edge1)
                .depth(&edge2)
                .normal(&mask)
                .linear_aux()
                .p2([
                    self.edge_strength,
                    self.edge_glow,
                    if self.use_pattern_texture { 1.0 } else { 0.0 },
                    0.0,
                ])
                .blend(PostFxBlend::AddAlpha);
            if let Some(tex) = &self.pattern_texture {
                if let Some(p) = upload::texture_target(ctx.renderer, tex, &mut self.pattern) {
                    overlay = overlay.noise(PostFxNoise::Color(p.id()));
                }
            }
            ctx.fx(overlay, Dest::Target(read));
        }
        if self.state.render_to_screen {
            ctx.fx(Fx::copy(read, 1.0).blend(PostFxBlend::Overwrite), Dest::Screen);
        }
    }

    fn params(&self) -> Vec<ParamInfo> {
        vec![
            ParamInfo::prop("selectedObjects", ParamKind::Objects),
            ParamInfo::prop("visibleEdgeColor", ParamKind::Color),
            ParamInfo::prop("hiddenEdgeColor", ParamKind::Color),
            ParamInfo::prop("edgeGlow", ParamKind::Float),
            ParamInfo::prop("usePatternTexture", ParamKind::Bool),
            ParamInfo::prop("edgeThickness", ParamKind::Float),
            ParamInfo::prop("edgeStrength", ParamKind::Float),
            ParamInfo::prop("downSampleRatio", ParamKind::Float),
            ParamInfo::prop("pulsePeriod", ParamKind::Float),
            ParamInfo::prop("patternTexture", ParamKind::Texture),
            ParamInfo::prop("resolution", ParamKind::Vec2),
        ]
    }

    fn get(&self, name: &str) -> Option<ParamValue> {
        let c = |c: Color| ParamValue::Color([c.r, c.g, c.b]);
        Some(match name {
            "selectedObjects" => ParamValue::Objects(self.selected.clone()),
            "visibleEdgeColor" => c(self.visible_edge_color),
            "hiddenEdgeColor" => c(self.hidden_edge_color),
            "edgeGlow" => ParamValue::Float(self.edge_glow),
            "usePatternTexture" => ParamValue::Bool(self.use_pattern_texture),
            "edgeThickness" => ParamValue::Float(self.edge_thickness),
            "edgeStrength" => ParamValue::Float(self.edge_strength),
            "downSampleRatio" => ParamValue::Float(self.down_sample_ratio),
            "pulsePeriod" => ParamValue::Float(self.pulse_period),
            "patternTexture" => ParamValue::Texture(self.pattern_texture.clone()),
            "resolution" => ParamValue::Vec2(self.resolution),
            _ => return None,
        })
    }

    fn set(&mut self, name: &str, value: ParamValue) -> bool {
        let color = |v: &ParamValue| v.as_array::<3>().map(|c| Color::new(c[0], c[1], c[2]));
        match name {
            "selectedObjects" => match value {
                ParamValue::Objects(ids) => {
                    self.selected = ids;
                    true
                }
                _ => false,
            },
            "visibleEdgeColor" => color(&value).map(|c| self.visible_edge_color = c).is_some(),
            "hiddenEdgeColor" => color(&value).map(|c| self.hidden_edge_color = c).is_some(),
            "edgeGlow" => assign_f32(&mut self.edge_glow, &value),
            "usePatternTexture" => assign_bool(&mut self.use_pattern_texture, &value),
            "edgeThickness" => assign_f32(&mut self.edge_thickness, &value),
            "edgeStrength" => assign_f32(&mut self.edge_strength, &value),
            "downSampleRatio" => assign_f32(&mut self.down_sample_ratio, &value),
            "pulsePeriod" => assign_f32(&mut self.pulse_period, &value),
            "patternTexture" => match value {
                ParamValue::Texture(t) => {
                    self.pattern_texture = t;
                    true
                }
                _ => false,
            },
            "resolution" => value.as_array::<2>().map(|r| self.resolution = r).is_some(),
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
