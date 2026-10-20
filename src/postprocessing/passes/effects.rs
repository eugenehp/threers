//! Single-shader effects and the passes built on a few of them: FilmPass,
//! DotScreenPass, HalftonePass, GlitchPass, AfterimagePass, LUTPass,
//! SMAAPass, BokehPass.

use std::sync::Arc;

use super::basic::{out, shader_fx, ShaderKind, Uniforms};
use crate::materials::{BasicMaterial, Material};
use crate::math::Color;
use crate::postprocessing::context::{kinds, Dest, Fx, PassContext, SceneDraw};
use crate::postprocessing::params::{assign_bool, assign_f32, ParamInfo, ParamKind, ParamValue};
use crate::postprocessing::pass::{pass_common, Pass, PassState};
use crate::postprocessing::random::{rand_float, rand_int, random};
use crate::postprocessing::target::Target;
use crate::renderer::{PostFxBlend, PostFxNoise};

/// The `get` / `set` / `params` of a pass whose parameters are all uniforms.
macro_rules! uniform_params {
    () => {
        fn params(&self) -> Vec<ParamInfo> {
            self.uniforms.infos()
        }
        fn get(&self, name: &str) -> Option<ParamValue> {
            self.uniforms.0.get(name).cloned()
        }
        fn set(&mut self, name: &str, value: ParamValue) -> bool {
            self.uniforms.set(name, value)
        }
    };
}

/// A shader draw to the pass's output, cleared first when the pass asks.
fn shader_draw(ctx: &mut PassContext<'_, '_>, state: &PassState, fx: Fx, write: &Target) {
    let dest = out(state, write);
    if state.clear && !state.render_to_screen {
        ctx.clear(dest, Color::BLACK, 0.0);
    }
    ctx.fx(fx.blend(PostFxBlend::Overwrite), dest);
}

// ---------------------------------------------------------------------------

/// three.js `FilmPass(intensity, grayscale)`: `uniforms.time` advances by
/// each frame's delta.
#[derive(Debug)]
pub struct FilmPass {
    pub state: PassState,
    pub uniforms: Uniforms,
}

impl Default for FilmPass {
    fn default() -> Self {
        Self::new(0.5, false)
    }
}

impl FilmPass {
    pub fn new(intensity: f32, grayscale: bool) -> Self {
        let mut uniforms = Uniforms::from_defaults(ShaderKind::Film.uniforms());
        uniforms.set("intensity", ParamValue::Float(intensity));
        uniforms.set("grayscale", ParamValue::Bool(grayscale));
        Self {
            state: PassState::default(),
            uniforms,
        }
    }
}

impl Pass for FilmPass {
    pass_common!("FilmPass");
    uniform_params!();

    fn render(
        &mut self,
        ctx: &mut PassContext<'_, '_>,
        write: &Target,
        read: &Target,
        delta: f32,
        _m: bool,
    ) {
        let t = self.uniforms.f("time") + delta;
        self.uniforms.set("time", ParamValue::Float(t));
        let fx = shader_fx(ShaderKind::Film, &self.uniforms, read);
        shader_draw(ctx, &self.state, fx, write);
    }
}

// ---------------------------------------------------------------------------

/// three.js `DotScreenPass(center, angle, scale)`: `uniforms.tSize` follows
/// the read buffer.
#[derive(Debug)]
pub struct DotScreenPass {
    pub state: PassState,
    pub uniforms: Uniforms,
}

impl Default for DotScreenPass {
    fn default() -> Self {
        Self::new(None, None, None)
    }
}

impl DotScreenPass {
    pub fn new(center: Option<[f32; 2]>, angle: Option<f32>, scale: Option<f32>) -> Self {
        let mut uniforms = Uniforms::from_defaults(ShaderKind::DotScreen.uniforms());
        if let Some(c) = center {
            uniforms.set("center", ParamValue::Vec2(c));
        }
        if let Some(a) = angle {
            uniforms.set("angle", ParamValue::Float(a));
        }
        if let Some(s) = scale {
            uniforms.set("scale", ParamValue::Float(s));
        }
        Self {
            state: PassState::default(),
            uniforms,
        }
    }
}

impl Pass for DotScreenPass {
    pass_common!("DotScreenPass");
    uniform_params!();

    fn render(
        &mut self,
        ctx: &mut PassContext<'_, '_>,
        write: &Target,
        read: &Target,
        _d: f32,
        _m: bool,
    ) {
        self.uniforms.set(
            "tSize",
            ParamValue::Vec2([read.width() as f32, read.height() as f32]),
        );
        let fx = shader_fx(ShaderKind::DotScreen, &self.uniforms, read);
        shader_draw(ctx, &self.state, fx, write);
    }
}

// ---------------------------------------------------------------------------

/// three.js `HalftonePass(width, height, params)`; `params` are uniform
/// names (`shape`, `radius`, `rotateR`, … `disable`).
#[derive(Debug)]
pub struct HalftonePass {
    pub state: PassState,
    pub uniforms: Uniforms,
}

impl HalftonePass {
    pub fn new(width: u32, height: u32, params: &[(&str, ParamValue)]) -> Self {
        let mut uniforms = Uniforms::from_defaults(ShaderKind::Halftone.uniforms());
        uniforms.set("width", ParamValue::Float(width as f32));
        uniforms.set("height", ParamValue::Float(height as f32));
        for (k, v) in params {
            uniforms.set(k, v.clone());
        }
        Self {
            state: PassState::default(),
            uniforms,
        }
    }
}

impl Pass for HalftonePass {
    pass_common!("HalftonePass");
    uniform_params!();

    fn set_size(&mut self, width: u32, height: u32) {
        self.uniforms.set("width", ParamValue::Float(width as f32));
        self.uniforms
            .set("height", ParamValue::Float(height as f32));
    }

    fn render(
        &mut self,
        ctx: &mut PassContext<'_, '_>,
        write: &Target,
        read: &Target,
        _d: f32,
        _m: bool,
    ) {
        let fx = shader_fx(ShaderKind::Halftone, &self.uniforms, read);
        shader_draw(ctx, &self.state, fx, write);
    }
}

// ---------------------------------------------------------------------------

/// three.js `GlitchPass(dt_size)`.
///
/// Its randomness — the displacement map, and every frame's trigger and
/// glitch — is drawn from [`random`], in three.js's order and number of
/// draws (the sixteen its constructor spends on object UUIDs included), so
/// seeding the stream with [`seed_random`](crate::postprocessing::random::seed_random)
/// reproduces a seeded three.js run frame for frame.
#[derive(Debug)]
pub struct GlitchPass {
    pub state: PassState,
    pub uniforms: Uniforms,
    pub go_wild: bool,
    pub cur_f: i64,
    pub rand_x: i64,
    /// three.js's DigitalGlitch adds "snow" (per-pixel noise). `false` draws
    /// the glitch without it — the parity scenes strip it from three.js's
    /// shader too, since a `sin` hash differs between GPUs.
    pub snow: bool,
    height_map: Vec<f32>,
    dt_size: u32,
}

impl Default for GlitchPass {
    fn default() -> Self {
        Self::new(64)
    }
}

impl GlitchPass {
    pub fn new(dt_size: u32) -> Self {
        let dt_size = dt_size.max(1);
        let height_map = (0..dt_size * dt_size)
            .map(|_| rand_float(0.0, 1.0) as f32)
            .collect();
        // three.js's constructor then makes a DataTexture (with its Source),
        // a ShaderMaterial and a FullScreenQuad's Mesh: four UUIDs, four
        // `Math.random()` each.
        for _ in 0..16 {
            random();
        }
        let mut p = Self {
            state: PassState::default(),
            uniforms: Uniforms::from_defaults(ShaderKind::DigitalGlitch.uniforms()),
            go_wild: false,
            cur_f: 0,
            rand_x: 0,
            snow: true,
            height_map,
            dt_size,
        };
        p.generate_trigger();
        p
    }

    /// three.js `generateTrigger`.
    pub fn generate_trigger(&mut self) {
        self.rand_x = rand_int(120, 240);
    }

    pub fn height_map(&self) -> &[f32] {
        &self.height_map
    }

    /// Advance the uniforms one frame, as three.js's `render` does before it
    /// draws.
    pub fn step(&mut self) {
        use std::f64::consts::PI;
        let u = &mut self.uniforms;
        let f = |v: f64| ParamValue::Float(v as f32);
        u.set("seed", f(random()));
        u.set("byp", ParamValue::Int(0));
        if self.cur_f % self.rand_x == 0 || self.go_wild {
            u.set("amount", f(random() / 30.0));
            u.set("angle", f(rand_float(-PI, PI)));
            u.set("seed_x", f(rand_float(-1.0, 1.0)));
            u.set("seed_y", f(rand_float(-1.0, 1.0)));
            u.set("distortion_x", f(rand_float(0.0, 1.0)));
            u.set("distortion_y", f(rand_float(0.0, 1.0)));
            self.cur_f = 0;
            self.generate_trigger();
        } else if ((self.cur_f % self.rand_x) as f64) < self.rand_x as f64 / 5.0 {
            u.set("amount", f(random() / 90.0));
            u.set("angle", f(rand_float(-PI, PI)));
            u.set("distortion_x", f(rand_float(0.0, 1.0)));
            u.set("distortion_y", f(rand_float(0.0, 1.0)));
            u.set("seed_x", f(rand_float(-0.3, 0.3)));
            u.set("seed_y", f(rand_float(-0.3, 0.3)));
        } else if !self.go_wild {
            u.set("byp", ParamValue::Int(1));
        }
        self.cur_f += 1;
    }
}

impl Pass for GlitchPass {
    pass_common!("GlitchPass");

    fn render(
        &mut self,
        ctx: &mut PassContext<'_, '_>,
        write: &Target,
        read: &Target,
        _d: f32,
        _m: bool,
    ) {
        self.step();
        ctx.renderer.set_glitch_disp(&self.height_map, self.dt_size);
        let mut fx = shader_fx(ShaderKind::DigitalGlitch, &self.uniforms, read);
        if !self.snow {
            fx.op.kind = kinds::GLITCH_NO_SNOW;
        }
        shader_draw(ctx, &self.state, fx, write);
    }

    fn params(&self) -> Vec<ParamInfo> {
        let mut p = self.uniforms.infos();
        p.extend([
            ParamInfo::prop("goWild", ParamKind::Bool),
            ParamInfo::prop("curF", ParamKind::Int),
            ParamInfo::prop("randX", ParamKind::Int),
            ParamInfo::prop("snow", ParamKind::Bool),
        ]);
        p
    }

    fn get(&self, name: &str) -> Option<ParamValue> {
        match name {
            "goWild" => Some(ParamValue::Bool(self.go_wild)),
            "curF" => Some(ParamValue::Int(self.cur_f as i32)),
            "randX" => Some(ParamValue::Int(self.rand_x as i32)),
            "snow" => Some(ParamValue::Bool(self.snow)),
            _ => self.uniforms.0.get(name).cloned(),
        }
    }

    fn set(&mut self, name: &str, value: ParamValue) -> bool {
        match name {
            "goWild" => assign_bool(&mut self.go_wild, &value),
            "snow" => assign_bool(&mut self.snow, &value),
            "curF" => value.as_f32().map(|v| self.cur_f = v as i64).is_some(),
            "randX" => value
                .as_f32()
                .map(|v| self.rand_x = (v as i64).max(1))
                .is_some(),
            _ => self.uniforms.set(name, value),
        }
    }
}

// ---------------------------------------------------------------------------

/// three.js `AfterimagePass(damp)`: each frame is the max of the new one and
/// the last composite faded by `damp`.
#[derive(Debug)]
pub struct AfterimagePass {
    pub state: PassState,
    pub uniforms: Uniforms,
    comp: Option<Target>,
    old: Option<Target>,
}

impl Default for AfterimagePass {
    fn default() -> Self {
        Self::new(0.96)
    }
}

impl AfterimagePass {
    pub fn new(damp: f32) -> Self {
        Self {
            state: PassState::default(),
            uniforms: Uniforms::from_defaults(vec![("damp", ParamValue::Float(damp))]),
            comp: None,
            old: None,
        }
    }
}

impl Pass for AfterimagePass {
    pass_common!("AfterimagePass");
    uniform_params!();

    fn render(
        &mut self,
        ctx: &mut PassContext<'_, '_>,
        write: &Target,
        read: &Target,
        _d: f32,
        _m: bool,
    ) {
        let (w, h) = read.size();
        let comp = ctx.target(&mut self.comp, w, h);
        let old = ctx.target(&mut self.old, w, h);
        ctx.fx(
            Fx::new(kinds::AFTERIMAGE).input(read).normal(&old).p2([
                self.uniforms.f("damp"),
                0.0,
                0.0,
                0.0,
            ]),
            Dest::Target(&comp),
        );
        // The copy is a MeshBasicMaterial's: converted onto the screen.
        let dest = out(&self.state, write);
        if self.state.clear && !self.state.render_to_screen {
            ctx.clear(dest, Color::BLACK, 0.0);
        }
        ctx.fx(Fx::basic_copy(&comp).blend(PostFxBlend::Overwrite), dest);
        std::mem::swap(&mut self.comp, &mut self.old);
    }
}

// ---------------------------------------------------------------------------

/// three.js `LUTPass({ lut, intensity })`. The LUT is a cube of RGBA texels,
/// three.js's `Data3DTexture` layout (`x + y·size + z·size²`), set with
/// [`set_lut`](Self::set_lut).
#[derive(Debug)]
pub struct LutPass {
    pub state: PassState,
    pub intensity: f32,
    lut: Option<(u32, Arc<Vec<[f32; 4]>>)>,
    atlas: Option<Target>,
    uploaded: Option<usize>,
}

impl Default for LutPass {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl LutPass {
    pub fn new(intensity: f32) -> Self {
        Self {
            state: PassState::default(),
            intensity,
            lut: None,
            atlas: None,
            uploaded: None,
        }
    }

    /// The LUT: `size³` texels, or `None` to pass the image through.
    pub fn set_lut(&mut self, size: u32, texels: Option<Vec<[f32; 4]>>) {
        self.lut = texels
            .filter(|t| size > 0 && t.len() >= (size * size * size) as usize)
            .map(|t| (size, Arc::new(t)));
        self.uploaded = None;
    }

    pub fn lut_size(&self) -> u32 {
        self.lut.as_ref().map(|(s, _)| *s).unwrap_or(0)
    }
}

impl Pass for LutPass {
    pass_common!("LUTPass");

    fn render(
        &mut self,
        ctx: &mut PassContext<'_, '_>,
        write: &Target,
        read: &Target,
        _d: f32,
        _m: bool,
    ) {
        let Some((size, texels)) = &self.lut else {
            // No LUT: three.js's shader samples nothing and mixes it in by
            // `intensity`; the useful reading of that is the image unchanged.
            shader_draw(ctx, &self.state, Fx::copy(read, 1.0), write);
            return;
        };
        let size = *size;
        let atlas = ctx.target(&mut self.atlas, size * size, size);
        let key = Arc::as_ptr(texels) as usize;
        if self.uploaded != Some(key) {
            // Slices side by side: (x, y, z) at (z·size + x, y).
            let mut data = vec![[0f32; 4]; (size * size * size) as usize];
            for z in 0..size {
                for y in 0..size {
                    for x in 0..size {
                        data[(y * size * size + z * size + x) as usize] =
                            texels[(x + y * size + z * size * size) as usize];
                    }
                }
            }
            ctx.renderer.upload_render_target_f16(atlas.id(), &data);
            self.uploaded = Some(key);
        }
        let fx = Fx::new(kinds::LUT)
            .input(read)
            .normal(&atlas)
            .linear_aux()
            .p2([size as f32, self.intensity, 0.0, 0.0]);
        shader_draw(ctx, &self.state, fx, write);
    }

    fn params(&self) -> Vec<ParamInfo> {
        vec![
            ParamInfo::prop("intensity", ParamKind::Float),
            ParamInfo::uniform("lutSize", ParamKind::Float),
        ]
    }

    fn get(&self, name: &str) -> Option<ParamValue> {
        match name {
            "intensity" => Some(ParamValue::Float(self.intensity)),
            "lutSize" => Some(ParamValue::Float(self.lut_size() as f32)),
            _ => None,
        }
    }

    fn set(&mut self, name: &str, value: ParamValue) -> bool {
        name == "intensity" && assign_f32(&mut self.intensity, &value)
    }
}

// ---------------------------------------------------------------------------

/// SMAA's lookup tables: the area table's width, height and texels, then the
/// search table's.
type SmaaTables = (u32, u32, Vec<[f32; 4]>, u32, u32, Vec<[f32; 4]>);

/// SMAA's lookup tables, decoded once.
fn smaa_tables() -> &'static SmaaTables {
    static TABLES: std::sync::OnceLock<SmaaTables> = std::sync::OnceLock::new();
    TABLES.get_or_init(|| {
        let decode = |bytes: &[u8]| {
            let img = crate::utils::png::decode_png(bytes).expect("SMAA table PNG");
            let texels = img
                .rgba
                .chunks_exact(4)
                .map(|p| {
                    [
                        p[0] as f32 / 255.0,
                        p[1] as f32 / 255.0,
                        p[2] as f32 / 255.0,
                        p[3] as f32 / 255.0,
                    ]
                })
                .collect::<Vec<_>>();
            (img.width, img.height, texels)
        };
        // three.js SMAAPass's `getAreaTexture()` / `getSearchTexture()` images.
        let (aw, ah, area) = decode(include_bytes!("../smaa_area.png"));
        let (sw, sh, search) = decode(include_bytes!("../smaa_search.png"));
        (aw, ah, area, sw, sh, search)
    })
}

/// three.js `SMAAPass(width, height)`: SMAA 1x medium — colour edges,
/// blending weights from the area and search tables, neighbourhood blend.
#[derive(Debug, Default)]
pub struct SmaaPass {
    pub state: PassState,
    width: u32,
    height: u32,
    edges: Option<Target>,
    weights: Option<Target>,
    area: Option<Target>,
    search: Option<Target>,
}

impl SmaaPass {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            state: PassState::default(),
            width: width.max(1),
            height: height.max(1),
            ..Default::default()
        }
    }
}

impl Pass for SmaaPass {
    pass_common!("SMAAPass");

    fn set_size(&mut self, width: u32, height: u32) {
        self.width = width.max(1);
        self.height = height.max(1);
    }

    fn render(
        &mut self,
        ctx: &mut PassContext<'_, '_>,
        write: &Target,
        read: &Target,
        _d: f32,
        _m: bool,
    ) {
        let (w, h) = (self.width, self.height);
        if self.area.is_none() || self.search.is_none() {
            let (aw, ah, area, sw, sh, search) = smaa_tables();
            let a = ctx.target(&mut self.area, *aw, *ah);
            ctx.renderer.upload_render_target_f16(a.id(), area);
            let s = ctx.target(&mut self.search, *sw, *sh);
            ctx.renderer.upload_render_target_f16(s.id(), search);
        }
        let (Some(area), Some(search)) = (self.area.clone(), self.search.clone()) else {
            return;
        };
        let edges = ctx.target(&mut self.edges, w, h);
        let weights = ctx.target(&mut self.weights, w, h);
        let blend = if self.state.clear {
            PostFxBlend::Replace
        } else {
            PostFxBlend::Overwrite
        };
        ctx.fx(
            Fx::new(kinds::SMAA_EDGES).input(read).blend(blend),
            Dest::Target(&edges),
        );
        ctx.fx(
            Fx::new(kinds::SMAA_WEIGHTS)
                .input(&edges)
                .depth(&area)
                .linear_aux()
                .noise(PostFxNoise::Color(search.id()))
                .blend(blend),
            Dest::Target(&weights),
        );
        let fx = Fx::new(kinds::SMAA_BLEND).input(read).normal(&weights);
        shader_draw(ctx, &self.state, fx, write);
    }
}

// ---------------------------------------------------------------------------

/// three.js `BokehPass(scene, camera, { focus, aperture, maxblur })`: depth
/// of field from a depth prepass of the scene.
#[derive(Debug)]
pub struct BokehPass {
    pub state: PassState,
    pub scene: usize,
    pub camera: usize,
    pub uniforms: Uniforms,
    depth: Option<Target>,
    depth_material: Arc<Material>,
}

impl BokehPass {
    pub fn new(scene: usize, camera: usize, focus: f32, aperture: f32, maxblur: f32) -> Self {
        Self {
            state: PassState::default(),
            scene,
            camera,
            uniforms: Uniforms::from_defaults(vec![
                ("focus", ParamValue::Float(focus)),
                ("aperture", ParamValue::Float(aperture)),
                ("maxblur", ParamValue::Float(maxblur)),
                ("aspect", ParamValue::Float(1.0)),
                ("nearClip", ParamValue::Float(0.1)),
                ("farClip", ParamValue::Float(1000.0)),
            ]),
            depth: None,
            // three.js draws the depth pass with an opaque MeshDepthMaterial;
            // only the depth it writes is read here, so any opaque material
            // does.
            depth_material: Arc::new(Material::Basic(BasicMaterial::new(Color::WHITE))),
        }
    }
}

impl Pass for BokehPass {
    pass_common!("BokehPass");
    uniform_params!();

    fn set_size(&mut self, width: u32, height: u32) {
        self.uniforms.set(
            "aspect",
            ParamValue::Float(width as f32 / height.max(1) as f32),
        );
    }

    fn render(
        &mut self,
        ctx: &mut PassContext<'_, '_>,
        write: &Target,
        read: &Target,
        _d: f32,
        _m: bool,
    ) {
        let depth = ctx.target(&mut self.depth, read.width(), read.height());
        ctx.render_scene(
            Dest::Target(&depth),
            &SceneDraw {
                scene: self.scene,
                camera: self.camera,
                override_material: Some(self.depth_material.clone()),
                clear: Some((Color::WHITE, 1.0)),
                ..SceneDraw::default()
            },
        );
        let mut camera = ctx.postfx_camera(self.camera);
        self.uniforms
            .set("nearClip", ParamValue::Float(camera.near));
        self.uniforms.set("farClip", ParamValue::Float(camera.far));
        camera.near = self.uniforms.f("nearClip");
        camera.far = self.uniforms.f("farClip");
        let fx = Fx::new(kinds::BOKEH)
            .input(read)
            .hw_depth(&depth)
            .camera(camera)
            .p2([
                self.uniforms.f("focus"),
                self.uniforms.f("aperture"),
                self.uniforms.f("maxblur"),
                self.uniforms.f("aspect"),
            ]);
        // three.js clears writeBuffer before the draw; the screen it does not.
        let dest = out(&self.state, write);
        let blend = if self.state.render_to_screen {
            PostFxBlend::Overwrite
        } else {
            PostFxBlend::Replace
        };
        ctx.fx(fx.blend(blend), dest);
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
