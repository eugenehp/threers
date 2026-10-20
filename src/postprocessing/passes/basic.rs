//! The structural passes: RenderPass, ShaderPass and its shaders, TexturePass,
//! CubeTexturePass, ClearPass, MaskPass / ClearMaskPass, OutputPass, and
//! DrawPass for a pass written as a closure.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::materials::Material;
use crate::math::Color;
use crate::postprocessing::context::{kinds, Dest, Fx, PassContext, SceneDraw};
use crate::postprocessing::params::{assign_bool, assign_f32, ParamInfo, ParamKind, ParamValue};
use crate::postprocessing::pass::{pass_common, MaskRole, Pass, PassState, PassThreading};
use crate::postprocessing::target::Target;
use crate::postprocessing::upload;
use crate::renderer::PostFxBlend;
use crate::textures::{CubeTexture, Texture};

/// Where a pass's output goes: the screen when it is the last, else `write`.
pub(crate) fn out<'t>(state: &PassState, write: &'t Target) -> Dest<'t> {
    if state.render_to_screen {
        Dest::Screen
    } else {
        Dest::Target(write)
    }
}

/// Where a pass that draws in place goes: the screen when it is the last,
/// else `read`.
pub(crate) fn in_place<'t>(state: &PassState, read: &'t Target) -> Dest<'t> {
    if state.render_to_screen {
        Dest::Screen
    } else {
        Dest::Target(read)
    }
}

// ---------------------------------------------------------------------------

/// three.js `RenderPass(scene, camera, overrideMaterial, clearColor, clearAlpha)`:
/// renders the scene into `readBuffer` (or the screen), without swapping.
#[derive(Debug)]
pub struct RenderPass {
    pub state: PassState,
    /// Index of the scene in the composer's list.
    pub scene: usize,
    /// Index of the camera.
    pub camera: usize,
    pub override_material: Option<Arc<Material>>,
    pub clear_color: Option<Color>,
    pub clear_alpha: Option<f32>,
    pub clear_depth: bool,
}

impl Default for RenderPass {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderPass {
    /// Scene 0 through camera 0.
    pub fn new() -> Self {
        Self {
            state: PassState {
                clear: true,
                ..PassState::swapping(false)
            },
            scene: 0,
            camera: 0,
            override_material: None,
            clear_color: None,
            clear_alpha: None,
            clear_depth: false,
        }
    }

    /// Scene `scene` through camera `camera`.
    pub fn with_slots(scene: usize, camera: usize) -> Self {
        Self {
            scene,
            camera,
            ..Self::new()
        }
    }
}

impl Pass for RenderPass {
    pass_common!("RenderPass");

    fn render(&mut self, ctx: &mut PassContext<'_, '_>, _write: &Target, read: &Target, _delta: f32, _mask: bool) {
        let clear = if self.state.clear && (self.clear_color.is_some() || self.clear_alpha.is_some()) {
            let background = ctx
                .scene_mut(self.scene)
                .map(|s| (s.background, s.background_alpha))
                .unwrap_or((Color::BLACK, 1.0));
            Some((
                self.clear_color.unwrap_or(background.0),
                self.clear_alpha.unwrap_or(background.1),
            ))
        } else {
            None
        };
        let draw = SceneDraw {
            scene: self.scene,
            camera: self.camera,
            override_material: self.override_material.clone(),
            clear,
            keep: !self.state.clear,
            ..SceneDraw::default()
        };
        ctx.render_scene(in_place(&self.state, read), &draw);
    }

    fn params(&self) -> Vec<ParamInfo> {
        vec![
            ParamInfo::prop("clearDepth", ParamKind::Bool),
            ParamInfo::prop("clearAlpha", ParamKind::Float),
            ParamInfo::prop("clearColor", ParamKind::Color),
        ]
    }

    fn get(&self, name: &str) -> Option<ParamValue> {
        match name {
            "clearDepth" => Some(ParamValue::Bool(self.clear_depth)),
            "clearAlpha" => self.clear_alpha.map(ParamValue::Float),
            "clearColor" => self.clear_color.map(|c| ParamValue::Color([c.r, c.g, c.b])),
            _ => None,
        }
    }

    fn set(&mut self, name: &str, value: ParamValue) -> bool {
        match name {
            "clearDepth" => assign_bool(&mut self.clear_depth, &value),
            "clearAlpha" => {
                self.clear_alpha = value.as_f32();
                true
            }
            "clearColor" => {
                self.clear_color = value.as_array::<3>().map(|c| Color::new(c[0], c[1], c[2]));
                true
            }
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

/// The shaders a [`ShaderPass`] can run: three.js's
/// `examples/jsm/shaders` that have a `POSTFX_SHADER` kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderKind {
    /// `CopyShader`: `opacity`.
    Copy,
    /// `FXAAShader`: `resolution` (the texel size; three.js's default is
    /// `1/1024, 1/512`, which the caller is expected to set).
    Fxaa,
    /// `FilmShader`: `intensity`, `grayscale`, `time`.
    Film,
    /// `DotScreenShader`: `center`, `angle`, `scale`, `tSize`.
    DotScreen,
    /// `HalftoneShader`.
    Halftone,
    /// `DigitalGlitch`, with its uniforms set by hand (GlitchPass sets them
    /// itself).
    DigitalGlitch,
    /// `LuminosityShader`.
    Luminosity,
    /// `LuminosityHighPassShader`.
    LuminosityHighPass,
    /// `OutputShader`: `toneMappingExposure`, plus `toneMapping` and
    /// `outputColorSpace` (which three.js takes from the renderer).
    Output,
}

impl ShaderKind {
    /// From a three.js shader's `name`.
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "CopyShader" => Self::Copy,
            "FXAAShader" => Self::Fxaa,
            "FilmShader" => Self::Film,
            "DotScreenShader" => Self::DotScreen,
            "HalftoneShader" => Self::Halftone,
            "DigitalGlitch" => Self::DigitalGlitch,
            "LuminosityShader" => Self::Luminosity,
            "LuminosityHighPassShader" => Self::LuminosityHighPass,
            "OutputShader" => Self::Output,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Copy => "CopyShader",
            Self::Fxaa => "FXAAShader",
            Self::Film => "FilmShader",
            Self::DotScreen => "DotScreenShader",
            Self::Halftone => "HalftoneShader",
            Self::DigitalGlitch => "DigitalGlitch",
            Self::Luminosity => "LuminosityShader",
            Self::LuminosityHighPass => "LuminosityHighPassShader",
            Self::Output => "OutputShader",
        }
    }

    /// The shader's uniforms and three.js's default values.
    pub fn uniforms(self) -> Vec<(&'static str, ParamValue)> {
        use std::f32::consts::PI;
        use ParamValue::*;
        match self {
            Self::Copy => vec![("opacity", Float(1.0))],
            Self::Fxaa => vec![("resolution", Vec2([1.0 / 1024.0, 1.0 / 512.0]))],
            Self::Film => vec![
                ("intensity", Float(0.5)),
                ("grayscale", Bool(false)),
                ("time", Float(0.0)),
            ],
            Self::DotScreen => vec![
                ("tSize", Vec2([256.0, 256.0])),
                ("center", Vec2([0.5, 0.5])),
                ("angle", Float(1.57)),
                ("scale", Float(1.0)),
            ],
            Self::Halftone => vec![
                ("shape", Int(1)),
                ("radius", Float(4.0)),
                ("rotateR", Float(PI / 12.0)),
                ("rotateG", Float(PI / 12.0 * 2.0)),
                ("rotateB", Float(PI / 12.0 * 3.0)),
                ("scatter", Float(0.0)),
                ("width", Float(1.0)),
                ("height", Float(1.0)),
                ("blending", Float(1.0)),
                ("blendingMode", Int(1)),
                ("greyscale", Bool(false)),
                ("disable", Bool(false)),
            ],
            Self::DigitalGlitch => vec![
                ("byp", Int(0)),
                ("amount", Float(0.08)),
                ("angle", Float(0.02)),
                ("seed", Float(0.02)),
                ("seed_x", Float(0.02)),
                ("seed_y", Float(0.02)),
                ("distortion_x", Float(0.5)),
                ("distortion_y", Float(0.6)),
                ("col_s", Float(0.05)),
            ],
            Self::Luminosity => vec![],
            Self::LuminosityHighPass => vec![
                ("luminosityThreshold", Float(1.0)),
                ("smoothWidth", Float(1.0)),
                ("defaultColor", Color([0.0, 0.0, 0.0])),
                ("defaultOpacity", Float(0.0)),
            ],
            Self::Output => vec![
                ("toneMappingExposure", Float(1.0)),
                ("toneMapping", Int(0)),
                ("srgbTransfer", Bool(true)),
            ],
        }
    }
}

/// The uniforms of a shader-backed pass, by name.
#[derive(Clone, Debug, Default)]
pub struct Uniforms(pub BTreeMap<&'static str, ParamValue>);

impl Uniforms {
    pub fn from_defaults(defaults: Vec<(&'static str, ParamValue)>) -> Self {
        Self(defaults.into_iter().collect())
    }

    pub fn f(&self, name: &str) -> f32 {
        self.0.get(name).and_then(|v| v.as_f32()).unwrap_or(0.0)
    }

    pub fn b(&self, name: &str) -> bool {
        self.0.get(name).and_then(|v| v.as_bool()).unwrap_or(false)
    }

    pub fn v2(&self, name: &str) -> [f32; 2] {
        self.0.get(name).and_then(|v| v.as_array::<2>()).unwrap_or([0.0; 2])
    }

    pub fn v3(&self, name: &str) -> [f32; 3] {
        self.0.get(name).and_then(|v| v.as_array::<3>()).unwrap_or([0.0; 3])
    }

    /// Assign `name` if the shader has it, keeping its shape.
    pub fn set(&mut self, name: &str, value: ParamValue) -> bool {
        let Some(slot) = self.0.get_mut(name) else {
            return false;
        };
        let new = match slot {
            ParamValue::Float(_) => value.as_f32().map(ParamValue::Float),
            ParamValue::Int(_) => value.as_f32().map(|v| ParamValue::Int(v as i32)),
            ParamValue::Bool(_) => value.as_bool().map(ParamValue::Bool),
            ParamValue::Vec2(_) => value.as_array::<2>().map(ParamValue::Vec2),
            ParamValue::Vec3(_) => value.as_array::<3>().map(ParamValue::Vec3),
            ParamValue::Vec4(_) => value.as_array::<4>().map(ParamValue::Vec4),
            ParamValue::Color(_) => value.as_array::<3>().map(ParamValue::Color),
            ParamValue::Floats(_) => value.as_floats().map(ParamValue::Floats),
            _ => Some(value),
        };
        match new {
            Some(v) => {
                *slot = v;
                true
            }
            None => false,
        }
    }

    pub fn infos(&self) -> Vec<ParamInfo> {
        self.0
            .iter()
            .map(|(name, v)| {
                let kind = match v {
                    ParamValue::Float(_) => ParamKind::Float,
                    ParamValue::Int(_) => ParamKind::Int,
                    ParamValue::Bool(_) => ParamKind::Bool,
                    ParamValue::Vec2(_) => ParamKind::Vec2,
                    ParamValue::Vec3(_) => ParamKind::Vec3,
                    ParamValue::Vec4(_) => ParamKind::Vec4,
                    ParamValue::Color(_) => ParamKind::Color,
                    ParamValue::Floats(_) => ParamKind::Floats,
                    ParamValue::Texture(_) => ParamKind::Texture,
                    ParamValue::CubeTexture(_) => ParamKind::CubeTexture,
                    ParamValue::Objects(_) => ParamKind::Objects,
                };
                ParamInfo::uniform(name, kind)
            })
            .collect()
    }
}

/// The draw for `kind`'s shader over `read` with `uniforms`.
pub(crate) fn shader_fx(kind: ShaderKind, u: &Uniforms, read: &Target) -> Fx {
    match kind {
        ShaderKind::Copy => Fx::copy(read, u.f("opacity")),
        ShaderKind::Fxaa => {
            let r = u.v2("resolution");
            Fx::new(kinds::FXAA).input(read).p2([r[0], r[1], 0.0, 0.0])
        }
        ShaderKind::Film => Fx::new(kinds::FILM)
            .input(read)
            .time(u.f("time"))
            .p2([u.f("intensity"), if u.b("grayscale") { 1.0 } else { 0.0 }, 0.0, 0.0]),
        ShaderKind::DotScreen => {
            let c = u.v2("center");
            let t = u.v2("tSize");
            Fx::new(kinds::DOT_SCREEN)
                .input(read)
                .p2([u.f("angle"), u.f("scale"), c[0], c[1]])
                .p3([t[0], t[1], 0.0, 0.0])
        }
        ShaderKind::Halftone => {
            let mut fx = Fx::new(kinds::HALFTONE)
                .input(read)
                .p2([u.f("radius"), u.f("scatter"), u.f("shape"), u.f("blending")])
                .p3([u.f("rotateR"), u.f("rotateG"), u.f("rotateB"), u.f("blendingMode")]);
            // `width` / `height` ride in params.yz and `greyscale` in the
            // camera block's first slot, where the halftone branch reads them.
            fx = fx.time(u.f("width")).p0z(u.f("height"));
            fx.op.camera.near = if u.b("greyscale") { 1.0 } else { 0.0 };
            if u.b("disable") {
                fx.op.kind = kinds::COPY;
                fx.op.p2 = [1.0, 0.0, 0.0, 0.0];
            }
            fx
        }
        ShaderKind::DigitalGlitch => Fx::new(kinds::GLITCH)
        .input(read)
        .time(u.f("seed"))
        .p0z(u.f("byp"))
        .p2([u.f("amount"), u.f("angle"), u.f("distortion_x"), u.f("distortion_y")])
        .p3([u.f("seed_x"), u.f("seed_y"), u.f("col_s"), u.f("byp")]),
        ShaderKind::Luminosity => Fx::new(kinds::LUMINOSITY).input(read),
        ShaderKind::LuminosityHighPass => {
            let c = u.v3("defaultColor");
            Fx::new(kinds::LUMINOSITY_HIGH_PASS)
                .input(read)
                .p2([u.f("luminosityThreshold"), u.f("smoothWidth"), u.f("defaultOpacity"), 0.0])
                .p3([c[0], c[1], c[2], 0.0])
        }
        ShaderKind::Output => Fx::new(kinds::OUTPUT).input(read).p2([
            u.f("toneMapping"),
            u.f("toneMappingExposure"),
            if u.b("srgbTransfer") { 1.0 } else { 0.0 },
            0.0,
        ]),
    }
}

/// three.js `ShaderPass(shader)` for the shaders in [`ShaderKind`]. A shader
/// it does not know — user GLSL — cannot run here; such a pass copies.
#[derive(Debug)]
pub struct ShaderPass {
    pub state: PassState,
    pub kind: ShaderKind,
    pub uniforms: Uniforms,
}

impl ShaderPass {
    pub fn new(kind: ShaderKind) -> Self {
        Self {
            state: PassState::default(),
            kind,
            uniforms: Uniforms::from_defaults(kind.uniforms()),
        }
    }

    /// `ShaderPass(CopyShader)`.
    pub fn copy() -> Self {
        Self::new(ShaderKind::Copy)
    }

    /// `ShaderPass(FXAAShader)` with `resolution` set to `width × height`'s
    /// texel size, as three.js examples do.
    pub fn fxaa(width: u32, height: u32) -> Self {
        let mut p = Self::new(ShaderKind::Fxaa);
        p.uniforms.set(
            "resolution",
            ParamValue::Vec2([1.0 / width.max(1) as f32, 1.0 / height.max(1) as f32]),
        );
        p
    }
}

impl Pass for ShaderPass {
    pass_common!("ShaderPass");

    fn render(&mut self, ctx: &mut PassContext<'_, '_>, write: &Target, read: &Target, _delta: f32, _mask: bool) {
        let dest = out(&self.state, write);
        if self.state.clear && !self.state.render_to_screen {
            ctx.clear(dest, Color::BLACK, 0.0);
        }
        ctx.fx(shader_fx(self.kind, &self.uniforms, read).blend(PostFxBlend::Overwrite), dest);
    }

    fn params(&self) -> Vec<ParamInfo> {
        self.uniforms.infos()
    }

    fn get(&self, name: &str) -> Option<ParamValue> {
        self.uniforms.0.get(name).cloned()
    }

    fn set(&mut self, name: &str, value: ParamValue) -> bool {
        self.uniforms.set(name, value)
    }
}

/// `ShaderPass(CopyShader)`.
pub type CopyPass = ShaderPass;

/// `ShaderPass(FXAAShader)`, sized.
pub fn fxaa_pass(width: u32, height: u32) -> ShaderPass {
    ShaderPass::fxaa(width, height)
}

// ---------------------------------------------------------------------------

/// three.js `TexturePass(map, opacity)`: copies a texture into `readBuffer`
/// (or onto the screen), blended when `opacity < 1`.
#[derive(Debug, Default)]
pub struct TexturePass {
    pub state: PassState,
    pub map: Option<Arc<Texture>>,
    pub opacity: f32,
    uploaded: Option<(u64, Target)>,
}

impl TexturePass {
    pub fn new(map: Option<Arc<Texture>>, opacity: f32) -> Self {
        Self {
            state: PassState::swapping(false),
            map,
            opacity,
            uploaded: None,
        }
    }
}

impl Pass for TexturePass {
    pass_common!("TexturePass");

    fn render(&mut self, ctx: &mut PassContext<'_, '_>, _write: &Target, read: &Target, _delta: f32, _mask: bool) {
        let dest = in_place(&self.state, read);
        if self.state.clear {
            ctx.clear(dest, Color::BLACK, 0.0);
        }
        let Some(map) = &self.map else {
            return;
        };
        let Some(src) = upload::texture_target(ctx.renderer, map, &mut self.uploaded) else {
            return;
        };
        // `premultipliedAlpha`, and blended only when `transparent`, which
        // three.js sets from `opacity < 1`.
        let blend = if self.opacity < 1.0 {
            PostFxBlend::NormalPremultiplied
        } else {
            PostFxBlend::Overwrite
        };
        ctx.fx(Fx::copy(&src, self.opacity).blend(blend), dest);
    }

    fn params(&self) -> Vec<ParamInfo> {
        vec![
            ParamInfo::prop("map", ParamKind::Texture),
            ParamInfo::prop("opacity", ParamKind::Float),
        ]
    }

    fn get(&self, name: &str) -> Option<ParamValue> {
        match name {
            "map" => Some(ParamValue::Texture(self.map.clone())),
            "opacity" => Some(ParamValue::Float(self.opacity)),
            _ => None,
        }
    }

    fn set(&mut self, name: &str, value: ParamValue) -> bool {
        match (name, value) {
            ("map", ParamValue::Texture(t)) => {
                self.map = t;
                true
            }
            ("opacity", v) => assign_f32(&mut self.opacity, &v),
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------

/// three.js `CubeTexturePass(camera, tCube, opacity)`: the cube texture as a
/// background seen through `camera`'s rotation and projection, into
/// `readBuffer`.
#[derive(Debug, Default)]
pub struct CubeTexturePass {
    pub state: PassState,
    pub camera: usize,
    pub cube: Option<Arc<CubeTexture>>,
    pub opacity: f32,
    /// three.js `tFlip`: -1 for a cube texture from images, 1 for a cube
    /// render target's texture.
    pub flip: f32,
    atlas: Option<(usize, Target)>,
}

impl CubeTexturePass {
    pub fn new(camera: usize, cube: Option<Arc<CubeTexture>>, opacity: f32) -> Self {
        Self {
            state: PassState::swapping(false),
            camera,
            cube,
            opacity,
            flip: -1.0,
            atlas: None,
        }
    }
}

impl Pass for CubeTexturePass {
    pass_common!("CubeTexturePass");

    fn render(&mut self, ctx: &mut PassContext<'_, '_>, _write: &Target, read: &Target, _delta: f32, _mask: bool) {
        let dest = in_place(&self.state, read);
        if self.state.clear {
            ctx.clear(dest, Color::BLACK, 0.0);
        }
        let Some(cube) = &self.cube else {
            return;
        };
        let Some(atlas) = upload::cube_atlas(ctx.renderer, cube, &mut self.atlas) else {
            return;
        };
        let mut camera = ctx.postfx_camera(self.camera);
        camera.proj = ctx.camera_world(self.camera).elements;
        let blend = if self.opacity < 1.0 {
            PostFxBlend::Normal
        } else {
            PostFxBlend::Overwrite
        };
        ctx.fx(
            Fx::new(kinds::CUBE_BACKGROUND)
                .normal(&atlas)
                .linear_aux()
                .camera(camera)
                .p2([self.opacity, self.flip, cube.size as f32, 0.0])
                .blend(blend),
            dest,
        );
    }

    fn params(&self) -> Vec<ParamInfo> {
        vec![
            ParamInfo::prop("tCube", ParamKind::CubeTexture),
            ParamInfo::prop("opacity", ParamKind::Float),
            ParamInfo::prop("tFlip", ParamKind::Float),
        ]
    }

    fn get(&self, name: &str) -> Option<ParamValue> {
        match name {
            "tCube" => Some(ParamValue::CubeTexture(self.cube.clone())),
            "opacity" => Some(ParamValue::Float(self.opacity)),
            "tFlip" => Some(ParamValue::Float(self.flip)),
            _ => None,
        }
    }

    fn set(&mut self, name: &str, value: ParamValue) -> bool {
        match (name, value) {
            ("tCube", ParamValue::CubeTexture(c)) => {
                self.cube = c;
                true
            }
            ("opacity", v) => assign_f32(&mut self.opacity, &v),
            ("tFlip", v) => assign_f32(&mut self.flip, &v),
            _ => false,
        }
    }

    fn scene_slots(&self) -> Vec<(usize, usize)> {
        Vec::new()
    }

    fn set_scene_slot(&mut self, n: usize, _scene: usize, camera: usize) {
        if n == 0 {
            self.camera = camera;
        }
    }
}

// ---------------------------------------------------------------------------

/// three.js `ClearPass(clearColor, clearAlpha)`: clears `readBuffer` (or the
/// screen). three.js only applies its colour when it is truthy — `0x000000`
/// is not — and otherwise clears with the renderer's, black and opaque by
/// default; `None` here is that.
#[derive(Debug)]
pub struct ClearPass {
    pub state: PassState,
    pub clear_color: Option<Color>,
    pub clear_alpha: f32,
}

impl ClearPass {
    pub fn new(clear_color: Option<Color>, clear_alpha: f32) -> Self {
        Self {
            state: PassState::swapping(false),
            clear_color,
            clear_alpha,
        }
    }
}

impl Pass for ClearPass {
    pass_common!("ClearPass");

    fn render(&mut self, ctx: &mut PassContext<'_, '_>, _write: &Target, read: &Target, _delta: f32, _mask: bool) {
        let (c, a) = match self.clear_color {
            Some(c) => (c, self.clear_alpha),
            None => (Color::BLACK, 1.0),
        };
        ctx.clear(in_place(&self.state, read), c, a);
    }

    fn params(&self) -> Vec<ParamInfo> {
        vec![
            ParamInfo::prop("clearColor", ParamKind::Color),
            ParamInfo::prop("clearAlpha", ParamKind::Float),
        ]
    }

    fn get(&self, name: &str) -> Option<ParamValue> {
        match name {
            "clearColor" => self.clear_color.map(|c| ParamValue::Color([c.r, c.g, c.b])),
            "clearAlpha" => Some(ParamValue::Float(self.clear_alpha)),
            _ => None,
        }
    }

    fn set(&mut self, name: &str, value: ParamValue) -> bool {
        match name {
            "clearColor" => {
                // three.js tests truthiness: black (0x000000) counts as unset.
                self.clear_color = value
                    .as_array::<3>()
                    .map(|c| Color::new(c[0], c[1], c[2]))
                    .filter(|c| c.r != 0.0 || c.g != 0.0 || c.b != 0.0);
                true
            }
            "clearAlpha" => assign_f32(&mut self.clear_alpha, &value),
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------

/// three.js `MaskPass(scene, camera)`: from here until a [`ClearMaskPass`],
/// passes only change the pixels the scene covers (or, `inverse`, those it
/// does not). three.js writes the stencil buffer; here the scene's coverage
/// is a depth buffer the composer tests each masked pass's output against.
#[derive(Debug)]
pub struct MaskPass {
    pub state: PassState,
    pub scene: usize,
    pub camera: usize,
    pub inverse: bool,
    mask: Option<Target>,
}

impl MaskPass {
    pub fn new(scene: usize, camera: usize) -> Self {
        Self {
            state: PassState {
                clear: true,
                ..PassState::swapping(false)
            },
            scene,
            camera,
            inverse: false,
            mask: None,
        }
    }
}

impl Pass for MaskPass {
    pass_common!("MaskPass");

    fn render(&mut self, ctx: &mut PassContext<'_, '_>, _write: &Target, read: &Target, _delta: f32, _mask: bool) {
        let t = ctx.target(&mut self.mask, read.width(), read.height());
        let draw = SceneDraw {
            scene: self.scene,
            camera: self.camera,
            clear: Some((Color::BLACK, 0.0)),
            ..SceneDraw::default()
        };
        ctx.render_scene(Dest::Target(&t), &draw);
    }

    fn mask_role(&self) -> MaskRole {
        match &self.mask {
            Some(t) => MaskRole::Set {
                target: t.clone(),
                inverse: self.inverse,
            },
            None => MaskRole::None,
        }
    }

    fn params(&self) -> Vec<ParamInfo> {
        vec![ParamInfo::prop("inverse", ParamKind::Bool)]
    }

    fn get(&self, name: &str) -> Option<ParamValue> {
        (name == "inverse").then_some(ParamValue::Bool(self.inverse))
    }

    fn set(&mut self, name: &str, value: ParamValue) -> bool {
        name == "inverse" && assign_bool(&mut self.inverse, &value)
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

/// three.js `ClearMaskPass`: ends a [`MaskPass`].
#[derive(Debug)]
pub struct ClearMaskPass {
    pub state: PassState,
}

impl Default for ClearMaskPass {
    fn default() -> Self {
        Self::new()
    }
}

impl ClearMaskPass {
    pub fn new() -> Self {
        Self {
            state: PassState::swapping(false),
        }
    }
}

impl Pass for ClearMaskPass {
    pass_common!("ClearMaskPass");

    fn render(&mut self, _ctx: &mut PassContext<'_, '_>, _w: &Target, _r: &Target, _d: f32, _m: bool) {}

    fn mask_role(&self) -> MaskRole {
        MaskRole::Clear
    }
}

// ---------------------------------------------------------------------------

/// three.js `OutputPass`: tone mapping and the sRGB transfer, as the renderer
/// would apply them drawing to the screen. three.js takes the operator,
/// exposure and colour space from the renderer; here they are fields
/// (`toneMapping` uses three.js's constants: 0 none, 1 linear, 2 Reinhard,
/// 3 Cineon, 4 ACES filmic, 6 AgX, 7 neutral).
#[derive(Debug)]
pub struct OutputPass {
    pub state: PassState,
    pub uniforms: Uniforms,
}

impl Default for OutputPass {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputPass {
    pub fn new() -> Self {
        Self {
            state: PassState::default(),
            uniforms: Uniforms::from_defaults(ShaderKind::Output.uniforms()),
        }
    }
}

impl Pass for OutputPass {
    pass_common!("OutputPass");

    fn render(&mut self, ctx: &mut PassContext<'_, '_>, write: &Target, read: &Target, _delta: f32, _mask: bool) {
        let dest = out(&self.state, write);
        if self.state.clear && !self.state.render_to_screen {
            ctx.clear(dest, Color::BLACK, 0.0);
        }
        ctx.fx(
            shader_fx(ShaderKind::Output, &self.uniforms, read).blend(PostFxBlend::Overwrite),
            dest,
        );
    }

    fn params(&self) -> Vec<ParamInfo> {
        self.uniforms.infos()
    }

    fn get(&self, name: &str) -> Option<ParamValue> {
        self.uniforms.0.get(name).cloned()
    }

    fn set(&mut self, name: &str, value: ParamValue) -> bool {
        self.uniforms.set(name, value)
    }
}

// ---------------------------------------------------------------------------

/// What a [`DrawPass`] is given: where the chain has got to, where its output
/// goes, and the device to draw with.
pub struct DrawContext<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    /// The depth the scene was rendered with into `input` — the render pass's
    /// — for a pass that needs the geometry the colour no longer carries.
    pub depth: Option<&'a wgpu::Texture>,
    /// Camera 0's near and far planes.
    pub near_far: (f32, f32),
    /// `tan(fov_y / 2)` and the aspect, from camera 0's projection.
    pub tan_half_fov: f32,
    pub aspect: f32,
    /// Record into this; the composer submits it after the pass returns.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// `readBuffer`.
    pub input: &'a wgpu::TextureView,
    pub input_texture: &'a wgpu::Texture,
    /// `writeBuffer`, or the screen when the pass is the last.
    pub target: &'a wgpu::TextureView,
    /// The texture behind `target`; `None` for a swapchain image.
    pub target_texture: Option<&'a wgpu::Texture>,
    pub format: wgpu::TextureFormat,
    pub width: u32,
    pub height: u32,
}

impl DrawContext<'_> {
    /// The target as a [`fullscreen::Target`](crate::renderer::fullscreen::Target).
    pub fn target(&self) -> crate::renderer::fullscreen::Target<'_> {
        crate::renderer::fullscreen::Target {
            view: self.target,
            format: self.format,
            width: self.width,
            height: self.height,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
type DrawFn = Box<dyn FnMut(&mut DrawContext<'_>) + Send>;
#[cfg(target_arch = "wasm32")]
type DrawFn = Box<dyn FnMut(&mut DrawContext<'_>)>;

/// A pass written as a closure, for anything that owns GPU state and wants to
/// run inside the chain: a fluid simulation, a particle system, a UI layer.
/// It reads `readBuffer` and writes `writeBuffer` (the screen when last).
///
/// ```no_run
/// # use threers::postprocessing::{DrawPass, EffectComposer, RenderPass};
/// # fn demo(composer: &mut EffectComposer) {
/// composer.add_pass(RenderPass::new());
/// composer.add_pass(DrawPass::new("fluid", move |ctx| {
///     // record into ctx.encoder, writing ctx.target
///     let _ = &ctx;
/// }));
/// # }
/// ```
pub struct DrawPass {
    pub state: PassState,
    name: &'static str,
    draw: std::sync::Mutex<DrawFn>,
}

impl DrawPass {
    pub fn new(
        name: &'static str,
        draw: impl FnMut(&mut DrawContext<'_>) + PassThreading + 'static,
    ) -> Self {
        Self {
            state: PassState::default(),
            name,
            draw: std::sync::Mutex::new(Box::new(draw)),
        }
    }
}

impl std::fmt::Debug for DrawPass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DrawPass").field("name", &self.name).finish()
    }
}

impl Pass for DrawPass {
    fn name(&self) -> &'static str {
        self.name
    }
    fn state(&self) -> PassState {
        self.state
    }
    fn set_state(&mut self, state: PassState) {
        self.state = state;
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn render(&mut self, ctx: &mut PassContext<'_, '_>, write: &Target, read: &Target, _delta: f32, _mask: bool) {
        let (near_far, tan_half_fov, aspect) = match ctx.camera(0) {
            Some(c) => {
                let p = c.projection_matrix().elements;
                let tan = if p[5].abs() > 1e-8 { 1.0 / p[5] } else { 1.0 };
                let aspect = if p[0].abs() > 1e-8 { p[5] / p[0] } else { 1.0 };
                (c.near_far(), tan, aspect)
            }
            None => ((0.1, 100.0), 1.0, 1.0),
        };
        let device = ctx.renderer.device_arc();
        let queue = ctx.renderer.queue_arc();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(self.name),
        });
        let screen_target = if self.state.render_to_screen {
            match ctx.screen {
                crate::postprocessing::Screen::Target(t) => Some((
                    &t.render_target().color_view,
                    Some(&t.render_target().color_texture),
                    t.format(),
                    t.width(),
                    t.height(),
                )),
                crate::postprocessing::Screen::View {
                    view,
                    format,
                    width,
                    height,
                } => Some((*view, None, *format, *width, *height)),
                crate::postprocessing::Screen::None => return,
            }
        } else {
            None
        };
        let w = write.render_target();
        let (target, target_texture, format, width, height) = screen_target.unwrap_or((
            &w.color_view,
            Some(&w.color_texture),
            w.format,
            w.width,
            w.height,
        ));
        let r = read.render_target();
        let mut draw_ctx = DrawContext {
            device: &device,
            queue: &queue,
            depth: Some(&r.depth_texture),
            near_far,
            tan_half_fov,
            aspect,
            encoder: &mut encoder,
            input: &r.color_view,
            input_texture: &r.color_texture,
            target,
            target_texture,
            format,
            width,
            height,
        };
        let mut f = match self.draw.lock() {
            Ok(f) => f,
            Err(poisoned) => poisoned.into_inner(),
        };
        f(&mut draw_ctx);
        drop(f);
        queue.submit(Some(encoder.finish()));
    }
}
