//! Post-processing bindings: `WebEffectComposer` and `WebPass` over
//! [`crate::postprocessing`], which the shim's `EffectComposer` and pass
//! classes wrap.
//!
//! A pass is shared between its JavaScript object and the composer it is
//! added to (`Rc<RefCell<..>>`), so a property the page sets on the pass
//! reaches the one the composer runs. Properties and uniforms are reached by
//! name ([`WebPass::get`] / [`WebPass::set`]), from the schema the pass
//! describes ([`WebPass::param_schema`]).
#![allow(unused_imports)]

use super::*;
use crate::postprocessing as pp;
use pp::{ParamKind, ParamValue, Pass};
use std::cell::RefCell;
use std::rc::Rc;

type Shared = Rc<RefCell<Box<dyn Pass>>>;

/// A shared pass, as the composer holds it.
struct SharedPass(Shared);

impl std::fmt::Debug for SharedPass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.borrow().fmt(f)
    }
}

impl Pass for SharedPass {
    fn name(&self) -> &'static str {
        self.0.borrow().name()
    }
    fn state(&self) -> pp::PassState {
        self.0.borrow().state()
    }
    fn set_state(&mut self, state: pp::PassState) {
        self.0.borrow_mut().set_state(state)
    }
    fn set_size(&mut self, width: u32, height: u32) {
        self.0.borrow_mut().set_size(width, height)
    }
    fn render(
        &mut self,
        ctx: &mut pp::PassContext<'_, '_>,
        write: &pp::Target,
        read: &pp::Target,
        delta: f32,
        mask_active: bool,
    ) {
        self.0
            .borrow_mut()
            .render(ctx, write, read, delta, mask_active)
    }
    fn params(&self) -> Vec<pp::ParamInfo> {
        self.0.borrow().params()
    }
    fn get(&self, name: &str) -> Option<ParamValue> {
        self.0.borrow().get(name)
    }
    fn set(&mut self, name: &str, value: ParamValue) -> bool {
        self.0.borrow_mut().set(name, value)
    }
    fn scene_slots(&self) -> Vec<(usize, usize)> {
        self.0.borrow().scene_slots()
    }
    fn set_scene_slot(&mut self, n: usize, scene: usize, camera: usize) {
        self.0.borrow_mut().set_scene_slot(n, scene, camera)
    }
    fn mask_role(&self) -> pp::MaskRole {
        self.0.borrow().mask_role()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

fn kind_name(k: ParamKind) -> &'static str {
    match k {
        ParamKind::Float => "float",
        ParamKind::Int => "int",
        ParamKind::Bool => "bool",
        ParamKind::Vec2 => "vec2",
        ParamKind::Vec3 => "vec3",
        ParamKind::Vec4 => "vec4",
        ParamKind::Color => "color",
        ParamKind::Floats => "floats",
        ParamKind::Texture => "texture",
        ParamKind::CubeTexture => "cubeTexture",
        ParamKind::Objects => "objects",
    }
}

fn to_js(v: ParamValue) -> JsValue {
    let arr = |s: &[f32]| {
        let a = js_sys::Array::new();
        for x in s {
            a.push(&JsValue::from_f64(*x as f64));
        }
        JsValue::from(a)
    };
    match v {
        ParamValue::Float(x) => JsValue::from_f64(x as f64),
        ParamValue::Int(x) => JsValue::from_f64(x as f64),
        ParamValue::Bool(b) => JsValue::from_bool(b),
        ParamValue::Vec2(v) => arr(&v),
        ParamValue::Vec3(v) | ParamValue::Color(v) => arr(&v),
        ParamValue::Vec4(v) => arr(&v),
        ParamValue::Floats(v) => arr(&v),
        ParamValue::Objects(ids) => {
            let a = js_sys::Array::new();
            for id in ids {
                a.push(&JsValue::from_f64(object_id_to_f64(id)));
            }
            JsValue::from(a)
        }
        ParamValue::Texture(_) | ParamValue::CubeTexture(_) => JsValue::NULL,
    }
}

fn from_js(v: &JsValue) -> Option<ParamValue> {
    if let Some(b) = v.as_bool() {
        return Some(ParamValue::Bool(b));
    }
    if let Some(x) = v.as_f64() {
        return Some(ParamValue::Float(x as f32));
    }
    if js_sys::Array::is_array(v)
        || v.is_instance_of::<js_sys::Float32Array>()
        || v.is_instance_of::<js_sys::Float64Array>()
    {
        let a = js_sys::Array::from(v);
        let out: Vec<f32> = a.iter().map(|e| e.as_f64().unwrap_or(0.0) as f32).collect();
        return Some(ParamValue::Floats(out));
    }
    None
}

fn camera_dyn(c: &WebCamera) -> &dyn crate::cameras::Camera {
    match &c.inner {
        CameraInner::Perspective(p) => p,
        CameraInner::Orthographic(o) => o,
    }
}

/// One post-processing pass — any of three.js's, made by the static
/// constructors below.
#[wasm_bindgen]
pub struct WebPass {
    inner: Shared,
}

impl WebPass {
    fn of(p: impl Pass + 'static) -> WebPass {
        WebPass {
            inner: Rc::new(RefCell::new(Box::new(p))),
        }
    }

    fn with<R>(&self, f: impl FnOnce(&mut dyn Pass) -> R) -> R {
        f(&mut **self.inner.borrow_mut())
    }

    fn downcast<T: 'static, R>(&self, f: impl FnOnce(&mut T) -> R) -> Option<R> {
        let mut b = self.inner.borrow_mut();
        b.as_any_mut().downcast_mut::<T>().map(f)
    }
}

#[wasm_bindgen]
impl WebPass {
    #[wasm_bindgen(js_name = renderPass)]
    pub fn render_pass() -> WebPass {
        Self::of(pp::RenderPass::new())
    }

    /// `ShaderPass(shader)` for a shader by its three.js `name`; one it does
    /// not have a port of copies, and reports so through `isKnownShader`.
    #[wasm_bindgen(js_name = shaderPass)]
    pub fn shader_pass(name: &str) -> WebPass {
        Self::of(pp::ShaderPass::new(
            pp::ShaderKind::from_name(name).unwrap_or(pp::ShaderKind::Copy),
        ))
    }

    #[wasm_bindgen(js_name = isKnownShader)]
    pub fn is_known_shader(name: &str) -> bool {
        pp::ShaderKind::from_name(name).is_some()
    }

    #[wasm_bindgen(js_name = texturePass)]
    pub fn texture_pass(opacity: f32) -> WebPass {
        Self::of(pp::TexturePass::new(None, opacity))
    }

    #[wasm_bindgen(js_name = cubeTexturePass)]
    pub fn cube_texture_pass(opacity: f32) -> WebPass {
        Self::of(pp::CubeTexturePass::new(0, None, opacity))
    }

    #[wasm_bindgen(js_name = clearPass)]
    pub fn clear_pass(r: f32, g: f32, b: f32, has_color: bool, alpha: f32) -> WebPass {
        let c =
            (has_color && (r != 0.0 || g != 0.0 || b != 0.0)).then(|| crate::Color::new(r, g, b));
        Self::of(pp::ClearPass::new(c, alpha))
    }

    #[wasm_bindgen(js_name = maskPass)]
    pub fn mask_pass() -> WebPass {
        Self::of(pp::MaskPass::new(0, 0))
    }

    #[wasm_bindgen(js_name = clearMaskPass)]
    pub fn clear_mask_pass() -> WebPass {
        Self::of(pp::ClearMaskPass::new())
    }

    #[wasm_bindgen(js_name = outputPass)]
    pub fn output_pass() -> WebPass {
        Self::of(pp::OutputPass::new())
    }

    #[wasm_bindgen(js_name = unrealBloomPass)]
    pub fn unreal_bloom_pass(
        width: u32,
        height: u32,
        strength: f32,
        radius: f32,
        threshold: f32,
    ) -> WebPass {
        Self::of(pp::UnrealBloomPass::new(
            width, height, strength, radius, threshold,
        ))
    }

    #[wasm_bindgen(js_name = bloomPass)]
    pub fn bloom_pass(strength: f32, kernel_size: u32, sigma: f32) -> WebPass {
        Self::of(pp::BloomPass::new(strength, kernel_size, sigma))
    }

    #[wasm_bindgen(js_name = filmPass)]
    pub fn film_pass(intensity: f32, grayscale: bool) -> WebPass {
        Self::of(pp::FilmPass::new(intensity, grayscale))
    }

    #[wasm_bindgen(js_name = dotScreenPass)]
    pub fn dot_screen_pass(cx: f32, cy: f32, angle: f32, scale: f32) -> WebPass {
        Self::of(pp::DotScreenPass::new(
            Some([cx, cy]),
            Some(angle),
            Some(scale),
        ))
    }

    #[wasm_bindgen(js_name = halftonePass)]
    pub fn halftone_pass(width: u32, height: u32) -> WebPass {
        Self::of(pp::HalftonePass::new(width, height, &[]))
    }

    /// `GlitchPass(dt_size)` — draws its heightmap from the shared random
    /// stream (`random`), as three.js's does from `Math.random`.
    #[wasm_bindgen(js_name = glitchPass)]
    pub fn glitch_pass(dt_size: u32) -> WebPass {
        Self::of(pp::GlitchPass::new(dt_size))
    }

    #[wasm_bindgen(js_name = afterimagePass)]
    pub fn afterimage_pass(damp: f32) -> WebPass {
        Self::of(pp::AfterimagePass::new(damp))
    }

    #[wasm_bindgen(js_name = lutPass)]
    pub fn lut_pass(intensity: f32) -> WebPass {
        Self::of(pp::LutPass::new(intensity))
    }

    #[wasm_bindgen(js_name = smaaPass)]
    pub fn smaa_pass(width: u32, height: u32) -> WebPass {
        Self::of(pp::SmaaPass::new(width, height))
    }

    #[wasm_bindgen(js_name = bokehPass)]
    pub fn bokeh_pass(focus: f32, aperture: f32, maxblur: f32) -> WebPass {
        Self::of(pp::BokehPass::new(0, 0, focus, aperture, maxblur))
    }

    /// `SSAOPass` — its kernel and noise come from the shared random stream.
    #[wasm_bindgen(js_name = ssaoPass)]
    pub fn ssao_pass(width: u32, height: u32, kernel_size: u32) -> WebPass {
        Self::of(pp::SsaoPass::new(0, 0, width, height, kernel_size))
    }

    #[wasm_bindgen(js_name = saoPass)]
    pub fn sao_pass(width: u32, height: u32) -> WebPass {
        Self::of(pp::SaoPass::new(0, 0, width, height))
    }

    #[wasm_bindgen(js_name = ssrPass)]
    pub fn ssr_pass(width: u32, height: u32) -> WebPass {
        Self::of(pp::SsrPass::new(0, 0, width, height))
    }

    #[wasm_bindgen(js_name = outlinePass)]
    pub fn outline_pass(width: u32, height: u32) -> WebPass {
        Self::of(pp::OutlinePass::new(width, height, 0, 0, Vec::new()))
    }

    #[wasm_bindgen(js_name = taaRenderPass)]
    pub fn taa_render_pass() -> WebPass {
        Self::of(pp::TaaRenderPass::new(0, 0))
    }

    #[wasm_bindgen(js_name = ssaaRenderPass)]
    pub fn ssaa_render_pass() -> WebPass {
        Self::of(pp::SsaaRenderPass::new(0, 0))
    }

    #[wasm_bindgen(js_name = renderPixelatedPass)]
    pub fn render_pixelated_pass(pixel_size: u32) -> WebPass {
        Self::of(pp::RenderPixelatedPass::new(pixel_size, 0, 0))
    }

    #[wasm_bindgen(js_name = renderTransitionPass)]
    pub fn render_transition_pass() -> WebPass {
        Self::of(pp::RenderTransitionPass::new(0, 0, 1, 1))
    }

    #[wasm_bindgen(js_name = adaptiveToneMappingPass)]
    pub fn adaptive_tone_mapping_pass(adaptive: bool, resolution: u32) -> WebPass {
        Self::of(pp::AdaptiveToneMappingPass::new(adaptive, resolution))
    }

    /// The three.js class name.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.inner.borrow().name().to_string()
    }

    #[wasm_bindgen(getter)]
    pub fn enabled(&self) -> bool {
        self.inner.borrow().state().enabled
    }
    #[wasm_bindgen(setter)]
    pub fn set_enabled(&self, v: bool) {
        self.with(|p| {
            p.set_state(pp::PassState {
                enabled: v,
                ..p.state()
            })
        });
    }
    #[wasm_bindgen(getter, js_name = needsSwap)]
    pub fn needs_swap(&self) -> bool {
        self.inner.borrow().state().needs_swap
    }
    #[wasm_bindgen(setter, js_name = needsSwap)]
    pub fn set_needs_swap(&self, v: bool) {
        self.with(|p| {
            p.set_state(pp::PassState {
                needs_swap: v,
                ..p.state()
            })
        });
    }
    #[wasm_bindgen(getter)]
    pub fn clear(&self) -> bool {
        self.inner.borrow().state().clear
    }
    #[wasm_bindgen(setter)]
    pub fn set_clear(&self, v: bool) {
        self.with(|p| {
            p.set_state(pp::PassState {
                clear: v,
                ..p.state()
            })
        });
    }
    #[wasm_bindgen(getter, js_name = renderToScreen)]
    pub fn render_to_screen(&self) -> bool {
        self.inner.borrow().state().render_to_screen
    }
    #[wasm_bindgen(setter, js_name = renderToScreen)]
    pub fn set_render_to_screen(&self, v: bool) {
        self.with(|p| {
            p.set_state(pp::PassState {
                render_to_screen: v,
                ..p.state()
            })
        });
    }

    #[wasm_bindgen(js_name = setSize)]
    pub fn set_size(&self, width: u32, height: u32) {
        self.with(|p| p.set_size(width, height));
    }

    /// `[{ name, kind, uniform }]` for every property and uniform.
    #[wasm_bindgen(js_name = paramSchema)]
    pub fn param_schema(&self) -> js_sys::Array {
        let out = js_sys::Array::new();
        for info in self.inner.borrow().params() {
            let o = js_sys::Object::new();
            let _ = js_sys::Reflect::set(&o, &"name".into(), &info.name.into());
            let _ = js_sys::Reflect::set(&o, &"kind".into(), &kind_name(info.kind).into());
            let _ = js_sys::Reflect::set(&o, &"uniform".into(), &JsValue::from_bool(info.uniform));
            out.push(&o);
        }
        out
    }

    /// A property or uniform: a number, a boolean, an array of numbers (a
    /// vector, colour or list, or object ids), or `undefined`.
    pub fn get(&self, name: &str) -> JsValue {
        match self.inner.borrow().get(name) {
            Some(v) => to_js(v),
            None => JsValue::UNDEFINED,
        }
    }

    /// Assign a property or uniform from a number, boolean or array; `false`
    /// if the pass has no such name or the value is not its shape.
    pub fn set(&self, name: &str, value: JsValue) -> bool {
        match from_js(&value) {
            Some(v) => self.with(|p| p.set(name, v)),
            None => false,
        }
    }

    /// A texture-valued property (`map`, `patternTexture`, `tMixTexture`).
    #[wasm_bindgen(js_name = setTexture)]
    pub fn set_texture(&self, name: &str, texture: &WebTexture) -> bool {
        let t = Some(texture.inner.clone());
        self.with(|p| p.set(name, ParamValue::Texture(t)))
    }

    /// Unset a texture-valued property.
    #[wasm_bindgen(js_name = clearTexture)]
    pub fn clear_texture(&self, name: &str) -> bool {
        self.with(|p| {
            p.set(name, ParamValue::Texture(None)) || p.set(name, ParamValue::CubeTexture(None))
        })
    }

    #[wasm_bindgen(js_name = setCubeTexture)]
    pub fn set_cube_texture(&self, name: &str, texture: &WebCubeTexture) -> bool {
        let t = Some(texture.inner.clone());
        self.with(|p| p.set(name, ParamValue::CubeTexture(t)))
    }

    /// An object-list property (`selectedObjects`, `selects`), by the ids
    /// `WebObjectHandle.id` reports.
    #[wasm_bindgen(js_name = setObjects)]
    pub fn set_objects(&self, name: &str, ids: Vec<f64>) -> bool {
        let ids = ids.into_iter().filter_map(object_id_from_f64).collect();
        self.with(|p| p.set(name, ParamValue::Objects(ids)))
    }

    /// RenderPass's / BokehPass's ... `overrideMaterial`.
    #[wasm_bindgen(js_name = setOverrideMaterial)]
    pub fn set_override_material(&self, material: &WebMaterial) -> bool {
        let m = Some(material.inner.clone());
        self.downcast::<pp::RenderPass, _>(|p| p.override_material = m)
            .is_some()
    }

    #[wasm_bindgen(js_name = clearOverrideMaterial)]
    pub fn clear_override_material(&self) -> bool {
        self.downcast::<pp::RenderPass, _>(|p| p.override_material = None)
            .is_some()
    }

    /// The `(scene, camera)` references the pass draws with, in order.
    #[wasm_bindgen(js_name = sceneSlotCount)]
    pub fn scene_slot_count(&self) -> u32 {
        self.inner.borrow().scene_slots().len() as u32
    }

    /// Point reference `n` at the composer's scene `scene` and camera
    /// `camera` for the next render.
    #[wasm_bindgen(js_name = setSceneSlot)]
    pub fn set_scene_slot(&self, n: u32, scene: u32, camera: u32) {
        self.with(|p| p.set_scene_slot(n as usize, scene as usize, camera as usize));
    }

    /// LUTPass's LUT: `size³` RGBA texels in three.js's `Data3DTexture`
    /// order, or none.
    #[wasm_bindgen(js_name = setLut)]
    pub fn set_lut(&self, size: u32, rgba: Option<Vec<f32>>) -> bool {
        let texels = rgba.map(|d| {
            d.chunks_exact(4)
                .map(|c| [c[0], c[1], c[2], c[3]])
                .collect()
        });
        self.downcast::<pp::LutPass, _>(|p| p.set_lut(size, texels))
            .is_some()
    }

    /// GlitchPass's displacement heightmap.
    #[wasm_bindgen(js_name = glitchHeightMap)]
    pub fn glitch_height_map(&self) -> Vec<f32> {
        self.downcast::<pp::GlitchPass, _>(|p| p.height_map().to_vec())
            .unwrap_or_default()
    }

    /// GlitchPass's per-frame uniform update, without drawing — what its
    /// `render` does first.
    #[wasm_bindgen(js_name = glitchStep)]
    pub fn glitch_step(&self) {
        self.downcast::<pp::GlitchPass, _>(|p| p.step());
    }

    /// SSAOPass's sample kernel, `xyz` per sample.
    #[wasm_bindgen(js_name = ssaoKernel)]
    pub fn ssao_kernel(&self) -> Vec<f32> {
        self.downcast::<pp::SsaoPass, _>(|p| p.kernel().iter().flatten().copied().collect())
            .unwrap_or_default()
    }

    /// SSAOPass's 4×4 rotation noise.
    #[wasm_bindgen(js_name = ssaoNoise)]
    pub fn ssao_noise(&self) -> Vec<f32> {
        self.downcast::<pp::SsaoPass, _>(|p| p.noise().to_vec())
            .unwrap_or_default()
    }

    /// SSAOPass's `generateSampleKernel` / `generateRandomKernelRotations`.
    #[wasm_bindgen(js_name = ssaoRegenerate)]
    pub fn ssao_regenerate(&self, kernel_size: u32) {
        self.downcast::<pp::SsaoPass, _>(|p| {
            p.generate_sample_kernel(kernel_size);
            p.generate_random_kernel_rotations();
        });
    }

    /// UnrealBloomPass's mip sizes, `[w, h]` per mip.
    #[wasm_bindgen(js_name = bloomMipSizes)]
    pub fn bloom_mip_sizes(&self) -> Vec<u32> {
        self.downcast::<pp::UnrealBloomPass, _>(|p| {
            p.mip_sizes().iter().flat_map(|(w, h)| [*w, *h]).collect()
        })
        .unwrap_or_default()
    }

    /// AdaptiveToneMappingPass's `reset`.
    pub fn reset(&self) {
        self.downcast::<pp::AdaptiveToneMappingPass, _>(|p| p.reset());
    }
}

/// three.js's `EffectComposer`, over [`crate::postprocessing::EffectComposer`].
#[wasm_bindgen]
pub struct WebEffectComposer {
    inner: pp::EffectComposer,
    passes: Vec<Shared>,
    user_target: Option<(u32, Arc<crate::renderer::RenderTarget>)>,
}

/// Seed the random stream the passes draw from (`GlitchPass`, `SSAOPass`):
/// the parity LCG, as the scenes' `seedRandom` installs for `Math.random`.
#[wasm_bindgen(js_name = seedRandom)]
pub fn seed_random(seed: u32) {
    pp::random::seed_random(seed);
}

#[wasm_bindgen(js_name = unseedRandom)]
pub fn unseed_random() {
    pp::random::unseed_random();
}

/// The next number from that stream — what the shim's seeded `Math.random`
/// returns, so JavaScript and the passes share one sequence.
#[wasm_bindgen(js_name = random)]
pub fn random() -> f64 {
    pp::random::random()
}

#[wasm_bindgen]
impl WebEffectComposer {
    /// A composer for a `width × height` drawing buffer at `pixel_ratio`.
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32, pixel_ratio: f32) -> WebEffectComposer {
        let mut inner = pp::EffectComposer::with_size(width, height);
        inner.set_pixel_ratio(pixel_ratio);
        WebEffectComposer {
            inner,
            passes: Vec::new(),
            user_target: None,
        }
    }

    /// `new EffectComposer(renderer, renderTarget)`: chain through `target`.
    #[wasm_bindgen(js_name = withTarget)]
    pub fn with_target(target: &WebRenderTarget) -> WebEffectComposer {
        let t = pp::Target::from_registered(target.id, target.inner.clone());
        WebEffectComposer {
            inner: pp::EffectComposer::with_target(t),
            passes: Vec::new(),
            user_target: Some((target.id, target.inner.clone())),
        }
    }

    #[wasm_bindgen(js_name = addPass)]
    pub fn add_pass(&mut self, pass: &WebPass) {
        self.passes.push(pass.inner.clone());
        self.inner
            .add_boxed(Box::new(SharedPass(pass.inner.clone())));
    }

    #[wasm_bindgen(js_name = insertPass)]
    pub fn insert_pass(&mut self, pass: &WebPass, index: u32) {
        let i = (index as usize).min(self.passes.len());
        self.passes.insert(i, pass.inner.clone());
        self.inner
            .insert_boxed(Box::new(SharedPass(pass.inner.clone())), i);
    }

    #[wasm_bindgen(js_name = removePass)]
    pub fn remove_pass(&mut self, pass: &WebPass) {
        if let Some(i) = self.passes.iter().position(|p| Rc::ptr_eq(p, &pass.inner)) {
            self.passes.remove(i);
            self.inner.remove_pass(i);
        }
    }

    #[wasm_bindgen(js_name = passNames)]
    pub fn pass_names(&self) -> Vec<String> {
        self.inner
            .pass_names()
            .into_iter()
            .map(String::from)
            .collect()
    }

    #[wasm_bindgen(js_name = setSize)]
    pub fn set_size(&mut self, width: u32, height: u32) {
        self.inner.set_size(width, height);
    }

    #[wasm_bindgen(js_name = setPixelRatio)]
    pub fn set_pixel_ratio(&mut self, ratio: f32) {
        self.inner.set_pixel_ratio(ratio);
    }

    /// The buffers' size in device pixels, `[w, h]`.
    #[wasm_bindgen(js_name = drawingSize)]
    pub fn drawing_size(&self) -> Vec<u32> {
        let (w, h) = self.inner.drawing_size();
        vec![w, h]
    }

    #[wasm_bindgen(getter, js_name = renderToScreen)]
    pub fn render_to_screen(&self) -> bool {
        self.inner.render_to_screen
    }
    #[wasm_bindgen(setter, js_name = renderToScreen)]
    pub fn set_render_to_screen(&mut self, v: bool) {
        self.inner.render_to_screen = v;
    }

    #[wasm_bindgen(js_name = swapBuffers)]
    pub fn swap_buffers(&mut self) {
        self.inner.swap_buffers();
    }

    /// three.js `reset()`: fresh buffers.
    pub fn reset(&mut self) {
        self.user_target = None;
        self.inner.reset(None);
    }

    /// `readBuffer` as a render target the page can sample or read back, once
    /// the composer has rendered (or `undefined`).
    #[wasm_bindgen(js_name = readBuffer)]
    pub fn read_buffer(&self) -> Option<WebRenderTarget> {
        self.inner.read_buffer().map(expose)
    }

    #[wasm_bindgen(js_name = writeBuffer)]
    pub fn write_buffer(&self) -> Option<WebRenderTarget> {
        self.inner.write_buffer().map(expose)
    }

    /// `render(delta)` for a chain that draws no scene.
    #[wasm_bindgen(js_name = render0)]
    pub fn render0(&mut self, renderer: &mut WebRenderer, delta: Option<f64>) {
        self.run(renderer, &mut [], &[], delta);
    }

    /// `render(delta)` with one scene and up to four cameras (repeat one to
    /// fill the slots).
    #[wasm_bindgen(js_name = render1)]
    #[allow(clippy::too_many_arguments)]
    pub fn render1(
        &mut self,
        renderer: &mut WebRenderer,
        scene: &mut WebScene,
        c0: &WebCamera,
        c1: &WebCamera,
        c2: &WebCamera,
        c3: &WebCamera,
        delta: Option<f64>,
    ) {
        let cams = [
            camera_dyn(c0),
            camera_dyn(c1),
            camera_dyn(c2),
            camera_dyn(c3),
        ];
        self.run(renderer, &mut [&mut scene.inner], &cams, delta);
    }

    /// `render(delta)` with two distinct scenes.
    #[wasm_bindgen(js_name = render2)]
    #[allow(clippy::too_many_arguments)]
    pub fn render2(
        &mut self,
        renderer: &mut WebRenderer,
        scene0: &mut WebScene,
        scene1: &mut WebScene,
        c0: &WebCamera,
        c1: &WebCamera,
        c2: &WebCamera,
        c3: &WebCamera,
        delta: Option<f64>,
    ) {
        let cams = [
            camera_dyn(c0),
            camera_dyn(c1),
            camera_dyn(c2),
            camera_dyn(c3),
        ];
        self.run(
            renderer,
            &mut [&mut scene0.inner, &mut scene1.inner],
            &cams,
            delta,
        );
    }
}

/// A composer target as a `WebRenderTarget`, registered for the page's
/// read-backs and `renderTargetTexture`.
fn expose(t: &pp::Target) -> WebRenderTarget {
    let rt = t.render_target().clone();
    ACTIVE_TARGETS.with(|m| m.borrow_mut().insert(t.id(), rt.clone()));
    WebRenderTarget {
        width: rt.width,
        height: rt.height,
        inner: rt,
        id: t.id(),
    }
}

impl WebEffectComposer {
    fn run(
        &mut self,
        renderer: &mut WebRenderer,
        scenes: &mut [&mut crate::Scene],
        cameras: &[&dyn crate::cameras::Camera],
        delta: Option<f64>,
    ) {
        // What `WebRenderer::render` does before drawing a scene: a cube
        // render target used as an environment has to be known by id.
        for scene in scenes.iter() {
            if let Some(id) = scene.environment_cube_rt {
                if let Some(t) = ACTIVE_CUBE_TARGETS.with(|m| m.borrow().get(&id).cloned()) {
                    renderer.renderer.register_cube_render_target(id, &t);
                }
            }
        }
        if let Some((id, rt)) = &self.user_target {
            renderer.renderer.register_render_target(*id, rt);
        }
        let delta = delta.map(|d| d as f32);
        if !self.inner.render_to_screen {
            self.inner.render_frame(
                &mut renderer.renderer,
                scenes,
                cameras,
                &pp::Screen::None,
                delta,
            );
            return;
        }
        // On WebGL2 the canvas is drawn through the persistent back buffer —
        // each acquire of the GL surface is a fresh texture, so a pass that
        // draws onto the screen twice would lose the first.
        if renderer.is_gl {
            renderer.multi_view = true;
            if let Some((id, back)) = renderer.gl_back_buffer() {
                let screen = pp::Screen::Target(pp::Target::from_registered(id, back));
                self.inner
                    .render_frame(&mut renderer.renderer, scenes, cameras, &screen, delta);
                renderer.present_gl_back_buffer(id);
                return;
            }
        }
        let frame = match renderer.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            _ => {
                renderer
                    .surface
                    .configure(&renderer.device, &renderer.config);
                match renderer.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(f)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
                    _ => return,
                }
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let screen = pp::Screen::View {
            view: &view,
            format: renderer.config.format,
            width: renderer.width,
            height: renderer.height,
        };
        self.inner
            .render_frame(&mut renderer.renderer, scenes, cameras, &screen, delta);
        renderer.renderer.queue_arc().present(frame);
    }
}
