//! [`EffectComposer`](crate::postprocessing::EffectComposer) post-processing
//! on the native Metal backend.
//!
//! Every three.js pass the composer has is built from the same few things: a
//! fullscreen draw of one of the post-fx shader's kinds
//! ([`PostFxOp`]), a scene drawn into a target, a clear, and a handful of
//! uploads. [`MetalPostBackend`] does each of them over Metal textures, so a
//! chain written once runs here unchanged.
//!
//! The post-fx shader is not a second copy: it is the WGSL the wgpu backend
//! runs (`POSTFX_SHADER`), put through naga's MSL writer once per process,
//! with its bindings laid out in Metal's argument tables. The uniform block
//! has the same bytes, the blends are the same equations, and an 8-bit
//! target is an sRGB texture — what the wgpu side gets by writing and
//! reading through sRGB views — so a pass cannot tell the backends apart.
//!
//! [`MetalPostFx`] is what lives between frames (targets, the compiled
//! pipelines, the kernel and noise textures); [`MetalPostFx::backend`] lends
//! it, with a [`MetalRenderer`] for the scene draws, to a composer.

use std::collections::HashMap;
use std::sync::OnceLock;

use super::device::{compile_library, MetalDevice, MetalError};
use super::enums::*;
use super::objc::{alloc_init, msg0, msg1, msg2, msg3, msg9, sel, class, AutoreleasePool, Id, Owned, NIL, YES};
use super::postfx::render_pass_descriptor;
use super::renderer::{MetalRenderer, RenderView};
use super::resources::{cpu_writable_storage, library_function, new_sampler, new_texture_2d, SamplerKey};
use super::target::MetalRenderTarget;
use crate::cameras::Camera;
use crate::postprocessing::{PostBackend, Target};
use crate::renderer::{PostFxBlend, PostFxNoise, PostFxOp, PostFxOutput};
use crate::scene::Scene;

/// `POSTFX_SHADER`'s uniform block, byte for byte.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    params: [f32; 4],
    params2: [f32; 4],
    params3: [f32; 4],
    resolution: [f32; 4],
    ssao0: [f32; 4],
    inv_proj: [f32; 16],
    proj: [f32; 16],
}

/// Where `POSTFX_SHADER`'s group-0 bindings land in Metal's argument tables:
/// `(binding, kind, slot)`, kind 0 a texture, 1 a sampler, 2 a buffer.
const SLOTS: [(u32, u8, u8); 10] = [
    (0, 0, 0), // input_tex
    (1, 1, 0), // input_sampler
    (2, 2, 0), // u
    (3, 0, 1), // depth_tex
    (4, 1, 1), // depth_sampler
    (5, 0, 2), // normal_tex
    (6, 0, 3), // hw_depth_tex
    (7, 2, 1), // ssao_kernel_buf
    (8, 0, 4), // noise_tex
    (9, 1, 2), // noise_sampler
];

/// `POSTFX_SHADER` as MSL, and its two entry points' names there — translated
/// once per process.
fn postfx_msl() -> &'static Result<(String, String, String), String> {
    static MSL: OnceLock<Result<(String, String, String), String>> = OnceLock::new();
    MSL.get_or_init(|| {
        use wgpu::naga::back::msl;
        use wgpu::naga::{self, valid};
        let source = crate::renderer::shader::postfx_shader();
        let module = naga::front::wgsl::parse_str(source).map_err(|e| e.emit_to_string(source))?;
        let info = valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::all())
            .validate(&module)
            .map_err(|e| format!("{e:?}"))?;
        let mut resources = msl::BindingMap::new();
        for (binding, kind, slot) in SLOTS {
            let mut target = msl::BindTarget::default();
            match kind {
                0 => target.texture = Some(slot),
                1 => target.sampler = Some(msl::BindSamplerTarget::Resource(slot)),
                _ => target.buffer = Some(slot),
            }
            resources.insert(naga::ResourceBinding { group: 0, binding }, target);
        }
        let entry = msl::EntryPointResources { resources, ..Default::default() };
        let options = msl::Options {
            lang_version: (2, 1),
            per_entry_point_map: [("vs_main".to_string(), entry.clone()), ("fs_main".to_string(), entry)]
                .into_iter()
                .collect(),
            fake_missing_bindings: false,
            ..Default::default()
        };
        let pipeline = msl::PipelineOptions::default();
        let (text, tinfo) = msl::write_string(&module, &info, &options, &pipeline).map_err(|e| format!("{e:?}"))?;
        let name = |wanted: &str| -> Result<String, String> {
            module
                .entry_points
                .iter()
                .zip(&tinfo.entry_point_names)
                .find(|(ep, _)| ep.name == wanted)
                .and_then(|(_, n)| n.as_ref().ok().cloned())
                .ok_or_else(|| format!("no `{wanted}` in the translated post-fx shader"))
        };
        Ok((text, name("vs_main")?, name("fs_main")?))
    })
}

/// The Metal format a composer target of `format` is made in. An 8-bit
/// target is sRGB, so it reads and writes linear as a half-float one does.
fn metal_format(format: wgpu::TextureFormat) -> NSUInteger {
    match format {
        wgpu::TextureFormat::Rgba16Float => pixel_format::RGBA16_FLOAT,
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
            pixel_format::BGRA8_UNORM_SRGB
        }
        _ => pixel_format::RGBA8_UNORM_SRGB,
    }
}

/// The wgpu format with the same storage as a Metal one, for the clear rule.
fn wgpu_equivalent(format: NSUInteger) -> wgpu::TextureFormat {
    match format {
        pixel_format::RGBA16_FLOAT => wgpu::TextureFormat::Rgba16Float,
        pixel_format::BGRA8_UNORM_SRGB => wgpu::TextureFormat::Bgra8UnormSrgb,
        _ => wgpu::TextureFormat::Rgba8UnormSrgb,
    }
}

/// A registered target: the Metal attachments and the format the composer
/// asked for.
struct Registered {
    rt: MetalRenderTarget,
    format: wgpu::TextureFormat,
}

/// What the composer keeps on Metal between frames.
pub struct MetalPostFx {
    device: MetalDevice,
    targets: HashMap<u32, Registered>,
    library: Option<Owned>,
    pipelines: HashMap<(PostFxBlend, NSUInteger), Owned>,
    kernel: [[f32; 4]; 32],
    ssao_noise: Owned,
    glitch: (Owned, u32),
    /// `default_normal` on wgpu: (0.5, 0.5, 1, 1).
    fallback_color: Owned,
    fallback_depth: Owned,
    linear: Owned,
    nearest: Owned,
    noise_sampler: Owned,
    jitter: Option<(f32, f32)>,
}

// SAFETY: as `MetalRenderer` — Metal objects are thread-safe to use from one
// thread at a time, and `&mut self` makes it one.
unsafe impl Send for MetalPostFx {}

impl MetalPostFx {
    /// Post-processing state on `device` (shared with the renderer that
    /// draws the scenes).
    ///
    /// Scene draws are single-sampled, as the wgpu renderer draws into a
    /// render target and as three.js draws into one without `samples`.
    pub fn new(device: &MetalDevice) -> Result<Self, MetalError> {
        let storage = cpu_writable_storage(device);
        let tex = |w: u32, h: u32, format: NSUInteger| {
            new_texture_2d(device, w, h, format, texture_usage::SHADER_READ, storage, 1, 1)
        };
        let ssao_noise = tex(4, 4, pixel_format::R32_FLOAT)?;
        replace(&ssao_noise, 4, 4, 4, bytemuck::cast_slice(&[0f32; 16]));
        let glitch = tex(64, 64, pixel_format::R32_FLOAT)?;
        replace(&glitch, 64, 64, 4, bytemuck::cast_slice(&[0f32; 64 * 64]));
        let fallback_color = tex(1, 1, pixel_format::RGBA8_UNORM)?;
        replace(&fallback_color, 1, 1, 4, &[128, 128, 255, 255]);
        let fallback_depth = tex(1, 1, pixel_format::R32_FLOAT)?;
        replace(&fallback_depth, 1, 1, 4, bytemuck::cast_slice(&[0f32]));
        let sampler = |linear: bool, repeat: bool| {
            new_sampler(
                device,
                SamplerKey {
                    mag_linear: linear,
                    min_linear: linear,
                    mipmapped: false,
                    mip_linear: false,
                    wrap_s: repeat as u8,
                    wrap_t: repeat as u8,
                },
            )
        };
        Ok(Self {
            device: device.clone(),
            targets: HashMap::new(),
            library: None,
            pipelines: HashMap::new(),
            kernel: [[0.0; 4]; 32],
            ssao_noise,
            glitch: (glitch, 64),
            fallback_color,
            fallback_depth,
            linear: sampler(true, false)?,
            nearest: sampler(false, false)?,
            noise_sampler: sampler(false, true)?,
            jitter: None,
        })
    }

    /// A [`PostBackend`] over this state, drawing scenes with `renderer`
    /// (which must be on the same device).
    pub fn backend<'a>(&'a mut self, renderer: &'a mut MetalRenderer) -> MetalPostBackend<'a> {
        MetalPostBackend { fx: self, renderer }
    }

    /// The Metal target registered under `id`.
    pub fn target(&self, id: u32) -> Option<&MetalRenderTarget> {
        self.targets.get(&id).map(|r| &r.rt)
    }

    /// Read a target back as 8-bit RGBA, top row first.
    pub fn read_rgba(&self, id: u32) -> Result<Vec<u8>, MetalError> {
        let t = self
            .targets
            .get(&id)
            .ok_or_else(|| MetalError::Unsupported(format!("no post-fx target {id}")))?;
        t.rt.read_rgba(&self.device)
    }

    /// Translate and compile the post-fx shader now rather than at the
    /// first draw, to learn whether this device takes it.
    pub fn prepare(&mut self) -> Result<(), MetalError> {
        self.library().map(|_| ())
    }

    /// Compile the translated shader into a library of its own, once.
    fn library(&mut self) -> Result<Id, MetalError> {
        if let Some(lib) = &self.library {
            return Ok(lib.id());
        }
        let (msl, _, _) = postfx_msl().as_ref().map_err(|e| MetalError::ShaderCompilation(e.clone()))?;
        let lib = unsafe { compile_library(self.device.device_id(), msl)? };
        let id = lib.id();
        self.library = Some(lib);
        Ok(id)
    }

    fn pipeline(&mut self, blend: PostFxBlend, format: NSUInteger) -> Result<Id, MetalError> {
        if let Some(p) = self.pipelines.get(&(blend, format)) {
            return Ok(p.id());
        }
        let library = self.library()?;
        let (_, vs_name, fs_name) = postfx_msl().as_ref().map_err(|e| MetalError::ShaderCompilation(e.clone()))?;
        let _pool = AutoreleasePool::new();
        let vs = library_function(library, vs_name)?;
        let fs = library_function(library, fs_name)?;
        let state = unsafe {
            let desc = Owned::from_retained(alloc_init(class!("MTLRenderPipelineDescriptor")))
                .ok_or_else(|| MetalError::Allocation("MTLRenderPipelineDescriptor".into()))?;
            let _: () = msg1(desc.id(), sel!("setVertexFunction:"), vs.id());
            let _: () = msg1(desc.id(), sel!("setFragmentFunction:"), fs.id());
            let attachments: Id = msg0(desc.id(), sel!("colorAttachments"));
            let color: Id = msg1(attachments, sel!("objectAtIndexedSubscript:"), 0usize);
            let _: () = msg1(color, sel!("setPixelFormat:"), format);
            if let Some([(cs, cd), (a_s, a_d)]) = blend_factors(blend) {
                let _: () = msg1(color, sel!("setBlendingEnabled:"), YES);
                let _: () = msg1(color, sel!("setRgbBlendOperation:"), blend_op::ADD);
                let _: () = msg1(color, sel!("setAlphaBlendOperation:"), blend_op::ADD);
                let _: () = msg1(color, sel!("setSourceRGBBlendFactor:"), cs);
                let _: () = msg1(color, sel!("setDestinationRGBBlendFactor:"), cd);
                let _: () = msg1(color, sel!("setSourceAlphaBlendFactor:"), a_s);
                let _: () = msg1(color, sel!("setDestinationAlphaBlendFactor:"), a_d);
            }
            let mut err: Id = NIL;
            let state: Id = msg2(
                self.device.device_id(),
                sel!("newRenderPipelineStateWithDescriptor:error:"),
                desc.id(),
                &mut err as *mut Id,
            );
            Owned::from_retained(state).ok_or_else(|| {
                MetalError::PipelineCreation(super::objc::error_message(err))
            })?
        };
        let id = state.id();
        self.pipelines.insert((blend, format), state);
        Ok(id)
    }

    fn register(&mut self, width: u32, height: u32, format: wgpu::TextureFormat) -> Result<Target, MetalError> {
        let rt = MetalRenderTarget::new(&self.device, width, height, metal_format(format), 1)?;
        let id = Target::next_id();
        let t = Target::external(id, width, height, format);
        self.targets.insert(id, Registered { rt, format });
        Ok(t)
    }

    fn color(&self, id: u32) -> Option<Id> {
        self.targets.get(&id).map(|r| r.rt.color_texture())
    }

    fn depth(&self, id: u32) -> Option<Id> {
        self.targets.get(&id).map(|r| r.rt.resolved_depth_texture())
    }

    fn run(&mut self, op: &PostFxOp, out_id: u32) -> Result<(), MetalError> {
        let Some(out) = self.targets.get(&out_id) else {
            return Ok(());
        };
        let (out_tex, out_format) = (out.rt.color_texture(), out.rt.color_format());
        let (width, height) = out.rt.size();
        let pipeline = self.pipeline(op.blend, out_format)?;
        let display = op.display;
        let fallback = self.fallback_color.id();
        let input = self.color(op.input).unwrap_or(fallback);
        let depth = self.color(op.depth).unwrap_or(fallback);
        let normal = self.color(op.normal).unwrap_or(fallback);
        let hw_depth = self.depth(op.hw_depth).unwrap_or(self.fallback_depth.id());
        let glitchy = op.kind == 5 || op.kind == 31;
        let noise = match op.noise {
            PostFxNoise::Default => None,
            PostFxNoise::Depth(id) => self.depth(id),
            PostFxNoise::Color(id) => self.color(id),
        }
        .unwrap_or(if glitchy { self.glitch.0.id() } else { self.ssao_noise.id() });
        let uniforms = Uniforms {
            params: [op.kind as f32, op.time, op.p0z, if op.hw_depth != 0 { 1.0 } else { 0.0 }],
            params2: op.p2,
            params3: op.p3,
            resolution: [
                width as f32,
                height as f32,
                if display { 0.0 } else { 1.0 },
                // Bit 1 as on wgpu: the display encodes on write. Depth here
                // is never reversed, so bit 16 stays clear.
                u32::from(display && pixel_format::is_srgb(out_format)) as f32,
            ],
            ssao0: [op.camera.near, op.camera.far, op.camera.kernel_radius, op.camera.kernel_size as f32],
            inv_proj: op.camera.inv_proj,
            proj: op.camera.proj,
        };
        let aux = if op.linear_aux || glitchy { self.linear.id() } else { self.nearest.id() };
        let clear = (op.blend == PostFxBlend::Replace).then(|| clear_color(op.clear, out_format, display));
        let _pool = AutoreleasePool::new();
        unsafe {
            let cmd = self.device.command_buffer();
            if cmd.is_null() {
                return Err(MetalError::NoCommandQueue);
            }
            let enc = begin(cmd, out_tex, clear)?;
            let _: () = msg1(enc, sel!("setRenderPipelineState:"), pipeline);
            for (slot, tex) in [input, depth, normal, hw_depth, noise].into_iter().enumerate() {
                let _: () = msg2(enc, sel!("setFragmentTexture:atIndex:"), tex, slot);
            }
            for (slot, s) in [self.linear.id(), aux, self.noise_sampler.id()].into_iter().enumerate() {
                let _: () = msg2(enc, sel!("setFragmentSamplerState:atIndex:"), s, slot);
            }
            let u = bytemuck::bytes_of(&uniforms);
            let _: () = msg3(enc, sel!("setFragmentBytes:length:atIndex:"), u.as_ptr(), u.len(), 0usize);
            let k: &[u8] = bytemuck::cast_slice(&self.kernel);
            let _: () = msg3(enc, sel!("setFragmentBytes:length:atIndex:"), k.as_ptr(), k.len(), 1usize);
            let _: () = msg3(enc, sel!("drawPrimitives:vertexStart:vertexCount:"), primitive::TRIANGLE, 0usize, 3usize);
            let _: () = msg0(enc, sel!("endEncoding"));
            let _: () = msg0(cmd, sel!("commit"));
        }
        Ok(())
    }

    fn clear(&mut self, out_id: u32, color: [f64; 4], display: bool) -> Result<(), MetalError> {
        let Some(out) = self.targets.get(&out_id) else {
            return Ok(());
        };
        let c = clear_color(color, out.rt.color_format(), display);
        let tex = out.rt.color_texture();
        let _pool = AutoreleasePool::new();
        unsafe {
            let cmd = self.device.command_buffer();
            if cmd.is_null() {
                return Err(MetalError::NoCommandQueue);
            }
            let enc = begin(cmd, tex, Some(c))?;
            let _: () = msg0(enc, sel!("endEncoding"));
            let _: () = msg0(cmd, sel!("commit"));
        }
        Ok(())
    }

    fn upload_f16(&mut self, id: u32, texels: &[[f32; 4]]) -> Result<(), MetalError> {
        let Some(t) = self.targets.get(&id) else {
            return Ok(());
        };
        let (w, h) = t.rt.size();
        if t.rt.color_format() != pixel_format::RGBA16_FLOAT || texels.len() < (w * h) as usize {
            return Ok(());
        }
        let mut bytes = Vec::with_capacity(texels.len() * 8);
        for texel in &texels[..(w * h) as usize] {
            for c in texel {
                bytes.extend_from_slice(&crate::renderer::gpu_texture::f32_to_f16_bits(*c).to_le_bytes());
            }
        }
        // The target is private to the GPU; the texels go through a buffer.
        let staging = self.device.new_buffer(&bytes)?;
        let _pool = AutoreleasePool::new();
        unsafe {
            let cmd = self.device.command_buffer();
            if cmd.is_null() {
                return Err(MetalError::NoCommandQueue);
            }
            let blit: Id = msg0(cmd, sel!("blitCommandEncoder"));
            let row = w as NSUInteger * 8;
            let _: () = msg9(
                blit,
                sel!("copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:"),
                staging.id(),
                0usize,
                row,
                row * h as NSUInteger,
                MTLSize { width: w as NSUInteger, height: h as NSUInteger, depth: 1 },
                t.rt.color_texture(),
                0usize,
                0usize,
                MTLOrigin::default(),
            );
            let _: () = msg0(blit, sel!("endEncoding"));
            let _: () = msg0(cmd, sel!("commit"));
        }
        Ok(())
    }

    /// Draw `scene` into target `id`.
    fn draw_scene(
        &mut self,
        renderer: &mut MetalRenderer,
        scene: &mut Scene,
        camera: &dyn Camera,
        id: u32,
    ) -> Result<(), MetalError> {
        let Some(t) = self.targets.get(&id) else {
            return Ok(());
        };
        let (w, h) = t.rt.size();
        let pass = t.rt.attachments();
        let mut view = RenderView::from_camera(camera);
        if let Some((ox, oy)) = self.jitter {
            // As the wgpu renderer jitters: in the z column for a
            // perspective projection, the translation for an orthographic one.
            let (jx, jy) = (ox * 2.0 / w as f32, oy * 2.0 / h as f32);
            let m = &mut view.projection_matrix.elements;
            if m[11].abs() > 0.5 {
                m[8] += jx;
                m[9] += jy;
            } else {
                m[12] -= jx;
                m[13] -= jy;
            }
        }
        // A composer's scene draws are linear, as three.js draws into a
        // render target: tone mapping is the output pass's to do. Depth is
        // kept for the passes that read it.
        let tone = renderer.tone_mapping();
        renderer.set_tone_mapping(crate::ToneMapping::None, tone.1);
        renderer.set_store_depth(true);
        let result = renderer.render_views(scene, &[view], &pass);
        renderer.set_store_depth(false);
        renderer.set_tone_mapping(tone.0, tone.1);
        result.map(|_| ())
    }
}

/// Begin a render pass on `tex`, clearing it to `clear` or loading it.
unsafe fn begin(cmd: Id, tex: Id, clear: Option<MTLClearColor>) -> Result<Id, MetalError> {
    let desc = render_pass_descriptor(tex);
    let attachments: Id = msg0(desc, sel!("colorAttachments"));
    let color: Id = msg1(attachments, sel!("objectAtIndexedSubscript:"), 0usize);
    match clear {
        Some(c) => {
            let _: () = msg1(color, sel!("setClearColor:"), c);
        }
        None => {
            let _: () = msg1(color, sel!("setLoadAction:"), load_action::LOAD);
        }
    }
    let enc: Id = msg1(cmd, sel!("renderCommandEncoderWithDescriptor:"), desc);
    if enc.is_null() {
        return Err(MetalError::Unsupported("post-fx render encoder failed".into()));
    }
    Ok(enc)
}

/// `stored_clear`, for a Metal format.
fn clear_color(c: [f64; 4], format: NSUInteger, display: bool) -> MTLClearColor {
    let c = crate::renderer::stored_clear(c, wgpu_equivalent(format), display);
    MTLClearColor { red: c.r, green: c.g, blue: c.b, alpha: c.a }
}

/// `[(colour src, dst), (alpha src, dst)]`, as `PostFxBlend::state` has them.
fn blend_factors(blend: PostFxBlend) -> Option<[(NSUInteger, NSUInteger); 2]> {
    use blend_factor::*;
    Some(match blend {
        PostFxBlend::Replace | PostFxBlend::Overwrite => return None,
        PostFxBlend::Add => [(ONE, ONE), (ONE, ONE)],
        PostFxBlend::AddAlpha => [(SOURCE_ALPHA, ONE), (SOURCE_ALPHA, ONE)],
        PostFxBlend::Multiply => [(DESTINATION_COLOR, ZERO), (DESTINATION_ALPHA, ZERO)],
        PostFxBlend::Normal => [(SOURCE_ALPHA, ONE_MINUS_SOURCE_ALPHA), (ONE, ONE_MINUS_SOURCE_ALPHA)],
        PostFxBlend::NormalPremultiplied => {
            [(ONE, ONE_MINUS_SOURCE_ALPHA), (ONE, ONE_MINUS_SOURCE_ALPHA)]
        }
    })
}

/// Overwrite a CPU-writable texture.
fn replace(tex: &Owned, width: u32, height: u32, bpp: usize, bytes: &[u8]) {
    let region = MTLRegion {
        origin: MTLOrigin::default(),
        size: MTLSize { width: width as NSUInteger, height: height as NSUInteger, depth: 1 },
    };
    unsafe {
        let _: () = super::objc::msg4(
            tex.id(),
            sel!("replaceRegion:mipmapLevel:withBytes:bytesPerRow:"),
            region,
            0usize,
            bytes.as_ptr() as *const std::ffi::c_void,
            (width as usize * bpp) as NSUInteger,
        );
    }
}

/// An [`EffectComposer`](crate::postprocessing::EffectComposer)'s renderer on
/// Metal: [`MetalPostFx`]'s state and a [`MetalRenderer`] for the scenes.
pub struct MetalPostBackend<'a> {
    fx: &'a mut MetalPostFx,
    renderer: &'a mut MetalRenderer,
}

impl MetalPostBackend<'_> {
    /// Report a failure once per kind; a pass that fails draws nothing.
    fn report(what: &str, e: MetalError) {
        static SAID: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
        let mut said = SAID.lock().unwrap_or_else(|e| e.into_inner());
        if !said.iter().any(|s| s == what) {
            said.push(what.to_string());
            eprintln!("threers: Metal post-processing: {what} failed: {e}");
        }
    }
}

impl PostBackend for MetalPostBackend<'_> {
    fn new_target(&mut self, width: u32, height: u32, format: wgpu::TextureFormat) -> Target {
        match self.fx.register(width, height, format) {
            Ok(t) => t,
            Err(e) => {
                Self::report("allocating a target", e);
                Target::external(Target::next_id(), width, height, format)
            }
        }
    }

    fn registered_target(&self, id: u32) -> Option<Target> {
        let r = self.fx.targets.get(&id)?;
        let (w, h) = r.rt.size();
        Some(Target::external(id, w, h, r.format))
    }

    fn unregister_render_target(&mut self, id: u32) {
        self.fx.targets.remove(&id);
    }

    fn upload_render_target_f16(&mut self, id: u32, texels: &[[f32; 4]]) {
        if let Err(e) = self.fx.upload_f16(id, texels) {
            Self::report("an upload", e);
        }
    }

    fn set_postfx_kernel(&mut self, data: &[[f32; 4]]) {
        self.fx.kernel = [[0.0; 4]; 32];
        for (dst, src) in self.fx.kernel.iter_mut().zip(data) {
            *dst = *src;
        }
    }

    fn set_ssao_noise(&mut self, noise: &[f32]) {
        let mut texels = [0f32; 16];
        for (dst, src) in texels.iter_mut().zip(noise) {
            *dst = *src;
        }
        replace(&self.fx.ssao_noise, 4, 4, 4, bytemuck::cast_slice(&texels));
    }

    fn set_glitch_disp(&mut self, data: &[f32], size: u32) {
        let sz = size.max(1);
        if data.len() < (sz * sz) as usize {
            return;
        }
        if self.fx.glitch.1 != sz {
            let storage = cpu_writable_storage(&self.fx.device);
            match new_texture_2d(&self.fx.device, sz, sz, pixel_format::R32_FLOAT, texture_usage::SHADER_READ, storage, 1, 1) {
                Ok(t) => self.fx.glitch = (t, sz),
                Err(e) => return Self::report("the glitch map", e),
            }
        }
        replace(&self.fx.glitch.0, sz, sz, 4, bytemuck::cast_slice(&data[..(sz * sz) as usize]));
    }

    fn run_postfx(&mut self, op: &PostFxOp, output: PostFxOutput<'_>) {
        // A wgpu view is no screen of Metal's; the composer is handed a
        // target for one instead.
        let PostFxOutput::Target(id) = output else {
            return;
        };
        if let Err(e) = self.fx.run(op, id) {
            Self::report("a post-fx draw", e);
        }
    }

    fn clear_postfx_output(&mut self, output: PostFxOutput<'_>, color: [f64; 4], display: bool) {
        let PostFxOutput::Target(id) = output else {
            return;
        };
        if let Err(e) = self.fx.clear(id, color, display) {
            Self::report("a clear", e);
        }
    }

    fn auto_clear(&self) -> bool {
        self.renderer.auto_clear()
    }

    fn set_auto_clear(&mut self, enabled: bool) {
        self.renderer.set_auto_clear(enabled);
    }

    fn set_jitter_override(&mut self, offset: Option<(f32, f32)>) {
        self.fx.jitter = offset;
    }

    fn render_to_target(&mut self, scene: &mut Scene, camera: &dyn Camera, target: &Target) {
        if let Err(e) = self.fx.draw_scene(self.renderer, scene, camera, target.id()) {
            Self::report("a scene draw", e);
        }
    }

    fn render_to_view(&mut self, _scene: &mut Scene, _camera: &dyn Camera, _view: &wgpu::TextureView) {}
}

#[cfg(test)]
mod tests {
    #[test]
    fn postfx_shader_translates_to_msl() {
        let (msl, vs, fs) = super::postfx_msl().as_ref().expect("POSTFX_SHADER translates");
        assert!(msl.contains(vs.as_str()) && msl.contains(fs.as_str()));
    }
}
