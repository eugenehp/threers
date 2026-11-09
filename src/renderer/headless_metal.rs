//! `HeadlessRenderer` drawing through the native Metal backend.
//!
//! Set `THREERS_BACKEND=metal` in a build with the `metal` feature, on an
//! Apple platform, and every [`HeadlessRenderer`](super::HeadlessRenderer)
//! draws with a [`MetalHeadlessRenderer`] instead of wgpu — no change to the
//! program. The pixels come back in the same layout: `render_to_rgba` at the
//! render size, `render_to_rgba_resolved` averaged to the output size, the
//! pipelined reads one frame behind.
//!
//! GPU captions blend into the Metal frame on the CPU, in linear light as the
//! wgpu pass does. Marks drawn through `mark_renderer` / `draw_marks` /
//! `pick_marks` and effect composers run through `compose` draw on Metal too.
//! What the Metal backend does not have, the renderer does not pretend to: a
//! call that needs wgpu itself — the device or queue to build resources with,
//! the target texture, `mark_target`, overlays, depth of field,
//! motion blur, the mapped RGB frames — hands the renderer back to wgpu for
//! the rest of its life and says so on stderr, once per call site kind. The
//! program still runs and its output is still right; the warning is the
//! list of what to port.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use crate::cameras::Camera;
use crate::metal::enums::pixel_format;
use crate::metal::{MetalHeadlessRenderer, SsaoSettings};
use crate::scene::Scene;

/// Whether a renderer on `backend` draws with the native Metal backend:
/// asked for by name, or, for `Auto`, by `THREERS_BACKEND=metal`.
pub(super) fn requested(backend: super::RenderBackend) -> bool {
    match backend {
        super::RenderBackend::Metal => true,
        super::RenderBackend::Wgpu => false,
        super::RenderBackend::Auto => {
            std::env::var("THREERS_BACKEND").is_ok_and(|v| v.eq_ignore_ascii_case("metal"))
        }
    }
}

/// Say why this renderer is on wgpu after all — once per reason per process,
/// so a frame loop does not print it every frame.
pub(super) fn warn_fallback(reason: &str) {
    static SAID: Mutex<Option<HashSet<String>>> = Mutex::new(None);
    let mut said = SAID.lock().unwrap_or_else(|e| e.into_inner());
    if said.get_or_insert_with(HashSet::new).insert(reason.to_string()) {
        eprintln!("threers: THREERS_BACKEND=metal: {reason} needs wgpu; drawing with wgpu from here");
    }
}

/// Unread frames after which the doubled drawing is said on stderr.
const UNCONFIRMED_NOTE: u32 = 120;

/// The Metal side of a headless renderer.
pub(super) struct MetalPath {
    hr: MetalHeadlessRenderer,
    format: crate::metal::enums::NSUInteger,
    size: (u32, u32),
    msaa: u32,
    /// The last frame drawn, at the render size.
    frame: Vec<u8>,
    /// The pipelined reads' frame in flight: one behind, as on wgpu.
    pending: Option<Vec<u8>>,
    /// wgpu's `set_ssao(strength, radius)`; the camera half is per frame.
    ssao: Option<(f32, f32)>,
    /// The grain's amount and seed. The seed advances a frame, as wgpu's
    /// does when it packs one, so the grain moves and both backends number
    /// the frames alike.
    grain: f32,
    grain_seed: f32,
    /// A frame has been read back from Metal. Until then wgpu draws each frame
    /// too, so a program that turns out to want wgpu itself — marks or
    /// captions over the frame, the device to build on — finds the frame it
    /// expects there rather than an empty target.
    confirmed: bool,
    /// Frames drawn before `confirmed`: each one drawn twice.
    unconfirmed: u32,
    /// `compose`'s post-processing state, and the screen it draws into.
    post: Option<crate::metal::MetalPostFx>,
    screen: Option<crate::postprocessing::Target>,
    /// What marks draw with: this renderer's device and queue.
    marks: Option<crate::marks::gpu::Gpu>,
    /// Something drew over the target since `frame` was read (marks) — or
    /// may have, once its `mark_target` was handed out; the next read reads
    /// it again.
    stale: std::sync::atomic::AtomicBool,
}

impl MetalPath {
    /// A Metal renderer the size of the wgpu one, or `None` (with the reason
    /// on stderr) when the format or the device is not available.
    pub(super) fn new(color_format: wgpu::TextureFormat, width: u32, height: u32) -> Option<Self> {
        // On a plain `Unorm` target the wgpu mesh shader encodes sRGB itself,
        // so the bytes read back are encoded either way. The Metal shaders
        // leave encoding to the attachment; an sRGB one gives the same bytes,
        // where a `Unorm` one would store linear light and read back dark.
        let format = match color_format {
            wgpu::TextureFormat::Rgba8UnormSrgb | wgpu::TextureFormat::Rgba8Unorm => {
                pixel_format::RGBA8_UNORM_SRGB
            }
            wgpu::TextureFormat::Bgra8UnormSrgb | wgpu::TextureFormat::Bgra8Unorm => {
                pixel_format::BGRA8_UNORM_SRGB
            }
            wgpu::TextureFormat::Rgba16Float => pixel_format::RGBA16_FLOAT,
            other => {
                warn_fallback(&format!("the {other:?} target format"));
                return None;
            }
        };
        let mut hr = build(format, width, height, 1)?;
        // Marks draw over a finished frame and load it: keep it whole.
        hr.renderer().set_keep_attachments(true);
        // A `Unorm` target is drawn through an sRGB one; a custom shader's
        // output is still stored as it stands, as on wgpu.
        hr.renderer().set_encoded_output(matches!(
            color_format,
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Bgra8Unorm
        ));
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| {
            eprintln!(
                "threers: THREERS_BACKEND=metal: drawing on the native Metal backend ({})",
                hr.device().name()
            )
        });
        Some(Self {
            hr,
            format,
            size: (width, height),
            msaa: 1,
            frame: Vec::new(),
            pending: None,
            ssao: None,
            grain: 0.0,
            grain_seed: 0.0,
            confirmed: false,
            unconfirmed: 0,
            post: None,
            screen: None,
            marks: None,
            stale: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// MSAA is fixed when a Metal target is built; a change rebuilds it.
    pub(super) fn set_msaa(&mut self, samples: u32) -> bool {
        let samples = if samples <= 1 { 1 } else { 4 };
        if samples == self.msaa {
            return true;
        }
        match build(self.format, self.size.0, self.size.1, samples) {
            Some(mut hr) => {
                hr.renderer().set_keep_attachments(true);
                self.hr = hr;
                self.msaa = samples;
                true
            }
            None => false,
        }
    }

    pub(super) fn set_bloom(&mut self, strength: f32, threshold: f32, radius: f32) {
        self.hr.set_bloom(strength, threshold, radius);
    }

    pub(super) fn set_ssao(&mut self, strength: f32, radius: f32) {
        self.ssao = Some((strength, radius));
    }

    pub(super) fn set_film_grade(&mut self, vignette: f32, grain: f32) {
        self.grain = grain;
        self.hr.set_film_grade(vignette, grain);
    }

    /// Carry what a program set on the wgpu renderer — tone mapping, and the
    /// viewport, scissor and auto-clear a [`QuadView`](crate::helpers::QuadView)
    /// sets per pane — to the Metal one, before it draws or clears.
    fn follow(&mut self, wgpu: &super::Renderer) {
        let m = self.hr.renderer();
        let (mode, exposure) = wgpu.tone_mapping();
        m.set_tone_mapping(mode, exposure);
        m.set_pixel_ratio(wgpu.pixel_ratio());
        match wgpu.viewport() {
            Some(r) => m.set_viewport(r.x, r.y, r.width, r.height),
            None => m.reset_viewport(),
        }
        match wgpu.scissor() {
            Some(r) => m.set_scissor(r.x, r.y, r.width, r.height),
            None => m.reset_scissor(),
        }
        m.set_scissor_test(wgpu.scissor_test());
        m.set_auto_clear(wgpu.auto_clear());
    }

    /// Draw a frame with the wgpu renderer's settings (see `follow`).
    pub(super) fn render(
        &mut self,
        scene: &mut Scene,
        camera: &dyn Camera,
        wgpu: &super::Renderer,
    ) -> Result<(), String> {
        self.follow(wgpu);
        if let Some((strength, radius)) = self.ssao {
            self.hr.set_ssao(Some(ssao_for(camera, strength, radius)));
        }
        if self.grain > 0.0 {
            self.grain_seed += 1.0;
            self.hr.set_grain_seed(self.grain_seed);
        }
        // Not read back here: `fresh` waits for the pixels when something
        // asks for them, so a program that only draws doesn't wait on the GPU.
        self.hr.render_queued(scene, camera).map_err(|e| e.to_string())?;
        self.stale.store(false, std::sync::atomic::Ordering::Relaxed);
        if !self.confirmed {
            self.unconfirmed += 1;
            if self.unconfirmed == UNCONFIRMED_NOTE {
                static SAID: std::sync::Once = std::sync::Once::new();
                SAID.call_once(|| {
                    eprintln!(
                        "threers: THREERS_BACKEND=metal: {UNCONFIRMED_NOTE} frames drawn and none read back; \
                         wgpu draws each one as well until a frame is read from Metal"
                    )
                });
            }
        }
        Ok(())
    }

    pub(super) fn clear(
        &mut self,
        color: crate::math::Color,
        alpha: f32,
        wgpu: &super::Renderer,
    ) -> Result<(), String> {
        self.follow(wgpu);
        self.hr.discard_queued();
        self.hr.clear(color, alpha).map_err(|e| e.to_string())?;
        self.frame = self.hr.read_rgba().map_err(|e| e.to_string())?;
        self.stale.store(false, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    pub(super) fn upload_geometry(&mut self, geometry: &Arc<crate::core::BufferGeometry>) {
        if let Err(e) = self.hr.renderer().upload_geometry(geometry) {
            log::warn!("metal: geometry upload failed: {e}");
        }
    }

    pub(super) fn vertex_buffer(&mut self, geometry: &Arc<crate::core::BufferGeometry>) -> Option<crate::marks::gpu::Buffer> {
        let (id, size) = self.hr.renderer().vertex_buffer(geometry)?;
        // SAFETY: the renderer's live buffer of `size` bytes.
        Some(unsafe { crate::marks::gpu::Buffer::from_metal(id, size as u64) })
    }

    /// The GPU marks draw with: this renderer's own Metal device and queue,
    /// so they draw straight into its frame.
    pub(super) fn marks_gpu(&mut self) -> crate::marks::gpu::Gpu {
        let device = self.hr.device();
        self.marks.get_or_insert_with(|| crate::marks::gpu::Gpu::from(device)).clone()
    }

    /// The frame just drawn as marks see it: its colour (multisampled, and
    /// what that resolves to) and depth, kept by the pass for this.
    pub(super) fn mark_target(&self) -> crate::marks::MarkTarget<'static> {
        // Whoever has it may draw into it.
        self.drawn_over();
        // SAFETY: the target's own live textures; the views retain them.
        unsafe { crate::marks::MarkTarget::from_metal_pass(&self.hr.target().attachments()) }
    }

    /// Something drew over the target (marks): read it again when next asked.
    pub(super) fn drawn_over(&self) {
        self.stale.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// `frame`: the queued frame's pixels, or the target read again if
    /// something drew over it since.
    fn fresh(&mut self) {
        if self.stale.swap(false, std::sync::atomic::Ordering::Relaxed) {
            self.hr.discard_queued();
            match self.hr.read_rgba() {
                Ok(f) => self.frame = f,
                Err(e) => log::warn!("THREERS_BACKEND=metal: reading the frame back after marks failed: {e}"),
            }
        } else if let Err(e) = self.hr.finish_queued_into(&mut self.frame) {
            log::warn!("THREERS_BACKEND=metal: reading the frame back failed: {e}");
        }
    }

    /// Wait for the frame in flight; its pixels are copied out when read.
    pub(super) fn wait_idle(&self) {
        if let Err(e) = self.hr.wait_queued() {
            log::warn!("THREERS_BACKEND=metal: waiting for the frame failed: {e}");
        }
    }

    /// Run an effect composer on Metal and read its screen back.
    pub(super) fn compose(
        &mut self,
        composer: &mut crate::postprocessing::EffectComposer,
        scene: &mut Scene,
        camera: &dyn Camera,
        wgpu: &super::Renderer,
    ) -> Result<Vec<u8>, String> {
        self.follow(wgpu);
        if self.post.is_none() {
            let mut post = crate::metal::MetalPostFx::new(self.hr.device()).map_err(|e| e.to_string())?;
            // Compiled up front, so a failure hands the composer to wgpu
            // before it has a target here.
            post.prepare().map_err(|e| e.to_string())?;
            self.post = Some(post);
        }
        let post = self.post.as_mut().expect("just made");
        let (w, h) = composer.drawing_size();
        let mut backend = post.backend(self.hr.renderer());
        let screen = crate::postprocessing::ensure_target(&mut self.screen, &mut backend, w, h, composer.screen_format());
        composer.render_to_target(&mut backend, scene, camera, screen.clone());
        self.confirmed = true;
        post.read_rgba(screen.id()).map_err(|e| e.to_string())
    }

    /// The last frame at the render size. Reading it confirms the Metal path.
    pub(super) fn frame(&mut self) -> Vec<u8> {
        self.fresh();
        self.confirmed = true;
        self.frame.clone()
    }

    /// Blend a straight-alpha sRGB overlay the size of the frame over it, in
    /// linear light — the arithmetic of the wgpu caption pass, on the bytes
    /// already read back.
    #[cfg(feature = "captions")]
    pub(super) fn blend_overlay(&mut self, rgba: &[u8]) {
        self.fresh();
        let lut = srgb_to_linear_lut();
        for (dst, src) in self
            .frame
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(rgba.as_chunks::<4>().0)
        {
            let a = src[3] as f32 / 255.0;
            if a <= 0.0 {
                continue;
            }
            for c in 0..3 {
                let v = lut[src[c] as usize] * a + lut[dst[c] as usize] * (1.0 - a);
                dst[c] = linear_to_srgb_byte(v);
            }
            let da = dst[3] as f32 / 255.0;
            dst[3] = ((a + da * (1.0 - a)) * 255.0 + 0.5) as u8;
        }
    }

    /// Whether wgpu can stop drawing alongside (see `confirmed`).
    pub(super) fn confirmed(&self) -> bool {
        self.confirmed
    }

    /// The last frame, one behind: what the pipelined reads hand out.
    pub(super) fn pipelined(&mut self, frame: Vec<u8>) -> Option<Vec<u8>> {
        self.pending.replace(frame)
    }

    pub(super) fn finish(&mut self) -> Option<Vec<u8>> {
        self.pending.take()
    }
}

fn build(format: crate::metal::enums::NSUInteger, width: u32, height: u32, msaa: u32) -> Option<MetalHeadlessRenderer> {
    match MetalHeadlessRenderer::builder()
        .size(width, height)
        .msaa(msaa)
        .color_format(format)
        .build()
    {
        Ok(hr) => Some(hr),
        Err(e) => {
            warn_fallback(&format!("this machine ({e})"));
            None
        }
    }
}

/// The wgpu pass reads the lens and clip planes off the frame; the Metal one
/// is told them. Both come from the projection: `P[5]` is `1/tan(fov/2)`,
/// `P[5]/P[0]` the aspect.
fn ssao_for(camera: &dyn Camera, strength: f32, radius: f32) -> SsaoSettings {
    let p = camera.projection_matrix().elements;
    let (near, far) = camera.near_far();
    let (tan_half_fov, aspect) = if p[5] != 0.0 && p[0] != 0.0 {
        (1.0 / p[5], p[5] / p[0])
    } else {
        (1.0, 1.0)
    };
    SsaoSettings { strength, radius, near, far, tan_half_fov, aspect }
}

#[cfg(feature = "captions")]
fn srgb_to_linear_lut() -> &'static [f32; 256] {
    static LUT: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    LUT.get_or_init(|| {
        std::array::from_fn(|i| {
            let s = i as f32 / 255.0;
            if s <= 0.040_45 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            }
        })
    })
}

#[cfg(feature = "captions")]
fn linear_to_srgb_byte(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let s = if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0 + 0.5) as u8
}
