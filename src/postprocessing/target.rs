//! Render targets as the composer and its passes hold them.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use super::PostBackend;
use crate::renderer::{RenderTarget, Renderer};

/// Ids for the targets the composer allocates. Every target a post-fx draw
/// reads or writes is registered with the renderer under an id, and the wasm
/// layer numbers its own from 1 — so these start far above anything it will
/// reach, and the two never name the same target.
static NEXT_ID: AtomicU32 = AtomicU32::new(0x4000_0000);

/// The colour format of every intermediate target: three.js's
/// `EffectComposer` and its passes use `HalfFloatType`, so the chain carries
/// linear light, above 1 included, from pass to pass.
pub const INTERMEDIATE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// A render target registered with a renderer — a [`PostBackend`](super::PostBackend):
/// three.js's `WebGLRenderTarget`, as `readBuffer` / `writeBuffer` and as
/// each pass's own scratch targets. Cheap to clone — the handles share the
/// texture.
///
/// The id is what a [`PostFxOp`](crate::renderer::PostFxOp) names it by; the
/// texture lives with the backend that registered it. On wgpu the handle
/// carries the [`RenderTarget`] too.
#[derive(Clone)]
pub struct Target {
    id: u32,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    rt: Option<Arc<RenderTarget>>,
}

impl std::fmt::Debug for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Target")
            .field("id", &self.id)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("format", &self.format)
            .finish()
    }
}

impl Target {
    /// A new `width × height` target of `format`, registered with `renderer`.
    pub fn new(
        renderer: &mut Renderer,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        let rt = Arc::new(RenderTarget::new(
            &renderer.device_arc(),
            width.max(1),
            height.max(1),
            format,
        ));
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        renderer.register_render_target(id, &rt);
        Self::from_registered(id, rt)
    }

    /// A fresh id for a target a backend other than wgpu allocates itself.
    pub fn next_id() -> u32 {
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    }

    /// A handle to a target another backend holds under `id`.
    pub fn external(id: u32, width: u32, height: u32, format: wgpu::TextureFormat) -> Self {
        Self {
            id,
            width: width.max(1),
            height: height.max(1),
            format,
            rt: None,
        }
    }

    /// A half-float target, the kind every pass works in.
    pub fn intermediate(renderer: &mut Renderer, width: u32, height: u32) -> Self {
        Self::new(renderer, width, height, INTERMEDIATE_FORMAT)
    }

    /// Wrap a render target registered elsewhere under `id` — one the caller
    /// handed the composer, or the wasm layer's.
    pub fn from_registered(id: u32, rt: Arc<RenderTarget>) -> Self {
        Self {
            id,
            width: rt.width,
            height: rt.height,
            format: rt.format,
            rt: Some(rt),
        }
    }

    pub fn id(&self) -> u32 {
        self.id
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    /// The wgpu render target behind this handle.
    ///
    /// # Panics
    /// On a target another backend holds — see
    /// [`wgpu_render_target`](Self::wgpu_render_target).
    pub fn render_target(&self) -> &Arc<RenderTarget> {
        self.rt
            .as_ref()
            .expect("a wgpu render target (this one belongs to another backend)")
    }

    /// The wgpu render target, if this target is one.
    pub fn wgpu_render_target(&self) -> Option<&Arc<RenderTarget>> {
        self.rt.as_ref()
    }

    /// Whether this target is this exact allocation.
    pub fn same(&self, other: &Target) -> bool {
        self.id == other.id
    }
}

/// `slot`, allocated or reallocated so it is `width × height` of `format`,
/// and a handle to it. The target it replaces is unregistered, so a resize
/// frees the old allocation rather than leaving the renderer holding it.
pub fn ensure(
    slot: &mut Option<Target>,
    renderer: &mut (impl PostBackend + ?Sized),
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> Target {
    let (width, height) = (width.max(1), height.max(1));
    if let Some(t) = slot {
        if t.size() == (width, height) && t.format() == format {
            return t.clone();
        }
        renderer.unregister_render_target(t.id);
    }
    let t = renderer.new_target(width, height, format);
    *slot = Some(t.clone());
    t
}

/// Drop `slot`'s target, unregistering it.
pub fn release(slot: &mut Option<Target>, renderer: &mut (impl PostBackend + ?Sized)) {
    if let Some(t) = slot.take() {
        renderer.unregister_render_target(t.id);
    }
}
