//! three.js viewport, scissor and auto-clear state for
//! [`MetalRenderer`] — the Metal side of [`crate::renderer::viewport`], with
//! the same coordinates and the same semantics.
//!
//! # Coordinates
//!
//! Rectangles are **bottom-left origin, in logical pixels**, multiplied by the
//! renderer's [`pixel_ratio`](MetalRenderer::set_pixel_ratio) and rounded to
//! get attachment pixels — exactly as on the wgpu renderer, which shares the
//! planning code ([`crate::renderer::viewport`]). Metal's own origin is
//! top-left; the flip happens there, once.
//!
//! # What a sub-rectangle changes
//!
//! With the default state (no viewport, no scissor test, auto-clear on) a
//! render is byte-for-byte what it always was. Otherwise:
//!
//! * colour is cleared to the background over the scissor rectangle when the
//!   scissor test is on, over the whole target when it is off, and not at all
//!   with [`auto_clear`](MetalRenderer::set_auto_clear) off. Depth is cleared
//!   over the whole target every render, which non-overlapping panes cannot
//!   see. Metal has no partial clear any more than WebGPU does: the
//!   rectangle is a scissored full-screen triangle writing the clear colour,
//!   converted exactly as the pass's own clear colour is.
//! * MSAA works in a pane. A resolve rewrites every pixel of the resolve
//!   texture, so a pane cannot resolve straight into the target without
//!   erasing the panes before it — the multisampled texture does not hold
//!   them (after a plain render it is resolved and discarded). So a render
//!   whose clear covers only the scissor rectangle is drawn into a
//!   pane-sized multisampled colour + depth pair (cached per size), resolved
//!   into a pane-sized texture, and composited into the target's resolve
//!   texture over exactly that rectangle — the same
//!   `PanePlan` the wgpu renderer
//!   uses. With auto-clear off (nothing to start a fresh pane target from),
//!   or for views carrying their own viewport or colour texture, the render
//!   draws single-sampled straight into the resolve texture, with a
//!   single-sampled scratch depth buffer.
//! * the state applies to [`render`](MetalRenderer::render) and to every view
//!   of [`render_views`](MetalRenderer::render_views) that has no viewport of
//!   its own. A view with its own viewport keeps it (scissored to it, and to
//!   the scissor rectangle when the test is on). The layered stereo path is
//!   used only in the default state.

use std::collections::HashMap;

use super::device::{compile_library, MetalDevice, MetalError};
use super::enums::*;
use super::objc::{
    alloc_init, class, error_message, msg0, msg1, msg2, msg3, nsstring, sel,
    AutoreleasePool, Id, Owned, NIL, NO,
};
use super::renderer::{clear_color, MetalRenderer, PassAttachments, RenderView};
use super::resources::new_texture_2d;
use crate::math::Color;
use crate::renderer::viewport::{ClearMode, PanePlan, PixelRect, ViewportPlan, ViewportSettings};
use crate::renderer::ViewportRect;

const FILL_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct FillOut { float4 position [[position]]; };
vertex FillOut vs_viewport_fill(uint i [[vertex_id]]) {
    float2 p = float2(float((i << 1) & 2u), float(i & 2u));
    FillOut o;
    o.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
    return o;
}
fragment float4 fs_viewport_fill(constant float4 &color [[buffer(0)]]) {
    return color;
}
fragment float4 fs_pane_composite(FillOut in [[stage_in]],
                                  texture2d<float, access::read> src [[texture(0)]],
                                  constant uint2 &origin [[buffer(0)]]) {
    return src.read(uint2(in.position.xy) - origin);
}
"#;

/// Distinct pane sizes kept at once; see the wgpu renderer's equivalent.
const MAX_PANE_TARGETS: usize = 8;

/// `(width, height, colour format, depth format, samples)`.
type PaneKey = (u32, u32, NSUInteger, NSUInteger, u32);

/// A pane-sized multisampled colour (and depth), and what it resolves into.
struct PaneTarget {
    color: Owned,
    depth: Option<Owned>,
    resolve: Owned,
}

/// Copy a resolved pane into `region` of `dest`'s resolve texture.
#[derive(Clone, Copy)]
pub(crate) struct PaneComposite {
    /// Top-left, attachment pixels.
    pub region: PixelRect,
    pub source: Id,
    /// The caller's attachments, drawable and all.
    pub dest: PassAttachments,
}

/// The renderer's viewport settings plus what the partial clear needs.
pub(crate) struct MetalViewportState {
    /// The same settings, and the same planning, as the wgpu renderer's.
    pub settings: ViewportSettings,
    /// Attachment pixels of the target last drawn or cleared (or set with
    /// [`MetalRenderer::set_drawing_buffer_size`]) — what `size` reports.
    pub buffer_size: (u32, u32),
    device: MetalDevice,
    library: Option<Owned>,
    /// `(colour format, depth format, samples)` → fill pipeline.
    pipelines: HashMap<(NSUInteger, NSUInteger, u32), Owned>,
    /// Single-sampled depth for an MSAA target drawn without MSAA.
    scratch_depth: Option<((u32, u32, NSUInteger), Owned)>,
    pane_targets: HashMap<PaneKey, PaneTarget>,
    /// Colour format → composite pipeline.
    composite_pipelines: HashMap<NSUInteger, Owned>,
}

/// A render's attachments and views once the state has been applied.
pub(crate) struct StagedPass {
    pub clear: ClearMode,
    /// The scissor rectangle (top-left, attachment pixels) while the scissor
    /// test is on.
    pub scissor: Option<PixelRect>,
    pub views: Vec<RenderView>,
    pub pass: PassAttachments,
    /// Set when `pass` is a pane target: what to copy back afterwards.
    pub composite: Option<PaneComposite>,
}

impl MetalViewportState {
    pub fn new(device: MetalDevice) -> Self {
        Self {
            settings: ViewportSettings::default(),
            buffer_size: (0, 0),
            device,
            library: None,
            pipelines: HashMap::new(),
            scratch_depth: None,
            pane_targets: HashMap::new(),
            composite_pipelines: HashMap::new(),
        }
    }

    /// Apply a non-legacy plan to `views` and `pass`.
    pub fn stage(
        &mut self,
        p: &ViewportPlan,
        views: &[RenderView],
        pass: &PassAttachments,
    ) -> Result<StagedPass, MetalError> {
        // Multisampled with a partial clear: draw into a pane target.
        if let Some(pane) = p.pane.filter(|_| {
            !pass.resolve.is_null()
                && pass.sample_count > 1
                && views
                    .iter()
                    .all(|v| v.viewport.is_none() && v.color.is_null())
        }) {
            return self.stage_pane(&pane, views, pass);
        }
        let mut staged = *pass;
        if p.clear != ClearMode::Full && !pass.resolve.is_null() {
            staged.color = pass.resolve;
            staged.resolve = NIL;
            staged.sample_count = 1;
            if !pass.depth.is_null() {
                staged.depth = self.scratch_depth(pass.width, pass.height, pass.depth_format)?;
            }
        }
        let v = p.viewport;
        let views = views
            .iter()
            .map(|view| {
                let mut view = view.clone();
                if view.viewport.is_none() {
                    view.viewport = Some(MTLViewport {
                        origin_x: v.x as f64,
                        origin_y: v.y as f64,
                        width: v.width as f64,
                        height: v.height as f64,
                        znear: 0.0,
                        zfar: 1.0,
                    });
                    if let Some(fix) = &p.projection_fix {
                        view.projection_matrix = fix.multiply(&view.projection_matrix);
                    }
                }
                view
            })
            .collect();
        Ok(StagedPass {
            clear: p.clear,
            scissor: self.settings.scissor_test.then_some(p.scissor),
            views,
            pass: staged,
            composite: None,
        })
    }

    fn stage_pane(
        &mut self,
        pane: &PanePlan,
        views: &[RenderView],
        pass: &PassAttachments,
    ) -> Result<StagedPass, MetalError> {
        let r = pane.region;
        let depth_format = if pass.depth.is_null() {
            0
        } else {
            pass.depth_format
        };
        let key = (r.width, r.height, pass.color_format, depth_format, pass.sample_count);
        if !self.pane_targets.contains_key(&key) {
            if self.pane_targets.len() >= MAX_PANE_TARGETS {
                self.pane_targets.clear();
            }
            let tex = |format, usage, samples| {
                new_texture_2d(
                    &self.device,
                    r.width,
                    r.height,
                    format,
                    usage,
                    storage_mode::PRIVATE,
                    samples,
                    1,
                )
            };
            let target = PaneTarget {
                color: tex(pass.color_format, texture_usage::RENDER_TARGET, pass.sample_count)?,
                depth: if depth_format == 0 {
                    None
                } else {
                    Some(tex(depth_format, texture_usage::RENDER_TARGET, pass.sample_count)?)
                },
                resolve: tex(
                    pass.color_format,
                    texture_usage::RENDER_TARGET | texture_usage::SHADER_READ,
                    1,
                )?,
            };
            self.pane_targets.insert(key, target);
        }
        let t = &self.pane_targets[&key];
        let mut staged = *pass;
        staged.color = t.color.id();
        staged.resolve = t.resolve.id();
        staged.depth = t.depth.as_ref().map(|d| d.id()).unwrap_or(NIL);
        staged.drawable = NIL;
        staged.width = r.width;
        staged.height = r.height;
        let v = pane.viewport;
        let views = views
            .iter()
            .map(|view| {
                let mut view = view.clone();
                view.viewport = Some(MTLViewport {
                    origin_x: v.x as f64,
                    origin_y: v.y as f64,
                    width: v.width as f64,
                    height: v.height as f64,
                    znear: 0.0,
                    zfar: 1.0,
                });
                if let Some(fix) = &pane.projection_fix {
                    view.projection_matrix = fix.multiply(&view.projection_matrix);
                }
                view
            })
            .collect();
        Ok(StagedPass {
            clear: ClearMode::Full,
            scissor: None,
            views,
            pass: staged,
            composite: Some(PaneComposite {
                region: r,
                source: t.resolve.id(),
                dest: *pass,
            }),
        })
    }

    /// Copy the resolved pane into its rectangle of the destination's resolve
    /// texture, on a command buffer of its own queued behind the pane's.
    /// `read` with no filtering, and an sRGB decode then encode, is exact.
    pub fn composite(&mut self, c: &PaneComposite) -> Result<(), MetalError> {
        let pipeline = self.composite_pipeline(c.dest.color_format)?;
        let _pool = AutoreleasePool::new();
        unsafe {
            let cmd = self.device.command_buffer();
            if cmd.is_null() {
                return Err(MetalError::NoCommandQueue);
            }
            let rpd: Id = msg0(class!("MTLRenderPassDescriptor"), sel!("renderPassDescriptor"));
            let attachments: Id = msg0(rpd, sel!("colorAttachments"));
            let a: Id = msg1(attachments, sel!("objectAtIndexedSubscript:"), 0usize);
            let _: () = msg1(a, sel!("setTexture:"), c.dest.resolve);
            let _: () = msg1(a, sel!("setLoadAction:"), load_action::LOAD);
            let _: () = msg1(a, sel!("setStoreAction:"), store_action::STORE);
            let encoder: Id = msg1(cmd, sel!("renderCommandEncoderWithDescriptor:"), rpd);
            if encoder.is_null() {
                return Err(MetalError::Unsupported(
                    "pane composite: render command encoder could not be created".into(),
                ));
            }
            let _: () = msg1(encoder, sel!("setScissorRect:"), scissor_rect(c.region));
            let _: () = msg1(encoder, sel!("setRenderPipelineState:"), pipeline);
            let _: () = msg2(
                encoder,
                sel!("setFragmentTexture:atIndex:"),
                c.source,
                0usize,
            );
            let origin = [c.region.x, c.region.y];
            let _: () = msg3(
                encoder,
                sel!("setFragmentBytes:length:atIndex:"),
                origin.as_ptr() as *const std::ffi::c_void,
                std::mem::size_of_val(&origin) as NSUInteger,
                0usize,
            );
            let _: () = msg3(
                encoder,
                sel!("drawPrimitives:vertexStart:vertexCount:"),
                primitive::TRIANGLE,
                0usize,
                3usize,
            );
            let _: () = msg0(encoder, sel!("endEncoding"));
            let _: () = msg0(cmd, sel!("commit"));
        }
        Ok(())
    }

    fn composite_pipeline(&mut self, color_format: NSUInteger) -> Result<Id, MetalError> {
        if let Some(p) = self.composite_pipelines.get(&color_format) {
            return Ok(p.id());
        }
        let state = self.build_pipeline("fs_pane_composite", color_format, 0, 1)?;
        let id = state.id();
        self.composite_pipelines.insert(color_format, state);
        Ok(id)
    }

    fn scratch_depth(&mut self, w: u32, h: u32, format: NSUInteger) -> Result<Id, MetalError> {
        let key = (w, h, format);
        if !matches!(&self.scratch_depth, Some((k, _)) if *k == key) {
            let tex = new_texture_2d(
                &self.device,
                w,
                h,
                format,
                texture_usage::RENDER_TARGET,
                storage_mode::PRIVATE,
                1,
                1,
            )?;
            self.scratch_depth = Some((key, tex));
        }
        Ok(self.scratch_depth.as_ref().expect("allocated above").1.id())
    }

    /// The scissored fill pipeline for these attachments, built on first use.
    pub fn fill_pipeline(
        &mut self,
        color_format: NSUInteger,
        depth_format: NSUInteger,
        samples: u32,
    ) -> Result<Id, MetalError> {
        let key = (color_format, depth_format, samples.max(1));
        if let Some(p) = self.pipelines.get(&key) {
            return Ok(p.id());
        }
        let state = self.build_pipeline("fs_viewport_fill", color_format, depth_format, key.2)?;
        let id = state.id();
        self.pipelines.insert(key, state);
        Ok(id)
    }

    /// A pipeline of the full-screen triangle and `fragment`, unblended.
    fn build_pipeline(
        &mut self,
        fragment: &str,
        color_format: NSUInteger,
        depth_format: NSUInteger,
        samples: u32,
    ) -> Result<Owned, MetalError> {
        let _pool = AutoreleasePool::new();
        if self.library.is_none() {
            self.library = Some(unsafe { compile_library(self.device.device_id(), FILL_MSL)? });
        }
        let lib = self.library.as_ref().expect("compiled above").id();
        let state = unsafe {
            let function = |name: &str| -> Result<Owned, MetalError> {
                let f: Id = msg1(lib, sel!("newFunctionWithName:"), nsstring(name));
                Owned::from_retained(f).ok_or_else(|| MetalError::MissingFunction(name.into()))
            };
            let (vs, fs) = (function("vs_viewport_fill")?, function(fragment)?);
            let desc = Owned::from_retained(alloc_init(class!("MTLRenderPipelineDescriptor")))
                .ok_or_else(|| MetalError::Allocation("MTLRenderPipelineDescriptor".into()))?;
            let _: () = msg1(desc.id(), sel!("setVertexFunction:"), vs.id());
            let _: () = msg1(desc.id(), sel!("setFragmentFunction:"), fs.id());
            let _: () = msg1(
                desc.id(),
                sel!("setRasterSampleCount:"),
                samples.max(1) as NSUInteger,
            );
            if depth_format != 0 {
                let _: () = msg1(desc.id(), sel!("setDepthAttachmentPixelFormat:"), depth_format);
            }
            let attachments: Id = msg0(desc.id(), sel!("colorAttachments"));
            let color: Id = msg1(attachments, sel!("objectAtIndexedSubscript:"), 0usize);
            let _: () = msg1(color, sel!("setPixelFormat:"), color_format);
            let _: () = msg1(color, sel!("setBlendingEnabled:"), NO);
            let mut err: Id = NIL;
            let state: Id = msg2(
                self.device.device_id(),
                sel!("newRenderPipelineStateWithDescriptor:error:"),
                desc.id(),
                &mut err as *mut Id,
            );
            Owned::from_retained(state)
                .ok_or_else(|| MetalError::PipelineCreation(error_message(err)))?
        };
        Ok(state)
    }
}

/// `rect` as Metal's scissor rectangle.
pub(crate) fn scissor_rect(rect: PixelRect) -> MTLScissorRect {
    MTLScissorRect {
        x: rect.x as NSUInteger,
        y: rect.y as NSUInteger,
        width: rect.width as NSUInteger,
        height: rect.height as NSUInteger,
    }
}

/// The viewport as a scissor rectangle inside a `width`×`height` attachment;
/// `None` when it misses the attachment entirely.
pub(crate) fn scissor_for_viewport(
    v: &MTLViewport,
    width: u32,
    height: u32,
) -> Option<MTLScissorRect> {
    let x0 = v.origin_x.floor().max(0.0);
    let y0 = v.origin_y.floor().max(0.0);
    let x1 = (v.origin_x + v.width).ceil().min(width as f64);
    let y1 = (v.origin_y + v.height).ceil().min(height as f64);
    (x1 > x0 && y1 > y0).then_some(MTLScissorRect {
        x: x0 as NSUInteger,
        y: y0 as NSUInteger,
        width: (x1 - x0) as NSUInteger,
        height: (y1 - y0) as NSUInteger,
    })
}

/// The intersection of two scissor rectangles; `None` when they miss.
pub(crate) fn intersect(a: MTLScissorRect, b: MTLScissorRect) -> Option<MTLScissorRect> {
    let x0 = a.x.max(b.x);
    let y0 = a.y.max(b.y);
    let x1 = (a.x + a.width).min(b.x + b.width);
    let y1 = (a.y + a.height).min(b.y + b.height);
    (x1 > x0 && y1 > y0).then_some(MTLScissorRect {
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
    })
}

/// Write `color` over `rect` and nothing else: a full-screen triangle under a
/// scissor. Leaves the scissor set to `rect`; the caller resets it.
///
/// # Safety
/// `encoder` is a live render command encoder whose attachments match the
/// formats `pipeline` was built for.
pub(crate) unsafe fn encode_fill(encoder: Id, pipeline: Id, rect: PixelRect, color: MTLClearColor) {
    let _: () = msg1(encoder, sel!("setScissorRect:"), scissor_rect(rect));
    let _: () = msg1(encoder, sel!("setRenderPipelineState:"), pipeline);
    let c = [
        color.red as f32,
        color.green as f32,
        color.blue as f32,
        color.alpha as f32,
    ];
    let _: () = msg3(
        encoder,
        sel!("setFragmentBytes:length:atIndex:"),
        c.as_ptr() as *const std::ffi::c_void,
        std::mem::size_of_val(&c) as NSUInteger,
        0usize,
    );
    let _: () = msg3(
        encoder,
        sel!("drawPrimitives:vertexStart:vertexCount:"),
        primitive::TRIANGLE,
        0usize,
        3usize,
    );
}

impl MetalRenderer {
    /// three.js `setViewport`: draw the next renders into this rectangle of
    /// the target. **Bottom-left origin, logical pixels** (multiplied by
    /// [`pixel_ratio`](Self::set_pixel_ratio)), as on the wgpu
    /// [`Renderer`](crate::Renderer). The rectangle may hang off the target;
    /// the visible part is drawn with the projection adjusted so the picture
    /// is shifted, not squashed. See [`crate::metal::viewport`].
    pub fn set_viewport(&mut self, x: f32, y: f32, width: f32, height: f32) {
        self.viewport_state.settings.viewport = Some(ViewportRect::new(x, y, width, height));
    }

    /// The viewport set by [`set_viewport`](Self::set_viewport), `None` for the
    /// whole target.
    pub fn viewport(&self) -> Option<ViewportRect> {
        self.viewport_state.settings.viewport
    }

    /// Back to drawing over the whole target.
    pub fn reset_viewport(&mut self) {
        self.viewport_state.settings.viewport = None;
    }

    /// three.js `setScissor`. Same coordinates as
    /// [`set_viewport`](Self::set_viewport); only takes effect while
    /// [`set_scissor_test`](Self::set_scissor_test) is on. With the test on and
    /// no rectangle set, the scissor is the whole target.
    pub fn set_scissor(&mut self, x: f32, y: f32, width: f32, height: f32) {
        self.viewport_state.settings.scissor = Some(ViewportRect::new(x, y, width, height));
    }

    pub fn scissor(&self) -> Option<ViewportRect> {
        self.viewport_state.settings.scissor
    }

    /// Forget the scissor rectangle: with the test on, scissor the whole target.
    pub fn reset_scissor(&mut self) {
        self.viewport_state.settings.scissor = None;
    }

    /// three.js `setScissorTest`. While on, the colour clear and every draw
    /// are confined to the scissor rectangle.
    pub fn set_scissor_test(&mut self, enabled: bool) {
        self.viewport_state.settings.scissor_test = enabled;
    }

    pub fn scissor_test(&self) -> bool {
        self.viewport_state.settings.scissor_test
    }

    /// three.js `setPixelRatio`, for the viewport and scissor rectangles only.
    /// [`MetalHeadlessRenderer`](super::MetalHeadlessRenderer) sets it to
    /// `supersample / render_scale`, so rectangles stay in output pixels.
    pub fn set_pixel_ratio(&mut self, ratio: f32) {
        self.viewport_state.settings.set_pixel_ratio(ratio);
    }

    pub fn pixel_ratio(&self) -> f32 {
        self.viewport_state.settings.pixel_ratio()
    }

    /// The drawing buffer in logical pixels — device size over
    /// [`pixel_ratio`](Self::pixel_ratio). three.js `getSize`.
    ///
    /// This renderer owns no target, so the size is that of the attachments
    /// it last drew or cleared, or what
    /// [`set_drawing_buffer_size`](Self::set_drawing_buffer_size) set since.
    pub fn size(&self) -> (f32, f32) {
        let (w, h) = self.viewport_state.buffer_size;
        let pr = self.viewport_state.settings.pixel_ratio();
        (w as f32 / pr, h as f32 / pr)
    }

    /// The drawing buffer in attachment pixels. three.js
    /// `getDrawingBufferSize`.
    pub fn drawing_buffer_size(&self) -> (u32, u32) {
        self.viewport_state.buffer_size
    }

    /// Tell the renderer the size of the target the next renders go to,
    /// before any of them has — what [`size`](Self::size) reports.
    pub fn set_drawing_buffer_size(&mut self, width: u32, height: u32) {
        self.viewport_state.buffer_size = (width, height);
    }

    /// three.js `autoClear`. On (the default), each render clears colour to
    /// the scene background — the scissor rectangle when the scissor test is
    /// on, the whole target otherwise. Off, colour is kept and the scene draws
    /// over it. Depth is cleared either way.
    pub fn set_auto_clear(&mut self, enabled: bool) {
        self.viewport_state.settings.auto_clear = enabled;
    }

    pub fn auto_clear(&self) -> bool {
        self.viewport_state.settings.auto_clear
    }

    /// three.js `renderer.clear()` for colour: fill `pass`'s colour with
    /// `color` (linear, as `scene.background` is) — the scissor rectangle when
    /// the scissor test is on, the whole target otherwise. Depth is not
    /// touched (every render clears it). Never presents `pass`'s drawable.
    pub fn clear(
        &mut self,
        pass: &PassAttachments,
        color: Color,
        alpha: f32,
    ) -> Result<(), MetalError> {
        let _pool = AutoreleasePool::new();
        self.viewport_state.buffer_size = (pass.width, pass.height);
        let settings = &self.viewport_state.settings;
        let Some(mode) = settings.clear_mode(pass.width, pass.height) else {
            return Ok(());
        };
        let value = clear_color(color, alpha);
        let rect = match mode {
            ClearMode::Rect(r) => Some(r),
            _ => None,
        };
        // A partial fill goes single-sampled into the resolved texture, which
        // is where a scissored render will load it from.
        let (texture, resolve, samples) = match rect {
            Some(_) if !pass.resolve.is_null() => (pass.resolve, NIL, 1),
            _ => (pass.color, pass.resolve, pass.sample_count.max(1)),
        };
        let pipeline = match rect {
            Some(_) => Some(self.viewport_state.fill_pipeline(pass.color_format, 0, samples)?),
            None => None,
        };
        unsafe {
            let cmd = self.device().command_buffer();
            if cmd.is_null() {
                return Err(MetalError::NoCommandQueue);
            }
            let rpd: Id = msg0(class!("MTLRenderPassDescriptor"), sel!("renderPassDescriptor"));
            let attachments: Id = msg0(rpd, sel!("colorAttachments"));
            let c: Id = msg1(attachments, sel!("objectAtIndexedSubscript:"), 0usize);
            let _: () = msg1(c, sel!("setTexture:"), texture);
            let _: () = msg1(
                c,
                sel!("setLoadAction:"),
                if rect.is_some() {
                    load_action::LOAD
                } else {
                    load_action::CLEAR
                },
            );
            let _: () = msg1(c, sel!("setClearColor:"), value);
            if resolve.is_null() {
                let _: () = msg1(c, sel!("setStoreAction:"), store_action::STORE);
            } else {
                let _: () = msg1(c, sel!("setResolveTexture:"), resolve);
                let _: () = msg1(c, sel!("setStoreAction:"), store_action::MULTISAMPLE_RESOLVE);
            }
            let encoder: Id = msg1(cmd, sel!("renderCommandEncoderWithDescriptor:"), rpd);
            if encoder.is_null() {
                return Err(MetalError::Unsupported(
                    "clear: render command encoder could not be created".into(),
                ));
            }
            if let (Some(r), Some(p)) = (rect, pipeline) {
                encode_fill(encoder, p, r, value);
            }
            let _: () = msg0(encoder, sel!("endEncoding"));
            let _: () = msg0(cmd, sel!("commit"));
        }
        Ok(())
    }

    /// Present `pass`'s drawable on a command buffer of its own, queued behind
    /// everything already committed. For a frame drawn in several renders —
    /// [`MetalViewSurface`](crate::helpers::quad_view::MetalViewSurface) draws
    /// with the drawable stripped and presents once at the end. No-op without
    /// a drawable.
    pub fn present(&self, pass: &PassAttachments) -> Result<(), MetalError> {
        if pass.drawable.is_null() {
            return Ok(());
        }
        let _pool = AutoreleasePool::new();
        unsafe {
            let cmd = self.device().command_buffer();
            if cmd.is_null() {
                return Err(MetalError::NoCommandQueue);
            }
            let _: () = msg1(cmd, sel!("presentDrawable:"), pass.drawable);
            let _: () = msg0(cmd, sel!("commit"));
        }
        Ok(())
    }
}

impl PassAttachments {
    /// The same attachments with no drawable, so a render does not present —
    /// for a frame drawn in several renders and presented once with
    /// [`MetalRenderer::present`].
    pub fn without_drawable(mut self) -> Self {
        self.drawable = NIL;
        self
    }
}
