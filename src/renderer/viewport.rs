//! Viewport and scissor state for [`Renderer`](super::Renderer) — what lets one
//! target hold several views, the way three.js does it with `setViewport`,
//! `setScissor` and `setScissorTest`.
//!
//! # Coordinates
//!
//! Rectangles are **bottom-left origin, in logical pixels**, exactly as in
//! three.js: `(0, 0)` is the lower-left corner of the target, `y` grows
//! upwards, and every value is multiplied by the renderer's
//! [`pixel_ratio`](super::Renderer::set_pixel_ratio) and rounded to get device
//! pixels. wgpu's own framebuffer origin is top-left; the flip happens here,
//! once, and nowhere else.
//!
//! # What a sub-rectangle changes
//!
//! With the default state (no viewport, no scissor test, auto-clear on) the
//! renderer behaves exactly as it always did. Otherwise, per render:
//!
//! * the colour clear covers the scissor rectangle when the scissor test is on,
//!   and the whole target when it is off — three.js's `gl.clear` semantics;
//!   with [`auto_clear`](super::Renderer::set_auto_clear) off there is no
//!   colour clear at all. Depth is always cleared for the whole target, which
//!   is invisible to non-overlapping panes.
//! * multisampling works in a pane. When the colour clear covers the whole
//!   target the ordinary multisampled pass resolves into it. When it covers
//!   only the scissor rectangle, the pane is drawn into a pane-sized
//!   multisampled colour + depth pair (cached per size), resolved into a
//!   pane-sized texture and composited into the target over exactly that
//!   rectangle — an MSAA resolve straight into the target would rewrite
//!   every pixel of it and erase the panes before. See `PanePlan`. With
//!   [`auto_clear`](super::Renderer::set_auto_clear) off the render has to
//!   draw over what is there, which a fresh pane target does not hold, so
//!   that case stays single-sampled.
//! * temporal AA and screen-space refraction are off: both treat the frame as
//!   one picture (a history buffer; a full-frame screen-space capture) and
//!   neither has a meaning per pane. Refracting draws are skipped, with a
//!   warning, rather than drawn wrongly.
//! * post-processing is not part of `render` here — `apply_postfx` and friends
//!   are separate calls over a whole texture. Run after the panes, they process
//!   the composite (bloom will bleed across the gaps); per-pane post-fx is not
//!   supported.

use std::collections::HashMap;

use crate::math::Matrix4;

/// A rectangle in logical pixels, bottom-left origin (three.js's convention).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewportRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl ViewportRect {
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Whether the point `(px, py)` (same units and origin) lies inside.
    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x && px < self.x + self.width && py >= self.y && py < self.y + self.height
    }

    /// Width over height, or 1 for a degenerate rectangle.
    pub fn aspect(&self) -> f32 {
        if self.height > 0.0 && self.width > 0.0 {
            self.width / self.height
        } else {
            1.0
        }
    }

    /// Device pixels, still bottom-left origin, rounded as three.js rounds
    /// (`multiplyScalar(pixelRatio).round()`). Signed: a viewport may hang off
    /// the target.
    pub(crate) fn to_device(self, pixel_ratio: f32) -> IRect {
        IRect {
            x: (self.x * pixel_ratio).round() as i64,
            y: (self.y * pixel_ratio).round() as i64,
            w: (self.width * pixel_ratio).round().max(0.0) as i64,
            h: (self.height * pixel_ratio).round().max(0.0) as i64,
        }
    }
}

/// Integer rectangle in device pixels, bottom-left origin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct IRect {
    pub x: i64,
    pub y: i64,
    pub w: i64,
    pub h: i64,
}

impl IRect {
    fn full(w: u32, h: u32) -> Self {
        Self {
            x: 0,
            y: 0,
            w: w as i64,
            h: h as i64,
        }
    }

    fn intersect(self, o: IRect) -> IRect {
        let x0 = self.x.max(o.x);
        let y0 = self.y.max(o.y);
        let x1 = (self.x + self.w).min(o.x + o.w);
        let y1 = (self.y + self.h).min(o.y + o.h);
        IRect {
            x: x0,
            y: y0,
            w: (x1 - x0).max(0),
            h: (y1 - y0).max(0),
        }
    }

    fn is_empty(self) -> bool {
        self.w <= 0 || self.h <= 0
    }

    /// To wgpu's top-left origin. Only valid for a rect inside the target.
    fn to_top_left(self, target_h: u32) -> PixelRect {
        PixelRect {
            x: self.x as u32,
            y: (target_h as i64 - self.y - self.h) as u32,
            width: self.w as u32,
            height: self.h as u32,
        }
    }
}

/// Device pixels, **top-left** origin — what `set_viewport`/`set_scissor_rect`
/// on a wgpu render pass take. Always inside the target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PixelRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// What a render clears its colour attachment with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ClearMode {
    /// `LoadOp::Clear` — the whole target.
    Full,
    /// Load, then draw the clear colour over this rect only.
    Rect(PixelRect),
    /// Load and keep everything (`auto_clear` off).
    None,
}

/// How a render with a partial clear is drawn multisampled: into a target the
/// size of `region` (the scissor rectangle the clear covers), then
/// composited into the real target at `region`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PanePlan {
    /// Where the pane lands in the target — top-left origin, device pixels.
    pub region: PixelRect,
    /// The viewport inside the pane-sized target, top-left origin.
    pub viewport: PixelRect,
    /// Left-multiplied onto the projection: the viewport clipped to the pane
    /// target, shown shifted rather than squashed (as
    /// [`ViewportPlan::projection_fix`] does for the whole target).
    pub projection_fix: Option<Matrix4>,
}

impl PanePlan {
    /// The plan for the pass that draws into the pane-sized target: a full
    /// clear, the local viewport, and the scissor confined to it.
    pub fn local(&self) -> ViewportPlan {
        ViewportPlan {
            legacy: false,
            viewport: self.viewport,
            scissor: self.viewport,
            clear: ClearMode::Full,
            projection_fix: None,
            pane: None,
        }
    }
}

/// three.js's `renderer.shadowMap.autoUpdate` / `needsUpdate`.
///
/// Shadow maps depend on the lights and the casters, not on the camera — in
/// this renderer a directional light's map is a fixed cube on the origin
/// ([`ShadowSettings::camera_size`](crate::lights::ShadowSettings)), a spot
/// light's is its own frustum and a point light's its cube — so
/// several renders of one unchanged scene from different cameras can share
/// them. With `auto_update` on (the default) a render redraws them when
/// anything they show changed since they were drawn: a caster's geometry,
/// transform, side or cut-out, a skinned caster's pose, an instanced
/// caster's instance data, a custom shader's uniform data, storage or
/// source, or a light's shadow projection. Only a custom-shader caster with
/// textures, which can be drawn into without any of that changing, redraws
/// them every render. With `auto_update` off, a render redraws
/// them only when `needs_update` is set, and clears the flag, as three.js
/// does; otherwise it samples what the last update left.
///
/// A render also redraws them, whatever the flags, when the maps cannot hold
/// what this render needs: the first render with shadows, a directional map
/// reallocated to a new size, or a different set of casting light kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShadowMap {
    pub auto_update: bool,
    pub needs_update: bool,
}

impl Default for ShadowMap {
    fn default() -> Self {
        Self {
            auto_update: true,
            needs_update: false,
        }
    }
}

impl ShadowMap {
    /// Whether this render redraws the maps, given whether the maps already
    /// hold what it needs (`valid`). Consumes `needs_update`.
    /// Whether this render redraws the maps. `valid`: they were last drawn
    /// for the same caster kinds and size. `unchanged`: nothing they depend
    /// on — casters, their geometry and transforms, the lights' projections
    /// — differs from then, so a redraw would produce the same maps.
    pub(crate) fn take_update(&mut self, valid: bool, unchanged: bool) -> bool {
        if !valid {
            self.needs_update = false;
            return true;
        }
        if self.auto_update {
            self.needs_update = false;
            return !unchanged;
        }
        std::mem::take(&mut self.needs_update)
    }
}

/// One render's resolved viewport state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ViewportPlan {
    /// Nothing differs from a plain full-target render. Every path that cannot
    /// work per pane (MSAA with a partial clear, TAA, screen-space refraction)
    /// keys off this, so the default state is byte-for-byte the old renderer.
    pub legacy: bool,
    /// The viewport, clipped to the target.
    pub viewport: PixelRect,
    /// Scissor for every pass that writes the target: the scissor rect when
    /// the test is on, else the clipped viewport (harmless — nothing draws
    /// outside the viewport anyway — and it confines full-screen resolves).
    pub scissor: PixelRect,
    pub clear: ClearMode,
    /// Left-multiplied onto the projection when the viewport hangs off the
    /// target: wgpu requires the viewport inside the attachment, so the
    /// visible part becomes the viewport and the projection is rescaled to
    /// keep the picture where an unclipped viewport would have put it.
    pub projection_fix: Option<Matrix4>,
    /// For a partial clear ([`ClearMode::Rect`]): how to draw the pane into a
    /// target of its own, for multisampling. `None` otherwise, and when the
    /// viewport misses the scissor (nothing but the clear to draw).
    pub pane: Option<PanePlan>,
}

impl ViewportPlan {
    pub fn full(w: u32, h: u32) -> Self {
        let r = PixelRect {
            x: 0,
            y: 0,
            width: w,
            height: h,
        };
        Self {
            legacy: true,
            viewport: r,
            scissor: r,
            clear: ClearMode::Full,
            projection_fix: None,
            pane: None,
        }
    }
}

/// The three.js viewport state itself — `setViewport`, `setScissor`,
/// `setScissorTest`, `setPixelRatio` and `autoClear` — shared by the wgpu
/// [`Renderer`](super::Renderer) and the Metal renderer, which differ only in
/// what they cache beside it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ViewportSettings {
    pub viewport: Option<ViewportRect>,
    pub scissor: Option<ViewportRect>,
    pub scissor_test: bool,
    pixel_ratio: f32,
    pub auto_clear: bool,
}

impl Default for ViewportSettings {
    fn default() -> Self {
        Self {
            viewport: None,
            scissor: None,
            scissor_test: false,
            pixel_ratio: 1.0,
            auto_clear: true,
        }
    }
}

impl ViewportSettings {
    pub fn pixel_ratio(&self) -> f32 {
        self.pixel_ratio
    }

    /// Anything but a finite positive ratio is taken as 1.
    pub fn set_pixel_ratio(&mut self, ratio: f32) {
        self.pixel_ratio = if ratio.is_finite() && ratio > 0.0 {
            ratio
        } else {
            1.0
        };
    }

    /// Resolve for a `w`×`h` device-pixel target. `None` when nothing would be
    /// drawn: the viewport or the scissor misses the target entirely.
    pub fn plan(&self, w: u32, h: u32) -> Option<ViewportPlan> {
        plan(
            self.viewport,
            self.scissor,
            self.scissor_test,
            self.auto_clear,
            self.pixel_ratio,
            w,
            h,
        )
    }

    /// What a bare `clear` covers: no viewport involved, and a clear always
    /// clears (auto-clear governs `render`, not an explicit clear).
    pub fn clear_mode(&self, w: u32, h: u32) -> Option<ClearMode> {
        plan(
            None,
            self.scissor,
            self.scissor_test,
            true,
            self.pixel_ratio,
            w,
            h,
        )
        .map(|p| p.clear)
    }
}

impl ViewportPlan {
    /// Set the pass's viewport and scissor to this plan's. Nothing for a
    /// legacy plan, whose pass keeps wgpu's whole-target defaults.
    pub fn apply(&self, pass: &mut wgpu::RenderPass<'_>) {
        if self.legacy {
            return;
        }
        let (v, sc) = (self.viewport, self.scissor);
        pass.set_viewport(
            v.x as f32,
            v.y as f32,
            v.width as f32,
            v.height as f32,
            0.0,
            1.0,
        );
        pass.set_scissor_rect(sc.x, sc.y, sc.width, sc.height);
    }
}

/// The renderer's viewport settings plus the little pipeline cache the
/// partial clear needs.
#[derive(Default)]
pub(crate) struct ViewportState {
    pub settings: ViewportSettings,
    /// `(format, with_depth)` → clear-colour pipeline.
    clear_pipelines: HashMap<(wgpu::TextureFormat, bool), wgpu::RenderPipeline>,
    pub warned_screen_space: bool,
    /// `(width, height, format, samples)` → a pane's multisampled target.
    pane_targets: HashMap<PaneKey, PaneTarget>,
    composite_bgl: Option<wgpu::BindGroupLayout>,
    /// Colour format → the pane composite pipeline.
    composite_pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
}

pub(crate) type PaneKey = (u32, u32, wgpu::TextureFormat, u32);

/// Distinct pane sizes kept at once. A quad view uses two or three (odd
/// remainders go to one column and one row); a window being dragged to a new
/// size makes a new one every frame, and those are dropped wholesale.
const MAX_PANE_TARGETS: usize = 8;

/// One pane size's multisampled colour and depth, the single-sampled
/// texture they resolve into, and the bind group the composite reads it by.
pub(crate) struct PaneTarget {
    pub msaa_color: wgpu::TextureView,
    /// `msaa_color` as its `Unorm` twin, when it is sRGB and the device can
    /// reinterpret views: the encoded-blend pass draws into it.
    pub msaa_color_unorm: Option<wgpu::TextureView>,
    pub msaa_depth: wgpu::TextureView,
    pub resolve: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
}

impl ViewportState {
    /// The fill pipeline for this attachment format, built on first use.
    pub fn ensure_clear_pipeline(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        depth: Option<wgpu::TextureFormat>,
    ) {
        self.clear_pipelines
            .entry((format, depth.is_some()))
            .or_insert_with(|| build_clear_pipeline(device, format, depth));
    }

    /// Write `color` over `rect` of the pass's colour attachment and nothing
    /// else — the partial clear. Leaves the scissor set to `rect`. A no-op
    /// until [`ensure_clear_pipeline`](Self::ensure_clear_pipeline) has built
    /// the pipeline for `(format, depth)`.
    pub fn fill_rect(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        format: wgpu::TextureFormat,
        depth: bool,
        rect: PixelRect,
        color: wgpu::Color,
    ) {
        if let Some(pipe) = self.clear_pipelines.get(&(format, depth)) {
            pass.set_scissor_rect(rect.x, rect.y, rect.width, rect.height);
            pass.set_pipeline(pipe);
            pass.set_blend_constant(color);
            pass.draw(0..3, 0..1);
        }
    }
}

impl ViewportState {
    /// The pane target for `key`, and the composite pipeline for its format,
    /// built on first use.
    pub fn ensure_pane_target(
        &mut self,
        device: &wgpu::Device,
        key: PaneKey,
        depth_format: wgpu::TextureFormat,
    ) {
        let (w, h, format, samples) = key;
        let bgl = self.composite_bgl.get_or_insert_with(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("threers pane composite bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            })
        });
        self.composite_pipelines
            .entry(format)
            .or_insert_with(|| build_composite_pipeline(device, bgl, format));
        if self.pane_targets.contains_key(&key) {
            return;
        }
        if self.pane_targets.len() >= MAX_PANE_TARGETS {
            self.pane_targets.clear();
        }
        let tex = |label, format, samples, usage| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: w.max(1),
                        height: h.max(1),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: samples,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let unorm = format.remove_srgb_suffix();
        let reinterpret = unorm != format && super::caps::supports_view_formats(device);
        let unorm_formats = [unorm];
        let msaa_color_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers pane msaa color"),
            size: wgpu::Extent3d {
                width: w.max(1),
                height: h.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: if reinterpret { &unorm_formats } else { &[] },
        });
        let msaa_color = msaa_color_tex.create_view(&Default::default());
        let msaa_color_unorm = reinterpret.then(|| {
            msaa_color_tex.create_view(&wgpu::TextureViewDescriptor {
                format: Some(unorm),
                ..Default::default()
            })
        });
        let msaa_depth = tex(
            "threers pane msaa depth",
            depth_format,
            samples,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let resolve = tex(
            "threers pane resolve",
            format,
            1,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers pane composite bg"),
            layout: bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&resolve),
            }],
        });
        self.pane_targets.insert(
            key,
            PaneTarget {
                msaa_color,
                msaa_color_unorm,
                msaa_depth,
                resolve,
                bind_group,
            },
        );
    }

    /// The target [`ensure_pane_target`](Self::ensure_pane_target) built.
    pub fn pane_target(&self, key: &PaneKey) -> Option<&PaneTarget> {
        self.pane_targets.get(key)
    }

    /// Copy the resolved pane `key` into `region` of `target` and nothing
    /// else. A draw rather than a texture copy: a window's surface texture is
    /// usually a render attachment only, and a draw also copes with a target
    /// whose view format differs in sRGB-ness. It is exact either way —
    /// `textureLoad` with no filtering, and an sRGB decode followed by an
    /// encode gives back the byte it started from.
    pub fn composite_pane(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        format: wgpu::TextureFormat,
        key: &PaneKey,
        region: PixelRect,
    ) {
        let (Some(pane), Some(pipe)) = (
            self.pane_targets.get(key),
            self.composite_pipelines.get(&format),
        ) else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("threers pane composite"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let r = region;
        pass.set_viewport(
            r.x as f32,
            r.y as f32,
            r.width as f32,
            r.height as f32,
            0.0,
            1.0,
        );
        pass.set_scissor_rect(r.x, r.y, r.width, r.height);
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, &pane.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// The fullscreen triangle over the pane's region, each pixel loading its
/// texel of the resolved pane. `uv` runs 0..1 across the viewport with the
/// texture's origin at the top left, so a pixel centre `(i + ½)/w` floors back
/// to `i`.
fn composite_wgsl() -> String {
    format!(
        "{}\n{}",
        super::fullscreen::VS,
        r#"
@group(0) @binding(0) var src : texture_2d<f32>;
@fragment
fn fs_main(v : FullscreenVertex) -> @location(0) vec4<f32> {
    let dims = vec2<i32>(textureDimensions(src));
    let p = clamp(vec2<i32>(floor(v.uv * vec2<f32>(dims))), vec2<i32>(0, 0), dims - vec2<i32>(1, 1));
    return textureLoad(src, p, 0);
}
"#
    )
}

fn build_composite_pipeline(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("threers pane composite"),
        source: wgpu::ShaderSource::Wgsl(composite_wgsl().into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("threers pane composite layout"),
        bind_group_layouts: &[Some(bgl)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("threers pane composite"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some(super::fullscreen::VS_ENTRY),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn plan(
    viewport: Option<ViewportRect>,
    scissor: Option<ViewportRect>,
    scissor_test: bool,
    auto_clear: bool,
    pixel_ratio: f32,
    w: u32,
    h: u32,
) -> Option<ViewportPlan> {
    let full = IRect::full(w, h);
    let pr = if pixel_ratio > 0.0 { pixel_ratio } else { 1.0 };
    let vp = viewport.map(|v| v.to_device(pr)).unwrap_or(full);
    let vp_clip = vp.intersect(full);
    if vp_clip.is_empty() {
        return None;
    }
    let sc = if scissor_test {
        let s = scissor
            .map(|s| s.to_device(pr))
            .unwrap_or(full)
            .intersect(full);
        if s.is_empty() {
            return None;
        }
        Some(s)
    } else {
        None
    };
    let clear = if !auto_clear {
        ClearMode::None
    } else {
        match sc {
            Some(s) if s != full => ClearMode::Rect(s.to_top_left(h)),
            _ => ClearMode::Full,
        }
    };
    let legacy = vp == full && clear == ClearMode::Full;
    let projection_fix = (vp != vp_clip).then(|| clip_fix(vp, vp_clip));
    let pane = match (clear, sc) {
        (ClearMode::Rect(_), Some(region)) => pane_plan(vp, region, h),
        _ => None,
    };
    Some(ViewportPlan {
        legacy,
        viewport: vp_clip.to_top_left(h),
        scissor: sc.unwrap_or(vp_clip).to_top_left(h),
        clear,
        projection_fix,
        pane,
    })
}

/// The viewport `vp` (device pixels, bottom-left, unclipped) as seen from a
/// target that is just `region` of an `h`-tall one. Everything a render may
/// write — the clear and the draws, which the scissor confines to `region` —
/// lies inside it, so drawing there and copying `region` back is the same
/// picture.
fn pane_plan(vp: IRect, region: IRect, h: u32) -> Option<PanePlan> {
    let local = IRect {
        x: vp.x - region.x,
        y: vp.y - region.y,
        w: vp.w,
        h: vp.h,
    };
    let clip = local.intersect(IRect {
        x: 0,
        y: 0,
        w: region.w,
        h: region.h,
    });
    if clip.is_empty() {
        return None;
    }
    Some(PanePlan {
        region: region.to_top_left(h),
        viewport: clip.to_top_left(region.h as u32),
        projection_fix: (local != clip).then(|| clip_fix(local, clip)),
    })
}

/// The clip-space remap that draws what viewport `v` would show into the
/// smaller viewport `c` it was clipped to. Per axis, with both rects in the
/// same (bottom-left, so NDC-parallel) frame:
/// `ndc' = ndc · v/c + (2(v₀ − c₀) + v − c) / c`.
fn clip_fix(v: IRect, c: IRect) -> Matrix4 {
    let (vx, vy, vw, vh) = (v.x as f32, v.y as f32, v.w as f32, v.h as f32);
    let (cx, cy, cw, ch) = (c.x as f32, c.y as f32, c.w as f32, c.h as f32);
    let mut m = Matrix4::identity();
    m.elements[0] = vw / cw;
    m.elements[12] = (2.0 * (vx - cx) + vw - cw) / cw;
    m.elements[5] = vh / ch;
    m.elements[13] = (2.0 * (vy - cy) + vh - ch) / ch;
    m
}

const CLEAR_WGSL: &str = r#"
@vertex
fn vs(@builtin(vertex_index) i : u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}
@fragment
fn fs() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
"#;

/// A full-screen triangle whose colour is the *blend constant*: the fragment
/// writes one, the blend multiplies by the constant and discards the
/// destination. Scissored, that is the partial clear WebGPU has no op for, and
/// it needs no bind group or uniform buffer — `set_blend_constant` carries the
/// colour, exactly as `LoadOp::Clear` would have written it.
fn build_clear_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    depth: Option<wgpu::TextureFormat>,
) -> wgpu::RenderPipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("threers viewport clear"),
        source: wgpu::ShaderSource::Wgsl(CLEAR_WGSL.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("threers viewport clear layout"),
        bind_group_layouts: &[],
        immediate_size: 0,
    });
    let constant = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::Constant,
        dst_factor: wgpu::BlendFactor::Zero,
        operation: wgpu::BlendOperation::Add,
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("threers viewport clear"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState {
                    color: constant,
                    alpha: constant,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: depth.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st() -> ViewportSettings {
        ViewportSettings::default()
    }

    #[test]
    fn default_state_is_the_old_renderer() {
        let p = st().plan(640, 480).unwrap();
        assert_eq!(p, ViewportPlan::full(640, 480));
        assert!(p.legacy);
    }

    #[test]
    fn a_full_viewport_set_explicitly_is_still_legacy() {
        let mut s = st();
        s.viewport = Some(ViewportRect::new(0.0, 0.0, 640.0, 480.0));
        assert!(s.plan(640, 480).unwrap().legacy);
        // A full scissor with the test on clears the whole target too.
        s.scissor_test = true;
        s.scissor = Some(ViewportRect::new(0.0, 0.0, 640.0, 480.0));
        assert!(s.plan(640, 480).unwrap().legacy);
    }

    #[test]
    fn origin_is_bottom_left_and_flips_to_top_left() {
        let mut s = st();
        // The lower-left quarter of a 200×100 target...
        s.viewport = Some(ViewportRect::new(0.0, 0.0, 100.0, 50.0));
        let p = s.plan(200, 100).unwrap();
        // ...starts 50 rows down in wgpu's top-left frame.
        assert_eq!(
            p.viewport,
            PixelRect {
                x: 0,
                y: 50,
                width: 100,
                height: 50
            }
        );
        assert!(!p.legacy);
        // Without the scissor test the clear is still the whole target.
        assert_eq!(p.clear, ClearMode::Full);
    }

    #[test]
    fn scissor_test_limits_the_clear() {
        let mut s = st();
        s.scissor_test = true;
        s.scissor = Some(ViewportRect::new(10.0, 20.0, 30.0, 40.0));
        let p = s.plan(100, 100).unwrap();
        let r = PixelRect {
            x: 10,
            y: 40,
            width: 30,
            height: 40,
        };
        assert_eq!(p.clear, ClearMode::Rect(r));
        assert_eq!(p.scissor, r);
        // The test without a rectangle is a full-target scissor.
        s.scissor = None;
        assert_eq!(s.plan(100, 100).unwrap().clear, ClearMode::Full);
    }

    #[test]
    fn auto_clear_off_keeps_everything() {
        let mut s = st();
        s.auto_clear = false;
        let p = s.plan(64, 64).unwrap();
        assert_eq!(p.clear, ClearMode::None);
        assert!(
            !p.legacy,
            "a load-only render cannot use the MSAA/TAA paths"
        );
    }

    #[test]
    fn pixel_ratio_scales_and_rounds() {
        let mut s = st();
        s.set_pixel_ratio(1.5);
        s.viewport = Some(ViewportRect::new(10.0, 10.0, 33.0, 21.0));
        let p = s.plan(300, 300).unwrap();
        // 15, 15, 49.5→50, 31.5→32; top = 300 − 15 − 32.
        assert_eq!(
            p.viewport,
            PixelRect {
                x: 15,
                y: 253,
                width: 50,
                height: 32
            }
        );
    }

    #[test]
    fn a_viewport_off_the_target_is_clipped_and_reprojected() {
        let mut s = st();
        // Hangs 50 px off the left and 25 px off the top of a 200×100 target.
        s.viewport = Some(ViewportRect::new(-50.0, 50.0, 100.0, 75.0));
        let p = s.plan(200, 100).unwrap();
        assert_eq!(
            p.viewport,
            PixelRect {
                x: 0,
                y: 0,
                width: 50,
                height: 50
            }
        );
        let fix = p.projection_fix.expect("clipped viewport needs a fix");
        // A point at NDC (x, y) in the unclipped viewport lands on device pixel
        // (vx + (x+1)/2·vw, vy + (y+1)/2·vh). Map it through the fix and the
        // clipped viewport and it must land on the same pixel.
        for &(nx, ny) in &[(-1.0f32, -1.0f32), (0.0, 0.0), (1.0, 1.0), (0.3, -0.6)] {
            let want_x = -50.0 + (nx + 1.0) * 0.5 * 100.0;
            let want_y = 50.0 + (ny + 1.0) * 0.5 * 75.0;
            let e = &fix.elements;
            let fx = e[0] * nx + e[12];
            let fy = e[5] * ny + e[13];
            let got_x = 0.0 + (fx + 1.0) * 0.5 * 50.0;
            let got_y = 50.0 + (fy + 1.0) * 0.5 * 50.0;
            assert!((got_x - want_x).abs() < 1e-4, "{nx}: {got_x} vs {want_x}");
            assert!((got_y - want_y).abs() < 1e-4, "{ny}: {got_y} vs {want_y}");
        }
        // The fix leaves w (and so perspective division) alone.
        assert_eq!(fix.elements[15], 1.0);
        assert_eq!(fix.elements[3], 0.0);
    }

    #[test]
    fn nothing_visible_plans_nothing() {
        let mut s = st();
        s.viewport = Some(ViewportRect::new(500.0, 0.0, 10.0, 10.0));
        assert!(s.plan(100, 100).is_none());
        s.viewport = None;
        s.scissor_test = true;
        s.scissor = Some(ViewportRect::new(0.0, 0.0, 0.0, 10.0));
        assert!(s.plan(100, 100).is_none());
    }

    #[test]
    fn a_partial_clear_plans_a_pane_target_the_size_of_the_scissor() {
        let mut s = st();
        s.scissor_test = true;
        s.viewport = Some(ViewportRect::new(10.0, 20.0, 30.0, 40.0));
        s.scissor = Some(ViewportRect::new(10.0, 20.0, 30.0, 40.0));
        let p = s.plan(100, 100).unwrap();
        let pane = p.pane.expect("partial clear plans a pane");
        // The region is the clear rectangle; the viewport fills the pane.
        assert_eq!(pane.region, p.scissor);
        assert_eq!(
            pane.viewport,
            PixelRect {
                x: 0,
                y: 0,
                width: 30,
                height: 40
            }
        );
        assert!(pane.projection_fix.is_none());
        let local = pane.local();
        assert_eq!(local.clear, ClearMode::Full);
        assert!(!local.legacy);
        // A full clear or none plans no pane.
        s.scissor_test = false;
        assert!(s.plan(100, 100).unwrap().pane.is_none());
        s.scissor_test = true;
        s.auto_clear = false;
        assert!(s.plan(100, 100).unwrap().pane.is_none());
    }

    #[test]
    fn a_scissor_inside_the_viewport_shifts_the_pane_picture() {
        let mut s = st();
        s.scissor_test = true;
        // Viewport 100×100 at (0,0); scissor its upper-right 40×30.
        s.viewport = Some(ViewportRect::new(0.0, 0.0, 100.0, 100.0));
        s.scissor = Some(ViewportRect::new(60.0, 70.0, 40.0, 30.0));
        let p = s.plan(200, 100).unwrap();
        let pane = p.pane.unwrap();
        assert_eq!(
            pane.region,
            PixelRect {
                x: 60,
                y: 0,
                width: 40,
                height: 30
            }
        );
        let fix = pane.projection_fix.expect("viewport larger than the pane");
        // NDC (x, y) of the full viewport lands on target pixel
        // (x+1)/2·100, (y+1)/2·100 (bottom-left). Through the fix and the pane
        // viewport it must land on the same pixel, offset by the region.
        for &(nx, ny) in &[(0.2f32, 0.4f32), (1.0, 1.0), (0.6, 0.9)] {
            let want = ((nx + 1.0) * 50.0, (ny + 1.0) * 50.0);
            let e = &fix.elements;
            let (fx, fy) = (e[0] * nx + e[12], e[5] * ny + e[13]);
            let v = pane.viewport;
            let got = (
                60.0 + v.x as f32 + (fx + 1.0) * 0.5 * v.width as f32,
                70.0 + (fy + 1.0) * 0.5 * v.height as f32,
            );
            assert!((got.0 - want.0).abs() < 1e-3, "{got:?} vs {want:?}");
            assert!((got.1 - want.1).abs() < 1e-3, "{got:?} vs {want:?}");
        }
        // A viewport that misses the scissor plans no pane: only the clear.
        s.viewport = Some(ViewportRect::new(0.0, 0.0, 50.0, 50.0));
        assert!(s.plan(200, 100).unwrap().pane.is_none());
    }

    #[test]
    fn shadow_map_flags_follow_three_js() {
        let mut m = ShadowMap::default();
        assert!(m.auto_update);
        assert!(m.take_update(true, false), "auto-update redraws what changed");
        assert!(!m.take_update(true, true), "and keeps what did not");
        m.auto_update = false;
        assert!(!m.take_update(true, false), "manual and not asked: reuse");
        m.needs_update = true;
        assert!(m.take_update(true, true), "asked: redrawn, changed or not");
        assert!(!m.needs_update, "the flag is consumed");
        assert!(!m.take_update(true, false));
        assert!(
            m.take_update(false, false),
            "maps that cannot be reused are redrawn"
        );
    }

    #[test]
    fn rect_helpers() {
        let r = ViewportRect::new(10.0, 20.0, 40.0, 20.0);
        assert_eq!(r.aspect(), 2.0);
        assert!(r.contains(10.0, 20.0));
        assert!(!r.contains(50.0, 20.0));
        assert!(!r.contains(10.0, 40.0));
    }
}
