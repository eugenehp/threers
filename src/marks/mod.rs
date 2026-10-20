//! Data-driven marks: points, lines, arcs, instanced shapes, text labels and
//! density fields, drawn from columns of per-instance data.
//!
//! The scene graph is the right model for a few hundred objects, each with a
//! material and a transform. It is the wrong one for a million rows of a
//! table, where every row is drawn the same way and differs only in its data.
//! A *mark* is that second model: one draw for the whole layer, with each
//! property — position, radius, colour — read from its own
//! [`Column`] of per-instance data.
//!
//! | | |
//! |---|---|
//! | [`PointMarks`] | Circles sized in pixels or world units, with strokes |
//! | [`SegmentMarks`] | Wide lines, polylines and arcs, in pixels or world units |
//! | [`ShapeMarks`] | Any geometry, instanced — extruded columns, 3D glyphs |
//! | [`PolygonMarks`] | Filled and extruded polygons with holes, many to a draw |
//! | [`TextLabels`] | Camera-facing text anchored in the world, with collision culling (`captions` feature) |
//! | [`DensityMarks`] | Kernels summed into a field and mapped through a colour ramp |
//!
//! Every mark kind shares the same machinery:
//!
//! - **High precision.** Positions are `f64` and drawn relative to an
//!   [`Origin`] — see [`precision`].
//! - **Hooks.** [`ShaderHooks`] splice WGSL into the vertex and colour stages,
//!   which is how filtering on the GPU works ([`ShaderHooks::range_filter`]).
//! - **Picking.** [`MarkRenderer::pick`] draws instance ids and reads back the
//!   pixel under the cursor, without blocking a browser's thread.
//! - **WebGL2.** Instanced vertex buffers and render passes only; nothing
//!   needs compute or storage buffers.
//!
//! Marks draw *over* a frame the [`Renderer`](crate::Renderer) has already
//! drawn, depth-tested against it, so they sit correctly among meshes:
//!
//! ```no_run
//! use threers::marks::{MarkRenderer, MarkTarget, MarkView, Origin, PointMarks, Mark};
//! # fn frame(renderer: &mut threers::Renderer, scene: &mut threers::Scene,
//! #          camera: &threers::PerspectiveCamera, frame_view: &wgpu::TextureView) {
//! let device = renderer.device_arc();
//! let queue = renderer.queue_arc();
//! let mut marks = MarkRenderer::new(&device);
//! let mut points = PointMarks::new();
//! points.set_positions(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
//! points.set_colors(&[[255, 80, 0, 255], [0, 160, 255, 255]]);
//! points.style_mut().size_scale = 6.0;
//!
//! renderer.render(scene, camera, frame_view, false);
//! let view = MarkView::new(camera, Origin::ZERO, renderer.drawing_buffer_size());
//! let target = MarkTarget::over_frame(renderer, frame_view);
//! marks.draw(&device, &queue, &target, &view, &mut [&mut points]);
//! # }
//! ```

mod atlas;
mod collision;
mod columns;
mod density;
pub mod geo;
pub mod geojson;
mod hooks;
mod images;
#[cfg(feature = "captions")]
mod labels;
mod map_view;
#[cfg(feature = "captions")]
mod msdf;
mod picking;
mod points;
mod polygon_mesh;
mod polygons;
pub mod precision;
mod segments;
pub(crate) mod shader;
mod shapes;
mod sprites;

pub use atlas::{AtlasError, AtlasRect, SharedAtlas, TextureAtlas};
pub use collision::{CollisionIndex, SharedCollision};
pub use columns::{Column, ColumnFormat};
pub use density::DensityMarks;
pub use hooks::{transition_offsets, ShaderHooks};
pub use images::{ImageMarks, ImageQuad};
#[cfg(feature = "captions")]
pub use labels::{GlyphField, LabelAnchor, LabelIcon, TextLabel, TextLabels};
pub use map_view::{MapFrame, MapView};
pub use picking::{PickHit, PickRectRequest, PickRequest};
pub use points::PointMarks;
pub use polygon_mesh::PolygonMesh;
pub use polygons::{PolygonFeature, PolygonMarks};
pub use precision::{split_f64, split_f64x3, Origin};
pub use segments::{LineCap, PathMesh, SegmentMarks};
pub use shader::{CustomShader, MarkKind, PREAMBLE};
pub use shapes::ShapeMarks;
pub use sprites::{SpriteMarks, SpriteMode};

use std::collections::HashMap;

/// The depth buffer format marks test against — the renderer's.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// How the camera sees the marks: the projection, the viewport and the origin.
#[derive(Debug, Clone, Copy)]
pub struct MarkView {
    /// Projection × view, in origin-relative coordinates.
    pub view_proj: [f32; 16],
    /// The projection's `[1][1]` — what turns world size into pixels.
    pub projection_scale: f32,
    /// Viewport size in pixels.
    pub viewport: (u32, u32),
    pub origin: Origin,
    /// Direction *towards* the light, for [`ShapeMarks`]. Normalised on use.
    pub light_direction: [f32; 3],
    /// The camera, origin-relative: its position with `w = 1` for a
    /// perspective view, or the direction towards it with `w = 0` for an
    /// orthographic one. Used by [`MarkStyle::depth_lift`].
    pub eye: [f32; 4],
    /// The scene's tone mapping and exposure, for layers with
    /// [`MarkStyle::tone_mapped`] set. [`from_renderer`](Self::from_renderer)
    /// copies the renderer's.
    pub tone_mapping: (crate::ToneMapping, f32),
    /// The depth buffer is reversed
    /// ([`Renderer::set_reversed_z`](crate::Renderer::set_reversed_z)):
    /// `view_proj` maps the near plane to 1, and marks test with `Greater`.
    /// [`from_renderer`](Self::from_renderer) sets both.
    pub reversed_z: bool,
}

impl MarkView {
    /// The view through `camera`, whose pose is in coordinates relative to
    /// `origin`.
    pub fn new(camera: &dyn crate::Camera, origin: Origin, viewport: (u32, u32)) -> Self {
        let proj = camera.projection_matrix();
        let view = camera.view_matrix();
        let vp = proj.multiply(&view);
        // The camera's world transform is the view's inverse: its translation
        // is the eye, its +Z the direction back towards the camera.
        let world = view.invert().elements;
        let eye = if proj.elements[15].abs() > 0.5 {
            [world[8], world[9], world[10], 0.0]
        } else {
            [world[12], world[13], world[14], 1.0]
        };
        Self {
            view_proj: vp.elements,
            projection_scale: proj.elements[5],
            viewport: (viewport.0.max(1), viewport.1.max(1)),
            origin,
            light_direction: [0.4, 0.8, 0.45],
            eye,
            tone_mapping: (crate::ToneMapping::None, 1.0),
            reversed_z: false,
        }
    }

    /// The view through `camera` at `renderer`'s drawing size and with its
    /// tone mapping — what marks drawn over that renderer's frames want.
    pub fn from_renderer(
        camera: &dyn crate::Camera,
        origin: Origin,
        renderer: &crate::Renderer,
    ) -> Self {
        let mut v = Self::new(camera, origin, renderer.drawing_buffer_size());
        v.tone_mapping = renderer.tone_mapping();
        if renderer.reversed_z() {
            v = v.with_reversed_z();
        }
        v
    }

    /// Remap to reversed depth, as a renderer with
    /// [`set_reversed_z`](crate::Renderer::set_reversed_z) draws: clip
    /// `z' = w - z`.
    pub fn with_reversed_z(mut self) -> Self {
        if !self.reversed_z {
            for c in 0..4 {
                self.view_proj[c * 4 + 2] = self.view_proj[c * 4 + 3] - self.view_proj[c * 4 + 2];
            }
            self.reversed_z = true;
        }
        self
    }

    /// Set [`tone_mapping`](Self::tone_mapping).
    pub fn with_tone_mapping(mut self, mode: crate::ToneMapping, exposure: f32) -> Self {
        self.tone_mapping = (mode, exposure);
        self
    }

    /// World units one pixel spans at absolute point `p` — what picks a
    /// tile zoom or a level of detail there. `None` behind the camera.
    pub fn units_per_pixel(&self, p: [f64; 3]) -> Option<f64> {
        let w = self.project(p)?[2] as f64;
        let px_per_unit = self.projection_scale as f64 * self.viewport.1 as f64 * 0.5;
        Some(w / px_per_unit.max(1e-12))
    }

    /// An absolute position on screen: `(x, y)` pixels from the top left and
    /// clip `w`, or `None` behind the camera.
    pub fn project(&self, p: [f64; 3]) -> Option<[f32; 3]> {
        let l = self.origin.to_local(p);
        let m = &self.view_proj;
        let x = m[0] * l.x + m[4] * l.y + m[8] * l.z + m[12];
        let y = m[1] * l.x + m[5] * l.y + m[9] * l.z + m[13];
        let w = m[3] * l.x + m[7] * l.y + m[11] * l.z + m[15];
        if w <= 1e-9 {
            return None;
        }
        let (vw, vh) = (self.viewport.0 as f32, self.viewport.1 as f32);
        Some([(x / w * 0.5 + 0.5) * vw, (0.5 - y / w * 0.5) * vh, w])
    }

    fn uniforms(&self, encode_srgb: bool, samples: u32) -> FrameUniforms {
        let (w, h) = (self.viewport.0 as f32, self.viewport.1 as f32);
        let (ohi, olo) = self.origin.split();
        let l = self.light_direction;
        let len = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt().max(1e-9);
        FrameUniforms {
            view_proj: self.view_proj,
            origin_hi: ohi,
            origin_lo: olo,
            viewport: [w, h, 1.0 / w, 1.0 / h],
            misc: [
                self.projection_scale * h * 0.5,
                if encode_srgb { 1.0 } else { 0.0 },
                self.tone_mapping.0.shader_mode(),
                self.tone_mapping.1,
            ],
            // w: reversed depth, which `mark_order` pushes the other way.
            light: [
                l[0] / len,
                l[1] / len,
                l[2] / len,
                if self.reversed_z { 1.0 } else { 0.0 },
            ],
            eye: self.eye,
            aa: [if samples > 1 { 1.0 } else { 0.5 }, 0.0, 0.0, 0.0],
        }
    }
}

/// Where marks are drawn: a colour target, and optionally the depth buffer of
/// the frame already in it.
pub struct MarkTarget<'a> {
    /// The view drawn into — the multisampled colour when `samples > 1`.
    pub color: &'a wgpu::TextureView,
    /// Where a multisampled `color` resolves to.
    pub resolve: Option<&'a wgpu::TextureView>,
    /// The frame's depth, [`DEPTH_FORMAT`], with `samples` samples.
    pub depth: Option<&'a wgpu::TextureView>,
    pub format: wgpu::TextureFormat,
    pub samples: u32,
}

impl<'a> MarkTarget<'a> {
    /// A plain colour target with no depth: marks draw over everything.
    pub fn new(color: &'a wgpu::TextureView, format: wgpu::TextureFormat) -> Self {
        Self {
            color,
            resolve: None,
            depth: None,
            format,
            samples: 1,
        }
    }

    /// Depth-test against `depth` ([`DEPTH_FORMAT`], same size and sample count).
    pub fn with_depth(mut self, depth: &'a wgpu::TextureView) -> Self {
        self.depth = Some(depth);
        self
    }

    /// A [`RenderTarget`](crate::RenderTarget) that
    /// [`Renderer::render_to`](crate::Renderer::render_to) just drew into.
    pub fn render_target(rt: &'a crate::RenderTarget) -> Self {
        Self::new(&rt.color_view, rt.format).with_depth(&rt.depth_view)
    }

    /// The frame [`Renderer::render`](crate::Renderer::render) just drew into
    /// `frame`, with that frame's depth — multisampled if it was.
    pub fn over_frame(renderer: &'a crate::Renderer, frame: &'a wgpu::TextureView) -> Self {
        let (msaa_color, depth, samples) = renderer.overlay_attachments();
        match msaa_color {
            Some(color) => Self {
                color,
                resolve: Some(frame),
                depth: Some(depth),
                format: renderer.color_format(),
                samples,
            },
            None => Self::new(frame, renderer.color_format()).with_depth(depth),
        }
    }
}

/// What a size is measured in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SizeUnits {
    /// Screen pixels: the same size at any distance.
    #[default]
    Pixels,
    /// World units: smaller with distance, like geometry.
    World,
}

/// How a mark's colour combines with what is under it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MarkBlend {
    /// Ordinary straight-alpha "over".
    #[default]
    Alpha,
    /// Colours add — glows, and dense layers that should brighten where they
    /// overlap.
    Additive,
}

/// Settings every mark has.
#[derive(Debug, Clone, PartialEq)]
pub struct MarkStyle {
    /// Multiplies every instance's size (radius, width, scale).
    pub size_scale: f32,
    pub units: SizeUnits,
    /// Pixel clamps on the drawn size, whatever the units.
    pub min_pixels: f32,
    pub max_pixels: f32,
    pub opacity: f32,
    /// Four `vec4`s the hooks read as `layer.params` — filter ranges, time.
    pub params: [[f32; 4]; 4],
    /// Hide behind the frame's geometry. Needs a target with depth.
    pub depth_test: bool,
    /// Write depth, so later marks and passes are hidden behind this one.
    pub depth_write: bool,
    /// How far towards the camera a mark drawn facing it (a point, a line, a
    /// label) is depth-tested, as a multiple of its drawn size.
    ///
    /// At 0 the whole sprite takes its anchor's depth, so a point resting on
    /// the ground loses the part of its disc that lies over nearer ground,
    /// and a line along the ground fights it for every pixel. Seen from an
    /// elevation θ, the ground under a disc's lower edge is `1 / tan θ`
    /// radii nearer than its centre, so that is the lift a mark needs to draw
    /// whole on a level surface: 1 for views steeper than 45°, 2 (the
    /// default) down to 27°, 4 down to 14°. The price is that geometry less
    /// than that far in front of a mark no longer hides it. Ignored by
    /// shapes, which are real geometry.
    pub depth_lift: f32,
    /// Draw order among layers lying in the same plane: a higher order wins
    /// the depth test over a lower one at the same depth, so water, land
    /// use and roads on the ground stack instead of fighting. Each step is
    /// 10⁻⁵ of the depth range, enough to separate coplanar layers and too
    /// little to pull a layer through anything above it.
    pub depth_order: i32,
    /// Draw only inside this ground rectangle, `[min_x, min_y, max_x,
    /// max_y]` in the ground convention of [`geo`] — a tile's bounds.
    /// Fills, lines and images are cut at its edge; points, sprites, shapes
    /// and labels are kept or dropped by their anchor, which also drops the
    /// copies of a point that vector tiles repeat in their buffer. Flat
    /// ground only.
    pub clip: Option<[f64; 4]>,
    /// One row drawn in [`highlight_color`](Self::highlight_color) — the
    /// hovered or selected one. A uniform: changing it rewrites no column.
    pub highlight: Option<u32>,
    /// sRGB; alpha is how far the row's colour moves towards it.
    pub highlight_color: [u8; 4],
    /// A planet as `(centre, radius)`: rows on its far side from the eye
    /// are not drawn — labels, icons and points that do not depth-test
    /// against the globe would otherwise show through it.
    pub horizon: Option<([f64; 3], f64)>,
    /// Run the layer's colours through the scene's tone mapping
    /// ([`MarkView::tone_mapping`]), so marks sit in a lit, exposed scene
    /// the way its materials do. Off by default: data colours then reach the
    /// screen exactly, and a legend matches them.
    pub tone_mapped: bool,
    pub blend: MarkBlend,
    pub visible: bool,
    pub pickable: bool,
}

impl Default for MarkStyle {
    fn default() -> Self {
        Self {
            size_scale: 1.0,
            units: SizeUnits::Pixels,
            min_pixels: 0.0,
            max_pixels: f32::MAX,
            opacity: 1.0,
            params: [[0.0; 4]; 4],
            depth_test: true,
            depth_write: false,
            depth_lift: 2.0,
            depth_order: 0,
            clip: None,
            highlight: None,
            highlight_color: [255, 220, 0, 255],
            horizon: None,
            tone_mapped: false,
            blend: MarkBlend::Alpha,
            visible: true,
            pickable: true,
        }
    }
}

/// The parts of a mark that are the same for every kind.
#[derive(Debug, Default)]
pub struct MarkCommon {
    pub style: MarkStyle,
    pub hooks: ShaderHooks,
    /// Set when the layer's positions are
    /// [`PositionLocal`](ColumnFormat::PositionLocal): the absolute point
    /// they are relative to. The `set_local_*` methods of each mark set it
    /// with the columns; set it yourself only for a mark of your own.
    pub local_origin: Option<[f64; 3]>,
    /// A stable id per row, reported by picks as [`PickHit::feature`] — the
    /// feature a tile's row came from, which another tile's row of the same
    /// feature shares. Shared, so a pick takes a snapshot for free.
    pub feature_ids: Option<std::sync::Arc<[u64]>>,
    custom: [Option<Column>; 2],
    pick_id: u32,
    /// Where this layer's uniforms sit in the renderer's arena, set by each
    /// prepare.
    layer_offset: u32,
}

impl MarkCommon {
    /// Custom column `slot` (0 or 1): up to four floats a row, read by hooks as
    /// `custom0` / `custom1`.
    pub fn set_custom(&mut self, slot: usize, components: u32, values: &[f32]) {
        assert!(slot < 2, "custom slot {slot} is not 0 or 1");
        assert!((1..=4).contains(&components), "1..=4 floats a row");
        let col = self.custom[slot].get_or_insert_with(|| Column::float(components));
        if col.format() != ColumnFormat::Float(components) {
            *col = Column::float(components);
        }
        col.set_f32(values);
    }

    /// Overwrite rows of a custom column from `start`.
    pub fn update_custom(&mut self, slot: usize, start: usize, values: &[f32]) {
        self.custom[slot]
            .as_mut()
            .expect("set_custom first")
            .update_f32(start, values);
    }

    /// Set [`feature_ids`](Self::feature_ids).
    pub fn set_feature_ids(&mut self, ids: Vec<u64>) {
        self.feature_ids = Some(ids.into());
    }

    pub fn clear_custom(&mut self, slot: usize) {
        self.custom[slot] = None;
    }

    /// Drop the custom columns' GPU copies.
    pub fn release_gpu(&mut self) {
        for c in self.custom.iter_mut().flatten() {
            c.release();
        }
    }

    /// Bytes the custom columns hold on the GPU. (A layer's uniforms live
    /// in the [`MarkRenderer`]'s shared arena.)
    pub fn gpu_bytes(&self) -> u64 {
        self.custom.iter().flatten().map(Column::gpu_bytes).sum()
    }

    pub fn custom(&self, slot: usize) -> Option<&Column> {
        self.custom[slot].as_ref()
    }

    fn custom_widths(&self) -> [u32; 2] {
        let w = |c: &Option<Column>| match c.as_ref().map(|c| c.format()) {
            Some(ColumnFormat::Float(n)) => n,
            _ => 0,
        };
        [w(&self.custom[0]), w(&self.custom[1])]
    }
}

/// Instanced geometry: a vertex buffer of position + normal, its vertex
/// count, and an optional `u32` index buffer with its count.
pub type InstancedGeometry<'a> = (&'a wgpu::Buffer, u32, Option<(&'a wgpu::Buffer, u32)>);

/// Per-kind uniform values beyond [`MarkStyle`]: `(style2, style3, stroke)`.
/// `style2.w` is always the opacity; the renderer fills it in.
pub type KindUniforms = ([f32; 4], [f32; 4], [f32; 4]);

/// What [`Mark::prepare`] gets to upload with.
pub struct PrepareContext<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub view: &'a MarkView,
    pub(crate) resources: &'a SharedResources,
}

impl PrepareContext<'_> {
    /// The renderer's clamp-to-edge, linearly filtered sampler, for a custom
    /// mark's bind group.
    pub fn linear_sampler(&self) -> &wgpu::Sampler {
        &self.resources.linear_sampler
    }
}

/// One layer of marks: the renderer's interface to it.
///
/// The mark types in this module implement it, and so can a crate of your
/// own: report a [`MarkKind::Custom`] from a [`CustomShader`], hand its
/// source back from [`custom_shader`](Self::custom_shader), list the columns
/// its vertex input reads, and — for a kind that samples textures — give
/// [`extra_layout`](Self::extra_layout) and
/// [`extra_bind_group`](Self::extra_bind_group). Uniforms, hooks, custom
/// columns, picking, depth and blending then work as they do for the
/// built-in kinds.
pub trait Mark {
    fn kind(&self) -> MarkKind;
    fn common(&self) -> &MarkCommon;
    fn common_mut(&mut self) -> &mut MarkCommon;
    /// Instances to draw.
    fn instance_count(&self) -> usize;
    /// Rows the columns are uploaded for: instances, or vertices for a mark
    /// whose columns are [`per_vertex`](Self::per_vertex).
    fn rows(&self) -> usize {
        self.instance_count()
    }
    /// Columns, custom ones included, advance per vertex rather than per
    /// instance.
    fn per_vertex(&self) -> bool {
        false
    }
    /// An index buffer for a mark with no instanced geometry, and its count.
    fn index_buffer(&self) -> Option<(&wgpu::Buffer, u32)> {
        None
    }

    /// Shorthand for `common_mut().style`.
    fn style_mut(&mut self) -> &mut MarkStyle {
        &mut self.common_mut().style
    }

    /// Vertices drawn per instance when there is no
    /// [`geometry`](Self::geometry): 6 for a quad.
    fn vertices_per_instance(&self) -> u32 {
        6
    }
    /// The kind's own uniforms, read as `layer.style2`, `layer.style3` and
    /// `layer.stroke`.
    fn kind_uniforms(&self) -> KindUniforms {
        ([0.0; 4], [0.0; 4], [0.0; 4])
    }
    /// A fourth vector of the kind's own uniforms, read as `layer.extra`.
    fn kind_extra(&self) -> [f32; 4] {
        [0.0; 4]
    }
    /// Bring GPU copies up to date. Column uploads happen in the renderer.
    fn prepare(&mut self, _ctx: &PrepareContext) {}
    /// The kind's own instance columns, in shader location order from 0
    /// (from 2 after [`geometry`](Self::geometry), which takes 0 and 1).
    fn columns(&mut self) -> Vec<&mut Column>;
    /// Instanced geometry: `(vertex buffer of position+normal, vertices,
    /// optional u32 index buffer and count)`.
    fn geometry(&self) -> Option<InstancedGeometry<'_>> {
        None
    }
    /// The source of a [`MarkKind::Custom`] kind.
    fn custom_shader(&self) -> Option<&CustomShader> {
        None
    }
    /// The layout of bind group 2, for a custom kind that binds one.
    fn extra_layout(&self) -> Option<&wgpu::BindGroupLayout> {
        None
    }
    /// Bind group 2, for kinds that sample a texture.
    fn extra_bind_group(&self) -> Option<&wgpu::BindGroup> {
        None
    }
    /// Free the layer's GPU memory and keep its data, so a tile cache can
    /// hold many more layers than it draws: the next draw uploads again.
    /// Kinds with GPU objects of their own (meshes, atlases' bind groups,
    /// fields) drop those too.
    fn release_gpu(&mut self) {
        release_columns(self);
    }
    /// Bytes this layer holds on the GPU: its columns, uniforms and any GPU
    /// objects of its own — what a cache budgets with.
    fn gpu_bytes(&mut self) -> u64 {
        column_bytes(self)
    }
    /// Density marks: the accumulation target and composite bind group.
    #[doc(hidden)]
    fn density(&self) -> Option<(&wgpu::TextureView, &wgpu::BindGroup)> {
        None
    }
    /// Density marks that scale to their own maximum: each reduction level,
    /// its target and the bind group reading the level before it, ending at
    /// one texel.
    #[doc(hidden)]
    fn density_reduction(&self) -> Option<&[density::ReduceLevel]> {
        None
    }
}

/// The part of [`Mark::release_gpu`] every kind shares: columns and common
/// state. For an override that frees GPU objects of its own as well.
pub fn release_columns<M: Mark + ?Sized>(m: &mut M) {
    for c in m.columns() {
        c.release();
    }
    m.common_mut().release_gpu();
}

/// The part of [`Mark::gpu_bytes`] every kind shares.
pub fn column_bytes<M: Mark + ?Sized>(m: &mut M) -> u64 {
    let own: u64 = m.columns().iter().map(|c| c.gpu_bytes()).sum();
    own + m.common().gpu_bytes()
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FrameUniforms {
    view_proj: [f32; 16],
    origin_hi: [f32; 4],
    origin_lo: [f32; 4],
    viewport: [f32; 4],
    misc: [f32; 4],
    light: [f32; 4],
    eye: [f32; 4],
    /// x: the quad margin past a mark's edge, pixels.
    aa: [f32; 4],
}

const LAYER_SIZE: u64 = std::mem::size_of::<LayerUniforms>() as u64;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LayerUniforms {
    params: [[f32; 4]; 4],
    style: [f32; 4],
    style2: [f32; 4],
    style3: [f32; 4],
    stroke: [f32; 4],
    ids: [u32; 4],
    /// x: [`MarkStyle::depth_lift`].
    depth: [f32; 4],
    /// What positions are measured from, split — see `mark_relative`.
    origin_hi: [f32; 4],
    origin_lo: [f32; 4],
    /// The clip rectangle, relative: `(x0, z0, x1, z1)`.
    clip: [f32; 4],
    /// [`Mark::kind_extra`].
    extra: [f32; 4],
    /// [`MarkStyle::highlight_color`], linear.
    highlight: [f32; 4],
    /// [`MarkStyle::horizon`], relative.
    horizon: [f32; 4],
}

/// Layouts and samplers the mark types build their bind groups against.
#[doc(hidden)]
pub struct SharedResources {
    pub(crate) frame_layout: wgpu::BindGroupLayout,
    pub(crate) layer_layout: wgpu::BindGroupLayout,
    /// Group 2 of a glyph mark: atlas + sampler.
    pub(crate) texture_layout: wgpu::BindGroupLayout,
    /// Group 2 of the density composite: field, ramp, sampler, maximum.
    pub(crate) density_layout: wgpu::BindGroupLayout,
    /// Group 0 of a density reduction step: the level before, a sampler.
    pub(crate) reduce_layout: wgpu::BindGroupLayout,
    pub(crate) nearest_sampler: wgpu::Sampler,
    pub(crate) linear_sampler: wgpu::Sampler,
    /// Linear between mip levels too, for atlases that keep them.
    pub(crate) mip_sampler: wgpu::Sampler,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DepthKey {
    /// The pass has no depth attachment.
    None,
    /// It has one, and this pipeline ignores it.
    Ignore,
    Test {
        write: bool,
        /// Reversed depth: nearer is larger.
        reversed: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PipelineKey {
    kind: MarkKind,
    hooks: u64,
    custom: [u32; 2],
    local: bool,
    format: wgpu::TextureFormat,
    samples: u32,
    depth: DepthKey,
    blend: MarkBlend,
    pick: bool,
}

/// Every layer's uniforms in one buffer, one bind group over it, a layer
/// picked by dynamic offset: a frame of hundreds of tile layers is one
/// buffer write rather than hundreds.
struct LayerArena {
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Layers it holds.
    capacity: usize,
}

/// Draws and picks marks. Holds the pipelines, built on first use and cached
/// for the life of the renderer.
pub struct MarkRenderer {
    resources: SharedResources,
    frame_uniform: wgpu::Buffer,
    frame_bind_group: wgpu::BindGroup,
    layers: Option<LayerArena>,
    /// Bytes between layers in the arena: the uniforms rounded up to the
    /// device's dynamic-offset alignment.
    layer_stride: u64,
    /// The layer uniforms of the prepare in progress.
    layer_bytes: Vec<u8>,
    modules: HashMap<(MarkKind, u64, [u32; 2], bool), wgpu::ShaderModule>,
    pipelines: HashMap<PipelineKey, wgpu::RenderPipeline>,
    composites: HashMap<(wgpu::TextureFormat, u32, DepthKey), wgpu::RenderPipeline>,
    composite_module: Option<wgpu::ShaderModule>,
    reduce_pipeline: Option<wgpu::RenderPipeline>,
    pick: picking::PickTargets,
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_entry(binding: u32, filterable: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn blend_state(blend: MarkBlend) -> wgpu::BlendState {
    match blend {
        MarkBlend::Alpha => wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        },
        MarkBlend::Additive => wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        },
    }
}

fn depth_state(depth: DepthKey) -> Option<wgpu::DepthStencilState> {
    let (write, compare) = match depth {
        DepthKey::None => return None,
        DepthKey::Ignore => (false, wgpu::CompareFunction::Always),
        DepthKey::Test { write, reversed } => (
            write,
            if reversed {
                wgpu::CompareFunction::GreaterEqual
            } else {
                wgpu::CompareFunction::LessEqual
            },
        ),
    };
    Some(wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: Some(write),
        depth_compare: Some(compare),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    })
}

impl MarkRenderer {
    pub fn new(device: &wgpu::Device) -> Self {
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers marks frame bgl"),
            entries: &[uniform_entry(0)],
        });
        let layer_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers marks layer bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(LAYER_SIZE),
                },
                count: None,
            }],
        });
        let align = u64::from(device.limits().min_uniform_buffer_offset_alignment);
        let layer_stride = LAYER_SIZE.div_ceil(align) * align;
        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers marks texture bgl"),
            entries: &[texture_entry(0, true), sampler_entry(1)],
        });
        let density_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers marks density bgl"),
            entries: &[
                // The field is sampled filtered: it may be smaller than the
                // target.
                texture_entry(0, true),
                texture_entry(1, true),
                sampler_entry(2),
                // The field's maximum, one texel, for `auto_max`.
                texture_entry(3, true),
            ],
        });
        let reduce_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers marks density reduce bgl"),
            entries: &[texture_entry(0, true), sampler_entry(1)],
        });
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers marks sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let frame_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threers marks frame uniforms"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers marks frame bind group"),
            layout: &frame_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_uniform.as_entire_binding(),
            }],
        });
        Self {
            resources: SharedResources {
                frame_layout,
                layer_layout,
                texture_layout,
                density_layout,
                reduce_layout,
                // Texel centres, read exactly: see `sample_postfx_depth` on
                // why a sample and not a `textureLoad` (WebGL2).
                nearest_sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("threers marks nearest sampler"),
                    ..Default::default()
                }),
                linear_sampler,
                mip_sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("threers marks mip sampler"),
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::MipmapFilterMode::Linear,
                    ..Default::default()
                }),
            },
            frame_uniform,
            frame_bind_group,
            layers: None,
            layer_stride,
            layer_bytes: Vec::new(),
            modules: HashMap::new(),
            pipelines: HashMap::new(),
            composites: HashMap::new(),
            composite_module: None,
            reduce_pipeline: None,
            pick: picking::PickTargets::default(),
        }
    }

    /// Pipelines built so far — for tests and diagnostics.
    pub fn pipeline_count(&self) -> usize {
        self.pipelines.len() + self.composites.len()
    }

    /// Upload everything `marks` need for `view`, and give each its pick id.
    fn prepare_all(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &MarkView,
        marks: &mut [&mut dyn Mark],
    ) {
        let stride = self.layer_stride as usize;
        let mut bytes = std::mem::take(&mut self.layer_bytes);
        bytes.clear();
        bytes.resize(marks.len().max(1) * stride, 0);
        for (i, m) in marks.iter_mut().enumerate() {
            m.common_mut().pick_id = i as u32 + 1;
            m.common_mut().layer_offset = (i * stride) as u32;
            let ctx = PrepareContext {
                device,
                queue,
                view,
                resources: &self.resources,
            };
            m.prepare(&ctx);
            let rows = m.rows();
            for c in m.columns() {
                c.upload(device, queue, rows);
            }
            for c in m.common_mut().custom.iter_mut().flatten() {
                c.upload(device, queue, rows);
            }
            let (mut style2, style3, stroke) = m.kind_uniforms();
            let extra = m.kind_extra();
            let common = m.common_mut();
            let s = &common.style;
            style2[3] = s.opacity;
            // Absolute: the view's origin. Local: minus the offset from the
            // view's origin to the layer's, so `(p + d_hi) + d_lo` comes out
            // of the same subtraction.
            let layer_origin = match common.local_origin {
                Some(o) => {
                    let v = view.origin.position;
                    let (hi, lo) = split_f64x3([-(o[0] - v[0]), -(o[1] - v[1]), -(o[2] - v[2])]);
                    ([hi[0], hi[1], hi[2], 0.0], [lo[0], lo[1], lo[2], 0.0])
                }
                None => view.origin.split(),
            };
            let u = LayerUniforms {
                params: s.params,
                style: [
                    s.size_scale,
                    s.min_pixels,
                    s.max_pixels,
                    if s.units == SizeUnits::World {
                        1.0
                    } else {
                        0.0
                    },
                ],
                style2,
                style3,
                stroke,
                ids: [common.pick_id, s.highlight.map_or(0, |r| r + 1), 0, 0],
                highlight: points::srgb_u8_to_linear(s.highlight_color),
                horizon: match s.horizon {
                    Some((c, r)) => {
                        let o = view.origin.position;
                        [
                            (c[0] - o[0]) as f32,
                            (c[1] - o[1]) as f32,
                            (c[2] - o[2]) as f32,
                            r as f32,
                        ]
                    }
                    None => [0.0; 4],
                },
                depth: [
                    s.depth_lift.max(0.0),
                    s.depth_order as f32 * 1e-5,
                    if s.clip.is_some() { 1.0 } else { 0.0 },
                    if s.tone_mapped { 1.0 } else { 0.0 },
                ],
                clip: match s.clip {
                    // Ground y runs along world -z.
                    Some([x0, y0, x1, y1]) => {
                        let o = view.origin.position;
                        [
                            (x0 - o[0]) as f32,
                            (-y1 - o[2]) as f32,
                            (x1 - o[0]) as f32,
                            (-y0 - o[2]) as f32,
                        ]
                    }
                    None => [0.0; 4],
                },
                origin_hi: layer_origin.0,
                origin_lo: layer_origin.1,
                extra,
            };
            bytes[i * stride..i * stride + LAYER_SIZE as usize]
                .copy_from_slice(bytemuck::bytes_of(&u));
        }
        let needed = marks.len().max(1);
        if self.layers.as_ref().is_none_or(|a| a.capacity < needed) {
            // Grown by half again, so a map adding tiles one at a time does
            // not rebuild it every frame.
            let capacity = needed.max(16) + needed / 2;
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("threers mark layer uniforms"),
                size: capacity as u64 * self.layer_stride,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("threers mark layer bind group"),
                layout: &self.resources.layer_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new(LAYER_SIZE),
                    }),
                }],
            });
            self.layers = Some(LayerArena {
                buffer,
                bind_group,
                capacity,
            });
        }
        let arena = self.layers.as_ref().unwrap();
        queue.write_buffer(&arena.buffer, 0, &bytes);
        self.layer_bytes = bytes;
    }

    /// The layer arena's bind group, after a prepare.
    fn layer_bind_group(&self) -> &wgpu::BindGroup {
        &self.layers.as_ref().expect("prepared").bind_group
    }

    /// Bytes the shared layer uniforms hold on the GPU.
    pub fn gpu_bytes(&self) -> u64 {
        self.layers.as_ref().map_or(0, |a| a.buffer.size())
    }

    fn module(
        &mut self,
        device: &wgpu::Device,
        kind: MarkKind,
        hooks: &ShaderHooks,
        custom: [u32; 2],
        local: bool,
        custom_shader: Option<&CustomShader>,
    ) -> &wgpu::ShaderModule {
        self.modules
            .entry((kind, hooks.key(), custom, local))
            .or_insert_with(|| {
                let source = match (custom_shader, local) {
                    (Some(cs), false) => cs.module_source(hooks, custom),
                    (Some(cs), true) => cs.module_source_local(hooks, custom),
                    (None, _) => shader::module_source_with(kind, hooks, custom, local),
                };
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("threers mark shader"),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                })
            })
    }

    fn ensure_pipeline(&mut self, device: &wgpu::Device, key: &PipelineKey, mark: &mut dyn Mark) {
        if self.pipelines.contains_key(key) {
            return;
        }
        let hooks = mark.common().hooks.clone();
        let custom_widths = key.custom;
        // Vertex layouts: geometry first for shapes, then the kind's columns,
        // then the custom columns.
        let mut layouts: Vec<(u64, wgpu::VertexStepMode, Vec<wgpu::VertexAttribute>)> = Vec::new();
        let mut loc = 0;
        if key.kind == MarkKind::Shape || mark.geometry().is_some() {
            layouts.push((
                24,
                wgpu::VertexStepMode::Vertex,
                vec![
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 12,
                        shader_location: 1,
                    },
                ],
            ));
            loc = 2;
        }
        let step = if mark.per_vertex() {
            wgpu::VertexStepMode::Vertex
        } else {
            wgpu::VertexStepMode::Instance
        };
        {
            let cols = mark.columns();
            let refs: Vec<&Column> = cols.iter().map(|c| &**c).collect();
            for (stride, attrs) in columns::instance_layouts(&refs, loc) {
                layouts.push((stride, step, attrs));
            }
        }
        if key.kind != MarkKind::Glyph {
            let first = match mark.custom_shader() {
                Some(cs) => cs.custom_location,
                None => shader::custom_location(key.kind),
            };
            for (cloc, c) in (first..).zip(mark.common().custom.iter().flatten()) {
                let (stride, attrs) = c.layout(cloc);
                layouts.push((stride, step, attrs));
            }
        }
        let buffers: Vec<Option<wgpu::VertexBufferLayout>> = layouts
            .iter()
            .map(|(stride, step, attrs)| {
                Some(wgpu::VertexBufferLayout {
                    array_stride: *stride,
                    step_mode: *step,
                    attributes: attrs,
                })
            })
            .collect();

        let mut groups: Vec<Option<&wgpu::BindGroupLayout>> = vec![
            Some(&self.resources.frame_layout),
            Some(&self.resources.layer_layout),
        ];
        if matches!(
            key.kind,
            MarkKind::Glyph | MarkKind::Sprite | MarkKind::Image
        ) {
            groups.push(Some(&self.resources.texture_layout));
        } else if let Some(l) = mark.extra_layout() {
            groups.push(Some(l));
        }
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers mark pipeline layout"),
            bind_group_layouts: &groups,
            immediate_size: 0,
        });
        let cs = mark.custom_shader().cloned();
        let module = self
            .module(
                device,
                key.kind,
                &hooks,
                custom_widths,
                key.local,
                cs.as_ref(),
            )
            .clone();
        let target = if key.pick {
            wgpu::ColorTargetState {
                format: key.format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }
        } else if key.kind == MarkKind::Splat {
            // Densities sum, unweighted by alpha.
            wgpu::ColorTargetState {
                format: key.format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            }
        } else {
            wgpu::ColorTargetState {
                format: key.format,
                blend: Some(blend_state(key.blend)),
                write_mask: wgpu::ColorWrites::ALL,
            }
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("threers mark pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &buffers,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some(if key.pick { "fs_pick" } else { "fs_main" }),
                targets: &[Some(target)],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: depth_state(key.depth).map(|mut d| {
                if matches!(key.kind, MarkKind::Polygon | MarkKind::Image) {
                    // Pull fills towards the camera, so one lying on a ground
                    // mesh at the same height wins instead of fighting it —
                    // towards larger depth when depth is reversed.
                    let toward = if matches!(key.depth, DepthKey::Test { reversed: true, .. }) {
                        -1
                    } else {
                        1
                    };
                    d.bias = wgpu::DepthBiasState {
                        constant: -2 * toward,
                        slope_scale: -(toward as f32),
                        clamp: 0.0,
                    };
                }
                d
            }),
            multisample: wgpu::MultisampleState {
                count: key.samples,
                ..Default::default()
            },
            multiview_mask: None,
            cache: None,
        });
        self.pipelines.insert(key.clone(), pipeline);
    }

    fn ensure_composite(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        samples: u32,
        depth: DepthKey,
    ) {
        if self.composites.contains_key(&(format, samples, depth)) {
            return;
        }
        let module = self
            .composite_module
            .get_or_insert_with(|| {
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("threers density composite"),
                    source: wgpu::ShaderSource::Wgsl(shader::composite_source().into()),
                })
            })
            .clone();
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers density composite layout"),
            bind_group_layouts: &[
                Some(&self.resources.frame_layout),
                Some(&self.resources.layer_layout),
                Some(&self.resources.density_layout),
            ],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("threers density composite pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(blend_state(MarkBlend::Alpha)),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: depth_state(depth),
            multisample: wgpu::MultisampleState {
                count: samples,
                ..Default::default()
            },
            multiview_mask: None,
            cache: None,
        });
        self.composites.insert((format, samples, depth), pipeline);
    }

    /// The max-of-4×4 step a density field's maximum is reduced by.
    fn ensure_reduce(&mut self, device: &wgpu::Device) {
        if self.reduce_pipeline.is_some() {
            return;
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers density reduce"),
            source: wgpu::ShaderSource::Wgsl(shader::reduce_source().into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers density reduce layout"),
            bind_group_layouts: &[Some(&self.resources.reduce_layout)],
            immediate_size: 0,
        });
        self.reduce_pipeline = Some(device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("threers density reduce pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: density::FIELD_FORMAT,
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
            },
        ));
    }

    fn key_for(
        mark: &dyn Mark,
        format: wgpu::TextureFormat,
        samples: u32,
        has_depth: bool,
        pick: bool,
        reversed: bool,
    ) -> PipelineKey {
        let s = &mark.common().style;
        let depth = if !has_depth {
            DepthKey::None
        } else if s.depth_test || s.depth_write {
            DepthKey::Test {
                write: s.depth_write,
                reversed,
            }
        } else {
            DepthKey::Ignore
        };
        PipelineKey {
            kind: mark.kind(),
            hooks: mark.common().hooks.key(),
            local: mark.common().local_origin.is_some(),
            custom: if mark.kind() == MarkKind::Glyph {
                [0, 0]
            } else {
                mark.common().custom_widths()
            },
            format,
            samples,
            depth,
            blend: s.blend,
            pick,
        }
    }

    /// Record one mark's draw into `pass`.
    fn record(
        pass: &mut wgpu::RenderPass<'_>,
        pipeline: &wgpu::RenderPipeline,
        layers: &wgpu::BindGroup,
        mark: &mut dyn Mark,
    ) {
        let count = mark.instance_count() as u32;
        let verts = mark.vertices_per_instance();
        pass.set_pipeline(pipeline);
        pass.set_bind_group(1, layers, &[mark.common().layer_offset]);
        if let Some(bg) = mark.extra_bind_group() {
            pass.set_bind_group(2, bg, &[]);
        }
        let mut slot = 0;
        let geometry = mark
            .geometry()
            .map(|(vb, n, ib)| (vb.clone(), n, ib.map(|(b, c)| (b.clone(), c))));
        if let Some((vb, _, _)) = &geometry {
            pass.set_vertex_buffer(0, vb.slice(..));
            slot = 1;
        }
        let kind = mark.kind();
        let mut buffers: Vec<wgpu::Buffer> = mark
            .columns()
            .iter()
            .map(|c| c.buffer.clone().expect("uploaded"))
            .collect();
        if kind != MarkKind::Glyph {
            buffers.extend(
                mark.common()
                    .custom
                    .iter()
                    .flatten()
                    .map(|c| c.buffer.clone().expect("uploaded")),
            );
        }
        for b in &buffers {
            pass.set_vertex_buffer(slot, b.slice(..));
            slot += 1;
        }
        let index = mark.index_buffer().map(|(b, n)| (b.clone(), n));
        match (geometry, index) {
            (Some((_, _, Some((ib, n)))), _) | (None, Some((ib, n))) => {
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..n, 0, 0..count);
            }
            (Some((_, n, None)), _) => pass.draw(0..n, 0..count),
            (None, None) => pass.draw(0..verts, 0..count),
        }
    }

    /// Draw `marks`, in order, over `target`.
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &MarkTarget,
        view: &MarkView,
        marks: &mut [&mut dyn Mark],
    ) {
        let encode = crate::renderer::shader_applies_transfer(target.format);
        queue.write_buffer(
            &self.frame_uniform,
            0,
            bytemuck::bytes_of(&view.uniforms(encode, target.samples)),
        );
        self.prepare_all(device, queue, view, marks);

        let has_depth = target.depth.is_some();
        // Build every pipeline first; recording borrows them.
        for m in marks.iter_mut() {
            if !m.common().style.visible || m.instance_count() == 0 {
                continue;
            }
            if m.kind() == MarkKind::Splat {
                let key = Self::key_for(
                    &**m,
                    density::FIELD_FORMAT,
                    1,
                    false,
                    false,
                    view.reversed_z,
                );
                self.ensure_pipeline(device, &key, &mut **m);
                let depth = if has_depth {
                    DepthKey::Ignore
                } else {
                    DepthKey::None
                };
                self.ensure_composite(device, target.format, target.samples, depth);
                if m.density_reduction().is_some() {
                    self.ensure_reduce(device);
                }
            } else {
                let key = Self::key_for(
                    &**m,
                    target.format,
                    target.samples,
                    has_depth,
                    false,
                    view.reversed_z,
                );
                self.ensure_pipeline(device, &key, &mut **m);
            }
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("threers marks"),
        });
        let mut i = 0;
        while i < marks.len() {
            // A run of ordinary marks shares one pass; a density mark needs
            // a pass of its own into its field first.
            let start = i;
            while i < marks.len() && marks[i].kind() != MarkKind::Splat {
                i += 1;
            }
            if i > start {
                let mut pass = Self::begin_target_pass(&mut encoder, target);
                pass.set_bind_group(0, &self.frame_bind_group, &[]);
                for m in marks[start..i].iter_mut() {
                    if !m.common().style.visible || m.instance_count() == 0 {
                        continue;
                    }
                    let key = Self::key_for(
                        &**m,
                        target.format,
                        target.samples,
                        has_depth,
                        false,
                        view.reversed_z,
                    );
                    let pipeline = &self.pipelines[&key];
                    Self::record(&mut pass, pipeline, self.layer_bind_group(), &mut **m);
                }
            }
            if i < marks.len() {
                let m = &mut marks[i];
                i += 1;
                if !m.common().style.visible || m.instance_count() == 0 {
                    continue;
                }
                let key = Self::key_for(
                    &**m,
                    density::FIELD_FORMAT,
                    1,
                    false,
                    false,
                    view.reversed_z,
                );
                {
                    let (field, _) = m.density().unwrap();
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("threers density field"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: field,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    pass.set_bind_group(0, &self.frame_bind_group, &[]);
                    Self::record(
                        &mut pass,
                        &self.pipelines[&key],
                        self.layer_bind_group(),
                        &mut **m,
                    );
                }
                if let Some(levels) = m.density_reduction() {
                    let reduce = self
                        .reduce_pipeline
                        .as_ref()
                        .expect("built with the pipelines");
                    for level in levels {
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("threers density reduce"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &level.target,
                                depth_slice: None,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        });
                        pass.set_pipeline(reduce);
                        pass.set_bind_group(0, &level.bind_group, &[]);
                        pass.draw(0..3, 0..1);
                    }
                }
                let depth = if has_depth {
                    DepthKey::Ignore
                } else {
                    DepthKey::None
                };
                let composite = &self.composites[&(target.format, target.samples, depth)];
                let mut pass = Self::begin_target_pass(&mut encoder, target);
                pass.set_pipeline(composite);
                pass.set_bind_group(0, &self.frame_bind_group, &[]);
                pass.set_bind_group(1, self.layer_bind_group(), &[m.common().layer_offset]);
                pass.set_bind_group(2, m.density().unwrap().1, &[]);
                pass.draw(0..3, 0..1);
            }
        }
        queue.submit(Some(encoder.finish()));
    }

    fn begin_target_pass<'e>(
        encoder: &'e mut wgpu::CommandEncoder,
        target: &MarkTarget<'_>,
    ) -> wgpu::RenderPass<'e> {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("threers marks pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target.color,
                depth_slice: None,
                resolve_target: target.resolve,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: target.depth.map(|d| {
                wgpu::RenderPassDepthStencilAttachment {
                    view: d,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WebGPU guarantees a pipeline only eight vertex buffers, and the
    /// ninth only fails when a layer binds both custom columns — which no
    /// test without hooks does.
    #[test]
    fn every_kind_fits_in_eight_vertex_buffers_with_both_custom_columns() {
        let geometry = crate::BoxGeometry::new(1.0, 1.0, 1.0);
        let mut marks: Vec<Box<dyn Mark>> = vec![
            Box::new(PointMarks::new()),
            Box::new(SegmentMarks::new()),
            Box::new(ShapeMarks::new(&geometry)),
            Box::new(DensityMarks::new()),
            Box::new(PolygonMarks::new()),
            Box::new(SpriteMarks::new(TextureAtlas::new(64).shared())),
            Box::new(ImageMarks::new(TextureAtlas::new(64).shared())),
        ];
        for m in &mut marks {
            let kind = m.kind();
            let own = m.columns().len();
            let mesh = usize::from(kind == MarkKind::Shape);
            let custom = if kind == MarkKind::Glyph { 0 } else { 2 };
            assert!(
                own + mesh + custom <= 8,
                "{kind:?}: {own} columns + {mesh} mesh + {custom} custom"
            );
        }
    }
}
