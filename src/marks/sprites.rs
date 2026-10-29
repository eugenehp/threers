//! Icons from an atlas, one per row.

use super::atlas::{AtlasRect, SharedAtlas};
use super::{Column, ColumnFormat, KindUniforms, Mark, MarkCommon, MarkKind, PrepareContext};

/// The view, collision generation, content version and box settings
/// (anchor, padding, size scale, min and max pixels, units) a placement was
/// made for.
type PlacedFor = ([f32; 16], (u32, u32), [f64; 3], u64, u64, [u32; 7]);

/// How a sprite's texels become its colour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpriteMode {
    /// The image's own colours, times the row's tint.
    #[default]
    Color,
    /// The image's alpha as a mask over the row's colour — one monochrome
    /// icon recoloured per row.
    Mask,
    /// The image's alpha as a signed distance field (0.5 on the edge, or
    /// [`SpriteSdf::edge`]), drawn sharp at any size in the row's colour,
    /// with an optional outline ([`SpriteMarks::sdf`]).
    DistanceField,
    /// No image: a rectangle the size of the row's rect (its aspect, at the
    /// row's size) in the row's colour, with rounded corners and a border
    /// ([`SpriteMarks::box_style`]) — a label's background.
    RoundedBox,
}

/// How a [`SpriteMode::DistanceField`] is drawn: deck.gl's `fontSettings`
/// (`buffer`, `smoothing`) and `outlineWidth`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpriteSdf {
    /// The field value on the edge, 0 to 1.
    pub edge: f32,
    /// Half the width of the edge's ramp, in field units; `None` for about
    /// a pixel at the size drawn.
    pub smoothing: Option<f32>,
    /// A border outside the edge, out to this field value, in this colour
    /// (sRGB, straight alpha).
    pub outline: Option<(f32, [u8; 4])>,
}

impl Default for SpriteSdf {
    fn default() -> Self {
        Self { edge: 0.5, smoothing: None, outline: None }
    }
}

/// How a [`SpriteMode::RoundedBox`] is drawn, in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SpriteBox {
    pub corner_radius: f32,
    pub border_width: f32,
    /// sRGB, straight alpha.
    pub border_color: [u8; 4],
}

/// Images from a [`TextureAtlas`](super::TextureAtlas), one per row, facing
/// the camera — deck.gl's `IconLayer`.
///
/// Each row has a position, a size (the image's height, in pixels or world
/// units, times [`MarkStyle::size_scale`](super::MarkStyle::size_scale); the
/// width follows the image's aspect), the image's rect in the atlas, a
/// rotation in radians (counter-clockwise on screen) and a colour.
/// [`anchor`](Self::anchor) says which point of the image sits on the
/// position: `[0, 0]` the centre, `[0, -1]` the bottom middle — a map pin.
///
/// Rows whose texel is transparent are not pickable there, so a pick lands
/// on the icon's ink rather than its box.
#[derive(Debug)]
pub struct SpriteMarks {
    pub common: MarkCommon,
    atlas: SharedAtlas,
    positions: Column,
    /// Per row: the size, and the offset of the image's centre from the
    /// anchored point in the same units, before turning (x right, y down).
    sizes: Column,
    rects: Column,
    /// Per row: the angle, and a pixel offset after turning (x right, y
    /// down).
    angles: Column,
    colors: Column,
    size_rows: Vec<f32>,
    offset_rows: Vec<[f32; 2]>,
    angle_rows: Vec<f32>,
    pixel_offset_rows: Vec<[f32; 2]>,
    pub mode: SpriteMode,
    /// Lie flat on the ground (the plane of +Y, the image's top toward -Z)
    /// instead of facing the camera — deck.gl's `IconLayer` without
    /// `billboard`. Sized as it would be facing the camera at the focal
    /// plane, so it keeps its size in the world and foreshortens with the
    /// view's tilt.
    pub ground: bool,
    /// Fragments fainter than this (after opacity) are not drawn: deck.gl's
    /// `alphaCutoff` (0.05 for its icons). 0.002 by default.
    pub alpha_cutoff: f32,
    /// For [`SpriteMode::DistanceField`].
    pub sdf: SpriteSdf,
    /// For [`SpriteMode::RoundedBox`].
    pub box_style: SpriteBox,
    /// The image point placed on the position, in `[-1, 1]²` with y up.
    pub anchor: [f32; 2],
    /// Hide icons whose box overlaps one placed before them — in this layer
    /// (higher [`priorities`](Self::set_priorities) first, then earlier
    /// rows) and, with a [`collision_index`](Self::collision_index), in every
    /// layer sharing it. Placed on the CPU each time the view changes.
    pub collision: bool,
    pub collision_index: Option<super::SharedCollision>,
    /// Extra space kept clear around each icon, in pixels.
    pub padding: f32,
    /// Show row `i` exactly when label `follow.1[i]` of a label layer
    /// shows (one that [publishes](super::TextLabels::publish_visibility)
    /// into `follow.0`), instead of placing on its own: icons that belong
    /// to labels, placed with them. Such a layer
    /// [prepares late](Mark::prepare_late), so it follows the same frame's
    /// placement wherever it draws.
    pub follow: Option<(super::SharedVisibility, Vec<u32>)>,
    /// Per row: shown (collision), the corners' depth offsets, and their
    /// screen nudges.
    shown: Column,
    /// The last placement's visibility; empty for all shown.
    shown_rows: Vec<f32>,
    depth_rows: Vec<[f32; 4]>,
    /// The corners' screen nudges: x of the four, then y.
    nudge_rows: Vec<[f32; 8]>,
    priorities: Vec<f32>,
    placed: Option<PlacedFor>,
    /// Bumped by every setter that moves, sizes or turns an icon.
    version: u64,
    /// Marks this layer's boxes in a shared index.
    owner: u64,
    gpu: Option<(u64, wgpu::BindGroup)>,
    atlas_size: (u32, u32),
    atlas_srgb: bool,
}

/// A layer dropped while the view holds still (a tile replaced) would leave
/// its boxes blocking whatever replaces it: take them out, and have the next
/// frame place everything again.
impl Drop for SpriteMarks {
    fn drop(&mut self) {
        if let (Some(index), true) = (&self.collision_index, self.placed.is_some()) {
            let mut index = index.lock().unwrap_or_else(|e| e.into_inner());
            index.remove_owner(self.owner);
            index.invalidate();
        }
    }
}

impl SpriteMarks {
    pub fn new(atlas: SharedAtlas) -> Self {
        let mut sizes = Column::float(3);
        sizes.set_default_f32(&[1.0, 0.0, 0.0]);
        let mut common = MarkCommon::default();
        common.style.size_scale = 24.0;
        Self {
            common,
            atlas,
            positions: Column::new(ColumnFormat::PositionF64),
            sizes,
            rects: Column::float(4),
            angles: Column::float(3),
            colors: Column::new(ColumnFormat::Unorm8x4),
            size_rows: Vec::new(),
            offset_rows: Vec::new(),
            angle_rows: Vec::new(),
            pixel_offset_rows: Vec::new(),
            mode: SpriteMode::Color,
            ground: false,
            alpha_cutoff: 0.002,
            sdf: SpriteSdf::default(),
            box_style: SpriteBox::default(),
            anchor: [0.0, 0.0],
            collision: false,
            collision_index: None,
            padding: 1.0,
            follow: None,
            shown: {
                let mut c = Column::new(ColumnFormat::Float4x4);
                c.set_default_f32(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
                c
            },
            shown_rows: Vec::new(),
            depth_rows: Vec::new(),
            nudge_rows: Vec::new(),
            priorities: Vec::new(),
            placed: None,
            version: 0,
            owner: super::collision::next_owner(),
            gpu: None,
            atlas_size: (1, 1),
            atlas_srgb: true,
        }
    }

    pub fn atlas(&self) -> &SharedAtlas {
        &self.atlas
    }

    pub fn set_positions(&mut self, p: &[[f64; 3]]) {
        self.version += 1;
        self.common.local_origin = None;
        self.positions.set_positions(p);
    }
    /// Positions relative to `origin` — half the memory of absolute ones,
    /// and what a tile decoder produces. See
    /// [`ColumnFormat::PositionLocal`](super::ColumnFormat::PositionLocal).
    pub fn set_local_positions(&mut self, origin: [f64; 3], p: &[[f32; 3]]) {
        self.version += 1;
        self.common.local_origin = Some(origin);
        self.positions.set_local_positions(p);
    }
    pub fn update_local_positions(&mut self, start: usize, p: &[[f32; 3]]) {
        self.version += 1;
        self.positions.update_local_positions(start, p);
    }
    pub fn update_positions(&mut self, start: usize, p: &[[f64; 3]]) {
        self.version += 1;
        self.positions.update_positions(start, p);
    }
    pub fn set_sizes(&mut self, s: &[f32]) {
        self.version += 1;
        self.size_rows = s.to_vec();
        self.pack_sizes();
    }
    pub fn update_sizes(&mut self, start: usize, s: &[f32]) {
        self.version += 1;
        let n = self.size_rows.len().max(start + s.len());
        self.size_rows.resize(n, 1.0);
        self.size_rows[start..start + s.len()].copy_from_slice(s);
        self.pack_sizes();
    }
    /// Each row's image centre relative to its anchored point, in the
    /// size's units (x right, y down), turned with the image: one glyph of
    /// a label, all a label's glyphs sharing its position. Empty for none.
    pub fn set_offsets(&mut self, o: &[[f32; 2]]) {
        self.version += 1;
        self.offset_rows = o.to_vec();
        self.pack_sizes();
    }
    fn pack_sizes(&mut self) {
        let n = self.size_rows.len().max(self.offset_rows.len());
        let packed: Vec<f32> = (0..n)
            .flat_map(|i| {
                let o = self.offset_rows.get(i).copied().unwrap_or([0.0; 2]);
                [self.size_rows.get(i).copied().unwrap_or(1.0), o[0], o[1]]
            })
            .collect();
        self.sizes.set_f32(&packed);
    }
    /// Each row's image, as rects in the atlas.
    pub fn set_rects(&mut self, r: &[AtlasRect]) {
        self.version += 1;
        let flat: Vec<f32> = r.iter().flat_map(|r| r.to_array()).collect();
        self.rects.set_f32(&flat);
    }
    /// Each row's image as `[x, y, width, height]` in the atlas's pixels,
    /// fractions kept: a frame placed between pixels samples there.
    pub fn set_pixel_rects(&mut self, r: &[[f32; 4]]) {
        self.version += 1;
        let flat: Vec<f32> = r.iter().flatten().copied().collect();
        self.rects.set_f32(&flat);
    }
    pub fn update_rects(&mut self, start: usize, r: &[AtlasRect]) {
        self.version += 1;
        let flat: Vec<f32> = r.iter().flat_map(|r| r.to_array()).collect();
        self.rects.update_f32(start, &flat);
    }
    /// For [`SpriteMode::RoundedBox`]: each row's box as `[dash, gap,
    /// width, height]` — its border dashed (`dash` and `gap` in border
    /// widths along the box's edge, from its top-left corner down; 0 and 0
    /// solid), and its aspect.
    pub fn set_box_rects(&mut self, r: &[[f32; 4]]) {
        self.version += 1;
        let flat: Vec<f32> = r.iter().flatten().copied().collect();
        self.rects.set_f32(&flat);
    }
    /// Each row's image by name; unknown names draw nothing.
    pub fn set_images(&mut self, names: &[&str]) {
        let atlas = self.atlas.read().unwrap();
        let r: Vec<AtlasRect> = names
            .iter()
            .map(|n| {
                atlas.get(n).unwrap_or(AtlasRect {
                    x: 0,
                    y: 0,
                    width: 0,
                    height: 0,
                })
            })
            .collect();
        drop(atlas);
        self.set_rects(&r);
    }
    /// The image of rows with no rect of their own.
    pub fn set_default_rect(&mut self, r: AtlasRect) {
        self.version += 1;
        self.rects.set_default_f32(&r.to_array());
    }
    pub fn set_angles(&mut self, a: &[f32]) {
        self.version += 1;
        self.angle_rows = a.to_vec();
        self.pack_angles();
    }
    pub fn update_angles(&mut self, start: usize, a: &[f32]) {
        self.version += 1;
        let n = self.angle_rows.len().max(start + a.len());
        self.angle_rows.resize(n, 0.0);
        self.angle_rows[start..start + a.len()].copy_from_slice(a);
        self.pack_angles();
    }
    /// Each row moved by this many pixels after turning (x right, y down):
    /// deck.gl's `getPixelOffset`. Empty for none.
    pub fn set_pixel_offsets(&mut self, o: &[[f32; 2]]) {
        self.version += 1;
        self.pixel_offset_rows = o.to_vec();
        self.pack_angles();
    }
    fn pack_angles(&mut self) {
        let n = self.angle_rows.len().max(self.pixel_offset_rows.len());
        let packed: Vec<f32> = (0..n)
            .flat_map(|i| {
                let o = self.pixel_offset_rows.get(i).copied().unwrap_or([0.0; 2]);
                [self.angle_rows.get(i).copied().unwrap_or(0.0), o[0], o[1]]
            })
            .collect();
        self.angles.set_f32(&packed);
    }
    /// sRGB colours, straight alpha: a tint in [`SpriteMode::Color`], the
    /// colour in the other modes.
    pub fn set_colors(&mut self, c: &[[u8; 4]]) {
        self.colors.set_rgba(c);
    }
    pub fn update_colors(&mut self, start: usize, c: &[[u8; 4]]) {
        self.colors.update_rgba(start, c);
    }
    pub fn set_default_color(&mut self, c: [u8; 4]) {
        self.colors.set_default_rgba(c);
    }
    /// Which icons win collisions: higher first. Rows without one rank 0.
    pub fn set_priorities(&mut self, p: &[f32]) {
        self.priorities = p.to_vec();
        self.version += 1;
    }
    /// Whether each row survived collision at the last placement; all shown
    /// without [`collision`](Self::collision).
    pub fn shown(&self) -> Vec<bool> {
        (0..self.positions.len())
            .map(|i| self.shown.row_f32(i)[0] > 0.5)
            .collect()
    }

    /// Each row's four corners moved this far in depth (the depth buffer's
    /// units, towards the far plane), in the quad's corner order: bottom
    /// left, bottom right, top right, top left of the image. What a renderer
    /// computing in float32 puts between coplanar icons — deck.gl's flat
    /// icons win their overlaps by it. Empty for none.
    pub fn set_depth_offsets(&mut self, d: &[[f32; 4]]) {
        self.version += 1;
        self.depth_rows = d.to_vec();
        self.pack_shown();
    }

    /// Each row's four corners moved this far on screen, in device pixels
    /// (x right, y down), in the quad's corner order (see
    /// [`set_depth_offsets`](Self::set_depth_offsets)), before any sub-pixel
    /// snap: where a renderer computing in float32 puts each corner, which
    /// the exact projection does not. Empty for none.
    pub fn set_corner_nudges(&mut self, n: &[[[f32; 2]; 4]]) {
        self.version += 1;
        self.nudge_rows = n.iter().map(|c| [c[0][0], c[1][0], c[2][0], c[3][0], c[0][1], c[1][1], c[2][1], c[3][1]]).collect();
        self.pack_shown();
    }

    /// The visibility, depth and nudge column, from the last placement,
    /// the depth offsets and the nudges.
    fn pack_shown(&mut self) {
        let n = self.shown_rows.len().max(self.depth_rows.len()).max(self.nudge_rows.len());
        let packed: Vec<f32> = (0..n)
            .flat_map(|i| {
                let d = self.depth_rows.get(i).copied().unwrap_or([0.0; 4]);
                let m = self.nudge_rows.get(i).copied().unwrap_or([0.0; 8]);
                let mut row = [0.0f32; 16];
                row[0] = self.shown_rows.get(i).copied().unwrap_or(1.0);
                row[4..8].copy_from_slice(&d);
                row[8..16].copy_from_slice(&m);
                row
            })
            .collect();
        self.shown.set_f32(&packed);
    }

    /// Cull overlapping icons for `view`.
    fn place(&mut self, view: &super::MarkView) {
        let shared = self.collision_index.clone();
        let generation = shared
            .as_ref()
            .map_or(0, |s| s.lock().unwrap().generation());
        let s = &self.common.style;
        let settings = [
            self.anchor[0].to_bits(),
            self.anchor[1].to_bits(),
            self.padding.to_bits(),
            s.size_scale.to_bits(),
            s.min_pixels.to_bits(),
            s.max_pixels.to_bits(),
            u32::from(s.units == super::SizeUnits::World),
        ];
        let key = (
            view.view_proj,
            view.viewport,
            view.origin.position,
            generation,
            self.version,
            settings,
        );
        if self.placed == Some(key) {
            return;
        }
        if let Some(sh) = &shared {
            sh.lock()
                .unwrap()
                .reclaim(self.owner, self.placed.map(|p| p.3));
        }
        self.placed = Some(key);
        let Some(pos) = self.positions.positions_f64(self.common.local_origin) else {
            return;
        };
        let s = &self.common.style;
        let (vw, vh) = (view.viewport.0 as f32, view.viewport.1 as f32);
        let mut order: Vec<usize> = (0..pos.len()).collect();
        let pr = |i: usize| self.priorities.get(i).copied().unwrap_or(0.0);
        order.sort_by(|&a, &b| {
            pr(b)
                .partial_cmp(&pr(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut private = super::CollisionIndex::new();
        let mut guard = shared.as_ref().map(|s| s.lock().unwrap());
        let index: &mut super::CollisionIndex = match guard.as_deref_mut() {
            Some(g) => g,
            None => &mut private,
        };
        let mut shown = vec![0.0f32; pos.len()];
        for i in order {
            let Some(p) = view.project(pos[i]) else {
                continue;
            };
            let rect = self.rects.row_f32(i);
            if rect[3] <= 0.0 {
                continue;
            }
            let size = self.sizes.row_f32(i);
            let mut h = size[0] * s.size_scale;
            if s.units == super::SizeUnits::World {
                h /= view.units_per_pixel(pos[i]).unwrap_or(1.0) as f32;
            }
            let h = h.clamp(s.min_pixels, s.max_pixels);
            let k = if size[0] != 0.0 { h / size[0] } else { 0.0 };
            let half = [h * rect[2] / rect[3] * 0.5, h * 0.5];
            let turn = self.angles.row_f32(i);
            let a = turn[0];
            let (sn, cs) = a.sin_cos();
            // The image's centre in screen space (y down): the anchor, the
            // row's offset turned with it, then its pixel offset.
            let local = [size[1] * k - self.anchor[0] * half[0], size[2] * k + self.anchor[1] * half[1]];
            let c = [
                p[0] + cs * local[0] + sn * local[1] + turn[1],
                p[1] - sn * local[0] + cs * local[1] + turn[2],
            ];
            let ex = (half[0] * cs).abs() + (half[1] * sn).abs() + self.padding;
            let ey = (half[0] * sn).abs() + (half[1] * cs).abs() + self.padding;
            let r = [c[0] - ex, c[1] - ey, c[0] + ex, c[1] + ey];
            if r[2] < 0.0 || r[3] < 0.0 || r[0] > vw || r[1] > vh {
                continue;
            }
            if index.try_insert_owned(self.owner, r) {
                shown[i] = 1.0;
            }
        }
        drop(guard);
        self.shown_rows = shown;
        self.pack_shown();
    }
}

/// Rebuild a texture bind group when the atlas's texture changed.
pub(crate) fn atlas_bind_group(
    ctx: &PrepareContext,
    atlas: &SharedAtlas,
    gpu: &mut Option<(u64, wgpu::BindGroup)>,
) -> (u32, u32) {
    let mut a = atlas.write().unwrap();
    let (view, generation) = a.texture(ctx.device, ctx.queue);
    let size = a.size();
    drop(a);
    if gpu.as_ref().is_none_or(|g| g.0 != generation) {
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers atlas bind group"),
            layout: &ctx.resources.texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&ctx.resources.mip_sampler),
                },
            ],
        });
        *gpu = Some((generation, bg));
    }
    size
}

impl Mark for SpriteMarks {
    fn kind(&self) -> MarkKind {
        MarkKind::Sprite
    }
    fn common(&self) -> &MarkCommon {
        &self.common
    }
    fn common_mut(&mut self) -> &mut MarkCommon {
        &mut self.common
    }
    fn instance_count(&self) -> usize {
        self.positions.len()
    }
    fn kind_uniforms(&self) -> KindUniforms {
        let mode = match self.mode {
            SpriteMode::Color => 0.0,
            SpriteMode::Mask => 1.0,
            SpriteMode::DistanceField => 2.0,
            SpriteMode::RoundedBox => 3.0,
        };
        (
            [mode, self.alpha_cutoff, if self.ground { 1.0 } else { 0.0 }, 0.0],
            [
                self.atlas_size.0 as f32,
                self.atlas_size.1 as f32,
                self.anchor[0],
                self.anchor[1],
            ],
            [0.0; 4],
        )
    }
    /// extra: the distance field's edge, ramp (0: a pixel) and outline edge
    /// (-1: none); or a box's corner radius and border width.
    fn kind_extra(&self) -> [f32; 4] {
        match self.mode {
            SpriteMode::RoundedBox => [self.box_style.corner_radius, self.box_style.border_width, 0.0, 0.0],
            // w: a colour image in a linear atlas holds sRGB bytes, read
            // after filtering as WebGL reads them.
            SpriteMode::Color => [0.0, 0.0, -1.0, if self.atlas_srgb { 0.0 } else { 1.0 }],
            _ => [
                self.sdf.edge,
                self.sdf.smoothing.unwrap_or(0.0),
                self.sdf.outline.map_or(-1.0, |o| o.0),
                0.0,
            ],
        }
    }
    /// material: the outline's or the border's colour, linear.
    fn kind_material(&self) -> [f32; 4] {
        let c = match self.mode {
            SpriteMode::RoundedBox => self.box_style.border_color,
            _ => self.sdf.outline.map_or([0; 4], |o| o.1),
        };
        super::points::srgb_u8_to_linear(c)
    }
    fn prepare(&mut self, ctx: &PrepareContext) {
        self.atlas_size = atlas_bind_group(ctx, &self.atlas, &mut self.gpu);
        self.atlas_srgb = self.atlas.read().map_or(true, |a| a.is_srgb());
        if let Some((vis, rows)) = &self.follow {
            let vis = vis.lock().unwrap();
            let shown: Vec<f32> =
                rows.iter().map(|&r| if vis.get(r as usize).copied().unwrap_or(false) { 1.0 } else { 0.0 }).collect();
            drop(vis);
            self.shown_rows = shown;
            self.pack_shown();
        } else if self.collision {
            self.place(ctx.view);
        } else if !self.shown_rows.is_empty() {
            self.shown_rows.clear();
            self.pack_shown();
            if let (Some(sh), Some(_)) = (&self.collision_index, self.placed) {
                // Its boxes would keep blocking others until the next clear.
                let mut index = sh.lock().unwrap();
                index.remove_owner(self.owner);
                index.invalidate();
            }
            self.placed = None;
        }
    }
    fn prepare_late(&self) -> bool {
        self.follow.is_some()
    }
    fn release_gpu(&mut self) {
        self.gpu = None;
        super::release_columns(self);
    }
    fn gpu_bytes(&mut self) -> u64 {
        // The atlas is shared, so it is counted by whoever owns it.
        super::column_bytes(self)
    }
    fn columns(&mut self) -> Vec<&mut Column> {
        vec![
            &mut self.positions,
            &mut self.sizes,
            &mut self.rects,
            &mut self.angles,
            &mut self.colors,
            &mut self.shown,
        ]
    }
    fn extra_bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.gpu.as_ref().map(|g| &g.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::marks::{MarkView, Origin, TextureAtlas};

    fn view() -> MarkView {
        // Identity view-projection over 200×100: x, y in [-1, 1] across it.
        let mut m = [0.0f32; 16];
        m[0] = 1.0;
        m[5] = 1.0;
        m[10] = 1.0;
        m[15] = 1.0;
        MarkView {
            view_proj: m,
            projection_scale: 1.0,
            viewport: (200, 100),
            origin: Origin::ZERO,
            light_direction: [0.0, 1.0, 0.0],
            eye: [0.0, 0.0, 1.0, 0.0],
            tone_mapping: (crate::ToneMapping::None, 1.0),
            reversed_z: false,
            focal_distance: 0.0,
            clip_epsilon: 0.001,
            subpixel_snap: None,
            flip_y: false,
            swiftshader_sampling: false,
            swiftshader_depth: false,
            scene_lights: None,
        }
    }

    /// The view holds still while the icons move apart: placement runs again.
    #[test]
    fn moving_icons_are_placed_again_under_a_still_view() {
        let mut atlas = TextureAtlas::new(8);
        let rect = atlas.insert("dot", 2, 2, &[255; 16]).unwrap();
        let mut s = SpriteMarks::new(atlas.shared());
        s.collision = true;
        s.set_default_rect(rect);
        s.set_positions(&[[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]]);
        s.place(&view());
        assert_eq!(s.shown(), vec![true, false]);
        s.set_positions(&[[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]);
        s.place(&view());
        assert_eq!(s.shown(), vec![true, true]);
        // Grown to overlap again, through a style field.
        s.common.style.size_scale = 150.0;
        s.common.style.max_pixels = 1000.0;
        s.place(&view());
        assert_eq!(s.shown(), vec![true, false]);
    }
}
