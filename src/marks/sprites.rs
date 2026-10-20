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
    /// The image's alpha as a signed distance field (0.5 on the edge), drawn
    /// sharp at any size in the row's colour.
    DistanceField,
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
    sizes: Column,
    rects: Column,
    angles: Column,
    colors: Column,
    pub mode: SpriteMode,
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
    shown: Column,
    priorities: Vec<f32>,
    placed: Option<PlacedFor>,
    /// Bumped by every setter that moves, sizes or turns an icon.
    version: u64,
    /// Marks this layer's boxes in a shared index.
    owner: u64,
    gpu: Option<(u64, wgpu::BindGroup)>,
    atlas_size: (u32, u32),
}

impl SpriteMarks {
    pub fn new(atlas: SharedAtlas) -> Self {
        let mut sizes = Column::float(1);
        sizes.set_default_f32(&[1.0]);
        let mut common = MarkCommon::default();
        common.style.size_scale = 24.0;
        Self {
            common,
            atlas,
            positions: Column::new(ColumnFormat::PositionF64),
            sizes,
            rects: Column::float(4),
            angles: Column::float(1),
            colors: Column::new(ColumnFormat::Unorm8x4),
            mode: SpriteMode::Color,
            anchor: [0.0, 0.0],
            collision: false,
            collision_index: None,
            padding: 1.0,
            shown: {
                let mut c = Column::float(1);
                c.set_default_f32(&[1.0]);
                c
            },
            priorities: Vec::new(),
            placed: None,
            version: 0,
            owner: super::collision::next_owner(),
            gpu: None,
            atlas_size: (1, 1),
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
        self.sizes.set_f32(s);
    }
    pub fn update_sizes(&mut self, start: usize, s: &[f32]) {
        self.version += 1;
        self.sizes.update_f32(start, s);
    }
    /// Each row's image, as rects in the atlas.
    pub fn set_rects(&mut self, r: &[AtlasRect]) {
        self.version += 1;
        let flat: Vec<f32> = r.iter().flat_map(|r| r.to_array()).collect();
        self.rects.set_f32(&flat);
    }
    pub fn update_rects(&mut self, start: usize, r: &[AtlasRect]) {
        self.version += 1;
        let flat: Vec<f32> = r.iter().flat_map(|r| r.to_array()).collect();
        self.rects.update_f32(start, &flat);
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
        self.angles.set_f32(a);
    }
    pub fn update_angles(&mut self, start: usize, a: &[f32]) {
        self.version += 1;
        self.angles.update_f32(start, a);
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
            let mut h = self.sizes.row_f32(i)[0] * s.size_scale;
            if s.units == super::SizeUnits::World {
                h /= view.units_per_pixel(pos[i]).unwrap_or(1.0) as f32;
            }
            let h = h.clamp(s.min_pixels, s.max_pixels);
            let half = [h * rect[2] / rect[3] * 0.5, h * 0.5];
            // The anchor in screen space (y down), then the turned box's
            // bounds.
            let c = [
                p[0] - self.anchor[0] * half[0],
                p[1] + self.anchor[1] * half[1],
            ];
            let a = self.angles.row_f32(i)[0];
            let (sn, cs) = a.sin_cos();
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
        self.shown.set_f32(&shown);
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
        };
        (
            [mode, 0.0, 0.0, 0.0],
            [
                self.atlas_size.0 as f32,
                self.atlas_size.1 as f32,
                self.anchor[0],
                self.anchor[1],
            ],
            [0.0; 4],
        )
    }
    fn prepare(&mut self, ctx: &PrepareContext) {
        self.atlas_size = atlas_bind_group(ctx, &self.atlas, &mut self.gpu);
        if self.collision {
            self.place(ctx.view);
        } else if !self.shown.is_empty() {
            self.shown.set_f32(&[]);
            if let (Some(sh), Some(_)) = (&self.collision_index, self.placed) {
                // Its boxes would keep blocking others until the next clear.
                let mut index = sh.lock().unwrap();
                index.remove_owner(self.owner);
                index.invalidate();
            }
            self.placed = None;
        }
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
