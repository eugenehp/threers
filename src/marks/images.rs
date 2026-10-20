//! Images on quads in the world: raster tiles, overlays, photographs.

use super::atlas::{AtlasRect, SharedAtlas};
use super::sprites::atlas_bind_group;
use super::{Column, ColumnFormat, KindUniforms, Mark, MarkCommon, MarkKind, PrepareContext};

/// One quad: a corner, the edge vectors to its two neighbours, and the
/// image it shows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageQuad {
    /// The corner the image's top-left texel sits on, absolute.
    pub corner: [f64; 3],
    /// Along the image's top edge, to the top-right corner.
    pub right: [f32; 3],
    /// Down the image's left edge, to the bottom-left corner.
    pub down: [f32; 3],
    pub rect: AtlasRect,
}

impl ImageQuad {
    /// A north-up rectangle on the ground plane from `min` to `max`
    /// (`[x, y]` in the convention of [`geo`](super::geo)), at `height`.
    pub fn ground(min: [f64; 2], max: [f64; 2], height: f64, rect: AtlasRect) -> Self {
        Self {
            corner: super::geo::ground_to_world([min[0], max[1]], height),
            right: [(max[0] - min[0]) as f32, 0.0, 0.0],
            down: [0.0, 0.0, (max[1] - min[1]) as f32],
            rect,
        }
    }
}

/// Images from a [`TextureAtlas`](super::TextureAtlas) on quads in the
/// world — deck.gl's `BitmapLayer`, and the raster half of a tile layer.
///
/// Each row is an [`ImageQuad`]. Quads are drawn as a grid of
/// [`subdivisions`](Self::subdivisions)² cells, which only matters once a
/// hook or a projection bends them; flat quads need one.
///
/// Like [`PolygonMarks`](super::PolygonMarks), image quads are depth-biased
/// towards the camera so one lying on a ground mesh wins, and opaque by
/// default; transparent texels are not drawn or picked.
#[derive(Debug)]
pub struct ImageMarks {
    pub common: MarkCommon,
    atlas: SharedAtlas,
    corners: Column,
    rights: Column,
    downs: Column,
    rects: Column,
    colors: Column,
    /// Cells along each edge of every quad, `1..=64`.
    pub subdivisions: u32,
    /// A sphere as `(centre, radius)`: every grid point of every quad is
    /// pushed out onto it, so raster tiles bend with a globe. Use enough
    /// [`subdivisions`](Self::subdivisions) for the curvature.
    pub sphere: Option<([f64; 3], f64)>,
    sphere_rel: [f32; 4],
    gpu: Option<(u64, wgpu::BindGroup)>,
    atlas_size: (u32, u32),
}

impl ImageMarks {
    pub fn new(atlas: SharedAtlas) -> Self {
        let mut common = MarkCommon::default();
        common.style.depth_write = true;
        Self {
            common,
            atlas,
            corners: Column::new(ColumnFormat::PositionF64),
            rights: Column::float(3),
            downs: Column::float(3),
            rects: Column::float(4),
            colors: Column::new(ColumnFormat::Unorm8x4),
            subdivisions: 1,
            sphere: None,
            sphere_rel: [0.0; 4],
            gpu: None,
            atlas_size: (1, 1),
        }
    }

    pub fn atlas(&self) -> &SharedAtlas {
        &self.atlas
    }

    pub fn set_quads(&mut self, quads: &[ImageQuad]) {
        let corners: Vec<[f64; 3]> = quads.iter().map(|q| q.corner).collect();
        let rights: Vec<f32> = quads.iter().flat_map(|q| q.right).collect();
        let downs: Vec<f32> = quads.iter().flat_map(|q| q.down).collect();
        let rects: Vec<f32> = quads.iter().flat_map(|q| q.rect.to_array()).collect();
        self.corners.set_positions(&corners);
        self.rights.set_f32(&rights);
        self.downs.set_f32(&downs);
        self.rects.set_f32(&rects);
    }

    pub fn update_quads(&mut self, start: usize, quads: &[ImageQuad]) {
        let corners: Vec<[f64; 3]> = quads.iter().map(|q| q.corner).collect();
        let rights: Vec<f32> = quads.iter().flat_map(|q| q.right).collect();
        let downs: Vec<f32> = quads.iter().flat_map(|q| q.down).collect();
        let rects: Vec<f32> = quads.iter().flat_map(|q| q.rect.to_array()).collect();
        self.corners.update_positions(start, &corners);
        self.rights.update_f32(start, &rights);
        self.downs.update_f32(start, &downs);
        self.rects.update_f32(start, &rects);
    }

    /// sRGB tint per quad, straight alpha (white by default).
    pub fn set_colors(&mut self, c: &[[u8; 4]]) {
        self.colors.set_rgba(c);
    }
    pub fn update_colors(&mut self, start: usize, c: &[[u8; 4]]) {
        self.colors.update_rgba(start, c);
    }
}

impl Mark for ImageMarks {
    fn kind(&self) -> MarkKind {
        MarkKind::Image
    }
    fn common(&self) -> &MarkCommon {
        &self.common
    }
    fn common_mut(&mut self) -> &mut MarkCommon {
        &mut self.common
    }
    fn instance_count(&self) -> usize {
        self.corners.len()
    }
    fn vertices_per_instance(&self) -> u32 {
        let n = self.subdivisions.clamp(1, 64);
        6 * n * n
    }
    fn kind_uniforms(&self) -> KindUniforms {
        (
            [self.subdivisions.clamp(1, 64) as f32, 0.0, 0.0, 0.0],
            [self.atlas_size.0 as f32, self.atlas_size.1 as f32, 0.0, 0.0],
            self.sphere_rel,
        )
    }
    fn prepare(&mut self, ctx: &PrepareContext) {
        self.atlas_size = atlas_bind_group(ctx, &self.atlas, &mut self.gpu);
        self.sphere_rel = match self.sphere {
            Some((c, r)) => {
                let o = ctx.view.origin.position;
                [
                    (c[0] - o[0]) as f32,
                    (c[1] - o[1]) as f32,
                    (c[2] - o[2]) as f32,
                    r as f32,
                ]
            }
            None => [0.0; 4],
        };
    }
    fn release_gpu(&mut self) {
        self.gpu = None;
        super::release_columns(self);
    }
    fn gpu_bytes(&mut self) -> u64 {
        super::column_bytes(self)
    }
    fn columns(&mut self) -> Vec<&mut Column> {
        vec![
            &mut self.corners,
            &mut self.rights,
            &mut self.downs,
            &mut self.rects,
            &mut self.colors,
        ]
    }
    fn extra_bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.gpu.as_ref().map(|g| &g.1)
    }
}
