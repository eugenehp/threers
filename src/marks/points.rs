//! Circles, one per row.

use super::{Column, ColumnFormat, KindUniforms, Mark, MarkCommon, MarkKind};

/// Antialiased circles, sized in pixels or world units, with an optional
/// stroke — deck.gl's `ScatterplotLayer`.
///
/// Radius is per instance (the `radii` column) times
/// [`MarkStyle::size_scale`](super::MarkStyle::size_scale), so an empty radius
/// column with a scale of 4 draws every point four pixels across in radius.
#[derive(Debug)]
pub struct PointMarks {
    pub common: MarkCommon,
    positions: Column,
    radii: Column,
    colors: Column,
    /// Stroke width in pixels, centred on the radius. 0 for none.
    pub stroke_width: f32,
    /// Stroke colour, sRGB bytes.
    pub stroke_color: [u8; 4],
    /// Draw only the stroke.
    pub hollow: bool,
}

impl Default for PointMarks {
    fn default() -> Self {
        Self::new()
    }
}

impl PointMarks {
    pub fn new() -> Self {
        let mut radii = Column::float(1);
        radii.set_default_f32(&[1.0]);
        Self {
            common: MarkCommon::default(),
            positions: Column::new(ColumnFormat::PositionF64),
            radii,
            colors: Column::new(ColumnFormat::Unorm8x4),
            stroke_width: 0.0,
            stroke_color: [0, 0, 0, 255],
            hollow: false,
        }
    }

    /// Absolute positions — the row count of the layer.
    pub fn set_positions(&mut self, p: &[[f64; 3]]) {
        self.common.local_origin = None;
        self.positions.set_positions(p);
    }
    /// Positions relative to `origin` — half the memory of absolute ones,
    /// and what a tile decoder produces. See
    /// [`ColumnFormat::PositionLocal`](super::ColumnFormat::PositionLocal).
    pub fn set_local_positions(&mut self, origin: [f64; 3], p: &[[f32; 3]]) {
        self.common.local_origin = Some(origin);
        self.positions.set_local_positions(p);
    }
    pub fn update_local_positions(&mut self, start: usize, p: &[[f32; 3]]) {
        self.positions.update_local_positions(start, p);
    }
    pub fn update_positions(&mut self, start: usize, p: &[[f64; 3]]) {
        self.positions.update_positions(start, p);
    }
    pub fn set_radii(&mut self, r: &[f32]) {
        self.radii.set_f32(r);
    }
    pub fn update_radii(&mut self, start: usize, r: &[f32]) {
        self.radii.update_f32(start, r);
    }
    /// sRGB colours with straight alpha.
    pub fn set_colors(&mut self, c: &[[u8; 4]]) {
        self.colors.set_rgba(c);
    }
    pub fn update_colors(&mut self, start: usize, c: &[[u8; 4]]) {
        self.colors.update_rgba(start, c);
    }
    /// The colour of rows with no colour of their own.
    pub fn set_default_color(&mut self, c: [u8; 4]) {
        self.colors.set_default_rgba(c);
    }

    pub fn positions(&self) -> &Column {
        &self.positions
    }
}

pub(crate) fn srgb_u8_to_linear(c: [u8; 4]) -> [f32; 4] {
    let f = |v: u8| {
        let s = v as f32 / 255.0;
        if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    };
    [f(c[0]), f(c[1]), f(c[2]), c[3] as f32 / 255.0]
}

impl Mark for PointMarks {
    fn kind(&self) -> MarkKind {
        MarkKind::Point
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
        (
            [
                self.stroke_width.max(0.0),
                if self.hollow { 1.0 } else { 0.0 },
                0.0,
                0.0,
            ],
            [0.0; 4],
            srgb_u8_to_linear(self.stroke_color),
        )
    }
    fn columns(&mut self) -> Vec<&mut Column> {
        vec![&mut self.positions, &mut self.radii, &mut self.colors]
    }
}
