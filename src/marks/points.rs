//! Circles, one per row.

use super::{Column, ColumnFormat, KindUniforms, Lighting, Mark, MarkCommon, MarkKind, ProceduralFrame};

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
    /// Per row: the radius, the stroke's width as a multiple of
    /// `stroke_width`, and the stroke's dash and gap.
    radii: Column,
    /// The parts of `radii` as set, empty when not per row.
    radius_rows: Vec<f32>,
    stroke_rows: Vec<f32>,
    dash_rows: Vec<[f32; 2]>,
    colors: Column,
    /// Per row: the stroke's colour; `stroke_color` where not set.
    stroke_colors: Column,
    /// Stroke width in pixels, centred on the radius, times each row's
    /// multiple ([`set_stroke_widths`](Self::set_stroke_widths)). 0 for
    /// none.
    pub stroke_width: f32,
    /// Stroke colour, sRGB bytes, of rows without their own
    /// ([`set_stroke_colors`](Self::set_stroke_colors)).
    pub stroke_color: [u8; 4],
    /// Draw only the stroke.
    pub hollow: bool,
    /// Lie flat on the ground (the plane of +Y) instead of facing the
    /// camera — deck.gl's `ScatterplotLayer` without `billboard`. A disc
    /// keeps the size it would have facing the camera at its centre, so
    /// it foreshortens with the view's tilt; it is drawn at the ground's
    /// own depth, so `depth_lift` does not apply.
    pub ground: bool,
    /// Drawn hard-edged as one triangle about the disc (its vertices at
    /// twice the radius) rather than a square: deck.gl's PointCloudLayer
    /// geometry, which a coarse rasterizer snaps differently.
    pub triangle: bool,
    /// Light each point by its normal ([`set_normals`](Self::set_normals);
    /// up, +Y, where not set), once at its centre — deck.gl's
    /// `PointCloudLayer`. Phong only: `pbr` and `scene` are not supported.
    pub lighting: Option<Lighting>,
    normals: Column,
    /// Per row, a procedural fill pattern ([`set_procedural`](Self::set_procedural)).
    procedural: Option<Vec<Option<ProceduralFrame>>>,
    /// World units one procedural pattern unit spans (before each row's
    /// scale).
    pub procedural_unit: f64,
    /// Pattern y runs along world +z (south on a map); else along -z.
    pub procedural_y_down: bool,
    procedural_built: Option<(u64, bool)>,
    /// Per row, the colour behind its pattern.
    backgrounds: Column,
}

impl Default for PointMarks {
    fn default() -> Self {
        Self::new()
    }
}

impl PointMarks {
    pub fn new() -> Self {
        let mut radii = Column::float(4);
        radii.set_default_f32(&[1.0, 1.0, 0.0, 0.0]);
        Self {
            common: MarkCommon::default(),
            positions: Column::new(ColumnFormat::PositionF64),
            radii,
            radius_rows: Vec::new(),
            stroke_rows: Vec::new(),
            dash_rows: Vec::new(),
            colors: Column::new(ColumnFormat::Unorm8x4),
            stroke_colors: Column::new(ColumnFormat::Unorm8x4),
            stroke_width: 0.0,
            stroke_color: [0, 0, 0, 255],
            hollow: false,
            ground: false,
            triangle: false,
            lighting: None,
            normals: {
                let mut n = Column::float(3);
                n.set_default_f32(&[0.0, 1.0, 0.0]);
                n
            },
            procedural: None,
            procedural_unit: 1.0,
            procedural_y_down: true,
            procedural_built: None,
            backgrounds: {
                let mut c = Column::new(ColumnFormat::Unorm8x4);
                c.set_default_rgba([0, 0, 0, 0]);
                c
            },
        }
    }

    /// The normal of rows with none of their own (up, +Y, unless set).
    pub fn set_default_normal(&mut self, n: [f32; 3]) {
        self.normals.set_default_f32(&n);
    }

    /// Each row's normal, for [`lighting`](Self::lighting); empty for the
    /// default everywhere. Need not be normalized.
    pub fn set_normals(&mut self, n: &[[f32; 3]]) {
        let flat: Vec<f32> = n.iter().flatten().copied().collect();
        self.normals.set_f32(&flat);
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
        self.radius_rows = r.to_vec();
        self.pack_radii();
    }
    pub fn update_radii(&mut self, start: usize, r: &[f32]) {
        let n = self.radius_rows.len().max(start + r.len());
        let whole = n != self.radius_rows.len() || self.radii.len() != n.max(self.stroke_rows.len()).max(self.dash_rows.len());
        self.radius_rows.resize(n, 1.0);
        self.radius_rows[start..start + r.len()].copy_from_slice(r);
        if whole {
            self.pack_radii();
            return;
        }
        let packed: Vec<f32> = (start..start + r.len()).flat_map(|i| self.packed(i)).collect();
        self.radii.update_f32(start, &packed);
    }
    /// Each row's stroke width as a multiple of
    /// [`stroke_width`](Self::stroke_width); empty for 1 everywhere.
    pub fn set_stroke_widths(&mut self, w: &[f32]) {
        self.stroke_rows = w.to_vec();
        self.pack_radii();
    }
    /// Each row's stroke colour, sRGB with straight alpha; empty for
    /// [`stroke_color`](Self::stroke_color) everywhere.
    pub fn set_stroke_colors(&mut self, c: &[[u8; 4]]) {
        self.stroke_colors.set_rgba(c);
    }
    /// Each row's stroke dashed: `[dash, gap]` in stroke widths along the
    /// circle through the stroke's middle, starting on the left; `[0, 0]`
    /// (or an empty slice) for solid. In a gap a filled circle shows its
    /// fill — deck.gl's `PathStyleExtension` on a `ScatterplotLayer`.
    pub fn set_dashes(&mut self, d: &[[f32; 2]]) {
        self.dash_rows = d.to_vec();
        self.pack_radii();
    }
    fn packed(&self, i: usize) -> [f32; 4] {
        let d = self.dash_rows.get(i).copied().unwrap_or([0.0, 0.0]);
        [self.radius_rows.get(i).copied().unwrap_or(1.0), self.stroke_rows.get(i).copied().unwrap_or(1.0), d[0], d[1]]
    }
    fn pack_radii(&mut self) {
        let n = self.radius_rows.len().max(self.stroke_rows.len()).max(self.dash_rows.len());
        if n == 0 {
            self.radii.set_f32(&[]);
            return;
        }
        let packed: Vec<f32> = (0..n).flat_map(|i| self.packed(i)).collect();
        self.radii.set_f32(&packed);
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

    /// Fill each row's disc, stroke included, with a procedural pattern
    /// (`None` in a row: solid) laid on the ground: its colour where the
    /// pattern is, the row's background elsewhere — deck.gl's
    /// `FillStyleExtension({proceduralPattern: true})` on a
    /// `ScatterplotLayer`. Uses both custom columns, so not with hooks that
    /// read them.
    pub fn set_procedural(&mut self, frames: Option<Vec<Option<ProceduralFrame>>>) {
        if frames.is_none() && self.procedural.is_some() {
            for slot in 0..2 {
                self.common.clear_custom(slot);
            }
        }
        self.procedural = frames;
        self.procedural_built = None;
    }

    /// Each row's colour behind its pattern, sRGB with straight alpha.
    pub fn set_pattern_backgrounds(&mut self, c: &[[u8; 4]]) {
        self.backgrounds.set_rgba(c);
    }
    /// The colour behind the pattern of rows without their own.
    pub fn set_default_pattern_background(&mut self, c: [u8; 4]) {
        self.backgrounds.set_default_rgba(c);
    }

    fn build_procedural_columns(&mut self) {
        let Some(frames) = &self.procedural else { return };
        let key = (self.procedural_unit.to_bits(), self.procedural_y_down);
        // Rebuilt too when something else cleared the custom columns.
        if self.procedural_built == Some(key) && self.common.has_custom(0) && self.common.has_custom(1) {
            return;
        }
        let n = frames.len();
        let (mut a, mut b) = (vec![0.0f32; n * 4], vec![0.0f32; n * 4]);
        for (k, f) in frames.iter().enumerate() {
            let Some(f) = f else { continue };
            let (p0, p1) = f.params(self.procedural_unit, self.procedural_y_down);
            a[k * 4..k * 4 + 4].copy_from_slice(&p0);
            b[k * 4..k * 4 + 4].copy_from_slice(&p1);
        }
        self.common.set_custom(0, 4, &a);
        self.common.set_custom(1, 4, &b);
        self.procedural_built = Some(key);
    }

    pub fn positions(&self) -> &Column {
        &self.positions
    }
}

impl PointMarks {
    fn lit(&self) -> Option<&Lighting> {
        self.lighting.as_ref().filter(|l| l.pbr.is_none() && !l.scene)
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
                if self.ground { 1.0 } else { 0.0 },
                0.0,
            ],
            // style3: the lighting's ambient, diffuse (0: unlit) and sRGB
            // flag; its lights go in stroke and extra.
            // w: the triangle.
            match self.lighting.as_ref().filter(|l| l.pbr.is_none() && !l.scene) {
                Some(l) => [l.ambient, l.diffuse.max(1e-6), if l.srgb { 1.0 } else { 0.0 }, if self.triangle { 1.0 } else { 0.0 }],
                None => [0.0, 0.0, 0.0, if self.triangle { 1.0 } else { 0.0 }],
            },
            Lighting::uniforms(self.lit()).1,
        )
    }

    fn kind_extra(&self) -> [f32; 4] {
        Lighting::uniforms(self.lit()).2
    }

    fn kind_material(&self) -> [f32; 4] {
        Lighting::material(self.lit())
    }
    fn prepare(&mut self, _ctx: &super::PrepareContext) {
        self.build_procedural_columns();
    }

    /// Bit 0: a procedural pattern.
    fn kind_flags(&self) -> u32 {
        u32::from(self.procedural.is_some())
    }

    fn columns(&mut self) -> Vec<&mut Column> {
        // Rows without their own stroke colour take the layer's.
        self.stroke_colors.set_default_rgba(self.stroke_color);
        vec![&mut self.positions, &mut self.radii, &mut self.colors, &mut self.stroke_colors, &mut self.normals, &mut self.backgrounds]
    }
}
