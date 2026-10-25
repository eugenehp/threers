//! Filled and extruded polygons, many to a draw.

use super::{
    Column, ColumnFormat, KindUniforms, Mark, MarkCommon, MarkKind, PolygonMesh, PrepareContext,
};

/// One polygon: an outer ring and any holes, as `[x, y]` on the ground, the
/// way GeoJSON gives them. A ring may repeat its first point at the end.
///
/// The ground plane maps `[x, y]` to world `(x, base, -y)`, as
/// [`PolygonGeometry`](crate::PolygonGeometry) does: with Y up, `y` (north)
/// runs away from a camera looking down −Z.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PolygonFeature {
    pub outer: Vec<[f64; 2]>,
    pub holes: Vec<Vec<[f64; 2]>>,
    /// The roof's triangles, given rather than found: indices into the
    /// outer ring then each hole, closing points dropped (deck.gl's binary
    /// `indices`). Not used on a curved ground, whose rings gain points.
    pub triangles: Option<Vec<[u32; 3]>>,
}

impl PolygonFeature {
    pub fn new(outer: Vec<[f64; 2]>) -> Self {
        Self {
            outer,
            holes: Vec::new(),
            triangles: None,
        }
    }
}

/// Polygons with a colour and an extrusion height each — deck.gl's
/// `PolygonLayer`, and the fill of its `GeoJsonLayer`.
///
/// Every feature is triangulated once, on the CPU, into one shared mesh: the
/// roof, and walls that hang from it. Each vertex carries its feature's
/// colour, elevation and custom values, so a whole layer is one draw however
/// many features it has, and changing one feature's values re-uploads only
/// its own vertices.
///
/// Elevation is per feature (the `elevations` column) times
/// [`MarkStyle::size_scale`](super::MarkStyle::size_scale), always in world
/// units. At 0 the walls have no height and the layer is a flat fill.
/// Feature `k` is row `k` for picking, and the `index` of the
/// [hooks'](super::ShaderHooks) `MarkInstance`, whose `size` is the
/// elevation. Custom values are per feature too, through
/// [`set_feature_custom`](Self::set_feature_custom).
///
/// Flat fills are drawn with a depth bias towards the camera, so a fill lying
/// on a ground mesh at the same height draws over it rather than fighting it.
/// Outlines are a [`SegmentMarks`](super::SegmentMarks) of the same rings.
#[derive(Debug)]
pub struct PolygonMarks {
    pub common: MarkCommon,
    positions: Column,
    /// Normal, and 1 for a vertex that rises with the elevation.
    normal_top: Column,
    feature: Column,
    elevations: Column,
    colors: Column,
    /// Per vertex, its feature's pattern background.
    backgrounds: Column,
    index: Vec<u32>,
    index_dirty: bool,
    index_gpu: Option<(wgpu::Buffer, u32)>,
    /// Each feature's vertices, `start..end`.
    ranges: Vec<(usize, usize)>,
    /// Each vertex's point in its feature ([`PolygonMesh::source`]).
    source: Vec<u32>,
    /// How much of a wall facing away from the light is lit: 1 is unlit.
    /// Roofs always show their colour exactly, so a choropleth reads true.
    pub ambient: f32,
    /// Light as deck.gl does instead (`ambient` is then unused, and roofs
    /// are lit too).
    pub lighting: Option<super::Lighting>,
    /// For a mesh built on a sphere: its centre, which extrusion rises away
    /// from. `None` rises along +Y.
    radial: Option<[f64; 3]>,
    /// `radial` relative to the view's origin, refreshed every frame.
    radial_rel: [f32; 4],
    pattern: Option<PolygonPattern>,
    /// World units one pattern texel spans (times each frame's scale):
    /// fixed for patterns sized in metres, set each frame for ones sized
    /// on screen.
    pub pattern_unit: f64,
    /// The ground point, `[x, z]` in world units, where pattern tiles start
    /// (the atlas image's top left).
    pub pattern_origin: [f64; 2],
    /// What the pattern columns were built for: (unit, origin, tile origin,
    /// atlas version), or none to build them.
    pattern_built: Option<(u64, [u64; 3], [u64; 2], u64)>,
    pattern_gpu: Option<(u64, wgpu::BindGroup)>,
    /// Per feature, a procedural pattern ([`set_procedural`](Self::set_procedural)).
    procedural: Option<Vec<Option<ProceduralFrame>>>,
    /// World units one procedural pattern unit spans (before each feature's
    /// scale): deck.gl's `fillPatternSizeUnits`, set each frame for pixels.
    pub procedural_unit: f64,
    /// Pattern y (procedural or image) runs along world +z (south on a
    /// map, down in a y-down view); otherwise along -z.
    pub procedural_y_down: bool,
    /// Where procedural pattern space starts: added to the ground point,
    /// in world units along the pattern's x and y (deck.gl measures from
    /// its common space's corner).
    pub procedural_offset: [f64; 2],
    procedural_built: Option<(u64, u64)>,
}

/// A fill pattern for [`PolygonMarks`] — deck.gl's `FillStyleExtension`,
/// MapLibre's `fill-pattern`: each feature's fill tiled with an image from
/// `atlas`, on the ground's x and z.
#[derive(Debug, Clone)]
pub struct PolygonPattern {
    pub atlas: super::SharedAtlas,
    /// The image's alpha masks the fill colour; otherwise the image's own
    /// colours show.
    pub mask: bool,
    /// Per feature: the atlas image, its scale (texels per tile texel), and
    /// its offset in tiles. `None`, or a name not in the atlas: a plain fill.
    pub frames: Vec<Option<PatternFrame>>,
}

/// A pattern drawn by the shader rather than from an image — deck.gl's
/// `FillStyleExtension({proceduralPattern: true})`. Sizes are in pattern
/// units ([`PolygonMarks::procedural_unit`] world units each).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProceduralShape {
    /// Parallel lines at `angle` (degrees), `stroke_width` wide, `gaps`
    /// apart edge to edge (two gaps alternate when `alternating`).
    Hatch { angle: f32, stroke_width: f32, gaps: [f32; 2], alternating: bool },
    /// Two sets of lines.
    CrossHatch { angles: [f32; 2], stroke_width: f32, gap: f32 },
    /// Dots of `radius`, `gap` apart edge to edge, on a grid turned by
    /// `angle` with its second axis tilted `skew` towards the first.
    Dots { radius: f32, gap: f32, angle: f32, skew: f32 },
}

/// One feature's procedural pattern and its scale (pattern units per
/// pattern unit, deck.gl's `getFillPatternScale`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProceduralFrame {
    pub shape: ProceduralShape,
    pub scale: f32,
}

impl ProceduralFrame {
    /// The shader's two columns: (shape, size, gaps) and (angles in
    /// radians, count, world units per pattern unit, negative when pattern
    /// y runs along -z).
    pub(crate) fn params(&self, unit: f64, y_down: bool) -> ([f32; 4], [f32; 4]) {
        let rad = |d: f32| d.to_radians();
        let (shape, size, gaps, angles, count) = match self.shape {
            ProceduralShape::Hatch { angle, stroke_width, gaps, alternating } => {
                (1.0, stroke_width, gaps, [rad(angle), 0.0], if alternating { 2.0 } else { 1.0 })
            }
            ProceduralShape::CrossHatch { angles, stroke_width, gap } => (2.0, stroke_width, [gap, gap], [rad(angles[0]), rad(angles[1])], 2.0),
            ProceduralShape::Dots { radius, gap, angle, skew } => (3.0, radius, [gap, gap], [rad(angle), rad(skew)], 2.0),
        };
        let unit = (unit * self.scale as f64) as f32 * if y_down { 1.0 } else { -1.0 };
        ([shape, size, gaps[0], gaps[1]], [angles[0], angles[1], count, unit])
    }
}

/// One feature's pattern.
#[derive(Debug, Clone, PartialEq)]
pub struct PatternFrame {
    pub name: String,
    pub scale: f32,
    pub offset: [f32; 2],
}

impl Default for PolygonMarks {
    fn default() -> Self {
        Self::new()
    }
}

impl PolygonMarks {
    pub fn new() -> Self {
        let mut common = MarkCommon::default();
        common.style.depth_write = true;
        Self {
            common,
            positions: Column::new(ColumnFormat::PositionF64),
            normal_top: Column::float(4),
            feature: Column::float(1),
            elevations: Column::float(1),
            colors: Column::new(ColumnFormat::Unorm8x4),
            backgrounds: {
                let mut c = Column::new(ColumnFormat::Unorm8x4);
                c.set_default_rgba([0, 0, 0, 0]);
                c
            },
            index: Vec::new(),
            index_dirty: true,
            index_gpu: None,
            ranges: Vec::new(),
            source: Vec::new(),
            ambient: 0.45,
            lighting: None,
            radial: None,
            radial_rel: [0.0; 4],
            pattern: None,
            pattern_unit: 1.0,
            pattern_origin: [0.0, 0.0],
            pattern_built: None,
            pattern_gpu: None,
            procedural: None,
            procedural_unit: 1.0,
            procedural_y_down: true,
            procedural_offset: [0.0, 0.0],
            procedural_built: None,
        }
    }

    /// Tile each feature's fill with an image ([`PolygonPattern`]), or
    /// stop. Patterns take the two custom slots, so they do not combine
    /// with hooks that read custom values.
    pub fn set_pattern(&mut self, pattern: Option<PolygonPattern>) {
        if pattern.is_none() && self.pattern.is_some() {
            self.common.clear_custom(0);
            self.common.clear_custom(1);
        }
        self.pattern = pattern;
        self.pattern_built = None;
        self.pattern_gpu = None;
    }

    pub fn pattern(&self) -> Option<&PolygonPattern> {
        self.pattern.as_ref()
    }

    /// Draw each feature's fill as a procedural pattern (`None` for a plain
    /// fill): the fill's colour where the pattern is, clear elsewhere. Uses
    /// both custom columns, so not with hooks that read them.
    pub fn set_procedural(&mut self, frames: Option<Vec<Option<ProceduralFrame>>>) {
        if frames.is_none() && self.procedural.is_some() {
            for slot in 0..2 {
                self.common.clear_custom(slot);
            }
        }
        self.procedural = frames;
        self.procedural_built = None;
    }

    /// The per-vertex procedural columns: each feature's shape (type, size,
    /// gaps) and (angles, count, world units per pattern unit, signed by y).
    fn build_procedural_columns(&mut self) {
        let Some(frames) = &self.procedural else { return };
        let key = (self.procedural_unit.to_bits(), u64::from(self.procedural_y_down));
        // Rebuilt too when something else cleared the custom columns.
        if self.procedural_built == Some(key) && self.common.has_custom(0) && self.common.has_custom(1) {
            return;
        }
        let n = self.ranges.len();
        let mut a = vec![0.0f32; n * 4];
        let mut b = vec![0.0f32; n * 4];
        for (k, f) in frames.iter().enumerate().take(n) {
            let Some(f) = f else { continue };
            let (p0, p1) = f.params(self.procedural_unit, self.procedural_y_down);
            a[k * 4..k * 4 + 4].copy_from_slice(&p0);
            b[k * 4..k * 4 + 4].copy_from_slice(&p1);
        }
        self.set_feature_custom(0, 4, &a);
        self.set_feature_custom(1, 4, &b);
        self.procedural_built = Some(key);
    }

    /// The per-vertex pattern columns: each feature's atlas frame (uv) and
    /// its tile size with the layer's origin reduced to within a tile — in
    /// f64 here, so the shader's f32 never sees large coordinates.
    fn build_pattern_columns(&mut self) {
        let Some(p) = &self.pattern else { return };
        let atlas = p.atlas.read().unwrap();
        let (aw, ah) = atlas.size();
        let version = atlas.version();
        let origin = self.common.local_origin.unwrap_or([0.0; 3]);
        let key = (self.pattern_unit.to_bits(), origin.map(f64::to_bits), self.pattern_origin.map(f64::to_bits), version ^ (u64::from(self.procedural_y_down) << 63));
        if self.pattern_built == Some(key) {
            return;
        }
        let n = self.ranges.len();
        let mut frames = vec![0.0f32; n * 4];
        let mut tiles = vec![0.0f32; n * 4];
        for (k, f) in p.frames.iter().enumerate().take(n) {
            let Some(f) = f else { continue };
            let Some(r) = atlas.get(&f.name) else { continue };
            frames[k * 4..k * 4 + 4].copy_from_slice(&[
                r.x as f32 / aw.max(1) as f32,
                r.y as f32 / ah.max(1) as f32,
                r.width as f32 / aw.max(1) as f32,
                r.height as f32 / ah.max(1) as f32,
            ]);
            let (px, pz) = (
                r.width as f64 * f.scale as f64 * self.pattern_unit,
                r.height as f64 * f.scale as f64 * self.pattern_unit,
            );
            if px <= 0.0 || pz <= 0.0 {
                continue;
            }
            let reduce = |o: f64, period: f64, offset: f32| (o.rem_euclid(period) + offset as f64 * period) as f32;
            let (tx, tz) = (origin[0] - self.pattern_origin[0], origin[2] - self.pattern_origin[1]);
            // A tile's y runs along -z when the pattern's y does (a negative
            // height turns it round in the shader).
            let pz_signed = if self.procedural_y_down { pz } else { -pz };
            tiles[k * 4..k * 4 + 4].copy_from_slice(&[px as f32, pz_signed as f32, reduce(tx, px, f.offset[0]), reduce(tz, pz, f.offset[1])]);
        }
        drop(atlas);
        self.set_feature_custom(0, 4, &frames);
        self.set_feature_custom(1, 4, &tiles);
        self.pattern_built = Some(key);
    }

    /// Replace every feature, with its ground at height `base`. Per-feature
    /// values set before this are dropped. Triangulates on this thread; for
    /// a tile decoded elsewhere, build a [`PolygonMesh`] there and hand it to
    /// [`set_mesh`](Self::set_mesh).
    pub fn set_polygons(&mut self, features: &[PolygonFeature], base: f64) {
        self.set_mesh(PolygonMesh::build(features, base));
    }

    /// Upload a mesh built by [`PolygonMesh::build`] — on this thread or any
    /// other. Per-feature values set before this are dropped.
    pub fn set_mesh(&mut self, mesh: PolygonMesh) {
        self.common.local_origin = Some(mesh.origin);
        self.positions.set_local_positions(&mesh.positions);
        self.normal_top
            .set_f32(bytemuck::cast_slice::<[f32; 4], f32>(&mesh.normal_top));
        let fid: Vec<f32> = mesh.feature.iter().map(|&f| f as f32).collect();
        self.feature.set_f32(&fid);
        self.elevations.set_f32(&[]);
        self.colors.set_rgba(&[]);
        self.backgrounds.set_rgba(&[]);
        for slot in 0..2 {
            self.common.clear_custom(slot);
        }
        // A pattern's columns follow the new features.
        self.pattern_built = None;
        self.procedural_built = None;
        self.ranges = mesh
            .ranges
            .iter()
            .map(|&(a, b)| (a as usize, b as usize))
            .collect();
        self.index = mesh.index;
        self.source = mesh.source;
        self.radial = mesh.radial;
        self.index_dirty = true;
    }

    pub fn feature_count(&self) -> usize {
        self.ranges.len()
    }

    /// Each feature's value, repeated over its vertices.
    fn expand<T: Copy>(&self, start: usize, values: &[T], components: usize) -> Vec<T> {
        let mut out = Vec::new();
        for (k, chunk) in values.chunks(components).enumerate() {
            let Some(&(a, b)) = self.ranges.get(start + k) else {
                break;
            };
            for _ in a..b {
                out.extend_from_slice(chunk);
            }
        }
        out
    }

    /// The first vertex of feature `k`.
    fn first_vertex(&self, k: usize) -> usize {
        self.ranges.get(k).map_or(0, |r| r.0)
    }

    /// Extrusion height per feature.
    pub fn set_elevations(&mut self, e: &[f32]) {
        let v = self.expand(0, e, 1);
        self.elevations.set_f32(&v);
    }
    pub fn update_elevations(&mut self, start: usize, e: &[f32]) {
        let v = self.expand(start, e, 1);
        let first = self.first_vertex(start);
        self.elevations.update_f32(first, &v);
    }
    /// sRGB colour per feature, straight alpha.
    pub fn set_colors(&mut self, c: &[[u8; 4]]) {
        let v = self.expand(0, c, 1);
        self.colors.set_rgba(&v);
    }
    /// sRGB colour per point of each feature — its outer ring then holes,
    /// closing points dropped — blended across the triangles between them
    /// (deck.gl's binary per-vertex `getFillColor`). A vertex with no point
    /// of its own, or past the colours given, takes its feature's first.
    pub fn set_point_colors(&mut self, colors: &[Vec<[u8; 4]>]) {
        let mut v = Vec::with_capacity(self.source.len());
        for (k, &(a, b)) in self.ranges.iter().enumerate() {
            let c = colors.get(k).map_or(&[][..], |c| c.as_slice());
            for i in a..b {
                let s = self.source.get(i).copied().unwrap_or(u32::MAX) as usize;
                v.push(c.get(s).or(c.first()).copied().unwrap_or([0, 0, 0, 255]));
            }
        }
        self.colors.set_rgba(&v);
    }
    pub fn update_colors(&mut self, start: usize, c: &[[u8; 4]]) {
        let v = self.expand(start, c, 1);
        let first = self.first_vertex(start);
        self.colors.update_rgba(first, &v);
    }
    pub fn set_default_color(&mut self, c: [u8; 4]) {
        self.colors.set_default_rgba(c);
    }
    /// Per feature, the colour behind its fill pattern (sRGB, straight
    /// alpha; clear for none): deck.gl's `getFillPatternBackgroundColor`.
    pub fn set_pattern_backgrounds(&mut self, c: &[[u8; 4]]) {
        let v = self.expand(0, c, 1);
        self.backgrounds.set_rgba(&v);
    }
    pub fn set_default_pattern_background(&mut self, c: [u8; 4]) {
        self.backgrounds.set_default_rgba(c);
    }
    /// A custom column, per feature — what
    /// [`MarkCommon::set_custom`](super::MarkCommon::set_custom) is for the
    /// other marks. Use this one here: the GPU reads custom values per vertex.
    pub fn set_feature_custom(&mut self, slot: usize, components: u32, values: &[f32]) {
        let v = self.expand(0, values, components as usize);
        self.common.set_custom(slot, components, &v);
    }
    pub fn update_feature_custom(&mut self, slot: usize, start: usize, values: &[f32]) {
        let components = self.common.custom(slot).map_or(1, |c| match c.format() {
            ColumnFormat::Float(n) => n as usize,
            _ => 1,
        });
        let v = self.expand(start, values, components);
        let first = self.first_vertex(start);
        self.common.update_custom(slot, first, &v);
    }

    /// Vertices in the shared mesh, roofs and walls.
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// The layer as an ordinary geometry for the scene renderer — lit by its
    /// lights, casting and receiving shadows, under its post effects — with
    /// positions relative to `origin`, each feature extruded by its elevation
    /// as the marks shader does, its normals, and its colour as the linear
    /// `"color"` attribute. Pair it with a material of white colour; the
    /// vertex colours tint it.
    ///
    /// A snapshot: later changes to the layer are not seen.
    pub fn to_buffer_geometry(&self, origin: &super::Origin) -> crate::BufferGeometry {
        use crate::core::BufferAttribute;
        let n = self.positions.len();
        let abs = self
            .positions
            .positions_f64(self.common.local_origin)
            .unwrap_or_default();
        let o = origin.position;
        let scale = self.common.style.size_scale as f64;
        let mut pos = Vec::with_capacity(n * 3);
        let mut nor = Vec::with_capacity(n * 3);
        let mut col = Vec::with_capacity(n * 4);
        for (i, p) in abs.iter().enumerate() {
            let nt = self.normal_top.row_f32(i);
            let rise = self.elevations.row_f32(i)[0] as f64 * scale * nt[3] as f64;
            let up = match self.radial {
                Some(c) => {
                    let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
                    let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-12);
                    [d[0] / l, d[1] / l, d[2] / l]
                }
                None => [0.0, 1.0, 0.0],
            };
            for k in 0..3 {
                pos.push((p[k] + up[k] * rise - o[k]) as f32);
            }
            nor.extend_from_slice(&nt[..3]);
            let c = self.colors.row_f32(i);
            let rgba: [u8; 4] = bytemuck::cast([c[0]]);
            col.extend_from_slice(&super::points::srgb_u8_to_linear(rgba));
        }
        let mut g = crate::BufferGeometry::new();
        g.set_attribute("position", BufferAttribute::new(pos, 3));
        g.set_attribute("normal", BufferAttribute::new(nor, 3));
        g.set_attribute("color", BufferAttribute::new(col, 4));
        g.set_index(self.index.clone());
        g
    }
}

impl Mark for PolygonMarks {
    fn kind(&self) -> MarkKind {
        MarkKind::Polygon
    }
    fn common(&self) -> &MarkCommon {
        &self.common
    }
    fn common_mut(&mut self) -> &mut MarkCommon {
        &mut self.common
    }
    /// One draw of the whole mesh.
    fn instance_count(&self) -> usize {
        usize::from(!self.index.is_empty())
    }
    fn rows(&self) -> usize {
        self.positions.len()
    }
    fn per_vertex(&self) -> bool {
        true
    }
    fn kind_uniforms(&self) -> KindUniforms {
        let (yz, l0, _) = super::Lighting::uniforms(self.lighting.as_ref());
        let ambient = self.lighting.map_or(self.ambient.clamp(0.0, 1.0), |l| l.ambient);
        (
            [ambient, yz[0], yz[1], 0.0],
            // style3: the sphere's centre, relative, and 1 to rise from it;
            // on flat ground, xy the procedural pattern's offset.
            if self.radial.is_some() {
                self.radial_rel
            } else {
                [self.procedural_offset[0] as f32, self.procedural_offset[1] as f32, 0.0, 0.0]
            },
            l0,
        )
    }

    fn kind_extra(&self) -> [f32; 4] {
        super::Lighting::uniforms(self.lighting.as_ref()).2
    }

    fn kind_material(&self) -> [f32; 4] {
        let mut m = super::Lighting::material(self.lighting.as_ref());
        m[2] = if self.pattern.as_ref().is_some_and(|p| p.mask) { 1.0 } else { 0.0 };
        m
    }

    fn patterned(&self) -> bool {
        self.pattern.is_some()
    }

    /// Bit 0: a procedural pattern.
    fn kind_flags(&self) -> u32 {
        u32::from(self.procedural.is_some())
    }

    fn extra_bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.pattern_gpu.as_ref().map(|g| &g.1)
    }

    fn prepare(&mut self, ctx: &PrepareContext) {
        self.build_procedural_columns();
        if let Some(atlas) = self.pattern.as_ref().map(|p| p.atlas.clone()) {
            super::sprites::atlas_bind_group(ctx, &atlas, &mut self.pattern_gpu);
            self.build_pattern_columns();
        }
        self.radial_rel = match self.radial {
            Some(c) => {
                let o = ctx.view.origin.position;
                [
                    (c[0] - o[0]) as f32,
                    (c[1] - o[1]) as f32,
                    (c[2] - o[2]) as f32,
                    1.0,
                ]
            }
            None => [0.0; 4],
        };
        if !self.index_dirty || self.index.is_empty() {
            return;
        }
        let b = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threers polygon marks indices"),
            size: (self.index.len() * 4) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue
            .write_buffer(&b, 0, bytemuck::cast_slice(&self.index));
        self.index_gpu = Some((b, self.index.len() as u32));
        self.index_dirty = false;
    }
    fn release_gpu(&mut self) {
        self.index_gpu = None;
        self.pattern_gpu = None;
        self.index_dirty = true;
        super::release_columns(self);
    }
    fn gpu_bytes(&mut self) -> u64 {
        super::column_bytes(self) + self.index_gpu.as_ref().map_or(0, |(b, _)| b.size())
    }
    fn columns(&mut self) -> Vec<&mut Column> {
        vec![
            &mut self.positions,
            &mut self.normal_top,
            &mut self.feature,
            &mut self.elevations,
            &mut self.colors,
            &mut self.backgrounds,
        ]
    }
    fn index_buffer(&self) -> Option<(&wgpu::Buffer, u32)> {
        self.index_gpu.as_ref().map(|(b, n)| (b, *n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x: f64, y: f64, s: f64) -> Vec<[f64; 2]> {
        vec![[x, y], [x + s, y], [x + s, y + s], [x, y + s], [x, y]]
    }

    #[test]
    fn features_get_contiguous_vertices_and_values() {
        let mut p = PolygonMarks::new();
        let mut holed = PolygonFeature::new(square(0.0, 0.0, 10.0));
        holed.holes.push(square(4.0, 4.0, 2.0));
        p.set_polygons(&[PolygonFeature::new(square(20.0, 0.0, 5.0)), holed], 0.0);
        assert_eq!(p.feature_count(), 2);
        // A square: 4 roof points and 4 wall quads. With a hole: 8 and 8.
        assert_eq!(p.ranges, vec![(0, 4 + 16), (20, 20 + 8 + 32)]);
        // Roofs: 2 triangles and 8; walls: 2 a quad.
        assert_eq!(p.index.len(), 3 * ((2 + 4 * 2) + (8 + 8 * 2)));

        p.set_colors(&[[1, 0, 0, 255], [2, 0, 0, 255]]);
        let c = p.colors.bytes();
        assert_eq!(c.len(), 4 * 60);
        assert_eq!(c[4 * 19], 1);
        assert_eq!(c[4 * 20], 2);

        // Updating feature 1 rewrites its vertices and no others.
        p.colors.mark_uploaded();
        p.update_colors(1, &[[3, 0, 0, 255]]);
        assert_eq!(p.colors.pending_upload(), Some(4 * 20..4 * 60));
    }

    #[test]
    fn walls_face_out_of_the_solid_whatever_the_winding() {
        for ring in [square(0.0, 0.0, 1.0), {
            let mut r = square(0.0, 0.0, 1.0);
            r.reverse();
            r
        }] {
            let mut p = PolygonMarks::new();
            let mut f = PolygonFeature::new(ring);
            f.holes.push(square(0.25, 0.25, 0.5));
            p.set_polygons(&[f], 0.0);
            let pos: Vec<f32> = bytemuck::pod_collect_to_vec(p.positions.bytes());
            let o = p.common.local_origin.unwrap();
            let nt: Vec<f32> = bytemuck::pod_collect_to_vec(p.normal_top.bytes());
            // Each wall vertex's normal points away from the square's centre
            // on the outer ring and towards it on the hole.
            for v in 8..p.vertex_count() {
                let (x, z) = (pos[v * 3] + o[0] as f32, pos[v * 3 + 2] + o[2] as f32);
                let n = [nt[v * 4], nt[v * 4 + 2]];
                let (cx, cz) = (x - 0.5, z + 0.5);
                let outward = n[0] * cx + n[1] * cz > 0.0;
                let on_outer = cx.abs() > 0.4 || cz.abs() > 0.4;
                assert_eq!(outward, on_outer, "vertex {v} at ({x}, {z}) normal {n:?}");
            }
        }
    }
}
