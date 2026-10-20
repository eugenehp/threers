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
}

impl PolygonFeature {
    pub fn new(outer: Vec<[f64; 2]>) -> Self {
        Self {
            outer,
            holes: Vec::new(),
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
    index: Vec<u32>,
    index_dirty: bool,
    index_gpu: Option<(wgpu::Buffer, u32)>,
    /// Each feature's vertices, `start..end`.
    ranges: Vec<(usize, usize)>,
    /// How much of a wall facing away from the light is lit: 1 is unlit.
    /// Roofs always show their colour exactly, so a choropleth reads true.
    pub ambient: f32,
    /// For a mesh built on a sphere: its centre, which extrusion rises away
    /// from. `None` rises along +Y.
    radial: Option<[f64; 3]>,
    /// `radial` relative to the view's origin, refreshed every frame.
    radial_rel: [f32; 4],
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
            index: Vec::new(),
            index_dirty: true,
            index_gpu: None,
            ranges: Vec::new(),
            ambient: 0.45,
            radial: None,
            radial_rel: [0.0; 4],
        }
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
        for slot in 0..2 {
            self.common.clear_custom(slot);
        }
        self.ranges = mesh
            .ranges
            .iter()
            .map(|&(a, b)| (a as usize, b as usize))
            .collect();
        self.index = mesh.index;
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
    pub fn update_colors(&mut self, start: usize, c: &[[u8; 4]]) {
        let v = self.expand(start, c, 1);
        let first = self.first_vertex(start);
        self.colors.update_rgba(first, &v);
    }
    pub fn set_default_color(&mut self, c: [u8; 4]) {
        self.colors.set_default_rgba(c);
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
        (
            [self.ambient.clamp(0.0, 1.0), 0.0, 0.0, 0.0],
            // style3: the sphere's centre, relative, and 1 to rise from it.
            self.radial_rel,
            [0.0; 4],
        )
    }

    fn prepare(&mut self, ctx: &PrepareContext) {
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
