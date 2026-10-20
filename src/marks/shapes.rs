//! Any geometry, drawn once per row.

use super::{
    Column, ColumnFormat, InstancedGeometry, KindUniforms, Mark, MarkCommon, MarkKind,
    PrepareContext,
};
use crate::core::BufferGeometry;

/// The uploaded mesh: vertices, vertex count, and indices with their count.
type MeshGpu = (wgpu::Buffer, u32, Option<(wgpu::Buffer, u32)>);

/// One mesh drawn at every row's position, scaled per row and lit by the
/// view's light — deck.gl's `ColumnLayer` and `SimpleMeshLayer`.
///
/// A unit box with its base at the origin and a per-row height makes a bar
/// chart on a map; a hexagonal prism makes a hexbin. The geometry's
/// `position` and `normal` attributes are used, and its index if it has one;
/// without normals it is drawn unlit.
///
/// Shapes write depth by default, so they hide each other and anything drawn
/// after them, as solid objects should.
#[derive(Debug)]
pub struct ShapeMarks {
    pub common: MarkCommon,
    /// position + normal, interleaved, and the index.
    mesh: (Vec<f32>, Option<Vec<u32>>),
    mesh_dirty: bool,
    mesh_gpu: Option<MeshGpu>,
    positions: Column,
    scales: Column,
    colors: Column,
    /// How much of the colour is lit regardless of the light: 1 is unlit.
    pub ambient: f32,
}

impl ShapeMarks {
    pub fn new(geometry: &BufferGeometry) -> Self {
        let mut scales = Column::float(3);
        scales.set_default_f32(&[1.0, 1.0, 1.0]);
        let mut common = MarkCommon::default();
        common.style.depth_write = true;
        let mut s = Self {
            common,
            mesh: (Vec::new(), None),
            mesh_dirty: true,
            mesh_gpu: None,
            positions: Column::new(ColumnFormat::PositionF64),
            scales,
            colors: Column::new(ColumnFormat::Unorm8x4),
            ambient: 0.35,
        };
        s.set_geometry(geometry);
        s
    }

    /// The layer as an [`InstancedMesh`](crate::core::InstancedMesh) for the
    /// scene renderer — lit, shadowed and post-processed with everything
    /// else: one instance per row, at its position relative to `origin`,
    /// scaled by its `scales` row times `style.size_scale`, tinted its colour.
    /// A snapshot: later changes to the layer are not seen.
    pub fn to_instanced_mesh(
        &self,
        material: crate::materials::Material,
        origin: &super::Origin,
    ) -> crate::core::InstancedMesh {
        use crate::core::BufferAttribute;
        use crate::math::{Matrix4, Quaternion, Vector3};
        let (verts, index) = &self.mesh;
        let mut g = crate::BufferGeometry::new();
        let (mut pos, mut nor) = (Vec::new(), Vec::new());
        for v in verts.chunks_exact(6) {
            pos.extend_from_slice(&v[..3]);
            nor.extend_from_slice(&v[3..]);
        }
        g.set_attribute("position", BufferAttribute::new(pos, 3));
        g.set_attribute("normal", BufferAttribute::new(nor, 3));
        if let Some(idx) = index {
            g.set_index(idx.clone());
        }
        let abs = self
            .positions
            .positions_f64(self.common.local_origin)
            .unwrap_or_default();
        let mut mesh = crate::core::InstancedMesh::new(g, material, abs.len());
        let s = self.common.style.size_scale;
        for (i, p) in abs.iter().enumerate() {
            let sc = self.scales.row_f32(i);
            mesh.set_matrix_at(
                i,
                Matrix4::compose(
                    origin.to_local(*p),
                    Quaternion::default(),
                    Vector3::new(sc[0] * s, sc[1] * s, sc[2] * s),
                ),
            );
            let c = self.colors.row_f32(i);
            let l = super::points::srgb_u8_to_linear(bytemuck::cast([c[0]]));
            mesh.colors.push(crate::math::Color::new(l[0], l[1], l[2]));
        }
        mesh
    }

    /// Replace the instanced geometry.
    pub fn set_geometry(&mut self, geometry: &BufferGeometry) {
        let pos = geometry
            .get_attribute("position")
            .filter(|a| a.item_size == 3);
        let nrm = geometry
            .get_attribute("normal")
            .filter(|a| a.item_size == 3);
        let n = pos.map(|p| p.count()).unwrap_or(0);
        let mut v = Vec::with_capacity(n * 6);
        for i in 0..n {
            let p = &pos.unwrap().array[i * 3..i * 3 + 3];
            v.extend_from_slice(p);
            match nrm {
                Some(a) if a.count() > i => v.extend_from_slice(&a.array[i * 3..i * 3 + 3]),
                // No normal: point it at the light, which makes it unlit.
                _ => v.extend_from_slice(&[0.0, 0.0, 0.0]),
            }
        }
        self.mesh = (v, geometry.index.clone());
        self.mesh_dirty = true;
    }

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
    /// Per-row scale on each axis.
    pub fn set_scales(&mut self, s: &[[f32; 3]]) {
        let flat: Vec<f32> = s.iter().flatten().copied().collect();
        self.scales.set_f32(&flat);
    }
    pub fn update_scales(&mut self, start: usize, s: &[[f32; 3]]) {
        let flat: Vec<f32> = s.iter().flatten().copied().collect();
        self.scales.update_f32(start, &flat);
    }
    pub fn set_colors(&mut self, c: &[[u8; 4]]) {
        self.colors.set_rgba(c);
    }
    pub fn update_colors(&mut self, start: usize, c: &[[u8; 4]]) {
        self.colors.update_rgba(start, c);
    }
    pub fn set_default_color(&mut self, c: [u8; 4]) {
        self.colors.set_default_rgba(c);
    }
}

impl Mark for ShapeMarks {
    fn kind(&self) -> MarkKind {
        MarkKind::Shape
    }
    fn common(&self) -> &MarkCommon {
        &self.common
    }
    fn common_mut(&mut self) -> &mut MarkCommon {
        &mut self.common
    }
    fn instance_count(&self) -> usize {
        if self.mesh.0.is_empty() {
            0
        } else {
            self.positions.len()
        }
    }
    fn kind_uniforms(&self) -> KindUniforms {
        (
            [self.ambient.clamp(0.0, 1.0), 0.0, 0.0, 0.0],
            [0.0; 4],
            [0.0; 4],
        )
    }
    fn prepare(&mut self, ctx: &PrepareContext) {
        if !self.mesh_dirty || self.mesh.0.is_empty() {
            return;
        }
        let vb = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threers shape marks vertices"),
            size: (self.mesh.0.len() * 4) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue
            .write_buffer(&vb, 0, bytemuck::cast_slice(&self.mesh.0));
        let ib = self.mesh.1.as_ref().filter(|i| !i.is_empty()).map(|idx| {
            let b = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("threers shape marks indices"),
                size: (idx.len() * 4) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            ctx.queue.write_buffer(&b, 0, bytemuck::cast_slice(idx));
            (b, idx.len() as u32)
        });
        self.mesh_gpu = Some((vb, (self.mesh.0.len() / 6) as u32, ib));
        self.mesh_dirty = false;
    }
    fn release_gpu(&mut self) {
        self.mesh_gpu = None;
        self.mesh_dirty = true;
        super::release_columns(self);
    }
    fn gpu_bytes(&mut self) -> u64 {
        let mesh = self.mesh_gpu.as_ref().map_or(0, |(vb, _, ib)| {
            vb.size() + ib.as_ref().map_or(0, |(b, _)| b.size())
        });
        super::column_bytes(self) + mesh
    }
    fn columns(&mut self) -> Vec<&mut Column> {
        vec![&mut self.positions, &mut self.scales, &mut self.colors]
    }
    fn geometry(&self) -> Option<InstancedGeometry<'_>> {
        self.mesh_gpu
            .as_ref()
            .map(|(vb, n, ib)| (vb, *n, ib.as_ref().map(|(b, c)| (b, *c))))
    }
}
