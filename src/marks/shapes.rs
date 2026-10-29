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
    /// Per vertex: the texture coordinate (empty: none), and the colour,
    /// linear RGBA (empty: white).
    uvs: Vec<f32>,
    vertex_colors: Vec<f32>,
    /// An image from an atlas the mesh is textured with, by its uvs.
    texture: Option<(super::SharedAtlas, String)>,
    texture_gpu: Option<(u64, wgpu::BindGroup)>,
    /// A texture made on the GPU instead of an atlas: its view and a
    /// generation that changes with it, holding sRGB bytes premultiplied
    /// by alpha in a linear format (a frame drawn by the marks).
    gpu_texture: Option<(std::sync::Arc<wgpu::TextureView>, u64)>,
    /// The atlas version the image's place was taken at, and the place:
    /// `[x, y, width, height]` of the atlas, 0 to 1.
    texture_version: Option<u64>,
    texture_rect: [f32; 4],
    /// Repeat the texture past uvs of 0 and 1 (glTF's default); otherwise
    /// clamp to its edge.
    pub texture_repeat: bool,
    /// With a [GPU texture](Self::set_gpu_texture): the texture is the
    /// colour as it is — its premultiplied texels lit as they are, empty
    /// ones transparent but still in the depth buffer (deck.gl's terrain
    /// `TERRAIN_MODE_USE_COVER_ONLY`). Otherwise empty texels are not drawn.
    pub cover_only: bool,
    mesh_dirty: bool,
    mesh_gpu: Option<MeshGpu>,
    positions: Column,
    scales: Column,
    colors: Column,
    /// Per row: a unit quaternion `[x, y, z, w]` turning the scaled mesh.
    rotations: Column,
    /// How much of the colour is lit regardless of the light: 1 is unlit.
    pub ambient: f32,
    /// Light as deck.gl does instead (`ambient` is then unused).
    pub lighting: Option<super::Lighting>,
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
            uvs: Vec::new(),
            vertex_colors: Vec::new(),
            texture: None,
            texture_gpu: None,
            gpu_texture: None,
            texture_version: None,
            texture_rect: [0.0, 0.0, 1.0, 1.0],
            texture_repeat: true,
            cover_only: false,
            mesh_dirty: true,
            mesh_gpu: None,
            positions: Column::new(ColumnFormat::PositionF64),
            scales,
            rotations: {
                let mut r = Column::float(4);
                r.set_default_f32(&[0.0, 0.0, 0.0, 1.0]);
                r
            },
            colors: Column::new(ColumnFormat::Unorm8x4),
            ambient: 0.35,
            lighting: None,
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
                    {
                        let q = self.rotations.row_f32(i);
                        Quaternion::new(q[0], q[1], q[2], q[3])
                    },
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
        self.uvs = geometry.get_attribute("uv").filter(|a| a.item_size == 2 && a.count() == n).map_or(Vec::new(), |a| a.array.clone());
        // three.js's vertex colours: linear, RGB or RGBA.
        self.vertex_colors = match geometry.get_attribute("color").filter(|a| a.count() == n) {
            Some(a) if a.item_size == 4 => a.array.clone(),
            Some(a) if a.item_size == 3 => a.array.chunks_exact(3).flat_map(|c| [c[0], c[1], c[2], 1.0]).collect(),
            _ => Vec::new(),
        };
        self.mesh_dirty = true;
    }

    /// Texture the mesh with image `name` of `atlas`, by its `uv`s (0 to 1
    /// across the image, clamped); `None` for no texture.
    /// Texture the mesh with a texture drawn on the GPU — a frame the marks
    /// drew into a plain `Rgba8Unorm` target, sRGB bytes premultiplied by
    /// alpha — over the whole of its uvs (0 to 1, clamped). `generation`
    /// changes when the view does. Replaces [`set_texture`](Self::set_texture)'s.
    pub fn set_gpu_texture(&mut self, texture: Option<(std::sync::Arc<wgpu::TextureView>, u64)>) {
        if texture.as_ref().map(|t| t.1) != self.gpu_texture.as_ref().map(|t| t.1) {
            self.texture_gpu = None;
        }
        self.gpu_texture = texture;
        if self.gpu_texture.is_some() {
            self.texture = None;
            self.texture_rect = [0.0, 0.0, 1.0, 1.0];
            self.texture_repeat = false;
        }
    }

    pub fn set_texture(&mut self, texture: Option<(super::SharedAtlas, String)>) {
        self.texture = texture;
        self.gpu_texture = None;
        self.texture_gpu = None;
        self.texture_version = None;
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
    /// Turn each row's mesh (after its scale) by a unit quaternion
    /// `[x, y, z, w]`; empty for none.
    pub fn set_rotations(&mut self, r: &[[f32; 4]]) {
        let flat: Vec<f32> = r.iter().flatten().copied().collect();
        self.rotations.set_f32(&flat);
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
        let (yz, l0, _) = super::Lighting::uniforms(self.lighting.as_ref());
        let ambient = self.lighting.map_or(self.ambient.clamp(0.0, 1.0), |l| l.ambient);
        // style3: the texture's place in its atlas; a negative width clamps
        // rather than repeats.
        let mut rect = self.texture_rect;
        if !self.texture_repeat {
            rect[2] = -rect[2];
        }
        ([ambient, yz[0], yz[1], 0.0], rect, l0)
    }

    fn kind_extra(&self) -> [f32; 4] {
        super::Lighting::uniforms(self.lighting.as_ref()).2
    }

    fn kind_material(&self) -> [f32; 4] {
        super::Lighting::material(self.lighting.as_ref())
    }
    fn prepare(&mut self, ctx: &PrepareContext) {
        if let Some((view, generation)) = &self.gpu_texture {
            if self.texture_gpu.as_ref().is_none_or(|g| g.0 != *generation) {
                let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("threers shape gpu texture"),
                    layout: &ctx.resources.texture_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&ctx.resources.mip_sampler) },
                    ],
                });
                self.texture_gpu = Some((*generation, bg));
            }
        }
        // The texture's place in its atlas, which may have moved.
        if let Some((atlas, name)) = &self.texture {
            let size = super::sprites::atlas_bind_group(ctx, atlas, &mut self.texture_gpu);
            let a = atlas.read().unwrap();
            let version = a.version();
            if self.texture_version != Some(version) {
                self.texture_version = Some(version);
                if let Some(r) = a.get(name) {
                    let (w, h) = (size.0.max(1) as f32, size.1.max(1) as f32);
                    self.texture_rect = [r.x as f32 / w, r.y as f32 / h, r.width as f32 / w, r.height as f32 / h];
                }
            }
        }
        if !self.mesh_dirty || self.mesh.0.is_empty() {
            return;
        }
        // position, normal, uv (in the atlas), colour, and the vertex's
        // triangle's corners — so the mesh is drawn unindexed, each vertex
        // knowing its triangle (for `MarkView::swiftshader_depth`).
        let n = self.mesh.0.len() / 6;
        let order: Vec<u32> = match &self.mesh.1 {
            Some(idx) if !idx.is_empty() => idx.clone(),
            _ => (0..n as u32).collect(),
        };
        let pos = |v: u32| &self.mesh.0[v as usize * 6..v as usize * 6 + 3];
        let mut data = Vec::with_capacity(order.len() * 21);
        for tri in order.chunks(3) {
            for &v in tri {
                let i = v as usize;
                data.extend_from_slice(&self.mesh.0[i * 6..i * 6 + 6]);
                data.extend_from_slice(self.uvs.get(i * 2..i * 2 + 2).unwrap_or(&[0.0, 0.0]));
                data.extend_from_slice(self.vertex_colors.get(i * 4..i * 4 + 4).unwrap_or(&[1.0; 4]));
                for k in 0..3 {
                    data.extend_from_slice(tri.get(k).map_or(&[0.0f32; 3][..], |&c| pos(c)));
                }
            }
        }
        let vertices = (data.len() / 21) as u32;
        // The same shape moved (a posed model's next frame): write into the
        // buffers it has.
        let bytes = (data.len() * 4) as u64;
        if let Some((vb, _, _)) = &self.mesh_gpu {
            if vb.size() == bytes {
                ctx.queue.write_buffer(vb, 0, bytemuck::cast_slice(&data));
                self.mesh_dirty = false;
                return;
            }
        }
        let vb = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threers shape marks vertices"),
            size: bytes,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.queue.write_buffer(&vb, 0, bytemuck::cast_slice(&data));
        self.mesh_gpu = Some((vb, vertices, None));
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
        vec![&mut self.positions, &mut self.scales, &mut self.colors, &mut self.rotations]
    }
    fn patterned(&self) -> bool {
        self.texture.is_some() || self.gpu_texture.is_some()
    }

    /// Bit 0: the texture holds premultiplied sRGB bytes in a linear
    /// format ([`set_gpu_texture`](Self::set_gpu_texture)); bit 1: it is
    /// drawn [as it is](Self::cover_only).
    fn kind_flags(&self) -> u32 {
        let gpu = self.gpu_texture.is_some();
        u32::from(gpu) | (u32::from(gpu && self.cover_only) << 1)
    }
    fn extra_bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.texture_gpu.as_ref().map(|g| &g.1)
    }
    fn geometry(&self) -> Option<InstancedGeometry<'_>> {
        self.mesh_gpu
            .as_ref()
            .map(|(vb, n, ib)| (vb, *n, ib.as_ref().map(|(b, c)| (b, *c))))
    }
}
