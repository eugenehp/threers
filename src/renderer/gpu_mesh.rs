use crate::core::BufferGeometry;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// GPU-side buffers for a single geometry, cached by `Arc<BufferGeometry>` pointer identity.
pub struct GpuMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: Option<wgpu::Buffer>,
    pub vertex_count: u32,
    pub index_count: u32,
    /// Line-list index buffer for `wireframe: true` materials. Matches three.js
    /// WebGLGeometries: each triangle emits (a,b,b,c,c,a) without deduping —
    /// unless the geometry carries polygons; see
    /// [`BufferGeometry::wireframe_indices`]. Built on first wireframe use by
    /// [`ensure_wire_indices`](Self::ensure_wire_indices), not by `upload`.
    pub wire_index_buffer: Option<wgpu::Buffer>,
    pub wire_index_count: u32,
    /// Line segments' ends, each segment's two as their own vertices in
    /// order (an index followed), and one spare vertex after, for lines drawn
    /// wider than a pixel (`LineBasicMaterial::line_width`): `vs_wide_lines`
    /// reads a segment as two consecutive vertices of an instance, its end
    /// through a binding a vertex further on — which a fetch of a whole
    /// stride would take past the last vertex without the spare. Built on
    /// first use by [`ensure_segments`](Self::ensure_segments).
    pub segment_buffer: Option<wgpu::Buffer>,
    /// The segment buffer's vertices, the spare left out.
    pub segment_vertices: u32,
    #[allow(dead_code)]
    pub has_vertex_colors: bool,
}

impl GpuMesh {
    /// Vertex layout (64 bytes):
    /// position(3) + normal(3) + uv(2) + color(4) + tangent(4).
    ///
    /// `tangent.xyz` is the surface tangent and `tangent.w` the bitangent
    /// handedness (glTF's convention). An all-zero tangent means "not
    /// supplied", and the fragment shader falls back to a derivative-derived
    /// frame — so geometries without tangents behave exactly as before.
    pub const VERTEX_STRIDE: u64 = 16 * 4;

    pub(crate) fn build_interleaved(geom: &BufferGeometry) -> Vec<f32> {
        let positions = geom
            .get_attribute("position")
            .expect("geometry needs position attribute");
        let normals = geom.get_attribute("normal");
        let uvs = geom.get_attribute("uv");
        let line_distances = geom.get_attribute("lineDistance");
        let colors = geom.get_attribute("color");
        let tangents = geom.get_attribute("tangent");

        let vert_count = positions.count();
        let mut interleaved = Vec::with_capacity(vert_count * 16);
        for i in 0..vert_count {
            interleaved.extend_from_slice(&positions.array[i * 3..i * 3 + 3]);
            if let Some(n) = normals {
                interleaved.extend_from_slice(&n.array[i * 3..i * 3 + 3]);
            } else {
                interleaved.extend_from_slice(&[0.0, 0.0, 1.0]);
            }
            if let Some(u) = uvs {
                interleaved.extend_from_slice(&u.array[i * 2..i * 2 + 2]);
            } else if let Some(ld) = line_distances {
                interleaved.extend_from_slice(&[ld.array[i], 0.0]);
            } else {
                interleaved.extend_from_slice(&[0.0, 0.0]);
            }
            if let Some(c) = colors {
                let s = c.item_size;
                if s == 3 {
                    interleaved.extend_from_slice(&c.array[i * 3..i * 3 + 3]);
                    interleaved.push(1.0);
                } else if s == 4 {
                    interleaved.extend_from_slice(&c.array[i * 4..i * 4 + 4]);
                } else {
                    interleaved.extend_from_slice(&[1.0, 1.0, 1.0, 1.0]);
                }
            } else {
                interleaved.extend_from_slice(&[1.0, 1.0, 1.0, 1.0]);
            }
            // Tangent (glTF TANGENT / `compute_tangents`). Zero = absent.
            match tangents {
                Some(t) if t.item_size == 4 => {
                    interleaved.extend_from_slice(&t.array[i * 4..i * 4 + 4]);
                }
                Some(t) if t.item_size == 3 => {
                    interleaved.extend_from_slice(&t.array[i * 3..i * 3 + 3]);
                    interleaved.push(1.0);
                }
                _ => interleaved.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]),
            }
        }
        interleaved
    }

    pub fn upload(device: &wgpu::Device, geom: &BufferGeometry) -> Self {
        let interleaved = Self::build_interleaved(geom);
        let vert_count = geom
            .get_attribute("position")
            .map(|p| p.count())
            .unwrap_or(0);
        let colors = geom.get_attribute("color").is_some();

        let mut vertex_usage = wgpu::BufferUsages::VERTEX;
        if geom.gpu_writable {
            // Opt-in: lets a user compute pass rewrite this mesh in place. See
            // `BufferGeometry::gpu_writable`.
            vertex_usage |= wgpu::BufferUsages::STORAGE;
        }
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("threers vertex buffer"),
            contents: bytemuck::cast_slice(&interleaved),
            usage: vertex_usage,
        });

        let (index_buffer, index_count) = if let Some(idx) = &geom.index {
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("threers index buffer"),
                contents: bytemuck::cast_slice(idx),
                usage: wgpu::BufferUsages::INDEX,
            });
            (Some(buf), idx.len() as u32)
        } else {
            (None, 0)
        };

        Self {
            vertex_buffer,
            index_buffer,
            vertex_count: vert_count as u32,
            index_count,
            wire_index_buffer: None,
            wire_index_count: 0,
            segment_buffer: None,
            segment_vertices: 0,
            has_vertex_colors: colors,
        }
    }

    /// Build the wireframe index buffer from `geom` — the geometry this mesh
    /// was uploaded from — if it is not built yet.
    ///
    /// Lazy, as on the Metal backend, because most meshes are never drawn as
    /// wireframes, and with a polygon table the line list costs a hash lookup
    /// per triangle edge. A deforming mesh re-uploads on every position write,
    /// so building it in `upload` would pay that every frame for nothing.
    pub fn ensure_wire_indices(&mut self, device: &wgpu::Device, geom: &BufferGeometry) {
        if self.wire_index_buffer.is_some() {
            return;
        }
        let flat = geom.wireframe_indices();
        if flat.is_empty() {
            return;
        }
        self.wire_index_buffer = Some(device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("threers wireframe index buffer"),
                contents: bytemuck::cast_slice(&flat),
                usage: wgpu::BufferUsages::INDEX,
            },
        ));
        self.wire_index_count = flat.len() as u32;
    }
}

impl GpuMesh {
    /// Build [`segment_buffer`](Self::segment_buffer) from `geom` (the
    /// geometry this mesh was uploaded from) if it is not built yet.
    pub fn ensure_segments(&mut self, device: &wgpu::Device, geom: &BufferGeometry) {
        if self.segment_buffer.is_some() {
            return;
        }
        let all = Self::build_interleaved(geom);
        let k = Self::VERTEX_STRIDE as usize / 4;
        let mut out = match &geom.index {
            Some(idx) => {
                let mut out = Vec::with_capacity((idx.len() + 1) * k);
                for &i in idx {
                    let i = i as usize;
                    match all.get(i * k..(i + 1) * k) {
                        Some(v) => out.extend_from_slice(v),
                        None => out.extend(std::iter::repeat_n(0.0, k)),
                    }
                }
                out
            }
            None => all,
        };
        if out.is_empty() {
            return;
        }
        self.segment_vertices = (out.len() / k) as u32;
        out.extend(std::iter::repeat_n(0.0, k));
        self.segment_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("threers wide line segments"),
            contents: bytemuck::cast_slice(&out),
            usage: wgpu::BufferUsages::VERTEX,
        }));
    }
}

pub fn geom_cache_key(g: &Arc<BufferGeometry>) -> *const BufferGeometry {
    Arc::as_ptr(g)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BufferAttribute, BufferGeometry};

    fn interleaved_uv_x(geom: &BufferGeometry, vert: usize) -> f32 {
        let interleaved = GpuMesh::build_interleaved(geom);
        // Derived from the stride rather than hardcoded, so adding a vertex
        // attribute can't silently invalidate this index.
        let floats_per_vert = (GpuMesh::VERTEX_STRIDE / 4) as usize;
        interleaved[vert * floats_per_vert + 6]
    }

    #[test]
    fn interleaved_carries_tangent_when_present() {
        let floats_per_vert = (GpuMesh::VERTEX_STRIDE / 4) as usize;
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(vec![0., 0., 0., 1., 0., 0.], 3),
        );
        // Without a tangent attribute the slot must be all zeros — that is the
        // sentinel the shader reads as "fall back to derivative tangents".
        let no_tan = GpuMesh::build_interleaved(&g);
        assert_eq!(&no_tan[12..16], &[0.0, 0.0, 0.0, 0.0]);

        g.set_attribute(
            "tangent",
            BufferAttribute::new(vec![1., 0., 0., 1., 0., 1., 0., -1.], 4),
        );
        let with_tan = GpuMesh::build_interleaved(&g);
        assert_eq!(&with_tan[12..16], &[1.0, 0.0, 0.0, 1.0]);
        // Second vertex, including the negative handedness in w.
        let o = floats_per_vert + 12;
        assert_eq!(&with_tan[o..o + 4], &[0.0, 1.0, 0.0, -1.0]);
    }

    #[test]
    fn interleaved_includes_uv_line_distance() {
        let mut g = BufferGeometry::new();
        g.set_attribute(
            "position",
            BufferAttribute::new(vec![0., 0., 0., 1.8, 0., 0.], 3),
        );
        g.set_attribute("lineDistance", BufferAttribute::new(vec![0., 1.8], 1));
        g.set_attribute("uv", BufferAttribute::new(vec![0., 0., 1.8, 0.], 2));
        assert!((interleaved_uv_x(&g, 1) - 1.8).abs() < 1e-5);
    }
}
