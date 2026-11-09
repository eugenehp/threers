use super::*;

impl Renderer {
    /// Upload `geom` to the GPU now, instead of waiting for the first draw.
    ///
    /// Only needed when something outside the renderer wants the buffer up
    /// front — a compute pass that writes it, typically. See
    /// [`vertex_buffer`](Self::vertex_buffer).
    pub fn upload_geometry(&mut self, geom: &std::sync::Arc<BufferGeometry>) {
        self.ensure_geometry(geom);
    }

    /// The GPU vertex buffer backing `geom`, if it has been uploaded.
    ///
    /// Interleaved, `GpuMesh::VERTEX_STRIDE` bytes per vertex, laid out
    /// position(3), normal(3), uv(2), colour(4), tangent(4) as `f32`.
    ///
    /// Writing it from a compute pass requires
    /// [`BufferGeometry::gpu_writable`], and the handle is only valid until the
    /// geometry is re-uploaded — which happens whenever
    /// [`geometry_version`](BufferGeometry::geometry_version) changes. Drive a
    /// mesh one way or the other, not both.
    pub fn vertex_buffer(&self, geom: &std::sync::Arc<BufferGeometry>) -> Option<&wgpu::Buffer> {
        self.geom_cache
            .get(&geom_cache_key(geom))
            .map(|e| &e.mesh.vertex_buffer)
    }

    pub(super) fn ensure_geometry(&mut self, geom: &std::sync::Arc<BufferGeometry>) {
        let key = geom_cache_key(geom);
        let ver = geom.geometry_version;
        let stale = self
            .geom_cache
            .get(&key)
            .map(|e| e.version != ver)
            .unwrap_or(true);
        if stale {
            let mesh = GpuMesh::upload(&self.device, geom);
            let sphere = local_bounding_sphere(geom);
            self.geom_cache.insert(
                key,
                CachedGpuMesh {
                    version: ver,
                    mesh,
                    sphere,
                },
            );
        }
        self.ensure_uv_sets(geom);
    }

    /// Upload `geom`'s `uv1`/`uv2`/`uv3` as its [`UvSetTexture`], once per
    /// geometry version; drop it when the geometry no longer has any.
    pub(super) fn ensure_uv_sets(&mut self, geom: &std::sync::Arc<BufferGeometry>) {
        let key = geom_cache_key(geom);
        let ver = geom.geometry_version;
        if self.uv_cache.get(&key).is_some_and(|u| u.version == ver) {
            return;
        }
        let sets = UV_SET_NAMES.map(|n| geom.get_attribute(n).filter(|a| a.item_size >= 2));
        let count = geom.get_attribute("position").map_or(0, |p| p.count());
        if sets.iter().all(Option::is_none) || count == 0 {
            self.uv_cache.remove(&key);
            return;
        }
        let texels = (count * 2) as u32;
        let max = self.device.limits().max_texture_dimension_2d.max(1);
        let width = texels.min(max);
        let height = texels.div_ceil(width);
        let mut data = vec![0.0f32; (width * height * 4) as usize];
        for (set, attr) in sets.iter().enumerate() {
            let Some(a) = attr else { continue };
            for v in 0..count.min(a.count()) {
                let at = (v * 2 + set / 2) * 4 + (set % 2) * 2;
                data[at] = a.array[v * a.item_size];
                data[at + 1] = a.array[v * a.item_size + 1];
            }
        }
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers uv sets"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 16),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.uv_cache.insert(
            key,
            UvSetTexture {
                version: ver,
                _texture: texture,
                view,
            },
        );
        self.tex_bg_cache.retain(|k, _| k.3 .0 != key as usize);
        self.variant_bg_cache.retain(|k, _| k.1 .0 != key as usize);
    }

    /// The identity of the uv-set texture a draw of geometry `key` binds —
    /// part of its bind group's cache key; `(0, 0)` for none.
    pub(super) fn uv_key(&self, key: *const BufferGeometry) -> (usize, u32) {
        self.uv_cache
            .get(&key)
            .map_or((0, 0), |u| (key as usize, u.version))
    }

    /// Upload `geom`'s morph targets as three.js's morph texture, once per
    /// geometry version. Returns the cache key and version to bind, or `None`
    /// when the geometry has no morph targets.
    pub(super) fn ensure_morph(
        &mut self,
        geom: &std::sync::Arc<BufferGeometry>,
    ) -> Option<(usize, u32)> {
        let key = geom_cache_key(geom);
        let ver = geom.geometry_version;
        if self.morph_cache.get(&key).is_some_and(|m| m.version == ver) {
            return Some((key as usize, ver));
        }
        let texels = crate::core::MorphTexels::build(geom)?;
        if texels.targets == 0 || texels.vertex_count == 0 {
            return None;
        }
        // three.js: one row of `vertexCount · stride` texels, wrapped onto
        // more rows when that passes the maximum texture width.
        let per_layer = texels.vertex_count * texels.stride;
        let max = self.device.limits().max_texture_dimension_2d.max(1);
        let width = per_layer.min(max);
        let height = per_layer.div_ceil(width);
        // At least two layers: WebGL2 (wgpu's GL backend) makes a
        // one-layer texture a plain `TEXTURE_2D`, and the shader's
        // `sampler2DArray` then reads zeros from it — a single morph target
        // collapsed every vertex to the origin there. The spare layer is
        // never read.
        let layers = texels.targets.max(2);
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers morph texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // Each layer padded out to whole rows.
        let row = (width * 4) as usize;
        let mut padded = vec![0.0f32; row * height as usize * layers as usize];
        let layer_len = (per_layer * 4) as usize;
        for t in 0..texels.targets as usize {
            let src = &texels.data[t * layer_len..(t + 1) * layer_len];
            let dst = t * row * height as usize;
            padded[dst..dst + layer_len].copy_from_slice(src);
        }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&padded),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 16),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: layers,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        self.morph_cache.insert(
            key,
            MorphTexture {
                version: ver,
                _texture: texture,
                view,
                instance_layer: None,
            },
        );
        Some((key as usize, ver))
    }

    /// An `InstancedMesh`'s morph texture: `geom`'s targets as in
    /// [`ensure_morph`](Self::ensure_morph), plus one more layer holding
    /// three.js's `InstancedMesh.morphTexture` — `instances` rows of
    /// `row_len` floats (base influence, then the influences), one float a
    /// texel in `.x`, row-major over the layer's width. Cached under `owner`
    /// (the object), rebuilt when the geometry or the row layout changes and
    /// otherwise only the rows are rewritten.
    ///
    /// One texture rather than a second binding because the mesh layout is
    /// already at WebGL2's sixteen texture units.
    /// The GPU buffer for `obj`'s instance data `slot`, holding `data`: last
    /// frame's buffer when it is big enough, rewritten only if `data`
    /// changed, else a new one with room to grow.
    pub(super) fn instance_buffer(
        &mut self,
        obj: crate::core::ObjectId,
        slot: u8,
        data: Vec<f32>,
        version: Option<(u64, usize)>,
    ) -> wgpu::Buffer {
        let frame = self.instance_frame;
        let bytes = (data.len() * 4) as u64;
        if let Some(c) = self.instance_cache.get_mut(&(obj, slot))
            && c.buffer.size() >= bytes.max(4)
        {
            // A versioned mesh got here because its version moved: write.
            if version.is_some() || c.data != data {
                self.queue
                    .write_buffer(&c.buffer, 0, bytemuck::cast_slice(&data));
                c.data = if version.is_some() { Vec::new() } else { data };
                self.instance_stamps += 1;
                c.stamp = self.instance_stamps;
            }
            c.frame = frame;
            c.version = version;
            return c.buffer.clone();
        }
        // Half again, so a mesh that grows a little each frame is not
        // reallocated every frame.
        let size = ((bytes + bytes / 2).max(16)).div_ceil(4) * 4;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threers instance buffer"),
            size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&buffer, 0, bytemuck::cast_slice(&data));
        self.instance_stamps += 1;
        self.instance_cache.insert(
            (obj, slot),
            InstanceBuffer {
                buffer: buffer.clone(),
                // A versioned mesh is never compared, so keep no copy.
                data: if version.is_some() { Vec::new() } else { data },
                frame,
                version,
                stamp: self.instance_stamps,
            },
        );
        buffer
    }

    /// The content stamp of `obj`'s instance slot, 0 when it has none.
    pub(super) fn instance_stamp(&self, obj: crate::core::ObjectId, slot: u8) -> u64 {
        self.instance_cache.get(&(obj, slot)).map_or(0, |c| c.stamp)
    }

    /// Last frame's buffer for `obj`'s slot when its `instance_version` (and
    /// count) say nothing changed — without reading the instance data.
    pub(super) fn cached_instances(
        &mut self,
        obj: crate::core::ObjectId,
        slot: u8,
        version: Option<(u64, usize)>,
    ) -> Option<wgpu::Buffer> {
        let v = version?;
        let c = self.instance_cache.get_mut(&(obj, slot))?;
        if c.version != Some(v) {
            return None;
        }
        c.frame = self.instance_frame;
        Some(c.buffer.clone())
    }

    pub(super) fn ensure_instance_morph(
        &mut self,
        owner: usize,
        geom: &std::sync::Arc<BufferGeometry>,
        rows: &[f32],
        instances: u32,
        row_len: u32,
    ) -> Option<(usize, u32)> {
        let key = owner as *const BufferGeometry;
        // High bit set, so it can never equal a plain geometry version.
        let ver = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            (
                geom_cache_key(geom),
                geom.geometry_version,
                instances,
                row_len,
            )
                .hash(&mut h);
            (h.finish() as u32) | 0x8000_0000
        };
        let texel_rows = |width: u32| -> Vec<f32> {
            let n = (instances * row_len) as usize;
            let rows_needed = (n as u32).div_ceil(width).max(1);
            let mut out = vec![0.0f32; (width * rows_needed * 4) as usize];
            for (i, v) in rows.iter().take(n).enumerate() {
                out[i * 4] = *v;
            }
            out
        };
        if let Some(m) = self.morph_cache.get(&key).filter(|m| m.version == ver) {
            let (layer, width) = m.instance_layer?;
            let data = texel_rows(width);
            let rows_needed = (data.len() / 4) as u32 / width;
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &m._texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&data),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 16),
                    rows_per_image: Some(rows_needed),
                },
                wgpu::Extent3d {
                    width,
                    height: rows_needed,
                    depth_or_array_layers: 1,
                },
            );
            return Some((owner, ver));
        }
        let texels = crate::core::MorphTexels::build(geom)?;
        if texels.targets == 0 || texels.vertex_count == 0 || instances == 0 {
            return None;
        }
        let per_layer = texels.vertex_count * texels.stride;
        let max = self.device.limits().max_texture_dimension_2d.max(1);
        let width = per_layer.min(max);
        let height = per_layer
            .div_ceil(width)
            .max((instances * row_len).div_ceil(width));
        if height > max {
            return None;
        }
        let layers = (texels.targets + 1).max(2);
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers instanced morph texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let row = (width * 4) as usize;
        let layer_size = row * height as usize;
        let mut padded = vec![0.0f32; layer_size * layers as usize];
        let layer_len = (per_layer * 4) as usize;
        for t in 0..texels.targets as usize {
            let src = &texels.data[t * layer_len..(t + 1) * layer_len];
            padded[t * layer_size..t * layer_size + layer_len].copy_from_slice(src);
        }
        let inst = texel_rows(width);
        let at = texels.targets as usize * layer_size;
        padded[at..at + inst.len()].copy_from_slice(&inst);
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&padded),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 16),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: layers,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        self.morph_cache.insert(
            key,
            MorphTexture {
                version: ver,
                _texture: texture,
                view,
                instance_layer: Some((texels.targets, width)),
            },
        );
        Some((owner, ver))
    }

    /// The wireframe line list of one partial span of `geom` — a group or a
    /// draw range — cached by geometry version and span.
    pub(super) fn ensure_wire_span(
        &mut self,
        geom: &std::sync::Arc<BufferGeometry>,
        start: u32,
        end: u32,
    ) {
        let k = (geom_cache_key(geom), geom.geometry_version, start, end);
        if self.wire_span_cache.contains_key(&k) {
            return;
        }
        // A runaway guard: spans are few and stable per geometry, and a
        // stale version's entries are simply never asked for again.
        if self.wire_span_cache.len() > 1024 {
            self.wire_span_cache.clear();
        }
        let flat = geom.wireframe_indices_in(start, end);
        let buf = wgpu::util::DeviceExt::create_buffer_init(
            &*self.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("threers wireframe span index buffer"),
                // An empty buffer cannot be bound; one dummy index, drawn zero
                // times, stands in.
                contents: bytemuck::cast_slice(if flat.is_empty() { &[0u32] } else { &flat }),
                usage: wgpu::BufferUsages::INDEX,
            },
        );
        self.wire_span_cache.insert(k, (buf, flat.len() as u32));
    }

    /// Give `geom`'s cached upload its wireframe line list. Call after
    /// [`ensure_geometry`](Self::ensure_geometry) for a `wireframe: true`
    /// material; a re-upload drops the list, so it follows `geometry_version`.
    pub(super) fn ensure_wire_indices(&mut self, geom: &std::sync::Arc<BufferGeometry>) {
        if let Some(e) = self.geom_cache.get_mut(&geom_cache_key(geom)) {
            e.mesh.ensure_wire_indices(&self.device, geom);
        }
    }

    /// Drop every cached GPU mesh, freeing their buffers.
    ///
    /// The cache is keyed by the address of the geometry's `Arc`, which is
    /// stable only while that `Arc` is alive. Code that feeds the renderer a
    /// *fresh* geometry every frame — a CAD model re-evaluated per frame, say —
    /// must call this between frames: without it the cache grows without bound,
    /// and an allocator that reuses a freed address could serve the old mesh
    /// for the new geometry.
    ///
    /// Scenes that keep their geometry alive (the usual case) never need this.
    pub fn clear_geometry_cache(&mut self) {
        self.geom_cache.clear();
        self.skin_attr_cache.clear();
        self.morph_cache.clear();
        self.wire_span_cache.clear();
    }

    /// How many GPU meshes are currently cached.
    pub fn cached_geometry_count(&self) -> usize {
        self.geom_cache.len()
    }
}
