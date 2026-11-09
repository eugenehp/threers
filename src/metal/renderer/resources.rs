use super::*;

impl MetalRenderer {
    /// Upload `geometry` now rather than at its first draw — for a compute
    /// pass that writes its [`vertex_buffer`](Self::vertex_buffer).
    pub fn upload_geometry(&mut self, geometry: &Arc<BufferGeometry>) -> Result<(), MetalError> {
        self.ensure_geometry(geometry, false).map(|_| ())
    }

    /// The `MTLBuffer` behind an uploaded geometry and its length in bytes:
    /// one `Vertex` a vertex, twelve `f32` — position(3), normal(3), uv(2),
    /// colour(3) and alpha — the first twelve of the wgpu renderer's sixteen
    /// ([`Renderer::vertex_buffer`](crate::Renderer::vertex_buffer)). Valid
    /// until the geometry's version changes and it is uploaded again.
    pub fn vertex_buffer(&self, geometry: &Arc<BufferGeometry>) -> Option<(Id, usize)> {
        let cached = self.geometries.get(&(Arc::as_ptr(geometry) as usize))?;
        (cached.version == geometry.geometry_version && !cached.slim)
            .then(|| (cached.vertices.id(), cached.vertex_count * std::mem::size_of::<Vertex>()))
    }

    /// Ensure `geometry` has current GPU buffers. `true` if it uploaded.
    pub(super) fn ensure_geometry(
        &mut self,
        geometry: &Arc<BufferGeometry>,
        slim: bool,
    ) -> Result<bool, MetalError> {
        let key = Arc::as_ptr(geometry) as usize;
        let frame = self.frame;
        if let Some(cached) = self.geometries.get_mut(&key)
            && cached.version == geometry.geometry_version
            && cached.slim == slim
        {
            cached.last_used = frame;
            return Ok(false);
        }
        // Built as one or the other, never both: the slim path also skips
        // `derive_normals`, whose scratch vector is the size of the geometry.
        let (vbuf, vertex_count) = if slim {
            let vertices = build_line_vertices(geometry);
            if vertices.is_empty() {
                return Ok(false);
            }
            let n = vertices.len();
            (self.device.new_buffer(bytemuck::cast_slice(&vertices))?, n)
        } else {
            let vertices = build_vertices(geometry);
            if vertices.is_empty() {
                return Ok(false);
            }
            let n = vertices.len();
            (self.device.new_buffer(bytemuck::cast_slice(&vertices))?, n)
        };
        // Skinning attributes, if this geometry has them. Named as three.js
        // names them.
        let skin = if !slim
            && geometry.attributes.contains_key("joint")
            && geometry.attributes.contains_key("weight")
        {
            let data = build_skin(geometry, vertex_count);
            Some(self.device.new_buffer(bytemuck::cast_slice(&data))?)
        } else {
            None
        };
        let (ibuf, index_count) = match &geometry.index {
            Some(idx) if !idx.is_empty() => (
                Some(self.device.new_buffer(bytemuck::cast_slice(idx))?),
                idx.len(),
            ),
            _ => (None, 0),
        };
        // Morph targets in three.js's morph-texture layout, built by the
        // same helper the wgpu backend uploads from.
        let morph = match (!slim)
            .then(|| crate::core::MorphTexels::build(geometry))
            .flatten()
            .filter(|t| t.targets > 0 && t.vertex_count as usize == vertex_count)
        {
            Some(t) => {
                let has_position = geometry
                    .morph_attributes
                    .get("position")
                    .is_some_and(|v| !v.is_empty());
                Some(MorphBuffer {
                    buffer: self.device.new_buffer(bytemuck::cast_slice(&t.data))?,
                    count: t.targets,
                    stride: t.stride,
                    // Normal and colour as the helper says; position only
                    // with position targets, as `Morph` blends it; colour
                    // alpha (all four channels) for a 4-component colour.
                    flags: (t.flags & 3)
                        | (has_position as u32) << 2
                        | (((t.flags & 4) != 0) as u32) << 3,
                    vertex_count: t.vertex_count,
                })
            }
            None => None,
        };
        let morph_bytes = morph
            .as_ref()
            .map(|m| (m.count * m.stride * m.vertex_count) as u64 * 16)
            .unwrap_or(0);
        let sets = ["uv1", "uv2", "uv3"].map(|n| geometry.get_attribute(n).filter(|a| a.item_size >= 2));
        let tangents = geometry.get_attribute("tangent").filter(|a| a.item_size >= 3);
        let has_sets = sets.iter().any(Option::is_some);
        let uv_sets = if !slim && (has_sets || tangents.is_some()) {
            let mut data = vec![0.0f32; vertex_count * 12];
            for (set, attr) in sets.iter().enumerate() {
                let Some(a) = attr else { continue };
                for v in 0..vertex_count.min(a.count()) {
                    let at = v * 12 + (set / 2) * 4 + (set % 2) * 2;
                    data[at] = a.array[v * a.item_size];
                    data[at + 1] = a.array[v * a.item_size + 1];
                }
            }
            if let Some(t) = tangents {
                for v in 0..vertex_count.min(t.count()) {
                    let at = v * 12 + 8;
                    data[at..at + 3].copy_from_slice(&t.array[v * t.item_size..v * t.item_size + 3]);
                    data[at + 3] = if t.item_size >= 4 { t.array[v * t.item_size + 3] } else { 1.0 };
                }
            }
            let flags = has_sets as u32 | (tangents.is_some() as u32) << 1;
            Some((self.device.new_buffer(bytemuck::cast_slice(&data))?, flags))
        } else {
            None
        };
        self.geometries.insert(
            key,
            CachedGeometry {
                morph,
                uv_sets,
                bytes: morph_bytes + (vertex_count * if slim {
                    std::mem::size_of::<LineVertex>()
                } else {
                    std::mem::size_of::<Vertex>()
                } + index_count * std::mem::size_of::<u32>()) as u64,
                sphere: geometry_sphere(geometry),
                _geometry: geometry.clone(),
                version: geometry.geometry_version,
                vertices: vbuf,
                skin,
                indices: ibuf,
                index_count,
                vertex_count,
                wire: None,
                slim,
                last_used: frame,
            },
        );
        Ok(true)
    }

    /// Ensure the cached `geometry` has its wireframe line-list indices.
    ///
    /// Call after [`ensure_geometry`](Self::ensure_geometry): a stale entry
    /// is rebuilt there without them, so they follow `geometry_version`.
    pub(super) fn ensure_wire_indices(&mut self, geometry: &Arc<BufferGeometry>) -> Result<(), MetalError> {
        let key = Arc::as_ptr(geometry) as usize;
        let Some(cached) = self.geometries.get_mut(&key) else {
            return Ok(());
        };
        if cached.wire.is_some() || cached.slim {
            return Ok(());
        }
        let flat = geometry.wireframe_indices();
        if flat.is_empty() {
            return Ok(());
        }
        let buf = self.device.new_buffer(bytemuck::cast_slice(&flat))?;
        cached.bytes += (flat.len() * std::mem::size_of::<u32>()) as u64;
        cached.wire = Some((buf, flat.len()));
        Ok(())
    }

    /// Upload `texture`, as colour or as data.
    ///
    /// `linear` asks for the bytes as they are, with no sRGB decode on sample.
    /// Normal, metallic-roughness and occlusion maps are *quantities*, not
    /// colours — the glTF spec says so in §3.9.2 — and decoding them through a
    /// transfer curve corrupts them: the 0.5 that means "flat" in a normal map
    /// comes out as 0.214, a tangent-space x of −0.57, so every flat texel
    /// arrives steeply tilted. The wgpu loader learned this the hard way and
    /// keeps two views of each image; this keeps two uploads, because a Metal
    /// texture's pixel format is fixed when it is created.
    ///
    /// The two are cached separately, so an image used as both a colour map and
    /// a data map gets one of each rather than whichever role asked first.
    pub(super) fn ensure_texture_as(
        &mut self,
        texture: &Arc<Texture>,
        linear: bool,
    ) -> Result<bool, MetalError> {
        let key = texture_key(texture, linear);
        let sampler_key = SamplerKey::from_texture(texture);
        if !self.samplers.contains_key(&sampler_key) {
            let sampler = new_sampler(&self.device, sampler_key)?;
            self.samplers.insert(sampler_key, sampler);
        }
        let frame = self.frame;
        if let Some(cached) = self.textures.get_mut(&key) {
            cached.last_used = frame;
            if let Some(stream) = &texture.stream
                && stream.frames() != cached.stream_frames
                && let Some(seq) =
                    self.streams
                        .write(&self.device, &cached.texture, texture, stream)?
            {
                cached.stream_frames = seq;
            }
            return Ok(false);
        }
        // An unsupported format is not fatal: the material falls back to its
        // flat colour, which is a great deal more useful than no frame at all.
        let Ok(gpu) = upload_texture_as(&self.device, texture, linear) else {
            return Ok(false);
        };
        let mut stream_frames = 0;
        if let Some(stream) = &texture.stream {
            stream_frames = self.streams.write(&self.device, &gpu, texture, stream)?.unwrap_or(0);
        }
        self.textures.insert(
            key,
            CachedTexture {
                _texture: texture.clone(),
                texture: gpu,
                sampler_key,
                last_used: frame,
                stream_frames,
            },
        );
        Ok(true)
    }

    /// The texture + sampler to bind for `material`, falling back to the 1x1
    /// white texture so the shader can sample unconditionally.
    pub(super) fn material_texture(&self, material: &Material) -> (Id, Id) {
        let slots = material.texture_slots();
        // A matcap's own texture has its own slot (`fs_mesh`'s `matcap_tex`).
        let map = slots.map.as_ref();
        if let Some(map) = map
            && let Some(cached) = self.textures.get(&(Arc::as_ptr(map) as usize))
            && let Some(sampler) = self.samplers.get(&cached.sampler_key)
        {
            return (cached.texture.id(), sampler.id());
        }
        // No colour map: the white stand-in, sampled with the first map
        // present's settings (`MaterialTextureSlots::sampler_source`), which
        // the other slots share.
        let from_maps = slots
            .sampler_source()
            // Cached under its role: sRGB for colour slots, linear for data.
            .and_then(|t| {
                self.textures
                    .get(&texture_key(t, false))
                    .or_else(|| self.textures.get(&texture_key(t, true)))
            })
            .and_then(|cached| self.samplers.get(&cached.sampler_key));
        let sampler = from_maps
            .or_else(|| self.samplers.get(&SamplerKey::default_linear()))
            .map(|s| s.id())
            .unwrap_or(NIL);
        (self.white.id(), sampler)
    }

    /// One uploaded texture and its sampler, or `None` if it has not been
    /// uploaded yet.
    ///
    /// Separate from [`material_texture`](Self::material_texture) because the
    /// extra slots have no fallback worth binding: a shader that is told the
    /// slot is empty never samples it, so binding white there would only hide
    /// an upload that did not happen.
    pub(super) fn slot_texture(&self, map: Option<&Arc<Texture>>, linear: bool) -> Option<(Id, Id)> {
        let map = map?;
        let cached = self.textures.get(&texture_key(map, linear))?;
        let sampler = self.samplers.get(&cached.sampler_key)?;
        Some((cached.texture.id(), sampler.id()))
    }

    /// Upload `scene.environment` if it is not the one already held.
    ///
    /// `scene.environment_cube_rt` — a live cube render target, fed by a
    /// `CubeCamera` — is not handled: it lives in the wgpu renderer's own
    /// resource table and this backend has no view of it. A scene using one
    /// gets no environment here rather than a wrong one.
    pub(super) fn ensure_environment(&mut self, scene: &Scene) {
        // Prefiltered, as three.js's `WebGLCubeUVMaps` does for every
        // Standard/Physical environment.
        let cube = scene.environment.as_ref().map(|c| self.prefiltered(c));
        // three.js `scene.environmentIntensity` and `environmentRotation`.
        self.env_light = (
            scene.environment_intensity,
            crate::materials::environment_lookup_rows(
                scene.environment_rotation,
                cube.as_ref().is_some_and(|c| c.cube_uv_atlas.is_none()),
            ),
        );
        let Some(cube) = cube.as_ref() else {
            self.env_cube = None;
            return;
        };
        let key = crate::renderer::gpu_texture::cube_cache_key(cube);
        if matches!(&self.env_cube, Some((k, _)) if *k == key) {
            return;
        }
        let sk = SamplerKey::environment();
        if !self.samplers.contains_key(&sk)
            && let Ok(smp) = new_sampler(&self.device, sk)
        {
            self.samplers.insert(sk, smp);
        }
        let sk = SamplerKey::default_linear();
        if !self.samplers.contains_key(&sk)
            && let Ok(smp) = new_sampler(&self.device, sk)
        {
            self.samplers.insert(sk, smp);
        }
        match EnvGpu::upload(&self.device, cube) {
            Ok(env) => self.env_cube = Some((key, env)),
            Err(e) => {
                // A scene that cannot have its environment uploaded is better
                // rendered without one than not at all, but silently is how
                // this backend earned a reputation for being broken.
                eprintln!("threers: metal environment upload failed: {e:?}");
                self.env_cube = None;
            }
        }
    }

    /// three.js `WebGLCubeUVMaps`: a cube texture lighting a Standard or
    /// Physical material is read through its PMREM — prefiltered once and
    /// kept — unless it already carries one.
    pub(super) fn prefiltered(&mut self, cube: &Arc<crate::textures::CubeTexture>) -> Arc<crate::textures::CubeTexture> {
        if cube.cube_uv_atlas.is_some() {
            return cube.clone();
        }
        self.pmrem_cache
            .entry(crate::renderer::gpu_texture::cube_cache_key(cube))
            .or_insert_with(|| crate::PmremGenerator::from_cube(cube.clone()))
            .clone()
    }

    /// [`prefiltered`](Self::prefiltered) once it has run for `cube`.
    pub(super) fn prefiltered_cached(&self, cube: &Arc<crate::textures::CubeTexture>) -> Arc<crate::textures::CubeTexture> {
        self.pmrem_cache
            .get(&crate::renderer::gpu_texture::cube_cache_key(cube))
            .cloned()
            .unwrap_or_else(|| cube.clone())
    }

    /// The extra material maps, in the order the shader binds them:
    /// emissive, normal, metallic-roughness, ambient occlusion.
    ///
    /// glTF packs roughness and metalness into one image — roughness in green,
    /// metalness in blue — and `threers` keeps them as separate slots because
    /// three.js does. When both point at the same texture, which is what a glTF
    /// load produces, one binding carries both. When they differ, roughness
    /// wins the slot and metalness falls back to its scalar, because there is
    /// no third channel to put it in and the alternative is a sixth binding for
    /// a case no real asset produces.
    ///
    /// Each with its own sampler, from the texture's own filter and wrap.
    pub(super) fn material_maps(&self, material: &Material) -> ([Option<(Id, Id)>; 4], [u32; 4]) {
        let slots = material.texture_slots();
        let mr = slots.roughness_map.as_ref().or(slots.metalness_map.as_ref());
        let ids = [
            self.slot_texture(slots.emissive_map.as_ref(), false),
            self.slot_texture(slots.normal_map.as_ref(), true),
            self.slot_texture(mr, true),
            self.slot_texture(slots.ao_map.as_ref(), true),
        ];
        let present = std::array::from_fn(|i| ids[i].is_some() as u32);
        (ids, present)
    }
}
