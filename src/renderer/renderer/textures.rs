use super::*;

impl Renderer {
    /// Upload a texture now, without drawing anything with it.
    ///
    /// Textures normally upload lazily, on the first draw that binds them. That
    /// is the wrong moment when the pixels are large and numerous: each has to
    /// be held in system memory until something draws it, so the peak is the
    /// sum of all of them. Preloading one at a time and swapping in
    /// [`Texture::handle`] keeps the peak at one.
    ///
    /// ```no_run
    /// # use threers::{HeadlessRenderer, Texture, TextureFormat};
    /// # use std::sync::Arc;
    /// # let mut hr = HeadlessRenderer::builder().size(64, 64).build().unwrap();
    /// # let load = |_: &str| Texture::new(1, 1, TextureFormat::Rgba8Unorm, vec![0; 4]);
    /// let full = Arc::new(load("tile.tex"));
    /// hr.renderer().preload_texture(&full);
    /// let light = Arc::new(full.handle()); // same id, no pixels
    /// drop(full);                          // system memory released
    /// ```
    ///
    /// A preloaded texture is kept however long nothing draws it: its handles
    /// have no pixels to upload it again from.
    /// [`clear_texture_cache`](Self::clear_texture_cache) lets it go.
    pub fn preload_texture(&mut self, tex: &std::sync::Arc<crate::textures::Texture>) {
        self.tex_pinned.insert(tex.id);
        self.ensure_slot_uploaded(&Some(tex.clone()));
    }

    /// Drop every cached GPU texture, preloaded ones included.
    pub fn clear_texture_cache(&mut self) {
        self.tex_cache.clear();
        self.tex_last_used.clear();
        self.tex_pinned.clear();
        self.cube_cache.clear();
    }

    /// How many frames a texture nothing draws keeps its GPU copy before it
    /// is let go (default [`DEFAULT_CACHE_RETENTION`](super::DEFAULT_CACHE_RETENTION));
    /// `u64::MAX` keeps every texture ever drawn. The Metal backend's
    /// `MetalRenderer::set_cache_retention` is the same setting.
    pub fn set_cache_retention(&mut self, frames: u64) {
        self.cache_retention = frames;
    }

    /// How many GPU textures are currently cached.
    pub fn cached_texture_count(&self) -> usize {
        self.tex_cache.len()
    }

    pub(super) fn ensure_slot_uploaded(&mut self, slot: &Option<Arc<Texture>>) {
        let Some(t) = slot else {
            return;
        };
        // Skip upload for render-target-backed textures — the view lives in
        // the RenderTarget itself.
        if t.external_rt_id.is_some() {
            return;
        }
        if let Some(stream) = &t.stream {
            self.ensure_stream_uploaded(t, stream);
            return;
        }
        let key = tex_cache_key(t);
        self.tex_last_used.insert(key, self.instance_frame);
        // No liveness check any more: the key is the texture's own id, which
        // does not depend on where — or whether — its bytes are. That is what
        // lets a data-less `Texture::handle` resolve to the upload its full
        // twin made, and it closes the recycled-address hole the weak handle
        // was guarding, since an id is never handed out twice.
        if !self.tex_cache.contains_key(&key) {
            // Drop stale GPU copies for this logical texture (older upload_seq).
            self.tex_cache.retain(|k, _| k.0 != t.id);
            self.tex_bg_cache.clear();
            self.variant_bg_cache.clear();
            if t.is_handle() {
                // A handle carries no pixels: it can only ever resolve to an
                // upload its full twin already made. Reaching here means that
                // upload is missing, and what gets created instead is a texture
                // of the right size with nothing in it -- which samples black
                // and raises no error anywhere.
                log::warn!(
                    "uploading a data-less handle: id {} {}x{} {:?} -- this samples black",
                    t.id,
                    t.width,
                    t.height,
                    t.format
                );
                if std::env::var("TEXWARN").is_ok() {
                    eprintln!(
                        "   WARN data-less handle uploaded: id {} {}x{} {:?}",
                        t.id, t.width, t.height, t.format
                    );
                }
            }
            // Downsampling is a texture read and a write, so let the GPU do it
            // rather than computing the whole chain serially on this thread and
            // uploading it level by level.
            let mipgen = self
                .mipgen
                .get_or_insert_with(|| MipGenerator::new(&self.device));
            let gt = GpuTexture::upload_with_mipgen(&self.device, &self.queue, t, mipgen);
            self.tex_cache.insert(key, gt);
            self.tex_bg_cache.clear();
            self.variant_bg_cache.clear();
        }
    }

    /// A streamed texture: made once, then the newest frame put into it
    /// whenever the stream has moved on: RGBA copied in, YUV converted by a
    /// GPU pass, and the mip chain rebuilt after. The key, the view and the
    /// bind groups stay the same from frame to frame.
    /// Let go of the GPU copies of textures nothing has drawn for
    /// `cache_retention` frames, and of every bind group, which may hold them.
    pub(super) fn evict_unused_textures(&mut self, now: u64) {
        if self.cache_retention == u64::MAX {
            return;
        }
        let cutoff = now.saturating_sub(self.cache_retention);
        let pinned = &self.tex_pinned;
        let last = &mut self.tex_last_used;
        last.retain(|k, &mut frame| frame >= cutoff || pinned.contains(&k.0));
        let before = self.tex_cache.len();
        self.tex_cache
            .retain(|k, _| last.contains_key(k) || pinned.contains(&k.0));
        if self.tex_cache.len() != before {
            self.tex_bg_cache.clear();
            self.variant_bg_cache.clear();
            let live: std::collections::HashSet<u64> = self.tex_cache.keys().map(|k| k.0).collect();
            self.stream_uploads.retain(|id, _| live.contains(id));
        }
    }

    pub(super) fn ensure_stream_uploaded(
        &mut self,
        t: &Arc<Texture>,
        stream: &crate::textures::TextureStream,
    ) {
        use crate::textures::StreamFrame;
        let key = tex_cache_key(t);
        self.tex_last_used.insert(key, self.instance_frame);
        let fresh = !self.tex_cache.contains_key(&key);
        if fresh {
            self.tex_cache.retain(|k, _| k.0 != t.id);
            let max = self.device.limits().max_texture_dimension_2d;
            self.tex_cache
                .insert(key, GpuTexture::streamed(&self.device, t, max));
            self.tex_bg_cache.clear();
            self.variant_bg_cache.clear();
        }
        if !fresh && self.stream_uploads.get(&t.id) == Some(&stream.frames()) {
            return;
        }
        let (width, height) = stream.size();
        if (width, height) != (t.width, t.height) {
            log::warn!(
                "a {}x{} stream on a {}x{} texture: frames are not shown",
                width,
                height,
                t.width,
                t.height
            );
            self.stream_uploads.insert(t.id, stream.frames());
            return;
        }
        let mut scratch = std::mem::take(&mut self.stream_scratch);
        let gpu = &self.tex_cache[&key];
        let format = gpu.texture.format();
        let yuv = &mut self.yuv;
        let (written, drew) = stream.read_frame(|frame, seq| match frame {
            // Nothing written yet: opaque black rather than the zeroed
            // texture's transparent black.
            StreamFrame::Rgba(rgba) if rgba.is_empty() => {
                if fresh {
                    let view = gpu.texture.create_view(&wgpu::TextureViewDescriptor {
                        mip_level_count: Some(1),
                        ..Default::default()
                    });
                    let mut encoder = self.device.create_command_encoder(&Default::default());
                    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("threers stream clear"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        ..Default::default()
                    });
                    self.queue.submit(Some(encoder.finish()));
                }
                (seq, fresh)
            }
            StreamFrame::Rgba(rgba) => {
                gpu.write_frame(&self.queue, rgba, width, height, t.flip_y, &mut scratch);
                (seq, true)
            }
            // The browser's own decoded picture, copied GPU to GPU.
            #[cfg(target_arch = "wasm32")]
            StreamFrame::Browser(image) => {
                use crate::textures::BrowserImage;
                let size = (gpu.texture.width(), gpu.texture.height());
                if image.size() != size {
                    return (seq, false);
                }
                let source = match image {
                    BrowserImage::Video(v) => {
                        wgpu::ExternalImageSource::HTMLVideoElement(Clone::clone(v))
                    }
                    // WebGL2 cannot take a `VideoFrame` (wgpu's GL backend
                    // needs web-sys's unstable APIs for it): the frame is drawn
                    // into a canvas, which the browser keeps on the GPU, and
                    // the canvas copied.
                    BrowserImage::Frame(f)
                        if self.device.adapter_info().backend == wgpu::Backend::Gl =>
                    {
                        match frame_canvas(&mut self.frame_canvas, f, size) {
                            Some(canvas) => wgpu::ExternalImageSource::HTMLCanvasElement(canvas),
                            None => return (seq, false),
                        }
                    }
                    // The Rust handle, not `VideoFrame.clone()`, which makes a second frame.
                    BrowserImage::Frame(f) => {
                        wgpu::ExternalImageSource::VideoFrame(Clone::clone(f))
                    }
                };
                self.queue.copy_external_image_to_texture(
                    &wgpu::CopyExternalImageSourceInfo {
                        source,
                        origin: wgpu::Origin2d::ZERO,
                        flip_y: t.flip_y,
                    },
                    wgpu::CopyExternalImageDestInfo {
                        texture: &gpu.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                        color_space: wgpu::PredefinedColorSpace::Srgb,
                        premultiplied_alpha: false,
                    },
                    wgpu::Extent3d {
                        width: size.0,
                        height: size.1,
                        depth_or_array_layers: 1,
                    },
                );
                (seq, true)
            }
            StreamFrame::Yuv(planes) => {
                yuv.get_or_insert_with(|| super::super::yuv::YuvConverter::new(&self.device))
                    .convert(
                        &self.device,
                        &self.queue,
                        t.id,
                        planes,
                        &gpu.texture,
                        format,
                        t.flip_y,
                    );
                (seq, true)
            }
        });
        self.stream_scratch = scratch;
        self.stream_uploads.insert(t.id, written);
        let levels = gpu.texture.mip_level_count();
        if drew && levels > 1 && MipGenerator::renderable(format) {
            let texture = gpu.texture.clone();
            self.mipgen
                .get_or_insert_with(|| MipGenerator::new(&self.device))
                .generate(&self.device, &self.queue, &texture, format, levels);
        }
    }

    pub(super) fn view_for<'a>(
        &'a self,
        slot: &Option<Arc<Texture>>,
        default: &'a GpuTexture,
    ) -> &'a wgpu::TextureView {
        match slot {
            Some(t) => {
                // If this Texture is RT-backed, sample from the RT's color view.
                if let Some(rt_id) = t.external_rt_id
                    && let Some(view) = self.rt_view_cache.get(&rt_id)
                {
                    return view;
                }
                &self.tex_cache[&tex_cache_key(t)].view
            }
            None => &default.view,
        }
    }

    /// Borrow an RT's sample-side view by id. Caller must have previously
    /// `register_render_target`'d this id. Used by post-fx to bind the input.
    pub fn rt_view_cache_get(&self, rt_id: u32) -> &wgpu::TextureView {
        self.rt_view_cache
            .get(&rt_id)
            .expect("rt id not registered")
    }

    /// Register a CubeRenderTarget so scene.environment_cube_rt can sample
    /// from its 6-face cube view as an env map.
    pub fn register_cube_render_target(
        &mut self,
        rt_id: u32,
        rt: &Arc<super::super::CubeRenderTarget>,
    ) {
        let sample_format = rt.format.add_srgb_suffix();
        let view = rt.color_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("threers cube rt sample view"),
            format: Some(sample_format),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            base_array_layer: 0,
            array_layer_count: Some(6),
            base_mip_level: 0,
            mip_level_count: Some(1),
            aspect: wgpu::TextureAspect::All,
            usage: None,
        });
        self.cube_rt_view_cache.insert(rt_id, view);
    }

    /// Register a RenderTarget so RT-backed Textures can sample from its color view.
    /// Creates a sample-side view using the sRGB-variant of the storage format
    /// (when applicable) so the hardware does sRGB→linear decode on read,
    /// matching three.js's WebGL behavior where RT textures auto-decode.
    pub fn register_render_target(&mut self, rt_id: u32, rt: &Arc<super::super::RenderTarget>) {
        let sample_format = rt_sample_format(rt.format);
        let view = rt.color_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("threers rt sample view"),
            format: Some(sample_format),
            ..Default::default()
        });
        self.rt_view_cache.insert(rt_id, view);
        self.tex_bg_cache.clear();
        self.variant_bg_cache.clear();
        let linear_view = rt.color_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("threers rt linear sample view"),
            format: Some(rt.format),
            ..Default::default()
        });
        self.rt_linear_view_cache.insert(rt_id, linear_view);
        let depth_view = rt.depth_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("threers rt depth sample view"),
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        self.rt_depth_view_cache.insert(rt_id, depth_view);
        self.rt_registry.insert(rt_id, Arc::clone(rt));
    }

    pub(super) fn build_tex_bg(
        &self,
        slots: &MaterialTextureSlots,
        morph: Option<&wgpu::TextureView>,
        uv_sets: Option<&wgpu::TextureView>,
    ) -> wgpu::BindGroup {
        let morph = morph.unwrap_or(&self.default_morph.1);
        let uv_sets = uv_sets.unwrap_or(&self.default_uv_sets.1);
        let albedo = self.view_for(&slots.map, &self.default_white_srgb);
        let normal = self.view_for(&slots.normal_map, &self.default_normal);
        let roughness = self.view_for(&slots.roughness_map, &self.default_white_linear);
        let metalness = self.view_for(&slots.metalness_map, &self.default_white_linear);
        let ao = self.view_for(&slots.ao_map, &self.default_white_linear);
        let emissive = self.view_for(&slots.emissive_map, &self.default_white_srgb);
        // Matcap and cloud cover share a binding. WebGPU guarantees only 16
        // sampled textures per fragment stage and we are at that ceiling, so a
        // seventeenth is not available — but the two can never both be set: a
        // matcap belongs to `MatcapMaterial`, a cloud deck to a Standard or
        // Physical surface.
        let matcap = match &slots.cloud_shadow_map {
            Some(_) => self.view_for(&slots.cloud_shadow_map, &self.default_transparent),
            None => self.view_for(&slots.matcap_map, &self.default_white_srgb),
        };

        // One sampler for every map of the material, from the colour map's
        // filter and wrap (or the first map present) — see
        // `MaterialTextureSlots::sampler_source`.
        let chosen = slots.sampler_source();
        let sampler = match chosen {
            Some(t) => self.pick_sampler(t),
            None => &self.samplers[SAMPLER_DEFAULT],
        };

        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers tex bg"),
            layout: &self.tex_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(albedo),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(normal),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(roughness),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(metalness),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(ao),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(emissive),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(matcap),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::TextureView(morph),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(uv_sets),
                },
            ],
        })
    }

    /// Select one of the 4 pre-built samplers based on a Texture's filter/wrap
    /// fields. Mirrors three.js: NearestFilter→nearest, LinearFilter→linear;
    /// RepeatWrapping→repeat, ClampToEdgeWrapping→clamp.
    /// Build material shader variant `v` — its module and group-2 layout — if
    /// it is not built yet.
    pub(super) fn ensure_map_variant(&mut self, v: super::super::material_maps::MapVariant) {
        use super::super::material_maps as mm;
        if self.map_variants.contains_key(&v) {
            return;
        }
        let source = super::super::shader::shader_source_with_maps(&mm::prelude(v));
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("threers material variant shader"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let b = mm::bindings(v);
        let mut entries = vec![
            uv_sets_entry(),
            wgpu::BindGroupLayoutEntry {
                binding: 10,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
        ];
        for (role, t, smp) in &b.maps {
            let visibility = if role.vertex() {
                wgpu::ShaderStages::VERTEX
            } else {
                wgpu::ShaderStages::FRAGMENT
            };
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: *t,
                visibility,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: *smp,
                visibility,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
        }
        if let Some((t, smp)) = b.env {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: t,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::Cube,
                    multisampled: false,
                },
                count: None,
            });
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: smp,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
        }
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: b.xforms,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("threers material variant bgl"),
                entries: &entries,
            });
        self.map_variants.insert(v, (module, bgl));
    }

    /// Group 2 for a draw on material variant `v`: each map with its own
    /// sampler, and the block of their uv transforms.
    pub(super) fn build_variant_tex_bg(
        &self,
        slots: &MaterialTextureSlots,
        morph: Option<&wgpu::TextureView>,
        uv_sets: Option<&wgpu::TextureView>,
        v: super::super::material_maps::MapVariant,
        uniforms: [[f32; 4]; 6],
    ) -> wgpu::BindGroup {
        use super::super::material_maps as mm;
        let b = mm::bindings(v);
        let textures: Vec<&std::sync::Arc<Texture>> = mm::role_textures(slots)
            .into_iter()
            .filter(|(r, _)| v.contains(*r))
            .filter_map(|(_, t)| t)
            .collect();
        let mut rows: Vec<[f32; 4]> = Vec::with_capacity(6 + 2 * textures.len().max(1));
        rows.extend(uniforms);
        // Row 0's w: the uv set the map reads (three.js `Texture.channel`).
        for t in &textures {
            let [r0, r1] = t.uv_transform_rows();
            rows.push([r0[0], r0[1], r0[2], t.channel.min(3) as f32]);
            rows.push([r1[0], r1[1], r1[2], 0.0]);
        }
        if textures.is_empty() {
            rows.extend([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]);
        }
        let xforms = wgpu::util::DeviceExt::create_buffer_init(
            &*self.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("threers material map transforms"),
                contents: bytemuck::cast_slice(&rows),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        );
        let morph = morph.unwrap_or(&self.default_morph.1);
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 10,
                resource: wgpu::BindingResource::TextureView(morph),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::TextureView(
                    uv_sets.unwrap_or(&self.default_uv_sets.1),
                ),
            },
        ];
        for ((_, t, smp), tex) in b.maps.iter().zip(&textures) {
            let slot = Some((*tex).clone());
            entries.push(wgpu::BindGroupEntry {
                binding: *t,
                resource: wgpu::BindingResource::TextureView(
                    self.view_for(&slot, &self.default_white_linear),
                ),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: *smp,
                resource: wgpu::BindingResource::Sampler(self.pick_sampler(tex)),
            });
        }
        if let (Some((t, smp)), Some(cube)) = (b.env, slots.env_map.as_ref()) {
            entries.push(wgpu::BindGroupEntry {
                binding: t,
                resource: wgpu::BindingResource::TextureView(
                    &self.cube_cache[&cube_cache_key(cube)].view,
                ),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: smp,
                resource: wgpu::BindingResource::Sampler(&self.samplers[cube_sampler_index(cube)]),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: b.xforms,
            resource: xforms.as_entire_binding(),
        });
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers material variant bg"),
            layout: &self.map_variants[&v].1,
            entries: &entries,
        })
    }

    /// three.js `WebGLCubeUVMaps`: a cube texture lighting a Standard or
    /// Physical material is read through its PMREM — prefiltered here once,
    /// and kept — unless it already carries one (`PMREMGenerator` output).
    pub(super) fn prefiltered(
        &mut self,
        cube: &std::sync::Arc<CubeTexture>,
    ) -> std::sync::Arc<CubeTexture> {
        if cube.cube_uv_atlas.is_some() {
            return cube.clone();
        }
        self.pmrem_cache
            .entry(cube_cache_key(cube))
            .or_insert_with(|| crate::PmremGenerator::from_cube(cube.clone()))
            .clone()
    }

    /// [`prefiltered`](Self::prefiltered) once it has run for `cube`.
    pub(super) fn prefiltered_cached(
        &self,
        cube: &std::sync::Arc<CubeTexture>,
    ) -> std::sync::Arc<CubeTexture> {
        self.pmrem_cache
            .get(&cube_cache_key(cube))
            .cloned()
            .unwrap_or_else(|| cube.clone())
    }

    pub(super) fn pick_sampler<'a>(&'a self, t: &Texture) -> &'a wgpu::Sampler {
        &self.samplers[sampler_index(t)]
    }

    pub(super) fn make_default_tex_bg(&self) -> wgpu::BindGroup {
        self.build_tex_bg(&MaterialTextureSlots::default(), None, None)
    }
}
