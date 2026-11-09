use super::*;

impl Renderer {
    /// three.js r165's `renderTransmissionPass`, when the scene has a visible
    /// transmissive material: the opaque scene drawn linear and untone-mapped
    /// into a half-float target the size of this one, its mips filled; then,
    /// for double-sided transmissive materials, their back faces drawn over it
    /// (reading a copy of it) and the mips filled again. The frame group then
    /// binds it for the colour pass that follows.
    pub(super) fn capture_transmission(&mut self, scene: &mut Scene, camera: &dyn Camera) {
        let (mut any, mut double) = (false, false);
        let mut see = |m: &Material| {
            if m.transmissive() {
                any = true;
                double |= m.side() == 2;
            }
        };
        scene
            .arena
            .traverse_visible(scene.root, &mut |_, obj| match &obj.kind {
                ObjectKind::Mesh(mesh) => mesh.draws().into_iter().for_each(|i| see(i.material)),
                ObjectKind::InstancedMesh(im) if !im.transforms.is_empty() => {
                    im.draws().into_iter().for_each(|i| see(i.material))
                }
                ObjectKind::SkinnedMesh(sm) => sm.draws().into_iter().for_each(|i| see(i.material)),
                _ => {}
            });
        if !any {
            if self.transmission.take().is_some() {
                self.frame_bind_group = self.frame_bind_group_dummy.clone();
            }
            return;
        }
        let (w, h) = self.depth_size;
        if self.transmission.as_ref().map(|c| c.size) != Some((w, h)) {
            let levels = 32 - w.max(h).max(1).leading_zeros();
            let make = |label: &str, usage: wgpu::TextureUsages| {
                self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: levels,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage,
                    view_formats: &[],
                })
            };
            let texture = make(
                "threers transmission capture",
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
            );
            let copy = make(
                "threers transmission capture copy",
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            );
            let depth = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers transmission depth"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.transmission = Some(TransmissionCapture {
                view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
                target_view: texture.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    ..Default::default()
                }),
                copy_view: copy.create_view(&wgpu::TextureViewDescriptor::default()),
                depth_view: depth.create_view(&wgpu::TextureViewDescriptor::default()),
                texture,
                copy,
                _depth: depth,
                size: (w, h),
                levels,
            });
        }
        let cap = self.transmission.as_ref().expect("just made");
        let (target, depth_view) = (cap.target_view.clone(), cap.depth_view.clone());
        let orig_depth = std::mem::replace(&mut self.depth_view, depth_view);
        let tone = std::mem::replace(&mut self.tone_map_pass, false);
        self.frame_bind_group = self.frame_bind_group_dummy.clone();

        self.capturing_transmission = Some(false);
        self.render_inner(scene, camera, &target, true, None, None);
        self.fill_capture_mips();
        if double {
            let cap = self.transmission.as_ref().expect("made above");
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("threers transmission copy"),
                });
            for level in 0..cap.levels {
                let size = wgpu::Extent3d {
                    width: (w >> level).max(1),
                    height: (h >> level).max(1),
                    depth_or_array_layers: 1,
                };
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &cap.texture,
                        mip_level: level,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &cap.copy,
                        mip_level: level,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    size,
                );
            }
            self.queue.submit([encoder.finish()]);
            self.frame_bind_group = frame_bg(
                &self.device,
                &self.frame_bgl,
                &self.frame_buffer_for_bg,
                &cap.copy_view,
                &self.transmission_sampler,
                "threers frame bg (back faces)",
            );
            let clear = std::mem::replace(&mut self.viewport_state.settings.auto_clear, false);
            self.capturing_transmission = Some(true);
            self.render_inner(scene, camera, &target, true, None, None);
            self.viewport_state.settings.auto_clear = clear;
            self.fill_capture_mips();
        }
        self.capturing_transmission = None;
        self.tone_map_pass = tone;
        self.depth_view = orig_depth;
        let cap = self.transmission.as_ref().expect("made above");
        self.frame_bind_group = frame_bg(
            &self.device,
            &self.frame_bgl,
            &self.frame_buffer_for_bg,
            &cap.view,
            &self.transmission_sampler,
            "threers frame bg",
        );
    }

    pub(super) fn fill_capture_mips(&mut self) {
        let Some(cap) = &self.transmission else {
            return;
        };
        let mipgen = self
            .mipgen
            .get_or_insert_with(|| MipGenerator::new(&self.device));
        mipgen.generate(
            &self.device,
            &self.queue,
            &cap.texture,
            wgpu::TextureFormat::Rgba16Float,
            cap.levels,
        );
    }

    /// (Re)create the OIT accumulation + revealage targets at `w×h` and their
    /// resolve bind group. No-op if already the right size.
    /// (Re)create the mipmapped opaque-scene capture used by the screen-space
    /// glass pass. Sized `w×h` in `format` (must match the framebuffer format so
    /// the opaque pipelines can render into mip 0). The mip chain feeds the
    /// roughness→LOD blur of the refracted/reflected image.
    /// The scratch a layer of overlapping glass draws into (`ss_layer`), the
    /// capture's size and format.
    pub(super) fn ensure_ss_layer(&mut self) {
        let Some(c) = self.ss_color.as_ref() else {
            return;
        };
        if matches!(&self.ss_layer, Some((w, h, t, _)) if *w == c.width && *h == c.height && t.format() == c.format)
        {
            return;
        }
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers ss glass layer"),
            size: wgpu::Extent3d {
                width: c.width,
                height: c.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: c.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.ss_layer = Some((c.width, c.height, texture, view));
    }

    pub(super) fn ensure_ss_targets(&mut self, w: u32, h: u32, format: wgpu::TextureFormat) {
        let (w, h) = (w.max(1), h.max(1));
        if let Some(c) = &self.ss_color
            && c.width == w
            && c.height == h
            && c.format == format
        {
            return;
        }
        // ~8 mips is plenty for a roughness blur; cap so the smallest is ≥1px.
        let max_mips = 32 - (w.max(h)).leading_zeros();
        let mips = max_mips.clamp(1, 8);
        let unorm = format.remove_srgb_suffix();
        let reinterpret =
            unorm != format && super::super::caps::supports_view_formats(&self.device);
        let unorm_formats = [unorm];
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers ss color capture"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            // COPY_DST: a layer of overlapping glass is copied in (`ss_layer`).
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: if reinterpret { &unorm_formats } else { &[] },
        });
        let all_mips_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mip0_unorm = reinterpret.then(|| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("threers ss mip 0 unorm"),
                format: Some(unorm),
                base_mip_level: 0,
                mip_level_count: Some(1),
                ..Default::default()
            })
        });
        let mut mip_render_views = Vec::with_capacity(mips as usize);
        let mut mip_sample_views = Vec::with_capacity(mips as usize);
        for m in 0..mips {
            mip_render_views.push(texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("threers ss mip render"),
                base_mip_level: m,
                mip_level_count: Some(1),
                ..Default::default()
            }));
            mip_sample_views.push(texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("threers ss mip sample"),
                base_mip_level: m,
                mip_level_count: Some(1),
                ..Default::default()
            }));
        }
        self.ss_color = Some(SsColor {
            width: w,
            height: h,
            texture,
            all_mips_view,
            mip_render_views,
            mip_sample_views,
            mip0_unorm,
            mips,
            format,
        });

        // Sampleable copy of the opaque depth (marched by the glass shader).
        let need_depth = match &self.ss_depth {
            Some((cw, ch, ..)) => *cw != w || *ch != h,
            None => true,
        };
        if need_depth {
            let dtex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers ss depth copy"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_FORMAT,
                // a render target too, for where it is filled by drawing (`depth_copy`)
                usage: wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let dview = dtex.create_view(&wgpu::TextureViewDescriptor::default());
            self.ss_depth = Some((w, h, dtex, dview));
            // Glass back-face depth target (rendered into, then sampled).
            let btex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers ss back depth"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let bview = btex.create_view(&wgpu::TextureViewDescriptor::default());
            self.ss_back_depth = Some((w, h, btex, bview));
            let stex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers glass shadows"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: GLASS_SHADOW_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let sview = stex.create_view(&wgpu::TextureViewDescriptor::default());
            self.ss_glass_shadow = Some((stex, sview));
        }
    }

    pub(super) fn ensure_taa_targets(&mut self, w: u32, h: u32, format: wgpu::TextureFormat) {
        let (w, h) = (w.max(1), h.max(1));
        // The format is part of the match, not just the size: turning
        // convergence on switches the history to float, and a pair of buffers
        // kept because they were merely the right *shape* would leave the
        // accumulation back in eight bits with nothing to show for it.
        if let Some((cw, ch, ..)) = &self.taa_history
            && *cw == w
            && *ch == h
            && self.taa_hist_format == Some(format)
        {
            return;
        }
        self.taa_hist_format = Some(format);
        // The cached views and bind groups point at the buffers about to be
        // replaced.
        self.taa_bind_cache = None;
        let mk = || {
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers taa history"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            (tex, view)
        };
        let (a, av) = mk();
        let (b, bv) = mk();
        self.taa_history = Some((w, h, a, av, b, bv));
        self.taa_frame = 0;
    }

    pub(super) fn ensure_oit_targets(&mut self, w: u32, h: u32) {
        let (w, h) = (w.max(1), h.max(1));
        if let Some((cw, ch, ..)) = &self.oit_targets
            && *cw == w
            && *ch == h
        {
            return;
        }
        let mk = |fmt: wgpu::TextureFormat, label: &str| {
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: fmt,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            (tex, view)
        };
        let (accum, accum_view) = mk(wgpu::TextureFormat::Rgba16Float, "threers oit accum");
        let (reveal, reveal_view) = mk(OIT_REVEAL_FORMAT, "threers oit reveal");
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers oit resolve bg"),
            layout: &self.oit_resolve_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&accum_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&reveal_view),
                },
            ],
        });
        self.oit_targets = Some((w, h, accum, accum_view, reveal, reveal_view, bg));
    }

    /// The OIT accumulation's multisampled targets, resolved into
    /// `oit_targets`.
    pub(super) fn ensure_oit_msaa_targets(&mut self, w: u32, h: u32) {
        let (w, h) = (w.max(1), h.max(1));
        if !take_sized(
            &mut self.oit_msaa_targets,
            &mut self.oit_msaa_targets_spare,
            w,
            h,
        ) {
            return;
        }
        let mk = |format: wgpu::TextureFormat, label: &str| {
            self.device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: MSAA_SAMPLES,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        let accum = mk(wgpu::TextureFormat::Rgba16Float, "threers oit accum msaa");
        let reveal = mk(OIT_REVEAL_FORMAT, "threers oit reveal msaa");
        self.oit_msaa_targets = Some((w, h, accum, reveal));
    }

    pub(super) fn ensure_msaa_targets(&mut self, w: u32, h: u32) {
        if !take_sized(&mut self.msaa_targets, &mut self.msaa_targets_spare, w, h) {
            return;
        }
        // The colour also viewable as its `Unorm` twin when it is sRGB, for
        // the encoded-blend pass.
        let unorm = self.color_format.remove_srgb_suffix();
        let reinterpret =
            unorm != self.color_format && super::super::caps::supports_view_formats(&self.device);
        let extra = [unorm];
        // The depth also sampled where the refraction pass resolves it.
        let sampled_depth = self.depth_resolve.is_some();
        let mk = |label, format, view_formats: &[wgpu::TextureFormat]| {
            let sampled = sampled_depth && format == DEPTH_FORMAT;
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: self.msaa,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: if sampled {
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
                } else {
                    wgpu::TextureUsages::RENDER_ATTACHMENT
                },
                view_formats,
            })
        };
        // wgpu keeps a texture alive as long as one of its views exists, so we can
        // store only the views.
        let color = mk(
            "threers msaa color",
            self.color_format,
            if reinterpret { &extra } else { &[] },
        );
        let cv = color.create_view(&Default::default());
        self.msaa_color_unorm = reinterpret.then(|| {
            color.create_view(&wgpu::TextureViewDescriptor {
                format: Some(unorm),
                ..Default::default()
            })
        });
        let dv = mk("threers msaa depth", DEPTH_FORMAT, &[]).create_view(&Default::default());
        self.msaa_targets = Some((w, h, cv, dv));
    }

    /// Render `scene` from `camera` into one face of a cube render target.
    /// Used by `CubeCamera.update()` which calls this 6 times per frame.
    /// Reallocate the directional shadow map if a caster asks for a different
    /// size. Powers of two only, and bounded: this is a render target, and a
    /// caller who types 65536 should get the largest sane map rather than an
    /// allocation failure three frames later.
    /// Size the shadow arrays for this frame's casters: a layer per
    /// directional and spot light, six per point light, at the size the
    /// directional lights ask for (`ShadowSettings::map_size`, the largest)
    /// and [`SPOT_SHADOW_SIZE`] for spots, shrunk when several share a kind
    /// ([`shadow_layer_size`]). Reallocated only to grow or resize.
    pub(super) fn ensure_shadow_arrays(
        &mut self,
        dir_size: u32,
        n_dir: usize,
        n_spot: usize,
        n_point: usize,
    ) {
        let mut changed = false;
        let mut fit = |array: &mut ShadowArray, label: &str, size: u32, layers: u32| {
            if array.size != size || array.layers < layers {
                *array = ShadowArray::new(&self.device, label, size, layers);
                changed = true;
            }
        };
        if n_dir > 0 {
            fit(
                &mut self.dir_shadows,
                "threers shadow depth",
                shadow_layer_size(dir_size, n_dir),
                n_dir as u32,
            );
        }
        if n_spot > 0 {
            fit(
                &mut self.spot_shadows,
                "threers spot shadow depth",
                shadow_layer_size(SPOT_SHADOW_SIZE, n_spot),
                n_spot as u32,
            );
        }
        if n_point > 0 {
            fit(
                &mut self.point_shadows,
                "threers point shadow depth",
                POINT_SHADOW_SIZE,
                6 * n_point as u32,
            );
        }
        if changed {
            // The environment bind group holds the old views. Invalidating the
            // cache key is what makes the next frame rebuild it; without this
            // the shader would keep sampling a texture nothing renders into
            // any more.
            self.env_cached_key = None;
            self.env_cached_rt_id = None;
            self.env_dirty = true;
        }
    }

    /// Room for `n` shadow passes' uniforms (`ShadowPassGpu`), one aligned
    /// slot each, and the group 0 the shadow pipelines read them through at
    /// a dynamic offset — beside the frame uniforms, which the passes' vertex
    /// stages may read too.
    ///
    /// One buffer for every pass, written once a frame, rather than a copy
    /// of the whole frame uniforms per pass: the passes are all encoded
    /// before anything runs and `queue.write_buffer` lands before any
    /// submitted command, so each pass needs its own slot either way.
    pub(super) fn ensure_shadow_pass_buffer(&mut self, n: usize) {
        let need = self.shadow_pass_stride * n.max(1) as u64;
        if self
            .shadow_pass_buffer
            .as_ref()
            .is_some_and(|(b, _)| b.size() >= need)
        {
            return;
        }
        let size = need.max(self.shadow_pass_stride * 8);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threers shadow pass uniforms"),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers shadow pass bg"),
            layout: &self.shadow_pass_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.frame_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<ShadowPassGpu>() as u64),
                    }),
                },
            ],
        });
        self.shadow_pass_buffer = Some((buffer, bg));
    }
}
