use super::*;

impl MetalRenderer {
    /// Draw `scene` from `camera` into `pass`, on a command buffer of its own.
    ///
    /// Does not block: the command buffer is committed and the call returns. A
    /// readback or a present sequences behind it on the same queue.
    pub fn render(
        &mut self,
        scene: &mut Scene,
        camera: &dyn Camera,
        pass: &PassAttachments,
    ) -> Result<MetalRenderStats, MetalError> {
        self.render_views(scene, &[RenderView::from_camera(camera)], pass)
    }

    /// Draw `scene` once for each view — one for a window, two for the eyes of
    /// a headset.
    ///
    /// Both eyes go through the scene graph, the sort and the encoder **once**
    /// when the views are *layered*: same colour texture, same viewport,
    /// different array slices. The vertex stage then reads its eye out of the
    /// instance id and writes `render_target_array_index`, so the second eye
    /// costs pixels and not CPU. Anything else — a texture or a viewport per
    /// eye — is one pass per view, and the clear happens once per distinct
    /// target so the second pass cannot wipe the first.
    ///
    /// At most [`MAX_VIEWS`] views; extra ones are ignored.
    pub fn render_views(
        &mut self,
        scene: &mut Scene,
        views: &[RenderView],
        pass: &PassAttachments,
    ) -> Result<MetalRenderStats, MetalError> {
        let _pool = AutoreleasePool::new();
        if views.is_empty() {
            return Err(MetalError::Unsupported("render with no views".into()));
        }
        let views = &views[..views.len().min(MAX_VIEWS)];
        // The viewport / scissor / auto-clear state. The default state plans
        // as `legacy` and takes the path below untouched.
        self.viewport_state.buffer_size = (pass.width, pass.height);
        let staged = match self.viewport_state.settings.plan(pass.width, pass.height) {
            None => {
                // Nothing visible: draw nothing, but still present a frame.
                self.present(pass)?;
                return Ok(MetalRenderStats::default());
            }
            Some(p) if p.legacy => None,
            Some(p) => Some(self.viewport_state.stage(&p, views, pass)?),
        };
        // A pane drawn into its own target is copied back afterwards.
        let composite = staged.as_ref().and_then(|s| s.composite);
        let (views, pass): (&[RenderView], &PassAttachments) = match &staged {
            Some(s) => (&s.views, &s.pass),
            None => (views, pass),
        };
        self.frame = self.frame.wrapping_add(1);
        let slot = (self.frame % FRAMES_IN_FLIGHT as u64) as usize;
        self.transients[slot].clear();

        scene.update_world();
        self.ensure_environment(scene);
        let (mut frame_u, mut items, mut stats, casters) = self.collect(scene, views, pass);
        // Each casting light's record — layers, matrices, biases, filter —
        // and the passes that draw its maps, as the wgpu renderer builds them.
        self.ensure_shadow_arrays(&casters)?;
        // `(kind: 0 directional, 1 spot, 2 point; layer; matrix; point light)`.
        let mut shadow_passes: Vec<(u8, usize, [f32; 16], [f32; 4])> = Vec::new();
        let mut next_layer = [0usize; 3];
        for (r, caster) in casters.iter().enumerate() {
            let settings = caster.settings();
            let filter = settings.filter.unwrap_or(self.shadow_type).code();
            let (bias, normal_bias) = (settings.bias.max(0.0), settings.normal_bias.max(0.0));
            let mut rec = ShadowRecordGpu::default();
            match *caster {
                Caster::Dir { dir, settings } => {
                    let texel = 1.0 / self.shadow_size.max(1) as f32;
                    let layer = next_layer[0];
                    let cascades = caster.layers();
                    next_layer[0] += cascades;
                    let angle = if settings.light_size > 0.0 {
                        settings.light_size
                    } else {
                        0.02
                    };
                    let (depth_range, width) = if cascades > 1 {
                        (12.0, 2.0)
                    } else {
                        (settings.camera_far - settings.camera_near, 2.0 * settings.camera_size)
                    };
                    let scale = angle * depth_range / width.max(1e-6);
                    rec.info = [0, layer as u32, cascades as u32, filter];
                    rec.params = [bias, normal_bias, settings.radius.max(0.0) * texel, texel];
                    rec.pcss = [(scale * 0.25).clamp(2.0 * texel, 0.1), 0.0, 0.0, scale];
                    if cascades > 1 {
                        let view = &views[0];
                        for (c, (vp, far)) in crate::lights::shadow::cascade_frusta(
                            &view.view_matrix,
                            &view.projection_matrix,
                            dir,
                            &settings,
                            self.shadow_size,
                        )
                        .into_iter()
                        .enumerate()
                        {
                            rec.vp[c] = vp;
                            rec.extra[c] = far;
                            shadow_passes.push((0, layer + c, vp, [0.0; 4]));
                        }
                    } else {
                        let s = settings.camera_size;
                        // A fixed cube on the origin, matching the wgpu path.
                        let eye = dir.normalize() * (-s * 1.5);
                        let view = Matrix4::look_at(eye, Vector3::ZERO, Vector3::UP);
                        let proj =
                            Matrix4::orthographic(-s, s, s, -s, settings.camera_near, settings.camera_far);
                        let vp = proj.multiply(&view).elements;
                        rec.vp[0] = vp;
                        rec.extra = [f32::MAX; 4];
                        shadow_passes.push((0, layer, vp, [0.0; 4]));
                    }
                }
                Caster::Spot { pos, dir, angle, settings } => {
                    let texel = 1.0 / self.spot_shadow_size.max(1) as f32;
                    let layer = next_layer[1];
                    next_layer[1] += 1;
                    let view = Matrix4::look_at(pos, pos + dir.normalize(), Vector3::UP);
                    // fov = 2 × the outer angle, so the frustum covers the whole cone.
                    let fov = (angle * 2.0).clamp(0.05, std::f32::consts::PI - 0.05);
                    let proj =
                        Matrix4::perspective(fov, 1.0, settings.camera_near, settings.camera_far);
                    let vp = proj.multiply(&view).elements;
                    let size = if settings.light_size > 0.0 { settings.light_size } else { 0.5 };
                    let scale = size / (2.0 * (fov * 0.5).tan());
                    rec.info = [1, layer as u32, 1, filter];
                    rec.params = [bias, normal_bias, settings.radius.max(0.0) * texel, texel];
                    rec.pcss = [(scale * 0.05).clamp(2.0 * texel, 0.1), settings.camera_near, settings.camera_far, scale];
                    rec.vp[0] = vp;
                    shadow_passes.push((1, layer, vp, [0.0; 4]));
                }
                Caster::Point { pos, radius, settings } => {
                    // A face texel (`2 / size` of a unit cube face) per unit
                    // of radius, across the direction to the light.
                    let texel = 2.0 / POINT_SHADOW_SIZE as f32;
                    let k = next_layer[2];
                    next_layer[2] += 6;
                    let at = [pos.x, pos.y, pos.z, radius];
                    let size = if settings.light_size > 0.0 { settings.light_size } else { 0.5 };
                    rec.info = [2, k as u32, 1, filter];
                    rec.params = [bias, normal_bias, settings.radius.max(0.0) * texel, texel];
                    rec.extra = at;
                    rec.pcss = [(size * 0.05).clamp(2.0 * texel, 0.2), 0.0, 0.0, size];
                    // Six faces, +X −X +Y −Y +Z −Z, each a 90° frustum. Metal's
                    // cube convention, which is not the GL one the wgpu path
                    // uses: +Y is up on the side faces rather than −Y. Getting
                    // this wrong leaves each face individually plausible and
                    // the shadow landing in the wrong place.
                    let faces: [(Vector3, Vector3); 6] = [
                        (Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0)),
                        (Vector3::new(-1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0)),
                        (Vector3::new(0.0, 1.0, 0.0), Vector3::new(0.0, 0.0, -1.0)),
                        (Vector3::new(0.0, -1.0, 0.0), Vector3::new(0.0, 0.0, 1.0)),
                        (Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.0, 1.0, 0.0)),
                        (Vector3::new(0.0, 0.0, -1.0), Vector3::new(0.0, 1.0, 0.0)),
                    ];
                    let proj = Matrix4::perspective(std::f32::consts::FRAC_PI_2, 1.0, 0.1, radius.max(1.0));
                    for (face, &(forward, up)) in faces.iter().enumerate() {
                        let view = Matrix4::look_at(pos, pos + forward, up);
                        let vp = mirror_clip_x(proj.multiply(&view)).elements;
                        shadow_passes.push((2, k + face, vp, at));
                    }
                }
            }
            frame_u.shadows[r] = rec;
        }
        frame_u.shadow_info = [casters.len() as u32, 0, 0, 0];
        // What the maps show, for reusing them (`ShadowMap`): taken here,
        // before the uploads below move the instance and bone data out.
        let shadow_content = (!shadow_passes.is_empty()).then(|| shadow_fingerprint(&items));

        // Opaque front-to-back (early-z rejects the overdraw), transparent
        // back-to-front (blending is order-dependent). `render_order` wins over
        // both, as in three.js.
        items.sort_by(|a, b| {
            a.transparent
                .cmp(&b.transparent)
                // three.js's transmissive list sits between the two.
                .then(a.transmissive.cmp(&b.transmissive))
                .then(a.render_order.cmp(&b.render_order))
                .then_with(|| {
                    let (x, y) = if a.transparent {
                        (b.view_depth, a.view_depth)
                    } else {
                        (a.view_depth, b.view_depth)
                    };
                    x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        // Upload before encoding: both touch `self`, and the encoder holds a
        // command buffer that a mipmap blit would otherwise interleave with.
        for item in &mut items {
            let slim = item.primitive == primitive::LINE && slim_lines_enabled();
            if self.ensure_geometry(&item.geometry, slim)? {
                stats.geometry_uploads += 1;
            }
            if item.primitive == primitive::TRIANGLE && item.material.wireframe() {
                let full = item.range == (0, item.geometry.draw_count() as u32);
                if full {
                    self.ensure_wire_indices(&item.geometry)?;
                } else {
                    // A group or a draw range: the line list of that span
                    // alone, for this frame.
                    let flat = item.geometry.wireframe_indices_in(item.range.0, item.range.1);
                    if !flat.is_empty() {
                        let buf = self.device.new_buffer(bytemuck::cast_slice(&flat))?;
                        item.wire_span = Some((self.transients[slot].len(), flat.len()));
                        self.transients[slot].push(buf);
                    }
                }
            }
            let slots = item.material.texture_slots();
            // Every slot the fragment shader can bind. Uploading only the ones
            // it used to sample is how the extra maps stayed invisible after
            // the shader learned to read them: the binding asked the cache for
            // a texture nobody had put there.
            let phys: Vec<(Arc<Texture>, bool)> = physical_maps(&slots)
                .into_iter()
                .filter_map(|(m, l)| m.map(|m| (m.clone(), l)))
                .collect();
            for (map, linear) in [
                (slots.map, false),
                (slots.matcap_map, false),
                // The cloud layer a planet's surface is shadowed by; it
                // shares the matcap slot (no material has both).
                (slots.cloud_shadow_map, false),
                // Heights, read in the vertex stage.
                (slots.displacement_map, true),
                (slots.gradient_map, false),
                (slots.emissive_map, false),
                (slots.normal_map, true),
                (slots.roughness_map, true),
                (slots.metalness_map, true),
                (slots.ao_map, true),
                (slots.alpha_map, true),
                // Colour, decoded by its texture's colour space, as three.js does.
                (slots.light_map, false),
                (slots.bump_map, true),
                (slots.specular_map, true),
            ]
            .into_iter()
            .filter_map(|(m, l)| m.map(|m| (m, l)))
            .chain(phys)
            {
                if self.ensure_texture_as(&map, linear)? {
                    stats.texture_uploads += 1;
                }
            }
            // A custom shader's `maps`, decoded by their own formats.
            if let Material::Shader(m) = &*item.material {
                if !m.volumes.is_empty() && !self.samplers.contains_key(&USER_VOLUME_SAMPLER) {
                    let sampler = new_sampler(&self.device, USER_VOLUME_SAMPLER)?;
                    self.samplers.insert(USER_VOLUME_SAMPLER, sampler);
                }
                // Linear and repeat for `textures` as for `maps`: a cascade
                // or wake map tiles, and an edge-clamped one goes flat.
                if (!m.maps.is_empty() || !m.textures.is_empty()) && !self.samplers.contains_key(&USER_SAMPLER) {
                    let sampler = new_sampler(&self.device, USER_SAMPLER)?;
                    self.samplers.insert(USER_SAMPLER, sampler);
                }
                for map in m.maps.iter().take(4) {
                    if self.ensure_texture_as(map, false)? {
                        stats.texture_uploads += 1;
                    }
                }
            }
            // A Standard/Physical material's own `envMap` prefiltered, as the
            // scene's is; a Basic/Lambert/Phong one sampled as it is.
            let pbr = slots.pbr_env_map.as_ref().map(|c| self.prefiltered(c));
            for cube in [slots.env_map.as_ref(), pbr.as_ref()].into_iter().flatten() {
                let key = crate::renderer::gpu_texture::cube_cache_key(cube);
                if !self.material_envs.contains_key(&key) {
                    if let Ok(env) = EnvGpu::upload(&self.device, cube) {
                        self.material_envs.insert(key, env);
                        stats.texture_uploads += 1;
                    }
                    let sk = SamplerKey::from_cube(cube);
                    if !self.samplers.contains_key(&sk)
                        && let Ok(smp) = new_sampler(&self.device, sk)
                    {
                        self.samplers.insert(sk, smp);
                    }
                }
            }
            if let Some(mut data) = item.instances.take() {
                item.instance_count = data.len() / 16;
                if let Some((rows, row_len)) = item.instance_morph.take() {
                    item.instance_morph_layout = Some((row_len as u32, data.len() as u32));
                    data.extend_from_slice(&rows);
                }
                if let Some(colors) = item.instance_colors.take() {
                    // float4-aligned, after everything else.
                    data.resize(data.len().div_ceil(4) * 4, 0.0);
                    item.instance_color_offset = (data.len() / 4) as u32;
                    data.extend_from_slice(&colors);
                }
                let buf = self.device.new_buffer(bytemuck::cast_slice(&data))?;
                item.instance_slot = Some(self.transients[slot].len());
                self.transients[slot].push(buf);
            }
            if let Some(data) = item.bones.take() {
                // Transient rather than cached: a pose changes every frame, so
                // there is nothing to reuse and a cache would only be a thing
                // to invalidate.
                if !data.is_empty() {
                    let buf = self.device.new_buffer(bytemuck::cast_slice(&data))?;
                    item.bone_slot = Some(self.transients[slot].len());
                    self.transients[slot].push(buf);
                }
            }
        }

        // Whether this render redraws them — see `ShadowMap`. They depend on
        // lights and casters, not the camera, so a render that skips them
        // samples what an identical redraw would have written: with
        // `auto_update` on, a render whose casters and lights match the last
        // one's keeps its maps.
        // Each pass's own content: the casters and its light's matrix. Only
        // the passes whose changed are redrawn — a cascade following the
        // camera does not take a still point light's faces with it.
        let sig = (next_layer[0], next_layer[1], next_layer[2], self.shadow_size, self.spot_shadow_size);
        let valid = self.shadow_sig == Some(sig);
        let pass_hashes: Vec<Option<u64>> = shadow_passes
            .iter()
            .map(|(kind, layer, vp, at)| {
                shadow_content.map(|c| {
                    let mut h = (c.rotate_left(5) ^ (*layer as u64 | (*kind as u64) << 32)).wrapping_mul(0x517c_c1b7_2722_0a95);
                    for v in vp.iter().chain(at) {
                        h = (h.rotate_left(5) ^ v.to_bits() as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
                    }
                    h
                })
            })
            .collect();
        let changed: Vec<bool> = pass_hashes
            .iter()
            .enumerate()
            .map(|(k, h)| !valid || self.shadow_pass_hashes.get(k) != Some(h))
            .collect();
        let update_shadows = !shadow_passes.is_empty() && {
            self.shadow_sig = Some(sig);
            self.shadow_flags.take_update(valid, !changed.iter().any(|&c| c))
        };
        // A forced update (`needs_update` with `auto_update` off) redraws
        // every pass; an automatic one, the ones that changed.
        let draw_pass: Vec<bool> = changed
            .iter()
            .map(|&c| update_shadows && (!self.shadow_flags.auto_update || c))
            .collect();
        if update_shadows {
            self.shadow_pass_hashes = pass_hashes;
        }
        if draw_pass.iter().any(|&d| d) {
            self.shadow_updates += 1;
        }
        // Shadow maps first: the main pass samples them, so they have to be
        // filled before anything reads them. Each pass writes its own matrix
        // into `shadow_vp` and shares one depth-only vertex function, rather
        // than having three that differ only in which matrix they read.
        for (&(kind, layer, vp, at), _) in shadow_passes.iter().zip(&draw_pass).filter(|&(_, &d)| d)
        {
            let map = match kind {
                0 => self.shadow_map.as_ref(),
                1 => self.spot_shadow_map.as_ref(),
                _ => self.point_shadow_map.as_ref(),
            };
            let Some(map) = map.map(|t| t.id()) else { continue };
            let pass_u = ShadowPassGpu { shadow_vp: vp, point_shadow_pos: at };
            // Casters outside this light's frustum are not drawn into it.
            let frustum = crate::math::Frustum::from_projection_matrix(&Matrix4 { elements: vp });
            unsafe { self.encode_shadow(&pass_u, &items, slot, map, layer, kind == 2, &frustum) }?;
        }
        // three.js's `renderTransmissionPass`, when anything is transmissive;
        // the same capture is what a screen-space `ShaderMaterial` reads.
        if items.iter().any(|i| i.transmissive || i.screen_space) {
            self.capture_transmission(pass, views, &frame_u, &mut items, scene, slot, &mut stats)?;
        } else {
            self.transmission = None;
            self.capture_bound = None;
        }
        frame_u.rect_counts[1] = self.transmission.as_ref().map_or(0, |c| c.levels);
        let vp = staged.as_ref().map(|s| (s.clear, s.scissor));
        unsafe { self.encode(pass, views, &frame_u, &items, scene, slot, &mut stats, vp) }?;
        if let Some(c) = composite {
            self.viewport_state.composite(&c)?;
            // The pane pass had the drawable stripped; present after the copy.
            self.present(&c.dest)?;
        }
        self.evict_unused();
        Ok(stats)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn encode(
        &mut self,
        pass: &PassAttachments,
        views: &[RenderView],
        frame_u: &FrameUniforms,
        items: &[DrawItem],
        scene: &Scene,
        slot: usize,
        stats: &mut MetalRenderStats,
        vp: Option<(ClearMode, Option<PixelRect>)>,
    ) -> Result<(), MetalError> {
        let cmd = self.device.command_buffer();
        if cmd.is_null() {
            return Err(MetalError::NoCommandQueue);
        }

        // Order-independent transparency in passes of its own after each
        // view's — not in a layered (stereo) pass, where `Oit` draws sorted
        // with the other transparent surfaces, as `Glass` does. The depth it
        // tests against has to outlive the main pass for it.
        let layered_pass = vp.is_none() && layered(views, pass);
        self.oit_active = !layered_pass
            && !pass.depth.is_null()
            && items.iter().any(|i| i.oit && !i.capture_skip);
        let store_depth = self.store_depth;
        self.store_depth |= self.oit_active;
        let result = unsafe { self.encode_views(
            cmd,
            pass,
            views,
            frame_u,
            items,
            scene,
            slot,
            stats,
            vp,
            layered_pass,
        ) };
        self.store_depth = store_depth;
        self.oit_active = false;
        result?;

        if !pass.drawable.is_null() {
            let _: () = unsafe { msg1(cmd, sel!("presentDrawable:"), pass.drawable) };
        }
        let _: () = unsafe { msg0(cmd, sel!("commit")) };
        Ok(())
    }

    /// `encode`'s passes, one per view (or one layered pass for them all).
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn encode_views(
        &mut self,
        cmd: Id,
        pass: &PassAttachments,
        views: &[RenderView],
        frame_u: &FrameUniforms,
        items: &[DrawItem],
        scene: &Scene,
        slot: usize,
        stats: &mut MetalRenderStats,
        vp: Option<(ClearMode, Option<PixelRect>)>,
        layered_pass: bool,
    ) -> Result<(), MetalError> {
        if layered_pass {
            let descriptor =
                unsafe { self.render_pass_descriptor(pass, scene, &views[0], true, false, views.len() as u32) };
            unsafe { self.encode_pass(
                cmd,
                descriptor,
                pass,
                frame_u,
                views[0].viewport,
                None,
                items,
                slot,
                stats,
                views.len() as u32,
                None,
                None,
            ) }?;
        } else {
            // One pass per view. The clear happens the first time a given
            // (texture, slice) is targeted; a second view sharing it loads
            // instead, or it would erase the first.
            let mut cleared: Vec<(Id, u32)> = Vec::with_capacity(views.len());
            for (index, view) in views.iter().enumerate() {
                let target = (
                    if view.color.is_null() {
                        pass.color
                    } else {
                        view.color
                    },
                    view.slice,
                );
                let first = !cleared.contains(&target);
                cleared.push(target);
                // Under the viewport state the first touch clears only when
                // the clear is the whole target; a scissored clear loads and
                // fills its rectangle, and auto-clear off just loads.
                let (clear, fill) = match vp {
                    None => (first, None),
                    Some((ClearMode::Full, _)) => (first, None),
                    Some((ClearMode::Rect(r), _)) if first => (
                        false,
                        Some((r, clear_color(scene.background, scene.background_alpha))),
                    ),
                    Some(_) => (false, None),
                };
                // A later view loads this target, so a multisampled colour has
                // to survive the resolve — `MULTISAMPLE_RESOLVE` alone discards
                // it and the next pane would load garbage.
                let keep = views[index + 1..].iter().any(|v| {
                    (if v.color.is_null() { pass.color } else { v.color }, v.slice) == target
                });

                // The mono shaders read view 0, so the view being drawn moves
                // into that slot for this pass.
                let mut per_view = *frame_u;
                per_view.promote_view(index);
                // The view now reads as view 0; its capture slice is `index`.
                per_view.rect_counts[3] = index as u32;
                // Pixel-sized effects (wide lines) convert through the viewport
                // they are drawn in, not the whole attachment.
                if let Some(v) = view.viewport {
                    let (w, h) = (v.width.max(1.0) as f32, v.height.max(1.0) as f32);
                    per_view.viewport = [w, h, 1.0 / w, 1.0 / h];
                }
                let descriptor =
                    // Kept multisampled when the OIT composite loads it next.
                    unsafe { self.render_pass_descriptor(pass, scene, view, clear, keep || self.oit_active, 1) };
                unsafe { self.encode_pass(
                    cmd,
                    descriptor,
                    pass,
                    &per_view,
                    view.viewport,
                    Some(view.layers),
                    items,
                    slot,
                    stats,
                    1,
                    fill,
                    vp.and_then(|(_, s)| s),
                ) }?;
                if self.oit_active {
                    unsafe { self.encode_oit(cmd, pass, view, &per_view, items, slot, stats, keep) }?;
                }
            }
        }
        Ok(())
    }

    /// An item's world bounding sphere, where its geometry's bounds bound
    /// what it draws: not instanced, skinned, morphed or a sprite (each
    /// draws somewhere its object-space bounds do not say).
    pub(super) fn item_sphere(&self, item: &DrawItem) -> Option<crate::math::Sphere> {
        if item.instance_count > 1
            || item.instance_slot.is_some()
            || item.bone_slot.is_some()
            || item.sprite.is_some()
            || item.morph_influences.iter().any(|&w| w != 0.0)
        {
            return None;
        }
        let g = self.geometries.get(&(Arc::as_ptr(&item.geometry) as usize))?;
        Some(g.sphere?.apply_matrix4(&item.world))
    }

    /// The OIT surfaces of one view: accumulated against the depth its main
    /// pass left — weighted light added up, transmittance multiplied down —
    /// then composited over its colour, multiplied by the transmittance and
    /// plus the light. Into the same (multisampled) colour the main pass drew
    /// and resolved again, so a later view's resolve keeps the composite.
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn encode_oit(
        &mut self,
        cmd: Id,
        pass: &PassAttachments,
        view: &RenderView,
        frame_u: &FrameUniforms,
        items: &[DrawItem],
        slot: usize,
        stats: &mut MetalRenderStats,
        keep_multisampled: bool,
    ) -> Result<(), MetalError> {
        if !items.iter().any(|i| i.oit && !i.capture_skip && view.layers.test(&i.layers)) {
            return Ok(());
        }
        let samples = pass.sample_count.max(1);
        let (w, h) = (pass.width.max(1), pass.height.max(1));
        if self.oit_targets.as_ref().map(|t| t.size) != Some((w, h, samples)) {
            let usage = texture_usage::RENDER_TARGET | texture_usage::SHADER_READ;
            let mk = |n| {
                crate::metal::resources::new_texture_2d(
                    &self.device,
                    w,
                    h,
                    pixel_format::RGBA16_FLOAT,
                    usage,
                    storage_mode::PRIVATE,
                    n,
                    1,
                )
            };
            let (accum, reveal) = (mk(1)?, mk(1)?);
            let (accum_ms, reveal_ms) = if samples > 1 { (Some(mk(samples)?), Some(mk(samples)?)) } else { (None, None) };
            self.oit_targets = Some(OitTargets { size: (w, h, samples), accum, reveal, accum_ms, reveal_ms });
        }
        let key = (pass.color_format, samples);
        if !self.oit_composite.contains_key(&key) {
            let tint = crate::metal::resources::oit_composite_pipeline(&self.device, pass.color_format, samples, false)?;
            let light = crate::metal::resources::oit_composite_pipeline(&self.device, pass.color_format, samples, true)?;
            self.oit_composite.insert(key, (tint, light));
        }
        let t = self.oit_targets.as_ref().expect("made above");
        let (accum, reveal) = (t.accum.id(), t.reveal.id());
        let (accum_ms, reveal_ms) = (t.accum_ms.as_ref().map(|t| t.id()), t.reveal_ms.as_ref().map(|t| t.id()));

        // 1) Accumulation, against the main pass's depth.
        let rpd: Id = unsafe { msg0(class!("MTLRenderPassDescriptor"), sel!("renderPassDescriptor")) };
        let attachments: Id = unsafe { msg0(rpd, sel!("colorAttachments")) };
        for (index, (single, multi, clear)) in
            [(accum, accum_ms, MTLClearColor { red: 0.0, green: 0.0, blue: 0.0, alpha: 0.0 }),
             (reveal, reveal_ms, MTLClearColor { red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0 })]
            .into_iter()
            .enumerate()
        {
            let a: Id = unsafe { msg1(attachments, sel!("objectAtIndexedSubscript:"), index) };
            let _: () = unsafe { msg1(a, sel!("setLoadAction:"), load_action::CLEAR) };
            let _: () = unsafe { msg1(a, sel!("setClearColor:"), clear) };
            match multi {
                Some(m) => {
                    let _: () = unsafe { msg1(a, sel!("setTexture:"), m) };
                    let _: () = unsafe { msg1(a, sel!("setResolveTexture:"), single) };
                    let _: () = unsafe { msg1(a, sel!("setStoreAction:"), store_action::MULTISAMPLE_RESOLVE) };
                }
                None => {
                    let _: () = unsafe { msg1(a, sel!("setTexture:"), single) };
                    let _: () = unsafe { msg1(a, sel!("setStoreAction:"), store_action::STORE) };
                }
            }
        }
        let depth_texture = if view.depth.is_null() { pass.depth } else { view.depth };
        let depth: Id = unsafe { msg0(rpd, sel!("depthAttachment")) };
        let _: () = unsafe { msg1(depth, sel!("setTexture:"), depth_texture) };
        let _: () = unsafe { msg1(depth, sel!("setSlice:"), view.slice as NSUInteger) };
        let _: () = unsafe { msg1(depth, sel!("setLoadAction:"), load_action::LOAD) };
        let keep_depth = if self.keep_attachments { store_action::STORE } else { store_action::DONT_CARE };
        let _: () = unsafe { msg1(depth, sel!("setStoreAction:"), keep_depth) };
        self.oit_phase = 1;
        let accumulated = unsafe { self.encode_pass(cmd, rpd, pass, frame_u, view.viewport, Some(view.layers), items, slot, stats, 1, None, None) };
        self.oit_phase = 0;
        accumulated?;

        // 2) The composite, over the colour the main pass drew.
        let rpd: Id = unsafe { msg0(class!("MTLRenderPassDescriptor"), sel!("renderPassDescriptor")) };
        let attachments: Id = unsafe { msg0(rpd, sel!("colorAttachments")) };
        let color: Id = unsafe { msg1(attachments, sel!("objectAtIndexedSubscript:"), 0usize) };
        let color_texture = if view.color.is_null() { pass.color } else { view.color };
        let _: () = unsafe { msg1(color, sel!("setTexture:"), color_texture) };
        let _: () = unsafe { msg1(color, sel!("setSlice:"), view.slice as NSUInteger) };
        let _: () = unsafe { msg1(color, sel!("setLoadAction:"), load_action::LOAD) };
        if pass.resolve.is_null() {
            let _: () = unsafe { msg1(color, sel!("setStoreAction:"), store_action::STORE) };
        } else {
            let _: () = unsafe { msg1(color, sel!("setResolveTexture:"), pass.resolve) };
            let _: () = unsafe { msg1(color, sel!("setResolveSlice:"), view.slice as NSUInteger) };
            let _: () = unsafe { msg1(
                color,
                sel!("setStoreAction:"),
                if keep_multisampled || self.keep_attachments {
                    store_action::STORE_AND_MULTISAMPLE_RESOLVE
                } else {
                    store_action::MULTISAMPLE_RESOLVE
                },
            ) };
        }
        let encoder: Id = unsafe { msg1(cmd, sel!("renderCommandEncoderWithDescriptor:"), rpd) };
        if encoder.is_null() {
            return Err(MetalError::NoCommandQueue);
        }
        if let Some(v) = view.viewport {
            let _: () = unsafe { msg1(encoder, sel!("setViewport:"), v) };
        }
        let (tint, light) = &self.oit_composite[&key];
        let _: () = unsafe { msg2(encoder, sel!("setFragmentTexture:atIndex:"), accum, 0usize) };
        let _: () = unsafe { msg2(encoder, sel!("setFragmentTexture:atIndex:"), reveal, 1usize) };
        for pipe in [tint.id(), light.id()] {
            let _: () = unsafe { msg1(encoder, sel!("setRenderPipelineState:"), pipe) };
            let _: () = unsafe { msg3(encoder, sel!("drawPrimitives:vertexStart:vertexCount:"), primitive::TRIANGLE, 0usize, 3usize) };
        }
        let _: () = unsafe { msg0(encoder, sel!("endEncoding")) };
        Ok(())
    }

    /// Encode every item into one render pass.
    ///
    /// `views_per_draw` is 1 for an ordinary pass and the view count for a
    /// layered one, where the instance count is multiplied by it and the vertex
    /// stage splits the id back into (view, instance).
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn encode_pass(
        &mut self,
        cmd: Id,
        descriptor: Id,
        pass: &PassAttachments,
        frame_u: &FrameUniforms,
        viewport: Option<MTLViewport>,
        view_layers: Option<Layers>,
        items: &[DrawItem],
        slot: usize,
        stats: &mut MetalRenderStats,
        views_per_draw: u32,
        fill: Option<(PixelRect, MTLClearColor)>,
        scissor_test: Option<PixelRect>,
    ) -> Result<(), MetalError> {
        // Built before the encoder exists: an error after it would have to
        // end the encoder first.
        let fill_pipeline = match fill {
            Some(_) => Some(self.viewport_state.fill_pipeline(
                pass.color_format,
                pass.depth_format,
                pass.sample_count.max(1),
            )?),
            None => None,
        };
        let encoder: Id = unsafe { msg1(cmd, sel!("renderCommandEncoderWithDescriptor:"), descriptor) };
        if encoder.is_null() {
            return Err(MetalError::Unsupported(
                "render command encoder could not be created for these attachments".into(),
            ));
        }

        let _: () = unsafe { msg1(
            encoder,
            sel!("setFrontFacingWinding:"),
            winding::COUNTER_CLOCKWISE,
        ) };
        // The scissored clear, before anything draws: the background over the
        // scissor rectangle and nowhere else.
        if let (Some((rect, color)), Some(pipeline)) = (fill, fill_pipeline) {
            unsafe { super::super::viewport::encode_fill(encoder, pipeline, rect, color) };
        }
        // With the scissor test on, nothing lands outside its rectangle.
        let mut items = items;
        if let Some(viewport) = viewport {
            let _: () = unsafe { msg1(encoder, sel!("setViewport:"), viewport) };
            // Scissor to the viewport, clamped to the attachment (Metal
            // rejects a scissor outside it), so nothing a pane draws — a wide
            // line, a point sprite — can land in the pane next to it.
            use super::super::viewport::{intersect, scissor_for_viewport, scissor_rect};
            let rect = scissor_for_viewport(&viewport, pass.width, pass.height).and_then(|r| {
                match scissor_test {
                    Some(s) => intersect(r, scissor_rect(s)),
                    None => Some(r),
                }
            });
            match rect {
                Some(rect) => {
                    let _: () = unsafe { msg1(encoder, sel!("setScissorRect:"), rect) };
                }
                // Viewport and scissor miss each other: nothing to draw.
                None => items = &[],
            }
        }
        // The frame uniforms outgrow `setVertexBytes`'s 4 KB with the shadow
        // records, so they go in a buffer of this frame's, kept until the
        // frame in flight is done with it.
        let frame_bytes = unsafe { std::slice::from_raw_parts(
            frame_u as *const FrameUniforms as *const u8,
            std::mem::size_of::<FrameUniforms>(),
        ) };
        let frame_buf = match self.device.new_buffer(frame_bytes) {
            Ok(b) => b,
            Err(e) => {
                let _: () = unsafe { msg0(encoder, sel!("endEncoding")) };
                return Err(e);
            }
        };
        let _: () = unsafe { msg3(encoder, sel!("setVertexBuffer:offset:atIndex:"), frame_buf.id(), 0usize, VB_FRAME) };
        let _: () = unsafe { msg3(encoder, sel!("setFragmentBuffer:offset:atIndex:"), frame_buf.id(), 0usize, FB_FRAME) };
        self.transients[slot].push(frame_buf);

        let mut last_pipeline = NIL;
        let mut last_cull = usize::MAX;
        let mut last_fill = usize::MAX;
        let mut last_depth = NIL;
        let mut last_bias = (0.0f32, 0.0f32);
        // A failure inside the loop still has to close the encoder: Metal
        // aborts the process if one is released without `endEncoding`.
        let mut failure = None;

        // Frustum culling, against this view's view-projection: an item whose
        // bounds are outside it is not encoded at all. Not in a layered pass,
        // where one draw serves both eyes and either could see it.
        let cull = (views_per_draw == 1)
            .then(|| crate::math::Frustum::from_projection_matrix(&Matrix4 { elements: frame_u.view_proj[0] }));
        for item in items {
            if let Some(f) = &cull
                && self.item_sphere(item).is_some_and(|s| !f.intersects_sphere(&s)) {
                    stats.culled += 1;
                    continue;
                }
            // With an OIT pass this frame, its items draw there and nowhere
            // else.
            if self.oit_active && item.oit != (self.oit_phase == 1) {
                continue;
            }
            if item.capture_skip {
                continue;
            }
            if let Some(layers) = view_layers
                && !layers.test(&item.layers) {
                    continue;
                }
            let key = Arc::as_ptr(&item.geometry) as usize;
            // Nothing uploaded for this geometry — an attribute set the vertex
            // builder could make nothing of. Counted, not drawn, not fatal.
            let Some(geom) = self.geometries.get(&key) else {
                stats.skipped += 1;
                continue;
            };
            // Everything needed from the cache entry, read out before anything
            // borrows `self` mutably again.
            let (vertices, indices, index_count, vertex_count, skin_buffer) = (
                geom.vertices.id(),
                geom.indices.as_ref().map(|b| b.id()),
                geom.index_count,
                geom.vertex_count,
                geom.skin.as_ref().map(|b| b.id()),
            );
            let morph_u = item_morph_uniforms(item, geom.morph.as_ref());
            let morph_buffer = geom
                .morph
                .as_ref()
                .filter(|_| morph_u.info[0] > 0)
                .map(|m| m.buffer.id())
                .unwrap_or(self.identity_instance.id());
            let uv_sets = geom.uv_sets.as_ref().map(|(b, f)| (b.id(), *f));
            // A wireframe mesh draws its line-list (polygon outlines, 1px
            // hardware lines, as the wgpu `TriangleWire` path does) rather
            // than its triangles in line fill mode, which would show every
            // fan diagonal. Fill-mode LINES stays only as the fallback for a
            // wireframe with no line list (nothing to outline). A partial
            // span draws the line list built for it.
            let wire = if item.primitive == primitive::TRIANGLE && item.material.wireframe() {
                match item.wire_span {
                    Some((i, n)) => self.transients[slot].get(i).map(|b| (b.id(), n)),
                    None if item.range == (0, item.geometry.draw_count() as u32) => {
                        geom.wire.as_ref().map(|(b, n)| (b.id(), *n))
                    }
                    None => None,
                }
            } else {
                None
            };
            let draw_primitive = if wire.is_some() {
                primitive::LINE
            } else {
                item.primitive
            };

            let material = &item.material;
            let point_sprites = item.primitive == primitive::POINT;
            let layered_draw = views_per_draw > 1;
            let wide = item.primitive == primitive::LINE
                && self.line_width > 0.0
                && !layered_draw
                && slim_lines_enabled();
            // Both halves have to be true: a mesh with a skeleton whose
            // geometry carries no weights has nothing to blend, and a geometry
            // with weights and no skeleton has nothing to blend *through*.
            // Read before `self.pipeline`, which borrows `self` mutably.
            let is_skinned = item.bone_slot.is_some() && skin_buffer.is_some() && !layered_draw;
            let fur_draw = item.fur.is_some() && !layered_draw && !point_sprites && item.primitive == primitive::TRIANGLE;
            // An `fx` surface draws in screen space from its own uniforms; a
            // layered (stereo) pass has no single viewport for it, so it is
            // skipped there.
            if item.fx_surface.is_some() && (layered_draw || item.primitive != primitive::TRIANGLE) {
                continue;
            }
            let fx_draw = item.fx_surface.is_some();
            // A custom shader is a triangle-mesh, single-view draw, like the
            // fx surface; anything else, or one that did not compile, draws
            // the fallback colour through the ordinary pipeline.
            let custom_draw = if item.custom
                && !layered_draw
                && !point_sprites
                && item.primitive == primitive::TRIANGLE
                && self.oit_phase == 0
            {
                let material = item.material.clone();
                self.custom_shader(&material)
            } else {
                0
            };
            let pipeline = match self.pipeline(PipelineKey {
                point_sprites,
                fur: fur_draw,
                fx_surface: fx_draw,
                custom: custom_draw,
                wide_lines: wide,
                // Must agree with what `ensure_geometry` uploaded, or the
                // vertex shader reads the wrong stride and the draw is garbage.
                slim_lines: item.primitive == primitive::LINE
                    && !layered_draw
                    && slim_lines_enabled(),
                blend: item.transparent,
                additive: item.state.blending == crate::materials::Blending::Additive,
                layered: layered_draw,
                // Both halves have to be true: a mesh with a skeleton whose
                // geometry carries no weights has nothing to blend, and a
                // geometry with weights and no skeleton has nothing to blend
                // *through*.
                skinned: is_skinned,
                // Only the layered path needs this, and naming it there costs a
                // pipeline per topology — so leave it unspecified elsewhere.
                topology: if layered_draw {
                    topology_class::of(draw_primitive)
                } else {
                    topology_class::UNSPECIFIED
                },
                color_format: pass.color_format,
                depth_format: pass.depth_format,
                sample_count: pass.sample_count.max(1),
                color_write: item.state.color_write,
                oit_accum: self.oit_phase == 1,
            }) {
                Ok(pipeline) => pipeline,
                Err(e) => {
                    failure = Some(e);
                    break;
                }
            };
            if pipeline != last_pipeline {
                let _: () = unsafe { msg1(encoder, sel!("setRenderPipelineState:"), pipeline) };
                last_pipeline = pipeline;
            }

            // The material's `depthTest` / `depthWrite` / `depthFunc`. By
            // default transparent surfaces test depth but do not write it, so
            // two blended layers both survive rather than the nearer one
            // erasing the further one's contribution; with the test off
            // nothing is written, as in WebGL. A pass with no depth attachment
            // gets no depth state at all: Metal rejects one that would write.
            if !pass.depth.is_null() {
                let mut func = item.state.effective_func();
                if pass.reverse_z {
                    func = func.reversed();
                }
                // The accumulation reads the opaque depth and writes none.
                let write = item.state.effective_write(!item.transparent) && self.oit_phase == 0;
                let depth_state = match self.depth_state(metal_compare(func), write) {
                    Ok(s) => s,
                    Err(e) => {
                        failure = Some(e);
                        break;
                    }
                };
                if depth_state != last_depth {
                    let _: () = unsafe { msg1(encoder, sel!("setDepthStencilState:"), depth_state) };
                    last_depth = depth_state;
                }
                // three.js `polygonOffset`: Metal's depth bias is
                // `glPolygonOffset`'s formula — `slopeScale · m + bias · r` —
                // so factor and units carry over, negated under reverse-Z
                // where nearer is larger.
                let sign = if pass.reverse_z { -1.0 } else { 1.0 };
                let bias = material
                    .polygon_offset()
                    .map(|o| (o.units * sign, o.factor * sign))
                    .unwrap_or((0.0, 0.0));
                if bias != last_bias {
                    let _: () = unsafe { msg3(
                        encoder,
                        sel!("setDepthBias:slopeScale:clamp:"),
                        bias.0,
                        bias.1,
                        0.0f32,
                    ) };
                    last_bias = bias;
                }
            }

            let cull = match item.side_override.unwrap_or(material.side()) {
                1 => cull::FRONT,
                2 => cull::NONE,
                _ => cull::BACK,
            };
            if cull != last_cull {
                let _: () = unsafe { msg1(encoder, sel!("setCullMode:"), cull) };
                last_cull = cull;
            }
            let fill = if material.wireframe() && wire.is_none() {
                fill_mode::LINES
            } else {
                fill_mode::FILL
            };
            if fill != last_fill {
                let _: () = unsafe { msg1(encoder, sel!("setTriangleFillMode:"), fill) };
                last_fill = fill;
            }

            let _: () = unsafe { msg3(
                encoder,
                sel!("setVertexBuffer:offset:atIndex:"),
                vertices,
                0usize,
                VB_VERTICES,
            ) };

            // Instanced draws read their matrices from this frame's transient
            // buffer; everything else still binds the 1x1 identity, because the
            // shader declares the buffer and Metal requires bound what is
            // declared.
            let instanced = item
                .instance_slot
                .and_then(|i| self.transients[slot].get(i))
                .map(|b| b.id());
            let instance_buffer = instanced.unwrap_or(self.identity_instance.id());
            let _: () = unsafe { msg3(
                encoder,
                sel!("setVertexBuffer:offset:atIndex:"),
                instance_buffer,
                0usize,
                VB_INSTANCES,
            ) };
            // Metal requires every resource a function declares to be bound,
            // and `vs_skinned` declares both — so bind them whenever they
            // exist, and fall back to the vertex buffer itself when they do
            // not. The non-skinned vertex functions never read either index.
            let skin_buffer = skin_buffer.unwrap_or(vertices);
            let _: () = unsafe { msg3(
                encoder,
                sel!("setVertexBuffer:offset:atIndex:"),
                skin_buffer,
                0usize,
                VB_SKIN,
            ) };
            let bone_buffer = item
                .bone_slot
                .and_then(|i| self.transients[slot].get(i))
                .map(|b| b.id())
                .unwrap_or(self.identity_instance.id());
            let _: () = unsafe { msg3(
                encoder,
                sel!("setVertexBuffer:offset:atIndex:"),
                bone_buffer,
                0usize,
                VB_BONES,
            ) };
            // Morph targets: the geometry's buffer, or a stand-in with count
            // 0 — the mesh functions declare both.
            let _: () = unsafe { msg3(
                encoder,
                sel!("setVertexBuffer:offset:atIndex:"),
                morph_buffer,
                0usize,
                VB_MORPH_DATA,
            ) };
            unsafe { set_bytes(encoder, true, &morph_u, VB_MORPH) };
            // The extra uv sets, or a stand-in the shader never reads
            // (`phys_flags.z` is 0 without them).
            let _: () = unsafe { msg3(
                encoder,
                sel!("setVertexBuffer:offset:atIndex:"),
                uv_sets.map_or(self.identity_instance.id(), |(b, _)| b),
                0usize,
                VB_UV_SETS,
            ) };
            if let (true, Some(fur)) = (fur_draw, &item.fur) {
                unsafe { set_bytes(encoder, true, fur, VB_FUR) };
                unsafe { set_bytes(encoder, false, fur, FB_FUR) };
            }
            // The surface reads its block at the fur's slots: the two never
            // share a draw.
            if let Some(fx) = &item.fx_surface {
                unsafe { set_bytes(encoder, true, &**fx, VB_FUR) };
                unsafe { set_bytes(encoder, false, &**fx, FB_FUR) };
            }
            // A custom shader's `data` at the fur slot, its storage arrays at
            // 3–5 and their lengths at 6 (`SharedShader::msl`'s `fs_custom`).
            if custom_draw != 0
                && let Material::Shader(m) = &*item.material {
                    let mut data = [[0.0f32; 4]; 16];
                    for (slot, v) in data.iter_mut().zip(&m.data) {
                        *slot = *v;
                    }
                    unsafe { set_bytes(encoder, false, &data, FB_FUR) };
                    // The vertex stage sees the same data (a custom vertex
                    // stage may read it), and the instances' custom columns.
                    unsafe { set_bytes(encoder, true, &data, 11) };
                    let custom_buf = match &item.instance_custom {
                        Some(c) if !c.is_empty() => match self.device.new_buffer(bytemuck::cast_slice(c)) {
                            Ok(b) => {
                                let id = b.id();
                                self.transients[slot].push(b);
                                id
                            }
                            Err(_) => self.identity_instance.id(),
                        },
                        _ => self.identity_instance.id(),
                    };
                    let _: () = unsafe { msg3(encoder, sel!("setVertexBuffer:offset:atIndex:"), custom_buf, 0usize, 10usize) };
                    let mut lens = [0u32; 4];
                    for (k, values) in [&m.storage0, &m.storage1, &m.storage2].into_iter().enumerate() {
                        lens[k] = values.len() as u32;
                        let buffer = if values.is_empty() {
                            self.identity_instance.id()
                        } else {
                            match self.device.new_buffer(bytemuck::cast_slice(values)) {
                                Ok(b) => {
                                    let id = b.id();
                                    self.transients[slot].push(b);
                                    id
                                }
                                Err(_) => {
                                    lens[k] = 0;
                                    self.identity_instance.id()
                                }
                            }
                        };
                        let _: () = unsafe { msg3(
                            encoder,
                            sel!("setFragmentBuffer:offset:atIndex:"),
                            buffer,
                            0usize,
                            3 + k as NSUInteger,
                        ) };
                        let _: () = unsafe { msg3(
                            encoder,
                            sel!("setVertexBuffer:offset:atIndex:"),
                            buffer,
                            0usize,
                            12 + k as NSUInteger,
                        ) };
                    }
                    unsafe { set_bytes(encoder, false, &lens, 6) };
                    unsafe { set_bytes(encoder, true, &lens, 15) };
                }

            let (texture, sampler) = self.material_texture(material);
            let _: () = unsafe { msg2(
                encoder,
                sel!("setFragmentTexture:atIndex:"),
                texture,
                0usize,
            ) };
            // three.js gives every texture its own filter and wrap: each map
            // takes its sampler's place in the pool, the base map's first.
            let mut pool = SamplerPool::new(sampler);
            let (extra, present) = self.material_maps(material);
            for (i, slot) in extra.iter().enumerate() {
                let _: () = unsafe { msg2(
                    encoder,
                    sel!("setFragmentTexture:atIndex:"),
                    slot.map_or(self.white.id(), |(t, _)| t),
                    i + 1,
                ) };
                // Emissive, normal, metallic-roughness, ambient occlusion.
                pool.set(1 + i, slot.map_or(sampler, |(_, s)| s));
            }
            // A Standard/Physical material's own `envMap` in place of the
            // scene's, with its own intensity and rotation.
            let own_env = material.pbr_env().and_then(|(c, intensity, rotation)| {
                let c = self.prefiltered_cached(c);
                self.material_envs
                    .get(&crate::renderer::gpu_texture::cube_cache_key(&c))
                    .map(|e| {
                        (e, intensity, crate::materials::environment_lookup_rows(rotation, e.atlas.is_none()))
                    })
            });
            let pbr_env = own_env.map(|e| e.0).or(self.env_cube.as_ref().map(|(_, e)| e));
            let env = pbr_env.map(|e| e.cube.id());
            // Its PMREM atlas, which `ibl` reads (texture 15, sampler 13).
            let _: () = unsafe { msg2(
                encoder,
                sel!("setFragmentTexture:atIndex:"),
                pbr_env.and_then(|e| e.atlas.as_ref()).map_or(self.white.id(), |a| a.id()),
                15usize,
            ) };
            let _: () = unsafe { msg2(encoder, sel!("setFragmentTexture:atIndex:"), self.ltc.id(), 28usize) };
            // The transmission capture (three.js `transmissionSamplerMap`).
            let _: () = unsafe { msg2(
                encoder,
                sel!("setFragmentTexture:atIndex:"),
                self.capture_bound.unwrap_or(self.capture_dummy.id()),
                29usize,
            ) };
            let _: () = unsafe { msg2(encoder, sel!("setFragmentSamplerState:atIndex:"), self.transmission_sampler.id(), 14usize) };
            // The capture's depth, a screen-space `ShaderMaterial`'s
            // `ss_depth_tex`: single-slice captures only (a stereo capture's
            // depth array is not readable), a stand-in otherwise.
            let capture_depth = self
                .transmission
                .as_ref()
                .filter(|c| c.slices == 1 && self.capture_bound.is_some())
                .map_or(self.capture_depth_dummy.id(), |c| c.target.depth_texture());
            let _: () = unsafe { msg2(encoder, sel!("setFragmentTexture:atIndex:"), capture_depth, 30usize) };
            let atlas_sampler = self
                .samplers
                .get(&SamplerKey::default_linear())
                .map_or(sampler, |s| s.id());
            let _: () = unsafe { msg2(encoder, sel!("setFragmentSamplerState:atIndex:"), atlas_sampler, 13usize) };
            let env_atlas_params = pbr_env.map_or([0.0; 4], |e| e.params);
            let _: () = unsafe { msg2(
                encoder,
                sel!("setFragmentTexture:atIndex:"),
                env.unwrap_or(self.black_cube.id()),
                5usize,
            ) };
            // A mipmapped sampler for the cube: the environment is sampled by
            // roughness across its own chain, and a sampler that cannot see
            // past level 0 makes every surface a mirror.
            let env_sampler = self
                .samplers
                .get(&SamplerKey::environment())
                .map(|s| s.id())
                .unwrap_or(sampler);
            let _: () = unsafe { msg2(
                encoder,
                sel!("setFragmentSamplerState:atIndex:"),
                env_sampler,
                10usize,
            ) };
            // The shadow map and its comparison sampler. Bound even when
            // nothing casts, because Metal requires every declared resource to
            // be bound and `shadow_factor` returns early on the flag anyway.
            let shadow_tex = self
                .shadow_map
                .as_ref()
                .map(|t| t.id())
                .unwrap_or(self.empty_shadow.id());
            let _: () = unsafe { msg2(
                encoder,
                sel!("setFragmentTexture:atIndex:"),
                shadow_tex,
                6usize,
            ) };
            let shadow_samp = self
                .shadow_sampler
                .as_ref()
                .map(|s| s.id())
                .unwrap_or(env_sampler);
            let _: () = unsafe { msg2(
                encoder,
                sel!("setFragmentSamplerState:atIndex:"),
                shadow_samp,
                11usize,
            ) };
            let spot_tex = self
                .spot_shadow_map
                .as_ref()
                .map(|t| t.id())
                .unwrap_or(self.empty_shadow.id());
            let _: () = unsafe { msg2(encoder, sel!("setFragmentTexture:atIndex:"), spot_tex, 7usize) };
            let point_tex = self
                .point_shadow_map
                .as_ref()
                .map(|t| t.id())
                .unwrap_or(self.empty_shadow.id());
            let _: () = unsafe { msg2(encoder, sel!("setFragmentTexture:atIndex:"), point_tex, 8usize) };
            // The variant-only maps: alpha, light, bump, specular at textures
            // 9–12, their samplers in the pool.
            let slots = material.texture_slots();
            // The light map is colour, the rest data.
            let extras = [
                (&slots.alpha_map, true),
                (&slots.light_map, false),
                (&slots.bump_map, true),
                (&slots.specular_map, true),
            ]
            .map(|(m, linear)| self.slot_texture(m.as_ref(), linear));
            for (i, slot) in extras.iter().enumerate() {
                let _: () = unsafe { msg2(
                    encoder,
                    sel!("setFragmentTexture:atIndex:"),
                    slot.map_or(self.white.id(), |(t, _)| t),
                    9 + i,
                ) };
                pool.set(5 + i, slot.map_or(sampler, |(_, s)| s));
            }
            // MeshPhysicalMaterial's maps at textures 16–27.
            let mut phys_bits = 0u32;
            for (i, (map, linear)) in physical_maps(&slots).into_iter().enumerate() {
                let slot = self.slot_texture(map, linear);
                let _: () = unsafe { msg2(
                    encoder,
                    sel!("setFragmentTexture:atIndex:"),
                    slot.map_or(self.white.id(), |(t, _)| t),
                    PHYS_TEXTURE0 + i,
                ) };
                if let Some((_, smp)) = slot {
                    pool.set(MAP_PHYS0 + i, smp);
                    phys_bits |= 1 << i;
                }
            }
            // A custom shader's `u_tex*` over them: it reads no physical map.
            if custom_draw != 0 {
                if let Material::Shader(m) = &*item.material {
                    // A view made on this device first, then a map, as on
                    // wgpu; white where the material has neither — the
                    // fragment slots would otherwise still hold the physical
                    // maps bound above.
                    for i in 0..4 {
                        let tex = match m.textures.get(i).and_then(|v| v.as_metal()) {
                            Some(t) => t.obj.id(),
                            None => m.maps.get(i).and_then(|map| self.slot_texture(Some(map), false)).map_or(self.white.id(), |(t, _)| t),
                        };
                        let _: () = unsafe { msg2(encoder, sel!("setFragmentTexture:atIndex:"), tex, USER_TEXTURE0 + i) };
                        let _: () = unsafe { msg2(encoder, sel!("setVertexTexture:atIndex:"), tex, USER_TEXTURE0 + i) };
                    }
                }
                let smp = self.samplers.get(&USER_SAMPLER).map_or(sampler, |s| s.id());
                let _: () = unsafe { msg2(encoder, sel!("setFragmentSamplerState:atIndex:"), smp, USER_SAMPLER_SLOT) };
                let _: () = unsafe { msg2(encoder, sel!("setVertexSamplerState:atIndex:"), smp, USER_SAMPLER_SLOT) };
                // Volumes made on this device (a wgpu view has no texture
                // here, and reads zero, as an empty slot does).
                let volumes = match &*item.material {
                    Material::Shader(m) => m.volumes.as_slice(),
                    _ => &[],
                };
                for i in 0..2 {
                    let tex = volumes.get(i).and_then(|v| v.as_metal()).map_or(self.zero_volume.id(), |t| t.obj.id());
                    let _: () = unsafe { msg2(encoder, sel!("setFragmentTexture:atIndex:"), tex, USER_VOLUME0 + i) };
                    let _: () = unsafe { msg2(encoder, sel!("setVertexTexture:atIndex:"), tex, USER_VOLUME0 + i) };
                }
                let smp = self.samplers.get(&USER_VOLUME_SAMPLER).map_or(sampler, |s| s.id());
                let _: () = unsafe { msg2(encoder, sel!("setFragmentSamplerState:atIndex:"), smp, USER_VOLUME_SAMPLER_SLOT) };
                let _: () = unsafe { msg2(encoder, sel!("setVertexSamplerState:atIndex:"), smp, USER_VOLUME_SAMPLER_SLOT) };
            }

            let mut draw_u = draw_uniforms(
                item,
                texture != self.white.id(),
                instanced.is_some(),
                &item.geometry,
            );
            draw_u.maps = present;
            draw_u.extra = extras.map(|e| e.is_some() as u32);
            // The material's own `envMap`, mipmapped like the scene's.
            let mat_env = slots
                .env_map
                .as_ref()
                .and_then(|c| self.material_envs.get(&crate::renderer::gpu_texture::cube_cache_key(c)))
                .map(|e| e.cube.id());
            draw_u.map_scalars[2] = mat_env.is_some() as u32 as f32;
            // A matcap's texture, sRGB like a colour map — or, on a toon
            // material, its `gradientMap`, decoded by its own colour space as
            // three.js does. No material has both.
            // The displacement map, at vertex texture 0 — bound for every draw,
            // white when there is none, because `vs_mesh` declares it.
            let disp = self.slot_texture(slots.displacement_map.as_ref(), true);
            let _: () = unsafe { msg2(encoder, sel!("setVertexTexture:atIndex:"), disp.map_or(self.white.id(), |(t, _)| t), 0usize) };
            let _: () = unsafe { msg2(encoder, sel!("setVertexSamplerState:atIndex:"), disp.map_or(sampler, |(_, s)| s), 0usize) };
            let matcap = self.slot_texture(
                slots.matcap_map.as_ref().or(slots.gradient_map.as_ref()).or(slots.cloud_shadow_map.as_ref()),
                false,
            );
            draw_u.map_scalars[3] = matcap.is_some() as u32 as f32;
            let _: () = unsafe { msg2(
                encoder,
                sel!("setFragmentTexture:atIndex:"),
                matcap.map_or(self.white.id(), |(t, _)| t),
                14usize,
            ) };
            pool.set(9, matcap.map_or(sampler, |(_, s)| s));
            let _: () = unsafe { msg2(
                encoder,
                sel!("setFragmentTexture:atIndex:"),
                mat_env.unwrap_or(self.black_cube.id()),
                13usize,
            ) };
            let mat_env_sampler = slots
                .env_map
                .as_ref()
                .and_then(|c| self.samplers.get(&SamplerKey::from_cube(c)))
                .map_or(env_sampler, |s| s.id());
            let _: () = unsafe { msg2(encoder, sel!("setFragmentSamplerState:atIndex:"), mat_env_sampler, 12usize) };
            for (i, smp) in pool.slots().into_iter().enumerate() {
                let _: () = unsafe { msg2(encoder, sel!("setFragmentSamplerState:atIndex:"), smp, i) };
            }
            draw_u.map_samplers = pool.packed();
            draw_u.phys_flags = [
                phys_bits,
                self.specular_antialiasing as u32,
                uv_sets.map_or(0, |(_, f)| f),
                item.blended_glass as u32,
            ];
            // three.js `envMapIntensity` with the material's own environment,
            // `scene.environmentIntensity` with the scene's.
            let (intensity, rows) = own_env.map_or(self.env_light, |e| (e.1, e.2));
            draw_u.map_params[2] = intensity;
            draw_u.map_params[3] = if env.is_some() { 1.0 } else { 0.0 };
            // A custom shader samples the scene's environment too
            // (`sample_env_cube`).
            if matches!(material.kind(), MaterialKind::Standard | MaterialKind::Physical) || custom_draw != 0 {
                draw_u.env_dir = rows;
                draw_u.env_params = env_atlas_params;
            }
            if wide {
                // misc.y is the point size, which a line never uses — the quad
                // path reads its width in pixels from the same slot.
                draw_u.misc[1] = self.line_width;
                // misc.x, otherwise unused: whether the quad blends, and so
                // can draw coverage rather than cut at half.
                draw_u.misc[0] = if item.transparent { 1.0 } else { 0.0 };
            }
            unsafe { set_bytes(encoder, true, &draw_u, VB_DRAW) };
            unsafe { set_bytes(encoder, false, &draw_u, FB_DRAW) };

            // Without a matrix buffer there is exactly one instance to draw,
            // whatever the object claimed — the alternative is the shader
            // indexing past the identity matrix.
            let instances = if instanced.is_some() {
                item.instance_count.max(1)
            } else {
                1
            };
            // A layered pass draws every instance once per view; the vertex
            // stage divides the id back out.
            let instances = instances * views_per_draw as usize;
            // The item's span: three.js's draw range cut to its group. The
            // wireframe line list replaces the triangle indices for the draw
            // only (it was built for the span, or is the whole list for a
            // whole span); the triangle count below reports the span.
            let (start, end) = (item.range.0 as usize, item.range.1 as usize);
            let (draw_indices, draw_first, draw_index_count) = match wire {
                Some((buf, n)) => (Some(buf), 0usize, n),
                None => (indices, start, end.min(index_count).saturating_sub(start)),
            };
            match draw_indices {
                Some(ibuf) if index_count > 0 || wire.is_some() => {
                    if draw_index_count > 0 {
                        let _: () = unsafe { msg6(
                            encoder,
                            sel!(
                                "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:"
                            ),
                            draw_primitive,
                            draw_index_count as NSUInteger,
                            index_type::UINT32,
                            ibuf,
                            draw_first * std::mem::size_of::<u32>(),
                            instances as NSUInteger,
                        ) };
                    }
                }
                _ => {
                    let end = end.min(vertex_count);
                    // Six vertices per segment instead of two, as triangles —
                    // the vertex buffer is unchanged, only how it is read.
                    let (prim, first, count) = if wide {
                        (primitive::TRIANGLE, (start / 2) * 6, (end.saturating_sub(start) / 2) * 6)
                    } else {
                        (item.primitive, start, end.saturating_sub(start))
                    };
                    if count > 0 {
                        let _: () = unsafe { msg4(
                            encoder,
                            sel!("drawPrimitives:vertexStart:vertexCount:instanceCount:"),
                            prim,
                            first,
                            count as NSUInteger,
                            instances as NSUInteger,
                        ) };
                    }
                }
            }

            stats.draw_calls += 1;
            if item.primitive == primitive::TRIANGLE {
                let verts = end.min(if index_count > 0 { index_count } else { vertex_count })
                    .saturating_sub(start);
                stats.triangles += (verts / 3 * instances) as u32;
            }
        }

        let _: () = unsafe { msg0(encoder, sel!("endEncoding")) };
        match failure {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Build the `MTLRenderPassDescriptor` for one view. Autoreleased.
    ///
    /// `clear` is false for a second view sharing a target with the first —
    /// side-by-side eyes in one texture, where clearing again would erase the
    /// eye already drawn.
    pub(super) unsafe fn render_pass_descriptor(
        &self,
        pass: &PassAttachments,
        scene: &Scene,
        view: &RenderView,
        clear: bool,
        keep_multisampled: bool,
        array_length: u32,
    ) -> Id {
        let rpd: Id = unsafe { msg0(
            class!("MTLRenderPassDescriptor"),
            sel!("renderPassDescriptor"),
        ) };
        // A layered pass renders into every slice at once, so it names how many
        // there are and starts at slice 0 — the shader's
        // `render_target_array_index` picks between them. Without this the
        // attachment is one slice deep and both eyes land on top of each other.
        if array_length > 1 {
            let _: () = unsafe { msg1(
                rpd,
                sel!("setRenderTargetArrayLength:"),
                array_length as NSUInteger,
            ) };
        }
        let slice = if array_length > 1 { 0 } else { view.slice };
        let color_attachments: Id = unsafe { msg0(rpd, sel!("colorAttachments")) };
        let color: Id = unsafe { msg1(color_attachments, sel!("objectAtIndexedSubscript:"), 0usize) };
        let color_texture = if view.color.is_null() {
            pass.color
        } else {
            view.color
        };
        let _: () = unsafe { msg1(color, sel!("setTexture:"), color_texture) };
        let _: () = unsafe { msg1(color, sel!("setSlice:"), slice as NSUInteger) };
        let _: () = unsafe { msg1(
            color,
            sel!("setLoadAction:"),
            if clear {
                load_action::CLEAR
            } else {
                load_action::LOAD
            },
        ) };
        let _: () = unsafe { msg1(
            color,
            sel!("setClearColor:"),
            clear_color(scene.background, scene.background_alpha),
        ) };
        if pass.resolve.is_null() {
            let _: () = unsafe { msg1(color, sel!("setStoreAction:"), store_action::STORE) };
        } else {
            let _: () = unsafe { msg1(color, sel!("setResolveTexture:"), pass.resolve) };
            let _: () = unsafe { msg1(color, sel!("setResolveSlice:"), slice as NSUInteger) };
            let _: () = unsafe { msg1(
                color,
                sel!("setStoreAction:"),
                if keep_multisampled || self.keep_attachments {
                    store_action::STORE_AND_MULTISAMPLE_RESOLVE
                } else {
                    store_action::MULTISAMPLE_RESOLVE
                },
            ) };
        }

        let depth_texture = if view.depth.is_null() {
            pass.depth
        } else {
            view.depth
        };
        if !depth_texture.is_null() {
            let depth: Id = unsafe { msg0(rpd, sel!("depthAttachment")) };
            let _: () = unsafe { msg1(depth, sel!("setTexture:"), depth_texture) };
            let _: () = unsafe { msg1(depth, sel!("setSlice:"), slice as NSUInteger) };
            // Always cleared, even for a view that loads its colour: depth is
            // usually not stored (below), so loading it reads undefined values
            // across the second view's whole viewport. Views sharing a target
            // do not overlap — or, if they do, the later one is an overlay —
            // so nothing needs the earlier view's depth.
            let _: () = unsafe { msg1(depth, sel!("setLoadAction:"), load_action::CLEAR) };
            // Normally discarded: depth is scratch for the pass that draws it,
            // and storing it costs bandwidth nothing reads. SSAO does read it
            // afterwards, and a discarded depth texture samples as the clear
            // value — which sends every pixel down the far-plane early-out and
            // produces occlusion that is uniformly, silently absent.
            // Under MSAA, stored depth is also resolved where the target
            // asks (`PassAttachments::with_depth_resolve`) — the multisampled
            // one kept as well while the OIT pass still has to test against
            // it.
            let store_depth = self.store_depth || self.keep_attachments;
            let resolve_depth =
                store_depth && view.depth.is_null() && !pass.depth_resolve.is_null() && pass.sample_count > 1;
            if resolve_depth {
                let _: () = unsafe { msg1(depth, sel!("setResolveTexture:"), pass.depth_resolve) };
                let _: () = unsafe { msg1(depth, sel!("setResolveSlice:"), slice as NSUInteger) };
            }
            let _: () = unsafe { msg1(
                depth,
                sel!("setStoreAction:"),
                match (store_depth, resolve_depth, self.oit_active || self.keep_attachments) {
                    (false, ..) => store_action::DONT_CARE,
                    (true, false, _) => store_action::STORE,
                    (true, true, false) => store_action::MULTISAMPLE_RESOLVE,
                    (true, true, true) => store_action::STORE_AND_MULTISAMPLE_RESOLVE,
                },
            ) };
            // Reverse-Z clears to the far plane, which is 0.
            let _: () = unsafe { msg1(
                depth,
                sel!("setClearDepth:"),
                if pass.reverse_z { 0.0f64 } else { 1.0f64 },
            ) };
        }
        rpd
    }
}
