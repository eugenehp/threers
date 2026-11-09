use super::*;

impl MetalRenderer {
    /// Drop cache entries no frame has touched in [`cache_retention`](Self::set_cache_retention)
    /// frames.
    pub(super) fn evict_unused(&mut self) {
        if self.cache_retention == u64::MAX {
            return;
        }
        let cutoff = self.frame.saturating_sub(self.cache_retention);
        self.geometries.retain(|_, g| g.last_used >= cutoff);
        self.textures.retain(|_, t| t.last_used >= cutoff);
    }

    pub(super) fn collect(
        &self,
        scene: &Scene,
        views: &[RenderView],
        pass: &PassAttachments,
    ) -> (FrameUniforms, Vec<DrawItem>, MetalRenderStats, Casters) {
        // View 0 is the one the traversal uses: its layer mask decides what is
        // drawn and its position orders the transparent items. Two eyes a few
        // centimetres apart do not disagree about either.
        let primary = &views[0];
        let eye = primary.position;
        // A perspective projection's last column is (0, 0, -1, 0) — the test
        // the raycaster and the wgpu sprite shader use too.
        let primary_perspective = primary.projection_matrix.elements[15] == 0.0;
        let mut view_proj = [Matrix4::identity().elements; MAX_VIEWS];
        let mut view_matrices = [Matrix4::identity().elements; MAX_VIEWS];
        let mut camera_pos = [[0.0f32; 4]; MAX_VIEWS];
        for (i, v) in views.iter().enumerate() {
            view_proj[i] = v.projection_matrix.multiply(&v.view_matrix).elements;
            view_matrices[i] = v.view_matrix.elements;
            // w: 1 with reversed depth, as the wgpu frame has it — what a
            // translated `ShaderMaterial` reading `ss_depth_tex` tests.
            camera_pos[i] = [v.position.x, v.position.y, v.position.z, if pass.reverse_z { 1.0 } else { 0.0 }];
        }
        let mut frame_u = FrameUniforms {
            view_proj,
            view: view_matrices,
            camera_pos,
            // y: bit 0, the target encodes sRGB on write, which data outputs
            // (the normal material) undo so the stored byte is the value;
            // bit 1, it stands in for a `Unorm` one (`set_encoded_output`).
            // z, w: the tone mapping mode and its exposure's bits.
            view_info: [
                views.len() as u32,
                {
                    let srgb = pixel_format::is_srgb(pass.color_format);
                    srgb as u32 | ((srgb && self.encoded_output) as u32) << 1
                },
                self.tone_mapping.0.shader_mode() as u32,
                self.tone_mapping.1.to_bits(),
            ],
            viewport: [
                pass.width as f32,
                pass.height as f32,
                1.0 / pass.width.max(1) as f32,
                1.0 / pass.height.max(1) as f32,
            ],
            fog_color: [
                scene.fog.color.r,
                scene.fog.color.g,
                scene.fog.color.b,
                scene.fog.mode as f32,
            ],
            fog_params: [scene.fog.near, scene.fog.far, scene.fog.density, 0.0],
            ..Default::default()
        };
        frame_u.rect_counts[2] = self.reflective_transparency as u32;

        let mut items = Vec::new();
        let mut stats = MetalRenderStats::default();
        let (mut n_dir, mut n_point, mut n_spot, mut n_hemi) = (0usize, 0, 0, 0);
        // Every light that casts, each owning a layer of its kind's maps.
        let mut casters = Casters::new();
        // Union every view's layer mask so split-screen / multi-viewport passes
        // collect both sides; each pass filters in [`encode_pass`].
        let cam_layers = views.iter().fold(Layers::with_mask(0), |mut acc, v| {
            acc.mask |= v.layers.mask;
            acc
        });
        let cam_layers = if cam_layers.mask == 0 {
            primary.layers
        } else {
            cam_layers
        };

        scene.arena.traverse_visible(scene.root, &mut |_id, obj| {
            if !cam_layers.test(&obj.layers) {
                return;
            }
            let world = obj.matrix_world;
            let obj_layers = obj.layers;
            // The instance colours for the next item pushed (instanced meshes
            // only), taken by `push_skinned`.
            let pending_custom: std::cell::RefCell<Option<Vec<f32>>> = std::cell::RefCell::new(None);
            let pending_colors: std::cell::RefCell<Option<Vec<f32>>> =
                std::cell::RefCell::new(None);
            #[allow(clippy::too_many_arguments)]
            let mut push_skinned = |geometry: &Arc<BufferGeometry>,
                            material: &Arc<Material>,
                            primitive: NSUInteger,
                            instances: Option<Vec<f32>>,
                            bones: Option<Vec<f32>>,
                            sprite: Option<[f32; 4]>,
                            range: Option<(u32, u32)>,
                            influences: &[f32],
                            instance_morph: Option<(Vec<f32>, usize)>| {
                if !geometry.attributes.contains_key("position") {
                    stats.skipped += 1;
                    return;
                }
                // An empty span draws nothing, as three.js's renderBufferDirect.
                let Some(range) = range else {
                    return;
                };
                let center =
                    Vector3::new(world.elements[12], world.elements[13], world.elements[14]);
                // three.js sorts on `transparent` alone: an opaque
                // `depthTest: false` material draws in the opaque list, in
                // `render_order`, like any other.
                let transparent = is_transparent(material);
                // A transmissive physical material's glass mode.
                let glass_mode = match &**material {
                    Material::Physical(m) if m.transmission > 0.0 => Some(m.transparency),
                    _ => None,
                };
                items.push(DrawItem {
                    fur: match &**material {
                        Material::Shader(m) if crate::fur::is_fur(m) => {
                            let mut d = [0.0f32; 64];
                            for (slot, v) in m.data.iter().take(16).enumerate() {
                                d[slot * 4..slot * 4 + 4].copy_from_slice(v);
                            }
                            Some(d)
                        }
                        _ => None,
                    },
                    custom: matches!(&**material, Material::Shader(m)
                        if !crate::fur::is_fur(m) && !crate::fx::surface::is_surface(m)),
                    fx_surface: match &**material {
                        Material::Shader(m) if crate::fx::surface::is_surface(m) => {
                            Some(crate::fx::surface::metal_bytes(m))
                        }
                        _ => None,
                    },
                    range,
                    morph_influences: influences.to_vec(),
                    wire_span: None,
                    bones,
                    bone_slot: None,
                    // three.js's shadow pass renders meshes, lines and points;
                    // a sprite casts nothing.
                    cast_shadow: obj.cast_shadow && sprite.is_none(),
                    sprite,
                    receive_shadow: obj.receive_shadow,
                    geometry: geometry.clone(),
                    material: material.clone(),
                    world,
                    layers: obj_layers,
                    primitive,
                    instance_count: instances.as_ref().map(|v| v.len() / 16).unwrap_or(1),
                    instances,
                    instance_slot: None,
                    instance_morph,
                    instance_morph_layout: None,
                    instance_colors: pending_colors.borrow_mut().take(),
                    instance_custom: pending_custom.borrow_mut().take(),
                    instance_color_offset: 0,
                    view_depth: center.distance_to(eye),
                    render_order: obj.render_order,
                    // `Refract` draws as three.js's transmission here: the same
                    // volume model over the capture, which is what that mode
                    // refines on the wgpu path.
                    transparent: transparent && glass_mode != Some(crate::materials::TransparencyMode::Refract),
                    transmissive: material.transmissive()
                        || glass_mode == Some(crate::materials::TransparencyMode::Refract),
                    // `Glass` and `Oit`: glass blended over what is behind,
                    // tinted by channel.
                    blended_glass: matches!(
                        glass_mode,
                        Some(crate::materials::TransparencyMode::Glass | crate::materials::TransparencyMode::Oit)
                    ),
                    // Order-independent: drawn in the accumulation pass rather
                    // than sorted with the transparent draws (`encode_oit`).
                    oit: transparent
                        && matches!(&**material, Material::Physical(m) if m.transparency == crate::materials::TransparencyMode::Oit),
                    screen_space: matches!(&**material, Material::Shader(m) if m.screen_space),
                    capture_skip: false,
                    side_override: None,
                    state: material.render_state(),
                });
            };
            // Everything that is not a skinned mesh, which is everything
            // else: same closure, no pose.
            // The whole draw range, one material: lines, points, instances.
            let mut push =
                |g: &Arc<BufferGeometry>,
                 m: &Arc<Material>,
                 p: NSUInteger,
                 inst: Option<Vec<f32>>| {
                    push_skinned(g, m, p, inst, None, None, g.draw_span(None), &[], None)
                };

            match &obj.kind {
                // One item per draw — per group with a material array — each
                // with its own material, transparency and sort, as three.js's
                // render list holds them.
                ObjectKind::Mesh(m) => {
                    // No positions: counted once, as any undrawable item is
                    // (its draw list is empty, so the loop would not see it).
                    if !m.geometry.attributes.contains_key("position") {
                        push_skinned(&m.geometry, &m.material, primitive::TRIANGLE, None, None, None, None, &[], None);
                        return;
                    }
                    for d in m.draws() {
                        push_skinned(
                            &m.geometry,
                            d.material,
                            primitive::TRIANGLE,
                            None,
                            None,
                            None,
                            Some((d.start, d.end)),
                            &m.morph_influences,
                            None,
                        );
                    }
                }
                ObjectKind::SkinnedMesh(sm) => {
                    // `bone_matrices` is world × inverse-bind per bone, already
                    // composed by `Skeleton::update`, so the vertex stage only
                    // has to blend them.
                    //
                    // Exactly `MAX_BONES` of them, padded with identities, as
                    // the wgpu backend uploads them: `vs_skinned` clamps joint
                    // indices to the last, so every index reads the matrix it
                    // reads there (and in the raycaster) — never past the end.
                    let identity = Matrix4::identity();
                    let bones: Vec<f32> = (0..crate::core::MAX_BONES)
                        .flat_map(|i| {
                            sm.skeleton
                                .bone_matrices
                                .get(i)
                                .unwrap_or(&identity)
                                .elements
                        })
                        .collect();
                    for d in sm.draws() {
                        push_skinned(
                            &sm.geometry,
                            d.material,
                            primitive::TRIANGLE,
                            None,
                            Some(bones.clone()),
                            None,
                            Some((d.start, d.end)),
                            &sm.morph_influences,
                            None,
                        );
                    }
                }
                ObjectKind::LineSegments(ls) => {
                    push(&ls.geometry, &ls.material, primitive::LINE, None)
                }
                ObjectKind::Points(p) => push(&p.geometry, &p.material, primitive::POINT, None),
                ObjectKind::Sprite(s) => {
                    if let Some(g) = sprite_geometry(s) {
                        let (rotation, size_attenuation) = match &*s.material {
                            Material::Sprite(m) => (m.rotation, m.size_attenuation),
                            _ => (0.0, true),
                        };
                        // The billboard faces view 0; a pass per view
                        // promotes its own view to slot 0, so each faces its
                        // own camera. Whether the size follows depth is read
                        // off the primary view's projection.
                        let by_depth = !size_attenuation && primary_perspective;
                        let params = [
                            s.center.x - 0.5,
                            s.center.y - 0.5,
                            rotation,
                            if by_depth { 1.0 } else { 0.0 },
                        ];
                        push_skinned(
                            &g,
                            &s.material,
                            primitive::TRIANGLE,
                            None,
                            None,
                            Some(params),
                            g.draw_span(None),
                            &[],
                            None,
                        );
                    }
                }
                ObjectKind::InstancedMesh(im) => {
                    if im.transforms.is_empty() {
                        return;
                    }
                    let mut data = Vec::with_capacity(im.transforms.len() * 16);
                    for m in &im.transforms {
                        data.extend_from_slice(&m.elements);
                    }
                    // Per-instance morph influences (`set_morph_at`), when the
                    // mesh has them for every instance.
                    let row_len = im.morph_row_len();
                    let rows = (!im.morph_texture.is_empty()
                        && im.morph_texture.len() == row_len * im.transforms.len())
                    .then(|| im.morph_texture.clone());
                    if !im.geometry.attributes.contains_key("position") {
                        push(&im.geometry, &im.material, primitive::TRIANGLE, Some(data));
                        return;
                    }
                    // three.js `instanceColor`: white past the end of `colors`.
                    let colors = (!im.colors.is_empty()).then(|| {
                        (0..im.transforms.len())
                            .flat_map(|i| {
                                let c = im.colors.get(i).copied().unwrap_or(crate::math::Color::WHITE);
                                [c.r, c.g, c.b, 1.0]
                            })
                            .collect::<Vec<f32>>()
                    });
                    // One instanced item per draw — per group with a material
                    // array — as three.js's render list holds them.
                    // The two custom columns, each padded to a float4.
                    let custom = (im.custom.iter().any(|c| !c.is_empty())).then(|| {
                        let n = im.transforms.len();
                        let mut out = vec![0.0f32; n * 8];
                        for (slot, values) in im.custom.iter().enumerate() {
                            let k = im.custom_components[slot].clamp(1, 4) as usize;
                            for i in 0..n {
                                for c in 0..k {
                                    out[i * 8 + slot * 4 + c] = values.get(i * k + c).copied().unwrap_or(0.0);
                                }
                            }
                        }
                        out
                    });
                    for d in im.draws() {
                        *pending_colors.borrow_mut() = colors.clone();
                        *pending_custom.borrow_mut() = custom.clone();
                        push_skinned(
                            &im.geometry,
                            d.material,
                            primitive::TRIANGLE,
                            Some(data.clone()),
                            None,
                            None,
                            Some((d.start, d.end)),
                            &im.morph_influences,
                            rows.clone().map(|r| (r, row_len)),
                        );
                    }
                }
                ObjectKind::Light(light) => match light {
                    Light::Ambient(l) => {
                        frame_u.ambient[0] += l.color.r * l.intensity;
                        frame_u.ambient[1] += l.color.g * l.intensity;
                        frame_u.ambient[2] += l.color.b * l.intensity;
                    }
                    Light::Directional(l) => {
                        if n_dir >= MAX_DIR_LIGHTS {
                            return;
                        }
                        // three.js: direction = normalize(target - position),
                        // default target the origin — matched to the wgpu path.
                        let pos = obj.world_position();
                        let d = Vector3::new(-pos.x, -pos.y, -pos.z);
                        let dir = if d.length_sq() > 1e-8 {
                            d.normalize()
                        } else {
                            transform_direction(&world, l.direction).normalize()
                        };
                        frame_u.dir[n_dir] = DirLightGpu {
                            direction: [dir.x, dir.y, dir.z, 0.0],
                            color: scaled(l.color, l.intensity),
                        };
                        // Every casting light gets a record and layers of
                        // its kind's map array, and names the record.
                        if l.cast_shadow && casters.len() < MAX_SHADOW_RECORDS {
                            casters.push(Caster::Dir { dir, settings: l.shadow });
                            frame_u.dir[n_dir].color[3] = casters.len() as f32;
                        }
                        n_dir += 1;
                    }
                    Light::Point(l) => {
                        if n_point >= MAX_POINT_LIGHTS {
                            return;
                        }
                        let p = obj.world_position();
                        frame_u.points[n_point] = PointLightGpu {
                            position: [p.x, p.y, p.z, 1.0],
                            color: scaled(l.color, l.intensity),
                            params: [l.distance, l.decay, 0.0, 0.0],
                        };
                        if l.cast_shadow && casters.len() < MAX_SHADOW_RECORDS {
                            // The cube's depth is normalised by the light's
                            // falloff `distance`, or — with none, as three.js's
                            // `PointLightShadow` does (`light.distance ||
                            // camera.far`) — by its shadow camera's far plane.
                            let radius = if l.distance > 0.0 { l.distance } else { l.shadow.camera_far };
                            casters.push(Caster::Point { pos: p, radius, settings: l.shadow });
                            frame_u.points[n_point].color[3] = casters.len() as f32;
                        }
                        n_point += 1;
                    }
                    Light::Spot(l) => {
                        if n_spot >= MAX_SPOT_LIGHTS {
                            return;
                        }
                        let p = obj.world_position();
                        let dir = transform_direction(&world, l.direction).normalize();
                        let casts = l.cast_shadow && casters.len() < MAX_SHADOW_RECORDS;
                        if casts {
                            casters.push(Caster::Spot { pos: p, dir, angle: l.angle, settings: l.shadow });
                        }
                        frame_u.spots[n_spot] = SpotLightGpu {
                            position: [p.x, p.y, p.z, 1.0],
                            direction: [dir.x, dir.y, dir.z, 0.0],
                            color: scaled(l.color, l.intensity),
                            params: [
                                l.distance,
                                l.decay,
                                l.angle.cos(),
                                (l.angle * (1.0 - l.penumbra)).cos(),
                            ],
                        };
                        if casts {
                            frame_u.spots[n_spot].color[3] = casters.len() as f32;
                        }
                        n_spot += 1;
                    }
                    Light::Hemisphere(l) => {
                        if n_hemi >= MAX_HEMI_LIGHTS {
                            return;
                        }
                        let up = transform_direction(&world, Vector3::UP).normalize();
                        frame_u.hemis[n_hemi] = HemiLightGpu {
                            sky: scaled(l.sky_color, l.intensity),
                            ground: scaled(l.ground_color, l.intensity),
                            up: [up.x, up.y, up.z, 0.0],
                        };
                        n_hemi += 1;
                    }
                    // three.js orients it by its world matrix's rotation
                    // alone (`extractRotation`): width along local +x,
                    // height along +y, emitting toward local −z.
                    Light::RectArea(l) => {
                        let n = frame_u.rect_counts[0] as usize;
                        if n >= MAX_RECT_LIGHTS {
                            return;
                        }
                        let p = obj.world_position();
                        let ax = transform_direction(&world, Vector3::new(1.0, 0.0, 0.0)).normalize();
                        let ay = transform_direction(&world, Vector3::new(0.0, 1.0, 0.0)).normalize();
                        let (hw, hh) = (l.width * 0.5, l.height * 0.5);
                        frame_u.rects[n] = RectLightGpu {
                            position: [p.x, p.y, p.z, 1.0],
                            color: scaled(l.color, l.intensity),
                            right: [ax.x * hw, ax.y * hw, ax.z * hw, 0.0],
                            up: [ay.x * hh, ay.y * hh, ay.z * hh, 0.0],
                        };
                        frame_u.rect_counts[0] += 1;
                    }
                },
                ObjectKind::Group => {}
                #[cfg(feature = "audio")]
                ObjectKind::Audio(_) => {}
            }
        });

        frame_u.counts = [n_dir as u32, n_point as u32, n_spot as u32, n_hemi as u32];
        stats.geometry_bytes = self.geometries.values().map(|g| g.bytes).sum();
        // Logged on meaningful change rather than per frame: on a scene that
        // grows as it animates this is the number that predicts a stall, and
        // it is invisible otherwise.
        let last = self
            .logged_geometry_bytes
            .load(std::sync::atomic::Ordering::Relaxed);
        if stats.geometry_bytes.abs_diff(last) > (last / 8).max(64 << 20) {
            self.logged_geometry_bytes
                .store(stats.geometry_bytes, std::sync::atomic::Ordering::Relaxed);
            log::info!(
                "geometry buffers: {:.2} GB across {} cached ({})",
                stats.geometry_bytes as f64 / (1u64 << 30) as f64,
                self.geometries.len(),
                if slim_lines_enabled() { "slim lines" } else { "full vertices" },
            );
        }
        (frame_u, items, stats, casters)
    }
}
