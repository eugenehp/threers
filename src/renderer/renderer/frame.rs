use super::*;

impl Renderer {
    /// three.js `renderer.clear()` for colour: fill `target_view` with `color`
    /// — the scissor rectangle when the scissor test is on, the whole target
    /// otherwise. `target_view` is taken to be drawing-buffer sized, as for
    /// [`render`](Self::render); `linear_framebuffer` means the same as there.
    pub fn clear(
        &mut self,
        target_view: &wgpu::TextureView,
        linear_framebuffer: bool,
        color: crate::math::Color,
        alpha: f32,
    ) {
        let size = self.depth_size;
        let format = if linear_framebuffer {
            wgpu::TextureFormat::Rgba16Float
        } else {
            self.color_format
        };
        self.clear_inner(target_view, size, format, color, alpha);
    }

    /// [`clear`](Self::clear) into an offscreen [`RenderTarget`](super::super::RenderTarget).
    pub fn clear_render_target(
        &mut self,
        target: &super::super::RenderTarget,
        color: crate::math::Color,
        alpha: f32,
    ) {
        self.clear_inner(
            &target.color_view,
            (target.width, target.height),
            target.format,
            color,
            alpha,
        );
    }

    pub(super) fn clear_inner(
        &mut self,
        view: &wgpu::TextureView,
        (w, h): (u32, u32),
        format: wgpu::TextureFormat,
        color: crate::math::Color,
        alpha: f32,
    ) {
        use super::super::viewport::ClearMode;
        let Some(mode) = self.viewport_state.settings.clear_mode(w, h) else {
            return;
        };
        let value = super::super::clear_value(format, color, alpha);
        if let ClearMode::Rect(_) = mode {
            self.viewport_state
                .ensure_clear_pipeline(&self.device, format, None);
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("threers clear"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("threers clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: if mode == ClearMode::Full {
                            wgpu::LoadOp::Clear(value)
                        } else {
                            wgpu::LoadOp::Load
                        },
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if let ClearMode::Rect(r) = mode {
                self.viewport_state
                    .fill_rect(&mut pass, format, false, r, value);
            }
        }
        self.queue.submit(Some(encoder.finish()));
    }

    pub fn render_to_cube_face(
        &mut self,
        scene: &mut crate::scene::Scene,
        camera: &dyn Camera,
        target: &super::super::CubeRenderTarget,
        face: usize,
    ) {
        // Tell the frame passes they are drawing a probe, not the picture.
        let was_cube = std::mem::replace(&mut self.cube_pass, true);
        let (orig_w, orig_h) = self.depth_size;
        let orig_depth = std::mem::replace(
            &mut self.depth_view,
            target
                .depth_texture
                .create_view(&wgpu::TextureViewDescriptor::default()),
        );
        self.depth_size = (target.side, target.side);
        let face_view = &target.face_views[face % 6];
        self.render(scene, camera, face_view, false);
        self.depth_view = orig_depth;
        self.depth_size = (orig_w, orig_h);
        self.cube_pass = was_cube;
    }

    /// Render `scene` from `camera` into an offscreen [`RenderTarget`](crate::RenderTarget). Uses
    /// the target's own depth buffer (required even when RT size matches the canvas).
    pub fn render_to(
        &mut self,
        scene: &mut crate::scene::Scene,
        camera: &dyn Camera,
        target: &super::super::RenderTarget,
    ) {
        let (orig_w, orig_h) = self.depth_size;
        let orig_depth = std::mem::replace(&mut self.depth_view, target.depth_view_clone());
        self.depth_size = (target.width, target.height);
        let linear_fb = target.format == wgpu::TextureFormat::Rgba16Float;
        let tone_map = std::mem::replace(&mut self.tone_map_pass, target.presents());
        // Resolving needs the multisampled colour in the target's format, and
        // the renderer's multisampled targets are in its own.
        if target.samples() > 1 && target.format == self.color_format {
            // three.js `samples`: drawn multisampled at the target's size,
            // colour resolved into it and depth resolved (or, on WebGL2,
            // redrawn) into its depth, which the glass and post-processing
            // read as they would a single-sampled target's. The renderer's own
            // MSAA setting is the canvas's and is left as it was.
            let msaa = std::mem::replace(&mut self.msaa, MSAA_SAMPLES);
            self.msaa_depth_out = Some(target.depth_texture.clone());
            self.render_inner(
                scene,
                camera,
                &target.color_view,
                linear_fb,
                None,
                Some((&target.depth_texture, &target.color_texture)),
            );
            self.msaa_depth_out = None;
            self.msaa = msaa;
        } else {
            // The target owns the live depth this pass — pass it as the copy
            // source for the screen-space glass ray march.
            self.render_inner(
                scene,
                camera,
                &target.color_view,
                linear_fb,
                Some(&target.depth_texture),
                Some((&target.depth_texture, &target.color_texture)),
            );
        }
        self.tone_map_pass = tone_map;
        self.depth_view = orig_depth;
        self.depth_size = (orig_w, orig_h);
    }

    /// Draw into a [`RenderTarget`](super::super::RenderTarget) the surface-style way —
    /// the path MSAA needs — while still handing the temporal pass the textures
    /// it reads.
    ///
    /// [`render`](Self::render) passes `None` for both, which silently turns TAA
    /// off: it is gated on having a colour source, and that path has none to
    /// give. Multisampling and temporal accumulation are complementary — one
    /// resolves coverage, the other resolves shading — and a caller that asked
    /// for both got only the first, with nothing said about it. Measured on a
    /// phone's side rail, thirty-two accumulated samples took the banding from
    /// 0.99 to 0.28 with MSAA off, and did exactly nothing with MSAA on.
    ///
    /// The depth handed over is the target's, which the multisampled path does
    /// not write. That is why this is separate from [`render_to`](Self::render_to)
    /// rather than folded into it: the temporal pass binds depth but only reads
    /// it when reprojecting, which [`set_taa_converge`](Self::set_taa_converge)
    /// does not do, and the screen-space glass march — the other consumer of a
    /// live depth — is deliberately not given it here.
    pub fn render_to_multisampled(
        &mut self,
        scene: &mut crate::scene::Scene,
        camera: &dyn Camera,
        target: &super::super::RenderTarget,
        linear_framebuffer: bool,
    ) {
        let tone_map = std::mem::replace(&mut self.tone_map_pass, target.presents());
        self.last_frame_msaa = false;
        self.render_inner(
            scene,
            camera,
            &target.color_view,
            linear_framebuffer,
            None,
            Some((&target.depth_texture, &target.color_texture)),
        );
        self.tone_map_pass = tone_map;
        // Antialiasing was asked for and the frame could not multisample
        // (`Oit` or `Refract` glass in view): FXAA it rather than hand back
        // hard edges. Temporal AA, when on, has already done the job.
        if self.msaa > 1 && !self.last_frame_msaa && !self.taa_enabled {
            self.fxaa_fallback(target);
        }
    }

    /// FXAA over `target`, read from a copy of it.
    ///
    /// On the stored bytes, as three.js runs FXAA over its 8-bit canvas: an
    /// sRGB copy decodes as it is read, so the pass is told (`raw_srgb` bit 2)
    /// and re-encodes, and the sRGB target encodes the result back to the
    /// same bytes. A flat area comes back byte for byte.
    pub(super) fn fxaa_fallback(&mut self, target: &super::super::RenderTarget) {
        let tex = &target.color_texture;
        let (w, h, format) = (tex.width(), tex.height(), tex.format());
        if format == wgpu::TextureFormat::Rgba16Float {
            // A linear target is a stage on the way somewhere else; whatever
            // presents it can antialias the result.
            return;
        }
        let stale = self
            .fxaa_scratch
            .as_ref()
            .is_none_or(|(t, _)| t.width() != w || t.height() != h || t.format() != format);
        if stale {
            let t = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers fxaa fallback source"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let v = t.create_view(&wgpu::TextureViewDescriptor::default());
            self.fxaa_scratch = Some((t, v));
        }
        let (scratch, view) = self.fxaa_scratch.as_ref().unwrap();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("threers fxaa fallback copy"),
            });
        encoder.copy_texture_to_texture(
            tex.as_image_copy(),
            scratch.as_image_copy(),
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));
        let view = view.clone();
        let normal = self
            .default_normal
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let hw_depth = self.hw_depth_tex.create_view(&wgpu::TextureViewDescriptor {
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        let raw_input = if format.is_srgb() { 2 } else { 0 };
        self.apply_postfx_views(
            &view,
            &target.color_view,
            &view,
            &normal,
            &hw_depth,
            1,
            0.0,
            [1.0 / w as f32, 1.0 / h as f32, 0.0, 0.0],
            [0.0; 4],
            w,
            h,
            false,
            PostFxCamera::default(),
            false,
            format,
            raw_input,
        );
    }

    /// Draw `scene` from `camera` into `target_view`.
    /// `linear_framebuffer`: when true, skip sRGB encode (HalfFloat / postfx RT).
    pub fn render(
        &mut self,
        scene: &mut Scene,
        camera: &dyn Camera,
        target_view: &wgpu::TextureView,
        linear_framebuffer: bool,
    ) {
        self.render_inner(scene, camera, target_view, linear_framebuffer, None, None);
    }

    /// Body of [`render`]. `depth_src` is the depth texture backing the current
    /// depth attachment (the external target's, via `render_to`) — the copy
    /// source for the screen-space glass `ss_depth`; `None` uses the renderer's
    /// own depth (the direct-`render`/surface path).
    pub(super) fn render_inner(
        &mut self,
        scene: &mut Scene,
        camera: &dyn Camera,
        target_view: &wgpu::TextureView,
        linear_framebuffer: bool,
        depth_src: Option<&wgpu::Texture>,
        // `(depth, colour)` for the temporal pass. Separate from `depth_src`,
        // which is the *live* depth for the screen-space glass march — the
        // multisampled path has one and not the other.
        taa_src: Option<(&wgpu::Texture, &wgpu::Texture)>,
    ) {
        scene.update_world();
        if self.capturing_transmission.is_none() && !self.cube_pass {
            self.capture_transmission(scene, camera);
        }

        // Viewport / scissor for this render. A cube face is always the whole
        // face. `None` means the viewport or scissor misses the target and
        // there is nothing to draw.
        let vp_plan = if self.cube_pass {
            super::super::viewport::ViewportPlan::full(self.depth_size.0, self.depth_size.1)
        } else {
            match self
                .viewport_state
                .settings
                .plan(self.depth_size.0, self.depth_size.1)
            {
                Some(p) => p,
                None => return,
            }
        };

        // Whether the shader writes linear light and lets something else encode
        // it. Two different reasons to: the target stores linear (a float
        // framebuffer, which is what `linear_framebuffer` says), or the target's
        // format is sRGB and the hardware encodes on write.
        //
        // This is *not* the same question as `linear_framebuffer`, which every
        // other use of that flag below is really asking — "is this a float
        // target, so pick the f16 pipeline". Conflating the two is what made an
        // sRGB target double-encode. Picking the format off `self.color_format`
        // rather than the target view is sound because every non-float target
        // the renderer draws into is the one its pipelines were built against.
        let shader_outputs_linear =
            linear_framebuffer || !super::super::shader_applies_transfer(self.color_format);

        // TAA active only when we have a sampleable color+depth target.
        // NOT during a probe. A cube face is 48 pixels and is not the picture:
        // running temporal AA over it reprojects the frame's history through
        // the probe's camera and writes the result back, so the next real frame
        // blends against six tiny views of the scene. On screen that reads as
        // the whole vehicle going half transparent — which is what it looked
        // like, and why the cause was not obvious from the symptom.
        let taa_active = self.taa_enabled && !self.cube_pass && taa_src.is_some() && vp_plan.legacy;

        let view_m = camera.view_matrix();
        let mut proj_m = camera.projection_matrix();
        if let Some(fix) = &vp_plan.projection_fix {
            proj_m = fix.multiply(&proj_m);
        }
        // Un-jittered view-projection is kept for TAA reprojection.
        let unjittered = proj_m.multiply(&view_m);
        if taa_active || self.jitter_override.is_some() {
            // Sub-pixel Halton jitter, half a pixel either way.
            //
            // Deliberately a box over the pixel and not a weighted
            // reconstruction: a pixel *is* a box, so weighting the samples by
            // a tent and widening the jitter to match was tried and measures
            // worse — the synthetic edge went from 0.0216 px of deviation to
            // 0.0318, because spreading the same samples over four times the
            // area undersamples it faster than the filter narrows it.
            let (ox, oy) = self.jitter_override.unwrap_or((
                halton(self.taa_frame + 1, 2) - 0.5,
                halton(self.taa_frame + 1, 3) - 0.5,
            ));
            // Per viewport pixel — the whole target unless a viewport is set.
            let jx = ox * 2.0 / vp_plan.viewport.width as f32;
            let jy = oy * 2.0 / vp_plan.viewport.height as f32;
            // Which element carries the offset depends on the projection, and
            // getting it wrong is not subtle. A perspective matrix divides by
            // `-z`, so a shift in the *z* column arrives in NDC as a constant:
            // `x_clip += jx·z` over `w = -z` is `-jx` at every depth. An
            // orthographic matrix does not divide — `w` is 1 — so that same
            // shift lands as `jx·z`, which is a shear that grows with distance
            // from the camera, not a sub-pixel offset. With the camera ten
            // units back it spread a hard edge over nine pixels. Orthographic
            // wants the translation column, where the offset is already in NDC.
            //
            // `elements[11]` is `-1` for perspective and `0` for orthographic,
            // which is the projection asked about rather than the camera type
            // guessed at.
            if proj_m.elements[11].abs() > 0.5 {
                proj_m.elements[8] += jx;
                proj_m.elements[9] += jy;
            } else {
                // Negated so the jitter means the same direction either way.
                proj_m.elements[12] -= jx;
                proj_m.elements[13] -= jy;
            }
        }
        // Reversed depth flips only what rasterises: TAA, motion blur and
        // culling keep the standard matrices, and every reader of the depth
        // buffer converts it back (`1 - d`) before using them.
        let reversed = self.state_pipelines.reversed();
        if reversed {
            proj_m = reverse_depth(&proj_m);
        }
        let view_proj = proj_m.multiply(&view_m).elements;
        let taa_inv_vp = unjittered.invert().elements;
        let taa_unjittered_vp = unjittered.elements;

        // tone_mapping_exposure[2] is the "IBL on" flag — the PBR shader's
        // env-map branch is gated on it. Enable whenever the scene has any
        // environment source (CPU cube texture or cube render target).
        let has_env = scene.environment.is_some() || scene.environment_cube_rt.is_some();
        let (cam_near, cam_far) = camera.near_far();
        // Remembered for the post passes. Depth linearisation needs these and
        // they are a property of the frame, not of the effect — making every
        // caller pass them again is an invitation to pass the wrong ones, and
        // the failure is silent: the occlusion or the blur is simply computed
        // at the wrong distances and still looks like an effect.
        self.last_camera_planes = (cam_near, cam_far);
        // The lens, from the projection itself: for a perspective matrix
        // `m[5]` is `1 / tan(fov/2)` and `m[0]` is that over the aspect.
        let proj = camera.projection_matrix();
        let (m0, m5) = (proj.elements[0], proj.elements[5]);
        if m5.abs() > 1e-9 && m0.abs() > 1e-9 {
            self.last_camera_lens = (1.0 / m5, m5 / m0);
        }
        let mut frame_u = FrameUniforms {
            view: view_m.elements,
            projection: proj_m.elements,
            view_proj,
            // w: 1 with reversed depth — the sky pin and the screen-space
            // marches read it.
            camera_position: [
                camera.position().x,
                camera.position().y,
                camera.position().z,
                if reversed { 1.0 } else { 0.0 },
            ],
            fog_color: [scene.fog.color.r, scene.fog.color.g, scene.fog.color.b, 1.0],
            fog_params: [
                scene.fog.near,
                scene.fog.far,
                scene.fog.density,
                scene.fog.mode as f32,
            ],
            tone_mapping_exposure: [
                if self.tone_map_pass && !self.cube_pass {
                    self.tone_mapping.shader_mode()
                } else {
                    0.0
                },
                self.exposure,
                if has_env { 1.0 } else { 0.0 },
                // 2: linear because the target is sRGB and encodes on write,
                // which the shader's data outputs (`framebuffer_data`) undo.
                if !shader_outputs_linear {
                    0.0
                } else if !linear_framebuffer && self.color_format.is_srgb() {
                    2.0
                } else {
                    1.0
                },
            ],
            viewport_size: [
                self.depth_size.0 as f32,
                self.depth_size.1 as f32,
                cam_near,
                cam_far,
            ],
            ..FrameUniforms::default()
        };

        let mut ambient = [0.0_f32; 3];
        let mut n_dir = 0usize;
        let mut n_point = 0usize;
        let mut n_spot = 0usize;
        let mut n_hemi = 0usize;
        let mut n_rect = 0usize;
        let oit_available = !self.pipeline_oit.is_empty();
        // Every shadow-casting light, by kind: its index in the kind's light
        // array, and what its shadow pass needs.
        // Every shadow-casting light, in the order its record takes. A light
        // names its record in its own `color[3]`.
        let mut casters: Vec<Caster> = Vec::new();

        #[derive(Clone, Copy, PartialEq, Eq)]
        enum Topology {
            Triangle,
            TriangleAlpha,
            TriangleWire,
            TriangleNoCull,
            Sky,
            Line,
            /// three.js `Points`: one instance per point, each a quad
            /// (`vs_points`).
            Point,
            /// `Point` with a transparent material: blended and sorted with
            /// the other transparent draws.
            PointAlpha,
            /// `LineSegments` wider than a pixel: one instance per segment,
            /// each a round-capped quad (`vs_wide_lines`).
            WideLine,
            /// `WideLine` blended: sorted with the transparent draws, its
            /// edges faded by coverage.
            WideLineAlpha,
            Sprite,
            Instanced,
            /// `Instanced` with a `wireframe: true` material: line list.
            InstancedWire,
            /// `Instanced` with a transparent material: alpha-blended, sorted
            /// with the other transparent draws, as three.js renders a
            /// transparent `InstancedMesh`.
            InstancedAlpha,
            Skinned,
            /// `Skinned` with a `wireframe: true` material: line list.
            SkinnedWire,
            /// `Skinned` with a transparent material.
            SkinnedAlpha,
            /// Single-layer glass depth prepass (writes depth, no color).
            GlassDepth,
            /// The tint half of the glass colour pass, drawn just before
            /// `GlassColor`: what is behind times the glass's transmittance.
            GlassTint,
            /// Single-layer glass color pass (LessEqual, no depth write): the
            /// light half, added over the tint.
            GlassColor,
            /// Weighted-blended OIT surface (drawn into accum+revealage MRT).
            Oit,
            /// Screen-space refraction glass (drawn in a dedicated pass that
            /// samples the captured opaque color + depth).
            Refract,
            /// Custom-shader material:
            /// (fragment-source hash, is-transparent, wants-screen-space).
            /// The last flag moves the draw into the screen-space pass, where the
            /// captured opaque colour and depth are bound at `@group(3)`.
            /// (shader hash, transparent, screen-space, side) — exactly the
            /// custom pipeline cache key minus the framebuffer format, so
            /// every lookup site can build the key from the topology alone.
            Custom(u64, bool, bool, u32),
        }
        impl Topology {
            /// A plain triangle mesh that a glass mode draws in passes of its
            /// own: in the shadow maps it is the triangles it always was.
            fn is_glass_mesh(self) -> bool {
                matches!(
                    self,
                    Topology::Refract | Topology::Oit | Topology::GlassColor
                )
            }
            /// Drawn from the mesh's wire (line-list) index buffer.
            fn is_wire(self) -> bool {
                matches!(
                    self,
                    Topology::TriangleWire | Topology::InstancedWire | Topology::SkinnedWire
                )
            }
            /// Reads per-instance matrices from vertex slot 1.
            fn is_instanced(self) -> bool {
                matches!(
                    self,
                    Topology::Instanced | Topology::InstancedWire | Topology::InstancedAlpha
                )
            }
            /// Binds the combined mesh+skin group and joint attributes.
            fn is_skinned(self) -> bool {
                matches!(
                    self,
                    Topology::Skinned | Topology::SkinnedWire | Topology::SkinnedAlpha
                )
            }
            /// Drawn as faces, so three.js's `side` culls them (`side_cull`).
            fn is_faces(self) -> bool {
                matches!(
                    self,
                    Topology::Triangle
                        | Topology::TriangleNoCull
                        | Topology::TriangleAlpha
                        | Topology::Instanced
                        | Topology::InstancedAlpha
                        | Topology::Skinned
                        | Topology::SkinnedAlpha
                )
            }
            /// One instance per point, a quad each, read from the geometry's
            /// own vertex buffer.
            fn is_points(self) -> bool {
                matches!(self, Topology::Point | Topology::PointAlpha)
            }
            /// One instance per line segment, read two vertices at a time.
            fn is_wide_lines(self) -> bool {
                matches!(self, Topology::WideLine | Topology::WideLineAlpha)
            }
            /// Blends with what is behind it — and so, on an sRGB target,
            /// draws in the encoded-blend pass (see `encoded_pass`). Glass
            /// included, prepass and all: it sorts with the transparent draws
            /// by `renderOrder`, so it has to be drawn in the same pass as
            /// them or a glass shell ordered after them still went first.
            fn encodes(self) -> bool {
                matches!(
                    self,
                    Topology::GlassDepth
                        | Topology::GlassTint
                        | Topology::GlassColor
                        | Topology::TriangleAlpha
                        | Topology::PointAlpha
                        | Topology::WideLineAlpha
                        | Topology::Sprite
                        | Topology::InstancedAlpha
                        | Topology::SkinnedAlpha
                        | Topology::Custom(_, true, false, ..)
                )
            }
        }
        #[derive(Clone)]
        struct DrawMesh {
            key: *const BufferGeometry,
            topology: Topology,
            /// Z in view space, used to back-to-front sort transparent draws.
            view_z: f32,
            render_order: i32,
            /// False when the camera frustum cannot see this. It still has to
            /// be drawn into the shadow maps — a caster behind the camera can
            /// throw a shadow across the whole view — so culled draws stay in
            /// this list and only the colour passes skip them.
            in_frustum: bool,
            /// three.js's transmissive material (`Material::transmissive`).
            transmissive: bool,
            /// Any transmissive physical material: cast with the cut-out shadow
            /// pipelines, which drop texels for the light it lets through.
            shadow_transmits: bool,
            cast_shadow: bool,
            refract_capture: bool,
            alpha_test: f32,
            /// The material's three.js `side` (0 front, 1 back, 2 double): the
            /// shadow passes cull by it, as three.js's `shadowSide` does.
            side: u32,
            /// The `side` the shadow passes use: the material's, even when the
            /// colour passes draw a blended DoubleSide mesh as its back and
            /// front halves (see the two-pass split before the sort).
            shadow_side: u32,
            model: [f32; 16],
            normal_matrix: [f32; 16],
            color: [f32; 4],
            emissive: [f32; 4],
            specular: [f32; 4],
            shininess_or_near: f32,
            opacity_or_far: f32,
            roughness: f32,
            metalness: f32,
            ao_intensity: f32,
            normal_scale: [f32; 2],
            physical: [f32; 4],
            physical2: [f32; 4],
            physical3: [f32; 4],
            physical4: [f32; 4],
            physical5: [f32; 4],
            displacement: [f32; 4],
            uv_transform: [f32; 4],
            atmos: [f32; 4],
            twilight: [f32; 4],
            eclipse: [f32; 4],
            scatter: [f32; 4],
            aurora: [f32; 4],
            shader_flags: u32,
            kind: MaterialKind,
            slots: MaterialTextureSlots,
            /// The material shader variant this draw needs, if its maps do not
            /// fit the legacy layout (see `material_maps`).
            map_variant: Option<super::super::material_maps::MapVariant>,
            /// `Material::map_uniforms`, a variant's uniform block.
            map_uniforms: [[f32; 4]; 6],
            /// A Standard/Physical material's own environment
            /// (`material_env_groups`), in place of the scene's.
            env_group: Option<MaterialEnvKey>,
            /// Index into `instance_buffers` for Topology::Instanced.
            instance_buf_idx: usize,
            /// The instance data's content stamp (`InstanceBuffer::stamp`),
            /// for the shadow fingerprint; 0 for a plain mesh.
            instance_stamp: u64,
            instance_count: u32,
            /// Index into `skin_data` for Topology::Skinned.
            skin_idx: usize,
            /// Index into `shader_data` for Topology::Custom.
            shader_idx: usize,
            /// The part of the index (vertices, unindexed) this draw covers —
            /// a group and/or the draw range, three.js's `renderBufferDirect`
            /// clipping. `None` is the whole geometry.
            range: Option<(u32, u32)>,
            /// Morph uniforms (count 0: no morphing) and the texture to bind.
            morph: [f32; 4],
            morph_weights: [f32; crate::core::MAX_MORPH_TARGETS],
            morph_tex: (usize, u32),
            /// three.js `polygonOffset`: (factor bits, units), drawn with a
            /// depth-biased pipeline (see `Renderer::biased_pipelines`).
            polygon_offset: Option<(u32, i32)>,
            /// three.js `depthTest` / `depthWrite` / `depthFunc` /
            /// `colorWrite`, applied over whichever pipeline the draw picks
            /// (see `pipeline_state`). Always the default for the glass,
            /// order-independent and refraction modes, which own theirs.
            state: crate::materials::RenderState,
            /// `Object3D::clip_plane`, packed for `MeshUniforms`.
            clip_plane: [f32; 4],
        }
        let mut draws: Vec<DrawMesh> = Vec::new();
        // The triangle topologies a three.js `polygonOffset` applies to, as
        // the code `build_biased_pipeline` builds them from: plain (back-face
        // culled), alpha-blended, double-sided.
        let biased_topology = |t: Topology| -> Option<u8> {
            match t {
                Topology::Triangle => Some(0),
                Topology::TriangleAlpha => Some(1),
                Topology::TriangleNoCull => Some(2),
                _ => None,
            }
        };
        // Per-custom-shader-material draw data (parallel to Topology::Custom).
        struct ShaderDrawData {
            fragment: String,
            vertex: Option<String>,
            data: [[f32; 4]; 16],
            storage: [Vec<f32>; 3],
            textures: Vec<crate::marks::gpu::TextureView>,
            /// `ShaderMaterial::maps`: the slots `textures` leaves empty.
            maps: Vec<Arc<Texture>>,
            volumes: Vec<crate::marks::gpu::TextureView>,
            /// Drawn from an `InstancedMesh`: the instanced pipeline layout,
            /// with the instance buffer at `instance_buf_idx` in slot 1 and
            /// the custom columns at `custom_buf` in slot 2.
            instanced: bool,
            custom_buf: usize,
            /// three.js `alphaToCoverage`.
            alpha_to_coverage: bool,
        }
        let mut shader_data: Vec<ShaderDrawData> = Vec::new();
        let mut instance_buffers: Vec<wgpu::Buffer> = Vec::new();
        struct SkinData {
            bone_buf: wgpu::Buffer,
            bg: Option<wgpu::BindGroup>,
            /// The bone matrices, hashed, for the shadow fingerprint.
            bone_hash: u64,
        }
        let mut skin_data: Vec<SkinData> = Vec::new();

        // Helper: build a DrawMesh from a geometry+material pair under a given topology.
        let make_draw = |geom: &std::sync::Arc<BufferGeometry>,
                         mat: &Material,
                         obj: &crate::core::Object3D,
                         topology: Topology|
         -> DrawMesh {
            let key = geom_cache_key(geom);
            let nm3 = Matrix3::normal_matrix(&obj.matrix_world);
            // Mirror materials repurpose the normal_matrix slot for the
            // projective texture matrix supplied per-frame from JS.
            let normal_matrix = match mat {
                Material::Mirror(m) => m.texture_matrix,
                _ => mat3_to_mat4_array(&nm3),
            };
            // Promote Triangle to TriangleAlpha / TriangleWire based on material
            // flags. Pre-set view_z for later sort.
            let world_pos = obj.world_position();
            let view_z = view_m.elements[2] * world_pos.x
                + view_m.elements[6] * world_pos.y
                + view_m.elements[10] * world_pos.z
                + view_m.elements[14];
            let topology = if topology == Topology::Instanced && mat.wireframe() {
                Topology::InstancedWire
            } else if topology == Topology::Instanced && mat.transparent() {
                Topology::InstancedAlpha
            } else if topology == Topology::Skinned && mat.wireframe() {
                Topology::SkinnedWire
            } else if topology == Topology::Skinned && mat.transparent() {
                Topology::SkinnedAlpha
            } else if topology == Topology::Triangle {
                if matches!(mat, Material::Sky(_)) {
                    Topology::Sky
                } else if mat.wireframe() {
                    Topology::TriangleWire
                } else if mat.transparent() {
                    Topology::TriangleAlpha
                } else if mat.side() != 0 {
                    Topology::TriangleNoCull
                } else {
                    Topology::Triangle
                }
            } else {
                topology
            };
            let c = mat.color();
            let e = mat.emissive();
            // For MeshDistanceMaterial, pack the reference position into the
            // specular channel (unused by that path) so the shader can read it.
            // For SkyMaterial, pack the sun position into specular.
            let s = match mat {
                Material::Distance(m) => crate::math::Color::new(
                    m.reference_position.x,
                    m.reference_position.y,
                    m.reference_position.z,
                ),
                Material::Sky(m) => {
                    crate::math::Color::new(m.sun_position.x, m.sun_position.y, m.sun_position.z)
                }
                _ => mat.specular(),
            };
            // For SkyMaterial, pack atmospheric tuning into params.x/y (replacing
            // shininess/opacity which it doesn't use).
            let (shin_or_near, opacity_or_far) = match mat {
                Material::Depth(m) => (m.near, m.far),
                Material::Distance(m) => (m.near_distance, m.far_distance),
                Material::Sky(m) => (m.turbidity, m.rayleigh),
                _ => (mat.shininess(), mat.opacity()),
            };
            let physical = match mat {
                Material::Physical(m) => {
                    [m.clearcoat, m.clearcoat_roughness, m.ior, m.transmission]
                }
                Material::Line(m) => [m.dash_scale, m.dash_size, m.gap_size, 0.0],
                // The atmosphere shader intersects the view ray with both
                // spheres, so it needs their radii rather than a surface
                // description.
                Material::Atmosphere(m) => {
                    [m.planet_radius, m.atmosphere_radius, m.intensity, m.falloff]
                }
                _ => [0.0; 4],
            };
            let physical2 = match mat {
                Material::Physical(m) => [
                    m.thickness,
                    m.dispersion,
                    m.vertex_emissive,
                    m.anisotropy_rotation,
                ],
                Material::Atmosphere(m) => [
                    m.sunset_color.r,
                    m.sunset_color.g,
                    m.sunset_color.b,
                    m.airglow,
                ],
                _ => [0.0; 4],
            };
            let physical3 = match mat {
                Material::Physical(m) => [
                    m.iridescence,
                    m.iridescence_ior,
                    m.iridescence_thickness_range[1],
                    m.sheen,
                ],
                Material::Atmosphere(m) => {
                    [m.airglow_color.r, m.airglow_color.g, m.airglow_color.b, 0.0]
                }
                _ => [0.0; 4],
            };
            let physical4 = match mat {
                Material::Physical(m) => [
                    m.sheen_color.r,
                    m.sheen_color.g,
                    m.sheen_color.b,
                    m.sheen_roughness,
                ],
                _ => [0.0; 4],
            };
            let physical5 = match mat {
                Material::Physical(m) => {
                    // attenuation_distance defaults to INFINITY. Feeding that to
                    // the GPU makes the Beer-Lambert coefficient -log(c)/inf,
                    // which is 0 for c>0 but NaN for a fully-black channel — and
                    // one NaN poisons the whole fragment. Collapse any
                    // non-finite/non-positive distance to 0, the shader's
                    // "absorption disabled" sentinel.
                    let dist = if m.attenuation_distance.is_finite() && m.attenuation_distance > 0.0
                    {
                        m.attenuation_distance
                    } else {
                        0.0
                    };
                    [
                        m.attenuation_color.r,
                        m.attenuation_color.g,
                        m.attenuation_color.b,
                        dist,
                    ]
                }
                _ => [0.0; 4],
            };
            let displacement = {
                let (scale, bias) = mat.displacement();
                let night = mat.emissive_night_side();
                [scale, bias, night, 0.0]
            };
            // three.js gives every map its own matrix. The legacy layout
            // holds one scale and translation, applied in the vertex stage to
            // every map; a material whose maps need more gets a shader variant
            // that transforms each map itself (`material_maps::needs_variant`).
            let uv_transform =
                super::super::material_maps::legacy_uv_transform(&mat.texture_slots());
            let atmos = mat.atmosphere_params();
            let (eclipse, sun_angle) = mat.eclipse();
            let twilight = {
                let c = mat.twilight_color();
                [c.r, c.g, c.b, sun_angle]
            };
            let scatter = mat.scatter_params();
            let aurora = mat.aurora_params();
            let mut shader_flags = match mat {
                Material::Line(m) if m.dash_size > 0.0 || m.gap_size > 0.0 => FLAG_DASHED,
                Material::Basic(m) if m.shadow_only => FLAG_SHADOW_MAT,
                _ => 0,
            };
            if obj.receive_shadow {
                shader_flags |= FLAG_RECEIVE_SHADOW;
            }
            if !mat.tone_mapped() {
                shader_flags |= FLAG_NO_TONE_MAP;
            }
            let shadow_transmits = matches!(mat, Material::Physical(m)
                if m.transmission > 0.0
                    && (m.thickness <= 0.0 || m.transparency == crate::materials::TransparencyMode::Glass));
            if shadow_transmits {
                shader_flags |= FLAG_SHADOW_TRANSMITS;
            }
            if let Material::Physical(m) = mat
                && m.transmission > 0.0
                && matches!(
                    m.transparency,
                    crate::materials::TransparencyMode::Glass
                        | crate::materials::TransparencyMode::Oit
                )
            {
                shader_flags |= FLAG_BLENDED_GLASS;
            }
            DrawMesh {
                key,
                topology,
                view_z,
                render_order: obj.render_order,
                clip_plane: clip_plane_of(obj),
                in_frustum: true,
                transmissive: mat.transmissive(),
                shadow_transmits,
                cast_shadow: obj.cast_shadow,
                refract_capture: obj.refract_capture,
                alpha_test: mat.alpha_test(),
                side: mat.side(),
                shadow_side: mat.side(),
                model: obj.matrix_world.elements,
                normal_matrix,
                color: [c.r, c.g, c.b, mat.opacity()],
                // emissive.w carries the physical-material anisotropy strength
                // (0 = isotropic); the shader stretches the specular lobe along
                // the surface tangent by this amount. Non-physical mats leave 0.
                emissive: [
                    e.r,
                    e.g,
                    e.b,
                    match mat {
                        Material::Physical(m) => m.anisotropy,
                        _ => 0.0,
                    },
                ],
                // Phong's specular colour; Physical's `specularColor` and, in
                // w, `specularIntensity`.
                specular: match mat {
                    Material::Physical(m) => {
                        let c = m.specular_color;
                        [c.r, c.g, c.b, m.specular_intensity]
                    }
                    _ => [s.r, s.g, s.b, 0.0],
                },
                shininess_or_near: shin_or_near,
                opacity_or_far,
                // For SkyMaterial, repurpose roughness/metalness as
                // params.z/w (mieCoefficient/mieDirectionalG).
                roughness: match mat {
                    Material::Sky(m) => m.mie_coefficient,
                    _ => mat.roughness(),
                },
                metalness: match mat {
                    Material::Sky(m) => m.mie_directional_g,
                    _ => mat.metalness(),
                },
                ao_intensity: mat.ao_intensity(),
                normal_scale: [mat.normal_scale().x, mat.normal_scale().y],
                physical,
                physical2,
                physical3,
                physical4,
                physical5,
                displacement,
                uv_transform,
                atmos,
                twilight,
                eclipse,
                scatter,
                aurora,
                shader_flags,
                kind: mat.kind(),
                slots: mat.texture_slots(),
                map_variant: None,
                map_uniforms: mat.map_uniforms(),
                env_group: mat.pbr_env().map(material_env_key),
                instance_buf_idx: usize::MAX,
                instance_stamp: 0,
                instance_count: 1,
                skin_idx: usize::MAX,
                shader_idx: usize::MAX,
                range: None,
                morph: [0.0; 4],
                morph_weights: [0.0; crate::core::MAX_MORPH_TARGETS],
                morph_tex: (0, 0),
                polygon_offset: mat
                    .polygon_offset()
                    .map(|o| (o.factor.to_bits(), o.units.round() as i32)),
                state: mat.render_state(),
            }
        };

        let cam_layers = camera.layers();
        let root = scene.root;
        self.instance_frame += 1;
        // Meshes not drawn for a while (removed, hidden, culled away) give
        // their buffers back. Generous, as a frame of several panes renders
        // more than once.
        let now = self.instance_frame;
        self.instance_cache.retain(|_, c| now - c.frame < 64);
        self.evict_unused_textures(now);
        scene.arena.traverse_visible(root, &mut |obj_id, obj| {
            if !cam_layers.test(&obj.layers) {
                return;
            }
            match &obj.kind {
                ObjectKind::Light(light) => match light {
                    Light::Ambient(l) => {
                        ambient[0] += l.color.r * l.intensity;
                        ambient[1] += l.color.g * l.intensity;
                        ambient[2] += l.color.b * l.intensity;
                    }
                    Light::Directional(l) => {
                        if n_dir >= MAX_DIR_LIGHTS {
                            return;
                        }
                        // three.js: direction = normalize(target - position); default target (0,0,0).
                        let pos = obj.world_position();
                        let dir = {
                            let d = Vector3::new(-pos.x, -pos.y, -pos.z);
                            if d.length_sq() > 1e-8 {
                                d.normalize()
                            } else {
                                transform_direction(&obj.matrix_world, l.direction).normalize()
                            }
                        };
                        let c = l.color;
                        frame_u.dir_lights[n_dir] = DirLightGpu {
                            direction: [dir.x, dir.y, dir.z, 0.0],
                            color: [c.r * l.intensity, c.g * l.intensity, c.b * l.intensity, 0.0],
                        };
                        if l.cast_shadow && casters.len() < MAX_SHADOW_RECORDS {
                            casters.push(Caster::Dir {
                                dir,
                                settings: l.shadow,
                            });
                            frame_u.dir_lights[n_dir].color[3] = casters.len() as f32;
                        }
                        n_dir += 1;
                    }
                    Light::Point(l) => {
                        if n_point >= MAX_POINT_LIGHTS {
                            return;
                        }
                        let p = obj.world_position();
                        let c = l.color;
                        frame_u.point_lights[n_point] = PointLightGpu {
                            position: [p.x, p.y, p.z, 0.0],
                            color: [c.r * l.intensity, c.g * l.intensity, c.b * l.intensity, 0.0],
                            params: [l.distance, l.decay, 0.0, 0.0],
                        };
                        if l.cast_shadow {
                            // three.js `PointLightShadow`: `light.distance ||
                            // camera.far`.
                            let radius = if l.distance > 0.0 {
                                l.distance
                            } else {
                                l.shadow.camera_far
                            };
                            if casters.len() < MAX_SHADOW_RECORDS {
                                casters.push(Caster::Point {
                                    pos: p,
                                    radius,
                                    settings: l.shadow,
                                });
                                frame_u.point_lights[n_point].color[3] = casters.len() as f32;
                            }
                        }
                        n_point += 1;
                    }
                    Light::Spot(l) => {
                        if n_spot >= MAX_SPOT_LIGHTS {
                            return;
                        }
                        let p = obj.world_position();
                        let dir = transform_direction(&obj.matrix_world, l.direction).normalize();
                        let c = l.color;
                        let cos_outer = l.angle.cos();
                        let cos_inner = (l.angle * (1.0 - l.penumbra)).cos();
                        frame_u.spot_lights[n_spot] = SpotLightGpu {
                            position: [p.x, p.y, p.z, 0.0],
                            direction: [dir.x, dir.y, dir.z, 0.0],
                            color: [c.r * l.intensity, c.g * l.intensity, c.b * l.intensity, 0.0],
                            params: [l.distance, l.decay, cos_outer, cos_inner],
                        };
                        if l.cast_shadow && casters.len() < MAX_SHADOW_RECORDS {
                            casters.push(Caster::Spot {
                                pos: p,
                                dir,
                                angle: l.angle,
                                settings: l.shadow,
                            });
                            frame_u.spot_lights[n_spot].color[3] = casters.len() as f32;
                        }
                        n_spot += 1;
                    }
                    Light::Hemisphere(l) => {
                        if n_hemi >= MAX_HEMI_LIGHTS {
                            return;
                        }
                        let up = transform_direction(&obj.matrix_world, Vector3::UP).normalize();
                        frame_u.hemi_lights[n_hemi] = HemiLightGpu {
                            sky_color: [
                                l.sky_color.r * l.intensity,
                                l.sky_color.g * l.intensity,
                                l.sky_color.b * l.intensity,
                                0.0,
                            ],
                            ground_color: [
                                l.ground_color.r * l.intensity,
                                l.ground_color.g * l.intensity,
                                l.ground_color.b * l.intensity,
                                0.0,
                            ],
                            direction: [up.x, up.y, up.z, 0.0],
                        };
                        n_hemi += 1;
                    }
                    Light::RectArea(l) => {
                        if n_rect >= MAX_RECT_LIGHTS {
                            return;
                        }
                        // three.js orients a RectAreaLight by its matrixWorld:
                        // width runs along local +X, height along local +Y, and
                        // the emitting face looks down local -Z. Store half
                        // extents so the shader can form corners by ± directly.
                        // Through the world matrix's rotation alone, as
                        // three.js's `extractRotation`: a scaled light keeps
                        // its `width` and `height`.
                        let pos = obj.world_position();
                        let unit = |v: Vector3| {
                            let d = transform_direction(&obj.matrix_world, v);
                            let len = (d.x * d.x + d.y * d.y + d.z * d.z).sqrt().max(1e-12);
                            Vector3::new(d.x / len, d.y / len, d.z / len)
                        };
                        let (ax, ay) = (
                            unit(Vector3::new(1.0, 0.0, 0.0)),
                            unit(Vector3::new(0.0, 1.0, 0.0)),
                        );
                        let right = Vector3::new(
                            ax.x * l.width * 0.5,
                            ax.y * l.width * 0.5,
                            ax.z * l.width * 0.5,
                        );
                        let up = Vector3::new(
                            ay.x * l.height * 0.5,
                            ay.y * l.height * 0.5,
                            ay.z * l.height * 0.5,
                        );
                        let c = l.color;
                        frame_u.rect_lights[n_rect] = RectLightGpu {
                            position: [pos.x, pos.y, pos.z, 0.0],
                            color: [c.r * l.intensity, c.g * l.intensity, c.b * l.intensity, 0.0],
                            right: [right.x, right.y, right.z, 0.0],
                            up: [up.x, up.y, up.z, 0.0],
                        };
                        n_rect += 1;
                    }
                },
                ObjectKind::Mesh(mesh) => {
                    self.ensure_geometry(&mesh.geometry);
                    // Morph targets: the per-draw uniforms and the geometry's
                    // morph texture, shared by every group of the mesh.
                    let morph = crate::core::morph_uniforms(&mesh.geometry, &mesh.morph_influences)
                        .and_then(|u| Some((u, self.ensure_morph(&mesh.geometry)?)));
                    let full = mesh.geometry.draw_count() as u32;
                    // One draw per three.js render item: the whole draw range
                    // with one material, or each group with its own — opaque
                    // and transparent sorted per group, as three.js's render
                    // lists are.
                    for item in mesh.draws() {
                        let material = item.material;
                        let range =
                            (item.start != 0 || item.end != full).then_some((item.start, item.end));
                        if material.wireframe() {
                            match range {
                                None => self.ensure_wire_indices(&mesh.geometry),
                                Some((a, b)) => self.ensure_wire_span(&mesh.geometry, a, b),
                            }
                        }
                        let with_span = |mut d: DrawMesh| -> DrawMesh {
                            d.range = range;
                            if let Some(((header, weights), tex)) = morph {
                                d.morph = header;
                                d.morph_weights = weights;
                                d.morph_tex = tex;
                            }
                            d
                        };
                        let mut d =
                            with_span(make_draw(&mesh.geometry, material, obj, Topology::Triangle));
                        // Custom-shader material: route to a per-source pipeline and
                        // stash its user data for the @group(4) bind group.
                        if let Material::Shader(sm) = &**material {
                            // Morph targets reach a shader material as three.js's
                            // `#include <morphtarget_vertex>` does: the default
                            // vertex stage (`standard_vertex_morphed`) applies
                            // them, and a custom one calls `morph_position` /
                            // `morph_normal` / `morph_color` itself. Its group 2
                            // then carries the geometry's morph texture.
                            let hash = custom_shader_hash(&sm.fragment, sm.vertex.as_deref())
                                ^ if sm.alpha_to_coverage {
                                    COVERAGE_CUSTOM_SALT
                                } else {
                                    0
                                };
                            let transparent = sm.transparent || sm.opacity < 1.0;
                            d.topology =
                                Topology::Custom(hash, transparent, sm.screen_space, sm.side);
                            d.shader_idx = shader_data.len();
                            let pad = |s: &Vec<f32>| {
                                if s.is_empty() {
                                    vec![0.0f32]
                                } else {
                                    s.clone()
                                }
                            };
                            shader_data.push(ShaderDrawData {
                                fragment: sm.fragment.clone(),
                                vertex: sm.vertex.clone(),
                                data: sm.data_slots(),
                                storage: [pad(&sm.storage0), pad(&sm.storage1), pad(&sm.storage2)],
                                textures: sm.textures.clone(),
                                maps: sm.maps.clone(),
                                volumes: sm.volumes.clone(),
                                instanced: false,
                                custom_buf: 0,
                                alpha_to_coverage: sm.alpha_to_coverage,
                            });
                        }
                        // Route transparent PhysicalMaterials to single-layer glass
                        // (depth prepass + color) or weighted-blended OIT.
                        if d.topology == Topology::TriangleAlpha {
                            // Where OIT cannot run (WebGL2 blends every attachment
                            // alike), its glass draws as `Glass` does — tinted,
                            // lit and shadowed, one layer deep — rather than not
                            // at all.
                            let mode = match material.transparency_mode() {
                                crate::materials::TransparencyMode::Oit if !oit_available => {
                                    crate::materials::TransparencyMode::Glass
                                }
                                m => m,
                            };
                            match mode {
                                crate::materials::TransparencyMode::Glass => {
                                    let mut prepass = with_span(make_draw(
                                        &mesh.geometry,
                                        material,
                                        obj,
                                        Topology::GlassDepth,
                                    ));
                                    prepass.state = crate::materials::RenderState::DEFAULT;
                                    draws.push(prepass);
                                    // The tint half, then (as `d`) the light half.
                                    let mut tint = d.clone();
                                    tint.topology = Topology::GlassTint;
                                    tint.state = crate::materials::RenderState::DEFAULT;
                                    tint.cast_shadow = false;
                                    draws.push(tint);
                                    d.topology = Topology::GlassColor;
                                }
                                crate::materials::TransparencyMode::Oit => {
                                    d.topology = Topology::Oit
                                }
                                crate::materials::TransparencyMode::Refract => {
                                    d.topology = Topology::Refract
                                }
                                crate::materials::TransparencyMode::Blend => {}
                            }
                            // These modes own their depth state (a prepass, an
                            // accumulation target); the material's does not apply.
                            if d.topology != Topology::TriangleAlpha {
                                d.state = crate::materials::RenderState::DEFAULT;
                            }
                        }
                        draws.push(d);
                    }
                }
                ObjectKind::LineSegments(ls) => {
                    self.ensure_geometry(&ls.geometry);
                    // three.js applies the draw range to lines too.
                    let full = ls.geometry.draw_count() as u32;
                    let Some((start, end)) = ls.geometry.draw_span(None) else {
                        return;
                    };
                    // three.js `linewidth`, which WebGL ignores: wider than
                    // a pixel, a capsule a segment (`vs_wide_lines`), its
                    // width in device pixels as a point's size is
                    let width = match &*ls.material {
                        Material::Line(m) => m.line_width,
                        _ => 1.0,
                    };
                    // Thinner than a pixel and blended, a capsule too: each
                    // pixel takes the share of it the line covers, where a
                    // line primitive lights whole pixels or none. Opaque, a
                    // capsule could only cut at half, so it stays a line.
                    let blended = ls.material.transparent();
                    if width > 1.0 || (blended && width > 0.0 && width < 1.0) {
                        let key = geom_cache_key(&ls.geometry);
                        if let Some(e) = self.geom_cache.get_mut(&key) {
                            e.mesh.ensure_segments(&self.device, &ls.geometry);
                        }
                        let topology = if blended {
                            Topology::WideLineAlpha
                        } else {
                            Topology::WideLine
                        };
                        let mut d = make_draw(&ls.geometry, &ls.material, obj, topology);
                        d.range = (start != 0 || end != full).then_some((start, end));
                        d.shininess_or_near = width * self.pixel_ratio();
                        d.roughness = if blended { 1.0 } else { 0.0 };
                        d.cast_shadow = false;
                        draws.push(d);
                        return;
                    }
                    let mut d = make_draw(&ls.geometry, &ls.material, obj, Topology::Line);
                    d.range = (start != 0 || end != full).then_some((start, end));
                    draws.push(d);
                }
                ObjectKind::Points(p) => {
                    // three.js `Points`: one draw, one instance per point,
                    // each expanded to a `size`-pixel quad by `vs_points`
                    // (WebGPU has no point size). `params.x` carries the size
                    // in device pixels (× the pixel ratio, as three.js's
                    // `size` uniform), `params.w` whether it attenuates.
                    self.ensure_geometry(&p.geometry);
                    let full = p.geometry.draw_count() as u32;
                    let Some((start, end)) = p.geometry.draw_span(None) else {
                        return;
                    };
                    let topology = if p.material.transparent() {
                        Topology::PointAlpha
                    } else {
                        Topology::Point
                    };
                    let mut d = make_draw(&p.geometry, &p.material, obj, topology);
                    d.range = (start != 0 || end != full).then_some((start, end));
                    d.shininess_or_near = p.material.point_size() * self.pixel_ratio();
                    d.metalness = p.material.point_size_attenuation() as u32 as f32;
                    d.cast_shadow = false;
                    draws.push(d);
                }
                ObjectKind::Sprite(sprite) => {
                    // Inline construction — the sprite quad is built-in, so we
                    // can't reuse make_draw (which keys off an Arc<BufferGeometry>).
                    let mat = &*sprite.material;
                    // three.js `sprite_vert` uniforms, in slots the sprite
                    // shader has to itself (see `vs_sprite`): the centre as
                    // an offset from the middle, so that zero — what point
                    // draws leave there — is three.js's default (0.5, 0.5),
                    // and the rotation; `params13.x` set means size
                    // attenuation off.
                    let (rotation, size_attenuation) = match mat {
                        Material::Sprite(m) => (m.rotation, m.size_attenuation),
                        _ => (0.0, true),
                    };
                    let sprite_params =
                        [sprite.center.x - 0.5, sprite.center.y - 0.5, 0.0, rotation];
                    let nm3 = Matrix3::normal_matrix(&obj.matrix_world);
                    let c = mat.color();
                    let sp = obj.world_position();
                    let svz = view_m.elements[2] * sp.x
                        + view_m.elements[6] * sp.y
                        + view_m.elements[10] * sp.z
                        + view_m.elements[14];
                    draws.push(DrawMesh {
                        key: std::ptr::null::<BufferGeometry>(),
                        topology: Topology::Sprite,
                        view_z: svz,
                        render_order: obj.render_order,
                        clip_plane: clip_plane_of(obj),
                        in_frustum: true,
                        transmissive: false,
                        shadow_transmits: false,
                        cast_shadow: false,
                        refract_capture: true,
                        alpha_test: mat.alpha_test(),
                        side: mat.side(),
                        shadow_side: mat.side(),
                        model: obj.matrix_world.elements,
                        normal_matrix: mat3_to_mat4_array(&nm3),
                        color: [c.r, c.g, c.b, mat.opacity()],
                        emissive: [0.0; 4],
                        specular: [0.0; 4],
                        shininess_or_near: 0.0,
                        opacity_or_far: mat.opacity(),
                        roughness: 1.0,
                        metalness: 0.0,
                        ao_intensity: 1.0,
                        normal_scale: [1.0, 1.0],
                        physical: [0.0; 4],
                        physical2: [0.0; 4],
                        physical3: [0.0; 4],
                        physical4: [0.0; 4],
                        physical5: [0.0; 4],
                        displacement: sprite_params,
                        uv_transform: [0.0, 0.0, 1.0, 1.0],
                        atmos: [0.0; 4],
                        twilight: [0.0; 4],
                        eclipse: [0.0; 4],
                        scatter: [if size_attenuation { 0.0 } else { 1.0 }, 0.0, 0.0, 0.0],
                        aurora: [0.0; 4],
                        shader_flags: (if mat.tone_mapped() {
                            0
                        } else {
                            FLAG_NO_TONE_MAP
                        }) | if mat.transparent() {
                            0
                        } else {
                            FLAG_OPAQUE_OUT
                        },
                        kind: mat.kind(),
                        slots: mat.texture_slots(),
                        map_variant: None,
                        map_uniforms: mat.map_uniforms(),
                        env_group: mat.pbr_env().map(material_env_key),
                        instance_buf_idx: usize::MAX,
                        instance_stamp: 0,
                        instance_count: 1,
                        skin_idx: usize::MAX,
                        shader_idx: usize::MAX,
                        range: None,
                        morph: [0.0; 4],
                        morph_weights: [0.0; crate::core::MAX_MORPH_TARGETS],
                        morph_tex: (0, 0),
                        polygon_offset: None,
                        state: mat.render_state(),
                    });
                }
                ObjectKind::InstancedMesh(im) => {
                    self.ensure_geometry(&im.geometry);
                    if im.transforms.is_empty() {
                        return;
                    }
                    // Its draws: the draw range with one material, or every
                    // group with its own from the material array — each one
                    // instanced, as three.js's render list holds an
                    // InstancedMesh with a material array.
                    let items: Vec<DrawItem<std::sync::Arc<Material>>> = {
                        let full = im.geometry.draw_count() as u32;
                        im.draws()
                            .into_iter()
                            .map(|i| {
                                (
                                    i.material.clone(),
                                    (i.start != 0 || i.end != full).then_some((i.start, i.end)),
                                )
                            })
                            .collect()
                    };
                    if items.is_empty() {
                        return;
                    }
                    for (material, range) in &items {
                        if material.wireframe() {
                            match range {
                                None => self.ensure_wire_indices(&im.geometry),
                                Some((a, b)) => self.ensure_wire_span(&im.geometry, *a, *b),
                            }
                        }
                    }
                    // Morph targets: one row per instance when the mesh has a
                    // morph texture (`set_morph_at`), else the shared
                    // influences as for a mesh.
                    let row_len = im.morph_row_len();
                    let per_instance = !im.morph_texture.is_empty()
                        && im.morph_texture.len() == row_len * im.transforms.len();
                    let morph = if per_instance {
                        crate::core::morph_header(&im.geometry).and_then(|mut h| {
                            h[3] = (h[3] as u32 | 16) as f32;
                            let tex = self.ensure_instance_morph(
                                im as *const crate::core::InstancedMesh as usize,
                                &im.geometry,
                                &im.morph_texture,
                                im.transforms.len() as u32,
                                row_len as u32,
                            )?;
                            Some(((h, [0.0; crate::core::MAX_MORPH_TARGETS]), tex))
                        })
                    } else {
                        crate::core::morph_uniforms(&im.geometry, &im.morph_influences)
                            .and_then(|u| Some((u, self.ensure_morph(&im.geometry)?)))
                    };
                    // Twenty floats an instance: the matrix, then the tint.
                    // Instances past the end of `colors` are white, so a caller
                    // that never sets one pays four floats and nothing else.
                    // A mesh with an `instance_version` that has not moved
                    // keeps last frame's buffer without being read at all.
                    let version = im.instance_version.map(|v| (v, im.transforms.len()));
                    let buf = match self.cached_instances(obj_id, 0, version) {
                        Some(b) => b,
                        None => {
                            let mut matrices: Vec<f32> =
                                Vec::with_capacity(im.transforms.len() * 20);
                            for (i, t) in im.transforms.iter().enumerate() {
                                matrices.extend_from_slice(&t.elements);
                                let c = im
                                    .colors
                                    .get(i)
                                    .copied()
                                    .unwrap_or(crate::math::Color::WHITE);
                                matrices.extend_from_slice(&[c.r, c.g, c.b, 1.0]);
                            }
                            self.instance_buffer(obj_id, 0, matrices, version)
                        }
                    };
                    let idx = instance_buffers.len();
                    instance_buffers.push(buf);
                    // A `ShaderMaterial` reads the two custom columns too:
                    // eight floats an instance, zero-padded, in slot 2.
                    let custom_idx = items
                        .iter()
                        .any(|(m, _)| matches!(&**m, Material::Shader(_)))
                        .then(|| {
                            let b = match self.cached_instances(obj_id, 1, version) {
                                Some(b) => b,
                                None => {
                                    let mut data: Vec<f32> =
                                        Vec::with_capacity(im.transforms.len() * 8);
                                    for i in 0..im.transforms.len() {
                                        data.extend_from_slice(&im.custom_row(0, i));
                                        data.extend_from_slice(&im.custom_row(1, i));
                                    }
                                    self.instance_buffer(obj_id, 1, data, version)
                                }
                            };
                            instance_buffers.push(b);
                            instance_buffers.len() - 1
                        });
                    for (material, range) in &items {
                        let mut d = make_draw(&im.geometry, material, obj, Topology::Instanced);
                        d.range = *range;
                        d.instance_buf_idx = idx;
                        d.instance_stamp = self.instance_stamp(obj_id, 0)
                            ^ self.instance_stamp(obj_id, 1).rotate_left(32);
                        d.instance_count = im.transforms.len() as u32;
                        if let Some(((header, weights), tex)) = morph {
                            d.morph = header;
                            d.morph_weights = weights;
                            d.morph_tex = tex;
                        }
                        // Custom-shader material: its own pipeline, on the
                        // instanced layout — salted so it never shares a cache
                        // entry with the same material on a plain mesh.
                        if let (Material::Shader(sm), Some(custom_buf)) = (&**material, custom_idx)
                        {
                            let hash = custom_shader_hash(&sm.fragment, sm.vertex.as_deref())
                                ^ INSTANCED_CUSTOM_SALT
                                ^ if sm.alpha_to_coverage {
                                    COVERAGE_CUSTOM_SALT
                                } else {
                                    0
                                };
                            let transparent = sm.transparent || sm.opacity < 1.0;
                            d.topology =
                                Topology::Custom(hash, transparent, sm.screen_space, sm.side);
                            d.shader_idx = shader_data.len();
                            let pad = |s: &Vec<f32>| {
                                if s.is_empty() {
                                    vec![0.0f32]
                                } else {
                                    s.clone()
                                }
                            };
                            shader_data.push(ShaderDrawData {
                                fragment: sm.fragment.clone(),
                                vertex: sm.vertex.clone(),
                                data: sm.data_slots(),
                                storage: [pad(&sm.storage0), pad(&sm.storage1), pad(&sm.storage2)],
                                textures: sm.textures.clone(),
                                maps: sm.maps.clone(),
                                volumes: sm.volumes.clone(),
                                instanced: true,
                                custom_buf,
                                alpha_to_coverage: sm.alpha_to_coverage,
                            });
                        }
                        draws.push(d);
                    }
                }
                ObjectKind::SkinnedMesh(sm) => {
                    self.ensure_geometry(&sm.geometry);
                    let morph = crate::core::morph_uniforms(&sm.geometry, &sm.morph_influences)
                        .and_then(|u| Some((u, self.ensure_morph(&sm.geometry)?)));
                    let full = sm.geometry.draw_count() as u32;
                    let items: Vec<DrawItem<&std::sync::Arc<Material>>> = sm
                        .draws()
                        .into_iter()
                        .map(|i| {
                            (
                                i.material,
                                (i.start != 0 || i.end != full).then_some((i.start, i.end)),
                            )
                        })
                        .collect();
                    for &(material, range) in &items {
                        if material.wireframe() {
                            match range {
                                None => self.ensure_wire_indices(&sm.geometry),
                                Some((a, b)) => self.ensure_wire_span(&sm.geometry, a, b),
                            }
                        }
                    }
                    let with_span = |mut d: DrawMesh, range: Option<(u32, u32)>| -> DrawMesh {
                        d.range = range;
                        if let Some(((header, weights), tex)) = morph {
                            d.morph = header;
                            d.morph_weights = weights;
                            d.morph_tex = tex;
                        }
                        d
                    };
                    let key = geom_cache_key(&sm.geometry);
                    // Upload joints+weights to slot-1 vertex buffer (cached per-geometry).
                    if !self.skin_attr_cache.contains_key(&key) {
                        let joints = sm.geometry.get_attribute("joint");
                        let weights = sm.geometry.get_attribute("weight");
                        let (Some(j), Some(w)) = (joints, weights) else {
                            // Geometry isn't skin-equipped; draw at rest
                            // (morphed).
                            for &(material, range) in &items {
                                draws.push(with_span(
                                    make_draw(&sm.geometry, material, obj, Topology::Triangle),
                                    range,
                                ));
                            }
                            return;
                        };
                        let count = j.count();
                        let mut interleaved: Vec<f32> = Vec::with_capacity(count * 8);
                        for i in 0..count {
                            interleaved.extend_from_slice(&j.array[i * 4..i * 4 + 4]);
                            interleaved.extend_from_slice(&w.array[i * 4..i * 4 + 4]);
                        }
                        let buf = wgpu::util::DeviceExt::create_buffer_init(
                            &*self.device,
                            &wgpu::util::BufferInitDescriptor {
                                label: Some("threers skin attrs"),
                                contents: bytemuck::cast_slice(&interleaved),
                                usage: wgpu::BufferUsages::VERTEX,
                            },
                        );
                        self.skin_attr_cache.insert(key, buf);
                    }
                    // Upload bone matrices (per-frame, per-skinned-mesh).
                    //
                    // Always exactly `MAX_BONES` of them, padded with
                    // identities: the shader's array is fixed because WebGL2
                    // cannot do a runtime-sized one, and a uniform binding has
                    // to match the size the layout declares.
                    const MAX_BONES: usize = 128;
                    let mut bone_data: Vec<f32> = Vec::with_capacity(MAX_BONES * 16);
                    for m in sm.skeleton.bone_matrices.iter().take(MAX_BONES) {
                        bone_data.extend_from_slice(&m.elements);
                    }
                    let identity = crate::math::Matrix4::identity();
                    while bone_data.len() < MAX_BONES * 16 {
                        bone_data.extend_from_slice(&identity.elements);
                    }
                    let bone_buf = wgpu::util::DeviceExt::create_buffer_init(
                        &*self.device,
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("threers bone matrices"),
                            contents: bytemuck::cast_slice(&bone_data),
                            usage: wgpu::BufferUsages::UNIFORM,
                        },
                    );
                    // Defer creating the combined mesh+skin bind group until
                    // after per_mesh is populated below — we need the per-mesh
                    // uniform buffer at binding 0. One per draw: each group
                    // has its own mesh uniform, and the bones are shared.
                    let bone_hash = hash_words(bone_data.iter().map(|v| v.to_bits() as u64));
                    for &(material, range) in &items {
                        let idx = skin_data.len();
                        skin_data.push(SkinData {
                            bone_buf: bone_buf.clone(),
                            bg: None,
                            bone_hash,
                        });
                        let mut d = with_span(
                            make_draw(&sm.geometry, material, obj, Topology::Skinned),
                            range,
                        );
                        d.skin_idx = idx;
                        draws.push(d);
                    }
                }
                ObjectKind::Group => {}
                #[cfg(feature = "audio")]
                ObjectKind::Audio(_) => {}
            }
        });

        // Sort: opaque first (stable order), then transparents back-to-front so
        // alpha blending composites correctly. Sprites are already alpha-blended
        // and we treat them as transparent for ordering purposes.
        // Draw order class: opaque(0) → glass depth prepass(1) → glass color(2) →
        // ordinary alpha & OIT(3). Glass must run after opaque (so its prepass
        // sees opaque depth) and before ordinary alpha (which blends over it).
        fn ord_class(t: Topology) -> u8 {
            match t {
                Topology::GlassDepth => 1,
                Topology::GlassTint | Topology::GlassColor => 2,
                Topology::TriangleAlpha
                | Topology::PointAlpha
                | Topology::WideLineAlpha
                | Topology::Sprite
                | Topology::Oit
                | Topology::Refract
                | Topology::InstancedAlpha
                | Topology::SkinnedAlpha => 3,
                Topology::Custom(_, transparent, ..) if transparent => 3,
                _ => 0,
            }
        }
        // three.js renders the transmissive list between the opaque and the
        // transparent ones, so it can read the opaque scene behind it.
        let class = |d: &DrawMesh| -> u8 {
            match ord_class(d.topology) {
                0 if d.transmissive => 1,
                0 => 0,
                c => c + 1,
            }
        };
        // three.js draws a blended DoubleSide mesh twice — its back faces,
        // then its front faces (`WebGLRenderer.renderObject`, with the default
        // `forceSinglePass: false`) — so a translucent surface's far side always
        // lies under its near side. In one unculled pass the triangles blend
        // in index order instead, and a closed translucent shell shows its own
        // back faces over its front ones as triangle-shaped shards. The sort
        // below is stable and the pair shares every key, so it stays adjacent
        // and in this order. The back half casts no shadow: the front half's
        // record already covers a double-sided caster. Meshes, instanced and
        // skinned alike; a custom shader's pipeline is built apart from the cull
        // variants and still draws in one pass.
        let mut two_pass = Vec::with_capacity(draws.len());
        for d in draws.drain(..) {
            let blended_faces = matches!(
                d.topology,
                Topology::TriangleAlpha | Topology::InstancedAlpha | Topology::SkinnedAlpha
            );
            if blended_faces && d.side == 2 && !d.transmissive {
                let mut back = d.clone();
                back.side = 1;
                back.cast_shadow = false;
                let mut front = d;
                front.side = 0;
                two_pass.push(back);
                two_pass.push(front);
            } else {
                two_pass.push(d);
            }
        }
        draws = two_pass;
        // Opaque, then transmissive, then the translucent family (glass
        // prepass, glass colour, ordinary alpha). Within the family
        // `renderOrder` comes first, as it does in three.js's transparent list,
        // and only then the family's own order — so with equal render orders
        // every glass prepass still precedes every glass colour pass and both
        // precede ordinary alpha, while a glass shell given a higher order is
        // drawn after the translucent things inside it, whose colour it then
        // blends over instead of hiding them behind its prepass depth.
        let tier = |d: &DrawMesh| -> u8 { class(d).min(2) };
        draws.sort_by(|a, b| {
            tier(a)
                .cmp(&tier(b))
                .then(a.render_order.cmp(&b.render_order))
                .then(class(a).cmp(&class(b)))
                .then_with(|| {
                    // For ordinary transparents we want farther-from-camera first.
                    // view_z is negative in front of the camera; smaller = farther.
                    if ord_class(a.topology) == 3 && ord_class(b.topology) == 3 {
                        a.view_z
                            .partial_cmp(&b.view_z)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    } else {
                        std::cmp::Ordering::Equal
                    }
                })
        });

        // --- Frustum culling.
        //
        // Nothing here used to test whether a draw was on screen: every mesh in
        // the scene went through the vertex shader and got rasterised every
        // frame, however far outside the view it was. `Frustum` has been in
        // `src/math` the whole time, correct for this crate's 0..1 clip depth
        // and exposed to wasm, and the renderer simply never called it.
        //
        // Culled draws stay in `draws`. The shadow passes walk the same list
        // and a caster behind the camera can still throw a shadow right across
        // the view, so only the colour passes read this flag. The indices also
        // key `per_mesh` and the bind-group caches, and removing entries would
        // shift all of them.
        //
        // The jitter-free view-projection is deliberate: culling off the
        // jittered matrix would let an object on the boundary flicker in and
        // out from frame to frame.
        if self.frustum_culling {
            let frustum = Frustum::from_projection_matrix(&unjittered);
            for d in draws.iter_mut() {
                // Only geometry whose object-space bounds really do bound what
                // gets rasterised. A skinned mesh moves outside its bind pose,
                // an instanced draw covers ground the source geometry does not,
                // a custom vertex shader may put the vertices anywhere, and the
                // sky is deliberately pinned to the far plane.
                let plain = matches!(
                    d.topology,
                    Topology::Triangle
                        | Topology::TriangleAlpha
                        | Topology::TriangleNoCull
                        | Topology::TriangleWire
                        | Topology::Line
                        | Topology::WideLine
                        | Topology::WideLineAlpha
                        | Topology::GlassDepth
                        | Topology::GlassTint
                        | Topology::GlassColor
                        | Topology::Oit
                        | Topology::Refract
                ) && d.instance_count == 1
                    && d.skin_idx == usize::MAX
                    && d.shader_idx == usize::MAX
                    // A morphed mesh can leave its rest-pose bounds.
                    && d.morph[0] == 0.0;
                if !plain {
                    continue;
                }
                let Some(sphere) = self.geom_cache.get(&d.key).and_then(|g| g.sphere) else {
                    continue;
                };
                let world = sphere.apply_matrix4(&Matrix4 { elements: d.model });
                d.in_frustum = frustum.intersects_sphere(&world);
            }
        }
        // The transmission capture passes (`capture_transmission`): the first
        // draws everything but the transmissive objects, the second only the
        // back faces of the double-sided ones, as three.js's
        // `renderTransmissionPass` does. Through `in_frustum`, so the shadow
        // passes still see every caster.
        if let Some(back) = self.capturing_transmission {
            for d in draws.iter_mut() {
                let keep = if back {
                    d.transmissive && d.side == 2
                } else {
                    !d.transmissive
                };
                if !keep {
                    d.in_frustum = false;
                } else if back {
                    d.side = 1;
                }
            }
        }
        self.cull_stats = (
            draws.iter().filter(|d| d.in_frustum).count(),
            draws.iter().filter(|d| !d.in_frustum).count(),
        );

        frame_u.ambient = [ambient[0], ambient[1], ambient[2], 0.0];
        frame_u.light_counts = [n_dir as u32, n_point as u32, n_spot as u32, n_hemi as u32];
        // w: the bound capture's mip count — the stand-in's one while a capture
        // pass draws, the capture's after it.
        let capture_levels = match (self.capturing_transmission, &self.transmission) {
            (Some(false), _) | (None, None) => 1,
            (Some(true), Some(c)) | (None, Some(c)) => c.levels,
            (Some(true), None) => 1,
        };
        frame_u.light_counts2 = [
            n_rect as u32,
            self.specular_antialiasing as u32,
            self.reflective_transparency as u32,
            capture_levels,
        ];

        // Each shadow-casting light's record — layers, matrices, biases and
        // filter — and the passes that draw its maps. A directional light's
        // single map fits a fixed cube on the world origin
        // (`ShadowSettings::camera_size`), so it depends on lights and
        // casters, never on the camera; cascades follow the camera.
        //
        // The maps are allocated before any light exists to ask for a size,
        // so the first frame that names one reallocates. `map_size` used to be
        // read and discarded, which is a worse failure than not offering it:
        // the caller sets 4096, sees no error, and gets 1024.
        let layers_of = |kind: u8| -> usize {
            casters
                .iter()
                .filter(|c| {
                    matches!(
                        (kind, c),
                        (0, Caster::Dir { .. })
                            | (1, Caster::Spot { .. })
                            | (2, Caster::Point { .. })
                    )
                })
                .map(Caster::layers)
                .sum()
        };
        let (dir_layers, spot_layers, point_layers) = (layers_of(0), layers_of(1), layers_of(2));
        self.ensure_shadow_arrays(
            casters
                .iter()
                .filter_map(|c| match c {
                    Caster::Dir { settings, .. } => Some(settings.map_size),
                    _ => None,
                })
                .max()
                .unwrap_or(0),
            dir_layers,
            spot_layers,
            point_layers / 6,
        );
        let mut shadow_passes: Vec<ShadowPass> = Vec::new();
        let mut next_layer = [0usize; 3];
        for (r, caster) in casters.iter().enumerate() {
            let settings = caster.settings();
            let filter = settings.filter.unwrap_or(self.shadow_type).code();
            let bias = settings.bias.max(0.0);
            let normal_bias = settings.normal_bias.max(0.0);
            let mut rec = ShadowRecordGpu::zeroed();
            match *caster {
                Caster::Dir { dir, settings } => {
                    let size = self.dir_shadows.size;
                    let texel = 1.0 / size as f32;
                    let layer = next_layer[0];
                    let cascades = caster.layers();
                    next_layer[0] += cascades;
                    // The light's size as a penumbra per unit of normalised
                    // depth, in map uv: an angle times the depth range over
                    // the map's width. Cascades keep that ratio fixed (see
                    // `cascade_frusta`), so one number serves them all.
                    let angle = if settings.light_size > 0.0 {
                        settings.light_size
                    } else {
                        0.02
                    };
                    let (depth_range, width) = if cascades > 1 {
                        (12.0, 2.0)
                    } else {
                        (
                            settings.camera_far - settings.camera_near,
                            2.0 * settings.camera_size,
                        )
                    };
                    let scale = angle * depth_range / width.max(1e-6);
                    rec.info = [0, layer as u32, cascades as u32, filter];
                    rec.params = [bias, normal_bias, settings.radius.max(0.0) * texel, texel];
                    rec.pcss = [(scale * 0.25).clamp(2.0 * texel, 0.1), 0.0, 0.0, scale];
                    if cascades > 1 {
                        for (c, (vp, far)) in cascade_frusta(
                            &view_m,
                            &camera.projection_matrix(),
                            dir,
                            &settings,
                            size,
                        )
                        .into_iter()
                        .enumerate()
                        {
                            rec.vp[c] = vp;
                            rec.extra[c] = far;
                            shadow_passes.push(ShadowPass {
                                kind: ShadowKind::Dir,
                                layer: (layer + c) as u32,
                                vp,
                                pos: [0.0; 4],
                            });
                        }
                    } else {
                        let s = settings.camera_size;
                        let light_eye = -dir.normalize() * (s * 1.5);
                        let light_view = Matrix4::look_at(light_eye, Vector3::ZERO, Vector3::UP);
                        let light_proj = Matrix4::orthographic(
                            -s,
                            s,
                            s,
                            -s,
                            settings.camera_near,
                            settings.camera_far,
                        );
                        let vp = light_proj.multiply(&light_view).elements;
                        rec.vp[0] = vp;
                        rec.extra = [f32::MAX; 4];
                        shadow_passes.push(ShadowPass {
                            kind: ShadowKind::Dir,
                            layer: layer as u32,
                            vp,
                            pos: [0.0; 4],
                        });
                    }
                }
                Caster::Spot {
                    pos,
                    dir,
                    angle,
                    settings,
                } => {
                    let texel = 1.0 / self.spot_shadows.size as f32;
                    let layer = next_layer[1];
                    next_layer[1] += 1;
                    let target = pos + dir.normalize();
                    let light_view = Matrix4::look_at(pos, target, Vector3::UP);
                    // Perspective frustum: fov = 2 * outer_angle to cover the cone.
                    let fov = (angle * 2.0).clamp(0.05, std::f32::consts::PI - 0.05);
                    let light_proj =
                        Matrix4::perspective(fov, 1.0, settings.camera_near, settings.camera_far);
                    let vp = light_proj.multiply(&light_view).elements;
                    // The light's width over the frustum's width at unit
                    // distance: the penumbra in uv per unit of (receiver −
                    // blocker) / (blocker · receiver).
                    let size = if settings.light_size > 0.0 {
                        settings.light_size
                    } else {
                        0.5
                    };
                    let scale = size / (2.0 * (fov * 0.5).tan());
                    rec.info = [1, layer as u32, 1, filter];
                    rec.params = [bias, normal_bias, settings.radius.max(0.0) * texel, texel];
                    rec.pcss = [
                        (scale * 0.05).clamp(2.0 * texel, 0.1),
                        settings.camera_near,
                        settings.camera_far,
                        scale,
                    ];
                    rec.vp[0] = vp;
                    shadow_passes.push(ShadowPass {
                        kind: ShadowKind::Spot,
                        layer: layer as u32,
                        vp,
                        pos: [0.0; 4],
                    });
                }
                Caster::Point {
                    pos,
                    radius,
                    settings,
                } => {
                    // The filter disc lies across the direction to the light,
                    // a face texel (`2 / size` of a unit cube face) per unit
                    // of radius.
                    let texel = 2.0 / POINT_SHADOW_SIZE as f32;
                    let layer = next_layer[2];
                    next_layer[2] += 6;
                    let at = [pos.x, pos.y, pos.z, radius];
                    let size = if settings.light_size > 0.0 {
                        settings.light_size
                    } else {
                        0.5
                    };
                    rec.info = [2, layer as u32, 1, filter];
                    rec.params = [bias, normal_bias, settings.radius.max(0.0) * texel, texel];
                    rec.extra = at;
                    rec.pcss = [(size * 0.05).clamp(2.0 * texel, 0.2), 0.0, 0.0, size];
                    for face in 0..6 {
                        shadow_passes.push(ShadowPass {
                            kind: ShadowKind::Point,
                            layer: (layer + face) as u32,
                            vp: point_face_vp(pos, radius, face),
                            pos: at,
                        });
                    }
                }
            }
            frame_u.shadows[r] = rec;
        }
        frame_u.shadow_info = [casters.len() as u32, 0, 0, 0];

        // Sync environment cubemap. Upload on first sight, rebuild bind group on change.
        // A cube texture is prefiltered first, as three.js's `WebGLCubeUVMaps`
        // does for every Standard/Physical environment.
        let scene_env_cube = scene.environment.as_ref().map(|e| self.prefiltered(e));
        let env_key = scene_env_cube.as_ref().map(cube_cache_key);
        let env_rt_id = scene.environment_cube_rt;
        if self.env_dirty || env_key != self.env_cached_key || env_rt_id != self.env_cached_rt_id {
            let mut env_map_params = [0.0_f32; 4];
            if let Some(env) = &scene_env_cube {
                let k = cube_cache_key(env);
                if !self.cube_cache.contains_key(&k) {
                    let gc = GpuCubeTexture::upload(&self.device, &self.queue, env);
                    self.cube_cache.insert(k, gc);
                }
                if let Some(atlas) = &env.cube_uv_atlas {
                    env_map_params = [
                        atlas.texel_width,
                        atlas.texel_height,
                        atlas.lod_max as f32,
                        1.0,
                    ];
                }
            }
            frame_u.env_map_params = env_map_params;
            let cube_uv_view: &wgpu::TextureView = if let Some(env) = &scene_env_cube {
                self.cube_cache
                    .get(&cube_cache_key(env))
                    .and_then(|gc| gc.cube_uv_view.as_ref())
                    .unwrap_or(&self.default_cube_uv_view)
            } else {
                &self.default_cube_uv_view
            };
            let view: &wgpu::TextureView = if let Some(rt_id) = scene.environment_cube_rt {
                self.cube_rt_view_cache
                    .get(&rt_id)
                    .unwrap_or(&self.default_env_cube.view)
            } else if let Some(env) = &scene_env_cube {
                &self.cube_cache[&cube_cache_key(env)].view
            } else {
                &self.default_env_cube.view
            };
            // Whatever `view` just became, record how many mips the shader may
            // sample from it. A cube render target's view is built with
            // `mip_level_count: Some(1)` and the placeholder cube is 1x1, so
            // both expose a single level; only an uploaded environment has a
            // chain worth walking.
            self.env_max_mip = if scene.environment_cube_rt.is_some() {
                0.0
            } else if let Some(env) = &scene_env_cube {
                let levels = self.cube_cache[&cube_cache_key(env)]
                    .texture
                    .mip_level_count();
                levels.saturating_sub(1) as f32
            } else {
                0.0
            };
            self.env_bind_group = env_bind_group!(
                self.device,
                &self.env_bgl,
                "threers env bg",
                view,
                &self.env_sampler,
                &self.dir_shadows.view,
                &self.shadow_sampler,
                &self.spot_shadows.view,
                &self.spot_shadow_sampler,
                &self.point_shadows.view,
                &self.point_shadow_sampler,
                cube_uv_view,
                &self.env_uniform,
                &self.ltc.1,
            );
            if let Some(bgl) = &self.custom_env_bgl {
                // The trimmed custom environment follows the same cube and 2D
                // fallback; its uniform is the shared `env_uniform` buffer.
                self.custom_env_bind_group = Some(custom_env_bind_group!(
                    self.device,
                    bgl,
                    "threers custom env bg",
                    view,
                    &self.env_sampler,
                    cube_uv_view,
                    &self.env_uniform,
                ));
            }
            self.env_dirty = false;
            // Theirs share the shadow-map views just rebound.
            self.material_env_groups.clear();
            self.env_cached_key = env_key;
            self.env_cached_rt_id = env_rt_id;
        }
        // The plain-cube path reads its chain length from here as well, so this
        // has to be set on every frame: the bind-group block above only runs
        // when the environment changes, and `frame_u` starts from `Default`.
        frame_u.env_map_params[2] = self.env_max_mip;
        if let Some(env) = &scene_env_cube
            && let Some(atlas) = &env.cube_uv_atlas
        {
            frame_u.env_map_params = [
                atlas.texel_width,
                atlas.texel_height,
                atlas.lod_max as f32,
                1.0,
            ];
        }
        // The scene's environment as group 3 reads it: three.js's
        // `scene.environmentIntensity` and `environmentRotation`.
        let scene_env = EnvUniforms {
            params: frame_u.env_map_params,
            info: [has_env as u32 as f32, scene.environment_intensity, 0.0, 0.0],
            rot: crate::materials::environment_lookup_rows(
                scene.environment_rotation,
                scene.environment_cube_rt.is_none()
                    && scene_env_cube
                        .as_ref()
                        .is_some_and(|e| e.cube_uv_atlas.is_none()),
            ),
        };
        self.queue
            .write_buffer(&self.env_uniform, 0, bytemuck::bytes_of(&scene_env));

        // -- Ensure every texture referenced in this frame is uploaded. --
        // Every role's map — the list `material_maps` binds from, so a new
        // role cannot be bound without being uploaded.
        for d in &draws {
            for (_, t) in super::super::material_maps::role_textures(&d.slots) {
                self.ensure_slot_uploaded(&t.cloned());
            }
            // A Standard/Physical material's own `envMap` prefiltered, as the
            // scene's is; a Basic/Lambert/Phong one sampled as it is.
            let pbr = d.slots.pbr_env_map.as_ref().map(|c| self.prefiltered(c));
            for cube in [d.slots.env_map.as_ref(), pbr.as_ref()]
                .into_iter()
                .flatten()
            {
                let k = cube_cache_key(cube);
                if !self.cube_cache.contains_key(&k) {
                    let gc = GpuCubeTexture::upload(&self.device, &self.queue, cube);
                    self.cube_cache.insert(k, gc);
                }
            }
        }
        // Group 3 for each Standard/Physical material's own `envMap`.
        for d in &draws {
            let (Some(key), Some(source)) = (d.env_group, d.slots.pbr_env_map.as_ref()) else {
                continue;
            };
            if self.material_env_groups.contains_key(&key) {
                continue;
            }
            let cube = &self.prefiltered(source);
            let gc = &self.cube_cache[&cube_cache_key(cube)];
            let u = EnvUniforms::of(
                cube,
                gc,
                f32::from_bits(key.1),
                crate::math::Euler {
                    x: f32::from_bits(key.2[0]),
                    y: f32::from_bits(key.2[1]),
                    z: f32::from_bits(key.2[2]),
                    order: crate::math::EulerOrder::ALL[key.2[3] as usize],
                },
            );
            let buf = wgpu::util::DeviceExt::create_buffer_init(
                &*self.device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some("threers material env uniforms"),
                    contents: bytemuck::bytes_of(&u),
                    usage: wgpu::BufferUsages::UNIFORM,
                },
            );
            let bg = env_bind_group!(
                self.device,
                &self.env_bgl,
                "threers material env bg",
                &gc.view,
                &self.env_sampler,
                &self.dir_shadows.view,
                &self.shadow_sampler,
                &self.spot_shadows.view,
                &self.spot_shadow_sampler,
                &self.point_shadows.view,
                &self.point_shadow_sampler,
                gc.cube_uv_view
                    .as_ref()
                    .unwrap_or(&self.default_cube_uv_view),
                &buf,
                &self.ltc.1,
            );
            self.material_env_groups.insert(key, (buf, bg));
        }

        // -- Ensure pool capacity: the uniform arena, and a mesh bind group
        // over each draw's slice of it. Grown by half again, so a scene
        // gaining meshes one at a time does not rebuild it every frame. --
        let stride = self.mesh_stride;
        let mesh_size = wgpu::BufferSize::new(std::mem::size_of::<MeshUniforms>() as u64);
        let capacity = self
            .mesh_arena
            .as_ref()
            .map_or(0, |b| (b.size() / stride) as usize);
        if capacity < draws.len() {
            let capacity = (draws.len() + draws.len() / 2).max(64);
            let arena = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("threers mesh uniforms"),
                size: capacity as u64 * stride,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind = |i: usize| {
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("threers mesh bg"),
                    layout: &self.mesh_bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &arena,
                            offset: i as u64 * stride,
                            size: mesh_size,
                        }),
                    }],
                })
            };
            let rebuilt: Vec<wgpu::BindGroup> = (0..capacity).map(bind).collect();
            for (i, mesh_bg) in rebuilt.into_iter().enumerate() {
                if let Some(entry) = self.per_mesh.get_mut(i) {
                    entry.0 = i as u64 * stride;
                    entry.1 = mesh_bg;
                } else {
                    // Texture bind group is rebuilt per-frame per-mesh below,
                    // but the pool entry needs a placeholder until then.
                    let tex_bg = std::sync::Arc::new(self.make_default_tex_bg());
                    self.per_mesh.push((i as u64 * stride, mesh_bg, tex_bg));
                }
            }
            self.mesh_arena = Some(arena);
        }
        let mut mesh_bytes = std::mem::take(&mut self.mesh_bytes);
        mesh_bytes.clear();
        mesh_bytes.resize(draws.len() * stride as usize, 0);

        // -- Build combined mesh+skin bind groups now that per_mesh is grown. --
        for (i, d) in draws.iter().enumerate() {
            if d.topology.is_skinned() && d.skin_idx < skin_data.len() {
                let sd = &mut skin_data[d.skin_idx];
                let arena = self.mesh_arena.as_ref().expect("grown above");
                sd.bg = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("threers mesh skinned bg"),
                    layout: &self.mesh_skinned_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: arena,
                                offset: self.per_mesh[i].0,
                                size: mesh_size,
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: sd.bone_buf.as_entire_binding(),
                        },
                    ],
                }));
            }
        }

        // The topologies the multisampled pipelines exist for.
        let msaa_drawable = |t: Topology| {
            matches!(
                t,
                Topology::Triangle
                    | Topology::TriangleAlpha
                    | Topology::TriangleNoCull
                    | Topology::Sky
                    | Topology::TriangleWire
                    | Topology::Line
                    | Topology::Point
                    | Topology::WideLine
                    | Topology::GlassDepth
                    | Topology::GlassTint
                    | Topology::GlassColor
                    | Topology::Skinned
                    | Topology::Instanced
                    // An opaque shader material (see `custom_msaa_pipelines`).
                    | Topology::Custom(_, false, false, _)
            )
        };
        // Multisampling in a pane with a partial clear: the pane is drawn into
        // a target of its own, the size of the clear rectangle, and
        // composited into this one — see `viewport::PanePlan`. Its viewport is
        // relative to that target, so the projection is redone for it here,
        // before the frame uniforms go up.
        let pane_msaa = vp_plan.pane.filter(|_| {
            self.msaa > 1
                && !self.cube_pass
                && depth_src.is_none()
                && !linear_framebuffer
                && self.jitter_override.is_none()
                && draws.iter().all(|d| msaa_drawable(d.topology))
        });
        if let Some(p) = &pane_msaa {
            let mut proj = camera.projection_matrix();
            if let Some(fix) = &p.projection_fix {
                proj = fix.multiply(&proj);
            }
            if self.state_pipelines.reversed() {
                proj = reverse_depth(&proj);
            }
            frame_u.projection = proj.elements;
            frame_u.view_proj = proj.multiply(&view_m).elements;
        }

        // Alpha blending as three.js's default renderer does it — on the
        // sRGB-encoded bytes of its 8-bit canvas, not in linear light. A plain
        // `Unorm` target already works that way (the shader encodes). An sRGB
        // one blends in linear in hardware, so its blended mesh draws go in a
        // pass of their own, after the rest, through a `Unorm` view of the same
        // texture with the shader encoding — the hardware then mixes encoded
        // values exactly as WebGL does, while the opaque pass, its multisample
        // resolve and everything downstream stay in linear light. Needs the
        // target texture (the `render_to` paths hand it over) and view-format
        // reinterpretation (not on WebGL2); elsewhere the blend stays linear.
        //
        // Every blended kind goes there (`Topology::encodes`): meshes,
        // sprites, transparent instanced and skinned meshes and transparent
        // custom shaders. In a pane drawn through its own multisampled target
        // the pass draws into that target's `Unorm` view; in a frame with
        // screen-space refraction, into the opaque capture's, which is then
        // blitted to the target as before.
        let encoded_target: Option<&wgpu::Texture> = taa_src
            .map(|(_, c)| c)
            .filter(|t| t.format().is_srgb() && t.format() == self.color_format)
            .filter(|t| t.sample_count() == 1 && t.mip_level_count() == 1);
        //
        // Without view reinterpretation (WebGL2) the pass draws into a `Unorm`
        // scratch copy of the target instead, copied back after
        // (`ensure_encoded_scratch`) — for a single-sampled frame without a
        // pane target or screen-space refraction, which is what a WebGL2
        // canvas drawn through its back buffer is.
        let reinterpret = super::super::caps::supports_view_formats(&self.device);
        let scratch_ok = !reinterpret
            && self.msaa <= 1
            && pane_msaa.is_none()
            && !draws.iter().any(|d| {
                d.topology == Topology::Refract
                    || matches!(d.topology, Topology::Custom(_, _, true, ..))
            })
            && encoded_target
                .is_some_and(|t| t.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING));
        let encoded_pass = encoded_target.is_some()
            && !linear_framebuffer
            && !self.cube_pass
            && self.encoded_blend_pipelines.is_some()
            && (reinterpret || scratch_ok)
            && draws.iter().any(|d| d.topology.encodes());
        let encoded_scratch = encoded_pass && !reinterpret;
        if let (true, Some(t)) = (encoded_scratch, encoded_target) {
            self.ensure_encoded_scratch(t.width(), t.height());
        }

        // Material shader variants: a draw whose maps the legacy layout cannot
        // hold (or whose maps disagree on filtering or transform) gets one,
        // built here, before any pass needs its pipelines. Custom shader
        // materials compile against the legacy layout and keep it.
        for d in draws.iter_mut() {
            d.map_variant = (!matches!(d.topology, Topology::Custom(..))
                && super::super::material_maps::needs_variant(&d.slots))
            .then(|| super::super::material_maps::MapVariant::of(&d.slots));
        }
        let wanted: std::collections::HashSet<super::super::material_maps::MapVariant> =
            draws.iter().filter_map(|d| d.map_variant).collect();
        for v in wanted {
            self.ensure_map_variant(v);
        }

        // -- Upload uniforms + rebuild per-mesh tex bind groups. --
        self.queue
            .write_buffer(&self.frame_buffer, 0, bytemuck::bytes_of(&frame_u));
        for (i, d) in draws.iter().enumerate() {
            let mut flags: u32 = 0;
            if d.slots.map.is_some() {
                flags |= FLAG_MAP;
            }
            if d.slots.normal_map.is_some() {
                flags |= FLAG_NORMAL_MAP;
            }
            if d.slots.roughness_map.is_some() {
                flags |= FLAG_ROUGHNESS_MAP;
            }
            if d.slots.metalness_map.is_some() {
                flags |= FLAG_METALNESS_MAP;
            }
            if d.slots.ao_map.is_some() {
                flags |= FLAG_AO_MAP;
            }
            if d.slots.emissive_map.is_some() {
                flags |= FLAG_EMISSIVE_MAP;
            }
            if d.slots.matcap_map.is_some() {
                flags |= FLAG_MATCAP_MAP;
            }
            if d.slots.cloud_shadow_map.is_some() && d.atmos[1] > 0.0 {
                flags |= FLAG_CLOUD_SHADOW;
            }

            let u = MeshUniforms {
                model: d.model,
                normal_matrix: d.normal_matrix,
                color: d.color,
                emissive: d.emissive,
                specular: d.specular,
                params: [
                    d.shininess_or_near,
                    d.opacity_or_far,
                    d.roughness,
                    d.metalness,
                ],
                params2: [d.ao_intensity, d.normal_scale[0], d.normal_scale[1], 0.0],
                params3: d.physical,
                params4: d.physical2,
                params5: d.physical3,
                params6: d.physical4,
                params7: d.physical5,
                params8: d.displacement,
                // A variant transforms each map's uv itself; its vertex stage
                // passes the raw uv through.
                params9: if d.map_variant.is_some() {
                    [0.0, 0.0, 1.0, 1.0]
                } else {
                    d.uv_transform
                },
                params10: d.atmos,
                params11: d.twilight,
                params12: d.eclipse,
                params13: d.scatter,
                params14: d.aurora,
                flags: [
                    d.kind as u32,
                    flags,
                    d.shader_flags
                        | if encoded_pass && d.topology.encodes() {
                            FLAG_ENCODED_OUT
                        } else {
                            0
                        },
                    d.alpha_test.to_bits(),
                ],
                // `morph.w` bit 32: the geometry has extra uv sets.
                morph: {
                    let mut m = d.morph;
                    if self.uv_cache.contains_key(&d.key) {
                        m[3] = (m[3] as u32 | 32) as f32;
                    }
                    m
                },
                morph_weights: bytemuck::cast(d.morph_weights),
                clip_plane: d.clip_plane,
            };
            let o = self.per_mesh[i].0 as usize;
            mesh_bytes[o..o + std::mem::size_of::<MeshUniforms>()]
                .copy_from_slice(bytemuck::bytes_of(&u));
        }
        if let (Some(arena), false) = (&self.mesh_arena, mesh_bytes.is_empty()) {
            self.queue.write_buffer(arena, 0, &mesh_bytes);
        }
        self.mesh_bytes = mesh_bytes;
        // Texture bind groups, from the cache. Keys are computed up front so
        // the miss path can borrow `self` immutably to build and then mutably
        // to insert, one draw at a time.
        let tex_keys: Vec<TexBgKey> = draws
            .iter()
            .map(|d| {
                let (maps, _, sampler, _) = tex_bg_key(&d.slots);
                (maps, d.morph_tex, sampler, self.uv_key(d.key))
            })
            .collect();
        // The set of distinct texture combinations in a scene is small and
        // static; this bound is a runaway guard, not a working limit.
        if self.tex_bg_cache.len() > 4096 {
            self.tex_bg_cache.clear();
            self.variant_bg_cache.clear();
        }
        for (i, d) in draws.iter().enumerate() {
            if !self.tex_bg_cache.contains_key(&tex_keys[i]) {
                let morph = self
                    .morph_cache
                    .get(&(d.morph_tex.0 as *const BufferGeometry))
                    .filter(|_| d.morph_tex.0 != 0)
                    .map(|m| &m.view);
                let uv = self.uv_cache.get(&d.key).map(|u| &u.view);
                let bg = std::sync::Arc::new(self.build_tex_bg(&d.slots, morph, uv));
                self.tex_bg_cache.insert(tex_keys[i], bg);
            }
        }
        for (i, k) in tex_keys.iter().enumerate() {
            self.per_mesh[i].2 = self.tex_bg_cache[k].clone();
        }
        // Variant draws: their own group 2.
        if self.variant_bg_cache.len() > 4096 {
            self.variant_bg_cache.clear();
        }
        for (i, d) in draws.iter().enumerate() {
            let Some(v) = d.map_variant else { continue };
            let key = (
                variant_bg_key(&d.slots, v, d.morph_tex, d.map_uniforms),
                self.uv_key(d.key),
            );
            if !self.variant_bg_cache.contains_key(&key) {
                let morph = self
                    .morph_cache
                    .get(&(d.morph_tex.0 as *const BufferGeometry))
                    .filter(|_| d.morph_tex.0 != 0)
                    .map(|m| &m.view);
                let uv = self.uv_cache.get(&d.key).map(|u| &u.view);
                let bg = std::sync::Arc::new(self.build_variant_tex_bg(
                    &d.slots,
                    morph,
                    uv,
                    v,
                    d.map_uniforms,
                ));
                self.variant_bg_cache.insert(key.clone(), bg);
            }
            self.per_mesh[i].2 = self.variant_bg_cache[&key].clone();
        }
        // A shader material's group 2 when its geometry is morphed or has
        // extra uv sets: the shared custom one with the geometry's morph
        // texture and uv sets in place of the empty defaults. (Its `morph.w`
        // bit 32 is set whenever the uv sets exist, so the shared group's
        // empty texture would be read as zeros.)
        let custom_morph_bgs: Vec<Option<wgpu::BindGroup>> = draws
            .iter()
            .map(|d| {
                let uv = self.uv_cache.get(&d.key).map(|u| &u.view);
                if !matches!(d.topology, Topology::Custom(..))
                    || (d.morph_tex.0 == 0 && uv.is_none())
                {
                    return None;
                }
                let view = if d.morph_tex.0 == 0 {
                    &self.default_morph.1
                } else {
                    &self
                        .morph_cache
                        .get(&(d.morph_tex.0 as *const BufferGeometry))?
                        .view
                };
                let uv = uv.unwrap_or(&self.default_uv_sets.1);
                Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("threers custom morph tex bg"),
                    layout: &self.custom_tex_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: wgpu::BindingResource::TextureView(uv),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: wgpu::BindingResource::Sampler(&self.sampler_linear_repeat),
                        },
                        wgpu::BindGroupEntry {
                            binding: 10,
                            resource: wgpu::BindingResource::TextureView(view),
                        },
                    ],
                }))
            })
            .collect();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("threers encoder"),
            });

        // Whether this render redraws the shadow maps: always, unless
        // `shadow_map.auto_update` is off — then only when asked, or when the
        // maps were last drawn for other casters or another size. They
        // depend on lights and casters, never on the camera (the directional
        // map is a fixed cube on the origin), so a skipped update samples the
        // same maps a redraw would have produced.
        //
        // With `auto_update` on (the default), a render whose casters and
        // lights are the same as last time keeps the maps it has: a still
        // scene under a moving camera stops redrawing every caster into every
        // map each frame.
        let mut pass_hashes: Vec<Option<u64>> = Vec::new();
        let mut pass_changed: Vec<bool> = Vec::new();
        let update_shadows = !shadow_passes.is_empty() && {
            let sig: ShadowSig = (
                dir_layers,
                spot_layers,
                point_layers,
                self.dir_shadows.size,
                self.spot_shadows.size,
            );
            let valid = self.shadow_sig == Some(sig);
            let content = {
                let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                let mut mix = |x: u64| h = mix_word(h, x);
                let mut known = true;
                for d in draws.iter().filter(|d| d.cast_shadow) {
                    // Skinned: the pose. Instanced: the instance data, through
                    // its content stamp.
                    if d.topology.is_skinned() {
                        mix(skin_data.get(d.skin_idx).map_or(0, |s| s.bone_hash));
                    }
                    mix(d.instance_stamp);
                    mix(d.instance_count as u64);
                    // A custom shader is a function of its inputs — there is
                    // no clock in the frame uniforms — so its data, storage and
                    // source stand for its output. Not its textures: a texture
                    // can be drawn into without anything here changing.
                    if let Topology::Custom(..) = d.topology {
                        match shader_data.get(d.shader_idx) {
                            Some(sd) if sd.textures.is_empty() && sd.volumes.is_empty() => {
                                // A map is pixels the renderer owns: its id
                                // and upload count stand for its contents.
                                sd.maps.iter().for_each(|m| {
                                    let k = tex_cache_key(m);
                                    mix(k.0);
                                    mix(k.1);
                                });
                                sd.data
                                    .iter()
                                    .flatten()
                                    .for_each(|v| mix(v.to_bits() as u64));
                                sd.storage.iter().for_each(|st| {
                                    mix(st.len() as u64);
                                    st.iter().for_each(|v| mix(v.to_bits() as u64));
                                });
                                mix(hash_words(sd.fragment.bytes().map(u64::from)));
                                mix(sd
                                    .vertex
                                    .as_deref()
                                    .map_or(0, |v| hash_words(v.bytes().map(u64::from))));
                            }
                            _ => {
                                known = false;
                                break;
                            }
                        }
                    }
                    mix(d.key as usize as u64);
                    mix(self.geom_cache.get(&d.key).map_or(0, |e| e.version) as u64);
                    d.model.iter().for_each(|v| mix(v.to_bits() as u64));
                    mix(d.shadow_side as u64);
                    mix(d.alpha_test.to_bits() as u64);
                    mix(d
                        .slots
                        .map
                        .as_ref()
                        .map_or(0, |t| std::sync::Arc::as_ptr(t) as usize as u64));
                    mix(d
                        .slots
                        .alpha_map
                        .as_ref()
                        .map_or(0, |t| std::sync::Arc::as_ptr(t) as usize as u64));
                    mix(d.shader_flags as u64);
                    d.color
                        .iter()
                        .chain(&d.physical)
                        .chain(&d.morph)
                        .for_each(|v| mix(v.to_bits() as u64));
                    d.morph_weights.iter().for_each(|v| mix(v.to_bits() as u64));
                    mix(d
                        .range
                        .map_or(u64::MAX, |(a, b)| ((a as u64) << 32) | b as u64));
                }
                known.then_some(h)
            };
            // Each pass's own: the casters and its light's matrix. A pass is
            // redrawn when its own changed — a cascade following the camera
            // does not take a still point light's six faces with it.
            pass_hashes = shadow_passes
                .iter()
                .map(|p| {
                    content.map(|c| {
                        let mut h = mix_word(c, p.layer as u64 | ((p.kind as u64) << 32));
                        for v in p.vp.iter().chain(&p.pos) {
                            h = mix_word(h, v.to_bits() as u64);
                        }
                        h
                    })
                })
                .collect();
            pass_changed = pass_hashes
                .iter()
                .enumerate()
                .map(|(k, h)| !valid || h.is_none() || self.shadow_pass_hashes.get(k) != Some(h))
                .collect();
            let unchanged = !pass_changed.iter().any(|&c| c);
            self.shadow_sig = Some(sig);
            self.shadow_map.take_update(valid, unchanged)
        };
        // A forced update (`needs_update` with `auto_update` off) redraws
        // every pass; an automatic one, the passes that changed.
        let draw_pass: Vec<bool> = (0..shadow_passes.len())
            .map(|k| {
                update_shadows
                    && (!self.shadow_map.auto_update
                        || pass_changed.get(k).copied().unwrap_or(true))
            })
            .collect();
        if update_shadows {
            self.shadow_pass_hashes = std::mem::take(&mut pass_hashes);
        }
        let update_shadows = draw_pass.iter().any(|&d| d);
        if update_shadows {
            self.shadow_updates += 1;
            // Every pass's matrix, at its own aligned offset in one buffer,
            // written once.
            self.ensure_shadow_pass_buffer(shadow_passes.len());
            let stride = self.shadow_pass_stride as usize;
            let mut bytes = vec![0u8; stride * shadow_passes.len()];
            for (k, p) in shadow_passes.iter().enumerate() {
                let g = ShadowPassGpu {
                    vp: p.vp,
                    light: p.pos,
                };
                bytes[k * stride..k * stride + std::mem::size_of::<ShadowPassGpu>()]
                    .copy_from_slice(bytemuck::bytes_of(&g));
            }
            if let Some((buffer, _)) = &self.shadow_pass_buffer {
                self.queue.write_buffer(buffer, 0, &bytes);
            }
        }

        // The shadow depth passes: every caster from each light, one layer
        // each (six for a point light). The timer brackets the first pass's
        // start and the last one's end.
        // Each caster's world bounding sphere, where its geometry's bounds
        // really bound what it draws (as the main pass culls): a pass skips
        // casters outside its light's frustum.
        let caster_spheres: Vec<Option<crate::math::Sphere>> = draws
            .iter()
            .map(|d| {
                let plain = d.cast_shadow
                    && (matches!(
                        d.topology,
                        Topology::Triangle | Topology::TriangleNoCull | Topology::TriangleAlpha
                    ) || d.topology.is_glass_mesh())
                    && d.instance_count == 1
                    && d.skin_idx == usize::MAX
                    && d.shader_idx == usize::MAX
                    && d.morph[0] == 0.0;
                if !plain {
                    return None;
                }
                let sphere = self.geom_cache.get(&d.key).and_then(|g| g.sphere)?;
                Some(sphere.apply_matrix4(&Matrix4 { elements: d.model }))
            })
            .collect();
        let drawn: Vec<usize> = (0..shadow_passes.len()).filter(|&k| draw_pass[k]).collect();
        self.shadow_stats.0 += drawn.len() as u64;
        for (n, &k) in drawn.iter().enumerate() {
            let p = &shadow_passes[k];
            let pass_frustum = Frustum::from_projection_matrix(&Matrix4 { elements: p.vp });
            let view = match p.kind {
                ShadowKind::Dir => self.dir_shadows.layer_view(p.layer),
                ShadowKind::Spot => self.spot_shadows.layer_view(p.layer),
                ShadowKind::Point => self.point_shadows.layer_view(p.layer),
            };
            let timestamp_writes =
                self.shadow_timer
                    .as_ref()
                    .and_then(|t| match (n == 0, n + 1 == drawn.len()) {
                        (true, true) => Some(t.mark_both(0, 1)),
                        (true, false) => Some(t.mark(0)),
                        (false, true) => Some(t.mark_end(1)),
                        _ => None,
                    });
            let mut spass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(match p.kind {
                    ShadowKind::Dir => "threers shadow pass",
                    ShadowKind::Spot => "threers spot shadow pass",
                    ShadowKind::Point => "threers point shadow face pass",
                }),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let Some((_, pass_bg)) = &self.shadow_pass_buffer else {
                break;
            };
            spass.set_bind_group(0, pass_bg, &[(k as u64 * self.shadow_pass_stride) as u32]);
            for (i, d) in draws.iter().enumerate() {
                // three.js draws only `castShadow` objects into a shadow map.
                if !d.cast_shadow {
                    continue;
                }
                if caster_spheres[i]
                    .as_ref()
                    .is_some_and(|s| !pass_frustum.intersects_sphere(s))
                {
                    continue;
                }
                self.shadow_stats.1 += 1;
                let skinned = matches!(d.topology, Topology::Skinned | Topology::SkinnedAlpha);
                // A `ShaderMaterial` casts its geometry, as three.js's depth
                // material draws it (and the Metal backend does): the mesh's
                // own triangles, on an `InstancedMesh` at each instance.
                let custom = matches!(d.topology, Topology::Custom(..));
                let instanced =
                    matches!(d.topology, Topology::Instanced | Topology::InstancedAlpha)
                        || (custom && shader_data.get(d.shader_idx).is_some_and(|sd| sd.instanced));
                // A transparent caster casts too: three.js's shadow map ignores
                // blending (only `alphaTest` cuts a shadow). So does glass in
                // the modes with passes of their own: solid glass fully, thin
                // glass through the cut-out pipelines (`shadow_transmits`).
                if d.topology != Topology::Triangle
                    && d.topology != Topology::TriangleNoCull
                    && d.topology != Topology::TriangleAlpha
                    && !d.topology.is_glass_mesh()
                    && !skinned
                    && !instanced
                    && !custom
                {
                    continue;
                }
                let gm = &self.geom_cache[&d.key].mesh;
                let mut instances = 0..1;
                // three.js's shadow depth material takes `map` and `alphaTest`;
                // the point passes have no cut-out variant.
                let cut = p.kind != ShadowKind::Point
                    && (d.shadow_transmits
                        || (d.alpha_test > 0.0
                            && (d.slots.map.is_some() || d.slots.alpha_map.is_some())));
                let c = &self.shadow_cut;
                let base = match (p.kind, skinned, instanced, cut) {
                    (ShadowKind::Dir, true, _, false) => &self.pipeline_shadow_skinned,
                    (ShadowKind::Dir, true, _, true) => &c.dir_skinned,
                    (ShadowKind::Dir, false, true, false) => &self.pipeline_shadow_instanced,
                    (ShadowKind::Dir, false, true, true) => &c.dir_instanced,
                    (ShadowKind::Dir, false, false, false) => &self.pipeline_shadow,
                    (ShadowKind::Dir, false, false, true) => &c.dir,
                    (ShadowKind::Spot, true, _, false) => &self.pipeline_shadow_spot_skinned,
                    (ShadowKind::Spot, true, _, true) => &c.spot_skinned,
                    (ShadowKind::Spot, false, true, false) => &self.pipeline_shadow_spot_instanced,
                    (ShadowKind::Spot, false, true, true) => &c.spot_instanced,
                    (ShadowKind::Spot, false, false, false) => &self.pipeline_shadow_spot,
                    (ShadowKind::Spot, false, false, true) => &c.spot,
                    (ShadowKind::Point, true, _, _) => &self.pipeline_shadow_point_skinned,
                    (ShadowKind::Point, false, true, _) => &self.pipeline_shadow_point_instanced,
                    (ShadowKind::Point, false, false, _) => &self.pipeline_shadow_point,
                };
                let cull = if p.kind == ShadowKind::Point {
                    shadow_cull_point(d.shadow_side)
                } else {
                    shadow_cull(d.shadow_side)
                };
                if skinned {
                    // The pose, not the bind pose: the skinned pipeline with
                    // the bones and the joint/weight buffer.
                    let (Some(bg), Some(attrs)) = (
                        skin_data.get(d.skin_idx).and_then(|s| s.bg.as_ref()),
                        self.skin_attr_cache.get(&d.key),
                    ) else {
                        continue;
                    };
                    let base = variant_pipeline(
                        &mut self.state_pipelines,
                        &self.map_variants,
                        &self.device,
                        &self.main_shader,
                        base,
                        d.map_variant,
                    );
                    spass.set_pipeline(&self.state_pipelines.cull_variant(
                        &self.device,
                        &base,
                        cull,
                    ));
                    spass.set_bind_group(1, bg, &[]);
                    spass.set_vertex_buffer(1, attrs.slice(..));
                } else {
                    let base = variant_pipeline(
                        &mut self.state_pipelines,
                        &self.map_variants,
                        &self.device,
                        &self.main_shader,
                        base,
                        d.map_variant,
                    );
                    spass.set_pipeline(&self.state_pipelines.cull_variant(
                        &self.device,
                        &base,
                        cull,
                    ));
                    spass.set_bind_group(1, &self.per_mesh[i].1, &[]);
                    if instanced {
                        // Every instance, at its matrix and with its morph row.
                        spass.set_vertex_buffer(1, instance_buffers[d.instance_buf_idx].slice(..));
                        instances = 0..d.instance_count;
                    }
                }
                spass.set_bind_group(2, &*self.per_mesh[i].2, &[]);
                spass.set_bind_group(3, &self.shadow_env_bind_group, &[]);
                spass.set_vertex_buffer(0, gm.vertex_buffer.slice(..));
                if let Some(ib) = &gm.index_buffer {
                    spass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    spass.draw_indexed(draw_span(d.range, gm.index_count), 0, instances);
                } else {
                    spass.draw(draw_span(d.range, gm.vertex_count), instances);
                }
            }
        }

        // Screen-space refraction glass: when any Refract surface is present, the
        // opaque scene is rendered into the mipped ss-color capture (instead of
        // straight to the frame target); the glass pass below then refracts and
        // reflects that capture. Otherwise the main pass targets the frame view
        // directly, exactly as before.
        // A custom shader that asked for the opaque-scene capture needs the same
        // machinery the refraction glass does, so it turns it on too.
        // SUPPRESSED while rendering an environment cube. The glass path
        // copies the frame's depth into a screen-sized scratch texture, and a
        // cube face's depth is 48 pixels, not 7680 — wgpu refuses the copy
        // outright ("the entire texture must be copied when copying from depth
        // texture") and the whole render dies. A reflection probe has no use
        // for screen-space refraction anyway: it exists to light other
        // surfaces, and what it would refract is not in it.
        let wants_screen_space = draws.iter().any(|d| {
            d.topology == Topology::Refract
                || matches!(d.topology, Topology::Custom(_, _, true, ..))
        });
        // Screen-space refraction captures and marches the *whole* frame, so
        // it has no per-pane meaning: with a viewport or a partial clear those
        // draws are skipped (the main pass already leaves them out) rather
        // than drawn against the wrong pixels.
        if wants_screen_space
            && !vp_plan.legacy
            && !self.cube_pass
            && !self.viewport_state.warned_screen_space
        {
            self.viewport_state.warned_screen_space = true;
            log::warn!(
                "screen-space refraction / screen-space shader materials are not drawn \
                 with a viewport or scissor set"
            );
        }
        let refract_present = !self.cube_pass && vp_plan.legacy && wants_screen_space;
        let ss_format = if linear_framebuffer {
            wgpu::TextureFormat::Rgba16Float
        } else {
            self.color_format
        };
        if refract_present {
            let (w, h) = self.depth_size;
            self.ensure_ss_targets(w, h, ss_format);
        }

        // Compile any not-yet-seen custom-shader pipelines and build their
        // @group(1) user bind groups (buffers kept alive until submit). Hoisted
        // above the passes because a screen-space custom material draws in the
        // glass pass rather than the opaque one, and both need these.
        let mut user_bgs: Vec<Option<wgpu::BindGroup>> = Vec::new();
        user_bgs.resize_with(draws.len(), || None);
        let mut _custom_keep_alive: Vec<wgpu::Buffer> = Vec::new();
        let mut _custom_keep_alive_tex: Vec<wgpu::Texture> = Vec::new();
        for (i, d) in draws.iter().enumerate() {
            if let Topology::Custom(hash, transparent, screen_space, side) = d.topology {
                let sd = &shader_data[d.shader_idx];
                for map in sd.maps.iter().take(4) {
                    self.ensure_slot_uploaded(&Some(map.clone()));
                }
                let key = (hash, linear_framebuffer, transparent, screen_space, side);
                let lit = self.custom_lit_pipeline_layout.is_some() && {
                    if !self.custom_lit.contains_key(&hash) {
                        let source = custom_shader_module_source(
                            &sd.fragment,
                            sd.vertex.as_deref(),
                            false,
                            false,
                            1,
                        );
                        let reach = custom_shader_reach(&source);
                        if (reach.full_env || screen_space) && reach.volumes {
                            log::error!(
                                "shader material skipped: shadows, area lights or screen space \
                                 and a volume slot do not fit on WebGL2's sixteen texture units"
                            );
                            self.custom_failed.insert(key);
                        }
                        self.custom_lit.insert(hash, reach.full_env);
                    }
                    self.custom_lit[&hash]
                };
                let no_volumes = lit || (screen_space && self.custom_lit_mesh_bgl.is_some());
                let layout = if screen_space {
                    self.custom_ss_pipeline_layout.as_ref()
                } else if lit {
                    self.custom_lit_pipeline_layout.as_ref()
                } else {
                    self.custom_pipeline_layout.as_ref()
                };
                let Some(custom_layout) = layout else {
                    // No layout on this device — skip the object rather than
                    // submit a draw that cannot validate.
                    continue;
                };
                if self.custom_failed.contains(&key) {
                    continue;
                }
                if !self.custom_pipelines.contains_key(&key) {
                    let fmt = if linear_framebuffer {
                        wgpu::TextureFormat::Rgba16Float
                    } else {
                        self.color_format
                    };
                    let pipeline = build_custom_pipeline(
                        &self.device,
                        custom_layout,
                        &sd.fragment,
                        sd.vertex.as_deref(),
                        fmt,
                        transparent,
                        screen_space,
                        side,
                        sd.instanced,
                        1,
                        sd.alpha_to_coverage,
                        &mut self.state_pipelines,
                    );
                    match pipeline {
                        Ok(p) => {
                            self.custom_pipelines.insert(key, p);
                        }
                        Err(e) => {
                            // One material is skipped; the rest of the frame
                            // draws. Previously this was a panic from inside
                            // wgpu, which took the program with it.
                            log::error!("shader material failed to compile:\n{e}");
                            self.custom_failed.insert(key);
                            continue;
                        }
                    }
                }
                // The encoded-blend pass's variant, into the `Unorm` view.
                let ekey = (hash, transparent, screen_space, side);
                if encoded_pass
                    && d.topology.encodes()
                    && !self.custom_encoded_pipelines.contains_key(&ekey)
                {
                    match build_custom_pipeline(
                        &self.device,
                        custom_layout,
                        &sd.fragment,
                        sd.vertex.as_deref(),
                        self.color_format.remove_srgb_suffix(),
                        transparent,
                        screen_space,
                        side,
                        sd.instanced,
                        1,
                        sd.alpha_to_coverage,
                        &mut self.state_pipelines,
                    ) {
                        Ok(p) => {
                            self.custom_encoded_pipelines.insert(ekey, p);
                        }
                        Err(e) => log::error!("shader material failed to compile:\n{e}"),
                    }
                }
                // The multisampled variant, for a frame drawn under MSAA: an
                // opaque material only (the blended passes draw single-sampled).
                let mkey = (hash, transparent, screen_space, side);
                if self.msaa > 1
                    && !transparent
                    && !screen_space
                    && !linear_framebuffer
                    && !self.custom_msaa_pipelines.contains_key(&mkey)
                {
                    match build_custom_pipeline(
                        &self.device,
                        custom_layout,
                        &sd.fragment,
                        sd.vertex.as_deref(),
                        self.color_format,
                        transparent,
                        screen_space,
                        side,
                        sd.instanced,
                        MSAA_SAMPLES,
                        sd.alpha_to_coverage,
                        &mut self.state_pipelines,
                    ) {
                        Ok(p) => {
                            self.custom_msaa_pipelines.insert(mkey, p);
                        }
                        Err(e) => log::error!("shader material failed to compile:\n{e}"),
                    }
                }
                let (bg, bufs, texs) = build_user_bind_group(
                    &self.device,
                    &self.queue,
                    if no_volumes {
                        self.custom_lit_mesh_bgl
                            .as_ref()
                            .expect("no_volumes implies the layout")
                    } else {
                        &self.custom_mesh_bgl
                    },
                    wgpu::BufferBinding {
                        buffer: self.mesh_arena.as_ref().expect("grown above"),
                        offset: self.per_mesh[i].0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<MeshUniforms>() as u64),
                    },
                    &sd.data,
                    &sd.storage,
                    &(0..4)
                        .map(|t| {
                            sd.textures.get(t).and_then(|v| v.as_wgpu()).or_else(|| {
                                sd.maps.get(t).map(|m| {
                                    self.view_for(&Some(m.clone()), &self.default_white_linear)
                                })
                            })
                        })
                        .collect::<Vec<_>>(),
                    &self.default_white_linear.view,
                    &self.default_data_tex_view,
                    &self.sampler_linear_repeat,
                    &sd.volumes,
                    &self.default_volume_view,
                    &self.sampler_volume,
                    !no_volumes
                        && crate::renderer::caps::supports_shader_material_volumes(&self.device),
                );
                _custom_keep_alive.extend(bufs);
                _custom_keep_alive_tex.extend(texs);
                user_bgs[i] = Some(bg);
            }
        }

        {
            let main_format = if linear_framebuffer {
                wgpu::TextureFormat::Rgba16Float
            } else {
                self.color_format
            };
            // `scene.background` is linear; the shared rule encodes it only
            // where the shader encodes too (a plain `Unorm` target), so a hex
            // background lands as that hex — see `super::super::clear_value`.
            let clear =
                super::super::clear_value(main_format, scene.background, scene.background_alpha);
            // Hardware MSAA on the direct-to-surface opaque pass. Falls back to
            // single-sample for render-to-texture, linear/postfx targets, and any
            // scene using glass/OIT/refraction/instanced/skinned/custom draws.
            // Not straight into the target when the colour is loaded (a partial
            // clear, or none): the resolve rewrites the whole target and would
            // erase the other panes. A partial clear goes through a pane-sized
            // target instead (`pane_msaa`); with no clear at all there is
            // nothing to start the pane target from, so that stays
            // single-sampled.
            let pane_key = pane_msaa
                .as_ref()
                .map(|p| (p.region.width, p.region.height, main_format, self.msaa));
            // OIT and refraction glass multisample on the whole target (not
            // in a pane), with a capture the colour can resolve into.
            let oit_msaa = !self.pipeline_oit_msaa.is_empty() && !linear_framebuffer;
            // Under MSAA the glass reads a single-sampled opaque depth: the
            // multisampled one resolved where a shader can read it, the
            // opaque scene's depth redrawn where it cannot (WebGL2).
            let refract_msaa = !linear_framebuffer && ss_format == self.color_format;
            let use_msaa = pane_msaa.is_some()
                || (self.msaa > 1
                    && vp_plan.clear == super::super::viewport::ClearMode::Full
                    && depth_src.is_none()
                    && (!linear_framebuffer || taa_src.is_some())
                    && (!refract_present || refract_msaa)
                    && draws.iter().all(|d| {
                        msaa_drawable(d.topology)
                            || (d.topology == Topology::Oit && oit_msaa)
                            || (d.topology == Topology::Refract && refract_msaa)
                    }));
            // Not over refraction glass: it is drawn single-sampled into the
            // frame, and resolving the multisampled colour again would erase it.
            self.overlay_msaa = use_msaa && pane_msaa.is_none() && !refract_present;
            self.last_frame_msaa = use_msaa;
            // three.js `polygonOffset`: the depth-biased pipelines this frame's
            // draws need, built once per (topology, target, offset).
            let bias_variant: u8 = if use_msaa {
                2
            } else if linear_framebuffer {
                1
            } else {
                0
            };
            // The encoded-blend pass's variant: the target's `Unorm` view,
            // single- or multisampled.
            let enc_variant: u8 = if use_msaa { 4 } else { 3 };
            if encoded_pass && use_msaa {
                self.ensure_msaa_targets(self.depth_size.0, self.depth_size.1);
            }
            for d in &draws {
                let in_encoded_pass = encoded_pass && d.topology == Topology::TriangleAlpha;
                let (factor, units) = match d.polygon_offset {
                    Some(o) => o,
                    None if in_encoded_pass => (0f32.to_bits(), 0),
                    None => continue,
                };
                let Some(code) = biased_topology(d.topology) else {
                    continue;
                };
                let variant = if in_encoded_pass {
                    enc_variant
                } else {
                    bias_variant
                };
                let key = (code, variant, factor, units);
                if !self.biased_pipelines.contains_key(&key) {
                    let (format, samples) = match variant {
                        4 => (self.color_format.remove_srgb_suffix(), MSAA_SAMPLES),
                        3 => (self.color_format.remove_srgb_suffix(), 1),
                        2 => (self.color_format, MSAA_SAMPLES),
                        1 => (wgpu::TextureFormat::Rgba16Float, 1),
                        _ => (self.color_format, 1),
                    };
                    let p = build_biased_pipeline(
                        &self.device,
                        &self.mesh_shader,
                        &self.mesh_pipeline_layout,
                        format,
                        samples,
                        code,
                        f32::from_bits(factor),
                        units,
                        &mut self.state_pipelines,
                    );
                    self.biased_pipelines.insert(key, p);
                }
            }
            if let Some(key) = pane_key {
                self.viewport_state
                    .ensure_pane_target(&self.device, key, DEPTH_FORMAT);
            } else if use_msaa {
                self.ensure_msaa_targets(self.depth_size.0, self.depth_size.1);
            }
            // The pane's own plan when drawing into a pane target.
            let main_plan = pane_msaa.as_ref().map(|p| p.local()).unwrap_or(vp_plan);
            if let super::super::viewport::ClearMode::Rect(_) = vp_plan.clear {
                self.viewport_state.ensure_clear_pipeline(
                    &self.device,
                    main_format,
                    Some(DEPTH_FORMAT),
                );
            }
            // Screen-space glass redirects the opaque pass into the ss-color capture
            // (mip 0); MSAA renders into the multisampled color and resolves into the
            // frame view; otherwise it targets the frame view directly.
            let (main_color_view, msaa_resolve, depth_pass_view): (
                &wgpu::TextureView,
                Option<&wgpu::TextureView>,
                &wgpu::TextureView,
            ) = if let Some(pane) = pane_key
                .as_ref()
                .and_then(|k| self.viewport_state.pane_target(k))
            {
                (&pane.msaa_color, Some(&pane.resolve), &pane.msaa_depth)
            } else if use_msaa {
                let (_, _, cv, dv) = self.msaa_targets.as_ref().unwrap();
                // Under refraction glass the opaque scene resolves into the
                // capture the glass refracts, as it is drawn there unsampled.
                let resolve = if refract_present {
                    &self.ss_color.as_ref().unwrap().mip_render_views[0]
                } else {
                    target_view
                };
                (cv, Some(resolve), dv)
            } else if refract_present {
                (
                    &self.ss_color.as_ref().unwrap().mip_render_views[0],
                    None,
                    &self.depth_view,
                )
            } else {
                (target_view, None, &self.depth_view)
            };

            // Timestamps go on the main pass alone. It is where the scene's
            // geometry is actually shaded — shadows, post-fx and the glass
            // passes are separate costs and averaging them together would hide
            // exactly the change anyone is trying to measure.
            let main_timestamps = self.timer.as_ref().map(|t| t.writes());
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("threers main pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: main_color_view,
                    depth_slice: None,
                    resolve_target: msaa_resolve,
                    ops: wgpu::Operations {
                        load: if main_plan.clear == super::super::viewport::ClearMode::Full {
                            wgpu::LoadOp::Clear(clear)
                        } else {
                            wgpu::LoadOp::Load
                        },
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_pass_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.depth_clear()),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: main_timestamps,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            // The partial clear first: the clear colour over the scissor
            // rectangle, as `gl.clear` does with the scissor test on.
            if let super::super::viewport::ClearMode::Rect(r) = main_plan.clear {
                self.viewport_state
                    .fill_rect(&mut pass, main_format, true, r, clear);
            }
            main_plan.apply(&mut pass);
            // Alpha-to-coverage pipelines store the blend constant as their
            // alpha: opaque. (The partial clear above sets its own.)
            pass.set_blend_constant(wgpu::Color::WHITE);
            pass.set_bind_group(0, &self.frame_bind_group, &[]);
            pass.set_bind_group(3, &self.env_bind_group, &[]);
            let mut last_topology: Option<Topology> = None;
            let mut last_bias: Option<(u32, i32)> = None;
            let mut last_state = crate::materials::RenderState::DEFAULT;
            let mut last_variant: Option<super::super::material_maps::MapVariant> = None;
            let mut last_side: u32 = u32::MAX;
            // Group 3 is the scene's environment unless a material has its own.
            let mut env_bound: (Option<MaterialEnvKey>, bool) = (None, false);
            for (i, d) in draws.iter().enumerate() {
                // A custom draw on a trimmed-environment backend (WebGL2) binds
                // the custom pipeline's own group 3; the full environment its
                // standard counterpart binds does not match that pipeline layout.
                let custom_env = self.custom_env_bind_group.is_some()
                    && matches!(d.topology, Topology::Custom(h, ..)
                        if !self.custom_lit.get(&h).copied().unwrap_or(false));
                if (d.env_group, custom_env) != env_bound {
                    pass.set_bind_group(
                        3,
                        if custom_env {
                            self.custom_env_bind_group.as_ref().unwrap()
                        } else {
                            d.env_group
                                .and_then(|k| self.material_env_groups.get(&k))
                                .map_or(&self.env_bind_group, |(_, g)| g)
                        },
                        &[],
                    );
                    env_bound = (d.env_group, custom_env);
                }
                // Off-screen. The shadow passes above deliberately did draw it:
                // a caster the camera cannot see still casts into the view.
                if !d.in_frustum {
                    continue;
                }
                // Writes neither colour nor depth: nothing to see here (a
                // shadow-only caster — fur's and hair's). Skipped rather than
                // drawn for nothing; and on WebGL2 a colour-masked draw left
                // the whole frame black.
                if !d.state.color_write && d.state.depth_write == Some(false) {
                    continue;
                }
                // OIT + screen-space-refraction surfaces are drawn in dedicated
                // later passes, not here.
                if d.topology == Topology::Oit
                    || d.topology == Topology::Refract
                    || matches!(d.topology, Topology::Custom(_, _, true, ..))
                {
                    continue;
                }
                // Objects opted out of glass capture draw *over* the glass (a
                // later overlay pass), so they're not refracted — skip here.
                if refract_present && !d.refract_capture {
                    continue;
                }
                // Blended draws on an sRGB target: the encoded-blend pass.
                if encoded_pass && d.topology.encodes() {
                    continue;
                }
                if last_topology != Some(d.topology)
                    || last_bias != d.polygon_offset
                    || last_state != d.state
                    || last_variant != d.map_variant
                    || last_side != d.side
                {
                    let biased = d.polygon_offset.and_then(|(factor, units)| {
                        let code = biased_topology(d.topology)?;
                        self.biased_pipelines
                            .get(&(code, bias_variant, factor, units))
                    });
                    let pipe = if let Some(p) = biased {
                        p
                    } else if use_msaa {
                        match d.topology {
                            Topology::Triangle => &self.pipeline_tri_msaa,
                            Topology::TriangleAlpha => &self.pipeline_tri_alpha_msaa,
                            Topology::TriangleNoCull => &self.pipeline_tri_nocull_msaa,
                            Topology::Sky => &self.pipeline_tri_sky_msaa,
                            Topology::TriangleWire => &self.pipeline_tri_wire_msaa,
                            Topology::Line => &self.pipeline_line_msaa,
                            Topology::Point => &self.pipeline_point_msaa,
                            Topology::WideLine => &self.pipeline_wide_line_msaa,
                            Topology::GlassDepth => &self.pipeline_glass_depth_msaa,
                            Topology::GlassColor => &self.pipeline_glass_color_msaa,
                            Topology::GlassTint => &self.pipeline_glass_tint_msaa,
                            Topology::Skinned => &self.pipeline_skinned_msaa,
                            Topology::Instanced => &self.pipeline_instanced_msaa,
                            Topology::Custom(h, tr, ss, sd) => {
                                match self.custom_msaa_pipelines.get(&(h, tr, ss, sd)) {
                                    Some(p) => p,
                                    None => continue,
                                }
                            }
                            // gated by `use_msaa`; other topologies never occur here
                            _ => &self.pipeline_tri_msaa,
                        }
                    } else if linear_framebuffer {
                        match d.topology {
                            Topology::Triangle => &self.pipeline_tri_f16,
                            Topology::TriangleAlpha => &self.pipeline_tri_alpha_f16,
                            Topology::TriangleNoCull => &self.pipeline_tri_nocull_f16,
                            Topology::GlassDepth => &self.pipeline_glass_depth_f16,
                            Topology::GlassColor => &self.pipeline_glass_color_f16,
                            Topology::GlassTint => &self.pipeline_glass_tint_f16,
                            Topology::Sky => &self.pipeline_tri_sky_f16,
                            Topology::TriangleWire => &self.pipeline_tri_wire_f16,
                            Topology::Line => &self.pipeline_line_f16,
                            Topology::Point => &self.pipeline_point_f16,
                            Topology::PointAlpha => &self.pipeline_point_alpha_f16,
                            Topology::WideLine => &self.pipeline_wide_line_f16,
                            Topology::WideLineAlpha => &self.pipeline_wide_line_alpha_f16,
                            Topology::Sprite => &self.pipeline_sprite_f16,
                            Topology::Instanced => &self.pipeline_instanced_f16,
                            Topology::InstancedWire => &self.pipeline_instanced_wire_f16,
                            Topology::InstancedAlpha => &self.pipeline_instanced_alpha_f16,
                            Topology::Skinned => &self.pipeline_skinned_f16,
                            Topology::SkinnedWire => &self.pipeline_skinned_wire_f16,
                            Topology::SkinnedAlpha => &self.pipeline_skinned_alpha_f16,
                            Topology::Oit | Topology::Refract => continue,
                            // A shader that would not compile has no
                            // pipeline. Skipping the draw costs one object;
                            // indexing the map costs the process.
                            Topology::Custom(h, tr, ss, sd) => {
                                match self.custom_pipelines.get(&(h, true, tr, ss, sd)) {
                                    Some(p) => p,
                                    None => continue,
                                }
                            }
                        }
                    } else {
                        match d.topology {
                            Topology::Triangle => &self.pipeline_tri,
                            Topology::TriangleAlpha => &self.pipeline_tri_alpha,
                            Topology::TriangleNoCull => &self.pipeline_tri_nocull,
                            Topology::GlassDepth => &self.pipeline_glass_depth,
                            Topology::GlassColor => &self.pipeline_glass_color,
                            Topology::GlassTint => &self.pipeline_glass_tint,
                            Topology::Sky => &self.pipeline_tri_sky,
                            Topology::TriangleWire => &self.pipeline_tri_wire,
                            Topology::Line => &self.pipeline_line,
                            Topology::Point => &self.pipeline_point,
                            Topology::PointAlpha => &self.pipeline_point_alpha,
                            Topology::WideLine => &self.pipeline_wide_line,
                            Topology::WideLineAlpha => &self.pipeline_wide_line_alpha,
                            Topology::Sprite => &self.pipeline_sprite,
                            Topology::Instanced => &self.pipeline_instanced,
                            Topology::InstancedWire => &self.pipeline_instanced_wire,
                            Topology::InstancedAlpha => &self.pipeline_instanced_alpha,
                            Topology::Skinned => &self.pipeline_skinned,
                            Topology::SkinnedWire => &self.pipeline_skinned_wire,
                            Topology::SkinnedAlpha => &self.pipeline_skinned_alpha,
                            Topology::Oit | Topology::Refract => continue,
                            // A shader that would not compile has no
                            // pipeline. Skipping the draw costs one object;
                            // indexing the map costs the process.
                            Topology::Custom(h, tr, ss, sd) => {
                                match self.custom_pipelines.get(&(h, false, tr, ss, sd)) {
                                    Some(p) => p,
                                    None => continue,
                                }
                            }
                        }
                    };
                    let pipe = variant_pipeline(
                        &mut self.state_pipelines,
                        &self.map_variants,
                        &self.device,
                        &self.main_shader,
                        pipe,
                        d.map_variant,
                    );
                    let pipe = self.state_pipelines.variant(&self.device, &pipe, &d.state);
                    let pipe = match side_cull(d.topology.is_faces(), d.side) {
                        Some(cull) => self.state_pipelines.cull_variant(&self.device, &pipe, cull),
                        None => pipe,
                    };
                    pass.set_pipeline(&pipe);
                    last_side = d.side;
                    last_topology = Some(d.topology);
                    last_bias = d.polygon_offset;
                    last_state = d.state;
                    last_variant = d.map_variant;
                }
                let gm = if d.topology == Topology::Sprite {
                    &self.sprite_quad
                } else {
                    &self.geom_cache[&d.key].mesh
                };
                if d.topology.is_skinned() {
                    // Skinned pipeline uses the combined mesh+skin BGL at group(1).
                    if let Some(bg) = skin_data.get(d.skin_idx).and_then(|s| s.bg.as_ref()) {
                        pass.set_bind_group(1, bg, &[]);
                    }
                    pass.set_bind_group(2, &*self.per_mesh[i].2, &[]);
                    if let Some(skin_attrs) = self.skin_attr_cache.get(&d.key) {
                        pass.set_vertex_buffer(1, skin_attrs.slice(..));
                    }
                } else if let Topology::Custom(..) = d.topology {
                    // Custom-shader draws use the combined mesh+user bind group
                    // at @group(1) (mesh uniform + user data/storage), and a
                    // group 2 holding only what the shared vertex entry needs —
                    // a `ShaderMaterial` has no standard maps, and declaring the
                    // eight it cannot use is what put this layout over WebGPU's
                    // per-stage texture limit.
                    if let Some(bg) = &user_bgs[i] {
                        pass.set_bind_group(1, bg, &[]);
                    }
                    pass.set_bind_group(
                        2,
                        custom_morph_bgs[i]
                            .as_ref()
                            .unwrap_or(&self.custom_tex_bind_group),
                        &[],
                    );
                } else {
                    pass.set_bind_group(1, &self.per_mesh[i].1, &[]);
                    pass.set_bind_group(2, &*self.per_mesh[i].2, &[]);
                }
                // Wide lines: a capsule (six corners) per segment in the range —
                // an index span, or a vertex span, two to a segment — the segment's
                // start through slot 0 and its end a vertex further through slot 1.
                if d.topology.is_wide_lines() {
                    let Some(buf) = &gm.segment_buffer else {
                        continue;
                    };
                    let span = draw_span(d.range, gm.segment_vertices);
                    let segs = span.start / 2..span.end / 2;
                    if segs.is_empty() {
                        continue;
                    }
                    pass.set_vertex_buffer(0, buf.slice(..));
                    pass.set_vertex_buffer(1, buf.slice(GpuMesh::VERTEX_STRIDE..));
                    pass.draw(0..6, segs);
                    continue;
                }
                pass.set_vertex_buffer(0, gm.vertex_buffer.slice(..));
                // Points: a quad's six corners for each point in the range.
                if d.topology.is_points() {
                    pass.draw(0..6, draw_span(d.range, gm.vertex_count));
                    continue;
                }
                // A `ShaderMaterial` on an `InstancedMesh` reads the instance
                // buffer too, and its custom columns from slot 2.
                let custom_instanced = matches!(d.topology, Topology::Custom(..))
                    && shader_data[d.shader_idx].instanced;
                let instance_count = if d.topology.is_instanced() || custom_instanced {
                    pass.set_vertex_buffer(1, instance_buffers[d.instance_buf_idx].slice(..));
                    if custom_instanced {
                        let custom = &instance_buffers[shader_data[d.shader_idx].custom_buf];
                        pass.set_vertex_buffer(2, custom.slice(..));
                    }
                    d.instance_count
                } else {
                    1
                };
                // Wire topologies draw the line-list index buffer: polygon
                // outlines, or three.js's a,b,b,c,c,a per triangle without a
                // polygon table (`BufferGeometry::wireframe_indices`).
                // A partial span (a group, a draw range) has a line list of
                // its own (`BufferGeometry::wireframe_indices_in`).
                let (ib_opt, idx_range) = if d.topology.is_wire() {
                    match d.range {
                        Some((a, b)) => {
                            let version = self.geom_cache.get(&d.key).map(|e| e.version);
                            match version.and_then(|v| self.wire_span_cache.get(&(d.key, v, a, b)))
                            {
                                Some((buf, n)) => (Some(buf), 0..*n),
                                None => (None, 0..0),
                            }
                        }
                        None => (gm.wire_index_buffer.as_ref(), 0..gm.wire_index_count),
                    }
                } else {
                    (gm.index_buffer.as_ref(), draw_span(d.range, gm.index_count))
                };
                if let Some(ib) = ib_opt {
                    pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(idx_range, 0, 0..instance_count);
                } else if d.topology.is_wire() && d.range.is_some() {
                    // Nothing to outline in this span.
                } else {
                    pass.draw(draw_span(d.range, gm.vertex_count), 0..instance_count);
                }
            }
            drop(pass);

            // The encoded-blend pass (see `encoded_pass`): the blended meshes,
            // back to front, into a `Unorm` view of the target (or of the
            // multisampled colour) with the shader encoding, so the hardware
            // blend mixes sRGB-encoded values as three.js's canvas does.
            if let (true, Some(tex)) = (encoded_pass, encoded_target) {
                let unorm = tex.format().remove_srgb_suffix();
                // (Not on the scratch path: there the device cannot take it.)
                let target_unorm = (!encoded_scratch).then(|| {
                    tex.create_view(&wgpu::TextureViewDescriptor {
                        format: Some(unorm),
                        ..Default::default()
                    })
                });
                // Where the main pass drew: a pane's own multisampled target,
                // the shared multisampled colour, the refraction capture, or
                // the target itself.
                let pane = pane_key
                    .as_ref()
                    .and_then(|k| self.viewport_state.pane_target(k));
                let color_view = if encoded_scratch {
                    self.encoded_scratch.as_ref().map(|(_, _, v)| v)
                } else if let Some(pane) = pane {
                    pane.msaa_color_unorm.as_ref()
                } else if use_msaa {
                    self.msaa_color_unorm.as_ref()
                } else if refract_present {
                    self.ss_color.as_ref().and_then(|c| c.mip0_unorm.as_ref())
                } else {
                    target_unorm.as_ref()
                };
                let target_srgb = encoded_scratch.then(|| tex.create_view(&Default::default()));
                if let (Some(color_view), Some(enc)) =
                    (color_view, self.encoded_blend_pipelines.as_ref())
                {
                    if let Some(t) = &target_srgb {
                        self.encoded_copy_pass(&mut encoder, false, t, color_view);
                    }
                    let mut ep = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("threers encoded blend pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: color_view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: depth_pass_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    main_plan.apply(&mut ep);
                    ep.set_bind_group(0, &self.frame_bind_group, &[]);
                    ep.set_bind_group(3, &self.env_bind_group, &[]);
                    // Group 3 is the scene's environment unless a material has its own.
                    let mut env_bound: (Option<MaterialEnvKey>, bool) = (None, false);
                    for (i, d) in draws.iter().enumerate() {
                        // A custom draw on a trimmed-environment backend (WebGL2) binds
                        // the custom pipeline's own group 3; the full environment its
                        // standard counterpart binds does not match that pipeline layout.
                        let custom_env = self.custom_env_bind_group.is_some()
                            && matches!(d.topology, Topology::Custom(h, ..)
                        if !self.custom_lit.get(&h).copied().unwrap_or(false));
                        if (d.env_group, custom_env) != env_bound {
                            ep.set_bind_group(
                                3,
                                if custom_env {
                                    self.custom_env_bind_group.as_ref().unwrap()
                                } else {
                                    d.env_group
                                        .and_then(|k| self.material_env_groups.get(&k))
                                        .map_or(&self.env_bind_group, |(_, g)| g)
                                },
                                &[],
                            );
                            env_bound = (d.env_group, custom_env);
                        }
                        if !d.in_frustum || !d.topology.encodes() {
                            continue;
                        }
                        // Opted out of the refraction capture: the overlay
                        // pass draws it, over the glass.
                        if refract_present && !d.refract_capture {
                            continue;
                        }
                        let pipe = match d.topology {
                            Topology::TriangleAlpha => {
                                let (factor, units) =
                                    d.polygon_offset.unwrap_or((0f32.to_bits(), 0));
                                self.biased_pipelines.get(&(1, enc_variant, factor, units))
                            }
                            Topology::Sprite => Some(&enc[0]),
                            Topology::PointAlpha => Some(&enc[3]),
                            Topology::WideLineAlpha => Some(&enc[7]),
                            Topology::InstancedAlpha => Some(&enc[1]),
                            Topology::SkinnedAlpha => Some(&enc[2]),
                            Topology::GlassDepth => {
                                Some(match (&self.encoded_glass_msaa, use_msaa) {
                                    (Some(m), true) => &m[0],
                                    _ => &enc[4],
                                })
                            }
                            Topology::GlassColor => {
                                Some(match (&self.encoded_glass_msaa, use_msaa) {
                                    (Some(m), true) => &m[1],
                                    _ => &enc[5],
                                })
                            }
                            Topology::GlassTint => {
                                Some(match (&self.encoded_glass_msaa, use_msaa) {
                                    (Some(m), true) => &m[2],
                                    _ => &enc[6],
                                })
                            }
                            Topology::Custom(h, tr, ss, sd) => {
                                self.custom_encoded_pipelines.get(&(h, tr, ss, sd))
                            }
                            _ => None,
                        };
                        let Some(pipe) = pipe else { continue };
                        let pipe = variant_pipeline(
                            &mut self.state_pipelines,
                            &self.map_variants,
                            &self.device,
                            &self.main_shader,
                            pipe,
                            d.map_variant,
                        );
                        let pipe = self.state_pipelines.variant(&self.device, &pipe, &d.state);
                        let pipe = match side_cull(d.topology.is_faces(), d.side) {
                            Some(cull) => {
                                self.state_pipelines.cull_variant(&self.device, &pipe, cull)
                            }
                            None => pipe,
                        };
                        ep.set_pipeline(&pipe);
                        let gm = if d.topology == Topology::Sprite {
                            &self.sprite_quad
                        } else {
                            &self.geom_cache[&d.key].mesh
                        };
                        if d.topology.is_skinned() {
                            let Some(bg) = skin_data.get(d.skin_idx).and_then(|s| s.bg.as_ref())
                            else {
                                continue;
                            };
                            let Some(skin_attrs) = self.skin_attr_cache.get(&d.key) else {
                                continue;
                            };
                            ep.set_bind_group(1, bg, &[]);
                            ep.set_bind_group(2, &*self.per_mesh[i].2, &[]);
                            ep.set_vertex_buffer(1, skin_attrs.slice(..));
                        } else if let Topology::Custom(..) = d.topology {
                            let Some(bg) = &user_bgs[i] else { continue };
                            ep.set_bind_group(1, bg, &[]);
                            ep.set_bind_group(
                                2,
                                custom_morph_bgs[i]
                                    .as_ref()
                                    .unwrap_or(&self.custom_tex_bind_group),
                                &[],
                            );
                        } else {
                            ep.set_bind_group(1, &self.per_mesh[i].1, &[]);
                            ep.set_bind_group(2, &*self.per_mesh[i].2, &[]);
                        }
                        // Wide lines: a capsule (six corners) per segment in the range —
                        // an index span, or a vertex span, two to a segment — the segment's
                        // start through slot 0 and its end a vertex further through slot 1.
                        if d.topology.is_wide_lines() {
                            let Some(buf) = &gm.segment_buffer else {
                                continue;
                            };
                            let span = draw_span(d.range, gm.segment_vertices);
                            let segs = span.start / 2..span.end / 2;
                            if segs.is_empty() {
                                continue;
                            }
                            ep.set_vertex_buffer(0, buf.slice(..));
                            ep.set_vertex_buffer(1, buf.slice(GpuMesh::VERTEX_STRIDE..));
                            ep.draw(0..6, segs);
                            continue;
                        }
                        ep.set_vertex_buffer(0, gm.vertex_buffer.slice(..));
                        if d.topology.is_points() {
                            ep.draw(0..6, draw_span(d.range, gm.vertex_count));
                            continue;
                        }
                        let custom_instanced = matches!(d.topology, Topology::Custom(..))
                            && shader_data[d.shader_idx].instanced;
                        let instances = if d.topology.is_instanced() || custom_instanced {
                            ep.set_vertex_buffer(1, instance_buffers[d.instance_buf_idx].slice(..));
                            if custom_instanced {
                                let custom =
                                    &instance_buffers[shader_data[d.shader_idx].custom_buf];
                                ep.set_vertex_buffer(2, custom.slice(..));
                            }
                            d.instance_count
                        } else {
                            1
                        };
                        if let Some(ib) = &gm.index_buffer {
                            ep.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            ep.draw_indexed(draw_span(d.range, gm.index_count), 0, 0..instances);
                        } else {
                            ep.draw(draw_span(d.range, gm.vertex_count), 0..instances);
                        }
                    }
                    drop(ep);
                    if let Some(t) = &target_srgb {
                        self.encoded_copy_pass(&mut encoder, true, color_view, t);
                    }
                    // Multisampled: resolve again, in linear light, now that
                    // the samples hold the blended surfaces too.
                    if use_msaa {
                        let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("threers encoded blend resolve"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: main_color_view,
                                depth_slice: None,
                                resolve_target: msaa_resolve,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        });
                    }
                }
            }
        }

        // A pane drawn into its own multisampled target: copy its resolved
        // picture into the clear rectangle, and only there.
        if let Some(p) = &pane_msaa {
            let key = (
                p.region.width,
                p.region.height,
                self.color_format,
                self.msaa,
            );
            self.viewport_state.composite_pane(
                &mut encoder,
                target_view,
                self.color_format,
                &key,
                p.region,
            );
        }

        // Which OIT draws composite where. Without refraction, all of them
        // over the frame. With it, each where the refraction glass would see
        // it: into the capture just before the first group of glass that
        // lies wholly in front of it (so that glass refracts it), or, nearer
        // than every group, over the finished glass.
        let (refract_groups, refract_group) = {
            let mut order: Vec<usize> = (0..draws.len())
                .filter(|&i| {
                    refract_present && draws[i].in_frustum && draws[i].topology == Topology::Refract
                })
                .collect();
            order.sort_by(|&a, &b| {
                draws[a]
                    .view_z
                    .partial_cmp(&draws[b].view_z)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let groups = self.refraction_layers.clamp(1, order.len().max(1));
            let mut group = vec![usize::MAX; draws.len()];
            for (k, &i) in order.iter().enumerate() {
                group[i] = k * groups / order.len();
            }
            (groups, group)
        };
        let oit_phase: Vec<usize> = {
            // Each group's farthest object, as a view depth (more negative is
            // farther).
            let mut farthest = vec![f32::INFINITY; refract_groups];
            for (i, &g) in refract_group.iter().enumerate() {
                if g != usize::MAX {
                    farthest[g] = farthest[g].min(draws[i].view_z);
                }
            }
            draws
                .iter()
                .map(|d| {
                    if d.topology != Topology::Oit {
                        usize::MAX
                    } else if !refract_present {
                        0
                    } else {
                        (0..refract_groups)
                            .find(|&g| d.view_z <= farthest[g])
                            .unwrap_or(refract_groups)
                    }
                })
                .collect()
        };
        // One OIT phase: accumulate its draws against the depth so far, then
        // composite them into `$target` — the frame, or the refraction
        // capture — multisampled when `$allow_msaa` and the frame is.
        macro_rules! oit_phase {
            ($phase:expr, $target:expr, $allow_msaa:expr) => {{
        // --- Weighted-blended OIT: accumulate the OIT surfaces, then resolve
        // (composite) over the opaque image already in `target_view`. ---
        // Not during a probe, and this one bites hard. The order-independent
        // transparency targets are sized from `depth_size`, which a cube pass
        // has set to the face's 48 pixels — so a probe RESIZES them, and the
        // next real frame resolves a 48-pixel transparency buffer stretched
        // across the whole picture. It reads as the entire vehicle going
        // translucent, which looks like an alpha bug in the materials and is
        // nothing of the kind.
        if !self.cube_pass
            && !self.pipeline_oit.is_empty()
            && draws.iter().enumerate().any(|(i, d)| d.topology == Topology::Oit && oit_phase[i] == $phase)
        {
            let (tw, th) = self.depth_size;
            self.ensure_oit_targets(tw, th);
            // Multisampled when the opaque pass was: against its depth, into
            // samples resolved at the end of the pass.
            let msaa = $allow_msaa && self.last_frame_msaa && pane_msaa.is_none() && !self.pipeline_oit_msaa.is_empty();
            if msaa {
                self.ensure_oit_msaa_targets(tw, th);
            }
            let msaa_views = msaa
                .then(|| Some((self.oit_msaa_targets.as_ref()?, self.msaa_targets.as_ref()?)))
                .flatten();
            if let Some((_, _, _, accum_view, _, reveal_view, resolve_bg)) =
                self.oit_targets.as_ref()
            {
                // 1) Accumulation pass (accum cleared to 0, revealage cleared to 1).
                let oit_pipes = if msaa_views.is_some() { &self.pipeline_oit_msaa } else { &self.pipeline_oit };
                for (part, oit_pipe) in oit_pipes.iter().enumerate() {
                    let attachments = [
                            Some(wgpu::RenderPassColorAttachment {
                                view: msaa_views.map_or(accum_view, |((_, _, a, _), _)| a),
                                depth_slice: None,
                                resolve_target: msaa_views.map(|_| accum_view),
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                    store: wgpu::StoreOp::Store,
                                },
                            }),
                            Some(wgpu::RenderPassColorAttachment {
                                view: msaa_views.map_or(reveal_view, |((_, _, _, r), _)| r),
                                depth_slice: None,
                                resolve_target: msaa_views.map(|_| reveal_view),
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color {
                                        r: 1.0,
                                        g: 1.0,
                                        b: 1.0,
                                        a: 1.0,
                                    }),
                                    store: wgpu::StoreOp::Store,
                                },
                            }),
                    ];
                    let mut opass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("threers oit accum pass"),
                        color_attachments: if oit_pipes.len() == 1 {
                            &attachments
                        } else {
                            &attachments[part..=part]
                        },
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: msaa_views.map_or(&self.depth_view, |(_, (_, _, _, dv))| dv),
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    vp_plan.apply(&mut opass);
                    opass.set_pipeline(oit_pipe);
                    opass.set_bind_group(0, &self.frame_bind_group, &[]);
                    opass.set_bind_group(3, &self.env_bind_group, &[]);
                    // Group 3 is the scene's environment unless a material has its own.
                    let mut env_bound: (Option<MaterialEnvKey>, bool) = (None, false);
                    for (i, d) in draws.iter().enumerate() {
                        // A custom draw on a trimmed-environment backend (WebGL2) binds
                        // the custom pipeline's own group 3; the full environment its
                        // standard counterpart binds does not match that pipeline layout.
                        let custom_env = self.custom_env_bind_group.is_some()
                            && matches!(d.topology, Topology::Custom(h, ..)
                        if !self.custom_lit.get(&h).copied().unwrap_or(false));
                        if (d.env_group, custom_env) != env_bound {
                            opass.set_bind_group(
                                3,
                                if custom_env {
                                    self.custom_env_bind_group.as_ref().unwrap()
                                } else {
                                    d.env_group
                                        .and_then(|k| self.material_env_groups.get(&k))
                                        .map_or(&self.env_bind_group, |(_, g)| g)
                                },
                                &[],
                            );
                            env_bound = (d.env_group, custom_env);
                        }
                        if !d.in_frustum {
                            continue;
                        }
                        if d.topology != Topology::Oit || oit_phase[i] != $phase {
                            continue;
                        }
                        // The draw's group 2 is its material variant's, so
                        // the pipeline has to be too.
                        let pipe = variant_pipeline(
                            &mut self.state_pipelines,
                            &self.map_variants,
                            &self.device,
                            &self.main_shader,
                            oit_pipe,
                            d.map_variant,
                        );
                        let pipe = self.state_pipelines.variant(
                            &self.device,
                            &pipe,
                            &crate::materials::RenderState::DEFAULT,
                        );
                        opass.set_pipeline(&pipe);
                        let gm = &self.geom_cache[&d.key].mesh;
                        opass.set_bind_group(1, &self.per_mesh[i].1, &[]);
                        opass.set_bind_group(2, &*self.per_mesh[i].2, &[]);
                        opass.set_vertex_buffer(0, gm.vertex_buffer.slice(..));
                        if let Some(ib) = &gm.index_buffer {
                            opass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            opass.draw_indexed(draw_span(d.range, gm.index_count), 0, 0..1);
                        } else {
                            opass.draw(draw_span(d.range, gm.vertex_count), 0..1);
                        }
                    }
                }
                // 2) Resolve over the opaque image: what is behind times the
                //    layers' transmittance, then plus their light.
                {
                    let resolve_pipe = if msaa_views.is_some() {
                        &self.pipeline_oit_resolve_msaa
                    } else if linear_framebuffer {
                        &self.pipeline_oit_resolve_f16
                    } else {
                        &self.pipeline_oit_resolve
                    };
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("threers oit resolve pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: msaa_views.map_or($target, |(_, (_, _, cv, _))| cv),
                            depth_slice: None,
                            resolve_target: msaa_views.map(|_| $target),
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    if !vp_plan.legacy {
                        // Full-screen composite: confine it to this pane.
                        let sc = vp_plan.scissor;
                        rpass.set_scissor_rect(sc.x, sc.y, sc.width, sc.height);
                    }
                    rpass.set_bind_group(0, resolve_bg, &[]);
                    for pipe in resolve_pipe {
                        rpass.set_pipeline(pipe);
                        rpass.draw(0..3, 0..1);
                    }
                }
            }
        }

            }};
        }
        if !refract_present {
            oit_phase!(0, target_view, true);
        }

        // A multisampled opaque pass left its depth multisampled: resolve it
        // into the single-sampled one — the renderer's own, which refraction
        // glass tests against and its march copies, or a multisampled render
        // target's, which post-processing reads (`RenderTarget::samples`).
        let depth_out = self.last_frame_msaa
            && pane_msaa.is_none()
            && (refract_present || self.msaa_depth_out.is_some());
        if depth_out && (self.depth_resolve.is_none() || self.force_depth_replay) {
            // The camera's view-projection as the replay's pass uniform.
            if self.depth_replay_pass.is_none() {
                let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("threers depth replay uniform"),
                    size: std::mem::size_of::<ShadowPassGpu>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("threers depth replay bg"),
                    layout: &self.shadow_pass_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.frame_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: buffer.as_entire_binding(),
                        },
                    ],
                });
                self.depth_replay_pass = Some((buffer, bg));
            }
            let (replay_buffer, replay_bg) = self.depth_replay_pass.as_ref().unwrap();
            let replay_u = ShadowPassGpu {
                vp: frame_u.view_proj,
                light: [0.0; 4],
            };
            self.queue
                .write_buffer(replay_buffer, 0, bytemuck::bytes_of(&replay_u));
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("threers depth replay pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.depth_clear()),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_bind_group(0, replay_bg, &[0]);
            rp.set_bind_group(3, &self.shadow_env_bind_group, &[]);
            for (i, d) in draws.iter().enumerate() {
                if !d.in_frustum {
                    continue;
                }
                // What the opaque pass wrote depth for. Lines, points,
                // sprites and custom shaders are left out: they rarely
                // stand behind glass, and have no depth-only pipeline.
                let skinned = d.topology == Topology::Skinned;
                let instanced = d.topology == Topology::Instanced;
                if !(matches!(d.topology, Topology::Triangle | Topology::TriangleNoCull)
                    || skinned
                    || instanced)
                {
                    continue;
                }
                let cut =
                    d.alpha_test > 0.0 && (d.slots.map.is_some() || d.slots.alpha_map.is_some());
                let base = &self.depth_replay[usize::from(cut) * 3
                    + if skinned {
                        1
                    } else if instanced {
                        2
                    } else {
                        0
                    }];
                let gm = &self.geom_cache[&d.key].mesh;
                let mut instances = 0..1;
                if skinned {
                    let (Some(bg), Some(attrs)) = (
                        skin_data.get(d.skin_idx).and_then(|s| s.bg.as_ref()),
                        self.skin_attr_cache.get(&d.key),
                    ) else {
                        continue;
                    };
                    rp.set_bind_group(1, bg, &[]);
                    rp.set_vertex_buffer(1, attrs.slice(..));
                } else {
                    rp.set_bind_group(1, &self.per_mesh[i].1, &[]);
                    if instanced {
                        rp.set_vertex_buffer(1, instance_buffers[d.instance_buf_idx].slice(..));
                        instances = 0..d.instance_count;
                    }
                }
                let pipe = variant_pipeline(
                    &mut self.state_pipelines,
                    &self.map_variants,
                    &self.device,
                    &self.main_shader,
                    base,
                    d.map_variant,
                );
                // The material's depth state, and with reversed depth the
                // flipped compare, as the opaque pass drew it.
                let pipe = self.state_pipelines.variant(&self.device, &pipe, &d.state);
                let pipe = match side_cull(true, d.side) {
                    Some(cull) => self.state_pipelines.cull_variant(&self.device, &pipe, cull),
                    None => pipe,
                };
                rp.set_pipeline(&pipe);
                rp.set_bind_group(2, &*self.per_mesh[i].2, &[]);
                rp.set_vertex_buffer(0, gm.vertex_buffer.slice(..));
                if let Some(ib) = &gm.index_buffer {
                    rp.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    rp.draw_indexed(draw_span(d.range, gm.index_count), 0, instances);
                } else {
                    rp.draw(draw_span(d.range, gm.vertex_count), instances);
                }
            }
        } else if depth_out
            && let (Some((bgl, pipe)), Some((_, _, _, msaa_depth))) =
                (&self.depth_resolve, &self.msaa_targets)
        {
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("threers depth resolve bg"),
                layout: bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(msaa_depth),
                }],
            });
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("threers depth resolve pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.depth_clear()),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_pipeline(pipe);
            rp.set_bind_group(0, &bg, &[]);
            rp.draw(0..3, 0..1);
        }
        // --- Screen-space refraction glass: build the ss-color mip chain from
        // the opaque capture, blit the sharp capture to the frame target, then
        // draw the glass surface sampling it for refraction + reflection. ---
        //
        if refract_present {
            // The refracting objects back to front, in `refract_groups` groups
            // (worked out above, with the OIT phases).
            if refract_groups > 1 {
                self.ensure_ss_layer();
            }
            // Overlapping glass, in layers (`set_refraction_layers`): the
            // refracting objects back to front in up to that many groups,
            // each refracting a capture of everything drawn before it — the
            // opaque scene for the farthest, that and the farther glass for
            // the next. Every group but the last draws into a scratch target
            // copied back into the capture; the last draws into the frame.
            for group in 0..refract_groups {
                let last_group = group + 1 == refract_groups;
                if group > 0 {
                    // What the previous group drew is the next one's capture.
                    let (ssc, (_, _, layer_tex, _)) = (
                        self.ss_color.as_ref().unwrap(),
                        self.ss_layer.as_ref().unwrap(),
                    );
                    encoder.copy_texture_to_texture(
                        layer_tex.as_image_copy(),
                        ssc.texture.as_image_copy(),
                        wgpu::Extent3d {
                            width: ssc.width,
                            height: ssc.height,
                            depth_or_array_layers: 1,
                        },
                    );
                }
                // OIT surfaces behind this group's glass, into its capture.
                oit_phase!(
                    group,
                    &self.ss_color.as_ref().unwrap().mip_render_views[0],
                    false
                );
                let layer_target: &wgpu::TextureView = if last_group {
                    target_view
                } else {
                    &self.ss_layer.as_ref().unwrap().3
                };
                let ssc = self.ss_color.as_ref().unwrap();
                // Copy the opaque depth into the sampleable ss_depth so the glass
                // shader can ray-march it for refraction/reflection.
                if let (Some((_, _, _, ss_dview)), Some((bgl, pipe, sampler))) =
                    (&self.ss_depth, &self.depth_copy)
                {
                    // where depth cannot be copied: drawn across instead
                    let src = depth_src
                        .or(self.msaa_depth_out.as_ref())
                        .unwrap_or(&self.depth_texture);
                    let src_view = src.create_view(&wgpu::TextureViewDescriptor {
                        aspect: wgpu::TextureAspect::DepthOnly,
                        ..Default::default()
                    });
                    let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("threers depth copy bg"),
                        layout: bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&src_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(sampler),
                            },
                        ],
                    });
                    let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("threers depth copy pass"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: ss_dview,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(self.depth_clear()),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    rp.set_pipeline(pipe);
                    rp.set_bind_group(0, &bg, &[]);
                    rp.draw(0..3, 0..1);
                } else if let Some((_, _, ss_dtex, _)) = &self.ss_depth {
                    let src = depth_src
                        .or(self.msaa_depth_out.as_ref())
                        .unwrap_or(&self.depth_texture);
                    encoder.copy_texture_to_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: src,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::DepthOnly,
                        },
                        wgpu::TexelCopyTextureInfo {
                            texture: ss_dtex,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::DepthOnly,
                        },
                        wgpu::Extent3d {
                            width: ssc.width,
                            height: ssc.height,
                            depth_or_array_layers: 1,
                        },
                    );
                }
                let ss_depth_view = &self.ss_depth.as_ref().unwrap().3;
                let ss_back_depth_view = &self.ss_back_depth.as_ref().unwrap().3;
                // Two-surface: render glass back faces (cull front) into ss_back_depth
                // so the shader can derive per-pixel glass thickness.
                {
                    let mut bp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("threers ss back-depth pass"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: ss_back_depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(self.depth_clear()),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    bp.set_pipeline(&self.pipeline_ss_back_depth);
                    bp.set_bind_group(0, &self.frame_bind_group, &[]);
                    for (i, d) in draws.iter().enumerate() {
                        if !d.in_frustum {
                            continue;
                        }
                        if d.topology != Topology::Refract || refract_group[i] != group {
                            continue;
                        }
                        let pipe = variant_pipeline(
                            &mut self.state_pipelines,
                            &self.map_variants,
                            &self.device,
                            &self.main_shader,
                            &self.pipeline_ss_back_depth,
                            d.map_variant,
                        );
                        let pipe = self.state_pipelines.variant(
                            &self.device,
                            &pipe,
                            &crate::materials::RenderState::DEFAULT,
                        );
                        bp.set_pipeline(&pipe);
                        let gm = &self.geom_cache[&d.key].mesh;
                        bp.set_bind_group(1, &self.per_mesh[i].1, &[]);
                        bp.set_bind_group(2, &*self.per_mesh[i].2, &[]);
                        bp.set_vertex_buffer(0, gm.vertex_buffer.slice(..));
                        if let Some(ib) = &gm.index_buffer {
                            bp.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            bp.draw_indexed(draw_span(d.range, gm.index_count), 0, 0..1);
                        } else {
                            bp.draw(draw_span(d.range, gm.vertex_count), 0..1);
                        }
                    }
                }
                let blit_pipe = if linear_framebuffer {
                    &self.pipeline_ss_blit_f16
                } else {
                    &self.pipeline_ss_blit
                };
                // 1) Downsample chain: build mip m by box-filtering mip m-1.
                for m in 1..ssc.mips as usize {
                    let src_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("threers ss mip src"),
                        layout: &self.ss_blit_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &ssc.mip_sample_views[m - 1],
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&self.ss_glass_sampler),
                            },
                        ],
                    });
                    let mut mp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("threers ss mip gen"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &ssc.mip_render_views[m],
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    mp.set_pipeline(blit_pipe);
                    mp.set_bind_group(0, &src_bg, &[]);
                    mp.draw(0..3, 0..1);
                }
                // 2) Blit the sharp capture (mip 0) to the frame target so non-glass
                //    pixels show through; the glass draws over it next.
                {
                    let blit_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("threers ss blit src"),
                        layout: &self.ss_blit_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &ssc.mip_sample_views[0],
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&self.ss_glass_sampler),
                            },
                        ],
                    });
                    let mut bp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("threers ss blit"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: layer_target,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    bp.set_pipeline(blit_pipe);
                    bp.set_bind_group(0, &blit_bg, &[]);
                    bp.draw(0..3, 0..1);
                }
                // 3) group(3) bind group with the real ss-color at binding 9 (env cube
                //    + shadows mirror the frame's env bind group).
                // The scene environment as the frame's group reads it: prefiltered
                // (earlier this frame).
                let scene_env_cube = scene
                    .environment
                    .as_ref()
                    .map(|e| self.prefiltered_cached(e));
                let cube_uv_view: &wgpu::TextureView = if let Some(env) = &scene_env_cube {
                    self.cube_cache
                        .get(&cube_cache_key(env))
                        .and_then(|gc| gc.cube_uv_view.as_ref())
                        .unwrap_or(&self.default_cube_uv_view)
                } else {
                    &self.default_cube_uv_view
                };
                let env_cube_view: &wgpu::TextureView =
                    if let Some(rt_id) = scene.environment_cube_rt {
                        self.cube_rt_view_cache
                            .get(&rt_id)
                            .unwrap_or(&self.default_env_cube.view)
                    } else if let Some(env) = &scene_env_cube {
                        &self.cube_cache[&cube_cache_key(env)].view
                    } else {
                        &self.default_env_cube.view
                    };
                // 4a) Each light's shadow on the glass, for the pass below, which
                //     has no room for the maps (`fs_glass_shadow`). It also lays
                //     down the glass depth that pass tests `LessEqual` against.
                let glass_shadow_view = &self.ss_glass_shadow.as_ref().unwrap().1;
                {
                    let mut sp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("threers glass shadow pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: glass_shadow_view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &self.depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    sp.set_bind_group(0, &self.frame_bind_group, &[]);
                    sp.set_bind_group(3, &self.env_bind_group, &[]);
                    for (i, d) in draws.iter().enumerate() {
                        if !d.in_frustum
                            || d.topology != Topology::Refract
                            || refract_group[i] != group
                        {
                            continue;
                        }
                        let pipe = variant_pipeline(
                            &mut self.state_pipelines,
                            &self.map_variants,
                            &self.device,
                            &self.main_shader,
                            &self.pipeline_glass_shadow,
                            d.map_variant,
                        );
                        let pipe = self.state_pipelines.variant(
                            &self.device,
                            &pipe,
                            &crate::materials::RenderState::DEFAULT,
                        );
                        sp.set_pipeline(&pipe);
                        let gm = &self.geom_cache[&d.key].mesh;
                        sp.set_bind_group(1, &self.per_mesh[i].1, &[]);
                        sp.set_bind_group(2, &*self.per_mesh[i].2, &[]);
                        sp.set_vertex_buffer(0, gm.vertex_buffer.slice(..));
                        if let Some(ib) = &gm.index_buffer {
                            sp.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            sp.draw_indexed(draw_span(d.range, gm.index_count), 0, 0..1);
                        } else {
                            sp.draw(draw_span(d.range, gm.vertex_count), 0..1);
                        }
                    }
                }
                // Built by hand rather than through `env_bind_group!`: this group
                // has the capture where the macro puts the shadow maps, and the
                // shadow maps are exactly what was dropped to make room for it.
                let glass_env_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("threers glass env bg"),
                    layout: &self.env_bgl_glass,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(env_cube_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.env_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: wgpu::BindingResource::TextureView(cube_uv_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 9,
                            resource: wgpu::BindingResource::TextureView(&ssc.all_mips_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 10,
                            resource: wgpu::BindingResource::Sampler(&self.ss_glass_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 11,
                            resource: wgpu::BindingResource::TextureView(ss_depth_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 12,
                            resource: wgpu::BindingResource::TextureView(ss_back_depth_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 13,
                            resource: self.env_uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 15,
                            resource: wgpu::BindingResource::TextureView(glass_shadow_view),
                        },
                    ],
                });
                // 4) Glass surface pass (Less + depth-write, opaque composite).
                let glass_pipe = if linear_framebuffer {
                    &self.pipeline_ss_glass_f16
                } else {
                    &self.pipeline_ss_glass
                };
                let mut gp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("threers ss glass pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: layer_target,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                gp.set_pipeline(glass_pipe);
                gp.set_bind_group(0, &self.frame_bind_group, &[]);
                gp.set_bind_group(3, &glass_env_bg, &[]);
                for (i, d) in draws.iter().enumerate() {
                    if !d.in_frustum {
                        continue;
                    }
                    if d.topology != Topology::Refract || refract_group[i] != group {
                        continue;
                    }
                    let pipe = variant_pipeline(
                        &mut self.state_pipelines,
                        &self.map_variants,
                        &self.device,
                        &self.main_shader,
                        glass_pipe,
                        d.map_variant,
                    );
                    let pipe = self.state_pipelines.variant(
                        &self.device,
                        &pipe,
                        &crate::materials::RenderState::DEFAULT,
                    );
                    gp.set_pipeline(&pipe);
                    let gm = &self.geom_cache[&d.key].mesh;
                    gp.set_bind_group(1, &self.per_mesh[i].1, &[]);
                    gp.set_bind_group(2, &*self.per_mesh[i].2, &[]);
                    gp.set_vertex_buffer(0, gm.vertex_buffer.slice(..));
                    if let Some(ib) = &gm.index_buffer {
                        gp.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        gp.draw_indexed(draw_span(d.range, gm.index_count), 0, 0..1);
                    } else {
                        gp.draw(draw_span(d.range, gm.vertex_count), 0..1);
                    }
                }
                // Custom shaders that asked for the capture draw here too, with the
                // real ss-colour/ss-depth at group(3) — the whole point of the flag.
                for (i, d) in draws.iter().enumerate() {
                    let Topology::Custom(h, tr, true, sd) = d.topology else {
                        continue;
                    };
                    if !last_group {
                        continue;
                    }
                    let Some(bg) = &user_bgs[i] else { continue };
                    let Some(pipe) =
                        self.custom_pipelines
                            .get(&(h, linear_framebuffer, tr, true, sd))
                    else {
                        continue;
                    };
                    let pipe = variant_pipeline(
                        &mut self.state_pipelines,
                        &self.map_variants,
                        &self.device,
                        &self.main_shader,
                        pipe,
                        d.map_variant,
                    );
                    let pipe = self.state_pipelines.variant(&self.device, &pipe, &d.state);
                    gp.set_pipeline(&pipe);
                    gp.set_bind_group(1, bg, &[]);
                    gp.set_bind_group(
                        2,
                        custom_morph_bgs[i]
                            .as_ref()
                            .unwrap_or(&self.custom_tex_bind_group),
                        &[],
                    );
                    let gm = &self.geom_cache[&d.key].mesh;
                    gp.set_vertex_buffer(0, gm.vertex_buffer.slice(..));
                    let sdd = &shader_data[d.shader_idx];
                    let instances = if sdd.instanced {
                        gp.set_vertex_buffer(1, instance_buffers[d.instance_buf_idx].slice(..));
                        gp.set_vertex_buffer(2, instance_buffers[sdd.custom_buf].slice(..));
                        0..d.instance_count
                    } else {
                        0..1
                    };
                    if let Some(ib) = &gm.index_buffer {
                        gp.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        gp.draw_indexed(draw_span(d.range, gm.index_count), 0, instances);
                    } else {
                        gp.draw(draw_span(d.range, gm.vertex_count), instances);
                    }
                }

                drop(gp); // end glass pass before the overlay pass reuses the encoder
            }
            // OIT surfaces in front of all the glass, over it.
            oit_phase!(refract_groups, target_view, false);

            // Overlay pass: objects opted out of glass capture draw here, over the
            // glass, depth-tested against it (so they stay crisp and un-refracted).
            if draws.iter().any(|d| !d.refract_capture) {
                let mut op = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("threers glass overlay pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                op.set_bind_group(0, &self.frame_bind_group, &[]);
                op.set_bind_group(3, &self.env_bind_group, &[]);
                let mut last: Option<(
                    Topology,
                    crate::materials::RenderState,
                    Option<super::super::material_maps::MapVariant>,
                    u32,
                )> = None;
                // Group 3 is the scene's environment unless a material has its own.
                let mut env_bound: (Option<MaterialEnvKey>, bool) = (None, false);
                for (i, d) in draws.iter().enumerate() {
                    // A custom draw on a trimmed-environment backend (WebGL2) binds
                    // the custom pipeline's own group 3; the full environment its
                    // standard counterpart binds does not match that pipeline layout.
                    let custom_env = self.custom_env_bind_group.is_some()
                        && matches!(d.topology, Topology::Custom(h, ..)
                        if !self.custom_lit.get(&h).copied().unwrap_or(false));
                    if (d.env_group, custom_env) != env_bound {
                        op.set_bind_group(
                            3,
                            if custom_env {
                                self.custom_env_bind_group.as_ref().unwrap()
                            } else {
                                d.env_group
                                    .and_then(|k| self.material_env_groups.get(&k))
                                    .map_or(&self.env_bind_group, |(_, g)| g)
                            },
                            &[],
                        );
                        env_bound = (d.env_group, custom_env);
                    }
                    if !d.in_frustum {
                        continue;
                    }
                    if d.refract_capture {
                        continue;
                    }
                    let pipe = match (linear_framebuffer, d.topology) {
                        (false, Topology::Triangle) => &self.pipeline_tri,
                        (false, Topology::TriangleAlpha) => &self.pipeline_tri_alpha,
                        (false, Topology::TriangleNoCull) => &self.pipeline_tri_nocull,
                        (false, Topology::Line) => &self.pipeline_line,
                        (false, Topology::Sprite) => &self.pipeline_sprite,
                        (false, Topology::Point) => &self.pipeline_point,
                        (false, Topology::PointAlpha) => &self.pipeline_point_alpha,
                        (false, Topology::WideLine) => &self.pipeline_wide_line,
                        (false, Topology::WideLineAlpha) => &self.pipeline_wide_line_alpha,
                        (false, Topology::Instanced) => &self.pipeline_instanced,
                        (false, Topology::InstancedAlpha) => &self.pipeline_instanced_alpha,
                        (true, Topology::Triangle) => &self.pipeline_tri_f16,
                        (true, Topology::TriangleAlpha) => &self.pipeline_tri_alpha_f16,
                        (true, Topology::TriangleNoCull) => &self.pipeline_tri_nocull_f16,
                        (true, Topology::Line) => &self.pipeline_line_f16,
                        (true, Topology::Sprite) => &self.pipeline_sprite_f16,
                        (true, Topology::Point) => &self.pipeline_point_f16,
                        (true, Topology::PointAlpha) => &self.pipeline_point_alpha_f16,
                        (true, Topology::WideLine) => &self.pipeline_wide_line_f16,
                        (true, Topology::WideLineAlpha) => &self.pipeline_wide_line_alpha_f16,
                        (true, Topology::Instanced) => &self.pipeline_instanced_f16,
                        (true, Topology::InstancedAlpha) => &self.pipeline_instanced_alpha_f16,
                        _ => continue, // skinned/custom/etc not supported as overlay
                    };
                    if last != Some((d.topology, d.state, d.map_variant, d.side)) {
                        let pipe = variant_pipeline(
                            &mut self.state_pipelines,
                            &self.map_variants,
                            &self.device,
                            &self.main_shader,
                            pipe,
                            d.map_variant,
                        );
                        let pipe = self.state_pipelines.variant(&self.device, &pipe, &d.state);
                        let pipe = match side_cull(d.topology.is_faces(), d.side) {
                            Some(cull) => {
                                self.state_pipelines.cull_variant(&self.device, &pipe, cull)
                            }
                            None => pipe,
                        };
                        op.set_pipeline(&pipe);
                        last = Some((d.topology, d.state, d.map_variant, d.side));
                    }
                    let gm = if d.topology == Topology::Sprite {
                        &self.sprite_quad
                    } else {
                        &self.geom_cache[&d.key].mesh
                    };
                    op.set_bind_group(1, &self.per_mesh[i].1, &[]);
                    op.set_bind_group(2, &*self.per_mesh[i].2, &[]);
                    // Wide lines: a capsule (six corners) per segment in the range —
                    // an index span, or a vertex span, two to a segment — the segment's
                    // start through slot 0 and its end a vertex further through slot 1.
                    if d.topology.is_wide_lines() {
                        let Some(buf) = &gm.segment_buffer else {
                            continue;
                        };
                        let span = draw_span(d.range, gm.segment_vertices);
                        let segs = span.start / 2..span.end / 2;
                        if segs.is_empty() {
                            continue;
                        }
                        op.set_vertex_buffer(0, buf.slice(..));
                        op.set_vertex_buffer(1, buf.slice(GpuMesh::VERTEX_STRIDE..));
                        op.draw(0..6, segs);
                        continue;
                    }
                    op.set_vertex_buffer(0, gm.vertex_buffer.slice(..));
                    if d.topology.is_points() {
                        op.draw(0..6, draw_span(d.range, gm.vertex_count));
                        continue;
                    }
                    let inst = if d.topology.is_instanced() {
                        op.set_vertex_buffer(1, instance_buffers[d.instance_buf_idx].slice(..));
                        d.instance_count
                    } else {
                        1
                    };
                    if let Some(ib) = &gm.index_buffer {
                        op.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        op.draw_indexed(draw_span(d.range, gm.index_count), 0, 0..inst);
                    } else {
                        op.draw(draw_span(d.range, gm.vertex_count), 0..inst);
                    }
                }
            }
        }

        // --- Temporal anti-aliasing: reproject + blend the rendered frame with
        // the accumulated history, then blit the result back to the target. ---
        if taa_active {
            let (depth_tex, color_tex) = taa_src.unwrap();
            let (w, h) = self.depth_size;
            // ACCUMULATE IN FLOAT, NOT IN THE FRAMEBUFFER'S FORMAT.
            //
            // A running mean folds each new sample in at `1/(n+1)`, and in an
            // eight-bit history that is below the quantum long before the mean
            // has settled: by the thirty-second sample the increment is a
            // 255th of the difference, which rounds to nothing, and every
            // sample after it is discarded. The mean stops improving while the
            // caller keeps paying for samples. Sixteen float bits cost one more
            // byte a pixel at output size and make every sample count.
            //
            // Only when converging. Ordinary TAA blends at a fixed 0.9 and
            // never gets near the quantum, and it is the interactive path,
            // where the bandwidth is worth more than the precision.
            let hist_format = if self.taa_converge {
                wgpu::TextureFormat::Rgba16Float
            } else {
                ss_format
            };
            self.ensure_taa_targets(w, h, hist_format);
            let first = self.taa_frame == 0;
            // Camera movement, carried for a caller that can distinguish its
            // own pinned geometry. This pass cannot: see TAA_SHADER.
            let cam = camera.position();
            let d = [
                cam.x - self.taa_prev_cam[0],
                cam.y - self.taa_prev_cam[1],
                cam.z - self.taa_prev_cam[2],
            ];
            let taa_u = TaaUniforms {
                inv_view_proj: taa_inv_vp,
                prev_view_proj: self.taa_prev_view_proj,
                // Converging: `n/(n+1)` makes the history the mean of every
                // sample so far. Otherwise the usual fixed feedback.
                params: [
                    w as f32,
                    h as f32,
                    if self.taa_converge {
                        self.taa_frame as f32 / (self.taa_frame as f32 + 1.0)
                    } else {
                        0.9
                    },
                    if first { 1.0 } else { 0.0 },
                ],
                // w: 0 feedback, 1 converging over sRGB-encoded bytes, 2
                // converging over linear light. The converge branch needs to
                // know which, because a running mean of encoded bytes is a mean
                // over a curve: see TAA_SHADER.
                cam_delta: [
                    d[0],
                    d[1],
                    d[2],
                    match (self.taa_converge, shader_outputs_linear) {
                        (false, _) => 0.0,
                        (true, false) => 1.0,
                        (true, true) => 2.0,
                    },
                ],
                flags: [
                    if self.state_pipelines.reversed() {
                        1.0
                    } else {
                        0.0
                    },
                    0.0,
                    0.0,
                    0.0,
                ],
            };
            self.queue
                .write_buffer(&self.taa_uniform, 0, bytemuck::bytes_of(&taa_u));
            let cur_view = color_tex.create_view(&wgpu::TextureViewDescriptor::default());
            let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
            // frame N reads what N-1 wrote, writes the other buffer.
            let even = self.taa_frame.is_multiple_of(2);
            if self.taa_bind_cache.is_none() {
                let (_, _, _, a_view, _, b_view) = self.taa_history.as_ref().unwrap();
                let mk_resolve = |read: &wgpu::TextureView| {
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("threers taa bg"),
                        layout: &self.taa_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&cur_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(read),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: wgpu::BindingResource::TextureView(&depth_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: wgpu::BindingResource::Sampler(&self.taa_sampler),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: self.taa_uniform.as_entire_binding(),
                            },
                        ],
                    })
                };
                let mk_blit = |write: &wgpu::TextureView| {
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("threers taa blit bg"),
                        layout: &self.ss_blit_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(write),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&self.ss_glass_sampler),
                            },
                        ],
                    })
                };
                // Index by `taa_frame % 2`: even frames read b and write a.
                self.taa_bind_cache = Some(TaaBindCache {
                    resolve: [mk_resolve(b_view), mk_resolve(a_view)],
                    blit: [mk_blit(a_view), mk_blit(b_view)],
                });
            }
            let parity = usize::from(!even);
            let (a_view, b_view) = {
                let (_, _, _, av, _, bv) = self.taa_history.as_ref().unwrap();
                (av, bv)
            };
            let write_view = if even { a_view } else { b_view };
            let resolve_bg = &self.taa_bind_cache.as_ref().unwrap().resolve[parity];
            // The resolve draws into the history, so its target format is the
            // history's — float whenever converging, whatever the framebuffer.
            let taa_pipe = if linear_framebuffer || self.taa_converge {
                &self.pipeline_taa_f16
            } else {
                &self.pipeline_taa
            };
            {
                let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("threers taa resolve"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: write_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                rp.set_pipeline(taa_pipe);
                rp.set_bind_group(0, resolve_bg, &[]);
                rp.draw(0..3, 0..1);
            }
            // Blit the resolved history back to the frame target — unless this
            // is an intermediate sample of an accumulation run, in which case
            // the mean is in the history buffer and nothing will read the
            // target before the next sample overwrites it anyway.
            if self.taa_blit {
                let blit_bg = &self.taa_bind_cache.as_ref().unwrap().blit[parity];
                let blit_pipe = if linear_framebuffer {
                    &self.pipeline_ss_blit_f16
                } else {
                    &self.pipeline_ss_blit
                };
                {
                    let mut bp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("threers taa blit"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: target_view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    bp.set_pipeline(blit_pipe);
                    bp.set_bind_group(0, blit_bg, &[]);
                    bp.draw(0..3, 0..1);
                }
            }
        }
        // Roll the motion-blur matrices on every frame -- but a PROBE IS NOT A
        // FRAME. An environment cube is six renders from a camera that is not
        // the picture's, and rolling on those leaves `mb_last_vp` holding a
        // 90-degree view from somewhere else. The next real frame then blurs
        // along the difference between the two, which is enormous: the vehicle
        // smears into a translucent double of itself and streaks run across the
        // whole image. It reads exactly like a transparency bug, and was chased
        // as one for a long time -- through the OIT buffers, the cube format,
        // the PMREM prefilter and the probe's brightness -- none of which it
        // ever was.
        if !self.cube_pass {
            self.mb_inv_vp = taa_inv_vp;
            self.mb_prev_vp = self.mb_last_vp;
            self.mb_last_vp = taa_unjittered_vp;
        }
        if taa_active && !self.cube_pass {
            // (separate block so the history-view borrows above have ended)
            self.taa_prev_view_proj = taa_unjittered_vp;
            {
                let c = camera.position();
                self.taa_prev_cam = [c.x, c.y, c.z];
            }
            self.taa_frame += 1;
        }

        if let Some(t) = &self.timer {
            t.resolve(&mut encoder);
        }
        if update_shadows && let Some(t) = &self.shadow_timer {
            t.resolve(&mut encoder);
        }
        self.queue.submit(Some(encoder.finish()));
        // After submit, so the map is queued behind this frame's work. Never
        // blocks — see `gpu_timer`.
        if let Some(t) = &mut self.timer {
            let device = self.device.clone();
            t.poll(&device);
        }
        if let Some(t) = &mut self.shadow_timer {
            let device = self.device.clone();
            t.poll(&device);
        }
        let _ = (&self.frame_bgl,);
    }
}
