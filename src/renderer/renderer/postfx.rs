use super::*;

impl Renderer {
    /// Ensure the multisampled color+depth targets exist at `w×h` for the MSAA pass.
    /// The `Unorm` scratch target and the copy pipelines for the encoded-blend
    /// pass on a device without view-format reinterpretation (WebGL2).
    pub(super) fn ensure_encoded_scratch(&mut self, w: u32, h: u32) {
        let unorm = self.color_format.remove_srgb_suffix();
        if !matches!(self.encoded_scratch, Some((cw, ch, _)) if cw == w && ch == h) {
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers encoded blend scratch"),
                size: wgpu::Extent3d {
                    width: w.max(1),
                    height: h.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.encoded_scratch = Some((w, h, tex.create_view(&Default::default())));
        }
        if self.encoded_copy.is_some() {
            return;
        }
        // `textureLoad` of the sRGB target decodes; `fs_in` encodes again, so
        // the `Unorm` scratch holds the target's bytes. `fs_out` reads the
        // scratch's bytes raw and decodes them, and the sRGB target encodes
        // on write: the bytes come back as they are. Both directions are
        // exact for 8-bit values.
        const COPY: &str = crate::wgsl!(
            r#"
@group(0) @binding(0) var src : texture_2d<f32>;
@vertex
fn vs(@builtin(vertex_index) i : u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}
@fragment
fn fs_in(@builtin(position) p : vec4<f32>) -> @location(0) vec4<f32> {
    let v = textureLoad(src, vec2<i32>(p.xy), 0);
    let c = clamp(v.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let e = select(pow(c, vec3<f32>(1.0 / 2.4)) * 1.055 - vec3<f32>(0.055), c * 12.92, c <= vec3<f32>(0.0031308));
    return vec4<f32>(e, v.a);
}
@fragment
fn fs_out(@builtin(position) p : vec4<f32>) -> @location(0) vec4<f32> {
    let v = textureLoad(src, vec2<i32>(p.xy), 0);
    let d = select(pow((v.rgb + 0.055) / 1.055, vec3<f32>(2.4)), v.rgb / 12.92, v.rgb <= vec3<f32>(0.04045));
    return vec4<f32>(d, v.a);
}
"#
        );
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("threers encoded blend copy"),
                source: wgpu::ShaderSource::Wgsl(COPY.into()),
            });
        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("threers encoded blend copy bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("threers encoded blend copy layout"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
        let pipe = |entry: &str, format: wgpu::TextureFormat| {
            self.state_pipelines.source().create_now(
                &self.device,
                &wgpu::RenderPipelineDescriptor {
                    label: Some("threers encoded blend copy"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some("vs"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some(entry),
                        targets: &[Some(wgpu::ColorTargetState {
                            format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                },
            )
        };
        let copy_in = pipe("fs_in", unorm);
        let copy_out = pipe("fs_out", self.color_format);
        self.encoded_copy = Some((bgl, copy_in, copy_out));
    }

    /// One fullscreen copy for the encoded-blend scratch (see
    /// [`ensure_encoded_scratch`](Self::ensure_encoded_scratch)).
    pub(super) fn encoded_copy_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        out: bool,
        src: &wgpu::TextureView,
        dst: &wgpu::TextureView,
    ) {
        let Some((bgl, copy_in, copy_out)) = &self.encoded_copy else {
            return;
        };
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers encoded blend copy bg"),
            layout: bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(src),
            }],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("threers encoded blend copy pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst,
                depth_slice: None,
                resolve_target: None,
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
        pass.set_pipeline(if out { copy_out } else { copy_in });
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Apply a post-fx pass from a registered render-target id (wasm path).
    ///
    /// Same as [`Self::apply_postfx`] but resolves `input_rt_id` internally so
    /// the caller does not hold a borrow against `&mut self`. When `additive`
    /// is true, output is blended onto existing framebuffer pixels.
    ///
    /// `kind` selects the WGSL branch in [`super::super::shader::postfx_shader`].
    /// `time` carries effect-specific data (e.g. glitch RNG seed). `params2` /
    /// `params3` are four-vectors whose meaning depends on `kind`.
    ///
    /// Every kind keeps the picture the right way up, into a render target
    /// (`output_rt_id != 0`) and onto the canvas (`0`) alike, so passes chain
    /// in any number without compensating flips.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_postfx_by_id(
        &mut self,
        input_rt_id: u32,
        output_rt_id: u32,
        output_view: &wgpu::TextureView,
        depth_rt_id: u32,
        normal_rt_id: u32,
        kind: u32,
        time: f32,
        params2: [f32; 4],
        params3: [f32; 4],
        width: u32,
        height: u32,
        additive: bool,
        camera: PostFxCamera,
    ) {
        let output_format = if output_rt_id != 0 {
            self.rt_registry
                .get(&output_rt_id)
                .map(|rt| rt.format)
                .unwrap_or(self.color_format)
        } else {
            self.color_format
        };
        let input_view = match self.rt_view_cache.remove(&input_rt_id) {
            Some(v) => v,
            _ => {
                if let Some(rt) = self.rt_registry.get(&input_rt_id) {
                    rt.color_texture.create_view(&wgpu::TextureViewDescriptor {
                        label: Some("threers postfx input fallback"),
                        format: Some(rt_sample_format(rt.format)),
                        ..Default::default()
                    })
                } else {
                    return;
                }
            }
        };
        let depth_owned = if depth_rt_id != 0 && depth_rt_id != input_rt_id {
            self.rt_view_cache.remove(&depth_rt_id).or_else(|| {
                self.rt_registry.get(&depth_rt_id).map(|rt| {
                    rt.color_texture.create_view(&wgpu::TextureViewDescriptor {
                        label: Some("threers postfx depth fallback"),
                        format: Some(rt_sample_format(rt.format)),
                        ..Default::default()
                    })
                })
            })
        } else {
            None
        };
        let normal_owned = if normal_rt_id != 0 {
            self.rt_view_cache.remove(&normal_rt_id)
        } else {
            None
        };
        let depth_ref = depth_owned.as_ref().unwrap_or(&input_view);
        let input_is_f16 = self
            .rt_registry
            .get(&input_rt_id)
            .map(|rt| rt.format == wgpu::TextureFormat::Rgba16Float)
            .unwrap_or(false);
        // Kind 0 copies stored RT texels 1:1. Default: linear (non-sRGB-decode)
        // view so textureLoad returns the gamma-encoded bytes the main shader
        // wrote (matches direct-to-canvas). params2.y > 0.5 selects the legacy
        // sRGB-decode view for OutlinePass scene copy (three.js darkens there).
        let kind0_legacy = kind == 0 && params2[1] > 0.5;
        let output_is_f16 = output_format == wgpu::TextureFormat::Rgba16Float;
        let kind0_byte_copy = kind == 0 && !kind0_legacy && !input_is_f16 && !output_is_f16;
        let use_linear_input = if kind == 0 {
            input_is_f16 && !kind0_legacy
        } else {
            (matches!(kind, 13 | 15) && !kind0_legacy)
                || input_is_f16
                || matches!(kind, 16 | 18 | 19)
        };
        let use_linear_depth = matches!(kind, 11 | 12 | 14);
        let input_linear = if use_linear_input {
            self.rt_registry.get(&input_rt_id).map(|rt| {
                rt.color_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("threers postfx linear input"),
                    format: Some(rt.format),
                    ..Default::default()
                })
            })
        } else {
            None
        };
        let input_raw = if kind0_byte_copy {
            self.rt_registry.get(&input_rt_id).map(|rt| {
                rt.color_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("threers postfx raw input"),
                    format: Some(rt.format),
                    ..Default::default()
                })
            })
        } else {
            None
        };
        let depth_linear = if use_linear_depth && depth_rt_id != 0 {
            self.rt_registry.get(&depth_rt_id).map(|rt| {
                rt.color_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("threers postfx linear depth"),
                    format: Some(rt.format),
                    ..Default::default()
                })
            })
        } else {
            None
        };
        let normal_linear = if use_linear_depth {
            self.rt_registry.get(&normal_rt_id).map(|rt| {
                rt.color_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("threers postfx linear normal"),
                    format: Some(rt.format),
                    ..Default::default()
                })
            })
        } else {
            None
        };
        let input_bind = input_raw
            .as_ref()
            .or(input_linear.as_ref())
            .unwrap_or(&input_view);
        let depth_bind = depth_linear.as_ref().unwrap_or(depth_ref);
        let mut params2 = params2;
        if kind0_byte_copy {
            params2[2] = 1.0;
        }
        // SSAO (11) and bokeh (26) read the hardware depth of the prepass
        // target, which the caller names as both depth and normal.
        let use_hw_depth =
            matches!(kind, 11 | 26) && depth_rt_id != 0 && depth_rt_id == normal_rt_id;
        let hw_depth_local = if use_hw_depth {
            self.rt_depth_view_cache.remove(&normal_rt_id)
        } else {
            None
        };
        let hw_depth_fallback = self.hw_depth_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("threers postfx hw depth fallback"),
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        let hw_depth_bind = hw_depth_local.as_ref().unwrap_or(&hw_depth_fallback);
        // A "raw" view above is the target's own format: without view formats
        // (WebGL2) an sRGB target cannot be viewed as anything else, so the
        // view decodes and the shader has to re-encode what it reads.
        let srgb_rt = |id: u32| {
            self.rt_registry
                .get(&id)
                .is_some_and(|rt| rt.format.is_srgb())
        };
        let mut raw_srgb = 0u32;
        if (input_raw.is_some() || input_linear.is_some()) && srgb_rt(input_rt_id) {
            raw_srgb |= 2;
        }
        if depth_linear.is_some() && srgb_rt(depth_rt_id) {
            raw_srgb |= 4;
        }
        if normal_linear.is_some() && srgb_rt(normal_rt_id) {
            raw_srgb |= 8;
        }
        match normal_owned {
            Some(n) => {
                let normal_bind = normal_linear.as_ref().unwrap_or(&n);
                self.apply_postfx_views(
                    input_bind,
                    output_view,
                    depth_bind,
                    normal_bind,
                    hw_depth_bind,
                    kind,
                    time,
                    params2,
                    params3,
                    width,
                    height,
                    additive,
                    camera,
                    use_hw_depth,
                    output_format,
                    raw_srgb,
                );
                self.rt_view_cache.insert(normal_rt_id, n);
                self.tex_bg_cache.clear();
                self.variant_bg_cache.clear();
            }
            None => {
                let fallback_normal = self
                    .default_normal
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let normal_bind = normal_linear.as_ref().unwrap_or(&fallback_normal);
                self.apply_postfx_views(
                    input_bind,
                    output_view,
                    depth_bind,
                    normal_bind,
                    hw_depth_bind,
                    kind,
                    time,
                    params2,
                    params3,
                    width,
                    height,
                    additive,
                    camera,
                    use_hw_depth,
                    output_format,
                    raw_srgb,
                );
            }
        }
        if let Some(v) = hw_depth_local {
            self.rt_depth_view_cache.insert(normal_rt_id, v);
        }
        self.rt_view_cache.insert(input_rt_id, input_view);
        self.tex_bg_cache.clear();
        self.variant_bg_cache.clear();
        if let Some(d) = depth_owned {
            self.rt_view_cache.insert(depth_rt_id, d);
            self.tex_bg_cache.clear();
            self.variant_bg_cache.clear();
        }
    }

    /// Apply a post-fx pass: read from `input_view`, write to `output_view`.
    /// Defaults to REPLACE blend / clear on load. Use `apply_postfx_ex` for
    /// additive blend (bloom composite) or load-preserve behavior.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_postfx(
        &mut self,
        input_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        kind: u32,
        time: f32,
        params2: [f32; 4],
        params3: [f32; 4],
        width: u32,
        height: u32,
    ) {
        let fallback_normal = self
            .default_normal
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let hw_fallback = self.hw_depth_tex.create_view(&wgpu::TextureViewDescriptor {
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        self.apply_postfx_ex(
            input_view,
            output_view,
            input_view,
            &fallback_normal,
            &hw_fallback,
            kind,
            time,
            params2,
            params3,
            width,
            height,
            false,
            PostFxCamera::default(),
            false,
            self.color_format,
        );
    }

    /// Extended post-fx pass with `additive` blend toggle.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_postfx_ex(
        &mut self,
        input_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        hw_depth_view: &wgpu::TextureView,
        kind: u32,
        time: f32,
        params2: [f32; 4],
        params3: [f32; 4],
        width: u32,
        height: u32,
        additive: bool,
        camera: PostFxCamera,
        use_hw_depth: bool,
        output_format: wgpu::TextureFormat,
    ) {
        self.apply_postfx_views(
            input_view,
            output_view,
            depth_view,
            normal_view,
            hw_depth_view,
            kind,
            time,
            params2,
            params3,
            width,
            height,
            additive,
            camera,
            use_hw_depth,
            output_format,
            0,
        );
    }

    /// [`Self::apply_postfx_ex`], told which bound views were meant to show
    /// the stored bytes but decode them: `raw_srgb` holds `POSTFX_SHADER`'s
    /// `postfx_flag` bits 2 (input), 4 (depth) and 8 (normal). Bit 1 — the
    /// output is an sRGB format — is worked out here from `output_format`.
    ///
    /// Each branch of the shader is written for a plain `Unorm` target: what
    /// it returns is the byte stored. An sRGB target encodes on write, so
    /// without bit 1 every computed pass was encoded once more on WebGL2 —
    /// where every 8-bit target and the canvas are sRGB — than on WebGPU.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn apply_postfx_views(
        &mut self,
        input_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        hw_depth_view: &wgpu::TextureView,
        kind: u32,
        time: f32,
        params2: [f32; 4],
        params3: [f32; 4],
        width: u32,
        height: u32,
        additive: bool,
        camera: PostFxCamera,
        use_hw_depth: bool,
        output_format: wgpu::TextureFormat,
        raw_srgb: u32,
    ) {
        let linear_output = output_format == wgpu::TextureFormat::Rgba16Float;
        // Bit 16: the hardware depth is reversed (`set_reversed_z`), and
        // `sample_postfx_depth` hands back `1 - d`.
        let srgb_bits = raw_srgb
            | u32::from(output_format.is_srgb())
            | if self.state_pipelines.reversed() {
                16
            } else {
                0
            };
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct PostFxUniforms {
            params: [f32; 4],
            params2: [f32; 4],
            params3: [f32; 4],
            resolution: [f32; 4],
            ssao0: [f32; 4],
            inv_proj: [f32; 16],
            proj: [f32; 16],
        }
        let uniforms = PostFxUniforms {
            params: [
                kind as f32,
                time,
                if matches!(kind, 5 | 31) {
                    params3[3]
                } else {
                    0.0
                },
                if use_hw_depth { 1.0 } else { 0.0 },
            ],
            params2,
            params3,
            resolution: [
                width as f32,
                height as f32,
                if linear_output { 1.0 } else { 0.0 },
                srgb_bits as f32,
            ],
            ssao0: [
                camera.near,
                camera.far,
                camera.kernel_radius,
                camera.kernel_size as f32,
            ],
            inv_proj: camera.inv_proj,
            proj: camera.proj,
        };
        let buffer = wgpu::util::DeviceExt::create_buffer_init(
            &*self.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("threers postfx uniforms"),
                contents: bytemuck::cast_slice(&[uniforms]),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        );
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers postfx bg"),
            layout: &self.postfx_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.postfx_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(if matches!(kind, 5 | 31) {
                        &self.postfx_sampler
                    } else {
                        &self.postfx_nearest_sampler
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(hw_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.ssao_kernel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(if kind == 5 || kind == 31 {
                        &self.glitch_disp_view
                    } else {
                        &self.ssao_noise_view
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::Sampler(&self.postfx_noise_sampler),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("threers postfx encoder"),
            });
        {
            let multiply = kind == 15;
            let load_op = if additive || multiply {
                wgpu::LoadOp::Load
            } else {
                wgpu::LoadOp::Clear(wgpu::Color::BLACK)
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("threers postfx pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: load_op,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let output_is_f16 = output_format == wgpu::TextureFormat::Rgba16Float;
            let pipe = if multiply {
                &self.pipeline_postfx_multiply
            } else if additive {
                if kind == 17 {
                    &self.pipeline_postfx_outline_add
                } else if output_is_f16 {
                    &self.pipeline_postfx_add_f16
                } else {
                    &self.pipeline_postfx_add
                }
            } else if output_is_f16 {
                &self.pipeline_postfx_f16
            } else {
                &self.pipeline_postfx
            };
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Upload SSAO hemisphere kernel samples (32 vec3 values, padded to vec4).
    pub fn set_ssao_kernel(&mut self, kernel: &[f32]) {
        let mut data = [0f32; 128];
        for i in 0..32 {
            let base = i * 3;
            if base + 2 < kernel.len() {
                data[i * 4] = kernel[base];
                data[i * 4 + 1] = kernel[base + 1];
                data[i * 4 + 2] = kernel[base + 2];
            }
        }
        self.queue
            .write_buffer(&self.ssao_kernel_buffer, 0, bytemuck::cast_slice(&data));
    }

    /// Blit tightly-packed RGBA8 bytes (row-major, top-first) to an output view.
    pub fn blit_rgba8(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        output_view: &wgpu::TextureView,
        output_format: wgpu::TextureFormat,
    ) {
        let w = width.max(1);
        let h = height.max(1);
        let need = (w * h * 4) as usize;
        if data.len() < need {
            return;
        }
        // RGBA8 bytes from a canvas or an image are sRGB-encoded and are to
        // reach the screen as they came. They are uploaded as plain `Unorm`,
        // so the copy reads the bytes themselves; onto an sRGB target the
        // post-fx pass decodes its output (`postfx_flag` bit 1) and the
        // target's encode gives the same bytes back.
        let upload_format = wgpu::TextureFormat::Rgba8Unorm;
        if self.rgba_upload_tex.width() != w
            || self.rgba_upload_tex.height() != h
            || self.rgba_upload_tex.format() != upload_format
        {
            self.rgba_upload_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers rgba upload"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: upload_format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.rgba_upload_view = self
                .rgba_upload_tex
                .create_view(&wgpu::TextureViewDescriptor::default());
        }
        const ALIGN: u32 = 256;
        let unpadded = w * 4;
        let padded = unpadded.div_ceil(ALIGN) * ALIGN;
        let mut upload = vec![0u8; (padded * h) as usize];
        for row in 0..h {
            let src = (row * w * 4) as usize;
            let dst = (row * padded) as usize;
            upload[dst..dst + (w * 4) as usize].copy_from_slice(&data[src..src + (w * 4) as usize]);
        }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.rgba_upload_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &upload,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        let upload_view = self
            .rgba_upload_tex
            .create_view(&wgpu::TextureViewDescriptor::default());
        let fallback_normal = self
            .default_normal
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let hw_fallback = self.hw_depth_tex.create_view(&wgpu::TextureViewDescriptor {
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        self.apply_postfx_ex(
            &upload_view,
            output_view,
            &upload_view,
            &fallback_normal,
            &hw_fallback,
            0,
            0.0,
            // A raw texel copy (`params2.z`): the bytes are already what the
            // screen should show, so no gamma is applied on the way out.
            [1.0, 0.0, 1.0, 0.0],
            [0.0; 4],
            w,
            h,
            false,
            PostFxCamera::default(),
            false,
            output_format,
        );
    }

    /// Blend a straight-alpha RGBA8 overlay over an already-rendered target.
    ///
    /// Unlike [`blit_rgba8`](Self::blit_rgba8), which replaces the target, this
    /// loads the existing contents and composites `data` on top — the 3D image
    /// stays visible wherever the overlay is transparent. `data` must be
    /// `width * height * 4` bytes, row-major from the top-left.
    ///
    /// Draw it after [`render`](Self::render) and before presenting the frame.
    pub fn draw_overlay_rgba8(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        target: &wgpu::TextureView,
        format: wgpu::TextureFormat,
    ) {
        self.composite_overlay(data, width, height, None, target, format);
    }

    /// Blend an [`Overlay`](crate::overlay::Overlay) over an already-rendered
    /// target, uploading only the rows that changed.
    ///
    /// Prefer this to [`draw_overlay_rgba8`](Self::draw_overlay_rgba8) when the
    /// pixels come from an `Overlay`. A full-frame upload is most of what
    /// compositing one costs — at 1920x1080 the upload alone measures 728 µs
    /// against ~124 µs for the quad and the submit together — and a read-out in
    /// the corner changes 95 rows out of 1080. Measured over a frame that
    /// clears the overlay, draws a two-line panel and composites it:
    /// **1.1 ms against 223 µs, 4.9x**.
    ///
    /// Taking the `Overlay` rather than its bytes is what makes that safe. The
    /// band to upload and the acknowledgement that it happened are two halves
    /// of one step, and a caller doing them by hand could skip a composite and
    /// leave the texture holding a frame nobody remembers.
    ///
    /// Draw it after [`render`](Self::render) and before presenting. Use one
    /// `Overlay` per `Renderer`: the dirty band describes what *this*
    /// renderer's texture is missing.
    pub fn draw_overlay(
        &mut self,
        overlay: &mut crate::overlay::Overlay,
        target: &wgpu::TextureView,
        format: wgpu::TextureFormat,
    ) {
        let (width, height) = overlay.size();
        // An empty band uploads nothing and still composites, which is what a
        // frame wants when the overlay has not changed: the image underneath it
        // is new even when the overlay is not.
        let bands = overlay.dirty_bands();
        self.composite_overlay(
            overlay.pixels(),
            width,
            height,
            Some(&bands),
            target,
            format,
        );
        overlay.mark_uploaded();
    }

    pub(super) fn composite_overlay(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        rows: Option<&[(u32, u32)]>,
        target: &wgpu::TextureView,
        format: wgpu::TextureFormat,
    ) {
        let pass = self
            .overlay_pass
            .get_or_insert_with(|| super::super::caption_pass::CaptionPass::new(&self.device));
        pass.draw(
            &self.device,
            &self.queue,
            data,
            width,
            height,
            rows,
            target,
            format,
        );
    }

    /// Draw the caption showing at `time` seconds over an already-rendered target.
    ///
    /// Sizes the overlay to `target_width × target_height`, rasterizes the
    /// active cue only when it changes, and blends it in. Cheap enough to call
    /// every frame — a frame with no cue change costs one texture upload, and a
    /// frame with no cue at all costs nothing.
    ///
    /// ```no_run
    /// # use threers::{Renderer, Scene, PerspectiveCamera, captions::{CaptionOverlay, CaptionTrack}};
    /// # fn demo(renderer: &mut Renderer, scene: &mut Scene, camera: &PerspectiveCamera,
    /// #         view: &wgpu::TextureView, format: wgpu::TextureFormat, time: f64) {
    /// let mut overlay = CaptionOverlay::new(CaptionTrack::new().cue(0.0, 3.0, "Hello"))
    ///     .auto_scale(true);
    /// renderer.render(scene, camera, view, false);
    /// renderer.draw_caption_overlay(&mut overlay, time, 1280, 720, view, format);
    /// # }
    /// ```
    #[cfg(feature = "captions")]
    pub fn draw_caption_overlay(
        &mut self,
        overlay: &mut crate::captions::CaptionOverlay,
        time: f64,
        target_width: u32,
        target_height: u32,
        target: &wgpu::TextureView,
        format: wgpu::TextureFormat,
    ) {
        overlay.set_size(target_width, target_height);
        overlay.update(time);
        if !overlay.is_visible() {
            return;
        }
        let (w, h) = overlay.size();
        // Borrow the buffer out of the overlay for the upload; the pass only
        // reads it, but the borrow checker needs the copy-free split.
        let pass = self
            .caption_pass
            .get_or_insert_with(|| super::super::caption_pass::CaptionPass::new(&self.device));
        pass.draw(
            &self.device,
            &self.queue,
            overlay.rgba(),
            w,
            h,
            None,
            target,
            format,
        );
    }

    /// Upload GlitchPass's `size × size` displacement heightmap (R32Float, read
    /// nearest, as three.js's `DataTexture` defaults to).
    pub fn set_glitch_disp(&mut self, data: &[f32], size: u32) {
        let sz = size.max(1);
        if data.len() < (sz * sz) as usize {
            return;
        }
        if self.glitch_disp_tex.width() != sz {
            self.glitch_disp_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers glitch disp"),
                size: wgpu::Extent3d {
                    width: sz,
                    height: sz,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.glitch_disp_view = self
                .glitch_disp_tex
                .create_view(&wgpu::TextureViewDescriptor::default());
        }
        let data = &data[..(sz * sz) as usize];
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.glitch_disp_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(sz * 4),
                rows_per_image: Some(sz),
            },
            wgpu::Extent3d {
                width: sz,
                height: sz,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Upload 4×4 SSAO rotation noise (R channel, repeat-wrapped).
    pub fn set_ssao_noise(&mut self, noise: &[f32]) {
        let mut texels = [0f32; 16];
        for (dst, src) in texels.iter_mut().zip(noise.iter()) {
            *dst = *src;
        }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.ssao_noise_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&texels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(16),
                rows_per_image: Some(4),
            },
            wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
        );
    }
}
