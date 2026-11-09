use super::*;

impl Renderer {
    /// Create pipelines and default 1×1 fallback textures.
    ///
    /// This eagerly constructs the initial pipelines. Use [`Self::new_lazy`]
    /// to opt into pipeline creation on first use instead.
    ///
    /// `color_format` is the swapchain or render-target format (typically
    /// `Rgba8Unorm` or `Rgba8UnormSrgb` on web, `Bgra8UnormSrgb` on some desktops).
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        color_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        Self::with_options(
            device,
            queue,
            color_format,
            width,
            height,
            RendererOptions::default(),
        )
    }

    /// Defer initial render pipelines until first use. Shader modules, layouts,
    /// textures and validation remain enabled. Pipeline validation errors and
    /// driver compilation can occur at first draw instead of construction.
    /// Warm representative scenes before exposing an interactive viewport.
    pub fn new_lazy(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        color_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        Self::with_options(
            device,
            queue,
            color_format,
            width,
            height,
            RendererOptions {
                lazy: true,
                ..Default::default()
            },
        )
    }

    /// [`new`](Self::new) or [`new_lazy`](Self::new_lazy), and optionally
    /// through a `wgpu::PipelineCache` — see [`RendererOptions`].
    pub fn with_options(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        color_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        options: RendererOptions,
    ) -> Self {
        let pipeline_source =
            super::super::pipeline_state::PipelineSource::new(options.lazy, options.pipeline_cache);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers shader"),
            source: wgpu::ShaderSource::Wgsl(super::super::shader::legacy_shader_source().into()),
        });

        let uniform_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        // 1–2: the transmission capture and its trilinear sampler.
        let frame_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers frame bgl"),
            entries: &[
                uniform_entry(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let mesh_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers mesh bgl"),
            entries: &[uniform_entry(0)],
        });

        // Skinned variant of the mesh BGL: same uniform at binding 0, plus a
        // storage buffer of bone matrices at binding 1. This lets the skinned
        // pipeline stay within the 4-bind-group WebGPU default limit.
        let mesh_skinned_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers mesh skinned bgl"),
            entries: &[
                uniform_entry(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // 7 fragment textures (albedo, normal, roughness, metalness, ao,
        // emissive, matcap) + 1 sampler, and for the vertex stage the morph
        // texture and the uv sets. Every other map — displacement included —
        // is a shader variant's (`material_maps`): with the frame group's
        // transmission capture and the environment group, this layout is at
        // WebGL2's sixteen texture units.
        let tex_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        // Binding 10: three.js's morph-target texture, read in the vertex
        // stage with `textureLoad` — unfilterable float, so `Rgba32Float`
        // works everywhere and it lowers to `texelFetch` on WebGL2. Eleven
        // textures here plus the environment group's five stays inside
        // WebGL2's sixteen units.
        let morph_entry = wgpu::BindGroupLayoutEntry {
            binding: 10,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        };
        let tex_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers tex bgl"),
            entries: &[
                tex_entry(0),
                tex_entry(1),
                tex_entry(2),
                tex_entry(3),
                tex_entry(4),
                tex_entry(5),
                tex_entry(6),
                uv_sets_entry(),
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    // The displacement sample needs the same sampler, from the
                    // vertex stage.
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                morph_entry,
            ],
        });

        // Bindings 9..12 are the screen-space capture the refraction pass reads.
        // They are left out where that pass cannot be compiled, and the reason is
        // the texture budget rather than tidiness: WebGL2 allots sixteen units,
        // and this group together with the nine material maps asks for
        // seventeen.
        let ltc = upload_ltc(&device, &queue);
        let env_entries: Vec<wgpu::BindGroupLayoutEntry> = vec![
            // 14: the `RectAreaLight` tables, filtered as three.js reads
            // them (`LinearFilter`).
            wgpu::BindGroupLayoutEntry {
                binding: 14,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::Cube,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // 13: `EnvUniforms`, this group's environment's parameters.
            wgpu::BindGroupLayoutEntry {
                binding: 13,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];
        // The main layout stops here, deliberately. Bindings 9..12 are the
        // screen-space capture, and no entry point outside the refraction pass
        // reads them — but a declared binding is not free: wgpu-hal's GL backend
        // assigns a texture unit per *layout* entry, not per use, which is why a
        // lit draw that samples none of them still ran out of units. They live
        // in a second layout below, used only by the pass that needs them.
        let env_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers env bgl"),
            entries: &env_entries,
        });

        // The refraction pass's own group 3: the environment and the capture, and
        // *not* the three shadow maps, which `fs_ss_glass` never samples. Those
        // three units are exactly what makes room for the capture's three, so
        // both layouts come to the same total and both fit inside WebGL2's
        // sixteen.
        let mut env_glass_entries: Vec<wgpu::BindGroupLayoutEntry> = env_entries
            .iter()
            .filter(|e| matches!(e.binding, 0 | 1 | 8 | 13))
            .cloned()
            .collect();
        env_glass_entries.extend([
            // 9/10: screen-space glass — mipmapped opaque-scene capture +
            // its trilinear sampler (only read by the fs_ss_glass entry).
            wgpu::BindGroupLayoutEntry {
                binding: 9,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 10,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            // 11: sampleable opaque depth; 12: glass back-face depth (both
            // marched/read by fs_ss_glass via textureLoad, no sampler).
            // 1×1 placeholders bound otherwise.
            // Unfilterable *float*, not `Depth`, even though the views are
            // depth-format — a depth format permits either. The shader reads
            // these with `textureLoad`, and GLSL has no `texelFetch` for a
            // depth sampler, so declared as depth the module cannot be
            // lowered for WebGL2 at all. As float textures the same reads
            // become `texelFetch` and the pass compiles everywhere.
            wgpu::BindGroupLayoutEntry {
                binding: 11,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 12,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // 15: each light's shadow on the glass, packed by
            // `fs_glass_shadow` — the one unit left, standing in for the
            // three shadow maps.
            wgpu::BindGroupLayoutEntry {
                binding: 15,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ]);
        let env_bgl_glass = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers env bgl (refraction)"),
            entries: &env_glass_entries,
        });
        let ss_glass_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("threers ss glass layout"),
                bind_group_layouts: &[
                    Some(&frame_bgl),
                    Some(&mesh_bgl),
                    Some(&tex_bgl),
                    Some(&env_bgl_glass),
                ],
                immediate_size: 0,
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers layout"),
            bind_group_layouts: &[
                Some(&frame_bgl),
                Some(&mesh_bgl),
                Some(&tex_bgl),
                Some(&env_bgl),
            ],
            immediate_size: 0,
        });

        // Skinned pipeline layout: same as the base 4 groups, but group 1 uses
        // the skinned mesh BGL (uniform + bones storage). Stays at 4 groups —
        // WebGPU's default `maxBindGroups` limit.
        let pipeline_layout_skinned =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("threers skinned layout"),
                bind_group_layouts: &[
                    Some(&frame_bgl),
                    Some(&mesh_skinned_bgl),
                    Some(&tex_bgl),
                    Some(&env_bgl),
                ],
                immediate_size: 0,
            });

        // The shadow passes' group 0: the frame uniforms and, at a dynamic
        // offset, the pass's own matrix (`ShadowPassGpu`). Their pipelines
        // get layouts of their own with it in place of the frame group.
        let shadow_pass_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers shadow pass bgl"),
            entries: &[
                uniform_entry(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<ShadowPassGpu>() as u64,
                        ),
                    },
                    count: None,
                },
            ],
        });
        let shadow_pass_stride = {
            let align = device.limits().min_uniform_buffer_offset_alignment as u64;
            (std::mem::size_of::<ShadowPassGpu>() as u64).div_ceil(align) * align
        };
        let shadow_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers shadow layout"),
            bind_group_layouts: &[
                Some(&shadow_pass_bgl),
                Some(&mesh_bgl),
                Some(&tex_bgl),
                Some(&env_bgl),
            ],
            immediate_size: 0,
        });
        let shadow_layout_skinned =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("threers skinned shadow layout"),
                bind_group_layouts: &[
                    Some(&shadow_pass_bgl),
                    Some(&mesh_skinned_bgl),
                    Some(&tex_bgl),
                    Some(&env_bgl),
                ],
                immediate_size: 0,
            });
        // Point passes: no environment group.
        let shadow_layout_point = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers point shadow layout"),
            bind_group_layouts: &[Some(&shadow_pass_bgl), Some(&mesh_bgl), Some(&tex_bgl)],
            immediate_size: 0,
        });

        // ShaderMaterial data rides in @group(1) alongside the mesh uniform
        // (WebGPU's default maxBindGroups is 4, so we can't add a 5th group):
        //   binding 0 = mesh uniform (as usual), binding 1 = user uniform block,
        //   bindings 2..4 = read-only storage arrays.
        let storage_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform_entry = |binding: u32, vis: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let volume_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D3,
                multisampled: false,
            },
            count: None,
        };
        let custom_tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        // The storage arrays again, as textures, for the devices that have no
        // storage buffers. `textureLoad` needs no filtering, so these stay
        // non-filterable and work with `Rgba32Float` everywhere.
        let data_tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        // The storage entries are omitted where the device has none — WebGL2's
        // limit is zero, and declaring them anyway fails layout creation
        // outright, which takes the whole renderer down before a frame is drawn.
        let mut custom_entries = vec![
            uniform_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
            // The user's own data is visible to the vertex stage as well, which
            // is what makes a custom vertex stage worth having: a ribbon or a
            // strand needs its control points there, not in the fragment.
            uniform_entry(1, wgpu::ShaderStages::VERTEX_FRAGMENT),
        ];
        if crate::renderer::caps::supports_storage_buffers(&device) {
            custom_entries.extend([storage_entry(2), storage_entry(3), storage_entry(4)]);
        }
        custom_entries.extend([
            // User textures + their sampler. Unbound slots get a 1x1 white
            // placeholder, so a shader that ignores them costs nothing.
            custom_tex_entry(5),
            custom_tex_entry(6),
            custom_tex_entry(7),
            custom_tex_entry(8),
            wgpu::BindGroupLayoutEntry {
                binding: 9,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            data_tex_entry(10),
            data_tex_entry(11),
            data_tex_entry(12),
        ]);
        // `ShaderMaterial::volumes`: two 3D textures and their clamp-to-edge
        // sampler, where they fit (not on WebGL2).
        if crate::renderer::caps::supports_shader_material_volumes(&device) {
            custom_entries.extend([
                volume_entry(13),
                volume_entry(14),
                wgpu::BindGroupLayoutEntry {
                    binding: 15,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ]);
        }
        let custom_mesh_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers custom mesh+user bgl"),
            entries: &custom_entries,
        });
        // Group 2 for shader materials: the sampler and the per-geometry
        // vertex data, and nothing else.
        //
        // `ShaderMaterial` has no standard maps — it carries a fragment source,
        // a uniform block and its own textures — so the eight material textures
        // in `tex_bgl` are definitionally unused by a custom shader. Declaring
        // them anyway cost eight fragment texture units, which is what put the
        // custom layout over the per-stage limit of sixteen and made shader
        // materials unavailable on baseline WebGPU altogether.
        //
        // Binding 7 stays for the shared `vs_main`'s sampler; the morph
        // texture and uv sets are the geometry's. A `ShaderMaterial` has no
        // displacement map, so one bind group serves every custom draw.
        let custom_tex_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers custom tex bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // The shared vertex stage reads the morph texture and the uv
                // sets too; a shader material binds the geometry's or the
                // empty defaults.
                morph_entry,
                uv_sets_entry(),
            ],
        });

        // The custom pipeline's group 3. A `ShaderMaterial` does its own
        // lighting and most samples neither the shadow maps nor the area-light
        // table the full environment carries, so where a backend charges a texture
        // unit per layout entry rather than per use — wgpu's GL backend, i.e.
        // WebGL2 — those are dead weight that would crowd out the volume slots:
        // the full group plus the volumes comes to eighteen there and WebGL2
        // allows sixteen, and going over panics inside wgpu-hal. On that backend
        // the pipeline takes a trimmed environment (the IBL cube, its 2D
        // fallback, the sampler and the uniform — bindings 0, 1, 8, 13), which
        // brings the layout back to fourteen. Where units are counted per use
        // (WebGPU, Metal, Vulkan) the full group costs nothing a shader does not
        // touch, so it stays and custom shaders keep shadows and area lights.
        // "No storage buffers" is the same backend, and the test used elsewhere.
        // `Some` only on the trimmed backend; elsewhere the custom pipeline
        // reuses the full `env_bgl` *object*, so the scene's `env_bind_group`
        // (built from it) binds to it without a separate compatible layout.
        let custom_env_trimmed = !crate::renderer::caps::supports_storage_buffers(&device);
        let custom_env_bgl = custom_env_trimmed.then(|| {
            let entries: Vec<wgpu::BindGroupLayoutEntry> = env_entries
                .iter()
                .filter(|e| matches!(e.binding, 0 | 1 | 8 | 13))
                .cloned()
                .collect();
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("threers custom env bgl (trimmed)"),
                entries: &entries,
            })
        });

        // Built only when the device can hold it.
        //
        // The four groups together once declared 20 sampled textures in the
        // fragment stage. WebGPU's floor is 16 and Chrome reports exactly that,
        // so that layout could not be created there at all; trimming the custom
        // tex group of the eight standard material maps a `ShaderMaterial` never
        // uses brought it to fourteen (see `custom_tex_bgl`). Below that floor —
        // genuinely old hardware — the feature is absent rather than broken, and
        // `None` is what says so. See `caps::CUSTOM_LAYOUT_TEXTURES` for the
        // count and how WebGL2's per-layout-entry budget is met.
        use crate::renderer::caps::CUSTOM_LAYOUT_TEXTURES;
        let custom_pipeline_layout =
            if device.limits().max_sampled_textures_per_shader_stage >= CUSTOM_LAYOUT_TEXTURES {
                Some(
                    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("threers custom (shader material) layout"),
                        bind_group_layouts: &[
                            Some(&frame_bgl),
                            Some(&custom_mesh_bgl),
                            Some(&custom_tex_bgl),
                            Some(custom_env_bgl.as_ref().unwrap_or(&env_bgl)),
                        ],
                        immediate_size: 0,
                    }),
                )
            } else {
                log::warn!(
                    "shader materials unavailable: this device allows {} sampled textures per \
                 fragment stage and the custom layout needs {CUSTOM_LAYOUT_TEXTURES}",
                    device.limits().max_sampled_textures_per_shader_stage
                );
                None
            };
        // A trimmed backend's `ShaderMaterial` that reads shadows or area lights
        // takes the full environment and gives up the volume slots: ten textures
        // in the other groups and six in that one come to sixteen, where the
        // shadow maps alone beside the volumes would be seventeen.
        let custom_lit_mesh_bgl = custom_pipeline_layout
            .as_ref()
            .filter(|_| custom_env_trimmed)
            .map(|_| {
                let entries: Vec<wgpu::BindGroupLayoutEntry> = custom_entries
                    .iter()
                    .filter(|e| e.binding < 13)
                    .cloned()
                    .collect();
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("threers custom mesh+user bgl (no volumes)"),
                    entries: &entries,
                })
            });
        let custom_lit_pipeline_layout = custom_lit_mesh_bgl.as_ref().map(|mesh_bgl| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("threers custom (lit shader material) layout"),
                bind_group_layouts: &[
                    Some(&frame_bgl),
                    Some(mesh_bgl),
                    Some(&custom_tex_bgl),
                    Some(&env_bgl),
                ],
                immediate_size: 0,
            })
        });

        // The screen-space (refraction) custom pipeline keeps the glass group 3,
        // whose capture textures leave no room for the volume slots on WebGL2,
        // so there it takes the group 1 without them, as a lit shader does.
        let custom_ss_pipeline_layout = custom_pipeline_layout.as_ref().map(|_| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("threers custom (screen-space shader material) layout"),
                bind_group_layouts: &[
                    Some(&frame_bgl),
                    Some(custom_lit_mesh_bgl.as_ref().unwrap_or(&custom_mesh_bgl)),
                    Some(&custom_tex_bgl),
                    Some(&env_bgl_glass),
                ],
                immediate_size: 0,
            })
        });

        let vertex_buffers = [wgpu::VertexBufferLayout {
            array_stride: GpuMesh::VERTEX_STRIDE as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 48,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }];
        // `buffers` takes a slice of `Option` in wgpu 30.
        let vertex_buffers = vertex_buffers.map(Some);
        // `Points`: the same vertex buffer, one vertex per *instance* — each
        // point is expanded to a quad by `vs_points`.
        let vertex_buffers_points = vertex_buffers.clone().map(|b| {
            b.map(|mut b| {
                b.step_mode = wgpu::VertexStepMode::Instance;
                b
            })
        });
        // Wide lines: a segment an instance, its start through slot 0 (as
        // `VsIn`) and its end through slot 1 — the same buffer bound a vertex
        // further on — each stepping two vertices (`vs_wide_lines`).
        let wide_stride = 2 * GpuMesh::VERTEX_STRIDE as wgpu::BufferAddress;
        let vertex_buffers_wide = [
            vertex_buffers[0].clone().map(|mut b| {
                b.step_mode = wgpu::VertexStepMode::Instance;
                b.array_stride = wide_stride;
                b
            }),
            Some(wgpu::VertexBufferLayout {
                array_stride: wide_stride,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 9,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: 24,
                        shader_location: 10,
                        format: wgpu::VertexFormat::Float32x2,
                    },
                    wgpu::VertexAttribute {
                        offset: 32,
                        shader_location: 11,
                        format: wgpu::VertexFormat::Float32x4,
                    },
                ],
            }),
        ];

        // Every pipeline a material draws with is created through this, which
        // keeps its descriptor so a material's `depthTest` / `depthWrite` /
        // `depthFunc` / `colorWrite` can be applied to it later — see
        // `pipeline_state`.
        let state_pipes = std::cell::RefCell::new(
            super::super::pipeline_state::StatePipelines::with_source(pipeline_source.clone()),
        );
        let record = |desc: &wgpu::RenderPipelineDescriptor| {
            state_pipes.borrow_mut().create_initial(&device, desc)
        };
        // The layouts whose group 2 is the material's maps, registered so a
        // pipeline on them can be rebuilt for a material shader variant
        // (`material_maps`, `StatePipelines::relayout`).
        {
            let mut sp = state_pipes.borrow_mut();
            sp.register_layout(
                &pipeline_layout,
                vec![
                    Some(frame_bgl.clone()),
                    Some(mesh_bgl.clone()),
                    Some(tex_bgl.clone()),
                    Some(env_bgl.clone()),
                ],
            );
            sp.register_layout(
                &pipeline_layout_skinned,
                vec![
                    Some(frame_bgl.clone()),
                    Some(mesh_skinned_bgl.clone()),
                    Some(tex_bgl.clone()),
                    Some(env_bgl.clone()),
                ],
            );
            sp.register_layout(
                &shadow_layout,
                vec![
                    Some(shadow_pass_bgl.clone()),
                    Some(mesh_bgl.clone()),
                    Some(tex_bgl.clone()),
                    Some(env_bgl.clone()),
                ],
            );
            sp.register_layout(
                &shadow_layout_skinned,
                vec![
                    Some(shadow_pass_bgl.clone()),
                    Some(mesh_skinned_bgl.clone()),
                    Some(tex_bgl.clone()),
                    Some(env_bgl.clone()),
                ],
            );
            sp.register_layout(
                &shadow_layout_point,
                vec![
                    Some(shadow_pass_bgl.clone()),
                    Some(mesh_bgl.clone()),
                    Some(tex_bgl.clone()),
                ],
            );
            sp.register_layout(
                &ss_glass_pipeline_layout,
                vec![
                    Some(frame_bgl.clone()),
                    Some(mesh_bgl.clone()),
                    Some(tex_bgl.clone()),
                    Some(env_bgl_glass.clone()),
                ],
            );
        }

        let make_pipeline = |fmt: wgpu::TextureFormat,
                             topology: wgpu::PrimitiveTopology,
                             cull: Option<wgpu::Face>,
                             label: &str| {
            let alpha = label.contains("alpha");
            // Note: PolygonMode::Line needs the POLYGON_MODE_LINE feature, which
            // is generally unavailable in browser WebGPU. The wireframe pipeline
            // instead uses LineList primitives over the triangle indices (fed as
            // edge indices at draw time), so every pipeline stays Fill.
            record(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &vertex_buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
                        blend: Some(if alpha {
                            wgpu::BlendState::ALPHA_BLENDING
                        } else {
                            wgpu::BlendState::REPLACE
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(topology, cull),
                // Transparent: don't write depth, so back fragments behind a
                // transparent surface still composite correctly. LessEqual is
                // three.js's default `depthFunc`.
                depth_stencil: Some(depth_stencil(!alpha, wgpu::CompareFunction::LessEqual)),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        // Single-layer "glass" pipelines. `GLASS_DEPTH` (the prepass) writes
        // the nearest glass depth with no color; the tint and light halves of
        // the colour pass (`GLASS_OUTPUT` in the shader) blend only the
        // fragments at that nearest depth (LessEqual) and don't write depth:
        // the tint multiplies what is behind by the glass's transmittance, the
        // light adds its reflection, highlights and emission.
        let make_glass_pipeline = |fmt: wgpu::TextureFormat, pass: u8, count: u32, label: &str| {
            let depth_only = pass == GLASS_DEPTH;
            let blend = match pass {
                GLASS_TINT => wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::Zero,
                        dst_factor: wgpu::BlendFactor::Src,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::Zero,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                },
                GLASS_LIGHT => wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                },
                _ => wgpu::BlendState::ALPHA_BLENDING,
            };
            let constants: &[(&str, f64)] = match pass {
                GLASS_TINT => &[("GLASS_OUTPUT", 1.0)],
                GLASS_LIGHT => &[("GLASS_OUTPUT", 2.0)],
                _ => &[],
            };
            record(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &vertex_buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
                        blend: Some(blend),
                        write_mask: if depth_only {
                            wgpu::ColorWrites::empty()
                        } else {
                            wgpu::ColorWrites::ALL
                        },
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants,
                        ..Default::default()
                    },
                }),
                primitive: primitive(
                    wgpu::PrimitiveTopology::TriangleList,
                    Some(wgpu::Face::Back),
                ),
                depth_stencil: Some(depth_stencil(
                    depth_only,
                    if depth_only {
                        wgpu::CompareFunction::Less
                    } else {
                        wgpu::CompareFunction::LessEqual
                    },
                )),
                multisample: wgpu::MultisampleState {
                    count,
                    ..Default::default()
                },
                multiview_mask: None,
                cache: None,
            })
        };
        let f16_format = wgpu::TextureFormat::Rgba16Float;
        let pipeline_glass_depth =
            make_glass_pipeline(color_format, GLASS_DEPTH, 1, "threers glass depth pipeline");
        let pipeline_glass_color =
            make_glass_pipeline(color_format, GLASS_LIGHT, 1, "threers glass color pipeline");
        let pipeline_glass_tint =
            make_glass_pipeline(color_format, GLASS_TINT, 1, "threers glass tint pipeline");
        let pipeline_glass_depth_f16 = make_glass_pipeline(
            f16_format,
            GLASS_DEPTH,
            1,
            "threers glass depth f16 pipeline",
        );
        let pipeline_glass_color_f16 = make_glass_pipeline(
            f16_format,
            GLASS_LIGHT,
            1,
            "threers glass color f16 pipeline",
        );
        let pipeline_glass_tint_f16 =
            make_glass_pipeline(f16_format, GLASS_TINT, 1, "threers glass tint f16 pipeline");
        // Single-layer glass multisampled, so a scene with glass keeps MSAA.
        let pipeline_glass_depth_msaa = make_glass_pipeline(
            color_format,
            GLASS_DEPTH,
            MSAA_SAMPLES,
            "threers glass depth msaa pipeline",
        );
        let pipeline_glass_color_msaa = make_glass_pipeline(
            color_format,
            GLASS_LIGHT,
            MSAA_SAMPLES,
            "threers glass color msaa pipeline",
        );
        let pipeline_glass_tint_msaa = make_glass_pipeline(
            color_format,
            GLASS_TINT,
            MSAA_SAMPLES,
            "threers glass tint msaa pipeline",
        );
        // And for the encoded-blend pass into the multisampled `Unorm` view.
        let encoded_glass_msaa = color_format.is_srgb().then(|| {
            let unorm = color_format.remove_srgb_suffix();
            [
                make_glass_pipeline(
                    unorm,
                    GLASS_DEPTH,
                    MSAA_SAMPLES,
                    "threers glass depth encoded msaa pipeline",
                ),
                make_glass_pipeline(
                    unorm,
                    GLASS_LIGHT,
                    MSAA_SAMPLES,
                    "threers glass color encoded msaa pipeline",
                ),
                make_glass_pipeline(
                    unorm,
                    GLASS_TINT,
                    MSAA_SAMPLES,
                    "threers glass tint encoded msaa pipeline",
                ),
            ]
        });

        // Screen-space refraction glass: opaque single-surface pass (writes depth,
        // Less test, no blend — it composites the refracted/reflected background
        // itself), sampling the captured ss-color from env-group binding 9.
        let make_ss_glass_pipeline = |fmt: wgpu::TextureFormat, label: &str| {
            record(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&ss_glass_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &vertex_buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_ss_glass"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(
                    wgpu::PrimitiveTopology::TriangleList,
                    Some(wgpu::Face::Back),
                ),
                // `LessEqual`: `fs_glass_shadow` has just written the front-most
                // glass depth, and this shades exactly that surface.
                depth_stencil: Some(depth_stencil(true, wgpu::CompareFunction::LessEqual)),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        // Built everywhere. It has its own group 3 — the screen-space capture in
        // place of the shadow maps it never samples — so it comes to the same
        // fourteen texture units the main layout does, inside WebGL2's sixteen.
        let pipeline_ss_glass = make_ss_glass_pipeline(color_format, "threers ss glass pipeline");
        // Each light's shadow on the refracting surfaces, for the pass above
        // (`fs_glass_shadow`): the lit layout, which has the shadow maps.
        let pipeline_glass_shadow = record(&wgpu::RenderPipelineDescriptor {
            label: Some("threers glass shadow pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &vertex_buffers,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_glass_shadow"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: GLASS_SHADOW_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: primitive(
                wgpu::PrimitiveTopology::TriangleList,
                Some(wgpu::Face::Back),
            ),
            depth_stencil: Some(depth_stencil(true, wgpu::CompareFunction::Less)),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let pipeline_ss_glass_f16 =
            make_ss_glass_pipeline(f16_format, "threers ss glass f16 pipeline");

        // Fullscreen "sample one texture → write" pipeline, used both to build
        // the ss-color mip chain (downsample) and to blit the capture back to the
        // frame target before the glass draws over it.
        let ss_blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers ss blit bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let ss_blit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers ss blit layout"),
            bind_group_layouts: &[Some(&ss_blit_bgl)],
            immediate_size: 0,
        });
        let ss_blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers ss blit shader"),
            source: wgpu::ShaderSource::Wgsl(super::super::shader::SS_BLIT_SHADER.into()),
        });
        let make_ss_blit_pipeline = |fmt: wgpu::TextureFormat, label: &str| {
            pipeline_source.create(
                &device,
                &wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&ss_blit_layout),
                    vertex: wgpu::VertexState {
                        module: &ss_blit_shader,
                        entry_point: Some("vs_main"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &ss_blit_shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: fmt,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: primitive(wgpu::PrimitiveTopology::TriangleList, None),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                },
            )
        };
        let pipeline_ss_blit = make_ss_blit_pipeline(color_format, "threers ss blit pipeline");
        let pipeline_ss_blit_f16 =
            make_ss_blit_pipeline(f16_format, "threers ss blit f16 pipeline");

        // Depth-only pass over glass back faces (cull front) → the exit depth of
        // the glass volume, for two-surface (thick) refraction. `vs_main` reads
        // the displacement map out of group 2, so this needs the texture layout
        // too — otherwise displaced glass would write back-face depth for
        // geometry that is not where it is drawn.
        let ss_back_depth_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers ss back depth layout"),
            bind_group_layouts: &[Some(&frame_bgl), Some(&mesh_bgl), Some(&tex_bgl)],
            immediate_size: 0,
        });
        state_pipes.borrow_mut().register_layout(
            &ss_back_depth_layout,
            vec![
                Some(frame_bgl.clone()),
                Some(mesh_bgl.clone()),
                Some(tex_bgl.clone()),
            ],
        );
        let pipeline_ss_back_depth = record(&wgpu::RenderPipelineDescriptor {
            label: Some("threers ss back depth pipeline"),
            layout: Some(&ss_back_depth_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &vertex_buffers,
                compilation_options: Default::default(),
            },
            fragment: None,
            primitive: primitive(
                wgpu::PrimitiveTopology::TriangleList,
                Some(wgpu::Face::Front),
            ),
            depth_stencil: Some(depth_stencil(true, wgpu::CompareFunction::Less)),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // --- Temporal anti-aliasing resolve ---
        let taa_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threers taa uniform"),
            size: std::mem::size_of::<TaaUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let taa_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers taa sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let tex_entry = |binding, depth| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                // Unfilterable float for the depth slot rather than `Depth`: the
                // resolve reads it with `textureLoad`, which has no GLSL
                // equivalent for a depth sampler. A depth format allows either
                // sample type, so this costs nothing and lets the pass compile
                // on WebGL2 as well.
                sample_type: if depth {
                    wgpu::TextureSampleType::Float { filterable: false }
                } else {
                    wgpu::TextureSampleType::Float { filterable: true }
                },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let taa_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers taa bgl"),
            entries: &[
                tex_entry(0, false), // current
                tex_entry(1, false), // history
                tex_entry(2, true),  // depth
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let taa_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers taa layout"),
            bind_group_layouts: &[Some(&taa_bgl)],
            immediate_size: 0,
        });
        let taa_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers taa shader"),
            source: wgpu::ShaderSource::Wgsl(super::super::shader::TAA_SHADER.into()),
        });
        let make_taa = |fmt: wgpu::TextureFormat, label: &str| {
            pipeline_source.create(
                &device,
                &wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&taa_layout),
                    vertex: wgpu::VertexState {
                        module: &taa_shader,
                        entry_point: Some("vs_main"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &taa_shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: fmt,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: primitive(wgpu::PrimitiveTopology::TriangleList, None),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                },
            )
        };
        // Built everywhere, including WebGL2: the resolve's depth slot is an
        // unfilterable float texture rather than a depth texture, so its
        // `textureLoad` lowers to `texelFetch` and the module compiles there too.
        let pipeline_taa = make_taa(color_format, "threers taa pipeline");
        let pipeline_taa_f16 = make_taa(f16_format, "threers taa f16 pipeline");

        // --- Weighted-blended OIT ---
        // Accumulation pass: fs_oit writes to accum (additive) + revealage
        // (multiplicative), depth-tested against opaque but not writing depth.
        let oit_accum_format = wgpu::TextureFormat::Rgba16Float;
        let oit_reveal_format = OIT_REVEAL_FORMAT;
        let add_blend = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let mul_blend = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Zero,
            dst_factor: wgpu::BlendFactor::OneMinusSrc,
            operation: wgpu::BlendOperation::Add,
        };
        // The accumulation writes weighted colour and coverage to two
        // attachments with different blend equations, which WebGL2 cannot do —
        // it blends every attachment alike. Where that is the case each gets a
        // pipeline and a pass of its own: the same fragments, drawn twice.
        let accum_target = wgpu::ColorTargetState {
            format: oit_accum_format,
            blend: Some(wgpu::BlendState {
                color: add_blend,
                alpha: add_blend,
            }),
            write_mask: wgpu::ColorWrites::ALL,
        };
        let reveal_target = wgpu::ColorTargetState {
            format: oit_reveal_format,
            blend: Some(wgpu::BlendState {
                color: mul_blend,
                alpha: mul_blend,
            }),
            write_mask: wgpu::ColorWrites::ALL,
        };
        let make_oit =
            |count: u32, label: &str, entry: &str, targets: &[Option<wgpu::ColorTargetState>]| {
                record(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        buffers: &vertex_buffers,
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some(entry),
                        targets,
                        // The light half of a blend, premultiplied, and the transmittance
                        // beside it (`shade_main`'s `GLASS_OUTPUT`).
                        compilation_options: wgpu::PipelineCompilationOptions {
                            constants: &[("GLASS_OUTPUT", 2.0)],
                            ..Default::default()
                        },
                    }),
                    primitive: primitive(
                        wgpu::PrimitiveTopology::TriangleList,
                        Some(wgpu::Face::Back),
                    ),
                    depth_stencil: Some(depth_stencil(false, wgpu::CompareFunction::Less)),
                    multisample: wgpu::MultisampleState {
                        count,
                        ..Default::default()
                    },
                    multiview_mask: None,
                    cache: None,
                })
            };
        let independent_blend = crate::renderer::caps::supports_independent_blend(&device);
        let make_oit_set = |count: u32, label: &str| {
            if independent_blend {
                vec![make_oit(
                    count,
                    label,
                    "fs_oit",
                    &[Some(accum_target.clone()), Some(reveal_target.clone())],
                )]
            } else {
                vec![
                    make_oit(count, label, "fs_oit_accum", &[Some(accum_target.clone())]),
                    make_oit(
                        count,
                        label,
                        "fs_oit_reveal",
                        &[Some(reveal_target.clone())],
                    ),
                ]
            }
        };
        let pipeline_oit = make_oit_set(1, "threers oit accum pipeline");
        // Multisampled, against the multisampled opaque depth, resolved
        // before the composite: a frame with OIT glass keeps its MSAA.
        let pipeline_oit_msaa = make_oit_set(MSAA_SAMPLES, "threers oit accum msaa pipeline");
        // Resolve pass: composite accum/revealage over the opaque image.
        let oit_resolve_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers oit resolve bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let resolve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers oit resolve shader"),
            source: wgpu::ShaderSource::Wgsl(super::super::shader::OIT_RESOLVE_SHADER.into()),
        });
        let resolve_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers oit resolve layout"),
            bind_group_layouts: &[Some(&oit_resolve_bgl)],
            immediate_size: 0,
        });
        // Two halves, as the glass colour pass is: the transmittance
        // multiplied over what is behind, then the light added.
        let make_resolve = |fmt: wgpu::TextureFormat, pass: u32, count: u32, label: &str| {
            let blend = if pass == 0 {
                wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::Zero,
                        dst_factor: wgpu::BlendFactor::Src,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::Zero,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                }
            } else {
                wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                }
            };
            pipeline_source.create(
                &device,
                &wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&resolve_layout),
                    vertex: wgpu::VertexState {
                        module: &resolve_shader,
                        entry_point: Some("vs_main"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &resolve_shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: fmt,
                            blend: Some(blend),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: wgpu::PipelineCompilationOptions {
                            constants: &[("RESOLVE_PASS", pass as f64)],
                            ..Default::default()
                        },
                    }),
                    primitive: primitive(wgpu::PrimitiveTopology::TriangleList, None),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState {
                        count,
                        ..Default::default()
                    },
                    multiview_mask: None,
                    cache: None,
                },
            )
        };
        let pipeline_oit_resolve = [
            make_resolve(color_format, 0, 1, "threers oit resolve tint pipeline"),
            make_resolve(color_format, 1, 1, "threers oit resolve light pipeline"),
        ];
        let pipeline_oit_resolve_f16 = [
            make_resolve(f16_format, 0, 1, "threers oit resolve tint f16 pipeline"),
            make_resolve(f16_format, 1, 1, "threers oit resolve light f16 pipeline"),
        ];
        // Into the multisampled colour, resolved to the frame: a pass drawn
        // over the frame afterwards (`overlay_attachments`) resolves the
        // multisampled colour again, and has to find the composite in it.
        let pipeline_oit_resolve_msaa = [
            make_resolve(
                color_format,
                0,
                MSAA_SAMPLES,
                "threers oit resolve tint msaa pipeline",
            ),
            make_resolve(
                color_format,
                1,
                MSAA_SAMPLES,
                "threers oit resolve light msaa pipeline",
            ),
        ];
        // A depth texture to another, by drawing, where they cannot be copied
        // (WebGL2): what the refraction pass's sampleable depth is filled by.
        // The depth is read through a nearest sampler, as `textureLoad` of a
        // depth texture reads 0 there.
        let depth_copy = (!crate::renderer::caps::supports_depth_copies(&device)).then(|| {
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("threers depth copy bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            });
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("threers depth copy shader"),
                source: wgpu::ShaderSource::Wgsl(DEPTH_COPY_SHADER.into()),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("threers depth copy layout"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
            let pipe = pipeline_source.create(
                &device,
                &wgpu::RenderPipelineDescriptor {
                    label: Some("threers depth copy pipeline"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &module,
                        entry_point: Some("vs"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &module,
                        entry_point: Some("fs"),
                        targets: &[],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: DEPTH_FORMAT,
                        depth_write_enabled: Some(true),
                        depth_compare: Some(wgpu::CompareFunction::Always),
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                },
            );
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("threers depth copy sampler"),
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });
            (bgl, pipe, sampler)
        });
        // A multisampled depth to a single-sampled one, for the refraction
        // pass under MSAA. Not where shaders cannot read multisampled
        // textures (WebGL2); there a refraction frame stays single-sampled.
        let depth_resolve =
            crate::renderer::caps::supports_multisampled_textures(&device).then(|| {
                let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("threers depth resolve bgl"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: true,
                        },
                        count: None,
                    }],
                });
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("threers depth resolve shader"),
                    source: wgpu::ShaderSource::Wgsl(
                        super::super::shader::DEPTH_RESOLVE_SHADER.into(),
                    ),
                });
                let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("threers depth resolve layout"),
                    bind_group_layouts: &[Some(&bgl)],
                    immediate_size: 0,
                });
                let pipeline = pipeline_source.create(
                    &device,
                    &wgpu::RenderPipelineDescriptor {
                        label: Some("threers depth resolve pipeline"),
                        layout: Some(&layout),
                        vertex: wgpu::VertexState {
                            module: &module,
                            entry_point: Some("vs_main"),
                            buffers: &[],
                            compilation_options: Default::default(),
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &module,
                            entry_point: Some("fs_main"),
                            targets: &[],
                            compilation_options: Default::default(),
                        }),
                        primitive: primitive(wgpu::PrimitiveTopology::TriangleList, None),
                        depth_stencil: Some(depth_stencil(true, wgpu::CompareFunction::Always)),
                        multisample: wgpu::MultisampleState::default(),
                        multiview_mask: None,
                        cache: None,
                    },
                );
                (bgl, pipeline)
            });
        let pipeline_tri = make_pipeline(
            color_format,
            wgpu::PrimitiveTopology::TriangleList,
            Some(wgpu::Face::Back),
            "threers tri pipeline",
        );
        let pipeline_tri_alpha = make_pipeline(
            color_format,
            wgpu::PrimitiveTopology::TriangleList,
            Some(wgpu::Face::Back),
            "threers tri alpha pipeline",
        );
        let pipeline_tri_nocull = make_pipeline(
            color_format,
            wgpu::PrimitiveTopology::TriangleList,
            None,
            "threers tri nocull pipeline",
        );
        let make_sky_pipeline = |fmt: wgpu::TextureFormat, label: &str| {
            record(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &vertex_buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(wgpu::PrimitiveTopology::TriangleList, None),
                depth_stencil: Some(depth_stencil(false, wgpu::CompareFunction::LessEqual)),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline_tri_sky = make_sky_pipeline(color_format, "threers tri sky pipeline");
        // `wireframe: true` materials: a LineList pipeline (Fill polygon mode, so no
        // NON_FILL_POLYGON_MODE feature needed and it works in browser WebGPU) drawn
        // over the mesh's separate wire index buffer — `BufferGeometry::wireframe_indices`,
        // i.e. polygon outlines when the geometry carries a polygon table, else every
        // triangle edge as three.js draws them. See `GpuMesh::wire_index_buffer`.
        let pipeline_tri_wire = make_pipeline(
            color_format,
            wgpu::PrimitiveTopology::LineList,
            None,
            "threers tri wire pipeline",
        );
        let pipeline_line = make_pipeline(
            color_format,
            wgpu::PrimitiveTopology::LineList,
            None,
            "threers line pipeline",
        );
        // three.js `Points`: a quad per point (`vs_points`), opaque — replace,
        // depth written — or blended, in either target format, single- or
        // multisampled.
        let make_point_pipeline =
            |fmt: wgpu::TextureFormat, blend: bool, samples: u32, label: &str| {
                record(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_points"),
                        buffers: &vertex_buffers_points,
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: fmt,
                            blend: Some(if blend {
                                wgpu::BlendState::ALPHA_BLENDING
                            } else {
                                wgpu::BlendState::REPLACE
                            }),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: primitive(wgpu::PrimitiveTopology::TriangleList, None),
                    depth_stencil: Some(depth_stencil(!blend, wgpu::CompareFunction::LessEqual)),
                    multisample: wgpu::MultisampleState {
                        count: samples,
                        ..Default::default()
                    },
                    multiview_mask: None,
                    cache: None,
                })
            };
        let pipeline_point = make_point_pipeline(color_format, false, 1, "threers point pipeline");
        let pipeline_point_alpha =
            make_point_pipeline(color_format, true, 1, "threers point alpha pipeline");
        // `LineSegments` wider than a pixel: a capsule a segment
        // (`vs_wide_lines` / `fs_wide_lines`), opaque or blended, as points are.
        let make_wide_line_pipeline =
            |fmt: wgpu::TextureFormat, blend: bool, samples: u32, label: &str| {
                record(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_wide_lines"),
                        buffers: &vertex_buffers_wide,
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_wide_lines"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: fmt,
                            blend: Some(if blend {
                                wgpu::BlendState::ALPHA_BLENDING
                            } else {
                                wgpu::BlendState::REPLACE
                            }),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: primitive(wgpu::PrimitiveTopology::TriangleList, None),
                    depth_stencil: Some(depth_stencil(!blend, wgpu::CompareFunction::LessEqual)),
                    multisample: wgpu::MultisampleState {
                        count: samples,
                        ..Default::default()
                    },
                    multiview_mask: None,
                    cache: None,
                })
            };
        let pipeline_wide_line =
            make_wide_line_pipeline(color_format, false, 1, "threers wide line pipeline");
        let pipeline_wide_line_alpha =
            make_wide_line_pipeline(color_format, true, 1, "threers wide line alpha pipeline");

        // --- Hardware-MSAA variants of the opaque forward pipelines (surface
        // path only). Identical to the ones above but with `count = MSAA_SAMPLES`;
        // used when antialiasing is enabled and the scene has no glass/OIT/refract
        // /instanced/skinned/custom draws (those fall back to the 1× path). ---
        let make_pipeline_ms = |fmt: wgpu::TextureFormat,
                                topology: wgpu::PrimitiveTopology,
                                cull: Option<wgpu::Face>,
                                label: &str| {
            let alpha = label.contains("alpha");
            record(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &vertex_buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
                        blend: Some(if alpha {
                            wgpu::BlendState::ALPHA_BLENDING
                        } else {
                            wgpu::BlendState::REPLACE
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(topology, cull),
                depth_stencil: Some(depth_stencil(!alpha, wgpu::CompareFunction::LessEqual)),
                multisample: wgpu::MultisampleState {
                    count: MSAA_SAMPLES,
                    ..Default::default()
                },
                multiview_mask: None,
                cache: None,
            })
        };
        let make_sky_pipeline_ms = |fmt: wgpu::TextureFormat, label: &str| {
            record(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &vertex_buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(wgpu::PrimitiveTopology::TriangleList, None),
                depth_stencil: Some(depth_stencil(false, wgpu::CompareFunction::LessEqual)),
                multisample: wgpu::MultisampleState {
                    count: MSAA_SAMPLES,
                    ..Default::default()
                },
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline_tri_msaa = make_pipeline_ms(
            color_format,
            wgpu::PrimitiveTopology::TriangleList,
            Some(wgpu::Face::Back),
            "threers tri msaa",
        );
        let pipeline_tri_alpha_msaa = make_pipeline_ms(
            color_format,
            wgpu::PrimitiveTopology::TriangleList,
            Some(wgpu::Face::Back),
            "threers tri alpha msaa",
        );
        let pipeline_tri_nocull_msaa = make_pipeline_ms(
            color_format,
            wgpu::PrimitiveTopology::TriangleList,
            None,
            "threers tri nocull msaa",
        );
        let pipeline_tri_wire_msaa = make_pipeline_ms(
            color_format,
            wgpu::PrimitiveTopology::LineList,
            None,
            "threers tri wire msaa",
        );
        let pipeline_line_msaa = make_pipeline_ms(
            color_format,
            wgpu::PrimitiveTopology::LineList,
            None,
            "threers line msaa",
        );
        let pipeline_point_msaa =
            make_point_pipeline(color_format, false, MSAA_SAMPLES, "threers point msaa");
        let pipeline_wide_line_msaa =
            make_wide_line_pipeline(color_format, false, MSAA_SAMPLES, "threers wide line msaa");
        let pipeline_tri_sky_msaa = make_sky_pipeline_ms(color_format, "threers tri sky msaa");

        let pipeline_tri_f16 = make_pipeline(
            f16_format,
            wgpu::PrimitiveTopology::TriangleList,
            Some(wgpu::Face::Back),
            "threers tri f16 pipeline",
        );
        let pipeline_tri_alpha_f16 = make_pipeline(
            f16_format,
            wgpu::PrimitiveTopology::TriangleList,
            Some(wgpu::Face::Back),
            "threers tri alpha f16 pipeline",
        );
        let pipeline_tri_nocull_f16 = make_pipeline(
            f16_format,
            wgpu::PrimitiveTopology::TriangleList,
            None,
            "threers tri nocull f16 pipeline",
        );
        let pipeline_tri_sky_f16 = make_sky_pipeline(f16_format, "threers tri sky f16 pipeline");
        let pipeline_tri_wire_f16 = make_pipeline(
            f16_format,
            wgpu::PrimitiveTopology::LineList,
            None,
            "threers tri wire f16 pipeline",
        );
        let pipeline_line_f16 = make_pipeline(
            f16_format,
            wgpu::PrimitiveTopology::LineList,
            None,
            "threers line f16 pipeline",
        );
        let pipeline_point_f16 =
            make_point_pipeline(f16_format, false, 1, "threers point f16 pipeline");
        let pipeline_point_alpha_f16 =
            make_point_pipeline(f16_format, true, 1, "threers point alpha f16 pipeline");
        let pipeline_wide_line_f16 =
            make_wide_line_pipeline(f16_format, false, 1, "threers wide line f16 pipeline");
        let pipeline_wide_line_alpha_f16 =
            make_wide_line_pipeline(f16_format, true, 1, "threers wide line alpha f16 pipeline");

        // Instanced pipeline: vertex layout slot 0 (per-vertex), slot 1 per-instance mat4.
        let vertex_buffers_instanced = [
            wgpu::VertexBufferLayout {
                array_stride: GpuMesh::VERTEX_STRIDE as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: 12,
                        shader_location: 1,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: 24,
                        shader_location: 2,
                        format: wgpu::VertexFormat::Float32x2,
                    },
                    wgpu::VertexAttribute {
                        offset: 32,
                        shader_location: 3,
                        format: wgpu::VertexFormat::Float32x4,
                    },
                    wgpu::VertexAttribute {
                        offset: 48,
                        shader_location: 8,
                        format: wgpu::VertexFormat::Float32x4,
                    },
                ],
            },
            wgpu::VertexBufferLayout {
                // Sixteen floats of matrix plus four of tint.
                array_stride: 80,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 4,
                        format: wgpu::VertexFormat::Float32x4,
                    },
                    wgpu::VertexAttribute {
                        offset: 16,
                        shader_location: 5,
                        format: wgpu::VertexFormat::Float32x4,
                    },
                    wgpu::VertexAttribute {
                        offset: 32,
                        shader_location: 6,
                        format: wgpu::VertexFormat::Float32x4,
                    },
                    wgpu::VertexAttribute {
                        offset: 48,
                        shader_location: 7,
                        format: wgpu::VertexFormat::Float32x4,
                    },
                    wgpu::VertexAttribute {
                        offset: 64,
                        shader_location: 9,
                        format: wgpu::VertexFormat::Float32x4,
                    },
                ],
            },
        ];
        // `buffers` takes a slice of `Option` in wgpu 30.
        let vertex_buffers_instanced = vertex_buffers_instanced.map(Some);
        // Skinned vertex layout: slot 0 = standard vertex, slot 1 = joints+weights.
        let vertex_buffers_skinned = [
            wgpu::VertexBufferLayout {
                array_stride: GpuMesh::VERTEX_STRIDE as wgpu::BufferAddress,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 0,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: 12,
                        shader_location: 1,
                        format: wgpu::VertexFormat::Float32x3,
                    },
                    wgpu::VertexAttribute {
                        offset: 24,
                        shader_location: 2,
                        format: wgpu::VertexFormat::Float32x2,
                    },
                    wgpu::VertexAttribute {
                        offset: 32,
                        shader_location: 3,
                        format: wgpu::VertexFormat::Float32x4,
                    },
                    wgpu::VertexAttribute {
                        offset: 48,
                        shader_location: 8,
                        format: wgpu::VertexFormat::Float32x4,
                    },
                ],
            },
            wgpu::VertexBufferLayout {
                array_stride: 32,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 4,
                        format: wgpu::VertexFormat::Float32x4,
                    },
                    wgpu::VertexAttribute {
                        offset: 16,
                        shader_location: 5,
                        format: wgpu::VertexFormat::Float32x4,
                    },
                ],
            },
        ];
        // `buffers` takes a slice of `Option` in wgpu 30.
        let vertex_buffers_skinned = vertex_buffers_skinned.map(Some);
        // Skinned and instanced pipelines, each in a filled (back-face culled)
        // and a `wireframe: true` LineList variant, for the surface format and
        // the half-float postfx format. The two kinds differ only in layout,
        // vertex entry point and vertex buffers.
        type MeshKind<'a> = (
            &'a wgpu::PipelineLayout,
            &'a str,
            &'a [Option<wgpu::VertexBufferLayout<'a>>],
        );
        let skinned: MeshKind = (
            &pipeline_layout_skinned,
            "vs_skinned",
            &vertex_buffers_skinned,
        );
        let instanced: MeshKind = (&pipeline_layout, "vs_instanced", &vertex_buffers_instanced);
        let make_mesh_pipeline_samples = |(layout, entry, buffers): MeshKind,
                                          fmt: wgpu::TextureFormat,
                                          topology: wgpu::PrimitiveTopology,
                                          cull: Option<wgpu::Face>,
                                          samples: u32,
                                          label: &str| {
            record(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(entry),
                    buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(topology, cull),
                depth_stencil: Some(depth_stencil(true, wgpu::CompareFunction::LessEqual)),
                multisample: wgpu::MultisampleState {
                    count: samples,
                    ..Default::default()
                },
                multiview_mask: None,
                cache: None,
            })
        };
        let make_mesh_pipeline = |kind: MeshKind,
                                  fmt: wgpu::TextureFormat,
                                  topology: wgpu::PrimitiveTopology,
                                  cull: Option<wgpu::Face>,
                                  label: &str| {
            make_mesh_pipeline_samples(kind, fmt, topology, cull, 1, label)
        };
        let tris = wgpu::PrimitiveTopology::TriangleList;
        let lines = wgpu::PrimitiveTopology::LineList;
        let back = Some(wgpu::Face::Back);
        // Multisampled, so a frame holding a skinned or instanced mesh keeps
        // its MSAA.
        let pipeline_skinned_msaa = make_mesh_pipeline_samples(
            skinned,
            color_format,
            tris,
            back,
            MSAA_SAMPLES,
            "threers skinned msaa pipeline",
        );
        let pipeline_instanced_msaa = make_mesh_pipeline_samples(
            instanced,
            color_format,
            tris,
            back,
            MSAA_SAMPLES,
            "threers instanced msaa pipeline",
        );
        let pipeline_skinned = make_mesh_pipeline(
            skinned,
            color_format,
            tris,
            back,
            "threers skinned pipeline",
        );
        let pipeline_skinned_f16 = make_mesh_pipeline(
            skinned,
            f16_format,
            tris,
            back,
            "threers skinned f16 pipeline",
        );
        let pipeline_skinned_wire = make_mesh_pipeline(
            skinned,
            color_format,
            lines,
            None,
            "threers skinned wire pipeline",
        );
        let pipeline_skinned_wire_f16 = make_mesh_pipeline(
            skinned,
            f16_format,
            lines,
            None,
            "threers skinned wire f16 pipeline",
        );
        let pipeline_instanced = make_mesh_pipeline(
            instanced,
            color_format,
            tris,
            back,
            "threers instanced pipeline",
        );
        let pipeline_instanced_f16 = make_mesh_pipeline(
            instanced,
            f16_format,
            tris,
            back,
            "threers instanced f16 pipeline",
        );
        let pipeline_instanced_wire = make_mesh_pipeline(
            instanced,
            color_format,
            lines,
            None,
            "threers instanced wire pipeline",
        );
        let pipeline_instanced_wire_f16 = make_mesh_pipeline(
            instanced,
            f16_format,
            lines,
            None,
            "threers instanced wire f16 pipeline",
        );

        let pipeline_sprite = record(&wgpu::RenderPipelineDescriptor {
            label: Some("threers sprite pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_sprite"),
                buffers: &vertex_buffers,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: primitive(wgpu::PrimitiveTopology::TriangleList, None),
            depth_stencil: Some(depth_stencil(false, wgpu::CompareFunction::LessEqual)),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let pipeline_sprite_f16 = record(&wgpu::RenderPipelineDescriptor {
            label: Some("threers sprite f16 pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_sprite"),
                buffers: &vertex_buffers,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: f16_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: primitive(wgpu::PrimitiveTopology::TriangleList, None),
            depth_stencil: Some(depth_stencil(false, wgpu::CompareFunction::LessEqual)),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Alpha-blended draws on the skinned, instanced and sprite layouts:
        // the transparent instanced and skinned pipelines in both target
        // formats, and — on an sRGB target — all three into its `Unorm` view
        // for the encoded-blend pass. No depth write, as `pipeline_tri_alpha`.
        let make_blend_pipeline = |(layout, entry, buffers): MeshKind,
                                   fmt: wgpu::TextureFormat,
                                   cull: Option<wgpu::Face>,
                                   compare: wgpu::CompareFunction,
                                   label: &str| {
            record(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(entry),
                    buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(wgpu::PrimitiveTopology::TriangleList, cull),
                depth_stencil: Some(depth_stencil(false, compare)),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        // three.js's default `depthFunc`.
        let less = wgpu::CompareFunction::LessEqual;
        let pipeline_instanced_alpha = make_blend_pipeline(
            instanced,
            color_format,
            back,
            less,
            "threers instanced alpha pipeline",
        );
        let pipeline_instanced_alpha_f16 = make_blend_pipeline(
            instanced,
            f16_format,
            back,
            less,
            "threers instanced alpha f16 pipeline",
        );
        let pipeline_skinned_alpha = make_blend_pipeline(
            skinned,
            color_format,
            back,
            less,
            "threers skinned alpha pipeline",
        );
        let pipeline_skinned_alpha_f16 = make_blend_pipeline(
            skinned,
            f16_format,
            back,
            less,
            "threers skinned alpha f16 pipeline",
        );
        let sprite_kind: MeshKind = (&pipeline_layout, "vs_sprite", &vertex_buffers);
        let encoded_blend_pipelines = color_format.is_srgb().then(|| {
            let unorm = color_format.remove_srgb_suffix();
            [
                make_blend_pipeline(
                    sprite_kind,
                    unorm,
                    None,
                    wgpu::CompareFunction::LessEqual,
                    "threers sprite encoded pipeline",
                ),
                make_blend_pipeline(
                    instanced,
                    unorm,
                    back,
                    less,
                    "threers instanced encoded pipeline",
                ),
                make_blend_pipeline(
                    skinned,
                    unorm,
                    back,
                    less,
                    "threers skinned encoded pipeline",
                ),
                make_point_pipeline(unorm, true, 1, "threers point encoded pipeline"),
                make_glass_pipeline(
                    unorm,
                    GLASS_DEPTH,
                    1,
                    "threers glass depth encoded pipeline",
                ),
                make_glass_pipeline(
                    unorm,
                    GLASS_LIGHT,
                    1,
                    "threers glass color encoded pipeline",
                ),
                make_glass_pipeline(unorm, GLASS_TINT, 1, "threers glass tint encoded pipeline"),
                make_wide_line_pipeline(unorm, true, 1, "threers wide line encoded pipeline"),
            ]
        });

        // Depth-only pipeline used to render the scene from each shadow-casting
        // light's POV. Reuses the per-vertex layout + per-mesh uniforms.
        let make_shadow_pipeline = |entry: &str, label: &str, cut: bool| {
            record(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&shadow_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(entry),
                    buffers: &vertex_buffers,
                    compilation_options: Default::default(),
                },
                // A cut-out caster (`alphaTest` with a map) tests the map's
                // alpha per fragment; every other caster is depth-only.
                fragment: cut.then(|| wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_shadow_cut"),
                    targets: &[],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(
                    wgpu::PrimitiveTopology::TriangleList,
                    Some(wgpu::Face::Front),
                ),
                // Shadow bias reduces acne; keep the explicit DepthStencilState.
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: Default::default(),
                    bias: wgpu::DepthBiasState {
                        constant: 2,
                        slope_scale: 2.0,
                        clamp: 0.0,
                    },
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline_shadow = make_shadow_pipeline("vs_shadow", "threers shadow pipeline", false);
        let pipeline_shadow_spot =
            make_shadow_pipeline("vs_shadow_spot", "threers spot shadow pipeline", false);
        // The only shadow pipeline with a fragment stage: see `fs_shadow_point`.
        let pipeline_shadow_point = {
            record(&wgpu::RenderPipelineDescriptor {
                label: Some("threers point shadow pipeline"),
                layout: Some(&shadow_layout_point),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_shadow_point"),
                    buffers: &vertex_buffers,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_shadow_point"),
                    targets: &[],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        // Shadow casters that are skinned: the same three passes on the
        // skinned layout and vertex buffers, so the map holds the pose the
        // colour pass draws rather than the bind pose (or nothing).
        let make_skinned_shadow_pipeline = |entry: &str, label: &str, point: bool, cut: bool| {
            record(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&shadow_layout_skinned),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(entry),
                    buffers: &vertex_buffers_skinned,
                    compilation_options: Default::default(),
                },
                fragment: (point || cut).then(|| wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(if point {
                        "fs_shadow_point"
                    } else {
                        "fs_shadow_cut"
                    }),
                    targets: &[],
                    compilation_options: Default::default(),
                }),
                primitive: if point {
                    wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        cull_mode: None,
                        ..Default::default()
                    }
                } else {
                    primitive(
                        wgpu::PrimitiveTopology::TriangleList,
                        Some(wgpu::Face::Front),
                    )
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: Default::default(),
                    bias: if point {
                        Default::default()
                    } else {
                        wgpu::DepthBiasState {
                            constant: 2,
                            slope_scale: 2.0,
                            clamp: 0.0,
                        }
                    },
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline_shadow_skinned = make_skinned_shadow_pipeline(
            "vs_shadow_skinned",
            "threers skinned shadow pipeline",
            false,
            false,
        );
        let pipeline_shadow_spot_skinned = make_skinned_shadow_pipeline(
            "vs_shadow_spot_skinned",
            "threers skinned spot shadow pipeline",
            false,
            false,
        );
        let pipeline_shadow_point_skinned = make_skinned_shadow_pipeline(
            "vs_shadow_point_skinned",
            "threers skinned point shadow pipeline",
            true,
            false,
        );

        // Instanced casters: the same three passes on the instanced vertex
        // layout. The point pass's layout is the point pipeline's — frame,
        // mesh and the morph texture's group — and the other two share the
        // mesh pipelines' layout, as their non-instanced twins do.
        let make_instanced_shadow_pipeline = |entry: &str, label: &str, point: bool, cut: bool| {
            record(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(if point {
                    &shadow_layout_point
                } else {
                    &shadow_layout
                }),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(entry),
                    buffers: &vertex_buffers_instanced,
                    compilation_options: Default::default(),
                },
                fragment: (point || cut).then(|| wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(if point {
                        "fs_shadow_point"
                    } else {
                        "fs_shadow_cut"
                    }),
                    targets: &[],
                    compilation_options: Default::default(),
                }),
                primitive: if point {
                    wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        cull_mode: None,
                        ..Default::default()
                    }
                } else {
                    primitive(
                        wgpu::PrimitiveTopology::TriangleList,
                        Some(wgpu::Face::Front),
                    )
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: Default::default(),
                    bias: if point {
                        Default::default()
                    } else {
                        wgpu::DepthBiasState {
                            constant: 2,
                            slope_scale: 2.0,
                            clamp: 0.0,
                        }
                    },
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline_shadow_instanced = make_instanced_shadow_pipeline(
            "vs_shadow_instanced",
            "threers instanced shadow pipeline",
            false,
            false,
        );
        let pipeline_shadow_spot_instanced = make_instanced_shadow_pipeline(
            "vs_shadow_spot_instanced",
            "threers instanced spot shadow pipeline",
            false,
            false,
        );
        let pipeline_shadow_point_instanced = make_instanced_shadow_pipeline(
            "vs_shadow_point_instanced",
            "threers instanced point shadow pipeline",
            true,
            false,
        );

        // Cut-out casters: the directional and spot passes again, with
        // `fs_shadow_cut` testing the map's alpha against `alphaTest`.
        let shadow_cut = ShadowCutPipelines {
            dir: make_shadow_pipeline("vs_shadow", "threers cut-out shadow pipeline", true),
            spot: make_shadow_pipeline(
                "vs_shadow_spot",
                "threers cut-out spot shadow pipeline",
                true,
            ),
            dir_skinned: make_skinned_shadow_pipeline(
                "vs_shadow_skinned",
                "threers skinned cut-out shadow pipeline",
                false,
                true,
            ),
            spot_skinned: make_skinned_shadow_pipeline(
                "vs_shadow_spot_skinned",
                "threers skinned cut-out spot shadow pipeline",
                false,
                true,
            ),
            dir_instanced: make_instanced_shadow_pipeline(
                "vs_shadow_instanced",
                "threers instanced cut-out shadow pipeline",
                false,
                true,
            ),
            spot_instanced: make_instanced_shadow_pipeline(
                "vs_shadow_spot_instanced",
                "threers instanced cut-out spot shadow pipeline",
                false,
                true,
            ),
        };

        // The opaque scene's depth again, single-sampled: for screen-space
        // refraction under MSAA where a shader cannot read the multisampled
        // depth to resolve it (WebGL2). The caster vertex stages, given the
        // camera's view-projection for their matrix; no colour, no bias.
        // `[plain, skinned, instanced]`, then the same with the cut-out
        // fragment stage for `alphaTest` with a map.
        let make_replay = |which: usize, cut: bool| {
            let (entry, layout, buffers): (
                &str,
                &wgpu::PipelineLayout,
                &[Option<wgpu::VertexBufferLayout>],
            ) = match which {
                0 => ("vs_shadow", &shadow_layout, &vertex_buffers[..]),
                1 => (
                    "vs_shadow_skinned",
                    &shadow_layout_skinned,
                    &vertex_buffers_skinned[..],
                ),
                _ => (
                    "vs_shadow_instanced",
                    &shadow_layout,
                    &vertex_buffers_instanced[..],
                ),
            };
            record(&wgpu::RenderPipelineDescriptor {
                label: Some("threers depth replay pipeline"),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(entry),
                    buffers,
                    compilation_options: Default::default(),
                },
                fragment: cut.then(|| wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_shadow_cut"),
                    targets: &[],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(
                    wgpu::PrimitiveTopology::TriangleList,
                    Some(wgpu::Face::Back),
                ),
                depth_stencil: Some(depth_stencil(true, wgpu::CompareFunction::LessEqual)),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let depth_replay: Vec<Pipeline> = [
            (0, false),
            (1, false),
            (2, false),
            (0, true),
            (1, true),
            (2, true),
        ]
        .iter()
        .map(|&(w, c)| make_replay(w, c))
        .collect();

        // Post-fx fullscreen-quad pipeline. Reads one input texture, writes
        // to a render attachment matching the surface format.
        let postfx_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers postfx shader"),
            source: wgpu::ShaderSource::Wgsl(crate::renderer::shader::postfx_shader().into()),
        });
        let postfx_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers postfx bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        // Unfilterable float rather than `Depth`, so the
                        // `textureLoad` in `sample_postfx_depth` lowers to
                        // `texelFetch`. A depth format allows either.
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        let postfx_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("threers postfx pipeline layout"),
                bind_group_layouts: &[Some(&postfx_bgl)],
                immediate_size: 0,
            });
        let pipeline_postfx = pipeline_source.create(
            &device,
            &wgpu::RenderPipelineDescriptor {
                label: Some("threers postfx pipeline"),
                layout: Some(&postfx_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &postfx_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &postfx_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(wgpu::PrimitiveTopology::TriangleList, None),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        );
        let postfx_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers postfx sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let postfx_nearest_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers postfx nearest sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let postfx_noise_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers postfx noise sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let ssao_kernel_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threers ssao kernel buffer"),
            size: (32 * 16) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ssao_noise_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers ssao noise"),
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let ssao_noise_view = ssao_noise_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let glitch_disp_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers glitch disp"),
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let glitch_disp_view = glitch_disp_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let rgba_upload_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers rgba upload"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let rgba_upload_view = rgba_upload_tex.create_view(&wgpu::TextureViewDescriptor::default());
        // Same pipeline as `pipeline_postfx`, but with additive blending so
        // the shader output adds to existing framebuffer pixels (bloom composite).
        let pipeline_postfx_add = pipeline_source.create(
            &device,
            &wgpu::RenderPipelineDescriptor {
                label: Some("threers postfx additive pipeline"),
                layout: Some(&postfx_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &postfx_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &postfx_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(wgpu::PrimitiveTopology::TriangleList, None),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        );
        let pipeline_postfx_outline_add = pipeline_source.create(
            &device,
            &wgpu::RenderPipelineDescriptor {
                label: Some("threers postfx outline additive pipeline"),
                layout: Some(&postfx_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &postfx_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &postfx_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::SrcAlpha,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(wgpu::PrimitiveTopology::TriangleList, None),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        );
        let pipeline_postfx_multiply = pipeline_source.create(
            &device,
            &wgpu::RenderPipelineDescriptor {
                label: Some("threers postfx multiply pipeline"),
                layout: Some(&postfx_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &postfx_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &postfx_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::Dst,
                                dst_factor: wgpu::BlendFactor::Zero,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(wgpu::PrimitiveTopology::TriangleList, None),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        );
        let postfx_f16_format = wgpu::TextureFormat::Rgba16Float;
        let pipeline_postfx_f16 = pipeline_source.create(
            &device,
            &wgpu::RenderPipelineDescriptor {
                label: Some("threers postfx f16 pipeline"),
                layout: Some(&postfx_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &postfx_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &postfx_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: postfx_f16_format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(wgpu::PrimitiveTopology::TriangleList, None),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        );
        let pipeline_postfx_add_f16 = pipeline_source.create(
            &device,
            &wgpu::RenderPipelineDescriptor {
                label: Some("threers postfx additive f16 pipeline"),
                layout: Some(&postfx_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &postfx_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &postfx_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: postfx_f16_format,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: primitive(wgpu::PrimitiveTopology::TriangleList, None),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        );

        let sprite_quad = {
            let mut g = crate::core::BufferGeometry::new();
            g.set_attribute(
                "position",
                crate::core::BufferAttribute::new(
                    vec![
                        -0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.5, 0.5, 0.0, -0.5, 0.5, 0.0,
                    ],
                    3,
                ),
            );
            g.set_attribute(
                "uv",
                crate::core::BufferAttribute::new(vec![0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], 2),
            );
            g.set_index(vec![0, 1, 2, 0, 2, 3]);
            GpuMesh::upload(&device, &g)
        };

        let (depth_texture, depth_view) = create_depth_view(&device, width, height);
        let hw_depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers postfx hw depth placeholder"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threers frame uniform"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let transmission_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers transmission sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let transmission_dummy = {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers transmission stand-in"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        };
        let frame_buffer_for_bg = frame_buffer.clone();
        let frame_bind_group = frame_bg(
            &device,
            &frame_bgl,
            &frame_buffer,
            &transmission_dummy.1,
            &transmission_sampler,
            "threers frame bg",
        );
        let frame_bind_group_dummy = frame_bg(
            &device,
            &frame_bgl,
            &frame_buffer,
            &transmission_dummy.1,
            &transmission_sampler,
            "threers frame bg (capture)",
        );

        let sampler_linear_repeat = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers linear-repeat sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            // Anisotropic filtering. Mip-mapping alone still blurs a texture seen
            // at a grazing angle — the whole surface of a sphere near its limb —
            // because the sampler has to pick one level for a footprint that is
            // long in one direction and short in the other.
            anisotropy_clamp: 16,
            ..Default::default()
        });
        let samplers = (0..SAMPLER_COUNT)
            .map(|i| make_sampler(&device, i))
            .collect();

        let default_white_srgb = GpuTexture::upload(
            &device,
            &queue,
            &Texture::solid([255, 255, 255, 255], TextureFormat::Rgba8UnormSrgb),
        );
        let default_data_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers user data placeholder"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let default_data_tex_view =
            default_data_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let default_volume_view = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("threers user volume placeholder"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());
        let sampler_volume = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers volume sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let default_white_linear = GpuTexture::upload(
            &device,
            &queue,
            &Texture::solid([255, 255, 255, 255], TextureFormat::Rgba8Unorm),
        );

        // The morph slot's placeholder: one texel, one layer, never read
        // (a draw without morph targets uploads a zero count).
        let default_morph = {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers default morph texture"),
                // Two layers so the GL backend makes it an array texture,
                // matching the shader's declaration (see `ensure_morph`).
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 2,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            });
            (texture, view)
        };
        let default_uv_sets = {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers default uv sets"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        };
        let custom_tex_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers custom tex bg"),
            layout: &custom_tex_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&default_uv_sets.1),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(&sampler_linear_repeat),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::TextureView(&default_morph.1),
                },
            ],
        });
        let default_normal = GpuTexture::upload(
            &device,
            &queue,
            &Texture::solid([128, 128, 255, 255], TextureFormat::Rgba8Unorm),
        );
        let default_transparent = GpuTexture::upload(
            &device,
            &queue,
            &Texture::solid([0, 0, 0, 0], TextureFormat::Rgba8Unorm),
        );

        // Default 1x1 environment cubemap: all-black faces (zero env contribution).
        let default_env_cube = {
            let face_data = [0u8; 4];
            let faces = [
                face_data.to_vec(),
                face_data.to_vec(),
                face_data.to_vec(),
                face_data.to_vec(),
                face_data.to_vec(),
                face_data.to_vec(),
            ];
            let src = CubeTexture::new(1, TextureFormat::Rgba8UnormSrgb, faces);
            GpuCubeTexture::upload(&device, &queue, &src)
        };
        let env_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers env sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        // Trilinear sampler for the screen-space glass capture — roughness maps
        // to a mip LOD so rough glass blurs its refracted/reflected image.
        let ss_glass_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers ss glass sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        // Shadow depth texture for the directional light's depth pre-pass.
        //
        // 4096, up from 1024. The map covers whatever the light's `camera_size`
        // spans, so its resolution is only meaningful as a ratio: a vehicle
        // 8 m across got 8 mm of shadow per texel at 1024, which is coarse
        // enough to see as staircase edges on every strut. At 4096 it is 2 mm,
        // and the cost is one 64 MB depth target and a depth pass over the same
        // geometry — nothing next to the colour pass at 8K.
        //
        // KEEP IN STEP with SHADOW_TEXEL in shader.rs, which sizes the PCF
        // kernel and cannot read this constant.
        //
        // Allocated at one texel until a light casts: `ensure_shadow_arrays`
        // sizes them for the frame's casters. A scene without shadows (most
        // UI views) no longer pays 2 × 128 MB of depth for nothing — per
        // renderer, so several views multiplied it.
        let dir_shadows = ShadowArray::new(&device, "threers shadow depth", 1, 1);
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers shadow sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let spot_shadows = ShadowArray::new(&device, "threers spot shadow depth", 1, 1);
        let spot_shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers spot shadow sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Point shadows: six faces per light, in a 2D array.
        let point_shadows =
            ShadowArray::new(&device, "threers point shadow depth", POINT_SHADOW_SIZE, 6);
        let point_shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers point shadow sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let default_cube_uv_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers default cube uv"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &default_cube_uv_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[0u8, 0, 0, 255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let default_cube_uv_view =
            default_cube_uv_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let make_env_uniform = |u: EnvUniforms, label: &str| {
            wgpu::util::DeviceExt::create_buffer_init(
                &*device,
                &wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::bytes_of(&u),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                },
            )
        };
        // The scene's, rewritten every frame; and one with no environment,
        // for the shadow passes, which read none.
        let env_uniform = make_env_uniform(EnvUniforms::NONE, "threers scene env uniforms");
        let no_env_uniform = make_env_uniform(EnvUniforms::NONE, "threers no env uniforms");
        let env_bind_group = env_bind_group!(
            device,
            &env_bgl,
            "threers env bg",
            &default_env_cube.view,
            &env_sampler,
            &dir_shadows.view,
            &shadow_sampler,
            &spot_shadows.view,
            &spot_shadow_sampler,
            &point_shadows.view,
            &point_shadow_sampler,
            &default_cube_uv_view,
            &env_uniform,
            &ltc.1,
        );
        // The trimmed custom-pipeline environment, where there is one. Rebuilt
        // alongside `env_bind_group` whenever the scene environment changes.
        let custom_env_bind_group = custom_env_bgl.as_ref().map(|bgl| {
            custom_env_bind_group!(
                device,
                bgl,
                "threers custom env bg",
                &default_env_cube.view,
                &env_sampler,
                &default_cube_uv_view,
                &env_uniform,
            )
        });

        // Dummy 1x1 depth textures used by `shadow_env_bind_group` so the
        // shadow passes can satisfy the pipeline layout without binding the
        // textures they're currently writing to (WebGPU forbids simultaneous
        // RenderAttachment + TextureBinding usage of the same resource).
        let dummy_depth = ShadowArray::new(&device, "threers dummy depth", 1, 1);
        let shadow_env_bind_group = env_bind_group!(
            device,
            &env_bgl,
            "threers shadow env bg (dummies)",
            &default_env_cube.view,
            &env_sampler,
            &dummy_depth.view,
            &shadow_sampler,
            &dummy_depth.view,
            &spot_shadow_sampler,
            &dummy_depth.view,
            &point_shadow_sampler,
            &default_cube_uv_view,
            &no_env_uniform,
            &ltc.1,
        );
        // Each draw's uniforms at an offset the device accepts for a binding.
        let mesh_stride = {
            let align = device.limits().min_uniform_buffer_offset_alignment as u64;
            (std::mem::size_of::<MeshUniforms>() as u64).div_ceil(align) * align
        };

        let mut renderer = Self {
            device,
            queue,
            pipeline_glass_depth,
            pipeline_glass_color,
            pipeline_glass_depth_f16,
            pipeline_glass_color_f16,
            pipeline_glass_depth_msaa,
            pipeline_glass_color_msaa,
            pipeline_glass_tint,
            pipeline_glass_tint_f16,
            pipeline_glass_tint_msaa,
            encoded_glass_msaa,
            pipeline_oit,
            pipeline_oit_resolve,
            pipeline_oit_resolve_f16,
            pipeline_oit_resolve_msaa,
            pipeline_oit_msaa,
            oit_msaa_targets: None,
            oit_msaa_targets_spare: None,
            depth_resolve,
            oit_resolve_bgl,
            oit_targets: None,
            ss_color: None,
            ss_glass_sampler,
            ss_depth: None,
            ss_back_depth: None,
            ss_layer: None,
            refraction_layers: 2,
            pipeline_ss_back_depth,
            pipeline_ss_blit,
            pipeline_ss_blit_f16,
            pipeline_ss_glass,
            pipeline_ss_glass_f16,
            pipeline_glass_shadow,
            ss_glass_shadow: None,
            ss_blit_bgl,
            taa_enabled: false,
            taa_frame: 0,
            taa_converge: false,
            taa_blit: true,
            taa_hist_format: None,
            jitter_override: None,
            taa_bind_cache: None,
            taa_prev_view_proj: [0.0; 16],
            mb_inv_vp: [0.0; 16],
            mb_prev_vp: [0.0; 16],
            mb_last_vp: [0.0; 16],
            taa_prev_cam: [0.0; 3],
            taa_history: None,
            taa_uniform,
            taa_bgl,
            taa_sampler,
            pipeline_taa,
            pipeline_taa_f16,
            pipeline_tri,
            pipeline_tri_alpha,
            pipeline_tri_nocull,
            pipeline_tri_sky,
            pipeline_tri_wire,
            pipeline_line,
            pipeline_point,
            pipeline_point_alpha,
            pipeline_wide_line,
            pipeline_wide_line_alpha,
            pipeline_sprite,
            pipeline_tri_msaa,
            pipeline_skinned_msaa,
            pipeline_instanced_msaa,
            pipeline_tri_alpha_msaa,
            pipeline_tri_nocull_msaa,
            pipeline_tri_sky_msaa,
            pipeline_tri_wire_msaa,
            pipeline_line_msaa,
            pipeline_point_msaa,
            pipeline_wide_line_msaa,
            msaa: 1,
            // Off by default: enabling tone mapping would silently re-grade
            // every existing scene and the parity renders along with them.
            tone_mapping: ToneMapping::None,
            specular_antialiasing: false,
            tone_map_pass: true,
            exposure: 1.0,
            msaa_targets: None,
            msaa_targets_spare: None,
            overlay_msaa: false,
            last_frame_msaa: false,
            fxaa_scratch: None,
            pipeline_tri_f16,
            pipeline_tri_alpha_f16,
            pipeline_tri_nocull_f16,
            pipeline_tri_sky_f16,
            pipeline_tri_wire_f16,
            pipeline_line_f16,
            pipeline_point_f16,
            pipeline_point_alpha_f16,
            pipeline_wide_line_f16,
            pipeline_wide_line_alpha_f16,
            pipeline_sprite_f16,
            pipeline_instanced_f16,
            pipeline_skinned_f16,
            pipeline_shadow,
            pipeline_shadow_spot,
            pipeline_shadow_point,
            pipeline_shadow_skinned,
            pipeline_shadow_spot_skinned,
            pipeline_shadow_point_skinned,
            pipeline_shadow_instanced,
            pipeline_shadow_spot_instanced,
            shadow_cut,
            pipeline_shadow_point_instanced,
            pipeline_instanced,
            pipeline_skinned,
            pipeline_instanced_wire,
            pipeline_instanced_wire_f16,
            pipeline_instanced_alpha,
            pipeline_instanced_alpha_f16,
            pipeline_skinned_alpha,
            pipeline_skinned_alpha_f16,
            encoded_blend_pipelines,
            pipeline_skinned_wire,
            pipeline_skinned_wire_f16,
            pipeline_postfx,
            pipeline_postfx_add,
            pipeline_postfx_outline_add,
            pipeline_postfx_multiply,
            pipeline_postfx_f16,
            pipeline_postfx_add_f16,
            postfx_bgl,
            postfx_sampler,
            postfx_nearest_sampler,
            postfx_noise_sampler,
            postfx_ext: Default::default(),
            ssao_kernel_buffer,
            ssao_noise_tex,
            ssao_noise_view,
            glitch_disp_tex,
            glitch_disp_view,
            rgba_upload_tex,
            rgba_upload_view,
            #[cfg(feature = "captions")]
            caption_pass: None,
            overlay_pass: None,
            mesh_skinned_bgl,
            sprite_quad,
            frame_bgl,
            mesh_bgl,
            tex_bgl,
            env_bgl,
            env_bgl_glass,
            ltc,
            custom_tex_bind_group,
            custom_tex_bgl,
            mesh_shader: shader.clone(),
            mesh_pipeline_layout: pipeline_layout.clone(),
            biased_pipelines: HashMap::new(),
            state_pipelines: state_pipes.into_inner(),
            msaa_color_unorm: None,
            encoded_scratch: None,
            encoded_copy: None,
            custom_mesh_bgl,
            custom_pipeline_layout,
            custom_ss_pipeline_layout,
            custom_lit_pipeline_layout,
            custom_lit_mesh_bgl,
            custom_lit: HashMap::new(),
            custom_pipelines: HashMap::new(),
            custom_encoded_pipelines: HashMap::new(),
            custom_msaa_pipelines: HashMap::new(),
            custom_failed: std::collections::HashSet::new(),
            depth_view,
            depth_texture,
            hw_depth_tex,
            depth_size: (width, height),
            geom_cache: IdentityKeyed::default(),
            frustum_culling: true,
            cull_stats: (0, 0),
            timer: None,
            shadow_timer: None,
            shadow_map: Default::default(),
            shadow_sig: None,
            shadow_pass_hashes: Vec::new(),
            shadow_stats: (0, 0),
            shadow_updates: 0,
            skin_attr_cache: IdentityKeyed::default(),
            morph_cache: IdentityKeyed::default(),
            instance_cache: HashMap::new(),
            instance_frame: 0,
            instance_stamps: 0,
            uv_cache: IdentityKeyed::default(),
            default_uv_sets,
            default_morph,
            wire_span_cache: IdentityKeyed::default(),
            tex_cache: HashMap::new(),
            tex_last_used: HashMap::new(),
            cache_retention: DEFAULT_CACHE_RETENTION,
            tex_pinned: std::collections::HashSet::new(),
            stream_uploads: HashMap::new(),
            stream_scratch: Vec::new(),
            yuv: None,
            #[cfg(target_arch = "wasm32")]
            frame_canvas: None,
            mipgen: None,
            rt_view_cache: HashMap::new(),
            rt_linear_view_cache: HashMap::new(),
            rt_registry: HashMap::new(),
            rt_depth_view_cache: HashMap::new(),
            cube_rt_view_cache: HashMap::new(),
            env_max_mip: 0.0,
            cube_cache: HashMap::new(),
            env_bind_group,
            shadow_env_bind_group,
            custom_env_bgl,
            custom_env_bind_group,
            env_uniform,
            material_env_groups: HashMap::new(),
            pmrem_cache: HashMap::new(),
            env_sampler,
            default_env_cube,
            default_cube_uv_view,
            shadow_pass_buffer: None,
            depth_replay,
            depth_replay_pass: None,
            force_depth_replay: false,
            msaa_depth_out: None,
            depth_copy,
            shadow_pass_bgl,
            shadow_pass_stride,
            shadow_type: crate::lights::ShadowType::default(),
            env_cached_key: None,
            env_dirty: true,
            env_cached_rt_id: None,
            dir_shadows,
            spot_shadows,
            point_shadows,
            cube_pass: false,
            viewport_state: Default::default(),
            shadow_sampler,
            spot_shadow_sampler,
            point_shadow_sampler,
            frame_buffer,
            frame_bind_group,
            frame_bind_group_dummy,
            frame_buffer_for_bg,
            transmission_sampler,
            transmission: None,
            capturing_transmission: None,
            reflective_transparency: false,
            per_mesh: Vec::new(),
            mesh_arena: None,
            mesh_stride,
            mesh_bytes: Vec::new(),
            tex_bg_cache: HashMap::new(),
            main_shader: shader.clone(),
            map_variants: HashMap::new(),
            variant_bg_cache: HashMap::new(),
            sampler_linear_repeat,
            samplers,
            default_white_srgb,
            default_white_linear,
            default_data_tex_view,
            default_volume_view,
            sampler_volume,
            last_camera_planes: (0.1, 1000.0),
            last_camera_lens: (0.414, 1.0),
            default_normal,
            default_transparent,
            color_format,
        };

        // Screen-space ambient occlusion needs a sample kernel and a rotation
        // noise texture, and until now the only code that ever uploaded them
        // was `wasm.rs`. Every native caller got a zero kernel — so every
        // sample offset was the zero vector, every sample landed on the
        // fragment it started from, `delta` never exceeded `min_dist`, and the
        // effect returned a constant for every pixel whatever radius it was
        // given. That is why SSAO has been switched off by default: it was not
        // subtle, it was doing nothing.
        //
        // Defaults are installed here so the feature works out of the box; the
        // setters still override them.
        renderer.timer = GpuTimer::new(&renderer.device, &renderer.queue);
        renderer.shadow_timer = GpuTimer::new(&renderer.device, &renderer.queue);
        renderer.set_ssao_kernel(&default_ssao_kernel());
        renderer.set_ssao_noise(&default_ssao_noise());
        renderer
    }
}
