//! [`PanelPass`]: a [`Panel`] drawn into each eye of any [`XrSource`],
//! with wgpu, so wherever wgpu draws (WebGPU, WebGL2, Metal, Vulkan, D3D12).
//!
//! The UI is drawn first into [`PanelPass::ui_target`] — by
//! [`UiPass`](crate::renderer::UiPass) or anything else that draws into a
//! texture — then [`PanelPass::draw`] puts it on the panel in every eye,
//! with the pointers, over a floor with a grid and a sky. Each eye has its
//! own uniform block (its view and projection); one pass draws them all, a
//! viewport each. No depth buffer: the room, the panel and the rays are
//! drawn in that order, which is the order they stand in.

use super::{Panel, Pointer, XrSource};
use crate::math::{Matrix4, Vector3};

const MAX_POINTERS: usize = 2;
const MAX_EYES: usize = 4;
/// Each eye's uniform block, spaced for dynamic offsets.
const BLOCK: u64 = 512;

/// One eye's uniforms (see the shader's `Eye`).
#[repr(C)]
#[derive(Clone, Copy)]
struct EyeBlock {
    view_proj: [f32; 16],
    inv_view_proj: [f32; 16],
    eye: [f32; 4],
    center: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    dot: [[f32; 4]; MAX_POINTERS],
    ray_from: [[f32; 4]; MAX_POINTERS],
    ray_to: [[f32; 4]; MAX_POINTERS],
}

impl EyeBlock {
    fn bytes(&self) -> &[u8] {
        // SAFETY: plain f32s, `repr(C)` with no padding.
        unsafe { std::slice::from_raw_parts((self as *const Self).cast::<u8>(), std::mem::size_of::<Self>()) }
    }
}

const SHADER: &str = r#"
struct Eye {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    // w: 1 to draw the room (VR), 0 to leave it see-through (AR).
    eye: vec4<f32>,
    // w: 1 when the target is sRGB (see `stored`).
    center: vec4<f32>,
    // Unit axes; w: half the panel's width and height in metres.
    right: vec4<f32>,
    up: vec4<f32>,
    // A pointer's dot: the point on the panel (0..1), pressed, shown.
    dot: array<vec4<f32>, 2>,
    // A ray: from and to; w: shown.
    ray_from: array<vec4<f32>, 2>,
    ray_to: array<vec4<f32>, 2>,
}
@group(0) @binding(0) var<uniform> u: Eye;
@group(0) @binding(1) var ui: texture_2d<f32>;
@group(0) @binding(2) var ui_sampler: sampler;

// Colours here are as stored (sRGB-encoded): the UI's texture holds them
// so, and the room's are given so. Into an sRGB target they go decoded,
// for the hardware to store the same bytes again (center.w says which).
fn stored(c: vec3<f32>) -> vec3<f32> {
    if (u.center.w < 0.5) {
        return c;
    }
    return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}

struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// The room: a sky, and a floor with a half-metre grid fading into the
// distance, from each pixel's view ray. Colours are as stored (sRGB).
@vertex fn vs_room(@builtin(vertex_index) i: u32) -> Out {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
    var o: Out;
    o.pos = vec4<f32>(p, 0.0, 1.0);
    o.uv = p;
    return o;
}

@fragment fn fs_room(in: Out) -> @location(0) vec4<f32> {
    let near = u.inv_view_proj * vec4<f32>(in.uv, 0.0, 1.0);
    let far = u.inv_view_proj * vec4<f32>(in.uv, 1.0, 1.0);
    let dir = normalize(far.xyz / far.w - near.xyz / near.w);
    let sky = mix(vec3<f32>(0.11, 0.13, 0.19), vec3<f32>(0.27, 0.32, 0.43), clamp(dir.y * 1.6 + 0.25, 0.0, 1.0));
    let down = dir.y < -0.001;
    let t = select(1000.0, -u.eye.y / min(dir.y, -0.001), down);
    let at = (u.eye.xyz + dir * t).xz * 2.0;
    let cell = abs(fract(at + 0.5) - 0.5) / max(fwidth(at), vec2<f32>(1e-4));
    let line = 1.0 - min(min(cell.x, cell.y), 1.0);
    let fade = exp(-t * 0.12);
    let ground = mix(vec3<f32>(0.15, 0.17, 0.22), vec3<f32>(0.42, 0.47, 0.58), line * fade);
    return vec4<f32>(stored(select(sky, mix(sky, ground, clamp(fade * 1.4, 0.0, 1.0)), down)), 1.0);
}

// The panel, with rounded corners and the pointers' dots on it.
@vertex fn vs_panel(@builtin(vertex_index) i: u32) -> Out {
    var corners = array<vec2<f32>, 6>(vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0));
    let q = corners[i];
    let w = u.center.xyz + u.right.xyz * (q.x * u.right.w) + u.up.xyz * (q.y * u.up.w);
    var o: Out;
    o.pos = u.view_proj * vec4<f32>(w, 1.0);
    o.uv = vec2<f32>(q.x * 0.5 + 0.5, 0.5 - q.y * 0.5);
    return o;
}

@fragment fn fs_panel(in: Out) -> @location(0) vec4<f32> {
    var c = textureSample(ui, ui_sampler, in.uv).rgb;
    let size = vec2<f32>(u.right.w, u.up.w) * 2.0;
    let p = (in.uv - 0.5) * size;
    let r = 0.025;
    let q = abs(p) - (size * 0.5 - r);
    let d = length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
    let aa = max(fwidth(d), 1e-5);
    let a = 1.0 - smoothstep(-aa, aa, d);
    for (var k = 0; k < 2; k++) {
        let dot = u.dot[k];
        if (dot.w > 0.5) {
            let m = length((in.uv - dot.xy) * size);
            let edge = 1.0 - smoothstep(0.0055, 0.0055 + aa * 2.0, m);
            let core = 1.0 - smoothstep(0.004, 0.004 + aa * 2.0, m);
            let fill = select(vec3<f32>(0.98, 0.98, 0.98), vec3<f32>(0.23, 0.51, 0.96), dot.z > 0.5);
            c = mix(c, vec3<f32>(0.08, 0.09, 0.12), edge);
            c = mix(c, fill, core);
        }
    }
    return vec4<f32>(stored(c) * a, a);
}

// A ray: a thin strip from the controller, turned to face the eye.
@vertex fn vs_ray(@builtin(vertex_index) i: u32, @builtin(instance_index) k: u32) -> Out {
    var corners = array<vec2<f32>, 6>(vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0));
    let q = corners[i];
    let a = u.ray_from[k].xyz;
    let b = u.ray_to[k].xyz;
    let s = cross(b - a, (a + b) * 0.5 - u.eye.xyz);
    let side = s * (0.002 / max(length(s), 1e-6));
    var o: Out;
    o.pos = u.view_proj * vec4<f32>(mix(a, b, q.x) + side * q.y, 1.0);
    if (u.ray_from[k].w < 0.5) {
        o.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    }
    o.uv = q;
    return o;
}

@fragment fn fs_ray(in: Out) -> @location(0) vec4<f32> {
    let a = (1.0 - abs(in.uv.y)) * mix(0.85, 0.2, in.uv.x);
    return vec4<f32>(stored(vec3<f32>(1.0)) * a, a);
}
"#;

/// The areas the system highlights when looked at: each a rounded box on
/// the panel, drawn as its value into an integer target.
const HOVER_SHADER: &str = r#"
struct Eye {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    center: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
}
@group(0) @binding(0) var<uniform> u: Eye;

struct Out {
    @builtin(position) pos: vec4<f32>,
    // Where in the area, from its middle, in UI pixels.
    @location(0) at: vec2<f32>,
    @location(1) @interpolate(flat) half_radius: vec3<f32>,
    @location(2) @interpolate(flat) value: u32,
}

// `rect`: x, y, width, height in UI pixels; `more`: the radius, the UI's
// width and height; `value`: what to write.
@vertex fn vs_hover(@builtin(vertex_index) i: u32, @location(0) rect: vec4<f32>, @location(1) more: vec3<f32>, @location(2) value: u32) -> Out {
    var corners = array<vec2<f32>, 6>(vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0));
    let q = corners[i];
    let px = rect.xy + q * rect.zw;
    let uv = px / more.yz;
    let w = u.center.xyz + u.right.xyz * ((uv.x * 2.0 - 1.0) * u.right.w) + u.up.xyz * ((1.0 - uv.y * 2.0) * u.up.w);
    var o: Out;
    o.pos = u.view_proj * vec4<f32>(w, 1.0);
    o.at = (q - 0.5) * rect.zw;
    o.half_radius = vec3<f32>(rect.zw * 0.5, more.x);
    o.value = value;
    return o;
}

@fragment fn fs_hover(in: Out) -> @location(0) vec4<u32> {
    let r = in.half_radius.z;
    let q = abs(in.at) - (in.half_radius.xy - r);
    let d = length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
    if (d > 0.0) {
        discard;
    }
    return vec4<u32>(in.value, 0u, 0u, 0u);
}
"#;

/// One area for [`PanelPass::draw_hover_regions`]: its id, where it is on
/// the panel in UI pixels, and its corners' radius.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HoverArea {
    pub id: u64,
    pub rect: [f32; 4],
    pub radius: f32,
}

/// The hover pass's pieces, made when first used.
struct HoverPass {
    module: wgpu::ShaderModule,
    pipeline_layout: wgpu::PipelineLayout,
    uniforms: wgpu::Buffer,
    group: wgpu::BindGroup,
    pipelines: std::collections::HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    instances: Option<(wgpu::Buffer, u64)>,
}

/// Draws a [`Panel`] into each eye, with wgpu: the UI's texture on it, the
/// pointers' rays and dots, over a floor and a sky (or nothing, where the
/// world shows through). See the module docs.
pub struct PanelPass {
    format: wgpu::TextureFormat,
    room: wgpu::RenderPipeline,
    panel: wgpu::RenderPipeline,
    rays: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniforms: wgpu::Buffer,
    sampler: wgpu::Sampler,
    /// The UI's texture, its size, and the bind group reading it.
    texture: Option<(wgpu::Texture, wgpu::TextureView, (u32, u32), wgpu::BindGroup)>,
    /// Its smaller copies (each half the one before), remade from the
    /// whole after each drawing of the UI: a panel seen from afar or at a
    /// slant reads them, smooth, where one texture would shimmer.
    mips: Option<Mips>,
    /// Whether the UI was drawn since its copies were made.
    mips_stale: std::cell::Cell<bool>,
    /// The texture is new since its copies were last made: they have to be.
    mips_new: std::cell::Cell<bool>,
    /// For pipelines made later: the panel and rays drawn over a scene, by
    /// its sample count, depth direction and whether it has a depth.
    module: wgpu::ShaderModule,
    pipeline_layout: wgpu::PipelineLayout,
    in_scene: std::collections::HashMap<(u32, bool, bool), (wgpu::RenderPipeline, wgpu::RenderPipeline)>,
    hover: Option<HoverPass>,
}

impl PanelPass {
    /// For targets (the canvas, a swap chain's image) and a UI texture of
    /// `format`.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> PanelPass {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("threers xr panel"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers xr panel"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<EyeBlock>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("threers xr panel"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let pipeline = |vs: &str, fs: &str, blend: Option<wgpu::BlendState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("threers xr panel"),
                layout: Some(&pl),
                vertex: wgpu::VertexState { module: &module, entry_point: Some(vs), compilation_options: Default::default(), buffers: &[] },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format, blend, write_mask: wgpu::ColorWrites::ALL })],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let over = Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING);
        let (room, panel, rays) = (pipeline("vs_room", "fs_room", None), pipeline("vs_panel", "fs_panel", over), pipeline("vs_ray", "fs_ray", over));
        PanelPass {
            room,
            panel,
            rays,
            in_scene: Default::default(),
            format,
            uniforms: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("threers xr panel eyes"),
                size: BLOCK * MAX_EYES as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            // Between copies too, and along a slant (text on a panel
            // turned away stays readable).
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("threers xr panel"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                anisotropy_clamp: 16,
                ..Default::default()
            }),
            layout,
            texture: None,
            module,
            pipeline_layout: pl,
            hover: None,
            mips: None,
            mips_stale: std::cell::Cell::new(false),
            mips_new: std::cell::Cell::new(false),
        }
    }

    /// The format of [`ui_target`](Self::ui_target): the target's without
    /// sRGB, the UI's bytes kept as they are drawn.
    pub fn ui_format(&self) -> wgpu::TextureFormat {
        self.format.remove_srgb_suffix()
    }

    /// The texture the UI is drawn into, `size` device pixels, in
    /// [`ui_format`](Self::ui_format). Asking for it marks the UI as drawn
    /// again: its smaller copies are remade before the panel is next drawn.
    pub fn ui_target(&mut self, device: &wgpu::Device, size: (u32, u32)) -> &wgpu::TextureView {
        if self.texture.as_ref().is_none_or(|t| t.2 != size) {
            let levels = mip_levels(size);
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers xr panel texture"),
                size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                mip_level_count: levels,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.ui_format(),
                // (COPY_DST: a UI may also be uploaded, as pixels.)
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            // Drawn into: the whole only.
            let view = texture.create_view(&wgpu::TextureViewDescriptor { base_mip_level: 0, mip_level_count: Some(1), ..Default::default() });
            let all = texture.create_view(&Default::default());
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("threers xr panel"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.uniforms,
                            offset: 0,
                            size: wgpu::BufferSize::new(std::mem::size_of::<EyeBlock>() as u64),
                        }),
                    },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&all) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            });
            self.mips = Some(Mips::new(device, &texture, self.ui_format()));
            self.mips_new.set(true);
            self.texture = Some((texture, view, size, group));
        }
        self.mips_stale.set(true);
        &self.texture.as_ref().expect("made above").1
    }

    /// The UI drawn into [`ui_target`](Self::ui_target) this frame is what
    /// was there (a retained draw found nothing changed): its smaller copies
    /// still stand, and are not made again.
    pub fn ui_unchanged(&self) {
        if !self.mips_new.get() {
            self.mips_stale.set(false);
        }
    }

    /// The UI's texture itself (its whole is level 0), for uploading pixels
    /// into; `None` before [`ui_target`](Self::ui_target) made it.
    pub fn ui_texture(&self) -> Option<&wgpu::Texture> {
        self.texture.as_ref().map(|t| &t.0)
    }

    /// The UI's smaller copies, from the whole, if it was drawn since.
    fn update_mips(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.mips_new.set(false);
        if self.mips_stale.replace(false)
            && let Some(m) = &self.mips
        {
            m.make(device, queue);
        }
    }

    /// How many device pixels each UI pixel of `panel` needs for the eyes
    /// of `source` to see every one they can (the panel's width as the
    /// sharpest eye sees it, over its width in UI pixels); `None` before
    /// the panel is placed or with no eyes. Draw the UI at about this
    /// (a little more costs little: the copies smooth it), and again when
    /// it changes much: as the viewer walks up to the panel, it grows.
    pub fn ui_scale_for(source: &(impl XrSource + ?Sized), panel: &Panel) -> Option<f32> {
        let plane = panel.plane()?;
        let eyes = source.eyes(&Matrix4::identity());
        eyes.iter()
            .filter_map(|eye| {
                let e = &eye.world.elements;
                let at = Vector3::new(e[12], e[13], e[14]);
                let distance = (plane.center - at).length();
                // Pixels per unit of tangent across the eye's view.
                let focal = eye.projection[0].abs() * eye.viewport[2] as f32 / 2.0;
                (distance > 1e-3 && focal.is_finite()).then(|| focal * 2.0 * plane.half.0 / distance / panel.size.0.max(1.0))
            })
            .fold(None, |m: Option<f32>, v| Some(m.map_or(v, |m| m.max(v))))
    }

    /// Each of `source`'s eyes into its viewport of `target` (sized to the
    /// source's framebuffer): the room unless the world shows through, then
    /// the panel, with what was drawn into [`ui_target`](Self::ui_target),
    /// and `pointers`. Nothing before the panel is placed.
    pub fn draw(&self, device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::TextureView, source: &(impl XrSource + ?Sized), panel: &Panel, pointers: &[Pointer]) {
        self.draw_eyes(device, queue, target, source, panel, pointers, !source.see_through());
    }

    /// `scene` drawn by `renderer` in each of `source`'s eyes (see
    /// [`Renderer::render_xr`](crate::renderer::Renderer::render_xr)), the
    /// panel and the pointers' rays standing in it: after each eye, they are
    /// drawn depth-tested against that eye's depth, so what stands in front
    /// of the panel hides it and the panel hides what is behind it. Works
    /// with the renderer's MSAA and reversed depth. `origin` places the
    /// reference space (and the panel, which stands in it) in the world.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_in_scene(
        &mut self,
        renderer: &mut crate::renderer::Renderer,
        scene: &mut crate::scene::Scene,
        camera: &crate::cameras::PerspectiveCamera,
        source: &(impl XrSource + ?Sized),
        origin: &Matrix4,
        target: &wgpu::TextureView,
        panel: &Panel,
        pointers: &[Pointer],
    ) -> bool {
        let (device, queue) = (renderer.device_arc(), renderer.queue_arc());
        self.update_mips(&device, &queue);
        renderer.render_xr_with(scene, camera, source, origin, target, |r, i, eye| {
            if i >= MAX_EYES || panel.plane().is_none() || self.texture.is_none() {
                return;
            }
            r.draw_over_last(target, |pass, info| {
                // The eye as the scene saw it, in the reference space the panel
                // stands in, through the projection the render used.
                let mut local = *eye;
                local.world = origin.invert().multiply(&eye.world);
                local.projection = info.projection(&Matrix4 { elements: eye.projection }).elements;
                queue.write_buffer(&self.uniforms, i as u64 * BLOCK, self.eye_block(&local, panel, pointers, false, info.color_format.is_srgb()).bytes());
                self.scene_pipelines(&device, info);
                let (panel_pipe, ray_pipe) = &self.in_scene[&(info.samples, info.reversed, info.depth)];
                let group = &self.texture.as_ref().expect("checked").3;
                pass.set_bind_group(0, group, &[(i as u64 * BLOCK) as u32]);
                pass.set_pipeline(panel_pipe);
                pass.draw(0..6, 0..1);
                pass.set_pipeline(ray_pipe);
                pass.draw(0..6, 0..MAX_POINTERS as u32);
            });
        })
    }

    /// Marks `areas` (in UI pixels on `panel`, painted last on top) for the
    /// system to highlight when they are looked at, in each of `source`'s
    /// eyes, where it can ([`XrSource::hover_target`]: Apple Vision Pro);
    /// elsewhere it does nothing. Call it each frame after the panel is
    /// drawn, with the controls as they are now (threers-ui's
    /// `Ui::hover_regions`). Returns how many were marked.
    ///
    /// The areas are drawn over everything: something standing between the
    /// eye and the panel does not hide its highlight.
    pub fn draw_hover_regions(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, source: &(impl XrSource + ?Sized), panel: &Panel, areas: &[HoverArea]) -> usize {
        let (Some(target), Some(_)) = (source.hover_target(), panel.plane()) else { return 0 };
        // Each area's value this frame; the system knows them by id.
        let mut data: Vec<u8> = Vec::with_capacity(areas.len() * 32);
        let mut count = 0u32;
        for a in areas {
            let Some(value) = source.declare_hover(a.id) else { continue };
            for v in [a.rect[0], a.rect[1], a.rect[2], a.rect[3], a.radius, panel.size.0, panel.size.1] {
                data.extend_from_slice(&v.to_le_bytes());
            }
            data.extend_from_slice(&value.to_le_bytes());
            count += 1;
        }
        let eyes = source.eyes(&Matrix4::identity());
        let blocks: Vec<EyeBlock> = eyes.iter().take(MAX_EYES).map(|eye| self.eye_block(eye, panel, &[], false, false)).collect();
        let hover = self.hover.get_or_insert_with(|| HoverPass::new(device));
        let pipeline = hover.pipeline(device, target.format).clone();
        for (i, b) in blocks.iter().enumerate() {
            queue.write_buffer(&hover.uniforms, i as u64 * BLOCK, b.bytes());
        }
        let need = (data.len() as u64).max(32);
        if hover.instances.as_ref().is_none_or(|(_, size)| *size < need) {
            let size = need.next_power_of_two();
            let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: Some("threers xr hover areas"), size, usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            hover.instances = Some((buffer, size));
        }
        let (buffer, _) = hover.instances.as_ref().expect("made above");
        if !data.is_empty() {
            queue.write_buffer(buffer, 0, &data);
        }
        // The eyes' viewports are in the framebuffer; the target may be
        // another size.
        let fb = source.framebuffer_size();
        let (sx, sy) = (target.size.0 as f32 / fb.0.max(1) as f32, target.size.1 as f32 / fb.1.max(1) as f32);
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("threers xr hover areas"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if count > 0 {
                pass.set_pipeline(&pipeline);
                pass.set_vertex_buffer(0, buffer.slice(..));
                for (i, eye) in eyes.iter().take(MAX_EYES).enumerate() {
                    let Some((x, top, w, h)) = top_left_viewport(eye.viewport, fb) else { continue };
                    let (x, top, w, h) = ((x as f32 * sx) as u32, (top as f32 * sy) as u32, ((w as f32 * sx) as u32).max(1), ((h as f32 * sy) as u32).max(1));
                    if x + w > target.size.0 || top + h > target.size.1 {
                        continue;
                    }
                    pass.set_viewport(x as f32, top as f32, w as f32, h as f32, 0.0, 1.0);
                    pass.set_scissor_rect(x, top, w, h);
                    pass.set_bind_group(0, &hover.group, &[(i as u64 * BLOCK) as u32]);
                    pass.draw(0..6, 0..count);
                }
            }
        }
        queue.submit(Some(encoder.finish()));
        count as usize
    }

    /// The panel's and rays' pipelines for a pass over a scene (see
    /// [`OverlayInfo`](crate::renderer::OverlayInfo)).
    fn scene_pipelines(&mut self, device: &wgpu::Device, info: &crate::renderer::OverlayInfo) {
        let (module, layout, format) = (&self.module, &self.pipeline_layout, info.color_format);
        let (samples, reversed, depth) = (info.samples, info.reversed, info.depth);
        let compare = info.depth_compare();
        self.in_scene.entry((samples, reversed, depth)).or_insert_with(|| {
            let make = |vs: &str, fs: &str, write: bool| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("threers xr panel in scene"),
                    layout: Some(layout),
                    vertex: wgpu::VertexState { module, entry_point: Some(vs), compilation_options: Default::default(), buffers: &[] },
                    fragment: Some(wgpu::FragmentState {
                        module,
                        entry_point: Some(fs),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })],
                    }),
                    primitive: Default::default(),
                    depth_stencil: depth.then(|| wgpu::DepthStencilState {
                        format: crate::renderer::OverlayInfo::DEPTH_FORMAT,
                        depth_write_enabled: Some(write),
                        depth_compare: Some(compare),
                        stencil: Default::default(),
                        bias: Default::default(),
                    }),
                    multisample: wgpu::MultisampleState { count: samples, ..Default::default() },
                    multiview_mask: None,
                    cache: None,
                })
            };
            // The panel hides what is behind it; a ray only hides nothing.
            (make("vs_panel", "fs_panel", true), make("vs_ray", "fs_ray", false))
        });
    }

    /// One eye's uniforms: its view and projection, the panel, the pointers.
    fn eye_block(&self, eye: &super::XrEye, panel: &Panel, pointers: &[Pointer], room: bool, srgb: bool) -> EyeBlock {
        let plane = panel.plane().expect("a placed panel");
        let v4 = |v: Vector3, w: f32| [v.x, v.y, v.z, w];
        let view_proj = Matrix4 { elements: eye.projection }.multiply(&eye.world.invert());
        let e = &eye.world.elements;
        let mut b = EyeBlock {
            view_proj: view_proj.elements,
            inv_view_proj: view_proj.invert().elements,
            eye: [e[12], e[13], e[14], if room { 1.0 } else { 0.0 }],
            center: v4(plane.center, if srgb { 1.0 } else { 0.0 }),
            right: v4(plane.right, plane.half.0),
            up: v4(plane.up, plane.half.1),
            dot: [[0.0; 4]; MAX_POINTERS],
            ray_from: [[0.0; 4]; MAX_POINTERS],
            ray_to: [[0.0; 4]; MAX_POINTERS],
        };
        for (k, p) in pointers.iter().take(MAX_POINTERS).enumerate() {
            if let Some((u, v)) = p.dot {
                b.dot[k] = [u, v, if p.pressed { 1.0 } else { 0.0 }, 1.0];
            }
            b.ray_from[k] = v4(p.from, if p.ray { 1.0 } else { 0.0 });
            b.ray_to[k] = v4(p.to, 1.0);
        }
        b
    }

    fn draw_eyes(&self, device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::TextureView, source: &(impl XrSource + ?Sized), panel: &Panel, pointers: &[Pointer], room: bool) {
        let (Some((_, _, _, group)), Some(_)) = (&self.texture, panel.plane()) else { return };
        self.update_mips(device, queue);
        let (eyes, fb, see_through) = (source.eyes(&Matrix4::identity()), source.framebuffer_size(), !room);
        for (i, eye) in eyes.iter().take(MAX_EYES).enumerate() {
            queue.write_buffer(&self.uniforms, i as u64 * BLOCK, self.eye_block(eye, panel, pointers, room, self.format.is_srgb()).bytes());
        }
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("threers xr panel"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            for (i, eye) in eyes.iter().take(MAX_EYES).enumerate() {
                let Some((x, top, w, h)) = top_left_viewport(eye.viewport, fb) else { continue };
                pass.set_viewport(x as f32, top as f32, w as f32, h as f32, 0.0, 1.0);
                pass.set_scissor_rect(x, top, w, h);
                pass.set_bind_group(0, group, &[(i as u64 * BLOCK) as u32]);
                if !see_through {
                    pass.set_pipeline(&self.room);
                    pass.draw(0..3, 0..1);
                }
                pass.set_pipeline(&self.panel);
                pass.draw(0..6, 0..1);
                pass.set_pipeline(&self.rays);
                pass.draw(0..6, 0..MAX_POINTERS as u32);
            }
        }
        queue.submit(Some(encoder.finish()));
    }
}

/// How many copies a texture `size` has, down to about 32 pixels across
/// (smaller would never be read: the panel is never that small).
fn mip_levels(size: (u32, u32)) -> u32 {
    let mut n = 1;
    let (mut w, mut h) = size;
    while (w > 32 || h > 32) && w > 1 && h > 1 {
        (w, h) = ((w / 2).max(1), (h / 2).max(1));
        n += 1;
    }
    n
}

const MIP_SHADER: &str = r#"
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var smooth_sampler: sampler;
struct Out {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}
@vertex fn vs(@builtin(vertex_index) i: u32) -> Out {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: Out;
    o.pos = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = vec2<f32>(p.x, 1.0 - p.y);
    return o;
}
// The four texels under each one: a box filter, as the copies halve.
@fragment fn fs(in: Out) -> @location(0) vec4<f32> {
    return textureSampleLevel(src, smooth_sampler, in.uv, 0.0);
}
"#;

/// A texture's smaller copies, each drawn from the one before.
struct Mips {
    pipeline: wgpu::RenderPipeline,
    /// Each copy's view to draw into, and the bind group reading the one
    /// before it.
    levels: Vec<(wgpu::TextureView, wgpu::BindGroup)>,
}

impl Mips {
    fn new(device: &wgpu::Device, texture: &wgpu::Texture, format: wgpu::TextureFormat) -> Mips {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("threers xr panel copies"), source: wgpu::ShaderSource::Wgsl(MIP_SHADER.into()) });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("threers xr panel copies"),
            layout: None,
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("threers xr panel copies"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let layout = pipeline.get_bind_group_layout(0);
        let level = |l: u32| texture.create_view(&wgpu::TextureViewDescriptor { base_mip_level: l, mip_level_count: Some(1), ..Default::default() });
        let levels = (1..texture.mip_level_count())
            .map(|l| {
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("threers xr panel copies"),
                    layout: &layout,
                    entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&level(l - 1)) }, wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) }],
                });
                (level(l), group)
            })
            .collect();
        Mips { pipeline, levels }
    }

    fn make(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.levels.is_empty() {
            return;
        }
        let mut encoder = device.create_command_encoder(&Default::default());
        for (view, group) in &self.levels {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("threers xr panel copy"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit(Some(encoder.finish()));
    }
}

impl HoverPass {
    fn new(device: &wgpu::Device) -> HoverPass {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("threers xr hover areas"), source: wgpu::ShaderSource::Wgsl(HOVER_SHADER.into()) });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers xr hover areas"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<EyeBlock>() as u64) },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("threers xr hover areas"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threers xr hover eyes"),
            size: BLOCK * MAX_EYES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers xr hover areas"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &uniforms, offset: 0, size: wgpu::BufferSize::new(std::mem::size_of::<EyeBlock>() as u64) }),
            }],
        });
        HoverPass { module, pipeline_layout, uniforms, group, pipelines: Default::default(), instances: None }
    }

    fn pipeline(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) -> &wgpu::RenderPipeline {
        let (module, layout) = (&self.module, &self.pipeline_layout);
        self.pipelines.entry(format).or_insert_with(|| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("threers xr hover areas"),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("vs_hover"),
                    compilation_options: Default::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: 32,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x3, 2 => Uint32],
                    })],
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some("fs_hover"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        })
    }
}

/// A device's eye rectangle (from the bottom left) as wgpu's (from the top
/// left), `None` if it is not inside the framebuffer.
fn top_left_viewport([x, y, w, h]: [i32; 4], fb: (u32, u32)) -> Option<(u32, u32, u32, u32)> {
    let (x, w, h) = (x.max(0) as u32, w.max(1) as u32, h.max(1) as u32);
    let top = (fb.1 as i32 - y - h as i32).max(0) as u32;
    (x + w <= fb.0 && top + h <= fb.1).then_some((x, top, w, h))
}
