//! Screen-space ambient occlusion from depth alone, as a compute pass.
//!
//! With one hard sun and a dim fill there is nothing to darken a crevice: two
//! surfaces meeting at right angles are lit identically to a flat plate, so
//! close-packed hardware reads as decals on a surface rather than objects
//! standing on it. Contact darkening is what gives the eye scale, and in a
//! vacuum scene — where there is no aerial perspective and no shadow softening
//! to lean on — it is doing more work than usual.
//!
//! # The normal, reconstructed
//!
//! Textbook SSAO wants a normal buffer and this renderer has no G-buffer, so
//! the first attempt here compared raw depths and skipped normals entirely.
//! That does not work, and the failure is worth recording: with nothing to bias
//! against, a surface slanted away from the camera has neighbours nearer than
//! itself EVERYWHERE and occludes itself uniformly. Measured on a deck shot it
//! darkened 71.6% of pixels by up to 29 levels — a global dim wearing
//! occlusion's clothes.
//!
//! So the normal is reconstructed from depth, taking care at the one place
//! naive reconstruction goes wrong. A plain central difference straddles a
//! silhouette and returns a normal facing nowhere real, which is worst exactly
//! at the creases the effect exists to find. The fix is standard: take forward
//! AND backward differences on each axis and keep whichever has the smaller
//! depth change, so the estimate always stays on the near side of an edge
//! rather than spanning it.
//!
//! With a normal in hand this becomes ordinary hemisphere SSAO: a sample only
//! occludes if it lies in front of the surface, which is what stops a plane
//! from shadowing itself. Measured on the same deck shot that exposed the first
//! attempt, the difference is 71.6% of pixels darkened before and 8.5% after,
//! with the darkening now on the turret bases, the deck edge and the gaps
//! between fittings, and none of it on the open ocean behind.
//!
//! Depth is linearised before comparison. Comparing raw depth-buffer values
//! would make the effect fall apart with distance, because the buffer's
//! precision is hyperbolic: the same numeric difference is millimetres near the
//! camera and kilometres far from it.

/// Depth-driven ambient occlusion, producing a single-channel factor.
pub struct Ssao {
    /// Kept so the depth range can be refreshed per frame without rebuilding.
    ubo: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    view: wgpu::TextureView,
    groups: (u32, u32),
    size: (u32, u32),
}

const TILE: u32 = 8;
/// AO is low-frequency; computing it at half resolution costs a quarter and
/// loses nothing the composite can see.
const SHRINK: u32 = 2;

impl Ssao {
    /// `radius` is in world units, `strength` scales the darkening, and
    /// `near`/`far` must match the camera the depth was rendered with — the
    /// linearisation is wrong otherwise, and wrong quietly.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        depth: &wgpu::Texture,
        near: f32,
        far: f32,
        radius: f32,
        strength: f32,
        // `tan(fov_y / 2)` and the viewport aspect, for the view-space
        // reconstruction. Wrong values tilt every normal and the occlusion
        // slides off the creases without any obvious sign it has.
        tan_half_fov: f32,
        aspect: f32,
    ) -> Option<Self> {
        if depth.sample_count() != 1 {
            return None;
        }
        let (dw, dh) = (depth.width(), depth.height());
        let (aw, ah) = ((dw / SHRINK).max(1), (dh / SHRINK).max(1));

        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor {
            label: Some("threers ssao depth"),
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("threers ssao"),
            size: wgpu::Extent3d {
                width: aw,
                height: ah,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // R32Float, not R8Unorm: WebGPU's list of storage-texture formats
            // does not include the 8-bit single channel, and wgpu rejects it at
            // creation. Four bytes a texel at half resolution is nothing.
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers ssao"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers ssao bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let params = [near, far, radius, strength, tan_half_fov, aspect, 0.0, 0.0];
        let ubo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threers ssao params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        ubo.slice(..)
            .get_mapped_range_mut().expect("buffer range is mapped")
            .slice(..)
            .copy_from_slice(bytemuck::cast_slice(&params));
        ubo.unmap();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers ssao bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: ubo.as_entire_binding(),
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers ssao layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        Some(Self {
            ubo,
            pipeline: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("threers ssao pipeline"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            }),
            bind_group,
            view: tex.create_view(&Default::default()),
            groups: (aw.div_ceil(TILE), ah.div_ceil(TILE)),
            size: (dw, dh),
        })
    }

    /// Refresh the camera's depth range. MUST be called on any frame where the
    /// camera's near or far changed, which for a shot that reframes itself is
    /// every frame — and getting it wrong does not fail loudly, it produces
    /// occlusion that merely looks a bit off.
    #[allow(clippy::too_many_arguments)]
    pub fn set_range(
        &self,
        queue: &wgpu::Queue,
        near: f32,
        far: f32,
        radius: f32,
        strength: f32,
        tan_half_fov: f32,
        aspect: f32,
    ) {
        let params = [near, far, radius, strength, tan_half_fov, aspect, 0.0, 0.0];
        queue.write_buffer(&self.ubo, 0, bytemuck::cast_slice(&params));
    }

    pub fn record(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("threers ssao pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.groups.0, self.groups.1, 1);
    }

    /// The occlusion factor: 1 is unoccluded, lower is darker.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Size of the depth buffer this was built for, so a resize is detectable.
    pub fn depth_size(&self) -> (u32, u32) {
        self.size
    }
}

const SHADER: &str = r#"
struct P {
    near : f32, far : f32, radius : f32, strength : f32,
    tan_half_fov : f32, aspect : f32, pad0 : f32, pad1 : f32,
};
@group(0) @binding(0) var depth : texture_depth_2d;
@group(0) @binding(1) var dst   : texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> p : P;

// Depth-buffer values are hyperbolic: the same numeric step is millimetres up
// close and kilometres far away. Everything below works in linear view depth.
fn linear_depth(d : f32) -> f32 {
    if (d >= 1.0) { return p.far; }
    return (p.near * p.far) / max(p.far - d * (p.far - p.near), 1e-6);
}

// View-space position from a pixel and its linear depth. Only the ratios
// matter for a normal, so this needs the projection's shape and not the matrix.
fn view_pos(c : vec2<i32>, dim : vec2<u32>) -> vec3<f32> {
    let z = linear_depth(textureLoad(depth, clamp(c, vec2<i32>(0), vec2<i32>(dim) - 1), 0));
    let uv = (vec2<f32>(c) + 0.5) / vec2<f32>(dim) * 2.0 - 1.0;
    return vec3<f32>(uv.x * p.aspect * p.tan_half_fov * z, -uv.y * p.tan_half_fov * z, z);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let out_dim = textureDimensions(dst);
    if (gid.x >= out_dim.x || gid.y >= out_dim.y) { return; }
    let in_dim = textureDimensions(depth);
    let c = vec2<i32>(vec2<u32>(gid.x * 2u, gid.y * 2u));

    let p0 = view_pos(c, in_dim);
    // The sky has no crevices, and the linearisation is least trustworthy at
    // the far plane — leave it rather than invent occlusion there.
    if (p0.z >= p.far * 0.999) {
        textureStore(dst, vec2<i32>(gid.xy), vec4<f32>(1.0, 0.0, 0.0, 1.0));
        return;
    }

    // NEAREST-SIDE DERIVATIVES. A central difference across a silhouette
    // returns a normal belonging to neither surface, and silhouettes are where
    // the creases are. Taking both sides and keeping the smaller depth step
    // holds the estimate on the near surface.
    let xr = view_pos(c + vec2<i32>(1, 0), in_dim);
    let xl = view_pos(c - vec2<i32>(1, 0), in_dim);
    let yd = view_pos(c + vec2<i32>(0, 1), in_dim);
    let yu = view_pos(c - vec2<i32>(0, 1), in_dim);
    var dx = xr - p0;
    if (abs(xl.z - p0.z) < abs(xr.z - p0.z)) { dx = p0 - xl; }
    var dy = yd - p0;
    if (abs(yu.z - p0.z) < abs(yd.z - p0.z)) { dy = p0 - yu; }
    let n = normalize(cross(dx, dy));

    // Screen-space radius shrinks with distance so the effect keeps a fixed
    // size in the WORLD rather than a fixed number of pixels.
    let px = clamp(p.radius / p0.z * f32(in_dim.y) * 0.5 / max(p.tan_half_fov, 1e-4),
                   2.0, 64.0);

    var occ = 0.0;
    for (var i = 0; i < 12; i = i + 1) {
        let a = f32(i) * 2.39996;                  // golden angle: even coverage
        let r = px * sqrt((f32(i) + 0.5) / 12.0);  // uniform over the disc
        let s = c + vec2<i32>(vec2<f32>(cos(a), sin(a)) * r);
        let ps = view_pos(s, in_dim);
        let v = ps - p0;
        let d = length(v);
        if (d < 1e-4) { continue; }
        // A sample occludes only if it sits IN FRONT of this surface — that
        // dot is the whole difference from the first attempt, and the reason a
        // slanted plane no longer shadows itself.
        let cosang = dot(n, v / d);
        // Falloff so a distant foreground object does not darken everything
        // behind it: that is a silhouette, not a contact.
        let fall = clamp(1.0 - d / p.radius, 0.0, 1.0);
        occ = occ + max(cosang, 0.0) * fall;
    }
    let ao = clamp(1.0 - p.strength * (occ / 12.0), 0.0, 1.0);
    textureStore(dst, vec2<i32>(gid.xy), vec4<f32>(ao, 0.0, 0.0, 1.0));
}
"#;
