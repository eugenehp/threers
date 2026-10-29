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
    passes: Passes,
    /// The texture `view` points into; held so the result can be copied out.
    tex: wgpu::Texture,
    view: wgpu::TextureView,
    size: (u32, u32),
    /// 1 when the depth is reversed; kept so `set_range` does not drop it.
    reversed: std::cell::Cell<f32>,
}

/// The same pass, recorded either way.
///
/// There is nothing compute-shaped about this: it reads a depth texture and
/// writes one value per output pixel. WebGL2 has no compute stage, so there it
/// is a fullscreen draw instead, and `the_two_ssao_paths_agree` holds the two to
/// each other on a device that has both.
enum Passes {
    Compute {
        pipeline: wgpu::ComputePipeline,
        bind_group: wgpu::BindGroup,
        groups: (u32, u32),
    },
    Raster {
        pipeline: wgpu::RenderPipeline,
        bind_group: wgpu::BindGroup,
    },
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
        // WebGL2 has no compute stage — wgpu signals that by zeroing the compute
        // limits rather than by refusing the device — so there the same pass is
        // recorded as a fullscreen draw instead.
        let raster = !crate::renderer::caps::supports_compute(device);
        Self::new_inner(
            device,
            depth,
            near,
            far,
            radius,
            strength,
            tan_half_fov,
            aspect,
            raster,
        )
    }

    /// The body of [`new`](Self::new), with the path chosen explicitly so the
    /// two can be compared against each other in a test.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_inner(
        device: &wgpu::Device,
        depth: &wgpu::Texture,
        near: f32,
        far: f32,
        radius: f32,
        strength: f32,
        tan_half_fov: f32,
        aspect: f32,
        raster: bool,
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
            // A fragment shader writes through the attachment, a compute shader
            // through a storage binding. COPY_SRC either way so the result can
            // be read back, which is how the two are compared.
            usage: if raster {
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
            } else {
                wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
            },
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers ssao"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    crate::renderer::fullscreen::VS,
                    if raster { RASTER_SHADER } else { SHADER }
                )
                .into(),
            ),
        });
        let stage = if raster {
            wgpu::ShaderStages::FRAGMENT
        } else {
            wgpu::ShaderStages::COMPUTE
        };
        // The depth slot is `Depth` for compute and unfilterable *float* for
        // raster. Both are legal for a depth format, and the difference matters:
        // the shader reads it with `textureLoad`, and GLSL has no `texelFetch`
        // for a depth sampler, so the raster module could not be lowered at all
        // if it were declared as depth.
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: stage,
            ty: wgpu::BindingType::Texture {
                sample_type: if raster {
                    wgpu::TextureSampleType::Float { filterable: false }
                } else {
                    wgpu::TextureSampleType::Depth
                },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }];
        // Binding 1 is the storage output, which exists only for compute: a
        // fragment shader returns its value instead.
        if !raster {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: stage,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::R32Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            });
        }
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: stage,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers ssao bgl"),
            entries: &entries,
        });
        let params = [near, far, radius, strength, tan_half_fov, aspect, 0.0, 0.0];
        let ubo = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threers ssao params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        ubo.slice(..)
            .get_mapped_range_mut()
            .expect("buffer range is mapped")
            .slice(..)
            .copy_from_slice(bytemuck::cast_slice(&params));
        ubo.unmap();

        let mut bg_entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&depth_view),
        }];
        if !raster {
            bg_entries.push(wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&view),
            });
        }
        bg_entries.push(wgpu::BindGroupEntry {
            binding: 2,
            resource: ubo.as_entire_binding(),
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers ssao bg"),
            layout: &bgl,
            entries: &bg_entries,
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers ssao layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let passes = if raster {
            Passes::Raster {
                pipeline: device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("threers ssao pipeline (raster)"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_main"),
                        // No blend state: R32Float is not blendable, and the
                        // pass writes every pixel so there is nothing to blend
                        // with.
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::R32Float,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        ..Default::default()
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                }),
                bind_group,
            }
        } else {
            Passes::Compute {
                pipeline: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("threers ssao pipeline"),
                    layout: Some(&layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                }),
                bind_group,
                groups: (aw.div_ceil(TILE), ah.div_ceil(TILE)),
            }
        };
        Some(Self {
            ubo,
            passes,
            view: tex.create_view(&Default::default()),
            tex,
            size: (dw, dh),
            reversed: std::cell::Cell::new(0.0),
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
        let params = [
            near,
            far,
            radius,
            strength,
            tan_half_fov,
            aspect,
            self.reversed.get(),
            0.0,
        ];
        queue.write_buffer(&self.ubo, 0, bytemuck::cast_slice(&params));
    }

    /// Read the depth as reversed — what
    /// [`Renderer::set_reversed_z`](crate::Renderer::set_reversed_z) renders.
    pub fn set_reversed_depth(&self, queue: &wgpu::Queue, on: bool) {
        let v = if on { 1.0f32 } else { 0.0 };
        self.reversed.set(v);
        queue.write_buffer(&self.ubo, 24, bytemuck::bytes_of(&v));
    }

    pub fn record(&self, encoder: &mut wgpu::CommandEncoder) {
        match &self.passes {
            Passes::Compute {
                pipeline,
                bind_group,
                groups,
            } => {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("threers ssao pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(groups.0, groups.1, 1);
            }
            Passes::Raster {
                pipeline,
                bind_group,
            } => {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("threers ssao pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
        }
    }

    /// The texture [`view`](Self::view) points into. Carries `COPY_SRC`.
    pub fn texture(&self) -> &wgpu::Texture {
        &self.tex
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
fn linear_depth(raw : f32) -> f32 {
    // pad0: the depth is reversed (`Renderer::set_reversed_z`).
    let d = select(raw, 1.0 - raw, p.pad0 > 0.5);
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
        let a = f32(i) * 2.39996 + fract(sin(dot(vec2<f32>(c), vec2<f32>(12.9898, 78.233))) * 43758.5453) * 6.2831853;  // golden angle + per-pixel rotation (break coherent streaks)
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

/// The same pass as [`SHADER`], as a fullscreen draw.
///
/// Everything between the two is identical arithmetic — the linearisation, the
/// nearest-side derivatives, the twelve golden-angle taps and the falloff are
/// the same lines. What differs is only how the invocation learns which pixel it
/// is, where the result goes, and that `depth` is declared as a float texture:
/// WGSL allows `textureLoad` on a depth texture but GLSL has no `texelFetch` for
/// a depth sampler, so this module could not be lowered for WebGL2 otherwise.
/// Depth arrives in `.r`.
const RASTER_SHADER: &str = r#"
struct P {
    near : f32, far : f32, radius : f32, strength : f32,
    tan_half_fov : f32, aspect : f32, pad0 : f32, pad1 : f32,
};
@group(0) @binding(0) var depth : texture_2d<f32>;
@group(0) @binding(2) var<uniform> p : P;

// Depth-buffer values are hyperbolic: the same numeric step is millimetres up
// close and kilometres far away. Everything below works in linear view depth.
fn linear_depth(raw : f32) -> f32 {
    // pad0: the depth is reversed (`Renderer::set_reversed_z`).
    let d = select(raw, 1.0 - raw, p.pad0 > 0.5);
    if (d >= 1.0) { return p.far; }
    return (p.near * p.far) / max(p.far - d * (p.far - p.near), 1e-6);
}

// View-space position from a pixel and its linear depth. Only the ratios
// matter for a normal, so this needs the projection's shape and not the matrix.
fn view_pos(c : vec2<i32>, dim : vec2<u32>) -> vec3<f32> {
    let z = linear_depth(textureLoad(depth, clamp(c, vec2<i32>(0), vec2<i32>(dim) - 1), 0).r);
    let uv = (vec2<f32>(c) + 0.5) / vec2<f32>(dim) * 2.0 - 1.0;
    return vec3<f32>(uv.x * p.aspect * p.tan_half_fov * z, -uv.y * p.tan_half_fov * z, z);
}

@fragment
fn fs_main(@builtin(position) pos : vec4<f32>) -> @location(0) vec4<f32> {
    // `position` is the pixel centre, so truncating gives the texel index. No
    // bounds test is needed: a fragment only runs for pixels that exist, which
    // is what the compute path's `out_dim` guard was for.
    let gid = vec2<u32>(pos.xy);
    let in_dim = textureDimensions(depth);
    let c = vec2<i32>(vec2<u32>(gid.x * 2u, gid.y * 2u));

    let p0 = view_pos(c, in_dim);
    // The sky has no crevices, and the linearisation is least trustworthy at
    // the far plane — leave it rather than invent occlusion there.
    if (p0.z >= p.far * 0.999) {
        return vec4<f32>(1.0, 0.0, 0.0, 1.0);
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
        let a = f32(i) * 2.39996 + fract(sin(dot(vec2<f32>(c), vec2<f32>(12.9898, 78.233))) * 43758.5453) * 6.2831853;  // golden angle + per-pixel rotation (break coherent streaks)
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
    return vec4<f32>(ao, 0.0, 0.0, 1.0);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// The compute and raster paths produce the same occlusion.
    ///
    /// The raster path exists for WebGL2, which is the one backend where it
    /// cannot be compared against the implementation it reproduces — so the two
    /// are compared here, on a device that has both, where a difference is the
    /// port and not the backend.
    ///
    /// The depth buffer is a step: half the field near, half far. That puts a
    /// silhouette down the middle, which is where the nearest-side derivative
    /// matters and where a naive central difference would return a normal
    /// belonging to neither surface — so a port that dropped that detail shows
    /// up here rather than only on real geometry.
    #[test]
    fn the_two_ssao_paths_agree() {
        let Some((device, queue)) = test_device() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        const W: u32 = 64;
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ssao test depth"),
            size: wgpu::Extent3d {
                width: W,
                height: W,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            // Rendered into rather than uploaded: `write_texture` to a depth
            // format is forbidden, so the pattern is drawn with `frag_depth`.
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        fill_depth(&device, &queue, &depth);

        let args = (0.1f32, 50.0f32, 1.0f32, 1.0f32, 0.5f32, 1.0f32);
        let compute = Ssao::new_inner(
            &device, &depth, args.0, args.1, args.2, args.3, args.4, args.5, false,
        )
        .expect("compute ssao");
        let raster = Ssao::new_inner(
            &device, &depth, args.0, args.1, args.2, args.3, args.4, args.5, true,
        )
        .expect("raster ssao");

        let a = run_and_read(&device, &queue, &compute, W / SHRINK);
        let b = run_and_read(&device, &queue, &raster, W / SHRINK);
        assert_eq!(a.len(), b.len());

        // The occlusion must actually vary, or agreement proves nothing: a pair
        // of all-ones buffers would match perfectly and mean the pass did
        // nothing.
        let (lo, hi) = a
            .iter()
            .fold((f32::MAX, f32::MIN), |(l, h), v| (l.min(*v), h.max(*v)));
        assert!(
            hi - lo > 0.02,
            "the compute occlusion is flat ({lo}..{hi}), so this comparison is vacuous"
        );
        let worst = a
            .iter()
            .zip(&b)
            .fold(0.0f32, |w, (x, y)| w.max((x - y).abs()));
        // Both write R32Float, so there is no format rounding to allow for; the
        // bar is tight for the same reason the bloom one is, and was set from
        // the measurement rather than guessed.
        assert!(
            worst < 1e-5,
            "the two ssao paths differ by {worst} at worst — the raster port \
             does not reproduce the compute one"
        );
    }

    /// Draw a depth pattern: a step across the middle with a slant on each side.
    ///
    /// The step puts a silhouette down the centre, which is where the
    /// nearest-side derivative earns its keep, and the slant keeps the
    /// reconstructed normal off the axes so a transposed derivative would show.
    /// `write_texture` cannot fill a depth format, hence a draw.
    fn fill_depth(device: &wgpu::Device, queue: &wgpu::Queue, depth: &wgpu::Texture) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssao test depth fill"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    crate::renderer::fullscreen::VS,
                    r#"
@fragment
fn fs_main(@builtin(position) pos : vec4<f32>) -> @builtin(frag_depth) f32 {
    if (pos.x < 32.0) { return 0.30 + pos.y * 0.002; }
    return 0.80 + pos.y * 0.001;
}
"#
                )
                .into(),
            ),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ssao test depth fill"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let view = depth.create_view(&Default::default());
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssao test depth fill"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            pass.draw(0..3, 0..1);
        }
        queue.submit(Some(encoder.finish()));
    }

    fn run_and_read(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        ssao: &Ssao,
        side: u32,
    ) -> Vec<f32> {
        let bytes_per_row = (side * 4).next_multiple_of(256);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssao readback"),
            size: (bytes_per_row * side) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        ssao.record(&mut encoder);
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: ssao.texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(side),
                },
            },
            wgpu::Extent3d {
                width: side,
                height: side,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));
        readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let data = readback
            .slice(..)
            .get_mapped_range()
            .expect("buffer range is mapped");
        let mut out = Vec::with_capacity((side * side) as usize);
        for row in 0..side {
            let start = (row * bytes_per_row) as usize;
            let end = start + (side * 4) as usize;
            out.extend(
                data[start..end]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
            );
        }
        out
    }

    fn test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let mut d = wgpu::InstanceDescriptor::new_without_display_handle();
        d.backends = wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(d);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("ssao test device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .ok()
    }
}
