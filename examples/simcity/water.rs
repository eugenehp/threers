//! Part of the `simcity` example; see `mod.rs`.
#![allow(dead_code)]

use super::*;

// ---------------------------------------------------------------------------
// Water, driven by a compute shader.
// ---------------------------------------------------------------------------

/// A shared, indexed lattice — not `MeshBuilder`, which emits four unshared
/// vertices per quad and would quadruple the buffer the compute pass walks.
/// Marked `gpu_writable` so its vertex buffer gets `STORAGE` usage and
/// `Renderer::vertex_buffer` will hand it back.
pub(crate) fn water_grid(x0: f32, z0: f32, x1: f32, z1: f32, y: f32, c: Color) -> (BufferGeometry, u32) {
    // Roughly 8 m a cell, capped so a wide bay does not become a million
    // triangles. The swells are ~15 m across, so this still resolves them.
    let nx = (((x1 - x0) / 8.0).round() as u32).clamp(2, 96);
    let nz = (((z1 - z0) / 8.0).round() as u32).clamp(2, 220);
    let (vx, vz) = (nx + 1, nz + 1);
    let count = (vx * vz) as usize;
    let mut pos = Vec::with_capacity(count * 3);
    let mut nrm = Vec::with_capacity(count * 3);
    let mut uv = Vec::with_capacity(count * 2);
    let mut col = Vec::with_capacity(count * 3);
    for j in 0..vz {
        for i in 0..vx {
            let u = i as f32 / nx as f32;
            let v = j as f32 / nz as f32;
            pos.extend_from_slice(&[mix(x0, x1, u), y, mix(z0, z1, v)]);
            nrm.extend_from_slice(&[0.0, 1.0, 0.0]);
            uv.extend_from_slice(&[u, v]);
            col.extend_from_slice(&[c.r, c.g, c.b]);
        }
    }
    let mut idx = Vec::with_capacity((nx * nz * 6) as usize);
    for j in 0..nz {
        for i in 0..nx {
            let a = j * vx + i;
            let b = a + 1;
            let d = a + vx;
            let e = d + 1;
            // Counter-clockwise seen from above, matching `add_slab`.
            idx.extend_from_slice(&[a, d, e, a, e, b]);
        }
    }
    let mut g = BufferGeometry::new();
    g.set_attribute("position", BufferAttribute::new(pos, 3));
    g.set_attribute("normal", BufferAttribute::new(nrm, 3));
    g.set_attribute("uv", BufferAttribute::new(uv, 2));
    g.set_attribute("color", BufferAttribute::new(col, 3));
    g.set_index(idx);
    g.gpu_writable = true;
    (g, (vx * vz))
}

/// The WGSL that moves the water.
///
/// It writes the mesh's vertex buffer in place — position `y` and the normal —
/// so the surface is never read back to the CPU and never re-uploaded. `x` and
/// `z` are read out of the buffer rather than derived from the invocation id,
/// which is what keeps this independent of how the lattice was laid out.
#[allow(dead_code)]
pub(crate) const WATER_WGSL: &str = r#"
struct Params {
    // x: seconds, y: amplitude, z: still-water level, w: unused
    wave : vec4<f32>,
    // x: vertex count, rest unused
    size : vec4<f32>,
};

@group(0) @binding(0) var<storage, read_write> verts : array<f32>;
@group(0) @binding(1) var<uniform> params : Params;

// Four crossed swells. Enough for a river, cheap enough to run every frame.
fn height(x : f32, z : f32, t : f32) -> f32 {
    var h = sin(x * 0.085 + t * 0.90) * 0.42;
    h = h + sin(z * 0.062 - t * 0.70) * 0.34;
    h = h + sin((x * 0.041 + z * 0.037) + t * 1.35) * 0.22;
    h = h + sin((x * 0.130 - z * 0.110) - t * 2.10) * 0.09;
    return h;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let n = u32(params.size.x);
    if (gid.x >= n) { return; }
    // Interleaved vertex: position(3), normal(3), uv(2), colour(4), tangent(4).
    let b = gid.x * 16u;

    let x = verts[b + 0u];
    let z = verts[b + 2u];
    let t = params.wave.x;
    let amp = params.wave.y;

    verts[b + 1u] = params.wave.z + height(x, z, t) * amp;

    // Central differences for the normal: one extra pair of evaluations an
    // axis, and far steadier than differencing neighbouring vertices, which
    // would tie the normal to the lattice spacing.
    let e = 1.5;
    let dx = (height(x + e, z, t) - height(x - e, z, t)) * amp;
    let dz = (height(x, z + e, t) - height(x, z - e, t)) * amp;
    let nrm = normalize(vec3<f32>(-dx, 2.0 * e, -dz));
    verts[b + 3u] = nrm.x;
    verts[b + 4u] = nrm.y;
    verts[b + 5u] = nrm.z;
}
"#;

/// Owns the compute pipeline that animates the water surface.
///
/// Raw wgpu on purpose: the renderer's job is to draw the mesh, and this
/// rewrites that mesh's vertices behind its back between frames. Nothing
/// crosses host memory.
#[allow(dead_code)]
pub(crate) struct WaterSim {
    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) layout: wgpu::BindGroupLayout,
    pub(crate) params: wgpu::Buffer,
    pub(crate) bind: Option<wgpu::BindGroup>,
    pub(crate) vertices: u32,
    pub(crate) level: f32,
}

#[allow(dead_code)]
impl WaterSim {
    pub(crate) fn new(device: &wgpu::Device, vertices: u32, level: f32) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("simcity water"),
            source: wgpu::ShaderSource::Wgsl(WATER_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("simcity water bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
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
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("simcity water layout"),
            // wgpu 30: layouts are optional per slot, and push constants are
            // now "immediates".
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("simcity water pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("simcity water params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            layout,
            params,
            bind: None,
            vertices,
            level,
        }
    }

    /// Run one step. The geometry has to be on the GPU first, which is what
    /// `upload_geometry` guarantees; after that its vertex buffer is ours to
    /// write, and the bind group is built once and kept.
    pub(crate) fn step(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        renderer: &mut threers::Renderer,
        geometry: &Arc<BufferGeometry>,
        time: f32,
        amplitude: f32,
    ) {
        if self.bind.is_none() {
            renderer.upload_geometry(geometry);
            let Some(buffer) = renderer.vertex_buffer(geometry) else {
                return;
            };
            self.bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("simcity water bind"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.params.as_entire_binding(),
                    },
                ],
            }));
        }
        let Some(bind) = &self.bind else {
            return;
        };
        let params: [f32; 8] = [
            time,
            amplitude,
            self.level,
            0.0,
            self.vertices as f32,
            0.0,
            0.0,
            0.0,
        ];
        let mut bytes = [0u8; 32];
        for (i, v) in params.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_ne_bytes());
        }
        queue.write_buffer(&self.params, 0, &bytes);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("simcity water encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("simcity water pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.dispatch_workgroups(self.vertices.div_ceil(64), 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }
}
