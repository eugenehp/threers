//! Points counted into screen cells on the GPU: a compute pass projects
//! every point with the view and adds it (and its weight) to its cell with
//! atomics — deck.gl's GPU aggregation for `ScreenGridLayer`. WebGPU only
//! (WebGL2 has no compute); check [`ScreenBinner::supported`].

use std::sync::{Arc, Mutex};

use wgpu::util::DeviceExt;

/// Weights are summed as fixed point: this many units a weight of 1.
const WEIGHT_SCALE: f32 = 4096.0;

const SHADER: &str = r#"
struct Params {
    view_proj : mat4x4<f32>,
    // Viewport in pixels, and the cell size in pixels.
    viewport : vec2<f32>,
    cell : f32,
    count : u32,
    cols : u32,
    rows : u32,
    weighted : u32,
    _pad : u32,
};

@group(0) @binding(0) var<storage, read> points : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> weights : array<f32>;
@group(0) @binding(2) var<storage, read_write> counts : array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> sums : array<atomic<i32>>;
@group(0) @binding(4) var<uniform> params : Params;

@compute @workgroup_size(256)
fn bin(@builtin(global_invocation_id) id : vec3<u32>) {
    let i = id.x;
    if (i >= params.count) {
        return;
    }
    let c = params.view_proj * vec4<f32>(points[i].xyz, 1.0);
    if (c.w <= 0.0) {
        return;
    }
    // Pixels from the top left, as deck.gl's screen grid counts them.
    let px = vec2<f32>((c.x / c.w + 1.0) * 0.5, (1.0 - c.y / c.w) * 0.5) * params.viewport;
    if (px.x < 0.0 || px.y < 0.0 || px.x >= params.viewport.x || px.y >= params.viewport.y) {
        return;
    }
    let cell = vec2<u32>(floor(px / params.cell));
    if (cell.x >= params.cols || cell.y >= params.rows) {
        return;
    }
    let k = cell.y * params.cols + cell.x;
    atomicAdd(&counts[k], 1u);
    if (params.weighted != 0u) {
        atomicAdd(&sums[k], i32(round(weights[i] * 4096.0)));
    }
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    view_proj: [f32; 16],
    viewport: [f32; 2],
    cell: f32,
    count: u32,
    cols: u32,
    rows: u32,
    weighted: u32,
    _pad: u32,
}

/// Each cell's count and summed weight, row by row from the top left.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ScreenBins {
    pub cols: u32,
    pub rows: u32,
    pub counts: Vec<u32>,
    pub sums: Vec<f32>,
}

/// A binning pass in flight: its result once the GPU is done.
pub struct BinRequest {
    result: Arc<Mutex<Option<ScreenBins>>>,
}

impl BinRequest {
    /// The bins, when they have arrived (poll the device meanwhile).
    pub fn try_take(&self) -> Option<ScreenBins> {
        self.result.lock().ok()?.take()
    }

    /// Block until they arrive. Native only: a browser cannot wait.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait(self, device: &wgpu::Device) -> Option<ScreenBins> {
        loop {
            if let Some(b) = self.try_take() {
                return Some(b);
            }
            device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        }
    }
}

/// Points kept on the GPU, binned into screen cells for any view.
pub struct ScreenBinner {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    points: wgpu::Buffer,
    weights: wgpu::Buffer,
    count: u32,
    weighted: bool,
}

impl ScreenBinner {
    /// Whether the device runs compute shaders (WebGPU and native do;
    /// WebGL2 does not).
    pub fn supported(device: &wgpu::Device) -> bool {
        device.limits().max_compute_workgroup_size_x >= 256 && device.limits().max_storage_buffers_per_shader_stage >= 4
    }

    /// `positions` relative to the origin the views will be given in, and
    /// optional weights.
    pub fn new(device: &wgpu::Device, positions: &[[f32; 3]], weights: Option<&[f32]>) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers screen binner"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let storage = |read_only| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers screen binner layout"),
            entries: &[
                entry(0, storage(true)),
                entry(1, storage(true)),
                entry(2, storage(false)),
                entry(3, storage(false)),
                entry(
                    4,
                    wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                ),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers screen binner pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("threers screen binner"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("bin"),
            compilation_options: Default::default(),
            cache: None,
        });
        let mut data: Vec<[f32; 4]> = positions.iter().map(|p| [p[0], p[1], p[2], 0.0]).collect();
        if data.is_empty() {
            data.push([0.0; 4]);
        }
        let points = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("threers screen binner points"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let w: Vec<f32> = match weights {
            Some(w) if !w.is_empty() => w.to_vec(),
            _ => vec![0.0],
        };
        let weights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("threers screen binner weights"),
            contents: bytemuck::cast_slice(&w),
            usage: wgpu::BufferUsages::STORAGE,
        });
        Self {
            pipeline,
            layout,
            points,
            weights: weights_buf,
            count: positions.len() as u32,
            weighted: weights.is_some_and(|w| !w.is_empty()),
        }
    }

    /// Bin the points seen through `view_proj` (origin-relative, as the
    /// positions) on a `viewport` of pixels into `cell`-pixel squares.
    pub fn bin(&self, device: &wgpu::Device, queue: &wgpu::Queue, view_proj: [f32; 16], viewport: (u32, u32), cell: f32) -> BinRequest {
        let cell = cell.max(1.0);
        let cols = ((viewport.0 as f32 / cell).ceil() as u32).max(1);
        let rows = ((viewport.1 as f32 / cell).ceil() as u32).max(1);
        let n = (cols * rows) as u64;
        let params = Params {
            view_proj,
            viewport: [viewport.0 as f32, viewport.1 as f32],
            cell,
            count: self.count,
            cols,
            rows,
            weighted: u32::from(self.weighted),
            _pad: 0,
        };
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("threers screen binner params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let make = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: n * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let (counts, sums) = (make("threers screen bin counts"), make("threers screen bin sums"));
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("threers screen bin readback"),
            size: n * 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("threers screen binner bind group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.points.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.weights.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: counts.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: sums.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: uniform.as_entire_binding() },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("threers screen binning") });
        // Fresh buffers are zeroed.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("threers screen binning"), timestamp_writes: None });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(self.count.div_ceil(256).max(1), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&counts, 0, &staging, 0, n * 4);
        encoder.copy_buffer_to_buffer(&sums, 0, &staging, n * 4, n * 4);
        queue.submit(Some(encoder.finish()));
        let result = Arc::new(Mutex::new(None));
        let out = result.clone();
        let staging = Arc::new(staging);
        let mapped = staging.clone();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            if r.is_err() {
                return;
            }
            let bins = {
                let Ok(data) = mapped.slice(..).get_mapped_range() else { return };
                let words: &[u32] = bytemuck::cast_slice(&data);
                let (c, s) = words.split_at(n as usize);
                ScreenBins {
                    cols,
                    rows,
                    counts: c.to_vec(),
                    sums: s.iter().map(|&v| v as i32 as f32 / WEIGHT_SCALE).collect(),
                }
            };
            mapped.unmap();
            if let Ok(mut o) = out.lock() {
                *o = Some(bins);
            }
        });
        BinRequest { result }
    }
}
