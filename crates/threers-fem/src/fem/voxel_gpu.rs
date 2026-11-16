//! The voxel solver's conjugate gradient on a wgpu compute device.
//!
//! The iteration is almost all one operation, `K p`, and on cubes that all
//! share one element matrix that operation is a gather: each node takes the
//! rows it owns of the up-to-eight cubes touching it, times their corners'
//! values. No two threads write the same place, so there are no atomics
//! (which WGSL does not have for floats). A table gives each node its
//! cubes, the body being any set of filled cells rather than a full grid.
//!
//! As in threers' lattice homogenisation (`gpu_solve.rs`, which this
//! follows), the whole iteration lives on the device — the vectors, the dot
//! products, and `α` and `β`, worked out by one-thread kernels — and the
//! host reads the residual every [`CHECK`] iterations and otherwise never
//! looks: a transfer a step would cost more than the arithmetic it saves.
//!
//! WebGPU has no `f64`, so the iteration runs in `f32` and stops at a
//! relative residual of [`TOLERANCE`], looser than the CPU's 10⁻⁷, which is
//! about as far as single precision goes. The displacements and stresses come
//! out within a fraction of a percent of the CPU's: `the_gpu_solves_as_the_cpu
//! _does` measures that rather than assuming it.

use std::borrow::Cow;
use std::sync::OnceLock;

use super::{Element, VoxelBody, MOST_ITERATIONS};
use threers::wgpu;

/// Iterations between convergence checks: a check is a round trip to the
/// device, worth about twenty iterations.
const CHECK: usize = 25;
/// Where the iteration stops, against the load's size: what single
/// precision reaches without stalling.
const TOLERANCE: f64 = 1e-5;
const GROUP: u32 = 64;
const REDUCE_GROUPS: u32 = 64;
const REDUCE_WIDTH: u32 = 256;
/// A node's corner with no cube there.
const NONE: u32 = u32::MAX;

/// The device and the pipelines, made once.
pub(super) struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    pipelines: [wgpu::ComputePipeline; 9],
}

const KERNELS: [&str; 9] = ["apply", "reduce", "finish", "seed", "alpha", "beta", "init", "update_xr", "update_p"];

/// The device, or `None` without an adapter.
pub(super) fn device() -> Option<&'static Gpu> {
    static GPU: OnceLock<Option<Gpu>> = OnceLock::new();
    GPU.get_or_init(Gpu::new).as_ref()
}

impl Gpu {
    fn new() -> Option<Gpu> {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(descriptor);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .ok()?;
        // The adapter's own limits: a temple is a few megabytes of vectors,
        // and ten storage buffers is more than the portable default.
        let limits = adapter.limits();
        if limits.max_storage_buffers_per_shader_stage < 10 {
            return None;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("threers-fem voxel"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            ..Default::default()
        }))
        .ok()?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers-fem voxel cg"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(
                SHADER
                    .replace("GROUP_N", &GROUP.to_string())
                    .replace("REDUCE_GROUPS_N", &REDUCE_GROUPS.to_string())
                    .replace("REDUCE_WIDTH_N", &REDUCE_WIDTH.to_string()),
            )),
        });
        let entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        // 1 ke, 2 cubes, 3 inverse diagonal are read; the rest worked on.
        entries.extend((1..4).map(|b| entry(b, true)));
        entries.extend((4..11).map(|b| entry(b, false)));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers-fem voxel layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers-fem voxel pipelines"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipelines = KERNELS.map(|name| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        Some(Gpu { device, queue, layout, pipelines })
    }
}

fn bytes_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_u32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// `x` with `K x = f` on the device, the `fixed` directions held at zero,
/// and how many iterations it took; `None` without a device, or should it
/// fail or not converge, for the caller to solve on the CPU instead.
pub(super) fn displacements(body: &VoxelBody, el: &Element, scale: f64, diag: &[f64], fixed: &[bool], f: &[f64]) -> Option<(Vec<f64>, usize)> {
    use wgpu::util::DeviceExt;
    let gpu = device()?;
    let nodes = body.nodes.len();
    let total = nodes * 3;
    // Each node's cube at each corner: the cube whose corner `c` it is.
    let mut owner = vec![NONE; nodes * 8];
    for (e, cube) in body.elements.iter().enumerate() {
        for (c, &node) in cube.iter().enumerate() {
            owner[node as usize * 8 + c] = e as u32;
        }
    }
    // One table for both: each node's eight cubes, then each cube's eight
    // nodes, from `cubes_at`.
    let cubes_at = owner.len() as u32;
    let mut table = owner;
    table.extend(body.elements.iter().flatten().copied());
    let ke: Vec<f32> = el.k.iter().flatten().map(|v| (v * scale) as f32).collect();
    // Zero where a direction is held: the preconditioner and the mask in one.
    let inverse: Vec<f32> = diag.iter().zip(fixed).map(|(d, held)| if *held || *d == 0.0 { 0.0 } else { (1.0 / d) as f32 }).collect();
    let load: Vec<f32> = f.iter().zip(fixed).map(|(v, held)| if *held { 0.0 } else { *v as f32 }).collect();
    let target = load.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt() * TOLERANCE;
    if target == 0.0 {
        return Some((vec![0.0; total], 0));
    }

    let device = &gpu.device;
    let storage = wgpu::BufferUsages::STORAGE;
    let init = |label: &str, contents: &[u8], usage: wgpu::BufferUsages| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage })
    };
    let zeros = bytes_f32(&vec![0.0f32; total]);
    let params = init("params", &bytes_u32(&[nodes as u32, total as u32, cubes_at, 0]), wgpu::BufferUsages::UNIFORM);
    let ke_b = init("ke", &bytes_f32(&ke), storage);
    let table_b = init("cubes", &bytes_u32(&table), storage);
    let inverse_b = init("inverse diagonal", &bytes_f32(&inverse), storage);
    let x = init("x", &zeros, storage | wgpu::BufferUsages::COPY_SRC);
    let r = init("r", &bytes_f32(&load), storage);
    let z = init("z", &zeros, storage);
    let p = init("p", &zeros, storage);
    let q = init("q", &zeros, storage);
    let partials = init("partials", &bytes_f32(&vec![0.0f32; 3 * REDUCE_GROUPS as usize]), storage);
    let scalars = init("scalars", &bytes_f32(&[0.0f32; 8]), storage | wgpu::BufferUsages::COPY_SRC);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: 32,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let solution = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("solution"),
        size: (total * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let buffers = [&params, &ke_b, &table_b, &inverse_b, &x, &r, &z, &p, &q, &partials, &scalars];
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("threers-fem voxel bind"),
        layout: &gpu.layout,
        entries: &buffers
            .iter()
            .enumerate()
            .map(|(binding, b)| wgpu::BindGroupEntry { binding: binding as u32, resource: b.as_entire_binding() })
            .collect::<Vec<_>>(),
    });
    let [apply, reduce, finish, seed, alpha, beta, init_k, update_xr, update_p] = &gpu.pipelines;
    let vector_groups = (total as u32).div_ceil(GROUP);
    let node_groups = (nodes as u32).div_ceil(GROUP);
    // Every kernel of a batch in one compute pass: a pass costs more to set
    // up than the arithmetic between two dispatches.
    let run = |steps: &[(&wgpu::ComputePipeline, u32)], copy: Option<(&wgpu::Buffer, &wgpu::Buffer, u64)>| {
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("threers-fem voxel cg"), timestamp_writes: None });
            pass.set_bind_group(0, &bind, &[]);
            for (pipeline, groups) in steps {
                pass.set_pipeline(pipeline);
                pass.dispatch_workgroups(*groups, 1, 1);
            }
        }
        if let Some((from, to, size)) = copy {
            encoder.copy_buffer_to_buffer(from, 0, to, 0, size);
        }
        gpu.queue.submit([encoder.finish()]);
    };
    run(&[(init_k, vector_groups), (reduce, REDUCE_GROUPS), (finish, 1), (seed, 1)], None);

    let mut done = 0usize;
    loop {
        if done >= MOST_ITERATIONS {
            return None;
        }
        let mut steps = Vec::with_capacity(CHECK * 9);
        for _ in 0..CHECK {
            steps.extend_from_slice(&[
                (apply, node_groups),
                (reduce, REDUCE_GROUPS),
                (finish, 1),
                (alpha, 1),
                (update_xr, vector_groups),
                (reduce, REDUCE_GROUPS),
                (finish, 1),
                (beta, 1),
                (update_p, vector_groups),
            ]);
        }
        run(&steps, Some((&scalars, &readback, 32)));
        done += CHECK;
        let s = read(gpu, &readback, 8)?;
        let residual = (s[2].max(0.0) as f64).sqrt();
        if !residual.is_finite() {
            return None;
        }
        if residual <= target {
            break;
        }
    }
    run(&[], Some((&x, &solution, (total * 4) as u64)));
    let x = read(gpu, &solution, total)?;
    Some((x.into_iter().map(f64::from).collect(), done))
}

/// Map a buffer and take `count` floats out of it.
fn read(gpu: &Gpu, buffer: &wgpu::Buffer, count: usize) -> Option<Vec<f32>> {
    let (tx, rx) = std::sync::mpsc::channel();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().ok()?.ok()?;
    let out = {
        let view = buffer.slice(..).get_mapped_range().ok()?;
        view.chunks_exact(4).take(count).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    };
    buffer.unmap();
    Some(out)
}

/// The iteration in nine kernels, on the CPU path's numbering: corner `c`
/// of a cube at `(c & 1, (c >> 1) & 1, (c >> 2) & 1)`, three displacements
/// a node.
const SHADER: &str = r#"
struct Params {
    nodes: u32,
    total: u32,
    cubes_at: u32,
    unused: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> ke: array<f32>;
@group(0) @binding(2) var<storage, read> cubes: array<u32>;
@group(0) @binding(3) var<storage, read> inverse: array<f32>;
@group(0) @binding(4) var<storage, read_write> x: array<f32>;
@group(0) @binding(5) var<storage, read_write> r: array<f32>;
@group(0) @binding(6) var<storage, read_write> z: array<f32>;
@group(0) @binding(7) var<storage, read_write> p: array<f32>;
@group(0) @binding(8) var<storage, read_write> q: array<f32>;
@group(0) @binding(9) var<storage, read_write> partials: array<f32>;
@group(0) @binding(10) var<storage, read_write> scalars: array<f32>;

// scalars: 0 = r·z carried between iterations, 1 = p·q, 2 = r·r,
//          3 = alpha, 4 = beta, 5 = the r·z just measured.

// q = K p at each node: of each cube touching it, the three rows of the
// element matrix for the corner the node is, times the cube's corners.
@compute @workgroup_size(GROUP_N)
fn apply(@builtin(global_invocation_id) id: vec3<u32>) {
    let nd = id.x;
    if (nd >= params.nodes) { return; }
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    for (var c = 0u; c < 8u; c = c + 1u) {
        let e = cubes[nd * 8u + c];
        if (e == 0xffffffffu) { continue; }
        let corners = params.cubes_at + e * 8u;
        for (var b = 0u; b < 8u; b = b + 1u) {
            let m = cubes[corners + b];
            let u = vec3<f32>(p[m * 3u], p[m * 3u + 1u], p[m * 3u + 2u]);
            for (var a = 0u; a < 3u; a = a + 1u) {
                let row = (c * 3u + a) * 24u + b * 3u;
                acc[a] = acc[a] + dot(vec3<f32>(ke[row], ke[row + 1u], ke[row + 2u]), u);
            }
        }
    }
    // Held directions take nothing.
    for (var a = 0u; a < 3u; a = a + 1u) {
        let i = nd * 3u + a;
        q[i] = select(0.0, acc[a], inverse[i] != 0.0);
    }
}

var<workgroup> scratch: array<vec3<f32>, REDUCE_WIDTH_N>;

// p·q, r·z and r·r in one pass, each thread striding the whole vector.
@compute @workgroup_size(REDUCE_WIDTH_N)
fn reduce(@builtin(global_invocation_id) id: vec3<u32>,
          @builtin(local_invocation_id) local_id: vec3<u32>,
          @builtin(workgroup_id) group: vec3<u32>) {
    var sum = vec3<f32>(0.0, 0.0, 0.0);
    let stride = REDUCE_WIDTH_Nu * REDUCE_GROUPS_Nu;
    for (var i = id.x; i < params.total; i = i + stride) {
        let ri = r[i];
        sum = sum + vec3<f32>(p[i] * q[i], ri * z[i], ri * ri);
    }
    scratch[local_id.x] = sum;
    workgroupBarrier();
    for (var step = REDUCE_WIDTH_Nu >> 1u; step > 0u; step = step >> 1u) {
        if (local_id.x < step) {
            scratch[local_id.x] = scratch[local_id.x] + scratch[local_id.x + step];
        }
        workgroupBarrier();
    }
    if (local_id.x == 0u) {
        partials[group.x * 3u] = scratch[0].x;
        partials[group.x * 3u + 1u] = scratch[0].y;
        partials[group.x * 3u + 2u] = scratch[0].z;
    }
}

@compute @workgroup_size(1)
fn finish() {
    var pq = 0.0;
    var rz = 0.0;
    var rr = 0.0;
    for (var g = 0u; g < REDUCE_GROUPS_Nu; g = g + 1u) {
        pq = pq + partials[g * 3u];
        rz = rz + partials[g * 3u + 1u];
        rr = rr + partials[g * 3u + 2u];
    }
    scalars[1] = pq;
    scalars[5] = rz;
    scalars[2] = rr;
}

@compute @workgroup_size(1)
fn seed() {
    scalars[0] = scalars[5];
}

@compute @workgroup_size(1)
fn alpha() {
    let pq = scalars[1];
    scalars[3] = select(0.0, scalars[0] / pq, abs(pq) > 1e-30);
}

@compute @workgroup_size(1)
fn beta() {
    let rz = scalars[0];
    scalars[4] = select(0.0, scalars[5] / rz, abs(rz) > 1e-30);
    scalars[0] = scalars[5];
}

// x = 0, z = r / diag (zero where held), p = z.
@compute @workgroup_size(GROUP_N)
fn init(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.total) { return; }
    x[i] = 0.0;
    let zi = r[i] * inverse[i];
    z[i] = zi;
    p[i] = zi;
}

@compute @workgroup_size(GROUP_N)
fn update_xr(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.total) { return; }
    let a = scalars[3];
    x[i] = x[i] + a * p[i];
    r[i] = r[i] - a * q[i];
    z[i] = r[i] * inverse[i];
}

@compute @workgroup_size(GROUP_N)
fn update_p(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.total) { return; }
    p[i] = z[i] + scalars[4] * p[i];
}
"#;
