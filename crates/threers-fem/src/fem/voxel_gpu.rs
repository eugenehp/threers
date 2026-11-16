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
//! WebGPU has no `f64`, and a residual small in single precision is not a
//! small error in a stiff, slender part: the error can be the residual times
//! the condition number. So the solve is refined: the device solves in `f32`
//! to [`TOLERANCE`], the host works out in `f64` what that leaves of the
//! load (one product), and the device solves for that, until what is left is
//! as small as the CPU's own solve leaves it ([`REFINED`]). The answer is the
//! CPU's to within its tolerance; `the_gpu_solves_as_the_cpu_does` measures
//! that rather than assuming it.
//!
//! What it cannot do is a body whose condition number times single
//! precision's 10⁻⁷ approaches one — a slender part in fine cubes, like a
//! spectacle temple at a millimetre — where the device's own products are
//! too rough for the refinement to gain on. It sees that after one round and
//! hands the solve back, and [`solve_gpu`](super::solve_gpu) finishes it on
//! the CPU.

use std::borrow::Cow;
use std::sync::OnceLock;

use super::{Element, VoxelBody, MOST_ITERATIONS};
use threers::wgpu;

/// Iterations between convergence checks: a check is a round trip to the
/// device, worth about twenty iterations.
const CHECK: usize = 25;
/// Where each single-precision solve stops, against its right-hand side,
/// and the most iterations it runs: short, because the residual a long
/// `f32` iteration keeps drifts away from the true one, and the refinement
/// has the accuracy in hand anyway.
const TOLERANCE: f64 = 1e-3;
const INNER: usize = 1000;
/// Where the refinement stops, against the load: as the CPU's solve does.
const REFINED: f64 = 1e-7;
/// Refinement rounds before giving the solve back to the CPU.
const ROUNDS: usize = 60;
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
pub(super) fn displacements(
    body: &VoxelBody,
    el: &Element,
    scale: f64,
    diag: &[f64],
    fixed: &[bool],
    f: &[f64],
    product: &dyn Fn(&[f64]) -> Vec<f64>,
) -> Option<(Vec<f64>, usize)> {
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
    let free = |v: &[f64]| -> Vec<f64> { v.iter().zip(fixed).map(|(v, held)| if *held { 0.0 } else { *v }).collect() };
    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let f = free(f);
    let size = norm(&f);
    if size == 0.0 {
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
    let r = init("r", &zeros, storage | wgpu::BufferUsages::COPY_DST);
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
    // One single-precision solve of `K dx = rhs` on the device, to a
    // residual `TOLERANCE` of the right-hand side's.
    let inner = |rhs: &[f64]| -> Option<(Vec<f64>, usize)> {
        let target = norm(rhs) * TOLERANCE;
        let rhs32: Vec<f32> = rhs.iter().map(|v| *v as f32).collect();
        gpu.queue.write_buffer(&r, 0, &bytes_f32(&rhs32));
        run(&[(init_k, vector_groups), (reduce, REDUCE_GROUPS), (finish, 1), (seed, 1)], None);
        let mut done = 0usize;
        loop {
            if done >= INNER {
                break;
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
        let dx = read(gpu, &solution, total)?;
        Some((dx.into_iter().map(f64::from).collect(), done))
    };

    // Iterative refinement: each round solves for what is left in single
    // precision on the device, and what is left is found again here in
    // double, until it is as small as the CPU's own solve leaves it.
    let mut x = vec![0.0f64; total];
    let mut rest = f.clone();
    let mut iterations = 0;
    let mut last = size;
    for _ in 0..ROUNDS {
        let (dx, n) = inner(&rest)?;
        if iterations + n > MOST_ITERATIONS {
            return None;
        }
        iterations += n;
        for (a, b) in x.iter_mut().zip(&dx) {
            *a += b;
        }
        let kx = product(&x);
        rest = free(&f.iter().zip(&kx).map(|(a, b)| a - b).collect::<Vec<_>>());
        let left = norm(&rest);
        if left <= size * REFINED {
            return Some((x, iterations));
        }
        // Refinement gains only while single precision can resolve the
        // problem: past a condition number of about 10⁷ (a long, thin part
        // in small cubes) the device's products are wrong in just the slow
        // modes the solve is after, and what is left does not shrink. Then
        // the CPU solves it, at the cost of one round here, not dozens.
        if left > 0.5 * last {
            return None;
        }
        last = left;
    }
    None
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
        view.as_chunks::<4>().0.iter().take(count).map(|b| f32::from_le_bytes(*b)).collect()
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

// ---------------------------------------------------------------------------
// Multigrid on the device: the V-cycle only.
//
// A cycle is a preconditioner, an approximation to `K⁻¹` by design, and
// single precision costs it nothing that matters. So for bodies big enough
// to have a hierarchy the conjugate gradient stays on the host in double
// precision — its products exact, the condition number no obstacle — and
// each iteration hands the device the residual and takes back the cycle's
// correction. The hierarchy is uploaded once a solve.
//
// Measured on an Apple M4 Pro (a busy machine, so the wall times are
// rough), against the CPU's multigrid on 14 cores: on bodies of about 5 000
// cubes the round trip each iteration costs more than the cycle saves, and
// the single-precision cycle needs a quarter to two thirds more iterations
// — 1.7 to 2.5 times the wall time, though a third less CPU time. At about
// 40 000 cubes it uses 40–45% of the CPU's time and is as fast or faster.
// Below some tens of thousands of cubes, use the CPU.

/// A cycle's pipelines, made once.
struct CyclePipelines {
    layout: wgpu::BindGroupLayout,
    zero: wgpu::ComputePipeline,
    apply: wgpu::ComputePipeline,
    jacobi: wgpu::ComputePipeline,
    restrict: wgpu::ComputePipeline,
    prolong: wgpu::ComputePipeline,
}

fn cycle_pipelines(gpu: &'static Gpu) -> &'static CyclePipelines {
    static PIPELINES: OnceLock<CyclePipelines> = OnceLock::new();
    PIPELINES.get_or_init(|| {
        let device = &gpu.device;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers-fem voxel multigrid"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(CYCLE_SHADER.replace("GROUP_N", &GROUP.to_string()))),
        });
        // 7, 8 … the read-only tables; 4, 5, 6, 13, 14 worked on.
        let writable = [4u32, 5, 6, 13, 14];
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        }];
        entries.extend((1..=16u32).map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: !writable.contains(&binding) },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("threers-fem multigrid layout"), entries: &entries });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers-fem multigrid pipelines"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |name: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        CyclePipelines { zero: pipeline("zero"), apply: pipeline("apply"), jacobi: pipeline("jacobi"), restrict: pipeline("to_coarse"), prolong: pipeline("prolong"), layout }
    })
}

/// The hierarchy on the device, ready to run cycles.
pub(super) struct DeviceCycle {
    gpu: &'static Gpu,
    pipelines: &'static CyclePipelines,
    binds: Vec<wgpu::BindGroup>,
    /// Each level's node count.
    nodes: Vec<u32>,
    b0: wgpu::Buffer,
    x0: wgpu::Buffer,
    readback: wgpu::Buffer,
}

impl DeviceCycle {
    /// `mg` uploaded, or `None` without a device.
    pub(super) fn new(mg: &super::Multigrid, el: &Element) -> Option<DeviceCycle> {
        use wgpu::util::DeviceExt;
        let gpu = device()?;
        let pipelines = cycle_pipelines(gpu);
        let device = &gpu.device;
        let storage = wgpu::BufferUsages::STORAGE;
        let init = |label: &str, contents: &[u8], usage: wgpu::BufferUsages| {
            // A binding cannot be empty.
            let contents = if contents.is_empty() { &[0u8; 4][..] } else { contents };
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage })
        };
        let ke = init("ke", &bytes_f32(&el.k.iter().flatten().map(|v| *v as f32).collect::<Vec<_>>()), storage);
        let levels = &mg.levels;
        // Each level's own buffers.
        struct Own {
            params: wgpu::Buffer,
            table: wgpu::Buffer,
            factor: wgpu::Buffer,
            inverse: wgpu::Buffer,
            x: wgpu::Buffer,
            b: wgpu::Buffer,
            q: wgpu::Buffer,
            prolong: [wgpu::Buffer; 3],
            restrict: [wgpu::Buffer; 3],
        }
        let rw = storage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let mut own: Vec<Own> = vec![];
        for (l, level) in levels.iter().enumerate() {
            let nodes = level.around.len();
            let mut table: Vec<u32> = level.around.iter().flatten().copied().collect();
            let cubes_at = table.len() as u32;
            table.extend(level.cubes.iter().flatten().copied());
            let coarse = levels.get(l + 1).map_or(0, |c| c.around.len());
            // Fine-from-coarse rows, and their transpose, coarse-from-fine.
            let csr = |rows: &[Vec<(u32, f64)>]| -> [Vec<u8>; 3] {
                let mut off = vec![0u32];
                let (mut idx, mut w) = (vec![], vec![]);
                for row in rows {
                    for &(i, v) in row {
                        idx.push(i);
                        w.push(v as f32);
                    }
                    off.push(idx.len() as u32);
                }
                [bytes_u32(&off), bytes_u32(&idx), bytes_f32(&w)]
            };
            let mut back: Vec<Vec<(u32, f64)>> = vec![vec![]; coarse];
            for (fine, row) in level.from_coarse.iter().enumerate() {
                for &(c, w) in row {
                    back[c as usize].push((fine as u32, w));
                }
            }
            let [po, pi, pw] = csr(&level.from_coarse);
            let [ro, ri, rwgt] = csr(&back);
            let params: Vec<u8> = [nodes as u32, cubes_at, coarse as u32].iter().flat_map(|v| v.to_le_bytes()).chain((level.damping as f32).to_le_bytes()).collect();
            let zeros = bytes_f32(&vec![0.0f32; nodes * 3]);
            own.push(Own {
                params: init("params", &params, wgpu::BufferUsages::UNIFORM),
                table: init("table", &bytes_u32(&table), storage),
                factor: init("factor", &bytes_f32(&level.factor.iter().map(|v| *v as f32).collect::<Vec<_>>()), storage),
                inverse: init("inverse", &bytes_f32(&level.inverse.iter().map(|v| *v as f32).collect::<Vec<_>>()), storage),
                x: init("x", &zeros, rw),
                b: init("b", &zeros, rw),
                q: init("q", &zeros, rw),
                prolong: [init("p off", &po, storage), init("p idx", &pi, storage), init("p w", &pw, storage)],
                restrict: [init("r off", &ro, storage), init("r idx", &ri, storage), init("r w", &rwgt, storage)],
            });
        }
        // A stand-in for the coarsest level's coarser neighbour.
        let dummy_rw = init("none", &[0u8; 4], rw);
        let dummy_r = init("none", &[0u8; 4], storage);
        let binds = (0..own.len())
            .map(|l| {
                let o = &own[l];
                let next = own.get(l + 1);
                let resources: Vec<&wgpu::Buffer> = vec![
                    &o.params,
                    &o.table,
                    &o.factor,
                    &o.inverse,
                    &o.x,
                    &o.b,
                    &o.q,
                    &o.prolong[0],
                    &o.prolong[1],
                    &o.prolong[2],
                    &o.restrict[0],
                    &o.restrict[1],
                    &o.restrict[2],
                    next.map_or(&dummy_rw, |n| &n.x),
                    next.map_or(&dummy_rw, |n| &n.b),
                    next.map_or(&dummy_r, |n| &n.inverse),
                    &ke,
                ];
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("threers-fem multigrid level"),
                    layout: &pipelines.layout,
                    entries: &resources.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect::<Vec<_>>(),
                })
            })
            .collect();
        let n0 = levels[0].around.len();
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cycle readback"),
            size: (n0 * 12) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let nodes = levels.iter().map(|l| l.around.len() as u32).collect();
        let mut own = own.into_iter();
        let first = own.next()?;
        Some(DeviceCycle { gpu, pipelines, binds, nodes, b0: first.b, x0: first.x, readback })
    }

    /// One V-cycle for the residual `r`, as [`super::Multigrid::cycle`]
    /// does it; `None` should the device fail.
    pub(super) fn cycle(&self, r: &[f64]) -> Option<Vec<f64>> {
        let gpu = self.gpu;
        let r32: Vec<f32> = r.iter().map(|v| *v as f32).collect();
        gpu.queue.write_buffer(&self.b0, 0, &bytes_f32(&r32));
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("threers-fem v-cycle"), timestamp_writes: None });
            self.encode(&mut pass, 0);
        }
        encoder.copy_buffer_to_buffer(&self.x0, 0, &self.readback, 0, (r.len() * 4) as u64);
        gpu.queue.submit([encoder.finish()]);
        let z = read(gpu, &self.readback, r.len())?;
        if z.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some(z.into_iter().map(f64::from).collect())
    }

    fn encode(&self, pass: &mut wgpu::ComputePass, level: usize) {
        let p = self.pipelines;
        let nodes = self.nodes[level];
        let (by_node, by_dof) = (nodes.div_ceil(GROUP), (nodes * 3).div_ceil(GROUP));
        let mut run = |pass: &mut wgpu::ComputePass, pipeline: &wgpu::ComputePipeline, groups: u32| {
            pass.set_bind_group(0, &self.binds[level], &[]);
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups(groups.max(1), 1, 1);
        };
        let smooth = |pass: &mut wgpu::ComputePass, run: &mut dyn FnMut(&mut wgpu::ComputePass, &wgpu::ComputePipeline, u32), sweeps: usize| {
            for _ in 0..sweeps {
                run(pass, &p.apply, by_node);
                run(pass, &p.jacobi, by_dof);
            }
        };
        run(pass, &p.zero, by_dof);
        if level + 1 == self.nodes.len() {
            smooth(pass, &mut run, super::COARSE_SWEEPS);
            return;
        }
        smooth(pass, &mut run, super::SMOOTH);
        run(pass, &p.apply, by_node);
        run(pass, &p.restrict, self.nodes[level + 1].div_ceil(GROUP));
        self.encode(pass, level + 1);
        run(pass, &p.prolong, by_node);
        smooth(pass, &mut run, super::SMOOTH);
    }
}

/// A V-cycle's five kernels, per level: the level's own buffers, and its
/// coarser neighbour's correction (`xc`), right-hand side (`bc`) and
/// inverse diagonal.
const CYCLE_SHADER: &str = r#"
struct Params {
    nodes: u32,
    cubes_at: u32,
    coarse: u32,
    damping: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> table: array<u32>;
@group(0) @binding(2) var<storage, read> factor: array<f32>;
@group(0) @binding(3) var<storage, read> inverse: array<f32>;
@group(0) @binding(4) var<storage, read_write> x: array<f32>;
@group(0) @binding(5) var<storage, read_write> b: array<f32>;
@group(0) @binding(6) var<storage, read_write> q: array<f32>;
@group(0) @binding(7) var<storage, read> p_off: array<u32>;
@group(0) @binding(8) var<storage, read> p_idx: array<u32>;
@group(0) @binding(9) var<storage, read> p_w: array<f32>;
@group(0) @binding(10) var<storage, read> r_off: array<u32>;
@group(0) @binding(11) var<storage, read> r_idx: array<u32>;
@group(0) @binding(12) var<storage, read> r_w: array<f32>;
@group(0) @binding(13) var<storage, read_write> xc: array<f32>;
@group(0) @binding(14) var<storage, read_write> bc: array<f32>;
@group(0) @binding(15) var<storage, read> inverse_c: array<f32>;
@group(0) @binding(16) var<storage, read> ke: array<f32>;

@compute @workgroup_size(GROUP_N)
fn zero(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.nodes * 3u) { return; }
    x[id.x] = 0.0;
}

// q = K x, a node at a time: its three rows of each cube around it.
@compute @workgroup_size(GROUP_N)
fn apply(@builtin(global_invocation_id) id: vec3<u32>) {
    let nd = id.x;
    if (nd >= params.nodes) { return; }
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    for (var c = 0u; c < 8u; c = c + 1u) {
        let e = table[nd * 8u + c];
        if (e == 0xffffffffu) { continue; }
        let corners = params.cubes_at + e * 8u;
        var cube = vec3<f32>(0.0, 0.0, 0.0);
        for (var k = 0u; k < 8u; k = k + 1u) {
            let m = table[corners + k];
            let u = vec3<f32>(x[m * 3u], x[m * 3u + 1u], x[m * 3u + 2u]);
            for (var a = 0u; a < 3u; a = a + 1u) {
                let row = (c * 3u + a) * 24u + k * 3u;
                cube[a] = cube[a] + dot(vec3<f32>(ke[row], ke[row + 1u], ke[row + 2u]), u);
            }
        }
        acc = acc + cube * factor[e];
    }
    for (var a = 0u; a < 3u; a = a + 1u) {
        let i = nd * 3u + a;
        q[i] = select(0.0, acc[a], inverse[i] != 0.0);
    }
}

// x += ω D⁻¹ (b − q), q being K x.
@compute @workgroup_size(GROUP_N)
fn jacobi(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.nodes * 3u) { return; }
    x[i] = x[i] + params.damping * inverse[i] * (b[i] - q[i]);
}

// The coarser level's right-hand side: the residual b − q gathered by the
// interpolation's transpose, held directions zero.
@compute @workgroup_size(GROUP_N)
fn to_coarse(@builtin(global_invocation_id) id: vec3<u32>) {
    let c = id.x;
    if (c >= params.coarse) { return; }
    var sum = vec3<f32>(0.0, 0.0, 0.0);
    for (var j = r_off[c]; j < r_off[c + 1u]; j = j + 1u) {
        let f = r_idx[j];
        let w = r_w[j];
        sum = sum + w * vec3<f32>(b[f * 3u] - q[f * 3u], b[f * 3u + 1u] - q[f * 3u + 1u], b[f * 3u + 2u] - q[f * 3u + 2u]);
    }
    for (var a = 0u; a < 3u; a = a + 1u) {
        bc[c * 3u + a] = select(0.0, sum[a], inverse_c[c * 3u + a] != 0.0);
    }
}

// x += the coarser correction, interpolated, held directions untouched.
@compute @workgroup_size(GROUP_N)
fn prolong(@builtin(global_invocation_id) id: vec3<u32>) {
    let nd = id.x;
    if (nd >= params.nodes) { return; }
    var sum = vec3<f32>(0.0, 0.0, 0.0);
    for (var j = p_off[nd]; j < p_off[nd + 1u]; j = j + 1u) {
        let c = p_idx[j];
        sum = sum + p_w[j] * vec3<f32>(xc[c * 3u], xc[c * 3u + 1u], xc[c * 3u + 2u]);
    }
    for (var a = 0u; a < 3u; a = a + 1u) {
        let i = nd * 3u + a;
        if (inverse[i] != 0.0) {
            x[i] = x[i] + sum[a];
        }
    }
}
"#;
