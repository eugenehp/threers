//! The homogenisation solver on a wgpu compute device.
//!
//! [`homogenize`](super::homogenize) spends nearly all of its time in one
//! place: applying `K` to a vector, once per conjugate-gradient iteration,
//! hundreds of times per macroscopic case. On a uniform periodic grid that
//! operator is a stencil — every node gathers the eight elements touching it
//! and multiplies by the one element matrix they all share — which is what a
//! GPU is for.
//!
//! # Why the whole iteration lives here
//!
//! The obvious split, operator on the device and the rest on the host, does not
//! work: the vectors are megabytes and the iteration touches all of them, so
//! every iteration would pay two transfers and the transfers cost more than the
//! arithmetic saved. Everything therefore stays in device memory — the
//! vectors, the dot products, and the two scalars `α` and `β`, which are
//! computed by a one-thread kernel rather than being read back. The host reads
//! five floats every [`CHECK`] iterations to test convergence and otherwise
//! never looks.
//!
//! # Single precision
//!
//! WebGPU has no `f64`, so this solves in `f32` where the CPU path solves in
//! `f64`, and the two do not agree to the last bit. That is affordable here for
//! a specific reason: the effective tensor is read off a *strain energy*, and
//! energy is stationary at the solution, so an error `ε` in the displacement
//! field shows up as `ε²` in the answer. A residual a thousand times looser
//! than the CPU's still lands within a fraction of a percent of it — which is
//! what `gpu_matches_cpu` measures rather than assumes.
//!
//! It is opt-in for that reason: see [`Solver`](super::Solver).

use std::borrow::Cow;
use std::sync::Arc;

/// How many iterations run between convergence checks.
///
/// Each check is a five-float readback, and a readback is a round trip to the
/// device — about as expensive as twenty iterations of the solve itself. Twenty
/// five iterations of overshoot at the end costs less than checking for it.
const CHECK: usize = 25;

/// Threads per workgroup for the stencil, and for each stage of the reduction.
const GROUP: u32 = 64;
/// Workgroups the reduction runs, whatever the vector length. Each thread walks
/// the vector with a stride, so this is a shape, not a size.
const REDUCE_GROUPS: u32 = 64;
/// Threads per reduction workgroup. The tree in `reduce` halves from this, so
/// it has to be a power of two.
const REDUCE_GROUP: u32 = 256;

/// A compute device, and the pipelines to run a conjugate gradient on it.
pub(super) struct GpuSolver {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    layout: wgpu::BindGroupLayout,
    apply: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
    finish: wgpu::ComputePipeline,
    seed: wgpu::ComputePipeline,
    alpha: wgpu::ComputePipeline,
    beta: wgpu::ComputePipeline,
    init: wgpu::ComputePipeline,
    update_xr: wgpu::ComputePipeline,
    update_p: wgpu::ComputePipeline,
}

impl GpuSolver {
    /// Acquire a device, or `None` if there is not one.
    ///
    /// Headless and native: a homogenisation is not a frame, and there is no
    /// surface to be compatible with.
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn new() -> Option<Self> {
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
        // The adapter's own limits, not the conservative defaults: a 48³ cell
        // is a nine-megabyte working set and the default storage-buffer
        // ceiling is smaller than that on some adapters.
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("threers lattice homogenisation"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            ..Default::default()
        }))
        .ok()?;
        Some(Self::with_device(Arc::new(device), Arc::new(queue)))
    }

    /// No device without a surface on the web, and no `pollster` either.
    #[cfg(target_arch = "wasm32")]
    pub(super) fn new() -> Option<Self> {
        None
    }

    fn with_device(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers lattice cg"),
            // The launch shapes are in the Rust constants and the shader is
            // written against them by name, so the two cannot drift.
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(
                SHADER
                    .replace("GROUP_N", &GROUP.to_string())
                    .replace("REDUCE_GROUPS_N", &REDUCE_GROUPS.to_string())
                    .replace("REDUCE_WIDTH_N", &REDUCE_GROUP.to_string()),
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
        // 1 ke, 2 stiff, 3 diag are read; the rest are worked on.
        entries.extend((1..4).map(|b| entry(b, true)));
        entries.extend((4..11).map(|b| entry(b, false)));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers lattice cg layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers lattice cg pipeline layout"),
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
        Self {
            apply: pipeline("apply"),
            reduce: pipeline("reduce"),
            finish: pipeline("finish"),
            seed: pipeline("seed"),
            alpha: pipeline("alpha"),
            beta: pipeline("beta"),
            init: pipeline("init"),
            update_xr: pipeline("update_xr"),
            update_p: pipeline("update_p"),
            layout,
            device,
            queue,
        }
    }

    /// Upload a cell and its element matrix, ready to solve any number of
    /// right-hand sides on.
    ///
    /// The geometry does not change between the six macroscopic cases, so it is
    /// uploaded once and the cases differ only in the load.
    pub(super) fn problem(
        &self,
        n: usize,
        per_node: usize,
        ke: &[f64],
        stiff: &[f64],
        diag: &[f64],
    ) -> GpuProblem<'_> {
        use wgpu::util::DeviceExt;
        let dofs = 8 * per_node;
        let total = stiff.len() * per_node;
        let params = [n as u32, per_node as u32, dofs as u32, total as u32];
        let storage = wgpu::BufferUsages::STORAGE;
        let buffer = |label: &str, data: &[f32], usage: wgpu::BufferUsages| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(data),
                    usage,
                })
        };
        let single = |v: &[f64]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };
        let zeros = vec![0.0f32; total];

        GpuProblem {
            solver: self,
            uniform: self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("cg params"),
                    contents: bytemuck::cast_slice(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                }),
            ke: buffer("cg ke", &single(ke), storage),
            stiff: buffer("cg stiff", &single(stiff), storage),
            diag: buffer("cg diag", &single(diag), storage),
            x: buffer("cg x", &zeros, storage | wgpu::BufferUsages::COPY_SRC),
            r: buffer("cg r", &zeros, storage | wgpu::BufferUsages::COPY_DST),
            z: buffer("cg z", &zeros, storage),
            p: buffer("cg p", &zeros, storage),
            q: buffer("cg q", &zeros, storage),
            partials: buffer("cg partials", &vec![0.0f32; 3 * REDUCE_GROUPS as usize], storage),
            scalars: buffer(
                "cg scalars",
                &[0.0f32; 8],
                storage | wgpu::BufferUsages::COPY_SRC,
            ),
            readback: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cg readback"),
                size: 32,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            solution: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cg solution"),
                size: (total * 4) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            nodes: stiff.len() as u32,
            total,
        }
    }
}

/// One cell, uploaded, with room to solve a load case on it.
pub(super) struct GpuProblem<'a> {
    solver: &'a GpuSolver,
    uniform: wgpu::Buffer,
    ke: wgpu::Buffer,
    stiff: wgpu::Buffer,
    diag: wgpu::Buffer,
    x: wgpu::Buffer,
    r: wgpu::Buffer,
    z: wgpu::Buffer,
    p: wgpu::Buffer,
    q: wgpu::Buffer,
    partials: wgpu::Buffer,
    scalars: wgpu::Buffer,
    readback: wgpu::Buffer,
    solution: wgpu::Buffer,
    nodes: u32,
    total: usize,
}

impl GpuProblem<'_> {
    /// Solve `K χ = f`, stopping when the residual falls below `target` or the
    /// iterations run out.
    ///
    /// `None` if the device fails to hand anything back, which leaves the
    /// caller free to fall back to the CPU rather than to guess.
    pub(super) fn solve(&self, f: &[f64], target: f64, limit: usize) -> Option<Vec<f64>> {
        let device = &self.solver.device;
        let queue = &self.solver.queue;
        let group = self.solver.layout.clone();
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cg bind"),
            layout: &group,
            entries: &[
                self.uniform.as_entire_binding(),
                self.ke.as_entire_binding(),
                self.stiff.as_entire_binding(),
                self.diag.as_entire_binding(),
                self.x.as_entire_binding(),
                self.r.as_entire_binding(),
                self.z.as_entire_binding(),
                self.p.as_entire_binding(),
                self.q.as_entire_binding(),
                self.partials.as_entire_binding(),
                self.scalars.as_entire_binding(),
            ]
            .iter()
            .enumerate()
            .map(|(binding, resource)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: resource.clone(),
            })
            .collect::<Vec<_>>(),
        });

        let load: Vec<f32> = f.iter().map(|&v| v as f32).collect();
        queue.write_buffer(&self.r, 0, bytemuck::cast_slice(&load));

        let vector_groups = (self.total as u32).div_ceil(GROUP);
        let node_groups = self.nodes.div_ceil(GROUP);
        // Every kernel of a batch in one compute pass. A pass is the unit the
        // driver sets up and tears down, and one per dispatch costs about a
        // millisecond here — more than the arithmetic between two of them.
        // Ordering inside a pass is guaranteed, so the hazards take care of
        // themselves.
        let run = |encoder: &mut wgpu::CommandEncoder, steps: &[(&wgpu::ComputePipeline, u32)]| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("threers lattice cg"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &bind, &[]);
            for (pipeline, groups) in steps {
                pass.set_pipeline(pipeline);
                pass.dispatch_workgroups(*groups, 1, 1);
            }
        };
        let solver = self.solver;
        // x = 0, z = r / diag, p = z, and the first `r·z` into the scalars.
        let mut encoder = device.create_command_encoder(&Default::default());
        run(
            &mut encoder,
            &[
                (&solver.init, vector_groups),
                (&solver.reduce, REDUCE_GROUPS),
                (&solver.finish, 1),
                (&solver.seed, 1),
            ],
        );
        queue.submit([encoder.finish()]);

        let mut done = 0usize;
        while done < limit {
            let batch = CHECK.min(limit - done);
            let mut steps = Vec::with_capacity(batch * 9);
            for _ in 0..batch {
                steps.extend_from_slice(&[
                    (&solver.apply, node_groups),
                    (&solver.reduce, REDUCE_GROUPS),
                    (&solver.finish, 1),
                    (&solver.alpha, 1),
                    (&solver.update_xr, vector_groups),
                    (&solver.reduce, REDUCE_GROUPS),
                    (&solver.finish, 1),
                    (&solver.beta, 1),
                    (&solver.update_p, vector_groups),
                ]);
            }
            let mut encoder = device.create_command_encoder(&Default::default());
            run(&mut encoder, &steps);
            encoder.copy_buffer_to_buffer(&self.scalars, 0, &self.readback, 0, 32);
            queue.submit([encoder.finish()]);
            done += batch;

            let scalars = self.read(&self.readback, 8)?;
            // scalars[2] is `r·r`, kept as a square to save a root per
            // iteration on a number only read every twenty five.
            let residual = scalars[2].max(0.0).sqrt() as f64;
            if !residual.is_finite() {
                return None;
            }
            if residual <= target {
                break;
            }
        }

        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.x, 0, &self.solution, 0, (self.total * 4) as u64);
        queue.submit([encoder.finish()]);
        let solution = self.read(&self.solution, self.total)?;
        Some(solution.into_iter().map(|v| v as f64).collect())
    }

    /// Map a buffer and take `count` floats out of it.
    fn read(&self, buffer: &wgpu::Buffer, count: usize) -> Option<Vec<f32>> {
        let (tx, rx) = std::sync::mpsc::channel();
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.solver.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().ok()?.ok()?;
        let out = {
            let view = buffer.slice(..).get_mapped_range().ok()?;
            bytemuck::cast_slice::<u8, f32>(&view)[..count].to_vec()
        };
        buffer.unmap();
        Some(out)
    }
}

/// The whole iteration, in nine kernels.
///
/// Written against the same element numbering the CPU path uses — corner `c` at
/// `(c & 1, (c >> 1) & 1, (c >> 2) & 1)`, `per_node` degrees of freedom each —
/// so that the two can be compared element for element and not just in the
/// aggregate.
const SHADER: &str = r#"
struct Params {
    n: u32,
    per_node: u32,
    dofs: u32,
    total: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> ke: array<f32>;
@group(0) @binding(2) var<storage, read> stiff: array<f32>;
@group(0) @binding(3) var<storage, read> diag: array<f32>;
@group(0) @binding(4) var<storage, read_write> x: array<f32>;
@group(0) @binding(5) var<storage, read_write> r: array<f32>;
@group(0) @binding(6) var<storage, read_write> z: array<f32>;
@group(0) @binding(7) var<storage, read_write> p: array<f32>;
@group(0) @binding(8) var<storage, read_write> q: array<f32>;
@group(0) @binding(9) var<storage, read_write> partials: array<f32>;
@group(0) @binding(10) var<storage, read_write> scalars: array<f32>;

// scalars: 0 = r·z carried between iterations, 1 = p·q, 2 = r·r,
//          3 = alpha, 4 = beta, 5 = the r·z just measured.

fn node_of(ei: u32, ej: u32, ek: u32, corner: u32) -> u32 {
    let n = params.n;
    let i = (ei + (corner & 1u)) % n;
    let j = (ej + ((corner >> 1u) & 1u)) % n;
    let k = (ek + ((corner >> 2u) & 1u)) % n;
    return (k * n + j) * n + i;
}

// q = K p, gathered per node: the eight elements that touch it, each
// contributing the row of its element matrix that this node owns.
@compute @workgroup_size(GROUP_N)
fn apply(@builtin(global_invocation_id) id: vec3<u32>) {
    let nd = id.x;
    let n = params.n;
    let nodes = n * n * n;
    if (nd >= nodes) { return; }
    let per_node = params.per_node;
    let dofs = params.dofs;

    var acc: array<f32, 3>;
    for (var a = 0u; a < per_node; a = a + 1u) { acc[a] = 0.0; }

    // The one node pinned to take out the rigid mode: it cannot move, so
    // nothing is applied to it.
    if (nd != 0u) {
        let i = nd % n;
        let j = (nd / n) % n;
        let k = nd / (n * n);
        var local: array<f32, 24>;
        for (var c = 0u; c < 8u; c = c + 1u) {
            // The element whose corner `c` this node is.
            let ei = (i + n - (c & 1u)) % n;
            let ej = (j + n - ((c >> 1u) & 1u)) % n;
            let ek = (k + n - ((c >> 2u) & 1u)) % n;
            let e = (ek * n + ej) * n + ei;
            let s = stiff[e];
            for (var b = 0u; b < dofs; b = b + 1u) {
                let node = node_of(ei, ej, ek, b / per_node);
                local[b] = p[node * per_node + (b % per_node)];
            }
            for (var a = 0u; a < per_node; a = a + 1u) {
                let row = (c * per_node + a) * dofs;
                var sum = 0.0;
                for (var b = 0u; b < dofs; b = b + 1u) {
                    sum = sum + ke[row + b] * local[b];
                }
                acc[a] = acc[a] + s * sum;
            }
        }
    }
    for (var a = 0u; a < per_node; a = a + 1u) {
        q[nd * per_node + a] = acc[a];
    }
}

var<workgroup> scratch: array<vec3<f32>, REDUCE_WIDTH_N>;

// One pass over the vectors for all three dot products at once. Each thread
// strides the whole array, so the launch shape does not depend on its length.
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

@compute @workgroup_size(GROUP_N)
fn init(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.total) { return; }
    x[i] = 0.0;
    let zi = r[i] / diag[i];
    z[i] = zi;
    p[i] = zi;
}

@compute @workgroup_size(GROUP_N)
fn update_xr(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.total) { return; }
    let a = scalars[3];
    x[i] = x[i] + a * p[i];
    let ri = r[i] - a * q[i];
    r[i] = ri;
    z[i] = ri / diag[i];
}

@compute @workgroup_size(GROUP_N)
fn update_p(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.total) { return; }
    p[i] = z[i] + scalars[4] * p[i];
}
"#;
