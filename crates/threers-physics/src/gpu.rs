//! GPU-accelerated broad phase, via a wgpu compute shader.
//!
//! # Why the broad phase, and only the broad phase
//!
//! Finding which bounding boxes overlap is the one stage of a rigid-body step
//! that is embarrassingly parallel: every box pair is independent, there is no
//! iteration to serialise, and the answer is a small list of indices. The
//! narrow phase and the solver are not — the solver is *sequential* impulses by
//! construction, and shipping bodies to the GPU and back each iteration would
//! cost far more than it saves.
//!
//! # This is not free
//!
//! A dispatch plus a buffer readback costs roughly 0.2–1 ms of latency
//! regardless of how little work it does. The CPU sweep-and-prune in
//! [`crate::broadphase`] handles a few thousand bodies in well under that. Only
//! reach for this when the body count is high enough that an O(n²) or
//! badly-clustered sweep dominates — measure both before switching.
//!
//! What it *does* unlock is parallel physics in the browser, where rayon cannot
//! run: `wasm32` has no threads without cross-origin isolation, but it does have
//! WebGPU.
//!
//! # Sharing the renderer's device
//!
//! Creating a second wgpu device is wasteful and, in a browser, may fail
//! outright. Pass the one the renderer already owns:
//!
//! ```no_run
//! # #[cfg(feature = "gpu")] {
//! use threers_physics::gpu::GpuBroadPhase;
//! # fn demo(renderer: &threers::Renderer) {
//! let broadphase = GpuBroadPhase::from_device(renderer.device_arc(), renderer.queue_arc());
//! # }
//! # }
//! ```

use crate::world::World;
use std::sync::Arc;
use threers::math::Vector3;

/// One body's bounds, laid out for the shader.
///
/// `min.w` doubles as the "this body can move" flag — a pair where neither body
/// can move is never worth reporting, and packing the flag into the existing
/// padding keeps the struct at 32 bytes with no extra buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuAabb {
    pub min: [f32; 4],
    pub max: [f32; 4],
}

impl GpuAabb {
    pub fn new(min: Vector3, max: Vector3, active: bool) -> Self {
        Self {
            min: [min.x, min.y, min.z, if active { 1.0 } else { 0.0 }],
            max: [max.x, max.y, max.z, 0.0],
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    count: u32,
    capacity: u32,
    _pad: [u32; 2],
}

const WORKGROUP_SIZE: u32 = 64;

const SHADER: &str = r#"
struct Aabb {
    lo: vec4<f32>,
    hi: vec4<f32>,
};

struct Params {
    count: u32,
    capacity: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> boxes: array<Aabb>;
@group(0) @binding(1) var<storage, read_write> counter: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> pairs: array<vec2<u32>>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn find_pairs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) {
        return;
    }
    let a = boxes[i];

    // Only the upper triangle, so each pair is produced exactly once.
    for (var j: u32 = i + 1u; j < params.count; j = j + 1u) {
        let b = boxes[j];

        // Two bodies that both cannot move can never begin to touch.
        if (a.lo.w == 0.0 && b.lo.w == 0.0) {
            continue;
        }
        if (a.lo.x > b.hi.x || a.hi.x < b.lo.x) { continue; }
        if (a.lo.y > b.hi.y || a.hi.y < b.lo.y) { continue; }
        if (a.lo.z > b.hi.z || a.hi.z < b.lo.z) { continue; }

        // Reserve a slot. Overflow is counted but not written, so the host can
        // see that the buffer was too small and grow it.
        let slot = atomicAdd(&counter[0], 1u);
        if (slot < params.capacity) {
            pairs[slot] = vec2<u32>(i, j);
        }
    }
}
"#;

/// All-pairs AABB overlap on the GPU.
pub struct GpuBroadPhase {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,

    box_buffer: wgpu::Buffer,
    counter_buffer: wgpu::Buffer,
    pair_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    readback: wgpu::Buffer,

    box_capacity: usize,
    pair_capacity: usize,
    /// Set when the last run produced more pairs than the buffer held.
    overflowed: bool,
}

impl GpuBroadPhase {
    /// Build on an existing device — normally the renderer's.
    pub fn from_device(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers-physics broadphase"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let storage = |read_only: bool, binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("threers-physics broadphase layout"),
            entries: &[
                storage(true, 0),
                storage(false, 1),
                storage(false, 2),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
            label: Some("threers-physics broadphase pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("threers-physics broadphase pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("find_pairs"),
            compilation_options: Default::default(),
            cache: None,
        });

        let box_capacity = 1024;
        let pair_capacity = 4096;
        let (box_buffer, counter_buffer, pair_buffer, params_buffer, readback) =
            Self::allocate(&device, box_capacity, pair_capacity);

        Self {
            device,
            queue,
            pipeline,
            layout,
            box_buffer,
            counter_buffer,
            pair_buffer,
            params_buffer,
            readback,
            box_capacity,
            pair_capacity,
            overflowed: false,
        }
    }

    /// Create a standalone device. Prefer [`Self::from_device`] when a renderer
    /// already exists — a second device wastes memory and may be refused in a
    /// browser.
    ///
    /// Returns `None` when no compute-capable adapter is available.
    pub async fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            // `request_adapter` returns a `Result` in wgpu 30; this function
            // reports "no GPU here" as `None`, which is the same answer.
            .await
            .ok()?;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("threers-physics"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                ..Default::default()
                },
            )
            .await
            .ok()?;
        Some(Self::from_device(Arc::new(device), Arc::new(queue)))
    }

    fn allocate(
        device: &wgpu::Device,
        boxes: usize,
        pairs: usize,
    ) -> (
        wgpu::Buffer,
        wgpu::Buffer,
        wgpu::Buffer,
        wgpu::Buffer,
        wgpu::Buffer,
    ) {
        let box_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aabbs"),
            size: (boxes.max(1) * std::mem::size_of::<GpuAabb>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let counter_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pair counter"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let pair_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pairs"),
            size: (pairs.max(1) * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // One readback buffer holds the counter (4 bytes, padded to 8 for
        // alignment) followed by the pair list.
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (8 + pairs.max(1) * 8) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (box_buffer, counter_buffer, pair_buffer, params_buffer, readback)
    }

    fn grow(&mut self, boxes: usize, pairs: usize) {
        if boxes <= self.box_capacity && pairs <= self.pair_capacity {
            return;
        }
        self.box_capacity = self.box_capacity.max(boxes.next_power_of_two());
        self.pair_capacity = self.pair_capacity.max(pairs.next_power_of_two());
        let (b, c, p, pa, r) = Self::allocate(&self.device, self.box_capacity, self.pair_capacity);
        self.box_buffer = b;
        self.counter_buffer = c;
        self.pair_buffer = p;
        self.params_buffer = pa;
        self.readback = r;
    }

    /// Whether the last run found more pairs than the buffer could hold. The
    /// buffer is grown automatically for the next call, so re-running produces
    /// the complete list.
    pub fn overflowed(&self) -> bool {
        self.overflowed
    }

    /// Overlapping pairs as `(i, j)` indices into `boxes`, with `i < j`.
    ///
    /// Results are sorted, so the output is identical to the CPU broad phase's
    /// and the simulation stays deterministic.
    pub async fn find_pairs(&mut self, boxes: &[GpuAabb]) -> Vec<(u32, u32)> {
        if boxes.len() < 2 {
            return Vec::new();
        }
        // Worst case is every pair, but reserving n²/2 for a big scene would be
        // absurd; start generous and grow when the counter says we overflowed.
        self.grow(boxes.len(), self.pair_capacity.max(boxes.len() * 8));

        self.queue
            .write_buffer(&self.box_buffer, 0, bytemuck::cast_slice(boxes));
        self.queue.write_buffer(&self.counter_buffer, 0, &0u32.to_ne_bytes());
        self.queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&Params {
                count: boxes.len() as u32,
                capacity: self.pair_capacity as u32,
                _pad: [0; 2],
            }),
        );

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("broadphase bind group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.box_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.counter_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.pair_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("broadphase"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let groups = (boxes.len() as u32).div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(groups.max(1), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.counter_buffer, 0, &self.readback, 0, 4);
        encoder.copy_buffer_to_buffer(
            &self.pair_buffer,
            0,
            &self.readback,
            8,
            (self.pair_capacity * 8) as u64,
        );
        self.queue.submit(Some(encoder.finish()));

        let (sender, receiver) = futures_channel::oneshot::channel();
        self.readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        // Native needs an explicit poll to drive the mapping to completion; in
        // the browser the runtime does it.
        #[cfg(not(target_arch = "wasm32"))]
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        if receiver.await.is_err() {
            return Vec::new();
        }

        let mut pairs = {
            let view = self.readback.slice(..).get_mapped_range().expect("buffer range is mapped");
            let found = u32::from_ne_bytes([view[0], view[1], view[2], view[3]]) as usize;
            self.overflowed = found > self.pair_capacity;
            let written = found.min(self.pair_capacity);
            let raw: &[u32] = bytemuck::cast_slice(&view[8..8 + written * 8]);
            raw.chunks_exact(2).map(|c| (c[0], c[1])).collect::<Vec<_>>()
        };
        self.readback.unmap();

        if self.overflowed {
            // Grow so the caller's next attempt succeeds, and say so.
            self.grow(boxes.len(), self.pair_capacity * 2);
        }

        // The shader appends in whatever order the workgroups finish. Sorting
        // makes the result independent of GPU scheduling — without it the
        // simulation would differ run to run on the same input.
        pairs.sort_unstable();
        pairs
    }

    /// Blocking wrapper for native code. Not available on `wasm32`, where
    /// blocking on the GPU would deadlock the event loop.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn find_pairs_blocking(&mut self, boxes: &[GpuAabb]) -> Vec<(u32, u32)> {
        pollster::block_on(self.find_pairs(boxes))
    }
}

impl World {
    /// The AABBs the broad phase would use this step, in body-slot order.
    ///
    /// Feed these to [`GpuBroadPhase::find_pairs`], then hand the result back
    /// with [`World::set_broadphase_pairs`].
    pub fn broadphase_aabbs(&self) -> Vec<GpuAabb> {
        (0..self.bodies().slot_count())
            .map(|i| match self.bodies().by_index(i) {
                Some(body) if body.enabled && !body.colliders().is_empty() => {
                    let mut aabb = body.compute_aabb();
                    if aabb.is_empty() {
                        // Keep slot order intact: an empty slot still needs an
                        // entry, or every index after it would be wrong.
                        return GpuAabb::new(Vector3::ZERO, Vector3::ZERO, false);
                    }
                    aabb.expand_by_scalar(self.prediction_distance);
                    GpuAabb::new(
                        aabb.min,
                        aabb.max,
                        !body.is_fixed() && !body.is_sleeping(),
                    )
                }
                _ => GpuAabb::new(Vector3::ZERO, Vector3::ZERO, false),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::RigidBody;
    use crate::shape::Shape;

    /// CPU reference: the same all-pairs test the shader performs.
    fn reference_pairs(boxes: &[GpuAabb]) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        for i in 0..boxes.len() {
            for j in i + 1..boxes.len() {
                let (a, b) = (boxes[i], boxes[j]);
                if a.min[3] == 0.0 && b.min[3] == 0.0 {
                    continue;
                }
                if a.min[0] > b.max[0] || a.max[0] < b.min[0] {
                    continue;
                }
                if a.min[1] > b.max[1] || a.max[1] < b.min[1] {
                    continue;
                }
                if a.min[2] > b.max[2] || a.max[2] < b.min[2] {
                    continue;
                }
                out.push((i as u32, j as u32));
            }
        }
        out
    }

    fn grid(n: usize, spacing: f32) -> Vec<GpuAabb> {
        (0..n)
            .map(|i| {
                let c = Vector3::new(i as f32 * spacing, 0.0, 0.0);
                GpuAabb::new(
                    c - Vector3::new(0.5, 0.5, 0.5),
                    c + Vector3::new(0.5, 0.5, 0.5),
                    true,
                )
            })
            .collect()
    }

    /// Skips rather than fails where there is no GPU — CI runners often have none.
    macro_rules! gpu_or_skip {
        () => {
            match pollster::block_on(GpuBroadPhase::new()) {
                Some(g) => g,
                None => {
                    eprintln!("no compute adapter available; skipping");
                    return;
                }
            }
        };
    }

    #[test]
    fn gpu_pairs_match_the_cpu_reference() {
        let mut gpu = gpu_or_skip!();
        // Overlapping (spacing 0.9 < box width 1.0) and disjoint cases.
        for spacing in [0.9f32, 1.5] {
            let boxes = grid(64, spacing);
            let got = gpu.find_pairs_blocking(&boxes);
            let mut want = reference_pairs(&boxes);
            want.sort_unstable();
            assert_eq!(got, want, "spacing {spacing}");
        }
    }

    #[test]
    fn a_dense_cluster_is_handled_and_stays_sorted() {
        let mut gpu = gpu_or_skip!();
        // Everything on top of everything: n*(n-1)/2 pairs.
        let boxes: Vec<GpuAabb> = (0..48)
            .map(|_| {
                GpuAabb::new(
                    Vector3::new(-1.0, -1.0, -1.0),
                    Vector3::new(1.0, 1.0, 1.0),
                    true,
                )
            })
            .collect();
        let got = gpu.find_pairs_blocking(&boxes);
        assert_eq!(got.len(), 48 * 47 / 2);
        assert!(got.windows(2).all(|w| w[0] < w[1]), "not sorted");
        assert!(got.iter().all(|(a, b)| a < b));
    }

    #[test]
    fn inactive_pairs_are_skipped() {
        let mut gpu = gpu_or_skip!();
        let overlapping = |active: bool| {
            GpuAabb::new(
                Vector3::new(-1.0, -1.0, -1.0),
                Vector3::new(1.0, 1.0, 1.0),
                active,
            )
        };
        // Two fixed bodies on top of each other: no pair.
        assert!(gpu
            .find_pairs_blocking(&[overlapping(false), overlapping(false)])
            .is_empty());
        // One active: reported.
        assert_eq!(
            gpu.find_pairs_blocking(&[overlapping(false), overlapping(true)]),
            vec![(0, 1)]
        );
    }

    #[test]
    fn trivial_inputs_do_not_dispatch() {
        let mut gpu = gpu_or_skip!();
        assert!(gpu.find_pairs_blocking(&[]).is_empty());
        assert!(gpu.find_pairs_blocking(&grid(1, 1.0)).is_empty());
    }

    #[test]
    fn world_aabbs_line_up_with_body_slots() {
        let mut world = World::new();
        world.add_body(RigidBody::fixed().shape(Shape::ground()));
        let ball = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.5))
                .translation(Vector3::new(0.0, 5.0, 0.0)),
        );
        let boxes = world.broadphase_aabbs();
        assert_eq!(boxes.len(), world.bodies().slot_count());
        // The ball's entry is at its own slot index, and is marked movable.
        let entry = boxes[ball.index()];
        assert_eq!(entry.min[3], 1.0);
        assert!((entry.max[1] - 5.5).abs() < 0.1, "{:?}", entry.max);
        // The fixed ground is not movable.
        assert_eq!(boxes[0].min[3], 0.0);
    }
}
