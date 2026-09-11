//! The wgpu compute backend: the same path tracer, as a kernel.
//!
//! # What makes it fast
//!
//! **One bind group, four storage buffers.** WebGPU's baseline guarantees only
//! four storage buffers per stage, and a scene needs a dozen arrays. They are
//! packed into `nodes` / `data` / `idx` / `accum` and addressed through offsets
//! in the uniform block. That keeps the pipeline valid on any device wgpu will
//! create — including one handed over from
//! [`HeadlessRenderer`](crate::HeadlessRenderer), which asks for downlevel
//! limits — and it means the bind group is built once per scene rather than per
//! draw.
//!
//! **One texture.** Every decoded map is shelf-packed into a single
//! `rgba16float` atlas, along with the six environment faces, and each material
//! carries the sub-rectangle it owns. Per-material texture bindings would need
//! binding arrays, which wgpu exposes only as a native-only feature and WebGPU
//! does not guarantee at all; an atlas gets the same result with one binding, no
//! rebinding between materials, and one code path on both targets.
//!
//! **Batched dispatches.** Samples are traced `samples_per_dispatch` at a time
//! rather than all at once. A single dispatch that runs for seconds trips the
//! GPU watchdog on Windows and macOS alike and takes the whole process with it;
//! batches of a few tens of milliseconds keep the device responsive, let a
//! progressive render show intermediate results, and cost one command
//! submission each.
//!
//! **Accumulate on the device.** The film lives in a storage buffer. Full-frame
//! traces read back immediately; tile/region traces defer readback until
//! [`GpuBackend::sync_film`] (called automatically from
//! [`RaytraceRenderer::resolve_rgba`] and friends).
//!
//! **One pixel per invocation, 8×8 workgroups.** No atomics: the only thread
//! that writes a pixel's accumulator is the one that owns it. 8×8 keeps
//! neighbouring pixels — whose rays go the same way and touch the same BVH
//! nodes — in the same workgroup.
//!
//! # Differences from the CPU backend
//!
//! The estimator is identical, but the random number generator is not, so the
//! two produce different *noise* from the same seed. They converge to the same
//! image; a per-pixel comparison at low sample counts will not match, and the
//! tests compare converged means instead.
//!
//! A texture's `rotation` is applied in the kernel (offset and repeat are too).
//! Any texture too large for the atlas falls back to its material's constant.
//! Both are reported by [`crate::raytrace::gpu::GpuBackend::report`].

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use crate::math::Vector3;
use crate::textures::TextureWrap;

use super::backend::{RaytraceBackend, RaytraceError};
use super::bsdf::ggx_albedo_table;
use super::camera::RtCamera;
use super::film::{Film, Pixel};
use super::scene::{RaytraceScene, RtLight, RtMaterial};
use super::settings::{BackgroundMode, RaytraceSettings};
use super::texture::CpuTexture;

/// Floats per material record in the `data` buffer.
const MAT_STRIDE: usize = 88;
/// Floats per analytic light.
const LIGHT_STRIDE: usize = 20;
/// Floats per triangle: 3 positions, 3 normals, 3 UVs.
const TRI_STRIDE: usize = 24;
/// Floats per emissive entry: area and cumulative weight.
const EMIT_STRIDE: usize = 2;
/// Floats per pixel in the accumulation buffer.
const ACCUM_STRIDE: usize = 14;
/// Side of the texture atlas when the device allows it. Packing and upload
/// clamp to [`GpuCaps::atlas_side`] on downlevel / WebGL2 / shared devices.
const ATLAS_SIZE: u32 = 8192;
/// Gutter between packed textures, so bilinear taps at a rectangle's edge
/// cannot reach its neighbour.
const ATLAS_PAD: u32 = 2;

/// Device limits that matter to the path-tracer kernel, derived once from the
/// [`wgpu::Device`] so packing and uploads fail with a clear message instead of
/// a validation error halfway through a progressive render.
#[derive(Debug, Clone, Copy)]
pub struct GpuCaps {
    /// `max_storage_buffer_binding_size` from the device.
    pub max_storage_binding: u64,
    /// `max_buffer_size` — total alloc including staging copies.
    pub max_buffer_size: u64,
    /// `max_texture_dimension_2d`.
    pub max_texture_2d: u32,
    /// `max_uniform_buffer_binding_size`.
    pub max_uniform_binding: u64,
    /// Shelf atlas side actually used (≤ [`ATLAS_SIZE`] and ≤ `max_texture_2d`).
    pub atlas_side: u32,
    /// Maximum film pixels the accum buffer may cover.
    pub max_accum_pixels: u64,
}

impl GpuCaps {
    pub fn from_device(device: &wgpu::Device) -> Self {
        Self::from_limits(&device.limits())
    }

    pub fn from_limits(limits: &wgpu::Limits) -> Self {
        let max_storage_binding = limits.max_storage_buffer_binding_size;
        let max_buffer_size = limits.max_buffer_size;
        let max_texture_2d = limits.max_texture_dimension_2d;
        let max_uniform_binding = limits.max_uniform_buffer_binding_size;

        // rgba16f scratch is 8 bytes/texel. Keep CPU packing under ~128 MiB on
        // downlevel devices unless storage headroom clearly allows more.
        let scratch_budget = max_storage_binding.min(512 << 20);
        let side_from_storage = isqrt_u64(scratch_budget / 8) as u32;
        let atlas_side = ATLAS_SIZE
            .min(max_texture_2d)
            .min(side_from_storage)
            .max(256);

        let accum_stride = (ACCUM_STRIDE * 4) as u64;
        let max_accum_pixels = (max_storage_binding / accum_stride)
            .min(max_buffer_size / accum_stride);

        Self {
            max_storage_binding,
            max_buffer_size,
            max_texture_2d,
            max_uniform_binding,
            atlas_side,
            max_accum_pixels,
        }
    }

    fn kernel_uniform_bytes() -> u64 {
        std::mem::size_of::<Uniforms>() as u64
    }

    /// Whether this device can run the compiled kernel at all.
    pub fn validate_kernel(&self) -> Result<(), RaytraceError> {
        let uniform = Self::kernel_uniform_bytes();
        if uniform > self.max_uniform_binding {
            return Err(RaytraceError::TooLarge(format!(
                "path-tracer uniforms are {} KB but this device allows {} KB per uniform binding",
                uniform / 1024,
                self.max_uniform_binding / 1024
            )));
        }
        if self.max_accum_pixels == 0 {
            return Err(RaytraceError::TooLarge(
                "device storage-buffer limits are too small for the path-tracer film"
                    .into(),
            ));
        }
        Ok(())
    }

    pub fn check_storage(&self, bytes: u64, label: &str) -> Result<(), RaytraceError> {
        if bytes > self.max_storage_binding {
            return Err(RaytraceError::TooLarge(format!(
                "{label} is {} MB but this device's storage-buffer binding limit is {} MB",
                bytes / (1 << 20),
                self.max_storage_binding / (1 << 20)
            )));
        }
        if bytes > self.max_buffer_size {
            return Err(RaytraceError::TooLarge(format!(
                "{label} is {} MB but this device's max buffer size is {} MB",
                bytes / (1 << 20),
                self.max_buffer_size / (1 << 20)
            )));
        }
        Ok(())
    }

    pub fn check_film(&self, width: u32, height: u32) -> Result<(), RaytraceError> {
        let pixels = width as u64 * height as u64;
        if pixels > self.max_accum_pixels {
            let side = isqrt_u64(self.max_accum_pixels) as u32;
            return Err(RaytraceError::TooLarge(format!(
                "film {width}×{height} ({} Mpx) exceeds this device's {} Mpx accum limit (~{side}×{side} max)",
                pixels / 1_000_000,
                self.max_accum_pixels / 1_000_000
            )));
        }
        Ok(())
    }

    pub fn check_texture_upload(&self, width: u32, height: u32) -> Result<(), RaytraceError> {
        if width > self.max_texture_2d || height > self.max_texture_2d {
            return Err(RaytraceError::TooLarge(format!(
                "atlas upload {width}×{height} exceeds max_texture_dimension_2d ({})",
                self.max_texture_2d
            )));
        }
        Ok(())
    }

    /// Largest square film side the accum buffer can cover on this device.
    pub fn max_film_side(&self) -> u32 {
        isqrt_u64(self.max_accum_pixels).max(1) as u32
    }

    /// Shrink a requested film size to fit [`Self::max_accum_pixels`], preserving
    /// aspect ratio when possible.
    pub fn clamp_film_size(&self, width: u32, height: u32) -> (u32, u32) {
        let mut w = width.max(1);
        let mut h = height.max(1);
        if self.check_film(w, h).is_ok() {
            return (w, h);
        }
        let max_side = self.max_film_side();
        let scale = (max_side as f64 / w.max(h) as f64).min(1.0);
        w = ((w as f64 * scale).floor() as u32).max(1);
        h = ((h as f64 * scale).floor() as u32).max(1);
        while w > 1 && h > 1 && self.check_film(w, h).is_err() {
            if w >= h {
                w -= 1;
            } else {
                h -= 1;
            }
        }
        (w, h)
    }
}

fn isqrt_u64(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let mut x = n;
    let mut y = x.div_ceil(2);
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    cam_pos: [f32; 4],
    cam_right: [f32; 4],
    cam_up: [f32; 4],
    cam_forward: [f32; 4],
    inv_view: [f32; 16],
    inv_proj: [f32; 16],
    background: [f32; 4],
    ambient: [f32; 4],
    dims: [u32; 4],
    counts: [u32; 4],
    data_offsets: [u32; 4],
    idx_offsets: [u32; 4],
    params: [f32; 4],
    adaptive: [f32; 4],
    limits: [u32; 4],
    flags: [u32; 4],
    env_offsets: [u32; 4],
    env_dims: [u32; 4],
    env_rect: [[f32; 4]; 6],
    hemi: [[f32; 4]; 12],
    /// Previous-frame camera for motion blur. Valid when `params[3] > 0`.
    motion_inv_view: [f32; 16],
    motion_inv_proj: [f32; 16],
    motion_prev: [f32; 4],
    motion_prev_right: [f32; 4],
    motion_prev_up: [f32; 4],
    motion_prev_forward: [f32; 4],
    /// Pixel clip: x, y, width, height. Width `0` means the full frame.
    clip: [u32; 4],
    /// x = scene scale, y = sample redistribution flag, zw unused.
    render_params: [f32; 4],
    /// rgb + mode (`0` off, `1` linear, `2` exp2).
    fog_color: [f32; 4],
    /// near, far, density, unused.
    fog_params: [f32; 4],
    albedo_lut: [[f32; 4]; 256],
}

/// The packed scene, ready to upload.
struct Packed {
    nodes: Vec<f32>,
    data: Vec<f32>,
    idx: Vec<u32>,
    tri_base: u32,
    emissive_base: u32,
    material_base: u32,
    light_base: u32,
    order_base: u32,
    tri_material_base: u32,
    emissive_slot_base: u32,
    emissive_tri_base: u32,
    atlas: Atlas,
    env_func_base: u32,
    env_cond_base: u32,
    env_marg_base: u32,
    env_dist_width: u32,
    env_dist_height: u32,
    env_dist_total: f32,
    env_rect: [[f32; 4]; 6],
    env_packed: bool,
    hemi: [[f32; 4]; 12],
    hemi_count: u32,
    notes: Vec<String>,
}

/// Device resources for one scene. Dropped and rebuilt when the scene changes.
struct SceneResources {
    bind_group: wgpu::BindGroup,
    accum: wgpu::Buffer,
    uniform: wgpu::Buffer,
    /// Identity of the scene these were built for, so a repeat `render` with
    /// the same scene reuses them.
    signature: (usize, usize, usize),
    accum_len: usize,
    /// Sample count the device accumulator holds, when it is known to match the
    /// host film. `None` forces a re-upload.
    accum_samples: Option<u32>,
}

/// A path tracer that runs on the GPU.
pub struct GpuBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    resources: Option<SceneResources>,
    /// Where each packed array starts, recomputed with the resources.
    offsets: Option<Offsets>,
    /// Samples traced per dispatch.
    samples_per_dispatch: u32,
    /// Skip readback after region traces; [`Self::sync_film`] pulls once.
    defer_readback: bool,
    /// Device accumulator is ahead of the host [`Film`] pixel buffers.
    readback_pending: bool,
    caps: GpuCaps,
    notes: Vec<String>,
}

impl std::fmt::Debug for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBackend")
            .field("samples_per_dispatch", &self.samples_per_dispatch)
            .field("prepared", &self.resources.is_some())
            .finish()
    }
}

impl GpuBackend {
    /// Acquire an adapter and build a device of our own. Blocking; native only.
    ///
    /// On `wasm32`, use [`Self::headless_browser`].
    pub fn headless() -> Result<Self, RaytraceError> {
        #[cfg(target_arch = "wasm32")]
        {
            return Err(RaytraceError::NoDevice(
                "GpuBackend::headless is blocking and native-only; use headless_browser()".into(),
            ));
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self::headless_sync()
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn headless_sync() -> Result<Self, RaytraceError> {
        // No `Default` for `InstanceDescriptor` in wgpu 30: a display handle is
        // either present or deliberately absent, and this path is headless.
        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(instance_desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .map_err(|e| RaytraceError::NoDevice(format!("no wgpu adapter: {e}")))?;

        // Ask for exactly what this adapter has. The kernel's own needs are
        // modest — four storage buffers and a 4K texture — but a large scene
        // needs the adapter's real buffer-size limit, not the conservative
        // default, and there is no portability to protect here: this device is
        // used on this adapter and nowhere else.
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("threers path tracer"),
                required_features: wgpu::Features::empty(),
                required_limits: Self::request_limits(&limits),
                ..Default::default()
            },
        ))
        .map_err(|e| RaytraceError::NoDevice(format!("request_device failed: {e:?}")))?;

        Ok(Self::with_device(Arc::new(device), Arc::new(queue)))
    }

    /// Acquire a WebGPU adapter in the browser. Async; wasm only.
    #[cfg(target_arch = "wasm32")]
    pub async fn headless_browser() -> Result<Self, RaytraceError> {
        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = wgpu::Backends::BROWSER_WEBGPU;
        let instance = wgpu::Instance::new(instance_desc);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|e| RaytraceError::NoDevice(format!("no wgpu adapter: {e}")))?;
        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("threers path tracer"),
                required_features: wgpu::Features::empty(),
                required_limits: Self::request_limits(&limits),
                ..Default::default()
            })
            .await
            .map_err(|e| RaytraceError::NoDevice(format!("request_device failed: {e:?}")))?;
        Ok(Self::with_device(Arc::new(device), Arc::new(queue)))
    }

    /// The device and queue this backend runs on.
    ///
    /// For building further backends on the same device — batch work that
    /// renders many scenes wants one adapter for the run, not one per scene,
    /// and [`Self::headless`] acquires a new device every time it is called.
    pub fn device_and_queue(&self) -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
        (Arc::clone(&self.device), Arc::clone(&self.queue))
    }

    /// Build on an existing device — the one from
    /// [`HeadlessRenderer::device`](crate::HeadlessRenderer::device), or a
    /// window's.
    pub fn with_device(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("threers pathtrace"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("pathtrace.wgsl"))),
        });

        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
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
            label: Some("threers pathtrace layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1, true),
                storage(2, true),
                storage(3, true),
                storage(4, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("threers pathtrace pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("threers pathtrace"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("trace"),
            compilation_options: Default::default(),
            cache: None,
        });

        let caps = GpuCaps::from_device(&device);
        Self {
            device,
            queue,
            layout,
            pipeline,
            resources: None,
            offsets: None,
            samples_per_dispatch: 4,
            defer_readback: true,
            readback_pending: false,
            caps,
            notes: Vec::new(),
        }
    }

    /// Limits the path tracer will respect on this device.
    pub fn caps(&self) -> GpuCaps {
        self.caps
    }

    /// Merge downlevel portability floors with whatever the adapter can actually
    /// do. wgpu/WASM/WebGL2 share a conservative baseline; Metal/CUDA stacks
    /// behind the same wgpu surface usually expose the adapter maximum.
    fn request_limits(adapter: &wgpu::Limits) -> wgpu::Limits {
        let mut limits = wgpu::Limits::downlevel_defaults().using_resolution(adapter.clone());
        limits.max_storage_buffer_binding_size = adapter.max_storage_buffer_binding_size;
        limits.max_buffer_size = adapter.max_buffer_size;
        limits.max_uniform_buffer_binding_size = adapter
            .max_uniform_buffer_binding_size
            .max(limits.max_uniform_buffer_binding_size);
        limits.max_compute_workgroup_size_x = adapter
            .max_compute_workgroup_size_x
            .max(limits.max_compute_workgroup_size_x);
        limits.max_compute_workgroup_size_y = adapter
            .max_compute_workgroup_size_y
            .max(limits.max_compute_workgroup_size_y);
        limits
    }

    /// Samples traced per dispatch. Larger amortises submission overhead;
    /// smaller keeps each dispatch short enough not to trip the GPU watchdog on
    /// a heavy scene. The default of 4 is safe for anything.
    pub fn with_samples_per_dispatch(mut self, n: u32) -> Self {
        self.samples_per_dispatch = n.max(1);
        self
    }

    /// Samples traced per compute dispatch (GPU watchdog batching).
    pub fn set_samples_per_dispatch(&mut self, n: u32) {
        self.samples_per_dispatch = n.max(1);
    }

    pub fn samples_per_dispatch(&self) -> u32 {
        self.samples_per_dispatch
    }

    /// When `true` (default), [`Self::render_regions`] skips readback until
    /// [`Self::sync_film`]. Full-frame [`RaytraceBackend::render`] always syncs.
    pub fn set_defer_readback(&mut self, on: bool) {
        self.defer_readback = on;
    }

    pub fn defer_readback(&self) -> bool {
        self.defer_readback
    }

    /// Whether the device accumulator has samples not yet copied to `film`.
    pub fn readback_pending(&self) -> bool {
        self.readback_pending
    }

    /// Whether scene BVH buffers are already resident on the device.
    pub fn scene_on_device(&self) -> bool {
        self.resources.is_some()
    }

    /// Drop deferred readback and force the next trace to re-seed the device
    /// accum buffer from the host `film`, without repacking scene geometry.
    pub fn reset_accum(&mut self) {
        self.readback_pending = false;
        if let Some(res) = self.resources.as_mut() {
            res.accum_samples = None;
        }
    }

    /// Copy the full device film into `film` when a deferred region trace ran.
    pub fn sync_film(&mut self, film: &mut Film) -> Result<(), RaytraceError> {
        if !self.readback_pending {
            return Ok(());
        }
        let Some(res) = self.resources.as_ref() else {
            return Err(RaytraceError::NoDevice("scene was not prepared".into()));
        };
        readback_full(&self.device, &self.queue, &res.accum, film)?;
        self.readback_pending = false;
        Ok(())
    }

    /// What the packing could not carry over — a texture that did not fit the
    /// atlas.
    pub fn report(&self) -> &[String] {
        &self.notes
    }

    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    /// Build (or reuse) the device-side scene.
    fn ensure_resources(
        &mut self,
        scene: &RaytraceScene,
        film: &Film,
    ) -> Result<(), RaytraceError> {
        // Cheap identity for "is this the same scene as last time". It is not a
        // hash of the contents: the contract is that a changed scene arrives
        // through `invalidate`, which `RaytraceRenderer::prepare` always calls.
        // This only has to catch the case of a caller driving the backend
        // directly and forgetting.
        let signature = (
            scene.tris.len(),
            scene.materials.len() << 20 | scene.lights.len() << 8 | scene.emissive.len().min(255),
            (film.width() as usize) << 16 | film.height() as usize,
        );
        if let Some(r) = &self.resources {
            if r.signature == signature {
                return Ok(());
            }
        }

        let packed = pack_scene(scene, self.caps);
        self.notes = packed.notes.clone();

        self.caps.validate_kernel()?;

        let nodes_bytes = (packed.nodes.len() * 4) as u64;
        let data_bytes = (packed.data.len() * 4) as u64;
        let idx_bytes = (packed.idx.len() * 4) as u64;
        self.caps.check_storage(nodes_bytes, "BVH nodes")?;
        self.caps.check_storage(data_bytes, "scene geometry")?;
        self.caps.check_storage(idx_bytes, "scene indices")?;
        self.caps.check_film(film.width(), film.height())?;

        let accum_len = film.width() as usize * film.height() as usize * ACCUM_STRIDE;
        let accum_bytes = (accum_len * 4) as u64;
        self.caps.check_storage(accum_bytes, "film accum")?;

        let mk = |label: &str, contents: &[u8], usage: wgpu::BufferUsages| {
            use wgpu::util::DeviceExt;
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage,
                })
        };
        // An empty storage buffer is invalid, so every array gets at least one
        // element; the kernel guards on the counts in the uniform anyway.
        let nodes = if packed.nodes.is_empty() {
            vec![0.0f32; 8]
        } else {
            packed.nodes
        };
        let data = if packed.data.is_empty() {
            vec![0.0f32; 4]
        } else {
            packed.data
        };
        let idx = if packed.idx.is_empty() {
            vec![0u32; 4]
        } else {
            packed.idx
        };

        let node_buf = mk(
            "pathtrace nodes",
            bytemuck::cast_slice(&nodes),
            wgpu::BufferUsages::STORAGE,
        );
        let data_buf = mk(
            "pathtrace data",
            bytemuck::cast_slice(&data),
            wgpu::BufferUsages::STORAGE,
        );
        let idx_buf = mk(
            "pathtrace idx",
            bytemuck::cast_slice(&idx),
            wgpu::BufferUsages::STORAGE,
        );

        let accum = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pathtrace accum"),
            size: (accum_len * 4) as u64,
            // COPY_DST as well as COPY_SRC: the film may already hold samples
            // from an earlier batch, and those are written in before tracing.
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pathtrace uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let atlas_view = packed.atlas.upload(&self.device, &self.queue, self.caps);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pathtrace bind group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: node_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: data_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: idx_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: accum.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
            ],
        });

        self.resources = Some(SceneResources {
            bind_group,
            accum,
            uniform,
            signature,
            accum_len,
            accum_samples: None,
        });
        // The offsets and rects computed during packing are needed at dispatch
        // time; stash them alongside the buffers.
        self.offsets = Some(Offsets {
            tri_base: packed.tri_base,
            emissive_base: packed.emissive_base,
            material_base: packed.material_base,
            light_base: packed.light_base,
            order_base: packed.order_base,
            tri_material_base: packed.tri_material_base,
            emissive_slot_base: packed.emissive_slot_base,
            emissive_tri_base: packed.emissive_tri_base,
            env_func_base: packed.env_func_base,
            env_cond_base: packed.env_cond_base,
            env_marg_base: packed.env_marg_base,
            env_dist_width: packed.env_dist_width,
            env_dist_height: packed.env_dist_height,
            env_dist_total: packed.env_dist_total,
            env_rect: packed.env_rect,
            env_packed: packed.env_packed,
            hemi: packed.hemi,
            hemi_count: packed.hemi_count,
        });
        Ok(())
    }
}

/// Where each array starts inside the packed buffers.
#[derive(Clone, Copy)]
struct Offsets {
    tri_base: u32,
    emissive_base: u32,
    material_base: u32,
    light_base: u32,
    order_base: u32,
    tri_material_base: u32,
    emissive_slot_base: u32,
    emissive_tri_base: u32,
    env_func_base: u32,
    env_cond_base: u32,
    env_marg_base: u32,
    env_dist_width: u32,
    env_dist_height: u32,
    env_dist_total: f32,
    env_rect: [[f32; 4]; 6],
    env_packed: bool,
    hemi: [[f32; 4]; 12],
    hemi_count: u32,
}

impl RaytraceBackend for GpuBackend {
    fn name(&self) -> &'static str {
        "gpu"
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn invalidate(&mut self) {
        self.resources = None;
        self.offsets = None;
        self.readback_pending = false;
    }

    fn gpu_caps(&self) -> Option<GpuCaps> {
        Some(self.caps)
    }

    fn render(
        &mut self,
        scene: &RaytraceScene,
        camera: &RtCamera,
        settings: &RaytraceSettings,
        film: &mut Film,
        first_sample: u32,
        samples: u32,
    ) -> Result<(), RaytraceError> {
        self.render_clipped(scene, camera, settings, film, first_sample, samples, None)
    }
}

impl GpuBackend {
    /// Trace only pixels inside `rect`. Untouched pixels keep their accumulators.
    pub fn render_rect(
        &mut self,
        scene: &RaytraceScene,
        camera: &RtCamera,
        settings: &RaytraceSettings,
        film: &mut Film,
        first_sample: u32,
        samples: u32,
        rect: super::RenderRect,
    ) -> Result<(), RaytraceError> {
        self.render_regions(
            scene,
            camera,
            settings,
            film,
            first_sample,
            samples,
            std::slice::from_ref(&rect),
        )
    }

    /// Trace several disjoint regions in one GPU submission.
    pub fn render_regions(
        &mut self,
        scene: &RaytraceScene,
        camera: &RtCamera,
        settings: &RaytraceSettings,
        film: &mut Film,
        first_sample: u32,
        samples: u32,
        regions: &[super::RenderRect],
    ) -> Result<(), RaytraceError> {
        if samples == 0 || regions.is_empty() {
            return Ok(());
        }
        self.trace_regions(
            scene,
            camera,
            settings,
            film,
            first_sample,
            samples,
            ReadbackScope::Regions(regions),
        )
    }

    fn render_clipped(
        &mut self,
        scene: &RaytraceScene,
        camera: &RtCamera,
        settings: &RaytraceSettings,
        film: &mut Film,
        first_sample: u32,
        samples: u32,
        clip: Option<super::RenderRect>,
    ) -> Result<(), RaytraceError> {
        if samples == 0 {
            return Ok(());
        }
        match clip {
            None => self.trace_regions(
                scene,
                camera,
                settings,
                film,
                first_sample,
                samples,
                ReadbackScope::Full,
            ),
            Some(r) => self.render_regions(
                scene,
                camera,
                settings,
                film,
                first_sample,
                samples,
                std::slice::from_ref(&r),
            ),
        }
    }

    fn trace_regions(
        &mut self,
        scene: &RaytraceScene,
        camera: &RtCamera,
        settings: &RaytraceSettings,
        film: &mut Film,
        first_sample: u32,
        samples: u32,
        scope: ReadbackScope<'_>,
    ) -> Result<(), RaytraceError> {
        self.ensure_resources(scene, film)?;
        let (Some(res), Some(off)) = (self.resources.as_ref(), self.offsets) else {
            return Err(RaytraceError::NoDevice("scene was not prepared".into()));
        };

        if res.accum_samples != Some(film.samples()) {
            let mut seed_accum = vec![0.0f32; res.accum_len];
            for (i, p) in film.pixels().iter().enumerate() {
                let b = i * ACCUM_STRIDE;
                seed_accum[b] = p.color[0];
                seed_accum[b + 1] = p.color[1];
                seed_accum[b + 2] = p.color[2];
                seed_accum[b + 3] = p.alpha;
                seed_accum[b + 4] = p.albedo[0];
                seed_accum[b + 5] = p.albedo[1];
                seed_accum[b + 6] = p.albedo[2];
                seed_accum[b + 7] = p.normal[0];
                seed_accum[b + 8] = p.normal[1];
                seed_accum[b + 9] = p.normal[2];
                seed_accum[b + 10] = p.depth;
                seed_accum[b + 11] = p.depth_samples as f32;
                seed_accum[b + 12] = p.samples as f32;
                seed_accum[b + 13] = p.lum_sq;
            }
            self.queue
                .write_buffer(&res.accum, 0, bytemuck::cast_slice(&seed_accum));
        }

        let epsilon = super::integrator::ray_epsilon(scene, settings);
        let clip_for_uniforms = match scope {
            ReadbackScope::Full => None,
            ReadbackScope::Regions(regions) if regions.len() == 1 => Some(regions[0]),
            ReadbackScope::Regions(_) => None,
        };
        let mut uniforms =
            build_uniforms(scene, camera, settings, film, &off, epsilon, clip_for_uniforms);

        let mut done = 0u32;
        while done < samples {
            let batch = self.samples_per_dispatch.min(samples - done);
            uniforms.dims[2] = first_sample + done;
            uniforms.dims[3] = batch;

            match scope {
                ReadbackScope::Full => {
                    self.queue.write_buffer(
                        &res.uniform,
                        0,
                        bytemuck::bytes_of(&uniforms),
                    );
                    let mut encoder = self.device.create_command_encoder(
                        &wgpu::CommandEncoderDescriptor {
                            label: Some("pathtrace batch"),
                        },
                    );
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("pathtrace"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.pipeline);
                    pass.set_bind_group(0, &res.bind_group, &[]);
                    pass.dispatch_workgroups(
                        film.width().div_ceil(8),
                        film.height().div_ceil(8),
                        1,
                    );
                    drop(pass);
                    self.queue.submit(Some(encoder.finish()));
                }
                ReadbackScope::Regions(regions) => {
                    for region in regions {
                        uniforms.clip = [
                            region.x,
                            region.y,
                            region.width,
                            region.height,
                        ];
                        self.queue.write_buffer(
                            &res.uniform,
                            0,
                            bytemuck::bytes_of(&uniforms),
                        );
                        let mut encoder = self.device.create_command_encoder(
                            &wgpu::CommandEncoderDescriptor {
                                label: Some("pathtrace batch"),
                            },
                        );
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("pathtrace region"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&self.pipeline);
                        pass.set_bind_group(0, &res.bind_group, &[]);
                        pass.dispatch_workgroups(
                            region.width.div_ceil(8),
                            region.height.div_ceil(8),
                            1,
                        );
                        drop(pass);
                        self.queue.submit(Some(encoder.finish()));
                    }
                }
            }
            done += batch;
        }

        let defer = self.defer_readback && matches!(scope, ReadbackScope::Regions(_));
        if defer {
            self.readback_pending = true;
        } else {
            readback_film(
                &self.device,
                &self.queue,
                &res.accum,
                film,
                scope,
            )?;
            self.readback_pending = false;
        }

        film.advance(samples);
        if let Some(res) = self.resources.as_mut() {
            res.accum_samples = Some(film.samples());
        }
        Ok(())
    }
}

enum ReadbackScope<'a> {
    Full,
    Regions(&'a [super::RenderRect]),
}

fn apply_accum_pixel(p: &mut Pixel, values: &[f32], b: usize) {
    p.color = [values[b], values[b + 1], values[b + 2]];
    p.alpha = values[b + 3];
    p.albedo = [values[b + 4], values[b + 5], values[b + 6]];
    p.normal = [values[b + 7], values[b + 8], values[b + 9]];
    p.depth = values[b + 10];
    p.depth_samples = values[b + 11] as u32;
    p.samples = values[b + 12] as u32;
    p.lum_sq = values[b + 13];
}

fn readback_film(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    accum: &wgpu::Buffer,
    film: &mut Film,
    scope: ReadbackScope<'_>,
) -> Result<(), RaytraceError> {
    let fw = film.width();
    match scope {
        ReadbackScope::Full => readback_full(device, queue, accum, film),
        ReadbackScope::Regions(regions) if regions.len() == 1 => {
            readback_one_region(device, queue, accum, film, fw, regions[0])
        }
        ReadbackScope::Regions(regions) => {
            readback_many_regions(device, queue, accum, film, fw, regions)
        }
    }
}

fn readback_full(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    accum: &wgpu::Buffer,
    film: &mut Film,
) -> Result<(), RaytraceError> {
    let staging_bytes = (film.pixels().len() * ACCUM_STRIDE * 4) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pathtrace readback"),
        size: staging_bytes.max(4),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("pathtrace readback"),
    });
    encoder.copy_buffer_to_buffer(accum, 0, &staging, 0, staging_bytes);
    queue.submit(Some(encoder.finish()));
    let values = map_staging_f32(device, &staging)?;
    for (i, p) in film.pixels_mut().iter_mut().enumerate() {
        apply_accum_pixel(p, &values, i * ACCUM_STRIDE);
    }
    Ok(())
}

fn readback_one_region(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    accum: &wgpu::Buffer,
    film: &mut Film,
    fw: u32,
    r: super::RenderRect,
) -> Result<(), RaytraceError> {
    let row_bytes = r.width as u64 * ACCUM_STRIDE as u64 * 4;
    let staging_bytes = row_bytes * r.height as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pathtrace readback"),
        size: staging_bytes.max(4),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("pathtrace readback"),
    });
    for row in 0..r.height {
        let src = ((r.y + row) * fw + r.x) as u64 * ACCUM_STRIDE as u64 * 4;
        let dst = row as u64 * row_bytes;
        encoder.copy_buffer_to_buffer(accum, src, &staging, dst, row_bytes);
    }
    queue.submit(Some(encoder.finish()));
    let values = map_staging_f32(device, &staging)?;
    let row_floats = (row_bytes / 4) as usize;
    for row in 0..r.height {
        for col in 0..r.width {
            let staging_base = row as usize * row_floats + col as usize * ACCUM_STRIDE;
            let idx = (r.y + row) as usize * fw as usize + (r.x + col) as usize;
            apply_accum_pixel(&mut film.pixels_mut()[idx], &values, staging_base);
        }
    }
    Ok(())
}

fn readback_many_regions(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    accum: &wgpu::Buffer,
    film: &mut Film,
    fw: u32,
    regions: &[super::RenderRect],
) -> Result<(), RaytraceError> {
    let mut layouts = Vec::with_capacity(regions.len());
    let mut staging_bytes = 0u64;
    for r in regions {
        let row_bytes = r.width as u64 * ACCUM_STRIDE as u64 * 4;
        layouts.push(( *r, staging_bytes, row_bytes));
        staging_bytes += row_bytes * r.height as u64;
    }
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pathtrace readback"),
        size: staging_bytes.max(4),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("pathtrace readback"),
    });
    for (r, base, row_bytes) in &layouts {
        for row in 0..r.height {
            let src = ((r.y + row) * fw + r.x) as u64 * ACCUM_STRIDE as u64 * 4;
            let dst = *base + row as u64 * *row_bytes;
            encoder.copy_buffer_to_buffer(accum, src, &staging, dst, *row_bytes);
        }
    }
    queue.submit(Some(encoder.finish()));
    let values = map_staging_f32(device, &staging)?;
    for (r, base, row_bytes) in layouts {
        let row_floats = (row_bytes / 4) as usize;
        let base = base as usize / 4;
        for row in 0..r.height {
            for col in 0..r.width {
                let staging_base =
                    base + row as usize * row_floats + col as usize * ACCUM_STRIDE;
                let idx = (r.y + row) as usize * fw as usize + (r.x + col) as usize;
                apply_accum_pixel(&mut film.pixels_mut()[idx], &values, staging_base);
            }
        }
    }
    Ok(())
}

fn map_staging_f32(device: &wgpu::Device, staging: &wgpu::Buffer) -> Result<Vec<f32>, RaytraceError> {
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    match rx.recv() {
        Ok(Ok(())) => {}
        other => {
            return Err(RaytraceError::DeviceLost(format!(
                "readback failed: {other:?}"
            )))
        }
    }
    let mapped = slice.get_mapped_range().expect("buffer range is mapped");
    let values: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    Ok(values)
}

fn build_uniforms(
    scene: &RaytraceScene,
    camera: &RtCamera,
    settings: &RaytraceSettings,
    film: &Film,
    off: &Offsets,
    epsilon: f32,
    clip: Option<super::RenderRect>,
) -> Uniforms {
    let inv_view = camera.inv_view_matrix();
    let inv_proj = camera.inv_proj_matrix();
    let pos = camera.position();
    let right = camera.right();
    let up = camera.up();
    let fwd = camera.forward();

    let mut lut = [[0.0f32; 4]; 256];
    for (i, v) in ggx_albedo_table().iter().enumerate() {
        lut[i / 4][i % 4] = *v;
    }

    let mut flags = 0u32;
    // The scene having an environment is not enough — it also has to have made
    // it into the atlas.
    if scene.world.has_environment() && off.env_packed {
        flags |= 1;
    }
    if scene.shadows_all_opaque {
        flags |= 2;
    }
    // Biased glass-shadow caustic hops (matches CPU visibility).
    if settings.caustic_glass_shadows {
        flags |= 4;
    }

    let motion_fields = motion_uniforms(camera);
    let motion_shutter = if camera.motion_shutter() > 0.0 && camera.motion_previous().is_some() {
        camera.motion_shutter()
    } else {
        0.0
    };
    let clip_arr = match clip {
        Some(r) => [r.x, r.y, r.width, r.height],
        None => [0, 0, 0, 0],
    };
    let redistribute = settings.sample_redistribution && settings.adaptive_threshold > 0.0;

    Uniforms {
        cam_pos: [
            pos.x,
            pos.y,
            pos.z,
            if camera.is_perspective() { 1.0 } else { 0.0 },
        ],
        cam_right: [right.x, right.y, right.z, settings.aperture],
        cam_up: [up.x, up.y, up.z, camera.focus_distance()],
        cam_forward: [fwd.x, fwd.y, fwd.z, epsilon],
        inv_view: inv_view.elements,
        inv_proj: inv_proj.elements,
        background: [
            scene.world.background.x,
            scene.world.background.y,
            scene.world.background.z,
            scene.world.background_alpha,
        ],
        ambient: [
            scene.world.ambient.x,
            scene.world.ambient.y,
            scene.world.ambient.z,
            scene.world.environment_intensity,
        ],
        dims: [film.width(), film.height(), 0, 0],
        counts: [
            scene.tris.len() as u32,
            scene.materials.len() as u32,
            scene.lights.len() as u32,
            scene.emissive.len() as u32,
        ],
        data_offsets: [
            off.tri_base,
            off.emissive_base,
            off.material_base,
            off.light_base,
        ],
        idx_offsets: [
            off.order_base,
            off.tri_material_base,
            off.emissive_slot_base,
            off.emissive_tri_base,
        ],
        params: [
            settings.clamp_direct,
            settings.clamp_indirect,
            off.env_dist_total,
            motion_shutter,
        ],
        adaptive: [
            settings.adaptive_threshold,
            settings.adaptive_min_samples as f32,
            0.0,
            0.0,
        ],
        limits: [
            settings.max_bounces,
            settings.min_bounces,
            settings.transparent_max_bounces,
            match settings.background {
                BackgroundMode::Color => 0,
                BackgroundMode::Environment => 1,
                BackgroundMode::Transparent => 2,
            },
        ],
        flags: [
            settings.seed as u32,
            (settings.seed >> 32) as u32,
            off.hemi_count,
            flags,
        ],
        env_offsets: [off.env_func_base, off.env_cond_base, off.env_marg_base, 0],
        env_dims: [
            off.env_dist_width,
            off.env_dist_height,
            // Only usable when the environment itself made it into the atlas —
            // importance-sampling a sky the kernel cannot read would aim every
            // sample at radiance of zero.
            u32::from(off.env_dist_width > 0 && (flags & 1) != 0),
            0,
        ],
        env_rect: off.env_rect,
        hemi: off.hemi,
        motion_inv_view: motion_fields.0,
        motion_inv_proj: motion_fields.1,
        motion_prev: motion_fields.2,
        motion_prev_right: motion_fields.3,
        motion_prev_up: motion_fields.4,
        motion_prev_forward: motion_fields.5,
        clip: clip_arr,
        render_params: [
            scene.scale(),
            if redistribute { 1.0 } else { 0.0 },
            0.0,
            0.0,
        ],
        fog_color: [
            scene.world.fog_color.x,
            scene.world.fog_color.y,
            scene.world.fog_color.z,
            scene.world.fog_mode as f32,
        ],
        fog_params: [
            scene.world.fog_near,
            scene.world.fog_far,
            scene.world.fog_density,
            0.0,
        ],
        albedo_lut: lut,
    }
}

/// Two matrices and four vectors: the previous view-projection, the current
/// one, and the shutter's origin and direction deltas.
type MotionUniforms = ([f32; 16], [f32; 16], [f32; 4], [f32; 4], [f32; 4], [f32; 4]);

fn motion_uniforms(camera: &RtCamera) -> MotionUniforms {
    let Some(prev) = camera.motion_previous() else {
        return ([0.0; 16], [0.0; 16], [0.0; 4], [0.0; 4], [0.0; 4], [0.0; 4]);
    };
    if camera.motion_shutter() <= 0.0 {
        return ([0.0; 16], [0.0; 16], [0.0; 4], [0.0; 4], [0.0; 4], [0.0; 4]);
    }
    let p = prev.position();
    (
        prev.inv_view_matrix().elements,
        prev.inv_proj_matrix().elements,
        [
            p.x,
            p.y,
            p.z,
            if prev.is_perspective() { 1.0 } else { 0.0 },
        ],
        [prev.right().x, prev.right().y, prev.right().z, 0.0],
        [prev.up().x, prev.up().y, prev.up().z, 0.0],
        [prev.forward().x, prev.forward().y, prev.forward().z, 0.0],
    )
}

// ------------------------------------------------------------------- packing

/// Shelf packer for the texture atlas. Textures arrive in no useful order, so
/// they are packed as they come into rows whose height is set by the first
/// texture in each — simple, and within a few percent of optimal for the
/// power-of-two sizes textures actually are.
struct Atlas {
    side: u32,
    pixels: Vec<u16>,
    shelf_x: u32,
    shelf_y: u32,
    shelf_height: u32,
    /// Rightmost column any shelf reached, so the upload can be cropped.
    used_width: u32,
    /// Rectangles already packed, by `Arc` identity — a map shared by twenty
    /// materials is copied in once, as it is decoded once.
    placed: HashMap<usize, [f32; 4]>,
}

impl Atlas {
    fn new(side: u32) -> Self {
        Self {
            side: side.max(256),
            pixels: Vec::new(),
            shelf_x: 0,
            shelf_y: 0,
            shelf_height: 0,
            used_width: 0,
            placed: HashMap::new(),
        }
    }

    /// As [`Self::insert`], but returns the existing rectangle when this exact
    /// texture has already been packed.
    fn insert_shared(&mut self, tex: &Arc<CpuTexture>) -> Option<[f32; 4]> {
        let key = Arc::as_ptr(tex) as usize;
        if let Some(rect) = self.placed.get(&key) {
            return Some(*rect);
        }
        let rect = self.insert(tex)?;
        self.placed.insert(key, rect);
        Some(rect)
    }

    fn ensure_allocated(&mut self) {
        if self.pixels.is_empty() {
            self.pixels = vec![0u16; (self.side * self.side * 4) as usize];
        }
    }

    /// Copy a decoded texture in and return its rectangle, or `None` if it does
    /// not fit.
    fn insert(&mut self, tex: &CpuTexture) -> Option<[f32; 4]> {
        let (w, h) = (tex.width(), tex.height());
        if w == 0 || h == 0 || w > self.side || h > self.side {
            return None;
        }
        if self.shelf_x + w + ATLAS_PAD > self.side {
            self.shelf_y += self.shelf_height + ATLAS_PAD;
            self.shelf_x = 0;
            self.shelf_height = 0;
        }
        if self.shelf_y + h > self.side {
            return None;
        }
        self.ensure_allocated();
        let (x0, y0) = (self.shelf_x, self.shelf_y);
        let flip = tex.flips_y();
        for y in 0..h {
            // `flip_y` is baked in here, so the kernel never has to know about
            // it — one branch removed from the innermost sampling path.
            let src_y = if flip { h - 1 - y } else { y };
            for x in 0..w {
                let c = tex.texel_at(x, src_y);
                let d = (((y0 + y) * self.side + x0 + x) * 4) as usize;
                for (dst, v) in self.pixels[d..d + 4].iter_mut().zip(c) {
                    // Clamped to the largest finite half. An HDR environment
                    // can legitimately hold values past this, and letting one
                    // become an infinity would put a NaN into the film on the
                    // first ray that found it — permanently, since the film is
                    // a running sum.
                    *dst =
                        crate::renderer::gpu_texture::f32_to_f16_bits(v.clamp(-65504.0, 65504.0));
                }
            }
        }
        self.shelf_x += w + ATLAS_PAD;
        self.used_width = self.used_width.max(x0 + w);
        self.shelf_height = self.shelf_height.max(h);
        Some([x0 as f32, y0 as f32, w as f32, h as f32])
    }

    /// Crop to what was packed and upload.
    ///
    /// The binding has to exist even for a scene with no textures at all, but
    /// it does not have to be 4096² — that is 134 MB of `rgba16float` allocated,
    /// zeroed and pushed across the bus to be sampled zero times. Packing runs
    /// at full width so the shelf arithmetic stays simple; only the upload is
    /// cropped, and since rectangles are measured from the origin they stay
    /// valid.
    fn upload(self, device: &wgpu::Device, queue: &wgpu::Queue, caps: GpuCaps) -> wgpu::TextureView {
        let used_w = self.used_width.max(1);
        let used_h = (self.shelf_y + self.shelf_height).max(1);
        let (upload_w, upload_h, pixels) = if caps.check_texture_upload(used_w, used_h).is_ok() {
            let mut pixels = vec![0u16; (used_w * used_h * 4) as usize];
            if !self.pixels.is_empty() {
                for y in 0..used_h {
                    let src = (y * self.side * 4) as usize;
                    let dst = (y * used_w * 4) as usize;
                    let n = (used_w * 4) as usize;
                    pixels[dst..dst + n].copy_from_slice(&self.pixels[src..src + n]);
                }
            }
            (used_w, used_h, pixels)
        } else {
            (1u32, 1u32, vec![0u16; 4])
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pathtrace atlas"),
            size: wgpu::Extent3d {
                width: upload_w,
                height: upload_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&pixels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(upload_w * 8),
                rows_per_image: Some(upload_h),
            },
            wgpu::Extent3d {
                width: upload_w,
                height: upload_h,
                depth_or_array_layers: 1,
            },
        );
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }
}

fn wrap_code(w: TextureWrap) -> u32 {
    match w {
        TextureWrap::ClampToEdge => 0,
        TextureWrap::Repeat => 1,
        TextureWrap::MirroredRepeat => 2,
    }
}

/// Flatten the scene into the four device buffers.
fn pack_scene(scene: &RaytraceScene, caps: GpuCaps) -> Packed {
    let n_tris = scene.tris.len();
    let mut atlas = Atlas::new(caps.atlas_side);
    let mut notes: Vec<String> = Vec::new();
    if caps.atlas_side < ATLAS_SIZE {
        notes.push(format!(
            "atlas capped at {}² texels (device max texture {}); oversized maps use material constants",
            caps.atlas_side, caps.max_texture_2d
        ));
    }

    // --- nodes: two vec4 each, with the child links bitcast into `w`.
    let mut nodes = Vec::with_capacity(scene.bvh.node_count() * 8);
    for i in 0..scene.bvh.node_count() {
        let n = scene.bvh.node(i);
        nodes.extend_from_slice(&[
            n.min[0],
            n.min[1],
            n.min[2],
            f32::from_bits(n.left_or_first),
            n.max[0],
            n.max[1],
            n.max[2],
            f32::from_bits(n.count),
        ]);
    }

    // --- data: triangles, then emissive, then materials, then lights.
    let mut data: Vec<f32> = Vec::with_capacity(n_tris * TRI_STRIDE + 4096);
    let tri_base = 0u32;
    for (i, t) in scene.tris.iter().enumerate() {
        let sh = &scene.shading[i];
        for v in t {
            data.extend_from_slice(&[v.x, v.y, v.z]);
        }
        for n in &sh.normals {
            data.extend_from_slice(&[n.x, n.y, n.z]);
        }
        for uv in &sh.uvs {
            data.extend_from_slice(&[uv.x, uv.y]);
        }
    }

    let emissive_base = data.len() as u32;
    for e in &scene.emissive.tris {
        let start = data.len();
        data.resize(start + EMIT_STRIDE, 0.0);
        data[start] = e.area;
        data[start + 1] = e.cdf;
    }

    let material_base = data.len() as u32;
    for (mi, m) in scene.materials.iter().enumerate() {
        let start = data.len();
        data.resize(start + MAT_STRIDE, 0.0);
        let s = &mut data[start..start + MAT_STRIDE];
        s[0] = m.base_color.x;
        s[1] = m.base_color.y;
        s[2] = m.base_color.z;
        s[3] = m.opacity;
        s[4] = m.emission.x;
        s[5] = m.emission.y;
        s[6] = m.emission.z;
        s[7] = m.alpha_test;
        s[8] = m.roughness;
        s[9] = m.metallic;
        s[10] = m.transmission;
        s[11] = m.ior;
        s[12] = m.attenuation_color.x;
        s[13] = m.attenuation_color.y;
        s[14] = m.attenuation_color.z;
        s[15] = m.attenuation_distance;
        s[16] = m.clearcoat;
        s[17] = m.clearcoat_roughness;
        s[18] = m.specular_tint.x;
        s[19] = m.specular_tint.y;
        s[20] = m.specular_tint.z;
        s[21] = m.anisotropy;
        s[22] = m.anisotropy_rotation;
        s[23] = if m.unlit { 1.0 } else { 0.0 };
        s[24] = m.sheen;
        s[25] = m.sheen_color.x;
        s[26] = m.sheen_color.y;
        s[27] = m.sheen_color.z;
        s[28] = m.sheen_roughness;
        s[29] = m.iridescence;
        s[30] = m.iridescence_ior;
        s[31] = m.iridescence_thickness;
        s[32] = m.dispersion;
        s[33] = m.subsurface;
        s[34] = m.subsurface_radius.x;
        s[35] = m.subsurface_radius.y;
        s[84] = m.subsurface_radius.z;
        s[81] = m.normal_scale.x;
        s[82] = m.normal_scale.y;

        let mut wrap_bits = 0u32;
        for (slot, tex) in material_maps(m).iter().enumerate() {
            let Some(tex) = tex else { continue };
            let o = 36 + slot * 9;
            match atlas.insert_shared(tex) {
                Some(rect) => {
                    s[o..o + 4].copy_from_slice(&rect);
                    let (offset, repeat, rotation) = tex.uv_transform();
                    s[o + 4] = offset.x;
                    s[o + 5] = offset.y;
                    s[o + 6] = repeat.x;
                    s[o + 7] = repeat.y;
                    s[o + 8] = rotation;
                    let (wrap_s, wrap_t) = tex.wrap_modes();
                    wrap_bits |= wrap_code(wrap_s) << (slot * 2);
                    wrap_bits |= wrap_code(wrap_t) << (10 + slot * 2);
                }
                None => notes.push(format!(
                    "material {mi}, map {slot}: {}x{} does not fit the atlas; the constant is used instead",
                    tex.width(),
                    tex.height()
                )),
            }
        }
        s[83] = wrap_bits as f32;
    }

    let light_base = data.len() as u32;
    for l in &scene.lights {
        let start = data.len();
        data.resize(start + LIGHT_STRIDE, 0.0);
        let s = &mut data[start..start + LIGHT_STRIDE];
        match *l {
            RtLight::Directional {
                direction,
                radiance,
                angular_radius,
            } => {
                s[0] = 0.0;
                write3(s, 1, direction);
                write3(s, 4, radiance);
                s[7] = angular_radius;
            }
            RtLight::Point {
                position,
                intensity,
                radius,
                distance,
                decay,
            } => {
                s[0] = 1.0;
                write3(s, 1, position);
                write3(s, 4, intensity);
                s[7] = radius;
                s[11] = distance;
                s[15] = decay;
            }
            RtLight::Spot {
                position,
                direction,
                intensity,
                radius,
                distance,
                decay,
                cos_outer,
                cos_inner,
            } => {
                s[0] = 2.0;
                write3(s, 1, position);
                write3(s, 4, intensity);
                s[7] = radius;
                write3(s, 8, direction);
                s[11] = distance;
                s[15] = decay;
                s[16] = cos_outer;
                s[17] = cos_inner;
            }
            RtLight::Rect {
                position,
                right,
                up,
                radiance,
            } => {
                s[0] = 3.0;
                write3(s, 1, position);
                write3(s, 4, radiance);
                write3(s, 8, right);
                write3(s, 12, up);
            }
        }
    }

    // --- the environment's sampling distribution. Only the CDFs and the
    // function are uploaded; the per-row integrals exist to *build* the
    // marginal CDF and are not read again.
    let mut env_func_base = 0u32;
    let mut env_cond_base = 0u32;
    let mut env_marg_base = 0u32;
    let mut env_dist_width = 0u32;
    let mut env_dist_height = 0u32;
    let mut env_dist_total = 0.0f32;
    if let Some(dist) = scene.world.env_distribution() {
        let (func, cond, _row, marg) = dist.arrays();
        env_func_base = data.len() as u32;
        data.extend_from_slice(func);
        env_cond_base = data.len() as u32;
        data.extend_from_slice(cond);
        env_marg_base = data.len() as u32;
        data.extend_from_slice(marg);
        env_dist_width = dist.width() as u32;
        env_dist_height = dist.height() as u32;
        env_dist_total = dist.total();
    }

    // --- idx: order, per-triangle material, emissive slot, emissive triangle.
    let order_base = 0u32;
    let mut idx: Vec<u32> = Vec::with_capacity(n_tris * 3 + scene.emissive.len());
    idx.extend_from_slice(scene.bvh.order());
    let tri_material_base = idx.len() as u32;
    idx.extend(scene.shading.iter().map(|s| s.material));
    let emissive_slot_base = idx.len() as u32;
    // The scene already keeps this mapping; searching the emitter list per
    // triangle would be O(triangles x emitters), which on a scene with a
    // hundred thousand triangles and a thousand emitters is a hundred million
    // comparisons for a table that already exists.
    if scene.emissive.slots().len() == n_tris {
        idx.extend_from_slice(scene.emissive.slots());
    } else {
        idx.resize(idx.len() + n_tris, u32::MAX);
    }
    let emissive_tri_base = idx.len() as u32;
    idx.extend(scene.emissive.tris.iter().map(|e| e.triangle));

    // --- environment, packed into the same atlas.
    let mut env_rect = [[0.0f32; 4]; 6];
    // All six faces or none. A missing rectangle reads as a full-white sample,
    // so a partial pack would light the scene from a white sky rather than
    // dropping the environment — the loud failure of the two.
    let mut env_packed = false;
    if let Some(faces) = scene.world.env_faces() {
        env_packed = true;
        for (i, face) in faces.iter().enumerate() {
            match atlas.insert(face) {
                Some(rect) => env_rect[i] = rect,
                None => env_packed = false,
            }
        }
        if !env_packed {
            env_rect = [[0.0f32; 4]; 6];
            notes.push(
                "environment map does not fit the texture atlas; the GPU backend renders no environment"
                    .into(),
            );
        }
    }

    let mut hemi = [[0.0f32; 4]; 12];
    let hemi_count = scene.world.hemispheres.len().min(4);
    for (i, h) in scene.world.hemispheres.iter().take(4).enumerate() {
        hemi[i * 3] = [h.sky.x, h.sky.y, h.sky.z, 0.0];
        hemi[i * 3 + 1] = [h.ground.x, h.ground.y, h.ground.z, 0.0];
        hemi[i * 3 + 2] = [h.up.x, h.up.y, h.up.z, 0.0];
    }
    if scene.world.hemispheres.len() > 4 {
        notes.push(format!(
            "{} hemisphere lights; the GPU kernel carries 4",
            scene.world.hemispheres.len()
        ));
    }

    Packed {
        nodes,
        data,
        idx,
        tri_base,
        emissive_base,
        material_base,
        light_base,
        order_base,
        tri_material_base,
        emissive_slot_base,
        emissive_tri_base,
        atlas,
        env_func_base,
        env_cond_base,
        env_marg_base,
        env_dist_width,
        env_dist_height,
        env_dist_total,
        env_rect,
        env_packed,
        hemi,
        hemi_count: hemi_count as u32,
        notes,
    }
}

/// The five map slots, in the order the kernel indexes them.
fn material_maps(m: &RtMaterial) -> [Option<Arc<CpuTexture>>; 5] {
    [
        m.base_color_map.clone(),
        m.roughness_map.clone(),
        m.metallic_map.clone(),
        m.emissive_map.clone(),
        m.normal_map.clone(),
    ]
}

fn write3(s: &mut [f32], at: usize, v: Vector3) {
    s[at] = v.x;
    s[at + 1] = v.y;
    s[at + 2] = v.z;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cameras::PerspectiveCamera;
    use crate::core::{Mesh, Object3D};
    use crate::geometries::{BoxGeometry, PlaneGeometry, SphereGeometry};
    use crate::lights::{AmbientLight, DirectionalLight};
    use crate::materials::{Material, StandardMaterial};
    use crate::math::Color;
    use crate::raytrace::backend::CpuBackend;
    use crate::raytrace::camera::RtCamera;
    use crate::raytrace::film::Film;
    use crate::raytrace::scene::RaytraceScene;
    use crate::raytrace::settings::RaytraceSettings;
    use crate::scene::Scene;

    /// Machines without a usable adapter (CI containers, mostly) skip rather
    /// than fail — but a machine that *has* one must not silently skip, so the
    /// reason is printed.
    fn backend() -> Option<GpuBackend> {
        match GpuBackend::headless() {
            Ok(b) => Some(b),
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                None
            }
        }
    }

    fn camera() -> PerspectiveCamera {
        let mut c = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
        c.position = Vector3::new(2.5, 2.0, 4.0);
        c.target = Vector3::ZERO;
        c
    }

    fn lit_box_scene() -> Scene {
        let mut scene = Scene::new();
        scene.background = Color::new(0.05, 0.06, 0.09);
        scene.add(Object3D::mesh(Mesh::new(
            BoxGeometry::new(2.0, 2.0, 2.0),
            Material::Standard(StandardMaterial::new(Color::new(0.75, 0.35, 0.2))),
        )));
        let mut floor = Object3D::mesh(Mesh::new(
            PlaneGeometry::new(20.0, 20.0),
            Material::Standard(StandardMaterial::new(Color::new(0.6, 0.6, 0.6))),
        ));
        floor.position = Vector3::new(0.0, -1.0, 0.0);
        floor.rotate_x(-std::f32::consts::FRAC_PI_2);
        scene.add(floor);
        let mut sun = Object3D::light(DirectionalLight::new(Color::WHITE, 2.5));
        sun.position = Vector3::new(3.0, 6.0, 4.0);
        scene.add(sun);
        scene.add_light(AmbientLight::new(Color::new(0.4, 0.5, 0.7), 0.3));
        scene
    }

    fn mean_channel(hdr: &[f32], c: usize) -> f32 {
        let mut sum = 0.0f64;
        let mut n = 0usize;
        for px in hdr.chunks_exact(4) {
            sum += px[c] as f64;
            n += 1;
        }
        (sum / n.max(1) as f64) as f32
    }

    #[test]
    fn the_kernel_compiles_and_runs() {
        let Some(mut gpu) = backend() else { return };
        let mut scene = Scene::new();
        scene.background = Color::new(0.25, 0.5, 0.75);
        let settings = RaytraceSettings::default()
            .with_samples(2)
            .with_denoise(false);
        let rt = RaytraceScene::build(&mut scene, &settings);
        let cam = RtCamera::new(&camera(), &settings);
        let mut film = Film::new(16, 16);
        gpu.render(&rt, &cam, &settings, &mut film, 0, 2).unwrap();
        let hdr = film.resolve_hdr();
        // An empty scene is the background, exactly.
        assert!((hdr[0] - 0.25).abs() < 2e-3, "{:?}", &hdr[..4]);
        assert!((hdr[1] - 0.5).abs() < 2e-3);
        assert!((hdr[2] - 0.75).abs() < 2e-3);
        assert!((hdr[3] - 1.0).abs() < 1e-4);
    }

    /// The furnace test again, this time on the device: a Lambertian of albedo
    /// `a` under uniform radiance `L` must render at `a·L`.
    #[test]
    fn gpu_furnace_test() {
        let Some(mut gpu) = backend() else { return };
        let mut scene = Scene::new();
        scene.background = Color::BLACK;
        scene.add(Object3D::mesh(Mesh::new(
            PlaneGeometry::new(50.0, 50.0),
            Material::Standard(StandardMaterial::new(Color::new(0.6, 0.6, 0.6))),
        )));
        scene.add_light(AmbientLight::new(Color::WHITE, 1.0));

        let settings = RaytraceSettings {
            samples_per_pixel: 256,
            max_bounces: 1,
            min_bounces: 1,
            clamp_indirect: 0.0,
            denoise: false,
            ..Default::default()
        };
        let rt = RaytraceScene::build(&mut scene, &settings);
        let mut cam = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
        cam.position = Vector3::new(0.0, 0.0, 4.0);
        cam.target = Vector3::ZERO;
        let rtc = RtCamera::new(&cam, &settings);
        let mut film = Film::new(16, 16);
        gpu.render(&rt, &rtc, &settings, &mut film, 0, 256).unwrap();
        let hdr = film.resolve_hdr();
        let i = (8 * 16 + 8) * 4;
        assert!((hdr[i] - 0.6).abs() < 0.03, "expected 0.6, got {}", hdr[i]);
    }

    /// The whole point of a second backend: it has to agree with the first.
    /// Compared as converged means over the frame, because the two use
    /// different random sequences and so produce different noise.
    #[test]
    fn gpu_and_cpu_agree_on_a_lit_scene() {
        let Some(mut gpu) = backend() else { return };
        let mut scene = lit_box_scene();
        let settings = RaytraceSettings {
            samples_per_pixel: 96,
            max_bounces: 4,
            min_bounces: 4,
            clamp_indirect: 0.0,
            denoise: false,
            ..Default::default()
        };
        let rt = RaytraceScene::build(&mut scene, &settings);
        let cam = RtCamera::new(&camera(), &settings);

        let mut gpu_film = Film::new(48, 48);
        gpu.render(&rt, &cam, &settings, &mut gpu_film, 0, 96)
            .unwrap();
        let mut cpu_film = Film::new(48, 48);
        CpuBackend::new()
            .render(&rt, &cam, &settings, &mut cpu_film, 0, 96)
            .unwrap();

        let g = gpu_film.resolve_hdr();
        let c = cpu_film.resolve_hdr();
        for channel in 0..3 {
            let gm = mean_channel(&g, channel);
            let cm = mean_channel(&c, channel);
            assert!(gm > 0.01, "channel {channel} rendered black on the GPU");
            assert!(
                (gm - cm).abs() < 0.06 * cm.max(0.05),
                "channel {channel}: gpu mean {gm}, cpu mean {cm}"
            );
        }

        // And the two must agree on the structure, not just the average: the
        // per-pixel difference has to be small where the image is smooth.
        let mut worst = 0.0f32;
        for (a, b) in g.chunks_exact(4).zip(c.chunks_exact(4)) {
            worst = worst.max((a[0] - b[0]).abs());
        }
        assert!(worst < 0.6, "worst per-pixel red difference {worst}");
    }

    #[test]
    fn gpu_handles_emissive_geometry_and_glass() {
        let Some(mut gpu) = backend() else { return };
        let mut scene = Scene::new();
        scene.background = Color::BLACK;
        let mut emitter = StandardMaterial::new(Color::BLACK);
        emitter.emissive = Color::WHITE;
        emitter.emissive_intensity = 12.0;
        let mut panel = Object3D::mesh(Mesh::new(
            PlaneGeometry::new(2.0, 2.0),
            Material::Standard(emitter),
        ));
        panel.position = Vector3::new(0.0, 0.0, 3.0);
        panel.rotate_y(std::f32::consts::PI);
        scene.add(panel);

        let mut glass = crate::materials::PhysicalMaterial::new(Color::WHITE);
        glass.transmission = 1.0;
        glass.roughness = 0.05;
        glass.ior = 1.5;
        scene.add(Object3D::mesh(Mesh::new(
            SphereGeometry::new(0.8, 24, 16),
            Material::Physical(glass),
        )));
        let mut wall = Object3D::mesh(Mesh::new(
            PlaneGeometry::new(20.0, 20.0),
            Material::Standard(StandardMaterial::new(Color::new(0.7, 0.7, 0.7))),
        ));
        wall.position = Vector3::new(0.0, 0.0, -3.0);
        scene.add(wall);

        let settings = RaytraceSettings {
            samples_per_pixel: 32,
            max_bounces: 6,
            denoise: false,
            ..Default::default()
        };
        let rt = RaytraceScene::build(&mut scene, &settings);
        assert!(!rt.emissive.is_empty());
        let mut cam = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
        cam.position = Vector3::new(0.0, 0.0, 6.0);
        cam.target = Vector3::ZERO;
        let rtc = RtCamera::new(&cam, &settings);
        let mut film = Film::new(32, 32);
        gpu.render(&rt, &rtc, &settings, &mut film, 0, 32).unwrap();
        let hdr = film.resolve_hdr();
        assert!(
            hdr.iter().all(|v| v.is_finite() && *v >= 0.0),
            "non-finite output"
        );
        assert!(mean_channel(&hdr, 0) > 0.005, "the emitter lit nothing");
    }

    /// Batching must not change the result, and the film has to survive being
    /// added to across calls.
    #[test]
    fn gpu_accumulation_is_additive() {
        let Some(mut gpu) = backend() else { return };
        let mut scene = lit_box_scene();
        let settings = RaytraceSettings {
            samples_per_pixel: 32,
            max_bounces: 3,
            denoise: false,
            ..Default::default()
        };
        let rt = RaytraceScene::build(&mut scene, &settings);
        let cam = RtCamera::new(&camera(), &settings);

        let mut one = Film::new(24, 24);
        gpu.render(&rt, &cam, &settings, &mut one, 0, 32).unwrap();

        let mut split = Film::new(24, 24);
        gpu.render(&rt, &cam, &settings, &mut split, 0, 12).unwrap();
        gpu.render(&rt, &cam, &settings, &mut split, 12, 20)
            .unwrap();
        assert_eq!(split.samples(), 32);

        let a = one.resolve_hdr();
        let b = split.resolve_hdr();
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < 1e-4,
                "channel {i}: {x} vs {y} — batching changed the sample sequence"
            );
        }
    }

    /// The kernel must handle the glass interface the same way the CPU does —
    /// the relative index flips on the way out, and getting that wrong costs a
    /// factor of eta^4.
    #[test]
    fn gpu_clear_glass_passes_light_through() {
        let Some(mut gpu) = backend() else { return };
        let build = |with_ball: bool| {
            let mut scene = Scene::new();
            scene.background = Color::BLACK;
            let mut em = StandardMaterial::new(Color::BLACK);
            em.emissive = Color::WHITE;
            em.emissive_intensity = 4.0;
            let mut wall = Object3D::mesh(Mesh::new(
                PlaneGeometry::new(20.0, 20.0),
                Material::Standard(em),
            ));
            wall.position = Vector3::new(0.0, 0.0, -4.0);
            scene.add(wall);
            if with_ball {
                let mut glass = crate::materials::PhysicalMaterial::new(Color::WHITE);
                glass.transmission = 1.0;
                glass.roughness = 0.0;
                glass.ior = 1.52;
                scene.add(Object3D::mesh(Mesh::new(
                    SphereGeometry::new(1.0, 48, 32),
                    Material::Physical(glass),
                )));
            }
            scene
        };
        let settings = RaytraceSettings {
            samples_per_pixel: 64,
            max_bounces: 12,
            min_bounces: 8,
            clamp_indirect: 0.0,
            denoise: false,
            ..Default::default()
        };
        let mut cam = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
        cam.position = Vector3::new(0.0, 0.0, 5.0);
        cam.target = Vector3::ZERO;

        let mut centre = |with_ball: bool| {
            let mut scene = build(with_ball);
            let rt = RaytraceScene::build(&mut scene, &settings);
            let rtc = RtCamera::new(&cam, &settings);
            let mut film = Film::new(32, 32);
            gpu.render(&rt, &rtc, &settings, &mut film, 0, 64).unwrap();
            film.resolve_hdr()[(16 * 32 + 16) * 4]
        };
        let clear = centre(false);
        let through_glass = centre(true);
        assert!(
            (clear - 4.0).abs() < 0.02,
            "backdrop should read 4.0, got {clear}"
        );
        assert!(
            through_glass > 0.85 * clear,
            "glass passed only {through_glass} of {clear}"
        );
    }

    #[test]
    fn a_transparent_background_reaches_the_film() {
        let Some(mut gpu) = backend() else { return };
        let mut scene = Scene::new();
        let settings = RaytraceSettings::default()
            .with_samples(4)
            .with_denoise(false)
            .with_background(BackgroundMode::Transparent);
        let rt = RaytraceScene::build(&mut scene, &settings);
        let cam = RtCamera::new(&camera(), &settings);
        let mut film = Film::new(8, 8);
        gpu.render(&rt, &cam, &settings, &mut film, 0, 4).unwrap();
        assert_eq!(film.resolve_hdr()[3], 0.0);
    }

    /// A checkerboard albedo map, which only reaches the kernel through the
    /// texture atlas — sub-rectangle addressing, per-axis wrapping and manual
    /// bilinear filtering all at once.
    #[test]
    fn gpu_matches_cpu_on_a_textured_surface() {
        let Some(mut gpu) = backend() else { return };
        let build = || {
            let mut scene = Scene::new();
            scene.background = Color::BLACK;
            let n = 8u32;
            let mut px = vec![0u8; (n * n * 4) as usize];
            for y in 0..n {
                for x in 0..n {
                    let on = (x + y) % 2 == 0;
                    let i = ((y * n + x) * 4) as usize;
                    px[i] = if on { 230 } else { 30 };
                    px[i + 1] = if on { 60 } else { 200 };
                    px[i + 2] = 90;
                    px[i + 3] = 255;
                }
            }
            let mut tex = crate::textures::Texture::new(
                n,
                n,
                crate::textures::TextureFormat::Rgba8UnormSrgb,
                px,
            );
            tex.wrap_s = crate::textures::TextureWrap::Repeat;
            tex.wrap_t = crate::textures::TextureWrap::Repeat;
            tex.mag_filter = crate::textures::TextureFilter::Linear;
            tex.repeat = crate::math::Vector2::new(3.0, 3.0);
            tex.flip_y = false;
            let mut m = StandardMaterial::new(Color::WHITE);
            m.map = Some(std::sync::Arc::new(tex));
            m.roughness = 0.8;
            scene.add(Object3D::mesh(Mesh::new(
                PlaneGeometry::new(6.0, 6.0),
                Material::Standard(m),
            )));
            scene.add_light(AmbientLight::new(Color::WHITE, 1.0));
            scene
        };
        let settings = RaytraceSettings {
            samples_per_pixel: 64,
            max_bounces: 2,
            denoise: false,
            ..Default::default()
        };
        let mut cam = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
        cam.position = Vector3::new(0.0, 0.0, 5.0);
        cam.target = Vector3::ZERO;

        let mut scene = build();
        let rt = RaytraceScene::build(&mut scene, &settings);
        assert!(
            rt.shadows_all_opaque,
            "an opaque map must keep the fast path"
        );
        let rtc = RtCamera::new(&cam, &settings);

        let mut g = Film::new(48, 48);
        gpu.render(&rt, &rtc, &settings, &mut g, 0, 64).unwrap();
        let mut c = Film::new(48, 48);
        CpuBackend::new()
            .render(&rt, &rtc, &settings, &mut c, 0, 64)
            .unwrap();

        let (gh, ch) = (g.resolve_hdr(), c.resolve_hdr());
        for channel in 0..3 {
            let (gm, cm) = (mean_channel(&gh, channel), mean_channel(&ch, channel));
            assert!(gm > 0.01, "channel {channel} came out black on the GPU");
            assert!(
                (gm - cm).abs() < 0.03 * cm.max(0.05),
                "channel {channel}: gpu {gm}, cpu {cm} — the atlas is not sampling like the CPU"
            );
        }
        // The checker has to survive: a constant image would mean the atlas
        // returned one texel, or white.
        let reds: Vec<f32> = gh.chunks_exact(4).map(|p| p[0]).collect();
        let spread = reds.iter().cloned().fold(0.0f32, f32::max)
            - reds.iter().cloned().fold(f32::MAX, f32::min);
        assert!(
            spread > 0.15,
            "the texture flattened to a constant ({spread})"
        );
    }

    /// An environment map, which reaches the kernel through the same atlas plus
    /// the cube direction mapping. The failure this guards is loud: an
    /// environment that does not pack used to sample as full white.
    #[test]
    fn gpu_matches_cpu_with_an_environment_map() {
        let Some(mut gpu) = backend() else { return };
        let size = 8u32;
        let mut faces: [Vec<u8>; 6] = Default::default();
        // A distinct constant per face, so a mis-mapped direction shows up as a
        // wrong colour rather than as noise.
        let tints: [[u8; 3]; 6] = [
            [220, 40, 40],
            [40, 220, 40],
            [40, 40, 220],
            [220, 220, 40],
            [220, 40, 220],
            [40, 220, 220],
        ];
        for (f, tint) in faces.iter_mut().zip(tints) {
            let mut px = vec![255u8; (size * size * 4) as usize];
            for p in px.chunks_exact_mut(4) {
                p[0] = tint[0];
                p[1] = tint[1];
                p[2] = tint[2];
            }
            *f = px;
        }
        let cube = std::sync::Arc::new(crate::textures::CubeTexture::new(
            size,
            crate::textures::TextureFormat::Rgba8Unorm,
            faces,
        ));

        let mut scene = Scene::new();
        scene.background = Color::BLACK;
        scene.environment = Some(cube);
        scene.add(Object3D::mesh(Mesh::new(
            SphereGeometry::new(1.2, 32, 24),
            Material::Standard(StandardMaterial::new(Color::new(0.8, 0.8, 0.8))),
        )));

        let settings = RaytraceSettings {
            samples_per_pixel: 96,
            max_bounces: 2,
            denoise: false,
            background: BackgroundMode::Environment,
            ..Default::default()
        };
        let rt = RaytraceScene::build(&mut scene, &settings);
        assert!(rt.world.has_environment(), "the cube should have decoded");
        let mut cam = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
        cam.position = Vector3::new(0.0, 0.0, 4.0);
        cam.target = Vector3::ZERO;
        let rtc = RtCamera::new(&cam, &settings);

        let mut g = Film::new(40, 40);
        gpu.render(&rt, &rtc, &settings, &mut g, 0, 192).unwrap();
        let mut c = Film::new(40, 40);
        CpuBackend::new()
            .render(&rt, &rtc, &settings, &mut c, 0, 96)
            .unwrap();

        let (gh, ch) = (g.resolve_hdr(), c.resolve_hdr());
        for channel in 0..3 {
            let (gm, cm) = (mean_channel(&gh, channel), mean_channel(&ch, channel));
            assert!(
                (gm - cm).abs() < 0.04 * cm.max(0.05),
                "channel {channel}: gpu {gm}, cpu {cm} — cube mapping disagrees"
            );
            // White would be the symptom of an environment that failed to pack.
            assert!(
                gm < 0.95,
                "channel {channel} reads {gm}: suspiciously white"
            );
        }
    }

    /// The kernel has to importance-sample the environment the same way the CPU
    /// does. A sun this small is found about once in eight hundred cosine-
    /// weighted samples, so a kernel that skipped the distribution would come
    /// back visibly darker at this budget and the means would not match.
    #[test]
    fn gpu_importance_samples_the_environment() {
        let Some(mut gpu) = backend() else { return };
        use crate::raytrace::backend::sun::{sun_scene, sun_settings};

        let mut cam = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
        cam.position = Vector3::new(0.0, 0.0, 5.0);
        cam.target = Vector3::ZERO;

        let settings = sun_settings(64);
        let mut scene = sun_scene();
        let rt = RaytraceScene::build(&mut scene, &settings);
        assert!(rt.world.env_distribution().is_some());
        let rtc = RtCamera::new(&cam, &settings);

        let mut g = Film::new(24, 24);
        gpu.render(&rt, &rtc, &settings, &mut g, 0, 64).unwrap();
        let mut c = Film::new(24, 24);
        CpuBackend::new()
            .render(&rt, &rtc, &settings, &mut c, 0, 64)
            .unwrap();

        let (gm, cm) = (
            mean_channel(&g.resolve_hdr(), 0),
            mean_channel(&c.resolve_hdr(), 0),
        );
        assert!(cm > 0.02, "the reference is not lit ({cm})");
        assert!(
            (gm - cm).abs() < 0.08 * cm,
            "gpu {gm} vs cpu {cm} — the kernel is not sampling the sun"
        );
    }

    /// The kernel's stopping rule has to be the CPU's stopping rule, or the two
    /// backends spend different budgets and diverge as they converge.
    #[test]
    fn gpu_adaptive_sampling_matches_the_cpu() {
        let Some(mut gpu) = backend() else { return };
        let mut scene = lit_box_scene();
        let settings = RaytraceSettings {
            samples_per_pixel: 128,
            max_bounces: 3,
            adaptive_threshold: 0.02,
            adaptive_min_samples: 16,
            denoise: false,
            ..Default::default()
        };
        let rt = RaytraceScene::build(&mut scene, &settings);
        let cam = RtCamera::new(&camera(), &settings);

        let mut g = Film::new(32, 32);
        gpu.render(&rt, &cam, &settings, &mut g, 0, 128).unwrap();
        let mut c = Film::new(32, 32);
        CpuBackend::new()
            .render(&rt, &cam, &settings, &mut c, 0, 128)
            .unwrap();

        let (g_lo, g_hi) = g.sample_range();
        let (c_lo, c_hi) = c.sample_range();
        assert!(g_lo < 128, "the kernel stopped nothing ({g_lo}..{g_hi})");
        assert!(c_lo < 128, "the CPU stopped nothing ({c_lo}..{c_hi})");

        let total = |f: &Film| {
            f.resolve_sample_counts()
                .iter()
                .map(|&n| n as f64)
                .sum::<f64>()
        };
        let (gt, ct) = (total(&g), total(&c));
        assert!(
            (gt - ct).abs() < 0.2 * ct,
            "budgets diverged: gpu {gt}, cpu {ct}"
        );

        let gm = mean_channel(&g.resolve_hdr(), 0);
        let cm = mean_channel(&c.resolve_hdr(), 0);
        assert!(
            (gm - cm).abs() < 0.05 * cm.max(0.02),
            "gpu {gm} vs cpu {cm}"
        );
    }

    /// Sample redistribution within an 8×8 workgroup must converge at least
    /// as many pixels as uniform adaptive sampling on the same mixed scene.
    #[test]
    fn gpu_sample_redistribution_improves_convergence() {
        let Some(mut gpu) = backend() else { return };
        let build = || {
            let mut scene = Scene::new();
            scene.background = Color::new(0.05, 0.05, 0.06);
            scene.add(Object3D::mesh(crate::core::Mesh::new(
                PlaneGeometry::new(20.0, 20.0),
                Material::Standard(StandardMaterial::new(Color::new(0.75, 0.75, 0.75))),
            )));
            scene.add(Object3D::mesh(crate::core::Mesh::new(
                BoxGeometry::new(0.4, 0.4, 0.4),
                Material::Standard(StandardMaterial::new(Color::new(0.9, 0.1, 0.1))),
            )));
            scene.add_light(AmbientLight::new(Color::WHITE, 0.4));
            scene
        };
        let mut render = |redistribute: bool| {
            let settings = RaytraceSettings {
                samples_per_pixel: 64,
                max_bounces: 3,
                adaptive_threshold: 0.05,
                adaptive_min_samples: 4,
                sample_redistribution: redistribute,
                denoise: false,
                ..Default::default()
            };
            let mut scene = build();
            let rt = RaytraceScene::build(&mut scene, &settings);
            let cam = RtCamera::new(&camera(), &settings);
            let mut film = Film::new(32, 32);
            gpu.render(&rt, &cam, &settings, &mut film, 0, 64).unwrap();
            let converged = film
                .pixels()
                .iter()
                .filter(|p| {
                    p.is_converged(settings.adaptive_threshold, settings.adaptive_min_samples)
                })
                .count();
            converged as f32 / film.pixels().len() as f32
        };
        let (conv_off, conv_on) = (render(false), render(true));
        assert!(
            conv_on >= conv_off,
            "gpu redistribution: {conv_on} vs {conv_off} converged fraction"
        );
    }

    /// Welford M₂ on the GPU must track the CPU or adaptive budgets diverge.
    #[test]
    fn gpu_welford_variance_tracks_the_cpu() {
        let Some(mut gpu) = backend() else { return };
        let mut scene = lit_box_scene();
        let settings = RaytraceSettings {
            samples_per_pixel: 64,
            max_bounces: 3,
            adaptive_threshold: 0.02,
            adaptive_min_samples: 8,
            denoise: false,
            ..Default::default()
        };
        let rt = RaytraceScene::build(&mut scene, &settings);
        let cam = RtCamera::new(&camera(), &settings);

        let mut g = Film::new(24, 24);
        gpu.render(&rt, &cam, &settings, &mut g, 0, 64).unwrap();
        let mut c = Film::new(24, 24);
        CpuBackend::new()
            .render(&rt, &cam, &settings, &mut c, 0, 64)
            .unwrap();

        let gv: f64 = g.resolve_variance().iter().filter(|v| v.is_finite()).map(|v| *v as f64).sum();
        let cv: f64 = c.resolve_variance().iter().filter(|v| v.is_finite()).map(|v| *v as f64).sum();
        assert!(gv > 0.0 && cv > 0.0);
        assert!(
            (gv - cv).abs() < 0.35 * cv.max(gv),
            "variance totals diverged: gpu {gv}, cpu {cv}"
        );
    }

    /// Both backends have to map a direction to the same cube texel, on every
    /// face.
    ///
    /// Rendered with nothing in the scene and the environment as the
    /// background, so a camera ray goes straight to the sky and back: no
    /// scattering, no light sampling, almost no noise, and the comparison is
    /// per pixel rather than per tile. Six camera directions cover all six
    /// faces — an object in the scene only ever exposes the two or three faces
    /// it happens to face, which is how a mis-mapped face survives an
    /// end-to-end test.
    #[test]
    fn gpu_and_cpu_map_the_cube_the_same_way() {
        let Some(mut gpu) = backend() else { return };
        // A sky that encodes the *direction* at each texel, as `0.5 + 0.5 * d`.
        //
        // Not longitude and latitude: those are discontinuous — longitude at
        // the ±pi seam, which is exactly where a frame pointed at -X sits, and
        // both of them at the poles, which is where the ±Y frames sit. A
        // discontinuity there makes a sub-pixel difference in jitter read as a
        // large difference in value, and the test fails on its own encoding
        // rather than on anything the renderer did. The direction itself is
        // smooth over the whole sphere and still identifies every texel.
        let (w, h) = (256u32, 128u32);
        let mut src = vec![0.0f32; (w * h * 4) as usize];
        for y in 0..h {
            let theta = (y as f32 + 0.5) / h as f32 * std::f32::consts::PI;
            for x in 0..w {
                let phi = ((x as f32 + 0.5) / w as f32 - 0.5) * 2.0 * std::f32::consts::PI;
                let d = Vector3::new(
                    theta.sin() * phi.cos(),
                    theta.cos(),
                    theta.sin() * phi.sin(),
                );
                let i = ((y * w + x) * 4) as usize;
                src[i] = 0.5 + 0.5 * d.x;
                src[i + 1] = 0.5 + 0.5 * d.y;
                src[i + 2] = 0.5 + 0.5 * d.z;
                src[i + 3] = 1.0;
            }
        }
        let cube = crate::extras::PmremGenerator::from_equirect_f32(&src, w, h, 128);
        let mut scene = Scene::new();
        scene.environment = Some(std::sync::Arc::new(cube));

        let settings = RaytraceSettings {
            samples_per_pixel: 4,
            max_bounces: 0,
            background: BackgroundMode::Environment,
            adaptive_threshold: 0.0,
            denoise: false,
            ..Default::default()
        };
        let rt = RaytraceScene::build(&mut scene, &settings);

        for (i, target) in [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(-1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, -1.0),
        ]
        .into_iter()
        .enumerate()
        {
            let mut cam = PerspectiveCamera::new(60.0, 1.0, 0.1, 100.0);
            cam.position = Vector3::ZERO;
            cam.target = target;
            // Looking straight up or down, the default up-vector is degenerate.
            cam.up = if target.y.abs() > 0.9 {
                Vector3::new(0.0, 0.0, 1.0)
            } else {
                Vector3::new(0.0, 1.0, 0.0)
            };
            let rtc = RtCamera::new(&cam, &settings);

            let mut g = Film::new(24, 24);
            gpu.render(&rt, &rtc, &settings, &mut g, 0, 4).unwrap();
            let mut c = Film::new(24, 24);
            CpuBackend::new()
                .render(&rt, &rtc, &settings, &mut c, 0, 4)
                .unwrap();

            let (gh, ch) = (g.resolve_hdr(), c.resolve_hdr());
            let mut worst = 0.0f32;
            for (a, b) in gh.chunks_exact(4).zip(ch.chunks_exact(4)) {
                for k in 0..3 {
                    worst = worst.max((a[k] - b[k]).abs());
                }
            }
            assert!(
                worst < 0.03,
                "looking at {target:?} (face {i}): worst channel difference {worst} \
                 — the two backends disagree about where the sky is"
            );
        }
    }

    /// An HDR environment with a *directional* feature in it, so a within-face
    /// rotation cannot hide.
    ///
    /// The per-face-constant test above cannot see an orientation error at all:
    /// rotating a constant face leaves it constant. This one encodes longitude
    /// and latitude into the sky and puts a bright sun in it, so the kernel and
    /// the CPU have to agree on *where* the sky is, not just which face it is
    /// on — and on its dynamic range, which only survives if both read the HDR
    /// faces rather than the tone-mapped display copy.
    #[test]
    fn gpu_matches_cpu_on_an_hdr_environment() {
        let Some(mut gpu) = backend() else { return };
        let (w, h) = (256u32, 128u32);
        let sun = Vector3::new(0.55, 0.62, 0.56).normalize();
        let mut src = vec![0.0f32; (w * h * 4) as usize];
        for y in 0..h {
            let v = (y as f32 + 0.5) / h as f32;
            let theta = v * std::f32::consts::PI;
            for x in 0..w {
                let u = (x as f32 + 0.5) / w as f32;
                let phi = (u - 0.5) * 2.0 * std::f32::consts::PI;
                let d = Vector3::new(
                    theta.sin() * phi.cos(),
                    theta.cos(),
                    theta.sin() * phi.sin(),
                );
                let i = ((y * w + x) * 4) as usize;
                let c = if d.dot(sun) > 0.06f32.cos() {
                    [400.0, 380.0, 350.0]
                } else {
                    // A gradient, so the two backends must agree on direction.
                    [0.2 + 0.6 * u, 0.25 + 0.5 * v, 0.6]
                };
                src[i] = c[0];
                src[i + 1] = c[1];
                src[i + 2] = c[2];
                src[i + 3] = 1.0;
            }
        }
        let cube = crate::extras::PmremGenerator::from_equirect_f32(&src, w, h, 64);

        let mut scene = Scene::new();
        scene.environment = Some(std::sync::Arc::new(cube));
        // A near-mirror, not a diffuse ball. A diffuse surface *integrates* the
        // sky over its whole hemisphere, which washes a rearranged face out of
        // the result — the test would then pass with the cube mapping wrong. A
        // sharp metal images the sky instead, so where it is matters. Roughness
        // is 0.08 rather than 0 so the lobe stays non-delta and next-event
        // estimation still runs.
        let mut m = StandardMaterial::new(Color::new(0.95, 0.95, 0.95));
        m.roughness = 0.08;
        m.metalness = 1.0;
        scene.add(Object3D::mesh(Mesh::new(
            SphereGeometry::new(1.2, 48, 32),
            Material::Standard(m),
        )));

        let settings = RaytraceSettings {
            samples_per_pixel: 192,
            max_bounces: 2,
            clamp_indirect: 0.0,
            background: BackgroundMode::Environment,
            // Adaptive stopping would give the two backends different budgets
            // per pixel, which is a second source of difference on top of the
            // one being measured.
            adaptive_threshold: 0.0,
            denoise: false,
            ..Default::default()
        };
        let rt = RaytraceScene::build(&mut scene, &settings);
        assert!(rt.world.env_distribution().is_some());
        // The HDR faces have to have survived the trip.
        let peak = rt.world.lighting_radiance(sun);
        assert!(
            peak.x > 300.0,
            "the sun read back at {} — clamped somewhere",
            peak.x
        );

        // Two viewpoints, because one only shows part of the cube: the
        // background is whatever face the camera looks at and the reflection is
        // whatever the sphere faces. A single angle leaves two faces untested,
        // and a mapping error on those would go unseen.
        for (i, eye) in [Vector3::new(0.0, 0.0, 4.5), Vector3::new(4.5, 0.6, 0.0)]
            .into_iter()
            .enumerate()
        {
            let mut cam = PerspectiveCamera::new(70.0, 1.0, 0.1, 100.0);
            cam.position = eye;
            cam.target = Vector3::ZERO;
            let rtc = RtCamera::new(&cam, &settings);

            let mut g = Film::new(40, 40);
            gpu.render(&rt, &rtc, &settings, &mut g, 0, 192).unwrap();
            let mut c = Film::new(40, 40);
            CpuBackend::new()
                .render(&rt, &rtc, &settings, &mut c, 0, 192)
                .unwrap();

            let (gh, ch) = (g.resolve_hdr(), c.resolve_hdr());
            for channel in 0..3 {
                let (gm, cm) = (mean_channel(&gh, channel), mean_channel(&ch, channel));
                assert!(cm > 0.05, "view {i} channel {channel} is not lit ({cm})");
                assert!(
                    (gm - cm).abs() < 0.05 * cm,
                    "view {i} channel {channel}: gpu {gm}, cpu {cm} — the two disagree about the sky"
                );
            }

            // Spatially, not just on average: a rotated face moves the sky
            // around without changing its total much. Compared as 4x4 tile
            // means over all three channels — red encodes longitude and green
            // latitude, so one channel alone leaves an axis untested — because
            // a per-pixel comparison at this sample count is dominated by the
            // two backends' different random sequences where the sun reflects.
            let tile_means = |v: &[f32], channel: usize| {
                let mut tiles = [[0.0f64; 4]; 4];
                let mut counts = [[0u32; 4]; 4];
                for (p, px) in v.chunks_exact(4).enumerate() {
                    let (x, y) = (p % 40, p / 40);
                    let (tx, ty) = (x * 4 / 40, y * 4 / 40);
                    tiles[ty][tx] += px[channel] as f64;
                    counts[ty][tx] += 1;
                }
                let mut out = [[0.0f64; 4]; 4];
                for y in 0..4 {
                    for x in 0..4 {
                        out[y][x] = tiles[y][x] / counts[y][x].max(1) as f64;
                    }
                }
                out
            };
            for channel in 0..3 {
                let (gt, ct) = (tile_means(&gh, channel), tile_means(&ch, channel));
                for y in 0..4 {
                    for x in 0..4 {
                        assert!(
                            // Loose enough for the Monte-Carlo difference
                            // between two different random sequences where a
                            // 400:1 sun reflects; a mis-mapped face moves a
                            // tile by a third, which is well clear of it.
                            (gt[y][x] - ct[y][x]).abs() < 0.18 * ct[y][x].max(0.05),
                            "view {i} channel {channel}, tile ({x},{y}): gpu {:.4} vs cpu {:.4} \
                             — the sky is in different places",
                            gt[y][x],
                            ct[y][x]
                        );
                    }
                }
            }
        }
    }

    /// The kernel's sampler has to be stratified too, or the GPU quietly needs
    /// several times the samples the CPU does for the same image.
    #[test]
    fn gpu_converges_as_fast_as_the_cpu_at_low_sample_counts() {
        let Some(mut gpu) = backend() else { return };
        use crate::raytrace::backend::sun::{
            area_light_camera, area_light_scene, area_light_settings,
        };

        let reference = {
            let settings = area_light_settings(1024);
            let mut scene = area_light_scene();
            let rt = RaytraceScene::build(&mut scene, &settings);
            let rtc = RtCamera::new(&area_light_camera(), &settings);
            let mut film = Film::new(24, 24);
            CpuBackend::new()
                .render(&rt, &rtc, &settings, &mut film, 0, 1024)
                .unwrap();
            film.resolve_hdr()
        };

        let settings = area_light_settings(16);
        let mut scene = area_light_scene();
        let rt = RaytraceScene::build(&mut scene, &settings);
        let rtc = RtCamera::new(&area_light_camera(), &settings);
        let mut film = Film::new(24, 24);
        gpu.render(&rt, &rtc, &settings, &mut film, 0, 16).unwrap();
        let quick = film.resolve_hdr();

        let mut sse = 0.0f64;
        let mut mean = 0.0f64;
        let mut n = 0usize;
        for (a, b) in quick.chunks_exact(4).zip(reference.chunks_exact(4)) {
            for k in 0..3 {
                let d = (a[k] - b[k]) as f64;
                sse += d * d;
                mean += b[k] as f64;
                n += 1;
            }
        }
        let rms = (sse / n as f64).sqrt();
        let mean = mean / n as f64;
        assert!(mean > 0.05, "the scene is not lit ({mean})");
        assert!(
            rms < 0.048 * mean,
            "16 GPU samples are {:.1}% off converged — the kernel's sampler is not stratified",
            100.0 * rms / mean
        );
    }

    /// The guide passes are computed in both backends and consumed by the
    /// denoiser, so they have to agree. They are also the channels a caller
    /// would hand to an external denoiser, which makes a silent divergence
    /// between the two especially unhelpful.
    #[test]
    fn gpu_and_cpu_agree_on_the_denoiser_guides() {
        let Some(mut gpu) = backend() else { return };
        let mut scene = lit_box_scene();
        // A mirror, so the guides have to follow a specular bounce rather than
        // stopping at the first hit — the case the two could disagree on.
        let mut mirror = StandardMaterial::new(Color::new(0.9, 0.9, 0.95));
        mirror.roughness = 0.03;
        mirror.metalness = 1.0;
        let mut ball = Object3D::mesh(Mesh::new(
            SphereGeometry::new(0.9, 32, 24),
            Material::Standard(mirror),
        ));
        ball.position = Vector3::new(0.0, 0.9, 1.5);
        scene.add(ball);

        let settings = RaytraceSettings {
            samples_per_pixel: 64,
            max_bounces: 3,
            adaptive_threshold: 0.0,
            denoise: false,
            ..Default::default()
        };
        let rt = RaytraceScene::build(&mut scene, &settings);
        let cam = RtCamera::new(&camera(), &settings);

        let mut g = Film::new(32, 32);
        gpu.render(&rt, &cam, &settings, &mut g, 0, 64).unwrap();
        let mut c = Film::new(32, 32);
        CpuBackend::new()
            .render(&rt, &cam, &settings, &mut c, 0, 64)
            .unwrap();

        let (ga, ca) = (g.resolve_albedo(), c.resolve_albedo());
        let mean =
            |v: &[[f32; 3]]| v.iter().map(|p| p[0] as f64).sum::<f64>() / v.len().max(1) as f64;
        let (gm, cm) = (mean(&ga), mean(&ca));
        assert!(cm > 0.02, "the albedo guide is empty ({cm})");
        assert!(
            (gm - cm).abs() < 0.06 * cm,
            "albedo guide: gpu {gm:.4} vs cpu {cm:.4}"
        );

        // Normals are unit vectors, so their mean length says whether the two
        // backends agree about direction and not merely about magnitude.
        let (gn, cn) = (g.resolve_normal(), c.resolve_normal());
        let mut worst = 0.0f32;
        for (a, b) in gn.iter().zip(cn.iter()) {
            let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            // Both zero is agreement; otherwise they must point the same way.
            let len_a = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
            let len_b = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
            if len_a > 0.1 && len_b > 0.1 {
                worst = worst.max(1.0 - dot);
            }
        }
        assert!(worst < 0.25, "normal guides diverge by {worst} (1 - cos)");
    }

    #[test]
    fn the_renderer_drives_the_gpu_backend() {
        let Some(gpu) = backend() else { return };
        let mut scene = lit_box_scene();
        let cam = camera();
        let mut r = super::super::RaytraceRenderer::with_backend(32, 32, Box::new(gpu));
        r.set_settings(
            RaytraceSettings::default()
                .with_samples(8)
                .with_denoise(false),
        );
        assert_eq!(r.backend_name(), "gpu");
        let rgba = r.render_to_rgba(&mut scene, &cam);
        assert_eq!(rgba.len(), 32 * 32 * 4);
        assert!(rgba.chunks_exact(4).any(|p| p[0] > 8), "image is black");
    }

    /// Reproducibility on the GPU: batching must not change which random
    /// sequence each pixel sees, or a progressive render flickers.
    #[test]
    fn gpu_batching_does_not_change_the_result() {
        let Some(mut one_gpu) = backend() else { return };
        let Some(mut split_gpu) = backend() else { return };
        let mut scene = lit_box_scene();
        let settings = RaytraceSettings {
            samples_per_pixel: 8,
            max_bounces: 3,
            denoise: false,
            adaptive_threshold: 0.0,
            ..Default::default()
        };
        let cam = camera();
        let rt = super::super::RaytraceScene::build(&mut scene, &settings);
        let rtc = super::super::RtCamera::new(&cam, &settings);

        let mut one = super::super::Film::new(12, 12);
        one_gpu
            .render(&rt, &rtc, &settings, &mut one, 0, 8)
            .unwrap();

        let mut split = super::super::Film::new(12, 12);
        split_gpu = split_gpu.with_samples_per_dispatch(3);
        split_gpu
            .render(&rt, &rtc, &settings, &mut split, 0, 3)
            .unwrap();
        split_gpu
            .render(&rt, &rtc, &settings, &mut split, 3, 5)
            .unwrap();

        let a = one.resolve_hdr();
        let b = split.resolve_hdr();
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < 1e-5, "pixel value {i}: {x} vs {y}");
        }
    }

    #[test]
    fn gpu_render_rect_only_updates_the_requested_region() {
        let Some(mut gpu) = backend() else { return };
        let mut scene = Scene::new();
        scene.background = Color::new(0.2, 0.3, 0.4);
        let settings = RaytraceSettings {
            samples_per_pixel: 4,
            adaptive_threshold: 0.0,
            sample_redistribution: false,
            denoise: false,
            ..Default::default()
        };
        let rt = RaytraceScene::build(&mut scene, &settings);
        let cam = RtCamera::new(&camera(), &settings);
        let mut film = Film::new(16, 16);
        gpu.render_rect(
            &rt,
            &cam,
            &settings,
            &mut film,
            0,
            4,
            super::super::RenderRect {
                x: 2,
                y: 3,
                width: 4,
                height: 2,
            },
        )
        .unwrap();
        gpu.sync_film(&mut film).unwrap();
        let outside = 0;
        let inside = (3 * 16 + 2) as usize;
        assert_eq!(film.pixels()[outside].samples, 0);
        assert_eq!(film.pixels()[inside].samples, 4);
    }

    #[test]
    fn gpu_batched_regions_trace_every_requested_pixel() {
        let Some(mut gpu) = backend() else { return };
        let mut scene = Scene::new();
        scene.background = Color::new(0.2, 0.3, 0.4);
        let settings = RaytraceSettings {
            samples_per_pixel: 4,
            adaptive_threshold: 0.0,
            sample_redistribution: false,
            denoise: false,
            ..Default::default()
        };
        let rt = RaytraceScene::build(&mut scene, &settings);
        let cam = RtCamera::new(&camera(), &settings);
        let rects = [
            super::super::RenderRect {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
            },
            super::super::RenderRect {
                x: 8,
                y: 8,
                width: 8,
                height: 8,
            },
        ];
        let mut film = Film::new(16, 16);
        gpu.render_regions(&rt, &cam, &settings, &mut film, 0, 4, &rects)
            .unwrap();
        gpu.sync_film(&mut film).unwrap();
        assert_eq!(film.samples(), 4);
        assert_eq!(film.pixels()[0].samples, 4);
        assert_eq!(film.pixels()[15 * 16 + 15].samples, 4);
        assert_eq!(film.pixels()[8].samples, 0);
    }

    #[test]
    fn caps_respect_downlevel_defaults() {
        let limits = wgpu::Limits::downlevel_defaults();
        let caps = GpuCaps::from_limits(&limits);
        assert!(caps.atlas_side <= limits.max_texture_dimension_2d);
        assert!(caps.atlas_side <= ATLAS_SIZE);
        assert!(caps.max_accum_pixels > 0);
        caps.validate_kernel().expect("downlevel should fit uniforms");
    }

    #[test]
    fn caps_check_film_rejects_oversized_accum() {
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: ACCUM_STRIDE as u64 * 4 * 1024,
            max_buffer_size: ACCUM_STRIDE as u64 * 4 * 1024,
            ..wgpu::Limits::downlevel_defaults()
        };
        let caps = GpuCaps::from_limits(&limits);
        assert!(caps.check_film(64, 64).is_err());
        assert!(caps.check_film(32, 32).is_ok());
    }

    #[test]
    fn clamp_film_size_fits_accum_and_preserves_aspect() {
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: ACCUM_STRIDE as u64 * 4 * 4096,
            max_buffer_size: ACCUM_STRIDE as u64 * 4 * 4096,
            ..wgpu::Limits::downlevel_defaults()
        };
        let caps = GpuCaps::from_limits(&limits);
        let (w, h) = caps.clamp_film_size(8192, 4096);
        assert!(caps.check_film(w, h).is_ok());
        assert!((w as f64 / h as f64 - 2.0).abs() < 0.05);
    }

    #[test]
    fn reset_accum_keeps_scene_on_device() {
        let Some(mut gpu) = backend() else {
            return;
        };
        let mut scene = lit_box_scene();
        let flat = RaytraceScene::build(&mut scene, &RaytraceSettings::default());
        let cam = RtCamera::new(&camera(), &RaytraceSettings::default());
        let mut film = Film::new(16, 16);
        gpu.render(&flat, &cam, &RaytraceSettings::default(), &mut film, 0, 1)
            .unwrap();
        assert!(gpu.scene_on_device());
        gpu.reset_accum();
        assert!(gpu.scene_on_device());
        gpu.invalidate();
        assert!(!gpu.scene_on_device());
    }

    #[test]
    fn renderer_clamps_oversized_gpu_film() {
        let Some(gpu) = backend() else { return };
        let caps = gpu.caps();
        let mut r = super::super::RaytraceRenderer::with_backend(1, 1, Box::new(gpu));
        let huge = caps.max_film_side().saturating_mul(4).max(4096);
        let (w, h) = r.set_size(huge, huge / 2);
        assert!(r.check_film_size(w, h).is_ok());
        assert!(w <= caps.max_film_side());
    }
}
