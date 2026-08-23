//! Buoyancy read back from the GPU's own surface.
//!
//! [`OceanFft::sample`](crate::ocean_fft::OceanFft::sample) answers "where is
//! the water" on the CPU from the 96 strongest modes of the swell cascade. That
//! is fast and it is right about the long waves that move a hull — but it is not
//! the surface that was *drawn*. It has no mid or fine cascade in it, and, more
//! importantly, no **shoaling**: run a boat into the shallows and the CPU
//! answer keeps reporting deep-water heights while the rendered wave beneath it
//! has grown, steepened and broken.
//!
//! This closes that gap. A compute pass evaluates the same `sample_surface` the
//! mesh and the pixels use, at a handful of query points, and copies the answers
//! back. What comes out is the surface as rendered, shoaling and all.
//!
//! # The latency, and why it is fine
//!
//! A readback cannot be synchronous — not on the web, where there is no way to
//! block, and not usefully on native either. So this is a pipeline: points go in
//! this frame, answers come out one or two frames later, and callers use the
//! most recent answer they have. At 60 Hz that is a lag of tens of milliseconds
//! on a hull that takes seconds to rise over a swell, which is invisible. What
//! is *not* invisible is a hull sitting a metre inside a breaking wave, which is
//! what the CPU approximation does in the surf.
//!
//! Exactly one readback is ever in flight. If the answer for a frame is not back
//! yet, the next frame simply does not queue another — no growing chain of
//! staging buffers, and no stall.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use wgpu::util::DeviceExt;

use crate::ocean_fft::OceanFft;

/// The most query points a single dispatch answers. One workgroup's worth.
pub const CAPACITY: usize = 64;

const SHADER: &str = r#"
@group(0) @binding(0) var cascade_samp : sampler;
@group(0) @binding(1) var<storage, read>       pts : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out : array<vec4<f32>>;
@group(0) @binding(3) var cascade_a : texture_2d<f32>;
@group(0) @binding(4) var cascade_b : texture_2d<f32>;
@group(0) @binding(5) var cascade_c : texture_2d<f32>;

$CASCADES

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&pts)) { return; }
    // A deliberately coarse footprint: a hull is metres across and does not ride
    // the capillary ripples, so band-limiting here is what stops a buoy
    // twitching at the fine cascade's frequency.
    let s = sample_surface(pts[i].xy, 0.6);
    out[i] = vec4<f32>(s.disp.y, s.normal.x, s.normal.y, s.normal.z);
}
"#;

pub struct Probe {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    points: wgpu::Buffer,
    result: wgpu::Buffer,
    /// Behind an `Arc` because the map callback outlives this call and needs
    /// its own handle; `wgpu::Buffer` is not itself cloneable.
    staging: Arc<wgpu::Buffer>,
    /// True from the moment a readback is queued until its answer lands.
    in_flight: Arc<AtomicBool>,
    /// The most recent answer, and how many of its entries are meaningful.
    latest: Arc<Mutex<(Vec<[f32; 4]>, usize)>>,
    /// Queued this frame, mapped after the submit that carries it.
    queued: Option<usize>,
}

impl Probe {
    pub fn new(device: &wgpu::Device, fft: &OceanFft, peak_wavelength: f32) -> Probe {
        let source = SHADER.replace(
            "$CASCADES",
            &crate::waves_gpu::cascade_wgsl(
                "cascade_a",
                "cascade_b",
                "cascade_c",
                "cascade_samp",
                peak_wavelength,
                &fft.cas,
            ),
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ocean probe"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let bytes = (CAPACITY * 16) as u64;
        let points = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ocean probe points"),
            contents: bytemuck::cast_slice(&[[0.0f32; 4]; CAPACITY]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let result = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ocean probe result"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ocean probe staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ocean probe sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let tex = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let buf = |b: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ocean probe bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                buf(1, true),
                buf(2, false),
                tex(3),
                tex(4),
                tex(5),
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ocean probe bg"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: points.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: result.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&fft.cascade_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&fft.cascade_views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&fft.cascade_views[2]),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ocean probe pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ocean probe pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Probe {
            pipeline,
            bind_group,
            points,
            result,
            staging,
            in_flight: Arc::new(AtomicBool::new(false)),
            latest: Arc::new(Mutex::new((vec![[0.0; 4]; CAPACITY], 0))),
            queued: None,
        }
    }

    /// Queue one dispatch, if the previous answer is already back.
    ///
    /// Encodes into the caller's encoder, so this costs no extra submission —
    /// it rides the same one every other compute pass in the frame does.
    pub fn dispatch(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        points: &[[f32; 2]],
    ) {
        if self.in_flight.load(Ordering::Acquire) || points.is_empty() {
            return;
        }
        let n = points.len().min(CAPACITY);
        let mut packed = [[0.0f32; 4]; CAPACITY];
        for (dst, src) in packed.iter_mut().zip(points) {
            *dst = [src[0], src[1], 0.0, 0.0];
        }
        queue.write_buffer(&self.points, 0, bytemuck::cast_slice(&packed));

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ocean probe pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.result, 0, &self.staging, 0, (CAPACITY * 16) as u64);
        self.queued = Some(n);
    }

    /// Start the map. Must be called *after* the encoder carrying
    /// [`Probe::dispatch`] has been submitted — mapping a buffer whose copy is
    /// still sitting in an un-submitted encoder would read the previous frame.
    pub fn after_submit(&mut self) {
        let Some(n) = self.queued.take() else {
            return;
        };
        self.in_flight.store(true, Ordering::Release);
        let flag = self.in_flight.clone();
        let out = self.latest.clone();
        let staging = self.staging.clone();
        self.staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |res| {
                if res.is_ok() {
                    {
                        let view = staging.slice(..).get_mapped_range().expect("buffer range is mapped");
                        let data: &[[f32; 4]] = bytemuck::cast_slice(&view);
                        if let Ok(mut slot) = out.lock() {
                            slot.0.copy_from_slice(data);
                            slot.1 = n;
                        }
                    }
                    staging.unmap();
                }
                flag.store(false, Ordering::Release);
            });
    }

    /// The most recent answers: height in `x`, surface normal in `yzw`.
    ///
    /// Empty until the first readback lands, which is the caller's cue to fall
    /// back to the CPU modes.
    pub fn read(&self) -> Vec<[f32; 4]> {
        match self.latest.lock() {
            Ok(slot) => slot.0[..slot.1].to_vec(),
            Err(_) => Vec::new(),
        }
    }
}
