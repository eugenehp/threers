//! Persistent surface state: foam that remembers, and wake behind moving things.
//!
//! Everything else about this ocean is closed-form — evaluate the spectrum at
//! `(x, z, t)` and you have the answer, with no state to carry. Foam is the one
//! thing that genuinely is not. A crest that broke five seconds ago left a patch
//! that is still there, drifting downwind and dissolving, and no function of the
//! current wave field can tell you about it. So this is the one piece that needs
//! memory: a world-anchored texture, advected and decayed each frame.
//!
//! Wake rides in the same texture for the same reason and at the same cost — it
//! is foam with a different source term.
//!
//! Two textures, ping-ponged: WebGPU's core feature set has write-only storage
//! textures, so a pass cannot read and write the same image. One is sampled, the
//! other written, and they swap.

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::ocean_fft::OceanFft;

/// Side of the world square the texture covers, metres.
///
/// Deliberately equal to the swell cascade's tile. The field is world-anchored
/// and *tiles*, which is what stops foam simply ending at its edge — the camera
/// can orbit further out than any fixed box would cover. Tiling is honest here
/// rather than a cheat: the wave field itself repeats on exactly this period, so
/// foam that repeats with it is foam in the right place, not a copy of somewhere
/// else's.
/// (The extent is [`Cascades::extent`](crate::ocean_fft::Cascades::extent), so
/// that it follows the swell when the lattice is reconfigured at runtime.)
///
/// Texels per side. At 1024 m that is ~1 m per texel — coarse for a bubble,
/// ample for a trail, which is the only scale that needs to persist.
pub const RESOLUTION: u32 = 1024;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    extent: f32,
    dt: f32,
    wake_count: u32,
    wind: [f32; 2],
    foam_softness: f32,
    foam_threshold: f32,
    decay: f32,
    _pad: [f32; 3],
}

/// A moving object that drags a wake behind it.
///
/// `velocity` is what turns a stirred patch into a **Kelvin wedge** — see
/// `kelvin_wake` in the shader below. Leave it at zero for something that is
/// bobbing rather than travelling.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WakeSource {
    /// World xz.
    pub position: [f32; 2],
    /// World xz, metres per second.
    pub velocity: [f32; 2],
    /// How hard it is stirring the water.
    pub strength: f32,
    /// Metres.
    pub radius: f32,
    pub _pad: [f32; 2],
}

const SHADER: &str = r#"
struct Params {
    extent         : f32,
    dt             : f32,
    wake_count     : u32,
    wind           : vec2<f32>,
    foam_softness  : f32,
    foam_threshold : f32,
    decay          : f32,
};

struct Wake {
    pos      : vec2<f32>,
    vel      : vec2<f32>,
    strength : f32,
    radius   : f32,
    pad      : vec2<f32>,
};

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var cascade_samp : sampler;
@group(0) @binding(2) var<storage, read> wakes : array<Wake>;
@group(0) @binding(3) var prev : texture_2d<f32>;
@group(0) @binding(5) var next : texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var cascade_a : texture_2d<f32>;
@group(0) @binding(7) var cascade_b : texture_2d<f32>;
@group(0) @binding(8) var cascade_c : texture_2d<f32>;

$CASCADES

const WAKE_G: f32 = 9.81;
const WAKE_TAU: f32 = 6.283185307179586;

// ---- Kelvin wake ---------------------------------------------------------
// A hull pushing water leaves a wedge, not a circle. The half-angle is
// arcsin(1/3) = 19.47 degrees, and the part that surprises people is that it
// does not depend on speed: going faster makes the pattern longer, not wider.
// What speed sets is the wavelength inside it, lambda = 2 pi U^2 / g.
//
// Two wave families ride in that wedge. The *divergent* ones pile up along its
// two arms — the sharp feathered lines you see from a bridge — and the
// *transverse* ones run across it behind the hull. Both spread their energy
// over a widening front, which is the 1/sqrt falloff.
fn kelvin_wake(world: vec2<f32>, src: Wake) -> f32 {
    let rel = world - src.pos;
    let hull = src.strength * (1.0 - smoothstep(0.0, max(src.radius, 0.5), length(rel)));
    let u = length(src.vel);
    // Below a walking pace there is no wedge to speak of — just stirred water.
    if (u < 0.4) { return hull; }

    let fwd = src.vel / u;
    // Distance behind the hull, and across its centreline.
    let along = -dot(rel, fwd);
    let across = abs(dot(rel, vec2<f32>(-fwd.y, fwd.x)));
    if (along < 0.0) { return hull; }

    let lam = clamp(WAKE_TAU * u * u / WAKE_G, 1.5, 400.0);
    let a = max(along, 0.001);
    // Energy spread over a front that widens with distance.
    let decay = src.strength / sqrt(1.0 + a / max(lam, 1.0));

    // tan(19.47 deg): the cusp line where the two families meet, and where the
    // amplitude is largest.
    let arm = a * 0.3536;
    let arm_w = max(lam * 0.18, src.radius * 0.6);
    var w = (1.0 - smoothstep(0.0, arm_w, abs(across - arm))) * decay;

    // Transverse crests: arcs of constant phase trailing the hull, spaced one
    // wavelength apart, and only inside the wedge.
    let inside = 1.0 - smoothstep(arm, arm + arm_w, across);
    let phase = (a - across * across / max(2.0 * a, 0.001)) / lam;
    let crest = 0.5 + 0.5 * cos(WAKE_TAU * phase);
    w = w + inside * decay * 0.5 * crest * crest * crest;

    return max(hull, w);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(next);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let world = (uv - 0.5) * params.extent;

    // Advect: read from where this texel's water came from. Foam floats, so it
    // travels with the wind rather than staying where it broke.
    let drift = params.wind * params.dt / params.extent;
    let prev_state = textureSampleLevel(prev, cascade_samp, uv - drift, 0.0);

    // Decay. Foam is bubbles, and bubbles pop; without this the sea silts up.
    var foam = prev_state.r * exp(-params.decay * params.dt);
    var wake = prev_state.g * exp(-params.decay * 0.55 * params.dt);

    // The same surface the mesh and the pixels read, at a deliberately coarse
    // footprint: persistent foam is what a *large* wave leaves when it breaks.
    // Deposit from every ripple as well and every texel sees breaking at some
    // point in a storm, so the field saturates into a white sheet.
    let s = sample_surface(world, 12.0);

    // Deposit from folding only. The slope term the *surface* also shades with
    // is instantaneous whitecapping, and in a storm it is true over most of the
    // sea most of the time. Folding is the rarer, real event: this patch of
    // water actually broke.
    let source = smoothstep(params.foam_threshold * 1.6,
                            params.foam_threshold * 1.6 + params.foam_softness, s.fold);
    // A rate, not a latch, so a crest has to keep breaking over a patch to
    // whiten it. Steady state is source * rate / decay.
    foam = min(1.0, foam + source * params.dt * 1.0);

    // Wake: a hull drags a trail, so this deposits rather than replaces.
    for (var k: u32 = 0u; k < params.wake_count; k = k + 1u) {
        wake = max(wake, kelvin_wake(world, wakes[k]));
    }

    textureStore(next, vec2<i32>(gid.xy),
                 vec4<f32>(clamp(foam, 0.0, 1.0), clamp(wake, 0.0, 1.0), 0.0, 1.0));
}
"#;

pub struct SurfaceState {
    pipeline: wgpu::ComputePipeline,
    /// One bind group per ping-pong direction.
    binds: [wgpu::BindGroup; 2],
    views: [Arc<wgpu::TextureView>; 2],
    params: wgpu::Buffer,
    wakes: wgpu::Buffer,
    flip: usize,
    groups: (u32, u32),
    /// Side of the world square this field covers — the swell cascade's tile,
    /// so the foam repeats on exactly the period the waves that made it do.
    extent: f32,
}

/// The largest number of wake sources the buffer is sized for.
const MAX_WAKES: usize = 16;

impl SurfaceState {
    pub fn new(device: &wgpu::Device, fft: &OceanFft, peak_wavelength: f32) -> SurfaceState {
        let cas = fft.cas;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ocean surface state"),
            source: wgpu::ShaderSource::Wgsl(
                SHADER
                    .replace(
                        "$CASCADES",
                        &crate::waves_gpu::cascade_wgsl(
                            "cascade_a",
                            "cascade_b",
                            "cascade_c",
                            "cascade_samp",
                            peak_wavelength,
                            &cas,
                        ),
                    )
                    .into(),
            ),
        });

        let make_tex = |label: &str| {
            let t = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: RESOLUTION,
                    height: RESOLUTION,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            Arc::new(t.create_view(&wgpu::TextureViewDescriptor::default()))
        };
        let views = [make_tex("ocean foam A"), make_tex("ocean foam B")];

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ocean foam sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ocean foam params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let wakes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ocean wake sources"),
            contents: bytemuck::cast_slice(&[WakeSource::default(); MAX_WAKES]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let cascade_tex = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let storage_buf = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ocean foam bgl"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                storage_buf(2),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                cascade_tex(6),
                cascade_tex(7),
                cascade_tex(8),
            ],
        });

        let make_bind = |src: usize, dst: usize| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ocean foam bg"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wakes.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&views[src]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&views[dst]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(&fft.cascade_views[0]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(&fft.cascade_views[1]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: wgpu::BindingResource::TextureView(&fft.cascade_views[2]),
                    },
                ],
            })
        };
        let binds = [make_bind(0, 1), make_bind(1, 0)];

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ocean foam pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ocean foam pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        SurfaceState {
            pipeline,
            binds,
            views,
            params,
            wakes,
            flip: 0,
            groups: (RESOLUTION.div_ceil(8), RESOLUTION.div_ceil(8)),
            extent: cas.extent(),
        }
    }

    /// The view the water shader should sample this frame — the one just written.
    pub fn current(&self) -> Arc<wgpu::TextureView> {
        self.views[(self.flip + 1) % 2].clone()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        dt: f32,
        wind: [f32; 2],
        foam_threshold: f32,
        foam_softness: f32,
        decay: f32,
        sources: &[WakeSource],
    ) {
        let n = sources.len().min(MAX_WAKES);
        if n > 0 {
            queue.write_buffer(&self.wakes, 0, bytemuck::cast_slice(&sources[..n]));
        }
        queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&Params {
                extent: self.extent,
                // A long pause (a breakpoint, a stalled frame) would otherwise
                // decay the whole field to nothing in one step.
                dt: dt.clamp(0.0, 0.1),
                wake_count: n as u32,
                wind,
                foam_softness,
                foam_threshold,
                decay,
                _pad: [0.0; 3],
            }),
        );

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ocean foam pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.binds[self.flip], &[]);
            pass.dispatch_workgroups(self.groups.0, self.groups.1, 1);
        }
        self.flip = (self.flip + 1) % 2;
    }
}

#[cfg(test)]
mod tests {
    /// The shader's `kelvin_wake`, transcribed. Kept in step with it by the
    /// assertions below, which are about the *shape* of a ship wave rather than
    /// about any particular number — if the transcription drifts, the wedge stops
    /// being 19.47 degrees and the tests say so.
    fn kelvin(world: [f32; 2], pos: [f32; 2], vel: [f32; 2], strength: f32, radius: f32) -> f32 {
        let rel = [world[0] - pos[0], world[1] - pos[1]];
        let dist = (rel[0] * rel[0] + rel[1] * rel[1]).sqrt();
        let hull = strength * (1.0 - smoothstep(0.0, radius.max(0.5), dist));
        let u = (vel[0] * vel[0] + vel[1] * vel[1]).sqrt();
        if u < 0.4 {
            return hull;
        }
        let fwd = [vel[0] / u, vel[1] / u];
        let along = -(rel[0] * fwd[0] + rel[1] * fwd[1]);
        let across = (rel[0] * -fwd[1] + rel[1] * fwd[0]).abs();
        if along < 0.0 {
            return hull;
        }
        let lam = (std::f32::consts::TAU * u * u / 9.81).clamp(1.5, 400.0);
        let a = along.max(0.001);
        let decay = strength / (1.0 + a / lam.max(1.0)).sqrt();
        let arm = a * 0.3536;
        let arm_w = (lam * 0.18).max(radius * 0.6);
        let mut w = (1.0 - smoothstep(0.0, arm_w, (across - arm).abs())) * decay;
        let inside = 1.0 - smoothstep(arm, arm + arm_w, across);
        let phase = (a - across * across / (2.0 * a).max(0.001)) / lam;
        let crest = 0.5 + 0.5 * (std::f32::consts::TAU * phase).cos();
        w += inside * decay * 0.5 * crest * crest * crest;
        w.max(hull)
    }

    fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
        let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    const V: [f32; 2] = [8.0, 0.0];

    /// The wedge half-angle is arcsin(1/3) = 19.47 degrees, and — the part that
    /// surprises people — it does not depend on speed.
    #[test]
    fn the_wedge_is_nineteen_and_a_half_degrees_at_any_speed() {
        for speed in [4.0f32, 8.0, 16.0] {
            let along = 90.0f32;
            // Sample across the wake at a fixed distance behind the hull and find
            // where the crest line is.
            let mut best = (0.0f32, 0.0f32);
            let mut x = 0.0f32;
            while x < along {
                let w = kelvin([-along, x], [0.0, 0.0], [speed, 0.0], 1.0, 5.0);
                if w > best.0 {
                    best = (w, x);
                }
                x += 0.25;
            }
            let angle = (best.1 / along).atan().to_degrees();
            assert!(
                (angle - 19.47).abs() < 1.5,
                "at {speed} m/s the arm sits at {angle:.2} deg, not 19.47"
            );
        }
    }

    #[test]
    fn nothing_runs_ahead_of_the_hull() {
        // 40 m in front, well clear of the hull's own radius.
        let ahead = kelvin([40.0, 0.0], [0.0, 0.0], V, 1.0, 5.0);
        let behind = kelvin([-40.0, 14.0], [0.0, 0.0], V, 1.0, 5.0);
        assert!(ahead < 1e-6, "wake ahead of the bow: {ahead}");
        assert!(behind > 0.05, "no wake behind the stern: {behind}");
    }

    #[test]
    fn the_trail_spreads_and_fades() {
        // On the arm at two distances: energy spread over a widening front.
        let near = kelvin([-40.0, 40.0 * 0.3536], [0.0, 0.0], V, 1.0, 5.0);
        let far = kelvin([-200.0, 200.0 * 0.3536], [0.0, 0.0], V, 1.0, 5.0);
        assert!(
            near > far,
            "wake should fade with distance: {near} then {far}"
        );
        assert!(far > 0.0, "and not vanish outright: {far}");
    }

    #[test]
    fn a_drifting_object_leaves_a_patch_not_a_wedge() {
        // Below a walking pace there is no wedge, just stirred water — and it is
        // centred on the object rather than trailing it.
        let slow = [0.05f32, 0.0];
        let on = kelvin([0.0, 0.0], [0.0, 0.0], slow, 1.0, 5.0);
        let off = kelvin([-40.0, 14.0], [0.0, 0.0], slow, 1.0, 5.0);
        assert!(
            on > 0.9,
            "the patch should be strongest on the object: {on}"
        );
        assert!(off < 1e-6, "and absent 40 m astern: {off}");
    }
}
