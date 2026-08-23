//! GPU wave displacement: a compute pass that rewrites the water's vertex
//! buffer in place.
//!
//! [`ShaderMaterial`](threers::ShaderMaterial) replaces the fragment stage only,
//! so there is no vertex hook to displace through. But the example owns the
//! `wgpu::Device` the renderer is built on, which is enough: mark the geometry
//! [`gpu_writable`](threers::BufferGeometry::gpu_writable), ask the renderer for
//! its vertex buffer, and bind it to a compute shader as `storage, read_write`.
//!
//! What that buys is not a faster CPU loop — it is the removal of the whole
//! round trip. Before, every frame evaluated ~40 components at every vertex on
//! the CPU, rebuilt the interleaved array, and re-uploaded a megabyte. Now the
//! CPU writes two small buffers (the component table and a 16-byte push of
//! parameters) and the vertices never leave the GPU.
//!
//! The buffer handle is only valid while the geometry is not re-uploaded, so
//! nothing here may touch `geometry_version` after setup — the mesh is driven
//! one way, from this pass.

use std::sync::Arc;

use threers::{BufferGeometry, Renderer};
use wgpu::util::DeviceExt;

use crate::ocean_fft::{Cascades, OceanFft};

/// Matches `Params` in the shader. `centre` is the disc's world position; the
/// mesh is stored in disc-local coordinates so the wave field can stay anchored
/// to the world while the sampling pattern follows the camera.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    centre: [f32; 2],
    vertex_count: u32,
    _pad: u32,
}

/// Threads per workgroup. 64 is the portable sweet spot — a multiple of every
/// current wave/warp width, and small enough that the tail of a dispatch wastes
/// little.
///
/// Substituted into the shader rather than written twice: a mismatch between
/// this and `@workgroup_size` would not error, it would silently leave part of
/// the mesh undisplaced.
const WORKGROUP: u32 = 64;

/// Shared by every consumer of the cascades: the same sampling, the same
/// band-limit, the same derivative scheme. Anything that reads the surface has to
/// read it identically or the mesh, the shading and the foam disagree about where
/// the water is.
///
/// `$L0/$L1/$L2` are substituted with the cascade tile sizes.
/// The shared cascade code, with the binding names its caller actually declared
/// substituted in. WGSL has no way to alias a binding — `alias` is for types —
/// so the names have to be textual.
pub fn cascade_wgsl(
    a: &str,
    b: &str,
    c: &str,
    sampler: &str,
    peak_wavelength: f32,
    cas: &Cascades,
) -> String {
    let k0 = std::f32::consts::TAU / peak_wavelength.max(1.0);
    CASCADE_WGSL
        .replace("$K0", &format!("{k0:?}"))
        .replace(
            "$SHORE_RADIUS",
            &format!("{:?}", crate::terrain::SHORE_RADIUS),
        )
        .replace("$BASE_DEPTH", &format!("{:?}", crate::terrain::BASE_DEPTH))
        .replace("$N", &format!("{:?}", cas.n as f32))
        .replace("$L0", &format!("{:?}", cas.tiles[0]))
        .replace("$L1", &format!("{:?}", cas.tiles[1]))
        .replace("$L2", &format!("{:?}", cas.tiles[2]))
        .replace("CASCADE_SAMP", sampler)
        .replace("CASCADE_A", a)
        .replace("CASCADE_B", b)
        .replace("CASCADE_C", c)
}

pub const CASCADE_WGSL: &str = r#"
// ---- the sea floor -------------------------------------------------------
// Lives here rather than in any one consumer: the mesh, the shading, the foam
// and the spray all need it, and shoaling makes them *disagree* visibly if they
// do not read the same bed.
fn seabed_height(p: vec2<f32>) -> f32 {
    let r = length(p);
    let land = 18.0 * (1.0 - smoothstep(0.0, $SHORE_RADIUS, r));
    let shelf = -($BASE_DEPTH * smoothstep($SHORE_RADIUS, 900.0, r)
        + 6.0 * smoothstep($SHORE_RADIUS, $SHORE_RADIUS + 140.0, r));
    // The beach face — see `seabed_height` in terrain.rs for why a slope has to
    // be put in by hand here.
    let face = -1.9 * smoothstep($SHORE_RADIUS - 16.0, $SHORE_RADIUS + 6.0, r);
    let relief = (land / 18.0)
        * (4.0 * sin(p.x * 0.031) * cos(p.y * 0.026) + 2.0 * sin(p.x * 0.09 + p.y * 0.07));
    let deep = -30.0 * smoothstep(1200.0, 2500.0, r);
    let ripple = 1.6 * sin(p.x * 0.021) * sin(p.y * 0.017);
    // Dunes and ridges on the dry island. The twin of `land_detail` in
    // terrain.rs — the two have to agree or the waterline the mesh draws is not
    // the waterline the water thinks it is.
    let mask = smoothstep(0.5, 6.0, land);
    let dunes = 1.35 * sin(p.x * 0.24 + p.y * 0.11) * cos(p.y * 0.19 - p.x * 0.07);
    let warp = 2.0 * sin(p.x * 0.06 + p.y * 0.05);
    let ridges = 0.34 * sin(p.x * 0.52 - p.y * 0.38 + warp);
    return land + shelf + face + relief + deep + ripple + mask * (dunes + ridges);
}

fn water_depth(p: vec2<f32>) -> f32 {
    return max(-seabed_height(p), 0.0);
}

/// Sea-floor normal, straight from the analytic bed rather than from the mesh.
///
/// The mesh is 4 m per quad; this is exact at any scale, which is what lets the
/// floor be shaded with more relief than it is tessellated with.
fn seabed_normal(p: vec2<f32>, e: f32) -> vec3<f32> {
    let hx = seabed_height(p + vec2<f32>(e, 0.0)) - seabed_height(p - vec2<f32>(e, 0.0));
    let hz = seabed_height(p + vec2<f32>(0.0, e)) - seabed_height(p - vec2<f32>(0.0, e));
    return normalize(vec3<f32>(-hx, 2.0 * e, -hz));
}

// ---- shoaling ------------------------------------------------------------
// A wave entering shallow water slows down, shortens, and grows. All three fall
// out of the finite-depth dispersion relation, w^2 = g k tanh(kd), which the
// cascades themselves cannot use: an FFT field is homogeneous, and depth is not.
// So the cascades are generated deep and shoaled here, per sample, from the
// analytic bed.

/// Local wavenumber for a deep-water `k0` over depth `d`.
///
/// Fenton & McKee's explicit inverse of the dispersion relation — good to about
/// 1.5%, and no iteration, which matters when this runs per vertex and per pixel.
fn local_wavenumber(k0: f32, d: f32) -> f32 {
    let x = k0 * d;
    // Past a few radians of kd, tanh is 1 and the water is deep by definition.
    if (x > 3.0) { return k0; }
    let t = tanh(pow(max(x, 1.0e-4), 0.75));
    return k0 * pow(t, -2.0 / 3.0);
}

struct Shoal {
    /// Amplitude gain from energy-flux conservation.
    gain   : f32,
    /// Extra horizontal displacement: shoaling waves peak up and flatten out.
    chop   : f32,
    /// Sample-position offset standing in for refraction.
    warp   : vec2<f32>,
    depth  : f32,
};

fn shoaling(p: vec2<f32>) -> Shoal {
    var out: Shoal;
    let k0 = $K0;
    let d = water_depth(p);
    out.depth = d;

    // Deep water: nothing to do, and this is most of the frame.
    if (k0 * d > 3.0) {
        out.gain = 1.0;
        out.chop = 1.0;
        out.warp = vec2<f32>(0.0);
        return out;
    }

    let k = local_wavenumber(k0, d);
    let kd = k * d;

    // Green's law, in the form that stays finite in both limits:
    // Ks = sqrt(k / (k0 (1 + 2kd/sinh 2kd))). Deep gives 1; shallow gives the
    // classic d^(-1/4) growth.
    let sh = select(2.0 * kd / sinh(2.0 * kd), 0.0, kd > 5.0);
    out.gain = sqrt(max(k / (k0 * (1.0 + sh)), 0.0));

    // Crests sharpen as they shoal — the wave becomes skewed long before it
    // breaks, and that asymmetry is most of what makes shallow water read as
    // shallow.
    out.chop = 1.0 + 1.6 * (1.0 - clamp(kd / 1.5, 0.0, 1.0));

    // Refraction, to first order. Slowing down means the phase lags, and a lag
    // that varies across the shore is what turns the crests to face it. The lag
    // is integrated coming in from deep water along the depth gradient, and
    // applied as a shift of the sample position along that same direction —
    // which for a wave travelling shoreward is exactly a phase offset.
    out.warp = vec2<f32>(0.0);
    // Only worth integrating where the bend is actually large. Past about one
    // radian of kd the local wavenumber is within a few percent of the deep-water
    // one, so the lag barely accumulates — and this loop is the most expensive
    // thing in the shader, four transcendentals a step. Gating it here is what
    // keeps the foam pass, which runs it over a million texels, affordable.
    if (kd < 1.2) {
        let e = 6.0;
        let gx = water_depth(p + vec2<f32>(e, 0.0)) - water_depth(p - vec2<f32>(e, 0.0));
        let gz = water_depth(p + vec2<f32>(0.0, e)) - water_depth(p - vec2<f32>(0.0, e));
        let g = vec2<f32>(gx, gz);
        let glen = length(g);
        if (glen > 1.0e-5) {
            let offshore = g / glen;
            var lag = 0.0;
            var q = p;
            let step = 18.0;
            for (var i = 0; i < 3; i = i + 1) {
                lag = lag + (local_wavenumber(k0, water_depth(q)) - k0) * step;
                q = q + offshore * step;
            }
            // Bound it.
            //
            // `local_wavenumber` grows as d^(-1/2) as the water shallows, so
            // this integral runs away: at five centimetres of depth it asks for
            // a sample some six hundred metres off. That is not refraction, it
            // is a swirl of unrelated sea — and because the mesh is displaced
            // from the same function, it also threw up spurious walls of water
            // that hid whatever was behind them. Half a wavelength is as far as
            // a first-order phase shift can honestly mean, and the shift is
            // faded out through the last half-metre, where the wave has broken
            // and the linear theory behind all of this has stopped applying.
            let lambda = 6.283185307179586 / k0;
            let shift = clamp(lag / k0, -0.5 * lambda, 0.5 * lambda);
            out.warp = offshore * shift * smoothstep(0.0, 0.75, d);
        }
    }
    return out;
}

// ---- the cascades --------------------------------------------------------
// Shared by every consumer: the same sampling, the same band-limit, the same
// derivative scheme. Anything that reads the surface has to read it identically
// or the mesh, the shading and the foam disagree about where the water is.
//
// `$L0/$L1/$L2` are substituted with the cascade tile sizes.

// Displacement summed over the cascades, each faded out once its texels are
// finer than the sampler can resolve. Without that the fine cascade aliases into
// exactly the boiling the band limit exists to prevent.
fn cascade_fade(footprint: f32) -> vec3<f32> {
    // $N is the lattice size, so `$Lk / $N` is one texel of that cascade: a
    // cascade starts fading once the sampler's footprint reaches its own texel
    // and is gone by eight of them. Writing it against the real resolution is
    // what lets the lattice change without the band limit going stale.
    return vec3<f32>(
        1.0 - smoothstep($L0 / $N, $L0 / ($N / 8.0), footprint),
        1.0 - smoothstep($L1 / $N, $L1 / ($N / 8.0), footprint),
        1.0 - smoothstep($L2 / $N, $L2 / ($N / 8.0), footprint));
}

fn cascade_disp(p: vec2<f32>, fade: vec3<f32>) -> vec3<f32> {
    return cascade_disp_warped(p, fade, vec2<f32>(0.0));
}

/// The cascades, with refraction applied where refraction means something.
///
/// The shoaling lag is computed for *one* wavelength — the spectral peak — and
/// applying that one shift to all three cascades is wrong twice over. A wave
/// only feels the bottom when the depth is a fraction of its own length, so the
/// ripples in the nine-metre tile barely refract at all over this bed; and the
/// shift is tens of metres, which against a nine-metre tile is not a phase
/// offset but a decorrelation. What that looks like is the swell bending
/// correctly while the fine detail smears into long radial streaks fanning out
/// from the island, because the shift follows the depth gradient and the depth
/// gradient is radial.
fn cascade_disp_warped(p: vec2<f32>, fade: vec3<f32>, warp: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(CASCADE_A, CASCADE_SAMP, (p + warp) / $L0, 0.0).xyz * fade.x
         + textureSampleLevel(CASCADE_B, CASCADE_SAMP, (p + warp * 0.2) / $L1, 0.0).xyz * fade.y
         + textureSampleLevel(CASCADE_C, CASCADE_SAMP, p / $L2, 0.0).xyz * fade.z;
}

/// Cascades, shoaled, with the breaking limit applied.
fn shoaled_disp(p: vec2<f32>, fade: vec3<f32>, sh: Shoal) -> vec3<f32> {
    var d = cascade_disp_warped(p, fade, sh.warp);
    d = vec3<f32>(d.x * sh.chop, d.y * sh.gain, d.z * sh.chop);
    // A wave cannot be taller than the water it is standing in: past about
    // 0.78 of the depth it breaks (McCowan). `tanh` saturates smoothly, so deep
    // water is untouched and the surf zone stops growing and starts spilling —
    // which is what puts a limit on shoaling instead of letting it run away.
    let limit = max(0.78 * sh.depth, 0.05);
    d.y = limit * tanh(d.y / limit);
    return d;
}

struct Surface {
    disp   : vec3<f32>,
    normal : vec3<f32>,
    // 1 - det(J) of the horizontal displacement: positive where the surface is
    // folding onto itself, which is what a breaking wave is.
    fold   : f32,
    // How close this wave is to its depth-limited height. 1 means breaking.
    breaking : f32,
    depth  : f32,
};

// Central differences rather than extra transforms: the derivative fields would
// be four more IFFTs and four more textures, and the surface is already a
// texture that can be differenced. The step doubles as the band limit.
fn sample_surface(p: vec2<f32>, footprint: f32) -> Surface {
    let fade = cascade_fade(footprint);
    // Shoaling is resolved once and shared by the difference taps: it varies on
    // the scale of the bed, which is far coarser than the step.
    let sh = shoaling(p);
    let e = max(footprint, $L2 / $N);
    let d = shoaled_disp(p, fade, sh);
    let px = shoaled_disp(p + vec2<f32>(e, 0.0), fade, sh);
    let mx = shoaled_disp(p - vec2<f32>(e, 0.0), fade, sh);
    let pz = shoaled_disp(p + vec2<f32>(0.0, e), fade, sh);
    let mz = shoaled_disp(p - vec2<f32>(0.0, e), fade, sh);

    let ddx = (px - mx) / (2.0 * e);
    let ddz = (pz - mz) / (2.0 * e);

    // Tangents of the displaced surface, then their cross product.
    let tx = vec3<f32>(1.0 + ddx.x, ddx.y, ddx.z);
    let tz = vec3<f32>(ddz.x, ddz.y, 1.0 + ddz.z);

    var out: Surface;
    out.disp = d;
    out.normal = normalize(cross(tz, tx));
    out.fold = 1.0 - ((1.0 + ddx.x) * (1.0 + ddz.z) - ddz.x * ddx.z);
    out.depth = sh.depth;
    // Unshoaled height against the depth limit: how hard this wave is breaking.
    out.breaking = clamp(cascade_disp_warped(p, fade, sh.warp).y * sh.gain
                         / max(0.78 * sh.depth, 0.05), 0.0, 1.0);
    return out;
}
"#;

const SHADER: &str = r#"
struct Params {
    centre       : vec2<f32>,
    vertex_count : u32,
};

@group(0) @binding(0) var<uniform>              params : Params;
@group(0) @binding(1) var cascade_samp : sampler;
// Per vertex: [x, z, spacing, _] in disc-local space. Never changes.
@group(0) @binding(2) var<storage, read>        base   : array<vec4<f32>>;
// The mesh itself: 16 floats per vertex, position(3) normal(3) uv(2)
// colour(4) tangent(4). Only the first six are ours.
@group(0) @binding(3) var<storage, read_write>  verts  : array<f32>;
@group(0) @binding(4) var cascade_a : texture_2d<f32>;
@group(0) @binding(5) var cascade_b : texture_2d<f32>;
@group(0) @binding(6) var cascade_c : texture_2d<f32>;

$CASCADES

@compute @workgroup_size($WORKGROUP)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.vertex_count) { return; }

    let b = base[i];
    let local = vec2<f32>(b.x, b.y);
    let s = sample_surface(local + params.centre, b.z);

    let o = i * 16u;
    verts[o]      = local.x + s.disp.x;
    verts[o + 1u] = s.disp.y;
    verts[o + 2u] = local.y + s.disp.z;
    verts[o + 3u] = s.normal.x;
    verts[o + 4u] = s.normal.y;
    verts[o + 5u] = s.normal.z;
}
"#;

pub struct WaveCompute {
    /// The geometry whose vertex buffer this pass owns, plus the version it had
    /// when the binding was made. Anything that bumps the version makes the
    /// renderer allocate a fresh buffer, leaving this pass writing the old one —
    /// the mesh would silently freeze rather than error, so it is checked.
    geometry: Arc<BufferGeometry>,
    bound_version: u32,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    params: wgpu::Buffer,
    vertex_count: u32,
    workgroups: u32,
}

impl WaveCompute {
    /// Build the pass against an already-uploaded geometry.
    ///
    /// `geometry` must be [`gpu_writable`](BufferGeometry::gpu_writable) and
    /// must already be on the GPU — call
    /// [`Renderer::upload_geometry`](threers::Renderer::upload_geometry) first.
    pub fn new(
        device: &wgpu::Device,
        renderer: &Renderer,
        geometry: &Arc<BufferGeometry>,
        base: &[[f32; 4]],
        fft: &OceanFft,
        peak_wavelength: f32,
    ) -> WaveCompute {
        let cas = fft.cas;
        let vertex_buffer = renderer
            .vertex_buffer(geometry)
            .expect("water geometry must be uploaded before the compute pass is built");

        let source = SHADER
            .replace(
                "$CASCADES",
                &cascade_wgsl(
                    "cascade_a",
                    "cascade_b",
                    "cascade_c",
                    "cascade_samp",
                    peak_wavelength,
                    &cas,
                ),
            )
            .replace("$WORKGROUP", &WORKGROUP.to_string());
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ocean wave displacement"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ocean wave params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let base_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ocean grid base"),
            contents: bytemuck::cast_slice(base),
            usage: wgpu::BufferUsages::STORAGE,
        });
        // Repeat, because every cascade tiles: that is what a tile size means.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ocean cascade sampler"),
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
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ocean wave bgl"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                tex(4),
                tex(5),
                tex(6),
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ocean wave bg"),
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
                    resource: base_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&fft.cascade_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&fft.cascade_views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&fft.cascade_views[2]),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ocean wave pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ocean wave pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let vertex_count = base.len() as u32;
        WaveCompute {
            geometry: geometry.clone(),
            bound_version: geometry.geometry_version,
            pipeline,
            bind_group,
            params,
            vertex_count,
            workgroups: vertex_count.div_ceil(WORKGROUP),
        }
    }

    /// Encode one frame's displacement. Must be submitted before the draw that
    /// reads the vertices; queue order takes care of the rest.
    pub fn dispatch(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        centre: [f32; 2],
    ) {
        assert_eq!(
            self.geometry.geometry_version, self.bound_version,
            "water geometry was re-uploaded behind the compute pass: it is driven \
             from the GPU, so nothing may touch its attributes after setup"
        );
        queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&Params {
                centre,
                vertex_count: self.vertex_count,
                _pad: 0,
            }),
        );

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ocean wave pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.workgroups, 1, 1);
        }
    }
}
