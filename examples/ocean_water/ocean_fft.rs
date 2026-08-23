//! The FFT ocean: a JONSWAP spectrum inverse-transformed into tiling cascades.
//!
//! A sum of forty Gerstner components is a sea with forty waves in it. Real
//! water has a continuum, and the give-away is not any single wave but the
//! *texture* between them — which is what a spectrum sampled on a full lattice
//! and inverse-transformed gives you: 65 536 modes per cascade instead of forty
//! total, for less work per pixel, because the result is a texture rather than a
//! loop.
//!
//! # Cascades
//!
//! One tile that both resolves capillary ripples and does not visibly repeat
//! would need to be enormous. Three tiles instead, each carrying its own band of
//! the spectrum: a long swell that repeats far past the haze, a mid band, and a
//! fine band whose repeat is too small to read as one. Each cascade takes the
//! wavenumbers below its own Nyquist and above the previous cascade's, so the
//! bands abut without double-counting. Swell, waves, ripples — the split is the
//! conventional one, and it falls out of the band limit rather than being a
//! stylistic choice.
//!
//! # Why not rlx
//!
//! rlx has an FFT and this project already depends on it, so it is the obvious
//! candidate. Measured, it is 0.55 ms for one axis of one 256² field on the CPU
//! backend — around 6.6 ms a frame for three cascades, against a whole-frame CPU
//! budget here of 0.2 ms. Its GPU backend is wgpu 30 while this renderer is wgpu
//! 0.20, so those are two separate devices and the result would still cross host
//! memory (the bridge hands back `Vec<f32>`), 3 MB of it every frame.
//!
//! So rlx is not in this path — but it *is* the oracle the butterfly table is
//! tested against, in [`crate::fft`]. Its FFT validates this one.

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::fft::{butterfly_table, log2_exact};
use crate::preset::Preset;
use crate::spectrum::G;

/// Texels per side, per cascade — the default. See [`Cascades`].
pub const N: usize = 256;

/// Tile size of each cascade, metres. The largest sets how far out the swell
/// repeats; the smallest sets the finest detail the surface can hold.
pub const TILES: [f32; CASCADES] = [1024.0, 96.0, 9.0];
pub const CASCADES: usize = 3;

/// The cascade lattice: how many texels per side, and how much world each tile
/// covers.
///
/// This used to be two constants, which meant resolution and extent were baked
/// into the build. They are runtime values because they are the two knobs that
/// genuinely trade quality for cost here: `n` sets how many modes the spectrum
/// is sampled at, and `tiles[0]` sets how far the swell travels before it
/// repeats.
///
/// Only `n` can change without recompiling anything — the FFT kernels read their
/// own extents from `textureDimensions`, and the stage count comes from a table.
/// The tile sizes are substituted into every consumer's WGSL, so changing them
/// rebuilds the materials. [`crate::world::World`] handles both.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cascades {
    /// Texels per side. Must be a power of two.
    pub n: usize,
    /// World size of each cascade's tile, metres, descending.
    pub tiles: [f32; CASCADES],
}

impl Default for Cascades {
    fn default() -> Self {
        Cascades { n: N, tiles: TILES }
    }
}

impl Cascades {
    /// Set the lattice size, clamped to the powers of two the butterfly table
    /// and the surface's detail budget both make sense over.
    pub fn with_resolution(mut self, n: usize) -> Self {
        let n = n.clamp(64, 1024);
        // Round to the nearest power of two: a radix-2 FFT has no meaning off it,
        // and silently transforming the wrong size is worse than snapping.
        self.n = 1usize << (n as f32).log2().round() as u32;
        self
    }

    /// Set the largest tile — how far the swell runs before it repeats.
    ///
    /// Only the swell cascade scales. The other two carry detail rather than
    /// extent, and stretching them would just make the ripples coarse.
    pub fn with_max_scale(mut self, metres: f32) -> Self {
        self.tiles[0] = metres.clamp(256.0, 8192.0);
        self
    }

    /// Side of the world square the swell repeats on. The persistent foam field
    /// tiles on exactly this, so foam repeats with the waves that made it.
    pub fn extent(&self) -> f32 {
        self.tiles[0]
    }
}

/// Wavenumber at which each cascade hands over to the next: its own Nyquist.
fn band_edges(cas: &Cascades) -> [f32; CASCADES] {
    let mut e = [0.0; CASCADES];
    for (i, l) in cas.tiles.iter().enumerate() {
        e[i] = std::f32::consts::PI * cas.n as f32 / l;
    }
    e
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SpectrumParams {
    time: f32,
    choppiness: f32,
    _pad: [f32; 2],
    /// `xyz` are the cascade tile sizes; `w` is unused.
    tiles: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct StageParams {
    stage: u32,
    vertical: u32,
    _pad: [u32; 2],
}

const SPECTRUM_WGSL: &str = r#"
struct Params {
    time       : f32,
    choppiness : f32,
    tiles      : vec4<f32>,
};
@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var h0  : texture_2d_array<f32>;
@group(0) @binding(2) var out_tex : texture_storage_2d_array<rgba32float, write>;

const G: f32 = 9.81;
const TAU: f32 = 6.283185307179586;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(out_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let layer = i32(gid.z);
    let n = i32(dims.x);
    let x = i32(gid.x);
    let y = i32(gid.y);

    // Wave vector, indices centred so the DC term sits in the middle.
    let l = params.tiles[layer];
    let k = vec2<f32>(f32(x - n / 2), f32(y - n / 2)) * (TAU / l);
    let kmag = length(k);
    if (kmag < 1e-6) {
        textureStore(out_tex, vec2<i32>(x, y), layer, vec4<f32>(0.0));
        return;
    }

    // Deep-water dispersion. This is the only place time enters: the spectrum is
    // fixed and each mode just rotates, which is why the sea cannot drift.
    let w = sqrt(G * kmag) * params.time;
    let rot = vec2<f32>(cos(w), sin(w));

    let seed = textureLoad(h0, vec2<i32>(x, y), layer, 0);
    // h(k,t) = h0(k) e^(iwt) + conj(h0(-k)) e^(-iwt), the pair that keeps the
    // field Hermitian and therefore its inverse transform real.
    let h = cmul(seed.xy, rot) + cmul(seed.zw, vec2<f32>(rot.x, -rot.y));

    // Horizontal displacement spectra: -i * k/|k| * h.
    let kn = k / kmag;
    let ih = vec2<f32>(h.y, -h.x);
    let hx = ih * kn.x * params.choppiness;
    let hz = ih * kn.y * params.choppiness;

    // Two real fields ride in one complex transform: pack Dx + i*Dz, and the
    // inverse comes back with Dx in the real part and Dz in the imaginary.
    let xz = vec2<f32>(hx.x - hz.y, hx.y + hz.x);
    textureStore(out_tex, vec2<i32>(x, y), layer, vec4<f32>(xz, h));
}
"#;

const BUTTERFLY_WGSL: &str = r#"
struct Stage {
    stage    : u32,
    vertical : u32,
};
@group(0) @binding(0) var<uniform> st : Stage;
@group(0) @binding(1) var butterfly : texture_2d<f32>;
@group(0) @binding(2) var src : texture_2d_array<f32>;
@group(0) @binding(3) var dst : texture_storage_2d_array<rgba32float, write>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// One butterfly stage, transcribed from `fft::fft_1d_with_table` — which is
// tested against both a naive DFT and rlx's FFT. The table carries the twiddle
// and both source indices, so this kernel has no index arithmetic of its own to
// get wrong.
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let layer = i32(gid.z);

    let idx = select(i32(gid.x), i32(gid.y), st.vertical == 1u);
    let e = textureLoad(butterfly, vec2<i32>(idx, i32(st.stage)), 0);
    let w = e.xy;
    var ca: vec2<i32>;
    var cb: vec2<i32>;
    if (st.vertical == 1u) {
        ca = vec2<i32>(i32(gid.x), i32(e.z));
        cb = vec2<i32>(i32(gid.x), i32(e.w));
    } else {
        ca = vec2<i32>(i32(e.z), i32(gid.y));
        cb = vec2<i32>(i32(e.w), i32(gid.y));
    }
    let a = textureLoad(src, ca, layer, 0);
    let b = textureLoad(src, cb, layer, 0);
    // Both complex fields in the RGBA advance together.
    textureStore(dst, vec2<i32>(gid.xy), layer,
                 vec4<f32>(a.xy + cmul(w, b.xy), a.zw + cmul(w, b.zw)));
}
"#;

const ASSEMBLE_WGSL: &str = r#"
@group(0) @binding(0) var src : texture_2d_array<f32>;
@group(0) @binding(1) var dst : texture_storage_2d_array<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let layer = i32(gid.z);
    let v = textureLoad(src, vec2<i32>(gid.xy), layer, 0);
    // The spectrum was indexed with DC at the centre, so the transform comes out
    // shifted by half a tile in each axis. (-1)^(x+y) undoes that.
    let sign = select(-1.0, 1.0, ((gid.x + gid.y) & 1u) == 0u);
    // Real part of the packed field is Dx, imaginary is Dz; the second complex
    // slot carries Dy.
    textureStore(dst, vec2<i32>(gid.xy), layer,
                 vec4<f32>(v.x * sign, v.z * sign, v.y * sign, 0.0));
}
"#;

/// One dominant mode, kept on the CPU so buoyancy can be answered without
/// reading a texture back.
#[derive(Clone, Copy)]
pub struct Mode {
    pub k: [f32; 2],
    pub omega: f32,
    /// `h0(k)`, complex.
    pub h0: [f32; 2],
    /// `conj(h0(-k))`, complex.
    pub h0c: [f32; 2],
}

pub struct OceanFft {
    spectrum: wgpu::ComputePipeline,
    butterfly: wgpu::ComputePipeline,
    assemble: wgpu::ComputePipeline,
    spectrum_bg: wgpu::BindGroup,
    /// `[stage][vertical][parity]` — 2 ping-pong directions per stage.
    stage_bgs: Vec<wgpu::BindGroup>,
    assemble_bgs: [wgpu::BindGroup; 2],
    spectrum_params: wgpu::Buffer,
    /// Held only to keep the per-stage uniforms alive for as long as the bind
    /// groups that point at them.
    #[allow(dead_code)]
    stage_params: Vec<wgpu::Buffer>,
    /// One `D2` view per cascade, for the material to sample.
    pub cascade_views: Vec<Arc<wgpu::TextureView>>,
    /// The strongest modes of the swell cascade, for CPU buoyancy.
    pub modes: Vec<Mode>,
    /// Held so the spectrum can be re-seeded — a wind change rewrites this
    /// texture and nothing else, which is why `set_wind` costs a texture upload
    /// rather than a rebuild.
    h0_tex: wgpu::Texture,
    pub cas: Cascades,
    stages: usize,
    groups: (u32, u32),
}

impl OceanFft {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        preset: &Preset,
        hs: f32,
        cas: Cascades,
    ) -> OceanFft {
        let n = cas.n;
        let stages = log2_exact(n);
        let (h0_data, modes) = build_h0(preset, hs, &cas);

        // --- textures ---
        let array_tex = |label: &str, format: wgpu::TextureFormat, extra: wgpu::TextureUsages| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: n as u32,
                    height: n as u32,
                    depth_or_array_layers: CASCADES as u32,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | extra,
                view_formats: &[],
            })
        };

        let h0_tex = array_tex(
            "ocean h0",
            wgpu::TextureFormat::Rgba32Float,
            wgpu::TextureUsages::COPY_DST,
        );
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &h0_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&h0_data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some((n * 16) as u32),
                rows_per_image: Some(n as u32),
            },
            wgpu::Extent3d {
                width: n as u32,
                height: n as u32,
                depth_or_array_layers: CASCADES as u32,
            },
        );

        let work: Vec<wgpu::Texture> = (0..2)
            .map(|i| {
                array_tex(
                    &format!("ocean fft work {i}"),
                    wgpu::TextureFormat::Rgba32Float,
                    wgpu::TextureUsages::STORAGE_BINDING,
                )
            })
            .collect();
        let disp = array_tex(
            "ocean displacement",
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::STORAGE_BINDING,
        );

        let array_view = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());
        let h0_view = array_view(&h0_tex);
        let work_views: Vec<wgpu::TextureView> = work.iter().map(array_view).collect();
        let disp_array = array_view(&disp);

        // A per-layer D2 view is what the material can bind: its texture slots
        // are plain `texture_2d`, and one layer of an array is a valid D2 view.
        let cascade_views: Vec<Arc<wgpu::TextureView>> = (0..CASCADES)
            .map(|i| {
                Arc::new(disp.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("ocean cascade"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: i as u32,
                    array_layer_count: Some(1),
                    ..Default::default()
                }))
            })
            .collect();

        let butterfly_data = butterfly_table(n, 1.0);
        let butterfly_tex = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("ocean butterfly"),
                size: wgpu::Extent3d {
                    width: n as u32,
                    height: stages as u32,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(&butterfly_data),
        );
        let butterfly_view = array_view(&butterfly_tex);

        // --- layouts ---
        let uniform = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let sampled = |b: u32, dim: wgpu::TextureViewDimension| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: dim,
                multisampled: false,
            },
            count: None,
        };
        let storage = |b: u32, format: wgpu::TextureFormat| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format,
                view_dimension: wgpu::TextureViewDimension::D2Array,
            },
            count: None,
        };
        let d2a = wgpu::TextureViewDimension::D2Array;
        let f32x4 = wgpu::TextureFormat::Rgba32Float;

        let spec_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ocean spectrum bgl"),
            entries: &[uniform(0), sampled(1, d2a), storage(2, f32x4)],
        });
        let stage_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ocean butterfly bgl"),
            entries: &[
                uniform(0),
                sampled(1, wgpu::TextureViewDimension::D2),
                sampled(2, d2a),
                storage(3, f32x4),
            ],
        });
        let asm_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ocean assemble bgl"),
            entries: &[
                sampled(0, d2a),
                storage(1, wgpu::TextureFormat::Rgba16Float),
            ],
        });

        let pipe = |src: &str, bgl: &wgpu::BindGroupLayout, label: &str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(bgl)],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let spectrum_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ocean spectrum params"),
            size: std::mem::size_of::<SpectrumParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let spectrum_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ocean spectrum bg"),
            layout: &spec_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: spectrum_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&h0_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&work_views[0]),
                },
            ],
        });

        // One uniform + bind group per (axis, stage). The ping-pong parity is
        // determined by the pass index, so each has exactly one source and one
        // destination and nothing has to be rebound mid-frame.
        let mut stage_params = Vec::new();
        let mut stage_bgs = Vec::new();
        for pass in 0..(2 * stages) {
            let vertical = (pass / stages) as u32;
            let stage = (pass % stages) as u32;
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ocean stage params"),
                contents: bytemuck::bytes_of(&StageParams {
                    stage,
                    vertical,
                    _pad: [0; 2],
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let src = pass % 2;
            let dst = 1 - src;
            stage_bgs.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ocean stage bg"),
                layout: &stage_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&butterfly_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&work_views[src]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&work_views[dst]),
                    },
                ],
            }));
            stage_params.push(buf);
        }

        let assemble_bgs = [0usize, 1].map(|src| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ocean assemble bg"),
                layout: &asm_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&work_views[src]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&disp_array),
                    },
                ],
            })
        });

        OceanFft {
            spectrum: pipe(SPECTRUM_WGSL, &spec_bgl, "ocean spectrum"),
            butterfly: pipe(BUTTERFLY_WGSL, &stage_bgl, "ocean butterfly"),
            assemble: pipe(ASSEMBLE_WGSL, &asm_bgl, "ocean assemble"),
            spectrum_bg,
            stage_bgs,
            assemble_bgs,
            spectrum_params,
            stage_params,
            cascade_views,
            modes,
            h0_tex,
            cas,
            stages,
            groups: ((n as u32).div_ceil(8), (n as u32).div_ceil(8)),
        }
    }

    /// Rebuild the cascades for time `t`: one spectrum pass, `2 log2(N)`
    /// butterfly passes, one assemble. Eighteen dispatches for three cascades,
    /// because all three ride as layers of one array texture.
    pub fn dispatch(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        t: f32,
        choppiness: f32,
    ) {
        let mut tiles = [0.0f32; 4];
        tiles[..CASCADES].copy_from_slice(&self.cas.tiles);
        queue.write_buffer(
            &self.spectrum_params,
            0,
            bytemuck::bytes_of(&SpectrumParams {
                time: t,
                choppiness,
                _pad: [0.0; 2],
                tiles,
            }),
        );

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ocean fft"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.spectrum);
            pass.set_bind_group(0, &self.spectrum_bg, &[]);
            pass.dispatch_workgroups(self.groups.0, self.groups.1, CASCADES as u32);

            pass.set_pipeline(&self.butterfly);
            for bg in &self.stage_bgs {
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(self.groups.0, self.groups.1, CASCADES as u32);
            }

            // 2*stages passes starting from work[0]: even count lands back in
            // work[0], odd in work[1].
            let final_src = (2 * self.stages) % 2;
            pass.set_pipeline(&self.assemble);
            pass.set_bind_group(0, &self.assemble_bgs[final_src], &[]);
            pass.dispatch_workgroups(self.groups.0, self.groups.1, CASCADES as u32);
        }
    }

    /// Re-seed the spectrum in place.
    ///
    /// Wind — both direction and speed — enters this ocean in exactly one place:
    /// `h0`. So changing it is a texture upload and a new mode table, and no
    /// pipeline, bind group or shader is touched. That is what makes turning the
    /// wind at runtime cost a fraction of a millisecond instead of a rebuild.
    pub fn set_spectrum(&mut self, queue: &wgpu::Queue, preset: &Preset, hs: f32) {
        let (data, modes) = build_h0(preset, hs, &self.cas);
        let n = self.cas.n as u32;
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.h0_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(n * 16),
                rows_per_image: Some(n),
            },
            wgpu::Extent3d {
                width: n,
                height: n,
                depth_or_array_layers: CASCADES as u32,
            },
        );
        self.modes = modes;
    }

    /// Vertical displacement and slope at one world point, from the dominant
    /// modes only. See [`Mode`].
    pub fn sample(&self, x: f32, z: f32, t: f32) -> (f32, [f32; 2]) {
        let mut y = 0.0f32;
        let mut dx = 0.0f32;
        let mut dz = 0.0f32;
        for m in &self.modes {
            let (s, c) = (m.omega * t).sin_cos();
            // h(k,t), then the real part of h e^(i k.x) — doubled, because the
            // conjugate half-plane was dropped when the modes were picked.
            let hr = m.h0[0] * c - m.h0[1] * s + m.h0c[0] * c + m.h0c[1] * s;
            let hi = m.h0[0] * s + m.h0[1] * c - m.h0c[0] * s + m.h0c[1] * c;
            let phase = m.k[0] * x + m.k[1] * z;
            let (ps, pc) = phase.sin_cos();
            y += 2.0 * (hr * pc - hi * ps);
            let d = -2.0 * (hr * ps + hi * pc);
            dx += d * m.k[0];
            dz += d * m.k[1];
        }
        (y, [dx, dz])
    }
}

/// Build `h0` for every cascade, and pick the modes buoyancy will use.
///
/// Returns the texture data as `[h0.re, h0.im, conj(h0(-k)).re, conj(h0(-k)).im]`
/// per texel, layer-major.
fn build_h0(p: &Preset, hs: f32, cas: &Cascades) -> (Vec<[f32; 4]>, Vec<Mode>) {
    let n = cas.n;
    let edges = band_edges(cas);
    let u = p.wind_speed.max(0.5);
    let peak_wavelength = if p.peak_wavelength > 0.0 {
        p.peak_wavelength
    } else {
        let w_p = 0.877 * G / u;
        2.0 * std::f32::consts::PI * G / (w_p * w_p)
    };
    let omega_peak = (G * (std::f32::consts::TAU / peak_wavelength)).sqrt();

    // Deterministic, so every machine builds the same sea — the same property
    // the analytic spectrum had, and for the same reason.
    let mut seed = 0x0CEA_11FEu32;
    let mut next = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 8) as f32 / 16_777_216.0
    };
    // Box–Muller: the sea surface is Gaussian, and using uniform noise here is
    // the difference between water and a moiré pattern.
    let mut gauss = move || {
        let u1 = next().max(1e-7);
        let u2 = next();
        let r = (-2.0 * u1.ln()).sqrt();
        let th = std::f32::consts::TAU * u2;
        (r * th.cos(), r * th.sin())
    };

    let mut data = vec![[0.0f32; 4]; n * n * CASCADES];
    // Amplitude of every mode, before the one global scale that sets H_s.
    let mut variance = 0.0f64;

    for c in 0..CASCADES {
        let l = cas.tiles[c];
        let dk = std::f32::consts::TAU / l;
        let lo = if c == 0 { 0.0 } else { edges[c - 1] };
        let hi = edges[c];

        for y in 0..n {
            for x in 0..n {
                let kx = (x as f32 - n as f32 / 2.0) * dk;
                let kz = (y as f32 - n as f32 / 2.0) * dk;
                let kmag = (kx * kx + kz * kz).sqrt();
                let (g1, g2) = gauss();

                // Each cascade owns one band. Outside it, this mode belongs to a
                // neighbour and must be zero here or it would be counted twice.
                if kmag < 1e-6 || kmag < lo || kmag >= hi {
                    continue;
                }

                let omega = (G * kmag).sqrt();
                // JONSWAP, as a function of frequency.
                let sigma = if omega <= omega_peak { 0.07 } else { 0.09 };
                let r = (-((omega - omega_peak).powi(2))
                    / (2.0 * sigma * sigma * omega_peak * omega_peak))
                    .exp();
                let pm =
                    0.0081 * G * G / omega.powi(5) * (-1.25 * (omega_peak / omega).powi(4)).exp();
                let s_omega = pm * 3.3f32.powf(r);

                // To a wavenumber spectrum: S(k) = S(w) * (dw/dk) / k.
                let s_k = s_omega * (0.5 * (G / kmag).sqrt()) / kmag;

                // Directional spread, cos^2 about the wind, and zero upwind.
                let theta = kz.atan2(kx) - p.wind_dir;
                let spread = theta.cos();
                let d = if spread > 0.0 { spread * spread } else { 0.02 };

                let amp = (s_k * d).max(0.0).sqrt() * dk * std::f32::consts::FRAC_1_SQRT_2;
                let h0 = [g1 * amp, g2 * amp];
                variance += 2.0 * (h0[0] * h0[0] + h0[1] * h0[1]) as f64;

                let i = (c * n * n) + y * n + x;
                data[i][0] = h0[0];
                data[i][1] = h0[1];
            }
        }
    }

    // One global scale so the field carries the significant wave height the wind
    // implies: H_s = 4 sqrt(m0), with m0 the variance summed above. The absolute
    // normalisation of the spectrum never has to be right — only its shape.
    let scale = if variance > 1e-20 {
        hs / (4.0 * (variance.sqrt() as f32))
    } else {
        0.0
    };
    for d in data.iter_mut() {
        d[0] *= scale;
        d[1] *= scale;
    }

    // conj(h0(-k)) for each mode, now that every h0 is final.
    let mut modes = Vec::new();
    for c in 0..CASCADES {
        for y in 0..n {
            for x in 0..n {
                let mx = (n - x) % n;
                let my = (n - y) % n;
                let i = (c * n * n) + y * n + x;
                let j = (c * n * n) + my * n + mx;
                data[i][2] = data[j][0];
                data[i][3] = -data[j][1];
            }
        }
    }

    // Buoyancy reads the swell cascade only, and only its strongest modes: a
    // buoy rides the long waves, and summing 65 536 of them per query point on
    // the CPU would cost more than the entire rest of the frame.
    let dk = std::f32::consts::TAU / cas.tiles[0];
    let mut candidates: Vec<(f32, usize, usize)> = Vec::new();
    for y in 0..n {
        for x in 0..n {
            // One half-plane only: the other is the conjugate, already folded in
            // by the doubling in `sample`.
            if (y as f32 - n as f32 / 2.0) < 0.0 {
                continue;
            }
            let i = y * n + x;
            let p2 = data[i][0] * data[i][0] + data[i][1] * data[i][1];
            if p2 > 0.0 {
                candidates.push((p2, x, y));
            }
        }
    }
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    for &(_, x, y) in candidates.iter().take(96) {
        let kx = (x as f32 - n as f32 / 2.0) * dk;
        let kz = (y as f32 - n as f32 / 2.0) * dk;
        let kmag = (kx * kx + kz * kz).sqrt();
        let i = y * n + x;
        modes.push(Mode {
            k: [kx, kz],
            omega: (G * kmag).sqrt(),
            h0: [data[i][0], data[i][1]],
            h0c: [data[i][2], data[i][3]],
        });
    }

    (data, modes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cascades_partition_the_spectrum() {
        let cas = Cascades::default();
        let e = band_edges(&cas);
        // Each cascade hands over at its own Nyquist, ascending, so the bands
        // abut without a gap or an overlap.
        assert!(e[0] < e[1] && e[1] < e[2]);
        for (i, l) in cas.tiles.iter().enumerate() {
            assert!((e[i] - std::f32::consts::PI * cas.n as f32 / l).abs() < 1e-3);
        }
        // And the finest cascade reaches well past the coarsest.
        assert!(e[2] / e[0] > 50.0);
    }

    #[test]
    fn spectrum_carries_the_requested_wave_height() {
        let p = crate::preset::all()[1];
        let hs = 1.44f32;
        let (data, modes) = build_h0(&p, hs, &Cascades::default());
        // H_s = 4 sqrt(m0), m0 = sum of 2|h0|^2 over every mode.
        let m0: f64 = data
            .iter()
            .map(|d| 2.0 * (d[0] * d[0] + d[1] * d[1]) as f64)
            .sum();
        let got = 4.0 * m0.sqrt() as f32;
        assert!((got - hs).abs() < 1e-2, "H_s {got} != {hs}");
        assert!(!modes.is_empty(), "buoyancy needs modes");
    }

    #[test]
    fn hermitian_pair_is_the_conjugate() {
        let (data, _) = build_h0(&crate::preset::all()[3], 8.0, &Cascades::default());
        // data[i].zw must be conj(h0(-k)); if it is not, the inverse transform
        // comes out complex and the surface gets an imaginary height.
        for &(x, y) in &[(10usize, 20usize), (200, 33), (0, 0), (128, 128)] {
            let i = y * N + x;
            let j = ((N - y) % N) * N + ((N - x) % N);
            assert!((data[i][2] - data[j][0]).abs() < 1e-12);
            assert!((data[i][3] + data[j][1]).abs() < 1e-12);
        }
    }
}
