//! Spray, rain and underwater motes — one GPU particle system, three spawn rules.
//!
//! threers' `Points` material is unlit and opaque, which gives hard square dots.
//! What these need is a soft round sprite that fades as it dies, so instead the
//! simulation writes **camera-facing quads** straight into a
//! [`gpu_writable`](threers::BufferGeometry::gpu_writable) mesh — the same trick
//! the water surface uses — and a transparent [`ShaderMaterial`] shades them.
//!
//! The particles never touch the CPU. State lives in a `read_write` storage
//! buffer, one compute pass integrates it and emits the quads, and the only
//! per-frame CPU work is sixty bytes of parameters.
//!
//! Spray is the interesting one: it spawns where the *surface* is breaking, which
//! it finds by sampling the same cascades everything else reads. So the spray is
//! thrown off the waves that are actually there rather than scattered over the
//! sea at random.

use std::sync::Arc;

use threers::{
    BufferAttribute, BufferGeometry, Material, Mesh, Object3D, ObjectId, Renderer, Scene,
    ShaderMaterial, Vector3,
};

use crate::ocean_fft::OceanFft;

/// What a system throws around, and where it gets it from.
#[derive(Clone, Copy, PartialEq)]
pub enum Kind {
    /// Torn off breaking crests, thrown downwind, pulled back by gravity.
    Spray = 0,
    /// Falls from above the camera, wrapping when it lands.
    Rain = 1,
    /// Neutrally buoyant specks, drifting. Only visible below the waterline.
    Motes = 2,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    camera: [f32; 3],
    dt: f32,
    right: [f32; 3],
    time: f32,
    up: [f32; 3],
    kind: u32,
    wind: [f32; 2],
    size: f32,
    count: u32,
    /// Half-extent of the box particles live in, metres.
    extent: f32,
    /// Fall or launch speed, m/s.
    speed: f32,
    /// Elongation along the direction of travel. 1 is a round sprite.
    stretch: f32,
    _pad: f32,
}

const SHADER: &str = r#"
struct Params {
    camera : vec3<f32>,
    dt     : f32,
    right  : vec3<f32>,
    time   : f32,
    up     : vec3<f32>,
    kind   : u32,
    wind   : vec2<f32>,
    size   : f32,
    count  : u32,
    extent : f32,
    speed  : f32,
    stretch: f32,
};

@group(0) @binding(0) var<uniform> params : Params;
// Two vec4 per particle: position + life, velocity + seed.
@group(0) @binding(1) var<storage, read_write> state : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> verts : array<f32>;
@group(0) @binding(3) var cascade_samp : sampler;
@group(0) @binding(4) var cascade_a : texture_2d<f32>;
@group(0) @binding(5) var cascade_b : texture_2d<f32>;
@group(0) @binding(6) var cascade_c : texture_2d<f32>;

const SPRAY: u32 = 0u;
const RAIN:  u32 = 1u;
const MOTES: u32 = 2u;

$CASCADES

fn hash(n: u32) -> f32 {
    var x = n * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return f32((x >> 22u) ^ x) / 4294967296.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.count) { return; }

    var pos_life = state[i * 2u];
    var vel_seed = state[i * 2u + 1u];
    var p = pos_life.xyz;
    var life = pos_life.w;
    var v = vel_seed.xyz;
    let seed = u32(vel_seed.w);

    // Stagger the respawns: seeding every particle from the same frame counter
    // makes the whole system blink in and out together.
    let r0 = hash(seed + u32(params.time * 60.0));
    let r1 = hash(seed * 7919u + u32(params.time * 60.0) + 13u);
    let r2 = hash(seed * 104729u + u32(params.time * 60.0) + 71u);

    life = life - params.dt;
    if (life <= 0.0) {
        // Respawn, in a box that follows the viewer: a particle you cannot see
        // is one you should not be simulating.
        let ox = (r0 - 0.5) * 2.0 * params.extent;
        let oz = (r1 - 0.5) * 2.0 * params.extent;
        if (params.kind == RAIN) {
            p = vec3<f32>(params.camera.x + ox, params.camera.y + params.extent * 0.8, params.camera.z + oz);
            v = vec3<f32>(params.wind.x, -params.speed, params.wind.y);
            life = 1.0 + r2 * 0.6;
        } else if (params.kind == MOTES) {
            p = params.camera + vec3<f32>(ox, (r2 - 0.5) * 2.0 * params.extent, oz);
            v = vec3<f32>(params.wind.x * 0.02, -0.03, params.wind.y * 0.02);
            life = 3.0 + r2 * 4.0;
        } else {
            // Spray: look at the surface where this particle would land, and
            // only launch if it is actually breaking there. A short life on a
            // miss means it retries next frame rather than stalling.
            let world = vec2<f32>(params.camera.x + ox, params.camera.z + oz);
            let s = sample_surface(world, 1.0);
            if (s.fold > 0.16 && s.disp.y > 0.0) {
                p = vec3<f32>(world.x + s.disp.x, s.disp.y, world.y + s.disp.z);
                // Off the crest: up, and downwind.
                v = vec3<f32>(params.wind.x * 1.4 + (r2 - 0.5) * 2.0,
                              params.speed * (0.5 + r2),
                              params.wind.y * 1.4 + (r0 - 0.5) * 2.0);
                life = 0.9 + r1 * 1.1;
            } else {
                life = 0.0;
                p = vec3<f32>(0.0, -1.0e5, 0.0);
            }
        }
    } else {
        p = p + v * params.dt;
        if (params.kind == SPRAY) {
            v.y = v.y - 9.81 * params.dt;
        }
    }

    state[i * 2u] = vec4<f32>(p, life);
    state[i * 2u + 1u] = vec4<f32>(v, f32(seed));

    // Fade in as it is born and out as it dies, so nothing pops.
    let alpha = clamp(min(life * 3.0, 1.0), 0.0, 1.0);
    // A dead particle collapses to a degenerate quad rather than being culled:
    // the index buffer is fixed, so there is nothing to compact.
    let size = select(params.size, 0.0, life <= 0.0);

    // Camera-facing quad. The vertex layout is threers' interleaved 16 floats:
    // position(3) normal(3) uv(2) colour(4) tangent(4).
    // `var`, not `let`: WGSL only allows a dynamic index into a variable.
    // `var`, not `let`: WGSL only allows a dynamic index into a variable.
    var corners = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0));
    // A falling drop is a streak, not a dot: the eye integrates its motion over
    // the exposure. Stretching the quad along the screen projection of the
    // velocity is that, and it costs one normalisation.
    var ax = params.right;
    var ay = params.up;
    if (params.stretch > 1.001) {
        let sv = vec2<f32>(dot(v, params.right), dot(v, params.up));
        let l = length(sv);
        if (l > 1.0e-3) {
            let d2 = sv / l;
            ay = params.right * d2.x + params.up * d2.y;
            ax = params.right * d2.y - params.up * d2.x;
        }
    }

    for (var c = 0u; c < 4u; c = c + 1u) {
        let q = corners[c];
        let world = p + (ax * q.x + ay * q.y * params.stretch) * size;
        let o = (i * 4u + c) * 16u;
        verts[o] = world.x;
        verts[o + 1u] = world.y;
        verts[o + 2u] = world.z;
        verts[o + 3u] = 0.0;
        verts[o + 4u] = 1.0;
        verts[o + 5u] = 0.0;
        verts[o + 6u] = q.x * 0.5 + 0.5;
        verts[o + 7u] = q.y * 0.5 + 0.5;
        verts[o + 8u] = 1.0;
        verts[o + 9u] = 1.0;
        verts[o + 10u] = 1.0;
        verts[o + 11u] = alpha;
    }
}
"#;

/// Soft round sprite. Unlit on purpose — spray and rain are scattering, not
/// shaded surfaces, and lighting them makes them read as beads.
const FRAGMENT: &str = r#"
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Corner offset from the uv, fade from the vertex colour's alpha.
    let d = length(in.uv - vec2<f32>(0.5)) * 2.0;
    // Square the falloff: a linear edge reads as a ring.
    let a = in.vertex_color.a * pow(clamp(1.0 - d, 0.0, 1.0), 2.0);
    if (a < 0.004) { discard; }
    let tint = u_data.data[0].xyz;
    return vec4<f32>(framebuffer_encode(tint), a * u_data.data[0].w);
}
"#;

pub struct Particles {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    params: wgpu::Buffer,
    count: u32,
    kind: Kind,
    extent: f32,
    speed: f32,
    stretch: f32,
    size: f32,
    pub object: ObjectId,
    geometry: Arc<BufferGeometry>,
    /// The version the vertex buffer was bound at. Anything that bumps it makes
    /// the renderer allocate a fresh buffer, leaving this pass writing the old
    /// one — the particles would silently freeze rather than error.
    bound_version: u32,
}

impl Particles {
    /// Build a system of `count` particles and add its mesh to the scene.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        renderer: &mut Renderer,
        scene: &mut Scene,
        fft: &OceanFft,
        kind: Kind,
        count: u32,
        size: f32,
        extent: f32,
        speed: f32,
        stretch: f32,
        tint: [f32; 4],
        peak_wavelength: f32,
    ) -> Particles {
        // Four vertices and six indices per particle, uploaded once. Only the
        // positions change, and those are written on the GPU.
        let verts = count as usize * 4;
        let mut geometry = BufferGeometry::new();
        geometry.set_attribute("position", BufferAttribute::new(vec![0.0; verts * 3], 3));
        geometry.set_attribute("normal", BufferAttribute::new(vec![0.0; verts * 3], 3));
        geometry.set_attribute("uv", BufferAttribute::new(vec![0.0; verts * 2], 2));
        geometry.set_attribute("color", BufferAttribute::new(vec![0.0; verts * 4], 4));
        let mut index = Vec::with_capacity(count as usize * 6);
        for i in 0..count {
            let b = i * 4;
            index.extend_from_slice(&[b, b + 1, b + 2, b, b + 2, b + 3]);
        }
        geometry.set_index(index);
        geometry.gpu_writable = true;

        let geometry = Arc::new(geometry);
        let material = ShaderMaterial::new(FRAGMENT)
            .with_data(vec![tint])
            .with_transparent(true)
            // Double-sided: the quads are built facing the camera, but a
            // one-frame-stale basis would flicker them out at grazing angles.
            .with_side(2);
        let object = scene.add(Object3D::mesh(Mesh::from_arc(
            geometry.clone(),
            Arc::new(Material::Shader(material)),
        )));
        renderer.upload_geometry(&geometry);
        let vertex_buffer = renderer
            .vertex_buffer(&geometry)
            .expect("particle geometry must be uploaded first");

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
            label: Some("ocean particles"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Seed each particle with its index and stagger the initial lives, so a
        // system does not begin by emitting everything at once.
        let mut init = vec![0.0f32; count as usize * 8];
        for i in 0..count as usize {
            init[i * 8 + 1] = -1.0e5;
            init[i * 8 + 3] = -(i as f32 % 97.0) * 0.01;
            init[i * 8 + 7] = i as f32 + 1.0;
        }
        let state = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("particle state"),
                contents: bytemuck::cast_slice(&init),
                usage: wgpu::BufferUsages::STORAGE,
            },
        );

        let storage = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
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
            label: Some("particle bgl"),
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
                storage(1),
                storage(2),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                tex(4),
                tex(5),
                tex(6),
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("particle cascade sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle bg"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: state.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler),
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
            label: Some("particle pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("particle pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Particles {
            pipeline,
            bind_group,
            params,
            count,
            kind,
            extent,
            speed,
            stretch,
            size,
            object,
            bound_version: geometry.geometry_version,
            geometry,
        }
    }

    pub fn kind(&self) -> Kind {
        self.kind
    }

    /// Integrate one step and rewrite the quads. `right`/`up` are the camera's
    /// screen axes in world space, which is what makes the sprites face it.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        camera: Vector3,
        right: Vector3,
        up: Vector3,
        wind: [f32; 2],
        dt: f32,
        time: f32,
    ) {
        debug_assert_eq!(
            self.geometry.geometry_version, self.bound_version,
            "particle geometry was re-uploaded behind the compute pass"
        );
        queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&Params {
                camera: [camera.x, camera.y, camera.z],
                dt: dt.clamp(0.0, 0.1),
                right: [right.x, right.y, right.z],
                time,
                up: [up.x, up.y, up.z],
                kind: self.kind as u32,
                wind,
                size: self.size,
                count: self.count,
                extent: self.extent,
                speed: self.speed,
                stretch: self.stretch,
                _pad: 0.0,
            }),
        );

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("particle pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.count.div_ceil(64), 1, 1);
        }
    }
}
