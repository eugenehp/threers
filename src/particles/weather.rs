//! Weather on the GPU: snow or rain filling a box of air, every flake or drop
//! worked out in the vertex shader from its index and the time — nothing
//! simulated on the CPU, nothing uploaded per frame. A hundred thousand cost
//! what a hundred thousand quads do to draw.
//!
//! What makes this possible is that falling weather needs no memory: a flake
//! at terminal speed in a steady wind is where its start and the time put it.
//! Each instance hashes its index into a start in the box, a fall speed, a
//! size and a phase; falls at that speed, carried by the wind and by a gentle
//! eddying sway (smaller and faster for rain); and wraps back to the top and
//! round the sides as it leaves, fading at the box's faces so nothing pops.
//! Drawn by the particle shader, so a flake is the same six-armed, lit,
//! glinting flake and a drop the same streak as the simulated ones.
//!
//! What it cannot do is land: for snow that piles up, pair it with a
//! [`presets::snowfall`](super::presets::snowfall) of a few thousand
//! simulated flakes settling on a [`SnowCover`](super::snow::SnowCover) —
//! the weather for the look, the simulation for the snow on the ground.
//!
//! ```no_run
//! use threers::particles::weather::Weather;
//! use threers::math::Vector3;
//! # let mut scene = threers::Scene::new();
//! let snow = Weather::snow(80_000);
//! let id = scene.add(snow.object());
//! // Each frame: the time, and where the box is centred (the camera).
//! snow.update(&mut scene, id, 1.5, Vector3::new(0.0, 0.0, 5.0));
//! ```

use std::sync::Arc;

use crate::core::{InstancedMesh, Object3D, ObjectId, ObjectKind};
use crate::geometries::PlaneGeometry;
use crate::materials::Material;
use crate::math::{Color, Matrix4, Vector3};
use crate::scene::Scene;

use super::{ParticleBlend, ParticleKind, ParticleLook};

/// Snow or rain filling a box of air round a point.
#[derive(Debug, Clone, PartialEq)]
pub struct Weather {
    /// Flakes or drops.
    pub count: u32,
    /// Half the box's width (x) and depth (z), and its height, metres.
    pub half_width: f32,
    pub half_depth: f32,
    pub height: f32,
    /// The box's floor: flakes fall from `floor + height` to here.
    pub floor: f32,
    /// Fall speeds, m/s (terminal speeds: ~1 for snow, ~9 for rain).
    pub fall: [f32; 2],
    /// Drawn sizes, metres.
    pub size: [f32; 2],
    /// The wind, m/s.
    pub wind: Vector3,
    /// The sway of eddies, metres either way.
    pub sway: f32,
    /// Spin, radians a second.
    pub spin: f32,
    /// Colour (linear) and opacity.
    pub color: Color,
    pub opacity: f32,
    pub look: ParticleLook,
}

impl Weather {
    /// Falling snow: flakes about 1 m/s, swaying and turning, lit, six-armed
    /// and glinting, in a 30 m × 30 m × 12 m box.
    pub fn snow(count: u32) -> Self {
        Self {
            count,
            half_width: 15.0,
            half_depth: 15.0,
            height: 12.0,
            floor: 0.0,
            fall: [0.7, 1.3],
            size: [0.012, 0.028],
            wind: Vector3::new(0.4, 0.0, 0.1),
            sway: 0.25,
            spin: 2.5,
            color: Color::new(1.0, 1.0, 1.0),
            opacity: 0.95,
            look: ParticleLook {
                lit: true,
                flake: true,
                glint: 1.0,
                scattering: 0.3,
                plates: 0.12,
                ..ParticleLook::new(
                    ParticleKind::Billboard { softness: 0.7 },
                    ParticleBlend::Alpha,
                )
            },
        }
    }

    /// Rain: drops at about 9 m/s, streaked as a camera sees them.
    pub fn rain(count: u32) -> Self {
        Self {
            count,
            half_width: 15.0,
            half_depth: 15.0,
            height: 12.0,
            floor: 0.0,
            fall: [8.0, 9.5],
            size: [0.006, 0.01],
            wind: Vector3::new(1.0, 0.0, 0.0),
            sway: 0.02,
            spin: 0.0,
            color: Color::new(0.75, 0.8, 0.88),
            opacity: 0.55,
            look: ParticleLook {
                lit: true,
                rainbow: 1.0,
                ..ParticleLook::new(
                    ParticleKind::Streak {
                        softness: 0.8,
                        shutter: 1.0 / 50.0,
                    },
                    ParticleBlend::Alpha,
                )
            },
        }
    }

    /// The object drawing it: `count` instances of a quad, placed by the
    /// shader. Its instances never change, so they are uploaded once.
    pub fn object(&self) -> Object3D {
        // The particle look's fragment (its constants and shading), with
        // the weather's own vertex stage.
        let mut material = self.look.material();
        material.vertex = Some(WEATHER_VERTEX_WGSL.to_string());
        material.data = self.data(0.0, Vector3::ZERO);
        let mut mesh = InstancedMesh::from_arc(
            Arc::new(PlaneGeometry::new(1.0, 1.0)),
            Arc::new(Material::Shader(material)),
            self.count as usize,
        );
        mesh.transforms = vec![Matrix4::identity(); self.count as usize];
        mesh.instance_version = Some(1);
        let mut o = Object3D::instanced_mesh(mesh);
        o.cast_shadow = false;
        o.receive_shadow = self.look.lit;
        o.name = "weather".into();
        o.render_order = 10;
        o
    }

    fn data(&self, time: f32, centre: Vector3) -> Vec<[f32; 4]> {
        vec![
            [time, self.sway, self.spin, self.opacity],
            [self.half_width, self.height, self.half_depth, self.floor],
            [centre.x, centre.z, self.fall[0], self.fall[1]],
            [self.wind.x, self.wind.y, self.wind.z, 0.0],
            [self.size[0], self.size[1], 0.0, 0.0],
            [self.color.r, self.color.g, self.color.b, 0.0],
        ]
    }

    /// Advance `id` (this weather's object) to `time` seconds, its box
    /// centred over `centre` (x and z; usually the camera, so it is always
    /// snowing where the camera looks).
    pub fn update(&self, scene: &mut Scene, id: ObjectId, time: f32, centre: Vector3) {
        if let Some(o) = scene.get_mut(id) {
            if let ObjectKind::InstancedMesh(mesh) = &mut o.kind {
                if let Material::Shader(m) = &*mesh.material {
                    let mut m = m.clone();
                    m.data = self.data(time, centre);
                    mesh.material = Arc::new(Material::Shader(m));
                }
            }
        }
    }
}

/// Each instance's flake, from its index and the time: a start in the box,
/// a fall speed, a size and a phase, all hashed; falling, carried by the
/// wind, swaying; wrapped round the box (centred on `data[2].xy`), and faded
/// near its faces. Then the same camera-facing quad as a particle's.
pub(crate) const WEATHER_VERTEX_WGSL: &str = r#"
fn weather_hash(n : f32) -> f32 {
    return fract(sin(n * 12.9898 + 78.233) * 43758.5453);
}

@vertex
fn vs_custom(in : InstancedCustomIn, @builtin(vertex_index) vid : u32,
             @builtin(instance_index) iid : u32) -> VsOut {
    var out : VsOut;
    let d0 = u_data.data[0];
    let box = u_data.data[1];
    let c = u_data.data[2];
    let wind = u_data.data[3].xyz;
    let t = d0.x;
    let i = f32(iid);
    let h1 = weather_hash(i * 1.371);
    let h2 = weather_hash(i * 2.913 + 7.0);
    let h3 = weather_hash(i * 4.177 + 13.0);
    let h4 = weather_hash(i * 0.613 + 29.0);
    let h5 = weather_hash(i * 3.311 + 41.0);
    let fall = mix(c.z, c.w, h4);
    // How far down its fall it is, wrapping to the top.
    let down = fract(h2 + t * fall / box.y);
    // Carried by the wind and swaying in the eddies, then wrapped round the
    // box about its centre.
    let sway = vec2<f32>(sin(t * (1.1 + h5) + h1 * 31.0), cos(t * (0.9 + h3) + h2 * 17.0)) * d0.y;
    let span = vec2<f32>(box.x, box.z) * 2.0;
    let start = vec2<f32>(h1, h3) * span;
    let moved = start + vec2<f32>(wind.x, wind.z) * t + sway - (c.xy - box.xz);
    let wrapped = moved - floor(moved / span) * span;
    let centre = vec3<f32>(c.x - box.x + wrapped.x, box.w + box.y * (1.0 - down), c.y - box.z + wrapped.y);
    // Fading in at the top and out at the box's sides, so a wrap is never
    // seen; and out right by the eye, where one flake would fill the view.
    let edge = min(min(wrapped.x, span.x - wrapped.x) / span.x, min(wrapped.y, span.y - wrapped.y) / span.y);
    let near = distance(centre, frame.camera_position.xyz);
    let fade = smoothstep(0.0, 0.08, edge) * smoothstep(0.0, 0.05, down) * smoothstep(0.0, 0.02, 1.0 - down)
        * smoothstep(0.3, 1.0, near);
    let size = mix(u_data.data[4].x, u_data.data[4].y, h5);
    let velocity = vec3<f32>(wind.x, -fall, wind.z);

    let right = vec3<f32>(frame.view[0][0], frame.view[1][0], frame.view[2][0]);
    let up = vec3<f32>(frame.view[0][1], frame.view[1][1], frame.view[2][1]);
    let to_cam = normalize(frame.camera_position.xyz - centre);
    let q = in.position.xy;
    var ax = right;
    var ay = up;
    var height = size;
    var shift = vec3<f32>(0.0);
    if (P_KIND == 1u) {
        let across = velocity - to_cam * dot(velocity, to_cam);
        let len = length(across) * P_SHUTTER;
        if (len > 1e-5) {
            ay = across / length(across);
            ax = normalize(cross(ay, to_cam));
            height = size + len;
            shift = -ay * (len * 0.5);
        }
    } else {
        let a = h2 * 6.2831853 + t * d0.z * (h4 - 0.5);
        let cs = cos(a);
        let sn = sin(a);
        ax = right * cs + up * sn;
        ay = up * cs - right * sn;
    }
    out.world_pos = centre + shift + ax * (q.x * size) + ay * (q.y * height);
    out.clip_pos = frame.view_proj * vec4<f32>(out.world_pos, 1.0);
    out.clip_zw = out.clip_pos.zw;
    out.view_z = -(frame.view * vec4<f32>(out.world_pos, 1.0)).z;
    out.uv = in.uv;
    out.vertex_color = vec4<f32>(u_data.data[5].rgb, d0.w * fade);
    out.world_normal = to_cam;
    out.world_tangent = vec4<f32>(ax, h2 * 6.2831853 + t * d0.z * (h4 - 0.5));
    out.local_pos = ay;
    let px = size * 0.5 * frame.projection[1][1] / max(out.clip_pos.w, 1e-4) * frame.viewport_size.y * 0.5;
    out.proj_uv = vec4<f32>(0.0, h3, size * 0.5, px);
    out.uv12 = vec4<f32>(centre + shift, 0.0);
    return out;
}
"#;
