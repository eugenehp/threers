//! Drawing particles: one instanced draw of camera-facing quads.
//!
//! Each particle is an instance: its centre and size in the instance matrix,
//! its colour in the instance tint, and in the two custom columns its
//! velocity and turn (for streaks and spin) and its opacity, life and a random
//! seed. The vertex stage turns the unit quad to face the camera — or, for a
//! [`ParticleKind::Streak`], stretches it along the velocity across the view
//! by the distance covered in a shutter time, the motion blur rain and sparks
//! are seen as — and the fragment stage cuts a soft disc or shades a lit
//! sphere, so a grain of sand is round and lit from the sun's side.
//!
//! Alpha-blended particles are drawn back to front from the eye given to
//! [`write_instances`]; additive ones need no order.

use std::sync::Arc;

use crate::core::{InstancedMesh, Object3D, ObjectId, ObjectKind};
use crate::geometries::PlaneGeometry;
use crate::materials::{Blending, Material, RenderState};
use crate::math::{Color, Matrix4, Vector3};
use crate::scene::Scene;
use crate::ShaderMaterial;

use super::{Heightfield, ParticleSystem};

/// The shape each particle is drawn as.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParticleKind {
    /// A disc facing the camera, its edge softened by `softness` (0: a hard
    /// edge; 1: a gentle fall-off; more: a glow).
    Billboard { softness: f32 },
    /// A disc stretched along the particle's motion across the view, by the
    /// distance it covers in `shutter` seconds: rain, sparks.
    Streak { softness: f32, shutter: f32 },
    /// A lit sphere: grains, hail, pellets.
    Sphere,
    /// Not drawn but seen through: what is behind it bent by a shimmer of
    /// `strength` (a share of the view's width) — the heat haze over a fire
    /// or an exhaust. Drawn in the screen-space pass, over the finished
    /// opaque scene.
    Distortion { strength: f32 },
}

/// How drawn particles combine with what is behind them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParticleBlend {
    /// Alpha blending, back to front: smoke, snow, dust.
    Alpha,
    /// Light added to what is behind, no order needed: fire, sparks, magic.
    Additive,
    /// Solid and depth-written, cut where the disc ends: sand, hail.
    Opaque,
}

/// How a system's particles look.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParticleLook {
    pub kind: ParticleKind,
    pub blend: ParticleBlend,
    /// Multiplies the over-life colour. Above 1 for things that glow (fire,
    /// sparks), which the renderer's tone mapping then rolls off.
    pub intensity: f32,
    /// Shaded by the scene's lights (and shadowed, if the object receives
    /// shadows); otherwise the colour is drawn as it is.
    pub lit: bool,
    /// How much a flat particle is a billow rather than a disc, 0 to 1:
    /// three octaves of noise, its own for each particle, eat into it and
    /// drift as it ages, and thin it away as it dies — smoke and dust as
    /// puffs, flame as tongues and wisps.
    pub detail: f32,
    /// A billboard stretched along its motion across the view, by its speed
    /// times this many seconds (flame licking upward). 0: round.
    pub stretch: f32,
    /// How a lit flat particle scatters light: the Henyey–Greenstein
    /// asymmetry `g`, from -1 (back toward the light) through 0 (every way
    /// alike) to 1 (straight on). Smoke and dust scatter forward (0.3–0.6),
    /// so they glow when the sun is behind them.
    pub scattering: f32,
    /// How much a billow shades itself, 0 to 1: its own density, looked up
    /// toward the light, dims its far side — the dark undersides of smoke.
    pub self_shadow: f32,
    /// Drawn as a six-armed snowflake rather than a disc.
    pub flake: bool,
    /// Glints, 0 and up: each particle (a flake, a grain) carries a tiny
    /// mirror at its own angle that flashes when it catches a light.
    pub glint: f32,
    /// Casts shadows: discs facing the light, as wide as the particle and
    /// shrunk by its opacity, drawn into the shadow maps (and nowhere else).
    /// For smoke, dust, sand; leave it off for fifty thousand snowflakes.
    pub casts_shadow: bool,
    /// Extinction, per metre, inside a particle at full opacity: 0 draws a
    /// soft disc. Above 0 a lit particle is a little sphere of participating
    /// medium — its opacity Beer–Lambert's `1 − e^(−σ·chord)` along each
    /// line of sight through it, and the sun and the sky reaching it dimmed
    /// by the optical depth of the system's other particles in the way
    /// (worked out each frame in a grid lined up with the light), with
    /// light scattered more than once filling in the shadowed side. Smoke
    /// 2–6, dust 1–3.
    pub extinction: f32,
    /// Plates: a flake's glinting face lies level, tilted by its flutter by
    /// up to this many radians, rather than at random — as falling plate
    /// crystals settle broadside to the air. Their glints then line up into
    /// pillars above and below a light. 0: every way at random.
    pub plates: f32,
    /// Raindrops' rainbow, 0 and up: each drop sends the sun's light back
    /// at the angles a sphere of water does — the primary bow, red at 42.3°
    /// and violet at 40.6° from the point opposite the sun, bright inside;
    /// Alexander's dark band; the fainter secondary, colours reversed, from
    /// 50.4° to 53.5°.
    pub rainbow: f32,
}

impl ParticleLook {
    pub const fn new(kind: ParticleKind, blend: ParticleBlend) -> Self {
        Self {
            kind,
            blend,
            intensity: 1.0,
            lit: false,
            detail: 0.0,
            stretch: 0.0,
            scattering: 0.0,
            self_shadow: 0.0,
            flake: false,
            glint: 0.0,
            casts_shadow: false,
            extinction: 0.0,
            plates: 0.0,
            rainbow: 0.0,
        }
    }

    /// The material drawing it.
    pub fn material(&self) -> ShaderMaterial {
        let (kind, softness, shutter) = match self.kind {
            ParticleKind::Billboard { softness } => (0.0, softness, 0.0),
            ParticleKind::Streak { softness, shutter } => (1.0, softness, shutter),
            ParticleKind::Sphere => (2.0, 0.0, 0.0),
            ParticleKind::Distortion { strength } => (3.0, strength, 0.0),
        };
        let distortion = matches!(self.kind, ParticleKind::Distortion { .. });
        let blending = match self.blend {
            ParticleBlend::Additive => Blending::Additive,
            _ => Blending::Normal,
        };
        let opaque = self.blend == ParticleBlend::Opaque;
        // Everything about the look, compiled in: each look is its own
        // pipeline, with no per-pixel tests of what never changes and the
        // code it does not use folded away.
        let consts = format!(
            "const P_KIND : u32 = {kind}u;\n\
             const P_SOFTNESS : f32 = {softness:?};\n\
             const P_SHUTTER : f32 = {shutter:?};\n\
             const P_LIT : bool = {lit};\n\
             const P_SCATTER : f32 = {g:?};\n\
             const P_SELF_SHADOW : f32 = {ss:?};\n\
             const P_FLAKE : bool = {flake};\n\
             const P_INTENSITY : f32 = {intensity:?};\n\
             const P_ALPHA_TEST : f32 = {at:?};\n\
             const P_GLINT : f32 = {glint:?};\n\
             const P_DETAIL : f32 = {detail:?};\n\
             const P_STRETCH : f32 = {stretch:?};\n\
             const P_EXTINCTION : f32 = {extinction:?};\n\
             const P_PLATES : f32 = {plates:?};\n\
             const P_RAINBOW : f32 = {rainbow:?};\n",
            kind = kind as u32,
            softness = softness.max(0.0),
            shutter = shutter,
            lit = self.lit,
            g = self.scattering.clamp(-0.95, 0.95),
            ss = self.self_shadow.clamp(0.0, 1.0),
            flake = self.flake,
            intensity = self.intensity,
            at = if opaque { 0.5f32 } else { 0.003 },
            glint = self.glint.max(0.0),
            detail = self.detail.clamp(0.0, 1.0),
            stretch = self.stretch.max(0.0),
            extinction = if self.lit {
                self.extinction.max(0.0)
            } else {
                0.0
            },
            plates = self.plates.max(0.0),
            rainbow = self.rainbow.max(0.0),
        );
        ShaderMaterial {
            // A distortion reads the opaque scene behind it: its own module,
            // since the capture is bound only in the screen-space pass.
            fragment: consts
                + if distortion {
                    PARTICLE_DISTORT_WGSL
                } else {
                    PARTICLE_WGSL
                },
            vertex: Some(PARTICLE_VERTEX_WGSL.to_string()),
            screen_space: distortion,
            data: vec![
                [
                    kind,
                    softness.max(0.0),
                    shutter,
                    if self.lit { 1.0 } else { 0.0 },
                ],
                [
                    self.scattering.clamp(-0.95, 0.95),
                    self.self_shadow.clamp(0.0, 1.0),
                    if self.flake { 1.0 } else { 0.0 },
                    self.intensity,
                ],
                [
                    if opaque { 0.5 } else { 0.003 },
                    self.glint.max(0.0),
                    0.0,
                    0.0,
                ],
                [self.detail.clamp(0.0, 1.0), self.stretch.max(0.0), 0.0, 0.0],
            ],
            transparent: !opaque,
            side: 2,
            render_state: RenderState {
                blending,
                ..RenderState::DEFAULT
            },
            fallback_color: Color::WHITE,
            ..Default::default()
        }
    }
}

/// An object drawing `system`'s particles: add it to a scene, then call
/// [`write_instances`] after each step (or use a [`ParticleEffect`], which
/// does both).
pub fn particles_object(system: &ParticleSystem) -> Object3D {
    let mesh = InstancedMesh::from_arc(
        Arc::new(PlaneGeometry::new(1.0, 1.0)),
        Arc::new(Material::Shader(system.look.material())),
        0,
    );
    let mut object = Object3D::instanced_mesh(mesh);
    // Particles' own shadows would be squares: quads unturned in the shadow
    // pass. They receive shadows, if they are lit.
    object.cast_shadow = false;
    object.receive_shadow = system.look.lit;
    object.name = "particles".into();
    // After the opaque scene, among the transparent things.
    object.render_order = 10;
    object
}

/// `system`'s particles into `mesh`'s instances, back to front from `eye`
/// for alpha-blended ones.
pub fn write_instances(system: &ParticleSystem, mesh: &mut InstancedMesh, eye: Option<Vector3>) {
    write_instances_lit(system, mesh, eye, None);
}

/// [`write_instances`], with each particle's optical depths toward the sun
/// and the sky (from [`volume::depths`](super::volume::depths)) for a look
/// with [`extinction`](ParticleLook::extinction).
pub fn write_instances_lit(
    system: &ParticleSystem,
    mesh: &mut InstancedMesh,
    eye: Option<Vector3>,
    depths: Option<&[[f32; 2]]>,
) {
    let depths = depths.filter(|d| d.len() == system.particles.len());
    let ps = &system.particles;
    let hot = system.thermal.is_some()
        && ps.temperature.len() == ps.len()
        && ps.remaining.len() == ps.len();
    let n = ps.len();
    let order: Vec<usize> = match (system.look.blend, eye) {
        (ParticleBlend::Alpha, Some(eye)) => far_to_near(&ps.position, eye),
        _ => (0..n).collect(),
    };
    mesh.transforms.clear();
    mesh.colors.clear();
    let mut motion = Vec::with_capacity(n * 4);
    let mut life = Vec::with_capacity(n * 4);
    for &i in &order {
        let t = system.life_fraction(i);
        let mut size = ps.size[i] * system.size_over_life.at(t);
        if hot {
            // Shrinking as it burns.
            size *= ps.remaining[i].sqrt();
        }
        let p = ps.position[i];
        mesh.transforms.push(Matrix4 {
            elements: [
                size, 0.0, 0.0, 0.0, 0.0, size, 0.0, 0.0, 0.0, 0.0, size, 0.0, p.x, p.y, p.z, 1.0,
            ],
        });
        let [mut r, mut g, mut b, a] = system.color_over_life.at(t);
        if hot {
            let k = ps.temperature[i];
            let [br, bg, bb] = blackbody(k);
            let e = (k / 1500.0).powi(4);
            r *= br * e;
            g *= bg * e;
            b *= bb * e;
        } else if let Some(temperature) = &system.temperature_over_life {
            // Glowing as a black body at that temperature: its colour, and
            // its brightness going as T⁴ (relative to 1500 K).
            let k = temperature.at(t);
            let [br, bg, bb] = blackbody(k);
            let e = (k / 1500.0).powi(4);
            r *= br * e;
            g *= bg * e;
            b *= bb * e;
        }
        mesh.colors.push(Color::new(r, g, b));
        let v = ps.velocity[i];
        motion.extend([v.x, v.y, v.z, ps.rotation[i]]);
        let light = depths.map_or(super::volume::CLEAR, |d| super::volume::pack(d[i]));
        life.extend([a.clamp(0.0, 1.0), t, ps.seed[i], light]);
    }
    mesh.set_custom(0, 4, &motion);
    mesh.set_custom(1, 4, &life);
    mesh.instance_version = None;
}

/// The colour of a black body at `kelvin`, linear RGB with its brightest
/// channel 1 — Tanner Helland's fit to the Planckian locus (1000–40 000 K),
/// decoded from sRGB.
pub fn blackbody(kelvin: f32) -> [f32; 3] {
    let t = kelvin.clamp(1000.0, 40_000.0) / 100.0;
    let r = if t <= 66.0 {
        255.0
    } else {
        329.698_73 * (t - 60.0).powf(-0.133_204_76)
    };
    let g = if t <= 66.0 {
        99.470_8 * t.ln() - 161.119_57
    } else {
        288.122_17 * (t - 60.0).powf(-0.075_514_85)
    };
    let b = if t >= 66.0 {
        255.0
    } else if t <= 19.0 {
        0.0
    } else {
        138.517_73 * (t - 10.0).ln() - 305.044_8
    };
    let lin = |c: f32| {
        let c = (c / 255.0).clamp(0.0, 1.0);
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    [lin(r), lin(g), lin(b)]
}

/// Indices of `positions`, farthest from `eye` first: a counting sort into
/// 1024 bands of distance, linear in the count — fifty thousand flakes a
/// frame is too many for a comparison sort to be free, and particles in one
/// band a few millimetres deep are drawn in either order alike.
fn far_to_near(positions: &[Vector3], eye: Vector3) -> Vec<usize> {
    const BANDS: usize = 1024;
    let d: Vec<f32> = positions.iter().map(|p| (*p - eye).length()).collect();
    let (lo, hi) = d
        .iter()
        .fold((f32::INFINITY, 0.0f32), |(a, b), &x| (a.min(x), b.max(x)));
    // (Also when every distance is NaN.)
    if hi.partial_cmp(&lo) != Some(std::cmp::Ordering::Greater) {
        return (0..positions.len()).collect();
    }
    let scale = (BANDS - 1) as f32 / (hi - lo);
    let band = |x: f32| BANDS - 1 - ((x - lo) * scale) as usize;
    let mut start = vec![0usize; BANDS + 1];
    for &x in &d {
        start[band(x) + 1] += 1;
    }
    for b in 0..BANDS {
        start[b + 1] += start[b];
    }
    let mut order = vec![0usize; positions.len()];
    for (i, &x) in d.iter().enumerate() {
        let b = band(x);
        order[start[b]] = i;
        start[b] += 1;
    }
    order
}

/// A [`ParticleSystem`] bound to the object that draws it (and, if its look
/// [casts shadows](ParticleLook::casts_shadow), the invisible object that
/// casts them).
#[derive(Debug, Clone)]
pub struct ParticleEffect {
    pub system: ParticleSystem,
    pub object: ObjectId,
    pub caster: Option<ObjectId>,
}

impl ParticleEffect {
    /// Add `system`'s object (and its shadow caster) to `scene`.
    pub fn add_to(scene: &mut Scene, system: ParticleSystem) -> Self {
        let object = scene.add(particles_object(&system));
        let caster = system
            .look
            .casts_shadow
            .then(|| scene.add_to(object, caster_object()));
        Self {
            system,
            object,
            caster,
        }
    }

    /// Step the system `dt` seconds (landing on `ground`, if given) and redraw
    /// it, sorted from `eye`.
    pub fn update(
        &mut self,
        scene: &mut Scene,
        dt: f32,
        ground: Option<&Heightfield>,
        eye: Option<Vector3>,
    ) {
        self.system.step(dt, ground);
        self.sync(scene, eye);
    }

    /// Redraw without stepping.
    pub fn sync(&self, scene: &mut Scene, eye: Option<Vector3>) {
        let look = &self.system.look;
        let depths = (look.lit && look.extinction > 0.0).then(|| {
            let toward = light_direction(scene).map_or(Vector3::new(0.0, 1.0, 0.0), |d| d * -1.0);
            super::volume::depths(&self.system, toward)
        });
        if let Some(object) = scene.get_mut(self.object) {
            if let ObjectKind::InstancedMesh(mesh) = &mut object.kind {
                write_instances_lit(&self.system, mesh, eye, depths.as_deref());
            }
        }
        if let Some(caster) = self.caster {
            let toward = sun_direction(scene).map_or(Vector3::new(0.0, 1.0, 0.0), |d| d * -1.0);
            let mut g = shadow_caster_geometry(&self.system, toward);
            // The caster is the particles' object's child: into its space.
            if let Some(o) = scene.get(self.object) {
                let to_local = o.matrix_world.invert();
                if let Some(pos) = g.get_attribute("position") {
                    let mut a = pos.clone();
                    for v in a.array.chunks_exact_mut(3) {
                        let q = Vector3::new(v[0], v[1], v[2]).apply_matrix4(&to_local);
                        v.copy_from_slice(&[q.x, q.y, q.z]);
                    }
                    g.set_attribute("position", a);
                }
            }
            if let Some(object) = scene.get_mut(caster) {
                object.visible = g.index.as_ref().is_some_and(|i| !i.is_empty());
                if let ObjectKind::Mesh(mesh) = &mut object.kind {
                    mesh.geometry = Arc::new(g);
                }
            }
        }
    }
}

/// The direction the scene's first shadow-casting directional light shines
/// (from it toward its target), as the renderer works it out.
pub(crate) fn sun_direction(scene: &Scene) -> Option<Vector3> {
    let mut found = None;
    scene.arena.traverse(scene.root, &mut |_, o| {
        if found.is_some() {
            return;
        }
        if let ObjectKind::Light(crate::lights::Light::Directional(l)) = &o.kind {
            if l.cast_shadow {
                let p = o.world_position();
                found = Some(if p.length_sq() > 1e-8 {
                    (p * -1.0).normalize()
                } else {
                    l.direction.normalize()
                });
            }
        }
    });
    found
}

/// The direction the scene's first directional light shines — the one the
/// particle shader takes for the sun (`frame.dir_lights[0]`).
pub(crate) fn light_direction(scene: &Scene) -> Option<Vector3> {
    let mut found = None;
    scene.arena.traverse(scene.root, &mut |_, o| {
        if found.is_some() || !o.visible {
            return;
        }
        if let ObjectKind::Light(crate::lights::Light::Directional(l)) = &o.kind {
            let p = o.world_position();
            found = Some(if p.length_sq() > 1e-8 {
                (p * -1.0).normalize()
            } else {
                l.direction.normalize()
            });
        }
    });
    found
}

/// An object casting particles' shadows: drawn into the shadow maps only.
pub(crate) fn caster_object() -> Object3D {
    // Double-sided: a one-sided caster's shadow pass draws only its back
    // faces (three.js's rule), and these discs face the light.
    let mut material = crate::fur::invisible_caster();
    material.side = 2;
    let mut o = Object3D::mesh(crate::core::Mesh::new(
        shadow_caster_geometry(
            &ParticleSystem::new(ParticleLook::new(
                ParticleKind::Billboard { softness: 1.0 },
                ParticleBlend::Alpha,
            )),
            Vector3::new(0.0, 1.0, 0.0),
        ),
        Material::Basic(material),
    ));
    // Hidden while there is nothing to cast.
    o.visible = false;
    o.cast_shadow = true;
    o.receive_shadow = false;
    o.name = "particle shadows".into();
    o
}

/// Each particle as an octagon facing `toward` (the light), shrunk by its
/// opacity: a shadow map has no partial cover, so a thin wisp casts a
/// smaller shadow (or none) rather than a fainter one, and the map's filter
/// softens the rest.
pub fn shadow_caster_geometry(
    system: &ParticleSystem,
    toward: Vector3,
) -> crate::core::BufferGeometry {
    let (u, v) = super::basis(toward.normalize());
    let ps = &system.particles;
    let mut p = Vec::with_capacity(ps.len() * 9 * 3);
    let mut ix = Vec::with_capacity(ps.len() * 24);
    for i in 0..ps.len() {
        let t = system.life_fraction(i);
        let alpha = system.color_over_life.at(t)[3].clamp(0.0, 1.0);
        // Only what is fairly opaque casts, and as much smaller as it is
        // thinner: the dense core of a plume, not its every wisp.
        let r = 0.5 * ps.size[i] * system.size_over_life.at(t) * ((alpha - 0.15) / 0.85).max(0.0);
        if r < 1e-4 {
            continue;
        }
        let c = ps.position[i];
        let first = (p.len() / 3) as u32;
        p.extend([c.x, c.y, c.z]);
        for k in 0..8 {
            let a = k as f32 * std::f32::consts::TAU / 8.0;
            let q = c + u * (r * a.cos()) + v * (r * a.sin());
            p.extend([q.x, q.y, q.z]);
        }
        for k in 0..8u32 {
            ix.extend([first, first + 1 + k, first + 1 + (k + 1) % 8]);
        }
    }
    // Always a position attribute, if an empty one: the renderer will not
    // take a geometry without. (The caster is hidden while it is empty.)
    let mut g = crate::core::BufferGeometry::new();
    g.set_attribute("position", crate::core::BufferAttribute::new(p, 3));
    g.set_index(ix);
    g
}

/// The vertex stage: the unit quad turned to face the camera at the
/// particle's centre and size, or stretched along its motion.
pub(crate) const PARTICLE_VERTEX_WGSL: &str = r#"
@vertex
fn vs_custom(in : InstancedCustomIn, @builtin(vertex_index) vid : u32,
             @builtin(instance_index) iid : u32) -> VsOut {
    var out : VsOut;
    // Particles live in world coordinates, whatever the object's own
    // transform (which a `ParticleAnimator` uses to place the emitters).
    let center = in.imat3.xyz;
    let size = length(in.imat0.xyz);
    let right = vec3<f32>(frame.view[0][0], frame.view[1][0], frame.view[2][0]);
    let up = vec3<f32>(frame.view[0][1], frame.view[1][1], frame.view[2][1]);
    let to_cam = normalize(frame.camera_position.xyz - center);
    let q = in.position.xy;
    var ax = right;
    var ay = up;
    var height = size;
    var shift = vec3<f32>(0.0);
    if (P_KIND == 1u) {
        // A streak: along the motion across the view, as long as the
        // distance covered in the shutter time, its head at the particle.
        let v = in.icustom0.xyz;
        let across = v - to_cam * dot(v, to_cam);
        let len = length(across) * P_SHUTTER;
        if (len > 1e-5) {
            ay = across / length(across);
            ax = normalize(cross(ay, to_cam));
            height = size + len;
            shift = -ay * (len * 0.5);
        }
    } else {
        // Turned by the particle's spin…
        let c = cos(in.icustom0.w);
        let s = sin(in.icustom0.w);
        ax = right * c + up * s;
        ay = up * c - right * s;
        // …or, asked to, drawn out along its motion: a tongue of flame.
        let v = in.icustom0.xyz;
        let across = v - to_cam * dot(v, to_cam);
        let len = length(across) * P_STRETCH;
        if (P_KIND == 0u && P_STRETCH > 0.0 && len > 1e-4) {
            ay = across / length(across);
            ax = normalize(cross(ay, to_cam));
            height = size * (1.0 + len / max(size, 1e-4));
        }
    }
    out.world_pos = center + shift + ax * (q.x * size) + ay * (q.y * height);
    out.clip_pos = frame.view_proj * vec4<f32>(out.world_pos, 1.0);
    // A flat particle is depth-tested its size nearer the camera, so
    // one crossing the ground or a wall is not sliced off along the line
    // they meet (a sphere grain is solid, and tested where it is).
    if (P_KIND != 2u) {
        let lifted = frame.view_proj * vec4<f32>(out.world_pos + to_cam * size, 1.0);
        out.clip_pos.z = lifted.z / lifted.w * out.clip_pos.w;
    }
    out.clip_zw = out.clip_pos.zw;
    out.view_z = -(frame.view * vec4<f32>(out.world_pos, 1.0)).z;
    out.uv = in.uv;
    out.vertex_color = vec4<f32>(in.itint.rgb, in.icustom1.x);
    out.world_normal = to_cam;
    // The quad's axis, and the particle's turn (a plate's rocking).
    out.world_tangent = vec4<f32>(ax, in.icustom0.w);
    // The quad's other axis, for a sphere's normal.
    out.local_pos = ay;
    // How many pixels across its radius is: small sprites skip detail no
    // one could see.
    let px = size * 0.5 * frame.projection[1][1] / max(out.clip_pos.w, 1e-4) * frame.viewport_size.y * 0.5;
    // (life, seed, half-width, pixel radius).
    out.proj_uv = vec4<f32>(in.icustom1.y, in.icustom1.z, size * 0.5, px);
    // The centre, for a flat particle's shadow lookup; and its optical
    // depths toward the sun and the sky, packed (see `volume::pack`).
    out.uv12 = vec4<f32>(center + shift, in.icustom1.w);
    return out;
}
"#;

/// A distortion: the opaque scene behind, looked up a little to the side —
/// by the slope of noise of its own, rising as it ages — and faded at the
/// edge and over its life.
pub(crate) const PARTICLE_DISTORT_WGSL: &str = r#"
fn haze_hash(p : vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(127.31, 311.71));
    q = q + vec2<f32>(dot(q, q + 19.19));
    return fract(q.x * q.y);
}
fn haze_noise(p : vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(haze_hash(i), haze_hash(i + vec2<f32>(1.0, 0.0)), u.x),
               mix(haze_hash(i + vec2<f32>(0.0, 1.0)), haze_hash(i + vec2<f32>(1.0, 1.0)), u.x), u.y);
}

@fragment
fn fs_main(in : VsOut) -> @location(0) vec4<f32> {
    let p = in.uv * 2.0 - vec2<f32>(1.0);
    let r2 = dot(p, p);
    if (r2 > 1.0) {
        discard;
    }
    let life = in.proj_uv.x;
    let seed = in.proj_uv.y;
    let q = p * 2.5 + vec2<f32>(seed * 37.0, seed * 11.0 - life * 4.0);
    let e = 0.05;
    let n0 = haze_noise(q);
    let gx = haze_noise(q + vec2<f32>(e, 0.0)) - n0;
    let gy = haze_noise(q + vec2<f32>(0.0, e)) - n0;
    let fade = (1.0 - r2) * (1.0 - r2) * in.vertex_color.a;
    let dims = vec2<f32>(textureDimensions(ss_color_tex));
    let uv = in.clip_pos.xy / dims + vec2<f32>(gx, gy) / e * (P_SOFTNESS * fade);
    let c = textureSampleLevel(ss_color_tex, ss_color_samp, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb;
    // The capture is linear and not yet tone-mapped: finish it as the frame is.
    return vec4<f32>(framebuffer_encode(apply_tone_mapping(c)), fade);
}
"#;

/// The fragment stage: a soft disc, or a lit sphere; lit by the scene's
/// lights if asked.
pub(crate) const PARTICLE_WGSL: &str = r#"
fn particle_hash(p : vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(127.31, 311.71));
    q = q + vec2<f32>(dot(q, q + 19.19));
    return fract(q.x * q.y);
}

// Value noise and its slope (d/dp), from the same four hashes.
fn particle_noise_d(p : vec2<f32>) -> vec3<f32> {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let du = 6.0 * f * (1.0 - f);
    let a = particle_hash(i);
    let b = particle_hash(i + vec2<f32>(1.0, 0.0));
    let c = particle_hash(i + vec2<f32>(0.0, 1.0));
    let d = particle_hash(i + vec2<f32>(1.0, 1.0));
    let k = a - b - c + d;
    return vec3<f32>(a + (b - a) * u.x + (c - a) * u.y + k * u.x * u.y,
                     du * vec2<f32>(b - a + k * u.y, c - a + k * u.x));
}

// Three octaves, each turned against the last, so no grid shows; and the
// slope, carried back through each octave's turn and scale — the billow's
// relief and self-shadow with no extra lookups.
fn particle_fbm_d(p : vec2<f32>) -> vec3<f32> {
    var s = vec3<f32>(0.0);
    var a = 0.5;
    var q = p;
    var j = mat2x2<f32>(1.0, 0.0, 0.0, 1.0);
    let turn = mat2x2<f32>(0.8, 0.6, -0.6, 0.8);
    for (var i = 0; i < 3; i = i + 1) {
        let n = particle_noise_d(q);
        s = s + a * vec3<f32>(n.x, transpose(j) * n.yz);
        q = turn * q * 2.03 + vec2<f32>(5.2, 1.3);
        j = turn * j * 2.03;
        a = a * 0.5;
    }
    return s / 0.875;
}

// A flat particle's shadow from record `r`: the share of four points across
// it — at its centre `c`, `radius` either way along `ax` and `ay` — that
// the light reaches. One compare each: the particle is soft already, so the
// filter a surface needs would only blur a blur, at a dozen compares a
// pixel through every particle overlapping it.
fn particle_shadow_record(r : i32, c : vec3<f32>, radius : f32, ax : vec3<f32>, ay : vec3<f32>) -> f32 {
    let rec = frame.shadows[r];
    let kind = rec.info.x;
    let layer = i32(rec.info.y);
    let zero = vec3<f32>(0.0);
    var lit = 0.0;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let o = vec2<f32>(select(-0.5, 0.5, (k & 1u) != 0u), select(-0.5, 0.5, (k & 2u) != 0u)) * radius;
        let p = c + ax * o.x + ay * o.y;
        if (kind == 2u) {
            let to = p - rec.extra.xyz;
            let dist = length(to);
            if (dist > rec.extra.w) {
                lit = lit + 1.0;
                continue;
            }
            let depth = dist / rec.extra.w - min(0.005, 0.25 / rec.extra.w) - rec.params.x;
            lit = lit + shadow_tap(2u, layer, to / max(dist, 1e-6), zero, zero, vec2<f32>(0.0), depth);
            continue;
        }
        var cascade = 0u;
        if (kind == 0u && rec.info.z > 1u) {
            let view_depth = -(frame.view * vec4<f32>(p, 1.0)).z;
            cascade = rec.info.z;
            for (var i = 0u; i < rec.info.z; i = i + 1u) {
                if (view_depth <= rec.extra[i]) {
                    cascade = i;
                    break;
                }
            }
        }
        let h = rec.vp[min(cascade, 3u)] * vec4<f32>(p, 1.0);
        let ndc = h.xyz / max(h.w, 0.0001);
        let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if (cascade >= max(rec.info.z, 1u) || any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) || ndc.z < 0.0 || ndc.z > 1.0) {
            lit = lit + 1.0;
        } else {
            lit = lit + shadow_compare(kind, uv, layer + i32(cascade), ndc.z - rec.params.x);
        }
    }
    return lit * 0.25;
}

fn particle_shadows(c : vec3<f32>, radius : f32, ax : vec3<f32>, ay : vec3<f32>) {
    for (var r = 0u; r < 16u; r = r + 1u) {
        shadow_cache[r] = 1.0;
    }
    let count = min(frame.shadow_info.x, 16u);
    for (var r = 0u; r < count; r = r + 1u) {
        shadow_cache[r] = particle_shadow_record(i32(r), c, radius, ax, ay);
    }
}

// A six-armed snowflake: arms, a branch pair on each, and a hexagonal
// heart — 1 inside, 0 outside, with a soft edge.
fn particle_flake(p : vec2<f32>, turn : f32) -> f32 {
    let r = length(p);
    let a = atan2(p.y, p.x) + turn;
    // The angle to the nearest arm, and where along it.
    let sector = (fract(a * 6.0 / 6.2831853 + 0.5) - 0.5) * 6.2831853 / 6.0;
    let x = r * cos(sector);
    let y = abs(r * sin(sector));
    let w = 0.055 + 0.04 * (1.0 - r);
    var m = 1.0 - smoothstep(w * 0.6, w, y);
    // Branches at 60° off the arm, at two places along it.
    for (var k = 0; k < 2; k = k + 1) {
        let x0 = 0.38 + 0.24 * f32(k);
        let len = 0.26 - 0.08 * f32(k);
        let along = (x - x0) * 0.5 + y * 0.8660254;
        let across = abs(-(x - x0) * 0.8660254 + y * 0.5);
        if (along > 0.0 && along < len) {
            m = max(m, 1.0 - smoothstep(w * 0.45, w * 0.8, across));
        }
    }
    m = max(m, 1.0 - smoothstep(0.15, 0.19, r * (0.87 + 0.13 * cos(6.0 * a))));
    return m * (1.0 - smoothstep(0.86, 1.0, r));
}

@fragment
fn fs_main(in : VsOut) -> @location(0) vec4<f32> {
    let p = in.uv * 2.0 - vec2<f32>(1.0);
    let r2 = dot(p, p);
    if (r2 > 1.0) {
        discard;
    }
    let sphere = P_KIND == 2u;
    let life = in.proj_uv.x;
    let seed = in.proj_uv.y;
    let px = in.proj_uv.w;
    var alpha = in.vertex_color.a;
    let to_cam = normalize(in.world_normal);
    let ax = normalize(in.world_tangent.xyz);
    let ay = normalize(in.local_pos);
    // A billow's detail, faded out on sprites too small to show it.
    let detail = P_DETAIL * smoothstep(3.0, 10.0, px);
    // A sphere's normal; for a billowing particle the same soft ball, which
    // is how a puff of smoke is lit. A small flat particle — a flake, a
    // mote — faces the eye: shaded as a ball, it reads as a snowball.
    var n = to_cam;
    if (sphere || P_DETAIL > 0.0) {
        n = normalize(ax * p.x + ay * p.y + to_cam * sqrt(1.0 - r2));
    }
    // The first sun, for a billow's self-shadowing and a flat particle's
    // shadow lookup.
    var sun = vec3<f32>(0.0, 1.0, 0.0);
    if (frame.light_counts.x > 0u) {
        sun = -normalize(frame.dir_lights[0].direction.xyz);
    }
    var density = 1.0;
    var self_lit = 1.0;
    if (!sphere) {
        // Arms and branches only where there are pixels enough for them.
        if (P_FLAKE && px > 3.0) {
            alpha = alpha * particle_flake(p, seed * 6.2831853);
        } else if (P_EXTINCTION == 0.0) {
            alpha = alpha * pow(1.0 - r2, P_SOFTNESS);
        }
        if (P_DETAIL > 0.0 && detail > 0.0) {
            // A billow: noise of its own (by its seed), drifting as it ages,
            // its threshold rising so it thins away toward the end.
            let q = p * 1.7 + vec2<f32>(seed * 61.0, seed * 23.0) + vec2<f32>(0.0, -life * 1.3);
            let fd = particle_fbm_d(q);
            let f = fd.x;
            let threshold = mix(0.3, 0.62, life) * detail;
            let edge = smoothstep(threshold - 0.18, threshold + 0.18, f + (1.0 - r2) * 0.25);
            alpha = alpha * mix(1.0, edge, detail);
            density = mix(1.0, 0.55 + 0.9 * f, detail);
            // The billow's slope (over a step `e`): its relief, in the
            // normal…
            let e = 0.08;
            let grad = fd.yz * e;
            n = normalize(n - (ax * grad.x + ay * grad.y) * (detail * 6.0));
            // …and, carried along it toward the sun, how much density light
            // crosses to get here: its own dark side, with no fourth lookup.
            if (P_SELF_SHADOW > 0.0) {
                let toward = vec2<f32>(dot(sun, ax), dot(sun, ay)) * 0.5;
                let blocked = f + dot(grad / e, toward) + 0.35 * dot(sun, -to_cam);
                self_lit = mix(1.0, exp(-3.0 * max(blocked - 0.3, 0.0)), P_SELF_SHADOW * detail);
            }
        }
    }
    // Faded out as the eye comes within two of its radii: a sprite the
    // camera is about to pass through fills the view as a flat disc.
    alpha = alpha * smoothstep(0.6, 2.0, in.view_z / max(in.proj_uv.z, 1e-4));
    // A little sphere of medium: as opaque as the optical depth along this
    // line of sight through it (its chord, times its extinction, times its
    // opacity and the billow's density).
    if (!sphere && P_EXTINCTION > 0.0) {
        alpha = 1.0 - exp(-P_EXTINCTION * alpha * 2.0 * in.proj_uv.z * sqrt(1.0 - r2));
    }
    if (alpha < P_ALPHA_TEST) {
        discard;
    }
    // The optical depth to the sun through the rest of the system, and the
    // skylight getting through from above.
    let tau_sun = floor(in.uv12.w) / 128.0;
    let sky_through = fract(in.uv12.w);
    // A grain's shade varies a little from grain to grain.
    let shade = select(1.0, 0.82 + 0.36 * seed, sphere);
    let base = in.vertex_color.rgb * (P_INTENSITY * shade * density);
    var col = base;
    if (P_LIT) {
        if ((mesh.flags.z & FLAG_RECEIVE_SHADOW) != 0u) {
            if (sphere) {
                compute_shadows(in.world_pos, n);
            } else {
                // A flat particle is a little volume, shadowed as one: at
                // its centre, nudged toward the sun past its own caster
                // disc. Looked up across the sprite, other particles' discs
                // would print hard edges on it.
                particle_shadows(in.uv12.xyz + sun * (in.proj_uv.z * 1.2), in.proj_uv.z, ax, ay);
            }
        } else {
            for (var r = 0u; r < 16u; r = r + 1u) {
                shadow_cache[r] = 1.0;
            }
        }
        let RECIP_PI = 0.3183098861837907;
        // Skylight, less where the billow is dense.
        var occlusion = mix(1.0, 0.55 + 0.45 * self_lit, P_SELF_SHADOW);
        if (P_EXTINCTION > 0.0) {
            // Skylight through the smoke above, some of it scattered on
            // down: half straight through, half as if through a quarter of
            // the depth.
            occlusion = occlusion * (0.5 * sky_through + 0.5 * pow(sky_through, 0.25));
        }
        col = base * frame.ambient.rgb * RECIP_PI * occlusion;
        for (var k = 0u; k < frame.light_counts.w; k = k + 1u) {
            if (k >= 8u) { break; }
            col = col + base * shade_hemi(n, frame.hemi_lights[k]) * RECIP_PI * occlusion;
        }
        // The tiny mirror each particle carries, at its own tilt.
        var facet = normalize(to_cam * 1.2 + vec3<f32>(seed - 0.5, fract(seed * 7.3) - 0.5, fract(seed * 13.1) - 0.5) * 2.0);
        if (P_PLATES > 0.0) {
            // A plate falling broadside: level, rocked by its flutter (its
            // spin's phase) by up to P_PLATES, each its own amount — either
            // face mirrors.
            let rock = in.world_tangent.w + seed * 17.0;
            let tilt = P_PLATES * (0.25 + 0.75 * fract(seed * 7.3));
            facet = normalize(vec3<f32>(cos(rock) * tilt, 1.0, sin(rock) * tilt));
        }
        for (var k = 0u; k < frame.light_counts.x; k = k + 1u) {
            if (k >= 8u) { break; }
            let l = -normalize(frame.dir_lights[k].direction.xyz);
            let light = frame.dir_lights[k].color.rgb * shadow_factor_dir(k);
            var through = particle_light(n, l, to_cam, sphere);
            if (P_EXTINCTION > 0.0 && k == 0u) {
                through = particle_volume_light(n, l, to_cam, tau_sun);
            }
            col = col + base * light * through * self_lit * RECIP_PI;
            if (P_GLINT > 0.0) {
                col = col + light * particle_glint(facet, l, to_cam);
            }
            if (P_RAINBOW > 0.0) {
                col = col + light * particle_rainbow(dot(to_cam, l)) * P_RAINBOW;
            }
        }
        for (var k = 0u; k < frame.light_counts.y; k = k + 1u) {
            if (k >= 8u) { break; }
            let pl = frame.point_lights[k];
            let to_light = pl.position.xyz - in.world_pos;
            let d = length(to_light);
            let l = to_light / max(d, 1e-4);
            let att = punctual_attenuation(d, pl.params.x, pl.params.y) * shadow_factor_point(k);
            col = col + base * pl.color.rgb * att * particle_light(n, l, to_cam, sphere) * RECIP_PI;
            if (P_GLINT > 0.0) {
                col = col + pl.color.rgb * att * particle_glint(facet, l, to_cam);
            }
        }
    }
    return vec4<f32>(framebuffer_encode(apply_tone_mapping(apply_fog(col, in.view_z))), alpha);
}

// Light a raindrop sends toward the eye, by the angle between the line of
// sight and the point opposite the sun (`cos_a` its cosine): the primary
// bow (one internal reflection, red outermost at 42.3°, violet at 40.6°)
// with the brighter sky inside it, Alexander's dark band, and the secondary
// (two reflections, reversed, 50.4°–53.5°, about 43% as bright).
fn particle_rainbow(cos_a : f32) -> vec3<f32> {
    let a = degrees(acos(clamp(cos_a, -1.0, 1.0)));
    let primary = vec3<f32>(42.3, 41.5, 40.6);
    let secondary = vec3<f32>(50.4, 51.6, 53.5);
    let w = 0.55;
    let p = exp(-pow((vec3<f32>(a) - primary) / w, vec3<f32>(2.0)));
    let s = exp(-pow((vec3<f32>(a) - secondary) / (w * 1.4), vec3<f32>(2.0))) * 0.43;
    // Inside the primary, light from rays short of the bow's minimum
    // deviation: a pale glow, fading toward the antisolar point.
    let inside = vec3<f32>(0.06) * smoothstep(20.0, 40.0, a) * (1.0 - smoothstep(40.0, 41.0, a));
    return (p + s) * 0.35 + inside;
}

// A glint: the particle's facet mirroring the light into the eye — a
// narrow lobe, so it flashes as the facet turns through it.
fn particle_glint(facet : vec3<f32>, l : vec3<f32>, v : vec3<f32>) -> f32 {
    let h = normalize(l + v);
    // Bright, but not so bright that bloom turns one flake into a lamp.
    return min(P_GLINT * pow(max(dot(facet, h), 0.0), 400.0) * 12.0, 2.5);
}

// The phase function relative to an isotropic scatterer: two
// Henyey–Greenstein lobes, most forward (`g`) and a little back, as smoke
// and dust measure.
fn particle_phase(g : f32, cos_theta : f32) -> f32 {
    let hg = (1.0 - g * g) / pow(max(1.0 + g * g - 2.0 * g * cos_theta, 1e-4), 1.5);
    let b = -0.5 * g;
    let back = (1.0 - b * b) / pow(max(1.0 + b * b - 2.0 * b * cos_theta, 1e-4), 1.5);
    return 0.85 * hg + 0.15 * back;
}

// The sun reaching a particle of medium through optical depth `tau` of the
// rest, and scattered toward the eye: single scattering, plus light
// scattered more than once — Wrenninge's octaves, each with half the
// extinction, half the weight and half the asymmetry of the last, so the
// far side of a plume is dim but not black and lit more evenly. Normalised
// to the single-scattering phase with nothing in the way.
fn particle_volume_light(n : vec3<f32>, l : vec3<f32>, v : vec3<f32>, tau : f32) -> f32 {
    let cos_theta = dot(-l, v);
    var ms = 0.0;
    var a = 1.0;
    var b = 1.0;
    var c = 1.0;
    for (var i = 0; i < 3; i = i + 1) {
        ms = ms + b * exp(-a * tau) * particle_phase(P_SCATTER * c, cos_theta);
        a = a * 0.5;
        b = b * 0.5;
        c = c * 0.5;
    }
    // A little of the ball's shape, for its form against the light.
    let wrap = max((dot(n, l) + 0.6) / 1.6, 0.0) * 0.8 + 0.2;
    return ms / 1.75 * mix(1.0, wrap, 0.35);
}

// How much of a light a particle passes on to the eye. A solid sphere:
// Lambert. Something thin and scattering — smoke, a flake, dust: light
// wrapped round its ball (lit through, not only on its face), weighted by
// the Henyey–Greenstein phase of its asymmetry `P_SCATTER` — the share
// scattered from the light's direction into the eye's, against what an
// isotropic scatterer sends.
fn particle_light(n : vec3<f32>, l : vec3<f32>, v : vec3<f32>, sphere : bool) -> f32 {
    if (sphere) {
        return max(dot(n, l), 0.0);
    }
    let wrap = max((dot(n, l) + 0.6) / 1.6, 0.0) * 0.8 + 0.2;
    if (P_SCATTER == 0.0) {
        return wrap;
    }
    let g = P_SCATTER;
    let cos_theta = dot(-l, v);
    let hg = (1.0 - g * g) / pow(max(1.0 + g * g - 2.0 * g * cos_theta, 1e-4), 1.5);
    return wrap * mix(1.0, hg, min(abs(g) * 2.0, 1.0));
}
"#;
