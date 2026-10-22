//! Particles from JavaScript (`threers::particles`): systems made from the
//! presets or built up by setters, added to a scene and run by
//! `WebScene.stepParticles`, landing in snow covers and sand beds.

use wasm_bindgen::prelude::*;

use super::{WebObjectHandle, WebScene};
use crate::math::{Color, Vector3};
use crate::particles::sand::{Downwind, SandBed};
use crate::particles::snow::SnowCover;
use crate::particles::{
    particles_object, presets, Collider, Curve, Direction, Emitter, EmitterShape, Field, Gradient,
    Heightfield, ParticleBlend, ParticleKind, ParticleLook, ParticleSystem, Range, Response,
    Turbulence, Weather,
};

/// A particle system (`threers::particles::ParticleSystem`): one of the
/// presets, or `new WebParticleSystem()` and the setters. Setters about
/// emission act on the first emitter.
#[wasm_bindgen]
pub struct WebParticleSystem {
    inner: ParticleSystem,
}

fn v(x: f32, y: f32, z: f32) -> Vector3 {
    Vector3::new(x, y, z)
}

impl WebParticleSystem {
    fn emitter(&mut self) -> &mut Emitter {
        if self.inner.emitters.is_empty() {
            self.inner
                .emitters
                .push(Emitter::new(EmitterShape::Point, Vector3::ZERO));
        }
        &mut self.inner.emitters[0]
    }
}

#[wasm_bindgen]
impl WebParticleSystem {
    /// An empty system: soft white billboards, alpha-blended, no emitter yet.
    #[wasm_bindgen(constructor)]
    pub fn new() -> WebParticleSystem {
        WebParticleSystem {
            inner: ParticleSystem::new(ParticleLook::new(
                ParticleKind::Billboard { softness: 1.0 },
                ParticleBlend::Alpha,
            )),
        }
    }

    pub fn fire(x: f32, y: f32, z: f32, radius: f32) -> WebParticleSystem {
        WebParticleSystem {
            inner: presets::fire(v(x, y, z), radius),
        }
    }
    /// Heat haze over a fire: invisible, bending the view behind it.
    #[wasm_bindgen(js_name = heatHaze)]
    pub fn heat_haze(x: f32, y: f32, z: f32, radius: f32) -> WebParticleSystem {
        WebParticleSystem {
            inner: presets::heat_haze(v(x, y, z), radius),
        }
    }
    pub fn embers(x: f32, y: f32, z: f32, radius: f32) -> WebParticleSystem {
        WebParticleSystem {
            inner: presets::embers(v(x, y, z), radius),
        }
    }
    pub fn smoke(x: f32, y: f32, z: f32, radius: f32) -> WebParticleSystem {
        WebParticleSystem {
            inner: presets::smoke(v(x, y, z), radius),
        }
    }
    pub fn sparks(x: f32, y: f32, z: f32) -> WebParticleSystem {
        WebParticleSystem {
            inner: presets::sparks(v(x, y, z)),
        }
    }
    /// Rain over `sx` × `sz` metres round (`cx`, `cy`, `cz`), from `height`,
    /// `rate` drops a second, in wind (`wx`, `wy`, `wz`) m/s.
    #[allow(clippy::too_many_arguments)]
    pub fn rain(
        cx: f32,
        cy: f32,
        cz: f32,
        sx: f32,
        sz: f32,
        height: f32,
        rate: f32,
        wx: f32,
        wy: f32,
        wz: f32,
    ) -> WebParticleSystem {
        WebParticleSystem {
            inner: presets::rain(v(cx, cy, cz), [sx, sz], height, rate, v(wx, wy, wz)),
        }
    }
    /// Snow over `sx` × `sz` metres round (`cx`, `cy`, `cz`), from `height`,
    /// `rate` flakes a second building `depthPerSecond` metres of snow a
    /// second, in wind (`wx`, `wy`, `wz`) m/s. Settle it on a snow cover
    /// (`WebScene.settleParticles`) to keep what falls.
    #[allow(clippy::too_many_arguments)]
    pub fn snowfall(
        cx: f32,
        cy: f32,
        cz: f32,
        sx: f32,
        sz: f32,
        height: f32,
        rate: f32,
        depth_per_second: f32,
        wx: f32,
        wy: f32,
        wz: f32,
    ) -> WebParticleSystem {
        WebParticleSystem {
            inner: presets::snowfall(
                v(cx, cy, cz),
                [sx, sz],
                height,
                rate,
                depth_per_second,
                v(wx, wy, wz),
            ),
        }
    }
    /// Sand poured from (`x`, `y`, `z`): `rate` grains a second carrying
    /// `flow` m³ a second. Settle it on a sand bed to pile it.
    #[wasm_bindgen(js_name = sandStream)]
    pub fn sand_stream(x: f32, y: f32, z: f32, rate: f32, flow: f32) -> WebParticleSystem {
        WebParticleSystem {
            inner: presets::sand_stream(v(x, y, z), rate, flow),
        }
    }
    pub fn dust(x: f32, y: f32, z: f32, count: u32) -> WebParticleSystem {
        WebParticleSystem {
            inner: presets::dust(v(x, y, z), count),
        }
    }
    pub fn fountain(x: f32, y: f32, z: f32, height: f32) -> WebParticleSystem {
        WebParticleSystem {
            inner: presets::fountain(v(x, y, z), height),
        }
    }

    /// How they are drawn: `kind` `"billboard"`, `"streak"` or `"sphere"`;
    /// `blend` `"alpha"`, `"additive"` or `"opaque"`; `softness` of a disc's
    /// edge; `shutter` seconds of motion a streak shows; `intensity` scales
    /// the colour; `lit` shades by the scene's lights; `detail` (0–1) makes a
    /// flat particle a billow; `stretch` (seconds) draws a billboard out along
    /// its motion.
    #[wasm_bindgen(js_name = setLook)]
    #[allow(clippy::too_many_arguments)]
    pub fn set_look(
        &mut self,
        kind: &str,
        blend: &str,
        softness: f32,
        shutter: f32,
        intensity: f32,
        lit: bool,
        detail: Option<f32>,
        stretch: Option<f32>,
    ) {
        self.inner.look = ParticleLook {
            kind: match kind {
                "streak" => ParticleKind::Streak { softness, shutter },
                "sphere" => ParticleKind::Sphere,
                _ => ParticleKind::Billboard { softness },
            },
            blend: match blend {
                "additive" => ParticleBlend::Additive,
                "opaque" => ParticleBlend::Opaque,
                _ => ParticleBlend::Alpha,
            },
            intensity,
            lit,
            detail: detail.unwrap_or(0.0),
            stretch: stretch.unwrap_or(0.0),
            // The rest as they were (set by the preset, or the setters below).
            ..self.inner.look
        };
    }

    /// Cinematic shading of a lit flat particle: `scattering` is the
    /// Henyey–Greenstein asymmetry (−1 to 1; smoke 0.3–0.6 glows when
    /// back-lit), `selfShadow` (0–1) how much a billow shades itself,
    /// `glint` the strength of each particle's catching the light, `flake`
    /// draws six-armed snowflakes, and `castsShadow` casts shadows.
    #[wasm_bindgen(js_name = setShading)]
    pub fn set_shading(
        &mut self,
        scattering: f32,
        self_shadow: f32,
        glint: f32,
        flake: bool,
        casts_shadow: bool,
    ) {
        let l = &mut self.inner.look;
        l.scattering = scattering;
        l.self_shadow = self_shadow;
        l.glint = glint;
        l.flake = flake;
        l.casts_shadow = casts_shadow;
    }
    /// Glow as a black body at a temperature over life: `[t, kelvin, …]`
    /// (fire, embers). Empty to stop.
    #[wasm_bindgen(js_name = setTemperatureOverLife)]
    pub fn set_temperature_over_life(&mut self, keys: &[f32]) {
        let k: Vec<(f32, f32)> = keys.chunks_exact(2).map(|c| (c[0], c[1])).collect();
        self.inner.temperature_over_life = (!k.is_empty()).then(|| crate::particles::Curve(k));
    }
    /// Flutter, m/s²: falling flakes and leaves zig-zag (0: none).
    #[wasm_bindgen(js_name = setFlutter)]
    pub fn set_flutter(&mut self, flutter: f32) {
        self.inner.forces.flutter = flutter;
    }
    /// Buoyancy (the gravity given) fading over `time` seconds toward
    /// (`gx`, `gy`, `gz`): a plume that climbs, slows and spreads.
    #[wasm_bindgen(js_name = setCooling)]
    pub fn set_cooling(&mut self, time: f32, gx: f32, gy: f32, gz: f32) {
        self.inner.forces.cooling = Some(crate::particles::Cooling {
            time,
            cold_gravity: v(gx, gy, gz),
        });
    }
    /// The wind slowing toward the ground at `ground` by the log law, with
    /// roughness length `roughness` metres; the wind given is the wind at
    /// `reference` metres up.
    #[wasm_bindgen(js_name = setBoundaryLayer)]
    pub fn set_boundary_layer(&mut self, ground: f32, roughness: f32, reference: f32) {
        self.inner.forces.boundary_layer = Some(crate::particles::BoundaryLayer {
            ground,
            roughness,
            reference,
        });
    }
    /// Particles thrown off where one hits: `minCount`–`maxCount` of them at
    /// `minSpeed`–`maxSpeed` m/s within `angle` radians of the surface's
    /// normal, living `minLife`–`maxLife` seconds, `sizeScale` times the
    /// size of the one that hit.
    #[wasm_bindgen(js_name = setSpray)]
    #[allow(clippy::too_many_arguments)]
    pub fn set_spray(
        &mut self,
        min_count: f32,
        max_count: f32,
        min_speed: f32,
        max_speed: f32,
        angle: f32,
        min_life: f32,
        max_life: f32,
        size_scale: f32,
    ) {
        self.inner.spray = Some(crate::particles::Spray {
            count: Range::new(min_count, max_count),
            speed: Range::new(min_speed, max_speed),
            angle,
            carry: 0.3,
            lifetime: Range::new(min_life, max_life),
            size: Range::fixed(size_scale),
            terminal_speed: Range::new(3.0, 5.0),
        });
    }

    /// Emit from a point.
    #[wasm_bindgen(js_name = setEmitterPoint)]
    pub fn set_emitter_point(&mut self, x: f32, y: f32, z: f32) {
        let e = self.emitter();
        e.shape = EmitterShape::Point;
        e.position = v(x, y, z);
    }
    /// Emit from a ball of `radius` (on its surface only, if `surface`).
    #[wasm_bindgen(js_name = setEmitterSphere)]
    pub fn set_emitter_sphere(&mut self, x: f32, y: f32, z: f32, radius: f32, surface: bool) {
        let e = self.emitter();
        e.shape = EmitterShape::Sphere { radius, surface };
        e.position = v(x, y, z);
    }
    /// Emit from a box of these half-extents.
    #[wasm_bindgen(js_name = setEmitterBox)]
    pub fn set_emitter_box(&mut self, x: f32, y: f32, z: f32, hx: f32, hy: f32, hz: f32) {
        let e = self.emitter();
        e.shape = EmitterShape::Box {
            half: v(hx, hy, hz),
        };
        e.position = v(x, y, z);
    }
    /// Emit from a disc of `radius` facing (`nx`, `ny`, `nz`).
    #[wasm_bindgen(js_name = setEmitterDisc)]
    #[allow(clippy::too_many_arguments)]
    pub fn set_emitter_disc(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        radius: f32,
        nx: f32,
        ny: f32,
        nz: f32,
    ) {
        let e = self.emitter();
        e.shape = EmitterShape::Disc {
            radius,
            normal: v(nx, ny, nz),
        };
        e.position = v(x, y, z);
    }
    /// Particles a second.
    #[wasm_bindgen(js_name = setRate)]
    pub fn set_rate(&mut self, rate: f32) {
        self.emitter().rate = rate;
    }
    /// `count` at once, `time` seconds after the system began.
    #[wasm_bindgen(js_name = addBurst)]
    pub fn add_burst(&mut self, time: f32, count: u32) {
        self.emitter().bursts.push((time, count));
    }
    /// Directions within `angle` radians of (`x`, `y`, `z`).
    #[wasm_bindgen(js_name = setDirectionCone)]
    pub fn set_direction_cone(&mut self, x: f32, y: f32, z: f32, angle: f32) {
        self.emitter().direction = Direction::Cone {
            axis: v(x, y, z),
            angle,
        };
    }
    /// Directions away from the shape (the default).
    #[wasm_bindgen(js_name = setDirectionOutward)]
    pub fn set_direction_outward(&mut self) {
        self.emitter().direction = Direction::Outward;
    }
    #[wasm_bindgen(js_name = setSpeed)]
    pub fn set_speed(&mut self, min: f32, max: f32) {
        self.emitter().speed = Range::new(min, max);
    }
    /// Added to every new particle's velocity.
    #[wasm_bindgen(js_name = setVelocity)]
    pub fn set_velocity(&mut self, x: f32, y: f32, z: f32) {
        self.emitter().velocity = v(x, y, z);
    }
    /// Seconds; `Infinity` lives until it lands.
    #[wasm_bindgen(js_name = setLifetime)]
    pub fn set_lifetime(&mut self, min: f32, max: f32) {
        self.emitter().lifetime = Range::new(min, max);
    }
    /// Drawn size, metres.
    #[wasm_bindgen(js_name = setSize)]
    pub fn set_size(&mut self, min: f32, max: f32) {
        self.emitter().size = Range::new(min, max);
    }
    /// Terminal speed in still air, m/s; 0 for no drag.
    #[wasm_bindgen(js_name = setTerminalSpeed)]
    pub fn set_terminal_speed(&mut self, min: f32, max: f32) {
        self.emitter().terminal_speed = Range::new(min, max);
    }
    #[wasm_bindgen(js_name = setSpin)]
    pub fn set_spin(&mut self, min: f32, max: f32) {
        self.emitter().spin = Range::new(min, max);
    }
    /// Material each particle carries, m³, for a snow cover or sand bed.
    #[wasm_bindgen(js_name = setVolume)]
    pub fn set_volume(&mut self, volume: f32) {
        self.emitter().volume = volume;
    }
    #[wasm_bindgen(js_name = setGravity)]
    pub fn set_gravity(&mut self, x: f32, y: f32, z: f32) {
        self.inner.forces.gravity = v(x, y, z);
    }
    /// The air's mean velocity, m/s.
    #[wasm_bindgen(js_name = setWind)]
    pub fn set_wind(&mut self, x: f32, y: f32, z: f32) {
        self.inner.forces.wind = v(x, y, z);
    }
    /// Eddies of `strength` m/s and `scale` metres; `churn` how fast they
    /// change (0: carried on the wind unchanged).
    #[wasm_bindgen(js_name = setTurbulence)]
    pub fn set_turbulence(&mut self, strength: f32, scale: f32, churn: f32) {
        self.inner.forces.turbulence = Turbulence {
            strength,
            scale,
            churn,
            anisotropy: 0.0,
        };
    }
    /// A pull of `strength` m/s² toward (`x`, `y`, `z`), fading to nothing at
    /// `radius` (negative pushes).
    #[wasm_bindgen(js_name = addAttractor)]
    pub fn add_attractor(&mut self, x: f32, y: f32, z: f32, strength: f32, radius: f32) {
        self.inner.forces.fields.push(Field::Attractor {
            position: v(x, y, z),
            strength,
            radius,
        });
    }
    /// Air swirling at `speed` m/s round the line through (`x`, `y`, `z`)
    /// along (`ax`, `ay`, `az`), fading to nothing at `radius`.
    #[wasm_bindgen(js_name = addVortex)]
    #[allow(clippy::too_many_arguments)]
    pub fn add_vortex(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        ax: f32,
        ay: f32,
        az: f32,
        speed: f32,
        radius: f32,
    ) {
        self.inner.forces.fields.push(Field::Vortex {
            position: v(x, y, z),
            axis: v(ax, ay, az),
            speed,
            radius,
        });
    }
    /// What particles do at a surface: `"bounce"` (with `restitution` and
    /// `friction`), `"stick"`, `"die"` or `"deposit"`.
    #[wasm_bindgen(js_name = setResponse)]
    pub fn set_response(&mut self, kind: &str, restitution: f32, friction: f32) {
        self.inner.response = match kind {
            "bounce" => Response::Bounce {
                restitution,
                friction,
            },
            "stick" => Response::Stick,
            "deposit" => Response::Deposit,
            _ => Response::Die,
        };
    }
    /// The plane y = `height`, from above.
    #[wasm_bindgen(js_name = addGround)]
    pub fn add_ground(&mut self, height: f32) {
        self.inner.colliders.push(Collider::Ground { height });
    }
    #[wasm_bindgen(js_name = addSphereCollider)]
    pub fn add_sphere_collider(&mut self, x: f32, y: f32, z: f32, radius: f32) {
        self.inner.colliders.push(Collider::Sphere {
            center: v(x, y, z),
            radius,
        });
    }
    #[wasm_bindgen(js_name = addBoxCollider)]
    #[allow(clippy::too_many_arguments)]
    pub fn add_box_collider(&mut self, x0: f32, y0: f32, z0: f32, x1: f32, y1: f32, z1: f32) {
        self.inner.colliders.push(Collider::Box {
            min: v(x0, y0, z0),
            max: v(x1, y1, z1),
        });
    }
    /// Colour (linear) and opacity over life: `[t, r, g, b, a, t, r, …]`
    /// with `t` from 0 (birth) to 1 (death).
    #[wasm_bindgen(js_name = setColorOverLife)]
    pub fn set_color_over_life(&mut self, keys: &[f32]) {
        let k: Vec<(f32, [f32; 4])> = keys
            .chunks_exact(5)
            .map(|c| (c[0], [c[1], c[2], c[3], c[4]]))
            .collect();
        if !k.is_empty() {
            self.inner.color_over_life = Gradient(k);
        }
    }
    /// One colour (linear) and opacity all its life.
    #[wasm_bindgen(js_name = setColor)]
    pub fn set_color(&mut self, r: f32, g: f32, b: f32, a: f32) {
        self.inner.color_over_life = Gradient::constant(Color::new(r, g, b), a);
    }
    /// Size over life as a multiple of each particle's own: `[t, s, t, s, …]`.
    #[wasm_bindgen(js_name = setSizeOverLife)]
    pub fn set_size_over_life(&mut self, keys: &[f32]) {
        let k: Vec<(f32, f32)> = keys.chunks_exact(2).map(|c| (c[0], c[1])).collect();
        if !k.is_empty() {
            self.inner.size_over_life = Curve(k);
        }
    }
    #[wasm_bindgen(js_name = setMaxParticles)]
    pub fn set_max_particles(&mut self, n: u32) {
        self.inner.max_particles = n as usize;
    }
    #[wasm_bindgen(js_name = setSeed)]
    pub fn set_seed(&mut self, seed: u32) {
        let look = self.inner.look;
        let s = std::mem::replace(&mut self.inner, ParticleSystem::new(look));
        self.inner = s.with_seed(seed as u64);
    }
}

impl Default for WebParticleSystem {
    fn default() -> Self {
        Self::new()
    }
}

fn downwind(dir: &str) -> Option<Downwind> {
    match dir {
        "+x" | "x" => Some(Downwind::PlusX),
        "-x" => Some(Downwind::MinusX),
        "+z" | "z" => Some(Downwind::PlusZ),
        "-z" => Some(Downwind::MinusZ),
        _ => None,
    }
}

#[wasm_bindgen]
impl WebScene {
    /// Add `system` to the scene, run by `stepParticles`. Its emitters are
    /// placed relative to its object: move the object and they follow.
    /// (The system is copied; later changes to it do not reach the scene.)
    #[wasm_bindgen(js_name = addParticles)]
    pub fn add_particles(&mut self, system: &WebParticleSystem) -> WebObjectHandle {
        let id = self
            .particle_animator
            .add(&mut self.inner, system.inner.clone());
        WebObjectHandle { id }
    }

    /// A new detached particle object running `system`, for the shim's
    /// scene mirror.
    #[wasm_bindgen(js_name = createParticles)]
    pub fn create_particles(&mut self, system: &WebParticleSystem) -> WebObjectHandle {
        let id = self.inner.arena.insert(particles_object(&system.inner));
        self.particle_animator.track(id, system.inner.clone());
        WebObjectHandle { id }
    }

    /// A snow cover over `sx` × `sz` metres round (`cx`, `cz`), `resolution`
    /// cells along its longer side, lying on whatever is in the scene below
    /// it — found at the next `stepParticles` by rays cast down from
    /// `castFrom` metres. Settle snowfall on it with `settleParticles`.
    #[wasm_bindgen(js_name = createSnowCover)]
    pub fn create_snow_cover(
        &mut self,
        cx: f32,
        cz: f32,
        sx: f32,
        sz: f32,
        resolution: u32,
        cast_from: f32,
    ) -> WebObjectHandle {
        let cover = SnowCover::new(Heightfield::new(
            [cx, cz],
            [sx, sz],
            resolution as usize,
            0.0,
        ));
        let id = self.inner.arena.insert(cover.object());
        self.particle_animator
            .track_snow(id, cover, Some(cast_from));
        WebObjectHandle { id }
    }

    /// A sand bed, as `createSnowCover`.
    #[wasm_bindgen(js_name = createSandBed)]
    pub fn create_sand_bed(
        &mut self,
        cx: f32,
        cz: f32,
        sx: f32,
        sz: f32,
        resolution: u32,
        cast_from: f32,
    ) -> WebObjectHandle {
        let bed = SandBed::new(Heightfield::new(
            [cx, cz],
            [sx, sz],
            resolution as usize,
            0.0,
        ));
        let id = self.inner.arena.insert(bed.object());
        self.particle_animator
            .track_sand(id, bed, (cast_from > 0.0).then_some(cast_from));
        WebObjectHandle { id }
    }

    /// `createSnowCover`, added to the scene's root.
    #[wasm_bindgen(js_name = addSnowCover)]
    pub fn add_snow_cover(
        &mut self,
        cx: f32,
        cz: f32,
        sx: f32,
        sz: f32,
        resolution: u32,
        cast_from: f32,
    ) -> WebObjectHandle {
        let h = self.create_snow_cover(cx, cz, sx, sz, resolution, cast_from);
        let root = self.inner.root;
        self.inner.arena.add_child(root, h.id);
        h
    }

    /// `createSandBed`, added to the scene's root.
    #[wasm_bindgen(js_name = addSandBed)]
    pub fn add_sand_bed(
        &mut self,
        cx: f32,
        cz: f32,
        sx: f32,
        sz: f32,
        resolution: u32,
        cast_from: f32,
    ) -> WebObjectHandle {
        let h = self.create_sand_bed(cx, cz, sx, sz, resolution, cast_from);
        let root = self.inner.root;
        self.inner.arena.add_child(root, h.id);
        h
    }

    /// Particles from `particles` land on and pile into `layer` (a snow
    /// cover or sand bed).
    #[wasm_bindgen(js_name = settleParticles)]
    pub fn settle_particles(&mut self, particles: &WebObjectHandle, layer: &WebObjectHandle) {
        self.particle_animator.settle(particles.id, layer.id);
    }

    /// A sheet of sand `depth` metres deep, give or take `roughness`, over a
    /// sand bed.
    #[wasm_bindgen(js_name = spreadSand)]
    pub fn spread_sand(&mut self, bed: &WebObjectHandle, depth: f32, roughness: f32) {
        if let Some(crate::particles::Layer::Sand(b)) = self.particle_animator.layer_mut(bed.id) {
            b.spread(depth, roughness);
        }
    }

    /// Wind over a sand bed or snow cover: `slabsPerSecond` of Werner's slab
    /// moves blowing toward `direction` (`"+x"`, `"-x"`, `"+z"`, `"-z"`) —
    /// dunes in sand, drifts in snow; 0 stills it.
    #[wasm_bindgen(js_name = setLayerWind)]
    pub fn set_layer_wind(
        &mut self,
        layer: &WebObjectHandle,
        direction: &str,
        slabs_per_second: f32,
    ) {
        let wind = downwind(direction)
            .filter(|_| slabs_per_second > 0.0)
            .map(|d| (d, slabs_per_second));
        self.particle_animator.set_layer_wind(layer.id, wind);
    }

    /// `slabs` of Werner's slab moves over a sand bed now, toward
    /// `direction`: a sheet shaped into dunes before it is first drawn.
    #[wasm_bindgen(js_name = blowSand)]
    pub fn blow_sand(&mut self, bed: &WebObjectHandle, direction: &str, slabs: u32) {
        let Some(d) = downwind(direction) else { return };
        if let Some(crate::particles::Layer::Sand(b)) = self.particle_animator.layer_mut(bed.id) {
            b.blow(d, slabs as usize);
        }
    }

    /// Advance every particle system, snow cover and sand bed `dt` seconds
    /// and redraw them, alpha-blended particles sorted from the camera at
    /// (`ex`, `ey`, `ez`). Once a frame, before rendering. (The three.js
    /// shim's renderer calls it for you.)
    #[wasm_bindgen(js_name = stepParticles)]
    pub fn step_particles(&mut self, dt: f32, ex: f32, ey: f32, ez: f32) {
        self.particle_animator
            .update(&mut self.inner, dt, Some(v(ex, ey, ez)));
    }

    /// `particles` drives `light` (a point or spot light's handle): it sits
    /// at the system's glowing centre with about `intensity`, flickering as
    /// the system's glow comes and goes — a fire lighting what is round it.
    #[wasm_bindgen(js_name = lightParticles)]
    pub fn light_particles(
        &mut self,
        particles: &WebObjectHandle,
        light: &WebObjectHandle,
        intensity: f32,
    ) {
        self.particle_animator
            .light_with(particles.id, light.id, intensity);
    }

    /// How strongly wind ripples mark a sand bed (0–1), lying across wind
    /// (`wx`, `wz`).
    #[wasm_bindgen(js_name = setSandRipples)]
    pub fn set_sand_ripples(&mut self, bed: &WebObjectHandle, strength: f32, wx: f32, wz: f32) {
        let Some(crate::particles::Layer::Sand(b)) = self.particle_animator.layer_mut(bed.id)
        else {
            return;
        };
        b.ripples = strength;
        b.ripple_wind = v(wx, 0.0, wz);
        let material = b.object();
        if let (Some(o), crate::core::ObjectKind::Mesh(m)) =
            (self.inner.get_mut(bed.id), material.kind)
        {
            if let crate::core::ObjectKind::Mesh(mesh) = &mut o.kind {
                mesh.material = m.material;
            }
        }
    }

    /// Particles alive in a system.
    #[wasm_bindgen(js_name = particleCount)]
    pub fn particle_count(&self, particles: &WebObjectHandle) -> u32 {
        self.particle_animator
            .system(particles.id)
            .map_or(0, |s| s.len() as u32)
    }

    /// Material lying in a snow cover or sand bed, m³.
    #[wasm_bindgen(js_name = layerVolume)]
    pub fn layer_volume(&self, layer: &WebObjectHandle) -> f32 {
        self.particle_animator
            .layer(layer.id)
            .map_or(0.0, |l| l.field().volume())
    }

    /// Particles a second from a system's first emitter: turn a stream on
    /// and off.
    #[wasm_bindgen(js_name = setParticleRate)]
    pub fn set_particle_rate(&mut self, particles: &WebObjectHandle, rate: f32) {
        if let Some(e) = self
            .particle_animator
            .system_mut(particles.id)
            .and_then(|s| s.emitters.first_mut())
        {
            e.rate = rate;
        }
    }

    /// `count` particles from a system's first emitter, now.
    #[wasm_bindgen(js_name = burstParticles)]
    pub fn burst_particles(&mut self, particles: &WebObjectHandle, count: u32) {
        if let Some(s) = self.particle_animator.system_mut(particles.id) {
            let t = s.time();
            if let Some(e) = s.emitters.first_mut() {
                e.bursts.push((t, count));
            }
        }
    }

    /// A system's wind, m/s.
    #[wasm_bindgen(js_name = setParticleWind)]
    pub fn set_particle_wind(&mut self, particles: &WebObjectHandle, x: f32, y: f32, z: f32) {
        if let Some(s) = self.particle_animator.system_mut(particles.id) {
            s.forces.wind = v(x, y, z);
        }
    }

    /// GPU weather (`threers::particles::Weather`): `count` snowflakes
    /// (`kind` "snow") or raindrops ("rain") filling a box `halfWidth` ×
    /// `height` × `halfDepth` above `floor`, kept over the camera, blown by
    /// the wind (`wx`, `wy`, `wz`). Placed in the vertex shader, so they
    /// cost nothing on the CPU; they do not land. A detached object, for the
    /// shim's scene mirror.
    #[wasm_bindgen(js_name = createWeather)]
    #[allow(clippy::too_many_arguments)]
    pub fn create_weather(
        &mut self,
        kind: &str,
        count: u32,
        half_width: f32,
        half_depth: f32,
        height: f32,
        floor: f32,
        wx: f32,
        wy: f32,
        wz: f32,
    ) -> WebObjectHandle {
        let mut w = if kind == "rain" {
            Weather::rain(count)
        } else {
            Weather::snow(count)
        };
        w.half_width = half_width;
        w.half_depth = half_depth;
        w.height = height;
        w.floor = floor;
        w.wind = v(wx, wy, wz);
        let id = self.inner.arena.insert(w.object());
        self.particle_animator.track_weather(id, w);
        WebObjectHandle { id }
    }

    /// Grow a dune field on sand bed `bed` (replacing its sand): crests
    /// about `wavelength` metres apart and `height` high, drawn out
    /// `stretch` times across the wind, worked by the wind toward
    /// `direction` (`'+x'`, …) for `shaping` slab moves a cell. See
    /// `threers::particles::dunes`.
    #[wasm_bindgen(js_name = shapeDunes)]
    #[allow(clippy::too_many_arguments)]
    pub fn shape_dunes(
        &mut self,
        bed: &WebObjectHandle,
        wavelength: f32,
        height: f32,
        stretch: f32,
        shaping: f32,
        seed: u32,
        direction: &str,
    ) {
        let Some(wind) = downwind(direction) else {
            return;
        };
        if let Some(crate::particles::Layer::Sand(b)) = self.particle_animator.layer_mut(bed.id) {
            b.dunes(crate::particles::dunes::DuneShape {
                wavelength,
                height,
                stretch,
                shaping,
                seed: seed as u64,
                wind,
                ..Default::default()
            });
        }
    }

    /// Sand bed `bed`'s look: its colour and its shadows' (linear), ripples
    /// (`ripples` 0–1, `rippleWavelength` metres), sand streaking up the
    /// windward faces (`streaks`, 0 for none), Journey's high-contrast
    /// shading (`stylize` 0–1) and the haze's glow toward the sun
    /// (`fogGlow`, as the desert sky's).
    #[wasm_bindgen(js_name = setSandLook)]
    #[allow(clippy::too_many_arguments)]
    pub fn set_sand_look(
        &mut self,
        bed: &WebObjectHandle,
        r: f32,
        g: f32,
        b: f32,
        sr: f32,
        sg: f32,
        sb: f32,
        ripples: f32,
        ripple_wavelength: f32,
        streaks: f32,
        stylize: f32,
        fog_glow: f32,
    ) {
        if let Some(crate::particles::Layer::Sand(s)) = self.particle_animator.layer_mut(bed.id) {
            s.color = Color::new(r, g, b);
            s.shadow_color = Color::new(sr, sg, sb);
            s.ripples = ripples;
            s.ripple_wavelength = ripple_wavelength;
            s.streaks = streaks;
            s.stylize = stylize;
            s.fog_glow = fog_glow;
            let s = s.clone();
            s.restyle(&mut self.inner);
        }
    }

    /// A desert sky dome `radius` metres across for the time of day `name`
    /// (`"dawn"`, `"day"`, `"sunset"`, `"night"`): keep it on the camera.
    /// A detached object, for the shim's scene mirror.
    #[wasm_bindgen(js_name = createDesertSky)]
    pub fn create_desert_sky(&mut self, name: &str, radius: f32) -> WebObjectHandle {
        let id = self
            .inner
            .arena
            .insert(crate::particles::dunes::sky_dome(&desert(name), radius));
        WebObjectHandle { id }
    }

    /// Change sky dome `sky` to the time of day `name`.
    #[wasm_bindgen(js_name = setDesertSky)]
    pub fn set_desert_sky(&mut self, sky: &WebObjectHandle, name: &str) {
        if let Some(o) = self.inner.get_mut(sky.id) {
            if let crate::core::ObjectKind::Mesh(m) = &mut o.kind {
                m.material = std::sync::Arc::new(desert(name).sky_material());
            }
        }
    }

    /// Sand blown off the crests of the dunes on `bed` into the wind
    /// (`wx`, `wy`, `wz`, m/s), `rate` puffs a second, in colour (`r`,
    /// `g`, `b`). A detached object, for the shim's scene mirror; call
    /// again after reshaping the dunes.
    #[wasm_bindgen(js_name = createSpindrift)]
    #[allow(clippy::too_many_arguments)]
    pub fn create_spindrift(
        &mut self,
        bed: &WebObjectHandle,
        wx: f32,
        wy: f32,
        wz: f32,
        rate: f32,
        r: f32,
        g: f32,
        b: f32,
    ) -> WebObjectHandle {
        let wind = v(wx, wy, wz);
        let along = if wx.abs() >= wz.abs() {
            if wx >= 0.0 {
                Downwind::PlusX
            } else {
                Downwind::MinusX
            }
        } else if wz >= 0.0 {
            Downwind::PlusZ
        } else {
            Downwind::MinusZ
        };
        let brinks = match self.particle_animator.layer(bed.id) {
            Some(crate::particles::Layer::Sand(s)) => {
                crate::particles::dunes::crests(&s.field, along, 18f32.to_radians(), 0.4, 3000)
            }
            _ => Vec::new(),
        };
        let system = presets::spindrift(brinks.into(), wind, rate, Color::new(r, g, b));
        let id = self.inner.arena.insert(particles_object(&system));
        self.particle_animator.track(id, system);
        WebObjectHandle { id }
    }

    /// Change `weather`'s wind.
    #[wasm_bindgen(js_name = setWeatherWind)]
    pub fn set_weather_wind(&mut self, weather: &WebObjectHandle, x: f32, y: f32, z: f32) {
        if let Some(w) = self.particle_animator.weather_mut(weather.id) {
            w.wind = v(x, y, z);
        }
    }
}

fn desert(name: &str) -> crate::particles::dunes::DesertTime {
    use crate::particles::dunes::DesertTime;
    match name {
        "dawn" => DesertTime::dawn(),
        "sunset" => DesertTime::sunset(),
        "night" => DesertTime::night(),
        _ => DesertTime::day(),
    }
}

/// The time of day `name` (`"dawn"`, `"day"`, `"sunset"`, `"night"`) over
/// the desert, for lighting a scene: `[sun x, y, z (toward it), sun r, g,
/// b, intensity, zenith r, g, b, horizon r, g, b, haze, glow, sky r, g, b,
/// ground r, g, b, sky intensity, sand r, g, b, shadow r, g, b, stars]`,
/// colours linear.
#[wasm_bindgen(js_name = desertTime)]
pub fn desert_time(name: &str) -> Vec<f32> {
    let t = desert(name);
    let s = t.sun.normalize();
    let c = |c: Color| [c.r, c.g, c.b];
    let mut out = vec![s.x, s.y, s.z];
    out.extend(c(t.sun_color));
    out.push(t.sun_intensity);
    out.extend(c(t.zenith));
    out.extend(c(t.horizon));
    out.extend([t.haze, t.glow]);
    out.extend(c(t.sky_light));
    out.extend(c(t.ground_light));
    out.push(t.sky_intensity);
    out.extend(c(t.sand));
    out.extend(c(t.shadow));
    out.push(t.stars);
    out
}
