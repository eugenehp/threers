//! Running a scene's particles: every system, the snow and sand they settle
//! into, stepped and redrawn once a frame.
//!
//! ```no_run
//! use threers::particles::{presets, ParticleAnimator};
//! use threers::math::Vector3;
//! use threers::Scene;
//!
//! let mut scene = Scene::new();
//! let mut particles = ParticleAnimator::new();
//! let torch = particles.add(&mut scene, presets::fire(Vector3::ZERO, 0.15));
//! // Move the torch: its fire follows.
//! scene.get_mut(torch).unwrap().position = Vector3::new(1.0, 1.5, 0.0);
//! particles.update(&mut scene, 1.0 / 60.0, Some(Vector3::new(0.0, 1.6, 4.0)));
//! ```

use crate::core::ObjectId;
use crate::math::Vector3;
use crate::scene::Scene;

use super::sand::{Downwind, SandBed};
use super::snow::SnowCover;
use super::{Heightfield, ParticleEffect, ParticleSystem, Weather};

/// Snow or sand, lying where particles land.
#[derive(Debug, Clone)]
pub enum Layer {
    Snow(SnowCover),
    Sand(SandBed),
}

impl Layer {
    pub fn field(&self) -> &Heightfield {
        match self {
            Layer::Snow(s) => &s.field,
            Layer::Sand(s) => &s.field,
        }
    }
    pub fn field_mut(&mut self) -> &mut Heightfield {
        match self {
            Layer::Snow(s) => &mut s.field,
            Layer::Sand(s) => &mut s.field,
        }
    }
    fn take_up(&mut self, system: &mut ParticleSystem) {
        match self {
            Layer::Snow(s) => s.take_up(system),
            Layer::Sand(s) => s.take_up(system),
        }
    }
    /// Slump, looking at no more than eight passes' worth of the grid: a
    /// dune field under wind always has a slip face avalanching somewhere,
    /// and a frame should cost a known amount.
    fn settle_some(&mut self) {
        let f = self.field();
        let budget = 8 * f.nx * f.nz;
        match self {
            Layer::Snow(s) => s.settle_within(budget),
            Layer::Sand(s) => s.settle_within(budget),
        }
    }
    fn sync(&mut self, scene: &mut Scene) {
        match self {
            Layer::Snow(s) => s.sync(scene),
            Layer::Sand(s) => s.sync(scene),
        }
    }
}

#[derive(Debug, Clone)]
struct Effect {
    fx: ParticleEffect,
    /// Each emitter's position relative to the object, as given.
    anchors: Vec<Vector3>,
    /// The layer it lands on, if any (its object id).
    settles_on: Option<ObjectId>,
    /// A light it drives (see [`ParticleAnimator::light_with`]).
    light: Option<DrivenLight>,
}

#[derive(Debug, Clone)]
struct DrivenLight {
    object: ObjectId,
    intensity: f32,
    /// The system's glow, averaged over the last second or two: what the
    /// light's intensity is relative to.
    mean: f32,
    /// Where it sits, eased toward the glow's centre.
    at: Option<Vector3>,
}

#[derive(Debug, Clone)]
struct LayerEntry {
    layer: Layer,
    object: ObjectId,
    /// Find the base by casting rays into the scene at the next update (the
    /// scene may not hold its objects yet when the layer is made), from this
    /// height.
    base_from: Option<f32>,
    /// Wind shaping the sand: a direction and slab moves a second.
    wind: Option<(Downwind, f32)>,
    carried: f32,
    /// Seconds since it last slumped, and since it was last redrawn.
    since_settled: f32,
    since_drawn: f32,
}

/// Layers slump and redraw this often, not every frame: material lands a
/// little at a time and a pile changes slowly, while slumping and rebuilding
/// a mesh of a grid's every cell are what a frame would otherwise spend its
/// time on.
const LAYER_INTERVAL: f32 = 1.0 / 15.0;

/// How often a snow cover's mesh is rebuilt and uploaded.
const SNOW_REDRAW: f32 = 0.25;

/// Every particle system in a scene, and the layers they settle into.
#[derive(Debug, Clone, Default)]
pub struct ParticleAnimator {
    effects: Vec<Effect>,
    layers: Vec<LayerEntry>,
    weather: Vec<(ObjectId, Weather)>,
    time: f32,
}

impl ParticleAnimator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add `system`'s object to `scene` and run it. Its emitters are placed
    /// relative to the object: move the object and they follow. Returns the
    /// object's id.
    pub fn add(&mut self, scene: &mut Scene, system: ParticleSystem) -> ObjectId {
        let anchors = system.emitters.iter().map(|e| e.position).collect();
        let fx = ParticleEffect::add_to(scene, system);
        let id = fx.object;
        self.effects.push(Effect {
            fx,
            anchors,
            settles_on: None,
            light: None,
        });
        id
    }

    /// Run `system`, drawn by `object` — already in the scene, made by
    /// [`particles_object`](super::particles_object) (for a scene built
    /// node by node, as the browser bindings do).
    pub fn track(&mut self, object: ObjectId, system: ParticleSystem) {
        let anchors = system.emitters.iter().map(|e| e.position).collect();
        self.effects.push(Effect {
            fx: ParticleEffect {
                system,
                object,
                caster: None,
            },
            anchors,
            settles_on: None,
            light: None,
        });
    }

    /// Keep `cover` drawn by `object` (made by [`SnowCover::object`]),
    /// already in the scene; `base_from` as for [`add_snow`](Self::add_snow).
    pub fn track_snow(&mut self, object: ObjectId, mut cover: SnowCover, base_from: Option<f32>) {
        cover.bind(object);
        self.layers.push(LayerEntry {
            layer: Layer::Snow(cover),
            object,
            base_from,
            wind: None,
            carried: 0.0,
            since_settled: 0.0,
            since_drawn: LAYER_INTERVAL,
        });
    }

    /// Keep `bed` drawn by `object` (made by [`SandBed::object`]), already
    /// in the scene; `base_from` as for [`add_snow`](Self::add_snow).
    pub fn track_sand(&mut self, object: ObjectId, mut bed: SandBed, base_from: Option<f32>) {
        bed.bind(object);
        self.layers.push(LayerEntry {
            layer: Layer::Sand(bed),
            object,
            base_from,
            wind: None,
            carried: 0.0,
            since_settled: 0.0,
            since_drawn: LAYER_INTERVAL,
        });
    }

    /// Add a snow cover's object to `scene`. With `base_from`, the cover's
    /// base is found at the next update by casting rays down from that
    /// height into everything else in the scene. Returns its object's id.
    pub fn add_snow(
        &mut self,
        scene: &mut Scene,
        mut cover: SnowCover,
        base_from: Option<f32>,
    ) -> ObjectId {
        let object = cover.add_to(scene);
        self.layers.push(LayerEntry {
            layer: Layer::Snow(cover),
            object,
            base_from,
            wind: None,
            carried: 0.0,
            since_settled: 0.0,
            since_drawn: LAYER_INTERVAL,
        });
        object
    }

    /// Add a sand bed's object to `scene`; `base_from` as for
    /// [`add_snow`](Self::add_snow).
    pub fn add_sand(
        &mut self,
        scene: &mut Scene,
        mut bed: SandBed,
        base_from: Option<f32>,
    ) -> ObjectId {
        let object = bed.add_to(scene);
        self.layers.push(LayerEntry {
            layer: Layer::Sand(bed),
            object,
            base_from,
            wind: None,
            carried: 0.0,
            since_settled: 0.0,
            since_drawn: LAYER_INTERVAL,
        });
        object
    }

    /// Particles from `system` (an id from [`add`](Self::add)) land on and
    /// deposit into `layer` (an id from `add_snow` or `add_sand`).
    pub fn settle(&mut self, system: ObjectId, layer: ObjectId) {
        if let Some(e) = self.effects.iter_mut().find(|e| e.fx.object == system) {
            e.settles_on = Some(layer);
        }
    }

    /// Let `system` (an id from [`add`](Self::add)) drive `light` (a point
    /// or spot light in the scene): it sits at the system's glowing centre
    /// and its intensity follows how much of it is alight, about `intensity`
    /// on average — so a fire lights what is round it, flickering as its
    /// flames come and go, and dies down as they do.
    pub fn light_with(&mut self, system: ObjectId, light: ObjectId, intensity: f32) {
        if let Some(e) = self.effects.iter_mut().find(|e| e.fx.object == system) {
            e.light = Some(DrivenLight {
                object: light,
                intensity,
                mean: 0.0,
                at: None,
            });
        }
    }

    /// A steady wind over a layer: `slabs_per_second` of Werner's slab
    /// moves blowing `downwind` — dunes in sand ([`SandBed::blow`]), drifts
    /// in snow ([`SnowCover::drift`]). `None` stills it.
    pub fn set_layer_wind(&mut self, layer: ObjectId, wind: Option<(Downwind, f32)>) {
        if let Some(l) = self.layers.iter_mut().find(|l| l.object == layer) {
            l.wind = wind;
        }
    }

    pub fn system(&self, id: ObjectId) -> Option<&ParticleSystem> {
        self.effects
            .iter()
            .find(|e| e.fx.object == id)
            .map(|e| &e.fx.system)
    }
    pub fn system_mut(&mut self, id: ObjectId) -> Option<&mut ParticleSystem> {
        self.effects
            .iter_mut()
            .find(|e| e.fx.object == id)
            .map(|e| &mut e.fx.system)
    }
    pub fn layer(&self, id: ObjectId) -> Option<&Layer> {
        self.layers
            .iter()
            .find(|l| l.object == id)
            .map(|l| &l.layer)
    }
    pub fn layer_mut(&mut self, id: ObjectId) -> Option<&mut Layer> {
        self.layers
            .iter_mut()
            .find(|l| l.object == id)
            .map(|l| &mut l.layer)
    }

    /// Stop running `id` (a system or a layer). Its object stays in the
    /// scene for the caller to remove.
    pub fn untrack(&mut self, id: ObjectId) {
        self.effects.retain(|e| e.fx.object != id);
        self.layers.retain(|l| l.object != id);
        self.weather.retain(|w| w.0 != id);
    }

    /// Add GPU [`Weather`] to `scene`, its box kept over the eye. Returns
    /// its object's id.
    pub fn add_weather(&mut self, scene: &mut Scene, weather: Weather) -> ObjectId {
        let id = scene.add(weather.object());
        self.weather.push((id, weather));
        id
    }

    /// Run `weather`, drawn by `object` (made by [`Weather::object`],
    /// already in the scene).
    pub fn track_weather(&mut self, object: ObjectId, weather: Weather) {
        self.weather.push((object, weather));
    }

    /// The weather drawn by `id`, to change (its wind, say).
    pub fn weather_mut(&mut self, id: ObjectId) -> Option<&mut Weather> {
        self.weather
            .iter_mut()
            .find(|w| w.0 == id)
            .map(|w| &mut w.1)
    }

    /// Step everything `dt` seconds and redraw it, alpha-blended particles
    /// sorted from `eye`.
    pub fn update(&mut self, scene: &mut Scene, dt: f32, eye: Option<Vector3>) {
        // Where the objects are now: emitters follow theirs.
        scene.update_world();
        // Layers that still need their base: everything in the scene but the
        // particles and the layers themselves.
        if self.layers.iter().any(|l| l.base_from.is_some()) {
            let ours: Vec<ObjectId> = self
                .effects
                .iter()
                .map(|e| e.fx.object)
                .chain(self.layers.iter().map(|l| l.object))
                .chain(self.weather.iter().map(|w| w.0))
                .collect();
            let solids: Vec<ObjectId> = scene
                .arena
                .get(scene.root)
                .map(|r| {
                    r.children
                        .iter()
                        .copied()
                        .filter(|c| !ours.contains(c))
                        .collect()
                })
                .unwrap_or_default();
            for l in &mut self.layers {
                if let Some(from) = l.base_from.take() {
                    l.layer
                        .field_mut()
                        .base_from_scene(&scene.arena, &solids, from);
                }
            }
        }
        self.time += dt;
        let centre = eye.unwrap_or(Vector3::ZERO);
        for (id, w) in &self.weather {
            w.update(scene, *id, self.time, centre);
        }
        for e in &mut self.effects {
            // A system tracked node by node gets its shadow caster here, as
            // its object's child.
            if e.fx.system.look.casts_shadow
                && e.fx.caster.is_none()
                && scene.get(e.fx.object).is_some()
            {
                e.fx.caster = Some(scene.add_to(e.fx.object, super::render::caster_object()));
            }
            // Emitters follow their object.
            if let Some(o) = scene.get(e.fx.object) {
                let m = o.matrix_world;
                for (emitter, anchor) in e.fx.system.emitters.iter_mut().zip(&e.anchors) {
                    emitter.position = anchor.apply_matrix4(&m);
                }
            }
            let layer = e
                .settles_on
                .and_then(|id| self.layers.iter_mut().find(|l| l.object == id));
            match layer {
                Some(l) => {
                    e.fx.system.step(dt, Some(l.layer.field()));
                    l.layer.take_up(&mut e.fx.system);
                }
                None => e.fx.system.step(dt, None),
            }
            e.fx.sync(scene, eye);
            if let Some(dl) = &mut e.light {
                drive_light(dl, &e.fx.system, scene, dt);
            }
        }
        for l in &mut self.layers {
            if let Some((downwind, rate)) = l.wind {
                l.carried += rate * dt;
                let slabs = l.carried.floor();
                l.carried -= slabs;
                if slabs > 0.0 {
                    match &mut l.layer {
                        Layer::Sand(bed) => bed.carry(downwind, slabs as usize),
                        Layer::Snow(cover) => cover.carry(downwind, slabs as usize),
                    }
                }
            }
            if let Layer::Sand(bed) = &mut l.layer {
                bed.advance(scene, dt);
            }
            l.since_settled += dt;
            l.since_drawn += dt;
            if l.since_settled >= LAYER_INTERVAL {
                l.since_settled = 0.0;
                l.layer.settle_some();
            }
            // Snow grows millimetres a second: a few redraws a second show
            // it. A pile being poured needs more.
            let redraw = match l.layer {
                Layer::Snow(_) => SNOW_REDRAW,
                Layer::Sand(_) => LAYER_INTERVAL,
            };
            if l.since_drawn >= redraw {
                l.since_drawn = 0.0;
                l.layer.sync(scene);
            }
        }
    }
}

/// A system's glow — each particle's opacity, times its black-body
/// brightness if it has a temperature — and where its centre is.
fn glow(system: &ParticleSystem) -> (f32, Vector3) {
    let ps = &system.particles;
    let (mut total, mut centre) = (0.0f32, Vector3::ZERO);
    for i in 0..ps.len() {
        let t = system.life_fraction(i);
        let mut w = system.color_over_life.at(t)[3].max(0.0);
        if let Some(k) = &system.temperature_over_life {
            w *= (k.at(t) / 1500.0).powi(4);
        }
        total += w;
        centre = centre + ps.position[i] * w;
    }
    (
        total,
        if total > 0.0 {
            centre * (1.0 / total)
        } else {
            Vector3::ZERO
        },
    )
}

fn drive_light(dl: &mut DrivenLight, system: &ParticleSystem, scene: &mut Scene, dt: f32) {
    let (now, centre) = glow(system);
    // A running mean over about 1.5 s; the light is the glow relative to it.
    let k = (dt / 1.5).min(1.0);
    dl.mean = if dl.mean <= 0.0 {
        now
    } else {
        dl.mean + (now - dl.mean) * k
    };
    let level = if dl.mean > 1e-6 {
        (now / dl.mean).clamp(0.0, 2.0)
    } else {
        0.0
    };
    let at = match dl.at {
        Some(p) => p + (centre - p) * (dt * 6.0).min(1.0),
        None => centre,
    };
    dl.at = Some(at);
    if let Some(o) = scene.get_mut(dl.object) {
        if now > 0.0 {
            o.position = at;
        }
        let i = dl.intensity * level;
        match &mut o.kind {
            crate::core::ObjectKind::Light(crate::lights::Light::Point(l)) => l.intensity = i,
            crate::core::ObjectKind::Light(crate::lights::Light::Spot(l)) => l.intensity = i,
            _ => {}
        }
    }
}
