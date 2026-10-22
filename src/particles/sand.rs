//! Sand: grains that pour and pile, and beds the wind shapes into dunes.
//!
//! Poured sand ([`presets::sand_stream`](super::presets::sand_stream)) falls as
//! grains — dense, so they cut through the air at a few metres a second — and
//! deposits where it lands. A [`SandBed`] takes the grains up and slumps
//! wherever the slope passes sand's **angle of repose**, about 34° for dry
//! sand: a poured pile grows as a cone at that angle, which is a property of
//! the material and not of how it was poured.
//!
//! Wind moves sand differently: grains hop (saltate) downwind in short jumps,
//! land, and are knocked on again — except in the lee of a crest, where the
//! air separates and they stop. [`SandBed::blow`] is **Werner's slab model**
//! of that (Werner, *Eolian dunes: computer simulations and attractor
//! interpretation*, Geology 23, 1995): slabs lifted at random, carried a hop
//! downwind at a time, settling more readily on sand than on bare ground and
//! always in a lee "shadow" behind a slope steeper than 15°, with avalanches
//! back to the angle of repose. Its dunes — barchans where sand is scarce,
//! transverse ridges where it is plentiful — form from a flat sheet, and
//! migrate downwind with a gentle windward slope and a steep slip face.

use crate::core::{Mesh, Object3D, ObjectId, ObjectKind};
use crate::math::Color;
use crate::scene::Scene;

use super::{Heightfield, ParticleSystem, Rng};

/// A downwind direction along one of the grid's axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Downwind {
    PlusX,
    MinusX,
    PlusZ,
    MinusZ,
}

impl Downwind {
    pub(crate) fn step(self) -> (i64, i64) {
        match self {
            Downwind::PlusX => (1, 0),
            Downwind::MinusX => (-1, 0),
            Downwind::PlusZ => (0, 1),
            Downwind::MinusZ => (0, -1),
        }
    }
}

/// Werner's model's parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Saltation {
    /// Slab thickness, metres: the unit of sand moved.
    pub slab: f32,
    /// Hop length, cells.
    pub hop: usize,
    /// Chance a slab settles where it lands on sand…
    pub on_sand: f32,
    /// …and on bare ground, where grains bounce on (less likely to stay).
    pub on_bare: f32,
    /// The lee shadow: behind a slope steeper than this (radians) the air
    /// separates and nothing is lifted, everything that lands stays.
    pub shadow: f32,
}

impl Default for Saltation {
    fn default() -> Self {
        Self {
            slab: 0.05,
            hop: 3,
            on_sand: 0.6,
            on_bare: 0.4,
            shadow: 15f32.to_radians(),
        }
    }
}

/// Sand lying on a heightfield.
#[derive(Debug, Clone)]
pub struct SandBed {
    pub field: Heightfield,
    /// Dry sand's: about 34°.
    pub angle_of_repose: f32,
    pub sweeps: usize,
    pub saltation: Saltation,
    /// The colour of the sand (linear).
    pub color: Color,
    /// The wind its ripples lie across (only the direction counts).
    pub ripple_wind: crate::math::Vector3,
    /// How strongly wind ripples mark it, 0 to 1: none on sand just poured,
    /// strong on a bed the wind has worked.
    pub ripples: f32,
    /// 0: physical shading. Toward 1, Journey's high-contrast sand, its
    /// shadowed side in `shadow_color`.
    pub stylize: f32,
    pub shadow_color: Color,
    /// Sand blown up the windward faces, streaking downwind: how strongly,
    /// 0 for none. Animated by [`advance`](Self::advance).
    pub streaks: f32,
    /// See [`SurfaceLook::fog_glow`](super::surface::SurfaceLook::fog_glow).
    pub fog_glow: f32,
    /// Ripples' spacing, metres (8 cm: wind ripples close up).
    pub ripple_wavelength: f32,
    rng: Rng,
    time: f32,
    object: Option<ObjectId>,
    drawn_version: u64,
}

impl SandBed {
    pub fn new(field: Heightfield) -> Self {
        Self {
            field,
            angle_of_repose: 34f32.to_radians(),
            sweeps: 24,
            saltation: Saltation::default(),
            color: Color::from_hex(0xd8b98a),
            ripple_wind: crate::math::Vector3::new(1.0, 0.0, 0.0),
            ripples: 0.7,
            stylize: 0.0,
            shadow_color: Color::new(0.34, 0.22, 0.13),
            streaks: 0.0,
            fog_glow: 0.0,
            ripple_wavelength: 0.08,
            rng: Rng::new(0x5a2d),
            time: 0.0,
            object: None,
            drawn_version: u64::MAX,
        }
    }

    /// A sheet of `depth` metres of sand everywhere, give or take `roughness`
    /// (a random bump per cell, so the wind has something to start from).
    pub fn spread(&mut self, depth: f32, roughness: f32) {
        for d in &mut self.field.depth {
            *d = (depth + (self.rng.next() * 2.0 - 1.0) * roughness).max(0.0);
        }
        self.field.touch_all();
        self.field.version += 1;
    }

    /// Take up the grains that have landed since the last call and let the
    /// bed slump.
    pub fn absorb(&mut self, grains: &mut ParticleSystem) {
        self.take_up(grains);
        self.settle();
    }

    /// Take up landed grains without slumping (see [`settle`](Self::settle)).
    pub fn take_up(&mut self, grains: &mut ParticleSystem) {
        for l in grains.take_landed() {
            self.field.deposit(l.position.x, l.position.z, l.volume);
        }
    }

    /// Slump to the angle of repose.
    pub fn settle(&mut self) {
        self.field.relax(self.angle_of_repose, self.sweeps);
    }

    /// [`settle`](Self::settle), looking at no more than `budget` cells:
    /// what is still moving carries on at the next call (see
    /// [`Heightfield::relax_within`]).
    pub fn settle_within(&mut self, budget: usize) {
        self.field
            .relax_within(self.angle_of_repose, self.sweeps, budget);
    }

    /// `slabs` slab moves of wind blowing `downwind`, then an avalanche back
    /// to the angle of repose. The sheet wraps at the grid's edges: what
    /// leaves downwind comes back in upwind, as on an endless sand sea.
    pub fn blow(&mut self, downwind: Downwind, slabs: usize) {
        self.carry(downwind, slabs);
        self.settle();
    }

    /// [`blow`](Self::blow) without slumping.
    pub(crate) fn carry(&mut self, downwind: Downwind, slabs: usize) {
        self.field
            .saltate(downwind, slabs, &self.saltation, &mut self.rng, true);
    }

    /// The bed's mesh, in a matte sand material, casting and receiving
    /// shadows.
    pub fn object(&self) -> Object3D {
        let mut look = super::surface::SurfaceLook::sand(self.color);
        look.wind = self.ripple_wind;
        look.ripple_strength = self.ripples;
        look.stylize = self.stylize;
        look.shadow_color = self.shadow_color;
        look.streaks = self.streaks;
        look.fog_glow = self.fog_glow;
        look.ripple_wavelength = self.ripple_wavelength;
        let mut o = Object3D::mesh(Mesh::new(self.field.mesh(), look.material()));
        o.cast_shadow = true;
        o.receive_shadow = true;
        o.name = "sand".into();
        o
    }

    /// Draw into `object` from now on (made by [`object`](Self::object)
    /// and added by the caller); the next [`sync`](Self::sync) redraws it.
    pub fn bind(&mut self, object: ObjectId) {
        self.object = Some(object);
        self.drawn_version = u64::MAX;
    }

    /// Add the bed's object to `scene`; [`sync`](Self::sync) keeps it
    /// current.
    pub fn add_to(&mut self, scene: &mut Scene) -> ObjectId {
        let id = scene.add(self.object());
        self.object = Some(id);
        self.drawn_version = self.field.version;
        id
    }

    /// Redraw the bed's material after its look (colour, ripples, streaks,
    /// stylize, haze) has changed.
    pub fn restyle(&self, scene: &mut Scene) {
        let Some(id) = self.object else { return };
        let material = self.object().kind;
        if let (Some(object), ObjectKind::Mesh(fresh)) = (scene.get_mut(id), material) {
            if let ObjectKind::Mesh(mesh) = &mut object.kind {
                mesh.material = fresh.material;
            }
        }
    }

    /// Run the streaks' clock `dt` seconds on (each frame, if there are
    /// streaks: they move as the wind does, not as the sand settles).
    pub fn advance(&mut self, scene: &mut Scene, dt: f32) {
        if self.streaks <= 0.0 {
            return;
        }
        self.time += dt;
        let Some(id) = self.object else { return };
        if let Some(object) = scene.get_mut(id) {
            if let ObjectKind::Mesh(mesh) = &mut object.kind {
                if let Some(m) = super::surface::with_time(&mesh.material, self.time) {
                    mesh.material = std::sync::Arc::new(m);
                }
            }
        }
    }

    /// Rewrite the bed's mesh if the sand has moved since it was drawn.
    pub fn sync(&mut self, scene: &mut Scene) {
        let Some(id) = self.object else { return };
        if self.drawn_version == self.field.version {
            return;
        }
        if let Some(object) = scene.get_mut(id) {
            if let ObjectKind::Mesh(mesh) = &mut object.kind {
                mesh.geometry = std::sync::Arc::new(self.field.mesh());
            }
        }
        self.drawn_version = self.field.version;
    }
}
