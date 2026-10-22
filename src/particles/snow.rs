//! Snow: flakes that fall, drift and settle, and the cover they build.
//!
//! [`presets::snowfall`](super::presets::snowfall) makes the flakes: light, so
//! their terminal speed is about a metre a second and every eddy of the
//! turbulence carries them, and set to [deposit](super::Response::Deposit)
//! where they land. A [`SnowCover`] takes those landings up into a
//! [`Heightfield`] laid over the scene — the ground, and the tops of whatever
//! stands on it, found by casting rays down — and lets the snow slump where it
//! is steeper than fresh snow holds (about 50°), so it slides off a roof's
//! edge rather than growing a wall there.
//!
//! Real snow falls a few centimetres an hour; a scene usually wants it faster.
//! The rate is in the flakes' [`volume`](super::Emitter::volume): see
//! `snowfall`'s `depth_per_second`.

use crate::core::{Mesh, Object3D, ObjectId, ObjectKind};
use crate::math::Color;
use crate::scene::Scene;

use super::sand::{Downwind, Saltation};
use super::{Heightfield, ParticleSystem, Rng};

/// Snow lying on a heightfield.
#[derive(Debug, Clone)]
pub struct SnowCover {
    pub field: Heightfield,
    /// The steepest slope snow holds, radians. Fresh, cohesive snow stands
    /// steeper than dry sand: about 50°.
    pub angle_of_repose: f32,
    /// Slumping passes per [`absorb`](Self::absorb).
    pub sweeps: usize,
    /// Each landing flake's snow is spread over a disc this wide (metres):
    /// in a time-lapse one drawn flake stands for a great many, landing all
    /// round it. 0 drops it where it fell.
    pub spread: f32,
    /// Snow thinner than this (metres) lets the ground show through, from
    /// bare at none to solid at this depth: a first dusting, not a white
    /// sheet. 0: solid at any depth.
    pub thin: f32,
    /// The colour of the snow (linear).
    pub color: Color,
    /// How wind drifts it (see [`drift`](Self::drift)): fine slabs, short
    /// hops, a 15° lee.
    pub saltation: Saltation,
    rng: Rng,
    object: Option<ObjectId>,
    drawn_version: u64,
}

impl SnowCover {
    pub fn new(field: Heightfield) -> Self {
        Self {
            field,
            angle_of_repose: 50f32.to_radians(),
            sweeps: 4,
            spread: 0.2,
            thin: 0.012,
            color: Color::from_hex(0xf2f5fa),
            saltation: Saltation {
                slab: 0.004,
                hop: 2,
                on_sand: 0.55,
                on_bare: 0.3,
                shadow: 15f32.to_radians(),
            },
            rng: Rng::new(0x5e0),
            object: None,
            drawn_version: u64::MAX,
        }
    }

    /// Take up the flakes that have landed since the last call and let the
    /// cover slump.
    pub fn absorb(&mut self, flakes: &mut ParticleSystem) {
        self.take_up(flakes);
        self.settle();
    }

    /// Take up landed flakes without slumping (see [`settle`](Self::settle)).
    pub fn take_up(&mut self, flakes: &mut ParticleSystem) {
        for l in flakes.take_landed() {
            self.field
                .deposit_spread(l.position.x, l.position.z, l.volume, self.spread);
        }
    }

    /// Slump where steeper than snow holds.
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

    /// The cover's mesh, in a matte snow material, casting and receiving
    /// shadows.
    pub fn object(&self) -> Object3D {
        let mut look = super::surface::SurfaceLook::snow(self.color);
        look.coverage = self.thin > 0.0;
        let mut o = Object3D::mesh(Mesh::new(self.mesh(), look.material()));
        o.cast_shadow = true;
        o.receive_shadow = true;
        o.name = "snow".into();
        o
    }

    /// Wind drifting the lying snow: `slabs` of Werner's slab moves
    /// toward `downwind` (see [`SandBed::blow`](super::sand::SandBed::blow)),
    /// with the snow's own slab size and hop — so snow is scoured from
    /// open ground and piles in the lee of whatever stands in the wind.
    /// Snow blown off the downwind edge is gone.
    pub fn drift(&mut self, downwind: Downwind, slabs: usize) {
        self.carry(downwind, slabs);
        self.settle();
    }

    /// [`drift`](Self::drift) without slumping.
    pub(crate) fn carry(&mut self, downwind: Downwind, slabs: usize) {
        self.field
            .saltate(downwind, slabs, &self.saltation, &mut self.rng, false);
    }

    /// The cover's mesh, see-through where thin (see [`thin`](Self::thin)).
    fn mesh(&self) -> crate::core::BufferGeometry {
        let thin = self.thin;
        self.field.mesh_with_coverage(|d| {
            if thin > 0.0 {
                let t = (d / thin).clamp(0.0, 1.0);
                t * t * (3.0 - 2.0 * t)
            } else {
                1.0
            }
        })
    }

    /// Draw into `object` from now on (made by [`object`](Self::object)
    /// and added by the caller); the next [`sync`](Self::sync) redraws it.
    pub fn bind(&mut self, object: ObjectId) {
        self.object = Some(object);
        self.drawn_version = u64::MAX;
    }

    /// Add the cover's object to `scene`; [`sync`](Self::sync) keeps it
    /// current.
    pub fn add_to(&mut self, scene: &mut Scene) -> ObjectId {
        let id = scene.add(self.object());
        self.object = Some(id);
        self.drawn_version = self.field.version;
        id
    }

    /// Rewrite the cover's mesh if the snow has changed since it was drawn.
    pub fn sync(&mut self, scene: &mut Scene) {
        let Some(id) = self.object else { return };
        if self.drawn_version == self.field.version {
            return;
        }
        if let Some(object) = scene.get_mut(id) {
            if let ObjectKind::Mesh(mesh) = &mut object.kind {
                mesh.geometry = std::sync::Arc::new(self.mesh());
            }
        }
        self.drawn_version = self.field.version;
    }
}
