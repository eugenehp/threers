//! Part of the `simcity` example; see `mod.rs`.
#![allow(dead_code)]

use super::*;

// ---------------------------------------------------------------------------
// Batches. One merged geometry per material; the whole city is ~20 draws.
// ---------------------------------------------------------------------------

#[derive(Default)]
pub(crate) struct Batches {
    /// Asphalt carriageway.
    pub(crate) road: MeshBuilder,
    /// Lane markings, crossings, parking stalls — flat, unlit paint.
    pub(crate) paint: MeshBuilder,
    /// Sidewalks, plazas, quay walls, bridge decks.
    pub(crate) pads: MeshBuilder,
    /// Parks and the terrain past the city limits.
    pub(crate) grass: MeshBuilder,
    pub(crate) water: MeshBuilder,
    /// Facades, one per architectural style.
    pub(crate) facades: [MeshBuilder; FACADE_STYLES],
    /// Untextured concrete/metal: plinths, parapets, roofs, plant, poles.
    pub(crate) trim: MeshBuilder,
    pub(crate) foliage: MeshBuilder,
    /// Vehicles at the kerb. Static, so they are merged geometry rather than
    /// instances: no draw call, no per-frame transform, and the paint is baked
    /// per vertex.
    pub(crate) parked: MeshBuilder,
    /// Fallen leaves and other flat scatter lying on the ground.
    ///
    /// Its own batch purely so the light bake can leave it out of the
    /// occluder tree. A leaf lying flat on a lawn blocks nothing, but there
    /// are tens of thousands of them and they doubled the time the bake spent
    /// tracing — they still *receive* shading, they just no longer cast it.
    pub(crate) litter: MeshBuilder,
    /// Standing water on the road. Its own batch because it is the one thing
    /// down there that is smooth.
    pub(crate) puddle: MeshBuilder,
    /// Unlit, night-only: the pools of light under street lamps.
    pub(crate) glow: MeshBuilder,
    /// How many surveillance cameras were mounted. Counted here because they
    /// are fitted from half a dozen unrelated places — signals, building
    /// corners, gates, platforms — and a feature scattered that widely is one
    /// nobody can eyeball the absence of.
    pub(crate) cameras: usize,
    /// Smoke and steam.
    ///
    /// Its own batch because it needs the treatment the cloud deck gets: the
    /// underside of a plume faces the ground, so the only light reaching it is
    /// the hemisphere's *ground* colour — which is grass — and left in `trim`
    /// every column of steam came out with olive-green shadows banding it. The
    /// clouds hit this exact problem and it was solved there by putting a wash
    /// of sky colour into the material's emissive; this batch exists so plumes
    /// can share that.
    pub(crate) plume: MeshBuilder,
    /// Unlit aircraft warning lights, which blink day and night.
    pub(crate) beacon: MeshBuilder,
    /// Neon tube and backlit sign faces: unlit, saturated, night only. Its own
    /// batch rather than part of `glow` because a lamp pool wants to be dim
    /// and a neon sign wants to be the brightest thing on the street.
    pub(crate) neon: MeshBuilder,
    /// The same, for signs that blink. Shown on one half of the signal cycle.
    pub(crate) neon_blink: MeshBuilder,
    /// Translucent cones under the street lamps. The one thing here that is
    /// not opaque, and the reason a lit street reads as lit rather than as a
    /// disc of paint on the tarmac.
    pub(crate) shaft: MeshBuilder,
    /// Traffic-signal lenses, one builder per half of the cycle. Exactly one
    /// is visible at a time, which is the whole of the signal logic.
    pub(crate) signal: [MeshBuilder; 2],
    /// What ground is already taken.
    ///
    /// Carried on the batches for the same reason the seats are: every
    /// placement function already has these in hand, so nothing needs a new
    /// parameter threaded through a dozen call sites.
    pub(crate) occ: Occupancy,
    /// Where a uniformed figure should stand: `(x, z, kind)` with 0 police
    /// and 1 firefighter. Collected the same way the benches are.
    pub(crate) posts: Vec<(f32, f32, u8)>,
    /// Every bench in the city as `(x, z, yaw)`.
    ///
    /// Collected here rather than threaded back through a dozen call sites:
    /// `add_bench` already has the batches in hand. The crowd generator reads
    /// it afterwards and puts people on them, which is the difference between
    /// a park with seating and a park where everyone is commuting.
    pub(crate) seats: Vec<(f32, f32, f32)>,
}

pub(crate) const C_ASPHALT: u32 = 0x393c42;
pub(crate) const C_PAINT: u32 = 0xb0ab9a;
pub(crate) const C_PAINT_WARN: u32 = 0xc9a227;
pub(crate) const C_SIDEWALK: u32 = 0x8d8b85;
pub(crate) const C_CONCRETE: u32 = 0x9a978f;
pub(crate) const C_ROOF: u32 = 0x3b3d40;
pub(crate) const C_GRASS: u32 = 0x4a6b3a;
pub(crate) const C_TERRAIN: u32 = 0x62704a;
pub(crate) const C_WATER: u32 = 0x16303f;
pub(crate) const C_TRUNK: u32 = 0x4b3b2c;
