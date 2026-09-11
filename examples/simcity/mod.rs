//! The procedural city shared by `simcity`, `simcity_render` and
//! `simcity_metal`.
//!
//! Split into modules rather than one included file: at six thousand
//! lines the single `.inc` had become the main obstacle to changing
//! anything in it. Each entry point pulls this in with
//! `#[path = "simcity/mod.rs"] mod city;`.
#![allow(dead_code)]
// The entry points pull this in as `mod city`, and it has a `city` module of
// its own; renaming either would touch three binaries to satisfy a lint about
// a path nobody types.
#![allow(clippy::module_inception)]
// A procedural building takes a position, a footprint, a height, a palette and
// a seed. They are independent quantities and the call sites read better with
// them spelled out than bundled to get under seven.
#![allow(clippy::too_many_arguments)]

pub(crate) use std::f32::consts::{PI, TAU};
pub(crate) use std::sync::Arc;

pub(crate) use threers::materials::SkyMaterial;
pub(crate) use threers::lights::SpotLight;
pub(crate) use threers::{
    wgpu, BasicMaterial, BufferAttribute, BufferGeometry, Color, CubeTexture, DirectionalLight,
    HemisphereLight, InstancedMesh, Light, Material, Matrix4, Mesh, Object3D, ObjectId, ObjectKind,
    PhysicalMaterial, PmremGenerator, Quaternion, Scene, ShadowSettings,
    SphereGeometry, StandardMaterial, Texture, TextureFormat, TextureWrap, Vector3,
};

// --- The city's unit system -------------------------------------------------
// One world unit is one metre. Facades are laid out on a bay/storey grid so
// that window columns line up with building edges instead of being sliced.

/// Horizontal spacing of one window bay.
const BAY: f32 = 3.0;
/// Floor-to-floor height.
const FLOOR: f32 = 3.6;
/// Bays (and storeys) covered by one wrap of the facade texture.
const TILE: f32 = 8.0;
/// How much larger than life the moving population is drawn.
///
/// Metres are metres everywhere else in this city, and for the buildings that
/// is right. For the things that move it is a mistake: at the aerial camera
/// this example ships with, a correctly-scaled person is about two pixels
/// across and a car about fifteen. Every city game oversizes its traffic and
/// its crowds for exactly this reason — not from sloppiness, but because a
/// simulation nobody can see is a simulation that may as well not run.
///
/// Vehicles get less than people do: they queue against each other and share
/// lanes, so their size feeds back into the simulation, and `Vehicle::length`
/// is scaled to match so following distances stay honest.
pub(crate) const PERSON_SCALE: f32 = 1.45;
pub(crate) const VEHICLE_SCALE: f32 = 1.22;
pub(crate) const CRITTER_SCALE: f32 = 1.6;

/// Height of the kerb: sidewalks, plazas and parks sit this far above asphalt.
const KERB: f32 = 0.22;
/// Water sits this far below road level.
const RIVER_DEPTH: f32 = 1.1;
/// How far the surrounding terrain and the water run past the city limits, as
/// a multiple of the city's half-width. It has to stay beyond the fog's far
/// plane, or the edge of the ground plane shows as a seam against the sky —
/// which is why this is a ratio and not a constant.
///
/// It was 4.6 against a fog that reached full strength at 5.2, so the ground
/// stopped *before* the haze had finished swallowing it and left a hard band
/// along the skyline at every hour of the day. Everything past the fog is
/// fog-coloured, so making the plate larger costs eight quads and nothing
/// else.
/// This has to stay beyond the camera's *far plane*, not
/// merely beyond the fog. At 11 it did not: the plate stopped at 11 half-widths
/// with 7 half-widths of visible distance left behind it, so the two edges of
/// the square that run away from the camera projected as straight lines from
/// the horizon up into the sky — pale by day against the white haze, and an
/// unmistakable dark seam at night, when the fogged ground is grey and the sky
/// behind it is navy. It reads as a crack in the world, which is what it is.
///
/// The plate past the countryside is eight quads of flat colour, so the only
/// cost of covering the whole far plane is those quads.
/// This has to stay beyond the camera's *far plane* (`extent * 45`), not
/// merely beyond the fog. At 11 it did not: the plate stopped at 11 half-widths
/// with seven half-widths of visible distance behind it, so the two edges of
/// the square that run away from the camera projected as straight lines out of
/// the horizon — faint by day against the white haze, and at night an
/// unmistakable arc, the fogged ground being grey where the sky behind it is
/// navy. The plate past the countryside is eight quads of flat colour, so
/// covering the whole far plane costs those quads and nothing else.
const OUTSKIRTS_REACH: f32 = 55.0;
const OUTSKIRTS_MIN: f32 = 7000.0;
/// Fog reaches full strength here, as a multiple of the half-width. Must stay
/// comfortably inside `OUTSKIRTS_REACH` less however far out the camera can
/// stand, or the seam comes back.
/// The haze used to close in at 4.6 half-widths, which cut the world off just
/// past the city and left the background a wall of fog. Pushed out: the ground
/// plate and the camera's far plane both grew with it, and what fills the
/// extra distance is countryside drawn at a coarser and coarser grain.
/// The haze used to have one job — hide where the ground plate stopped — and
/// it was tuned tight enough to do it at four half-widths. The plate now runs
/// past the far plane, so nothing needs hiding, and the fog is free to be
/// atmosphere: it thins out over most of the visible distance instead of
/// closing the world off just past the city.
const FOG_FAR: f32 = 26.0;
const FOG_NEAR: f32 = 6.0;


pub(crate) mod rng;
pub(crate) mod maths;
pub(crate) mod mesh;
pub(crate) mod facade;
pub(crate) mod layout;
pub(crate) mod batches;
pub(crate) mod occupancy;
pub(crate) mod buildings;
pub(crate) mod props;
pub(crate) mod park;
pub(crate) mod country;
pub(crate) mod civic;
pub(crate) mod industry;
pub(crate) mod venues;
pub(crate) mod retail;
pub(crate) mod suburb;
pub(crate) mod roads;
pub(crate) mod tunnel;
pub(crate) mod zoo;
pub(crate) mod airport;
pub(crate) mod monuments;
pub(crate) mod creatures;
pub(crate) mod bake;
pub(crate) mod signs;
pub(crate) mod rail;
pub(crate) mod traffic;
pub(crate) mod sky;
pub(crate) mod water;
pub(crate) mod ibl;
pub(crate) mod city;

pub(crate) use rng::*;
pub(crate) use maths::*;
pub(crate) use mesh::*;
pub(crate) use facade::*;
pub(crate) use layout::*;
pub(crate) use batches::*;
pub(crate) use occupancy::*;
pub(crate) use buildings::*;
pub(crate) use props::*;
pub(crate) use park::*;
pub(crate) use country::*;
pub(crate) use civic::*;
pub(crate) use industry::*;
pub(crate) use venues::*;
pub(crate) use retail::*;
pub(crate) use suburb::*;
pub(crate) use roads::*;
pub(crate) use tunnel::*;
pub(crate) use zoo::*;
pub(crate) use airport::*;
pub(crate) use monuments::*;
pub(crate) use creatures::*;
pub(crate) use bake::*;
pub(crate) use signs::*;
pub(crate) use rail::*;
pub(crate) use traffic::*;
pub(crate) use sky::*;
pub(crate) use water::*;
pub(crate) use ibl::*;
pub(crate) use city::*;
