//! Part of the `simcity` example; see `mod.rs`.
#![allow(dead_code)]

use super::*;

// ---------------------------------------------------------------------------
// The city.
// ---------------------------------------------------------------------------

pub(crate) struct CityParams {
    pub(crate) seed: u64,
    /// Blocks per side. One column of them may be the river.
    pub(crate) blocks: usize,
    pub(crate) cars: usize,
    pub(crate) layout: Layout,
    /// Run the raycast light bake. On by default; off is for callers that
    /// build a great many cities and do not look at any of them — the test
    /// suite builds twenty-eight, and baking each of them takes the run from
    /// a second to ten minutes.
    pub(crate) bake: bool,
}

impl Default for CityParams {
    fn default() -> Self {
        Self {
            seed: 12,
            blocks: 8,
            cars: 240,
            layout: Layout::Manhattan,
            bake: true,
        }
    }
}

/// Spot lights the renderer will accept. The first is the only one that can
/// cast a shadow.
pub(crate) const MAX_SPOTS: usize = 4;

/// A crossroads, as a box on the ground.
pub(crate) struct Junction {
    pub(crate) cx: f32,
    pub(crate) cz: f32,
    pub(crate) hx: f32,
    pub(crate) hz: f32,
    /// Signalled crossings are governed by the lights, not by give-way.
    pub(crate) signalled: bool,
    /// Span index of the road running along Z (its x is `cx`) and of the one
    /// running along X (its z is `cz`) — the lane ids a turn has to adopt.
    pub(crate) road_z: usize,
    pub(crate) road_x: usize,
    /// Whether each of those is an avenue, and so signalled.
    pub(crate) avenue_z: bool,
    pub(crate) avenue_x: bool,
    /// How far a vehicle may travel on each after turning onto it. The X range
    /// is the bank this junction is on where a street stops at the water.
    pub(crate) x_lo: f32,
    pub(crate) x_hi: f32,
    pub(crate) z_lo: f32,
    pub(crate) z_hi: f32,
}

/// One instanced mesh and the run of movers it may draw.
pub(crate) struct Group {
    pub(crate) mesh: ObjectId,
    pub(crate) start: usize,
    pub(crate) count: usize,
    /// Which `pose * variants + tone` slot this mesh holds. Meshes with other
    /// slots cover the same run of movers; each mover lands in exactly one.
    pub(crate) pose: usize,
}

/// One population of movers sharing a gait.
///
/// LEVEL OF DETAIL, in a renderer whose budget is draw calls rather than
/// triangles. Swapping in a simpler mesh per level would *add* a draw call per
/// level per colour, which is the wrong direction here. So the far level is a
/// single shared impostor mesh — one draw for the whole population, whatever
/// its colour — and everything past `lod` metres goes into it. The near model
/// can then be far better than it could otherwise afford, because only the
/// instances close enough to see it are paying for it.
///
/// Instances not drawn by a given mesh get a zero-scale matrix, which
/// collapses to a point and rasterises nothing.
pub(crate) struct Fleet {
    pub(crate) movers: Vec<Mover>,
    /// Paint per mover, applied as a per-instance tint. Parallel to `movers`.
    pub(crate) paints: Vec<Color>,
    pub(crate) groups: Vec<Group>,
    /// Shared low-detail mesh for everything past `lod`.
    pub(crate) far: Option<ObjectId>,
    pub(crate) poses: usize,
    /// How many skin-tone meshes each pose is split across. `1` for anything
    /// that does not vary.
    pub(crate) variants: usize,
    /// Metres from the camera at which an instance drops to the impostor.
    pub(crate) lod: f32,
    pub(crate) gait: Gait,
}

/// The animated water: the mesh the compute pass writes, and what it needs to
/// know to write it. Unused on the Metal path, which has no WGSL to run.
#[allow(dead_code)]
pub(crate) struct WaterSurface {
    pub(crate) id: ObjectId,
    pub(crate) geometry: Arc<BufferGeometry>,
    pub(crate) vertices: u32,
    pub(crate) level: f32,
}

#[allow(dead_code)]
pub(crate) struct CityStats {
    /// The ceiling on mover triangles — every instance at full detail. What is
    /// drawn is this scaled by the level-of-detail split, which moves.
    pub(crate) mover_triangles: usize,
    /// Blocks per side actually used — `CityParams::blocks` after clamping.
    pub(crate) blocks: usize,
    pub(crate) buildings: usize,
    /// How many of the four civic buildings found a block to stand on. Fewer
    /// than four means the size or zone criteria were too tight for this
    /// layout, which is invisible in a render and obvious here.
    pub(crate) civic: usize,
    /// What the infrastructure pass placed. Counted for the same reason the
    /// civic buildings are: a power station that silently failed to find a
    /// clear footprint looks exactly like one that was never asked for.
    pub(crate) works: Works,
    /// Blocks laid out as a close or a crescent rather than as a grid of lots.
    pub(crate) suburbs: usize,
    pub(crate) walkers: usize,
    /// People placed and never moved: on benches, standing about.
    pub(crate) idlers: usize,
    /// `(ground claimed, placements refused for overlapping something)`.
    pub(crate) footprints: (usize, usize),
    pub(crate) triangles: usize,
    pub(crate) draws: usize,
    pub(crate) cars: usize,
    /// Half the width of the built area, in metres.
    pub(crate) extent: f32,
    /// What the raycast light bake did, or that it was compiled out.
    pub(crate) bake: BakeReport,
}

pub(crate) struct City {
    pub(crate) stats: CityStats,
    /// Facade meshes — their emissive intensity tracks the clock.
    pub(crate) windows: Vec<(ObjectId, f32)>,
    /// Unlit geometry that only exists after dark.
    pub(crate) night_only: Vec<ObjectId>,
    pub(crate) beacons: Vec<ObjectId>,
    /// Neon and backlit signage, and whether each batch blinks.
    pub(crate) neon: Vec<(ObjectId, bool)>,
    /// The translucent cones under the street lamps.
    pub(crate) shafts: Option<ObjectId>,
    /// The cloud deck.
    pub(crate) clouds: Option<ObjectId>,
    /// Smoke and steam, lit like the clouds are.
    pub(crate) plumes: Option<ObjectId>,
    /// Stars and the moon. Their own object because they must not be dimmed
    /// along with the lamp pools — a star is either there or it is not.
    pub(crate) night_sky: Option<ObjectId>,
    /// Every moving population: cars, buses, lorries, people, boats, dogs.
    pub(crate) fleets: Vec<Fleet>,
    /// Birds, pigeons and waterfowl. Not lane-bound, so not fleets.
    pub(crate) swarms: Vec<Swarm>,
    /// Which fleet the car lamps follow.
    pub(crate) car_fleet: usize,
    /// Which fleet the skin meshes follow.
    pub(crate) people_fleet: usize,
    /// `(mesh, pose)` for the shared skin overlay: one mesh a walk pose,
    /// covering every pedestrian whatever their coat.
    pub(crate) walker_skin: Vec<(ObjectId, usize)>,
    /// `(fleet, mover)` for every road vehicle, grouped by lane. Built once:
    /// a vehicle never changes lane, so the grouping never changes.
    /// Scratch for the traffic pass: `(lane, position, fleet, mover)`, sorted
    /// once a frame. Grouping cannot be precomputed any more — a vehicle that
    /// turns leaves its lane, and a stale index would have it queueing behind
    /// traffic on a road it is no longer on.
    pub(crate) lane_scratch: Vec<(u16, f32, u32, u32)>,
    /// Scratch for sorting one lane by position. Kept on the struct so the
    /// per-frame traffic pass allocates nothing.
    /// Signalled crossings as `(centre, half width)` along each axis: where a
    /// vehicle travelling that way has to stop when its phase is red.
    pub(crate) stops_along_x: Vec<(f32, f32)>,
    pub(crate) stops_along_z: Vec<(f32, f32)>,
    /// Every crossroads, and a scratch marking which are occupied this frame.
    pub(crate) junctions: Vec<Junction>,
    pub(crate) busy: Vec<u8>,
    /// The river or bay surface, when this layout has one.
    #[allow(dead_code)]
    pub(crate) water: Option<WaterSurface>,
    /// The x range the water occupies, for anything that has to keep out of
    /// it. `None` when the layout is landlocked.
    pub(crate) water_span: Option<(f32, f32)>,
    /// `(centre z, half width)` of every road that bridges the water. The only
    /// places a vehicle may legitimately be over it.
    pub(crate) bridges: Vec<(f32, f32)>,
    pub(crate) dome: ObjectId,
    /// Every street lamp's bulb position, so the nearest few can be lit for
    /// real rather than faked with an unlit disc.
    pub(crate) lamps: Vec<Vector3>,
    pub(crate) spots: Vec<ObjectId>,
    /// `(mesh, first mover, count)`. Movers are sorted by colour, so each
    /// instanced mesh owns a contiguous run.
    /// Head and tail lamps and the beam on the road: instanced meshes that
    /// track the cars, with the emissive strength each reaches after dark.
    pub(crate) car_lights: Vec<(ObjectId, f32)>,
    /// Every lamp mesh with what it is and how bright it burns after dark.
    /// The last field is a bitmask of the fleets a lamp applies to; 0 means
    /// every road fleet. Beacons belong to the three emergency fleets and
    /// nothing else.
    pub(crate) car_lamps: Vec<(ObjectId, Lamp, f32, u32)>,
    /// The two halves of the signal cycle: exactly one is visible at a time.
    pub(crate) signal_phases: [Option<ObjectId>; 2],
    /// A ready-made eye/target pair standing on an avenue looking downtown.
    pub(crate) vista: (Vector3, Vector3),
    pub(crate) sun: ObjectId,
    pub(crate) moon: ObjectId,
    pub(crate) hemi: ObjectId,
    /// Which slice of the day the current environment cube was built for.
    /// Rebuilding it every frame is pure waste; the sky does not move that
    /// fast, and a cube upload is not free.
    pub(crate) env_slot: Option<i32>,
    /// What fraction of the population is out, from the clock.
    ///
    /// The time of day was a slider with nothing behind it: the same number of
    /// cars at three in the morning as at nine, the same crowds. All the
    /// machinery for it already existed — the animals have shifts — it was
    /// simply never wired to the people or the traffic.
    pub(crate) activity: f32,
}

/// Push one merged batch into the scene. Returns `None` for an empty builder.
/// Debug aid: hide named objects from the command line. Finding which object
/// draws a given artefact by commenting code out and rebuilding costs a minute
/// a guess; this costs one run. `SIMCITY_HIDE=?` lists the names.
pub(crate) fn hide_check(obj: &mut Object3D) {
    if let Ok(hide) = std::env::var("SIMCITY_HIDE") {
        if hide == "?" {
            eprintln!("batch {}", obj.name);
        }
        if hide.split(',').any(|h| h == obj.name) {
            obj.visible = false;
        }
    }
}

/// What the infrastructure pass managed to place.
///
/// Reported rather than assumed. The civic buildings taught this lesson the
/// expensive way: thresholds that read as reasonable placed *nothing at all*
/// on four layouts out of seven, and the only reason anybody found out is that
/// the generator counted them.
#[derive(Default, Clone)]
pub(crate) struct Works {
    pub(crate) plant: Option<(f32, f32)>,
    pub(crate) pylons: usize,
    pub(crate) port: bool,
    pub(crate) stadium: Option<(f32, f32)>,
    pub(crate) ballpark: bool,
    pub(crate) cricket: bool,
    pub(crate) mall: bool,
    pub(crate) grocery: usize,
    pub(crate) strips: usize,
    pub(crate) decks: usize,
    pub(crate) courts: usize,
    /// Cars standing in a bay rather than at a kerb.
    pub(crate) parked_cars: usize,
    /// Access roads built from an outlying site to the network.
    pub(crate) spurs: usize,
    /// Summits on the horizon.
    pub(crate) peaks: usize,
    /// Vehicles running on the motorway.
    pub(crate) motorway_vehicles: usize,
    /// Surveillance cameras fitted across the whole city.
    pub(crate) cameras: usize,
    pub(crate) depot: bool,
    pub(crate) rail_yard: bool,
    /// Bridges over the motorway: the junction, the green bridge, the footbridge.
    pub(crate) crossings: usize,
    /// Animals in the zoo. An empty zoo and a missing one look identical.
    pub(crate) zoo_animals: usize,
    /// Aircraft parked on the apron.
    pub(crate) aircraft: usize,
    /// People riding in the cycle lanes.
    pub(crate) cyclists: usize,
    /// Parking meters and pay-and-display machines on the footways.
    pub(crate) meters: usize,
    /// Angled kerbside bays, which only the broad avenues can take.
    pub(crate) echelon_bays: usize,
    /// Whether the city got a data centre.
    pub(crate) data_centre: bool,
    /// Telecommunications masts on the fringe.
    pub(crate) masts: usize,
    /// Which landmarks this city got.
    pub(crate) monuments: Vec<&'static str>,
    /// Whether the water is crossed by a suspension bridge.
    pub(crate) suspension: bool,
    /// The airfield's claimed ground as `[x0, z0, x1, z1]`, and the railway as
    /// `(runs along x, its fixed coordinate on the other axis)`.
    ///
    /// Recorded only so a test can check they do not overlap. The railway is
    /// laid on a street and runs the full width of the map, and for a long
    /// time that put the viaduct straight down the runway.
    /// The widest stretch of horizon, in radians, with no mountain on it.
    pub(crate) skyline_gap: f32,
    /// Road tunnels driven through the mountain ring. Each has two portals.
    pub(crate) tunnels: Vec<Tunnel>,
    pub(crate) airfield: Option<[f32; 4]>,
    pub(crate) rail: Option<(bool, f32)>,
}

/// Which street the railway is laid along: the fixed coordinate on the axis
/// the line does *not* run down.
///
/// Roads are the even entries and blocks the odd ones. When the line runs
/// parallel to the river it must not sit *along* the channel — that stands
/// every pier in the water and gives the barges something to hit — so step to
/// the next road until the chosen one is clear of it.
///
/// Pulled out of `generate_city` so it can be answered before the airport is
/// sited; see the airfield offset there.
fn rail_street(
    along_x: bool,
    ax: &Axis,
    az: &Axis,
    blocks: usize,
    water: Water,
    rx0: f32,
    rx1: f32,
) -> f32 {
    let axis = if along_x { az } else { ax };
    let mut k = ((blocks * 2 / 3).max(1) * 2).min(blocks * 2);
    let dry = |c: f32| water != Water::River || along_x || c < rx0 - 8.0 || c > rx1 + 8.0;
    for _ in 0..blocks {
        let (a, c) = axis.span(k);
        if dry((a + c) * 0.5) {
            break;
        }
        k = if k >= 4 { k - 2 } else { k + 2 };
    }
    let (a, c) = axis.span(k.min(blocks * 2));
    (a + c) * 0.5
}

pub(crate) fn add_batch(
    scene: &mut Scene,
    name: &str,
    mb: MeshBuilder,
    material: Material,
    cast: bool,
    receive: bool,
) -> Option<ObjectId> {
    if mb.is_empty() {
        return None;
    }
    let mut obj = Object3D::mesh(Mesh::new(mb.build(), material));
    obj.name = name.to_string();
    // Debug aid: hide named batches from the command line. Finding which batch
    // draws a given artefact by commenting code out and rebuilding costs a
    // minute a guess; this costs one run.
    hide_check(&mut obj);
    obj.cast_shadow = cast;
    obj.receive_shadow = receive;
    Some(scene.add(obj))
}

pub(crate) fn lit_standard(color: u32, roughness: f32, metalness: f32) -> Material {
    Material::Standard(
        StandardMaterial::new(Color::from_hex(color))
            .with_roughness(roughness)
            .with_metalness(metalness),
    )
}

pub(crate) fn generate_city(scene: &mut Scene, params: &CityParams) -> City {
    let mut rng = Rng::new(params.seed);
    let blocks = params.blocks.clamp(3, 12);
    let plan = params.layout.plan();

    // Where the water goes, in grid terms. A river takes one block column; a
    // bay swallows everything from one block column outward.
    let quarter = (blocks / 4).max(1);
    let water_block = match plan.water {
        // Off-centre on purpose: a river through the exact middle erases
        // downtown, which is the one part worth looking at.
        Water::River => {
            if rng.chance(0.5) {
                quarter
            } else {
                blocks - 1 - quarter
            }
        }
        Water::Bay => quarter.max(1),
        Water::Dry => usize::MAX,
    }
    .min(blocks.saturating_sub(1));

    let wide = match plan.water {
        Water::River => Some(water_block),
        _ => None,
    };
    let mut ax = Axis::build(blocks, &plan, &mut rng, wide);
    let az = Axis::build(blocks, &plan, &mut rng, None);
    let water_a = if plan.water == Water::Dry {
        usize::MAX
    } else {
        2 * water_block + 1
    };

    // A bay leaves the built area on one side of the origin, which would put
    // the city half outside the shadow camera's box. Slide the grid so the
    // land is centred again.
    if plan.water == Water::Bay {
        let shore = ax.span(water_a).1;
        ax.shift(-(shore - ax.min()) * 0.5);
    }

    let (cx0, cx1) = (ax.min(), ax.max());
    let (cz0, cz1) = (az.min(), az.max());
    // The seaward edge of the land, or the river channel.
    let (rx0, rx1) = match plan.water {
        Water::Dry => (f32::INFINITY, f32::INFINITY),
        _ => ax.span(water_a),
    };
    // The landward edge of the bay, or the near bank of the river.
    let shore = match plan.water {
        Water::Bay => rx1,
        _ => rx0,
    };
    let land_x0 = match plan.water {
        Water::Bay => shore,
        _ => cx0,
    };
    let extent = land_x0.abs().max(cx1).max(cz0.abs()).max(cz1);
    let land_span = cx1 - land_x0;
    let e = (extent * OUTSKIRTS_REACH).max(OUTSKIRTS_MIN);
    // How far the moving water runs. Deliberately *not* `e`: the land plate is
    // eight quads and may as well cover the far plane, but the water is a
    // lattice whose row count is capped, so stretching it to the world's edge
    // just makes each cell longer. At `e` a cell was a hundred metres, and the
    // one cell that straddled the camera clipped against the near plane into a
    // pale wedge that climbed out of the horizon and across the sky — the
    // "streak" that survived every other fix in this file.
    //
    // Half again past the point where the haze is complete: beyond that the
    // river is fog colour anyway, so its end cannot be seen.
    let wr = (extent * FOG_FAR * 1.5).max(600.0);
    // Only avenues get a bridge; bridging every side street would leave the
    // river reading as a chain of ponds rather than as a river.
    let is_bridge = |k: usize| is_avenue(&plan, k / 2);
    // True for any cell the water has taken.
    let drowned = |a: usize| match plan.water {
        Water::River => a == water_a,
        Water::Bay => a <= water_a,
        Water::Dry => false,
    };
    // Parkway keeps its middle green: a square of blocks rather than a
    // scatter of them, with the towers ringed outside.
    // Big enough to read as one park from the air rather than as a scatter of
    // gaps between towers.
    let park_r = if plan.core == Core::Ring {
        extent * 0.56
    } else {
        0.0
    };

    let mut b = Batches::default();
    // Parked cars draw from a stream of their own. Taking from the city's
    // `rng` would shift every later decision — block subdivision, building
    // heights, where the traffic starts — so adding decoration would silently
    // rearrange the city and any test tuned to a seed with it.
    let mut park_rng = Rng::new(params.seed ^ 0x9a3c_1f77_2b04_d519);
    // One body per type, built once and stamped along the kerbs.
    let parked_bodies: Vec<MeshBuilder> = (0..ROAD_FLEETS)
        .map(|i| {
            [
                Vehicle::Car,
                Vehicle::Hatch,
                Vehicle::Sports,
                Vehicle::Taxi,
                Vehicle::Pickup,
                Vehicle::Van,
                Vehicle::Bus,
                Vehicle::Lorry,
                Vehicle::Police,
                Vehicle::Estate,
                Vehicle::Suv,
                Vehicle::Ambulance,
                Vehicle::FireEngine,
                Vehicle::Delivery,
                // Nobody parks a bicycle at a kerb here, but the table is
                // indexed by fleet and has to stay the same length as one.
                Vehicle::Bicycle,
            ][i]
                .body_parked()
        })
        .collect();
    // One texture metre for every six world metres on the ground.
    b.road.uv_override = Some(Uv::World {
        u: 1.0 / 6.0,
        v: 1.0 / 6.0,
    });
    b.pads.uv_override = b.road.uv_override;
    let mut buildings = 0usize;
    let mut suburbs = 0usize;
    let mut cyclists = 0usize;
    let mut meters = 0usize;
    let mut echelon_bays = 0usize;
    // Collected while the streets are surfaced, which happens well before the
    // road fleets exist; merged into them below.
    //
    // On its own stream, not the city's. Drawing these from the shared `rng`
    // shifts every draw after them — and everything is after them, because the
    // streets are surfaced first. It moved the power station off its site and
    // lost the rail freight yard entirely, which is the hazard the parked cars
    // already carry a comment about and which I walked straight into anyway.
    let mut bike_rng = Rng::new(params.seed ^ 0x8b1c_9d33_47a2_e015);
    // Which roads carry a segregated cycle lane.
    //
    // Avenues only, and every other one, so the network is a grid of routes
    // rather than paint on everything. Deterministic on the road index so the
    // paint and the parked cars agree about it without sharing any state —
    // they are laid in different loops hundreds of lines apart.
    //
    // The lane goes *where the parking was*, and the parking is suppressed on
    // those roads. There is no third option on a fourteen-metre street: a
    // traffic lane, a parked car and a cycle lane do not fit across it, which
    // is the argument every city has about this and the reason they take the
    // parking out.
    //
    // Takes the half-width, because the lane is only *painted* where one fits
    // (`half >= 6.6`). Deciding this on the road index alone cleared the kerb
    // on alternate avenues of every layout — including the ones whose avenues
    // are eleven metres wide and never get a cycle lane at all, which is how
    // `oldtown` ended up with no kerbside parking anywhere in it.
    //
    // "Every other avenue" has to be counted in avenues, not in road indices.
    // Avenues fall every `avenue_every` roads, so on a layout where that
    // spacing is even *every* avenue has an even index — and the whole set
    // became cycle routes, which meant no avenue in the city had kerbside
    // parking on it and the echelon bays could never appear anywhere.
    let cycle_route = |idx: usize, half: f32| -> bool {
        let road = idx / 2;
        let every = plan.avenue_every.max(1);
        is_avenue(&plan, road) && (road / every).is_multiple_of(2) && half >= 6.6
    };
    let mut bike_movers: Vec<(usize, Mover)> = Vec::new();
    let mut landmark_left = if plan.peak_floors >= 14.0 { 1 } else { 0 };
    let mut lamps: Vec<Vector3> = Vec::new();
    // Palms belong on the waterfront and nowhere else: one among the conifers
    // of a temperate grid reads as a mistake.
    let warm = plan.water == Water::Bay;
    // Park cells, so the crowd generator can put people on their paths.
    let mut park_walks: Vec<(f32, f32, f32, f32)> = Vec::new();
    // Placed once each, in the order they matter: a city without a hospital is
    // more obviously missing something than one without a police station.
    let mut civic_wanted: Vec<Civic> = vec![
        Civic::Hospital,
        Civic::School,
        Civic::FireStation,
        Civic::PoliceStation,
    ];
    // Where the wildlife goes: open water for waterfowl, open hard ground for
    // anything that gathers to be fed.
    let mut ponds: Vec<(f32, f32, f32)> = Vec::new();
    let mut gathers: Vec<(f32, f32, f32)> = Vec::new();
    let mut runs: Vec<(f32, f32, f32)> = Vec::new();
    // Where the refuse sacks are, which is where the rats are.
    let mut trash: Vec<(f32, f32)> = Vec::new();

    // --- Terrain and water. The land always runs to the horizon, with a hole
    // --- cut for the city and, where there is one, for the channel.
    let terrain = Color::from_hex(C_TERRAIN);
    let water_col = Color::from_hex(C_WATER);
    // The open water is its own mesh rather than part of the merged batch: a
    // compute shader rewrites its vertices every frame, which needs a buffer
    // of its own.
    let mut water_surface: Option<(BufferGeometry, u32)> = None;
    match plan.water {
        Water::River => {
            b.grass.add_slab(-e, -e, cx0, e, 0.0, terrain, Uv::Unit);
            b.grass.add_slab(cx1, -e, e, e, 0.0, terrain, Uv::Unit);
            for (a0, a1) in [(cx0, rx0), (rx1, cx1)] {
                b.grass.add_slab(a0, -e, a1, cz0, 0.0, terrain, Uv::Unit);
                b.grass.add_slab(a0, cz1, a1, e, 0.0, terrain, Uv::Unit);
            }
            water_surface = Some(water_grid(rx0, -wr, rx1, wr, -RIVER_DEPTH, water_col));
        }
        Water::Bay => {
            b.grass.add_slab(cx1, -e, e, e, 0.0, terrain, Uv::Unit);
            b.grass.add_slab(shore, -e, cx1, cz0, 0.0, terrain, Uv::Unit);
            b.grass.add_slab(shore, cz1, cx1, e, 0.0, terrain, Uv::Unit);
            water_surface = Some(water_grid(-wr, -wr, shore.min(wr), wr, -RIVER_DEPTH, water_col));
        }
        Water::Dry => {
            b.grass.add_slab(-e, -e, cx0, e, 0.0, terrain, Uv::Unit);
            b.grass.add_slab(cx1, -e, e, e, 0.0, terrain, Uv::Unit);
            b.grass.add_slab(cx0, -e, cx1, cz0, 0.0, terrain, Uv::Unit);
            b.grass.add_slab(cx0, cz1, cx1, e, 0.0, terrain, Uv::Unit);
        }
    }

    // Embankment walls, set inside the water so they do not fight the roadway.
    let quay = Color::from_hex(0x7d7a72);
    let quay_spans: &[(f32, f32)] = match plan.water {
        Water::River => &[(rx0, rx0 + 1.1), (rx1 - 1.1, rx1)],
        Water::Bay => &[(shore - 1.3, shore)],
        Water::Dry => &[],
    };
    for (qx0, qx1) in quay_spans {
        b.pads.add_box(
            Vector3::new(*qx0, -RIVER_DEPTH - 0.9, cz0),
            Vector3::new(*qx1, KERB, cz1),
            quay,
            Uv::Unit,
        );
    }
    // Piers, so a bay is a working waterfront rather than a painted edge.
    if plan.water == Water::Bay {
        for i in 0..3 {
            let cz = mix(cz0 + 30.0, cz1 - 30.0, (i as f32 + 0.5) / 3.0);
            let len = rng.range(26.0, 48.0);
            b.pads.add_box(
                Vector3::new(shore - 1.3 - len, -0.55, cz - 5.0),
                Vector3::new(shore - 1.3, KERB, cz + 5.0),
                Color::from_hex(C_CONCRETE),
                Uv::Unit,
            );
            for pz in [cz - 5.0, cz + 5.0] {
                b.trim.add_box(
                    Vector3::new(shore - 1.3 - len, KERB, pz - 0.18),
                    Vector3::new(shore - 1.3, KERB + 0.9, pz + 0.18),
                    Color::from_hex(0x9b988f),
                    Uv::Unit,
                );
            }
        }
    }

    // --- Cells: roads, bridges, blocks.
    for a in 0..ax.len() {
        for k in 0..az.len() {
            let (x0, x1) = ax.span(a);
            let (z0, z1) = az.span(k);

            if drowned(a) {
                // Under water, except where an avenue bridges a river. A bay
                // is never crossed at all.
                if plan.water != Water::River || !Axis::is_road(k) || !is_bridge(k) {
                    continue;
                }
                // Bridge: a deck slung under the roadway, plus railings.
                b.pads.add_box(
                    Vector3::new(x0 - 1.2, -0.75, z0),
                    Vector3::new(x1 + 1.2, 0.0, z1),
                    Color::from_hex(C_CONCRETE),
                    Uv::Unit,
                );
                b.road.add_slab(
                    x0 - 1.2,
                    z0,
                    x1 + 1.2,
                    z1,
                    0.0,
                    Color::from_hex(C_ASPHALT),
                    Uv::Unit,
                );
                for zz in [z0, z1] {
                    b.trim.add_box(
                        Vector3::new(x0 - 1.2, 0.0, zz - 0.16),
                        Vector3::new(x1 + 1.2, 1.15, zz + 0.16),
                        Color::from_hex(0x9b988f),
                        Uv::Unit,
                    );
                }
                continue;
            }

            if Axis::is_road(a) || Axis::is_road(k) {
                b.road.add_slab(
                    x0,
                    z0,
                    x1,
                    z1,
                    0.0,
                    scale_color(Color::from_hex(C_ASPHALT), rng.range(0.92, 1.08)),
                    Uv::Unit,
                );
                // --- Cycle lanes.
                //
                // On the running length of a street, not in the junction
                // cells: a cycle lane painted across a junction is a cycle
                // lane painted over the give-way markings, which is both wrong
                // and unreadable. Wide streets only — a nine-metre street with
                // parking on both sides has nowhere to put one, which is
                // exactly the argument every city has about them.
                let along_z = Axis::is_road(a) && !Axis::is_road(k);
                let along_x = Axis::is_road(k) && !Axis::is_road(a);
                let road_idx = if along_z { a } else { k };
                let half_here = if along_z { (x1 - x0) * 0.5 } else { (z1 - z0) * 0.5 };
                if (along_z || along_x) && cycle_route(road_idx, half_here) {
                    let (w0, w1) = if along_z { (x0, x1) } else { (z0, z1) };
                    let mid = (w0 + w1) * 0.5;
                    let half = (w1 - w0) * 0.5;
                    let lane_w = 1.8f32;
                    // Where the kerbside parking would have been.
                    let lane_c = half - 1.6;
                    // Still has to clear the running lane, which sits at
                    // 0.44 of the half-width with a car's width around it.
                    if half >= 6.6 && lane_c - lane_w * 0.5 > half * 0.44 + 1.05 {
                        let track =
                            scale_color(Color::from_hex(0x2f6b4f), rng.range(0.92, 1.08));
                        for side in [-1.0f32, 1.0] {
                            let c = mid + side * lane_c;
                            let (a0, a1) = (c - lane_w * 0.5, c + lane_w * 0.5);
                            let ln = c - side * (lane_w * 0.5 + 0.06);
                            if along_z {
                                b.paint.add_slab(a0, z0, a1, z1, 0.012, track, Uv::Unit);
                                b.paint.add_slab(
                                    ln - 0.06,
                                    z0,
                                    ln + 0.06,
                                    z1,
                                    0.014,
                                    Color::from_hex(0xd8d4c4),
                                    Uv::Unit,
                                );
                            } else {
                                b.paint.add_slab(x0, a0, x1, a1, 0.012, track, Uv::Unit);
                                b.paint.add_slab(
                                    x0,
                                    ln - 0.06,
                                    x1,
                                    ln + 0.06,
                                    0.014,
                                    Color::from_hex(0xd8d4c4),
                                    Uv::Unit,
                                );
                            }
                            // Somebody in it. A painted lane with nothing in it
                            // is paint — the same argument that put lorries on
                            // the motorway.
                            let dir = if along_z { -side } else { side };
                            let (s_lo, s_hi) = if along_z { (z0, z1) } else { (x0, x1) };
                            let riders = ((s_hi - s_lo) / 20.0).ceil() as usize;
                            for _ in 0..riders {
                                if !bike_rng.chance(0.65) {
                                    continue;
                                }
                                let cruise = bike_rng.range(
                                    Vehicle::Bicycle.speed().0,
                                    Vehicle::Bicycle.speed().1,
                                );
                                bike_movers.push((
                                    bike_rng.below(Vehicle::Bicycle.colors().len()),
                                    Mover {
                                        along_x: !along_z,
                                        fixed: c,
                                        s: bike_rng.range(s_lo, s_hi),
                                        s_lo,
                                        s_hi,
                                        speed: cruise,
                                        cruise,
                                        length: Vehicle::Bicycle.length(),
                                        lane: 0x2000
                                            | ((a as u16) << 6)
                                            | ((k as u16) << 1)
                                            | u16::from(side > 0.0),
                                        signalled: false,
                                        dir,
                                        base_y: 0.03,
                                        scale: VEHICLE_SCALE * bike_rng.range(0.94, 1.06),
                                        seed: bike_rng.next_u32(),
                                        turn_s0: f32::NAN,
                                        turn: [0.0; 4],
                                        tone: 0,
                                    },
                                ));
                                cyclists += 1;
                            }
                        }
                    }
                }

                // Standing water. The cheapest reflection in a city: a low
                // roughness patch picks up the sky by day and the lamps at
                // night, through the same environment map the glass uses.
                for _ in 0..rng.below(3) {
                    let edge = if rng.chance(0.5) { x0 + 0.9 } else { x1 - 0.9 };
                    add_ground_blob(
                        &mut b.puddle,
                        edge + rng.range(-0.5, 0.5),
                        rng.range(z0, z1),
                        0.008,
                        rng.range(0.5, 1.5),
                        rng.range(0.4, 1.2),
                        &mut rng,
                        Color::from_hex(0x1b1d21),
                    );
                }
                // Oil and grime down the middle of the lane.
                for _ in 0..rng.below(3) {
                    add_ground_blob(
                        &mut b.paint,
                        rng.range(x0 + 1.0, x1 - 1.0),
                        rng.range(z0, z1),
                        0.010,
                        rng.range(0.4, 1.1),
                        rng.range(0.4, 1.1),
                        &mut rng,
                        Color::from_hex(0x2a2b2c),
                    );
                }
                // What blows into the gutter and stays there.
                for _ in 0..rng.below(5) {
                    let edge = if rng.chance(0.5) { x0 + 0.5 } else { x1 - 0.5 };
                    add_ground_quad(
                        &mut b.paint,
                        edge + rng.range(-0.3, 0.3),
                        rng.range(z0, z1),
                        0.012,
                        rng.range(0.10, 0.28),
                        rng.range(0.08, 0.22),
                        rng.range(0.0, TAU),
                        scale_color(Color::from_hex(0x9c968a), rng.range(0.5, 1.1)),
                    );
                }
                continue;
            }

            // --- A city block.
            let cell = Rect { x0, z0, x1, z1 };
            // One of each civic building per city, taken from the middle ring:
            // not downtown, where the land is towers, and not the outskirts,
            // where nothing would ever pass one.
            if let Some(kind) = civic_wanted.first().copied() {
                let d0 = core_distance(&plan, cell.cx(), cell.cz(), extent, shore, land_span, park_r);
                if (0.20..0.94).contains(&d0) && add_civic(&mut b, kind, cell, &mut rng) {
                    civic_wanted.remove(0);
                    buildings += 1;
                    continue;
                }
            }
            let d = core_distance(&plan, cell.cx(), cell.cz(), extent, shore, land_span, park_r);
            let n = city_noise(cell.cx() * 0.012 + 21.5, cell.cz() * 0.012 + 7.25);
            let zone = (d * 1.02 + (n - 0.5) * 0.50).clamp(0.0, 1.25);

            // Designated green comes from the layout's park pattern and is
            // graded from the middle of the green area outwards; the rest is
            // the old per-block roll.
            let central = park_role(&plan, cell.cx(), cell.cz(), extent, park_r);
            let park = central.is_some()
                || (d > 0.14 && rng.chance(plan.park_base + plan.park_slope * zone));
            if park {
                let kind = choose_park(&cell, zone, central, &mut rng);
                let out = add_park(&mut b, cell, kind, warm, true, &mut rng);
                park_walks.extend(out.walks);
                ponds.extend(out.ponds);
                gathers.extend(out.gathers);
                runs.extend(out.runs);
                continue;
            }

            b.pads.add_box(
                Vector3::new(x0, 0.0, z0),
                Vector3::new(x1, KERB, z1),
                scale_color(Color::from_hex(C_SIDEWALK), rng.range(0.9, 1.1)),
                Uv::Unit,
            );

            // Out at the edge, some blocks stop being blocks. A close or a
            // crescent replaces the grid subdivision entirely: its own road
            // goes in first and the houses front that instead of the four
            // streets around the outside. Only where the land is cheap enough
            // for houses and the block is big enough to get a turning head
            // into — which is exactly where suburbs are.
            if zone > 0.66 && rng.chance(0.60) {
                // Try the other form if the first will not fit. A close needs
                // a turning head and a crescent needs a bow; a block that
                // refuses one often takes the other, and refusing both when
                // only one was tried is how a layout ends up with a single
                // suburban street.
                let close_first = rng.chance(0.62);
                let mut made = if close_first {
                    add_close(&mut b, &plan, cell, &mut rng)
                } else {
                    add_crescent(&mut b, &plan, cell, &mut rng)
                };
                if made == 0 {
                    made = if close_first {
                        add_crescent(&mut b, &plan, cell, &mut rng)
                    } else {
                        add_close(&mut b, &plan, cell, &mut rng)
                    };
                }
                if made > 0 {
                    buildings += made;
                    suburbs += 1;
                    continue;
                }
            }

            // Parcels. Downtown lots are large (one tower fills a lot); the
            // outskirts subdivide down to house plots.
            let buildable = cell.inset((plan.street_w * 0.34).clamp(1.9, 4.0));
            if buildable.w() > 6.0 && buildable.d() > 6.0 {
                // Downtown parcels take the whole block (one tower fills
                // it); the outskirts subdivide down to house plots.
                let target = mix(plan.lot_core, plan.lot_edge, smoothstep(0.10, 0.78, zone));
                let mut lots = Vec::new();
                subdivide(buildable, target, 4, &mut rng, &mut lots);
                for lot in lots {
                    let landmark = landmark_left > 0 && zone < 0.16 && lot.area() > 260.0;
                    if landmark {
                        landmark_left -= 1;
                    }
                    if (!landmark && rng.chance(0.015 + 0.06 * zone))
                        || !add_building(&mut b, &plan, lot, zone, landmark, &mut rng)
                    {
                        // Nothing fits, or the dice said car park. Downtown
                        // that means surface parking; further out it is a
                        // garden, which is what the gaps between houses are.
                        // The lots nothing was built on are where the bins,
                        // pallets and skips end up.
                        if rng.chance(0.30) && lot.w() > 6.0 && lot.d() > 6.0 {
                            add_service_yard(&mut b, lot.inset(0.9), &mut rng);
                            continue;
                        }
                        if rng.chance(0.30 + 0.45 * (1.0 - zone).clamp(0.0, 1.0)) {
                            let g = lot.inset(0.5);
                            if g.w() > 1.5 && g.d() > 1.5 {
                                b.grass.add_slab(
                                    g.x0,
                                    g.z0,
                                    g.x1,
                                    g.z1,
                                    KERB + 0.01,
                                    scale_color(
                                        Color::from_hex(C_GRASS),
                                        rng.range(0.8, 1.15),
                                    ),
                                    Uv::Unit,
                                );
                                if g.w() > 4.0 && g.d() > 4.0 && rng.chance(0.55) {
                                    add_tree(&mut b, g.cx(), g.cz(), KERB, true, warm, &mut rng);
                                }
                            }
                        } else {
                            add_parking(&mut b, lot, &mut rng);
                        }
                    } else {
                        buildings += 1;
                    }
                }
            }

            // Street furniture in the pavement strip. Without this layer a
            // pavement is a grey ribbon, and no amount of work on the towers
            // fixes that.
            let strip = (plan.street_w * 0.34).clamp(1.9, 4.0);
            for _ in 0..2 + rng.below(4) {
                let inset = rng.range(0.7, strip - 0.4);
                let (fx, fz, toward) = match rng.below(4) {
                    0 => (
                        rng.range(x0 + 3.0, x1 - 3.0),
                        z0 + inset,
                        Vector3::new(0.0, 0.0, -1.0),
                    ),
                    1 => (
                        rng.range(x0 + 3.0, x1 - 3.0),
                        z1 - inset,
                        Vector3::new(0.0, 0.0, 1.0),
                    ),
                    2 => (
                        x0 + inset,
                        rng.range(z0 + 3.0, z1 - 3.0),
                        Vector3::new(-1.0, 0.0, 0.0),
                    ),
                    _ => (
                        x1 - inset,
                        rng.range(z0 + 3.0, z1 - 3.0),
                        Vector3::new(1.0, 0.0, 0.0),
                    ),
                };
                add_furniture(&mut b, fx, fz, toward, &mut rng);
                // Somewhere to get on whatever runs under the street. Rare,
                // and only on the wider pavements — an entrance takes three
                // metres of footway and would block a side street.
                if rng.chance(0.03) {
                    add_metro_entrance(
                        &mut b,
                        fx,
                        fz,
                        facing(-toward.x, -toward.z),
                        &mut rng,
                    );
                }
                // Refuse waiting for collection, against the wall rather than
                // out on the flags. Remembered so the rats know where to go.
                if rng.chance(0.22) {
                    let tx = fx - toward.x * 0.6;
                    let tz = fz - toward.z * 0.6;
                    if add_trash(&mut b, tx, tz, &mut rng) {
                        trash.push((tx, tz));
                    }
                }
            }

            // Street trees in the setback strip, thinned out at random so the
            // rows do not look planted by a for-loop.
            let inset = 1.6;
            let step = 11.0;
            let mut t = z0 + 5.0;
            while t < z1 - 5.0 {
                for tx in [x0 + inset, x1 - inset] {
                    if rng.chance(0.7) {
                        add_tree(&mut b, tx, t, KERB, true, warm, &mut rng);
                    }
                }
                t += step;
            }
            let mut t = x0 + 5.0;
            while t < x1 - 5.0 {
                for tz in [z0 + inset, z1 - inset] {
                    if rng.chance(0.7) {
                        add_tree(&mut b, t, tz, KERB, true, warm, &mut rng);
                    }
                }
                t += step;
            }
        }
    }

    // --- Markings and street lighting, one pass per road segment.
    for a in 0..ax.len() {
        for k in 0..az.len() {
            let road_z = Axis::is_road(a) && !Axis::is_road(k); // runs along Z
            let road_x = !Axis::is_road(a) && Axis::is_road(k); // runs along X
            if !(road_z || road_x) {
                continue;
            }
            // A road buried under the water, or one that stops at the bank.
            if drowned(a) && (plan.water != Water::River || !is_bridge(k)) {
                continue;
            }
            let (center, half, s0, s1, avenue) = if road_z {
                (
                    ax.center(a),
                    ax.half(a),
                    az.span(k).0,
                    az.span(k).1,
                    is_avenue(&plan, a / 2),
                )
            } else {
                (
                    az.center(k),
                    az.half(k),
                    ax.span(a).0,
                    ax.span(a).1,
                    is_avenue(&plan, k / 2),
                )
            };
            paint_road(&mut b, road_x, center, s0, s1, half, avenue);

            // Cars at the kerb. Static, so they merge into a batch: a parked
            // car has no simulation to do and an empty kerb is the single most
            // conspicuous thing missing from a street.
            //
            // Only where there is room. A side street's carriageway is about
            // nine metres; a traffic lane sits 0.44 of the half-width out from
            // the centre and a parked car is 1.8 wide, and the two do not both
            // fit — the first pass had cars parked *in* the running lane. An
            // avenue has room; a lane does not, and in a real city it would be
            // single yellow lines rather than cars.
            // No kerbside parking where the cycle lane took the kerb.
            let parks = !cycle_route(if road_z { a } else { k }, half);
            // Wide roads get *echelon* bays — angled, nose-in, marked with
            // diagonals — rather than a parallel strip. It is the arrangement
            // you actually see on a broad avenue or beside a park, it fits half
            // as many cars again into the same kerb, and unlike parallel
            // parking it is unmistakable from any distance: a row of cars all
            // canted the same way off the kerb reads as a parking lane at a
            // glance where a line of parallel ones reads as traffic.
            //
            // The gate is wide on purpose. A car at forty-five degrees needs
            // nearly five metres of depth, and the running lane sits about
            // 0.44 of the half-width out with a car's width around it — so a
            // fourteen-metre avenue cannot take echelon bays without putting
            // them in the traffic, and only the genuinely broad boulevards can.
            let echelon = parks && half >= 9.5;
            // Double yellows where parking is not allowed — the approaches to
            // every junction. This is the other half of making a kerb legible:
            // marked bays say where you may leave a car, and the yellows say
            // where you may not, and a street with neither reads as a car park
            // with a road through it.
            if half >= 4.6 {
                let cross = if road_z { &az } else { &ax };
                for ci in (0..cross.len()).step_by(2) {
                    let c = cross.center(ci);
                    let reach = cross.half(ci) + 9.0;
                    let (y0, y1) = (c - reach, c + reach);
                    if y1 < s0 || y0 > s1 {
                        continue;
                    }
                    let (y0, y1) = (y0.max(s0), y1.min(s1));
                    for side in [-1.0f32, 1.0] {
                        for off in [half - 0.55, half - 0.95] {
                            let e = center + side * off;
                            let (q0, q1) = if road_z {
                                ((e - 0.07, y0), (e + 0.07, y1))
                            } else {
                                ((y0, e - 0.07), (y1, e + 0.07))
                            };
                            b.paint.add_slab(
                                q0.0,
                                q0.1,
                                q1.0,
                                q1.1,
                                0.013,
                                Color::from_hex(0xd8b13a),
                                Uv::Unit,
                            );
                        }
                    }
                }
            }
            // A machine every few bays, on the footway rather than the kerb.
            if parks && half >= 4.6 {
                let mut m = s0 + 8.0;
                while m < s1 - 8.0 {
                    let cross = if road_z { &az } else { &ax };
                    let near_junction = (0..cross.len())
                        .step_by(2)
                        // Clear of the junction, but only just: at ten metres
                        // the exclusion zones from adjacent junctions
                        // overlapped on any normal block and the whole city
                        // got three machines.
                        .any(|ci| (m - cross.center(ci)).abs() < cross.half(ci) + 3.5);
                    if !near_junction {
                        for side in [-1.0f32, 1.0] {
                            if !park_rng.chance(0.8) {
                                continue;
                            }
                            let f = center + side * (half + 1.4);
                            let (mx, mz) = if road_z { (f, m) } else { (m, f) };
                            if !b.occ.try_spot(mx, mz, 0.5) {
                                continue;
                            }
                            let toward = if road_z {
                                Vector3::new(-side, 0.0, 0.0)
                            } else {
                                Vector3::new(0.0, 0.0, -side)
                            };
                            add_meter(
                                &mut b,
                                mx,
                                mz,
                                toward,
                                if park_rng.chance(0.72) {
                                    Meter::PayDisplay
                                } else {
                                    Meter::Single
                                },
                                &mut park_rng,
                            );
                            meters += 1;
                        }
                    }
                    m += park_rng.range(15.0, 24.0);
                }
            }
            let mut p = s0 + 9.0;
            while parks && half >= 4.6 && p < s1 - 9.0 {
                // Never across a junction or its crossings.
                let cross = if road_z { &az } else { &ax };
                // Clear of the junction and its crossings — but only just. At
                // seven metres the exclusion zones from the junctions at each
                // end of a block met in the middle, so most kerbs had no
                // parking on them at all and the bay markings were drawn in
                // about thirty places in the entire city.
                let blocked = (0..cross.len()).step_by(2).any(|k| {
                    (p - cross.center(k)).abs() < cross.half(k) + 3.2
                });
                if blocked {
                    p += 4.0;
                    continue;
                }
                if echelon {
                    // A 45-degree bay: the car sits diagonally, so the lane is
                    // deeper than a parallel one and the markings run at an
                    // angle from the kerb.
                    for side in [-1.0f32, 1.0] {
                        let depth = 4.9f32;
                        let outer = half - 0.10;
                        let inner = outer - depth;
                        // The surface of the lane.
                        let (o0, o1) = (
                            center + side * inner.min(outer),
                            center + side * inner.max(outer),
                        );
                        let (q0, q1) = if road_z {
                            ((o0.min(o1), p - 1.6), (o0.max(o1), p + 1.6))
                        } else {
                            ((p - 1.6, o0.min(o1)), (p + 1.6, o0.max(o1)))
                        };
                        b.paint.add_slab(
                            q0.0,
                            q0.1,
                            q1.0,
                            q1.1,
                            0.011,
                            scale_color(Color::from_hex(0x4e4e54), rng.range(0.95, 1.05)),
                            Uv::Unit,
                        );
                        // The diagonal bay line, from the kerb inward at 45
                        // degrees. One rotated quad — drawing it as a run of
                        // axis-aligned slabs makes a staircase, which is the
                        // same mistake as the wall over the triumphal arch and
                        // the motorway slip roads.
                        let a_out = center + side * outer;
                        let a_in = center + side * inner;
                        let (s_out, s_in) = (p - 1.6, p - 1.6 - depth);
                        let n = 0.09f32;
                        let corners: [(f32, f32); 4] = if road_z {
                            [
                                (a_out, s_out - n),
                                (a_in, s_in - n),
                                (a_in, s_in + n),
                                (a_out, s_out + n),
                            ]
                        } else {
                            [
                                (s_out - n, a_out),
                                (s_in - n, a_in),
                                (s_in + n, a_in),
                                (s_out + n, a_out),
                            ]
                        };
                        b.paint.add_ground_quad(
                            corners,
                            0.014,
                            Color::from_hex(0xd8d4c4),
                            Uv::Unit,
                        );
                    }
                    // A car in most of them, canted off the kerb.
                    for side in [-1.0f32, 1.0] {
                        if !park_rng.chance(0.66) {
                            continue;
                        }
                        let kind = match park_rng.below(10) {
                            0 => Vehicle::Van,
                            1 => Vehicle::Pickup,
                            2 => Vehicle::Suv,
                            3..=5 => Vehicle::Hatch,
                            _ => Vehicle::Car,
                        };
                        let paint = Color::from_hex(
                            kind.colors()[park_rng.below(kind.colors().len())],
                        );
                        let lane = center + side * (half - 2.7);
                        let (px, pz) = if road_z { (lane, p - 1.2) } else { (p - 1.2, lane) };
                        // Nose to the kerb, forty-five degrees off it.
                        let cant = PI * 0.25 * side * if road_z { 1.0 } else { -1.0 };
                        let yaw = if road_z { cant } else { PI * 0.5 + cant };
                        if !b.occ.try_claim([px - 2.6, pz - 2.6, px + 2.6, pz + 2.6]) {
                            continue;
                        }
                        b.parked.append_at(
                            &parked_bodies[kind as usize],
                            Vector3::new(px, 0.0, pz),
                            yaw,
                            VEHICLE_SCALE,
                            paint,
                        );
                    }
                    echelon_bays += 1;
                    p += park_rng.range(3.2, 3.8);
                    continue;
                }
                // The parking lane itself: a strip of surface along the kerb,
                // the width of a car, that the bays are marked out on. Bay
                // ticks alone read as stray lines on a wide road — what makes
                // a kerb legible as *parking* is that the strip is a different
                // surface from the running lanes, which is how it is built.
                for side in [-1.0f32, 1.0] {
                    let outer = center + side * (half - 0.10);
                    let inner = center + side * (half - 2.40);
                    let (o0, o1) = (outer.min(inner), outer.max(inner));
                    let (q0, q1) = if road_z {
                        ((o0, p - 3.0), (o1, p + 3.0))
                    } else {
                        ((p - 3.0, o0), (p + 3.0, o1))
                    };
                    b.paint.add_slab(
                        q0.0,
                        q0.1,
                        q1.0,
                        q1.1,
                        0.011,
                        scale_color(Color::from_hex(0x4e4e54), rng.range(0.95, 1.05)),
                        Uv::Unit,
                    );
                }
                // Bay markings. A car standing on bare asphalt is a car that
                // has stopped; a car standing in a marked bay is a car that is
                // parked, and the difference is one white line.
                for side in [-1.0f32, 1.0] {
                    let edge = center + side * (half - 0.12);
                    let inner = center + side * (half - 2.35);
                    for end in [p - 2.9, p + 2.9] {
                        let (l0, l1) = if road_z {
                            ((edge.min(inner), end - 0.06), (edge.max(inner), end + 0.06))
                        } else {
                            ((end - 0.06, edge.min(inner)), (end + 0.06, edge.max(inner)))
                        };
                        b.paint.add_slab(
                            l0.0,
                            l0.1,
                            l1.0,
                            l1.1,
                            0.013,
                            Color::from_hex(0xd8d4c4),
                            Uv::Unit,
                        );
                    }
                }
                for side in [-1.0f32, 1.0] {
                    if !park_rng.chance(0.62) {
                        continue;
                    }
                    let kind = match park_rng.below(10) {
                        0 => Vehicle::Van,
                        1 => Vehicle::Pickup,
                        2 => Vehicle::Taxi,
                        3..=5 => Vehicle::Hatch,
                        _ => Vehicle::Car,
                    };
                    let paint = Color::from_hex(
                        kind.colors()[park_rng.below(kind.colors().len())],
                    );
                    // Nose in or nose out, and never quite square to the kerb.
                    let flip = if park_rng.chance(0.5) { 0.0 } else { PI };
                    let skew = park_rng.range(-0.05, 0.05);
                    // Hard against the kerb, outboard of the running lane.
                    let lane = center + side * (half - 1.05);
                    let (px, pz, yaw) = if road_z {
                        (lane, p, flip + skew)
                    } else {
                        (p, lane, PI * 0.5 + flip + skew)
                    };
                    // Six metres of kerb, and nothing else may be in it.
                    let (hw, hd) = if road_z {
                        (1.2 * VEHICLE_SCALE, 2.9 * VEHICLE_SCALE)
                    } else {
                        (2.9 * VEHICLE_SCALE, 1.2 * VEHICLE_SCALE)
                    };
                    if !b.occ.try_claim([px - hw, pz - hd, px + hw, pz + hd]) {
                        continue;
                    }
                    b.parked.append_at(
                        &parked_bodies[kind as usize],
                        Vector3::new(px, KERB * 0.0, pz),
                        yaw,
                        VEHICLE_SCALE,
                        paint,
                    );
                }
                p += park_rng.range(6.5, 11.0);
            }

            let mut s = s0 + 6.0;
            while s < s1 - 4.0 {
                for side in [-1.0f32, 1.0] {
                    let off = center + side * (half + 0.9);
                    let toward = if road_z {
                        Vector3::new(-side, 0.0, 0.0)
                    } else {
                        Vector3::new(0.0, 0.0, -side)
                    };
                    // Sodium or LED, chosen per road rather than per lamp:
                    // a street is relit all at once, and a mixture down one
                    // carriageway reads as a bug rather than as a city part
                    // way through a replacement programme.
                    let warm = city_hash2(road_x as i32 * 7 + 1, road_z as i32 * 13) < 0.55;
                    if road_z {
                        add_lamp(&mut b, &mut lamps, off, s, toward, warm);
                    } else {
                        add_lamp(&mut b, &mut lamps, s, off, toward, warm);
                    }
                }
                s += 23.0;
            }
        }
    }

    // --- Woodland past the city limits. A flat green sheet to the horizon
    // reads as a missing texture; scattered trees read as countryside.
    let mut planted = 0;
    for _ in 0..2000 {
        if planted >= 170 {
            break;
        }
        let x = rng.range(-e * 0.55, e * 0.55);
        let z = rng.range(-e * 0.55, e * 0.55);
        let on_city = x > cx0 - 14.0 && x < cx1 + 14.0 && z > cz0 - 14.0 && z < cz1 + 14.0;
        let in_water = match plan.water {
            Water::River => x > rx0 - 8.0 && x < rx1 + 8.0,
            Water::Bay => x < shore + 8.0,
            Water::Dry => false,
        };
        if on_city || in_water {
            continue;
        }
        add_tree(&mut b, x, z, 0.0, false, warm, &mut rng);
        planted += 1;
    }

    // --- Signals, where two avenues meet.
    for a in (0..ax.len()).step_by(2) {
        if !is_avenue(&plan, a / 2) || drowned(a) {
            continue;
        }
        for k in (0..az.len()).step_by(2) {
            if !is_avenue(&plan, k / 2) {
                continue;
            }
            let (cx, hx) = (ax.center(a), ax.half(a));
            let (cz, hz) = (az.center(k), az.half(k));
            let (ox, oz) = (hx + 1.3, hz + 1.3);
            // Arms along X hold the east-west phase, arms along Z the
            // north-south one, so the two never show green together.
            add_signal(&mut b, cx - ox, cz - oz, Vector3::new(1.0, 0.0, 0.0), 1);
            add_signal(&mut b, cx + ox, cz + oz, Vector3::new(-1.0, 0.0, 0.0), 1);
            add_signal(&mut b, cx + ox, cz - oz, Vector3::new(0.0, 0.0, 1.0), 0);
            add_signal(&mut b, cx - ox, cz + oz, Vector3::new(0.0, 0.0, -1.0), 0);
        }
    }

    // --- The stretch of an east-west road a vehicle or a pedestrian may use:
    // the whole width where the water is bridged or absent, one bank where a
    // street stops at a river, and everything landward of a bay's quay.
    let x_run = |k: usize, rng: &mut Rng| -> (f32, f32) {
        match plan.water {
            Water::Dry => (cx0, cx1),
            Water::Bay => (shore + 3.0, cx1),
            Water::River if is_bridge(k) => (cx0, cx1),
            Water::River => {
                if rng.chance(0.5) {
                    (cx0, rx0)
                } else {
                    (rx1, cx1)
                }
            }
        }
    };

    // --- Road traffic. Mostly cars, with enough larger vehicles that the
    // streets still read from the air, where a car is two pixels across.
    let lines = 2 * (blocks + 1);
    let per_line = (params.cars / lines.max(1)).max(1);
    let mut road: [Vec<(usize, Mover)>; ROAD_FLEETS] = Default::default();
    road[Vehicle::Bicycle as usize] = std::mem::take(&mut bike_movers);
    let push_vehicle = |road: &mut [Vec<(usize, Mover)>; ROAD_FLEETS],
                        rng: &mut Rng,
                        along_x: bool,
                        road_index: usize,
                        signalled: bool,
                        center: f32,
                        half: f32,
                        s_lo: f32,
                        s_hi: f32| {
        // Weighted like real traffic: mostly ordinary cars and hatchbacks,
        // a fifth commercial, and the rarities rare. One police car per
        // couple of hundred vehicles is enough to be worth noticing.
        let roll = rng.f();
        let kind = if roll < 0.004 {
            Vehicle::Police
        } else if roll < 0.008 {
            Vehicle::Ambulance
        } else if roll < 0.011 {
            Vehicle::FireEngine
        } else if roll < 0.025 {
            Vehicle::Sports
        } else if roll < 0.075 {
            Vehicle::Bus
        } else if roll < 0.135 {
            Vehicle::Lorry
        } else if roll < 0.175 {
            Vehicle::Delivery
        } else if roll < 0.235 {
            Vehicle::Van
        } else if roll < 0.285 {
            Vehicle::Pickup
        } else if roll < 0.355 {
            Vehicle::Taxi
        } else if roll < 0.44 {
            Vehicle::Suv
        } else if roll < 0.51 {
            Vehicle::Estate
        } else if roll < 0.72 {
            Vehicle::Hatch
        } else {
            Vehicle::Car
        };
        let dir = if rng.chance(0.5) { 1.0 } else { -1.0 };
        let cruise = rng.range(kind.speed().0, kind.speed().1);
        // The bounds hold the CENTRE, so a long vehicle at the end of a run
        // hangs half its body past it — which on a street that dead-ends at
        // the river is a bus over the water.
        let (s_lo, s_hi) = fit_run(s_lo, s_hi, kind.length());
        // One lane per (axis, road, direction). Both directions of a road are
        // separate streams, and a stream may span all four vehicle fleets.
        let lane = ((road_index as u16) << 2)
            | ((along_x as u16) << 1)
            | u16::from(dir > 0.0);
        road[kind as usize].push((
            rng.below(kind.colors().len()),
            Mover {
                along_x,
                // The running lane sits 0.44 of the half-width out — except on
                // a street narrow enough that a parked car would then be in
                // it, where it pulls inboard instead. That is what a
                // residential street actually does: the carriageway narrows to
                // whatever is left between the parked cars, and it is the
                // reason kerbside parking is not confined to avenues.
                fixed: center - dir * (half * 0.44).min(half - 3.3).max(half * 0.25),
                s: rng.range(s_lo, s_hi),
                s_lo,
                s_hi,
                speed: cruise,
                cruise,
                length: kind.length(),
                lane,
                signalled,
                dir,
                base_y: 0.03,
                scale: VEHICLE_SCALE * rng.range(0.95, 1.05),
                seed: rng.next_u32(),
                turn_s0: f32::NAN,
                turn: [0.0; 4],
                tone: 0,
            },
        ));
    };
    for a in (0..ax.len()).step_by(2) {
        // Roads running along Z never meet a river, which runs along Z too.
        let (c, h) = (ax.center(a), ax.half(a));
        if drowned(a) {
            continue;
        }
        for _ in 0..per_line {
            push_vehicle(
                &mut road,
                &mut rng,
                false,
                a,
                is_avenue(&plan, a / 2),
                c,
                h,
                cz0,
                cz1,
            );
        }
    }
    for k in (0..az.len()).step_by(2) {
        let (c, h) = (az.center(k), az.half(k));
        for _ in 0..per_line {
            let (s_lo, s_hi) = x_run(k, &mut rng);
            push_vehicle(
                &mut road,
                &mut rng,
                true,
                k,
                is_avenue(&plan, k / 2),
                c,
                h,
                s_lo,
                s_hi,
            );
        }
    }
    for v in road.iter_mut() {
        v.sort_by_key(|(c, _)| *c);
    }

    // --- People, on the pavement rather than in the lane. Denser than the
    // traffic: a city street has far more of them than it has cars.
    let mut crowd: Vec<(usize, Mover)> = Vec::new();
    let per_pavement = (params.cars / lines.max(1)).max(2);
    let walk = |crowd: &mut Vec<(usize, Mover)>,
                rng: &mut Rng,
                along_x: bool,
                road_index: usize,
                side: f32,
                fixed: f32,
                s_lo: f32,
                s_hi: f32| {
        let cruise = rng.range(1.0, 1.8);
        let dir = if rng.chance(0.5) { 1.0 } else { -1.0 };
        // One lane a pavement a direction: people queue behind the person in
        // front of them, not behind everyone on the street.
        let lane = ((road_index as u16) << 3)
            | ((along_x as u16) << 2)
            | (u16::from(side > 0.0) << 1)
            | u16::from(dir > 0.0);
        crowd.push((
            rng.below(WALKER_COLORS.len()),
            Mover {
                along_x,
                fixed: fixed + rng.range(-0.45, 0.45),
                s: rng.range(s_lo, s_hi),
                s_lo,
                s_hi,
                speed: cruise,
                cruise,
                length: 0.55,
                lane,
                signalled: false,
                dir,
                base_y: KERB + 0.03,
                scale: PERSON_SCALE * rng.range(0.88, 1.12),
                seed: rng.next_u32(),
                turn_s0: f32::NAN,
                turn: [0.0; 4],
                tone: rng.below(SKIN_TONES) as u8,
            },
        ));
    };
    for a in (0..ax.len()).step_by(2) {
        let (c, h) = (ax.center(a), ax.half(a));
        for side in [-1.0f32, 1.0] {
            // The pavement on the water side of an embankment road is water.
            let neighbour = if side > 0.0 { a + 1 } else { a.wrapping_sub(1) };
            if drowned(a) || drowned(neighbour) {
                continue;
            }
            for _ in 0..per_pavement {
                walk(
                    &mut crowd,
                    &mut rng,
                    false,
                    a,
                    side,
                    c + side * (h + 1.9),
                    cz0,
                    cz1,
                );
            }
        }
    }
    for k in (0..az.len()).step_by(2) {
        let (c, h) = (az.center(k), az.half(k));
        for side in [-1.0f32, 1.0] {
            for _ in 0..per_pavement {
                let (s_lo, s_hi) = x_run(k, &mut rng);
                walk(
                    &mut crowd,
                    &mut rng,
                    true,
                    k,
                    side,
                    c + side * (h + 1.9),
                    s_lo,
                    s_hi,
                );
            }
        }
    }
    // And a scatter of people in the parks, strolling the long way across.
    for (px0, pz0, px1, pz1) in park_walks.iter().copied() {
        for _ in 0..3 {
            if px1 - px0 > pz1 - pz0 {
                let across = rng.range(pz0, pz1);
                walk(&mut crowd, &mut rng, true, 900, 1.0, across, px0, px1);
            } else {
                let across = rng.range(px0, px1);
                walk(&mut crowd, &mut rng, false, 901, 1.0, across, pz0, pz1);
            }
        }
    }
    crowd.sort_by_key(|(c, _)| *c);

    // --- River traffic. A few barges, running well past the city limits so
    // they arrive out of the haze rather than popping in at the bank.
    let mut river_traffic: [Vec<(usize, Mover)>; 4] = Default::default();
    let boat_count = match plan.water {
        Water::River => 4,
        Water::Bay => 6,
        Water::Dry => 0,
    };
    // The reaches between the bridges. A barge stands two and a half metres
    // out of the water and a bridge deck sits at kerb height a metre above it,
    // so nothing on this river can pass under anything — the boats were
    // sailing straight through every bridge in the city. On a real river with
    // low crossings, craft that cannot clear them work the reach between two,
    // which is what these do now.
    let mut reaches: Vec<(f32, f32)> = Vec::new();
    {
        let mut spans: Vec<(f32, f32)> = (0..az.len())
            .step_by(2)
            .filter(|k| plan.water == Water::River && is_bridge(*k))
            .map(|k| {
                let (c, half) = (az.center(k), az.half(k));
                (c - half - 12.0, c + half + 12.0)
            })
            .collect();
        spans.sort_by(|a, c| a.0.partial_cmp(&c.0).unwrap());
        let mut cursor = cz0 - 150.0;
        for (a, c) in &spans {
            if a - cursor > 60.0 {
                reaches.push((cursor, *a));
            }
            cursor = cursor.max(*c);
        }
        if cz1 + 150.0 - cursor > 60.0 {
            reaches.push((cursor, cz1 + 150.0));
        }
        if reaches.is_empty() {
            reaches.push((cz0 - 150.0, cz1 + 150.0));
        }
    }
    for _ in 0..boat_count {
        let dir = if rng.chance(0.5) { 1.0 } else { -1.0 };
        let (lo, hi) = if plan.water == Water::River {
            let r = reaches[rng.below(reaches.len())];
            fit_run(r.0, r.1, 24.0)
        } else {
            (cz0 - 150.0, cz1 + 150.0)
        };
        // What kind of vessel: mostly working craft, with the odd yacht.
        let boat = match rng.below(10) {
            0..=3 => Boat::Barge,
            4..=6 => Boat::Tug,
            7 => Boat::Container,
            _ => Boat::Yacht,
        };
        let boat_cruise = rng.range(boat.speed().0, boat.speed().1);
        let (lo, hi) = fit_run(lo, hi, boat.length());
        river_traffic[boat as usize].push((
            rng.below(BOAT_COLORS.len()),
            Mover {
                along_x: false,
                // In a channel, keep to the correct side of it, as they do;
                // in open water, spread out.
                fixed: match plan.water {
                    Water::Bay => shore - rng.range(25.0, 170.0),
                    _ => (rx0 + rx1) * 0.5 - dir * (rx1 - rx0) * 0.22,
                },
                s: rng.range(lo, hi),
                s_lo: lo,
                s_hi: hi,
                speed: boat_cruise,
                cruise: boat_cruise,
                length: boat.length(),
                lane: u16::MAX,
                signalled: false,
                dir,
                base_y: -RIVER_DEPTH + 0.85,
                scale: rng.range(0.85, 1.25),
                seed: rng.next_u32(),
                turn_s0: f32::NAN,
                turn: [0.0; 4],
                tone: 0,
            },
        ));
    }
    for list in river_traffic.iter_mut() {
        list.sort_by_key(|(c, _)| *c);
    }

    // --- The road network outside the grid, laid before anything it serves.
    //
    // Order matters and it is the reverse of what is natural to write: the
    // roads claim their corridors first, so every works, retail park and
    // playing field placed afterwards is placed *beside* a road rather than in
    // a field, and the countryside built last keeps off all of it.
    let mut routes: Vec<Route> = Vec::new();
    let mut motorway_vehicles = 0usize;
    let crossings;
    let ring = add_ring(&mut b, extent * 1.12, extent * 1.12, &mut rng);
    routes.push(ring.clone());
    // Lanes out across the fields. Eight bearings, skipping any that would run
    // straight down the water.
    for i in 0..8 {
        let a = (i as f32 + 0.5) / 8.0 * TAU;
        let start = (
            extent * 1.12 * a.cos() * 0.98,
            extent * 1.12 * a.sin() * 0.98,
        );
        if let Water::River = plan.water {
            // Not along the channel, and not starting in it.
            if start.0 > rx0 - 20.0 && start.0 < rx1 + 20.0 {
                continue;
            }
        }
        // Out to where the haze is complete. A lane that stops short of that
        // stops *visibly*, in the middle of a field, going nowhere — which is
        // what every one of them did at two half-widths.
        routes.push(add_lane(&mut b, start, a, extent * 30.0, &mut rng));
    }
    // The bypass, out beyond the retail ring, running parallel to the water so
    // it needs no crossing of its own.
    let hx = match plan.water {
        Water::Bay => shore + extent * 3.4,
        _ => extent * 3.3,
    };
    // Likewise the bypass: it has to run off the edge of the world rather
    // than end in a field. Past the fog its ends cannot be seen at all, which
    // is the only honest way to terminate a road that is meant to be going
    // somewhere else.
    let hlen = extent * 30.0;
    // A straight connector from the ring out to the junction, which is what
    // the overbridge carries.
    let connector = Route {
        pts: vec![(extent * 1.10, 0.0), (hx + extent * 0.5, 0.0)],
        width: 7.0,
    };
    pave(&mut b, &connector.pts, 7.0, false, &mut rng);
    let highway = add_highway(
        &mut b,
        false,
        hx,
        -hlen,
        hlen,
        0.0,
        Some(&connector),
        3,
        &mut rng,
    );
    routes.push(connector);

    // --- The other two crossings. A motorway has more than one kind, and the
    // junction bridge is the only one that carries cars: a green bridge for
    // wildlife, wide and planted so an animal on it never sees the traffic,
    // and a footbridge with switchback ramps rather than steps.
    {
        let span = extent * 0.10 + 26.0;
        add_green_bridge(&mut b, false, hx, extent * 1.35, span, RURAL_Y, &mut rng);
        add_foot_bridge(&mut b, false, hx, -extent * 0.85, span, RURAL_Y, &mut rng);
        crossings = 3;
    }

    // --- Motorway traffic.
    //
    // A motorway with nothing on it is a painted field. It is also where the
    // freight is: the mix here is deliberately nothing like the city's, which
    // is four-fifths cars — a trunk road at any hour is a third lorries, and
    // that is what makes the port, the depot and the rail yard look like parts
    // of the same system rather than three separate models.
    //
    // These movers never turn. `maybe_turn` only considers a junction whose
    // crossing road the vehicle is actually within, and the motorway sits
    // three half-widths outside the furthest street, so nothing matches.
    {
        // Only the stretch that can be seen. The road runs thirty half-widths
        // so its ends vanish into haze, but the haze is complete at twenty-six
        // and there is no sense simulating vehicles nobody can ever see.
        let run = extent * 6.0;
        for (i, (t, dir)) in highway.lanes.iter().enumerate() {
            // Lorries keep left, the outer lanes are cars overtaking. `i % n`
            // counts out from the median, so this reads the lane position.
            let per_side = highway.lanes.len() / 2;
            let from_median = i % per_side;
            let outer = from_median as f32 / (per_side.max(2) - 1) as f32;
            let count = (14.0 * (1.0 - outer * 0.45)) as usize;
            for _ in 0..count {
                let roll = rng.f() + outer * 0.55;
                let kind = if roll < 0.20 {
                    Vehicle::Lorry
                } else if roll < 0.34 {
                    Vehicle::Delivery
                } else if roll < 0.44 {
                    Vehicle::Van
                } else if roll < 0.50 {
                    Vehicle::Bus
                } else if roll < 0.60 {
                    Vehicle::Pickup
                } else if roll < 0.70 {
                    Vehicle::Suv
                } else if roll < 0.80 {
                    Vehicle::Estate
                } else if roll < 0.90 {
                    Vehicle::Hatch
                } else {
                    Vehicle::Car
                };
                // Motorway speeds, and the outer lanes move faster.
                let (s0, s1) = kind.speed();
                let cruise = rng.range(s0, s1) * (1.85 + outer * 0.45);
                let (a, c) = fit_run(-run, run, kind.length());
                road[kind as usize].push((
                    rng.below(kind.colors().len()),
                    Mover {
                        along_x: highway.along_x,
                        fixed: highway.fixed + t,
                        s: rng.range(a, c),
                        s_lo: a,
                        s_hi: c,
                        speed: cruise,
                        cruise,
                        length: kind.length(),
                        // Well clear of the street lanes, which are packed as
                        // `(road_index << 2) | axis | direction` and never
                        // reach four figures.
                        lane: 0x4000 | i as u16,
                        signalled: false,
                        dir: *dir,
                        base_y: highway.deck + 0.03,
                        scale: VEHICLE_SCALE * rng.range(0.95, 1.05),
                        seed: rng.next_u32(),
                        turn_s0: f32::NAN,
                        turn: [0.0; 4],
                        tone: 0,
                    },
                ));
                motorway_vehicles += 1;
            }
        }
    }

    // --- Infrastructure: the ugly half of a city, sited out where the land is
    // cheap. All of it is placed after the countryside so it claims ground
    // that fields have already taken, and before the railway so the line can
    // be refused a footprint the power station is standing on.
    let mut works = Works {
        motorway_vehicles,
        crossings,
        ..Default::default()
    };
    let mut served: Vec<(f32, f32)> = Vec::new();
    {
        // Away from the water, and far enough out that the city is between it
        // and nothing. Four candidate corners, first one that fits wins —
        // there is a river through two of them.
        let d = extent * 2.4;
        for (sx, sz) in [(-1.0f32, -1.0f32), (1.0, 1.0), (-1.0, 1.0), (1.0, -1.0)] {
            let (px, pz) = (sx * d, sz * d * 0.8);
            if add_power_station(&mut b, px, pz, &mut rng) {
                works.plant = Some((px, pz));
                served.push((px, pz));
                break;
            }
        }
        // The line runs from the station past the city and out the far side:
        // transmission lines do not stop at the town they feed, and a run that
        // ends at the city edge looks like it was drawn for the picture.
        if let Some((px, pz)) = works.plant {
            let far = (-px * 1.9, -pz * 1.9);
            works.pylons = add_power_line(&mut b, (px, pz), far, 92.0, &mut rng);
            // A branch off at right angles, so the grid is a network rather
            // than one line.
            let (bx, bz) = (-pz * 1.5, px * 1.5);
            works.pylons += add_power_line(&mut b, (px * 0.35, pz * 0.35), (bx, bz), 92.0, &mut rng);
        }
    }

    // --- The port. Only where there is water to put it on, and clear of the
    // city rather than through the middle of it.
    //
    // Several candidate berths rather than one: a single fixed site is fine
    // for the city size it was tuned at and silently places nothing at every
    // other, which is what it did — a six-block city had no port at all.
    {
        let mut berths: Vec<(f32, f32, f32, f32)> = Vec::new();
        match plan.water {
            Water::River => {
                for k in [0.35f32, 1.10, 1.90] {
                    // Downstream and upstream, on each bank.
                    berths.push((rx1, cz1 + extent * k, 1.0, 1.0));
                    berths.push((rx0, cz1 + extent * k, -1.0, 1.0));
                    berths.push((rx1, cz0 - extent * (k + 1.5), 1.0, 1.0));
                    berths.push((rx0, cz0 - extent * (k + 1.5), -1.0, 1.0));
                }
            }
            Water::Bay => {
                for k in [1.2f32, 0.2, 2.2] {
                    berths.push((shore, cz0 - extent * k, 1.0, 1.0));
                    berths.push((shore, cz1 + extent * (k - 0.8), 1.0, 1.0));
                }
            }
            Water::Dry => {}
        }
        for (quay, lo, inland, _) in berths {
            let hi = lo + extent * 1.5;
            if add_port(&mut b, false, quay, lo, hi, inland, &mut rng) {
                works.port = true;
                served.push((quay + inland * 54.0, lo + extent * 0.75));
                break;
            }
        }
    }

    // --- The fringe: sport, retail and parking.
    //
    // All of it goes in the ring between the last street and the fields, which
    // is where every city in the world puts it, and for the same reason: these
    // are the uses that need more ground than a block has and can pay least
    // for it. It also puts them exactly where the eye goes on the way out to
    // the countryside, so the transition from city to farmland stops being a
    // hard edge and becomes what it is in life — a scatter of sheds, car parks
    // and playing fields.
    //
    // Sites are tried on a ring, spiralling outward, and every one of them
    // goes through `Occupancy`, so a site that collides is skipped rather than
    // overlapped. What actually landed is counted, because a placement rule
    // that quietly fails looks identical to one that was never written.
    {
        // A flat palette for anything that parks a car.
        let paints: Vec<Color> = [
            Vehicle::Car,
            Vehicle::Hatch,
            Vehicle::Taxi,
            Vehicle::Pickup,
            Vehicle::Van,
            Vehicle::Estate,
            Vehicle::Suv,
        ]
        .iter()
        .flat_map(|k| k.colors().iter().map(|c| Color::from_hex(*c)))
        .collect();
        let lot_bodies: Vec<MeshBuilder> = [
            Vehicle::Car,
            Vehicle::Hatch,
            Vehicle::Taxi,
            Vehicle::Pickup,
            Vehicle::Van,
            Vehicle::Estate,
            Vehicle::Suv,
        ]
        .iter()
        .map(|k| k.body())
        .collect();

        // Candidate sites, ordered so the first ring fills before the second.
        // Generous, because a candidate that collides costs nothing and a
        // shortage of them costs everything: at six blocks a side the first
        // version had fifty-six sites and five layouts out of seven ended up
        // with no courts at all and a single strip. Nothing in the render says
        // so — the counters do.
        let mut sites: Vec<(f32, f32, f32)> = Vec::new();
        for ring in 0..7 {
            let rad = extent * (1.26 + ring as f32 * 0.27);
            let n = 10 + ring * 5;
            for i in 0..n {
                let a = (i as f32 + ring as f32 * 0.37) / n as f32 * TAU;
                sites.push((rad * a.cos(), rad * a.sin(), a));
            }
        }
        let mut next = 0usize;
        // Try each remaining site with a rect of the given half-size, squared
        // to the world rather than to the ring: everything here is a shed and
        // a car park, and sheds are rectangular.
        let take = |occ: &Occupancy, hw: f32, hd: f32, sites: &[(f32, f32, f32)], next: &mut usize| -> Option<(Rect, bool)> {
            // Wraps rather than running off the end. The first version
            // consumed a site on every *attempt*, so a few large footprints
            // that collided with the city walked the cursor past all
            // fifty-six candidates and everything asked for afterwards —
            // every court, every parking deck — silently got nothing.
            for step in 0..sites.len() {
                let idx = (*next + step) % sites.len();
                let (x, z, a) = sites[idx];
                // Long axis across the radius, so frontages face the city.
                let along_x = a.cos().abs() < 0.5;
                let (w, d) = if along_x { (hw, hd) } else { (hd, hw) };
                let r = Rect { x0: x - w, z0: z - d, x1: x + w, z1: z + d };
                // Never in the water.
                if let Water::River = plan.water {
                    if r.x1 > rx0 - 12.0 && r.x0 < rx1 + 12.0 {
                        continue;
                    }
                }
                if let Water::Bay = plan.water {
                    if r.x0 < shore + 12.0 {
                        continue;
                    }
                }
                if occ.free([r.x0, r.z0, r.x1, r.z1]) {
                    *next = idx + 1;
                    return Some((r, along_x));
                }
            }
            None
        };

        // The big draws first: they need the most ground. Through the same
        // site picker as everything else — the stadium used to have four
        // hard-coded corners, which worked until the ring road and the lanes
        // started claiming ground at a similar radius and then it placed
        // nothing at all.
        for _ in 0..4 {
            if let Some((r, _)) = take(&b.occ, 66.0, 54.0, &sites, &mut next) {
                if add_stadium(&mut b, r.cx(), r.cz(), rng.range(0.0, TAU), &mut rng) {
                    works.stadium = Some((r.cx(), r.cz()));
                    served.push((r.cx(), r.cz()));
                    break;
                }
            }
        }
        // Each of these gets several attempts. They are the largest footprints
        // in the city and the ring road and lanes have already broken the
        // fringe into pieces, so the first free-looking site frequently is not
        // one — a single attempt is how the cricket ground vanished from the
        // dense layouts.
        // Progressively smaller as well as progressively further out: a ground
        // that will not fit anywhere at full size is better shrunk than
        // omitted, and nobody can tell a sixty-metre outfield from a
        // seventy-metre one from the air.
        for want in [70.0f32, 58.0, 46.0] {
            if works.ballpark {
                break;
            }
            if let Some((r, _)) = take(&b.occ, want, want, &sites, &mut next) {
                works.ballpark =
                    add_ballpark(&mut b, r.cx(), r.cz(), want * 0.88, rng.range(0.0, TAU), &mut rng);
                if works.ballpark {
                    served.push((r.cx(), r.cz()));
                }
            }
        }
        for want in [76.0f32, 62.0, 50.0, 42.0] {
            if works.cricket {
                break;
            }
            if let Some((r, _)) = take(&b.occ, want, want, &sites, &mut next) {
                works.cricket =
                    add_cricket_ground(&mut b, r.cx(), r.cz(), want * 0.9, rng.range(0.0, TAU), &mut rng);
                if works.cricket {
                    served.push((r.cx(), r.cz()));
                }
            }
        }
        for (hw, hd) in [(78.0f32, 52.0f32), (62.0, 44.0), (48.0, 34.0)] {
            if works.mall {
                break;
            }
            if let Some((r, _)) = take(&b.occ, hw, hd, &sites, &mut next) {
                works.mall = add_mall(&mut b, &r, 2 + rng.below(2), &lot_bodies, &paints, &mut rng);
                if works.mall {
                    served.push((r.cx(), r.cz()));
                }
            }
        }
        for _ in 0..2 {
            if let Some((r, along_x)) = take(&b.occ, 38.0, 34.0, &sites, &mut next) {
                if add_grocery(&mut b, &r, along_x, &mut rng) {
                    works.grocery += 1;
                    served.push((r.cx(), r.cz()));
                    // The car park a supermarket always has in front of it.
                    let lot = if along_x {
                        Rect { x0: r.x0, z0: r.z0, x1: r.x1, z1: r.z1 - r.d() * 0.48 }
                    } else {
                        Rect { x0: r.x0, z0: r.z0, x1: r.x1 - r.w() * 0.48, z1: r.z1 }
                    };
                    works.parked_cars +=
                        add_parking_lot(&mut b, &lot, along_x, 0.5, &lot_bodies, &paints, &mut rng);
                }
            }
        }
        for _ in 0..3 {
            if let Some((r, along_x)) = take(&b.occ, 40.0, 30.0, &sites, &mut next) {
                if add_strip_mall(&mut b, &r, along_x, &mut rng) {
                    works.strips += 1;
                    served.push((r.cx(), r.cz()));
                    let lot = if along_x {
                        Rect { x0: r.x0, z0: r.z0, x1: r.x1, z1: r.z1 - r.d() * 0.36 }
                    } else {
                        Rect { x0: r.x0, z0: r.z0, x1: r.x1 - r.w() * 0.36, z1: r.z1 }
                    };
                    works.parked_cars +=
                        add_parking_lot(&mut b, &lot, along_x, 0.42, &lot_bodies, &paints, &mut rng);
                }
            }
        }
        for _ in 0..2 {
            if let Some((r, _)) = take(&b.occ, 20.0, 17.0, &sites, &mut next) {
                if add_parking_deck(&mut b, &r, 3 + rng.below(3), &lot_bodies, &paints, &mut rng) {
                    works.decks += 1;
                    served.push((r.cx(), r.cz()));
                }
            }
        }
        for _ in 0..2 {
            if let Some((r, along_x)) = take(&b.occ, 26.0, 20.0, &sites, &mut next) {
                works.parked_cars +=
                    add_parking_lot(&mut b, &r, along_x, rng.range(0.25, 0.8), &lot_bodies, &paints, &mut rng);
                served.push((r.cx(), r.cz()));
            }
        }
        // The zoo. Large, so it goes in before the small things take the
        // gaps, and it shrinks rather than being dropped.
        for (hw, hd) in [(72.0f32, 58.0f32), (62.0, 50.0), (54.0, 44.0), (44.0, 37.0)] {
            if works.zoo_animals > 0 {
                break;
            }
            if let Some((r, _)) = take(&b.occ, hw, hd, &sites, &mut next) {
                let n = add_zoo(&mut b, &r, &mut rng);
                if n > 0 {
                    works.zoo_animals = n;
                    served.push((r.cx(), r.cz()));
                }
            }
        }

        // --- Monuments. After the zoo, which is bigger and fussier about
        // ground: monuments placed first took the sites it needed and the
        // dense layouts ended up with no zoo at all.
        //
        // Two or three a city, drawn from a shuffled list so
        // that different seeds are recognisable as different places — which is
        // the whole point of a landmark and the one thing a generator built
        // entirely from rules cannot produce.
        {
            // Its own stream. Drawing these from the city's `rng` shifts
            // every placement after them — it lost the rail freight yard on
            // one layout — which is the third time this exact hazard has
            // caught me and the reason the parked cars have a comment about it.
            // Seeded on the *layout* as well as the seed. Seeded on the seed
            // alone, every layout drew the same shuffle and so got the same
            // two landmarks — which is the opposite of the point: a monument
            // exists to make one city recognisable as not another.
            let li = Layout::ALL
                .iter()
                .position(|l| *l == params.layout)
                .unwrap_or(0) as u64;
            let mut mon_rng =
                Rng::new(params.seed ^ 0x51d2_7fa4_1c93_6e08 ^ (li.wrapping_mul(0x9e37_79b9_7f4a_7c15)));
            let mut kinds = MONUMENTS;
            for i in (1..kinds.len()).rev() {
                kinds.swap(i, mon_rng.below(i + 1));
            }
            let want = 3 + mon_rng.below(2);
            for kind in kinds.into_iter() {
                if works.monuments.len() >= want {
                    break;
                }
                let f = kind.footprint();
                // Tried on the fringe ring like everything else, but from the
                // inside out: a monument nobody passes is not a monument.
                for _ in 0..3 {
                    if let Some((r, _)) = take(&b.occ, f + 6.0, f + 6.0, &sites, &mut next) {
                        if add_monument(&mut b, kind, r.cx(), r.cz(), mon_rng.range(0.0, TAU), &mut mon_rng)
                        {
                            works.monuments.push(kind.label());
                            served.push((r.cx(), r.cz()));
                            break;
                        }
                    }
                }
            }
        }

        // A data centre, and masts. Both belong on the fringe: one because it
        // needs a substation and cheap ground, the other because a mast is
        // sited for coverage and coverage means high and clear.
        for (hw, hd) in [(58.0f32, 44.0f32), (46.0, 36.0), (38.0, 30.0), (31.0, 24.0)] {
            if works.data_centre {
                break;
            }
            if let Some((r, along_x)) = take(&b.occ, hw, hd, &sites, &mut next) {
                if add_data_centre(&mut b, &r, along_x, &mut rng) {
                    works.data_centre = true;
                    served.push((r.cx(), r.cz()));
                }
            }
        }
        for _ in 0..3 {
            if let Some((r, _)) = take(&b.occ, 12.0, 12.0, &sites, &mut next) {
                if add_telecom_mast(
                    &mut b,
                    r.cx(),
                    r.cz(),
                    rng.range(42.0, 78.0),
                    &mut rng,
                ) {
                    works.masts += 1;
                }
            }
        }

        // Courts: small, so they fit in the gaps the big things left — but
        // only if what is asked for is the size of the bank rather than a
        // fixed rectangle that happens to suit one layout. A dense old town
        // has no 68-by-60 gaps in its fringe at all, so it got two courts;
        // asking for the space a bank actually needs, and settling for a
        // shorter bank when the long one will not fit, gets it a full set.
        for (kind, want) in [
            (Court::Basketball, 2usize),
            (Court::Tennis, 3),
            (Court::Pickleball, 4),
            (Court::Netball, 2),
        ] {
            let (l, w) = kind.size();
            for n in (1..=want).rev() {
                let hw = l * 0.5 + 5.0;
                let hd = (n as f32 * (w + 7.0)) * 0.5 + 3.0;
                if let Some((r, along_x)) = take(&b.occ, hw, hd, &sites, &mut next) {
                    let made = add_court_bank(&mut b, r.cx(), r.cz(), kind, n, along_x, &mut rng);
                    if made > 0 {
                        served.push((r.cx(), r.cz()));
                        works.courts += made;
                        break;
                    }
                }
            }
        }
    }

    // --- Logistics. A distribution depot where the motorway meets the
    // connector, and a rail freight yard alongside the line. Between them and
    // the container terminal, a box has somewhere to arrive, somewhere to be
    // trans-shipped and somewhere to leave from — which is what turns three
    // separate models into one system.
    {
        let hgv: Vec<MeshBuilder> = [Vehicle::Lorry, Vehicle::Delivery, Vehicle::Van]
            .iter()
            .map(|k| k.body())
            .collect();
        // Depots go where the lorries are. Beside the junction, on the city
        // side of the motorway so the traffic does not have to cross it.
        // Several berths along the motorway rather than two, and a smaller
        // footprint to fall back on. One fixed site is how the power station,
        // the port, the stadium and the cricket ground each managed to place
        // nothing at all on some layout or other; there is no reason to
        // believe a depot is different.
        'depot: for half in [46.0f32, 34.0] {
            // Both sides of the motorway. On a waterfront layout the city side
            // of it is the bay.
            for x0f in [-1.15f32, 0.24] {
                for k in [1.9f32, -1.9, 1.15, -1.15, 2.7, -2.7] {
                let cz = k * extent;
                let r = Rect {
                    x0: hx + extent * x0f,
                    z0: cz - half,
                    x1: hx + extent * (x0f + 0.93),
                    z1: cz + half,
                };
                if add_depot(&mut b, &r, false, &hgv, &mut rng) {
                    works.depot = true;
                    served.push((r.cx(), r.cz()));
                    break 'depot;
                }
                }
            }
        }
    }

    // --- The airport.
    //
    // Where the railway will go, answered early. The line is dead straight,
    // laid along a street, and spans the whole map, so an airfield sited on
    // the same axis gets a viaduct down the runway — piers on the asphalt and
    // the deck across the threshold, both structures rendering happily because
    // neither knew about the other. Steering the *line* round the field does
    // not work: on a small city every street lies inside the field's
    // three-hundred-metre band. So the field moves instead. Neither this nor
    // the offset below draws from `rng`, so nothing downstream shifts.
    let rail_along_x = !matches!(plan.water, Water::River);
    let rail_fixed = rail_street(rail_along_x, &ax, &az, blocks, plan.water, rx0, rx1);
    let mut airfield: Option<[f32; 4]> = None;
    //
    // A runway is fourteen hundred metres of dead straight ground and nothing
    // else in this world needs that, so it is fussy about where it goes. Two
    // things had to be got right. It is placed *before* the countryside, like
    // everything else this large — afterwards it never found clear ground,
    // because the fields had taken it. And it is laid *along* the gap between
    // two country lanes rather than across one: the lanes radiate on eight
    // bearings, so the gaps between them are centred on the axes, and a runway
    // pointed down an axis sits in a gap instead of crossing two.
    {
        let cands: [(f32, f32, bool); 4] = [
            (0.0, 1.0, false),
            (0.0, -1.0, false),
            (-1.0, 0.0, true),
            (1.0, 0.0, true),
        ];
        // Shorter runways as a fallback, and more radii. A regional field with
        // a nine-hundred-metre strip is still an airport; no airport at all is
        // what the first version produced on most seeds.
        'airport: for (d, rw) in [
            (extent * 6.5, 1400.0f32),
            (extent * 8.2, 1400.0),
            (extent * 5.2, 1100.0),
            (extent * 7.3, 1100.0),
            (extent * 4.4, 900.0),
            (extent * 9.6, 900.0),
        ] {
            for (ux, uz, along) in cands {
                // Never over the water.
                let (mut px, mut pz) = (ux * d, uz * d);
                // Slide the field off the railway. Only the coordinate on the
                // line's fixed axis matters; the runway stays parallel to the
                // axis it was chosen for, so it still sits in the gap between
                // two country lanes — those radiate on eight bearings and are
                // kilometres apart this far out.
                {
                    let f = airport_field(px, pz, along, rw);
                    let (lo, hi) = if rail_along_x { (f[1], f[3]) } else { (f[0], f[2]) };
                    let here = if rail_along_x { pz } else { px };
                    let half = (hi - lo) * 0.5;
                    if rail_fixed > lo - 30.0 && rail_fixed < hi + 30.0 {
                        let push = half + 30.0;
                        let moved = if here >= rail_fixed {
                            rail_fixed + push
                        } else {
                            rail_fixed - push
                        };
                        if rail_along_x {
                            pz = moved;
                        } else {
                            px = moved;
                        }
                    }
                }
                if matches!(plan.water, Water::River) && px.abs() < rx1.abs() + 200.0 && ux != 0.0 {
                    continue;
                }
                let n = add_airport(&mut b, px, pz, along, rw, &mut rng);
                if n > 0 {
                    works.aircraft = n;
                    served.push((px, pz));
                    airfield = Some(airport_field(px, pz, along, rw));
                    break 'airport;
                }
            }
        }
    }

    // --- The bridge. One crossing of the water is a landmark rather than a
    // deck on piers: towers, a catenary and hangers, painted the colour the
    // famous one is. Sited downstream of the city so it is seen against open
    // water rather than against the skyline.
    match plan.water {
        Water::River => {
            for k in [1.35f32, -1.35, 1.9, -1.9, 2.5, -2.5, 1.05, -1.05, 3.1, -3.1] {
                if add_suspension_bridge(
                    &mut b,
                    true,
                    cz1 * k,
                    rx0,
                    rx1,
                    KERB + 9.0,
                    &mut rng,
                ) {
                    works.suspension = true;
                    break;
                }
            }
        }
        Water::Bay => {
            // Span as well as position. A bay has no far bank to aim at, so
            // the crossing is a headland-to-headland span picked by trial —
            // and one fixed width found nowhere to stand on the layout that
            // has the most water in it.
            'bay: for wdt in [0.9f32, 0.62, 0.42, 0.28] {
                for k in [1.3f32, -1.3, 1.8, -1.8, 2.4, -2.4, 1.0, -1.0, 3.0, -3.0] {
                    if add_suspension_bridge(
                        &mut b,
                        true,
                        cz1 * k,
                        shore - extent * wdt,
                        shore,
                        KERB + 9.0,
                        &mut rng,
                    ) {
                        works.suspension = true;
                        break 'bay;
                    }
                }
            }
        }
        Water::Dry => {}
    }

    // --- Access roads. Every site placed above gets one, to whichever route
    // passes nearest. A car park in a field with no road to it is not a car
    // park, and once noticed it cannot be unnoticed.
    for site in &served {
        if add_spur(&mut b, *site, &routes, &mut rng) {
            works.spurs += 1;
        }
    }

    // --- The countryside. Everything past the last road was a flat green
    // plane running to the fog: fine at street level, and the first thing you
    // notice from the air where it is half the frame.
    add_countryside(
        &mut b,
        extent,
        e,
        match plan.water {
            Water::River => Some((rx0, rx1)),
            Water::Bay => Some((-e, shore)),
            Water::Dry => None,
        },
        warm,
        &mut rng,
    );

    // --- Mountains, out past everything. Built after the countryside so they
    // take whatever ground is left, and they need a lot of it.
    let range = add_mountains(&mut b, extent, e, &mut rng);
    works.peaks = range.peaks;
    works.skyline_gap = range.gap;

    // --- Tunnels, once the hills are up and the lanes are already laid
    // through them. Its own stream: everything downstream of here is placed
    // from `rng`, and drawing even one number from it moves the railway, the
    // freight yard and the monuments — which is how this file has twice lost a
    // power station to a feature that had nothing to do with power stations.
    let mut tunnel_rng = Rng::new(params.seed ^ 0x5d41_402a_bc4b_2a76);
    works.tunnels = add_road_tunnels(&mut b, &routes, &range.summits, &mut tunnel_rng);

    // --- The railway. It ignores the street grid entirely, which is the
    // point of it: everything else in this city is at ground level and
    // parallel to something. Laid along whichever axis the water does not run
    // down, so the viaduct does not spend its length over a river.
    // Over a street, not over the blocks. A viaduct on an arbitrary line runs
    // straight through the buildings it passes; above a carriageway it does
    // what a real one does, and the road underneath keeps working.
    // Roads are the even entries; blocks are the odd ones. When the line runs
    // parallel to the river it must not be laid *along* the channel — that
    // stands every pier in the water and gives the barges something to hit —
    // so step to the next road until the chosen one is clear of it.
    let rail_fixed = rail_street(rail_along_x, &ax, &az, blocks, plan.water, rx0, rx1);
    let rail_lo = -e * 0.55;
    let rail_hi = e * 0.55;
    // Where the line crosses open water, in the line's own coordinate.
    let rail_channel = match plan.water {
        Water::River if !rail_along_x => None,
        Water::River => Some((rx0, rx1)),
        Water::Bay if rail_along_x => Some((-e, shore)),
        _ => None,
    };
    let railway = add_railway(
        &mut b,
        rail_along_x,
        rail_fixed,
        rail_lo,
        rail_hi,
        rail_channel,
        extent * 3.0,
        &mut rng,
    );

    // The freight yard goes beside the line, out past the built area where
    // there is room for five sidings and a gantry.
    // Close enough to be seen. At 0.62 of the line's half-length the yard sat
    // nearly four kilometres out, where the haze is two-thirds of the way to
    // complete and a container is a grey speck.
    // Several berths along the line and a shorter yard to fall back on. Four
    // fixed spots was enough until the monuments started claiming ground, and
    // then one layout quietly lost its freight yard — the same failure this
    // file has now had from the power station, the port, the stadium, the
    // cricket ground, the mall, the depot, the zoo and the airport.
    'yard: for len in [0.8f32, 0.58, 0.42] {
        for frac in [0.26f32, -0.26, 0.44, -0.44, 0.10, -0.10, 0.62, -0.62] {
            for side in [1.0f32, -1.0] {
                let mid = frac * e * 0.55;
                let (ylo, yhi) = (mid - extent * len, mid + extent * len);
                if add_rail_yard(&mut b, rail_along_x, rail_fixed, ylo, yhi, side, &mut rng) {
                    works.rail_yard = true;
                    break 'yard;
                }
            }
        }
    }

    works.cameras = b.cameras;
    works.cyclists = cyclists;
    works.meters = meters;
    works.echelon_bays = echelon_bays;
    works.airfield = airfield;
    works.rail = Some((rail_along_x, rail_fixed));
    let (b_claims, b_refusals) = (b.occ.claimed(), b.occ.refused());

    // --- Light transport, before anything is uploaded. Sky access and one
    // bounce, folded into vertex colour; see `bake.rs` for why it goes there.
    let bake = if params.bake {
        bake_light(&mut b, 0.85)
    } else {
        BakeReport::default()
    };

    // --- Hand the batches to the scene.
    let styles = facade_styles();
    let mut triangles = 0usize;
    let mut draws = 0usize;
    let mut windows = Vec::new();
    let mut night_only = Vec::new();
    let mut beacons = Vec::new();
    let mut neon: Vec<(ObjectId, bool)> = Vec::new();
    let mut night_sky: Option<ObjectId> = None;
    let mut clouds: Option<ObjectId> = None;
    let mut plumes: Option<ObjectId> = None;
    let mut shafts: Option<ObjectId> = None;

    let mut facades = std::mem::take(&mut b.facades);
    for (i, mb) in facades.iter_mut().enumerate() {
        let mb = std::mem::take(mb);
        if mb.is_empty() {
            continue;
        }
        triangles += mb.tri_count();
        let (albedo, lit, rough) = facade_textures(&styles[i], params.seed ^ (0x51 + i as u64));
        // Physical, not Standard: a curtain wall is glazing set in a frame,
        // and the clearcoat lobe is exactly that — a second, sharper specular
        // over the base, which a single-lobe BRDF cannot produce. Masonry gets
        // little of it and glass a lot.
        let mut phys = PhysicalMaterial::new(Color::WHITE)
            .with_roughness(styles[i].roughness)
            .with_metalness(styles[i].metalness)
            .with_clearcoat(styles[i].clearcoat, 0.14)
            .with_map(albedo)
            .with_roughness_map(rough)
            .with_emissive(Color::from_hex(0xfff0d8), 0.0);
        phys.ior = 1.52;
        phys.emissive_map = Some(lit);
        let mat = Material::Physical(phys);
        if let Some(id) = add_batch(scene, &format!("facades-{i}"), mb, mat, true, true) {
            windows.push((id, styles[i].emissive_gain));
            draws += 1;
        }
    }

    let (asphalt, paving) = ground_textures();
    let opaque: [(&str, MeshBuilder, Material, bool, bool); 9] = [
        (
            "terrain",
            std::mem::take(&mut b.grass),
            lit_standard(0xffffff, 0.97, 0.0),
            false,
            true,
        ),
        (
            "roads",
            std::mem::take(&mut b.road),
            Material::Standard(
                StandardMaterial::new(Color::WHITE)
                    .with_roughness(0.92)
                    .with_metalness(0.0)
                    .with_map(asphalt),
            ),
            false,
            true,
        ),
        (
            "markings",
            std::mem::take(&mut b.paint),
            lit_standard(0xffffff, 0.62, 0.0),
            false,
            true,
        ),
        (
            "sidewalks",
            std::mem::take(&mut b.pads),
            Material::Standard(
                StandardMaterial::new(Color::WHITE)
                    .with_roughness(0.88)
                    .with_metalness(0.0)
                    .with_map(paving),
            ),
            true,
            true,
        ),
        (
            "water",
            std::mem::take(&mut b.water),
            Material::Physical(
                PhysicalMaterial::new(Color::WHITE)
                    .with_roughness(0.11)
                    .with_metalness(0.0)
                    .with_clearcoat(0.55, 0.05),
            ),
            false,
            true,
        ),
        (
            "trim",
            std::mem::take(&mut b.trim),
            lit_standard(0xffffff, 0.80, 0.06),
            true,
            true,
        ),
        (
            "parked",
            std::mem::take(&mut b.parked),
            // Painted metal under lacquer, like the traffic it is parked
            // among.
            Material::Physical(
                PhysicalMaterial::new(Color::WHITE)
                    .with_roughness(0.34)
                    .with_metalness(0.22)
                    .with_clearcoat(0.85, 0.06),
            ),
            true,
            true,
        ),
        (
            // Flat scatter on the ground: same material as the foliage it fell
            // off, its own batch only so the light bake can skip it.
            "litter",
            std::mem::take(&mut b.litter),
            Material::Physical(
                PhysicalMaterial::new(Color::WHITE)
                    .with_roughness(0.94)
                    .with_metalness(0.0),
            ),
            false,
            true,
        ),
        (
            "foliage",
            std::mem::take(&mut b.foliage),
            // Sheen: leaves are covered in fine structure and go pale and
            // bright at grazing angles, which a plain diffuse lobe cannot do.
            Material::Physical(
                PhysicalMaterial::new(Color::WHITE)
                    .with_roughness(0.92)
                    .with_metalness(0.0)
                    .with_sheen(0.26, Color::from_hex(0xcadd9e), 0.6),
            ),
            true,
            true,
        ),
    ];
    for (name, mb, mat, cast, receive) in opaque {
        triangles += mb.tri_count();
        if add_batch(scene, name, mb, mat, cast, receive).is_some() {
            draws += 1;
        }
    }

    let puddles = std::mem::take(&mut b.puddle);
    triangles += puddles.tri_count();
    if add_batch(
        scene,
        "puddles",
        puddles,
        Material::Standard(
            StandardMaterial::new(Color::WHITE)
                .with_roughness(0.06)
                .with_metalness(0.0),
        ),
        false,
        true,
    )
    .is_some()
    {
        draws += 1;
    }

    // Smoke and steam, on their own object so they can take the same
    // sky-coloured emissive the cloud deck does — see `Batches::plume`.
    let plume = std::mem::take(&mut b.plume);
    triangles += plume.tri_count();
    if let Some(id) = add_batch(
        scene,
        "plumes",
        plume,
        Material::Standard(
            StandardMaterial::new(Color::from_hex(0xf2f4f6))
                .with_roughness(0.98)
                .with_metalness(0.0),
        ),
        false,
        false,
    ) {
        plumes = Some(id);
        draws += 1;
    }

    let glow = std::mem::take(&mut b.glow);
    triangles += glow.tri_count();
    if let Some(id) = add_batch(
        scene,
        "lamp-pools",
        glow,
        Material::Basic(BasicMaterial {
            color: Color::WHITE,
            ..Default::default()
        }),
        false,
        false,
    ) {
        night_only.push(id);
        draws += 1;
    }

    // Neon and backlit signage: unlit, night only, and brighter than the lamp
    // pools next to them.
    for (name, mb, blinks) in [
        ("neon", std::mem::take(&mut b.neon), false),
        ("neon-blink", std::mem::take(&mut b.neon_blink), true),
    ] {
        triangles += mb.tri_count();
        if let Some(id) = add_batch(
            scene,
            name,
            mb,
            Material::Basic(BasicMaterial {
                color: Color::WHITE,
                ..Default::default()
            }),
            false,
            false,
        ) {
            neon.push((id, blinks));
            draws += 1;
        }
    }

    // The lamp shafts. Translucent, so they are drawn after everything opaque
    // and never write depth over the street they are lighting.
    let shaft = std::mem::take(&mut b.shaft);
    triangles += shaft.tri_count();
    if let Some(id) = add_batch(
        scene,
        "lamp-shafts",
        shaft,
        Material::Basic(BasicMaterial {
            color: Color::WHITE,
            transparent: true,
            opacity: 0.16,
            side: 2,
            ..Default::default()
        }),
        false,
        false,
    ) {
        shafts = Some(id);
        draws += 1;
    }

    // --- Clouds, on a deck between the towers and the stars.
    {
        let mb = cloud_deck(e * 0.60, extent * 1.7, 0.62, params.seed);
        triangles += mb.tri_count();
        if let Some(id) = add_batch(
            scene,
            "clouds",
            mb,
            // Lit, not unlit: they take the sun's colour and go warm at dawn
            // with everything else, which is most of what makes a sky read as
            // being at a time of day.
            Material::Standard(
                StandardMaterial::new(Color::from_hex(0xf4f6fa))
                    .with_roughness(0.98)
                    .with_metalness(0.0),
            ),
            false,
            false,
        ) {
            clouds = Some(id);
            draws += 1;
        }
    }

    // --- The night sky. A shell of stars and a moon, well inside the
    // camera's far plane and outside everything else, shown only after dark.
    {
        let r = extent * 7.0;
        // The rim colour the stars fade into: the night background, so they
        // blend rather than each carrying a dark halo.
        let night_bg = sky_at(0.75).background;
        let mut sky_mb = star_field(r, params.seed, night_bg);
        let moon_dir = Vector3::new(-0.42, 0.66, 0.62);
        let moon_mb = moon_disc(r * 0.97, moon_dir);
        sky_mb.append_at(&moon_mb, Vector3::ZERO, 0.0, 1.0, Color::WHITE);
        triangles += sky_mb.tri_count();
        if let Some(id) = add_batch(
            scene,
            "night-sky",
            sky_mb,
            Material::Basic(BasicMaterial {
                color: Color::WHITE,
                ..Default::default()
            }),
            false,
            false,
        ) {
            night_sky = Some(id);
            draws += 1;
        }
    }

    let beacon = std::mem::take(&mut b.beacon);
    triangles += beacon.tri_count();
    if let Some(id) = add_batch(
        scene,
        "beacons",
        beacon,
        Material::Basic(BasicMaterial {
            color: Color::WHITE,
            ..Default::default()
        }),
        false,
        false,
    ) {
        beacons.push(id);
        draws += 1;
    }

    // Everything after this point is instanced, and what it costs depends on
    // the camera, so the static total is frozen here.
    let static_triangles = triangles;

    // --- Movers. Each population gets its near meshes and one shared
    // impostor; see `Fleet` for why the far level is a single mesh.
    let mut fleets: Vec<Fleet> = Vec::new();
    let car_fleet = 0usize;
    for (kind, src) in [
        (Vehicle::Car, std::mem::take(&mut road[0])),
        (Vehicle::Hatch, std::mem::take(&mut road[1])),
        (Vehicle::Sports, std::mem::take(&mut road[2])),
        (Vehicle::Taxi, std::mem::take(&mut road[3])),
        (Vehicle::Pickup, std::mem::take(&mut road[4])),
        (Vehicle::Van, std::mem::take(&mut road[5])),
        (Vehicle::Bus, std::mem::take(&mut road[6])),
        (Vehicle::Lorry, std::mem::take(&mut road[7])),
        (Vehicle::Police, std::mem::take(&mut road[8])),
        (Vehicle::Estate, std::mem::take(&mut road[9])),
        (Vehicle::Suv, std::mem::take(&mut road[10])),
        (Vehicle::Ambulance, std::mem::take(&mut road[11])),
        (Vehicle::FireEngine, std::mem::take(&mut road[12])),
        (Vehicle::Delivery, std::mem::take(&mut road[13])),
        (Vehicle::Bicycle, std::mem::take(&mut road[14])),
    ] {
        let near = vec![Arc::new(kind.geometry())];
        // A bus is legible three times further off than a car is.
        let lod = match kind {
            Vehicle::Sports | Vehicle::Hatch => 220.0,
            Vehicle::Car | Vehicle::Taxi | Vehicle::Police => 240.0,
            Vehicle::Estate | Vehicle::Suv => 260.0,
            Vehicle::Pickup | Vehicle::Van => 300.0,
            Vehicle::Delivery => 340.0,
            Vehicle::Ambulance => 380.0,
            Vehicle::FireEngine => 460.0,
            Vehicle::Bus | Vehicle::Lorry => 420.0,
            // Small, so it stops being legible sooner than anything else.
            Vehicle::Bicycle => 150.0,
        };
        fleets.push(spawn_fleet(
            scene,
            kind.label(),
            &near,
            Some(Arc::new(vehicle_impostor(kind))),
            kind.colors(),
            0.34,
            0.22,
            0.85,
            lod,
            Gait::Drive,
            src,
            &mut triangles,
            &mut draws,
        ));
    }
    let walker_poses = vec![
        Arc::new(walker_geometry(1.0)),
        Arc::new(walker_geometry(-1.0)),
    ];
    let walkers_len = crowd.len();
    let people_fleet = fleets.len();
    fleets.push(spawn_fleet(
        scene,
        "people",
        &walker_poses,
        Some(Arc::new(walker_impostor())),
        &WALKER_COLORS,
        0.85,
        0.0,
        0.0,
        170.0,
        Gait::Walk,
        crowd,
        &mut triangles,
        &mut draws,
    ));
    // Skin: one mesh a pose for the whole population, drawn over the clothed
    // meshes. Two draws so nobody's face is the colour of their jacket.
    let mut walker_skin = Vec::new();
    for pose in 0..2 {
        let geom = Arc::new(walker_skin_geometry(if pose == 0 { 1.0 } else { -1.0 }));
        let tris = geom.index.as_ref().map(|i| i.len() / 3).unwrap_or(0);
        {
            let mut im = InstancedMesh::new(
                BufferGeometry::new(),
                // White: the tone arrives per instance.
                Material::Standard(
                    StandardMaterial::new(Color::WHITE)
                        .with_roughness(0.76)
                        .with_metalness(0.0),
                ),
                walkers_len,
            );
            im.geometry = geom.clone();
            let mut obj = Object3D::instanced_mesh(im);
            obj.name = format!("people-skin-{pose}");
            walker_skin.push((scene.add(obj), pose));
            triangles += tris * walkers_len;
            draws += 1;
        }
    }

    // --- Trains. Ordinary movers on a lane of their own at the deck's height:
    // the lane machinery already handles a long vehicle on a bounded run, and
    // a train is nothing but a very long vehicle that never turns. Its wrap
    // point sits inside a tunnel, so the one place the simulation teleports is
    // the one place nobody can see it.
    const TRAIN_LIVERY: [u32; 4] = [0xb0342c, 0x2f5f8a, 0xd8d9dd, 0x2e6f4f];
    let mut train_src: Vec<(usize, Mover)> = Vec::new();
    for i in 0..2 {
        let cars = 3 + rng.below(3);
        let length = 19.0 * cars as f32;
        let (s_lo, s_hi) = fit_run(railway.s_lo, railway.s_hi, length);
        let dir = if i % 2 == 0 { 1.0 } else { -1.0 };
        train_src.push((
            cars - 3,
            Mover {
                along_x: railway.along_x,
                fixed: railway.fixed,
                s: mix(s_lo, s_hi, (i as f32 + 0.5) / 2.0),
                s_lo,
                s_hi,
                speed: rng.range(18.0, 26.0),
                cruise: rng.range(18.0, 26.0),
                length,
                // No lane: a train must not be swept into the road queueing
                // pass, which would have it braking for a bus underneath it.
                lane: u16::MAX,
                signalled: false,
                dir,
                base_y: DECK + 0.55,
                scale: 1.0,
                seed: rng.next_u32(),
                turn_s0: f32::NAN,
                turn: [0.0; 4],
                tone: 0,
            },
        ));
    }
    train_src.sort_by_key(|(c, _)| *c);
    {
        let poses: Vec<Arc<BufferGeometry>> = vec![Arc::new(train_geometry(
            4,
            Color::WHITE,
        ))];
        fleets.push(spawn_fleet(
            scene,
            "trains",
            &poses,
            None,
            &TRAIN_LIVERY,
            0.42,
            0.20,
            0.55,
            900.0,
            Gait::Drive,
            train_src,
            &mut triangles,
            &mut draws,
        ));
    }

    // --- Dogs. They are not their own population: each one is a copy of a
    // pedestrian's mover, pushed to one side of the pavement and set back a
    // stride, so it walks its owner's route at its owner's pace for nothing.
    let dog_coats: [u32; 6] = [0x3a2c22, 0x8a6a44, 0xc9b48c, 0x2b2b2e, 0x6d5136, 0xa8a49c];
    let mut dog_src: Vec<(usize, Mover)> = Vec::new();
    for (i, m) in fleets[people_fleet].movers.iter().enumerate() {
        if i % 11 != 3 {
            continue;
        }
        let mut d = *m;
        let side = if i % 2 == 0 { 0.62 } else { -0.62 };
        d.fixed += side;
        d.s -= 1.15 * d.dir;
        d.scale = PERSON_SCALE * (0.85 + (i % 5) as f32 * 0.06);
        d.tone = 0;
        dog_src.push((i % dog_coats.len(), d));
    }
    dog_src.sort_by_key(|(c, _)| *c);
    let dog_poses = vec![
        Arc::new(dog_geometry(1.0, Color::WHITE)),
        Arc::new(dog_geometry(-1.0, Color::WHITE)),
    ];
    if !dog_src.is_empty() {
        fleets.push(spawn_fleet(
            scene,
            "dogs",
            &dog_poses,
            None,
            &dog_coats,
            0.88,
            0.0,
            0.0,
            110.0,
            Gait::Walk,
            dog_src,
            &mut triangles,
            &mut draws,
        ));
    }

    // --- People who are not going anywhere.
    //
    // Placed once and never touched again: no mover, no lane, no pose cycle.
    // One instanced mesh per variant, so the whole idle population of a city
    // is a handful of draws whatever its size.
    let mut idle = Rng::new(params.seed ^ 0x51d1_e007);
    let seats = std::mem::take(&mut b.seats);
    let mut sitting: Vec<[Matrix4; 1]> = Vec::new();
    let mut sit_slots: Vec<Vec<Matrix4>> = vec![Vec::new(); IDLE_VARIANTS];
    let mut stand_slots: Vec<Vec<Matrix4>> = vec![Vec::new(); IDLE_VARIANTS];
    for (sx, sz, yaw) in &seats {
        // Not every bench is occupied, and a full one is two people.
        let n = match idle.f() {
            f if f < 0.46 => 0,
            f if f < 0.84 => 1,
            _ => 2,
        };
        for k in 0..n {
            let off = if n == 1 {
                idle.range(-0.35, 0.35)
            } else {
                (k as f32 - 0.5) * 0.86
            };
            // `add_bench` builds along the wall frame, so the seat runs across
            // the yaw the bench faces.
            let (rx, rz) = (yaw.cos(), yaw.sin());
            let m = Matrix4::compose(
                Vector3::new(sx + rx * off, KERB, sz + rz * off),
                Quaternion::from_axis_angle(Vector3::UP, -yaw + PI * 0.5),
                Vector3::ONE * PERSON_SCALE * idle.range(0.94, 1.06),
            );
            sit_slots[idle.below(IDLE_VARIANTS)].push(m);
        }
    }
    let _ = &mut sitting;
    // On duty outside the stations. Uniforms, not the ordinary palette: a
    // police officer in a random coat colour is a person standing outside a
    // police station, which is not the same thing.
    let posts = std::mem::take(&mut b.posts);
    let mut on_duty = 0usize;
    for (px, pz, which) in &posts {
        let (coat, trouser) = if *which == 0 {
            (Color::from_hex(0x1b2740), Color::from_hex(0x161d2c))
        } else {
            (Color::from_hex(0x2a2d30), Color::from_hex(0xd8b62a))
        };
        let skin = Color::from_hex(SKIN_PALETTE[idle.below(SKIN_PALETTE.len())]);
        let geom = Arc::new(standing_geometry(idle.below(2), coat, skin, trouser));
        let tris = geom.index.as_ref().map(|i| i.len() / 3).unwrap_or(0);
        let mut im = InstancedMesh::new(
            BufferGeometry::new(),
            Material::Standard(
                StandardMaterial::new(Color::WHITE)
                    .with_roughness(0.85)
                    .with_metalness(0.0),
            ),
            1,
        );
        im.geometry = geom;
        im.transforms = vec![Matrix4::compose(
            Vector3::new(*px, KERB, *pz),
            Quaternion::from_axis_angle(Vector3::UP, idle.range(0.0, TAU)),
            Vector3::ONE * PERSON_SCALE * idle.range(0.95, 1.05),
        )];
        let mut obj = Object3D::instanced_mesh(im);
        obj.name = format!("on-duty-{which}");
        scene.add(obj);
        triangles += tris;
        draws += 1;
        on_duty += 1;
    }

    // Standing about, on the hard ground where people actually loiter.
    for (gx, gz, gr) in &gathers {
        for _ in 0..idle.below(4) {
            let a = idle.range(0.0, TAU);
            let d = gr * idle.range(0.2, 1.0);
            let m = Matrix4::compose(
                Vector3::new(gx + d * a.cos(), KERB, gz + d * a.sin()),
                Quaternion::from_axis_angle(Vector3::UP, idle.range(0.0, TAU)),
                Vector3::ONE * PERSON_SCALE * idle.range(0.94, 1.06),
            );
            stand_slots[idle.below(IDLE_VARIANTS)].push(m);
        }
    }
    let mut idlers = on_duty;
    for (name, slots, seated) in [
        ("sitting", &sit_slots, true),
        ("standing", &stand_slots, false),
    ] {
        for (v, list) in slots.iter().enumerate() {
            if list.is_empty() {
                continue;
            }
            idlers += list.len();
            let coat = Color::from_hex(WALKER_COLORS[v % WALKER_COLORS.len()]);
            let skin = Color::from_hex(SKIN_PALETTE[v % SKIN_PALETTE.len()]);
            let trouser = scale_color(coat, 0.55);
            let geom = Arc::new(if seated {
                seated_geometry(v as f32 / (IDLE_VARIANTS - 1) as f32, coat, skin, trouser)
            } else {
                standing_geometry(v % 2, coat, skin, trouser)
            });
            let tris = geom.index.as_ref().map(|i| i.len() / 3).unwrap_or(0);
            let mut im = InstancedMesh::new(
                BufferGeometry::new(),
                Material::Standard(
                    StandardMaterial::new(Color::WHITE)
                        .with_roughness(0.85)
                        .with_metalness(0.0),
                ),
                list.len(),
            );
            im.geometry = geom;
            im.transforms = list.clone();
            let mut obj = Object3D::instanced_mesh(im);
            obj.name = format!("idle-{name}-{v}");
            scene.add(obj);
            triangles += tris * list.len();
            draws += 1;
        }
    }


    // --- Wildlife.
    //
    // Two populations that barely overlap. Pigeons, squirrels and butterflies
    // work daylight; foxes and bats work the dark. `Shift` decides which are
    // awake, and `apply_sky` flips them on the same clock as the street lamps.
    let mut swarms: Vec<Swarm> = Vec::new();
    let mut wild = Rng::new(params.seed ^ 0x5eed_b12d);
    let spot = |list: &mut Vec<Critter>,
                    wild: &mut Rng,
                    (cx, cz, cr): (f32, f32, f32),
                    y: f32,
                    n: usize,
                    r: (f32, f32),
                    rate: (f32, f32),
                    wander: (f32, f32),
                    scale: (f32, f32),
                    climb: f32| {
        for _ in 0..n {
            let a = wild.range(0.0, TAU);
            let d = cr * wild.range(0.0, 0.85);
            list.push(Critter {
                hx: cx + d * a.cos(),
                hz: cz + d * a.sin(),
                r: wild.range(r.0, r.1),
                y,
                phase: wild.f(),
                rate: wild.range(rate.0, rate.1) * if wild.chance(0.5) { 1.0 } else { -1.0 },
                wander: wild.range(wander.0, wander.1),
                seed: wild.next_u32(),
                scale: CRITTER_SCALE * wild.range(scale.0, scale.1),
                climb,
            });
        }
    };

    // Pigeons, on the hard ground of every plaza, square and playground. A
    // pigeon moves in fits and starts: a slow loop plus the peck cycle reads
    // as that where a smooth cruise does not.
    let mut pigeons: Vec<Critter> = Vec::new();
    for g in &gathers {
        let n = 4 + wild.below(9);
        spot(&mut pigeons, &mut wild, *g, KERB, n, (0.35, 1.4), (0.012, 0.045), (0.25, 0.6), (0.85, 1.15), 0.0);
    }

    // Squirrels, wherever there are trees to be between. Quicker and more
    // erratic than a pigeon, and far fewer of them.
    let mut squirrels: Vec<Critter> = Vec::new();
    for w in &park_walks {
        if !wild.chance(0.55) {
            continue;
        }
        let (cx, cz) = ((w.0 + w.2) * 0.5, (w.1 + w.3) * 0.5);
        let cr = ((w.2 - w.0).min(w.3 - w.1) * 0.45).max(1.0);
        let n = 1 + wild.below(4);
        spot(&mut squirrels, &mut wild, (cx, cz, cr), KERB, n, (0.6, 2.6), (0.05, 0.14), (0.4, 0.9), (0.8, 1.2), 0.0);
    }

    // Butterflies over the same green, low and slow, in daylight only.
    let mut butterflies: Vec<Critter> = Vec::new();
    for w in &park_walks {
        if !wild.chance(0.5) {
            continue;
        }
        let (cx, cz) = ((w.0 + w.2) * 0.5, (w.1 + w.3) * 0.5);
        let cr = ((w.2 - w.0).min(w.3 - w.1) * 0.45).max(1.0);
        let n = 3 + wild.below(7);
        spot(&mut butterflies, &mut wild, (cx, cz, cr), KERB + 0.9, n, (0.5, 2.2), (0.05, 0.16), (0.5, 1.0), (0.8, 1.3), 0.35);
    }

    // Waterfowl. Ducks on anything, swans only where there is room for them.
    let mut ducks: Vec<Critter> = Vec::new();
    let mut swans: Vec<Critter> = Vec::new();
    for (px, pz, pr) in &ponds {
        let n = 2 + wild.below(6);
        spot(&mut ducks, &mut wild, (*px, *pz, pr * 0.55), POND_Y, n, ((pr * 0.12).max(0.6), (pr * 0.34).max(0.9)), (0.008, 0.030), (0.15, 0.45), (0.9, 1.1), 0.012);
        if *pr > 7.0 && wild.chance(0.55) {
            let n = 1 + wild.below(3);
            spot(&mut swans, &mut wild, (*px, *pz, pr * 0.45), POND_Y - 0.02, n, ((pr * 0.14).max(0.9), (pr * 0.30).max(1.4)), (0.005, 0.016), (0.10, 0.30), (0.95, 1.15), 0.010);
        }
    }

    // Dogs off the lead, inside the runs. Faster and wider-ranging than
    // anything else on the ground here, because that is the point of a run.
    let mut park_dogs: Vec<Critter> = Vec::new();
    for r in &runs {
        let n = 2 + wild.below(4);
        spot(&mut park_dogs, &mut wild, *r, KERB, n, (1.5, r.2.max(2.0)), (0.06, 0.16), (0.35, 0.85), (0.85, 1.15), 0.0);
    }

    // Where the water is, for anything that lives over it.
    let water_x = match plan.water {
        Water::River => Some((rx0 + rx1) * 0.5),
        Water::Bay => Some((-e + shore) * 0.5),
        Water::Dry => None,
    };

    // --- Aircraft. Placed by hand rather than through `spot`, because the
    // scale rules for a thirty-metre airliner are not the scale rules for a
    // pigeon, and because a plane's loop has to be centred a long way off.
    let mut planes: Vec<Critter> = Vec::new();
    for _ in 0..(1 + wild.below(2)) {
        // A loop two kilometres across, centred so that the city sits on it:
        // over three hundred metres of town that arc reads as a straight line,
        // which is what an airliner does, and it costs nothing extra.
        let a = wild.range(0.0, TAU);
        let r = wild.range(1700.0, 2900.0);
        planes.push(Critter {
            hx: r * a.cos(),
            hz: r * a.sin(),
            r,
            y: wild.range(230.0, 430.0),
            // Start it somewhere near the half of the loop that crosses town.
            phase: (a + PI) / TAU + wild.range(-0.02, 0.02),
            rate: wild.range(0.0022, 0.0042) * if wild.chance(0.5) { 1.0 } else { -1.0 },
            wander: 0.0,
            seed: wild.next_u32(),
            scale: wild.range(0.9, 1.25),
            climb: wild.range(0.0, 6.0),
        });
    }

    // Helicopters orbit at working height, which is what a helicopter over a
    // city is usually doing.
    let mut helis: Vec<Critter> = Vec::new();
    for _ in 0..(1 + wild.below(2)) {
        helis.push(Critter {
            hx: wild.range(-extent * 0.6, extent * 0.6),
            hz: wild.range(-extent * 0.6, extent * 0.6),
            r: wild.range(110.0, 280.0),
            y: wild.range(95.0, 190.0),
            phase: wild.f(),
            rate: wild.range(0.020, 0.045) * if wild.chance(0.5) { 1.0 } else { -1.0 },
            wander: wild.range(0.05, 0.18),
            seed: wild.next_u32(),
            scale: wild.range(0.9, 1.15),
            climb: wild.range(1.0, 5.0),
        });
    }

    // Quadcopters, low and quick, over anywhere.
    let mut drones: Vec<Critter> = Vec::new();
    for _ in 0..(3 + wild.below(6)) {
        drones.push(Critter {
            hx: wild.range(-extent, extent),
            hz: wild.range(-extent, extent),
            r: wild.range(14.0, 55.0),
            y: wild.range(26.0, 72.0),
            phase: wild.f(),
            rate: wild.range(0.06, 0.20) * if wild.chance(0.5) { 1.0 } else { -1.0 },
            wander: wild.range(0.3, 0.8),
            seed: wild.next_u32(),
            scale: CRITTER_SCALE * wild.range(0.9, 1.4),
            climb: wild.range(1.5, 5.0),
        });
    }

    // Dragonflies over standing water, in daylight. They dart rather than
    // cruise: a fast rate on a very bent path, which is what separates one
    // from a butterfly at the same size.
    let mut dragonflies: Vec<Critter> = Vec::new();
    for (px, pz, pr) in &ponds {
        let n = 2 + wild.below(5);
        spot(&mut dragonflies, &mut wild, (*px, *pz, pr * 0.9), POND_Y + 0.55, n, (0.6, pr.max(1.2)), (0.10, 0.30), (0.5, 0.95), (0.9, 1.3), 0.22);
    }
    if let Some(wx) = water_x {
        for _ in 0..(6 + (extent / 40.0) as usize) {
            let z = wild.range(-extent, extent);
            let off = wild.range(-14.0, 14.0);
            let n = 1 + wild.below(3);
            spot(&mut dragonflies, &mut wild, (wx + off, z, 3.0), -RIVER_DEPTH + 0.9, n, (0.8, 3.5), (0.08, 0.26), (0.5, 0.95), (0.9, 1.3), 0.30);
        }
    }

    // Flies over the refuse. A knot of them going nowhere is the point, so the
    // loops are tiny and the rate is high.
    let mut flies: Vec<Critter> = Vec::new();
    for t in &trash {
        if !wild.chance(0.7) {
            continue;
        }
        let n = 5 + wild.below(9);
        spot(&mut flies, &mut wild, (t.0, t.1, 0.5), KERB + 0.75, n, (0.10, 0.45), (0.35, 0.95), (0.6, 1.0), (0.8, 1.3), 0.16);
    }

    // Rats, after dark, against the refuse. They do not wander: a rat by a
    // bin stays by that bin, which is what the very small loop radius is.
    let mut rats: Vec<Critter> = Vec::new();
    for t in &trash {
        if !wild.chance(0.55) {
            continue;
        }
        let n = 1 + wild.below(4);
        spot(&mut rats, &mut wild, (t.0, t.1, 0.9), KERB, n, (0.20, 0.75), (0.10, 0.30), (0.5, 1.0), (0.8, 1.25), 0.0);
    }

    // Foxes, after dark, along the edges of the green where it meets the
    // street. Solitary — a pack of them in a park is a different animal.
    let mut foxes: Vec<Critter> = Vec::new();
    for w in &park_walks {
        if !wild.chance(0.30) {
            continue;
        }
        let (cx, cz) = ((w.0 + w.2) * 0.5, (w.1 + w.3) * 0.5);
        let cr = ((w.2 - w.0).min(w.3 - w.1) * 0.42).max(1.5);
        spot(&mut foxes, &mut wild, (cx, cz, cr), KERB, 1, (2.0, 6.0), (0.02, 0.06), (0.3, 0.8), (0.9, 1.1), 0.0);
    }

    // Flocks in the air. Each flock is a shared centre with the birds spread
    // round it at their own radii and phases, which is enough for it to hold
    // together as a flock while no two birds trace the same path. Gulls get
    // the water, and fly higher and slower; bats come out over the streets
    // after dark and fly like nothing else here.
    let mut birds: Vec<Critter> = Vec::new();
    let mut gulls: Vec<Critter> = Vec::new();
    let mut bats: Vec<Critter> = Vec::new();
    for _ in 0..(3 + (extent / 90.0) as usize) {
        let fx = wild.range(-extent, extent);
        let fz = wild.range(-extent, extent);
        let fy = wild.range(26.0, 78.0);
        let fr = wild.range(14.0, 46.0);
        let rate = wild.range(0.020, 0.055) * if wild.chance(0.5) { 1.0 } else { -1.0 };
        for _ in 0..(5 + wild.below(12)) {
            birds.push(Critter {
                hx: fx + wild.range(-6.0, 6.0),
                hz: fz + wild.range(-6.0, 6.0),
                r: fr * wild.range(0.82, 1.18),
                y: fy + wild.range(-5.0, 5.0),
                phase: wild.range(0.0, 1.0),
                rate,
                wander: wild.range(0.05, 0.2),
                seed: wild.next_u32(),
                scale: CRITTER_SCALE * wild.range(0.8, 1.25),
                climb: wild.range(0.8, 2.6),
            });
        }
    }
    if let Some(wx) = water_x {
        for _ in 0..(2 + (extent / 140.0) as usize) {
            let fz = wild.range(-extent, extent);
            let fy = wild.range(14.0, 40.0);
            let fr = wild.range(18.0, 44.0);
            let rate = wild.range(0.012, 0.032) * if wild.chance(0.5) { 1.0 } else { -1.0 };
            for _ in 0..(4 + wild.below(9)) {
                gulls.push(Critter {
                    hx: wx + wild.range(-10.0, 10.0),
                    hz: fz + wild.range(-8.0, 8.0),
                    r: fr * wild.range(0.8, 1.2),
                    y: fy + wild.range(-4.0, 4.0),
                    phase: wild.range(0.0, 1.0),
                    rate,
                    wander: wild.range(0.08, 0.26),
                    seed: wild.next_u32(),
                    scale: CRITTER_SCALE * wild.range(0.95, 1.35),
                    climb: wild.range(1.2, 3.4),
                });
            }
        }
    }
    for lamp in &lamps {
        if !wild.chance(0.10) {
            continue;
        }
        for _ in 0..(1 + wild.below(3)) {
            bats.push(Critter {
                hx: lamp.x + wild.range(-4.0, 4.0),
                hz: lamp.z + wild.range(-4.0, 4.0),
                r: wild.range(2.5, 7.0),
                y: lamp.y + wild.range(-1.5, 3.0),
                phase: wild.range(0.0, 1.0),
                // Fast and wildly non-circular. A bat that cruises is a bird.
                rate: wild.range(0.28, 0.55) * if wild.chance(0.5) { 1.0 } else { -1.0 },
                wander: wild.range(0.55, 0.95),
                seed: wild.next_u32(),
                scale: CRITTER_SCALE * wild.range(0.8, 1.2),
                climb: wild.range(0.6, 1.6),
            });
        }
    }

    let wing_colours = [0xe8b83c, 0xd8643c, 0xf0f0e8, 0x6f7fc0, 0xc85a90];
    for (name, critters, kind, args, beat, lod, rough, shift) in [
        ("planes", planes, CritterKind::Plane, vec![0.0f32, 1.0], 1.4f32, 4000.0f32, 0.42f32, Shift::Always),
        ("helicopters", helis, CritterKind::Helicopter, vec![0.0, 0.33, 0.66], 13.0, 1600.0, 0.55, Shift::Always),
        ("drones", drones, CritterKind::Drone, vec![0.0, 0.33, 0.66], 22.0, 340.0, 0.62, Shift::Always),
        ("birds", birds, CritterKind::Bird, vec![-1.0f32, 0.0, 1.0, 0.0], 7.0f32, 520.0f32, 0.72f32, Shift::Day),
        ("gulls", gulls, CritterKind::Gull, vec![-1.0, 0.0, 1.0, 0.0], 5.0, 620.0, 0.70, Shift::Day),
        ("pigeons", pigeons, CritterKind::Pigeon, vec![0.0, 1.0], 0.8, 95.0, 0.80, Shift::Day),
        ("squirrels", squirrels, CritterKind::Squirrel, vec![0.0, 1.0], 1.6, 85.0, 0.86, Shift::Day),
        ("butterflies", butterflies, CritterKind::Butterfly, vec![0.0, 0.5, 1.0, 0.5], 9.0, 55.0, 0.90, Shift::Day),
        ("dragonflies", dragonflies, CritterKind::Dragonfly, vec![-1.0, 0.0, 1.0, 0.0], 14.0, 60.0, 0.72, Shift::Day),
        ("flies", flies, CritterKind::Fly, vec![-1.0, 1.0], 22.0, 32.0, 0.92, Shift::Day),
        ("ducks", ducks, CritterKind::Duck, vec![0.0, 1.0], 0.0, 110.0, 0.62, Shift::Always),
        ("swans", swans, CritterKind::Swan, vec![0.0, 1.0], 0.0, 150.0, 0.60, Shift::Always),
        ("park-dogs", park_dogs, CritterKind::ParkDog, vec![1.0, -1.0], 2.6, 130.0, 0.88, Shift::Day),
        ("rats", rats, CritterKind::Rat, vec![0.0, 1.0], 3.0, 60.0, 0.88, Shift::Night),
        ("foxes", foxes, CritterKind::Fox, vec![1.0, -1.0], 2.2, 130.0, 0.86, Shift::Night),
        ("bats", bats, CritterKind::Bat, vec![-1.0, 0.2, 1.0, 0.2], 12.0, 190.0, 0.88, Shift::Night),
    ] {
        if critters.is_empty() {
            continue;
        }
        let n = critters.len();
        let mut groups = Vec::new();
        for (pose, arg) in args.iter().enumerate() {
            let geom = Arc::new(match kind {
                CritterKind::Bird => bird_geometry(*arg),
                CritterKind::Gull => gull_geometry(*arg),
                CritterKind::Pigeon => pigeon_geometry(*arg > 0.5),
                CritterKind::Duck => duck_geometry(*arg > 0.5),
                CritterKind::Swan => swan_geometry(*arg > 0.5),
                CritterKind::Squirrel => squirrel_geometry(*arg > 0.5),
                CritterKind::Fox => fox_geometry(*arg),
                CritterKind::Bat => bat_geometry(*arg),
                CritterKind::ParkDog => dog_geometry(*arg, Color::from_hex(0x8a6a44)),
                CritterKind::Rat => rat_geometry(*arg > 0.5),
                CritterKind::Plane => plane_geometry(*arg > 0.5),
                CritterKind::Helicopter => heli_geometry(*arg),
                CritterKind::Drone => drone_geometry(*arg),
                CritterKind::Dragonfly => dragonfly_geometry(
                    *arg,
                    Color::from_hex(
                        [0x2f7fa8u32, 0x3f8f57, 0xa83f5a, 0x2f4a8a][pose % 4],
                    ),
                ),
                CritterKind::Fly => fly_geometry(*arg),
                CritterKind::Butterfly => butterfly_geometry(
                    *arg,
                    Color::from_hex(wing_colours[pose % wing_colours.len()]),
                ),
            });
            let tris = geom.index.as_ref().map(|i| i.len() / 3).unwrap_or(0);
            let mut im = InstancedMesh::new(
                BufferGeometry::new(),
                Material::Standard(
                    StandardMaterial::new(Color::WHITE)
                        .with_roughness(rough)
                        .with_metalness(0.0),
                ),
                n,
            );
            im.geometry = geom;
            let mut obj = Object3D::instanced_mesh(im);
            obj.name = format!("{name}-{pose}");
            groups.push((scene.add(obj), pose));
            triangles += tris * n;
            draws += 1;
        }
        swarms.push(Swarm {
            critters,
            groups,
            poses: args.len(),
            beat,
            lod,
            kind,
            shift,
            awake: true,
        });
    }

    // One fleet a hull. A handful of vessels on open water, so nothing to
    // gain by simplifying them at distance.
    for (i, kind) in BOAT_KINDS.into_iter().enumerate() {
        let src = std::mem::take(&mut river_traffic[i]);
        if src.is_empty() {
            continue;
        }
        fleets.push(spawn_fleet(
            scene,
            kind.label(),
            &[Arc::new(kind.geometry())],
            None,
            &BOAT_COLORS,
            0.62,
            0.10,
            0.25,
            f32::INFINITY,
            Gait::Float,
            src,
            &mut triangles,
            &mut draws,
        ));
    }
    let mover_triangles = triangles - static_triangles;
    triangles = static_triangles;
    let cars_len: usize = fleets[..4].iter().map(|f| f.movers.len()).sum();

    // --- The sky. The crate has a full Preetham model behind `SkyMaterial`,
    // rendered on its own pipeline — no cull, no depth write, pinned at the
    // far plane — so it costs one draw and needs no fog or tone mapping of its
    // own. A flat clear colour was the weakest thing in every frame.
    let dome = {
        let mut obj = Object3D::mesh(Mesh::new(
            // The dome has to enclose everything the camera can see, and at
            // 8 half-widths it did not: the ground runs to 22 and the far
            // plane to 18, so the ground plane *cut through the sphere*. The
            // intersection is a circle at 8 half-widths, and that circle is
            // the faint pale line that ran diagonally up out of the horizon in
            // every daylight shot — ground on one side of it, dome on the
            // other, two nearly-equal whites that do not quite match.
            //
            // At 16 it sits inside the far plane and far outside the fog, so
            // wherever the seam still falls it falls in total haze. The night
            // shots never showed the line because the dome retires after dusk,
            // which is what made it look like a daylight-only artefact.
            SphereGeometry::new(extent * 16.0, 48, 24),
            Material::Sky(SkyMaterial {
                sun_position: Vector3::new(0.0, 1.0, 0.0),
                turbidity: 3.4,
                rayleigh: 2.1,
                mie_coefficient: 0.005,
                mie_directional_g: 0.78,
            }),
        ));
        obj.name = "sky".into();
        hide_check(&mut obj);
        draws += 1;
        scene.add(obj)
    };

    // --- The open water. Same material as the still pools; the difference
    // is that this one moves.
    let water = water_surface.map(|(geometry, vertices)| {
        let geometry = Arc::new(geometry);
        let mut obj = Object3D::mesh(Mesh::from_arc(
            geometry.clone(),
            Arc::new(Material::Physical(
                PhysicalMaterial::new(Color::WHITE)
                    .with_roughness(0.11)
                    .with_metalness(0.0)
                    .with_clearcoat(0.55, 0.05),
            )),
        ));
        obj.name = "open-water".into();
        obj.receive_shadow = true;
        hide_check(&mut obj);
        triangles += geometry.index.as_ref().map(|i| i.len() / 3).unwrap_or(0);
        draws += 1;
        WaterSurface {
            id: scene.add(obj),
            geometry,
            vertices,
            level: -RIVER_DEPTH,
        }
    });

    // --- Lane index. A vehicle never changes lane, so this is built once
    // and the per-frame traffic pass just sorts each lane by position. Lanes
    // deliberately span fleets: a car has to queue behind a bus.
    // Roads that carry traffic over the water.
    let bridges: Vec<(f32, f32)> = match plan.water {
        Water::River => (0..az.len())
            .step_by(2)
            .filter(|k| is_bridge(*k))
            .map(|k| (az.center(k), az.half(k)))
            .collect(),
        _ => Vec::new(),
    };

    // Where a vehicle has to stop: the signalled crossings on each axis.
    // Signals sit where two avenues meet, so a car running along Z stops at
    // the avenue roads that run along X, and the other way round.
    let stops_along_z: Vec<(f32, f32)> = (0..az.len())
        .step_by(2)
        .filter(|k| is_avenue(&plan, k / 2))
        .map(|k| (az.center(k), az.half(k)))
        .collect();
    let stops_along_x: Vec<(f32, f32)> = (0..ax.len())
        .step_by(2)
        .filter(|a| is_avenue(&plan, a / 2) && !drowned(*a))
        .map(|a| (ax.center(a), ax.half(a)))
        .collect();

    // Every crossroads, as a box. A vehicle inside one occupies it; a vehicle
    // approaching one that is occupied by cross traffic has to give way.
    // Without this, two streams simply drive through each other.
    let mut junctions: Vec<Junction> = Vec::new();
    for a in (0..ax.len()).step_by(2) {
        if drowned(a) {
            continue;
        }
        for k in (0..az.len()).step_by(2) {
            // A street that stops at the water gives a turning vehicle the
            // bank this junction is on, not the whole width.
            let (x_lo, x_hi) = match plan.water {
                Water::Dry => (cx0, cx1),
                Water::Bay => (shore + 3.0, cx1),
                Water::River if is_bridge(k) => (cx0, cx1),
                Water::River => {
                    if ax.center(a) < rx0 {
                        (cx0, rx0)
                    } else {
                        (rx1, cx1)
                    }
                }
            };
            junctions.push(Junction {
                cx: ax.center(a),
                cz: az.center(k),
                hx: ax.half(a) + 1.0,
                hz: az.half(k) + 1.0,
                // Signalled crossings are governed by the lights instead.
                signalled: is_avenue(&plan, a / 2) && is_avenue(&plan, k / 2),
                road_z: a,
                road_x: k,
                avenue_z: is_avenue(&plan, a / 2),
                avenue_x: is_avenue(&plan, k / 2),
                x_lo,
                x_hi,
                z_lo: cz0,
                z_hi: cz1,
            });
        }
    }

    // --- Car lamps. One instanced mesh per end covering every car whatever
    // its paint, so night traffic costs two draws rather than one a colour.
    let mut car_lights = Vec::new();
    let mut car_lamps: Vec<(ObjectId, Lamp, f32, u32)> = Vec::new();
    // Every road vehicle, not just the cars: vans, buses and lorries drove
    // around at night with no lights on them at all.
    let lamp_count: usize = fleets
        .iter()
        .take(ROAD_FLEETS)
        .map(|f| f.movers.len())
        .sum();
    // The last field restricts a lamp to one fleet: only the marked cars
    // carry beacons, and the beacon fleet is the last road fleet spawned.
    // Police, ambulance and fire engine: fleets 8, 11 and 12.
    let emergency = (1u32 << 8) | (1 << 11) | (1 << 12);
    for (part, name, tint, glow, only) in [
        (Lamp::Head, "car-headlights", 0xfff3d6u32, 3.4f32, 0u32),
        (Lamp::Tail, "car-taillights", 0xff2a12, 1.7, 0),
        (Lamp::Beam, "car-beams", 0xffe9b8, 0.85, 0),
        (Lamp::Brake, "car-brakes", 0xff1c0c, 4.2, 0),
        (Lamp::Indicate, "car-indicators", 0xffa617, 4.6, 0),
        (Lamp::BeaconA, "emergency-beacon-a", 0x2a6cff, 6.0, emergency),
        (Lamp::BeaconB, "emergency-beacon-b", 0xff2418, 6.0, emergency),
    ] {
        let geom = Arc::new(car_lamp_geometry(part));
        let tris = geom.index.as_ref().map(|i| i.len() / 3).unwrap_or(0);
        let mut im = InstancedMesh::new(
            BufferGeometry::new(),
            Material::Standard(
                StandardMaterial::new(Color::from_hex(tint))
                    .with_roughness(0.35)
                    .with_emissive(Color::from_hex(tint), 0.0),
            ),
            lamp_count,
        );
        im.geometry = geom;
        let mut obj = Object3D::instanced_mesh(im);
        obj.name = name.into();
        // Lamps are off until `apply_sky` says otherwise — except the ones
        // that signal intent, which are as useful in daylight as at night.
        obj.visible = matches!(part, Lamp::Brake | Lamp::Indicate | Lamp::BeaconA | Lamp::BeaconB);
        let id = scene.add(obj);
        car_lights.push((id, glow));
        car_lamps.push((id, part, glow, only));
        triangles += tris * lamp_count;
        draws += 1;
    }

    // --- Signal lenses. Unlit, always on, and exactly one phase visible.
    let mut signal_phases = [None, None];
    let signal_builders = std::mem::take(&mut b.signal);
    for (i, mb) in signal_builders.into_iter().enumerate() {
        triangles += mb.tri_count();
        if let Some(id) = add_batch(
            scene,
            &format!("signals-{i}"),
            mb,
            Material::Basic(BasicMaterial {
                color: Color::WHITE,
                ..Default::default()
            }),
            false,
            false,
        ) {
            signal_phases[i] = Some(id);
            draws += 1;
        }
    }

    // --- A camera position that is guaranteed to be standing on tarmac:
    // the avenue nearest the middle, looking downtown along it.
    let vista_x = (0..ax.len())
        .step_by(2)
        .filter(|a| is_avenue(&plan, a / 2) && !drowned(*a))
        .map(|a| ax.center(a))
        .min_by(|p, q| p.abs().total_cmp(&q.abs()))
        .unwrap_or(0.0);
    let vista = (
        Vector3::new(vista_x, 6.5, cz1 - 8.0),
        Vector3::new(vista_x, 30.0, -extent * 0.30),
    );

    // --- Lights. The sun must be the first directional light in the scene:
    // the renderer only lets `dir_lights[0]` cast the shadow map.
    let mut sun_obj = Object3D::light(DirectionalLight::new(Color::WHITE, 3.0));
    sun_obj.name = "sun".into();
    if let ObjectKind::Light(Light::Directional(l)) = &mut sun_obj.kind {
        l.cast_shadow = true;
        l.shadow = ShadowSettings {
            map_size: 4096,
            // The shadow camera is a cube on the origin, so it has to be at
            // least the city's half-width or the far corners lose their
            // shadows entirely.
            camera_size: extent * 1.06,
            camera_near: 0.5,
            camera_far: extent * 6.0,
            bias: 0.0012,
            normal_bias: 0.05,
        };
    }
    let sun = scene.add(sun_obj);

    let mut moon_obj = Object3D::light(DirectionalLight::new(Color::from_hex(0x9db4e8), 0.3));
    moon_obj.name = "moon".into();
    moon_obj.position = Vector3::new(-0.42, 0.80, 0.44).normalize() * 400.0;
    let moon = scene.add(moon_obj);

    // Real lights for the four street lamps nearest the camera. The glow
    // discs fake every lamp in the city at once and cannot cast anything;
    // these four are the ones close enough for it to matter, and the first of
    // them casts a shadow — the renderer allows exactly one spot caster.
    let mut spots = Vec::new();
    for i in 0..MAX_SPOTS {
        let mut light = SpotLight::new(Color::from_hex(0xffd9a0), 0.0);
        light.direction = Vector3::new(0.0, -1.0, 0.0);
        light.distance = 22.0;
        light.decay = 2.0;
        light.angle = 0.62;
        light.penumbra = 0.55;
        light.cast_shadow = i == 0;
        light.shadow = ShadowSettings {
            map_size: 2048,
            camera_size: 8.0,
            camera_near: 0.4,
            camera_far: 26.0,
            bias: 0.0016,
            normal_bias: 0.03,
        };
        let mut obj = Object3D::light(light);
        obj.name = format!("lamp-{i}");
        spots.push(scene.add(obj));
    }

    let mut hemi_obj = Object3D::light(HemisphereLight::new(
        Color::from_hex(0x9cc3ea),
        Color::from_hex(0x4b4438),
        0.7,
    ));
    hemi_obj.name = "sky".into();
    let hemi = scene.add(hemi_obj);


    City {
        stats: CityStats {
            mover_triangles,
            blocks,
            buildings,
            civic: 4 - civic_wanted.len(),
            works,
            suburbs,
            triangles,
            draws,
            cars: cars_len,
            bake,
            idlers,
            footprints: (b_claims, b_refusals),
            walkers: walkers_len,
            extent,
        },
        windows,
        night_only,
        beacons,
        neon,
        shafts,
        clouds,
        plumes,
        night_sky,
        fleets,
        car_fleet,
        people_fleet,
        walker_skin,
        lane_scratch: Vec::new(),
        stops_along_x,
        stops_along_z,
        junctions,
        busy: Vec::new(),
        water,
        water_span: match plan.water {
            Water::River => Some((rx0, rx1)),
            Water::Bay => Some((-e, shore)),
            Water::Dry => None,
        },
        bridges,
        dome,
        lamps,
        spots,
        car_lights,
        car_lamps,
        signal_phases,
        swarms,
        vista,
        sun,
        moon,
        hemi,
        env_slot: None,
        activity: 1.0,
    }
}

/// How many seated and standing variants the idle population is drawn from.
/// Each is one instanced mesh and therefore one draw, so this is a budget
/// rather than a taste: four is enough that a full bench is not two clones.
pub(crate) const IDLE_VARIANTS: usize = 4;

/// Slices of a day the environment cube is rebuilt on. 96 is about every 15
/// minutes of city time — below the threshold where the step is visible.
pub(crate) const ENV_SLICES: f32 = 96.0;

impl City {
    /// Point the lights, tint the air, and switch the windows on. Returns the
    /// sky it applied so the caller can reuse the numbers (background, fog).
    /// Move the four spot lights onto the street lamps nearest `eye`.
    ///
    /// Nearest-first rather than a fixed set: the four that matter are the
    /// ones you are standing among, and which those are changes as the camera
    /// moves. Off entirely in daylight, where they would only wash the road.
    pub(crate) fn place_lights(&self, scene: &mut Scene, eye: Vector3, lights: f32) {
        let mut best: Vec<(f32, usize)> = Vec::with_capacity(MAX_SPOTS + 1);
        if lights > 0.03 {
            for (i, p) in self.lamps.iter().enumerate() {
                let d = (p.x - eye.x).powi(2) + (p.z - eye.z).powi(2);
                let at = best.partition_point(|(bd, _)| *bd < d);
                if at < MAX_SPOTS {
                    best.insert(at, (d, i));
                    best.truncate(MAX_SPOTS);
                }
            }
        }
        for (slot, id) in self.spots.iter().enumerate() {
            let Some(obj) = scene.get_mut(*id) else {
                continue;
            };
            let on = best.get(slot).copied();
            if let Some((_, lamp)) = on {
                obj.position = self.lamps[lamp];
            }
            if let ObjectKind::Light(Light::Spot(l)) = &mut obj.kind {
                // Inverse-square from six metres up, so the number is large
                // and the pool on the ground is not.
                l.intensity = if on.is_some() { 62.0 * lights } else { 0.0 };
                l.cast_shadow = slot == 0 && on.is_some();
            }
        }
    }

    pub(crate) fn apply_sky(&mut self, scene: &mut Scene, t: f32) -> Sky {
        let sky = sky_at(t);

        // Two peaks: an hour or so after sunrise and again before sunset, with
        // a long trough overnight. A city at 3 a.m. is not a city at 9 a.m.
        // with the lights turned off.
        let bump = |c: f32, w: f32| (-(((t.rem_euclid(1.0) - c) / w).powi(2))).exp();
        let rush = bump(0.085, 0.045) + bump(0.405, 0.052);
        self.activity = (0.20 + 0.55 * sky.day + 0.30 * rush).clamp(0.16, 1.0);

        let slot = (t.rem_euclid(1.0) * ENV_SLICES).floor() as i32;
        if self.env_slot != Some(slot) {
            self.env_slot = Some(slot);
            scene.environment = Some(sky_environment(&sky, 64));
        }

        if let Some(obj) = scene.get_mut(self.sun) {
            obj.position = sky.sun_dir * 400.0;
            if let ObjectKind::Light(Light::Directional(l)) = &mut obj.kind {
                l.color = sky.sun_color;
                l.intensity = sky.sun_intensity;
                // Below the horizon there is nothing to cast, and a shadow
                // camera looking up through the ground is worse than none.
                l.cast_shadow = sky.elevation > 0.04;
                // The shadow camera is a box on the origin sized to the city,
                // which is right at noon and useless at dawn: a ninety-metre
                // tower at ten degrees throws its shadow half a kilometre, and
                // ground that far out is not in the map — so sunrise and
                // sunset had no cast shadows at all, the one time of day they
                // matter most. Grow the box as the sun drops. Texels get
                // coarser with it, which at these angles is the right trade:
                // a soft half-kilometre shadow beats none.
                let e = sky.elevation.max(0.055);
                let reach = (0.16 / e - 0.16).clamp(0.0, 4.5);
                let s = self.stats.extent * (1.06 + reach);
                l.shadow.camera_size = s;
                l.shadow.camera_far = s * 6.0;
                // Bias in world units has to grow with the texel it hides.
                l.shadow.bias = 0.0012 * (1.0 + reach * 1.6);
            }
        }
        if let Some(obj) = scene.get_mut(self.moon) {
            if let ObjectKind::Light(Light::Directional(l)) = &mut obj.kind {
                l.intensity = sky.moon_intensity;
            }
        }
        if let Some(obj) = scene.get_mut(self.hemi) {
            if let ObjectKind::Light(Light::Hemisphere(l)) = &mut obj.kind {
                l.sky_color = sky.hemi_sky;
                l.ground_color = sky.hemi_ground;
                l.intensity = sky.hemi_intensity;
            }
        }

        for (id, gain) in &self.windows {
            if let Some(obj) = scene.get_mut(*id) {
                if let ObjectKind::Mesh(m) = &mut obj.kind {
                    // Both arms. The facades are `Physical` — they have a
                    // clearcoat lobe, which is what a curtain wall is — and
                    // this matched `Standard` only, so the intensity it was
                    // setting went nowhere and every window in the city was
                    // lit by whatever the default happened to be.
                    match Arc::make_mut(&mut m.material) {
                        Material::Standard(s) => s.emissive_intensity = sky.lights * gain,
                        Material::Physical(p) => p.emissive_intensity = sky.lights * gain,
                        _ => {}
                    }
                }
            }
        }
        for (id, glow) in &self.car_lights {
            let signal = self
                .car_lamps
                .iter()
                .any(|(m, p, _, _)| {
                    m == id
                        && matches!(
                            p,
                            Lamp::Brake | Lamp::Indicate | Lamp::BeaconA | Lamp::BeaconB
                        )
                });
            if let Some(obj) = scene.get_mut(*id) {
                // A brake light and an indicator are visible in broad daylight
                // — that is what they are for. Head and tail lamps and the
                // beam on the road are not.
                obj.visible = signal || sky.lights > 0.05;
                if let ObjectKind::InstancedMesh(im) = &mut obj.kind {
                    if let Material::Standard(sm) = Arc::make_mut(&mut im.material) {
                        sm.emissive_intensity = if signal {
                            // Bright enough to read against a sunlit road, and
                            // brighter still once the light goes.
                            glow * mix(0.42, 1.0, sky.lights)
                        } else {
                            sky.lights * glow
                        };
                    }
                }
            }
        }
        // Who is about. Foxes and bats come out as the lamps do; pigeons,
        // squirrels and butterflies go wherever it is they go.
        for i in 0..self.swarms.len() {
            let awake = self.swarms[i].shift.awake(sky.lights);
            self.swarms[i].awake = awake;
            for (id, _) in self.swarms[i].groups.clone() {
                if let Some(obj) = scene.get_mut(id) {
                    obj.visible = awake;
                }
            }
        }

        if let Some(id) = self.shafts {
            if let Some(obj) = scene.get_mut(id) {
                obj.visible = sky.lights > 0.05;
                if let ObjectKind::Mesh(m) = &mut obj.kind {
                    if let Material::Basic(bm) = Arc::make_mut(&mut m.material) {
                        bm.color = scale_color(Color::WHITE, sky.lights);
                        // Barely there at dusk, and never enough to hide the
                        // road: a visible cone is a hint, not a fog bank.
                        bm.opacity = 0.058 * sky.lights;
                    }
                }
            }
        }
        for (id, _) in &self.neon {
            if let Some(obj) = scene.get_mut(*id) {
                if let ObjectKind::Mesh(m) = &mut obj.kind {
                    if let Material::Basic(bm) = Arc::make_mut(&mut m.material) {
                        // Full brightness: neon is the brightest thing on the
                        // street and the hue is carried per vertex.
                        bm.color = scale_color(Color::WHITE, sky.lights);
                    }
                }
            }
        }
        for id in &self.night_only {
            if let Some(obj) = scene.get_mut(*id) {
                obj.visible = sky.lights > 0.02;
                if let ObjectKind::Mesh(m) = &mut obj.kind {
                    if let Material::Basic(bm) = Arc::make_mut(&mut m.material) {
                        // White: the hue is baked per vertex, so lamp orange
                        // and shopfront warm-white share one material. Pulled
                        // down since the four nearest lamps got real spot
                        // lights — where both apply they were doubling up.
                        bm.color = scale_color(Color::WHITE, sky.lights * 0.62);
                    }
                }
            }
        }

        // The dome is an analytic daylight model and has no night in it: after
        // dusk it renders a flat grey wash, brighter than any star drawn in
        // front of it, which is why there was no night sky at all. Below the
        // horizon it is retired and the background — a dark navy with the
        // city's own glow mixed into it — is what shows.
        let dome_on = sky.elevation > -0.16;
        if let Some(obj) = scene.get_mut(self.dome) {
            obj.visible = dome_on;
        }
        if let Some(id) = self.clouds {
            if let Some(obj) = scene.get_mut(id) {
                if let ObjectKind::Mesh(m) = &mut obj.kind {
                    if let Material::Standard(sm) = Arc::make_mut(&mut m.material) {
                        // The underside of a cloud faces the ground, so the
                        // only light reaching it in this scene is the
                        // hemisphere's ground colour — which is grass. Left
                        // alone the shadow side of every cloud came out olive.
                        // Real undersides are lit by the rest of the sky, so
                        // that is what this puts back: a dim wash of the
                        // horizon colour, which also carries them through
                        // sunset and lets them go properly dark at night.
                        sm.emissive = scale_color(sky.hemi_sky, 0.30 + 0.26 * sky.day);
                        // Overcast whites blow out under a full sun; pulling
                        // the albedo down keeps the tops inside the range the
                        // tone map can still show shape in.
                        sm.color = scale_color(Color::from_hex(0xf4f6fa), 0.80);
                    }
                }
            }
        }
        if let Some(id) = self.plumes {
            if let Some(obj) = scene.get_mut(id) {
                if let ObjectKind::Mesh(m) = &mut obj.kind {
                    if let Material::Standard(sm) = Arc::make_mut(&mut m.material) {
                        // Exactly the clouds' treatment, and for exactly the
                        // same reason: without it the shadow side of a column
                        // of steam is lit only by grass.
                        sm.emissive = scale_color(sky.hemi_sky, 0.34 + 0.30 * sky.day);
                        sm.color = scale_color(Color::from_hex(0xf2f4f6), 0.86);
                    }
                }
            }
        }
        if let Some(id) = self.night_sky {
            if let Some(obj) = scene.get_mut(id) {
                // Fade in as the dome retires, and stay at full brightness
                // after: stars do not get dimmer as the night goes on.
                let show = smoothstep(0.02, -0.16, sky.elevation);
                obj.visible = show > 0.02;
                if let ObjectKind::Mesh(m) = &mut obj.kind {
                    if let Material::Basic(bm) = Arc::make_mut(&mut m.material) {
                        bm.color = scale_color(Color::WHITE, show);
                    }
                }
            }
        }
        if let Some(obj) = scene.get_mut(self.dome) {
            if let ObjectKind::Mesh(m) = &mut obj.kind {
                if let Material::Sky(sm) = Arc::make_mut(&mut m.material) {
                    // The model wants a position, not a direction, and reads
                    // its own height off it for the earth-shadow term.
                    sm.sun_position = sky.sun_dir * 450_000.0;
                    // Thicker air at the horizon: more haze near sunrise and
                    // sunset, cleaner overhead.
                    // Low on purpose. This shader is a port of three.js's Sky,
                    // which expects an exposure of about 0.5 and ACES after
                    // it; here it writes straight to the framebuffer, so the
                    // published defaults (turbidity 10, rayleigh 3) clip to
                    // white. These are the values that still read as sky.
                    sm.turbidity = mix(2.4, 1.7, sky.day);
                    // Rayleigh drives the horizon, and the horizon is where a
                    // wide shot spends most of its sky. Measured looking along
                    // the ground at noon: 0.33 clipped 86% of the sky to flat
                    // white, 0.10 clipped 14%, 0.05 none. The old values were
                    // tuned on a steep aerial where the camera barely sees the
                    // horizon at all, and they are hopeless the moment it does.
                    sm.rayleigh = mix(0.30, 0.12, sky.day);
                    // Mie was left at the library default of 0.005, and the
                    // dome clipped to flat white over a sixth of the sky at
                    // noon and a tenth at sunrise. Raising it is the fix and
                    // the direction is worth recording, because it is the
                    // opposite of the intuitive one: more Mie means more
                    // extinction along the view ray, so the sky gets *darker*,
                    // not hazier-brighter. Swept: 0.005 -> 10.0%/16.4% clipped
                    // at sunrise/noon, 0.030 -> 3.0%/0.0%. A little more haze
                    // low in the sky, where a city has more of it anyway.
                    sm.mie_coefficient = mix(0.036, 0.026, sky.day);
                    // Forward scattering, tightening the aureole around the
                    // sun rather than smearing it across the whole dome.
                    sm.mie_directional_g = 0.82;
                }
            }
        }

        // Still set, though the dome covers it: it is what shows if the sky
        // mesh is ever culled, and it is the colour the fog fades to.
        scene.background = shaded_like_a_fragment(sky.background);
        // The haze has one job: hide where the terrain plane stops. It has to
        // be *complete* by then, and it was not — the ground runs to
        // `OUTSKIRTS_REACH` and the fog reached full strength a fifth further
        // out than that, so the far edge arrived at about three-quarters
        // fogged and left a hard band along the skyline at every hour. Tie
        // both distances to the ground's own extent instead of guessing.
        scene.fog = threers::scene::FogParams {
            color: sky.fog_color,
            near: self.stats.extent * FOG_NEAR,
            far: self.stats.extent * FOG_FAR,
            density: 0.0,
            mode: 1,
        };
        sky
    }

    /// Where one mover is, and which way it faces. For aiming a camera at a
    /// model, or riding along with it, rather than hunting for a good vantage
    /// by hand — which is how you end up inside a building.
    #[allow(dead_code)]
    pub(crate) fn mover_pose(&self, gait: Gait, index: usize) -> Option<(Vector3, Vector3)> {
        let fleet = self.fleets.iter().find(|f| f.gait == gait)?;
        let m = fleet.movers.get(index % fleet.movers.len().max(1))?;
        let pos = if m.along_x {
            Vector3::new(m.s, m.base_y, m.fixed)
        } else {
            Vector3::new(m.fixed, m.base_y, m.s)
        };
        let facing = if m.along_x {
            Vector3::new(m.dir, 0.0, 0.0)
        } else {
            Vector3::new(0.0, 0.0, m.dir)
        };
        Some((pos, facing))
    }

    /// The two things that run on their own clock rather than the sun's:
    /// aircraft warning lights (on for about a third of every 1.6 s) and the
    /// traffic signals (one phase per 8 s).
    pub(crate) fn animate_lights(&self, scene: &mut Scene, time: f32) {
        let beacon_on = (time * 0.625).fract() < 0.34;
        for id in &self.beacons {
            if let Some(obj) = scene.get_mut(*id) {
                obj.visible = beacon_on;
            }
        }
        let phase = signal_phase(time);
        // Blinking signage rides the same clock as the traffic signals. One
        // shared phase means every blinking sign in the city blinks together,
        // which is wrong; the per-sign offset is baked into which batch a sign
        // went into and into a slow flicker on the colour below.
        for (id, blinks) in &self.neon {
            if let Some(obj) = scene.get_mut(*id) {
                obj.visible = !*blinks || phase == 0;
            }
        }
        for (i, id) in self.signal_phases.iter().enumerate() {
            if let Some(id) = id {
                if let Some(obj) = scene.get_mut(*id) {
                    obj.visible = i == phase;
                }
            }
        }
    }

    /// Mark which crossroads have a vehicle *crossing* them, and on which
    /// axis. One pass over the traffic; the give-way rule below reads it.
    ///
    /// Only moving vehicles count. Marking stationary ones deadlocks the whole
    /// network within a minute: a car queued at a red light with its tail in
    /// the junction behind it blocks that cross street permanently, the cross
    /// street backs up into ITS junction, and it spreads. Measured, it took
    /// 234 vehicles from a mean 8.4 m/s to 2.8 and still falling.
    pub(crate) fn mark_junctions(&mut self) {
        let mut busy = std::mem::take(&mut self.busy);
        busy.clear();
        busy.resize(self.junctions.len(), 0);
        for fleet in self.fleets.iter().take(ROAD_FLEETS) {
            for m in &fleet.movers {
                if m.speed < 1.2 {
                    continue;
                }
                let (x, z) = if m.along_x {
                    (m.s, m.fixed)
                } else {
                    (m.fixed, m.s)
                };
                for (i, j) in self.junctions.iter().enumerate() {
                    if (x - j.cx).abs() < j.hx && (z - j.cz).abs() < j.hz {
                        busy[i] |= if m.along_x { 2 } else { 1 };
                        break;
                    }
                }
            }
        }
        self.busy = busy;
    }

    /// Hold station behind whatever is in front, stop at a red light, and give
    /// way to cross traffic at an unsignalled crossroads.
    ///
    /// Lanes span fleets, so a car queues behind a bus. The grouping is built
    /// here rather than precomputed: a vehicle that turns leaves its lane, and
    /// a stale index would have it queueing behind traffic on a road it is no
    /// longer on. One sort a frame gives both the grouping and the order
    /// within each lane.
    pub(crate) fn follow_pass(&mut self, vehicles: bool, phase: usize, dt: f32) {
        // A driver at a full stop leaves a couple of metres; a pedestrian
        // leaves half a metre, brakes harder and accelerates slower.
        let (follow, clear, accel, brake) = if vehicles {
            (26.0f32, 2.2f32, 3.0f32, 9.0f32)
        } else {
            (2.8, 0.42, 1.8, 3.5)
        };
        /// How far ahead a driver reads the signal.
        const LOOK: f32 = 36.0;

        let mut scratch = std::mem::take(&mut self.lane_scratch);
        scratch.clear();
        // Every road fleet, derived rather than listed: this was a literal
        // `[0, 1, 2, 3]` and adding five vehicle types silently left taxis,
        // pickups, sports cars and police cars driving through red lights and
        // through each other.
        let road: Vec<usize> = (0..ROAD_FLEETS).collect();
        let fleets: &[usize] = if vehicles {
            &road
        } else {
            std::slice::from_ref(&self.people_fleet)
        };
        for fi in fleets {
            for (mi, m) in self.fleets[*fi].movers.iter().enumerate() {
                if m.lane != u16::MAX {
                    scratch.push((m.lane, m.s, *fi as u32, mi as u32));
                }
            }
        }
        scratch.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));

        let mut lo = 0;
        while lo < scratch.len() {
            let lane = scratch[lo].0;
            let mut hi = lo;
            while hi < scratch.len() && scratch[hi].0 == lane {
                hi += 1;
            }
            let n = hi - lo;
            for oi in lo..hi {
                let (_, s_self, fi, mi) = scratch[oi];
                let m = self.fleets[fi as usize].movers[mi as usize];
                let span = m.s_hi - m.s_lo;
                let mut target = m.cruise;

                if n > 1 {
                    // The run is sorted by position, so the one ahead is the
                    // next along in the direction of travel, wrapping at the
                    // end of the lane.
                    let ai = if m.dir > 0.0 {
                        lo + (oi - lo + 1) % n
                    } else {
                        lo + (oi - lo + n - 1) % n
                    };
                    let (_, s_ahead, afi, ami) = scratch[ai];
                    let ahead_len = self.fleets[afi as usize].movers[ami as usize].length;
                    let raw = (s_ahead - s_self) * m.dir;
                    let along = if raw < 0.0 { raw + span } else { raw };
                    let gap = along - (m.length + ahead_len) * 0.5 - clear;
                    target = target.min(m.cruise * (gap / follow).clamp(0.0, 1.0).powf(0.7));
                }

                if vehicles && m.signalled {
                    // Arms along Z hold phase 0, so that is when traffic
                    // running along Z has green.
                    let red = if m.along_x { phase != 1 } else { phase != 0 };
                    if red {
                        let stops = if m.along_x {
                            &self.stops_along_x
                        } else {
                            &self.stops_along_z
                        };
                        let mut nearest = f32::MAX;
                        for (center, half) in stops.iter() {
                            let line = center - m.dir * (half + 1.6);
                            let mut ahead = (line - s_self) * m.dir;
                            if ahead < -span * 0.5 {
                                ahead += span;
                            }
                            if (0.0..nearest).contains(&ahead) {
                                nearest = ahead;
                            }
                        }
                        if nearest < LOOK {
                            target =
                                target.min(m.cruise * ((nearest - 1.0) / 14.0).clamp(0.0, 1.0));
                        }
                    }
                }

                if vehicles {
                    // Give way. Anything approaching an unsignalled crossroads
                    // that cross traffic is already inside stops short of it.
                    let mine = if m.along_x { 2u8 } else { 1u8 };
                    let mut nearest = f32::MAX;
                    for (i, j) in self.junctions.iter().enumerate() {
                        if j.signalled || self.busy[i] & !mine & 3 == 0 {
                            continue;
                        }
                        let (center, half) = if m.along_x {
                            (j.cx, j.hx)
                        } else {
                            (j.cz, j.hz)
                        };
                        let lane_hit = if m.along_x {
                            (m.fixed - j.cz).abs() < j.hz
                        } else {
                            (m.fixed - j.cx).abs() < j.hx
                        };
                        if !lane_hit {
                            continue;
                        }
                        let line = center - m.dir * (half + 1.2);
                        let mut ahead = (line - s_self) * m.dir;
                        if ahead < -span * 0.5 {
                            ahead += span;
                        }
                        // Already inside: carry on rather than stop in it.
                        if ahead < -0.5 {
                            continue;
                        }
                        if (0.0..nearest).contains(&ahead) {
                            nearest = ahead;
                        }
                    }
                    // Short look-ahead on purpose: hesitating fifteen metres
                    // out from every crossing turns give-way into a crawl.
                    if nearest < 11.0 {
                        target = target.min(m.cruise * ((nearest - 0.8) / 7.0).clamp(0.0, 1.0));
                    }
                }

                let m = &mut self.fleets[fi as usize].movers[mi as usize];
                let delta = target - m.speed;
                m.speed = (m.speed + delta.clamp(-brake * dt, accel * dt)).max(0.0);
            }
            lo = hi;
        }
        self.lane_scratch = scratch;
    }

    /// Decide a speed for every road vehicle and every pedestrian.
    pub(crate) fn plan_traffic(&mut self, phase: usize, dt: f32) {
        self.mark_junctions();
        self.follow_pass(true, phase, dt);
        self.follow_pass(false, phase, dt);
    }

    /// Advance the traffic and the pavements, then push the new transforms
    /// into the instanced meshes. `eye` drives the level-of-detail split and
    /// `time` the signal phase; returns `(drawn at full detail, total)`.
    pub(crate) fn drive(&mut self, scene: &mut Scene, dt: f32, eye: Vector3, time: f32) -> (usize, usize) {
        if dt > 0.0 {
            self.plan_traffic(signal_phase(time), dt);
        }
        // Junctions are read while the fleets that index them are written.
        let junctions = std::mem::take(&mut self.junctions);
        for (fi, fleet) in self.fleets.iter_mut().enumerate() {
            let turns = fi < ROAD_FLEETS;
            for m in fleet.movers.iter_mut() {
                let before = m.s;
                m.s += m.speed * m.dir * dt;
                let span = m.s_hi - m.s_lo;
                if m.turn_s0.is_finite() && m.cornering().is_none() {
                    m.turn_s0 = f32::NAN;
                }
                if span <= 0.01 {
                    // `fit_run` collapses a run shorter than the vehicle to a
                    // point. Wrapping by a zero span does nothing, so the
                    // mover would drift away for ever; park it instead.
                    m.s = m.s_lo;
                } else if m.s > m.s_hi {
                    m.s -= span;
                    m.turn_s0 = f32::NAN;
                } else if m.s < m.s_lo {
                    m.s += span;
                    m.turn_s0 = f32::NAN;
                } else if turns {
                    // Only when the step did not wrap: `before` and `s` are
                    // not comparable across the seam.
                    maybe_turn(m, before, &junctions);
                }
            }
        }
        self.junctions = junctions;
        let mut near = 0usize;
        let mut total = 0usize;
        for fleet in &self.fleets {
            // Pedestrians empty the streets harder than traffic does: there is
            // always some traffic, and at four in the morning there is nobody
            // walking.
            let share = match fleet.gait {
                Gait::Walk => self.activity * self.activity,
                Gait::Float => 1.0,
                Gait::Drive => self.activity,
            };
            near += write_instances(scene, fleet, eye, share);
            total += fleet.movers.len();
        }
        // Lamps track the traffic at the same level of detail: a headlamp on a
        // vehicle too far away to see is not worth drawing. Brakes and
        // indicators are filtered further, by what the vehicle is doing.
        write_lamps(scene, &self.fleets, &self.car_lamps, eye, time, self.activity);

        // Skin follows the people fleet exactly: same movers, same poses, same
        // level-of-detail cut.
        let people = &self.fleets[self.people_fleet];
        let skin = Fleet {
            movers: people.movers.clone(),
            // Skin tone is the tint now, so the six meshes this used to need
            // — two poses by three tones — are two.
            paints: people
                .movers
                .iter()
                .map(|m| Color::from_hex(SKIN_PALETTE[m.tone as usize % SKIN_PALETTE.len()]))
                .collect(),
            groups: self
                .walker_skin
                .iter()
                .map(|(mesh, pose)| Group {
                    mesh: *mesh,
                    start: 0,
                    count: people.movers.len(),
                    pose: *pose,
                })
                .collect(),
            far: None,
            poses: people.poses,
            variants: 1,
            lod: people.lod,
            gait: Gait::Walk,
        };
        write_instances(scene, &skin, eye, self.activity * self.activity);

        // Wildlife. Position and heading are a pure function of the clock, so
        // there is nothing to integrate here — only to place.
        for swarm in &self.swarms {
            near += write_swarm(scene, swarm, eye, time);
            total += swarm.critters.len();
        }
        (near, total)
    }
}

/// Place every mover in this fleet: near instances into their pose mesh and
/// far ones into the shared impostor.
///
/// The transform list of each mesh is REBUILT rather than written slot by
/// slot. A zero-scale matrix rasterises nothing, but it is still uploaded and
/// still runs the vertex shader over the whole model — with two poses and a
/// separate skin mesh that is four wasted copies of every pedestrian in the
/// city. Packing the visible ones to the front and letting the list shrink is
/// what turns the level-of-detail cut into an actual saving.
///
/// Returns how many landed in the near meshes, which is the only LOD figure
/// worth reporting.
/// How many fleets are road traffic, and therefore turn at junctions, queue,
/// obey signals and carry lamps. The rest — people, boats, dogs — do not.
pub(crate) const ROAD_FLEETS: usize = 15;

/// Place the lamps for every road vehicle.
///
/// Not `write_instances` with a cloned fleet, which is what this used to be:
/// that placed every lamp on every car unconditionally and on nothing else at
/// all. Brakes come on when the simulation has slowed a vehicle down —
/// queueing at a red, or behind a bus — and indicators only while a corner is
/// actually in progress, on half the blink cycle. Both read state the traffic
/// pass already maintains, so neither costs anything to know.
pub(crate) fn write_lamps(
    scene: &mut Scene,
    fleets: &[Fleet],
    lamps: &[(ObjectId, Lamp, f32, u32)],
    eye: Vector3,
    time: f32,
    share: f32,
) {
    let blink = (time * 1.5).fract() < 0.5;
    for (mesh, part, _, only) in lamps {
        let Some(obj) = scene.get_mut(*mesh) else {
            continue;
        };
        let ObjectKind::InstancedMesh(im) = &mut obj.kind else {
            continue;
        };
        im.transforms.clear();
        for (fi, fleet) in fleets.iter().take(ROAD_FLEETS).enumerate() {
            if *only != 0 && *only & (1 << fi) == 0 {
                continue;
            }
            let lod2 = fleet.lod * fleet.lod;
            for m in &fleet.movers {
                if !out_today(m, share) {
                    continue;
                }
                let show = match part {
                    // A vehicle the traffic pass has pulled below walking pace
                    // is one standing on its brakes.
                    Lamp::Brake => m.speed < 1.6,
                    Lamp::Indicate => blink && m.cornering().is_some(),
                    Lamp::BeaconA => blink,
                    Lamp::BeaconB => !blink,
                    _ => true,
                };
                if !show {
                    continue;
                }
                let (px, pz, yaw) = m.pose();
                if (px - eye.x).powi(2) + (pz - eye.z).powi(2) > lod2 {
                    continue;
                }
                // The lamp meshes are laid out for a four-metre car. Stretching
                // them along Z by the vehicle's own length puts a bus's lights
                // on the ends of the bus, from one mesh.
                im.transforms.push(Matrix4::compose(
                    Vector3::new(px, m.base_y, pz),
                    Quaternion::from_axis_angle(Vector3::UP, yaw),
                    Vector3::new(m.scale, m.scale, m.scale * m.length / 4.4),
                ));
            }
        }
    }
}

/// Whether this mover is out at all right now.
///
/// Keyed on the mover's own seed rather than on its index or the clock, so the
/// *same* vehicles are absent from one frame to the next. Thinning by index
/// would make traffic flicker in and out as the share moved.
fn out_today(m: &Mover, share: f32) -> bool {
    if share >= 1.0 {
        return true;
    }
    ((m.seed >> 7) & 0x3ff) as f32 / 1024.0 < share
}

pub(crate) fn write_instances(
    scene: &mut Scene,
    fleet: &Fleet,
    eye: Vector3,
    share: f32,
) -> usize {
    let lod2 = fleet.lod * fleet.lod;
    let mut near_count = 0usize;

    let place = |m: &Mover, gait: Gait| -> Matrix4 {
        // Position and heading both come from `Mover::pose`, which follows the
        // corner arc while one is in progress and the lane otherwise.
        let (px, pz, yaw) = m.pose();
        let y = match gait {
            Gait::Drive => m.base_y,
            // A pedestrian at a constant height reads as a sliding chess
            // piece. The bob is in step with the leg swing below.
            Gait::Walk => m.base_y + 0.035 * (m.s * 4.2).sin().abs(),
            Gait::Float => m.base_y + 0.09 * (m.s * 0.14).sin(),
        };
        Matrix4::compose(
            Vector3::new(px, y, pz),
            Quaternion::from_axis_angle(Vector3::UP, yaw),
            Vector3::ONE * m.scale,
        )
    };

    // World position of a mover, for the distance test only.
    let at = |m: &Mover| -> (f32, f32) {
        let (x, z, _) = m.pose();
        (x, z)
    };

    for g in &fleet.groups {
        let mut n = 0usize;
        let Some(obj) = scene.get_mut(g.mesh) else {
            continue;
        };
        let ObjectKind::InstancedMesh(im) = &mut obj.kind else {
            continue;
        };
        im.transforms.clear();
        im.colors.clear();
        for i in 0..g.count {
            let m = &fleet.movers[g.start + i];
            if !out_today(m, share) {
                continue;
            }
            let (wx, wz) = at(m);
            let far = (wx - eye.x).powi(2) + (wz - eye.z).powi(2) > lod2;
            // One stride per 0.8 m walked, so the cycle is tied to the ground
            // rather than to the clock: a slow walker takes slow steps.
            let pose = if fleet.poses > 1 {
                ((m.s / 0.8).floor() as i64).rem_euclid(fleet.poses as i64) as usize
            } else {
                0
            };
            // Tone only participates where the fleet is actually split by it.
            // Folding it in unconditionally makes a tone-1 walker look for a
            // clothes group that does not exist, and it loses its body.
            let slot = if fleet.variants > 1 {
                pose * fleet.variants + m.tone as usize
            } else {
                pose
            };
            if !far && slot == g.pose {
                im.transforms.push(place(m, fleet.gait));
                // Parallel to the transform: instances past the end of the
                // list are white, so a fleet with no paints costs nothing.
                if let Some(c) = fleet.paints.get(g.start + i) {
                    im.colors.push(*c);
                }
                n += 1;
            }
        }
        near_count += n;
    }

    if let Some(far_mesh) = fleet.far {
        let Some(obj) = scene.get_mut(far_mesh) else {
            return near_count;
        };
        let ObjectKind::InstancedMesh(im) = &mut obj.kind else {
            return near_count;
        };
        im.transforms.clear();
        im.colors.clear();
        for (i, m) in fleet.movers.iter().enumerate() {
            if !out_today(m, share) {
                continue;
            }
            let (wx, wz) = at(m);
            if (wx - eye.x).powi(2) + (wz - eye.z).powi(2) > lod2 {
                im.transforms.push(place(m, fleet.gait));
                if let Some(c) = fleet.paints.get(i) {
                    im.colors.push(*c);
                }
            }
        }
    }

    near_count
}
