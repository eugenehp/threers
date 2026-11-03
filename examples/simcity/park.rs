//! Part of the `simcity` example; see `mod.rs`.
//!
//! Parks. Every green block used to be the same object: a lawn, a cross of two
//! straight paths, a rectangular pool and a scatter of random trees. Repeated
//! thirty times across a city that reads as wallpaper, not as landscape.
//!
//! A park is a *designed* thing, and the design is what makes it legible from
//! the air: a formal square is symmetrical about a centrepiece, a sports ground
//! is a marked rectangle with nothing in the middle, a pond park is organised
//! around water, an allotment is a grid of beds. So each green block picks a
//! kind and is built to that kind's own rules.
#![allow(dead_code)]

use super::*;

/// What a green block actually is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ParkKind {
    /// Formal urban square: perimeter walk, diagonals, a centrepiece.
    Square,
    /// Informal planting: serpentine walk, beds, hedges, a pergola.
    Garden,
    /// Organised around an irregular pond, with a jetty and willows.
    Pond,
    /// A marked pitch, goals, floodlights and a stand. Nothing in the middle.
    Sports,
    /// Playground: sandpit, climbing frame, swings, a fence.
    Play,
    /// Hard landscape — no grass at all. Paving, fountain, raised planters.
    Plaza,
    /// Community allotment: raised beds, sheds, a greenhouse.
    Allotment,
    /// Long grass and wildflowers with one mown path through it.
    Meadow,
    /// Dense trees, a path threaded between them, benches in the shade.
    Grove,
    /// Fenced dog run: gravel, agility equipment, a double gate, and dogs off
    /// the lead.
    DogRun,
}

/// What the rest of the generator needs back from a park.
#[derive(Default)]
pub(crate) struct ParkOut {
    /// Rectangles the crowd generator may put people on.
    pub(crate) walks: Vec<(f32, f32, f32, f32)>,
    /// Open water: `(cx, cz, r)`. Waterfowl go here.
    pub(crate) ponds: Vec<(f32, f32, f32)>,
    /// Open hard ground where a flock of pigeons would gather: `(cx, cz, r)`.
    pub(crate) gathers: Vec<(f32, f32, f32)>,
    /// Fenced dog runs: `(cx, cz, r)`. Dogs go round inside these off the
    /// lead, which is the only place in the city they are not attached to a
    /// pedestrian.
    pub(crate) runs: Vec<(f32, f32, f32)>,
}

/// Yaw for `add_yaw_box` such that the box's local `+Z` points along `(dx, dz)`.
///
/// `add_yaw_box` builds its frame as local X = `(cos, sin)` and local Z =
/// `(-sin, cos)`. Reaching for `dx.atan2(dz)` instead gives the mirrored basis,
/// which is subtle enough to survive review and obvious enough on screen: the
/// cast ends of a bench end up several metres from the bench.
pub(crate) fn facing(dx: f32, dz: f32) -> f32 {
    (-dx).atan2(dz)
}

const C_PATH: u32 = 0xa8a498;
const C_GRAVEL: u32 = 0x9c9280;
const C_SOIL: u32 = 0x4b3a2a;
const C_TIMBER: u32 = 0x7a5a3a;
const C_STONE: u32 = 0x93908a;
const C_PONDWATER: u32 = 0x27546a;
const C_SAND: u32 = 0xc4a878;

/// Path surface height. Everything laid on the lawn stacks up from here in
/// centimetre steps, which is enough to keep coplanar quads apart at the
/// depth precision this scene runs at.
const PATH_Y: f32 = KERB + 0.012;

/// Surface of a park pond, a little below the lawn it is cut into. Waterfowl
/// float here, which is why it is a constant and not a local.
pub(crate) const POND_Y: f32 = KERB - 0.16;

/// Pick a park kind for a block.
///
/// `zone` is the built-up intensity, `central` is the block's radial position
/// inside a big central park (0 at the middle, 1 at its rim) or `None` for an
/// ordinary green block. The central case is graded deliberately: a lake in the
/// middle, meadow and grove around it, gardens and squares at the edge where it
/// meets the streets. That gradient is what makes eleven adjacent green blocks
/// read as one large park rather than as eleven small ones.
pub(crate) fn choose_park(cell: &Rect, zone: f32, central: Option<f32>, rng: &mut Rng) -> ParkKind {
    let small = cell.w() < 24.0 || cell.d() < 24.0;
    if let Some(t) = central {
        return if t < 0.22 {
            ParkKind::Pond
        } else if t < 0.52 {
            if rng.chance(0.35) {
                ParkKind::Grove
            } else {
                ParkKind::Meadow
            }
        } else if t < 0.78 {
            match rng.below(3) {
                0 => ParkKind::Sports,
                1 => ParkKind::Grove,
                _ => ParkKind::Garden,
            }
        } else {
            match rng.below(3) {
                0 => ParkKind::Garden,
                1 => ParkKind::Square,
                _ => ParkKind::Play,
            }
        };
    }
    // Downtown gets hard landscape; the outskirts get allotments and meadows.
    // `zone` runs 0 in the core to about 1.25 at the edge.
    let roll = rng.f();
    if zone < 0.35 {
        if roll < 0.45 {
            ParkKind::Plaza
        } else if roll < 0.72 {
            ParkKind::Square
        } else {
            ParkKind::Garden
        }
    } else if zone < 0.75 {
        if small {
            if roll < 0.34 {
                ParkKind::Square
            } else if roll < 0.60 {
                ParkKind::Play
            } else {
                ParkKind::Garden
            }
        } else if roll < 0.22 {
            ParkKind::Pond
        } else if roll < 0.46 {
            ParkKind::Sports
        } else if roll < 0.66 {
            ParkKind::Garden
        } else if roll < 0.78 {
            ParkKind::Play
        } else if roll < 0.90 {
            ParkKind::DogRun
        } else {
            ParkKind::Grove
        }
    } else if roll < 0.26 {
        ParkKind::Allotment
    } else if roll < 0.50 {
        ParkKind::Meadow
    } else if roll < 0.70 {
        ParkKind::Grove
    } else if roll < 0.80 {
        ParkKind::Play
    } else if roll < 0.90 {
        ParkKind::DogRun
    } else {
        ParkKind::Sports
    }
}

// ---------------------------------------------------------------------------
// Shared pieces.
// ---------------------------------------------------------------------------

fn lawn(b: &mut Batches, cell: &Rect, tint: f32, rng: &mut Rng) {
    b.grass.add_box(
        Vector3::new(cell.x0, 0.0, cell.z0),
        Vector3::new(cell.x1, KERB, cell.z1),
        scale_color(Color::from_hex(C_GRASS), tint * rng.range(0.92, 1.08)),
        Uv::Unit,
    );
}

/// Short paths from the middle of each edge into the block.
///
/// Whatever a park does inside itself, it has to meet its neighbours: without
/// these, two adjacent park blocks in a large park are two islands with a seam
/// between them. With them the walks join up across the whole green area.
fn edge_stubs(b: &mut Batches, cell: &Rect, width: f32, col: Color) {
    let inset = 4.0f32.min(cell.w().min(cell.d()) * 0.25);
    for pts in [
        [(cell.cx(), cell.z0), (cell.cx(), cell.z0 + inset)],
        [(cell.cx(), cell.z1), (cell.cx(), cell.z1 - inset)],
        [(cell.x0, cell.cz()), (cell.x0 + inset, cell.cz())],
        [(cell.x1, cell.cz()), (cell.x1 - inset, cell.cz())],
    ] {
        b.pads.add_ground_path(&pts, width, PATH_Y, col, Uv::Unit);
    }
}

/// Lawn with a circular hole cut in its surface.
///
/// A block is a solid box from the ground up to the kerb, so anything sunk
/// into it — a pond, a basin — is simply hidden underneath the top face. The
/// surface has to have a hole in it. The ring between the hole and the pond's
/// bounding square is tiled with quads, and the four corner angles are forced
/// into the sample set so a segment spanning a corner does not cut it off.
fn lawn_with_hole(
    b: &mut Batches,
    cell: &Rect,
    cx: f32,
    cz: f32,
    r: f32,
    segs: usize,
    wobble: f32,
    col: Color,
) {
    let reach = (r * 1.28)
        .min(cx - cell.x0)
        .min(cell.x1 - cx)
        .min(cz - cell.z0)
        .min(cell.z1 - cz);
    let bb = Rect {
        x0: cx - reach,
        z0: cz - reach,
        x1: cx + reach,
        z1: cz + reach,
    };
    // Everything outside the bounding square is ordinary lawn.
    for (x0, z0, x1, z1) in [
        (cell.x0, cell.z0, cell.x1, bb.z0),
        (cell.x0, bb.z1, cell.x1, cell.z1),
        (cell.x0, bb.z0, bb.x0, bb.z1),
        (bb.x1, bb.z0, cell.x1, bb.z1),
    ] {
        if x1 - x0 > 1e-3 && z1 - z0 > 1e-3 {
            b.grass.add_box(
                Vector3::new(x0, 0.0, z0),
                Vector3::new(x1, KERB, z1),
                col,
                Uv::Unit,
            );
        }
    }
    // The rim points are the disc's own vertices, sampled exactly as
    // `add_ground_disc` samples them so the hole and the pond wall share an
    // edge rather than leaving a crack along it.
    let rim = |i: usize| -> (f32, f32) {
        let a = (i % segs) as f32 / segs as f32 * TAU;
        let rr = r * (1.0 + wobble * (city_hash2((i % segs) as i32 * 13, 7) - 0.5) * 2.0);
        (cx + rr * a.cos(), cz + rr * a.sin())
    };
    // Where a ray at `a` leaves the bounding square.
    let edge = |a: f32| -> (f32, f32) {
        let (c, si) = (a.cos(), a.sin());
        let t = (reach / c.abs().max(1e-6)).min(reach / si.abs().max(1e-6));
        (cx + t * c, cz + t * si)
    };
    let corners = [
        PI * 0.25,
        PI * 0.75,
        PI * 1.25,
        PI * 1.75,
    ];
    for i in 0..segs {
        let (a0, a1) = (
            i as f32 / segs as f32 * TAU,
            (i + 1) as f32 / segs as f32 * TAU,
        );
        let (i0, i1) = (rim(i), rim(i + 1));
        // Outer polyline: the two ray hits, with any square corner the segment
        // spans inserted between them. Without this each corner of the hole's
        // bounding square is sliced off and the ground shows through.
        let mut outer = vec![edge(a0)];
        for c in corners {
            if c > a0 + 1e-4 && c < a1 - 1e-4 {
                outer.push(edge(c));
            }
        }
        outer.push(edge(a1));
        // Fan from the inner edge. Winding is inner-then-outer: the other way
        // round the whole ring faces down and vanishes.
        let uv = [[0.5, 0.5]; 3];
        b.grass.tri(
            [
                [i0.0, KERB, i0.1],
                [i1.0, KERB, i1.1],
                [outer[outer.len() - 1].0, KERB, outer[outer.len() - 1].1],
            ],
            [0.0, 1.0, 0.0],
            uv,
            [col; 3],
        );
        for k in (1..outer.len()).rev() {
            b.grass.tri(
                [
                    [i0.0, KERB, i0.1],
                    [outer[k].0, KERB, outer[k].1],
                    [outer[k - 1].0, KERB, outer[k - 1].1],
                ],
                [0.0, 1.0, 0.0],
                uv,
                [col; 3],
            );
        }
    }
}

/// A rectangular loop path inset from the block edge.
fn loop_path(b: &mut Batches, r: &Rect, width: f32, col: Color) {
    b.pads.add_ground_path(
        &[
            (r.x0, r.z0),
            (r.x1, r.z0),
            (r.x1, r.z1),
            (r.x0, r.z1),
            (r.x0, r.z0),
            (r.x1, r.z0),
        ],
        width,
        PATH_Y,
        col,
        Uv::Unit,
    );
}

/// A circle approximated by a closed polyline, for track markings and rims.
fn ring_path(
    b: &mut MeshBuilder,
    cx: f32,
    cz: f32,
    r: f32,
    width: f32,
    y: f32,
    col: Color,
    segs: usize,
) {
    let mut pts: Vec<(f32, f32)> = (0..=segs)
        .map(|i| {
            let a = i as f32 / segs as f32 * TAU;
            (cx + r * a.cos(), cz + r * a.sin())
        })
        .collect();
    pts.push(pts[1]);
    b.add_ground_path(&pts, width, y, col, Uv::Unit);
}

/// A raised bed: a mound of one colour with a visible side wall of another.
/// Flower beds, sandpits and vegetable plots are all this shape.
fn raised_bed(
    b: &mut MeshBuilder,
    cx: f32,
    cz: f32,
    r: f32,
    h: f32,
    top: Color,
    side: Color,
    rng: &mut Rng,
) {
    let segs = 9;
    let wob = rng.range(0.10, 0.24);
    b.add_ground_disc(cx, cz, r, segs, wob, KERB + h, top);
    b.add_ground_disc_wall(cx, cz, r, segs, wob, KERB + h, KERB, side);
}

/// A park bench, facing along `+yaw`.
pub(crate) fn add_bench(b: &mut Batches, x: f32, z: f32, yaw: f32, rng: &mut Rng) {
    if !b.occ.try_spot(x, z, 1.15) {
        return;
    }
    b.seats.push((x, z, yaw));
    let wood = scale_color(Color::from_hex(0x6b4a30), rng.range(0.85, 1.15));
    let iron = Color::from_hex(0x33383c);
    // Local (right, up, forward) into world, in the same frame `add_yaw_box`
    // uses: right is local X, forward is local Z.
    let c = |dx: f32, dy: f32, dz: f32| {
        Vector3::new(
            x + dx * yaw.cos() - dz * yaw.sin(),
            KERB + dy,
            z + dx * yaw.sin() + dz * yaw.cos(),
        )
    };
    // Seat, back, and two cast ends.
    b.trim
        .add_yaw_box(c(0.0, 0.44, 0.0), Vector3::new(0.85, 0.04, 0.24), yaw, wood, Uv::Unit);
    b.trim.add_yaw_box(
        c(0.0, 0.70, -0.22),
        Vector3::new(0.85, 0.22, 0.04),
        yaw,
        wood,
        Uv::Unit,
    );
    for s in [-1.0f32, 1.0] {
        b.trim.add_yaw_box(
            c(s * 0.78, 0.22, 0.0),
            Vector3::new(0.05, 0.22, 0.24),
            yaw,
            iron,
            Uv::Unit,
        );
    }
}

/// A low clipped hedge running between two points.
fn hedge(b: &mut Batches, a: (f32, f32), c: (f32, f32), h: f32, rng: &mut Rng) {
    let (dx, dz) = (c.0 - a.0, c.1 - a.1);
    let len = (dx * dx + dz * dz).sqrt();
    if len < 0.5 {
        return;
    }
    let yaw = facing(dx, dz);
    let col = scale_color(Color::from_hex(0x2c4a24), rng.range(0.85, 1.15));
    b.foliage.add_yaw_box(
        Vector3::new((a.0 + c.0) * 0.5, KERB + h * 0.5, (a.1 + c.1) * 0.5),
        Vector3::new(0.34, h * 0.5, len * 0.5),
        yaw,
        col,
        Uv::Unit,
    );
}

/// A fountain: basin, bowl, a jet and the water falling off it.
///
/// The water is the hard part. With nothing translucent to hand, a plume drawn
/// as fat cylinders reads as a cluster of pale poles — which is exactly what
/// the first attempt looked like. Thin tapered limbs on ballistic arcs read as
/// water because the *shape* is right: rising, turning over, falling into the
/// basin.
fn fountain(b: &mut Batches, cx: f32, cz: f32, r: f32, rng: &mut Rng) {
    let stone = scale_color(Color::from_hex(C_STONE), rng.range(0.92, 1.08));
    let spray = Color::from_hex(0xd6e7ef);
    b.trim
        .add_cylinder(Vector3::new(cx, KERB, cz), r, r, 0.55, 20, stone, true, Uv::Unit);
    b.water
        .add_ground_disc(cx, cz, r - 0.34, 20, 0.0, KERB + 0.42, Color::from_hex(C_PONDWATER));
    // Pedestal and the bowl it carries.
    let ped = KERB + 0.42;
    b.trim.add_cylinder(
        Vector3::new(cx, ped, cz),
        r * 0.17,
        r * 0.12,
        0.95,
        12,
        stone,
        true,
        Uv::Unit,
    );
    let bowl_y = ped + 0.95;
    let bowl_r = (r * 0.30).clamp(0.5, 1.4);
    b.trim.add_cylinder(
        Vector3::new(cx, bowl_y, cz),
        bowl_r * 0.75,
        bowl_r,
        0.16,
        14,
        stone,
        true,
        Uv::Unit,
    );
    // The jet, and the crown of water falling off it into the bowl.
    let jet = bowl_y + 0.16;
    let top = jet + (r * 0.28).clamp(0.7, 1.5);
    b.trim.add_cylinder(
        Vector3::new(cx, jet, cz),
        0.055,
        0.018,
        top - jet,
        6,
        spray,
        false,
        Uv::Unit,
    );
    // Water off the bowl and a ring of jets from the basin rim.
    //
    // Full arcs were tried and abandoned: with nothing translucent available,
    // a metre-long limb two centimetres thick reads as wire however well the
    // parabola is sampled. What does read as water is the *break* — a flare
    // where the jet loses coherence, and froth where it lands. So the jets are
    // short and the spray is volume.
    b.trim.add_cylinder(
        Vector3::new(cx, top - 0.06, cz),
        0.05,
        (bowl_r * 0.55).max(0.18),
        0.34,
        12,
        spray,
        false,
        Uv::Unit,
    );
    for i in 0..8 {
        let a = i as f32 / 8.0 * TAU;
        let (ax, az) = (a.cos(), a.sin());
        b.trim.add_ellipsoid(
            Vector3::new(
                cx + ax * bowl_r * 0.80,
                bowl_y + 0.30,
                cz + az * bowl_r * 0.80,
            ),
            0.10,
            0.16,
            0.10,
            3,
            5,
            spray,
        );
    }
    // Rim jets: the first stretch of the throw only, where it is still a jet.
    let rim = (r - 0.55).max(0.5);
    for i in 0..6 {
        let a = i as f32 / 6.0 * TAU + 0.4;
        let (ax, az) = (a.cos(), a.sin());
        b.trim.add_limb(
            Vector3::new(cx + ax * rim, KERB + 0.50, cz + az * rim),
            Vector3::new(
                cx + ax * (rim - 0.42),
                KERB + 1.28,
                cz + az * (rim - 0.42),
            ),
            0.038,
            0.012,
            5,
            spray,
            false,
        );
    }
    // Froth where the water lands.
    for i in 0..7 {
        let a = i as f32 / 7.0 * TAU + 0.9;
        let d = r * rng.range(0.35, 0.80);
        b.trim.add_ellipsoid(
            Vector3::new(cx + d * a.cos(), KERB + 0.44, cz + d * a.sin()),
            rng.range(0.14, 0.30),
            0.07,
            rng.range(0.14, 0.30),
            3,
            6,
            spray,
        );
    }
}

/// A plinth with a figure on it — the other half of the centrepieces.
fn statue(b: &mut Batches, cx: f32, cz: f32, rng: &mut Rng) {
    let stone = scale_color(Color::from_hex(0x8f8b82), rng.range(0.9, 1.1));
    let bronze = Color::from_hex(0x4d5f47);
    b.trim.add_box(
        Vector3::new(cx - 1.05, KERB, cz - 1.05),
        Vector3::new(cx + 1.05, KERB + 0.30, cz + 1.05),
        stone,
        Uv::Unit,
    );
    b.trim.add_box(
        Vector3::new(cx - 0.72, KERB + 0.30, cz - 0.72),
        Vector3::new(cx + 0.72, KERB + 2.30, cz + 0.72),
        stone,
        Uv::Unit,
    );
    let yaw = rng.range(0.0, TAU);
    b.trim.add_yaw_box(
        Vector3::new(cx, KERB + 3.05, cz),
        Vector3::new(0.30, 0.75, 0.20),
        yaw,
        bronze,
        Uv::Unit,
    );
    b.trim
        .add_ellipsoid(Vector3::new(cx, KERB + 3.95, cz), 0.22, 0.26, 0.22, 6, 8, bronze);
    // One arm out, which is what makes a bronze block read as a person.
    b.trim.add_yaw_box(
        Vector3::new(cx + 0.42 * yaw.cos(), KERB + 3.60, cz + 0.42 * yaw.sin()),
        Vector3::new(0.42, 0.10, 0.10),
        yaw,
        bronze,
        Uv::Unit,
    );
}

// ---------------------------------------------------------------------------
// The kinds.
// ---------------------------------------------------------------------------

/// Build a green block. Returns what the crowd and the wildlife need from it.
pub(crate) fn add_park(
    b: &mut Batches,
    cell: Rect,
    kind: ParkKind,
    warm: bool,
    near: bool,
    rng: &mut Rng,
) -> ParkOut {
    let mut out = ParkOut::default();
    match kind {
        ParkKind::Square => square(b, &cell, warm, near, rng, &mut out),
        ParkKind::Garden => garden(b, &cell, warm, near, rng, &mut out),
        ParkKind::Pond => pond(b, &cell, warm, near, rng, &mut out),
        ParkKind::Sports => sports(b, &cell, warm, near, rng, &mut out),
        ParkKind::Play => play(b, &cell, warm, near, rng, &mut out),
        ParkKind::Plaza => plaza(b, &cell, warm, near, rng, &mut out),
        ParkKind::Allotment => allotment(b, &cell, warm, near, rng, &mut out),
        ParkKind::Meadow => meadow(b, &cell, warm, near, rng, &mut out),
        ParkKind::Grove => grove(b, &cell, warm, near, rng, &mut out),
        ParkKind::DogRun => dog_run(b, &cell, warm, near, rng, &mut out),
    }
    out
}

/// Formal square: symmetrical about a centrepiece, walked round and across.
fn square(b: &mut Batches, cell: &Rect, warm: bool, near: bool, rng: &mut Rng, out: &mut ParkOut) {
    lawn(b, cell, 1.0, rng);
    let path = Color::from_hex(C_PATH);
    let walk = cell.inset(3.0);
    loop_path(b, &walk, 2.4, path);
    edge_stubs(b, cell, 2.2, path);

    // Diagonals, cut short of the middle so the roundel is not crossed twice.
    let r = (cell.w().min(cell.d()) * 0.16).clamp(2.4, 5.5);
    for (a, c) in [
        ((walk.x0, walk.z0), (walk.x1, walk.z1)),
        ((walk.x1, walk.z0), (walk.x0, walk.z1)),
    ] {
        let (mx, mz) = (cell.cx(), cell.cz());
        let stop = |p: (f32, f32)| {
            let (dx, dz) = (p.0 - mx, p.1 - mz);
            let l = (dx * dx + dz * dz).sqrt().max(1e-3);
            (mx + dx / l * (l - r + 0.4), mz + dz / l * (l - r + 0.4))
        };
        b.pads
            .add_ground_path(&[a, stop(a)], 2.0, PATH_Y, path, Uv::Unit);
        b.pads
            .add_ground_path(&[c, stop(c)], 2.0, PATH_Y, path, Uv::Unit);
    }
    b.pads
        .add_ground_disc(cell.cx(), cell.cz(), r, 18, 0.0, PATH_Y, path);

    let centre_r = r * 0.52;
    match rng.below(3) {
        0 => fountain(b, cell.cx(), cell.cz(), centre_r, rng),
        1 => statue(b, cell.cx(), cell.cz(), rng),
        _ => add_bandstand(b, cell.cx(), cell.cz(), (r * 0.62).clamp(1.8, 3.6), rng),
    }
    // Clear of whatever stands in the middle.
    for k in 0..3 {
        let a = k as f32 / 3.0 * TAU;
        out.gathers.push((
            cell.cx() + (centre_r + 1.6) * a.cos(),
            cell.cz() + (centre_r + 1.6) * a.sin(),
            1.4,
        ));
    }
    // Benches round the roundel, all facing in.
    for i in 0..4 {
        let a = i as f32 / 4.0 * TAU + PI * 0.25;
        add_bench(
            b,
            cell.cx() + (r + 1.5) * a.cos(),
            cell.cz() + (r + 1.5) * a.sin(),
            facing(-a.cos(), -a.sin()),
            rng,
        );
    }
    // One species, planted on a regular spacing outside the walk. A formal
    // square is an avenue of matched trees; mixing them makes it a wood.
    let species = if warm {
        Species::Palm
    } else if rng.chance(0.4) {
        Species::Columnar
    } else {
        Species::Broadleaf
    };
    let step = 7.0;
    let n = ((walk.w() - 2.0) / step).floor().max(1.0) as usize;
    let m = ((walk.d() - 2.0) / step).floor().max(1.0) as usize;
    for i in 0..=n {
        let x = walk.x0 + (walk.w()) * i as f32 / n as f32;
        for z in [cell.z0 + 1.6, cell.z1 - 1.6] {
            add_tree_as(b, x, z, KERB, near, species, None, 0.95, rng);
        }
    }
    for j in 1..m {
        let z = walk.z0 + (walk.d()) * j as f32 / m as f32;
        for x in [cell.x0 + 1.6, cell.x1 - 1.6] {
            add_tree_as(b, x, z, KERB, near, species, None, 0.95, rng);
        }
    }
    // Railings, which is what makes a square a square rather than a lawn: it
    // is enclosed, and it has a gate.
    if cell.w() > 20.0 && cell.d() > 20.0 && rng.chance(0.7) {
        add_railings(b, &cell.inset(0.8), rng.below(4), rng);
    }
    for i in 0..3 {
        let a = i as f32 / 3.0 * TAU + 0.7;
        add_park_furniture(
            b,
            cell.cx() + (r + 3.4) * a.cos(),
            cell.cz() + (r + 3.4) * a.sin(),
            rng,
        );
    }
    out.walks.push((walk.x0, walk.z0, walk.x1, walk.z1));
}

/// Informal garden: a curved walk, beds, hedges and a pergola.
fn garden(b: &mut Batches, cell: &Rect, warm: bool, near: bool, rng: &mut Rng, out: &mut ParkOut) {
    lawn(b, cell, 1.04, rng);
    let path = Color::from_hex(C_GRAVEL);
    edge_stubs(b, cell, 2.0, path);

    // A serpentine spine across the long axis, plus one branch off it. The
    // amplitude is a fraction of the short side so it never leaves the block.
    let along_x = cell.w() >= cell.d();
    let amp = (if along_x { cell.d() } else { cell.w() }) * 0.22;
    let phase = rng.range(0.0, TAU);
    let spine: Vec<(f32, f32)> = (0..=14)
        .map(|i| {
            let t = i as f32 / 14.0;
            let wig = (phase + t * TAU * 1.15).sin() * amp;
            if along_x {
                (cell.x0 + t * cell.w(), cell.cz() + wig)
            } else {
                (cell.cx() + wig, cell.z0 + t * cell.d())
            }
        })
        .collect();
    b.pads.add_ground_path(&spine, 2.0, PATH_Y, path, Uv::Unit);
    let mid = spine[spine.len() / 2];
    let branch = if along_x {
        [(mid.0, mid.1), (mid.0 + rng.range(-6.0, 6.0), cell.z1 - 3.0)]
    } else {
        [(mid.0, mid.1), (cell.x1 - 3.0, mid.1 + rng.range(-6.0, 6.0))]
    };
    b.pads.add_ground_path(&branch, 1.6, PATH_Y, path, Uv::Unit);

    // Beds. Bedding plants are the one thing in a city that is allowed to be
    // a saturated colour, so they are what the eye finds first from the air.
    // Bedding plants are the one thing in a city allowed to be a saturated
    // colour — but one big disc of it reads as a blob of paint from the air.
    // A bed is a cluster of small mounds in one colour family, sitting in a
    // ring of turned soil, which is what a planted bed actually looks like.
    const BLOOM: [u32; 5] = [0xb0566f, 0xc09a3e, 0xcfc9bc, 0x8a6198, 0xb05a44];
    let beds = 3 + rng.below(4);
    for _ in 0..beds {
        let bx = rng.range(cell.x0 + 3.0, cell.x1 - 3.0);
        let bz = rng.range(cell.z0 + 3.0, cell.z1 - 3.0);
        let br = rng.range(1.3, 2.4);
        let family = Color::from_hex(BLOOM[rng.below(BLOOM.len())]);
        b.grass
            .add_ground_disc(bx, bz, br, 9, 0.18, KERB + 0.02, Color::from_hex(C_SOIL));
        for _ in 0..(2 + rng.below(3)) {
            let a = rng.range(0.0, TAU);
            let d = br * rng.range(0.0, 0.55);
            raised_bed(
                &mut b.foliage,
                bx + d * a.cos(),
                bz + d * a.sin(),
                br * rng.range(0.35, 0.60),
                rng.range(0.22, 0.40),
                scale_color(family, rng.range(0.82, 1.14)),
                Color::from_hex(C_SOIL),
                rng,
            );
        }
    }
    // Clipped hedges framing one corner.
    let h = cell.inset(2.4);
    hedge(b, (h.x0, h.z0), (h.x0 + h.w() * 0.4, h.z0), 0.85, rng);
    hedge(b, (h.x0, h.z0), (h.x0, h.z0 + h.d() * 0.4), 0.85, rng);

    // Pergola: two rows of posts with beams across, over the branch path.
    if cell.w() > 20.0 && cell.d() > 20.0 {
        let (px, pz) = (branch[1].0, branch[1].1);
        let yaw = rng.range(0.0, PI);
        let wood = Color::from_hex(C_TIMBER);
        for i in 0..5 {
            let t = (i as f32 - 2.0) * 1.6;
            for s in [-1.2f32, 1.2] {
                b.trim.add_cylinder(
                    Vector3::new(
                        px + t * yaw.cos() - s * yaw.sin(),
                        KERB,
                        pz + t * yaw.sin() + s * yaw.cos(),
                    ),
                    0.09,
                    0.09,
                    2.35,
                    6,
                    wood,
                    true,
                    Uv::Unit,
                );
            }
            b.trim.add_yaw_box(
                Vector3::new(px + t * yaw.cos(), KERB + 2.45, pz + t * yaw.sin()),
                Vector3::new(0.06, 0.06, 1.45),
                yaw,
                wood,
                Uv::Unit,
            );
        }
    }

    for _ in 0..(cell.area() / 70.0) as usize {
        let tx = rng.range(cell.x0 + 2.0, cell.x1 - 2.0);
        let tz = rng.range(cell.z0 + 2.0, cell.z1 - 2.0);
        // Ornamental blossom among the ordinary planting.
        if rng.chance(0.16) {
            let bloom = mix_color(
                Color::from_hex(0xd98fa8),
                Color::from_hex(0xc9a3b4),
                rng.f(),
            );
            add_tree_as(b, tx, tz, KERB, near, Species::Broadleaf, Some(bloom), 0.7, rng);
        } else {
            add_tree(b, tx, tz, KERB, near, warm, rng);
        }
    }
    for _ in 0..2 {
        let i = 2 + rng.below(spine.len() - 4);
        add_bench(b, spine[i].0, spine[i].1 + 1.6, facing(0.0, -1.0), rng);
    }
    if cell.w() > 26.0 && cell.d() > 26.0 && rng.chance(0.45) {
        add_kiosk(b, mix(cell.x0 + 4.0, cell.x1 - 4.0, rng.f()), cell.z0 + 4.0, rng);
    }
    for _ in 0..2 {
        let i = 1 + rng.below(spine.len() - 2);
        add_park_furniture(b, spine[i].0 + rng.range(-1.6, 1.6), spine[i].1 + 1.5, rng);
    }
    out.walks
        .push((cell.x0 + 2.0, cell.z0 + 2.0, cell.x1 - 2.0, cell.z1 - 2.0));
}

/// A pond park: irregular water, a shore walk, a jetty and willows.
fn pond(b: &mut Batches, cell: &Rect, warm: bool, near: bool, rng: &mut Rng, out: &mut ParkOut) {
    // A pond is water plus a shore walk plus benches facing it, and those
    // reach a good five metres past the water. On a small block that puts the
    // walk on the pavement and the benches in the road, so build something
    // else instead.
    let short = cell.w().min(cell.d());
    if short < 25.0 {
        return garden(b, cell, warm, near, rng, out);
    }
    let cxp = cell.cx() + rng.range(-1.5, 1.5);
    let czp = cell.cz() + rng.range(-1.5, 1.5);
    // Sized from what has to fit around it, not just from the block: the walk
    // and the seating are what set the limit.
    let r = (short * 0.30).clamp(4.0, 18.0).min(short * 0.5 - 6.5);
    let segs = 15;
    let wob = 0.22;
    let turf = scale_color(Color::from_hex(C_GRASS), 1.02 * rng.range(0.92, 1.08));
    lawn_with_hole(b, cell, cxp, czp, r, segs, wob, turf);
    // The bank drops from the lawn to the bed, and the water sits below its
    // lip — which is what makes the shoreline an edge and not a painted circle.
    b.grass.add_ground_disc_wall(
        cxp,
        czp,
        r,
        segs,
        wob,
        KERB,
        KERB - 0.62,
        Color::from_hex(C_SOIL),
    );
    b.grass
        .add_ground_disc(cxp, czp, r, segs, wob, KERB - 0.62, Color::from_hex(C_SOIL));
    b.water.add_ground_disc(
        cxp,
        czp,
        r * 0.985,
        segs,
        wob,
        POND_Y,
        Color::from_hex(C_PONDWATER),
    );
    out.ponds.push((cxp, czp, r * 0.8));

    let path = Color::from_hex(C_GRAVEL);
    ring_path(&mut b.pads, cxp, czp, r + 2.6, 2.2, PATH_Y, path, 16);
    edge_stubs(b, cell, 2.0, path);

    // Timber jetty out over the water, on posts.
    let ja = rng.range(0.0, TAU);
    let (jx, jz) = (cxp + ja.cos(), czp + ja.sin());
    let reach = r * 0.62;
    b.trim.add_yaw_box(
        Vector3::new(jx + ja.cos() * reach * 0.5, KERB + 0.04, jz + ja.sin() * reach * 0.5),
        Vector3::new(0.9, 0.06, reach * 0.5),
        facing(ja.cos(), ja.sin()),
        Color::from_hex(C_TIMBER),
        Uv::Unit,
    );
    for i in 0..3 {
        let t = (i as f32 + 0.5) / 3.0 * reach;
        for s in [-0.7f32, 0.7] {
            b.trim.add_cylinder(
                Vector3::new(
                    jx + ja.cos() * t - s * ja.sin(),
                    KERB - 0.62,
                    jz + ja.sin() * t + s * ja.cos(),
                ),
                0.09,
                0.09,
                0.70,
                5,
                Color::from_hex(0x53412d),
                false,
                Uv::Unit,
            );
        }
    }

    // Reeds round the rim, and willows leaning over the bank.
    let reed = Color::from_hex(0x5c7a34);
    for i in 0..(if near { 90 } else { 26 }) {
        let a = i as f32 / 90.0 * TAU + rng.range(0.0, 0.4);
        let rr = r * rng.range(0.99, 1.06);
        let (bx, bz) = (cxp + rr * a.cos(), czp + rr * a.sin());
        let h = rng.range(0.5, 1.3);
        b.foliage.add_limb(
            Vector3::new(bx, KERB - 0.30, bz),
            Vector3::new(bx + rng.range(-0.3, 0.3), KERB - 0.30 + h, bz + rng.range(-0.3, 0.3)),
            0.06,
            0.01,
            3,
            scale_color(reed, rng.range(0.8, 1.2)),
            false,
        );
    }
    for i in 0..5 {
        let a = i as f32 / 5.0 * TAU + rng.range(0.0, 1.0);
        let rr = r + rng.range(3.4, 5.5);
        let (tx, tz) = (cxp + rr * a.cos(), czp + rr * a.sin());
        if tx < cell.x0 + 1.5 || tx > cell.x1 - 1.5 || tz < cell.z0 + 1.5 || tz > cell.z1 - 1.5 {
            continue;
        }
        add_tree_as(b, tx, tz, KERB, near, Species::Weeping, None, 1.15, rng);
    }
    for i in 0..3 {
        let a = i as f32 / 3.0 * TAU + 0.6;
        add_bench(
            b,
            cxp + (r + 3.9) * a.cos(),
            czp + (r + 3.9) * a.sin(),
            facing(-a.cos(), -a.sin()),
            rng,
        );
    }
    let _ = warm;
    out.walks
        .push((cell.x0 + 2.0, cell.z0 + 2.0, cell.x1 - 2.0, cell.z1 - 2.0));
}

/// A sports ground: a marked pitch with floodlights and a stand.
///
/// The mown stripes are the point. A pitch is the one piece of grass in a city
/// that is cut in bands, and from the air that alternation identifies it
/// instantly — more than the markings do, which are only a few centimetres
/// wide.
fn sports(b: &mut Batches, cell: &Rect, warm: bool, near: bool, rng: &mut Rng, out: &mut ParkOut) {
    lawn(b, cell, 0.94, rng);
    let p = cell.inset(3.2);
    if p.w() < 8.0 || p.d() < 8.0 {
        return grove(b, cell, warm, near, rng, out);
    }
    // A hard court where the block is too small for a pitch, or one time in
    // three anyway: not every sports ground is a football field.
    if p.w() < 22.0 || p.d() < 22.0 || rng.chance(0.34) {
        hard_court(b, cell, &p, near, warm, rng, out);
        return;
    }
    let along_x = p.w() >= p.d();
    let (len, wid) = if along_x { (p.w(), p.d()) } else { (p.d(), p.w()) };
    let bands = ((len / 5.0).round() as usize).clamp(4, 14);
    let dark = scale_color(Color::from_hex(C_GRASS), 0.88);
    let pale = scale_color(Color::from_hex(C_GRASS), 1.12);
    for i in 0..bands {
        let (t0, t1) = (i as f32 / bands as f32, (i + 1) as f32 / bands as f32);
        let c = if i % 2 == 0 { dark } else { pale };
        if along_x {
            b.grass.add_slab(
                p.x0 + t0 * len,
                p.z0,
                p.x0 + t1 * len,
                p.z1,
                KERB + 0.008,
                c,
                Uv::Unit,
            );
        } else {
            b.grass.add_slab(
                p.x0,
                p.z0 + t0 * len,
                p.x1,
                p.z0 + t1 * len,
                KERB + 0.008,
                c,
                Uv::Unit,
            );
        }
    }

    // Markings.
    let white = Color::from_hex(0xd8d6cc);
    let my = KERB + 0.022;
    let m = p.inset(1.2);
    b.paint.add_ground_path(
        &[
            (m.x0, m.z0),
            (m.x1, m.z0),
            (m.x1, m.z1),
            (m.x0, m.z1),
            (m.x0, m.z0),
            (m.x1, m.z0),
        ],
        0.18,
        my,
        white,
        Uv::Unit,
    );
    if along_x {
        b.paint
            .add_ground_path(&[(m.cx(), m.z0), (m.cx(), m.z1)], 0.18, my, white, Uv::Unit);
    } else {
        b.paint
            .add_ground_path(&[(m.x0, m.cz()), (m.x1, m.cz())], 0.18, my, white, Uv::Unit);
    }
    ring_path(
        &mut b.paint,
        m.cx(),
        m.cz(),
        (wid * 0.16).min(9.0),
        0.18,
        my,
        white,
        18,
    );
    // Penalty areas at each end.
    for s in [0.0f32, 1.0] {
        let box_len = (len * 0.14).min(16.0);
        let box_wid = (wid * 0.55).min(40.0);
        let (bx0, bz0, bx1, bz1) = if along_x {
            let x = m.x0 + s * m.w();
            let x2 = x + if s == 0.0 { box_len } else { -box_len };
            (x.min(x2), m.cz() - box_wid * 0.5, x.max(x2), m.cz() + box_wid * 0.5)
        } else {
            let z = m.z0 + s * m.d();
            let z2 = z + if s == 0.0 { box_len } else { -box_len };
            (m.cx() - box_wid * 0.5, z.min(z2), m.cx() + box_wid * 0.5, z.max(z2))
        };
        b.paint.add_ground_path(
            &[
                (bx0, bz0),
                (bx1, bz0),
                (bx1, bz1),
                (bx0, bz1),
                (bx0, bz0),
                (bx1, bz0),
            ],
            0.16,
            my,
            white,
            Uv::Unit,
        );
        // Goal: two posts and a bar, standing on the goal line.
        let goal_w = (wid * 0.16).clamp(2.4, 7.3);
        let (gx, gz) = if along_x {
            (m.x0 + s * m.w(), m.cz())
        } else {
            (m.cx(), m.z0 + s * m.d())
        };
        // The goal stands on the goal line, so its posts run across the pitch.
        let (ux, uz) = if along_x { (0.0f32, 1.0f32) } else { (1.0, 0.0) };
        for side in [-1.0f32, 1.0] {
            b.trim.add_cylinder(
                Vector3::new(
                    gx + side * goal_w * 0.5 * ux,
                    KERB,
                    gz + side * goal_w * 0.5 * uz,
                ),
                0.07,
                0.07,
                2.44,
                6,
                white,
                false,
                Uv::Unit,
            );
        }
        b.trim.add_yaw_box(
            Vector3::new(gx, KERB + 2.48, gz),
            Vector3::new(0.07, 0.07, goal_w * 0.5),
            facing(ux, uz),
            white,
            Uv::Unit,
        );
    }

    // Floodlight masts, and a small stand along one side.
    let mast = Color::from_hex(0x5a5f63);
    for (cx, cz) in [
        (p.x0, p.z0),
        (p.x1, p.z0),
        (p.x0, p.z1),
        (p.x1, p.z1),
    ] {
        b.trim.add_cylinder(
            Vector3::new(cx, KERB, cz),
            0.24,
            0.14,
            11.0,
            7,
            mast,
            true,
            Uv::Unit,
        );
        b.trim.add_box(
            Vector3::new(cx - 1.1, KERB + 11.0, cz - 0.30),
            Vector3::new(cx + 1.1, KERB + 11.7, cz + 0.30),
            Color::from_hex(0x3d4247),
            Uv::Unit,
        );
        b.glow.add_slab(
            cx - 1.05,
            cz - 0.26,
            cx + 1.05,
            cz + 0.26,
            KERB + 10.98,
            Color::from_hex(0xfff2cf),
            Uv::Unit,
        );
    }
    if cell.w() > 30.0 && cell.d() > 30.0 {
        let seat = Color::from_hex(0x7f8489);
        for step in 0..4 {
            let h = 0.45 * (step + 1) as f32;
            let inset = 0.9 * step as f32;
            if along_x {
                b.trim.add_box(
                    Vector3::new(p.x0 + 2.0, KERB, cell.z1 - 3.0 + inset),
                    Vector3::new(p.x1 - 2.0, KERB + h, cell.z1 - 2.1 + inset),
                    scale_color(seat, 1.0 - 0.05 * step as f32),
                    Uv::Unit,
                );
            } else {
                b.trim.add_box(
                    Vector3::new(cell.x1 - 3.0 + inset, KERB, p.z0 + 2.0),
                    Vector3::new(cell.x1 - 2.1 + inset, KERB + h, p.z1 - 2.0),
                    scale_color(seat, 1.0 - 0.05 * step as f32),
                    Uv::Unit,
                );
            }
        }
    }
    // Trees only outside the pitch, in the corners the markings leave over.
    for _ in 0..4 {
        let cx = if rng.chance(0.5) { cell.x0 + 1.5 } else { cell.x1 - 1.5 };
        let cz = if rng.chance(0.5) { cell.z0 + 1.5 } else { cell.z1 - 1.5 };
        add_tree(b, cx, cz, KERB, near, warm, rng);
    }
    out.walks.push((cell.x0 + 1.0, cell.z0 + 1.0, cell.x1 - 1.0, cell.z0 + 2.6));
}

/// Playground: sand, a climbing frame, swings, a slide and a fence.
fn play(b: &mut Batches, cell: &Rect, warm: bool, near: bool, rng: &mut Rng, out: &mut ParkOut) {
    lawn(b, cell, 1.0, rng);
    let yard = cell.inset(2.6);
    let sand = Color::from_hex(C_SAND);
    b.pads.add_ground_disc(
        yard.cx(),
        yard.cz(),
        (yard.w().min(yard.d()) * 0.42).max(3.0),
        14,
        0.14,
        KERB + 0.02,
        sand,
    );
    edge_stubs(b, cell, 1.8, Color::from_hex(C_GRAVEL));

    let steel = Color::from_hex(0x9099a0);
    let paint = [0xc4442b, 0x2f74b5, 0xd8a02a, 0x3f8f57];
    let pick = |rng: &mut Rng| Color::from_hex(paint[rng.below(paint.len())]);
    let (cx, cz) = (yard.cx(), yard.cz());

    // Climbing frame: a cube of bars.
    let s = 2.0f32;
    for (dx, dz) in [(-s, -s), (s, -s), (s, s), (-s, s)] {
        b.trim.add_cylinder(
            Vector3::new(cx + dx, KERB, cz + dz),
            0.06,
            0.06,
            2.5,
            5,
            steel,
            false,
            Uv::Unit,
        );
    }
    for h in [1.0f32, 1.75, 2.5] {
        for (a, c) in [
            ((-s, -s), (s, -s)),
            ((s, -s), (s, s)),
            ((s, s), (-s, s)),
            ((-s, s), (-s, -s)),
        ] {
            let mid = Vector3::new(cx + (a.0 + c.0) * 0.5, KERB + h, cz + (a.1 + c.1) * 0.5);
            let yaw = facing(c.0 - a.0, c.1 - a.1);
            b.trim.add_yaw_box(
                mid,
                Vector3::new(0.05, 0.05, s),
                yaw,
                pick(rng),
                Uv::Unit,
            );
        }
    }

    // Swings: two A-frames, a beam, and seats on chains.
    let sx = cx + yard.w() * 0.28;
    let beam_y = KERB + 2.3;
    for side in [-1.6f32, 1.6] {
        for lean in [-0.55f32, 0.55] {
            b.trim.add_limb(
                Vector3::new(sx + lean, KERB, cz + side),
                Vector3::new(sx, beam_y, cz + side),
                0.06,
                0.05,
                5,
                steel,
                false,
            );
        }
    }
    b.trim.add_yaw_box(
        Vector3::new(sx, beam_y, cz),
        Vector3::new(0.06, 0.06, 1.7),
        0.0,
        steel,
        Uv::Unit,
    );
    for t in [-0.8f32, 0.8] {
        let swing = rng.range(-0.25, 0.25);
        let seat = Vector3::new(sx + swing * 1.3, KERB + 0.55, cz + t);
        b.trim.add_limb(
            Vector3::new(sx, beam_y, cz + t),
            seat,
            0.015,
            0.015,
            4,
            Color::from_hex(0x6b7076),
            false,
        );
        b.trim.add_yaw_box(
            seat,
            Vector3::new(0.22, 0.03, 0.12),
            0.0,
            pick(rng),
            Uv::Unit,
        );
    }

    // Slide: a platform with a ramp off it.
    let lx = cx - yard.w() * 0.28;
    b.trim.add_cylinder(
        Vector3::new(lx, KERB, cz),
        0.07,
        0.07,
        1.6,
        5,
        steel,
        false,
        Uv::Unit,
    );
    b.trim.add_box(
        Vector3::new(lx - 0.55, KERB + 1.55, cz - 0.55),
        Vector3::new(lx + 0.55, KERB + 1.68, cz + 0.55),
        Color::from_hex(C_TIMBER),
        Uv::Unit,
    );
    let chute = pick(rng);
    b.trim.add_limb_flat(
        Vector3::new(lx, KERB + 1.62, cz + 0.4),
        Vector3::new(lx, KERB + 0.25, cz + 3.2),
        0.34,
        0.34,
        0.22,
        4,
        chute,
        true,
    );

    // Roundabout.
    b.trim.add_cylinder(
        Vector3::new(cx, KERB + 0.02, cz - yard.d() * 0.30),
        1.35,
        1.35,
        0.30,
        14,
        pick(rng),
        true,
        Uv::Unit,
    );

    // A low fence with a gap for the gate, and benches for the adults.
    let f = cell.inset(1.4);
    let rail = Color::from_hex(0x4c6b48);
    for (a, c) in [
        ((f.x0, f.z0), (f.x1, f.z0)),
        ((f.x1, f.z0), (f.x1, f.z1)),
        ((f.x1, f.z1), (f.x0, f.z1)),
    ] {
        let yaw = facing(c.0 - a.0, c.1 - a.1);
        let len = ((c.0 - a.0).powi(2) + (c.1 - a.1).powi(2)).sqrt();
        b.trim.add_yaw_box(
            Vector3::new((a.0 + c.0) * 0.5, KERB + 0.62, (a.1 + c.1) * 0.5),
            Vector3::new(0.04, 0.05, len * 0.5),
            yaw,
            rail,
            Uv::Unit,
        );
        for i in 0..(len / 1.8) as usize + 1 {
            let t = i as f32 / ((len / 1.8) as usize).max(1) as f32;
            b.trim.add_cylinder(
                Vector3::new(a.0 + (c.0 - a.0) * t, KERB, a.1 + (c.1 - a.1) * t),
                0.045,
                0.045,
                0.72,
                4,
                rail,
                false,
                Uv::Unit,
            );
        }
    }
    add_bench(b, cell.x0 + 2.4, cell.cz(), facing(1.0, 0.0), rng);
    for _ in 0..3 {
        add_tree(
            b,
            rng.range(cell.x0 + 1.2, cell.x1 - 1.2),
            if rng.chance(0.5) { cell.z0 + 1.2 } else { cell.z1 - 1.2 },
            KERB,
            near,
            warm,
            rng,
        );
    }
    out.gathers.push((cx, cz, yard.w().min(yard.d()) * 0.42));
    out.walks
        .push((cell.x0 + 1.5, cell.z0 + 1.5, cell.x1 - 1.5, cell.z1 - 1.5));
}

/// A hard-landscaped plaza. No grass at all — that is what makes it downtown.
fn plaza(b: &mut Batches, cell: &Rect, warm: bool, near: bool, rng: &mut Rng, out: &mut ParkOut) {
    let stone = Color::from_hex(C_CONCRETE);
    b.pads.add_box(
        Vector3::new(cell.x0, 0.0, cell.z0),
        Vector3::new(cell.x1, KERB, cell.z1),
        scale_color(stone, rng.range(0.94, 1.06)),
        Uv::Unit,
    );
    // Banding: alternate courses of a darker granite, running the short way.
    let along_x = cell.w() < cell.d();
    let n = ((if along_x { cell.w() } else { cell.d() }) / 3.4).max(2.0) as usize;
    let band = scale_color(stone, 0.86);
    for i in (0..n).step_by(2) {
        let (t0, t1) = (i as f32 / n as f32, (i as f32 + 1.0) / n as f32);
        if along_x {
            b.pads.add_slab(
                cell.x0 + t0 * cell.w(),
                cell.z0,
                cell.x0 + t1 * cell.w(),
                cell.z1,
                KERB + 0.006,
                band,
                Uv::Unit,
            );
        } else {
            b.pads.add_slab(
                cell.x0,
                cell.z0 + t0 * cell.d(),
                cell.x1,
                cell.z0 + t1 * cell.d(),
                KERB + 0.006,
                band,
                Uv::Unit,
            );
        }
    }

    let r = (cell.w().min(cell.d()) * 0.17).clamp(2.2, 6.0);
    fountain(b, cell.cx(), cell.cz(), r, rng);
    // Feeding ground, on the paving clear of the basin. Pushing one disc over
    // the fountain itself puts the whole flock inside the bowl.
    for k in 0..4 {
        let a = k as f32 / 4.0 * TAU + PI * 0.25;
        out.gathers.push((
            cell.cx() + (r + 3.0) * a.cos(),
            cell.cz() + (r + 3.0) * a.sin(),
            2.2,
        ));
    }

    // Raised granite planters with one tree in each, on a regular spacing.
    let inset = 3.4;
    let cols = ((cell.w() - inset * 2.0) / 9.0).floor().max(1.0) as usize;
    let rows = ((cell.d() - inset * 2.0) / 9.0).floor().max(1.0) as usize;
    for i in 0..=cols {
        for j in 0..=rows {
            let x = cell.x0 + inset + (cell.w() - inset * 2.0) * i as f32 / cols as f32;
            let z = cell.z0 + inset + (cell.d() - inset * 2.0) * j as f32 / rows as f32;
            if (x - cell.cx()).hypot(z - cell.cz()) < r * 1.5 {
                continue;
            }
            let pr = 1.5;
            b.trim.add_box(
                Vector3::new(x - pr, KERB, z - pr),
                Vector3::new(x + pr, KERB + 0.55, z + pr),
                scale_color(Color::from_hex(0x86837c), rng.range(0.92, 1.08)),
                Uv::Unit,
            );
            b.grass.add_slab(
                x - pr + 0.16,
                z - pr + 0.16,
                x + pr - 0.16,
                z + pr - 0.16,
                KERB + 0.52,
                Color::from_hex(C_SOIL),
                Uv::Unit,
            );
            add_tree_as(
                b,
                x,
                z,
                KERB + 0.5,
                near,
                if warm { Species::Palm } else { Species::Columnar },
                None,
                0.85,
                rng,
            );
        }
    }
    // Bollards along the street edge, and flag poles by the fountain.
    for i in 0..(cell.w() / 4.0) as usize {
        let x = cell.x0 + 2.0 + i as f32 * 4.0;
        for z in [cell.z0 + 1.0, cell.z1 - 1.0] {
            b.trim.add_cylinder(
                Vector3::new(x, KERB, z),
                0.11,
                0.10,
                0.92,
                8,
                Color::from_hex(0x3a3d42),
                true,
                Uv::Unit,
            );
        }
    }
    for s in [-1.0f32, 1.0] {
        let x = cell.cx() + s * (r + 3.2);
        b.trim.add_cylinder(
            Vector3::new(x, KERB, cell.cz()),
            0.10,
            0.07,
            9.0,
            6,
            Color::from_hex(0xb9bcc0),
            true,
            Uv::Unit,
        );
        b.trim.add_yaw_box(
            Vector3::new(x + 0.9, KERB + 8.1, cell.cz()),
            Vector3::new(0.9, 0.55, 0.02),
            0.0,
            Color::from_hex(0xb8453c),
            Uv::Unit,
        );
    }
    for i in 0..4 {
        let a = i as f32 / 4.0 * TAU + PI * 0.25;
        add_bench(
            b,
            cell.cx() + (r + 2.6) * a.cos(),
            cell.cz() + (r + 2.6) * a.sin(),
            facing(-a.cos(), -a.sin()),
            rng,
        );
    }
    out.walks
        .push((cell.x0 + 1.5, cell.z0 + 1.5, cell.x1 - 1.5, cell.z1 - 1.5));
}

/// Community allotment: raised beds on a grid, sheds, a greenhouse.
fn allotment(
    b: &mut Batches,
    cell: &Rect,
    warm: bool,
    near: bool,
    rng: &mut Rng,
    out: &mut ParkOut,
) {
    b.grass.add_box(
        Vector3::new(cell.x0, 0.0, cell.z0),
        Vector3::new(cell.x1, KERB, cell.z1),
        scale_color(Color::from_hex(0x5a5b3a), rng.range(0.92, 1.08)),
        Uv::Unit,
    );
    let p = cell.inset(2.2);
    // Crops. Each plot is one crop, and neighbouring plots are deliberately
    // never the same: an allotment reads as a patchwork or it reads as a field.
    const CROP: [u32; 6] = [0x53702f, 0x7e8b3a, 0x9a5a2c, 0x3f5f38, 0x86913f, 0x6b4f2a];
    let cw = 4.6f32;
    let cols = (p.w() / cw).floor().max(1.0) as usize;
    let rows = (p.d() / cw).floor().max(1.0) as usize;
    let mut last = usize::MAX;
    for i in 0..cols {
        for j in 0..rows {
            let x0 = p.x0 + p.w() * i as f32 / cols as f32;
            let x1 = p.x0 + p.w() * (i + 1) as f32 / cols as f32;
            let z0 = p.z0 + p.d() * j as f32 / rows as f32;
            let z1 = p.z0 + p.d() * (j + 1) as f32 / rows as f32;
            let bed = Rect { x0, z0, x1, z1 }.inset(0.55);
            if bed.w() < 0.6 || bed.d() < 0.6 {
                continue;
            }
            let mut k = rng.below(CROP.len());
            if k == last {
                k = (k + 1) % CROP.len();
            }
            last = k;
            // Timber rim, soil, then the crop just proud of it.
            b.trim.add_box(
                Vector3::new(bed.x0, KERB, bed.z0),
                Vector3::new(bed.x1, KERB + 0.26, bed.z1),
                scale_color(Color::from_hex(C_TIMBER), rng.range(0.85, 1.1)),
                Uv::Unit,
            );
            b.foliage.add_box(
                Vector3::new(bed.x0 + 0.12, KERB + 0.24, bed.z0 + 0.12),
                Vector3::new(bed.x1 - 0.12, KERB + rng.range(0.36, 0.70), bed.z1 - 0.12),
                scale_color(Color::from_hex(CROP[k]), rng.range(0.88, 1.12)),
                Uv::Unit,
            );
            // Bean canes over about one plot in five.
            if rng.chance(0.20) {
                for _ in 0..5 {
                    let bx = rng.range(bed.x0, bed.x1);
                    let bz = rng.range(bed.z0, bed.z1);
                    b.trim.add_limb(
                        Vector3::new(bx, KERB + 0.2, bz),
                        Vector3::new(bed.cx(), KERB + 2.1, bed.cz()),
                        0.025,
                        0.015,
                        3,
                        Color::from_hex(0x9a8a5c),
                        false,
                    );
                }
            }
        }
    }
    // Sheds and a greenhouse along one edge.
    for i in 0..3 {
        let x = cell.x0 + 3.0 + i as f32 * (cell.w() - 6.0) / 3.0;
        let z = cell.z1 - 2.4;
        if i == 1 {
            let glass = Color::from_hex(0xa8c6cc);
            b.trim.add_box(
                Vector3::new(x - 1.4, KERB, z - 1.0),
                Vector3::new(x + 1.4, KERB + 1.7, z + 1.0),
                glass,
                Uv::Unit,
            );
            b.trim.add_gable(
                Vector3::new(x - 1.5, KERB + 1.7, z - 1.1),
                Vector3::new(x + 1.5, KERB + 1.7, z + 1.1),
                0.55,
                glass,
            );
        } else {
            let wood = scale_color(Color::from_hex(0x6a5136), rng.range(0.85, 1.15));
            b.trim.add_box(
                Vector3::new(x - 1.1, KERB, z - 0.9),
                Vector3::new(x + 1.1, KERB + 1.9, z + 0.9),
                wood,
                Uv::Unit,
            );
            b.trim.add_gable(
                Vector3::new(x - 1.2, KERB + 1.9, z - 1.0),
                Vector3::new(x + 1.2, KERB + 1.9, z + 1.0),
                0.5,
                scale_color(wood, 0.8),
            );
            // Water butt.
            b.trim.add_cylinder(
                Vector3::new(x + 1.5, KERB, z),
                0.35,
                0.35,
                1.1,
                8,
                Color::from_hex(0x2f4a35),
                true,
                Uv::Unit,
            );
        }
    }
    let _ = (warm, near);
    edge_stubs(b, cell, 1.6, Color::from_hex(C_GRAVEL));
    out.walks
        .push((cell.x0 + 1.0, cell.z0 + 1.0, cell.x1 - 1.0, cell.z1 - 1.0));
}

/// Rough grass and wildflowers with one mown path through it.
fn meadow(b: &mut Batches, cell: &Rect, warm: bool, near: bool, rng: &mut Rng, out: &mut ParkOut) {
    lawn(b, cell, 1.10, rng);
    // Drifts of long grass: overlapping patches slightly proud of the sward,
    // which is enough to break the flat green without any extra material.
    for _ in 0..(cell.area() / 90.0) as usize {
        let cx = rng.range(cell.x0, cell.x1);
        let cz = rng.range(cell.z0, cell.z1);
        b.grass.add_ground_disc(
            cx,
            cz,
            rng.range(2.5, 7.0),
            7,
            0.30,
            KERB + 0.01,
            scale_color(Color::from_hex(0x6d7a3c), rng.range(0.85, 1.15)),
        );
    }
    // Wildflowers: tiny bright squares, thick enough to read as a haze of
    // colour from a distance and as individual flowers from the path.
    const WILD: [u32; 4] = [0xd8c73a, 0xc44a5e, 0xe4e0d2, 0x8f6fbf];
    let flowers = if near { 260 } else { 60 };
    for _ in 0..flowers {
        let fx = rng.range(cell.x0, cell.x1);
        let fz = rng.range(cell.z0, cell.z1);
        let s = rng.range(0.10, 0.22);
        b.foliage.add_slab(
            fx - s,
            fz - s,
            fx + s,
            fz + s,
            KERB + rng.range(0.25, 0.55),
            Color::from_hex(WILD[rng.below(WILD.len())]),
            Uv::Unit,
        );
    }
    // The mown path: a wandering strip of short grass, not gravel.
    let along_x = cell.w() >= cell.d();
    let phase = rng.range(0.0, TAU);
    let amp = (if along_x { cell.d() } else { cell.w() }) * 0.24;
    let pts: Vec<(f32, f32)> = (0..=12)
        .map(|i| {
            let t = i as f32 / 12.0;
            let w = (phase + t * TAU * 0.9).sin() * amp;
            if along_x {
                (cell.x0 + t * cell.w(), cell.cz() + w)
            } else {
                (cell.cx() + w, cell.z0 + t * cell.d())
            }
        })
        .collect();
    b.grass.add_ground_path(
        &pts,
        2.4,
        KERB + 0.02,
        scale_color(Color::from_hex(C_GRASS), 1.06),
        Uv::Unit,
    );
    for _ in 0..(cell.area() / 260.0) as usize {
        add_tree(
            b,
            rng.range(cell.x0 + 2.0, cell.x1 - 2.0),
            rng.range(cell.z0 + 2.0, cell.z1 - 2.0),
            KERB,
            near,
            warm,
            rng,
        );
    }
    out.walks
        .push((cell.x0 + 2.0, cell.z0 + 2.0, cell.x1 - 2.0, cell.z1 - 2.0));
}

/// Dense trees with a path threaded between them.
fn grove(b: &mut Batches, cell: &Rect, warm: bool, near: bool, rng: &mut Rng, out: &mut ParkOut) {
    lawn(b, cell, 0.90, rng);
    let path = Color::from_hex(C_SOIL);
    let along_x = cell.w() >= cell.d();
    let phase = rng.range(0.0, TAU);
    let amp = (if along_x { cell.d() } else { cell.w() }) * 0.20;
    let pts: Vec<(f32, f32)> = (0..=10)
        .map(|i| {
            let t = i as f32 / 10.0;
            let w = (phase + t * TAU * 1.3).sin() * amp;
            if along_x {
                (cell.x0 + t * cell.w(), cell.cz() + w)
            } else {
                (cell.cx() + w, cell.z0 + t * cell.d())
            }
        })
        .collect();
    b.pads.add_ground_path(&pts, 1.8, PATH_Y, path, Uv::Unit);
    edge_stubs(b, cell, 1.8, path);

    // One dominant species with a scatter of others, which is what a planted
    // grove looks like — a random mix reads as scrub.
    let lead = match rng.below(4) {
        0 => Species::Conifer,
        1 => Species::Columnar,
        2 => Species::Weeping,
        _ => Species::Broadleaf,
    };
    let count = (cell.area() / 26.0) as usize;
    for _ in 0..count {
        let tx = rng.range(cell.x0 + 1.5, cell.x1 - 1.5);
        let tz = rng.range(cell.z0 + 1.5, cell.z1 - 1.5);
        // Keep the path walkable.
        if pts
            .iter()
            .any(|p| (p.0 - tx).hypot(p.1 - tz) < 2.2)
        {
            continue;
        }
        if rng.chance(0.72) {
            add_tree_as(b, tx, tz, KERB, near, lead, None, rng.range(0.85, 1.25), rng);
        } else {
            add_tree(b, tx, tz, KERB, near, warm, rng);
        }
    }
    for i in [3usize, 7] {
        if i < pts.len() {
            add_bench(b, pts[i].0, pts[i].1 + 1.5, facing(0.0, -1.0), rng);
        }
    }
    if rng.chance(0.7) {
        let i = 2 + rng.below(pts.len() - 3);
        add_park_furniture(b, pts[i].0 + 1.4, pts[i].1, rng);
    }
    out.walks
        .push((cell.x0 + 2.0, cell.z0 + 2.0, cell.x1 - 2.0, cell.z1 - 2.0));
}

// ---------------------------------------------------------------------------
// Park buildings and fittings.
// ---------------------------------------------------------------------------

/// A bandstand: a raised octagonal deck, columns, and a conical roof.
///
/// The centrepiece a Victorian park was built around, and the one park
/// structure that is unmistakable from the air — an octagon with a point on it
/// is a bandstand and nothing else.
pub(crate) fn add_bandstand(b: &mut Batches, cx: f32, cz: f32, r: f32, rng: &mut Rng) {
    let stone = scale_color(Color::from_hex(0x9a968c), rng.range(0.92, 1.08));
    let iron = scale_color(Color::from_hex(0x2f4a3c), rng.range(0.85, 1.15));
    let roof = scale_color(Color::from_hex(0x4a5560), rng.range(0.9, 1.1));
    let deck = KERB + 0.85;
    b.trim
        .add_cylinder(Vector3::new(cx, KERB, cz), r + 0.30, r + 0.20, 0.70, 8, stone, true, Uv::Unit);
    b.trim
        .add_cylinder(Vector3::new(cx, KERB + 0.70, cz), r, r, 0.15, 8, scale_color(stone, 1.08), true, Uv::Unit);
    // Steps down one side.
    for k in 0..3 {
        let t = k as f32;
        b.trim.add_box(
            Vector3::new(cx - 0.9, KERB + 0.22 * t, cz + r + 0.30 + 0.30 * (2.0 - t)),
            Vector3::new(cx + 0.9, KERB + 0.22 * (t + 1.0), cz + r + 0.60 + 0.30 * (2.0 - t)),
            stone,
            Uv::Unit,
        );
    }
    let posts = 8;
    let eaves = deck + 2.75;
    for i in 0..posts {
        let a = i as f32 / posts as f32 * TAU;
        let (px, pz) = (cx + r * 0.88 * a.cos(), cz + r * 0.88 * a.sin());
        b.trim
            .add_cylinder(Vector3::new(px, deck, pz), 0.075, 0.065, 2.75, 6, iron, false, Uv::Unit);
        // Fretwork bracket at the head of each column.
        let a2 = (i + 1) as f32 / posts as f32 * TAU;
        let (qx, qz) = (cx + r * 0.88 * a2.cos(), cz + r * 0.88 * a2.sin());
        b.trim.add_limb(
            Vector3::new(px, eaves - 0.12, pz),
            Vector3::new(qx, eaves - 0.12, qz),
            0.045,
            0.045,
            4,
            iron,
            false,
        );
        // Balustrade, left open where the steps arrive.
        if a.sin() < 0.75 {
            b.trim.add_limb(
                Vector3::new(px, deck + 0.55, pz),
                Vector3::new(qx, deck + 0.55, qz),
                0.05,
                0.05,
                4,
                iron,
                false,
            );
        }
    }
    // Roof: a cone of triangles with a finial on top.
    let apex = Vector3::new(cx, eaves + 1.5, cz);
    for i in 0..posts {
        let (a0, a1) = (
            i as f32 / posts as f32 * TAU,
            (i + 1) as f32 / posts as f32 * TAU,
        );
        let rr = r * 1.18;
        let p0 = [cx + rr * a0.cos(), eaves, cz + rr * a0.sin()];
        let p1 = [cx + rr * a1.cos(), eaves, cz + rr * a1.sin()];
        let mid = ((a0 + a1) * 0.5).cos();
        b.trim.tri(
            [[apex.x, apex.y, apex.z], p1, p0],
            [mid, 0.55, ((a0 + a1) * 0.5).sin()],
            [[0.5, 0.5]; 3],
            [roof; 3],
        );
    }
    b.trim.add_cylinder(
        Vector3::new(cx, eaves + 1.5, cz),
        0.07,
        0.03,
        0.55,
        5,
        scale_color(roof, 1.3),
        true,
        Uv::Unit,
    );
}

/// A park kiosk with tables and parasols outside it.
pub(crate) fn add_kiosk(b: &mut Batches, cx: f32, cz: f32, rng: &mut Rng) {
    let wood = scale_color(Color::from_hex(0x6d5238), rng.range(0.9, 1.1));
    let roof = scale_color(Color::from_hex(0x3f4a44), rng.range(0.9, 1.1));
    b.trim.add_box(
        Vector3::new(cx - 1.7, KERB, cz - 1.3),
        Vector3::new(cx + 1.7, KERB + 2.5, cz + 1.3),
        wood,
        Uv::Unit,
    );
    // Serving hatch, lit from inside after dark.
    b.glow.add_box(
        Vector3::new(cx - 1.15, KERB + 1.05, cz + 1.28),
        Vector3::new(cx + 1.15, KERB + 1.95, cz + 1.36),
        Color::from_hex(0xffe2ac),
        Uv::Unit,
    );
    b.trim.add_gable(
        Vector3::new(cx - 2.0, KERB + 2.5, cz - 1.6),
        Vector3::new(cx + 2.0, KERB + 2.5, cz + 1.6),
        0.75,
        roof,
    );
    // Awning over the hatch, on two struts.
    b.trim.add_box(
        Vector3::new(cx - 1.8, KERB + 2.35, cz + 1.3),
        Vector3::new(cx + 1.8, KERB + 2.45, cz + 2.9),
        scale_color(Color::from_hex(0xb04a3a), rng.range(0.9, 1.1)),
        Uv::Unit,
    );
    for s in [-1.0f32, 1.0] {
        b.trim.add_cylinder(
            Vector3::new(cx + s * 1.7, KERB, cz + 2.8),
            0.05,
            0.05,
            2.35,
            5,
            wood,
            false,
            Uv::Unit,
        );
    }

    // Tables with parasols, which is what makes it read as a cafe rather than
    // as a shed.
    for i in 0..(2 + rng.below(3)) {
        let a = rng.range(0.0, TAU);
        let d = rng.range(3.2, 6.0);
        let (tx, tz) = (cx + d * a.cos(), cz + 2.5 + d * a.sin() * 0.5);
        b.trim.add_cylinder(
            Vector3::new(tx, KERB, tz),
            0.06,
            0.06,
            0.72,
            5,
            Color::from_hex(0x54585d),
            false,
            Uv::Unit,
        );
        b.trim.add_cylinder(
            Vector3::new(tx, KERB + 0.72, tz),
            0.42,
            0.42,
            0.05,
            10,
            scale_color(Color::from_hex(0xb9b3a6), rng.range(0.9, 1.1)),
            true,
            Uv::Unit,
        );
        for k in 0..3 {
            let ca = a + k as f32 * TAU / 3.0;
            b.trim.add_cylinder(
                Vector3::new(tx + 0.75 * ca.cos(), KERB, tz + 0.75 * ca.sin()),
                0.16,
                0.16,
                0.44,
                6,
                Color::from_hex(0x4b5f52),
                true,
                Uv::Unit,
            );
        }
        if i % 2 == 0 {
            // Parasol: a mast and a shallow cone.
            b.trim.add_cylinder(
                Vector3::new(tx, KERB, tz),
                0.035,
                0.035,
                2.15,
                5,
                Color::from_hex(0x9a8f7d),
                false,
                Uv::Unit,
            );
            let shade = mix_color(
                Color::from_hex(0xd8d2c2),
                Color::from_hex(0xc06a4a),
                rng.f(),
            );
            const N: usize = 8;
            for k in 0..N {
                let (a0, a1) = (
                    k as f32 / N as f32 * TAU,
                    (k + 1) as f32 / N as f32 * TAU,
                );
                let rr = 1.25;
                b.trim.tri(
                    [
                        [tx, KERB + 2.35, tz],
                        [tx + rr * a1.cos(), KERB + 1.95, tz + rr * a1.sin()],
                        [tx + rr * a0.cos(), KERB + 1.95, tz + rr * a0.sin()],
                    ],
                    [0.0, 1.0, 0.0],
                    [[0.5, 0.5]; 3],
                    [shade; 3],
                );
            }
        }
    }
}

/// Cast-iron railings round a park, with a gap for the gate.
pub(crate) fn add_railings(b: &mut Batches, r: &Rect, gate_side: usize, rng: &mut Rng) {
    let iron = scale_color(Color::from_hex(0x24312c), rng.range(0.85, 1.15));
    let h = 1.15;
    let sides = [
        ((r.x0, r.z0), (r.x1, r.z0)),
        ((r.x1, r.z0), (r.x1, r.z1)),
        ((r.x1, r.z1), (r.x0, r.z1)),
        ((r.x0, r.z1), (r.x0, r.z0)),
    ];
    for (i, (a, c)) in sides.iter().enumerate() {
        let len = ((c.0 - a.0).powi(2) + (c.1 - a.1).powi(2)).sqrt();
        let n = (len / 0.42).round().max(2.0) as usize;
        // The gate is a gap in the middle of one side, with a pier each way.
        let gap = if i == gate_side { 0.18f32 } else { -1.0 };
        for k in 0..=n {
            let t = k as f32 / n as f32;
            if (t - 0.5).abs() < gap {
                continue;
            }
            let (px, pz) = (a.0 + (c.0 - a.0) * t, a.1 + (c.1 - a.1) * t);
            b.trim.add_cylinder(
                Vector3::new(px, KERB, pz),
                0.022,
                0.018,
                h,
                4,
                iron,
                false,
                Uv::Unit,
            );
            // Finial.
            b.trim.add_cylinder(
                Vector3::new(px, KERB + h, pz),
                0.030,
                0.004,
                0.10,
                4,
                iron,
                true,
                Uv::Unit,
            );
        }
        // Top and bottom rails, broken at the gate.
        for (t0, t1) in if gap > 0.0 {
            vec![(0.0, 0.5 - gap), (0.5 + gap, 1.0)]
        } else {
            vec![(0.0, 1.0)]
        } {
            for y in [KERB + 0.18, KERB + h - 0.10] {
                b.trim.add_limb(
                    Vector3::new(a.0 + (c.0 - a.0) * t0, y, a.1 + (c.1 - a.1) * t0),
                    Vector3::new(a.0 + (c.0 - a.0) * t1, y, a.1 + (c.1 - a.1) * t1),
                    0.024,
                    0.024,
                    4,
                    iron,
                    false,
                );
            }
        }
        // Gate piers.
        if gap > 0.0 {
            for s in [-1.0f32, 1.0] {
                let t = 0.5 + s * gap;
                let (px, pz) = (a.0 + (c.0 - a.0) * t, a.1 + (c.1 - a.1) * t);
                b.trim.add_box(
                    Vector3::new(px - 0.20, KERB, pz - 0.20),
                    Vector3::new(px + 0.20, KERB + 1.75, pz + 0.20),
                    scale_color(Color::from_hex(0x8d8880), rng.range(0.9, 1.1)),
                    Uv::Unit,
                );
            }
        }
    }
}

/// The small stuff that lines a path: bins, drinking fountains, signposts and
/// bicycle stands. Individually trivial; collectively the difference between a
/// lawn with a path on it and a park.
pub(crate) fn add_park_furniture(b: &mut Batches, x: f32, z: f32, rng: &mut Rng) {
    if !b.occ.try_spot(x, z, 0.9) {
        return;
    }
    let iron = Color::from_hex(0x2a352f);
    match rng.below(5) {
        // Litter bin: a slatted drum on a post.
        0 => {
            b.trim
                .add_cylinder(Vector3::new(x, KERB, z), 0.09, 0.09, 0.55, 5, iron, false, Uv::Unit);
            b.trim.add_cylinder(
                Vector3::new(x, KERB + 0.55, z),
                0.28,
                0.31,
                0.62,
                9,
                scale_color(iron, 1.25),
                true,
                Uv::Unit,
            );
        }
        // Drinking fountain.
        1 => {
            b.trim
                .add_cylinder(Vector3::new(x, KERB, z), 0.16, 0.13, 0.85, 8, Color::from_hex(0x7d8a80), true, Uv::Unit);
            b.trim.add_cylinder(
                Vector3::new(x, KERB + 0.85, z),
                0.20,
                0.20,
                0.10,
                9,
                Color::from_hex(0x93a099),
                true,
                Uv::Unit,
            );
            b.trim.add_limb(
                Vector3::new(x, KERB + 0.92, z - 0.14),
                Vector3::new(x, KERB + 1.10, z),
                0.020,
                0.016,
                4,
                Color::from_hex(0xa8b2ab),
                false,
            );
        }
        // Fingerpost.
        2 => {
            b.trim
                .add_cylinder(Vector3::new(x, KERB, z), 0.055, 0.045, 2.15, 5, iron, false, Uv::Unit);
            let yaw = rng.range(0.0, TAU);
            for k in 0..2 {
                b.trim.add_yaw_box(
                    Vector3::new(x, KERB + 1.75 - 0.28 * k as f32, z),
                    Vector3::new(0.42, 0.075, 0.02),
                    yaw + k as f32 * 1.9,
                    Color::from_hex(0xd8d2c4),
                    Uv::Unit,
                );
            }
        }
        // Bicycle stands: a row of hoops.
        3 => {
            let yaw = rng.range(0.0, TAU);
            for k in 0..3 {
                let off = (k as f32 - 1.0) * 0.85;
                let (bx, bz) = (x + off * yaw.cos(), z + off * yaw.sin());
                const N: usize = 6;
                let mut prev = Vector3::new(bx - 0.34 * yaw.sin(), KERB, bz + 0.34 * yaw.cos());
                for i in 1..=N {
                    let a = PI * i as f32 / N as f32;
                    let p = Vector3::new(
                        bx - 0.34 * a.cos() * yaw.sin(),
                        KERB + 0.72 * a.sin(),
                        bz + 0.34 * a.cos() * yaw.cos(),
                    );
                    b.trim.add_limb(prev, p, 0.026, 0.026, 4, Color::from_hex(0x5a6a62), false);
                    prev = p;
                }
            }
        }
        // Notice board.
        _ => {
            let yaw = rng.range(0.0, TAU);
            for s in [-1.0f32, 1.0] {
                b.trim.add_cylinder(
                    Vector3::new(x + s * 0.5 * yaw.cos(), KERB, z + s * 0.5 * yaw.sin()),
                    0.045,
                    0.040,
                    1.35,
                    4,
                    Color::from_hex(0x5c4632),
                    false,
                    Uv::Unit,
                );
            }
            b.trim.add_yaw_box(
                Vector3::new(x, KERB + 1.15, z),
                Vector3::new(0.62, 0.42, 0.04),
                facing(-yaw.sin(), yaw.cos()),
                Color::from_hex(0x2f5b4a),
                Uv::Unit,
            );
        }
    }
}

/// A run of chain-link between two points: rails top, middle and bottom.
///
/// Drawn as a solid panel first, which at any distance reads as sheet metal —
/// a compound wall rather than a fence you can watch a dog through. Three thin
/// rails cost about the same and read correctly, and the gap between them is
/// what says *mesh*.
fn fence_run(b: &mut Batches, a: (f32, f32), c: (f32, f32), h: f32, col: Color) {
    let len = ((c.0 - a.0).powi(2) + (c.1 - a.1).powi(2)).sqrt();
    if len < 0.1 {
        return;
    }
    for t in [0.12f32, 0.52, 0.94] {
        b.trim.add_limb(
            Vector3::new(a.0, KERB + h * t, a.1),
            Vector3::new(c.0, KERB + h * t, c.1),
            0.022,
            0.022,
            4,
            col,
            false,
        );
    }
    // A sparse set of uprights between the posts, which is what carries the
    // impression of mesh at a distance without drawing any.
    let n = (len / 0.75).round().max(1.0) as usize;
    for k in 1..n {
        let t = k as f32 / n as f32;
        b.trim.add_limb(
            Vector3::new(a.0 + (c.0 - a.0) * t, KERB + h * 0.10, a.1 + (c.1 - a.1) * t),
            Vector3::new(a.0 + (c.0 - a.0) * t, KERB + h * 0.96, a.1 + (c.1 - a.1) * t),
            0.010,
            0.010,
            3,
            col,
            false,
        );
    }
}

/// A fenced dog run: gravel, agility equipment, a double gate, and dogs off
/// the lead.
///
/// The double gate is the giveaway. Every dog run has an airlock — two gates
/// with a pen between them so nothing gets out while someone comes in — and
/// nothing else in a park is built that way.
fn dog_run(b: &mut Batches, cell: &Rect, warm: bool, near: bool, rng: &mut Rng, out: &mut ParkOut) {
    lawn(b, cell, 0.96, rng);
    let yard = cell.inset(2.2);
    if yard.w() < 8.0 || yard.d() < 8.0 {
        return play(b, cell, warm, near, rng, out);
    }
    // Worn ground inside the fence: grass does not survive this.
    b.pads.add_slab(
        yard.x0,
        yard.z0,
        yard.x1,
        yard.z1,
        KERB + 0.015,
        scale_color(Color::from_hex(0x8a7f68), rng.range(0.92, 1.08)),
        Uv::Unit,
    );

    let mesh = Color::from_hex(0x77817c);
    let post = Color::from_hex(0x36403c);
    let h = 1.35;
    let sides = [
        ((yard.x0, yard.z0), (yard.x1, yard.z0)),
        ((yard.x1, yard.z0), (yard.x1, yard.z1)),
        ((yard.x1, yard.z1), (yard.x0, yard.z1)),
        ((yard.x0, yard.z1), (yard.x0, yard.z0)),
    ];
    let gate_side = rng.below(4);
    for (i, (a, c)) in sides.iter().enumerate() {
        let len = ((c.0 - a.0).powi(2) + (c.1 - a.1).powi(2)).sqrt();
        let n = (len / 2.4).round().max(2.0) as usize;
        for k in 0..=n {
            let t = k as f32 / n as f32;
            let (px, pz) = (a.0 + (c.0 - a.0) * t, a.1 + (c.1 - a.1) * t);
            b.trim
                .add_cylinder(Vector3::new(px, KERB, pz), 0.055, 0.050, h, 5, post, false, Uv::Unit);
        }
        // The mesh itself: a thin panel, drawn as a low box so it reads as a
        // barrier without needing anything transparent.
        let gap = if i == gate_side { 0.16f32 } else { -1.0 };
        for (t0, t1) in if gap > 0.0 {
            vec![(0.0, 0.5 - gap), (0.5 + gap, 1.0)]
        } else {
            vec![(0.0, 1.0)]
        } {
            let (ax, az) = (a.0 + (c.0 - a.0) * t0, a.1 + (c.1 - a.1) * t0);
            let (bx, bz) = (a.0 + (c.0 - a.0) * t1, a.1 + (c.1 - a.1) * t1);
            fence_run(b, (ax, az), (bx, bz), h, mesh);
        }
        // The airlock: a small pen outside the gap with a second gate in it.
        if gap > 0.0 {
            let (mx, mz) = ((a.0 + c.0) * 0.5, (a.1 + c.1) * 0.5);
            let (ox, oz) = (mx - cell.cx(), mz - cell.cz());
            let l = ox.hypot(oz).max(1e-3);
            let (nx, nz) = (ox / l, oz / l);
            let (tx, tz) = (-nz, nx);
            let pen = 1.6f32;
            for s in [-1.0f32, 1.0] {
                let (px, pz) = (mx + tx * pen * s, mz + tz * pen * s);
                fence_run(
                    b,
                    (px, pz),
                    (px + nx * pen * 2.0, pz + nz * pen * 2.0),
                    h,
                    mesh,
                );
            }
            fence_run(
                b,
                (mx + nx * pen * 2.0 - tx * pen, mz + nz * pen * 2.0 - tz * pen),
                (mx + nx * pen * 2.0 + tx * pen, mz + nz * pen * 2.0 + tz * pen),
                h,
                mesh,
            );
        }
    }

    // Agility equipment: an A-frame, a row of weave poles, and a hoop.
    let (cx, cz) = (yard.cx(), yard.cz());
    let paint = [0xc4442b, 0x2f74b5, 0xd8a02a];
    let pick = |rng: &mut Rng| Color::from_hex(paint[rng.below(paint.len())]);
    let ramp = pick(rng);
    for s in [-1.0f32, 1.0] {
        b.trim.add_limb_flat(
            Vector3::new(cx + s * 1.5, KERB, cz - yard.d() * 0.22),
            Vector3::new(cx, KERB + 1.25, cz - yard.d() * 0.22),
            0.55,
            0.55,
            0.10,
            4,
            ramp,
            true,
        );
    }
    for k in 0..6 {
        b.trim.add_cylinder(
            Vector3::new(cx - 2.5 + k as f32, KERB, cz + yard.d() * 0.18),
            0.045,
            0.040,
            1.0,
            5,
            pick(rng),
            false,
            Uv::Unit,
        );
    }
    // Hoop on two legs.
    let hoop = pick(rng);
    let (hx, hz) = (cx + yard.w() * 0.26, cz);
    for s in [-1.0f32, 1.0] {
        b.trim.add_cylinder(
            Vector3::new(hx, KERB, hz + s * 0.62),
            0.05,
            0.05,
            1.55,
            5,
            Color::from_hex(0x6b7570),
            false,
            Uv::Unit,
        );
    }
    const N: usize = 12;
    for i in 0..N {
        let (a0, a1) = (
            i as f32 / N as f32 * TAU,
            (i + 1) as f32 / N as f32 * TAU,
        );
        let r = 0.55;
        b.trim.add_limb(
            Vector3::new(hx, KERB + 1.0 + r * a0.sin(), hz + r * a0.cos()),
            Vector3::new(hx, KERB + 1.0 + r * a1.sin(), hz + r * a1.cos()),
            0.035,
            0.035,
            4,
            hoop,
            false,
        );
    }

    for _ in 0..2 {
        add_bench(
            b,
            rng.range(yard.x0 + 1.0, yard.x1 - 1.0),
            yard.z0 + 0.8,
            facing(0.0, 1.0),
            rng,
        );
    }
    add_park_furniture(b, yard.x0 + 1.2, yard.z1 - 1.2, rng);
    for _ in 0..3 {
        add_tree(
            b,
            rng.range(cell.x0 + 1.0, cell.x1 - 1.0),
            if rng.chance(0.5) { cell.z0 + 1.0 } else { cell.z1 - 1.0 },
            KERB,
            near,
            warm,
            rng,
        );
    }
    out.runs
        .push((cx, cz, yard.w().min(yard.d()) * 0.38));
    out.walks
        .push((yard.x0 + 1.0, yard.z0 + 1.0, yard.x1 - 1.0, yard.z1 - 1.0));
}

/// A hard court: tarmac, a net or a pair of hoops, and a high fence.
///
/// The fence is what makes it read. A tennis court without one is a blue
/// rectangle; with one it is unmistakable from three streets away.
fn hard_court(
    b: &mut Batches,
    cell: &Rect,
    p: &Rect,
    near: bool,
    warm: bool,
    rng: &mut Rng,
    out: &mut ParkOut,
) {
    let tennis = rng.chance(0.55);
    let surface = if tennis {
        scale_color(Color::from_hex(0x2f6b52), rng.range(0.9, 1.1))
    } else {
        scale_color(Color::from_hex(0x4a5560), rng.range(0.9, 1.1))
    };
    b.pads
        .add_slab(p.x0, p.z0, p.x1, p.z1, KERB + 0.02, surface, Uv::Unit);
    let white = Color::from_hex(0xe4e0d6);
    let m = p.inset(1.0);
    let my = KERB + 0.035;
    b.paint.add_ground_path(
        &[
            (m.x0, m.z0),
            (m.x1, m.z0),
            (m.x1, m.z1),
            (m.x0, m.z1),
            (m.x0, m.z0),
            (m.x1, m.z0),
        ],
        0.12,
        my,
        white,
        Uv::Unit,
    );
    let along_x = m.w() >= m.d();
    if tennis {
        // Service lines and a net across the middle.
        if along_x {
            b.paint
                .add_ground_path(&[(m.cx(), m.z0), (m.cx(), m.z1)], 0.10, my, white, Uv::Unit);
            for s in [-1.0f32, 1.0] {
                let x = m.cx() + s * m.w() * 0.22;
                b.paint
                    .add_ground_path(&[(x, m.z0 + 0.8), (x, m.z1 - 0.8)], 0.10, my, white, Uv::Unit);
            }
        } else {
            b.paint
                .add_ground_path(&[(m.x0, m.cz()), (m.x1, m.cz())], 0.10, my, white, Uv::Unit);
            for s in [-1.0f32, 1.0] {
                let z = m.cz() + s * m.d() * 0.22;
                b.paint
                    .add_ground_path(&[(m.x0 + 0.8, z), (m.x1 - 0.8, z)], 0.10, my, white, Uv::Unit);
            }
        }
        let net = Color::from_hex(0x2b2f33);
        let (a, c) = if along_x {
            ((m.cx(), m.z0), (m.cx(), m.z1))
        } else {
            ((m.x0, m.cz()), (m.x1, m.cz()))
        };
        let len = ((c.0 - a.0).powi(2) + (c.1 - a.1).powi(2)).sqrt();
        b.trim.add_yaw_box(
            Vector3::new((a.0 + c.0) * 0.5, KERB + 0.52, (a.1 + c.1) * 0.5),
            Vector3::new(0.02, 0.45, len * 0.5),
            facing(c.0 - a.0, c.1 - a.1),
            net,
            Uv::Unit,
        );
        for (px, pz) in [a, c] {
            b.trim.add_cylinder(
                Vector3::new(px, KERB, pz),
                0.05,
                0.05,
                1.07,
                5,
                Color::from_hex(0x5a625f),
                false,
                Uv::Unit,
            );
        }
    } else {
        // Basketball: a key at each end and a backboard on a post.
        ring_path(&mut b.paint, m.cx(), m.cz(), (m.w().min(m.d()) * 0.16).min(1.8), 0.10, my, white, 16);
        for s in [0.0f32, 1.0] {
            let (bx, bz) = if along_x {
                (m.x0 + s * m.w(), m.cz())
            } else {
                (m.cx(), m.z0 + s * m.d())
            };
            let inward = if s == 0.0 { 1.0f32 } else { -1.0 };
            let (ix, iz) = if along_x { (inward, 0.0) } else { (0.0, inward) };
            b.trim.add_cylinder(
                Vector3::new(bx - ix * 0.3, KERB, bz - iz * 0.3),
                0.07,
                0.06,
                3.05,
                5,
                Color::from_hex(0x62696b),
                false,
                Uv::Unit,
            );
            b.trim.add_yaw_box(
                Vector3::new(bx + ix * 0.5, KERB + 3.15, bz + iz * 0.5),
                Vector3::new(0.03, 0.55, 0.9),
                facing(ix, iz),
                Color::from_hex(0xdad6cc),
                Uv::Unit,
            );
            const N: usize = 9;
            for k in 0..N {
                let (a0, a1) = (
                    k as f32 / N as f32 * TAU,
                    (k + 1) as f32 / N as f32 * TAU,
                );
                let r = 0.23;
                let c = (bx + ix * 1.0, bz + iz * 1.0);
                // A hoop is a horizontal circle. Scaling one axis by the
                // backboard's facing collapses it to a line whenever that
                // component is zero, which is half the courts in the city.
                b.trim.add_limb(
                    Vector3::new(c.0 + r * a0.cos(), KERB + 3.05, c.1 + r * a0.sin()),
                    Vector3::new(c.0 + r * a1.cos(), KERB + 3.05, c.1 + r * a1.sin()),
                    0.02,
                    0.02,
                    3,
                    Color::from_hex(0xd06a2a),
                    false,
                );
            }
        }
    }

    // The fence: posts and a mesh panel, three metres up.
    let f = p.inset(-0.6);
    let fh = 3.2;
    let mesh_c = Color::from_hex(0x848c8d);
    let sides = [
        ((f.x0, f.z0), (f.x1, f.z0)),
        ((f.x1, f.z0), (f.x1, f.z1)),
        ((f.x1, f.z1), (f.x0, f.z1)),
        ((f.x0, f.z1), (f.x0, f.z0)),
    ];
    let gate = rng.below(4);
    for (i, (a, c)) in sides.iter().enumerate() {
        let len = ((c.0 - a.0).powi(2) + (c.1 - a.1).powi(2)).sqrt();
        let n = (len / 3.0).round().max(2.0) as usize;
        for k in 0..=n {
            let t = k as f32 / n as f32;
            b.trim.add_cylinder(
                Vector3::new(a.0 + (c.0 - a.0) * t, KERB, a.1 + (c.1 - a.1) * t),
                0.06,
                0.05,
                fh,
                5,
                Color::from_hex(0x3d4547),
                false,
                Uv::Unit,
            );
        }
        let gap = if i == gate { 0.14f32 } else { -1.0 };
        for (t0, t1) in if gap > 0.0 {
            vec![(0.0, 0.5 - gap), (0.5 + gap, 1.0)]
        } else {
            vec![(0.0, 1.0)]
        } {
            let (ax, az) = (a.0 + (c.0 - a.0) * t0, a.1 + (c.1 - a.1) * t0);
            let (bx, bz) = (a.0 + (c.0 - a.0) * t1, a.1 + (c.1 - a.1) * t1);
            fence_run(b, (ax, az), (bx, bz), fh, mesh_c);
        }
    }
    add_bench(b, cell.x0 + 1.6, cell.cz(), facing(1.0, 0.0), rng);
    for _ in 0..3 {
        add_tree(
            b,
            rng.range(cell.x0 + 1.0, cell.x1 - 1.0),
            if rng.chance(0.5) { cell.z0 + 1.0 } else { cell.z1 - 1.0 },
            KERB,
            near,
            warm,
            rng,
        );
    }
    out.walks
        .push((cell.x0 + 1.0, cell.z0 + 1.0, cell.x1 - 1.0, cell.z0 + 2.4));
}
