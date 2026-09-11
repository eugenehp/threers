//! Part of the `simcity` example; see `mod.rs`.
//!
//! Where the city shops, and where it leaves the car.
//!
//! Both halves of that are missing from almost every procedural city, and they
//! are missing together, because they are the same thing: retail at this scale
//! is a shed with a car park in front of it, and the car park is usually the
//! larger of the two. Leaving it out is why generated cities read as European
//! old towns no matter what you put in them — the ground between buildings is
//! all pavement and grass, never asphalt with white lines on it.
//!
//! The three retail forms here are genuinely different plans, not the same box
//! at three sizes: a strip is a long thin row facing its own parking, a
//! supermarket is one deep box with a service yard behind, and a mall is a
//! blank hulk with skylights and parking decks stuck to the side.
#![allow(dead_code)]

use super::*;

/// Bay dimensions. Everything in a car park is derived from these.
const BAY_W: f32 = 2.6;
const BAY_L: f32 = 5.0;
const AISLE: f32 = 6.4;

/// A surface car park: painted bays in double rows either side of an aisle,
/// with cars in some of them.
///
/// `fill` is the fraction of bays occupied, which is the one number that says
/// what time of day it is out here.
pub(crate) fn add_parking_lot(
    b: &mut Batches,
    r: &Rect,
    along_x: bool,
    fill: f32,
    bodies: &[MeshBuilder],
    paints: &[Color],
    rng: &mut Rng,
) -> usize {
    if r.w() < 14.0 || r.d() < 14.0 {
        return 0;
    }
    b.pads.add_slab(
        r.x0,
        r.z0,
        r.x1,
        r.z1,
        KERB + 0.02,
        scale_color(Color::from_hex(0x4a4a4c), rng.range(0.92, 1.08)),
        Uv::Unit,
    );
    // A module is two rows of bays nose to nose with an aisle beside them.
    let module = BAY_L * 2.0 + AISLE;
    let (span, run) = if along_x { (r.d(), r.w()) } else { (r.w(), r.d()) };
    let modules = (span / module).floor().max(1.0) as usize;
    let bays = (run / BAY_W).floor() as usize;
    let mut cars = 0;
    for m in 0..modules {
        for side in 0..2 {
            // Where this row of bays sits across the lot, and which way the
            // cars in it face.
            let v = m as f32 * module + if side == 0 { 0.0 } else { BAY_L + AISLE * 0.0 };
            let v = v + BAY_L * 0.5;
            for k in 0..bays {
                let u = (k as f32 + 0.5) * BAY_W;
                let (x, z) = if along_x {
                    (r.x0 + u, r.z0 + v)
                } else {
                    (r.x0 + v, r.z0 + u)
                };
                if x > r.x1 - 0.5 || z > r.z1 - 0.5 {
                    continue;
                }
                // The bay line: one stripe on the left of each bay, which is
                // how they are actually painted.
                let (lx, lz, lx1, lz1) = if along_x {
                    (x - BAY_W * 0.5, z - BAY_L * 0.5, x - BAY_W * 0.5 + 0.12, z + BAY_L * 0.5)
                } else {
                    (x - BAY_L * 0.5, z - BAY_W * 0.5, x + BAY_L * 0.5, z - BAY_W * 0.5 + 0.12)
                };
                b.paint
                    .add_slab(lx, lz, lx1, lz1, KERB + 0.04, Color::from_hex(0xd8d8d0), Uv::Unit);
                if bodies.is_empty() || rng.f() > fill {
                    continue;
                }
                let kind = rng.below(bodies.len());
                let yaw = if along_x { 0.0 } else { PI * 0.5 } + if side == 0 { 0.0 } else { PI };
                let (hw, hd) = if along_x {
                    (1.1 * VEHICLE_SCALE, 2.4 * VEHICLE_SCALE)
                } else {
                    (2.4 * VEHICLE_SCALE, 1.1 * VEHICLE_SCALE)
                };
                if !b.occ.try_claim([x - hw, z - hd, x + hw, z + hd]) {
                    continue;
                }
                b.parked.append_at(
                    &bodies[kind],
                    Vector3::new(x, KERB + 0.02, z),
                    yaw,
                    VEHICLE_SCALE,
                    paints[rng.below(paints.len())],
                );
                cars += 1;
            }
        }
    }
    // Lighting columns down the aisles, and a tree island or two — the things
    // that stop a lot reading as a grey rectangle.
    let masts = (run / 26.0).floor().max(1.0) as usize;
    for i in 0..masts {
        let u = (i as f32 + 0.5) / masts as f32 * run;
        for m in 0..modules {
            let v = m as f32 * module + BAY_L;
            let (x, z) = if along_x { (r.x0 + u, r.z0 + v) } else { (r.x0 + v, r.z0 + u) };
            b.trim.add_limb(
                Vector3::new(x, KERB, z),
                Vector3::new(x, 8.5, z),
                0.16,
                0.11,
                5,
                Color::from_hex(0x4a4f55),
                false,
            );
            b.glow.add_box(
                Vector3::new(x - 0.7, 8.2, z - 0.4),
                Vector3::new(x + 0.7, 8.5, z + 0.4),
                Color::from_hex(0xffeec4),
                Uv::Unit,
            );
        }
    }
    cars
}

/// A multi-storey car park.
///
/// Open decks with a spandrel band along each one, columns you can see between
/// them, a ramp on one end, and cars on the decks. The open sides are the
/// point: a solid box with windows is an office, and the thing that says "car
/// park" is being able to see straight through it at every level.
pub(crate) fn add_parking_deck(
    b: &mut Batches,
    r: &Rect,
    decks: usize,
    bodies: &[MeshBuilder],
    paints: &[Color],
    rng: &mut Rng,
) -> bool {
    if r.w() < 24.0 || r.d() < 24.0 {
        return false;
    }
    if !b.occ.try_claim([r.x0, r.z0, r.x1, r.z1]) {
        return false;
    }
    let floor = 3.1f32;
    let concrete = scale_color(Color::from_hex(0xa9a69e), rng.range(0.92, 1.08));
    for d in 0..decks {
        let y = KERB + d as f32 * floor;
        // The slab.
        b.trim.add_box(
            Vector3::new(r.x0, y, r.z0),
            Vector3::new(r.x1, y + 0.35, r.z1),
            concrete,
            Uv::Unit,
        );
        // Spandrel: a waist-high band round the edge, leaving the gap above it
        // open.
        for (sx0, sz0, sx1, sz1) in [
            (r.x0, r.z0, r.x1, r.z0 + 0.4),
            (r.x0, r.z1 - 0.4, r.x1, r.z1),
            (r.x0, r.z0, r.x0 + 0.4, r.z1),
            (r.x1 - 0.4, r.z0, r.x1, r.z1),
        ] {
            b.trim.add_box(
                Vector3::new(sx0, y + 0.35, sz0),
                Vector3::new(sx1, y + 1.35, sz1),
                scale_color(concrete, 0.94),
                Uv::Unit,
            );
        }
        // Cars on the deck, in two rows.
        if d + 1 < decks && !bodies.is_empty() {
            let rows = 2;
            for row in 0..rows {
                let z = mix(r.z0 + 5.0, r.z1 - 5.0, (row as f32 + 0.5) / rows as f32);
                let n = ((r.w() - 8.0) / BAY_W) as usize;
                for k in 0..n {
                    if rng.chance(0.45) {
                        continue;
                    }
                    let x = r.x0 + 4.0 + (k as f32 + 0.5) * BAY_W;
                    b.parked.append_at(
                        &bodies[rng.below(bodies.len())],
                        Vector3::new(x, y + 0.35, z),
                        if row == 0 { 0.0 } else { PI },
                        VEHICLE_SCALE,
                        paints[rng.below(paints.len())],
                    );
                }
            }
        }
    }
    // Corner and mid columns, visible through the open sides.
    let top = KERB + decks as f32 * floor;
    let nx = ((r.w() / 8.0).round() as usize).max(2);
    let nz = ((r.d() / 8.0).round() as usize).max(2);
    for i in 0..=nx {
        for j in 0..=nz {
            if i != 0 && i != nx && j != 0 && j != nz {
                continue;
            }
            let x = mix(r.x0 + 0.5, r.x1 - 0.5, i as f32 / nx as f32);
            let z = mix(r.z0 + 0.5, r.z1 - 0.5, j as f32 / nz as f32);
            b.trim.add_box(
                Vector3::new(x - 0.35, KERB, z - 0.35),
                Vector3::new(x + 0.35, top, z + 0.35),
                scale_color(concrete, 0.9),
                Uv::Unit,
            );
        }
    }
    // Stair and lift core on one corner: the only solid part of the building.
    b.trim.add_box(
        Vector3::new(r.x1 - 7.0, KERB, r.z0),
        Vector3::new(r.x1, top + 2.6, r.z0 + 6.0),
        scale_color(concrete, 1.05),
        Uv::Unit,
    );
    true
}

/// A strip mall: a long single-storey row of units behind a shared canopy,
/// facing its own car park.
pub(crate) fn add_strip_mall(
    b: &mut Batches,
    r: &Rect,
    along_x: bool,
    rng: &mut Rng,
) -> bool {
    if r.w() < 30.0 || r.d() < 30.0 {
        return false;
    }
    // The building takes the back third; the rest is parking.
    let depth = (if along_x { r.d() } else { r.w() }) * 0.34;
    let shop = if along_x {
        Rect { x0: r.x0, z0: r.z1 - depth, x1: r.x1, z1: r.z1 }
    } else {
        Rect { x0: r.x1 - depth, z0: r.z0, x1: r.x1, z1: r.z1 }
    };
    // Claim the *building*, not the site. Claiming the whole rectangle is what
    // it looks like it should do, and it means the car park in front can never
    // claim a bay — every lot came out empty and the only clue was the counter
    // reading zero.
    if !b.occ.try_claim([shop.x0, shop.z0, shop.x1, shop.z1]) {
        return false;
    }
    let h = 5.4;
    // Units, each a different colour, because a strip is let unit by unit and
    // every tenant paints their own frontage.
    let run = if along_x { shop.w() } else { shop.d() };
    let units = ((run / 12.0).round() as usize).max(2);
    const FRONT: [u32; 6] = [0xc9c2b4, 0xb08a6a, 0x8fa3ad, 0xc4a05c, 0xa8b09a, 0xbd8f8a];
    for i in 0..units {
        let (u0, u1) = (i as f32 / units as f32, (i + 1) as f32 / units as f32);
        let unit = if along_x {
            Rect {
                x0: mix(shop.x0, shop.x1, u0),
                z0: shop.z0,
                x1: mix(shop.x0, shop.x1, u1),
                z1: shop.z1,
            }
        } else {
            Rect {
                x0: shop.x0,
                z0: mix(shop.z0, shop.z1, u0),
                x1: shop.x1,
                z1: mix(shop.z0, shop.z1, u1),
            }
        };
        let col = scale_color(Color::from_hex(FRONT[rng.below(FRONT.len())]), rng.range(0.9, 1.1));
        b.trim.add_box(
            Vector3::new(unit.x0, KERB, unit.z0),
            Vector3::new(unit.x1, h, unit.z1),
            col,
            Uv::Unit,
        );
        // Parapet, one course taller than the roof and a shade darker: the
        // flat-roof-with-a-raised-front that every strip in the world has.
        b.trim.add_box(
            Vector3::new(unit.x0, h, unit.z0),
            Vector3::new(unit.x1, h + 1.1, unit.z1),
            scale_color(col, 0.82),
            Uv::Unit,
        );
        // Fascia sign on the front face.
        let sign = scale_color(
            Color::from_hex([0xd04a3a, 0x2f6ab5, 0x3f8f5c, 0xd8a52e][rng.below(4)]),
            rng.range(0.9, 1.1),
        );
        let (fx0, fz0, fx1, fz1) = if along_x {
            (unit.x0 + 1.0, unit.z0 - 0.12, unit.x1 - 1.0, unit.z0)
        } else {
            (unit.x0 - 0.12, unit.z0 + 1.0, unit.x0, unit.z1 - 1.0)
        };
        // A backlit fascia is a *coloured panel* by day and a lit one by
        // night. On `neon` alone it was neither: that batch is not hidden in
        // daylight, it is drawn with its colour scaled by `sky.lights`, so
        // every shop in the row had a black board over the door and the whole
        // parade read as derelict. The lit panel carries the colour; `glow`,
        // which *is* hidden by day, carries the light.
        b.trim.add_box(
            Vector3::new(fx0, h - 1.4, fz0),
            Vector3::new(fx1, h + 0.5, fz1),
            sign,
            Uv::Unit,
        );
        b.glow.add_box(
            Vector3::new(fx0, h - 1.45, fz0 - 0.05),
            Vector3::new(fx1, h + 0.55, fz1 + 0.05),
            scale_color(sign, 1.25),
            Uv::Unit,
        );
        // Glazing under the sign.
        b.trim.add_box(
            Vector3::new(fx0, KERB, fz0 - 0.05),
            Vector3::new(fx1, h - 1.6, fz1),
            Color::from_hex(0x2a3238),
            Uv::Unit,
        );
    }
    // The canopy over the walkway, on posts.
    let (cx0, cz0, cx1, cz1) = if along_x {
        (shop.x0, shop.z0 - 3.2, shop.x1, shop.z0)
    } else {
        (shop.x0 - 3.2, shop.z0, shop.x0, shop.z1)
    };
    b.trim.add_box(
        Vector3::new(cx0, h - 0.9, cz0),
        Vector3::new(cx1, h - 0.5, cz1),
        Color::from_hex(0x6f747a),
        Uv::Unit,
    );
    let posts = (run / 8.0) as usize;
    for i in 0..=posts {
        let t = i as f32 / posts.max(1) as f32;
        let (px, pz) = if along_x {
            (mix(cx0, cx1, t), cz0 + 0.4)
        } else {
            (cx0 + 0.4, mix(cz0, cz1, t))
        };
        b.trim.add_limb(
            Vector3::new(px, KERB, pz),
            Vector3::new(px, h - 0.9, pz),
            0.11,
            0.11,
            4,
            Color::from_hex(0x6f747a),
            false,
        );
    }
    true
}

/// A supermarket: one deep box, a big sign, trolley bays, and a service yard
/// with a loading dock at the back.
pub(crate) fn add_grocery(b: &mut Batches, r: &Rect, along_x: bool, rng: &mut Rng) -> bool {
    if r.w() < 34.0 || r.d() < 34.0 {
        return false;
    }
    let depth = (if along_x { r.d() } else { r.w() }) * 0.46;
    let shop = if along_x {
        Rect { x0: r.x0 + 2.0, z0: r.z1 - depth, x1: r.x1 - 2.0, z1: r.z1 - 2.0 }
    } else {
        Rect { x0: r.x1 - depth, z0: r.z0 + 2.0, x1: r.x1 - 2.0, z1: r.z1 - 2.0 }
    };
    if !b.occ.try_claim([shop.x0, shop.z0, shop.x1, shop.z1]) {
        return false;
    }
    let h = 8.2;
    let brand = Color::from_hex([0x2f6ab5, 0x3f8f5c, 0xd04a3a, 0xd8a52e][rng.below(4)]);
    b.trim.add_box(
        Vector3::new(shop.x0, KERB, shop.z0),
        Vector3::new(shop.x1, h, shop.z1),
        scale_color(Color::from_hex(0xd8d4c8), rng.range(0.94, 1.06)),
        Uv::Unit,
    );
    // A band of brand colour round the parapet — how every one of these is
    // liveried.
    b.trim.add_box(
        Vector3::new(shop.x0 - 0.2, h, shop.z0 - 0.2),
        Vector3::new(shop.x1 + 0.2, h + 1.6, shop.z1 + 0.2),
        brand,
        Uv::Unit,
    );
    // Roof plant: the condensers and ducts a shed this size is covered in.
    for _ in 0..6 + rng.below(6) {
        let (px, pz) = (
            rng.range(shop.x0 + 3.0, shop.x1 - 3.0),
            rng.range(shop.z0 + 3.0, shop.z1 - 3.0),
        );
        let (w, d, ph) = (rng.range(1.6, 3.4), rng.range(1.6, 3.0), rng.range(0.8, 1.8));
        b.trim.add_box(
            Vector3::new(px - w * 0.5, h, pz - d * 0.5),
            Vector3::new(px + w * 0.5, h + ph, pz + d * 0.5),
            Color::from_hex(0x9aa0a6),
            Uv::Unit,
        );
    }
    // Shopfront: glazing and a sign the size of a bus.
    let (fx0, fz0, fx1, fz1) = if along_x {
        (shop.x0 + 2.0, shop.z0 - 0.15, shop.x1 - 2.0, shop.z0)
    } else {
        (shop.x0 - 0.15, shop.z0 + 2.0, shop.x0, shop.z1 - 2.0)
    };
    b.trim.add_box(
        Vector3::new(fx0, KERB, fz0),
        Vector3::new(fx1, 4.6, fz1),
        Color::from_hex(0x2a3238),
        Uv::Unit,
    );
    b.trim.add_box(
        Vector3::new(mix(fx0, fx1, 0.28), 5.4, fz0 - 0.06),
        Vector3::new(mix(fx0, fx1, 0.72), 7.4, fz1),
        brand,
        Uv::Unit,
    );
    b.glow.add_box(
        Vector3::new(mix(fx0, fx1, 0.275), 5.35, fz0 - 0.11),
        Vector3::new(mix(fx0, fx1, 0.725), 7.45, fz1 + 0.05),
        scale_color(brand, 1.3),
        Uv::Unit,
    );
    // Entrance canopy, trolley bays out in the lot.
    for i in 0..3 {
        let t = 0.2 + i as f32 * 0.3;
        let (bx, bz) = if along_x {
            (mix(r.x0 + 6.0, r.x1 - 6.0, t), shop.z0 - 16.0)
        } else {
            (shop.x0 - 16.0, mix(r.z0 + 6.0, r.z1 - 6.0, t))
        };
        if bx < r.x0 || bz < r.z0 {
            continue;
        }
        let (bw, bd) = if along_x { (1.4f32, 5.0f32) } else { (5.0, 1.4) };
        b.trim.add_box(
            Vector3::new(bx - bw, 2.4, bz - bd),
            Vector3::new(bx + bw, 2.7, bz + bd),
            Color::from_hex(0x8a9096),
            Uv::Unit,
        );
        for s in [-1.0f32, 1.0] {
            b.trim.add_limb(
                Vector3::new(bx + s * bw * 0.8, KERB, bz + s * bd * 0.8),
                Vector3::new(bx + s * bw * 0.8, 2.4, bz + s * bd * 0.8),
                0.08,
                0.08,
                4,
                Color::from_hex(0x8a9096),
                false,
            );
        }
    }
    true
}

/// A shopping mall: a hulk with skylights, entrance pavilions and a deck.
///
/// The distinguishing feature of a mall from the air is that it has almost no
/// windows and a very large roof, and that the roof is the interesting part —
/// rooflights over the malls themselves, plant everywhere else. Two or three
/// levels, so it stands above the strip retail around it.
pub(crate) fn add_mall(
    b: &mut Batches,
    r: &Rect,
    levels: usize,
    bodies: &[MeshBuilder],
    paints: &[Color],
    rng: &mut Rng,
) -> bool {
    // A small mall is still a mall. The floor was set at the size the first
    // one happened to be given, which meant a dense layout with no room for
    // that got no mall at all.
    if r.w() < 58.0 || r.d() < 44.0 {
        return false;
    }
    // The building takes two thirds; the rest gets a deck and surface parking.
    let split = r.x0 + r.w() * 0.66;
    let hull = Rect { x0: r.x0, z0: r.z0, x1: split - 6.0, z1: r.z1 };
    if !b.occ.try_claim([hull.x0, hull.z0, hull.x1, hull.z1]) {
        return false;
    }
    let h = KERB + levels as f32 * 5.6;
    let wall = scale_color(Color::from_hex(0xbfb6a6), rng.range(0.94, 1.06));
    b.trim.add_box(
        Vector3::new(hull.x0, KERB, hull.z0),
        Vector3::new(hull.x1, h, hull.z1),
        wall,
        Uv::Unit,
    );
    // A stepped-back top level on part of the footprint, so it is not one
    // extruded rectangle.
    b.trim.add_box(
        Vector3::new(mix(hull.x0, hull.x1, 0.18), h, mix(hull.z0, hull.z1, 0.15)),
        Vector3::new(mix(hull.x0, hull.x1, 0.72), h + 5.0, mix(hull.z0, hull.z1, 0.85)),
        scale_color(wall, 0.95),
        Uv::Unit,
    );
    // Rooflights: two runs down the length, which is where the malls are.
    for k in 0..2 {
        let t = 0.3 + k as f32 * 0.4;
        let z = mix(hull.z0, hull.z1, t);
        b.glow.add_box(
            Vector3::new(hull.x0 + 6.0, h + 0.1, z - 2.2),
            Vector3::new(hull.x1 - 6.0, h + 1.9, z + 2.2),
            Color::from_hex(0xd8e4ee),
            Uv::Unit,
        );
    }
    // Roof plant.
    for _ in 0..10 + rng.below(10) {
        let (px, pz) = (
            rng.range(hull.x0 + 4.0, hull.x1 - 4.0),
            rng.range(hull.z0 + 4.0, hull.z1 - 4.0),
        );
        let (w, d, ph) = (rng.range(2.0, 5.0), rng.range(2.0, 4.0), rng.range(1.0, 2.4));
        b.trim.add_box(
            Vector3::new(px - w * 0.5, h, pz - d * 0.5),
            Vector3::new(px + w * 0.5, h + ph, pz + d * 0.5),
            Color::from_hex(0x9aa0a6),
            Uv::Unit,
        );
    }
    // Entrance pavilions: glazed, taller than the wall, one per long side.
    for s in [0.0f32, 1.0] {
        let z = mix(hull.z0 + 12.0, hull.z1 - 12.0, s);
        b.trim.add_box(
            Vector3::new(hull.x1 - 1.0, KERB, z - 8.0),
            Vector3::new(hull.x1 + 6.0, h * 0.72, z + 8.0),
            Color::from_hex(0x2f3a42),
            Uv::Unit,
        );
        b.trim.add_box(
            Vector3::new(hull.x1 + 6.0, h * 0.30, z - 6.0),
            Vector3::new(hull.x1 + 6.2, h * 0.52, z + 6.0),
            Color::from_hex(0xe8d9a8),
            Uv::Unit,
        );
        b.glow.add_box(
            Vector3::new(hull.x1 + 5.95, h * 0.295, z - 6.1),
            Vector3::new(hull.x1 + 6.3, h * 0.525, z + 6.1),
            Color::from_hex(0xfff0c8),
            Uv::Unit,
        );
    }
    // The parking deck alongside, and surface bays in front of it.
    let deck = Rect { x0: split, z0: r.z0 + 4.0, x1: r.x1, z1: mix(r.z0, r.z1, 0.55) };
    add_parking_deck(b, &deck, 3, bodies, paints, rng);
    let lot = Rect { x0: split, z0: mix(r.z0, r.z1, 0.58), x1: r.x1, z1: r.z1 - 4.0 };
    add_parking_lot(b, &lot, true, 0.55, bodies, paints, rng);
    true
}
