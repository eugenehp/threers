//! Part of the `simcity` example; see `mod.rs`.
//!
//! Residential streets that are not part of the grid.
//!
//! Every road in this city was one of two families of parallel bands, which
//! means every road went somewhere and every block was a rectangle bounded by
//! four through routes. Real cities are like that in the middle and nothing
//! like it at the edge: the outskirts are culs-de-sac, crescents and loops,
//! laid out precisely so that traffic *cannot* pass through them, and they are
//! the single clearest signal of "suburb" from the air.
//!
//! The two forms here are the ones that do the work. A close is a stub with a
//! turning head — the bulb is the tell, because nothing on a grid is round. A
//! crescent leaves a street and rejoins it, bowing out around a green. Both
//! are drawn inside a block the grid has already handed over, so nothing about
//! the axis machinery has to change: from the outside the block still has four
//! straight sides, and inside it stops being a rectangle of lots.
#![allow(dead_code)]

use super::*;

/// Carriageway width of a residential street. Narrow — that is half of why a
/// close reads as one.
const LANE: f32 = 5.0;

/// A footway and a carriageway along a polyline.
///
/// Everything here is laid *on top of* the block's pavement slab, not cut into
/// it. The through streets sit below kerb level because they run over bare
/// ground; a close runs inside a block, and the block is a solid pad up to
/// `KERB`, so a carriageway at the streets' height is a carriageway nobody can
/// see. The kerb reads from the colour change instead of from the step.
fn kerbed(b: &mut Batches, pts: &[(f32, f32)], width: f32, rng: &mut Rng) {
    b.pads.add_ground_path(
        pts,
        width + 3.4,
        KERB + 0.01,
        scale_color(Color::from_hex(C_SIDEWALK), rng.range(0.98, 1.06)),
        Uv::Unit,
    );
    b.road.add_ground_path(
        pts,
        width,
        KERB + 0.03,
        scale_color(Color::from_hex(0x3c3c40), rng.range(0.94, 1.06)),
        Uv::Unit,
    );
}

/// One house on its plot, with the things a house has and a block of flats
/// does not: a drive, a garage, a hedge and a lawn.
fn dwelling(b: &mut Batches, plan: &LayoutPlan, plot: Rect, front: (f32, f32), rng: &mut Rng) -> bool {
    if plot.w() < 8.0 || plot.d() < 8.0 {
        return false;
    }
    // Front garden: the house sits back from the road, which is the other half
    // of why a suburb does not look like a terrace. Never so deep that the
    // house left behind it falls under the two-bay minimum `add_building`
    // enforces — a garden that eats the house is how a whole close came out
    // with a road, a turning head and no dwellings on it.
    let short = plot.d().min(plot.w());
    let depth = (short * 0.22).clamp(1.4, 6.0).min((short - 7.4).max(0.8));
    let (fx, fz) = front;
    let house = Rect {
        x0: plot.x0 + if fx < 0.0 { depth } else { 1.0 },
        z0: plot.z0 + if fz < 0.0 { depth } else { 1.0 },
        x1: plot.x1 - if fx > 0.0 { depth } else { 1.0 },
        z1: plot.z1 - if fz > 0.0 { depth } else { 1.0 },
    };
    if house.w() < 4.5 || house.d() < 4.5 {
        return false;
    }
    // Lawn over the whole plot, then the drive over that.
    b.grass.add_slab(
        plot.x0,
        plot.z0,
        plot.x1,
        plot.z1,
        KERB + 0.02,
        scale_color(Color::from_hex(0x6f9a4a), rng.range(0.85, 1.15)),
        Uv::Unit,
    );
    // The drive runs from the frontage to the house, on one side of the plot.
    let side = rng.chance(0.5);
    let dw = 3.0f32.min(plot.w().min(plot.d()) * 0.34);
    let drive = if fx.abs() > fz.abs() {
        let z = if side { plot.z0 + dw * 0.5 + 0.6 } else { plot.z1 - dw * 0.5 - 0.6 };
        Rect {
            x0: if fx < 0.0 { plot.x0 } else { house.x1 },
            z0: z - dw * 0.5,
            x1: if fx < 0.0 { house.x0 } else { plot.x1 },
            z1: z + dw * 0.5,
        }
    } else {
        let x = if side { plot.x0 + dw * 0.5 + 0.6 } else { plot.x1 - dw * 0.5 - 0.6 };
        Rect {
            x0: x - dw * 0.5,
            z0: if fz < 0.0 { plot.z0 } else { house.z1 },
            x1: x + dw * 0.5,
            z1: if fz < 0.0 { house.z0 } else { plot.z1 },
        }
    };
    if drive.w() > 0.5 && drive.d() > 0.5 {
        b.pads.add_slab(
            drive.x0,
            drive.z0,
            drive.x1,
            drive.z1,
            KERB + 0.05,
            scale_color(Color::from_hex(0x84807a), rng.range(0.9, 1.1)),
            Uv::Unit,
        );
    }
    // The house goes through the ordinary builder where it can, at the
    // outskirts end of the zone scale, so it gets the forms, pitched roofs and
    // window idioms the rest of the city already has.
    //
    // Where it cannot, it gets a cottage instead. `add_building` snaps
    // frontages to the three-metre bay grid and refuses anything under two
    // bays, which is right for a street of shops and wrong for a plot at the
    // end of a close: several closes came out with a road, a turning head and
    // a single house on them because every other plot was a few centimetres
    // short of a bay.
    if !add_building(b, plan, house, 1.15, false, rng) {
        if !b.occ.try_claim([house.x0 - 0.4, house.z0 - 0.4, house.x1 + 0.4, house.z1 + 0.4]) {
            return false;
        }
        const WALL: [u32; 6] = [0xd8cdb8, 0xc4a58a, 0xb8bcc0, 0xd0b9a0, 0xa8b0a4, 0xcfc0a8];
        const ROOF: [u32; 4] = [0x6b4038, 0x4a4f55, 0x7a5a3c, 0x59463c];
        // One in six gets a flat roof and a terrace instead of a pitch: the
        // modern infill that every older suburb has a few of.
        let terrace = rng.chance(0.17);
        let storeys = if rng.chance(0.35) { 1.0f32 } else { 2.0 };
        let h = KERB + storeys * 2.9;
        let wall = scale_color(Color::from_hex(WALL[rng.below(WALL.len())]), rng.range(0.9, 1.1));
        b.trim.add_box(
            Vector3::new(house.x0, KERB, house.z0),
            Vector3::new(house.x1, h, house.z1),
            wall,
            Uv::Unit,
        );
        if terrace {
            // Parapet, decking, planters and a table on the roof.
            b.trim.add_box(
                Vector3::new(house.x0 - 0.2, h, house.z0 - 0.2),
                Vector3::new(house.x1 + 0.2, h + 0.9, house.z1 + 0.2),
                scale_color(wall, 0.9),
                Uv::Unit,
            );
            b.pads.add_slab(
                house.x0 + 0.3,
                house.z0 + 0.3,
                house.x1 - 0.3,
                house.z1 - 0.3,
                h + 0.06,
                Color::from_hex(0xa98a62),
                Uv::Unit,
            );
            for k in 0..3 {
                let t = (k as f32 + 0.5) / 3.0;
                let x = mix(house.x0 + 0.8, house.x1 - 0.8, t);
                b.foliage.add_box(
                    Vector3::new(x - 0.4, h + 0.06, house.z0 + 0.5),
                    Vector3::new(x + 0.4, h + 0.8, house.z0 + 1.2),
                    Color::from_hex(0x4a6f2c),
                    Uv::Unit,
                );
            }
            b.trim.add_box(
                Vector3::new(house.cx() - 0.6, h + 0.06, house.cz() - 0.6),
                Vector3::new(house.cx() + 0.6, h + 0.72, house.cz() + 0.6),
                Color::from_hex(0xe4dccc),
                Uv::Unit,
            );
        } else {
            b.trim.add_gable(
                Vector3::new(house.x0 - 0.35, h, house.z0 - 0.35),
                Vector3::new(house.x1 + 0.35, h, house.z1 + 0.35),
                rng.range(1.4, 2.6),
                scale_color(Color::from_hex(ROOF[rng.below(ROOF.len())]), rng.range(0.9, 1.1)),
            );
        }
        // Windows on the frontage, and a door under them.
        let glass = Color::from_hex(0x2a3238);
        let (wx, wz) = (fx, fz);
        let (ax, az) = if wx.abs() > wz.abs() {
            (if wx < 0.0 { house.x0 } else { house.x1 }, 0.0f32)
        } else {
            (0.0, if wz < 0.0 { house.z0 } else { house.z1 })
        };
        for s in 0..storeys as usize {
            let y = KERB + 0.9 + s as f32 * 2.9;
            if wx.abs() > wz.abs() {
                for k in 0..2 {
                    let z = mix(house.z0 + 1.0, house.z1 - 1.0, (k as f32 + 0.5) / 2.0);
                    b.trim.add_box(
                        Vector3::new(ax - 0.06, y, z - 0.7),
                        Vector3::new(ax + 0.06, y + 1.3, z + 0.7),
                        glass,
                        Uv::Unit,
                    );
                }
            } else {
                for k in 0..2 {
                    let x = mix(house.x0 + 1.0, house.x1 - 1.0, (k as f32 + 0.5) / 2.0);
                    b.trim.add_box(
                        Vector3::new(x - 0.7, y, az - 0.06),
                        Vector3::new(x + 0.7, y + 1.3, az + 0.06),
                        glass,
                        Uv::Unit,
                    );
                }
            }
        }
    }
    // A garage at the head of the drive on about half of them.
    if rng.chance(0.55) && drive.w() > 2.2 && drive.d() > 2.2 {
        let g = if fx.abs() > fz.abs() {
            Rect {
                x0: if fx < 0.0 { house.x0 - 5.4 } else { house.x1 },
                z0: drive.cz() - 2.6,
                x1: if fx < 0.0 { house.x0 } else { house.x1 + 5.4 },
                z1: drive.cz() + 2.6,
            }
        } else {
            Rect {
                x0: drive.cx() - 2.6,
                z0: if fz < 0.0 { house.z0 - 5.4 } else { house.z1 },
                x1: drive.cx() + 2.6,
                z1: if fz < 0.0 { house.z0 } else { house.z1 + 5.4 },
            }
        };
        if b.occ.try_claim([g.x0, g.z0, g.x1, g.z1]) {
            let wall = scale_color(Color::from_hex(0xc9c0ae), rng.range(0.85, 1.15));
            b.trim.add_box(
                Vector3::new(g.x0, KERB, g.z0),
                Vector3::new(g.x1, KERB + 2.5, g.z1),
                wall,
                Uv::Unit,
            );
            b.trim.add_gable(
                Vector3::new(g.x0 - 0.3, KERB + 2.5, g.z0 - 0.3),
                Vector3::new(g.x1 + 0.3, KERB + 2.5, g.z1 + 0.3),
                rng.range(0.6, 1.2),
                scale_color(Color::from_hex(0x6b4038), rng.range(0.85, 1.15)),
            );
        }
    }
    // --- The back garden. What is behind a house is most of what tells one
    // suburb from another, and from above it is the only part of a house you
    // actually see much of.
    let back = if fx.abs() > fz.abs() {
        Rect {
            x0: if fx < 0.0 { house.x1 } else { plot.x0 },
            z0: plot.z0 + 0.5,
            x1: if fx < 0.0 { plot.x1 } else { house.x0 },
            z1: plot.z1 - 0.5,
        }
    } else {
        Rect {
            x0: plot.x0 + 0.5,
            z0: if fz < 0.0 { house.z1 } else { plot.z0 },
            x1: plot.x1 - 0.5,
            z1: if fz < 0.0 { plot.z1 } else { house.z0 },
        }
    };
    if back.w() > 3.0 && back.d() > 3.0 {
        let roll = rng.f();
        if roll < 0.16 && back.w() > 7.0 && back.d() > 6.0 {
            // A pool, with a paved surround and a couple of loungers.
            let pool = back.inset(1.6);
            b.pads.add_slab(
                back.x0,
                back.z0,
                back.x1,
                back.z1,
                KERB + 0.03,
                Color::from_hex(0xd6d0c2),
                Uv::Unit,
            );
            b.water.add_slab(
                pool.x0,
                pool.z0,
                pool.x1,
                pool.z1,
                KERB + 0.05,
                Color::from_hex(0x2f8fc4),
                Uv::Unit,
            );
            for k in 0..2 {
                let x = mix(back.x0 + 0.8, back.x1 - 0.8, (k as f32 + 0.5) / 2.0);
                b.trim.add_box(
                    Vector3::new(x - 0.35, KERB + 0.03, back.z0 + 0.3),
                    Vector3::new(x + 0.35, KERB + 0.45, back.z0 + 1.6),
                    Color::from_hex(0xe8e4d8),
                    Uv::Unit,
                );
            }
        } else if roll < 0.44 {
            // A vegetable garden: beds in rows, which is unmistakable from
            // above and is what half the gardens in any suburb are.
            let along_x = back.w() >= back.d();
            let n = ((if along_x { back.d() } else { back.w() }) / 1.6) as usize;
            for i in 0..n {
                let t = (i as f32 + 0.5) / n as f32;
                let c = scale_color(
                    Color::from_hex([0x5f7a34u32, 0x6f8a3c, 0x7a6b3c][rng.below(3)]),
                    rng.range(0.85, 1.15),
                );
                if along_x {
                    let z = mix(back.z0, back.z1, t);
                    b.foliage.add_box(
                        Vector3::new(back.x0 + 0.4, KERB + 0.02, z - 0.5),
                        Vector3::new(back.x1 - 0.4, KERB + 0.42, z + 0.5),
                        c,
                        Uv::Unit,
                    );
                } else {
                    let x = mix(back.x0, back.x1, t);
                    b.foliage.add_box(
                        Vector3::new(x - 0.5, KERB + 0.02, back.z0 + 0.4),
                        Vector3::new(x + 0.5, KERB + 0.42, back.z1 - 0.4),
                        c,
                        Uv::Unit,
                    );
                }
            }
        } else if roll < 0.68 {
            // A patio with a table, and a tree at the bottom of the garden.
            let patio = Rect {
                x0: back.x0 + 0.4,
                z0: back.z0 + 0.4,
                x1: back.x0 + (back.w() * 0.55).min(5.0),
                z1: back.z0 + (back.d() * 0.55).min(5.0),
            };
            b.pads.add_slab(
                patio.x0,
                patio.z0,
                patio.x1,
                patio.z1,
                KERB + 0.03,
                scale_color(Color::from_hex(0xb8ab96), rng.range(0.9, 1.1)),
                Uv::Unit,
            );
            b.trim.add_limb(
                Vector3::new(patio.cx(), KERB + 0.03, patio.cz()),
                Vector3::new(patio.cx(), KERB + 0.74, patio.cz()),
                0.09,
                0.09,
                5,
                Color::from_hex(0x6f747a),
                false,
            );
            b.trim.add_box(
                Vector3::new(patio.cx() - 0.7, KERB + 0.74, patio.cz() - 0.7),
                Vector3::new(patio.cx() + 0.7, KERB + 0.82, patio.cz() + 0.7),
                Color::from_hex(0xe4dccc),
                Uv::Unit,
            );
            add_tree(b, back.x1 - 1.2, back.z1 - 1.2, KERB, false, false, rng);
        }
        // A shed in the corner of a good many of them, whatever else is there.
        if rng.chance(0.34) && back.w() > 4.0 && back.d() > 4.0 {
            let (sx, sz) = (back.x1 - 1.4, back.z0 + 1.4);
            b.trim.add_box(
                Vector3::new(sx - 1.1, KERB, sz - 0.9),
                Vector3::new(sx + 1.1, KERB + 2.0, sz + 0.9),
                scale_color(Color::from_hex(0x7a6244), rng.range(0.85, 1.15)),
                Uv::Unit,
            );
            b.trim.add_gable(
                Vector3::new(sx - 1.25, KERB + 2.0, sz - 1.05),
                Vector3::new(sx + 1.25, KERB + 2.0, sz + 1.05),
                0.45,
                Color::from_hex(0x4a4f55),
            );
        }
    }

    // A camera over the door on about one house in five — which is roughly
    // the rate a modern suburb runs at, and is the detail that dates one.
    if rng.chance(0.20) {
        let (ax, az) = if fx.abs() > fz.abs() {
            (if fx < 0.0 { house.x0 } else { house.x1 }, house.cz())
        } else {
            (house.cx(), if fz < 0.0 { house.z0 } else { house.z1 })
        };
        add_cctv(
            b,
            Vector3::new(ax + fx * 0.12, KERB + 2.6, az + fz * 0.12),
            Vector3::new(fx, 0.0, fz),
            if rng.chance(0.3) { Cctv::Dome } else { Cctv::Bullet },
            rng,
        );
    }

    // A hedge or a wall along the frontage, with a gap for the drive.
    let hedge = scale_color(Color::from_hex(0x3f5a2b), rng.range(0.85, 1.15));
    if rng.chance(0.7) {
        let hh = rng.range(0.8, 1.4);
        if fx.abs() > fz.abs() {
            let x = if fx < 0.0 { plot.x0 } else { plot.x1 - 0.5 };
            for (a, c) in [(plot.z0, drive.z0), (drive.z1, plot.z1)] {
                if c - a > 1.0 {
                    b.foliage.add_box(
                        Vector3::new(x, KERB, a),
                        Vector3::new(x + 0.5, KERB + hh, c),
                        hedge,
                        Uv::Unit,
                    );
                }
            }
        } else {
            let z = if fz < 0.0 { plot.z0 } else { plot.z1 - 0.5 };
            for (a, c) in [(plot.x0, drive.x0), (drive.x1, plot.x1)] {
                if c - a > 1.0 {
                    b.foliage.add_box(
                        Vector3::new(a, KERB, z),
                        Vector3::new(c, KERB + hh, z + 0.5),
                        hedge,
                        Uv::Unit,
                    );
                }
            }
        }
    }
    true
}

/// A cul-de-sac: a stub off one side of the block, ending in a turning head,
/// with houses down both sides and around the head.
///
/// Returns how many houses were built, or 0 if the block could not take one.
pub(crate) fn add_close(
    b: &mut Batches,
    plan: &LayoutPlan,
    cell: Rect,
    rng: &mut Rng,
) -> usize {
    // Sized to the block rather than to a fixed ideal. The first version
    // wanted forty metres each way; blocks in this city are fourteen to
    // forty-seven, so it fitted almost nowhere. Check what the generator
    // actually offers before picking a threshold — this is the third time.
    //
    // The gate is on the *plot* that comes out the far end, not on the block:
    // a block can be wide enough for a road and a turning head and still leave
    // strips too shallow to put a house on, and this function draws the road
    // before it finds that out.
    if cell.w() < 26.0 || cell.d() < 28.0 && cell.w() < 28.0 {
        return 0;
    }
    let across0 = cell.w().min(cell.d());
    if (across0 * 0.5) - LANE * 0.5 - 1.2 < 9.6 {
        return 0;
    }
    let bulb = (cell.w().min(cell.d()) * 0.17).clamp(3.6, 8.0);
    // Enter from whichever side leaves the most depth, and jog the spine a
    // little off centre so it is not a mirror of the block.
    let along_x = cell.w() >= cell.d();
    let (near, far, span_lo, span_hi) = if along_x {
        (cell.x0, cell.x1, cell.z0, cell.z1)
    } else {
        (cell.z0, cell.z1, cell.x0, cell.x1)
    };
    let flip = rng.chance(0.5);
    let (mouth, head) = if flip { (far, near) } else { (near, far) };
    let dirn = if flip { -1.0f32 } else { 1.0 };
    let mid = mix(span_lo, span_hi, rng.range(0.42, 0.58));
    // The head stops short of the far side so there are plots behind it.
    let head_at = mouth + dirn * ((head - mouth).abs() - bulb - 4.0).max(6.0);
    // A slight dog-leg: a close that runs dead straight is still a grid stub.
    let bend = rng.range(-1.0, 1.0) * (cell.w().min(cell.d()) * 0.06);
    let pts: Vec<(f32, f32)> = if along_x {
        vec![
            (mouth, mid),
            (mix(mouth, head_at, 0.45), mid),
            (head_at, mid + bend),
        ]
    } else {
        vec![
            (mid, mouth),
            (mid, mix(mouth, head_at, 0.45)),
            (mid + bend, head_at),
        ]
    };
    let (bx, bz) = *pts.last().unwrap();
    // Claim the road and the head so nothing else is scattered into them.
    let road_rect = if along_x {
        [mouth.min(head_at), mid - LANE, mouth.max(head_at), mid + LANE]
    } else {
        [mid - LANE, mouth.min(head_at), mid + LANE, mouth.max(head_at)]
    };
    if !b.occ.try_claim(road_rect) {
        return 0;
    }
    b.occ.claim([bx - bulb, bz - bulb, bx + bulb, bz + bulb]);

    // Green over the whole block before anything else. The block arrives as a
    // slab of pavement, which is right for a city block and wrong for a
    // close — what is between the houses out here is grass, and leaving the
    // pad showing made the suburb read as a car park with houses in it.
    b.grass.add_slab(
        cell.x0,
        cell.z0,
        cell.x1,
        cell.z1,
        KERB + 0.005,
        scale_color(Color::from_hex(0x6f9a4a), rng.range(0.9, 1.1)),
        Uv::Unit,
    );
    kerbed(b, &pts, LANE, rng);
    // The turning head. A disc, which is the one shape the grid never makes.
    b.pads.add_ground_disc(
        bx,
        bz,
        bulb + 1.5,
        16,
        0.0,
        KERB + 0.02,
        scale_color(Color::from_hex(C_SIDEWALK), rng.range(0.94, 1.06)),
    );
    b.road.add_ground_disc(
        bx,
        bz,
        bulb,
        16,
        0.0,
        KERB + 0.04,
        scale_color(Color::from_hex(0x3c3c40), rng.range(0.94, 1.06)),
    );
    // Some closes have a little green in the middle of the head, which is
    // where the one tree in the street goes.
    if rng.chance(0.45) {
        b.grass
            .add_ground_disc(bx, bz, bulb * 0.42, 12, 0.0, KERB + 0.06, Color::from_hex(0x5f8f3f));
        add_tree(b, bx, bz, KERB + 0.06, false, false, rng);
    }
    // A no-through-road sign at the mouth.
    let (sx, sz) = pts[0];
    let off = LANE * 0.5 + 1.4;
    b.trim.add_limb(
        Vector3::new(if along_x { sx } else { sx + off }, KERB, if along_x { sz + off } else { sz }),
        Vector3::new(
            if along_x { sx } else { sx + off },
            KERB + 2.2,
            if along_x { sz + off } else { sz },
        ),
        0.06,
        0.06,
        4,
        Color::from_hex(0x9aa0a6),
        false,
    );

    // Plots. Down both sides of the spine, and a fan of three round the head.
    let mut built = 0;
    // Plot depth is whatever is left between the kerb and the block boundary.
    let across = if along_x { cell.d() } else { cell.w() };
    let depth = ((across * 0.5) - LANE * 0.5 - 1.2).clamp(9.6, 17.0);
    let run = ((head_at - mouth).abs() - bulb - 1.0).max(0.0);
    let plots = ((run / 9.0).floor() as usize).max(1);
    for i in 0..plots {
        let (t0, t1) = (i as f32 / plots as f32, (i + 1) as f32 / plots as f32);
        for s in [-1.0f32, 1.0] {
            let a = mouth + dirn * run * t0;
            let c = mouth + dirn * run * t1;
            let (lo, hi) = (a.min(c) + 0.6, a.max(c) - 0.6);
            let inner = mid + s * (LANE * 0.5 + 2.0);
            let outer = mid + s * (LANE * 0.5 + 2.0 + depth);
            let plot = if along_x {
                Rect { x0: lo, z0: inner.min(outer), x1: hi, z1: inner.max(outer) }
            } else {
                Rect { x0: inner.min(outer), z0: lo, x1: outer.max(inner), z1: hi }
            };
            // Which way the house faces: back towards the road.
            let front = if along_x { (0.0, -s) } else { (-s, 0.0) };
            if plot.w() > 8.0 && plot.d() > 8.0 && dwelling(b, plan, plot, front, rng) {
                built += 1;
            }
        }
    }
    // Round the head.
    for k in 0..3 {
        let a = -PI * 0.5 + (k as f32 - 1.0) * 0.72;
        let ang = if along_x { a * dirn } else { a * dirn + PI * 0.5 };
        let rr = bulb + 2.0 + depth * 0.5;
        let (px, pz) = (bx + rr * ang.cos() * dirn.abs(), bz + rr * ang.sin());
        let half = depth * 0.46;
        let plot = Rect { x0: px - half, z0: pz - half, x1: px + half, z1: pz + half };
        if plot.x0 > cell.x0 && plot.x1 < cell.x1 && plot.z0 > cell.z0 && plot.z1 < cell.z1 {
            let (fx, fz) = (bx - px, bz - pz);
            let l = fx.hypot(fz).max(0.001);
            if dwelling(b, plan, plot, (fx / l, fz / l), rng) {
                built += 1;
            }
        }
    }
    built
}

/// A crescent: leaves the street, bows out around a green, and rejoins it.
///
/// The other suburban form, and the one that says "planned estate" rather than
/// "infill". Houses face outward from the bow.
pub(crate) fn add_crescent(
    b: &mut Batches,
    plan: &LayoutPlan,
    cell: Rect,
    rng: &mut Rng,
) -> usize {
    if cell.w() < 30.0 || cell.d() < 30.0 {
        return 0;
    }
    let along_x = cell.w() >= cell.d();
    let (lo, hi, near, far) = if along_x {
        (cell.x0, cell.x1, cell.z0, cell.z1)
    } else {
        (cell.z0, cell.z1, cell.x0, cell.x1)
    };
    let flip = rng.chance(0.5);
    let (base, depth_dir) = if flip { (far, -1.0f32) } else { (near, 1.0) };
    // How far the bow reaches in from the street it leaves.
    let bow = ((far - near).abs() * rng.range(0.34, 0.46)).clamp(6.0, 26.0);
    let (a, c) = (mix(lo, hi, 0.16), mix(lo, hi, 0.84));
    let segs = 9;
    let pts: Vec<(f32, f32)> = (0..=segs)
        .map(|i| {
            let t = i as f32 / segs as f32;
            let u = mix(a, c, t);
            // A half-sine bow: zero at both ends, deepest in the middle.
            let v = base + depth_dir * bow * (t * PI).sin();
            if along_x {
                (u, v)
            } else {
                (v, u)
            }
        })
        .collect();
    let claim = if along_x {
        [a - 2.0, (base + depth_dir * bow).min(base) - 2.0, c + 2.0, (base + depth_dir * bow).max(base) + 2.0]
    } else {
        [(base + depth_dir * bow).min(base) - 2.0, a - 2.0, (base + depth_dir * bow).max(base) + 2.0, c + 2.0]
    };
    if !b.occ.try_claim(claim) {
        return 0;
    }
    b.grass.add_slab(
        cell.x0,
        cell.z0,
        cell.x1,
        cell.z1,
        KERB + 0.005,
        scale_color(Color::from_hex(0x6f9a4a), rng.range(0.9, 1.1)),
        Uv::Unit,
    );
    kerbed(b, &pts, LANE, rng);
    // The green inside the bow: what a crescent is built around.
    let (gx, gz) = if along_x {
        ((a + c) * 0.5, base + depth_dir * bow * 0.42)
    } else {
        (base + depth_dir * bow * 0.42, (a + c) * 0.5)
    };
    b.grass.add_ground_disc(
        gx,
        gz,
        bow * 0.34,
        14,
        0.0,
        KERB + 0.06,
        Color::from_hex(0x639640),
    );
    for _ in 0..2 + rng.below(3) {
        let ang = rng.range(0.0, TAU);
        let rr = bow * rng.range(0.05, 0.26);
        add_tree(b, gx + rr * ang.cos(), gz + rr * ang.sin(), KERB, false, false, rng);
    }
    // Houses on the outside of the bow, facing in.
    let mut built = 0;
    let depth = (((far - near).abs() - bow) * 0.62).clamp(9.6, 16.0);
    for i in 0..segs {
        let (x0, z0) = pts[i];
        let (x1, z1) = pts[i + 1];
        let (mx, mz) = ((x0 + x1) * 0.5, (z0 + z1) * 0.5);
        // Outward normal from the arc's own tangent, not from the green in the
        // middle of it. Taking it from the green points nearly *along* the
        // street at both ends of the bow, which threw those plots sideways out
        // of the block, and the block-bounds check then rejected them — so a
        // crescent came out as a road with no houses on it.
        let (tx, tz) = (x1 - x0, z1 - z0);
        let tl = tx.hypot(tz).max(0.001);
        let (mut nx, mut nz) = (-tz / tl, tx / tl);
        // Point it into the block, away from the street the crescent leaves.
        let inward = if along_x { nz * depth_dir } else { nx * depth_dir };
        if inward < 0.0 {
            nx = -nx;
            nz = -nz;
        }
        let _ = (gx, gz);
        let (px, pz) = (
            mx + nx * (LANE * 0.5 + 2.0 + depth * 0.5),
            mz + nz * (LANE * 0.5 + 2.0 + depth * 0.5),
        );
        let half = depth * 0.44;
        let plot = Rect { x0: px - half, z0: pz - half, x1: px + half, z1: pz + half };
        // Slide the plot back inside the block rather than discarding it: at
        // the ends of the bow it will always overhang a little.
        let mut plot = plot;
        let dx = (cell.x0 - plot.x0).max(0.0) + (cell.x1 - plot.x1).min(0.0);
        let dz = (cell.z0 - plot.z0).max(0.0) + (cell.z1 - plot.z1).min(0.0);
        plot = Rect {
            x0: plot.x0 + dx,
            z0: plot.z0 + dz,
            x1: plot.x1 + dx,
            z1: plot.z1 + dz,
        };
        if plot.x0 < cell.x0 - 0.1 || plot.x1 > cell.x1 + 0.1 || plot.z0 < cell.z0 - 0.1 || plot.z1 > cell.z1 + 0.1 {
            continue;
        }
        if dwelling(b, plan, plot, (-nx, -nz), rng) {
            built += 1;
        }
    }
    built
}
