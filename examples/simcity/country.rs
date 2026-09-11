//! Part of the `simcity` example; see `mod.rs`.
//!
//! What is outside the city.
//!
//! Everything past the last road was a flat green plane running to the fog —
//! which is fine at street level and the first thing you notice from the air,
//! where half the frame is empty. A city sits *in* something, and the something
//! is fields, woods and water.
//!
//! It is laid out on a coarse grid over the ring around the built area, one
//! parcel a cell, and everything in it is drawn at the far level of detail: at
//! this range a tree is a few pixels, and paying near-detail prices for a
//! thousand of them out here would cost more than the city does.
#![allow(dead_code)]

use super::*;

/// What a parcel of countryside is.
///
/// The split that matters is not how many kinds there are but which side of
/// one line each falls on. **Worked** land is regular: straight rows, square
/// corners, one crop to a field, a hedge marking where somebody's ownership
/// stops. **Wild** land is not: no boundary, density that varies across the
/// parcel, a mix of species and sizes, gaps.
///
/// Farms and woods read as the same thing when a wood is a rectangle of evenly
/// scattered trees — because that is a plantation, not a wood — and when a
/// field's only feature is the hedge of trees round it. So `Plantation` exists
/// deliberately as the regular one, to give `Wood` something to be unlike.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Parcel {
    /// A planted field, hedged, with tramlines.
    Field,
    /// Bare earth under the plough.
    Plough,
    /// Fruit trees on a grid, on mown grass.
    Orchard,
    /// Dense parallel rows on wires.
    Vineyard,
    /// Rough grazing: grass, a few trees, no hedges.
    Pasture,
    /// Long grass gone to flower.
    Meadow,
    /// Wild trees: clumped, mixed, and no straight edge anywhere.
    Wood,
    /// Conifers in rows with a firebreak. The regular one.
    Plantation,
    /// Gorse, bracken and stone. Too poor to farm.
    Heath,
    /// Open water.
    Lake,
    /// Barns, silos and a yard.
    Farmyard,
    /// A rise in the ground. What makes the fields around it read as a valley.
    Hill,
}

/// How much of a parcel to build.
///
/// There were two tiers, and the gap between them did the damage: full detail
/// out to three half-widths and a flat coloured rectangle past it. That is
/// fine when the haze closes in at four, and it is the reason the middle
/// distance went bare the moment the background stopped being clipped early.
///
/// The middle tier is the cheap one and it earns its place twice over. It
/// keeps the *pattern* — furrows, rows, drifts, a lumpy canopy — which is all
/// that survives at that range anyway, and it lets the expensive ring shrink,
/// so this costs less than the two-tier version it replaces while covering
/// four times the ground.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Lod {
    /// Individual trees, understorey, buildings.
    Near,
    /// Pattern and silhouette. No individual anything.
    Mid,
    /// A colour and an edge.
    Far,
}

/// Crops, and the seasons they are in. Fields being *different colours* is the
/// whole of what makes farmland read as farmland from above — a uniform green
/// is a lawn however large it is.
const CROP: [u32; 8] = [
    0x8a9a3c, 0xb8a05a, 0x6f8a34, 0xc2a44a, 0x5c7a2e, 0x9c8a3a, 0x7a6b3c, 0xa8b04a,
];

/// Fill the ring around the city with countryside.
///
/// `inner` is the half-width of the built area and `outer` how far the terrain
/// runs. Parcels are placed on a grid and jittered, so the boundaries are not
/// a visible lattice.
pub(crate) fn add_countryside(
    b: &mut Batches,
    inner: f32,
    outer: f32,
    water: Option<(f32, f32)>,
    warm: bool,
    rng: &mut Rng,
) {
    // Dressed a long way out now that the fog does not close in at four
    // half-widths — but the grain coarsens with distance. Past the near ring
    // a parcel is a colour and a hedgerow and nothing else: at that range a
    // tree is a pixel, and a thousand pixels are not worth a hundred thousand
    // triangles.
    // Three rings, each with its own grain. Cells grow with distance because
    // a 62 m parcel two miles out is four pixels: paying for the boundary
    // between it and its neighbour buys nothing, and there are thousands of
    // them.
    let near_ring = inner * 2.2;
    let mid_ring = inner * 8.0;
    let reach = outer.min(inner * 26.0);

    // The floor of the world, out to wherever the camera can still see.
    //
    // There was none: the dressed parcels *were* the ground, so past the last
    // ring there was nothing at all, and the river — built to `outer`, far
    // beyond — carried on through empty space.
    let base = scale_color(
        Color::from_hex(if warm { 0x8fa254 } else { 0x789a52 }),
        0.94,
    );
    // Laid either side of the water rather than under it: the surface sits at
    // `-RIVER_DEPTH`, so one plate across the whole world covers the river up
    // and the city loses its river altogether.
    //
    // Split into cells rather than one quad per side. Two triangles would do
    // for the geometry, but the terrain is lit per vertex, and a quad this
    // size gets its shading from four corners a kilometre apart.
    let mut plate = |x0: f32, x1: f32| {
        if x1 - x0 <= 1.0 {
            return;
        }
        let n = 8usize;
        for i in 0..n {
            for j in 0..n {
                let a = mix(x0, x1, i as f32 / n as f32);
                let b2 = mix(x0, x1, (i + 1) as f32 / n as f32);
                let c = mix(-outer, outer, j as f32 / n as f32);
                let d = mix(-outer, outer, (j + 1) as f32 / n as f32);
                b.grass.add_slab(a, c, b2, d, -0.35, base, Uv::Unit);
            }
        }
    };
    match water {
        Some((w0, w1)) => {
            plate(-outer, w0.max(-outer));
            plate(w1.min(outer), outer);
        }
        None => plate(-outer, outer),
    }

    for (cell, lo, hi) in [(62.0f32, 0.0f32, mid_ring), (196.0, mid_ring, reach)] {
        if hi <= lo {
            continue;
        }
        let n = ((hi * 2.0) / cell).ceil() as i32;
        let half = n as f32 * cell * 0.5;
        for ix in 0..n {
            for iz in 0..n {
                let x0 = -half + ix as f32 * cell;
                let z0 = -half + iz as f32 * cell;
                let (cx, cz) = (x0 + cell * 0.5, z0 + cell * 0.5);
                // Outside the city, inside this ring.
                if cx.abs() < inner * 1.12 && cz.abs() < inner * 1.12 {
                    continue;
                }
                let d = cx.hypot(cz);
                if d > hi || d <= lo {
                    continue;
                }
                // Never over the water.
                if let Some((w0, w1)) = water {
                    if cx > w0 - cell && cx < w1 + cell {
                        continue;
                    }
                }
                // Nor over anything the infrastructure pass has already taken.
                // It runs first precisely so it gets first refusal on ground:
                // a power station is a hundred and fifty metres across and
                // there is no arrangement of fields it can be squeezed into
                // afterwards. Tested on the whole cell rather than the inset
                // parcel so hedges and headlands stay outside the fence too.
                if !b.occ.free([x0, z0, x0 + cell, z0 + cell]) {
                    continue;
                }

                let lod = if d < near_ring {
                    Lod::Near
                } else if d < mid_ring {
                    Lod::Mid
                } else {
                    Lod::Far
                };
                let roll = rng.f();
                let kind = if lod == Lod::Far {
                    // Out here everything is a colour, so the kinds worth
                    // keeping are the ones that differ *as* colours: green
                    // crop, bare earth, flowering meadow, dark timber, water.
                    if roll < 0.40 {
                        Parcel::Field
                    } else if roll < 0.58 {
                        Parcel::Plough
                    } else if roll < 0.72 {
                        Parcel::Meadow
                    } else if roll < 0.92 {
                        Parcel::Wood
                    } else {
                        Parcel::Lake
                    }
                } else if roll < 0.22 {
                    Parcel::Field
                } else if roll < 0.32 {
                    Parcel::Plough
                } else if roll < 0.39 {
                    Parcel::Orchard
                } else if roll < 0.45 {
                    Parcel::Vineyard
                } else if roll < 0.56 {
                    Parcel::Pasture
                } else if roll < 0.64 {
                    Parcel::Meadow
                } else if roll < 0.79 {
                    Parcel::Wood
                } else if roll < 0.85 {
                    Parcel::Plantation
                } else if roll < 0.88 {
                    Parcel::Heath
                } else if roll < 0.93 {
                    Parcel::Hill
                } else if roll < 0.97 {
                    Parcel::Lake
                } else {
                    Parcel::Farmyard
                };
                // Inset by a strongly jittered margin, independently on each
                // of the four sides. A uniform inset leaves every parcel the
                // same size and the ring reads as a lattice — which is the one
                // thing farmland never looks like. Occasionally take a very
                // deep bite out of one side so a few parcels are half the size
                // of the rest.
                let k = cell / 62.0;
                let bite = |rng: &mut Rng| {
                    if rng.chance(0.18) {
                        rng.range(10.0, 24.0) * k
                    } else {
                        rng.range(1.0, 5.0) * k
                    }
                };
                let p = Rect {
                    x0: x0 + bite(rng),
                    z0: z0 + bite(rng),
                    x1: x0 + cell - bite(rng),
                    z1: z0 + cell - bite(rng),
                };
                if p.w() < 12.0 * k || p.d() < 12.0 * k {
                    // Too small to be worth hedging: leave it as rough ground,
                    // so the grid has gaps in it as well as variety.
                    if lod == Lod::Near {
                        pasture(b, &p.inset(-4.0), warm, rng);
                    } else {
                        meadow(b, &p.inset(-4.0), lod, rng);
                    }
                    continue;
                }
                match kind {
                    Parcel::Field => field(b, &p, lod, rng),
                    Parcel::Plough => plough(b, &p, lod, rng),
                    Parcel::Orchard => orchard(b, &p, lod, warm, rng),
                    Parcel::Vineyard => vineyard(b, &p, lod, rng),
                    Parcel::Meadow => meadow(b, &p, lod, rng),
                    Parcel::Wood => wood(b, &p, lod, warm, rng),
                    Parcel::Plantation => plantation(b, &p, lod, rng),
                    Parcel::Heath => heath(b, &p, lod, rng),
                    Parcel::Lake => lake(b, &p, lod, rng),
                    Parcel::Hill => hill(b, &p, lod, warm, rng),
                    Parcel::Pasture => pasture(b, &p, warm, rng),
                    Parcel::Farmyard => farmyard(b, &p, rng),
                }
            }
        }
    }
}

fn field(b: &mut Batches, p: &Rect, lod: Lod, rng: &mut Rng) {
    let crop = scale_color(
        Color::from_hex(CROP[rng.below(CROP.len())]),
        rng.range(0.88, 1.12),
    );
    b.grass
        .add_slab(p.x0, p.z0, p.x1, p.z1, 0.05, crop, Uv::Unit);
    if lod == Lod::Far {
        // Just the colour. A hedge is twelve triangles and there are more than
        // a thousand parcels out here; at this range what separates one field
        // from the next is that they are different colours, which they are.
        return;
    }
    // Tramlines: the wheel marks a tractor leaves, which is the other half of
    // what says "field" rather than "green rectangle".
    let along_x = p.w() >= p.d();
    let lines = ((if along_x { p.d() } else { p.w() }) / 7.0) as usize;
    let pale = scale_color(crop, 1.14);
    for i in 1..lines {
        let t = i as f32 / lines as f32;
        if along_x {
            let z = mix(p.z0, p.z1, t);
            b.grass
                .add_slab(p.x0, z - 0.5, p.x1, z + 0.5, 0.06, pale, Uv::Unit);
        } else {
            let x = mix(p.x0, p.x1, t);
            b.grass
                .add_slab(x - 0.5, p.z0, x + 0.5, p.z1, 0.06, pale, Uv::Unit);
        }
    }
    // Hedgerows on the boundary: the mark of worked land.
    hedge_sides(b, p, 0x35502a, rng);
    if lod != Lod::Near {
        return;
    }
    // A barn or a farmhouse on one field in five.
    if rng.chance(0.20) {
        let (bx, bz) = (
            mix(p.x0 + 6.0, p.x1 - 6.0, rng.f()),
            mix(p.z0 + 6.0, p.z1 - 6.0, rng.f()),
        );
        let wall = scale_color(Color::from_hex(0x9a8f7d), rng.range(0.85, 1.15));
        let roof = scale_color(Color::from_hex(0x6b4038), rng.range(0.85, 1.15));
        let (w, d, h) = (rng.range(5.0, 9.0), rng.range(4.0, 7.0), rng.range(3.5, 5.5));
        b.trim.add_box(
            Vector3::new(bx - w * 0.5, 0.0, bz - d * 0.5),
            Vector3::new(bx + w * 0.5, h, bz + d * 0.5),
            wall,
            Uv::Unit,
        );
        b.trim.add_gable(
            Vector3::new(bx - w * 0.55, h, bz - d * 0.55),
            Vector3::new(bx + w * 0.55, h, bz + d * 0.55),
            rng.range(1.6, 2.6),
            roof,
        );
    }
}

/// Wild trees.
///
/// Everything here exists to *not* be a rectangle of evenly spaced trees.
/// Density comes from a handful of clump centres and falls off away from them,
/// so the wood has a thick middle and a ragged edge; a couple of clearings are
/// punched out of it; species are mixed with one dominant rather than drawn
/// uniformly, because a wood that is a third of each reads as an arboretum;
/// and the trees are allowed to overrun the parcel boundary, which is the
/// single most effective thing, since the boundary is what makes a stand of
/// trees look planted.
fn wood(b: &mut Batches, p: &Rect, lod: Lod, warm: bool, rng: &mut Rng) {
    // Mottled floor rather than one flat colour: leaf litter, moss and the
    // odd sunlit gap. Three overlapping patches cost three quads.
    let floor = scale_color(Color::from_hex(0x3d5628), rng.range(0.9, 1.1));
    b.grass
        .add_slab(p.x0, p.z0, p.x1, p.z1, 0.05, floor, Uv::Unit);
    for _ in 0..3 {
        let (sx, sz) = (rng.range(0.25, 0.6), rng.range(0.25, 0.6));
        let (ox, oz) = (rng.range(0.0, 1.0 - sx), rng.range(0.0, 1.0 - sz));
        b.grass.add_slab(
            mix(p.x0, p.x1, ox),
            mix(p.z0, p.z1, oz),
            mix(p.x0, p.x1, ox + sx),
            mix(p.z0, p.z1, oz + sz),
            0.06,
            scale_color(floor, rng.range(0.82, 1.24)),
            Uv::Unit,
        );
    }
    if lod != Lod::Near {
        // At range a wood is a dark patch with a lumpy top edge. A handful of
        // blobs give it that for a fraction of what the trees cost, and the
        // silhouette is the only part of a wood that carries this far. Scaled
        // to the parcel, because the far ring's cells are three times the
        // width of the middle ring's.
        let crown = scale_color(Color::from_hex(0x354d24), rng.range(0.9, 1.1));
        let k = (p.w().min(p.d()) / 55.0).max(1.0);
        for _ in 0..if lod == Lod::Mid { 7 } else { 4 } {
            let (cx, cz) = (
                rng.range(p.x0, p.x1),
                rng.range(p.z0, p.z1),
            );
            let r = rng.range(6.0, 14.0) * k;
            b.foliage.add_blob(
                Vector3::new(cx, rng.range(4.0, 8.0) * k, cz),
                r,
                r * rng.range(0.5, 0.8),
                r * rng.range(0.8, 1.2),
                // Three rings, not two. `add_blob` at two rings is a
                // bipyramid — a diamond in silhouette — which is invisible
                // from above and unmistakable from anywhere near the ground,
                // where a wood at range came out as a row of floating
                // gemstones. The plumes had the same fault.
                3,
                7,
                0.22,
                rng.next_u32() as i32 & 0xffff,
                0.55,
                crown,
            );
        }
        return;
    }
    // One species dominates; the rest are seasoning.
    let dominant = if warm && rng.chance(0.35) {
        Species::Palm
    } else {
        match rng.below(3) {
            0 => Species::Conifer,
            1 => Species::Columnar,
            _ => Species::Broadleaf,
        }
    };
    // Clumps, and gaps between them.
    let clumps: Vec<(f32, f32, f32)> = (0..rng.range(2.0, 5.0) as usize + 2)
        .map(|_| {
            (
                rng.range(p.x0, p.x1),
                rng.range(p.z0, p.z1),
                rng.range(9.0, 22.0),
            )
        })
        .collect();
    let clearings: Vec<(f32, f32, f32)> = (0..rng.below(3))
        .map(|_| {
            (
                rng.range(p.x0, p.x1),
                rng.range(p.z0, p.z1),
                rng.range(6.0, 13.0),
            )
        })
        .collect();
    // Sampled over the parcel *and a margin outside it*, so the wood spills
    // into whatever is next door instead of stopping on a survey line.
    let m = 5.0;
    let tries = (p.area() / 22.0) as usize;
    for _ in 0..tries {
        let x = rng.range(p.x0 - m, p.x1 + m);
        let z = rng.range(p.z0 - m, p.z1 + m);
        // Density is the strongest clump's falloff at this point.
        let d = clumps.iter().fold(0.0f32, |acc, (cx, cz, r)| {
            acc.max(1.0 - (x - cx).hypot(z - cz) / r)
        });
        if d <= 0.0 || rng.f() > d * 0.9 {
            continue;
        }
        if clearings
            .iter()
            .any(|(cx, cz, r)| (x - cx).hypot(z - cz) < *r)
        {
            continue;
        }
        // Thicker in the middle of a clump, and the odd emergent standing
        // clear of the canopy.
        let scale = if rng.chance(0.07) {
            rng.range(1.35, 1.75)
        } else {
            rng.range(0.55, 1.05) * (0.75 + 0.45 * d)
        };
        let species = if rng.chance(0.62) {
            dominant
        } else {
            match rng.below(5) {
                0 => Species::Conifer,
                1 => Species::Columnar,
                2 => Species::Weeping,
                3 => Species::Bare,
                _ => Species::Broadleaf,
            }
        };
        add_tree_as(b, x, z, 0.0, false, species, None, scale, rng);
    }
    // Understorey: bramble and deadfall in the gaps, which is what a wood has
    // and a plantation does not.
    for _ in 0..(p.area() / 130.0) as usize {
        let (x, z) = (rng.range(p.x0, p.x1), rng.range(p.z0, p.z1));
        if rng.chance(0.7) {
            let r = rng.range(1.2, 2.8);
            b.foliage.add_blob(
                Vector3::new(x, r * 0.55, z),
                r,
                r * 0.55,
                r * rng.range(0.7, 1.3),
                3,
                6,
                0.3,
                rng.next_u32() as i32 & 0xffff,
                0.5,
                scale_color(Color::from_hex(0x3f5a2b), rng.range(0.85, 1.15)),
            );
        } else {
            // A fallen trunk, lying whichever way it fell.
            let a = rng.range(0.0, TAU);
            let l = rng.range(3.0, 7.0);
            b.foliage.add_limb(
                Vector3::new(x, 0.35, z),
                Vector3::new(x + l * a.cos(), 0.30, z + l * a.sin()),
                0.22,
                0.16,
                5,
                Color::from_hex(0x584434),
                false,
            );
        }
    }
}

/// Conifers in rows, with a firebreak through the middle.
///
/// This is the regular one, and it is here on purpose: a wood only reads as
/// wild if there is something planted nearby to be wild *against*.
fn plantation(b: &mut Batches, p: &Rect, lod: Lod, rng: &mut Rng) {
    b.grass.add_slab(
        p.x0,
        p.z0,
        p.x1,
        p.z1,
        0.05,
        scale_color(Color::from_hex(0x36461f), rng.range(0.95, 1.05)),
        Uv::Unit,
    );
    if lod != Lod::Near {
        // Rows of blobs: a plantation at range is a dark block with a ruled
        // edge and a pale lane through it, and that is exactly what reads.
        let crown = scale_color(Color::from_hex(0x2f4a2a), rng.range(0.92, 1.08));
        let rows = 3;
        for i in 0..rows {
            let t = (i as f32 + 0.5) / rows as f32;
            let (cx, cz) = if p.w() >= p.d() {
                (p.cx(), mix(p.z0, p.z1, t))
            } else {
                (mix(p.x0, p.x1, t), p.cz())
            };
            let (rx, rz) = if p.w() >= p.d() {
                (p.w() * 0.46, p.d() / rows as f32 * 0.46)
            } else {
                (p.w() / rows as f32 * 0.46, p.d() * 0.46)
            };
            b.foliage.add_blob(
                Vector3::new(cx, 5.0, cz),
                rx,
                4.5,
                rz,
                3,
                7,
                0.10,
                rng.next_u32() as i32 & 0xffff,
                0.6,
                crown,
            );
        }
        return;
    }
    let along_x = p.w() >= p.d();
    let pitch = rng.range(5.0, 7.0);
    // One lane left unplanted, straight across.
    let break_t = rng.range(0.3, 0.7);
    let tint = Some(scale_color(Color::from_hex(0x2f4a2a), rng.range(0.92, 1.08)));
    let scale = rng.range(0.8, 1.0);
    let (nx, nz) = ((p.w() / pitch) as usize, (p.d() / pitch) as usize);
    for i in 0..nx {
        for j in 0..nz {
            let x = p.x0 + (i as f32 + 0.5) * pitch;
            let z = p.z0 + (j as f32 + 0.5) * pitch;
            let t = if along_x {
                (z - p.z0) / p.d()
            } else {
                (x - p.x0) / p.w()
            };
            if (t - break_t).abs() < 0.06 {
                continue;
            }
            // Uniform to within a hand's width: that is the whole point.
            add_tree_as(
                b,
                x + rng.range(-0.4, 0.4),
                z + rng.range(-0.4, 0.4),
                0.0,
                false,
                Species::Conifer,
                tint,
                scale * rng.range(0.96, 1.04),
                rng,
            );
        }
    }
}

/// Fruit trees on a grid, on mown grass. Worked land that happens to be trees.
fn orchard(b: &mut Batches, p: &Rect, lod: Lod, warm: bool, rng: &mut Rng) {
    b.grass.add_slab(
        p.x0,
        p.z0,
        p.x1,
        p.z1,
        0.05,
        scale_color(Color::from_hex(0x6f8f3e), rng.range(0.94, 1.06)),
        Uv::Unit,
    );
    if lod != Lod::Near {
        // The grid is the point, so keep the grid and drop the trees: a
        // stipple of small dark squares on mown grass.
        let dot = scale_color(Color::from_hex(0x4a6b2c), rng.range(0.9, 1.1));
        let pitch = 12.0f32.max(p.w().min(p.d()) / 4.0);
        let (nx, nz) = ((p.w() / pitch) as usize, (p.d() / pitch) as usize);
        for i in 0..nx {
            for j in 0..nz {
                let x = p.x0 + (i as f32 + 0.5) * pitch;
                let z = p.z0 + (j as f32 + 0.5) * pitch;
                let r = pitch * 0.3;
                b.foliage.add_box(
                    Vector3::new(x - r, 0.0, z - r),
                    Vector3::new(x + r, r * 1.6, z + r),
                    dot,
                    Uv::Unit,
                );
            }
        }
        hedge_sides(b, p, 0x3f5a2b, rng);
        return;
    }
    let pitch = rng.range(7.0, 9.5);
    let species = if warm && rng.chance(0.4) {
        Species::Palm
    } else {
        Species::Broadleaf
    };
    let tint = Some(scale_color(Color::from_hex(0x5c8a35), rng.range(0.9, 1.1)));
    let (nx, nz) = ((p.w() / pitch) as usize, (p.d() / pitch) as usize);
    for i in 0..nx {
        for j in 0..nz {
            add_tree_as(
                b,
                p.x0 + (i as f32 + 0.5) * pitch,
                p.z0 + (j as f32 + 0.5) * pitch,
                0.0,
                false,
                species,
                tint,
                rng.range(0.42, 0.56),
                rng,
            );
        }
    }
    hedge_sides(b, p, 0x3f5a2b, rng);
}

/// Vines on wires: many thin parallel rows, all one way.
fn vineyard(b: &mut Batches, p: &Rect, lod: Lod, rng: &mut Rng) {
    b.grass.add_slab(
        p.x0,
        p.z0,
        p.x1,
        p.z1,
        0.05,
        scale_color(Color::from_hex(0x8a7c4e), rng.range(0.94, 1.06)),
        Uv::Unit,
    );
    let along_x = p.w() >= p.d();
    // The rows *are* the parcel, so they survive to the middle tier — just
    // wider apart, because at that range a 3 m pitch is finer than a pixel and
    // aliases into a shimmer.
    let pitch = match lod {
        Lod::Near => rng.range(2.6, 3.4),
        Lod::Mid => rng.range(5.5, 7.0),
        Lod::Far => rng.range(11.0, 15.0),
    };
    let leaf = scale_color(Color::from_hex(0x4e6f2c), rng.range(0.92, 1.08));
    let span = if along_x { p.d() } else { p.w() };
    let rows = (span / pitch) as usize;
    for i in 0..rows {
        let t = (i as f32 + 0.5) * pitch;
        let h = rng.range(1.5, 1.9);
        if along_x {
            let z = p.z0 + t;
            b.foliage.add_box(
                Vector3::new(p.x0 + 1.5, 0.4, z - 0.45),
                Vector3::new(p.x1 - 1.5, h, z + 0.45),
                leaf,
                Uv::Unit,
            );
        } else {
            let x = p.x0 + t;
            b.foliage.add_box(
                Vector3::new(x - 0.45, 0.4, p.z0 + 1.5),
                Vector3::new(x + 0.45, h, p.z1 - 1.5),
                leaf,
                Uv::Unit,
            );
        }
    }
}

/// Bare earth, furrowed. The one parcel that is not green.
fn plough(b: &mut Batches, p: &Rect, lod: Lod, rng: &mut Rng) {
    let soil = scale_color(
        Color::from_hex(if rng.chance(0.4) { 0x6a4f34 } else { 0x7d6242 }),
        rng.range(0.9, 1.1),
    );
    b.grass
        .add_slab(p.x0, p.z0, p.x1, p.z1, 0.05, soil, Uv::Unit);
    if lod == Lod::Far {
        return;
    }
    // Furrows: close-spaced and high contrast, because turned earth throws a
    // hard shadow and that corduroy is what says "ploughed" from the air.
    let along_x = p.w() >= p.d();
    let pitch = rng.range(1.8, 2.6);
    let dark = scale_color(soil, 0.78);
    let span = if along_x { p.d() } else { p.w() };
    for i in 0..(span / pitch) as usize {
        let t = (i as f32 + 0.5) * pitch;
        if along_x {
            let z = p.z0 + t;
            b.grass
                .add_slab(p.x0, z - pitch * 0.28, p.x1, z + pitch * 0.28, 0.06, dark, Uv::Unit);
        } else {
            let x = p.x0 + t;
            b.grass
                .add_slab(x - pitch * 0.28, p.z0, x + pitch * 0.28, p.z1, 0.06, dark, Uv::Unit);
        }
    }
    hedge_sides(b, p, 0x35502a, rng);
}

/// Long grass gone to flower.
fn meadow(b: &mut Batches, p: &Rect, lod: Lod, rng: &mut Rng) {
    let base = scale_color(Color::from_hex(0x93a84c), rng.range(0.92, 1.1));
    b.grass
        .add_slab(p.x0, p.z0, p.x1, p.z1, 0.05, base, Uv::Unit);
    // Drifts of flower colour. Even at distance these are what tells a meadow
    // from a lawn, so they are the one thing kept at the far level of detail.
    let hues = [0xd8d066u32, 0xc98fb0, 0xe0e0e0, 0xb8c46a];
    for _ in 0..if lod == Lod::Far { 2 } else { 7 } {
        let (sx, sz) = (rng.range(0.15, 0.45), rng.range(0.15, 0.45));
        let (ox, oz) = (rng.range(0.0, 1.0 - sx), rng.range(0.0, 1.0 - sz));
        b.grass.add_slab(
            mix(p.x0, p.x1, ox),
            mix(p.z0, p.z1, oz),
            mix(p.x0, p.x1, ox + sx),
            mix(p.z0, p.z1, oz + sz),
            0.06,
            scale_color(
                mix_color(base, Color::from_hex(hues[rng.below(hues.len())]), 0.55),
                rng.range(0.9, 1.1),
            ),
            Uv::Unit,
        );
    }
}

/// Gorse, bracken and stone: land nobody bothered to farm.
fn heath(b: &mut Batches, p: &Rect, lod: Lod, rng: &mut Rng) {
    b.grass.add_slab(
        p.x0,
        p.z0,
        p.x1,
        p.z1,
        0.05,
        scale_color(Color::from_hex(0x7a7440), rng.range(0.9, 1.12)),
        Uv::Unit,
    );
    for _ in 0..(p.area() / if lod == Lod::Near { 55.0 } else { 520.0 }) as usize {
        let (x, z) = (rng.range(p.x0, p.x1), rng.range(p.z0, p.z1));
        if rng.chance(0.78) {
            let r = rng.range(0.9, 2.4);
            b.foliage.add_blob(
                Vector3::new(x, r * 0.5, z),
                r,
                r * rng.range(0.4, 0.7),
                r * rng.range(0.7, 1.3),
                3,
                6,
                0.34,
                rng.next_u32() as i32 & 0xffff,
                0.5,
                scale_color(
                    Color::from_hex(if rng.chance(0.35) { 0xa8a13c } else { 0x4c5a2c }),
                    rng.range(0.85, 1.15),
                ),
            );
        } else {
            let r = rng.range(0.6, 1.8);
            b.trim.add_blob(
                Vector3::new(x, r * 0.35, z),
                r,
                r * 0.5,
                r * rng.range(0.7, 1.2),
                2,
                5,
                0.28,
                rng.next_u32() as i32 & 0xffff,
                0.6,
                scale_color(Color::from_hex(0x8d8b84), rng.range(0.85, 1.15)),
            );
        }
    }
}

/// A working yard: barns round three sides, silos, and hard standing.
fn farmyard(b: &mut Batches, p: &Rect, rng: &mut Rng) {
    b.grass.add_slab(
        p.x0,
        p.z0,
        p.x1,
        p.z1,
        0.05,
        scale_color(Color::from_hex(0x7f8a4e), rng.range(0.92, 1.08)),
        Uv::Unit,
    );
    let y = p.inset(rng.range(5.0, 10.0));
    b.pads.add_slab(
        y.x0,
        y.z0,
        y.x1,
        y.z1,
        0.09,
        scale_color(Color::from_hex(0x8e8b80), rng.range(0.9, 1.1)),
        Uv::Unit,
    );
    // Sheds along the long sides, gable ends to the yard.
    let along_x = y.w() >= y.d();
    for side in [0.0f32, 1.0] {
        if rng.chance(0.25) {
            continue;
        }
        let (w, d, h) = (
            rng.range(9.0, 16.0),
            rng.range(5.0, 8.0),
            rng.range(4.0, 6.5),
        );
        let (cx, cz) = if along_x {
            (
                mix(y.x0 + w * 0.6, y.x1 - w * 0.6, rng.f()),
                mix(y.z0 + d * 0.6, y.z1 - d * 0.6, side),
            )
        } else {
            (
                mix(y.x0 + w * 0.6, y.x1 - w * 0.6, side),
                mix(y.z0 + d * 0.6, y.z1 - d * 0.6, rng.f()),
            )
        };
        let wall = scale_color(Color::from_hex(0x8f8a7c), rng.range(0.85, 1.15));
        let roof = scale_color(
            Color::from_hex(if rng.chance(0.5) { 0x54585c } else { 0x6b4038 }),
            rng.range(0.85, 1.15),
        );
        b.trim.add_box(
            Vector3::new(cx - w * 0.5, 0.0, cz - d * 0.5),
            Vector3::new(cx + w * 0.5, h, cz + d * 0.5),
            wall,
            Uv::Unit,
        );
        b.trim.add_gable(
            Vector3::new(cx - w * 0.55, h, cz - d * 0.55),
            Vector3::new(cx + w * 0.55, h, cz + d * 0.55),
            rng.range(1.4, 2.4),
            roof,
        );
    }
    // Silos: the tall thing that says farm from a mile off.
    for i in 0..1 + rng.below(3) {
        let r = rng.range(1.6, 2.6);
        let x = y.x0 + 3.0 + i as f32 * (r * 2.4 + 0.6);
        if x + r > y.x1 {
            break;
        }
        let z = y.z0 + r + 1.0;
        let h = rng.range(7.0, 12.0);
        b.trim.add_limb(
            Vector3::new(x, 0.0, z),
            Vector3::new(x, h, z),
            r,
            r,
            9,
            scale_color(Color::from_hex(0xb9bcc0), rng.range(0.9, 1.1)),
            true,
        );
        b.trim.add_blob(
            Vector3::new(x, h + r * 0.3, z),
            r,
            r * 0.55,
            r,
            3,
            9,
            0.0,
            0,
            0.7,
            Color::from_hex(0x9aa0a6),
        );
    }
    // Round bales, stacked where they fell off the trailer.
    for _ in 0..rng.below(7) {
        let (x, z) = (rng.range(y.x0, y.x1), rng.range(y.z0, y.z1));
        let r = rng.range(0.7, 1.0);
        let a = rng.range(0.0, TAU);
        b.trim.add_limb(
            Vector3::new(x - r * a.cos() * 0.6, r, z - r * a.sin() * 0.6),
            Vector3::new(x + r * a.cos() * 0.6, r, z + r * a.sin() * 0.6),
            r,
            r,
            8,
            scale_color(Color::from_hex(0xcdb877), rng.range(0.9, 1.1)),
            true,
        );
    }
}

/// Mountains, beyond everything else.
///
/// These are scenery and are built like it: a ring of ridges out past the last
/// field, each a run of overlapping cones sharing a base line, with the peaks
/// snow-capped above a height that is the same for all of them. That last part
/// is what sells it — a snow line is a horizontal plane cutting through a
/// range, so every summit above it is white and every one below is not, and
/// the eye reads the whole range as one landform rather than as a row of
/// separate lumps.
///
/// They sit outside the fog's reach, so most of the time they are a pale grey
/// silhouette. That is what distant mountains look like.
/// One cone of the mountain ring: where it stands, how far its foot reaches,
/// and how tall it is.
///
/// Handed back so the roads can be driven through it — see `tunnel.rs`. A lane
/// crossing a cone was already hidden inside it, which read as a road that
/// stops dead in a hillside; knowing where the cones are is what turns that
/// into a tunnel.
#[derive(Clone, Copy)]
pub(crate) struct Summit {
    pub(crate) x: f32,
    pub(crate) z: f32,
    /// Circumradius of the cone where it meets the ground.
    pub(crate) foot: f32,
    pub(crate) height: f32,
}

/// Sides in a drawn cone.
///
/// This matters outside the drawing code because the cones are not round. A
/// nine-sided pyramid reaches `foot` at its corners but only `foot * cos(pi/9)`
/// across its flats — 94% — and at the toe, where the cone has almost no
/// height, that last 6% is forty-odd metres of ground that a circle test calls
/// hillside and the screen shows as open field. The first tunnels were sited
/// with a circle and their portals stood in the open with the mountain a
/// couple of hundred metres behind them.
pub(crate) const CONE_SIDES: usize = 9;

impl Summit {
    /// Depth of rock over a point, or zero outside the cone.
    ///
    /// Exact, not a bound. A circle overstates the footprint by up to 6% and a
    /// tunnel mouth sited on it stands clear of the mountain; the inradius
    /// understates it by the same amount, and a mouth sited on *that* is
    /// buried under as much as eighty metres of hillside that the estimate
    /// said was twelve. Both were built. The cone is a nonagon at a known
    /// orientation, so the honest answer is available and neither bound is
    /// worth the trouble it causes.
    ///
    /// `add_limb` puts its ring at `(sin(a), cos(a)) * r` with the first
    /// vertex on +Z, which is where the angle below comes from.
    pub(crate) fn cover_at(&self, x: f32, z: f32) -> f32 {
        let (dx, dz) = (x - self.x, z - self.z);
        let d = (dx * dx + dz * dz).sqrt();
        if self.foot <= 1e-3 || d >= self.foot {
            return 0.0;
        }
        let sector = TAU / CONE_SIDES as f32;
        // Angle from the midpoint of whichever facet this direction crosses.
        let off = dx.atan2(dz).rem_euclid(sector) - sector * 0.5;
        // The facet is a straight edge: its distance from the axis is the
        // inradius, and the polygon's radius along `off` is that over cos.
        let reach = self.foot * (sector * 0.5).cos() / off.cos().max(1e-3);
        if d >= reach {
            return 0.0;
        }
        self.height * (1.0 - d / reach)
    }
}

/// What `add_mountains` built.
pub(crate) struct Mountains {
    pub(crate) peaks: usize,
    /// Widest stretch of horizon, in radians, with no mountain on it. Reported
    /// because it is the thing that goes wrong: the peaks are easy, an even
    /// ring of them is not, and a hole in the ring only shows if you happen to
    /// look down that bearing. See `mountains_ring_the_horizon`.
    pub(crate) gap: f32,
    pub(crate) summits: Vec<Summit>,
}

/// Builds the ring of hills.
pub(crate) fn add_mountains(b: &mut Batches, inner: f32, outer: f32, rng: &mut Rng) -> Mountains {
    // Inside the haze, not beyond it. At four-fifths of the world's radius
    // they sat well past the point where the fog is complete, which does not
    // make them a pale silhouette — it makes them exactly fog colour, and the
    // range was invisible. Far enough to read as distance, near enough that
    // some of their own colour still gets through.
    // Twelve half-widths, not eighteen. The haze is three-quarters complete by
    // eighteen and the range came out as a rumour on the skyline; at twelve
    // enough of the rock colour survives to read as a silhouette, and a city
    // ringed by hills a couple of kilometres out is an ordinary thing to be.
    let radius = (inner * 16.5).min(outer * 0.9);
    let snow_line = inner * 1.9;
    let rock = Color::from_hex(0x6d7079);
    let mut peaks = 0;
    // `(bearing, angular half-width)` per summit, for the gap measurement.
    let mut seen: Vec<(f32, f32)> = Vec::new();
    let mut summits: Vec<Summit> = Vec::new();
    // Ranges rather than an even ring: mountains come in chains with gaps.
    // Enough ranges, spread widely enough, that most bearings out of the city
    // have something on the skyline. Five narrow arcs covered about a third of
    // the circle, so which direction you looked decided whether this city had
    // mountains at all.
    let ranges = 9 + rng.below(6);
    // One sector of the horizon per range, jittered inside it, rather than a
    // free bearing for each. Uniformly random bearings leave holes: with a
    // dozen arcs averaging 0.63 rad thrown at 6.28 rad of circle, the expected
    // uncovered fraction is (1 - 0.63/6.28)^12, about a third. Measured over
    // five seeds it was worse than the model — 64% to 82% of the skyline
    // covered, with the largest hole running from 27 to 57 degrees. A 57
    // degree gap is a quarter of a view with no mountains in it, which is
    // exactly the complaint this was meant to have fixed.
    let sector = TAU / ranges as f32;
    for r in 0..ranges {
        // Jitter stays inside a third of a sector either way, so ranges cannot
        // pile onto one bearing and reopen the holes.
        let a0 = (r as f32 + 0.5) * sector + rng.range(-sector * 0.34, sector * 0.34);
        // At least a sector and a quarter wide, so neighbours overlap at the
        // ends — where a range's peaks are smallest and cover least.
        let arc = rng.range(0.34, 0.92).max(sector * 1.25);
        // Enough peaks that a wide arc is a chain rather than a row of
        // separate cones with sky between them.
        let n = (4 + rng.below(7)).max((arc / 0.16).ceil() as usize + 1);
        // One range shares a bearing offset and a height scale, so its peaks
        // rise and fall together instead of at random.
        // Tall. The haze is complete at twenty-six half-widths, so a range has
        // to stand inside that to be seen at all — and at three and a half
        // kilometres a three-hundred-metre summit subtends about five degrees,
        // which is a hill. These are four to eight half-widths high, which is
        // the difference between a lump on the skyline and a mountain.
        let base_h = inner * rng.range(2.6, 5.6);
        for i in 0..n {
            let t = i as f32 / (n - 1).max(1) as f32;
            let a = a0 + (t - 0.5) * arc;
            let rr = radius * rng.range(0.88, 1.14);
            let (x, z) = (rr * a.cos(), rr * a.sin());
            // Peaks fall away towards the ends of the range.
            let shape = (t * PI).sin().max(0.25);
            let h = base_h * shape * rng.range(0.75, 1.25);
            // Steeper than they were. A base radius larger than the height
            // gives a cone under twenty-five degrees, and a dozen of those
            // overlapping is not a range — it is one grey wall across a third
            // of the horizon, which is what this was.
            let foot = h * rng.range(0.42, 0.72);
            summits.push(Summit {
                x,
                z,
                foot,
                height: h,
            });
            let r = (x * x + z * z).sqrt();
            if r > 1.0 {
                seen.push((z.atan2(x).rem_euclid(TAU), (foot / r).min(0.999).asin()));
            }
            let segs = CONE_SIDES;
            // The cone, in two bands so the snow line is a real edge.
            let cut = ((snow_line / h).clamp(0.0, 1.0)).min(0.92);
            for (t0, t1, col) in [
                (0.0f32, cut, rock),
                (cut, 1.0f32, Color::from_hex(0xe8edf2)),
            ] {
                if t1 - t0 < 0.01 {
                    continue;
                }
                b.grass.add_limb(
                    Vector3::new(x, h * t0, z),
                    Vector3::new(x, h * t1, z),
                    foot * (1.0 - t0),
                    foot * (1.0 - t1),
                    segs,
                    scale_color(col, rng.range(0.9, 1.1)),
                    false,
                );
            }
            peaks += 1;
        }
    }
    // The largest stretch of horizon with nothing on it. Sort by bearing and
    // walk the arcs, carrying the reach of the widest one seen so far — the
    // arcs overlap heavily, so a plain neighbour-to-neighbour difference would
    // report gaps that a wider earlier peak already fills.
    seen.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut gap: f32 = 0.0;
    if let Some(&(a0, w0)) = seen.first() {
        // Start from the last arc's reach, wrapped, so the seam is measured
        // like any other bearing.
        let mut reach = seen
            .iter()
            .map(|&(a, w)| a + w - TAU)
            .fold(a0 - w0, f32::max);
        for &(a, w) in &seen {
            gap = gap.max(a - w - reach);
            reach = reach.max(a + w);
        }
        gap = gap.max(a0 - w0 + TAU - reach);
    }
    Mountains {
        peaks,
        gap: gap.max(0.0),
        summits,
    }
}

/// A hill: a broad, low mound with trees on it.
///
/// The countryside here is dead flat, and flat ground has no valleys in it —
/// a valley is only ever the space between two rises. So the rises come first
/// and the fields between them become the valley floor by default, which is
/// the cheap way to get relief into a world whose terrain is a handful of
/// enormous slabs that nothing displaces.
///
/// Broad and low on purpose: the ratio is what separates a hill from a slag
/// heap, and anything with a slope you would notice reads as spoil.
fn hill(b: &mut Batches, p: &Rect, lod: Lod, warm: bool, rng: &mut Rng) {
    let r = p.w().min(p.d()) * 0.52;
    let (cx, cz) = (p.cx(), p.cz());
    let h = r * rng.range(0.24, 0.46);
    let green = scale_color(
        Color::from_hex(if warm { 0x7f8f4a } else { 0x6b8f42 }),
        rng.range(0.9, 1.1),
    );
    // Concentric rings rather than a blob: a mound wants a flat-ish top and a
    // skirt that meets the ground tangentially, and stacked discs give that
    // for a fraction of the triangles a sphere would.
    let rings = if lod == Lod::Near { 5 } else { 3 };
    for k in 0..rings {
        let t = k as f32 / rings as f32;
        // Cosine profile: steepest half way up, flat at the top and the toe.
        let y = h * (1.0 - (t * PI * 0.5).cos());
        let rr = r * (1.0 - t * t * 0.86) * rng.range(0.94, 1.06);
        b.grass.add_ground_disc(
            cx,
            cz,
            rr,
            if lod == Lod::Near { 14 } else { 9 },
            0.0,
            0.06 + y,
            scale_color(green, 1.0 + t * 0.12),
        );
    }
    if lod == Lod::Far {
        return;
    }
    // Trees up the flanks, thinning towards the top — which is how a hill in
    // farmland is wooded, because the top is where the wind is.
    for _ in 0..(p.area() / 130.0) as usize {
        let a = rng.range(0.0, TAU);
        let u = rng.f().sqrt();
        let (x, z) = (cx + r * u * a.cos(), cz + r * u * a.sin());
        if rng.f() < u * 0.8 {
            continue;
        }
        let y = 0.06 + h * (1.0 - ((u * u).min(1.0) * PI * 0.5).cos()).max(0.0);
        add_tree_as(
            b,
            x,
            z,
            y,
            false,
            if rng.chance(0.5) { Species::Conifer } else { Species::Broadleaf },
            None,
            rng.range(0.6, 1.1),
            rng,
        );
    }
}

/// Hedges on all four sides. What marks worked land as somebody's.
fn hedge_sides(b: &mut Batches, p: &Rect, hex: u32, rng: &mut Rng) {
    let hedge = scale_color(Color::from_hex(hex), rng.range(0.85, 1.15));
    for (hx0, hz0, hx1, hz1) in [
        (p.x0, p.z0, p.x1, p.z0 + 1.4),
        (p.x0, p.z1 - 1.4, p.x1, p.z1),
        (p.x0, p.z0, p.x0 + 1.4, p.z1),
        (p.x1 - 1.4, p.z0, p.x1, p.z1),
    ] {
        b.foliage.add_box(
            Vector3::new(hx0, 0.0, hz0),
            Vector3::new(hx1, rng.range(1.6, 2.6), hz1),
            hedge,
            Uv::Unit,
        );
    }
}

fn lake(b: &mut Batches, p: &Rect, lod: Lod, rng: &mut Rng) {
    let r = (p.w().min(p.d()) * 0.42).max(4.0);
    let (cx, cz) = (p.cx(), p.cz());
    b.grass.add_slab(
        p.x0,
        p.z0,
        p.x1,
        p.z1,
        0.05,
        scale_color(Color::from_hex(0x5c7a3a), rng.range(0.9, 1.1)),
        Uv::Unit,
    );
    if lod == Lod::Far {
        // Water and nothing else. The bank ring, the reeds and the bankside
        // trees are two hundred triangles that resolve, at this range, to one
        // pixel of brown around a blue dot.
        b.water
            .add_ground_disc(cx, cz, r, 9, 0.20, 0.09, Color::from_hex(0x2a5468));
        return;
    }
    // A bank ring, then water inside it. No hole cutting out here: the terrain
    // is flat and unbuilt, so a rim sitting proud of it reads perfectly well
    // and costs a fraction of what the park ponds do.
    b.grass.add_ground_disc(
        cx,
        cz,
        r * 1.10,
        13,
        0.20,
        0.07,
        scale_color(Color::from_hex(0x6b6150), rng.range(0.9, 1.1)),
    );
    b.water
        .add_ground_disc(cx, cz, r, 13, 0.20, 0.09, Color::from_hex(0x2a5468));
    // Reeds and a few trees along the bank.
    for i in 0..14 {
        let a = i as f32 / 14.0 * TAU + rng.range(0.0, 0.3);
        let rr = r * rng.range(1.02, 1.14);
        b.foliage.add_box(
            Vector3::new(cx + rr * a.cos() - 0.5, 0.0, cz + rr * a.sin() - 0.5),
            Vector3::new(cx + rr * a.cos() + 0.5, rng.range(0.8, 1.6), cz + rr * a.sin() + 0.5),
            scale_color(Color::from_hex(0x5c7a34), rng.range(0.85, 1.15)),
            Uv::Unit,
        );
    }
    for _ in 0..(2 + rng.below(4)) {
        let a = rng.range(0.0, TAU);
        let rr = r * rng.range(1.25, 1.6);
        add_tree(b, cx + rr * a.cos(), cz + rr * a.sin(), 0.0, false, false, rng);
    }
}

fn pasture(b: &mut Batches, p: &Rect, warm: bool, rng: &mut Rng) {
    b.grass.add_slab(
        p.x0,
        p.z0,
        p.x1,
        p.z1,
        0.05,
        scale_color(Color::from_hex(0x5f7a38), rng.range(0.88, 1.14)),
        Uv::Unit,
    );
    // Rough ground: drifts of longer grass and a scatter of standing trees.
    for _ in 0..(p.area() / 260.0) as usize {
        b.grass.add_ground_disc(
            rng.range(p.x0, p.x1),
            rng.range(p.z0, p.z1),
            rng.range(3.0, 9.0),
            7,
            0.3,
            0.06,
            scale_color(Color::from_hex(0x6d7a3c), rng.range(0.85, 1.15)),
        );
    }
    for _ in 0..(p.area() / 700.0) as usize {
        add_tree(
            b,
            rng.range(p.x0 + 2.0, p.x1 - 2.0),
            rng.range(p.z0 + 2.0, p.z1 - 2.0),
            0.0,
            false,
            warm,
            rng,
        );
    }
}
