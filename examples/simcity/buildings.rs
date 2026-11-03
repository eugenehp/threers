//! Part of the `simcity` example; see `mod.rs`.
#![allow(dead_code)]

use super::*;

// ---------------------------------------------------------------------------
// Buildings.
// ---------------------------------------------------------------------------

/// Small plant, tanks and masts on a finished roof. This is most of what makes
/// a skyline read as a skyline rather than a bar chart.
pub(crate) fn add_roof_kit(b: &mut Batches, r: Rect, top: f32, tall: bool, rng: &mut Rng) {
    let metal = Color::from_hex(0x77787c);
    let dark = Color::from_hex(0x54565a);
    let n = 1 + rng.below(if r.area() > 260.0 { 4 } else { 2 });
    for _ in 0..n {
        let w = rng.range(1.4, 3.2).min(r.w() * 0.4);
        let d = rng.range(1.4, 3.0).min(r.d() * 0.4);
        let h = rng.range(0.7, 1.8);
        let cx = rng.range(r.x0 + w, r.x1 - w);
        let cz = rng.range(r.z0 + d, r.z1 - d);
        b.trim.add_box(
            Vector3::new(cx - w * 0.5, top, cz - d * 0.5),
            Vector3::new(cx + w * 0.5, top + h, cz + d * 0.5),
            scale_color(metal, rng.range(0.85, 1.1)),
            Uv::Unit,
        );
    }
    if rng.chance(0.35) && r.w() > 8.0 && r.d() > 8.0 {
        // Water tank on a short pedestal.
        let rr = rng.range(1.1, 1.8);
        let cx = rng.range(r.x0 + rr + 1.0, r.x1 - rr - 1.0);
        let cz = rng.range(r.z0 + rr + 1.0, r.z1 - rr - 1.0);
        b.trim.add_box(
            Vector3::new(cx - rr, top, cz - rr),
            Vector3::new(cx + rr, top + 1.2, cz + rr),
            dark,
            Uv::Unit,
        );
        b.trim.add_cylinder(
            Vector3::new(cx, top + 1.2, cz),
            rr,
            rr,
            rng.range(2.2, 3.6),
            10,
            Color::from_hex(0x6b5a48),
            true,
            Uv::Unit,
        );
    }
    if rng.chance(0.55) && r.w() > 6.0 && r.d() > 6.0 {
        // Stair and lift overrun. Taller than the plant around it, so it
        // reads on the skyline rather than only from directly above.
        let w = rng.range(2.4, 4.0).min(r.w() * 0.35);
        let d = rng.range(2.4, 3.6).min(r.d() * 0.35);
        let cx = rng.range(r.x0 + w, r.x1 - w);
        let cz = rng.range(r.z0 + d, r.z1 - d);
        b.trim.add_box(
            Vector3::new(cx - w * 0.5, top, cz - d * 0.5),
            Vector3::new(cx + w * 0.5, top + rng.range(2.4, 3.4), cz + d * 0.5),
            scale_color(Color::from_hex(0x8b8880), rng.range(0.9, 1.1)),
            Uv::Unit,
        );
    }
    if tall && rng.chance(0.5) {
        let h = rng.range(5.0, 16.0);
        b.trim.add_cylinder(
            Vector3::new(r.cx(), top, r.cz()),
            0.28,
            0.09,
            h,
            6,
            dark,
            false,
            Uv::Unit,
        );
        b.beacon.add_ellipsoid(
            Vector3::new(r.cx(), top + h + 0.25, r.cz()),
            0.34,
            0.34,
            0.34,
            4,
            8,
            Color::from_hex(0xff2a1a),
        );
    }
}

/// The massing of a tower. A city of nothing but boxes has no skyline; these
/// are the three shapes that carry most of a real one.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Form {
    /// Rectangular, with rectangular setbacks.
    Block,
    /// Corners cut off — an octagonal prism on a square plot.
    Chamfer,
    /// A round tower.
    Round,
    /// A hexagonal prism — narrower than the octagon and more obviously
    /// faceted, which reads differently again at a distance.
    Hex,
    /// Square, but tapering as it rises. The obelisk profile.
    Taper,
    /// A cruciform plan: a square with a bay pushed out on each face, which is
    /// what a great many pre-war towers actually are.
    Cross,
}

impl Form {
    pub(crate) fn sides(self) -> usize {
        match self {
            Form::Block | Form::Taper | Form::Cross => 4,
            Form::Hex => 6,
            Form::Chamfer => 8,
            Form::Round => 20,
        }
    }

    /// Whether this form needs a roughly square plot. A prism inscribed in a
    /// long thin parcel wastes most of it.
    pub(crate) fn wants_square(self) -> bool {
        matches!(self, Form::Chamfer | Form::Round | Form::Hex | Form::Taper)
    }
}

/// One tier's mass. A block gets a box; the prism forms get a regular polygon
/// inscribed in the plot, which is why they are only chosen for square-ish
/// ones.
#[allow(clippy::too_many_arguments)]
pub(crate) fn add_mass(
    b: &mut Batches,
    style: usize,
    form: Form,
    r: Rect,
    y: f32,
    h: f32,
    tint: Color,
    uv: Uv,
) {
    if form == Form::Block {
        b.facades[style].add_box(
            Vector3::new(r.x0, y, r.z0),
            Vector3::new(r.x1, y + h, r.z1),
            tint,
            uv,
        );
    } else if form == Form::Cross {
        // A square core with a bay pushed out on each face. Four boxes and a
        // middle, which from any angle gives a stepped silhouette rather than
        // a flat one.
        let (iw, id) = (r.w() * 0.30, r.d() * 0.30);
        b.facades[style].add_box(
            Vector3::new(r.x0 + iw, y, r.z0),
            Vector3::new(r.x1 - iw, y + h, r.z1),
            tint,
            uv,
        );
        b.facades[style].add_box(
            Vector3::new(r.x0, y, r.z0 + id),
            Vector3::new(r.x1, y + h, r.z1 - id),
            tint,
            uv,
        );
    } else if form == Form::Taper {
        // Same footprint, drawn as a frustum: a cylinder with four sides and a
        // smaller top is a square obelisk.
        let radius = r.w().min(r.d()) * 0.5;
        b.facades[style].add_cylinder(
            Vector3::new(r.cx(), y, r.cz()),
            radius * std::f32::consts::SQRT_2,
            radius * std::f32::consts::SQRT_2 * 0.72,
            h,
            4,
            tint,
            false,
            uv,
        );
    } else {
        let radius = r.w().min(r.d()) * 0.5;
        b.facades[style].add_cylinder(
            Vector3::new(r.cx(), y, r.cz()),
            radius,
            radius,
            h,
            form.sides(),
            tint,
            false,
            uv,
        );
    }
}

/// One parcel's worth of building: plinth, stacked and set-back tiers, a
/// parapet and roof per tier, and roof plant on top.
pub(crate) fn add_building(
    b: &mut Batches,
    plan: &LayoutPlan,
    lot: Rect,
    zone: f32,
    landmark: bool,
    rng: &mut Rng,
) -> bool {
    // Cap the setback on small plots: a fixed one eats a house lot whole and
    // the whole block ends up paved over instead of built on.
    let setback = rng.range(0.4, 1.8).min(lot.w().min(lot.d()) * 0.10);
    let plot = lot.inset(setback);
    let Some((x0, x1)) = snap_span(plot.x0, plot.x1, BAY, 2.0) else {
        return false;
    };
    let Some((z0, z1)) = snap_span(plot.z0, plot.z1, BAY, 2.0) else {
        return false;
    };

    // The building takes its plot before anything is scattered near it, so a
    // street tree cannot grow through a wall and a parked car cannot end up
    // inside a shopfront.
    b.occ.claim([x0 - 0.5, z0 - 0.5, x1 + 0.5, z1 + 0.5]);

    // Height falls off from the core, with enough noise that the drop is not a
    // clean cone, plus the occasional landmark that breaks the envelope.
    let core = smoothstep(plan.core_reach, 0.06, zone);
    let mut floors = (mix(1.6, plan.peak_floors, core.powf(1.7)) * rng.range(0.55, 1.45)).round();
    // One plot in forty is a building site rather than a building. Rare
    // enough to be an event, common enough that a large city has two or three
    // cranes on its skyline — and a crane is the one piece of a skyline that
    // says the skyline is still changing.
    if floors >= 4.0 && rng.chance(0.05) {
        add_construction(b, lot, floors, rng);
        return true;
    }

    if plan.peak_floors >= 14.0 && zone < 0.24 && rng.chance(0.10) {
        floors *= rng.range(1.4, 1.9);
    }
    // The landmark: one building a city, head and shoulders above the rest.
    // A skyline needs a peak, and a purely statistical height distribution
    // gives it a plateau instead.
    if landmark {
        floors = floors.max(plan.peak_floors) * rng.range(1.7, 2.1);
    }
    let floors = floors.round().max(1.0);

    // A one-or-two storey box on a wide suburban lot is a house, not an
    // office: give it a pitched roof instead of a parapet.
    // Two storeys was the whole of "house". Three is a townhouse and one is a
    // bungalow, and a street that is all the same height is a housing estate
    // rather than a suburb.
    let house = floors <= 3.0 && zone > 0.52;
    // 0 curtain wall, 1 precast, 2 brick, 3 ribbon glazing, 4 bronze glass.
    let style = if house {
        2
    } else if floors >= 13.0 {
        match rng.below(10) {
            0..=4 => 0,
            5..=6 => 4,
            7..=8 => 1,
            _ => 3,
        }
    } else if floors >= 5.0 {
        match rng.below(10) {
            0..=2 => 1,
            3..=4 => 0,
            5..=6 => 3,
            7 => 4,
            _ => 2,
        }
    } else if rng.chance(0.62) {
        2
    } else if rng.chance(0.5) {
        1
    } else {
        3
    };

    // Prisms only on a square-ish plot: inscribing a circle in a long thin
    // lot throws away most of it.
    let squareish = (x1 - x0).max(z1 - z0) / (x1 - x0).min(z1 - z0) < 1.3;
    let form = if floors < 6.0 {
        Form::Block
    } else if !squareish {
        // A long plot cannot take a prism, but it can take a cruciform: the
        // bays push out along its length rather than being inscribed in it.
        if rng.chance(0.22) {
            Form::Cross
        } else {
            Form::Block
        }
    } else {
        match rng.below(100) {
            0..=13 => Form::Round,
            14..=31 => Form::Chamfer,
            32..=41 => Form::Hex,
            42..=50 => Form::Taper,
            51..=62 => Form::Cross,
            _ => Form::Block,
        }
    };

    let tint = Color::new(
        rng.range(0.80, 1.14),
        rng.range(0.82, 1.10),
        rng.range(0.80, 1.12),
    );
    let uv = Uv::World {
        u: 1.0 / (TILE * BAY),
        v: 1.0 / (TILE * FLOOR),
    };

    if house {
        let h = floors * FLOOR * 0.85;
        b.facades[style].add_box(
            Vector3::new(x0, KERB, z0),
            Vector3::new(x1, KERB + h, z1),
            tint,
            uv,
        );
        let slate = scale_color(Color::from_hex(0x5a4038), rng.range(0.8, 1.25));
        let eaves = 0.35;
        // Roof form. A pitched lid on everything is what made every house on
        // the street the same house; the shape of the roof is most of what
        // distinguishes one from the next at any distance.
        let roof = rng.below(5);
        match roof {
            // Hipped: a gable with the ends brought in, so it reads as four
            // slopes rather than two and a wall.
            0 => {
                let rise = rng.range(1.5, 2.4);
                b.trim.add_gable(
                    Vector3::new(x0 - eaves, KERB + h, z0 - eaves),
                    Vector3::new(x1 + eaves, KERB + h, z1 + eaves),
                    rise,
                    slate,
                );
                for (ex0, ez0, ex1, ez1) in [
                    (x0 - eaves, z0 - eaves, x1 + eaves, z0 + 0.9),
                    (x0 - eaves, z1 - 0.9, x1 + eaves, z1 + eaves),
                ] {
                    b.trim.add_gable(
                        Vector3::new(ex0, KERB + h, ez0),
                        Vector3::new(ex1, KERB + h, ez1),
                        rise * 0.55,
                        scale_color(slate, 0.94),
                    );
                }
            }
            // Mansard: a steep lower slope with a shallow deck on top, which
            // is what an extra storey in the roof looks like.
            1 => {
                b.trim.add_box(
                    Vector3::new(x0 - 0.15, KERB + h, z0 - 0.15),
                    Vector3::new(x1 + 0.15, KERB + h + 1.5, z1 + 0.15),
                    scale_color(slate, 0.86),
                    Uv::Unit,
                );
                b.trim.add_gable(
                    Vector3::new(x0 - eaves, KERB + h + 1.5, z0 - eaves),
                    Vector3::new(x1 + eaves, KERB + h + 1.5, z1 + eaves),
                    0.7,
                    slate,
                );
                // Dormers in the mansard, which is the point of one.
                let n = ((x1 - x0) / 3.2).floor().max(1.0) as usize;
                for i in 0..n {
                    let dx = mix(x0 + 1.2, x1 - 1.2, (i as f32 + 0.5) / n as f32);
                    b.trim.add_box(
                        Vector3::new(dx - 0.55, KERB + h + 0.25, z0 - 0.3),
                        Vector3::new(dx + 0.55, KERB + h + 1.35, z0 + 0.5),
                        scale_color(slate, 1.15),
                        Uv::Unit,
                    );
                    b.glow.add_box(
                        Vector3::new(dx - 0.38, KERB + h + 0.45, z0 - 0.34),
                        Vector3::new(dx + 0.38, KERB + h + 1.15, z0 - 0.28),
                        Color::from_hex(0xffe0a8),
                        Uv::Unit,
                    );
                }
            }
            // Cross-gable: a second ridge at right angles to the first.
            2 => {
                let rise = rng.range(1.7, 2.6);
                b.trim.add_gable(
                    Vector3::new(x0 - eaves, KERB + h, z0 - eaves),
                    Vector3::new(x1 + eaves, KERB + h, z1 + eaves),
                    rise,
                    slate,
                );
                let w = (x1 - x0) * 0.42;
                let cx = mix(x0 + w, x1 - w, rng.f());
                b.trim.add_gable(
                    Vector3::new(cx - w * 0.5, KERB + h, z0 - eaves - 0.6),
                    Vector3::new(cx + w * 0.5, KERB + h, (z0 + z1) * 0.5),
                    rise * 0.92,
                    scale_color(slate, 1.06),
                );
            }
            // Flat with a parapet: the post-war infill on any street.
            3 => {
                b.trim.add_slab(
                    x0 - 0.2,
                    z0 - 0.2,
                    x1 + 0.2,
                    z1 + 0.2,
                    KERB + h + 0.02,
                    scale_color(Color::from_hex(C_ROOF), rng.range(0.9, 1.1)),
                    Uv::Unit,
                );
                for (px0, pz0, px1, pz1) in [
                    (x0 - 0.24, z0 - 0.24, x1 + 0.24, z0 + 0.06),
                    (x0 - 0.24, z1 - 0.06, x1 + 0.24, z1 + 0.24),
                    (x0 - 0.24, z0, x0 + 0.06, z1),
                    (x1 - 0.06, z0, x1 + 0.24, z1),
                ] {
                    b.trim.add_box(
                        Vector3::new(px0, KERB + h, pz0),
                        Vector3::new(px1, KERB + h + 0.42, pz1),
                        scale_color(tint, 0.92),
                        Uv::Unit,
                    );
                }
            }
            // The plain gable it always had.
            _ => {
                b.trim.add_gable(
                    Vector3::new(x0 - eaves, KERB + h, z0 - eaves),
                    Vector3::new(x1 + eaves, KERB + h, z1 + eaves),
                    rng.range(1.6, 2.8),
                    slate,
                );
            }
        }
        // A chimney, and a porch over the door. Both are small, and both are
        // most of what makes a box with a pitched lid read as a house.
        if roof != 3 {
            let stack = scale_color(Color::from_hex(0x6b4a3c), rng.range(0.85, 1.15));
            let cx = mix(x0 + 1.4, x1 - 1.4, rng.f());
            b.trim.add_box(
                Vector3::new(cx - 0.42, KERB + h - 0.2, (z0 + z1) * 0.5 - 0.42),
                Vector3::new(cx + 0.42, KERB + h + rng.range(2.2, 3.2), (z0 + z1) * 0.5 + 0.42),
                stack,
                Uv::Unit,
            );
        }
        let door = mix(x0 + 1.2, x1 - 1.2, rng.f());
        b.trim.add_box(
            Vector3::new(door - 1.0, KERB + 2.3, z0 - 1.2),
            Vector3::new(door + 1.0, KERB + 2.5, z0),
            Color::from_hex(0x8f8a80),
            Uv::Unit,
        );
        // Porch posts, so the canopy is held up by something.
        for s in [-1.0f32, 1.0] {
            b.trim.add_cylinder(
                Vector3::new(door + s * 0.9, KERB, z0 - 1.05),
                0.075,
                0.070,
                2.3,
                5,
                Color::from_hex(0xd8d3c8),
                false,
                Uv::Unit,
            );
        }
        // Window boxes on the front, at first-floor sill height.
        if rng.chance(0.5) {
            let n = ((x1 - x0) / 3.2).floor().max(1.0) as usize;
            for i in 0..n {
                if !rng.chance(0.6) {
                    continue;
                }
                let wx = mix(x0 + 1.2, x1 - 1.2, (i as f32 + 0.5) / n as f32);
                add_window_box(b, wx, KERB + FLOOR * 1.05, z0, -1.0, rng);
            }
        }

        // A bay window on the front, which is the other half of a terrace.
        if rng.chance(0.42) && x1 - x0 > 7.0 {
            let bx = mix(x0 + 1.6, x1 - 1.6, rng.f());
            if (bx - door).abs() > 2.6 {
                b.facades[style].add_box(
                    Vector3::new(bx - 1.15, KERB, z0 - 0.9),
                    Vector3::new(bx + 1.15, KERB + FLOOR * 0.86, z0 + 0.1),
                    tint,
                    uv,
                );
                b.trim.add_slab(
                    bx - 1.3,
                    z0 - 1.05,
                    bx + 1.3,
                    z0 + 0.1,
                    KERB + FLOOR * 0.86,
                    slate,
                    Uv::Unit,
                );
            }
        }
        // Garage: attached to one flank, with a door and a drive out to the
        // kerb. A suburb without them is a film set.
        if lot.w() > 12.0 && rng.chance(0.55) {
            let side = if rng.chance(0.5) { 1.0f32 } else { -1.0 };
            let gx = if side > 0.0 { x1 } else { x0 - 3.2 };
            let (gx0, gx1) = (gx, gx + 3.2);
            if gx0 > lot.x0 + 0.4 && gx1 < lot.x1 - 0.4 {
                let gh = 2.6;
                b.facades[style].add_box(
                    Vector3::new(gx0, KERB, z0 + 0.4),
                    Vector3::new(gx1, KERB + gh, z0 + 6.0),
                    tint,
                    uv,
                );
                b.trim.add_gable(
                    Vector3::new(gx0 - 0.2, KERB + gh, z0 + 0.2),
                    Vector3::new(gx1 + 0.2, KERB + gh, z0 + 6.2),
                    0.8,
                    slate,
                );
                // The door: panelled, and a different material from the wall.
                b.trim.add_box(
                    Vector3::new(gx0 + 0.25, KERB, z0 + 0.32),
                    Vector3::new(gx1 - 0.25, KERB + 2.1, z0 + 0.44),
                    scale_color(Color::from_hex(0xb8b2a4), rng.range(0.85, 1.15)),
                    Uv::Unit,
                );
                // Drive.
                b.road.add_slab(
                    gx0 + 0.2,
                    lot.z0,
                    gx1 - 0.2,
                    z0 + 0.4,
                    KERB + 0.01,
                    scale_color(Color::from_hex(0x5c5f63), rng.range(0.9, 1.1)),
                    Uv::Unit,
                );
            }
        }
        // The back garden. Fenced, and with something in it.
        if lot.z1 - z1 > 3.0 {
            add_back_garden(
                b,
                Rect {
                    x0: lot.x0 + 0.3,
                    z0: z1 + 0.2,
                    x1: lot.x1 - 0.3,
                    z1: lot.z1 - 0.3,
                },
                rng,
            );
        }

        // The front garden. A path to the door and nothing else is a verge;
        // what makes it a garden is a low wall, a bed against the house and
        // something clipped by the gate.
        if lot.z0 < z0 - 2.0 {
            let front = Rect {
                x0: lot.x0 + 0.4,
                z0: lot.z0 + 0.3,
                x1: lot.x1 - 0.4,
                z1: z0 - 0.4,
            };
            if front.w() > 3.0 && front.d() > 1.2 {
                b.grass.add_slab(
                    front.x0,
                    front.z0,
                    front.x1,
                    front.z1,
                    KERB + 0.012,
                    scale_color(Color::from_hex(C_GRASS), rng.range(0.9, 1.15)),
                    Uv::Unit,
                );
                // Dwarf wall along the pavement, with a gap for the gate.
                let wall = scale_color(Color::from_hex(0x8d857a), rng.range(0.88, 1.12));
                let gate = mix(front.x0 + 1.0, front.x1 - 1.0, rng.f());
                for (wx0, wx1) in [(front.x0, gate - 0.6), (gate + 0.6, front.x1)] {
                    if wx1 - wx0 > 0.3 {
                        b.trim.add_box(
                            Vector3::new(wx0, KERB, front.z0 - 0.12),
                            Vector3::new(wx1, KERB + rng.range(0.5, 0.8), front.z0 + 0.12),
                            wall,
                            Uv::Unit,
                        );
                    }
                }
                // A planted bed against the house, and a shrub or two.
                add_flowers(
                    b,
                    Vector3::new(front.x0 + 0.4, KERB + 0.02, front.z1 - 0.35),
                    Vector3::new(front.x1 - 0.4, KERB + 0.02, front.z1 - 0.35),
                    0.34,
                    0.42,
                    rng,
                );
                for _ in 0..(1 + rng.below(3)) {
                    let sx = rng.range(front.x0 + 0.5, front.x1 - 0.5);
                    let sz = rng.range(front.z0 + 0.4, front.z1 - 0.8);
                    b.foliage.add_blob(
                        Vector3::new(sx, KERB + 0.34, sz),
                        rng.range(0.34, 0.6),
                        rng.range(0.30, 0.52),
                        rng.range(0.34, 0.6),
                        4,
                        7,
                        0.22,
                        rng.next_u32() as i32 & 0xffff,
                        0.45,
                        scale_color(Color::from_hex(0x35602a), rng.range(0.85, 1.15)),
                    );
                }
            }
        }

        // Garden path out to the kerb.
        if lot.w() > 9.0 {
            b.pads.add_slab(
                lot.cx() - 0.7,
                lot.z0,
                lot.cx() + 0.7,
                z0,
                KERB + 0.01,
                Color::from_hex(C_CONCRETE),
                Uv::Unit,
            );
        }
        return true;
    }

    // Plinth: a slightly wider, darker base storey. Reads as retail frontage
    // and stops every tower from meeting the pavement with a hard edge.
    let plinth_h = FLOOR * rng.range(0.9, 1.3);
    b.trim.add_box(
        Vector3::new(x0 - 0.35, KERB, z0 - 0.35),
        Vector3::new(x1 + 0.35, KERB + plinth_h, z1 + 0.35),
        scale_color(Color::from_hex(0x6d6a64), rng.range(0.8, 1.15)),
        Uv::Unit,
    );

    // Entrance canopy. Small, but it is the thing that tells you where the
    // door is from street level.
    if rng.chance(0.6) {
        let door = mix(x0 + 2.0, x1 - 2.0, rng.f());
        b.trim.add_box(
            Vector3::new(door - 2.2, KERB + 3.1, z0 - 2.4),
            Vector3::new(door + 2.2, KERB + 3.35, z0 - 0.35),
            Color::from_hex(0x5c5f64),
            Uv::Unit,
        );
    }

    // Shopfront glazing round the plinth. Unlit and night-only, so a lit
    // ground floor spills onto the pavement after dark and costs nothing in
    // daylight. Sits just proud of the plinth so it is not z-fighting it.
    if rng.chance(0.78 - 0.38 * zone.clamp(0.0, 1.0)) {
        // A *band* of glazing, not a box wrapped round the whole plinth. The
        // box read as a metre-and-a-half-tall ribbon of light round every
        // building in the city, all the same colour, and with bloom on it that
        // is what the night skyline was made of.
        //
        // Colour per shop, too. A parade of units is a butcher's fluorescent
        // white next to a bar's warm amber next to a shuttered one showing
        // nothing at all — the uniform cream was the other half of why every
        // night render looked the same.
        const SHOPFRONT: [u32; 6] = [
            0xffe6bc, 0xfff4e2, 0xe8f0ff, 0xffd28a, 0xd8ffe6, 0xffc4d2,
        ];
        let sill = KERB + 0.95;
        let head = (KERB + plinth_h * 0.62).min(sill + 1.35);
        if head > sill + 0.3 {
            // One unit every few metres along each frontage, lit or not.
            for (a0, a1, fixed, along_x) in [
                (x0, x1, z0 - 0.42, true),
                (x0, x1, z1 + 0.42, true),
                (z0, z1, x0 - 0.42, false),
                (z0, z1, x1 + 0.42, false),
            ] {
                let units = (((a1 - a0) / 4.5).round() as usize).max(1);
                for u in 0..units {
                    // A shuttered unit is as much a part of a parade as a lit
                    // one, and a run of them all lit is a shopping centre.
                    if !rng.chance(0.62) {
                        continue;
                    }
                    let (u0, u1) = (
                        mix(a0, a1, u as f32 / units as f32) + 0.35,
                        mix(a0, a1, (u + 1) as f32 / units as f32) - 0.35,
                    );
                    if u1 - u0 < 0.6 {
                        continue;
                    }
                    let tint = scale_color(
                        Color::from_hex(SHOPFRONT[rng.below(SHOPFRONT.len())]),
                        rng.range(0.55, 1.0),
                    );
                    let (lo, hi) = if along_x {
                        (
                            Vector3::new(u0, sill, fixed - 0.06),
                            Vector3::new(u1, head, fixed + 0.06),
                        )
                    } else {
                        (
                            Vector3::new(fixed - 0.06, sill, u0),
                            Vector3::new(fixed + 0.06, head, u1),
                        )
                    };
                    b.glow.add_box(lo, hi, tint, Uv::Unit);
                }
            }
        }
    }

    // A fire escape on the brick walk-ups, and an awning over the shop.
    let front = if rng.chance(0.5) { 1.0 } else { -1.0 };
    let face_z = if front > 0.0 { z1 + 0.36 } else { z0 - 0.36 };
    if style == 2 && (4.0..10.0).contains(&floors) && rng.chance(0.6) {
        add_fire_escape(b, x0 + 0.4, x1 - 0.4, face_z, front, KERB + plinth_h, floors);
    }
    if rng.chance(0.55) && x1 - x0 > 5.0 {
        let shop = mix(x0 + 2.2, x1 - 2.2, rng.f());
        add_awning(
            b,
            shop - 1.9,
            shop + 1.9,
            face_z,
            front,
            KERB + (plinth_h * 0.78).min(3.0),
            mix_color(
                Color::from_hex(0x8c3b32),
                Color::from_hex(0x2f5b6e),
                rng.f(),
            ),
        );
    }

    // --- Advertising. Where it goes follows what the building is: shops sign
    // their own fascia, a blank flank gets sold to whoever wants it, and a
    // low roof downtown carries a hoarding aimed at the taller blocks around
    // it. Nothing goes above the fourth floor of a tower, because nobody
    // reads a poster from up there.
    // Two frontages, not one: the street the awning faces, and — for a corner
    // plot, which most of these are — the one round the side. Signing only the
    // Z-facing wall leaves every north-south avenue in the city bare.
    let fascia_y = KERB + (plinth_h * 0.80).min(3.4);
    if floors <= 14.0 && rng.chance(0.80) {
        add_shop_signage(b, x0 + 0.6, x1 - 0.6, face_z, front, true, fascia_y, rng);
    }
    if floors <= 14.0 && z1 - z0 > 5.0 && rng.chance(0.62) {
        let side = if rng.chance(0.5) { 1.0f32 } else { -1.0 };
        let face_x = if side > 0.0 { x1 + 0.36 } else { x0 - 0.36 };
        add_shop_signage(b, z0 + 0.6, z1 - 0.6, face_x, side, false, fascia_y, rng);
    }
    // A big wall panel wants a wall with nothing else on it, which in practice
    // means the side that did not get the fire escape or the awning.
    let hoarding = rng.chance(0.30 + 0.22 * (1.0 - zone.clamp(0.0, 1.0)));
    if hoarding && floors >= 3.0 {
        // On the blank side — the one that did not get the awning, the fire
        // escape and the fascia.
        let base = KERB + plinth_h + FLOOR * rng.range(0.4, 1.6);
        if x1 - x0 > 9.0 && rng.chance(0.5) {
            let back = -front;
            let back_z = if back > 0.0 { z1 + 0.36 } else { z0 - 0.36 };
            add_wall_billboard(b, x0, x1, back_z, back, true, base, rng);
        } else if z1 - z0 > 9.0 {
            let side = if rng.chance(0.5) { 1.0f32 } else { -1.0 };
            let face_x = if side > 0.0 { x1 + 0.36 } else { x0 - 0.36 };
            add_wall_billboard(b, z0, z1, face_x, side, false, base, rng);
        }
    }

    let tiers = if floors >= 17.0 {
        3
    } else if floors >= 9.0 {
        2
    } else {
        1
    };
    let mut rect = Rect { x0, z0, x1, z1 };
    let mut y = KERB + plinth_h;
    let mut left = floors;
    for t in 0..tiers {
        let last = t == tiers - 1;
        let f = if last {
            left
        } else {
            (left * rng.range(0.45, 0.72)).round().max(1.0)
        };
        let h = f * FLOOR;
        add_mass(b, style, form, rect, y, h, tint, uv);
        let top = y + h;
        // Roof deck. The facade box has a top face too, but it samples the
        // window tile from above — this covers it with something roof-like.
        b.trim.add_slab(
            rect.x0,
            rect.z0,
            rect.x1,
            rect.z1,
            top + 0.02,
            scale_color(Color::from_hex(C_ROOF), rng.range(0.85, 1.15)),
            Uv::Unit,
        );
        // Parapet: four walls, not a slab. A solid box here would cap the
        // roof and hide everything standing on it.
        let ph = rng.range(0.55, 1.0);
        let pc = scale_color(Color::from_hex(0x8a877f), rng.range(0.9, 1.1));
        let (px0, pz0) = (rect.x0 - 0.22, rect.z0 - 0.22);
        let (px1, pz1) = (rect.x1 + 0.22, rect.z1 + 0.22);
        let wall = 0.44f32.min(rect.w().min(rect.d()) * 0.2);
        for (bx0, bz0, bx1, bz1) in [
            (px0, pz0, px1, pz0 + wall),
            (px0, pz1 - wall, px1, pz1),
            (px0, pz0 + wall, px0 + wall, pz1 - wall),
            (px1 - wall, pz0 + wall, px1, pz1 - wall),
        ] {
            b.trim.add_box(
                Vector3::new(bx0, top, bz0),
                Vector3::new(bx1, top + ph, bz1),
                pc,
                Uv::Unit,
            );
        }
        add_roof_kit(b, rect, top + 0.04, last && floors >= 12.0, rng);
        // A pad on the roof of a tall block, where there is deck to put one
        // on and a reason to have it.
        // The top tier is set back, so it is usually a good deal narrower
        // than the block below — requiring sixteen metres of it meant almost
        // nothing qualified.
        if last && floors >= 12.0 && rect.w().min(rect.d()) > 11.0 && rng.chance(0.6) {
            add_helipad(
                b,
                rect.cx(),
                rect.cz(),
                (rect.w().min(rect.d()) * 0.34).clamp(3.5, 9.0),
                top + 0.06,
                rng,
            );
        }
        // A hoarding on the roof of a low block, aimed across at whatever is
        // taller nearby. Only on the top tier, and only where there is roof to
        // stand it on.
        if last && (2.0..11.0).contains(&floors) && rng.chance(0.34) {
            add_roof_billboard(b, rect, top + ph * 0.5, rng);
        }
        if last && floors >= 15.0 {
            add_crown(b, form, rect, top, rng);
        }

        if last {
            break;
        }
        left -= f;
        // Step in on a random subset of the four sides rather than all of
        // them: a tower that always insets symmetrically is a wedding cake,
        // and a setback with a direction is what a real envelope produces.
        // Both edges stay on the bay grid, so `snap_span` returns them
        // unchanged and the asymmetry survives.
        let mut step = [0.0f32; 4];
        for e in step.iter_mut() {
            if rng.chance(0.62) {
                *e = BAY;
            }
        }
        if step.iter().all(|e| *e == 0.0) {
            step[rng.below(4)] = BAY;
        }
        let nx = snap_span(rect.x0 + step[0], rect.x1 - step[1], BAY, 2.0);
        let nz = snap_span(rect.z0 + step[2], rect.z1 - step[3], BAY, 2.0);
        match (nx, nz) {
            (Some((ax0, ax1)), Some((az0, az1))) => {
                rect = Rect {
                    x0: ax0,
                    z0: az0,
                    x1: ax1,
                    z1: az1,
                };
            }
            // Too slim to step: run the remaining floors straight up.
            _ => {
                let h2 = left * FLOOR;
                add_mass(b, style, form, rect, top, h2, tint, uv);
                b.trim.add_slab(
                    rect.x0,
                    rect.z0,
                    rect.x1,
                    rect.z1,
                    top + h2 + 0.02,
                    Color::from_hex(C_ROOF),
                    Uv::Unit,
                );
                add_roof_kit(b, rect, top + h2 + 0.04, floors >= 12.0, rng);
                break;
            }
        }
        y = top;
    }

    // --- Surveillance.
    //
    // High on a ground-floor corner, angled out and down along the frontage —
    // which is where every one of these actually is, and why they read as
    // cameras at a glance even when they are four pixels across. Not on the
    // roof: a camera on a roof is watching nothing.
    //
    // Commoner on the big commercial buildings than on houses, which is also
    // true, and one in seven gets the dome instead.
    let watched = if house { 0.14 } else { 0.34 + 0.30 * (1.0 - zone) };
    if rng.chance(watched.clamp(0.0, 0.85)) {
        let h = KERB + 4.2 + rng.range(0.0, 1.4);
        let corners = [
            (x0, z0, -1.0f32, -1.0f32),
            (x1, z0, 1.0, -1.0),
            (x1, z1, 1.0, 1.0),
            (x0, z1, -1.0, 1.0),
        ];
        let n = if rng.chance(0.30) { 2 } else { 1 };
        let first = rng.below(4);
        for k in 0..n {
            let (cx, cz, sx, sz) = corners[(first + k * 2) % 4];
            let kind = if rng.chance(0.15) { Cctv::Dome } else { Cctv::Bullet };
            add_cctv(
                b,
                Vector3::new(cx + sx * 0.10, h, cz + sz * 0.10),
                // Out along the diagonal, so it covers both elevations that
                // meet at the corner.
                Vector3::new(sx, 0.0, sz),
                kind,
                rng,
            );
        }
    }
    true
}

/// What the top of a tall building does against the sky. A flat parapet at
/// 90 m reads as an unfinished box; a crown is the cheapest thing that fixes
/// a skyline.
pub(crate) fn add_crown(b: &mut Batches, form: Form, r: Rect, top: f32, rng: &mut Rng) {
    let stone = scale_color(Color::from_hex(0x8e8b83), rng.range(0.88, 1.12));
    match form {
        Form::Block => {
            // Stepped setbacks, shrinking and shortening — the deco move.
            let mut cur = r;
            let mut y = top;
            for i in 0..3 {
                let inset = cur.w().min(cur.d()) * rng.range(0.12, 0.20);
                cur = cur.inset(inset);
                if cur.w() < 1.5 || cur.d() < 1.5 {
                    break;
                }
                let h = rng.range(2.4, 4.2) * (1.0 - i as f32 * 0.22);
                b.trim.add_box(
                    Vector3::new(cur.x0, y, cur.z0),
                    Vector3::new(cur.x1, y + h, cur.z1),
                    stone,
                    Uv::Unit,
                );
                y += h;
            }
        }
        _ => {
            let radius = r.w().min(r.d()) * 0.5;
            b.trim.add_cylinder(
                Vector3::new(r.cx(), top, r.cz()),
                radius * 1.04,
                radius * rng.range(0.35, 0.60),
                rng.range(5.0, 11.0),
                form.sides(),
                stone,
                true,
                Uv::Unit,
            );
        }
    }
}

/// A fire escape zig-zagging up one face. The single most recognisable thing
/// on a walk-up, and cheap: a platform, a rail and two diagonal stringers a
/// floor, with `add_limb` doing the diagonals.
pub(crate) fn add_fire_escape(
    b: &mut Batches,
    x0: f32,
    x1: f32,
    z_face: f32,
    // +1 puts it on the +Z face, -1 on the -Z face. Which one is the street
    // varies block by block, and a fire escape on the hidden side is no use.
    front: f32,
    base: f32,
    floors: f32,
) {
    let iron = Color::from_hex(0x3f4348);
    let cx = (x0 + x1) * 0.5;
    let out = 1.15 * front;
    let half = 0.72;
    let mut side = 1.0f32;
    for i in 1..floors as usize {
        let y = base + i as f32 * FLOOR;
        // Platform.
        b.trim.add_box(
            Vector3::new(cx - half, y, z_face - out),
            Vector3::new(cx + half, y + 0.06, z_face),
            iron,
            Uv::Unit,
        );
        // Rail: two posts and a top rail, along the outer edge.
        for px in [cx - half, cx + half] {
            b.trim.add_box(
                Vector3::new(px - 0.035, y, z_face - out),
                Vector3::new(px + 0.035, y + 0.95, z_face - out + 0.07),
                iron,
                Uv::Unit,
            );
        }
        b.trim.add_box(
            Vector3::new(cx - half, y + 0.88, z_face - out),
            Vector3::new(cx + half, y + 0.95, z_face - out + 0.07),
            iron,
            Uv::Unit,
        );
        // Stair to the floor above, alternating which end it starts from.
        if i + 1 < floors as usize {
            for off in [-0.28f32, 0.28] {
                b.trim.add_limb(
                    Vector3::new(cx + side * (half - 0.1) + off * 0.0, y + 0.06, z_face - out + 0.15),
                    Vector3::new(cx - side * (half - 0.1), y + FLOOR, z_face - out + 0.85),
                    0.035,
                    0.035,
                    4,
                    iron,
                    false,
                );
                let _ = off;
            }
        }
        side = -side;
    }
}

/// A shop awning: a sloped double-sided quad with a valance hanging off the
/// front. One quad is single-sided, and an awning seen from underneath is the
/// whole point of an awning.
pub(crate) fn add_awning(b: &mut Batches, x0: f32, x1: f32, z: f32, front: f32, y: f32, c: Color) {
    let depth = 1.35 * front;
    let drop = 0.42;
    let corners = [
        [x0, y, z],
        [x1, y, z],
        [x1, y - drop, z - depth],
        [x0, y - drop, z - depth],
    ];
    let n = Vector3::new(0.0, depth.abs(), drop * front).normalize();
    b.trim.quad(
        [corners[0], corners[1], corners[2], corners[3]],
        [0.0, n.y, n.z],
        Uv::Unit,
        c,
    );
    b.trim.quad(
        [corners[3], corners[2], corners[1], corners[0]],
        [0.0, -n.y, -n.z],
        Uv::Unit,
        scale_color(c, 0.72),
    );
    // Valance.
    b.trim.add_box(
        Vector3::new(x0, y - drop - 0.22, (z - depth) - 0.03),
        Vector3::new(x1, y - drop, (z - depth) + 0.03),
        scale_color(c, 0.88),
        Uv::Unit,
    );
}

/// A lot too small to build on becomes surface parking, striped and all.
pub(crate) fn add_parking(b: &mut Batches, lot: Rect, rng: &mut Rng) {
    let p = lot.inset(0.6);
    if p.w() < 4.0 || p.d() < 4.0 {
        return;
    }
    b.road.add_slab(
        p.x0,
        p.z0,
        p.x1,
        p.z1,
        KERB + 0.01,
        Color::from_hex(C_ASPHALT),
        Uv::Unit,
    );
    let along_x = p.w() >= p.d();
    let stall = 2.6;
    let n = ((if along_x { p.w() } else { p.d() }) / stall).floor() as usize;
    let paint = scale_color(Color::from_hex(C_PAINT), 0.75);
    for i in 1..n {
        let t = i as f32 * stall;
        if along_x {
            b.paint.add_slab(
                p.x0 + t - 0.07,
                p.z0,
                p.x0 + t + 0.07,
                p.z1,
                KERB + 0.02,
                paint,
                Uv::Unit,
            );
        } else {
            b.paint.add_slab(
                p.x0,
                p.z0 + t - 0.07,
                p.x1,
                p.z0 + t + 0.07,
                KERB + 0.02,
                paint,
                Uv::Unit,
            );
        }
    }
    let _ = rng;
}

/// The back garden of a house: fenced, lawned, and with something in it.
///
/// Everything behind the houses was bare pavement, which is the one part of a
/// suburb nobody builds and everybody notices — a street of houses with no
/// gardens behind them reads as a film set. All of this is cheap: a lawn, a
/// fence, and one of a handful of things people put on a lawn.
pub(crate) fn add_back_garden(b: &mut Batches, plot: Rect, rng: &mut Rng) {
    if plot.w() < 3.0 || plot.d() < 2.5 {
        return;
    }
    b.grass.add_slab(
        plot.x0,
        plot.z0,
        plot.x1,
        plot.z1,
        KERB + 0.012,
        scale_color(Color::from_hex(C_GRASS), rng.range(0.88, 1.14)),
        Uv::Unit,
    );
    // A patio against the house, which is where the door comes out.
    let patio = plot.d() * rng.range(0.18, 0.34);
    b.pads.add_slab(
        plot.x0 + 0.3,
        plot.z0,
        plot.x1 - 0.3,
        plot.z0 + patio,
        KERB + 0.02,
        scale_color(Color::from_hex(0x9c9890), rng.range(0.9, 1.1)),
        Uv::Unit,
    );

    // Close-boarded fence on three sides. Boards, not a panel: a solid box
    // reads as a wall, and every garden here would then be a compound.
    let fence = scale_color(Color::from_hex(0x7a5c3c), rng.range(0.85, 1.15));
    let fh = 1.7;
    for (a, c) in [
        ((plot.x0, plot.z0), (plot.x0, plot.z1)),
        ((plot.x1, plot.z0), (plot.x1, plot.z1)),
        ((plot.x0, plot.z1), (plot.x1, plot.z1)),
    ] {
        let len = ((c.0 - a.0).powi(2) + (c.1 - a.1).powi(2)).sqrt();
        let n = (len / 0.22).round().max(2.0) as usize;
        for k in 0..n {
            let t = k as f32 / n as f32;
            let (px, pz) = (a.0 + (c.0 - a.0) * t, a.1 + (c.1 - a.1) * t);
            let (dx, dz) = if (c.0 - a.0).abs() > (c.1 - a.1).abs() {
                (0.10, 0.035)
            } else {
                (0.035, 0.10)
            };
            b.trim.add_box(
                Vector3::new(px - dx, KERB, pz - dz),
                Vector3::new(px + dx, KERB + fh * rng.range(0.97, 1.0), pz + dz),
                scale_color(fence, rng.range(0.92, 1.08)),
                Uv::Unit,
            );
        }
    }

    let (cx, cz) = (plot.cx(), plot.z0 + patio + (plot.d() - patio) * 0.5);
    match rng.below(6) {
        // Shed in a corner.
        0 => {
            let sx = if rng.chance(0.5) { plot.x0 + 1.3 } else { plot.x1 - 1.3 };
            let wood = scale_color(Color::from_hex(0x6a5136), rng.range(0.85, 1.15));
            b.trim.add_box(
                Vector3::new(sx - 1.1, KERB, plot.z1 - 2.2),
                Vector3::new(sx + 1.1, KERB + 1.9, plot.z1 - 0.4),
                wood,
                Uv::Unit,
            );
            b.trim.add_gable(
                Vector3::new(sx - 1.2, KERB + 1.9, plot.z1 - 2.3),
                Vector3::new(sx + 1.2, KERB + 1.9, plot.z1 - 0.3),
                0.5,
                scale_color(wood, 0.78),
            );
        }
        // Washing line: two posts and a sag of white between them.
        1 => {
            let post = Color::from_hex(0x8f8a80);
            for s in [-1.0f32, 1.0] {
                b.trim.add_cylinder(
                    Vector3::new(cx + s * plot.w() * 0.30, KERB, cz),
                    0.05,
                    0.045,
                    1.9,
                    5,
                    post,
                    false,
                    Uv::Unit,
                );
            }
            for k in 0..4 {
                let t = (k as f32 + 0.5) / 4.0;
                let x = mix(cx - plot.w() * 0.28, cx + plot.w() * 0.28, t);
                let sag = 1.75 - (t - 0.5).abs().mul_add(-0.5, 0.28);
                b.trim.add_box(
                    Vector3::new(x - 0.22, KERB + sag - 0.55, cz - 0.02),
                    Vector3::new(x + 0.22, KERB + sag, cz + 0.02),
                    scale_color(Color::WHITE, rng.range(0.7, 1.0)),
                    Uv::Unit,
                );
            }
        }
        // Trampoline.
        2 => {
            b.trim.add_cylinder(
                Vector3::new(cx, KERB, cz),
                1.4,
                1.4,
                0.75,
                12,
                Color::from_hex(0x2f4a6b),
                false,
                Uv::Unit,
            );
            b.trim
                .add_ground_disc(cx, cz, 1.25, 12, 0.0, KERB + 0.76, Color::from_hex(0x2b2d31));
        }
        // A small above-ground pool.
        3 => {
            b.trim.add_cylinder(
                Vector3::new(cx, KERB, cz),
                1.6,
                1.6,
                0.85,
                14,
                Color::from_hex(0xb8bec2),
                false,
                Uv::Unit,
            );
            b.water
                .add_ground_disc(cx, cz, 1.45, 14, 0.0, KERB + 0.78, Color::from_hex(0x2f7fa8));
        }
        // Vegetable beds.
        4 => {
            for k in 0..3 {
                let z = cz - 1.2 + k as f32 * 1.2;
                if z > plot.z1 - 0.8 {
                    break;
                }
                b.trim.add_box(
                    Vector3::new(cx - 1.3, KERB, z - 0.4),
                    Vector3::new(cx + 1.3, KERB + 0.24, z + 0.4),
                    Color::from_hex(0x7a5a3a),
                    Uv::Unit,
                );
                b.foliage.add_box(
                    Vector3::new(cx - 1.15, KERB + 0.22, z - 0.3),
                    Vector3::new(cx + 1.15, KERB + rng.range(0.42, 0.7), z + 0.3),
                    scale_color(Color::from_hex(0x53702f), rng.range(0.85, 1.15)),
                    Uv::Unit,
                );
            }
        }
        // Just a tree and a table.
        _ => {
            add_tree(b, cx, cz, KERB, true, false, rng);
            b.trim.add_cylinder(
                Vector3::new(plot.x0 + 1.4, KERB, plot.z0 + patio * 0.5),
                0.05,
                0.05,
                0.70,
                5,
                Color::from_hex(0x54585d),
                false,
                Uv::Unit,
            );
            b.trim.add_cylinder(
                Vector3::new(plot.x0 + 1.4, KERB + 0.70, plot.z0 + patio * 0.5),
                0.42,
                0.42,
                0.05,
                10,
                Color::from_hex(0xb9b3a6),
                true,
                Uv::Unit,
            );
        }
    }
}

/// A helipad on a roof: the circle, the H, edge lights and a windsock.
///
/// It is the one rooftop marking that is legible from the air, which is the
/// angle this city is usually looked at from, and it gives the helicopters
/// somewhere to be going.
pub(crate) fn add_helipad(b: &mut Batches, cx: f32, cz: f32, r: f32, y: f32, rng: &mut Rng) {
    let deck = scale_color(Color::from_hex(0x3f4a44), rng.range(0.92, 1.08));
    let paint = Color::from_hex(0xe8e4d8);
    b.trim.add_ground_disc(cx, cz, r, 18, 0.0, y + 0.06, deck);
    // The touchdown circle, then the H inside it.
    const N: usize = 20;
    for i in 0..N {
        let (a0, a1) = (
            i as f32 / N as f32 * TAU,
            (i + 1) as f32 / N as f32 * TAU,
        );
        let (i0, i1) = (r * 0.66, r * 0.74);
        b.paint.add_ground_quad(
            [
                (cx + i0 * a0.cos(), cz + i0 * a0.sin()),
                (cx + i1 * a0.cos(), cz + i1 * a0.sin()),
                (cx + i1 * a1.cos(), cz + i1 * a1.sin()),
                (cx + i0 * a1.cos(), cz + i0 * a1.sin()),
            ],
            y + 0.08,
            paint,
            Uv::Unit,
        );
    }
    let h = r * 0.34;
    for (x0, z0, x1, z1) in [
        (-h * 0.62, -h, -h * 0.28, h),
        (h * 0.28, -h, h * 0.62, h),
        (-h * 0.28, -h * 0.20, h * 0.28, h * 0.20),
    ] {
        b.paint.add_slab(cx + x0, cz + z0, cx + x1, cz + z1, y + 0.08, paint, Uv::Unit);
    }
    // Edge lights: green, and only after dark.
    for i in 0..12 {
        let a = i as f32 / 12.0 * TAU;
        b.glow.add_ellipsoid(
            Vector3::new(cx + r * 0.94 * a.cos(), y + 0.16, cz + r * 0.94 * a.sin()),
            0.18,
            0.14,
            0.18,
            3,
            5,
            Color::from_hex(0x2aff7a),
        );
    }
    // A windsock on a mast, which is the other thing every pad has.
    let (wx, wz) = (cx + r * 1.15, cz + r * 0.5);
    b.trim.add_cylinder(
        Vector3::new(wx, y, wz),
        0.07,
        0.05,
        3.0,
        5,
        Color::from_hex(0x9a958c),
        false,
        Uv::Unit,
    );
    let bearing = rng.range(0.0, TAU);
    for k in 0..3 {
        let t0 = k as f32 * 0.55;
        let t1 = t0 + 0.55;
        let c = if k % 2 == 0 {
            Color::from_hex(0xe85a2a)
        } else {
            Color::from_hex(0xf0efe8)
        };
        b.trim.add_limb(
            Vector3::new(wx + bearing.cos() * t0, y + 2.9 - t0 * 0.12, wz + bearing.sin() * t0),
            Vector3::new(wx + bearing.cos() * t1, y + 2.9 - t1 * 0.12, wz + bearing.sin() * t1),
            0.26 - k as f32 * 0.05,
            0.21 - k as f32 * 0.05,
            7,
            c,
            false,
        );
    }
}

/// A plot in the middle of being built on.
///
/// Every city here is finished and perfect, which is most of why it reads as a
/// model rather than a place: nothing is in progress, nothing is worn, nothing
/// is half-done. One site per few blocks is the cheapest possible injection of
/// "something is happening here", and a tower crane is visible from anywhere
/// in the city — it is the one piece of a skyline that says the skyline is
/// still changing.
pub(crate) fn add_construction(b: &mut Batches, lot: Rect, floors: f32, rng: &mut Rng) {
    let plot = lot.inset(0.8);
    if plot.w() < 8.0 || plot.d() < 8.0 {
        return;
    }
    b.occ.claim([plot.x0, plot.z0, plot.x1, plot.z1]);

    // Churned ground inside the hoarding.
    b.pads.add_slab(
        plot.x0,
        plot.z0,
        plot.x1,
        plot.z1,
        KERB + 0.02,
        scale_color(Color::from_hex(0x6e6353), rng.range(0.9, 1.1)),
        Uv::Unit,
    );

    // --- Hoarding: painted panels on posts, with a gate.
    let panel = [0x2f6bb5u32, 0x3f8f57, 0xc4442b, 0xd8a02a];
    let gate_side = rng.below(4);
    let sides = [
        ((plot.x0, plot.z0), (plot.x1, plot.z0)),
        ((plot.x1, plot.z0), (plot.x1, plot.z1)),
        ((plot.x1, plot.z1), (plot.x0, plot.z1)),
        ((plot.x0, plot.z1), (plot.x0, plot.z0)),
    ];
    for (i, (a, c)) in sides.iter().enumerate() {
        let len = ((c.0 - a.0).powi(2) + (c.1 - a.1).powi(2)).sqrt();
        let n = ((len / 2.4).round() as usize).max(1);
        for k in 0..n {
            let (t0, t1) = (k as f32 / n as f32, (k + 1) as f32 / n as f32);
            // The gate is a two-panel gap in the middle of one side.
            if i == gate_side && (t0 - 0.5).abs() < 0.18 {
                continue;
            }
            let (ax, az) = (a.0 + (c.0 - a.0) * t0, a.1 + (c.1 - a.1) * t0);
            let (bx, bz) = (a.0 + (c.0 - a.0) * t1, a.1 + (c.1 - a.1) * t1);
            let seg = ((bx - ax).powi(2) + (bz - az).powi(2)).sqrt();
            b.trim.add_yaw_box(
                Vector3::new((ax + bx) * 0.5, KERB + 1.15, (az + bz) * 0.5),
                Vector3::new(0.05, 1.15, seg * 0.48),
                facing(bx - ax, bz - az),
                scale_color(
                    Color::from_hex(panel[(k + i) % panel.len()]),
                    rng.range(0.85, 1.15),
                ),
                Uv::Unit,
            );
        }
    }

    // --- The frame going up: columns and slabs, fewer floors as it rises, and
    // the topmost one only half poured.
    let done = ((floors * rng.range(0.25, 0.75)).round()).clamp(1.0, 14.0);
    let core = plot.inset(2.2);
    let concrete = scale_color(Color::from_hex(0x9a968c), rng.range(0.92, 1.08));
    let cols = ((core.w() / 6.0).round() as usize).max(2);
    let rows = ((core.d() / 6.0).round() as usize).max(2);
    for f in 0..done as usize {
        let y = KERB + f as f32 * FLOOR;
        // A slab, shrinking to a partial pour on the top floor.
        let frac = if f + 1 == done as usize { rng.range(0.35, 0.85) } else { 1.0 };
        b.trim.add_box(
            Vector3::new(core.x0, y, core.z0),
            Vector3::new(mix(core.x0, core.x1, frac), y + 0.28, core.z1),
            concrete,
            Uv::Unit,
        );
        for i in 0..=cols {
            for j in 0..=rows {
                let x = mix(core.x0, core.x1, i as f32 / cols as f32);
                let z = mix(core.z0, core.z1, j as f32 / rows as f32);
                if x > mix(core.x0, core.x1, frac) {
                    continue;
                }
                b.trim.add_box(
                    Vector3::new(x - 0.24, y + 0.28, z - 0.24),
                    Vector3::new(x + 0.24, y + FLOOR, z + 0.24),
                    concrete,
                    Uv::Unit,
                );
            }
        }
    }
    // Rebar sticking out of the top columns, which is what an unfinished
    // frame looks like from the street.
    let top = KERB + done * FLOOR;
    for i in 0..=cols {
        for j in 0..=rows {
            if !rng.chance(0.55) {
                continue;
            }
            let x = mix(core.x0, core.x1, i as f32 / cols as f32);
            let z = mix(core.z0, core.z1, j as f32 / rows as f32);
            for _ in 0..3 {
                b.trim.add_limb(
                    Vector3::new(x + rng.range(-0.18, 0.18), top, z + rng.range(-0.18, 0.18)),
                    Vector3::new(x + rng.range(-0.3, 0.3), top + rng.range(0.5, 1.1), z + rng.range(-0.3, 0.3)),
                    0.022,
                    0.018,
                    3,
                    Color::from_hex(0x8a6a4a),
                    false,
                );
            }
        }
    }

    // --- Tower crane. Mast, slewing unit, jib, counter-jib and the hook.
    let yellow = scale_color(Color::from_hex(0xd8a02a), rng.range(0.9, 1.1));
    let (mx, mz) = (
        mix(plot.x0 + 2.0, plot.x1 - 2.0, rng.f()),
        mix(plot.z0 + 2.0, plot.z1 - 2.0, rng.f()),
    );
    // Tall enough to clear what is around it: a crane hidden among the towers
    // it is building is a crane nobody sees.
    let mast = top.max(KERB + 18.0) + rng.range(14.0, 30.0);
    // A lattice mast: four legs with cross-bracing, which reads at any range.
    for (sx, sz) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        b.trim.add_box(
            Vector3::new(mx + sx * 0.62 - 0.09, KERB, mz + sz * 0.62 - 0.09),
            Vector3::new(mx + sx * 0.62 + 0.09, mast, mz + sz * 0.62 + 0.09),
            yellow,
            Uv::Unit,
        );
    }
    let bays = ((mast - KERB) / 2.4) as usize;
    for k in 0..bays {
        let y = KERB + k as f32 * 2.4;
        for (a, c) in [
            ((-0.62f32, -0.62f32), (0.62f32, -0.62f32)),
            ((0.62, -0.62), (0.62, 0.62)),
            ((0.62, 0.62), (-0.62, 0.62)),
            ((-0.62, 0.62), (-0.62, -0.62)),
        ] {
            let flip = k % 2 == 0;
            let (p, q) = if flip { (a, c) } else { (c, a) };
            b.trim.add_limb(
                Vector3::new(mx + p.0, y, mz + p.1),
                Vector3::new(mx + q.0, y + 2.4, mz + q.1),
                0.045,
                0.045,
                3,
                yellow,
                false,
            );
        }
    }
    // Slewing unit, cab, jib and counter-jib, turned to a random bearing.
    let bearing = rng.range(0.0, TAU);
    b.trim.add_box(
        Vector3::new(mx - 0.85, mast, mz - 0.85),
        Vector3::new(mx + 0.85, mast + 1.6, mz + 0.85),
        scale_color(yellow, 0.85),
        Uv::Unit,
    );
    let jib = rng.range(18.0, 32.0);
    let counter = jib * 0.34;
    let jy = mast + 1.9;
    for (len, dir, depth) in [(jib, 1.0f32, 1.3f32), (counter, -1.0, 1.0)] {
        b.trim.add_yaw_box(
            Vector3::new(
                mx + bearing.cos() * dir * len * 0.5,
                jy + depth * 0.5,
                mz + bearing.sin() * dir * len * 0.5,
            ),
            Vector3::new(0.22, depth * 0.5, len * 0.5),
            facing(bearing.cos() * dir, bearing.sin() * dir),
            yellow,
            Uv::Unit,
        );
    }
    // Counterweight, and the hoist block hanging on its rope.
    b.trim.add_box(
        Vector3::new(mx - bearing.cos() * counter - 0.9, jy - 0.4, mz - bearing.sin() * counter - 0.9),
        Vector3::new(mx - bearing.cos() * counter + 0.9, jy + 1.5, mz - bearing.sin() * counter + 0.9),
        Color::from_hex(0x5a5f66),
        Uv::Unit,
    );
    let hook_t = rng.range(0.4, 0.9);
    let (hx, hz) = (
        mx + bearing.cos() * jib * hook_t,
        mz + bearing.sin() * jib * hook_t,
    );
    let hook_y = KERB + rng.range(2.0, (top - KERB).max(4.0));
    b.trim.add_limb(
        Vector3::new(hx, jy, hz),
        Vector3::new(hx, hook_y, hz),
        0.03,
        0.03,
        3,
        Color::from_hex(0x3a3d42),
        false,
    );
    b.trim.add_box(
        Vector3::new(hx - 0.28, hook_y - 0.5, hz - 0.28),
        Vector3::new(hx + 0.28, hook_y, hz + 0.28),
        Color::from_hex(0x4a4f52),
        Uv::Unit,
    );
    // An aircraft light on the mast, which blinks with the tower beacons.
    b.beacon.add_ellipsoid(
        Vector3::new(mx, mast + 1.75, mz),
        0.28,
        0.28,
        0.28,
        3,
        6,
        Color::from_hex(0xff2a1a),
    );

    // --- The yard: a portacabin, a skip and heaps of aggregate.
    let cabin = scale_color(Color::from_hex(0xd8d3c4), rng.range(0.9, 1.1));
    let (cx, cz) = (plot.x0 + 2.6, plot.z1 - 2.0);
    b.trim.add_box(
        Vector3::new(cx - 2.4, KERB, cz - 1.3),
        Vector3::new(cx + 2.4, KERB + 2.6, cz + 1.3),
        cabin,
        Uv::Unit,
    );
    b.trim.add_slab(
        cx - 2.5,
        cz - 1.4,
        cx + 2.5,
        cz + 1.4,
        KERB + 2.62,
        scale_color(cabin, 0.7),
        Uv::Unit,
    );
    for _ in 0..(1 + rng.below(3)) {
        let (sx, sz) = (
            rng.range(plot.x0 + 2.0, plot.x1 - 2.0),
            rng.range(plot.z0 + 2.0, plot.z1 - 2.0),
        );
        b.trim.add_box(
            Vector3::new(sx - 1.5, KERB, sz - 0.9),
            Vector3::new(sx + 1.5, KERB + 1.1, sz + 0.9),
            scale_color(Color::from_hex(0xb0542a), rng.range(0.85, 1.15)),
            Uv::Unit,
        );
    }
    for _ in 0..(2 + rng.below(4)) {
        let (px, pz) = (
            rng.range(plot.x0 + 1.5, plot.x1 - 1.5),
            rng.range(plot.z0 + 1.5, plot.z1 - 1.5),
        );
        // A heap, not a box: a cone of aggregate.
        let r = rng.range(0.9, 2.0);
        b.trim.add_cylinder(
            Vector3::new(px, KERB, pz),
            r,
            0.05,
            r * rng.range(0.5, 0.8),
            9,
            scale_color(Color::from_hex(0x8a8074), rng.range(0.85, 1.15)),
            true,
            Uv::Unit,
        );
    }
}
