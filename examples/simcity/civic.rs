//! Part of the `simcity` example; see `mod.rs`.
//!
//! The buildings a city has one of.
//!
//! Everything else here is generated from a rule — a height falloff, a parcel
//! subdivision, a facade idiom — and the result is a city entirely made of
//! *housing and offices*. A real one has a handful of buildings that are
//! nothing like the blocks around them and that everybody can point to: the
//! hospital, the school, the fire station, the police station. They are worth
//! having precisely because they are exceptions, so each is placed once and
//! built by hand rather than by the rule.
#![allow(dead_code)]

use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Civic {
    Hospital,
    School,
    FireStation,
    PoliceStation,
}

impl Civic {
    /// The smallest block worth putting one on.
    ///
    /// These started at 34 and 32 metres, which is larger than a Manhattan
    /// block and more than twice an old-town one — so four layouts out of
    /// seven placed *none of these buildings at all*, and the only reason
    /// anybody knew is that the generator counts them. A threshold that reads
    /// as reasonable is worth nothing next to the sizes actually on offer.
    pub(crate) fn wants(self) -> f32 {
        match self {
            Civic::Hospital => 24.0,
            Civic::School => 19.0,
            Civic::FireStation => 13.0,
            Civic::PoliceStation => 13.0,
        }
    }
}

/// A perimeter railing, low enough to see over.
fn railing(b: &mut Batches, r: &Rect, gate: usize, col: Color) {
    let sides = [
        ((r.x0, r.z0), (r.x1, r.z0)),
        ((r.x1, r.z0), (r.x1, r.z1)),
        ((r.x1, r.z1), (r.x0, r.z1)),
        ((r.x0, r.z1), (r.x0, r.z0)),
    ];
    for (i, (a, c)) in sides.iter().enumerate() {
        let len = ((c.0 - a.0).powi(2) + (c.1 - a.1).powi(2)).sqrt();
        let n = (len / 0.5).round().max(2.0) as usize;
        for k in 0..=n {
            let t = k as f32 / n as f32;
            if i == gate && (t - 0.5).abs() < 0.14 {
                continue;
            }
            let (px, pz) = (a.0 + (c.0 - a.0) * t, a.1 + (c.1 - a.1) * t);
            b.trim.add_cylinder(
                Vector3::new(px, KERB, pz),
                0.028,
                0.024,
                1.2,
                4,
                col,
                false,
                Uv::Unit,
            );
        }
    }
}

/// Build one. Returns false if the block is too small to take it.
pub(crate) fn add_civic(b: &mut Batches, kind: Civic, lot: Rect, rng: &mut Rng) -> bool {
    if lot.w() < kind.wants() || lot.d() < kind.wants() {
        return false;
    }
    b.occ.claim([lot.x0, lot.z0, lot.x1, lot.z1]);
    match kind {
        Civic::Hospital => hospital(b, lot, rng),
        Civic::School => school(b, lot, rng),
        Civic::FireStation => fire_station(b, lot, rng),
        Civic::PoliceStation => police_station(b, lot, rng),
    }
    true
}

fn hospital(b: &mut Batches, lot: Rect, rng: &mut Rng) {
    let plot = lot.inset(2.0);
    b.pads.add_box(
        Vector3::new(lot.x0, 0.0, lot.z0),
        Vector3::new(lot.x1, KERB, lot.z1),
        scale_color(Color::from_hex(C_SIDEWALK), rng.range(0.95, 1.05)),
        Uv::Unit,
    );
    // A broad slab block: wide, not tall, which is what a hospital is.
    let ward = Rect {
        x0: plot.x0 + 1.0,
        z0: plot.z0 + plot.d() * 0.30,
        x1: plot.x1 - 1.0,
        z1: plot.z1 - 1.0,
    };
    let floors = 5.0f32;
    let h = KERB + floors * FLOOR;
    let wall = Color::from_hex(0xdfe3e6);
    b.facades[1].add_box(
        Vector3::new(ward.x0, KERB, ward.z0),
        Vector3::new(ward.x1, h, ward.z1),
        wall,
        Uv::World { u: 1.0 / BAY, v: 1.0 / FLOOR },
    );
    b.trim.add_slab(
        ward.x0 - 0.3,
        ward.z0 - 0.3,
        ward.x1 + 0.3,
        ward.z1 + 0.3,
        h,
        Color::from_hex(C_ROOF),
        Uv::Unit,
    );
    // The pad on the roof, which is why a hospital reads from the air.
    add_helipad(
        b,
        ward.cx(),
        ward.cz(),
        (ward.w().min(ward.d()) * 0.28).clamp(4.0, 8.0),
        h + 0.08,
        rng,
    );
    // Ambulance bay: a canopy on columns over a marked apron.
    let bay = Rect {
        x0: plot.x0 + 2.0,
        z0: plot.z0,
        x1: plot.x0 + plot.w() * 0.55,
        z1: ward.z0 - 0.5,
    };
    b.road.add_slab(
        bay.x0,
        bay.z0,
        bay.x1,
        bay.z1,
        KERB + 0.01,
        Color::from_hex(0x53585c),
        Uv::Unit,
    );
    b.paint.add_ground_path(
        &[(bay.x0 + 1.0, bay.cz()), (bay.x1 - 1.0, bay.cz())],
        0.16,
        KERB + 0.02,
        Color::from_hex(0xd8d2c0),
        Uv::Unit,
    );
    b.trim.add_slab(
        bay.x0 - 0.4,
        bay.z0,
        bay.x1 + 0.4,
        bay.z1 + 0.4,
        KERB + 4.2,
        Color::from_hex(0x9a958c),
        Uv::Unit,
    );
    for (px, pz) in [
        (bay.x0, bay.z0 + 0.6),
        (bay.x1, bay.z0 + 0.6),
        (bay.x0, bay.z1),
        (bay.x1, bay.z1),
    ] {
        b.trim.add_cylinder(
            Vector3::new(px, KERB, pz),
            0.16,
            0.14,
            4.2,
            6,
            Color::from_hex(0xb9b3a6),
            false,
            Uv::Unit,
        );
    }
    // Red cross on the wall, lit at night, and the entrance under it.
    let cross = Color::from_hex(0xd8241c);
    for (cw, ch) in [(2.2f32, 0.7f32), (0.7, 2.2)] {
        // On `glow`, not `neon`. The pairing below was meant to give the cross
        // a lit daylight body, but a `neon` batch is drawn at every hour and
        // only scaled by `sky.lights` — and this box sat 0.02 PROUD of the
        // trim copy, so by day it was a black cross covering the red one. On
        // `glow` it is hidden outright while the sun is up.
        b.glow.add_box(
            Vector3::new(ward.cx() - cw * 0.5, KERB + FLOOR * 1.4 - ch * 0.5, ward.z0 - 0.14),
            Vector3::new(ward.cx() + cw * 0.5, KERB + FLOOR * 1.4 + ch * 0.5, ward.z0 - 0.06),
            cross,
            Uv::Unit,
        );
        b.trim.add_box(
            Vector3::new(ward.cx() - cw * 0.5, KERB + FLOOR * 1.4 - ch * 0.5, ward.z0 - 0.12),
            Vector3::new(ward.cx() + cw * 0.5, KERB + FLOOR * 1.4 + ch * 0.5, ward.z0 - 0.05),
            cross,
            Uv::Unit,
        );
    }
    b.glow.add_box(
        Vector3::new(ward.cx() - 2.2, KERB + 0.6, ward.z0 - 0.10),
        Vector3::new(ward.cx() + 2.2, KERB + 2.8, ward.z0 - 0.02),
        Color::from_hex(0xffeec8),
        Uv::Unit,
    );
    // Car park on the remaining ground.
    add_parking(
        b,
        Rect {
            x0: plot.x0 + plot.w() * 0.58,
            z0: plot.z0,
            x1: plot.x1,
            z1: ward.z0 - 0.5,
        },
        rng,
    );
}

fn school(b: &mut Batches, lot: Rect, rng: &mut Rng) {
    let plot = lot.inset(1.5);
    b.grass.add_box(
        Vector3::new(lot.x0, 0.0, lot.z0),
        Vector3::new(lot.x1, KERB, lot.z1),
        scale_color(Color::from_hex(C_GRASS), rng.range(0.92, 1.08)),
        Uv::Unit,
    );
    // Two low wings meeting at a corner: a school is long and two storeys, not
    // a block, and the shape is what tells it from an office.
    let wall = Color::from_hex(0xc8a882);
    let uv = Uv::World { u: 1.0 / BAY, v: 1.0 / FLOOR };
    let h = KERB + 2.0 * FLOOR;
    // The wings are a fraction of the plot, not a fixed twelve metres: on a
    // small block a fixed depth swallows the playground and the two wings
    // merge into one box.
    let wing = (plot.d() * 0.30).clamp(8.0, 13.0);
    let a = Rect {
        x0: plot.x0 + 1.0,
        z0: plot.z1 - wing,
        x1: plot.x1 - plot.w() * 0.34,
        z1: plot.z1 - 1.0,
    };
    let c = Rect {
        x0: plot.x1 - wing,
        z0: plot.z1 - plot.d() * 0.62,
        x1: plot.x1 - 1.0,
        z1: plot.z1 - 1.0,
    };
    for r in [a, c] {
        b.facades[1].add_box(
            Vector3::new(r.x0, KERB, r.z0),
            Vector3::new(r.x1, h, r.z1),
            wall,
            uv,
        );
        b.trim.add_slab(
            r.x0 - 0.35,
            r.z0 - 0.35,
            r.x1 + 0.35,
            r.z1 + 0.35,
            h,
            Color::from_hex(C_ROOF),
            Uv::Unit,
        );
    }
    // Playground: tarmac with markings, and a pitch beyond it.
    let yard = Rect {
        x0: plot.x0 + 1.0,
        z0: plot.z0 + 1.0,
        x1: plot.x1 - wing - 1.5,
        z1: a.z0 - 1.0,
    };
    if yard.w() > 8.0 && yard.d() > 8.0 {
        b.road.add_slab(
            yard.x0,
            yard.z0,
            yard.x1,
            yard.z1,
            KERB + 0.01,
            Color::from_hex(0x4e5358),
            Uv::Unit,
        );
        let paint = Color::from_hex(0xe8e2d0);
        // A court, a hopscotch grid and a painted circle: the markings are
        // what makes tarmac a playground.
        b.paint.add_ground_path(
            &[
                (yard.x0 + 1.5, yard.z0 + 1.5),
                (yard.x1 - 1.5, yard.z0 + 1.5),
                (yard.x1 - 1.5, yard.cz()),
                (yard.x0 + 1.5, yard.cz()),
                (yard.x0 + 1.5, yard.z0 + 1.5),
                (yard.x1 - 1.5, yard.z0 + 1.5),
            ],
            0.12,
            KERB + 0.02,
            paint,
            Uv::Unit,
        );
        for k in 0..6 {
            let z = yard.cz() + 1.2 + k as f32 * 1.1;
            if z > yard.z1 - 1.0 {
                break;
            }
            b.paint.add_slab(
                yard.x0 + 2.0,
                z,
                yard.x0 + 3.0,
                z + 0.9,
                KERB + 0.02,
                paint,
                Uv::Unit,
            );
        }
        // A bike shed against the wall.
        b.trim.add_box(
            Vector3::new(yard.x1 - 4.0, KERB, yard.z1 - 2.2),
            Vector3::new(yard.x1 - 0.6, KERB + 2.3, yard.z1 - 0.6),
            Color::from_hex(0x6b7076),
            Uv::Unit,
        );
    }
    railing(b, &plot, rng.below(4), Color::from_hex(0x2f4a5c));
    for _ in 0..4 {
        add_tree(
            b,
            rng.range(plot.x0 + 2.0, plot.x1 - 2.0),
            rng.range(plot.z0 + 2.0, plot.z1 - 2.0),
            KERB,
            true,
            false,
            rng,
        );
    }
}

/// A shared shape for the two emergency stations: a hall with vehicle doors
/// and an apron in front of it. What differs is the livery and the yard.
fn station(b: &mut Batches, lot: Rect, bays: usize, livery: Color, rng: &mut Rng) -> Rect {
    let plot = lot.inset(1.2);
    b.pads.add_box(
        Vector3::new(lot.x0, 0.0, lot.z0),
        Vector3::new(lot.x1, KERB, lot.z1),
        scale_color(Color::from_hex(C_SIDEWALK), rng.range(0.95, 1.05)),
        Uv::Unit,
    );
    let hall = Rect {
        x0: plot.x0 + 0.5,
        z0: plot.z0 + plot.d() * 0.40,
        x1: plot.x1 - 0.5,
        z1: plot.z1 - 0.5,
    };
    let h = KERB + 6.4;
    b.facades[1].add_box(
        Vector3::new(hall.x0, KERB, hall.z0),
        Vector3::new(hall.x1, h, hall.z1),
        Color::from_hex(0xc9c4b8),
        Uv::World { u: 1.0 / BAY, v: 1.0 / FLOOR },
    );
    b.trim.add_slab(
        hall.x0 - 0.35,
        hall.z0 - 0.35,
        hall.x1 + 0.35,
        hall.z1 + 0.35,
        h,
        Color::from_hex(C_ROOF),
        Uv::Unit,
    );
    // A band of livery under the parapet — every station has one.
    b.trim.add_box(
        Vector3::new(hall.x0 - 0.06, h - 1.1, hall.z0 - 0.10),
        Vector3::new(hall.x1 + 0.06, h - 0.3, hall.z0 - 0.02),
        livery,
        Uv::Unit,
    );
    // Vehicle doors: tall roller shutters facing the apron.
    let door_w = (hall.w() / bays as f32) * 0.72;
    for i in 0..bays {
        let cx = mix(hall.x0, hall.x1, (i as f32 + 0.5) / bays as f32);
        b.trim.add_box(
            Vector3::new(cx - door_w * 0.5, KERB, hall.z0 - 0.12),
            Vector3::new(cx + door_w * 0.5, KERB + 4.4, hall.z0 - 0.04),
            Color::from_hex(0xb0aa9c),
            Uv::Unit,
        );
        b.glow.add_box(
            Vector3::new(cx - door_w * 0.46, KERB + 0.3, hall.z0 - 0.16),
            Vector3::new(cx + door_w * 0.46, KERB + 4.1, hall.z0 - 0.13),
            Color::from_hex(0xffe6bc),
            Uv::Unit,
        );
    }
    // Apron in front, marked with a stop line.
    let apron = Rect {
        x0: plot.x0,
        z0: plot.z0,
        x1: plot.x1,
        z1: hall.z0 - 0.5,
    };
    b.road.add_slab(
        apron.x0,
        apron.z0,
        apron.x1,
        apron.z1,
        KERB + 0.01,
        Color::from_hex(0x4a4f54),
        Uv::Unit,
    );
    b.paint.add_ground_path(
        &[(apron.x0 + 1.0, apron.z0 + 1.2), (apron.x1 - 1.0, apron.z0 + 1.2)],
        0.22,
        KERB + 0.02,
        Color::from_hex(0xd8c23a),
        Uv::Unit,
    );
    apron
}

fn fire_station(b: &mut Batches, lot: Rect, rng: &mut Rng) {
    let red = Color::from_hex(0xc0231c);
    let apron = station(b, lot, 3, red, rng);
    // A drill tower: the one piece of a fire station visible over the rooftops.
    let (tx, tz) = (lot.x1 - 3.2, lot.z1 - 3.2);
    b.trim.add_box(
        Vector3::new(tx - 2.0, KERB, tz - 2.0),
        Vector3::new(tx + 2.0, KERB + 14.0, tz + 2.0),
        Color::from_hex(0xb5b0a4),
        Uv::Unit,
    );
    for f in 1..4 {
        b.glow.add_box(
            Vector3::new(tx - 1.1, KERB + f as f32 * 3.4, tz - 2.06),
            Vector3::new(tx + 1.1, KERB + f as f32 * 3.4 + 1.5, tz - 2.02),
            Color::from_hex(0xffe6bc),
            Uv::Unit,
        );
    }
    b.trim.add_box(
        Vector3::new(tx - 2.2, KERB + 14.0, tz - 2.2),
        Vector3::new(tx + 2.2, KERB + 14.5, tz + 2.2),
        Color::from_hex(0x8a857c),
        Uv::Unit,
    );
    // Firefighters on the apron.
    for k in 0..3 {
        b.posts.push((
            mix(apron.x0 + 2.0, apron.x1 - 2.0, (k as f32 + 0.5) / 3.0),
            apron.z1 - 1.6,
            1,
        ));
    }
}

fn police_station(b: &mut Batches, lot: Rect, rng: &mut Rng) {
    let blue = Color::from_hex(0x1d4e8a);
    let apron = station(b, lot, 2, blue, rng);
    // The blue lamp over the door, which is the whole signature.
    let (lx, lz) = (lot.cx(), lot.z0 + lot.d() * 0.40 - 0.6);
    b.trim.add_cylinder(
        Vector3::new(lx, KERB + 4.6, lz),
        0.10,
        0.10,
        0.5,
        6,
        Color::from_hex(0x2b2d31),
        true,
        Uv::Unit,
    );
    // Same fault as the hospital cross: this shell ENCLOSES the lit copy just
    // below, so on `neon` it blacked the lamp out for every daylight hour.
    b.glow.add_ellipsoid(
        Vector3::new(lx, KERB + 4.35, lz),
        0.42,
        0.52,
        0.42,
        4,
        7,
        Color::from_hex(0x2a6cff),
    );
    b.trim.add_ellipsoid(
        Vector3::new(lx, KERB + 4.35, lz),
        0.40,
        0.50,
        0.40,
        4,
        7,
        Color::from_hex(0x2a6cff),
    );
    for k in 0..2 {
        b.posts.push((
            mix(apron.x0 + 3.0, apron.x1 - 3.0, (k as f32 + 0.5) / 2.0),
            apron.z1 - 1.8,
            0,
        ));
    }
}
