//! Part of the `simcity` example; see `mod.rs`.
//!
//! The zoo.
//!
//! What makes a zoo read from the air is not the animals — at any sane camera
//! distance they are three or four pixels. It is the **plan**: a compound with
//! one way in, a path that loops rather than goes anywhere, and inside it a
//! patchwork of enclosures that are all different shapes and all fenced. No
//! other land use looks like that. A park has paths that cross; a farm has
//! rectangles in rows; a zoo has irregular pens strung along a circuit.
//!
//! So the enclosures come first and the animals are decoration on top — but
//! the animals are still worth having, because the one thing that gives a zoo
//! away at close range is that the things standing in the fields are the wrong
//! shape for a field.
#![allow(dead_code)]

use super::*;

/// What lives in an enclosure, which decides its ground, its barrier and the
/// silhouette standing in it.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Pen {
    /// Elephants: dust, a pool, a big shelter.
    Elephant,
    /// Giraffe and zebra on dry savannah, with a tall feeding platform.
    Savannah,
    /// Big cats behind a deep moat and a glass wall, with rocks to lie on.
    BigCat,
    /// Penguins: a shallow pool with a tiled surround.
    Penguin,
    /// Flamingos on a shallow lagoon.
    Flamingo,
    /// A walk-through aviary under a mesh dome.
    Aviary,
    /// Primates on an island surrounded by water, with climbing frames.
    Primate,
    /// Bears on rock and grass.
    Bear,
}

const PENS: [Pen; 8] = [
    Pen::Elephant,
    Pen::Savannah,
    Pen::BigCat,
    Pen::Penguin,
    Pen::Flamingo,
    Pen::Aviary,
    Pen::Primate,
    Pen::Bear,
];

/// A four-legged animal, built from a body, a neck, a head and four legs.
///
/// Deliberately one function with parameters rather than eight models: what
/// separates a giraffe from a zebra in silhouette is the neck length and the
/// leg length, and what separates a big cat from a bear is how low the body
/// sits. Those are numbers, not meshes.
#[allow(clippy::too_many_arguments)]
fn quadruped(
    b: &mut Batches,
    x: f32,
    y: f32,
    z: f32,
    yaw: f32,
    body_len: f32,
    body_r: f32,
    leg: f32,
    neck: f32,
    neck_lean: f32,
    hide: Color,
    rng: &mut Rng,
) {
    let (sa, ca) = yaw.sin_cos();
    let fwd = |d: f32, s: f32| (x + ca * d - sa * s, z + sa * d + ca * s);
    let hip = y + leg + body_r;
    // Barrel.
    let (ax, az) = fwd(-body_len * 0.5, 0.0);
    let (bx, bz) = fwd(body_len * 0.5, 0.0);
    b.foliage.add_limb(
        Vector3::new(ax, hip, az),
        Vector3::new(bx, hip, bz),
        body_r,
        body_r * 0.92,
        7,
        hide,
        true,
    );
    // Legs, splayed a little so it stands rather than balances.
    for (d, s) in [
        (-body_len * 0.34, body_r * 0.62),
        (-body_len * 0.34, -body_r * 0.62),
        (body_len * 0.34, body_r * 0.62),
        (body_len * 0.34, -body_r * 0.62),
    ] {
        let (lx, lz) = fwd(d, s);
        b.foliage.add_limb(
            Vector3::new(lx, y, lz),
            Vector3::new(lx, hip, lz),
            body_r * 0.20,
            body_r * 0.26,
            5,
            scale_color(hide, 0.9),
            false,
        );
    }
    // Neck and head.
    let (nx, nz) = fwd(body_len * 0.46, 0.0);
    let (hx, hz) = fwd(body_len * 0.46 + neck * neck_lean, 0.0);
    let head_y = hip + neck * (1.0 - neck_lean * 0.5);
    b.foliage.add_limb(
        Vector3::new(nx, hip + body_r * 0.4, nz),
        Vector3::new(hx, head_y, hz),
        body_r * 0.42,
        body_r * 0.30,
        6,
        hide,
        false,
    );
    b.foliage.add_ellipsoid(
        Vector3::new(hx, head_y, hz),
        body_r * 0.40,
        body_r * 0.34,
        body_r * 0.34,
        3,
        7,
        scale_color(hide, 1.05),
    );
    let _ = rng;
}

/// A standing bird: penguin, flamingo, crane. Legs, a body and a head.
fn wader(b: &mut Batches, x: f32, y: f32, z: f32, leg: f32, body: f32, c: Color, head: Color) {
    if leg > 0.05 {
        for s in [-1.0f32, 1.0] {
            b.foliage.add_limb(
                Vector3::new(x + s * body * 0.22, y, z),
                Vector3::new(x + s * body * 0.10, y + leg, z),
                body * 0.10,
                body * 0.10,
                4,
                head,
                false,
            );
        }
    }
    b.foliage.add_ellipsoid(
        Vector3::new(x, y + leg + body * 0.55, z),
        body * 0.42,
        body * 0.58,
        body * 0.40,
        3,
        6,
        c,
    );
    b.foliage.add_ellipsoid(
        Vector3::new(x, y + leg + body * 1.15, z),
        body * 0.24,
        body * 0.26,
        body * 0.24,
        2,
        6,
        head,
    );
}

/// One enclosure: its ground, its barrier, its furniture and its animals.
fn enclosure(b: &mut Batches, p: &Rect, kind: Pen, rng: &mut Rng) -> usize {
    let ground = match kind {
        Pen::Elephant => 0x9a8b70,
        Pen::Savannah => 0xb5a256,
        Pen::BigCat => 0x7f8a4e,
        Pen::Penguin => 0xc8ccd0,
        Pen::Flamingo => 0x8fa36a,
        Pen::Aviary => 0x5f8a3f,
        Pen::Primate => 0x6f9a4a,
        Pen::Bear => 0x7d8a5c,
    };
    b.grass.add_slab(
        p.x0,
        p.z0,
        p.x1,
        p.z1,
        KERB + 0.02,
        scale_color(Color::from_hex(ground), rng.range(0.92, 1.08)),
        Uv::Unit,
    );

    // The barrier. Which one it is says as much about the animal as the animal
    // does: a moat means something that could get out, a mesh dome means
    // something that flies, and a post-and-rail means something that will not
    // try.
    let water = |b: &mut Batches, r: Rect, col: u32| {
        b.water.add_slab(
            r.x0,
            r.z0,
            r.x1,
            r.z1,
            KERB + 0.04,
            Color::from_hex(col),
            Uv::Unit,
        );
    };
    match kind {
        Pen::BigCat | Pen::Primate | Pen::Bear => {
            // A moat inside the perimeter, and a low wall the public leans on.
            let m = p.inset(1.6);
            water(b, Rect { x0: p.x0 + 0.6, z0: p.z0 + 0.6, x1: p.x1 - 0.6, z1: m.z0 }, 0x2f6a86);
            b.trim.add_box(
                Vector3::new(p.x0, KERB, p.z0),
                Vector3::new(p.x1, KERB + 1.0, p.z0 + 0.4),
                Color::from_hex(0x8e8b83),
                Uv::Unit,
            );
        }
        Pen::Aviary => {
            // A mesh dome: ribs and a ring, which reads as netting without
            // needing an alpha pass.
            let (cx, cz) = (p.cx(), p.cz());
            let r = p.w().min(p.d()) * 0.46;
            let h = r * 0.9;
            for k in 0..10 {
                let a = k as f32 / 10.0 * TAU;
                b.trim.add_limb(
                    Vector3::new(cx + r * a.cos(), KERB, cz + r * a.sin()),
                    Vector3::new(cx, KERB + h, cz),
                    0.10,
                    0.06,
                    4,
                    Color::from_hex(0x8a9096),
                    false,
                );
            }
            for ring in [0.35f32, 0.7] {
                let rr = r * (1.0 - ring * 0.55);
                for k in 0..14 {
                    let a = k as f32 / 14.0 * TAU;
                    b.trim.add_limb(
                        Vector3::new(cx + rr * a.cos(), KERB + h * ring, cz + rr * a.sin()),
                        Vector3::new(
                            cx + rr * ((k + 1) as f32 / 14.0 * TAU).cos(),
                            KERB + h * ring,
                            cz + rr * ((k + 1) as f32 / 14.0 * TAU).sin(),
                        ),
                        0.05,
                        0.05,
                        3,
                        Color::from_hex(0x8a9096),
                        false,
                    );
                }
            }
        }
        _ => {
            // Post and rail all the way round.
            let n = ((p.w() + p.d()) / 2.4) as usize;
            for k in 0..n {
                let t = k as f32 / n as f32 * 4.0;
                let (px, pz) = match t as usize {
                    0 => (mix(p.x0, p.x1, t.fract()), p.z0),
                    1 => (p.x1, mix(p.z0, p.z1, t.fract())),
                    2 => (mix(p.x1, p.x0, t.fract()), p.z1),
                    _ => (p.x0, mix(p.z1, p.z0, t.fract())),
                };
                b.trim.add_limb(
                    Vector3::new(px, KERB, pz),
                    Vector3::new(px, KERB + 1.35, pz),
                    0.07,
                    0.06,
                    4,
                    Color::from_hex(0x6b5a44),
                    false,
                );
            }
        }
    }

    // Water features and shelters.
    let inner = p.inset(3.0);
    if inner.w() < 3.0 || inner.d() < 3.0 {
        return 0;
    }
    match kind {
        Pen::Elephant => {
            water(
                b,
                Rect {
                    x0: inner.x0,
                    z0: inner.z1 - inner.d() * 0.34,
                    x1: inner.x0 + inner.w() * 0.5,
                    z1: inner.z1,
                },
                0x5a6f52,
            );
            // A big open-sided shelter.
            b.trim.add_box(
                Vector3::new(inner.x1 - 7.0, KERB, inner.z0),
                Vector3::new(inner.x1, KERB + 4.6, inner.z0 + 5.0),
                Color::from_hex(0xb0a894),
                Uv::Unit,
            );
        }
        Pen::Penguin => {
            water(b, inner.inset(1.0), 0x2f8fc4);
            // Tiled surround and a few rocks.
            for _ in 0..4 {
                let (rx, rz) = (rng.range(p.x0 + 1.0, p.x1 - 1.0), rng.range(p.z0 + 1.0, p.z1 - 1.0));
                b.trim.add_blob(
                    Vector3::new(rx, KERB + 0.3, rz),
                    0.9,
                    0.5,
                    0.9,
                    2,
                    5,
                    0.3,
                    rng.next_u32() as i32 & 0xffff,
                    0.6,
                    Color::from_hex(0x9aa0a6),
                );
            }
        }
        Pen::Flamingo => water(b, inner, 0x4f7f8a),
        Pen::Primate => {
            // An island, with a climbing frame on it.
            water(b, inner, 0x2f6a86);
            let isl = inner.inset(inner.w().min(inner.d()) * 0.24);
            b.grass.add_slab(
                isl.x0,
                isl.z0,
                isl.x1,
                isl.z1,
                KERB + 0.06,
                Color::from_hex(0x6f9a4a),
                Uv::Unit,
            );
            for k in 0..3 {
                let t = (k as f32 + 0.5) / 3.0;
                let x = mix(isl.x0, isl.x1, t);
                b.trim.add_limb(
                    Vector3::new(x, KERB + 0.06, isl.cz()),
                    Vector3::new(x, KERB + 4.0 + k as f32 * 0.8, isl.cz()),
                    0.13,
                    0.11,
                    5,
                    Color::from_hex(0x6b5a44),
                    false,
                );
            }
            b.trim.add_box(
                Vector3::new(isl.x0, KERB + 4.2, isl.cz() - 0.1),
                Vector3::new(isl.x1, KERB + 4.35, isl.cz() + 0.1),
                Color::from_hex(0x6b5a44),
                Uv::Unit,
            );
        }
        Pen::BigCat | Pen::Bear => {
            // Rocks to lie on, and a dead tree to climb.
            for _ in 0..3 + rng.below(3) {
                let (rx, rz) = (
                    rng.range(inner.x0, inner.x1),
                    rng.range(inner.z0, inner.z1),
                );
                let s = rng.range(1.2, 2.8);
                b.trim.add_blob(
                    Vector3::new(rx, KERB + s * 0.35, rz),
                    s,
                    s * 0.55,
                    s * rng.range(0.7, 1.2),
                    2,
                    6,
                    0.3,
                    rng.next_u32() as i32 & 0xffff,
                    0.55,
                    scale_color(Color::from_hex(0x8d8b84), rng.range(0.85, 1.15)),
                );
            }
            add_tree_as(
                b,
                inner.cx(),
                inner.cz(),
                KERB,
                false,
                Species::Bare,
                None,
                0.9,
                rng,
            );
        }
        Pen::Savannah | Pen::Aviary => {
            for _ in 0..2 {
                add_tree_as(
                    b,
                    rng.range(inner.x0, inner.x1),
                    rng.range(inner.z0, inner.z1),
                    KERB,
                    false,
                    Species::Broadleaf,
                    None,
                    rng.range(0.5, 0.8),
                    rng,
                );
            }
        }
    }

    // The animals.
    let mut n = 0usize;
    let spot = |rng: &mut Rng| {
        (
            rng.range(inner.x0 + 1.0, inner.x1 - 1.0),
            rng.range(inner.z0 + 1.0, inner.z1 - 1.0),
        )
    };
    match kind {
        Pen::Elephant => {
            for _ in 0..2 + rng.below(2) {
                let (x, z) = spot(rng);
                quadruped(b, x, KERB, z, rng.range(0.0, TAU), 4.2, 1.5, 1.7, 1.1, 0.8,
                    Color::from_hex(0x7d7a76), rng);
                n += 1;
            }
        }
        Pen::Savannah => {
            for _ in 0..2 + rng.below(2) {
                let (x, z) = spot(rng);
                // Giraffe: the neck is the whole animal.
                quadruped(b, x, KERB, z, rng.range(0.0, TAU), 2.6, 0.72, 2.9, 3.4, 0.18,
                    Color::from_hex(0xc9a352), rng);
                n += 1;
            }
            for _ in 0..3 + rng.below(3) {
                let (x, z) = spot(rng);
                quadruped(b, x, KERB, z, rng.range(0.0, TAU), 2.0, 0.52, 1.0, 0.7, 0.6,
                    Color::from_hex(0xe0dcd4), rng);
                n += 1;
            }
        }
        Pen::BigCat => {
            for _ in 0..1 + rng.below(3) {
                let (x, z) = spot(rng);
                quadruped(b, x, KERB, z, rng.range(0.0, TAU), 1.9, 0.40, 0.55, 0.42, 0.9,
                    Color::from_hex(0xc9924a), rng);
                n += 1;
            }
        }
        Pen::Bear => {
            for _ in 0..1 + rng.below(2) {
                let (x, z) = spot(rng);
                quadruped(b, x, KERB, z, rng.range(0.0, TAU), 1.9, 0.62, 0.62, 0.45, 0.85,
                    Color::from_hex(0x584434), rng);
                n += 1;
            }
        }
        Pen::Penguin => {
            for _ in 0..7 + rng.below(8) {
                let (x, z) = spot(rng);
                wader(b, x, KERB, z, 0.12, 0.62, Color::from_hex(0x23262b),
                    Color::from_hex(0xe8e4dc));
                n += 1;
            }
        }
        Pen::Flamingo => {
            for _ in 0..6 + rng.below(8) {
                let (x, z) = spot(rng);
                wader(b, x, KERB, z, 0.95, 0.5, Color::from_hex(0xe08fa8),
                    Color::from_hex(0xd06f8f));
                n += 1;
            }
        }
        Pen::Aviary => {
            for _ in 0..5 + rng.below(6) {
                let (x, z) = spot(rng);
                wader(b, x, KERB + rng.range(0.0, 2.5), z, 0.10, 0.34,
                    Color::from_hex([0xd04a3au32, 0x2f6ab5, 0xd8a52e][rng.below(3)]),
                    Color::from_hex(0x2a2d31));
                n += 1;
            }
        }
        Pen::Primate => {
            for _ in 0..4 + rng.below(5) {
                let (x, z) = spot(rng);
                wader(b, x, KERB, z, 0.22, 0.44, Color::from_hex(0x6b5a44),
                    Color::from_hex(0x8a7458));
                n += 1;
            }
        }
    }
    n
}

/// The zoo: a walled compound with one entrance and a circuit of enclosures.
///
/// Returns how many animals ended up in it, which is the number worth
/// counting — an empty zoo and a missing zoo look identical from the air.
pub(crate) fn add_zoo(b: &mut Batches, r: &Rect, rng: &mut Rng) -> usize {
    // A small zoo is still a zoo. The floor was set at the size of the first
    // one that happened to fit, which meant the dense layouts got none.
    if r.w() < 84.0 || r.d() < 70.0 {
        return 0;
    }
    if !b.occ.try_claim([r.x0, r.z0, r.x1, r.z1]) {
        return 0;
    }
    // Grounds.
    b.grass.add_slab(
        r.x0,
        r.z0,
        r.x1,
        r.z1,
        KERB,
        scale_color(Color::from_hex(0x5f8f3f), rng.range(0.94, 1.06)),
        Uv::Unit,
    );
    // Perimeter wall, with the entrance block on one side.
    let wall = scale_color(Color::from_hex(0x8e8b83), rng.range(0.9, 1.1));
    for (wx0, wz0, wx1, wz1) in [
        (r.x0, r.z0, r.x1, r.z0 + 0.6),
        (r.x0, r.z1 - 0.6, r.x1, r.z1),
        (r.x0, r.z0, r.x0 + 0.6, r.z1),
        (r.x1 - 0.6, r.z0, r.x1, r.z1),
    ] {
        b.trim.add_box(
            Vector3::new(wx0, KERB, wz0),
            Vector3::new(wx1, KERB + 2.4, wz1),
            wall,
            Uv::Unit,
        );
    }
    // Entrance: a gap in the south wall with a ticket hall and a canopy.
    let gate = mix(r.x0, r.x1, 0.5);
    b.pads.add_slab(
        gate - 7.0,
        r.z0 - 2.0,
        gate + 7.0,
        r.z0 + 12.0,
        KERB + 0.03,
        Color::from_hex(0xb4b0a6),
        Uv::Unit,
    );
    b.trim.add_box(
        Vector3::new(gate - 10.0, KERB, r.z0 + 1.0),
        Vector3::new(gate - 4.0, KERB + 4.2, r.z0 + 6.0),
        Color::from_hex(0xd8d4c8),
        Uv::Unit,
    );
    b.trim.add_box(
        Vector3::new(gate + 4.0, KERB, r.z0 + 1.0),
        Vector3::new(gate + 10.0, KERB + 4.2, r.z0 + 6.0),
        Color::from_hex(0xd8d4c8),
        Uv::Unit,
    );
    b.trim.add_box(
        Vector3::new(gate - 10.5, KERB + 4.2, r.z0 + 0.5),
        Vector3::new(gate + 10.5, KERB + 4.9, r.z0 + 6.5),
        Color::from_hex(0x3f8f5c),
        Uv::Unit,
    );
    add_cctv(
        b,
        Vector3::new(gate, KERB + 4.6, r.z0 + 6.4),
        Vector3::new(0.0, 0.0, -1.0),
        Cctv::Ptz,
        rng,
    );

    // The circuit: a loop path round the inside, which is what a zoo has
    // instead of a street pattern.
    let loop_r = r.inset(14.0);
    let path: Vec<(f32, f32)> = vec![
        (gate, r.z0 + 10.0),
        (loop_r.x0, loop_r.z0),
        (loop_r.x0, loop_r.z1),
        (loop_r.x1, loop_r.z1),
        (loop_r.x1, loop_r.z0),
        (gate, r.z0 + 10.0),
    ];
    b.pads.add_ground_path(
        &path,
        4.2,
        KERB + 0.03,
        scale_color(Color::from_hex(0xb0a894), rng.range(0.94, 1.06)),
        Uv::Unit,
    );

    // Enclosures, strung along the circuit. A grid of cells with a random
    // subset used, so the plan is irregular — which is the point.
    let mut animals = 0usize;
    // The grid adapts to the compound, so a small zoo gets fewer, larger cells
    // instead of a dozen pens all under the minimum and therefore skipped.
    let cols = if r.w() > 130.0 { 4usize } else { 3 };
    let rows = if r.d() > 105.0 { 3usize } else { 2 };
    let mut kinds = PENS;
    // Shuffle, so the same pen is not always in the same corner.
    for i in (1..kinds.len()).rev() {
        kinds.swap(i, rng.below(i + 1));
    }
    let mut ki = 0usize;
    for cx in 0..cols {
        for cz in 0..rows {
            // Leave the middle open: that is where the paths and the cafe go.
            if cx > 0 && cx < cols - 1 && cz > 0 && cz < rows - 1 {
                continue;
            }
            let cell = Rect {
                x0: mix(r.x0 + 3.0, r.x1 - 3.0, cx as f32 / cols as f32),
                z0: mix(r.z0 + 3.0, r.z1 - 3.0, cz as f32 / rows as f32),
                x1: mix(r.x0 + 3.0, r.x1 - 3.0, (cx + 1) as f32 / cols as f32),
                z1: mix(r.z0 + 3.0, r.z1 - 3.0, (cz + 1) as f32 / rows as f32),
            };
            // Jittered inset, so no two pens are the same size.
            // Inset scaled to the cell, so a small zoo is not all margin.
            let m = (cell.w().min(cell.d()) * 0.10).clamp(1.2, 5.0);
            let pen = Rect {
                x0: cell.x0 + rng.range(m * 0.6, m),
                z0: cell.z0 + rng.range(m * 0.6, m),
                x1: cell.x1 - rng.range(m * 0.6, m),
                z1: cell.z1 - rng.range(m * 0.6, m),
            };
            if pen.w() < 11.0 || pen.d() < 9.0 {
                continue;
            }
            animals += enclosure(b, &pen, kinds[ki % kinds.len()], rng);
            ki += 1;
        }
    }
    // A cafe and some benches in the middle.
    let mid = Rect {
        x0: mix(r.x0, r.x1, 0.40),
        z0: mix(r.z0, r.z1, 0.40),
        x1: mix(r.x0, r.x1, 0.60),
        z1: mix(r.z0, r.z1, 0.60),
    };
    b.pads.add_slab(
        mid.x0,
        mid.z0,
        mid.x1,
        mid.z1,
        KERB + 0.03,
        Color::from_hex(0xb4b0a6),
        Uv::Unit,
    );
    b.trim.add_box(
        Vector3::new(mid.x0, KERB, mid.z0),
        Vector3::new(mid.x0 + 12.0, KERB + 4.0, mid.z0 + 8.0),
        Color::from_hex(0xd8d4c8),
        Uv::Unit,
    );
    b.trim.add_gable(
        Vector3::new(mid.x0 - 0.4, KERB + 4.0, mid.z0 - 0.4),
        Vector3::new(mid.x0 + 12.4, KERB + 4.0, mid.z0 + 8.4),
        1.2,
        Color::from_hex(0x3f8f5c),
    );
    for k in 0..4 {
        let t = (k as f32 + 0.5) / 4.0;
        add_tree(b, mix(mid.x0, mid.x1, t), mid.z1 + 2.0, KERB, false, false, rng);
    }
    animals
}
