//! Part of the `simcity` example; see `mod.rs`.
#![allow(dead_code)]

use super::*;

// ---------------------------------------------------------------------------
// Planting.
// ---------------------------------------------------------------------------

/// Which shape of tree. Each has a silhouette the others cannot fake, which is
/// the only reason to carry more than one.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Species {
    /// Trunk forking into primaries, a cluster on each.
    Broadleaf,
    /// Single leader with whorls of tiers narrowing to a point.
    Conifer,
    /// Tall and narrow — poplar, lombardy.
    Columnar,
    /// Broadleaf with the clusters hung below the branch tips.
    Weeping,
    /// Bare stem with radiating fronds. Warm layouts only.
    Palm,
    /// Dead or winter: the armature with nothing on it.
    Bare,
}

/// Leaf colour. Most trees are green; the rest is what stops an avenue of them
/// reading as one repeated asset.
fn leaf_colour(rng: &mut Rng) -> Color {
    let roll = rng.f();
    if roll < 0.05 {
        // Blossom.
        mix_color(
            Color::from_hex(0xf0c6d4),
            Color::from_hex(0xfaf0f2),
            rng.f(),
        )
    } else if roll < 0.17 {
        // Turning: amber through rust.
        mix_color(
            Color::from_hex(0xc98a2e),
            Color::from_hex(0x9c4522),
            rng.f(),
        )
    } else {
        mix_color(
            Color::from_hex(0x2f5323),
            Color::from_hex(0x6a8b3c),
            rng.f(),
        )
    }
}

/// A tree, built around its branches rather than around a blob on a stick.
///
/// The crown is not one mass: each primary limb carries its own cluster, so
/// the outline is made of several smaller lumps at different heights and there
/// is visible structure underneath it. That is the difference between a tree
/// and a lollipop, and it only shows at street level — hence the `near` split,
/// which drops the whole armature for the outskirts and returns a single blob.
///
/// `warm` admits palms; a palm among the conifers of a temperate grid looks
/// like a mistake, so only the waterfront layout asks for them.
pub(crate) fn add_tree(
    b: &mut Batches,
    x: f32,
    z: f32,
    ground: f32,
    near: bool,
    warm: bool,
    rng: &mut Rng,
) {
    // A tree needs its trunk's worth of ground and a little around it. If
    // something is already there — a lamp, a bin, a bench — do not plant it.
    if !b.occ.try_spot(x, z, 1.15) {
        return;
    }
    let roll = rng.f();
    let species = if warm && roll < 0.30 {
        Species::Palm
    } else if roll < 0.18 {
        Species::Conifer
    } else if roll < 0.28 {
        Species::Columnar
    } else if roll < 0.40 {
        Species::Weeping
    } else if roll < 0.44 {
        Species::Bare
    } else {
        Species::Broadleaf
    };
    add_tree_as(b, x, z, ground, near, species, None, 1.0, rng);
}

/// The same tree with the species chosen for it, and optionally a crown colour
/// and a size multiplier.
///
/// Parks are planted, not seeded: a formal square wants one species in a row,
/// a pond wants willows on the bank and a spring garden wants blossom. Letting
/// the caller name the tree is what separates a designed landscape from a
/// scatter of random ones.
pub(crate) fn add_tree_as(
    b: &mut Batches,
    x: f32,
    z: f32,
    ground: f32,
    near: bool,
    species: Species,
    tint: Option<Color>,
    size: f32,
    rng: &mut Rng,
) {
    let scale = rng.range(0.72, 1.40) * size;
    // Bark by species: conifers run redder, planes and limes greyer, palms
    // pale and fibrous.
    let bark = scale_color(
        match species {
            Species::Conifer => Color::from_hex(0x5d3b28),
            Species::Columnar => Color::from_hex(0x585044),
            Species::Palm => Color::from_hex(0x8a7a5c),
            Species::Bare => Color::from_hex(0x4c443c),
            _ => Color::from_hex(C_TRUNK),
        },
        rng.range(0.85, 1.2),
    );
    let trunk_h = match species {
        Species::Conifer => 1.0,
        Species::Columnar => 2.4,
        Species::Palm => 4.4,
        _ => 2.1,
    } * scale;
    let seed = rng.next_u32() as i32 & 0xffff;
    let leaf = tint.unwrap_or_else(|| leaf_colour(rng));
    // Resolution: a crown at four rings by seven sectors is legible but
    // plainly faceted a few metres away, which is where half these trees are.
    let (rings, sectors) = if near { (6, 10) } else { (3, 6) };
    let bark_segs = if near { 9 } else { 4 };
    let jitter = if near { 0.24 } else { 0.28 };

    // Root flare: a short, much wider taper at the foot. A trunk that meets
    // the ground at its own diameter reads as a pole stuck in a hole.
    b.trim.add_cylinder(
        Vector3::new(x, ground, z),
        0.34 * scale,
        0.23 * scale,
        0.28 * scale,
        bark_segs,
        bark,
        false,
        Uv::Unit,
    );

    // Everything that forks: broadleaf, weeping and bare share an armature.
    let forked = matches!(
        species,
        Species::Broadleaf | Species::Weeping | Species::Bare
    );
    if forked {
        let fork = ground + trunk_h;
        b.trim.add_cylinder(
            Vector3::new(x, ground + 0.26 * scale, z),
            0.23 * scale,
            0.15 * scale,
            trunk_h,
            bark_segs,
            bark,
            false,
            Uv::Unit,
        );
        if !near {
            if species != Species::Bare {
                let r = rng.range(1.2, 1.8) * scale;
                b.foliage.add_blob(
                    Vector3::new(x, fork + r * 0.7, z),
                    r,
                    r * 0.85,
                    r,
                    rings,
                    sectors,
                    jitter,
                    seed,
                    0.62,
                    leaf,
                );
            }
            return;
        }
        let limbs = 4 + rng.below(3);
        let phase = rng.range(0.0, TAU);
        for i in 0..limbs {
            let a = phase + i as f32 * TAU / limbs as f32 + rng.range(-0.30, 0.30);
            // Reach and cluster size are tuned against each other: too much
            // reach and the crown reads as a bunch of grapes, too little and
            // the clusters merge back into one mass.
            let reach = rng.range(0.52, 0.95) * scale;
            let rise = rng.range(0.85, 1.55) * scale;
            let elbow = Vector3::new(
                x + a.cos() * reach * 0.5,
                fork + rise * 0.6,
                z + a.sin() * reach * 0.5,
            );
            let tip = Vector3::new(x + a.cos() * reach, fork + rise, z + a.sin() * reach);
            b.trim.add_limb(
                Vector3::new(x, fork - 0.20 * scale, z),
                elbow,
                0.115 * scale,
                0.070 * scale,
                6,
                bark,
                false,
            );
            b.trim
                .add_limb(elbow, tip, 0.070 * scale, 0.032 * scale, 5, bark, false);
            if species == Species::Bare {
                // Twigs, since there is no foliage to hide the ends.
                for j in 0..2 {
                    let ta = a + rng.range(-0.9, 0.9);
                    b.trim.add_limb(
                        tip,
                        Vector3::new(
                            tip.x + ta.cos() * 0.35 * scale,
                            tip.y + rng.range(0.25, 0.55) * scale,
                            tip.z + ta.sin() * 0.35 * scale,
                        ),
                        0.030 * scale,
                        0.012 * scale,
                        4,
                        bark,
                        false,
                    );
                    let _ = j;
                }
                continue;
            }
            let cr = rng.range(0.68, 1.02) * scale;
            // A weeping crown hangs below its branch tips instead of sitting
            // on them, which is the whole of the shape.
            let drop = if species == Species::Weeping {
                -cr * 0.55
            } else {
                cr * 0.22
            };
            b.foliage.add_blob(
                Vector3::new(tip.x, tip.y + drop, tip.z),
                cr,
                cr * if species == Species::Weeping {
                    rng.range(1.15, 1.5)
                } else {
                    rng.range(0.76, 0.98)
                },
                cr * rng.range(0.88, 1.10),
                rings,
                sectors,
                jitter,
                seed + i as i32 * 613,
                0.62,
                scale_color(leaf, rng.range(0.86, 1.14)),
            );
        }
        if species != Species::Bare {
            // A smaller mass over the fork, tying the clusters together so the
            // crown is not a ring of separate balls.
            let cr = rng.range(0.80, 1.10) * scale;
            b.foliage.add_blob(
                Vector3::new(x, fork + cr * 0.58, z),
                cr,
                cr * 0.8,
                cr,
                rings,
                sectors,
                jitter,
                seed + 97,
                0.58,
                scale_color(leaf, rng.range(0.9, 1.05)),
            );
        }
        return;
    }

    match species {
        Species::Conifer => {
            // A single leader running the whole height, with the tiers hung on
            // it. The trunk showing between the lowest branches is most of
            // what says conifer.
            let height = rng.range(4.4, 6.4) * scale;
            b.trim.add_cylinder(
                Vector3::new(x, ground + 0.26 * scale, z),
                0.22 * scale,
                0.05 * scale,
                trunk_h + height * 0.92,
                bark_segs,
                bark,
                false,
                Uv::Unit,
            );
            let base = ground + trunk_h * 0.85;
            let tiers = if near { 6 + rng.below(3) } else { 3 };
            for i in 0..tiers {
                let t = i as f32 / (tiers - 1).max(1) as f32;
                let r = (1.62 - 1.25 * t) * scale * rng.range(0.88, 1.12);
                // Wide and shallow, set well apart. Tall overlapping lumps
                // merge into one taper and the tree is a party hat again; the
                // notches between tiers are the whole silhouette.
                b.foliage.add_blob(
                    Vector3::new(x, base + height * t * 0.86, z),
                    r,
                    height * (0.15 - 0.03 * t),
                    r,
                    if near { 4 } else { 3 },
                    sectors,
                    jitter * 1.4,
                    seed + i as i32 * 977,
                    0.46,
                    scale_color(leaf, rng.range(0.88, 1.12)),
                );
            }
            b.foliage.add_blob(
                Vector3::new(x, base + height * 0.94, z),
                0.30 * scale,
                height * 0.22,
                0.30 * scale,
                3,
                6,
                jitter,
                seed + 41,
                0.46,
                leaf,
            );
        }

        Species::Columnar => {
            let h = rng.range(3.6, 5.4) * scale;
            b.trim.add_cylinder(
                Vector3::new(x, ground + 0.26 * scale, z),
                0.22 * scale,
                0.09 * scale,
                trunk_h + h * 0.7,
                bark_segs,
                bark,
                false,
                Uv::Unit,
            );
            // Stacked masses rather than one capsule: a poplar is a column of
            // foliage, not a sausage.
            let lumps = if near { 4 } else { 1 };
            let r = rng.range(0.58, 0.88) * scale;
            for i in 0..lumps {
                let t = if lumps == 1 {
                    0.5
                } else {
                    i as f32 / (lumps - 1) as f32
                };
                let lr = r * (1.0 - 0.30 * (t - 0.35).abs());
                b.foliage.add_blob(
                    Vector3::new(x, ground + trunk_h * 0.5 + h * (0.18 + t * 0.62), z),
                    lr,
                    h * if lumps == 1 { 0.5 } else { 0.24 },
                    lr * rng.range(0.9, 1.1),
                    rings,
                    sectors,
                    jitter * 1.15,
                    seed + i * 331,
                    0.6,
                    scale_color(leaf, rng.range(0.9, 1.1)),
                );
            }
        }

        Species::Palm => {
            // A bare stem with a slight lean, and fronds radiating from the
            // top. `add_limb_flat` makes each frond a blade rather than a
            // tube — a round frond reads as a sausage on a stick.
            let lean = rng.range(0.0, 0.55) * scale;
            let la = rng.range(0.0, TAU);
            let crown = Vector3::new(x + la.cos() * lean, ground + trunk_h, z + la.sin() * lean);
            b.trim.add_limb(
                Vector3::new(x, ground + 0.26 * scale, z),
                crown,
                0.20 * scale,
                0.13 * scale,
                bark_segs,
                bark,
                false,
            );
            let fronds = if near { 7 + rng.below(4) } else { 5 };
            let phase = rng.range(0.0, TAU);
            let palm_green = mix_color(leaf, Color::from_hex(0x4c7a2e), 0.5);
            for i in 0..fronds {
                let a = phase + i as f32 * TAU / fronds as f32 + rng.range(-0.18, 0.18);
                let reach = rng.range(1.5, 2.3) * scale;
                let droop = rng.range(-0.55, 0.25) * scale;
                b.foliage.add_limb_flat(
                    Vector3::new(crown.x, crown.y + 0.12 * scale, crown.z),
                    Vector3::new(
                        crown.x + a.cos() * reach,
                        crown.y + droop,
                        crown.z + a.sin() * reach,
                    ),
                    0.20 * scale,
                    0.03 * scale,
                    0.16,
                    if near { 5 } else { 3 },
                    scale_color(palm_green, rng.range(0.85, 1.15)),
                    false,
                );
            }
        }

        _ => {}
    }

    // Fruit, and what has fallen off. Both are small and both are the sort of
    // thing that only shows at street level, which is where half these trees
    // are — a tree that drops nothing and carries nothing is a lollipop with
    // structure.
    if near && matches!(species, Species::Broadleaf | Species::Weeping) {
        add_fruit_and_litter(b, x, z, ground, scale, rng);
    }
}

/// Colours bedding plants are drawn from. Saturated on purpose: everything
/// else at this scale is a construction material.
pub(crate) const BLOOM: [u32; 8] = [
    0xd8456b, 0xe8a02a, 0xf0ece0, 0x9f4fb8, 0xd4442b, 0xe86a9a, 0xf2d43a, 0x7a5ec8,
];

/// A row of flowers in a container: a trough of soil with heads standing out
/// of it. Window boxes, hanging baskets and planted borders are all this.
pub(crate) fn add_flowers(
    b: &mut Batches,
    a: Vector3,
    c: Vector3,
    width: f32,
    height: f32,
    rng: &mut Rng,
) {
    let d = c - a;
    let len = (d.x * d.x + d.z * d.z).sqrt();
    if len < 0.1 {
        return;
    }
    let family = Color::from_hex(BLOOM[rng.below(BLOOM.len())]);
    // One head every quarter metre, not every sixth: at two ellipsoids a head
    // a long border was costing thousands of triangles on its own, and at any
    // distance a denser row reads identically.
    let n = ((len / 0.26).round() as usize).clamp(3, 26);
    for i in 0..n {
        let t = (i as f32 + 0.5) / n as f32;
        let p = Vector3::new(a.x + d.x * t, a.y, a.z + d.z * t);
        // Foliage first, then a head above it. Two blobs is enough: what reads
        // is a band of green with a band of colour standing on it.
        let jitter = |r: &mut Rng| r.range(-width * 0.35, width * 0.35);
        b.foliage.add_ellipsoid(
            Vector3::new(p.x + jitter(rng), p.y + height * 0.28, p.z + jitter(rng)),
            width * 0.36,
            height * 0.36,
            width * 0.36,
            // Two rings by four sectors is a tetrahedron. These are small and
            // numerous, so the count matters — but not that much, and a bed of
            // planting made of tetrahedra reads as gravel.
            3,
            6,
            scale_color(Color::from_hex(0x3f6b2c), rng.range(0.82, 1.18)),
        );
        if rng.chance(0.72) {
            b.foliage.add_ellipsoid(
                Vector3::new(p.x + jitter(rng), p.y + height * rng.range(0.62, 0.95), p.z + jitter(rng)),
                width * 0.24,
                height * 0.22,
                width * 0.24,
                2,
                4,
                scale_color(family, rng.range(0.78, 1.2)),
            );
        }
    }
}

/// A window box under a sill: a trough with flowers in it.
pub(crate) fn add_window_box(b: &mut Batches, x: f32, y: f32, z: f32, front: f32, rng: &mut Rng) {
    let w = rng.range(0.9, 1.5);
    let trough = scale_color(Color::from_hex(0x6b5136), rng.range(0.85, 1.15));
    b.trim.add_box(
        Vector3::new(x - w * 0.5, y, z - 0.11 + front * 0.11),
        Vector3::new(x + w * 0.5, y + 0.22, z + 0.11 + front * 0.11),
        trough,
        Uv::Unit,
    );
    add_flowers(
        b,
        Vector3::new(x - w * 0.42, y + 0.20, z + front * 0.11),
        Vector3::new(x + w * 0.42, y + 0.20, z + front * 0.11),
        0.26,
        0.34,
        rng,
    );
}

/// A hanging basket on a lamp column: a bracket, a bowl and a spill of colour.
pub(crate) fn add_hanging_basket(b: &mut Batches, x: f32, y: f32, z: f32, rng: &mut Rng) {
    let iron = Color::from_hex(0x2f3a34);
    for s in [-1.0f32, 1.0] {
        b.trim.add_limb(
            Vector3::new(x, y, z),
            Vector3::new(x + s * 0.42, y + 0.10, z),
            0.022,
            0.018,
            4,
            iron,
            false,
        );
        let c = Vector3::new(x + s * 0.42, y - 0.02, z);
        b.trim
            .add_cylinder(c, 0.20, 0.24, 0.20, 8, iron, false, Uv::Unit);
        // The planting spills over the rim, which is what a basket looks like.
        let family = Color::from_hex(BLOOM[rng.below(BLOOM.len())]);
        for _ in 0..(7 + rng.below(7)) {
            let a = rng.range(0.0, TAU);
            let r = rng.range(0.0, 0.26);
            b.foliage.add_ellipsoid(
                Vector3::new(c.x + r * a.cos(), c.y + rng.range(-0.16, 0.12), c.z + r * a.sin()),
                rng.range(0.07, 0.12),
                rng.range(0.06, 0.11),
                rng.range(0.07, 0.12),
                3,
                5,
                if rng.chance(0.55) {
                    scale_color(family, rng.range(0.8, 1.2))
                } else {
                    scale_color(Color::from_hex(0x3f6b2c), rng.range(0.85, 1.15))
                },
            );
        }
    }
}

/// Fruit in the crown and leaf litter under it.
///
/// The litter is the cheaper half and does more work: a scatter of flat
/// coloured quads on the ground under every broadleaf turns a lawn with trees
/// on it into a lawn under trees.
pub(crate) fn add_fruit_and_litter(
    b: &mut Batches,
    x: f32,
    z: f32,
    ground: f32,
    scale: f32,
    rng: &mut Rng,
) {
    // Roughly one broadleaf in five is carrying something.
    if rng.chance(0.20) {
        const FRUIT: [u32; 5] = [0xc4342a, 0xd88b2a, 0xe0c23a, 0x8f2f4a, 0xd86a86];
        let colour = Color::from_hex(FRUIT[rng.below(FRUIT.len())]);
        let n = 9 + rng.below(16);
        for _ in 0..n {
            // Hung around the outside of the crown, where the light is and
            // where they would actually be visible.
            let a = rng.range(0.0, TAU);
            let r = rng.range(0.55, 1.05) * scale * rng.range(1.0, 1.35);
            let y = ground + (2.3 + rng.range(-0.5, 1.5)) * scale;
            b.foliage.add_ellipsoid(
                Vector3::new(x + r * a.cos(), y, z + r * a.sin()),
                0.085 * scale,
                0.095 * scale,
                0.085 * scale,
                3,
                5,
                scale_color(colour, rng.range(0.82, 1.15)),
            );
        }
        // Windfalls under it.
        for _ in 0..(2 + rng.below(5)) {
            let a = rng.range(0.0, TAU);
            let d = rng.range(0.4, 1.9) * scale;
            b.foliage.add_ellipsoid(
                Vector3::new(x + d * a.cos(), ground + 0.07, z + d * a.sin()),
                0.075,
                0.055,
                0.075,
                3,
                5,
                scale_color(colour, rng.range(0.6, 0.95)),
            );
        }
    }

    // Leaf litter: flat, at every angle, thickest near the trunk.
    const LITTER: [u32; 5] = [0xa8752a, 0x8f5a24, 0xc19a3a, 0x6b7a30, 0x9c4522];
    let n = 14 + rng.below(18);
    for _ in 0..n {
        let a = rng.range(0.0, TAU);
        // Square-rooted so the scatter is even in area rather than crowding
        // the middle, then biased back in a little.
        let d = rng.f().sqrt() * 2.4 * scale * rng.range(0.7, 1.15);
        add_ground_quad(
            &mut b.litter,
            x + d * a.cos(),
            z + d * a.sin(),
            ground + 0.008 + rng.range(0.0, 0.006),
            rng.range(0.09, 0.20),
            rng.range(0.07, 0.16),
            rng.range(0.0, TAU),
            scale_color(
                Color::from_hex(LITTER[rng.below(LITTER.len())]),
                rng.range(0.75, 1.15),
            ),
        );
    }
}

/// A street lamp: pole, arm, head, and the pool of light it throws after dark.
/// A street lamp: column, arm, luminaire, the pool it throws, and the shaft of
/// light between them.
///
/// `warm` picks low-pressure sodium over LED. Real cities are part way through
/// swapping one for the other and the mixture is visible from any bridge —
/// an avenue of orange with one cold white side street off it.
pub(crate) fn add_lamp(
    b: &mut Batches,
    lamps: &mut Vec<Vector3>,
    x: f32,
    z: f32,
    toward: Vector3,
    warm: bool,
) {
    // A lamp column is small but its base is not, and it is the one thing on
    // the pavement that everything else should defer to — so it claims early,
    // before the trees and the furniture are scattered.
    if !b.occ.try_spot(x, z, 0.85) {
        return;
    }
    let metal = Color::from_hex(0x3a3d42);
    let h = 6.4;
    // Base casting, then the column in two tapers: a single cylinder from
    // pavement to lantern reads as a broom handle.
    b.trim.add_cylinder(
        Vector3::new(x, KERB, z),
        0.20,
        0.155,
        0.55,
        8,
        scale_color(metal, 0.85),
        true,
        Uv::Unit,
    );
    b.trim.add_cylinder(
        Vector3::new(x, KERB + 0.55, z),
        0.135,
        0.105,
        h * 0.62,
        7,
        metal,
        false,
        Uv::Unit,
    );
    b.trim.add_cylinder(
        Vector3::new(x, KERB + 0.55 + h * 0.62, z),
        0.105,
        0.075,
        h * 0.38,
        7,
        metal,
        false,
        Uv::Unit,
    );

    // The arm sweeps out rather than jutting: three short limbs on a quarter
    // circle, which at this scale is the difference between a lamp post and a
    // gallows.
    let arm = 1.7;
    let top = KERB + h + 0.55;
    let mut prev = Vector3::new(x, top - 0.55, z);
    for i in 1..=3 {
        let t = i as f32 / 3.0;
        let p = Vector3::new(
            x + toward.x * arm * t,
            top - 0.55 + (1.0 - (1.0 - t) * (1.0 - t)) * 0.5,
            z + toward.z * arm * t,
        );
        b.trim.add_limb(prev, p, 0.075, 0.062, 6, metal, false);
        prev = p;
    }
    let (ex, ez) = (prev.x, prev.z);
    let lens_y = prev.y - 0.28;

    // Luminaire: a shallow shade with the lens tucked under its lip, so the
    // fitting is what glows and not the whole head.
    b.trim.add_cylinder(
        Vector3::new(ex, prev.y - 0.30, ez),
        0.40,
        0.20,
        0.30,
        9,
        Color::from_hex(0x50545a),
        true,
        Uv::Unit,
    );
    let lamp_col = if warm {
        Color::from_hex(0xffb35c)
    } else {
        Color::from_hex(0xdfeaff)
    };
    b.glow
        .add_glow_disc(Vector3::new(ex, lens_y, ez), 0.33, 9, lamp_col);

    // Remembered so the nearest few can be given real spot lights at night.
    lamps.push(Vector3::new(ex, lens_y - 0.04, ez));

    // Hanging baskets on the column. Not every lamp — a whole avenue of them
    // reads as municipal pride rather than as a street.
    if city_hash2((x * 3.0) as i32, (z * 3.0) as i32) < 0.22 {
        let mut r = Rng::new(((x * 100.0) as i64 as u64) ^ ((z * 100.0) as i64 as u64) << 20);
        add_hanging_basket(b, x, KERB + h * 0.52, z, &mut r);
    }

    // The shaft. A cone from the lens to the pool, translucent and unlit — the
    // single cheapest thing that makes a night street read as lit rather than
    // as tarmac with bright decals painted on it. Double sided: from under it
    // you are inside the cone and would otherwise see nothing at all.
    let pool_r = 5.2;
    // The cone is narrower than the pool it lights. A shaft as wide as the
    // pool spans the carriageway and reads as a tent pitched over the road.
    let cone_r = 2.5;
    const SEGS: usize = 16;
    for i in 0..SEGS {
        let (a0, a1) = (
            i as f32 / SEGS as f32 * TAU,
            (i + 1) as f32 / SEGS as f32 * TAU,
        );
        let top_r = 0.30;
        let p = |a: f32, r: f32, y: f32| [ex + r * a.cos(), y, ez + r * a.sin()];
        // Bright at the fitting, gone by the time it reaches the road. Fading
        // to black rather than to a dim tint is what stops the cone having a
        // visible bottom edge across the tarmac.
        let hot = scale_color(lamp_col, 0.85);
        let cold = Color::new(0.0, 0.0, 0.0);
        b.shaft.quad_shaded(
            [
                p(a0, top_r, lens_y),
                p(a0, cone_r, 0.04),
                p(a1, cone_r, 0.04),
                p(a1, top_r, lens_y),
            ],
            [a0.cos(), 0.25, a0.sin()],
            [hot, cold, cold, hot],
        );
    }

    // The pool on the road, elliptical along the kerb rather than a circle.
    b.glow.add_glow_disc(
        Vector3::new(ex, 0.03, ez),
        pool_r,
        16,
        scale_color(lamp_col, 0.72),
    );
}

/// A pile of refuse sacks on the pavement, and sometimes a fox's worth of
/// spilled rubbish beside it.
///
/// The one thing every real street has that a generated one never does. Also
/// the reason there are rats: they are placed against these.
pub(crate) fn add_trash(b: &mut Batches, x: f32, z: f32, rng: &mut Rng) -> bool {
    if !b.occ.try_spot(x, z, 1.1) {
        return false;
    }
    let n = 2 + rng.below(5);
    for _ in 0..n {
        let (bx, bz) = (x + rng.range(-0.7, 0.7), z + rng.range(-0.5, 0.5));
        let r = rng.range(0.26, 0.42);
        // Black sacks mostly, with the odd council-issue colour among them.
        let sack = if rng.chance(0.78) {
            scale_color(Color::from_hex(0x24262a), rng.range(0.85, 1.2))
        } else {
            scale_color(
                Color::from_hex([0x2f5f3a, 0x2f4f7a, 0x6b5a2f][rng.below(3)]),
                rng.range(0.85, 1.15),
            )
        };
        // Slumped rather than spherical: a sack of rubbish is wider than it is
        // tall and never symmetrical.
        b.trim.add_blob(
            Vector3::new(bx, KERB + r * 0.72, bz),
            r * rng.range(0.95, 1.25),
            r * rng.range(0.62, 0.85),
            r * rng.range(0.95, 1.25),
            4,
            7,
            0.16,
            rng.next_u32() as i32 & 0xffff,
            0.5,
            sack,
        );
        // The knot at the neck.
        b.trim.add_limb(
            Vector3::new(bx, KERB + r * 1.28, bz),
            Vector3::new(bx + rng.range(-0.1, 0.1), KERB + r * 1.62, bz + rng.range(-0.1, 0.1)),
            0.05,
            0.02,
            4,
            scale_color(sack, 0.8),
            true,
        );
    }
    // Spillage: flat scraps on the pavement round the pile.
    if rng.chance(0.45) {
        for _ in 0..(2 + rng.below(5)) {
            let a = rng.range(0.0, TAU);
            let d = rng.range(0.6, 1.6);
            add_ground_quad(
                &mut b.trim,
                x + d * a.cos(),
                z + d * a.sin(),
                KERB + 0.012,
                rng.range(0.10, 0.26),
                rng.range(0.10, 0.26),
                rng.range(0.0, TAU),
                scale_color(Color::from_hex(0xb9b2a2), rng.range(0.7, 1.1)),
            );
        }
    }
    true
}

/// An irregular flat disc, for standing water and stains. A circle reads as a
/// decal; jittering the radius per segment reads as a puddle.
pub(crate) fn add_ground_blob(
    m: &mut MeshBuilder,
    cx: f32,
    cz: f32,
    y: f32,
    rx: f32,
    rz: f32,
    rng: &mut Rng,
    c: Color,
) {
    const SEGS: usize = 11;
    let mut r = [0.0f32; SEGS];
    for v in r.iter_mut() {
        *v = rng.range(0.62, 1.0);
    }
    let at = |i: usize| {
        let a = i as f32 / SEGS as f32 * TAU;
        [cx + a.cos() * rx * r[i % SEGS], y, cz + a.sin() * rz * r[i % SEGS]]
    };
    for i in 0..SEGS {
        m.tri(
            [[cx, y, cz], at((i + 1) % SEGS), at(i)],
            [0.0, 1.0, 0.0],
            [[0.5, 0.5]; 3],
            [c; 3],
        );
    }
}

/// A flat quad lying on the ground at an arbitrary angle. `add_slab` is
/// axis-aligned, and litter that all points the same way is not litter.
pub(crate) fn add_ground_quad(m: &mut MeshBuilder, cx: f32, cz: f32, y: f32, w: f32, d: f32, a: f32, c: Color) {
    let (s, k) = (a.sin(), a.cos());
    let (hw, hd) = (w * 0.5, d * 0.5);
    let corner = |sx: f32, sz: f32| {
        [
            cx + (sx * hw) * k - (sz * hd) * s,
            y,
            cz + (sx * hw) * s + (sz * hd) * k,
        ]
    };
    m.quad(
        [
            corner(-1.0, -1.0),
            corner(-1.0, 1.0),
            corner(1.0, 1.0),
            corner(1.0, -1.0),
        ],
        [0.0, 1.0, 0.0],
        Uv::Unit,
        c,
    );
}

/// One piece of street furniture at `(x, z)`, facing `toward` the kerb.
///
/// This is the layer a city is actually made of at eye level: without it a
/// pavement is a grey ribbon, and no amount of work on the towers fixes that.
pub(crate) fn add_furniture(b: &mut Batches, x: f32, z: f32, toward: Vector3, rng: &mut Rng) {
    if !b.occ.try_spot(x, z, 0.75) {
        return;
    }
    let dark = Color::from_hex(0x3a3d42);
    let pick = rng.f();
    if pick < 0.20 {
        // Wheelie bin, lid slightly proud of the body.
        let body = scale_color(Color::from_hex(0x2f4a35), rng.range(0.8, 1.2));
        b.trim.add_box(
            Vector3::new(x - 0.31, KERB, z - 0.27),
            Vector3::new(x + 0.31, KERB + 0.96, z + 0.27),
            body,
            Uv::Unit,
        );
        b.trim.add_box(
            Vector3::new(x - 0.34, KERB + 0.96, z - 0.30),
            Vector3::new(x + 0.34, KERB + 1.06, z + 0.30),
            scale_color(body, 0.82),
            Uv::Unit,
        );
    } else if pick < 0.34 {
        // Bollard.
        b.trim.add_cylinder(
            Vector3::new(x, KERB, z),
            0.11,
            0.10,
            0.92,
            8,
            dark,
            true,
            Uv::Unit,
        );
    } else if pick < 0.44 {
        // Hydrant: a stubby column with a cap and two side outlets.
        let red = Color::from_hex(0xb3372c);
        b.trim.add_cylinder(
            Vector3::new(x, KERB, z),
            0.16,
            0.13,
            0.62,
            8,
            red,
            true,
            Uv::Unit,
        );
        b.trim.add_cylinder(
            Vector3::new(x, KERB + 0.62, z),
            0.20,
            0.09,
            0.20,
            8,
            red,
            true,
            Uv::Unit,
        );
        for s in [-1.0f32, 1.0] {
            b.trim.add_box(
                Vector3::new(x + s * 0.14 - 0.06, KERB + 0.34, z - 0.06),
                Vector3::new(x + s * 0.14 + 0.06, KERB + 0.46, z + 0.06),
                red,
                Uv::Unit,
            );
        }
    } else if pick < 0.56 {
        // Bench, turned to face the road.
        let wood = scale_color(Color::from_hex(0x6b4a30), rng.range(0.85, 1.15));
        let along = Vector3::new(-toward.z, 0.0, toward.x);
        let (hx, hz) = (along.x * 0.85, along.z * 0.85);
        b.trim.add_box(
            Vector3::new(x - hx.abs() - 0.22, KERB + 0.42, z - hz.abs() - 0.22),
            Vector3::new(x + hx.abs() + 0.22, KERB + 0.50, z + hz.abs() + 0.22),
            wood,
            Uv::Unit,
        );
        b.trim.add_box(
            Vector3::new(
                x - hx.abs() - 0.22 + toward.x * 0.20,
                KERB + 0.50,
                z - hz.abs() - 0.22 + toward.z * 0.20,
            ),
            Vector3::new(
                x + hx.abs() + 0.22 + toward.x * 0.28,
                KERB + 0.94,
                z + hz.abs() + 0.22 + toward.z * 0.28,
            ),
            wood,
            Uv::Unit,
        );
    } else if pick < 0.66 {
        // Post box.
        b.trim.add_cylinder(
            Vector3::new(x, KERB, z),
            0.28,
            0.28,
            1.18,
            10,
            Color::from_hex(0xa8271f),
            true,
            Uv::Unit,
        );
    } else if pick < 0.78 {
        // Utility cabinet, the grey box on every corner.
        b.trim.add_box(
            Vector3::new(x - 0.40, KERB, z - 0.24),
            Vector3::new(x + 0.40, KERB + 1.22, z + 0.24),
            scale_color(Color::from_hex(0x7c8288), rng.range(0.85, 1.1)),
            Uv::Unit,
        );
    } else if pick < 0.88 {
        // Planter with a shrub.
        let stone = Color::from_hex(0x8b8880);
        b.trim.add_box(
            Vector3::new(x - 0.46, KERB, z - 0.46),
            Vector3::new(x + 0.46, KERB + 0.52, z + 0.46),
            stone,
            Uv::Unit,
        );
        b.foliage.add_ellipsoid(
            Vector3::new(x, KERB + 0.78, z),
            0.44,
            0.36,
            0.44,
            3,
            6,
            scale_color(Color::from_hex(0x40662e), rng.range(0.85, 1.15)),
        );
    } else {
        // Traffic cone, in a pair.
        for i in 0..2 {
            let ox = x + toward.x * 0.1 + (i as f32 - 0.5) * 0.7;
            let oz = z + toward.z * 0.1;
            b.trim.add_cylinder(
                Vector3::new(ox, KERB, oz),
                0.22,
                0.03,
                0.62,
                7,
                Color::from_hex(0xd4551f),
                false,
                Uv::Unit,
            );
        }
    }
}

/// The back-of-house layer: skips, pallets and crates in the gap behind a
/// block, plus the litter that collects around them.
pub(crate) fn add_service_yard(b: &mut Batches, r: Rect, rng: &mut Rng) {
    let cx = rng.range(r.x0 + 1.6, r.x1 - 1.6);
    let cz = rng.range(r.z0 + 1.6, r.z1 - 1.6);
    let along_x = rng.chance(0.5);
    let (hw, hd) = if along_x { (1.05, 0.62) } else { (0.62, 1.05) };

    // Skip. The lid sits open more often than not.
    let hull = scale_color(Color::from_hex(0x6b4a2c), rng.range(0.7, 1.3));
    b.trim.add_box(
        Vector3::new(cx - hw, KERB, cz - hd),
        Vector3::new(cx + hw, KERB + 0.92, cz + hd),
        hull,
        Uv::Unit,
    );
    b.trim.add_box(
        Vector3::new(cx - hw - 0.06, KERB + 0.86, cz - hd - 0.06),
        Vector3::new(cx + hw + 0.06, KERB + 0.98, cz + hd + 0.06),
        scale_color(hull, 0.8),
        Uv::Unit,
    );
    // Whatever is sticking out of it.
    for _ in 0..3 {
        let jx = rng.range(cx - hw * 0.7, cx + hw * 0.7);
        let jz = rng.range(cz - hd * 0.7, cz + hd * 0.7);
        let s = rng.range(0.18, 0.42);
        b.trim.add_box(
            Vector3::new(jx - s, KERB + 0.9, jz - s * 0.7),
            Vector3::new(jx + s, KERB + 0.9 + rng.range(0.12, 0.4), jz + s * 0.7),
            scale_color(Color::from_hex(0x8a8377), rng.range(0.6, 1.3)),
            Uv::Unit,
        );
    }

    // A stack of pallets, and a couple of crates.
    if rng.chance(0.6) {
        let px = rng.range(r.x0 + 1.0, r.x1 - 1.0);
        let pz = rng.range(r.z0 + 1.0, r.z1 - 1.0);
        let pale = Color::from_hex(0xa5865c);
        for i in 0..3 + rng.below(4) {
            let y = KERB + i as f32 * 0.15;
            let j = rng.range(-0.06, 0.06);
            b.trim.add_box(
                Vector3::new(px - 0.55 + j, y, pz - 0.45),
                Vector3::new(px + 0.55 + j, y + 0.11, pz + 0.45),
                scale_color(pale, rng.range(0.85, 1.1)),
                Uv::Unit,
            );
        }
    }
    for _ in 0..rng.below(3) {
        let bx = rng.range(r.x0 + 0.8, r.x1 - 0.8);
        let bz = rng.range(r.z0 + 0.8, r.z1 - 0.8);
        let s = rng.range(0.22, 0.40);
        b.trim.add_box(
            Vector3::new(bx - s, KERB, bz - s),
            Vector3::new(bx + s, KERB + s * rng.range(1.2, 1.8), bz + s),
            scale_color(Color::from_hex(0x9a7d55), rng.range(0.8, 1.2)),
            Uv::Unit,
        );
    }

    // The ground under a skip is never clean.
    add_ground_blob(
        &mut b.paint,
        cx,
        cz,
        KERB + 0.004,
        hw * 2.4,
        hd * 2.4,
        rng,
        Color::from_hex(0x4a4740),
    );

    // Litter, thickest where the bins are.
    for _ in 0..8 + rng.below(10) {
        let lx = cx + rng.range(-2.6, 2.6);
        let lz = cz + rng.range(-2.6, 2.6);
        if lx < r.x0 || lx > r.x1 || lz < r.z0 || lz > r.z1 {
            continue;
        }
        add_ground_quad(
            &mut b.paint,
            lx,
            lz,
            KERB + 0.006,
            rng.range(0.12, 0.34),
            rng.range(0.10, 0.26),
            rng.range(0.0, TAU),
            scale_color(Color::from_hex(0xb9b2a2), rng.range(0.55, 1.15)),
        );
    }
}

/// A traffic signal on a corner: pole, mast arm over the road, dark housing,
/// and a lens in each half of the cycle — green in `phase`, red in the other.
/// Toggling which of the two lens batches is visible runs every junction in
/// the city at once, which is the whole of the signal logic.
pub(crate) fn add_signal(b: &mut Batches, x: f32, z: f32, toward: Vector3, phase: usize) {
    let metal = Color::from_hex(0x3b3e43);
    let h = 5.4;
    // A camera on the pole, looking back down the road the signal faces. A
    // signalised junction is the single most-watched place in any city and the
    // one spot where a camera is never a surprise.
    {
        let mut rng = Rng::new(
            ((x * 7.0) as i64 as u64) ^ ((z * 13.0) as i64 as u64).rotate_left(21) ^ 0x0cc7_0aa1,
        );
        add_cctv(
            b,
            Vector3::new(x + toward.x * 0.22, KERB + h - 0.5, z + toward.z * 0.22),
            toward,
            if rng.chance(0.35) { Cctv::Ptz } else { Cctv::Bullet },
            &mut rng,
        );
    }
    b.trim
        .add_cylinder(Vector3::new(x, KERB, z), 0.15, 0.11, h, 6, metal, false, Uv::Unit);
    let arm = 3.2;
    let ex = x + toward.x * arm;
    let ez = z + toward.z * arm;
    b.trim.add_box(
        Vector3::new(x.min(ex) - 0.08, KERB + h - 0.18, z.min(ez) - 0.08),
        Vector3::new(x.max(ex) + 0.08, KERB + h, z.max(ez) + 0.08),
        metal,
        Uv::Unit,
    );
    let top = KERB + h - 0.18;
    b.trim.add_box(
        Vector3::new(ex - 0.20, top - 1.05, ez - 0.20),
        Vector3::new(ex + 0.20, top, ez + 0.20),
        Color::from_hex(0x24262a),
        Uv::Unit,
    );
    // Red at the top of the housing, green at the bottom, as on the street.
    let lens = |b: &mut Batches, i: usize, y: f32, c: Color| {
        b.signal[i].add_box(
            Vector3::new(ex - 0.22, y - 0.13, ez - 0.22),
            Vector3::new(ex + 0.22, y + 0.13, ez + 0.22),
            c,
            Uv::Unit,
        );
    };
    lens(b, phase, top - 0.88, Color::from_hex(0x2bff62));
    lens(b, 1 - phase, top - 0.20, Color::from_hex(0xff2a18));
}

/// Lane markings and zebra crossings for one road segment.
///
/// `center` is the fixed coordinate (x for a road running along Z, z for one
/// running along X); `s0..s1` is the run, `half` the half-width.
pub(crate) fn paint_road(
    b: &mut Batches,
    along_x: bool,
    center: f32,
    s0: f32,
    s1: f32,
    half: f32,
    avenue: bool,
) {
    let y = 0.015;
    let white = Color::from_hex(C_PAINT);
    let yellow = Color::from_hex(C_PAINT_WARN);
    let mut bar = |sa: f32, sb: f32, ta: f32, tb: f32, c: Color| {
        if along_x {
            b.paint.add_slab(sa, ta, sb, tb, y, c, Uv::Unit);
        } else {
            b.paint.add_slab(ta, sa, tb, sb, y, c, Uv::Unit);
        }
    };

    // Centre line: avenues get a double yellow, streets a white dash.
    if avenue {
        for off in [-0.42, 0.24] {
            bar(s0, s1, center + off, center + off + 0.18, yellow);
        }
    } else {
        let (on, gap) = (2.6, 3.2);
        let mut s = s0 + 1.5;
        while s + on < s1 - 1.5 {
            bar(s, s + on, center - 0.09, center + 0.09, white);
            s += on + gap;
        }
    }

    // Zebra crossing at each end, bars running with the traffic.
    let usable = 2.0 * half - 1.4;
    let bars = 6;
    let w = usable / bars as f32 * 0.55;
    for (end, into) in [(s0, 1.0f32), (s1, -1.0f32)] {
        let (ca, cb) = if into > 0.0 {
            (end + 0.4, end + 3.6)
        } else {
            (end - 3.6, end - 0.4)
        };
        if cb - ca < 0.5 || (s1 - s0) < 12.0 {
            continue;
        }
        for i in 0..bars {
            let t = center - usable * 0.5 + usable * (i as f32 + 0.22) / bars as f32;
            bar(ca, cb, t, t + w, white);
        }
    }
}

/// What a surveillance camera is mounted on, which decides its shape.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Cctv {
    /// Bullet camera on a bracket: the one bolted to a wall or a post.
    Bullet,
    /// Dome: the half-sphere under a canopy, used where the direction it is
    /// pointing is meant to be unreadable.
    Dome,
    /// Pan-tilt-zoom on a short mast, with a sunshield. The one that watches a
    /// junction or a yard.
    Ptz,
}

/// A CCTV camera, mounted at a point and looking along `toward`.
///
/// Small enough that it is a silhouette at any distance and a recognisable
/// object up close, which is the whole job: what makes a modern street read as
/// modern is not the cameras being legible, it is their being *there*, on
/// every corner, at the angle they always sit at — high, and tilted down.
///
/// `toward` need not be normalised and its Y is ignored; the downward tilt is
/// applied here, because a camera pointing along the horizontal is a camera
/// watching the sky.
pub(crate) fn add_cctv(
    b: &mut Batches,
    at: Vector3,
    toward: Vector3,
    kind: Cctv,
    rng: &mut Rng,
) {
    let l = toward.x.hypot(toward.z).max(1e-4);
    let (fx, fz) = (toward.x / l, toward.z / l);
    let body = Color::from_hex(0xd8d4cc);
    let dark = Color::from_hex(0x2a2d31);
    let lens = Color::from_hex(0x11161c);
    match kind {
        Cctv::Bullet => {
            // Bracket back to the wall, then the barrel, tilted down.
            let root = Vector3::new(at.x - fx * 0.42, at.y + 0.12, at.z - fz * 0.42);
            b.trim
                .add_limb(root, at, 0.045, 0.045, 4, dark, false);
            let nose = Vector3::new(at.x + fx * 0.46, at.y - 0.20, at.z + fz * 0.46);
            b.trim.add_limb(at, nose, 0.10, 0.085, 6, body, true);
            // Sunshield over the top of the barrel.
            b.trim.add_limb(
                Vector3::new(at.x - fx * 0.04, at.y + 0.10, at.z - fz * 0.04),
                Vector3::new(nose.x - fx * 0.02, nose.y + 0.12, nose.z - fz * 0.02),
                0.115,
                0.10,
                6,
                scale_color(body, 0.92),
                false,
            );
            // The glass.
            b.trim.add_limb(
                nose,
                Vector3::new(nose.x + fx * 0.05, nose.y - 0.02, nose.z + fz * 0.05),
                0.075,
                0.075,
                6,
                lens,
                true,
            );
        }
        Cctv::Dome => {
            b.trim.add_limb(
                Vector3::new(at.x, at.y + 0.10, at.z),
                Vector3::new(at.x, at.y, at.z),
                0.19,
                0.19,
                8,
                body,
                true,
            );
            b.trim.add_ellipsoid(
                Vector3::new(at.x, at.y, at.z),
                0.17,
                0.13,
                0.17,
                2,
                8,
                lens,
            );
        }
        Cctv::Ptz => {
            // A short mast head: housing on a yoke, pointing down and out.
            b.trim.add_limb(
                Vector3::new(at.x, at.y + 0.30, at.z),
                Vector3::new(at.x, at.y, at.z),
                0.05,
                0.05,
                4,
                dark,
                false,
            );
            let nose = Vector3::new(at.x + fx * 0.34, at.y - 0.16, at.z + fz * 0.34);
            b.trim.add_box(
                Vector3::new(at.x - 0.16, at.y - 0.24, at.z - 0.16),
                Vector3::new(at.x + 0.16, at.y + 0.02, at.z + 0.16),
                body,
                Uv::Unit,
            );
            b.trim.add_limb(
                Vector3::new(at.x, at.y - 0.11, at.z),
                nose,
                0.09,
                0.075,
                6,
                lens,
                true,
            );
        }
    }
    b.cameras += 1;
    let _ = rng;
}

/// What kind of machine collects for the space.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Meter {
    /// A single-space meter on a post: one head, one bay.
    Single,
    /// A pay-and-display machine serving a run of bays, with a solar panel.
    PayDisplay,
}

/// A parking meter on the footway.
///
/// Small, and worth having for the same reason cameras are: what dates a
/// street and tells you the kerb is *managed* rather than just occupied is the
/// row of machines beside it. A kerb full of cars and nothing else reads as a
/// car park with a road through it.
///
/// `toward` points at the carriageway; the machine faces the person standing
/// in front of it, so the display faces away from the road.
pub(crate) fn add_meter(
    b: &mut Batches,
    x: f32,
    z: f32,
    toward: Vector3,
    kind: Meter,
    rng: &mut Rng,
) {
    let l = toward.x.hypot(toward.z).max(1e-4);
    let (fx, fz) = (toward.x / l, toward.z / l);
    // Contrast is the whole job. The first version had a slate body, a
    // near-black face and a navy panel: three dark greys, so the machine read
    // as a solid block — a bin with a lid, not something with a screen and a
    // slot on it. The body is now light enough that the dark face reads *as*
    // a face.
    let post = Color::from_hex(0x3f4348);
    let body = scale_color(Color::from_hex(0x8a949c), rng.range(0.94, 1.06));
    let dark = Color::from_hex(0x14181c);
    let trim_c = Color::from_hex(0xd8d4c4);
    match kind {
        Meter::Single => {
            b.trim.add_limb(
                Vector3::new(x, KERB, z),
                Vector3::new(x, KERB + 1.05, z),
                0.055,
                0.05,
                6,
                post,
                false,
            );
            // A squat head with a domed top and a dark face.
            b.trim.add_box(
                Vector3::new(x - 0.17, KERB + 1.05, z - 0.14),
                Vector3::new(x + 0.17, KERB + 1.46, z + 0.14),
                body,
                Uv::Unit,
            );
            b.trim.add_limb(
                Vector3::new(x, KERB + 1.46, z),
                Vector3::new(x, KERB + 1.56, z),
                0.19,
                0.10,
                7,
                scale_color(body, 0.88),
                false,
            );
            b.trim.add_box(
                Vector3::new(x - fx * 0.15 - 0.12, KERB + 1.14, z - fz * 0.15 - 0.12),
                Vector3::new(x - fx * 0.13 + 0.12, KERB + 1.40, z - fz * 0.13 + 0.12),
                dark,
                Uv::Unit,
            );
        }
        Meter::PayDisplay => {
            // Plinth, cabinet, hood, face, panel — each a different value, so
            // the thing has parts.
            b.trim.add_box(
                Vector3::new(x - 0.30, KERB, z - 0.26),
                Vector3::new(x + 0.30, KERB + 0.14, z + 0.26),
                post,
                Uv::Unit,
            );
            b.trim.add_box(
                Vector3::new(x - 0.27, KERB + 0.14, z - 0.23),
                Vector3::new(x + 0.27, KERB + 1.34, z + 0.23),
                body,
                Uv::Unit,
            );
            // A band of livery round the cabinet: the one bright thing on it,
            // and what makes it legible at ten metres.
            b.trim.add_box(
                Vector3::new(x - 0.285, KERB + 1.02, z - 0.245),
                Vector3::new(x + 0.285, KERB + 1.16, z + 0.245),
                Color::from_hex(0x2f6ab5),
                Uv::Unit,
            );
            // Hood, sloping down towards the user.
            b.trim.add_limb(
                Vector3::new(x - fx * 0.20, KERB + 1.34, z - fz * 0.20),
                Vector3::new(x + fx * 0.12, KERB + 1.56, z + fz * 0.12),
                0.30,
                0.26,
                4,
                scale_color(body, 0.86),
                false,
            );
            // The face: a dark recess with a paler screen and a keypad in it.
            let (ax, az) = (x - fx * 0.235, z - fz * 0.235);
            b.trim.add_box(
                Vector3::new(ax - 0.20, KERB + 0.68, az - 0.20),
                Vector3::new(ax + 0.20, KERB + 1.30, az + 0.20),
                dark,
                Uv::Unit,
            );
            let (sx2, sz2) = (x - fx * 0.26, z - fz * 0.26);
            b.trim.add_box(
                Vector3::new(sx2 - 0.13, KERB + 1.04, sz2 - 0.13),
                Vector3::new(sx2 + 0.13, KERB + 1.24, sz2 + 0.13),
                Color::from_hex(0x9fd8c8),
                Uv::Unit,
            );
            // Coin slot and ticket tray, in white so they read as openings.
            b.trim.add_box(
                Vector3::new(sx2 - 0.10, KERB + 0.92, sz2 - 0.10),
                Vector3::new(sx2 + 0.10, KERB + 0.97, sz2 + 0.10),
                trim_c,
                Uv::Unit,
            );
            b.trim.add_box(
                Vector3::new(sx2 - 0.09, KERB + 0.74, sz2 - 0.09),
                Vector3::new(sx2 + 0.09, KERB + 0.80, sz2 + 0.09),
                trim_c,
                Uv::Unit,
            );
            // Solar panel: framed, tilted, and a blue that is not the body's.
            b.trim.add_box(
                Vector3::new(x - 0.28, KERB + 1.56, z - 0.24),
                Vector3::new(x + 0.28, KERB + 1.62, z + 0.24),
                scale_color(body, 0.8),
                Uv::Unit,
            );
            b.trim.add_box(
                Vector3::new(x - 0.25, KERB + 1.62, z - 0.21),
                Vector3::new(x + 0.25, KERB + 1.66, z + 0.21),
                Color::from_hex(0x24467a),
                Uv::Unit,
            );
        }
    }
    b.occ.take_spot(x, z, 0.4);
}
