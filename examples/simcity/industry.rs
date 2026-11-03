//! Part of the `simcity` example; see `mod.rs`.
//!
//! The infrastructure layer: where the power comes from, where the freight
//! goes, and the one building everybody in the city can name.
//!
//! This is the part of a city-builder that is not housing, not offices and not
//! parkland, and it is the part this generator had none of. A city made only
//! of the places people live and work reads as a model village however good
//! the buildings are — what makes it look like somewhere that *runs* is the
//! ugly half: a power station on the skyline, transmission lines marching
//! across the fields to it, a container terminal on the water with somewhere
//! for the ships to actually go, and a stadium.
//!
//! All of it is deliberately sited *outside* the street grid. Infrastructure
//! is what a city puts where the land is cheap and the neighbours are few, and
//! putting it out in the countryside also means it fills the middle distance —
//! which, now the background is no longer clipped a few hundred metres out, is
//! most of the frame.
#![allow(dead_code)]

use super::*;

/// A drifting plume.
///
/// Stacked blobs that grow, rise and lean downwind, thinning as they go. The
/// deck of cloud uses the same wind bearing, so a plume and the weather agree
/// about which way the air is moving — which nobody consciously notices and
/// everybody notices when they disagree.
///
/// Opaque, like everything else here. Real smoke is not, but this renderer has
/// no alpha-blended pass that plumes could join without sorting artefacts, and
/// an opaque plume with a dark underside reads perfectly well at the range you
/// see one from.
pub(crate) fn add_plume(
    b: &mut Batches,
    x: f32,
    y: f32,
    z: f32,
    r0: f32,
    reach: f32,
    tint: Color,
    rng: &mut Rng,
) {
    let (sa, ca) = (0.55f32).sin_cos();
    // Enough of them that consecutive puffs overlap. At ten across a
    // hundred-metre rise they did not, and the result was a row of separate
    // lumps climbing the sky like beads on a string — `add_blob` at two rings
    // is a bipyramid, so they read as diamonds rather than even as smoke.
    // Spacing has to come out *below* the puff radius or the union of them is
    // a string of beads rather than a column. At fourteen over a hundred-metre
    // rise it was six metres between puffs of radius six and a half, which is
    // only just touching; at twenty-four it is four, and the lumps read as the
    // edge of one plume instead of as separate balloons.
    let puffs = 24 + rng.below(6);
    for i in 0..puffs {
        let t = i as f32 / (puffs - 1).max(1) as f32;
        // Rises fast at first, then the wind wins and it flattens out.
        //
        // The proportions matter more than the shape: a plume is *tall and
        // narrow*, and the first version grew four-fold over a short rise, so
        // two cooling towers produced one white mass wider than they were high
        // and the towers disappeared into it. Buoyant gas goes up far further
        // than it spreads.
        // A steady rise rather than a fast one: the eased curve bunched the
        // puffs at the top and left gaps at the bottom, which is the opposite
        // of how a plume thins.
        let up = reach * t.powf(0.82) * 0.92;
        let along = reach * t * t * 0.55;
        // Grows as it entrains air, and goes from dirty to pale as it cools.
        let r = r0 * (1.0 + t * 1.25) * rng.range(0.85, 1.15);
        b.plume.add_blob(
            Vector3::new(
                x + along * ca + rng.range(-r * 0.3, r * 0.3),
                y + up + rng.range(-r * 0.2, r * 0.2),
                z + along * sa + rng.range(-r * 0.3, r * 0.3),
            ),
            r,
            r * rng.range(0.8, 1.1),
            r * rng.range(0.8, 1.2),
            3,
            9,
            0.20,
            rng.next_u32() as i32 & 0xffff,
            // Smoke has no dark underside. `add_blob`'s `floor` exists for
            // foliage, where the bottom of a crown genuinely is in shadow; at
            // 0.52 every puff got a dark rim and a plume read as a stack of
            // dinner plates rather than as a column of vapour.
            0.86,
            mix_color(tint, Color::from_hex(0xe8ecef), t * 0.8),
        );
    }
}

/// A lattice transmission pylon, and the conductors leaving it.
///
/// The lattice is four legs that taper to a waist and two crossarms — enough
/// to read as a pylon in silhouette, which is the only way anybody sees one.
fn pylon(b: &mut Batches, x: f32, z: f32, h: f32, along: Vector3, steel: Color) {
    let base = h * 0.13;
    let waist = h * 0.045;
    // Legs. Splayed at the ground, gathered at the waist, vertical above it.
    for (sx, sz) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        b.trim.add_limb(
            Vector3::new(x + sx * base, 0.0, z + sz * base),
            Vector3::new(x + sx * waist, h * 0.55, z + sz * waist),
            0.30,
            0.20,
            4,
            steel,
            false,
        );
        b.trim.add_limb(
            Vector3::new(x + sx * waist, h * 0.55, z + sz * waist),
            Vector3::new(x + sx * waist * 0.7, h, z + sz * waist * 0.7),
            0.20,
            0.16,
            4,
            steel,
            false,
        );
    }
    // Bracing: one X per panel is all the eye needs to call it a lattice.
    for k in 0..4 {
        let t0 = 0.12 + k as f32 * 0.2;
        let t1 = t0 + 0.18;
        let r0 = mix(base, waist, (t0 / 0.55).min(1.0));
        let r1 = mix(base, waist, (t1 / 0.55).min(1.0));
        for (ax, az, bx, bz) in [(-1.0f32, -1.0f32, 1.0f32, -1.0f32), (1.0, 1.0, -1.0, 1.0)] {
            b.trim.add_limb(
                Vector3::new(x + ax * r0, h * t0, z + az * r0),
                Vector3::new(x + bx * r1, h * t1, z + bz * r1),
                0.10,
                0.10,
                3,
                steel,
                false,
            );
            b.trim.add_limb(
                Vector3::new(x + bx * r0, h * t0, z + bz * r0),
                Vector3::new(x + ax * r1, h * t1, z + az * r1),
                0.10,
                0.10,
                3,
                steel,
                false,
            );
        }
    }
    // Crossarms, square to the run of the line.
    let side = Vector3::new(-along.z, 0.0, along.x);
    for (ty, arm) in [(0.72f32, h * 0.30), (0.93, h * 0.22)] {
        b.trim.add_limb(
            Vector3::new(x - side.x * arm, h * ty, z - side.z * arm),
            Vector3::new(x + side.x * arm, h * ty, z + side.z * arm),
            0.16,
            0.16,
            4,
            steel,
            false,
        );
    }
}

/// A run of pylons carrying conductors, from one point to another.
///
/// The conductors sag. That is the whole reason to draw them as several
/// segments rather than one straight limb: a taut horizontal wire between two
/// towers looks like scaffolding, and the catenary is the line everybody
/// recognises even when the pylons themselves are a few pixels.
pub(crate) fn add_power_line(
    b: &mut Batches,
    from: (f32, f32),
    to: (f32, f32),
    span: f32,
    rng: &mut Rng,
) -> usize {
    let (dx, dz) = (to.0 - from.0, to.1 - from.1);
    let len = dx.hypot(dz);
    if len < span * 1.5 {
        return 0;
    }
    let along = Vector3::new(dx / len, 0.0, dz / len);
    let side = Vector3::new(-along.z, 0.0, along.x);
    let n = (len / span).round().max(2.0) as usize;
    let steel = Color::from_hex(0x8d9196);
    let wire = Color::from_hex(0x3a3d42);
    let h = 34.0;
    let mut towers = 0;
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let (x, z) = (from.0 + dx * t, from.1 + dz * t);
        // A pylon needs its footprint clear, and the run crosses whatever the
        // countryside put down first. Skip the ones that land on something
        // rather than moving them: a line of pylons with a gap in it still
        // reads as a line, and one that jinks sideways does not.
        if !b.occ.try_spot(x, z, 7.0) {
            continue;
        }
        pylon(b, x, z, h, along, steel);
        towers += 1;
        if i == n {
            continue;
        }
        // Six conductors and an earth wire, sagging between this tower and the
        // next.
        let (nx, nz) = (from.0 + dx * (t + 1.0 / n as f32), from.1 + dz * (t + 1.0 / n as f32));
        for (ty, arm, sag) in [
            (0.72f32, h * 0.30, 2.6f32),
            (0.72, h * 0.18, 2.6),
            (0.93, h * 0.22, 2.2),
        ] {
            for s in [-1.0f32, 1.0] {
                let y = h * ty;
                let segs = 4;
                for k in 0..segs {
                    let (u0, u1) = (k as f32 / segs as f32, (k + 1) as f32 / segs as f32);
                    // 4u(1-u) peaks at 1 in the middle and is 0 at both ends,
                    // which is the sag profile to within what anybody can see.
                    let d0 = sag * 4.0 * u0 * (1.0 - u0);
                    let d1 = sag * 4.0 * u1 * (1.0 - u1);
                    b.trim.add_limb(
                        Vector3::new(
                            mix(x, nx, u0) + side.x * arm * s,
                            y - d0,
                            mix(z, nz, u0) + side.z * arm * s,
                        ),
                        Vector3::new(
                            mix(x, nx, u1) + side.x * arm * s,
                            y - d1,
                            mix(z, nz, u1) + side.z * arm * s,
                        ),
                        0.09,
                        0.09,
                        3,
                        wire,
                        false,
                    );
                }
            }
        }
        let _ = rng;
    }
    towers
}

/// The power station the lines run to.
///
/// Cooling towers, a turbine hall, chimneys and a switchyard. The towers are
/// the thing: a hyperboloid is the one industrial shape nobody mistakes for
/// anything else, and it is cheap — a stack of rings whose radius pinches in
/// at the waist and flares at the lip.
pub(crate) fn add_power_station(b: &mut Batches, cx: f32, cz: f32, rng: &mut Rng) -> bool {
    let r = Rect {
        x0: cx - 78.0,
        z0: cz - 62.0,
        x1: cx + 78.0,
        z1: cz + 62.0,
    };
    if !b.occ.try_claim([r.x0, r.z0, r.x1, r.z1]) {
        return false;
    }
    b.pads.add_slab(
        r.x0,
        r.z0,
        r.x1,
        r.z1,
        0.10,
        scale_color(Color::from_hex(0x8b8880), rng.range(0.92, 1.08)),
        Uv::Unit,
    );
    let concrete = Color::from_hex(0xa8a49c);

    // --- Cooling towers.
    for i in 0..2 {
        let tx = r.x0 + 34.0 + i as f32 * 52.0;
        let tz = r.z0 + 40.0;
        let h = rng.range(52.0, 62.0);
        let foot = 19.0;
        let rings = 9;
        let radius = |t: f32| {
            // Waist at 70% of the way up, then a slight flare to the lip.
            let waist = 0.62 + 0.38 * (1.0 - (t / 0.75).min(1.0)).powi(2);
            let flare = if t > 0.75 { (t - 0.75) * 0.5 } else { 0.0 };
            foot * (waist + flare)
        };
        for k in 0..rings {
            let (t0, t1) = (k as f32 / rings as f32, (k + 1) as f32 / rings as f32);
            b.trim.add_limb(
                Vector3::new(tx, h * t0, tz),
                Vector3::new(tx, h * t1, tz),
                radius(t0),
                radius(t1),
                14,
                scale_color(concrete, 1.0 - t0 * 0.12),
                false,
            );
        }
        // The legs a real one stands on, and the shadow gap under it.
        for k in 0..12 {
            let a = k as f32 / 12.0 * TAU;
            b.trim.add_limb(
                Vector3::new(tx + foot * 1.02 * a.cos(), 0.0, tz + foot * 1.02 * a.sin()),
                Vector3::new(tx + foot * a.cos(), 5.0, tz + foot * a.sin()),
                0.7,
                0.7,
                4,
                scale_color(concrete, 0.8),
                false,
            );
        }
        add_plume(b, tx, h, tz, foot * 0.34, h * 1.15, Color::from_hex(0xeef1f4), rng);
    }

    // --- Turbine hall: one long shed, because that is what they are.
    let hall = Rect {
        x0: r.x0 + 12.0,
        z0: r.z1 - 46.0,
        x1: r.x1 - 22.0,
        z1: r.z1 - 14.0,
    };
    b.trim.add_box(
        Vector3::new(hall.x0, 0.0, hall.z0),
        Vector3::new(hall.x1, 20.0, hall.z1),
        Color::from_hex(0x9aa0a6),
        Uv::Unit,
    );
    b.trim.add_gable(
        Vector3::new(hall.x0 - 0.6, 20.0, hall.z0 - 0.6),
        Vector3::new(hall.x1 + 0.6, 20.0, hall.z1 + 0.6),
        4.0,
        Color::from_hex(0x6d7378),
    );
    // --- Chimneys, banded, with the plume that says it is running.
    for i in 0..2 {
        let sx = hall.x1 + 8.0;
        let sz = mix(hall.z0 + 6.0, hall.z1 - 6.0, i as f32);
        let h = rng.range(74.0, 92.0);
        let bands = 7;
        for k in 0..bands {
            let (t0, t1) = (k as f32 / bands as f32, (k + 1) as f32 / bands as f32);
            b.trim.add_limb(
                Vector3::new(sx, h * t0, sz),
                Vector3::new(sx, h * t1, sz),
                mix(4.2, 2.6, t0),
                mix(4.2, 2.6, t1),
                10,
                if k % 2 == 0 {
                    Color::from_hex(0xc9c4ba)
                } else {
                    Color::from_hex(0xa8453a)
                },
                false,
            );
        }
        add_plume(b, sx, h, sz, 2.4, h * 0.72, Color::from_hex(0x8f9498), rng);
    }
    // --- Switchyard: the gantries and transformers the lines actually leave
    // from. Without it the pylons appear to start in a field.
    for i in 0..5 {
        let gx = r.x0 + 10.0 + i as f32 * 12.0;
        for s in [-1.0f32, 1.0] {
            b.trim.add_limb(
                Vector3::new(gx, 0.0, r.z0 + 12.0 + s * 7.0),
                Vector3::new(gx, 11.0, r.z0 + 12.0 + s * 7.0),
                0.45,
                0.35,
                5,
                Color::from_hex(0x8d9196),
                false,
            );
        }
        b.trim.add_limb(
            Vector3::new(gx, 11.0, r.z0 + 5.0),
            Vector3::new(gx, 11.0, r.z0 + 19.0),
            0.30,
            0.30,
            4,
            Color::from_hex(0x8d9196),
            false,
        );
        b.trim.add_box(
            Vector3::new(gx - 2.2, 0.0, r.z0 + 22.0),
            Vector3::new(gx + 2.2, 4.4, r.z0 + 27.0),
            Color::from_hex(0x6b7075),
            Uv::Unit,
        );
    }
    true
}

/// A container terminal.
///
/// The city already had container ships and nowhere for them to go, which is
/// the sort of thing that reads as wrong long before anybody works out why.
/// Gantry cranes over a quay, stacks of boxes behind them, and sheds behind
/// those.
pub(crate) fn add_port(
    b: &mut Batches,
    along_x: bool,
    quay: f32,
    lo: f32,
    hi: f32,
    inland: f32,
    rng: &mut Rng,
) -> bool {
    // Deep enough for the three bands a terminal actually has: a quayside
    // strip the cranes straddle, the stacking yard, and sheds at the back. At
    // 74 m the yard came out one row of boxes wide, which reads as a lorry
    // park.
    let depth = 108.0;
    // `inland` is +1 or -1: which side of the quay line the land is on.
    let (d0, d1) = if inland > 0.0 {
        (quay, quay + depth)
    } else {
        (quay - depth, quay)
    };
    let r = if along_x {
        Rect { x0: lo, z0: d0, x1: hi, z1: d1 }
    } else {
        Rect { x0: d0, z0: lo, x1: d1, z1: hi }
    };
    if !b.occ.try_claim([r.x0, r.z0, r.x1, r.z1]) {
        return false;
    }
    b.pads.add_slab(
        r.x0,
        r.z0,
        r.x1,
        r.z1,
        KERB,
        scale_color(Color::from_hex(0x6f6d69), rng.range(0.94, 1.06)),
        Uv::Unit,
    );

    // Position along the quay, and distance in from the water, in world terms.
    let at = |u: f32, v: f32| -> (f32, f32) {
        let s = mix(lo, hi, u);
        let t = quay + inland * v;
        if along_x {
            (s, t)
        } else {
            (t, s)
        }
    };
    let run = hi - lo;

    // --- Gantry cranes. Legs to the waterside, a boom out over the ship, and
    // a counterweight boom inland. The boom is what makes the silhouette.
    let cranes = 2 + rng.below(3);
    for i in 0..cranes {
        let u = (i as f32 + 0.6) / (cranes as f32 + 0.2);
        let h = rng.range(38.0, 48.0);
        let leg = 11.0;
        let steel = scale_color(
            Color::from_hex(if rng.chance(0.5) { 0xc25a2e } else { 0x3f6da8 }),
            rng.range(0.9, 1.1),
        );
        for du in [-1.0f32, 1.0] {
            for dv in [6.0f32, 6.0 + leg * 2.0] {
                let (px, pz) = at(u + du * (7.0 / run), dv);
                b.trim.add_limb(
                    Vector3::new(px, KERB, pz),
                    Vector3::new(px, h, pz),
                    0.9,
                    0.7,
                    5,
                    steel,
                    false,
                );
            }
        }
        // The boom: out over the water and back over the yard.
        let (ax, az) = at(u, -26.0);
        let (bx, bz) = at(u, 46.0);
        for du in [-1.0f32, 1.0] {
            let off = du * 7.0 / run;
            let (a2x, a2z) = at(u + off, -26.0);
            let (b2x, b2z) = at(u + off, 46.0);
            b.trim.add_limb(
                Vector3::new(a2x, h, a2z),
                Vector3::new(b2x, h, b2z),
                0.8,
                0.8,
                4,
                steel,
                false,
            );
        }
        // Machinery house and the trolley hanging off the boom.
        let (mx, mz) = at(u, 30.0);
        b.trim.add_box(
            Vector3::new(mx - 5.0, h - 1.0, mz - 5.0),
            Vector3::new(mx + 5.0, h + 5.0, mz + 5.0),
            scale_color(steel, 0.85),
            Uv::Unit,
        );
        let ht = rng.f();
        let (tx, tz) = (mix(ax, bx, ht * 0.35), mix(az, bz, ht * 0.35));
        b.trim.add_box(
            Vector3::new(tx - 2.4, h - 3.2, tz - 2.4),
            Vector3::new(tx + 2.4, h - 0.6, tz + 2.4),
            Color::from_hex(0x4a4d52),
            Uv::Unit,
        );
    }

    // --- Container stacks. Boxes are 12 m by 2.4 m and they are stacked in
    // blocks with lanes between: the regularity is the point, and the colours
    // are what makes a yard read from the air.
    const BOX: [u32; 7] = [
        0xb5432f, 0x2f6ab5, 0x3f8f5c, 0xd0a13a, 0x8f4a8a, 0x9aa0a6, 0xc25a2e,
    ];
    // The yard occupies the middle of the apron: quayside lanes in front of it
    // for the cranes to straddle, sheds behind. These two used to overlap —
    // the stacks ran to 62 m inland and the sheds started at 48 — so half the
    // boxes were inside a warehouse.
    let (yard0, yard1) = (30.0f32, depth - 34.0);
    let rows = ((yard1 - yard0) / 14.0).max(1.0) as usize;
    for row in 0..rows {
        let v = yard0 + row as f32 * 14.0;
        if v > yard1 {
            break;
        }
        let per = (run / 12.6) as usize;
        for k in 0..per {
            if rng.chance(0.22) {
                continue;
            }
            let u0 = (k as f32 * 12.6 + 1.0) / run;
            let u1 = u0 + 12.0 / run;
            let high = 1 + rng.below(4);
            for s in 0..high {
                let (x0, z0) = at(u0, v - 1.2);
                let (x1, z1) = at(u1, v + 1.2);
                b.trim.add_box(
                    Vector3::new(x0.min(x1), KERB + s as f32 * 2.6, z0.min(z1)),
                    Vector3::new(x0.max(x1), KERB + s as f32 * 2.6 + 2.5, z0.max(z1)),
                    scale_color(Color::from_hex(BOX[rng.below(BOX.len())]), rng.range(0.85, 1.15)),
                    Uv::Unit,
                );
            }
        }
    }

    // --- Yard masts. A container terminal works around the clock and is lit
    // like it: without these the whole apron goes black at dusk while the city
    // beside it is at its best.
    for i in 0..4 {
        let (mx, mz) = at((i as f32 + 0.5) / 4.0, depth * 0.55);
        b.trim.add_limb(
            Vector3::new(mx, KERB, mz),
            Vector3::new(mx, 26.0, mz),
            0.5,
            0.35,
            5,
            Color::from_hex(0x7d8288),
            false,
        );
        // Housing on `trim`, lens on `glow`. `neon` is drawn at every hour and
        // merely scaled by `sky.lights`, so a floodlight head parked on it was
        // a black slab against the sky all day.
        b.trim.add_box(
            Vector3::new(mx - 2.0, 25.0, mz - 0.8),
            Vector3::new(mx + 2.0, 27.2, mz + 0.8),
            Color::from_hex(0x8e9298),
            Uv::Unit,
        );
        b.glow.add_box(
            Vector3::new(mx - 1.9, 25.1, mz - 0.86),
            Vector3::new(mx + 1.9, 27.1, mz + 0.86),
            Color::from_hex(0xffeec4),
            Uv::Unit,
        );
        // Ports are watched harder than anywhere else in a city, for reasons
        // that have nothing to do with crime and everything to do with
        // customs.
        add_cctv(
            b,
            Vector3::new(mx, 23.2, mz),
            Vector3::new(-inland, 0.0, 0.0),
            Cctv::Ptz,
            rng,
        );
    }

    // --- Sheds at the back.
    for i in 0..3 {
        let u0 = (i as f32 * 0.33) + 0.03;
        let u1 = u0 + 0.26;
        let (x0, z0) = at(u0, depth - 3.0);
        let (x1, z1) = at(u1, depth - 22.0);
        let (mnx, mxx) = (x0.min(x1), x0.max(x1));
        let (mnz, mxz) = (z0.min(z1), z0.max(z1));
        if mxx - mnx < 6.0 || mxz - mnz < 6.0 {
            continue;
        }
        let wall = scale_color(Color::from_hex(0x9a9186), rng.range(0.88, 1.12));
        b.trim.add_box(
            Vector3::new(mnx, KERB, mnz),
            Vector3::new(mxx, 11.0, mxz),
            wall,
            Uv::Unit,
        );
        b.trim.add_gable(
            Vector3::new(mnx - 0.5, 11.0, mnz - 0.5),
            Vector3::new(mxx + 0.5, 11.0, mxz + 0.5),
            2.4,
            Color::from_hex(0x5f6469),
        );
    }
    true
}

/// The stadium.
///
/// One building the whole city can point to, and the only large enclosed
/// volume in the model that is not a box. A ring of raked stands round an oval
/// of grass, a lip of roof over the seats, and floodlight masts at the
/// corners — which at night are the brightest thing outside the centre.
pub(crate) fn add_stadium(b: &mut Batches, cx: f32, cz: f32, rot: f32, rng: &mut Rng) -> bool {
    let (rx, rz) = (54.0f32, 42.0f32);
    if !b.occ.try_claim([cx - rx - 8.0, cz - rz - 8.0, cx + rx + 8.0, cz + rz + 8.0]) {
        return false;
    }
    let (sa, ca) = rot.sin_cos();
    let at = |u: f32, r: f32| -> (f32, f32) {
        let (lx, lz) = (rx * r * u.cos(), rz * r * u.sin());
        (cx + lx * ca - lz * sa, cz + lx * sa + lz * ca)
    };
    b.pads.add_slab(
        cx - rx - 8.0,
        cz - rz - 8.0,
        cx + rx + 8.0,
        cz + rz + 8.0,
        KERB,
        Color::from_hex(0x8e8b84),
        Uv::Unit,
    );
    let segs = 28;
    let concrete = Color::from_hex(0xb6b2aa);
    let seats = Color::from_hex(0x2f5c8f);
    for i in 0..segs {
        let (u0, u1) = (
            i as f32 / segs as f32 * TAU,
            (i + 1) as f32 / segs as f32 * TAU,
        );
        let (ix0, iz0) = at(u0, 0.62);
        let (ix1, iz1) = at(u1, 0.62);
        let (ox0, oz0) = at(u0, 1.0);
        let (ox1, oz1) = at(u1, 1.0);
        // The rake: inner edge low at the pitch, outer edge high at the back.
        b.trim.quad(
            [
                [ix0, KERB + 1.0, iz0],
                [ix1, KERB + 1.0, iz1],
                [ox1, KERB + 15.0, oz1],
                [ox0, KERB + 15.0, oz0],
            ],
            [0.0, 0.85, 0.0],
            Uv::Unit,
            scale_color(seats, rng.range(0.9, 1.1)),
        );
        // Outer wall.
        b.trim.quad(
            [
                [ox0, KERB, oz0],
                [ox0, KERB + 17.0, oz0],
                [ox1, KERB + 17.0, oz1],
                [ox1, KERB, oz1],
            ],
            [
                (ox0 + ox1) * 0.5 - cx,
                0.0,
                (oz0 + oz1) * 0.5 - cz,
            ],
            Uv::Unit,
            scale_color(concrete, rng.range(0.94, 1.06)),
        );
        // Roof lip, cantilevered in over the back rows.
        let (rx0, rz0) = at(u0, 0.80);
        let (rx1, rz1) = at(u1, 0.80);
        b.trim.quad(
            [
                [ox0, KERB + 17.0, oz0],
                [rx0, KERB + 19.0, rz0],
                [rx1, KERB + 19.0, rz1],
                [ox1, KERB + 17.0, oz1],
            ],
            [0.0, 1.0, 0.0],
            Uv::Unit,
            Color::from_hex(0x6f747a),
        );
    }
    // The lit pitch. On `glow` rather than `neon` — this is a wash of light on
    // grass, not a lamp, and it wants to be dim enough that the floodlights
    // above it are still obviously the source.
    let (gx, gz) = (rx * 0.66, rz * 0.66);
    b.glow.add_ground_disc(
        cx,
        cz,
        gx.min(gz),
        24,
        0.0,
        KERB + 0.5,
        Color::from_hex(0x54804a),
    );
    // The pitch, with a centre circle so it is obviously a pitch.
    let (px, pz) = (rx * 0.60, rz * 0.60);
    b.grass.add_ground_disc(cx, cz, px.min(pz), 24, 0.0, KERB + 0.6, Color::from_hex(0x2f7a34));
    b.paint
        .add_ground_disc(cx, cz, px.min(pz) * 0.18, 16, 0.0, KERB + 0.7, Color::WHITE);
    // Floodlight masts.
    for k in 0..4 {
        let u = (k as f32 + 0.5) / 4.0 * TAU;
        let (mx, mz) = at(u, 1.10);
        // Which way is the pitch. The lamps face it, and the housing sits
        // behind them — the first version put the lit box *inside* the housing
        // box, where it was perfectly correct and completely invisible.
        let (ix, iz) = (cx - mx, cz - mz);
        let il = ix.hypot(iz).max(0.001);
        let (ix, iz) = (ix / il, iz / il);
        let (px, pz) = (-iz, ix);
        b.trim.add_limb(
            Vector3::new(mx, KERB, mz),
            Vector3::new(mx, 40.0, mz),
            0.8,
            0.5,
            6,
            Color::from_hex(0x7d8288),
            false,
        );
        // Housing: a bar across the mast, set back from the lamps.
        let (hx, hz) = (mx - ix * 0.9, mz - iz * 0.9);
        b.trim.quad(
            [
                [hx - px * 4.2, 38.0, hz - pz * 4.2],
                [hx + px * 4.2, 38.0, hz + pz * 4.2],
                [hx + px * 4.2, 43.4, hz + pz * 4.2],
                [hx - px * 4.2, 43.4, hz - pz * 4.2],
            ],
            [-ix, 0.0, -iz],
            Uv::Unit,
            Color::from_hex(0x53585e),
        );
        // The lit face, proud of the housing and pointing at the grass.
        let (lx, lz) = (mx + ix * 0.5, mz + iz * 0.5);
        // A pale lamp face by day, an emissive one at night. On `neon` this
        // large panel read as a black rectangle every daylight hour.
        b.trim.quad(
            [
                [lx + px * 3.8, 38.6, lz + pz * 3.8],
                [lx - px * 3.8, 38.6, lz - pz * 3.8],
                [lx - px * 3.8, 42.8, lz - pz * 3.8],
                [lx + px * 3.8, 42.8, lz + pz * 3.8],
            ],
            [ix, 0.0, iz],
            Uv::Unit,
            Color::from_hex(0xdcd8cc),
        );
        b.glow.quad(
            [
                [lx + px * 3.7 + ix * 0.06, 38.7, lz + pz * 3.7 + iz * 0.06],
                [lx - px * 3.7 + ix * 0.06, 38.7, lz - pz * 3.7 + iz * 0.06],
                [lx - px * 3.7 + ix * 0.06, 42.7, lz - pz * 3.7 + iz * 0.06],
                [lx + px * 3.7 + ix * 0.06, 42.7, lz + pz * 3.7 + iz * 0.06],
            ],
            [ix, 0.0, iz],
            Uv::Unit,
            Color::from_hex(0xfff4d2),
        );
    }
    true
}

/// A distribution depot: the building the lorries on the motorway are going to.
///
/// The identifying feature is not the shed, which is just a shed. It is the
/// **dock face** — one long elevation that is nothing but doors, with trailers
/// backed onto it at a slight angle and a yard deep enough to swing a
/// thirteen-metre trailer round in. Every one of these ever built has that,
/// and nothing else in a city does.
pub(crate) fn add_depot(
    b: &mut Batches,
    r: &Rect,
    along_x: bool,
    bodies: &[MeshBuilder],
    rng: &mut Rng,
) -> bool {
    if r.w() < 86.0 || r.d() < 60.0 {
        return false;
    }
    if !b.occ.try_claim([r.x0, r.z0, r.x1, r.z1]) {
        return false;
    }
    // Hardstanding over the lot: this is a place made of concrete.
    b.pads.add_slab(
        r.x0,
        r.z0,
        r.x1,
        r.z1,
        KERB,
        scale_color(Color::from_hex(0x8b8880), rng.range(0.94, 1.06)),
        Uv::Unit,
    );
    // The shed takes the back third; the rest is the yard the trailers use.
    let depth = (if along_x { r.d() } else { r.w() }) * 0.34;
    let shed = if along_x {
        Rect { x0: r.x0 + 4.0, z0: r.z1 - depth, x1: r.x1 - 4.0, z1: r.z1 - 4.0 }
    } else {
        Rect { x0: r.x1 - depth, z0: r.z0 + 4.0, x1: r.x1 - 4.0, z1: r.z1 - 4.0 }
    };
    let h = 12.0;
    let wall = scale_color(Color::from_hex(0xb9bcc0), rng.range(0.94, 1.06));
    b.trim.add_box(
        Vector3::new(shed.x0, KERB, shed.z0),
        Vector3::new(shed.x1, h, shed.z1),
        wall,
        Uv::Unit,
    );
    // A band of company colour along the top, and roof plant.
    b.trim.add_box(
        Vector3::new(shed.x0 - 0.2, h, shed.z0 - 0.2),
        Vector3::new(shed.x1 + 0.2, h + 1.4, shed.z1 + 0.2),
        scale_color(
            Color::from_hex([0x2f6ab5u32, 0xd04a3a, 0x3f8f5c, 0xd8a52e][rng.below(4)]),
            rng.range(0.9, 1.1),
        ),
        Uv::Unit,
    );
    // The dock face. Doors at a fixed pitch along the yard elevation, each
    // with a shallow canopy and a rubber buffer.
    let run = if along_x { shed.w() } else { shed.d() };
    let docks = ((run / 4.6) as usize).max(4);
    let mut trailers = 0;
    for i in 0..docks {
        let t = (i as f32 + 0.5) / docks as f32;
        let (dx, dz) = if along_x {
            (mix(shed.x0, shed.x1, t), shed.z0)
        } else {
            (shed.x0, mix(shed.z0, shed.z1, t))
        };
        let (hw, hd) = if along_x { (1.7f32, 0.25f32) } else { (0.25, 1.7) };
        // Door.
        b.trim.add_box(
            Vector3::new(dx - hw, KERB, dz - hd - 0.1),
            Vector3::new(dx + hw, KERB + 4.2, dz + hd),
            Color::from_hex(0x3a3f45),
            Uv::Unit,
        );
        // Canopy over it.
        b.trim.add_box(
            Vector3::new(dx - hw - 0.4, KERB + 4.2, dz - hd - 1.1),
            Vector3::new(dx + hw + 0.4, KERB + 4.7, dz + hd),
            scale_color(wall, 0.85),
            Uv::Unit,
        );
        // A trailer backed onto two docks in three, sitting on its legs with
        // no tractor unit — which is what a dock face mostly looks like.
        if rng.chance(0.66) {
            let (tl, tw) = (13.6f32, 2.5f32);
            let (bx0, bz0, bx1, bz1) = if along_x {
                (dx - tw * 0.5, dz - tl, dx + tw * 0.5, dz - 0.3)
            } else {
                (dx - tl, dz - tw * 0.5, dx - 0.3, dz + tw * 0.5)
            };
            b.trim.add_box(
                Vector3::new(bx0, KERB + 1.15, bz0),
                Vector3::new(bx1, KERB + 4.25, bz1),
                scale_color(
                    Color::from_hex([0xe4e4e0u32, 0xd8dce0, 0xc8ccd0, 0xdcd4c4][rng.below(4)]),
                    rng.range(0.9, 1.1),
                ),
                Uv::Unit,
            );
            // Bogie and landing legs.
            for (wx, wz) in [
                ((bx0 + bx1) * 0.5, mix(bz0, bz1, if along_x { 0.18 } else { 0.5 })),
                ((bx0 + bx1) * 0.5, mix(bz0, bz1, if along_x { 0.30 } else { 0.5 })),
            ] {
                b.trim.add_box(
                    Vector3::new(wx - 1.1, KERB, wz - 0.4),
                    Vector3::new(wx + 1.1, KERB + 1.15, wz + 0.4),
                    Color::from_hex(0x2f3338),
                    Uv::Unit,
                );
            }
            trailers += 1;
        }
    }
    let _ = trailers;
    // Lorry park along the front: tractor units nose-in, waiting.
    let front = if along_x {
        Rect { x0: r.x0 + 6.0, z0: r.z0 + 4.0, x1: r.x1 - 6.0, z1: r.z0 + 16.0 }
    } else {
        Rect { x0: r.x0 + 4.0, z0: r.z0 + 6.0, x1: r.x0 + 16.0, z1: r.z1 - 6.0 }
    };
    let bays = ((if along_x { front.w() } else { front.d() } / 4.2) as usize).max(2);
    for i in 0..bays {
        if rng.chance(0.45) {
            continue;
        }
        let t = (i as f32 + 0.5) / bays as f32;
        let (px, pz) = if along_x {
            (mix(front.x0, front.x1, t), front.cz())
        } else {
            (front.cx(), mix(front.z0, front.z1, t))
        };
        if bodies.is_empty() {
            continue;
        }
        b.parked.append_at(
            &bodies[rng.below(bodies.len())],
            Vector3::new(px, KERB, pz),
            if along_x { 0.0 } else { PI * 0.5 },
            VEHICLE_SCALE,
            Color::from_hex([0xd8dce0u32, 0x2f6ab5, 0xd04a3a, 0x3f8f5c][rng.below(4)]),
        );
    }
    // Gatehouse and barrier at the entrance, and a mast or two over the yard.
    let (gx, gz) = if along_x { (r.x0 + 8.0, r.z0 + 3.0) } else { (r.x0 + 3.0, r.z0 + 8.0) };
    b.trim.add_box(
        Vector3::new(gx - 2.0, KERB, gz - 1.6),
        Vector3::new(gx + 2.0, KERB + 3.0, gz + 1.6),
        Color::from_hex(0xd8d4c8),
        Uv::Unit,
    );
    for i in 0..2 {
        let t = (i as f32 + 0.5) / 2.0;
        let (mx, mz) = (mix(r.x0 + 10.0, r.x1 - 10.0, t), mix(r.z0 + 10.0, r.z1 - 10.0, 0.4));
        b.trim.add_limb(
            Vector3::new(mx, KERB, mz),
            Vector3::new(mx, 22.0, mz),
            0.4,
            0.28,
            5,
            Color::from_hex(0x7d8288),
            false,
        );
        // Same split as the port masts: a grey housing that reads by day, a
        // lens that only exists after dark.
        b.trim.add_box(
            Vector3::new(mx - 1.8, 21.2, mz - 0.7),
            Vector3::new(mx + 1.8, 23.0, mz + 0.7),
            Color::from_hex(0x8e9298),
            Uv::Unit,
        );
        b.glow.add_box(
            Vector3::new(mx - 1.7, 21.3, mz - 0.76),
            Vector3::new(mx + 1.7, 22.9, mz + 0.76),
            Color::from_hex(0xffeec4),
            Uv::Unit,
        );
        // A camera on each yard mast, watching the dock face and the gate.
        add_cctv(
            b,
            Vector3::new(mx, 19.4, mz),
            Vector3::new(0.0, 0.0, -1.0),
            Cctv::Ptz,
            rng,
        );
    }
    // And one on the gatehouse, which is the point every vehicle passes.
    add_cctv(
        b,
        Vector3::new(gx, KERB + 3.2, gz),
        Vector3::new(0.0, 0.0, -1.0),
        Cctv::Ptz,
        rng,
    );
    true
}

/// A rail freight yard: sidings, wagons and a container gantry.
///
/// This is the other end of the same chain. The port lifts boxes off ships,
/// the motorway carries them by road, and this is where they go by rail —
/// without it the container terminal is a place things arrive at and never
/// leave.
pub(crate) fn add_rail_yard(
    b: &mut Batches,
    along_x: bool,
    fixed: f32,
    lo: f32,
    hi: f32,
    side: f32,
    rng: &mut Rng,
) -> bool {
    let sidings = 5;
    let pitch = 5.2f32;
    let depth = sidings as f32 * pitch + 16.0;
    let (d0, d1) = if side > 0.0 {
        (fixed + 6.0, fixed + 6.0 + depth)
    } else {
        (fixed - 6.0 - depth, fixed - 6.0)
    };
    let r = if along_x {
        Rect { x0: lo, z0: d0, x1: hi, z1: d1 }
    } else {
        Rect { x0: d0, z0: lo, x1: d1, z1: hi }
    };
    if !b.occ.try_claim([r.x0, r.z0, r.x1, r.z1]) {
        return false;
    }
    b.pads.add_slab(
        r.x0,
        r.z0,
        r.x1,
        r.z1,
        KERB * 0.5,
        scale_color(Color::from_hex(0x6b6862), rng.range(0.94, 1.06)),
        Uv::Unit,
    );
    let at = |s: f32, t: f32| -> (f32, f32) {
        if along_x {
            (s, fixed + side * t)
        } else {
            (fixed + side * t, s)
        }
    };
    const BOX: [u32; 7] = [
        0xb5432f, 0x2f6ab5, 0x3f8f5c, 0xd0a13a, 0x8f4a8a, 0x9aa0a6, 0xc25a2e,
    ];
    for k in 0..sidings {
        let t = 10.0 + k as f32 * pitch;
        // Ballast, then two rails.
        let (b0x, b0z) = at(lo, t - 1.9);
        let (b1x, b1z) = at(hi, t + 1.9);
        b.pads.add_slab(
            b0x.min(b1x),
            b0z.min(b1z),
            b0x.max(b1x),
            b0z.max(b1z),
            KERB * 0.5 + 0.05,
            scale_color(Color::from_hex(0x847d72), rng.range(0.92, 1.08)),
            Uv::Unit,
        );
        for rail in [-0.72f32, 0.72] {
            let (r0x, r0z) = at(lo, t + rail - 0.07);
            let (r1x, r1z) = at(hi, t + rail + 0.07);
            b.trim.add_box(
                Vector3::new(r0x.min(r1x), KERB * 0.5 + 0.05, r0z.min(r1z)),
                Vector3::new(r0x.max(r1x), KERB * 0.5 + 0.22, r0z.max(r1z)),
                Color::from_hex(0x6b5f52),
                Uv::Unit,
            );
        }
        // A rake of flat wagons, most of them loaded.
        let mut s = lo + rng.range(6.0, 40.0);
        while s < hi - 22.0 {
            let len = 19.0;
            if rng.chance(0.78) {
                let (w0x, w0z) = at(s, t - 1.4);
                let (w1x, w1z) = at(s + len, t + 1.4);
                b.trim.add_box(
                    Vector3::new(w0x.min(w1x), KERB * 0.5 + 0.22, w0z.min(w1z)),
                    Vector3::new(w0x.max(w1x), KERB * 0.5 + 1.25, w0z.max(w1z)),
                    Color::from_hex(0x4a4038),
                    Uv::Unit,
                );
                // One forty-foot box or two twenties.
                let boxes = if rng.chance(0.5) { 1 } else { 2 };
                for i in 0..boxes {
                    let (u0, u1) = (
                        i as f32 / boxes as f32 + 0.02,
                        (i + 1) as f32 / boxes as f32 - 0.02,
                    );
                    let (c0x, c0z) = at(s + len * u0, t - 1.25);
                    let (c1x, c1z) = at(s + len * u1, t + 1.25);
                    b.trim.add_box(
                        Vector3::new(c0x.min(c1x), KERB * 0.5 + 1.25, c0z.min(c1z)),
                        Vector3::new(c0x.max(c1x), KERB * 0.5 + 3.8, c0z.max(c1z)),
                        scale_color(
                            Color::from_hex(BOX[rng.below(BOX.len())]),
                            rng.range(0.85, 1.15),
                        ),
                        Uv::Unit,
                    );
                }
            }
            s += len + rng.range(0.6, 3.0);
        }
    }
    // --- The connection to the running line.
    //
    // The line through this city is on a viaduct eight metres up, so a yard
    // laid at ground level beside it is a yard connected to nothing — the same
    // fault as a road that stops in a field, and just as visible once you look
    // for it. This is the branch: a ballast ramp falling from the deck to the
    // yard throat, on a bank, with the rails carried down it.
    {
        let throat = lo + (hi - lo) * 0.12;
        let steps = 10;
        for k in 0..steps {
            let (u0, u1) = (k as f32 / steps as f32, (k + 1) as f32 / steps as f32);
            // Runs back along the line while it falls, at about one in thirty,
            // which is as steep as a freight branch is ever built.
            let s0 = throat - 150.0 * (1.0 - u0);
            let s1 = throat - 150.0 * (1.0 - u1);
            let (y0, y1) = (mix(DECK, KERB * 0.5, u0), mix(DECK, KERB * 0.5, u1));
            // Slews across from the running line to the first siding as it
            // drops, which is what a throat is.
            let (t0, t1) = (mix(1.0, 10.0, u0 * u0), mix(1.0, 10.0, u1 * u1));
            let (p0x, p0z) = at(s0, t0);
            let (p1x, p1z) = at(s1, t1);
            let (dx, dz) = (p1x - p0x, p1z - p0z);
            let l = dx.hypot(dz).max(0.001);
            let (nx, nz) = (-dz / l * 2.4, dx / l * 2.4);
            // Ballast top.
            b.pads.quad(
                [
                    [p0x - nx, y0, p0z - nz],
                    [p1x - nx, y1, p1z - nz],
                    [p1x + nx, y1, p1z + nz],
                    [p0x + nx, y0, p0z + nz],
                ],
                [0.0, 1.0, 0.0],
                Uv::Unit,
                scale_color(Color::from_hex(0x847d72), rng.range(0.92, 1.08)),
            );
            // The bank it sits on, battered either side.
            for sg in [-1.0f32, 1.0] {
                let toe = |y: f32| 2.4 + y * 1.4;
                let a = [p0x + nx / 2.4 * 2.4 * sg, y0, p0z + nz / 2.4 * 2.4 * sg];
                let d = [p1x + nx / 2.4 * 2.4 * sg, y1, p1z + nz / 2.4 * 2.4 * sg];
                let e = [p1x + nx / 2.4 * toe(y1) * sg, 0.0, p1z + nz / 2.4 * toe(y1) * sg];
                let f = [p0x + nx / 2.4 * toe(y0) * sg, 0.0, p0z + nz / 2.4 * toe(y0) * sg];
                let q = if sg > 0.0 { [a, d, e, f] } else { [f, e, d, a] };
                b.grass.quad(
                    q,
                    [nx * sg, 0.6, nz * sg],
                    Uv::Unit,
                    scale_color(Color::from_hex(0x6f8f4a), rng.range(0.92, 1.08)),
                );
            }
            // Rails down the ramp.
            for rail in [-0.72f32, 0.72] {
                let (r0x, r0z) = (p0x + nx / 2.4 * rail, p0z + nz / 2.4 * rail);
                let (r1x, r1z) = (p1x + nx / 2.4 * rail, p1z + nz / 2.4 * rail);
                b.trim.add_limb(
                    Vector3::new(r0x, y0 + 0.16, r0z),
                    Vector3::new(r1x, y1 + 0.16, r1z),
                    0.08,
                    0.08,
                    3,
                    Color::from_hex(0x6b5f52),
                    false,
                );
            }
        }
    }

    // A rail-mounted gantry straddling the sidings, which is the thing that
    // makes it a freight terminal rather than a place trains are stored.
    let gs = mix(lo, hi, rng.range(0.3, 0.7));
    let (l0, l1) = (6.0f32, 10.0 + sidings as f32 * pitch + 2.0);
    for du in [-9.0f32, 9.0] {
        for &t in [l0, l1].iter() {
            let (px, pz) = at(gs + du, t);
            b.trim.add_limb(
                Vector3::new(px, KERB * 0.5, pz),
                Vector3::new(px, 17.0, pz),
                0.6,
                0.45,
                5,
                Color::from_hex(0xd8a52e),
                false,
            );
        }
        let (a0x, a0z) = at(gs + du, l0 - 3.0);
        let (a1x, a1z) = at(gs + du, l1 + 3.0);
        b.trim.add_box(
            Vector3::new(a0x.min(a1x) - 0.4, 17.0, a0z.min(a1z) - 0.4),
            Vector3::new(a0x.max(a1x) + 0.4, 18.6, a0z.max(a1z) + 0.4),
            Color::from_hex(0xd8a52e),
            Uv::Unit,
        );
    }
    // Boxes stacked on the hardstanding beside the sidings.
    let stack_t = 10.0 + sidings as f32 * pitch + 3.0;
    let mut s = lo + 12.0;
    while s < hi - 14.0 {
        if rng.chance(0.35) {
            s += 13.0;
            continue;
        }
        let high = 1 + rng.below(3);
        for k in 0..high {
            let (c0x, c0z) = at(s, stack_t - 1.3);
            let (c1x, c1z) = at(s + 12.0, stack_t + 1.3);
            b.trim.add_box(
                Vector3::new(c0x.min(c1x), KERB * 0.5 + k as f32 * 2.6, c0z.min(c1z)),
                Vector3::new(c0x.max(c1x), KERB * 0.5 + k as f32 * 2.6 + 2.5, c0z.max(c1z)),
                scale_color(Color::from_hex(BOX[rng.below(BOX.len())]), rng.range(0.85, 1.15)),
                Uv::Unit,
            );
        }
        s += 13.0;
    }
    true
}

/// A data centre.
///
/// The identifying feature is what it *hasn't* got: windows. A data hall is a
/// long windowless shed, and the giveaway that it is not a warehouse is the
/// plant — ranks of dry coolers on the roof and a row of generator containers
/// down one flank, which together take up as much ground as the building does.
/// Nothing else in a city is a blank box with that much machinery bolted to it.
pub(crate) fn add_data_centre(b: &mut Batches, r: &Rect, along_x: bool, rng: &mut Rng) -> bool {
    // A small data centre is one hall instead of two, not no data centre.
    if r.w() < 58.0 || r.d() < 44.0 {
        return false;
    }
    if !b.occ.try_claim([r.x0, r.z0, r.x1, r.z1]) {
        return false;
    }
    b.pads.add_slab(
        r.x0,
        r.z0,
        r.x1,
        r.z1,
        KERB,
        scale_color(Color::from_hex(0x8b8880), rng.range(0.94, 1.06)),
        Uv::Unit,
    );
    // Two data halls with a service yard between them.
    let halls = if (if along_x { r.d() } else { r.w() }) > 84.0 { 2 } else { 1 };
    let h = 11.0f32;
    let wall = scale_color(Color::from_hex(0xb4b8bc), rng.range(0.95, 1.05));
    let mut plant = 0usize;
    for k in 0..halls {
        let t0 = 0.06 + k as f32 * 0.50;
        let t1 = t0 + 0.38;
        let hall = if along_x {
            Rect {
                x0: r.x0 + 8.0,
                z0: mix(r.z0, r.z1, t0),
                x1: r.x1 - 22.0,
                z1: mix(r.z0, r.z1, t1),
            }
        } else {
            Rect {
                x0: mix(r.x0, r.x1, t0),
                z0: r.z0 + 8.0,
                x1: mix(r.x0, r.x1, t1),
                z1: r.z1 - 22.0,
            }
        };
        if hall.w() < 12.0 || hall.d() < 12.0 {
            continue;
        }
        // The hall: blank, and banded rather than glazed.
        b.trim.add_box(
            Vector3::new(hall.x0, KERB, hall.z0),
            Vector3::new(hall.x1, h, hall.z1),
            wall,
            Uv::Unit,
        );
        for band in [0.34f32, 0.68] {
            b.trim.add_box(
                Vector3::new(hall.x0 - 0.15, h * band, hall.z0 - 0.15),
                Vector3::new(hall.x1 + 0.15, h * band + 0.5, hall.z1 + 0.15),
                scale_color(wall, 0.86),
                Uv::Unit,
            );
        }
        b.trim.add_box(
            Vector3::new(hall.x0 - 0.3, h, hall.z0 - 0.3),
            Vector3::new(hall.x1 + 0.3, h + 1.2, hall.z1 + 0.3),
            scale_color(wall, 0.78),
            Uv::Unit,
        );
        // Dry coolers on the roof, in ranks. This is the part that says what
        // the building is: a grid of identical fan units covering most of it.
        let pitch = 4.6f32;
        let (nx, nz) = (
            ((hall.w() - 4.0) / pitch) as usize,
            ((hall.d() - 4.0) / pitch) as usize,
        );
        for i in 0..nx {
            for j in 0..nz {
                let cx = hall.x0 + 2.0 + (i as f32 + 0.5) * pitch;
                let cz = hall.z0 + 2.0 + (j as f32 + 0.5) * pitch;
                b.trim.add_box(
                    Vector3::new(cx - 1.7, h + 1.2, cz - 1.5),
                    Vector3::new(cx + 1.7, h + 2.6, cz + 1.5),
                    Color::from_hex(0x8f969c),
                    Uv::Unit,
                );
                // The fan cowl on top, which is what makes it read as plant
                // rather than as a crate.
                b.trim.add_limb(
                    Vector3::new(cx, h + 2.6, cz),
                    Vector3::new(cx, h + 3.0, cz),
                    1.25,
                    1.35,
                    8,
                    Color::from_hex(0x6f767c),
                    false,
                );
                plant += 1;
            }
        }
    }
    // Generator containers down one flank, each with an exhaust stack.
    let n_gen = 5 + rng.below(4);
    for i in 0..n_gen {
        let t = (i as f32 + 0.5) / n_gen as f32;
        let (gx, gz) = if along_x {
            (r.x1 - 12.0, mix(r.z0 + 8.0, r.z1 - 8.0, t))
        } else {
            (mix(r.x0 + 8.0, r.x1 - 8.0, t), r.z1 - 12.0)
        };
        let (hw, hd) = if along_x { (5.0f32, 1.6f32) } else { (1.6, 5.0) };
        b.trim.add_box(
            Vector3::new(gx - hw, KERB, gz - hd),
            Vector3::new(gx + hw, KERB + 3.4, gz + hd),
            scale_color(Color::from_hex(0x5f6a52), rng.range(0.92, 1.08)),
            Uv::Unit,
        );
        b.trim.add_limb(
            Vector3::new(gx + hw * 0.6, KERB + 3.4, gz + hd * 0.6),
            Vector3::new(gx + hw * 0.6, KERB + 8.4, gz + hd * 0.6),
            0.34,
            0.30,
            7,
            Color::from_hex(0x4a4f52),
            false,
        );
    }
    // Transformers and the incoming gantry: a data hall is a substation with a
    // building attached, and the yard is never hidden.
    for i in 0..3 {
        let t = (i as f32 + 0.5) / 3.0;
        let (tx, tz) = if along_x {
            (r.x1 - 4.5, mix(r.z0 + 14.0, r.z1 - 14.0, t))
        } else {
            (mix(r.x0 + 14.0, r.x1 - 14.0, t), r.z1 - 4.5)
        };
        b.trim.add_box(
            Vector3::new(tx - 2.2, KERB, tz - 2.2),
            Vector3::new(tx + 2.2, KERB + 3.6, tz + 2.2),
            Color::from_hex(0x6b7075),
            Uv::Unit,
        );
        for s in [-1.0f32, 1.0] {
            b.trim.add_limb(
                Vector3::new(tx + s * 1.4, KERB + 3.6, tz),
                Vector3::new(tx + s * 1.4, KERB + 6.4, tz),
                0.20,
                0.16,
                5,
                Color::from_hex(0x8d9196),
                false,
            );
        }
    }
    // A small office at the front — the only part with any glass on it.
    let (ox, oz) = if along_x {
        (r.x0 + 14.0, r.z0 + 5.0)
    } else {
        (r.x0 + 5.0, r.z0 + 14.0)
    };
    let (ow, od) = if along_x { (12.0f32, 4.0f32) } else { (4.0, 12.0) };
    b.trim.add_box(
        Vector3::new(ox - ow, KERB, oz - od),
        Vector3::new(ox + ow, KERB + 7.0, oz + od),
        Color::from_hex(0xd4d0c8),
        Uv::Unit,
    );
    b.trim.add_box(
        Vector3::new(ox - ow + 1.0, KERB + 1.4, oz - od - 0.15),
        Vector3::new(ox + ow - 1.0, KERB + 5.6, oz - od + 0.05),
        Color::from_hex(0x2a3238),
        Uv::Unit,
    );
    // Security fence and cameras. These places are watched harder than banks.
    let peri = 30;
    for k in 0..peri {
        let t = k as f32 / peri as f32 * 4.0;
        let (px, pz) = match t as usize {
            0 => (r.x0, mix(r.z0, r.z1, t.fract())),
            1 => (mix(r.x0, r.x1, t.fract()), r.z1),
            2 => (r.x1, mix(r.z1, r.z0, t.fract())),
            _ => (mix(r.x1, r.x0, t.fract()), r.z0),
        };
        b.trim.add_limb(
            Vector3::new(px, KERB, pz),
            Vector3::new(px, KERB + 3.0, pz),
            0.07,
            0.06,
            4,
            Color::from_hex(0x6f747a),
            false,
        );
        if k % 6 == 0 {
            add_cctv(
                b,
                Vector3::new(px, KERB + 2.8, pz),
                Vector3::new(r.cx() - px, 0.0, r.cz() - pz),
                Cctv::Bullet,
                rng,
            );
        }
    }
    plant > 0
}

/// A telecommunications mast.
///
/// A lattice tower carrying panel antennas in triads and microwave dishes at
/// two levels, with a waveguide run up one face and a lamp on top. The triad
/// is what identifies it: three flat panels at a hundred and twenty degrees is
/// a shape nothing else in a city makes.
pub(crate) fn add_telecom_mast(b: &mut Batches, cx: f32, cz: f32, h: f32, rng: &mut Rng) -> bool {
    if !b.occ.try_claim([cx - 9.0, cz - 9.0, cx + 9.0, cz + 9.0]) {
        return false;
    }
    let steel = Color::from_hex(0x9aa0a6);
    let base = h * 0.11;
    // Legs, tapering, with X-bracing every bay.
    const BAYS: usize = 12;
    let radius = |t: f32| base * (1.0 - t * 0.62) + 0.5;
    for k in 0..BAYS {
        let (t0, t1) = (k as f32 / BAYS as f32, (k + 1) as f32 / BAYS as f32);
        let (r0, r1) = (radius(t0), radius(t1));
        let corner = |r: f32, i: usize| {
            let a = i as f32 / 3.0 * TAU;
            (cx + r * a.cos(), cz + r * a.sin())
        };
        for i in 0..3 {
            let (ax, az) = corner(r0, i);
            let (bx, bz) = corner(r1, i);
            b.trim.add_limb(
                Vector3::new(ax, KERB + h * t0, az),
                Vector3::new(bx, KERB + h * t1, bz),
                0.16,
                0.15,
                4,
                steel,
                false,
            );
            let j = (i + 1) % 3;
            let (jx, jz) = corner(r1, j);
            let (kx, kz) = corner(r0, j);
            b.trim.add_limb(
                Vector3::new(ax, KERB + h * t0, az),
                Vector3::new(jx, KERB + h * t1, jz),
                0.07,
                0.07,
                3,
                steel,
                false,
            );
            b.trim.add_limb(
                Vector3::new(kx, KERB + h * t0, kz),
                Vector3::new(bx, KERB + h * t1, bz),
                0.07,
                0.07,
                3,
                steel,
                false,
            );
        }
    }
    // Antenna triads at two or three levels.
    let levels = 2 + rng.below(2);
    for l in 0..levels {
        let t = 0.72 + l as f32 * 0.11;
        if t > 1.0 {
            break;
        }
        let y = KERB + h * t;
        let r = radius(t) + 1.5;
        for i in 0..3 {
            let a = i as f32 / 3.0 * TAU + 0.4;
            let (px, pz) = (cx + r * a.cos(), cz + r * a.sin());
            // Support arm out to the panel.
            b.trim.add_limb(
                Vector3::new(cx + radius(t) * a.cos(), y, cz + radius(t) * a.sin()),
                Vector3::new(px, y, pz),
                0.09,
                0.08,
                4,
                steel,
                false,
            );
            // The panel: tall, flat, and square to the arm.
            let (nx, nz) = (-a.sin(), a.cos());
            b.trim.quad(
                [
                    [px - nx * 0.35, y - 1.3, pz - nz * 0.35],
                    [px + nx * 0.35, y - 1.3, pz + nz * 0.35],
                    [px + nx * 0.35, y + 1.3, pz + nz * 0.35],
                    [px - nx * 0.35, y + 1.3, pz - nz * 0.35],
                ],
                [a.cos(), 0.0, a.sin()],
                Uv::Unit,
                Color::from_hex(0xd8dce0),
            );
            b.trim.add_box(
                Vector3::new(px - 0.36, y - 1.3, pz - 0.36),
                Vector3::new(px + 0.36, y + 1.3, pz + 0.36),
                Color::from_hex(0xd0d4d8),
                Uv::Unit,
            );
        }
    }
    // Microwave dishes: drums facing outward, at two heights.
    for (t, rr) in [(0.55f32, 1.6f32), (0.66, 1.2)] {
        let a = rng.range(0.0, TAU);
        let y = KERB + h * t;
        let (px, pz) = (cx + (radius(t) + rr * 0.4) * a.cos(), cz + (radius(t) + rr * 0.4) * a.sin());
        b.trim.add_limb(
            Vector3::new(px, y, pz),
            Vector3::new(px + a.cos() * 0.5, y, pz + a.sin() * 0.5),
            rr,
            rr * 0.94,
            10,
            Color::from_hex(0xe0e4e8),
            true,
        );
    }
    // Waveguide run up one face, and the lamp on top.
    let a = 0.0f32;
    for k in 0..10 {
        let (t0, t1) = (k as f32 / 10.0, (k + 1) as f32 / 10.0);
        b.trim.add_limb(
            Vector3::new(cx + (radius(t0) + 0.3) * a.cos(), KERB + h * t0, cz),
            Vector3::new(cx + (radius(t1) + 0.3) * a.cos(), KERB + h * t1, cz),
            0.13,
            0.12,
            4,
            Color::from_hex(0x6f747a),
            false,
        );
    }
    b.trim.add_limb(
        Vector3::new(cx, KERB + h, cz),
        Vector3::new(cx, KERB + h + 3.0, cz),
        0.22,
        0.10,
        5,
        steel,
        false,
    );
    b.beacon.add_box(
        Vector3::new(cx - 0.4, KERB + h + 3.0, cz - 0.4),
        Vector3::new(cx + 0.4, KERB + h + 3.9, cz + 0.4),
        Color::from_hex(0xff5a3a),
        Uv::Unit,
    );
    // An equipment cabin at the foot, fenced.
    b.trim.add_box(
        Vector3::new(cx + 3.0, KERB, cz - 2.2),
        Vector3::new(cx + 7.0, KERB + 2.8, cz + 2.2),
        Color::from_hex(0xc4c8cc),
        Uv::Unit,
    );
    true
}
