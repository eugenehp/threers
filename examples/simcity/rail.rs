//! Part of the `simcity` example; see `mod.rs`.
//!
//! The railway: a viaduct across the city, a station on it, tunnel portals at
//! both ends, and the entrances to whatever runs underneath.
//!
//! Rail is the one piece of city infrastructure that ignores the street grid
//! entirely — it goes where it needs to and the streets pass under it — which
//! is exactly why it is worth having. Every other line in this city is at
//! ground level and parallel to something.
//!
//! The train is an ordinary [`Mover`] on a lane of its own, at the deck's
//! height. It gets that for free: the lane machinery already handles a long
//! vehicle on a bounded run, and a train is nothing but a very long vehicle
//! that never turns. The wrap point is put *inside a tunnel*, so the one place
//! the simulation teleports is the one place nobody can see.
#![allow(dead_code)]

use super::*;

/// Rail level above the street.
pub(crate) const DECK: f32 = 8.4;
/// Half the width of the formation.
pub(crate) const HALF_W: f32 = 3.6;

/// Where the line runs and how far the train may travel on it.
pub(crate) struct Railway {
    /// True when the line runs along X.
    pub(crate) along_x: bool,
    /// The other coordinate: the line's own position across the city.
    pub(crate) fixed: f32,
    pub(crate) s_lo: f32,
    pub(crate) s_hi: f32,
    /// Where the platforms are, for putting people on them.
    pub(crate) platform: (f32, f32),
}

/// Build the viaduct, the portals at each end and the station.
///
/// `lo`/`hi` are the ends of the line, which run a good way past the built
/// area so the portals sit out in the fields where a tunnel mouth belongs.
pub(crate) fn add_railway(
    b: &mut Batches,
    along_x: bool,
    fixed: f32,
    lo: f32,
    hi: f32,
    // The stretch of the line that crosses open water, if any. A pier in a
    // navigable channel is wrong on its own terms and, here, something for a
    // barge to sail through — so the channel gets one clear span.
    channel: Option<(f32, f32)>,
    // How far from the middle the masonry is worth modelling in full.
    //
    // The line runs the width of the world — twelve kilometres of it — and an
    // arch every twelve metres at full detail came to five hundred and seventy
    // thousand triangles: thirty per cent of everything static in the city,
    // for a structure whose voussoirs stop being legible a few hundred metres
    // away and which is entirely inside the haze past five kilometres.
    near_reach: f32,
    rng: &mut Rng,
) -> Railway {
    let stone = scale_color(Color::from_hex(0x7d7468), rng.range(0.92, 1.08));
    let dark = scale_color(Color::from_hex(0x5a5349), rng.range(0.92, 1.08));
    let steel = Color::from_hex(0x4a4f52);

    // Position along the line at `s`, offset `t` across it.
    let at = |s: f32, t: f32| -> (f32, f32) {
        if along_x {
            (s, fixed + t)
        } else {
            (fixed + t, s)
        }
    };
    let bx = |s0: f32, s1: f32, t0: f32, t1: f32, y0: f32, y1: f32| -> (Vector3, Vector3) {
        let (a, c) = (at(s0, t0), at(s1, t1));
        (
            Vector3::new(a.0.min(c.0), y0, a.1.min(c.1)),
            Vector3::new(a.0.max(c.0), y1, a.1.max(c.1)),
        )
    };

    // --- The deck, in one piece, with a parapet each side.
    let (d0, d1) = bx(lo, hi, -HALF_W, HALF_W, DECK - 0.9, DECK);
    b.trim.add_box(d0, d1, stone, Uv::Unit);
    for t in [-HALF_W, HALF_W - 0.35] {
        let (p0, p1) = bx(lo, hi, t, t + 0.35, DECK, DECK + 0.95);
        b.trim.add_box(p0, p1, scale_color(stone, 1.06), Uv::Unit);
    }

    // --- Piers and arches.
    //
    // These were a rectangular column and a stack of five blocks stepping
    // inwards to suggest a curve. From any distance where the viaduct is
    // interesting that reads as a staircase, and a masonry viaduct has no
    // straight lines in it below the deck at all.
    //
    // So: piers that batter — wider at the foot than at the springing, which
    // is how they are built and what stops them looking like scaffolding —
    // and a true semicircular arch, with a curved barrel between the two faces
    // and a voussoir ring on each. The spandrel above is genuinely flat
    // masonry, so that stays a wall; it is the *opening* that has to curve.
    let span = 12.0f32;
    let n = ((hi - lo) / span).floor().max(1.0) as usize;
    let pier_w = 3.0f32;
    let springing = DECK - 0.9 - (span - pier_w) * 0.5;
    // Nothing may stand between the banks, and the abutments each side are
    // pulled out to the water's edge so the clear span reads as deliberate.
    let in_channel = |a: f32, c: f32| {
        channel.is_some_and(|(w0, w1)| a < w1 + 2.0 && c > w0 - 2.0)
    };
    for i in 0..=n {
        let s = mix(lo, hi, i as f32 / n as f32);
        if in_channel(s - pier_w * 0.5, s + pier_w * 0.5) {
            continue;
        }
        // Three tiers. Near the city the masonry is modelled; beyond that the
        // arch keeps its shape at a quarter of the segments and loses the
        // voussoir rings; far out it is piers and deck, which is all that
        // survives the haze anyway.
        let d = s.abs();
        let near = d < near_reach;
        let mid = d < near_reach * 4.0;
        if !mid && i % 3 != 0 {
            // Out here even the piers thin out: at this range they are one
            // pixel wide and there is a deck over them.
            continue;
        }
        // A battered pier. `add_limb` with four segments looked like the
        // obvious way to taper one, but four segments give a square rotated
        // forty-five degrees — a diamond in plan — so every pier read as a
        // thin fin edge-on to the viaduct. Stacked courses instead: each one
        // slightly narrower than the one below, which is both how the thing is
        // built and how the batter reads.
        let courses: usize = if near { 5 } else { 2 };
        for c in 0..courses {
            let (u0, u1) = (c as f32 / courses as f32, (c + 1) as f32 / courses as f32);
            let w0 = mix(pier_w * 0.62, pier_w * 0.46, u0);
            let t0 = mix(HALF_W * 0.98, HALF_W * 0.84, u0);
            let (c0, c1) = bx(
                s - w0,
                s + w0,
                -t0,
                t0,
                mix(0.9, springing, u0),
                mix(0.9, springing, u1) + 0.02,
            );
            b.trim.add_box(c0, c1, scale_color(dark, 1.0 - u0 * 0.06), Uv::Unit);
        }
        // A splayed plinth at the foot, and an impost band at the springing —
        // the two horizontal lines that give a pier its scale.
        let (b0, b1) = bx(s - pier_w * 0.62, s + pier_w * 0.62, -HALF_W - 0.25, HALF_W + 0.25, 0.0, 1.1);
        b.trim.add_box(b0, b1, scale_color(dark, 0.92), Uv::Unit);
        let (i0, i1) = bx(s - pier_w * 0.60, s + pier_w * 0.60, -HALF_W - 0.18, HALF_W + 0.18, springing - 0.45, springing);
        b.trim.add_box(i0, i1, scale_color(stone, 0.96), Uv::Unit);

        if i == n {
            continue;
        }
        let s1 = mix(lo, hi, (i + 1) as f32 / n as f32);
        if in_channel(s, s1) {
            continue;
        }
        if !mid {
            continue;
        }
        // The arch: a semicircle springing from pier face to pier face.
        let (a0, a1) = (s + pier_w * 0.5, s1 - pier_w * 0.5);
        let arch_mid = (a0 + a1) * 0.5;
        let r = (a1 - a0) * 0.5;
        let segs: usize = if near { 10 } else { 3 };
        let arc = |k: usize| -> (f32, f32) {
            let th = PI * k as f32 / segs as f32;
            (arch_mid - r * th.cos(), springing + r * th.sin())
        };
        for k in 0..segs {
            let (u0, y0) = arc(k);
            let (u1, y1) = arc(k + 1);
            // Barrel: the underside of the arch, spanning between the faces.
            let (q0x, q0z) = at(u0, -HALF_W + 0.35);
            let (q1x, q1z) = at(u0, HALF_W - 0.35);
            let (q2x, q2z) = at(u1, HALF_W - 0.35);
            let (q3x, q3z) = at(u1, -HALF_W + 0.35);
            // Normal points down and outward from the arch centre.
            let nx = (u0 + u1) * 0.5 - arch_mid;
            let ny = -((y0 + y1) * 0.5 - springing);
            let nl = (nx * nx + ny * ny).sqrt().max(1e-4);
            let nrm = if along_x {
                [nx / nl, ny / nl, 0.0]
            } else {
                [0.0, ny / nl, nx / nl]
            };
            b.trim.quad(
                [
                    [q0x, y0, q0z],
                    [q3x, y1, q3z],
                    [q2x, y1, q2z],
                    [q1x, y0, q1z],
                ],
                nrm,
                Uv::Unit,
                scale_color(dark, 0.88),
            );
            // Voussoir ring on each face: a band of masonry following the
            // curve, slightly proud of the spandrel behind it. Near work only —
            // a fifty-five millimetre step in the stonework is not visible from
            // the next parish.
            for (t0, t1) in [
                (-HALF_W - 0.12, -HALF_W + 0.35),
                (HALF_W - 0.35, HALF_W + 0.12),
            ] {
                if !near {
                    continue;
                }
                let ring = 0.55f32;
                let (e0x, e0z) = at(u0, t0);
                let (e1x, e1z) = at(u1, t1);
                let (o0, o1) = (
                    (u0 - arch_mid) / r * ring,
                    (u1 - arch_mid) / r * ring,
                );
                let (h0, h1) = (
                    (y0 - springing) / r * ring,
                    (y1 - springing) / r * ring,
                );
                let (f0x, f0z) = at(u0 + o0, t0);
                let (f1x, f1z) = at(u1 + o1, t1);
                b.trim.quad(
                    [
                        [e0x, y0, e0z],
                        [f0x, y0 + h0, f0z],
                        [f1x, y1 + h1, f1z],
                        [e1x, y1, e1z],
                    ],
                    if along_x { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] },
                    Uv::Unit,
                    scale_color(stone, 1.02),
                );
            }
            // Spandrel: the flat wall above the extrados, up to the deck.
            let top = DECK - 0.9;
            let ext0 = y0 + (y0 - springing) / r * 0.55;
            let ext1 = y1 + (y1 - springing) / r * 0.55;
            if ext0.min(ext1) < top {
                for (t0, t1) in [
                    (-HALF_W, -HALF_W + 0.35),
                    (HALF_W - 0.35, HALF_W),
                ] {
                    let (g0x, g0z) = at(u0 + (u0 - arch_mid) / r * 0.55, t0);
                    let (g1x, g1z) = at(u1 + (u1 - arch_mid) / r * 0.55, t1);
                    let (gx0, gz0) = (g0x.min(g1x), g0z.min(g1z));
                    let (gx1, gz1) = (g0x.max(g1x), g0z.max(g1z));
                    if gx1 - gx0 > 0.02 && gz1 - gz0 > 0.02 {
                        b.trim.add_box(
                            Vector3::new(gx0, ext0.min(ext1).min(top), gz0),
                            Vector3::new(gx1, top, gz1),
                            stone,
                            Uv::Unit,
                        );
                    }
                }
            }
        }
    }

    // A truss over the clear span, which is what carries a railway across a
    // river when it cannot be propped in the middle of it.
    if let Some((w0, w1)) = channel {
        let (t0, t1) = bx(w0 - 3.0, w1 + 3.0, -HALF_W, HALF_W, DECK - 1.4, DECK - 0.9);
        b.trim.add_box(t0, t1, steel, Uv::Unit);
        for t in [-HALF_W, HALF_W - 0.3] {
            let n_bays = ((w1 - w0) / 5.0).max(2.0) as usize;
            for k in 0..n_bays {
                let a = mix(w0 - 3.0, w1 + 3.0, k as f32 / n_bays as f32);
                let c = mix(w0 - 3.0, w1 + 3.0, (k + 1) as f32 / n_bays as f32);
                // Diagonals, alternating direction: a Warren truss.
                let (p, q) = if k % 2 == 0 {
                    (at(a, t + 0.15), at(c, t + 0.15))
                } else {
                    (at(c, t + 0.15), at(a, t + 0.15))
                };
                b.trim.add_limb(
                    Vector3::new(p.0, DECK + 0.1, p.1),
                    Vector3::new(q.0, DECK + 3.4, q.1),
                    0.14,
                    0.14,
                    4,
                    steel,
                    false,
                );
            }
            let (u0, u1) = bx(w0 - 3.0, w1 + 3.0, t, t + 0.3, DECK + 3.3, DECK + 3.7);
            b.trim.add_box(u0, u1, steel, Uv::Unit);
        }
    }

    // --- Ballast, sleepers and rail. The sleepers are what make it read as
    // track rather than as two lines painted on a bridge.
    let (b0, b1) = bx(lo, hi, -2.5, 2.5, DECK, DECK + 0.30);
    b.trim
        .add_box(b0, b1, scale_color(Color::from_hex(0x6b6157), 0.9), Uv::Unit);
    let ties = ((hi - lo) / 0.65) as usize;
    for i in 0..ties {
        let s = mix(lo, hi, (i as f32 + 0.5) / ties as f32);
        let (t0, t1) = bx(s - 0.12, s + 0.12, -1.9, 1.9, DECK + 0.28, DECK + 0.40);
        b.trim.add_box(
            t0,
            t1,
            scale_color(Color::from_hex(0x4a3a2c), 0.85 + 0.3 * city_hash2(i as i32, 3)),
            Uv::Unit,
        );
    }
    for t in [-0.75f32, 0.75] {
        let (r0, r1) = bx(lo, hi, t - 0.07, t + 0.07, DECK + 0.40, DECK + 0.55);
        b.trim.add_box(r0, r1, steel, Uv::Unit);
    }

    // --- Tunnel portals. An embankment with a stone face and a black opening
    // in it, at each end. The train wraps inside, where it cannot be seen.
    for (end, dir) in [(lo, 1.0f32), (hi, -1.0)] {
        let mouth = end + dir * 16.0;
        let (e0, e1) = bx(
            end - dir * 26.0,
            mouth,
            -HALF_W - 5.0,
            HALF_W + 5.0,
            0.0,
            DECK + 4.2,
        );
        b.grass
            .add_box(e0, e1, scale_color(Color::from_hex(C_TERRAIN), 0.92), Uv::Unit);
        // The face, and the hole.
        let (f0, f1) = bx(mouth - dir * 0.9, mouth, -HALF_W - 1.2, HALF_W + 1.2, 0.0, DECK + 3.4);
        b.trim.add_box(f0, f1, stone, Uv::Unit);
        let (h0, h1) = bx(
            mouth - dir * 1.4,
            mouth + dir * 0.2,
            -2.7,
            2.7,
            DECK - 0.1,
            DECK + 2.6,
        );
        b.trim.add_box(h0, h1, Color::new(0.02, 0.02, 0.025), Uv::Unit);
        // A keystone and coping, so the mouth has an arch over it.
        let (k0, k1) = bx(mouth - dir * 1.1, mouth + dir * 0.1, -0.5, 0.5, DECK + 2.5, DECK + 3.1);
        b.trim.add_box(k0, k1, scale_color(stone, 1.15), Uv::Unit);
    }

    // --- Station: a pair of platforms on the deck with a canopy over them.
    let mid = (lo + hi) * 0.5;
    let plat = (mid - 22.0, mid + 22.0);
    for t in [-3.1f32, 2.1] {
        let (p0, p1) = bx(plat.0, plat.1, t, t + 1.0, DECK + 0.40, DECK + 0.95);
        b.trim
            .add_box(p0, p1, scale_color(Color::from_hex(0x9a958c), 1.0), Uv::Unit);
    }
    // Canopy on columns, and a valance along its edge.
    let cols = 7;
    for i in 0..=cols {
        let s = mix(plat.0, plat.1, i as f32 / cols as f32);
        for t in [-2.7f32, 2.6] {
            let (x, z) = at(s, t);
            b.trim.add_cylinder(
                Vector3::new(x, DECK + 0.95, z),
                0.10,
                0.09,
                3.1,
                6,
                steel,
                false,
                Uv::Unit,
            );
        }
    }
    let (c0, c1) = bx(plat.0 - 1.0, plat.1 + 1.0, -3.4, 3.3, DECK + 4.05, DECK + 4.25);
    b.trim
        .add_box(c0, c1, scale_color(Color::from_hex(0x3f4a50), 1.0), Uv::Unit);
    // Lit platform edge after dark, which is what a station looks like at
    // night from anywhere else in the city.
    for t in [-3.1f32, 3.05] {
        let (g0, g1) = bx(plat.0, plat.1, t, t + 0.06, DECK + 0.90, DECK + 0.96);
        b.glow.add_box(g0, g1, Color::from_hex(0xffeec4), Uv::Unit);
    }

    // Cameras along both platform edges, under the canopy and looking down
    // the platform. A station is the most heavily watched building most people
    // pass through in a day.
    for t in [-3.0f32, 3.0] {
        for k in 0..4 {
            let u = (k as f32 + 0.5) / 4.0;
            let s0 = mix(plat.0, plat.1, u);
            let (px, pz) = if along_x { (s0, fixed + t) } else { (fixed + t, s0) };
            add_cctv(
                b,
                Vector3::new(px, DECK + 3.7, pz),
                if along_x {
                    Vector3::new(if k % 2 == 0 { 1.0 } else { -1.0 }, 0.0, -t.signum() * 0.35)
                } else {
                    Vector3::new(-t.signum() * 0.35, 0.0, if k % 2 == 0 { 1.0 } else { -1.0 })
                },
                if k % 3 == 0 { Cctv::Dome } else { Cctv::Bullet },
                rng,
            );
        }
    }

    Railway {
        along_x,
        fixed,
        // The runnable stretch reaches into both tunnels, so the wrap happens
        // out of sight.
        s_lo: lo + 4.0,
        s_hi: hi - 4.0,
        platform: plat,
    }
}

/// A train: a driving car at each end and flat-sided stock between them.
///
/// Built along `+Z` like every other vehicle, so the same lane machinery
/// places it. `cars` sets the length; the geometry is one mesh because a train
/// never articulates on a straight line.
pub(crate) fn train_geometry(cars: usize, livery: Color) -> BufferGeometry {
    let mut m = MeshBuilder::default();
    let glass = Color::new(0.10, 0.12, 0.15);
    let skirt = Color::from_hex(0x2b2d31);
    let car_len = 19.0f32;
    let total = car_len * cars as f32;
    for i in 0..cars {
        let z0 = -total * 0.5 + i as f32 * car_len + 0.35;
        let z1 = z0 + car_len - 0.7;
        // Body, with the roof narrowed so the section is not a brick.
        m.add_box(
            Vector3::new(-1.45, 0.75, z0),
            Vector3::new(1.45, 3.05, z1),
            livery,
            Uv::Unit,
        );
        m.add_box(
            Vector3::new(-1.28, 3.05, z0 + 0.2),
            Vector3::new(1.28, 3.40, z1 - 0.2),
            scale_color(livery, 0.86),
            Uv::Unit,
        );
        m.add_box(
            Vector3::new(-1.30, 0.35, z0 + 0.5),
            Vector3::new(1.30, 0.75, z1 - 0.5),
            skirt,
            Uv::Unit,
        );
        // A window band down each side, and doors breaking it.
        for sx in [-1.0f32, 1.0] {
            m.add_box(
                Vector3::new(sx * 1.46 - 0.03, 1.85, z0 + 0.9),
                Vector3::new(sx * 1.46 + 0.03, 2.72, z1 - 0.9),
                glass,
                Uv::Unit,
            );
            for d in [0.28f32, 0.72] {
                let dz = mix(z0, z1, d);
                m.add_box(
                    Vector3::new(sx * 1.47 - 0.035, 0.80, dz - 0.65),
                    Vector3::new(sx * 1.47 + 0.035, 2.80, dz + 0.65),
                    scale_color(livery, 0.72),
                    Uv::Unit,
                );
            }
        }
        // Bogies.
        for t in [0.22f32, 0.78] {
            let bz = mix(z0, z1, t);
            for sx in [-1.0f32, 1.0] {
                m.add_box(
                    Vector3::new(sx * 0.95 - 0.16, 0.0, bz - 1.5),
                    Vector3::new(sx * 0.95 + 0.16, 0.55, bz + 1.5),
                    skirt,
                    Uv::Unit,
                );
            }
        }
    }
    // Cab ends: a raked front, which is the whole silhouette of a train from
    // the platform.
    for (zs, dir) in [(total * 0.5, 1.0f32), (-total * 0.5, -1.0)] {
        m.add_box(
            Vector3::new(-1.45, 0.75, zs - dir * 0.35),
            Vector3::new(1.45, 2.70, zs),
            livery,
            Uv::Unit,
        );
        m.add_box(
            Vector3::new(-1.20, 1.90, zs - dir * 0.42),
            Vector3::new(1.20, 2.62, zs + dir * 0.03),
            glass,
            Uv::Unit,
        );
    }
    m.build()
}

/// The entrance to whatever runs under the street: a stair down, railings, a
/// canopy and a sign.
pub(crate) fn add_metro_entrance(b: &mut Batches, x: f32, z: f32, yaw: f32, rng: &mut Rng) {
    // Three metres of footway, railings and a sign: the largest thing that
    // ever lands on a pavement here, so it wants the most room.
    if !b.occ.try_spot(x, z, 2.6) {
        return;
    }
    let iron = scale_color(Color::from_hex(0x1f3d33), rng.range(0.9, 1.1));
    let stone = Color::from_hex(0x8f8a80);
    // The opening, and the steps falling away into it.
    b.trim.add_yaw_box(
        Vector3::new(x, KERB - 0.6, z),
        Vector3::new(1.5, 0.6, 2.1),
        yaw,
        Color::new(0.03, 0.03, 0.035),
        Uv::Unit,
    );
    for k in 0..5 {
        let t = k as f32 / 5.0;
        b.trim.add_yaw_box(
            Vector3::new(
                x - (t * 1.5) * yaw.sin(),
                KERB - 0.12 - t * 0.55,
                z + (t * 1.5) * yaw.cos(),
            ),
            Vector3::new(1.35, 0.09, 0.30),
            yaw,
            stone,
            Uv::Unit,
        );
    }
    // Railings round three sides of the hole.
    for (dx, dz, len, ry) in [
        (1.45f32, 0.0f32, 2.1f32, yaw),
        (-1.45, 0.0, 2.1, yaw),
        (0.0, -2.05, 1.45, yaw + PI * 0.5),
    ] {
        let px = x + dx * yaw.cos() - dz * yaw.sin();
        let pz = z + dx * yaw.sin() + dz * yaw.cos();
        b.trim.add_yaw_box(
            Vector3::new(px, KERB + 0.55, pz),
            Vector3::new(0.04, 0.05, len),
            ry,
            iron,
            Uv::Unit,
        );
        for k in 0..4 {
            let t = (k as f32 / 3.0 - 0.5) * len * 1.8;
            b.trim.add_cylinder(
                Vector3::new(px - t * ry.sin(), KERB, pz + t * ry.cos()),
                0.035,
                0.030,
                1.05,
                4,
                iron,
                false,
                Uv::Unit,
            );
        }
    }
    // Sign on a post: a roundel, which is the one piece of a metro anybody
    // recognises from across a street.
    let sx = x + 1.9 * yaw.cos();
    let sz = z + 1.9 * yaw.sin();
    b.trim
        .add_cylinder(Vector3::new(sx, KERB, sz), 0.06, 0.055, 2.6, 5, iron, false, Uv::Unit);
    // Painted, on `trim`, NOT on `neon`. A neon batch is a Basic material
    // scaled by `sky.lights`, which is zero whenever the sun is up — so the
    // roundel the comment above calls the one recognisable piece of a metro
    // was a solid black disc for the whole daylight half of the clock. The lit
    // copy goes on `glow`, which is genuinely hidden by day rather than
    // blackened, and sits proud so it reads in front after dark.
    let roundel = Color::from_hex(0xd8322c);
    let bar = Color::from_hex(0x1d4e8a);
    b.trim.add_yaw_box(
        Vector3::new(sx, KERB + 2.75, sz),
        Vector3::new(0.62, 0.62, 0.05),
        yaw,
        roundel,
        Uv::Unit,
    );
    b.trim.add_yaw_box(
        Vector3::new(sx, KERB + 2.75, sz),
        Vector3::new(0.70, 0.20, 0.07),
        yaw,
        bar,
        Uv::Unit,
    );
    let (gx, gz) = (0.02 * yaw.cos(), 0.02 * yaw.sin());
    b.glow.add_yaw_box(
        Vector3::new(sx + gx, KERB + 2.75, sz + gz),
        Vector3::new(0.62, 0.62, 0.05),
        yaw,
        roundel,
        Uv::Unit,
    );
    b.glow.add_yaw_box(
        Vector3::new(sx + gx, KERB + 2.75, sz + gz),
        Vector3::new(0.70, 0.20, 0.07),
        yaw,
        bar,
        Uv::Unit,
    );
}
