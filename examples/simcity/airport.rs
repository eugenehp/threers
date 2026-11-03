//! Part of the `simcity` example; see `mod.rs`.
//!
//! The airport.
//!
//! There were aircraft in the sky over this city from the moment it had a sky,
//! and nowhere for any of them to have come from — the same fault as a road
//! that ends in a field, only three hundred metres up where it is harder to
//! notice.
//!
//! An airport is one of the few things whose plan is legible from orbit, and
//! all of that legibility is in three elements: a **runway**, which is longer
//! and straighter than anything else people build; a **taxiway** parallel to
//! it; and an **apron** with aircraft parked nose-in along one edge. Terminal,
//! tower and car park are dressing. The markings matter more than the
//! buildings — a strip of grey with a dashed centreline and paired threshold
//! bars is unmistakably a runway, and the same strip without them is a road.
#![allow(dead_code)]

use super::*;

/// A parked airliner, nose-in to a stand.
///
/// Built here rather than reusing the flying model because the flying one is
/// sized to be seen from a kilometre below and this one is seen from the
/// apron, and because a parked aircraft needs its undercarriage down and its
/// engines under the wing rather than a silhouette.
fn airliner(b: &mut Batches, x: f32, z: f32, yaw: f32, len: f32, tail: Color, rng: &mut Rng) {
    let (sa, ca) = yaw.sin_cos();
    let at = |d: f32, s: f32| (x + ca * d - sa * s, z + sa * d + ca * s);
    let r = len * 0.055;
    let y = KERB + r + 2.2;
    let white = Color::from_hex(0xe8eaec);
    // Fuselage, with a tapered tail cone.
    let (n0x, n0z) = at(-len * 0.5, 0.0);
    let (n1x, n1z) = at(len * 0.34, 0.0);
    b.trim
        .add_limb(Vector3::new(n0x, y, n0z), Vector3::new(n1x, y, n1z), r, r, 9, white, true);
    let (t0x, t0z) = at(len * 0.34, 0.0);
    let (t1x, t1z) = at(len * 0.5, 0.0);
    b.trim.add_limb(
        Vector3::new(t0x, y, t0z),
        Vector3::new(t1x, y + r * 0.9, t1z),
        r,
        r * 0.18,
        8,
        white,
        true,
    );
    // Wings, swept back, with an engine slung under each.
    for s in [-1.0f32, 1.0] {
        let (w0x, w0z) = at(-len * 0.04, 0.0);
        let (w1x, w1z) = at(len * 0.13, s * len * 0.46);
        b.trim.add_limb(
            Vector3::new(w0x, y - r * 0.3, w0z),
            Vector3::new(w1x, y + r * 0.1, w1z),
            r * 0.55,
            r * 0.12,
            5,
            white,
            false,
        );
        let (e0x, e0z) = at(-len * 0.02, s * len * 0.20);
        let (e1x, e1z) = at(len * 0.07, s * len * 0.20);
        b.trim.add_limb(
            Vector3::new(e0x, y - r * 0.85, e0z),
            Vector3::new(e1x, y - r * 0.85, e1z),
            r * 0.36,
            r * 0.34,
            7,
            scale_color(white, 0.9),
            true,
        );
        // Main gear.
        let (gx, gz) = at(len * 0.02, s * len * 0.06);
        b.trim.add_limb(
            Vector3::new(gx, KERB, gz),
            Vector3::new(gx, y - r, gz),
            r * 0.10,
            r * 0.10,
            4,
            Color::from_hex(0x3a3f45),
            false,
        );
    }
    // Nose gear.
    let (gx, gz) = at(-len * 0.36, 0.0);
    b.trim.add_limb(
        Vector3::new(gx, KERB, gz),
        Vector3::new(gx, y - r, gz),
        r * 0.08,
        r * 0.08,
        4,
        Color::from_hex(0x3a3f45),
        false,
    );
    // Fin, in the airline's colour — the only part of an airliner that is not
    // white, and the only part anybody identifies it by.
    let (f0x, f0z) = at(len * 0.34, 0.0);
    let (f1x, f1z) = at(len * 0.47, 0.0);
    b.trim.add_limb(
        Vector3::new(f0x, y + r * 0.6, f0z),
        Vector3::new(f1x, y + r * 3.4, f1z),
        r * 0.7,
        r * 0.34,
        4,
        tail,
        false,
    );
    // Tailplane.
    for s in [-1.0f32, 1.0] {
        let (h0x, h0z) = at(len * 0.42, 0.0);
        let (h1x, h1z) = at(len * 0.48, s * len * 0.15);
        b.trim.add_limb(
            Vector3::new(h0x, y + r * 0.8, h0z),
            Vector3::new(h1x, y + r * 1.0, h1z),
            r * 0.3,
            r * 0.08,
            4,
            white,
            false,
        );
    }
    let _ = rng;
}

/// The airport. Returns how many aircraft are on the ground.
/// The ground an airfield takes: runway, taxiway, apron and the mown grass
/// round them, as `[x0, z0, x1, z1]`.
///
/// Public because the railway needs it. The line is laid along a street and
/// runs the full width of the map, so unless it is told where the airfield is
/// it puts a viaduct straight down the runway — which is exactly what it did,
/// because nothing outside this file knew how much ground the airport claimed.
pub(crate) fn airport_field(cx: f32, cz: f32, along_x: bool, rw_len: f32) -> [f32; 4] {
    let field_w = (rw_len * 0.21).clamp(190.0, 300.0);
    let (hw, hd) = if along_x {
        (rw_len * 0.5 + 40.0, field_w * 0.5)
    } else {
        (field_w * 0.5, rw_len * 0.5 + 40.0)
    };
    [cx - hw, cz - hd, cx + hw, cz + hd]
}

pub(crate) fn add_airport(
    b: &mut Batches,
    cx: f32,
    cz: f32,
    along_x: bool,
    rw_len: f32,
    rng: &mut Rng,
) -> usize {
    let rw_w = 45.0f32;
    let field = airport_field(cx, cz, along_x, rw_len);
    let (hw, hd) = ((field[2] - field[0]) * 0.5, (field[3] - field[1]) * 0.5);
    if !b.occ.try_claim(field) {
        return 0;
    }
    // Airfield grass: mown, and a different green from farmland.
    b.grass.add_slab(
        cx - hw,
        cz - hd,
        cx + hw,
        cz + hd,
        0.07,
        scale_color(Color::from_hex(0x74964a), rng.range(0.96, 1.04)),
        Uv::Unit,
    );
    // `s` runs along the runway, `t` across it.
    let at = |s: f32, t: f32| -> (f32, f32) {
        if along_x {
            (cx + s, cz + t)
        } else {
            (cx + t, cz + s)
        }
    };
    let strip = |b: &mut Batches, s0: f32, s1: f32, t0: f32, t1: f32, y: f32, c: Color, paint: bool| {
        let (a0, a1) = at(s0, t0);
        let (c0, c1) = at(s1, t1);
        let mb = if paint { &mut b.paint } else { &mut b.road };
        mb.add_slab(a0.min(c0), a1.min(c1), a0.max(c0), a1.max(c1), y, c, Uv::Unit);
    };

    let asphalt = Color::from_hex(0x3c3f45);
    let white = Color::from_hex(0xdedacb);
    // --- Runway.
    strip(b, -rw_len * 0.5, rw_len * 0.5, -rw_w * 0.5, rw_w * 0.5, 0.10, asphalt, false);
    // Centreline: long dashes, widely spaced. This is the marking that says
    // "runway" rather than "road", because no road has dashes this long.
    let n = (rw_len / 60.0) as usize;
    for k in 0..n {
        let s0 = -rw_len * 0.5 + k as f32 * 60.0 + 12.0;
        strip(b, s0, s0 + 30.0, -0.45, 0.45, 0.12, white, true);
    }
    // Edge lines.
    for s2 in [-1.0f32, 1.0] {
        strip(
            b,
            -rw_len * 0.5,
            rw_len * 0.5,
            s2 * (rw_w * 0.5 - 0.9),
            s2 * (rw_w * 0.5 - 0.3),
            0.12,
            white,
            true,
        );
    }
    // Threshold bars at both ends: the piano keys.
    for end in [-1.0f32, 1.0] {
        for k in 0..8 {
            let t = (k as f32 - 3.5) * 3.2;
            let s0 = end * (rw_len * 0.5 - 34.0);
            let s1 = end * (rw_len * 0.5 - 8.0);
            strip(b, s0.min(s1), s0.max(s1), t - 1.1, t + 1.1, 0.12, white, true);
        }
        // Touchdown zone markers.
        for k in 1..4 {
            let s0 = end * (rw_len * 0.5 - 60.0 - k as f32 * 45.0);
            for s2 in [-1.0f32, 1.0] {
                strip(b, s0 - 12.0, s0 + 12.0, s2 * 3.0, s2 * 5.4, 0.12, white, true);
            }
        }
        // Approach lighting off the end, on stalks over the grass.
        for k in 1..7 {
            let (lx, lz) = at(end * (rw_len * 0.5 + k as f32 * 22.0), 0.0);
            b.trim.add_limb(
                Vector3::new(lx, 0.07, lz),
                Vector3::new(lx, 1.4, lz),
                0.07,
                0.07,
                4,
                Color::from_hex(0x6f747a),
                false,
            );
            // The housing is `trim` so it survives daylight, and only the
            // lens is emissive. On `neon` the whole bar went black every hour
            // the sun was up, because that batch is a Basic material scaled by
            // `sky.lights` and drawn regardless — it does not hide, it
            // blackens. `glow` is the batch that actually hides.
            b.trim.add_box(
                Vector3::new(lx - 1.4, 1.4, lz - 0.2),
                Vector3::new(lx + 1.4, 1.7, lz + 0.2),
                Color::from_hex(0xd9d4c6),
                Uv::Unit,
            );
            b.glow.add_box(
                Vector3::new(lx - 1.4, 1.44, lz - 0.24),
                Vector3::new(lx + 1.4, 1.66, lz + 0.24),
                Color::from_hex(0xfff0c8),
                Uv::Unit,
            );
        }
    }
    // Runway edge lights, blue, all the way down both sides.
    for s2 in [-1.0f32, 1.0] {
        let m = (rw_len / 55.0) as usize;
        for k in 0..=m {
            let s0 = -rw_len * 0.5 + k as f32 * 55.0;
            let (lx, lz) = at(s0, s2 * (rw_w * 0.5 + 2.5));
            // Same split as the approach lights: a pale fitting that reads by
            // day, a blue lens that only exists at night.
            b.trim.add_box(
                Vector3::new(lx - 0.28, 0.10, lz - 0.28),
                Vector3::new(lx + 0.28, 0.44, lz + 0.28),
                Color::from_hex(0xc8ccd2),
                Uv::Unit,
            );
            b.glow.add_box(
                Vector3::new(lx - 0.24, 0.40, lz - 0.24),
                Vector3::new(lx + 0.24, 0.58, lz + 0.24),
                Color::from_hex(0x5aa8ff),
                Uv::Unit,
            );
        }
    }

    // --- Taxiway, parallel, with two links to the runway.
    let tw_t = rw_w * 0.5 + 60.0;
    strip(b, -rw_len * 0.42, rw_len * 0.42, tw_t - 11.0, tw_t + 11.0, 0.10, asphalt, false);
    strip(b, -rw_len * 0.42, rw_len * 0.42, tw_t - 0.3, tw_t + 0.3, 0.12, Color::from_hex(0xd8b13a), true);
    for link in [-0.34f32, 0.34] {
        let s0 = rw_len * link;
        strip(b, s0 - 11.0, s0 + 11.0, rw_w * 0.5, tw_t - 11.0, 0.10, asphalt, false);
        strip(b, s0 - 0.3, s0 + 0.3, rw_w * 0.5, tw_t - 11.0, 0.12, Color::from_hex(0xd8b13a), true);
    }

    // --- Apron and terminal.
    let ap_t0 = tw_t + 11.0;
    let ap_t1 = ap_t0 + 95.0;
    strip(b, -rw_len * 0.26, rw_len * 0.26, ap_t0, ap_t1, 0.10, Color::from_hex(0x8b8880), false);
    // Terminal along the back of the apron: a long low block with a glazed
    // face and a saw-tooth roof.
    let (tm0x, tm0z) = at(-rw_len * 0.22, ap_t1 - 34.0);
    let (tm1x, tm1z) = at(rw_len * 0.22, ap_t1 - 2.0);
    let term = Rect {
        x0: tm0x.min(tm1x),
        z0: tm0z.min(tm1z),
        x1: tm0x.max(tm1x),
        z1: tm0z.max(tm1z),
    };
    b.trim.add_box(
        Vector3::new(term.x0, KERB, term.z0),
        Vector3::new(term.x1, KERB + 13.0, term.z1),
        Color::from_hex(0xd4d0c8),
        Uv::Unit,
    );
    b.trim.add_box(
        Vector3::new(term.x0 - 0.3, KERB + 13.0, term.z0 - 0.3),
        Vector3::new(term.x1 + 0.3, KERB + 14.4, term.z1 + 0.3),
        Color::from_hex(0x6f747a),
        Uv::Unit,
    );
    // Glazing on the apron side.
    if along_x {
        b.trim.add_box(
            Vector3::new(term.x0 + 3.0, KERB + 1.5, term.z0 - 0.2),
            Vector3::new(term.x1 - 3.0, KERB + 11.0, term.z0 + 0.2),
            Color::from_hex(0x2a3238),
            Uv::Unit,
        );
    } else {
        b.trim.add_box(
            Vector3::new(term.x0 - 0.2, KERB + 1.5, term.z0 + 3.0),
            Vector3::new(term.x0 + 0.2, KERB + 11.0, term.z1 - 3.0),
            Color::from_hex(0x2a3238),
            Uv::Unit,
        );
    }
    // Control tower: the tallest thing on an airfield and the one silhouette
    // that names it.
    let (cwx, cwz) = at(rw_len * 0.28, ap_t1 - 16.0);
    b.trim.add_limb(
        Vector3::new(cwx, KERB, cwz),
        Vector3::new(cwx, KERB + 34.0, cwz),
        3.4,
        2.6,
        10,
        Color::from_hex(0xc8c4bc),
        false,
    );
    b.trim.add_limb(
        Vector3::new(cwx, KERB + 34.0, cwz),
        Vector3::new(cwx, KERB + 40.0, cwz),
        5.2,
        4.4,
        10,
        Color::from_hex(0x2a3238),
        false,
    );
    b.trim.add_limb(
        Vector3::new(cwx, KERB + 40.0, cwz),
        Vector3::new(cwx, KERB + 41.6, cwz),
        5.4,
        4.0,
        10,
        Color::from_hex(0x8f8b83),
        true,
    );
    b.beacon.add_box(
        Vector3::new(cwx - 0.5, KERB + 41.6, cwz - 0.5),
        Vector3::new(cwx + 0.5, KERB + 43.0, cwz + 0.5),
        Color::from_hex(0xff5a3a),
        Uv::Unit,
    );
    add_cctv(
        b,
        Vector3::new(cwx, KERB + 33.0, cwz + 3.0),
        Vector3::new(0.0, 0.0, -1.0),
        Cctv::Ptz,
        rng,
    );

    // --- Stands: aircraft nose-in to the terminal, with a jetty each.
    const TAIL: [u32; 6] = [0x2f6ab5, 0xd04a3a, 0x3f8f5c, 0xd8a52e, 0x8f4a8a, 0x2a3238];
    let stands = 5;
    let mut aircraft = 0;
    for k in 0..stands {
        if rng.chance(0.20) {
            continue;
        }
        let s0 = mix(-rw_len * 0.19, rw_len * 0.19, (k as f32 + 0.5) / stands as f32);
        // Stand centreline lead-in.
        strip(b, s0 - 0.25, s0 + 0.25, ap_t0 + 6.0, ap_t1 - 36.0, 0.12, Color::from_hex(0xd8b13a), true);
        let (ax, az) = at(s0, ap_t1 - 52.0);
        let len = rng.range(34.0, 46.0);
        // Nose points at the terminal.
        let yaw = if along_x { PI * 0.5 } else { 0.0 };
        airliner(b, ax, az, yaw, len, Color::from_hex(TAIL[rng.below(TAIL.len())]), rng);
        aircraft += 1;
        // Air bridge from the terminal to the forward door.
        let (j0x, j0z) = at(s0 - 3.0, ap_t1 - 34.0);
        let (j1x, j1z) = at(s0 + 3.0, ap_t1 - 44.0);
        b.trim.add_box(
            Vector3::new(j0x.min(j1x), KERB + 3.4, j0z.min(j1z)),
            Vector3::new(j0x.max(j1x), KERB + 6.4, j0z.max(j1z)),
            Color::from_hex(0xb9bcc0),
            Uv::Unit,
        );
    }

    // --- Perimeter fence, and cameras on it. An airfield boundary is the most
    // consistently watched line in any city.
    let peri = 42;
    for k in 0..peri {
        let t = k as f32 / peri as f32 * 4.0;
        let (px, pz) = match t as usize {
            0 => (cx - hw, mix(cz - hd, cz + hd, t.fract())),
            1 => (mix(cx - hw, cx + hw, t.fract()), cz + hd),
            2 => (cx + hw, mix(cz + hd, cz - hd, t.fract())),
            _ => (mix(cx + hw, cx - hw, t.fract()), cz - hd),
        };
        b.trim.add_limb(
            Vector3::new(px, 0.07, pz),
            Vector3::new(px, 3.0, pz),
            0.07,
            0.06,
            4,
            Color::from_hex(0x6f747a),
            false,
        );
        if k % 7 == 0 {
            add_cctv(
                b,
                Vector3::new(px, 2.8, pz),
                Vector3::new(cx - px, 0.0, cz - pz),
                Cctv::Bullet,
                rng,
            );
        }
    }
    aircraft
}
