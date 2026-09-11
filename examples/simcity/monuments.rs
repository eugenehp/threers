//! Part of the `simcity` example; see `mod.rs`.
//!
//! Monuments: the buildings a city is recognised by.
//!
//! Everything else in this generator comes from a rule — a height falloff, a
//! parcel subdivision, a facade idiom — and rules produce *typical* buildings
//! by construction. A skyline made only of typical buildings has no landmark
//! in it, and a city with no landmark is a city nobody can describe. These are
//! the exceptions: each is built by hand, each is placed once, and each is
//! deliberately unlike anything the rules can make.
//!
//! What they have in common is that their silhouette is legible at any size.
//! A lattice tower is four curving legs and two platforms; a colossus is a
//! figure with one arm up; a suspension bridge is two towers and a curve. You
//! can recognise all three at forty pixels, which is the only test that
//! matters for something meant to be seen from across a city.
#![allow(dead_code)]

use super::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Monument {
    /// A wrought-iron lattice tower on four splayed legs.
    LatticeTower,
    /// A colossal robed figure on a plinth, one arm raised with a torch.
    Colossus,
    /// A triumphal arch on a roundabout.
    Arch,
    /// An observation wheel on the waterfront.
    Wheel,
    /// An obelisk on a paved square.
    Obelisk,
    /// A domed rotunda.
    Dome,
}

pub(crate) const MONUMENTS: [Monument; 6] = [
    Monument::LatticeTower,
    Monument::Colossus,
    Monument::Arch,
    Monument::Wheel,
    Monument::Obelisk,
    Monument::Dome,
];

impl Monument {
    /// Ground it needs, as a half-extent.
    pub(crate) fn footprint(self) -> f32 {
        match self {
            Monument::LatticeTower => 42.0,
            Monument::Colossus => 30.0,
            Monument::Arch => 26.0,
            Monument::Wheel => 44.0,
            Monument::Obelisk => 22.0,
            Monument::Dome => 34.0,
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Monument::LatticeTower => "lattice tower",
            Monument::Colossus => "colossus",
            Monument::Arch => "triumphal arch",
            Monument::Wheel => "observation wheel",
            Monument::Obelisk => "obelisk",
            Monument::Dome => "rotunda",
        }
    }
}

/// A square lattice bay: four uprights and a cross-brace on each face.
///
/// The whole tower is this repeated at shrinking radii, which is both how the
/// real thing is built and why it reads as ironwork rather than as a solid
/// pyramid — you can see sky through it.
fn lattice_bay(
    b: &mut Batches,
    cx: f32,
    cz: f32,
    y0: f32,
    y1: f32,
    r0: f32,
    r1: f32,
    iron: Color,
) {
    let corners = |r: f32| {
        [
            (cx - r, cz - r),
            (cx + r, cz - r),
            (cx + r, cz + r),
            (cx - r, cz + r),
        ]
    };
    let (c0, c1) = (corners(r0), corners(r1));
    for k in 0..4 {
        // Upright.
        b.trim.add_limb(
            Vector3::new(c0[k].0, y0, c0[k].1),
            Vector3::new(c1[k].0, y1, c1[k].1),
            r0 * 0.055 + 0.10,
            r1 * 0.055 + 0.09,
            4,
            iron,
            false,
        );
        // Cross-brace on this face.
        let j = (k + 1) % 4;
        b.trim.add_limb(
            Vector3::new(c0[k].0, y0, c0[k].1),
            Vector3::new(c1[j].0, y1, c1[j].1),
            0.10,
            0.09,
            3,
            iron,
            false,
        );
        b.trim.add_limb(
            Vector3::new(c0[j].0, y0, c0[j].1),
            Vector3::new(c1[k].0, y1, c1[k].1),
            0.10,
            0.09,
            3,
            iron,
            false,
        );
        // Horizontal tie at the top of the bay.
        b.trim.add_limb(
            Vector3::new(c1[k].0, y1, c1[k].1),
            Vector3::new(c1[j].0, y1, c1[j].1),
            0.12,
            0.12,
            3,
            iron,
            false,
        );
    }
}

/// The lattice tower.
fn lattice_tower(b: &mut Batches, cx: f32, cz: f32, rng: &mut Rng) {
    let h = 172.0f32;
    let iron = Color::from_hex(0x7a6a52);
    // Base plinths under each leg.
    for (sx, sz) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        b.trim.add_box(
            Vector3::new(cx + sx * 26.0 - 4.5, KERB, cz + sz * 26.0 - 4.5),
            Vector3::new(cx + sx * 26.0 + 4.5, KERB + 3.0, cz + sz * 26.0 + 4.5),
            Color::from_hex(0x8e8b83),
            Uv::Unit,
        );
    }
    // The legs: bays whose radius falls off as a curve, not a straight taper.
    // The curve is the tower — a straight-sided lattice pyramid reads as a
    // pylon, and the difference is entirely in this exponent.
    const BAYS: usize = 16;
    let radius = |t: f32| 26.0 * (1.0 - t).powf(1.9) + 3.4;
    let first = 0.40f32; // the first platform
    let second = 0.66f32;
    for k in 0..BAYS {
        let (t0, t1) = (k as f32 / BAYS as f32, (k + 1) as f32 / BAYS as f32);
        lattice_bay(
            b,
            cx,
            cz,
            KERB + h * t0,
            KERB + h * t1,
            radius(t0),
            radius(t1),
            iron,
        );
    }
    // Two platforms, each a deck with a balustrade — the horizontals that stop
    // it being one continuous taper.
    for t in [first, second] {
        let r = radius(t) + 3.0;
        b.trim.add_box(
            Vector3::new(cx - r, KERB + h * t, cz - r),
            Vector3::new(cx + r, KERB + h * t + 0.7, cz + r),
            Color::from_hex(0x8a7a60),
            Uv::Unit,
        );
        for s in [-1.0f32, 1.0] {
            b.trim.add_box(
                Vector3::new(cx - r, KERB + h * t + 0.7, cz + s * r - 0.2),
                Vector3::new(cx + r, KERB + h * t + 2.0, cz + s * r + 0.2),
                iron,
                Uv::Unit,
            );
            b.trim.add_box(
                Vector3::new(cx + s * r - 0.2, KERB + h * t + 0.7, cz - r),
                Vector3::new(cx + s * r + 0.2, KERB + h * t + 2.0, cz + r),
                iron,
                Uv::Unit,
            );
        }
    }
    // The arches between the legs at ground level: the other thing everyone
    // draws when they draw one of these.
    for (ax, az, bx2, bz2) in [
        (-26.0f32, -26.0f32, 26.0f32, -26.0f32),
        (26.0, -26.0, 26.0, 26.0),
        (26.0, 26.0, -26.0, 26.0),
        (-26.0, 26.0, -26.0, -26.0),
    ] {
        const SEG: usize = 8;
        for k in 0..SEG {
            let (u0, u1) = (k as f32 / SEG as f32, (k + 1) as f32 / SEG as f32);
            let arc = |u: f32| {
                let th = PI * u;
                (
                    cx + mix(ax, bx2, u),
                    KERB + 20.0 + 16.0 * th.sin(),
                    cz + mix(az, bz2, u),
                )
            };
            let (p0x, p0y, p0z) = arc(u0);
            let (p1x, p1y, p1z) = arc(u1);
            b.trim.add_limb(
                Vector3::new(p0x, p0y, p0z),
                Vector3::new(p1x, p1y, p1z),
                0.9,
                0.9,
                4,
                iron,
                false,
            );
        }
    }
    // Lantern and mast.
    b.trim.add_limb(
        Vector3::new(cx, KERB + h, cz),
        Vector3::new(cx, KERB + h + 6.0, cz),
        3.2,
        2.4,
        8,
        iron,
        false,
    );
    b.trim.add_limb(
        Vector3::new(cx, KERB + h + 6.0, cz),
        Vector3::new(cx, KERB + h + 26.0, cz),
        0.6,
        0.2,
        6,
        Color::from_hex(0x9aa0a6),
        false,
    );
    b.beacon.add_box(
        Vector3::new(cx - 0.6, KERB + h + 26.0, cz - 0.6),
        Vector3::new(cx + 0.6, KERB + h + 27.4, cz + 0.6),
        Color::from_hex(0xff5a3a),
        Uv::Unit,
    );
    let _ = rng;
}

/// A lofted body of revolution with a scalloped section: drapery.
///
/// The robe was a smooth cone with a few thin ribs laid against it, and thin
/// ribs on a smooth cone read as a smooth cone with wires on. Cloth at this
/// scale is not decoration applied to a surface — the folds *are* the surface,
/// so the section itself has to be scalloped and the loft carries it up.
///
/// `lean` shifts each ring sideways as it rises, which is contrapposto: the
/// figure stands on one leg and the hip and shoulder go opposite ways. Costs
/// nothing and is the difference between a person standing and a bollard.
#[allow(clippy::too_many_arguments)]
fn draped(
    b: &mut Batches,
    cx: f32,
    cz: f32,
    base: f32,
    height: f32,
    profile: &[(f32, f32)],
    flutes: usize,
    depth: f32,
    lean: f32,
    yaw: f32,
    col: Color,
) {
    // Resolution. This is one object in the entire city, so segments here are
    // free in a way they are nowhere else: the whole statue at this density is
    // a few thousand triangles against nearly two million in the scene.
    const N: usize = 56;
    // Rows *between* control points. The profile is a dozen measurements down
    // a human figure; lofting straight between them makes the silhouette a
    // polyline, and the eye reads the kinks at the waist and shoulder as
    // faceting. Sampling it as a spline is what makes the outline a curve.
    const SUB: usize = 5;

    // Catmull-Rom through the control radii, clamped at the ends. This is the
    // "precision" half: the shape is now defined by the curve the control
    // points describe rather than by the segments between them.
    let radius_at = |t: f32| -> f32 {
        let n = profile.len();
        // Find the span.
        let mut i = 0;
        while i + 2 < n && profile[i + 1].0 < t {
            i += 1;
        }
        let (t1, r1) = profile[i];
        let (t2, r2) = profile[(i + 1).min(n - 1)];
        let (_, r0) = profile[i.saturating_sub(1)];
        let (_, r3) = profile[(i + 2).min(n - 1)];
        let u = if (t2 - t1).abs() < 1e-6 {
            0.0
        } else {
            ((t - t1) / (t2 - t1)).clamp(0.0, 1.0)
        };
        let u2 = u * u;
        let u3 = u2 * u;
        0.5 * ((2.0 * r1)
            + (-r0 + r2) * u
            + (2.0 * r0 - 5.0 * r1 + 4.0 * r2 - r3) * u2
            + (-r0 + 3.0 * r1 - 3.0 * r2 + r3) * u3)
    };

    let ring = |t: f32| -> Vec<[f32; 3]> {
        let r = radius_at(t);
        // The lean is strongest at the hem and dies away at the shoulders, so
        // the figure sways rather than tilts.
        let off = lean * (1.0 - t) * t * 4.0;
        (0..=N)
            .map(|k| {
                let th = k as f32 / N as f32 * TAU;
                // Deeper folds low down, shallowing as the cloth is drawn in.
                let d = depth * (1.0 - t * 0.45);
                let rr = r * (1.0 + d * (flutes as f32 * th).cos());
                [
                    cx + rr * th.cos() + off * yaw.cos(),
                    base + height * t,
                    cz + rr * th.sin() + off * yaw.sin(),
                ]
            })
            .collect()
    };

    // Walk the whole profile at SUB rows per span.
    let mut ts: Vec<f32> = Vec::new();
    for w in profile.windows(2) {
        for k in 0..SUB {
            ts.push(mix(w[0].0, w[1].0, k as f32 / SUB as f32));
        }
    }
    ts.push(profile[profile.len() - 1].0);

    for pair in ts.windows(2) {
        let (a, c) = (ring(pair[0]), ring(pair[1]));
        for k in 0..N {
            let th = (k as f32 + 0.5) / N as f32 * TAU;
            // Normal follows the flute, not just the ring: on a fluted surface
            // the facets face sideways as much as outwards, and shading them
            // as a plain cylinder flattens the folds back out.
            let dr = -depth * flutes as f32 * (flutes as f32 * th).sin();
            let nl = (1.0 + dr * dr).sqrt();
            b.trim.quad(
                [a[k], c[k], c[k + 1], a[k + 1]],
                [
                    (th.cos() - dr * th.sin()) / nl,
                    0.20,
                    (th.sin() + dr * th.cos()) / nl,
                ],
                Uv::Unit,
                col,
            );
        }
    }
}

/// The colossus: a robed figure on a stepped plinth, one arm raised.
///
/// The first attempt was a cone with a ball on top and the arm leaving the
/// crown, which read as a green traffic cone with a spear through it. What was
/// missing was not detail — it was *anatomy*. Three things do all the work at
/// this scale and none of them is the face:
///
/// - a **waist**. The robe has to pinch in and flare back out; a monotonic
///   taper from hem to head is a cone whatever colour it is.
/// - **shoulders** wider than the neck, with the head sitting above them
///   rather than on the point of the body.
/// - the raised arm leaving a **shoulder**, offset to one side, not the
///   centreline. That single offset is what makes the pose read as a person
///   holding something up.
fn colossus(b: &mut Batches, cx: f32, cz: f32, yaw: f32, rng: &mut Rng) {
    let bronze = Color::from_hex(0x7fae9a);
    let stone = Color::from_hex(0x9a968e);
    let (sa, ca) = yaw.sin_cos();
    // Local frame: `d` is forward (the way it faces), `s` is to its left.
    let at = |d: f32, s: f32| (cx + ca * d - sa * s, cz + sa * d + ca * s);

    // --- The pedestal. Half the height of the whole thing, which is the
    // proportion the real one has and the reason it reads as monumental
    // rather than as a statue on a box.
    // An eleven-point star fort.
    for k in 0..11 {
        let a = k as f32 / 11.0 * TAU;
        let (px, pz) = (cx + 19.0 * a.cos(), cz + 19.0 * a.sin());
        b.trim.add_limb(
            Vector3::new(px, KERB, pz),
            Vector3::new(px, KERB + 4.5, pz),
            7.2,
            6.4,
            4,
            scale_color(stone, 0.9),
            false,
        );
    }
    // Battered plinth, then the pedestal proper with a cornice top and bottom.
    let ped = |b: &mut Batches, w: f32, h0: f32, h1: f32, c: Color| {
        b.trim.add_box(
            Vector3::new(cx - w, KERB + h0, cz - w),
            Vector3::new(cx + w, KERB + h1, cz + w),
            c,
            Uv::Unit,
        );
    };
    ped(b, 14.5, 4.5, 8.0, scale_color(stone, 0.95));
    ped(b, 13.0, 8.0, 10.5, scale_color(stone, 1.05)); // lower cornice
    ped(b, 11.0, 10.5, 27.0, stone); // the shaft
    ped(b, 12.4, 27.0, 30.0, scale_color(stone, 1.06)); // upper cornice
    ped(b, 9.6, 30.0, 34.0, scale_color(stone, 0.98)); // attic
    // Loggia: recessed vertical openings round the shaft, which is what stops
    // the pedestal being a plain block.
    for face in 0..4 {
        let ang = face as f32 * PI * 0.5;
        let (fx, fz) = (ang.cos(), ang.sin());
        let (px, pz) = (-fz, fx);
        for k in 0..5 {
            let u = (k as f32 - 2.0) * 3.6;
            b.trim.add_box(
                Vector3::new(
                    cx + fx * 10.6 + px * u - 1.0,
                    KERB + 13.0,
                    cz + fz * 10.6 + pz * u - 1.0,
                ),
                Vector3::new(
                    cx + fx * 11.2 + px * u + 1.0,
                    KERB + 24.0,
                    cz + fz * 11.2 + pz * u + 1.0,
                ),
                Color::from_hex(0x4a4740),
                Uv::Unit,
            );
        }
    }

    let foot = KERB + 34.0;
    let fh = 36.0f32;
    // The robe. Asymmetric hem, a waist, and a chest — and now *fluted*, so it
    // is cloth rather than a cone.
    const PROFILE: [(f32, f32); 12] = [
        (0.000, 7.2),  // hem
        (0.070, 6.9),
        (0.180, 6.5),
        (0.330, 6.0),  // knees
        (0.500, 5.3),  // hips
        (0.610, 4.7),  // waist
        (0.710, 5.1),  // chest
        (0.780, 5.3),  // shoulders
        (0.812, 4.4),
        (0.848, 2.7),
        (0.888, 1.5),  // neck
        (0.930, 1.4),
    ];
    draped(b, cx, cz, foot, fh, &PROFILE, 11, 0.075, 0.9, yaw, bronze);
    // The hem sweeps back on one side and forward on the other, so the figure
    // is striding rather than standing to attention.
    for k in 0..14 {
        let th = k as f32 / 14.0 * TAU;
        let sweep = 1.0 + 0.22 * (th - yaw + PI * 0.5).cos();
        let r = 7.2 * (1.0 + 0.075 * (11.0 * th).cos());
        b.trim.add_limb(
            Vector3::new(cx + r * th.cos(), foot, cz + r * th.sin()),
            Vector3::new(
                cx + r * sweep * th.cos(),
                foot - 1.6 * sweep,
                cz + r * sweep * th.sin(),
            ),
            0.9,
            0.7,
            4,
            scale_color(bronze, 0.94),
            false,
        );
    }
    // The stola: a sash over one shoulder and across the body.
    for k in 0..7 {
        let t = k as f32 / 6.0;
        let (px, pz) = at(mix(1.0, -0.6, t) * 3.4, mix(-3.6, 3.2, t));
        b.trim.add_limb(
            Vector3::new(px, foot + fh * mix(0.775, 0.60, t), pz),
            Vector3::new(px, foot + fh * mix(0.775, 0.60, t) - 0.9, pz),
            1.05,
            0.95,
            4,
            scale_color(bronze, 1.07),
            false,
        );
    }

    let shoulder = foot + fh * 0.78;
    for s2 in [-1.0f32, 1.0] {
        let (kx, kz) = at(0.0, s2 * 4.4);
        b.trim
            .add_ellipsoid(Vector3::new(kx, shoulder + 0.2, kz), 1.9, 1.6, 1.9, 6, 18, bronze);
    }

    // Head, above the shoulders on a real neck.
    let head_y = foot + fh * 1.005;
    b.trim
        .add_ellipsoid(Vector3::new(cx, head_y, cz), 2.7, 3.3, 2.8, 8, 22, bronze);
    // A brow and a nose. Two primitives, and at any distance where you can see
    // the head at all they are the difference between a face and a ball.
    let (bx0, bz0) = at(2.2, 0.0);
    b.trim.add_yaw_box(
        Vector3::new(bx0, head_y + 0.75, bz0),
        Vector3::new(0.45, 0.35, 1.7),
        yaw,
        scale_color(bronze, 0.9),
        Uv::Unit,
    );
    let (nx0, nz0) = at(2.5, 0.0);
    b.trim.add_limb(
        Vector3::new(nx0, head_y + 0.55, nz0),
        Vector3::new(nx0 + ca * 0.5, head_y - 0.45, nz0 + sa * 0.5),
        0.42,
        0.30,
        4,
        bronze,
        false,
    );
    // Hair, gathered at the back.
    let (hx0, hz0) = at(-1.5, 0.0);
    b.trim
        .add_ellipsoid(Vector3::new(hx0, head_y - 0.6, hz0), 1.7, 1.9, 2.4, 6, 18, scale_color(bronze, 0.93));
    // Crown: a diadem round the forehead, narrower than the head, and seven
    // long rays.
    b.trim.add_limb(
        Vector3::new(cx, head_y + 0.35, cz),
        Vector3::new(cx, head_y + 1.25, cz),
        2.3,
        2.15,
        22,
        scale_color(bronze, 1.08),
        false,
    );
    for k in 0..7 {
        let a = yaw + (k as f32 / 6.0 - 0.5) * PI * 1.55;
        let (u0, v0) = (a.cos(), a.sin());
        // Tapered blades, not spikes: a ray is a flat plate on edge.
        b.trim.add_limb(
            Vector3::new(cx + u0 * 2.1, head_y + 1.1, cz + v0 * 2.1),
            Vector3::new(cx + u0 * 7.6, head_y + 6.6, cz + v0 * 7.6),
            0.52,
            0.09,
            7,
            bronze,
            false,
        );
    }

    // The raised arm: from the right shoulder, high and slightly forward.
    let (sx, sz) = at(0.2, -4.4);
    let (ex, ez) = at(1.0, -5.8);
    let (hx, hz) = at(1.9, -5.2);
    b.trim.add_limb(
        Vector3::new(sx, shoulder + 0.5, sz),
        Vector3::new(ex, shoulder + 9.5, ez),
        1.55,
        1.15,
        14,
        bronze,
        false,
    );
    // A sleeve falling from the upper arm, which is the detail that keeps the
    // arm from looking like a pipe.
    b.trim.add_limb(
        Vector3::new(sx, shoulder + 1.6, sz),
        Vector3::new(sx - ca * 0.4, shoulder - 2.6, sz - sa * 0.4),
        2.1,
        1.3,
        14,
        scale_color(bronze, 0.95),
        false,
    );
    b.trim.add_limb(
        Vector3::new(ex, shoulder + 9.5, ez),
        Vector3::new(hx, shoulder + 18.0, hz),
        1.15,
        0.92,
        14,
        bronze,
        false,
    );
    b.trim
        .add_ellipsoid(Vector3::new(hx, shoulder + 18.7, hz), 1.15, 1.3, 1.15, 6, 16, bronze);
    // Torch: a fluted handle, a bowl, and a flame that is lit by day and
    // glows by night.
    b.trim.add_limb(
        Vector3::new(hx, shoulder + 18.4, hz),
        Vector3::new(hx, shoulder + 20.2, hz),
        0.75,
        1.05,
        18,
        Color::from_hex(0xc9a352),
        false,
    );
    b.trim.add_limb(
        Vector3::new(hx, shoulder + 20.2, hz),
        Vector3::new(hx, shoulder + 21.4, hz),
        2.05,
        1.85,
        20,
        Color::from_hex(0xd8b466),
        true,
    );
    b.trim.add_ellipsoid(
        Vector3::new(hx, shoulder + 23.0, hz),
        1.85,
        2.7,
        1.85,
        7,
        18,
        Color::from_hex(0xe8c46a),
    );
    // The emissive shell goes *outside* the lit one, not inside it — inside,
    // the lit flame occludes the glow completely and the torch is merely a
    // gold shape after dark.
    //
    // And it goes on `glow`, not `neon`. `neon` is not hidden in daylight: it
    // is a `Basic` material whose colour is scaled by `sky.lights`, so by day
    // it is still drawn and simply renders **black**. An oversized neon shell
    // therefore put a black box on top of the torch at noon — which is exactly
    // the fault I had already fixed once here and reintroduced by enlarging
    // the same shape. `glow` is in `night_only` and is genuinely hidden.
    b.glow.add_ellipsoid(
        Vector3::new(hx, shoulder + 23.0, hz),
        2.05,
        2.95,
        2.05,
        3,
        8,
        Color::from_hex(0xffe6a8),
    );

    // The left arm, bent, with the tablet resting on the forearm against the
    // body — not hanging from a hand.
    let (ax, az) = at(0.2, 4.4);
    let (elx, elz) = at(0.4, 5.2);
    let (wx, wz) = at(3.0, 2.6);
    b.trim.add_limb(
        Vector3::new(ax, shoulder + 0.2, az),
        Vector3::new(elx, shoulder - 6.4, elz),
        1.5,
        1.1,
        14,
        bronze,
        false,
    );
    b.trim.add_limb(
        Vector3::new(elx, shoulder - 6.4, elz),
        Vector3::new(wx, shoulder - 4.6, wz),
        1.1,
        0.85,
        14,
        bronze,
        false,
    );
    b.trim.add_limb(
        Vector3::new(ax, shoulder + 1.4, az),
        Vector3::new(ax + ca * 0.3, shoulder - 2.8, az + sa * 0.3),
        2.0,
        1.25,
        14,
        scale_color(bronze, 0.95),
        false,
    );
    // The tablet, tilted back against the chest.
    let (tx, tz) = at(3.4, 2.0);
    b.trim.add_yaw_box(
        Vector3::new(tx, shoulder - 3.4, tz),
        Vector3::new(0.6, 4.6, 3.1),
        yaw,
        scale_color(bronze, 0.96),
        Uv::Unit,
    );
    let _ = rng;
}

/// A triumphal arch.
///
/// The first version put the springing of the vault at three-quarters of the
/// height, which left an eight-metre slot between two tall blocks with a
/// separate lump of arch sitting on top of them. An arch is not a hole with a
/// curve above it: the piers stop at the springing and the curve *is* the top
/// of the opening. Everything above that is wall.
fn arch(b: &mut Batches, cx: f32, cz: f32, yaw: f32, rng: &mut Rng) {
    let stone = scale_color(Color::from_hex(0xcfc4ac), rng.range(0.95, 1.05));
    let (w, d, h) = (30.0f32, 11.0f32, 36.0f32);
    let r = 6.5f32; // half-width of the opening
    let spring = 15.0f32; // where the vault starts
    let wall_top = h * 0.74; // where the entablature begins
    let (sa, ca) = yaw.sin_cos();
    let at = |dd: f32, s: f32| (cx + ca * dd - sa * s, cz + sa * dd + ca * s);

    // Paved setting, so it stands on something.
    b.pads
        .add_ground_disc(cx, cz, w * 0.85, 22, 0.0, KERB + 0.02, Color::from_hex(0xb0a894));
    // The two piers, from the ground to the springing.
    for s in [-1.0f32, 1.0] {
        let inner = s * r;
        let outer = s * w * 0.5;
        let (px, pz) = at(0.0, (inner + outer) * 0.5);
        // Full height, not just to the springing: the wall above the opening
        // is continuous masonry and modelling it in pieces is what produced a
        // staircase last time.
        b.trim.add_yaw_box(
            Vector3::new(px, KERB + wall_top * 0.5, pz),
            Vector3::new(d * 0.5, wall_top * 0.5, (outer - inner).abs() * 0.5),
            yaw,
            stone,
            Uv::Unit,
        );
        // A pilaster on each face, which is what a pier of this size has.
        for f in [-1.0f32, 1.0] {
            let (qx, qz) = at(f * (d * 0.5 + 0.35), s * (r + (w * 0.5 - r) * 0.55));
            b.trim.add_limb(
                Vector3::new(qx, KERB, qz),
                Vector3::new(qx, KERB + wall_top, qz),
                1.6,
                1.4,
                7,
                scale_color(stone, 1.05),
                false,
            );
        }
    }
    // The mass above the crown of the arch, full width of the opening.
    b.trim.add_yaw_box(
        Vector3::new(cx, KERB + (spring + r + wall_top) * 0.5, cz),
        Vector3::new(d * 0.5, (wall_top - spring - r) * 0.5, r),
        yaw,
        stone,
        Uv::Unit,
    );
    // The vault: a real semicircle.
    const SEG: usize = 12;
    for k in 0..SEG {
        let (u0, u1) = (k as f32 / SEG as f32, (k + 1) as f32 / SEG as f32);
        let p = |u: f32| {
            let th = PI * u;
            (-r * th.cos(), spring + r * th.sin())
        };
        let (s0, y0) = p(u0);
        let (s1, y1) = p(u1);
        // The barrel, spanning the depth of the arch.
        let (a0x, a0z) = at(-d * 0.5, s0);
        let (a1x, a1z) = at(d * 0.5, s0);
        let (b0x, b0z) = at(-d * 0.5, s1);
        let (b1x, b1z) = at(d * 0.5, s1);
        let nrm = {
            let (mx, my) = ((s0 + s1) * 0.5, (y0 + y1) * 0.5 - spring);
            let l = (mx * mx + my * my).sqrt().max(1e-4);
            [-(-sa * mx / l), -(my / l), -(ca * mx / l)]
        };
        b.trim.quad(
            [
                [a0x, KERB + y0, a0z],
                [b0x, KERB + y1, b0z],
                [b1x, KERB + y1, b1z],
                [a1x, KERB + y0, a1z],
            ],
            nrm,
            Uv::Unit,
            scale_color(stone, 0.88),
        );
        // Voussoirs on both faces, standing proud of the wall behind.
        for f in [-1.0f32, 1.0] {
            let (v0x, v0z) = at(f * (d * 0.5 + 0.3), s0);
            let (v1x, v1z) = at(f * (d * 0.5 + 0.3), s1);
            let (e0, e1) = (1.0 + 1.15 / r, 1.0 + 1.15 / r);
            let (w0x, w0z) = at(f * (d * 0.5 + 0.3), s0 * e0);
            let (w1x, w1z) = at(f * (d * 0.5 + 0.3), s1 * e1);
            b.trim.quad(
                [
                    [v0x, KERB + y0, v0z],
                    [w0x, KERB + spring + (y0 - spring) * e0, w0z],
                    [w1x, KERB + spring + (y1 - spring) * e1, w1z],
                    [v1x, KERB + y1, v1z],
                ],
                [ca * f, 0.0, sa * f],
                Uv::Unit,
                scale_color(stone, 1.05),
            );
        }
        // The lune: the sliver of wall between the curve and the vertical
        // line the piers stop at, on each face. Filling the whole spandrel
        // with one box per arch segment gave a staircase down both sides of
        // the opening — this follows the curve exactly and the mass above the
        // crown is a single block instead.
        let edge = if (s0 + s1) * 0.5 < 0.0 { -r } else { r };
        for f in [-1.0f32, 1.0] {
            let (p0x, p0z) = at(f * d * 0.5, s0);
            let (p1x, p1z) = at(f * d * 0.5, s1);
            let (e0x, e0z) = at(f * d * 0.5, edge);
            b.trim.quad(
                [
                    [p0x, KERB + y0, p0z],
                    [p1x, KERB + y1, p1z],
                    [e0x, KERB + y1, e0z],
                    [e0x, KERB + y0, e0z],
                ],
                [ca * f, 0.0, sa * f],
                Uv::Unit,
                stone,
            );
        }
    }
    // Entablature, attic and cornice.
    b.trim.add_yaw_box(
        Vector3::new(cx, KERB + wall_top + 1.4, cz),
        Vector3::new(d * 0.56, 1.4, w * 0.5 + 0.5),
        yaw,
        scale_color(stone, 1.06),
        Uv::Unit,
    );
    b.trim.add_yaw_box(
        Vector3::new(cx, KERB + wall_top + 5.4, cz),
        Vector3::new(d * 0.5, 2.6, w * 0.46),
        yaw,
        scale_color(stone, 0.98),
        Uv::Unit,
    );
    b.trim.add_yaw_box(
        Vector3::new(cx, KERB + h - 0.8, cz),
        Vector3::new(d * 0.58, 0.8, w * 0.5 + 0.8),
        yaw,
        scale_color(stone, 1.08),
        Uv::Unit,
    );
    // A quadriga: four horses abreast and a chariot behind them.
    for k in 0..4 {
        let (qx, qz) = at(-1.0, (k as f32 - 1.5) * 3.0);
        b.trim.add_limb(
            Vector3::new(qx, KERB + h, qz),
            Vector3::new(qx, KERB + h + 3.6, qz),
            1.15,
            0.85,
            6,
            Color::from_hex(0x6f6a5c),
            false,
        );
        let (hx, hz) = at(1.6, (k as f32 - 1.5) * 3.0);
        b.trim.add_limb(
            Vector3::new(qx, KERB + h + 3.2, qz),
            Vector3::new(hx, KERB + h + 4.6, hz),
            0.8,
            0.55,
            5,
            Color::from_hex(0x6f6a5c),
            false,
        );
    }
    let (chx, chz) = at(-4.4, 0.0);
    b.trim.add_yaw_box(
        Vector3::new(chx, KERB + h + 1.8, chz),
        Vector3::new(1.6, 1.8, 3.4),
        yaw,
        Color::from_hex(0x7a7466),
        Uv::Unit,
    );
}

/// An observation wheel.
fn wheel(b: &mut Batches, cx: f32, cz: f32, yaw: f32, rng: &mut Rng) {
    let r = 52.0f32;
    let hub = KERB + r + 8.0;
    let steel = Color::from_hex(0xd8dce0);
    let (sa, ca) = yaw.sin_cos();
    // The wheel stands in a plane; `n` is its normal.
    let (nx, nz) = (-sa, ca);
    let pt = |a: f32| {
        (
            cx + ca * r * a.cos(),
            hub + r * a.sin(),
            cz + sa * r * a.cos(),
        )
    };
    // A-frame legs, splayed across the plane of the wheel.
    for s in [-1.0f32, 1.0] {
        for d in [-1.0f32, 1.0] {
            b.trim.add_limb(
                Vector3::new(cx + nx * 16.0 * s + ca * d * 22.0, KERB, cz + nz * 16.0 * s + sa * d * 22.0),
                Vector3::new(cx + nx * 3.0 * s, hub, cz + nz * 3.0 * s),
                1.5,
                0.9,
                6,
                steel,
                false,
            );
        }
    }
    // Rim and spokes.
    const N: usize = 32;
    for k in 0..N {
        let (a0, a1) = (
            k as f32 / N as f32 * TAU,
            (k + 1) as f32 / N as f32 * TAU,
        );
        let (p0x, p0y, p0z) = pt(a0);
        let (p1x, p1y, p1z) = pt(a1);
        b.trim.add_limb(
            Vector3::new(p0x, p0y, p0z),
            Vector3::new(p1x, p1y, p1z),
            0.55,
            0.55,
            4,
            steel,
            false,
        );
        // Cable spokes to the hub, and a capsule hung outside the rim.
        b.trim.add_limb(
            Vector3::new(cx, hub, cz),
            Vector3::new(p0x, p0y, p0z),
            0.10,
            0.10,
            3,
            steel,
            false,
        );
        if k % 2 == 0 {
            let (gx, gy, gz) = (
                cx + ca * (r + 3.2) * a0.cos(),
                hub + (r + 3.2) * a0.sin(),
                cz + sa * (r + 3.2) * a0.cos(),
            );
            // A pod: a pale shell with a band of glazing round it. Two rings
            // gives `add_ellipsoid` a bipyramid, so these hung off the rim as
            // black diamonds — dark because the body colour was near-black and
            // faceted because of the ring count.
            b.trim.add_ellipsoid(
                Vector3::new(gx, gy, gz),
                2.1,
                1.7,
                2.1,
                3,
                9,
                Color::from_hex(0xe4e8ec),
            );
            b.trim.add_limb(
                Vector3::new(gx, gy - 0.45, gz),
                Vector3::new(gx, gy + 0.55, gz),
                2.02,
                2.02,
                9,
                Color::from_hex(0x2f3a44),
                false,
            );
            // Lit from inside after dark. On `glow`, which is genuinely hidden
            // in daylight, and larger than the shell so the glow is not
            // occluded by it.
            b.glow.add_ellipsoid(
                Vector3::new(gx, gy, gz),
                2.3,
                1.9,
                2.3,
                3,
                9,
                Color::from_hex([0x6fd8ffu32, 0xffd98a, 0xff8ab0][(k / 2) % 3]),
            );
        }
    }
    // Hub.
    b.trim.add_limb(
        Vector3::new(cx + nx * 4.0, hub, cz + nz * 4.0),
        Vector3::new(cx - nx * 4.0, hub, cz - nz * 4.0),
        3.0,
        3.0,
        9,
        steel,
        true,
    );
    let _ = rng;
}

/// An obelisk on a paved square.
fn obelisk(b: &mut Batches, cx: f32, cz: f32, rng: &mut Rng) {
    let stone = scale_color(Color::from_hex(0xc4b89c), rng.range(0.95, 1.05));
    b.pads.add_ground_disc(cx, cz, 20.0, 20, 0.0, KERB + 0.02, Color::from_hex(0xb0a894));
    for (w, h0, h1) in [(7.0f32, 0.0f32, 3.0f32), (5.0, 3.0, 6.0)] {
        b.trim.add_box(
            Vector3::new(cx - w, KERB + h0, cz - w),
            Vector3::new(cx + w, KERB + h1, cz + w),
            scale_color(stone, 0.94),
            Uv::Unit,
        );
    }
    // The shaft: a very slight taper, and a pyramidion on top.
    const N: usize = 8;
    let h = 56.0f32;
    for k in 0..N {
        let (t0, t1) = (k as f32 / N as f32, (k + 1) as f32 / N as f32);
        b.trim.add_limb(
            Vector3::new(cx, KERB + 6.0 + h * t0, cz),
            Vector3::new(cx, KERB + 6.0 + h * t1, cz),
            mix(3.1, 2.1, t0),
            mix(3.1, 2.1, t1),
            4,
            stone,
            false,
        );
    }
    b.trim.add_limb(
        Vector3::new(cx, KERB + 6.0 + h, cz),
        Vector3::new(cx, KERB + 6.0 + h + 5.5, cz),
        2.1,
        0.05,
        4,
        Color::from_hex(0xd8c98a),
        false,
    );
}

/// A domed rotunda.
fn dome(b: &mut Batches, cx: f32, cz: f32, rng: &mut Rng) {
    let stone = scale_color(Color::from_hex(0xd8d0bc), rng.range(0.96, 1.04));
    let r = 22.0f32;
    b.pads
        .add_ground_disc(cx, cz, r + 12.0, 24, 0.0, KERB + 0.02, Color::from_hex(0xb0a894));
    // Steps.
    for k in 0..4 {
        let t = k as f32;
        b.trim.add_limb(
            Vector3::new(cx, KERB + t * 0.55, cz),
            Vector3::new(cx, KERB + (t + 1.0) * 0.55, cz),
            r + 6.0 - t * 1.4,
            r + 6.0 - (t + 1.0) * 1.4,
            24,
            scale_color(stone, 0.92),
            false,
        );
    }
    let base = KERB + 2.2;
    // Peristyle: a ring of columns, then the drum behind them.
    for k in 0..20 {
        let a = k as f32 / 20.0 * TAU;
        b.trim.add_limb(
            Vector3::new(cx + r * a.cos(), base, cz + r * a.sin()),
            Vector3::new(cx + r * a.cos(), base + 16.0, cz + r * a.sin()),
            1.15,
            1.0,
            7,
            stone,
            false,
        );
    }
    b.trim.add_limb(
        Vector3::new(cx, base, cz),
        Vector3::new(cx, base + 16.0, cz),
        r - 4.5,
        r - 4.5,
        22,
        scale_color(stone, 0.96),
        true,
    );
    // Entablature, then the dome: rings of falling radius on a sine profile.
    b.trim.add_limb(
        Vector3::new(cx, base + 16.0, cz),
        Vector3::new(cx, base + 18.4, cz),
        r + 1.8,
        r + 1.2,
        24,
        scale_color(stone, 1.05),
        false,
    );
    const D: usize = 9;
    let dh = 20.0f32;
    for k in 0..D {
        let (t0, t1) = (k as f32 / D as f32, (k + 1) as f32 / D as f32);
        b.trim.add_limb(
            Vector3::new(cx, base + 18.4 + dh * t0, cz),
            Vector3::new(cx, base + 18.4 + dh * t1, cz),
            (r - 2.0) * (1.0 - t0 * t0).max(0.0).sqrt(),
            (r - 2.0) * (1.0 - t1 * t1).max(0.0).sqrt(),
            24,
            Color::from_hex(0x7fae9a),
            false,
        );
    }
    // Lantern on top.
    b.trim.add_limb(
        Vector3::new(cx, base + 18.4 + dh, cz),
        Vector3::new(cx, base + 18.4 + dh + 7.0, cz),
        3.0,
        2.6,
        10,
        stone,
        false,
    );
    b.trim.add_limb(
        Vector3::new(cx, base + 18.4 + dh + 7.0, cz),
        Vector3::new(cx, base + 18.4 + dh + 11.0, cz),
        2.6,
        0.1,
        10,
        Color::from_hex(0xc9a352),
        false,
    );
}

/// Build one. Returns false if the ground was already taken.
pub(crate) fn add_monument(
    b: &mut Batches,
    kind: Monument,
    cx: f32,
    cz: f32,
    yaw: f32,
    rng: &mut Rng,
) -> bool {
    let f = kind.footprint();
    if !b.occ.try_claim([cx - f, cz - f, cx + f, cz + f]) {
        return false;
    }
    match kind {
        Monument::LatticeTower => lattice_tower(b, cx, cz, rng),
        Monument::Colossus => colossus(b, cx, cz, yaw, rng),
        Monument::Arch => arch(b, cx, cz, yaw, rng),
        Monument::Wheel => wheel(b, cx, cz, yaw, rng),
        Monument::Obelisk => obelisk(b, cx, cz, rng),
        Monument::Dome => dome(b, cx, cz, rng),
    }
    true
}

/// A suspension bridge across the water.
///
/// Three things make one recognisable and none of them is the deck. The
/// **towers**, which are far taller than anything else on the water. The
/// **main cable**, which hangs in a catenary from tower top down to deck level
/// at mid-span and back — that curve is the single most identifiable shape in
/// civil engineering. And the **hangers**, the vertical ties between cable and
/// deck, which get shorter towards the middle and are what tell you the curve
/// is holding something up.
///
/// Painted the orange-red that the famous one is, because a grey suspension
/// bridge reads as infrastructure and a coloured one reads as a landmark.
pub(crate) fn add_suspension_bridge(
    b: &mut Batches,
    along_x: bool,
    fixed: f32,
    w0: f32,
    w1: f32,
    deck_y: f32,
    rng: &mut Rng,
) -> bool {
    let clear = w1 - w0;
    if clear < 30.0 {
        return false;
    }
    // Towers stand just outside the water, so the main span is the channel.
    let (t0, t1) = (w0 - clear * 0.12, w1 + clear * 0.12);
    let span = t1 - t0;
    let tower_h = deck_y + (span * 0.13).clamp(28.0, 78.0);
    let half_w = 7.0f32;
    let paint = Color::from_hex(0xc1502e);
    let at = |s: f32, t: f32| -> (f32, f32) {
        if along_x {
            (s, fixed + t)
        } else {
            (fixed + t, s)
        }
    };
    if !b.occ.try_claim(if along_x {
        [t0 - 12.0, fixed - half_w - 6.0, t1 + 12.0, fixed + half_w + 6.0]
    } else {
        [fixed - half_w - 6.0, t0 - 12.0, fixed + half_w + 6.0, t1 + 12.0]
    }) {
        return false;
    }

    // Deck, carried right through both towers to the abutments.
    let (d0x, d0z) = at(t0 - 26.0, -half_w);
    let (d1x, d1z) = at(t1 + 26.0, half_w);
    b.road.add_slab(
        d0x.min(d1x),
        d0z.min(d1z),
        d0x.max(d1x),
        d0z.max(d1z),
        deck_y,
        Color::from_hex(0x43434a),
        Uv::Unit,
    );
    // Stiffening truss under it: the depth that stops the deck looking like a
    // ribbon of tape stretched between two towers.
    b.trim.add_box(
        Vector3::new(d0x.min(d1x), deck_y - 1.9, d0z.min(d1z)),
        Vector3::new(d0x.max(d1x), deck_y - 0.1, d0z.max(d1z)),
        scale_color(paint, 0.85),
        Uv::Unit,
    );
    // Centre line and edge lines.
    for (t, w, c) in [
        (0.0f32, 0.22f32, Color::from_hex(0xd8d4c4)),
        (-half_w + 0.7, 0.20, Color::from_hex(0xd8d4c4)),
        (half_w - 0.7, 0.20, Color::from_hex(0xd8d4c4)),
    ] {
        let (l0x, l0z) = at(t0 - 26.0, t - w * 0.5);
        let (l1x, l1z) = at(t1 + 26.0, t + w * 0.5);
        b.paint.add_slab(
            l0x.min(l1x),
            l0z.min(l1z),
            l0x.max(l1x),
            l0z.max(l1z),
            deck_y + 0.02,
            c,
            Uv::Unit,
        );
    }

    // Towers: two legs with cross-braces, standing above the deck.
    for s in [t0, t1] {
        for side in [-1.0f32, 1.0] {
            let (px, pz) = at(s, side * (half_w - 1.2));
            // Pier down into the water.
            b.trim.add_limb(
                Vector3::new(px, -RIVER_DEPTH - 1.0, pz),
                Vector3::new(px, deck_y, pz),
                3.2,
                2.4,
                8,
                scale_color(paint, 0.8),
                false,
            );
            // The tower leg, tapering.
            b.trim.add_limb(
                Vector3::new(px, deck_y - 2.0, pz),
                Vector3::new(px, tower_h, pz),
                2.4,
                1.3,
                6,
                paint,
                false,
            );
        }
        // Cross-braces: the ladder between the legs, which is the other half
        // of a suspension tower's silhouette.
        let n = 4;
        for k in 0..=n {
            let y = mix(deck_y + 3.0, tower_h - 1.5, k as f32 / n as f32);
            let (a0x, a0z) = at(s, -(half_w - 1.2));
            let (a1x, a1z) = at(s, half_w - 1.2);
            b.trim.add_limb(
                Vector3::new(a0x, y, a0z),
                Vector3::new(a1x, y, a1z),
                1.0,
                1.0,
                4,
                paint,
                false,
            );
        }
    }

    // Main cables, and the hangers under them.
    const SEG: usize = 26;
    let sag = tower_h - deck_y - 2.0;
    for side in [-1.0f32, 1.0] {
        let t = side * (half_w - 1.2);
        // Main span: a catenary from tower top to tower top, approximated by
        // `4u(1-u)`, which is within a whisker of the real curve over one span
        // and is what the eye is checking for.
        for k in 0..SEG {
            let (u0, u1) = (k as f32 / SEG as f32, (k + 1) as f32 / SEG as f32);
            let y = |u: f32| tower_h - sag * 4.0 * u * (1.0 - u);
            let (p0x, p0z) = at(mix(t0, t1, u0), t);
            let (p1x, p1z) = at(mix(t0, t1, u1), t);
            b.trim.add_limb(
                Vector3::new(p0x, y(u0), p0z),
                Vector3::new(p1x, y(u1), p1z),
                0.45,
                0.45,
                5,
                paint,
                false,
            );
            // Hangers, every other segment.
            if k % 2 == 0 && k > 0 {
                let (hx, hz) = at(mix(t0, t1, u0), t);
                b.trim.add_limb(
                    Vector3::new(hx, y(u0), hz),
                    Vector3::new(hx, deck_y, hz),
                    0.10,
                    0.10,
                    3,
                    paint,
                    false,
                );
            }
        }
        // Back stays, running from each tower top down to the abutment.
        for (s, dir) in [(t0, -1.0f32), (t1, 1.0)] {
            let n = 6;
            for k in 0..n {
                let (u0, u1) = (k as f32 / n as f32, (k + 1) as f32 / n as f32);
                let y = |u: f32| mix(tower_h, deck_y + 1.5, u * u);
                let (p0x, p0z) = at(s + dir * 26.0 * u0, t);
                let (p1x, p1z) = at(s + dir * 26.0 * u1, t);
                b.trim.add_limb(
                    Vector3::new(p0x, y(u0), p0z),
                    Vector3::new(p1x, y(u1), p1z),
                    0.4,
                    0.4,
                    5,
                    paint,
                    false,
                );
            }
        }
    }
    // Lit at night: a string of lamps down the deck, and the cable picked out.
    let lamps = (span / 34.0).max(2.0) as usize;
    for k in 0..=lamps {
        let u = k as f32 / lamps as f32;
        for side in [-1.0f32, 1.0] {
            let (lx, lz) = at(mix(t0 - 20.0, t1 + 20.0, u), side * (half_w - 2.2));
            b.trim.add_limb(
                Vector3::new(lx, deck_y, lz),
                Vector3::new(lx, deck_y + 7.0, lz),
                0.16,
                0.11,
                5,
                scale_color(paint, 0.9),
                false,
            );
            b.glow.add_box(
                Vector3::new(lx - 0.7, deck_y + 6.7, lz - 0.4),
                Vector3::new(lx + 0.7, deck_y + 7.0, lz + 0.4),
                Color::from_hex(0xffeec4),
                Uv::Unit,
            );
        }
    }
    let _ = rng;
    true
}
