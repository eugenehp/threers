//! Part of the `simcity` example; see `mod.rs`.
//!
//! Places people go to watch or play a game.
//!
//! Cities in this genre are usually all housing, offices and shops, and the
//! thing that fixes that is not more buildings — it is the flat, painted,
//! fenced ground in between. A ballpark, a cricket ground and a row of hard
//! courts are three completely different *plan shapes*, and plan shape is what
//! you read from the air. A stadium is a closed oval, a ballpark is a wedge, a
//! cricket ground is a circle with a strip across the middle, and courts are a
//! grid of rectangles. None of them can be mistaken for a block of flats.
#![allow(dead_code)]

use super::*;

/// Surfaces the small games are played on, and the colours they are painted.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Court {
    Basketball,
    Tennis,
    Pickleball,
    Netball,
}

impl Court {
    /// Playing area in metres, close enough to the real thing that the
    /// proportions read: a tennis court is long and narrow, a pickleball court
    /// is small, a basketball court is neither.
    pub(crate) fn size(self) -> (f32, f32) {
        match self {
            Court::Basketball => (28.0, 15.0),
            Court::Tennis => (23.8, 11.0),
            Court::Pickleball => (13.4, 6.1),
            Court::Netball => (30.5, 15.25),
        }
    }

    pub(crate) fn surface(self) -> u32 {
        match self {
            Court::Basketball => 0x2f5f7a,
            Court::Tennis => 0x2d6b46,
            Court::Pickleball => 0x3f5aa0,
            Court::Netball => 0x8a5a3a,
        }
    }

    pub(crate) fn apron(self) -> u32 {
        match self {
            Court::Basketball => 0x8a4a3a,
            Court::Tennis => 0x8a5a3a,
            Court::Pickleball => 0x2e6f52,
            Court::Netball => 0x4a6a86,
        }
    }
}

/// One court: apron, playing surface, painted lines, net or hoops, and a fence.
///
/// The apron is the wide surround in a contrasting colour, and it is doing
/// more work than the court: a bare rectangle of blue on grass reads as a
/// tarpaulin, and the same rectangle inside a red surround reads as a court
/// from three hundred metres up.
pub(crate) fn add_court(
    b: &mut Batches,
    cx: f32,
    cz: f32,
    kind: Court,
    along_x: bool,
    fenced: bool,
    rng: &mut Rng,
) -> bool {
    let (l, w) = kind.size();
    let (dx, dz) = if along_x { (l, w) } else { (w, l) };
    let (ax, az) = (dx * 0.5 + 3.0, dz * 0.5 + 3.0);
    if !b.occ.try_claim([cx - ax, cz - az, cx + ax, cz + az]) {
        return false;
    }
    b.pads.add_slab(
        cx - ax,
        cz - az,
        cx + ax,
        cz + az,
        KERB + 0.02,
        scale_color(Color::from_hex(kind.apron()), rng.range(0.92, 1.08)),
        Uv::Unit,
    );
    b.pads.add_slab(
        cx - dx * 0.5,
        cz - dz * 0.5,
        cx + dx * 0.5,
        cz + dz * 0.5,
        KERB + 0.04,
        scale_color(Color::from_hex(kind.surface()), rng.range(0.95, 1.05)),
        Uv::Unit,
    );
    // Lines: the boundary, and one across the middle. Enough to say "marked".
    let line = Color::WHITE;
    let t = 0.14;
    for (lx0, lz0, lx1, lz1) in [
        (-dx * 0.5, -dz * 0.5, dx * 0.5, -dz * 0.5 + t),
        (-dx * 0.5, dz * 0.5 - t, dx * 0.5, dz * 0.5),
        (-dx * 0.5, -dz * 0.5, -dx * 0.5 + t, dz * 0.5),
        (dx * 0.5 - t, -dz * 0.5, dx * 0.5, dz * 0.5),
        (-t * 0.5, -dz * 0.5, t * 0.5, dz * 0.5),
    ] {
        b.paint.add_slab(
            cx + lx0,
            cz + lz0,
            cx + lx1,
            cz + lz1,
            KERB + 0.05,
            line,
            Uv::Unit,
        );
    }
    match kind {
        Court::Basketball | Court::Netball => {
            // Hoops at both ends: a post, a backboard and a ring.
            for s in [-1.0f32, 1.0] {
                let (hx, hz) = if along_x {
                    (cx + s * (dx * 0.5 - 1.2), cz)
                } else {
                    (cx, cz + s * (dz * 0.5 - 1.2))
                };
                b.trim.add_limb(
                    Vector3::new(hx, KERB, hz),
                    Vector3::new(hx, 3.6, hz),
                    0.10,
                    0.08,
                    5,
                    Color::from_hex(0x53585e),
                    false,
                );
                let (bx, bz) = if along_x { (0.10f32, 0.9f32) } else { (0.9, 0.10) };
                b.trim.add_box(
                    Vector3::new(hx - bx - s * 0.0, 2.9, hz - bz),
                    Vector3::new(hx + bx, 3.9, hz + bz),
                    Color::WHITE,
                    Uv::Unit,
                );
                b.trim.add_limb(
                    Vector3::new(hx - if along_x { s * 0.5 } else { 0.0 }, 3.05, hz - if along_x { 0.0 } else { s * 0.5 }),
                    Vector3::new(hx - if along_x { s * 0.9 } else { 0.0 }, 3.05, hz - if along_x { 0.0 } else { s * 0.9 }),
                    0.22,
                    0.22,
                    6,
                    Color::from_hex(0xd4622e),
                    false,
                );
            }
        }
        Court::Tennis | Court::Pickleball => {
            // A net across the middle, with a post each end.
            let (nx, nz) = if along_x { (0.06f32, dz * 0.5 + 0.5) } else { (dx * 0.5 + 0.5, 0.06) };
            b.trim.add_box(
                Vector3::new(cx - nx, KERB, cz - nz),
                Vector3::new(cx + nx, KERB + 1.05, cz + nz),
                Color::from_hex(0x3a3f45),
                Uv::Unit,
            );
        }
    }
    if fenced {
        // Chain-link: posts and a rail. The mesh itself would be a texture
        // this renderer has no alpha pass for, so it is left to the posts —
        // which is what you see at this distance anyway.
        let n = 14;
        for i in 0..n {
            let u = i as f32 / n as f32 * TAU;
            let (px, pz) = (cx + ax * u.cos() * 1.02, cz + az * u.sin() * 1.02);
            b.trim.add_limb(
                Vector3::new(px, KERB, pz),
                Vector3::new(px, KERB + 3.4, pz),
                0.06,
                0.06,
                4,
                Color::from_hex(0x6f747a),
                false,
            );
        }
    }
    true
}

/// A row of courts sharing one apron, which is how they are actually built.
pub(crate) fn add_court_bank(
    b: &mut Batches,
    cx: f32,
    cz: f32,
    kind: Court,
    n: usize,
    along_x: bool,
    rng: &mut Rng,
) -> usize {
    let (l, w) = kind.size();
    let pitch = if along_x { w + 7.0 } else { l + 7.0 };
    let mut made = 0;
    for i in 0..n {
        let off = (i as f32 - (n as f32 - 1.0) * 0.5) * pitch;
        let (x, z) = if along_x { (cx, cz + off) } else { (cx + off, cz) };
        if add_court(b, x, z, kind, along_x, i == 0, rng) {
            made += 1;
        }
    }
    made
}

/// A baseball park.
///
/// The wedge is the whole identity: the diamond in one corner, the outfield
/// fanning out from it, the stands wrapped round the back of home plate and
/// nowhere else. Drawn as a sector rather than an oval for exactly that
/// reason.
pub(crate) fn add_ballpark(b: &mut Batches, cx: f32, cz: f32, r: f32, rot: f32, rng: &mut Rng) -> bool {
    if !b.occ.try_claim([cx - r, cz - r, cx + r, cz + r]) {
        return false;
    }
    let (sa, ca) = rot.sin_cos();
    let at = |ang: f32, rad: f32| -> (f32, f32) {
        let (lx, lz) = (rad * ang.cos(), rad * ang.sin());
        (cx + lx * ca - lz * sa, cz + lx * sa + lz * ca)
    };
    // Home plate at the apex; the field opens through a right angle.
    let (a0, a1) = (-PI * 0.25, PI * 0.25);
    let segs = 12;
    let turf = Color::from_hex(0x35803a);
    let dirt = Color::from_hex(0x9a6a44);
    let (hx, hz) = at(PI, 4.0);
    for i in 0..segs {
        let u0 = mix(a0, a1, i as f32 / segs as f32);
        let u1 = mix(a0, a1, (i + 1) as f32 / segs as f32);
        let (x0, z0) = at(u0, r * 0.86);
        let (x1, z1) = at(u1, r * 0.86);
        b.grass.quad(
            [[hx, KERB + 0.02, hz], [x0, KERB + 0.02, z0], [x1, KERB + 0.02, z1], [hx, KERB + 0.02, hz]],
            [0.0, 1.0, 0.0],
            Uv::Unit,
            scale_color(turf, rng.range(0.95, 1.05)),
        );
    }
    // The infield: a dirt sector, the diamond's bases, and the mound.
    for i in 0..segs {
        let u0 = mix(a0, a1, i as f32 / segs as f32);
        let u1 = mix(a0, a1, (i + 1) as f32 / segs as f32);
        let (x0, z0) = at(u0, r * 0.44);
        let (x1, z1) = at(u1, r * 0.44);
        b.pads.quad(
            [[hx, KERB + 0.04, hz], [x0, KERB + 0.04, z0], [x1, KERB + 0.04, z1], [hx, KERB + 0.04, hz]],
            [0.0, 1.0, 0.0],
            Uv::Unit,
            scale_color(dirt, rng.range(0.95, 1.05)),
        );
    }
    let (mx, mz) = at(0.0, r * 0.32);
    b.pads
        .add_ground_disc(mx, mz, r * 0.044, 10, 0.0, KERB + 0.06, Color::from_hex(0xa87a52));
    for (ang, rad) in [(a0, r * 0.44), (0.0, r * 0.44), (a1, r * 0.44)] {
        let (bx, bz) = at(ang, rad * 0.82);
        b.paint
            .add_slab(bx - 0.7, bz - 0.7, bx + 0.7, bz + 0.7, KERB + 0.07, Color::WHITE, Uv::Unit);
    }
    // Stands behind the plate, and the outfield wall.
    let seats = Color::from_hex(0x27568c);
    for i in 0..10 {
        let u0 = mix(a1, a0 + TAU, i as f32 / 10.0);
        let u1 = mix(a1, a0 + TAU, (i + 1) as f32 / 10.0);
        let (ix0, iz0) = at(u0, r * 0.35);
        let (ix1, iz1) = at(u1, r * 0.35);
        let (ox0, oz0) = at(u0, r * 0.65);
        let (ox1, oz1) = at(u1, r * 0.65);
        b.trim.quad(
            [
                [ix0, KERB + 1.0, iz0],
                [ix1, KERB + 1.0, iz1],
                [ox1, KERB + 13.0, oz1],
                [ox0, KERB + 13.0, oz0],
            ],
            [0.0, 0.9, 0.0],
            Uv::Unit,
            scale_color(seats, rng.range(0.9, 1.1)),
        );
        b.trim.quad(
            [
                [ox0, KERB, oz0],
                [ox0, KERB + 15.0, oz0],
                [ox1, KERB + 15.0, oz1],
                [ox1, KERB, oz1],
            ],
            [(ox0 + ox1) * 0.5 - cx, 0.0, (oz0 + oz1) * 0.5 - cz],
            Uv::Unit,
            Color::from_hex(0xb0aca4),
        );
    }
    for i in 0..segs {
        let u0 = mix(a0, a1, i as f32 / segs as f32);
        let u1 = mix(a0, a1, (i + 1) as f32 / segs as f32);
        let (x0, z0) = at(u0, r * 0.88);
        let (x1, z1) = at(u1, r * 0.88);
        b.trim.quad(
            [
                [x0, KERB, z0],
                [x0, KERB + 3.4, z0],
                [x1, KERB + 3.4, z1],
                [x1, KERB, z1],
            ],
            [cx - (x0 + x1) * 0.5, 0.0, cz - (z0 + z1) * 0.5],
            Uv::Unit,
            Color::from_hex(0x2b4a34),
        );
    }
    true
}

/// A cricket ground.
///
/// A circle of mown grass with a paler strip across the middle and a rope
/// round the edge. No stands worth the name: most grounds are a pavilion and
/// a bank, and the pavilion is the only building.
pub(crate) fn add_cricket_ground(b: &mut Batches, cx: f32, cz: f32, r: f32, rot: f32, rng: &mut Rng) -> bool {
    if !b.occ.try_claim([cx - r, cz - r, cx + r, cz + r]) {
        return false;
    }
    b.grass.add_ground_disc(
        cx,
        cz,
        r,
        26,
        0.0,
        KERB + 0.02,
        scale_color(Color::from_hex(0x4f8a3c), rng.range(0.95, 1.05)),
    );
    // Mown rings, which is what a ground actually looks like from above.
    for k in 1..4 {
        let rr = r * (1.0 - k as f32 * 0.22);
        b.grass.add_ground_disc(
            cx,
            cz,
            rr,
            22,
            0.0,
            KERB + 0.03 + k as f32 * 0.005,
            scale_color(Color::from_hex(0x4f8a3c), if k % 2 == 0 { 1.08 } else { 0.93 }),
        );
    }
    // The square: a pale rectangle of bare-ish turf across the middle.
    let (sa, ca) = rot.sin_cos();
    let (hw, hd) = (r * 0.18, r * 0.047);
    b.pads.quad(
        [
            [cx - hw * ca + hd * sa, KERB + 0.05, cz - hw * sa - hd * ca],
            [cx + hw * ca + hd * sa, KERB + 0.05, cz + hw * sa - hd * ca],
            [cx + hw * ca - hd * sa, KERB + 0.05, cz + hw * sa + hd * ca],
            [cx - hw * ca - hd * sa, KERB + 0.05, cz - hw * sa + hd * ca],
        ],
        [0.0, 1.0, 0.0],
        Uv::Unit,
        Color::from_hex(0xbdc08a),
    );
    // The boundary rope, and sightscreens at each end of the square.
    for i in 0..40 {
        let a = i as f32 / 40.0 * TAU;
        let (px, pz) = (cx + r * 0.94 * a.cos(), cz + r * 0.94 * a.sin());
        b.trim.add_box(
            Vector3::new(px - 0.3, KERB, pz - 0.3),
            Vector3::new(px + 0.3, KERB + 0.35, pz + 0.3),
            Color::WHITE,
            Uv::Unit,
        );
    }
    for s in [-1.0f32, 1.0] {
        let (px, pz) = (cx + s * r * 0.82 * ca, cz + s * r * 0.82 * sa);
        b.trim.add_yaw_box(
            Vector3::new(px, KERB + 2.5, pz),
            Vector3::new(4.0, 2.5, 0.25),
            rot,
            Color::from_hex(0xf0efe6),
            Uv::Unit,
        );
    }
    // The pavilion.
    let (px, pz) = (cx - r * 0.86 * sa, cz + r * 0.86 * ca);
    b.trim.add_yaw_box(
        Vector3::new(px, KERB + 3.0, pz),
        Vector3::new(8.0, 3.0, 4.0),
        rot,
        Color::from_hex(0xe4ded0),
        Uv::Unit,
    );
    b.trim.add_yaw_box(
        Vector3::new(px, KERB + 6.3, pz),
        Vector3::new(8.5, 0.3, 4.5),
        rot,
        Color::from_hex(0x6b4038),
        Uv::Unit,
    );
    true
}
