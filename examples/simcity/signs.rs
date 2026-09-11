//! Part of the `simcity` example; see `mod.rs`.
//!
//! Advertising: fascia signs, projecting blades, wall panels, roof hoardings
//! and neon.
//!
//! A city at night is mostly not lit by street lamps. It is lit by the things
//! trying to sell you something — and those are also what tell you which floor
//! is a shop, which corner is a junction worth stopping at, and which block is
//! the busy one. Without them a night render is a grid of orange pools with
//! dark buildings between them.
//!
//! Everything here is built from two pieces: a *structure* in the lit `trim`
//! batch, which is there in daylight, and a *face* in the unlit `neon` batch,
//! which only appears after dark. So a hoarding is a grey board by day and a
//! lit board by night, without a second material or a second draw.
#![allow(dead_code)]

use super::*;

/// Colours a sign face is drawn from. Saturated on purpose: everything else in
/// the city is a desaturated construction material, and the eye finds these.
const INK: [u32; 10] = [
    0xff2d5e, 0x24d6ff, 0xffd23f, 0x35e08a, 0xff7a1a, 0xd44bff, 0xff4444, 0x54ffe0, 0xffffff,
    0x6f8bff,
];

/// A quad in an arbitrary plane. `right` and `up` are unit vectors; the face
/// points along `right x up`.
fn panel(
    m: &mut MeshBuilder,
    o: Vector3,
    right: Vector3,
    up: Vector3,
    w: f32,
    h: f32,
    c: Color,
) {
    let p = |a: f32, b: f32| [o.x + right.x * a + up.x * b, o.y + right.y * a + up.y * b, o.z + right.z * a + up.z * b];
    let n = right.cross(up);
    m.quad(
        [p(0.0, 0.0), p(w, 0.0), p(w, h), p(0.0, h)],
        [n.x, n.y, n.z],
        Uv::Unit,
        c,
    );
}

/// Fill a sign face with an abstract advert.
///
/// Not text — a legible word needs a texture and this is one draw call shared
/// by every sign in the city. What reads at a distance is *layout*: a field, a
/// bold band across it, a mark, and a line of small blocks standing in for a
/// strapline. Any poster in any city is that, and at fifty metres it is all
/// you get from a real one too.
fn ad_face(
    m: &mut MeshBuilder,
    o: Vector3,
    right: Vector3,
    up: Vector3,
    w: f32,
    h: f32,
    rng: &mut Rng,
) {
    let ground = Color::from_hex(INK[rng.below(INK.len())]);
    let mark = Color::from_hex(INK[rng.below(INK.len())]);
    let at = |a: f32, b: f32| {
        Vector3::new(
            o.x + right.x * a + up.x * b,
            o.y + right.y * a + up.y * b,
            o.z + right.z * a + up.z * b,
        )
    };
    panel(m, o, right, up, w, h, scale_color(ground, 0.55));

    // Push each layer a millimetre forward of the last so nothing z-fights.
    let n = right.cross(up).normalize();
    let step = |k: f32| Vector3::new(n.x * 0.004 * k, n.y * 0.004 * k, n.z * 0.004 * k);

    match rng.below(4) {
        // A band across the middle with a mark on it.
        0 => {
            let bh = h * rng.range(0.28, 0.44);
            let by = (h - bh) * rng.range(0.25, 0.75);
            panel(m, at(0.0, by) + step(1.0), right, up, w, bh, mark);
            let r = bh * 0.34;
            panel(
                m,
                at(w * 0.12, by + bh * 0.5 - r) + step(2.0),
                right,
                up,
                r * 2.0,
                r * 2.0,
                Color::from_hex(0xffffff),
            );
        }
        // A block of colour down one side, a stack of straplines beside it.
        1 => {
            let bw = w * rng.range(0.28, 0.42);
            let flip = rng.chance(0.5);
            let bx = if flip { w - bw } else { 0.0 };
            panel(m, at(bx, 0.0) + step(1.0), right, up, bw, h, mark);
            let lines = 2 + rng.below(3);
            for i in 0..lines {
                let ly = h * (0.24 + 0.18 * i as f32);
                if ly + h * 0.09 > h {
                    break;
                }
                let lw = (w - bw - w * 0.10) * rng.range(0.45, 0.95);
                let lx = if flip { w * 0.05 } else { bw + w * 0.05 };
                panel(
                    m,
                    at(lx, ly) + step(2.0),
                    right,
                    up,
                    lw,
                    h * 0.09,
                    Color::from_hex(0xf0f0f0),
                );
            }
        }
        // One huge mark, poster style.
        2 => {
            let s = h * rng.range(0.5, 0.78);
            panel(
                m,
                at((w - s) * rng.range(0.1, 0.9), (h - s) * 0.55) + step(1.0),
                right,
                up,
                s,
                s,
                mark,
            );
            panel(
                m,
                at(w * 0.08, h * 0.08) + step(1.0),
                right,
                up,
                w * 0.5,
                h * 0.10,
                Color::from_hex(0xf0f0f0),
            );
        }
        // Vertical stripes.
        _ => {
            let bars = 3 + rng.below(5);
            for i in 0..bars {
                if !rng.chance(0.7) {
                    continue;
                }
                let bw = w / bars as f32;
                panel(
                    m,
                    at(i as f32 * bw, h * 0.10) + step(1.0),
                    right,
                    up,
                    bw * 0.82,
                    h * rng.range(0.4, 0.8),
                    Color::from_hex(INK[rng.below(INK.len())]),
                );
            }
        }
    }
}

/// A run of neon tube, as glyph-like strokes.
///
/// Real letterforms need a texture; what carries at street level is the rhythm
/// of a word — upright strokes at a regular pitch with the odd bowl and
/// crossbar, all in one saturated colour, all the same tube thickness.
fn neon_word(
    m: &mut MeshBuilder,
    o: Vector3,
    right: Vector3,
    up: Vector3,
    w: f32,
    h: f32,
    c: Color,
    rng: &mut Rng,
) {
    let at = |a: f32, b: f32| {
        Vector3::new(
            o.x + right.x * a + up.x * b,
            o.y + right.y * a + up.y * b,
            o.z + right.z * a + up.z * b,
        )
    };
    let glyphs = ((w / (h * 0.72)).floor() as usize).clamp(2, 9);
    let pitch = w / glyphs as f32;
    let tube = (h * 0.09).clamp(0.025, 0.09);
    for g in 0..glyphs {
        let x = g as f32 * pitch + pitch * 0.14;
        let gw = pitch * 0.62;
        match rng.below(5) {
            // Upright with a crossbar.
            0 => {
                m.add_limb(at(x, 0.0), at(x, h), tube, tube, 5, c, true);
                m.add_limb(
                    at(x, h * 0.55),
                    at(x + gw, h * 0.55),
                    tube,
                    tube,
                    5,
                    c,
                    true,
                );
            }
            // Two uprights bridged.
            1 => {
                m.add_limb(at(x, 0.0), at(x, h), tube, tube, 5, c, true);
                m.add_limb(at(x + gw, 0.0), at(x + gw, h), tube, tube, 5, c, true);
                m.add_limb(at(x, h * 0.5), at(x + gw, h * 0.5), tube, tube, 5, c, true);
            }
            // A bowl.
            2 => {
                const N: usize = 7;
                for i in 0..N {
                    let (a0, a1) = (
                        i as f32 / N as f32 * TAU,
                        (i + 1) as f32 / N as f32 * TAU,
                    );
                    let (rx, ry) = (gw * 0.5, h * 0.5);
                    m.add_limb(
                        at(x + gw * 0.5 + rx * a0.cos(), h * 0.5 + ry * a0.sin()),
                        at(x + gw * 0.5 + rx * a1.cos(), h * 0.5 + ry * a1.sin()),
                        tube,
                        tube,
                        4,
                        c,
                        false,
                    );
                }
            }
            // A diagonal.
            3 => {
                m.add_limb(at(x, 0.0), at(x, h), tube, tube, 5, c, true);
                m.add_limb(at(x, h), at(x + gw, 0.0), tube, tube, 5, c, true);
            }
            // Three horizontals.
            _ => {
                for k in 0..3 {
                    let y = h * (0.06 + 0.44 * k as f32);
                    m.add_limb(at(x, y), at(x + gw, y), tube, tube, 4, c, true);
                }
                m.add_limb(at(x, 0.0), at(x, h), tube, tube, 5, c, true);
            }
        }
    }
}

/// Which neon batch a sign goes into. Roughly one sign in four blinks; more
/// than that and the street strobes.
fn neon_batch<'a>(b: &'a mut Batches, rng: &mut Rng) -> &'a mut MeshBuilder {
    if rng.chance(0.24) {
        &mut b.neon_blink
    } else {
        &mut b.neon
    }
}

/// A wall's frame: where a point on it is, which way its surface runs, and
/// which way it faces.
///
/// Signage used to be hard-wired to walls facing along Z, because the awning
/// and fire-escape code it was bolted next to is. On a street running the
/// other way that puts every sign in the city edge-on to the camera — a whole
/// avenue with nothing on it. `along_x` picks which pair of walls is being
/// dressed.
struct Wall {
    along_x: bool,
    /// The wall's own coordinate on the axis it does *not* run along.
    fixed: f32,
    /// +1 or -1: which way the wall faces on that axis.
    front: f32,
}

impl Wall {
    /// World position `s` metres along the wall at height `y`, `out` metres
    /// proud of it.
    fn at(&self, s: f32, y: f32, out: f32) -> Vector3 {
        if self.along_x {
            Vector3::new(s, y, self.fixed + self.front * out)
        } else {
            Vector3::new(self.fixed + self.front * out, y, s)
        }
    }

    fn normal(&self) -> Vector3 {
        if self.along_x {
            Vector3::new(0.0, 0.0, self.front)
        } else {
            Vector3::new(self.front, 0.0, 0.0)
        }
    }

    /// `right` such that `right x up` is the outward normal, and the offset
    /// along the wall that `right` starts from for a panel `w` wide at `s`.
    fn frame(&self, s: f32, w: f32) -> (Vector3, f32) {
        let right = Vector3::new(0.0, 1.0, 0.0).cross(self.normal());
        let along = if self.along_x { right.x } else { right.z };
        (right, if along > 0.0 { s } else { s + w })
    }
}

/// The signage over a shopfront: a fascia board with neon on it, and sometimes
/// a blade projecting over the pavement.
pub(crate) fn add_shop_signage(
    b: &mut Batches,
    a0: f32,
    a1: f32,
    fixed: f32,
    front: f32,
    along_x: bool,
    y: f32,
    rng: &mut Rng,
) {
    let wall = Wall {
        along_x,
        fixed,
        front,
    };
    let (x0, x1) = (a0, a1);
    if x1 - x0 < 3.0 {
        return;
    }
    let up = Vector3::new(0.0, 1.0, 0.0);
    let w = (x1 - x0) * rng.range(0.55, 0.92);
    let ox = mix(x0, x1 - w, rng.f());
    let h = rng.range(0.7, 1.15);
    let (right, start) = wall.frame(ox, w);
    let _o = wall.at(start, y, 0.06);

    // The board itself, lit, so it is a grey fascia by day.
    let board = scale_color(Color::from_hex(0x33363b), rng.range(0.85, 1.2));
    let (bmin, bmax) = if along_x {
        (
            Vector3::new(ox, y, fixed + front * 0.06 - 0.10),
            Vector3::new(ox + w, y + h, fixed + front * 0.06 + 0.10),
        )
    } else {
        (
            Vector3::new(fixed + front * 0.06 - 0.10, y, ox),
            Vector3::new(fixed + front * 0.06 + 0.10, y + h, ox + w),
        )
    };
    b.trim.add_box(bmin, bmax, board, Uv::Unit);
    let ink = Color::from_hex(INK[rng.below(INK.len())]);
    // A painted sign board by day, in the ink the neon will use after dark.
    panel(
        &mut b.trim,
        wall.at(start, y + h * 0.12, 0.13),
        right,
        up,
        w * 0.9,
        h * 0.62,
        scale_color(ink, 0.45),
    );
    let m = neon_batch(b, rng);
    if rng.chance(0.55) {
        neon_word(
            m,
            wall.at(start, y + h * 0.18, 0.16),
            right,
            up,
            w * 0.86,
            h * 0.62,
            ink,
            rng,
        );
    } else {
        // A backlit fascia instead: the whole board glows and a bar runs
        // under it.
        panel(
            m,
            wall.at(start, y + h * 0.12, 0.16),
            right,
            up,
            w * 0.9,
            h * 0.60,
            scale_color(ink, 0.7),
        );
        panel(
            m,
            wall.at(start, y + h * 0.04, 0.17),
            right,
            up,
            w * 0.9,
            h * 0.06,
            Color::from_hex(0xffffff),
        );
    }

    // A projecting blade, double sided, with a bulb border. This is the piece
    // that reads from down the street rather than from in front of the shop.
    if rng.chance(0.38) {
        let bh = rng.range(1.9, 3.6);
        let bw = rng.range(0.6, 1.0);
        let bs = mix(ox + 0.4, ox + w - 0.4, rng.f());
        let steel = Color::from_hex(0x2c2f34);
        // The bracket, running out from the wall.
        let (mn, mx) = {
            let near = wall.at(bs, y + 0.1, 0.10);
            let far = wall.at(bs, y + 0.1 + bh, 0.10 + bw);
            (
                Vector3::new(
                    near.x.min(far.x) - 0.06,
                    near.y,
                    near.z.min(far.z) - 0.06,
                ),
                Vector3::new(
                    near.x.max(far.x) + 0.06,
                    far.y,
                    near.z.max(far.z) + 0.06,
                ),
            )
        };
        b.trim.add_box(mn, mx, steel, Uv::Unit);

        let ink = Color::from_hex(INK[rng.below(INK.len())]);
        let m = neon_batch(b, rng);
        // Both faces, each with its own `right` so neither is back-facing.
        for s in [-1.0f32, 1.0] {
            let right = if along_x {
                Vector3::new(s, 0.0, 0.0)
            } else {
                Vector3::new(0.0, 0.0, s)
            };
            // Start at the end the face's own `right` walks away from.
            let out0 = 0.14;
            let corner = wall.at(bs, y + 0.18, out0);
            let o = if along_x {
                Vector3::new(corner.x + s * 0.07, corner.y, corner.z)
            } else {
                Vector3::new(corner.x, corner.y, corner.z + s * 0.07)
            };
            let up = Vector3::new(0.0, 1.0, 0.0);
            // Only draw the face whose normal points away from the wall's
            // plane on this side; `right x up` gives it.
            panel(m, o, right, up, bw * 0.85, bh * 0.85, scale_color(ink, 0.8));
            for i in 0..((bh * 0.85 / 0.30) as usize) {
                let t = 0.12 + i as f32 * 0.30;
                for e in [0.03f32, bw * 0.85 - 0.10] {
                    let p = Vector3::new(
                        o.x + right.x * e,
                        o.y + t,
                        o.z + right.z * e,
                    );
                    panel(m, p, right, up, 0.075, 0.075, Color::from_hex(0xfff3cf));
                }
            }
        }
    }
}

/// A large backlit advert on an otherwise blank flank wall.
pub(crate) fn add_wall_billboard(
    b: &mut Batches,
    a0: f32,
    a1: f32,
    fixed: f32,
    front: f32,
    along_x: bool,
    base_y: f32,
    rng: &mut Rng,
) {
    let wall = Wall {
        along_x,
        fixed,
        front,
    };
    let (x0, x1) = (a0, a1);
    let span = x1 - x0;
    if span < 7.0 {
        return;
    }
    let w = span * rng.range(0.5, 0.85);
    let h = (w * rng.range(0.42, 0.62)).min(9.0);
    let ox = mix(x0 + 0.5, x1 - w - 0.5, rng.f());
    let up = Vector3::new(0.0, 1.0, 0.0);
    let (right, start) = wall.frame(ox, w);

    // Frame, then a row of gantry lights leaning out over the top of it.
    let lo = wall.at(ox - 0.22, base_y - 0.22, 0.10 - 0.14);
    let hi = wall.at(ox + w + 0.22, base_y + h + 0.22, 0.10 + 0.14);
    b.trim.add_box(
        Vector3::new(lo.x.min(hi.x), lo.y, lo.z.min(hi.z)),
        Vector3::new(lo.x.max(hi.x), hi.y, lo.z.max(hi.z)),
        Color::from_hex(0x3c4045),
        Uv::Unit,
    );
    for k in 0..3 {
        let ls = ox + w * (0.2 + 0.3 * k as f32);
        b.trim.add_limb(
            wall.at(ls, base_y + h + 0.2, 0.10),
            wall.at(ls, base_y + h + 0.75, 0.65),
            0.05,
            0.05,
            4,
            Color::from_hex(0x2f3237),
            false,
        );
    }
    // The artwork goes on twice: once lit, so the hoarding is a printed
    // poster in daylight, and once unlit so the same poster is backlit after
    // dark. Drawing it only into the night batch left a black rectangle on
    // every wall in the city between sunrise and sunset. The clone is what
    // makes the two agree — both draws walk the same random sequence.
    let mut night_rng = rng.clone();
    ad_face(
        &mut b.trim,
        wall.at(start, base_y, 0.26),
        right,
        up,
        w,
        h,
        rng,
    );
    ad_face(
        neon_batch(b, &mut night_rng),
        wall.at(start, base_y, 0.30),
        right,
        up,
        w,
        h,
        &mut night_rng,
    );
}

/// A sign's frame: the direction its face runs, its origin, and where its legs
/// stand for a given offset along it.
type SignFrame = (Vector3, Vector3, Box<dyn Fn(f32) -> (f32, f32)>);

/// A hoarding standing on a roof, on legs, facing whichever way the block does.
pub(crate) fn add_roof_billboard(b: &mut Batches, lot: Rect, roof_y: f32, rng: &mut Rng) {
    let along_x = lot.w() >= lot.d();
    let span = if along_x { lot.w() } else { lot.d() };
    if span < 8.0 {
        return;
    }
    let w = span * rng.range(0.55, 0.9);
    let h = (w * rng.range(0.3, 0.5)).clamp(2.5, 8.0);
    let legs = rng.range(1.2, 2.6);
    let up = Vector3::new(0.0, 1.0, 0.0);
    let front = if rng.chance(0.5) { 1.0f32 } else { -1.0 };

    let (right, o, leg_at): SignFrame = if along_x {
        let ox = lot.cx() - w * 0.5;
        let z = lot.cz();
        let n = Vector3::new(0.0, 0.0, front);
        let r = up.cross(n);
        let start = if r.x > 0.0 { ox } else { ox + w };
        (
            r,
            Vector3::new(start, roof_y + legs, z),
            Box::new(move |t: f32| (ox + w * t, z)),
        )
    } else {
        let oz = lot.cz() - w * 0.5;
        let x = lot.cx();
        let n = Vector3::new(front, 0.0, 0.0);
        let r = up.cross(n);
        let start = if r.z > 0.0 { oz } else { oz + w };
        (
            r,
            Vector3::new(x, roof_y + legs, start),
            Box::new(move |t: f32| (x, oz + w * t)),
        )
    };

    let steel = Color::from_hex(0x44484d);
    for k in 0..4 {
        let (lx, lz) = leg_at(0.08 + 0.28 * k as f32);
        b.trim.add_cylinder(
            Vector3::new(lx, roof_y, lz),
            0.10,
            0.09,
            legs,
            5,
            steel,
            false,
            Uv::Unit,
        );
    }
    // Board back and frame, then the lit face on the front only — a hoarding
    // is blank from behind, and that is worth having when the camera swings
    // round to the other side of the block.
    let back = Vector3::new(
        o.x - right.cross(up).x * 0.10,
        o.y,
        o.z - right.cross(up).z * 0.10,
    );
    panel(
        &mut b.trim,
        back + Vector3::new(right.x * w, 0.0, right.z * w),
        Vector3::new(-right.x, 0.0, -right.z),
        up,
        w,
        h,
        Color::from_hex(0x35383d),
    );
    let mut night_rng = rng.clone();
    ad_face(&mut b.trim, o, right, up, w, h, rng);
    let lift = right.cross(up).normalize();
    ad_face(
        neon_batch(b, &mut night_rng),
        Vector3::new(o.x + lift.x * 0.04, o.y, o.z + lift.z * 0.04),
        right,
        up,
        w,
        h,
        &mut night_rng,
    );
}
