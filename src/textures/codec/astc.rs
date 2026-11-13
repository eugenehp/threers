//! ASTC, LDR profile, every 2D footprint: the decoder of the Khronos Data
//! Format specification (section "ASTC Compressed Texture Image Formats").
//!
//! A block is 128 bits: an 11-bit block mode giving the weight grid and its
//! quantisation, up to four partitions each with its own colour endpoint
//! mode, colour endpoints and weights both packed with integer sequence
//! encoding (bits, trits and quints), the weights from the top of the block
//! downwards. Anything malformed — and any HDR endpoint mode, which the LDR
//! profile does not decode — is the error colour, magenta.
//!
//! Interpolation is at 16 bits as the specification has it: an sRGB texel is
//! the top 8 bits of the result; a linear one is the 16-bit value rounded to 8.

const ERROR: [u8; 4] = [255, 0, 255, 255];

/// Bit `i` of a block upwards from bit 0.
struct Block(u128);

impl Block {
    fn bits(&self, lo: u32, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        ((self.0 >> lo) & ((1u128 << n) - 1)) as u32
    }
}

/// An ISE range: its trits/quints and plain bits.
#[derive(Clone, Copy)]
struct Range {
    trits: bool,
    quints: bool,
    bits: u32,
}

/// The 21 ISE ranges, by index: maximum values 1, 2, 3, 4, 5, 7, 9, 11, 15,
/// 19, 23, 31, 39, 47, 63, 79, 95, 127, 159, 191, 255.
const RANGES: [Range; 21] = {
    const fn b(bits: u32) -> Range {
        Range {
            trits: false,
            quints: false,
            bits,
        }
    }
    const fn t(bits: u32) -> Range {
        Range {
            trits: true,
            quints: false,
            bits,
        }
    }
    const fn q(bits: u32) -> Range {
        Range {
            trits: false,
            quints: true,
            bits,
        }
    }
    [
        b(1),
        t(0),
        b(2),
        q(0),
        t(1),
        b(3),
        q(1),
        t(2),
        b(4),
        q(2),
        t(3),
        b(5),
        q(3),
        t(4),
        b(6),
        q(4),
        t(5),
        b(7),
        q(5),
        t(6),
        b(8),
    ]
};

/// Bits `count` values take at this range.
fn ise_bits(r: Range, count: u32) -> u32 {
    let base = count * r.bits;
    if r.trits {
        base + (8 * count).div_ceil(5)
    } else if r.quints {
        base + (7 * count).div_ceil(3)
    } else {
        base
    }
}

/// The trit or quint (`.0`) and plain bits (`.1`) of each value.
type Ise = Vec<(u32, u32)>;

/// Decode `count` values of range `r` from `src` starting at bit `start`.
fn ise_decode(src: &Block, start: u32, r: Range, count: usize) -> Ise {
    let mut out = Vec::with_capacity(count);
    // A last, partial group's trit or quint bits run past the encoded data;
    // the specification has them read as zero, not as whatever follows in
    // the block. Reading on corrupted that group's earlier values too, since
    // its trits (or quints) are packed together.
    let end = start + ise_bits(r, count as u32);
    let mut pos = start;
    let mut read = |n: u32| {
        let take = n.min(end.saturating_sub(pos));
        let v = src.bits(pos, take);
        pos += n;
        v
    };
    if r.trits {
        while out.len() < count {
            // Five values: m0 T[1:0] m1 T[3:2] m2 T[4] m3 T[6:5] m4 T[7].
            let mut m = [0u32; 5];
            let mut t = 0u32;
            m[0] = read(r.bits);
            t |= read(2);
            m[1] = read(r.bits);
            t |= read(2) << 2;
            m[2] = read(r.bits);
            t |= read(1) << 4;
            m[3] = read(r.bits);
            t |= read(2) << 5;
            m[4] = read(r.bits);
            t |= read(1) << 7;
            let tr = trits(t);
            for k in 0..5 {
                if out.len() < count {
                    out.push((tr[k], m[k]));
                }
            }
        }
    } else if r.quints {
        while out.len() < count {
            // Three values: m0 Q[2:0] m1 Q[4:3] m2 Q[6:5].
            let mut m = [0u32; 3];
            let mut q = 0u32;
            m[0] = read(r.bits);
            q |= read(3);
            m[1] = read(r.bits);
            q |= read(2) << 3;
            m[2] = read(r.bits);
            q |= read(2) << 5;
            let qu = quints(q);
            for k in 0..3 {
                if out.len() < count {
                    out.push((qu[k], m[k]));
                }
            }
        }
    } else {
        for _ in 0..count {
            out.push((0, read(r.bits)));
        }
    }
    out
}

fn bit(v: u32, i: u32) -> u32 {
    (v >> i) & 1
}

/// Five trits from their 8-bit packing.
fn trits(t: u32) -> [u32; 5] {
    let (c, t4, t3);
    if (t >> 2) & 7 == 7 {
        c = ((t >> 5) & 7) << 2 | (t & 3);
        t4 = 2;
        t3 = 2;
    } else {
        c = t & 0x1f;
        if (t >> 5) & 3 == 3 {
            t4 = 2;
            t3 = bit(t, 7);
        } else {
            t4 = bit(t, 7);
            t3 = (t >> 5) & 3;
        }
    }
    let (t2, t1, t0);
    if c & 3 == 3 {
        t2 = 2;
        t1 = bit(c, 4);
        t0 = (bit(c, 3) << 1) | (bit(c, 2) & !bit(c, 3) & 1);
    } else if (c >> 2) & 3 == 3 {
        t2 = 2;
        t1 = 2;
        t0 = c & 3;
    } else {
        t2 = bit(c, 4);
        t1 = (c >> 2) & 3;
        t0 = (bit(c, 1) << 1) | (bit(c, 0) & !bit(c, 1) & 1);
    }
    [t0, t1, t2, t3, t4]
}

/// Three quints from their 7-bit packing.
fn quints(q: u32) -> [u32; 3] {
    let (q0, q1, q2);
    if (q >> 1) & 3 == 3 && (q >> 5) & 3 == 0 {
        q2 = (bit(q, 0) << 2) | ((bit(q, 4) & !bit(q, 0) & 1) << 1) | (bit(q, 3) & !bit(q, 0) & 1);
        q1 = 4;
        q0 = 4;
    } else {
        let c;
        if (q >> 1) & 3 == 3 {
            q2 = 4;
            c = (((q >> 3) & 3) << 3) | ((!(q >> 5) & 3) << 1) | bit(q, 0);
        } else {
            q2 = (q >> 5) & 3;
            c = q & 0x1f;
        }
        if c & 7 == 5 {
            q1 = 4;
            q0 = (c >> 3) & 3;
        } else {
            q1 = (c >> 3) & 3;
            q0 = c & 7;
        }
    }
    [q0, q1, q2]
}

/// Replicate the low `from` bits of `v` to fill `to` bits.
fn replicate(v: u32, from: u32, to: u32) -> u32 {
    if from == 0 {
        return 0;
    }
    let mut out = 0;
    let mut have = 0;
    while have < to {
        out = (out << from) | v;
        have += from;
    }
    out >> (have - to)
}

/// A colour endpoint value to 8 bits.
fn unquantize_colour(r: Range, (tq, m): (u32, u32)) -> u32 {
    if !r.trits && !r.quints {
        return replicate(m, r.bits, 8);
    }
    let a = if m & 1 == 1 { 0x1ff } else { 0 };
    let (b, c) = match (r.trits, r.bits) {
        (true, 1) => (0, 204),
        (true, 2) => {
            let b = bit(m, 1);
            ((b << 8) | (b << 4) | (b << 2) | (b << 1), 93)
        }
        (true, 3) => {
            let (cb, bb) = (bit(m, 2), bit(m, 1));
            (
                (cb << 8) | (bb << 7) | (cb << 3) | (bb << 2) | (cb << 1) | bb,
                44,
            )
        }
        (true, 4) => {
            let x = (m >> 1) & 7; // dcb
            ((x << 6) | x, 22)
        }
        (true, 5) => {
            let x = (m >> 1) & 0xf; // edcb
            ((x << 5) | (x >> 2), 11)
        }
        (true, 6) => {
            let x = (m >> 1) & 0x1f; // fedcb
            ((x << 4) | (x >> 4), 5)
        }
        (false, 1) => (0, 113),
        (false, 2) => {
            let b = bit(m, 1);
            ((b << 8) | (b << 3) | (b << 2), 54)
        }
        (false, 3) => {
            let x = (m >> 1) & 3; // cb
            ((x << 7) | (x << 1) | (x >> 1), 26)
        }
        (false, 4) => {
            let x = (m >> 1) & 7; // dcb
            ((x << 6) | (x >> 1), 13)
        }
        (false, 5) => {
            let x = (m >> 1) & 0xf; // edcb
            ((x << 5) | (x >> 3), 6)
        }
        _ => unreachable!("colour ranges start at 6 levels"),
    };
    let t = (tq * c + b) ^ a;
    (a & 0x80) | (t >> 2)
}

/// A weight to 0..=64.
fn unquantize_weight(r: Range, (tq, m): (u32, u32)) -> u32 {
    let v = if !r.trits && !r.quints {
        replicate(m, r.bits, 6)
    } else if r.bits == 0 {
        if r.trits {
            [0, 32, 63][tq as usize]
        } else {
            [0, 16, 32, 47, 63][tq as usize]
        }
    } else {
        let a = if m & 1 == 1 { 0x7f } else { 0 };
        let (b, c) = match (r.trits, r.bits) {
            (true, 1) => (0, 50),
            (true, 2) => {
                let b = bit(m, 1);
                ((b << 6) | (b << 2) | b, 23)
            }
            (true, 3) => {
                let x = (m >> 1) & 3; // cb
                ((x << 5) | x, 11)
            }
            (false, 1) => (0, 28),
            (false, 2) => {
                let b = bit(m, 1);
                ((b << 6) | (b << 1), 13)
            }
            _ => unreachable!("weight ranges stop at 32 levels"),
        };
        let t = (tq * c + b) ^ a;
        (a & 0x20) | (t >> 2)
    };
    if v > 32 {
        v + 1
    } else {
        v
    }
}

/// The weight grid a block mode gives: `(width, height, dual plane, range)`.
fn block_mode(mode: u32) -> Option<(u32, u32, bool, usize)> {
    let mut base = (mode >> 4) & 1;
    let mut high = (mode >> 9) & 1;
    let mut dual = (mode >> 10) & 1;
    let a = (mode >> 5) & 3;
    let (w, h);
    if mode & 3 != 0 {
        base |= (mode & 3) << 1;
        let b = (mode >> 7) & 3;
        match (mode >> 2) & 3 {
            0 => (w, h) = (b + 4, a + 2),
            1 => (w, h) = (b + 8, a + 2),
            2 => (w, h) = (a + 2, b + 8),
            _ => {
                let b = b & 1;
                if mode & 0x100 != 0 {
                    (w, h) = (b + 2, a + 2);
                } else {
                    (w, h) = (a + 2, b + 6);
                }
            }
        }
    } else {
        base |= ((mode >> 2) & 3) << 1;
        if (mode >> 2) & 3 == 0 {
            return None;
        }
        let b = (mode >> 9) & 3;
        match (mode >> 7) & 3 {
            0 => (w, h) = (12, a + 2),
            1 => (w, h) = (a + 2, 12),
            2 => {
                (w, h) = (a + 6, b + 6);
                dual = 0;
                high = 0;
            }
            _ => match (mode >> 5) & 3 {
                0 => (w, h) = (6, 10),
                1 => (w, h) = (10, 6),
                _ => return None,
            },
        }
    }
    let range = (base as i32 - 2 + 6 * high as i32) as usize;
    Some((w, h, dual == 1, range))
}

/// Which partition texel `(x, y)` is in: the specification's hash.
fn select_partition(seed: u32, x: u32, y: u32, partitions: u32, small: bool) -> usize {
    let (x, y) = if small { (x << 1, y << 1) } else { (x, y) };
    let seed = seed + (partitions - 1) * 1024;
    let mut p = seed;
    p ^= p >> 15;
    p = p.wrapping_sub(p << 17);
    p = p.wrapping_add(p << 7);
    p = p.wrapping_add(p << 4);
    p ^= p >> 5;
    p = p.wrapping_add(p << 16);
    p ^= p >> 7;
    p ^= p >> 3;
    p ^= p << 6;
    p ^= p >> 17;
    let rnum = p;
    let mut s = [
        rnum & 0xf,
        (rnum >> 4) & 0xf,
        (rnum >> 8) & 0xf,
        (rnum >> 12) & 0xf,
        (rnum >> 16) & 0xf,
        (rnum >> 20) & 0xf,
        (rnum >> 24) & 0xf,
        (rnum >> 28) & 0xf,
        (rnum >> 18) & 0xf,
        (rnum >> 22) & 0xf,
        (rnum >> 26) & 0xf,
        rnum.rotate_left(2) & 0xf,
    ];
    for v in s.iter_mut() {
        *v *= *v;
    }
    let (sh1, sh2) = if seed & 1 != 0 {
        (
            if seed & 2 != 0 { 4 } else { 5 },
            if partitions == 3 { 6 } else { 5 },
        )
    } else {
        (
            if partitions == 3 { 6 } else { 5 },
            if seed & 2 != 0 { 4 } else { 5 },
        )
    };
    let sh3 = if seed & 0x10 != 0 { sh1 } else { sh2 };
    for (i, v) in s.iter_mut().enumerate() {
        *v >>= if i < 8 {
            if i % 2 == 0 {
                sh1
            } else {
                sh2
            }
        } else {
            sh3
        };
    }
    let z = 0;
    let a = (s[0] * x + s[1] * y + s[10] * z + (rnum >> 14)) & 0x3f;
    let b = (s[2] * x + s[3] * y + s[11] * z + (rnum >> 10)) & 0x3f;
    let mut c = (s[4] * x + s[5] * y + s[8] * z + (rnum >> 6)) & 0x3f;
    let mut d = (s[6] * x + s[7] * y + s[9] * z + (rnum >> 2)) & 0x3f;
    if partitions < 4 {
        d = 0;
    }
    if partitions < 3 {
        c = 0;
    }
    if a >= b && a >= c && a >= d {
        0
    } else if b >= c && b >= d {
        1
    } else if c >= d {
        2
    } else {
        3
    }
}

fn clamp8(v: i32) -> u32 {
    v.clamp(0, 255) as u32
}

/// `a` takes the top bit of `b`'s bit transfer; returns `(a, b)`.
fn bit_transfer_signed(a: i32, b: i32) -> (i32, i32) {
    let b = (b >> 1) | (a & 0x80);
    let mut a = (a >> 1) & 0x3f;
    if a & 0x20 != 0 {
        a -= 0x40;
    }
    (a, b)
}

fn blue_contract(c: [i32; 4]) -> [i32; 4] {
    [(c[0] + c[2]) >> 1, (c[1] + c[2]) >> 1, c[2], c[3]]
}

/// An LDR colour endpoint mode's two endpoints, or `None` for an HDR mode.
fn endpoints(cem: u32, v: &[i32]) -> Option<([u32; 4], [u32; 4])> {
    let c = |x: [i32; 4]| x.map(clamp8);
    Some(match cem {
        0 => (c([v[0], v[0], v[0], 255]), c([v[1], v[1], v[1], 255])),
        1 => {
            let l0 = (v[0] >> 2) | (v[1] & 0xc0);
            let l1 = (l0 + (v[1] & 0x3f)).min(255);
            (c([l0, l0, l0, 255]), c([l1, l1, l1, 255]))
        }
        4 => (c([v[0], v[0], v[0], v[2]]), c([v[1], v[1], v[1], v[3]])),
        5 => {
            let (a1, a0) = bit_transfer_signed(v[1], v[0]);
            let (b1, b0) = bit_transfer_signed(v[3], v[2]);
            (c([a0, a0, a0, b0]), c([a0 + a1, a0 + a1, a0 + a1, b0 + b1]))
        }
        6 => (
            c([
                (v[0] * v[3]) >> 8,
                (v[1] * v[3]) >> 8,
                (v[2] * v[3]) >> 8,
                255,
            ]),
            c([v[0], v[1], v[2], 255]),
        ),
        8 => {
            if v[1] + v[3] + v[5] >= v[0] + v[2] + v[4] {
                (c([v[0], v[2], v[4], 255]), c([v[1], v[3], v[5], 255]))
            } else {
                (
                    c(blue_contract([v[1], v[3], v[5], 255])),
                    c(blue_contract([v[0], v[2], v[4], 255])),
                )
            }
        }
        9 => {
            let (r1, r0) = bit_transfer_signed(v[1], v[0]);
            let (g1, g0) = bit_transfer_signed(v[3], v[2]);
            let (b1, b0) = bit_transfer_signed(v[5], v[4]);
            if r1 + g1 + b1 >= 0 {
                (c([r0, g0, b0, 255]), c([r0 + r1, g0 + g1, b0 + b1, 255]))
            } else {
                (
                    c(blue_contract([r0 + r1, g0 + g1, b0 + b1, 255])),
                    c(blue_contract([r0, g0, b0, 255])),
                )
            }
        }
        10 => (
            c([
                (v[0] * v[3]) >> 8,
                (v[1] * v[3]) >> 8,
                (v[2] * v[3]) >> 8,
                v[4],
            ]),
            c([v[0], v[1], v[2], v[5]]),
        ),
        12 => {
            if v[1] + v[3] + v[5] >= v[0] + v[2] + v[4] {
                (c([v[0], v[2], v[4], v[6]]), c([v[1], v[3], v[5], v[7]]))
            } else {
                (
                    c(blue_contract([v[1], v[3], v[5], v[7]])),
                    c(blue_contract([v[0], v[2], v[4], v[6]])),
                )
            }
        }
        13 => {
            let (r1, r0) = bit_transfer_signed(v[1], v[0]);
            let (g1, g0) = bit_transfer_signed(v[3], v[2]);
            let (b1, b0) = bit_transfer_signed(v[5], v[4]);
            let (a1, a0) = bit_transfer_signed(v[7], v[6]);
            if r1 + g1 + b1 >= 0 {
                (c([r0, g0, b0, a0]), c([r0 + r1, g0 + g1, b0 + b1, a0 + a1]))
            } else {
                (
                    c(blue_contract([r0 + r1, g0 + g1, b0 + b1, a0 + a1])),
                    c(blue_contract([r0, g0, b0, a0])),
                )
            }
        }
        _ => return None, // 2, 3, 7, 11, 14, 15: HDR
    })
}

/// The 16-bit interpolation, then to 8 bits for the format's colour space.
fn interpolate(e0: u32, e1: u32, w: u32, srgb: bool) -> u8 {
    let (a, b) = if srgb {
        ((e0 << 8) | 0x80, (e1 << 8) | 0x80)
    } else {
        (e0 * 257, e1 * 257)
    };
    let c = (a * (64 - w) + b * w + 32) >> 6;
    if srgb {
        (c >> 8) as u8
    } else {
        ((c * 255 + 32767) / 65535) as u8
    }
}

fn fill(out: &mut [u8], n: usize, c: [u8; 4]) {
    for px in out[..n * 4].as_chunks_mut::<4>().0 {
        *px = c;
    }
}

/// One ASTC block of footprint `bw x bh` to `bw * bh` RGBA8 texels, row-major.
pub fn block(data: &[u8], bw: u32, bh: u32, srgb: bool, out: &mut [u8]) {
    let n = (bw * bh) as usize;
    let blk = Block(u128::from_le_bytes(data[..16].try_into().unwrap()));
    let mode = blk.bits(0, 11);

    // Void extent: one colour, as four UNORM16 channels.
    if mode & 0x1ff == 0x1fc {
        if blk.bits(9, 1) == 1 || blk.bits(10, 2) != 3 {
            return fill(out, n, ERROR); // HDR, or reserved bits not set
        }
        let ch = |i: u32| blk.bits(64 + 16 * i, 16);
        let c = [ch(0), ch(1), ch(2), ch(3)].map(|v| {
            if srgb {
                (v >> 8) as u8
            } else {
                ((v * 255 + 32767) / 65535) as u8
            }
        });
        // Extents that are not all ones must be ordered, or the block is malformed.
        let (s0, s1, t0, t1) = (
            blk.bits(12, 13),
            blk.bits(25, 13),
            blk.bits(38, 13),
            blk.bits(51, 13),
        );
        let all_ones = s0 == 0x1fff && s1 == 0x1fff && t0 == 0x1fff && t1 == 0x1fff;
        if !all_ones && (s0 >= s1 || t0 >= t1) {
            return fill(out, n, ERROR);
        }
        return fill(out, n, c);
    }

    let Some((gw, gh, dual, wrange)) = block_mode(mode) else {
        return fill(out, n, ERROR);
    };
    if gw > bw || gh > bh || wrange >= 12 {
        return fill(out, n, ERROR);
    }
    let planes = 1 + dual as u32;
    let wcount = gw * gh * planes;
    let wr = RANGES[wrange];
    let wbits = ise_bits(wr, wcount);
    if wcount > 64 || !(24..=96).contains(&wbits) {
        return fill(out, n, ERROR);
    }
    let partitions = blk.bits(11, 2) + 1;
    if partitions == 4 && dual {
        return fill(out, n, ERROR);
    }

    // Colour endpoint modes.
    let mut cems = [0u32; 4];
    let (colour_start, extra_bits);
    let below_weights = 128 - wbits;
    if partitions == 1 {
        cems[0] = blk.bits(13, 4);
        colour_start = 17;
        extra_bits = 0;
    } else {
        colour_start = 29;
        let low = blk.bits(23, 6);
        if low & 3 == 0 {
            let cem = (low >> 2) & 0xf;
            cems[..partitions as usize].fill(cem);
            extra_bits = 0;
        } else {
            extra_bits = 3 * partitions - 4;
            let high = blk.bits(below_weights - extra_bits, extra_bits);
            let enc = low | (high << 6);
            let class = (enc & 3) - 1;
            let mut pos = 2;
            let mut c = [0u32; 4];
            for v in c.iter_mut().take(partitions as usize) {
                *v = (enc >> pos) & 1;
                pos += 1;
            }
            for (i, v) in cems.iter_mut().take(partitions as usize).enumerate() {
                let m = (enc >> pos) & 3;
                pos += 2;
                *v = ((class + c[i]) << 2) | m;
            }
        }
    }
    let ccs_bits = if dual { 2 } else { 0 };
    let colour_end = below_weights - extra_bits - ccs_bits;
    let ccs = if dual {
        blk.bits(colour_end, 2) as usize
    } else {
        0
    };
    if colour_end < colour_start {
        return fill(out, n, ERROR);
    }
    let ncolour: u32 = cems[..partitions as usize]
        .iter()
        .map(|c| 2 * ((c >> 2) + 1))
        .sum();
    if ncolour > 18 {
        return fill(out, n, ERROR);
    }
    let avail = colour_end - colour_start;
    let Some(crange) = (4..21)
        .rev()
        .find(|&r| ise_bits(RANGES[r], ncolour) <= avail)
    else {
        return fill(out, n, ERROR);
    };
    let cr = RANGES[crange];
    let colours: Vec<i32> = ise_decode(&blk, colour_start, cr, ncolour as usize)
        .into_iter()
        .map(|v| unquantize_colour(cr, v) as i32)
        .collect();

    // Endpoints per partition. An HDR endpoint mode, which the LDR profile
    // does not decode, makes its own partition's texels the error colour and
    // leaves the others alone — as the hardware does.
    let mut ends = [None; 4];
    let mut at = 0;
    for p in 0..partitions as usize {
        let k = 2 * ((cems[p] >> 2) + 1) as usize;
        ends[p] = endpoints(cems[p], &colours[at..at + k]);
        at += k;
    }

    // Weights, read from bit 127 downwards: reverse the block and read up.
    let rev = Block(blk.0.reverse_bits());
    let weights: Vec<u32> = ise_decode(&rev, 0, wr, wcount as usize)
        .into_iter()
        .map(|v| unquantize_weight(wr, v))
        .collect();

    // Infill from the grid to the texels.
    let ds = (1024 + bw / 2) / (bw - 1);
    let dt = (1024 + bh / 2) / (bh - 1);
    let seed = blk.bits(13, 10);
    let small = n < 31;
    for t in 0..bh {
        for s in 0..bw {
            let cs = ds * s;
            let ct = dt * t;
            let gs = (cs * (gw - 1) + 32) >> 6;
            let gt = (ct * (gh - 1) + 32) >> 6;
            let (js, fs, jt, ft) = (gs >> 4, gs & 0xf, gt >> 4, gt & 0xf);
            let w11 = (fs * ft + 8) >> 4;
            let w10 = ft - w11;
            let w01 = fs - w11;
            let w00 = 16 - fs - ft + w11;
            let weight = |plane: u32| {
                let at = |x: u32, y: u32| -> u32 {
                    if x >= gw || y >= gh {
                        0
                    } else {
                        weights[((y * gw + x) * planes + plane) as usize]
                    }
                };
                (at(js, jt) * w00
                    + at(js + 1, jt) * w01
                    + at(js, jt + 1) * w10
                    + at(js + 1, jt + 1) * w11
                    + 8)
                    >> 4
            };
            let w0 = weight(0);
            let w1 = if dual { weight(1) } else { w0 };
            let p = if partitions > 1 {
                select_partition(seed, s, t, partitions, small)
            } else {
                0
            };
            let d = ((t * bw + s) * 4) as usize;
            let Some((e0, e1)) = ends[p] else {
                out[d..d + 4].copy_from_slice(&ERROR);
                continue;
            };
            for c in 0..4 {
                let w = if dual && c == ccs { w1 } else { w0 };
                out[d + c] = interpolate(e0[c], e1[c], w, srgb);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trits_and_quints_decode_every_packing_to_valid_digits() {
        let mut seen = std::collections::HashSet::new();
        for t in 0..256 {
            let tr = trits(t);
            assert!(tr.iter().all(|&d| d < 3), "{t:#x} -> {tr:?}");
            seen.insert(tr);
        }
        assert_eq!(
            seen.len(),
            243,
            "every combination of five trits is reachable"
        );
        let mut seen = std::collections::HashSet::new();
        for q in 0..128 {
            let qu = quints(q);
            assert!(qu.iter().all(|&d| d < 5), "{q:#x} -> {qu:?}");
            seen.insert(qu);
        }
        assert_eq!(
            seen.len(),
            125,
            "every combination of three quints is reachable"
        );
    }

    #[test]
    fn weights_unquantize_monotonically_to_0_and_64() {
        for (i, r) in RANGES.iter().enumerate().take(12) {
            let levels = if r.trits {
                3
            } else if r.quints {
                5
            } else {
                1
            } << r.bits;
            let mut vals: Vec<u32> = (0..levels)
                .map(|k| {
                    let (tq, m) = if r.trits || r.quints {
                        (k >> r.bits, k & ((1 << r.bits) - 1))
                    } else {
                        (0, k)
                    };
                    unquantize_weight(*r, (tq, m))
                })
                .collect();
            vals.sort();
            vals.dedup();
            assert_eq!(vals.len() as u32, levels, "range {i}: distinct levels");
            assert_eq!((vals[0], *vals.last().unwrap()), (0, 64), "range {i}");
        }
    }

    #[test]
    fn colours_unquantize_to_distinct_levels_spanning_0_to_255() {
        for (i, r) in RANGES.iter().enumerate().skip(4) {
            let levels = if r.trits {
                3
            } else if r.quints {
                5
            } else {
                1
            } << r.bits;
            let mut vals: Vec<u32> = (0..levels)
                .map(|k| {
                    let (tq, m) = if r.trits || r.quints {
                        (k >> r.bits, k & ((1 << r.bits) - 1))
                    } else {
                        (0, k)
                    };
                    unquantize_colour(*r, (tq, m))
                })
                .collect();
            vals.sort();
            vals.dedup();
            assert_eq!(vals.len() as u32, levels, "range {i}");
            assert_eq!((vals[0], *vals.last().unwrap()), (0, 255), "range {i}");
        }
    }
}
