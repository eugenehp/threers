//! ETC2 RGB and the EAC alpha block of ETC2 RGBA, as the Khronos Data Format
//! and OpenGL ES 3.0 specifications define them.
//!
//! ETC blocks are big-endian 64-bit words, and their pixel indices run down
//! columns (index `x * 4 + y`). The output here is row-major RGBA8 like every
//! other decoder in this module.

/// ETC1's intensity modifiers, by table codeword: `[a, b]` give the four
/// values `+a, +b, -a, -b` for pixel index `(msb, lsb)` = 00, 01, 10, 11.
const MODIFIERS: [[i32; 2]; 8] = [
    [2, 8],
    [5, 17],
    [9, 29],
    [13, 42],
    [18, 60],
    [24, 80],
    [33, 106],
    [47, 183],
];

/// T and H mode distances.
const DISTANCES: [i32; 8] = [3, 6, 11, 16, 23, 32, 41, 64];

/// EAC's sixteen modifier tables.
const EAC: [[i32; 8]; 16] = [
    [-3, -6, -9, -15, 2, 5, 8, 14],
    [-3, -7, -10, -13, 2, 6, 9, 12],
    [-2, -5, -8, -13, 1, 4, 7, 12],
    [-2, -4, -6, -13, 1, 3, 5, 12],
    [-3, -6, -8, -12, 2, 5, 7, 11],
    [-3, -7, -9, -11, 2, 6, 8, 10],
    [-4, -7, -8, -11, 3, 6, 7, 10],
    [-3, -5, -8, -11, 2, 4, 7, 10],
    [-2, -6, -8, -10, 1, 5, 7, 9],
    [-2, -5, -8, -10, 1, 4, 7, 9],
    [-2, -4, -8, -10, 1, 3, 7, 9],
    [-2, -5, -7, -10, 1, 4, 6, 9],
    [-3, -4, -7, -10, 2, 3, 6, 9],
    [-1, -2, -3, -10, 0, 1, 2, 9],
    [-4, -6, -8, -9, 3, 5, 7, 8],
    [-3, -5, -7, -9, 2, 4, 6, 8],
];

fn clamp8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

fn ext4(v: u64) -> i32 {
    let v = (v & 0xf) as i32;
    (v << 4) | v
}

fn ext5(v: i32) -> i32 {
    (v << 3) | (v >> 2)
}

fn bits(w: u64, lo: u32, n: u32) -> u64 {
    (w >> lo) & ((1 << n) - 1)
}

/// The 2-bit index of pixel `(x, y)`: its MSB from the high half of the low
/// word, its LSB from the low half.
fn pixel_index(w: u64, x: usize, y: usize) -> usize {
    let i = x * 4 + y;
    ((((w >> (16 + i)) & 1) << 1) | ((w >> i) & 1)) as usize
}

/// One ETC2 RGB block (8 bytes) to 16 RGBA8 texels, row-major, alpha 255.
pub fn rgb_block(block: &[u8], out: &mut [u8]) {
    let w = u64::from_be_bytes(block[..8].try_into().unwrap());
    let put = |out: &mut [u8], x: usize, y: usize, c: [i32; 3]| {
        let d = (y * 4 + x) * 4;
        out[d..d + 4].copy_from_slice(&[clamp8(c[0]), clamp8(c[1]), clamp8(c[2]), 255]);
    };
    let diff = (w >> 33) & 1 == 1;
    let flip = (w >> 32) & 1 == 1;

    if !diff {
        // Individual: two 4-bit base colours.
        let c1 = [ext4(w >> 60), ext4(w >> 52), ext4(w >> 44)];
        let c2 = [ext4(w >> 56), ext4(w >> 48), ext4(w >> 40)];
        let (t1, t2) = (bits(w, 37, 3) as usize, bits(w, 34, 3) as usize);
        return etc1_subblocks(w, flip, [c1, c2], [t1, t2], out, put);
    }

    let r = bits(w, 59, 5) as i32;
    let g = bits(w, 51, 5) as i32;
    let b = bits(w, 43, 5) as i32;
    let dr = ((bits(w, 56, 3) as i32) << 29) >> 29;
    let dg = ((bits(w, 48, 3) as i32) << 29) >> 29;
    let db = ((bits(w, 40, 3) as i32) << 29) >> 29;
    let (r2, g2, b2) = (r + dr, g + dg, b + db);

    if !(0..=31).contains(&r2) {
        // T mode.
        let r1 = (bits(w, 59, 2) << 2) | bits(w, 56, 2);
        let c1 = [ext4(r1), ext4(w >> 52), ext4(w >> 48)];
        let c2 = [ext4(w >> 44), ext4(w >> 40), ext4(w >> 36)];
        let d = DISTANCES[((bits(w, 34, 2) << 1) | bits(w, 32, 1)) as usize];
        let pal = [c1, c2.map(|v| v + d), c2, c2.map(|v| v - d)];
        for y in 0..4 {
            for x in 0..4 {
                put(out, x, y, pal[pixel_index(w, x, y)]);
            }
        }
    } else if !(0..=31).contains(&g2) {
        // H mode.
        let g1 = (bits(w, 56, 3) << 1) | bits(w, 52, 1);
        let b1 = (bits(w, 51, 1) << 3) | bits(w, 47, 3);
        let c1 = [ext4(w >> 59), ext4(g1), ext4(b1)];
        let c2 = [ext4(w >> 43), ext4(w >> 39), ext4(w >> 35)];
        let key = |c: [i32; 3]| (c[0] << 16) | (c[1] << 8) | c[2];
        let d_idx = (bits(w, 34, 1) << 2) | (bits(w, 32, 1) << 1) | (key(c1) >= key(c2)) as u64;
        let d = DISTANCES[d_idx as usize];
        let pal = [
            c1.map(|v| v + d),
            c1.map(|v| v - d),
            c2.map(|v| v + d),
            c2.map(|v| v - d),
        ];
        for y in 0..4 {
            for x in 0..4 {
                put(out, x, y, pal[pixel_index(w, x, y)]);
            }
        }
    } else if !(0..=31).contains(&b2) {
        // Planar: three colours, origin, horizontal and vertical.
        let e6 = |v: u64| ((v << 2) | (v >> 4)) as i32;
        let e7 = |v: u64| ((v << 1) | (v >> 6)) as i32;
        let ro = e6(bits(w, 57, 6));
        let go = e7((bits(w, 56, 1) << 6) | bits(w, 49, 6));
        let bo = e6((bits(w, 48, 1) << 5) | (bits(w, 43, 2) << 3) | bits(w, 39, 3));
        let rh = e6((bits(w, 34, 5) << 1) | bits(w, 32, 1));
        let gh = e7(bits(w, 25, 7));
        let bh = e6(bits(w, 19, 6));
        let rv = e6(bits(w, 13, 6));
        let gv = e7(bits(w, 6, 7));
        let bv = e6(bits(w, 0, 6));
        for y in 0..4i32 {
            for x in 0..4i32 {
                let f = |o: i32, h: i32, v: i32| (x * (h - o) + y * (v - o) + 4 * o + 2) >> 2;
                put(
                    out,
                    x as usize,
                    y as usize,
                    [f(ro, rh, rv), f(go, gh, gv), f(bo, bh, bv)],
                );
            }
        }
    } else {
        // Differential: a 5-bit base colour and a 3-bit offset to the second.
        let c1 = [ext5(r), ext5(g), ext5(b)];
        let c2 = [ext5(r2), ext5(g2), ext5(b2)];
        let (t1, t2) = (bits(w, 37, 3) as usize, bits(w, 34, 3) as usize);
        etc1_subblocks(w, flip, [c1, c2], [t1, t2], out, put);
    }
}

/// ETC1's two subblocks, side by side (2x4 each) or, flipped, stacked (4x2).
fn etc1_subblocks(
    w: u64,
    flip: bool,
    base: [[i32; 3]; 2],
    table: [usize; 2],
    out: &mut [u8],
    put: impl Fn(&mut [u8], usize, usize, [i32; 3]),
) {
    for y in 0..4 {
        for x in 0..4 {
            let sub = if flip {
                (y >= 2) as usize
            } else {
                (x >= 2) as usize
            };
            let [a, b] = MODIFIERS[table[sub]];
            let m = [a, b, -a, -b][pixel_index(w, x, y)];
            put(out, x, y, base[sub].map(|c| c + m));
        }
    }
}

/// One EAC block (8 bytes) to 16 values, written every `stride` bytes of
/// `out` from 0: ETC2 RGBA's alpha.
pub fn eac_block(block: &[u8], out: &mut [u8], stride: usize) {
    let w = u64::from_be_bytes(block[..8].try_into().unwrap());
    let base = bits(w, 56, 8) as i32;
    let mult = bits(w, 52, 4) as i32;
    let table = &EAC[bits(w, 48, 4) as usize];
    for y in 0..4 {
        for x in 0..4 {
            let i = x * 4 + y;
            let idx = bits(w, 45 - 3 * i as u32, 3) as usize;
            out[(y * 4 + x) * stride] = clamp8(base + table[idx] * mult);
        }
    }
}

/// ETC2 RGBA: an EAC alpha block, then an ETC2 RGB block.
pub fn rgba_block(block: &[u8], out: &mut [u8]) {
    rgb_block(&block[8..16], out);
    eac_block(&block[0..8], &mut out[3..], 4);
}
