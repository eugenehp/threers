//! BC7 (BPTC unorm): eight block modes, one to three subsets, 2- to 4-bit
//! indices, optional p-bits and a channel rotation. Exact integer decoding as
//! the D3D11 and Khronos specifications give it, so a decoded block matches
//! the hardware bit for bit.

/// A little-endian bit reader over one 128-bit block.
pub(super) struct Bits {
    v: u128,
    pos: u32,
}

impl Bits {
    pub(super) fn new(block: &[u8]) -> Self {
        let mut b = [0u8; 16];
        b.copy_from_slice(&block[..16]);
        Self {
            v: u128::from_le_bytes(b),
            pos: 0,
        }
    }

    pub(super) fn read(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        let r = ((self.v >> self.pos) & ((1u128 << n) - 1)) as u32;
        self.pos += n;
        r
    }
}

/// Subset of each texel for the 64 two-subset partitions, bit `i` set when
/// texel `i` (x + 4y) is in subset 1. BC6H uses the first 32.
pub(super) const PARTITION2: [u16; 64] = [
    0xcccc, 0x8888, 0xeeee, 0xecc8, 0xc880, 0xfeec, 0xfec8, 0xec80, 0xc800, 0xffec, 0xfe80, 0xe800,
    0xffe8, 0xff00, 0xfff0, 0xf000, 0xf710, 0x008e, 0x7100, 0x08ce, 0x008c, 0x7310, 0x3100, 0x8cce,
    0x088c, 0x3110, 0x6666, 0x366c, 0x17e8, 0x0ff0, 0x718e, 0x399c, 0xaaaa, 0xf0f0, 0x5a5a, 0x33cc,
    0x3c3c, 0x55aa, 0x9696, 0xa55a, 0x73ce, 0x13c8, 0x324c, 0x3bdc, 0x6996, 0xc33c, 0x9966, 0x0660,
    0x0272, 0x04e4, 0x4e40, 0x2720, 0xc936, 0x936c, 0x39c6, 0x639c, 0x9336, 0x9cc6, 0x817e, 0xe718,
    0xccf0, 0x0fcc, 0x7744, 0xee22,
];

/// Subset of each texel for the 64 three-subset partitions, two bits a texel.
const PARTITION3: [u32; 64] = [
    0xaa685050, 0x6a5a5040, 0x5a5a4200, 0x5450a0a8, 0xa5a50000, 0xa0a05050, 0x5555a0a0, 0x5a5a5050,
    0xaa550000, 0xaa555500, 0xaaaa5500, 0x90909090, 0x94949494, 0xa4a4a4a4, 0xa9a59450, 0x2a0a4250,
    0xa5945040, 0x0a425054, 0xa5a5a500, 0x55a0a0a0, 0xa8a85454, 0x6a6a4040, 0xa4a45000, 0x1a1a0500,
    0x0050a4a4, 0xaaa59090, 0x14696914, 0x69691400, 0xa08585a0, 0xaa821414, 0x50a4a450, 0x6a5a0200,
    0xa9a58000, 0x5090a0a8, 0xa8a09050, 0x24242424, 0x00aa5500, 0x24924924, 0x24499224, 0x50a50a50,
    0x500aa550, 0xaaaa4444, 0x66660000, 0xa5a0a5a0, 0x50a050a0, 0x69286928, 0x44aaaa44, 0x66666600,
    0xaa444444, 0x54a854a8, 0x95809580, 0x96969600, 0xa85454a8, 0x80959580, 0xaa141414, 0x96960000,
    0xaaaa1414, 0xa05050a0, 0xa0a5a5a0, 0x96000000, 0x40804080, 0xa9a8a9a8, 0xaaaaaa44, 0x2a4a5254,
];

/// Anchor texel of subset 1 in each two-subset partition (subset 0's is 0).
pub(super) const ANCHOR2: [u8; 64] = [
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15, 2, 8,
    2, 2, 8, 8, 2, 2, 15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6, 6, 2, 6, 8, 15, 15, 2,
    2, 15, 15, 15, 15, 15, 2, 2, 15,
];

/// Anchor texels of subsets 1 and 2 in each three-subset partition.
const ANCHOR3A: [u8; 64] = [
    3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3, 3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6, 8, 5,
    15, 15, 8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15, 3, 15, 5, 5, 5, 8, 5, 10, 5,
    10, 8, 13, 15, 12, 3, 3,
];
const ANCHOR3B: [u8; 64] = [
    15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8, 15, 8, 15, 3, 15, 8, 15, 8, 3, 15, 6,
    10, 15, 15, 10, 8, 15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8, 15, 3, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8,
];

/// Interpolation weights out of 64, by index width.
pub(super) const WEIGHTS2: [u32; 4] = [0, 21, 43, 64];
pub(super) const WEIGHTS3: [u32; 8] = [0, 9, 18, 27, 37, 46, 55, 64];
pub(super) const WEIGHTS4: [u32; 16] =
    [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

pub(super) fn weights(bits: u32) -> &'static [u32] {
    match bits {
        2 => &WEIGHTS2,
        3 => &WEIGHTS3,
        _ => &WEIGHTS4,
    }
}

/// Subset of texel `i` in partition `p` of an `n`-subset mode.
fn subset(n: usize, p: usize, i: usize) -> usize {
    match n {
        1 => 0,
        2 => ((PARTITION2[p] >> i) & 1) as usize,
        _ => ((PARTITION3[p] >> (2 * i)) & 3) as usize,
    }
}

/// Whether texel `i` is an anchor (its index is stored one bit short).
fn is_anchor(n: usize, p: usize, i: usize) -> bool {
    i == 0
        || match n {
            2 => i == ANCHOR2[p] as usize,
            3 => i == ANCHOR3A[p] as usize || i == ANCHOR3B[p] as usize,
            _ => false,
        }
}

struct Mode {
    subsets: usize,
    partition_bits: u32,
    rotation_bits: u32,
    index_select_bits: u32,
    colour_bits: u32,
    alpha_bits: u32,
    endpoint_pbits: bool,
    shared_pbits: bool,
    index_bits: u32,
    index2_bits: u32,
}

#[rustfmt::skip]
const MODES: [Mode; 8] = [
    Mode { subsets: 3, partition_bits: 4, rotation_bits: 0, index_select_bits: 0, colour_bits: 4, alpha_bits: 0, endpoint_pbits: true, shared_pbits: false, index_bits: 3, index2_bits: 0 },
    Mode { subsets: 2, partition_bits: 6, rotation_bits: 0, index_select_bits: 0, colour_bits: 6, alpha_bits: 0, endpoint_pbits: false, shared_pbits: true, index_bits: 3, index2_bits: 0 },
    Mode { subsets: 3, partition_bits: 6, rotation_bits: 0, index_select_bits: 0, colour_bits: 5, alpha_bits: 0, endpoint_pbits: false, shared_pbits: false, index_bits: 2, index2_bits: 0 },
    Mode { subsets: 2, partition_bits: 6, rotation_bits: 0, index_select_bits: 0, colour_bits: 7, alpha_bits: 0, endpoint_pbits: true, shared_pbits: false, index_bits: 2, index2_bits: 0 },
    Mode { subsets: 1, partition_bits: 0, rotation_bits: 2, index_select_bits: 1, colour_bits: 5, alpha_bits: 6, endpoint_pbits: false, shared_pbits: false, index_bits: 2, index2_bits: 3 },
    Mode { subsets: 1, partition_bits: 0, rotation_bits: 2, index_select_bits: 0, colour_bits: 7, alpha_bits: 8, endpoint_pbits: false, shared_pbits: false, index_bits: 2, index2_bits: 2 },
    Mode { subsets: 1, partition_bits: 0, rotation_bits: 0, index_select_bits: 0, colour_bits: 7, alpha_bits: 7, endpoint_pbits: true, shared_pbits: false, index_bits: 4, index2_bits: 0 },
    Mode { subsets: 2, partition_bits: 6, rotation_bits: 0, index_select_bits: 0, colour_bits: 5, alpha_bits: 5, endpoint_pbits: true, shared_pbits: false, index_bits: 2, index2_bits: 0 },
];

/// Interpolate two 8-bit endpoints by a 0..=64 weight, as BC7 does.
fn interp(e0: u32, e1: u32, w: u32) -> u8 {
    (((64 - w) * e0 + w * e1 + 32) >> 6) as u8
}

/// One BC7 block to 16 RGBA8 texels, row-major.
pub fn block(block: &[u8], out: &mut [u8]) {
    let mut r = Bits::new(block);
    let mut mode_n = 0;
    while mode_n < 8 && r.read(1) == 0 {
        mode_n += 1;
    }
    if mode_n == 8 {
        // Reserved: the specification decodes it to transparent black.
        out[..64].fill(0);
        return;
    }
    let m = &MODES[mode_n];
    let partition = r.read(m.partition_bits) as usize;
    let rotation = r.read(m.rotation_bits);
    let index_select = r.read(m.index_select_bits);

    // Endpoints: every R, then every G, then B, then A; two per subset.
    let ne = m.subsets * 2;
    let mut ep = [[0u32; 4]; 6];
    for c in 0..3 {
        for e in ep.iter_mut().take(ne) {
            e[c] = r.read(m.colour_bits);
        }
    }
    for e in ep.iter_mut().take(ne) {
        e[3] = r.read(m.alpha_bits);
    }
    // P-bits: one per endpoint, or one per subset shared by its two.
    let mut pbits = [0u32; 6];
    if m.endpoint_pbits {
        for p in pbits.iter_mut().take(ne) {
            *p = r.read(1);
        }
    } else if m.shared_pbits {
        for s in 0..m.subsets {
            let p = r.read(1);
            pbits[2 * s] = p;
            pbits[2 * s + 1] = p;
        }
    }
    let has_p = m.endpoint_pbits || m.shared_pbits;
    let expand = |v: u32, bits: u32| -> u32 {
        // Left-align to 8 bits and fill the bottom with the top bits.
        (v << (8 - bits)) | (v >> (2 * bits).saturating_sub(8).min(bits))
    };
    for (e, p) in ep.iter_mut().take(ne).zip(pbits) {
        for (c, v) in e.iter_mut().enumerate() {
            let bits = if c < 3 { m.colour_bits } else { m.alpha_bits };
            if bits == 0 {
                *v = 255;
                continue;
            }
            let (val, bits) = if has_p {
                ((*v << 1) | p, bits + 1)
            } else {
                (*v, bits)
            };
            *v = if bits >= 8 { val } else { expand(val, bits) };
        }
    }

    // Indices: primary for every texel, then the secondary set (modes 4, 5).
    let mut idx = [0u32; 16];
    for (i, v) in idx.iter_mut().enumerate() {
        let short = is_anchor(m.subsets, partition, i);
        *v = r.read(m.index_bits - short as u32);
    }
    let mut idx2 = [0u32; 16];
    if m.index2_bits > 0 {
        for (i, v) in idx2.iter_mut().enumerate() {
            *v = r.read(m.index2_bits - (i == 0) as u32);
        }
    }

    for i in 0..16 {
        let s = subset(m.subsets, partition, i);
        let (e0, e1) = (ep[2 * s], ep[2 * s + 1]);
        let mut px = [0u8; 4];
        if m.index2_bits > 0 {
            // Colour and alpha from different index sets; mode 4's selector
            // swaps which is which.
            let (ci, cb, ai, ab) = if index_select == 0 {
                (idx[i], m.index_bits, idx2[i], m.index2_bits)
            } else {
                (idx2[i], m.index2_bits, idx[i], m.index_bits)
            };
            let cw = weights(cb)[ci as usize];
            let aw = weights(ab)[ai as usize];
            for c in 0..3 {
                px[c] = interp(e0[c], e1[c], cw);
            }
            px[3] = interp(e0[3], e1[3], aw);
        } else {
            let w = weights(m.index_bits)[idx[i] as usize];
            for c in 0..4 {
                px[c] = interp(e0[c], e1[c], w);
            }
        }
        match rotation {
            1 => px.swap(0, 3),
            2 => px.swap(1, 3),
            3 => px.swap(2, 3),
            _ => {}
        }
        out[i * 4..i * 4 + 4].copy_from_slice(&px);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_anchor_is_in_its_own_subset() {
        for p in 0..64 {
            assert_eq!(
                subset(2, p, ANCHOR2[p] as usize),
                1,
                "2-subset partition {p}"
            );
            assert_eq!(subset(2, p, 0), 0);
            assert_eq!(
                subset(3, p, ANCHOR3A[p] as usize),
                1,
                "3-subset partition {p}"
            );
            assert_eq!(
                subset(3, p, ANCHOR3B[p] as usize),
                2,
                "3-subset partition {p}"
            );
            assert_eq!(subset(3, p, 0), 0);
        }
    }

    #[test]
    fn every_block_reads_exactly_128_bits() {
        for (n, m) in MODES.iter().enumerate() {
            let ne = m.subsets as u32 * 2;
            let pbits = if m.endpoint_pbits {
                ne
            } else if m.shared_pbits {
                m.subsets as u32
            } else {
                0
            };
            let bits = (n as u32 + 1)
                + m.partition_bits
                + m.rotation_bits
                + m.index_select_bits
                + ne * (3 * m.colour_bits + m.alpha_bits)
                + pbits
                + 16 * m.index_bits
                - m.subsets as u32
                + if m.index2_bits > 0 {
                    16 * m.index2_bits - 1
                } else {
                    0
                };
            assert_eq!(bits, 128, "mode {n}");
        }
    }
}
