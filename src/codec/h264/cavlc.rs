//! CAVLC residual coding (ISO/IEC 14496-10 §9.2).
//!
//! Baseline profile's entropy coder. Where CABAC adapts a probability per bin,
//! CAVLC switches between fixed VLC tables using context that both sides can
//! derive — chiefly `nC`, the neighbouring blocks' coefficient counts. Coding a
//! block is five stages: `coeff_token`, the trailing ±1 signs, the remaining
//! levels, `total_zeros`, and the runs between coefficients.
//!
//! Coefficients are handled in **reverse** scan order throughout: the highest
//! frequency first, because that is where the runs of zeros are.
//!
//! # On the tables
//!
//! These are transcribed constants, and a wrong entry is silently wrong in a way
//! an encode↔decode round-trip through the same table cannot see. So the tests
//! below check properties that hold regardless of the values: every table must
//! be prefix-free and satisfy Kraft's inequality, with the right number of
//! entries. That is a real constraint — it caught a `coeff_token` entry coded
//! one bit short, which pushed its table's Kraft sum above 1 (impossible) and
//! made it a prefix of another codeword. It is not a proof of correctness
//! though: two different tables can both be valid prefix codes. Conformance
//! comes from `tests/h264_compress.rs`, which decodes with ffmpeg.

use crate::codec::bitstream::BitWriter;

/// One variable-length code: `len` bits, value right-aligned in `code`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vlc {
    pub len: u8,
    pub code: u32,
}

const fn v(len: u8, code: u32) -> Vlc {
    Vlc { len, code }
}

/// Zig-zag scan for a 4×4 block (Table 8-13): scan position → raster index.
pub const ZIGZAG4: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

/// `coeff_token`, indexed `[trailing_ones][total_coeff]` (Table 9-5).
/// Entries where `trailing_ones > total_coeff` are unused.
type CoeffTokenTable = [[Vlc; 17]; 4];

const NA: Vlc = v(0, 0);

/// `0 <= nC < 2`.
const CT0: CoeffTokenTable = [
    [
        v(1, 0b1), v(6, 0b000101), v(8, 0b00000111), v(9, 0b000000111),
        v(10, 0b0000000111), v(11, 0b00000000111), v(13, 0b0000000001111),
        v(13, 0b0000000001011), v(13, 0b0000000001000), v(14, 0b00000000001111),
        v(14, 0b00000000001011), v(15, 0b000000000001111), v(15, 0b000000000001011),
        v(16, 0b0000000000001111), v(16, 0b0000000000001011), v(16, 0b0000000000000111),
        v(16, 0b0000000000000100),
    ],
    [
        NA, v(2, 0b01), v(6, 0b000100), v(8, 0b00000110), v(9, 0b000000110),
        v(10, 0b0000000110), v(11, 0b00000000110), v(13, 0b0000000001110),
        v(13, 0b0000000001010), v(14, 0b00000000001110), v(14, 0b00000000001010),
        v(15, 0b000000000001110), v(15, 0b000000000001010), v(15, 0b000000000000001),
        v(16, 0b0000000000001110), v(16, 0b0000000000001010), v(16, 0b0000000000000110),
    ],
    [
        NA, NA, v(3, 0b001), v(7, 0b0000101), v(8, 0b00000101), v(9, 0b000000101),
        v(10, 0b0000000101), v(11, 0b00000000101), v(13, 0b0000000001101),
        v(13, 0b0000000001001), v(14, 0b00000000001101), v(14, 0b00000000001001),
        v(15, 0b000000000001101), v(15, 0b000000000001001), v(16, 0b0000000000001101),
        v(16, 0b0000000000001001), v(16, 0b0000000000000101),
    ],
    [
        NA, NA, NA, v(5, 0b00011), v(6, 0b000011), v(7, 0b0000100), v(8, 0b00000100),
        v(9, 0b000000100), v(10, 0b0000000100), v(11, 0b00000000100),
        v(13, 0b0000000001100), v(14, 0b00000000001100), v(14, 0b00000000001000),
        v(15, 0b000000000001100), v(15, 0b000000000001000), v(16, 0b0000000000001100),
        v(16, 0b0000000000001000),
    ],
];

/// `2 <= nC < 4`.
const CT1: CoeffTokenTable = [
    [
        v(2, 0b11), v(6, 0b001011), v(6, 0b000111), v(7, 0b0000111), v(8, 0b00000111),
        v(8, 0b00000100), v(9, 0b000000111), v(11, 0b00000001111), v(11, 0b00000001011),
        v(12, 0b000000001111), v(12, 0b000000001011), v(12, 0b000000001000),
        v(13, 0b0000000001111), v(13, 0b0000000001011), v(13, 0b0000000000111),
        v(14, 0b00000000001001), v(14, 0b00000000000111),
    ],
    [
        NA, v(2, 0b10), v(5, 0b00111), v(6, 0b001010), v(6, 0b000110), v(7, 0b0000110),
        v(8, 0b00000110), v(9, 0b000000110), v(11, 0b00000001110), v(11, 0b00000001010),
        v(12, 0b000000001110), v(12, 0b000000001010), v(13, 0b0000000001110),
        v(13, 0b0000000001010), v(14, 0b00000000001011), v(14, 0b00000000001000),
        v(14, 0b00000000000110),
    ],
    [
        NA, NA, v(3, 0b011), v(6, 0b001001), v(6, 0b000101), v(7, 0b0000101),
        v(8, 0b00000101), v(9, 0b000000101), v(11, 0b00000001101), v(11, 0b00000001001),
        v(12, 0b000000001101), v(12, 0b000000001001), v(13, 0b0000000001101),
        v(13, 0b0000000001001), v(13, 0b0000000000110), v(14, 0b00000000001010),
        v(14, 0b00000000000101),
    ],
    [
        NA, NA, NA, v(4, 0b0101), v(4, 0b0100), v(5, 0b00110), v(6, 0b001000),
        v(6, 0b000100), v(7, 0b0000100), v(9, 0b000000100), v(11, 0b00000001100),
        v(11, 0b00000001000), v(12, 0b000000001100), v(13, 0b0000000001100),
        v(13, 0b0000000001000), v(13, 0b0000000000001), v(14, 0b00000000000100),
    ],
];

/// `4 <= nC < 8`.
const CT2: CoeffTokenTable = [
    [
        v(4, 0b1111), v(6, 0b001111), v(6, 0b001011), v(6, 0b001000), v(7, 0b0001111),
        v(7, 0b0001011), v(7, 0b0001001), v(7, 0b0001000), v(8, 0b00001111),
        v(8, 0b00001011), v(9, 0b000001111), v(9, 0b000001011), v(9, 0b000001000),
        v(10, 0b0000001101), v(10, 0b0000001001), v(10, 0b0000000101),
        v(10, 0b0000000001),
    ],
    [
        NA, v(4, 0b1110), v(5, 0b01111), v(5, 0b01100), v(5, 0b01010), v(5, 0b01000),
        v(6, 0b001110), v(6, 0b001010), v(7, 0b0001110), v(8, 0b00001110),
        v(8, 0b00001010), v(9, 0b000001110), v(9, 0b000001010), v(9, 0b000000111),
        v(10, 0b0000001100), v(10, 0b0000001000), v(10, 0b0000000100),
    ],
    [
        NA, NA, v(4, 0b1101), v(5, 0b01110), v(5, 0b01011), v(5, 0b01001),
        v(6, 0b001101), v(6, 0b001001), v(7, 0b0001101), v(7, 0b0001010),
        v(8, 0b00001101), v(8, 0b00001001), v(9, 0b000001101), v(9, 0b000001001),
        v(10, 0b0000001011), v(10, 0b0000000111), v(10, 0b0000000011),
    ],
    [
        NA, NA, NA, v(4, 0b1100), v(4, 0b1011), v(4, 0b1010), v(4, 0b1001),
        v(4, 0b1000), v(5, 0b01101), v(6, 0b001100), v(7, 0b0001100), v(8, 0b00001100),
        v(8, 0b00001000), v(9, 0b000001100), v(10, 0b0000001010), v(10, 0b0000000110),
        v(10, 0b0000000010),
    ],
];

/// `nC == -1` — the 2×2 chroma DC block, which has at most four coefficients.
const CT_CHROMA_DC: [[Vlc; 5]; 4] = [
    [v(2, 0b01), v(6, 0b000111), v(6, 0b000100), v(6, 0b000011), v(6, 0b000010)],
    [NA, v(1, 0b1), v(6, 0b000110), v(7, 0b0000011), v(8, 0b00000011)],
    [NA, NA, v(3, 0b001), v(7, 0b0000010), v(8, 0b00000010)],
    [NA, NA, NA, v(6, 0b000101), v(7, 0b0000000)],
];

/// `total_zeros` for 4×4 blocks, indexed `[tz_vlc_index - 1][total_zeros]`
/// (Tables 9-7 and 9-8).
const TOTAL_ZEROS_4X4: [&[Vlc]; 15] = [
    &[
        v(1, 0b1), v(3, 0b011), v(3, 0b010), v(4, 0b0011), v(4, 0b0010), v(5, 0b00011),
        v(5, 0b00010), v(6, 0b000011), v(6, 0b000010), v(7, 0b0000011), v(7, 0b0000010),
        v(8, 0b00000011), v(8, 0b00000010), v(9, 0b000000011), v(9, 0b000000010),
        v(9, 0b000000001),
    ],
    &[
        v(3, 0b111), v(3, 0b110), v(3, 0b101), v(3, 0b100), v(3, 0b011), v(4, 0b0101),
        v(4, 0b0100), v(4, 0b0011), v(4, 0b0010), v(5, 0b00011), v(5, 0b00010),
        v(6, 0b000011), v(6, 0b000010), v(6, 0b000001), v(6, 0b000000),
    ],
    &[
        v(4, 0b0101), v(3, 0b111), v(3, 0b110), v(3, 0b101), v(4, 0b0100), v(4, 0b0011),
        v(3, 0b100), v(3, 0b011), v(4, 0b0010), v(5, 0b00011), v(5, 0b00010),
        v(6, 0b000001), v(5, 0b00001), v(6, 0b000000),
    ],
    &[
        v(5, 0b00011), v(3, 0b111), v(4, 0b0101), v(4, 0b0100), v(3, 0b110), v(3, 0b101),
        v(3, 0b100), v(4, 0b0011), v(3, 0b011), v(4, 0b0010), v(5, 0b00010),
        v(5, 0b00001), v(5, 0b00000),
    ],
    &[
        v(4, 0b0101), v(4, 0b0100), v(4, 0b0011), v(3, 0b111), v(3, 0b110), v(3, 0b101),
        v(3, 0b100), v(3, 0b011), v(4, 0b0010), v(5, 0b00001), v(4, 0b0001),
        v(5, 0b00000),
    ],
    &[
        v(6, 0b000001), v(5, 0b00001), v(3, 0b111), v(3, 0b110), v(3, 0b101), v(3, 0b100),
        v(3, 0b011), v(3, 0b010), v(4, 0b0001), v(3, 0b001), v(6, 0b000000),
    ],
    &[
        v(6, 0b000001), v(5, 0b00001), v(3, 0b101), v(3, 0b100), v(3, 0b011), v(2, 0b11),
        v(3, 0b010), v(4, 0b0001), v(3, 0b001), v(6, 0b000000),
    ],
    &[
        v(6, 0b000001), v(4, 0b0001), v(5, 0b00001), v(3, 0b011), v(2, 0b11), v(2, 0b10),
        v(3, 0b010), v(3, 0b001), v(6, 0b000000),
    ],
    &[
        v(6, 0b000001), v(6, 0b000000), v(4, 0b0001), v(2, 0b11), v(2, 0b10), v(3, 0b001),
        v(2, 0b01), v(5, 0b00001),
    ],
    &[
        v(5, 0b00001), v(5, 0b00000), v(3, 0b001), v(2, 0b11), v(2, 0b10), v(2, 0b01),
        v(4, 0b0001),
    ],
    &[v(4, 0b0000), v(4, 0b0001), v(3, 0b001), v(3, 0b010), v(1, 0b1), v(3, 0b011)],
    &[v(4, 0b0000), v(4, 0b0001), v(2, 0b01), v(1, 0b1), v(3, 0b001)],
    &[v(3, 0b000), v(3, 0b001), v(1, 0b1), v(2, 0b01)],
    &[v(2, 0b00), v(2, 0b01), v(1, 0b1)],
    &[v(1, 0b0), v(1, 0b1)],
];

/// `total_zeros` for the 2×2 chroma DC block (Table 9-9a).
const TOTAL_ZEROS_CHROMA_DC: [&[Vlc]; 3] = [
    &[v(1, 0b1), v(2, 0b01), v(3, 0b001), v(3, 0b000)],
    &[v(1, 0b1), v(2, 0b01), v(2, 0b00)],
    &[v(1, 0b1), v(1, 0b0)],
];

/// `run_before`, indexed `[min(zeros_left, 7) - 1][run_before]` (Table 9-10).
/// The last row covers every `zeros_left > 6`.
const RUN_BEFORE: [&[Vlc]; 7] = [
    &[v(1, 0b1), v(1, 0b0)],
    &[v(1, 0b1), v(2, 0b01), v(2, 0b00)],
    &[v(2, 0b11), v(2, 0b10), v(2, 0b01), v(2, 0b00)],
    &[v(2, 0b11), v(2, 0b10), v(2, 0b01), v(3, 0b001), v(3, 0b000)],
    &[v(2, 0b11), v(2, 0b10), v(3, 0b011), v(3, 0b010), v(3, 0b001), v(3, 0b000)],
    &[
        v(2, 0b11), v(3, 0b000), v(3, 0b001), v(3, 0b011), v(3, 0b010), v(3, 0b101),
        v(3, 0b100),
    ],
    &[
        v(3, 0b111), v(3, 0b110), v(3, 0b101), v(3, 0b100), v(3, 0b011), v(3, 0b010),
        v(3, 0b001), v(4, 0b0001), v(5, 0b00001), v(6, 0b000001), v(7, 0b0000001),
        v(8, 0b00000001), v(9, 0b000000001), v(10, 0b0000000001), v(11, 0b00000000001),
    ],
];

/// Which block flavour is being coded — it selects the coefficient count and,
/// for chroma DC, an entirely different set of tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockKind {
    /// A full 4×4 residual block (`LumaLevel4x4`).
    Luma4x4,
    /// The 2×2 chroma DC block (`ChromaDCLevel`), `nC` fixed at −1.
    ChromaDc,
    /// The 15 AC coefficients of a chroma 4×4 block (`ChromaACLevel`).
    ChromaAc,
    /// The sixteen Hadamard-transformed DCs of an `I_16x16` macroblock
    /// (`Intra16x16DCLevel`). Same tables as a luma 4×4 block; `nC` comes from
    /// the neighbours of luma block 0.
    Luma16x16Dc,
    /// The 15 AC coefficients of one 4×4 block of an `I_16x16` macroblock
    /// (`Intra16x16ACLevel`).
    Luma16x16Ac,
}

impl BlockKind {
    fn max_coeff(self) -> usize {
        match self {
            BlockKind::Luma4x4 => 16,
            BlockKind::ChromaDc => 4,
            BlockKind::ChromaAc => 15,
            BlockKind::Luma16x16Dc => 16,
            BlockKind::Luma16x16Ac => 15,
        }
    }
}

/// Pick the `coeff_token` code for `(trailing_ones, total_coeff)` at context `nc`.
fn coeff_token(kind: BlockKind, nc: i32, t1: usize, total: usize) -> Vlc {
    if kind == BlockKind::ChromaDc {
        return CT_CHROMA_DC[t1][total];
    }
    if nc < 2 {
        CT0[t1][total]
    } else if nc < 4 {
        CT1[t1][total]
    } else if nc < 8 {
        CT2[t1][total]
    } else {
        // nC >= 8 is a 6-bit fixed-length code (Table 9-5, last column).
        if total == 0 {
            v(6, 0b000011)
        } else {
            v(6, (((total - 1) << 2) | t1) as u32)
        }
    }
}

/// Write `level_code` with the prefix/suffix scheme of §9.2.2.1.
fn write_level(w: &mut BitWriter, level_code: u32, suffix_length: u32) {
    let unary = |w: &mut BitWriter, prefix: u32| {
        for _ in 0..prefix {
            w.write_bit(0);
        }
        w.write_bit(1);
    };

    if suffix_length == 0 {
        if level_code < 14 {
            unary(w, level_code);
            return;
        }
        if level_code < 30 {
            unary(w, 14);
            w.write_bits(level_code - 14, 4);
            return;
        }
    } else {
        let prefix = level_code >> suffix_length;
        if prefix < 15 {
            unary(w, prefix);
            w.write_bits(level_code & ((1 << suffix_length) - 1), suffix_length);
            return;
        }
    }

    // Escape. `level_prefix >= 15` carries a 12-bit suffix, and each prefix past
    // that widens the suffix by one bit; the ranges tile without gaps, so the
    // first that contains `level_code` is the shortest encoding of it.
    let base = (15u32 << suffix_length) + if suffix_length == 0 { 15 } else { 0 };
    let mut prefix = 15u32;
    loop {
        let size = prefix - 3;
        let extra = if prefix >= 16 {
            (1u32 << size) - 4096
        } else {
            0
        };
        let lo = base + extra;
        let hi = lo + (1u32 << size) - 1;
        if level_code >= lo && level_code <= hi {
            unary(w, prefix);
            w.write_bits(level_code - lo, size);
            return;
        }
        prefix += 1;
        debug_assert!(prefix < 28, "level {level_code} does not fit any prefix");
    }
}

/// Code one residual block (§9.2). `coeff` is in **scan order** with `.len()`
/// equal to the block's coefficient count; `nc` is the neighbour-derived context
/// (ignored for chroma DC).
///
/// Returns the number of non-zero coefficients, which is what neighbouring
/// blocks need for their own `nC`.
pub fn encode_block(w: &mut BitWriter, coeff: &[i32], kind: BlockKind, nc: i32) -> usize {
    debug_assert_eq!(coeff.len(), kind.max_coeff());

    // Walk backwards: highest frequency first.
    //
    // The zeros *above* the highest-frequency coefficient are not coded at all —
    // `coeff_token` already implies where the block ends. Only the gaps between
    // coefficients are runs, and `total_zeros` counts the zeros before the
    // highest-frequency one. Counting the leading run makes a DC-only block
    // claim 15 zeros instead of 0.
    let mut levels: Vec<i32> = Vec::new();
    let mut runs: Vec<usize> = Vec::new();
    let mut run = 0usize;
    let mut started = false;
    for &c in coeff.iter().rev() {
        if c == 0 {
            if started {
                run += 1;
            }
        } else {
            if started {
                runs.push(run);
            }
            levels.push(c);
            started = true;
            run = 0;
        }
    }
    // Whatever is left is the run before the lowest-frequency coefficient, which
    // the decoder infers from `zerosLeft` rather than reading.
    let trailing_run = run;
    let total = levels.len();

    // Trailing ones: up to three ±1s at the high-frequency end.
    let mut t1 = 0usize;
    while t1 < total.min(3) && levels[t1].abs() == 1 {
        t1 += 1;
    }

    let tok = coeff_token(kind, nc, t1, total);
    w.write_bits(tok.code, tok.len as u32);
    if total == 0 {
        return 0;
    }

    // Signs of the trailing ones, same order.
    for &l in levels.iter().take(t1) {
        w.write_bit(u32::from(l < 0));
    }

    // Remaining levels.
    let mut suffix_length = u32::from(total > 10 && t1 < 3);
    for (i, &l) in levels.iter().enumerate().skip(t1) {
        let mut code = 2 * (l.unsigned_abs() - 1) + u32::from(l < 0);
        // The first coded level cannot be a ±1 unless all three trailing-one
        // slots were used, so the alphabet starts two later.
        if i == t1 && t1 < 3 {
            code -= 2;
        }
        write_level(w, code, suffix_length);
        if suffix_length == 0 {
            suffix_length = 1;
        }
        if l.unsigned_abs() > (3 << (suffix_length - 1)) && suffix_length < 6 {
            suffix_length += 1;
        }
    }

    // total_zeros, then the run before each coefficient.
    let total_zeros: usize = runs.iter().sum::<usize>() + trailing_run;
    if total < kind.max_coeff() {
        let tz = if kind == BlockKind::ChromaDc {
            TOTAL_ZEROS_CHROMA_DC[total - 1][total_zeros]
        } else {
            TOTAL_ZEROS_4X4[total - 1][total_zeros]
        };
        w.write_bits(tz.code, tz.len as u32);
    }

    let mut zeros_left = total_zeros;
    for &r in runs.iter() {
        if zeros_left == 0 {
            break;
        }
        let row = zeros_left.min(7) - 1;
        let rb = RUN_BEFORE[row][r];
        w.write_bits(rb.code, rb.len as u32);
        zeros_left -= r;
    }

    total
}

/// `nC` for a luma or chroma AC block from its neighbours' coefficient counts
/// (§9.2.1). `None` means that neighbour is outside the picture or slice.
pub fn derive_nc(left: Option<usize>, above: Option<usize>) -> i32 {
    match (left, above) {
        (Some(a), Some(b)) => ((a + b + 1) >> 1) as i32,
        (Some(a), None) => a as i32,
        (None, Some(b)) => b as i32,
        (None, None) => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A code table has to be uniquely decodable: no code may be a prefix of
    /// another. A mistyped entry almost always breaks this, which is why it is
    /// worth checking independently of the values themselves.
    fn assert_prefix_free(name: &str, codes: &[Vlc]) {
        for (i, a) in codes.iter().enumerate() {
            assert!(a.len > 0 && a.len <= 32, "{name}: bad length {}", a.len);
            assert!(
                a.code < (1u32 << a.len),
                "{name}: code {:b} wider than its {} bits",
                a.code,
                a.len
            );
            for (j, b) in codes.iter().enumerate() {
                if i == j {
                    continue;
                }
                let (short, long) = if a.len <= b.len { (a, b) } else { (b, a) };
                let shifted = long.code >> (long.len - short.len);
                assert_ne!(
                    shifted, short.code,
                    "{name}: entry {i} ({:0w1$b}) prefixes entry {j} ({:0w2$b})",
                    a.code,
                    b.code,
                    w1 = a.len as usize,
                    w2 = b.len as usize
                );
            }
        }
    }

    /// Kraft's inequality: `Σ 2^-len <= 1` for any prefix code. Exceeding 1 means
    /// two codewords must overlap.
    ///
    /// Equality would mean *complete*, and H.264's tables are not — several
    /// leave a codeword unused (`total_zeros` at `tzVlcIndex = 1` is short by
    /// exactly one 9-bit word, `CT0` by one 15-bit word). So the lower bound is
    /// a sanity floor rather than an exact expectation: it still catches a
    /// systematically wrong set of lengths, which is what a transcription slip
    /// looks like.
    fn assert_kraft_sane(name: &str, codes: &[Vlc]) {
        let sum: f64 = codes.iter().map(|c| 2f64.powi(-(c.len as i32))).sum();
        assert!(
            sum <= 1.0 + 1e-9,
            "{name}: Kraft sum {sum} exceeds 1 — codewords must overlap"
        );
        assert!(
            sum > 0.9,
            "{name}: Kraft sum {sum} far below 1 — codewords are missing or too long"
        );
    }

    fn live(table: &CoeffTokenTable) -> Vec<Vlc> {
        let mut out = Vec::new();
        for (t1, row) in table.iter().enumerate() {
            for (total, e) in row.iter().enumerate() {
                if total >= t1 && e.len > 0 {
                    out.push(*e);
                }
            }
        }
        out
    }

    #[test]
    fn coeff_token_tables_are_prefix_free_and_complete() {
        for (name, t) in [("CT0", &CT0), ("CT1", &CT1), ("CT2", &CT2)] {
            let codes = live(t);
            assert_eq!(codes.len(), 62, "{name}: expected 62 live entries");
            assert_prefix_free(name, &codes);
            assert_kraft_sane(name, &codes);
        }
        let cdc: Vec<Vlc> = CT_CHROMA_DC
            .iter()
            .enumerate()
            .flat_map(|(t1, row)| {
                row.iter()
                    .enumerate()
                    .filter(move |(total, e)| *total >= t1 && e.len > 0)
                    .map(|(_, e)| *e)
            })
            .collect();
        assert_eq!(cdc.len(), 14, "chroma DC: expected 14 live entries");
        assert_prefix_free("CT_CHROMA_DC", &cdc);
        assert_kraft_sane("CT_CHROMA_DC", &cdc);
    }

    #[test]
    fn total_zeros_tables_are_prefix_free_and_complete() {
        for (i, t) in TOTAL_ZEROS_4X4.iter().enumerate() {
            let name = format!("total_zeros[tzVlcIndex={}]", i + 1);
            // tzVlcIndex n covers total_zeros 0..=(16-n).
            assert_eq!(t.len(), 16 - i, "{name}: wrong entry count");
            assert_prefix_free(&name, t);
            assert_kraft_sane(&name, t);
        }
        for (i, t) in TOTAL_ZEROS_CHROMA_DC.iter().enumerate() {
            let name = format!("total_zeros_chroma_dc[{}]", i + 1);
            assert_eq!(t.len(), 4 - i, "{name}: wrong entry count");
            assert_prefix_free(&name, t);
            assert_kraft_sane(&name, t);
        }
    }

    #[test]
    fn run_before_tables_are_prefix_free() {
        for (i, t) in RUN_BEFORE.iter().enumerate() {
            let name = format!("run_before[zerosLeft={}]", i + 1);
            assert_prefix_free(&name, t);
            if i < 6 {
                assert_eq!(t.len(), i + 2, "{name}: wrong entry count");
                assert_kraft_sane(&name, t);
            }
        }
    }

    #[test]
    fn zigzag_is_a_permutation() {
        let mut seen = [false; 16];
        for &i in &ZIGZAG4 {
            assert!(!seen[i], "zigzag repeats index {i}");
            seen[i] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn an_empty_block_is_just_its_token() {
        let mut w = BitWriter::new();
        let n = encode_block(&mut w, &[0; 16], BlockKind::Luma4x4, 0);
        assert_eq!(n, 0);
        // nC < 2, (0,0) → the single bit `1`.
        assert_eq!(w.bit_len(), 1);
    }

    #[test]
    fn nc_derivation_follows_availability() {
        assert_eq!(derive_nc(Some(3), Some(4)), 4); // (3+4+1)>>1
        assert_eq!(derive_nc(Some(5), None), 5);
        assert_eq!(derive_nc(None, Some(2)), 2);
        assert_eq!(derive_nc(None, None), 0);
    }

    /// Levels beyond the 12-bit escape still encode — the prefix widens instead
    /// of the value being silently clamped.
    #[test]
    fn very_large_levels_use_the_widening_escape() {
        for code in [30u32, 4125, 4126, 12000, 20000, 60000] {
            let mut w = BitWriter::new();
            write_level(&mut w, code, 0);
            assert!(w.bit_len() > 0 && w.bit_len() < 64, "code {code}");
        }
    }
}
