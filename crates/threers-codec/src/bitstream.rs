//! Bit-level output primitives shared by the native codecs.
//!
//! [`BitWriter`] packs bits MSB-first into a byte buffer and knows the two
//! variable-length codes the HEVC high-level syntax needs: unsigned/signed
//! Exp-Golomb (`ue(v)` / `se(v)`). [`rbsp_trailing_bits`] and
//! [`emulation_prevention`] turn a raw bit payload into the byte-stuffed form a
//! NAL unit carries.
//!
//! Pure `core`/`alloc` + `std::Vec` — no OS, threads, or time — so it builds and
//! runs identically on native and `wasm32`.

/// Packs bits MSB-first into a growable byte buffer.
///
/// The first bit written lands in bit 7 (the most-significant bit) of byte 0,
/// matching the bit order every video/image bitstream uses. Partial trailing
/// bits stay in an internal accumulator until a byte fills or the writer is
/// byte-aligned.
#[derive(Clone, Debug, Default)]
pub struct BitWriter {
    bytes: Vec<u8>,
    /// Accumulator for the current, not-yet-complete byte (bits left-aligned).
    cur: u8,
    /// Number of valid bits currently held in `cur` (0..=7).
    nbits: u8,
}

impl BitWriter {
    /// A new, empty writer.
    pub fn new() -> Self {
        Self::default()
    }

    /// A writer preallocated for roughly `bytes` bytes of output.
    pub fn with_capacity(bytes: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(bytes),
            cur: 0,
            nbits: 0,
        }
    }

    /// Append a single bit (only the low bit of `bit` is used).
    #[inline]
    pub fn write_bit(&mut self, bit: u32) {
        self.cur |= ((bit & 1) as u8) << (7 - self.nbits);
        self.nbits += 1;
        if self.nbits == 8 {
            self.bytes.push(self.cur);
            self.cur = 0;
            self.nbits = 0;
        }
    }

    /// Append the low `n` bits of `value`, MSB-first. `n` must be `<= 32`.
    #[inline]
    pub fn write_bits(&mut self, value: u32, n: u32) {
        debug_assert!(n <= 32);
        for i in (0..n).rev() {
            self.write_bit((value >> i) & 1);
        }
    }

    /// Append the low `n` bits of a 64-bit `value`, MSB-first. `n` must be `<= 64`.
    pub fn write_bits64(&mut self, value: u64, n: u32) {
        debug_assert!(n <= 64);
        for i in (0..n).rev() {
            self.write_bit(((value >> i) & 1) as u32);
        }
    }

    /// Append a boolean as a `u(1)` flag.
    #[inline]
    pub fn flag(&mut self, b: bool) {
        self.write_bit(b as u32);
    }

    /// Append a whole byte. Fast when byte-aligned (the common case for raw PCM
    /// samples); falls back to bit-by-bit otherwise.
    #[inline]
    pub fn write_byte(&mut self, b: u8) {
        if self.nbits == 0 {
            self.bytes.push(b);
        } else {
            self.write_bits(b as u32, 8);
        }
    }

    /// Unsigned Exp-Golomb, `ue(v)` (H.265 §9.2). Encodes `value` as
    /// `leadingZeros` `0`s, a `1`, then the `leadingZeros`-bit suffix of
    /// `value + 1`.
    pub fn write_ue(&mut self, value: u32) {
        // codeNum + 1; its bit length is the total code length.
        let v1 = value as u64 + 1;
        let len = 64 - v1.leading_zeros(); // bits in v1
        for _ in 0..(len - 1) {
            self.write_bit(0);
        }
        self.write_bits64(v1, len);
    }

    /// Signed Exp-Golomb, `se(v)` (H.265 §9.2). Maps `value` to a `ue(v)`
    /// codeNum via the standard zig-zag (`0→0, 1→1, -1→2, 2→3, -2→4, …`).
    pub fn write_se(&mut self, value: i32) {
        let code = if value <= 0 {
            (-(value as i64) as u64) * 2
        } else {
            (value as u64) * 2 - 1
        };
        self.write_ue(code as u32);
    }

    /// Total number of bits written so far.
    pub fn bit_len(&self) -> usize {
        self.bytes.len() * 8 + self.nbits as usize
    }

    /// Whether the next bit would start a fresh byte.
    pub fn is_byte_aligned(&self) -> bool {
        self.nbits == 0
    }

    /// Pad to the next byte boundary with `0` bits (no-op if already aligned).
    pub fn align_zero(&mut self) {
        while self.nbits != 0 {
            self.write_bit(0);
        }
    }

    /// Consume the writer, padding any partial final byte with zeros, and return
    /// the bytes.
    pub fn finish(mut self) -> Vec<u8> {
        self.align_zero();
        self.bytes
    }

    /// Borrow the completed bytes; panics unless byte-aligned. Prefer
    /// [`finish`](Self::finish) unless you must keep writing.
    pub fn as_bytes(&self) -> &[u8] {
        debug_assert!(
            self.is_byte_aligned(),
            "as_bytes on a non-byte-aligned writer"
        );
        &self.bytes
    }
}

/// Append RBSP trailing bits: a single `1` stop bit followed by `0`s to the next
/// byte boundary (H.265 §7.3.2.11). Every RBSP ends with this.
pub fn rbsp_trailing_bits(w: &mut BitWriter) {
    w.write_bit(1);
    w.align_zero();
}

/// Insert emulation-prevention bytes, turning a raw byte-aligned RBSP into the
/// EBSP a NAL unit carries (H.265 §7.3.1.1).
///
/// Any run `00 00 00`, `00 00 01`, `00 00 02`, or `00 00 03` in the RBSP would
/// collide with a start code, so a `0x03` byte is spliced in after the two
/// zeros: `00 00 XX` → `00 00 03 XX` (for `XX <= 03`).
pub fn emulation_prevention(rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rbsp.len() + rbsp.len() / 128 + 4);
    let mut zeros = 0u32;
    for &b in rbsp {
        if zeros >= 2 && b <= 0x03 {
            out.push(0x03);
            zeros = 0;
        }
        out.push(b);
        if b == 0 {
            zeros += 1;
        } else {
            zeros = 0;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits_of(w: &BitWriter) -> String {
        let mut s = String::new();
        let full = w.bytes.len();
        for &byte in &w.bytes {
            for i in (0..8).rev() {
                s.push(if (byte >> i) & 1 == 1 { '1' } else { '0' });
            }
        }
        let _ = full;
        for i in (0..w.nbits).rev() {
            s.push(if (w.cur >> (7 - (w.nbits - 1 - i))) & 1 == 1 {
                '1'
            } else {
                '0'
            });
        }
        s
    }

    #[test]
    fn ue_known_vectors() {
        // H.265 Table: codeNum → bit string.
        let cases: &[(u32, &str)] = &[
            (0, "1"),
            (1, "010"),
            (2, "011"),
            (3, "00100"),
            (4, "00101"),
            (5, "00110"),
            (6, "00111"),
            (7, "0001000"),
            (8, "0001001"),
        ];
        for &(v, expect) in cases {
            let mut w = BitWriter::new();
            w.write_ue(v);
            assert_eq!(bits_of(&w), expect, "ue({v})");
        }
    }

    #[test]
    fn se_known_vectors() {
        let cases: &[(i32, &str)] = &[
            (0, "1"),
            (1, "010"),
            (-1, "011"),
            (2, "00100"),
            (-2, "00101"),
            (3, "00110"),
            (-3, "00111"),
        ];
        for &(v, expect) in cases {
            let mut w = BitWriter::new();
            w.write_se(v);
            assert_eq!(bits_of(&w), expect, "se({v})");
        }
    }

    #[test]
    fn write_bits_msb_first() {
        let mut w = BitWriter::new();
        w.write_bits(0b101, 3);
        w.write_bits(0b1, 1);
        w.write_bits(0b0000, 4);
        assert_eq!(w.finish(), vec![0b1011_0000]);
    }

    #[test]
    fn rbsp_trailing_aligns_with_stop_bit() {
        let mut w = BitWriter::new();
        w.write_bits(0b101, 3); // 3 bits used
        rbsp_trailing_bits(&mut w);
        // 101 + stop 1 + pad 0000 = 1011_0000
        assert_eq!(w.finish(), vec![0b1011_0000]);
    }

    #[test]
    fn emulation_prevention_inserts_03() {
        assert_eq!(
            emulation_prevention(&[0x00, 0x00, 0x00]),
            vec![0x00, 0x00, 0x03, 0x00]
        );
        assert_eq!(
            emulation_prevention(&[0x00, 0x00, 0x01]),
            vec![0x00, 0x00, 0x03, 0x01]
        );
        assert_eq!(
            emulation_prevention(&[0x00, 0x00, 0x02]),
            vec![0x00, 0x00, 0x03, 0x02]
        );
        assert_eq!(
            emulation_prevention(&[0x00, 0x00, 0x03]),
            vec![0x00, 0x00, 0x03, 0x03]
        );
        // No collision → untouched.
        assert_eq!(
            emulation_prevention(&[0x00, 0x00, 0x04]),
            vec![0x00, 0x00, 0x04]
        );
        assert_eq!(
            emulation_prevention(&[0x12, 0x34, 0x56]),
            vec![0x12, 0x34, 0x56]
        );
        // Long zero run: 00 00 00 00 → 00 00 03 00 00 (the extra 03 resets the count).
        assert_eq!(
            emulation_prevention(&[0x00, 0x00, 0x00, 0x00]),
            vec![0x00, 0x00, 0x03, 0x00, 0x00]
        );
    }
}
