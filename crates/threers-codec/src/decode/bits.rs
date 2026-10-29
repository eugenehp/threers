//! Reading side of the bitstream layer: an MSB-first bit reader with
//! Exp-Golomb, and NAL unit splitting and unescaping.
//!
//! The mirror of [`crate::bitstream`]. Reads past the end return zeros
//! rather than failing, because a truncated slice is something a decoder has to
//! survive, not something it can always detect before it has read too far —
//! callers check [`BitReader::overrun`] at the points where it matters.

/// MSB-first reader over an RBSP (emulation prevention already removed).
#[derive(Clone)]
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Bit position from the start of `data`.
    pos: usize,
}

impl<'a> BitReader<'a> {
    /// A reader at the first bit of `data`.
    pub fn new(data: &'a [u8]) -> Self {
        BitReader { data, pos: 0 }
    }

    /// The whole buffer being read.
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// Bit position from the start of the buffer.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Move to bit `pos` from the start of the buffer.
    pub fn set_pos(&mut self, pos: usize) {
        self.pos = pos;
    }

    /// The buffer's length in bits.
    pub fn len_bits(&self) -> usize {
        self.data.len() * 8
    }

    /// Bits left before the end of the buffer (negative once overrun).
    pub fn bits_left(&self) -> isize {
        self.len_bits() as isize - self.pos as isize
    }

    /// Whether reads have run past the end of the buffer.
    pub fn overrun(&self) -> bool {
        self.pos > self.len_bits()
    }

    /// Whether the next read starts on a byte boundary.
    pub fn is_byte_aligned(&self) -> bool {
        self.pos & 7 == 0
    }

    /// Skip to the next byte boundary (no-op when already on one).
    pub fn byte_align(&mut self) {
        self.pos = (self.pos + 7) & !7;
    }

    /// Byte offset of the reader, rounded up to the next byte boundary.
    pub fn byte_pos(&self) -> usize {
        (self.pos + 7) >> 3
    }

    #[inline(always)]
    fn load64(&self, byte: usize) -> u64 {
        if byte + 8 <= self.data.len() {
            u64::from_be_bytes(self.data[byte..byte + 8].try_into().unwrap())
        } else {
            let mut b = [0u8; 8];
            for (i, slot) in b.iter_mut().enumerate() {
                if let Some(&v) = self.data.get(byte + i) {
                    *slot = v;
                }
            }
            u64::from_be_bytes(b)
        }
    }

    /// The next `n` bits (`n <= 32`) without consuming them.
    #[inline(always)]
    pub fn peek(&self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        let v = self.load64(self.pos >> 3) << (self.pos & 7);
        (v >> (64 - n)) as u32
    }

    /// Consume `n` bits.
    #[inline(always)]
    pub fn skip(&mut self, n: u32) {
        self.pos += n as usize;
    }

    /// `u(n)`, `n <= 32`.
    #[inline(always)]
    pub fn u(&mut self, n: u32) -> u32 {
        let v = self.peek(n);
        self.pos += n as usize;
        v
    }

    /// `u(1)` as a `bool`.
    #[inline(always)]
    pub fn flag(&mut self) -> bool {
        self.u(1) != 0
    }

    /// `ue(v)`. Saturates on codes longer than 32 bits, which no conforming
    /// stream contains.
    pub fn ue(&mut self) -> u32 {
        let peek = self.peek(32);
        let lz = peek.leading_zeros();
        if lz < 16 {
            let len = 2 * lz + 1;
            let v = self.peek(len);
            self.pos += len as usize;
            return v - 1;
        }
        if lz >= 32 {
            // 32+ zeros: either garbage or the end of the buffer.
            self.pos += 32;
            return u32::MAX;
        }
        self.pos += lz as usize;
        let v = self.u(lz + 1) as u64;
        (v - 1) as u32
    }

    /// `se(v)`.
    pub fn se(&mut self) -> i32 {
        let k = self.ue() as i64;
        if k & 1 == 1 {
            ((k + 1) / 2) as i32
        } else {
            (-(k / 2)) as i32
        }
    }

    /// `more_rbsp_data()`: whether anything other than the stop bit and its
    /// alignment zeros is left.
    pub fn more_rbsp_data(&self) -> bool {
        let Some(last) = self.data.iter().rposition(|&b| b != 0) else {
            return false;
        };
        let stop = last * 8 + 7 - self.data[last].trailing_zeros() as usize;
        self.pos < stop
    }
}

/// Remove emulation prevention bytes: every `00 00 03` becomes `00 00`.
pub fn unescape(nal: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(nal.len());
    let mut zeros = 0usize;
    let mut i = 0;
    while i < nal.len() {
        let b = nal[i];
        if zeros >= 2 && b == 3 {
            zeros = 0;
            i += 1;
            continue;
        }
        if b == 0 {
            zeros += 1;
        } else {
            zeros = 0;
        }
        out.push(b);
        i += 1;
    }
    out
}

/// Split a length-prefixed sample (`avcC`/`hvcC` style) into NAL units.
pub fn split_length_prefixed(sample: &[u8], len_size: usize) -> Result<Vec<&[u8]>, String> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while at < sample.len() {
        if at + len_size > sample.len() {
            return Err("sample ended inside a NAL length".into());
        }
        let mut n = 0usize;
        for &b in &sample[at..at + len_size] {
            n = (n << 8) | b as usize;
        }
        at += len_size;
        if at + n > sample.len() {
            return Err(format!("NAL of {n} bytes runs past the sample"));
        }
        if n > 0 {
            out.push(&sample[at..at + n]);
        }
        at += n;
    }
    Ok(out)
}

/// Split an Annex B byte stream on its start codes.
pub fn split_annexb(stream: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= stream.len() {
        if stream[i] == 0 && stream[i + 1] == 0 && stream[i + 2] == 1 {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut out = Vec::with_capacity(starts.len());
    for (k, &s) in starts.iter().enumerate() {
        let mut e = if k + 1 < starts.len() {
            starts[k + 1] - 3
        } else {
            stream.len()
        };
        // Trailing zeros belong to the next start code (`00 00 00 01`) or
        // are padding.
        while e > s && stream[e - 1] == 0 {
            e -= 1;
        }
        if e > s {
            out.push(&stream[s..e]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::BitWriter;

    #[test]
    fn exp_golomb_round_trips() {
        let mut w = BitWriter::new();
        let vals = [0u32, 1, 2, 3, 7, 8, 254, 255, 65534, 65535, 1 << 20, 0xFFFF_FFFE];
        for &v in &vals {
            w.write_ue(v);
        }
        let svals = [0i32, 1, -1, 2, -2, 1000, -1000, 1 << 20, -(1 << 20)];
        for &v in &svals {
            w.write_se(v);
        }
        w.write_bits(0b101, 3);
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        for &v in &vals {
            assert_eq!(r.ue(), v);
        }
        for &v in &svals {
            assert_eq!(r.se(), v);
        }
        assert_eq!(r.u(3), 0b101);
    }

    #[test]
    fn unescape_drops_only_emulation_bytes() {
        assert_eq!(unescape(&[0, 0, 3, 1, 0, 0, 3, 0, 0, 3]), vec![0, 0, 1, 0, 0, 0, 0]);
        assert_eq!(unescape(&[0, 3, 0, 0, 3, 3]), vec![0, 3, 0, 0, 3]);
    }

    #[test]
    fn more_rbsp_data_stops_at_the_stop_bit() {
        // 1 data bit, then the stop bit and alignment.
        let d = [0b1100_0000u8];
        let mut r = BitReader::new(&d);
        assert!(r.more_rbsp_data());
        r.skip(1);
        assert!(!r.more_rbsp_data());
    }

    #[test]
    fn annexb_splits_three_and_four_byte_start_codes() {
        let s = [0, 0, 0, 1, 0x67, 1, 2, 0, 0, 1, 0x68, 3, 0, 0, 0, 1, 0x65];
        let n = split_annexb(&s);
        assert_eq!(n, vec![&[0x67u8, 1, 2][..], &[0x68, 3][..], &[0x65][..]]);
    }
}
