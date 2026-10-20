//! The CABAC arithmetic decoding engine (H.264 §9.3.3.2, H.265 §9.3.4.3).
//!
//! The two standards specify the same engine — the same `rangeTabLPS`, state
//! transitions and renormalisation — and differ only in how a context's state
//! is initialised, which is what [`Ctx::h264`] and [`Ctx::hevc`] are for.
//!
//! This follows the specification's 9-bit `codIOffset` formulation literally,
//! reading renormalisation bits straight from a [`BitReader`]. That keeps the
//! reader's position equal to the number of bits the specification says have
//! been consumed, which is what PCM samples, WPP substreams and tile entry
//! points need: after a terminating bin the next byte-aligned position is
//! exactly where the following data starts.

use super::bits::BitReader;
use crate::codec::hevc::tables::{RANGE_TAB_LPS, TRANS_IDX_LPS, TRANS_IDX_MPS};

/// One context variable: probability state and most probable symbol.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Ctx {
    pub state: u8,
    pub mps: u8,
}

impl Ctx {
    fn from_pre(pre: i32) -> Ctx {
        if pre <= 63 {
            Ctx {
                state: (63 - pre) as u8,
                mps: 0,
            }
        } else {
            Ctx {
                state: (pre - 64) as u8,
                mps: 1,
            }
        }
    }

    /// H.264 §9.3.1.1: from the `(m, n)` pair and `SliceQPY`.
    pub fn h264(m: i8, n: i8, qp: i32) -> Ctx {
        let pre = (((m as i32) * qp.clamp(0, 51)) >> 4) + n as i32;
        Ctx::from_pre(pre.clamp(1, 126))
    }

    /// H.265 §9.3.2.2: from the 8-bit `initValue` and `SliceQpY`.
    pub fn hevc(init: u8, qp: i32) -> Ctx {
        let slope = (init >> 4) as i32 * 5 - 45;
        let offset = (((init & 15) as i32) << 3) - 16;
        let pre = ((slope * qp.clamp(0, 51)) >> 4) + offset;
        Ctx::from_pre(pre.clamp(1, 126))
    }
}

/// The arithmetic decoder.
pub struct Cabac<'a> {
    r: BitReader<'a>,
    range: u32,
    offset: u32,
}

impl<'a> Cabac<'a> {
    /// Start decoding at byte `start` of `data` (§9.3.1.2 / §9.3.2.5).
    pub fn new(data: &'a [u8], start: usize) -> Self {
        let mut r = BitReader::new(data);
        r.set_pos(start * 8);
        let offset = r.u(9);
        Cabac {
            r,
            range: 510,
            offset,
        }
    }

    /// Re-initialise at byte `start` — after PCM samples, or at the start of
    /// a WPP row or tile.
    pub fn restart(&mut self, start: usize) {
        self.r.set_pos(start * 8);
        self.range = 510;
        self.offset = self.r.u(9);
    }

    /// Whether the decoder has read past the end of its data — a truncated
    /// or corrupt slice.
    pub fn overrun(&self) -> bool {
        self.r.bits_left() < -64
    }

    /// The byte at which data following a terminating bin begins: the bits
    /// consumed so far rounded up to a byte.
    pub fn aligned_byte_pos(&self) -> usize {
        self.r.byte_pos()
    }

    pub fn data(&self) -> &'a [u8] {
        self.r.data()
    }

    #[inline(always)]
    pub fn decision(&mut self, ctx: &mut Ctx) -> u32 {
        let q = ((self.range >> 6) & 3) as usize;
        let lps = RANGE_TAB_LPS[ctx.state as usize][q] as u32;
        self.range -= lps;
        let bin;
        if self.offset >= self.range {
            bin = 1 - ctx.mps as u32;
            self.offset -= self.range;
            self.range = lps;
            if ctx.state == 0 {
                ctx.mps = 1 - ctx.mps;
            }
            ctx.state = TRANS_IDX_LPS[ctx.state as usize];
        } else {
            bin = ctx.mps as u32;
            ctx.state = TRANS_IDX_MPS[ctx.state as usize];
        }
        if self.range < 256 {
            let n = self.range.leading_zeros() - 23;
            self.range <<= n;
            self.offset = (self.offset << n) | self.r.u(n);
        }
        bin
    }

    #[inline(always)]
    pub fn bypass(&mut self) -> u32 {
        self.offset = (self.offset << 1) | self.r.u(1);
        if self.offset >= self.range {
            self.offset -= self.range;
            1
        } else {
            0
        }
    }

    /// `n` bypass bins, first one most significant.
    #[inline]
    pub fn bypass_bits(&mut self, n: u32) -> u32 {
        let mut v = 0;
        for _ in 0..n {
            v = (v << 1) | self.bypass();
        }
        v
    }

    /// A terminating bin (`end_of_slice_flag`, `pcm_flag`, ...).
    pub fn terminate(&mut self) -> u32 {
        self.range -= 2;
        if self.offset >= self.range {
            1
        } else {
            if self.range < 256 {
                let n = self.range.leading_zeros() - 23;
                self.range <<= n;
                self.offset = (self.offset << n) | self.r.u(n);
            }
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::hevc::cabac::{CabacEncoder, CtxModel};

    /// Decode what the crate's own encoder wrote, decisions, bypass and a
    /// terminate, and land exactly on the byte that follows.
    #[test]
    fn decodes_the_encoders_output() {
        let mut enc = CabacEncoder::new();
        let mut ectx = [CtxModel::from_state(0, 0); 4];
        let mut seq = Vec::new();
        let mut x = 12345u32;
        for i in 0..5000 {
            x = x.wrapping_mul(1103515245).wrapping_add(12345);
            let c = (i % 4) as usize;
            let bias = [1, 3, 7, 15][c];
            let bit = ((x >> 16) & bias == 0) as u32;
            seq.push((c, bit, i % 7 == 0));
            if i % 7 == 0 {
                enc.encode_bypass(bit);
            } else {
                enc.encode_bin(&mut ectx[c], bit);
            }
        }
        enc.encode_terminate(1);
        let mut bytes = enc.finish();
        let tail = bytes.len();
        bytes.extend_from_slice(&[0xAB, 0xCD]);

        let mut dec = Cabac::new(&bytes, 0);
        let mut dctx = [Ctx { state: 0, mps: 0 }; 4];
        for &(c, bit, by) in &seq {
            let got = if by {
                dec.bypass()
            } else {
                dec.decision(&mut dctx[c])
            };
            assert_eq!(got, bit);
        }
        assert_eq!(dec.terminate(), 1);
        assert_eq!(dec.aligned_byte_pos(), tail);
    }
}
