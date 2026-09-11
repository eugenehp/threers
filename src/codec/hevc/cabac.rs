//! HEVC CABAC — Context-Adaptive Binary Arithmetic Coding (H.265 §9.3).
//!
//! HEVC has no CAVLC fallback: *every* slice-data syntax element is arithmetic
//! coded, so this engine is mandatory for any conformant stream. It is also the
//! single trickiest kernel to get right — an off-by-one in renormalization or a
//! wrong table entry corrupts the whole slice with no local symptom.
//!
//! This module implements the encoder ([`CabacEncoder`]) with the three bin
//! coders the spec defines — regular (context-modeled), bypass (equiprobable),
//! and terminate — plus the context model and its QP-dependent initialization
//! (§9.3.2.2). Correctness is pinned by a matching decoder in the tests: random
//! bin scripts are encoded then decoded and must come back identical.
//!
//! Pure arithmetic over `Vec<u8>`; identical on native and `wasm32`.

use super::tables::{RANGE_TAB_LPS, TRANS_IDX_LPS, TRANS_IDX_MPS};
use crate::codec::bitstream::BitWriter;

/// One CABAC context: an adaptive probability state (`0..=63`) and the current
/// most-probable-bin value (`0`/`1`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CtxModel {
    /// `pStateIdx` — probability-state index into the range/transition tables.
    pub state: u8,
    /// `valMps` — the value (0 or 1) currently treated as most probable.
    pub mps: u8,
}

impl CtxModel {
    /// Initialize from an 8-bit `init_value` and the slice QP (H.265 §9.3.2.2).
    ///
    /// The high/low nibbles of `init_value` give a slope/offset that map the QP
    /// to a starting probability. This is how every context is seeded at the
    /// start of a slice from the per-syntax-element init constants.
    pub fn init(init_value: u8, slice_qp: i32) -> Self {
        let slope_idx = (init_value >> 4) as i32;
        let offset_idx = (init_value & 0x0F) as i32;
        let m = slope_idx * 5 - 45;
        let n = (offset_idx << 3) - 16;
        let qp = slice_qp.clamp(0, 51);
        let pre = ((m * qp) >> 4) + n;
        let pre = pre.clamp(1, 126);
        if pre <= 63 {
            CtxModel {
                state: (63 - pre) as u8,
                mps: 0,
            }
        } else {
            CtxModel {
                state: (pre - 64) as u8,
                mps: 1,
            }
        }
    }

    /// A context pinned to an explicit `(state, mps)` — for tests and for the
    /// equiprobable-ish defaults some tools use before real init constants land.
    pub fn from_state(state: u8, mps: u8) -> Self {
        debug_assert!(state < 64);
        CtxModel {
            state,
            mps: mps & 1,
        }
    }
}

/// CABAC arithmetic encoder (H.265 §9.3.4.3).
///
/// Feed bins via [`encode_bin`](Self::encode_bin) (context-modeled),
/// [`encode_bypass`](Self::encode_bypass), and [`encode_terminate`](Self::encode_terminate).
/// The final `encode_terminate(1)` (i.e. `end_of_slice_segment_flag = 1`) flushes
/// the engine; [`finish`](Self::finish) then returns the byte-aligned slice data.
///
/// `Clone` is what makes rate estimation possible: an encoder can copy the live
/// engine and contexts, code a candidate into the copy, read [`bit_len`](Self::bit_len),
/// and throw it away without disturbing the real bitstream.
#[derive(Clone)]
pub struct CabacEncoder {
    /// `ivlLow` — low bound of the current interval.
    low: u32,
    /// `ivlCurrRange` — width of the current interval (kept in `[256, 510]`
    /// except transiently inside a bin).
    range: u32,
    /// Count of undecided (outstanding) bits awaiting a resolving carry.
    bits_outstanding: u32,
    /// The very first output bit is discarded (it carries no information).
    first_bit: bool,
    out: BitWriter,
}

impl Default for CabacEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl CabacEncoder {
    /// A freshly initialized engine (§9.3.2.5): range `510`, low `0`.
    pub fn new() -> Self {
        Self {
            low: 0,
            range: 510,
            bits_outstanding: 0,
            first_bit: true,
            out: BitWriter::new(),
        }
    }

    /// Emit a resolved bit `b`, then flush any outstanding bits as `1 - b`
    /// (§9.3.4.3.5 `PutBit`). The first ever bit is swallowed.
    #[inline]
    fn put_bit(&mut self, b: u32) {
        if self.first_bit {
            self.first_bit = false;
        } else {
            self.out.write_bit(b);
        }
        while self.bits_outstanding > 0 {
            self.out.write_bit(1 - (b & 1));
            self.bits_outstanding -= 1;
        }
    }

    /// Renormalize so `range` returns to `[256, 510]` (§9.3.4.3.3).
    #[inline]
    fn renorm(&mut self) {
        while self.range < 256 {
            if self.low < 256 {
                self.put_bit(0);
            } else if self.low >= 512 {
                self.low -= 512;
                self.put_bit(1);
            } else {
                self.low -= 256;
                self.bits_outstanding += 1;
            }
            self.range <<= 1;
            self.low <<= 1;
        }
    }

    /// Encode one context-modeled decision (§9.3.4.3.2), updating `ctx`.
    pub fn encode_bin(&mut self, ctx: &mut CtxModel, bin: u32) {
        let q = ((self.range >> 6) & 3) as usize;
        let lps = RANGE_TAB_LPS[ctx.state as usize][q] as u32;
        self.range -= lps;
        if (bin & 1) != ctx.mps as u32 {
            // Least-probable symbol.
            self.low += self.range;
            self.range = lps;
            if ctx.state == 0 {
                ctx.mps = 1 - ctx.mps;
            }
            ctx.state = TRANS_IDX_LPS[ctx.state as usize];
        } else {
            // Most-probable symbol.
            ctx.state = TRANS_IDX_MPS[ctx.state as usize];
        }
        self.renorm();
    }

    /// Encode one equiprobable bin with no context (§9.3.4.3.4). Faster and used
    /// for sign bits, high-order coefficient bits, etc.
    pub fn encode_bypass(&mut self, bin: u32) {
        self.low <<= 1;
        if (bin & 1) != 0 {
            self.low += self.range;
        }
        if self.low >= 1024 {
            self.put_bit(1);
            self.low -= 1024;
        } else if self.low < 512 {
            self.put_bit(0);
        } else {
            self.low -= 512;
            self.bits_outstanding += 1;
        }
    }

    /// Encode `n` equiprobable bins from the low bits of `value`, MSB-first.
    pub fn encode_bypass_bits(&mut self, value: u32, n: u32) {
        for i in (0..n).rev() {
            self.encode_bypass((value >> i) & 1);
        }
    }

    /// Encode a termination decision (§9.3.4.3.5). `bin = 1` ends the slice and
    /// flushes the engine; `bin = 0` just renormalizes and coding continues.
    pub fn encode_terminate(&mut self, bin: u32) {
        self.range -= 2;
        if (bin & 1) != 0 {
            self.low += self.range;
            self.flush();
        } else {
            self.renorm();
        }
    }

    /// Encode-flush (§9.3.4.3.6): drain the interval to the bitstream. Called by
    /// `encode_terminate(1)`.
    fn flush(&mut self) {
        self.range = 2;
        self.renorm();
        self.put_bit((self.low >> 9) & 1);
        self.out.write_bits(((self.low >> 7) & 3) | 1, 2);
    }

    /// The arithmetic state with an empty output buffer.
    ///
    /// Rate-distortion decisions need to know what a candidate *would* cost, so
    /// they encode it into a copy and read [`bit_len`](Self::bit_len). Cloning
    /// the whole encoder would copy every byte written so far, once per
    /// candidate per level — quadratic in picture size. The bit count is a
    /// difference, so the buffer can start empty as long as `low`, `range` and
    /// the outstanding-bit count carry over.
    pub fn probe(&self) -> Self {
        Self {
            low: self.low,
            range: self.range,
            bits_outstanding: self.bits_outstanding,
            first_bit: self.first_bit,
            out: BitWriter::new(),
        }
    }

    /// Number of bits emitted so far (for rate estimation / debugging).
    pub fn bit_len(&self) -> usize {
        self.out.bit_len()
    }

    /// Direct access to the underlying bit writer. Only valid to use between a
    /// flush (`encode_terminate(1)`, as `I_PCM` does) and [`reinit`](Self::reinit)
    /// — e.g. to emit `pcm_alignment_zero_bit`s and raw PCM samples that bypass
    /// the arithmetic coder (H.265 §7.3.8.5 / §9.3.1).
    pub fn writer_mut(&mut self) -> &mut BitWriter {
        &mut self.out
    }

    /// Re-initialize the arithmetic engine (range `510`, low `0`), keeping the
    /// already-emitted bytes. The decoder does the same after reading an `I_PCM`
    /// CU's samples, so the two stay in lockstep across the PCM interruption.
    pub fn reinit(&mut self) {
        self.low = 0;
        self.range = 510;
        self.bits_outstanding = 0;
        self.first_bit = true;
    }

    /// Finish and return the byte-aligned coded bytes. Call after the terminating
    /// `encode_terminate(1)`.
    pub fn finish(self) -> Vec<u8> {
        self.out.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal MSB-first bit reader that returns `0` once the buffer is
    /// exhausted — matching how a real decoder sees an over-read at a NAL's end.
    struct BitReader<'a> {
        bytes: &'a [u8],
        pos: usize,
    }
    impl<'a> BitReader<'a> {
        fn new(bytes: &'a [u8]) -> Self {
            Self { bytes, pos: 0 }
        }
        fn read_bit(&mut self) -> u32 {
            let byte = self.pos >> 3;
            let bit = 7 - (self.pos & 7);
            self.pos += 1;
            if byte < self.bytes.len() {
                ((self.bytes[byte] >> bit) & 1) as u32
            } else {
                0
            }
        }
        fn read_bits(&mut self, n: u32) -> u32 {
            let mut v = 0;
            for _ in 0..n {
                v = (v << 1) | self.read_bit();
            }
            v
        }
    }

    /// The decoding counterpart of [`CabacEncoder`] (§9.3.4.3), used only to
    /// prove the encoder roundtrips.
    struct CabacDecoder<'a> {
        range: u32,
        offset: u32,
        r: BitReader<'a>,
    }
    impl<'a> CabacDecoder<'a> {
        fn new(bytes: &'a [u8]) -> Self {
            let mut r = BitReader::new(bytes);
            let offset = r.read_bits(9);
            Self {
                range: 510,
                offset,
                r,
            }
        }
        fn renorm(&mut self) {
            while self.range < 256 {
                self.range <<= 1;
                self.offset = (self.offset << 1) | self.r.read_bit();
            }
        }
        fn decode_bin(&mut self, ctx: &mut CtxModel) -> u32 {
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
            self.renorm();
            bin
        }
        fn decode_bypass(&mut self) -> u32 {
            self.offset = (self.offset << 1) | self.r.read_bit();
            if self.offset >= self.range {
                self.offset -= self.range;
                1
            } else {
                0
            }
        }
        fn decode_terminate(&mut self) -> u32 {
            self.range -= 2;
            if self.offset >= self.range {
                1
            } else {
                self.renorm();
                0
            }
        }
    }

    /// A tiny deterministic PRNG (xorshift32) so the roundtrip is reproducible
    /// without `rand` or `Math.random` (unavailable in this environment).
    struct Rng(u32);
    impl Rng {
        fn next(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = x;
            x
        }
        fn bit(&mut self) -> u32 {
            self.next() & 1
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum Op {
        Regular(usize, u32),
        Bypass(u32),
        Terminate(u32),
    }

    #[test]
    fn ctx_init_matches_spec_examples() {
        // Hand-computed from §9.3.2.2 at SliceQpY = 26.
        // initValue = 154: slope=9,offset=10 → m=0, n=64, pre=64 → state 0, mps 1.
        assert_eq!(CtxModel::init(154, 26), CtxModel { state: 0, mps: 1 });
        // initValue = 0: m=-45, n=-16, pre=clip(-90)=1 → state 62, mps 0.
        assert_eq!(CtxModel::init(0, 26), CtxModel { state: 62, mps: 0 });
    }

    fn run_roundtrip(ops: &[Op], ctx_seed: &[CtxModel]) {
        // Encode.
        let mut enc = CabacEncoder::new();
        let mut ectx: Vec<CtxModel> = ctx_seed.to_vec();
        for &op in ops {
            match op {
                Op::Regular(c, b) => enc.encode_bin(&mut ectx[c], b),
                Op::Bypass(b) => enc.encode_bypass(b),
                Op::Terminate(b) => enc.encode_terminate(b),
            }
        }
        let bytes = enc.finish();

        // Decode with identical initial contexts and op structure.
        let mut dec = CabacDecoder::new(&bytes);
        let mut dctx: Vec<CtxModel> = ctx_seed.to_vec();
        for (i, &op) in ops.iter().enumerate() {
            match op {
                Op::Regular(c, b) => {
                    assert_eq!(dec.decode_bin(&mut dctx[c]), b, "regular bin #{i}");
                }
                Op::Bypass(b) => assert_eq!(dec.decode_bypass(), b, "bypass bin #{i}"),
                Op::Terminate(b) => assert_eq!(dec.decode_terminate(), b, "terminate bin #{i}"),
            }
        }
    }

    #[test]
    fn roundtrip_simple() {
        let seed = vec![CtxModel::from_state(0, 0), CtxModel::from_state(31, 1)];
        let ops = vec![
            Op::Regular(0, 1),
            Op::Regular(0, 0),
            Op::Regular(1, 1),
            Op::Bypass(1),
            Op::Bypass(0),
            Op::Regular(1, 0),
            Op::Terminate(0),
            Op::Regular(0, 1),
            Op::Terminate(1),
        ];
        run_roundtrip(&ops, &seed);
    }

    #[test]
    fn roundtrip_randomized() {
        // Many random scripts across seeds — the strong correctness signal for
        // the arithmetic engine and all three coding tables.
        for trial in 0..200u32 {
            let mut rng = Rng(0x9E37_79B9 ^ trial.wrapping_mul(2654435761));
            let seed = vec![
                CtxModel::init((rng.next() & 0xFF) as u8, (rng.next() % 52) as i32),
                CtxModel::init((rng.next() & 0xFF) as u8, (rng.next() % 52) as i32),
                CtxModel::init((rng.next() & 0xFF) as u8, (rng.next() % 52) as i32),
                CtxModel::init((rng.next() & 0xFF) as u8, (rng.next() % 52) as i32),
            ];
            let len = 20 + (rng.next() % 400) as usize;
            let mut ops = Vec::with_capacity(len + 1);
            for _ in 0..len {
                match rng.next() % 10 {
                    0..=6 => ops.push(Op::Regular((rng.next() % 4) as usize, rng.bit())),
                    7..=8 => ops.push(Op::Bypass(rng.bit())),
                    _ => ops.push(Op::Terminate(0)),
                }
            }
            ops.push(Op::Terminate(1)); // end_of_slice_segment_flag
            run_roundtrip(&ops, &seed);
        }
    }

    #[test]
    fn roundtrip_all_bypass() {
        let seed = vec![CtxModel::from_state(0, 0)];
        let mut rng = Rng(0x1234_5678);
        let mut ops: Vec<Op> = (0..500).map(|_| Op::Bypass(rng.bit())).collect();
        ops.push(Op::Terminate(1));
        run_roundtrip(&ops, &seed);
    }
}
