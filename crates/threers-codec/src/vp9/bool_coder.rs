//! The VP8/VP9 boolean entropy coder — a binary range coder.
//!
//! Every compressed VP9 syntax element is coded as a series of boolean decisions,
//! each with an 8-bit probability (the chance the bit is 0). This is VP9's analog
//! of HEVC's CABAC engine, and like it, it is the one kernel that must be
//! bit-exact: a single wrong renormalization corrupts the whole partition.
//!
//! The arithmetic matches RFC 6386 §7, but the start/stop convention follows
//! libvpx's `vpx_writer` (a leading equiprobable `0` on start, thirty-two
//! equiprobable `0`s on stop) so the bitstream is accepted by ffmpeg, browsers,
//! and every other libvpx-derived decoder. Pure arithmetic over `Vec<u8>` —
//! identical on native and `wasm32`.
//!
//! [`BoolEncoder`] is the encode side. Correctness is pinned by a matching
//! decoder in the tests: random `(prob, bit)` scripts are encoded then decoded
//! and must come back identical.

/// VP8/VP9 boolean-range encoder (libvpx `vpx_writer` / RFC 6386 §7.3).
pub struct BoolEncoder {
    out: Vec<u8>,
    /// `range` — width of the current interval, kept in `[128, 255]`.
    range: u32,
    /// `bottom` — low end of the current interval (with pending output bits).
    bottom: u32,
    /// Shifts remaining before the next output byte is complete.
    bit_count: i32,
}

impl Default for BoolEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl BoolEncoder {
    /// A fresh encoder primed like libvpx `vpx_start_encode`: `range = 255`,
    /// `bottom = 0`, and one leading equiprobable `0` so the first emitted byte
    /// lines up with a libvpx/ffmpeg bool decoder.
    pub fn new() -> Self {
        let mut e = Self {
            out: Vec::new(),
            range: 255,
            bottom: 0,
            bit_count: 24,
        };
        e.put_bit(false); // libvpx primes with vpx_write(0, 128)
        e
    }

    /// Propagate a carry (`+1`) backward through already-emitted bytes
    /// (RFC 6386 `add_one_to_output`). The final `insert` is a defensive
    /// backstop for a whole-number carry; the priming above makes it unreachable
    /// in normal operation.
    fn carry(&mut self) {
        let mut i = self.out.len();
        while i > 0 {
            i -= 1;
            if self.out[i] == 0xff {
                self.out[i] = 0;
            } else {
                self.out[i] += 1;
                return;
            }
        }
        self.out.insert(0, 1);
    }

    /// One renormalization shift: emit a bit of `bottom`, completing a byte every
    /// eight shifts.
    #[inline]
    fn shift(&mut self) {
        if self.bottom & 0x8000_0000 != 0 {
            self.carry();
        }
        self.bottom <<= 1;
        self.bit_count -= 1;
        if self.bit_count == 0 {
            self.out.push((self.bottom >> 24) as u8);
            self.bottom &= (1 << 24) - 1;
            self.bit_count = 8;
        }
    }

    /// Encode one boolean `bit` whose probability of being `0` is `prob/256`
    /// (RFC 6386 `write_bool`).
    pub fn put_bool(&mut self, prob: u8, bit: bool) {
        let split = 1 + (((self.range - 1) * prob as u32) >> 8);
        if bit {
            self.bottom += split;
            self.range -= split;
        } else {
            self.range = split;
        }
        while self.range < 128 {
            self.range <<= 1;
            self.shift();
        }
    }

    /// Encode one equiprobable bit (probability 128, i.e. a literal bit).
    pub fn put_bit(&mut self, bit: bool) {
        self.put_bool(128, bit);
    }

    /// Encode the `n` low bits of `value`, MSB first, as literals.
    pub fn put_literal(&mut self, value: u32, n: u32) {
        for i in (0..n).rev() {
            self.put_bit((value >> i) & 1 != 0);
        }
    }

    /// Flush like libvpx `vpx_stop_encode`: thirty-two equiprobable `0`s so the
    /// pending interval is fully emitted, then return the coded bytes.
    pub fn finish(mut self) -> Vec<u8> {
        for _ in 0..32 {
            self.put_bit(false);
        }
        self.out
    }

    /// Bytes emitted so far (renormalized; excludes the pending interval).
    pub fn len(&self) -> usize {
        self.out.len()
    }

    pub fn is_empty(&self) -> bool {
        self.out.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The decode counterpart (RFC 6386 §7.3), used only to prove the encoder
    /// round-trips.
    struct BoolDecoder<'a> {
        input: &'a [u8],
        pos: usize,
        range: u32,
        value: u32,
        bit_count: i32,
    }
    impl<'a> BoolDecoder<'a> {
        fn new(input: &'a [u8]) -> Self {
            let b0 = input.first().copied().unwrap_or(0) as u32;
            let b1 = input.get(1).copied().unwrap_or(0) as u32;
            let mut d = Self {
                input,
                pos: 2,
                range: 255,
                value: (b0 << 8) | b1,
                bit_count: 0,
            };
            // libvpx `vpx_reader_init` consumes the encoder's priming 0 bit.
            assert!(!d.get_bool(128), "bool coder marker bit must be 0");
            d
        }
        fn get_bool(&mut self, prob: u8) -> bool {
            let split = 1 + (((self.range - 1) * prob as u32) >> 8);
            let big_split = split << 8;
            let ret = if self.value >= big_split {
                self.range -= split;
                self.value -= big_split;
                true
            } else {
                self.range = split;
                false
            };
            while self.range < 128 {
                self.value <<= 1;
                self.range <<= 1;
                self.bit_count += 1;
                if self.bit_count == 8 {
                    self.bit_count = 0;
                    let byte = self.input.get(self.pos).copied().unwrap_or(0) as u32;
                    self.pos += 1;
                    self.value |= byte;
                }
            }
            ret
        }
        fn get_literal(&mut self, n: u32) -> u32 {
            let mut v = 0;
            for _ in 0..n {
                v = (v << 1) | self.get_bool(128) as u32;
            }
            v
        }
    }

    /// A tiny deterministic xorshift PRNG (no `rand` / `Math.random`).
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
    }

    #[test]
    fn roundtrip_simple() {
        let script: &[(u8, bool)] = &[
            (128, true),
            (128, false),
            (10, true),
            (250, false),
            (200, true),
            (5, false),
        ];
        let mut enc = BoolEncoder::new();
        for &(p, b) in script {
            enc.put_bool(p, b);
        }
        let bytes = enc.finish();
        let mut dec = BoolDecoder::new(&bytes);
        for &(p, b) in script {
            assert_eq!(dec.get_bool(p), b);
        }
    }

    #[test]
    fn roundtrip_randomized() {
        // Many random scripts, mixing modeled bools and literals — the strong
        // correctness signal for the range coder and its carry propagation.
        for trial in 0..300u32 {
            let mut rng = Rng(0x9E37_79B9 ^ trial.wrapping_mul(2654435761));
            let len = 20 + (rng.next() % 500) as usize;
            let ops: Vec<(u8, bool)> = (0..len)
                .map(|_| {
                    // Bias probabilities toward the extremes to stress renorm.
                    let p = match rng.next() % 4 {
                        0 => 1,
                        1 => 255,
                        _ => (rng.next() % 256) as u8,
                    };
                    (p.max(1), rng.next() & 1 != 0)
                })
                .collect();

            let mut enc = BoolEncoder::new();
            for &(p, b) in &ops {
                enc.put_bool(p, b);
            }
            let bytes = enc.finish();

            let mut dec = BoolDecoder::new(&bytes);
            for (i, &(p, b)) in ops.iter().enumerate() {
                assert_eq!(dec.get_bool(p), b, "trial {trial} bin {i} (prob {p})");
            }
        }
    }

    #[test]
    fn roundtrip_literals() {
        let mut rng = Rng(0x1234_5678);
        let vals: Vec<(u32, u32)> = (0..200)
            .map(|_| {
                let n = 1 + rng.next() % 16;
                (rng.next() & ((1 << n) - 1), n)
            })
            .collect();
        let mut enc = BoolEncoder::new();
        for &(v, n) in &vals {
            enc.put_literal(v, n);
        }
        let bytes = enc.finish();
        let mut dec = BoolDecoder::new(&bytes);
        for &(v, n) in &vals {
            assert_eq!(dec.get_literal(n), v, "literal {v} ({n} bits)");
        }
    }

    #[test]
    fn roundtrip_all_ones_carry_stress() {
        // Long runs of the least-probable symbol force maximal carry propagation.
        let mut enc = BoolEncoder::new();
        let ops: Vec<(u8, bool)> = (0..2000).map(|i| (1u8, i % 7 != 0)).collect();
        for &(p, b) in &ops {
            enc.put_bool(p, b);
        }
        let bytes = enc.finish();
        let mut dec = BoolDecoder::new(&bytes);
        for &(p, b) in &ops {
            assert_eq!(dec.get_bool(p), b);
        }
    }
}
