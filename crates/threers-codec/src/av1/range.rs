//! AV1's multi-symbol range encoder, the inverse of
//! [`decode::av1::msac`](crate::decode::av1::msac).
//!
//! CDFs are inverted (`32768 - cdf`) as the decoder keeps them. Probabilities
//! never adapt here: frames set `disable_cdf_update`.

const PROB_SHIFT: u32 = 6;
const MIN_PROB: u32 = 4;

pub(super) struct RangeEncoder {
    low: u64,
    rng: u32,
    cnt: i32,
    /// Output before carries are resolved, up to 16 bits a slot.
    precarry: Vec<u16>,
}

impl RangeEncoder {
    pub(super) fn new() -> Self {
        RangeEncoder {
            low: 0,
            rng: 0x8000,
            cnt: -9,
            precarry: Vec::new(),
        }
    }

    /// `s` of the `n + 1` symbols `cdf` describes.
    pub(super) fn symbol(&mut self, cdf: &[u16], n: usize, s: usize) {
        let scaled = |f: u32, k: usize, r: u32| {
            (((r >> 8) * (f >> PROB_SHIFT)) >> (7 - PROB_SHIFT)) + MIN_PROB * k as u32
        };
        let r = self.rng;
        let fh = if s < n { cdf[s] as u32 } else { 0 };
        let v = scaled(fh, n - s, r);
        if s > 0 {
            let u = scaled(cdf[s - 1] as u32, n - s + 1, r);
            self.low += (r - u) as u64;
            self.normalize(u - v);
        } else {
            self.normalize(r - v);
        }
    }

    /// A binary symbol with `f` as its inverted probability of 0.
    pub(super) fn bool(&mut self, f: u32, bit: bool) {
        self.symbol(&[f as u16], 1, bit as usize);
    }

    pub(super) fn bool_equi(&mut self, bit: bool) {
        self.bool(1 << 14, bit);
    }

    /// `n` raw bits, most significant first.
    pub(super) fn bools(&mut self, value: u32, n: u32) {
        for i in (0..n).rev() {
            self.bool_equi((value >> i) & 1 != 0);
        }
    }

    /// Exp-Golomb, as the decoder's `read_golomb`.
    pub(super) fn golomb(&mut self, value: u32) {
        let x = value + 1;
        let len = 31 - x.leading_zeros();
        for _ in 0..len {
            self.bool_equi(false);
        }
        self.bool_equi(true);
        self.bools(x & ((1 << len) - 1), len);
    }

    /// A coefficient level from 3 to 15, in steps of up to three.
    pub(super) fn hi_tok(&mut self, cdf: &[u16], tok: u32) {
        let mut rest = tok - 3;
        for _ in 0..4 {
            let s = rest.min(3);
            self.symbol(cdf, 3, s as usize);
            if s < 3 {
                break;
            }
            rest -= 3;
        }
    }

    fn normalize(&mut self, rng: u32) {
        let d = 16 - (32 - rng.leading_zeros()) as i32;
        let mut c = self.cnt;
        let mut s = c + d;
        if s >= 0 {
            c += 16;
            let mut m = (1u64 << c) - 1;
            if s >= 8 {
                self.precarry.push((self.low >> c) as u16);
                self.low &= m;
                c -= 8;
                m >>= 8;
            }
            self.precarry.push((self.low >> c) as u16);
            s = c + d - 24;
            self.low &= m;
        }
        self.low <<= d;
        self.rng = rng << d;
        self.cnt = s;
    }

    /// The bytes, with carries propagated.
    pub(super) fn finish(mut self) -> Vec<u8> {
        let m = 0x3fffu64;
        let mut e = ((self.low + m) & !m) | (m + 1);
        let mut c = self.cnt;
        let mut s = 10 + c;
        if s > 0 {
            let mut n = (1u64 << (c + 16)) - 1;
            loop {
                self.precarry.push((e >> (c + 16)) as u16);
                e &= n;
                s -= 8;
                c -= 8;
                n >>= 8;
                if s <= 0 {
                    break;
                }
            }
        }
        let mut out = vec![0u8; self.precarry.len()];
        let mut carry = 0u32;
        for (o, &p) in out.iter_mut().zip(&self.precarry).rev() {
            carry += p as u32;
            *o = carry as u8;
            carry >>= 8;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::av1::msac::Msac;

    /// Against the decoder: symbols of every alphabet size, skewed and flat,
    /// with enough of them that carries occur.
    #[test]
    fn the_decoder_reads_back_what_was_written() {
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let cdf4 = [24000u16, 9000, 1000, 0, 0];
        let cdf2 = [30000u16, 0, 0];
        let mut script = Vec::new();
        let mut enc = RangeEncoder::new();
        for _ in 0..20000 {
            let kind = next() % 5;
            let v = next();
            match kind {
                0 => enc.symbol(&cdf4, 3, (v % 4) as usize),
                1 => enc.bool(cdf2[0] as u32, v & 1 != 0),
                2 => enc.bools((v & 0xff) as u32, 8),
                3 => enc.golomb((v % 5000) as u32),
                _ => enc.hi_tok(&cdf4, 3 + (v % 13) as u32),
            }
            script.push((kind, v));
        }
        let bytes = enc.finish();
        let mut dec = Msac::new(&bytes, true);
        let (mut c4, mut c2) = (cdf4, cdf2);
        for (kind, v) in script {
            match kind {
                0 => assert_eq!(dec.symbol_adapt(&mut c4, 3), (v % 4) as u32),
                1 => assert_eq!(dec.bool_adapt(&mut c2), v & 1 != 0),
                2 => assert_eq!(dec.bools(8), (v & 0xff) as u32),
                3 => {
                    let mut len = 0;
                    while !dec.bool_equi() {
                        len += 1;
                    }
                    let mut x = 1u32;
                    for _ in 0..len {
                        x = (x << 1) + dec.bool_equi() as u32;
                    }
                    assert_eq!(x - 1, (v % 5000) as u32);
                }
                _ => assert_eq!(dec.hi_tok(&mut c4), 3 + (v % 13) as u32),
            }
        }
    }
}
