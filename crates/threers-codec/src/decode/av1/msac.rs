//! AV1 multi-symbol arithmetic decoder (dav1d `src/msac.c`, C paths).
//!
//! CDFs are stored inverted (`32768 - cdf`), with the adaptation counter in
//! the slot after the last symbol.

const EC_PROB_SHIFT: u32 = 6;
const EC_MIN_PROB: u32 = 4;
const EC_WIN_SIZE: i32 = 64;

pub struct Msac {
    buf: Vec<u8>,
    pos: usize,
    dif: u64,
    rng: u32,
    cnt: i32,
    pub allow_update_cdf: bool,
}

#[inline]
pub fn inv_recenter(r: u32, v: u32) -> u32 {
    if v > (r << 1) {
        v
    } else if v & 1 == 0 {
        (v >> 1) + r
    } else {
        r - ((v + 1) >> 1)
    }
}

#[inline]
pub fn ulog2(v: u32) -> i32 {
    31 ^ v.leading_zeros() as i32
}

impl Msac {
    pub fn new(data: &[u8], disable_cdf_update: bool) -> Self {
        let mut s = Msac { buf: data.to_vec(), pos: 0, dif: 0, rng: 0x8000, cnt: -15, allow_update_cdf: !disable_cdf_update };
        s.refill();
        s
    }

    #[inline]
    fn refill(&mut self) {
        let mut c = EC_WIN_SIZE - self.cnt - 24;
        let mut dif = self.dif;
        loop {
            if self.pos >= self.buf.len() {
                // Set the remaining bits to 1.
                dif |= !(!0xffu64 << c);
                break;
            }
            dif |= ((self.buf[self.pos] ^ 0xff) as u64) << c;
            self.pos += 1;
            c -= 8;
            if c < 0 {
                break;
            }
        }
        self.dif = dif;
        self.cnt = EC_WIN_SIZE - c - 24;
    }

    #[inline]
    fn norm(&mut self, dif: u64, rng: u32) {
        let d = 15 ^ (31 ^ rng.leading_zeros() as i32);
        let cnt = self.cnt;
        self.dif = dif << d;
        self.rng = rng << d;
        self.cnt = cnt - d;
        if (cnt as u32) < (d as u32) {
            self.refill();
        }
    }

    #[inline]
    pub fn bool_equi(&mut self) -> bool {
        let r = self.rng;
        let mut dif = self.dif;
        let mut v = ((r >> 8) << 7) + EC_MIN_PROB;
        let vw = (v as u64) << (EC_WIN_SIZE - 16);
        let ret = dif >= vw;
        if ret {
            dif -= vw;
            v = r - v;
        }
        self.norm(dif, v);
        !ret
    }

    /// `f`: the (inverted) probability.
    #[inline]
    pub fn bool(&mut self, f: u32) -> bool {
        let r = self.rng;
        let mut dif = self.dif;
        let mut v = ((r >> 8) * (f >> EC_PROB_SHIFT) >> (7 - EC_PROB_SHIFT)) + EC_MIN_PROB;
        let vw = (v as u64) << (EC_WIN_SIZE - 16);
        let ret = dif >= vw;
        if ret {
            dif -= vw;
            v = r - v;
        }
        self.norm(dif, v);
        !ret
    }

    pub fn symbol_adapt(&mut self, cdf: &mut [u16], n_symbols: usize) -> u32 {
        let c = (self.dif >> (EC_WIN_SIZE - 16)) as u32;
        let r = self.rng >> 8;
        let mut v = self.rng;
        let mut u;
        let mut val: u32 = u32::MAX;
        loop {
            val = val.wrapping_add(1);
            u = v;
            v = r * (cdf[val as usize] as u32 >> EC_PROB_SHIFT);
            v >>= 7 - EC_PROB_SHIFT;
            v += EC_MIN_PROB * (n_symbols as u32 - val);
            if c >= v {
                break;
            }
        }
        self.norm(self.dif - ((v as u64) << (EC_WIN_SIZE - 16)), u - v);
        if self.allow_update_cdf {
            let count = cdf[n_symbols] as u32;
            let rate = 4 + (count >> 4) + (n_symbols > 2) as u32;
            let val = val as usize;
            for x in cdf.iter_mut().take(val) {
                *x += (32768 - *x) >> rate;
            }
            for x in cdf.iter_mut().take(n_symbols).skip(val) {
                *x -= *x >> rate;
            }
            cdf[n_symbols] = (count + (count < 32) as u32) as u16;
        }
        val
    }

    pub fn bool_adapt(&mut self, cdf: &mut [u16]) -> bool {
        let bit = self.bool(cdf[0] as u32);
        if self.allow_update_cdf {
            let count = cdf[1] as u32;
            let rate = 4 + (count >> 4);
            if bit {
                cdf[0] += (32768 - cdf[0]) >> rate;
            } else {
                cdf[0] -= cdf[0] >> rate;
            }
            cdf[1] = (count + (count < 32) as u32) as u16;
        }
        bit
    }

    pub fn hi_tok(&mut self, cdf: &mut [u16]) -> u32 {
        let mut tok_br = self.symbol_adapt(cdf, 3);
        let mut tok = 3 + tok_br;
        if tok_br == 3 {
            tok_br = self.symbol_adapt(cdf, 3);
            tok = 6 + tok_br;
            if tok_br == 3 {
                tok_br = self.symbol_adapt(cdf, 3);
                tok = 9 + tok_br;
                if tok_br == 3 {
                    tok = 12 + self.symbol_adapt(cdf, 3);
                }
            }
        }
        tok
    }

    /// `check_trailing_bits_after_symbol_coder`: the marker bit and zero
    /// padding after the last symbol (true if valid).
    pub fn trailing_ok(&self) -> bool {
        let n_bits = -(self.cnt + 14);
        let n_bytes = (n_bits + 7) >> 3;
        let p = self.pos as isize + n_bytes as isize;
        if p < 1 || p as usize > self.buf.len() {
            return false;
        }
        let pattern = 128i32 >> ((n_bits - 1) & 7);
        if (self.buf[p as usize - 1] as i32 & (2 * pattern - 1)) != pattern {
            return false;
        }
        self.buf[p as usize..].iter().all(|&b| b == 0)
    }

    /// Bits consumed past the end of the data (dav1d `cnt <= -15` check).
    pub fn overread(&self) -> bool {
        self.cnt <= -15
    }

    #[inline]
    pub fn bools(&mut self, n: u32) -> u32 {
        let mut v = 0;
        for _ in 0..n {
            v = (v << 1) | self.bool_equi() as u32;
        }
        v
    }

    pub fn uniform(&mut self, n: u32) -> u32 {
        let l = ulog2(n) + 1;
        let m = (1u32 << l) - n;
        let v = self.bools(l as u32 - 1);
        if v < m {
            v
        } else {
            (v << 1) - m + self.bool_equi() as u32
        }
    }

    pub fn subexp(&mut self, r: i32, n: i32, mut k: u32) -> i32 {
        let mut a = 0u32;
        if self.bool_equi() {
            if self.bool_equi() {
                k += self.bool_equi() as u32 + 1;
            }
            a = 1 << k;
        }
        let v = self.bools(k) + a;
        if r * 2 <= n {
            inv_recenter(r as u32, v) as i32
        } else {
            n - 1 - inv_recenter((n - 1 - r) as u32, v) as i32
        }
    }
}
