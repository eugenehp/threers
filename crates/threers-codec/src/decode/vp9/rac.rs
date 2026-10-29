//! The VP9 boolean range decoder (ffmpeg `vpx_rac.h` / `vp89_rac.h`).
//!
//! Reads two bytes per refill like ffmpeg; bytes past the end read as zero
//! (ffmpeg's packets carry zeroed padding), so exhausted data decodes the same.

const NORM_SHIFT: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = if i == 0 { 8 } else { (i as u8).leading_zeros() as u8 };
        i += 1;
    }
    t
};

#[derive(Clone)]
pub struct Rac<'a> {
    high: u32,
    bits: i32,
    buf: &'a [u8],
    pos: usize,
    code_word: u32,
    end_reached: u32,
}

impl<'a> Rac<'a> {
    pub fn new(buf: &'a [u8]) -> Result<Self, String> {
        if buf.is_empty() {
            return Err("empty VP9 range-coded partition".into());
        }
        let b = |i: usize| buf.get(i).copied().unwrap_or(0) as u32;
        Ok(Rac { high: 255, bits: -16, buf, pos: 3, code_word: (b(0) << 16) | (b(1) << 8) | b(2), end_reached: 0 })
    }

    /// ffmpeg `vpx_rac_is_end`: true once the data has been overrun for a
    /// while.
    pub fn is_end(&mut self) -> bool {
        if self.pos >= self.buf.len() && self.bits >= 0 {
            self.end_reached += 1;
        }
        self.end_reached > 10
    }

    #[inline(always)]
    fn renorm(&mut self) -> u32 {
        let shift = NORM_SHIFT[self.high as usize] as i32;
        self.high <<= shift;
        let mut code_word = self.code_word << shift;
        let mut bits = self.bits + shift;
        if bits >= 0 && self.pos < self.buf.len() {
            let b0 = self.buf[self.pos] as u32;
            let b1 = self.buf.get(self.pos + 1).copied().unwrap_or(0) as u32;
            self.pos += 2;
            code_word |= ((b0 << 8) | b1) << bits;
            bits -= 16;
        }
        self.bits = bits;
        code_word
    }

    /// `vpx_rac_get_prob` (and the identical `_branchy`).
    #[inline(always)]
    pub fn get_prob(&mut self, prob: u8) -> bool {
        let code_word = self.renorm();
        let low = 1 + (((self.high - 1) * prob as u32) >> 8);
        let low_shift = low << 16;
        let bit = code_word >= low_shift;
        if bit {
            self.high -= low;
            self.code_word = code_word - low_shift;
        } else {
            self.high = low;
            self.code_word = code_word;
        }
        bit
    }

    /// `vp89_rac_get`: probability 128.
    #[inline(always)]
    pub fn get(&mut self) -> bool {
        self.get_prob(128)
    }

    /// `vp89_rac_get_uint`.
    pub fn get_uint(&mut self, bits: u32) -> u32 {
        let mut v = 0;
        for _ in 0..bits {
            v = (v << 1) | self.get() as u32;
        }
        v
    }

    /// `vp89_rac_get_tree`.
    #[inline]
    pub fn get_tree(&mut self, tree: &[[i8; 2]], probs: &[u8]) -> u8 {
        let mut i = 0i32;
        loop {
            i = tree[i as usize][self.get_prob(probs[i as usize]) as usize] as i32;
            if i <= 0 {
                return (-i) as u8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_shift_matches_ffmpeg() {
        assert_eq!(NORM_SHIFT[0], 8);
        assert_eq!(NORM_SHIFT[1], 7);
        assert_eq!(NORM_SHIFT[2], 6);
        assert_eq!(NORM_SHIFT[4], 5);
        assert_eq!(NORM_SHIFT[127], 1);
        assert_eq!(NORM_SHIFT[128], 0);
        assert_eq!(NORM_SHIFT[255], 0);
    }

    #[test]
    fn roundtrips_the_encoder() {
        // threers' VP9 encoder bool coder writes; this reads it back.
        let mut e = crate::vp9::bool_coder::BoolEncoder::new();
        let bits: Vec<(u8, bool)> = (0..500u32).map(|i| (((i * 37) % 254 + 1) as u8, (i * 7919) % 3 == 0)).collect();
        for &(p, b) in &bits {
            e.put_bool(p, b);
        }
        let data = e.finish();
        let mut d = Rac::new(&data).unwrap();
        // libvpx's writer starts with a 0 at probability 128 (the marker
        // every VP9 partition begins with).
        assert!(!d.get_prob(128));
        for &(p, b) in &bits {
            assert_eq!(d.get_prob(p), b);
        }
    }
}
