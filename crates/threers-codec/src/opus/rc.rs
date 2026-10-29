//! ffmpeg n7.1's Opus range decoder (`libavcodec/opus/rc.c`).
//!
//! The forward byte reader is ffmpeg's `GetBitContext`: reads past the end
//! of the frame see whatever follows it in the packet (then zeros — the
//! packet padding), with the bit index clamped at `size * 8 + 8`, exactly as
//! the checked bitstream reader does. The raw-bit reader walks backwards
//! from the end of the frame and never overreads.

#[derive(Clone)]
pub struct Rc<'a> {
    /// The packet from the frame's first byte on (later frames included).
    buf: &'a [u8],
    /// `size_in_bits_plus8`.
    limit: usize,
    index: usize,
    // RawBitsContext
    rb_pos: usize,
    rb_bytes: u32,
    rb_cachelen: u32,
    rb_cacheval: u32,
    pub range: u32,
    pub value: u32,
    pub total_bits: u32,
}

const SYM: u32 = 8;
const CEIL: u32 = (1 << SYM) - 1;
const TOP: u32 = 1 << 31;
const BOT: u32 = TOP >> SYM;

#[inline]
pub fn ilog(i: u32) -> u32 {
    if i == 0 {
        0
    } else {
        32 - i.leading_zeros()
    }
}

#[inline]
fn log2(v: u32) -> u32 {
    31 - (v | 1).leading_zeros()
}

impl<'a> Rc<'a> {
    /// `ff_opus_rc_dec_init(rc, data, size)`, `data` being `buf[..size]`.
    pub fn new(buf: &'a [u8], size: usize) -> Self {
        let mut rc = Rc {
            buf,
            limit: size * 8 + 8,
            index: 0,
            rb_pos: 0,
            rb_bytes: 0,
            rb_cachelen: 0,
            rb_cacheval: 0,
            range: 128,
            value: 0,
            total_bits: 9,
        };
        rc.value = 127 - rc.get_bits(7);
        rc.normalize();
        rc
    }

    #[inline]
    fn byte(&self, i: usize) -> u32 {
        self.buf.get(i).copied().unwrap_or(0) as u32
    }

    #[inline]
    fn get_bits(&mut self, n: u32) -> u32 {
        let i = self.index;
        let b = i >> 3;
        let w = (self.byte(b) << 24) | (self.byte(b + 1) << 16) | (self.byte(b + 2) << 8) | self.byte(b + 3);
        let v = (w << (i & 7)) >> (32 - n);
        self.index = (i + n as usize).min(self.limit);
        v
    }

    /// `ff_opus_rc_dec_raw_init(rc, data + end, bytes)`.
    pub fn raw_init(&mut self, end: usize, bytes: u32) {
        self.rb_pos = end;
        self.rb_bytes = bytes;
        self.rb_cachelen = 0;
        self.rb_cacheval = 0;
    }

    pub fn rb_bytes(&self) -> u32 {
        self.rb_bytes
    }

    #[inline]
    fn normalize(&mut self) {
        while self.range <= BOT {
            self.value = ((self.value << SYM) | (self.get_bits(SYM) ^ CEIL)) & (TOP - 1);
            self.range <<= SYM;
            self.total_bits += SYM;
        }
    }

    #[inline]
    fn update(&mut self, scale: u32, low: u32, high: u32, total: u32) {
        self.value = self.value.wrapping_sub(scale.wrapping_mul(total - high));
        self.range = if low != 0 { scale * (high - low) } else { self.range - scale * (total - high) };
        self.normalize();
    }

    #[inline]
    pub fn tell(&self) -> u32 {
        self.total_bits.wrapping_sub(log2(self.range)).wrapping_sub(1)
    }

    pub fn tell_frac(&self) -> u32 {
        let total_bits = self.total_bits << 3;
        let mut rcbuffer = log2(self.range) + 1;
        let mut range = self.range >> (rcbuffer - 16);
        for _ in 0..3 {
            range = (range * range) >> 15;
            let bit = range >> 16;
            rcbuffer = rcbuffer << 1 | bit;
            range >>= bit;
        }
        total_bits.wrapping_sub(rcbuffer)
    }

    /// `ff_opus_rc_dec_cdf`: `cdf[0]` is the total.
    pub fn dec_cdf(&mut self, cdf: &[u16]) -> u32 {
        let total = cdf[0] as u32;
        let cdf = &cdf[1..];
        let scale = self.range / total;
        let symbol = self.value / scale + 1;
        let symbol = total - symbol.min(total);
        let mut k = 0;
        while cdf[k] as u32 <= symbol {
            k += 1;
        }
        let high = cdf[k] as u32;
        let low = if k > 0 { cdf[k - 1] as u32 } else { 0 };
        self.update(scale, low, high, total);
        k as u32
    }

    pub fn dec_log(&mut self, bits: u32) -> u32 {
        let scale = self.range >> bits;
        let k = if self.value >= scale {
            self.value -= scale;
            self.range -= scale;
            0
        } else {
            self.range = scale;
            1
        };
        self.normalize();
        k
    }

    pub fn get_raw(&mut self, count: u32) -> u32 {
        while self.rb_bytes > 0 && self.rb_cachelen < count {
            self.rb_pos -= 1;
            self.rb_cacheval |= (self.buf[self.rb_pos] as u32) << self.rb_cachelen;
            self.rb_cachelen += 8;
            self.rb_bytes -= 1;
        }
        let value = if count >= 32 { self.rb_cacheval } else { self.rb_cacheval & ((1u32 << count) - 1) };
        self.rb_cacheval = if count >= 32 { 0 } else { self.rb_cacheval >> count };
        self.rb_cachelen = self.rb_cachelen.wrapping_sub(count);
        self.total_bits += count;
        value
    }

    pub fn dec_uint(&mut self, size: u32) -> u32 {
        let bits = ilog(size - 1);
        let total = if bits > 8 { ((size - 1) >> (bits - 8)) + 1 } else { size };
        let scale = self.range / total;
        let k = self.value / scale + 1;
        let k = total - k.min(total);
        self.update(scale, k, k + 1, total);
        if bits > 8 {
            let k = k << (bits - 8) | self.get_raw(bits - 8);
            k.min(size - 1)
        } else {
            k
        }
    }

    pub fn dec_uint_step(&mut self, k0: u32) -> u32 {
        let total = (k0 + 1) * 3 + k0;
        let scale = self.range / total;
        let symbol = self.value / scale + 1;
        let symbol = total - symbol.min(total);
        let k = if symbol < (k0 + 1) * 3 { symbol / 3 } else { symbol - (k0 + 1) * 2 };
        let (lo, hi) = if k <= k0 { (3 * k, 3 * (k + 1)) } else { ((k - 1 - k0) + 3 * (k0 + 1), (k - k0) + 3 * (k0 + 1)) };
        self.update(scale, lo, hi, total);
        k
    }

    pub fn dec_uint_tri(&mut self, qn: u32) -> u32 {
        let total = ((qn >> 1) + 1) * ((qn >> 1) + 1);
        let scale = self.range / total;
        let center = self.value / scale + 1;
        let center = total - center.min(total);
        let (k, low, symbol);
        if center < total >> 1 {
            k = (ff_sqrt(8 * center + 1) - 1) >> 1;
            low = (k * (k + 1)) >> 1;
            symbol = k + 1;
        } else {
            k = (2 * (qn + 1) - ff_sqrt(8 * (total - center - 1) + 1)) >> 1;
            low = total - (((qn + 1 - k) * (qn + 2 - k)) >> 1);
            symbol = qn + 1 - k;
        }
        self.update(scale, low, low + symbol, total);
        k
    }

    pub fn dec_laplace(&mut self, symbol: u32, decay: i32) -> i32 {
        let mut symbol = symbol;
        let mut value: i32 = 0;
        let mut low = 0u32;
        let scale = self.range >> 15;
        let center = self.value / scale + 1;
        let center = (1u32 << 15) - center.min(1 << 15);
        if center >= symbol {
            value += 1;
            low = symbol;
            symbol = 1 + (((32768 - 32 - symbol) as i32 * (16384 - decay)) >> 15) as u32;
            while symbol > 1 && center >= low + 2 * symbol {
                value += 1;
                symbol *= 2;
                low += symbol;
                symbol = ((((symbol - 2) as i32) * decay) >> 15) as u32 + 1;
            }
            if symbol <= 1 {
                let distance = (center - low) >> 1;
                value += distance as i32;
                low += 2 * distance;
            }
            if center < low + symbol {
                value = -value;
            } else {
                low += symbol;
            }
        }
        self.update(scale, low, (low + symbol).min(32768), 32768);
        value
    }
}

/// `ff_sqrt`: the integer square root (floor).
pub fn ff_sqrt(a: u32) -> u32 {
    let mut r = (a as f64).sqrt() as u32;
    while r * r > a {
        r -= 1;
    }
    while (r + 1) * (r + 1) <= a {
        r += 1;
    }
    r
}
