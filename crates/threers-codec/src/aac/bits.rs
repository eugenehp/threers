//! MSB-first bit reader with libavcodec `GetBitContext` semantics: reads past
//! the end return zeros (ffmpeg pads every packet with zeroed bytes), and
//! [`BitReader::left`] goes negative on overread.

#[derive(Clone)]
pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn left(&self) -> isize {
        (self.data.len() * 8) as isize - self.pos as isize
    }

    /// The next 32 bits, MSB-aligned (`GET_CACHE`).
    #[inline]
    pub fn peek32(&self) -> u32 {
        let byte = self.pos >> 3;
        let mut v: u64 = 0;
        for i in 0..5 {
            v = (v << 8) | *self.data.get(byte + i).unwrap_or(&0) as u64;
        }
        ((v << (self.pos & 7)) >> 8) as u32
    }

    #[inline]
    pub fn show(&self, n: u32) -> u32 {
        if n == 0 {
            0
        } else {
            self.peek32() >> (32 - n)
        }
    }

    #[inline]
    pub fn skip(&mut self, n: u32) {
        self.pos += n as usize;
    }

    #[inline]
    pub fn bits(&mut self, n: u32) -> u32 {
        let v = self.show(n);
        self.skip(n);
        v
    }

    #[inline]
    pub fn bit(&mut self) -> bool {
        self.bits(1) != 0
    }

    /// Bits beyond 32 are read in pieces (`skip_bits_long`).
    pub fn skip_long(&mut self, n: isize) {
        if n > 0 {
            self.pos += n as usize;
        }
    }

    pub fn align(&mut self) {
        self.pos = (self.pos + 7) & !7;
    }
}
