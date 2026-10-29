//! FLAC decoding (native `.flac` streams and FLAC-in-Ogg via
//! [`super::ogg`]): STREAMINFO, frames with constant / verbatim / fixed /
//! LPC subframes, wasted bits, Rice and escaped residuals, and the three
//! stereo decorrelations, with the frame CRC-8 / CRC-16 checked. Lossless,
//! so the integer samples are the encoder's exactly.

// Ported loop structure is kept as in the C reference.
#![allow(clippy::needless_range_loop, clippy::manual_memcpy, clippy::unnecessary_cast, clippy::manual_is_multiple_of)]

/// The `STREAMINFO` metadata block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamInfo {
    /// Smallest block size in the stream, in samples.
    pub min_block: u16,
    /// Largest block size in the stream, in samples.
    pub max_block: u16,
    /// Samples per second.
    pub sample_rate: u32,
    /// Channel count, 1 to 8.
    pub channels: u8,
    /// Bits per sample, 4 to 32.
    pub bits_per_sample: u8,
    /// Samples per channel, or 0 when the encoder did not know.
    pub total_samples: u64,
}

/// Decoded FLAC: per-channel integer samples at `bits_per_sample`.
#[derive(Debug, Clone)]
pub struct Flac {
    /// The stream's `STREAMINFO`.
    pub info: StreamInfo,
    /// One `Vec` per channel.
    pub channels: Vec<Vec<i32>>,
}

impl Flac {
    /// `int / 2^(bits-1)` (libsndfile's float normalisation), per channel.
    pub fn to_f32(&self) -> Vec<Vec<f32>> {
        let s = (1u64 << (self.info.bits_per_sample - 1)) as f64;
        self.channels.iter().map(|c| c.iter().map(|&v| (v as f64 / s) as f32).collect()).collect()
    }
}

struct Bits<'a> {
    d: &'a [u8],
    pos: usize,
}

impl<'a> Bits<'a> {
    fn new(d: &'a [u8]) -> Self {
        Self { d, pos: 0 }
    }
    fn bit(&mut self) -> Result<u32, String> {
        let b = *self.d.get(self.pos >> 3).ok_or("FLAC stream truncated")?;
        let v = (b >> (7 - (self.pos & 7))) & 1;
        self.pos += 1;
        Ok(v as u32)
    }
    fn bits(&mut self, n: u32) -> Result<u64, String> {
        let mut v = 0u64;
        let mut n = n;
        while n > 0 {
            if self.pos & 7 == 0 && n >= 8 {
                let b = *self.d.get(self.pos >> 3).ok_or("FLAC stream truncated")?;
                v = (v << 8) | b as u64;
                self.pos += 8;
                n -= 8;
            } else {
                v = (v << 1) | self.bit()? as u64;
                n -= 1;
            }
        }
        Ok(v)
    }
    fn signed(&mut self, n: u32) -> Result<i64, String> {
        if n == 0 {
            return Ok(0);
        }
        let v = self.bits(n)? as i64;
        Ok((v << (64 - n)) >> (64 - n))
    }
    fn unary(&mut self) -> Result<u32, String> {
        let mut n = 0;
        while self.bit()? == 0 {
            n += 1;
        }
        Ok(n)
    }
    fn align(&mut self) {
        self.pos = (self.pos + 7) & !7;
    }
    fn byte_pos(&self) -> usize {
        self.pos >> 3
    }
}

fn crc8(d: &[u8]) -> u8 {
    let mut c = 0u8;
    for &b in d {
        c ^= b;
        for _ in 0..8 {
            c = if c & 0x80 != 0 { (c << 1) ^ 0x07 } else { c << 1 };
        }
    }
    c
}

fn crc16(d: &[u8]) -> u16 {
    let mut c = 0u16;
    for &b in d {
        c ^= (b as u16) << 8;
        for _ in 0..8 {
            c = if c & 0x8000 != 0 { (c << 1) ^ 0x8005 } else { c << 1 };
        }
    }
    c
}

/// Parse a STREAMINFO block body (34 bytes).
pub fn parse_streaminfo(b: &[u8]) -> Result<StreamInfo, String> {
    if b.len() < 34 {
        return Err("short STREAMINFO".into());
    }
    let mut r = Bits::new(b);
    let min_block = r.bits(16)? as u16;
    let max_block = r.bits(16)? as u16;
    r.bits(24)?;
    r.bits(24)?;
    let sample_rate = r.bits(20)? as u32;
    let channels = r.bits(3)? as u8 + 1;
    let bits_per_sample = r.bits(5)? as u8 + 1;
    let total_samples = r.bits(36)?;
    Ok(StreamInfo { min_block, max_block, sample_rate, channels, bits_per_sample, total_samples })
}

/// Decode one frame starting at `d[0]`; returns its samples per channel and
/// its length in bytes.
pub fn decode_frame(d: &[u8], info: &StreamInfo) -> Result<(Vec<Vec<i32>>, usize), String> {
    let mut r = Bits::new(d);
    if r.bits(14)? != 0x3ffe {
        return Err("FLAC frame sync not found".into());
    }
    r.bits(1)?;
    let _variable = r.bit()?;
    let bs_code = r.bits(4)? as u32;
    let sr_code = r.bits(4)? as u32;
    let ch_code = r.bits(4)? as u32;
    let ss_code = r.bits(3)? as u32;
    r.bits(1)?;
    // UTF-8-style coded frame/sample number.
    let first = r.bits(8)? as u32;
    let extra = (!first as u8).leading_zeros().saturating_sub(1) as u32;
    let extra = if first & 0x80 == 0 { 0 } else { extra };
    for _ in 0..extra {
        r.bits(8)?;
    }
    let block = match bs_code {
        1 => 192,
        2..=5 => 576 << (bs_code - 2),
        6 => r.bits(8)? as u32 + 1,
        7 => r.bits(16)? as u32 + 1,
        8..=15 => 256 << (bs_code - 8),
        _ => return Err("reserved FLAC block size".into()),
    } as usize;
    match sr_code {
        12 => {
            r.bits(8)?;
        }
        13 | 14 => {
            r.bits(16)?;
        }
        15 => return Err("invalid FLAC sample rate code".into()),
        _ => {}
    }
    let hdr_end = r.byte_pos();
    let crc = r.bits(8)? as u8;
    if crc8(&d[..hdr_end]) != crc {
        return Err("FLAC frame header CRC mismatch".into());
    }
    let bps = match ss_code {
        0 => info.bits_per_sample as u32,
        1 => 8,
        2 => 12,
        4 => 16,
        5 => 20,
        6 => 24,
        7 => 32,
        _ => return Err("reserved FLAC sample size".into()),
    };
    let (n_ch, decor) = match ch_code {
        0..=7 => (ch_code as usize + 1, 0),
        8..=10 => (2, ch_code),
        _ => return Err("reserved FLAC channel assignment".into()),
    };
    let mut chans = Vec::with_capacity(n_ch);
    for c in 0..n_ch {
        // The side channel carries one extra bit.
        let side = (decor == 8 && c == 1) || (decor == 9 && c == 0) || (decor == 10 && c == 1);
        chans.push(subframe(&mut r, block, bps + side as u32)?);
    }
    r.align();
    let end = r.byte_pos();
    let crc = r.bits(16)? as u16;
    if crc16(&d[..end]) != crc {
        return Err("FLAC frame CRC mismatch".into());
    }
    let out: Vec<Vec<i32>> = match decor {
        8 => {
            // left/side
            let (l, s) = (&chans[0], &chans[1]);
            let r: Vec<i64> = l.iter().zip(s).map(|(&l, &s)| l - s).collect();
            vec![to32(l), to32(&r)]
        }
        9 => {
            // side/right
            let (s, rr) = (&chans[0], &chans[1]);
            let l: Vec<i64> = s.iter().zip(rr).map(|(&s, &r)| s + r).collect();
            vec![to32(&l), to32(rr)]
        }
        10 => {
            // mid/side
            let (m, s) = (&chans[0], &chans[1]);
            let mut l = Vec::with_capacity(block);
            let mut rr = Vec::with_capacity(block);
            for (&m, &s) in m.iter().zip(s) {
                let m = (m << 1) | (s & 1);
                l.push((m + s) >> 1);
                rr.push((m - s) >> 1);
            }
            vec![to32(&l), to32(&rr)]
        }
        _ => chans.iter().map(|c| to32(c)).collect(),
    };
    Ok((out, end + 2))
}

fn to32(v: &[i64]) -> Vec<i32> {
    v.iter().map(|&x| x as i32).collect()
}

fn subframe(r: &mut Bits, block: usize, bps: u32) -> Result<Vec<i64>, String> {
    if r.bit()? != 0 {
        return Err("FLAC subframe padding bit set".into());
    }
    let kind = r.bits(6)? as u32;
    let mut wasted = 0;
    if r.bit()? == 1 {
        wasted = r.unary()? + 1;
    }
    let bps = bps - wasted;
    let mut s = match kind {
        0 => vec![r.signed(bps)?; block],
        1 => (0..block).map(|_| r.signed(bps)).collect::<Result<_, _>>()?,
        8..=12 => {
            let order = (kind - 8) as usize;
            let mut s: Vec<i64> = (0..order).map(|_| r.signed(bps)).collect::<Result<_, _>>()?;
            residual(r, block, order, &mut s)?;
            for i in order..block {
                let p = match order {
                    0 => 0,
                    1 => s[i - 1],
                    2 => 2 * s[i - 1] - s[i - 2],
                    3 => 3 * s[i - 1] - 3 * s[i - 2] + s[i - 3],
                    _ => 4 * s[i - 1] - 6 * s[i - 2] + 4 * s[i - 3] - s[i - 4],
                };
                s[i] += p;
            }
            s
        }
        32..=63 => {
            let order = (kind - 31) as usize;
            let mut s: Vec<i64> = (0..order).map(|_| r.signed(bps)).collect::<Result<_, _>>()?;
            let prec = r.bits(4)? as u32 + 1;
            if prec == 16 {
                return Err("invalid FLAC LPC precision".into());
            }
            let shift = r.signed(5)?;
            if shift < 0 {
                return Err("negative FLAC LPC shift".into());
            }
            let coefs: Vec<i64> = (0..order).map(|_| r.signed(prec)).collect::<Result<_, _>>()?;
            residual(r, block, order, &mut s)?;
            for i in order..block {
                let mut acc: i64 = 0;
                for (j, &c) in coefs.iter().enumerate() {
                    acc += c * s[i - 1 - j];
                }
                s[i] += acc >> shift;
            }
            s
        }
        _ => return Err(format!("reserved FLAC subframe type {kind}")),
    };
    if wasted > 0 {
        for v in &mut s {
            *v <<= wasted;
        }
    }
    Ok(s)
}

/// Rice-coded residual; appends `block - order` values to `s`.
fn residual(r: &mut Bits, block: usize, order: usize, s: &mut Vec<i64>) -> Result<(), String> {
    let method = r.bits(2)?;
    let param_bits = match method {
        0 => 4,
        1 => 5,
        _ => return Err("reserved FLAC residual coding".into()),
    };
    let escape = (1u64 << param_bits) - 1;
    let part_order = r.bits(4)? as u32;
    let parts = 1usize << part_order;
    if block % parts != 0 || block / parts < order && part_order > 0 {
        return Err("bad FLAC partition order".into());
    }
    for p in 0..parts {
        let n = if p == 0 { block / parts - order } else { block / parts };
        let k = r.bits(param_bits)?;
        if k == escape {
            let nb = r.bits(5)? as u32;
            for _ in 0..n {
                s.push(r.signed(nb)?);
            }
        } else {
            let k = k as u32;
            for _ in 0..n {
                let q = r.unary()? as u64;
                let v = (q << k) | r.bits(k)?;
                s.push(((v >> 1) as i64) ^ -((v & 1) as i64));
            }
        }
    }
    Ok(())
}

/// Decode a native FLAC file (`fLaC` + metadata + frames). ID3v2 tags in
/// front of the stream are skipped.
pub fn decode(data: &[u8]) -> Result<Flac, String> {
    let mut off = 0;
    if data.starts_with(b"ID3") && data.len() > 10 {
        let sz = data[6..10].iter().fold(0usize, |a, &b| (a << 7) | (b & 0x7f) as usize);
        off = 10 + sz;
    }
    if data.get(off..off + 4) != Some(b"fLaC".as_slice()) {
        return Err("not a FLAC file".into());
    }
    off += 4;
    let mut info = None;
    loop {
        let h = *data.get(off).ok_or("truncated FLAC metadata")?;
        let len = u32::from_be_bytes([0, data[off + 1], data[off + 2], data[off + 3]]) as usize;
        let body = data.get(off + 4..off + 4 + len).ok_or("truncated FLAC metadata")?;
        if h & 0x7f == 0 {
            info = Some(parse_streaminfo(body)?);
        }
        off += 4 + len;
        if h & 0x80 != 0 {
            break;
        }
    }
    let info = info.ok_or("FLAC without STREAMINFO")?;
    let mut channels = vec![Vec::new(); info.channels as usize];
    decode_frames(&data[off..], &info, &mut channels)?;
    Ok(Flac { info, channels })
}

/// Decode consecutive frames from `d` into `channels`.
pub fn decode_frames(d: &[u8], info: &StreamInfo, channels: &mut [Vec<i32>]) -> Result<(), String> {
    let mut off = 0;
    while off + 2 <= d.len() {
        let (frame, len) = decode_frame(&d[off..], info)?;
        if frame.len() != channels.len() {
            return Err("FLAC frame channel count changed".into());
        }
        for (c, f) in channels.iter_mut().zip(frame) {
            c.extend(f);
        }
        off += len;
    }
    if info.total_samples > 0 {
        for c in channels.iter_mut() {
            c.truncate(info.total_samples as usize);
        }
    }
    Ok(())
}
