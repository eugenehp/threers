//! The block compressors of OpenEXR beyond ZIP: RLE, PIZ, PXR24 and B44/B44A,
//! and through [`exr_dwa`](super::exr_dwa) DWAA and DWAB.
//!
//! Each turns one chunk into the layout an uncompressed file stores: for every
//! scanline, every channel in name order, little-endian. Bad input returns
//! `None`, and nothing here indexes on a value read from the file unchecked.

use super::deflate::inflate_zlib;
use super::exr::exr_reorder;

const HALF: u8 = 1;

/// What a compressor needs to know of a channel.
pub(super) struct Channel<'a> {
    pub name: &'a str,
    pub pixel_type: u8,
    /// Marked perceptually linear in the channel list.
    pub p_linear: bool,
}

/// 16-bit words in one sample: the wavelet and B44 both work on these.
fn words(pixel_type: u8) -> usize {
    if pixel_type == HALF {
        1
    } else {
        2
    }
}

/// Decompress one chunk of `lines` scanlines, `width` samples per channel.
/// `channels` are in name order.
pub(super) fn decompress(
    compression: u8,
    data: &[u8],
    channels: &[Channel],
    width: usize,
    lines: usize,
) -> Option<Vec<u8>> {
    let types: Vec<u8> = channels.iter().map(|c| c.pixel_type).collect();
    let types = &types[..];
    let size = types.iter().map(|&t| words(t) * 2 * width).sum::<usize>() * lines;
    let out = match compression {
        1 => exr_reorder(&rle(data, size)?),
        2 | 3 => exr_reorder(&inflate_zlib(data).ok()?),
        4 => piz(data, types, width, lines, size)?,
        5 => pxr24(data, types, width, lines, size)?,
        6 | 7 => b44(data, channels, width, lines)?,
        8 | 9 => super::exr_dwa::decompress(data, channels, width, lines)?,
        _ => return None,
    };
    Some(out)
}

/// A negative count copies that many literals; any other repeats the next
/// byte count + 1 times.
pub(super) fn rle(data: &[u8], size: usize) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(size);
    let mut i = 0;
    while i < data.len() {
        let count = data[i] as i8;
        i += 1;
        if count < 0 {
            let n = -(count as i32) as usize;
            out.extend_from_slice(data.get(i..i + n)?);
            i += n;
        } else {
            out.resize(out.len() + count as usize + 1, *data.get(i)?);
            i += 1;
        }
        if out.len() > size {
            return None;
        }
    }
    Some(out)
}

/// Each channel's samples are stored one byte plane at a time per scanline,
/// delta-coded across the row; FLOAT keeps only its top 24 bits.
fn pxr24(data: &[u8], types: &[u8], width: usize, lines: usize, size: usize) -> Option<Vec<u8>> {
    let packed = inflate_zlib(data).ok()?;
    let mut out = Vec::with_capacity(size);
    let mut pos = 0;
    for _ in 0..lines {
        for &t in types {
            let planes = match t {
                HALF => 2,
                2 => 3,
                _ => 4,
            };
            let row = packed.get(pos..pos + planes * width)?;
            pos += planes * width;
            let mut pixel = 0u32;
            for x in 0..width {
                let mut diff = 0u32;
                for p in 0..planes {
                    diff = diff << 8 | row[p * width + x] as u32;
                }
                if t == 2 {
                    diff <<= 8;
                }
                pixel = pixel.wrapping_add(diff);
                if t == HALF {
                    out.extend_from_slice(&(pixel as u16).to_le_bytes());
                } else {
                    out.extend_from_slice(&pixel.to_le_bytes());
                }
            }
        }
    }
    Some(out)
}

/// Channel planes, one after another, back into scanlines.
fn planes_to_scanlines(tmp: &[u16], types: &[u8], width: usize, lines: usize) -> Vec<u8> {
    let mut starts = Vec::with_capacity(types.len());
    let mut at = 0;
    for &t in types {
        starts.push(at);
        at += words(t) * width * lines;
    }
    let mut out = Vec::with_capacity(tmp.len() * 2);
    for y in 0..lines {
        for (&t, &start) in types.iter().zip(&starts) {
            let n = words(t) * width;
            for w in &tmp[start + y * n..start + (y + 1) * n] {
                out.extend_from_slice(&w.to_le_bytes());
            }
        }
    }
    out
}

fn piz(data: &[u8], types: &[u8], width: usize, lines: usize, size: usize) -> Option<Vec<u8>> {
    const BITMAP_SIZE: usize = 8192;
    let u16_at = |i: usize| Some(u16::from_le_bytes(data.get(i..i + 2)?.try_into().ok()?));
    let (min, max) = (u16_at(0)? as usize, u16_at(2)? as usize);
    let mut pos = 4;
    let mut bitmap = vec![0u8; BITMAP_SIZE];
    if min <= max {
        if max >= BITMAP_SIZE {
            return None;
        }
        let n = max - min + 1;
        bitmap[min..=max].copy_from_slice(data.get(pos..pos + n)?);
        pos += n;
    }

    // The values present, in order: the wavelet ran on their indices.
    let mut lut = vec![0u16; 1 << 16];
    let mut k = 0usize;
    for i in 0..1usize << 16 {
        if i == 0 || bitmap[i >> 3] & (1 << (i & 7)) != 0 {
            lut[k] = i as u16;
            k += 1;
        }
    }
    let max_value = k - 1;

    let length = u32::from_le_bytes(data.get(pos..pos + 4)?.try_into().ok()?) as usize;
    pos += 4;
    let mut tmp = huf_uncompress(data.get(pos..pos.checked_add(length)?)?, size / 2)?;

    let mut start = 0;
    for &t in types {
        let w = words(t);
        for j in 0..w {
            wav2_decode(&mut tmp, start + j, width, w, lines, width * w, max_value);
        }
        start += w * width * lines;
    }
    for v in &mut tmp {
        *v = lut[*v as usize];
    }
    Some(planes_to_scanlines(&tmp, types, width, lines))
}

fn wdec14(l: u16, h: u16) -> (u16, u16) {
    let (ls, hi) = (l as i16 as i32, h as i16 as i32);
    let ai = ls + (hi & 1) + (hi >> 1);
    (ai as u16, (ai - hi) as u16)
}

fn wdec16(l: u16, h: u16) -> (u16, u16) {
    const A_OFFSET: i32 = 1 << 15;
    let (m, d) = (l as i32, h as i32);
    let bb = (m - (d >> 1)) & 0xffff;
    let aa = (d + bb - A_OFFSET) & 0xffff;
    (aa as u16, bb as u16)
}

/// Undo the 2D Haar pyramid over `nx` by `ny` samples `ox` and `oy` apart.
fn wav2_decode(b: &mut [u16], at: usize, nx: usize, ox: usize, ny: usize, oy: usize, mx: usize) {
    if mx < 1 << 14 {
        wav2(b, at, nx, ox, ny, oy, wdec14);
    } else {
        wav2(b, at, nx, ox, ny, oy, wdec16);
    }
}

fn wav2(
    b: &mut [u16],
    at: usize,
    nx: usize,
    ox: usize,
    ny: usize,
    oy: usize,
    wdec: impl Fn(u16, u16) -> (u16, u16),
) {
    let n = nx.min(ny);
    if n == 0 {
        return;
    }
    let mut p = 1;
    while p <= n {
        p <<= 1;
    }
    p >>= 1;
    let mut p2 = p;
    p >>= 1;
    while p >= 1 {
        let (oy1, oy2, ox1, ox2) = (oy * p, oy * p2, ox * p, ox * p2);
        let ey = at + oy * (ny - p2);
        let mut py = at;
        while py <= ey {
            let ex = py + ox * (nx - p2);
            let mut px = py;
            while px <= ex {
                let (p01, p10) = (px + ox1, px + oy1);
                let p11 = p10 + ox1;
                let (i00, i10) = wdec(b[px], b[p10]);
                let (i01, i11) = wdec(b[p01], b[p11]);
                (b[px], b[p01]) = wdec(i00, i01);
                (b[p10], b[p11]) = wdec(i10, i11);
                px += ox2;
            }
            if nx & p != 0 {
                let p10 = px + oy1;
                (b[px], b[p10]) = wdec(b[px], b[p10]);
            }
            py += oy2;
        }
        if ny & p != 0 {
            let ex = py + ox * (nx - p2);
            let mut px = py;
            while px <= ex {
                let p01 = px + ox1;
                (b[px], b[p01]) = wdec(b[px], b[p01]);
                px += ox2;
            }
        }
        p2 = p;
        p >>= 1;
    }
}

const HUF_ENCSIZE: usize = (1 << 16) + 1;
const HUF_DECBITS: u32 = 14;
const HUF_DECSIZE: usize = 1 << HUF_DECBITS;
const SHORT_ZEROCODE_RUN: u64 = 59;
const LONG_ZEROCODE_RUN: u64 = 63;
const SHORTEST_LONG_RUN: usize = 2 + (LONG_ZEROCODE_RUN - SHORT_ZEROCODE_RUN) as usize;

/// Bits taken from a byte slice, most significant first.
struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    acc: u64,
    count: u32,
}

impl Bits<'_> {
    fn fill(&mut self) -> Option<()> {
        self.acc = self.acc << 8 | *self.data.get(self.pos)? as u64;
        self.pos += 1;
        self.count += 8;
        Some(())
    }

    fn take(&mut self, n: u32) -> Option<u64> {
        while self.count < n {
            self.fill()?;
        }
        self.count -= n;
        Some(self.acc >> self.count & ((1 << n) - 1))
    }
}

/// A code's bits and length, as `code << 6 | length`.
fn code_bits(code: u64) -> u64 {
    code >> 6
}

fn code_len(code: u64) -> u32 {
    (code & 63) as u32
}

/// Read the run-length-coded code lengths for `min..=max` and give each symbol
/// its canonical code. Returns the table and the bytes it took.
fn huf_table(data: &[u8], min: usize, max: usize) -> Option<(Vec<u64>, usize)> {
    let mut codes = vec![0u64; HUF_ENCSIZE];
    let mut bits = Bits {
        data,
        pos: 0,
        acc: 0,
        count: 0,
    };
    let mut i = min;
    while i <= max {
        let l = bits.take(6)?;
        let run = if l == LONG_ZEROCODE_RUN {
            bits.take(8)? as usize + SHORTEST_LONG_RUN
        } else if l >= SHORT_ZEROCODE_RUN {
            (l - SHORT_ZEROCODE_RUN) as usize + 2
        } else {
            codes[i] = l;
            i += 1;
            continue;
        };
        if i + run > max + 1 {
            return None;
        }
        i += run;
    }

    let mut first = [0u64; 59];
    for &l in &codes {
        first[l as usize] += 1;
    }
    let mut c = 0;
    for l in (1..59).rev() {
        let next = (c + first[l]) >> 1;
        first[l] = c;
        c = next;
    }
    for code in &mut codes {
        let l = *code as usize;
        if l > 0 {
            *code = l as u64 | first[l] << 6;
            first[l] += 1;
        }
    }
    Some((codes, bits.pos))
}

/// A slot for the next 14 bits: a short code and its symbol, or the symbols
/// of the longer codes that begin with those bits.
#[derive(Clone, Default)]
struct DecEntry {
    len: u32,
    symbol: u32,
    long: Vec<u32>,
}

fn huf_decoder(codes: &[u64], min: usize, max: usize) -> Option<Vec<DecEntry>> {
    let mut table = vec![DecEntry::default(); HUF_DECSIZE];
    for (i, &code) in codes.iter().enumerate().take(max + 1).skip(min) {
        let (c, l) = (code_bits(code), code_len(code));
        if c >> l != 0 {
            return None;
        }
        if l > HUF_DECBITS {
            let slot = &mut table[(c >> (l - HUF_DECBITS)) as usize];
            if slot.len != 0 {
                return None;
            }
            slot.long.push(i as u32);
        } else if l > 0 {
            let first = (c << (HUF_DECBITS - l)) as usize;
            for slot in &mut table[first..first + (1 << (HUF_DECBITS - l))] {
                if slot.len != 0 || !slot.long.is_empty() {
                    return None;
                }
                slot.len = l;
                slot.symbol = i as u32;
            }
        }
    }
    Some(table)
}

/// Emit a symbol. The run symbol repeats the last value by a count in the next
/// eight bits.
fn huf_emit(symbol: u32, run: usize, bits: &mut Bits, out: &mut Vec<u16>, n: usize) -> Option<()> {
    if symbol as usize == run {
        if bits.count < 8 {
            bits.fill()?;
        }
        bits.count -= 8;
        let count = (bits.acc >> bits.count) as u8 as usize;
        let last = *out.last()?;
        if out.len() + count > n {
            return None;
        }
        out.resize(out.len() + count, last);
    } else {
        if out.len() >= n {
            return None;
        }
        out.push(symbol as u16);
    }
    Some(())
}

pub(super) fn huf_uncompress(data: &[u8], n: usize) -> Option<Vec<u16>> {
    if data.is_empty() {
        return (n == 0).then(Vec::new);
    }
    let u32_at = |i: usize| Some(u32::from_le_bytes(data.get(i..i + 4)?.try_into().ok()?) as usize);
    let (min, max, n_bits) = (u32_at(0)?, u32_at(4)?, u32_at(12)?);
    if min >= HUF_ENCSIZE || max >= HUF_ENCSIZE {
        return None;
    }
    let body = data.get(20..)?;
    let (codes, used) = huf_table(body, min, max)?;
    let table = huf_decoder(&codes, min, max)?;
    let stream = body.get(used..)?;
    let bytes = n_bits.div_ceil(8);
    if bytes > stream.len() {
        return None;
    }

    let mut bits = Bits {
        data: &stream[..bytes],
        pos: 0,
        acc: 0,
        count: 0,
    };
    let mut out = Vec::with_capacity(n);
    while bits.pos < bytes {
        bits.fill()?;
        while bits.count >= HUF_DECBITS {
            let entry =
                &table[(bits.acc >> (bits.count - HUF_DECBITS)) as usize & (HUF_DECSIZE - 1)];
            if entry.len != 0 {
                bits.count -= entry.len;
                huf_emit(entry.symbol, max, &mut bits, &mut out, n)?;
                continue;
            }
            let mut found = false;
            for &symbol in &entry.long {
                let code = codes[symbol as usize];
                let l = code_len(code);
                while bits.count < l && bits.pos < bytes {
                    bits.fill()?;
                }
                if bits.count >= l
                    && code_bits(code) == (bits.acc >> (bits.count - l)) & ((1 << l) - 1)
                {
                    bits.count -= l;
                    huf_emit(symbol, max, &mut bits, &mut out, n)?;
                    found = true;
                    break;
                }
            }
            if !found {
                return None;
            }
        }
    }

    // The last byte holds padding, and fewer than 14 bits may be left.
    let pad = 8u32.wrapping_sub(n_bits as u32) & 7;
    bits.count = bits.count.checked_sub(pad)?;
    bits.acc >>= pad;
    while bits.count > 0 {
        let entry = &table[(bits.acc << (HUF_DECBITS - bits.count)) as usize & (HUF_DECSIZE - 1)];
        if entry.len == 0 || entry.len > bits.count {
            return None;
        }
        bits.count -= entry.len;
        huf_emit(entry.symbol, max, &mut bits, &mut out, n)?;
    }
    (out.len() == n).then_some(out)
}

/// A 4x4 block of half floats from its 14-byte form: a first sample, then
/// six-bit differences down the first column and along each row.
fn b44_unpack14(b: &[u8]) -> [u16; 16] {
    let b = |i: usize| b[i] as u32;
    let shift = b(2) >> 2;
    let bias = 0x20u32 << shift;
    let d = |v: u32| (v & 0x3f) << shift;
    let mut s = [0u32; 16];
    let step = |from: u32, v: u32| from.wrapping_add(d(v)).wrapping_sub(bias) & 0xffff;
    s[0] = b(0) << 8 | b(1);
    s[4] = step(s[0], b(2) << 4 | b(3) >> 4);
    s[8] = step(s[4], b(3) << 2 | b(4) >> 6);
    s[12] = step(s[8], b(4));
    s[1] = step(s[0], b(5) >> 2);
    s[5] = step(s[4], b(5) << 4 | b(6) >> 4);
    s[9] = step(s[8], b(6) << 2 | b(7) >> 6);
    s[13] = step(s[12], b(7));
    s[2] = step(s[1], b(8) >> 2);
    s[6] = step(s[5], b(8) << 4 | b(9) >> 4);
    s[10] = step(s[9], b(9) << 2 | b(10) >> 6);
    s[14] = step(s[13], b(10));
    s[3] = step(s[2], b(11) >> 2);
    s[7] = step(s[6], b(11) << 4 | b(12) >> 4);
    s[11] = step(s[10], b(12) << 2 | b(13) >> 6);
    s[15] = step(s[14], b(13));
    s.map(|v| b44_order(v as u16))
}

/// Blocks hold halves mapped so that their integer order is their numeric one.
fn b44_order(v: u16) -> u16 {
    if v & 0x8000 != 0 {
        v & 0x7fff
    } else {
        !v
    }
}

fn b44(data: &[u8], channels: &[Channel], width: usize, lines: usize) -> Option<Vec<u8>> {
    let types: Vec<u8> = channels.iter().map(|c| c.pixel_type).collect();
    let types = &types[..];
    let total: usize = types.iter().map(|&t| words(t) * width * lines).sum();
    let mut tmp = vec![0u16; total];
    let mut pos = 0;
    let mut start = 0;
    for channel in channels {
        let t = channel.pixel_type;
        if t != HALF {
            // B44 only compresses HALF: other channels sit there as they are.
            let n = words(t) * width * lines;
            let raw = data.get(pos..pos + n * 2)?;
            for (slot, w) in tmp[start..start + n].iter_mut().zip(raw.as_chunks::<2>().0) {
                *slot = u16::from_le_bytes([w[0], w[1]]);
            }
            pos += n * 2;
            start += n;
            continue;
        }
        for y in (0..lines).step_by(4) {
            for x in (0..width).step_by(4) {
                let flat = *data.get(pos + 2)? >= 13 << 2;
                let mut s = if flat {
                    let first = data.get(pos..pos + 3)?;
                    pos += 3;
                    [b44_order((first[0] as u16) << 8 | first[1] as u16); 16]
                } else {
                    let block = data.get(pos..pos + 14)?;
                    pos += 14;
                    b44_unpack14(block)
                };
                if channel.p_linear {
                    let table = super::exr_dwa::b44_to_linear();
                    s = s.map(|v| table[v as usize]);
                }
                for r in 0..4.min(lines - y) {
                    for c in 0..4.min(width - x) {
                        tmp[start + (y + r) * width + x + c] = s[r * 4 + c];
                    }
                }
            }
        }
        start += width * lines;
    }
    Some(planes_to_scanlines(&tmp, types, width, lines))
}
