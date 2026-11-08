//! OpenEXR's DWAA and DWAB compressors: DreamWorks' lossy 8x8 DCT.
//!
//! A chunk opens with eleven little-endian `u64` counters and, from version 2,
//! the rules that say how each channel was coded by the suffix of its name.
//! Four sections follow: channels of no known kind (deflated), AC coefficients
//! (Huffman or deflate), DC coefficients (deflate, with the same byte transform
//! as ZIP) and run-length channels. RGB triples of one prefix are coded as
//! Y'CbCr, and coefficients are in a non-linear space, so a decoded block goes
//! through an inverse DCT, the inverse colour matrix and the
//! [`to_linear`] curve, in that order.

use std::sync::OnceLock;

use super::deflate::inflate_zlib;
use super::exr::exr_reorder;
use super::exr_codec::{huf_uncompress, rle, Channel};

const UNKNOWN: u8 = 0;
const LOSSY_DCT: u8 = 1;
const RLE: u8 = 2;

/// Raster position `i` of a block reads zigzag entry `DWA_ZIGZAG[i]`.
const DWA_ZIGZAG: [u8; 64] = [
    0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42, 3, 8, 12, 17, 25, 30, 41, 43, 9, 11,
    18, 24, 31, 40, 44, 53, 10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60, 21, 34,
    37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63,
];

/// The channel rules of a version 1 stream, which carries none of its own:
/// suffix (matched ignoring case), scheme, pixel type, colour index.
const DWA_LEGACY_RULES: &[(&str, u8, u8, i8)] = &[
    ("r", 1, 1, 0),
    ("r", 1, 2, 0),
    ("red", 1, 1, 0),
    ("red", 1, 2, 0),
    ("g", 1, 1, 1),
    ("g", 1, 2, 1),
    ("grn", 1, 1, 1),
    ("grn", 1, 2, 1),
    ("green", 1, 1, 1),
    ("green", 1, 2, 1),
    ("b", 1, 1, 2),
    ("b", 1, 2, 2),
    ("blu", 1, 1, 2),
    ("blu", 1, 2, 2),
    ("blue", 1, 1, 2),
    ("blue", 1, 2, 2),
    ("y", 1, 1, -1),
    ("y", 1, 2, -1),
    ("by", 1, 1, -1),
    ("by", 1, 2, -1),
    ("ry", 1, 1, -1),
    ("ry", 1, 2, -1),
    ("a", 2, 0, -1),
    ("a", 2, 1, -1),
    ("a", 2, 2, -1),
];

/// A channel rule: the suffix it matches, how such a channel is coded, the
/// pixel type it applies to and its place in an RGB triple (or -1).
struct Rule {
    suffix: String,
    scheme: u8,
    pixel_type: u8,
    csc: i8,
    ignore_case: bool,
}

impl Rule {
    fn matches(&self, suffix: &str, pixel_type: u8) -> bool {
        self.pixel_type == pixel_type
            && if self.ignore_case {
                self.suffix.eq_ignore_ascii_case(suffix)
            } else {
                self.suffix == suffix
            }
    }
}

fn legacy_rules() -> Vec<Rule> {
    DWA_LEGACY_RULES
        .iter()
        .map(|&(suffix, scheme, pixel_type, csc)| Rule {
            suffix: suffix.to_string(),
            scheme,
            pixel_type,
            csc,
            ignore_case: true,
        })
        .collect()
}

/// The rules at the start of `data`, and the bytes they took.
fn read_rules(data: &[u8]) -> Option<(Vec<Rule>, usize)> {
    let size = u16::from_le_bytes(data.get(..2)?.try_into().ok()?) as usize;
    if size < 2 {
        return None;
    }
    let mut body = data.get(2..size)?;
    let mut rules = Vec::new();
    while !body.is_empty() {
        let end = body.iter().position(|&b| b == 0)?;
        if end > 128 {
            return None;
        }
        let suffix = String::from_utf8_lossy(&body[..end]).into_owned();
        let (&value, &pixel_type) = (body.get(end + 1)?, body.get(end + 2)?);
        let csc = (value >> 4) as i8 - 1;
        let scheme = (value >> 2) & 3;
        if csc >= 3 || scheme > RLE || pixel_type > 2 {
            return None;
        }
        rules.push(Rule {
            suffix,
            scheme,
            pixel_type,
            csc,
            ignore_case: value & 1 != 0,
        });
        body = &body[end + 3..];
    }
    Some((rules, size))
}

/// What each channel's rules say: its scheme, and the RGB triples found among
/// channels sharing a prefix.
fn classify(channels: &[Channel], rules: &[Rule]) -> (Vec<u8>, Vec<[usize; 3]>) {
    let mut schemes = vec![UNKNOWN; channels.len()];
    let mut prefixes: Vec<(&str, [Option<usize>; 3])> = Vec::new();
    for (c, chan) in channels.iter().enumerate() {
        let cut = chan.name.rfind('.').map_or(0, |i| i + 1);
        let (prefix, suffix) = chan.name.split_at(cut);
        let at = match prefixes.iter().position(|p| p.0 == prefix) {
            Some(at) => at,
            None => {
                prefixes.push((prefix, [None; 3]));
                prefixes.len() - 1
            }
        };
        for rule in rules.iter().filter(|r| r.matches(suffix, chan.pixel_type)) {
            schemes[c] = rule.scheme;
            if rule.csc >= 0 {
                prefixes[at].1[rule.csc as usize] = Some(c);
            }
        }
    }
    let sets = prefixes
        .iter()
        .filter_map(|p| Some([p.1[0]?, p.1[1]?, p.1[2]?]))
        .collect();
    (schemes, sets)
}

/// Single precision to half, rounding to nearest even and keeping subnormals
/// and infinities, as the format's own arithmetic does.
fn half_bits(f: f32) -> u16 {
    crate::math::float::f32_to_f16(f)
}

fn half_float(h: u16) -> f32 {
    crate::renderer::gpu_texture::f16_bits_to_f32(h)
}

/// Decoded non-linear half to linear half: a gamma of 2.2 up to 1 and an
/// exponential above it, the two meeting smoothly there.
fn to_linear() -> &'static [u16] {
    static TABLE: OnceLock<Vec<u16>> = OnceLock::new();
    TABLE.get_or_init(|| {
        (0..=u16::MAX)
            .map(|x| {
                if x == 0 || x & 0x7c00 == 0x7c00 {
                    return 0;
                }
                let f = half_float(x);
                let sign = if f < 0.0 { -1.0f32 } else { 1.0 };
                let f = f.abs();
                let (px, py) = if f <= 1.0 {
                    (f, 2.2f32)
                } else {
                    (9.025_013f32, f - 1.0)
                };
                half_bits(sign * px.powf(py))
            })
            .collect()
    })
}

/// B44's table for a channel marked perceptually linear: blocks hold
/// `exp(x / 8)` of those, so decoding takes `8 ln(x)`.
pub(super) fn b44_to_linear() -> &'static [u16] {
    static TABLE: OnceLock<Vec<u16>> = OnceLock::new();
    TABLE.get_or_init(|| {
        (0..=u16::MAX)
            .map(|x| {
                if x & 0x7c00 == 0x7c00 || x > 0x8000 {
                    return 0;
                }
                half_bits(8.0 * half_float(x).ln())
            })
            .collect()
    })
}

/// The separable inverse DCT, in place.
fn inverse_dct(data: &mut [f32; 64]) {
    // The format's constants are built from this value of pi.
    #[allow(clippy::approx_constant)]
    const PI: f32 = 3.14159;
    let a = 0.5 * (PI / 4.0).cos();
    let b = 0.5 * (PI / 16.0).cos();
    let c = 0.5 * (PI / 8.0).cos();
    let d = 0.5 * (3.0 * PI / 16.0).cos();
    let e = 0.5 * (5.0 * PI / 16.0).cos();
    let f = 0.5 * (3.0 * PI / 8.0).cos();
    let g = 0.5 * (7.0 * PI / 16.0).cos();
    let line = |v: [f32; 8]| -> [f32; 8] {
        let alpha = [c * v[2], f * v[2], c * v[6], f * v[6]];
        let beta = [
            b * v[1] + d * v[3] + e * v[5] + g * v[7],
            d * v[1] - g * v[3] - b * v[5] - e * v[7],
            e * v[1] - b * v[3] + g * v[5] + d * v[7],
            g * v[1] - e * v[3] + d * v[5] - b * v[7],
        ];
        let theta = [
            a * (v[0] + v[4]),
            alpha[0] + alpha[3],
            alpha[1] - alpha[2],
            a * (v[0] - v[4]),
        ];
        let gamma = [
            theta[0] + theta[1],
            theta[3] + theta[2],
            theta[3] - theta[2],
            theta[0] - theta[1],
        ];
        [
            gamma[0] + beta[0],
            gamma[1] + beta[1],
            gamma[2] + beta[2],
            gamma[3] + beta[3],
            gamma[3] - beta[3],
            gamma[2] - beta[2],
            gamma[1] - beta[1],
            gamma[0] - beta[0],
        ]
    };
    for row in data.chunks_exact_mut(8) {
        let v: [f32; 8] = (&*row).try_into().unwrap();
        row.copy_from_slice(&line(v));
    }
    for x in 0..8 {
        let v = std::array::from_fn(|y| data[y * 8 + x]);
        for (y, o) in line(v).into_iter().enumerate() {
            data[y * 8 + x] = o;
        }
    }
}

/// One block's AC run: `0xff` in the high byte skips that many coefficients,
/// and a count of 0 ends the block. Returns the last one written.
fn un_rle_ac(ac: &[u16], at: &mut usize, block: &mut [u16; 64]) -> Option<usize> {
    let (mut comp, mut last) = (1, 0);
    while comp < 64 {
        let v = *ac.get(*at)?;
        *at += 1;
        if v & 0xff00 == 0xff00 {
            let count = (v & 0xff) as usize;
            comp += if count == 0 { 64 } else { count };
        } else {
            last = comp;
            block[comp] = v;
            comp += 1;
        }
    }
    Some(last)
}

/// Decode one DCT group, a Y'CbCr triple or a lone channel, into `planes`:
/// half bits, `width` by `lines` each.
#[allow(clippy::too_many_arguments)]
fn decode_dct(
    planes: &mut [&mut Vec<u16>],
    ac: &[u16],
    ac_at: &mut usize,
    dc: &[u16],
    dc_at: &mut usize,
    linear: Option<&[u16]>,
    width: usize,
    lines: usize,
) -> Option<()> {
    let comps = planes.len();
    let (bx, by) = (width.div_ceil(8), lines.div_ceil(8));
    let per = bx * by;
    let dc = dc.get(*dc_at..*dc_at + comps * per)?;
    *dc_at += comps * per;
    let mut blocks = [[0f32; 64]; 3];
    for y in 0..by {
        for x in 0..bx {
            for c in 0..comps {
                let mut zig = [0u16; 64];
                zig[0] = dc[c * per + y * bx + x];
                let last = un_rle_ac(ac, ac_at, &mut zig)?;
                let block = &mut blocks[c];
                if last == 0 {
                    *block = [half_float(zig[0]) * 3.535536e-01f32 * 3.535536e-01f32; 64];
                } else {
                    for (slot, &z) in block.iter_mut().zip(&DWA_ZIGZAG) {
                        *slot = half_float(zig[z as usize]);
                    }
                    inverse_dct(block);
                }
            }
            if comps == 3 {
                let [b0, b1, b2] = &mut blocks;
                for ((v0, v1), v2) in b0.iter_mut().zip(b1.iter_mut()).zip(b2.iter_mut()) {
                    let (s0, s1, s2) = (*v0, *v1, *v2);
                    *v0 = s0 + 1.5747 * s2;
                    *v1 = s0 - 0.1873 * s1 - 0.4682 * s2;
                    *v2 = s0 + 1.8556 * s1;
                }
            }
            for (c, plane) in planes.iter_mut().enumerate() {
                for r in 0..8.min(lines - y * 8) {
                    for k in 0..8.min(width - x * 8) {
                        let h = half_bits(blocks[c][r * 8 + k]);
                        plane[(y * 8 + r) * width + x * 8 + k] =
                            linear.map_or(h, |t| t[h as usize]);
                    }
                }
            }
        }
    }
    Some(())
}

fn words(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}

pub(super) fn decompress(
    data: &[u8],
    channels: &[Channel],
    width: usize,
    lines: usize,
) -> Option<Vec<u8>> {
    const HEADER: usize = 11 * 8;
    let counter = |i: usize| -> Option<usize> {
        usize::try_from(u64::from_le_bytes(
            data.get(i * 8..i * 8 + 8)?.try_into().ok()?,
        ))
        .ok()
    };
    let version = counter(0)?;
    if version > 2 {
        return None;
    }
    let (unknown_raw, unknown_size) = (counter(1)?, counter(2)?);
    let (ac_size, dc_size) = (counter(3)?, counter(4)?);
    let (rle_size, rle_packed, rle_raw) = (counter(5)?, counter(6)?, counter(7)?);
    let (ac_count, dc_count, ac_compression) = (counter(8)?, counter(9)?, counter(10)?);

    let mut pos = HEADER;
    let rules = if version < 2 {
        legacy_rules()
    } else {
        let (rules, size) = read_rules(data.get(pos..)?)?;
        pos += size;
        rules
    };
    let mut section = |size: usize| -> Option<&[u8]> {
        let bytes = data.get(pos..pos.checked_add(size)?)?;
        pos += size;
        Some(bytes)
    };
    let (unknown_data, ac_data) = (section(unknown_size)?, section(ac_size)?);
    let (dc_data, rle_data) = (section(dc_size)?, section(rle_size)?);

    let (schemes, sets) = classify(channels, &rules);
    let sample = |c: &Channel| if c.pixel_type == 1 { 2usize } else { 4 };
    let pixels = width * lines;
    let budget: usize = channels.iter().map(|c| sample(c) * pixels).sum();
    if unknown_raw > budget || rle_raw > budget || ac_count > budget || dc_count > budget {
        return None;
    }

    let unknown = if unknown_size > 0 {
        inflate_zlib(unknown_data).ok()?
    } else {
        Vec::new()
    };
    if unknown.len() != if unknown_size > 0 { unknown_raw } else { 0 } {
        return None;
    }
    let ac = if ac_size == 0 {
        Vec::new()
    } else if ac_compression == 0 {
        huf_uncompress(ac_data, ac_count)?
    } else if ac_compression == 1 {
        let packed = inflate_zlib(ac_data).ok()?;
        if packed.len() != ac_count * 2 {
            return None;
        }
        words(&packed)
    } else {
        return None;
    };
    let dc = if dc_size > 0 {
        let packed = inflate_zlib(dc_data).ok()?;
        if packed.len() != dc_count * 2 {
            return None;
        }
        words(&exr_reorder(&packed))
    } else if dc_count != 0 {
        return None;
    } else {
        Vec::new()
    };
    let rle_bytes = if rle_raw > 0 {
        let packed = inflate_zlib(rle_data).ok()?;
        if packed.len() != rle_packed {
            return None;
        }
        let bytes = rle(&packed, rle_raw)?;
        if bytes.len() != rle_raw {
            return None;
        }
        bytes
    } else {
        Vec::new()
    };

    // Half bits per DCT channel, filled below.
    let mut dct: Vec<Vec<u16>> = channels
        .iter()
        .zip(&schemes)
        .map(|(_, &s)| {
            if s == LOSSY_DCT {
                vec![0u16; pixels]
            } else {
                Vec::new()
            }
        })
        .collect();
    let mut done = vec![false; channels.len()];
    let (mut ac_at, mut dc_at) = (0, 0);
    for set in &sets {
        if set.iter().any(|&c| schemes[c] != LOSSY_DCT) {
            return None;
        }
        let [r, g, b] = *set;
        if r == g || g == b || r == b {
            return None;
        }
        let mut taken = [
            std::mem::take(&mut dct[r]),
            std::mem::take(&mut dct[g]),
            std::mem::take(&mut dct[b]),
        ];
        let [p0, p1, p2] = &mut taken;
        decode_dct(
            &mut [p0, p1, p2],
            &ac,
            &mut ac_at,
            &dc,
            &mut dc_at,
            Some(to_linear()),
            width,
            lines,
        )?;
        let [p0, p1, p2] = taken;
        (dct[r], dct[g], dct[b]) = (p0, p1, p2);
        done[r] = true;
        done[g] = true;
        done[b] = true;
    }
    for c in 0..channels.len() {
        if schemes[c] == LOSSY_DCT && !done[c] {
            let linear = (!channels[c].p_linear).then(to_linear);
            decode_dct(
                &mut [&mut dct[c]],
                &ac,
                &mut ac_at,
                &dc,
                &mut dc_at,
                linear,
                width,
                lines,
            )?;
        }
    }

    // Every channel as rows of its own samples.
    let (mut rle_at, mut unknown_at) = (0, 0);
    let mut rows: Vec<Vec<u8>> = Vec::with_capacity(channels.len());
    for (c, chan) in channels.iter().enumerate() {
        let size = sample(chan);
        let n = size * pixels;
        rows.push(match schemes[c] {
            LOSSY_DCT if chan.pixel_type == 1 => {
                dct[c].iter().flat_map(|h| h.to_le_bytes()).collect()
            }
            LOSSY_DCT => dct[c]
                .iter()
                .flat_map(|&h| half_float(h).to_le_bytes())
                .collect(),
            RLE => {
                // Byte planes: all of one byte position, then the next.
                let planes = rle_bytes.get(rle_at..rle_at + n)?;
                rle_at += n;
                let mut out = vec![0u8; n];
                for i in 0..pixels {
                    for byte in 0..size {
                        out[i * size + byte] = planes[byte * pixels + i];
                    }
                }
                out
            }
            _ => {
                let bytes = unknown.get(unknown_at..unknown_at + n)?.to_vec();
                unknown_at += n;
                bytes
            }
        });
    }

    let mut out = Vec::with_capacity(budget);
    for y in 0..lines {
        for (chan, plane) in channels.iter().zip(&rows) {
            let n = sample(chan) * width;
            out.extend_from_slice(&plane[y * n..(y + 1) * n]);
        }
    }
    Some(out)
}
