//! BC1 encoding — 4x4 pixel blocks in 8 bytes, an eighth of RGBA8.
//!
//! This exists because of one number: Blue Marble at its native 500 m is
//! 86400x43200, which is 13.9 GB as RGBA and fits nothing. As BC1 it is 1.74 GB,
//! or 2.32 with a full mip chain, and a whole planet at native resolution stops
//! being a streaming problem and becomes eighteen textures.
//!
//! # The format
//!
//! Each block is two RGB565 endpoints and sixteen 2-bit indices. The decoder
//! builds four colours — the two endpoints and two thirds-points between them —
//! so a block can express four colours along a single line in RGB. That is a
//! good fit for photographic ground, which is locally close to one hue, and a
//! poor one for wide smooth gradients, where four steps show as banding.
//!
//! The order of the two endpoint words is a MODE FLAG, not just an order: if the
//! first is greater the decoder uses four opaque colours, and if it is not it
//! uses three colours plus transparent black. So an encoder that emits equal
//! endpoints for a flat block — the easiest block there is — silently selects
//! the transparent mode. Everything here keeps `c0 > c1`, or emits index 0
//! everywhere when it cannot.
//!
//! # Quality
//!
//! Endpoints come from the block's principal axis, then two rounds of
//! assign-and-refit: index each pixel to its nearest palette entry, least-
//! squares fit new endpoints to that assignment, requantise. That is what makes
//! the difference on the cases that matter — a coastline, where a bounding-box
//! fit smears land into sea because the box corners are not on the data's axis.
//!
//! Encoding happens on the sRGB-ENCODED bytes, not in linear light, because
//! that is where the hardware interpolates: `Bc1RgbaUnormSrgb` decodes to sRGB
//! after the interpolation between endpoints, so matching it means fitting the
//! same numbers it will blend. Mip levels are a different question and go the
//! other way — see [`crate::textures::bc1::encode_with_mips`].

use super::texture::TextureFormat;

/// Expand a 5-bit or 6-bit channel to 8 bits the way the hardware does: repeat
/// the high bits into the low ones, so 31 -> 255 rather than 248.
#[inline]
fn expand5(v: u8) -> u8 {
    (v << 3) | (v >> 2)
}
#[inline]
fn expand6(v: u8) -> u8 {
    (v << 2) | (v >> 4)
}

/// Pack an 8-bit RGB triple into RGB565, rounding to nearest.
#[inline]
fn to565(c: [f32; 3]) -> u16 {
    let q = |v: f32, bits: f32| -> u16 {
        let m = (1u16 << bits as u16) - 1;
        ((v / 255.0 * m as f32).round().clamp(0.0, m as f32)) as u16
    };
    (q(c[0], 5.0) << 11) | (q(c[1], 6.0) << 5) | q(c[2], 5.0)
}

/// The 8-bit colour the hardware will actually see for a 565 word.
#[inline]
fn from565(w: u16) -> [f32; 3] {
    [
        expand5(((w >> 11) & 0x1f) as u8) as f32,
        expand6(((w >> 5) & 0x3f) as u8) as f32,
        expand5((w & 0x1f) as u8) as f32,
    ]
}

/// The four colours a decoder derives from two endpoint words, in the opaque
/// four-colour mode.
#[inline]
fn palette(c0: u16, c1: u16) -> [[f32; 3]; 4] {
    let a = from565(c0);
    let b = from565(c1);
    let mut p = [[0.0; 3]; 4];
    for i in 0..3 {
        p[0][i] = a[i];
        p[1][i] = b[i];
        // Exactly the decoder's thirds, computed in the same 8-bit space.
        p[2][i] = ((2.0 * a[i] + b[i]) / 3.0).round();
        p[3][i] = ((a[i] + 2.0 * b[i]) / 3.0).round();
    }
    p
}

#[inline]
fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let (x, y, z) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    x * x + y * y + z * z
}

/// Endpoints along the block's principal axis.
///
/// The alternative — the bounding box's corners — is cheaper and worse in
/// exactly the places that matter: on a coastline the box spans both land and
/// sea, and its corners are colours that appear nowhere in the block, so the
/// four palette entries land off the data and the edge smears.
fn principal_endpoints(px: &[[f32; 3]; 16]) -> ([f32; 3], [f32; 3]) {
    let mut mean = [0.0f32; 3];
    for p in px.iter() {
        for i in 0..3 {
            mean[i] += p[i];
        }
    }
    for m in mean.iter_mut() {
        *m /= 16.0;
    }
    // Covariance, upper triangle.
    let (mut xx, mut xy, mut xz, mut yy, mut yz, mut zz) = (0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0);
    for p in px.iter() {
        let d = [p[0] - mean[0], p[1] - mean[1], p[2] - mean[2]];
        xx += d[0] * d[0];
        xy += d[0] * d[1];
        xz += d[0] * d[2];
        yy += d[1] * d[1];
        yz += d[1] * d[2];
        zz += d[2] * d[2];
    }
    // Power iteration for the dominant eigenvector. Starting from the largest
    // variance axis rather than an arbitrary vector so a block that is already
    // axis-aligned converges immediately.
    let mut v = if xx > yy && xx > zz {
        [1.0, 0.0, 0.0]
    } else if yy > zz {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    for _ in 0..8 {
        let n = [
            xx * v[0] + xy * v[1] + xz * v[2],
            xy * v[0] + yy * v[1] + yz * v[2],
            xz * v[0] + yz * v[1] + zz * v[2],
        ];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len < 1e-8 {
            break; // flat block: any axis will do
        }
        v = [n[0] / len, n[1] / len, n[2] / len];
    }
    // Extremes of the projection onto that axis.
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for p in px.iter() {
        let t = (p[0] - mean[0]) * v[0] + (p[1] - mean[1]) * v[1] + (p[2] - mean[2]) * v[2];
        lo = lo.min(t);
        hi = hi.max(t);
    }
    let at = |t: f32| {
        [
            (mean[0] + v[0] * t).clamp(0.0, 255.0),
            (mean[1] + v[1] * t).clamp(0.0, 255.0),
            (mean[2] + v[2] * t).clamp(0.0, 255.0),
        ]
    };
    (at(hi), at(lo))
}

/// Index every pixel to its nearest palette entry, and report the total error.
fn assign(px: &[[f32; 3]; 16], pal: &[[f32; 3]; 4]) -> ([u8; 16], f32) {
    let mut idx = [0u8; 16];
    let mut err = 0.0;
    for (i, p) in px.iter().enumerate() {
        let (mut best, mut bd) = (0u8, f32::MAX);
        for (j, c) in pal.iter().enumerate() {
            let d = dist2(*p, *c);
            if d < bd {
                bd = d;
                best = j as u8;
            }
        }
        idx[i] = best;
        err += bd;
    }
    (idx, err)
}

/// Least-squares refit of the two endpoints, holding the assignment fixed.
///
/// Each index carries a known weight along the endpoint line — 0, 1/3, 2/3, 1 —
/// so with the indices fixed this is an ordinary two-unknown linear fit per
/// channel, and it moves the endpoints onto the data instead of onto the
/// extremes of it.
fn refit(px: &[[f32; 3]; 16], idx: &[u8; 16]) -> Option<([f32; 3], [f32; 3])> {
    const W: [f32; 4] = [1.0, 0.0, 2.0 / 3.0, 1.0 / 3.0];
    let (mut aa, mut ab, mut bb) = (0.0f32, 0.0, 0.0);
    let mut ax = [0.0f32; 3];
    let mut bx = [0.0f32; 3];
    for (i, p) in px.iter().enumerate() {
        let a = W[idx[i] as usize];
        let b = 1.0 - a;
        aa += a * a;
        ab += a * b;
        bb += b * b;
        for c in 0..3 {
            ax[c] += a * p[c];
            bx[c] += b * p[c];
        }
    }
    let det = aa * bb - ab * ab;
    if det.abs() < 1e-6 {
        return None; // every pixel on one endpoint: nothing to solve
    }
    let mut e0 = [0.0f32; 3];
    let mut e1 = [0.0f32; 3];
    for c in 0..3 {
        e0[c] = ((bb * ax[c] - ab * bx[c]) / det).clamp(0.0, 255.0);
        e1[c] = ((aa * bx[c] - ab * ax[c]) / det).clamp(0.0, 255.0);
    }
    Some((e0, e1))
}

/// Encode one 4x4 block of 8-bit RGB into 8 bytes.
fn encode_block(px: &[[f32; 3]; 16]) -> [u8; 8] {
    let (mut a, mut b) = principal_endpoints(px);
    let (mut best_w, mut best_idx, mut best_err) = (None, [0u8; 16], f32::MAX);

    // One pass with the principal-axis endpoints, then two refinements. More
    // rounds stop paying: on Blue Marble the third changes under 0.1 % of blocks.
    for _ in 0..3 {
        let (mut w0, mut w1) = (to565(a), to565(b));
        // Keep the opaque four-colour mode. If the words come out equal the
        // block is flat and index 0 alone reproduces it exactly.
        if w0 == w1 {
            let mut out = [0u8; 8];
            out[0..2].copy_from_slice(&w0.to_le_bytes());
            out[2..4].copy_from_slice(&w1.to_le_bytes());
            return out; // indices all zero
        }
        let mut swapped = false;
        if w0 < w1 {
            std::mem::swap(&mut w0, &mut w1);
            swapped = true;
        }
        let pal = palette(w0, w1);
        let (idx, err) = assign(px, &pal);
        if err < best_err {
            best_err = err;
            best_idx = idx;
            best_w = Some((w0, w1));
        }
        // Refit for the next round, in the endpoint order the indices assume.
        match refit(px, &idx) {
            Some((n0, n1)) => {
                // `refit` returns them in palette order (index 0 = first
                // endpoint), which after a swap is the reverse of a and b.
                if swapped {
                    b = n0;
                    a = n1;
                } else {
                    a = n0;
                    b = n1;
                }
            }
            None => break,
        }
    }

    let (w0, w1) = best_w.expect("at least one round always runs");
    let mut out = [0u8; 8];
    out[0..2].copy_from_slice(&w0.to_le_bytes());
    out[2..4].copy_from_slice(&w1.to_le_bytes());
    for (i, &v) in best_idx.iter().enumerate() {
        out[4 + i / 4] |= v << ((i % 4) * 2);
    }
    out
}

/// Encode RGBA8 pixels as BC1. Alpha is discarded — the format has none in the
/// mode used here.
///
/// Dimensions need not be multiples of four: partial blocks at the right and
/// bottom edges are padded by repeating the last row and column, which is what
/// keeps the edge from being pulled toward black.
pub fn encode(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    let (w, h) = (width.max(1) as usize, height.max(1) as usize);
    assert!(
        rgba.len() >= w * h * 4,
        "bc1::encode: {} bytes for {w}x{h}, need {}",
        rgba.len(),
        w * h * 4
    );
    let (bx, by) = (w.div_ceil(4), h.div_ceil(4));
    let mut out = Vec::with_capacity(bx * by * 8);
    let mut block = [[0.0f32; 3]; 16];
    for byi in 0..by {
        for bxi in 0..bx {
            for r in 0..4 {
                // Clamp rather than wrap: a block that runs off the edge repeats
                // its last real pixel.
                let y = (byi * 4 + r).min(h - 1);
                for c in 0..4 {
                    let x = (bxi * 4 + c).min(w - 1);
                    let i = (y * w + x) * 4;
                    block[r * 4 + c] = [rgba[i] as f32, rgba[i + 1] as f32, rgba[i + 2] as f32];
                }
            }
            out.extend_from_slice(&encode_block(&block));
        }
    }
    out
}

/// Decode BC1 back to RGBA8. For tests, and for anything that needs to inspect
/// what a block actually became.
pub fn decode(blocks: &[u8], width: u32, height: u32) -> Vec<u8> {
    let (w, h) = (width.max(1) as usize, height.max(1) as usize);
    let (bx, by) = (w.div_ceil(4), h.div_ceil(4));
    let mut out = vec![0u8; w * h * 4];
    for byi in 0..by {
        for bxi in 0..bx {
            let o = (byi * bx + bxi) * 8;
            if o + 8 > blocks.len() {
                break;
            }
            let c0 = u16::from_le_bytes([blocks[o], blocks[o + 1]]);
            let c1 = u16::from_le_bytes([blocks[o + 2], blocks[o + 3]]);
            let pal = if c0 > c1 {
                palette(c0, c1)
            } else {
                // Three-colour mode: the midpoint, and index 3 transparent.
                let (a, b) = (from565(c0), from565(c1));
                let mid = [
                    ((a[0] + b[0]) / 2.0).round(),
                    ((a[1] + b[1]) / 2.0).round(),
                    ((a[2] + b[2]) / 2.0).round(),
                ];
                [a, b, mid, [0.0, 0.0, 0.0]]
            };
            let bits =
                u32::from_le_bytes([blocks[o + 4], blocks[o + 5], blocks[o + 6], blocks[o + 7]]);
            for r in 0..4 {
                let y = byi * 4 + r;
                if y >= h {
                    break;
                }
                for c in 0..4 {
                    let x = bxi * 4 + c;
                    if x >= w {
                        break;
                    }
                    let i = r * 4 + c;
                    let v = ((bits >> (i * 2)) & 3) as usize;
                    let p = pal[v];
                    let d = (y * w + x) * 4;
                    out[d] = p[0] as u8;
                    out[d + 1] = p[1] as u8;
                    out[d + 2] = p[2] as u8;
                    out[d + 3] = if c0 <= c1 && v == 3 { 0 } else { 255 };
                }
            }
        }
    }
    out
}

/// Encode level 0 and a full mip chain down to 1x1.
///
/// Each level is box filtered from the level above IN LINEAR LIGHT and only
/// then encoded. Both halves of that matter and for different reasons: sRGB is
/// a transfer function, so averaging its bytes darkens every edge and a planet
/// is mostly coastline; and filtering has to happen before encoding because a
/// BC1 block cannot be averaged at all.
///
/// # How far the chain goes
///
/// All the way to 1x1, at any size. A 2x2 level of a block format still costs a
/// whole 4x4 block, and the copy is made against that PHYSICAL size — get that
/// wrong and wgpu rejects the level, which is why an earlier version stopped at
/// the last block-aligned level and gave a 14400-wide tile four mips instead of
/// twelve. Every sampler here asks for trilinear filtering, so a planet-sized
/// texture with a truncated chain does not merely soften under minification, it
/// aliases.
///
/// Returns `(level0, levels 1..N)`, which is exactly what `Texture.data` and
/// `Texture.mips` want.
pub fn encode_with_mips(rgba: &[u8], width: u32, height: u32) -> (Vec<u8>, Vec<Vec<u8>>) {
    let level0 = encode(rgba, width, height);
    let mut mips = Vec::new();
    let (mut w, mut h) = (width.max(1), height.max(1));
    let mut src = rgba.to_vec();
    // sRGB decode/encode, as tables to keep this off the transcendental path.
    let dec: Vec<f32> = (0..256)
        .map(|i| {
            let c = i as f32 / 255.0;
            if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        })
        .collect();
    let enc = |v: f32| -> u8 {
        let c = if v <= 0.0031308 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        };
        (c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
    };
    while w > 1 || h > 1 {
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        let mut next = vec![0u8; (nw * nh * 4) as usize];
        for y in 0..nh {
            for x in 0..nw {
                let mut acc = [0.0f32; 4];
                let mut n = 0.0f32;
                for dy in 0..2 {
                    let sy = (y * 2 + dy).min(h - 1);
                    for dx in 0..2 {
                        let sx = (x * 2 + dx).min(w - 1);
                        let i = ((sy * w + sx) * 4) as usize;
                        acc[0] += dec[src[i] as usize];
                        acc[1] += dec[src[i + 1] as usize];
                        acc[2] += dec[src[i + 2] as usize];
                        acc[3] += src[i + 3] as f32;
                        n += 1.0;
                    }
                }
                let o = ((y * nw + x) * 4) as usize;
                next[o] = enc(acc[0] / n);
                next[o + 1] = enc(acc[1] / n);
                next[o + 2] = enc(acc[2] / n);
                next[o + 3] = (acc[3] / n) as u8;
            }
        }
        mips.push(encode(&next, nw, nh));
        src = next;
        w = nw;
        h = nh;
    }
    (level0, mips)
}

/// Bytes a BC1 image of this size occupies, mip chain included.
pub fn size_with_mips(width: u32, height: u32) -> usize {
    let f = TextureFormat::Bc1RgbaUnormSrgb;
    let (mut w, mut h) = (width.max(1), height.max(1));
    let mut total = f.data_len(w, h);
    while w > 1 || h > 1 {
        w = (w / 2).max(1);
        h = (h / 2).max(1);
        total += f.data_len(w, h);
    }
    total
}

/// How many levels [`crate::textures::bc1::encode_with_mips`] will produce below level 0, so a caller
/// can size a tile to get the chain depth it wants.
///
/// Halving stops at the first odd number, so 12288 (3 x 2^12) gives ten levels
/// while 12720 gives two. If a tiling scheme wants deep chains, that is the
/// number to choose tile sizes against.
pub fn mip_levels(width: u32, height: u32) -> usize {
    let (mut w, mut h) = (width.max(1), height.max(1));
    let mut n = 0;
    while w > 1 || h > 1 {
        w = (w / 2).max(1);
        h = (h / 2).max(1);
        n += 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rmse(a: &[u8], b: &[u8]) -> f64 {
        let mut s = 0.0;
        let mut n = 0.0;
        for (x, y) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
            for c in 0..3 {
                let d = x[c] as f64 - y[c] as f64;
                s += d * d;
                n += 1.0;
            }
        }
        (s / n).sqrt()
    }

    /// A flat block must not land in the transparent mode.
    ///
    /// Equal endpoints make `c0 > c1` false, which selects three colours plus
    /// transparent black — so the easiest block in the format is also the one
    /// that silently punches a hole in an opaque texture.
    #[test]
    fn flat_blocks_stay_opaque_and_exact() {
        for colour in [[0u8, 0, 0], [255, 255, 255], [37, 140, 201], [8, 8, 8]] {
            let px: Vec<u8> = std::iter::repeat_n([colour[0], colour[1], colour[2], 255], 16)
                .flatten()
                .collect();
            let enc = encode(&px, 4, 4);
            let dec = decode(&enc, 4, 4);
            for p in dec.chunks_exact(4) {
                assert_eq!(p[3], 255, "flat {colour:?} decoded transparent");
                // 565 quantisation is the only error allowed here.
                for c in 0..3 {
                    assert!(
                        (p[c] as i32 - colour[c] as i32).abs() <= 4,
                        "flat {colour:?} -> {:?}",
                        &p[..3]
                    );
                }
            }
        }
    }

    /// A hard two-colour edge is what BC1 should be perfect at: two endpoints,
    /// two indices. If the mode flag or the index packing is wrong this is where
    /// it shows.
    #[test]
    fn a_hard_edge_survives_exactly() {
        let (land, sea) = ([120u8, 96, 48], [16u8, 40, 120]);
        let mut px = Vec::new();
        for r in 0..4 {
            for _ in 0..4 {
                let c = if r < 2 { land } else { sea };
                px.extend_from_slice(&[c[0], c[1], c[2], 255]);
            }
        }
        let dec = decode(&encode(&px, 4, 4), 4, 4);
        assert!(rmse(&px, &dec) < 3.0, "edge rmse {}", rmse(&px, &dec));
        for p in dec.chunks_exact(4) {
            assert_eq!(p[3], 255);
        }
    }

    /// The principal-axis fit has to beat the obvious bounding-box one, or the
    /// extra work is not worth having. A diagonal ramp through colour space is
    /// where they differ most.
    #[test]
    fn principal_axis_beats_a_bounding_box_fit() {
        let mut px = Vec::new();
        for i in 0..16 {
            let t = i as f32 / 15.0;
            px.extend_from_slice(&[
                (20.0 + 200.0 * t) as u8,
                (200.0 - 150.0 * t) as u8,
                (60.0 + 40.0 * t) as u8,
                255,
            ]);
        }
        let ours = rmse(&px, &decode(&encode(&px, 4, 4), 4, 4));

        // The bounding box's corners: max of every channel and min of every
        // channel, which for this ramp are colours the block does not contain.
        let mut lo = [255f32; 3];
        let mut hi = [0f32; 3];
        for p in px.chunks_exact(4) {
            for c in 0..3 {
                lo[c] = lo[c].min(p[c] as f32);
                hi[c] = hi[c].max(p[c] as f32);
            }
        }
        let (w0, w1) = (to565(hi), to565(lo));
        let (w0, w1) = if w0 > w1 { (w0, w1) } else { (w1, w0) };
        let pal = palette(w0, w1);
        let mut boxed = Vec::new();
        for p in px.chunks_exact(4) {
            let c = [p[0] as f32, p[1] as f32, p[2] as f32];
            let best = pal
                .iter()
                .min_by(|a, b| dist2(c, **a).partial_cmp(&dist2(c, **b)).unwrap())
                .unwrap();
            boxed.extend_from_slice(&[best[0] as u8, best[1] as u8, best[2] as u8, 255]);
        }
        let box_err = rmse(&px, &boxed);
        assert!(
            ours <= box_err,
            "principal axis {ours:.2} should beat bounding box {box_err:.2}"
        );
    }

    #[test]
    fn sizes_and_partial_blocks() {
        let f = TextureFormat::Bc1RgbaUnormSrgb;
        // A 4x4 block is 8 bytes; anything up to 4 pixels still costs one block.
        assert_eq!(f.data_len(4, 4), 8);
        assert_eq!(f.data_len(1, 1), 8);
        assert_eq!(f.data_len(5, 5), 4 * 8);
        assert_eq!(f.data_len(1024, 512), (1024 / 4) * (512 / 4) * 8);
        // Odd sizes must encode without panicking and produce whole blocks.
        let px = vec![128u8; 7 * 5 * 4];
        assert_eq!(encode(&px, 7, 5).len(), f.data_len(7, 5));
    }

    #[test]
    fn mip_chain_is_complete_and_correctly_sized() {
        let (w, h) = (64u32, 32u32);
        let mut px = Vec::new();
        for y in 0..h {
            for x in 0..w {
                px.extend_from_slice(&[(x * 4) as u8, (y * 8) as u8, 128, 255]);
            }
        }
        let (level0, mips) = encode_with_mips(&px, w, h);
        let f = TextureFormat::Bc1RgbaUnormSrgb;
        assert_eq!(level0.len(), f.data_len(w, h));
        // 64x32 -> 32x16 -> 16x8 -> 8x4 -> 4x2 -> 2x1 -> 1x1
        assert_eq!(mips.len(), 6, "chain must reach 1x1");
        assert_eq!(mip_levels(w, h), mips.len());
        let (mut lw, mut lh) = (w, h);
        for (i, m) in mips.iter().enumerate() {
            lw = (lw / 2).max(1);
            lh = (lh / 2).max(1);
            assert_eq!(m.len(), f.data_len(lw, lh), "mip {} size", i + 1);
        }
        assert_eq!(
            size_with_mips(w, h),
            level0.len() + mips.iter().map(|m| m.len()).sum::<usize>()
        );
    }
}
