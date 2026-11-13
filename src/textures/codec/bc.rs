//! BC1, BC3, BC4 and BC5: the S3TC family and its one- and two-channel
//! relatives. Each block is 4x4 texels; output is row-major within it.
//!
//! The D3D specification gives the interpolated values in exact arithmetic
//! (thirds, fifths, sevenths) and lets hardware land anywhere within a small
//! tolerance of them. These follow what Apple's GPUs were measured to do
//! instead, which is inside that tolerance: BC7's weights in 64ths — 21 and 43
//! for BC1's thirds, 9 to 55 for BC4's sevenths, 13 to 51 for its fifths —
//! rounded to the nearest 8-bit value. BC1 then decodes bit for bit as the
//! hardware does; BC4 and BC5, which the hardware returns unrounded, to within
//! half a step.

/// Expand a 5-, 6-bit colour channel to 8 by repeating its top bits.
fn expand565(c: u16) -> [u32; 3] {
    let r = ((c >> 11) & 31) as u32;
    let g = ((c >> 5) & 63) as u32;
    let b = (c & 31) as u32;
    [
        (r << 3) | (r >> 2),
        (g << 2) | (g >> 4),
        (b << 3) | (b >> 2),
    ]
}

/// A BC1 colour block to 16 RGBA8 texels. `four_colour` is BC3's colour
/// block, which always interpolates four colours whatever the endpoint order;
/// BC1 itself switches to three colours and transparent black when
/// `c0 <= c1`.
pub fn bc1_block(block: &[u8], four_colour: bool, out: &mut [u8]) {
    let c0 = u16::from_le_bytes([block[0], block[1]]);
    let c1 = u16::from_le_bytes([block[2], block[3]]);
    let (a, b) = (expand565(c0), expand565(c1));
    let mut pal = [[0u8; 4]; 4];
    // `w` of 64 towards the second endpoint.
    let mix = |w: u32| -> [u8; 4] {
        let ch = |i: usize| ((a[i] * (64 - w) + b[i] * w + 32) >> 6) as u8;
        [ch(0), ch(1), ch(2), 255]
    };
    pal[0] = [a[0] as u8, a[1] as u8, a[2] as u8, 255];
    pal[1] = [b[0] as u8, b[1] as u8, b[2] as u8, 255];
    if four_colour || c0 > c1 {
        pal[2] = mix(21);
        pal[3] = mix(43);
    } else {
        pal[2] = mix(32);
        pal[3] = [0, 0, 0, 0];
    }
    let bits = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    for i in 0..16 {
        out[i * 4..i * 4 + 4].copy_from_slice(&pal[((bits >> (2 * i)) & 3) as usize]);
    }
}

/// A BC4 block (one channel) to 16 values, written every `stride` bytes of
/// `out` starting at 0 — so BC3 can put it in alpha and BC5 in R and G.
pub fn bc4_block(block: &[u8], out: &mut [u8], stride: usize) {
    let (r0, r1) = (block[0] as u32, block[1] as u32);
    let mut pal = [0u8; 8];
    pal[0] = r0 as u8;
    pal[1] = r1 as u8;
    let mix = |w: u32| (((64 - w) * r0 + w * r1 + 32) >> 6) as u8;
    if r0 > r1 {
        for (i, w) in [9, 18, 27, 37, 46, 55].into_iter().enumerate() {
            pal[i + 2] = mix(w);
        }
    } else {
        for (i, w) in [13, 26, 38, 51].into_iter().enumerate() {
            pal[i + 2] = mix(w);
        }
        pal[6] = 0;
        pal[7] = 255;
    }
    let mut bits = 0u64;
    for (k, b) in block[2..8].iter().enumerate() {
        bits |= (*b as u64) << (8 * k);
    }
    for i in 0..16 {
        out[i * stride] = pal[((bits >> (3 * i)) & 7) as usize];
    }
}

/// BC3: a BC4 alpha block, then a four-colour BC1 colour block.
pub fn bc3_block(block: &[u8], out: &mut [u8]) {
    bc1_block(&block[8..16], true, out);
    bc4_block(&block[0..8], &mut out[3..], 4);
}

/// BC5: red then green, each a BC4 block. Out is RG8.
pub fn bc5_block(block: &[u8], out: &mut [u8]) {
    bc4_block(&block[0..8], &mut out[0..], 2);
    bc4_block(&block[8..16], &mut out[1..], 2);
}
