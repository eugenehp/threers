//! BC6H: HDR RGB in 128-bit blocks. Fourteen modes, each with its own way of
//! scattering the endpoint bits through the block, one or two subsets, and
//! endpoints either stored whole or as deltas from the first. Output is the
//! half-float bit patterns, as the hardware produces them, so a texel decodes
//! exactly to what a GPU's BC6H sampler returns.

use super::bc7::{weights, Bits, ANCHOR2, PARTITION2};

/// One step of a mode's bit layout: the next `n` stream bits go to endpoint
/// component `field` (`R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3` as 0..12) at bit
/// `shift` upward — or, with `reversed`, from bit `shift` downward.
#[derive(Clone, Copy)]
struct Step {
    field: u8,
    shift: u8,
    n: u8,
    reversed: bool,
}

const fn s(field: u8, shift: u8, n: u8) -> Step {
    Step {
        field,
        shift,
        n,
        reversed: false,
    }
}
/// `n` bits, reversed: the first stream bit becomes bit `top`.
const fn rev(field: u8, top: u8, n: u8) -> Step {
    Step {
        field,
        shift: top,
        n,
        reversed: true,
    }
}

// Endpoint component ids: endpoint e (0..4), channel c (0..3) → e * 3 + c.
const R0: u8 = 0;
const G0: u8 = 1;
const B0: u8 = 2;
const R1: u8 = 3;
const G1: u8 = 4;
const B1: u8 = 5;
const R2: u8 = 6;
const G2: u8 = 7;
const B2: u8 = 8;
const R3: u8 = 9;
const G3: u8 = 10;
const B3: u8 = 11;

struct Mode {
    /// Endpoint precision in bits, and each channel's delta bits (equal to
    /// the precision where the endpoints are not transformed).
    endpoint: u32,
    delta: [u32; 3],
    transformed: bool,
    two_subsets: bool,
    layout: &'static [Step],
}

// The layouts, in stream order after the mode bits, from the D3D11 tables.
#[rustfmt::skip]
const L1: &[Step] = &[
    s(G2, 4, 1), s(B2, 4, 1), s(B3, 4, 1), s(R0, 0, 10), s(G0, 0, 10), s(B0, 0, 10),
    s(R1, 0, 5), s(G3, 4, 1), s(G2, 0, 4), s(G1, 0, 5), s(B3, 0, 1), s(G3, 0, 4), s(B1, 0, 5),
    s(B3, 1, 1), s(B2, 0, 4), s(R2, 0, 5), s(B3, 2, 1), s(R3, 0, 5), s(B3, 3, 1),
];
#[rustfmt::skip]
const L2: &[Step] = &[
    s(G2, 5, 1), s(G3, 4, 1), s(G3, 5, 1), s(R0, 0, 7), s(B3, 0, 1), s(B3, 1, 1), s(B2, 4, 1),
    s(G0, 0, 7), s(B2, 5, 1), s(B3, 2, 1), s(G2, 4, 1), s(B0, 0, 7), s(B3, 3, 1), s(B3, 5, 1),
    s(B3, 4, 1), s(R1, 0, 6), s(G2, 0, 4), s(G1, 0, 6), s(G3, 0, 4), s(B1, 0, 6), s(B2, 0, 4),
    s(R2, 0, 6), s(R3, 0, 6),
];
#[rustfmt::skip]
const L3: &[Step] = &[
    s(R0, 0, 10), s(G0, 0, 10), s(B0, 0, 10), s(R1, 0, 5), s(R0, 10, 1), s(G2, 0, 4), s(G1, 0, 4),
    s(G0, 10, 1), s(B3, 0, 1), s(G3, 0, 4), s(B1, 0, 4), s(B0, 10, 1), s(B3, 1, 1), s(B2, 0, 4),
    s(R2, 0, 5), s(B3, 2, 1), s(R3, 0, 5), s(B3, 3, 1),
];
#[rustfmt::skip]
const L4: &[Step] = &[
    s(R0, 0, 10), s(G0, 0, 10), s(B0, 0, 10), s(R1, 0, 4), s(R0, 10, 1), s(G3, 4, 1), s(G2, 0, 4),
    s(G1, 0, 5), s(G0, 10, 1), s(G3, 0, 4), s(B1, 0, 4), s(B0, 10, 1), s(B3, 1, 1), s(B2, 0, 4),
    s(R2, 0, 4), s(B3, 0, 1), s(B3, 2, 1), s(R3, 0, 4), s(G2, 4, 1), s(B3, 3, 1),
];
#[rustfmt::skip]
const L5: &[Step] = &[
    s(R0, 0, 10), s(G0, 0, 10), s(B0, 0, 10), s(R1, 0, 4), s(R0, 10, 1), s(B2, 4, 1), s(G2, 0, 4),
    s(G1, 0, 4), s(G0, 10, 1), s(B3, 0, 1), s(G3, 0, 4), s(B1, 0, 5), s(B0, 10, 1), s(B2, 0, 4),
    s(R2, 0, 4), s(B3, 1, 1), s(B3, 2, 1), s(R3, 0, 4), s(B3, 4, 1), s(B3, 3, 1),
];
#[rustfmt::skip]
const L6: &[Step] = &[
    s(R0, 0, 9), s(B2, 4, 1), s(G0, 0, 9), s(G2, 4, 1), s(B0, 0, 9), s(B3, 4, 1), s(R1, 0, 5),
    s(G3, 4, 1), s(G2, 0, 4), s(G1, 0, 5), s(B3, 0, 1), s(G3, 0, 4), s(B1, 0, 5), s(B3, 1, 1),
    s(B2, 0, 4), s(R2, 0, 5), s(B3, 2, 1), s(R3, 0, 5), s(B3, 3, 1),
];
#[rustfmt::skip]
const L7: &[Step] = &[
    s(R0, 0, 8), s(G3, 4, 1), s(B2, 4, 1), s(G0, 0, 8), s(B3, 2, 1), s(G2, 4, 1), s(B0, 0, 8),
    s(B3, 3, 1), s(B3, 4, 1), s(R1, 0, 6), s(G2, 0, 4), s(G1, 0, 5), s(B3, 0, 1), s(G3, 0, 4),
    s(B1, 0, 5), s(B3, 1, 1), s(B2, 0, 4), s(R2, 0, 6), s(R3, 0, 6),
];
#[rustfmt::skip]
const L8: &[Step] = &[
    s(R0, 0, 8), s(B3, 0, 1), s(B2, 4, 1), s(G0, 0, 8), s(G2, 5, 1), s(G2, 4, 1), s(B0, 0, 8),
    s(G3, 5, 1), s(B3, 4, 1), s(R1, 0, 5), s(G3, 4, 1), s(G2, 0, 4), s(G1, 0, 6), s(G3, 0, 4),
    s(B1, 0, 5), s(B3, 1, 1), s(B2, 0, 4), s(R2, 0, 5), s(B3, 2, 1), s(R3, 0, 5), s(B3, 3, 1),
];
#[rustfmt::skip]
const L9: &[Step] = &[
    s(R0, 0, 8), s(B3, 1, 1), s(B2, 4, 1), s(G0, 0, 8), s(B2, 5, 1), s(G2, 4, 1), s(B0, 0, 8),
    s(B3, 5, 1), s(B3, 4, 1), s(R1, 0, 5), s(G3, 4, 1), s(G2, 0, 4), s(G1, 0, 5), s(B3, 0, 1),
    s(G3, 0, 4), s(B1, 0, 6), s(B2, 0, 4), s(R2, 0, 5), s(B3, 2, 1), s(R3, 0, 5), s(B3, 3, 1),
];
#[rustfmt::skip]
const L10: &[Step] = &[
    s(R0, 0, 6), s(G3, 4, 1), s(B3, 0, 1), s(B3, 1, 1), s(B2, 4, 1), s(G0, 0, 6), s(G2, 5, 1),
    s(B2, 5, 1), s(B3, 2, 1), s(G2, 4, 1), s(B0, 0, 6), s(G3, 5, 1), s(B3, 3, 1), s(B3, 5, 1),
    s(B3, 4, 1), s(R1, 0, 6), s(G2, 0, 4), s(G1, 0, 6), s(G3, 0, 4), s(B1, 0, 6), s(B2, 0, 4),
    s(R2, 0, 6), s(R3, 0, 6),
];
#[rustfmt::skip]
const L11: &[Step] = &[
    s(R0, 0, 10), s(G0, 0, 10), s(B0, 0, 10), s(R1, 0, 10), s(G1, 0, 10), s(B1, 0, 10),
];
#[rustfmt::skip]
const L12: &[Step] = &[
    s(R0, 0, 10), s(G0, 0, 10), s(B0, 0, 10), s(R1, 0, 9), s(R0, 10, 1), s(G1, 0, 9), s(G0, 10, 1),
    s(B1, 0, 9), s(B0, 10, 1),
];
#[rustfmt::skip]
const L13: &[Step] = &[
    s(R0, 0, 10), s(G0, 0, 10), s(B0, 0, 10), s(R1, 0, 8), rev(R0, 11, 2), s(G1, 0, 8),
    rev(G0, 11, 2), s(B1, 0, 8), rev(B0, 11, 2),
];
#[rustfmt::skip]
const L14: &[Step] = &[
    s(R0, 0, 10), s(G0, 0, 10), s(B0, 0, 10), s(R1, 0, 4), rev(R0, 15, 6), s(G1, 0, 4),
    rev(G0, 15, 6), s(B1, 0, 4), rev(B0, 15, 6),
];

const fn two(endpoint: u32, d: [u32; 3], layout: &'static [Step]) -> Mode {
    Mode {
        endpoint,
        delta: d,
        transformed: true,
        two_subsets: true,
        layout,
    }
}

/// The mode for a block's mode bits, `None` for the reserved ones.
#[rustfmt::skip]
fn mode(block: &mut Bits) -> Option<&'static Mode> {
    static M1: Mode = two(10, [5, 5, 5], L1);
    static M2: Mode = two(7, [6, 6, 6], L2);
    static M3: Mode = two(11, [5, 4, 4], L3);
    static M4: Mode = two(11, [4, 5, 4], L4);
    static M5: Mode = two(11, [4, 4, 5], L5);
    static M6: Mode = two(9, [5, 5, 5], L6);
    static M7: Mode = two(8, [6, 5, 5], L7);
    static M8: Mode = two(8, [5, 6, 5], L8);
    static M9: Mode = two(8, [5, 5, 6], L9);
    static M10: Mode = Mode { endpoint: 6, delta: [6, 6, 6], transformed: false, two_subsets: true, layout: L10 };
    static M11: Mode = Mode { endpoint: 10, delta: [10, 10, 10], transformed: false, two_subsets: false, layout: L11 };
    static M12: Mode = Mode { endpoint: 11, delta: [9, 9, 9], transformed: true, two_subsets: false, layout: L12 };
    static M13: Mode = Mode { endpoint: 12, delta: [8, 8, 8], transformed: true, two_subsets: false, layout: L13 };
    static M14: Mode = Mode { endpoint: 16, delta: [4, 4, 4], transformed: true, two_subsets: false, layout: L14 };
    let low = block.read(2);
    if low < 2 {
        return Some(if low == 0 { &M1 } else { &M2 });
    }
    let m = low | (block.read(3) << 2);
    Some(match m {
        0x02 => &M3,
        0x06 => &M4,
        0x0a => &M5,
        0x0e => &M6,
        0x12 => &M7,
        0x16 => &M8,
        0x1a => &M9,
        0x1e => &M10,
        0x03 => &M11,
        0x07 => &M12,
        0x0b => &M13,
        0x0f => &M14,
        _ => return None,
    })
}

fn sign_extend(v: i32, bits: u32) -> i32 {
    let shift = 32 - bits;
    (v << shift) >> shift
}

/// Endpoint value to the 16-bit range the interpolation works in.
fn unquantize(v: i32, bits: u32, signed: bool) -> i32 {
    if !signed {
        if bits >= 15 {
            v
        } else if v == 0 {
            0
        } else if v == (1 << bits) - 1 {
            0xffff
        } else {
            ((v << 16) + 0x8000) >> bits
        }
    } else {
        if bits >= 16 {
            return v;
        }
        let (neg, a) = if v < 0 { (true, -v) } else { (false, v) };
        let q = if a == 0 {
            0
        } else if a >= (1 << (bits - 1)) - 1 {
            0x7fff
        } else {
            ((a << 15) + 0x4000) >> (bits - 1)
        };
        if neg {
            -q
        } else {
            q
        }
    }
}

/// The interpolated value's last step to half-float bits.
fn finish(v: i32, signed: bool) -> u16 {
    if !signed {
        ((v * 31) >> 6) as u16
    } else if v < 0 {
        0x8000 | ((((-v) * 31) >> 5) as u16)
    } else {
        ((v * 31) >> 5) as u16
    }
}

/// One BC6H block to 16 RGBA half-float texels (alpha 1.0), row-major.
pub fn block(block: &[u8], signed: bool, out: &mut [u16]) {
    let mut r = Bits::new(block);
    let Some(m) = mode(&mut r) else {
        // Reserved modes decode to zero.
        for px in out[..64].as_chunks_mut::<4>().0 {
            *px = [0, 0, 0, 0x3c00];
        }
        return;
    };
    let mut e = [0i32; 12];
    for st in m.layout {
        let v = r.read(st.n as u32) as i32;
        if st.reversed {
            for k in 0..st.n {
                e[st.field as usize] |= ((v >> k) & 1) << (st.shift - k);
            }
        } else {
            e[st.field as usize] |= v << st.shift;
        }
    }
    let partition = if m.two_subsets { r.read(5) as usize } else { 0 };
    let n_end = if m.two_subsets { 4 } else { 2 };

    // Sign-extend, and turn deltas into endpoints.
    let mask = (1i32 << m.endpoint) - 1;
    for c in 0..3 {
        if signed {
            e[c] = sign_extend(e[c], m.endpoint);
        }
        for k in 1..n_end {
            let i = k * 3 + c;
            if m.transformed {
                let d = sign_extend(e[i], m.delta[c]);
                e[i] = (e[c] + d) & mask;
                if signed {
                    e[i] = sign_extend(e[i], m.endpoint);
                }
            } else if signed {
                e[i] = sign_extend(e[i], m.endpoint);
            }
        }
    }
    for v in e.iter_mut().take(n_end * 3) {
        *v = unquantize(*v, m.endpoint, signed);
    }

    let index_bits = if m.two_subsets { 3 } else { 4 };
    let w = weights(index_bits);
    for i in 0..16 {
        let sub = if m.two_subsets {
            ((PARTITION2[partition] >> i) & 1) as usize
        } else {
            0
        };
        let anchor = i == 0 || (m.two_subsets && i == ANCHOR2[partition] as usize);
        let idx = r.read(index_bits - anchor as u32) as usize;
        let wt = w[idx] as i32;
        let (a, b) = (sub * 6, sub * 6 + 3);
        for c in 0..3 {
            let v = ((64 - wt) * e[a + c] + wt * e[b + c] + 32) >> 6;
            out[i * 4 + c] = finish(v, signed);
        }
        out[i * 4 + 3] = 0x3c00;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_mode_fills_exactly_128_bits() {
        for code in [
            0u32, 1, 0x02, 0x06, 0x0a, 0x0e, 0x12, 0x16, 0x1a, 0x1e, 0x03, 0x07, 0x0b, 0x0f,
        ] {
            let mut blk = [0u8; 16];
            blk[0] = code as u8;
            let mut r = Bits::new(&blk);
            let m = mode(&mut r).expect("a valid mode");
            let mode_bits = if code < 2 { 2 } else { 5 };
            let layout: u32 = m.layout.iter().map(|s| s.n as u32).sum();
            let rest = if m.two_subsets {
                5 + 16 * 3 - 2
            } else {
                16 * 4 - 1
            };
            assert_eq!(mode_bits + layout + rest, 128, "mode {code:#x}");
        }
    }

    #[test]
    fn layouts_set_each_endpoint_bit_once() {
        for code in [
            0u32, 1, 0x02, 0x06, 0x0a, 0x0e, 0x12, 0x16, 0x1a, 0x1e, 0x03, 0x07, 0x0b, 0x0f,
        ] {
            let mut blk = [0u8; 16];
            blk[0] = code as u8;
            let m = mode(&mut Bits::new(&blk)).unwrap();
            let mut seen = [0u32; 12];
            for st in m.layout {
                for k in 0..st.n {
                    let bit = if st.reversed {
                        st.shift - k
                    } else {
                        st.shift + k
                    };
                    assert_eq!(
                        seen[st.field as usize] & (1 << bit),
                        0,
                        "mode {code:#x} field {}",
                        st.field
                    );
                    seen[st.field as usize] |= 1 << bit;
                }
            }
            // Endpoint 0 has every bit of the precision; deltas their own widths.
            let n_end = if m.two_subsets { 4 } else { 2 };
            for c in 0..3 {
                assert_eq!(
                    seen[c],
                    (1 << m.endpoint) - 1,
                    "mode {code:#x} endpoint 0 channel {c}"
                );
                for k in 1..n_end {
                    assert_eq!(
                        seen[k * 3 + c],
                        (1 << m.delta[c]) - 1,
                        "mode {code:#x} endpoint {k} channel {c}"
                    );
                }
            }
        }
    }
}
