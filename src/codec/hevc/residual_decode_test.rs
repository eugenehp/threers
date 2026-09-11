// Included into `residual::tests`. A CABAC decoder + a residual decoder mirroring
// `encode`, used to round-trip random coefficient blocks and prove the coding
// logic (last position, significance, greater-1/2, signs, Golomb-Rice) is
// self-consistent regardless of the context init values.

use crate::codec::hevc::tables::{RANGE_TAB_LPS, TRANS_IDX_LPS, TRANS_IDX_MPS};

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Reader<'a> {
    fn bit(&mut self) -> u32 {
        let (byte, off) = (self.p >> 3, 7 - (self.p & 7));
        self.p += 1;
        if byte < self.b.len() {
            ((self.b[byte] >> off) & 1) as u32
        } else {
            0
        }
    }
    fn bits(&mut self, n: u32) -> u32 {
        let mut v = 0;
        for _ in 0..n {
            v = (v << 1) | self.bit();
        }
        v
    }
}

struct Dec<'a> {
    range: u32,
    offset: u32,
    r: Reader<'a>,
}
impl<'a> Dec<'a> {
    fn new(b: &'a [u8]) -> Self {
        let mut r = Reader { b, p: 0 };
        let offset = r.bits(9);
        Self { range: 510, offset, r }
    }
    fn bin(&mut self, c: &mut CtxModel) -> u32 {
        let q = ((self.range >> 6) & 3) as usize;
        let lps = RANGE_TAB_LPS[c.state as usize][q] as u32;
        self.range -= lps;
        let v;
        if self.offset >= self.range {
            v = 1 - c.mps as u32;
            self.offset -= self.range;
            self.range = lps;
            if c.state == 0 {
                c.mps = 1 - c.mps;
            }
            c.state = TRANS_IDX_LPS[c.state as usize];
        } else {
            v = c.mps as u32;
            c.state = TRANS_IDX_MPS[c.state as usize];
        }
        while self.range < 256 {
            self.range <<= 1;
            self.offset = (self.offset << 1) | self.r.bit();
        }
        v
    }
    fn bypass(&mut self) -> u32 {
        self.offset = (self.offset << 1) | self.r.bit();
        if self.offset >= self.range {
            self.offset -= self.range;
            1
        } else {
            0
        }
    }
    fn bypass_bits(&mut self, n: u32) -> u32 {
        let mut v = 0;
        for _ in 0..n {
            v = (v << 1) | self.bypass();
        }
        v
    }
    fn decode_terminate(&mut self) -> u32 {
        self.range -= 2;
        if self.offset >= self.range {
            1
        } else {
            while self.range < 256 {
                self.range <<= 1;
                self.offset = (self.offset << 1) | self.r.bit();
            }
            0
        }
    }
}

fn decode_last_prefix(dec: &mut Dec, ctxs: &mut [CtxModel], log2n: usize, chroma: bool) -> usize {
    let c_max = (log2n << 1) - 1;
    let mut group = 0;
    while group < c_max && dec.bin(&mut ctxs[last_ctx(group, log2n, chroma)]) == 1 {
        group += 1;
    }
    group
}

fn decode_last_suffix(dec: &mut Dec, group: usize) -> usize {
    if group > 3 {
        let nbits = (group >> 1) - 1;
        MIN_IN_GROUP[group] + dec.bypass_bits(nbits as u32) as usize
    } else {
        group
    }
}

fn read_remaining(dec: &mut Dec, rice: u32) -> u32 {
    let mut prefix = 0u32;
    while dec.bypass() == 1 {
        prefix += 1;
    }
    if prefix < 3 {
        let suffix = if rice > 0 { dec.bypass_bits(rice) } else { 0 };
        (prefix << rice) + suffix
    } else {
        let len = prefix - 3 + rice;
        let suffix = dec.bypass_bits(len);
        ((3 << rice) + ((1u32 << len) - (1u32 << rice))) + suffix
    }
}

fn decode(dec: &mut Dec, ctx: &mut ResidualCtx, n: usize, chroma: bool, scan_idx: u8) -> Vec<i32> {
    let log2n = log2(n);
    let scan = full_scan(n, scan_idx);
    let inner = scan_kxk(4, scan_idx);
    let side = n / 4;

    let gx = decode_last_prefix(dec, &mut ctx.last_x, log2n, chroma);
    let gy = decode_last_prefix(dec, &mut ctx.last_y, log2n, chroma);
    let last_x = decode_last_suffix(dec, gx);
    let last_y = decode_last_suffix(dec, gy);
    // A vertical scan signals the last position transposed (§7.4.9.11).
    let (last_x, last_y) = if scan_idx == 2 {
        (last_y, last_x)
    } else {
        (last_x, last_y)
    };
    let last_scan = scan.iter().position(|&p| p == (last_x, last_y)).unwrap();
    let last_sb = last_scan / 16;
    let last_pos = last_scan % 16;

    let mut coeff = vec![0i32; n * n];
    let mut csbf = vec![vec![false; side]; side];
    let sb_scan = scan_kxk(side, scan_idx);
    let mut c1_carry = 1u32;

    for si in (0..=last_sb).rev() {
        let (sxs, sys) = sb_scan[si];
        let (is_first, is_last) = (si == 0, si == last_sb);
        let mut infer_dc = false;
        if !is_first && !is_last {
            let rb = csbf_right_below(&csbf, sxs, sys, side);
            let ci = (rb.min(1) as usize) + if chroma { 2 } else { 0 };
            let bit = dec.bin(&mut ctx.csbf[ci]) == 1;
            csbf[sys][sxs] = bit;
            if !bit {
                continue;
            }
            infer_dc = true;
        } else {
            csbf[sys][sxs] = true;
        }
        let csbf_rb = csbf_right_below(&csbf, sxs, sys, side);

        let start = if is_last { last_pos as i32 - 1 } else { 15 };
        let mut sig = [false; 16];
        if is_last {
            sig[last_pos] = true;
        }
        let sub_nonzero = sxs + sys != 0;
        let mut num_sig = if is_last { 1 } else { 0 };
        for p in (0..=start).rev() {
            let (px, py) = inner[p as usize];
            let (xc, yc) = (sxs * 4 + px, sys * 4 + py);
            if p == 0 && infer_dc && num_sig == 0 {
                sig[0] = true;
                num_sig += 1;
                break;
            }
            let ci = sig_ctx(xc, yc, log2n, chroma, sub_nonzero, csbf_rb, scan_idx);
            if dec.bin(&mut ctx.sig[ci]) == 1 {
                sig[p as usize] = true;
                num_sig += 1;
            }
        }
        if num_sig == 0 {
            continue;
        }

        let mut positions = Vec::with_capacity(num_sig);
        for p in (0..16).rev() {
            if sig[p] {
                let (px, py) = inner[p];
                positions.push((sxs * 4 + px, sys * 4 + py));
            }
        }

        let ctx_set_base = if si > 0 && !chroma { 2 } else { 0 };
        let ctx_set = ctx_set_base + if c1_carry == 0 { 1 } else { 0 };
        let mut c1 = 1u32;
        let mut first_gt1: i32 = -1;
        let n_gt1 = positions.len().min(8);
        let mut gt1 = vec![0u32; n_gt1];
        for idx in 0..n_gt1 {
            let base = if chroma { 16 } else { 0 };
            let ci = base + (ctx_set << 2) + c1 as usize;
            let bin = dec.bin(&mut ctx.gt1[ci]);
            gt1[idx] = bin;
            if bin == 1 {
                c1 = 0;
                if first_gt1 < 0 {
                    first_gt1 = idx as i32;
                }
            } else if c1 > 0 && c1 < 3 {
                c1 += 1;
            }
        }
        c1_carry = c1;

        let mut gt2 = 0u32;
        if first_gt1 >= 0 {
            let base = if chroma { 4 } else { 0 };
            gt2 = dec.bin(&mut ctx.gt2[base + ctx_set]);
        }

        let signs: Vec<u32> = (0..positions.len()).map(|_| dec.bypass()).collect();

        let mut first_coeff2 = 1u32;
        let mut rice = 0u32;
        for idx in 0..positions.len() {
            let base_level = if idx < 8 { 2 + first_coeff2 } else { 1 };
            let g1 = if idx < n_gt1 { gt1[idx] } else { 0 };
            let g2 = if idx as i32 == first_gt1 { gt2 } else { 0 };
            let coded = 1 + g1 + g2;
            let abs = if coded == base_level {
                let rem = read_remaining(dec, rice);
                let a = base_level + rem;
                if a > 3 * (1 << rice) {
                    rice = (rice + 1).min(4);
                }
                a
            } else {
                coded
            };
            if abs >= 2 {
                first_coeff2 = 0;
            }
            let (xc, yc) = positions[idx];
            coeff[yc * n + xc] = if signs[idx] == 1 { -(abs as i32) } else { abs as i32 };
        }
    }
    coeff
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn upto(&mut self, m: u32) -> u32 {
        (self.next() % m as u64) as u32
    }
}

fn roundtrip_block(n: usize, chroma: bool, scan_idx: u8, coeff: &[i32], qp: i32) {
    let mut enc = CabacEncoder::new();
    let mut ectx = ResidualCtx::new(qp);
    encode(&mut enc, &mut ectx, coeff, n, chroma, scan_idx);
    enc.encode_terminate(1);
    let bytes = enc.finish();

    let mut dec = Dec::new(&bytes);
    let mut dctx = ResidualCtx::new(qp);
    let got = decode(&mut dec, &mut dctx, n, chroma, scan_idx);
    assert_eq!(got, coeff, "residual roundtrip n={n} chroma={chroma} scan={scan_idx}");
}

#[test]
fn residual_roundtrips_random() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    for trial in 0..300u32 {
        let n = *[4usize, 8, 16].get((trial % 3) as usize).unwrap();
        let chroma = trial % 2 == 0;
        let scan_idx = (trial % 3) as u8;
        let mut coeff = vec![0i32; n * n];
        // Sparse-ish coefficients with a mix of magnitudes and signs.
        let density = 1 + rng.upto(6);
        let mut any = false;
        for c in coeff.iter_mut() {
            if rng.upto(density) == 0 {
                let mag = match rng.upto(10) {
                    0..=5 => 1,
                    6..=7 => 2,
                    8 => 3 + rng.upto(5) as i32,
                    _ => 1 + rng.upto(300) as i32,
                };
                *c = if rng.upto(2) == 0 { -mag } else { mag };
                any = true;
            }
        }
        if !any {
            coeff[0] = 1; // must have ≥1 significant coeff
        }
        roundtrip_block(n, chroma, scan_idx, &coeff, 26);
    }
}

#[test]
fn residual_roundtrips_edge_cases() {
    // DC only, single high coeff, full block.
    roundtrip_block(4, false, 0, &{ let mut c = vec![0; 16]; c[0] = 5; c }, 26);
    roundtrip_block(16, false, 0, &{ let mut c = vec![0; 256]; c[255] = -1; c }, 26);
    roundtrip_block(8, true, 0, &vec![7; 64], 26);
}

/// Chain many blocks with `encode_terminate(0)` between them (as CTBs do) — this
/// exercises the CABAC engine across the residual→terminate→residual boundary
/// that a single-block test misses.
#[test]
fn residual_roundtrips_chained() {
    let mut rng = Rng(0xdead_beef_cafe_0001);
    let blocks: Vec<(usize, bool, Vec<i32>)> = (0..40)
        .map(|t| {
            let n = *[4usize, 8, 16].get((t % 3) as usize).unwrap();
            let mut c = vec![0i32; n * n];
            for x in c.iter_mut() {
                if rng.upto(4) == 0 {
                    let m = 1 + rng.upto(50) as i32;
                    *x = if rng.upto(2) == 0 { -m } else { m };
                }
            }
            if !c.iter().any(|&v| v != 0) {
                c[0] = 1;
            }
            (n, t % 2 == 0, c)
        })
        .collect();

    // Encode all, terminate(0) between, terminate(1) at the end.
    let mut enc = CabacEncoder::new();
    let mut ectx = ResidualCtx::new(26);
    for (i, (n, chroma, c)) in blocks.iter().enumerate() {
        encode(&mut enc, &mut ectx, c, *n, *chroma, 0);
        enc.encode_terminate((i + 1 == blocks.len()) as u32);
    }
    let bytes = enc.finish();

    // Decode all back and compare.
    let mut dec = Dec::new(&bytes);
    let mut dctx = ResidualCtx::new(26);
    for (i, (n, chroma, c)) in blocks.iter().enumerate() {
        let got = decode(&mut dec, &mut dctx, *n, *chroma, 0);
        assert_eq!(&got, c, "chained block {i}");
        let term = dec.decode_terminate();
        assert_eq!(term, (i + 1 == blocks.len()) as u32, "terminate {i}");
    }
}
