//! CAVLC residual block parsing (§9.2), built on the encoder's code tables.
//!
//! The tables are the encoder's `(length, code)` pairs, which ffmpeg already
//! decodes correctly; here each is turned into a decode list sorted by length.
//! A codeword is found by comparing the next 16 bits against each entry's
//! prefix — the lists are short and the common codes are first.

use crate::decode::bits::BitReader;
use crate::h264::cavlc::{
    Vlc, CT0, CT1, CT2, CT_CHROMA_DC, RUN_BEFORE, TOTAL_ZEROS_4X4, TOTAL_ZEROS_CHROMA_DC,
};

/// A decode list: `(len, code, value)`, shortest first.
#[derive(Clone, Default)]
struct DecList(Vec<(u8, u32, u16)>);

impl DecList {
    fn from(entries: impl Iterator<Item = (Vlc, u16)>) -> Self {
        let mut v: Vec<(u8, u32, u16)> = entries
            .filter(|(c, _)| c.len > 0 || c.code != 0)
            .map(|(c, val)| (c.len, c.code, val))
            .collect();
        v.sort_by_key(|e| e.0);
        DecList(v)
    }

    fn read(&self, r: &mut BitReader) -> Option<u16> {
        let peek = r.peek(16);
        for &(len, code, val) in &self.0 {
            if peek >> (16 - len as u32) == code {
                r.skip(len as u32);
                return Some(val);
            }
        }
        None
    }
}

/// 4:2:2 chroma DC `coeff_token` `(TotalCoeff, TrailingOnes, len, code)`
/// (ffmpeg `chroma422_dc_coeff_token_{len,bits}`, `nC == -2`).
const CT_CHROMA422_DC: &[(u16, u16, u8, u32)] = &[(0, 0, 1, 1), (1, 0, 7, 15), (1, 1, 2, 1), (2, 0, 7, 14), (2, 1, 7, 13), (2, 2, 3, 1), (3, 0, 9, 7), (3, 1, 7, 12), (3, 2, 7, 11), (3, 3, 5, 1), (4, 0, 9, 6), (4, 1, 9, 5), (4, 2, 7, 10), (4, 3, 6, 1), (5, 0, 10, 7), (5, 1, 10, 6), (5, 2, 9, 4), (5, 3, 7, 9), (6, 0, 11, 7), (6, 1, 11, 6), (6, 2, 10, 5), (6, 3, 7, 8), (7, 0, 12, 7), (7, 1, 12, 6), (7, 2, 11, 5), (7, 3, 10, 4), (8, 0, 13, 7), (8, 1, 12, 5), (8, 2, 12, 4), (8, 3, 11, 4)];
/// 4:2:2 chroma DC `total_zeros` `(len, code)` by `TotalCoeff - 1`.
const TOTAL_ZEROS_CHROMA422_DC: [&[(u8, u32)]; 7] = [
    &[(1, 1), (3, 2), (3, 3), (4, 2), (4, 3), (4, 1), (5, 1), (5, 0)],
    &[(3, 0), (2, 1), (3, 1), (3, 4), (3, 5), (3, 6), (3, 7)],
    &[(3, 0), (3, 1), (2, 1), (2, 2), (3, 6), (3, 7)],
    &[(3, 6), (2, 0), (2, 1), (2, 2), (3, 7)],
    &[(2, 0), (2, 1), (2, 2), (2, 3)],
    &[(2, 0), (2, 1), (1, 1)],
    &[(1, 0), (1, 1)],
];

pub struct CavlcTables {
    /// `coeff_token` for `0 <= nC < 2`, `2 <= nC < 4`, `4 <= nC < 8`, 4:2:0
    /// chroma DC and 4:2:2 chroma DC; value `total << 2 | trailing_ones`.
    coeff_token: [DecList; 5],
    total_zeros: Vec<DecList>,
    total_zeros_dc: Vec<DecList>,
    total_zeros_dc422: Vec<DecList>,
    run_before: Vec<DecList>,
}

impl CavlcTables {
    pub fn new() -> Self {
        let ct = |t: &[[Vlc; 17]; 4]| {
            DecList::from((0..4).flat_map(move |t1| {
                (0..17usize)
                    .filter(move |&tc| t1 <= tc)
                    .map(move |tc| (t[t1][tc], ((tc << 2) | t1) as u16))
            }))
        };
        let dc = DecList::from((0..4).flat_map(|t1| {
            (0..5usize)
                .filter(move |&tc| t1 <= tc)
                .map(move |tc| (CT_CHROMA_DC[t1][tc], ((tc << 2) | t1) as u16))
        }));
        let list = |t: &[Vlc]| DecList::from(t.iter().enumerate().map(|(i, v)| (*v, i as u16)));
        CavlcTables {
            coeff_token: [
                ct(&CT0),
                ct(&CT1),
                ct(&CT2),
                dc,
                DecList::from(CT_CHROMA422_DC.iter().map(|&(tc, t1, len, code)| (Vlc { len, code }, (tc << 2) | t1))),
            ],
            total_zeros: TOTAL_ZEROS_4X4.iter().map(|t| list(t)).collect(),
            total_zeros_dc: TOTAL_ZEROS_CHROMA_DC.iter().map(|t| list(t)).collect(),
            total_zeros_dc422: TOTAL_ZEROS_CHROMA422_DC
                .iter()
                .map(|t| DecList::from(t.iter().enumerate().map(|(i, &(len, code))| (Vlc { len, code }, i as u16))))
                .collect(),
            run_before: RUN_BEFORE.iter().map(|t| list(t)).collect(),
        }
    }

    /// Parse one `residual_block_cavlc`. `nc` is the neighbour context, `-1`
    /// for 4:2:0 chroma DC, `-2` for 4:2:2 chroma DC; `max` is `maxNumCoeff`.
    ///
    /// Levels land in `out` by list index, exactly as `coeffLevel` in
    /// §7.3.5.3.2 with `startIdx = 0` — so for an AC block `out[0]` is scan
    /// position 1. `out` must be zeroed by the caller. Returns `TotalCoeff`.
    pub fn residual_block(
        &self,
        r: &mut BitReader,
        nc: i32,
        max: usize,
        out: &mut [i32],
    ) -> Result<usize, String> {
        // coeff_token
        let tok = if nc == -2 {
            self.coeff_token[4].read(r)
        } else if nc == -1 {
            self.coeff_token[3].read(r)
        } else if nc < 2 {
            self.coeff_token[0].read(r)
        } else if nc < 4 {
            self.coeff_token[1].read(r)
        } else if nc < 8 {
            self.coeff_token[2].read(r)
        } else {
            let c = r.u(6) as u16;
            if c == 3 {
                Some(0)
            } else {
                let total = (c >> 2) + 1;
                let t1 = c & 3;
                if t1 > total {
                    None
                } else {
                    Some((total << 2) | t1)
                }
            }
        }
        .ok_or("bad coeff_token")?;
        let total = (tok >> 2) as usize;
        let t1 = (tok & 3) as usize;
        if total == 0 {
            return Ok(0);
        }
        if total > max {
            return Err("TotalCoeff exceeds maxNumCoeff".into());
        }
        let mut levels = [0i32; 16];
        let mut suffix_length: u32 = if total > 10 && t1 < 3 { 1 } else { 0 };
        for (i, lvl) in levels.iter_mut().enumerate().take(total) {
            if i < t1 {
                *lvl = if r.u(1) == 1 { -1 } else { 1 };
                continue;
            }
            // level_prefix: leading zeros before a 1.
            let mut prefix = 0u32;
            while r.u(1) == 0 {
                prefix += 1;
                if prefix > 32 {
                    return Err("level_prefix too long".into());
                }
            }
            let mut code: i64 = (prefix.min(15) as i64) << suffix_length;
            if suffix_length > 0 || prefix >= 14 {
                let size = if prefix == 14 && suffix_length == 0 {
                    4
                } else if prefix >= 15 {
                    prefix - 3
                } else {
                    suffix_length
                };
                if size > 0 {
                    code += r.u(size) as i64;
                }
            }
            if prefix >= 15 && suffix_length == 0 {
                code += 15;
            }
            if prefix >= 16 {
                code += (1i64 << (prefix - 3)) - 4096;
            }
            if i == t1 && t1 < 3 {
                code += 2;
            }
            let level = if code % 2 == 0 {
                (code + 2) >> 1
            } else {
                (-code - 1) >> 1
            };
            *lvl = level as i32;
            if suffix_length == 0 {
                suffix_length = 1;
            }
            if level.abs() > (3i64 << (suffix_length - 1)) && suffix_length < 6 {
                suffix_length += 1;
            }
        }
        let mut zeros_left = if total < max {
            let t = if nc == -2 {
                self.total_zeros_dc422.get(total - 1)
            } else if nc == -1 {
                self.total_zeros_dc.get(total - 1)
            } else {
                self.total_zeros.get(total - 1)
            }
            .ok_or("bad total_zeros table")?;
            t.read(r).ok_or("bad total_zeros")? as usize
        } else {
            0
        };
        if total + zeros_left > max {
            return Err("total_zeros overflows the block".into());
        }
        let mut runs = [0usize; 16];
        for run in runs.iter_mut().take(total - 1) {
            if zeros_left > 0 {
                let rb = self.run_before[zeros_left.min(7) - 1]
                    .read(r)
                    .ok_or("bad run_before")? as usize;
                if rb > zeros_left {
                    return Err("run_before exceeds zerosLeft".into());
                }
                *run = rb;
                zeros_left -= rb;
            }
        }
        runs[total - 1] = zeros_left;
        let mut coeff_num: isize = -1;
        for i in (0..total).rev() {
            coeff_num += runs[i] as isize + 1;
            out[coeff_num as usize] = levels[i];
        }
        Ok(total)
    }
}

impl Default for CavlcTables {
    fn default() -> Self {
        Self::new()
    }
}
