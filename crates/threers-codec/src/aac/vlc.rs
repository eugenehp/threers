//! Huffman lookup for the AAC spectral and scalefactor codebooks: a direct
//! table indexed by the next `max_len` bits, built once from the
//! `ff_aac_spectral_codes/bits` and `ff_aac_scalefactor_code/bits` tables
//! with `ff_aac_codebook_vector_idx` as the spectral symbols, as
//! `ff_vlc_init_tables_sparse` does.

use super::bits::BitReader;
use super::tables;
use std::sync::OnceLock;

pub struct Vlc {
    max_len: u32,
    /// `(symbol << 8) | len`; `len == 0` marks an invalid code.
    table: Vec<u32>,
}

impl Vlc {
    fn build(codes: &[u32], lens: &[u8], syms: &[u32]) -> Self {
        let max_len = *lens.iter().max().unwrap() as u32;
        let mut table = vec![0u32; 1 << max_len];
        for ((&c, &l), &s) in codes.iter().zip(lens).zip(syms) {
            if l == 0 {
                continue;
            }
            let l = l as u32;
            let base = (c as usize) << (max_len - l);
            for e in &mut table[base..base + (1 << (max_len - l))] {
                *e = (s << 8) | l;
            }
        }
        Self { max_len, table }
    }

    /// Decode one symbol; `None` on an invalid code.
    #[inline]
    pub fn read(&self, br: &mut BitReader) -> Option<u32> {
        let e = self.table[br.show(self.max_len) as usize];
        let len = e & 0xff;
        if len == 0 {
            return None;
        }
        br.skip(len);
        Some(e >> 8)
    }
}

pub fn spectral(cb_m1: usize) -> &'static Vlc {
    static T: OnceLock<Vec<Vlc>> = OnceLock::new();
    &T.get_or_init(|| {
        (0..11)
            .map(|i| {
                let codes: Vec<u32> = tables::CODES[i].iter().map(|&c| c as u32).collect();
                let syms: Vec<u32> = tables::VEC_IDX[i].iter().map(|&s| s as u32).collect();
                Vlc::build(&codes, tables::BITS[i], &syms)
            })
            .collect()
    })[cb_m1]
}

pub fn scalefactor() -> &'static Vlc {
    static T: OnceLock<Vlc> = OnceLock::new();
    T.get_or_init(|| {
        let syms: Vec<u32> = (0..tables::SF_CODE.len() as u32).collect();
        Vlc::build(&tables::SF_CODE, &tables::SF_BITS, &syms)
    })
}
