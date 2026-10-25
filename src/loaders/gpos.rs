//! Pair kerning from `GPOS`: the `kern` feature's pair-adjustment lookups
//! for Latin text (type 2, directly or through an extension), glyph pairs
//! and class pairs, with variable fonts' deltas applied through `GDEF`'s
//! variation store.
//!
//! A modern font keeps its kerning here and nowhere else — SF Compact, Inter,
//! Roboto have no `kern` table at all — so a reader of `kern` alone sets every
//! "AV", "To" and "Ty" loose.

use super::ttf_var::ItemVariationStore;
use std::collections::HashMap;

fn u16_at(b: &[u8], p: usize) -> Option<u16> {
    Some(u16::from_be_bytes([*b.get(p)?, *b.get(p + 1)?]))
}
fn u32_at(b: &[u8], p: usize) -> Option<u32> {
    Some(u32::from_be_bytes([*b.get(p)?, *b.get(p + 1)?, *b.get(p + 2)?, *b.get(p + 3)?]))
}

/// One pair-adjustment subtable, resolved at the font's location.
#[derive(Debug, Clone)]
enum PairTable {
    /// Explicit pairs: first glyph's coverage, then the second glyph.
    Glyphs { coverage: HashMap<u16, u16>, pairs: Vec<HashMap<u16, f32>> },
    /// Class pairs: `values[c1 * class2_count + c2]`.
    Classes {
        coverage: HashMap<u16, u16>,
        class1: HashMap<u16, u16>,
        class2: HashMap<u16, u16>,
        class2_count: usize,
        values: Vec<f32>,
    },
}

/// The `kern` feature's pair adjustments, as lookups of subtables.
#[derive(Debug, Clone, Default)]
pub struct GposKerning {
    lookups: Vec<Vec<PairTable>>,
}

impl GposKerning {
    /// Whether the font had any pair kerning in `GPOS`.
    pub fn is_empty(&self) -> bool {
        self.lookups.is_empty()
    }

    /// The x-advance adjustment between two glyphs, in font units.
    ///
    /// Each lookup contributes once: its first subtable that covers `left`
    /// (and, for explicit pairs, lists `right`) decides, as a shaper applies
    /// it.
    pub fn kern(&self, left: u16, right: u16) -> f32 {
        let mut total = 0.0;
        for lookup in &self.lookups {
            for table in lookup {
                match table {
                    PairTable::Glyphs { coverage, pairs } => {
                        let Some(&i) = coverage.get(&left) else { continue };
                        if let Some(v) = pairs.get(i as usize).and_then(|p| p.get(&right)) {
                            total += v;
                            break;
                        }
                    }
                    PairTable::Classes { coverage, class1, class2, class2_count, values } => {
                        if !coverage.contains_key(&left) {
                            continue;
                        }
                        let c1 = class1.get(&left).copied().unwrap_or(0) as usize;
                        let c2 = class2.get(&right).copied().unwrap_or(0) as usize;
                        total += values.get(c1 * class2_count + c2).copied().unwrap_or(0.0);
                        break;
                    }
                }
            }
        }
        total
    }
}

/// The lookups the features tagged `tags` name in a `GSUB` or `GPOS` table
/// at `table`, for one script's default language system — Latin, else the
/// default script, else the first — in lookup-list order, as a shaper
/// applies them to a run of that script. Different scripts name different
/// lookups, and the union of them all would kern (or ligate) Latin text
/// with another script's rules. The language system's required feature
/// counts when its tag is asked for.
pub(crate) fn feature_lookups(bytes: &[u8], table: usize, tags: &[&[u8; 4]]) -> Option<Vec<u16>> {
    let scripts = table + u16_at(bytes, table + 4)? as usize;
    let features = table + u16_at(bytes, table + 6)? as usize;
    let script_count = u16_at(bytes, scripts)? as usize;
    let script_at = |tag: &[u8; 4]| -> Option<usize> {
        (0..script_count).find_map(|i| {
            let r = scripts + 2 + 6 * i;
            (bytes.get(r..r + 4)? == tag).then(|| u16_at(bytes, r + 4).map(|o| scripts + o as usize)).flatten()
        })
    };
    let script = script_at(b"latn")
        .or_else(|| script_at(b"DFLT"))
        .or_else(|| (script_count > 0).then(|| u16_at(bytes, scripts + 6).map(|o| scripts + o as usize)).flatten())?;
    let lang_sys = u16_at(bytes, script)? as usize;
    if lang_sys == 0 {
        return Some(Vec::new());
    }
    let lang_sys = script + lang_sys;
    let feature_count = u16_at(bytes, features)? as usize;
    let required = u16_at(bytes, lang_sys + 2)? as usize;
    let n = u16_at(bytes, lang_sys + 4)? as usize;
    let chosen = (0..n)
        .filter_map(|k| u16_at(bytes, lang_sys + 6 + 2 * k).map(usize::from))
        .chain((required != 0xFFFF).then_some(required))
        .filter(|&i| i < feature_count);
    let mut indices = Vec::new();
    for i in chosen {
        let r = features + 2 + 6 * i;
        let tag = bytes.get(r..r + 4)?;
        if !tags.iter().any(|t| t[..] == *tag) {
            continue;
        }
        let f = features + u16_at(bytes, r + 4)? as usize;
        let n = u16_at(bytes, f + 2)? as usize;
        for k in 0..n {
            indices.push(u16_at(bytes, f + 4 + 2 * k)?);
        }
    }
    indices.sort_unstable();
    indices.dedup();
    Some(indices)
}

/// Lookup `index`'s subtables in a `GSUB` or `GPOS` table at `table`, as
/// `(type, offset)`, through extension lookups (`extension`: 7 in `GSUB`, 9
/// in `GPOS`).
pub(crate) fn lookup_subtables(bytes: &[u8], table: usize, index: u16, extension: u16) -> Option<Vec<(u16, usize)>> {
    let lookups = table + u16_at(bytes, table + 8)? as usize;
    if index >= u16_at(bytes, lookups)? {
        return None;
    }
    let l = lookups + u16_at(bytes, lookups + 2 + 2 * index as usize)? as usize;
    let kind = u16_at(bytes, l)?;
    let n = u16_at(bytes, l + 4)? as usize;
    let mut out = Vec::with_capacity(n);
    for s in 0..n {
        let mut sub = l + u16_at(bytes, l + 6 + 2 * s)? as usize;
        let mut sub_kind = kind;
        if kind == extension {
            sub_kind = u16_at(bytes, sub + 2)?;
            sub += u32_at(bytes, sub + 4)? as usize;
        }
        out.push((sub_kind, sub));
    }
    Some(out)
}

/// `GPOS` at `gpos`; `store` resolves variation deltas when the font has them.
pub(crate) fn parse(bytes: &[u8], gpos: usize, store: Option<&ItemVariationStore>) -> Option<GposKerning> {
    let mut out = GposKerning::default();
    for index in feature_lookups(bytes, gpos, &[b"kern"])? {
        let Some(subtables) = lookup_subtables(bytes, gpos, index, 9) else { continue };
        let mut tables = Vec::new();
        for (kind, sub) in subtables {
            if kind != 2 {
                continue;
            }
            if let Some(t) = pair_table(bytes, sub, store) {
                tables.push(t);
            }
        }
        if !tables.is_empty() {
            out.lookups.push(tables);
        }
    }
    Some(out)
}

fn pair_table(b: &[u8], sub: usize, store: Option<&ItemVariationStore>) -> Option<PairTable> {
    let format = u16_at(b, sub)?;
    let coverage = parse_coverage(b, sub + u16_at(b, sub + 2)? as usize)?;
    let vf1 = u16_at(b, sub + 4)?;
    let vf2 = u16_at(b, sub + 6)?;
    let (size1, size2) = (vf1.count_ones() as usize * 2, vf2.count_ones() as usize * 2);
    match format {
        1 => {
            let sets = u16_at(b, sub + 8)? as usize;
            let mut pairs = Vec::with_capacity(sets);
            for i in 0..sets {
                let set = sub + u16_at(b, sub + 10 + 2 * i)? as usize;
                let n = u16_at(b, set)? as usize;
                let mut map = HashMap::with_capacity(n);
                let rec = 2 + size1 + size2;
                for k in 0..n {
                    let r = set + 2 + k * rec;
                    let second = u16_at(b, r)?;
                    // A pair set's device offsets are from the pair set.
                    let v = x_advance(b, r + 2, vf1, set, store)?;
                    map.insert(second, v);
                }
                pairs.push(map);
            }
            Some(PairTable::Glyphs { coverage, pairs })
        }
        2 => {
            let class1 = parse_class_def(b, sub + u16_at(b, sub + 8)? as usize)?;
            let class2 = parse_class_def(b, sub + u16_at(b, sub + 10)? as usize)?;
            let c1n = u16_at(b, sub + 12)? as usize;
            let c2n = u16_at(b, sub + 14)? as usize;
            let rec = size1 + size2;
            let mut values = Vec::with_capacity(c1n * c2n);
            for c1 in 0..c1n {
                for c2 in 0..c2n {
                    let r = sub + 16 + (c1 * c2n + c2) * rec;
                    values.push(x_advance(b, r, vf1, sub, store)?);
                }
            }
            Some(PairTable::Classes { coverage, class1, class2, class2_count: c2n, values })
        }
        _ => None,
    }
}

/// A value record's XAdvance plus its variation delta, in font units.
fn x_advance(b: &[u8], record: usize, format: u16, base: usize, store: Option<&ItemVariationStore>) -> Option<f32> {
    if format & 0x0004 == 0 {
        return Some(0.0);
    }
    // Fields before XAdvance: XPlacement, YPlacement.
    let at = record + 2 * (format & 0x0003).count_ones() as usize;
    let mut v = u16_at(b, at)? as i16 as f32;
    if format & 0x0040 != 0 {
        // XAdvDevice: after XAdvance, YAdvance, XPlaDevice, YPlaDevice.
        let dev_at = record + 2 * (format & 0x003F).count_ones() as usize;
        let dev = u16_at(b, dev_at)? as usize;
        if dev != 0 {
            if let Some(store) = store {
                let d = base + dev;
                // A VariationIndex table (deltaFormat 0x8000); hinting device
                // tables are for one ppem and are not applied.
                if u16_at(b, d + 4)? == 0x8000 {
                    v += store.delta(u16_at(b, d)?, u16_at(b, d + 2)?);
                }
            }
        }
    }
    Some(v)
}

/// A coverage table: glyph id to coverage index.
pub(crate) fn parse_coverage(b: &[u8], off: usize) -> Option<HashMap<u16, u16>> {
    let format = u16_at(b, off)?;
    let n = u16_at(b, off + 2)? as usize;
    let mut out = HashMap::new();
    match format {
        1 => {
            for i in 0..n {
                out.insert(u16_at(b, off + 4 + 2 * i)?, i as u16);
            }
        }
        2 => {
            for i in 0..n {
                let r = off + 4 + 6 * i;
                let (start, end, index) = (u16_at(b, r)?, u16_at(b, r + 2)?, u16_at(b, r + 4)?);
                for g in start..=end.max(start) {
                    out.insert(g, index + (g - start));
                }
            }
        }
        _ => return None,
    }
    Some(out)
}

/// A class definition: glyph id to class, absent glyphs class 0.
pub(crate) fn parse_class_def(b: &[u8], off: usize) -> Option<HashMap<u16, u16>> {
    let format = u16_at(b, off)?;
    let mut out = HashMap::new();
    match format {
        1 => {
            let start = u16_at(b, off + 2)?;
            let n = u16_at(b, off + 4)? as usize;
            for i in 0..n {
                let c = u16_at(b, off + 6 + 2 * i)?;
                if c != 0 {
                    out.insert(start.wrapping_add(i as u16), c);
                }
            }
        }
        2 => {
            let n = u16_at(b, off + 2)? as usize;
            for i in 0..n {
                let r = off + 4 + 6 * i;
                let (start, end, c) = (u16_at(b, r)?, u16_at(b, r + 2)?, u16_at(b, r + 4)?);
                if c != 0 {
                    for g in start..=end.max(start) {
                        out.insert(g, c);
                    }
                }
            }
        }
        _ => return None,
    }
    Some(out)
}

/// `GDEF`'s item variation store offset (version 1.3), absolute.
pub(crate) fn gdef_var_store(b: &[u8], gdef: usize) -> Option<usize> {
    if u16_at(b, gdef)? != 1 || u16_at(b, gdef + 2)? < 3 {
        return None;
    }
    let off = u32_at(b, gdef + 14)? as usize;
    (off != 0).then_some(gdef + off)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coverage_ranges_number_glyphs_from_their_start_index() {
        // Format 2: glyphs 10..=12 from index 0, 20..=20 from index 3.
        let b = [0, 2, 0, 2, 0, 10, 0, 12, 0, 0, 0, 20, 0, 20, 0, 3];
        let c = parse_coverage(&b, 0).unwrap();
        assert_eq!(c.get(&11), Some(&1));
        assert_eq!(c.get(&20), Some(&3));
        assert_eq!(c.get(&13), None);
    }

    #[test]
    fn a_class_pair_applies_even_when_its_value_is_zero() {
        // A shaper stops at the first subtable covering the left glyph, so a
        // later subtable must not add to a pair the first one decided.
        let coverage: HashMap<u16, u16> = [(1, 0)].into();
        let first = PairTable::Classes {
            coverage: coverage.clone(),
            class1: HashMap::new(),
            class2: HashMap::new(),
            class2_count: 1,
            values: vec![0.0],
        };
        let second = PairTable::Glyphs { coverage, pairs: vec![[(2, -50.0)].into()] };
        let k = GposKerning { lookups: vec![vec![first, second]] };
        assert_eq!(k.kern(1, 2), 0.0);
    }
}
