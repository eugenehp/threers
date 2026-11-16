//! Text shaping as HarfBuzz does it for horizontal text in Latin and the
//! other simple scripts, with the font's advances as FreeType gives them
//! unhinted (`hb_ft_font_create`): characters to glyphs through `cmap`,
//! the default GSUB features (`ccmp`, `locl`, `rlig`, `liga`, `clig`,
//! `calt`, `rclt`) in lookup order, then GPOS `kern`, `mark` and `mkmk`, or
//! the `kern` table when GPOS has no kerning.

use crate::fixed::mul_div;
use crate::sfnt::{Face, Reader};
use crate::size::Size;

/// One shaped glyph, its place in 26.6 pixels; `cluster` is the byte
/// offset of its first character in the text.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GlyphPosition {
    pub glyph: u16,
    pub cluster: u32,
    pub x_advance: i32,
    pub y_advance: i32,
    pub x_offset: i32,
    pub y_offset: i32,
}

/// Why text cannot be shaped as HarfBuzz would.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unsupported(pub &'static str);

/// A glyph in the buffer with its GDEF class.
#[derive(Clone, Copy, Debug)]
struct Info {
    glyph: u16,
    cluster: u32,
    class: u16,
}

/// GDEF glyph classes.
const BASE: u16 = 1;
const LIGATURE: u16 = 2;
const MARK: u16 = 3;

/// The features HarfBuzz enables by default for horizontal text, by
/// stage: GSUB then GPOS.
const GSUB_FEATURES: [&[u8; 4]; 8] = [b"rvrn", b"ccmp", b"locl", b"rlig", b"liga", b"clig", b"calt", b"rclt"];
const GPOS_FEATURES: [&[u8; 4]; 5] = [b"kern", b"mark", b"mkmk", b"curs", b"dist"];

/// A layout table (GSUB or GPOS) of a face.
struct Layout<'a> {
    r: Reader<'a>,
    scripts: usize,
    features: usize,
    lookups: usize,
}

impl<'a> Layout<'a> {
    fn new(data: &'a [u8]) -> Option<Layout<'a>> {
        let r = Reader::new(data);
        Some(Layout { r, scripts: r.u16(4).ok()? as usize, features: r.u16(6).ok()? as usize, lookups: r.u16(8).ok()? as usize })
    }

    /// The default language system of the script HarfBuzz picks for Latin
    /// text: `latn`, else `DFLT`, else `dflt`.
    fn langsys(&self) -> Option<usize> {
        let r = self.r;
        let n = r.u16(self.scripts).ok()? as usize;
        let mut found = None;

        for want in [b"latn", b"DFLT", b"dflt"] {
            for i in 0..n {
                let at = self.scripts + 2 + 6 * i;

                if r.slice(at, 4).ok()? == want {
                    found = Some(self.scripts + r.u16(at + 4).ok()? as usize);
                    break;
                }
            }

            if found.is_some() {
                break;
            }
        }

        let script = found?;
        let default = r.u16(script).ok()? as usize;
        (default != 0).then_some(script + default)
    }

    /// The lookups of the features asked for, in ascending order.
    fn lookups_for(&self, tags: &[&[u8; 4]]) -> Vec<u16> {
        let r = self.r;
        let Some(ls) = self.langsys() else { return Vec::new() };
        let mut out = Vec::new();
        let count = r.u16(ls + 4).unwrap_or(0) as usize;
        let mut indices: Vec<usize> = (0..count).filter_map(|i| r.u16(ls + 6 + 2 * i).ok().map(usize::from)).collect();

        if let Ok(req) = r.u16(ls + 2)
            && req != 0xFFFF {
                indices.push(usize::from(req));
            }

        for f in indices {
            let at = self.features + 2 + 6 * f;

            let Ok(tag) = r.slice(at, 4) else { continue };

            if !tags.iter().any(|t| t.as_slice() == tag) {
                continue;
            }

            let Ok(off) = r.u16(at + 4) else { continue };
            let feature = self.features + off as usize;
            let n = r.u16(feature + 2).unwrap_or(0) as usize;

            for i in 0..n {
                if let Ok(l) = r.u16(feature + 4 + 2 * i) {
                    out.push(l);
                }
            }
        }

        out.sort_unstable();
        out.dedup();
        out
    }

    /// A lookup's type, flags, mark filtering set and subtables, extension
    /// subtables resolved.
    fn lookup(&self, index: u16, extension: u16) -> Option<(u16, u16, Option<u16>, Vec<usize>)> {
        let r = self.r;
        let lookup = self.lookups + r.u16(self.lookups + 2 + 2 * index as usize).ok()? as usize;
        let mut kind = r.u16(lookup).ok()?;
        let flags = r.u16(lookup + 2).ok()?;
        let n = r.u16(lookup + 4).ok()? as usize;
        let mut subtables = Vec::with_capacity(n);

        for i in 0..n {
            let mut st = lookup + r.u16(lookup + 6 + 2 * i).ok()? as usize;

            if kind == extension {
                let real = r.u16(st + 2).ok()?;
                st += r.u32(st + 4).ok()? as usize;
                subtables.push(st);
                kind = real;
                continue;
            }

            subtables.push(st);
        }

        if let Some(&first) = subtables.first()
            && r.u16(lookup).ok()? == extension {
                let real = r.u16(lookup + 6).ok().map(|o| lookup + o as usize).and_then(|st| r.u16(st + 2).ok())?;
                kind = real;
                let _ = first;
            }

        let mark_set = if flags & 0x10 != 0 { r.u16(lookup + 6 + 2 * n).ok() } else { None };
        Some((kind, flags, mark_set, subtables))
    }
}

/// The index of a glyph in a coverage table.
fn coverage(r: Reader, at: usize, glyph: u16) -> Option<usize> {
    match r.u16(at).ok()? {
        1 => {
            let n = r.u16(at + 2).ok()? as usize;
            let (mut lo, mut hi) = (0, n);

            while lo < hi {
                let mid = (lo + hi) / 2;
                let g = r.u16(at + 4 + 2 * mid).ok()?;

                match g.cmp(&glyph) {
                    std::cmp::Ordering::Equal => return Some(mid),
                    std::cmp::Ordering::Less => lo = mid + 1,
                    std::cmp::Ordering::Greater => hi = mid,
                }
            }

            None
        }
        2 => {
            let n = r.u16(at + 2).ok()? as usize;

            for i in 0..n {
                let rec = at + 4 + 6 * i;
                let (start, end) = (r.u16(rec).ok()?, r.u16(rec + 2).ok()?);

                if (start..=end).contains(&glyph) {
                    return Some(r.u16(rec + 4).ok()? as usize + (glyph - start) as usize);
                }
            }

            None
        }
        _ => None,
    }
}

/// A glyph's class in a class definition table.
fn class_of(r: Reader, at: usize, glyph: u16) -> u16 {
    let lookup = || -> Option<u16> {
        match r.u16(at).ok()? {
            1 => {
                let start = r.u16(at + 2).ok()?;
                let n = r.u16(at + 4).ok()?;

                if glyph >= start && glyph - start < n {
                    r.u16(at + 6 + 2 * (glyph - start) as usize).ok()
                } else {
                    Some(0)
                }
            }
            2 => {
                let n = r.u16(at + 2).ok()? as usize;

                for i in 0..n {
                    let rec = at + 4 + 6 * i;

                    if (r.u16(rec).ok()?..=r.u16(rec + 2).ok()?).contains(&glyph) {
                        return r.u16(rec + 4).ok();
                    }
                }

                Some(0)
            }
            _ => Some(0),
        }
    };

    lookup().unwrap_or(0)
}

/// The shaper's view of a face: GDEF, GSUB, GPOS and `kern`.
struct Shaper<'a> {
    face: &'a Face,
    gdef: Option<Reader<'a>>,
}

impl<'a> Shaper<'a> {
    fn glyph_class(&self, glyph: u16) -> u16 {
        let Some(g) = self.gdef else { return 0 };
        let off = g.u16(4).unwrap_or(0) as usize;
        if off == 0 { 0 } else { class_of(g, off, glyph) }
    }

    fn mark_attach_class(&self, glyph: u16) -> u16 {
        let Some(g) = self.gdef else { return 0 };
        let off = g.u16(10).unwrap_or(0) as usize;
        if off == 0 { 0 } else { class_of(g, off, glyph) }
    }

    fn in_mark_set(&self, set: u16, glyph: u16) -> bool {
        let Some(g) = self.gdef else { return false };

        if g.u32(0).unwrap_or(0) < 0x0001_0002 {
            return false;
        }

        let off = g.u16(12).unwrap_or(0) as usize;

        if off == 0 {
            return false;
        }

        let Ok(cov) = g.u32(off + 4 + 4 * set as usize) else { return false };
        coverage(g, off + cov as usize, glyph).is_some()
    }

    /// Whether a lookup's flags skip a glyph.
    fn ignored(&self, info: &Info, flags: u16, mark_set: Option<u16>) -> bool {
        match info.class {
            BASE if flags & 0x2 != 0 => true,
            LIGATURE if flags & 0x4 != 0 => true,
            MARK => {
                if flags & 0x8 != 0 {
                    return true;
                }

                if let Some(set) = mark_set {
                    return !self.in_mark_set(set, info.glyph);
                }

                let attach = flags >> 8;
                attach != 0 && self.mark_attach_class(info.glyph) != attach
            }
            _ => false,
        }
    }

    /// The next glyph the lookup sees after `i`, if any.
    fn next(&self, buf: &[Info], mut i: usize, flags: u16, mark_set: Option<u16>) -> Option<usize> {
        loop {
            i += 1;

            if i >= buf.len() {
                return None;
            }

            if !self.ignored(&buf[i], flags, mark_set) {
                return Some(i);
            }
        }
    }

    fn prev(&self, buf: &[Info], mut i: usize, flags: u16, mark_set: Option<u16>) -> Option<usize> {
        loop {
            if i == 0 {
                return None;
            }

            i -= 1;

            if !self.ignored(&buf[i], flags, mark_set) {
                return Some(i);
            }
        }
    }

    // --- GSUB --------------------------------------------------------

    /// Runs a GSUB lookup over the buffer.
    fn gsub_lookup(&self, gsub: &Layout, index: u16, buf: &mut Vec<Info>) -> Result<(), Unsupported> {
        let Some((kind, flags, mark_set, subtables)) = gsub.lookup(index, 7) else { return Ok(()) };
        let mut i = 0;

        while i < buf.len() {
            if self.ignored(&buf[i], flags, mark_set) {
                i += 1;
                continue;
            }

            let mut applied = None;

            for &st in &subtables {
                applied = self.gsub_apply(gsub, kind, flags, mark_set, st, buf, i)?;

                if applied.is_some() {
                    break;
                }
            }

            i = applied.unwrap_or(i + 1);
        }

        Ok(())
    }

    /// One subtable at `i`; where to go on from when it applied.
    #[allow(clippy::too_many_arguments)]
    fn gsub_apply(
        &self,
        gsub: &Layout,
        kind: u16,
        flags: u16,
        mark_set: Option<u16>,
        st: usize,
        buf: &mut Vec<Info>,
        i: usize,
    ) -> Result<Option<usize>, Unsupported> {
        let r = gsub.r;
        let glyph = buf[i].glyph;
        let format = r.u16(st).unwrap_or(0);
        let cov = |at: usize| r.u16(at).ok().and_then(|o| coverage(r, st + o as usize, glyph));

        match kind {
            1 => {
                let Some(c) = cov(st + 2) else { return Ok(None) };

                let new = match format {
                    1 => glyph.wrapping_add(r.u16(st + 4).unwrap_or(0)),
                    2 => r.u16(st + 6 + 2 * c).unwrap_or(glyph),
                    _ => return Ok(None),
                };

                buf[i].glyph = new;
                buf[i].class = self.glyph_class(new);
                Ok(Some(i + 1))
            }
            2 => {
                let Some(c) = cov(st + 2) else { return Ok(None) };
                let seq = st + r.u16(st + 6 + 2 * c).unwrap_or(0) as usize;
                let n = r.u16(seq).unwrap_or(0) as usize;

                if n == 0 {
                    buf.remove(i);
                    return Ok(Some(i));
                }

                let template = buf[i];
                let glyphs: Vec<u16> = (0..n).filter_map(|k| r.u16(seq + 2 + 2 * k).ok()).collect();
                buf.splice(i..=i, glyphs.iter().map(|&g| Info { glyph: g, class: self.glyph_class(g), ..template }));
                Ok(Some(i + n))
            }
            4 => {
                let Some(c) = cov(st + 2) else { return Ok(None) };
                let set = st + r.u16(st + 6 + 2 * c).unwrap_or(0) as usize;
                let n = r.u16(set).unwrap_or(0) as usize;

                for k in 0..n {
                    let lig = set + r.u16(set + 2 + 2 * k).unwrap_or(0) as usize;
                    let lig_glyph = r.u16(lig).unwrap_or(0);
                    let comps = r.u16(lig + 2).unwrap_or(0) as usize;
                    let mut positions = vec![i];
                    let mut at = i;
                    let mut ok = true;

                    for m in 1..comps {
                        match self.next(buf, at, flags, mark_set) {
                            Some(j) if buf[j].glyph == r.u16(lig + 4 + 2 * (m - 1)).unwrap_or(0) => {
                                positions.push(j);
                                at = j;
                            }
                            _ => {
                                ok = false;
                                break;
                            }
                        }
                    }

                    if !ok {
                        continue;
                    }

                    if positions.windows(2).any(|w| w[1] != w[0] + 1) {
                        return Err(Unsupported("a ligature over skipped marks"));
                    }

                    let cluster = positions.iter().map(|&p| buf[p].cluster).min().unwrap_or(buf[i].cluster);
                    let last = *positions.last().unwrap_or(&i);
                    let lig_class = self.glyph_class(lig_glyph);
                    buf.splice(i..=last, std::iter::once(Info { glyph: lig_glyph, cluster, class: if lig_class == 0 { LIGATURE } else { lig_class } }));

                    // HarfBuzz merges the clusters of what follows into the ligature's
                    return Ok(Some(i + 1));
                }

                Ok(None)
            }
            6 => self.chain_context(gsub, st, format, flags, mark_set, buf, i, true),
            5 => Err(Unsupported("GSUB contextual substitutions")),
            8 => Err(Unsupported("GSUB reverse chaining substitutions")),
            _ => Ok(None),
        }
    }

    /// A chaining contextual lookup (GSUB 6, GPOS 8) at `i`.
    #[allow(clippy::too_many_arguments)]
    fn chain_context(
        &self,
        table: &Layout,
        st: usize,
        format: u16,
        flags: u16,
        mark_set: Option<u16>,
        buf: &mut Vec<Info>,
        i: usize,
        gsub: bool,
    ) -> Result<Option<usize>, Unsupported> {
        let r = table.r;

        if format != 3 {
            // formats 1 and 2: glyph and class sequences
            let glyph = buf[i].glyph;
            let Some(c) = r.u16(st + 2).ok().and_then(|o| coverage(r, st + o as usize, glyph)) else { return Ok(None) };
            let _ = c;
            return Err(Unsupported("chaining contextual lookups of format 1 or 2"));
        }

        let mut at = st + 2;
        let read_seq = |at: &mut usize| -> Vec<usize> {
            let n = r.u16(*at).unwrap_or(0) as usize;
            let v = (0..n).map(|k| st + r.u16(*at + 2 + 2 * k).unwrap_or(0) as usize).collect();
            *at += 2 + 2 * n;
            v
        };

        let backtrack = read_seq(&mut at);
        let input = read_seq(&mut at);
        let lookahead = read_seq(&mut at);

        if input.is_empty() || coverage(r, input[0], buf[i].glyph).is_none() {
            return Ok(None);
        }

        let mut positions = vec![i];
        let mut p = i;

        for &cov in &input[1..] {
            match self.next(buf, p, flags, mark_set) {
                Some(j) if coverage(r, cov, buf[j].glyph).is_some() => {
                    positions.push(j);
                    p = j;
                }
                _ => return Ok(None),
            }
        }

        let mut q = p;

        for &cov in &lookahead {
            match self.next(buf, q, flags, mark_set) {
                Some(j) if coverage(r, cov, buf[j].glyph).is_some() => q = j,
                _ => return Ok(None),
            }
        }

        let mut b = i;

        for &cov in &backtrack {
            match self.prev(buf, b, flags, mark_set) {
                Some(j) if coverage(r, cov, buf[j].glyph).is_some() => b = j,
                _ => return Ok(None),
            }
        }

        let n = r.u16(at).unwrap_or(0) as usize;

        if n > 0 {
            if !gsub {
                return Err(Unsupported("GPOS chaining contextual positioning"));
            }

            for k in 0..n {
                let seq_index = r.u16(at + 2 + 4 * k).unwrap_or(0) as usize;
                let lookup = r.u16(at + 4 + 4 * k).unwrap_or(0);
                let Some(&pos) = positions.get(seq_index) else { continue };
                let Some((kind, f2, ms2, subs)) = table.lookup(lookup, 7) else { continue };

                if !matches!(kind, 1) {
                    return Err(Unsupported("a nested contextual lookup other than a single substitution"));
                }

                for &s in &subs {
                    if self.gsub_apply(table, kind, f2, ms2, s, buf, pos)?.is_some() {
                        break;
                    }
                }
            }
        }

        Ok(Some(p + 1))
    }

    // --- GPOS ----------------------------------------------------------

    /// A value record at `at` of the format: x placement, y placement, x
    /// advance, y advance in font units; and its size.
    fn value_record(r: Reader, at: usize, format: u16) -> ([i32; 4], usize) {
        let mut v = [0; 4];
        let mut p = at;

        for (bit, slot) in [(0x1, 0), (0x2, 1), (0x4, 2), (0x8, 3)] {
            if format & bit != 0 {
                v[slot] = i32::from(r.i16(p).unwrap_or(0));
                p += 2;
            }
        }

        for bit in [0x10, 0x20, 0x40, 0x80] {
            if format & bit != 0 {
                p += 2;
            }
        }

        (v, p - at)
    }

    fn gpos_lookup(&self, gpos: &Layout, index: u16, buf: &[Info], pos: &mut [GlyphPosition], em: &dyn Fn(i32) -> i32) -> Result<(), Unsupported> {
        let Some((kind, flags, mark_set, subtables)) = gpos.lookup(index, 9) else { return Ok(()) };
        let r = gpos.r;

        for i in 0..buf.len() {
            if self.ignored(&buf[i], flags, mark_set) {
                continue;
            }

            for &st in &subtables {
                let format = r.u16(st).unwrap_or(0);
                let glyph = buf[i].glyph;
                let Some(c) = r.u16(st + 2).ok().and_then(|o| coverage(r, st + o as usize, glyph)) else { continue };

                let applied = match kind {
                    1 => {
                        let vf = r.u16(st + 4).unwrap_or(0);
                        let (v, size) = match format {
                            1 => Self::value_record(r, st + 6, vf),
                            _ => {
                                let (_, size) = Self::value_record(r, st + 8, vf);
                                Self::value_record(r, st + 8 + c * size, vf)
                            }
                        };
                        let _ = size;
                        Self::apply_value(&mut pos[i], v, em);
                        true
                    }
                    2 => {
                        let Some(j) = self.next(buf, i, flags, mark_set) else { continue };
                        let (vf1, vf2) = (r.u16(st + 4).unwrap_or(0), r.u16(st + 6).unwrap_or(0));
                        let (_, s1) = Self::value_record(r, 0, vf1);
                        let (_, s2) = Self::value_record(r, 0, vf2);

                        let found = match format {
                            1 => {
                                let set = st + r.u16(st + 10 + 2 * c).unwrap_or(0) as usize;
                                let n = r.u16(set).unwrap_or(0) as usize;
                                let rec = 2 + s1 + s2;
                                (0..n)
                                    .map(|k| set + 2 + k * rec)
                                    .find(|&at| r.u16(at).ok() == Some(buf[j].glyph))
                                    .map(|at| (Self::value_record(r, at + 2, vf1).0, Self::value_record(r, at + 2 + s1, vf2).0))
                            }
                            _ => {
                                let c1 = class_of(r, st + r.u16(st + 8).unwrap_or(0) as usize, glyph) as usize;
                                let c2 = class_of(r, st + r.u16(st + 10).unwrap_or(0) as usize, buf[j].glyph) as usize;
                                let n2 = r.u16(st + 14).unwrap_or(0) as usize;
                                let at = st + 16 + (c1 * n2 + c2) * (s1 + s2);
                                Some((Self::value_record(r, at, vf1).0, Self::value_record(r, at + s1, vf2).0))
                            }
                        };

                        match found {
                            Some((v1, v2)) => {
                                Self::apply_value(&mut pos[i], v1, em);
                                Self::apply_value(&mut pos[j], v2, em);
                                true
                            }
                            None => false,
                        }
                    }
                    4 | 6 => return Err(Unsupported("mark attachment")),
                    _ => return Err(Unsupported("a GPOS lookup type")),
                };

                if applied {
                    break;
                }
            }
        }

        Ok(())
    }

    fn apply_value(p: &mut GlyphPosition, v: [i32; 4], em: &dyn Fn(i32) -> i32) {
        p.x_offset += em(v[0]);
        p.y_offset += em(v[1]);
        p.x_advance += em(v[2]);
        p.y_advance += em(v[3]);
    }

    /// The `kern` table's pair value (format 0 subtables, horizontal).
    fn kern_pair(&self, left: u16, right: u16) -> Option<i32> {
        let kern = self.face.table(b"kern")?;
        let r = Reader::new(kern);

        if r.u16(0).ok()? != 0 {
            return None;
        }

        let n = r.u16(2).ok()? as usize;
        let mut at = 4;
        let mut total = None;

        for _ in 0..n {
            let len = r.u16(at + 2).ok()? as usize;
            let coverage = r.u16(at + 4).ok()?;

            // horizontal, format 0, not minimum or cross-stream
            if coverage & 1 == 1 && coverage >> 8 == 0 && coverage & 0x6 == 0 {
                let pairs = r.u16(at + 6).ok()? as usize;
                let key = (u32::from(left) << 16) | u32::from(right);
                let (mut lo, mut hi) = (0, pairs);

                while lo < hi {
                    let mid = (lo + hi) / 2;
                    let rec = at + 14 + 6 * mid;
                    let k = r.u32(rec).ok()?;

                    match k.cmp(&key) {
                        std::cmp::Ordering::Equal => {
                            let v = i32::from(r.i16(rec + 4).ok()?);
                            total = Some(if coverage & 0x8 != 0 { v } else { total.unwrap_or(0) + v });
                            break;
                        }
                        std::cmp::Ordering::Less => lo = mid + 1,
                        std::cmp::Ordering::Greater => hi = mid,
                    }
                }
            }

            at += len;
        }

        total
    }
}

/// Combining marks (which HarfBuzz may compose or position) and default
/// ignorable characters (which it hides).
fn is_mark_or_ignorable(c: u32) -> bool {
    matches!(c,
        0x300..=0x36F | 0x483..=0x489 | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF | 0x20D0..=0x20FF | 0xFE20..=0xFE2F
        | 0xAD | 0x61C | 0x115F..=0x1160 | 0x17B4..=0x17B5 | 0x180B..=0x180F | 0x200B..=0x200F
        | 0x2028..=0x202E | 0x2060..=0x206F | 0x3164 | 0xFE00..=0xFE0F | 0xFEFF | 0xFFA0 | 0xFFF0..=0xFFF8)
}

/// Characters HarfBuzz shapes left to right with its default shaper:
/// Latin, Greek, Cyrillic and the symbol and punctuation blocks.
fn is_simple_script(c: u32) -> bool {
    matches!(c,
        0x20..=0x7E | 0xA0..=0x52F | 0x1D00..=0x1FFF | 0x2000..=0x2BFF | 0x2C60..=0x2C7F | 0x2DE0..=0x2E7F
        | 0xA640..=0xA69F | 0xA720..=0xA7FF | 0xAB30..=0xAB6F | 0xFB00..=0xFB06 | 0xFFFD)
}

/// `hb_shape` of a line of text with a FreeType font at `size`
/// (`hb_ft_font_create_referenced`, the default load flags with no
/// hinting): the glyphs and their places in 26.6.
pub fn shape(face: &Face, size: &Size, text: &str) -> Result<Vec<GlyphPosition>, Unsupported> {
    let shaper = Shaper { face, gdef: face.table(b"GDEF").map(Reader::new) };

    // characters to glyphs, each its byte offset as its cluster
    let mut buf: Vec<Info> = Vec::with_capacity(text.len());

    for (offset, ch) in text.char_indices() {
        let c = ch as u32;

        if is_mark_or_ignorable(c) {
            return Err(Unsupported("combining marks, default ignorables or variation selectors"));
        }

        if !is_simple_script(c) {
            return Err(Unsupported("characters of scripts with shapers of their own"));
        }

        let glyph = face.glyph_index(c);
        buf.push(Info { glyph, cluster: offset as u32, class: shaper.glyph_class(glyph) });
    }

    // GSUB: rvrn in a stage of its own, then the rest, each in lookup order
    if let Some(gsub) = face.table(b"GSUB").and_then(Layout::new) {
        for stage in [&GSUB_FEATURES[..1], &GSUB_FEATURES[1..]] {
            for l in gsub.lookups_for(stage) {
                shaper.gsub_lookup(&gsub, l, &mut buf)?;
            }
        }
    }

    // advances: hb_ft's, FreeType's unhinted advance in 16.16, rounded
    let xs = size.metrics.x_scale;
    let mut pos: Vec<GlyphPosition> = buf
        .iter()
        .map(|info| {
            let (advance, _) = face.h_metrics(info.glyph);
            let v = mul_div(i64::from(advance), xs, 64);
            GlyphPosition { glyph: info.glyph, cluster: info.cluster, x_advance: ((v + (1 << 9)) >> 10) as i32, ..Default::default() }
        })
        .collect();

    // the font's scale as hb_ft sets it, and HarfBuzz's em_mult
    let upem = i64::from(face.units_per_em);
    let hb_scale = ((xs as u64 * upem as u64 + (1 << 15)) >> 16) as i64;
    let mult = (hb_scale << 16) / upem;
    let em = move |v: i32| ((i64::from(v) * mult + 32768) >> 16) as i32;

    // GPOS, or the kern table when GPOS has no kerning
    let gpos = face.table(b"GPOS").and_then(Layout::new);
    let gpos_kern = gpos.as_ref().is_some_and(|g| !g.lookups_for(&[b"kern"]).is_empty());

    if let Some(gpos) = &gpos {
        for l in gpos.lookups_for(&GPOS_FEATURES) {
            shaper.gpos_lookup(gpos, l, &buf, &mut pos, &em)?;
        }
    }

    if !gpos_kern && face.table(b"kern").is_some() {
        // hb_kern_machine_t: the pair's value split, the second half on the
        // right glyph's advance and offset
        for i in 0..buf.len() {
            if buf[i].class == MARK {
                continue;
            }

            let Some(j) = (i + 1..buf.len()).find(|&j| buf[j].class != MARK) else { break };
            let Some(kern) = shaper.kern_pair(buf[i].glyph, buf[j].glyph) else { continue };
            let kern = em(kern);

            if kern == 0 {
                continue;
            }

            let k1 = kern >> 1;
            let k2 = kern - k1;
            pos[i].x_advance += k1;
            pos[j].x_advance += k2;
            pos[j].x_offset += k2;
        }
    }

    // zero the advances of marks (by GDEF, after positioning)
    for (info, p) in buf.iter().zip(&mut pos) {
        if info.class == MARK {
            p.x_advance = 0;
            p.y_advance = 0;
        }
    }

    Ok(pos)
}
