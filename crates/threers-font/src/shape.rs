//! Text shaping as HarfBuzz does it for horizontal text, with the font's
//! advances as FreeType gives them unhinted (`hb_ft_font_create`):
//! characters to glyphs through `cmap`, the default GSUB features (`ccmp`,
//! `locl`, `rlig`, `liga`, `clig`, `calt`, `rclt`) in lookup order, then
//! GPOS `kern`, `mark`, `mkmk`, `abvm` and `blwm`, or the `kern` table when
//! GPOS has no kerning. Lookups of every type are applied (contextual and
//! chaining ones in all their formats, but cursive attachment).
//!
//! The scripts with shapers of their own are shaped as HarfBuzz's are:
//! Arabic and the other joining scripts (Syriac, N'Ko, Mongolian…) in
//! their forms, with cursive attachment; Hebrew with its points composed
//! where the font wants; the Indic scripts, Khmer and Myanmar syllable by
//! syllable, reordered ([`indic`]); the scripts of the Universal Shaping
//! Engine (Javanese, Balinese, Tai Tham…, [`useng`]); Thai and Lao with
//! SARA AM split. Other scripts take the default features, accents
//! composed with Unicode's compositions where the font has the whole. A
//! font with no positioning for its marks has them placed by its glyphs'
//! boxes; a font with only Apple's `morx` is shaped by it ([`aat`]), its
//! kerning from Apple's `kern`.

mod aat;
mod compose;
mod indic;
mod joining;
mod use_table;
mod useng;

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
    /// The Arabic joining form (a `FORM_*`), 0 for none: which of the
    /// form features may change this glyph.
    mask: u8,
    /// The features that may change it, as bits (all of them but where a
    /// shaper says otherwise).
    fmask: u32,
    /// What an Indic shaper made of its character: its category, its
    /// place in the syllable.
    cat: u8,
    pos: u8,
    /// Made by a ligature; made by a multiple substitution.
    ligated: bool,
    multiplied: bool,
    /// Not seen (a joiner that has done its work): no advance.
    hidden: bool,
    /// Changed by a substitution (of any kind) since last cleared.
    substituted: bool,
    /// The character's combining class as the fallback for placing marks
    /// sees it (0 for what is not a mark).
    ccc: u8,
    /// HarfBuzz's ligature properties: the ligature a glyph belongs to (0
    /// none); for a mark in it, the component it is on (from 1); for the
    /// ligature itself, how many components it has.
    lig_id: u8,
    lig_comp: u8,
    lig_comps: u8,
}

impl Default for Info {
    fn default() -> Self {
        Info { glyph: 0, cluster: 0, class: 0, mask: 0, fmask: u32::MAX, cat: 0, pos: 0, ligated: false, multiplied: false, hidden: false, substituted: false, ccc: 0, lig_id: 0, lig_comp: 0, lig_comps: 0 }
    }
}

impl Info {
    /// How many components it stands for (1 unless it is a ligature).
    fn components(&self) -> u8 {
        if self.class == LIGATURE && self.lig_id != 0 && self.lig_comps > 0 { self.lig_comps } else { 1 }
    }
}

/// Arabic joining forms, as masks for `isol`, `init`, `medi` and `fina`.
const FORM_ISOL: u8 = 1;
const FORM_INIT: u8 = 2;
const FORM_MEDI: u8 = 3;
const FORM_FINA: u8 = 4;

/// GDEF glyph classes.
const BASE: u16 = 1;
const LIGATURE: u16 = 2;
const MARK: u16 = 3;

/// The features HarfBuzz enables by default for horizontal text, by
/// stage: GSUB then GPOS.
const GSUB_FEATURES: [&[u8; 4]; 8] = [b"rvrn", b"ccmp", b"locl", b"rlig", b"liga", b"clig", b"calt", b"rclt"];
const GPOS_FEATURES: [&[u8; 4]; 7] = [b"kern", b"mark", b"mkmk", b"curs", b"dist", b"abvm", b"blwm"];

/// A layout table (GSUB or GPOS) of a face.
struct Layout<'a> {
    r: Reader<'a>,
    scripts: usize,
    features: usize,
    lookups: usize,
    /// The OpenType script tag the text is in.
    script: [u8; 4],
    /// Feature tables a variable font's instance has in place of others
    /// (FeatureVariations): feature index, where its table is.
    swaps: Vec<(usize, usize)>,
}

impl<'a> Layout<'a> {
    fn new(data: &'a [u8]) -> Option<Layout<'a>> {
        let r = Reader::new(data);
        Some(Layout {
            r,
            scripts: r.u16(4).ok()? as usize,
            features: r.u16(6).ok()? as usize,
            lookups: r.u16(8).ok()? as usize,
            script: *b"latn",
            swaps: Vec::new(),
        })
    }

    /// The face's GSUB or GPOS, with the feature tables its instance has
    /// in place of others: the first FeatureVariations record whose
    /// conditions it meets (at the default instance too, as HarfBuzz).
    fn of(face: &'a Face, tag: &[u8; 4]) -> Option<Layout<'a>> {
        let mut l = Layout::new(face.table(tag)?)?;
        let r = l.r;
        if r.u16(0).ok()? != 1 || r.u16(2).ok()? < 1 {
            return Some(l);
        }
        let fv = r.u32(10).ok()? as usize;
        if fv == 0 {
            return Some(l);
        }
        let coords = face.coords();
        let coord = |axis: usize| coords.get(axis).map_or(0, |c| (c * 16384.0).round() as i32);
        let records = r.u32(fv + 4).ok()? as usize;
        for k in 0..records {
            let rec = fv + 8 + 8 * k;
            let (set, subst) = (r.u32(rec).ok()? as usize, r.u32(rec + 4).ok()? as usize);
            let met = set == 0 || {
                let set = fv + set;
                let n = r.u16(set).ok()? as usize;
                (0..n).all(|i| {
                    let c = set + r.u32(set + 2 + 4 * i).map_or(0, |o| o as usize);
                    match r.u16(c) {
                        Ok(1) => {
                            let axis = r.u16(c + 2).unwrap_or(0) as usize;
                            let (lo, hi) = (i32::from(r.i16(c + 4).unwrap_or(0)), i32::from(r.i16(c + 6).unwrap_or(0)));
                            (lo..=hi).contains(&coord(axis))
                        }
                        // conditions of kinds not known: not met
                        _ => false,
                    }
                })
            };
            if met {
                if subst != 0 {
                    let t = fv + subst;
                    let n = r.u16(t + 4).ok()? as usize;
                    l.swaps = (0..n).filter_map(|i| Some((r.u16(t + 6 + 6 * i).ok()? as usize, t + r.u32(t + 8 + 6 * i).ok()? as usize))).collect();
                }
                break;
            }
        }
        Some(l)
    }

    /// Whether the table has the script `tag` of its own.
    fn has_script(&self, tag: &[u8; 4]) -> bool {
        let r = self.r;
        let n = r.u16(self.scripts).unwrap_or(0) as usize;
        (0..n).any(|i| r.slice(self.scripts + 2 + 6 * i, 4).is_ok_and(|t| t == tag))
    }

    fn with_script(mut self, script: [u8; 4]) -> Self {
        self.script = script;
        self
    }

    /// The default language system of the script HarfBuzz picks for the
    /// text: its own (`latn` for Latin), else `DFLT`, else `dflt`.
    fn langsys(&self) -> Option<usize> {
        let r = self.r;
        let n = r.u16(self.scripts).ok()? as usize;
        let mut found = None;

        for want in [&self.script, b"DFLT", b"dflt"] {
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
        let mut out: Vec<u16> = self.lookups_tagged(tags).into_iter().map(|(l, _)| l).collect();
        out.dedup();
        out
    }

    /// The lookups of the features asked for with the feature each is
    /// for, in ascending lookup order.
    fn lookups_tagged(&self, tags: &[&[u8; 4]]) -> Vec<(u16, [u8; 4])> {
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
            let feature = self.swaps.iter().find(|s| s.0 == f).map_or(self.features + off as usize, |s| s.1);
            let n = r.u16(feature + 2).unwrap_or(0) as usize;

            let mut t = [0; 4];
            t.copy_from_slice(tag);

            for i in 0..n {
                if let Ok(l) = r.u16(feature + 4 + 2 * i) {
                    out.push((l, t));
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
        // every subtable of an extension lookup is an extension (the type
        // read from the first must not stop the rest being resolved)
        let extended = kind == extension;

        for i in 0..n {
            let mut st = lookup + r.u16(lookup + 6 + 2 * i).ok()? as usize;

            if extended {
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

/// Glyphs `start..end` made one cluster (HarfBuzz's `merge_clusters`: the
/// range grown over neighbours of its ends' clusters).
fn merge(info: &mut [Info], mut start: usize, mut end: usize) {
    if end <= start + 1 || end > info.len() {
        return;
    }
    let cluster = info[start..end].iter().map(|g| g.cluster).min().unwrap_or(0);
    if cluster != info[end - 1].cluster {
        while end < info.len() && info[end - 1].cluster == info[end].cluster {
            end += 1;
        }
    }
    if cluster != info[start].cluster {
        while start > 0 && info[start - 1].cluster == info[start].cluster {
            start -= 1;
        }
    }
    for g in &mut info[start..end] {
        g.cluster = cluster;
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

/// Fonts whose GDEF HarfBuzz ignores (it calls ordinary glyphs marks):
/// the lengths of their GDEF, GSUB and GPOS (from its `is_blocklisted`).
const GDEF_BLOCKLIST: [(usize, usize, usize); 40] = [(442, 2874, 42038), (430, 2874, 40662), (442, 2874, 39116), (430, 2874, 39374), (490, 3046, 41638), (478, 3046, 41902), (898, 12554, 46470), (910, 12566, 47732), (928, 23298, 59332), (940, 23310, 60732), (964, 23836, 60072), (976, 23832, 61456), (994, 24474, 60336), (1006, 24470, 61740), (1006, 24576, 61346), (1018, 24572, 62828), (1006, 24576, 61352), (1018, 24572, 62834), (832, 7324, 47162), (844, 7302, 45474), (180, 13054, 7254), (192, 12638, 7254), (192, 12690, 7254), (188, 248, 3852), (188, 264, 3426), (1058, 47032, 11818), (1046, 47030, 12600), (1058, 71796, 16770), (1046, 71790, 17862), (1046, 71788, 17112), (1058, 71794, 17514), (1330, 109904, 57938), (1330, 109904, 58972), (1004, 59092, 14836), (588, 5078, 14418), (588, 5078, 14238), (894, 17162, 33960), (894, 17154, 34472), (816, 7868, 17052), (816, 7868, 17138)];

/// A face's GDEF, unless HarfBuzz ignores it.
fn gdef(face: &Face) -> Option<Reader<'_>> {
    let g = face.table(b"GDEF")?;
    let len = |t: &[u8; 4]| face.table(t).map_or(0, <[u8]>::len);
    if GDEF_BLOCKLIST.contains(&(g.len(), len(b"GSUB"), len(b"GPOS"))) {
        return None;
    }
    Some(Reader::new(g))
}

/// The shaper's view of a face: GDEF, GSUB, GPOS and `kern`.
struct Shaper<'a> {
    face: &'a Face,
    gdef: Option<Reader<'a>>,
    /// The feature bits of the lookup being applied: it changes only
    /// glyphs with one of them.
    lookup_mask: std::cell::Cell<u32>,
    /// How deep nested contextual lookups are.
    depth: std::cell::Cell<u32>,
    /// Marks lose their advances (not in the Indic scripts, whose fonts
    /// give their marks the advances they mean).
    zero_marks: bool,
    /// Marks lose their advances before positioning, not after (the
    /// Universal Shaping Engine's and Myanmar's way).
    zero_early: bool,
    /// Right to left (for cursive attachment's advances).
    rtl: std::cell::Cell<bool>,
    /// What each glyph is cursively attached to, and each mark to.
    cursive: std::cell::RefCell<Vec<Option<usize>>>,
    marks: std::cell::RefCell<Vec<Option<usize>>>,
    /// Substituted by Apple's `morx` (its GSUB unused).
    aat: bool,
    /// The last ligature id given (HarfBuzz's, 1 to 7 round).
    lig_id: std::cell::Cell<u8>,
}

/// A contextual rule matched: where its input glyphs are, and its nested
/// lookups (sequence index, lookup index).
struct Matched {
    positions: Vec<usize>,
    records: Vec<(usize, u16)>,
}

impl<'a> Shaper<'a> {
    fn new(face: &'a Face) -> Shaper<'a> {
        Shaper { face, gdef: gdef(face), lookup_mask: std::cell::Cell::new(u32::MAX), depth: std::cell::Cell::new(0), zero_marks: true, zero_early: false, rtl: std::cell::Cell::new(false), cursive: Default::default(), marks: Default::default(), aat: false, lig_id: std::cell::Cell::new(0) }
    }

    /// Whether GDEF gives glyphs classes (else HarfBuzz takes them from
    /// Unicode's categories).
    fn has_classes(&self) -> bool {
        self.gdef.is_some_and(|g| g.u16(4).unwrap_or(0) != 0)
    }

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

    /// Whether the lookup being applied may change this glyph.
    fn takes(&self, info: &Info) -> bool {
        info.fmask & self.lookup_mask.get() != 0
    }

    /// The input after `i` (each glyph tested by `input`), then the
    /// lookahead and the backtrack: the input's places.
    #[allow(clippy::too_many_arguments)]
    fn match_rule(
        &self,
        buf: &[Info],
        i: usize,
        flags: u16,
        mark_set: Option<u16>,
        backtrack: (usize, &dyn Fn(usize, u16) -> bool),
        input: (usize, &dyn Fn(usize, u16) -> bool),
        lookahead: (usize, &dyn Fn(usize, u16) -> bool),
    ) -> Option<Vec<usize>> {
        let mut positions = vec![i];
        let mut p = i;

        for k in 0..input.0 {
            let j = self.next(buf, p, flags, mark_set)?;

            if !input.1(k, buf[j].glyph) || !self.takes(&buf[j]) {
                return None;
            }

            positions.push(j);
            p = j;
        }

        let mut q = p;

        for k in 0..lookahead.0 {
            let j = self.next(buf, q, flags, mark_set)?;

            if !lookahead.1(k, buf[j].glyph) {
                return None;
            }

            q = j;
        }

        let mut b = i;

        for k in 0..backtrack.0 {
            let j = self.prev(buf, b, flags, mark_set)?;

            if !backtrack.1(k, buf[j].glyph) {
                return None;
            }

            b = j;
        }

        Some(positions)
    }

    /// A contextual (GSUB 5, GPOS 7) or chaining contextual (GSUB 6, GPOS
    /// 8) subtable at `i`, in any of its three formats: the rule that
    /// matches, if one does.
    #[allow(clippy::too_many_arguments)]
    fn match_context(&self, r: Reader, chain: bool, st: usize, flags: u16, mark_set: Option<u16>, buf: &[Info], i: usize) -> Option<Matched> {
        let glyph = buf[i].glyph;
        let format = r.u16(st).ok()?;
        let u = |at: usize| r.u16(at).unwrap_or(0);
        let records = |at: usize, n: usize| -> Vec<(usize, u16)> { (0..n).map(|k| (u(at + 4 * k) as usize, u(at + 2 + 4 * k))).collect() };
        let by_glyph = |at: usize| move |k: usize, g: u16| u(at + 2 * k) == g;

        match (chain, format) {
            (_, 3) if !chain => {
                let n = u(st + 2) as usize;
                let covs: Vec<usize> = (0..n).map(|k| st + u(st + 6 + 2 * k) as usize).collect();

                if n == 0 || coverage(r, covs[0], glyph).is_none() {
                    return None;
                }

                let input = |k: usize, g: u16| coverage(r, covs[k + 1], g).is_some();
                let none = |_: usize, _: u16| false;
                let positions = self.match_rule(buf, i, flags, mark_set, (0, &none), (n - 1, &input), (0, &none))?;
                Some(Matched { positions, records: records(st + 6 + 2 * n, u(st + 4) as usize) })
            }
            (true, 3) => {
                let mut at = st + 2;
                let mut read = || {
                    let n = u(at) as usize;
                    let v: Vec<usize> = (0..n).map(|k| st + u(at + 2 + 2 * k) as usize).collect();
                    at += 2 + 2 * n;
                    v
                };
                let (back, input, ahead) = (read(), read(), read());

                if input.is_empty() || coverage(r, input[0], glyph).is_none() {
                    return None;
                }

                let fb = |k: usize, g: u16| coverage(r, back[k], g).is_some();
                let fi = |k: usize, g: u16| coverage(r, input[k + 1], g).is_some();
                let fa = |k: usize, g: u16| coverage(r, ahead[k], g).is_some();
                let positions = self.match_rule(buf, i, flags, mark_set, (back.len(), &fb), (input.len() - 1, &fi), (ahead.len(), &fa))?;
                Some(Matched { positions, records: records(at + 2, u(at) as usize) })
            }
            (_, 1 | 2) => {
                let c = coverage(r, st + u(st + 2) as usize, glyph)?;
                // format 2 picks the rule set by the first glyph's class, and matches classes
                let (classes, sets_at, set_index) = if format == 1 {
                    ([0usize; 3], if chain { st + 6 } else { st + 6 }, c)
                } else if chain {
                    let defs = [st + u(st + 4) as usize, st + u(st + 6) as usize, st + u(st + 8) as usize];
                    (defs, st + 12, class_of(r, defs[1], glyph) as usize)
                } else {
                    let def = st + u(st + 4) as usize;
                    ([def, def, def], st + 8, class_of(r, def, glyph) as usize)
                };
                let count = u(if format == 1 { st + 4 } else if chain { st + 10 } else { st + 6 }) as usize;

                if set_index >= count || u(sets_at + 2 * set_index) == 0 {
                    return None;
                }

                let set = st + u(sets_at + 2 * set_index) as usize;

                for k in 0..u(set) as usize {
                    let rule = set + u(set + 2 + 2 * k) as usize;
                    let test = |def: usize, at: usize| -> Box<dyn Fn(usize, u16) -> bool + '_> {
                        if format == 1 { Box::new(by_glyph(at)) } else { Box::new(move |k: usize, g: u16| class_of(r, def, g) == u(at + 2 * k)) }
                    };
                    let none = |_: usize, _: u16| false;

                    let (found, rec_at) = if chain {
                        let nb = u(rule) as usize;
                        let ni_at = rule + 2 + 2 * nb;
                        let ni = u(ni_at) as usize;
                        let na_at = ni_at + 2 + 2 * ni.saturating_sub(1);
                        let na = u(na_at) as usize;
                        let rec_at = na_at + 2 + 2 * na;
                        let (fb, fi, fa) = (test(classes[0], rule + 2), test(classes[1], ni_at + 2), test(classes[2], na_at + 2));
                        (self.match_rule(buf, i, flags, mark_set, (nb, &*fb), (ni.saturating_sub(1), &*fi), (na, &*fa)), rec_at)
                    } else {
                        let ni = u(rule) as usize;
                        let fi = test(classes[1], rule + 4);
                        (self.match_rule(buf, i, flags, mark_set, (0, &none), (ni.saturating_sub(1), &*fi), (0, &none)), rule + 4 + 2 * ni.saturating_sub(1))
                    };

                    if let Some(positions) = found {
                        let n = u(if chain { rec_at } else { rule + 2 }) as usize;
                        let at = if chain { rec_at + 2 } else { rec_at };
                        return Some(Matched { positions, records: records(at, n) });
                    }
                }

                None
            }
            _ => None,
        }
    }

    /// Whether the lookups of `features` would change `glyphs` standing
    /// alone (HarfBuzz's `would_substitute`).
    fn would_substitute(&self, gsub: &Layout, features: &[&[u8; 4]], glyphs: &[u16]) -> bool {
        let mut buf: Vec<Info> = glyphs.iter().map(|&g| Info { glyph: g, class: self.glyph_class(g), ..Default::default() }).collect();
        let before = buf.clone();
        let saved = self.lookup_mask.replace(u32::MAX);

        for l in gsub.lookups_for(features) {
            if self.gsub_lookup(gsub, l, &mut buf).is_err() {
                break;
            }
        }

        self.lookup_mask.set(saved);
        buf.len() != before.len() || buf.iter().zip(&before).any(|(a, b)| a.glyph != b.glyph)
    }

    // --- GSUB --------------------------------------------------------

    /// Runs a GSUB lookup over the glyphs with one of the feature bits.
    fn gsub_lookup_bits(&self, gsub: &Layout, index: u16, buf: &mut Vec<Info>, bits: u32) -> Result<(), Unsupported> {
        let saved = self.lookup_mask.replace(bits);
        let r = self.gsub_lookup_masked(gsub, index, buf, None);
        self.lookup_mask.set(saved);
        r
    }

    /// Runs a GSUB lookup over the buffer.
    fn gsub_lookup(&self, gsub: &Layout, index: u16, buf: &mut Vec<Info>) -> Result<(), Unsupported> {
        self.gsub_lookup_masked(gsub, index, buf, None)
    }

    /// Runs a GSUB lookup over the glyphs with the joining form `mask`
    /// (all of them for `None`).
    fn gsub_lookup_masked(&self, gsub: &Layout, index: u16, buf: &mut Vec<Info>, mask: Option<u8>) -> Result<(), Unsupported> {
        let Some((kind, flags, mark_set, subtables)) = gsub.lookup(index, 7) else { return Ok(()) };

        if kind == 8 {
            // reverse chaining single substitution: from the end
            for i in (0..buf.len()).rev() {
                if self.ignored(&buf[i], flags, mark_set) || !self.takes(&buf[i]) || mask.is_some_and(|m| buf[i].mask != m) {
                    continue;
                }

                for &st in &subtables {
                    if self.reverse_chain(gsub.r, st, flags, mark_set, buf, i) {
                        break;
                    }
                }
            }

            return Ok(());
        }

        let mut i = 0;

        while i < buf.len() {
            if self.ignored(&buf[i], flags, mark_set) || !self.takes(&buf[i]) || mask.is_some_and(|m| buf[i].mask != m) {
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
                buf[i].substituted = true;
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
                buf.splice(i..=i, glyphs.iter().map(|&g| Info { glyph: g, class: self.glyph_class(g), multiplied: n > 1, substituted: true, ..template }));
                Ok(Some(i + n))
            }
            3 => {
                // an alternate: the first
                let Some(c) = cov(st + 2) else { return Ok(None) };
                let set = st + r.u16(st + 6 + 2 * c).unwrap_or(0) as usize;

                if r.u16(set).unwrap_or(0) == 0 {
                    return Ok(None);
                }

                let new = r.u16(set + 2).unwrap_or(glyph);
                buf[i].glyph = new;
                buf[i].class = self.glyph_class(new);
                buf[i].substituted = true;
                Ok(Some(i + 1))
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
                            Some(j) if buf[j].glyph == r.u16(lig + 4 + 2 * (m - 1)).unwrap_or(0) && self.takes(&buf[j]) => {
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

                    // one cluster, over what follows of the last component's too
                    let last = positions.last().copied().unwrap_or(i);
                    merge(buf, i, last + 1);

                    // as HarfBuzz's ligate_input: a ligature of a letter and
                    // marks counts as a letter, of marks as a mark; else the
                    // marks passed over (and those after it on its last
                    // component) remember which component they were on
                    let base_ligature = buf[i].class != MARK && positions[1..].iter().all(|&p| buf[p].class == MARK);
                    let mark_ligature = positions.iter().all(|&p| buf[p].class == MARK);
                    let ligature = !base_ligature && !mark_ligature;
                    let lig_id = if ligature {
                        let n = self.lig_id.get() % 7 + 1;
                        self.lig_id.set(n);
                        n
                    } else {
                        0
                    };
                    let total: u32 = positions.iter().map(|&p| u32::from(buf[p].components())).sum();
                    if ligature {
                        let mut so_far = u32::from(buf[i].components());
                        let mut last_comps = so_far;
                        for w in positions.windows(2) {
                            for g in &mut buf[w[0] + 1..w[1]] {
                                let this = if g.lig_comp == 0 { last_comps } else { u32::from(g.lig_comp) };
                                g.lig_id = lig_id;
                                g.lig_comp = (so_far - last_comps + this.min(last_comps)) as u8;
                            }
                            last_comps = u32::from(buf[w[1]].components());
                            so_far += last_comps;
                        }
                        let last_id = buf[last].lig_id;
                        if last_id != 0 {
                            for g in &mut buf[last + 1..] {
                                if g.lig_id != last_id || g.lig_comp == 0 {
                                    break;
                                }
                                let this = u32::from(g.lig_comp);
                                g.lig_id = lig_id;
                                g.lig_comp = (so_far - last_comps + this.min(last_comps)) as u8;
                            }
                        }
                    }

                    let lig_class = self.glyph_class(lig_glyph);
                    let (id, comps) = if mark_ligature { (buf[i].lig_id, buf[i].lig_comps) } else { (lig_id, total.min(15) as u8) };
                    buf[i] = Info { glyph: lig_glyph, class: if lig_class == 0 { LIGATURE } else { lig_class }, ligated: true, multiplied: false, substituted: true, lig_id: id, lig_comp: if mark_ligature { buf[i].lig_comp } else { 0 }, lig_comps: comps, ..buf[i] };

                    // the other components out; marks passed over stay, after the ligature
                    for &p in positions[1..].iter().rev() {
                        buf.remove(p);
                    }

                    // HarfBuzz merges the clusters of what follows into the ligature's
                    return Ok(Some(i + 1));
                }

                Ok(None)
            }
            5 | 6 => {
                let _ = format;
                let Some(m) = self.match_context(r, kind == 6, st, flags, mark_set, buf, i) else { return Ok(None) };
                self.nested_gsub(gsub, m, buf).map(Some)
            }
            _ => Ok(None),
        }
    }

    /// A matched rule's nested lookups applied, each at its input glyph
    /// (the places after it moved as the buffer grows or shrinks); where
    /// to go on from.
    fn nested_gsub(&self, gsub: &Layout, m: Matched, buf: &mut Vec<Info>) -> Result<usize, Unsupported> {
        let mut positions = m.positions;
        let mut end = positions.last().copied().unwrap_or(0) + 1;

        if self.depth.get() > 8 {
            return Ok(end);
        }

        self.depth.set(self.depth.get() + 1);

        for (seq, lookup) in m.records {
            let Some(&at) = positions.get(seq) else { continue };

            if at >= buf.len() {
                continue;
            }

            let Some((kind, f2, ms2, subs)) = gsub.lookup(lookup, 7) else { continue };
            let before = buf.len();

            if kind == 8 {
                for &st in &subs {
                    if self.reverse_chain(gsub.r, st, f2, ms2, buf, at) {
                        break;
                    }
                }

                continue;
            }

            for &st in &subs {
                match self.gsub_apply(gsub, kind, f2, ms2, st, buf, at) {
                    Ok(Some(_)) => break,
                    Ok(None) => {}
                    Err(e) => {
                        self.depth.set(self.depth.get() - 1);
                        return Err(e);
                    }
                }
            }

            let delta = buf.len() as isize - before as isize;

            if delta != 0 {
                for p in positions.iter_mut().skip(seq + 1) {
                    *p = (*p as isize + delta).max(at as isize) as usize;
                }

                end = (end as isize + delta).max(at as isize + 1) as usize;
            }
        }

        self.depth.set(self.depth.get() - 1);
        Ok(end.min(buf.len()).max(1))
    }

    /// A reverse chaining single substitution (GSUB 8) at `i`; whether it applied.
    fn reverse_chain(&self, r: Reader, st: usize, flags: u16, mark_set: Option<u16>, buf: &mut [Info], i: usize) -> bool {
        let u = |at: usize| r.u16(at).unwrap_or(0);

        if u(st) != 1 {
            return false;
        }

        let Some(c) = coverage(r, st + u(st + 2) as usize, buf[i].glyph) else { return false };
        let nb = u(st + 4) as usize;
        let back: Vec<usize> = (0..nb).map(|k| st + u(st + 6 + 2 * k) as usize).collect();
        let na_at = st + 6 + 2 * nb;
        let na = u(na_at) as usize;
        let ahead: Vec<usize> = (0..na).map(|k| st + u(na_at + 2 + 2 * k) as usize).collect();
        let subs_at = na_at + 2 + 2 * na;
        let fb = |k: usize, g: u16| coverage(r, back[k], g).is_some();
        let fa = |k: usize, g: u16| coverage(r, ahead[k], g).is_some();
        let none = |_: usize, _: u16| false;

        if self.match_rule(buf, i, flags, mark_set, (nb, &fb), (0, &none), (na, &fa)).is_none() || c >= u(subs_at) as usize {
            return false;
        }

        let g = u(subs_at + 2 + 2 * c);
        buf[i].glyph = g;
        buf[i].class = self.glyph_class(g);
        true
    }

    // --- GPOS ----------------------------------------------------------

    /// A value record at `at` of the format: x placement, y placement, x
    /// advance, y advance in font units; and its size.
    /// Its device tables (offsets from `base`), a variable font's
    /// instance's deltas to the values.
    fn value_record(&self, r: Reader, base: usize, at: usize, format: u16) -> ([i32; 4], usize) {
        let mut v = [0; 4];
        let mut p = at;

        for (bit, slot) in [(0x1, 0), (0x2, 1), (0x4, 2), (0x8, 3)] {
            if format & bit != 0 {
                v[slot] = i32::from(r.i16(p).unwrap_or(0));
                p += 2;
            }
        }

        for (bit, slot) in [(0x10, 0), (0x20, 1), (0x40, 2), (0x80, 3)] {
            if format & bit != 0 {
                let off = r.u16(p).unwrap_or(0) as usize;
                if off != 0 {
                    v[slot] += crate::var::hb_round(self.device_delta(r, base + off)) as i32;
                }
                p += 2;
            }
        }

        (v, p - at)
    }

    /// A value record's size.
    fn value_size(format: u16) -> usize {
        2 * (format & 0xFF).count_ones() as usize
    }

    /// A device table's delta for the face's instance: a variation index
    /// into GDEF's item variation store (hinting's pixel deltas ignored, as
    /// HarfBuzz ignores them unhinted).
    fn device_delta(&self, r: Reader, at: usize) -> f32 {
        let coords = self.face.coords();
        if coords.is_empty() || r.u16(at + 4).ok() != Some(0x8000) {
            return 0.0;
        }
        let Some(g) = self.gdef else { return 0.0 };
        if g.u32(0).unwrap_or(0) < 0x0001_0003 {
            return 0.0;
        }
        let store = g.u32(14).unwrap_or(0) as usize;
        if store == 0 {
            return 0.0;
        }
        let (outer, inner) = (r.u16(at).unwrap_or(0), r.u16(at + 2).unwrap_or(0));
        crate::var::delta(g, store, outer, inner, coords)
    }

    fn gpos_lookup(&self, gpos: &Layout, index: u16, buf: &[Info], pos: &mut [GlyphPosition], em: &dyn Fn(i32) -> i32) -> Result<(), Unsupported> {
        let Some((kind, flags, mark_set, subtables)) = gpos.lookup(index, 9) else { return Ok(()) };

        for i in 0..buf.len() {
            if self.ignored(&buf[i], flags, mark_set) {
                continue;
            }

            for &st in &subtables {
                if self.gpos_apply(gpos, kind, flags, mark_set, st, buf, pos, i, em) {
                    break;
                }
            }
        }

        Ok(())
    }

    /// One GPOS subtable at `i`; whether it applied.
    #[allow(clippy::too_many_arguments)]
    fn gpos_apply(&self, gpos: &Layout, kind: u16, flags: u16, mark_set: Option<u16>, st: usize, buf: &[Info], pos: &mut [GlyphPosition], i: usize, em: &dyn Fn(i32) -> i32) -> bool {
        let r = gpos.r;

        if matches!(kind, 7 | 8) {
            let Some(m) = self.match_context(r, kind == 8, st, flags, mark_set, buf, i) else { return false };

            if self.depth.get() > 8 {
                return true;
            }

            self.depth.set(self.depth.get() + 1);

            for (seq, lookup) in m.records {
                let Some(&at) = m.positions.get(seq) else { continue };
                let Some((k2, f2, ms2, subs)) = gpos.lookup(lookup, 9) else { continue };

                for &s2 in &subs {
                    if self.gpos_apply(gpos, k2, f2, ms2, s2, buf, pos, at, em) {
                        break;
                    }
                }
            }

            self.depth.set(self.depth.get() - 1);
            return true;
        }

        let format = r.u16(st).unwrap_or(0);
        let glyph = buf[i].glyph;
        let Some(c) = r.u16(st + 2).ok().and_then(|o| coverage(r, st + o as usize, glyph)) else { return false };

        match kind {
            1 => {
                let vf = r.u16(st + 4).unwrap_or(0);
                let v = match format {
                    1 => self.value_record(r, st, st + 6, vf).0,
                    _ => self.value_record(r, st, st + 8 + c * Self::value_size(vf), vf).0,
                };
                Self::apply_value(&mut pos[i], v, em);
                true
            }
            2 => {
                let Some(j) = self.next(buf, i, flags, mark_set) else { return false };
                let (vf1, vf2) = (r.u16(st + 4).unwrap_or(0), r.u16(st + 6).unwrap_or(0));
                let (s1, s2) = (Self::value_size(vf1), Self::value_size(vf2));

                let found = match format {
                    1 => {
                        let set = st + r.u16(st + 10 + 2 * c).unwrap_or(0) as usize;
                        let n = r.u16(set).unwrap_or(0) as usize;
                        let rec = 2 + s1 + s2;
                        (0..n)
                            .map(|k| set + 2 + k * rec)
                            .find(|&at| r.u16(at).ok() == Some(buf[j].glyph))
                            .map(|at| (self.value_record(r, set, at + 2, vf1).0, self.value_record(r, set, at + 2 + s1, vf2).0))
                    }
                    _ => {
                        let c1 = class_of(r, st + r.u16(st + 8).unwrap_or(0) as usize, glyph) as usize;
                        let c2 = class_of(r, st + r.u16(st + 10).unwrap_or(0) as usize, buf[j].glyph) as usize;
                        let n2 = r.u16(st + 14).unwrap_or(0) as usize;
                        let at = st + 16 + (c1 * n2 + c2) * (s1 + s2);
                        Some((self.value_record(r, st, at, vf1).0, self.value_record(r, st, at + s1, vf2).0))
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
            3 => self.attach_cursive(r, st, c, buf, pos, i, flags, mark_set, em),
            4..=6 => self.attach_mark(r, kind, st, c, buf, pos, i, flags, mark_set, em),
            _ => false,
        }
    }

    /// Cursive attachment (GPOS 3) at `i` (its coverage index `c`): the
    /// glyph before's exit meets this one's entry (HarfBuzz's
    /// `CursivePosFormat1`); the one that hangs is placed later, after its
    /// chain (`propagate_cursive`).
    #[allow(clippy::too_many_arguments)]
    fn attach_cursive(&self, r: Reader, st: usize, c: usize, buf: &[Info], pos: &mut [GlyphPosition], i: usize, flags: u16, mark_set: Option<u16>, em: &dyn Fn(i32) -> i32) -> bool {
        let u = |at: usize| r.u16(at).unwrap_or(0) as usize;
        let entry = u(st + 6 + 4 * c);
        if entry == 0 {
            return false;
        }
        let Some(j) = self.prev(buf, i, flags, mark_set) else { return false };
        let Some(cj) = coverage(r, st + u(st + 2), buf[j].glyph) else { return false };
        let exit = u(st + 6 + 4 * cj + 2);
        if exit == 0 {
            return false;
        }
        let (Some((ex, ey)), Some((nx, ny))) = (self.anchor(r, st + exit), self.anchor(r, st + entry)) else { return false };
        if self.rtl.get() {
            let d = em(ex) + pos[j].x_offset;
            pos[j].x_advance -= d;
            pos[j].x_offset -= d;
            pos[i].x_advance = em(nx) + pos[i].x_offset;
        } else {
            pos[j].x_advance = em(ex) + pos[j].x_offset;
            let d = em(nx) + pos[i].x_offset;
            pos[i].x_advance -= d;
            pos[i].x_offset -= d;
        }
        // the one that hangs from the other: the first, right to left
        let (child, parent, y) = if flags & 1 != 0 { (j, i, em(ny - ey)) } else { (i, j, -em(ny - ey)) };
        let mut links = self.cursive.borrow_mut();
        if links.len() < buf.len() {
            links.resize(buf.len(), None);
        }
        // no loops
        let mut k = parent;
        for _ in 0..buf.len() {
            match links[k] {
                Some(p) if p == child => return true,
                Some(p) => k = p,
                None => break,
            }
        }
        links[child] = Some(parent);
        pos[child].y_offset = y;
        true
    }

    /// Cursive chains' offsets added up, each glyph's after its parent's;
    /// the marks on them moved with them.
    fn propagate_cursive(&self, pos: &mut [GlyphPosition]) {
        let links = self.cursive.borrow();
        if links.iter().all(Option::is_none) {
            return;
        }
        let raw: Vec<i32> = pos.iter().map(|p| p.y_offset).collect();
        let mut done = vec![false; pos.len()];
        fn settle(i: usize, links: &[Option<usize>], pos: &mut [GlyphPosition], done: &mut [bool], depth: usize) {
            if done[i] || depth > 64 {
                return;
            }
            done[i] = true;
            if let Some(Some(p)) = links.get(i).copied() {
                settle(p, links, pos, done, depth + 1);
                pos[i].y_offset += pos[p].y_offset;
            }
        }
        for i in 0..pos.len() {
            settle(i, &links, pos, &mut done, 0);
        }
        // marks: with what they sit on (marks on marks, in order)
        let marks = self.marks.borrow();
        let mut delta: Vec<i32> = pos.iter().zip(&raw).map(|(p, r)| p.y_offset - r).collect();
        for i in 0..pos.len() {
            if let Some(Some(t)) = marks.get(i).copied()
                && t < pos.len()
            {
                pos[i].y_offset += delta[t];
                delta[i] += delta[t];
            }
        }
    }

    /// An anchor's point, in font units.
    fn anchor(&self, r: Reader, at: usize) -> Option<(i32, i32)> {
        let (mut x, mut y) = (i32::from(r.u16(at + 2).ok()? as i16), i32::from(r.u16(at + 4).ok()? as i16));
        // format 3: device tables, a variable font's deltas
        if r.u16(at).ok()? == 3 {
            for (k, v) in [(6, &mut x), (8, &mut y)] {
                let off = r.u16(at + k).unwrap_or(0) as usize;
                if off != 0 {
                    *v += crate::var::hb_round(self.device_delta(r, at + off)) as i32;
                }
            }
        }
        Some((x, y))
    }

    /// Mark-to-base (4), mark-to-ligature (5) and mark-to-mark (6): the
    /// mark at `i` (its coverage index `c`) moved so its anchor meets the
    /// one it attaches to. Marks count as having no advance, as they will.
    #[allow(clippy::too_many_arguments)]
    fn attach_mark(
        &self,
        r: Reader,
        kind: u16,
        st: usize,
        c: usize,
        buf: &[Info],
        pos: &mut [GlyphPosition],
        i: usize,
        flags: u16,
        mark_set: Option<u16>,
        em: &dyn Fn(i32) -> i32,
    ) -> bool {
        let attach = || -> Option<(usize, (i32, i32), (i32, i32))> {
            if buf[i].class != MARK && kind != 6 {
                return None;
            }
            let classes = r.u16(st + 6).ok()? as usize;
            let marks = st + r.u16(st + 8).ok()? as usize;
            let targets = st + r.u16(st + 10).ok()? as usize;
            let rec = marks + 2 + 4 * c;
            let class = r.u16(rec).ok()? as usize;
            let mark_anchor = self.anchor(r, marks + r.u16(rec + 2).ok()? as usize)?;
            // what it attaches to: the letter before (marks passed over),
            // or for mark-to-mark the mark just before
            let target = if kind == 6 {
                let j = self.prev(buf, i, flags, mark_set)?;
                (buf[j].class == MARK).then_some(j)?;
                // two marks on the same letter, or the same component of a ligature
                let (a, b) = (&buf[i], &buf[j]);
                let same = if a.lig_id == b.lig_id { a.lig_id == 0 || a.lig_comp == b.lig_comp } else { (a.lig_id > 0 && a.lig_comp == 0) || (b.lig_id > 0 && b.lig_comp == 0) };
                same.then_some(j)?
            } else {
                (0..i).rev().find(|&j| buf[j].class != MARK)?
            };
            let cov = st + r.u16(st + 4).ok()? as usize;
            let t = coverage(r, cov, buf[target].glyph)?;
            let anchor_at = if kind == 5 {
                // a ligature: the component the mark is on (where it
                // remembers it), else the last
                let lig = targets + r.u16(targets + 2 + 2 * t).ok()? as usize;
                let comps = r.u16(lig).ok()? as usize;
                let (l, m) = (&buf[target], &buf[i]);
                let comp = if l.lig_id != 0 && l.lig_id == m.lig_id && m.lig_comp > 0 { comps.min(m.lig_comp as usize).checked_sub(1)? } else { comps.checked_sub(1)? };
                let off = r.u16(lig + 2 + 2 * (comp * classes + class)).ok()? as usize;
                (off != 0).then_some(lig + off)?
            } else {
                let off = r.u16(targets + 2 + 2 * (t * classes + class)).ok()? as usize;
                (off != 0).then_some(targets + off)?
            };
            Some((target, self.anchor(r, anchor_at)?, mark_anchor))
        };
        let Some((target, base, mark)) = attach() else { return false };
        {
            let mut m = self.marks.borrow_mut();
            if m.len() < buf.len() {
                m.resize(buf.len(), None);
            }
            m[i] = Some(target);
        }
        // the pen moved on past the target (and the marks between) since
        let passed = |k: &usize| buf[*k].class != MARK || !self.zero_marks;
        let between: i32 = (target..i).filter(passed).map(|k| pos[k].x_advance).sum();
        // (vertical text: the pen moved down past them, their advances negative)
        let below: i32 = (target..i).filter(passed).map(|k| pos[k].y_advance).sum();
        pos[i].x_offset = pos[target].x_offset + em(base.0 - mark.0) - between;
        pos[i].y_offset = pos[target].y_offset + em(base.1 - mark.1) - below;
        true
    }

    fn apply_value(p: &mut GlyphPosition, v: [i32; 4], em: &dyn Fn(i32) -> i32) {
        p.x_offset += em(v[0]);
        p.y_offset += em(v[1]);
        p.x_advance += em(v[2]);
        p.y_advance += em(v[3]);
    }

    /// A `kern` format 2 subtable at `st` (its fields at `body`): the
    /// value for a pair, at the subtable plus the two glyphs' class
    /// offsets.
    fn class_kern(r: Reader, st: usize, body: usize, left: u16, right: u16) -> Option<i32> {
        // only glyphs within the class tables kern (as HarfBuzz)
        let class = |table: usize, g: u16| -> Option<usize> {
            let (first, n) = (r.u16(table).ok()?, r.u16(table + 2).ok()?);
            (g >= first && g - first < n).then(|| r.u16(table + 4 + 2 * (g - first) as usize).unwrap_or(0) as usize)
        };
        let l = class(st + r.u16(body + 2).ok()? as usize, left)?;
        let rc = class(st + r.u16(body + 4).ok()? as usize, right)?;
        let v = i32::from(r.i16(st + l + rc).ok()?);
        (v != 0).then_some(v)
    }

    /// Apple's `kern` table (version 1): its horizontal format 0 subtables' pair value.
    fn apple_kern_pair(r: Reader, left: u16, right: u16) -> Option<i32> {
        let n = r.u32(4).ok()? as usize;
        let mut at = 8;
        let mut total = None;

        for _ in 0..n {
            let len = r.u32(at).ok()? as usize;
            let coverage = r.u16(at + 4).ok()?;

            // format 2: a class table for each side, the value at the sum of their classes
            if coverage & 0xE000 == 0
                && coverage & 0xFF == 2
                && let Some(v) = Self::class_kern(r, at, at + 8, left, right)
            {
                total = Some(total.unwrap_or(0) + v);
            }

            // format 3: glyphs' classes, an index of values for each pair of them
            if coverage & 0xE000 == 0 && coverage & 0xFF == 3 {
                let b = at + 8;
                let glyphs = r.u16(b).ok()?;
                let byte = |o: usize| r.slice(o, 1).ok().map(|x| x[0] as usize);
                let (values, lefts, rights) = (byte(b + 2)?, byte(b + 3)?, byte(b + 4)?);
                if left < glyphs && right < glyphs {
                    let classes = b + 6 + 2 * values;
                    let (lc, rc) = (byte(classes + left as usize)?, byte(classes + glyphs as usize + right as usize)?);
                    if lc < lefts && rc < rights {
                        let index = byte(classes + 2 * glyphs as usize + lc * rights + rc)?;
                        let v = i32::from(r.i16(b + 6 + 2 * index).ok()?);
                        if v != 0 {
                            total = Some(total.unwrap_or(0) + v);
                        }
                    }
                }
            }

            // horizontal, not cross-stream or variation, format 0
            if coverage & 0xE000 == 0 && coverage & 0xFF == 0 {
                let pairs = r.u16(at + 8).ok()? as usize;
                let key = (u32::from(left) << 16) | u32::from(right);
                let (mut lo, mut hi) = (0, pairs);

                while lo < hi {
                    let mid = (lo + hi) / 2;
                    let rec = at + 16 + 6 * mid;
                    match r.u32(rec).ok()?.cmp(&key) {
                        std::cmp::Ordering::Equal => {
                            total = Some(total.unwrap_or(0) + i32::from(r.i16(rec + 4).ok()?));
                            break;
                        }
                        std::cmp::Ordering::Less => lo = mid + 1,
                        std::cmp::Ordering::Greater => hi = mid,
                    }
                }
            }

            at += len.max(1);
        }

        total
    }

    /// The `kern` table's pair value (format 0 subtables, horizontal).
    fn kern_pair(&self, left: u16, right: u16) -> Option<i32> {
        let kern = self.face.table(b"kern")?;
        let r = Reader::new(kern);

        if r.u32(0).ok()? == 0x0001_0000 {
            return Self::apple_kern_pair(r, left, right);
        }

        if r.u16(0).ok()? != 0 {
            return None;
        }

        let n = r.u16(2).ok()? as usize;
        let mut at = 4;
        let mut total = None;

        for _ in 0..n {
            let len = r.u16(at + 2).ok()? as usize;
            let coverage = r.u16(at + 4).ok()?;

            // horizontal, format 2 (classes), not minimum or cross-stream
            if coverage & 1 == 1 && coverage >> 8 == 2 && coverage & 0x6 == 0
                && let Some(v) = Self::class_kern(r, at, at + 6, left, right)
            {
                total = Some(if coverage & 0x8 != 0 { v } else { total.unwrap_or(0) + v });
            }

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

/// A mark's combining class (0 for what is not a mark), as HarfBuzz
/// orders marks: Unicode's, but Hebrew's and Arabic's points in the order
/// they are typeset (HarfBuzz's `_hb_modified_combining_class`).
fn ccc(c: u32) -> u8 {
    let t = compose::CCC;
    let class = match t.binary_search_by(|&(a, b, _)| if c < a { std::cmp::Ordering::Greater } else if c > b { std::cmp::Ordering::Less } else { std::cmp::Ordering::Equal }) {
        Ok(i) => t[i].2,
        Err(_) => 0,
    };
    match class {
        10 => 22,
        11 => 15,
        12 => 16,
        13 => 17,
        14 => 23,
        15 => 18,
        16 => 19,
        17 => 20,
        18 => 21,
        19 => 14,
        20 => 24,
        21 => 12,
        22 => 25,
        23 => 13,
        24 => 10,
        25 => 11,
        27 => 28,
        28 => 29,
        29 => 30,
        30 => 31,
        31 => 32,
        32 => 33,
        33 => 27,
        84 => 4,
        91 => 5,
        103 => 3,
        130 => 132,
        132 => 131,
        c => c,
    }
}

/// A mark's class for placing it without the font's help (HarfBuzz's
/// `recategorize_combining_class`): above, below, left, right and so on.
fn placement_class(c: u32, class: u8) -> u8 {
    if class >= 200 {
        return class;
    }
    let class = if c & !0xFF == 0xE00 {
        match (class, c) {
            (0, 0xE31 | 0xE34..=0xE37 | 0xE47 | 0xE4C..=0xE4E) => 232,
            (0, 0xEB1 | 0xEB4..=0xEB7 | 0xEBB | 0xECC..=0xECD) => 230,
            (0, 0xEBC) => 220,
            (_, 0xE3A) => 222,
            (k, _) => k,
        }
    } else {
        class
    };
    match class {
        22 | 15 | 16 | 17 | 23 | 18 | 19 | 20 | 21 | 24 | 25 => 220,
        13 => 214,
        10 => 232,
        11 | 14 => 228,
        26 => 230,
        28 | 29 | 31 | 32 | 27 | 34 | 35 | 36 => 230,
        30 | 33 => 220,
        3 => 222,
        107 => 232,
        118 => 220,
        122 => 230,
        129 => 220,
        132 => 230,
        131 => 220,
        k => k,
    }
}

/// A glyph's ink box in font units, as HarfBuzz gives extents: left, top,
/// width, height (negative, downward). From `glyf` where the face has it.
fn extents(face: &Face, gid: u16) -> Option<(i32, i32, i32, i32)> {
    if face.table(b"glyf").is_none() {
        // CFF: the outline's box, its control points in it, rounded as HarfBuzz rounds
        let path = crate::cff::Cff::new(face.table(b"CFF ")?)?.path(gid)?;
        let Some((x0, y0, x1, y1)) = crate::cff::bounds(&path) else { return Some((0, 0, 0, 0)) };
        let r = |v: f64| crate::var::hb_round(v as f32) as i32;
        let (xb, w) = if x0 >= x1 { (0, 0) } else { (r(x0), r(x1 - f64::from(r(x0)))) };
        let (yb, h) = if y0 >= y1 { (0, 0) } else { (r(y1), r(y0 - f64::from(r(y1)))) };
        return Some((xb, yb, w, h));
    }
    let d = face.glyph_data(gid);
    if d.len() < 10 {
        return Some((0, 0, 0, 0));
    }
    let v = |k: usize| i32::from(i16::from_be_bytes([d[k], d[k + 1]]));
    let (x0, y0, x1, y1) = (v(2), v(4), v(6), v(8));
    // the left from `hmtx`'s side bearing, as rasterizers (and HarfBuzz) shift the glyph there
    let lsb = i32::from(face.h_metrics(gid).1);
    Some((lsb, y0.max(y1), (x1 - x0).abs(), y0.min(y1) - y0.max(y1)))
}

/// Marks placed round their letters from the glyphs' boxes, for a font with
/// no mark positioning of its own (HarfBuzz's fallback mark positioning).
fn fallback_marks(face: &Face, buf: &[Info], pos: &mut [GlyphPosition], rtl: bool, em: &dyn Fn(i32) -> i32) {
    fallback_marks_dir(face, buf, pos, rtl, None, em);
}

/// [`fallback_marks`], for vertical text with the glyphs' horizontal
/// advances (`widths`) the boxes are measured across by.
fn fallback_marks_dir(face: &Face, buf: &[Info], pos: &mut [GlyphPosition], rtl: bool, widths: Option<&[i32]>, em: &dyn Fn(i32) -> i32) {
    let gap = em(i32::from(face.units_per_em) / 16);
    let mut i = 0;
    while i < buf.len() {
        if buf[i].ccc != 0 {
            i += 1;
            continue;
        }
        let base = i;
        let mut end = i + 1;
        while end < buf.len() && (buf[end].ccc != 0 || buf[end].hidden) {
            end += 1;
        }
        if end - base > 1 {
            place_round(face, buf, pos, base, end, rtl, gap, widths, em);
        }
        i = end;
    }
}

#[allow(clippy::too_many_arguments)]
fn place_round(face: &Face, buf: &[Info], pos: &mut [GlyphPosition], base: usize, end: usize, rtl: bool, gap: i32, widths: Option<&[i32]>, em: &dyn Fn(i32) -> i32) {
    let Some((_, by, _, bh)) = extents(face, buf[base].glyph) else {
        for p in &mut pos[base + 1..end] {
            p.x_advance = 0;
            p.y_advance = 0;
        }
        return;
    };
    // extents in 26.6: by the advance across, the ink up and down (in
    // vertical text, from the letter's origin across its horizontal advance)
    let base_ext = match widths {
        Some(w) => (0, em(by) + pos[base].y_offset, w[base], em(bh)),
        None => (pos[base].x_offset, em(by) + pos[base].y_offset, pos[base].x_advance, em(bh)),
    };
    let mut x_off = if rtl { 0 } else { -pos[base].x_advance };
    // the pen gone down past the letter (vertical text)
    let mut y_off = -pos[base].y_advance;
    let mut last = 255u8;
    let mut cluster = base_ext;
    for i in base + 1..end {
        let class = buf[i].ccc;
        if class == 0 {
            x_off -= if rtl { -pos[i].x_advance } else { pos[i].x_advance };
            y_off -= pos[i].y_advance;
            continue;
        }
        if class != last {
            last = class;
            cluster = base_ext;
        }
        let Some((mx, my, mw, mh)) = extents(face, buf[i].glyph) else { continue };
        let (mx, my, mw, mh) = (em(mx), em(my), em(mw), em(mh));
        let p = &mut pos[i];
        p.x_offset = 0;
        p.y_offset = 0;
        p.x_offset = match class {
            233 | 234 if !rtl => cluster.0 + cluster.2 - mw / 2 - mx,
            233 | 234 => cluster.0 - mw / 2 - mx,
            200 | 218 | 228 => cluster.0 - mx,
            216 | 222 | 232 => cluster.0 + cluster.2 - mw - mx,
            _ => cluster.0 + (cluster.2 - mw) / 2 - mx,
        };
        match class {
            233 | 218 | 220 | 222 | 200 | 202 => {
                if matches!(class, 233 | 218 | 220 | 222) {
                    cluster.3 -= gap;
                }
                p.y_offset = cluster.1 + cluster.3 - my;
                if (gap > 0) == (p.y_offset > 0) {
                    cluster.3 -= p.y_offset;
                    p.y_offset = 0;
                }
                cluster.3 += mh;
            }
            234 | 228 | 230 | 232 | 214 | 216 => {
                if matches!(class, 234 | 228 | 230 | 232) {
                    cluster.1 += gap;
                    cluster.3 -= gap;
                }
                p.y_offset = cluster.1 - my - mh;
                if (gap > 0) != (p.y_offset > 0) {
                    let c = -(p.y_offset / 2);
                    cluster.1 += c;
                    cluster.3 -= c;
                    p.y_offset += c;
                }
                cluster.1 -= mh;
                cluster.3 += mh;
            }
            _ => {}
        }
        p.x_advance = 0;
        p.y_advance = 0;
        p.x_offset += x_off;
        p.y_offset += y_off;
    }
}

/// A letter and a mark as one character, where Unicode has one.
fn compose_pair(a: u32, b: u32) -> Option<u32> {
    let t = compose::COMPOSE;
    t.binary_search_by(|&(x, y, _)| (x, y).cmp(&(a, b))).ok().map(|i| t[i].2)
}

/// Whether `c` is a non-spacing mark (Mn).
fn is_nonspacing(c: u32) -> bool {
    compose::MN.binary_search_by(|&(a, b)| if c < a { std::cmp::Ordering::Greater } else if c > b { std::cmp::Ordering::Less } else { std::cmp::Ordering::Equal }).is_ok()
}

/// A Hebrew letter and point as their presentation form.
fn compose_hebrew(a: u32, b: u32) -> Option<u32> {
    let t = compose::HEBREW;
    t.binary_search_by(|&(x, y, _)| (x, y).cmp(&(a, b))).ok().map(|i| t[i].2)
}

/// A character as its letter and mark, where Unicode has them.
fn decompose(c: u32) -> Option<(u32, u32)> {
    compose::COMPOSE.iter().find(|t| t.2 == c).map(|t| (t.0, t.1))
}

/// Characters as HarfBuzz normalizes them for a font: one the font lacks
/// split into its letter and marks, the marks in their canonical order,
/// then a letter and a mark made one where the font has the whole.
fn normalize(face: &Face, chars: Vec<(usize, u32)>, hebrew_forms: bool) -> Vec<(usize, u32)> {
    // as HarfBuzz: its mark has a glyph, and its letter has one or splits so too
    fn splits(face: &Face, c: u32) -> bool {
        decompose(c).is_some_and(|(a, b)| face.glyph_index(b) != 0 && (face.glyph_index(a) != 0 || splits(face, a)))
    }
    fn split(face: &Face, o: usize, c: u32, out: &mut Vec<(usize, u32)>) {
        if face.glyph_index(c) == 0
            && splits(face, c)
            && let Some((a, b)) = decompose(c)
        {
            split(face, o, a, out);
            out.push((o, b));
        } else {
            out.push((o, c));
        }
    }

    let mut out = Vec::with_capacity(chars.len());
    for (o, c) in chars {
        split(face, o, c, &mut out);
    }

    // each run of marks sorted by combining class (stably)
    let mut i = 0;
    while i < out.len() {
        if ccc(out[i].1) == 0 {
            i += 1;
            continue;
        }
        let start = i;
        while i < out.len() && ccc(out[i].1) != 0 {
            i += 1;
        }
        out[start..i].sort_by_key(|&(_, c)| ccc(c));
    }

    // letters and marks composed where the font has the composite
    let mut res = Vec::with_capacity(out.len());
    let mut i = 0;
    while i < out.len() {
        let (o, mut base) = out[i];
        i += 1;
        if ccc(base) != 0 {
            res.push((o, base));
            continue;
        }
        let mut kept: Vec<(usize, u32)> = Vec::new();
        let mut last = 0;
        while i < out.len() && ccc(out[i].1) != 0 {
            let m = out[i].1;
            let class = ccc(m);
            match compose_pair(base, m).or_else(|| if hebrew_forms { compose_hebrew(base, m) } else { None }) {
                Some(whole) if (kept.is_empty() || last < class) && face.glyph_index(whole) != 0 => base = whole,
                _ => {
                    kept.push(out[i]);
                    last = class;
                }
            }
            i += 1;
        }
        res.push((o, base));
        res.extend(kept);
    }
    res
}

/// Characters drawn as nothing (joiners, variation selectors, bidi controls).
fn is_ignorable(c: u32) -> bool {
    matches!(c, 0xAD | 0x34F | 0x61C | 0x115F..=0x1160 | 0x17B4..=0x17B5 | 0x180B..=0x180F | 0x200B..=0x200F | 0x202A..=0x202E | 0x2060..=0x206F | 0x3164 | 0xFE00..=0xFE0F | 0xFEFF | 0xFFA0 | 0xE0000..=0xE0FFF)
}

/// Combining marks of the scripts the default shaper takes: each in the
/// cluster of the letter before it.
fn is_combining(c: u32) -> bool {
    matches!(c,
        0x300..=0x36F | 0x483..=0x489 | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF | 0x20D0..=0x20FF | 0xFE20..=0xFE2F
        | 0xF18..=0xF19 | 0xF35 | 0xF37 | 0xF39 | 0xF3E..=0xF3F | 0xF71..=0xF84 | 0xF86..=0xF87 | 0xF8D..=0xFBC | 0xFC6
        | 0x135D..=0x135F | 0x302A..=0x302F | 0x3099..=0x309A | 0x1F3FB..=0x1F3FF)
}

/// Scripts HarfBuzz shapes with its default shaper (no reordering, no
/// joining), beyond Latin, Greek and Cyrillic: their OpenType tags.
fn default_script(c: u32) -> Option<[u8; 4]> {
    Some(match c {
        0x530..=0x58F | 0xFB13..=0xFB17 => *b"armn",
        0x10A0..=0x10FF | 0x1C90..=0x1CBF | 0x2D00..=0x2D2F => *b"geor",
        0xF00..=0xFFF => *b"tibt",
        0x1200..=0x139F | 0x2D80..=0x2DDF | 0xAB00..=0xAB2F => *b"ethi",
        0x13A0..=0x13FF | 0xAB70..=0xABBF => *b"cher",
        0x1400..=0x167F | 0x18B0..=0x18FF => *b"cans",
        0x1680..=0x169F => *b"ogam",
        0x13000..=0x1345F => *b"egyp",
        0x16A0..=0x16FF => *b"runr",
        0x3040..=0x30FF | 0x31F0..=0x31FF => *b"kana",
        0x2E80..=0x2FDF | 0x3000..=0x303F | 0x3190..=0x319F | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0x20000..=0x3FFFF => *b"hani",
        0x3100..=0x312F | 0x31A0..=0x31BF => *b"bopo",
        0xAC00..=0xD7AF | 0x3130..=0x318F => *b"hang",
        0xFF00..=0xFFEF | 0xFE30..=0xFE4F | 0x2C00..=0x2C5F | 0x1D00..=0x1DBF | 0xA720..=0xA7FF => *b"latn",
        0x1F000..=0x1FAFF => *b"DFLT",
        c if is_ignorable(c) || is_combining(c) => *b"latn",
        _ => return None,
    })
}

/// Characters HarfBuzz shapes left to right with its default shaper:
/// Latin, Greek, Cyrillic and the symbol and punctuation blocks.
fn is_simple_script(c: u32) -> bool {
    matches!(c,
        0x20..=0x7E | 0xA0..=0x52F | 0x1D00..=0x1FFF | 0x2000..=0x2BFF | 0x2C60..=0x2C7F | 0x2DE0..=0x2E7F
        | 0xA640..=0xA69F | 0xA720..=0xA7FF | 0xAB30..=0xAB6F | 0xFB00..=0xFB06 | 0xFFFD)
}

/// The right-to-left scripts this shaper knows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Script {
    Simple,
    Arabic,
    Hebrew,
}

/// A mark of the Arabic or Hebrew blocks, which rides on the letter before it.
fn is_rtl_mark(c: u32) -> bool {
    matches!(c, 0x591..=0x5BD | 0x5BF | 0x5C1..=0x5C2 | 0x5C4..=0x5C5 | 0x5C7 | 0x610..=0x61A | 0x64B..=0x65F | 0x670 | 0x6D6..=0x6DC | 0x6DF..=0x6E4 | 0x6E7..=0x6E8 | 0x6EA..=0x6ED | 0x8D3..=0x8FF)
}

/// How a character joins (Unicode's `ArabicShaping.txt`): `D` both sides,
/// `R` to the right only (the one before it), `L` to the left only, `C`
/// joins both but has no forms, `T` transparent, `U` not at all.
fn joining(c: u32) -> u8 {
    let t = joining::JOINING;
    match t.binary_search_by(|&(a, b, _)| if c < a { std::cmp::Ordering::Greater } else if c > b { std::cmp::Ordering::Less } else { std::cmp::Ordering::Equal }) {
        Ok(i) => t[i].2,
        Err(_) => b'U',
    }
}

/// The scripts shaped as HarfBuzz's Arabic shaper shapes them (their
/// letters joining): their tags, and whether right to left.
fn joining_script(c: u32) -> Option<([u8; 4], bool)> {
    Some(match c {
        0x600..=0x6FF | 0x750..=0x77F | 0x870..=0x8FF | 0xFB50..=0xFDFF | 0xFE70..=0xFEFF | 0x10EC0..=0x10EFF => (*b"arab", true),
        0x700..=0x74F | 0x860..=0x86F => (*b"syrc", true),
        0x7C0..=0x7FF => (*b"nko ", true),
        0x840..=0x85F => (*b"mand", true),
        0x1800..=0x18AF | 0x11660..=0x1167F => (*b"mong", false),
        0x1E900..=0x1E95F => (*b"adlm", true),
        0x10AC0..=0x10AFF => (*b"mani", true),
        0x10B80..=0x10BAF => (*b"phlp", true),
        0x10D00..=0x10D3F => (*b"rohg", true),
        0x10F30..=0x10F6F => (*b"sogd", true),
        0xA840..=0xA87F => (*b"phag", false),
        _ => return None,
    })
}

/// Right-to-left scripts whose letters do not join: their tags.
fn plain_rtl_script(c: u32) -> Option<[u8; 4]> {
    Some(match c {
        0x590..=0x5FF | 0xFB1D..=0xFB4F => *b"hebr",
        0x780..=0x7BF => *b"thaa",
        0x800..=0x83F => *b"samr",
        _ => return None,
    })
}

/// Each character's joining form, by its neighbours (transparent ones
/// passed over): `FORM_*`, 0 for what takes no form.
fn joining_forms(chars: &[u32]) -> Vec<u8> {
    let mut forms = vec![0u8; chars.len()];
    let mut prev: Option<usize> = None;

    for i in 0..chars.len() {
        let t = joining(chars[i]);

        if t == b'T' {
            continue;
        }

        let joins_prev = prev.is_some_and(|p| matches!(joining(chars[p]), b'D' | b'C' | b'L')) && matches!(t, b'D' | b'R' | b'C');

        if joins_prev {
            let p = prev.unwrap();
            forms[p] = match forms[p] {
                FORM_ISOL => FORM_INIT,
                FORM_FINA => FORM_MEDI,
                f => f,
            };
        }

        forms[i] = match t {
            b'D' | b'R' if joins_prev => FORM_FINA,
            b'D' | b'R' | b'L' => FORM_ISOL,
            _ => 0,
        };
        prev = Some(i);
    }

    forms
}

/// Glyphs in their vertical forms, for text set upright in a vertical
/// line (punctuation turned and moved, as Japanese and Chinese set it):
/// the face's `vert` substitutions for the script `tag` (`hani`, `kana`,
/// `hang`…), as HarfBuzz applies them in vertical text.
pub fn vertical_forms(face: &Face, glyphs: &[u16], tag: [u8; 4]) -> Vec<u16> {
    let shaper = Shaper::new(face);
    let mut buf: Vec<Info> = glyphs.iter().enumerate().map(|(i, &g)| Info { glyph: g, cluster: i as u32, class: shaper.glyph_class(g), ..Default::default() }).collect();
    if let Some(gsub) = Layout::of(face, b"GSUB").map(|l| l.with_script(tag)) {
        for l in gsub.lookups_for(&[b"vert"]) {
            if shaper.gsub_lookup(&gsub, l, &mut buf).is_err() {
                break;
            }
        }
    }
    // one glyph for one (vertical forms are single substitutions)
    if buf.len() == glyphs.len() { buf.iter().map(|g| g.glyph).collect() } else { glyphs.to_vec() }
}

/// A character's vertical presentation form, where Unicode has one.
fn vertical_char(c: u32) -> Option<u32> {
    Some(match c {
        0x2013 => 0xFE32,
        0x2014 => 0xFE31,
        0x2025 => 0xFE30,
        0x2026 => 0xFE19,
        0x3001 => 0xFE11,
        0x3002 => 0xFE12,
        0x3008 => 0xFE3F,
        0x3009 => 0xFE40,
        0x300A => 0xFE3D,
        0x300B => 0xFE3E,
        0x300C => 0xFE41,
        0x300D => 0xFE42,
        0x300E => 0xFE43,
        0x300F => 0xFE44,
        0x3010 => 0xFE3B,
        0x3011 => 0xFE3C,
        0x3014 => 0xFE39,
        0x3015 => 0xFE3A,
        0x3016 => 0xFE17,
        0x3017 => 0xFE18,
        0xFE4F => 0xFE34,
        0xFF01 => 0xFE15,
        0xFF08 => 0xFE35,
        0xFF09 => 0xFE36,
        0xFF0C => 0xFE10,
        0xFF1A => 0xFE13,
        0xFF1B => 0xFE14,
        0xFF1F => 0xFE16,
        0xFF3B => 0xFE47,
        0xFF3D => 0xFE48,
        0xFF3F => 0xFE33,
        0xFF5B => 0xFE37,
        0xFF5D => 0xFE38,
        _ => return None,
    })
}

/// The script tag HarfBuzz picks for text set vertically, from its first
/// letter of a script: East Asian, else Latin.
fn vertical_script(text: &str) -> [u8; 4] {
    for c in text.chars() {
        let tag = match c as u32 {
            0x3040..=0x30FF | 0x31F0..=0x31FF | 0xFF66..=0xFF9F => *b"kana",
            0x1100..=0x11FF | 0x3130..=0x318F | 0xA960..=0xA97F | 0xAC00..=0xD7FF => *b"hang",
            0x3100..=0x312F | 0x31A0..=0x31BF => *b"bopo",
            0x2E80..=0x2FDF | 0x3005 | 0x3007 | 0x3021..=0x3029 | 0x3038..=0x303B | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0x20000..=0x3FFFF => *b"hani",
            0x41..=0x5A | 0x61..=0x7A | 0xC0..=0x24F => *b"latn",
            _ => continue,
        };
        return tag;
    }
    *b"DFLT"
}

/// `hb_shape` of text set top to bottom (`HB_DIRECTION_TTB`): the
/// substitutions of every direction and `vert`; each glyph advancing down
/// by its vertical advance (`vmtx`, else the face's height) and placed
/// from its vertical origin (`VORG`, else its box's top and `vmtx`'s top
/// side bearing, else the ascender), centred across the line; marks
/// attached by GPOS. Positions as [`shape`]'s (y up, advances negative
/// going down). Not kerned (`vkrn` is not on by default in HarfBuzz either).
pub fn shape_vertical(face: &Face, size: &Size, text: &str) -> Result<Vec<GlyphPosition>, Unsupported> {
    let shaper = Shaper::new(face);
    let tag = vertical_script(text);
    let mut chars: Vec<(usize, u32)> = normalize(face, text.char_indices().map(|(o, c)| (o, c as u32)).collect(), false);
    // Apple's substitutions in vertical text only where there are no
    // OpenType ones (as HarfBuzz)
    let aat = face.table(b"morx").is_some() && face.table(b"GSUB").is_none();
    // a face with no `vert` of its own: the punctuation's vertical
    // presentation forms where it has them (HarfBuzz's hb_vert_char_for)
    let has_vert = !aat && Layout::of(face, b"GSUB").is_some_and(|g| !g.with_script(tag).lookups_for(&[b"vert"]).is_empty());
    if !has_vert {
        for (_, c) in &mut chars {
            if let Some(v) = vertical_char(*c)
                && face.glyph_index(v) != 0
            {
                *c = v;
            }
        }
    }
    let mut buf: Vec<Info> = Vec::with_capacity(chars.len());
    for &(o, c) in &chars {
        let glyph = face.glyph_index(c);
        let mark = ccc(c) != 0 || is_combining(c);
        let cluster = match buf.last() {
            Some(last) if mark => last.cluster,
            _ => o as u32,
        };
        let class = if !shaper.has_classes() && is_nonspacing(c) { MARK } else { shaper.glyph_class(glyph) };
        buf.push(Info { glyph, cluster, class, ccc: placement_class(c, ccc(c)), ..Default::default() });
    }
    if aat {
        // Apple's substitutions, with their vertical forms (feature 4, setting 0)
        aat::morx_with(face, &mut buf, false, true, &[(4, 0)]);
        if shaper.has_classes() {
            for g in &mut buf {
                g.class = shaper.glyph_class(g.glyph);
            }
        }
    } else if let Some(gsub) = Layout::of(face, b"GSUB").map(|l| l.with_script(tag)) {
        for stage in [&[b"rvrn"][..], &[b"ccmp", b"locl", b"rlig", b"vert"]] {
            for l in gsub.lookups_for(stage) {
                shaper.gsub_lookup(&gsub, l, &mut buf)?;
            }
        }
    }

    // the scale as `position` has it
    let xs = size.metrics.x_scale;
    let upem = i64::from(face.units_per_em);
    let hb_scale = ((xs as u64 * upem as u64 + (1 << 15)) >> 16) as i64;
    let mult = (hb_scale << 16) / upem;
    let em = move |v: i32| ((i64::from(v) * mult + 32768) >> 16) as i32;

    // the face's ascender and descender as HarfBuzz reads them
    let (asc, desc) = match face.os2 {
        Some(o) if o.fs_selection & 0x80 != 0 && (o.typo_ascender != 0 || o.typo_descender != 0) => (i32::from(o.typo_ascender), i32::from(o.typo_descender)),
        _ if face.ascender != 0 || face.descender != 0 => (i32::from(face.ascender), i32::from(face.descender)),
        Some(o) => (i32::from(o.typo_ascender), i32::from(o.typo_descender)),
        None => (upem as i32 * 4 / 5, upem as i32 * 4 / 5 - upem as i32),
    };
    let vmtx = match (face.table(b"vhea"), face.table(b"vmtx")) {
        (Some(h), Some(m)) if h.len() >= 36 => Some((Reader::new(m), Reader::new(h).u16(34).unwrap_or(0).max(1) as usize)),
        _ => None,
    };
    // a glyph's advance down and top side bearing, from `vmtx`
    let v_metrics = |g: u16| -> Option<(i32, i32)> {
        let (r, n) = vmtx?;
        let g = g as usize;
        let adv = r.u16(4 * g.min(n - 1)).ok()?;
        let tsb = if g < n { r.i16(4 * g + 2).ok()? } else { r.i16(4 * n + 2 * (g - n)).ok()? };
        Some((i32::from(adv), i32::from(tsb)))
    };
    let vorg = face.table(b"VORG").map(Reader::new);
    let origin_y = |g: u16| -> i32 {
        if let Some(r) = vorg {
            let n = r.u16(6).unwrap_or(0) as usize;
            let (mut lo, mut hi) = (0, n);
            while lo < hi {
                let mid = (lo + hi) / 2;
                match r.u16(8 + 4 * mid).unwrap_or(0).cmp(&g) {
                    std::cmp::Ordering::Equal => return em(i32::from(r.i16(10 + 4 * mid).unwrap_or(0))),
                    std::cmp::Ordering::Less => lo = mid + 1,
                    std::cmp::Ordering::Greater => hi = mid,
                }
            }
            return em(i32::from(r.i16(4).unwrap_or(0)));
        }
        match extents(face, g) {
            Some((_, top, _, height)) => match v_metrics(g) {
                Some((_, tsb)) => em(top) + em(tsb),
                None => em(top) + ((em(asc - desc) - -em(height)) >> 1),
            },
            _ => em(asc),
        }
    };

    let mut out = Vec::with_capacity(buf.len());
    for info in &buf {
        let g = info.glyph;
        let h_adv = mul_div(i64::from(face.h_metrics(g).0), xs, 64);
        let h_adv = ((h_adv + (1 << 9)) >> 10) as i32;
        let v_adv = match v_metrics(g) {
            Some((a, _)) => em(a),
            None => em(asc - desc),
        };
        let (h_adv, v_adv) = if info.hidden { (0, 0) } else { (h_adv, v_adv) };
        out.push(GlyphPosition { glyph: g, cluster: info.cluster, x_advance: 0, y_advance: -v_adv, x_offset: -(h_adv / 2), y_offset: -origin_y(g) });
    }
    // GPOS's mark attachment (the features of every direction), then
    // marks no advance down the line (as HarfBuzz)
    if !aat
        && let Some(gpos) = Layout::of(face, b"GPOS").map(|l| l.with_script(tag))
    {
        for l in gpos.lookups_for(&[b"abvm", b"blwm", b"mark", b"mkmk"]) {
            let _ = shaper.gpos_lookup(&gpos, l, &buf, &mut out, &em);
        }
    }
    // a face that cannot place marks: placed by the glyphs' boxes (as
    // horizontally, for the scripts HarfBuzz does that for)
    let unplaced = text.chars().any(|c| {
        let c = c as u32;
        indic::Indic::of(c).is_some() || useng::script(c).is_some() || is_thai(c) || matches!(c, 0xF00..=0xFFF | 0x7C0..=0x7FF | 0x1800..=0x18AF)
    });
    let kern_across = face.table(b"kern").is_some() && aat::cross_kern(face);
    if face.table(b"GPOS").is_none() && face.table(b"kerx").is_none() && !kern_across && !unplaced && buf.iter().any(|b| b.ccc != 0) {
        let widths: Vec<i32> = buf.iter().map(|b| ((mul_div(i64::from(face.h_metrics(b.glyph).0), xs, 64) + (1 << 9)) >> 10) as i32).collect();
        fallback_marks_dir(face, &buf, &mut out, false, Some(&widths), &em);
    }
    let zero = !(aat::machine_kern(face) || face.table(b"kerx").is_some() && face.table(b"GPOS").is_none());
    for (info, p) in buf.iter().zip(&mut out) {
        if info.class == MARK && zero {
            p.x_advance = 0;
            p.y_advance = 0;
        }
    }
    // what Apple's substitutions deleted: gone
    if aat {
        out.retain(|p| p.glyph != aat::DELETED_GLYPH);
    }
    Ok(out)
}

/// `hb_shape` of a line of text with a FreeType font at `size`
/// (`hb_ft_font_create_referenced`, the default load flags with no
/// hinting): the glyphs and their places in 26.6.
///
/// Text in Arabic or Hebrew is shaped right to left, as HarfBuzz shapes it
/// with its direction guessed: letters take their joining forms (`isol`,
/// `init`, `medi`, `fina`), then the ligatures (`rlig`'s lam-alef), and
/// the glyphs come back in visual order (the clusters descending), marks
/// in the cluster of the letter they sit on. Give it one direction's text
/// at a time (a paragraph's runs split by the bidirectional algorithm).
pub fn shape(face: &Face, size: &Size, text: &str) -> Result<Vec<GlyphPosition>, Unsupported> {
    let mut shaper = Shaper::new(face);

    // a face with Apple's substitutions and no OpenType ones for the text
    if let Some(buf) = aat_buffer(&shaper, face, text) {
        let rtl = text.chars().any(|c| joining_script(c as u32).is_some_and(|s| s.1) || plain_rtl_script(c as u32).is_some());
        // marks lose their advances in the scripts of HarfBuzz's default
        // shaper (the others get its "dumber" one, which leaves them)
        shaper.zero_marks = !text.chars().any(|c| {
            let c = c as u32;
            joining_script(c).is_some() || plain_rtl_script(c).is_some() || indic::Indic::of(c).is_some() || useng::script(c).is_some() || is_thai(c) || matches!(c, 0xF00..=0xFFF | 0x7C0..=0x7FF | 0x1800..=0x18AF)
        });
        shaper.aat = true;
        // marks placed by their boxes where nothing of Apple's places them,
        // but not in the scripts whose HarfBuzz shapers place none (the
        // Indic ones, Khmer, Myanmar, Thai and Lao, Tibetan, the Universal
        // Shaping Engine's with N'Ko and Mongolian)
        let unplaced = text.chars().any(|c| {
            let c = c as u32;
            indic::Indic::of(c).is_some() || useng::script(c).is_some() || is_thai(c) || matches!(c, 0xF00..=0xFFF | 0x7C0..=0x7FF | 0x1800..=0x18AF)
        });
        let fallback = face.table(b"kerx").is_none() && !unplaced;
        // GPOS (where it applies) for the text's script, as HarfBuzz picks its tag
        let gpos = Layout::of(face, b"GPOS");
        let tag = text
            .chars()
            .find_map(|c| {
                let c = c as u32;
                indic::Indic::of(c)
                    .map(|s| {
                        let (v2, v1) = s.tags();
                        if gpos.as_ref().is_some_and(|g| !g.has_script(&v2)) { v1 } else { v2 }
                    })
                    .or_else(|| joining_script(c).map(|s| s.0))
                    .or_else(|| plain_rtl_script(c))
                    .or_else(|| is_thai(c).then_some(if (0xE80..=0xEFF).contains(&c) { *b"lao " } else { *b"thai" }))
                    .or_else(|| useng::script(c))
                    .or_else(|| default_script(c).filter(|t| t != b"DFLT"))
            })
            .unwrap_or(*b"latn");
        return Ok(position(&shaper, size, &buf, tag, rtl, fallback));
    }

    // a script with a shaper of its own, in a face made for no script in
    // particular: the default shaper (as HarfBuzz)
    let gsub_has = |tags: &[[u8; 4]]| Layout::of(face, b"GSUB").is_some_and(|g| tags.iter().any(|t| g.has_script(t)));
    let own = text.chars().find_map(|c| {
        let c = c as u32;
        indic::Indic::of(c).map(|s| {
            let (a, b) = s.tags();
            vec![a, b]
        })
        .or_else(|| useng::script(c).map(|t| vec![t]))
    });
    if let Some(tags) = own
        && !gsub_has(&tags)
    {
        let buf = default_buffer(&shaper, face, text);
        let mut buf = buf;
        if let Some(gsub) = Layout::of(face, b"GSUB").map(|l| l.with_script(*b"DFLT")) {
            for stage in [&GSUB_FEATURES[..1], &GSUB_FEATURES[1..]] {
                for l in gsub.lookups_for(stage) {
                    shaper.gsub_lookup(&gsub, l, &mut buf)?;
                }
            }
        }
        return Ok(position(&shaper, size, &buf, *b"DFLT", false, false));
    }

    // the Indic scripts and Khmer, syllable by syllable
    if let Some(script) = text.chars().find_map(|c| indic::Indic::of(c as u32)) {
        shaper.zero_marks = false;
        let buf = indic::substitute(&shaper, face, script, text)?;
        let (v2, v1) = script.tags();
        let tag = match Layout::of(face, b"GPOS") {
            Some(g) if !g.has_script(&v2) => v1,
            _ => v2,
        };
        return Ok(position(&shaper, size, &buf, tag, false, false));
    }

    // the scripts of the Universal Shaping Engine
    if let Some(tag) = text.chars().find_map(|c| useng::script(c as u32)) {
        shaper.zero_marks = false;
        shaper.zero_early = true;
        let buf = useng::substitute(&shaper, face, tag, text)?;
        return Ok(position(&shaper, size, &buf, tag, false, false));
    }

    // Thai and Lao: SARA AM split, marks in their letter's cluster
    if text.chars().any(|c| is_thai(c as u32)) {
        let lao = text.chars().any(|c| (0xE80..=0xEFF).contains(&(c as u32)));
        let tag = if lao { *b"lao " } else { *b"thai" };
        let mut buf = thai_buffer(&shaper, face, text)?;
        if let Some(gsub) = Layout::of(face, b"GSUB").map(|l| l.with_script(tag)) {
            for stage in [&GSUB_FEATURES[..1], &GSUB_FEATURES[1..]] {
                for l in gsub.lookups_for(stage) {
                    shaper.gsub_lookup(&gsub, l, &mut buf)?;
                }
            }
        }
        return Ok(position(&shaper, size, &buf, tag, false, false));
    }

    let (script, script_tag, rtl) = if let Some((tag, rtl)) = text.chars().find_map(|c| joining_script(c as u32)) {
        (Script::Arabic, tag, rtl)
    } else if let Some(tag) = text.chars().find_map(|c| plain_rtl_script(c as u32)) {
        (Script::Hebrew, tag, true)
    } else {
        (Script::Simple, *b"latn", false)
    };

    // characters to glyphs, each its byte offset as its cluster (a mark,
    // in right-to-left text, its letter's)
    let mut buf: Vec<Info> = Vec::with_capacity(text.len());
    let mut chars: Vec<(usize, u32)> = text.char_indices().map(|(o, c)| (o, c as u32)).collect();
    if script == Script::Simple && chars.iter().any(|&(_, c)| c >= 0xC0) {
        chars = normalize(face, chars, false);
    } else if script == Script::Arabic && chars.iter().any(|&(_, c)| ccc(c) != 0) {
        // marks in their order (shadda first), composed where the font has it
        chars = normalize(face, chars, false);
    } else if script_tag == *b"hebr" {
        // points in their order; a font that cannot place them, its forms with them
        let places = Layout::of(face, b"GPOS").is_some_and(|g| !g.with_script(*b"hebr").lookups_for(&[b"mark"]).is_empty());
        chars = normalize(face, chars, !places);
    }
    let codes: Vec<u32> = chars.iter().map(|c| c.1).collect();
    let forms = if script == Script::Arabic { joining_forms(&codes) } else { vec![0; codes.len()] };

    // the script of text in other scripts shaped as HarfBuzz's default
    // shaper shapes them (its tag), if any
    let mut other: Option<[u8; 4]> = None;
    let space = face.glyph_index(0x20);

    for (k, &(offset, c)) in chars.iter().enumerate() {
        let ours = match script {
            Script::Arabic => joining_script(c).is_some() || c == 0x200C || c == 0x200D,
            Script::Hebrew => plain_rtl_script(c).is_some(),
            Script::Simple => false,
        };
        // left-to-right text: marks and the scripts with no shaper of their own too
        let default = !ours && script == Script::Simple && (is_mark_or_ignorable(c) || default_script(c).is_some());

        if !ours && !default && is_mark_or_ignorable(c) {
            return Err(Unsupported("combining marks, default ignorables or variation selectors"));
        }

        if !ours && !default && !is_simple_script(c) {
            return Err(Unsupported("characters of scripts with shapers of their own"));
        }

        if default && other.is_none() {
            other = default_script(c);
        }

        let hidden = default && is_ignorable(c);
        let glyph = if hidden { space } else { face.glyph_index(c) };
        let cluster = match buf.last() {
            Some(last) if script != Script::Simple && (is_rtl_mark(c) || ours && joining(c) == b'T' && c != 0x200B) => last.cluster,
            // a mark, a joiner, a variation selector: in its letter's cluster
            Some(last) if default && (is_combining(c) || is_ignorable(c)) => last.cluster,
            _ => offset as u32,
        };
        // no glyph classes of its own: Unicode's non-spacing marks are marks (as HarfBuzz)
        let class = if !shaper.has_classes() && is_nonspacing(c) { MARK } else { shaper.glyph_class(glyph) };
        buf.push(Info { glyph, cluster, class, mask: forms[k], hidden, ccc: placement_class(c, ccc(c)), ..Default::default() });
    }

    // as HarfBuzz guesses it: the script of the first letter that has one
    let guessed = text.chars().find_map(|c| {
        let c = c as u32;
        match c {
            0x41..=0x5A | 0x61..=0x7A | 0xAA | 0xBA | 0xC0..=0xD6 | 0xD8..=0xF6 | 0xF8..=0x24F | 0x1E00..=0x1EFF => Some(*b"latn"),
            0x370..=0x373 | 0x376..=0x3FF | 0x1F00..=0x1FFF => Some(*b"grek"),
            0x400..=0x52F | 0x1C80..=0x1C8F | 0x2DE0..=0x2DFF | 0xA640..=0xA69F => Some(*b"cyrl"),
            c if is_ignorable(c) || is_combining(c) => None,
            c => default_script(c),
        }
    });
    let tag = match script {
        Script::Simple => guessed.or(other).unwrap_or(*b"latn"),
        Script::Arabic | Script::Hebrew => script_tag,
    };

    if let Some(gsub) = Layout::of(face, b"GSUB").map(|l| l.with_script(tag)) {
        if script == Script::Arabic {
            // the composition features, the forms (each on its own
            // letters), then the ligatures, as HarfBuzz's Arabic shaper
            for l in gsub.lookups_for(&[b"rvrn"]) {
                shaper.gsub_lookup(&gsub, l, &mut buf)?;
            }
            for l in gsub.lookups_for(&[b"ccmp", b"locl"]) {
                shaper.gsub_lookup(&gsub, l, &mut buf)?;
            }
            let forms: [(&[u8; 4], u8); 4] = [(b"isol", FORM_ISOL), (b"fina", FORM_FINA), (b"medi", FORM_MEDI), (b"init", FORM_INIT)];
            let tags: Vec<&[u8; 4]> = forms.iter().map(|f| f.0).collect();
            for (l, t) in gsub.lookups_tagged(&tags) {
                let mask = forms.iter().find(|f| *f.0 == t).map(|f| f.1);
                shaper.gsub_lookup_masked(&gsub, l, &mut buf, mask)?;
            }
            // HarfBuzz's stages: rlig; rclt and calt; mset with the common ligatures
            for stage in [&[b"rlig"][..], &[b"rclt", b"calt"], &[b"liga", b"clig", b"mset"]] {
                for l in gsub.lookups_for(stage) {
                    shaper.gsub_lookup(&gsub, l, &mut buf)?;
                }
            }
        } else {
            // rvrn in a stage of its own, then the rest, each in lookup order
            for stage in [&GSUB_FEATURES[..1], &GSUB_FEATURES[1..]] {
                for l in gsub.lookups_for(stage) {
                    shaper.gsub_lookup(&gsub, l, &mut buf)?;
                }
            }
        }
    }

    Ok(position(&shaper, size, &buf, tag, rtl, true))
}

/// Glyphs substituted to glyphs placed: advances, GPOS (or `kern`), marks'
/// advances taken away, and right-to-left text in visual order.
fn position(shaper: &Shaper, size: &Size, buf: &[Info], tag: [u8; 4], rtl: bool, fallback: bool) -> Vec<GlyphPosition> {
    let face = shaper.face;
    // advances: hb_ft's, FreeType's unhinted advance in 16.16, rounded
    let xs = size.metrics.x_scale;
    let mut pos: Vec<GlyphPosition> = buf
        .iter()
        .map(|info| {
            let (advance, _) = face.h_metrics(info.glyph);
            let advance = if info.hidden { 0 } else { advance };
            let v = mul_div(i64::from(advance), xs, 64);
            GlyphPosition { glyph: info.glyph, cluster: info.cluster, x_advance: ((v + (1 << 9)) >> 10) as i32, ..Default::default() }
        })
        .collect();

    // the font's scale as hb_ft sets it, and HarfBuzz's em_mult
    let upem = i64::from(face.units_per_em);
    let hb_scale = ((xs as u64 * upem as u64 + (1 << 15)) >> 16) as i64;
    let mult = (hb_scale << 16) / upem;
    let em = move |v: i32| ((i64::from(v) * mult + 32768) >> 16) as i32;

    if shaper.zero_early {
        for (info, p) in buf.iter().zip(&mut pos) {
            if info.class == MARK {
                p.x_advance = 0;
            }
        }
    }
    shaper.rtl.set(rtl);
    shaper.cursive.borrow_mut().clear();
    shaper.marks.borrow_mut().clear();

    // as HarfBuzz chooses: Apple's `kerx` unless GSUB (applied) and GPOS
    // are both there, else GPOS; and `kerx` or `kern` too when GPOS has
    // no kerning
    let has_kerx = face.table(b"kerx").is_some();
    let has_gsub = !shaper.aat && face.table(b"GSUB").is_some();
    let mut gpos = Layout::of(face, b"GPOS").map(|l| l.with_script(tag));
    // Hebrew is positioned by a GPOS that has Hebrew, or not at all (as HarfBuzz)
    if tag == *b"hebr" && gpos.as_ref().is_some_and(|g| !g.has_script(b"hebr")) {
        gpos = None;
    }
    if has_kerx && !(has_gsub && gpos.is_some()) {
        gpos = None;
    }
    let gpos_kern = gpos.as_ref().is_some_and(|g| !g.lookups_for(&[b"kern"]).is_empty());

    if let Some(gpos) = &gpos {
        for l in gpos.lookups_for(&GPOS_FEATURES) {
            let _ = shaper.gpos_lookup(gpos, l, buf, &mut pos, &em);
        }
    }

    // pair kerning (hb_kern_machine_t): each pair's value split, the
    // second half on the right glyph's advance and offset; marks passed
    // over; in visual order where the table reads that way
    let pairs = |pos: &mut [GlyphPosition], reverse: bool, kern: &dyn Fn(u16, u16) -> Option<i32>| {
        let order: Vec<usize> = if reverse { (0..buf.len()).rev().collect() } else { (0..buf.len()).collect() };
        for (n, &i) in order.iter().enumerate() {
            if buf[i].class == MARK {
                continue;
            }
            let Some(&j) = order[n + 1..].iter().find(|&&j| buf[j].class != MARK) else { break };
            let Some(v) = kern(buf[i].glyph, buf[j].glyph) else { continue };
            let v = em(v);
            if v == 0 {
                continue;
            }
            let k1 = v >> 1;
            let k2 = v - k1;
            pos[i].x_advance += k1;
            pos[j].x_advance += k2;
            pos[j].x_offset += k2;
        }
    };

    // Apple's `kerx`: in place of `kern`
    let kerx = has_kerx && (gpos.is_none() || !gpos_kern);
    if kerx {
        pairs(&mut pos, aat::kerx_backwards(face) != rtl, &|a, b| aat::kerx_pair(face, a, b));
        aat::kerx_states(face, buf, &mut pos, rtl, &em);
    }

    // `kerx`'s attachments: each glyph by the one it is attached to (as
    // GPOS attaches marks), after the pen has passed what is between
    if kerx {
        for (i, target, dx, dy) in aat::kerx_attachments(face, buf, rtl) {
            if target >= i {
                continue;
            }
            // from the pen where the target was (left to right: back over
            // what is between; right to left: on over it and itself)
            let between: i32 = if rtl { (target + 1..=i).map(|k| pos[k].x_advance).sum() } else { -(target..i).map(|k| pos[k].x_advance).sum::<i32>() };
            pos[i].x_offset = pos[target].x_offset + em(dx) + between;
            pos[i].y_offset = pos[target].y_offset + em(dy);
        }
    }

    let kern = !kerx && !gpos_kern && face.table(b"kern").is_some();
    if kern {
        pairs(&mut pos, rtl, &|a, b| shaper.kern_pair(a, b));
    }

    // Apple's contextual kerning (`kern` format 1), with its pairs
    if !kerx && !gpos_kern {
        aat::kern_states(face, buf, &mut pos, rtl, &em);
    }

    // Apple's tracking: on the first glyph of each cluster (not with
    // Apple's substitutions, as HarfBuzz leaves it then), before marks
    // are placed by their glyphs' boxes
    let track = if shaper.aat { 0.0 } else { face.tracking() };
    if track != 0.0 {
        let add = crate::var::hb_round((f64::from(track) * mult as f64 / 65536.0) as f32) as i32;
        let mut last = None;
        for p in &mut pos {
            if last != Some(p.cluster) {
                p.x_advance += add;
            }
            last = Some(p.cluster);
        }
    }

    // a font that cannot place marks: placed from the glyphs' boxes
    // (as HarfBuzz: not where `kerx`, or a `kern` that moves across the line, places them)
    let adjust = gpos.is_none() && !kerx && !(kern && aat::cross_kern(face));
    if fallback && adjust && buf.iter().any(|b| b.ccc != 0) {
        fallback_marks(face, buf, &mut pos, rtl, &em);
    }

    // zero the advances of marks (by GDEF, after positioning); not where
    // `kerx` places them without GPOS, or `kern`'s state machines do (as HarfBuzz)
    let zero = shaper.zero_marks && !(kerx && gpos.is_none()) && !(kern && aat::machine_kern(face));
    for (info, p) in buf.iter().zip(&mut pos) {
        if info.class == MARK && zero {
            p.x_advance = 0;
            p.y_advance = 0;
        }
    }

    shaper.propagate_cursive(&mut pos);

    if rtl {
        // a mark was placed from the pen after its letter; drawn right to
        // left, its pen is the letter's own
        let mut base = None;
        let mut since = 0;
        let attached = shaper.marks.borrow();
        for i in 0..pos.len() {
            if buf[i].class == MARK {
                // (a mark nothing attached stays where its pen left it)
                if base.is_some() && attached.get(i).is_some_and(Option::is_some) {
                    pos[i].x_offset += since;
                }
            } else {
                base = Some(i);
                since = 0;
            }
            since += pos[i].x_advance;
        }
        pos.reverse();
    }

    // glyphs Apple's substitutions deleted, kept till now: gone
    if shaper.aat {
        pos.retain(|p| p.glyph != aat::DELETED_GLYPH);
    }

    pos
}

/// Each character's glyph, a grapheme a cluster (marks with their letter),
/// for the default shaper.
fn default_buffer(shaper: &Shaper, face: &Face, text: &str) -> Vec<Info> {
    let mut buf: Vec<Info> = Vec::with_capacity(text.len());
    // composed and decomposed as the font has the characters (as HarfBuzz always does)
    let chars = normalize(face, text.char_indices().map(|(o, c)| (o, c as u32)).collect(), false);
    for (offset, c) in chars {
        let glyph = face.glyph_index(c);
        let mark = ccc(c) != 0 || is_combining(c) || indic::is_mark_char(c) || useng::is_mark_char(c) || c == 0x200D;
        let cluster = match buf.last() {
            Some(last) if mark => last.cluster,
            _ => offset as u32,
        };
        buf.push(Info { glyph, cluster, class: shaper.glyph_class(glyph), ..Default::default() });
    }
    buf
}

/// The text's glyphs through the face's `morx`, when the face has it and
/// no OpenType substitutions for the text's script: each character's
/// glyph, clusters by grapheme (marks with their letters).
fn aat_buffer(shaper: &Shaper, face: &Face, text: &str) -> Option<Vec<Info>> {
    // as HarfBuzz: Apple's substitutions whenever the face has them (in
    // horizontal text), its OpenType ones then unused
    face.table(b"morx")?;
    let mut buf: Vec<Info> = Vec::with_capacity(text.len());
    // composed and decomposed as the font has the characters (as HarfBuzz always does)
    let chars = normalize(face, text.char_indices().map(|(o, c)| (o, c as u32)).collect(), false);
    for (offset, c) in chars {
        let glyph = face.glyph_index(c);
        let mark = ccc(c) != 0 || is_combining(c) || is_rtl_mark(c) || indic::is_mark_char(c) || is_thai_mark(c) || c == 0x200D;
        let cluster = match buf.last() {
            Some(last) if mark => last.cluster,
            _ => offset as u32,
        };
        // no glyph classes of its own: Unicode's non-spacing marks are marks
        let class = if !shaper.has_classes() && is_nonspacing(c) { MARK } else { shaper.glyph_class(glyph) };
        buf.push(Info { glyph, cluster, class, ccc: placement_class(c, ccc(c)), ..Default::default() });
    }
    let rtl = text.chars().any(|c| joining_script(c as u32).is_some_and(|s| s.1) || plain_rtl_script(c as u32).is_some());
    aat::morx(face, &mut buf, rtl);
    // the glyphs it made take their classes from GDEF
    if shaper.has_classes() {
        for g in &mut buf {
            g.class = shaper.glyph_class(g.glyph);
        }
    }
    Some(buf)
}

fn is_thai(c: u32) -> bool {
    matches!(c, 0xE01..=0xE7F | 0xE80..=0xEFF)
}

/// A Thai or Lao mark: in the cluster of the letter it is on.
fn is_thai_mark(c: u32) -> bool {
    matches!(c, 0xE31 | 0xE34..=0xE3A | 0xE47..=0xE4E | 0xEB1 | 0xEB4..=0xEBC | 0xEC8..=0xECE)
}

/// Thai or Lao marks drawn above the letter.
fn is_thai_above(c: u32) -> bool {
    matches!(c, 0xE31 | 0xE34..=0xE37 | 0xE47..=0xE4E | 0xEB1 | 0xEB4..=0xEB7 | 0xEBB | 0xEC8..=0xECD)
}

/// Thai or Lao text as glyphs (HarfBuzz's Thai shaper, the fonts' own
/// features doing the rest): SARA AM as NIKHAHIT and SARA AA, the
/// NIKHAHIT before the tone marks above the letter it follows.
fn thai_buffer(shaper: &Shaper, face: &Face, text: &str) -> Result<Vec<Info>, Unsupported> {
    let mut buf: Vec<Info> = Vec::with_capacity(text.len());
    let mut codes: Vec<u32> = Vec::with_capacity(text.len());

    for (offset, ch) in text.char_indices() {
        let c = ch as u32;

        if !is_thai(c) && (is_mark_or_ignorable(c) || !is_simple_script(c)) {
            return Err(Unsupported("characters of another script among Thai text"));
        }

        let cluster = match buf.last() {
            Some(last) if is_thai_mark(c) => last.cluster,
            _ => offset as u32,
        };
        let parts: &[u32] = match c {
            0xE33 => &[0xE4D, 0xE32],
            0xEB3 => &[0xECD, 0xEB2],
            _ => &[c],
        };

        for (k, &p) in parts.iter().enumerate() {
            let glyph = face.glyph_index(p);
            let cluster = if parts.len() > 1 && buf.last().is_some() { buf.last().unwrap().cluster } else { cluster };
            buf.push(Info { glyph, cluster, class: shaper.glyph_class(glyph), ccc: placement_class(p, ccc(p)), ..Default::default() });
            codes.push(p);

            // the NIKHAHIT before the marks above that come before it
            if k == 0 && parts.len() > 1 {
                let mut at = buf.len() - 1;
                while at > 0 && is_thai_above(codes[at - 1]) && codes[at - 1] != p {
                    at -= 1;
                }
                let moved = buf.pop().unwrap();
                codes.pop();
                buf.insert(at, moved);
                codes.insert(at, p);
            }
        }
    }

    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face(path: &str) -> Option<Face> {
        let data = std::fs::read(path).ok()?;
        Face::new(data, 0).ok()
    }

    #[test]
    fn arabic_letters_join() {
        // lam, ba, kaf: init, medi, fina
        assert_eq!(joining_forms(&[0x644, 0x628, 0x643]), [FORM_INIT, FORM_MEDI, FORM_FINA]);
        // alef joins only to its right: what follows starts again
        assert_eq!(joining_forms(&[0x628, 0x627, 0x628]), [FORM_INIT, FORM_FINA, FORM_ISOL]);
        // a mark is passed over
        assert_eq!(joining_forms(&[0x628, 0x64E, 0x628]), [FORM_INIT, 0, FORM_FINA]);
    }

    #[test]
    fn arabic_is_shaped_right_to_left_in_its_forms() {
        let Some(f) = ["/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"].iter().find_map(|p| face(p)) else {
            return;
        };
        let size = Size::char_size(&f, 0, i64::from(f.units_per_em) * 64, 72, 72);
        // "بيت": three letters, the middle one medial
        let g = shape(&f, &size, "بيت").unwrap();
        assert_eq!(g.len(), 3);
        // visual order: the last letter first
        assert_eq!(g.iter().map(|g| g.cluster).collect::<Vec<_>>(), [4, 2, 0]);
        // the forms differ from the letters' own glyphs
        let isolated: Vec<u16> = "بيت".chars().map(|c| f.glyph_index(c as u32)).collect();
        assert_ne!(g[2].glyph, isolated[0], "ba took its initial form");
        // Hebrew: reversed, no forms
        let h = shape(&f, &size, "שלום").unwrap();
        assert_eq!(h.iter().map(|g| g.cluster).collect::<Vec<_>>(), [6, 4, 2, 0]);
        // a point under a letter: in its cluster, no advance, placed under it
        let p = shape(&f, &size, "בַּ").unwrap();
        assert!(p.iter().all(|g| g.cluster == 0));
        assert_eq!(p.iter().filter(|g| g.x_advance != 0).count(), 1, "{p:?}");
    }

    fn size(f: &Face) -> Size {
        Size::char_size(f, 0, i64::from(f.units_per_em) * 64, 72, 72)
    }

    #[test]
    fn devanagari_vowel_signs_and_rephs_move_where_they_are_drawn() {
        let Some(f) = ["/System/Library/Fonts/Kohinoor.ttc", "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"].iter().find_map(|p| face(p)) else {
            return;
        };
        let s = size(&f);
        let ka = f.glyph_index(0x915);
        // कि: the vowel sign I, typed after KA, drawn before it
        let g = shape(&f, &s, "कि").unwrap();
        assert_eq!(g.len(), 2);
        assert_eq!(g[1].glyph, ka, "{g:?}");
        assert!(g.iter().all(|g| g.cluster == 0), "one cluster: {g:?}");
        // र्क: RA and the virama become a reph, drawn after KA
        let g = shape(&f, &s, "र्क").unwrap();
        assert_eq!(g.len(), 2, "{g:?}");
        assert_eq!(g[0].glyph, ka);
        assert_ne!(g[1].glyph, f.glyph_index(0x930));
        // क्ष: a conjunct, one glyph
        assert_eq!(shape(&f, &s, "क्ष").unwrap().len(), 1);
        // a vowel sign with nothing to sit on: on a dotted circle
        let g = shape(&f, &s, "ि").unwrap();
        assert!(g.iter().any(|g| g.glyph == f.glyph_index(0x25CC)), "{g:?}");
        // words keep their own clusters, spaces between
        let g = shape(&f, &s, "नमस्ते दुनिया").unwrap();
        assert!(g.windows(2).all(|w| w[0].cluster <= w[1].cluster), "clusters in order: {g:?}");
    }

    #[test]
    fn two_part_vowel_signs_are_split_round_their_consonant() {
        let Some(f) = ["/System/Library/Fonts/KohinoorBangla.ttc", "/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf"].iter().find_map(|p| face(p)) else {
            return;
        };
        // কো: E before KA, AA after it
        let g = shape(&f, &size(&f), "কো").unwrap();
        assert_eq!(g.len(), 3, "{g:?}");
        assert_eq!(g[1].glyph, f.glyph_index(0x995));
        assert!(g.iter().all(|g| g.cluster == 0));
    }

    #[test]
    fn khmer_coeng_ro_and_thai_sara_am_are_reordered() {
        if let Some(f) = ["/System/Library/Fonts/Supplemental/Khmer Sangam MN.ttf", "/usr/share/fonts/truetype/noto/NotoSansKhmer-Regular.ttf"].iter().find_map(|p| face(p)) {
            // ស្រ: COENG RO comes before SA
            let g = shape(&f, &size(&f), "ស្រ").unwrap();
            assert_ne!(g[0].glyph, f.glyph_index(0x179F), "{g:?}");
            assert!(g.iter().any(|x| x.glyph == f.glyph_index(0x179F)));
        }
        if let Some(f) = ["/System/Library/Fonts/ThonburiUI.ttc", "/usr/share/fonts/truetype/tlwg/Garuda.ttf"].iter().find_map(|p| face(p)) {
            // น้ำ: NIKHAHIT before the tone mark, SARA AA last, one cluster
            let g = shape(&f, &size(&f), "น้ำ").unwrap();
            // (the font may join NIKHAHIT and the tone mark into one glyph)
            assert_eq!(g.first().map(|g| g.glyph), Some(f.glyph_index(0xE19)));
            assert_eq!(g.last().map(|g| g.glyph), Some(f.glyph_index(0xE32)), "SARA AA after the marks: {g:?}");
            assert!(g.iter().all(|g| g.cluster == 0));
        }
    }

    #[test]
    fn accents_are_composed_where_the_font_has_the_whole_and_myanmar_is_reordered() {
        if let Some(f) = ["/System/Library/Fonts/Supplemental/Times New Roman.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"].iter().find_map(|p| face(p)) {
            let s = size(&f);
            // e and a combining acute: the font's é
            let g = shape(&f, &s, "e\u{301}").unwrap();
            assert_eq!(g.iter().map(|g| g.glyph).collect::<Vec<_>>(), [f.glyph_index(0xE9)]);
            // a mark with no composite: its own glyph, in its letter's cluster
            let g = shape(&f, &s, "x\u{303}y").unwrap();
            assert_eq!(g.iter().map(|g| g.cluster).collect::<Vec<_>>(), [0, 0, 3]);
            // a joiner: not seen, no advance
            let g = shape(&f, &s, "a\u{200D}b").unwrap();
            assert_eq!(g[1].x_advance, 0);
        }
        if let Some(f) = ["/System/Library/Fonts/NotoSansMyanmar.ttc", "/usr/share/fonts/truetype/noto/NotoSansMyanmar-Regular.ttf"].iter().find_map(|p| face(p)) {
            // မြ: the medial Ra drawn before MA, one cluster
            let g = shape(&f, &size(&f), "မြ").unwrap();
            assert_eq!(g.len(), 2, "{g:?}");
            assert_eq!(g[1].glyph, f.glyph_index(0x1019));
            assert!(g.iter().all(|g| g.cluster == 0));
        }
    }

    #[test]
    fn apples_own_tables_shape_a_face_that_has_only_them() {
        // Helvetica: an fi ligature by `morx`, kerning by Apple's `kern`
        let Some(f) = face("/System/Library/Fonts/Helvetica.ttc") else { return };
        let s = size(&f);
        let g = shape(&f, &s, "office").unwrap();
        assert_eq!(g.len(), 5, "ffi made one: {g:?}");
        assert_eq!(g.iter().map(|g| g.cluster).collect::<Vec<_>>(), [0, 1, 2, 4, 5]);
        let w = shape(&f, &s, "Wa").unwrap();
        assert!(w[1].x_offset < 0, "W and a kerned: {w:?}");
        // Tibetan in Kokonor: stacks made by `morx`
        if let Some(t) = face("/System/Library/Fonts/Supplemental/Kokonor.ttf") {
            let g = shape(&t, &size(&t), "སྒྲུབ").unwrap();
            assert!(g.len() < "སྒྲུབ".chars().count(), "{g:?}");
        }
    }

    #[test]
    fn the_universal_engine_moves_vowels_before_their_bases() {
        // Javanese: the vowel sign E (typed after KA) drawn before it
        let Some(f) = face("/System/Library/Fonts/Supplemental/NotoSansJavanese-Regular.otf") else { return };
        let s = size(&f);
        let g = shape(&f, &s, "ꦏꦺ").unwrap();
        assert_eq!(g.len(), 2, "{g:?}");
        assert_eq!(g[1].glyph, f.glyph_index(0xA98F), "KA after the vowel sign: {g:?}");
        assert!(g.iter().all(|g| g.cluster == 0));
        // a vowel sign alone: on a dotted circle
        let g = shape(&f, &s, "ꦺ").unwrap();
        assert!(g.iter().any(|x| x.glyph == f.glyph_index(0x25CC)), "{g:?}");
    }

    /// Glyphs as `hb-shape --no-glyph-names --utf8-clusters` prints them
    /// (at a size of one unit to the pixel).
    fn hb(g: &[GlyphPosition]) -> String {
        g.iter()
            .map(|g| {
                let off = if g.x_offset != 0 || g.y_offset != 0 { format!("@{},{}", (g.x_offset + 32) >> 6, (g.y_offset + 32) >> 6) } else { String::new() };
                let ya = if g.y_advance != 0 { format!(",{}", (g.y_advance + 32) >> 6) } else { String::new() };
                format!("{}={}{}+{}{}", g.glyph, g.cluster, off, (g.x_advance + 32) >> 6, ya)
            })
            .collect::<Vec<_>>()
            .join("|")
    }

    fn shaped(path: &str, text: &str) -> Option<String> {
        let f = face(path)?;
        Some(hb(&shape(&f, &size(&f), text).unwrap()))
    }

    #[test]
    fn a_variable_fonts_instance_and_apples_tracking_are_as_harfbuzz_has_them() {
        // SF Pro bold: HVAR's advances, GPOS kerning's deltas
        if let Some(mut f) = face("/System/Library/Fonts/SFNS.ttf") {
            f.set_variations(&[(*b"wght", 700.0)]);
            assert_eq!(hb(&shape(&f, &size(&f), "AVAT Wa").unwrap()), "19=0+1238|301=1+1209|19=2+1223|261=3+1232|3=4+405|304=5+1842|588=6+1121");
        }
        // New York's `trak`: at 12 points unless told, as HarfBuzz
        if let Some(mut f) = face("/System/Library/Fonts/NewYork.ttf") {
            assert_eq!(hb(&shape(&f, &size(&f), "Wave").unwrap()), "251=0+1863|783=1+892|1111=2+833|845=3+952");
            f.point_size = 36.0;
            assert_eq!(hb(&shape(&f, &size(&f), "Wave").unwrap()), "251=0+1823|783=1+852|1111=2+793|845=3+912");
        }
    }

    #[test]
    fn apples_tables_are_used_as_harfbuzz_uses_them() {
        let s = "/System/Library/Fonts/Supplemental";
        let cases = [
            // `morx` before GSUB; marks sorted (shadda first) and ligated by
            // it (HarfBuzz, at whole units, has the mark at 232: half a unit)
            (format!("{s}/Damascus.ttc"), "لَّ", "167=0@233,105+0|344=0+860"),
            // `kerx` format 4: a point attached by `ankr`'s anchors, right to left
            ("/System/Library/Fonts/ArialHB.ttc".to_string(), "שָׁ", "81=0@70,0+0|157=0+694"),
            // `kern` format 1, across the line too
            ("/System/Library/Fonts/GeezaPro.ttc".to_string(), "بِسْمِ", "884=8@-80,0+-80|347=8@80,0+980|886=4@450,-200+450|278=4@-450,0+1179|884=0@80,290+80|243=0@-80,0+576"),
            // `kern` format 2
            (format!("{s}/Apple Chancery.ttf"), "Yo", "60=0+1271|82=1@-50,0+828"),
            // what `morx` puts in: in the cluster of the glyph it goes with
            (format!("{s}/Bangla MN.ttc"), "কো", "214=0+719|133=0+1330|167=0+474"),
            // a symbol font's characters at U+F000 up
            (format!("{s}/Wingdings.ttf"), "ab", "72=0+2233|73=1+1927"),
        ];
        for (path, text, want) in cases {
            if let Some(got) = shaped(&path, text) {
                assert_eq!(got, want, "{path} {text}");
            }
        }
    }

    #[test]
    fn marks_go_on_the_ligature_component_they_followed() {
        // Noto Nastaliq: the shadda and fatha on the second lam of the
        // lam-lam-heh ligature, the kasra on its heh
        if let Some(got) = shaped("/System/Library/Fonts/NotoNastaliq.ttc", "اللَّهِ") {
            assert_eq!(got, "44=2@183,70+0|42=2@528,-93+0|51=2@528,-256+0|222=2+802|224=0+330");
        }
    }

    #[test]
    fn vertical_text_takes_its_vertical_forms() {
        // Hiragino: brackets, comma, full stop and the long vowel mark turned, as HarfBuzz has them
        let Some(f) = face("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc") else { return };
        let glyphs: Vec<u16> = "「日本、。ー」".chars().map(|c| f.glyph_index(c as u32)).collect();
        assert_eq!(vertical_forms(&f, &glyphs, *b"hani"), [7911, 3284, 3722, 7887, 7888, 7891, 7912]);
    }

    #[test]
    fn text_set_vertically_is_as_harfbuzz_sets_it() {
        let cases = [
            // GSUB's `vert`, `VORG`, `vmtx`
            ("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", "7911=0@-500,-880+0,-1000|3284=3@-500,-880+0,-1000|3722=6@-500,-880+0,-1000|7887=9@-500,-880+0,-1000|7888=12@-500,-880+0,-1000|7891=15@-500,-880+0,-1000|7912=18@-500,-880+0,-1000"),
            // `morx`'s vertical subtable, and the vertical presentation forms
            ("/System/Library/Fonts/STHeiti Light.ttc", "49637=0@-500,-860+0,-1000|363=3@-500,-860+0,-1000|469=6@-500,-860+0,-1000|49264=9@-500,-860+0,-1000|49266=12@-500,-860+0,-1000|49775=15@-500,-860+0,-1000|49638=18@-500,-860+0,-1000"),
            // the origin from the glyph's box and its top side bearing
            ("/System/Library/Fonts/AppleSDGothicNeo.ttc", "8069=0@-432,-900+0,-1200|6464=3@-432,-900+0,-1200|5091=6@-432,-900+0,-1200|8056=9@-432,-900+0,-1200|8057=12@-432,-900+0,-1200|9330=15@-432,-900+0,-1200|8070=18@-432,-900+0,-1200"),
        ];
        for (path, want) in cases {
            let Some(f) = face(path) else { continue };
            assert_eq!(hb(&shape_vertical(&f, &size(&f), "「日本、。ー」").unwrap()), want, "{path}");
        }
    }

    #[test]
    fn marks_in_vertical_text_are_placed_as_harfbuzz_places_them() {
        let cases = [
            // GPOS's mark attachment, down the line
            ("/System/Library/Fonts/Supplemental/Arial.ttf", "x̂ ạ", "91=0@-512,-1675+0,-2288|2114=0@-12,638+0|3=3@-284,-1144+0,-2288|1189=4@-569,-1524+0,-2288"),
            // no GPOS: placed by the glyphs' boxes
            ("/System/Library/Fonts/Times.ttc", "x̂", "91=0@-512,-1483+0,-2048|367=0@226,1361+0"),
        ];
        for (path, text, want) in cases {
            let Some(f) = face(path) else { continue };
            assert_eq!(hb(&shape_vertical(&f, &size(&f), text).unwrap()), want, "{path}");
        }
    }
}
