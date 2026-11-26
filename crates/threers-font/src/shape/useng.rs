//! The Universal Shaping Engine, as HarfBuzz runs it, for the many
//! scripts with no shaper of their own (Javanese, Balinese, Sundanese, Tai
//! Tham, Chakma, Batak…): the text cut into clusters by the engine's
//! grammar (joiners passed over), marks with no base on a dotted circle;
//! `locl`, `ccmp`, `nukt`, `akhn`; `rphf` on a cluster's start (what it
//! makes becomes a reph), `pref` (what it makes goes before the base);
//! the basic features at once; the reph moved on to before the first
//! post-base glyph, pre-base vowels moved back to the cluster's start
//! (after any virama); the presentation features last.

use super::use_table::*;
use super::*;

/// The engine's category of a character of its scripts (`None`: not one of them).
fn category(c: u32) -> Option<u8> {
    let t = USE;
    t.binary_search_by(|&(a, b, _)| if c < a { std::cmp::Ordering::Greater } else if c > b { std::cmp::Ordering::Less } else { std::cmp::Ordering::Equal })
        .ok()
        .map(|i| t[i].2)
}

/// The OpenType tag of the engine's script a character is in.
pub(super) fn script(c: u32) -> Option<[u8; 4]> {
    let t = TAGS;
    t.binary_search_by(|&(a, b, _)| if c < a { std::cmp::Ordering::Greater } else if c > b { std::cmp::Ordering::Less } else { std::cmp::Ordering::Equal })
        .ok()
        .map(|i| t[i].2)
}

/// A mark of the engine's scripts (what goes on in the cluster before it).
pub(super) fn is_mark_char(c: u32) -> bool {
    category(c).is_some_and(|k| !matches!(k, B | GB | N | O | R | CS))
}

const CGJ_: u8 = 0xF0;
const ZWNJ_: u8 = 0xF1;
const OTHER: u8 = 0xF2;

fn is_h(c: u8) -> bool {
    matches!(c, H | IS | SK)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Cluster,
    Broken,
    Other,
}

/// A greedy reading of the engine's grammar from `i` (over categories with
/// joiners left out): where each kind of cluster would end.
struct Grammar<'a> {
    c: &'a [u8],
}

impl Grammar<'_> {
    fn at(&self, i: usize) -> u8 {
        self.c.get(i).copied().unwrap_or(0xFF)
    }

    fn many(&self, mut i: usize, set: &[u8]) -> usize {
        while set.contains(&self.at(i)) {
            i += 1;
        }
        i
    }

    fn opt(&self, i: usize, set: &[u8]) -> usize {
        if set.contains(&self.at(i)) { i + 1 } else { i }
    }

    fn start(&self, i: usize) -> Option<usize> {
        let i = self.opt(i, &[R, CS]);
        matches!(self.at(i), B | GB).then_some(i + 1)
    }

    fn consonant_modifiers(&self, mut i: usize) -> usize {
        i = self.many(i, &[CMABV]);
        i = self.many(i, &[CMBLW]);
        loop {
            let j = if is_h(self.at(i)) && self.at(i + 1) == B {
                i + 2
            } else if self.at(i) == SUB {
                i + 1
            } else {
                return i;
            };
            i = self.many(self.many(j, &[CMABV]), &[CMBLW]);
        }
    }

    fn middle(&self, mut i: usize) -> usize {
        i = self.consonant_modifiers(i);
        for set in [MPRE, MABV, MBLW, MPST] {
            i = self.opt(i, &[set]);
        }
        if self.at(i) == H {
            i += 1;
        } else {
            for set in [VPRE, VABV, VBLW, VPST] {
                i = self.many(i, &[set]);
            }
        }
        for set in [VMPRE, VMABV, VMBLW, VMPST] {
            i = self.many(i, &[set]);
        }
        while self.at(i) == SK && self.at(i + 1) == B {
            i += 2;
        }
        i
    }

    fn tail(&self, i: usize) -> usize {
        let mut i = self.middle(i);
        for set in [FABV, FBLW, FPST] {
            i = self.many(i, &[set]);
        }
        let j = self.many(self.many(i, &[FMABV]), &[FMBLW]);
        if j > i { j } else { i }
    }

    /// The longest cluster from `i`, and its kind.
    fn cluster(&self, i: usize) -> (usize, Kind) {
        if let Some(s) = self.start(i) {
            let standard = self.tail(s);
            let virama = {
                let j = self.consonant_modifiers(s);
                if matches!(self.at(j), IS | RK) { j + 1 } else { 0 }
            };
            let sakot = {
                let j = self.middle(s);
                if self.at(j) == SK { j + 1 } else { 0 }
            };
            let end = standard.max(virama).max(sakot);
            return (self.opt(end, &[ZWNJ_]), Kind::Cluster);
        }
        if self.at(i) == N {
            let mut j = i + 1;
            while self.at(j) == HN && self.at(j + 1) == N {
                j += 2;
            }
            return (self.opt(self.opt(j, &[HN]), &[ZWNJ_]), Kind::Cluster);
        }
        if matches!(self.at(i), O | GB) {
            let j = self.tail(i + 1).max(self.many(self.many(i + 1, &[SMABV]), &[0xFE]));
            return (self.opt(j.max(i + 1), &[ZWNJ_]), Kind::Cluster);
        }
        // marks with nothing to sit on
        let r = self.opt(i, &[R]);
        let t = self.tail(r);
        if t > r || (r > i && t == r) {
            return (self.opt(t.max(i + 1), &[ZWNJ_]), Kind::Broken);
        }
        (i + 1, Kind::Other)
    }
}

/// Clusters over the characters: from, to (joiners in the cluster before).
fn clusters(cats: &[u8]) -> Vec<(usize, usize, Kind)> {
    let index: Vec<usize> = (0..cats.len()).filter(|&k| cats[k] != CGJ_).collect();
    let filtered: Vec<u8> = index.iter().map(|&k| cats[k]).collect();
    let g = Grammar { c: &filtered };
    let mut out = Vec::new();
    let mut i = 0;
    let mut from = 0;
    while i < filtered.len() {
        let (end, kind) = g.cluster(i);
        let to = if end < index.len() { index[end] } else { cats.len() };
        out.push((from, to, kind));
        from = to;
        i = end;
    }
    if from < cats.len() {
        if let Some(last) = out.last_mut() {
            last.1 = cats.len();
        } else {
            out.push((from, cats.len(), Kind::Other));
        }
    }
    out
}

const F_RPHF: u32 = 2;

/// The text's glyphs, substituted, with the engine.
pub(super) fn substitute(shaper: &Shaper, face: &Face, tag: [u8; 4], text: &str) -> Result<Vec<Info>, Unsupported> {
    let gsub = Layout::of(face, b"GSUB").map(|l| l.with_script(tag));
    let mut buf: Vec<Info> = Vec::with_capacity(text.len());
    let mut cats: Vec<u8> = Vec::with_capacity(text.len());
    for (offset, ch) in text.char_indices() {
        let c = ch as u32;
        let cat = match c {
            0x200D | 0x34F | 0xFE00..=0xFE0F => CGJ_,
            0x200C => ZWNJ_,
            0x25CC => GB,
            _ => category(c).unwrap_or(OTHER),
        };
        if category(c).is_none() && !matches!(cat, CGJ_ | ZWNJ_ | GB) && (is_mark_or_ignorable(c) || !is_simple_script(c)) {
            return Err(Unsupported("characters of another script among the engine's"));
        }
        let glyph = face.glyph_index(c);
        // a grapheme a cluster to start with: marks with their letter
        let mark = !matches!(cat, B | GB | N | O | R | CS | OTHER | ZWNJ_);
        let cluster = match buf.last() {
            Some(last) if mark => last.cluster,
            _ => offset as u32,
        };
        buf.push(Info { glyph, cluster, class: shaper.glyph_class(glyph), cat, fmask: F_GLOBAL_, ..Default::default() });
        cats.push(cat);
    }
    let Some(gsub) = gsub else { return Ok(buf) };
    let dotted = face.glyph_index(0x25CC);
    let mut out = Vec::with_capacity(buf.len());
    let stage = |syl: &mut Vec<Info>, tags: &[&[u8; 4]], bits: u32| -> Result<(), Unsupported> {
        for l in gsub.lookups_for(tags) {
            shaper.gsub_lookup_bits(&gsub, l, syl, bits)?;
        }
        Ok(())
    };
    for (a, b, kind) in clusters(&cats) {
        let mut syl: Vec<Info> = buf[a..b].to_vec();
        if kind == Kind::Broken && dotted != 0 {
            let at = usize::from(syl[0].cat == R);
            let cluster = syl[0].cluster;
            syl.insert(at, Info { glyph: dotted, cluster, class: shaper.glyph_class(dotted), cat: GB, fmask: F_GLOBAL_, ..Default::default() });
        }
        if kind != Kind::Other {
            stage(&mut syl, &[b"locl", b"ccmp", b"nukt", b"akhn"], F_GLOBAL_)?;
            // a reph: the first glyphs (just a repha character, if it starts)
            let limit = if syl[0].cat == R { 1 } else { syl.len().min(3) };
            for g in &mut syl[..limit] {
                g.fmask |= F_RPHF;
            }
            for g in &mut syl {
                g.substituted = false;
            }
            stage(&mut syl, &[b"rphf"], F_RPHF)?;
            if let Some(g) = syl.iter_mut().take_while(|g| g.fmask & F_RPHF != 0).find(|g| g.substituted) {
                g.cat = R;
            }
            for g in &mut syl {
                g.substituted = false;
            }
            stage(&mut syl, &[b"pref"], F_GLOBAL_)?;
            if let Some(g) = syl.iter_mut().find(|g| g.substituted) {
                g.cat = VPRE;
            }
            stage(&mut syl, &[b"rkrf", b"abvf", b"blwf", b"half", b"pstf", b"vatu", b"cjct"], F_GLOBAL_)?;
            reorder(&mut syl);
        } else {
            stage(&mut syl, &[b"locl", b"ccmp"], F_GLOBAL_)?;
        }
        out.extend(syl);
    }
    for l in gsub.lookups_for(&[b"abvs", b"blws", b"haln", b"pres", b"psts", b"rlig", b"calt", b"clig", b"liga", b"rclt"]) {
        shaper.gsub_lookup_bits(&gsub, l, &mut out, F_GLOBAL_)?;
    }
    Ok(out)
}

const F_GLOBAL_: u32 = 1;

fn is_halant(g: &Info) -> bool {
    matches!(g.cat, H | IS) && !g.ligated
}

/// HarfBuzz's `reorder_syllable_use`.
fn reorder(info: &mut [Info]) {
    let (start, end) = (0, info.len());
    let post_base = |c: u8| matches!(c, FABV | FBLW | FPST | FMABV | FMBLW | MABV | MBLW | MPST | MPRE | VABV | VBLW | VPST | VPRE | VMABV | VMBLW | VMPST | VMPRE);
    if end > 1 && info[start].cat == R {
        for i in start + 1..end {
            let post = post_base(info[i].cat) || is_halant(&info[i]);
            if post || i == end - 1 {
                let i = if post { i - 1 } else { i };
                merge(info, start, i + 1);
                info[start..=i].rotate_left(1);
                break;
            }
        }
    }
    let mut j = start;
    for i in start..end {
        if is_halant(&info[i]) {
            j = i + 1;
        } else if matches!(info[i].cat, VPRE | VMPRE) && !info[i].multiplied && j < i {
            merge(info, j, i + 1);
            info[j..=i].rotate_right(1);
        }
    }
}
