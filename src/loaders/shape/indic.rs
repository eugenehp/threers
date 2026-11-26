//! Indic shaping: Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil,
//! Telugu, Kannada, Malayalam and Sinhala, as OpenType's Indic spec has it.
//!
//! A word of these scripts isn't its letters side by side: a consonant
//! before a virama joins the next (स + ् + त is one conjunct, स्त), a Ra
//! before a virama goes up over the syllable as a reph (र्म), a Ra after
//! one goes under it (क्र), and some vowel signs are written before the
//! consonant they follow in the text (कि). The font has every form; this
//! says which to use, where:
//!
//! 1. The text is split into syllables (a consonant, its nukta, more
//!    consonants after viramas, then vowel signs and marks).
//! 2. Each syllable's base consonant is found (the last, except a Ra
//!    that takes its below-base form), and each character labelled by
//!    where it stands: a reph, before the base (half forms), the base,
//!    after it (below- and post-base forms), or its vowel signs.
//! 3. A vowel sign written before the base is moved to the syllable's
//!    front (one in two parts, Bengali's ো, split first).
//! 4. The features run in the spec's order, each over the part of the
//!    syllable it's for (`rphf` the reph, `half` before the base,
//!    `blwf`/`pstf` after it), then the presentation features over all.
//! 5. A reph formed moves after the base and its below-base forms.
//!
//! Joiners (ZWJ, ZWNJ) steer it and draw nothing.

use super::Shaped;
use super::gsub::{Buffer, Gsub};

/// A script shaped here: its new tag (version 2), given its Unicode one.
#[must_use]
pub fn v2_tag(script: &[u8; 4]) -> Option<[u8; 4]> {
    Some(match script {
        b"deva" | b"dev2" => *b"dev2",
        b"beng" | b"bng2" => *b"bng2",
        b"guru" | b"gur2" => *b"gur2",
        b"gujr" | b"gjr2" => *b"gjr2",
        b"orya" | b"ory2" => *b"ory2",
        b"taml" | b"tml2" => *b"tml2",
        b"telu" | b"tel2" => *b"tel2",
        b"knda" | b"knd2" => *b"knd2",
        b"mlym" | b"mlm2" => *b"mlm2",
        b"sinh" => *b"sinh",
        _ => return None,
    })
}

/// The old tag (version 1) for a version 2 one.
fn v1_tag(v2: &[u8; 4]) -> [u8; 4] {
    match v2 {
        b"dev2" => *b"deva",
        b"bng2" => *b"beng",
        b"gur2" => *b"guru",
        b"gjr2" => *b"gujr",
        b"ory2" => *b"orya",
        b"tml2" => *b"taml",
        b"tel2" => *b"telu",
        b"knd2" => *b"knda",
        b"mlm2" => *b"mlym",
        t => *t,
    }
}

/// What a character is to a syllable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    Consonant,
    /// An independent vowel (a syllable's start, as a consonant is).
    Vowel,
    Nukta,
    Halant,
    /// A dependent vowel sign.
    Matra,
    /// Candrabindu, anusvara, visarga: a syllable's end.
    Modifier,
    Zwj,
    Zwnj,
    Other,
}

fn class(c: char) -> Class {
    let v = c as u32;
    match v {
        0x200D => return Class::Zwj,
        0x200C => return Class::Zwnj,
        _ => {}
    }
    // Sinhala's block isn't laid out as the others
    if (0x0D80..=0x0DFF).contains(&v) {
        return match v {
            0x0D81..=0x0D83 => Class::Modifier,
            0x0D85..=0x0D96 => Class::Vowel,
            0x0D9A..=0x0DC6 => Class::Consonant,
            0x0DCA => Class::Halant,
            0x0DCF..=0x0DDF | 0x0DF2 | 0x0DF3 => Class::Matra,
            _ => Class::Other,
        };
    }
    if !(0x0900..=0x0D7F).contains(&v) {
        return Class::Other;
    }
    let block = v & !0x7F;
    let o = v - block;
    // (the few that break the blocks' common layout)
    match v {
        0x09CE | 0x09F0 | 0x09F1 | 0x0D7A..=0x0D7F => return Class::Consonant,
        0x0B71 => return Class::Consonant,
        0x0BCD | 0x0C4D | 0x0CCD | 0x0D4D | 0x0D3B | 0x0D3C => {
            return if matches!(v, 0x0D3B | 0x0D3C) { Class::Halant } else { Class::Halant };
        }
        _ => {}
    }
    match o {
        0x00..=0x03 => Class::Modifier,
        0x04..=0x14 | 0x60 | 0x61 | 0x72..=0x77 => Class::Vowel,
        0x15..=0x39 | 0x58..=0x5F | 0x78..=0x7F if block == 0x0900 => Class::Consonant,
        0x15..=0x39 | 0x5C..=0x5F => Class::Consonant,
        0x3C => Class::Nukta,
        0x3A | 0x3B | 0x3E..=0x4C | 0x4E | 0x4F | 0x55..=0x57 | 0x62 | 0x63 => Class::Matra,
        0x4D => Class::Halant,
        _ => Class::Other,
    }
}

/// Whether `c` is its script's Ra (a reph before a virama, below after one).
fn is_ra(c: char) -> bool {
    matches!(c as u32, 0x0930 | 0x09B0 | 0x09F0 | 0x0A30 | 0x0AB0 | 0x0B30 | 0x0BB0 | 0x0C30 | 0x0CB0 | 0x0D30 | 0x0DBB)
}

/// Whether a Ra after a virama takes a below-base form here (not a base of
/// its own): the north's scripts.
fn ra_below(c: char) -> bool {
    matches!(c as u32, 0x0930 | 0x09B0 | 0x09F0 | 0x0A30 | 0x0AB0 | 0x0B30)
}

/// A vowel sign written before its consonant.
fn pre_base(c: char) -> bool {
    matches!(
        c as u32,
        0x093F | 0x094E | 0x09BF | 0x09C7 | 0x09C8 | 0x0A3F | 0x0ABF | 0x0B47 | 0x0BC6 | 0x0BC7 | 0x0BC8 | 0x0D46 | 0x0D47
            | 0x0D48 | 0x0DD9 | 0x0DDA | 0x0DDB
    )
}

/// A vowel sign in two parts (one before, one after): its parts.
fn split(c: char) -> Option<&'static [char]> {
    Some(match c as u32 {
        0x09CB => &['\u{09C7}', '\u{09BE}'],
        0x09CC => &['\u{09C7}', '\u{09D7}'],
        0x0B48 => &['\u{0B47}', '\u{0B56}'],
        0x0B4B => &['\u{0B47}', '\u{0B3E}'],
        0x0B4C => &['\u{0B47}', '\u{0B57}'],
        0x0BCA => &['\u{0BC6}', '\u{0BBE}'],
        0x0BCB => &['\u{0BC7}', '\u{0BBE}'],
        0x0BCC => &['\u{0BC6}', '\u{0BD7}'],
        0x0D4A => &['\u{0D46}', '\u{0D3E}'],
        0x0D4B => &['\u{0D47}', '\u{0D3E}'],
        0x0D4C => &['\u{0D46}', '\u{0D57}'],
        0x0DDA => &['\u{0DD9}', '\u{0DCA}'],
        0x0DDC => &['\u{0DD9}', '\u{0DCF}'],
        0x0DDD => &['\u{0DD9}', '\u{0DCF}', '\u{0DCA}'],
        0x0DDE => &['\u{0DD9}', '\u{0DDF}'],
        _ => return None,
    })
}

/// Where a character stands in its syllable (which features it gets).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Place {
    /// A vowel sign moved before the base.
    Front,
    Reph,
    /// Before the base: half forms.
    Pre,
    Base,
    /// After the base: below- and post-base forms.
    Below,
    /// Vowel signs and marks after.
    Tail,
}

/// Syllables: `(start, end)` in `text`.
fn syllables(text: &[char]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < text.len() {
        let start = i;
        let first = class(text[i]);
        i += 1;
        if matches!(first, Class::Consonant | Class::Vowel) {
            let mut vowel_seen = first == Class::Vowel;
            while i < text.len() {
                match class(text[i]) {
                    Class::Nukta | Class::Zwj => i += 1,
                    Class::Halant => {
                        i += 1;
                        // a joiner after it, then (no vowel yet) the next consonant
                        if i < text.len() && matches!(class(text[i]), Class::Zwj | Class::Zwnj) {
                            i += 1;
                        }
                        if !vowel_seen && i < text.len() && class(text[i]) == Class::Consonant {
                            i += 1;
                        }
                    }
                    Class::Matra | Class::Modifier => {
                        vowel_seen = true;
                        i += 1;
                    }
                    _ => break,
                }
            }
        } else {
            // a vowel sign or mark with no letter: on its own
            while i < text.len() && matches!(class(text[i]), Class::Matra | Class::Modifier | Class::Nukta | Class::Halant) {
                i += 1;
            }
        }
        out.push((start, i));
    }
    out
}

/// One syllable's characters placed (and a pre-base vowel sign moved):
/// `(char, cluster, place)` in the order they're shaped.
fn place(syl: &[(char, usize)]) -> Vec<(char, usize, Place)> {
    let n = syl.len();
    let cls: Vec<Class> = syl.iter().map(|(c, _)| class(*c)).collect();
    if n == 0 || cls[0] != Class::Consonant {
        return syl.iter().map(|(c, k)| (*c, *k, Place::Base)).collect();
    }
    // consonants, and whether the syllable starts with a reph (Ra, virama,
    // more consonants after; not Ra, virama, ZWJ: that's an eyelash Ra)
    let consonants: Vec<usize> = (0..n).filter(|&i| cls[i] == Class::Consonant).collect();
    let reph = consonants.len() > 1 && is_ra(syl[0].0) && cls.get(1) == Some(&Class::Halant) && cls.get(2) != Some(&Class::Zwj);
    let from = if reph { 2 } else { 0 };
    let candidates: Vec<usize> = consonants.iter().copied().filter(|&i| i >= from).collect();
    let mut base = *candidates.last().unwrap_or(&0);
    // a last Ra after a virama: below the one before (क्र)
    if candidates.len() > 1 && ra_below(syl[base].0) && base >= 1 && cls[base - 1] == Class::Halant {
        base = candidates[candidates.len() - 2];
    }
    let mut out: Vec<(char, usize, Place)> = Vec::with_capacity(n);
    let mut front: Vec<(char, usize, Place)> = Vec::new();
    for i in 0..n {
        let (c, k) = syl[i];
        let p = if reph && i < 2 {
            Place::Reph
        } else if i < base {
            Place::Pre
        } else if i == base || (cls[i] == Class::Nukta && i == base + 1) {
            Place::Base
        } else if matches!(cls[i], Class::Matra | Class::Modifier) {
            if pre_base(c) {
                front.push((c, k, Place::Front));
                continue;
            }
            Place::Tail
        } else {
            Place::Below
        };
        out.push((c, k, p));
    }
    front.extend(out);
    front
}

/// The features run over one place only, in order (the spec's basic
/// shaping); `None`: over the whole syllable.
const BASIC: &[([u8; 4], Option<Place>)] = &[
    (*b"locl", None),
    (*b"ccmp", None),
    (*b"nukt", None),
    (*b"akhn", None),
    (*b"rphf", Some(Place::Reph)),
    (*b"rkrf", None),
    (*b"pref", Some(Place::Below)),
    (*b"blwf", Some(Place::Below)),
    (*b"abvf", Some(Place::Below)),
    (*b"half", Some(Place::Pre)),
    (*b"pstf", Some(Place::Below)),
    (*b"vatu", None),
    (*b"cjct", None),
];

/// Then over all of it: the presentation forms.
const PRESENTATION: &[[u8; 4]] = &[*b"init", *b"pres", *b"abvs", *b"blws", *b"psts", *b"haln", *b"calt", *b"clig", *b"liga"];

/// `feature` over the glyphs placed `at` (they're together), or all.
fn apply_at(gsub: &Gsub<'_>, script: &[u8; 4], feature: &[u8; 4], buf: &mut Buffer, places: &mut Vec<Place>, at: Option<Place>) {
    let (lo, hi) = match at {
        None => (0, buf.len()),
        Some(p) => {
            let Some(lo) = places.iter().position(|q| *q == p) else { return };
            let hi = places.iter().rposition(|q| *q == p).map_or(lo, |h| h + 1);
            (lo, hi)
        }
    };
    let pairs: Vec<(u16, usize)> = (lo..hi).map(|i| (buf.glyphs[i], buf.clusters[i])).collect();
    let mut part = Buffer::new(&pairs);
    gsub.apply_feature(script, feature, &mut part);
    if part.glyphs == buf.glyphs[lo..hi] {
        return;
    }
    // (each glyph out keeps the place of the glyph its cluster came from)
    let place_of: Vec<(usize, Place)> = (lo..hi).map(|i| (buf.clusters[i], places[i])).collect();
    let new_places: Vec<Place> = part
        .clusters
        .iter()
        .map(|c| place_of.iter().find(|(k, _)| k == c).map_or(at.unwrap_or(Place::Base), |(_, p)| *p))
        .collect();
    buf.glyphs.splice(lo..hi, part.glyphs);
    buf.clusters.splice(lo..hi, part.clusters);
    places.splice(lo..hi, new_places);
}

/// `text` (one script's run) shaped.
#[must_use]
pub fn shape(glyph_of: &impl Fn(char) -> Option<u16>, gsub: Option<&Gsub<'_>>, script: &[u8; 4], text: &[char]) -> Vec<Shaped> {
    // two-part vowel signs split (each part keeps the sign's place in the text)
    let mut chars: Vec<(char, usize)> = Vec::with_capacity(text.len());
    for (i, c) in text.iter().enumerate() {
        match split(*c) {
            Some(parts) => chars.extend(parts.iter().map(|p| (*p, i))),
            None => chars.push((*c, i)),
        }
    }
    let plain: Vec<char> = chars.iter().map(|(c, _)| *c).collect();
    let v2 = v2_tag(script).unwrap_or(*script);
    let tag = match gsub {
        Some(g) if !g.lists_script(&v2) && g.lists_script(&v1_tag(&v2)) => v1_tag(&v2),
        _ => v2,
    };
    let mut out = Vec::with_capacity(chars.len());
    for (a, b) in syllables(&plain) {
        let placed = place(&chars[a..b]);
        // joiners steer, and draw nothing
        let mut pairs = Vec::new();
        let mut places = Vec::new();
        for (c, k, p) in &placed {
            if matches!(class(*c), Class::Zwj | Class::Zwnj) {
                continue;
            }
            if let Some(g) = glyph_of(*c) {
                pairs.push((g, *k));
                places.push(*p);
            }
        }
        let mut buf = Buffer::new(&pairs);
        if let Some(gsub) = gsub {
            let reph_pair = places.iter().filter(|p| **p == Place::Reph).count();
            for (feature, at) in BASIC {
                apply_at(gsub, &tag, feature, &mut buf, &mut places, *at);
            }
            // a reph that didn't form (the font has none): a half form, where it is
            let reph_now = places.iter().filter(|p| **p == Place::Reph).count();
            if reph_pair == 2 && reph_now == 2 {
                for p in places.iter_mut().filter(|p| **p == Place::Reph) {
                    *p = Place::Pre;
                }
                apply_at(gsub, &tag, b"half", &mut buf, &mut places, Some(Place::Pre));
            } else if let Some(r) = places.iter().position(|p| *p == Place::Reph) {
                // a reph formed: after the base and what's below it, before the vowel signs
                let to = places.iter().rposition(|p| matches!(p, Place::Base | Place::Below)).map_or(buf.len() - 1, |i| i);
                if to > r {
                    let (g, k, p) = (buf.glyphs.remove(r), buf.clusters.remove(r), places.remove(r));
                    buf.glyphs.insert(to, g);
                    buf.clusters.insert(to, k);
                    places.insert(to, p);
                }
            }
            for feature in PRESENTATION {
                apply_at(gsub, &tag, feature, &mut buf, &mut places, None);
            }
        }
        out.extend(buf.glyphs.iter().zip(&buf.clusters).map(|(g, c)| Shaped { glyph: *g, cluster: *c }));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chars(s: &str) -> Vec<char> {
        s.chars().collect()
    }

    #[test]
    fn syllables_are_found() {
        // न | म | स्ते
        assert_eq!(syllables(&chars("नमस्ते")), vec![(0, 1), (1, 2), (2, 6)]);
        // हि | न्दी
        assert_eq!(syllables(&chars("हिन्दी")), vec![(0, 2), (2, 6)]);
    }

    #[test]
    fn a_pre_base_vowel_sign_goes_first() {
        let s: Vec<(char, usize)> = chars("कि").into_iter().zip(0..).collect();
        let p: Vec<char> = place(&s).iter().map(|x| x.0).collect();
        assert_eq!(p, chars("िक"));
    }

    #[test]
    fn ra_before_a_virama_is_a_reph_after_one_below() {
        let s: Vec<(char, usize)> = chars("र्म").into_iter().zip(0..).collect();
        let places: Vec<Place> = place(&s).iter().map(|x| x.2).collect();
        assert_eq!(places, vec![Place::Reph, Place::Reph, Place::Base]);
        let s: Vec<(char, usize)> = chars("क्र").into_iter().zip(0..).collect();
        let places: Vec<Place> = place(&s).iter().map(|x| x.2).collect();
        assert_eq!(places, vec![Place::Base, Place::Below, Place::Below]);
        let s: Vec<(char, usize)> = chars("स्त").into_iter().zip(0..).collect();
        let places: Vec<Place> = place(&s).iter().map(|x| x.2).collect();
        assert_eq!(places, vec![Place::Pre, Place::Pre, Place::Base]);
    }

    #[test]
    fn a_two_part_vowel_sign_is_split() {
        // Bengali কো: ে before ক, া after
        let shaped = shape(&|c| Some(c as u16), None, b"beng", &chars("কো"));
        let got: Vec<u16> = shaped.iter().map(|s| s.glyph).collect();
        assert_eq!(got, vec![0x09C7, 0x0995, 0x09BE]);
    }
}
