//! The Indic scripts (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya,
//! Tamil, Telugu, Kannada, Malayalam, Sinhala), Khmer and Myanmar, shaped
//! as HarfBuzz's Indic, Khmer and Myanmar shapers shape them: two-part vowel signs
//! split, the text cut into syllables, each syllable's base consonant
//! found and its characters put in the order the font's features expect
//! (a pre-base vowel sign first, a reph's Ra kept for `rphf`), the basic
//! features applied one at a time on the glyphs each is for (`half` on
//! the consonants before the base, `blwf`, `abvf`, `pstf` after it), the
//! reph and pre-base vowel signs then moved to where they are drawn, and
//! the presentation features applied last. Clusters are graphemes (a
//! vowel sign, a virama, a nukta in its letter's), merged where glyphs
//! are reordered or joined, as HarfBuzz merges them.

use std::collections::HashMap;

use super::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Indic {
    Deva,
    Beng,
    Guru,
    Gujr,
    Orya,
    Taml,
    Telu,
    Knda,
    Mlym,
    Sinh,
    Khmr,
    Mymr,
}

// categories
const X: u8 = 0;
const C: u8 = 1;
const V: u8 = 2;
const N: u8 = 3;
const H: u8 = 4;
const ZWNJ: u8 = 5;
const ZWJ: u8 = 6;
const M: u8 = 7;
const SM: u8 = 8;
const VD: u8 = 9;
const PLACEHOLDER: u8 = 10;
const DOTTED: u8 = 11;
const RA: u8 = 12;
const REPHA: u8 = 13;
/// Myanmar's asat (a dead consonant's sign) and its medial Ra, drawn before the base.
const ASAT: u8 = 14;
const MEDIAL_RA: u8 = 15;
/// Myanmar's vowel signs below (not its medials below).
const VBLW: u8 = 16;

// places in a syllable, in the order they are sorted into
const POS_START: u8 = 0;
const POS_REPH: u8 = 1;
const POS_PRE_M: u8 = 2;
const POS_PRE_C: u8 = 3;
const POS_BASE: u8 = 4;
const POS_AFTER_MAIN: u8 = 5;
const POS_BEFORE_SUB: u8 = 7;
const POS_BELOW_C: u8 = 8;
const POS_AFTER_SUB: u8 = 9;
const POS_POST_C: u8 = 11;
const POS_AFTER_POST: u8 = 12;
const POS_SMVD: u8 = 13;
const POS_END: u8 = 14;

// feature bits
const F_GLOBAL: u32 = 1;
const F_RPHF: u32 = 2;
const F_HALF: u32 = 4;
const F_BLWF: u32 = 8;
const F_ABVF: u32 = 16;
const F_PSTF: u32 = 32;
const F_PREF: u32 = 64;
const F_CFAR: u32 = 128;
const F_INIT: u32 = 256;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Matra {
    Left,
    Right,
    Top,
    Bottom,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RephPos {
    AfterMain,
    BeforeSub,
    AfterSub,
    BeforePost,
    AfterPost,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RephMode {
    Implicit,
    Explicit,
    Logical,
    None,
}

impl Indic {
    pub(super) fn of(c: u32) -> Option<Indic> {
        Some(match c {
            0x900..=0x97F | 0xA8E0..=0xA8FF | 0x1CD0..=0x1CFF => Indic::Deva,
            0x980..=0x9FF => Indic::Beng,
            0xA00..=0xA7F => Indic::Guru,
            0xA80..=0xAFF => Indic::Gujr,
            0xB00..=0xB7F => Indic::Orya,
            0xB80..=0xBFF => Indic::Taml,
            0xC00..=0xC7F => Indic::Telu,
            0xC80..=0xCFF => Indic::Knda,
            0xD00..=0xD7F => Indic::Mlym,
            0xD80..=0xDFF => Indic::Sinh,
            0x1780..=0x17FF | 0x19E0..=0x19FF => Indic::Khmr,
            0x1000..=0x109F | 0xA9E0..=0xA9FF | 0xAA60..=0xAA7F => Indic::Mymr,
            _ => return None,
        })
    }

    /// The script's OpenType tags: the new one, the old.
    pub(super) fn tags(self) -> ([u8; 4], [u8; 4]) {
        match self {
            Indic::Deva => (*b"dev2", *b"deva"),
            Indic::Beng => (*b"bng2", *b"beng"),
            Indic::Guru => (*b"gur2", *b"guru"),
            Indic::Gujr => (*b"gjr2", *b"gujr"),
            Indic::Orya => (*b"ory2", *b"orya"),
            Indic::Taml => (*b"tml2", *b"taml"),
            Indic::Telu => (*b"tel2", *b"telu"),
            Indic::Knda => (*b"knd2", *b"knda"),
            Indic::Mlym => (*b"mlm2", *b"mlym"),
            Indic::Sinh => (*b"sinh", *b"sinh"),
            Indic::Khmr => (*b"khmr", *b"khmr"),
            Indic::Mymr => (*b"mym2", *b"mymr"),
        }
    }

    fn block(self) -> u32 {
        match self {
            Indic::Deva => 0x900,
            Indic::Beng => 0x980,
            Indic::Guru => 0xA00,
            Indic::Gujr => 0xA80,
            Indic::Orya => 0xB00,
            Indic::Taml => 0xB80,
            Indic::Telu => 0xC00,
            Indic::Knda => 0xC80,
            Indic::Mlym => 0xD00,
            Indic::Sinh => 0xD80,
            Indic::Khmr => 0x1780,
            Indic::Mymr => 0x1000,
        }
    }

    fn virama(self) -> u32 {
        match self {
            Indic::Sinh => 0xDCA,
            Indic::Khmr => 0x17D2,
            Indic::Mymr => 0x1039,
            s => s.block() + 0x4D,
        }
    }

    fn reph(self) -> (RephPos, RephMode) {
        match self {
            Indic::Deva | Indic::Gujr => (RephPos::BeforePost, RephMode::Implicit),
            Indic::Beng => (RephPos::AfterSub, RephMode::Implicit),
            Indic::Guru => (RephPos::BeforeSub, RephMode::Implicit),
            Indic::Orya => (RephPos::AfterMain, RephMode::Implicit),
            Indic::Taml | Indic::Knda => (RephPos::AfterPost, RephMode::Implicit),
            Indic::Telu => (RephPos::AfterPost, RephMode::Explicit),
            Indic::Mlym => (RephPos::AfterMain, RephMode::Logical),
            Indic::Sinh => (RephPos::AfterMain, RephMode::Explicit),
            Indic::Khmr | Indic::Mymr => (RephPos::AfterMain, RephMode::None),
        }
    }

    /// Whether consonants before the base may take below-base forms too.
    fn blwf_before_base(self) -> bool {
        !matches!(self, Indic::Telu | Indic::Knda)
    }

    /// Where a vowel sign of each kind goes (HarfBuzz's `matra_position_indic`).
    fn matra_pos(self, kind: Matra, c: u32) -> u8 {
        use Indic::*;
        match kind {
            Matra::Left => POS_PRE_M,
            Matra::Right => match self {
                Deva | Sinh => POS_AFTER_SUB,
                Telu => if c <= 0xC42 { POS_BEFORE_SUB } else { POS_AFTER_SUB },
                Knda => if !(0xCC3..=0xCD6).contains(&c) { POS_BEFORE_SUB } else { POS_AFTER_SUB },
                Khmr => POS_AFTER_POST,
                _ => POS_AFTER_POST,
            },
            Matra::Top => match self {
                Guru => POS_AFTER_POST,
                Orya => POS_AFTER_MAIN,
                Telu | Knda => POS_BEFORE_SUB,
                _ => POS_AFTER_SUB,
            },
            Matra::Bottom => match self {
                Guru | Gujr | Taml | Mlym => POS_AFTER_POST,
                Telu | Knda => POS_BEFORE_SUB,
                _ => POS_AFTER_SUB,
            },
        }
    }

    /// A character's category, and for a vowel sign its kind.
    fn classify(self, c: u32) -> (u8, Option<Matra>) {
        use Matra::*;
        match c {
            0x200C => return (ZWNJ, None),
            0x200D => return (ZWJ, None),
            0x25CC => return (DOTTED, None),
            0xA0 | 0x2010..=0x2014 | 0xD7 => return (PLACEHOLDER, None),
            0x1CD0..=0x1CFF | 0xA8E0..=0xA8F1 => return (VD, None),
            _ => {}
        }
        if self == Indic::Khmr {
            return match c {
                0x179A => (RA, None),
                0x1780..=0x17A2 => (C, None),
                0x17A3..=0x17B3 => (V, None),
                0x17B6 => (M, Some(Right)),
                0x17B7..=0x17BA | 0x17BE..=0x17C0 => (M, Some(Top)),
                0x17BB..=0x17BD => (M, Some(Bottom)),
                0x17C1..=0x17C3 => (M, Some(Left)),
                0x17C4..=0x17C5 => (M, Some(Right)),
                0x17C6..=0x17D1 | 0x17D3 | 0x17DD => (SM, None),
                0x17D2 => (H, None),
                _ => (X, None),
            };
        }
        if self == Indic::Mymr {
            return match c {
                // the asat and the medials, by their places
                0x103A => (ASAT, None),
                0x103B | 0x105E..=0x105F => (M, Some(Right)),
                0x103C => (MEDIAL_RA, None),
                0x103D..=0x103E | 0x1060 | 0x1082 => (M, Some(Bottom)),
                0x1004 | 0x101B | 0x105A => (RA, None),
                0x1000..=0x1020 | 0x103F | 0x1050..=0x1051 | 0x105B..=0x105D | 0x1061 | 0x1065..=0x1066 | 0x106E..=0x1070 | 0x1075..=0x1081 | 0x108E => (C, None),
                0x1021..=0x102A | 0x1052..=0x1055 => (V, None),
                0x1031 | 0x1084 => (M, Some(Left)),
                0x102B..=0x102C | 0x1056..=0x1057 | 0x1062 | 0x1067..=0x1068 | 0x1083 => (M, Some(Right)),
                0x102D..=0x102E | 0x1032..=0x1035 | 0x1071..=0x1074 | 0x1085..=0x1086 => (M, Some(Top)),
                0x102F..=0x1030 | 0x1058..=0x1059 => (VBLW, None),
                0x1036..=0x1038 | 0x1063..=0x1064 | 0x1069..=0x106D | 0x1087..=0x108D | 0x108F | 0x109A..=0x109D => (SM, None),
                0x1039 => (H, None),
                _ => (X, None),
            };
        }
        if self == Indic::Sinh {
            return match c {
                0xDBB => (RA, None),
                0xD81..=0xD83 => (SM, None),
                0xD85..=0xD96 => (V, None),
                0xD9A..=0xDC6 => (C, None),
                0xDCA => (H, None),
                0xDCF..=0xDD1 | 0xDD8 | 0xDDF | 0xDF2..=0xDF3 => (M, Some(Right)),
                0xDD2..=0xDD3 => (M, Some(Top)),
                0xDD4 | 0xDD6 => (M, Some(Bottom)),
                0xDD9..=0xDDB => (M, Some(Left)),
                0xDDC..=0xDDE => (M, Some(Right)),
                _ => (X, None),
            };
        }
        let o = c.wrapping_sub(self.block());
        if o >= 0x80 {
            // the danda and double danda, shared
            return (X, None);
        }
        // what differs from script to script
        let special = match (self, c) {
            (Indic::Deva, 0x93A) => Some((M, Some(Top))),
            (Indic::Deva, 0x93B | 0x94F) => Some((M, Some(Right))),
            (Indic::Deva, 0x94E) => Some((M, Some(Left))),
            (Indic::Deva, 0x955) => Some((M, Some(Top))),
            (Indic::Deva, 0x956..=0x957) => Some((M, Some(Bottom))),
            (Indic::Deva, 0x951..=0x954) => Some((VD, None)),
            (Indic::Deva, 0x972..=0x977) => Some((V, None)),
            (Indic::Deva, 0x978..=0x97F) => Some((C, None)),
            (Indic::Beng, 0x9C7..=0x9C8) => Some((M, Some(Left))),
            (Indic::Beng, 0x9CE) => Some((C, None)),
            (Indic::Beng, 0x9D7) => Some((M, Some(Right))),
            (Indic::Beng, 0x9F0) => Some((RA, None)),
            (Indic::Beng, 0x9F1) => Some((C, None)),
            (Indic::Guru, 0xA47..=0xA48 | 0xA4B..=0xA4C) => Some((M, Some(Top))),
            (Indic::Guru, 0xA70..=0xA71) => Some((SM, None)),
            (Indic::Guru, 0xA72..=0xA73) => Some((C, None)),
            (Indic::Guru, 0xA75) => Some((M, Some(Bottom))),
            (Indic::Gujr, 0xAC5 | 0xAC7..=0xAC8) => Some((M, Some(Top))),
            (Indic::Gujr, 0xAC9 | 0xACB..=0xACC) => Some((M, Some(Right))),
            (Indic::Gujr, 0xAF9) => Some((C, None)),
            (Indic::Orya, 0xB3F | 0xB55..=0xB56) => Some((M, Some(Top))),
            (Indic::Orya, 0xB47) => Some((M, Some(Left))),
            (Indic::Orya, 0xB57) => Some((M, Some(Right))),
            (Indic::Orya, 0xB71) => Some((C, None)),
            (Indic::Taml, 0xB83) => Some((SM, None)),
            (Indic::Taml, 0xBBE..=0xBBF | 0xBC1..=0xBC2 | 0xBD7) => Some((M, Some(Right))),
            (Indic::Taml, 0xBC0) => Some((M, Some(Top))),
            (Indic::Taml, 0xBC6..=0xBC8) => Some((M, Some(Left))),
            (Indic::Telu, 0xC00 | 0xC04) => Some((SM, None)),
            (Indic::Telu, 0xC3E..=0xC40 | 0xC46..=0xC47 | 0xC4A..=0xC4C | 0xC55) => Some((M, Some(Top))),
            (Indic::Telu, 0xC41..=0xC44) => Some((M, Some(Right))),
            (Indic::Telu, 0xC56) => Some((M, Some(Bottom))),
            (Indic::Telu, 0xC58..=0xC5A) => Some((C, None)),
            (Indic::Knda, 0xCBF | 0xCC6 | 0xCCC) => Some((M, Some(Top))),
            (Indic::Knda, 0xCC1..=0xCC4 | 0xCD5..=0xCD6) => Some((M, Some(Right))),
            (Indic::Knda, 0xCDE | 0xCF1..=0xCF2) => Some((C, None)),
            (Indic::Mlym, 0xD3B..=0xD3C) => Some((H, None)),
            (Indic::Mlym, 0xD3E..=0xD40 | 0xD57) => Some((M, Some(Right))),
            (Indic::Mlym, 0xD46..=0xD48) => Some((M, Some(Left))),
            (Indic::Mlym, 0xD4E) => Some((REPHA, None)),
            (Indic::Mlym, 0xD54..=0xD56 | 0xD7A..=0xD7F) => Some((C, None)),
            (Indic::Mlym, 0xD5F..=0xD61) => Some((V, None)),
            _ => None,
        };
        if let Some(s) = special {
            return s;
        }
        match o {
            0x00..=0x02 => (SM, None),
            0x03 => (SM, None),
            0x04..=0x14 => (V, None),
            0x30 => (RA, None),
            0x15..=0x39 => (C, None),
            0x3C => (N, None),
            0x3E | 0x40 => (M, Some(Right)),
            0x3F => (M, Some(Left)),
            0x41..=0x44 => (M, Some(Bottom)),
            0x45..=0x48 => (M, Some(Top)),
            0x49..=0x4C => (M, Some(Right)),
            0x4D => (H, None),
            0x58..=0x5F => (C, None),
            0x60..=0x61 => (V, None),
            0x62..=0x63 => (M, Some(Bottom)),
            _ => (X, None),
        }
    }
}

/// A two-part vowel sign as its parts (and Khmer's, the part before the base added).
fn split_matra(c: u32) -> Option<&'static [u32]> {
    Some(match c {
        0x9CB => &[0x9C7, 0x9BE],
        0x9CC => &[0x9C7, 0x9D7],
        0xB48 => &[0xB47, 0xB56],
        0xB4B => &[0xB47, 0xB3E],
        0xB4C => &[0xB47, 0xB57],
        0xBCA => &[0xBC6, 0xBBE],
        0xBCB => &[0xBC7, 0xBBE],
        0xBCC => &[0xBC6, 0xBD7],
        0xC48 => &[0xC46, 0xC56],
        0xCC0 => &[0xCBF, 0xCD5],
        0xCC7 => &[0xCC6, 0xCD5],
        0xCC8 => &[0xCC6, 0xCD6],
        0xCCA => &[0xCC6, 0xCC2],
        0xCCB => &[0xCC6, 0xCC2, 0xCD5],
        0xD4A => &[0xD46, 0xD3E],
        0xD4B => &[0xD47, 0xD3E],
        0xD4C => &[0xD46, 0xD57],
        0xDDA => &[0xDD9, 0xDCA],
        0xDDC => &[0xDD9, 0xDCF],
        0xDDD => &[0xDD9, 0xDCF, 0xDCA],
        0xDDE => &[0xDD9, 0xDDF],
        0x17BE => &[0x17C1, 0x17BE],
        0x17BF => &[0x17C1, 0x17BF],
        0x17C0 => &[0x17C1, 0x17C0],
        0x17C4 => &[0x17C1, 0x17C4],
        0x17C5 => &[0x17C1, 0x17C5],
        _ => return None,
    })
}

/// A letter with a nukta that Unicode keeps decomposed (its composition
/// excluded): as its letter and nukta, which the font's `nukt` joins.
fn nukta_letter(c: u32) -> Option<&'static [u32]> {
    Some(match c {
        0x958 => &[0x915, 0x93C],
        0x959 => &[0x916, 0x93C],
        0x95A => &[0x917, 0x93C],
        0x95B => &[0x91C, 0x93C],
        0x95C => &[0x921, 0x93C],
        0x95D => &[0x922, 0x93C],
        0x95E => &[0x92B, 0x93C],
        0x95F => &[0x92F, 0x93C],
        0x9DC => &[0x9A1, 0x9BC],
        0x9DD => &[0x9A2, 0x9BC],
        0x9DF => &[0x9AF, 0x9BC],
        0xA33 => &[0xA32, 0xA3C],
        0xA36 => &[0xA38, 0xA3C],
        0xA59 => &[0xA16, 0xA3C],
        0xA5A => &[0xA17, 0xA3C],
        0xA5B => &[0xA1C, 0xA3C],
        0xA5E => &[0xA2B, 0xA3C],
        0xB5C => &[0xB21, 0xB3C],
        0xB5D => &[0xB22, 0xB3C],
        _ => return None,
    })
}

/// A vowel sign, virama, nukta or modifier of these scripts (for clusters
/// where Apple's tables shape them).
pub(super) fn is_mark_char(c: u32) -> bool {
    Indic::of(c).is_some_and(|s| matches!(s.classify(c).0, M | N | H | SM | VD | ASAT | MEDIAL_RA | VBLW))
}

/// Whether `c` may stand in text shaped as `script`.
pub(super) fn belongs(script: Indic, c: u32) -> bool {
    Indic::of(c) == Some(script) || matches!(c, 0x964 | 0x965 | 0x200C | 0x200D | 0x25CC | 0xA0) || is_simple_script(c) && !is_mark_or_ignorable(c)
}

fn is_consonant(cat: u8) -> bool {
    matches!(cat, C | RA | V | PLACEHOLDER | DOTTED | REPHA)
}

fn is_joiner(cat: u8) -> bool {
    matches!(cat, ZWJ | ZWNJ)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Around a consonant (or an independent vowel).
    Cluster,
    /// Marks with nothing to sit on: on a dotted circle.
    Broken,
    Other,
}

/// The syllables: from, to, kind.
fn syllables(script: Indic, cats: &[u8]) -> Vec<(usize, usize, Kind)> {
    let n = cats.len();
    let mut out = Vec::new();
    let mut i = 0;
    let is_mark = |c: u8| matches!(c, M | N | H | SM | VD | ZWJ | ZWNJ | ASAT | MEDIAL_RA | VBLW);
    while i < n {
        let c = cats[i];
        if is_consonant(c) {
            let mut j = i + 1;
            if script == Indic::Mymr && c == RA && j + 1 < n && cats[j] == ASAT && cats[j + 1] == H {
                // kinzi: NGA, asat, virama before the consonant it goes over
                j += 2;
                if j < n && matches!(cats[j], C | RA | V) {
                    j += 1;
                }
            }
            if c == REPHA {
                while j < n && is_joiner(cats[j]) {
                    j += 1;
                }
                if j < n && matches!(cats[j], C | RA) {
                    j += 1;
                }
            }
            loop {
                while j < n && cats[j] == N {
                    j += 1;
                }
                if j < n && cats[j] == H {
                    let mut k = j + 1;
                    if k < n && is_joiner(cats[k]) {
                        k += 1;
                    }
                    if k < n && (matches!(cats[k], C | RA) || matches!(script, Indic::Khmr | Indic::Mymr) && cats[k] == V) {
                        j = k + 1;
                        continue;
                    }
                    j = k;
                    break;
                }
                if j + 2 < n && is_joiner(cats[j]) && cats[j + 1] == H && matches!(cats[j + 2], C | RA) {
                    j += 3;
                    continue;
                }
                break;
            }
            while j < n && matches!(cats[j], M | N | H | ZWJ | ZWNJ | ASAT | MEDIAL_RA | VBLW) {
                j += 1;
            }
            while j < n && matches!(cats[j], SM | VD | N) {
                j += 1;
            }
            out.push((i, j, Kind::Cluster));
            i = j;
        } else if is_mark(c) {
            let mut j = i + 1;
            while j < n && is_mark(cats[j]) {
                j += 1;
            }
            out.push((i, j, Kind::Broken));
            i = j;
        } else {
            out.push((i, i + 1, Kind::Other));
            i += 1;
        }
    }
    out
}

/// What the basic features of a script are, in order, and the bits that
/// say which glyphs each may change.
const BASIC: [(&[u8; 4], u32); 11] = [
    (b"nukt", F_GLOBAL),
    (b"akhn", F_GLOBAL),
    (b"rphf", F_RPHF),
    (b"rkrf", F_GLOBAL),
    (b"pref", F_PREF),
    (b"blwf", F_BLWF),
    (b"abvf", F_ABVF),
    (b"half", F_HALF),
    (b"pstf", F_PSTF),
    (b"vatu", F_GLOBAL),
    (b"cjct", F_GLOBAL),
];

const KHMER_BASIC: [(&[u8; 4], u32); 5] = [(b"pref", F_PREF), (b"blwf", F_BLWF), (b"abvf", F_ABVF), (b"pstf", F_PSTF), (b"cfar", F_CFAR)];

const PRESENTATION: [&[u8; 4]; 10] = [b"pres", b"abvs", b"blws", b"psts", b"haln", b"rlig", b"calt", b"clig", b"liga", b"rclt"];
const KHMER_PRESENTATION: [&[u8; 4]; 9] = [b"pres", b"abvs", b"blws", b"psts", b"rlig", b"calt", b"clig", b"liga", b"rclt"];

/// Characters to shaped glyphs (GSUB done) in logical order, a syllable a
/// cluster.
pub(super) fn substitute(shaper: &Shaper, face: &Face, script: Indic, text: &str) -> Result<Vec<Info>, Unsupported> {
    let (v2, v1) = script.tags();
    let gsub = Layout::of(face, b"GSUB").map(|l| {
        let tag = if l.has_script(&v2) { v2 } else { v1 };
        l.with_script(tag)
    });

    // characters, two-part vowel signs split, each with its category
    let mut buf: Vec<Info> = Vec::with_capacity(text.len());
    let mut matras: Vec<Option<Matra>> = Vec::with_capacity(text.len());
    let mut codes: Vec<u32> = Vec::with_capacity(text.len());
    for (offset, ch) in text.char_indices() {
        let c = ch as u32;
        if !belongs(script, c) {
            return Err(Unsupported("characters of another script among Indic text"));
        }
        let parts: Vec<u32> = match split_matra(c).or_else(|| nukta_letter(c)) {
            Some(p) => p.to_vec(),
            None => vec![c],
        };
        for p in parts {
            let (cat, matra) = script.classify(p);
            let glyph = face.glyph_index(p);
            // a mark (or a ZWJ) goes on in the grapheme before it
            let cluster = match buf.last() {
                Some(last) if matches!(cat, M | N | H | SM | VD | ZWJ | ASAT | MEDIAL_RA | VBLW) => last.cluster,
                _ => offset as u32,
            };
            buf.push(Info { glyph, cluster, class: shaper.glyph_class(glyph), cat, fmask: F_GLOBAL, ..Default::default() });
            matras.push(matra);
            codes.push(p);
        }
    }

    let Some(gsub) = gsub else { return Ok(buf) };
    let cats: Vec<u8> = buf.iter().map(|b| b.cat).collect();
    let virama = face.glyph_index(script.virama());
    let dotted = face.glyph_index(0x25CC);
    // a consonant's place by the forms the font has for it after a virama
    let mut forms: HashMap<u16, u8> = HashMap::new();
    let mut place = |g: u16| -> u8 {
        *forms.entry(g).or_insert_with(|| {
            let pairs = [[virama, g], [g, virama]];
            let would = |f: &[u8; 4]| virama != 0 && pairs.iter().any(|p| shaper.would_substitute(&gsub, &[f], p));
            if would(b"blwf") {
                POS_BELOW_C
            } else if would(b"pstf") || would(b"pref") {
                POS_POST_C
            } else {
                POS_BASE
            }
        })
    };
    for (k, b) in buf.iter_mut().enumerate() {
        b.pos = match b.cat {
            C | RA => place(b.glyph),
            V | PLACEHOLDER | DOTTED | REPHA => POS_BASE,
            M => script.matra_pos(matras[k].unwrap_or(Matra::Right), codes[k]),
            ASAT | MEDIAL_RA | VBLW => POS_END,
            SM | VD => POS_SMVD,
            _ => POS_END,
        };
    }

    let mut out = Vec::with_capacity(buf.len());
    for (a, b, kind) in syllables(script, &cats) {
        let mut syl: Vec<Info> = buf[a..b].to_vec();
        let cluster = syl.iter().map(|g| g.cluster).min().unwrap_or(0);
        if kind == Kind::Broken && dotted != 0 {
            // marks with nothing under them, on a dotted circle
            let at = usize::from(syl[0].cat == REPHA);
            syl.insert(at, Info { glyph: dotted, cluster, class: shaper.glyph_class(dotted), cat: DOTTED, pos: POS_BASE, fmask: F_GLOBAL, ..Default::default() });
        }
        for f in [&b"rvrn", &b"locl", &b"ccmp"] {
            for l in gsub.lookups_for(&[f]) {
                shaper.gsub_lookup_bits(&gsub, l, &mut syl, F_GLOBAL)?;
            }
        }
        if kind != Kind::Other {
            if script == Indic::Mymr {
                myanmar_syllable(shaper, &gsub, &mut syl)?;
            } else if script == Indic::Khmr {
                khmer_syllable(shaper, &gsub, &mut syl)?;
            } else {
                indic_syllable(shaper, &gsub, script, &mut syl)?;
            }
        }
        // `init`: a vowel sign before the base at a word's start
        let word_start = a == 0 || !matches!(cats[a - 1], C | RA | V | N | H | M | SM | VD | ZWJ | ZWNJ | PLACEHOLDER | DOTTED | REPHA);
        if !matches!(script, Indic::Khmr | Indic::Mymr) && word_start && syl.first().is_some_and(|g| g.pos == POS_PRE_M) {
            syl[0].fmask |= F_INIT;
            for l in gsub.lookups_for(&[b"init"]) {
                shaper.gsub_lookup_bits(&gsub, l, &mut syl, F_INIT)?;
            }
        }
        let presentation: &[&[u8; 4]] = if matches!(script, Indic::Khmr | Indic::Mymr) { &KHMER_PRESENTATION } else { &PRESENTATION };
        for l in gsub.lookups_for(presentation) {
            shaper.gsub_lookup_bits(&gsub, l, &mut syl, F_GLOBAL)?;
        }
        // joiners left: a space with no advance, as HarfBuzz hides them
        let space = face.glyph_index(0x20);
        for g in &mut syl {
            if is_joiner(g.cat) && !g.ligated {
                g.glyph = space;
                g.class = shaper.glyph_class(space);
                g.hidden = true;
            }
        }
        out.extend(syl);
    }
    Ok(out)
}

/// A syllable's initial reordering, its basic features and its final
/// reordering (HarfBuzz's `initial_reordering_consonant_syllable` and
/// `final_reordering_syllable_indic`).
fn indic_syllable(shaper: &Shaper, gsub: &Layout, script: Indic, info: &mut Vec<Info>) -> Result<(), Unsupported> {
    let (start, end) = (0, info.len());
    let (reph_pos, reph_mode) = script.reph();

    // 1. the base consonant, and whether a Ra starting it becomes a reph
    let mut base = end;
    let mut has_reph = false;
    let mut limit = start;
    let has_rphf = !gsub.lookups_for(&[b"rphf"]).is_empty();
    if has_rphf
        && start + 3 <= end
        && info[start].cat == RA
        && info[start + 1].cat == H
        && (reph_mode == RephMode::Implicit && !is_joiner(info[start + 2].cat) || reph_mode == RephMode::Explicit && info[start + 2].cat == ZWJ)
    {
        let glyphs = [info[start].glyph, info[start + 1].glyph];
        if shaper.would_substitute(gsub, &[b"rphf"], &glyphs) || reph_mode == RephMode::Explicit && shaper.would_substitute(gsub, &[b"rphf"], &[glyphs[0], glyphs[1], info[start + 2].glyph]) {
            limit += 2;
            while limit < end && is_joiner(info[limit].cat) {
                limit += 1;
            }
            base = start;
            has_reph = true;
        }
    } else if reph_mode == RephMode::Logical && info[start].cat == REPHA {
        limit += 1;
        while limit < end && is_joiner(info[limit].cat) {
            limit += 1;
        }
        base = start;
        has_reph = true;
    }
    {
        // from the end back: the last consonant without a below- or post-base form
        let mut i = end;
        let mut seen_below = false;
        loop {
            if i <= limit {
                break;
            }
            i -= 1;
            if is_consonant(info[i].cat) {
                if info[i].pos != POS_BELOW_C && (info[i].pos != POS_POST_C || seen_below) {
                    base = i;
                    break;
                }
                if info[i].pos == POS_BELOW_C {
                    seen_below = true;
                }
                base = i;
            } else if start < i && info[i].cat == ZWJ && info[i - 1].cat == H {
                break;
            }
        }
    }
    if has_reph && base == start && limit - base <= 2 {
        // no other consonant: the Ra is the base
        has_reph = false;
    }

    // places: before the base, the base, the reph
    for g in &mut info[start..base.min(end)] {
        g.pos = g.pos.min(POS_PRE_C);
    }
    if base < end {
        info[base].pos = POS_BASE;
    }
    if has_reph {
        info[start].pos = POS_REPH;
    }
    // marks go with what they follow
    let mut last = POS_START;
    for i in start..end {
        if matches!(info[i].cat, ZWJ | ZWNJ | N | H) {
            info[i].pos = last;
            if info[i].cat == H && info[i].pos == POS_PRE_M {
                for j in (start + 1..=i).rev() {
                    if info[j - 1].pos != POS_PRE_M {
                        info[i].pos = info[j - 1].pos;
                        break;
                    }
                }
            }
        } else if info[i].pos != POS_SMVD {
            last = info[i].pos;
        }
    }
    // a post-base consonant owns what lies between it and the last consonant or vowel sign
    {
        let mut last = base;
        for i in base + 1..end {
            if is_consonant(info[i].cat) {
                for j in last + 1..i {
                    if info[j].pos < POS_SMVD {
                        info[j].pos = info[i].pos;
                    }
                }
                last = i;
            } else if info[i].cat == M {
                last = i;
            }
        }
    }
    // in order (a stable sort)
    let mut order: Vec<usize> = (start..end).collect();
    order.sort_by_key(|&k| info[k].pos);
    let sorted: Vec<Info> = order.iter().map(|&k| info[k]).collect();
    info.copy_from_slice(&sorted);
    base = info.iter().position(|g| g.pos == POS_BASE).unwrap_or(end);
    // after the base, what changed places is one cluster: each cycle of the
    // permutation, from its first place to its last
    {
        let mut seen = vec![false; end];
        for i in base..end {
            if seen[i] {
                continue;
            }
            let (mut min, mut max) = (i, i);
            let mut j = order[i];
            while j != i {
                min = min.min(j);
                max = max.max(j);
                seen[j] = true;
                j = order[j];
            }
            if i != max {
                merge(info, min, max + 1);
            }
        }
    }
    // left vowel signs, more than one: in reverse
    let lefts: Vec<usize> = (start..base.min(end)).filter(|&i| info[i].pos == POS_PRE_M).collect();
    if lefts.len() > 1 {
        let (a, b) = (lefts[0], *lefts.last().unwrap());
        info[a..=b].reverse();
    }

    // which features may change which glyphs
    for g in info.iter_mut().take_while(|g| g.pos == POS_REPH) {
        g.fmask |= F_RPHF;
    }
    let pre = F_HALF | if script.blwf_before_base() { F_BLWF } else { 0 };
    for g in &mut info[start..base.min(end)] {
        g.fmask |= pre;
    }
    for g in info.iter_mut().skip(base + 1) {
        g.fmask |= F_BLWF | F_ABVF | F_PSTF;
    }
    if !gsub.lookups_for(&[b"pref"]).is_empty() && base + 2 < end {
        for i in base + 1..end - 1 {
            if shaper.would_substitute(gsub, &[b"pref"], &[info[i].glyph, info[i + 1].glyph]) {
                info[i].fmask |= F_PREF;
                info[i + 1].fmask |= F_PREF;
                break;
            }
        }
    }
    // a ZWNJ stops the half form of what it follows
    for i in start + 1..end {
        if info[i].cat == ZWNJ {
            let mut j = i;
            loop {
                j -= 1;
                info[j].fmask &= !F_HALF;
                if j <= start || is_consonant(info[j].cat) {
                    break;
                }
            }
        }
    }

    // the basic features, one at a time
    for (tag, bits) in BASIC {
        for l in gsub.lookups_for(&[tag]) {
            shaper.gsub_lookup_bits(gsub, l, info, bits)?;
        }
    }

    final_reordering(script, reph_pos, info);
    Ok(())
}

fn final_reordering(script: Indic, reph_pos: RephPos, info: &mut [Info]) {
    let (start, end) = (0, info.len());
    if end == 0 {
        return;
    }
    let is_halant = |g: &Info| g.cat == H && !g.ligated;

    // the base again
    let mut base = (start..end).find(|&i| info[i].pos >= POS_BASE).unwrap_or(end);
    if base < end && start < base && info[base].pos > POS_BASE {
        base -= 1;
    }
    if base == end && start < base && info[base - 1].cat == ZWJ {
        base -= 1;
    }
    if base < end {
        while start < base && matches!(info[base].cat, N | H) {
            base -= 1;
        }
    }

    // a pre-base vowel sign: after the last virama still standing before the base
    if start + 1 < end && start < base {
        let mut new_pos = if base == end { base.saturating_sub(2) } else { base - 1 };
        if !matches!(script, Indic::Mlym | Indic::Taml) {
            loop {
                while new_pos > start && !matches!(info[new_pos].cat, M | H) {
                    new_pos -= 1;
                }
                if is_halant(&info[new_pos]) && info[new_pos].pos != POS_PRE_M {
                    if new_pos + 1 < end && info[new_pos + 1].cat == ZWJ && new_pos > start {
                        new_pos -= 1;
                        continue;
                    }
                } else {
                    new_pos = start;
                }
                break;
            }
        }
        if start < new_pos && info[new_pos].pos != POS_PRE_M {
            let mut i = new_pos;
            while i > start {
                if info[i - 1].pos == POS_PRE_M {
                    let old = i - 1;
                    if old < base && base <= new_pos {
                        base -= 1;
                    }
                    info[old..=new_pos].rotate_left(1);
                    merge(info, new_pos, end.min(base + 1));
                    new_pos -= 1;
                }
                i -= 1;
            }
        } else if let Some(i) = (start..base).find(|&i| info[i].pos == POS_PRE_M) {
            merge(info, i, end.min(base + 1));
        }
    }

    // the reph, to where it is drawn
    if start + 1 < end && info[start].pos == POS_REPH && ((info[start].cat == REPHA) ^ (info[start].ligated && !info[start].multiplied)) {
        let explicit_halant = |info: &[Info]| -> Option<usize> {
            let mut p = start + 1;
            while p < base && !is_halant(&info[p]) {
                p += 1;
            }
            (p < base && is_halant(&info[p])).then(|| if p + 1 < base && is_joiner(info[p + 1].cat) { p + 1 } else { p })
        };
        let mut new_reph = None;
        if reph_pos != RephPos::AfterPost {
            new_reph = explicit_halant(info);
            if new_reph.is_none() && reph_pos == RephPos::AfterMain {
                let mut p = base;
                while p + 1 < end && info[p + 1].pos <= POS_AFTER_MAIN {
                    p += 1;
                }
                if p < end {
                    new_reph = Some(p);
                }
            }
            if new_reph.is_none() && reph_pos == RephPos::AfterSub {
                let mut p = base;
                while p + 1 < end && !matches!(info[p + 1].pos, POS_POST_C | POS_AFTER_POST | POS_SMVD) {
                    p += 1;
                }
                if p < end {
                    new_reph = Some(p);
                }
            }
        }
        let new_reph = new_reph.or_else(|| explicit_halant(info)).unwrap_or_else(|| {
            let mut p = end - 1;
            while p > start && info[p].pos == POS_SMVD {
                p -= 1;
            }
            if is_halant(&info[p]) && (base + 1..p).any(|i| info[i].cat == M) {
                p -= 1;
            }
            p
        });
        if new_reph > start {
            merge(info, start, new_reph + 1);
            info[start..=new_reph].rotate_left(1);
            if start < base && base <= new_reph {
                base -= 1;
            }
        }
    }

    // a pre-base-reordering consonant made by `pref`: before the base
    if base + 1 < end {
        if let Some(i) = (base + 1..end).find(|&i| info[i].fmask & F_PREF != 0) {
            if info[i].ligated && !info[i].multiplied {
                let mut new_pos = base;
                if !matches!(script, Indic::Mlym | Indic::Taml) {
                    while new_pos > start && !matches!(info[new_pos - 1].cat, M | H) {
                        new_pos -= 1;
                    }
                }
                if new_pos > start && is_halant(&info[new_pos - 1]) && new_pos < end && is_joiner(info[new_pos].cat) {
                    new_pos += 1;
                }
                merge(info, new_pos, i + 1);
                info[new_pos..=i].rotate_right(1);
            }
        }
    }
}

/// A Khmer syllable: a Coeng and Ro, and a vowel sign that goes before
/// the base, moved to its start; the basic features.
fn khmer_syllable(shaper: &Shaper, gsub: &Layout, info: &mut Vec<Info>) -> Result<(), Unsupported> {
    let end = info.len();
    let mut syl = std::mem::take(info);
    for g in syl.iter_mut().skip(1) {
        g.fmask |= F_BLWF | F_ABVF | F_PSTF;
    }
    let mut coengs = 0;
    let mut i = 1;
    while i < end {
        if syl[i].cat == H && coengs <= 2 && i + 1 < end {
            coengs += 1;
            if syl[i + 1].cat == RA {
                syl[i].fmask |= F_PREF;
                syl[i + 1].fmask |= F_PREF;
                merge(&mut syl, 0, i + 2);
                syl[0..i + 2].rotate_right(2);
                for g in &mut syl[i + 2..] {
                    g.fmask |= F_CFAR;
                }
                coengs = 2;
            }
        } else if syl[i].cat == M && syl[i].pos == POS_PRE_M {
            merge(&mut syl, 0, i + 1);
            syl[0..=i].rotate_right(1);
        }
        i += 1;
    }
    // one stage: each lookup once, for the glyphs of all its features
    let tags: Vec<&[u8; 4]> = KHMER_BASIC.iter().map(|f| f.0).collect();
    let mut lookups: Vec<(u16, u32)> = Vec::new();
    for (l, t) in gsub.lookups_tagged(&tags) {
        let bits = KHMER_BASIC.iter().find(|f| *f.0 == t).map_or(0, |f| f.1);
        match lookups.iter_mut().find(|x| x.0 == l) {
            Some(x) => x.1 |= bits,
            None => lookups.push((l, bits)),
        }
    }
    for (l, bits) in lookups {
        shaper.gsub_lookup_bits(gsub, l, &mut syl, bits)?;
    }
    *info = syl;
    Ok(())
}

/// A Myanmar syllable (HarfBuzz's Myanmar shaper): a kinzi after the
/// base, the medial Ra and the vowel sign E before it, the basic features
/// each on the whole syllable.
fn myanmar_syllable(shaper: &Shaper, gsub: &Layout, info: &mut Vec<Info>) -> Result<(), Unsupported> {
    let (start, end) = (0, info.len());
    let mut limit = start;
    let mut base = end;
    let kinzi = start + 3 <= end && info[start].cat == RA && info[start + 1].cat == ASAT && info[start + 2].cat == H;
    if kinzi {
        limit += 3;
        base = start;
    }
    if !kinzi {
        base = limit;
    }
    if let Some(i) = (limit..end).find(|&i| is_consonant(info[i].cat)) {
        base = i;
    }
    let mut i = start;
    while i < start + if kinzi { 3 } else { 0 } {
        info[i].pos = POS_AFTER_MAIN;
        i += 1;
    }
    while i < base.min(end) {
        info[i].pos = POS_PRE_C;
        i += 1;
    }
    if i < end {
        info[i].pos = POS_BASE;
        i += 1;
    }
    let mut pos = POS_AFTER_MAIN;
    while i < end {
        let cat = info[i].cat;
        let vowel_below = cat == VBLW;
        info[i].pos = if cat == MEDIAL_RA {
            POS_PRE_C
        } else if cat == M && info[i].pos == POS_PRE_M {
            POS_PRE_M
        } else if pos == POS_AFTER_MAIN && vowel_below {
            pos = POS_BELOW_C;
            pos
        } else if pos == POS_BELOW_C && cat == ASAT {
            POS_BEFORE_SUB
        } else if pos == POS_BELOW_C && vowel_below {
            pos
        } else if pos == POS_BELOW_C {
            pos = POS_AFTER_SUB;
            pos
        } else {
            pos
        };
        i += 1;
    }
    let mut order: Vec<usize> = (start..end).collect();
    order.sort_by_key(|&k| info[k].pos);
    if order.iter().enumerate().any(|(n, &k)| n != k) {
        let first = order.iter().enumerate().position(|(n, &k)| n != k).unwrap_or(0);
        let last = order.iter().enumerate().rposition(|(n, &k)| n != k).unwrap_or(0);
        merge(info, first, last + 1);
    }
    let sorted: Vec<Info> = order.iter().map(|&k| info[k]).collect();
    info.copy_from_slice(&sorted);
    for tag in [b"rphf", b"pref", b"blwf", b"pstf"] {
        for l in gsub.lookups_for(&[tag]) {
            shaper.gsub_lookup_bits(gsub, l, info, F_GLOBAL)?;
        }
    }
    Ok(())
}
