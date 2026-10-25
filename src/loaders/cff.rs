//! `CFF`: PostScript outlines in an sfnt, drawn by Type 2 charstrings.
//!
//! The system font is `glyf`, but 59 of the 372 font files macOS ships keep their
//! outlines here instead — every STIX face, all of Hiragino, Apple SD Gothic Neo,
//! Kohinoor, ITF Devanagari, Tamil Sangam, Sign Painter, Last Resort and seven of the
//! Noto faces. None of them is reachable through [`crate::glyf`], so without this the
//! crate reads their metrics and draws nothing.
//!
//! # What this format is, as these fonts use it
//!
//! A charstring is a little stack program, not a point list: `rrcurveto` takes six
//! deltas off the stack and leaves the current point where it ends. Two consequences
//! shape the code below.
//!
//! The first is that the **operand count is part of the operator**. `hvcurveto` draws as
//! many curves as it has arguments for, alternating its starting direction each time, and
//! the last group may carry a fifth argument the others do not. So may `hhcurveto`'s
//! first, as a one-off `dy` before an otherwise horizontal run. Reading the count wrong
//! does not fail — it draws a different shape.
//!
//! The second is that **a glyph is not self-contained**. Repeated fragments live in
//! subroutines, two arrays of them: one global to the font and one local, and on a
//! CID-keyed font the local one depends on which font DICT `FDSelect` assigns the glyph.
//! Hiragino's 20,339 glyphs share 837 global subroutines between them. Subroutine
//! numbers are *biased* by the array's own size, so the same number means different
//! things in a 100-entry and a 1500-entry array.
//!
//! # The `FontMatrix` goes through `Fract`
//!
//! A CFF font states its own size twice: `head.unitsPerEm`, and the `FontMatrix` in the
//! Top DICT that maps a charstring coordinate into the em. Two fonts here carry a matrix
//! — Tamil Sangam MN and Last Resort, both `1/2048` against a 2048 em — and it is not
//! quite the identity it looks like, because the matrix is written as the decimal
//! `.00048828` and not as the exact `1/2048`.
//!
//! Core Text's `H` in Tamil Sangam starts at x `160.99969291687012` where the charstring
//! says 161. That is `161 * (1 - 2^-19)` to the last bit, so the scale Core Graphics used
//! is `524287 / 2^19` — and `524287 / 2^30` is what `.00048828` becomes when it is
//! **rounded to a multiple of `2^-30`**, which is `Fract`, the 2.30 fixed point of the
//! Mac OS numerics these fonts' parser long predates. No other power of two fits: `2^-29`
//! gives `1 - 2^-18`, `2^-31` gives `1 - 3*2^-20`, and `2^-28` gives exactly 1.
//! [`fract`] is that rounding, and [`Cff::parse`] folds it into the em so there is a
//! single exact multiply on the way out.
//!
//! When the Top DICT has **no** `FontMatrix`, the spec's default is `0.001` — and Core
//! Graphics does not use it. Hiragino settles that: the second face of every weight has a
//! **1083** em over charstrings drawn on 1000, and through a 0.001 default its `A` would
//! span `6.498 .. 829.578` where Core Text reports `6 .. 766`. So an absent matrix means
//! the em cancels, not that it is 0.001, and that face's glyphs really do sit 8% small
//! inside their own em. `examples/cff_matrix.rs` in `swiftui-native` is the measurement;
//! it reaches that face through `CTFontManagerCreateFontDescriptorsFromURL`, because
//! `CGFontCreateWithDataProvider` hands back only the first face of a collection.
//!
//! # `CFF2`
//!
//! One file on the machine — `SFIndia.ttc`, nine faces — uses `CFF2` instead, which is the
//! same charstrings with the variation machinery folded in. The differences that matter
//! here are small: the Top DICT is a plain run of bytes rather than an `INDEX` of one, the
//! `INDEX`es count in `u32`, a charstring carries no leading width and has no `endchar`,
//! and two operators are new. `vsindex` selects a variation data set and `blend` replaces
//! operands with a base plus a delta per region.
//!
//! Its `opsz` axis runs 17..28 with the default at **28**, so a comparison made at the em
//! clamps to the default and sees no variation at all — which is why `blend` could read
//! the region *count* alone for a while and drop the deltas. At any other instance they
//! are the font: SF India's `:` is 57 cells out at six sizes without them.
//!
//! `blend` takes `n` base values followed by `n * k` deltas **grouped by operand** — all
//! of the first value's, then the second's — and `vsindex` selects which
//! `ItemVariationData` names the regions. The arithmetic is the item variation store's and
//! not `gvar`'s: double, with one fused multiply-add per region. See
//! `ItemVariationStore::region_scalars` and `Cff::region_scalars`.
//!
//! # Nothing is dropped here
//!
//! A charstring's `0 vlineto` draws a line to where the pen already is, and a `0 0 0 0
//! vhcurveto` a curve with all three points there. The interpreter emits both, literally:
//! what removes them is [`crate::normalise`], the pass Core Graphics runs afterwards, and
//! leaving them in matters because that pass **steps over the element after a removal**.
//! STIX Two Math's glyph 3501 ends on a zero-length line, a curve that goes nowhere and a
//! real curve; the pass drops the line, skips the curve, and Core Text reports the curve
//! that goes nowhere. Dropping it here instead takes the glyph one element short and the
//! pass then removes the wrong one.
//!
//! Points are streamed through [`Outline`] as cubics, which is what `CGPathApply`
//! reports for these fonts and the reason that method exists on the trait at all.

use crate::curves::Path;
use crate::math::Vector2;

/// A sink for one glyph's contours, which is all a charstring produces.
///
/// `glyf` and `CFF` are two outline formats with two curve degrees and this takes both:
/// elevating a quadratic to a cubic is exact and the reverse is not, so neither can stand
/// in for the other. `glyf` never calls [`Self::cubic_to`]; `CFF` never calls
/// [`Self::quad_to`].
pub trait Outline {
    fn move_to(&mut self, x: f64, y: f64);
    fn line_to(&mut self, x: f64, y: f64);
    /// One quadratic Bézier, which is the only curve `glyf` has.
    fn quad_to(&mut self, cx: f64, cy: f64, x: f64, y: f64);
    /// One cubic Bézier, which is the only curve `CFF` has.
    fn cubic_to(&mut self, c1x: f64, c1y: f64, c2x: f64, c2y: f64, x: f64, y: f64);
    fn close(&mut self);
}

/// Every contour of a glyph in one [`Path`], the way `shape_from_contours` builds a
/// `glyf` glyph — each contour opened by a `move_to`, holes sorted out later by
/// `generate_shapes`.
#[derive(Default)]
pub struct PathSink {
    /// The contours, ready for `Shape { outline, .. }`.
    pub path: Path,
    /// Where the open contour started, so `close` can return to it.
    start: Option<(f64, f64)>,
    last: Option<(f64, f64)>,
}

impl PathSink {
    fn to(&mut self, x: f64, y: f64) -> Vector2 {
        self.last = Some((x, y));
        Vector2::new(x as f32, y as f32)
    }
}

impl Outline for PathSink {
    fn move_to(&mut self, x: f64, y: f64) {
        self.start = Some((x, y));
        let p = self.to(x, y);
        self.path.move_to(p);
    }
    fn line_to(&mut self, x: f64, y: f64) {
        let p = self.to(x, y);
        self.path.line_to(p);
    }
    fn quad_to(&mut self, cx: f64, cy: f64, x: f64, y: f64) {
        let c = Vector2::new(cx as f32, cy as f32);
        let p = self.to(x, y);
        self.path.quadratic_curve_to(c, p);
    }
    fn cubic_to(&mut self, c1x: f64, c1y: f64, c2x: f64, c2y: f64, x: f64, y: f64) {
        let c1 = Vector2::new(c1x as f32, c1y as f32);
        let c2 = Vector2::new(c2x as f32, c2y as f32);
        let p = self.to(x, y);
        self.path.bezier_curve_to(c1, c2, p);
    }
    fn close(&mut self) {
        // A charstring's contours are implicitly closed, and `Path` has no close of its
        // own — a line back to the start is the same thing, and only needed when the
        // contour did not already end there.
        if let (Some(s), Some(l)) = (self.start, self.last) {
            if (s.0 - l.0).abs() > 1e-6 || (s.1 - l.1).abs() > 1e-6 {
                let p = self.to(s.0, s.1);
                self.path.line_to(p);
            }
        }
    }
}

/// A glyph id. `CFF` indexes its charstrings by this directly.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GlyphId(pub u16);

/// A 2×3 transform, for the `FontMatrix`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Affine {
    pub(crate) a: f64,
    pub(crate) b: f64,
    pub(crate) c: f64,
    pub(crate) d: f64,
    pub(crate) x: f64,
    pub(crate) y: f64,
}

impl Affine {
    pub(crate) const IDENTITY: Self =
        Self { a: 1.0, b: 0.0, c: 0.0, d: 1.0, x: 0.0, y: 0.0 };

    pub(crate) fn apply(self, px: f64, py: f64) -> (f64, f64) {
        (self.a * px + self.c * py + self.x, self.b * px + self.d * py + self.y)
    }
}

/// Half away from zero, which is how a `FontMatrix` lands back on its own grid.
fn round_half_away(v: f64) -> f64 {
    if v >= 0.0 {
        ((v + 0.5) as i64) as f64
    } else {
        -(((-v + 0.5) as i64) as f64)
    }
}

fn i16_at(b: &[u8], p: usize) -> Option<i16> {
    u16_at(b, p).map(|v| v as i16)
}

fn i32_at(b: &[u8], p: usize) -> Option<i32> {
    u32_at(b, p).map(|v| v as i32)
}

fn u8_at(b: &[u8], p: usize) -> Option<u8> {
    b.get(p).copied()
}

fn u16_at(b: &[u8], p: usize) -> Option<u16> {
    Some(u16::from_be_bytes([*b.get(p)?, *b.get(p + 1)?]))
}

fn u32_at(b: &[u8], p: usize) -> Option<u32> {
    Some(u32::from_be_bytes([
        *b.get(p)?,
        *b.get(p + 1)?,
        *b.get(p + 2)?,
        *b.get(p + 3)?,
    ]))
}

/// A big-endian unsigned integer of `size` bytes, 1 to 4 — `CFF`'s `offSize`.
fn uint_at(b: &[u8], p: usize, size: usize) -> Option<u32> {
    let mut v = 0u32;
    for i in 0..size {
        v = (v << 8) | u32::from(*b.get(p + i)?);
    }
    Some(v)
}

/// How deep subroutine calls may nest. The spec's own limit is 10.
const MAX_DEPTH: u8 = 10;

/// The Type 2 operand stack.
///
/// `CFF`'s limit is 48 and `CFF2`'s is **513**, and the difference is not academic: SF
/// India's charstrings push fifty-odd operands before a `blend`, and its Private DICT
/// `BlueValues` is a 73-entry blended array. A 48-slot stack silently drops the overflow
/// and the glyph stops drawing after its first `moveto`.
const MAX_STACK: usize = 513;

/// How many variation regions one `ItemVariationData` may name.
///
/// `CFF2`'s `blend` multiplies every operand by one scalar per region, so these are held
/// on the stack rather than recomputed per operand. SF India — the one `CFF2` font on
/// this machine — names **four** across its nine faces; the ceiling is for a font that
/// names more, which stops the glyph rather than reading past the array.
const MAX_REGIONS: usize = 64;

/// An `INDEX`: CFF's one container, a count and a run of offsets into the bytes that
/// follow it.
///
/// Offsets are 1-based and `offSize` bytes wide, so `base` is one *before* the data —
/// not a mistake, it is how the format is written.
#[derive(Clone, Copy, Debug, Default)]
struct Index<'a> {
    /// The whole `CFF ` table. Kept rather than a sub-slice so that a bad offset is
    /// caught against the table's real end.
    data: &'a [u8],
    count: usize,
    off_size: usize,
    /// Where the offset array starts.
    offs: usize,
    /// One byte before the data, which is where a 1-based offset is measured from.
    base: usize,
}

impl<'a> Index<'a> {
    /// Read the `INDEX` at `at`; returns it and the offset just past its data.
    fn parse(data: &'a [u8], at: usize) -> Option<(Self, usize)> {
        Self::parse_wide(data, at, false)
    }

    /// The same, with `CFF2`'s `u32` count when `wide`.
    ///
    /// The only difference between the two formats' container, and it shifts everything
    /// after it by two bytes.
    fn parse_wide(data: &'a [u8], at: usize, wide: bool) -> Option<(Self, usize)> {
        let head = if wide { 4 } else { 2 };
        let count = if wide {
            u32_at(data, at)? as usize
        } else {
            usize::from(u16_at(data, at)?)
        };
        if count == 0 {
            // An empty INDEX is its count and no offset array at all.
            return Some((Self { data, ..Self::default() }, at.checked_add(head)?));
        }
        let off_size = usize::from(u8_at(data, at.checked_add(head)?)?);
        if off_size == 0 || off_size > 4 {
            return None;
        }
        let offs = at.checked_add(head + 1)?;
        let base = offs.checked_add(count.checked_add(1)?.checked_mul(off_size)?)? - 1;
        let last = uint_at(data, offs.checked_add(count.checked_mul(off_size)?)?, off_size)?;
        let end = base.checked_add(last as usize)?;
        if end > data.len() {
            return None;
        }
        Some((Self { data, count, off_size, offs, base }, end))
    }

    fn get(&self, i: usize) -> Option<&'a [u8]> {
        if i >= self.count {
            return None;
        }
        let at = |k: usize| -> Option<usize> {
            let o = uint_at(self.data, self.offs.checked_add(k.checked_mul(self.off_size)?)?, self.off_size)?;
            self.base.checked_add(o as usize)
        };
        let (a, b) = (at(i)?, at(i + 1)?);
        self.data.get(a..b)
    }

    /// The number a `callsubr` has to add to its argument.
    ///
    /// Subroutine numbers are signed and centred, so that the short integer encodings
    /// cover the middle of the array wherever it starts. The three thresholds are the
    /// spec's.
    fn bias(&self) -> i32 {
        if self.count < 1240 {
            107
        } else if self.count < 33900 {
            1131
        } else {
            32768
        }
    }
}

/// Call `f` for each operator in a DICT, with the operands that preceded it.
///
/// The two-byte escapes are keyed `0x0c00 | b2`, which keeps one `match` arm per
/// operator and matches how the spec numbers them (`12 7` is `FontMatrix`).
fn dict_scan(d: &[u8], mut f: impl FnMut(u16, &[f64])) {
    let mut ops = [0.0f64; MAX_STACK];
    let mut n = 0usize;
    let mut i = 0usize;
    let push = |ops: &mut [f64; MAX_STACK], n: &mut usize, v: f64| {
        if *n < MAX_STACK {
            ops[*n] = v;
            *n += 1;
        }
    };
    while i < d.len() {
        let b = d[i];
        match b {
            // The operand prefixes come first, because the operator range runs past them:
            // `CFF2` added `vstore` at **24** and `maxstack` at 25, and a `CFF` reader that
            // stops at 21 stops in the middle of SF India's Top DICT — which is how its
            // variation store went missing and every glyph in it drew nothing.
            28 => {
                let Some(v) = i16_at(d, i + 1) else { return };
                push(&mut ops, &mut n, f64::from(v));
                i += 3;
            }
            29 => {
                let Some(v) = i32_at(d, i + 1) else { return };
                push(&mut ops, &mut n, f64::from(v));
                i += 5;
            }
            30 => {
                let (v, next) = real_at(d, i + 1);
                push(&mut ops, &mut n, v);
                i = next;
            }
            32..=246 => {
                push(&mut ops, &mut n, f64::from(i16::from(b) - 139));
                i += 1;
            }
            247..=250 => {
                let Some(&lo) = d.get(i + 1) else { return };
                let v = (i32::from(b) - 247) * 256 + i32::from(lo) + 108;
                push(&mut ops, &mut n, f64::from(v));
                i += 2;
            }
            251..=254 => {
                let Some(&lo) = d.get(i + 1) else { return };
                let v = -((i32::from(b) - 251) * 256) - i32::from(lo) - 108;
                push(&mut ops, &mut n, f64::from(v));
                i += 2;
            }
            // Everything below 32 that is not an operand prefix is an operator. 22, 23,
            // 26, 27 and 31 are reserved and will simply match no key below.
            0..=31 => {
                let (key, adv) = if b == 12 {
                    match d.get(i + 1) {
                        Some(&b2) => (0x0c00 | u16::from(b2), 2),
                        None => return,
                    }
                } else {
                    (u16::from(b), 1)
                };
                f(key, &ops[..n]);
                n = 0;
                i += adv;
            }
            // 255 does not appear in a DICT. Stop rather than guess.
            _ => return,
        }
    }
}

/// A DICT's real-number encoding: nibbles, digits as themselves and five escapes.
///
/// Returns the value and the offset past it. Used only by `FontMatrix` in practice,
/// which is why it reassembles decimal text and parses that: `0.00048828` has to come
/// out as the same `f64` the font means, and a hand-rolled mantissa would not.
fn real_at(d: &[u8], mut i: usize) -> (f64, usize) {
    let mut buf = [0u8; 40];
    let mut n = 0usize;
    let mut put = |b: u8, n: &mut usize| {
        if *n < buf.len() {
            buf[*n] = b;
            *n += 1;
        }
    };
    'outer: while i < d.len() {
        let byte = d[i];
        i += 1;
        for nib in [byte >> 4, byte & 0x0f] {
            match nib {
                0..=9 => put(b'0' + nib, &mut n),
                0x0a => put(b'.', &mut n),
                0x0b => put(b'E', &mut n),
                0x0c => {
                    put(b'E', &mut n);
                    put(b'-', &mut n);
                }
                0x0e => put(b'-', &mut n),
                0x0f => break 'outer,
                // 0x0d is reserved; skipping it is what every reader does.
                _ => {}
            }
        }
    }
    let v = core::str::from_utf8(&buf[..n]).ok().and_then(|s| s.parse::<f64>().ok());
    (v.unwrap_or(0.0), i)
}

/// `Fract`: the 2.30 fixed point a `FontMatrix` component lands on.
///
/// Solved for rather than assumed — see the module docs. `2^30` is exact in `f64` and so
/// is the quotient of any integer by it, so this introduces no error of its own.
fn fract(v: f64) -> f64 {
    let g = 1_073_741_824.0;
    round_half_away(v * g) / g
}

/// A font's `CFF ` table, resolved down to the four things drawing a glyph needs.
///
/// Not `Copy`, unlike the version this was ported from: `CFF2`'s variation store owns its
/// per-region scalars, computed once for the location the font was read at.
///
/// `charset`, `glyph_for_sid` and `std_stems` are read by `swiftui-font` and by nothing
/// here — the first two name glyphs and the third feeds an autohinter, neither of which
/// threers has. They are kept rather than deleted so this file stays diffable against the
/// original when that one changes.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub(crate) struct Cff<'a> {
    /// The whole table. Every offset in a DICT is measured from its start.
    data: &'a [u8],
    char_strings: Index<'a>,
    gsubrs: Index<'a>,
    /// The Top DICT's own `Subrs`. Empty on a CID-keyed font, where each glyph's come
    /// from the font DICT `FDSelect` picks instead.
    subrs: Index<'a>,
    /// `FDArray`, present with `FDSelect` on a CID-keyed font. Its font DICTs are
    /// resolved per glyph rather than up front: Hiragino has 20,339 glyphs across them
    /// and this struct has to stay copyable.
    fd_array: Option<Index<'a>>,
    fd_select: Option<usize>,
    /// The Top DICT's `FontMatrix` through [`fract`] and scaled by the em, so one
    /// multiply turns a charstring coordinate into a font unit. Identity when the Top
    /// DICT has no matrix at all, which is every font here but two.
    xf: Affine,
    /// `true` for `CFF2`, which changes the container, the charstring preamble and two
    /// operators. See the module docs.
    wide: bool,
    /// `CFF2`'s own `ItemVariationStore`, opened at the location the font was read at —
    /// the offset comes from the Top DICT's `vstore` operator.
    store: Option<super::ttf_var::ItemVariationStore<'a>>,
    /// The Top DICT's `charset` offset, which names the glyphs. `0`, `1` and `2` are the
    /// three predefined charsets rather than offsets. See [`Cff::glyph_for_sid`].
    charset: usize,
    /// The Private DICT's `StdHW` and `StdVW` — the standard horizontal and vertical stem
    /// widths, in charstring units. `(0, 0)` when it carries neither, which is how a
    /// `CFF` font says it has no answer. See [`crate::Font::stem`].
    pub(crate) std_stems: (f64, f64),
}

#[allow(dead_code)]
impl Cff<'_> {
    /// The glyph a standard string identifies, through the `charset`.
    ///
    /// A `CFF` font keeps its glyph names here and not in `post` — which is version 3 and
    /// nameless in every OpenType `CFF` face — so this is the lookup
    /// `TFPFont::GetGlyph("'X'")` ends up in for them. `X` is standard string 57 and `o`
    /// is 80, and STIX General, whose charset is the predefined `ISOAdobe` one, therefore
    /// answers glyph 57 and glyph 80 — which is exactly what Core Graphics asks
    /// `CGFontGetGlyphBBoxes` about.
    ///
    /// The two expert charsets are not handled: no font on this machine uses one, and a
    /// wrong answer there would be worse than none.
    pub(crate) fn glyph_for_sid(&self, sid: u16) -> Option<u16> {
        match self.charset {
            // `ISOAdobe`: the first 229 glyphs are the first 229 standard strings, in
            // order, so the identity is the mapping.
            0 => (sid < 229 && usize::from(sid) < self.char_strings.count).then_some(sid),
            1 | 2 => None,
            at => {
                let fmt = u8_at(self.data, at)?;
                let n = u32::try_from(self.char_strings.count).ok()?;
                let mut gid = 1u32;
                match fmt {
                    0 => {
                        while gid < n {
                            let s = u16_at(self.data, at + 1 + (gid as usize - 1) * 2)?;
                            if s == sid {
                                return u16::try_from(gid).ok();
                            }
                            gid += 1;
                        }
                        None
                    }
                    1 | 2 => {
                        let step = if fmt == 1 { 3 } else { 4 };
                        let mut p = at + 1;
                        while gid < n {
                            let first = u16_at(self.data, p)?;
                            let left = if fmt == 1 {
                                u32::from(u8_at(self.data, p + 2)?)
                            } else {
                                u32::from(u16_at(self.data, p + 2)?)
                            };
                            if sid >= first && u32::from(sid - first) <= left {
                                return u16::try_from(gid + u32::from(sid - first)).ok();
                            }
                            gid += left + 1;
                            p += step;
                        }
                        None
                    }
                    _ => None,
                }
            }
        }
    }
}

impl<'a> Cff<'a> {
    /// Read the table's INDEXes and Top DICT.
    ///
    /// `upem` is `head.unitsPerEm`, which `FontMatrix` is expressed against.
    /// `coords` are the normalized axis coordinates, which `CFF2`'s own variation store
    /// is opened at. Empty for a static font, and harmless for one: a `CFF` table has no
    /// `vstore` operator, so there is nothing to open.
    pub(crate) fn parse(data: &'a [u8], upem: u16, coords: &[f32]) -> Option<Self> {
        // The header's length is its third byte in `CFF` and `CFF2` alike: what follows
        // starts there, not at a fixed offset, because a header may carry more than the
        // bytes the version defines.
        let wide = u8_at(data, 0)? >= 2;
        let hdr = usize::from(u8_at(data, 2)?);
        if hdr < 4 {
            return None;
        }
        // `CFF2` keeps its Top DICT as a plain run of bytes with its length in the
        // header; `CFF` keeps it as an `INDEX` of one, behind the Name `INDEX`.
        let (top, gsubrs) = if wide {
            let len = usize::from(u16_at(data, 3)?);
            let top = data.get(hdr..hdr.checked_add(len)?)?;
            let (gsubrs, _) = Index::parse_wide(data, hdr.checked_add(len)?, true)?;
            (top, gsubrs)
        } else {
            let (_names, p) = Index::parse(data, hdr)?;
            let (tops, p) = Index::parse(data, p)?;
            let (_strings, p) = Index::parse(data, p)?;
            let (gsubrs, _) = Index::parse(data, p)?;
            (tops.get(0)?, gsubrs)
        };

        let (mut cs_off, mut priv_at, mut fda, mut fds) = (0usize, None, None, None);
        let mut charset = 0usize;
        let mut matrix = None;
        let mut vstore = None;
        dict_scan(top, |key, ops| match (key, ops) {
            (17, [o, ..]) => cs_off = *o as usize,
            (15, [o, ..]) => charset = *o as usize,
            // `vstore` points at a `u16` length and then the store itself.
            (24, [o, ..]) => vstore = Some(*o as usize + 2),
            (18, [size, off, ..]) => priv_at = Some((*off as usize, *size as usize)),
            (0x0c24, [o, ..]) => fda = Some(*o as usize),
            (0x0c25, [o, ..]) => fds = Some(*o as usize),
            (0x0c07, [a, b, c, d, x, y]) => {
                let e = f64::from(upem);
                matrix = Some(Affine {
                    a: fract(*a) * e,
                    b: fract(*b) * e,
                    c: fract(*c) * e,
                    d: fract(*d) * e,
                    x: fract(*x) * e,
                    y: fract(*y) * e,
                });
            }
            _ => {}
        });
        if cs_off == 0 {
            return None;
        }
        let (char_strings, _) = Index::parse_wide(data, cs_off, wide)?;
        if char_strings.count == 0 {
            return None;
        }

        Some(Self {
            data,
            char_strings,
            gsubrs,
            subrs: priv_at
                .and_then(|(o, n)| private_subrs(data, o, n, wide))
                .unwrap_or_default(),
            fd_array: fda.and_then(|o| Index::parse_wide(data, o, wide).map(|(i, _)| i)),
            fd_select: fds,
            xf: matrix.unwrap_or(Affine::IDENTITY),
            wide,
            store: vstore
                .and_then(|at| super::ttf_var::ItemVariationStore::new(data, at, coords)),
            charset,
            std_stems: priv_at
                .and_then(|(o, n)| private_stems(data, o, n))
                // A CID-keyed font has no Top DICT Private; its widths live in the
                // **first** font DICT of the `FDArray`. Hiragino Kaku Gothic W4 is the
                // case that says which: its `meta` header carries 84 and 95 and its
                // `FDArray[0]` carries 77 and 91, and `FPFontGetStemH` answers 77.
                .or_else(|| {
                    let fda = fda.and_then(|o| Index::parse_wide(data, o, wide).map(|(i, _)| i))?;
                    let fd = fda.get(0)?;
                    let mut at = None;
                    dict_scan(fd, |key, ops| {
                        if let (18, [size, off, ..]) = (key, ops) {
                            at = Some((*off as usize, *size as usize));
                        }
                    });
                    let (o, n) = at?;
                    private_stems(data, o, n)
                })
                .unwrap_or((0.0, 0.0)),
        })
    }

    /// Which local subroutines a glyph uses.
    ///
    /// On a CID-keyed font that is the Private DICT of the font DICT `FDSelect` names;
    /// on every other font it is the Top DICT's own, and `FDSelect` is absent.
    fn local(&self, glyph: GlyphId) -> Index<'a> {
        let Some(fda) = self.fd_array else { return self.subrs };
        // `FDSelect` is absent when every glyph shares one font DICT, which `CFF2` allows
        // and SF India uses. Falling back to the Top DICT's `Subrs` there finds nothing at
        // all: `CFF2` has no Top DICT `Private`.
        let fd = match self.fd_select {
            Some(at) => fd_select(self.data, at, glyph).unwrap_or(0),
            None => 0,
        };
        let Some(dict) = fda.get(usize::from(fd)) else { return self.subrs };
        let mut priv_at = None;
        dict_scan(dict, |key, ops| {
            if let (18, [size, off, ..]) = (key, ops) {
                priv_at = Some((*off as usize, *size as usize));
            }
        });
        priv_at
            .and_then(|(o, n)| private_subrs(self.data, o, n, self.wide))
            .unwrap_or_default()
    }

    /// The scalar of every variation region the `ItemVariationData` at `index` names.
    ///
    /// Returns the count and the scalars, in the subtable's own region order — which is
    /// the order `blend`'s deltas arrive in. A count of zero, or an instance where every
    /// scalar is zero, makes `blend` reduce to its base values.
    ///
    /// The store is **`CFF2`'s own**, at the offset its `vstore` operator gives, inside
    /// the `CFF2` table — *not* `GDEF`'s, which is what `HVAR` and `GPOS` vary through.
    /// Reading the wrong one is how this port first behaved: it produced no outline at
    /// all for every variable glyph, which reads as "`CFF2` unsupported" rather than as a
    /// bug, and the oracle in `swiftui-map/examples/cff_oracle` is what named it.
    /// Same type, same precomputed scalars, different table.
    fn region_scalars(&self, index: usize) -> Option<(usize, [f64; MAX_REGIONS])> {
        let scalars = self.store.as_ref()?.region_scalars(u16::try_from(index).ok()?)?;
        if scalars.len() > MAX_REGIONS {
            return None;
        }
        let mut out = [0.0f64; MAX_REGIONS];
        for (slot, s) in out.iter_mut().zip(&scalars) {
            *slot = f64::from(*s);
        }
        Some((scalars.len(), out))
    }

    /// Draw one glyph, in font units with `y` up.
    ///
    /// `None` if the charstring is missing, malformed, or uses an operator this does not
    /// implement — a wrong outline is worse than none, so an unknown operator stops the
    /// glyph rather than being skipped.
    pub(crate) fn draw(
        &self,
        glyph: GlyphId,
        sink: &mut dyn Outline,
    ) -> Option<()> {
        let code = self.char_strings.get(usize::from(glyph.0))?;
        let mut run = Run {
            cff: self,
            subrs: self.local(glyph),
            sink,
            xf: self.xf,
            st: [0.0; MAX_STACK],
            n: 0,
            x: 0.0,
            y: 0.0,
            open: false,
            stems: 0,
            // `CFF2` charstrings carry no leading width, so there is nothing to skip.
            width: self.wide,
            regions: None,
            vsindex: 0,
        };
        run.exec(code, 0)?;
        if run.open {
            run.sink.close();
        }
        Some(())
    }
}

/// A Private DICT's `Subrs`, whose offset is relative to the Private DICT itself rather
/// than to the table.
fn private_subrs<'a>(data: &'a [u8], off: usize, size: usize, wide: bool) -> Option<Index<'a>> {
    let dict = data.get(off..off.checked_add(size)?)?;
    let mut subrs = None;
    dict_scan(dict, |key, ops| {
        if let (19, [o, ..]) = (key, ops) {
            subrs = Some(*o as usize);
        }
    });
    let at = off.checked_add(subrs?)?;
    Index::parse_wide(data, at, wide).map(|(i, _)| i)
}

/// The Private DICT's `StdHW` and `StdVW`, which is where a `CFF` font keeps the stem
/// widths Core Graphics' font descriptor carries.
///
/// `FPFontGetStemH` answers these before it looks at any metadata: STIX General has
/// `StdHW = 66` and no `meta` table, and the live call returns 66.
fn private_stems(data: &[u8], off: usize, size: usize) -> Option<(f64, f64)> {
    let dict = data.get(off..off.checked_add(size)?)?;
    let (mut h, mut v) = (0.0, 0.0);
    dict_scan(dict, |key, ops| match (key, ops) {
        (10, [x, ..]) => h = *x,
        (11, [x, ..]) => v = *x,
        _ => {}
    });
    Some((h, v))
}

/// `FDSelect`: which font DICT a glyph belongs to.
fn fd_select(data: &[u8], at: usize, glyph: GlyphId) -> Option<u8> {
    let g = usize::from(glyph.0);
    match u8_at(data, at)? {
        // Format 0: one byte per glyph.
        0 => u8_at(data, at.checked_add(1)?.checked_add(g)?),
        // Format 3: sorted ranges of (first glyph, FD), with a sentinel past the end.
        3 => {
            let n = usize::from(u16_at(data, at + 1)?);
            let recs = at.checked_add(3)?;
            let sentinel = usize::from(u16_at(data, recs.checked_add(n.checked_mul(3)?)?)?);
            if g >= sentinel {
                return None;
            }
            // Binary search: Hiragino asks this 20,339 times.
            let (mut lo, mut hi) = (0usize, n);
            while lo + 1 < hi {
                let mid = (lo + hi) / 2;
                if usize::from(u16_at(data, recs + mid * 3)?) <= g {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            if usize::from(u16_at(data, recs + lo * 3)?) > g {
                return None;
            }
            u8_at(data, recs + lo * 3 + 2)
        }
        _ => None,
    }
}

/// One charstring in flight: the operand stack, the current point, and where the
/// subroutines are.
struct Run<'a, 'b> {
    cff: &'b Cff<'a>,
    subrs: Index<'a>,
    sink: &'b mut dyn Outline,
    xf: Affine,
    st: [f64; MAX_STACK],
    n: usize,
    x: f64,
    y: f64,
    /// Whether a contour is open. A `moveto` closes the previous one and so does the
    /// end of the glyph: a charstring never says `closepath`.
    open: bool,
    /// How many stem hints have been declared, which is how many bits a `hintmask`
    /// carries.
    stems: usize,
    /// Whether the leading width argument has been accounted for. It is optional and
    /// appears on whichever stack-clearing operator comes first. `CFF2` has no width, so
    /// this starts out already done there.
    width: bool,
    /// The scalar of each variation region the current `vsindex` names, which is what
    /// every `blend` operand's deltas are multiplied by. `None` until the first `blend`
    /// needs them; a count of zero means the instance is the default one.
    regions: Option<(usize, [f64; MAX_REGIONS])>,
    /// The `vsindex` in force; 0 until the operator says otherwise.
    vsindex: usize,
}

impl Run<'_, '_> {
    fn push(&mut self, v: f64) -> Option<()> {
        if self.n >= MAX_STACK {
            return None;
        }
        self.st[self.n] = v;
        self.n += 1;
        Some(())
    }

    /// Where the arguments start, once the optional leading width is accounted for.
    ///
    /// The width is present on the *first* stack-clearing operator and only if it has
    /// one argument more than it needs, so the caller passes what it needs. `None` means
    /// "an even number", which is what the stem and mask operators take.
    fn args(&mut self, nominal: Option<usize>) -> usize {
        if self.width {
            return 0;
        }
        self.width = true;
        let extra = match nominal {
            Some(k) => self.n > k,
            None => self.n % 2 == 1,
        };
        usize::from(extra)
    }

    fn move_to(&mut self, x: f64, y: f64) {
        if self.open {
            self.sink.close();
        }
        self.x = x;
        self.y = y;
        let (px, py) = self.xf.apply(x, y);
        self.sink.move_to(px, py);
        self.open = true;
    }

    fn line_to(&mut self, x: f64, y: f64) {
        self.x = x;
        self.y = y;
        let (px, py) = self.xf.apply(x, y);
        self.sink.line_to(px, py);
    }

    fn curve_to(&mut self, c1: (f64, f64), c2: (f64, f64), to: (f64, f64)) {
        self.x = to.0;
        self.y = to.1;
        let a = self.xf.apply(c1.0, c1.1);
        let b = self.xf.apply(c2.0, c2.1);
        let e = self.xf.apply(to.0, to.1);
        self.sink.cubic_to(a.0, a.1, b.0, b.1, e.0, e.1);
    }

    /// Run one charstring; `Ok(true)` means `endchar` was reached and the glyph is done.
    fn exec(&mut self, code: &[u8], depth: u8) -> Option<bool> {
        if depth > MAX_DEPTH {
            return None;
        }
        let mut i = 0usize;
        while i < code.len() {
            let b = code[i];
            i += 1;
            match b {
                // 32.. and 28 are operands, not operators.
                28 => {
                    let v = i16_at(code, i)?;
                    i += 2;
                    self.push(f64::from(v))?;
                }
                32..=246 => self.push(f64::from(i16::from(b) - 139))?,
                247..=250 => {
                    let lo = *code.get(i)?;
                    i += 1;
                    self.push(f64::from((i32::from(b) - 247) * 256 + i32::from(lo) + 108))?;
                }
                251..=254 => {
                    let lo = *code.get(i)?;
                    i += 1;
                    self.push(f64::from(-((i32::from(b) - 251) * 256) - i32::from(lo) - 108))?;
                }
                // 16.16, and in a charstring rather than a DICT: the only place a
                // non-integer coordinate can come from.
                255 => {
                    let v = i32_at(code, i)?;
                    i += 4;
                    self.push(f64::from(v) / 65536.0)?;
                }

                // Stem hints. The values are for the hinter, which this is not; only
                // the count matters, because `hintmask` has one bit per stem.
                1 | 3 | 18 | 23 => {
                    let s = self.args(None);
                    self.stems += (self.n - s) / 2;
                    self.n = 0;
                }
                // `hintmask` and `cntrmask` carry their mask inline, after the operator.
                // Any arguments still on the stack are implicit `vstem`s.
                19 | 20 => {
                    let s = self.args(None);
                    self.stems += (self.n - s) / 2;
                    self.n = 0;
                    let bytes = self.stems.div_ceil(8);
                    i = i.checked_add(bytes)?;
                    if i > code.len() {
                        return None;
                    }
                }

                // rmoveto, hmoveto, vmoveto.
                21 => {
                    let s = self.args(Some(2));
                    let (dx, dy) = (*self.st.get(s)?, *self.st.get(s + 1)?);
                    self.move_to(self.x + dx, self.y + dy);
                    self.n = 0;
                }
                22 | 4 => {
                    let s = self.args(Some(1));
                    let d = *self.st.get(s)?;
                    if b == 22 {
                        self.move_to(self.x + d, self.y);
                    } else {
                        self.move_to(self.x, self.y + d);
                    }
                    self.n = 0;
                }

                // rlineto: as many pairs as there are.
                5 => {
                    let mut k = 0;
                    while k + 2 <= self.n {
                        self.line_to(self.x + self.st[k], self.y + self.st[k + 1]);
                        k += 2;
                    }
                    self.n = 0;
                }
                // hlineto, vlineto: one coordinate each, alternating direction.
                6 | 7 => {
                    let mut horizontal = b == 6;
                    for k in 0..self.n {
                        let d = self.st[k];
                        if horizontal {
                            self.line_to(self.x + d, self.y);
                        } else {
                            self.line_to(self.x, self.y + d);
                        }
                        horizontal = !horizontal;
                    }
                    self.n = 0;
                }

                // rrcurveto: as many sextuples as there are.
                8 => {
                    let mut k = 0;
                    while k + 6 <= self.n {
                        self.rel_curve(k);
                        k += 6;
                    }
                    self.n = 0;
                }
                // rcurveline: curves, then one line from the last pair.
                24 => {
                    let mut k = 0;
                    while self.n - k >= 8 {
                        self.rel_curve(k);
                        k += 6;
                    }
                    if k + 2 <= self.n {
                        self.line_to(self.x + self.st[k], self.y + self.st[k + 1]);
                    }
                    self.n = 0;
                }
                // rlinecurve: lines, then one curve from the last sextuple.
                25 => {
                    let mut k = 0;
                    while self.n - k >= 8 {
                        self.line_to(self.x + self.st[k], self.y + self.st[k + 1]);
                        k += 2;
                    }
                    if k + 6 <= self.n {
                        self.rel_curve(k);
                    }
                    self.n = 0;
                }
                // vvcurveto, hhcurveto: a run in one direction, with an optional
                // one-off cross-axis delta on the *first* curve only.
                26 | 27 => {
                    let mut k = 0;
                    let mut off = 0.0;
                    if (self.n - k) % 4 == 1 {
                        off = self.st[0];
                        k = 1;
                    }
                    while k + 4 <= self.n {
                        let (a, b2, c, d) = (self.st[k], self.st[k + 1], self.st[k + 2], self.st[k + 3]);
                        let (c1, c2, to) = if b == 26 {
                            let c1 = (self.x + off, self.y + a);
                            let c2 = (c1.0 + b2, c1.1 + c);
                            (c1, c2, (c2.0, c2.1 + d))
                        } else {
                            let c1 = (self.x + a, self.y + off);
                            let c2 = (c1.0 + b2, c1.1 + c);
                            (c1, c2, (c2.0 + d, c2.1))
                        };
                        off = 0.0;
                        self.curve_to(c1, c2, to);
                        k += 4;
                    }
                    self.n = 0;
                }
                // vhcurveto, hvcurveto: four arguments per curve, the starting
                // direction alternating, and a fifth on the last group if it is there.
                30 | 31 => {
                    let mut horizontal = b == 31;
                    let mut k = 0;
                    while self.n - k >= 4 {
                        let last = self.n - k == 5;
                        let (a, b2, c, d) = (self.st[k], self.st[k + 1], self.st[k + 2], self.st[k + 3]);
                        let e = if last { self.st[k + 4] } else { 0.0 };
                        let (c1, c2, to) = if horizontal {
                            let c1 = (self.x + a, self.y);
                            let c2 = (c1.0 + b2, c1.1 + c);
                            (c1, c2, (c2.0 + e, c2.1 + d))
                        } else {
                            let c1 = (self.x, self.y + a);
                            let c2 = (c1.0 + b2, c1.1 + c);
                            (c1, c2, (c2.0 + d, c2.1 + e))
                        };
                        self.curve_to(c1, c2, to);
                        horizontal = !horizontal;
                        k += 4;
                    }
                    self.n = 0;
                }

                // callsubr, callgsubr: the argument is biased by the array's size.
                10 | 29 => {
                    if self.n == 0 {
                        return None;
                    }
                    self.n -= 1;
                    let v = self.st[self.n];
                    let arr = if b == 10 { self.subrs } else { self.cff.gsubrs };
                    #[allow(clippy::cast_possible_truncation)]
                    let idx = v as i32 + arr.bias();
                    if idx < 0 {
                        return None;
                    }
                    let sub = arr.get(idx as usize)?;
                    if self.exec(sub, depth + 1)? {
                        return Some(true);
                    }
                }
                11 => return Some(false),
                // `vsindex` and `blend`, both `CFF2` only.
                15 if self.cff.wide => {
                    if self.n == 0 {
                        return None;
                    }
                    self.n -= 1;
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let v = self.st[self.n] as usize;
                    self.vsindex = v;
                    self.regions = None;
                }
                // `blend` replaces `k` operands with `k` blended values. Above the `k`
                // bases sit `k * regions` deltas, grouped by operand — all of the first
                // value's, then the second's — which is the order
                // `ItemVariationStore::ValueFromBlendRequest` walks them in, and its
                // accumulation is one fused multiply-add per region, in double:
                //
                //     ldr    d1, [x2, x10, lsl #3]     ; the region's scalar
                //     ldr    d2, [x4], #0x8            ; the delta
                //     fmadd  d0, d1, d2, d0
                16 if self.cff.wide => {
                    if self.n == 0 {
                        return None;
                    }
                    self.n -= 1;
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let k = self.st[self.n] as usize;
                    let (regions, scalars) = match self.regions {
                        Some(r) => r,
                        None => {
                            let r = self.cff.region_scalars(self.vsindex)?;
                            self.regions = Some(r);
                            r
                        }
                    };
                    let deltas = k.checked_mul(regions)?;
                    if deltas > self.n || k > self.n - deltas {
                        return None;
                    }
                    let base = self.n - deltas - k;
                    for i in 0..k {
                        let mut v = self.st[base + i];
                        for (j, s) in scalars.iter().enumerate().take(regions) {
                            // `mul_add` is one rounding, which is what a blend needs when a delta and
                            // its scalar nearly cancel. rswift emulates it with Dekker's exact
                            // product because `core` has no `mul_add` in `no_std`; here the
                            // hardware instruction is reachable and is the same answer.
                            v = f64::mul_add(*s, self.st[base + k + i * regions + j], v);
                        }
                        self.st[base + i] = v;
                    }
                    self.n = base + k;
                }
                14 if !self.cff.wide => {
                    // `endchar` with four arguments left is `seac`: an accented glyph
                    // built from two others. Nothing on this machine uses it, and
                    // guessing would draw the base without its accent.
                    let s = self.args(Some(0));
                    if self.n - s >= 4 {
                        return None;
                    }
                    return Some(true);
                }

                12 => {
                    let b2 = *code.get(i)?;
                    i += 1;
                    match b2 {
                        // flex: two curves that a hinter may flatten into one line. The
                        // final argument is the flex depth, which only the hinter reads.
                        35 if self.n >= 13 => {
                            self.rel_curve(0);
                            self.rel_curve(6);
                            self.n = 0;
                        }
                        // hflex: a flex whose ends share the starting y, so six of the
                        // twelve deltas are implied.
                        34 if self.n >= 7 => {
                            let y0 = self.y;
                            let a = self.st;
                            let c1 = (self.x + a[0], y0);
                            let c2 = (c1.0 + a[1], c1.1 + a[2]);
                            self.curve_to(c1, c2, (c2.0 + a[3], c2.1));
                            let c3 = (self.x + a[4], self.y);
                            let c4 = (c3.0 + a[5], y0);
                            self.curve_to(c3, c4, (c4.0 + a[6], y0));
                            self.n = 0;
                        }
                        // hflex1: as hflex, but the first curve's control points move
                        // in y as well; only the final point returns to the start.
                        36 if self.n >= 9 => {
                            let y0 = self.y;
                            let a = self.st;
                            let c1 = (self.x + a[0], self.y + a[1]);
                            let c2 = (c1.0 + a[2], c1.1 + a[3]);
                            self.curve_to(c1, c2, (c2.0 + a[4], c2.1));
                            let c3 = (self.x + a[5], self.y);
                            let c4 = (c3.0 + a[6], c3.1 + a[7]);
                            self.curve_to(c3, c4, (c4.0 + a[8], y0));
                            self.n = 0;
                        }
                        // flex1: the last point is one delta, and which axis it is on
                        // depends on which way the flex travelled furthest.
                        37 if self.n >= 11 => {
                            let (x0, y0) = (self.x, self.y);
                            let a = self.st;
                            let dx = a[0] + a[2] + a[4] + a[6] + a[8];
                            let dy = a[1] + a[3] + a[5] + a[7] + a[9];
                            self.rel_curve(0);
                            let c3 = (self.x + a[6], self.y + a[7]);
                            let c4 = (c3.0 + a[8], c3.1 + a[9]);
                            let to = if dx.abs() > dy.abs() {
                                (c4.0 + a[10], y0)
                            } else {
                                (x0, c4.1 + a[10])
                            };
                            self.curve_to(c3, c4, to);
                            self.n = 0;
                        }
                        // The arithmetic and storage operators. No font on this machine
                        // uses one, and a wrong outline is worse than no outline.
                        _ => return None,
                    }
                }

                // 9, 13, 15, 16, 17 are reserved.
                _ => return None,
            }
        }
        Some(false)
    }

    /// One `rrcurveto`-shaped curve from six deltas at `k`.
    fn rel_curve(&mut self, k: usize) {
        let a = self.st;
        let c1 = (self.x + a[k], self.y + a[k + 1]);
        let c2 = (c1.0 + a[k + 2], c1.1 + a[k + 3]);
        self.curve_to(c1, c2, (c2.0 + a[k + 4], c2.1 + a[k + 5]));
    }
}
