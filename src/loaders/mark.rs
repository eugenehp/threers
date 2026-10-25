//! `GPOS` mark attachment: types 4, 5 and 6, the `mark` and `mkmk` features.
//!
//! This places a mark glyph at an anchor on the glyph it belongs to, and it is **not**
//! only about combining characters. `SFArabic` shapes a letter into a skeleton *plus its
//! dots as separate mark glyphs*, so `كتاب` — four letters, nothing combining in the
//! string at all — has a mark in it, and without this the dot sits at its skeleton's
//! origin instead of 0.3857pt to the right of it. Measured in `rswift` against
//! `CTRunGetPositions`: every other glyph of that line was already exact to the bit.
//!
//! Ported from `rswift`'s `swiftui-font`, where it took the pens from 7 of 13
//! right-to-left cases to 12 of 13 and made `كتاب` pixel-exact against SwiftUI at 24pt.
//!
//! # What a mark is, and why `GDEF` is the only answer
//!
//! A dot the shaper decomposed a letter into has **no codepoint** — nothing in `cmap`
//! reaches it — so no property of a character can identify it. `GDEF`'s `GlyphClassDef`
//! says so, by putting it in class 3. [`Marks::attach`] reads that map **once** per run
//! rather than once per glyph, which is why there is no public "is this a mark".
//!
//! # Every script, unlike `GPOS` kerning
//!
//! [`super::gpos::feature_lookups`] prefers the `latn` script, which is right for kerning
//! and wrong here: an Arabic font registers `mark` under `arab`, and asking for `latn`
//! finds nothing. So this scans the `FeatureList` directly and takes every `mark` and
//! `mkmk` feature whatever script registers it. Anchors are per glyph pair and do not
//! conflict between scripts the way `GSUB`'s alternates do.
//!
//! `SFArabic` has **no `curs`**, so cursive attachment is absent here too: its census is
//! `kern` 2 pair, `mark` 5 mark-to-base and 2 mark-to-ligature, `mkmk` 2 mark-to-mark.
//! Adding what no font in reach uses would be unmeasurable.

use super::ttf_var::ItemVariationStore;

fn u16_at(b: &[u8], p: usize) -> Option<u16> {
    Some(u16::from_be_bytes([*b.get(p)?, *b.get(p + 1)?]))
}

fn i16_at(b: &[u8], p: usize) -> Option<i16> {
    u16_at(b, p).map(|v| v as i16)
}

/// `GDEF`'s glyph class for a mark. 1 is a base, 2 a ligature, 4 a component.
const CLASS_MARK: u16 = 3;

/// Where one glyph of a run goes, beyond its own advance.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct Attachment {
    /// The run index of the glyph this one hangs off, if it hangs off anything.
    pub to: Option<u16>,
    /// How far right of that glyph's origin, in font units.
    pub dx: f32,
    /// How far above its baseline, in font units.
    pub dy: f32,
}

/// `GPOS`'s mark lookups and `GDEF`'s classes, resolved at one location.
pub(crate) struct Marks<'a> {
    gpos: &'a [u8],
    gdef: &'a [u8],
    /// `mark` lookups then `mkmk` ones, in that order — a mark attaches to a base before
    /// it attaches to another mark, and the later answer wins.
    lookups: Vec<u16>,
    /// Where the `mkmk` lookups begin within `lookups`.
    mkmk_at: usize,
    /// `GDEF`'s `GlyphClassDef`, absolute within `gdef`.
    classes: Option<usize>,
    store: Option<&'a ItemVariationStore<'a>>,
}

impl<'a> Marks<'a> {
    /// `None` when the font has no mark positioning, which is most Latin fonts.
    pub(crate) fn parse(
        gpos: &'a [u8],
        gdef: &'a [u8],
        store: Option<&'a ItemVariationStore<'a>>,
    ) -> Option<Self> {
        if u16_at(gpos, 0)? != 1 {
            return None;
        }
        let features = usize::from(u16_at(gpos, 6)?);
        let count = usize::from(u16_at(gpos, features)?);
        // Two passes, so every `mark` lookup precedes every `mkmk` one.
        let mut lookups = Vec::new();
        let mut mkmk_at = 0usize;
        for tag in [b"mark", b"mkmk"] {
            if tag == b"mkmk" {
                mkmk_at = lookups.len();
            }
            for i in 0..count {
                let rec = features + 2 + 6 * i;
                if gpos.get(rec..rec + 4) != Some(&tag[..]) {
                    continue;
                }
                let f = features + usize::from(u16_at(gpos, rec + 4)?);
                for k in 0..usize::from(u16_at(gpos, f + 2)?) {
                    if let Some(l) = u16_at(gpos, f + 4 + 2 * k) {
                        lookups.push(l);
                    }
                }
            }
        }
        if lookups.is_empty() {
            return None;
        }
        // `GlyphClassDef` is `GDEF`'s first `Offset16`; 0 means absent, in which case
        // nothing is a mark and the lookups never fire.
        let classes = match u16_at(gdef, 4) {
            Some(0) | None => None,
            Some(o) => Some(usize::from(o)),
        };
        Some(Self { gpos, gdef, lookups, mkmk_at, classes, store })
    }

    /// Resolve attachment for a run in **logical** order, one entry per glyph.
    ///
    /// Logical order matters: a mark attaches to what precedes it in the text, which
    /// after a right-to-left reversal is what follows it on the screen.
    pub(crate) fn attach(&self, glyphs: &[u16]) -> Vec<Attachment> {
        // One class lookup for the whole run rather than one per glyph: `parse_class_def`
        // builds a map, and asking it per glyph would rebuild it per glyph.
        let classes = self
            .classes
            .and_then(|at| super::gpos::parse_class_def(self.gdef, at))
            .unwrap_or_default();
        let is_mark = |g: u16| classes.get(&g).copied() == Some(CLASS_MARK);
        let mut out = vec![Attachment::default(); glyphs.len()];
        for (i, g) in glyphs.iter().enumerate() {
            if !is_mark(*g) {
                continue;
            }
            // `mark` attaches to the nearest preceding glyph that is not a mark; `mkmk`
            // to the immediately preceding mark, and overwrites — which is what stacks a
            // second mark on the first rather than on the base both of them cover.
            if let Some(b) = (0..i).rev().find(|j| !is_mark(glyphs[*j])) {
                if let Some((dx, dy)) = self.anchor_pair(*g, glyphs[b], 0, self.mkmk_at) {
                    out[i] = Attachment { to: u16::try_from(b).ok(), dx, dy };
                }
            }
            if i > 0 && is_mark(glyphs[i - 1]) {
                if let Some((dx, dy)) =
                    self.anchor_pair(*g, glyphs[i - 1], self.mkmk_at, self.lookups.len())
                {
                    out[i] = Attachment { to: u16::try_from(i - 1).ok(), dx, dy };
                }
            }
        }
        out
    }

    /// The anchor delta between `mark` and `to`, over the lookups in a range.
    ///
    /// Within a lookup the subtables are alternatives and the first lookup with an answer
    /// wins: a mark is attached once, not several times.
    fn anchor_pair(
        &self,
        mark: u16,
        to: u16,
        from: usize,
        until: usize,
    ) -> Option<(f32, f32)> {
        for index in self.lookups.get(from..until.min(self.lookups.len()))? {
            // 9 is `GPOS`'s extension type.
            for (kind, sub) in super::gpos::lookup_subtables(self.gpos, 0, *index, 9)? {
                if let Some(d) = self.mark_pos(kind, sub, mark, to) {
                    return Some(d);
                }
            }
        }
        None
    }

    /// One mark-positioning subtable, types 4, 5 and 6.
    ///
    /// All three have the same shape — a mark coverage, a second coverage, a class count,
    /// a `MarkArray` and an array of anchors per covered second glyph — and differ only in
    /// what the second array is indexed by. Type 5's rows are *ligature components*, and
    /// since this does not track which component a mark belongs to it takes the **last**,
    /// which is where a mark on a lam-alef belongs.
    fn mark_pos(&self, kind: u16, sub: usize, mark: u16, to: u16) -> Option<(f32, f32)> {
        if !(4..=6).contains(&kind) || u16_at(self.gpos, sub)? != 1 {
            return None;
        }
        let b = self.gpos;
        let mark_cov = super::gpos::parse_coverage(b, sub + usize::from(u16_at(b, sub + 2)?))?;
        let to_cov = super::gpos::parse_coverage(b, sub + usize::from(u16_at(b, sub + 4)?))?;
        let mark_index = usize::from(mark_cov.get(&mark).copied()?);
        let to_index = usize::from(to_cov.get(&to).copied()?);
        let class_count = usize::from(u16_at(b, sub + 6)?);
        let mark_array = sub + usize::from(u16_at(b, sub + 8)?);
        let second = sub + usize::from(u16_at(b, sub + 10)?);

        // `MarkArray`: a count, then (markClass, markAnchorOffset) per covered mark.
        if mark_index >= usize::from(u16_at(b, mark_array)?) {
            return None;
        }
        let rec = mark_array + 2 + mark_index * 4;
        let class = usize::from(u16_at(b, rec)?);
        if class >= class_count {
            return None;
        }
        let mark_anchor = usize::from(u16_at(b, rec + 2)?);
        if mark_anchor == 0 {
            return None;
        }
        let (mx, my) = self.anchor(mark_array + mark_anchor)?;

        // The second array: one row per covered glyph, `class_count` anchor offsets wide.
        // For a ligature the row is itself a list of components, each that wide.
        if to_index >= usize::from(u16_at(b, second)?) {
            return None;
        }
        let row = if kind == 5 {
            let lig = second + usize::from(u16_at(b, second + 2 + to_index * 2)?);
            let components = usize::from(u16_at(b, lig)?);
            if components == 0 {
                return None;
            }
            lig + 2 + (components - 1) * class_count * 2
        } else {
            second + 2 + to_index * class_count * 2
        };
        let to_anchor = usize::from(u16_at(b, row + class * 2)?);
        if to_anchor == 0 {
            return None;
        }
        let (bx, by) = self.anchor(second + to_anchor)?;
        Some((bx - mx, by - my))
    }

    /// One `Anchor` table, in font units.
    ///
    /// Format 2 adds a contour point for hinted alignment, which this ignores — the
    /// coordinates are still there and are what an unhinted renderer uses. Format 3 adds
    /// device tables, and in a variable font those are `VariationIndex` into the same
    /// `GDEF` store a kern value varies through.
    fn anchor(&self, at: usize) -> Option<(f32, f32)> {
        let b = self.gpos;
        let format = u16_at(b, at)?;
        let mut x = f32::from(i16_at(b, at + 2)?);
        let mut y = f32::from(i16_at(b, at + 4)?);
        if format == 3 {
            for (slot, v) in [(6usize, &mut x), (8, &mut y)] {
                let device = usize::from(u16_at(b, at + slot)?);
                if device == 0 {
                    continue;
                }
                let d = at + device;
                // In a `VariationIndex` the delta format is 0x8000 and the two leading
                // fields are the delta-set indices rather than a ppem range.
                if u16_at(b, d + 4)? != 0x8000 {
                    continue;
                }
                let (outer, inner) = (u16_at(b, d)?, u16_at(b, d + 2)?);
                if let Some(s) = self.store {
                    *v += s.delta(outer, inner);
                }
            }
        }
        Some((x, y))
    }
}
