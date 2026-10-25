//! `GSUB`: the substitutions that turn characters into the glyphs actually drawn.
//!
//! # What is covered
//!
//! Lookup types **1** (single), **2** (multiple), **3** (alternate), **4** (ligature),
//! **5** (contextual), **6** (chaining contextual) and **7** (extension). Type 8, reverse
//! chaining, is not covered; a lookup of an unhandled type is skipped rather than guessed
//! at, so a font that needs one loses that feature and nothing else.
//!
//! **Type 6 is not optional for Arabic.** `SFArabic` registers eighteen lookups under
//! `rclt` — required contextual alternates — and sixteen of them are chaining contextual.
//! That is where the real shaping happens: the four joining features only pick a letter's
//! base form, and `rclt` then rewrites it according to its neighbours. Without type 6 this
//! shaper agreed with Core Text on 5 of 16 Arabic words; the four joining features alone
//! are not Arabic shaping.
//!
//! # A buffer, not a glyph at a time
//!
//! A contextual lookup matches a run and then applies **nested lookups** at positions
//! inside it, and a nested lookup may be a ligature that shortens the run. So
//! substitution works on a [`Buffer`] rather than answering questions about single
//! glyphs, and [`Gsub::apply_feature`] walks it the way the specification describes.
//!
//! # The script has to be selected
//!
//! A feature tag appears **once per script**, and the lookup sets differ. `SFArabic`
//! registers `rlig` twice: under `arab` with three ligature lookups, and under `DFLT`
//! with those three plus a single substitution. Taking the union applies that extra
//! lookup, which rewrites the lam-alef ligature 965 into 1414 — a different glyph, drawn
//! correctly, in the wrong place for this script. So [`Gsub::lookups_for`] takes a script
//! tag and reads the `ScriptList` for it, falling back to `DFLT`.
//!
//! Language systems are **not** selected: a script's default is used, never a
//! language-specific one. No font in this stack's cascade distinguishes them, and
//! guessing a language from the characters is a different problem.


use super::read::{tag_at, u16_at, u32_at};

/// A run of glyphs being substituted, and which character each came from.
///
/// The cluster travels with the glyph: a ligature keeps its first component's cluster and
/// a decomposition gives all its outputs the same one, so a caller can still map a glyph
/// back to the text it came from after any amount of substitution.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Buffer {
    pub glyphs: Vec<u16>,
    pub clusters: Vec<usize>,
}

impl Buffer {
    /// A buffer of `(glyph, cluster)` pairs.
    #[must_use]
    pub fn new(pairs: &[(u16, usize)]) -> Self {
        Self {
            glyphs: pairs.iter().map(|(g, _)| *g).collect(),
            clusters: pairs.iter().map(|(_, c)| *c).collect(),
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.glyphs.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.glyphs.is_empty()
    }

    /// Replace `at..at + take` with `with`, all carrying `at`'s cluster.
    fn splice(&mut self, at: usize, take: usize, with: &[u16]) {
        let cluster = self.clusters.get(at).copied().unwrap_or(0);
        let end = (at + take).min(self.glyphs.len());
        self.glyphs.splice(at..end, with.iter().copied());
        self.clusters.splice(at..end, core::iter::repeat_n(cluster, with.len()));
    }
}

/// How deep nested contextual lookups are followed.
///
/// Real fonts nest two or three deep. The cap stops a font whose lookups refer to each
/// other from recursing forever, which is a thing a malformed font can express.
const MAX_DEPTH: usize = 8;

/// A `GSUB` table, with its lookups resolved.
#[derive(Clone, Copy, Debug)]
pub struct Gsub<'a> {
    data: &'a [u8],
    script_list: usize,
    feature_list: usize,
    lookup_list: usize,
}

impl<'a> Gsub<'a> {
    /// Parse the header. `None` if it is not a version this understands.
    #[must_use]
    pub fn parse(data: &'a [u8]) -> Option<Self> {
        if u16_at(data, 0)? != 1 {
            return None;
        }
        Some(Self {
            data,
            script_list: usize::from(u16_at(data, 4)?),
            feature_list: usize::from(u16_at(data, 6)?),
            lookup_list: usize::from(u16_at(data, 8)?),
        })
    }

    /// The feature indices `script`'s default language system lists.
    ///
    /// `None` when the script is absent, which the caller treats as "try `DFLT`".
    fn features_of(&self, script: &[u8; 4]) -> Option<Vec<u16>> {
        let count = usize::from(u16_at(self.data, self.script_list)?);
        for i in 0..count {
            let rec = self.script_list + 2 + 6 * i;
            if tag_at(self.data, rec)? != *script {
                continue;
            }
            let at = self.script_list + usize::from(u16_at(self.data, rec + 4)?);
            let default = usize::from(u16_at(self.data, at)?);
            if default == 0 {
                return Some(Vec::new());
            }
            let lang = at + default;
            let n = usize::from(u16_at(self.data, lang + 4)?);
            let mut out = Vec::with_capacity(n + 1);
            // A required feature applies whether or not it is in the list.
            let required = u16_at(self.data, lang + 2)?;
            if required != 0xFFFF {
                out.push(required);
            }
            for j in 0..n {
                out.push(u16_at(self.data, lang + 6 + 2 * j)?);
            }
            return Some(out);
        }
        None
    }

    /// Every lookup index registered under `tag` for `script`, in feature-list order.
    ///
    /// Restricted to the features that script's default language system lists, because a
    /// tag means different lookups under different scripts — see the module docs.
    #[must_use]
    pub fn lookups_for_script(&self, script: &[u8; 4], tag: &[u8; 4]) -> Vec<u16> {
        // A script the table does not list falls back to `DFLT`; if that is absent too,
        // **nothing** applies. Allowing everything in that case would reintroduce exactly
        // the bug script selection exists to prevent, on the fonts least likely to have
        // been tested against.
        let allowed = self
            .features_of(script)
            .or_else(|| self.features_of(b"DFLT"))
            .unwrap_or_default();
        let mut out = Vec::new();
        let Some(count) = u16_at(self.data, self.feature_list) else { return out };
        for i in 0..usize::from(count) {
            let Some(rec) = self.feature_list.checked_add(2 + 6 * i) else { break };
            if tag_at(self.data, rec) != Some(*tag) {
                continue;
            }
            if !allowed.contains(&u16::try_from(i).unwrap_or(u16::MAX)) {
                continue;
            }
            let Some(off) = u16_at(self.data, rec + 4) else { continue };
            let feature = self.feature_list + usize::from(off);
            let Some(n) = u16_at(self.data, feature + 2) else { continue };
            for j in 0..usize::from(n) {
                if let Some(idx) = u16_at(self.data, feature + 4 + 2 * j) {
                    if !out.contains(&idx) {
                        out.push(idx);
                    }
                }
            }
        }
        out
    }

    /// Whether `script` registers `tag`.
    #[must_use]
    pub fn has(&self, script: &[u8; 4], tag: &[u8; 4]) -> bool {
        !self.lookups_for_script(script, tag).is_empty()
    }

    /// One lookup's type and subtable offsets, following an extension if present.
    #[must_use]
    pub fn subtables(&self, index: u16) -> Option<(u16, Vec<usize>)> {
        let total = u16_at(self.data, self.lookup_list)?;
        if index >= total {
            return None;
        }
        let off = u16_at(self.data, self.lookup_list + 2 + 2 * usize::from(index))?;
        let lookup = self.lookup_list + usize::from(off);
        let kind = u16_at(self.data, lookup)?;
        let n = usize::from(u16_at(self.data, lookup + 4)?);
        let mut out = Vec::with_capacity(n);
        let mut real = kind;
        for j in 0..n {
            let s = lookup + usize::from(u16_at(self.data, lookup + 6 + 2 * j)?);
            if kind == 7 {
                // An extension subtable names the real type and reaches it with a
                // 32-bit offset from its own start, which is how a large table gets past
                // the 16-bit limit.
                real = u16_at(self.data, s + 2)?;
                out.push(s + u32_at(self.data, s + 4)? as usize);
            } else {
                out.push(s);
            }
        }
        Some((real, out))
    }

    /// Apply every lookup of `tag` across `buf`, left to right.
    ///
    /// One pass per lookup, which is what the specification describes: a feature's
    /// lookups are applied in lookup-list order, and each walks the buffer once.
    pub fn apply_feature(&self, script: &[u8; 4], tag: &[u8; 4], buf: &mut Buffer) {
        for index in self.lookups_for_script(script, tag) {
            let mut at = 0;
            while at < buf.len() {
                if self.apply_at(index, buf, at, 0) {
                    // A substitution may have shortened the buffer; carry on from here
                    // rather than assuming a width.
                    at += 1;
                } else {
                    at += 1;
                }
            }
        }
    }

    /// Apply one lookup at one position. `true` if it did something.
    fn apply_at(&self, index: u16, buf: &mut Buffer, at: usize, depth: usize) -> bool {
        if depth > MAX_DEPTH || at >= buf.len() {
            return false;
        }
        let Some((kind, subs)) = self.subtables(index) else { return false };
        for sub in subs {
            let did = match kind {
                1 => self.single(sub, buf.glyphs[at]).map(|g| {
                    buf.glyphs[at] = g;
                    true
                }),
                2 => self.multiple(sub, buf.glyphs[at]).map(|many| {
                    buf.splice(at, 1, &many);
                    true
                }),
                // Alternate: the first alternate, since there is no caller asking for
                // the third stylistic variant of anything here.
                3 => self.alternate(sub, buf.glyphs[at]).map(|g| {
                    buf.glyphs[at] = g;
                    true
                }),
                4 => self.ligature_in(sub, buf.glyphs[at], &buf.glyphs[at..]).map(|(g, n)| {
                    buf.splice(at, n, &[g]);
                    true
                }),
                5 | 6 => self.contextual(sub, buf, at, kind, depth),
                _ => None,
            };
            if did == Some(true) {
                return true;
            }
        }
        false
    }

    /// A contextual subtable in **format 1**: rules as explicit glyph sequences.
    ///
    /// The first glyph is matched by coverage and its index selects a rule set; each rule
    /// then spells out the rest of the input, and for a chaining rule the backtrack and
    /// lookahead too. The backtrack is stored **in reverse** — nearest glyph first —
    /// which reads backwards from the input's start.
    fn contextual_1(
        &self,
        sub: usize,
        buf: &mut Buffer,
        at: usize,
        kind: u16,
        depth: usize,
    ) -> Option<bool> {
        let coverage = sub + usize::from(u16_at(self.data, sub + 2)?);
        let index = usize::from(coverage_index(self.data, coverage, buf.glyphs[at])?);
        let sets = usize::from(u16_at(self.data, sub + 4)?);
        if index >= sets {
            return None;
        }
        let set = sub + usize::from(u16_at(self.data, sub + 6 + 2 * index)?);
        let rules = usize::from(u16_at(self.data, set)?);
        for r in 0..rules {
            let rule = set + usize::from(u16_at(self.data, set + 2 + 2 * r)?);
            let mut p = rule;
            let backtrack = if kind == 6 {
                let n = usize::from(u16_at(self.data, p)?);
                p += 2;
                let g: Vec<u16> = (0..n)
                    .map(|i| u16_at(self.data, p + 2 * i))
                    .collect::<Option<Vec<u16>>>()?;
                p += 2 * n;
                g
            } else {
                Vec::new()
            };
            let input_n = usize::from(u16_at(self.data, p)?);
            p += 2;
            // The input array omits the first glyph, which coverage already matched.
            let input: Vec<u16> = (0..input_n.saturating_sub(1))
                .map(|i| u16_at(self.data, p + 2 * i))
                .collect::<Option<Vec<u16>>>()?;
            p += 2 * input.len();
            let lookahead = if kind == 6 {
                let n = usize::from(u16_at(self.data, p)?);
                p += 2;
                let g: Vec<u16> = (0..n)
                    .map(|i| u16_at(self.data, p + 2 * i))
                    .collect::<Option<Vec<u16>>>()?;
                p += 2 * n;
                g
            } else {
                Vec::new()
            };
            let record_n = usize::from(u16_at(self.data, p)?);
            p += 2;

            if input_n == 0 || at + input_n > buf.len() || backtrack.len() > at {
                continue;
            }
            if !input.iter().enumerate().all(|(k, g)| buf.glyphs[at + 1 + k] == *g) {
                continue;
            }
            if !backtrack.iter().enumerate().all(|(k, g)| buf.glyphs[at - 1 - k] == *g) {
                continue;
            }
            let after = at + input_n;
            if after + lookahead.len() > buf.len() {
                continue;
            }
            if !lookahead.iter().enumerate().all(|(k, g)| buf.glyphs[after + k] == *g) {
                continue;
            }
            let mut did = false;
            for i in 0..record_n {
                let rec = p + 4 * i;
                let seq = usize::from(u16_at(self.data, rec)?);
                let lookup = u16_at(self.data, rec + 2)?;
                if self.apply_at(lookup, buf, at + seq, depth + 1) {
                    did = true;
                }
            }
            return Some(did);
        }
        None
    }

    /// One alternate-substitution subtable: the first alternate.
    fn alternate(&self, sub: usize, glyph: u16) -> Option<u16> {
        if u16_at(self.data, sub)? != 1 {
            return None;
        }
        let coverage = sub + usize::from(u16_at(self.data, sub + 2)?);
        let index = usize::from(coverage_index(self.data, coverage, glyph)?);
        let count = usize::from(u16_at(self.data, sub + 4)?);
        if index >= count {
            return None;
        }
        let set = sub + usize::from(u16_at(self.data, sub + 6 + 2 * index)?);
        if u16_at(self.data, set)? == 0 {
            return None;
        }
        u16_at(self.data, set + 2)
    }

    /// A glyph's class from a `ClassDef`, which is **0** for anything unlisted.
    ///
    /// Format 1 is a dense array from a start glyph; format 2 is a range list. Defaulting
    /// to 0 is not a fallback — class 0 is a real class that rules match on, and it is how
    /// a rule says "anything else".
    fn class_of(&self, at: usize, glyph: u16) -> u16 {
        let one = |()| -> Option<u16> {
            let start = u16_at(self.data, at + 2)?;
            let n = u16_at(self.data, at + 4)?;
            let i = glyph.checked_sub(start)?;
            if i >= n {
                return None;
            }
            u16_at(self.data, at + 6 + 2 * usize::from(i))
        };
        let two = |()| -> Option<u16> {
            let n = usize::from(u16_at(self.data, at + 2)?);
            for i in 0..n {
                let rec = at + 4 + 6 * i;
                let (first, last) = (u16_at(self.data, rec)?, u16_at(self.data, rec + 2)?);
                if glyph >= first && glyph <= last {
                    return u16_at(self.data, rec + 4);
                }
            }
            None
        };
        match u16_at(self.data, at) {
            Some(1) => one(()).unwrap_or(0),
            Some(2) => two(()).unwrap_or(0),
            _ => 0,
        }
    }

    /// A contextual subtable in **format 2**: rules as class sequences.
    ///
    /// Two things differ from format 1 beyond the comparison itself, and both are silent
    /// if got wrong: the rule set is chosen by the first glyph's **class**, not by its
    /// coverage index, and a chaining subtable has **three** class definitions — the
    /// backtrack, input and lookahead positions are classified separately, so the same
    /// glyph can be class 2 as a predecessor and class 7 as input.
    fn contextual_2(
        &self,
        sub: usize,
        buf: &mut Buffer,
        at: usize,
        kind: u16,
        depth: usize,
    ) -> Option<bool> {
        let coverage = sub + usize::from(u16_at(self.data, sub + 2)?);
        coverage_index(self.data, coverage, buf.glyphs[at])?;
        let (back_def, input_def, ahead_def, head) = if kind == 6 {
            (
                sub + usize::from(u16_at(self.data, sub + 4)?),
                sub + usize::from(u16_at(self.data, sub + 6)?),
                sub + usize::from(u16_at(self.data, sub + 8)?),
                12,
            )
        } else {
            let d = sub + usize::from(u16_at(self.data, sub + 4)?);
            (d, d, d, 8)
        };
        let sets = usize::from(u16_at(self.data, sub + head - 2)?);
        let index = usize::from(self.class_of(input_def, buf.glyphs[at]));
        if index >= sets {
            return None;
        }
        let off = usize::from(u16_at(self.data, sub + head + 2 * index)?);
        if off == 0 {
            return None; // A class with no rules is a null offset.
        }
        let set = sub + off;
        let rules = usize::from(u16_at(self.data, set)?);
        for r in 0..rules {
            let rule = set + usize::from(u16_at(self.data, set + 2 + 2 * r)?);
            let mut p = rule;
            let read = |p: usize, n: usize| -> Option<Vec<u16>> {
                (0..n).map(|i| u16_at(self.data, p + 2 * i)).collect()
            };
            let backtrack = if kind == 6 {
                let n = usize::from(u16_at(self.data, p)?);
                p += 2;
                let g = read(p, n)?;
                p += 2 * n;
                g
            } else {
                Vec::new()
            };
            let input_n = usize::from(u16_at(self.data, p)?);
            p += 2;
            // As in format 1, the input array omits the position coverage matched.
            let input = read(p, input_n.saturating_sub(1))?;
            p += 2 * input.len();
            let lookahead = if kind == 6 {
                let n = usize::from(u16_at(self.data, p)?);
                p += 2;
                let g = read(p, n)?;
                p += 2 * n;
                g
            } else {
                Vec::new()
            };
            let record_n = usize::from(u16_at(self.data, p)?);
            p += 2;

            if input_n == 0 || at + input_n > buf.len() || backtrack.len() > at {
                continue;
            }
            let class = |def: usize, i: usize| self.class_of(def, buf.glyphs[i]);
            if !input.iter().enumerate().all(|(k, c)| class(input_def, at + 1 + k) == *c) {
                continue;
            }
            if !backtrack.iter().enumerate().all(|(k, c)| class(back_def, at - 1 - k) == *c) {
                continue;
            }
            let after = at + input_n;
            if after + lookahead.len() > buf.len() {
                continue;
            }
            if !lookahead.iter().enumerate().all(|(k, c)| class(ahead_def, after + k) == *c) {
                continue;
            }
            let mut did = false;
            for i in 0..record_n {
                let rec = p + 4 * i;
                let seq = usize::from(u16_at(self.data, rec)?);
                let lookup = u16_at(self.data, rec + 2)?;
                if self.apply_at(lookup, buf, at + seq, depth + 1) {
                    did = true;
                }
            }
            return Some(did);
        }
        None
    }

    /// A contextual or chaining-contextual subtable.
    ///
    /// Formats **1** and **3** are read. Format 3 is the coverage-array form a feature
    /// file compiles to; format 1 lists rules as explicit glyph sequences, keyed by a
    /// coverage of first glyphs. `SFArabic`'s `rclt` uses fifteen of format 3 and **two
    /// of format 1**, and the two are what rewrite a decomposed letter's skeleton in
    /// context — skipping them left 20 of 2000 fuzzed Arabic words with the wrong base
    /// glyph while the dots were right.
    ///
    /// Format **2** keys rules by *class*, and the class — not the coverage index — is
    /// what selects the rule set; coverage only decides whether the subtable applies at
    /// all. `SFNS` uses it, which a corpus of Latin words could not show: it is what gives
    /// a colon its alternate form after a digit, three times in 2000 fuzzed strings.
    fn contextual(
        &self,
        sub: usize,
        buf: &mut Buffer,
        at: usize,
        kind: u16,
        depth: usize,
    ) -> Option<bool> {
        match u16_at(self.data, sub)? {
            1 => return self.contextual_1(sub, buf, at, kind, depth),
            2 => return self.contextual_2(sub, buf, at, kind, depth),
            3 => {}
            _ => return None,
        }
        let mut p = sub + 2;
        // A non-chaining context has no backtrack or lookahead.
        let backtrack = if kind == 6 {
            let n = usize::from(u16_at(self.data, p)?);
            p += 2;
            let offs: Vec<usize> = (0..n)
                .map(|i| u16_at(self.data, p + 2 * i).map(usize::from))
                .collect::<Option<Vec<usize>>>()?;
            p += 2 * n;
            offs
        } else {
            Vec::new()
        };
        let input_n = usize::from(u16_at(self.data, p)?);
        p += 2;
        let input: Vec<usize> = (0..input_n)
            .map(|i| u16_at(self.data, p + 2 * i).map(usize::from))
            .collect::<Option<Vec<usize>>>()?;
        p += 2 * input_n;
        let lookahead = if kind == 6 {
            let n = usize::from(u16_at(self.data, p)?);
            p += 2;
            let offs: Vec<usize> = (0..n)
                .map(|i| u16_at(self.data, p + 2 * i).map(usize::from))
                .collect::<Option<Vec<usize>>>()?;
            p += 2 * n;
            offs
        } else {
            Vec::new()
        };
        let record_n = usize::from(u16_at(self.data, p)?);
        p += 2;

        // The input sequence starts at `at`; the backtrack runs **backwards** from just
        // before it, which is why its coverage array is stored in reverse order.
        if input.is_empty() || at + input.len() > buf.len() {
            return None;
        }
        for (k, off) in input.iter().enumerate() {
            coverage_index(self.data, sub + off, buf.glyphs[at + k])?;
        }
        if backtrack.len() > at {
            return None;
        }
        for (k, off) in backtrack.iter().enumerate() {
            coverage_index(self.data, sub + off, buf.glyphs[at - 1 - k])?;
        }
        let after = at + input.len();
        if after + lookahead.len() > buf.len() {
            return None;
        }
        for (k, off) in lookahead.iter().enumerate() {
            coverage_index(self.data, sub + off, buf.glyphs[after + k])?;
        }

        // Matched: apply the nested lookups at their positions inside the input.
        let mut did = false;
        for i in 0..record_n {
            let rec = p + 4 * i;
            let seq = usize::from(u16_at(self.data, rec)?);
            let lookup = u16_at(self.data, rec + 2)?;
            if self.apply_at(lookup, buf, at + seq, depth + 1) {
                did = true;
            }
        }
        Some(did)
    }

    /// One single-substitution subtable.
    fn single(&self, sub: usize, glyph: u16) -> Option<u16> {
        let coverage = sub + usize::from(u16_at(self.data, sub + 2)?);
        let index = coverage_index(self.data, coverage, glyph)?;
        match u16_at(self.data, sub)? {
            // Format 1 adds a delta to the glyph id, which is how a contiguous block of
            // forms is expressed in four bytes.
            1 => {
                let delta = u16_at(self.data, sub + 4)? as i32;
                u16::try_from((i32::from(glyph) + delta) & 0xFFFF).ok()
            }
            // Format 2 lists a replacement per covered glyph.
            2 => {
                if index >= u16_at(self.data, sub + 4)? {
                    return None;
                }
                u16_at(self.data, sub + 6 + 2 * usize::from(index))
            }
            _ => None,
        }
    }

    /// One multiple-substitution subtable: one glyph in, a sequence out.
    ///
    /// How a font that draws a letter as a dotless skeleton plus a separate dot mark
    /// expresses it. `SFArabic` reaches these through `rclt`, not `ccmp`.
    fn multiple(&self, sub: usize, glyph: u16) -> Option<Vec<u16>> {
        if u16_at(self.data, sub)? != 1 {
            return None;
        }
        let coverage = sub + usize::from(u16_at(self.data, sub + 2)?);
        let index = usize::from(coverage_index(self.data, coverage, glyph)?);
        if index >= usize::from(u16_at(self.data, sub + 4)?) {
            return None;
        }
        let seq = sub + usize::from(u16_at(self.data, sub + 6 + 2 * index)?);
        let n = usize::from(u16_at(self.data, seq)?);
        (0..n).map(|i| u16_at(self.data, seq + 2 + 2 * i)).collect()
    }

    /// One ligature-substitution subtable: the longest match wins.
    fn ligature_in(&self, sub: usize, first: u16, glyphs: &[u16]) -> Option<(u16, usize)> {
        let coverage = sub + usize::from(u16_at(self.data, sub + 2)?);
        let index = usize::from(coverage_index(self.data, coverage, first)?);
        if index >= usize::from(u16_at(self.data, sub + 4)?) {
            return None;
        }
        let set = sub + usize::from(u16_at(self.data, sub + 6 + 2 * index)?);
        let n = usize::from(u16_at(self.data, set)?);
        let mut best: Option<(u16, usize)> = None;
        for i in 0..n {
            let lig = set + usize::from(u16_at(self.data, set + 2 + 2 * i)?);
            let glyph = u16_at(self.data, lig)?;
            let comps = usize::from(u16_at(self.data, lig + 2)?);
            if comps == 0 || comps > glyphs.len() {
                continue;
            }
            // The component array omits the first glyph, which coverage already matched.
            let matches = (1..comps)
                .all(|k| u16_at(self.data, lig + 2 + 2 * k).is_some_and(|w| w == glyphs[k]));
            if matches && best.is_none_or(|(_, m)| comps > m) {
                best = Some((glyph, comps));
            }
        }
        best
    }
}

/// Where `glyph` sits in a coverage table, or `None` if it is not covered.
#[must_use]
pub fn coverage_index(d: &[u8], at: usize, glyph: u16) -> Option<u16> {
    match u16_at(d, at)? {
        1 => {
            let n = usize::from(u16_at(d, at + 2)?);
            let mut lo = 0usize;
            let mut hi = n;
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                let g = u16_at(d, at + 4 + 2 * mid)?;
                if glyph < g {
                    hi = mid;
                } else if glyph > g {
                    lo = mid + 1;
                } else {
                    return u16::try_from(mid).ok();
                }
            }
            None
        }
        2 => {
            let n = usize::from(u16_at(d, at + 2)?);
            for i in 0..n {
                let rec = at + 4 + 6 * i;
                let (first, last, start) =
                    (u16_at(d, rec)?, u16_at(d, rec + 2)?, u16_at(d, rec + 4)?);
                if glyph >= first && glyph <= last {
                    return start.checked_add(glyph - first);
                }
            }
            None
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    /// A `GSUB` with one feature under one script, pointing at `lookups`.
    ///
    /// Built by hand rather than checked in as a binary, so the test says which bytes
    /// matter — the same reason `swiftui-font`'s tests build a font.
    fn table(tag: &[u8; 4], lookups: &[Vec<u8>]) -> Vec<u8> {
        // Header: version, scriptList, featureList, lookupList.
        let mut out = vec![0u8; 10];
        out[0..2].copy_from_slice(&1u16.to_be_bytes());
        out[2..4].copy_from_slice(&0u16.to_be_bytes());

        // ScriptList with one script, one default LangSys listing feature 0.
        let script_at = out.len();
        out.extend_from_slice(&1u16.to_be_bytes()); // scriptCount
        out.extend_from_slice(b"arab");
        out.extend_from_slice(&8u16.to_be_bytes()); // scriptOffset, past this record
        // Relative to the Script table, which is four bytes: the offset itself and
        // `langSysCount`.
        out.extend_from_slice(&4u16.to_be_bytes()); // defaultLangSysOffset
        out.extend_from_slice(&0u16.to_be_bytes()); // langSysCount
        out.extend_from_slice(&0u16.to_be_bytes()); // lookupOrder
        out.extend_from_slice(&0xFFFFu16.to_be_bytes()); // requiredFeatureIndex: none
        out.extend_from_slice(&1u16.to_be_bytes()); // featureIndexCount
        out.extend_from_slice(&0u16.to_be_bytes()); // featureIndices[0]

        // FeatureList with one feature listing every lookup.
        let feature_at = out.len();
        out.extend_from_slice(&1u16.to_be_bytes());
        out.extend_from_slice(tag);
        out.extend_from_slice(&8u16.to_be_bytes()); // featureOffset
        out.extend_from_slice(&0u16.to_be_bytes()); // featureParams
        #[allow(clippy::cast_possible_truncation)]
        out.extend_from_slice(&(lookups.len() as u16).to_be_bytes());
        for i in 0..lookups.len() {
            #[allow(clippy::cast_possible_truncation)]
            out.extend_from_slice(&(i as u16).to_be_bytes());
        }

        // LookupList.
        let lookup_at = out.len();
        #[allow(clippy::cast_possible_truncation)]
        out.extend_from_slice(&(lookups.len() as u16).to_be_bytes());
        let mut at = 2 + 2 * lookups.len();
        for l in lookups {
            #[allow(clippy::cast_possible_truncation)]
            out.extend_from_slice(&(at as u16).to_be_bytes());
            at += l.len();
        }
        for l in lookups {
            out.extend_from_slice(l);
        }
        #[allow(clippy::cast_possible_truncation)]
        {
            out[4..6].copy_from_slice(&(script_at as u16).to_be_bytes());
            out[6..8].copy_from_slice(&(feature_at as u16).to_be_bytes());
            out[8..10].copy_from_slice(&(lookup_at as u16).to_be_bytes());
        }
        out
    }

    /// A lookup header wrapping one subtable.
    fn lookup(kind: u16, sub: &[u8]) -> Vec<u8> {
        let mut out = vec![];
        out.extend_from_slice(&kind.to_be_bytes());
        out.extend_from_slice(&0u16.to_be_bytes()); // flags
        out.extend_from_slice(&1u16.to_be_bytes()); // subTableCount
        out.extend_from_slice(&8u16.to_be_bytes()); // offset, past this header
        out.extend_from_slice(sub);
        out
    }

    /// A format-1 coverage listing `glyphs`.
    fn coverage(glyphs: &[u16]) -> Vec<u8> {
        let mut out = vec![];
        out.extend_from_slice(&1u16.to_be_bytes());
        #[allow(clippy::cast_possible_truncation)]
        out.extend_from_slice(&(glyphs.len() as u16).to_be_bytes());
        for g in glyphs {
            out.extend_from_slice(&g.to_be_bytes());
        }
        out
    }

    #[test]
    fn a_single_substitution_by_delta_and_by_list() {
        // Format 1: coverage, then a delta added to the glyph id.
        let mut sub = vec![];
        sub.extend_from_slice(&1u16.to_be_bytes());
        sub.extend_from_slice(&6u16.to_be_bytes()); // coverageOffset
        sub.extend_from_slice(&100u16.to_be_bytes()); // deltaGlyphID
        sub.extend_from_slice(&coverage(&[10, 11]));
        let t = table(b"init", &[lookup(1, &sub)]);
        let g = Gsub::parse(&t).expect("parses");
        // Each stage, so a broken fixture names itself rather than looking like a broken
        // reader.
        assert_eq!(g.features_of(b"arab"), Some(vec![0]), "the script's features");
        assert_eq!(g.lookups_for_script(b"arab", b"init"), vec![0], "the feature's lookups");
        assert_eq!(g.subtables(0).map(|(k, s)| (k, s.len())), Some((1, 1)), "the subtables");
        let mut b = Buffer::new(&[(10, 0), (11, 1), (12, 2)]);
        g.apply_feature(b"arab", b"init", &mut b);
        assert_eq!(b.glyphs, vec![110, 111, 12], "the uncovered glyph changed");

        // Format 2: a replacement listed per covered glyph.
        let mut sub = vec![];
        sub.extend_from_slice(&2u16.to_be_bytes()); // substFormat
        // The header is six bytes plus two per substitute, so the coverage goes after.
        sub.extend_from_slice(&10u16.to_be_bytes()); // coverageOffset
        sub.extend_from_slice(&2u16.to_be_bytes()); // glyphCount
        sub.extend_from_slice(&77u16.to_be_bytes());
        sub.extend_from_slice(&88u16.to_be_bytes());
        sub.extend_from_slice(&coverage(&[10, 11]));
        let t = table(b"init", &[lookup(1, &sub)]);
        let g = Gsub::parse(&t).expect("parses");
        let mut b = Buffer::new(&[(10, 0), (11, 1)]);
        g.apply_feature(b"arab", b"init", &mut b);
        assert_eq!(b.glyphs, vec![77, 88]);
    }

    #[test]
    fn a_multiple_substitution_keeps_the_cluster() {
        let mut sub = vec![];
        sub.extend_from_slice(&1u16.to_be_bytes());
        sub.extend_from_slice(&8u16.to_be_bytes()); // coverageOffset
        sub.extend_from_slice(&1u16.to_be_bytes()); // sequenceCount
        // A one-glyph coverage is six bytes, 8..14, so the sequence follows it.
        sub.extend_from_slice(&14u16.to_be_bytes()); // sequenceOffsets[0]
        sub.extend_from_slice(&coverage(&[10]));
        sub.extend_from_slice(&2u16.to_be_bytes()); // glyphCount
        sub.extend_from_slice(&20u16.to_be_bytes());
        sub.extend_from_slice(&21u16.to_be_bytes());
        let t = table(b"ccmp", &[lookup(2, &sub)]);
        let g = Gsub::parse(&t).expect("parses");
        let mut b = Buffer::new(&[(10, 5), (11, 6)]);
        g.apply_feature(b"arab", b"ccmp", &mut b);
        assert_eq!(b.glyphs, vec![20, 21, 11]);
        // Both halves of the decomposition belong to the character they came from, which
        // is what lets a caller map a glyph back to the text.
        assert_eq!(b.clusters, vec![5, 5, 6]);
    }

    #[test]
    fn a_ligature_takes_the_longest_match() {
        // Two ligatures from glyph 10: 10+11 and 10+11+12. The longer must win.
        let mut set = vec![];
        set.extend_from_slice(&2u16.to_be_bytes()); // ligatureCount
        set.extend_from_slice(&6u16.to_be_bytes()); // the two-glyph one
        set.extend_from_slice(&12u16.to_be_bytes()); // the three-glyph one
        set.extend_from_slice(&50u16.to_be_bytes()); // ligatureGlyph
        set.extend_from_slice(&2u16.to_be_bytes()); // componentCount
        set.extend_from_slice(&11u16.to_be_bytes());
        set.extend_from_slice(&51u16.to_be_bytes());
        set.extend_from_slice(&3u16.to_be_bytes());
        set.extend_from_slice(&11u16.to_be_bytes());
        set.extend_from_slice(&12u16.to_be_bytes());
        let mut sub = vec![];
        sub.extend_from_slice(&1u16.to_be_bytes());
        sub.extend_from_slice(&8u16.to_be_bytes()); // coverageOffset
        sub.extend_from_slice(&1u16.to_be_bytes()); // ligatureSetCount
        sub.extend_from_slice(&14u16.to_be_bytes()); // ligatureSetOffsets[0], past coverage
        sub.extend_from_slice(&coverage(&[10]));
        sub.extend_from_slice(&set);
        let t = table(b"rlig", &[lookup(4, &sub)]);
        let g = Gsub::parse(&t).expect("parses");
        let mut b = Buffer::new(&[(10, 0), (11, 1), (12, 2)]);
        g.apply_feature(b"arab", b"rlig", &mut b);
        assert_eq!(b.glyphs, vec![51], "the shorter ligature won");
        assert_eq!(b.clusters, vec![0]);

        // Without the third glyph, the two-glyph one applies.
        let mut b = Buffer::new(&[(10, 0), (11, 1)]);
        g.apply_feature(b"arab", b"rlig", &mut b);
        assert_eq!(b.glyphs, vec![50]);
    }

    #[test]
    fn a_feature_under_another_script_is_not_applied() {
        // The whole reason script selection exists: the same tag means different lookups
        // under different scripts, and taking the union applies the wrong ones.
        let mut sub = vec![];
        sub.extend_from_slice(&1u16.to_be_bytes());
        sub.extend_from_slice(&6u16.to_be_bytes());
        sub.extend_from_slice(&100u16.to_be_bytes());
        sub.extend_from_slice(&coverage(&[10]));
        let t = table(b"rlig", &[lookup(1, &sub)]);
        let g = Gsub::parse(&t).expect("parses");
        assert!(g.has(b"arab", b"rlig"));
        // A script the table does not list falls back to `DFLT`, which it also does not
        // list — so nothing is allowed and nothing applies.
        let mut b = Buffer::new(&[(10, 0)]);
        g.apply_feature(b"latn", b"rlig", &mut b);
        assert_eq!(b.glyphs, vec![10], "a lookup from another script was applied");
    }

    #[test]
    fn a_truncated_table_is_refused_rather_than_read() {
        let mut sub = vec![];
        sub.extend_from_slice(&1u16.to_be_bytes());
        sub.extend_from_slice(&6u16.to_be_bytes());
        sub.extend_from_slice(&100u16.to_be_bytes());
        sub.extend_from_slice(&coverage(&[10]));
        let full = table(b"init", &[lookup(1, &sub)]);
        for cut in 0..full.len() {
            // Either it refuses to parse or it substitutes nothing; it must not panic.
            if let Some(g) = Gsub::parse(&full[..cut]) {
                let mut b = Buffer::new(&[(10, 0)]);
                g.apply_feature(b"arab", b"init", &mut b);
            }
        }
    }
}
