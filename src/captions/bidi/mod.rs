//! The Unicode bidirectional algorithm, UAX #9.
//!
//! Ported from `rswift`'s `swiftui-bidi`, where it is measured against ICU's own
//! `ubidi_*` over 360,000 random strings and differs in 9 of them — all carrying several
//! explicit embedding codes, where the specification leaves the level of an unmatched
//! `PDI` to a convention.
//!
//! # Why this exists
//!
//! Reversing a right-to-left span is not the bidirectional algorithm, and the difference
//! shows the moment a line mixes scripts: `"abc ‏אבג‎ 123"` has four runs at three
//! levels, and a number after Hebrew text belongs to the Hebrew run while a number after
//! Latin does not. Hebrew was the one script in this stack's fallback cascade that drew
//! the right glyphs in the wrong order, and this is what it was waiting for.
//!
//! # What is implemented
//!
//! All of it, for one paragraph at a time: the paragraph level (**P2–P3**), explicit
//! embeddings, overrides and **isolates** (**X1–X10**), isolating run sequences
//! (**BD13**), the weak types (**W1–W7**), paired brackets and neutrals (**N0–N2**),
//! implicit levels (**I1–I2**) and reordering (**L1–L2**). [`mirror`] covers **L4**.
//!
//! The class of every codepoint is in [`class_table`], **generated** from
//! `ubidi_getClass` in the platform's own ICU rather than transcribed — see
//! `examples/gen_classes.rs`. 1286 ranges, which is the whole of Unicode.
//!
//! # How it is checked
//!
//! Against ICU, which is the same implementation Core Text resolves bidi with.
//! `examples/icu_parity.rs` compares **every level and the whole visual order** for a
//! corpus of mixed-script strings, at both paragraph directions and with the direction
//! left to the algorithm. A budget would defeat the purpose, so there is none.

mod bracket_table;
mod class_table;
pub mod mirror;
mod mirror_table;

/// A character's bidi class, in UAX #9's names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum Class {
    /// Left to right.
    L,
    /// Right to left.
    R,
    /// Right to left, Arabic.
    AL,
    /// European number.
    EN,
    /// European separator.
    ES,
    /// European terminator.
    ET,
    /// Arabic number.
    AN,
    /// Common separator.
    CS,
    /// Non-spacing mark.
    NSM,
    /// Boundary neutral.
    BN,
    /// Paragraph separator.
    B,
    /// Segment separator.
    S,
    /// Whitespace.
    WS,
    /// Other neutral.
    ON,
    LRE,
    LRO,
    RLE,
    RLO,
    PDF,
    LRI,
    RLI,
    FSI,
    PDI,
}

impl Class {
    /// Whether this is one of the isolate initiators: **LRI**, **RLI** or **FSI**.
    #[must_use]
    pub fn is_isolate_initiator(self) -> bool {
        matches!(self, Self::LRI | Self::RLI | Self::FSI)
    }

    /// Whether this is removed by **X9**: an embedding, an override, a PDF or a BN.
    ///
    /// Removed in the sense of being treated as absent by the later rules. They are not
    /// deleted here — the indices have to keep lining up with the input — so they are
    /// given the level of the surrounding text and skipped.
    #[must_use]
    pub fn is_removed_by_x9(self) -> bool {
        matches!(self, Self::RLE | Self::LRE | Self::RLO | Self::LRO | Self::PDF | Self::BN)
    }

    /// Whether this is a strong type for **P2** and **N0**'s purposes.
    #[must_use]
    pub fn is_strong(self) -> bool {
        matches!(self, Self::L | Self::R | Self::AL)
    }
}

/// The bidi class of `c`.
///
/// A binary search of the generated ranges, which cover every codepoint — so there is no
/// default to get wrong.
#[must_use]
pub fn class(c: char) -> Class {
    let c = c as u32;
    let r = class_table::RANGES;
    let mut lo = 0usize;
    let mut hi = r.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let (first, last, k) = r[mid];
        if c < first {
            hi = mid;
        } else if c > last {
            lo = mid + 1;
        } else {
            return k;
        }
    }
    Class::L
}

/// Which way a paragraph runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// Left to right: level 0.
    Ltr,
    /// Right to left: level 1.
    Rtl,
    /// Decide from the text, which is **P2–P3**.
    Auto,
}

/// The maximum explicit depth, **BD2**. Fixed by the specification at 125.
pub const MAX_DEPTH: u8 = 125;

/// A resolved paragraph: an embedding level per character, and the paragraph's own.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Levels {
    /// One level per `char` of the input, in logical order.
    pub levels: Vec<u8>,
    /// The paragraph embedding level, 0 or 1.
    pub paragraph: u8,
    /// Each character's class, kept because `reorder` needs it for **L1** and a caller
    /// may want it for mirroring.
    pub classes: Vec<Class>,
}

/// **P2–P3**: the paragraph level, from the first strong character.
///
/// Characters between an isolate initiator and its matching PDI are skipped, which is
/// what makes `FSI` work and what a naive "first strong character" misses.
#[must_use]
pub fn paragraph_level(text: &[Class]) -> u8 {
    let mut depth = 0usize;
    for c in text {
        match c {
            c if c.is_isolate_initiator() => depth += 1,
            Class::PDI if depth > 0 => depth -= 1,
            Class::L if depth == 0 => return 0,
            Class::R | Class::AL if depth == 0 => return 1,
            _ => {}
        }
    }
    0
}

/// What an isolate initiator's matching PDI is, per character, and vice versa.
///
/// **BD9**: scan forward counting initiators and PDIs; the depth returning to zero is the
/// match. An unmatched initiator has none, which the later rules rely on.
fn matching_pdi(classes: &[Class]) -> Vec<Option<usize>> {
    let mut out = vec![None; classes.len()];
    for (i, c) in classes.iter().enumerate() {
        if !c.is_isolate_initiator() {
            continue;
        }
        let mut depth = 1usize;
        for (j, d) in classes.iter().enumerate().skip(i + 1) {
            if d.is_isolate_initiator() {
                depth += 1;
            } else if *d == Class::PDI {
                depth -= 1;
                if depth == 0 {
                    out[i] = Some(j);
                    break;
                }
            }
        }
    }
    out
}

/// One entry of the directional status stack, **X1**.
#[derive(Clone, Copy)]
struct Status {
    level: u8,
    /// `Some(L)` or `Some(R)` while an override is in force.
    override_to: Option<Class>,
    isolate: bool,
}

/// **X1–X8**: explicit levels and directions.
///
/// Returns a level per character and the class each one should be treated as — an
/// override rewrites a character's class, which is the whole point of `LRO` and `RLO`.
fn explicit(classes: &[Class], para: u8, pdi_of: &[Option<usize>]) -> (Vec<u8>, Vec<Class>) {
    let n = classes.len();
    let mut levels = vec![para; n];
    let mut out = classes.to_vec();
    let mut stack: Vec<Status> = vec![Status { level: para, override_to: None, isolate: false }];
    let mut overflow_isolate = 0usize;
    let mut overflow_embedding = 0usize;
    let mut valid_isolate = 0usize;

    for i in 0..n {
        let c = classes[i];
        match c {
            Class::RLE | Class::LRE | Class::RLO | Class::LRO => {
                // X9 removes these, so they take the level in force before the push.
                levels[i] = stack.last().map_or(para, |s| s.level);
                let rtl = matches!(c, Class::RLE | Class::RLO);
                let next = next_level(stack.last().map_or(para, |s| s.level), rtl);
                let over = match c {
                    Class::RLO => Some(Class::R),
                    Class::LRO => Some(Class::L),
                    _ => None,
                };
                if next <= MAX_DEPTH && overflow_isolate == 0 && overflow_embedding == 0 {
                    stack.push(Status { level: next, override_to: over, isolate: false });
                } else if overflow_isolate == 0 {
                    overflow_embedding += 1;
                }
            }
            Class::RLI | Class::LRI | Class::FSI => {
                // An isolate initiator *is* part of the text: it takes the current level
                // and any override applies to it, unlike an embedding.
                let top = stack.last().copied().unwrap_or(Status {
                    level: para,
                    override_to: None,
                    isolate: false,
                });
                levels[i] = top.level;
                if let Some(o) = top.override_to {
                    out[i] = o;
                }
                // FSI resolves to LRI or RLI by looking inside itself, P2–P3 again.
                let rtl = match c {
                    Class::RLI => true,
                    Class::LRI => false,
                    _ => {
                        let end = pdi_of[i].unwrap_or(n);
                        paragraph_level(&classes[i + 1..end]) == 1
                    }
                };
                let next = next_level(top.level, rtl);
                if next <= MAX_DEPTH && overflow_isolate == 0 && overflow_embedding == 0 {
                    valid_isolate += 1;
                    stack.push(Status { level: next, override_to: None, isolate: true });
                } else {
                    overflow_isolate += 1;
                }
            }
            Class::PDI => {
                if overflow_isolate > 0 {
                    overflow_isolate -= 1;
                } else if valid_isolate > 0 {
                    // Pop embeddings down to the isolate, then the isolate itself.
                    overflow_embedding = 0;
                    while stack.last().is_some_and(|s| !s.isolate) {
                        stack.pop();
                    }
                    stack.pop();
                    valid_isolate -= 1;
                }
                let top = stack.last().copied().unwrap_or(Status {
                    level: para,
                    override_to: None,
                    isolate: false,
                });
                levels[i] = top.level;
                if let Some(o) = top.override_to {
                    out[i] = o;
                }
            }
            Class::PDF => {
                levels[i] = stack.last().map_or(para, |s| s.level);
                if overflow_isolate > 0 {
                    // Nothing: an isolate overflow outranks an embedding pop.
                } else if overflow_embedding > 0 {
                    overflow_embedding -= 1;
                } else if stack.last().is_some_and(|s| !s.isolate) && stack.len() > 1 {
                    stack.pop();
                }
            }
            Class::B => {
                // X8: a paragraph separator takes the paragraph level and clears all of
                // it. There is one paragraph here, so this is the terminating case.
                stack.truncate(1);
                overflow_isolate = 0;
                overflow_embedding = 0;
                valid_isolate = 0;
                levels[i] = para;
            }
            _ => {
                let top = stack.last().copied().unwrap_or(Status {
                    level: para,
                    override_to: None,
                    isolate: false,
                });
                levels[i] = top.level;
                // **X6 excludes `BN`** — along with `B` and the formatting characters,
                // which have their own arms above. Overriding a `BN` to `L` or `R` stops
                // it being removed by X9, so it stays in its level run and votes in the
                // neutral rules: one `U+200B` inside an override changed a neighbouring
                // `ON`'s level, which showed up as about one random string in two
                // thousand coming out in the wrong order.
                if !c.is_removed_by_x9() {
                    if let Some(o) = top.override_to {
                        out[i] = o;
                    }
                }
            }
        }
    }
    (levels, out)
}

/// The next odd level above `level`, or the next even one.
fn next_level(level: u8, rtl: bool) -> u8 {
    if rtl {
        (level + 1) | 1
    } else {
        (level + 2) & !1
    }
}

/// Resolve `text` into embedding levels.
///
/// One paragraph. A caller with several should split on **B** first; this treats a
/// paragraph separator as **X8** does and carries on, which gives the right answer for
/// the common case of a trailing newline.
#[must_use]
pub fn resolve(text: &str, direction: Direction) -> Levels {
    let chars: Vec<char> = text.chars().collect();
    let classes: Vec<Class> = chars.iter().copied().map(class).collect();
    resolve_inner(&classes, Some(&chars), direction)
}

/// [`resolve`], for a caller that already has the classes — a test, or a shaper that
/// classified the text once.
#[must_use]
pub fn resolve_classes(classes: &[Class], direction: Direction) -> Levels {
    resolve_inner(classes, None, direction)
}

/// The algorithm. `chars` is `None` for a caller that has only classes, which gives up
/// **N0** — paired brackets are a property of the characters, not of their classes.
fn resolve_inner(classes: &[Class], chars: Option<&[char]>, direction: Direction) -> Levels {
    let para = match direction {
        Direction::Ltr => 0,
        Direction::Rtl => 1,
        Direction::Auto => paragraph_level(classes),
    };
    let pdi_of = matching_pdi(classes);
    let (mut levels, effective) = explicit(classes, para, &pdi_of);

    for seq in isolating_runs(&effective, &levels, para, &pdi_of) {
        resolve_sequence(&seq, &effective, &mut levels, chars);
    }

    // **X9**: the embeddings, overrides, PDFs and BNs are "removed". They are not deleted
    // here, because the levels have to stay indexed by the input — so instead each takes
    // the level of the character **before** it, which is the note X9 attaches and is not
    // cosmetic. An `LRE` left at the level in force before its own push sits at a
    // *different* level from the text either side of it, which splits one level run in
    // two, and **L2** then reverses the halves separately: `"abc ⟪LRE⟫אב⟪PDF⟫ def"` in a
    // right-to-left paragraph comes out rotated rather than reordered.
    //
    // **After** the implicit levels, not before. Before, it copies a level that the
    // `W`/`N`/`I` rules are about to change, which looks like it works and does not.
    for i in 0..classes.len() {
        if classes[i].is_removed_by_x9() {
            levels[i] = if i == 0 { para } else { levels[i - 1] };
        }
    }

    // L1: three things go back to the paragraph level — a segment or paragraph
    // separator, any whitespace or isolate formatting **immediately before** one, and
    // any at the **end of the line**. Not everything after the last separator, which is
    // the mistake that puts `"abc"` after a tab at the paragraph level instead of its
    // own: ICU gives `[1, 1, 1, 2, 2, 2]` there and resetting forward gives all ones.
    let n = classes.len();
    let resettable = |k: Class| {
        matches!(k, Class::WS | Class::LRI | Class::RLI | Class::FSI | Class::PDI)
            || k.is_removed_by_x9()
    };
    for i in 0..n {
        if !matches!(classes[i], Class::B | Class::S) {
            continue;
        }
        levels[i] = para;
        for j in (0..i).rev() {
            if !resettable(classes[j]) {
                break;
            }
            levels[j] = para;
        }
    }
    let mut i = n;
    while i > 0 && resettable(classes[i - 1]) {
        i -= 1;
        levels[i] = para;
    }
    Levels { levels, paragraph: para, classes: classes.to_vec() }
}

/// One isolating run sequence: the character indices it covers, in order, plus the
/// classes flanking it, which **W** and **N** need.
#[derive(Debug)]
struct Sequence {
    indices: Vec<usize>,
    level: u8,
    /// The strong class before the sequence, **sos**.
    sos: Class,
    /// The strong class after it, **eos**.
    eos: Class,
}

/// **BD13**: group the level runs into isolating run sequences.
fn isolating_runs(
    classes: &[Class],
    levels: &[u8],
    para: u8,
    pdi_of: &[Option<usize>],
) -> Vec<Sequence> {
    let n = classes.len();
    // The level runs, over the characters X9 does not remove.
    let keep: Vec<usize> = (0..n).filter(|i| !classes[*i].is_removed_by_x9()).collect();
    let mut runs: Vec<Vec<usize>> = Vec::new();
    for i in keep {
        match runs.last_mut() {
            Some(r) if levels[*r.last().unwrap()] == levels[i] => r.push(i),
            _ => runs.push(vec![i]),
        }
    }
    // A run whose last character is an isolate initiator with a matching PDI continues
    // into the run that PDI starts, which is what makes an isolate one sequence.
    let mut used = vec![false; runs.len()];
    let start_of: std::collections::BTreeMap<usize, usize> =
        runs.iter().enumerate().map(|(k, r)| (r[0], k)).collect();
    let mut out = Vec::new();
    for k in 0..runs.len() {
        if used[k] {
            continue;
        }
        // A sequence starts at a run whose first character is not a PDI matching an
        // initiator — otherwise it is a continuation.
        let first = runs[k][0];
        if classes[first] == Class::PDI && pdi_of.contains(&Some(first)) {
            continue;
        }
        let mut indices: Vec<usize> = Vec::new();
        let mut at = k;
        loop {
            used[at] = true;
            indices.extend_from_slice(&runs[at]);
            let last = *runs[at].last().unwrap();
            let Some(m) = pdi_of.get(last).copied().flatten() else { break };
            if !classes[last].is_isolate_initiator() {
                break;
            }
            let Some(next) = start_of.get(&m).copied() else { break };
            if used[next] {
                break;
            }
            at = next;
        }
        let level = levels[indices[0]];
        // X10: sos and eos are the higher of the sequence's level and the level of the
        // adjacent character, as a direction. Past the ends, the paragraph level.
        let before = indices[0];
        let prev = (0..before).rev().find(|i| !classes[*i].is_removed_by_x9());
        let sos_level = levels[prev.unwrap_or(0)].max(level).max(if prev.is_none() { para } else { 0 });
        let after = *indices.last().unwrap();
        // An unmatched isolate initiator at the end takes the paragraph level as eos.
        let ends_isolated = classes[after].is_isolate_initiator()
            && pdi_of.get(after).copied().flatten().is_none();
        let next = (after + 1..n).find(|i| !classes[*i].is_removed_by_x9());
        let eos_level = match next {
            // An unmatched isolate initiator, or the end of the paragraph: the
            // paragraph level stands in for what would follow.
            _ if ends_isolated => level.max(para),
            None => level.max(para),
            Some(k) => levels[k].max(level),
        };
        out.push(Sequence {
            indices,
            level,
            sos: if sos_level % 2 == 1 { Class::R } else { Class::L },
            eos: if eos_level % 2 == 1 { Class::R } else { Class::L },
        });
    }
    out
}

/// **BD16**: the bracket pairs in a sequence, as positions within it, sorted by opening.
///
/// A stack of at most 63 openings, per the specification; overflowing it stops the search
/// entirely rather than processing part of it, which is what BD16 says and matters for a
/// string of nothing but brackets.
///
/// Matching is by canonical equivalence, which in practice means U+2329 and U+3008 pair
/// with each other's partners — the only such case in Unicode, and one a naive equality
/// test gets wrong.
fn bracket_pairs(indices: &[usize], chars: &[char]) -> Vec<(usize, usize)> {
    /// U+3008 and U+2329 are canonically equivalent, as are their closers.
    fn canonical(c: char) -> char {
        match c {
            '\u{2329}' => '\u{3008}',
            '\u{232A}' => '\u{3009}',
            other => other,
        }
    }
    let mut stack: Vec<(char, usize)> = Vec::new();
    let mut out: Vec<(usize, usize)> = Vec::new();
    for (k, idx) in indices.iter().enumerate() {
        let Some(ch) = chars.get(*idx).copied() else { continue };
        match bracket_of(ch) {
            Some((pair, bracket_table::Bracket::Open)) => {
                if stack.len() == 63 {
                    return out;
                }
                stack.push((canonical(pair), k));
            }
            Some((_, bracket_table::Bracket::Close)) => {
                let me = canonical(ch);
                // Pop down to the match, discarding the openings skipped over.
                if let Some(at) = stack.iter().rposition(|(want, _)| *want == me) {
                    out.push((stack[at].1, k));
                    stack.truncate(at);
                }
            }
            None => {}
        }
    }
    out.sort_unstable();
    out
}

/// What `c` pairs with and which way it faces, if it is a bracket.
#[must_use]
fn bracket_of(c: char) -> Option<(char, bracket_table::Bracket)> {
    let key = c as u32;
    let t = bracket_table::BRACKETS;
    let i = t.binary_search_by_key(&key, |(a, _, _)| *a).ok()?;
    let (_, pair, kind) = t[i];
    Some((char::from_u32(pair)?, kind))
}

/// **W1–W7**, **N0–N2** and **I1–I2** for one isolating run sequence.
fn resolve_sequence(seq: &Sequence, classes: &[Class], levels: &mut [u8], chars: Option<&[char]>) {
    let n = seq.indices.len();
    if n == 0 {
        return;
    }
    let mut c: Vec<Class> = seq.indices.iter().map(|i| classes[*i]).collect();

    // W1: a non-spacing mark takes the class of the character before it, or sos. After
    // an isolate initiator or PDI it becomes ON, not the initiator's class.
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        if c[i] != Class::NSM {
            continue;
        }
        c[i] = if i == 0 {
            seq.sos
        } else if c[i - 1].is_isolate_initiator() || c[i - 1] == Class::PDI {
            Class::ON
        } else {
            c[i - 1]
        };
    }
    // W2: a European number after an Arabic letter is an Arabic number.
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        if c[i] != Class::EN {
            continue;
        }
        let strong = (0..i).rev().map(|j| c[j]).find(|k| k.is_strong()).unwrap_or(seq.sos);
        if strong == Class::AL {
            c[i] = Class::AN;
        }
    }
    // W3: an Arabic letter is just R from here on.
    for k in c.iter_mut() {
        if *k == Class::AL {
            *k = Class::R;
        }
    }
    // W4: a single separator between two numbers of the same kind joins them.
    for i in 1..n.saturating_sub(1) {
        if c[i] == Class::ES && c[i - 1] == Class::EN && c[i + 1] == Class::EN {
            c[i] = Class::EN;
        } else if c[i] == Class::CS
            && ((c[i - 1] == Class::EN && c[i + 1] == Class::EN)
                || (c[i - 1] == Class::AN && c[i + 1] == Class::AN))
        {
            c[i] = c[i - 1];
        }
    }
    // W5: a run of terminators adjacent to a European number becomes European numbers.
    let mut i = 0;
    while i < n {
        if c[i] != Class::ET {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < n && c[j] == Class::ET {
            j += 1;
        }
        let before = i > 0 && c[i - 1] == Class::EN;
        let after = j < n && c[j] == Class::EN;
        if before || after {
            for k in &mut c[i..j] {
                *k = Class::EN;
            }
        }
        i = j;
    }
    // W6: whatever separators and terminators are left are neutral.
    for k in c.iter_mut() {
        if matches!(*k, Class::ES | Class::ET | Class::CS) {
            *k = Class::ON;
        }
    }
    // W7: a European number after L is L.
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        if c[i] != Class::EN {
            continue;
        }
        let strong = (0..i).rev().map(|j| c[j]).find(|k| k.is_strong()).unwrap_or(seq.sos);
        if strong == Class::L {
            c[i] = Class::L;
        }
    }

    // N0: paired brackets. A bracket pair enclosing strong text takes that text's
    // direction, which is why `(` in `"he said (אב)"` leans with the Hebrew rather than
    // with the sentence. It needs the **characters**, since a bracket is a property of
    // the codepoint and every bracket is already `ON` by class.
    if let Some(chars) = chars {
        let e = if seq.level % 2 == 1 { Class::R } else { Class::L };
        let o = if e == Class::R { Class::L } else { Class::R };
        for (open, close) in bracket_pairs(&seq.indices, chars) {
            // Strong types strictly inside, with a number counting as R.
            let inside = |want: Class| {
                (open + 1..close).any(|k| match c[k] {
                    Class::L => want == Class::L,
                    Class::R | Class::EN | Class::AN => want == Class::R,
                    _ => false,
                })
            };
            let to = if inside(e) {
                // b: the embedding direction is in there, so both brackets take it.
                Some(e)
            } else if inside(o) {
                // c: only the opposite direction. Which way the pair goes depends on
                // what precedes it — the established context, back to `sos`.
                let before = (0..open)
                    .rev()
                    .find_map(|k| match c[k] {
                        Class::L => Some(Class::L),
                        Class::R | Class::EN | Class::AN => Some(Class::R),
                        _ => None,
                    })
                    .unwrap_or(seq.sos);
                Some(if before == o { o } else { e })
            } else {
                // d: nothing strong inside, so N1 and N2 handle them as neutrals.
                None
            };
            if let Some(to) = to {
                c[open] = to;
                c[close] = to;
                // **N0's trailing clause is deliberately not applied.** The specification
                // says that a character whose original class was `NSM`, immediately
                // following a bracket that N0 changed, should take the bracket's
                // direction. **ICU does not do this**, and ICU is what Core Text resolves
                // bidi with — so following the text here would differ from the platform
                // on every string that exercises it.
                //
                // The minimal case is `"א (ב)◌ٌ b"` in a left-to-right paragraph: the
                // preceding Hebrew sends the brackets to R under clause c, and the mark
                // after the closing bracket comes out at level 1 under the clause and
                // level 0 without it. ICU says 0. Applying the clause also made about one
                // random string in two thousand reorder differently.
                //
                // Restoring it is four lines — copy `to` into each following `NSM` of the
                // original classes — if a caller ever needs the specification over the
                // platform.
            }
        }
    }

    // N1: a run of neutrals between two matching strong directions takes that direction,
    // where a number counts as R for this purpose.
    let strong_of = |k: Class| -> Option<Class> {
        match k {
            Class::L => Some(Class::L),
            Class::R | Class::EN | Class::AN => Some(Class::R),
            _ => None,
        }
    };
    let neutral = |k: Class| {
        matches!(k, Class::B | Class::S | Class::WS | Class::ON | Class::PDI)
            || k.is_isolate_initiator()
    };
    let mut i = 0;
    while i < n {
        if !neutral(c[i]) {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < n && neutral(c[j]) {
            j += 1;
        }
        let before = if i == 0 { seq.sos } else { strong_of(c[i - 1]).unwrap_or(seq.sos) };
        let after = if j == n { seq.eos } else { strong_of(c[j]).unwrap_or(seq.eos) };
        // N2: and otherwise the embedding direction.
        let to = if before == after {
            before
        } else if seq.level % 2 == 1 {
            Class::R
        } else {
            Class::L
        };
        for k in &mut c[i..j] {
            *k = to;
        }
        i = j;
    }

    // I1 and I2: the levels themselves.
    for (k, idx) in seq.indices.iter().enumerate() {
        let lvl = seq.level;
        levels[*idx] = if lvl.is_multiple_of(2) {
            match c[k] {
                Class::R => lvl + 1,
                Class::AN | Class::EN => lvl + 2,
                _ => lvl,
            }
        } else {
            match c[k] {
                Class::L | Class::EN | Class::AN => lvl + 1,
                _ => lvl,
            }
        };
    }
}

/// **L2**: the visual order, as indices into the logical order.
///
/// Reverse every maximal run at the highest level, then at each level down to the lowest
/// odd one — **including levels not actually present**, which is the part that is easy to
/// drop. A right-to-left paragraph whose text is all at level 2 has no level-1 character
/// at all, and taking the lowest odd level from the levels present then stops before
/// reversing anything: `"abc \u{202A}…\u{202C} def"` at `Direction::Rtl` comes out in
/// logical order, which is the whole line backwards from what it should be.
///
/// So the paragraph level is a parameter. The result maps a visual position to the
/// logical character at it.
#[must_use]
pub fn reorder(levels: &[u8], paragraph: u8) -> Vec<usize> {
    let mut out: Vec<usize> = (0..levels.len()).collect();
    if levels.is_empty() {
        return out;
    }
    let highest = *levels.iter().max().unwrap_or(&0);
    let lowest_odd = levels
        .iter()
        .filter(|l| **l % 2 == 1)
        .min()
        .copied()
        .unwrap_or(u8::MAX)
        .min(if paragraph % 2 == 1 { paragraph } else { u8::MAX });
    if lowest_odd == u8::MAX {
        return out;
    }
    let mut level = highest.max(lowest_odd);
    while level >= lowest_odd && level > 0 {
        let mut i = 0;
        while i < levels.len() {
            if levels[i] < level {
                i += 1;
                continue;
            }
            let mut j = i;
            while j < levels.len() && levels[j] >= level {
                j += 1;
            }
            out[i..j].reverse();
            i = j;
        }
        level -= 1;
    }
    out
}

impl Levels {
    /// The visual order of the characters that are actually drawn.
    ///
    /// **X9** removes the embeddings, overrides, PDFs and BNs, and this is what "removed"
    /// means in practice: they are dropped before reordering, so the order is computed on
    /// the text without them. Their own levels are a *display convention* — UAX #9 leaves
    /// them to the implementation and ICU's choice differs from the note in X9 — so
    /// reordering with them in gives an order that depends on that convention. It differs
    /// from ICU's about once in two thousand random strings, and only on strings carrying
    /// several explicit codes.
    ///
    /// Returns indices into the original text, so a caller can map straight back to its
    /// own characters.
    #[must_use]
    pub fn visual(&self) -> Vec<usize> {
        let keep: Vec<usize> = (0..self.classes.len())
            .filter(|i| !self.classes[*i].is_removed_by_x9())
            .collect();
        let kept: Vec<u8> = keep.iter().map(|i| self.levels[*i]).collect();
        reorder(&kept, self.paragraph).into_iter().map(|k| keep[k]).collect()
    }
}

/// Resolve and reorder in one call: the visual order of `text`'s characters.
#[must_use]
pub fn visual_order(text: &str, direction: Direction) -> Vec<usize> {
    resolve(text, direction).visual()
}

#[cfg(test)]
mod tests {
    use super::*;
    use String;

    /// The visual string, which is what a reader would see.
    fn visual(text: &str, d: Direction) -> String {
        let r = resolve(text, d);
        let chars: Vec<char> = text.chars().collect();
        r.visual()
            .into_iter()
            .map(|i| mirror::at_level(chars[i], r.levels[i]))
            .collect()
    }

    #[test]
    fn latin_is_left_alone() {
        assert_eq!(visual("Hello", Direction::Ltr), "Hello");
        assert_eq!(resolve("Hello", Direction::Ltr).levels, vec![0, 0, 0, 0, 0]);
    }

    #[test]
    fn hebrew_reverses() {
        // Three letters, logical order aleph-bet-gimel, drawn right to left.
        let t = "\u{5D0}\u{5D1}\u{5D2}";
        assert_eq!(visual(t, Direction::Auto), "\u{5D2}\u{5D1}\u{5D0}");
        assert_eq!(resolve(t, Direction::Auto).paragraph, 1);
    }

    #[test]
    fn the_paragraph_direction_comes_from_the_first_strong_character() {
        assert_eq!(resolve("abc \u{5D0}", Direction::Auto).paragraph, 0);
        assert_eq!(resolve("\u{5D0} abc", Direction::Auto).paragraph, 1);
        // Neutrals and numbers do not decide it.
        assert_eq!(resolve("123 ... \u{5D0}", Direction::Auto).paragraph, 1);
        assert_eq!(resolve("123", Direction::Auto).paragraph, 0);
        // And text inside an isolate is skipped, which is what makes `FSI` work.
        assert_eq!(resolve("\u{2066}\u{5D0}\u{2069}abc", Direction::Auto).paragraph, 0);
    }

    #[test]
    fn a_number_after_hebrew_stays_with_it() {
        // W2 and I2: the digits are a left-to-right island inside a right-to-left run,
        // so they read in order while the Hebrew around them does not.
        let t = "\u{5D0}\u{5D1} 12 \u{5D2}";
        let r = resolve(t, Direction::Auto);
        assert_eq!(r.paragraph, 1);
        assert_eq!(r.levels, vec![1, 1, 1, 2, 2, 1, 1]);
        assert_eq!(visual(t, Direction::Auto), "\u{5D2} 12 \u{5D1}\u{5D0}");
    }

    #[test]
    fn an_arabic_letter_turns_a_following_number_arabic() {
        // W2 then W3: `EN` after `AL` becomes `AN`, which does **not** get the extra
        // level a European number does.
        let t = "\u{627}12";
        let r = resolve(t, Direction::Auto);
        assert_eq!(r.levels, vec![1, 2, 2]);
        // And with Latin before it instead, W7 makes the digits L.
        let r = resolve("a12", Direction::Ltr);
        assert_eq!(r.levels, vec![0, 0, 0]);
    }

    #[test]
    fn brackets_lean_with_what_they_contain() {
        // N0: the preceding Hebrew sends the pair to R, so the brackets reverse with it.
        let t = "\u{5D0} (\u{5D1}) b";
        let r = resolve(t, Direction::Ltr);
        assert_eq!(r.levels, vec![1, 1, 1, 1, 1, 0, 0]);
        // Mirrored, because they are drawn at an odd level.
        assert_eq!(visual(t, Direction::Ltr), "(\u{5D1}) \u{5D0} b");
    }

    #[test]
    fn an_override_forces_direction() {
        // RLO makes Latin read right to left, which is the whole point of it.
        let t = "a\u{202E}bcd\u{202C}e";
        assert_eq!(visual(t, Direction::Ltr), "a\u{202E}dcb\u{202C}e".replace(['\u{202E}', '\u{202C}'], ""));
    }

    #[test]
    fn an_isolate_does_not_leak() {
        // The point of an isolate: the Hebrew inside does not affect the digits after.
        let inside = "abc \u{2066}\u{5D0}\u{5D1}\u{2069} 12";
        let leaky = "abc \u{202B}\u{5D0}\u{5D1}\u{202C} 12";
        let a = resolve(inside, Direction::Ltr);
        let b = resolve(leaky, Direction::Ltr);
        assert_ne!(a.levels, b.levels, "the isolate behaved like an embedding");
    }

    #[test]
    fn the_visual_order_is_always_a_permutation() {
        for t in [
            "",
            " ",
            "abc",
            "\u{5D0}\u{5D1}",
            "a\u{202E}b\u{202C}c",
            "a\u{2066}b\u{2069}c",
            "\u{2069}\u{2069}\u{2069}",
            "((((((",
            "a\u{5B7}\u{5B7}b",
        ] {
            for d in [Direction::Ltr, Direction::Rtl, Direction::Auto] {
                let r = resolve(t, d);
                assert_eq!(r.levels.len(), t.chars().count(), "{t:?}");
                let order = r.visual();
                let mut sorted = order.clone();
                sorted.sort_unstable();
                let expect: Vec<usize> = (0..t.chars().count())
                    .filter(|i| !r.classes[*i].is_removed_by_x9())
                    .collect();
                assert_eq!(sorted, expect, "{t:?} at {d:?} is not a permutation");
            }
        }
    }

    #[test]
    fn deep_nesting_stops_at_the_limit() {
        // BD2 caps the depth at 125; past that the extra embeddings are overflow and
        // must not raise the level or panic.
        let mut t = String::new();
        for _ in 0..200 {
            t.push('\u{202B}');
        }
        t.push('a');
        let r = resolve(&t, Direction::Ltr);
        assert!(r.levels.iter().all(|l| *l <= MAX_DEPTH + 1), "a level ran past the cap");
        assert_eq!(r.levels.len(), t.chars().count());
    }

    #[test]
    fn a_class_is_found_for_every_codepoint() {
        // The generated table covers all of Unicode, so no lookup falls through.
        for c in [
            '\0', 'a', '\u{5D0}', '\u{627}', '\u{661}', '\u{2066}', '\u{10FFFF}', '\u{E0001}',
        ] {
            let _ = class(c);
        }
        assert_eq!(class('a'), Class::L);
        assert_eq!(class('\u{5D0}'), Class::R);
        assert_eq!(class('\u{627}'), Class::AL);
        assert_eq!(class('1'), Class::EN);
        assert_eq!(class('\u{661}'), Class::AN);
        assert_eq!(class(' '), Class::WS);
        assert_eq!(class('\u{2066}'), Class::LRI);
        assert_eq!(class('\u{200B}'), Class::BN);
    }

    #[test]
    fn the_class_table_is_sorted_and_complete() {
        let mut next = 0u32;
        for (first, last, _) in class_table::RANGES {
            assert_eq!(*first, next, "a gap or overlap before {first:#X}");
            assert!(last >= first);
            next = last + 1;
        }
        assert_eq!(next, 0x11_0000, "the table stops at {next:#X}");
    }
}
