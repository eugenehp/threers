//! Arabic joining and OpenType `GSUB` substitution.
//!
//! Ported from `rswift`'s `swiftui-shape`, where it is measured against Core Text: 16 of
//! 16 corpus words and 1987 of 2000 fuzzed Arabic words exact, and 0 of 20,000 fuzzed
//! Latin strings differing in each of six fonts. The one coupling to that crate's font
//! type — a single `font.glyph(c)` — is a closure here, so this reads glyphs from
//! whatever the caller has.

/// A character's Arabic joining type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Joining {
    NonJoining,
    JoinCausing,
    DualJoining,
    LeftJoining,
    RightJoining,
    Transparent,
}

pub mod gsub;
pub mod joining;
pub mod read;
pub mod script;
mod joining_table;
mod script_table;

/// One shaped glyph and which character it came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shaped {
    /// The glyph to draw.
    pub glyph: u16,
    /// The index in the input `char` slice this glyph starts at.
    ///
    /// A ligature carries the index of its first component, and the characters it
    /// swallowed produce no glyph of their own — so this is the cluster, which is what a
    /// caller needs for hit testing and cursor movement.
    pub cluster: usize,
}

/// The features applied after the four joining forms, in order.
///
/// `rclt` is **required** contextual alternates and is where Arabic is actually shaped:
/// `SFArabic` registers eighteen lookups under it, sixteen of them chaining contextual,
/// and they rewrite a letter's base form according to its neighbours and decompose it
/// into a skeleton plus marks. The joining features alone pick the base form and nothing
/// else — with `rclt` missing, this shaper agreed with Core Text on 5 of 16 words.
///
/// `rlig` is required ligatures, where lam-alef lives, and it comes **before** `rclt`.
/// The order is not cosmetic: with `rclt` first, lam-alef in isolation ligates to the
/// wrong glyph — 1414 where Core Text says 965 — because `rclt` has already rewritten the
/// lam's form by the time the ligature lookup sees it.
pub const AFTER_FORMS: &[[u8; 4]] =
    &[*b"ccmp", *b"rlig", *b"rclt", *b"calt", *b"liga"];

/// Shape `text` with `font`: characters in, glyphs out.
///
/// Three stages, and the **order is the whole thing**:
///
/// 1. `cmap` maps each character to its nominal glyph.
/// 2. The **joining form** of each character selects a single substitution from `isol`,
///    `init`, `medi` or `fina`. The form belongs to the *character*, so it is looked up by
///    cluster.
/// 3. The features in [`AFTER_FORMS`] run over the whole buffer, in order. These are the
///    ones that can change the glyph count — a decomposition makes one glyph several, a
///    ligature makes several one — so from here on a glyph and a character are not in
///    correspondence and the cluster is what relates them.
///
/// `script` is the OpenType script tag — `arab` for Arabic — and it matters: a feature
/// tag means different lookups under different scripts. See [`gsub`].
///
/// A character with no glyph in the font contributes nothing, so a caller can spot it by
/// a missing cluster.
#[must_use]
pub fn shape(
    glyph_of: impl Fn(char) -> Option<u16>,
    gsub: Option<&gsub::Gsub<'_>>,
    script: &[u8; 4],
    text: &[char],
) -> Vec<Shaped> {
    let forms = joining::forms(text);
    let mut pairs: Vec<(u16, usize)> = Vec::with_capacity(text.len());
    for (i, c) in text.iter().enumerate() {
        if let Some(g) = glyph_of(*c) {
            pairs.push((g, i));
        }
    }
    let Some(gsub) = gsub else {
        return pairs.into_iter().map(|(glyph, cluster)| Shaped { glyph, cluster }).collect();
    };
    let mut buf = gsub::Buffer::new(&pairs);

    // The joining forms, by the cluster's character rather than by glyph position.
    for i in 0..buf.len() {
        let Some(form) = forms.get(buf.clusters[i]) else { continue };
        let mut one = gsub::Buffer::new(&[(buf.glyphs[i], buf.clusters[i])]);
        gsub.apply_feature(script, &form.tag(), &mut one);
        if one.len() == 1 {
            buf.glyphs[i] = one.glyphs[0];
        }
    }

    for tag in AFTER_FORMS {
        gsub.apply_feature(script, tag, &mut buf);
    }
    buf.glyphs
        .iter()
        .zip(&buf.clusters)
        .map(|(glyph, cluster)| Shaped { glyph: *glyph, cluster: *cluster })
        .collect()
}
