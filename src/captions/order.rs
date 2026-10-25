//! A caption's text as it is actually drawn: composed, reordered, mirrored.
//!
//! Between a string and a row of glyphs there are three transformations, and the caption
//! painter used to do none of them — it walked `text.chars()` and drew each one. Each
//! omission has a visible consequence:
//!
//! | missing | what the caption looked like |
//! |---|---|
//! | composition | `"Noe\u{0308}l"` drew a floating diaeresis after the `e` |
//! | reordering | Hebrew and Arabic read backwards |
//! | mirroring | a `(` opened the wrong way inside a right-to-left phrase |
//!
//! | shaping | Arabic letters stood apart instead of joining |
//!
//! The ordering had to be right before the shaping: a shaper's output laid out backwards
//! is no better than unshaped text laid out backwards.

use super::font::CaptionFont;
use crate::loaders::shape;

use super::{bidi, compose};

/// One glyph of a line, in the order it is drawn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placed {
    /// The glyph to draw, by the index [`CaptionFont::gid`] gives.
    pub gid: u32,
    /// Which character of the composed text it came from.
    ///
    /// Shaping breaks the one-to-one correspondence in both directions — a ligature makes
    /// several characters one glyph, a decomposition makes one character several — so this
    /// is what relates the two afterwards.
    pub cluster: usize,
    /// Where this glyph hangs off another, and by how much, in **pixels**.
    ///
    /// `Some((index into this line, dx, dy))` for a mark: the dots an Arabic letter is
    /// decomposed into are mark glyphs, so this is not only about combining characters.
    /// `dy` is above the baseline. Resolved against the glyph's *base*, which in a
    /// reversed run comes **after** it on screen — so the caller has to place every
    /// unattached glyph before it can place these.
    pub attach: Option<(usize, f32, f32)>,
    /// Whether it came from a right-to-left run.
    ///
    /// The kerning needs this. A kerning pair is keyed in **logical** order, so for two
    /// visually adjacent glyphs of a reversed run the pair to look up is the other way
    /// round. Asking for the visual pair finds a different pair or none, and every letter
    /// lands a fraction of a pixel out — the words read correctly and the spacing is
    /// subtly wrong, which is the hardest kind of error to notice. (Measured in `rswift`,
    /// where fixing exactly this took a right-to-left pixel sweep from 3604 differing
    /// cells to 149.)
    pub rtl: bool,
}

/// `text` as a sequence of glyphs in drawing order.
///
/// The order of the four stages is not interchangeable:
///
/// 1. **Compose**, because it changes *which characters there are* — and the bidi classes
///    are then resolved on the result, since a combining mark has a class of its own and
///    resolving first would count marks that are about to disappear.
/// 2. **Resolve levels**, which decides both the direction of each run and the next step.
/// 3. **Mirror** (**L4**), which is a *character* operation: a `(` inside a right-to-left
///    phrase becomes a `)` and then maps to that glyph. Mirroring after the `cmap` lookup
///    would have nothing to mirror.
/// 4. **Shape**, per run of one level and one script, in **logical** order — joining asks
///    what precedes and follows a letter — and the result reversed where the level is odd.
///
/// The runs themselves are then ordered by the same rule at run granularity.
pub fn visual(text: &str, font: &CaptionFont, size: f32) -> Vec<Placed> {
    let typed: Vec<char> = text.chars().collect();
    let mut chars = compose::composed(&typed);
    if chars.is_empty() {
        return Vec::new();
    }
    let classes: Vec<bidi::Class> = chars.iter().map(|c| bidi::class(*c)).collect();
    let levels = bidi::resolve_classes(&classes, bidi::Direction::Auto);
    for (i, c) in chars.iter_mut().enumerate() {
        *c = bidi::mirror::at_level(*c, levels.levels.get(i).copied().unwrap_or(0));
    }
    let scripts = shape::script::runs(&chars);

    // A run breaks where either the level or the script changes.
    let mut runs: Vec<(usize, usize, u8, [u8; 4])> = Vec::new();
    for i in 0..chars.len() {
        let level = levels.levels.get(i).copied().unwrap_or(0);
        let script = scripts
            .iter()
            .find(|(a, b, _)| i >= *a && i < *b)
            .map_or(shape::script::DEFAULT, |(_, _, t)| *t);
        match runs.last_mut() {
            Some((_, end, l, s)) if *l == level && *s == script && *end == i => *end = i + 1,
            _ => runs.push((i, i + 1, level, script)),
        }
    }

    let mut shaped: Vec<Vec<Placed>> = Vec::with_capacity(runs.len());
    for (a, b, level, script) in &runs {
        let rtl = level % 2 == 1;
        let ids = font.shape_run(&chars[*a..*b], script, size);
        // Mark attachment is resolved on the **logical** sequence, because that is how
        // `GPOS` defines it, and then turned round with the glyphs.
        let marks = font.attachments(&ids.iter().map(|(g, _)| *g).collect::<Vec<_>>(), size);
        let mut run: Vec<Placed> = ids
            .into_iter()
            .zip(marks)
            .map(|((gid, cluster), m)| Placed {
                gid,
                cluster: a + cluster,
                attach: m.to.map(|t| (usize::from(t), m.dx, m.dy)),
                rtl,
            })
            .collect();
        if rtl {
            let n = run.len();
            for p in &mut run {
                // An index into the reversed sequence.
                p.attach = p.attach.map(|(t, dx, dy)| (n - 1 - t, dx, dy));
            }
            run.reverse();
        }
        shaped.push(run);
    }
    let run_levels: Vec<u8> = runs.iter().map(|(_, _, l, _)| *l).collect();
    // The runs are concatenated, so an attachment index inside a run has to be shifted
    // by where that run lands in the line.
    let mut out: Vec<Placed> = Vec::new();
    for i in bidi::reorder(&run_levels, levels.paragraph) {
        let Some(run) = shaped.get(i) else { continue };
        let base = out.len();
        out.extend(run.iter().map(|p| Placed {
            attach: p.attach.map(|(t, dx, dy)| (base + t, dx, dy)),
            ..*p
        }));
    }
    out
}

/// The kerning pair between two drawn glyphs, in logical order.
///
/// Returns `(left, right)` as the font stores them, which for a reversed run is the
/// reverse of what is on the screen.
pub fn pair(previous: Placed, current: Placed) -> (u32, u32) {
    if previous.rtl && current.rtl {
        (current.gid, previous.gid)
    } else {
        (previous.gid, current.gid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The built-in bitmap face, whose glyph index **is** the codepoint — so a test can
    /// read the drawn text back as a string and still be exercising the glyph-id path.
    fn face() -> CaptionFont {
        CaptionFont::builtin()
    }

    fn text(s: &str) -> String {
        visual(s, &face(), 16.0)
            .into_iter()
            .filter_map(|p| char::from_u32(p.gid))
            .collect()
    }

    #[test]
    fn latin_is_left_alone() {
        assert_eq!(text("Hello, world"), "Hello, world");
        assert_eq!(text(""), "");
    }

    #[test]
    fn an_accent_composes() {
        assert_eq!(text("Noe\u{0308}l"), "Noël");
    }

    #[test]
    fn hebrew_is_reversed() {
        let typed: Vec<char> = "אבגד".chars().collect();
        let drawn: Vec<char> = text("אבגד").chars().collect();
        assert_eq!(drawn, typed.iter().rev().copied().collect::<Vec<_>>());
        assert!(visual("אבגד", &face(), 16.0).iter().all(|p| p.rtl));
    }

    #[test]
    fn a_number_after_hebrew_stays_in_order() {
        // The case a blanket reverse gets wrong: the digits are a run of their own and
        // read left to right inside the Hebrew.
        assert_eq!(text("א123ב"), "ב123א");
    }

    #[test]
    fn latin_inside_hebrew_keeps_its_direction() {
        assert_eq!(text("אabcב"), "בabcא");
    }

    #[test]
    fn a_bracket_mirrors_in_a_right_to_left_run() {
        // The point of **L4**, and it takes both steps to see it. Reversing alone would
        // put the typed `(` on the right still opening leftward — brackets around the
        // wrong side of the word. Reversed *and* mirrored, they still enclose `ב`.
        assert_eq!(text("א(ב)ג"), "ג(ב)א");
    }

    #[test]
    fn a_kerning_pair_is_logical() {
        let f = face();
        let run = visual("אב", &f, 16.0);
        assert_eq!(
            pair(run[0], run[1]),
            ('א' as u32, 'ב' as u32),
            "an RTL pair was taken visually"
        );
        let latin = visual("AV", &f, 16.0);
        assert_eq!(pair(latin[0], latin[1]), ('A' as u32, 'V' as u32));
    }

    #[test]
    fn a_cluster_points_back_at_the_composed_text() {
        // `Noël` is four characters after composing five, and the `l` is cluster 3 — not
        // 4, which is where it sat before the diaeresis was folded in.
        let run = visual("Noe\u{0308}l", &face(), 16.0);
        assert_eq!(run.len(), 4);
        assert_eq!(run[3].cluster, 3);
        assert_eq!(char::from_u32(run[3].gid), Some('l'));
    }

    #[test]
    fn a_right_to_left_run_keeps_its_clusters_with_its_glyphs() {
        // Reversing the glyphs must carry each one's cluster with it, or a caret or a
        // highlight lands on the wrong letter.
        let run = visual("אבג", &face(), 16.0);
        let clusters: Vec<usize> = run.iter().map(|p| p.cluster).collect();
        assert_eq!(clusters, vec![2, 1, 0]);
    }
}
