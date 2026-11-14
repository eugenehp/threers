//! Line breaking and measurement for caption text.
//!
//! Explicit `\n` in the cue text always breaks a line — subtitle authors use it
//! deliberately. Lines that still overflow the available width are wrapped
//! greedily at spaces, and a single word too long to fit is broken mid-word so
//! it can never run off the frame.

use super::{CaptionFont, CaptionStyle};

/// One laid-out line with its measured width.
#[derive(Clone, Debug, PartialEq)]
pub struct LineBox {
    pub text: String,
    /// Advance width in pixels, including letter spacing.
    pub width: f32,
}

/// A wrapped block of caption text and the metrics needed to place it.
#[derive(Clone, Debug, PartialEq)]
pub struct LaidOutText {
    pub lines: Vec<LineBox>,
    /// Width of the widest line, in pixels.
    pub width: f32,
    /// Total block height, in pixels.
    pub height: f32,
    /// Baseline-to-baseline distance, in pixels.
    pub line_advance: f32,
    /// Distance from the top of the block to the first baseline, in pixels.
    pub first_baseline: f32,
}

impl LaidOutText {
    /// Whether the block has no visible lines.
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }
}

/// Wrap and measure `text` for `style`, fitting lines into `max_width` pixels.
///
/// ```
/// use threers::captions::{layout, CaptionFont, CaptionStyle};
/// let mut font = CaptionFont::builtin();
/// let style = CaptionStyle::default().font_size(16.0);
/// // The built-in face is 16 px per glyph here, so 5 glyphs fit in 80 px.
/// let out = layout::layout(&mut font, "hello world", &style, 80.0);
/// assert_eq!(out.lines.len(), 2);
/// ```
pub fn layout(
    font: &mut CaptionFont,
    text: &str,
    style: &CaptionStyle,
    max_width: f32,
) -> LaidOutText {
    // Two generations of this many: when the newer fills, the older goes,
    // so what is still in use survives (a cache cleared whole at its limit
    // starts over just when it is busiest).
    const KEEP: usize = 1024;
    // Laid out at its natural width, a block fits any width at least as
    // wide unchanged (every line already fits): one entry serves them all.
    let natural = layout_key(text, style, None);
    if let Some(laid) = cached(font, natural)
        && laid.width <= max_width
    {
        return laid;
    }
    let bounded = layout_key(text, style, Some(max_width));
    if let Some(laid) = cached(font, bounded) {
        return laid;
    }
    let laid = layout_uncached(font, text, style, max_width);
    // Nothing wrapped: it is the natural layout.
    let unwrapped = laid.lines.len() == text.split('\n').count() || laid.lines.is_empty();
    let key = if unwrapped { natural } else { bounded };
    if font.layouts.len() >= KEEP {
        font.layouts_old = std::mem::take(&mut font.layouts);
    }
    font.layouts.insert(key, laid.clone());
    laid
}

/// A layout from the cache, the older generation's moved to the newer.
fn cached(font: &mut CaptionFont, key: u64) -> Option<LaidOutText> {
    if let Some(l) = font.layouts.get(&key) {
        return Some(l.clone());
    }
    let l = font.layouts_old.remove(&key)?;
    font.layouts.insert(key, l.clone());
    Some(l)
}

/// What a layout depends on: the text, the size, spacing and line height,
/// and the width (`None`: as wide as it is). FNV-1a, as this runs for every
/// string measured or drawn.
fn layout_key(text: &str, style: &CaptionStyle, width: Option<f32>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    };
    eat(text.as_bytes());
    eat(&style.font_size.to_bits().to_le_bytes());
    eat(&style.letter_spacing.to_bits().to_le_bytes());
    eat(&style.line_height.to_bits().to_le_bytes());
    match width {
        Some(w) => eat(&w.to_bits().to_le_bytes()),
        None => eat(b"natural"),
    }
    h
}

fn layout_uncached(
    font: &mut CaptionFont,
    text: &str,
    style: &CaptionStyle,
    max_width: f32,
) -> LaidOutText {
    let size = style.font_size;
    // The box of the faces the text is drawn in, which is the first face's alone unless it falls back.
    let metrics = font.metrics_of(text, size);
    let line_advance = size * style.line_height;
    let mut lines: Vec<LineBox> = Vec::new();

    for paragraph in text.split('\n') {
        let paragraph = paragraph.trim_end();
        if paragraph.is_empty() {
            lines.push(LineBox {
                text: String::new(),
                width: 0.0,
            });
            continue;
        }
        wrap_paragraph(font, paragraph, style, max_width, &mut lines);
    }

    // A block of only blank lines carries no information — drop it.
    if lines.iter().all(|l| l.text.is_empty()) {
        lines.clear();
    }

    let width = lines.iter().fold(0.0f32, |acc, l| acc.max(l.width));
    let height = if lines.is_empty() {
        0.0
    } else {
        // Ink height of the first line plus one advance for each line after it.
        metrics.ascent + metrics.descent + line_advance * (lines.len() - 1) as f32
    };

    LaidOutText {
        lines,
        width,
        height,
        line_advance,
        first_baseline: metrics.ascent,
    }
}

/// Greedy word wrap of one paragraph into `out`.
fn wrap_paragraph(
    font: &mut CaptionFont,
    paragraph: &str,
    style: &CaptionStyle,
    max_width: f32,
    out: &mut Vec<LineBox>,
) {
    let size = style.font_size;
    let spacing = style.letter_spacing;
    // Most lines fit whole: measured once, not word by word.
    let whole = font.measure(paragraph.trim_start(), size, spacing);
    if whole <= max_width && !paragraph.trim().is_empty() {
        let text = paragraph.split_whitespace().collect::<Vec<_>>().join(" ");
        let width = if text.len() == paragraph.trim_start().len() {
            whole
        } else {
            font.measure(&text, size, spacing)
        };
        out.push(LineBox { width, text });
        return;
    }
    let mut current = String::new();

    for word in paragraph.split_whitespace() {
        let candidate = if current.is_empty() {
            word.to_string()
        } else {
            format!("{current} {word}")
        };
        let candidate_width = font.measure(&candidate, size, spacing);
        if candidate_width <= max_width || current.is_empty() {
            // A first word wider than the line still has to go somewhere; it is
            // split below.
            current = candidate;
        } else {
            out.push(LineBox {
                width: font.measure(&current, size, spacing),
                text: std::mem::take(&mut current),
            });
            current = word.to_string();
        }

        // Break an over-long single word rather than letting it overflow.
        while font.measure(&current, size, spacing) > max_width && current.chars().count() > 1 {
            let split = break_point(font, &current, size, spacing, max_width);
            let head: String = current.chars().take(split).collect();
            let tail: String = current.chars().skip(split).collect();
            out.push(LineBox {
                width: font.measure(&head, size, spacing),
                text: head,
            });
            current = tail;
        }
    }

    if !current.is_empty() {
        out.push(LineBox {
            width: font.measure(&current, size, spacing),
            text: current,
        });
    }
}

/// The largest character count of `text` that still fits `max_width` (at least 1).
fn break_point(
    font: &mut CaptionFont,
    text: &str,
    size: f32,
    spacing: f32,
    max_width: f32,
) -> usize {
    let mut width = 0.0;
    let mut count = 0usize;
    for c in text.chars() {
        let advance = font.advance(c, size) + if count > 0 { spacing } else { 0.0 };
        if count > 0 && width + advance > max_width {
            break;
        }
        width += advance;
        count += 1;
    }
    count.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_layout_serves_every_width_it_fits_and_any_colour() {
        let mut font = CaptionFont::builtin();
        let s = style(16.0);
        let wide = layout(&mut font, "hello world", &s, 1000.0);
        assert_eq!(wide.lines.len(), 1);
        let red = CaptionStyle {
            color: [255, 0, 0, 255],
            ..s.clone()
        };
        let again = layout(&mut font, "hello world", &red, 400.0);
        assert_eq!((again.lines.len(), again.width), (1, wide.width));
        assert_eq!(
            font.layouts.len(),
            1,
            "measured once, for both widths and colours"
        );
        // Too narrow for it: wrapped, and kept apart.
        let narrow = layout(&mut font, "hello world", &s, 80.0);
        assert_eq!(narrow.lines.len(), 2);
        assert_eq!(font.layouts.len(), 2);
        assert_eq!(layout(&mut font, "hello world", &s, 80.0).lines.len(), 2);
    }

    fn style(size: f32) -> CaptionStyle {
        CaptionStyle::default().font_size(size).line_height(1.0)
    }

    #[test]
    fn explicit_newlines_always_break() {
        let mut font = CaptionFont::builtin();
        let out = layout(&mut font, "one\ntwo", &style(10.0), 10_000.0);
        assert_eq!(out.lines.len(), 2);
        assert_eq!(out.lines[0].text, "one");
        assert_eq!(out.lines[1].text, "two");
    }

    #[test]
    fn greedy_wrap_fits_words_into_the_width() {
        let mut font = CaptionFont::builtin();
        // Built-in glyphs are exactly `size` wide, so 10 px × 11 chars = 110 px.
        let out = layout(&mut font, "aaa bbb ccc", &style(10.0), 80.0);
        assert_eq!(out.lines.len(), 2);
        assert_eq!(out.lines[0].text, "aaa bbb");
        assert_eq!(out.lines[1].text, "ccc");
        assert_eq!(out.width, 70.0);
    }

    #[test]
    fn over_long_words_are_broken_instead_of_overflowing() {
        let mut font = CaptionFont::builtin();
        let out = layout(&mut font, "aaaaaaaaaa", &style(10.0), 30.0);
        assert!(out.lines.len() >= 4, "{:?}", out.lines);
        assert!(out.width <= 30.0, "no line may exceed the wrap width");
        let joined: String = out.lines.iter().map(|l| l.text.as_str()).collect();
        assert_eq!(joined, "aaaaaaaaaa", "no characters lost");
    }

    #[test]
    fn block_height_counts_ink_plus_one_advance_per_extra_line() {
        let mut font = CaptionFont::builtin();
        let s = style(16.0); // line_height 1.0 → advance 16
        let one = layout(&mut font, "a", &s, 1000.0);
        assert_eq!(one.height, 16.0); // ascent 14 + descent 2
        let two = layout(&mut font, "a\nb", &s, 1000.0);
        assert_eq!(two.height, 32.0);
        assert_eq!(two.line_advance, 16.0);
        assert_eq!(two.first_baseline, 14.0);
    }

    #[test]
    fn whitespace_only_text_lays_out_empty() {
        let mut font = CaptionFont::builtin();
        let out = layout(&mut font, "   \n  ", &style(16.0), 1000.0);
        assert!(out.is_empty());
        assert_eq!(out.height, 0.0);
    }
}
