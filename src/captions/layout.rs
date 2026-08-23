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
    let size = style.font_size;
    let metrics = font.metrics(size);
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
