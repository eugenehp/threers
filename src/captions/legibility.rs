//! Whether a [`CaptionStyle`] stays readable over whatever the video shows.
//!
//! Captions burned into video are text (WCAG 1.4.3 asks 4.5:1 of them, as
//! of any text), but what is behind them changes every frame. So the text
//! must contrast with something the style itself guarantees — the box
//! behind it, or an edge around each glyph (the outline, or a drop shadow)
//! — over the brightest frame and the darkest. That is what caption
//! guidance (the FCC's caption quality rules, the DCMP Captioning Key) asks
//! for in other words: a background, or edge attributes, that keep the text
//! distinct from the picture.

use super::CaptionStyle;
use crate::utils::contrast::contrast_ratio;

/// What makes a caption style hard to read; see [`CaptionStyle::legibility_issues`].
#[derive(Debug, Clone, PartialEq)]
pub enum LegibilityIssue {
    /// Nothing between the text and the picture: no box, no outline, no
    /// shadow. White text vanishes in snow; black, at night.
    NoBackgroundOrEdge,
    /// The box is there but the text does not stand out from it, over a
    /// white frame and a black one composited through the box: the worse
    /// of the two.
    LowContrastWithBox { ratio: f32 },
    /// The text relies on an edge (no box) that is too close to it in
    /// colour to separate it from the picture.
    LowContrastWithEdge { ratio: f32 },
}

fn rgb(c: [u8; 4]) -> [f32; 3] {
    [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0]
}

fn over(top: [u8; 4], under: [f32; 3]) -> [f32; 3] {
    let a = top[3] as f32 / 255.0;
    let t = rgb(top);
    [0, 1, 2].map(|k| t[k] * a + under[k] * (1.0 - a))
}

impl CaptionStyle {
    /// The least contrast the text has with what is directly behind it,
    /// over any frame: with the box over the frame when there is one, over
    /// a white frame and a black one, the worse of the two. 1:1 with no box.
    pub fn worst_box_contrast(&self) -> f32 {
        [[1.0f32; 3], [0.0; 3]]
            .into_iter()
            .map(|frame| {
                let behind = over(self.background, frame);
                contrast_ratio(over(self.color, behind), behind)
            })
            .fold(f32::MAX, f32::min)
    }

    /// Why this style may be hard to read over some frames; empty when it
    /// is readable over all of them.
    ///
    /// Readable means at least 4.5:1 (WCAG 1.4.3, AA) between the text and
    /// one thing the style guarantees: the box behind it, over the brightest
    /// and the darkest frame, or — with no box to speak of — an outline or
    /// drop shadow at least a pixel wide around every glyph. A box that
    /// already gives the contrast needs no edge; an edge that gives it needs
    /// no box.
    pub fn legibility_issues(&self) -> Vec<LegibilityIssue> {
        const MIN: f32 = 4.5;
        let boxed = self.background[3] > 0;
        let outline = self.outline_width >= 1.0 && self.outline_color[3] > 0;
        let shadow = self.shadow_color[3] > 0 && (self.shadow_offset[0].abs() + self.shadow_offset[1].abs()) >= 1.0;
        let box_ratio = self.worst_box_contrast();
        if boxed && box_ratio >= MIN {
            return Vec::new();
        }
        // The edge's contrast with the text, the edge as drawn over the box
        // (or the frame) at its own alpha — the worse over light and dark.
        let edge = |c: [u8; 4]| {
            [[1.0f32; 3], [0.0; 3]]
                .into_iter()
                .map(|frame| {
                    let behind = over(self.background, frame);
                    contrast_ratio(over(self.color, behind), over(c, behind))
                })
                .fold(f32::MAX, f32::min)
        };
        let edge_ratio = [outline.then(|| edge(self.outline_color)), shadow.then(|| edge(self.shadow_color))]
            .into_iter()
            .flatten()
            .fold(0.0f32, f32::max);
        if (outline || shadow) && edge_ratio >= MIN {
            return Vec::new();
        }
        let mut issues = Vec::new();
        match (boxed, outline || shadow) {
            (false, false) => issues.push(LegibilityIssue::NoBackgroundOrEdge),
            (true, false) => issues.push(LegibilityIssue::LowContrastWithBox { ratio: box_ratio }),
            (false, true) => issues.push(LegibilityIssue::LowContrastWithEdge { ratio: edge_ratio }),
            (true, true) => {
                issues.push(LegibilityIssue::LowContrastWithBox { ratio: box_ratio });
                issues.push(LegibilityIssue::LowContrastWithEdge { ratio: edge_ratio });
            }
        }
        issues
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The broadcast default — white, a black outline, a translucent box —
    /// reads over anything; so does white on a solid black box alone, and
    /// white with only a black outline.
    #[test]
    fn the_default_and_the_classics_are_legible() {
        assert_eq!(CaptionStyle::default().legibility_issues(), vec![]);
        let boxed = CaptionStyle { outline_width: 0.0, background: [0, 0, 0, 255], ..Default::default() };
        assert_eq!(boxed.legibility_issues(), vec![]);
        assert!((boxed.worst_box_contrast() - 21.0).abs() < 1e-3);
        let edged = CaptionStyle { background: [0; 4], ..Default::default() };
        assert_eq!(edged.legibility_issues(), vec![]);
    }

    /// Bare text has nothing to stand on; a faint box over a white frame
    /// leaves white text at 1.3:1; a grey outline does not separate white
    /// text from the picture.
    #[test]
    fn styles_that_vanish_on_some_frames_are_flagged() {
        let bare = CaptionStyle { outline_width: 0.0, background: [0; 4], ..Default::default() };
        assert_eq!(bare.legibility_issues(), vec![LegibilityIssue::NoBackgroundOrEdge]);
        let faint = CaptionStyle { outline_width: 0.0, background: [0, 0, 0, 40], ..Default::default() };
        match faint.legibility_issues().as_slice() {
            [LegibilityIssue::LowContrastWithBox { ratio }] => assert!(*ratio < 1.5, "{ratio}"),
            other => panic!("{other:?}"),
        }
        let grey = CaptionStyle { background: [0; 4], outline_color: [190, 190, 190, 255], ..Default::default() };
        assert!(matches!(grey.legibility_issues().as_slice(), [LegibilityIssue::LowContrastWithEdge { .. }]));
    }
}
