//! A system's legibility settings, followed: [`Ui::set_bold_text`] shows
//! every text node a step bolder (Bold Text on Apple's systems), and
//! [`Ui::set_increase_contrast`] makes the theme's quieter colours stronger
//! (Increase Contrast; `prefers-contrast: more` on the web): secondary text
//! nearer the text's colour, borders and separators darker in light and
//! lighter in dark. With the text size ([`Ui::set_text_scale`]) and less
//! motion ([`Ui::set_reduce_motion`]), what a platform's accessibility
//! settings ask of a UI.
//!
//! ```
//! use threers_ui::{style, FontWeight, Ui};
//! use threers_ui::environment::Token;
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let t = ui.add_text(root, "Hello");
//! ui.set_bold_text(true);
//! assert_eq!(ui.style(t).font_weight, FontWeight::Weight(600));
//! let quiet = ui.token(Token::SecondaryText);
//! ui.set_increase_contrast(true);
//! assert_ne!(ui.token(Token::SecondaryText), quiet);
//! ```

use crate::environment::Token;
use crate::tree::{NodeId, NodeKind, Ui};
use crate::{FontWeight, Rgba};

/// A weight a step bolder: regular to semibold, bold to extra bold.
fn bolder(w: FontWeight) -> FontWeight {
    let n = match w {
        FontWeight::Normal => 400,
        FontWeight::Bold => 700,
        FontWeight::Weight(n) => n,
    };
    FontWeight::Weight((n + 200).min(900))
}

impl Ui {
    /// Every text node a step bolder (or as it was), now and as text is
    /// added or restyled: a user's Bold Text.
    pub fn set_bold_text(&mut self, on: bool) {
        if on == self.env.bold_text {
            return;
        }
        self.env.bold_text = on;
        let texts: Vec<NodeId> = (0..self.nodes.len())
            .filter(|&i| !self.nodes[i].removed && matches!(self.nodes[i].kind, NodeKind::Text(_)))
            .map(|i| self.id_at(i))
            .collect();
        for id in texts {
            self.embolden_text(id);
        }
    }

    pub fn bold_text(&self) -> bool {
        self.env.bold_text
    }

    /// A text node's weight from its own and the setting.
    pub(crate) fn embolden_text(&mut self, id: NodeId) {
        let own = *self
            .env
            .unbolded
            .entry(id)
            .or_insert(self.nodes[id.0].style.font_weight);
        let want = if self.env.bold_text { bolder(own) } else { own };
        if self.nodes[id.0].style.font_weight != want {
            self.style_mut(id).font_weight = want;
        }
    }

    /// A text node was given `weight` of its own (a style set, a parent's
    /// font taken): that is its own weight now.
    pub(crate) fn reembolden_text(&mut self, id: NodeId, weight: FontWeight) {
        if self.env.bold_text {
            self.env.unbolded.insert(id, weight);
            self.embolden_text(id);
        } else {
            self.env.unbolded.remove(&id);
        }
    }

    /// The text node's weight before Bold Text made it bolder.
    pub(crate) fn own_weight(&self, id: NodeId) -> FontWeight {
        self.env
            .unbolded
            .get(&id)
            .copied()
            .filter(|_| self.env.bold_text)
            .unwrap_or(self.nodes[id.0].style.font_weight)
    }

    /// The theme's quieter colours stronger (or as they were): a user's
    /// Increase Contrast. Every node coloured by [`Token`] takes them.
    pub fn set_increase_contrast(&mut self, on: bool) {
        if on != self.env.increase_contrast {
            self.env.increase_contrast = on;
            self.restyle_all();
        }
    }

    pub fn increase_contrast(&self) -> bool {
        self.env.increase_contrast
    }
}

/// `token`'s colour `c`, stronger: secondary text half way to the text's
/// colour `text`, borders and separators a third of the way.
pub(crate) fn stronger(token: Token, c: Rgba, text: Rgba) -> Rgba {
    let toward = |k: f32| -> Rgba {
        [
            c[0] + (text[0] - c[0]) * k,
            c[1] + (text[1] - c[1]) * k,
            c[2] + (text[2] - c[2]) * k,
            c[3] + (1.0 - c[3]) * k,
        ]
    };
    match token {
        Token::SecondaryText => toward(0.5),
        Token::Border | Token::Separator => toward(0.35),
        _ => c,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{contrast_ratio, style, FixedWidthText};

    #[test]
    fn bold_text_holds_through_new_and_restyled_text() {
        let mut ui = Ui::new();
        let root = ui.root();
        let heading = ui.add_box(root, style! { font-weight: bold; });
        let h = ui.add_text(heading, "Title");
        let body = ui.add_text(root, "Body");
        ui.set_bold_text(true);
        assert_eq!(ui.style(h).font_weight, FontWeight::Weight(900));
        assert_eq!(ui.style(body).font_weight, FontWeight::Weight(600));
        // Added after, and given a style of its own after: still bolder.
        let later = ui.add_text(root, "Later");
        assert_eq!(ui.style(later).font_weight, FontWeight::Weight(600));
        let mut s = ui.style(later).clone();
        s.font_weight = FontWeight::Bold;
        ui.set_style(later, s);
        assert_eq!(ui.style(later).font_weight, FontWeight::Weight(900));
        // The parent's font changing reaches its text, made bolder.
        let mut ps = ui.style(heading).clone();
        ps.font_weight = FontWeight::Normal;
        ui.set_style(heading, ps);
        assert_eq!(ui.style(h).font_weight, FontWeight::Weight(600));
        // And laying out again does not keep restyling it.
        ui.layout(300.0, 200.0, &mut FixedWidthText(7.0));
        let passes = ui.layout_passes();
        let mut ps = ui.style(heading).clone();
        ps.font_weight = FontWeight::Normal;
        ui.set_style(heading, ps);
        ui.layout(300.0, 200.0, &mut FixedWidthText(7.0));
        assert_eq!(ui.layout_passes(), passes);
        // Off: back as each was.
        ui.set_bold_text(false);
        assert_eq!(ui.style(h).font_weight, FontWeight::Normal);
        assert_eq!(ui.style(later).font_weight, FontWeight::Bold);
        assert_eq!(ui.style(body).font_weight, FontWeight::Normal);
    }

    #[test]
    fn increased_contrast_strengthens_quiet_colours() {
        for dark in [false, true] {
            let mut ui = Ui::new();
            ui.set_theme(if dark {
                crate::states::Theme::Dark
            } else {
                crate::states::Theme::Light
            });
            let root = ui.root();
            let note = ui.add_text(root, "A note");
            ui.token_style(note, &[("color", Token::SecondaryText)]);
            let rgb = |c: Rgba| [c[0], c[1], c[2]];
            let bg = rgb(ui.token(Token::Background));
            let before = contrast_ratio(rgb(ui.token(Token::SecondaryText)), bg);
            ui.set_increase_contrast(true);
            let after = contrast_ratio(rgb(ui.token(Token::SecondaryText)), bg);
            assert!(after > before * 1.3, "dark {dark}: {before} to {after}");
            // (Through CSS: to a 255th.)
            let (got, want) = (ui.style(note).color, ui.token(Token::SecondaryText));
            assert!(
                (0..4).all(|i| (got[i] - want[i]).abs() < 1.0 / 255.0),
                "the node took it: {got:?} {want:?}"
            );
            assert_eq!(ui.token(Token::Accent), {
                let mut u = Ui::new();
                u.set_theme(if dark {
                    crate::states::Theme::Dark
                } else {
                    crate::states::Theme::Light
                });
                u.token(Token::Accent)
            });
        }
    }
}
